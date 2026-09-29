//! Online collapsed variational SGPR.

use std::collections::HashMap;
use std::marker::PhantomData;

use faer::Mat;

use crate::error::GprError;
use crate::kernel::KernelScalar;
use crate::kernel::ScalarOps;
use crate::linalg::{chol_rank1_downdate, chol_rank1_update, frobenius2};
use crate::optimizer::{Lbfgs, Optimizer};
use crate::points::PointId;
use crate::points::PointRegistry;
use crate::policy::with_kernel_exp;
use crate::precision::{DoublePrecision, ModelPrecision};
use crate::sgpr::SgprObjective;
use crate::sparse::{
    KernelScratch, PredictScratch, SparseCore, SparseScratch, sparse_core_accessors,
};
use crate::{PredictOptions, Prediction, PredictiveCovariance};

use super::FixedInducing;
use super::factor::{
    VfeState, VfeSystem, append_column, append_point, assemble_vfe, inducing_delete,
    inducing_insert, kernel_column, kernel_diag_at, point_at, predict_vfe_covariance,
    predict_vfe_into, publish_sgpr_weights, refresh_w, remove_column, remove_point, solve_lmm,
    vfe_neg_log_marginal_likelihood,
};
use super::fitted::FittedSgpr;

/// Stable identity of one inducing point on [`OnlineSgpr`].
///
/// [`crate::FittedSgpr::into_online`] assigns identifiers `0 .. m-1` in
/// buffer order. Later [`OnlineSgpr::insert_inducing`] values increase
/// monotonically and are never reused after
/// [`OnlineSgpr::delete_inducing`]. There is no public constructor.
///
/// # Examples
///
/// ```rust
/// use gprx::kernel::{KernelSpec, RbfKernel};
/// use gprx::{Fixed, GaussianLikelihood, Sgpr};
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let fitted = Sgpr::new(
///     KernelSpec::from(RbfKernel::new(1.0)?),
///     GaussianLikelihood::new(0.1)?,
/// )
/// .with_optimizer(Fixed)
/// .factor(
///     &[0.0, 1.0, 2.0, 3.0],
///     4,
///     1,
///     &[0.0, 1.0, 0.5, 0.25],
///     &[0.5, 2.5],
///     2,
/// )
/// .map_err(|(_, e)| e)?;
/// let mut online = fitted.into_online();
/// let id = online.insert_inducing(&[1.5])?;
/// assert_eq!(online.inducing_ids().last().copied(), Some(id));
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct InducingId(u64);

impl InducingId {
    pub(crate) fn from_raw(raw: u64) -> Self {
        Self(raw)
    }
}

#[derive(Clone, Debug)]
struct InducingRegistry {
    id_to_index: HashMap<InducingId, usize>,
    index_to_id: Vec<InducingId>,
    next_id: u64,
}

impl InducingRegistry {
    fn from_count(m: usize) -> Self {
        let index_to_id: Vec<InducingId> = (0..m as u64).map(InducingId::from_raw).collect();
        let id_to_index = index_to_id
            .iter()
            .copied()
            .enumerate()
            .map(|(index, id)| (id, index))
            .collect();
        Self {
            id_to_index,
            index_to_id,
            next_id: m as u64,
        }
    }

    fn ids(&self) -> &[InducingId] {
        &self.index_to_id
    }

    fn index_of(&self, id: InducingId) -> Result<usize, GprError> {
        self.id_to_index
            .get(&id)
            .copied()
            .ok_or(GprError::InvalidInducingId)
    }

    fn insert(&mut self) -> InducingId {
        let id = InducingId::from_raw(self.next_id);
        let index = self.index_to_id.len();
        self.next_id = self.next_id.saturating_add(1);
        self.index_to_id.push(id);
        self.id_to_index.insert(id, index);
        id
    }

    fn remove_at(&mut self, index: usize) {
        let id = self.index_to_id.remove(index);
        self.id_to_index.remove(&id);
        for (shifted, remaining) in self.index_to_id.iter().enumerate().skip(index) {
            self.id_to_index.insert(*remaining, shifted);
        }
    }
}

/// Online collapsed variational SGPR after [`FittedSgpr::into_online`].
///
/// [`Self::insert`] appends one training point and returns a [`PointId`].
/// [`Self::delete`] removes one point by that identifier. VFE factors for
/// `X` update with a rank-1 cholupdate of `B` while `K_mm` stays put.
/// [`Self::insert_inducing`] appends one inducing point and returns an
/// [`InducingId`]. [`Self::delete_inducing`] removes one inducing point
/// by that identifier. Those updates follow ADR 0005.
/// Parameters are kernel `θ` then likelihood `θ`. [`Self::set_params`] and
/// [`Self::refit`] rebuild the VFE system.
///
/// # Examples
///
/// ```rust
/// use gprx::kernel::{KernelSpec, RbfKernel};
/// use gprx::{Fixed, GaussianLikelihood, Sgpr};
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let fitted = Sgpr::new(
///     KernelSpec::from(RbfKernel::new(1.0)?),
///     GaussianLikelihood::new(0.1)?,
/// )
/// .with_optimizer(Fixed)
/// .factor(
///     &[0.0, 1.0, 2.0, 3.0],
///     4,
///     1,
///     &[0.0, 1.0, 0.5, 0.25],
///     &[0.5, 2.5],
///     2,
/// )
/// .map_err(|(_, e)| e)?;
/// let mut online = fitted.into_online();
/// online.insert(&[4.0], 0.1)?;
/// let pred = online.predict(&[0.5], 1, 1)?;
/// assert_eq!(pred.mean.len(), 1);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct OnlineSgpr<O = Lbfgs, P: ModelPrecision = DoublePrecision> {
    pub(super) core: SparseCore,
    /// Kernel scratch kept between `&mut self` calls.
    pub(super) scratch: SparseScratch<P::Storage>,
    optimizer: O,
    k_mm_l: Mat<P::Storage>,
    a: Mat<P::Storage>,
    b_l: Mat<P::Storage>,
    w: Vec<P::Storage>,
    predict_w: Vec<P::Refine>,
    k_diag_sum: P::Storage,
    a_frobenius2: P::Storage,
    registry: PointRegistry,
    inducing: InducingRegistry,
}

impl<O, P> OnlineSgpr<O, P>
where
    P: crate::precision::GpScalar,
{
    pub(crate) fn from_fitted<I>(fitted: FittedSgpr<O, I, P>) -> Self {
        let registry = PointRegistry::from_count(fitted.core.n);
        let inducing = InducingRegistry::from_count(fitted.core.m);
        Self {
            core: fitted.core,
            scratch: fitted.scratch,
            optimizer: fitted.optimizer,
            k_mm_l: fitted.k_mm_l,
            a: fitted.a,
            b_l: fitted.b_l,
            w: fitted.w,
            predict_w: fitted.predict_w,
            k_diag_sum: fitted.k_diag_sum,
            a_frobenius2: fitted.a_frobenius2,
            registry,
            inducing,
        }
    }

    fn snapshot_fitted(&mut self) -> FittedSgpr<O, FixedInducing, P>
    where
        O: Clone,
    {
        FittedSgpr {
            core: self.core.clone(),
            // Lent for the call; `adopt_fitted` takes it back.
            scratch: std::mem::take(&mut self.scratch),
            optimizer: self.optimizer.clone(),
            inducing: PhantomData,
            k_mm_l: self.k_mm_l.clone(),
            a: self.a.clone(),
            b_l: self.b_l.clone(),
            w: self.w.clone(),
            predict_w: self.predict_w.clone(),
            k_diag_sum: self.k_diag_sum,
            a_frobenius2: self.a_frobenius2,
        }
    }

    fn adopt_fitted(&mut self, fitted: FittedSgpr<O, FixedInducing, P>) {
        self.core = fitted.core;
        self.scratch = fitted.scratch;
        self.optimizer = fitted.optimizer;
        self.k_mm_l = fitted.k_mm_l;
        self.a = fitted.a;
        self.b_l = fitted.b_l;
        self.w = fitted.w;
        self.predict_w = fitted.predict_w;
        self.k_diag_sum = fitted.k_diag_sum;
        self.a_frobenius2 = fitted.a_frobenius2;
    }

    fn vfe_state(&self) -> VfeState<P::Storage> {
        VfeState::<P::Storage> {
            k_mm_l: self.k_mm_l.clone(),
            a: self.a.clone(),
            b_l: self.b_l.clone(),
            w: self.w.clone(),
            k_diag_sum: self.k_diag_sum,
            a_frobenius2: self.a_frobenius2,
        }
    }

    fn apply_vfe(&mut self, state: VfeState<P::Storage>) -> Result<(), GprError> {
        self.k_mm_l = state.k_mm_l;
        self.a = state.a;
        self.b_l = state.b_l;
        self.w = state.w;
        self.k_diag_sum = state.k_diag_sum;
        self.a_frobenius2 = state.a_frobenius2;
        self.refresh_predict_w()
    }

    fn refresh_predict_w(&mut self) -> Result<(), GprError> {
        if P::REFINES_IN_F64 {
            self.predict_w = with_kernel_exp!(self.core.math, M => assemble_vfe::<M, f64>(
                &self.core.kernel,
                self.core.jitter,
                self.core.likelihood,
                &self.core.x_train,
                self.core.n,
                self.core.d,
                &self.core.y_train,
                &self.core.z_train,
                self.core.m,
                &mut self.scratch.f64,
                &mut KernelScratch::new(),
            ))?
            .w
            .into_iter()
            .map(P::Refine::from_f64)
            .collect();
            return Ok(());
        }
        self.predict_w = with_kernel_exp!(self.core.math, M => publish_sgpr_weights::<M, P>(
            &self.core.kernel,
            self.core.jitter,
            self.a.as_ref(),
            self.b_l.as_ref(),
            &self.w,
            &self.core.x_train,
            &self.core.y_train,
            &self.core.z_train,
            self.core.likelihood.noise_variance(),
            self.core.n,
            self.core.m,
            self.core.d,
        ))?;
        Ok(())
    }

    fn recompute_w(&mut self) -> Result<(), GprError> {
        let w = {
            let mut y_cast = P::Storage::empty_rows();
            let y_s = P::Storage::storage_rows(&self.core.y_train, &mut y_cast);
            refresh_w(self.a.as_ref(), self.b_l.as_ref(), y_s)
        };
        self.w = w;
        self.refresh_predict_w()
    }

    sparse_core_accessors!();

    /// Returns training-point identifiers in buffer order.
    pub fn point_ids(&self) -> &[PointId] {
        self.registry.ids()
    }

    /// Returns inducing-point identifiers in buffer order.
    pub fn inducing_ids(&self) -> &[InducingId] {
        self.inducing.ids()
    }

    /// Returns the concatenated kernel and likelihood parameter count.
    ///
    /// Inducing coordinates are not counted.
    pub fn num_params(&self) -> usize {
        self.core.theta_len()
    }

    /// Writes kernel `θ` then likelihood `θ` into `out`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::LengthMismatch`] if `out` is the wrong length
    /// or a custom leaf rejects the write.
    pub fn get_params(&self, out: &mut [f64]) -> Result<(), GprError> {
        self.core.read_theta(out)
    }

    /// Sets kernel then likelihood `θ` and rebuilds the VFE factors.
    ///
    /// `params` matches [`Self::get_params`]. Training `X` / `y` and
    /// inducing `Z` are not changed. Values are committed together only
    /// after the VFE system factors.
    ///
    /// # Errors
    ///
    /// Same as [`FittedSgpr::set_params`].
    pub fn set_params(&mut self, params: &[f64]) -> Result<(), GprError>
    where
        O: Clone,
    {
        let mut fitted = self.snapshot_fitted();
        fitted.set_params(params)?;
        self.adopt_fitted(fitted);
        Ok(())
    }

    /// Sets parameters, rebuilds the VFE system, and writes `∂L/∂θ` of the
    /// negative ELBO.
    ///
    /// `params` and `out` match [`Self::get_params`].
    ///
    /// # Errors
    ///
    /// Same as [`FittedSgpr::value_and_gradient_into`].
    pub fn value_and_gradient_into(
        &mut self,
        params: &[f64],
        out: &mut [f64],
    ) -> Result<f64, GprError>
    where
        O: Clone,
    {
        let mut fitted = self.snapshot_fitted();
        let value = fitted.value_and_gradient_into(params, out)?;
        self.adopt_fitted(fitted);
        Ok(value)
    }

    /// Writes the Hessian of the negative ELBO (row-major `p×p`) into `out`.
    ///
    /// # Errors
    ///
    /// Same as [`FittedSgpr::hessian_into`].
    pub fn hessian_into(&mut self, params: &[f64], out: &mut [f64]) -> Result<(), GprError>
    where
        O: Clone,
    {
        let mut fitted = self.snapshot_fitted();
        fitted.hessian_into(params, out)?;
        self.adopt_fitted(fitted);
        Ok(())
    }

    /// Returns the negative VFE evidence lower bound (the sparse NLML).
    ///
    /// # Errors
    ///
    /// The stored factors are already valid, so this returns `Ok`.
    pub fn neg_log_marginal_likelihood(&self) -> Result<f64, GprError> {
        vfe_neg_log_marginal_likelihood(
            self.a.as_ref(),
            self.b_l.as_ref(),
            &self.w,
            &self.core.y_train,
            self.k_diag_sum,
            self.a_frobenius2,
            self.core.likelihood.noise_variance(),
            self.core.n,
            self.core.m,
        )
    }

    /// Predicts at `xs` with [`PredictOptions::default`] (observation variance).
    ///
    /// # Errors
    ///
    /// Same as [`FittedSgpr::predict`].
    pub fn predict(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
    ) -> Result<Prediction<P::Refine>, GprError> {
        self.predict_with(xs, n_rows, n_cols, PredictOptions::default())
    }

    /// Predicts at `xs` with an explicit variance kind.
    ///
    /// # Errors
    ///
    /// Same as [`FittedSgpr::predict_with`].
    pub fn predict_with(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        options: PredictOptions,
    ) -> Result<Prediction<P::Refine>, GprError> {
        let mut out = Prediction::default();
        predict_vfe_into::<P>(
            &self.core,
            &VfeSystem::new(
                &self.core,
                self.k_mm_l.as_ref(),
                self.b_l.as_ref(),
                &self.predict_w,
            ),
            xs,
            n_rows,
            n_cols,
            options,
            &mut PredictScratch::default(),
            &mut out,
        )?;
        Ok(out)
    }

    /// Returns the predictive mean and query–query covariance at `xs`.
    ///
    /// Default [`PredictOptions`] uses [`crate::VarianceKind::Observation`]:
    /// `σn²` is added on the diagonal in the transformed space. The diagonal
    /// is exactly the variance of [`Self::predict`] for the same query. This
    /// path allocates an `m × m` matrix; [`Self::predict`] stays diagonal.
    ///
    /// # Errors
    ///
    /// Same as [`FittedSgpr::predict`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Fixed, GaussianLikelihood, Sgpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let mut fitted = Sgpr::new(
    ///     KernelSpec::from(RbfKernel::new(1.0)?),
    ///     GaussianLikelihood::new(0.1)?,
    /// )
    /// .with_optimizer(Fixed)
    /// .factor(&[0.0, 1.0, 2.0], 3, 1, &[0.0, 1.0, 0.5], &[0.5, 1.5], 2)
    /// .map_err(|(_, e)| e)?
    /// .into_online();
    /// fitted.insert(&[3.0], 0.2)?;
    /// let cov = fitted.predict_covariance(&[0.25, 0.75], 2, 1)?;
    /// assert_eq!(cov.covariance.len(), 4);
    /// assert_eq!(cov.covariance[0], fitted.predict(&[0.25, 0.75], 2, 1)?.variance[0]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn predict_covariance(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
    ) -> Result<PredictiveCovariance<P::Refine>, GprError> {
        self.predict_covariance_with(xs, n_rows, n_cols, PredictOptions::default())
    }

    /// Returns query–query covariance with an explicit variance kind.
    ///
    /// The VFE posterior covariance is `K** − Q** + K*m Σ Km*` with
    /// `Q** = K*m K_mm⁻¹ Km*` and `Σ = (K_mm + σn⁻² K_mn K_nm)⁻¹`.
    /// Latent diagonals are clipped at 0. Observation adds `σn²` on the
    /// diagonal in the transformed space, then the target transform scales
    /// the whole matrix.
    ///
    /// # Errors
    ///
    /// Same as [`FittedSgpr::predict`].
    pub fn predict_covariance_with(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        options: PredictOptions,
    ) -> Result<PredictiveCovariance<P::Refine>, GprError> {
        predict_vfe_covariance::<P>(
            &self.core,
            &VfeSystem::new(
                &self.core,
                self.k_mm_l.as_ref(),
                self.b_l.as_ref(),
                &self.predict_w,
            ),
            xs,
            n_rows,
            n_cols,
            options,
        )
    }

    /// Draws posterior samples at `xs` from [`Self::predict_covariance`].
    ///
    /// Each column of the returned column-major `m × n_draws` matrix is
    /// `μ + L z` with `z ∼ N(0, I)` and `L` the Cholesky factor of the
    /// posterior covariance, the same draw as [`crate::FittedGpr::sample`].
    /// `seed` is the crate [`rand::rngs::SmallRng`] start state. Zero draws
    /// returns an empty vector after the covariance is formed.
    ///
    /// # Errors
    ///
    /// Same as [`FittedSgpr::predict`], plus [`GprError::CholeskyFailed`] with
    /// [`CholeskyStage::Predict`](crate::CholeskyStage::Predict) if the
    /// posterior covariance cannot be factored after the retries of
    /// [`Self::jitter_policy`].
    pub fn sample(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        n_draws: usize,
        seed: u64,
    ) -> Result<Vec<P::Refine>, GprError> {
        self.sample_with(xs, n_rows, n_cols, PredictOptions::default(), n_draws, seed)
    }

    /// Draws posterior samples with an explicit variance kind.
    ///
    /// # Errors
    ///
    /// Same as [`Self::sample`].
    pub fn sample_with(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        options: PredictOptions,
        n_draws: usize,
        seed: u64,
    ) -> Result<Vec<P::Refine>, GprError> {
        self.predict_covariance_with(xs, n_rows, n_cols, options)?
            .draw(n_draws, seed, self.core.jitter)
    }

    /// Predicts at `xs` with [`PredictOptions::default`] into `out`.
    ///
    /// Same values as [`Self::predict`]. The buffers are kept on the model
    /// and `out` keeps its capacity, so a call after one with the same
    /// shapes allocates nothing.
    ///
    /// # Errors
    ///
    /// Same as [`FittedSgpr::predict`].
    pub fn predict_into(
        &mut self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        out: &mut Prediction<P::Refine>,
    ) -> Result<(), GprError> {
        self.predict_with_into(xs, n_rows, n_cols, PredictOptions::default(), out)
    }

    /// Predicts at `xs` with an explicit variance kind into `out`.
    ///
    /// Same values as [`Self::predict_with`], with the buffers of
    /// [`Self::predict_into`].
    ///
    /// # Errors
    ///
    /// Same as [`FittedSgpr::predict`].
    pub fn predict_with_into(
        &mut self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        options: PredictOptions,
        out: &mut Prediction<P::Refine>,
    ) -> Result<(), GprError> {
        predict_vfe_into::<P>(
            &self.core,
            &VfeSystem::new(
                &self.core,
                self.k_mm_l.as_ref(),
                self.b_l.as_ref(),
                &self.predict_w,
            ),
            xs,
            n_rows,
            n_cols,
            options,
            &mut self.scratch.predict,
            out,
        )
    }

    /// Appends one training point at the current `θ` with a rank-1 VFE update.
    ///
    /// `x_new` has length [`Self::d`]. `x_new` and `y_new` are in the
    /// original units and go through the transforms fitted at training.
    /// Inducing coordinates are not moved.
    /// The returned [`PointId`] is new and is never reused after a later
    /// [`Self::delete`].
    ///
    /// # Errors
    ///
    /// Returns [`GprError::DimensionMismatch`] if `x_new` is the wrong length,
    /// [`GprError::NonFiniteInput`] if a value is `NaN` or `Inf`, or
    /// [`GprError::EmptyInput`] if `d` is zero.
    pub fn insert(&mut self, x_new: &[f64], y_new: f64) -> Result<PointId, GprError> {
        if x_new.len() != self.core.d {
            return Err(GprError::DimensionMismatch {
                x_dim: x_new.len(),
                expected_dim: self.core.d,
            });
        }
        if self.core.d == 0 {
            return Err(GprError::EmptyInput);
        }
        if x_new.iter().any(|v| !v.is_finite()) || !y_new.is_finite() {
            return Err(GprError::NonFiniteInput);
        }
        let mut mapped = std::mem::take(&mut self.scratch.point);
        let result = self.insert_mapped(x_new, y_new, &mut mapped);
        self.scratch.point = mapped;
        result
    }

    /// [`Self::insert`] after the checks. `mapped` receives `x_obs` through
    /// the input transform.
    fn insert_mapped(
        &mut self,
        x_obs: &[f64],
        y_obs: f64,
        mapped: &mut Vec<f64>,
    ) -> Result<PointId, GprError> {
        self.core.map_point(x_obs, mapped)?;
        let x_new = mapped.as_slice();
        let y_new = self.core.map_target(y_obs)?;
        let mut a_col = with_kernel_exp!(self.core.math, M => kernel_column::<M, P::Storage>(
            &self.core.kernel,
            &self.core.z_train,
            self.core.m,
            x_new,
            self.core.d,
            &mut self.scratch.storage,
        ))?;
        solve_lmm(self.k_mm_l.as_ref(), a_col.as_mut());
        let mut v = vec![P::Storage::from_f64(0.0); self.core.m];
        for (i, slot) in v.iter_mut().enumerate() {
            *slot = a_col[(i, 0)];
        }
        self.a_frobenius2 += frobenius2(a_col.as_ref());
        self.k_diag_sum += kernel_diag_at::<P::Storage>(&self.core.kernel, x_new, self.core.d)?;
        self.a = append_column(&self.a, a_col.as_ref());
        chol_rank1_update(&mut self.b_l, &mut v);
        append_point(&mut self.core.x_train, self.core.n, self.core.d, x_new);
        append_point(&mut self.core.x_obs, self.core.n, self.core.d, x_obs);
        self.core.y_train.push(y_new);
        self.core.y_obs.push(y_obs);
        self.core.n += 1;
        self.recompute_w()?;
        Ok(self.registry.insert())
    }

    /// Removes the training point identified by `id` and packs every buffer.
    ///
    /// Updates `B` with a rank-1 downdate. If that loses positive
    /// definiteness, the remaining points are factored again. The last
    /// remaining point cannot be deleted.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::EmptyInput`] when `n == 1`, or
    /// [`GprError::InvalidPointId`] when `id` is unknown or already deleted.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Fixed, GaussianLikelihood, Sgpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let fitted = Sgpr::new(
    ///     KernelSpec::from(RbfKernel::new(1.0)?),
    ///     GaussianLikelihood::new(0.1)?,
    /// )
    /// .with_optimizer(Fixed)
    /// .factor(
    ///     &[0.0, 1.0, 2.0, 3.0],
    ///     4,
    ///     1,
    ///     &[0.0, 1.0, 0.5, 0.25],
    ///     &[0.5, 2.5],
    ///     2,
    /// )
    /// .map_err(|(_, e)| e)?;
    /// let mut online = fitted.into_online();
    /// let id = online.point_ids()[1];
    /// online.delete(id)?;
    /// assert_eq!(online.n(), 3);
    /// # Ok(())
    /// # }
    /// ```
    pub fn delete(&mut self, id: PointId) -> Result<(), GprError> {
        if self.core.n <= 1 {
            return Err(GprError::EmptyInput);
        }
        let idx = self.registry.index_of(id)?;
        let mut v = vec![P::Storage::from_f64(0.0); self.core.m];
        for (i, slot) in v.iter_mut().enumerate() {
            *slot = self.a[(i, idx)];
        }
        let x_pt = point_at(&self.core.x_train, self.core.n, self.core.d, idx);
        let diag = kernel_diag_at::<P::Storage>(&self.core.kernel, &x_pt, self.core.d)?;
        let mut col_norm = P::Storage::from_f64(0.0);
        for value in &v {
            col_norm += *value * *value;
        }
        let x_next = remove_point(&self.core.x_train, self.core.n, self.core.d, idx);
        let mut y_next = self.core.y_train.clone();
        y_next.remove(idx);
        let x_obs_next = remove_point(&self.core.x_obs, self.core.n, self.core.d, idx);
        let mut y_obs_next = self.core.y_obs.clone();
        y_obs_next.remove(idx);
        let mut b_trial = self.b_l.clone();
        let mut v_trial = v;
        if chol_rank1_downdate(&mut b_trial, &mut v_trial) {
            self.k_diag_sum -= diag;
            self.a_frobenius2 -= col_norm;
            self.a = remove_column(&self.a, idx);
            self.b_l = b_trial;
            self.core.x_train = x_next;
            self.core.y_train = y_next;
            self.core.x_obs = x_obs_next;
            self.core.y_obs = y_obs_next;
            self.core.n -= 1;
            self.recompute_w()?;
        } else {
            let state = with_kernel_exp!(self.core.math, M => assemble_vfe::<M, P::Storage>(
                &self.core.kernel,
                self.core.jitter,
                self.core.likelihood,
                &x_next,
                self.core.n - 1,
                self.core.d,
                &y_next,
                &self.core.z_train,
                self.core.m,
                &mut self.scratch.storage,
                &mut self.scratch.f64,
            ))?;
            self.core.x_train = x_next;
            self.core.y_train = y_next;
            self.core.x_obs = x_obs_next;
            self.core.y_obs = y_obs_next;
            self.core.n -= 1;
            self.k_mm_l = state.k_mm_l;
            self.a = state.a;
            self.b_l = state.b_l;
            self.w = state.w;
            self.k_diag_sum = state.k_diag_sum;
            self.a_frobenius2 = state.a_frobenius2;
        }
        self.registry.remove_at(idx);
        Ok(())
    }

    /// Appends one inducing point at the current `θ` with a bordered VFE update.
    ///
    /// `z_new` has length [`Self::d`], in the original coordinates of `X`;
    /// it goes through the input transform fitted at training. Training
    /// `X` / `y` are not moved.
    /// The returned [`InducingId`] is new and is never reused after a later
    /// [`Self::delete_inducing`]. If the bordered Schur complement is
    /// non-positive, the enlarged inducing set is assembled again.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::DimensionMismatch`] if `z_new` is the wrong length,
    /// [`GprError::NonFiniteInput`] if a value is `NaN` or `Inf`,
    /// [`GprError::EmptyInput`] if `d` is zero, or
    /// [`GprError::CholeskyFailed`] if the full reassemble of the enlarged
    /// inducing set fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Fixed, GaussianLikelihood, Sgpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let fitted = Sgpr::new(
    ///     KernelSpec::from(RbfKernel::new(1.0)?),
    ///     GaussianLikelihood::new(0.1)?,
    /// )
    /// .with_optimizer(Fixed)
    /// .factor(
    ///     &[0.0, 1.0, 2.0, 3.0],
    ///     4,
    ///     1,
    ///     &[0.0, 1.0, 0.5, 0.25],
    ///     &[0.5, 2.5],
    ///     2,
    /// )
    /// .map_err(|(_, e)| e)?;
    /// let mut online = fitted.into_online();
    /// online.insert_inducing(&[1.5])?;
    /// assert_eq!(online.m(), 3);
    /// # Ok(())
    /// # }
    /// ```
    pub fn insert_inducing(&mut self, z_new: &[f64]) -> Result<InducingId, GprError> {
        if z_new.len() != self.core.d {
            return Err(GprError::DimensionMismatch {
                x_dim: z_new.len(),
                expected_dim: self.core.d,
            });
        }
        if self.core.d == 0 {
            return Err(GprError::EmptyInput);
        }
        if z_new.iter().any(|v| !v.is_finite()) {
            return Err(GprError::NonFiniteInput);
        }
        let z_obs = z_new;
        let mut z_new = Vec::with_capacity(z_obs.len());
        self.core.map_point(z_obs, &mut z_new)?;
        let z_new = z_new.as_slice();
        let mut state = self.vfe_state();
        match with_kernel_exp!(self.core.math, M => inducing_insert::<M, _>(
            &mut state,
            &self.core.kernel,
            self.core.likelihood.noise_variance(),
            &self.core.x_train,
            self.core.n,
            self.core.d,
            &self.core.y_train,
            &self.core.z_train,
            self.core.m,
            z_new,
            &mut self.scratch.storage,
        )) {
            Ok(()) | Err(GprError::CholeskyFailed { .. }) => {}
            Err(err) => return Err(err),
        }
        let (m, d) = (self.core.m, self.core.d);
        let mut z_train = self.core.z_train.clone();
        append_point(&mut z_train, m, d, z_new);
        let mut z_obs_next = self.core.z_obs.clone();
        append_point(&mut z_obs_next, m, d, z_obs);
        self.commit_inducing(z_train, z_obs_next, m + 1)?;
        Ok(self.inducing.insert())
    }

    /// Factors the VFE system at the inducing set `z_train` (`m × d`) and
    /// commits it with `z_obs`. A factor failure, such as `K_mm` not
    /// factoring under the jitter policy, leaves the model unchanged.
    ///
    /// ADR 0005 applies first. This full factor keeps `L` aligned with
    /// `k(Z, Z)` so a long insert/delete sequence stays within the public
    /// 1e-12 check.
    fn commit_inducing(
        &mut self,
        z_train: Vec<f64>,
        z_obs: Vec<f64>,
        m: usize,
    ) -> Result<(), GprError> {
        let state = with_kernel_exp!(self.core.math, M => assemble_vfe::<M, P::Storage>(
            &self.core.kernel,
            self.core.jitter,
            self.core.likelihood,
            &self.core.x_train,
            self.core.n,
            self.core.d,
            &self.core.y_train,
            &z_train,
            m,
            &mut self.scratch.storage,
            &mut self.scratch.f64,
        ))?;
        self.core.z_train = z_train;
        self.core.z_obs = z_obs;
        self.core.m = m;
        self.apply_vfe(state)
    }

    /// Removes the inducing point identified by `id` and packs every buffer.
    ///
    /// Updates `K_mm` with a trailing cholupdate and rebuilds `A` / `B`
    /// from the reduced inducing set. The last remaining inducing point
    /// cannot be deleted.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::EmptyInput`] when `m == 1`, or
    /// [`GprError::InvalidInducingId`] when `id` is unknown or already
    /// deleted.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Fixed, GaussianLikelihood, Sgpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let fitted = Sgpr::new(
    ///     KernelSpec::from(RbfKernel::new(1.0)?),
    ///     GaussianLikelihood::new(0.1)?,
    /// )
    /// .with_optimizer(Fixed)
    /// .factor(
    ///     &[0.0, 1.0, 2.0, 3.0],
    ///     4,
    ///     1,
    ///     &[0.0, 1.0, 0.5, 0.25],
    ///     &[0.5, 2.5],
    ///     2,
    /// )
    /// .map_err(|(_, e)| e)?;
    /// let mut online = fitted.into_online();
    /// let id = online.inducing_ids()[0];
    /// online.delete_inducing(id)?;
    /// assert_eq!(online.m(), 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn delete_inducing(&mut self, id: InducingId) -> Result<(), GprError> {
        if self.core.m <= 1 {
            return Err(GprError::EmptyInput);
        }
        let idx = self.inducing.index_of(id)?;
        let mut state = self.vfe_state();
        inducing_delete(
            &mut state,
            self.core.likelihood.noise_variance(),
            &self.core.y_train,
            idx,
        )?;
        let (m, d) = (self.core.m, self.core.d);
        let z_train = remove_point(&self.core.z_train, m, d, idx);
        let z_obs = remove_point(&self.core.z_obs, m, d, idx);
        self.commit_inducing(z_train, z_obs, m - 1)?;
        self.inducing.remove_at(idx);
        Ok(())
    }

    /// Converts this model back to a batch sparse GPR with fixed inducing
    /// points.
    ///
    /// Point and inducing identifiers are discarded. Parameters stay
    /// kernel then likelihood `θ`. The snapshot `Z` is the current
    /// inducing coordinates.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Fixed, GaussianLikelihood, Sgpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let fitted = Sgpr::new(
    ///     KernelSpec::from(RbfKernel::new(1.0)?),
    ///     GaussianLikelihood::new(0.1)?,
    /// )
    /// .with_optimizer(Fixed)
    /// .factor(
    ///     &[0.0, 1.0, 2.0, 3.0],
    ///     4,
    ///     1,
    ///     &[0.0, 1.0, 0.5, 0.25],
    ///     &[0.5, 2.5],
    ///     2,
    /// )
    /// .map_err(|(_, e)| e)?;
    /// let online = fitted.into_online();
    /// let fitted = online.into_fitted();
    /// assert_eq!(fitted.n(), 4);
    /// # Ok(())
    /// # }
    /// ```
    pub fn into_fitted(self) -> FittedSgpr<O, FixedInducing, P> {
        FittedSgpr {
            core: self.core,
            scratch: self.scratch,
            optimizer: self.optimizer,
            inducing: PhantomData,
            k_mm_l: self.k_mm_l,
            a: self.a,
            b_l: self.b_l,
            w: self.w,
            predict_w: self.predict_w,
            k_diag_sum: self.k_diag_sum,
            a_frobenius2: self.a_frobenius2,
        }
    }
}

impl<O, P> OnlineSgpr<O, P>
where
    P: crate::precision::GpScalar,
    O: Clone + for<'a> Optimizer<SgprObjective<'a, O, FixedInducing, P>>,
{
    /// Re-runs the stored optimizer on the stored training data.
    ///
    /// Inducing coordinates stay fixed. The VFE system is rebuilt after the
    /// search.
    ///
    /// # Errors
    ///
    /// Same as [`Sgpr::fit`](crate::Sgpr::fit).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{GaussianLikelihood, Sgpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let fitted = Sgpr::new(
    ///     KernelSpec::from(RbfKernel::new(1.0)?),
    ///     GaussianLikelihood::new(0.1)?,
    /// )
    /// .fit(
    ///     &[0.0, 1.0, 2.0, 3.0],
    ///     4,
    ///     1,
    ///     &[0.0, 1.0, 0.5, 0.25],
    ///     &[0.5, 2.5],
    ///     2,
    /// )
    /// .map_err(|(_, e)| e)?;
    /// let mut online = fitted.into_online();
    /// online.refit()?;
    /// assert!(online.neg_log_marginal_likelihood()?.is_finite());
    /// # Ok(())
    /// # }
    /// ```
    pub fn refit(&mut self) -> Result<(), GprError> {
        let mut fitted = self.snapshot_fitted();
        fitted.optimize_hyperparameters()?;
        self.adopt_fitted(fitted);
        Ok(())
    }
}
