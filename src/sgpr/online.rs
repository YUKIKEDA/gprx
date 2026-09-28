//! Online collapsed variational SGPR.

use std::collections::HashMap;
use std::marker::PhantomData;

use faer::Mat;

use crate::error::GprError;
use crate::gpr::PointId;
use crate::gpr::online::PointRegistry;
use crate::kernel::ScalarOps;
use crate::kernel::{KernelScalar, KernelSpec};
use crate::likelihood::GaussianLikelihood;
use crate::linalg::{chol_rank1_downdate, chol_rank1_update, frobenius2};
use crate::objective::SgprObjective;
use crate::optimizer::{Lbfgs, Optimizer};
use crate::param::write_params;
use crate::precision::{DoublePrecision, ModelPrecision};
use crate::{PredictOptions, Prediction};

use super::FixedInducing;
use super::factor::{
    PublishSgprWeights, VfeState, append_column, append_point, assemble_vfe, inducing_delete,
    inducing_insert, kernel_column, kernel_diag_at, point_at, publish_sgpr_weights, refresh_w,
    remove_column, remove_point, solve_lmm, vfe_neg_log_marginal_likelihood, vfe_predict,
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
#[allow(private_bounds)]
pub struct OnlineSgpr<O = Lbfgs, M = crate::math::Accurate, P: ModelPrecision = DoublePrecision> {
    kernel: KernelSpec,
    likelihood: GaussianLikelihood,
    optimizer: O,
    x_obs: Vec<f64>,
    z_obs: Vec<f64>,
    y: Vec<f64>,
    k_mm_l: Mat<P::Storage>,
    a: Mat<P::Storage>,
    b_l: Mat<P::Storage>,
    w: Vec<P::Storage>,
    predict_w: Vec<P::Refine>,
    k_diag_sum: P::Storage,
    a_frobenius2: P::Storage,
    n: usize,
    m: usize,
    d: usize,
    registry: PointRegistry,
    inducing: InducingRegistry,
    _math: PhantomData<M>,
}

#[allow(private_bounds)]
impl<O, M, P> OnlineSgpr<O, M, P>
where
    M: crate::math::KernelMath,
    P: crate::precision::GpScalar + super::factor::MeanDot + PublishSgprWeights,
    crate::kernel::CompiledKernel<P::Storage>: crate::kernel::GramKernel<T = P::Storage>,
{
    pub(crate) fn from_fitted<I>(fitted: FittedSgpr<O, I, M, P>) -> Self {
        let registry = PointRegistry::from_count(fitted.n);
        let inducing = InducingRegistry::from_count(fitted.m);
        Self {
            kernel: fitted.kernel,
            likelihood: fitted.likelihood,
            optimizer: fitted.optimizer,
            x_obs: fitted.x_obs,
            z_obs: fitted.z_obs,
            y: fitted.y,
            k_mm_l: fitted.k_mm_l,
            a: fitted.a,
            b_l: fitted.b_l,
            w: fitted.w,
            predict_w: fitted.predict_w,
            k_diag_sum: fitted.k_diag_sum,
            a_frobenius2: fitted.a_frobenius2,
            n: fitted.n,
            m: fitted.m,
            d: fitted.d,
            registry,
            inducing,
            _math: PhantomData,
        }
    }

    fn snapshot_fitted(&self) -> FittedSgpr<O, FixedInducing, M, P>
    where
        O: Clone,
    {
        FittedSgpr {
            kernel: self.kernel.clone(),
            likelihood: self.likelihood,
            optimizer: self.optimizer.clone(),
            inducing: PhantomData,
            _math: PhantomData,
            x_obs: self.x_obs.clone(),
            z_obs: self.z_obs.clone(),
            y: self.y.clone(),
            k_mm_l: self.k_mm_l.clone(),
            a: self.a.clone(),
            b_l: self.b_l.clone(),
            w: self.w.clone(),
            predict_w: self.predict_w.clone(),
            k_diag_sum: self.k_diag_sum,
            a_frobenius2: self.a_frobenius2,
            n: self.n,
            m: self.m,
            d: self.d,
        }
    }

    fn adopt_fitted(&mut self, fitted: FittedSgpr<O, FixedInducing, M, P>) {
        self.kernel = fitted.kernel;
        self.likelihood = fitted.likelihood;
        self.optimizer = fitted.optimizer;
        self.x_obs = fitted.x_obs;
        self.z_obs = fitted.z_obs;
        self.y = fitted.y;
        self.k_mm_l = fitted.k_mm_l;
        self.a = fitted.a;
        self.b_l = fitted.b_l;
        self.w = fitted.w;
        self.predict_w = fitted.predict_w;
        self.k_diag_sum = fitted.k_diag_sum;
        self.a_frobenius2 = fitted.a_frobenius2;
        self.n = fitted.n;
        self.m = fitted.m;
        self.d = fitted.d;
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

    /// Rebuilds the stored VFE factors from the current `X` / `Z` / `θ`.
    ///
    /// ADR 0005 applies first. This refresh keeps `L` aligned with `k(Z, Z)`
    /// so a long insert/delete sequence stays within the public 1e-12 check.
    fn refresh_vfe(&mut self) -> Result<(), GprError> {
        let state = assemble_vfe::<M, P::Storage>(
            &self.kernel,
            self.likelihood,
            &self.x_obs,
            self.n,
            self.d,
            &self.y,
            &self.z_obs,
            self.m,
        )?;
        self.apply_vfe(state)
    }

    fn refresh_predict_w(&mut self) -> Result<(), GprError> {
        if P::REFINES_IN_F64 {
            self.predict_w = assemble_vfe::<M, f64>(
                &self.kernel,
                self.likelihood,
                &self.x_obs,
                self.n,
                self.d,
                &self.y,
                &self.z_obs,
                self.m,
            )?
            .w
            .into_iter()
            .map(P::Refine::from_f64)
            .collect();
            return Ok(());
        }
        self.predict_w = publish_sgpr_weights::<M, P>(
            &self.kernel,
            self.a.as_ref(),
            self.b_l.as_ref(),
            &self.w,
            &self.x_obs,
            &self.y,
            &self.z_obs,
            self.likelihood.noise_variance(),
            self.n,
            self.m,
            self.d,
        )?;
        Ok(())
    }

    fn recompute_w(&mut self) -> Result<(), GprError> {
        let w = {
            let mut y_cast = P::Storage::empty_rows();
            let y_s = P::Storage::storage_rows(&self.y, &mut y_cast);
            refresh_w(self.a.as_ref(), self.b_l.as_ref(), y_s)
        };
        self.w = w;
        self.refresh_predict_w()
    }

    /// Returns the number of training points.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Returns the number of inducing points.
    pub fn m(&self) -> usize {
        self.m
    }

    /// Returns the feature dimension.
    pub fn d(&self) -> usize {
        self.d
    }

    /// Returns the kernel whose hyperparameters this model owns.
    pub fn kernel(&self) -> &KernelSpec {
        &self.kernel
    }

    /// Returns the observation-noise model.
    pub fn likelihood(&self) -> &GaussianLikelihood {
        &self.likelihood
    }

    /// Returns the original training features in column-major order.
    pub fn x(&self) -> &[f64] {
        &self.x_obs
    }

    /// Returns the inducing features in column-major order.
    pub fn z(&self) -> &[f64] {
        &self.z_obs
    }

    /// Returns the original training targets.
    pub fn y(&self) -> &[f64] {
        &self.y
    }

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
        self.kernel.num_params() + self.likelihood.num_params()
    }

    /// Writes kernel `θ` then likelihood `θ` into `out`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if `out` is the wrong length
    /// or a custom leaf rejects the write.
    pub fn get_params(&self, out: &mut [f64]) -> Result<(), GprError> {
        write_params(&self.kernel, &self.likelihood, out)
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
            &self.y,
            self.k_diag_sum,
            self.a_frobenius2,
            self.likelihood.noise_variance(),
            self.n,
            self.m,
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
        vfe_predict::<M, P>(
            &self.kernel,
            &self.z_obs,
            self.k_mm_l.as_ref(),
            self.b_l.as_ref(),
            &self.predict_w,
            self.likelihood.noise_variance(),
            self.m,
            self.d,
            xs,
            n_rows,
            n_cols,
            options,
        )
    }

    /// Appends one training point at the current `θ` with a rank-1 VFE update.
    ///
    /// `x_new` has length [`Self::d`]. Inducing coordinates are not moved.
    /// The returned [`PointId`] is new and is never reused after a later
    /// [`Self::delete`].
    ///
    /// # Errors
    ///
    /// Returns [`GprError::DimensionMismatch`] if `x_new` is the wrong length,
    /// [`GprError::NonFiniteInput`] if a value is `NaN` or `Inf`, or
    /// [`GprError::EmptyInput`] if `d` is zero.
    pub fn insert(&mut self, x_new: &[f64], y_new: f64) -> Result<PointId, GprError> {
        if x_new.len() != self.d {
            return Err(GprError::DimensionMismatch {
                x_dim: x_new.len(),
                expected_dim: self.d,
            });
        }
        if self.d == 0 {
            return Err(GprError::EmptyInput);
        }
        if x_new.iter().any(|v| !v.is_finite()) || !y_new.is_finite() {
            return Err(GprError::NonFiniteInput);
        }
        let mut a_col =
            kernel_column::<M, P::Storage>(&self.kernel, &self.z_obs, self.m, x_new, self.d)?;
        solve_lmm(self.k_mm_l.as_ref(), a_col.as_mut());
        let mut v = vec![P::Storage::from_f64(0.0); self.m];
        for (i, slot) in v.iter_mut().enumerate() {
            *slot = a_col[(i, 0)];
        }
        self.a_frobenius2 += frobenius2(a_col.as_ref());
        self.k_diag_sum += kernel_diag_at::<P::Storage>(&self.kernel, x_new, self.d)?;
        self.a = append_column(&self.a, a_col.as_ref());
        chol_rank1_update(&mut self.b_l, &mut v);
        self.x_obs = append_point(&self.x_obs, self.n, self.d, x_new);
        self.y.push(y_new);
        self.n += 1;
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
        if self.n <= 1 {
            return Err(GprError::EmptyInput);
        }
        let idx = self.registry.index_of(id)?;
        let mut v = vec![P::Storage::from_f64(0.0); self.m];
        for (i, slot) in v.iter_mut().enumerate() {
            *slot = self.a[(i, idx)];
        }
        let x_pt = point_at(&self.x_obs, self.n, self.d, idx);
        let diag = kernel_diag_at::<P::Storage>(&self.kernel, &x_pt, self.d)?;
        let mut col_norm = P::Storage::from_f64(0.0);
        for value in &v {
            col_norm += *value * *value;
        }
        let x_next = remove_point(&self.x_obs, self.n, self.d, idx);
        let mut y_next = self.y.clone();
        y_next.remove(idx);
        let mut b_trial = self.b_l.clone();
        let mut v_trial = v;
        if chol_rank1_downdate(&mut b_trial, &mut v_trial) {
            self.k_diag_sum -= diag;
            self.a_frobenius2 -= col_norm;
            self.a = remove_column(&self.a, idx);
            self.b_l = b_trial;
            self.x_obs = x_next;
            self.y = y_next;
            self.n -= 1;
            self.recompute_w()?;
        } else {
            let state = assemble_vfe::<M, P::Storage>(
                &self.kernel,
                self.likelihood,
                &x_next,
                self.n - 1,
                self.d,
                &y_next,
                &self.z_obs,
                self.m,
            )?;
            self.x_obs = x_next;
            self.y = y_next;
            self.n -= 1;
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
    /// `z_new` has length [`Self::d`]. Training `X` / `y` are not moved.
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
        if z_new.len() != self.d {
            return Err(GprError::DimensionMismatch {
                x_dim: z_new.len(),
                expected_dim: self.d,
            });
        }
        if self.d == 0 {
            return Err(GprError::EmptyInput);
        }
        if z_new.iter().any(|v| !v.is_finite()) {
            return Err(GprError::NonFiniteInput);
        }
        let mut state = self.vfe_state();
        match inducing_insert::<M, _>(
            &mut state,
            &self.kernel,
            self.likelihood.noise_variance(),
            &self.x_obs,
            self.n,
            self.d,
            &self.y,
            &self.z_obs,
            self.m,
            z_new,
        ) {
            Ok(()) => {
                self.z_obs = append_point(&self.z_obs, self.m, self.d, z_new);
                self.apply_vfe(state)?;
                self.m += 1;
            }
            Err(GprError::CholeskyFailed { .. }) => {
                self.z_obs = append_point(&self.z_obs, self.m, self.d, z_new);
                self.m += 1;
            }
            Err(err) => return Err(err),
        }
        self.refresh_vfe()?;
        Ok(self.inducing.insert())
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
        if self.m <= 1 {
            return Err(GprError::EmptyInput);
        }
        let idx = self.inducing.index_of(id)?;
        let mut state = self.vfe_state();
        inducing_delete(&mut state, self.likelihood.noise_variance(), &self.y, idx)?;
        self.z_obs = remove_point(&self.z_obs, self.m, self.d, idx);
        self.apply_vfe(state)?;
        self.m -= 1;
        self.inducing.remove_at(idx);
        self.refresh_vfe()?;
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
    pub fn into_fitted(self) -> FittedSgpr<O, FixedInducing, M, P> {
        FittedSgpr {
            kernel: self.kernel,
            likelihood: self.likelihood,
            optimizer: self.optimizer,
            inducing: PhantomData,
            _math: PhantomData,
            x_obs: self.x_obs,
            z_obs: self.z_obs,
            y: self.y,
            k_mm_l: self.k_mm_l,
            a: self.a,
            b_l: self.b_l,
            w: self.w,
            predict_w: self.predict_w,
            k_diag_sum: self.k_diag_sum,
            a_frobenius2: self.a_frobenius2,
            n: self.n,
            m: self.m,
            d: self.d,
        }
    }
}

#[allow(private_bounds)]
impl<O, M, P> OnlineSgpr<O, M, P>
where
    M: crate::math::KernelMath,
    P: crate::precision::GpScalar + super::factor::MeanDot + PublishSgprWeights,
    crate::kernel::CompiledKernel<P::Storage>: crate::kernel::GramKernel<T = P::Storage>,
    O: Clone + for<'a> Optimizer<SgprObjective<'a, O, FixedInducing, M, P>>,
{
    /// Re-runs the stored optimizer on the stored training data.
    ///
    /// Inducing coordinates stay fixed. The VFE system is rebuilt after the
    /// search.
    ///
    /// # Errors
    ///
    /// Same as [`Sgpr<O, FixedInducing>::fit`].
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
