//! Online collapsed variational SGPR.

use crate::sparse::QueryDist;
use std::marker::PhantomData;

use faer::Mat;

use crate::error::GprError;
use crate::error::PersistErrorKind;
use crate::kernel::KernelScalar;
use crate::linalg::{chol_rank1_downdate, chol_rank1_update, frobenius2, solve_llt};
use crate::optimizer::{Lbfgs, Optimizer};
use crate::points::PointId;
use crate::points::{IdRegistry, PointRegistry, RegistryId};
use crate::policy::with_kernel_exp;
use crate::precision::{DoublePrecision, ModelPrecision};
use crate::sgpr::SgprObjective;
use crate::sparse::{
    KernelScratch, PredictScratch, SparseCore, SparseScratch, sparse_core_accessors,
    sparse_kernel_accessor, sparse_point_accessors,
};
use crate::{PredictOptions, Prediction, PredictiveCovariance};

use super::FixedInducing;
use super::factor::{
    VfeState, VfeSystem, a_times_y, append_point, assemble_vfe, assemble_vfe_with_f64_w,
    inducing_delete, inducing_insert, kernel_column, kernel_diag_at, point_at,
    predict_vfe_covariance, predict_vfe_into, publish_sgpr_weights, push_column,
    remove_column_in_place, remove_point, solve_lmm, vfe_loo, vfe_neg_log_marginal_likelihood,
};
use super::fitted::FittedSgpr;

/// Represents the stable identity of one inducing point on [`OnlineSgpr`].
///
/// [`crate::FittedSgpr::into_online`] assigns identifiers `0.. m-1` in buffer order. Later
/// [`OnlineSgpr::insert_inducing`] values increase monotonically and are never reused after
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

impl RegistryId for InducingId {
    const PERSIST_KEY: &'static str = "inducing_ids";

    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    fn raw(self) -> u64 {
        self.0
    }

    fn unknown() -> GprError {
        GprError::InvalidInducingId
    }
}

pub(crate) type InducingRegistry = IdRegistry<InducingId>;

/// Represents the online collapsed variational SGPR after [`FittedSgpr::into_online`].
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
    /// Everything an online update writes; [`OnlineSgpr::atomically`]
    /// copies and restores it as one value.
    pub(super) state: OnlineState<P>,
    /// Kernel scratch kept between `&mut self` calls.
    pub(super) scratch: SparseScratch<P::Storage>,
    optimizer: O,
}

/// The fields of an [`OnlineSgpr`] an online update writes. A field added
/// here is undone by [`OnlineSgpr::atomically`] with the rest.
#[derive(Clone, Debug)]
pub(super) struct OnlineState<P: ModelPrecision> {
    pub(super) core: SparseCore,
    k_mm_l: Mat<P::Storage>,
    a: Mat<P::Storage>,
    b_l: Mat<P::Storage>,
    w: Vec<P::Storage>,
    predict_w: Vec<P::Refine>,
    k_diag_sum: P::Storage,
    a_frobenius2: P::Storage,
    /// `A y` in `f64`, kept through rank-1 updates so `w = B⁻¹ A y` costs
    /// one `O(m²)` solve instead of a pass over all `n` columns.
    ay: Vec<f64>,
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
        let ay = a_times_y(fitted.a.as_ref(), &fitted.core.y_train);
        Self {
            state: OnlineState {
                ay,
                core: fitted.core,
                k_mm_l: fitted.k_mm_l,
                a: fitted.a,
                b_l: fitted.b_l,
                w: fitted.w,
                predict_w: fitted.predict_w,
                k_diag_sum: fitted.k_diag_sum,
                a_frobenius2: fitted.a_frobenius2,
                registry,
                inducing,
            },
            scratch: fitted.scratch,
            optimizer: fitted.optimizer,
        }
    }

    fn snapshot_fitted(&mut self) -> FittedSgpr<O, FixedInducing, P>
    where
        O: Clone,
    {
        FittedSgpr {
            core: self.state.core.clone(),
            // Lent for the call; `adopt_fitted` takes it back.
            scratch: std::mem::take(&mut self.scratch),
            optimizer: self.optimizer.clone(),
            inducing: PhantomData,
            _kernel: PhantomData,
            k_mm_l: self.state.k_mm_l.clone(),
            a: self.state.a.clone(),
            b_l: self.state.b_l.clone(),
            w: self.state.w.clone(),
            predict_w: self.state.predict_w.clone(),
            k_diag_sum: self.state.k_diag_sum,
            a_frobenius2: self.state.a_frobenius2,
        }
    }

    fn adopt_fitted(&mut self, fitted: FittedSgpr<O, FixedInducing, P>) {
        self.state.core = fitted.core;
        self.scratch = fitted.scratch;
        self.optimizer = fitted.optimizer;
        self.state.k_mm_l = fitted.k_mm_l;
        self.state.a = fitted.a;
        self.state.b_l = fitted.b_l;
        self.state.w = fitted.w;
        self.state.predict_w = fitted.predict_w;
        self.state.k_diag_sum = fitted.k_diag_sum;
        self.state.a_frobenius2 = fitted.a_frobenius2;
        self.state.ay = a_times_y(self.state.a.as_ref(), &self.state.core.y_train);
    }

    /// Runs one online update so that it either completes or changes nothing.
    ///
    /// Every update runs its failing steps (kernel evaluation, transforms,
    /// re-assembly, Cholesky) before its first write, and the writes after
    /// that cannot fail, except one: a precision that refines in `f64`
    /// ([`ModelPrecision::REFINES_IN_F64`]) assembles its predict weights
    /// again from the updated data, and that can fail after the buffers
    /// changed. For such a precision this copies [`OnlineState`], every
    /// field an update writes, and puts it back on failure. Other precisions
    /// skip the copy, so their updates allocate nothing for it; they rely on
    /// every failing step coming before the first write.
    pub(super) fn atomically<R>(
        &mut self,
        update: impl FnOnce(&mut Self) -> Result<R, GprError>,
    ) -> Result<R, GprError> {
        if !P::REFINES_IN_F64 {
            return update(self);
        }
        let undo = self.state.clone();
        let result = update(self);
        if result.is_err() {
            self.state = undo;
        }
        result
    }

    fn vfe_state(&self) -> VfeState<P::Storage> {
        VfeState::<P::Storage> {
            k_mm_l: self.state.k_mm_l.clone(),
            a: self.state.a.clone(),
            b_l: self.state.b_l.clone(),
            w: self.state.w.clone(),
            k_diag_sum: self.state.k_diag_sum,
            a_frobenius2: self.state.a_frobenius2,
        }
    }

    fn apply_vfe(&mut self, state: VfeState<P::Storage>) -> Result<(), GprError> {
        self.set_vfe(state);
        self.refresh_predict_w()
    }

    /// Takes `state` as the VFE system. The predict weights are the caller's.
    fn set_vfe(&mut self, state: VfeState<P::Storage>) {
        self.state.k_mm_l = state.k_mm_l;
        self.state.a = state.a;
        self.state.b_l = state.b_l;
        self.state.w = state.w;
        self.state.k_diag_sum = state.k_diag_sum;
        self.state.a_frobenius2 = state.a_frobenius2;
        self.state.ay = a_times_y(self.state.a.as_ref(), &self.state.core.y_train);
    }

    fn refresh_predict_w(&mut self) -> Result<(), GprError> {
        if P::REFINES_IN_F64 {
            self.state.predict_w =
                with_kernel_exp!(self.state.core.math, M => assemble_vfe::<M, f64>(
                    &self.state.core.kernel,
                    self.state.core.jitter,
                    self.state.core.likelihood,
                    &self.state.core.x_train,
                    self.state.core.n,
                    self.state.core.d,
                    &self.state.core.y_train,
                    &self.state.core.z_train,
                    self.state.core.m,
                    self.state.core.dist.as_ref(),
                    &mut self.scratch.f64,
                    &mut KernelScratch::new(),
                ))?
                .w
                .into_iter()
                .map(P::Refine::from_f64)
                .collect();
            return Ok(());
        }
        self.state.predict_w = with_kernel_exp!(self.state.core.math, M => publish_sgpr_weights::<M, P>(
            &self.state.core.kernel,
            self.state.core.jitter,
            self.state.a.as_ref(),
            self.state.b_l.as_ref(),
            &self.state.w,
            &self.state.core.x_train,
            &self.state.core.y_train,
            &self.state.core.z_train,
            self.state.core.likelihood.noise_variance(),
            self.state.core.n,
            self.state.core.m,
            self.state.core.d,
            self.state.core.dist.as_ref(),
        ))?;
        Ok(())
    }

    /// `w = B⁻¹ A y` from the kept `A y`: one `O(m²)` solve.
    fn recompute_w(&mut self) -> Result<(), GprError> {
        let m = self.state.core.m;
        let mut rhs = Mat::<P::Storage>::zeros(m, 1);
        for (row, value) in self.state.ay.iter().enumerate() {
            rhs[(row, 0)] = P::Storage::from_f64(*value);
        }
        solve_llt(self.state.b_l.as_ref(), rhs.as_mut());
        self.state.w.clear();
        self.state.w.extend((0..m).map(|row| rhs[(row, 0)]));
        self.refresh_predict_w()
    }

    sparse_core_accessors!(state.core);
    sparse_point_accessors!(state.core);
    sparse_kernel_accessor!(state.core);

    /// Returns training-point identifiers in buffer order.
    ///
    /// See the example on [`OnlineSgpr`].
    pub fn point_ids(&self) -> &[PointId] {
        self.state.registry.ids()
    }

    /// Returns inducing-point identifiers in buffer order.
    ///
    /// See the example on [`OnlineSgpr`].
    pub fn inducing_ids(&self) -> &[InducingId] {
        self.state.inducing.ids()
    }

    /// Returns the concatenated kernel and likelihood parameter count.
    ///
    /// Inducing coordinates are not counted.
    ///
    /// See the example on [`OnlineSgpr`].
    pub fn num_params(&self) -> usize {
        self.state.core.theta_len()
    }

    /// Writes kernel `θ` then likelihood `θ` into `out`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::LengthMismatch`] if `out` is the wrong length
    /// or a custom leaf rejects the write.
    ///
    /// See the example on [`OnlineSgpr`].
    pub fn get_params(&self, out: &mut [f64]) -> Result<(), GprError> {
        self.state.core.read_theta(out)
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
    ///
    /// See the example on [`OnlineSgpr`].
    pub fn set_params(&mut self, params: &[f64]) -> Result<(), GprError>
    where
        O: Clone,
    {
        let mut fitted = self.snapshot_fitted();
        fitted.set_params(params)?;
        self.adopt_fitted(fitted);
        Ok(())
    }

    /// Sets parameters, rebuilds the VFE system, and writes `∂L/∂θ` of the negative ELBO.
    ///
    /// `params` and `out` match [`Self::get_params`].
    ///
    /// # Errors
    ///
    /// Same as [`FittedSgpr::value_and_gradient_into`].
    ///
    /// See the example on [`OnlineSgpr`].
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
    ///
    /// See the example on [`OnlineSgpr`].
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
    ///
    /// See the example on [`OnlineSgpr`].
    pub fn neg_log_marginal_likelihood(&self) -> Result<f64, GprError> {
        vfe_neg_log_marginal_likelihood(
            self.state.a.as_ref(),
            self.state.b_l.as_ref(),
            &self.state.w,
            &self.state.core.y_train,
            self.state.k_diag_sum,
            self.state.a_frobenius2,
            self.state.core.likelihood.noise_variance(),
            self.state.core.n,
            self.state.core.m,
        )
    }

    /// Predicts at `xs` with [`PredictOptions::default`] (observation variance).
    ///
    /// # Errors
    ///
    /// Same as [`FittedSgpr::predict`].
    ///
    /// See the example on [`OnlineSgpr`].
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
    ///
    /// See the example on [`OnlineSgpr`].
    pub fn predict_with(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        options: PredictOptions,
    ) -> Result<Prediction<P::Refine>, GprError> {
        let mut out = Prediction::default();
        predict_vfe_into::<P>(
            &self.state.core,
            &VfeSystem::new(
                &self.state.core,
                self.state.k_mm_l.as_ref(),
                self.state.b_l.as_ref(),
                &self.state.predict_w,
            ),
            xs,
            n_rows,
            n_cols,
            &QueryDist::default(),
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
    ///
    /// See the example on [`OnlineSgpr`].
    pub fn predict_covariance_with(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        options: PredictOptions,
    ) -> Result<PredictiveCovariance<P::Refine>, GprError> {
        predict_vfe_covariance::<P>(
            &self.state.core,
            &VfeSystem::new(
                &self.state.core,
                self.state.k_mm_l.as_ref(),
                self.state.b_l.as_ref(),
                &self.state.predict_w,
            ),
            xs,
            n_rows,
            n_cols,
            &QueryDist::default(),
            options,
        )
    }

    /// Draws posterior samples at `xs` from [`Self::predict_covariance`].
    ///
    /// Each column of the returned column-major `m × n_draws` matrix is
    /// `μ + L z` with `z ∼ N(0, I)` and `L` the Cholesky factor of the
    /// posterior covariance, the same draw as [`crate::FittedGpr::sample`].
    /// `seed` is the start state of gprx's seeded generator (Xoshiro256++; the same seed gives the same draws on every platform). Zero draws
    /// returns an empty vector after the covariance is formed.
    ///
    /// # Errors
    ///
    /// Same as [`FittedSgpr::predict`], plus [`GprError::CholeskyFailed`] with
    /// [`CholeskyStage::Predict`](crate::CholeskyStage::Predict) if the
    /// posterior covariance cannot be factored after the retries of
    /// [`Self::jitter_policy`].
    ///
    /// See the example on [`OnlineSgpr`].
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
    ///
    /// See the example on [`OnlineSgpr`].
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
            .draw(n_draws, seed, self.state.core.jitter)
    }

    /// Returns the leave-one-out mean and observation variance at every training point.
    ///
    /// `p(y_i | X, y_{-i}, θ, Z)` of the collapsed VFE posterior: the
    /// optimal `q(u)` without point `i` at fixed `θ` and `Z`, predicted at
    /// `x_i`. It is a rank-1 downdate of `B = σn² I + A Aᵀ` per point
    /// (Sherman–Morrison), `O(n m²)` in all. At `Z = X` it matches
    /// [`crate::FittedGpr::loo_predict`]. Mean and variance are
    /// inverse-transformed like [`Self::predict`].
    ///
    /// # Errors
    ///
    /// Returns [`GprError::NonPositiveDefiniteMatrix`] if a downdated `B` is
    /// not positive definite, or [`GprError::CholeskyFailed`] if an `f32` storage cannot factor `K_mm` again in `f64`.
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
    /// let loo = fitted.loo_predict()?;
    /// assert_eq!(loo.mean.len(), fitted.n());
    /// # Ok(())
    /// # }
    /// ```
    pub fn loo_predict(&self) -> Result<Prediction<P::Refine>, GprError> {
        self.loo_predict_with(PredictOptions::default())
    }

    /// Returns leave-one-out mean and variance with an explicit variance kind.
    ///
    /// Latent variance is the VFE variance of `f(x_i)` without point `i`;
    /// observation variance adds `σn²`.
    ///
    /// # Errors
    ///
    /// Same as [`Self::loo_predict`].
    ///
    /// See the example on [`OnlineSgpr`].
    pub fn loo_predict_with(
        &self,
        options: PredictOptions,
    ) -> Result<Prediction<P::Refine>, GprError> {
        vfe_loo::<P>(
            &self.state.core,
            self.state.a.as_ref(),
            self.state.b_l.as_ref(),
            &self.state.w,
            options,
        )
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
    ///
    /// See the example on [`OnlineSgpr`].
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
    ///
    /// See the example on [`OnlineSgpr`].
    pub fn predict_with_into(
        &mut self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        options: PredictOptions,
        out: &mut Prediction<P::Refine>,
    ) -> Result<(), GprError> {
        predict_vfe_into::<P>(
            &self.state.core,
            &VfeSystem::new(
                &self.state.core,
                self.state.k_mm_l.as_ref(),
                self.state.b_l.as_ref(),
                &self.state.predict_w,
            ),
            xs,
            n_rows,
            n_cols,
            &QueryDist::default(),
            options,
            &mut self.scratch.predict,
            out,
        )
    }

    /// Appends one training point at the current `θ` with a rank-1 VFE update.
    ///
    /// Costs `O(m² + n·d)`, amortized: `A` grows in place, `A y` is updated
    /// with the new column, and `w = B⁻¹ A y` is one solve with the updated
    /// factor of `B`. The `O(n·d)` part moves the column-major training `X`.
    /// A precision that refines in `f64` ([`crate::MixedPrecision`]) also
    /// assembles its `f64` predict weights again from all `n` points,
    /// `O(n·m²)`: its weights are the exact `f64` solution, which no rank-1
    /// update of the stored `f32` factor reproduces.
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
    /// [`GprError::NonFiniteInput`] if a value is `NaN` or `Inf`,
    /// [`GprError::IndexOutOfRange`] if no new [`PointId`] is left (only a
    /// loaded `next_point_id` near `u64::MAX` reaches this), or
    /// [`GprError::EmptyInput`] if `d` is zero.
    ///
    /// See the example on [`OnlineSgpr`].
    pub fn insert(&mut self, x_new: &[f64], y_new: f64) -> Result<PointId, GprError> {
        if x_new.len() != self.state.core.d {
            return Err(GprError::DimensionMismatch {
                x_dim: x_new.len(),
                expected_dim: self.state.core.d,
            });
        }
        if self.state.core.d == 0 {
            return Err(GprError::EmptyInput);
        }
        if x_new.iter().any(|v| !v.is_finite()) || !y_new.is_finite() {
            return Err(GprError::NonFiniteInput);
        }
        self.state.registry.require_room()?;
        let mut mapped = std::mem::take(&mut self.scratch.point);
        let result = self.atomically(|model| model.insert_mapped(x_new, y_new, &mut mapped));
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
        self.state.core.map_point(x_obs, mapped)?;
        let x_new = mapped.as_slice();
        let y_new = self.state.core.map_target(y_obs)?;
        let mut a_col = with_kernel_exp!(self.state.core.math, M => kernel_column::<M, P::Storage>(
            &self.state.core.kernel,
            &self.state.core.z_train,
            self.state.core.m,
            x_new,
            self.state.core.d,
            &mut self.scratch.storage,
        ))?;
        solve_lmm(self.state.k_mm_l.as_ref(), a_col.as_mut());
        let mut v = vec![P::Storage::from_f64(0.0); self.state.core.m];
        for (i, slot) in v.iter_mut().enumerate() {
            *slot = a_col[(i, 0)];
        }
        let k_diag =
            kernel_diag_at::<P::Storage>(&self.state.core.kernel, x_new, self.state.core.d)?;
        // No step below fails until the predict weights (see `atomically`).
        self.state.a_frobenius2 += frobenius2(a_col.as_ref());
        self.state.k_diag_sum += k_diag;
        for (row, slot) in self.state.ay.iter_mut().enumerate() {
            *slot += a_col[(row, 0)].to_f64() * y_new;
        }
        push_column(&mut self.state.a, a_col.as_ref());
        chol_rank1_update(&mut self.state.b_l, &mut v);
        append_point(
            &mut self.state.core.x_train,
            self.state.core.n,
            self.state.core.d,
            x_new,
        );
        append_point(
            &mut self.state.core.x_obs,
            self.state.core.n,
            self.state.core.d,
            x_obs,
        );
        self.state.core.y_train.push(y_new);
        self.state.core.y_obs.push(y_obs);
        self.state.core.n += 1;
        self.recompute_w()?;
        Ok(self.state.registry.insert())
    }

    /// Removes the training point identified by `id` and packs every buffer.
    ///
    /// Updates `B` with a rank-1 downdate. If that loses positive
    /// definiteness, the remaining points are factored again. The last
    /// remaining point cannot be deleted.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InsufficientData`] when `n == 1`, or
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
        if self.state.core.n <= 1 {
            return Err(GprError::InsufficientData {
                n: self.state.core.n,
                min: 2,
            });
        }
        let idx = self.state.registry.index_of(id)?;
        self.atomically(|model| model.delete_at(idx))
    }

    /// [`Self::delete`] of the point at buffer index `idx`.
    fn delete_at(&mut self, idx: usize) -> Result<(), GprError> {
        let mut v = vec![P::Storage::from_f64(0.0); self.state.core.m];
        for (i, slot) in v.iter_mut().enumerate() {
            *slot = self.state.a[(i, idx)];
        }
        let x_pt = point_at(
            &self.state.core.x_train,
            self.state.core.n,
            self.state.core.d,
            idx,
        );
        let diag = kernel_diag_at::<P::Storage>(&self.state.core.kernel, &x_pt, self.state.core.d)?;
        let mut col_norm = P::Storage::from_f64(0.0);
        for value in &v {
            col_norm += *value * *value;
        }
        let x_next = remove_point(
            &self.state.core.x_train,
            self.state.core.n,
            self.state.core.d,
            idx,
        );
        let mut y_next = self.state.core.y_train.clone();
        y_next.remove(idx);
        let x_obs_next = remove_point(
            &self.state.core.x_obs,
            self.state.core.n,
            self.state.core.d,
            idx,
        );
        let mut y_obs_next = self.state.core.y_obs.clone();
        y_obs_next.remove(idx);
        let mut b_trial = self.state.b_l.clone();
        let mut v_trial = v;
        if chol_rank1_downdate(&mut b_trial, &mut v_trial) {
            self.state.k_diag_sum -= diag;
            self.state.a_frobenius2 -= col_norm;
            let y_idx = self.state.core.y_train[idx];
            for (row, slot) in self.state.ay.iter_mut().enumerate() {
                *slot -= self.state.a[(row, idx)].to_f64() * y_idx;
            }
            remove_column_in_place(&mut self.state.a, idx);
            self.state.b_l = b_trial;
            self.state.core.x_train = x_next;
            self.state.core.y_train = y_next;
            self.state.core.x_obs = x_obs_next;
            self.state.core.y_obs = y_obs_next;
            self.state.core.n -= 1;
            self.recompute_w()?;
        } else {
            self.delete_by_reassembly(x_next, y_next, x_obs_next, y_obs_next)?;
        }
        self.state.registry.remove_at(idx);
        Ok(())
    }

    /// The [`Self::delete`] path when the downdate of `B` fails: assembles
    /// the VFE system again from the remaining points (one fewer than now)
    /// and publishes its predict weights with it.
    pub(super) fn delete_by_reassembly(
        &mut self,
        x_next: Vec<f64>,
        y_next: Vec<f64>,
        x_obs_next: Vec<f64>,
        y_obs_next: Vec<f64>,
    ) -> Result<(), GprError> {
        let state = with_kernel_exp!(self.state.core.math, M => assemble_vfe::<M, P::Storage>(
            &self.state.core.kernel,
            self.state.core.jitter,
            self.state.core.likelihood,
            &x_next,
            self.state.core.n - 1,
            self.state.core.d,
            &y_next,
            &self.state.core.z_train,
            self.state.core.m,
            self.state.core.dist.as_ref(),
            &mut self.scratch.storage,
            &mut self.scratch.f64,
        ))?;
        self.state.core.x_train = x_next;
        self.state.core.y_train = y_next;
        self.state.core.x_obs = x_obs_next;
        self.state.core.y_obs = y_obs_next;
        self.state.core.n -= 1;
        self.apply_vfe(state)
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
    /// [`GprError::EmptyInput`] if `d` is zero,
    /// [`GprError::IndexOutOfRange`] if no new [`InducingId`] is left (only a
    /// loaded `next_inducing_id` near `u64::MAX` reaches this), or
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
        if z_new.len() != self.state.core.d {
            return Err(GprError::DimensionMismatch {
                x_dim: z_new.len(),
                expected_dim: self.state.core.d,
            });
        }
        if self.state.core.d == 0 {
            return Err(GprError::EmptyInput);
        }
        if z_new.iter().any(|v| !v.is_finite()) {
            return Err(GprError::NonFiniteInput);
        }
        self.state.inducing.require_room()?;
        self.atomically(|model| model.insert_inducing_checked(z_new))
    }

    /// [`Self::insert_inducing`] after the checks.
    fn insert_inducing_checked(&mut self, z_new: &[f64]) -> Result<InducingId, GprError> {
        let z_obs = z_new;
        let mut z_new = Vec::with_capacity(z_obs.len());
        self.state.core.map_point(z_obs, &mut z_new)?;
        let z_new = z_new.as_slice();
        let mut state = self.vfe_state();
        match with_kernel_exp!(self.state.core.math, M => inducing_insert::<M, _>(
            &mut state,
            &self.state.core.kernel,
            self.state.core.likelihood.noise_variance(),
            &self.state.core.x_train,
            self.state.core.n,
            self.state.core.d,
            &self.state.core.y_train,
            &self.state.core.z_train,
            self.state.core.m,
            z_new,
            &mut self.scratch.storage,
        )) {
            Ok(()) | Err(GprError::CholeskyFailed { .. }) => {}
            Err(err) => return Err(err),
        }
        let (m, d) = (self.state.core.m, self.state.core.d);
        let mut z_train = self.state.core.z_train.clone();
        append_point(&mut z_train, m, d, z_new);
        let mut z_obs_next = self.state.core.z_obs.clone();
        append_point(&mut z_obs_next, m, d, z_obs);
        self.commit_inducing(z_train, z_obs_next, m + 1)?;
        Ok(self.state.inducing.insert())
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
        let (state, w64) = with_kernel_exp!(self.state.core.math, M => assemble_vfe_with_f64_w::<M, P::Storage>(
            &self.state.core.kernel,
            self.state.core.jitter,
            self.state.core.likelihood,
            &self.state.core.x_train,
            self.state.core.n,
            self.state.core.d,
            &self.state.core.y_train,
            &z_train,
            m,
            self.state.core.dist.as_ref(),
            &mut self.scratch.storage,
            &mut self.scratch.f64,
        ))?;
        self.state.core.z_train = z_train;
        self.state.core.z_obs = z_obs;
        self.state.core.m = m;
        match w64 {
            // The `f64` weights of this assembly are the refined predict weights.
            Some(w64) if P::REFINES_IN_F64 => {
                self.state.predict_w = w64.into_iter().map(P::Refine::from_f64).collect();
                self.set_vfe(state);
                Ok(())
            }
            _ => self.apply_vfe(state),
        }
    }

    /// Removes the inducing point identified by `id` and packs every buffer.
    ///
    /// Updates `K_mm` with a trailing cholupdate and rebuilds `A` / `B`
    /// from the reduced inducing set. The last remaining inducing point
    /// cannot be deleted.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InsufficientData`] when `m == 1`, or
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
        if self.state.core.m <= 1 {
            return Err(GprError::InsufficientData {
                n: self.state.core.m,
                min: 2,
            });
        }
        let idx = self.state.inducing.index_of(id)?;
        self.atomically(|model| model.delete_inducing_at(idx))
    }

    /// [`Self::delete_inducing`] of the inducing point at index `idx`.
    fn delete_inducing_at(&mut self, idx: usize) -> Result<(), GprError> {
        let mut state = self.vfe_state();
        inducing_delete(
            &mut state,
            self.state.core.likelihood.noise_variance(),
            &self.state.core.y_train,
            idx,
        )?;
        let (m, d) = (self.state.core.m, self.state.core.d);
        let z_train = remove_point(&self.state.core.z_train, m, d, idx);
        let z_obs = remove_point(&self.state.core.z_obs, m, d, idx);
        self.commit_inducing(z_train, z_obs, m - 1)?;
        self.state.inducing.remove_at(idx);
        Ok(())
    }

    pub(crate) fn core(&self) -> &SparseCore {
        &self.state.core
    }

    pub(crate) fn point_registry(&self) -> &PointRegistry {
        &self.state.registry
    }

    pub(crate) fn inducing_registry(&self) -> &InducingRegistry {
        &self.state.inducing
    }

    /// The online model of a persist directory: `fitted` with the saved
    /// point and inducing identifiers.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::PersistFailed`] when an identifier list is
    /// invalid or its length is not `n` / `m`.
    pub(crate) fn from_persisted<I>(
        fitted: FittedSgpr<O, I, P>,
        points: PointRegistry,
        inducing: InducingRegistry,
    ) -> Result<Self, GprError> {
        let mut online = Self::from_fitted(fitted);
        if points.len() != online.state.core.n || inducing.len() != online.state.core.m {
            return Err(crate::persist::persist_err(
                PersistErrorKind::Config,
                format!(
                    "config has {} point ids and {} inducing ids, expected n = {} and m = {}",
                    points.len(),
                    inducing.len(),
                    online.state.core.n,
                    online.state.core.m
                ),
            ));
        }
        online.state.registry = points;
        online.state.inducing = inducing;
        Ok(online)
    }

    /// Writes this model to `dir` as `config.json` and `model.safetensors`.
    ///
    /// Stores the kernel, likelihood, kernel `exp`, `K_mm` jitter policy,
    /// precision, transforms (unfitted and fitted), the original `X`, `y`,
    /// and `Z`, and `Z` in transformed coordinates, and the point and inducing identifiers. The factors are
    /// not stored; [`crate::LoadedSgpr::load`] factors the system again at the saved `θ`
    /// and `Z`. Caller-defined kernels and transforms need their
    /// `persist_id` / `persist_state` and a [`crate::PersistRegistry`] entry.
    /// The optimizer and the inducing-point search are not stored.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::PersistFailed`] when the directory cannot be
    /// written or a kernel or transform has no persist form.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Fixed, GaussianLikelihood, Sgpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let mut model = Sgpr::new(KernelSpec::from(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?)
    ///     .with_optimizer(Fixed)
    ///     .factor(&[0.0, 1.0, 2.0], 3, 1, &[0.0, 1.0, 0.5], &[0.5, 1.5], 2)
    ///     .map_err(|(_, e)| e)?
    ///     .into_online();
    /// model.insert(&[3.0], 0.2)?;
    /// let dir = std::env::temp_dir().join(format!("gprx-doctest-save-online-{}", std::process::id()));
    /// let _ = std::fs::remove_dir_all(&dir);
    /// model.save(&dir)?;
    /// let loaded = gprx::LoadedSgpr::load(&dir, &gprx::PersistRegistry::new())?;
    /// assert_eq!(loaded.n(), model.n());
    /// let _ = std::fs::remove_dir_all(&dir);
    /// # Ok(())
    /// # }
    /// ```
    pub fn save(&self, dir: impl AsRef<std::path::Path>) -> Result<(), GprError> {
        crate::persist::save_online_sgpr(self, dir.as_ref())
    }

    /// Converts this model back to a batch sparse GPR with fixed inducing points.
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
            core: self.state.core,
            scratch: self.scratch,
            optimizer: self.optimizer,
            inducing: PhantomData,
            _kernel: PhantomData,
            k_mm_l: self.state.k_mm_l,
            a: self.state.a,
            b_l: self.state.b_l,
            w: self.state.w,
            predict_w: self.state.predict_w,
            k_diag_sum: self.state.k_diag_sum,
            a_frobenius2: self.state.a_frobenius2,
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
