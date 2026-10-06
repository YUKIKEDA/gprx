//! [`FittedGpr`]: the fitted Exact GPR and its public API.

use std::fmt;
use std::marker::PhantomData;

use dyn_stack::MemBuffer;
use faer::linalg::cholesky::llt;
use faer::{Mat, MatRef};

use crate::data::{pack_points, validate_training};
use crate::error::GprError;
use crate::error::PersistErrorKind;
use crate::gpr::GprObjective;
use crate::kernel::ScalarOps;
use crate::kernel::{
    DistanceKernel, DistanceSlot, DistanceSource, KernelScalar, KernelSpec, ModelKernel,
    ModelKernelParts, PointKernel, PointUse, SpecOf, spec_slots,
};
use crate::likelihood::GaussianLikelihood;
use crate::linalg::{faer_par_dims, solve_llt_in_place};
use crate::optimizer::{Fixed, Lbfgs, Optimizer};
use crate::persist::{self, PersistedModel};
use crate::precision::{DoublePrecision, GpScalar, StoredFactor};
use crate::transform::IdentityInput;
use crate::transform::{TargetTransform, Transform, UnfittedTarget, UnfittedTransform};
use crate::workspace::{FitWorkspace, QueryWorkspace};
use crate::{PredictOptions, Prediction, PredictiveCovariance};

use super::shared::{Query, bind_training};
use super::{ExactFit, Gpr, GprCore, LdltStore, LltStore, OnlineGpr, Policies, fit_buffers};

/// Stores a fitted Exact GPR: `L`, `α`, training `X` / `y`, kernel, and transforms.
///
/// [`Self::neg_log_marginal_likelihood`] is
/// `½ yᵀ α + ½ log|A| + (n/2) log(2π)` with `log|A| = 2 Σ log(L_ii)`.
/// [`Self::value_and_gradient_into`] rebuilds `L`, `α`, and `W` once and
/// writes `∂L/∂θ = -½ ⟨W, ∂A/∂θ⟩`. [`Self::predict`] returns the mean and
/// a diagonal variance; [`Self::predict_into`] writes into a reused
/// [`Prediction`](crate::Prediction) and crate-private query buffers (not the fit workspace).
/// [`Self::predict_covariance`] returns the query–query matrix as
/// [`PredictiveCovariance`](crate::PredictiveCovariance). [`Self::sample`] draws from that posterior.
/// [`Self::loo_predict`] is the GPML leave-one-out at every
/// training point, from `L` and `α`. [`Self::refit`] re-runs the trainer's
/// optimizer (`Gpr<O>`) or re-factors (`Gpr<Fixed>`) on the same training
/// data.
///
/// [`Clone`] copies the factorization, training observations, transforms, and
/// optimizer. [`Self::kernel`] is a shared reference; writes go through
/// [`Self::set_params`].
///
/// # Examples
///
/// ```rust
/// use gprx::kernel::{KernelSpec, RbfKernel};
/// use gprx::{Gpr, GaussianLikelihood};
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
/// let likelihood = GaussianLikelihood::new(0.1)?;
/// let fitted = Gpr::new(kernel, likelihood)
///     .fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
///     .map_err(|(_, e)| e)?;
/// let pred = fitted.predict(&[0.5], 1, 1)?;
/// assert_eq!(pred.mean.len(), 1);
/// # Ok(())
/// # }
/// ```
pub struct FittedGpr<O = Lbfgs, P: GpScalar = DoublePrecision, K: ModelKernel = KernelSpec> {
    pub(super) core: GprCore<P, K>,
    pub(super) optimizer: O,
    pub(super) store: LltStore<P>,
    pub(super) _kernel: PhantomData<K>,
}

impl<O, P, K> Clone for FittedGpr<O, P, K>
where
    O: Clone,
    P: GpScalar,
    K: ModelKernel,
{
    fn clone(&self) -> Self {
        Self {
            core: self.core.clone(),
            optimizer: self.optimizer.clone(),
            store: self.store.clone(),
            _kernel: PhantomData,
        }
    }
}

/// The training data of one fit: coordinates (`n_rows × n_cols`,
/// column-major; no columns for a kernel of supplied distances alone),
/// targets, and the sources of a distance kernel.
pub(crate) struct TrainInput<'a> {
    pub(crate) x: &'a [f64],
    pub(crate) n_rows: usize,
    pub(crate) n_cols: usize,
    pub(crate) y: &'a [f64],
    pub(crate) sources: Vec<DistanceSource<'a>>,
}

impl<'a> TrainInput<'a> {
    pub(crate) fn points(x: &'a [f64], n_rows: usize, n_cols: usize, y: &'a [f64]) -> Self {
        Self {
            x,
            n_rows,
            n_cols,
            y,
            sources: Vec::new(),
        }
    }

    /// Checks the coordinates and targets. A kernel that reads `points`
    /// needs at least one feature column.
    fn validate(&self, points: bool) -> Result<(), GprError> {
        if points {
            return validate_training(self.x, self.n_rows, self.n_cols, self.y);
        }
        crate::data::require_nonempty(self.n_rows)?;
        crate::data::require_count(self.x.len(), 0, "feature values")?;
        crate::data::require_count(self.y.len(), self.n_rows, "targets")?;
        crate::data::require_finite(self.y)
    }
}

impl<O, P, K> fmt::Debug for FittedGpr<O, P, K>
where
    O: fmt::Debug,
    P: GpScalar,
    K: ModelKernel,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FittedGpr")
            .field("n", &self.core.n)
            .field("d", &self.core.d)
            .field("kernel", &self.core.kernel)
            .field("likelihood", &self.core.likelihood)
            .field("optimizer", &self.optimizer)
            .field("distance_cache", &self.core.policies.distance_cache)
            .field("cholesky_buffer", &self.core.policies.cholesky_buffer)
            .field("math", &self.core.policies.math)
            .field("jitter_policy", &self.core.policies.jitter)
            .finish_non_exhaustive()
    }
}

impl<O, P, K> FittedGpr<O, P, K>
where
    P: GpScalar,
    K: ModelKernel,
{
    /// Checks `input`, fits the transforms, binds the training distances,
    /// and allocates the fit buffers.
    #[allow(clippy::result_large_err)] // failure returns the trainer so the caller can retry
    pub(crate) fn prepare(
        gpr: Gpr<O, P, K>,
        input: TrainInput<'_>,
    ) -> Result<Self, (Gpr<O, P, K>, GprError)> {
        if let Err(err) = input.validate(<K as ModelKernelParts>::POINTS) {
            return Err((gpr, err));
        }
        let TrainInput {
            x,
            n_rows,
            n_cols,
            y,
            sources,
        } = input;
        let sources = match bind_training::<P::Storage, P::Sources, _>(&gpr.kernel, sources, n_rows)
        {
            Ok(bound) => bound,
            Err(err) => return Err((gpr, err)),
        };
        let mut x_buf = x.to_vec();
        let x_fitted: Box<dyn Transform> = if n_cols == 0 {
            Box::new(IdentityInput)
        } else {
            match gpr.x_transform.clone_box().fit(&x_buf, n_rows, n_cols) {
                Ok(t) => t,
                Err(err) => return Err((gpr, err)),
            }
        };
        if n_cols > 0
            && let Err(err) = x_fitted.apply(&mut x_buf, n_rows, n_cols)
        {
            return Err((gpr, err));
        }
        let mut y_buf = y.to_vec();
        let y_fitted = match gpr.y_transform.clone_box().fit(&y_buf) {
            Ok(t) => t,
            Err(err) => return Err((gpr, err)),
        };
        if let Err(err) = y_fitted.transform(&mut y_buf) {
            return Err((gpr, err));
        }
        let compiled = gpr.kernel.compile_as::<P::Storage>();
        let workspace = match fit_buffers::<P, _>(n_rows, gpr.policies, &compiled) {
            Ok(ws) => ws,
            Err(err) => return Err((gpr, err)),
        };
        Ok(Self {
            core: GprCore {
                slots: spec_slots(&gpr.kernel),
                kernel: gpr.kernel,
                compiled,
                likelihood: gpr.likelihood,
                x_unfitted: gpr.x_transform,
                y_unfitted: gpr.y_transform,
                x_transform: x_fitted,
                y_transform: y_fitted,
                policies: gpr.policies,
                query: QueryWorkspace::new(),
                x_obs: x.to_vec(),
                y_obs: y.to_vec(),
                x: pack_points(&x_buf, n_rows, n_cols),
                y_train: y_buf,
                factor_alpha: vec![P::Storage::from_f64(0.0); n_rows],
                alpha: vec![P::Refine::from_f64(0.0); n_rows],
                x_cast: P::Storage::empty_cols(),
                y_cast: P::Storage::empty_rows(),
                sources,
                n: n_rows,
                d: n_cols,
            },
            optimizer: gpr.optimizer,
            store: LltStore::new(workspace),
            _kernel: PhantomData,
        })
    }

    /// Drops `L` / `α` / training data and returns a trainer with the current kernel, likelihood, transforms, optimizer, and policies.
    ///
    /// See the example on [`FittedGpr`].
    pub fn into_trainer(self) -> Gpr<O, P, K> {
        self.core.into_trainer(self.optimizer)
    }

    /// Converts this LLT factorization into an [`OnlineGpr`] for tail inserts.
    ///
    /// Writes `D[j] = L_jj²` and `L_ldlt[i,j] = L_llt[i,j] / L_jj`, then
    /// rebuilds `A = K + σn² I` on the online workspace. [`OnlineGpr::insert`]
    /// updates that LDLT in place.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::EmptyInput`] if `n` is zero, or
    /// [`GprError::CholeskyFailed`] if a diagonal of `L` is not positive.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Fixed, GaussianLikelihood, Gpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let fitted = Gpr::new(
    ///     KernelSpec::from(RbfKernel::new(1.0)?),
    ///     GaussianLikelihood::new(0.1)?,
    /// )
    /// .with_optimizer(Fixed)
    /// .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
    /// .map_err(|(_, e)| e)?;
    /// let mut online = fitted.into_online()?;
    /// online.insert(&[1.5], 0.5)?;
    /// let pred = online.predict(&[0.5], 1, 1)?;
    /// assert_eq!(pred.mean.len(), 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn into_online(self) -> Result<OnlineGpr<O, P, K>, GprError> {
        let n = self.core.n;
        let mut workspace = LdltStore::from_active(n)?;
        workspace.fill_ld_from_llt(self.chol_l(), n)?;
        workspace.factor_jitter = self.store.buffers.core().factor_jitter;
        LdltStore::set_f64_prefix(&mut workspace.y, &self.core.y_train);
        LdltStore::set_vector_prefix(&mut workspace.alpha, &self.core.factor_alpha);
        Ok(OnlineGpr::from_core(self.core, self.optimizer, workspace))
    }

    /// Returns the number of training points.
    ///
    /// See the example on [`FittedGpr`].
    pub fn n(&self) -> usize {
        self.core.n
    }

    /// Returns the observation-noise model.
    ///
    /// See the example on [`FittedGpr`].
    pub fn likelihood(&self) -> &GaussianLikelihood {
        &self.core.likelihood
    }

    /// Returns `α = A⁻¹ y` from the last successful fit.
    ///
    /// See the example on [`FittedGpr`].
    pub fn alpha(&self) -> &[P::Refine] {
        &self.core.alpha
    }

    /// Returns the original training targets.
    ///
    /// Values are on the scale passed to fit, before the target transform.
    ///
    /// See the example on [`FittedGpr`].
    pub fn y(&self) -> &[f64] {
        &self.core.y_obs
    }

    /// Writes this fitted model to `dir/config.json` and `dir/model.safetensors`.
    ///
    /// Omits `L` and `α`. [`crate::persist::LoadedGpr::load`] rebuilds them
    /// by factorizing. The Cholesky buffer policy is not written; load
    /// reconstructs [`crate::CholeskyBuffer::Retain`].
    ///
    /// # Errors
    ///
    /// Returns [`GprError::PersistFailed`] when the directory cannot be
    /// created or a Custom leaf / caller transform has no persist form.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{GaussianLikelihood, Gpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let fitted = Gpr::new(
    ///     KernelSpec::from(RbfKernel::new(1.0)?),
    ///     GaussianLikelihood::new(0.1)?,
    /// )
    /// .fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
    /// .map_err(|(_, e)| e)?;
    /// let dir = std::env::temp_dir().join(format!(
    ///     "gprx-doctest-save-{}",
    ///     std::process::id()
    /// ));
    /// let _ = std::fs::remove_dir_all(&dir);
    /// fitted.save(&dir)?;
    /// let _ = std::fs::remove_dir_all(&dir);
    /// # Ok(())
    /// # }
    /// ```
    pub fn save(&self, dir: impl AsRef<std::path::Path>) -> Result<(), GprError> {
        persist::save_fitted(self, dir.as_ref(), false)
    }

    /// Writes this fitted model including the Cholesky factor `L` and `α`.
    ///
    /// `L` is stored column-major. Its dtype is `F64` when storage is `f64`
    /// and `F32` when storage is `f32`. `α` uses the predict scalar: `F32`
    /// for [`crate::SinglePrecision`], `F64` for [`crate::DoublePrecision`]
    /// and [`crate::MixedPrecision`]. [`crate::persist::LoadedGpr::load`]
    /// keeps an `f64` factor memory-mapped.
    ///
    /// # Errors
    ///
    /// Same as [`Self::save`].
    ///
    /// See the example on [`FittedGpr`].
    pub fn save_with_factor(&self, dir: impl AsRef<std::path::Path>) -> Result<(), GprError> {
        persist::save_fitted(self, dir.as_ref(), true)
    }

    /// Replaces the optimizer used by a later [`Self::refit`].
    ///
    /// Does not write a solver into a persist directory. A model loaded as
    /// [`crate::persist::LoadedGpr`] is [`Fixed`]; call this before `refit`
    /// to search again. Incremental rebuilds follow the same rule as
    /// [`Gpr::with_optimizer`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::persist::{LoadedGpr, PersistRegistry};
    /// use gprx::{GaussianLikelihood, Gpr, Lbfgs};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let fitted = Gpr::new(
    ///     KernelSpec::from(RbfKernel::new(1.0)?),
    ///     GaussianLikelihood::new(0.1)?,
    /// )
    /// .fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
    /// .map_err(|(_, e)| e)?;
    /// let dir = std::env::temp_dir().join(format!(
    ///     "gprx-doctest-refit-{}",
    ///     std::process::id()
    /// ));
    /// let _ = std::fs::remove_dir_all(&dir);
    /// fitted.save(&dir)?;
    /// let LoadedGpr::Double(model) = LoadedGpr::load(&dir, &PersistRegistry::new())? else {
    ///     return Ok(());
    /// };
    /// let mut model = model.with_optimizer(Lbfgs::new());
    /// model.refit()?;
    /// let _ = std::fs::remove_dir_all(&dir);
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_optimizer<O2>(self, optimizer: O2) -> FittedGpr<O2, P, K> {
        FittedGpr {
            core: self.core,
            optimizer,
            store: self.store,
            _kernel: PhantomData,
        }
    }

    /// Diagonal jitter the current factor was built with.
    pub(crate) fn factor_jitter(&self) -> f64 {
        self.store.buffers.core().factor_jitter
    }

    pub(crate) fn policies(&self) -> Policies {
        self.core.policies
    }

    /// Returns the distance-cache policy carried from the trainer.
    ///
    /// See the example on [`FittedGpr`].
    pub fn distance_cache_policy(&self) -> crate::DistanceCachePolicy {
        self.core.policies.distance_cache
    }

    /// Returns the Cholesky buffer policy carried from the trainer.
    ///
    /// See the example on [`FittedGpr`].
    pub fn cholesky_buffer(&self) -> crate::CholeskyBuffer {
        self.core.policies.cholesky_buffer
    }

    /// Returns the kernel `exp` used by fit and predict.
    ///
    /// See the example on [`FittedGpr`].
    pub fn math(&self) -> crate::KernelExp {
        self.core.policies.math
    }

    /// Returns the jitter retries used when `K + σn² I` fails to factor.
    ///
    /// See the example on [`FittedGpr`].
    pub fn jitter_policy(&self) -> crate::JitterPolicy {
        self.core.policies.jitter
    }

    /// The kernel tree (with its distance leaves).
    pub(crate) fn kernel_spec(&self) -> &SpecOf<K> {
        &self.core.kernel
    }

    /// The feature count (`0` for a kernel of supplied distances alone).
    pub(crate) fn feature_dim(&self) -> usize {
        self.core.d
    }

    /// The training features on the caller's scale.
    pub(crate) fn x_obs(&self) -> &[f64] {
        &self.core.x_obs
    }

    /// The training squared distances (empty for a coordinate kernel).
    pub(crate) fn sources(&self) -> &P::Sources {
        &self.core.sources
    }

    pub(crate) fn x_unfitted(&self) -> &dyn UnfittedTransform {
        self.core.x_unfitted.as_ref()
    }

    pub(crate) fn y_unfitted(&self) -> &dyn UnfittedTarget {
        self.core.y_unfitted.as_ref()
    }

    pub(crate) fn x_transform(&self) -> &dyn Transform {
        self.core.x_transform.as_ref()
    }

    pub(crate) fn y_transform(&self) -> &dyn TargetTransform {
        self.core.y_transform.as_ref()
    }

    pub(crate) fn chol_l(&self) -> MatRef<'_, P::Storage> {
        self.store.l()
    }

    pub(crate) fn factor(&self) -> StoredFactor<'_, P::Storage> {
        StoredFactor::Llt(self.chol_l())
    }

    /// Lends the core and the LLT buffers to the fit code.
    pub(crate) fn fit_view(&mut self) -> ExactFit<'_, P, K> {
        ExactFit {
            core: &mut self.core,
            store: &mut self.store,
        }
    }

    /// Predicts `q` into `out` through the model's query buffers.
    pub(crate) fn predict_query_into(
        &mut self,
        q: Query<'_, P::Storage>,
        options: PredictOptions,
        out: &mut Prediction<P::Refine>,
    ) -> Result<(), GprError> {
        let (l, thread_scratch) = self.store.l_and_thread_scratch();
        self.core
            .predict_with_into(StoredFactor::Llt(l), thread_scratch, q, options, out)
    }

    /// Returns the negative log marginal likelihood of the last successful fit.
    ///
    /// Evaluates `½ yᵀ A⁻¹ y + ½ log|A| + (n/2) log(2π)` from the stored
    /// `α` and the Cholesky factor `L` in the workspace, using
    /// `log|A| = 2 Σ log(L_ii)`. `y` is the target after the target
    /// transform.
    ///
    /// # Errors
    ///
    /// The stored factor is already valid, so this returns [`Ok`] and does not return [`GprError`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Gpr, GaussianLikelihood};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    /// let likelihood = GaussianLikelihood::new(0.1)?;
    /// let gpr = Gpr::new(kernel, likelihood);
    /// let fitted = gpr.fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0]).map_err(|(_, e)| e)?;
    /// let nlml = fitted.neg_log_marginal_likelihood()?;
    /// assert!(nlml.is_finite());
    /// # Ok(())
    /// # }
    /// ```
    pub fn neg_log_marginal_likelihood(&self) -> Result<f64, GprError> {
        Ok(self
            .core
            .neg_log_marginal_likelihood(self.factor(), &self.core.factor_alpha))
    }

    /// Returns the concatenated kernel and likelihood parameter count.
    ///
    /// See the example on [`FittedGpr`].
    pub fn num_params(&self) -> usize {
        self.core.num_params()
    }

    /// Writes kernel `θ` then likelihood `θ` into `out`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::LengthMismatch`] if `out` is the wrong length
    /// or a custom leaf rejects the write.
    ///
    /// See the example on [`FittedGpr`].
    pub fn get_params(&self, out: &mut [f64]) -> Result<(), GprError> {
        self.core.get_params(out)
    }

    /// Sets kernel then likelihood `θ` and rebuilds `L` / `α`.
    ///
    /// `params` is kernel parameters followed by the likelihood parameter,
    /// matching [`Self::get_params`]. Transforms and training `X` / `y` are
    /// not changed. [`Self::kernel`] stays a shared reference; this is the
    /// write path. After success, [`Gpr<Fixed>::factor`] on
    /// [`Self::into_trainer`] with [`Self::x`] / [`Self::y`] rebuilds the
    /// same factorization from the stored observations.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::LengthMismatch`] if `params` is the wrong
    /// length, [`GprError::InvalidNoiseVariance`] if the likelihood `θ` is
    /// invalid, or [`GprError::CholeskyFailed`] if `A` cannot be factored.
    /// Kernel and likelihood `θ` are committed together only after `A`
    /// factors. A rejected slice or a Cholesky failure leaves stored `θ`
    /// and `L` / `α` unchanged.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Fixed, GaussianLikelihood, Gpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let mut fitted = Gpr::new(
    ///     KernelSpec::from(RbfKernel::new(1.0)?),
    ///     GaussianLikelihood::new(0.1)?,
    /// )
    /// .fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
    /// .map_err(|(_, e)| e)?;
    /// let mut params = [0.0; 2];
    /// fitted.get_params(&mut params)?;
    /// params[0] = 0.5_f64.ln();
    /// fitted.set_params(&params)?;
    /// let x = fitted.x().to_vec();
    /// let y = fitted.y().to_vec();
    /// let n = fitted.n();
    /// let d = fitted.d();
    /// let _fitted = fitted
    ///     .into_trainer()
    ///     .with_optimizer(Fixed)
    ///     .factor(&x, n, d, &y)
    ///     .map_err(|(_, e)| e)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_params(&mut self, params: &[f64]) -> Result<(), GprError> {
        self.fit_view().set_params(params)
    }

    #[cfg(test)]
    pub(crate) fn objective(&mut self) -> GprObjective<'_, P, K> {
        GprObjective::new(self.fit_view())
    }

    /// Sets kernel and likelihood `θ`, rebuilds `L` / `α` / `W`, and writes `∂L/∂θ`.
    ///
    /// `params` and `out` are kernel parameters followed by the likelihood
    /// parameter. One Cholesky produces `L` and `α`; `W = ααᵀ - A⁻¹` is
    /// formed from that factor. [`crate::CholeskyBuffer::Retain`] keeps `W` in a dedicated
    /// buffer. [`crate::CholeskyBuffer::Reuse`] writes `W` over `L` and this method
    /// refactors afterwards so [`Self::predict`] still sees `L`. Kernel
    /// `∂A/∂θ` goes through `exp_buf`. Product trees also use
    /// `kernel_scratch`. The returned value is the same as
    /// [`Self::neg_log_marginal_likelihood`] after a successful call.
    ///
    /// Training `X` / `y` come from [`Gpr::fit`]. Transforms
    /// are not re-fit.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::LengthMismatch`] if a slice length is wrong,
    /// [`GprError::InvalidNoiseVariance`] if the likelihood `θ` is invalid,
    /// [`GprError::CholeskyFailed`] if `A` cannot be factored, or
    /// [`GprError::UnsupportedKernelOperation`] if the compiled tree cannot
    /// evaluate at this `θ`. Distance-mode and points-mode product trees are
    /// supported. Kernel and likelihood `θ` are committed together only after
    /// `A` factors. A rejected slice or a Cholesky failure leaves stored `θ`
    /// unchanged.
    /// Cholesky failure restores `L` and `α` at the previous `θ` so this
    /// value stays a usable [`FittedGpr`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Gpr, GaussianLikelihood};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    /// let likelihood = GaussianLikelihood::new(0.1)?;
    /// let gpr = Gpr::new(kernel, likelihood);
    /// let mut fitted = gpr.fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0]).map_err(|(_, e)| e)?;
    /// let mut params = [0.0; 2];
    /// fitted.get_params(&mut params)?;
    /// let mut grad = [0.0; 2];
    /// let nlml = fitted.value_and_gradient_into(&params, &mut grad)?;
    /// assert!(nlml.is_finite());
    /// # Ok(())
    /// # }
    /// ```
    pub fn value_and_gradient_into(
        &mut self,
        params: &[f64],
        out: &mut [f64],
    ) -> Result<f64, GprError> {
        self.fit_view().value_and_gradient_into(params, out)
    }

    /// Writes the analytic NLML Hessian (row-major `p×p`) at `params`.
    ///
    /// `params` is kernel `θ` followed by likelihood `θ`. After a successful
    /// call the stored kernel and likelihood match `params`. [`crate::CholeskyBuffer::Reuse`]
    /// rebuilds `L` before return, matching [`Self::value_and_gradient_into`].
    ///
    /// # Errors
    ///
    /// Returns [`GprError`] when a slice length is wrong, `params` is
    /// rejected, or the Gram matrix does not factor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Gpr, GaussianLikelihood};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    /// let likelihood = GaussianLikelihood::new(0.1)?;
    /// let mut fitted = Gpr::new(kernel, likelihood)
    ///     .fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
    ///     .map_err(|(_, e)| e)?;
    /// let mut params = [0.0; 2];
    /// fitted.get_params(&mut params)?;
    /// let mut hess = [0.0; 4];
    /// fitted.hessian_into(&params, &mut hess)?;
    /// assert!(hess.iter().all(|h| h.is_finite()));
    /// # Ok(())
    /// # }
    /// ```
    pub fn hessian_into(&mut self, params: &[f64], out: &mut [f64]) -> Result<(), GprError> {
        self.fit_view().hessian_into(params, out)
    }

    /// Returns leave-one-out mean and observation variance at every training point.
    ///
    /// Uses the GPML identities `μ_i = y_i - α_i / Q_ii` and
    /// `σ_i² = 1 / Q_ii` with `Q = A⁻¹` and `A = K + σn² I`. This is
    /// `p(y_i | X, y_{-i}, θ)`, not a query at a new `x*`. Mean and
    /// variance are inverse-transformed like [`Self::predict`].
    ///
    /// # Errors
    ///
    /// Returns [`GprError::NonPositiveDefiniteMatrix`] if a diagonal of `A⁻¹`
    /// is not positive and finite.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Gpr, GaussianLikelihood};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    /// let likelihood = GaussianLikelihood::new(0.1)?;
    /// let gpr = Gpr::new(kernel, likelihood);
    /// let fitted = gpr.fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0]).map_err(|(_, e)| e)?;
    /// let loo = fitted.loo_predict()?;
    /// assert_eq!(loo.mean.len(), 2);
    /// # Ok(())
    /// # }
    /// ```
    pub fn loo_predict(&self) -> Result<Prediction<P::Refine>, GprError> {
        self.loo_predict_with(PredictOptions::default())
    }

    /// Returns leave-one-out mean and variance with an explicit variance kind.
    ///
    /// Observation variance is `1 / Q_ii`. Latent variance is
    /// `max(0, 1 / Q_ii - σn²)` in the transformed space, then both mean
    /// and variance are mapped back by the target transform.
    ///
    /// # Errors
    ///
    /// Same as [`Self::loo_predict`].
    ///
    /// See the example on [`FittedGpr`].
    pub fn loo_predict_with(
        &self,
        options: PredictOptions,
    ) -> Result<Prediction<P::Refine>, GprError> {
        self.core
            .loo_predict_with(self.factor(), &self.core.alpha, options)
    }
}

impl<O, P: GpScalar> FittedGpr<O, P> {
    /// Returns the kernel whose hyperparameters this model owns.
    ///
    /// See the example on [`FittedGpr`].
    pub fn kernel(&self) -> &KernelSpec {
        &self.core.kernel
    }

    /// Predicts at `xs` with [`PredictOptions::default`] (observation variance).
    ///
    /// `xs` is column-major with `n_rows` query points and `n_cols` features.
    /// Allocates query buffers for this call. Reuse [`Self::predict_into`]
    /// after a warmup call for a zero-allocation path. See [`Gpr`] for a
    /// complete fit→predict example.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::DimensionMismatch`] if `n_cols` differs from the
    /// training features, [`GprError::EmptyInput`] if a dimension is zero, or
    /// [`GprError::LengthMismatch`] / [`GprError::NonFiniteInput`] for a
    /// badly packed or non-finite `xs`.
    pub fn predict(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
    ) -> Result<Prediction<P::Refine>, GprError> {
        self.predict_with(xs, n_rows, n_cols, PredictOptions::default())
    }

    /// Writes [`Self::predict`] into `out`, reusing `mean` / `variance` capacity when the query length matches.
    ///
    /// # Errors
    ///
    /// Same as [`Self::predict`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Gpr, GaussianLikelihood, Prediction};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    /// let likelihood = GaussianLikelihood::new(0.1)?;
    /// let gpr = Gpr::new(kernel, likelihood);
    /// let mut fitted = gpr.fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0]).map_err(|(_, e)| e)?;
    /// let mut pred = Prediction::default();
    /// fitted.predict_into(&[0.5], 1, 1, &mut pred)?;
    /// assert_eq!(pred.mean.len(), 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn predict_into(
        &mut self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        out: &mut Prediction<P::Refine>,
    ) -> Result<(), GprError> {
        self.predict_with_into(xs, n_rows, n_cols, PredictOptions::default(), out)
    }

    /// Predicts at `xs` with an explicit variance kind.
    ///
    /// Latent variance is `k(x*, x*) - ‖L⁻¹ k_*‖²`. Observation variance adds
    /// `σn²` in the transformed space, then both mean and variance are mapped
    /// back by the target transform.
    ///
    /// # Errors
    ///
    /// Same as [`Self::predict`].
    ///
    /// See the example on [`FittedGpr`].
    pub fn predict_with(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        options: PredictOptions,
    ) -> Result<Prediction<P::Refine>, GprError> {
        let mut out = Prediction::default();
        self.core.write_prediction(
            self.factor(),
            &self.core.alpha,
            Query::points(xs, n_rows, n_cols),
            options,
            &mut out,
        )?;
        Ok(out)
    }

    /// Writes [`Self::predict_with`] into `out`, reusing `mean` / `variance` capacity when the query length matches.
    ///
    /// # Errors
    ///
    /// Same as [`Self::predict`].
    ///
    /// See the example on [`FittedGpr`].
    pub fn predict_with_into(
        &mut self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        options: PredictOptions,
        out: &mut Prediction<P::Refine>,
    ) -> Result<(), GprError> {
        self.predict_query_into(Query::points(xs, n_rows, n_cols), options, out)
    }

    /// Returns the predictive mean and query–query covariance at `xs`.
    ///
    /// Default [`PredictOptions`] uses [`crate::VarianceKind::Observation`]: `σn²`
    /// is added on the diagonal in the transformed space. The diagonal
    /// matches [`Self::predict`] for the same query. This path allocates
    /// an `m × m` matrix; the default [`Self::predict`] stays diagonal-only.
    ///
    /// # Errors
    ///
    /// Same as [`Self::predict`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Gpr, GaussianLikelihood};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    /// let likelihood = GaussianLikelihood::new(0.1)?;
    /// let gpr = Gpr::new(kernel, likelihood);
    /// let fitted = gpr.fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0]).map_err(|(_, e)| e)?;
    /// let cov = fitted.predict_covariance(&[0.25, 0.75], 2, 1)?;
    /// assert_eq!(cov.mean.len(), 2);
    /// assert_eq!(cov.covariance.len(), 4);
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
    /// Posterior covariance is `K** − VᵀV` with `V = L⁻¹ K_*`. Latent
    /// diagonals are clipped at 0. Observation adds `σn²` on the diagonal
    /// in the transformed space, then the target transform scales the
    /// whole matrix.
    ///
    /// # Errors
    ///
    /// Same as [`Self::predict`].
    ///
    /// See the example on [`FittedGpr`].
    pub fn predict_covariance_with(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        options: PredictOptions,
    ) -> Result<PredictiveCovariance<P::Refine>, GprError> {
        self.core.write_covariance(
            self.factor(),
            &self.core.alpha,
            Query::points(xs, n_rows, n_cols),
            options,
        )
    }

    /// Draws posterior samples at `xs` from [`Self::predict_covariance`].
    ///
    /// Each column of the returned column-major `m × n_draws` matrix is
    /// `μ + L z` with `z ∼ N(0, I)` and `L` the Cholesky factor of the
    /// posterior covariance. `seed` is gprx's seeded generator (Xoshiro256++, the same on every platform)
    /// start state. Zero draws returns an empty vector after the covariance
    /// is formed.
    ///
    /// # Errors
    ///
    /// Same as [`Self::predict`], plus [`GprError::CholeskyFailed`] with
    /// [`CholeskyStage::Predict`](crate::CholeskyStage::Predict) if the posterior covariance cannot be
    /// factored after [`crate::JitterPolicy`] retries.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Gpr, GaussianLikelihood};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    /// let likelihood = GaussianLikelihood::new(0.1)?;
    /// let gpr = Gpr::new(kernel, likelihood);
    /// let fitted = gpr.fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0]).map_err(|(_, e)| e)?;
    /// let draws = fitted.sample(&[0.25, 0.75], 2, 1, 4, 1)?;
    /// assert_eq!(draws.len(), 8);
    /// # Ok(())
    /// # }
    /// ```
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
    /// See the example on [`FittedGpr`].
    pub fn sample_with(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        options: PredictOptions,
        n_draws: usize,
        seed: u64,
    ) -> Result<Vec<P::Refine>, GprError> {
        self.core.sample_with(
            self.factor(),
            &self.core.alpha,
            Query::points(xs, n_rows, n_cols),
            options,
            n_draws,
            seed,
        )
    }
}

impl<O, P: GpScalar, K: PointKernel> FittedGpr<O, P, K> {
    /// Returns the feature dimension from the last successful fit.
    ///
    /// See the example on [`FittedGpr`].
    pub fn d(&self) -> usize {
        self.core.d
    }

    /// Returns the original training features in column-major order.
    ///
    /// Same packing as [`Gpr::fit`] / [`Gpr<Fixed>::factor`]: `n` points by
    /// `d` features. Values are on the scale passed to fit, before the input
    /// transform.
    ///
    /// See the example on [`FittedGpr`].
    pub fn x(&self) -> &[f64] {
        &self.core.x_obs
    }
}

impl<O, P: GpScalar, C: PointUse> FittedGpr<O, P, DistanceKernel<C>> {
    /// Returns a copy of the kernel whose hyperparameters this model owns.
    ///
    /// See the example on [`DistanceKernel`].
    pub fn to_kernel(&self) -> DistanceKernel<C> {
        <DistanceKernel<C> as ModelKernelParts>::from_spec(self.core.kernel.clone())
    }

    /// Returns the slots of the kernel, in the order of
    /// [`DistanceKernel::slots`]. A model loaded from disk has slots of its
    /// own; bind its supplies to these.
    ///
    /// See the example on [`DistanceKernel`].
    pub fn slots(&self) -> Vec<DistanceSlot> {
        spec_slots(&self.core.kernel)
    }
}

impl<O, P, K> FittedGpr<O, P, K>
where
    P: GpScalar,
    K: ModelKernel,
    O: for<'a> Optimizer<GprObjective<'a, P, K>>,
{
    /// Re-runs the stored optimizer on the stored training data from the current `θ`.
    ///
    /// This is the same `O` that [`Gpr::with_optimizer`] installed. Transforms
    /// are not re-fit. `n` and `d` stay the same.
    ///
    /// # Errors
    ///
    /// Same as [`Gpr::fit`].
    ///
    /// See the example on [`FittedGpr`].
    pub fn refit(&mut self) -> Result<(), GprError> {
        let mut view = ExactFit {
            core: &mut self.core,
            store: &mut self.store,
        };
        view.optimize(&self.optimizer)
    }
}

impl<P: GpScalar, K: ModelKernel> FittedGpr<Fixed, P, K> {
    /// Rebuilds `L` and `α` at the current `θ` without a search.
    ///
    /// Transforms are not re-fit. `n` and `d` stay the same.
    ///
    /// # Errors
    ///
    /// Same as [`Gpr<Fixed>::factor`].
    ///
    /// See the example on [`FittedGpr`].
    pub fn refit(&mut self) -> Result<(), GprError> {
        self.fit_view().refactor()
    }

    pub(crate) fn into_online_preserving_factor(self) -> Result<OnlineGpr<Fixed, P, K>, GprError> {
        let n = self.core.n;
        let mut workspace = LdltStore::<P::Storage>::from_active(n)?;
        workspace.copy_ld_from(self.chol_l(), n)?;
        workspace.factor_jitter = self.store.buffers.core().factor_jitter;
        LdltStore::set_f64_prefix(&mut workspace.y, &self.core.y_train);
        LdltStore::set_vector_prefix(&mut workspace.alpha, &self.core.factor_alpha);
        Ok(OnlineGpr::from_core(self.core, self.optimizer, workspace))
    }
}

impl<P: GpScalar, K: ModelKernel> FittedGpr<Fixed, P, K> {
    pub(crate) fn from_persisted(mut parts: PersistedModel<P, K>) -> Result<Self, GprError> {
        let n = parts.y_obs.len();
        if n == 0 {
            return Err(GprError::EmptyInput);
        }
        if parts.x_obs.len() % n != 0 {
            return Err(persist::persist_err(
                PersistErrorKind::Tensor,
                "persisted x length is not n * d",
            ));
        }
        let d = parts.x_obs.len() / n;
        if parts.alpha.len() != n {
            return Err(persist::persist_err(
                PersistErrorKind::Tensor,
                format!("alpha has {} values, expected n = {n}", parts.alpha.len()),
            ));
        }
        let mut x_buf = parts.x_obs.clone();
        if d > 0 {
            parts.x_transform.apply(&mut x_buf, n, d)?;
        }
        let mut y_buf = parts.y_obs.clone();
        parts.y_transform.transform(&mut y_buf)?;
        let sources = bind_training::<P::Storage, P::Sources, _>(
            &parts.kernel,
            std::mem::take(&mut parts.sources),
            n,
        )?;
        let compiled = parts.kernel.compile_as::<P::Storage>();
        let mut workspace = fit_buffers::<P, _>(n, parts.policies, &compiled)?;
        workspace.core_mut().factor_jitter = parts.factor_jitter;
        if let Some(l) = parts.owned_l.take() {
            let mut dest = workspace.core_mut().k_matrix.as_mut();
            for col in 0..n {
                for row in 0..n {
                    dest[(row, col)] = l[(row, col)];
                }
            }
        }
        let factor_alpha = storage_alpha_from_saved::<P>(
            workspace.core().k_matrix.as_ref(),
            &y_buf,
            &parts.alpha,
        )?;
        Ok(Self {
            core: GprCore {
                slots: spec_slots(&parts.kernel),
                kernel: parts.kernel,
                compiled,
                likelihood: parts.likelihood,
                x_unfitted: parts.x_unfitted,
                y_unfitted: parts.y_unfitted,
                x_transform: parts.x_transform,
                y_transform: parts.y_transform,
                policies: parts.policies,
                query: QueryWorkspace::new(),
                x_obs: parts.x_obs,
                y_obs: parts.y_obs,
                x: pack_points(&x_buf, n, d),
                y_train: y_buf,
                factor_alpha,
                alpha: parts.alpha,
                x_cast: P::Storage::empty_cols(),
                y_cast: P::Storage::empty_rows(),
                sources,
                n,
                d,
            },
            optimizer: Fixed,
            store: LltStore::with_mapped(workspace, parts.mapped),
            _kernel: PhantomData,
        })
    }
}

fn storage_alpha_from_saved<P: GpScalar>(
    l: MatRef<'_, P::Storage>,
    y: &[f64],
    saved: &[P::Refine],
) -> Result<Vec<P::Storage>, GprError> {
    let mixed = P::REFINES_IN_F64;
    if !mixed {
        return Ok(saved
            .iter()
            .map(|weight| P::Storage::from_f64(weight.to_f64()))
            .collect());
    }
    let n = y.len();
    let mut rhs = Mat::<P::Storage>::zeros(n, 1);
    for (i, &yi) in y.iter().enumerate() {
        rhs[(i, 0)] = P::Storage::from_f64(yi);
    }
    let par = faer_par_dims(n, 1);
    let req = llt::solve::solve_in_place_scratch::<P::Storage>(n, 1, par);
    let mut scratch = MemBuffer::new(req);
    solve_llt_in_place(l, rhs.as_mut(), &mut scratch);
    Ok((0..n).map(|i| rhs[(i, 0)]).collect())
}
