//! [`Gpr`] trainer and [`FittedGpr`] factorization.

use std::fmt;
use std::marker::PhantomData;

use dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt;
use faer::{Mat, MatMut, MatRef};

use crate::error::{CholeskyStage, GprError};
use crate::kernel::{
    CompiledKernel, CoordMode, KernelSpec, MixedKernelViews, Triangle, fill_squared_euclidean,
    fill_squared_euclidean_cross,
};
use crate::likelihood::GaussianLikelihood;
use crate::objective::GprObjective;
use crate::optimizer::{
    AcceptsRecompute, Fixed, FullRecompute, Lbfgs, OptResult, Optimizer, RecomputeStrategy,
};
use crate::param::Interval;
use crate::persist::{self, MappedTensors, PersistedModel};
use crate::precision::DoublePrecision;
use crate::transform::{
    IdentityInput, IdentityTarget, TargetTransform, Transform, UnfittedTarget, UnfittedTransform,
};
use crate::workspace::{QueryWorkspace, Workspace, empty_thread_scratch, faer_par};

use super::factor::{
    FactorPolicy, cholesky_lower_with_policy, factor_train_with_policy, fill_identity,
    form_w_lower, frobenius_lower, inv_diag_from_chol_l, neg_mll_from_factor, pack_points,
    pack_points_into, require_param_len, validate_query, validate_training, write_kernel_grad,
    write_params,
};
use super::{
    DistanceCachePolicy, DistanceCacheSlot, JitterPolicy, NoDistanceCache, PredictOptions,
    Prediction, PredictiveCovariance, VarianceKind,
};

/// Unfitted Exact GPR trainer: kernel, likelihood, transforms, optimizer, and
/// recompute strategy.
///
/// [`Self::fit`] consumes [`Gpr<O>`] where `O: `[`Optimizer`] and searches
/// hyperparameters. [`Gpr<Fixed>::factor`] factors at the current `θ` with no
/// search. Success returns [`FittedGpr`]. Failure returns the trainer with
/// [`GprError`] so the caller can change `θ` or data and try again.
/// Input and target transforms default to identity. Trainers from
/// [`Gpr::new`] store [`DistanceCachePolicy`] (default [`DistanceCachePolicy::Always`]).
/// Standalone Linear, Constant, and White kernels use [`Gpr::from_points`],
/// which has no distance-cache slot. The default type is
/// [`Gpr<Lbfgs, FullRecompute>`]. [`Clone`] copies kernel, likelihood,
/// transforms, optimizer, and policies.
///
/// Isotropic distance fills and lower-triangle kernel writes use Rayon.
/// There is no parallel on/off flag. Thread count is the process-wide
/// pool (`RAYON_NUM_THREADS`, or `ThreadPoolBuilder::build_global` before
/// the first fit); one worker is sequential. See the [crate-level
/// parallelism notes](crate).
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
/// // Column-major `X` with n = 2 points and d = 1 feature.
/// let fitted = gpr.fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0]).map_err(|(_, e)| e)?;
/// let pred = fitted.predict(&[0.5], 1, 1)?;
/// assert_eq!(pred.mean.len(), 1);
/// let _nlml = fitted.neg_log_marginal_likelihood()?;
/// # Ok(())
/// # }
/// ```
pub struct Gpr<O = Lbfgs, S = FullRecompute, C = DistanceCachePolicy> {
    kernel: KernelSpec,
    likelihood: GaussianLikelihood,
    x_transform: Box<dyn UnfittedTransform>,
    y_transform: Box<dyn UnfittedTarget>,
    optimizer: O,
    distance_cache: C,
    jitter_policy: JitterPolicy,
    _recompute: PhantomData<S>,
}

impl<O, S, C> fmt::Debug for Gpr<O, S, C>
where
    O: fmt::Debug,
    C: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Gpr")
            .field("kernel", &self.kernel)
            .field("likelihood", &self.likelihood)
            .field("optimizer", &self.optimizer)
            .field("distance_cache", &self.distance_cache)
            .field("jitter_policy", &self.jitter_policy)
            .finish_non_exhaustive()
    }
}

impl<O: Clone, S, C: Copy> Clone for Gpr<O, S, C> {
    fn clone(&self) -> Self {
        Self {
            kernel: self.kernel.clone(),
            likelihood: self.likelihood,
            x_transform: self.x_transform.clone_box(),
            y_transform: self.y_transform.clone_box(),
            optimizer: self.optimizer.clone(),
            distance_cache: self.distance_cache,
            jitter_policy: self.jitter_policy,
            _recompute: PhantomData,
        }
    }
}

/// Fitted Exact GPR: `L`, `α`, training `X` / `y`, kernel, and transforms.
///
/// [`Self::neg_log_marginal_likelihood`] is
/// `½ yᵀ α + ½ log|A| + (n/2) log(2π)` with `log|A| = 2 Σ log(L_ii)`.
/// [`Self::value_and_gradient_into`] rebuilds `L`, `α`, and `W` once and
/// writes `∂L/∂θ = -½ ⟨W, ∂A/∂θ⟩`. [`Self::predict`] returns the mean and
/// a diagonal variance; [`Self::predict_into`] writes into a reused
/// [`Prediction`] and crate-private query buffers (not the fit workspace).
/// [`Self::predict_covariance`] returns the query–query matrix as
/// [`PredictiveCovariance`]. [`Self::sample`] draws from that posterior.
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
pub struct FittedGpr<O = Lbfgs, S = FullRecompute, C = DistanceCachePolicy> {
    kernel: KernelSpec,
    compiled: CompiledKernel,
    likelihood: GaussianLikelihood,
    x_unfitted: Box<dyn UnfittedTransform>,
    y_unfitted: Box<dyn UnfittedTarget>,
    x_transform: Box<dyn Transform>,
    y_transform: Box<dyn TargetTransform>,
    optimizer: O,
    distance_cache: C,
    jitter_policy: JitterPolicy,
    workspace: Workspace<DoublePrecision>,
    query: QueryWorkspace<DoublePrecision>,
    x_obs: Vec<f64>,
    y_obs: Vec<f64>,
    x: Mat<f64>,
    y_train: Vec<f64>,
    alpha: Vec<f64>,
    n: usize,
    d: usize,
    mapped_factor: Option<MappedTensors>,
    _recompute: PhantomData<S>,
}

impl<O: Clone, S, C: Copy> Clone for FittedGpr<O, S, C> {
    fn clone(&self) -> Self {
        let mut workspace = self.workspace.clone();
        if let Some(mapped) = &self.mapped_factor {
            persist::copy_l_into(workspace.k_matrix.as_mut(), mapped.l_view());
        }
        Self {
            kernel: self.kernel.clone(),
            compiled: self.compiled.clone(),
            likelihood: self.likelihood,
            x_unfitted: self.x_unfitted.clone_box(),
            y_unfitted: self.y_unfitted.clone_box(),
            x_transform: self.x_transform.clone_box(),
            y_transform: self.y_transform.clone_box(),
            optimizer: self.optimizer.clone(),
            distance_cache: self.distance_cache,
            jitter_policy: self.jitter_policy,
            workspace,
            query: self.query.clone(),
            x_obs: self.x_obs.clone(),
            y_obs: self.y_obs.clone(),
            x: self.x.clone(),
            y_train: self.y_train.clone(),
            alpha: self.alpha.clone(),
            n: self.n,
            d: self.d,
            mapped_factor: None,
            _recompute: PhantomData,
        }
    }
}

impl<O, S, C> fmt::Debug for FittedGpr<O, S, C>
where
    O: fmt::Debug,
    C: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FittedGpr")
            .field("n", &self.n)
            .field("d", &self.d)
            .field("kernel", &self.kernel)
            .field("likelihood", &self.likelihood)
            .field("distance_cache", &self.distance_cache)
            .field("jitter_policy", &self.jitter_policy)
            .finish_non_exhaustive()
    }
}

impl Gpr {
    /// Builds an unfitted trainer that owns the kernel and observation noise.
    ///
    /// Input and target maps default to identity. The optimizer is [`Lbfgs`].
    /// Call [`Self::with_input_transform`] / [`Self::with_target_transform`]
    /// before [`Self::fit`] to standardize. Those methods take an unfitted
    /// map; `fit` / `factor` produce the fitted map stored on [`FittedGpr`].
    /// Call [`Self::with_optimizer`] to
    /// switch to [`Fixed`] or another [`Optimizer`]. Distance kernels use this
    /// constructor; the cache slot is [`DistanceCachePolicy`]. Standalone
    /// Linear, Constant, and White kernels use [`Self::from_points`].
    pub fn new(kernel: KernelSpec, likelihood: GaussianLikelihood) -> Self {
        Self {
            kernel,
            likelihood,
            x_transform: Box::new(IdentityInput),
            y_transform: Box::new(IdentityTarget),
            optimizer: Lbfgs::new(),
            distance_cache: DistanceCachePolicy::Always,
            jitter_policy: JitterPolicy::default(),
            _recompute: PhantomData,
        }
    }

    /// Builds a trainer for a kernel that evaluates from coordinates, not
    /// pairwise distances.
    ///
    /// Use this for a standalone Linear, Constant, or White kernel. The
    /// trainer has no [`DistanceCachePolicy`]. Compositions that still fill
    /// distances (`RBF + White`, `Constant * RBF`) use [`Self::new`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, LinearKernel};
    /// use gprx::{GaussianLikelihood, Gpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let kernel = KernelSpec::from(LinearKernel::new(1.0)?);
    /// let likelihood = GaussianLikelihood::new(0.1)?;
    /// let fitted = Gpr::from_points(kernel, likelihood)
    ///     .fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
    ///     .map_err(|(_, e)| e)?;
    /// let pred = fitted.predict(&[0.5], 1, 1)?;
    /// assert_eq!(pred.mean.len(), 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_points(
        kernel: KernelSpec,
        likelihood: GaussianLikelihood,
    ) -> Gpr<Lbfgs, FullRecompute, NoDistanceCache> {
        Gpr {
            kernel,
            likelihood,
            x_transform: Box::new(IdentityInput),
            y_transform: Box::new(IdentityTarget),
            optimizer: Lbfgs::new(),
            distance_cache: NoDistanceCache,
            jitter_policy: JitterPolicy::default(),
            _recompute: PhantomData,
        }
    }
}

impl<O, S, C> Gpr<O, S, C> {
    /// Replaces the input (`X`) transform. Intended to be called before fit.
    ///
    /// A single map, a [`crate::transform::Pipeline`], or
    /// [`crate::transform::ColumnwiseInput`]. One-step maps still use this
    /// method.
    pub fn with_input_transform(mut self, transform: impl UnfittedTransform + 'static) -> Self {
        self.x_transform = Box::new(transform);
        self
    }

    /// Replaces the target (`y`) transform. Intended to be called before fit.
    ///
    /// A single map or a [`crate::transform::TargetPipeline`]. One-step maps
    /// still use this method.
    pub fn with_target_transform(mut self, transform: impl UnfittedTarget + 'static) -> Self {
        self.y_transform = Box::new(transform);
        self
    }

    pub(crate) fn with_boxed_input_transform(
        mut self,
        transform: Box<dyn UnfittedTransform>,
    ) -> Self {
        self.x_transform = transform;
        self
    }

    pub(crate) fn with_boxed_target_transform(
        mut self,
        transform: Box<dyn UnfittedTarget>,
    ) -> Self {
        self.y_transform = transform;
        self
    }

    /// Sets the Cholesky jitter policy. Does not change observation noise.
    ///
    /// Intended to be called before [`Gpr::fit`] / [`Gpr<Fixed>::factor`].
    /// The default is [`JitterPolicy::fixed`]`(0.0)`. See [`JitterPolicy`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{GaussianLikelihood, Gpr, JitterPolicy};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let gpr = Gpr::new(
    ///     KernelSpec::from(RbfKernel::new(1.0)?),
    ///     GaussianLikelihood::new(0.1)?,
    /// )
    /// .with_jitter_policy(JitterPolicy::fixed(1e-8)?);
    /// let _fitted = gpr
    ///     .with_optimizer(gprx::Fixed)
    ///     .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
    ///     .map_err(|(_, e)| e)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_jitter_policy(mut self, policy: JitterPolicy) -> Self {
        self.jitter_policy = policy;
        self
    }

    /// Replaces the optimizer, changing the type parameter `O`.
    ///
    /// The recompute strategy becomes [`FullRecompute`]. Call
    /// [`Gpr::with_recompute_strategy`] afterwards when the new optimizer
    /// implements [`crate::UsesChangeIndices`].
    ///
    /// [`Fixed`] is not an [`Optimizer`]; use [`Gpr<Fixed>::factor`] after
    /// this switch. argmin solvers are [`crate::Lbfgs`], [`crate::NonlinearCg`],
    /// and [`crate::NelderMead`]. A user type that implements [`Optimizer`]
    /// uses this same method; there is no second solver slot.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{GaussianLikelihood, Gpr, NonlinearCg};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let gpr = Gpr::new(
    ///     KernelSpec::from(RbfKernel::new(1.0)?),
    ///     GaussianLikelihood::new(0.1)?,
    /// )
    /// .with_optimizer(NonlinearCg::new());
    /// let _fitted = gpr.fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0]).map_err(|(_, e)| e)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_optimizer<O2>(self, optimizer: O2) -> Gpr<O2, FullRecompute, C> {
        Gpr {
            kernel: self.kernel,
            likelihood: self.likelihood,
            x_transform: self.x_transform,
            y_transform: self.y_transform,
            optimizer,
            distance_cache: self.distance_cache,
            jitter_policy: self.jitter_policy,
            _recompute: PhantomData,
        }
    }

    /// Returns the kernel whose hyperparameters this trainer owns.
    pub fn kernel(&self) -> &KernelSpec {
        &self.kernel
    }

    /// Returns the observation-noise model.
    pub fn likelihood(&self) -> &GaussianLikelihood {
        &self.likelihood
    }

    /// Returns the concatenated kernel and likelihood parameter count.
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
}

impl<O, S> Gpr<O, S, DistanceCachePolicy> {
    /// Sets whether training distances are cached across kernel builds.
    ///
    /// Intended to be called before [`Gpr::fit`] / [`Gpr<Fixed>::factor`].
    /// The default is [`DistanceCachePolicy::Always`]. This method exists
    /// only on trainers from [`Gpr::new`]. See [`DistanceCachePolicy`].
    pub fn with_distance_cache_policy(mut self, policy: DistanceCachePolicy) -> Self {
        self.distance_cache = policy;
        self
    }
}

#[allow(private_bounds)] // `GprObjective` is crate-private; `fit` still needs `O: Optimizer` for it.
impl<O, S, C> Gpr<O, S, C>
where
    S: RecomputeStrategy,
    C: DistanceCacheSlot,
    O: Clone + for<'a> Optimizer<GprObjective<'a, O, S, C>>,
{
    /// Replaces the recompute-strategy marker.
    ///
    /// [`FullRecompute`] is valid for every optimizer. [`crate::IncrementalRecompute`]
    /// requires `O: `[`crate::UsesChangeIndices`]. L-BFGS does not implement
    /// that marker. The incremental evaluation body is P2B-18.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{FullRecompute, GaussianLikelihood, Gpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let gpr = Gpr::new(
    ///     KernelSpec::from(RbfKernel::new(1.0)?),
    ///     GaussianLikelihood::new(0.1)?,
    /// )
    /// .with_recompute_strategy(FullRecompute);
    /// let _fitted = gpr.fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0]).map_err(|(_, e)| e)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_recompute_strategy<S2: AcceptsRecompute<O>>(self, _: S2) -> Gpr<O, S2, C> {
        Gpr {
            kernel: self.kernel,
            likelihood: self.likelihood,
            x_transform: self.x_transform,
            y_transform: self.y_transform,
            optimizer: self.optimizer,
            distance_cache: self.distance_cache,
            jitter_policy: self.jitter_policy,
            _recompute: PhantomData,
        }
    }

    /// Factors `A = K + σn² I`, solves `A α = y`, and updates `θ` with `O`.
    ///
    /// `x` is column-major with `n_rows` points and `n_cols` features. After
    /// success, `L` remains in the workspace and `α` is stored on
    /// [`FittedGpr`]. A failed factorization or optimizer step returns this
    /// trainer and does not produce a [`FittedGpr`]. Kernel and likelihood
    /// `θ` are restored to the values from the start of the call when
    /// optimization fails.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::EmptyInput`] if `n_rows` or `n_cols` is zero,
    /// [`GprError::InvalidHyperparameter`] if `x` or `y` has the wrong length,
    /// [`GprError::NonFiniteInput`] if a value is `NaN` or `Inf`,
    /// [`GprError::CholeskyFailed`] if `A` cannot be factored, or
    /// [`GprError::OptimizationNotConverged`] if the optimizer does not
    /// produce a best parameter vector.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Fixed, Gpr, GaussianLikelihood};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    /// let likelihood = GaussianLikelihood::new(0.1)?;
    /// let gpr = Gpr::new(kernel, likelihood);
    /// let fitted = gpr.fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0]).map_err(|(_, e)| e)?;
    /// let _fitted = fitted
    ///     .into_trainer()
    ///     .with_optimizer(Fixed)
    ///     .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
    ///     .map_err(|(_, e)| e)?;
    /// # Ok(())
    /// # }
    /// ```
    #[allow(clippy::result_large_err)] // failure returns the trainer so the caller can retry
    pub fn fit(
        self,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
        y: &[f64],
    ) -> Result<FittedGpr<O, S, C>, (Self, GprError)> {
        let mut model = FittedGpr::prepare(self, x, n_rows, n_cols, y)?;
        match model.optimize_hyperparameters() {
            Ok(()) => Ok(model),
            Err(err) => Err((model.into_trainer(), err)),
        }
    }
}

#[allow(private_bounds)] // `DistanceCacheSlot` is crate-private; `factor` still needs it.
impl<C: DistanceCacheSlot> Gpr<Fixed, FullRecompute, C> {
    /// Factors at the current kernel and likelihood `θ` without a search.
    ///
    /// Same data contract as [`Gpr::fit`]. There are no optimizer knobs.
    ///
    /// # Errors
    ///
    /// Same as [`Gpr::fit`], except [`GprError::OptimizationNotConverged`]
    /// does not apply.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Fixed, Gpr, GaussianLikelihood};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    /// let likelihood = GaussianLikelihood::new(0.1)?;
    /// let fitted = Gpr::new(kernel, likelihood)
    ///     .with_optimizer(Fixed)
    ///     .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
    ///     .map_err(|(_, e)| e)?;
    /// let pred = fitted.predict(&[0.5], 1, 1)?;
    /// assert_eq!(pred.mean.len(), 1);
    /// # Ok(())
    /// # }
    /// ```
    #[allow(clippy::result_large_err)] // failure returns the trainer so the caller can retry
    pub fn factor(
        self,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
        y: &[f64],
    ) -> Result<FittedGpr<Fixed, FullRecompute, C>, (Self, GprError)> {
        let mut model = FittedGpr::prepare(self, x, n_rows, n_cols, y)?;
        match model.factorize_current() {
            Ok(()) => Ok(model),
            Err(err) => Err((model.into_trainer(), err)),
        }
    }
}

/// Drops the trainer and keeps the error so `?` works in `Result<_, GprError>`.
impl<O, S, C> From<(Gpr<O, S, C>, GprError)> for GprError {
    fn from((_, err): (Gpr<O, S, C>, GprError)) -> Self {
        err
    }
}

#[allow(private_bounds)] // `DistanceCacheSlot` is crate-private; factorization reads it.
impl<O, S, C: DistanceCacheSlot> FittedGpr<O, S, C> {
    #[allow(clippy::result_large_err)] // failure returns the trainer so the caller can retry
    fn prepare(
        gpr: Gpr<O, S, C>,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
        y: &[f64],
    ) -> Result<Self, (Gpr<O, S, C>, GprError)> {
        if let Err(err) = validate_training(x, n_rows, n_cols, y) {
            return Err((gpr, err));
        }
        let mut x_buf = x.to_vec();
        let x_fitted = match gpr.x_transform.clone_box().fit(&x_buf, n_rows, n_cols) {
            Ok(t) => t,
            Err(err) => return Err((gpr, err)),
        };
        if let Err(err) = x_fitted.apply(&mut x_buf, n_rows, n_cols) {
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
        let mut workspace = match Workspace::new(n_rows) {
            Ok(ws) => ws,
            Err(err) => return Err((gpr, err)),
        };
        let compiled = gpr.kernel.compile();
        if gpr.distance_cache.policy() == DistanceCachePolicy::Always
            && compiled.needs_ard_sq_diff()
        {
            if let Err(err) = workspace.ensure_ard_sq_diff(n_rows, n_cols) {
                return Err((gpr, err));
            }
        } else {
            workspace.clear_ard_sq_diff();
        }
        Ok(Self {
            kernel: gpr.kernel,
            compiled,
            likelihood: gpr.likelihood,
            x_unfitted: gpr.x_transform,
            y_unfitted: gpr.y_transform,
            x_transform: x_fitted,
            y_transform: y_fitted,
            optimizer: gpr.optimizer,
            distance_cache: gpr.distance_cache,
            jitter_policy: gpr.jitter_policy,
            workspace,
            query: QueryWorkspace::new(),
            x_obs: x.to_vec(),
            y_obs: y.to_vec(),
            x: pack_points(&x_buf, n_rows, n_cols),
            y_train: y_buf,
            alpha: vec![0.0; n_rows],
            n: n_rows,
            d: n_cols,
            mapped_factor: None,
            _recompute: PhantomData,
        })
    }

    /// Drops `L` / `α` / training data and returns a trainer with the current
    /// kernel, likelihood, transforms, optimizer, distance-cache slot, and
    /// jitter policy.
    pub fn into_trainer(self) -> Gpr<O, S, C> {
        Gpr {
            kernel: self.kernel,
            likelihood: self.likelihood,
            x_transform: self.x_unfitted,
            y_transform: self.y_unfitted,
            optimizer: self.optimizer,
            distance_cache: self.distance_cache,
            jitter_policy: self.jitter_policy,
            _recompute: PhantomData,
        }
    }

    /// Returns the number of training points.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Returns the feature dimension from the last successful fit.
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

    /// Returns `α = A⁻¹ y` from the last successful fit.
    pub fn alpha(&self) -> &[f64] {
        &self.alpha
    }

    /// Returns the original training features in column-major order.
    ///
    /// Same packing as [`Gpr::fit`] / [`Gpr<Fixed>::factor`]: `n` points by
    /// `d` features. Values are on the scale passed to fit, before the input
    /// transform.
    pub fn x(&self) -> &[f64] {
        &self.x_obs
    }

    /// Returns the original training targets.
    ///
    /// Values are on the scale passed to fit, before the target transform.
    pub fn y(&self) -> &[f64] {
        &self.y_obs
    }

    /// Writes this fitted model to `dir/config.json` and `dir/model.safetensors`.
    ///
    /// Omits `L` and `α`. [`crate::persist::LoadedGpr::load`] rebuilds them
    /// by factorizing.
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
    /// `L` is stored as a column-major `n×n` `f64` tensor; the lower triangle
    /// is canonical. [`crate::persist::LoadedGpr::load`] keeps the safetensors
    /// file mapped for `L`.
    ///
    /// # Errors
    ///
    /// Same as [`Self::save`].
    pub fn save_with_factor(&self, dir: impl AsRef<std::path::Path>) -> Result<(), GprError> {
        persist::save_fitted(self, dir.as_ref(), true)
    }

    /// Replaces the optimizer used by a later [`Self::refit`].
    ///
    /// Does not write a solver into a persist directory. A model loaded as
    /// [`crate::persist::LoadedGpr`] is [`Fixed`]; call this before `refit`
    /// to search again.
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
    /// let LoadedGpr::Distance(model) = LoadedGpr::load(&dir, &PersistRegistry::new())? else {
    ///     return Ok(());
    /// };
    /// let mut model = model.with_optimizer(Lbfgs::new());
    /// model.refit()?;
    /// let _ = std::fs::remove_dir_all(&dir);
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_optimizer<O2>(self, optimizer: O2) -> FittedGpr<O2, S, C> {
        FittedGpr {
            kernel: self.kernel,
            compiled: self.compiled,
            likelihood: self.likelihood,
            x_unfitted: self.x_unfitted,
            y_unfitted: self.y_unfitted,
            x_transform: self.x_transform,
            y_transform: self.y_transform,
            optimizer,
            distance_cache: self.distance_cache,
            jitter_policy: self.jitter_policy,
            workspace: self.workspace,
            query: self.query,
            x_obs: self.x_obs,
            y_obs: self.y_obs,
            x: self.x,
            y_train: self.y_train,
            alpha: self.alpha,
            n: self.n,
            d: self.d,
            mapped_factor: self.mapped_factor,
            _recompute: PhantomData,
        }
    }

    pub(crate) fn jitter_policy(&self) -> JitterPolicy {
        self.jitter_policy
    }

    pub(crate) fn distance_cache_slot(&self) -> C {
        self.distance_cache
    }

    pub(crate) fn x_unfitted(&self) -> &dyn UnfittedTransform {
        self.x_unfitted.as_ref()
    }

    pub(crate) fn y_unfitted(&self) -> &dyn UnfittedTarget {
        self.y_unfitted.as_ref()
    }

    pub(crate) fn x_transform(&self) -> &dyn Transform {
        self.x_transform.as_ref()
    }

    pub(crate) fn y_transform(&self) -> &dyn TargetTransform {
        self.y_transform.as_ref()
    }

    pub(crate) fn chol_l(&self) -> MatRef<'_, f64> {
        match &self.mapped_factor {
            Some(mapped) => mapped.l_view(),
            None => self.workspace.k_matrix.as_ref(),
        }
    }

    /// Returns the negative log marginal likelihood of the last successful fit.
    ///
    /// Evaluates `½ yᵀ A⁻¹ y + ½ log|A| + (n/2) log(2π)` from the stored
    /// `α` and the Cholesky factor `L` in the workspace, using
    /// `log|A| = 2 Σ log(L_ii)`. `y` is the target after the target
    /// transform.
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
        Ok(neg_mll_from_factor(
            self.chol_l(),
            &self.y_train,
            &self.alpha,
            self.n,
        ))
    }

    /// Returns the concatenated kernel and likelihood parameter count.
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
        let n_kernel = self.kernel.num_params();
        require_param_len(out.len(), self.num_params())?;
        self.kernel.get_params(&mut out[..n_kernel])?;
        self.likelihood.get_params(&mut out[n_kernel..])
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
    /// Returns [`GprError::InvalidHyperparameter`] if `params` is the wrong
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
        let n_kernel = self.kernel.num_params();
        require_param_len(params.len(), self.num_params())?;
        let (kernel, compiled, likelihood) = self.prepared_params(params, n_kernel)?;
        let workspace = self.workspace.clone();
        let alpha = self.alpha.clone();
        if let Err(err) = factor_train_with_policy(
            &compiled,
            self.x.as_ref(),
            &mut self.workspace,
            &self.y_train,
            likelihood.noise_variance(),
            FactorPolicy {
                cache: self.distance_cache.policy(),
                jitter: self.jitter_policy,
                stage: CholeskyStage::Fit,
            },
        ) {
            self.workspace = workspace;
            self.alpha = alpha;
            return Err(err);
        }
        self.copy_alpha_from_rhs();
        self.kernel = kernel;
        self.compiled = compiled;
        self.likelihood = likelihood;
        self.mapped_factor = None;
        Ok(())
    }

    pub(crate) fn objective(&mut self) -> GprObjective<'_, O, S, C> {
        GprObjective::new(self)
    }

    pub(crate) fn fill_intervals(&self, out: &mut [Interval]) -> Result<(), GprError> {
        let n = self.num_params();
        if out.len() != n {
            return Err(GprError::InvalidHyperparameter {
                reason: format!("expected {n} intervals, got {}", out.len()),
            });
        }
        let n_kernel = self.kernel.num_params();
        let mut offset = 0;
        self.kernel
            .write_intervals(&mut out[..n_kernel], &mut offset)?;
        out[n_kernel] = self.likelihood.bounds();
        Ok(())
    }

    /// Sets kernel and likelihood `θ`, rebuilds `L` / `α` / `W`, and writes `∂L/∂θ`.
    ///
    /// `params` and `out` are kernel parameters followed by the likelihood
    /// parameter. One Cholesky produces `L` and `α`; `W = ααᵀ - A⁻¹` is
    /// formed in the workspace without overwriting `L`. Kernel `∂A/∂θ` goes
    /// through `exp_buf`. Product trees also use `kernel_scratch`. The
    /// returned value is the same as
    /// [`Self::neg_log_marginal_likelihood`] after a successful call.
    ///
    /// Training `X` / `y` come from [`Gpr::fit`]. Transforms
    /// are not re-fit.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if a slice length is wrong,
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
        let n_kernel = self.kernel.num_params();
        let n_params = self.num_params();
        require_param_len(params.len(), n_params)?;
        require_param_len(out.len(), n_params)?;
        let (kernel, compiled, likelihood) = self.prepared_params(params, n_kernel)?;
        let n = self.n;
        if let Err(err) = factor_train_with_policy(
            &compiled,
            self.x.as_ref(),
            &mut self.workspace,
            &self.y_train,
            likelihood.noise_variance(),
            FactorPolicy {
                cache: self.distance_cache.policy(),
                jitter: self.jitter_policy,
                stage: CholeskyStage::Fit,
            },
        ) {
            let _ = self.factorize_current();
            return Err(err);
        }
        if self.alpha.len() != n {
            self.alpha.resize(n, 0.0);
        }
        for (i, slot) in self.alpha.iter_mut().enumerate() {
            *slot = self.workspace.rhs[(i, 0)];
        }
        self.kernel = kernel;
        self.likelihood = likelihood;
        self.compiled = compiled;
        self.mapped_factor = None;
        let nlml = neg_mll_from_factor(
            self.workspace.k_matrix.as_ref(),
            &self.y_train,
            &self.alpha,
            n,
        );
        {
            let compiled = &self.compiled;
            fill_identity(self.workspace.w_matrix.as_mut());
            {
                let stack = MemStack::new(&mut self.workspace.faer_scratch);
                llt::solve::solve_in_place(
                    self.workspace.k_matrix.as_ref(),
                    self.workspace.w_matrix.as_mut(),
                    faer_par(n),
                    stack,
                );
            }
            form_w_lower(self.workspace.w_matrix.as_mut(), &self.alpha, n);
            if compiled.needs_product_grad_scratch() {
                self.workspace.ensure_kernel_scratch(n)?;
            }
            let thread_scratch = std::mem::take(&mut self.workspace.thread_scratch);
            let ard_cache = if compiled.needs_ard_sq_diff() && self.workspace.ard_sq_diff_ready {
                Some(self.workspace.ard_sq_diff.as_ref())
            } else {
                None
            };
            let result = (|| {
                for (i, slot) in out.iter_mut().enumerate().take(n_kernel) {
                    write_kernel_grad(
                        compiled,
                        self.workspace.dist_cache.as_ref(),
                        self.x.as_ref(),
                        ard_cache,
                        self.workspace.exp_buf.as_mut(),
                        self.workspace.kernel_scratch.as_mut(),
                        i,
                    )?;
                    let inner = frobenius_lower(
                        self.workspace.w_matrix.as_ref(),
                        self.workspace.exp_buf.as_ref(),
                        n,
                    );
                    *slot = -0.5 * inner;
                }
                Ok::<(), GprError>(())
            })();
            self.workspace.thread_scratch = thread_scratch;
            result?;
            let mut noise_inner = 0.0;
            let d_noise = self.likelihood.noise_variance();
            for i in 0..n {
                noise_inner += self.workspace.w_matrix[(i, i)] * d_noise;
            }
            out[n_kernel] = -0.5 * noise_inner;
        }
        Ok(nlml)
    }

    /// Builds kernel, compiled kernel, and likelihood `θ` without storing them.
    ///
    /// Each `set_params` is atomic on its own type. The caller commits the
    /// triple only after `A` factors, so a later Cholesky failure cannot
    /// leave stored kernel and likelihood `θ` mixed or half-applied.
    fn prepared_params(
        &self,
        params: &[f64],
        n_kernel: usize,
    ) -> Result<(KernelSpec, CompiledKernel, GaussianLikelihood), GprError> {
        let mut likelihood = self.likelihood;
        likelihood.set_params(&params[n_kernel..])?;
        let mut kernel = self.kernel.clone();
        kernel.set_params(&params[..n_kernel])?;
        let mut compiled = self.compiled.clone();
        compiled.set_params(&params[..n_kernel])?;
        Ok((kernel, compiled, likelihood))
    }

    fn optimize_hyperparameters(&mut self) -> Result<(), GprError>
    where
        O: Clone + for<'a> Optimizer<GprObjective<'a, O, S, C>>,
    {
        let mut init = vec![0.0; self.num_params()];
        self.get_params(&mut init)?;
        let kernel_before = self.kernel.clone();
        let likelihood_before = self.likelihood;
        let optimizer = self.optimizer.clone();
        let result = {
            let mut obj = self.objective();
            optimizer.minimize(&mut obj, &init)
        };
        self.commit_or_revert_optimize(kernel_before, likelihood_before, result)
    }

    fn commit_or_revert_optimize(
        &mut self,
        kernel_before: KernelSpec,
        likelihood_before: GaussianLikelihood,
        result: Result<OptResult, GprError>,
    ) -> Result<(), GprError> {
        match result {
            Ok(opt) => {
                if opt.params.len() != self.num_params() || !opt.value.is_finite() {
                    self.revert_theta(kernel_before, likelihood_before);
                    return Err(GprError::OptimizationNotConverged {
                        iterations: opt.iterations as usize,
                    });
                }
                Ok(())
            }
            Err(err) => {
                self.revert_theta(kernel_before, likelihood_before);
                Err(err)
            }
        }
    }

    fn revert_theta(&mut self, kernel: KernelSpec, likelihood: GaussianLikelihood) {
        self.kernel = kernel;
        self.likelihood = likelihood;
        self.compiled = self.kernel.compile();
        let _ = self.factorize_current();
    }

    fn factorize_current(&mut self) -> Result<(), GprError> {
        self.mapped_factor = None;
        factor_train_with_policy(
            &self.compiled,
            self.x.as_ref(),
            &mut self.workspace,
            &self.y_train,
            self.likelihood.noise_variance(),
            FactorPolicy {
                cache: self.distance_cache.policy(),
                jitter: self.jitter_policy,
                stage: CholeskyStage::Fit,
            },
        )?;
        self.copy_alpha_from_rhs();
        Ok(())
    }

    fn copy_alpha_from_rhs(&mut self) {
        let n = self.n;
        if self.alpha.len() != n {
            self.alpha.resize(n, 0.0);
        }
        for (i, slot) in self.alpha.iter_mut().enumerate() {
            *slot = self.workspace.rhs[(i, 0)];
        }
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
    /// [`GprError::InvalidHyperparameter`] / [`GprError::NonFiniteInput`] for a
    /// badly packed or non-finite `xs`.
    pub fn predict(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
    ) -> Result<Prediction, GprError> {
        self.predict_with(xs, n_rows, n_cols, PredictOptions::default())
    }

    /// Writes [`Self::predict`] into `out`, reusing `mean` / `variance`
    /// capacity when the query length matches.
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
        out: &mut Prediction,
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
    pub fn predict_with(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        options: PredictOptions,
    ) -> Result<Prediction, GprError> {
        let mut out = Prediction::default();
        self.write_prediction(xs, n_rows, n_cols, options, &mut out)?;
        Ok(out)
    }

    /// Writes [`Self::predict_with`] into `out`, reusing `mean` / `variance`
    /// capacity when the query length matches.
    ///
    /// # Errors
    ///
    /// Same as [`Self::predict`].
    pub fn predict_with_into(
        &mut self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        options: PredictOptions,
        out: &mut Prediction,
    ) -> Result<(), GprError> {
        if n_cols != self.d {
            return Err(GprError::DimensionMismatch {
                x_dim: n_cols,
                expected_dim: self.d,
            });
        }
        validate_query(xs, n_rows, n_cols)?;
        let n = self.n;
        let m = n_rows;
        self.query.ensure(n, m, n_cols)?;
        self.query.query_xs.copy_from_slice(xs);
        self.x_transform
            .apply(&mut self.query.query_xs, n_rows, n_cols)?;
        pack_points_into(
            &self.query.query_xs,
            n_rows,
            n_cols,
            self.query.query_x.as_mut(),
        );
        let compiled = &self.compiled;
        let x_train = self.x.as_ref();
        let alpha = self.alpha.as_slice();
        let ws = &mut self.workspace;
        let query = &mut self.query;
        match compiled.coord_mode()? {
            CoordMode::Dist | CoordMode::Either => {
                let mut thread_scratch = std::mem::take(&mut ws.thread_scratch);
                fill_squared_euclidean_cross(
                    x_train.as_ref(),
                    query.query_x.as_ref(),
                    query.query_dist.as_mut(),
                    &mut thread_scratch,
                );
                ws.thread_scratch = thread_scratch;
                compiled.apply_cross(
                    query.query_dist.as_ref(),
                    query.query_k_star.as_mut(),
                    query.query_scratch.as_mut(),
                )?;
            }
            CoordMode::Points => {
                compiled.apply_cross_points(
                    x_train.as_ref(),
                    query.query_x.as_ref(),
                    query.query_k_star.as_mut(),
                    query.query_scratch.as_mut(),
                )?;
            }
            CoordMode::Mixed => {
                let mut thread_scratch = std::mem::take(&mut ws.thread_scratch);
                fill_squared_euclidean_cross(
                    x_train.as_ref(),
                    query.query_x.as_ref(),
                    query.query_dist.as_mut(),
                    &mut thread_scratch,
                );
                ws.thread_scratch = thread_scratch;
                compiled.apply_cross_mixed(
                    query.query_dist.as_ref(),
                    x_train.as_ref(),
                    query.query_x.as_ref(),
                    query.query_k_star.as_mut(),
                    query.query_scratch.as_mut(),
                )?;
            }
        }
        if out.mean.len() != m {
            out.mean.resize(m, 0.0);
        }
        if out.variance.len() != m {
            out.variance.resize(m, 0.0);
        }
        for (col, mean) in out.mean.iter_mut().enumerate() {
            let mut sum = 0.0;
            for (row, &a) in alpha.iter().enumerate() {
                sum += query.query_k_star[(row, col)] * a;
            }
            *mean = sum;
        }
        let l = match &self.mapped_factor {
            Some(mapped) => mapped.l_view(),
            None => self.workspace.k_matrix.as_ref(),
        };
        faer::linalg::triangular_solve::solve_lower_triangular_in_place(
            l,
            query.query_k_star.as_mut(),
            faer_par(n),
        );
        match compiled.coord_mode()? {
            CoordMode::Dist | CoordMode::Either => compiled.fill_diag(&mut query.query_kss)?,
            CoordMode::Points | CoordMode::Mixed => {
                compiled.fill_diag_points(query.query_x.as_ref(), &mut query.query_kss)?
            }
        }
        let noise = self.likelihood.noise_variance();
        for col in 0..m {
            let mut vnorm = 0.0;
            for row in 0..n {
                let v = query.query_k_star[(row, col)];
                vnorm += v * v;
            }
            let mut latent = query.query_kss[col] - vnorm;
            if latent < 0.0 {
                latent = 0.0;
            }
            out.variance[col] = match options.variance_kind {
                VarianceKind::Latent => latent,
                VarianceKind::Observation => latent + noise,
            };
        }
        self.y_transform.inverse_transform_mean(&mut out.mean)?;
        self.y_transform
            .inverse_transform_variance(&mut out.variance)?;
        out.variance_kind = options.variance_kind;
        Ok(())
    }

    fn write_prediction(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        options: PredictOptions,
        out: &mut Prediction,
    ) -> Result<(), GprError> {
        if n_cols != self.d {
            return Err(GprError::DimensionMismatch {
                x_dim: n_cols,
                expected_dim: self.d,
            });
        }
        validate_query(xs, n_rows, n_cols)?;
        let compiled = &self.compiled;
        let x_train = self.x.as_ref();
        let alpha = self.alpha.as_slice();
        let n = self.n;
        let m = n_rows;
        let mut query_xs = xs.to_vec();
        self.x_transform.apply(&mut query_xs, n_rows, n_cols)?;
        let mut query_x = Mat::zeros(m, n_cols);
        pack_points_into(&query_xs, n_rows, n_cols, query_x.as_mut());
        let mut query_dist = Mat::zeros(n, m);
        let mut query_k_star = Mat::zeros(n, m);
        let mut query_scratch = Mat::zeros(n, m);
        let mut query_kss = vec![0.0; m];
        let mut thread_scratch = empty_thread_scratch();
        match compiled.coord_mode()? {
            CoordMode::Dist | CoordMode::Either => {
                fill_squared_euclidean_cross(
                    x_train.as_ref(),
                    query_x.as_ref(),
                    query_dist.as_mut(),
                    &mut thread_scratch,
                );
                compiled.apply_cross(
                    query_dist.as_ref(),
                    query_k_star.as_mut(),
                    query_scratch.as_mut(),
                )?;
            }
            CoordMode::Points => {
                compiled.apply_cross_points(
                    x_train.as_ref(),
                    query_x.as_ref(),
                    query_k_star.as_mut(),
                    query_scratch.as_mut(),
                )?;
            }
            CoordMode::Mixed => {
                fill_squared_euclidean_cross(
                    x_train.as_ref(),
                    query_x.as_ref(),
                    query_dist.as_mut(),
                    &mut thread_scratch,
                );
                compiled.apply_cross_mixed(
                    query_dist.as_ref(),
                    x_train.as_ref(),
                    query_x.as_ref(),
                    query_k_star.as_mut(),
                    query_scratch.as_mut(),
                )?;
            }
        }
        if out.mean.len() != m {
            out.mean.resize(m, 0.0);
        }
        if out.variance.len() != m {
            out.variance.resize(m, 0.0);
        }
        for (col, mean) in out.mean.iter_mut().enumerate() {
            let mut sum = 0.0;
            for (row, &a) in alpha.iter().enumerate() {
                sum += query_k_star[(row, col)] * a;
            }
            *mean = sum;
        }
        faer::linalg::triangular_solve::solve_lower_triangular_in_place(
            self.chol_l(),
            query_k_star.as_mut(),
            faer_par(n),
        );
        match compiled.coord_mode()? {
            CoordMode::Dist | CoordMode::Either => compiled.fill_diag(&mut query_kss)?,
            CoordMode::Points | CoordMode::Mixed => {
                compiled.fill_diag_points(query_x.as_ref(), &mut query_kss)?
            }
        }
        let noise = self.likelihood.noise_variance();
        for col in 0..m {
            let mut vnorm = 0.0;
            for row in 0..n {
                let v = query_k_star[(row, col)];
                vnorm += v * v;
            }
            let mut latent = query_kss[col] - vnorm;
            if latent < 0.0 {
                latent = 0.0;
            }
            out.variance[col] = match options.variance_kind {
                VarianceKind::Latent => latent,
                VarianceKind::Observation => latent + noise,
            };
        }
        self.y_transform.inverse_transform_mean(&mut out.mean)?;
        self.y_transform
            .inverse_transform_variance(&mut out.variance)?;
        out.variance_kind = options.variance_kind;
        Ok(())
    }

    /// Returns the predictive mean and query–query covariance at `xs`.
    ///
    /// Default [`PredictOptions`] uses [`VarianceKind::Observation`]: `σn²`
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
    ) -> Result<PredictiveCovariance, GprError> {
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
    pub fn predict_covariance_with(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        options: PredictOptions,
    ) -> Result<PredictiveCovariance, GprError> {
        self.write_covariance(xs, n_rows, n_cols, options)
    }

    /// Draws posterior samples at `xs` from [`Self::predict_covariance`].
    ///
    /// Each column of the returned column-major `m × n_draws` matrix is
    /// `μ + L z` with `z ∼ N(0, I)` and `L` the Cholesky factor of the
    /// posterior covariance. `seed` is the crate [`rand::rngs::SmallRng`]
    /// start state. Zero draws returns an empty vector after the covariance
    /// is formed.
    ///
    /// # Errors
    ///
    /// Same as [`Self::predict`], plus [`GprError::CholeskyFailed`] with
    /// [`CholeskyStage::Predict`] if the posterior covariance cannot be
    /// factored after [`JitterPolicy`] retries.
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
    ) -> Result<Vec<f64>, GprError> {
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
    ) -> Result<Vec<f64>, GprError> {
        let cov = self.write_covariance(xs, n_rows, n_cols, options)?;
        if n_draws == 0 {
            return Ok(Vec::new());
        }
        let m = cov.mean.len();
        let mut a = Mat::zeros(m, m);
        for col in 0..m {
            for row in 0..m {
                a[(row, col)] = cov.covariance[col * m + row];
            }
        }
        let req = llt::factor::cholesky_in_place_scratch::<f64>(m, faer_par(m), Default::default());
        let mut scratch = MemBuffer::new(req);
        cholesky_lower_with_policy(
            &mut a,
            &mut scratch,
            self.jitter_policy,
            CholeskyStage::Predict,
        )?;
        let mut rng = crate::rng::small_rng(seed);
        let mut out = vec![0.0; m * n_draws];
        let mut z = vec![0.0; m];
        let mut lz = vec![0.0; m];
        for draw in 0..n_draws {
            for slot in &mut z {
                *slot = crate::rng::unit_normal(&mut rng);
            }
            mul_lower_chol(a.as_ref(), &z, &mut lz);
            let col = &mut out[draw * m..(draw + 1) * m];
            for i in 0..m {
                col[i] = cov.mean[i] + lz[i];
            }
        }
        Ok(out)
    }

    fn write_covariance(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        options: PredictOptions,
    ) -> Result<PredictiveCovariance, GprError> {
        if n_cols != self.d {
            return Err(GprError::DimensionMismatch {
                x_dim: n_cols,
                expected_dim: self.d,
            });
        }
        validate_query(xs, n_rows, n_cols)?;
        let compiled = &self.compiled;
        let x_train = self.x.as_ref();
        let alpha = self.alpha.as_slice();
        let n = self.n;
        let m = n_rows;
        let mut query_xs = xs.to_vec();
        self.x_transform.apply(&mut query_xs, n_rows, n_cols)?;
        let mut query_x = Mat::zeros(m, n_cols);
        pack_points_into(&query_xs, n_rows, n_cols, query_x.as_mut());
        let mut query_dist = Mat::zeros(n, m);
        let mut query_k_star = Mat::zeros(n, m);
        let mut query_scratch = Mat::zeros(n, m);
        let mut thread_scratch = empty_thread_scratch();
        match compiled.coord_mode()? {
            CoordMode::Dist | CoordMode::Either => {
                fill_squared_euclidean_cross(
                    x_train.as_ref(),
                    query_x.as_ref(),
                    query_dist.as_mut(),
                    &mut thread_scratch,
                );
                compiled.apply_cross(
                    query_dist.as_ref(),
                    query_k_star.as_mut(),
                    query_scratch.as_mut(),
                )?;
            }
            CoordMode::Points => {
                compiled.apply_cross_points(
                    x_train.as_ref(),
                    query_x.as_ref(),
                    query_k_star.as_mut(),
                    query_scratch.as_mut(),
                )?;
            }
            CoordMode::Mixed => {
                fill_squared_euclidean_cross(
                    x_train.as_ref(),
                    query_x.as_ref(),
                    query_dist.as_mut(),
                    &mut thread_scratch,
                );
                compiled.apply_cross_mixed(
                    query_dist.as_ref(),
                    x_train.as_ref(),
                    query_x.as_ref(),
                    query_k_star.as_mut(),
                    query_scratch.as_mut(),
                )?;
            }
        }
        let mut mean = vec![0.0; m];
        for (col, slot) in mean.iter_mut().enumerate() {
            let mut sum = 0.0;
            for (row, &a) in alpha.iter().enumerate() {
                sum += query_k_star[(row, col)] * a;
            }
            *slot = sum;
        }
        faer::linalg::triangular_solve::solve_lower_triangular_in_place(
            self.chol_l(),
            query_k_star.as_mut(),
            faer_par(n),
        );
        let mut kss = Mat::zeros(m, m);
        let mut kss_scratch = Mat::zeros(m, m);
        fill_query_query_kernel(
            compiled,
            query_x.as_ref(),
            kss.as_mut(),
            kss_scratch.as_mut(),
            &mut thread_scratch,
        )?;
        for col in 0..m {
            for row in 0..m {
                let mut dot = 0.0;
                for k in 0..n {
                    dot += query_k_star[(k, row)] * query_k_star[(k, col)];
                }
                kss[(row, col)] -= dot;
            }
        }
        let noise = self.likelihood.noise_variance();
        for i in 0..m {
            let mut latent = kss[(i, i)];
            if latent < 0.0 {
                latent = 0.0;
            }
            kss[(i, i)] = match options.variance_kind {
                VarianceKind::Latent => latent,
                VarianceKind::Observation => latent + noise,
            };
        }
        self.y_transform.inverse_transform_mean(&mut mean)?;
        let mut covariance = vec![0.0; m * m];
        for col in 0..m {
            for row in 0..m {
                covariance[col * m + row] = kss[(row, col)];
            }
        }
        self.y_transform
            .inverse_transform_covariance(&mut covariance)?;
        Ok(PredictiveCovariance {
            mean,
            covariance,
            variance_kind: options.variance_kind,
        })
    }

    /// Returns leave-one-out mean and observation variance at every training
    /// point.
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
    pub fn loo_predict(&self) -> Result<Prediction, GprError> {
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
    pub fn loo_predict_with(&self, options: PredictOptions) -> Result<Prediction, GprError> {
        let y = self.y_train.as_slice();
        let alpha = self.alpha.as_slice();
        let n = self.n;
        let mut q_diag = vec![0.0; n];
        inv_diag_from_chol_l(self.chol_l(), &mut q_diag);
        let noise = self.likelihood.noise_variance();
        let mut mean = vec![0.0; n];
        let mut variance = vec![0.0; n];
        for i in 0..n {
            let qii = q_diag[i];
            if !qii.is_finite() || qii <= 0.0 {
                return Err(GprError::NonPositiveDefiniteMatrix);
            }
            mean[i] = y[i] - alpha[i] / qii;
            let obs = 1.0 / qii;
            variance[i] = match options.variance_kind {
                VarianceKind::Observation => obs,
                VarianceKind::Latent => (obs - noise).max(0.0),
            };
        }
        self.y_transform.inverse_transform_mean(&mut mean)?;
        self.y_transform.inverse_transform_variance(&mut variance)?;
        Ok(Prediction {
            mean,
            variance,
            variance_kind: options.variance_kind,
        })
    }
}

#[allow(private_bounds)] // `GprObjective` is crate-private; `refit` still needs `O: Optimizer` for it.
impl<O, S, C> FittedGpr<O, S, C>
where
    C: DistanceCacheSlot,
    O: Clone + for<'a> Optimizer<GprObjective<'a, O, S, C>>,
{
    /// Re-runs the stored optimizer on the stored training data from the current `θ`.
    ///
    /// This is the same `O` that [`Gpr::with_optimizer`] installed. Transforms
    /// are not re-fit. `n` and `d` stay the same.
    ///
    /// # Errors
    ///
    /// Same as [`Gpr::fit`].
    pub fn refit(&mut self) -> Result<(), GprError> {
        self.optimize_hyperparameters()
    }
}

#[allow(private_bounds)] // `DistanceCacheSlot` is crate-private; `refit` still needs it.
impl<C: DistanceCacheSlot> FittedGpr<Fixed, FullRecompute, C> {
    pub(crate) fn from_persisted(parts: PersistedModel<C>) -> Result<Self, GprError> {
        let n = parts.y_obs.len();
        if n == 0 {
            return Err(GprError::EmptyInput);
        }
        if parts.x_obs.len() % n != 0 {
            return Err(persist::persist_err("persisted x length is not n * d"));
        }
        let d = parts.x_obs.len() / n;
        if parts.alpha.len() != n {
            return Err(persist::persist_err(format!(
                "alpha has {} values, expected n = {n}",
                parts.alpha.len()
            )));
        }
        let mut x_buf = parts.x_obs.clone();
        parts.x_transform.apply(&mut x_buf, n, d)?;
        let mut y_buf = parts.y_obs.clone();
        parts.y_transform.transform(&mut y_buf)?;
        let mut workspace = Workspace::new(n)?;
        let compiled = parts.kernel.compile();
        if parts.distance_cache.policy() == DistanceCachePolicy::Always
            && compiled.needs_ard_sq_diff()
        {
            workspace.ensure_ard_sq_diff(n, d)?;
        } else {
            workspace.clear_ard_sq_diff();
        }
        Ok(Self {
            kernel: parts.kernel,
            compiled,
            likelihood: parts.likelihood,
            x_unfitted: parts.x_unfitted,
            y_unfitted: parts.y_unfitted,
            x_transform: parts.x_transform,
            y_transform: parts.y_transform,
            optimizer: Fixed,
            distance_cache: parts.distance_cache,
            jitter_policy: parts.jitter_policy,
            workspace,
            query: QueryWorkspace::new(),
            x: pack_points(&x_buf, n, d),
            y_train: y_buf,
            x_obs: parts.x_obs,
            y_obs: parts.y_obs,
            alpha: parts.alpha,
            n,
            d,
            mapped_factor: parts.mapped,
            _recompute: PhantomData,
        })
    }

    /// Rebuilds `L` and `α` at the current `θ` without a search.
    ///
    /// Transforms are not re-fit. `n` and `d` stay the same.
    ///
    /// # Errors
    ///
    /// Same as [`Gpr<Fixed>::factor`].
    pub fn refit(&mut self) -> Result<(), GprError> {
        self.factorize_current()
    }
}

fn fill_query_query_kernel(
    compiled: &CompiledKernel,
    query_x: MatRef<'_, f64>,
    kss: MatMut<'_, f64>,
    scratch: MatMut<'_, f64>,
    thread_scratch: &mut [Mat<f64>],
) -> Result<(), GprError> {
    match compiled.coord_mode()? {
        CoordMode::Dist | CoordMode::Either => {
            let m = query_x.nrows();
            let mut dist_ss = Mat::zeros(m, m);
            fill_squared_euclidean(query_x, dist_ss.as_mut(), thread_scratch);
            compiled.apply(dist_ss.as_ref(), kss, Triangle::Full, scratch)
        }
        CoordMode::Points => compiled.apply_points(query_x, kss, Triangle::Full, scratch),
        CoordMode::Mixed => {
            let m = query_x.nrows();
            let mut dist_ss = Mat::zeros(m, m);
            fill_squared_euclidean(query_x, dist_ss.as_mut(), thread_scratch);
            compiled.apply_mixed(
                MixedKernelViews::new(dist_ss.as_ref(), query_x),
                kss,
                Triangle::Full,
                scratch,
            )
        }
    }
}

fn mul_lower_chol(l: MatRef<'_, f64>, z: &[f64], out: &mut [f64]) {
    let m = l.nrows();
    debug_assert_eq!(z.len(), m);
    debug_assert_eq!(out.len(), m);
    for i in 0..m {
        let mut s = 0.0;
        for (j, &zj) in z.iter().enumerate().take(i + 1) {
            s += l[(i, j)] * zj;
        }
        out[i] = s;
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
