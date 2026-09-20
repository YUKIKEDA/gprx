//! [`Gpr`] trainer and [`FittedGpr`] factorization.

use std::fmt;
use std::marker::PhantomData;

use faer::Mat;

use crate::error::GprError;
use crate::kernel::{CompiledKernel, KernelSpec};
use crate::likelihood::GaussianLikelihood;
use crate::objective::GprObjective;
use crate::optimizer::{AcceptsRecompute, Fixed, FullRecompute, Lbfgs, Optimizer, PoleRecompute};
use crate::persist::{self, MappedTensors};
use crate::precision::DoublePrecision;
use crate::transform::{
    IdentityInput, IdentityTarget, TargetTransform, Transform, UnfittedTarget, UnfittedTransform,
};
use crate::workspace::{FitWorkspace, QueryWorkspace};

use super::factor::write_params;
use super::{
    AllocWorkspace, CachedDistances, DistanceCachePolicy, DistanceCacheSlot, FitBuffers,
    JitterPolicy, NoDistanceCache, RetainCholesky, ReuseCholesky, UncachedDistances,
};

/// Unfitted Exact GPR trainer: kernel, likelihood, transforms, optimizer, and
/// recompute strategy.
///
/// [`Self::fit`] consumes [`Gpr<O>`] where `O: `[`Optimizer`] and searches
/// hyperparameters. [`Gpr<Fixed>::factor`] factors at the current `θ` with no
/// search. Success returns [`FittedGpr`]. Failure returns the trainer with
/// [`GprError`] so the caller can change `θ` or data and try again.
/// Input and target transforms default to identity. Trainers from
/// [`Gpr::new`] store [`CachedDistances`] and [`RetainCholesky`] by default
/// (the speed pole). [`Gpr::with_prefer_memory`] switches to
/// [`UncachedDistances`] and [`ReuseCholesky`]. Standalone Linear, Constant,
/// and White kernels use [`Gpr::from_points`], which has no distance-cache
/// slot; the same prefer methods change only the Cholesky buffer.
/// The default type is
/// [`Gpr<Lbfgs, FullRecompute, CachedDistances, RetainCholesky>`].
/// [`Clone`] copies kernel, likelihood,
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
pub struct Gpr<O = Lbfgs, S = FullRecompute, C = CachedDistances, B = RetainCholesky> {
    kernel: KernelSpec,
    likelihood: GaussianLikelihood,
    x_transform: Box<dyn UnfittedTransform>,
    y_transform: Box<dyn UnfittedTarget>,
    optimizer: O,
    distance_cache: C,
    jitter_policy: JitterPolicy,
    _recompute: PhantomData<S>,
    _cholesky: PhantomData<B>,
}

impl<O, S, C, B> fmt::Debug for Gpr<O, S, C, B>
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

impl<O: Clone, S, C: Copy, B> Clone for Gpr<O, S, C, B> {
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
            _cholesky: PhantomData,
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
#[allow(private_bounds)] // `AllocWorkspace` is crate-private; the public slot is `CholeskyBuffer`.
pub struct FittedGpr<
    O = Lbfgs,
    S = FullRecompute,
    C: DistanceCacheSlot = CachedDistances,
    B: AllocWorkspace = RetainCholesky,
> {
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
    workspace: FitBuffers<C, B>,
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

impl<O: Clone, S, C: Copy + DistanceCacheSlot, B: AllocWorkspace> Clone for FittedGpr<O, S, C, B> {
    fn clone(&self) -> Self {
        let mut workspace = self.workspace.clone();
        if let Some(mapped) = &self.mapped_factor {
            persist::copy_l_into(workspace.core_mut().k_matrix.as_mut(), mapped.l_view());
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

impl<O, S, C, B: AllocWorkspace> fmt::Debug for FittedGpr<O, S, C, B>
where
    O: fmt::Debug,
    C: fmt::Debug + DistanceCacheSlot,
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
    /// constructor; the cache slot is [`CachedDistances`] and the Cholesky
    /// buffer is [`RetainCholesky`]. [`Self::with_prefer_memory`] / [`Self::with_prefer_speed`]
    /// switch those poles. Standalone Linear, Constant, and White kernels
    /// use [`Self::from_points`].
    pub fn new(kernel: KernelSpec, likelihood: GaussianLikelihood) -> Self {
        Self {
            kernel,
            likelihood,
            x_transform: Box::new(IdentityInput),
            y_transform: Box::new(IdentityTarget),
            optimizer: Lbfgs::new(),
            distance_cache: CachedDistances,
            jitter_policy: JitterPolicy::default(),
            _recompute: PhantomData,
            _cholesky: PhantomData,
        }
    }

    /// Builds a trainer for a kernel that evaluates from coordinates, not
    /// pairwise distances.
    ///
    /// Use this for a standalone Linear, Constant, or White kernel. The
    /// trainer has no [`DistanceCachePolicy`]. [`Self::with_prefer_memory`]
    /// / [`Self::with_prefer_speed`] still exist and change only the
    /// Cholesky buffer. Compositions that still fill distances
    /// (`RBF + White`, `Constant * RBF`) use [`Self::new`].
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
            _cholesky: PhantomData,
        }
    }
}

impl<O, S, C, B> Gpr<O, S, C, B> {
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
    /// The recompute strategy follows the Cholesky pole `B`:
    /// [`crate::ReuseCholesky`] is always [`FullRecompute`].
    /// [`crate::RetainCholesky`] is [`crate::IncrementalRecompute`] when
    /// `O2: `[`crate::UsesChangeIndices`], otherwise [`FullRecompute`].
    /// L-BFGS cannot be incremental. A custom optimizer that does not
    /// implement [`crate::UsesChangeIndices`] implements
    /// [`crate::PoleRecompute`]`<`[`crate::RetainCholesky`]`>` with
    /// [`FullRecompute`].
    ///
    /// [`Fixed`] is not an [`Optimizer`]; use [`Gpr<Fixed>::factor`] after
    /// this switch. argmin solvers are [`crate::Lbfgs`], [`crate::NonlinearCg`],
    /// [`crate::NelderMead`], and [`crate::Newton`]. A user type that implements [`Optimizer`]
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
    pub fn with_optimizer<O2: PoleRecompute<B>>(
        self,
        optimizer: O2,
    ) -> Gpr<O2, O2::Strategy, C, B> {
        Gpr {
            kernel: self.kernel,
            likelihood: self.likelihood,
            x_transform: self.x_transform,
            y_transform: self.y_transform,
            optimizer,
            distance_cache: self.distance_cache,
            jitter_policy: self.jitter_policy,
            _recompute: PhantomData,
            _cholesky: PhantomData,
        }
    }

    /// Selects whether the Cholesky factor keeps a dedicated `W` buffer.
    ///
    /// Public callers use [`Gpr::with_prefer_memory`] / [`Gpr::with_prefer_speed`].
    pub(crate) fn with_cholesky_buffer<B2>(self, _: B2) -> Gpr<O, O::Strategy, C, B2>
    where
        B2: crate::CholeskyBuffer,
        O: PoleRecompute<B2>,
    {
        Gpr {
            kernel: self.kernel,
            likelihood: self.likelihood,
            x_transform: self.x_transform,
            y_transform: self.y_transform,
            optimizer: self.optimizer,
            distance_cache: self.distance_cache,
            jitter_policy: self.jitter_policy,
            _recompute: PhantomData,
            _cholesky: PhantomData,
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

impl<O, S, C: DistanceCachePolicy, B> Gpr<O, S, C, B> {
    /// Selects the memory pole: no distance cache and a reused Cholesky buffer.
    ///
    /// The returned trainer is [`UncachedDistances`] + [`ReuseCholesky`].
    /// Training distances are computed from `X` each kernel build. `W`
    /// overwrites `L` during a gradient. Call before [`Gpr::fit`] /
    /// [`Gpr<Fixed>::factor`]. [`Self::with_prefer_speed`] restores the
    /// default. This method exists only on trainers from [`Gpr::new`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{GaussianLikelihood, Gpr, ReuseCholesky, UncachedDistances};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let gpr = Gpr::new(
    ///     KernelSpec::from(RbfKernel::new(1.0)?),
    ///     GaussianLikelihood::new(0.1)?,
    /// )
    /// .with_prefer_memory();
    /// let _: gprx::Gpr<_, _, UncachedDistances, ReuseCholesky> = gpr;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_prefer_memory(self) -> Gpr<O, FullRecompute, UncachedDistances, ReuseCholesky>
    where
        O: PoleRecompute<ReuseCholesky, Strategy = FullRecompute>,
    {
        self.with_distance_cache_policy(UncachedDistances)
            .with_cholesky_buffer(ReuseCholesky)
    }

    /// Selects the speed pole: cached distances and a dedicated `W` buffer.
    ///
    /// This is the default [`Gpr::new`] layout ([`CachedDistances`] +
    /// [`RetainCholesky`]). Use it to undo [`Self::with_prefer_memory`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{CachedDistances, GaussianLikelihood, Gpr, RetainCholesky};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let gpr = Gpr::new(
    ///     KernelSpec::from(RbfKernel::new(1.0)?),
    ///     GaussianLikelihood::new(0.1)?,
    /// )
    /// .with_prefer_memory()
    /// .with_prefer_speed();
    /// let _: gprx::Gpr<_, _, CachedDistances, RetainCholesky> = gpr;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_prefer_speed(self) -> Gpr<O, O::Strategy, CachedDistances, RetainCholesky>
    where
        O: PoleRecompute<RetainCholesky>,
    {
        self.with_distance_cache_policy(CachedDistances)
            .with_cholesky_buffer(RetainCholesky)
    }

    /// Sets whether training distances are cached across kernel builds.
    ///
    /// Public callers use [`Self::with_prefer_memory`] / [`Self::with_prefer_speed`].
    pub(crate) fn with_distance_cache_policy<C2: DistanceCachePolicy>(
        self,
        _: C2,
    ) -> Gpr<O, S, C2, B> {
        Gpr {
            kernel: self.kernel,
            likelihood: self.likelihood,
            x_transform: self.x_transform,
            y_transform: self.y_transform,
            optimizer: self.optimizer,
            distance_cache: C2::default(),
            jitter_policy: self.jitter_policy,
            _recompute: PhantomData,
            _cholesky: PhantomData,
        }
    }
}

impl<O, S, B> Gpr<O, S, NoDistanceCache, B> {
    /// Selects the memory-pole Cholesky layout. The cache slot stays
    /// [`NoDistanceCache`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, LinearKernel};
    /// use gprx::{GaussianLikelihood, Gpr, NoDistanceCache, ReuseCholesky};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let gpr = Gpr::from_points(
    ///     KernelSpec::from(LinearKernel::new(1.0)?),
    ///     GaussianLikelihood::new(0.1)?,
    /// )
    /// .with_prefer_memory();
    /// let _: gprx::Gpr<_, _, NoDistanceCache, ReuseCholesky> = gpr;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_prefer_memory(self) -> Gpr<O, FullRecompute, NoDistanceCache, ReuseCholesky>
    where
        O: PoleRecompute<ReuseCholesky, Strategy = FullRecompute>,
    {
        self.with_cholesky_buffer(ReuseCholesky)
    }

    /// Selects the speed-pole Cholesky layout. The cache slot stays
    /// [`NoDistanceCache`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, LinearKernel};
    /// use gprx::{GaussianLikelihood, Gpr, NoDistanceCache, RetainCholesky};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let gpr = Gpr::from_points(
    ///     KernelSpec::from(LinearKernel::new(1.0)?),
    ///     GaussianLikelihood::new(0.1)?,
    /// )
    /// .with_prefer_memory()
    /// .with_prefer_speed();
    /// let _: gprx::Gpr<_, _, NoDistanceCache, RetainCholesky> = gpr;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_prefer_speed(self) -> Gpr<O, O::Strategy, NoDistanceCache, RetainCholesky>
    where
        O: PoleRecompute<RetainCholesky>,
    {
        self.with_cholesky_buffer(RetainCholesky)
    }
}

#[allow(private_bounds)] // `GprObjective` is crate-private; `fit` still needs `O: Optimizer` for it.
impl<O, S, C, B> Gpr<O, S, C, B>
where
    S: AcceptsRecompute<O>,
    C: DistanceCacheSlot,
    B: AllocWorkspace,
    O: Clone + for<'a> Optimizer<GprObjective<'a, O, S, C, B>>,
{
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
    ) -> Result<FittedGpr<O, S, C, B>, (Self, GprError)> {
        let mut model = FittedGpr::prepare(self, x, n_rows, n_cols, y)?;
        match model.optimize_hyperparameters() {
            Ok(()) => {
                if let Err(err) = model.restore_cholesky_if_overwritten() {
                    return Err((model.into_trainer(), err));
                }
                Ok(model)
            }
            Err(err) => Err((model.into_trainer(), err)),
        }
    }
}

#[allow(private_bounds)] // `DistanceCacheSlot` is crate-private; `factor` still needs it.
impl<C: DistanceCacheSlot, B: AllocWorkspace> Gpr<Fixed, FullRecompute, C, B> {
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
    ) -> Result<FittedGpr<Fixed, FullRecompute, C, B>, (Self, GprError)> {
        let mut model = FittedGpr::prepare(self, x, n_rows, n_cols, y)?;
        match model.factorize_current() {
            Ok(()) => Ok(model),
            Err(err) => Err((model.into_trainer(), err)),
        }
    }
}

/// Drops the trainer and keeps the error so `?` works in `Result<_, GprError>`.
impl<O, S, C, B> From<(Gpr<O, S, C, B>, GprError)> for GprError {
    fn from((_, err): (Gpr<O, S, C, B>, GprError)) -> Self {
        err
    }
}

#[path = "fitted.rs"]
mod fitted;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
