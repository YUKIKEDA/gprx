//! Unfitted [`Gpr`] trainer.

use std::fmt;
use std::marker::PhantomData;

use crate::error::GprError;
use crate::gpr::GprObjective;
use crate::kernel::KernelSpec;
use crate::likelihood::GaussianLikelihood;
use crate::optimizer::{Fixed, Lbfgs, Optimizer};
use crate::param::write_params;
use crate::precision::{DoublePrecision, GpScalar};
use crate::transform::{IdentityInput, IdentityTarget, UnfittedTarget, UnfittedTransform};

use super::{ExactFit, FittedGpr, Policies};
use crate::policy::{CholeskyBuffer, DistanceCachePolicy, JitterPolicy, KernelExp};

/// Unfitted Exact GPR trainer: kernel, likelihood, transforms, optimizer, and
/// recompute strategy.
///
/// [`Self::fit`] consumes [`Gpr<O>`] where `O: `[`Optimizer`] and searches
/// hyperparameters. [`Gpr<Fixed>::factor`] factors at the current `θ` with no
/// search. Success returns [`FittedGpr`]. Failure returns the trainer with
/// [`GprError`] so the caller can change `θ` or data and try again.
/// Input and target transforms default to identity. The trainer stores three
/// runtime policies: [`DistanceCachePolicy`] (default
/// [`DistanceCachePolicy::Cached`]), [`CholeskyBuffer`] (default
/// [`CholeskyBuffer::Retain`]), and [`KernelExp`] (default
/// [`KernelExp::Accurate`]). [`Gpr::with_prefer_memory`] /
/// [`Gpr::with_prefer_speed`] set the memory / speed pole at once. The type
/// parameters are the optimizer `O` (default [`Lbfgs`]; [`Fixed`] for
/// [`Gpr<Fixed>::factor`]) and the precision `P` (default
/// [`DoublePrecision`]). [`Clone`] copies kernel, likelihood, transforms,
/// optimizer, and policies.
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
pub struct Gpr<O = Lbfgs, P = DoublePrecision> {
    pub(super) kernel: KernelSpec,
    pub(super) likelihood: GaussianLikelihood,
    pub(super) x_transform: Box<dyn UnfittedTransform>,
    pub(super) y_transform: Box<dyn UnfittedTarget>,
    pub(super) optimizer: O,
    pub(super) policies: Policies,
    pub(super) _precision: PhantomData<P>,
}

impl<O, P> fmt::Debug for Gpr<O, P>
where
    O: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Gpr")
            .field("kernel", &self.kernel)
            .field("likelihood", &self.likelihood)
            .field("optimizer", &self.optimizer)
            .field("distance_cache", &self.policies.distance_cache)
            .field("cholesky_buffer", &self.policies.cholesky_buffer)
            .field("math", &self.policies.math)
            .field("jitter_policy", &self.policies.jitter)
            .finish_non_exhaustive()
    }
}

impl<O: Clone, P> Clone for Gpr<O, P> {
    fn clone(&self) -> Self {
        Self {
            kernel: self.kernel.clone(),
            likelihood: self.likelihood,
            x_transform: self.x_transform.clone_box(),
            y_transform: self.y_transform.clone_box(),
            optimizer: self.optimizer.clone(),
            policies: self.policies,
            _precision: PhantomData,
        }
    }
}

impl Gpr {
    /// Builds an unfitted trainer that owns the kernel and observation noise.
    ///
    /// Input and target maps default to identity. The optimizer is [`Lbfgs`].
    /// Call [`Self::with_input_transform`] / [`Self::with_target_transform`]
    /// before [`Self::fit`] to standardize. Those methods take an unfitted
    /// map; `fit` / `factor` produce the fitted map stored on [`FittedGpr`].
    /// Call [`Self::with_optimizer`] to switch to [`Fixed`] or another
    /// [`Optimizer`]. Policies start at [`DistanceCachePolicy::Cached`],
    /// [`CholeskyBuffer::Retain`], and [`KernelExp::Accurate`]. A kernel that
    /// does not read pairwise distances (standalone Linear, Constant, White)
    /// never allocates the distance cache.
    pub fn new(kernel: KernelSpec, likelihood: GaussianLikelihood) -> Self {
        Self {
            kernel,
            likelihood,
            x_transform: Box::new(IdentityInput),
            y_transform: Box::new(IdentityTarget),
            optimizer: Lbfgs::new(),
            policies: Policies::default(),
            _precision: PhantomData,
        }
    }
}

impl<O, P> Gpr<O, P> {
    /// The one place a trainer changes its type parameters.
    fn retype<O2, P2>(self, optimizer: O2) -> (Gpr<O2, P2>, O) {
        (
            Gpr {
                kernel: self.kernel,
                likelihood: self.likelihood,
                x_transform: self.x_transform,
                y_transform: self.y_transform,
                optimizer,
                policies: self.policies,
                _precision: PhantomData,
            },
            self.optimizer,
        )
    }

    /// Replaces the input (`X`) transform. Intended to be called before fit.
    ///
    /// A single map, a [`crate::transform::Pipeline`], or
    /// [`crate::transform::ColumnwiseInput`]. One-step maps still use this
    /// method.
    pub fn with_input_transform(mut self, transform: impl UnfittedTransform + 'static) -> Self {
        self.x_transform = Box::new(transform);
        self
    }

    /// Selects the storage precision. Omitting it leaves [`DoublePrecision`].
    pub fn with_precision<P2: GpScalar>(self) -> Gpr<O, P2> {
        let (trainer, optimizer) = self.retype::<(), P2>(());
        trainer.retype(optimizer).0
    }

    /// Selects the kernel `exp`. Omitting it leaves [`KernelExp::Accurate`].
    ///
    /// `fit` and predict use the same `exp`. Hyperparameter `exp(θ)` is
    /// unchanged.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{GaussianLikelihood, Gpr, KernelExp};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let gpr = Gpr::new(
    ///     KernelSpec::from(RbfKernel::new(1.0)?),
    ///     GaussianLikelihood::new(0.1)?,
    /// )
    /// .with_math(KernelExp::FastApprox);
    /// let _fitted = gpr
    ///     .fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
    ///     .map_err(|(_, e)| e)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_math(mut self, math: KernelExp) -> Self {
        self.policies.math = math;
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

    pub(crate) fn from_owned(
        kernel: KernelSpec,
        likelihood: GaussianLikelihood,
        x_transform: Box<dyn UnfittedTransform>,
        y_transform: Box<dyn UnfittedTarget>,
        optimizer: O,
        policies: Policies,
    ) -> Self {
        Self {
            kernel,
            likelihood,
            x_transform,
            y_transform,
            optimizer,
            policies,
            _precision: PhantomData,
        }
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
        self.policies.jitter = policy;
        self
    }

    /// Replaces the optimizer, changing the type parameter `O`.
    ///
    /// A fit rebuilds only the kernel leaves a step touches when the
    /// optimizer reports changed coordinates
    /// ([`Optimizer::USES_CHANGE_INDICES`]) and the buffer is
    /// [`CholeskyBuffer::Retain`]; otherwise it rebuilds the whole kernel.
    ///
    /// [`Fixed`] is not an [`Optimizer`]; use [`Gpr<Fixed>::factor`] after
    /// this switch. argmin solvers are [`crate::Lbfgs`],
    /// [`crate::NelderMead`], and [`crate::TrustRegion`]. A user type that implements [`Optimizer`]
    /// uses this same method; there is no second solver slot.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{GaussianLikelihood, Gpr, NelderMead};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let gpr = Gpr::new(
    ///     KernelSpec::from(RbfKernel::new(1.0)?),
    ///     GaussianLikelihood::new(0.1)?,
    /// )
    /// .with_optimizer(NelderMead::new());
    /// let _fitted = gpr.fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0]).map_err(|(_, e)| e)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_optimizer<O2>(self, optimizer: O2) -> Gpr<O2, P> {
        self.retype(optimizer).0
    }

    /// Sets whether training distances are cached across kernel builds.
    ///
    /// See [`DistanceCachePolicy`]. [`Self::with_prefer_memory`] /
    /// [`Self::with_prefer_speed`] set this together with the Cholesky buffer.
    pub fn with_distance_cache_policy(mut self, policy: DistanceCachePolicy) -> Self {
        self.policies.distance_cache = policy;
        self
    }

    /// Sets where the gradient matrix `W` lives. See [`CholeskyBuffer`].
    pub fn with_cholesky_buffer(mut self, buffer: CholeskyBuffer) -> Self {
        self.policies.cholesky_buffer = buffer;
        self
    }

    /// Selects the memory pole: [`DistanceCachePolicy::Uncached`] and
    /// [`CholeskyBuffer::Reuse`].
    ///
    /// Training distances are computed from `X` each kernel build, and `W`
    /// overwrites `L` during a gradient. [`Self::with_prefer_speed`] restores
    /// the default.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{CholeskyBuffer, DistanceCachePolicy, GaussianLikelihood, Gpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let gpr = Gpr::new(
    ///     KernelSpec::from(RbfKernel::new(1.0)?),
    ///     GaussianLikelihood::new(0.1)?,
    /// )
    /// .with_prefer_memory();
    /// assert_eq!(gpr.distance_cache_policy(), DistanceCachePolicy::Uncached);
    /// assert_eq!(gpr.cholesky_buffer(), CholeskyBuffer::Reuse);
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_prefer_memory(self) -> Self {
        self.with_distance_cache_policy(DistanceCachePolicy::Uncached)
            .with_cholesky_buffer(CholeskyBuffer::Reuse)
    }

    /// Selects the speed pole: [`DistanceCachePolicy::Cached`] and
    /// [`CholeskyBuffer::Retain`] (the default).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{CholeskyBuffer, DistanceCachePolicy, GaussianLikelihood, Gpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let gpr = Gpr::new(
    ///     KernelSpec::from(RbfKernel::new(1.0)?),
    ///     GaussianLikelihood::new(0.1)?,
    /// )
    /// .with_prefer_memory()
    /// .with_prefer_speed();
    /// assert_eq!(gpr.distance_cache_policy(), DistanceCachePolicy::Cached);
    /// assert_eq!(gpr.cholesky_buffer(), CholeskyBuffer::Retain);
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_prefer_speed(self) -> Self {
        self.with_distance_cache_policy(DistanceCachePolicy::Cached)
            .with_cholesky_buffer(CholeskyBuffer::Retain)
    }

    /// Returns the distance-cache policy.
    pub fn distance_cache_policy(&self) -> DistanceCachePolicy {
        self.policies.distance_cache
    }

    /// Returns the Cholesky buffer policy.
    pub fn cholesky_buffer(&self) -> CholeskyBuffer {
        self.policies.cholesky_buffer
    }

    /// Returns the kernel `exp`.
    pub fn math(&self) -> KernelExp {
        self.policies.math
    }

    /// Returns the jitter retries used when `K + σn² I` fails to factor.
    pub fn jitter_policy(&self) -> JitterPolicy {
        self.policies.jitter
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
    /// Returns [`GprError::LengthMismatch`] if `out` is the wrong length
    /// or a custom leaf rejects the write.
    pub fn get_params(&self, out: &mut [f64]) -> Result<(), GprError> {
        write_params(&self.kernel, &self.likelihood, out)
    }
}

impl<O, P> Gpr<O, P>
where
    P: GpScalar,
    O: for<'a> Optimizer<GprObjective<'a, P>>,
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
    /// [`GprError::LengthMismatch`] if `x` or `y` has the wrong length,
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
    ) -> Result<FittedGpr<O, P>, (Self, GprError)> {
        let mut model = FittedGpr::prepare(self, x, n_rows, n_cols, y)?;
        let mut view = ExactFit {
            core: &mut model.core,
            store: &mut model.store,
        };
        match view.optimize(&model.optimizer) {
            Ok(()) => Ok(model),
            Err(err) => Err((model.into_trainer(), err)),
        }
    }
}

impl<P: GpScalar> Gpr<Fixed, P> {
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
    ) -> Result<FittedGpr<Fixed, P>, (Self, GprError)> {
        let mut model = FittedGpr::prepare(self, x, n_rows, n_cols, y)?;
        match model.fit_view().refactor() {
            Ok(()) => Ok(model),
            Err(err) => Err((model.into_trainer(), err)),
        }
    }
}

/// Drops the trainer and keeps the error so `?` works in `Result<_, GprError>`.
impl<O, P> From<(Gpr<O, P>, GprError)> for GprError {
    fn from((_, err): (Gpr<O, P>, GprError)) -> Self {
        err
    }
}
