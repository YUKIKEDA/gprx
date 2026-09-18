//! Batch Gaussian process regression: `A = K + σn² I`, LLT, and `α`.

use std::fmt;
use std::marker::PhantomData;

use dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt;
use faer::linalg::cholesky::llt::factor::{LltError, LltRegularization};
use faer::{Mat, MatMut, MatRef, Par};

use crate::error::{CholeskyStage, GprError};
use crate::kernel::{
    CompiledKernel, CoordMode, KernelSpec, Triangle, fill_ard_squared_diff, fill_squared_euclidean,
    fill_squared_euclidean_cross,
};
use crate::likelihood::GaussianLikelihood;
use crate::objective::GprObjective;
use crate::optimizer::{
    AcceptsRecompute, Fixed, FullRecompute, Lbfgs, OptResult, Optimizer, RecomputeStrategy,
};
use crate::param::Interval;
use crate::precision::DoublePrecision;
use crate::transform::{IdentityInput, IdentityTarget, TargetTransform, Transform};
use crate::workspace::{QueryWorkspace, Workspace, empty_thread_scratch};

/// Which predictive variance [`Prediction`] reports.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum VarianceKind {
    /// Variance of the latent function `f*`, without observation noise.
    Latent,
    /// Variance of a new observation `y*`, including `σn²`. This is the default.
    #[default]
    Observation,
}

/// Options for [`FittedGpr::predict`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PredictOptions {
    /// Which variance to return. Defaults to [`VarianceKind::Observation`].
    pub variance_kind: VarianceKind,
}

impl Default for PredictOptions {
    fn default() -> Self {
        Self {
            variance_kind: VarianceKind::Observation,
        }
    }
}

/// Selects whether training distances are reused across kernel builds.
///
/// Isotropic RBF, Matérn, Periodic, and RQ evaluate from an `n×n` squared
/// Euclidean matrix. ARD RBF / Matérn / RQ evaluate from raw `(Δx_d)²` stored
/// as `n × (n·d)`. [`Self::Always`] fills the matching tensor once per fit.
/// [`Self::Never`] recomputes it on every kernel build. Linear ignores this
/// setting.
///
/// # Examples
///
/// ```rust
/// use gprx::kernel::{KernelSpec, RbfKernel};
/// use gprx::{DistanceCachePolicy, GaussianLikelihood, Gpr};
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
/// let likelihood = GaussianLikelihood::new(0.1)?;
/// let gpr = Gpr::new(kernel, likelihood)
///     .with_distance_cache_policy(DistanceCachePolicy::Always);
/// let _fitted = gpr.fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0]).map_err(|(_, e)| e)?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum DistanceCachePolicy {
    /// Recompute isotropic `n×n` distances or ARD `(Δx_d)²` on every kernel build.
    Never,
    /// Fill distances once per fit and reuse them while `X` is unchanged.
    /// This is the default. ARD fits store an extra `n×(n·d)` tensor.
    #[default]
    Always,
}

/// Numerical Cholesky stabilizer, distinct from observation noise.
///
/// The first factorization always tries `A = K + σn² I` with no extra
/// diagonal. [`Self::Fixed`] retries once with that `j` if the first factor
/// fails. [`Self::Adaptive`] retries with `initial`, then
/// `initial * multiplier` on each later attempt, stopping at `max_retries`
/// or when `j` would exceed `max_jitter`. A successful retry factors
/// `A + j I`; this crate does not iteratively refine back to `A`.
/// Observation noise stays on [`GaussianLikelihood`].
///
/// The default is [`Self::fixed`]`(0.0)`: no retry, matching an unregularized
/// factor. [`GprError::CholeskyFailed::jitter`] is the last `j` that was
/// tried (`0.0` when the unregularized factor was the only attempt).
///
/// # Errors
///
/// Constructors return [`GprError::InvalidHyperparameter`] if a value is
/// non-finite or outside the domain below.
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
/// .with_jitter_policy(JitterPolicy::adaptive(1e-10, 10.0, 5, 1e-3)?);
/// let _fitted = gpr.fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0]).map_err(|(_, e)| e)?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum JitterPolicy {
    /// Retry the failed factor with this non-negative `j` on the diagonal.
    Fixed(FixedJitter),
    /// Retry with a growing `j` after the unregularized factor fails.
    Adaptive(AdaptiveJitter),
}

/// Non-negative diagonal offset for [`JitterPolicy::Fixed`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FixedJitter {
    jitter: f64,
}

/// Growing diagonal offsets for [`JitterPolicy::Adaptive`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdaptiveJitter {
    initial: f64,
    multiplier: f64,
    max_retries: usize,
    max_jitter: f64,
}

impl Default for JitterPolicy {
    fn default() -> Self {
        Self::Fixed(FixedJitter { jitter: 0.0 })
    }
}

impl JitterPolicy {
    /// Builds a single retry offset `j ≥ 0`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if `jitter` is not finite
    /// or is negative.
    pub fn fixed(jitter: f64) -> Result<Self, GprError> {
        if !jitter.is_finite() || jitter < 0.0 {
            return Err(GprError::InvalidHyperparameter {
                reason: format!("jitter must be finite and non-negative, got {jitter}"),
            });
        }
        Ok(Self::Fixed(FixedJitter { jitter }))
    }

    /// Builds a growing retry sequence after an unregularized factor fails.
    ///
    /// `initial` must be positive, `multiplier` must be greater than 1,
    /// `max_retries` must be at least 1, and `max_jitter` must be at least
    /// `initial`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if a value is non-finite
    /// or outside that domain.
    pub fn adaptive(
        initial: f64,
        multiplier: f64,
        max_retries: usize,
        max_jitter: f64,
    ) -> Result<Self, GprError> {
        if !initial.is_finite() || initial <= 0.0 {
            return Err(GprError::InvalidHyperparameter {
                reason: format!(
                    "adaptive initial jitter must be finite and positive, got {initial}"
                ),
            });
        }
        if !multiplier.is_finite() || multiplier <= 1.0 {
            return Err(GprError::InvalidHyperparameter {
                reason: format!(
                    "adaptive jitter multiplier must be finite and greater than 1, got {multiplier}"
                ),
            });
        }
        if max_retries < 1 {
            return Err(GprError::InvalidHyperparameter {
                reason: "adaptive jitter max_retries must be at least 1".to_owned(),
            });
        }
        if !max_jitter.is_finite() || max_jitter < initial {
            return Err(GprError::InvalidHyperparameter {
                reason: format!(
                    "adaptive max_jitter must be finite and at least initial ({initial}), got {max_jitter}"
                ),
            });
        }
        Ok(Self::Adaptive(AdaptiveJitter {
            initial,
            multiplier,
            max_retries,
            max_jitter,
        }))
    }

    fn retry_jitters(self) -> RetryJitters {
        match self {
            Self::Fixed(fixed) => {
                if fixed.jitter > 0.0 {
                    RetryJitters {
                        next: Some(fixed.jitter),
                        multiplier: 1.0,
                        remaining: 1,
                        max_jitter: fixed.jitter,
                    }
                } else {
                    RetryJitters {
                        next: None,
                        multiplier: 1.0,
                        remaining: 0,
                        max_jitter: 0.0,
                    }
                }
            }
            Self::Adaptive(adaptive) => RetryJitters {
                next: Some(adaptive.initial),
                multiplier: adaptive.multiplier,
                remaining: adaptive.max_retries,
                max_jitter: adaptive.max_jitter,
            },
        }
    }
}

impl FixedJitter {
    /// Returns the retry offset `j`.
    pub fn jitter(&self) -> f64 {
        self.jitter
    }
}

impl AdaptiveJitter {
    /// Returns the first retry offset.
    pub fn initial(&self) -> f64 {
        self.initial
    }

    /// Returns the factor applied after each failed retry.
    pub fn multiplier(&self) -> f64 {
        self.multiplier
    }

    /// Returns the maximum number of jittered attempts.
    pub fn max_retries(&self) -> usize {
        self.max_retries
    }

    /// Returns the largest retry offset that may be tried.
    pub fn max_jitter(&self) -> f64 {
        self.max_jitter
    }
}

struct RetryJitters {
    next: Option<f64>,
    multiplier: f64,
    remaining: usize,
    max_jitter: f64,
}

impl Iterator for RetryJitters {
    type Item = f64;

    fn next(&mut self) -> Option<f64> {
        if self.remaining == 0 {
            return None;
        }
        let j = self.next?;
        if j > self.max_jitter {
            return None;
        }
        self.remaining -= 1;
        let grown = j * self.multiplier;
        self.next = if grown.is_finite() { Some(grown) } else { None };
        Some(j)
    }
}

/// Predictive mean and (diagonal) variance at the query points.
///
/// [`FittedGpr::predict_into`] reuses `mean` / `variance` capacity when the
/// query length matches a previous call.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Prediction {
    /// Predictive mean on the original target scale.
    pub mean: Vec<f64>,
    /// Predictive variance on the original target scale.
    pub variance: Vec<f64>,
    /// Whether [`Self::variance`] is latent or observation variance.
    pub variance_kind: VarianceKind,
}

/// Unfitted Exact GPR trainer: kernel, likelihood, transforms, optimizer, and
/// recompute strategy.
///
/// [`Self::fit`] consumes [`Gpr<O>`] where `O: `[`Optimizer`] and searches
/// hyperparameters. [`Gpr<Fixed>::factor`] factors at the current `θ` with no
/// search. Success returns [`FittedGpr`]. Failure returns the trainer with
/// [`GprError`] so the caller can change `θ` or data and try again.
/// Input and target transforms default to identity. Training squared
/// distances default to [`DistanceCachePolicy::Always`]. The default type is
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
pub struct Gpr<O = Lbfgs, S = FullRecompute> {
    kernel: KernelSpec,
    likelihood: GaussianLikelihood,
    x_transform: Box<dyn Transform>,
    y_transform: Box<dyn TargetTransform>,
    optimizer: O,
    distance_cache_policy: DistanceCachePolicy,
    jitter_policy: JitterPolicy,
    _recompute: PhantomData<S>,
}

impl<O, S> fmt::Debug for Gpr<O, S>
where
    O: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Gpr")
            .field("kernel", &self.kernel)
            .field("likelihood", &self.likelihood)
            .field("optimizer", &self.optimizer)
            .field("distance_cache_policy", &self.distance_cache_policy)
            .field("jitter_policy", &self.jitter_policy)
            .finish_non_exhaustive()
    }
}

impl<O: Clone, S> Clone for Gpr<O, S> {
    fn clone(&self) -> Self {
        Self {
            kernel: self.kernel.clone(),
            likelihood: self.likelihood,
            x_transform: self.x_transform.clone_box(),
            y_transform: self.y_transform.clone_box(),
            optimizer: self.optimizer.clone(),
            distance_cache_policy: self.distance_cache_policy,
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
pub struct FittedGpr<O = Lbfgs, S = FullRecompute> {
    kernel: KernelSpec,
    compiled: CompiledKernel,
    likelihood: GaussianLikelihood,
    x_transform: Box<dyn Transform>,
    y_transform: Box<dyn TargetTransform>,
    optimizer: O,
    distance_cache_policy: DistanceCachePolicy,
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
    _recompute: PhantomData<S>,
}

impl<O: Clone, S> Clone for FittedGpr<O, S> {
    fn clone(&self) -> Self {
        Self {
            kernel: self.kernel.clone(),
            compiled: self.compiled.clone(),
            likelihood: self.likelihood,
            x_transform: self.x_transform.clone_box(),
            y_transform: self.y_transform.clone_box(),
            optimizer: self.optimizer.clone(),
            distance_cache_policy: self.distance_cache_policy,
            jitter_policy: self.jitter_policy,
            workspace: self.workspace.clone(),
            query: self.query.clone(),
            x_obs: self.x_obs.clone(),
            y_obs: self.y_obs.clone(),
            x: self.x.clone(),
            y_train: self.y_train.clone(),
            alpha: self.alpha.clone(),
            n: self.n,
            d: self.d,
            _recompute: PhantomData,
        }
    }
}

impl<O, S> fmt::Debug for FittedGpr<O, S>
where
    O: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FittedGpr")
            .field("n", &self.n)
            .field("d", &self.d)
            .field("kernel", &self.kernel)
            .field("likelihood", &self.likelihood)
            .field("distance_cache_policy", &self.distance_cache_policy)
            .field("jitter_policy", &self.jitter_policy)
            .finish_non_exhaustive()
    }
}

impl Gpr {
    /// Builds an unfitted trainer that owns the kernel and observation noise.
    ///
    /// Input and target maps default to identity. The optimizer is [`Lbfgs`].
    /// Call [`Self::with_input_transform`] / [`Self::with_target_transform`]
    /// before [`Self::fit`] to standardize. Call [`Self::with_optimizer`] to
    /// switch to [`Fixed`] or another [`Optimizer`].
    pub fn new(kernel: KernelSpec, likelihood: GaussianLikelihood) -> Self {
        Self {
            kernel,
            likelihood,
            x_transform: Box::new(IdentityInput),
            y_transform: Box::new(IdentityTarget),
            optimizer: Lbfgs::new(),
            distance_cache_policy: DistanceCachePolicy::Always,
            jitter_policy: JitterPolicy::default(),
            _recompute: PhantomData,
        }
    }
}

impl<O, S> Gpr<O, S> {
    /// Replaces the input (`X`) transform. Intended to be called before fit.
    pub fn with_input_transform(mut self, transform: impl Transform + 'static) -> Self {
        self.x_transform = Box::new(transform);
        self
    }

    /// Replaces the target (`y`) transform. Intended to be called before fit.
    pub fn with_target_transform(mut self, transform: impl TargetTransform + 'static) -> Self {
        self.y_transform = Box::new(transform);
        self
    }

    /// Sets whether training distances are cached across kernel builds.
    ///
    /// Intended to be called before [`Gpr::fit`] / [`Gpr<Fixed>::factor`].
    /// The default is [`DistanceCachePolicy::Always`]. See
    /// [`DistanceCachePolicy`].
    pub fn with_distance_cache_policy(mut self, policy: DistanceCachePolicy) -> Self {
        self.distance_cache_policy = policy;
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
    /// and [`crate::NelderMead`].
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
    pub fn with_optimizer<O2>(self, optimizer: O2) -> Gpr<O2, FullRecompute> {
        Gpr {
            kernel: self.kernel,
            likelihood: self.likelihood,
            x_transform: self.x_transform,
            y_transform: self.y_transform,
            optimizer,
            distance_cache_policy: self.distance_cache_policy,
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

#[allow(private_bounds)] // `GprObjective` is crate-private; `fit` still needs `O: Optimizer` for it.
impl<O, S> Gpr<O, S>
where
    S: RecomputeStrategy,
    O: Clone + for<'a> Optimizer<GprObjective<'a, O, S>>,
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
    pub fn with_recompute_strategy<S2: AcceptsRecompute<O>>(self, _: S2) -> Gpr<O, S2> {
        Gpr {
            kernel: self.kernel,
            likelihood: self.likelihood,
            x_transform: self.x_transform,
            y_transform: self.y_transform,
            optimizer: self.optimizer,
            distance_cache_policy: self.distance_cache_policy,
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
    ) -> Result<FittedGpr<O, S>, (Self, GprError)> {
        let mut model = FittedGpr::prepare(self, x, n_rows, n_cols, y)?;
        match model.optimize_hyperparameters() {
            Ok(()) => Ok(model),
            Err(err) => Err((model.into_trainer(), err)),
        }
    }
}

impl Gpr<Fixed> {
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
    ) -> Result<FittedGpr<Fixed>, (Self, GprError)> {
        let mut model = FittedGpr::prepare(self, x, n_rows, n_cols, y)?;
        match model.factorize_current() {
            Ok(()) => Ok(model),
            Err(err) => Err((model.into_trainer(), err)),
        }
    }
}

/// Drops the trainer and keeps the error so `?` works in `Result<_, GprError>`.
impl<O, S> From<(Gpr<O, S>, GprError)> for GprError {
    fn from((_, err): (Gpr<O, S>, GprError)) -> Self {
        err
    }
}

impl<O, S> FittedGpr<O, S> {
    #[allow(clippy::result_large_err)] // failure returns the trainer so the caller can retry
    fn prepare(
        mut gpr: Gpr<O, S>,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
        y: &[f64],
    ) -> Result<Self, (Gpr<O, S>, GprError)> {
        if let Err(err) = validate_training(x, n_rows, n_cols, y) {
            return Err((gpr, err));
        }
        let mut x_buf = x.to_vec();
        if let Err(err) = gpr.x_transform.fit(&x_buf, n_rows, n_cols) {
            return Err((gpr, err));
        }
        if let Err(err) = gpr.x_transform.apply(&mut x_buf, n_rows, n_cols) {
            return Err((gpr, err));
        }
        let mut y_buf = y.to_vec();
        if let Err(err) = gpr.y_transform.fit(&y_buf) {
            return Err((gpr, err));
        }
        if let Err(err) = gpr.y_transform.transform(&mut y_buf) {
            return Err((gpr, err));
        }
        let mut workspace = match Workspace::new(n_rows) {
            Ok(ws) => ws,
            Err(err) => return Err((gpr, err)),
        };
        let compiled = gpr.kernel.compile();
        if gpr.distance_cache_policy == DistanceCachePolicy::Always && compiled.needs_ard_sq_diff()
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
            x_transform: gpr.x_transform,
            y_transform: gpr.y_transform,
            optimizer: gpr.optimizer,
            distance_cache_policy: gpr.distance_cache_policy,
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
            _recompute: PhantomData,
        })
    }

    /// Drops `L` / `α` / training data and returns a trainer with the current
    /// kernel, likelihood, transforms, optimizer, distance-cache policy, and
    /// jitter policy.
    pub fn into_trainer(self) -> Gpr<O, S> {
        Gpr {
            kernel: self.kernel,
            likelihood: self.likelihood,
            x_transform: self.x_transform,
            y_transform: self.y_transform,
            optimizer: self.optimizer,
            distance_cache_policy: self.distance_cache_policy,
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
            self.workspace.k_matrix.as_ref(),
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
        let old_kernel = self.kernel.clone();
        let old_compiled = self.compiled.clone();
        let old_likelihood = self.likelihood;
        self.kernel = kernel;
        self.compiled = compiled;
        self.likelihood = likelihood;
        if let Err(err) = self.factorize_current() {
            self.kernel = old_kernel;
            self.compiled = old_compiled;
            self.likelihood = old_likelihood;
            let _ = self.factorize_current();
            return Err(err);
        }
        Ok(())
    }

    pub(crate) fn objective(&mut self) -> GprObjective<'_, O, S> {
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
    /// [`GprError::UnsupportedKernelOperation`] if a points-mode product tree
    /// needs a gradient. Distance-mode product trees are supported. Kernel
    /// and likelihood `θ` are committed together only after `A` factors. A
    /// rejected slice or a Cholesky failure leaves stored `θ` unchanged.
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
                cache: self.distance_cache_policy,
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
                    Par::Seq,
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
        O: Clone + for<'a> Optimizer<GprObjective<'a, O, S>>,
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
        let n_rows = self.n;
        factor_train_with_policy(
            &self.compiled,
            self.x.as_ref(),
            &mut self.workspace,
            &self.y_train,
            self.likelihood.noise_variance(),
            FactorPolicy {
                cache: self.distance_cache_policy,
                jitter: self.jitter_policy,
                stage: CholeskyStage::Fit,
            },
        )?;
        if self.alpha.len() != n_rows {
            self.alpha.resize(n_rows, 0.0);
        }
        for (i, slot) in self.alpha.iter_mut().enumerate() {
            *slot = self.workspace.rhs[(i, 0)];
        }
        Ok(())
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
        faer::linalg::triangular_solve::solve_lower_triangular_in_place(
            ws.k_matrix.as_ref(),
            query.query_k_star.as_mut(),
            Par::Seq,
        );
        match compiled.coord_mode()? {
            CoordMode::Dist | CoordMode::Either => compiled.fill_diag(&mut query.query_kss)?,
            CoordMode::Points => {
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
        let ws = &self.workspace;
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
            ws.k_matrix.as_ref(),
            query_k_star.as_mut(),
            Par::Seq,
        );
        match compiled.coord_mode()? {
            CoordMode::Dist | CoordMode::Either => compiled.fill_diag(&mut query_kss)?,
            CoordMode::Points => compiled.fill_diag_points(query_x.as_ref(), &mut query_kss)?,
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
        let ws = &self.workspace;
        let n = self.n;
        let mut q_diag = vec![0.0; n];
        inv_diag_from_chol_l(ws.k_matrix.as_ref(), &mut q_diag);
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

/// Writes the training Gram matrix.
///
/// Distance-mode leaves use squared Euclidean distances in `dist_cache`.
/// ARD leaves under [`DistanceCachePolicy::Always`] use `ard_sq_diff`
/// (`n × (n·d)` raw `(Δx_d)²`). [`DistanceCachePolicy::Never`] refills
/// every call so a stale cache cannot leak into MLL/grad.
fn apply_train_kernel(
    compiled: &CompiledKernel,
    x: MatRef<'_, f64>,
    ws: &mut Workspace<DoublePrecision>,
    policy: DistanceCachePolicy,
) -> Result<(), GprError> {
    match compiled.coord_mode()? {
        CoordMode::Dist | CoordMode::Either => {
            let refill = match policy {
                DistanceCachePolicy::Never => true,
                DistanceCachePolicy::Always => !ws.dist_ready,
            };
            if refill {
                let mut thread_scratch = std::mem::take(&mut ws.thread_scratch);
                fill_squared_euclidean(x, ws.dist_cache.as_mut(), &mut thread_scratch);
                ws.thread_scratch = thread_scratch;
                ws.dist_ready = policy == DistanceCachePolicy::Always;
            }
            compiled.apply(
                ws.dist_cache.as_ref(),
                ws.k_matrix.as_mut(),
                Triangle::Lower,
                ws.exp_buf.as_mut(),
            )
        }
        CoordMode::Points => {
            if compiled.needs_ard_sq_diff()
                && policy == DistanceCachePolicy::Always
                && ws.ard_sq_diff.ncols() > 0
            {
                let refill = !ws.ard_sq_diff_ready;
                if refill {
                    let mut thread_scratch = std::mem::take(&mut ws.thread_scratch);
                    fill_ard_squared_diff(x, ws.ard_sq_diff.as_mut(), &mut thread_scratch);
                    ws.thread_scratch = thread_scratch;
                    ws.ard_sq_diff_ready = true;
                }
                compiled.apply_from_ard_cache(
                    ws.ard_sq_diff.as_ref(),
                    x,
                    ws.k_matrix.as_mut(),
                    Triangle::Lower,
                    ws.exp_buf.as_mut(),
                )
            } else {
                compiled.apply_points(
                    x,
                    ws.k_matrix.as_mut(),
                    Triangle::Lower,
                    ws.exp_buf.as_mut(),
                )
            }
        }
    }
}

fn validate_training(x: &[f64], n_rows: usize, n_cols: usize, y: &[f64]) -> Result<(), GprError> {
    if n_rows == 0 || n_cols == 0 {
        return Err(GprError::EmptyInput);
    }
    let expected_x = n_rows.checked_mul(n_cols).ok_or(GprError::EmptyInput)?;
    if x.len() != expected_x {
        return Err(GprError::InvalidHyperparameter {
            reason: format!("expected {expected_x} feature values, got {}", x.len()),
        });
    }
    if y.len() != n_rows {
        return Err(GprError::InvalidHyperparameter {
            reason: format!("expected {n_rows} targets, got {}", y.len()),
        });
    }
    if x.iter().any(|v| !v.is_finite()) || y.iter().any(|v| !v.is_finite()) {
        return Err(GprError::NonFiniteInput);
    }
    Ok(())
}

fn validate_query(xs: &[f64], n_rows: usize, n_cols: usize) -> Result<(), GprError> {
    if n_rows == 0 || n_cols == 0 {
        return Err(GprError::EmptyInput);
    }
    let expected = n_rows.checked_mul(n_cols).ok_or(GprError::EmptyInput)?;
    if xs.len() != expected {
        return Err(GprError::InvalidHyperparameter {
            reason: format!("expected {expected} feature values, got {}", xs.len()),
        });
    }
    if xs.iter().any(|v| !v.is_finite()) {
        return Err(GprError::NonFiniteInput);
    }
    Ok(())
}

fn pack_points(x: &[f64], n_rows: usize, n_cols: usize) -> Mat<f64> {
    let mut dest = Mat::zeros(n_rows, n_cols);
    pack_points_into(x, n_rows, n_cols, dest.as_mut());
    dest
}

fn pack_points_into(x: &[f64], n_rows: usize, n_cols: usize, mut dest: MatMut<'_, f64>) {
    debug_assert_eq!(dest.nrows(), n_rows);
    debug_assert_eq!(dest.ncols(), n_cols);
    for col in 0..n_cols {
        for row in 0..n_rows {
            dest[(row, col)] = x[col * n_rows + row];
        }
    }
}

fn add_noise_to_diag(mut k: MatMut<'_, f64>, noise: f64) {
    let n = k.nrows();
    for i in 0..n {
        k[(i, i)] += noise;
    }
}

fn assemble_train_system(
    compiled: &CompiledKernel,
    x: MatRef<'_, f64>,
    ws: &mut Workspace<DoublePrecision>,
    y: &[f64],
    noise: f64,
    extra_diag: f64,
    cache: DistanceCachePolicy,
) -> Result<(), GprError> {
    apply_train_kernel(compiled, x, ws, cache)?;
    add_noise_to_diag(ws.k_matrix.as_mut(), noise);
    if extra_diag != 0.0 {
        add_noise_to_diag(ws.k_matrix.as_mut(), extra_diag);
    }
    for (i, &yi) in y.iter().enumerate() {
        ws.rhs[(i, 0)] = yi;
    }
    Ok(())
}

struct FactorPolicy {
    cache: DistanceCachePolicy,
    jitter: JitterPolicy,
    stage: CholeskyStage,
}

fn map_cholesky_jitter(err: GprError, jitter: f64) -> GprError {
    match err {
        GprError::CholeskyFailed {
            matrix_size, stage, ..
        } => GprError::CholeskyFailed {
            jitter,
            matrix_size,
            stage,
        },
        other => other,
    }
}

fn factor_train_with_policy(
    compiled: &CompiledKernel,
    x: MatRef<'_, f64>,
    ws: &mut Workspace<DoublePrecision>,
    y: &[f64],
    noise: f64,
    policy: FactorPolicy,
) -> Result<(), GprError> {
    assemble_train_system(compiled, x, ws, y, noise, 0.0, policy.cache)?;
    match cholesky_and_solve(
        &mut ws.k_matrix,
        &mut ws.rhs,
        &mut ws.faer_scratch,
        0.0,
        policy.stage,
    ) {
        Ok(()) => return Ok(()),
        Err(GprError::CholeskyFailed { .. }) => {}
        Err(err) => return Err(err),
    }
    let mut last_j = 0.0;
    for j in policy.jitter.retry_jitters() {
        last_j = j;
        assemble_train_system(compiled, x, ws, y, noise, j, policy.cache)?;
        match cholesky_and_solve(
            &mut ws.k_matrix,
            &mut ws.rhs,
            &mut ws.faer_scratch,
            0.0,
            policy.stage,
        ) {
            Ok(()) => return Ok(()),
            Err(GprError::CholeskyFailed { .. }) => {}
            Err(err) => return Err(map_cholesky_jitter(err, j)),
        }
    }
    let n = ws.k_matrix.nrows();
    Err(GprError::CholeskyFailed {
        jitter: last_j,
        matrix_size: n,
        stage: policy.stage,
    })
}

fn log_det_from_l(l: MatRef<'_, f64>, n: usize) -> f64 {
    let mut log_diag = 0.0;
    for i in 0..n {
        log_diag += l[(i, i)].ln();
    }
    2.0 * log_diag
}

fn neg_mll_from_factor(l: MatRef<'_, f64>, y: &[f64], alpha: &[f64], n: usize) -> f64 {
    let mut quad = 0.0;
    for i in 0..n {
        quad += y[i] * alpha[i];
    }
    let log_det = log_det_from_l(l, n);
    let log_two_pi = (2.0 * std::f64::consts::PI).ln();
    0.5 * (quad + log_det + n as f64 * log_two_pi)
}

fn write_params(
    kernel: &KernelSpec,
    likelihood: &GaussianLikelihood,
    out: &mut [f64],
) -> Result<(), GprError> {
    let n_kernel = kernel.num_params();
    require_param_len(out.len(), n_kernel + likelihood.num_params())?;
    kernel.get_params(&mut out[..n_kernel])?;
    likelihood.get_params(&mut out[n_kernel..])
}

fn require_param_len(actual: usize, expected: usize) -> Result<(), GprError> {
    if actual == expected {
        Ok(())
    } else {
        Err(GprError::InvalidHyperparameter {
            reason: format!("expected {expected} parameters, got {actual}"),
        })
    }
}

fn fill_identity(mut a: MatMut<'_, f64>) {
    let n = a.nrows();
    for col in 0..n {
        for row in 0..n {
            a[(row, col)] = if row == col { 1.0 } else { 0.0 };
        }
    }
}

fn form_w_lower(mut w: MatMut<'_, f64>, alpha: &[f64], n: usize) {
    for col in 0..n {
        for row in col..n {
            w[(row, col)] = alpha[row] * alpha[col] - w[(row, col)];
        }
    }
}

fn frobenius_lower(w: MatRef<'_, f64>, d_k: MatRef<'_, f64>, n: usize) -> f64 {
    let mut inner = 0.0;
    for col in 0..n {
        inner += w[(col, col)] * d_k[(col, col)];
        for row in col + 1..n {
            inner += 2.0 * w[(row, col)] * d_k[(row, col)];
        }
    }
    inner
}

fn write_kernel_grad(
    compiled: &CompiledKernel,
    dist: MatRef<'_, f64>,
    x: MatRef<'_, f64>,
    ard_cache: Option<MatRef<'_, f64>>,
    d_k: MatMut<'_, f64>,
    scratch: MatMut<'_, f64>,
    param_idx: usize,
) -> Result<(), GprError> {
    match compiled.coord_mode()? {
        CoordMode::Dist | CoordMode::Either => {
            compiled.grad(dist, d_k, param_idx, Triangle::Lower, scratch)
        }
        CoordMode::Points => {
            if let Some(cache) = ard_cache {
                compiled.grad_from_ard_cache(cache, x, d_k, param_idx, Triangle::Lower, scratch)
            } else {
                compiled.grad_points(x, d_k, param_idx, Triangle::Lower, scratch)
            }
        }
    }
}

/// Writes `diag(A⁻¹)` given the lower Cholesky factor `L` of `A = L Lᵀ`.
///
/// `A⁻¹ = L^{-T} L^{-1}`, so entry `i` is the squared Euclidean norm of
/// column `i` of `L⁻¹`.
fn inv_diag_from_chol_l(l: MatRef<'_, f64>, q_diag: &mut [f64]) {
    let n = l.nrows();
    debug_assert_eq!(q_diag.len(), n);
    let mut inv_l = Mat::from_fn(n, n, |row, col| if row == col { 1.0 } else { 0.0 });
    faer::linalg::triangular_solve::solve_lower_triangular_in_place(l, inv_l.as_mut(), Par::Seq);
    for (i, qi) in q_diag.iter_mut().enumerate() {
        let mut q = 0.0;
        for k in 0..n {
            let v = inv_l[(k, i)];
            q += v * v;
        }
        *qi = q;
    }
}

/// Factors `A` in place as `L Lᵀ` and overwrites `rhs` with `A⁻¹ rhs`.
///
/// P1A-18 can call this on the same `Workspace` buffers as [`Gpr::fit`].
pub(crate) fn cholesky_and_solve(
    a: &mut Mat<f64>,
    rhs: &mut Mat<f64>,
    scratch: &mut MemBuffer,
    jitter: f64,
    stage: CholeskyStage,
) -> Result<(), GprError> {
    let n = a.nrows();
    let regularization = LltRegularization {
        dynamic_regularization_delta: jitter,
        dynamic_regularization_epsilon: 0.0,
    };
    {
        let stack = MemStack::new(scratch);
        match llt::factor::cholesky_in_place(
            a.as_mut(),
            regularization,
            Par::Seq,
            stack,
            Default::default(),
        ) {
            Ok(_) => {}
            Err(LltError::NonPositivePivot { .. }) => {
                return Err(GprError::CholeskyFailed {
                    jitter,
                    matrix_size: n,
                    stage,
                });
            }
        }
    }
    let stack = MemStack::new(scratch);
    llt::solve::solve_in_place(a.as_ref(), rhs.as_mut(), Par::Seq, stack);
    Ok(())
}

#[allow(private_bounds)] // `GprObjective` is crate-private; `refit` still needs `O: Optimizer` for it.
impl<O, S> FittedGpr<O, S>
where
    O: Clone + for<'a> Optimizer<GprObjective<'a, O, S>>,
{
    /// Re-runs the stored optimizer on the stored training data from the current `θ`.
    ///
    /// Transforms are not re-fit. `n` and `d` stay the same.
    ///
    /// # Errors
    ///
    /// Same as [`Gpr::fit`].
    pub fn refit(&mut self) -> Result<(), GprError> {
        self.optimize_hyperparameters()
    }
}

impl FittedGpr<Fixed> {
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

#[cfg(test)]
mod tests {
    use super::{
        FittedGpr, Gpr, JitterPolicy, OptResult, Prediction, cholesky_and_solve, pack_points,
    };
    use crate::error::{CholeskyStage, GprError};
    use crate::kernel::{
        ConstantKernel, KernelSpec, KernelTerm, LinearKernel, MaternArdKernel, MaternKernel,
        MaternNu, PeriodicKernel, RationalQuadraticArdKernel, RationalQuadraticKernel,
        RbfArdKernel, RbfKernel, Triangle, WhiteKernel,
    };
    use crate::likelihood::GaussianLikelihood;
    use crate::objective::Objective;
    use crate::optimizer::{
        Fixed, FullRecompute, IncrementalRecompute, Lbfgs, NelderMead, NonlinearCg, Optimizer,
        UsesChangeIndices,
    };
    use crate::param::Interval;
    use crate::precision::DoublePrecision;
    use crate::transform::{MinMaxInput, StandardizeTarget, TargetTransform};
    use crate::workspace::Workspace;
    use faer::{Mat, MatMut, MatRef};

    const TOL: f64 = 1e-9;

    fn assert_close(actual: f64, expected: f64) {
        let scale = expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= TOL * scale,
            "actual={actual}, expected={expected}"
        );
    }

    fn assert_send_sync<T: Send + Sync>() {}

    fn rbf_gpr(ell: f64, noise: f64) -> Gpr {
        Gpr::new(
            KernelSpec::from(RbfKernel::new(ell).expect("valid")),
            GaussianLikelihood::new(noise).expect("valid"),
        )
    }

    fn rbf_ard_gpr(ells: &[f64], noise: f64) -> Gpr {
        Gpr::new(
            KernelSpec::from(RbfArdKernel::new(ells).expect("valid")),
            GaussianLikelihood::new(noise).expect("valid"),
        )
    }

    fn dense_a(kernel: &KernelSpec, noise: f64, x: &[f64], n: usize, d: usize) -> Mat<f64> {
        let compiled = kernel.compile();
        let x_mat = pack_points(x, n, d);
        let mut dist = Mat::zeros(n, n);
        crate::kernel::fill_squared_euclidean(x_mat.as_ref(), dist.as_mut(), &mut []);
        let mut k = Mat::zeros(n, n);
        let mut scratch = Mat::zeros(n, n);
        compiled
            .apply(dist.as_ref(), k.as_mut(), Triangle::Full, scratch.as_mut())
            .expect("shape");
        super::add_noise_to_diag(k.as_mut(), noise);
        k
    }

    fn copy_lower(src: faer::MatRef<'_, f64>) -> Mat<f64> {
        let n = src.nrows();
        Mat::from_fn(n, n, |i, j| if i >= j { src[(i, j)] } else { 0.0 })
    }

    fn matvec_sym(a: &Mat<f64>, x: &[f64]) -> Vec<f64> {
        let n = a.nrows();
        let mut out = vec![0.0; n];
        for col in 0..n {
            for row in 0..n {
                out[row] += a[(row, col)] * x[col];
            }
        }
        out
    }

    #[test]
    fn is_send_sync() {
        assert_send_sync::<Gpr>();
        assert_send_sync::<Gpr<NonlinearCg>>();
        assert_send_sync::<Gpr<NelderMead>>();
        assert_send_sync::<FittedGpr>();
        assert_send_sync::<FittedGpr<NonlinearCg>>();
        assert_send_sync::<FittedGpr<NelderMead>>();
        assert_send_sync::<super::Prediction>();
        assert_send_sync::<super::VarianceKind>();
        assert_send_sync::<super::PredictOptions>();
        assert_send_sync::<super::JitterPolicy>();
        assert_send_sync::<super::FixedJitter>();
        assert_send_sync::<super::AdaptiveJitter>();
        fn assert_clone<T: Clone>() {}
        assert_clone::<Gpr>();
        assert_clone::<Gpr<Fixed>>();
        assert_clone::<FittedGpr>();
        assert_clone::<FittedGpr<Fixed>>();
    }

    #[test]
    fn fit_restores_thread_scratch() {
        let gpr = rbf_gpr(1.0, 0.1)
            .fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
            .expect("spd");
        let ws = &gpr.workspace;
        assert_eq!(ws.thread_scratch.len(), rayon::current_num_threads().max(1));
        assert!(
            ws.thread_scratch
                .iter()
                .all(|m| m.nrows() == 0 && m.ncols() == 0)
        );
    }

    #[test]
    fn predict_restores_thread_scratch_when_apply_cross_fails() {
        let mut gpr = rbf_gpr(1.0, 0.1)
            .fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
            .expect("spd");
        gpr.query.ensure(2, 1, 1).expect("query");
        gpr.query.query_scratch = Mat::<f64>::zeros(1, 1);
        assert!(matches!(
            gpr.predict_into(&[0.5], 1, 1, &mut Prediction::default()),
            Err(GprError::WorkspaceTooSmall)
        ));
        let ws = &gpr.workspace;
        assert_eq!(ws.thread_scratch.len(), rayon::current_num_threads().max(1));
        assert!(
            ws.thread_scratch
                .iter()
                .all(|m| m.nrows() == 0 && m.ncols() == 0)
        );
    }

    #[test]
    fn predict_into_matches_predict() {
        let mut gpr = rbf_gpr(1.0, 0.1)
            .fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
            .expect("spd");
        let owned = gpr.predict(&[0.5], 1, 1).expect("fitted");
        let mut into = super::Prediction::default();
        gpr.predict_into(&[0.5], 1, 1, &mut into).expect("fitted");
        assert_eq!(into.mean, owned.mean);
        assert_eq!(into.variance, owned.variance);
        assert_eq!(into.variance_kind, owned.variance_kind);
        let ws = &gpr.workspace;
        assert_eq!(ws.thread_scratch.len(), rayon::current_num_threads().max(1));
        assert!(
            ws.thread_scratch
                .iter()
                .all(|m| m.nrows() == 0 && m.ncols() == 0)
        );
    }

    #[test]
    fn fit_solves_a_alpha_equals_y() {
        let x = [0.0, 0.5, 1.5, 0.0, 1.0, 0.5];
        let y = [0.2, -1.0, 0.7];
        let gpr = rbf_gpr(1.25, 0.1)
            .with_optimizer(Fixed)
            .factor(&x, 3, 2, &y)
            .expect("spd");
        assert_eq!(gpr.n(), 3);
        assert_eq!(gpr.d(), 2);
        let a = dense_a(gpr.kernel(), gpr.likelihood().noise_variance(), &x, 3, 2);
        let alpha = gpr.alpha();
        let restored = matvec_sym(&a, alpha);
        for i in 0..3 {
            assert_close(restored[i], y[i]);
        }
        let ws = &gpr.workspace;
        let l = copy_lower(ws.k_matrix.as_ref());
        let a_from_l = &l * l.transpose();
        for col in 0..3 {
            for row in col..3 {
                assert_close(a_from_l[(row, col)], a[(row, col)]);
            }
        }
    }

    #[test]
    fn refit_replaces_size_and_still_solves() {
        let gpr = rbf_gpr(1.25, 0.1)
            .with_optimizer(Fixed)
            .factor(&[0.0, 0.5, 1.5, 0.0, 1.0, 0.5], 3, 2, &[0.2, -1.0, 0.7])
            .expect("spd");
        let gpr = gpr
            .into_trainer()
            .with_optimizer(Fixed)
            .factor(&[0.0, 1.0], 2, 1, &[0.5, -0.25])
            .expect("refit");
        assert_eq!(gpr.n(), 2);
        assert_eq!(gpr.d(), 1);
        let a = dense_a(
            gpr.kernel(),
            gpr.likelihood().noise_variance(),
            &[0.0, 1.0],
            2,
            1,
        );
        let alpha = gpr.alpha();
        let restored = matvec_sym(&a, alpha);
        assert_close(restored[0], 0.5);
        assert_close(restored[1], -0.25);
    }

    #[test]
    fn fit_n_one_matches_scalar_solve() {
        let noise = 0.25;
        let gpr = rbf_gpr(1.0, noise)
            .with_optimizer(Fixed)
            .factor(&[0.0], 1, 1, &[2.0])
            .expect("spd");
        let a = 1.0 + noise;
        assert_close(gpr.alpha()[0], 2.0 / a);
    }

    #[test]
    fn validation_error_does_not_yield_fitted_model() {
        let gpr = rbf_gpr(1.0, 0.1)
            .with_optimizer(Fixed)
            .factor(&[0.0, 1.0], 2, 1, &[1.0, 2.0])
            .expect("spd");
        let alpha = gpr.alpha().to_vec();
        assert!(matches!(
            gpr.into_trainer().factor(&[0.0], 0, 1, &[]),
            Err((_, GprError::EmptyInput))
        ));
        let gpr = rbf_gpr(1.0, 0.1)
            .with_optimizer(Fixed)
            .factor(&[0.0, 1.0], 2, 1, &[1.0, 2.0])
            .expect("spd");
        assert_close(gpr.alpha()[0], alpha[0]);
        assert_close(gpr.alpha()[1], alpha[1]);
    }

    #[test]
    fn indefinite_matrix_returns_cholesky_failed() {
        let mut a = faer::mat![[1.0, 2.0], [2.0, 1.0]];
        let mut rhs = faer::mat![[1.0], [0.0]];
        let mut ws = Workspace::<DoublePrecision>::new(2).expect("n > 0");
        let err = cholesky_and_solve(
            &mut a,
            &mut rhs,
            &mut ws.faer_scratch,
            0.0,
            CholeskyStage::Fit,
        )
        .expect_err("indefinite");
        assert!(matches!(
            err,
            GprError::CholeskyFailed {
                stage: CholeskyStage::Fit,
                matrix_size: 2,
                jitter: 0.0,
            }
        ));
    }

    #[derive(Clone, Debug)]
    struct IndefiniteLeaf;

    impl KernelTerm for IndefiniteLeaf {
        fn num_params(&self) -> usize {
            0
        }

        fn get_params(&self, out: &mut [f64]) -> Result<(), GprError> {
            if out.is_empty() {
                Ok(())
            } else {
                Err(GprError::InvalidHyperparameter {
                    reason: "indefinite leaf has no parameters".to_owned(),
                })
            }
        }

        fn set_params(&mut self, params: &[f64]) -> Result<(), GprError> {
            self.get_params(&mut params.to_vec())
        }

        fn bounds_into(&self, out: &mut [Interval]) -> Result<(), GprError> {
            if out.is_empty() {
                Ok(())
            } else {
                Err(GprError::InvalidHyperparameter {
                    reason: "indefinite leaf has no parameters".to_owned(),
                })
            }
        }

        fn apply(
            &self,
            dist: MatRef<'_, f64>,
            mut out: MatMut<'_, f64>,
            uplo: Triangle,
        ) -> Result<(), GprError> {
            let n = dist.nrows();
            if n == 0 || dist.ncols() != n || out.nrows() != n || out.ncols() != n {
                return Err(GprError::InvalidHyperparameter {
                    reason: "indefinite leaf needs matching square matrices".to_owned(),
                });
            }
            for col in 0..n {
                let start = match uplo {
                    Triangle::Lower => col,
                    Triangle::Upper | Triangle::Full => 0,
                };
                let end = match uplo {
                    Triangle::Upper => col + 1,
                    Triangle::Lower | Triangle::Full => n,
                };
                for row in start..end {
                    out[(row, col)] = if row == col { 1.0 } else { 2.0 };
                }
            }
            Ok(())
        }

        fn apply_cross(
            &self,
            dist: MatRef<'_, f64>,
            mut out: MatMut<'_, f64>,
        ) -> Result<(), GprError> {
            for col in 0..out.ncols() {
                for row in 0..out.nrows() {
                    out[(row, col)] = if row == col { 1.0 } else { 2.0 };
                }
            }
            let _ = dist;
            Ok(())
        }

        fn fill_diag(&self, out: &mut [f64]) -> Result<(), GprError> {
            out.fill(1.0);
            Ok(())
        }

        fn grad(
            &self,
            _dist: MatRef<'_, f64>,
            _d_k: MatMut<'_, f64>,
            param_idx: usize,
            _uplo: Triangle,
        ) -> Result<(), GprError> {
            Err(GprError::InvalidHyperparameter {
                reason: format!("indefinite leaf has no parameter {param_idx}"),
            })
        }

        fn clone_box(&self) -> Box<dyn KernelTerm> {
            Box::new(self.clone())
        }
    }

    fn indefinite_gpr(noise: f64) -> Gpr {
        Gpr::new(
            KernelSpec::custom(IndefiniteLeaf),
            GaussianLikelihood::new(noise).expect("valid"),
        )
    }

    #[test]
    fn jitter_constructors_reject_invalid_values() {
        assert!(JitterPolicy::fixed(-1e-8).is_err());
        assert!(JitterPolicy::fixed(f64::NAN).is_err());
        assert!(JitterPolicy::adaptive(0.0, 10.0, 3, 1.0).is_err());
        assert!(JitterPolicy::adaptive(1e-8, 1.0, 3, 1.0).is_err());
        assert!(JitterPolicy::adaptive(1e-8, 10.0, 0, 1.0).is_err());
        assert!(JitterPolicy::adaptive(1.0, 10.0, 3, 0.5).is_err());
    }

    #[test]
    fn default_jitter_policy_reports_zero_on_failure() {
        let err = indefinite_gpr(0.1)
            .with_optimizer(Fixed)
            .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
            .expect_err("indefinite")
            .1;
        assert!(matches!(
            err,
            GprError::CholeskyFailed {
                jitter: 0.0,
                matrix_size: 2,
                stage: CholeskyStage::Fit,
            }
        ));
    }

    #[test]
    fn fixed_jitter_recovers_without_changing_noise() {
        let noise = 0.1;
        let fitted = indefinite_gpr(noise)
            .with_jitter_policy(JitterPolicy::fixed(1.0).expect("valid"))
            .with_optimizer(Fixed)
            .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
            .expect("A + j I is spd");
        assert_close(fitted.likelihood().noise_variance(), noise);
        assert_eq!(fitted.alpha().len(), 2);
        assert!(fitted.alpha().iter().all(|a| a.is_finite()));
    }

    #[test]
    fn adaptive_jitter_recovers_after_growth() {
        let fitted = indefinite_gpr(0.1)
            .with_jitter_policy(JitterPolicy::adaptive(0.1, 10.0, 3, 10.0).expect("valid"))
            .with_optimizer(Fixed)
            .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
            .expect("j grows past the negative eigenvalue");
        assert_close(fitted.likelihood().noise_variance(), 0.1);
    }

    #[test]
    fn adaptive_jitter_reports_last_attempt_when_capped() {
        let err = indefinite_gpr(0.1)
            .with_jitter_policy(JitterPolicy::adaptive(0.1, 10.0, 5, 0.5).expect("valid"))
            .with_optimizer(Fixed)
            .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
            .expect_err("max_jitter too small")
            .1;
        assert!(matches!(
            err,
            GprError::CholeskyFailed {
                jitter: j,
                matrix_size: 2,
                stage: CholeskyStage::Fit,
            } if (j - 0.1).abs() <= 1e-18
        ));
    }

    #[test]
    fn set_params_refactors_and_training_xy_roundtrip_through_factor() {
        let x = [0.0, 1.0];
        let y = [0.25, -0.5];
        let mut fitted = rbf_gpr(1.0, 0.1)
            .with_optimizer(Fixed)
            .factor(&x, 2, 1, &y)
            .expect("spd");
        assert_eq!(fitted.x(), x.as_slice());
        assert_eq!(fitted.y(), y.as_slice());
        let mut params = [0.0; 2];
        fitted.get_params(&mut params).expect("len 2");
        params[0] = 2.0_f64.ln();
        fitted.set_params(&params).expect("spd at new theta");
        let mut got = [0.0; 2];
        fitted.get_params(&mut got).expect("len 2");
        assert_close(got[0], params[0]);
        assert_close(got[1], params[1]);
        let pred = fitted.predict(&[0.5], 1, 1).expect("fitted");
        assert!(pred.mean[0].is_finite());

        let n = fitted.n();
        let d = fitted.d();
        let x_obs = fitted.x().to_vec();
        let y_obs = fitted.y().to_vec();
        let alpha = fitted.alpha().to_vec();
        let rebuilt = fitted
            .into_trainer()
            .with_optimizer(Fixed)
            .factor(&x_obs, n, d, &y_obs)
            .expect("same observations");
        assert_eq!(rebuilt.alpha().len(), alpha.len());
        for (a, b) in rebuilt.alpha().iter().zip(alpha.iter()) {
            assert_close(*a, *b);
        }
    }

    #[test]
    fn set_params_rejects_wrong_length_without_changing_theta() {
        let mut fitted = rbf_gpr(1.0, 0.1)
            .with_optimizer(Fixed)
            .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
            .expect("spd");
        let mut before = [0.0; 2];
        fitted.get_params(&mut before).expect("len 2");
        assert!(fitted.set_params(&[0.0]).is_err());
        let mut after = [0.0; 2];
        fitted.get_params(&mut after).expect("len 2");
        assert_close(before[0], after[0]);
        assert_close(before[1], after[1]);
    }

    #[test]
    fn clone_preserves_trainer_and_fitted_predict() {
        let trainer = rbf_gpr(1.25, 0.1);
        let trainer_clone = trainer.clone();
        let fitted = trainer
            .with_optimizer(Fixed)
            .factor(&[0.0, 1.0], 2, 1, &[0.5, -0.25])
            .expect("spd");
        let fitted_clone = fitted.clone();
        let p1 = fitted.predict(&[0.25], 1, 1).expect("fitted");
        let p2 = fitted_clone.predict(&[0.25], 1, 1).expect("clone");
        assert_close(p1.mean[0], p2.mean[0]);
        assert_close(p1.variance[0], p2.variance[0]);
        let other = trainer_clone
            .with_optimizer(Fixed)
            .factor(&[0.0, 1.0], 2, 1, &[0.5, -0.25])
            .expect("spd");
        let p3 = other.predict(&[0.25], 1, 1).expect("cloned trainer");
        assert_close(p1.mean[0], p3.mean[0]);
    }

    #[test]
    fn training_y_is_original_scale_with_standardize_target() {
        let y = [1.0, 3.0];
        let fitted = rbf_gpr(1.0, 0.1)
            .with_target_transform(StandardizeTarget::new())
            .with_optimizer(Fixed)
            .factor(&[0.0, 1.0], 2, 1, &y)
            .expect("spd");
        assert_eq!(fitted.y(), y.as_slice());
        let x_obs = fitted.x().to_vec();
        let y_obs = fitted.y().to_vec();
        let n = fitted.n();
        let d = fitted.d();
        let pred = fitted.predict(&[0.5], 1, 1).expect("fitted");
        let rebuilt = fitted
            .into_trainer()
            .with_optimizer(Fixed)
            .factor(&x_obs, n, d, &y_obs)
            .expect("roundtrip");
        let pred2 = rebuilt.predict(&[0.5], 1, 1).expect("rebuilt");
        assert_close(pred.mean[0], pred2.mean[0]);
        assert_close(pred.variance[0], pred2.variance[0]);
    }

    #[test]
    fn minmax_input_fit_predicts() {
        let gpr = rbf_gpr(1.0, 0.1)
            .with_input_transform(MinMaxInput::new())
            .with_optimizer(Fixed)
            .factor(&[0.0, 10.0], 2, 1, &[0.0, 1.0])
            .expect("spd");
        let pred = gpr.predict(&[5.0], 1, 1).expect("fitted");
        assert!(pred.mean[0].is_finite());
        assert!(pred.variance[0].is_finite());
        assert!(pred.variance[0] >= 0.0);
    }

    #[test]
    fn fit_rejects_bad_shapes_and_non_finite() {
        assert!(matches!(
            rbf_gpr(1.0, 0.1).fit(&[0.0], 2, 1, &[0.0, 1.0]),
            Err((_, GprError::InvalidHyperparameter { .. }))
        ));
        assert!(matches!(
            rbf_gpr(1.0, 0.1).fit(&[0.0, 1.0], 2, 1, &[0.0]),
            Err((_, GprError::InvalidHyperparameter { .. }))
        ));
        assert!(matches!(
            rbf_gpr(1.0, 0.1).fit(&[0.0, f64::NAN], 2, 1, &[0.0, 1.0]),
            Err((_, GprError::NonFiniteInput))
        ));
    }

    #[test]
    fn neg_mll_n_one_matches_closed_form() {
        let noise = 0.25;
        let y = 2.0;
        let gpr = rbf_gpr(1.0, noise)
            .with_optimizer(Fixed)
            .factor(&[0.0], 1, 1, &[y])
            .expect("spd");
        let a = 1.0 + noise;
        let log_det = a.ln();
        let ws = &gpr.workspace;
        assert_close(super::log_det_from_l(ws.k_matrix.as_ref(), 1), log_det);
        let quad = y * y / a;
        let expected = 0.5 * (quad + log_det + (2.0 * std::f64::consts::PI).ln());
        assert_close(gpr.neg_log_marginal_likelihood().expect("fitted"), expected);
    }

    #[test]
    fn neg_mll_n_two_matches_analytic_det_and_quad() {
        let ell = 1.0;
        let noise = 0.1;
        let x = [0.0, 1.0];
        let y = [0.5, -0.25];
        let gpr = rbf_gpr(ell, noise)
            .with_optimizer(Fixed)
            .factor(&x, 2, 1, &y)
            .expect("spd");
        let k01 = (-0.5 * (1.0 / ell) * (1.0 / ell)).exp();
        let diag = 1.0 + noise;
        let det = diag * diag - k01 * k01;
        let log_det = det.ln();
        let ws = &gpr.workspace;
        assert_close(super::log_det_from_l(ws.k_matrix.as_ref(), 2), log_det);
        let inv_scale = 1.0 / det;
        let quad =
            inv_scale * (y[0] * (diag * y[0] - k01 * y[1]) + y[1] * (-k01 * y[0] + diag * y[1]));
        let expected = 0.5 * (quad + log_det + 2.0 * (2.0 * std::f64::consts::PI).ln());
        assert_close(gpr.neg_log_marginal_likelihood().expect("fitted"), expected);
    }

    #[test]
    fn neg_mll_uses_transformed_targets() {
        let noise = 0.16;
        let y = [0.0, 4.0];
        let gpr = rbf_gpr(1.0, noise)
            .with_target_transform(StandardizeTarget::new())
            .with_optimizer(Fixed)
            .factor(&[0.0, 1.0], 2, 1, &y)
            .expect("spd");
        let mut t = StandardizeTarget::new();
        t.fit(&y).expect("finite");
        let mut y_t = y;
        t.transform(&mut y_t).expect("fitted");
        let k01 = (-0.5_f64).exp();
        let diag = 1.0 + noise;
        let det = diag * diag - k01 * k01;
        let log_det = det.ln();
        let inv_scale = 1.0 / det;
        let quad = inv_scale
            * (y_t[0] * (diag * y_t[0] - k01 * y_t[1]) + y_t[1] * (-k01 * y_t[0] + diag * y_t[1]));
        let expected = 0.5 * (quad + log_det + 2.0 * (2.0 * std::f64::consts::PI).ln());
        assert_close(gpr.neg_log_marginal_likelihood().expect("fitted"), expected);
        let raw = 0.5
            * (inv_scale
                * (y[0] * (diag * y[0] - k01 * y[1]) + y[1] * (-k01 * y[0] + diag * y[1]))
                + log_det
                + 2.0 * (2.0 * std::f64::consts::PI).ln());
        assert!((gpr.neg_log_marginal_likelihood().expect("fitted") - raw).abs() > TOL);
    }

    #[test]
    fn value_and_gradient_rejects_bad_len() {
        let mut gpr = rbf_gpr(1.0, 0.1)
            .with_optimizer(Fixed)
            .factor(&[0.0, 1.0], 2, 1, &[0.5, -0.25])
            .expect("spd");
        let mut grad = [0.0, 0.0];
        assert!(matches!(
            gpr.value_and_gradient_into(&[0.0], &mut grad),
            Err(GprError::InvalidHyperparameter { .. })
        ));
        assert!(matches!(
            gpr.get_params(&mut [0.0]),
            Err(GprError::InvalidHyperparameter { .. })
        ));
    }

    #[test]
    fn value_and_gradient_set_params_is_atomic() {
        let mut gpr = rbf_gpr(1.0, 0.1)
            .with_optimizer(Fixed)
            .factor(&[0.0, 1.0], 2, 1, &[0.5, -0.25])
            .expect("spd");
        let mut before = [0.0; 2];
        gpr.get_params(&mut before).expect("len 2");
        let mut bad = before;
        bad[0] = 0.5;
        bad[1] = f64::INFINITY;
        let mut grad = [0.0; 2];
        assert!(matches!(
            gpr.value_and_gradient_into(&bad, &mut grad),
            Err(GprError::InvalidNoiseVariance { .. })
        ));
        let mut after = [0.0; 2];
        gpr.get_params(&mut after).expect("len 2");
        assert_close(after[0], before[0]);
        assert_close(after[1], before[1]);
    }

    #[test]
    fn value_and_gradient_cholesky_failure_keeps_params() {
        let mut gpr = Gpr::new(
            KernelSpec::from(RbfKernel::new(1.0).expect("valid")),
            GaussianLikelihood::new(0.1)
                .expect("valid")
                .with_bounds(Interval::new(1e-30, 1e5).expect("open"))
                .expect("inside"),
        )
        .with_optimizer(Fixed)
        .factor(&[0.0, 0.0], 2, 1, &[0.5, -0.25])
        .expect("spd");
        let mut before = [0.0; 2];
        gpr.get_params(&mut before).expect("len 2");
        let mut bad = before;
        bad[0] = 0.5;
        bad[1] = (1e-20_f64).ln();
        let mut grad = [0.0; 2];
        assert!(matches!(
            gpr.value_and_gradient_into(&bad, &mut grad),
            Err(GprError::CholeskyFailed { .. })
        ));
        let mut after = [0.0; 2];
        gpr.get_params(&mut after).expect("len 2");
        assert_close(after[0], before[0]);
        assert_close(after[1], before[1]);
        gpr.value_and_gradient_into(&before, &mut grad)
            .expect("restore");
    }

    #[test]
    fn value_and_gradient_refills_stale_dist_cache() {
        let mut gpr = rbf_gpr(1.25, 0.16)
            .with_distance_cache_policy(super::DistanceCachePolicy::Never)
            .with_optimizer(Fixed)
            .factor(&[0.0, 0.8, 1.7], 3, 1, &[0.4, -0.2, 0.9])
            .expect("spd");
        {
            let ws = &mut gpr.workspace;
            let n = ws.dist_cache.nrows();
            ws.dist_cache = Mat::from_fn(n, n, |_, _| 999.0);
            ws.dist_ready = false;
        }
        let mut params = [0.0; 2];
        gpr.get_params(&mut params).expect("len 2");
        let mut grad = [0.0; 2];
        let value = gpr
            .value_and_gradient_into(&params, &mut grad)
            .expect("spd");
        assert_close(value, gpr.neg_log_marginal_likelihood().expect("fitted"));
        assert!(grad.iter().all(|g| g.is_finite()));
    }

    #[test]
    fn always_reuses_poisoned_dist_cache() {
        let x = [0.0, 0.8, 1.7];
        let y = [0.4, -0.2, 0.9];
        let mut gpr = rbf_gpr(1.25, 0.16)
            .with_distance_cache_policy(super::DistanceCachePolicy::Always)
            .with_optimizer(Fixed)
            .factor(&x, 3, 1, &y)
            .expect("spd");
        let mut params = [0.0; 2];
        gpr.get_params(&mut params).expect("len 2");
        let mut grad = [0.0; 2];
        let good = gpr
            .value_and_gradient_into(&params, &mut grad)
            .expect("spd");
        {
            let ws = &mut gpr.workspace;
            let n = ws.dist_cache.nrows();
            ws.dist_cache = Mat::from_fn(n, n, |_, _| 999.0);
            ws.dist_ready = true;
        }
        let poisoned = gpr
            .value_and_gradient_into(&params, &mut grad)
            .expect("spd");
        assert!(
            (poisoned - good).abs() > 1e-3,
            "Always should keep the poisoned distances: good={good}, poisoned={poisoned}"
        );
    }

    #[test]
    fn never_and_always_match_rbf_nlml_grad_and_predict() {
        let x = [0.0, 0.8, 1.7];
        let y = [0.4, -0.2, 0.9];
        let mut never = rbf_gpr(1.25, 0.16)
            .with_distance_cache_policy(super::DistanceCachePolicy::Never)
            .with_optimizer(Fixed)
            .factor(&x, 3, 1, &y)
            .expect("spd");
        let mut always = rbf_gpr(1.25, 0.16)
            .with_distance_cache_policy(super::DistanceCachePolicy::Always)
            .with_optimizer(Fixed)
            .factor(&x, 3, 1, &y)
            .expect("spd");
        let mut params = [0.0; 2];
        never.get_params(&mut params).expect("len 2");
        let mut grad_n = [0.0; 2];
        let mut grad_a = [0.0; 2];
        let vn = never
            .value_and_gradient_into(&params, &mut grad_n)
            .expect("spd");
        let va = always
            .value_and_gradient_into(&params, &mut grad_a)
            .expect("spd");
        assert_close(vn, va);
        assert_close(grad_n[0], grad_a[0]);
        assert_close(grad_n[1], grad_a[1]);
        let pn = never.predict(&[0.5], 1, 1).expect("fitted");
        let pa = always.predict(&[0.5], 1, 1).expect("fitted");
        assert_close(pn.mean[0], pa.mean[0]);
        assert_close(pn.variance[0], pa.variance[0]);
    }

    #[test]
    fn never_and_always_match_rbf_ard_nlml_grad_and_predict() {
        let x = [0.0, 0.8, 1.7, 0.2, -0.4, 0.9];
        let y = [0.4, -0.2, 0.9];
        let xs = [0.5, 0.1];
        let mut never = rbf_ard_gpr(&[1.25, 0.8], 0.16)
            .with_distance_cache_policy(super::DistanceCachePolicy::Never)
            .with_optimizer(Fixed)
            .factor(&x, 3, 2, &y)
            .expect("spd");
        let mut always = rbf_ard_gpr(&[1.25, 0.8], 0.16)
            .with_distance_cache_policy(super::DistanceCachePolicy::Always)
            .with_optimizer(Fixed)
            .factor(&x, 3, 2, &y)
            .expect("spd");
        assert_eq!(never.workspace.ard_sq_diff.ncols(), 0);
        assert_eq!(always.workspace.ard_sq_diff.ncols(), 6);
        let mut params = [0.0; 3];
        never.get_params(&mut params).expect("len 3");
        let mut grad_n = [0.0; 3];
        let mut grad_a = [0.0; 3];
        let vn = never
            .value_and_gradient_into(&params, &mut grad_n)
            .expect("spd");
        let va = always
            .value_and_gradient_into(&params, &mut grad_a)
            .expect("spd");
        assert_close(vn, va);
        assert_close(grad_n[0], grad_a[0]);
        assert_close(grad_n[1], grad_a[1]);
        assert_close(grad_n[2], grad_a[2]);
        let pn = never.predict(&xs, 1, 2).expect("fitted");
        let pa = always.predict(&xs, 1, 2).expect("fitted");
        assert_close(pn.mean[0], pa.mean[0]);
        assert_close(pn.variance[0], pa.variance[0]);
    }

    #[test]
    fn rbf_ard_fit_optimizes_with_always_cache() {
        let x = [0.0, 0.8, 1.7, 0.2, -0.4, 0.9];
        let y = [0.4, -0.2, 0.9];
        let gpr = rbf_ard_gpr(&[1.25, 0.8], 0.16)
            .with_distance_cache_policy(super::DistanceCachePolicy::Always);
        let mut before = [0.0; 3];
        gpr.get_params(&mut before).expect("len 3");
        let gpr = gpr.fit(&x, 3, 2, &y).expect("optimize");
        let mut after = [0.0; 3];
        gpr.get_params(&mut after).expect("len 3");
        assert!(
            before.iter().zip(&after).any(|(a, b)| (a - b).abs() > 1e-9),
            "L-BFGS should move ARD θ: before={before:?}, after={after:?}"
        );
        assert_eq!(gpr.workspace.ard_sq_diff.ncols(), 6);
        assert!(gpr.workspace.ard_sq_diff_ready);
    }

    #[test]
    fn always_reuses_poisoned_ard_cache() {
        let x = [0.0, 0.8, 1.7, 0.2, -0.4, 0.9];
        let y = [0.4, -0.2, 0.9];
        let mut gpr = rbf_ard_gpr(&[1.25, 0.8], 0.16)
            .with_distance_cache_policy(super::DistanceCachePolicy::Always)
            .with_optimizer(Fixed)
            .factor(&x, 3, 2, &y)
            .expect("spd");
        let mut params = [0.0; 3];
        gpr.get_params(&mut params).expect("len 3");
        let mut grad = [0.0; 3];
        let good = gpr
            .value_and_gradient_into(&params, &mut grad)
            .expect("spd");
        {
            let ws = &mut gpr.workspace;
            let n = 3;
            for dim in 0..2 {
                for col in 0..n {
                    for row in col..n {
                        ws.ard_sq_diff[(row, dim * n + col)] = 999.0;
                    }
                }
            }
            ws.ard_sq_diff_ready = true;
        }
        let poisoned = gpr
            .value_and_gradient_into(&params, &mut grad)
            .expect("spd");
        assert!(
            (poisoned - good).abs() > 1e-3,
            "Always should keep the poisoned ARD cache: good={good}, poisoned={poisoned}"
        );
    }

    #[test]
    fn always_ard_cache_retiling_follows_n() {
        let gpr = rbf_ard_gpr(&[1.0, 1.5], 0.16)
            .with_distance_cache_policy(super::DistanceCachePolicy::Always)
            .with_optimizer(Fixed)
            .factor(&[0.0, 0.8, 1.7, 0.2, -0.4, 0.9], 3, 2, &[0.4, -0.2, 0.9])
            .expect("spd n=3");
        {
            let ws = &gpr.workspace;
            assert_eq!(ws.ard_sq_diff.nrows(), 3);
            assert_eq!(ws.ard_sq_diff.ncols(), 6);
        }
        let gpr = gpr
            .into_trainer()
            .with_optimizer(Fixed)
            .factor(
                &[0.0, 0.8, 1.7, 2.1, 0.2, -0.4, 0.9, 0.3],
                4,
                2,
                &[0.4, -0.2, 0.9, 0.1],
            )
            .expect("spd n=4");
        let ws = &gpr.workspace;
        assert_eq!(ws.ard_sq_diff.nrows(), 4);
        assert_eq!(ws.ard_sq_diff.ncols(), 8);
        assert!(ws.ard_sq_diff_ready);
    }

    #[test]
    fn isotropic_always_leaves_ard_cache_empty() {
        let gpr = rbf_gpr(1.25, 0.16)
            .with_distance_cache_policy(super::DistanceCachePolicy::Always)
            .with_optimizer(Fixed)
            .factor(&[0.0, 0.8, 1.7], 3, 1, &[0.4, -0.2, 0.9])
            .expect("spd");
        let ws = &gpr.workspace;
        assert_eq!(ws.ard_sq_diff.nrows(), 0);
        assert_eq!(ws.ard_sq_diff.ncols(), 0);
        assert!(!ws.ard_sq_diff_ready);
    }

    #[test]
    fn never_and_always_match_matern_ard_nlml() {
        let x = [0.0, 0.8, 1.7, 0.2, -0.4, 0.9];
        let y = [0.4, -0.2, 0.9];
        let nu = MaternNu::ThreeHalves;
        let mut never = Gpr::new(
            KernelSpec::from(MaternArdKernel::new(&[1.25, 0.8], nu).expect("valid")),
            GaussianLikelihood::new(0.16).expect("valid"),
        )
        .with_distance_cache_policy(super::DistanceCachePolicy::Never)
        .with_optimizer(Fixed)
        .factor(&x, 3, 2, &y)
        .expect("spd");
        let mut always = Gpr::new(
            KernelSpec::from(MaternArdKernel::new(&[1.25, 0.8], nu).expect("valid")),
            GaussianLikelihood::new(0.16).expect("valid"),
        )
        .with_distance_cache_policy(super::DistanceCachePolicy::Always)
        .with_optimizer(Fixed)
        .factor(&x, 3, 2, &y)
        .expect("spd");
        let mut params = [0.0; 3];
        never.get_params(&mut params).expect("len 3");
        let mut grad_n = [0.0; 3];
        let mut grad_a = [0.0; 3];
        let vn = never
            .value_and_gradient_into(&params, &mut grad_n)
            .expect("spd");
        let va = always
            .value_and_gradient_into(&params, &mut grad_a)
            .expect("spd");
        assert_close(vn, va);
        assert_close(grad_n[0], grad_a[0]);
        assert_close(grad_n[1], grad_a[1]);
        assert_close(grad_n[2], grad_a[2]);
    }

    #[test]
    fn value_and_gradient_matches_nlml_and_finite_difference() {
        let mut gpr = rbf_gpr(1.25, 0.16)
            .with_optimizer(Fixed)
            .factor(&[0.0, 0.8, 1.7], 3, 1, &[0.4, -0.2, 0.9])
            .expect("spd");
        let mut params = [0.0; 2];
        gpr.get_params(&mut params).expect("len 2");
        let mut grad = [0.0; 2];
        let value = gpr
            .value_and_gradient_into(&params, &mut grad)
            .expect("spd");
        assert_close(value, gpr.neg_log_marginal_likelihood().expect("fitted"));
        let h = 1e-5;
        let mut dummy = [0.0; 2];
        for i in 0..2 {
            let mut plus = params;
            let mut minus = params;
            plus[i] += h;
            minus[i] -= h;
            let v_plus = gpr
                .value_and_gradient_into(&plus, &mut dummy)
                .expect("plus");
            let v_minus = gpr
                .value_and_gradient_into(&minus, &mut dummy)
                .expect("minus");
            let fd = (v_plus - v_minus) / (2.0 * h);
            let scale = fd.abs().max(1.0);
            assert!(
                (grad[i] - fd).abs() <= 1e-5 * scale,
                "param {i}: analytic={}, fd={}",
                grad[i],
                fd
            );
        }
        gpr.value_and_gradient_into(&params, &mut dummy)
            .expect("restore");
    }

    #[test]
    fn value_and_gradient_sum_rbf_matches_finite_difference() {
        let kernel = KernelSpec::from(RbfKernel::new(1.25).expect("valid"))
            + KernelSpec::from(RbfKernel::new(0.7).expect("valid"));
        let mut gpr = Gpr::new(kernel, GaussianLikelihood::new(0.16).expect("valid"))
            .with_optimizer(Fixed)
            .factor(&[0.0, 0.8, 1.7], 3, 1, &[0.4, -0.2, 0.9])
            .expect("spd");
        let n_params = gpr.num_params();
        let mut params = vec![0.0; n_params];
        gpr.get_params(&mut params).expect("len");
        let mut grad = vec![0.0; n_params];
        gpr.value_and_gradient_into(&params, &mut grad)
            .expect("spd");
        let h = 1e-5;
        let mut dummy = vec![0.0; n_params];
        for i in 0..n_params {
            let mut plus = params.clone();
            let mut minus = params.clone();
            plus[i] += h;
            minus[i] -= h;
            let v_plus = gpr
                .value_and_gradient_into(&plus, &mut dummy)
                .expect("plus");
            let v_minus = gpr
                .value_and_gradient_into(&minus, &mut dummy)
                .expect("minus");
            let fd = (v_plus - v_minus) / (2.0 * h);
            let scale = fd.abs().max(1.0);
            assert!(
                (grad[i] - fd).abs() <= 1e-5 * scale,
                "param {i}: analytic={}, fd={}",
                grad[i],
                fd
            );
        }
    }

    #[test]
    fn value_and_gradient_product_matches_finite_difference() {
        let kernel = KernelSpec::from(RbfKernel::new(1.0).expect("valid"))
            * KernelSpec::from(RbfKernel::new(2.0).expect("valid"));
        let mut gpr = Gpr::new(kernel, GaussianLikelihood::new(0.1).expect("valid"))
            .with_optimizer(Fixed)
            .factor(&[0.0, 1.0], 2, 1, &[0.5, -0.25])
            .expect("spd");
        let n_params = gpr.num_params();
        let mut params = vec![0.0; n_params];
        gpr.get_params(&mut params).expect("len");
        let mut grad = vec![0.0; n_params];
        gpr.value_and_gradient_into(&params, &mut grad)
            .expect("spd");
        let h = 1e-5;
        let mut dummy = vec![0.0; n_params];
        for i in 0..n_params {
            let mut plus = params.clone();
            let mut minus = params.clone();
            plus[i] += h;
            minus[i] -= h;
            let v_plus = gpr
                .value_and_gradient_into(&plus, &mut dummy)
                .expect("plus");
            let v_minus = gpr
                .value_and_gradient_into(&minus, &mut dummy)
                .expect("minus");
            let fd = (v_plus - v_minus) / (2.0 * h);
            let scale = fd.abs().max(1.0);
            assert!(
                (grad[i] - fd).abs() <= 1e-5 * scale,
                "param {i}: analytic={}, fd={}",
                grad[i],
                fd
            );
        }
    }

    #[test]
    fn value_and_gradient_sum_of_product_matches_finite_difference() {
        let kernel = KernelSpec::from(ConstantKernel::new(1.5).expect("valid"))
            * KernelSpec::from(RbfKernel::new(1.0).expect("valid"))
            + KernelSpec::from(RbfKernel::new(2.0).expect("valid"));
        let mut gpr = Gpr::new(kernel, GaussianLikelihood::new(0.1).expect("valid"))
            .with_optimizer(Fixed)
            .factor(&[0.0, 1.0], 2, 1, &[0.5, -0.25])
            .expect("spd");
        let n_params = gpr.num_params();
        let mut params = vec![0.0; n_params];
        gpr.get_params(&mut params).expect("len");
        let mut grad = vec![0.0; n_params];
        gpr.value_and_gradient_into(&params, &mut grad)
            .expect("spd");
        let h = 1e-5;
        let mut dummy = vec![0.0; n_params];
        for i in 0..n_params {
            let mut plus = params.clone();
            let mut minus = params.clone();
            plus[i] += h;
            minus[i] -= h;
            let v_plus = gpr
                .value_and_gradient_into(&plus, &mut dummy)
                .expect("plus");
            let v_minus = gpr
                .value_and_gradient_into(&minus, &mut dummy)
                .expect("minus");
            let fd = (v_plus - v_minus) / (2.0 * h);
            let scale = fd.abs().max(1.0);
            assert!(
                (grad[i] - fd).abs() <= 1e-5 * scale,
                "param {i}: analytic={}, fd={}",
                grad[i],
                fd
            );
        }
    }

    #[test]
    fn value_and_gradient_n_one_noise_matches_closed_form() {
        let noise = 0.25;
        let y = 2.0;
        let mut gpr = rbf_gpr(1.0, noise)
            .with_optimizer(Fixed)
            .factor(&[0.0], 1, 1, &[y])
            .expect("spd");
        let mut params = [0.0; 2];
        gpr.get_params(&mut params).expect("len 2");
        let mut grad = [0.0; 2];
        gpr.value_and_gradient_into(&params, &mut grad)
            .expect("spd");
        let a = 1.0 + noise;
        let w = (y / a) * (y / a) - 1.0 / a;
        assert_close(grad[0], 0.0);
        assert_close(grad[1], -0.5 * w * noise);
    }

    #[test]
    fn predict_rejects_wrong_dim() {
        let gpr = rbf_gpr(1.0, 0.1)
            .with_optimizer(Fixed)
            .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
            .expect("spd");
        assert!(matches!(
            gpr.predict(&[0.0, 1.0], 1, 2),
            Err(GprError::DimensionMismatch {
                x_dim: 2,
                expected_dim: 1
            })
        ));
    }

    #[test]
    fn loo_n_one_is_prior() {
        let noise = 0.25;
        let gpr = rbf_gpr(1.0, noise)
            .with_optimizer(Fixed)
            .factor(&[0.0], 1, 1, &[2.0])
            .expect("spd");
        let loo = gpr.loo_predict().expect("fitted");
        assert_eq!(loo.variance_kind, super::VarianceKind::Observation);
        assert_close(loo.mean[0], 0.0);
        assert_close(loo.variance[0], 1.0 + noise);
        let lat = gpr
            .loo_predict_with(super::PredictOptions {
                variance_kind: super::VarianceKind::Latent,
            })
            .expect("fitted");
        assert_close(lat.mean[0], 0.0);
        assert_close(lat.variance[0], 1.0);
    }

    #[test]
    fn loo_n_two_matches_closed_form() {
        let ell = 1.0;
        let noise = 0.25;
        let x = [0.0, 1.0];
        let y = [0.5, 1.5];
        let gpr = rbf_gpr(ell, noise)
            .with_optimizer(Fixed)
            .factor(&x, 2, 1, &y)
            .expect("spd");
        let k01 = (-0.5 / (ell * ell)).exp();
        let a = 1.0 + noise;
        let det = a * a - k01 * k01;
        let qii = a / det;
        let inv01 = -k01 / det;
        let alpha0 = qii * y[0] + inv01 * y[1];
        let alpha1 = inv01 * y[0] + qii * y[1];
        let loo = gpr.loo_predict().expect("fitted");
        assert_eq!(loo.mean.len(), 2);
        assert_eq!(loo.variance_kind, super::VarianceKind::Observation);
        assert_close(loo.mean[0], y[0] - alpha0 / qii);
        assert_close(loo.mean[1], y[1] - alpha1 / qii);
        assert_close(loo.variance[0], 1.0 / qii);
        assert_close(loo.variance[1], 1.0 / qii);
        let lat = gpr
            .loo_predict_with(super::PredictOptions {
                variance_kind: super::VarianceKind::Latent,
            })
            .expect("fitted");
        assert_close(lat.variance[0], (1.0 / qii - noise).max(0.0));
        assert_close(lat.variance[1], (1.0 / qii - noise).max(0.0));
    }

    fn omit_training_row(
        x: &[f64],
        y: &[f64],
        n: usize,
        d: usize,
        skip: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n_out = n - 1;
        let mut xo = vec![0.0; n_out * d];
        let mut yo = Vec::with_capacity(n_out);
        let mut xs = vec![0.0; d];
        let mut o = 0;
        for i in 0..n {
            if i == skip {
                for dim in 0..d {
                    xs[dim] = x[dim * n + i];
                }
                continue;
            }
            for dim in 0..d {
                xo[dim * n_out + o] = x[dim * n + i];
            }
            yo.push(y[i]);
            o += 1;
        }
        (xo, yo, xs)
    }

    #[test]
    fn loo_n_three_matches_refit_predict() {
        let ell = 1.25;
        let noise = 0.16;
        let n = 3;
        let d = 1;
        let x = [0.0, 0.5, 1.5];
        let y = [0.2, -1.0, 0.7];
        let gpr = rbf_gpr(ell, noise)
            .with_optimizer(Fixed)
            .factor(&x, n, d, &y)
            .expect("spd");
        let loo_obs = gpr.loo_predict().expect("fitted");
        let loo_lat = gpr
            .loo_predict_with(super::PredictOptions {
                variance_kind: super::VarianceKind::Latent,
            })
            .expect("fitted");
        for skip in 0..n {
            let (xo, yo, xs) = omit_training_row(&x, &y, n, d, skip);
            let held = rbf_gpr(ell, noise)
                .with_optimizer(Fixed)
                .factor(&xo, n - 1, d, &yo)
                .expect("spd");
            let pred_obs = held.predict(&xs, 1, d).expect("fitted");
            let pred_lat = held
                .predict_with(
                    &xs,
                    1,
                    d,
                    super::PredictOptions {
                        variance_kind: super::VarianceKind::Latent,
                    },
                )
                .expect("fitted");
            assert_close(loo_obs.mean[skip], pred_obs.mean[0]);
            assert_close(loo_obs.variance[skip], pred_obs.variance[0]);
            assert_close(loo_lat.mean[skip], pred_lat.mean[0]);
            assert_close(loo_lat.variance[skip], pred_lat.variance[0]);
        }
    }

    #[test]
    fn loo_observation_is_latent_plus_noise_after_inverse() {
        let noise = 0.16;
        let y = [0.0, 4.0];
        let gpr = rbf_gpr(1.0, noise)
            .with_target_transform(StandardizeTarget::new())
            .with_optimizer(Fixed)
            .factor(&[0.0, 1.0], 2, 1, &y)
            .expect("spd");
        let mut t = StandardizeTarget::new();
        t.fit(&y).expect("finite");
        let scale = t.std().expect("fitted");
        let scale_sq = scale * scale;
        let lat = gpr
            .loo_predict_with(super::PredictOptions {
                variance_kind: super::VarianceKind::Latent,
            })
            .expect("fitted");
        let obs = gpr.loo_predict().expect("fitted");
        assert_close(obs.variance[0], lat.variance[0] + scale_sq * noise);
        assert_close(obs.variance[1], lat.variance[1] + scale_sq * noise);
    }

    #[test]
    fn predict_n_one_matches_closed_form() {
        let noise = 0.25;
        let gpr = rbf_gpr(1.0, noise)
            .with_optimizer(Fixed)
            .factor(&[0.0], 1, 1, &[2.0])
            .expect("spd");
        let pred = gpr
            .predict_with(
                &[0.0],
                1,
                1,
                super::PredictOptions {
                    variance_kind: super::VarianceKind::Latent,
                },
            )
            .expect("fitted");
        let a = 1.0 + noise;
        assert_close(pred.mean[0], 2.0 / a);
        assert_close(pred.variance[0], 1.0 - 1.0 / a);
        let obs = gpr.predict(&[0.0], 1, 1).expect("fitted");
        assert_eq!(obs.variance_kind, super::VarianceKind::Observation);
        assert_close(obs.variance[0], pred.variance[0] + noise);
    }

    #[test]
    fn observation_variance_is_latent_plus_noise_after_inverse() {
        let noise = 0.16;
        let y = [0.0, 4.0];
        let gpr = rbf_gpr(1.0, noise)
            .with_target_transform(StandardizeTarget::new())
            .with_optimizer(Fixed)
            .factor(&[0.0, 1.0], 2, 1, &y)
            .expect("spd");
        let mut t = StandardizeTarget::new();
        t.fit(&y).expect("finite");
        let scale = t.std().expect("fitted");
        let scale_sq = scale * scale;
        let lat = gpr
            .predict_with(
                &[0.5],
                1,
                1,
                super::PredictOptions {
                    variance_kind: super::VarianceKind::Latent,
                },
            )
            .expect("fitted");
        let obs = gpr.predict(&[0.5], 1, 1).expect("fitted");
        assert_close(obs.variance[0], lat.variance[0] + scale_sq * noise);
        let mut recovered = y;
        t.transform(&mut recovered).expect("fitted");
        t.inverse_transform_mean(&mut recovered).expect("fitted");
        assert_close(recovered[0], y[0]);
        assert_close(recovered[1], y[1]);
    }

    #[test]
    fn ard_equal_lengthscales_match_isotropic_predict() {
        let ell = 1.25;
        let noise = 0.1;
        let x = [0.0, 0.5, 1.5, 0.0, 1.0, 0.5];
        let y = [0.2, -1.0, 0.7];
        let xs = [0.25, 1.0];
        let iso = rbf_gpr(ell, noise)
            .with_optimizer(Fixed)
            .factor(&x, 3, 2, &y)
            .expect("spd");
        let mut ard = Gpr::new(
            KernelSpec::from(RbfArdKernel::new(&[ell, ell]).expect("valid")),
            GaussianLikelihood::new(noise).expect("valid"),
        )
        .with_optimizer(Fixed)
        .factor(&x, 3, 2, &y)
        .expect("spd");
        let p_iso = iso.predict(&xs, 1, 2).expect("fitted");
        let p_ard = ard.predict(&xs, 1, 2).expect("fitted");
        assert_close(p_ard.mean[0], p_iso.mean[0]);
        assert_close(p_ard.variance[0], p_iso.variance[0]);
        let mut params = vec![0.0; ard.num_params()];
        ard.get_params(&mut params).expect("len");
        let mut grad = vec![0.0; params.len()];
        let nlml = ard
            .value_and_gradient_into(&params, &mut grad)
            .expect("spd");
        assert!(nlml.is_finite());
        assert_eq!(params.len(), 3);
        assert!(grad.iter().all(|g| g.is_finite()));
    }

    #[test]
    fn rbf_plus_white_fits() {
        let gpr = Gpr::new(
            KernelSpec::from(RbfKernel::new(1.0).expect("valid"))
                + KernelSpec::from(WhiteKernel::new(0.05).expect("valid")),
            GaussianLikelihood::new(0.1).expect("valid"),
        )
        .with_optimizer(Fixed)
        .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
        .expect("spd");
        let pred = gpr.predict(&[0.5], 1, 1).expect("fitted");
        assert!(pred.mean[0].is_finite());
        assert!(pred.variance[0] > 0.0);
    }

    #[test]
    fn linear_kernel_fits_and_predicts() {
        let x = [0.0, 1.0, 2.0];
        let y = [0.0, 1.0, 2.0];
        let mut gpr = Gpr::new(
            KernelSpec::from(LinearKernel::new(1.0).expect("valid")),
            GaussianLikelihood::new(0.1).expect("valid"),
        )
        .with_optimizer(Fixed)
        .factor(&x, 3, 1, &y)
        .expect("spd");
        let pred = gpr.predict(&[1.5], 1, 1).expect("fitted");
        assert!(pred.mean[0].is_finite());
        let mut params = vec![0.0; gpr.num_params()];
        gpr.get_params(&mut params).expect("len");
        let mut grad = vec![0.0; params.len()];
        let nlml = gpr
            .value_and_gradient_into(&params, &mut grad)
            .expect("spd");
        assert!(nlml.is_finite());
        assert!(grad.iter().all(|g| g.is_finite()));
    }

    #[test]
    fn matern_fits_and_predicts() {
        let mut gpr = Gpr::new(
            KernelSpec::from(MaternKernel::new(1.0, MaternNu::FiveHalves).expect("valid")),
            GaussianLikelihood::new(0.1).expect("valid"),
        )
        .with_optimizer(Fixed)
        .factor(&[0.0, 1.0, 2.0], 3, 1, &[0.0, 0.5, 1.0])
        .expect("spd");
        let pred = gpr.predict(&[0.5], 1, 1).expect("fitted");
        assert!(pred.mean[0].is_finite());
        assert!(pred.variance[0] > 0.0);
        let mut params = vec![0.0; gpr.num_params()];
        gpr.get_params(&mut params).expect("len");
        let mut grad = vec![0.0; params.len()];
        let nlml = gpr
            .value_and_gradient_into(&params, &mut grad)
            .expect("spd");
        assert!(nlml.is_finite());
        assert!(grad.iter().all(|g| g.is_finite()));
    }

    #[test]
    fn matern_ard_equal_lengthscales_match_isotropic() {
        let ell = 1.25;
        let noise = 0.1;
        let nu = MaternNu::ThreeHalves;
        let x = [0.0, 0.5, 1.5, 0.0, 1.0, 0.5];
        let y = [0.2, -1.0, 0.7];
        let xs = [0.25, 1.0];
        let iso = Gpr::new(
            KernelSpec::from(MaternKernel::new(ell, nu).expect("valid")),
            GaussianLikelihood::new(noise).expect("valid"),
        )
        .with_optimizer(Fixed)
        .factor(&x, 3, 2, &y)
        .expect("spd");
        let mut ard = Gpr::new(
            KernelSpec::from(MaternArdKernel::new(&[ell, ell], nu).expect("valid")),
            GaussianLikelihood::new(noise).expect("valid"),
        )
        .with_optimizer(Fixed)
        .factor(&x, 3, 2, &y)
        .expect("spd");
        let p_iso = iso.predict(&xs, 1, 2).expect("fitted");
        let p_ard = ard.predict(&xs, 1, 2).expect("fitted");
        assert_close(p_ard.mean[0], p_iso.mean[0]);
        assert_close(p_ard.variance[0], p_iso.variance[0]);
        let mut params = vec![0.0; ard.num_params()];
        ard.get_params(&mut params).expect("len");
        let mut grad = vec![0.0; params.len()];
        let nlml = ard
            .value_and_gradient_into(&params, &mut grad)
            .expect("spd");
        assert!(nlml.is_finite());
        assert_eq!(params.len(), 3);
        assert!(grad.iter().all(|g| g.is_finite()));
    }

    #[test]
    fn periodic_fits_and_predicts() {
        let mut gpr = Gpr::new(
            KernelSpec::from(PeriodicKernel::new(1.0, 2.0).expect("valid")),
            GaussianLikelihood::new(0.1).expect("valid"),
        )
        .with_optimizer(Fixed)
        .factor(&[0.0, 0.5, 1.0], 3, 1, &[0.0, 0.4, 0.1])
        .expect("spd");
        let pred = gpr.predict(&[2.0], 1, 1).expect("fitted");
        assert!(pred.mean[0].is_finite());
        assert!(pred.variance[0] > 0.0);
        let mut params = vec![0.0; gpr.num_params()];
        gpr.get_params(&mut params).expect("len");
        assert_eq!(params.len(), 3);
        let mut grad = vec![0.0; params.len()];
        let nlml = gpr
            .value_and_gradient_into(&params, &mut grad)
            .expect("spd");
        assert!(nlml.is_finite());
        assert!(grad.iter().all(|g| g.is_finite()));
    }

    #[test]
    fn rational_quadratic_fits_and_predicts() {
        let mut gpr = Gpr::new(
            KernelSpec::from(RationalQuadraticKernel::new(1.0, 1.5).expect("valid")),
            GaussianLikelihood::new(0.1).expect("valid"),
        )
        .with_optimizer(Fixed)
        .factor(&[0.0, 0.5, 1.0], 3, 1, &[0.0, 0.4, 0.1])
        .expect("spd");
        let pred = gpr.predict(&[0.25], 1, 1).expect("fitted");
        assert!(pred.mean[0].is_finite());
        assert!(pred.variance[0] > 0.0);
        let mut params = vec![0.0; gpr.num_params()];
        gpr.get_params(&mut params).expect("len");
        assert_eq!(params.len(), 3);
        let mut grad = vec![0.0; params.len()];
        let nlml = gpr
            .value_and_gradient_into(&params, &mut grad)
            .expect("spd");
        assert!(nlml.is_finite());
        assert!(grad.iter().all(|g| g.is_finite()));
    }

    #[test]
    fn rational_quadratic_ard_equal_lengthscales_match_isotropic() {
        let ell = 1.25;
        let alpha = 0.8;
        let noise = 0.1;
        let x = [0.0, 0.5, 1.5, 0.0, 1.0, 0.5];
        let y = [0.2, -1.0, 0.7];
        let xs = [0.25, 1.0];
        let iso = Gpr::new(
            KernelSpec::from(RationalQuadraticKernel::new(ell, alpha).expect("valid")),
            GaussianLikelihood::new(noise).expect("valid"),
        )
        .with_optimizer(Fixed)
        .factor(&x, 3, 2, &y)
        .expect("spd");
        let mut ard = Gpr::new(
            KernelSpec::from(RationalQuadraticArdKernel::new(&[ell, ell], alpha).expect("valid")),
            GaussianLikelihood::new(noise).expect("valid"),
        )
        .with_optimizer(Fixed)
        .factor(&x, 3, 2, &y)
        .expect("spd");
        let p_iso = iso.predict(&xs, 1, 2).expect("fitted");
        let p_ard = ard.predict(&xs, 1, 2).expect("fitted");
        assert_close(p_ard.mean[0], p_iso.mean[0]);
        assert_close(p_ard.variance[0], p_iso.variance[0]);
        let mut params = vec![0.0; ard.num_params()];
        ard.get_params(&mut params).expect("len");
        let mut grad = vec![0.0; params.len()];
        let nlml = ard
            .value_and_gradient_into(&params, &mut grad)
            .expect("spd");
        assert!(nlml.is_finite());
        assert_eq!(params.len(), 4);
        assert!(grad.iter().all(|g| g.is_finite()));
    }

    #[test]
    fn fit_optimizes_and_keeps_l_and_alpha() {
        let gpr = rbf_gpr(2.0, 0.1)
            .fit(&[0.0, 1.0], 2, 1, &[0.5, -0.25])
            .expect("lbfgs");
        let alpha = gpr.alpha().to_vec();
        assert_eq!(alpha.len(), 2);
        assert!(alpha.iter().all(|a| a.is_finite()));
        {
            let ws = &gpr.workspace;
            assert_eq!(ws.k_matrix.nrows(), 2);
        }
        let pred = gpr.predict(&[0.5], 1, 1).expect("fitted");
        assert_eq!(pred.mean.len(), 1);
        assert!(pred.mean[0].is_finite());
        assert!(
            gpr.neg_log_marginal_likelihood()
                .expect("fitted")
                .is_finite()
        );
        let a = dense_a(
            gpr.kernel(),
            gpr.likelihood().noise_variance(),
            &[0.0, 1.0],
            2,
            1,
        );
        let restored = matvec_sym(&a, &alpha);
        assert_close(restored[0], 0.5);
        assert_close(restored[1], -0.25);
    }

    #[test]
    fn fit_fixed_keeps_construction_params() {
        let gpr = rbf_gpr(1.25, 0.16)
            .with_optimizer(Fixed)
            .factor(&[0.0, 1.0], 2, 1, &[0.5, -0.25])
            .expect("spd");
        let mut params = [0.0; 2];
        gpr.get_params(&mut params).expect("len 2");
        assert_close(params[0], 1.25_f64.ln());
        assert_close(params[1], 0.16_f64.ln());
        let pred = gpr.predict(&[0.5], 1, 1).expect("fitted");
        assert_eq!(pred.mean.len(), 1);
    }

    #[test]
    fn non_finite_optimize_result_restores_theta() {
        let mut gpr = rbf_gpr(1.25, 0.16)
            .with_optimizer(Fixed)
            .factor(&[0.0, 1.0], 2, 1, &[0.5, -0.25])
            .expect("spd");
        let kernel_before = gpr.kernel().clone();
        let likelihood_before = *gpr.likelihood();
        let mut before = [0.0; 2];
        gpr.get_params(&mut before).expect("len 2");
        let moved = [2.0_f64.ln(), 0.5_f64.ln()];
        let mut grad = [0.0; 2];
        gpr.value_and_gradient_into(&moved, &mut grad)
            .expect("moved");
        let mut mid = [0.0; 2];
        gpr.get_params(&mut mid).expect("len 2");
        assert!((mid[0] - before[0]).abs() > TOL);
        let err = gpr
            .commit_or_revert_optimize(
                kernel_before,
                likelihood_before,
                Ok(OptResult {
                    params: moved.to_vec(),
                    value: f64::NAN,
                    iterations: 4,
                }),
            )
            .expect_err("nan nlml");
        assert!(matches!(
            err,
            GprError::OptimizationNotConverged { iterations: 4 }
        ));
        let mut after = [0.0; 2];
        gpr.get_params(&mut after).expect("len 2");
        assert_close(after[0], before[0]);
        assert_close(after[1], before[1]);
        assert_eq!(gpr.alpha().len(), 2);
        assert!(gpr.alpha().iter().all(|a| a.is_finite()));
    }

    #[test]
    fn failed_optimize_err_restores_theta() {
        let mut gpr = rbf_gpr(1.25, 0.16)
            .with_optimizer(Fixed)
            .factor(&[0.0, 1.0], 2, 1, &[0.5, -0.25])
            .expect("spd");
        let kernel_before = gpr.kernel().clone();
        let likelihood_before = *gpr.likelihood();
        let mut before = [0.0; 2];
        gpr.get_params(&mut before).expect("len 2");
        let moved = [2.0_f64.ln(), 0.5_f64.ln()];
        let mut grad = [0.0; 2];
        gpr.value_and_gradient_into(&moved, &mut grad)
            .expect("moved");
        gpr.commit_or_revert_optimize(
            kernel_before,
            likelihood_before,
            Err(GprError::CholeskyFailed {
                jitter: 0.0,
                matrix_size: 2,
                stage: CholeskyStage::Fit,
            }),
        )
        .expect_err("chol");
        let mut after = [0.0; 2];
        gpr.get_params(&mut after).expect("len 2");
        assert_close(after[0], before[0]);
        assert_close(after[1], before[1]);
    }

    #[test]
    fn factor_refit_keeps_construction_theta() {
        let mut gpr = rbf_gpr(1.25, 0.16)
            .with_optimizer(Fixed)
            .factor(&[0.0, 1.0], 2, 1, &[0.5, -0.25])
            .expect("spd");
        let mut before = [0.0; 2];
        gpr.get_params(&mut before).expect("len 2");
        gpr.refit().expect("refit");
        let mut after = [0.0; 2];
        gpr.get_params(&mut after).expect("len 2");
        assert_close(after[0], before[0]);
        assert_close(after[1], before[1]);
    }

    #[test]
    fn lbfgs_knobs_affect_fit_and_refit() {
        let x = [0.0, 0.25, 0.6, 1.0];
        let y = [0.1, -0.4, 0.2, 0.8];
        let frozen = rbf_gpr(2.0, 0.2)
            .with_optimizer(Lbfgs::new().with_max_iterations(0))
            .fit(&x, 4, 1, &y)
            .expect("zero iters");
        let mut frozen_params = [0.0; 2];
        frozen.get_params(&mut frozen_params).expect("len 2");
        assert_close(frozen_params[0], 2.0_f64.ln());
        assert_close(frozen_params[1], 0.2_f64.ln());

        let searched = rbf_gpr(2.0, 0.2)
            .with_optimizer(
                Lbfgs::new()
                    .with_max_iterations(80)
                    .with_history_size(std::num::NonZeroUsize::MIN)
                    .with_tolerance(1e-8)
                    .expect("tol"),
            )
            .fit(&x, 4, 1, &y)
            .expect("search");
        let mut searched_params = [0.0; 2];
        searched.get_params(&mut searched_params).expect("len 2");
        assert!(
            frozen_params
                .iter()
                .zip(&searched_params)
                .any(|(a, b)| (a - b).abs() > 1e-9),
            "a real search should move θ: frozen={frozen_params:?} searched={searched_params:?}"
        );

        let mut restarted = rbf_gpr(2.0, 0.2)
            .with_optimizer(Lbfgs::new().with_restarts(std::num::NonZeroU32::MIN, 11))
            .fit(&x, 4, 1, &y)
            .expect("restarts");
        let nlml_fit = restarted.neg_log_marginal_likelihood().expect("nlml");
        restarted.refit().expect("refit");
        let nlml_refit = restarted.neg_log_marginal_likelihood().expect("nlml");
        assert!(
            nlml_refit <= nlml_fit + 1e-9,
            "refit should not raise NLML: fit={nlml_fit}, refit={nlml_refit}"
        );
    }

    #[test]
    fn ncg_knobs_affect_fit_and_refit() {
        let x = [0.0, 0.25, 0.6, 1.0];
        let y = [0.1, -0.4, 0.2, 0.8];
        let frozen = rbf_gpr(2.0, 0.2)
            .with_optimizer(NonlinearCg::new().with_max_iterations(0))
            .fit(&x, 4, 1, &y)
            .expect("zero iters");
        let mut frozen_params = [0.0; 2];
        frozen.get_params(&mut frozen_params).expect("len 2");
        assert_close(frozen_params[0], 2.0_f64.ln());
        assert_close(frozen_params[1], 0.2_f64.ln());

        let searched = rbf_gpr(2.0, 0.2)
            .with_optimizer(
                NonlinearCg::new()
                    .with_max_iterations(80)
                    .with_tolerance(1e-8)
                    .expect("tol"),
            )
            .fit(&x, 4, 1, &y)
            .expect("search");
        let mut searched_params = [0.0; 2];
        searched.get_params(&mut searched_params).expect("len 2");
        assert!(
            frozen_params
                .iter()
                .zip(&searched_params)
                .any(|(a, b)| (a - b).abs() > 1e-9),
            "a real search should move θ: frozen={frozen_params:?} searched={searched_params:?}"
        );

        let mut restarted = rbf_gpr(2.0, 0.2)
            .with_optimizer(NonlinearCg::new().with_restarts(std::num::NonZeroU32::MIN, 11))
            .fit(&x, 4, 1, &y)
            .expect("restarts");
        let nlml_fit = restarted.neg_log_marginal_likelihood().expect("nlml");
        restarted.refit().expect("refit");
        let nlml_refit = restarted.neg_log_marginal_likelihood().expect("nlml");
        assert!(
            nlml_refit <= nlml_fit + 1e-9,
            "refit should not raise NLML: fit={nlml_fit}, refit={nlml_refit}"
        );
    }

    #[test]
    fn neldermead_fit_lowers_nlml() {
        let x = [0.0, 0.25, 0.6, 1.0];
        let y = [0.1, -0.4, 0.2, 0.8];
        let at_init = rbf_gpr(2.0, 0.2)
            .with_optimizer(Fixed)
            .factor(&x, 4, 1, &y)
            .expect("spd");
        let nlml_init = at_init.neg_log_marginal_likelihood().expect("init");
        let mut fitted = rbf_gpr(2.0, 0.2)
            .with_optimizer(NelderMead::new().with_max_iterations(80))
            .fit(&x, 4, 1, &y)
            .expect("nm");
        let nlml_fit = fitted.neg_log_marginal_likelihood().expect("nlml");
        assert!(
            nlml_fit < nlml_init,
            "NLML should fall: init={nlml_init}, fit={nlml_fit}"
        );
        fitted.refit().expect("refit");
        let nlml_refit = fitted.neg_log_marginal_likelihood().expect("nlml");
        assert!(
            nlml_refit <= nlml_fit + 1e-9,
            "refit should not raise NLML: fit={nlml_fit}, refit={nlml_refit}"
        );
    }

    #[derive(Clone, Copy, Debug)]
    struct IndexUsingOpt;

    impl UsesChangeIndices for IndexUsingOpt {}

    impl<P: Objective> Optimizer<P> for IndexUsingOpt {
        fn minimize(&self, objective: &mut P, init: &[f64]) -> Result<OptResult, GprError> {
            let value = objective.value(init)?;
            Ok(OptResult {
                params: init.to_vec(),
                value,
                iterations: 0,
            })
        }
    }

    #[test]
    fn incremental_recompute_is_gated_on_uses_change_indices() {
        let fitted = rbf_gpr(1.0, 0.1)
            .with_optimizer(IndexUsingOpt)
            .with_recompute_strategy(IncrementalRecompute)
            .fit(&[0.0, 1.0], 2, 1, &[0.5, -0.25])
            .expect("fit");
        assert_eq!(fitted.n(), 2);
        let _full = rbf_gpr(1.0, 0.1)
            .with_recompute_strategy(FullRecompute)
            .fit(&[0.0, 1.0], 2, 1, &[0.5, -0.25])
            .expect("fit");
    }
}
