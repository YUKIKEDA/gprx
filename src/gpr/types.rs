//! Public prediction, cache, and jitter types for [`crate::Gpr`].

use crate::error::GprError;

/// Which predictive variance [`Prediction`] reports.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum VarianceKind {
    /// Variance of the latent function `f*`, without observation noise.
    Latent,
    /// Variance of a new observation `y*`, including `σn²`. This is the default.
    #[default]
    Observation,
}

/// Options for [`crate::FittedGpr::predict`], [`crate::FittedGpr::predict_covariance`],
/// and [`crate::FittedGpr::sample`].
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

/// Whether [`crate::Gpr`] caches training distances between kernel builds.
///
/// [`Self::Cached`] (the default) fills the distances once per fit and reuses
/// them while `X` is unchanged: isotropic fits store an `n×n`
/// squared-Euclidean matrix, ARD fits also store raw `(Δx_d)²` as
/// `n × (n·d)`. [`Self::Uncached`] recomputes them from `X` on every kernel
/// build. A kernel that does not read distances (standalone Linear,
/// Constant, White) never allocates the cache.
///
/// # Examples
///
/// ```rust
/// use gprx::kernel::{KernelSpec, RbfKernel};
/// use gprx::{DistanceCachePolicy, GaussianLikelihood, Gpr};
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
/// let gpr = Gpr::new(kernel, GaussianLikelihood::new(0.1)?)
///     .with_distance_cache_policy(DistanceCachePolicy::Uncached);
/// assert_eq!(gpr.distance_cache_policy(), DistanceCachePolicy::Uncached);
/// let _fitted = gpr.fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0]).map_err(|(_, e)| e)?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum DistanceCachePolicy {
    /// Fill once per fit and reuse (the speed pole).
    #[default]
    Cached,
    /// Recompute from `X` on every kernel build (the memory pole).
    Uncached,
}

/// Where the gradient matrix `W = ααᵀ - K⁻¹` lives during a fit.
///
/// [`Self::Retain`] (the default) keeps a dedicated `n×n` `W`, so the
/// Cholesky factor stays in place and optimizers that report changed
/// parameters rebuild only the touched kernel leaves. [`Self::Reuse`]
/// overwrites the factor with `W` and refactors afterwards; it saves one
/// `n×n` matrix and always rebuilds the full kernel.
///
/// # Examples
///
/// ```rust
/// use gprx::kernel::{KernelSpec, RbfKernel};
/// use gprx::{CholeskyBuffer, GaussianLikelihood, Gpr};
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
/// let gpr = Gpr::new(kernel, GaussianLikelihood::new(0.1)?)
///     .with_cholesky_buffer(CholeskyBuffer::Reuse);
/// assert_eq!(gpr.cholesky_buffer(), CholeskyBuffer::Reuse);
/// let _fitted = gpr.fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0]).map_err(|(_, e)| e)?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CholeskyBuffer {
    /// Keep a dedicated `W` (the speed pole).
    #[default]
    Retain,
    /// Overwrite the factor with `W`, then refactor (the memory pole).
    Reuse,
}

/// Which kernel `exp` [`crate::Gpr`] evaluates with.
///
/// [`Self::Accurate`] (the default) is libm / SIMD `exp`
/// ([`crate::Accurate`]). [`Self::FastApprox`] is the polynomial
/// ([`crate::FastApprox`]); fit and predict use the same one.
/// Hyperparameter `exp(θ)` is unchanged.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum KernelExp {
    /// Exact libm / SIMD `exp`.
    #[default]
    Accurate,
    /// Degree-7 polynomial `exp`.
    FastApprox,
}

/// Runs `$body` with the type alias `$M` bound to the [`crate::KernelMath`]
/// type of `$mode`. The body is monomorphized once per mode, as a generic
/// `M` parameter was before.
macro_rules! with_kernel_exp {
    ($mode:expr, $M:ident => $body:expr) => {
        match $mode {
            $crate::gpr::KernelExp::Accurate => {
                type $M = $crate::math::Accurate;
                $body
            }
            $crate::gpr::KernelExp::FastApprox => {
                type $M = $crate::math::FastApprox;
                $body
            }
        }
    };
}
pub(crate) use with_kernel_exp;

/// Numerical Cholesky stabilizer, distinct from observation noise.
///
/// The first factorization always tries `A = K + σn² I` with no extra
/// diagonal. [`Self::Fixed`] retries once with that `j` if the first factor
/// fails. [`Self::Adaptive`] retries with `initial`, then
/// `initial * multiplier` on each later attempt, stopping at `max_retries`
/// or when `j` would exceed `max_jitter`. A successful retry factors
/// `A + j I`; this crate does not iteratively refine back to `A`.
/// Observation noise stays on [`GaussianLikelihood`](crate::GaussianLikelihood).
///
/// The default is [`Self::fixed`]`(0.0)`: no retry, matching an unregularized
/// factor. [`GprError::CholeskyFailed::jitter`] is the last `j` that was
/// tried (`0.0` when the unregularized factor was the only attempt).
///
/// # Errors
///
/// Constructors return [`GprError::InvalidConfig`] if a value is
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
    /// Returns [`GprError::InvalidConfig`] if `jitter` is not finite
    /// or is negative.
    pub fn fixed(jitter: f64) -> Result<Self, GprError> {
        if !jitter.is_finite() || jitter < 0.0 {
            return Err(GprError::InvalidConfig {
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
    /// Returns [`GprError::InvalidConfig`] if a value is non-finite
    /// or outside that domain.
    pub fn adaptive(
        initial: f64,
        multiplier: f64,
        max_retries: usize,
        max_jitter: f64,
    ) -> Result<Self, GprError> {
        if !initial.is_finite() || initial <= 0.0 {
            return Err(GprError::InvalidConfig {
                reason: format!(
                    "adaptive initial jitter must be finite and positive, got {initial}"
                ),
            });
        }
        if !multiplier.is_finite() || multiplier <= 1.0 {
            return Err(GprError::InvalidConfig {
                reason: format!(
                    "adaptive jitter multiplier must be finite and greater than 1, got {multiplier}"
                ),
            });
        }
        if max_retries < 1 {
            return Err(GprError::InvalidConfig {
                reason: "adaptive jitter max_retries must be at least 1".to_owned(),
            });
        }
        if !max_jitter.is_finite() || max_jitter < initial {
            return Err(GprError::InvalidConfig {
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

    pub(crate) fn retry_jitters(self) -> RetryJitters {
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

pub(crate) struct RetryJitters {
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
/// [`FittedGpr::predict_into`](crate::FittedGpr::predict_into) reuses `mean` / `variance` capacity when the
/// query length matches a previous call. Query–query covariance is
/// [`PredictiveCovariance`], not a field here.
#[derive(Clone, Debug, PartialEq)]
pub struct Prediction<T = f64> {
    /// Predictive mean on the original target scale.
    pub mean: Vec<T>,
    /// Predictive variance on the original target scale.
    pub variance: Vec<T>,
    /// Whether [`Self::variance`] is latent or observation variance.
    pub variance_kind: VarianceKind,
}

/// Predictive mean and query–query covariance at the query points.
///
/// [`crate::FittedGpr::predict_covariance`] returns this type. `covariance` is
/// packed column-major as an `m × m` matrix: index `col * m + row`. The
/// diagonal matches [`Prediction::variance`] for the same query and
/// [`PredictOptions`]. Observation covariance adds `σn²` on the diagonal in
/// the transformed space, then the target transform scales every entry by
/// the same `s²` as variance.
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
/// assert_eq!(cov.covariance.len(), cov.mean.len() * cov.mean.len());
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct PredictiveCovariance<T = f64> {
    /// Predictive mean on the original target scale.
    pub mean: Vec<T>,
    /// Predictive covariance, packed column-major `m × m`.
    pub covariance: Vec<T>,
    /// Whether the diagonal of [`Self::covariance`] is latent or observation.
    pub variance_kind: VarianceKind,
}

impl<T> Default for Prediction<T> {
    fn default() -> Self {
        Self {
            mean: Vec::new(),
            variance: Vec::new(),
            variance_kind: VarianceKind::default(),
        }
    }
}

impl<T> Default for PredictiveCovariance<T> {
    fn default() -> Self {
        Self {
            mean: Vec::new(),
            covariance: Vec::new(),
            variance_kind: VarianceKind::default(),
        }
    }
}

/// Stable identity of one training point on [`crate::OnlineGpr`].
///
/// [`crate::FittedGpr::into_online`] assigns identifiers `0 .. n-1` in buffer
/// order. Later [`crate::OnlineGpr::insert`] values increase monotonically and
/// are never reused after [`crate::OnlineGpr::delete`]. There is no public
/// constructor.
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
/// let id = online.insert(&[1.5], 0.5)?;
/// assert_eq!(online.point_ids().last().copied(), Some(id));
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PointId(u64);

impl PointId {
    pub(crate) fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    pub(crate) fn raw(self) -> u64 {
        self.0
    }
}
