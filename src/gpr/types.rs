//! Public prediction, cache, and jitter types for [`crate::Gpr`].

use std::fmt;

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

/// Selects whether training distances are reused across kernel builds.
///
/// Isotropic RBF, Matérn, Periodic, and RQ evaluate from an `n×n` squared
/// Euclidean matrix. ARD RBF / Matérn / RQ evaluate from raw `(Δx_d)²` stored
/// as `n × (n·d)`. [`Self::Always`] fills the matching tensor once per fit.
/// [`Self::Never`] recomputes it on every kernel build.
///
/// This type is the cache slot on trainers from [`crate::Gpr::new`].
/// Standalone Linear, Constant, and White trainers use
/// [`crate::Gpr::from_points`] and have no
/// [`crate::Gpr::with_distance_cache_policy`].
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

/// Maps a trainer cache slot to the policy used when factorizing `A`.
pub(crate) trait DistanceCacheSlot:
    Copy + Clone + fmt::Debug + Default + Eq + PartialEq + Send + Sync + 'static
{
    fn policy(self) -> DistanceCachePolicy;

    fn persist(self) -> Option<DistanceCachePolicy>;
}

impl DistanceCacheSlot for DistanceCachePolicy {
    fn policy(self) -> DistanceCachePolicy {
        self
    }

    fn persist(self) -> Option<DistanceCachePolicy> {
        Some(self)
    }
}

/// Marks a trainer that does not store a [`DistanceCachePolicy`].
///
/// [`crate::Gpr::from_points`] builds this slot for a standalone Linear,
/// Constant, or White kernel. Distance kernels keep [`DistanceCachePolicy`]
/// on [`crate::Gpr::new`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NoDistanceCache;

impl DistanceCacheSlot for NoDistanceCache {
    fn policy(self) -> DistanceCachePolicy {
        DistanceCachePolicy::Never
    }

    fn persist(self) -> Option<DistanceCachePolicy> {
        None
    }
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
/// [`FittedGpr::predict_into`] reuses `mean` / `variance` capacity when the
/// query length matches a previous call. Query–query covariance is
/// [`PredictiveCovariance`], not a field here.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Prediction {
    /// Predictive mean on the original target scale.
    pub mean: Vec<f64>,
    /// Predictive variance on the original target scale.
    pub variance: Vec<f64>,
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
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PredictiveCovariance {
    /// Predictive mean on the original target scale.
    pub mean: Vec<f64>,
    /// Predictive covariance, packed column-major `m × m`.
    pub covariance: Vec<f64>,
    /// Whether the diagonal of [`Self::covariance`] is latent or observation.
    pub variance_kind: VarianceKind,
}
