//! Open intervals and bounded hyperparameters in user units.

use thiserror::Error;

use crate::error::GprError;
use crate::kernel::KernelSpec;
use crate::likelihood::GaussianLikelihood;

/// Why an [`Interval`] or [`BoundedParam`] could not be constructed.
///
/// Invalid bounds are not [`crate::GprError::InvalidHyperparameter`]. That
/// variant stays for data (wrong slice length, `NaN` in `X`).
#[derive(Clone, Copy, Debug, Error, PartialEq)]
pub enum IntervalError {
    /// `lo` or `hi` is non-finite, or `lo >= hi`.
    #[error("interval bounds must be finite and satisfy lo < hi (got lo={lo}, hi={hi})")]
    InvalidBounds {
        /// Requested lower endpoint.
        lo: f64,
        /// Requested upper endpoint.
        hi: f64,
    },
    /// `value` is non-finite or not strictly inside `(lo, hi)`.
    #[error("value {value} is not strictly inside ({lo}, {hi})")]
    OutOfRange {
        /// Requested parameter in user units.
        value: f64,
        /// Lower endpoint of the open interval.
        lo: f64,
        /// Upper endpoint of the open interval.
        hi: f64,
    },
}

/// Finite open interval `(lo, hi)` with `lo < hi`.
///
/// Fields are private so `±inf` and `lo >= hi` cannot be stored. Values live
/// in user units (`ℓ`, `σn²`, constant `c`, …), not optimizer `log` space.
///
/// # Examples
///
/// ```rust
/// use gprx::Interval;
///
/// # fn main() -> Result<(), gprx::IntervalError> {
/// let interval = Interval::new(1e-5, 1e5)?;
/// assert!(interval.lo() < interval.hi());
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Interval {
    lo: f64,
    hi: f64,
}

impl Interval {
    /// Default open interval `(1e-5, 1e5)` for positive kernel and likelihood
    /// parameters (`ℓ`, `σn²`, constant `c`, …).
    pub const DEFAULT_POSITIVE: Self = Self { lo: 1e-5, hi: 1e5 };

    /// Builds a finite open interval.
    ///
    /// # Errors
    ///
    /// Returns [`IntervalError::InvalidBounds`] if a bound is non-finite or
    /// `lo >= hi`.
    pub fn new(lo: f64, hi: f64) -> Result<Self, IntervalError> {
        if lo.is_finite() && hi.is_finite() && lo < hi {
            Ok(Self { lo, hi })
        } else {
            Err(IntervalError::InvalidBounds { lo, hi })
        }
    }

    /// Returns the exclusive lower endpoint.
    pub fn lo(self) -> f64 {
        self.lo
    }

    /// Returns the exclusive upper endpoint.
    pub fn hi(self) -> f64 {
        self.hi
    }

    /// Returns whether `value` is finite and strictly inside `(lo, hi)`.
    pub fn contains(self, value: f64) -> bool {
        value.is_finite() && value > self.lo && value < self.hi
    }

    pub(crate) fn width(self) -> f64 {
        self.hi - self.lo
    }
}

/// User-unit parameter that cannot sit on or outside its [`Interval`].
///
/// Leaves and [`crate::GaussianLikelihood`] store this instead of a bare `f64`
/// plus a separate bounds field. [`Self::new`] is the only constructor.
///
/// # Examples
///
/// ```rust
/// use gprx::{BoundedParam, Interval};
///
/// # fn main() -> Result<(), gprx::IntervalError> {
/// let param = BoundedParam::new(1.0, Interval::DEFAULT_POSITIVE)?;
/// assert!((param.value() - 1.0).abs() < 1e-15);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundedParam {
    value: f64,
    interval: Interval,
}

impl BoundedParam {
    /// Builds a parameter strictly inside `interval`.
    ///
    /// # Errors
    ///
    /// Returns [`IntervalError::OutOfRange`] if `value` is non-finite or not
    /// strictly inside `(lo, hi)`.
    pub fn new(value: f64, interval: Interval) -> Result<Self, IntervalError> {
        if interval.contains(value) {
            Ok(Self { value, interval })
        } else {
            Err(IntervalError::OutOfRange {
                value,
                lo: interval.lo(),
                hi: interval.hi(),
            })
        }
    }

    /// Builds a parameter on [`Interval::DEFAULT_POSITIVE`].
    ///
    /// # Errors
    ///
    /// Same as [`Self::new`].
    pub fn default_positive(value: f64) -> Result<Self, IntervalError> {
        Self::new(value, Interval::DEFAULT_POSITIVE)
    }

    /// Returns the value in user units.
    pub fn value(self) -> f64 {
        self.value
    }

    /// Returns the open interval this value belongs to.
    pub fn interval(self) -> Interval {
        self.interval
    }

    /// Returns `log(value)` for the concatenated optimizer `θ`.
    pub fn ln(self) -> f64 {
        self.value.ln()
    }

    /// Rebuilds this parameter with the same interval.
    ///
    /// # Errors
    ///
    /// Same as [`Self::new`].
    pub fn with_value(self, value: f64) -> Result<Self, IntervalError> {
        Self::new(value, self.interval)
    }

    /// Rebuilds this parameter with the same value and a new interval.
    ///
    /// # Errors
    ///
    /// Same as [`Self::new`].
    pub fn with_interval(self, interval: Interval) -> Result<Self, IntervalError> {
        Self::new(self.value, interval)
    }
}

/// Writes kernel `θ` then likelihood `θ` into `out`, the layout every model's
/// optimizer sees.
pub(crate) fn write_params(
    kernel: &KernelSpec,
    likelihood: &GaussianLikelihood,
    out: &mut [f64],
) -> Result<(), GprError> {
    let n_kernel = kernel.num_params();
    crate::data::require_count(out.len(), n_kernel + likelihood.num_params(), "parameters")?;
    kernel.get_params(&mut out[..n_kernel])?;
    likelihood.get_params(&mut out[n_kernel..])
}

#[cfg(test)]
mod tests {
    use super::{BoundedParam, Interval, IntervalError};

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn is_send_sync() {
        assert_send_sync::<Interval>();
        assert_send_sync::<BoundedParam>();
        assert_send_sync::<IntervalError>();
    }

    #[test]
    fn new_rejects_non_finite_and_unordered() {
        assert!(matches!(
            Interval::new(f64::NAN, 1.0),
            Err(IntervalError::InvalidBounds { .. })
        ));
        assert!(matches!(
            Interval::new(0.0, f64::INFINITY),
            Err(IntervalError::InvalidBounds { .. })
        ));
        assert!(matches!(
            Interval::new(1.0, 1.0),
            Err(IntervalError::InvalidBounds { .. })
        ));
        assert!(matches!(
            Interval::new(2.0, 1.0),
            Err(IntervalError::InvalidBounds { .. })
        ));
    }

    #[test]
    fn bounded_param_rejects_endpoint_and_outside() {
        let interval = Interval::DEFAULT_POSITIVE;
        assert!(matches!(
            BoundedParam::new(interval.lo(), interval),
            Err(IntervalError::OutOfRange { .. })
        ));
        assert!(matches!(
            BoundedParam::new(interval.hi(), interval),
            Err(IntervalError::OutOfRange { .. })
        ));
        assert!(matches!(
            BoundedParam::new(1e6, interval),
            Err(IntervalError::OutOfRange { .. })
        ));
        let param = BoundedParam::default_positive(1.0).expect("inside");
        assert!((param.value() - 1.0).abs() < 1e-15);
        assert!((param.ln() - 1.0_f64.ln()).abs() < 1e-15);
    }
}
