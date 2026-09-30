//! Target (`y`) transforms: identity, standardize, and min-max.

use std::any::Any;

use super::population_std;
use crate::error::GprError;

/// Unfitted target map. [`Self::fit`] consumes it and returns a
/// [`TargetTransform`].
pub trait UnfittedTarget: Send + Sync {
    /// Estimates transform parameters from training targets.
    ///
    /// # Errors
    ///
    /// Returns [`GprError`] when `y` is empty, non-finite, or otherwise invalid
    /// for this transform.
    fn fit(self: Box<Self>, y: &[f64]) -> Result<Box<dyn TargetTransform>, GprError>;

    /// Clones this map into a new box. Used by [`crate::Gpr`] clone.
    fn clone_box(&self) -> Box<dyn UnfittedTarget>;

    /// Downcast handle used when encoding a built-in map for persist.
    fn as_any(&self) -> &dyn Any;

    /// Registry key for a caller-defined map. Built-ins return [`None`].
    fn persist_id(&self) -> Option<&'static str> {
        None
    }

    /// JSON state paired with [`Self::persist_id`].
    ///
    /// # Errors
    ///
    /// Returns [`GprError::PersistFailed`] when this map has no persist form.
    fn persist_state(&self) -> Result<serde_json::Value, GprError> {
        Err(GprError::PersistFailed {
            reason: "this target transform does not implement persist_state".to_owned(),
        })
    }
}

/// Fitted target map: forward and inverse maps exist only here.
///
/// [`StandardizeTarget`] is the usual unfitted choice when the mean function
/// is zero. Variance undoes `y' = (y - μ) / s` as `Var(y) = s² Var(y')`.
/// Covariance uses the same `s²` on every entry.
pub trait TargetTransform: Send + Sync {
    /// Applies the forward map to targets in place.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::NonFiniteInput`] when `y` contains `NaN` or `Inf`.
    fn transform(&self, y: &mut [f64]) -> Result<(), GprError>;

    /// Maps latent or observation means from transformed space to `y` scale.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::NonFiniteInput`] when `mean` contains `NaN` or `Inf`.
    fn inverse_transform_mean(&self, mean: &mut [f64]) -> Result<(), GprError>;

    /// Maps predictive variances from transformed space to `y` scale.
    ///
    /// For the affine map `y' = (y - μ) / s`, each entry is multiplied by `s²`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::NonFiniteInput`] when `var` contains `NaN` or `Inf`.
    fn inverse_transform_variance(&self, var: &mut [f64]) -> Result<(), GprError>;

    /// Maps a packed query–query covariance from transformed space to `y` scale.
    ///
    /// Affine maps multiply every entry by the same `s²` as
    /// [`Self::inverse_transform_variance`]. The default implementation is
    /// that scaling.
    ///
    /// # Errors
    ///
    /// Same as [`Self::inverse_transform_variance`].
    fn inverse_transform_covariance(&self, cov: &mut [f64]) -> Result<(), GprError> {
        self.inverse_transform_variance(cov)
    }

    /// Clones this map into a new box. Used by [`crate::FittedGpr`] clone.
    fn clone_box(&self) -> Box<dyn TargetTransform>;

    /// Downcast handle used when encoding a built-in map for persist.
    fn as_any(&self) -> &dyn Any;

    /// Registry key for a caller-defined map. Built-ins return [`None`].
    fn persist_id(&self) -> Option<&'static str> {
        None
    }

    /// JSON state paired with [`Self::persist_id`].
    ///
    /// # Errors
    ///
    /// Returns [`GprError::PersistFailed`] when this map has no persist form.
    fn persist_state(&self) -> Result<serde_json::Value, GprError> {
        Err(GprError::PersistFailed {
            reason: "this target transform does not implement persist_state".to_owned(),
        })
    }
}

/// Leaves targets and predictions unchanged.
///
/// Identity has no unfitted state. [`Self::fit`] only checks that `y` is
/// finite.
///
/// # Examples
///
/// ```rust
/// use gprx::transform::{IdentityTarget, TargetTransform};
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let t = IdentityTarget.fit(&[1.0, 2.0])?;
/// let mut mean = [0.5, 1.5];
/// t.inverse_transform_mean(&mut mean)?;
/// # let _ = mean;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct IdentityTarget;

impl IdentityTarget {
    /// Checks that `y` is finite and returns this map.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::NonFiniteInput`] when `y` contains `NaN` or `Inf`.
    pub fn fit(self, y: &[f64]) -> Result<Self, GprError> {
        crate::data::require_finite(y)?;
        Ok(self)
    }
}

impl UnfittedTarget for IdentityTarget {
    fn fit(self: Box<Self>, y: &[f64]) -> Result<Box<dyn TargetTransform>, GprError> {
        (*self).fit(y).map(|t| Box::new(t) as _)
    }

    fn clone_box(&self) -> Box<dyn UnfittedTarget> {
        Box::new(*self)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl TargetTransform for IdentityTarget {
    fn transform(&self, y: &mut [f64]) -> Result<(), GprError> {
        crate::data::require_finite(y)
    }

    fn inverse_transform_mean(&self, mean: &mut [f64]) -> Result<(), GprError> {
        crate::data::require_finite(mean)
    }

    fn inverse_transform_variance(&self, var: &mut [f64]) -> Result<(), GprError> {
        crate::data::require_finite(var)
    }

    fn clone_box(&self) -> Box<dyn TargetTransform> {
        Box::new(*self)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Unfitted center-and-scale map. [`Self::fit`] returns
/// [`FittedStandardizeTarget`].
///
/// # Examples
///
/// ```rust
/// use gprx::transform::{StandardizeTarget, TargetTransform};
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let t = StandardizeTarget::new().fit(&[1.0, 3.0, 5.0])?;
/// let mut y = [1.0, 3.0, 5.0];
/// t.transform(&mut y)?;
/// t.inverse_transform_mean(&mut y)?;
/// # let _ = y;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct StandardizeTarget;

impl StandardizeTarget {
    /// Returns an unfitted transform.
    pub fn new() -> Self {
        Self
    }

    /// Estimates `μ` and `s` and returns the fitted map.
    ///
    /// `s` is the population standard deviation. A constant `y` uses `s = 1`
    /// so centering stays well-defined and the inverse is a shift by the mean.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::EmptyInput`] when `y` is empty, or
    /// [`GprError::NonFiniteInput`] when `y` contains `NaN` or `Inf`.
    pub fn fit(self, y: &[f64]) -> Result<FittedStandardizeTarget, GprError> {
        crate::data::require_nonempty(y.len())?;
        crate::data::require_finite(y)?;
        let n = y.len() as f64;
        let mean = y.iter().sum::<f64>() / n;
        Ok(FittedStandardizeTarget {
            mean,
            std: population_std(y, mean),
        })
    }
}

impl UnfittedTarget for StandardizeTarget {
    fn fit(self: Box<Self>, y: &[f64]) -> Result<Box<dyn TargetTransform>, GprError> {
        (*self).fit(y).map(|t| Box::new(t) as _)
    }

    fn clone_box(&self) -> Box<dyn UnfittedTarget> {
        Box::new(*self)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Fitted center-and-scale map with training `μ` and `s`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FittedStandardizeTarget {
    mean: f64,
    std: f64,
}

impl FittedStandardizeTarget {
    /// Returns the training mean.
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Returns the training scale `s`.
    pub fn std(&self) -> f64 {
        self.std
    }

    pub(crate) fn from_parts(mean: f64, std: f64) -> Result<Self, GprError> {
        if !mean.is_finite() || !std.is_finite() || std <= 0.0 {
            return Err(GprError::InvalidHyperparameter {
                reason: format!(
                    "fitted standardize target stats must be finite with positive scale, got mean={mean}, std={std}"
                ),
            });
        }
        Ok(Self { mean, std })
    }
}

impl TargetTransform for FittedStandardizeTarget {
    fn transform(&self, y: &mut [f64]) -> Result<(), GprError> {
        crate::data::require_finite(y)?;
        let mean = self.mean;
        let std = self.std;
        for value in y {
            *value = (*value - mean) / std;
        }
        Ok(())
    }

    fn inverse_transform_mean(&self, mean: &mut [f64]) -> Result<(), GprError> {
        crate::data::require_finite(mean)?;
        let loc = self.mean;
        let std = self.std;
        for value in mean {
            *value = loc + std * *value;
        }
        Ok(())
    }

    fn inverse_transform_variance(&self, var: &mut [f64]) -> Result<(), GprError> {
        crate::data::require_finite(var)?;
        let scale = self.std * self.std;
        for value in var {
            *value *= scale;
        }
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn TargetTransform> {
        Box::new(*self)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Unfitted min-max map onto a closed interval, default `[0, 1]`.
///
/// [`Self::fit`] returns [`FittedMinMaxTarget`].
///
/// # Examples
///
/// ```rust
/// use gprx::transform::{MinMaxTarget, TargetTransform};
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let t = MinMaxTarget::new().fit(&[1.0, 3.0, 5.0])?;
/// let mut y = [1.0, 3.0, 5.0];
/// t.transform(&mut y)?;
/// t.inverse_transform_mean(&mut y)?;
/// # let _ = y;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MinMaxTarget {
    range_lo: f64,
    range_hi: f64,
}

impl MinMaxTarget {
    /// Returns an unfitted map onto `[0, 1]`.
    pub fn new() -> Self {
        Self {
            range_lo: 0.0,
            range_hi: 1.0,
        }
    }

    /// Returns an unfitted map onto `[lo, hi]`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidConfig`] if `lo` or `hi` is not
    /// finite, or if `hi <= lo`.
    pub fn with_feature_range(lo: f64, hi: f64) -> Result<Self, GprError> {
        require_feature_range(lo, hi)?;
        Ok(Self {
            range_lo: lo,
            range_hi: hi,
        })
    }

    /// Returns the output interval `[lo, hi]`.
    pub fn feature_range(&self) -> (f64, f64) {
        (self.range_lo, self.range_hi)
    }

    /// Estimates training min / max and returns the fitted map.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::EmptyInput`] when `y` is empty, or
    /// [`GprError::NonFiniteInput`] when `y` contains `NaN` or `Inf`.
    pub fn fit(self, y: &[f64]) -> Result<FittedMinMaxTarget, GprError> {
        crate::data::require_nonempty(y.len())?;
        crate::data::require_finite(y)?;
        let mut min = y[0];
        let mut max = y[0];
        for &value in &y[1..] {
            if value < min {
                min = value;
            }
            if value > max {
                max = value;
            }
        }
        Ok(FittedMinMaxTarget {
            data_min: min,
            data_max: max,
            range_lo: self.range_lo,
            range_hi: self.range_hi,
        })
    }
}

impl Default for MinMaxTarget {
    fn default() -> Self {
        Self::new()
    }
}

impl UnfittedTarget for MinMaxTarget {
    fn fit(self: Box<Self>, y: &[f64]) -> Result<Box<dyn TargetTransform>, GprError> {
        (*self).fit(y).map(|t| Box::new(t) as _)
    }

    fn clone_box(&self) -> Box<dyn UnfittedTarget> {
        Box::new(*self)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Fitted min-max map with training extrema and the output interval.
///
/// `y' = lo + (hi - lo) * (y - min) / (max - min)`. A constant `y` uses
/// denominator `1`, so every entry maps to `lo` and the inverse is a shift
/// by `min`. Variance undoes the affine map as `Var(y) = s² Var(y')` with
/// `s = (max - min) / (hi - lo)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FittedMinMaxTarget {
    data_min: f64,
    data_max: f64,
    range_lo: f64,
    range_hi: f64,
}

impl FittedMinMaxTarget {
    /// Returns the output interval `[lo, hi]`.
    pub fn feature_range(&self) -> (f64, f64) {
        (self.range_lo, self.range_hi)
    }

    /// Returns the training minimum.
    pub fn min(&self) -> f64 {
        self.data_min
    }

    /// Returns the training maximum.
    pub fn max(&self) -> f64 {
        self.data_max
    }

    fn data_span(&self) -> f64 {
        let span = self.data_max - self.data_min;
        if span > 0.0 && span.is_finite() {
            span
        } else {
            1.0
        }
    }

    fn out_span(&self) -> f64 {
        self.range_hi - self.range_lo
    }

    pub(crate) fn from_parts(
        data_min: f64,
        data_max: f64,
        range_lo: f64,
        range_hi: f64,
    ) -> Result<Self, GprError> {
        require_feature_range(range_lo, range_hi)?;
        if !data_min.is_finite() || !data_max.is_finite() {
            return Err(GprError::NonFiniteInput);
        }
        Ok(Self {
            data_min,
            data_max,
            range_lo,
            range_hi,
        })
    }
}

impl TargetTransform for FittedMinMaxTarget {
    fn transform(&self, y: &mut [f64]) -> Result<(), GprError> {
        crate::data::require_finite(y)?;
        let min = self.data_min;
        let span = self.data_span();
        let lo = self.range_lo;
        let out_span = self.out_span();
        for value in y {
            *value = lo + out_span * (*value - min) / span;
        }
        Ok(())
    }

    fn inverse_transform_mean(&self, mean: &mut [f64]) -> Result<(), GprError> {
        crate::data::require_finite(mean)?;
        let min = self.data_min;
        let span = self.data_span();
        let lo = self.range_lo;
        let out_span = self.out_span();
        for value in mean {
            *value = min + span * (*value - lo) / out_span;
        }
        Ok(())
    }

    fn inverse_transform_variance(&self, var: &mut [f64]) -> Result<(), GprError> {
        crate::data::require_finite(var)?;
        let scale = self.data_span() / self.out_span();
        let scale2 = scale * scale;
        for value in var {
            *value *= scale2;
        }
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn TargetTransform> {
        Box::new(*self)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn require_feature_range(lo: f64, hi: f64) -> Result<(), GprError> {
    if !lo.is_finite() || !hi.is_finite() || hi <= lo {
        return Err(GprError::InvalidConfig {
            reason: format!("feature range must satisfy lo < hi and both finite, got [{lo}, {hi}]"),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        FittedMinMaxTarget, FittedStandardizeTarget, IdentityTarget, MinMaxTarget,
        StandardizeTarget, TargetTransform,
    };
    use crate::error::GprError;

    const TOL: f64 = 1e-10;

    use crate::test_check::{assert_close, assert_send_sync};

    #[test]
    fn is_send_sync() {
        assert_send_sync::<IdentityTarget>();
        assert_send_sync::<StandardizeTarget>();
        assert_send_sync::<FittedStandardizeTarget>();
        assert_send_sync::<MinMaxTarget>();
        assert_send_sync::<FittedMinMaxTarget>();
    }

    #[test]
    fn identity_is_a_no_op() {
        let t = IdentityTarget.fit(&[1.0, 2.0]).expect("finite");
        let mut y = [1.0, 2.0];
        t.transform(&mut y).expect("finite");
        assert_close(y[0], 1.0, TOL);
        assert_close(y[1], 2.0, TOL);
        t.inverse_transform_mean(&mut y).expect("finite");
        t.inverse_transform_variance(&mut y).expect("finite");
        assert_close(y[0], 1.0, TOL);
        assert_close(y[1], 2.0, TOL);
    }

    #[test]
    fn standardize_centers_and_unit_scales() {
        let y = [1.0, 3.0, 5.0];
        let t = StandardizeTarget::new().fit(&y).expect("valid");
        assert_close(t.mean(), 3.0, TOL);
        assert_close(t.std(), (8.0 / 3.0_f64).sqrt(), TOL);
        let mut z = y;
        t.transform(&mut z).expect("finite");
        let z_mean = z.iter().sum::<f64>() / 3.0;
        assert_close(z_mean, 0.0, TOL);
        let z_var = z.iter().map(|v| v * v).sum::<f64>() / 3.0;
        assert_close(z_var, 1.0, TOL);
    }

    #[test]
    fn inverse_mean_recovers_original_targets() {
        let y = [1.0, 3.0, 5.0];
        let t = StandardizeTarget::new().fit(&y).expect("valid");
        let mut z = y;
        t.transform(&mut z).expect("finite");
        t.inverse_transform_mean(&mut z).expect("finite");
        assert_close(z[0], y[0], TOL);
        assert_close(z[1], y[1], TOL);
        assert_close(z[2], y[2], TOL);
    }

    #[test]
    fn inverse_variance_is_std_squared() {
        let y = [1.0, 3.0, 5.0];
        let t = StandardizeTarget::new().fit(&y).expect("valid");
        let s = t.std();
        let mut var = [0.25, 1.0, 4.0];
        t.inverse_transform_variance(&mut var).expect("finite");
        assert_close(var[0], 0.25 * s * s, TOL);
        assert_close(var[1], 1.0 * s * s, TOL);
        assert_close(var[2], 4.0 * s * s, TOL);
    }

    #[test]
    fn inverse_covariance_is_std_squared() {
        let y = [1.0, 3.0, 5.0];
        let t = StandardizeTarget::new().fit(&y).expect("valid");
        let s = t.std();
        let mut cov = [1.0, 0.5, 0.5, 4.0];
        t.inverse_transform_covariance(&mut cov).expect("finite");
        let s2 = s * s;
        assert_close(cov[0], 1.0 * s2, TOL);
        assert_close(cov[1], 0.5 * s2, TOL);
        assert_close(cov[2], 0.5 * s2, TOL);
        assert_close(cov[3], 4.0 * s2, TOL);
    }

    #[test]
    fn affine_closed_form_matches_unstandardized_scale() {
        let t = StandardizeTarget::new()
            .fit(&[2.0, 4.0, 6.0, 8.0])
            .expect("valid");
        let mu = t.mean();
        let s = t.std();
        let mut mean = [0.0, 1.0, -0.5];
        let mut var = [1.0, 0.25, 4.0];
        t.inverse_transform_mean(&mut mean).expect("finite");
        t.inverse_transform_variance(&mut var).expect("finite");
        assert_close(mean[0], mu, TOL);
        assert_close(mean[1], mu + s, TOL);
        assert_close(mean[2], mu - 0.5 * s, TOL);
        assert_close(var[0], s * s, TOL);
        assert_close(var[1], 0.25 * s * s, TOL);
        assert_close(var[2], 4.0 * s * s, TOL);
    }

    #[test]
    fn constant_target_uses_unit_scale() {
        let t = StandardizeTarget::new()
            .fit(&[4.0, 4.0, 4.0])
            .expect("valid");
        assert_close(t.mean(), 4.0, TOL);
        assert_close(t.std(), 1.0, TOL);
        let mut y = [4.0, 4.0];
        t.transform(&mut y).expect("finite");
        assert_close(y[0], 0.0, TOL);
        t.inverse_transform_mean(&mut y).expect("finite");
        assert_close(y[0], 4.0, TOL);
    }

    #[test]
    fn standardize_rejects_empty_and_non_finite() {
        let t = StandardizeTarget::new();
        assert!(matches!(t.fit(&[]), Err(GprError::EmptyInput)));
        assert!(matches!(
            StandardizeTarget::new().fit(&[1.0, f64::NAN]),
            Err(GprError::NonFiniteInput)
        ));
    }

    #[test]
    fn identity_rejects_non_finite() {
        assert!(matches!(
            IdentityTarget.fit(&[f64::INFINITY]),
            Err(GprError::NonFiniteInput)
        ));
    }

    #[test]
    fn minmax_scales_and_inverts() {
        let y = [1.0, 3.0, 5.0];
        let t = MinMaxTarget::new().fit(&y).expect("valid");
        assert_close(t.min(), 1.0, TOL);
        assert_close(t.max(), 5.0, TOL);
        let mut z = y;
        t.transform(&mut z).expect("finite");
        assert_close(z[0], 0.0, TOL);
        assert_close(z[1], 0.5, TOL);
        assert_close(z[2], 1.0, TOL);
        t.inverse_transform_mean(&mut z).expect("finite");
        assert_close(z[0], y[0], TOL);
        assert_close(z[1], y[1], TOL);
        assert_close(z[2], y[2], TOL);
        let mut var = [1.0, 0.25];
        t.inverse_transform_variance(&mut var).expect("finite");
        assert_close(var[0], 16.0, TOL);
        assert_close(var[1], 4.0, TOL);
    }

    #[test]
    fn minmax_constant_target_maps_to_lo() {
        let t = MinMaxTarget::new().fit(&[4.0, 4.0, 4.0]).expect("valid");
        let mut y = [4.0, 4.0];
        t.transform(&mut y).expect("finite");
        assert_close(y[0], 0.0, TOL);
        t.inverse_transform_mean(&mut y).expect("finite");
        assert_close(y[0], 4.0, TOL);
    }

    #[test]
    fn minmax_target_rejects() {
        assert!(MinMaxTarget::with_feature_range(1.0, 0.0).is_err());
    }
}
