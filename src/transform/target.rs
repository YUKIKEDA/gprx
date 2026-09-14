//! Target (`y`) transforms: identity and standardize.

use super::{population_std, require_finite, require_nonempty};
use crate::error::GpError;

/// Maps observations `y` into the space the GP fits, and maps predictive
/// mean and variance back to the original scale.
///
/// [`StandardizeTarget`] is the usual choice when the mean function is zero:
/// it centers and scales `y`, then undoes that affine map on predictions.
/// Variance undoes `y' = (y - μ) / s` as `Var(y) = s² Var(y')`.
pub trait TargetTransform: Send + Sync {
    /// Estimates transform parameters from training targets.
    ///
    /// # Errors
    ///
    /// Returns [`GpError`] when `y` is empty, non-finite, or otherwise invalid
    /// for this transform.
    fn fit(&mut self, y: &[f64]) -> Result<(), GpError>;

    /// Applies the forward map to targets in place.
    ///
    /// # Errors
    ///
    /// Returns [`GpError::NotFitted`] when [`Self::fit`] has not succeeded, or
    /// [`GpError::NonFiniteInput`] when `y` contains `NaN` or `Inf`.
    fn transform(&self, y: &mut [f64]) -> Result<(), GpError>;

    /// Maps latent or observation means from transformed space to `y` scale.
    ///
    /// # Errors
    ///
    /// Returns [`GpError::NotFitted`] when [`Self::fit`] has not succeeded, or
    /// [`GpError::NonFiniteInput`] when `mean` contains `NaN` or `Inf`.
    fn inverse_transform_mean(&self, mean: &mut [f64]) -> Result<(), GpError>;

    /// Maps predictive variances from transformed space to `y` scale.
    ///
    /// For the affine map `y' = (y - μ) / s`, each entry is multiplied by `s²`.
    ///
    /// # Errors
    ///
    /// Returns [`GpError::NotFitted`] when [`Self::fit`] has not succeeded, or
    /// [`GpError::NonFiniteInput`] when `var` contains `NaN` or `Inf`.
    fn inverse_transform_variance(&self, var: &mut [f64]) -> Result<(), GpError>;
}

/// Leaves targets and predictions unchanged.
///
/// # Examples
///
/// ```rust
/// use gprx::transform::{IdentityTarget, TargetTransform};
///
/// # fn main() -> Result<(), gprx::GpError> {
/// let mut t = IdentityTarget;
/// t.fit(&[1.0, 2.0])?;
/// let mut mean = [0.5, 1.5];
/// t.inverse_transform_mean(&mut mean)?;
/// # let _ = mean;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct IdentityTarget;

impl TargetTransform for IdentityTarget {
    fn fit(&mut self, y: &[f64]) -> Result<(), GpError> {
        require_finite(y)
    }

    fn transform(&self, y: &mut [f64]) -> Result<(), GpError> {
        require_finite(y)
    }

    fn inverse_transform_mean(&self, mean: &mut [f64]) -> Result<(), GpError> {
        require_finite(mean)
    }

    fn inverse_transform_variance(&self, var: &mut [f64]) -> Result<(), GpError> {
        require_finite(var)
    }
}

/// Centers and scales targets to mean 0 and variance 1, then inverts that map.
///
/// `s` is the population standard deviation. A constant `y` uses `s = 1` so
/// centering stays well-defined and the inverse is a shift by the mean.
///
/// # Examples
///
/// ```rust
/// use gprx::transform::{StandardizeTarget, TargetTransform};
///
/// # fn main() -> Result<(), gprx::GpError> {
/// let mut t = StandardizeTarget::new();
/// t.fit(&[1.0, 3.0, 5.0])?;
/// let mut y = [1.0, 3.0, 5.0];
/// t.transform(&mut y)?;
/// t.inverse_transform_mean(&mut y)?;
/// # let _ = y;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StandardizeTarget {
    mean: f64,
    std: f64,
    fitted: bool,
}

impl StandardizeTarget {
    /// Returns an unfitted transform. Call [`TargetTransform::fit`] before use.
    pub fn new() -> Self {
        Self {
            mean: 0.0,
            std: 1.0,
            fitted: false,
        }
    }

    /// Returns the training mean after a successful fit.
    pub fn mean(&self) -> Option<f64> {
        self.fitted.then_some(self.mean)
    }

    /// Returns the training scale `s` after a successful fit.
    pub fn std(&self) -> Option<f64> {
        self.fitted.then_some(self.std)
    }

    fn require_fitted(&self) -> Result<(), GpError> {
        if self.fitted {
            Ok(())
        } else {
            Err(GpError::NotFitted)
        }
    }
}

impl Default for StandardizeTarget {
    fn default() -> Self {
        Self::new()
    }
}

impl TargetTransform for StandardizeTarget {
    fn fit(&mut self, y: &[f64]) -> Result<(), GpError> {
        require_nonempty(y.len())?;
        require_finite(y)?;
        let n = y.len() as f64;
        let mean = y.iter().sum::<f64>() / n;
        self.mean = mean;
        self.std = population_std(y, mean);
        self.fitted = true;
        Ok(())
    }

    fn transform(&self, y: &mut [f64]) -> Result<(), GpError> {
        self.require_fitted()?;
        require_finite(y)?;
        let mean = self.mean;
        let std = self.std;
        for value in y {
            *value = (*value - mean) / std;
        }
        Ok(())
    }

    fn inverse_transform_mean(&self, mean: &mut [f64]) -> Result<(), GpError> {
        self.require_fitted()?;
        require_finite(mean)?;
        let loc = self.mean;
        let std = self.std;
        for value in mean {
            *value = loc + std * *value;
        }
        Ok(())
    }

    fn inverse_transform_variance(&self, var: &mut [f64]) -> Result<(), GpError> {
        self.require_fitted()?;
        require_finite(var)?;
        let scale = self.std * self.std;
        for value in var {
            *value *= scale;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{IdentityTarget, StandardizeTarget, TargetTransform};
    use crate::error::GpError;

    const TOL: f64 = 1e-10;

    fn assert_close(actual: f64, expected: f64) {
        let scale = expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= TOL * scale,
            "actual={actual}, expected={expected}"
        );
    }

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn is_send_sync() {
        assert_send_sync::<IdentityTarget>();
        assert_send_sync::<StandardizeTarget>();
    }

    #[test]
    fn identity_is_a_no_op() {
        let mut t = IdentityTarget;
        t.fit(&[1.0, 2.0]).expect("finite");
        let mut y = [1.0, 2.0];
        t.transform(&mut y).expect("finite");
        assert_close(y[0], 1.0);
        assert_close(y[1], 2.0);
        t.inverse_transform_mean(&mut y).expect("finite");
        t.inverse_transform_variance(&mut y).expect("finite");
        assert_close(y[0], 1.0);
        assert_close(y[1], 2.0);
    }

    #[test]
    fn standardize_centers_and_unit_scales() {
        let y = [1.0, 3.0, 5.0];
        let mut t = StandardizeTarget::new();
        t.fit(&y).expect("valid");
        assert_close(t.mean().expect("fitted"), 3.0);
        assert_close(t.std().expect("fitted"), (8.0 / 3.0_f64).sqrt());
        let mut z = y;
        t.transform(&mut z).expect("fitted");
        let z_mean = z.iter().sum::<f64>() / 3.0;
        assert_close(z_mean, 0.0);
        let z_var = z.iter().map(|v| v * v).sum::<f64>() / 3.0;
        assert_close(z_var, 1.0);
    }

    #[test]
    fn inverse_mean_recovers_original_targets() {
        let y = [1.0, 3.0, 5.0];
        let mut t = StandardizeTarget::new();
        t.fit(&y).expect("valid");
        let mut z = y;
        t.transform(&mut z).expect("fitted");
        t.inverse_transform_mean(&mut z).expect("fitted");
        assert_close(z[0], y[0]);
        assert_close(z[1], y[1]);
        assert_close(z[2], y[2]);
    }

    #[test]
    fn inverse_variance_is_std_squared() {
        let y = [1.0, 3.0, 5.0];
        let mut t = StandardizeTarget::new();
        t.fit(&y).expect("valid");
        let s = t.std().expect("fitted");
        let mut var = [0.25, 1.0, 4.0];
        t.inverse_transform_variance(&mut var).expect("fitted");
        assert_close(var[0], 0.25 * s * s);
        assert_close(var[1], 1.0 * s * s);
        assert_close(var[2], 4.0 * s * s);
    }

    #[test]
    fn affine_closed_form_matches_unstandardized_scale() {
        // A GP on standardized y that predicts m' and v' is the same affine
        // map as training on raw y only after inverse_transform_*.
        let mut t = StandardizeTarget::new();
        t.fit(&[2.0, 4.0, 6.0, 8.0]).expect("valid");
        let mu = t.mean().expect("fitted");
        let s = t.std().expect("fitted");
        let mut mean = [0.0, 1.0, -0.5];
        let mut var = [1.0, 0.25, 4.0];
        t.inverse_transform_mean(&mut mean).expect("fitted");
        t.inverse_transform_variance(&mut var).expect("fitted");
        assert_close(mean[0], mu);
        assert_close(mean[1], mu + s);
        assert_close(mean[2], mu - 0.5 * s);
        assert_close(var[0], s * s);
        assert_close(var[1], 0.25 * s * s);
        assert_close(var[2], 4.0 * s * s);
    }

    #[test]
    fn constant_target_uses_unit_scale() {
        let mut t = StandardizeTarget::new();
        t.fit(&[4.0, 4.0, 4.0]).expect("valid");
        assert_close(t.mean().expect("fitted"), 4.0);
        assert_close(t.std().expect("fitted"), 1.0);
        let mut y = [4.0, 4.0];
        t.transform(&mut y).expect("fitted");
        assert_close(y[0], 0.0);
        t.inverse_transform_mean(&mut y).expect("fitted");
        assert_close(y[0], 4.0);
    }

    #[test]
    fn standardize_rejects_empty_and_non_finite() {
        let mut t = StandardizeTarget::new();
        assert!(matches!(t.fit(&[]), Err(GpError::EmptyInput)));
        assert!(matches!(
            t.fit(&[1.0, f64::NAN]),
            Err(GpError::NonFiniteInput)
        ));
        assert!(matches!(t.transform(&mut [1.0]), Err(GpError::NotFitted)));
    }

    #[test]
    fn identity_rejects_non_finite() {
        let mut t = IdentityTarget;
        assert!(matches!(
            t.fit(&[f64::INFINITY]),
            Err(GpError::NonFiniteInput)
        ));
    }
}
