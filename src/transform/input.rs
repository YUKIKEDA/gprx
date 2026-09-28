//! Input (`X`) transforms: identity, per-feature standardize, and min-max.

use std::any::Any;

use super::population_std;
use crate::error::GprError;

/// Unfitted input map. [`Self::fit`] consumes it and returns a [`Transform`].
pub trait UnfittedTransform: Send + Sync {
    /// Estimates transform parameters from training features.
    ///
    /// # Errors
    ///
    /// Returns [`GprError`] when `x` is empty, packed incorrectly, or contains
    /// a non-finite value.
    fn fit(
        self: Box<Self>,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
    ) -> Result<Box<dyn Transform>, GprError>;

    /// Clones this map into a new box. Used by [`crate::Gpr`] clone.
    fn clone_box(&self) -> Box<dyn UnfittedTransform>;

    /// Downcast handle used when encoding a built-in map for persist.
    fn as_any(&self) -> &dyn Any;

    /// Registry key for a caller-defined map. Built-ins return [`None`].
    ///
    /// The id must not start with `gprx.`.
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
            reason: "this input transform does not implement persist_state".to_owned(),
        })
    }
}

/// Fitted input map. [`Self::apply`] exists only here.
///
/// `x` is column-major: column `j` occupies `x[j * n_rows .. (j + 1) * n_rows]`.
pub trait Transform: Send + Sync {
    /// Applies the fitted map to a feature matrix in place.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::DimensionMismatch`] when `n_cols` differs from the
    /// fit, or [`GprError::NonFiniteInput`] when `x` contains `NaN` or `Inf`.
    fn apply(&self, x: &mut [f64], n_rows: usize, n_cols: usize) -> Result<(), GprError>;

    /// Clones this map into a new box. Used by [`crate::FittedGpr`] clone.
    fn clone_box(&self) -> Box<dyn Transform>;

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
            reason: "this input transform does not implement persist_state".to_owned(),
        })
    }
}

fn require_pack(x: &[f64], n_rows: usize, n_cols: usize) -> Result<(), GprError> {
    let expected = crate::data::column_major_len(n_rows, n_cols)?;
    crate::data::require_count(x.len(), expected, "values")
}

/// Leaves features unchanged.
///
/// Identity has no unfitted state. [`Self::fit`] only checks packing and
/// finiteness.
///
/// # Examples
///
/// ```rust
/// use gprx::transform::{IdentityInput, Transform};
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let t = IdentityInput.fit(&[1.0, 2.0, 3.0, 4.0], 2, 2)?;
/// let mut x = [1.0, 2.0, 3.0, 4.0];
/// t.apply(&mut x, 2, 2)?;
/// # let _ = x;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct IdentityInput;

impl IdentityInput {
    /// Checks packing and finiteness and returns this map.
    ///
    /// # Errors
    ///
    /// Returns [`GprError`] when `x` is empty, packed incorrectly, or contains
    /// a non-finite value.
    pub fn fit(self, x: &[f64], n_rows: usize, n_cols: usize) -> Result<Self, GprError> {
        require_pack(x, n_rows, n_cols)?;
        crate::data::require_finite(x)?;
        Ok(self)
    }
}

impl UnfittedTransform for IdentityInput {
    fn fit(
        self: Box<Self>,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
    ) -> Result<Box<dyn Transform>, GprError> {
        (*self).fit(x, n_rows, n_cols).map(|t| Box::new(t) as _)
    }

    fn clone_box(&self) -> Box<dyn UnfittedTransform> {
        Box::new(*self)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Transform for IdentityInput {
    fn apply(&self, x: &mut [f64], n_rows: usize, n_cols: usize) -> Result<(), GprError> {
        require_pack(x, n_rows, n_cols)?;
        crate::data::require_finite(x)
    }

    fn clone_box(&self) -> Box<dyn Transform> {
        Box::new(*self)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Unfitted per-column center-and-scale map. [`Self::fit`] returns
/// [`FittedStandardizeInput`].
///
/// # Examples
///
/// ```rust
/// use gprx::transform::{StandardizeInput, Transform};
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let t = StandardizeInput::new().fit(&[0.0, 2.0, 10.0, 30.0], 2, 2)?;
/// let mut x = [0.0, 2.0, 10.0, 30.0];
/// t.apply(&mut x, 2, 2)?;
/// # let _ = x;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct StandardizeInput;

impl StandardizeInput {
    /// Returns an unfitted transform.
    pub fn new() -> Self {
        Self
    }

    /// Estimates per-column `μ` and `s` and returns the fitted map.
    ///
    /// A constant column uses scale `1`, matching [`super::StandardizeTarget`].
    ///
    /// # Errors
    ///
    /// Returns [`GprError`] when `x` is empty, packed incorrectly, or contains
    /// a non-finite value.
    pub fn fit(
        self,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
    ) -> Result<FittedStandardizeInput, GprError> {
        require_pack(x, n_rows, n_cols)?;
        crate::data::require_finite(x)?;
        let mut mean = Vec::with_capacity(n_cols);
        let mut std = Vec::with_capacity(n_cols);
        for col in 0..n_cols {
            let start = col * n_rows;
            let column = &x[start..start + n_rows];
            let m = column.iter().sum::<f64>() / n_rows as f64;
            mean.push(m);
            std.push(population_std(column, m));
        }
        Ok(FittedStandardizeInput { mean, std })
    }
}

impl UnfittedTransform for StandardizeInput {
    fn fit(
        self: Box<Self>,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
    ) -> Result<Box<dyn Transform>, GprError> {
        (*self).fit(x, n_rows, n_cols).map(|t| Box::new(t) as _)
    }

    fn clone_box(&self) -> Box<dyn UnfittedTransform> {
        Box::new(*self)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Fitted per-column center-and-scale map.
#[derive(Clone, Debug, PartialEq)]
pub struct FittedStandardizeInput {
    mean: Vec<f64>,
    std: Vec<f64>,
}

impl FittedStandardizeInput {
    /// Returns per-column training means.
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    /// Returns per-column training scales.
    pub fn std(&self) -> &[f64] {
        &self.std
    }

    pub(crate) fn from_parts(mean: Vec<f64>, std: Vec<f64>) -> Result<Self, GprError> {
        crate::data::require_count(std.len(), mean.len(), "values")?;
        crate::data::require_finite(&mean)?;
        crate::data::require_finite(&std)?;
        if mean.is_empty() {
            return Err(GprError::EmptyInput);
        }
        if std.iter().any(|s| *s <= 0.0) {
            return Err(GprError::InvalidHyperparameter {
                reason: "fitted standardize scale must be positive".to_owned(),
            });
        }
        Ok(Self { mean, std })
    }
}

impl Transform for FittedStandardizeInput {
    fn apply(&self, x: &mut [f64], n_rows: usize, n_cols: usize) -> Result<(), GprError> {
        let expected_cols = self.mean.len();
        if n_cols != expected_cols {
            return Err(GprError::DimensionMismatch {
                x_dim: n_cols,
                expected_dim: expected_cols,
            });
        }
        crate::data::require_count(
            x.len(),
            crate::data::column_major_len(n_rows, n_cols)?,
            "values",
        )?;
        crate::data::require_finite(x)?;
        for col in 0..n_cols {
            let mean = self.mean[col];
            let std = self.std[col];
            let start = col * n_rows;
            for value in &mut x[start..start + n_rows] {
                *value = (*value - mean) / std;
            }
        }
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn Transform> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Unfitted per-column min-max map onto a closed interval, default `[0, 1]`.
///
/// [`Self::fit`] returns [`FittedMinMaxInput`].
///
/// # Examples
///
/// ```rust
/// use gprx::transform::{MinMaxInput, Transform};
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let t = MinMaxInput::new().fit(&[0.0, 2.0, 4.0], 3, 1)?;
/// let mut x = [0.0, 2.0, 4.0];
/// t.apply(&mut x, 3, 1)?;
/// assert!((x[0] - 0.0).abs() < 1e-12);
/// assert!((x[2] - 1.0).abs() < 1e-12);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MinMaxInput {
    range_lo: f64,
    range_hi: f64,
}

impl MinMaxInput {
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

    /// Estimates per-column min / max and returns the fitted map.
    ///
    /// # Errors
    ///
    /// Returns [`GprError`] when `x` is empty, packed incorrectly, or contains
    /// a non-finite value.
    pub fn fit(
        self,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
    ) -> Result<FittedMinMaxInput, GprError> {
        require_pack(x, n_rows, n_cols)?;
        crate::data::require_finite(x)?;
        let mut data_min = Vec::with_capacity(n_cols);
        let mut data_max = Vec::with_capacity(n_cols);
        for col in 0..n_cols {
            let start = col * n_rows;
            let column = &x[start..start + n_rows];
            let mut min = column[0];
            let mut max = column[0];
            for &value in &column[1..] {
                if value < min {
                    min = value;
                }
                if value > max {
                    max = value;
                }
            }
            data_min.push(min);
            data_max.push(max);
        }
        Ok(FittedMinMaxInput {
            data_min,
            data_max,
            range_lo: self.range_lo,
            range_hi: self.range_hi,
        })
    }
}

impl Default for MinMaxInput {
    fn default() -> Self {
        Self::new()
    }
}

impl UnfittedTransform for MinMaxInput {
    fn fit(
        self: Box<Self>,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
    ) -> Result<Box<dyn Transform>, GprError> {
        (*self).fit(x, n_rows, n_cols).map(|t| Box::new(t) as _)
    }

    fn clone_box(&self) -> Box<dyn UnfittedTransform> {
        Box::new(*self)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Fitted per-column min-max map.
///
/// `x' = lo + (hi - lo) * (x - min) / (max - min)`. A constant column uses
/// denominator `1`, so every entry maps to `lo`.
#[derive(Clone, Debug, PartialEq)]
pub struct FittedMinMaxInput {
    data_min: Vec<f64>,
    data_max: Vec<f64>,
    range_lo: f64,
    range_hi: f64,
}

impl FittedMinMaxInput {
    /// Returns the output interval `[lo, hi]`.
    pub fn feature_range(&self) -> (f64, f64) {
        (self.range_lo, self.range_hi)
    }

    /// Returns per-column training minima.
    pub fn min(&self) -> &[f64] {
        &self.data_min
    }

    /// Returns per-column training maxima.
    pub fn max(&self) -> &[f64] {
        &self.data_max
    }

    pub(crate) fn from_parts(
        data_min: Vec<f64>,
        data_max: Vec<f64>,
        range_lo: f64,
        range_hi: f64,
    ) -> Result<Self, GprError> {
        require_feature_range(range_lo, range_hi)?;
        crate::data::require_count(data_max.len(), data_min.len(), "values")?;
        crate::data::require_finite(&data_min)?;
        crate::data::require_finite(&data_max)?;
        if data_min.is_empty() {
            return Err(GprError::EmptyInput);
        }
        Ok(Self {
            data_min,
            data_max,
            range_lo,
            range_hi,
        })
    }
}

impl Transform for FittedMinMaxInput {
    fn apply(&self, x: &mut [f64], n_rows: usize, n_cols: usize) -> Result<(), GprError> {
        let expected_cols = self.data_min.len();
        if n_cols != expected_cols {
            return Err(GprError::DimensionMismatch {
                x_dim: n_cols,
                expected_dim: expected_cols,
            });
        }
        crate::data::require_count(
            x.len(),
            crate::data::column_major_len(n_rows, n_cols)?,
            "values",
        )?;
        crate::data::require_finite(x)?;
        let out_span = self.range_hi - self.range_lo;
        for col in 0..n_cols {
            let min = self.data_min[col];
            let span = column_span(self.data_max[col], min);
            let start = col * n_rows;
            for value in &mut x[start..start + n_rows] {
                *value = self.range_lo + out_span * (*value - min) / span;
            }
        }
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn Transform> {
        Box::new(self.clone())
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

fn column_span(max: f64, min: f64) -> f64 {
    let span = max - min;
    if span > 0.0 && span.is_finite() {
        span
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::{
        FittedMinMaxInput, FittedStandardizeInput, IdentityInput, MinMaxInput, StandardizeInput,
        Transform,
    };
    use crate::error::GprError;

    const TOL: f64 = 1e-10;

    use crate::test_check::{assert_close, assert_send_sync};

    #[test]
    fn is_send_sync() {
        assert_send_sync::<IdentityInput>();
        assert_send_sync::<StandardizeInput>();
        assert_send_sync::<FittedStandardizeInput>();
        assert_send_sync::<MinMaxInput>();
        assert_send_sync::<FittedMinMaxInput>();
    }

    #[test]
    fn identity_leaves_features() {
        let t = IdentityInput
            .fit(&[1.0, 2.0, 3.0, 4.0], 2, 2)
            .expect("valid");
        let mut x = [1.0, 2.0, 3.0, 4.0];
        t.apply(&mut x, 2, 2).expect("valid");
        assert_close(x[0], 1.0, TOL);
        assert_close(x[3], 4.0, TOL);
    }

    #[test]
    fn standardize_each_column() {
        let x = [0.0, 2.0, 4.0, 1.0, 1.0, 1.0];
        let t = StandardizeInput::new().fit(&x, 3, 2).expect("valid");
        let mean = t.mean();
        let std = t.std();
        assert_close(mean[0], 2.0, TOL);
        assert_close(std[0], (8.0 / 3.0_f64).sqrt(), TOL);
        assert_close(mean[1], 1.0, TOL);
        assert_close(std[1], 1.0, TOL);
        let mut z = x;
        t.apply(&mut z, 3, 2).expect("fitted");
        let z0 = z[0] + z[1] + z[2];
        assert_close(z0 / 3.0, 0.0, TOL);
        assert_close(z[3], 0.0, TOL);
        assert_close(z[4], 0.0, TOL);
        assert_close(z[5], 0.0, TOL);
    }

    #[test]
    fn apply_rejects_column_mismatch() {
        let x = [0.0, 1.0, 2.0, 3.0];
        let t = StandardizeInput::new().fit(&x, 2, 2).expect("valid");
        let mut other = [0.0, 1.0];
        assert!(matches!(
            t.apply(&mut other, 2, 1),
            Err(GprError::DimensionMismatch {
                x_dim: 1,
                expected_dim: 2
            })
        ));
    }

    #[test]
    fn standardize_rejects_empty() {
        assert!(matches!(
            StandardizeInput::new().fit(&[], 0, 1),
            Err(GprError::EmptyInput)
        ));
        assert!(matches!(
            StandardizeInput::new().fit(&[1.0, f64::NAN], 1, 2),
            Err(GprError::NonFiniteInput)
        ));
    }

    #[test]
    fn minmax_scales_column_to_unit_interval() {
        let x = [0.0, 2.0, 4.0, 1.0, 1.0, 1.0];
        let t = MinMaxInput::new().fit(&x, 3, 2).expect("valid");
        let min = t.min();
        let max = t.max();
        assert_close(min[0], 0.0, TOL);
        assert_close(max[0], 4.0, TOL);
        assert_close(min[1], 1.0, TOL);
        assert_close(max[1], 1.0, TOL);
        assert_eq!(t.feature_range(), (0.0, 1.0));
        let mut z = x;
        t.apply(&mut z, 3, 2).expect("fitted");
        assert_close(z[0], 0.0, TOL);
        assert_close(z[1], 0.5, TOL);
        assert_close(z[2], 1.0, TOL);
        assert_close(z[3], 0.0, TOL);
        assert_close(z[4], 0.0, TOL);
        assert_close(z[5], 0.0, TOL);
    }

    #[test]
    fn minmax_custom_range_and_rejects() {
        let spec = MinMaxInput::with_feature_range(-1.0, 1.0).expect("valid range");
        let x = [0.0, 4.0];
        let t = spec.fit(&x, 2, 1).expect("valid");
        let mut z = x;
        t.apply(&mut z, 2, 1).expect("fitted");
        assert_close(z[0], -1.0, TOL);
        assert_close(z[1], 1.0, TOL);
        assert!(MinMaxInput::with_feature_range(1.0, 1.0).is_err());
        assert!(MinMaxInput::with_feature_range(0.0, f64::NAN).is_err());
    }
}
