//! Per-column input maps.

use std::fmt;

use super::{Transform, UnfittedTransform};
use crate::error::GprError;

fn require_pack(x: &[f64], n_rows: usize, n_cols: usize) -> Result<(), GprError> {
    let expected = crate::data::column_major_len(n_rows, n_cols)?;
    crate::data::require_count(x.len(), expected, "values")
}

fn require_column_count(n_cols: usize, expected: usize) -> Result<(), GprError> {
    if n_cols == expected {
        Ok(())
    } else {
        Err(GprError::DimensionMismatch {
            x_dim: n_cols,
            expected_dim: expected,
        })
    }
}

/// Unfitted per-column input maps. [`Self::fit`] returns [`FittedColumnwiseInput`].
///
/// The number of maps must equal the feature count `d`. A uniform column
/// typically uses [`super::MinMaxInput`]; a near-normal column typically
/// uses [`super::StandardizeInput`]. A column may also be
/// [`super::IdentityInput`] or a caller-supplied [`UnfittedTransform`].
///
/// Pass this to [`crate::Gpr::with_input_transform`].
///
/// # Examples
///
/// ```rust
/// use gprx::transform::{ColumnwiseInput, MinMaxInput, StandardizeInput, Transform};
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let t = ColumnwiseInput::new()
///     .then(MinMaxInput::new())
///     .then(StandardizeInput::new())
///     .fit(&[0.0, 2.0, 4.0, 1.0, 3.0, 5.0], 3, 2)?;
/// let mut x = [0.0, 2.0, 4.0, 1.0, 3.0, 5.0];
/// t.apply(&mut x, 3, 2)?;
/// # let _ = x;
/// # Ok(())
/// # }
/// ```
pub struct ColumnwiseInput {
    maps: Vec<Box<dyn UnfittedTransform>>,
}

impl ColumnwiseInput {
    /// Returns an empty column list. Append one map per feature with [`Self::then`].
    pub fn new() -> Self {
        Self { maps: Vec::new() }
    }

    /// Appends the map for the next feature column.
    pub fn then(mut self, map: impl UnfittedTransform + 'static) -> Self {
        self.maps.push(Box::new(map));
        self
    }

    /// Returns the number of column maps.
    pub fn len(&self) -> usize {
        self.maps.len()
    }

    /// Returns whether this list has no column maps.
    pub fn is_empty(&self) -> bool {
        self.maps.is_empty()
    }

    pub(crate) fn from_maps(maps: Vec<Box<dyn UnfittedTransform>>) -> Self {
        Self { maps }
    }

    pub(crate) fn maps(&self) -> &[Box<dyn UnfittedTransform>] {
        &self.maps
    }

    /// Fits each map on its own column of `x`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::DimensionMismatch`] when the number of maps is not
    /// `n_cols`, or [`GprError`] from packing, finiteness, or a column map.
    pub fn fit(
        self,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
    ) -> Result<FittedColumnwiseInput, GprError> {
        require_pack(x, n_rows, n_cols)?;
        crate::data::require_finite(x)?;
        require_column_count(n_cols, self.maps.len())?;
        let mut fitted = Vec::with_capacity(self.maps.len());
        for (col, map) in self.maps.into_iter().enumerate() {
            let start = col * n_rows;
            let column = &x[start..start + n_rows];
            fitted.push(map.fit(column, n_rows, 1)?);
        }
        Ok(FittedColumnwiseInput { maps: fitted })
    }
}

impl Default for ColumnwiseInput {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for ColumnwiseInput {
    fn clone(&self) -> Self {
        Self {
            maps: self.maps.iter().map(|map| map.clone_box()).collect(),
        }
    }
}

impl fmt::Debug for ColumnwiseInput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ColumnwiseInput")
            .field("len", &self.maps.len())
            .finish()
    }
}

impl UnfittedTransform for ColumnwiseInput {
    fn fit(
        self: Box<Self>,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
    ) -> Result<Box<dyn Transform>, GprError> {
        (*self).fit(x, n_rows, n_cols).map(|t| Box::new(t) as _)
    }

    fn clone_box(&self) -> Box<dyn UnfittedTransform> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Fitted per-column input maps.
///
/// [`Self::apply`] runs each map on its own column.
pub struct FittedColumnwiseInput {
    maps: Vec<Box<dyn Transform>>,
}

impl FittedColumnwiseInput {
    /// Returns the number of column maps.
    pub fn len(&self) -> usize {
        self.maps.len()
    }

    /// Returns whether this list has no column maps.
    pub fn is_empty(&self) -> bool {
        self.maps.is_empty()
    }

    pub(crate) fn from_maps(maps: Vec<Box<dyn Transform>>) -> Self {
        Self { maps }
    }

    pub(crate) fn maps(&self) -> &[Box<dyn Transform>] {
        &self.maps
    }
}

impl Clone for FittedColumnwiseInput {
    fn clone(&self) -> Self {
        Self {
            maps: self.maps.iter().map(|map| map.clone_box()).collect(),
        }
    }
}

impl fmt::Debug for FittedColumnwiseInput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FittedColumnwiseInput")
            .field("len", &self.maps.len())
            .finish()
    }
}

impl Transform for FittedColumnwiseInput {
    fn apply(&self, x: &mut [f64], n_rows: usize, n_cols: usize) -> Result<(), GprError> {
        require_column_count(n_cols, self.maps.len())?;
        crate::data::require_count(
            x.len(),
            crate::data::column_major_len(n_rows, n_cols)?,
            "values",
        )?;
        crate::data::require_finite(x)?;
        for (col, map) in self.maps.iter().enumerate() {
            let start = col * n_rows;
            map.apply(&mut x[start..start + n_rows], n_rows, 1)?;
        }
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn Transform> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ColumnwiseInput, FittedColumnwiseInput, Transform, UnfittedTransform, require_pack,
    };
    use crate::error::GprError;
    use crate::transform::{IdentityInput, MinMaxInput, StandardizeInput};

    const TOL: f64 = 1e-10;

    fn assert_close(actual: f64, expected: f64) {
        let scale = expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= TOL * scale,
            "actual={actual}, expected={expected}"
        );
    }

    fn assert_send_sync<T: Send + Sync>() {}

    /// Caller-supplied column map used to cover a custom [`UnfittedTransform`].
    #[derive(Clone, Copy, Debug)]
    struct TimesTwo;

    impl TimesTwo {
        fn fit(self, x: &[f64], n_rows: usize, n_cols: usize) -> Result<Self, GprError> {
            require_pack(x, n_rows, n_cols)?;
            crate::data::require_finite(x)?;
            Ok(self)
        }
    }

    impl UnfittedTransform for TimesTwo {
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

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    impl Transform for TimesTwo {
        fn apply(&self, x: &mut [f64], n_rows: usize, n_cols: usize) -> Result<(), GprError> {
            require_pack(x, n_rows, n_cols)?;
            crate::data::require_finite(x)?;
            for value in x {
                *value *= 2.0;
            }
            Ok(())
        }

        fn clone_box(&self) -> Box<dyn Transform> {
            Box::new(*self)
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[test]
    fn is_send_sync() {
        assert_send_sync::<ColumnwiseInput>();
        assert_send_sync::<FittedColumnwiseInput>();
    }

    #[test]
    fn minmax_and_standardize_each_column() {
        let x = [0.0, 2.0, 4.0, 1.0, 3.0, 5.0];
        let spec = ColumnwiseInput::new()
            .then(MinMaxInput::new())
            .then(StandardizeInput::new());
        assert_eq!(spec.len(), 2);
        let t = spec.fit(&x, 3, 2).expect("valid");
        let minmax = MinMaxInput::new().fit(&x[0..3], 3, 1).expect("valid");
        let std = StandardizeInput::new().fit(&x[3..6], 3, 1).expect("valid");
        let mut expected = x;
        minmax.apply(&mut expected[0..3], 3, 1).expect("ok");
        std.apply(&mut expected[3..6], 3, 1).expect("ok");
        let mut got = x;
        t.apply(&mut got, 3, 2).expect("ok");
        for (actual, want) in got.iter().zip(expected.iter()) {
            assert_close(*actual, *want);
        }
    }

    #[test]
    fn custom_map_scales_its_column() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let t = ColumnwiseInput::new()
            .then(IdentityInput)
            .then(TimesTwo)
            .fit(&x, 2, 2)
            .expect("valid");
        let mut z = x;
        t.apply(&mut z, 2, 2).expect("ok");
        assert_close(z[0], 1.0);
        assert_close(z[1], 2.0);
        assert_close(z[2], 6.0);
        assert_close(z[3], 8.0);
    }

    #[test]
    fn fit_rejects_length_mismatch() {
        let x = [0.0, 1.0, 2.0, 3.0];
        assert!(matches!(
            ColumnwiseInput::new()
                .then(MinMaxInput::new())
                .fit(&x, 2, 2),
            Err(GprError::DimensionMismatch {
                x_dim: 2,
                expected_dim: 1
            })
        ));
        assert!(matches!(
            ColumnwiseInput::new()
                .then(MinMaxInput::new())
                .then(StandardizeInput::new())
                .then(IdentityInput)
                .fit(&x, 2, 2),
            Err(GprError::DimensionMismatch {
                x_dim: 2,
                expected_dim: 3
            })
        ));
    }

    #[test]
    fn apply_rejects_column_mismatch() {
        let x = [0.0, 1.0, 2.0, 3.0];
        let t = ColumnwiseInput::new()
            .then(MinMaxInput::new())
            .then(StandardizeInput::new())
            .fit(&x, 2, 2)
            .expect("valid");
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
    fn clone_keeps_column_count() {
        let spec = ColumnwiseInput::new()
            .then(MinMaxInput::new())
            .then(StandardizeInput::new());
        assert_eq!(spec.clone().len(), 2);
    }
}
