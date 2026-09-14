//! Input (`X`) transforms: identity and per-feature standardize.

use super::{column_major_len, population_std, require_finite, require_len, require_nonempty};
use crate::error::GprError;

/// Maps feature matrices in place before kernel evaluation.
///
/// `x` is column-major: column `j` occupies `x[j * n_rows .. (j + 1) * n_rows]`.
pub trait Transform: Send + Sync {
    /// Estimates transform parameters from training features.
    ///
    /// # Errors
    ///
    /// Returns [`GprError`] when `x` is empty, packed incorrectly, or contains
    /// a non-finite value.
    fn fit(&mut self, x: &[f64], n_rows: usize, n_cols: usize) -> Result<(), GprError>;

    /// Applies the fitted map to a feature matrix in place.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::NotFitted`] when [`Self::fit`] has not succeeded,
    /// [`GprError::DimensionMismatch`] when `n_cols` differs from the fit, or
    /// [`GprError::NonFiniteInput`] when `x` contains `NaN` or `Inf`.
    fn apply(&self, x: &mut [f64], n_rows: usize, n_cols: usize) -> Result<(), GprError>;
}

fn require_pack(x: &[f64], n_rows: usize, n_cols: usize) -> Result<(), GprError> {
    let expected = column_major_len(n_rows, n_cols)?;
    require_len(x, expected)
}

/// Leaves features unchanged.
///
/// # Examples
///
/// ```rust
/// use gprx::transform::{IdentityInput, Transform};
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let mut t = IdentityInput;
/// let mut x = [1.0, 2.0, 3.0, 4.0];
/// t.fit(&x, 2, 2)?;
/// t.apply(&mut x, 2, 2)?;
/// # let _ = x;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct IdentityInput;

impl Transform for IdentityInput {
    fn fit(&mut self, x: &[f64], n_rows: usize, n_cols: usize) -> Result<(), GprError> {
        require_pack(x, n_rows, n_cols)?;
        require_finite(x)
    }

    fn apply(&self, x: &mut [f64], n_rows: usize, n_cols: usize) -> Result<(), GprError> {
        require_pack(x, n_rows, n_cols)?;
        require_finite(x)
    }
}

/// Centers and scales each feature column to mean 0 and variance 1.
///
/// A constant column uses scale `1`, matching [`super::StandardizeTarget`].
///
/// # Examples
///
/// ```rust
/// use gprx::transform::{StandardizeInput, Transform};
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let mut t = StandardizeInput::new();
/// let mut x = [0.0, 2.0, 10.0, 30.0];
/// t.fit(&x, 2, 2)?;
/// t.apply(&mut x, 2, 2)?;
/// # let _ = x;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct StandardizeInput {
    mean: Vec<f64>,
    std: Vec<f64>,
}

impl StandardizeInput {
    /// Returns an unfitted transform. Call [`Transform::fit`] before [`Transform::apply`].
    pub fn new() -> Self {
        Self {
            mean: Vec::new(),
            std: Vec::new(),
        }
    }

    /// Returns per-column training means after a successful fit.
    pub fn mean(&self) -> Option<&[f64]> {
        if self.mean.is_empty() {
            None
        } else {
            Some(self.mean.as_slice())
        }
    }

    /// Returns per-column training scales after a successful fit.
    pub fn std(&self) -> Option<&[f64]> {
        if self.std.is_empty() {
            None
        } else {
            Some(self.std.as_slice())
        }
    }

    fn n_cols(&self) -> Option<usize> {
        if self.mean.is_empty() {
            None
        } else {
            Some(self.mean.len())
        }
    }
}

impl Default for StandardizeInput {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform for StandardizeInput {
    fn fit(&mut self, x: &[f64], n_rows: usize, n_cols: usize) -> Result<(), GprError> {
        require_pack(x, n_rows, n_cols)?;
        require_finite(x)?;
        self.mean = Vec::with_capacity(n_cols);
        self.std = Vec::with_capacity(n_cols);
        for col in 0..n_cols {
            let start = col * n_rows;
            let column = &x[start..start + n_rows];
            let mean = column.iter().sum::<f64>() / n_rows as f64;
            self.mean.push(mean);
            self.std.push(population_std(column, mean));
        }
        Ok(())
    }

    fn apply(&self, x: &mut [f64], n_rows: usize, n_cols: usize) -> Result<(), GprError> {
        let expected_cols = self.n_cols().ok_or(GprError::NotFitted)?;
        if n_cols != expected_cols {
            return Err(GprError::DimensionMismatch {
                x_dim: n_cols,
                expected_dim: expected_cols,
            });
        }
        require_nonempty(n_rows)?;
        require_len(x, n_rows * n_cols)?;
        require_finite(x)?;
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
}

#[cfg(test)]
mod tests {
    use super::{IdentityInput, StandardizeInput, Transform};
    use crate::error::GprError;

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
        assert_send_sync::<IdentityInput>();
        assert_send_sync::<StandardizeInput>();
    }

    #[test]
    fn identity_leaves_features() {
        let mut t = IdentityInput;
        let mut x = [1.0, 2.0, 3.0, 4.0];
        t.fit(&x, 2, 2).expect("valid");
        t.apply(&mut x, 2, 2).expect("valid");
        assert_close(x[0], 1.0);
        assert_close(x[3], 4.0);
    }

    #[test]
    fn standardize_each_column() {
        // Column-major 3×2: col0 = [0, 2, 4], col1 = [1, 1, 1]
        let x = [0.0, 2.0, 4.0, 1.0, 1.0, 1.0];
        let mut t = StandardizeInput::new();
        t.fit(&x, 3, 2).expect("valid");
        let mean = t.mean().expect("fitted");
        let std = t.std().expect("fitted");
        assert_close(mean[0], 2.0);
        assert_close(std[0], (8.0 / 3.0_f64).sqrt());
        assert_close(mean[1], 1.0);
        assert_close(std[1], 1.0);
        let mut z = x;
        t.apply(&mut z, 3, 2).expect("fitted");
        let z0 = z[0] + z[1] + z[2];
        assert_close(z0 / 3.0, 0.0);
        assert_close(z[3], 0.0);
        assert_close(z[4], 0.0);
        assert_close(z[5], 0.0);
    }

    #[test]
    fn apply_rejects_column_mismatch() {
        let x = [0.0, 1.0, 2.0, 3.0];
        let mut t = StandardizeInput::new();
        t.fit(&x, 2, 2).expect("valid");
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
    fn standardize_rejects_empty_and_unfitted() {
        let mut t = StandardizeInput::new();
        assert!(matches!(t.fit(&[], 0, 1), Err(GprError::EmptyInput)));
        assert!(matches!(
            t.apply(&mut [1.0], 1, 1),
            Err(GprError::NotFitted)
        ));
        assert!(matches!(
            t.fit(&[1.0, f64::NAN], 1, 2),
            Err(GprError::NonFiniteInput)
        ));
    }
}
