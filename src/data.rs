//! Boundary checks and column-major packing for caller data.
//!
//! Every public entry point that receives raw `f64` slices validates them
//! here. Past this boundary the types are trusted (`.cursor/rules/types.mdc`).

use faer::{Mat, MatMut, MatRef};

use crate::error::GprError;
use crate::kernel::KernelScalar;

/// Returns `n_rows · n_cols` for column-major data.
///
/// # Errors
///
/// Returns [`GprError::EmptyInput`] if either side is zero, or
/// [`GprError::SizeOverflow`] if the product overflows `usize`.
pub(crate) fn column_major_len(n_rows: usize, n_cols: usize) -> Result<usize, GprError> {
    require_nonempty(n_rows)?;
    require_nonempty(n_cols)?;
    n_rows.checked_mul(n_cols).ok_or(GprError::SizeOverflow)
}

/// Rejects a zero count with [`GprError::EmptyInput`].
pub(crate) fn require_nonempty(n: usize) -> Result<(), GprError> {
    if n == 0 {
        Err(GprError::EmptyInput)
    } else {
        Ok(())
    }
}

/// Rejects `NaN` / `Inf` with [`GprError::NonFiniteInput`].
pub(crate) fn require_finite(values: &[f64]) -> Result<(), GprError> {
    if values.iter().any(|value| !value.is_finite()) {
        Err(GprError::NonFiniteInput)
    } else {
        Ok(())
    }
}

/// Rejects a `NaN` / `Inf` coordinate with [`GprError::NonFiniteInput`].
pub(crate) fn require_finite_points<T: KernelScalar>(x: MatRef<'_, T>) -> Result<(), GprError> {
    for col in 0..x.ncols() {
        for row in 0..x.nrows() {
            if !x[(row, col)].is_finite() {
                return Err(GprError::NonFiniteInput);
            }
        }
    }
    Ok(())
}

/// Checks a count: `expected {expected} {what}, got {actual}`.
pub(crate) fn require_count(actual: usize, expected: usize, what: &str) -> Result<(), GprError> {
    if actual == expected {
        Ok(())
    } else {
        Err(GprError::LengthMismatch {
            reason: format!("expected {expected} {what}, got {actual}"),
        })
    }
}

pub(crate) fn validate_training(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    y: &[f64],
) -> Result<(), GprError> {
    let expected_x = column_major_len(n_rows, n_cols)?;
    if x.len() != expected_x {
        return Err(GprError::LengthMismatch {
            reason: format!("expected {expected_x} feature values, got {}", x.len()),
        });
    }
    if y.len() != n_rows {
        return Err(GprError::LengthMismatch {
            reason: format!("expected {n_rows} targets, got {}", y.len()),
        });
    }
    if x.iter().any(|v| !v.is_finite()) || y.iter().any(|v| !v.is_finite()) {
        return Err(GprError::NonFiniteInput);
    }
    Ok(())
}

pub(crate) fn validate_query(xs: &[f64], n_rows: usize, n_cols: usize) -> Result<(), GprError> {
    let expected = column_major_len(n_rows, n_cols)?;
    if xs.len() != expected {
        return Err(GprError::LengthMismatch {
            reason: format!("expected {expected} feature values, got {}", xs.len()),
        });
    }
    if xs.iter().any(|v| !v.is_finite()) {
        return Err(GprError::NonFiniteInput);
    }
    Ok(())
}

pub(crate) fn pack_points(x: &[f64], n_rows: usize, n_cols: usize) -> Mat<f64> {
    let mut dest = Mat::zeros(n_rows, n_cols);
    pack_points_into(x, n_rows, n_cols, dest.as_mut());
    dest
}

pub(crate) fn pack_points_into(x: &[f64], n_rows: usize, n_cols: usize, mut dest: MatMut<'_, f64>) {
    debug_assert_eq!(dest.nrows(), n_rows);
    debug_assert_eq!(dest.ncols(), n_cols);
    for col in 0..n_cols {
        for row in 0..n_rows {
            dest[(row, col)] = x[col * n_rows + row];
        }
    }
}

pub(crate) fn pack_storage<T: KernelScalar>(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    mut dest: MatMut<'_, T>,
) {
    debug_assert_eq!(dest.nrows(), n_rows);
    debug_assert_eq!(dest.ncols(), n_cols);
    for col in 0..n_cols {
        for row in 0..n_rows {
            dest[(row, col)] = T::from_f64(x[col * n_rows + row]);
        }
    }
}

pub(crate) fn validate_inducing(z: &[f64], m: usize, d: usize) -> Result<(), GprError> {
    if m == 0 || d == 0 {
        return Err(GprError::EmptyInput);
    }
    if z.len().is_multiple_of(m) {
        let z_dim = z.len() / m;
        if z_dim != d {
            return Err(GprError::DimensionMismatch {
                x_dim: z_dim,
                expected_dim: d,
            });
        }
    }
    let expected = m.checked_mul(d).ok_or(GprError::SizeOverflow)?;
    if z.len() != expected {
        return Err(GprError::LengthMismatch {
            reason: format!(
                "expected {expected} inducing feature values, got {}",
                z.len()
            ),
        });
    }
    if z.iter().any(|v| !v.is_finite()) {
        return Err(GprError::NonFiniteInput);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{column_major_len, require_count};
    use crate::error::GprError;
    use crate::transform::{MinMaxInput, StandardizeInput, Transform};

    #[test]
    fn column_major_len_rejects_zero_and_overflow() {
        assert_eq!(column_major_len(3, 2), Ok(6));
        assert_eq!(column_major_len(0, 2), Err(GprError::EmptyInput));
        assert_eq!(column_major_len(3, 0), Err(GprError::EmptyInput));
        assert_eq!(column_major_len(usize::MAX, 2), Err(GprError::SizeOverflow));
    }

    #[test]
    fn require_count_keeps_the_message() {
        let err = require_count(2, 3, "kernel parameters").err();
        assert_eq!(
            err,
            Some(GprError::LengthMismatch {
                reason: "expected 3 kernel parameters, got 2".to_owned(),
            })
        );
    }

    #[test]
    fn fitted_input_maps_reject_an_overflowing_shape() {
        let x = [0.0, 1.0, 2.0, 3.0];
        let standardize = StandardizeInput::new().fit(&x, 2, 2).expect("fit");
        let min_max = MinMaxInput::new().fit(&x, 2, 2).expect("fit");
        let mut buf = x;
        assert_eq!(
            standardize.apply(&mut buf, usize::MAX, 2),
            Err(GprError::SizeOverflow)
        );
        assert_eq!(
            min_max.apply(&mut buf, usize::MAX, 2),
            Err(GprError::SizeOverflow)
        );
    }
}
