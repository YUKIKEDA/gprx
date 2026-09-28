//! Linear kernel `k(x, x') = σ² xᵀ x'`.

use super::{Triangle, validate_log_positive, validate_positive_finite, write_square};
use crate::error::GprError;
use crate::param::{BoundedParam, Interval};
use faer::{MatMut, MatRef};

/// Linear kernel: `k = σ² xᵀ x'`.
///
/// The optimizer parameter is `θ = log(σ²)`. This is not a lengthscale kernel
/// and has no ARD form here. Add [`super::ConstantKernel`] for an intercept
/// (`σ0² + σ² xᵀ x'`). `apply` takes the `n×d` coordinate matrix.
///
/// # Examples
///
/// ```rust
/// use gprx::kernel::LinearKernel;
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let k = LinearKernel::new(1.0)?;
/// assert!(k.variance() > 0.0);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinearKernel {
    variance: BoundedParam,
}

impl LinearKernel {
    /// Builds a linear kernel from a positive finite variance `σ²`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if `variance` is not finite
    /// or not strictly positive.
    pub fn new(variance: f64) -> Result<Self, GprError> {
        validate_positive_finite(variance, "linear variance")?;
        Ok(Self {
            variance: BoundedParam::default_positive(variance)?,
        })
    }

    /// Builds a linear kernel from `θ = log(σ²)`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if `θ` is not finite, if
    /// `exp(θ)` overflows, or if `exp(θ)` underflows to zero.
    pub fn from_log_variance(log_variance: f64) -> Result<Self, GprError> {
        let log_variance = validate_log_positive(log_variance, "linear variance")?;
        Ok(Self {
            variance: BoundedParam::default_positive(log_variance.exp())?,
        })
    }

    /// Returns `σ² = exp(θ)`.
    pub fn variance(&self) -> f64 {
        self.variance.value()
    }

    /// Returns `θ = log(σ²)`.
    pub fn log_variance(&self) -> f64 {
        self.variance.ln()
    }

    /// Returns the open interval on `σ²`.
    pub fn bounds(&self) -> Interval {
        self.variance.interval()
    }

    /// Rebuilds this kernel with a new interval on `σ²`.
    ///
    /// # Errors
    ///
    /// Returns [`crate::IntervalError`] if the current `σ²` is not strictly
    /// inside `interval`.
    pub fn with_bounds(self, interval: Interval) -> Result<Self, crate::IntervalError> {
        Ok(Self {
            variance: self.variance.with_interval(interval)?,
        })
    }

    /// Returns the number of optimizer parameters (always 1).
    pub fn num_params(&self) -> usize {
        1
    }

    /// Writes `θ` into a length-1 slice.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::LengthMismatch`] if `out` is not length 1.
    pub fn get_params(&self, out: &mut [f64]) -> Result<(), GprError> {
        crate::data::require_count(out.len(), 1, "linear parameter")?;
        out[0] = self.variance.ln();
        Ok(())
    }

    /// Replaces `θ` from a length-1 slice.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::LengthMismatch`] if `params` is not length 1,
    /// or [`GprError::InvalidHyperparameter`] if the new `θ` is invalid.
    pub fn set_params(&mut self, params: &[f64]) -> Result<(), GprError> {
        crate::data::require_count(params.len(), 1, "linear parameter")?;
        let log_variance = validate_log_positive(params[0], "linear variance")?;
        self.variance = BoundedParam::new(log_variance.exp(), self.variance.interval())?;
        Ok(())
    }

    /// Writes `k(x, x')` into `out` for the requested triangle.
    ///
    /// # Errors
    ///
    /// Returns [`GprError`] if `x` is empty, `out` is not `n×n`, or a
    /// coordinate is non-finite.
    pub fn apply(
        &self,
        x: MatRef<'_, f64>,
        out: MatMut<'_, f64>,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        require_square_points(x, out.as_ref())?;
        crate::data::require_finite_points(x)?;
        let var = self.variance();
        let d = x.ncols();
        write_square(out, uplo, |row, col| Ok(var * dot_at(x, row, x, col, d)?))
    }

    /// Writes rectangular `k(x, xs)` into `out`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError`] if a matrix is empty, feature dimensions differ,
    /// `out` is the wrong shape, or a coordinate is non-finite.
    pub fn apply_cross(
        &self,
        x: MatRef<'_, f64>,
        xs: MatRef<'_, f64>,
        mut out: MatMut<'_, f64>,
    ) -> Result<(), GprError> {
        require_feature_pair(x, xs)?;
        if out.nrows() != x.nrows() || out.ncols() != xs.nrows() {
            return Err(GprError::ShapeMismatch {
                reason: format!(
                    "output is {}x{}, expected {}x{}",
                    out.nrows(),
                    out.ncols(),
                    x.nrows(),
                    xs.nrows()
                ),
            });
        }
        crate::data::require_finite_points(x)?;
        crate::data::require_finite_points(xs)?;
        let var = self.variance();
        let d = x.ncols();
        for col in 0..xs.nrows() {
            for row in 0..x.nrows() {
                out[(row, col)] = var * dot_at(x, row, xs, col, d)?;
            }
        }
        Ok(())
    }

    /// Writes the diagonal `k(x_i, x_i) = σ² ‖x_i‖²` into `out`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::EmptyInput`] if `x` is empty,
    /// [`GprError::LengthMismatch`] if `out.len()` is not `x.nrows()`,
    /// or [`GprError::NonFiniteInput`] if a coordinate is non-finite.
    pub fn fill_diag_points(&self, x: MatRef<'_, f64>, out: &mut [f64]) -> Result<(), GprError> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(GprError::EmptyInput);
        }
        if out.len() != x.nrows() {
            return Err(GprError::LengthMismatch {
                reason: format!("expected {} diagonal entries, got {}", x.nrows(), out.len()),
            });
        }
        crate::data::require_finite_points(x)?;
        let var = self.variance();
        let d = x.ncols();
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = var * dot_at(x, i, x, i, d)?;
        }
        Ok(())
    }

    /// Writes `∂K/∂θ` for `θ = log(σ²)` (`∂k/∂θ = k`).
    ///
    /// # Errors
    ///
    /// Returns [`GprError::IndexOutOfRange`] if `param_idx` is not 0, or
    /// the same shape / non-finite errors as [`Self::apply`].
    pub fn grad(
        &self,
        x: MatRef<'_, f64>,
        d_k: MatMut<'_, f64>,
        param_idx: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        if param_idx != 0 {
            return Err(GprError::IndexOutOfRange {
                reason: "linear kernel has a single parameter at index 0".to_owned(),
            });
        }
        self.apply(x, d_k, uplo)
    }

    /// Writes `∂²K/∂θ²` for `θ = log(σ²)` (`∂²k/∂θ² = k`).
    ///
    /// # Errors
    ///
    /// Returns [`GprError::IndexOutOfRange`] if `i` or `j` is not 0, or
    /// the same shape / non-finite errors as [`Self::apply`].
    pub fn hess(
        &self,
        x: MatRef<'_, f64>,
        d2_k: MatMut<'_, f64>,
        i: usize,
        j: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        if i != 0 || j != 0 {
            return Err(GprError::IndexOutOfRange {
                reason: format!("linear kernel has a single parameter; got pair ({i}, {j})"),
            });
        }
        self.apply(x, d2_k, uplo)
    }
}

fn dot_at(
    x: MatRef<'_, f64>,
    row: usize,
    xs: MatRef<'_, f64>,
    col: usize,
    d: usize,
) -> Result<f64, GprError> {
    let mut sum = 0.0;
    for dim in 0..d {
        sum += x[(row, dim)] * xs[(col, dim)];
    }
    if sum.is_finite() {
        Ok(sum)
    } else {
        Err(GprError::NonFiniteKernelValue)
    }
}

fn require_feature_pair(x: MatRef<'_, f64>, xs: MatRef<'_, f64>) -> Result<(), GprError> {
    if x.nrows() == 0 || x.ncols() == 0 || xs.nrows() == 0 || xs.ncols() == 0 {
        return Err(GprError::EmptyInput);
    }
    if x.ncols() != xs.ncols() {
        return Err(GprError::DimensionMismatch {
            x_dim: xs.ncols(),
            expected_dim: x.ncols(),
        });
    }
    Ok(())
}

fn require_square_points(x: MatRef<'_, f64>, out: MatRef<'_, f64>) -> Result<usize, GprError> {
    if x.nrows() == 0 || x.ncols() == 0 {
        return Err(GprError::EmptyInput);
    }
    if out.nrows() != x.nrows() || out.ncols() != x.nrows() {
        return Err(GprError::ShapeMismatch {
            reason: format!(
                "output is {}x{}, expected {}x{}",
                out.nrows(),
                out.ncols(),
                x.nrows(),
                x.nrows()
            ),
        });
    }
    Ok(x.nrows())
}

#[cfg(test)]
mod tests {
    use super::LinearKernel;
    use crate::error::GprError;
    use crate::kernel::Triangle;

    const TOL: f64 = 1e-8;

    use crate::test_check::{assert_close, assert_lower_close, fill, points_2d};

    #[test]
    fn known_inner_products() {
        let kernel = LinearKernel::new(2.0).expect("valid");
        let x = points_2d(&[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]);
        let mut k = fill(3, 0.0);
        kernel
            .apply(x.as_ref(), k.as_mut(), Triangle::Full)
            .expect("shape");
        assert_close(k[(0, 0)], 2.0, TOL);
        assert_close(k[(1, 0)], 0.0, TOL);
        assert_close(k[(2, 0)], 2.0, TOL);
        assert_close(k[(2, 1)], 2.0, TOL);
        assert_close(k[(2, 2)], 4.0, TOL);
        for col in 0..3 {
            for row in 0..3 {
                assert_close(k[(row, col)], k[(col, row)], TOL);
            }
        }
    }

    #[test]
    fn lower_matches_full_and_leaves_upper() {
        let kernel = LinearKernel::new(1.0).expect("valid");
        let x = points_2d(&[[0.5, -1.0], [1.0, 0.2], [2.0, 0.0]]);
        let mut full = fill(3, 0.0);
        kernel
            .apply(x.as_ref(), full.as_mut(), Triangle::Full)
            .expect("shape");
        let mut lower = fill(3, 42.0);
        kernel
            .apply(x.as_ref(), lower.as_mut(), Triangle::Lower)
            .expect("shape");
        assert_lower_close(lower.as_ref(), full.as_ref(), TOL);
        assert_close(lower[(0, 1)], 42.0, TOL);
    }

    #[test]
    fn grad_matches_finite_difference() {
        let kernel = LinearKernel::from_log_variance(-0.4).expect("valid");
        let theta = kernel.log_variance();
        let h = 1e-6;
        let plus = LinearKernel::from_log_variance(theta + h).expect("valid");
        let minus = LinearKernel::from_log_variance(theta - h).expect("valid");
        let x = points_2d(&[[0.0, 1.0], [1.2, -0.3], [0.4, 0.8]]);
        let mut kp = fill(3, 0.0);
        let mut km = fill(3, 0.0);
        let mut dk = fill(3, 0.0);
        plus.apply(x.as_ref(), kp.as_mut(), Triangle::Full)
            .expect("shape");
        minus
            .apply(x.as_ref(), km.as_mut(), Triangle::Full)
            .expect("shape");
        kernel
            .grad(x.as_ref(), dk.as_mut(), 0, Triangle::Full)
            .expect("idx 0");
        for col in 0..3 {
            for row in 0..3 {
                let fd = (kp[(row, col)] - km[(row, col)]) / (2.0 * h);
                assert_close(dk[(row, col)], fd, TOL);
            }
        }
    }

    #[test]
    fn fill_diag_points_matches_square() {
        let kernel = LinearKernel::new(0.5).expect("valid");
        let x = points_2d(&[[1.0, 2.0], [0.0, 3.0]]);
        let mut k = fill(2, 0.0);
        kernel
            .apply(x.as_ref(), k.as_mut(), Triangle::Full)
            .expect("shape");
        let mut diag = [0.0, 0.0];
        kernel
            .fill_diag_points(x.as_ref(), &mut diag)
            .expect("diag");
        assert_close(diag[0], k[(0, 0)], TOL);
        assert_close(diag[1], k[(1, 1)], TOL);
    }

    #[test]
    fn get_set_params_roundtrip() {
        let mut kernel = LinearKernel::new(2.0).expect("valid");
        let mut params = [0.0];
        kernel.get_params(&mut params).expect("len 1");
        assert_close(params[0], 2.0_f64.ln(), TOL);
        params[0] = 0.25_f64.ln();
        kernel.set_params(&params).expect("len 1");
        assert_close(kernel.variance(), 0.25, TOL);
    }

    #[test]
    fn rejects_non_positive_and_bad_index() {
        assert!(matches!(
            LinearKernel::new(0.0),
            Err(GprError::InvalidHyperparameter { .. })
        ));
        let kernel = LinearKernel::new(1.0).expect("valid");
        let x = points_2d(&[[0.0, 1.0], [1.0, 0.0]]);
        let mut dk = fill(2, 0.0);
        assert!(matches!(
            kernel.grad(x.as_ref(), dk.as_mut(), 1, Triangle::Lower),
            Err(GprError::IndexOutOfRange { .. })
        ));
    }
}
