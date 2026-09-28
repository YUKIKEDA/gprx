//! Constant kernel `k(x, x') = c`.

use super::{
    Triangle, require_same_shape, require_square_pair, validate_log_positive,
    validate_positive_finite, write_square,
};
use crate::error::GprError;
use crate::param::{BoundedParam, Interval};
use faer::{MatMut, MatRef};

/// Constant kernel: `k = c` for every pair of points.
///
/// The optimizer parameter is `θ = log(c)`. Compose with a stationary leaf
/// (for example [`super::RbfKernel`]) to set the signal variance. `dist` is
/// used only for shape checks.
///
/// # Examples
///
/// ```rust
/// use gprx::kernel::ConstantKernel;
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let k = ConstantKernel::new(1.5)?;
/// assert!(k.constant() > 0.0);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConstantKernel {
    constant: BoundedParam,
}

impl ConstantKernel {
    /// Builds a constant kernel from a positive finite value `c`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if `constant` is not finite
    /// or not strictly positive.
    pub fn new(constant: f64) -> Result<Self, GprError> {
        validate_positive_finite(constant, "constant value")?;
        Ok(Self {
            constant: BoundedParam::default_positive(constant)?,
        })
    }

    /// Builds a constant kernel from `θ = log(c)`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if `θ` is not finite, if
    /// `exp(θ)` overflows, or if `exp(θ)` underflows to zero.
    pub fn from_log_constant(log_constant: f64) -> Result<Self, GprError> {
        let log_constant = validate_log_positive(log_constant, "constant value")?;
        Ok(Self {
            constant: BoundedParam::default_positive(log_constant.exp())?,
        })
    }

    /// Returns `c = exp(θ)`.
    pub fn constant(&self) -> f64 {
        self.constant.value()
    }

    /// Returns `θ = log(c)`.
    pub fn log_constant(&self) -> f64 {
        self.constant.ln()
    }

    /// Returns the open interval on `c`.
    pub fn bounds(&self) -> Interval {
        self.constant.interval()
    }

    /// Rebuilds this kernel with a new interval on `c`.
    ///
    /// # Errors
    ///
    /// Returns [`crate::IntervalError`] if the current `c` is not strictly
    /// inside `interval`.
    pub fn with_bounds(self, interval: Interval) -> Result<Self, crate::IntervalError> {
        Ok(Self {
            constant: self.constant.with_interval(interval)?,
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
    /// Returns [`GprError::InvalidHyperparameter`] if `out` is not length 1.
    pub fn get_params(&self, out: &mut [f64]) -> Result<(), GprError> {
        crate::data::require_count(out.len(), 1, "constant parameter")?;
        out[0] = self.constant.ln();
        Ok(())
    }

    /// Replaces `θ` from a length-1 slice.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if `params` is not length 1
    /// or if the new `θ` is invalid.
    pub fn set_params(&mut self, params: &[f64]) -> Result<(), GprError> {
        crate::data::require_count(params.len(), 1, "constant parameter")?;
        let log_constant = validate_log_positive(params[0], "constant value")?;
        self.constant = BoundedParam::new(log_constant.exp(), self.constant.interval())?;
        Ok(())
    }

    /// Writes `c` into `out` for the requested triangle.
    ///
    /// # Errors
    ///
    /// Returns [`GprError`] if the matrices are empty, not square, or size
    /// mismatched.
    pub fn apply(
        &self,
        dist: MatRef<'_, f64>,
        out: MatMut<'_, f64>,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        require_square_pair(dist, out.as_ref())?;
        self.write_square(out, uplo)
    }

    /// Writes rectangular `k = c` into `out`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError`] if the matrices are empty or size mismatched.
    pub fn apply_cross(
        &self,
        dist: MatRef<'_, f64>,
        mut out: MatMut<'_, f64>,
    ) -> Result<(), GprError> {
        require_same_shape(dist, out.as_ref())?;
        let c = self.constant();
        for col in 0..out.ncols() {
            for row in 0..out.nrows() {
                out[(row, col)] = c;
            }
        }
        Ok(())
    }

    /// Writes `c` from the number of rows of `x` into a square `out`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError`] if `x` is empty or `out` is not `n×n`.
    pub fn apply_points(
        &self,
        x: MatRef<'_, f64>,
        out: MatMut<'_, f64>,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        require_points_square(x, out.as_ref())?;
        self.write_square(out, uplo)
    }

    /// Writes rectangular `k = c` for train × test coordinates.
    ///
    /// # Errors
    ///
    /// Returns [`GprError`] if a matrix is empty or `out` is the wrong shape.
    pub fn apply_cross_points(
        &self,
        x: MatRef<'_, f64>,
        xs: MatRef<'_, f64>,
        mut out: MatMut<'_, f64>,
    ) -> Result<(), GprError> {
        require_cross_points(x, xs, out.as_ref())?;
        let c = self.constant();
        for col in 0..out.ncols() {
            for row in 0..out.nrows() {
                out[(row, col)] = c;
            }
        }
        Ok(())
    }

    /// Writes the diagonal `k(x, x) = c` into `out`.
    pub fn fill_diag(&self, out: &mut [f64]) {
        out.fill(self.constant());
    }

    /// Writes `∂K/∂θ` for `θ = log(c)` into `d_k` (`∂k/∂θ = c`).
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if `param_idx` is not 0, or
    /// the same shape errors as [`Self::apply`].
    pub fn grad(
        &self,
        dist: MatRef<'_, f64>,
        d_k: MatMut<'_, f64>,
        param_idx: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        require_param_idx(param_idx)?;
        require_square_pair(dist, d_k.as_ref())?;
        self.write_square(d_k, uplo)
    }

    /// Writes `∂K/∂θ` from coordinates. `x` is used only for shape.
    ///
    /// # Errors
    ///
    /// Same as [`Self::grad`], with `x` in place of `dist`.
    pub fn grad_points(
        &self,
        x: MatRef<'_, f64>,
        d_k: MatMut<'_, f64>,
        param_idx: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        require_param_idx(param_idx)?;
        require_points_square(x, d_k.as_ref())?;
        self.write_square(d_k, uplo)
    }

    /// Writes `∂²K/∂θ²` for `θ = log(c)` into `d2_k` (`∂²k/∂θ² = c`).
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if `i` or `j` is not 0, or
    /// the same shape errors as [`Self::apply`].
    pub fn hess(
        &self,
        dist: MatRef<'_, f64>,
        d2_k: MatMut<'_, f64>,
        i: usize,
        j: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        require_hess_idx(i, j)?;
        require_square_pair(dist, d2_k.as_ref())?;
        self.write_square(d2_k, uplo)
    }

    /// Writes `∂²K/∂θ²` from coordinates. `x` is used only for shape.
    ///
    /// # Errors
    ///
    /// Same as [`Self::hess`], with `x` in place of `dist`.
    pub fn hess_points(
        &self,
        x: MatRef<'_, f64>,
        d2_k: MatMut<'_, f64>,
        i: usize,
        j: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        require_hess_idx(i, j)?;
        require_points_square(x, d2_k.as_ref())?;
        self.write_square(d2_k, uplo)
    }

    fn write_square(&self, out: MatMut<'_, f64>, uplo: Triangle) -> Result<(), GprError> {
        let c = self.constant();
        write_square(out, uplo, |_, _| Ok(c))
    }
}

fn require_param_idx(param_idx: usize) -> Result<(), GprError> {
    if param_idx == 0 {
        Ok(())
    } else {
        Err(GprError::InvalidHyperparameter {
            reason: "constant kernel has a single parameter at index 0".to_owned(),
        })
    }
}

fn require_hess_idx(i: usize, j: usize) -> Result<(), GprError> {
    if i == 0 && j == 0 {
        Ok(())
    } else {
        Err(GprError::InvalidHyperparameter {
            reason: format!("constant kernel has a single parameter; got pair ({i}, {j})"),
        })
    }
}

fn require_points_square(x: MatRef<'_, f64>, out: MatRef<'_, f64>) -> Result<(), GprError> {
    if x.nrows() == 0 || x.ncols() == 0 {
        return Err(GprError::EmptyInput);
    }
    if out.nrows() != x.nrows() || out.ncols() != x.nrows() {
        return Err(GprError::InvalidHyperparameter {
            reason: format!(
                "output is {}x{}, expected {}x{}",
                out.nrows(),
                out.ncols(),
                x.nrows(),
                x.nrows()
            ),
        });
    }
    Ok(())
}

fn require_cross_points(
    x: MatRef<'_, f64>,
    xs: MatRef<'_, f64>,
    out: MatRef<'_, f64>,
) -> Result<(), GprError> {
    if x.nrows() == 0 || x.ncols() == 0 || xs.nrows() == 0 || xs.ncols() == 0 {
        return Err(GprError::EmptyInput);
    }
    if out.nrows() != x.nrows() || out.ncols() != xs.nrows() {
        return Err(GprError::InvalidHyperparameter {
            reason: format!(
                "output is {}x{}, expected {}x{}",
                out.nrows(),
                out.ncols(),
                x.nrows(),
                xs.nrows()
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::ConstantKernel;
    use crate::error::GprError;
    use crate::kernel::Triangle;
    use faer::{Mat, MatRef, mat};

    const TOL: f64 = 1e-8;

    fn assert_close(actual: f64, expected: f64) {
        let scale = expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= TOL * scale,
            "actual={actual}, expected={expected}"
        );
    }

    fn fill(n: usize, value: f64) -> Mat<f64> {
        Mat::from_fn(n, n, |_, _| value)
    }

    fn dummy_dist(n: usize) -> Mat<f64> {
        Mat::from_fn(n, n, |i, j| ((i + j) as f64) * 0.1)
    }

    fn lower_matches(actual: MatRef<'_, f64>, expected: MatRef<'_, f64>) {
        let n = actual.nrows();
        for col in 0..n {
            for row in col..n {
                assert_close(actual[(row, col)], expected[(row, col)]);
            }
        }
    }

    #[test]
    fn apply_is_constant_and_symmetric() {
        let kernel = ConstantKernel::new(2.5).expect("valid");
        let dist = dummy_dist(3);
        let mut k = fill(3, 0.0);
        kernel
            .apply(dist.as_ref(), k.as_mut(), Triangle::Full)
            .expect("shape");
        for col in 0..3 {
            for row in 0..3 {
                assert_close(k[(row, col)], 2.5);
                assert_close(k[(row, col)], k[(col, row)]);
            }
        }
    }

    #[test]
    fn lower_matches_full_and_leaves_upper() {
        let kernel = ConstantKernel::new(0.75).expect("valid");
        let dist = dummy_dist(3);
        let mut full = fill(3, 0.0);
        kernel
            .apply(dist.as_ref(), full.as_mut(), Triangle::Full)
            .expect("shape");
        let mut lower = fill(3, 42.0);
        kernel
            .apply(dist.as_ref(), lower.as_mut(), Triangle::Lower)
            .expect("shape");
        lower_matches(lower.as_ref(), full.as_ref());
        assert_close(lower[(0, 1)], 42.0);
    }

    #[test]
    fn grad_matches_finite_difference() {
        let kernel = ConstantKernel::from_log_constant(-0.2).expect("valid");
        let theta = kernel.log_constant();
        let h = 1e-6;
        let plus = ConstantKernel::from_log_constant(theta + h).expect("valid");
        let minus = ConstantKernel::from_log_constant(theta - h).expect("valid");
        let dist = dummy_dist(2);
        let mut kp = fill(2, 0.0);
        let mut km = fill(2, 0.0);
        let mut dk = fill(2, 0.0);
        plus.apply(dist.as_ref(), kp.as_mut(), Triangle::Full)
            .expect("shape");
        minus
            .apply(dist.as_ref(), km.as_mut(), Triangle::Full)
            .expect("shape");
        kernel
            .grad(dist.as_ref(), dk.as_mut(), 0, Triangle::Full)
            .expect("idx 0");
        let fd = (kp[(0, 1)] - km[(0, 1)]) / (2.0 * h);
        assert_close(dk[(0, 1)], fd);
        assert_close(dk[(0, 1)], kernel.constant());
    }

    #[test]
    fn get_set_params_roundtrip() {
        let mut kernel = ConstantKernel::new(2.0).expect("valid");
        let mut params = [0.0];
        kernel.get_params(&mut params).expect("len 1");
        assert_close(params[0], 2.0_f64.ln());
        params[0] = 0.5_f64.ln();
        kernel.set_params(&params).expect("len 1");
        assert_close(kernel.constant(), 0.5);
    }

    #[test]
    fn apply_cross_and_diag() {
        let kernel = ConstantKernel::new(3.0).expect("valid");
        let dist = mat![[0.0, 1.0], [4.0, 0.0]];
        let mut out = Mat::zeros(2, 2);
        kernel
            .apply_cross(dist.as_ref(), out.as_mut())
            .expect("rect");
        assert_close(out[(0, 1)], 3.0);
        let mut diag = [0.0, 0.0];
        kernel.fill_diag(&mut diag);
        assert_close(diag[0], 3.0);
    }

    #[test]
    fn rejects_non_positive() {
        assert!(matches!(
            ConstantKernel::new(0.0),
            Err(GprError::InvalidHyperparameter { .. })
        ));
        let kernel = ConstantKernel::new(1.0).expect("valid");
        let dist = dummy_dist(2);
        let mut dk = fill(2, 0.0);
        assert!(matches!(
            kernel.grad(dist.as_ref(), dk.as_mut(), 1, Triangle::Lower),
            Err(GprError::InvalidHyperparameter { .. })
        ));
    }
}
