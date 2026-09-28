//! ARD rational quadratic kernel.

use super::rq::{finite_kernel, rq_d2k_ard, rq_dk_dtheta_alpha, rq_dk_dtheta_ard_dim, rq_from_r2};
use super::{
    ArdLengthscales, Triangle, validate_log_positive, validate_positive_finite, visit_triangle,
};
use crate::error::GprError;
use crate::param::{BoundedParam, Interval};
use faer::{MatMut, MatRef};

/// ARD rational quadratic: `k = (1 + r² / (2α))^(-α)` with
/// `r² = Σ_d (x_d-x'_d)² / ℓ_d²`.
///
/// Optimizer parameters are `[log(ℓ_1), …, log(ℓ_d), log(α)]` via
/// [`ArdLengthscales`] plus a scalar `α`. When every `ℓ_d` equals a scalar `ℓ`
/// and `α` matches, values match isotropic
/// [`super::RationalQuadraticKernel`]. Amplitude is not stored here.
///
/// Cloning copies the lengthscale vectors. When
/// [`crate::CachedDistances`] is set, [`crate::Gpr`] caches raw
/// `(Δx_d)²`.
///
/// # Examples
///
/// ```rust
/// use gprx::kernel::RationalQuadraticArdKernel;
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let k = RationalQuadraticArdKernel::new(&[1.0, 2.5], 1.5)?;
/// assert_eq!(k.num_params(), 3);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct RationalQuadraticArdKernel {
    lengthscales: ArdLengthscales,
    alpha: BoundedParam,
}

impl RationalQuadraticArdKernel {
    /// Builds an ARD RQ kernel from positive finite `ℓ_d` and `α`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if the slice is empty or a
    /// value is invalid.
    pub fn new(lengthscales: &[f64], alpha: f64) -> Result<Self, GprError> {
        validate_positive_finite(alpha, "alpha")?;
        Ok(Self {
            lengthscales: ArdLengthscales::new(lengthscales)?,
            alpha: BoundedParam::default_positive(alpha)?,
        })
    }

    /// Builds an ARD RQ kernel from `θ_d = log(ℓ_d)` and `θ_α = log(α)`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if the slice is empty or a
    /// `θ` is invalid.
    pub fn from_log_lengthscales(
        log_lengthscales: &[f64],
        log_alpha: f64,
    ) -> Result<Self, GprError> {
        Ok(Self {
            lengthscales: ArdLengthscales::from_log_lengthscales(log_lengthscales)?,
            alpha: BoundedParam::default_positive(
                validate_log_positive(log_alpha, "alpha")?.exp(),
            )?,
        })
    }

    /// Returns `α = exp(θ_α)`.
    pub fn alpha(&self) -> f64 {
        self.alpha.value()
    }

    /// Returns `θ_α = log(α)`.
    pub fn log_alpha(&self) -> f64 {
        self.alpha.ln()
    }

    /// Returns the open interval on `α`.
    pub fn alpha_bounds(&self) -> Interval {
        self.alpha.interval()
    }

    /// Rebuilds this kernel with new intervals on every `ℓ_d` and on `α`.
    ///
    /// # Errors
    ///
    /// Returns [`crate::IntervalError`] if a current value is not strictly
    /// inside the matching interval.
    pub fn with_bounds(
        self,
        lengthscale: Interval,
        alpha: Interval,
    ) -> Result<Self, crate::IntervalError> {
        Ok(Self {
            lengthscales: self.lengthscales.with_bounds(lengthscale)?,
            alpha: self.alpha.with_interval(alpha)?,
        })
    }

    /// Returns the shared ARD lengthscale mouth.
    pub fn lengthscales(&self) -> &ArdLengthscales {
        &self.lengthscales
    }

    pub(crate) fn from_ard(lengthscales: ArdLengthscales, alpha: BoundedParam) -> Self {
        Self {
            lengthscales,
            alpha,
        }
    }

    /// Returns `ℓ_d` for feature `dim`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if `dim` is out of range.
    pub fn lengthscale(&self, dim: usize) -> Result<f64, GprError> {
        self.lengthscales.lengthscale(dim)
    }

    /// Returns `θ_d = log(ℓ_d)`.
    pub fn log_lengthscales(&self) -> &[f64] {
        self.lengthscales.log_lengthscales()
    }

    /// Returns the number of optimizer parameters (`d + 1`).
    pub fn num_params(&self) -> usize {
        self.lengthscales.num_params() + 1
    }

    /// Writes `[log(ℓ_d)…, log(α)]` into `out`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if `out` is the wrong length.
    pub fn get_params(&self, out: &mut [f64]) -> Result<(), GprError> {
        let d = self.lengthscales.num_params();
        if out.len() != d + 1 {
            return Err(GprError::InvalidHyperparameter {
                reason: format!(
                    "expected {} rational quadratic parameters, got {}",
                    d + 1,
                    out.len()
                ),
            });
        }
        self.lengthscales.get_params(&mut out[..d])?;
        out[d] = self.alpha.ln();
        Ok(())
    }

    /// Replaces `[log(ℓ_d)…, log(α)]`. Previous values are kept on error.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if `params` is the wrong
    /// length or a `θ` is invalid.
    pub fn set_params(&mut self, params: &[f64]) -> Result<(), GprError> {
        let d = self.lengthscales.num_params();
        if params.len() != d + 1 {
            return Err(GprError::InvalidHyperparameter {
                reason: format!(
                    "expected {} rational quadratic parameters, got {}",
                    d + 1,
                    params.len()
                ),
            });
        }
        let mut lengthscales = self.lengthscales.clone();
        lengthscales.set_params(&params[..d])?;
        let log_alpha = validate_log_positive(params[d], "alpha")?;
        let alpha = BoundedParam::new(log_alpha.exp(), self.alpha.interval())?;
        self.lengthscales = lengthscales;
        self.alpha = alpha;
        Ok(())
    }

    /// Writes `k(x, x')` into `out` for the requested triangle.
    ///
    /// `x` is `n×d` (rows are points). The default contract is
    /// [`Triangle::Lower`]. Entries outside that triangle are left unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`GprError`] if `x` is empty, `d` does not match the
    /// lengthscales, `out` is not `n×n`, or a coordinate is non-finite.
    pub fn apply(
        &self,
        x: MatRef<'_, f64>,
        mut out: MatMut<'_, f64>,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        let n = require_square_points(x, out.as_ref(), self.lengthscales.num_params())?;
        let inv_ell_sq = self.lengthscales.inv_ell_sq();
        let alpha = self.alpha();
        let mut err = None;
        visit_triangle(n, uplo, |row, col| {
            if err.is_some() {
                return;
            }
            match ard_kernel(x, row, col, inv_ell_sq, alpha) {
                Ok(value) => out[(row, col)] = value,
                Err(e) => err = Some(e),
            }
        });
        match err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// Writes rectangular `k(x, xs)` (train × test) into `out`.
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
        let d = self.lengthscales.num_params();
        require_feature_dim(x, d)?;
        require_feature_dim(xs, d)?;
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
        crate::data::require_finite_points(x)?;
        crate::data::require_finite_points(xs)?;
        let inv_ell_sq = self.lengthscales.inv_ell_sq();
        let alpha = self.alpha();
        for col in 0..xs.nrows() {
            for row in 0..x.nrows() {
                out[(row, col)] = ard_kernel_pair(x, row, xs, col, inv_ell_sq, alpha)?;
            }
        }
        Ok(())
    }

    /// Writes the stationary diagonal `k(x, x) = 1` into `out`.
    pub fn fill_diag(&self, out: &mut [f64]) {
        out.fill(1.0);
    }

    /// Writes `∂K/∂θ` into `d_k`. Indices `0..d` are `log(ℓ_d)`; index `d` is
    /// `log(α)`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if `param_idx` is out of
    /// range, or the same shape / non-finite errors as [`Self::apply`].
    pub fn grad(
        &self,
        x: MatRef<'_, f64>,
        mut d_k: MatMut<'_, f64>,
        param_idx: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        let d = self.lengthscales.num_params();
        if param_idx > d {
            return Err(GprError::InvalidHyperparameter {
                reason: format!(
                    "ARD rational quadratic parameter index {param_idx} is out of range (d={d})"
                ),
            });
        }
        let n = require_square_points(x, d_k.as_ref(), d)?;
        let inv_ell_sq = self.lengthscales.inv_ell_sq();
        let alpha = self.alpha();
        let mut err = None;
        visit_triangle(n, uplo, |row, col| {
            if err.is_some() {
                return;
            }
            match ard_kernel_grad(x, row, col, inv_ell_sq, param_idx, d, alpha) {
                Ok(value) => d_k[(row, col)] = value,
                Err(e) => err = Some(e),
            }
        });
        match err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    pub(crate) fn apply_from_sq_diff(
        &self,
        cache: MatRef<'_, f64>,
        mut out: MatMut<'_, f64>,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        let n = out.nrows();
        if out.ncols() != n {
            return Err(GprError::InvalidHyperparameter {
                reason: format!("output is {}x{}, expected square", out.nrows(), out.ncols()),
            });
        }
        let d = self.lengthscales.num_params();
        super::dist::require_ard_sq_diff_shape(cache, n, d)?;
        let inv_ell_sq = self.lengthscales.inv_ell_sq();
        let alpha = self.alpha();
        let mut err = None;
        visit_triangle(n, uplo, |row, col| {
            if err.is_some() {
                return;
            }
            match super::dist::weighted_r2_from_cache(cache, n, row, col, inv_ell_sq, None)
                .and_then(|(r2, _)| finite_kernel(rq_from_r2(r2.max(0.0), alpha)))
            {
                Ok(value) => out[(row, col)] = value,
                Err(e) => err = Some(e),
            }
        });
        match err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    pub(crate) fn grad_from_sq_diff(
        &self,
        cache: MatRef<'_, f64>,
        mut d_k: MatMut<'_, f64>,
        param_idx: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        let d = self.lengthscales.num_params();
        if param_idx > d {
            return Err(GprError::InvalidHyperparameter {
                reason: format!(
                    "ARD rational quadratic parameter index {param_idx} is out of range (d={d})"
                ),
            });
        }
        let n = d_k.nrows();
        if d_k.ncols() != n {
            return Err(GprError::InvalidHyperparameter {
                reason: format!("output is {}x{}, expected square", d_k.nrows(), d_k.ncols()),
            });
        }
        super::dist::require_ard_sq_diff_shape(cache, n, d)?;
        let inv_ell_sq = self.lengthscales.inv_ell_sq();
        let alpha = self.alpha();
        let mut err = None;
        visit_triangle(n, uplo, |row, col| {
            if err.is_some() {
                return;
            }
            match super::dist::weighted_r2_from_cache(
                cache,
                n,
                row,
                col,
                inv_ell_sq,
                if param_idx < d { Some(param_idx) } else { None },
            )
            .and_then(|(r2, dim_term)| {
                let r2 = r2.max(0.0);
                let dk = if param_idx == d {
                    rq_dk_dtheta_alpha(r2, alpha)
                } else {
                    rq_dk_dtheta_ard_dim(r2, alpha, dim_term)
                };
                finite_kernel(dk)
            }) {
                Ok(value) => d_k[(row, col)] = value,
                Err(e) => err = Some(e),
            }
        });
        match err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// Writes `∂²K/∂θ_i ∂θ_j`. Indices `0..d` are `log(ℓ_d)`; index `d` is
    /// `log(α)`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if `i` or `j` is out of
    /// range, or the same shape / non-finite errors as [`Self::apply`].
    pub fn hess(
        &self,
        x: MatRef<'_, f64>,
        mut d2_k: MatMut<'_, f64>,
        i: usize,
        j: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        let d = self.lengthscales.num_params();
        if i > d || j > d {
            return Err(GprError::InvalidHyperparameter {
                reason: format!(
                    "ARD rational quadratic parameter pair ({i}, {j}) is out of range (d={d})"
                ),
            });
        }
        let n = require_square_points(x, d2_k.as_ref(), d)?;
        let inv_ell_sq = self.lengthscales.inv_ell_sq();
        let alpha = self.alpha();
        let mut err = None;
        visit_triangle(n, uplo, |row, col| {
            if err.is_some() {
                return;
            }
            match ard_kernel_hess(x, row, col, inv_ell_sq, (i, j), d, alpha) {
                Ok(value) => d2_k[(row, col)] = value,
                Err(e) => err = Some(e),
            }
        });
        match err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    pub(crate) fn hess_from_sq_diff(
        &self,
        cache: MatRef<'_, f64>,
        mut d2_k: MatMut<'_, f64>,
        i: usize,
        j: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        let d = self.lengthscales.num_params();
        if i > d || j > d {
            return Err(GprError::InvalidHyperparameter {
                reason: format!(
                    "ARD rational quadratic parameter pair ({i}, {j}) is out of range (d={d})"
                ),
            });
        }
        let n = d2_k.nrows();
        if d2_k.ncols() != n {
            return Err(GprError::InvalidHyperparameter {
                reason: format!(
                    "output is {}x{}, expected square",
                    d2_k.nrows(),
                    d2_k.ncols()
                ),
            });
        }
        super::dist::require_ard_sq_diff_shape(cache, n, d)?;
        let inv_ell_sq = self.lengthscales.inv_ell_sq();
        let alpha = self.alpha();
        let mut err = None;
        visit_triangle(n, uplo, |row, col| {
            if err.is_some() {
                return;
            }
            match ard_hess_from_cache(cache, n, (row, col), inv_ell_sq, (i, j), d, alpha) {
                Ok(value) => d2_k[(row, col)] = value,
                Err(e) => err = Some(e),
            }
        });
        match err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

fn require_feature_dim(x: MatRef<'_, f64>, expected_d: usize) -> Result<(), GprError> {
    if x.nrows() == 0 || x.ncols() == 0 {
        return Err(GprError::EmptyInput);
    }
    if x.ncols() != expected_d {
        return Err(GprError::DimensionMismatch {
            x_dim: x.ncols(),
            expected_dim: expected_d,
        });
    }
    Ok(())
}

fn require_square_points(
    x: MatRef<'_, f64>,
    out: MatRef<'_, f64>,
    expected_d: usize,
) -> Result<usize, GprError> {
    require_feature_dim(x, expected_d)?;
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
    crate::data::require_finite_points(x)?;
    Ok(x.nrows())
}

fn ard_r2_pair(
    x: MatRef<'_, f64>,
    row: usize,
    xs: MatRef<'_, f64>,
    col: usize,
    inv_ell_sq: &[f64],
) -> Result<f64, GprError> {
    let mut r2 = 0.0;
    for (dim, &w) in inv_ell_sq.iter().enumerate() {
        let diff = x[(row, dim)] - xs[(col, dim)];
        if !diff.is_finite() {
            return Err(GprError::NonFiniteInput);
        }
        r2 += diff * diff * w;
    }
    if r2.is_finite() {
        Ok(r2.max(0.0))
    } else {
        Err(GprError::NonFiniteKernelValue)
    }
}

fn ard_kernel_pair(
    x: MatRef<'_, f64>,
    row: usize,
    xs: MatRef<'_, f64>,
    col: usize,
    inv_ell_sq: &[f64],
    alpha: f64,
) -> Result<f64, GprError> {
    let r2 = ard_r2_pair(x, row, xs, col, inv_ell_sq)?;
    finite_kernel(rq_from_r2(r2, alpha))
}

fn ard_kernel(
    x: MatRef<'_, f64>,
    row: usize,
    col: usize,
    inv_ell_sq: &[f64],
    alpha: f64,
) -> Result<f64, GprError> {
    ard_kernel_pair(x, row, x, col, inv_ell_sq, alpha)
}

fn ard_kernel_grad(
    x: MatRef<'_, f64>,
    row: usize,
    col: usize,
    inv_ell_sq: &[f64],
    param_idx: usize,
    d: usize,
    alpha: f64,
) -> Result<f64, GprError> {
    let mut r2 = 0.0;
    let mut dim_term = 0.0;
    for (dim, &w) in inv_ell_sq.iter().enumerate() {
        let diff = x[(row, dim)] - x[(col, dim)];
        if !diff.is_finite() {
            return Err(GprError::NonFiniteInput);
        }
        let term = diff * diff * w;
        r2 += term;
        if dim == param_idx {
            dim_term = term;
        }
    }
    if !r2.is_finite() {
        return Err(GprError::NonFiniteKernelValue);
    }
    let r2 = r2.max(0.0);
    let dk = if param_idx == d {
        rq_dk_dtheta_alpha(r2, alpha)
    } else {
        rq_dk_dtheta_ard_dim(r2, alpha, dim_term)
    };
    finite_kernel(dk)
}

fn ard_kernel_hess(
    x: MatRef<'_, f64>,
    row: usize,
    col: usize,
    inv_ell_sq: &[f64],
    pair: (usize, usize),
    d: usize,
    alpha: f64,
) -> Result<f64, GprError> {
    let (i, j) = pair;
    let mut r2 = 0.0;
    let mut dim_i = 0.0;
    let mut dim_j = 0.0;
    for (dim, &w) in inv_ell_sq.iter().enumerate() {
        let diff = x[(row, dim)] - x[(col, dim)];
        if !diff.is_finite() {
            return Err(GprError::NonFiniteInput);
        }
        let term = diff * diff * w;
        r2 += term;
        if dim == i {
            dim_i = term;
        }
        if dim == j {
            dim_j = term;
        }
    }
    if !r2.is_finite() {
        return Err(GprError::NonFiniteKernelValue);
    }
    finite_kernel(rq_d2k_ard(r2.max(0.0), alpha, dim_i, dim_j, i, j, d))
}

fn ard_hess_from_cache(
    cache: MatRef<'_, f64>,
    n: usize,
    rc: (usize, usize),
    inv_ell_sq: &[f64],
    pair: (usize, usize),
    d: usize,
    alpha: f64,
) -> Result<f64, GprError> {
    let (row, col) = rc;
    let (i, j) = pair;
    let mut r2 = 0.0;
    let mut dim_i = 0.0;
    let mut dim_j = 0.0;
    for (dim, &w) in inv_ell_sq.iter().enumerate() {
        let v = cache[(row, dim * n + col)];
        if !v.is_finite() {
            return Err(GprError::NonFiniteInput);
        }
        let term = v * w;
        r2 += term;
        if dim == i {
            dim_i = term;
        }
        if dim == j {
            dim_j = term;
        }
    }
    if !r2.is_finite() {
        return Err(GprError::NonFiniteKernelValue);
    }
    finite_kernel(rq_d2k_ard(r2.max(0.0), alpha, dim_i, dim_j, i, j, d))
}

#[cfg(test)]
mod tests {
    use super::RationalQuadraticArdKernel;
    use crate::error::GprError;
    use crate::kernel::{RationalQuadraticKernel, Triangle};
    use faer::{Mat, MatRef};

    const TOL: f64 = 1e-8;

    fn assert_close(actual: f64, expected: f64) {
        let scale = expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= TOL * scale,
            "actual={actual}, expected={expected}"
        );
    }

    fn assert_send_sync<T: Send + Sync>() {}

    fn fill(n: usize, value: f64) -> Mat<f64> {
        Mat::from_fn(n, n, |_, _| value)
    }

    fn points_2d(rows: &[[f64; 2]]) -> Mat<f64> {
        Mat::from_fn(rows.len(), 2, |i, j| rows[i][j])
    }

    fn sq_dist(x: MatRef<'_, f64>) -> Mat<f64> {
        let n = x.nrows();
        let d = x.ncols();
        Mat::from_fn(n, n, |row, col| {
            let mut sum = 0.0;
            for dim in 0..d {
                let diff = x[(row, dim)] - x[(col, dim)];
                sum += diff * diff;
            }
            sum
        })
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
    fn is_send_sync() {
        assert_send_sync::<RationalQuadraticArdKernel>();
    }

    #[test]
    fn diagonal_is_one() {
        let kernel = RationalQuadraticArdKernel::new(&[1.0, 2.0], 0.8).expect("valid");
        let x = points_2d(&[[0.0, 0.0], [1.0, 0.5], [0.2, 1.3]]);
        let mut k = fill(3, f64::NAN);
        kernel
            .apply(x.as_ref(), k.as_mut(), Triangle::Full)
            .expect("shape");
        assert_close(k[(0, 0)], 1.0);
        assert_close(k[(1, 1)], 1.0);
        assert_close(k[(2, 2)], 1.0);
    }

    #[test]
    fn equal_lengthscales_match_isotropic() {
        let ell = 1.25;
        let alpha = 0.7;
        let iso = RationalQuadraticKernel::new(ell, alpha).expect("valid");
        let ard = RationalQuadraticArdKernel::new(&[ell, ell], alpha).expect("valid");
        let x = points_2d(&[[0.0, 0.0], [0.8, -0.4], [1.5, 0.2]]);
        let dist = sq_dist(x.as_ref());
        let mut k_iso = fill(3, 0.0);
        let mut k_ard = fill(3, 0.0);
        iso.apply(dist.as_ref(), k_iso.as_mut(), Triangle::Full)
            .expect("iso");
        ard.apply(x.as_ref(), k_ard.as_mut(), Triangle::Full)
            .expect("ard");
        for col in 0..3 {
            for row in 0..3 {
                assert_close(k_ard[(row, col)], k_iso[(row, col)]);
            }
        }
    }

    #[test]
    fn full_is_symmetric() {
        let kernel = RationalQuadraticArdKernel::new(&[0.8, 1.4], 1.2).expect("valid");
        let x = points_2d(&[[0.0, 0.0], [0.5, 1.0], [1.2, -0.3]]);
        let mut k = fill(3, 0.0);
        kernel
            .apply(x.as_ref(), k.as_mut(), Triangle::Full)
            .expect("shape");
        for col in 0..3 {
            for row in 0..3 {
                assert_close(k[(row, col)], k[(col, row)]);
            }
        }
    }

    #[test]
    fn lower_matches_full_and_leaves_upper() {
        let kernel = RationalQuadraticArdKernel::new(&[1.0, 0.5], 1.0).expect("valid");
        let x = points_2d(&[[0.0, 0.0], [1.0, 0.5], [0.2, 1.0]]);
        let mut full = fill(3, 0.0);
        kernel
            .apply(x.as_ref(), full.as_mut(), Triangle::Full)
            .expect("shape");
        let mut lower = fill(3, 42.0);
        kernel
            .apply(x.as_ref(), lower.as_mut(), Triangle::Lower)
            .expect("shape");
        lower_matches(lower.as_ref(), full.as_ref());
        assert_close(lower[(0, 1)], 42.0);
    }

    #[test]
    fn grad_matches_finite_difference_per_param() {
        let kernel =
            RationalQuadraticArdKernel::from_log_lengthscales(&[-0.2, 0.4], 0.1).expect("valid");
        let mut theta = vec![0.0; kernel.num_params()];
        kernel.get_params(&mut theta).expect("len");
        let h = 1e-6;
        let x = points_2d(&[[0.0, 0.0], [0.7, 1.1], [-0.3, 0.4]]);
        for idx in 0..theta.len() {
            let mut plus_th = theta.clone();
            let mut minus_th = theta.clone();
            plus_th[idx] += h;
            minus_th[idx] -= h;
            let plus = RationalQuadraticArdKernel::from_log_lengthscales(&plus_th[..2], plus_th[2])
                .expect("plus");
            let minus =
                RationalQuadraticArdKernel::from_log_lengthscales(&minus_th[..2], minus_th[2])
                    .expect("minus");
            let mut k_plus = fill(3, 0.0);
            let mut k_minus = fill(3, 0.0);
            let mut dk = fill(3, 0.0);
            plus.apply(x.as_ref(), k_plus.as_mut(), Triangle::Full)
                .expect("plus");
            minus
                .apply(x.as_ref(), k_minus.as_mut(), Triangle::Full)
                .expect("minus");
            kernel
                .grad(x.as_ref(), dk.as_mut(), idx, Triangle::Full)
                .expect("idx");
            for col in 0..3 {
                for row in 0..3 {
                    let fd = (k_plus[(row, col)] - k_minus[(row, col)]) / (2.0 * h);
                    assert_close(dk[(row, col)], fd);
                }
            }
        }
    }

    #[test]
    fn unused_dimension_has_zero_grad() {
        let kernel = RationalQuadraticArdKernel::new(&[1.0, 2.0], 1.0).expect("valid");
        let x = points_2d(&[[0.0, 0.0], [1.0, 0.0]]);
        let mut dk0 = fill(2, 0.0);
        let mut dk1 = fill(2, 0.0);
        kernel
            .grad(x.as_ref(), dk0.as_mut(), 0, Triangle::Full)
            .expect("dim 0");
        kernel
            .grad(x.as_ref(), dk1.as_mut(), 1, Triangle::Full)
            .expect("dim 1");
        assert_close(dk1[(1, 0)], 0.0);
        assert!(dk0[(1, 0)].abs() > 1e-8);
    }

    #[test]
    fn grad_lower_matches_full() {
        let kernel = RationalQuadraticArdKernel::new(&[1.0, 0.5], 0.9).expect("valid");
        let x = points_2d(&[[0.0, 0.0], [0.8, 0.3], [1.6, -0.2]]);
        let mut full = fill(3, 0.0);
        let mut lower = fill(3, 99.0);
        kernel
            .grad(x.as_ref(), full.as_mut(), 2, Triangle::Full)
            .expect("alpha");
        kernel
            .grad(x.as_ref(), lower.as_mut(), 2, Triangle::Lower)
            .expect("alpha");
        lower_matches(lower.as_ref(), full.as_ref());
        assert_close(lower[(0, 1)], 99.0);
    }

    #[test]
    fn get_set_params_roundtrip_and_atomic() {
        let mut kernel = RationalQuadraticArdKernel::new(&[2.0, 0.5], 1.5).expect("valid");
        let mut params = [0.0; 3];
        kernel.get_params(&mut params).expect("len 3");
        assert_close(params[0], 2.0_f64.ln());
        assert_close(params[2], 1.5_f64.ln());
        params[0] = 0.5_f64.ln();
        kernel.set_params(&params).expect("len 3");
        assert_close(kernel.lengthscale(0).expect("dim 0"), 0.5);
        let before = kernel.clone();
        assert!(kernel.set_params(&[0.0, 0.0, f64::INFINITY]).is_err());
        assert_eq!(kernel, before);
    }

    #[test]
    fn apply_cross_matches_square_block() {
        let kernel = RationalQuadraticArdKernel::new(&[1.0, 2.0], 1.0).expect("valid");
        let train = points_2d(&[[0.0, 0.0], [1.0, 0.5]]);
        let test = points_2d(&[[0.2, -0.1], [1.0, 0.5]]);
        let mut square = fill(2, 0.0);
        kernel
            .apply(train.as_ref(), square.as_mut(), Triangle::Full)
            .expect("square");
        let mut cross = fill(2, 0.0);
        kernel
            .apply_cross(train.as_ref(), test.as_ref(), cross.as_mut())
            .expect("rect");
        assert_close(cross[(0, 1)], square[(0, 1)]);
        assert_close(cross[(1, 1)], square[(1, 1)]);
    }

    #[test]
    fn rejects_bad_index_dim_and_non_finite() {
        assert!(matches!(
            RationalQuadraticArdKernel::new(&[1.0, 0.0], 1.0),
            Err(GprError::InvalidHyperparameter { .. })
        ));
        let kernel = RationalQuadraticArdKernel::new(&[1.0, 2.0], 1.0).expect("valid");
        let x = points_2d(&[[0.0, 0.0], [1.0, 1.0]]);
        let mut dk = fill(2, 0.0);
        assert!(matches!(
            kernel.grad(x.as_ref(), dk.as_mut(), 3, Triangle::Lower),
            Err(GprError::InvalidHyperparameter { .. })
        ));
        let bad_d = Mat::from_fn(2, 3, |_, _| 0.0);
        let mut k = fill(2, 0.0);
        assert!(matches!(
            kernel.apply(bad_d.as_ref(), k.as_mut(), Triangle::Full),
            Err(GprError::DimensionMismatch { .. })
        ));
        let nan = points_2d(&[[0.0, 0.0], [f64::NAN, 1.0]]);
        assert!(matches!(
            kernel.apply(nan.as_ref(), k.as_mut(), Triangle::Full),
            Err(GprError::NonFiniteInput)
        ));
    }
}
