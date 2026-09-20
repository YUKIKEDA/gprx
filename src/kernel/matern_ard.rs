//! ARD Matérn kernel for `ν = 1/2`, `3/2`, and `5/2`.

use super::matern::{
    MaternNu, finite_kernel, matern_d2k_dtheta_ard, matern_dk_dtheta_ard, matern_from_r,
};
use super::{ArdLengthscales, Triangle, visit_triangle};
use crate::error::GprError;
use faer::{MatMut, MatRef};

/// ARD Matérn: `k` is a function of `r = √(Σ_d (x_d-x'_d)² / ℓ_d²)`.
///
/// Optimizer parameters are `θ_d = log(ℓ_d)` via [`ArdLengthscales`]. `ν` is
/// not an optimizer parameter. When every `ℓ_d` equals a scalar `ℓ`, values
/// match isotropic [`super::MaternKernel`]. `apply` / `grad` take the `n×d`
/// coordinate matrix. Amplitude is not stored here.
///
/// Cloning copies the lengthscale vectors. When
/// [`crate::CachedDistances`] is set, [`crate::Gpr`] caches raw
/// `(Δx_d)²`.
///
/// # Examples
///
/// ```rust
/// use gprx::kernel::{MaternArdKernel, MaternNu};
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let k = MaternArdKernel::new(&[1.0, 2.5], MaternNu::FiveHalves)?;
/// assert_eq!(k.num_params(), 2);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct MaternArdKernel {
    nu: MaternNu,
    lengthscales: ArdLengthscales,
}

impl MaternArdKernel {
    /// Builds an ARD Matérn kernel from positive finite `ℓ_d` and `ν`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if the slice is empty or a
    /// lengthscale is invalid.
    pub fn new(lengthscales: &[f64], nu: MaternNu) -> Result<Self, GprError> {
        Ok(Self {
            nu,
            lengthscales: ArdLengthscales::new(lengthscales)?,
        })
    }

    /// Builds an ARD Matérn kernel from `θ_d = log(ℓ_d)` and `ν`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if the slice is empty or a
    /// `θ_d` is invalid.
    pub fn from_log_lengthscales(log_lengthscales: &[f64], nu: MaternNu) -> Result<Self, GprError> {
        Ok(Self {
            nu,
            lengthscales: ArdLengthscales::from_log_lengthscales(log_lengthscales)?,
        })
    }

    /// Returns the smoothness `ν`.
    pub fn nu(&self) -> MaternNu {
        self.nu
    }

    /// Returns the shared ARD lengthscale mouth.
    pub fn lengthscales(&self) -> &ArdLengthscales {
        &self.lengthscales
    }

    pub(crate) fn from_ard(lengthscales: ArdLengthscales, nu: MaternNu) -> Self {
        Self { nu, lengthscales }
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

    /// Rebuilds every `ℓ_d` with the same open interval.
    ///
    /// # Errors
    ///
    /// Returns [`crate::IntervalError`] if any current `ℓ_d` is not strictly
    /// inside `interval`.
    pub fn with_bounds(
        self,
        interval: crate::param::Interval,
    ) -> Result<Self, crate::IntervalError> {
        Ok(Self {
            nu: self.nu,
            lengthscales: self.lengthscales.with_bounds(interval)?,
        })
    }

    /// Returns the number of optimizer parameters (`d`).
    pub fn num_params(&self) -> usize {
        self.lengthscales.num_params()
    }

    /// Writes `θ_d` into `out`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if `out` is the wrong length.
    pub fn get_params(&self, out: &mut [f64]) -> Result<(), GprError> {
        self.lengthscales.get_params(out)
    }

    /// Replaces `θ_d` from `params`. The previous values and `ν` are kept on
    /// error.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if `params` is the wrong
    /// length or a `θ_d` is invalid.
    pub fn set_params(&mut self, params: &[f64]) -> Result<(), GprError> {
        self.lengthscales.set_params(params)
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
        let n = require_square_points(x, out.as_ref(), self.num_params())?;
        let inv_ell_sq = self.lengthscales.inv_ell_sq();
        let nu = self.nu;
        let mut err = None;
        visit_triangle(n, uplo, |row, col| {
            if err.is_some() {
                return;
            }
            match ard_kernel(x, row, col, inv_ell_sq, nu) {
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
        let d = self.num_params();
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
        require_finite_points(x)?;
        require_finite_points(xs)?;
        let inv_ell_sq = self.lengthscales.inv_ell_sq();
        let nu = self.nu;
        for col in 0..xs.nrows() {
            for row in 0..x.nrows() {
                out[(row, col)] = ard_kernel_pair(x, row, xs, col, inv_ell_sq, nu)?;
            }
        }
        Ok(())
    }

    /// Writes the stationary diagonal `k(x, x) = 1` into `out`.
    pub fn fill_diag(&self, out: &mut [f64]) {
        out.fill(1.0);
    }

    /// Writes `∂K/∂θ_d` for `θ_d = log(ℓ_d)` into `d_k`.
    ///
    /// This is not `∂k/∂ℓ_d`.
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
        if param_idx >= self.num_params() {
            return Err(GprError::InvalidHyperparameter {
                reason: format!(
                    "ARD Matern parameter index {param_idx} is out of range (d={})",
                    self.num_params()
                ),
            });
        }
        let n = require_square_points(x, d_k.as_ref(), self.num_params())?;
        let inv_ell_sq = self.lengthscales.inv_ell_sq();
        let nu = self.nu;
        let mut err = None;
        visit_triangle(n, uplo, |row, col| {
            if err.is_some() {
                return;
            }
            match ard_kernel_grad(x, row, col, inv_ell_sq, param_idx, nu) {
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
        let d = self.num_params();
        super::dist::require_ard_sq_diff_shape(cache, n, d)?;
        let inv_ell_sq = self.lengthscales.inv_ell_sq();
        let nu = self.nu;
        let mut err = None;
        visit_triangle(n, uplo, |row, col| {
            if err.is_some() {
                return;
            }
            match super::dist::weighted_r2_from_cache(cache, n, row, col, inv_ell_sq, None)
                .and_then(|(r2, _)| finite_kernel(matern_from_r(nu, r2.max(0.0).sqrt())))
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
        if param_idx >= self.num_params() {
            return Err(GprError::InvalidHyperparameter {
                reason: format!(
                    "ARD Matern parameter index {param_idx} is out of range (d={})",
                    self.num_params()
                ),
            });
        }
        let n = d_k.nrows();
        if d_k.ncols() != n {
            return Err(GprError::InvalidHyperparameter {
                reason: format!("output is {}x{}, expected square", d_k.nrows(), d_k.ncols()),
            });
        }
        let d = self.num_params();
        super::dist::require_ard_sq_diff_shape(cache, n, d)?;
        let inv_ell_sq = self.lengthscales.inv_ell_sq();
        let nu = self.nu;
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
                Some(param_idx),
            )
            .and_then(|(r2, dim_term)| {
                if !r2.is_finite() {
                    return Err(GprError::NonFiniteKernelValue);
                }
                finite_kernel(matern_dk_dtheta_ard(nu, r2.max(0.0).sqrt(), dim_term))
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

    /// Writes `∂²K/∂θ_i ∂θ_j` for ARD `θ_d = log(ℓ_d)`.
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
        let d = self.num_params();
        if i >= d || j >= d {
            return Err(GprError::InvalidHyperparameter {
                reason: format!("ARD Matern parameter pair ({i}, {j}) is out of range (d={d})"),
            });
        }
        let n = require_square_points(x, d2_k.as_ref(), d)?;
        let inv_ell_sq = self.lengthscales.inv_ell_sq();
        let nu = self.nu;
        let mut err = None;
        visit_triangle(n, uplo, |row, col| {
            if err.is_some() {
                return;
            }
            match ard_kernel_hess(x, row, col, inv_ell_sq, i, j, nu) {
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
        let d = self.num_params();
        if i >= d || j >= d {
            return Err(GprError::InvalidHyperparameter {
                reason: format!("ARD Matern parameter pair ({i}, {j}) is out of range (d={d})"),
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
        let nu = self.nu;
        let mut err = None;
        visit_triangle(n, uplo, |row, col| {
            if err.is_some() {
                return;
            }
            match ard_hess_from_cache(cache, n, row, col, inv_ell_sq, (i, j), nu) {
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

fn require_finite_points(x: MatRef<'_, f64>) -> Result<(), GprError> {
    for col in 0..x.ncols() {
        for row in 0..x.nrows() {
            if !x[(row, col)].is_finite() {
                return Err(GprError::NonFiniteInput);
            }
        }
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
    require_finite_points(x)?;
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
        Ok(r2)
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
    nu: MaternNu,
) -> Result<f64, GprError> {
    let r2 = ard_r2_pair(x, row, xs, col, inv_ell_sq)?;
    finite_kernel(matern_from_r(nu, r2.max(0.0).sqrt()))
}

fn ard_kernel(
    x: MatRef<'_, f64>,
    row: usize,
    col: usize,
    inv_ell_sq: &[f64],
    nu: MaternNu,
) -> Result<f64, GprError> {
    ard_kernel_pair(x, row, x, col, inv_ell_sq, nu)
}

fn ard_kernel_grad(
    x: MatRef<'_, f64>,
    row: usize,
    col: usize,
    inv_ell_sq: &[f64],
    param_idx: usize,
    nu: MaternNu,
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
    let r = r2.max(0.0).sqrt();
    finite_kernel(matern_dk_dtheta_ard(nu, r, dim_term))
}

fn ard_dim_pair(
    x: MatRef<'_, f64>,
    row: usize,
    col: usize,
    inv_ell_sq: &[f64],
    i: usize,
    j: usize,
) -> Result<(f64, f64, f64), GprError> {
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
    if r2.is_finite() {
        Ok((r2, dim_i, dim_j))
    } else {
        Err(GprError::NonFiniteKernelValue)
    }
}

fn ard_kernel_hess(
    x: MatRef<'_, f64>,
    row: usize,
    col: usize,
    inv_ell_sq: &[f64],
    i: usize,
    j: usize,
    nu: MaternNu,
) -> Result<f64, GprError> {
    let (r2, dim_i, dim_j) = ard_dim_pair(x, row, col, inv_ell_sq, i, j)?;
    let r = r2.max(0.0).sqrt();
    finite_kernel(matern_d2k_dtheta_ard(nu, r, dim_i, dim_j, i == j))
}

fn ard_hess_from_cache(
    cache: MatRef<'_, f64>,
    n: usize,
    row: usize,
    col: usize,
    inv_ell_sq: &[f64],
    pair: (usize, usize),
    nu: MaternNu,
) -> Result<f64, GprError> {
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
    let r = r2.max(0.0).sqrt();
    finite_kernel(matern_d2k_dtheta_ard(nu, r, dim_i, dim_j, i == j))
}

#[cfg(test)]
mod tests {
    use super::MaternArdKernel;
    use crate::error::GprError;
    use crate::kernel::{MaternKernel, MaternNu, Triangle};
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

    fn all_nu() -> [MaternNu; 3] {
        [MaternNu::Half, MaternNu::ThreeHalves, MaternNu::FiveHalves]
    }

    #[test]
    fn is_send_sync() {
        assert_send_sync::<MaternArdKernel>();
    }

    #[test]
    fn diagonal_is_one() {
        let kernel = MaternArdKernel::new(&[1.0, 2.0], MaternNu::ThreeHalves).expect("valid");
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
        for nu in all_nu() {
            let ell = 1.25;
            let iso = MaternKernel::new(ell, nu).expect("valid");
            let ard = MaternArdKernel::new(&[ell, ell], nu).expect("valid");
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
    }

    #[test]
    fn known_values_use_per_dimension_lengthscales() {
        let kernel = MaternArdKernel::new(&[1.0, 2.0], MaternNu::Half).expect("valid");
        let x = points_2d(&[[0.0, 0.0], [1.0, 0.0]]);
        let mut k = fill(2, 0.0);
        kernel
            .apply(x.as_ref(), k.as_mut(), Triangle::Full)
            .expect("shape");
        let r = 1.0_f64;
        assert_close(k[(1, 0)], (-r).exp());
    }

    #[test]
    fn full_is_symmetric() {
        let kernel = MaternArdKernel::new(&[0.8, 1.4], MaternNu::FiveHalves).expect("valid");
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
        let kernel = MaternArdKernel::new(&[1.0, 0.5], MaternNu::ThreeHalves).expect("valid");
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
    fn grad_matches_finite_difference_per_dimension() {
        for nu in all_nu() {
            let kernel = MaternArdKernel::from_log_lengthscales(&[-0.2, 0.4], nu).expect("valid");
            let theta: Vec<f64> = kernel.log_lengthscales().to_vec();
            let h = 1e-6;
            let x = points_2d(&[[0.0, 0.0], [0.7, 1.1], [-0.3, 0.4]]);
            for dim in 0..2 {
                let mut plus_th = theta.clone();
                let mut minus_th = theta.clone();
                plus_th[dim] += h;
                minus_th[dim] -= h;
                let plus = MaternArdKernel::from_log_lengthscales(&plus_th, nu).expect("valid");
                let minus = MaternArdKernel::from_log_lengthscales(&minus_th, nu).expect("valid");
                let mut k_plus = fill(3, 0.0);
                let mut k_minus = fill(3, 0.0);
                let mut dk = fill(3, 0.0);
                plus.apply(x.as_ref(), k_plus.as_mut(), Triangle::Full)
                    .expect("plus");
                minus
                    .apply(x.as_ref(), k_minus.as_mut(), Triangle::Full)
                    .expect("minus");
                kernel
                    .grad(x.as_ref(), dk.as_mut(), dim, Triangle::Full)
                    .expect("dim");
                for col in 0..3 {
                    for row in 0..3 {
                        let fd = (k_plus[(row, col)] - k_minus[(row, col)]) / (2.0 * h);
                        assert_close(dk[(row, col)], fd);
                    }
                }
            }
        }
    }

    #[test]
    fn unused_dimension_has_zero_grad() {
        let kernel = MaternArdKernel::new(&[1.0, 2.0], MaternNu::FiveHalves).expect("valid");
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
        let kernel = MaternArdKernel::new(&[1.0, 0.5], MaternNu::Half).expect("valid");
        let x = points_2d(&[[0.0, 0.0], [0.8, 0.3], [1.6, -0.2]]);
        let mut full = fill(3, 0.0);
        let mut lower = fill(3, 99.0);
        kernel
            .grad(x.as_ref(), full.as_mut(), 1, Triangle::Full)
            .expect("index 1");
        kernel
            .grad(x.as_ref(), lower.as_mut(), 1, Triangle::Lower)
            .expect("index 1");
        lower_matches(lower.as_ref(), full.as_ref());
        assert_close(lower[(0, 1)], 99.0);
    }

    #[test]
    fn get_set_params_roundtrip() {
        let mut kernel = MaternArdKernel::new(&[2.0, 0.5], MaternNu::ThreeHalves).expect("valid");
        let mut params = [0.0; 2];
        kernel.get_params(&mut params).expect("len 2");
        assert_close(params[0], 2.0_f64.ln());
        assert_close(params[1], 0.5_f64.ln());
        params[0] = 0.5_f64.ln();
        kernel.set_params(&params).expect("len 2");
        assert_close(kernel.lengthscale(0).expect("dim 0"), 0.5);
        assert_eq!(kernel.nu(), MaternNu::ThreeHalves);
    }

    #[test]
    fn apply_cross_matches_square_block() {
        let kernel = MaternArdKernel::new(&[1.0, 2.0], MaternNu::ThreeHalves).expect("valid");
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
            MaternArdKernel::new(&[1.0, 0.0], MaternNu::Half),
            Err(GprError::InvalidHyperparameter { .. })
        ));
        let kernel = MaternArdKernel::new(&[1.0, 2.0], MaternNu::Half).expect("valid");
        let x = points_2d(&[[0.0, 0.0], [1.0, 1.0]]);
        let mut dk = fill(2, 0.0);
        assert!(matches!(
            kernel.grad(x.as_ref(), dk.as_mut(), 2, Triangle::Lower),
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
