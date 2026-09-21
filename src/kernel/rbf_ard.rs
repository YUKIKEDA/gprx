//! ARD squared-exponential (RBF) kernel.

use super::dist::require_ard_sq_diff_shape;
use super::simd::{
    try_apply_rbf_ard_cache, try_apply_rbf_ard_cross, try_apply_rbf_ard_points,
    try_grad_rbf_ard_cache, try_grad_rbf_ard_points,
};
use super::{ArdLengthscales, Triangle, visit_triangle};
use crate::error::GprError;
use faer::reborrow::ReborrowMut;
use faer::{MatMut, MatRef};

/// ARD RBF: `k = exp( -½ Σ_d (x_d - x'_d)² / ℓ_d² )`.
///
/// Optimizer parameters are `θ_d = log(ℓ_d)` via [`ArdLengthscales`]. When every
/// `ℓ_d` equals a scalar `ℓ`, values match isotropic [`super::RbfKernel`].
/// `apply` / `grad` take the `n×d` coordinate matrix; a scalar squared-distance
/// matrix is not enough for `∂K/∂θ_d`. Amplitude is not stored here.
///
/// Cloning copies the lengthscale vectors. When
/// [`crate::CachedDistances`] is set, [`crate::Gpr`] caches raw
/// `(Δx_d)²` as `n × (n·d)` and evaluates from that tensor. Column-major
/// views with unit row stride use `wide::f64x4` for [`Self::apply`] and
/// [`Self::grad`].
///
/// # Examples
///
/// ```rust
/// use gprx::kernel::RbfArdKernel;
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let rbf = RbfArdKernel::new(&[1.0, 2.5])?;
/// assert_eq!(rbf.num_params(), 2);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct RbfArdKernel {
    lengthscales: ArdLengthscales,
}

impl RbfArdKernel {
    /// Builds an ARD RBF kernel from positive finite `ℓ_d`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if the slice is empty or a
    /// lengthscale is invalid.
    pub fn new(lengthscales: &[f64]) -> Result<Self, GprError> {
        Ok(Self {
            lengthscales: ArdLengthscales::new(lengthscales)?,
        })
    }

    /// Builds an ARD RBF kernel from `θ_d = log(ℓ_d)`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if the slice is empty or a
    /// `θ_d` is invalid.
    pub fn from_log_lengthscales(log_lengthscales: &[f64]) -> Result<Self, GprError> {
        Ok(Self {
            lengthscales: ArdLengthscales::from_log_lengthscales(log_lengthscales)?,
        })
    }

    /// Returns the shared ARD lengthscale mouth.
    pub fn lengthscales(&self) -> &ArdLengthscales {
        &self.lengthscales
    }

    pub(crate) fn from_ard(lengthscales: ArdLengthscales) -> Self {
        Self { lengthscales }
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

    /// Replaces `θ_d` from `params`. The previous values are kept on error.
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
        if try_apply_rbf_ard_points(x, out.rb_mut(), uplo, inv_ell_sq)? {
            return Ok(());
        }
        let mut err = None;
        visit_triangle(n, uplo, |row, col| {
            if err.is_some() {
                return;
            }
            match ard_kernel(x, row, col, inv_ell_sq) {
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
        if try_apply_rbf_ard_cross(x, xs, out.rb_mut(), inv_ell_sq)? {
            return Ok(());
        }
        for col in 0..xs.nrows() {
            for row in 0..x.nrows() {
                out[(row, col)] = ard_kernel_pair(x, row, xs, col, inv_ell_sq)?;
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
    /// `∂k/∂θ_d = k · (x_d - x'_d)² / ℓ_d²`. This is not `∂k/∂ℓ_d`.
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
                    "ARD RBF parameter index {param_idx} is out of range (d={})",
                    self.num_params()
                ),
            });
        }
        let n = require_square_points(x, d_k.as_ref(), self.num_params())?;
        let inv_ell_sq = self.lengthscales.inv_ell_sq();
        if try_grad_rbf_ard_points(x, d_k.rb_mut(), uplo, inv_ell_sq, param_idx)? {
            return Ok(());
        }
        let mut err = None;
        visit_triangle(n, uplo, |row, col| {
            if err.is_some() {
                return;
            }
            match ard_kernel_grad(x, row, col, inv_ell_sq, param_idx) {
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
        let n = require_square_out(out.as_ref())?;
        let d = self.num_params();
        require_ard_sq_diff_shape(cache, n, d)?;
        let inv_ell_sq = self.lengthscales.inv_ell_sq();
        if try_apply_rbf_ard_cache(cache, out.rb_mut(), uplo, inv_ell_sq)? {
            return Ok(());
        }
        let mut err = None;
        visit_triangle(n, uplo, |row, col| {
            if err.is_some() {
                return;
            }
            match ard_kernel_from_cache(cache, n, row, col, inv_ell_sq) {
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
                    "ARD RBF parameter index {param_idx} is out of range (d={})",
                    self.num_params()
                ),
            });
        }
        let n = require_square_out(d_k.as_ref())?;
        let d = self.num_params();
        require_ard_sq_diff_shape(cache, n, d)?;
        let inv_ell_sq = self.lengthscales.inv_ell_sq();
        if try_grad_rbf_ard_cache(cache, d_k.rb_mut(), uplo, inv_ell_sq, param_idx)? {
            return Ok(());
        }
        let mut err = None;
        visit_triangle(n, uplo, |row, col| {
            if err.is_some() {
                return;
            }
            match ard_kernel_grad_from_cache(cache, n, row, col, inv_ell_sq, param_idx) {
                Ok(value) => d_k[(row, col)] = value,
                Err(e) => err = Some(e),
            }
        });
        match err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// Writes `∂²K/∂θ_i ∂θ_j` for `θ_d = log(ℓ_d)` into `d2_k`.
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
                reason: format!("ARD RBF parameter pair ({i}, {j}) is out of range (d={d})"),
            });
        }
        let n = require_square_points(x, d2_k.as_ref(), d)?;
        let inv_ell_sq = self.lengthscales.inv_ell_sq();
        let mut err = None;
        visit_triangle(n, uplo, |row, col| {
            if err.is_some() {
                return;
            }
            match ard_kernel_hess(x, row, col, inv_ell_sq, i, j) {
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
                reason: format!("ARD RBF parameter pair ({i}, {j}) is out of range (d={d})"),
            });
        }
        let n = require_square_out(d2_k.as_ref())?;
        require_ard_sq_diff_shape(cache, n, d)?;
        let inv_ell_sq = self.lengthscales.inv_ell_sq();
        let mut err = None;
        visit_triangle(n, uplo, |row, col| {
            if err.is_some() {
                return;
            }
            match ard_kernel_hess_from_cache(cache, n, row, col, inv_ell_sq, i, j) {
                Ok(value) => d2_k[(row, col)] = value,
                Err(e) => err = Some(e),
            }
        });
        match err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// Writes `∂K(X1, X2)/∂X2[*, dim]` into `d_k`.
    ///
    /// `∂k/∂x2_e = k (x1_e - x2_e) / ℓ_e²`.
    ///
    /// # Errors
    ///
    /// Same shape / non-finite errors as [`RbfKernel::grad_wrt_coord_dim`], or
    /// [`GprError::InvalidHyperparameter`] when `dim` does not match `ℓ_d`.
    pub fn grad_wrt_coord_dim(
        &self,
        x1: MatRef<'_, f64>,
        x2: MatRef<'_, f64>,
        mut d_k: MatMut<'_, f64>,
        dim: usize,
    ) -> Result<(), GprError> {
        super::require_coord_grad(x1, x2, d_k.as_ref(), dim)?;
        if dim >= self.num_params() {
            return Err(GprError::InvalidHyperparameter {
                reason: format!(
                    "coordinate dimension {dim} is out of range for d={}",
                    self.num_params()
                ),
            });
        }
        let inv_ell_sq = self.lengthscales.inv_ell_sq();
        for col in 0..x2.nrows() {
            for row in 0..x1.nrows() {
                let k = ard_kernel_pair(x1, row, x2, col, inv_ell_sq)?;
                let delta = x1[(row, dim)] - x2[(col, dim)];
                d_k[(row, col)] = k * delta * inv_ell_sq[dim];
            }
        }
        Ok(())
    }

    pub(crate) fn hess_wrt_coord_dims(
        &self,
        x1: MatRef<'_, f64>,
        x2: MatRef<'_, f64>,
        mut d2_k: MatMut<'_, f64>,
        dim_a: usize,
        dim_b: usize,
    ) -> Result<(), GprError> {
        super::require_coord_grad(x1, x2, d2_k.as_ref(), dim_a)?;
        super::require_coord_grad(x1, x2, d2_k.as_ref(), dim_b)?;
        let inv_ell_sq = self.lengthscales.inv_ell_sq();
        for col in 0..x2.nrows() {
            for row in 0..x1.nrows() {
                let k = ard_kernel_pair(x1, row, x2, col, inv_ell_sq)?;
                let da = x1[(row, dim_a)] - x2[(col, dim_a)];
                let db = x1[(row, dim_b)] - x2[(col, dim_b)];
                let mut value = k * da * db * inv_ell_sq[dim_a] * inv_ell_sq[dim_b];
                if dim_a == dim_b {
                    value -= k * inv_ell_sq[dim_a];
                }
                d2_k[(row, col)] = value;
            }
        }
        Ok(())
    }

    pub(crate) fn hess_wrt_coord_mixed(
        &self,
        x1: MatRef<'_, f64>,
        x2: MatRef<'_, f64>,
        mut d2_k: MatMut<'_, f64>,
        dim_x1: usize,
        dim_x2: usize,
    ) -> Result<(), GprError> {
        super::require_coord_grad(x1, x2, d2_k.as_ref(), dim_x1)?;
        super::require_coord_grad(x1, x2, d2_k.as_ref(), dim_x2)?;
        let inv_ell_sq = self.lengthscales.inv_ell_sq();
        for col in 0..x2.nrows() {
            for row in 0..x1.nrows() {
                let k = ard_kernel_pair(x1, row, x2, col, inv_ell_sq)?;
                let d1 = x1[(row, dim_x1)] - x2[(col, dim_x1)];
                let d2 = x1[(row, dim_x2)] - x2[(col, dim_x2)];
                let mut value = -k * d1 * d2 * inv_ell_sq[dim_x1] * inv_ell_sq[dim_x2];
                if dim_x1 == dim_x2 {
                    value += k * inv_ell_sq[dim_x1];
                }
                d2_k[(row, col)] = value;
            }
        }
        Ok(())
    }

    pub(crate) fn hess_theta_coord_dim(
        &self,
        x1: MatRef<'_, f64>,
        x2: MatRef<'_, f64>,
        mut d2_k: MatMut<'_, f64>,
        param_idx: usize,
        dim: usize,
    ) -> Result<(), GprError> {
        if param_idx >= self.num_params() {
            return Err(GprError::InvalidHyperparameter {
                reason: format!(
                    "ARD RBF parameter {param_idx} is out of range (d={})",
                    self.num_params()
                ),
            });
        }
        super::require_coord_grad(x1, x2, d2_k.as_ref(), dim)?;
        let inv_ell_sq = self.lengthscales.inv_ell_sq();
        for col in 0..x2.nrows() {
            for row in 0..x1.nrows() {
                let k = ard_kernel_pair(x1, row, x2, col, inv_ell_sq)?;
                let delta_theta = x1[(row, param_idx)] - x2[(col, param_idx)];
                let delta_dim = x1[(row, dim)] - x2[(col, dim)];
                let dk_dtheta = k * delta_theta * delta_theta * inv_ell_sq[param_idx];
                let mut value = dk_dtheta * delta_dim * inv_ell_sq[dim];
                if param_idx == dim {
                    value += k * delta_dim * (-2.0 * inv_ell_sq[dim]);
                }
                d2_k[(row, col)] = value;
            }
        }
        Ok(())
    }

    pub(crate) fn grad_cross_from_coords(
        &self,
        x1: MatRef<'_, f64>,
        x2: MatRef<'_, f64>,
        mut d_k: MatMut<'_, f64>,
        param_idx: usize,
    ) -> Result<(), GprError> {
        if param_idx >= self.num_params() {
            return Err(GprError::InvalidHyperparameter {
                reason: format!(
                    "ARD RBF parameter {param_idx} is out of range (d={})",
                    self.num_params()
                ),
            });
        }
        super::require_coord_grad(x1, x2, d_k.as_ref(), 0)?;
        let inv_ell_sq = self.lengthscales.inv_ell_sq();
        for col in 0..x2.nrows() {
            for row in 0..x1.nrows() {
                d_k[(row, col)] = ard_kernel_grad_pair(x1, row, x2, col, inv_ell_sq, param_idx)?;
            }
        }
        Ok(())
    }

    pub(crate) fn hess_cross_from_coords(
        &self,
        x1: MatRef<'_, f64>,
        x2: MatRef<'_, f64>,
        mut d2_k: MatMut<'_, f64>,
        i: usize,
        j: usize,
    ) -> Result<(), GprError> {
        let d = self.num_params();
        if i >= d || j >= d {
            return Err(GprError::InvalidHyperparameter {
                reason: format!("ARD RBF parameter pair ({i}, {j}) is out of range (d={d})"),
            });
        }
        super::require_coord_grad(x1, x2, d2_k.as_ref(), 0)?;
        let inv_ell_sq = self.lengthscales.inv_ell_sq();
        for col in 0..x2.nrows() {
            for row in 0..x1.nrows() {
                d2_k[(row, col)] = ard_kernel_hess_pair(x1, row, x2, col, inv_ell_sq, i, j)?;
            }
        }
        Ok(())
    }
}

fn ard_kernel_grad_pair(
    x: MatRef<'_, f64>,
    row: usize,
    xs: MatRef<'_, f64>,
    col: usize,
    inv_ell_sq: &[f64],
    param_idx: usize,
) -> Result<f64, GprError> {
    let mut r2 = 0.0;
    let mut dim_term = 0.0;
    for (dim, &w) in inv_ell_sq.iter().enumerate() {
        let diff = x[(row, dim)] - xs[(col, dim)];
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
    let k = (-0.5 * r2).exp();
    let dk = k * dim_term;
    if dk.is_finite() {
        Ok(dk)
    } else {
        Err(GprError::NonFiniteKernelValue)
    }
}

fn ard_kernel_hess_pair(
    x: MatRef<'_, f64>,
    row: usize,
    xs: MatRef<'_, f64>,
    col: usize,
    inv_ell_sq: &[f64],
    i: usize,
    j: usize,
) -> Result<f64, GprError> {
    let mut r2 = 0.0;
    let mut dim_i = 0.0;
    let mut dim_j = 0.0;
    for (dim, &w) in inv_ell_sq.iter().enumerate() {
        let diff = x[(row, dim)] - xs[(col, dim)];
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
    let k = (-0.5 * r2).exp();
    ard_rbf_hess_from_terms(k, dim_i, dim_j, i == j)
}

fn require_square_out(out: MatRef<'_, f64>) -> Result<usize, GprError> {
    if out.nrows() == 0 || out.ncols() == 0 {
        return Err(GprError::EmptyInput);
    }
    if out.nrows() != out.ncols() {
        return Err(GprError::InvalidHyperparameter {
            reason: format!("output is {}x{}, expected square", out.nrows(), out.ncols()),
        });
    }
    Ok(out.nrows())
}

fn ard_kernel_from_cache(
    cache: MatRef<'_, f64>,
    n: usize,
    row: usize,
    col: usize,
    inv_ell_sq: &[f64],
) -> Result<f64, GprError> {
    let mut r2 = 0.0;
    for (dim, &w) in inv_ell_sq.iter().enumerate() {
        let v = cache[(row, dim * n + col)];
        if !v.is_finite() {
            return Err(GprError::NonFiniteInput);
        }
        r2 += v * w;
    }
    if !r2.is_finite() {
        return Err(GprError::NonFiniteKernelValue);
    }
    let k = (-0.5 * r2).exp();
    if k.is_finite() {
        Ok(k)
    } else {
        Err(GprError::NonFiniteKernelValue)
    }
}

fn ard_kernel_grad_from_cache(
    cache: MatRef<'_, f64>,
    n: usize,
    row: usize,
    col: usize,
    inv_ell_sq: &[f64],
    param_idx: usize,
) -> Result<f64, GprError> {
    let mut r2 = 0.0;
    let mut dim_term = 0.0;
    for (dim, &w) in inv_ell_sq.iter().enumerate() {
        let v = cache[(row, dim * n + col)];
        if !v.is_finite() {
            return Err(GprError::NonFiniteInput);
        }
        let term = v * w;
        r2 += term;
        if dim == param_idx {
            dim_term = term;
        }
    }
    if !r2.is_finite() {
        return Err(GprError::NonFiniteKernelValue);
    }
    let k = (-0.5 * r2).exp();
    let dk = k * dim_term;
    if dk.is_finite() {
        Ok(dk)
    } else {
        Err(GprError::NonFiniteKernelValue)
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
) -> Result<f64, GprError> {
    let r2 = ard_r2_pair(x, row, xs, col, inv_ell_sq)?;
    let k = (-0.5 * r2).exp();
    if k.is_finite() {
        Ok(k)
    } else {
        Err(GprError::NonFiniteKernelValue)
    }
}

fn ard_kernel(
    x: MatRef<'_, f64>,
    row: usize,
    col: usize,
    inv_ell_sq: &[f64],
) -> Result<f64, GprError> {
    ard_kernel_pair(x, row, x, col, inv_ell_sq)
}

fn ard_kernel_grad(
    x: MatRef<'_, f64>,
    row: usize,
    col: usize,
    inv_ell_sq: &[f64],
    param_idx: usize,
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
    let k = (-0.5 * r2).exp();
    let dk = k * dim_term;
    if dk.is_finite() {
        Ok(dk)
    } else {
        Err(GprError::NonFiniteKernelValue)
    }
}

fn ard_rbf_hess_from_terms(k: f64, dim_i: f64, dim_j: f64, same: bool) -> Result<f64, GprError> {
    let h = if same {
        k * dim_i * (dim_i - 2.0)
    } else {
        k * dim_i * dim_j
    };
    if h.is_finite() {
        Ok(h)
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
) -> Result<f64, GprError> {
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
    let k = (-0.5 * r2).exp();
    ard_rbf_hess_from_terms(k, dim_i, dim_j, i == j)
}

fn ard_kernel_hess_from_cache(
    cache: MatRef<'_, f64>,
    n: usize,
    row: usize,
    col: usize,
    inv_ell_sq: &[f64],
    i: usize,
    j: usize,
) -> Result<f64, GprError> {
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
    let k = (-0.5 * r2).exp();
    ard_rbf_hess_from_terms(k, dim_i, dim_j, i == j)
}

#[cfg(test)]
mod tests {
    use super::RbfArdKernel;
    use crate::error::GprError;
    use crate::kernel::{RbfKernel, Triangle};
    use faer::{Mat, MatRef};

    const TOL: f64 = 1e-10;

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
        assert_send_sync::<RbfArdKernel>();
    }

    #[test]
    fn diagonal_is_one() {
        let rbf = RbfArdKernel::new(&[1.0, 2.0]).expect("valid");
        let x = points_2d(&[[0.0, 0.0], [1.0, 0.5], [0.2, 1.3]]);
        let mut k = fill(3, f64::NAN);
        rbf.apply(x.as_ref(), k.as_mut(), Triangle::Full)
            .expect("shape");
        assert_close(k[(0, 0)], 1.0);
        assert_close(k[(1, 1)], 1.0);
        assert_close(k[(2, 2)], 1.0);
    }

    #[test]
    fn known_values_use_per_dimension_lengthscales() {
        let rbf = RbfArdKernel::new(&[1.0, 2.0]).expect("valid");
        // (0,1) differs only in dim 0 by 1 ⇒ k = exp(-1/(2ℓ₀²)) = exp(-1/2)
        // (0,2) differs only in dim 1 by 2 ⇒ k = exp(-4/(2ℓ₁²)) = exp(-1/2)
        let x = points_2d(&[[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]]);
        let mut k = fill(3, 0.0);
        rbf.apply(x.as_ref(), k.as_mut(), Triangle::Full)
            .expect("shape");
        assert_close(k[(1, 0)], (-0.5_f64).exp());
        assert_close(k[(2, 0)], (-0.5_f64).exp());
        assert_close(k[(2, 1)], (-0.5_f64 * (1.0 + 1.0)).exp());
    }

    #[test]
    fn equal_lengthscales_match_isotropic_rbf() {
        let ell = 1.4;
        let x = points_2d(&[[0.0, 0.0], [1.0, 0.5], [0.2, 1.3], [-0.4, 0.8]]);
        let dist = sq_dist(x.as_ref());
        let iso = RbfKernel::new(ell).expect("valid");
        let ard = RbfArdKernel::new(&[ell, ell]).expect("valid");
        let mut k_iso = fill(4, 0.0);
        let mut k_ard = fill(4, 0.0);
        iso.apply(dist.as_ref(), k_iso.as_mut(), Triangle::Full)
            .expect("shape");
        ard.apply(x.as_ref(), k_ard.as_mut(), Triangle::Full)
            .expect("shape");
        for col in 0..4 {
            for row in 0..4 {
                assert_close(k_ard[(row, col)], k_iso[(row, col)]);
            }
        }
    }

    #[test]
    fn apply_from_sq_diff_matches_apply_lower() {
        let rbf = RbfArdKernel::new(&[1.25, 0.8]).expect("valid");
        let x = points_2d(&[[0.0, 0.0], [1.0, 0.5], [0.2, 1.3], [-0.4, 0.8]]);
        let n = 4;
        let d = 2;
        let mut cache = Mat::zeros(n, n * d);
        crate::kernel::fill_ard_squared_diff(x.as_ref(), cache.as_mut(), &mut []);
        let mut from_points = fill(n, 0.0);
        let mut from_cache = fill(n, f64::NAN);
        rbf.apply(x.as_ref(), from_points.as_mut(), Triangle::Lower)
            .expect("points");
        rbf.apply_from_sq_diff(cache.as_ref(), from_cache.as_mut(), Triangle::Lower)
            .expect("cache");
        lower_matches(from_cache.as_ref(), from_points.as_ref());
    }

    #[test]
    fn grad_from_sq_diff_matches_grad_lower() {
        let rbf = RbfArdKernel::new(&[1.25, 0.8]).expect("valid");
        let x = points_2d(&[[0.0, 0.0], [1.0, 0.5], [0.2, 1.3], [-0.4, 0.8], [0.7, -1.1]]);
        let n = 5;
        let d = 2;
        let mut cache = Mat::zeros(n, n * d);
        crate::kernel::fill_ard_squared_diff(x.as_ref(), cache.as_mut(), &mut []);
        for param_idx in 0..d {
            let mut from_points = fill(n, 0.0);
            let mut from_cache = fill(n, f64::NAN);
            rbf.grad(x.as_ref(), from_points.as_mut(), param_idx, Triangle::Lower)
                .expect("points");
            rbf.grad_from_sq_diff(
                cache.as_ref(),
                from_cache.as_mut(),
                param_idx,
                Triangle::Lower,
            )
            .expect("cache");
            lower_matches(from_cache.as_ref(), from_points.as_ref());
        }
    }

    #[test]
    fn full_is_symmetric() {
        let rbf = RbfArdKernel::new(&[0.8, 1.7]).expect("valid");
        let x = points_2d(&[[0.0, 0.0], [0.5, 1.0], [2.0, -0.3], [2.5, 0.4]]);
        let mut k = fill(4, 0.0);
        rbf.apply(x.as_ref(), k.as_mut(), Triangle::Full)
            .expect("shape");
        for col in 0..4 {
            for row in 0..4 {
                assert_close(k[(row, col)], k[(col, row)]);
            }
        }
    }

    #[test]
    fn lower_matches_full_and_leaves_upper() {
        let rbf = RbfArdKernel::new(&[0.75, 1.25]).expect("valid");
        let x = points_2d(&[[0.0, 0.0], [1.0, 0.2], [2.0, -0.5]]);
        let mut full = fill(3, 0.0);
        rbf.apply(x.as_ref(), full.as_mut(), Triangle::Full)
            .expect("shape");
        let sentinel = 42.0;
        let mut lower = fill(3, sentinel);
        rbf.apply(x.as_ref(), lower.as_mut(), Triangle::Lower)
            .expect("shape");
        lower_matches(lower.as_ref(), full.as_ref());
        assert_close(lower[(0, 1)], sentinel);
        assert_close(lower[(0, 2)], sentinel);
        assert_close(lower[(1, 2)], sentinel);
    }

    #[test]
    fn upper_matches_full() {
        let rbf = RbfArdKernel::new(&[1.0, 1.5]).expect("valid");
        let x = points_2d(&[[0.0, 0.0], [1.0, 1.0], [2.0, 0.5]]);
        let mut full = fill(3, 0.0);
        let mut upper = fill(3, -1.0);
        rbf.apply(x.as_ref(), full.as_mut(), Triangle::Full)
            .expect("shape");
        rbf.apply(x.as_ref(), upper.as_mut(), Triangle::Upper)
            .expect("shape");
        for col in 0..3 {
            for row in 0..=col {
                assert_close(upper[(row, col)], full[(row, col)]);
            }
        }
        assert_close(upper[(1, 0)], -1.0);
        assert_close(upper[(2, 0)], -1.0);
        assert_close(upper[(2, 1)], -1.0);
    }

    #[test]
    fn hess_matches_finite_difference_of_grad() {
        let rbf = RbfArdKernel::from_log_lengthscales(&[-0.3, 0.4]).expect("valid");
        let x = points_2d(&[[0.0, 0.0], [1.2, -0.4], [0.3, 2.4]]);
        let h = 1e-6;
        let mut theta = [0.0; 2];
        rbf.get_params(&mut theta).expect("len 2");
        for i in 0..2 {
            for j in 0..2 {
                let mut plus_th = theta;
                let mut minus_th = theta;
                plus_th[j] += h;
                minus_th[j] -= h;
                let plus = RbfArdKernel::from_log_lengthscales(&plus_th).expect("valid");
                let minus = RbfArdKernel::from_log_lengthscales(&minus_th).expect("valid");
                let mut g_plus = fill(3, 0.0);
                let mut g_minus = fill(3, 0.0);
                let mut d2 = fill(3, 0.0);
                plus.grad(x.as_ref(), g_plus.as_mut(), i, Triangle::Full)
                    .expect("plus");
                minus
                    .grad(x.as_ref(), g_minus.as_mut(), i, Triangle::Full)
                    .expect("minus");
                rbf.hess(x.as_ref(), d2.as_mut(), i, j, Triangle::Full)
                    .expect("pair");
                for col in 0..3 {
                    for row in 0..3 {
                        let fd = (g_plus[(row, col)] - g_minus[(row, col)]) / (2.0 * h);
                        assert_close(d2[(row, col)], fd);
                    }
                }
            }
        }
    }

    #[test]
    fn grad_matches_finite_difference_per_dimension() {
        let rbf = RbfArdKernel::from_log_lengthscales(&[-0.3, 0.4]).expect("valid");
        let x = points_2d(&[[0.0, 0.0], [1.2, -0.4], [0.3, 2.4]]);
        let h = 1e-6;
        for dim in 0..2 {
            let mut params = [0.0; 2];
            rbf.get_params(&mut params).expect("len 2");
            params[dim] += h;
            let plus = RbfArdKernel::from_log_lengthscales(&params).expect("valid");
            params[dim] -= 2.0 * h;
            let minus = RbfArdKernel::from_log_lengthscales(&params).expect("valid");
            let mut k_plus = fill(3, 0.0);
            let mut k_minus = fill(3, 0.0);
            let mut dk = fill(3, 0.0);
            plus.apply(x.as_ref(), k_plus.as_mut(), Triangle::Full)
                .expect("shape");
            minus
                .apply(x.as_ref(), k_minus.as_mut(), Triangle::Full)
                .expect("shape");
            rbf.grad(x.as_ref(), dk.as_mut(), dim, Triangle::Full)
                .expect("index");
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
        let rbf = RbfArdKernel::new(&[1.0, 2.0]).expect("valid");
        let x = points_2d(&[[0.0, 0.0], [1.0, 0.0]]);
        let mut dk0 = fill(2, 0.0);
        let mut dk1 = fill(2, 0.0);
        rbf.grad(x.as_ref(), dk0.as_mut(), 0, Triangle::Full)
            .expect("dim 0");
        rbf.grad(x.as_ref(), dk1.as_mut(), 1, Triangle::Full)
            .expect("dim 1");
        assert_close(dk1[(1, 0)], 0.0);
        assert!(dk0[(1, 0)].abs() > 1e-8);
    }

    #[test]
    fn grad_lower_matches_full() {
        let rbf = RbfArdKernel::new(&[1.0, 0.5]).expect("valid");
        let x = points_2d(&[[0.0, 0.0], [0.8, 0.3], [1.6, -0.2]]);
        let mut full = fill(3, 0.0);
        let mut lower = fill(3, 99.0);
        rbf.grad(x.as_ref(), full.as_mut(), 1, Triangle::Full)
            .expect("index 1");
        rbf.grad(x.as_ref(), lower.as_mut(), 1, Triangle::Lower)
            .expect("index 1");
        lower_matches(lower.as_ref(), full.as_ref());
        assert_close(lower[(0, 1)], 99.0);
    }

    #[test]
    fn get_set_params_roundtrip() {
        let mut rbf = RbfArdKernel::new(&[2.0, 0.5]).expect("valid");
        let mut params = [0.0; 2];
        rbf.get_params(&mut params).expect("len 2");
        assert_close(params[0], 2.0_f64.ln());
        assert_close(params[1], 0.5_f64.ln());
        params[0] = 0.5_f64.ln();
        rbf.set_params(&params).expect("len 2");
        assert_close(rbf.lengthscale(0).expect("dim 0"), 0.5);
    }

    #[test]
    fn apply_cross_matches_square_block() {
        let rbf = RbfArdKernel::new(&[1.0, 2.0]).expect("valid");
        let train = points_2d(&[[0.0, 0.0], [1.0, 0.5]]);
        let test = points_2d(&[[0.2, -0.1], [1.0, 0.5]]);
        let mut square = fill(2, 0.0);
        rbf.apply(train.as_ref(), square.as_mut(), Triangle::Full)
            .expect("square");
        let mut cross = fill(2, 0.0);
        rbf.apply_cross(train.as_ref(), test.as_ref(), cross.as_mut())
            .expect("rect");
        // test[:, 1] == train[:, 1]
        assert_close(cross[(0, 1)], square[(0, 1)]);
        assert_close(cross[(1, 1)], square[(1, 1)]);
    }

    #[test]
    fn rejects_bad_index_dim_and_non_finite() {
        assert!(matches!(
            RbfArdKernel::new(&[1.0, 0.0]),
            Err(GprError::InvalidHyperparameter { .. })
        ));
        let rbf = RbfArdKernel::new(&[1.0, 2.0]).expect("valid");
        let x = points_2d(&[[0.0, 0.0], [1.0, 1.0]]);
        let mut dk = fill(2, 0.0);
        assert!(matches!(
            rbf.grad(x.as_ref(), dk.as_mut(), 2, Triangle::Lower),
            Err(GprError::InvalidHyperparameter { .. })
        ));
        let bad_d = Mat::from_fn(2, 3, |_, _| 0.0);
        let mut k = fill(2, 0.0);
        assert!(matches!(
            rbf.apply(bad_d.as_ref(), k.as_mut(), Triangle::Full),
            Err(GprError::DimensionMismatch { .. })
        ));
        let nan = points_2d(&[[0.0, 0.0], [f64::NAN, 1.0]]);
        assert!(matches!(
            rbf.apply(nan.as_ref(), k.as_mut(), Triangle::Full),
            Err(GprError::NonFiniteInput)
        ));
    }
}
