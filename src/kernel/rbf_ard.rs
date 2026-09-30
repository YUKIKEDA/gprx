//! ARD squared-exponential (RBF) kernel.

use super::ard::{self, ArdR2, Pick};
use super::dist::require_ard_sq_diff_shape;
use super::scalar::f64_pair;
use super::simd::{
    try_apply_rbf_ard_cache, try_apply_rbf_ard_cross, try_apply_rbf_ard_points,
    try_grad_rbf_ard_cache, try_grad_rbf_ard_points,
};
use super::{ArdLengthscales, KernelScalar, Triangle, finite_kernel, write_square};
use crate::error::GprError;
use crate::math::KernelMath;
use faer::reborrow::ReborrowMut;
use faer::{Mat, MatMut, MatRef};
use rayon::prelude::*;
use wide::f64x4;

/// ARD RBF: `k = exp( -½ Σ_d (x_d - x'_d)² / ℓ_d² )`.
///
/// Optimizer parameters are `θ_d = log(ℓ_d)` via [`ArdLengthscales`]. When every
/// `ℓ_d` equals a scalar `ℓ`, values match isotropic [`super::RbfKernel`].
/// `apply` / `grad` take the `n×d` coordinate matrix; a scalar squared-distance
/// matrix is not enough for `∂K/∂θ_d`. Amplitude is not stored here.
///
/// Cloning copies the lengthscale vectors. When
/// [`crate::DistanceCachePolicy::Cached`] is set, [`crate::Gpr`] caches raw
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
    /// Returns [`GprError::IndexOutOfRange`] if `dim` is out of range.
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
    /// Returns [`GprError::LengthMismatch`] if `out` is the wrong length.
    pub fn get_params(&self, out: &mut [f64]) -> Result<(), GprError> {
        self.lengthscales.get_params(out)
    }

    /// Replaces `θ_d` from `params`. The previous values are kept on error.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::LengthMismatch`] if `params` is the wrong
    /// length, or [`GprError::InvalidHyperparameter`] if a `θ_d` is invalid.
    pub fn set_params(&mut self, params: &[f64]) -> Result<(), GprError> {
        self.lengthscales.set_params(params)
    }

    /// Writes `k(x, x')` into `out` for the requested triangle.
    ///
    /// Writes `k(x, x')` into `out` for the requested triangle.
    ///
    /// `x` is `n×d` (rows are points). The default contract is
    /// [`Triangle::Lower`]. Entries outside that triangle are left unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`GprError`] if `x` is empty, `d` does not match the
    /// lengthscales, `out` is not `n×n`, or a coordinate is non-finite.
    pub fn apply<T: KernelScalar>(
        &self,
        x: MatRef<'_, T>,
        out: MatMut<'_, T>,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        self.apply_math::<crate::math::Accurate, T>(x, out, uplo)
    }

    pub(crate) fn apply_math<M: KernelMath, T: KernelScalar>(
        &self,
        x: MatRef<'_, T>,
        mut out: MatMut<'_, T>,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        ard::require_square_points(x, out.as_ref(), self.num_params())?;
        let w = self.lengthscales.inv_ell_sq();
        if let Some((xf, of)) = f64_pair(x, out.rb_mut())
            && try_apply_rbf_ard_points::<M>(xf, of, uplo, w)?
        {
            return Ok(());
        }
        write_square(out, uplo, |row, col| {
            rbf_value::<M, T>(ard::r2_from_coords(x, row, x, col, w, Pick::NONE)?)
        })
    }

    /// Writes rectangular `k(x, xs)` (train × test) into `out`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError`] if a matrix is empty, feature dimensions differ,
    /// `out` is the wrong shape, or a coordinate is non-finite.
    pub fn apply_cross<T: KernelScalar>(
        &self,
        x: MatRef<'_, T>,
        xs: MatRef<'_, T>,
        out: MatMut<'_, T>,
    ) -> Result<(), GprError> {
        self.apply_cross_math::<crate::math::Accurate, T>(x, xs, out)
    }

    pub(crate) fn apply_cross_math<M: KernelMath, T: KernelScalar>(
        &self,
        x: MatRef<'_, T>,
        xs: MatRef<'_, T>,
        mut out: MatMut<'_, T>,
    ) -> Result<(), GprError> {
        ard::require_cross(x, xs, out.as_ref(), self.num_params())?;
        let w = self.lengthscales.inv_ell_sq();
        if let (Some(xf), Some((xsf, of))) = (T::as_f64_ref(x), f64_pair(xs, out.rb_mut()))
            && try_apply_rbf_ard_cross::<M>(xf, xsf, of, w)?
        {
            return Ok(());
        }
        super::write_rect(out, |row, col| {
            rbf_value::<M, T>(ard::r2_from_coords(x, row, xs, col, w, Pick::NONE)?)
        })
    }

    /// Writes the stationary diagonal `k(x, x) = 1` into `out`.
    pub fn fill_diag<T: KernelScalar>(&self, out: &mut [T]) {
        out.fill(T::from_f64(1.0));
    }

    /// Writes `∂K/∂θ_d` for `θ_d = log(ℓ_d)` into `d_k`.
    ///
    /// `∂k/∂θ_d = k · (x_d - x'_d)² / ℓ_d²`. This is not `∂k/∂ℓ_d`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::IndexOutOfRange`] if `param_idx` is out of
    /// range, or the same shape / non-finite errors as [`Self::apply`].
    pub fn grad<T: KernelScalar>(
        &self,
        x: MatRef<'_, T>,
        d_k: MatMut<'_, T>,
        param_idx: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        self.grad_math::<crate::math::Accurate, T>(x, d_k, param_idx, uplo)
    }

    pub(crate) fn grad_math<M: KernelMath, T: KernelScalar>(
        &self,
        x: MatRef<'_, T>,
        mut d_k: MatMut<'_, T>,
        param_idx: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        ard::require_param(NAME, param_idx, self.num_params())?;
        ard::require_square_points(x, d_k.as_ref(), self.num_params())?;
        let w = self.lengthscales.inv_ell_sq();
        if let Some((xf, of)) = f64_pair(x, d_k.rb_mut())
            && try_grad_rbf_ard_points::<M>(xf, of, uplo, w, param_idx)?
        {
            return Ok(());
        }
        write_square(d_k, uplo, |row, col| {
            rbf_grad::<M, T>(ard::r2_from_coords(
                x,
                row,
                x,
                col,
                w,
                Pick::one(param_idx),
            )?)
        })
    }

    pub(crate) fn apply_from_sq_diff<M: KernelMath, T: KernelScalar>(
        &self,
        cache: MatRef<'_, T>,
        mut out: MatMut<'_, T>,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        let n = ard::require_square_out(out.as_ref())?;
        require_ard_sq_diff_shape(cache, n, self.num_params())?;
        let w = self.lengthscales.inv_ell_sq();
        if let Some((cf, of)) = f64_pair(cache, out.rb_mut())
            && try_apply_rbf_ard_cache::<M>(cf, of, uplo, w)?
        {
            return Ok(());
        }
        write_square(out, uplo, |row, col| {
            rbf_value::<M, T>(ard::r2_from_cache(cache, n, row, col, w, Pick::NONE)?)
        })
    }

    pub(crate) fn grad_from_sq_diff<M: KernelMath, T: KernelScalar>(
        &self,
        cache: MatRef<'_, T>,
        mut d_k: MatMut<'_, T>,
        param_idx: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        ard::require_param(NAME, param_idx, self.num_params())?;
        let n = ard::require_square_out(d_k.as_ref())?;
        require_ard_sq_diff_shape(cache, n, self.num_params())?;
        let w = self.lengthscales.inv_ell_sq();
        if let Some((cf, of)) = f64_pair(cache, d_k.rb_mut())
            && try_grad_rbf_ard_cache::<M>(cf, of, uplo, w, param_idx)?
        {
            return Ok(());
        }
        write_square(d_k, uplo, |row, col| {
            rbf_grad::<M, T>(ard::r2_from_cache(
                cache,
                n,
                row,
                col,
                w,
                Pick::one(param_idx),
            )?)
        })
    }

    /// Writes `∂²K/∂θ_i ∂θ_j` for `θ_d = log(ℓ_d)` into `d2_k`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::IndexOutOfRange`] if `i` or `j` is out of
    /// range, or the same shape / non-finite errors as [`Self::apply`].
    pub fn hess<T: KernelScalar>(
        &self,
        x: MatRef<'_, T>,
        d2_k: MatMut<'_, T>,
        i: usize,
        j: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        self.hess_math::<crate::math::Accurate, T>(x, d2_k, i, j, uplo)
    }

    pub(crate) fn hess_math<M: KernelMath, T: KernelScalar>(
        &self,
        x: MatRef<'_, T>,
        d2_k: MatMut<'_, T>,
        i: usize,
        j: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        ard::require_param_pair(NAME, i, j, self.num_params())?;
        let w = self.lengthscales.inv_ell_sq();
        ard::write_from_points(x, d2_k, self.num_params(), uplo, |row, col| {
            rbf_hess::<M, T>(
                ard::r2_from_coords(x, row, x, col, w, Pick::pair(i, j))?,
                i == j,
            )
        })
    }

    pub(crate) fn hess_from_sq_diff<M: KernelMath, T: KernelScalar>(
        &self,
        cache: MatRef<'_, T>,
        d2_k: MatMut<'_, T>,
        i: usize,
        j: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        ard::require_param_pair(NAME, i, j, self.num_params())?;
        let w = self.lengthscales.inv_ell_sq();
        ard::write_from_cache(cache, d2_k, self.num_params(), uplo, |n, row, col| {
            rbf_hess::<M, T>(
                ard::r2_from_cache(cache, n, row, col, w, Pick::pair(i, j))?,
                i == j,
            )
        })
    }

    /// Writes `∂K(X1, X2)/∂X2[*, dim]` into `d_k`.
    ///
    /// `∂k/∂x2_e = k (x1_e - x2_e) / ℓ_e²`.
    ///
    /// # Errors
    ///
    /// Same shape / non-finite errors as [`RbfKernel::grad_wrt_coord_dim`](super::RbfKernel::grad_wrt_coord_dim), or
    /// [`GprError::IndexOutOfRange`] when `dim` does not match `ℓ_d`.
    pub fn grad_wrt_coord_dim<T: KernelScalar>(
        &self,
        x1: MatRef<'_, T>,
        x2: MatRef<'_, T>,
        d_k: MatMut<'_, T>,
        dim: usize,
    ) -> Result<(), GprError> {
        self.grad_wrt_coord_dim_math::<crate::math::Accurate, T>(x1, x2, d_k, dim)
    }

    pub(crate) fn grad_wrt_coord_dim_math<M: KernelMath, T: KernelScalar>(
        &self,
        x1: MatRef<'_, T>,
        x2: MatRef<'_, T>,
        d_k: MatMut<'_, T>,
        dim: usize,
    ) -> Result<(), GprError> {
        super::require_coord_grad(x1, x2, d_k.as_ref(), dim)?;
        if dim >= self.num_params() {
            return Err(GprError::IndexOutOfRange {
                reason: format!(
                    "coordinate dimension {dim} is out of range for d={}",
                    self.num_params()
                ),
            });
        }
        let inv_ell_sq = self.lengthscales.inv_ell_sq();
        let w_dim = T::from_f64(inv_ell_sq[dim]);
        super::write_rect(d_k, |row, col| {
            let t = ard::r2_from_coords(x1, row, x2, col, inv_ell_sq, Pick::NONE)?;
            let delta = x1[(row, dim)] - x2[(col, dim)];
            Ok(ard_d1::<M, T>(t.r2) * delta * w_dim)
        })
    }

    pub(crate) fn hess_wrt_coord_dims<M: KernelMath, T: KernelScalar>(
        &self,
        x1: MatRef<'_, T>,
        x2: MatRef<'_, T>,
        d2_k: MatMut<'_, T>,
        dim_a: usize,
        dim_b: usize,
    ) -> Result<(), GprError> {
        self.coord_hess::<M, T>(x1, x2, d2_k, (dim_a, dim_b), T::from_f64(-1.0))
    }

    pub(crate) fn hess_wrt_coord_mixed<M: KernelMath, T: KernelScalar>(
        &self,
        x1: MatRef<'_, T>,
        x2: MatRef<'_, T>,
        d2_k: MatMut<'_, T>,
        dim_x1: usize,
        dim_x2: usize,
    ) -> Result<(), GprError> {
        self.coord_hess::<M, T>(x1, x2, d2_k, (dim_x1, dim_x2), T::from_f64(1.0))
    }

    /// `sign · (−k'' Δ_a Δ_b w_a w_b + [a = b] k' w_a)`: `sign = −1` for two
    /// derivatives in `X2`, `+1` for one in `X1` and one in `X2`.
    fn coord_hess<M: KernelMath, T: KernelScalar>(
        &self,
        x1: MatRef<'_, T>,
        x2: MatRef<'_, T>,
        d2_k: MatMut<'_, T>,
        (dim_a, dim_b): (usize, usize),
        sign: T,
    ) -> Result<(), GprError> {
        super::require_coord_grad(x1, x2, d2_k.as_ref(), dim_a)?;
        super::require_coord_grad(x1, x2, d2_k.as_ref(), dim_b)?;
        let inv_ell_sq = self.lengthscales.inv_ell_sq();
        let wa = T::from_f64(inv_ell_sq[dim_a]);
        let wb = T::from_f64(inv_ell_sq[dim_b]);
        super::write_rect(d2_k, |row, col| {
            let r2 = ard::r2_from_coords(x1, row, x2, col, inv_ell_sq, Pick::NONE)?.r2;
            let da = x1[(row, dim_a)] - x2[(col, dim_a)];
            let db = x1[(row, dim_b)] - x2[(col, dim_b)];
            let (d1, d2) = if M::ACCURATE {
                let k = ard_value::<M, T>(r2);
                (k, k)
            } else {
                let jet = M::jet(T::from_f64(-0.5) * r2);
                (jet.d1, jet.d2)
            };
            let mut value = -sign * d2 * da * db * wa * wb;
            if dim_a == dim_b {
                value += sign * d1 * wa;
            }
            Ok(value)
        })
    }

    pub(crate) fn hess_theta_coord_dim<M: KernelMath, T: KernelScalar>(
        &self,
        x1: MatRef<'_, T>,
        x2: MatRef<'_, T>,
        d2_k: MatMut<'_, T>,
        param_idx: usize,
        dim: usize,
    ) -> Result<(), GprError> {
        ard::require_param(NAME, param_idx, self.num_params())?;
        super::require_coord_grad(x1, x2, d2_k.as_ref(), dim)?;
        let inv_ell_sq = self.lengthscales.inv_ell_sq();
        let w_theta = T::from_f64(inv_ell_sq[param_idx]);
        let w_dim = T::from_f64(inv_ell_sq[dim]);
        let minus_two = T::from_f64(-2.0);
        super::write_rect(d2_k, |row, col| {
            let r2 = ard::r2_from_coords(x1, row, x2, col, inv_ell_sq, Pick::NONE)?.r2;
            let delta_theta = x1[(row, param_idx)] - x2[(col, param_idx)];
            let delta_dim = x1[(row, dim)] - x2[(col, dim)];
            let value = if M::ACCURATE {
                let k = ard_value::<M, T>(r2);
                let dk_dtheta = k * delta_theta * delta_theta * w_theta;
                let mut value = dk_dtheta * delta_dim * w_dim;
                if param_idx == dim {
                    value += k * delta_dim * (minus_two * w_dim);
                }
                value
            } else {
                let jet = M::jet(T::from_f64(-0.5) * r2);
                let dim_term = delta_theta * delta_theta * w_theta;
                let mut value = jet.d2 * dim_term * delta_dim * w_dim;
                if param_idx == dim {
                    value += jet.d1 * delta_dim * (minus_two * w_dim);
                }
                value
            };
            Ok(value)
        })
    }

    pub(crate) fn grad_cross_from_coords<M: KernelMath, T: KernelScalar>(
        &self,
        x1: MatRef<'_, T>,
        x2: MatRef<'_, T>,
        mut d_k: MatMut<'_, T>,
        param_idx: usize,
    ) -> Result<(), GprError> {
        ard::require_param(NAME, param_idx, self.num_params())?;
        super::require_coord_grad(x1, x2, d_k.as_ref(), 0)?;
        let w = self.lengthscales.inv_ell_sq();
        if let (Some(af), Some((bf, of))) = (T::as_f64_ref(x1), f64_pair(x2, d_k.rb_mut()))
            && try_grad_rbf_ard_cross::<M>(af, bf, of, w, param_idx)?
        {
            return Ok(());
        }
        super::write_rect(d_k, |row, col| {
            rbf_grad::<M, T>(ard::r2_from_coords(
                x1,
                row,
                x2,
                col,
                w,
                Pick::one(param_idx),
            )?)
        })
    }

    /// One pass of `∂k/∂θ_d` for every lengthscale. Same values as
    /// [`Self::grad_cross_from_coords`] called once per `d`.
    pub(crate) fn grad_cross_all_from_coords<M: KernelMath, T: KernelScalar>(
        &self,
        x1: MatRef<'_, T>,
        x2: MatRef<'_, T>,
    ) -> Result<Vec<Mat<T>>, GprError> {
        let d = self.num_params();
        let mut out: Vec<Mat<T>> = (0..d).map(|_| Mat::zeros(x1.nrows(), x2.nrows())).collect();
        if d == 0 {
            return Ok(out);
        }
        super::require_coord_grad(x1, x2, out[0].as_ref(), 0)?;
        let w = self.lengthscales.inv_ell_sq();
        let slots: Option<Vec<MatMut<'_, f64>>> =
            out.iter_mut().map(|m| T::as_f64_mut(m.as_mut())).collect();
        if let (Some(af), Some(bf), Some(mut slots)) = (T::as_f64_ref(x1), T::as_f64_ref(x2), slots)
            && try_grad_rbf_ard_cross_all::<M>(af, bf, &mut slots, w)?
        {
            return Ok(out);
        }
        let mut terms = vec![T::from_f64(0.0); d];
        for col in 0..x2.nrows() {
            for row in 0..x1.nrows() {
                let k = ard_grad_terms::<M, T>(x1, row, x2, col, w, &mut terms)?;
                for (dest, &term) in out.iter_mut().zip(&terms) {
                    dest[(row, col)] = k * term;
                }
            }
        }
        Ok(out)
    }

    pub(crate) fn hess_cross_from_coords<M: KernelMath, T: KernelScalar>(
        &self,
        x1: MatRef<'_, T>,
        x2: MatRef<'_, T>,
        d2_k: MatMut<'_, T>,
        i: usize,
        j: usize,
    ) -> Result<(), GprError> {
        ard::require_param_pair(NAME, i, j, self.num_params())?;
        super::require_coord_grad(x1, x2, d2_k.as_ref(), 0)?;
        let w = self.lengthscales.inv_ell_sq();
        super::write_rect(d2_k, |row, col| {
            rbf_hess::<M, T>(
                ard::r2_from_coords(x1, row, x2, col, w, Pick::pair(i, j))?,
                i == j,
            )
        })
    }
}

const NAME: &str = "RBF";

fn try_grad_rbf_ard_cross<M: KernelMath>(
    x1: MatRef<'_, f64>,
    x2: MatRef<'_, f64>,
    d_k: MatMut<'_, f64>,
    inv_ell_sq: &[f64],
    param_idx: usize,
) -> Result<bool, GprError> {
    if param_idx >= inv_ell_sq.len() {
        return Ok(false);
    }
    let mut write = vec![false; inv_ell_sq.len()];
    write[param_idx] = true;
    let mut one = [d_k];
    try_fill_ard_cross::<M>(x1, x2, &mut one, inv_ell_sq, &write)
}

fn try_grad_rbf_ard_cross_all<M: KernelMath>(
    x1: MatRef<'_, f64>,
    x2: MatRef<'_, f64>,
    slots: &mut [MatMut<'_, f64>],
    inv_ell_sq: &[f64],
) -> Result<bool, GprError> {
    if slots.len() != inv_ell_sq.len() {
        return Ok(false);
    }
    let write = vec![true; inv_ell_sq.len()];
    try_fill_ard_cross::<M>(x1, x2, slots, inv_ell_sq, &write)
}

/// `∂k/∂θ_d = k · (Δ_d)² / ℓ_d²` with one `exp` for every lengthscale.
fn try_fill_ard_cross<M: KernelMath>(
    x1: MatRef<'_, f64>,
    x2: MatRef<'_, f64>,
    out: &mut [MatMut<'_, f64>],
    inv_ell_sq: &[f64],
    write: &[bool],
) -> Result<bool, GprError> {
    let m = x1.nrows();
    let n = x2.nrows();
    let d = inv_ell_sq.len();
    if d == 0 || write.len() != d || out.len() != d || x1.ncols() != d || x2.ncols() != d {
        return Ok(false);
    }
    if !ard_unit_cols(x1) || !ard_unit_cols(x2) {
        return Ok(false);
    }
    for dest in out.iter() {
        if dest.nrows() != m || dest.ncols() != n || !ard_unit_cols(dest.as_ref()) {
            return Ok(false);
        }
    }
    for dim in 0..d {
        ard_finite(ard_col(x1, dim)?)?;
        ard_finite(ard_col(x2, dim)?)?;
    }
    if n <= 1024 {
        write_ard_cross::<M>(x1, x2, out, inv_ell_sq, write, 0, n)?;
        return Ok(true);
    }
    let n_parts = super::dist::worker_count();
    let mut slots = Vec::with_capacity(d);
    for mat in out.iter_mut() {
        slots.push(packed_mut(mat));
    }
    let shared = ShareBases(slots);
    let results: Vec<Result<(), GprError>> = (0..n_parts)
        .into_par_iter()
        .map(|idx| {
            let (start, len) = super::dist::col_chunk(n, idx, n_parts);
            write_packed::<M>(
                x1,
                x2,
                shared.slots(),
                inv_ell_sq,
                write,
                ArdSpan {
                    x_begin: start,
                    dest_col: start,
                    len,
                },
            )
        })
        .collect();
    for result in results {
        result?;
    }
    Ok(true)
}

fn write_ard_cross<M: KernelMath>(
    x1: MatRef<'_, f64>,
    x2: MatRef<'_, f64>,
    dest: &mut [MatMut<'_, f64>],
    inv_ell_sq: &[f64],
    write: &[bool],
    start: usize,
    len: usize,
) -> Result<(), GprError> {
    let d = inv_ell_sq.len();
    for mat in dest.iter() {
        if mat.ncols() > 0 && mat.row_stride() != 1 {
            return Err(GprError::UnsupportedKernelOperation {
                reason: "expected unit row-stride for ARD cross grad".to_owned(),
            });
        }
    }
    let mut packed = Vec::with_capacity(d);
    for mat in dest.iter_mut() {
        packed.push(packed_mut(mat));
    }
    write_packed::<M>(
        x1,
        x2,
        &packed,
        inv_ell_sq,
        write,
        ArdSpan {
            x_begin: start,
            dest_col: 0,
            len,
        },
    )
}

fn write_packed<M: KernelMath>(
    x1: MatRef<'_, f64>,
    x2: MatRef<'_, f64>,
    packed: &[PackedMut],
    inv_ell_sq: &[f64],
    write: &[bool],
    span: ArdSpan,
) -> Result<(), GprError> {
    let m = x1.nrows();
    let d = inv_ell_sq.len();
    let len = span.len;
    let mut r2 = vec![0.0; len];
    let mut scratch = vec![0.0; len];
    let mut saved: Vec<Vec<f64>> = write
        .iter()
        .map(|flag| if *flag { vec![0.0; len] } else { Vec::new() })
        .collect();
    let mut k = vec![0.0; len];
    let half = f64x4::new([-0.5; 4]);
    let mut x_dim = Vec::with_capacity(d);
    for dim in 0..d {
        x_dim.push(ard_col(x2, dim)?);
    }
    for row in 0..m {
        r2.fill(0.0);
        for dim in 0..d {
            let z = ard_col(x1, dim)?[row];
            let acc = if write[dim] {
                &mut saved[dim]
            } else {
                &mut scratch
            };
            acc.fill(0.0);
            add_weighted_sq(
                &x_dim[dim][span.x_begin..span.x_begin + len],
                z,
                inv_ell_sq[dim],
                acc,
            );
            add_slice(acc, &mut r2);
        }
        exp_scaled::<M>(&r2, &mut k, half)?;
        for dim in 0..d {
            if !write[dim] {
                continue;
            }
            let src = &saved[dim];
            let mut i = 0;
            while i + 4 <= len {
                let dk = load4(&k, i) * load4(src, i);
                if !all_finite4(dk) {
                    return Err(GprError::NonFiniteKernelValue);
                }
                let lanes = dk.to_array();
                let col = span.dest_col + i;
                store_packed(&packed[dim], row, col, lanes[0]);
                store_packed(&packed[dim], row, col + 1, lanes[1]);
                store_packed(&packed[dim], row, col + 2, lanes[2]);
                store_packed(&packed[dim], row, col + 3, lanes[3]);
                i += 4;
            }
            while i < len {
                let dk = k[i] * src[i];
                if !dk.is_finite() {
                    return Err(GprError::NonFiniteKernelValue);
                }
                store_packed(&packed[dim], row, span.dest_col + i, dk);
                i += 1;
            }
        }
    }
    Ok(())
}

struct ArdSpan {
    x_begin: usize,
    dest_col: usize,
    len: usize,
}

struct PackedMut {
    ptr: *mut f64,
    stride: isize,
}

struct ShareBases(Vec<PackedMut>);

impl ShareBases {
    fn slots(&self) -> &[PackedMut] {
        &self.0
    }
}

/// # Safety
///
/// Each pointer is a column-major matrix. Parallel callers write disjoint
/// columns of those matrices and do not read the columns they write.
unsafe impl Send for ShareBases {}

/// # Safety
///
/// Each pointer is a column-major matrix. Parallel callers write disjoint
/// columns of those matrices and do not read the columns they write.
unsafe impl Sync for ShareBases {}

fn packed_mut(mat: &mut MatMut<'_, f64>) -> PackedMut {
    let view = mat.rb_mut();
    debug_assert!(view.ncols() == 0 || view.row_stride() == 1);
    PackedMut {
        ptr: view.as_ptr_mut(),
        stride: view.col_stride(),
    }
}

/// # Safety
///
/// `slot` addresses a column-major matrix whose row stride is `+1`.
/// `row` is inside that matrix and `col` is inside its column count.
#[inline(always)]
fn store_packed(slot: &PackedMut, row: usize, col: usize, value: f64) {
    // SAFETY: row stride is +1 and `(row, col)` is inside this matrix.
    unsafe {
        *slot.ptr.offset(row as isize + col as isize * slot.stride) = value;
    }
}

fn add_slice(src: &[f64], acc: &mut [f64]) {
    let mut i = 0;
    while i + 4 <= src.len() {
        store4(acc, i, load4(acc, i) + load4(src, i));
        i += 4;
    }
    while i < src.len() {
        acc[i] += src[i];
        i += 1;
    }
}

fn ard_unit_cols(mat: MatRef<'_, f64>) -> bool {
    mat.ncols() == 0 || mat.col(0).try_as_col_major().is_some()
}

fn ard_col<'a>(mat: MatRef<'a, f64>, col: usize) -> Result<&'a [f64], GprError> {
    mat.col(col)
        .try_as_col_major()
        .map(|c| c.as_slice())
        .ok_or_else(|| GprError::UnsupportedKernelOperation {
            reason: "expected unit row-stride for ARD cross grad".to_owned(),
        })
}

fn ard_finite(values: &[f64]) -> Result<(), GprError> {
    if values.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(GprError::NonFiniteInput)
    }
}

fn load4(src: &[f64], i: usize) -> f64x4 {
    f64x4::new([src[i], src[i + 1], src[i + 2], src[i + 3]])
}

fn store4(dest: &mut [f64], i: usize, v: f64x4) {
    let a = v.to_array();
    dest[i] = a[0];
    dest[i + 1] = a[1];
    dest[i + 2] = a[2];
    dest[i + 3] = a[3];
}

fn all_finite4(v: f64x4) -> bool {
    let a = v.to_array();
    a[0].is_finite() && a[1].is_finite() && a[2].is_finite() && a[3].is_finite()
}

fn add_weighted_sq(x: &[f64], x0: f64, w: f64, acc: &mut [f64]) {
    let x0v = f64x4::new([x0; 4]);
    let wv = f64x4::new([w; 4]);
    let mut i = 0;
    while i + 4 <= x.len() {
        let d = load4(x, i) - x0v;
        let av = load4(acc, i);
        store4(acc, i, av + d * d * wv);
        i += 4;
    }
    while i < x.len() {
        let d = x[i] - x0;
        acc[i] += d * d * w;
        i += 1;
    }
}

fn ard_value<M: KernelMath, T: KernelScalar>(r2: T) -> T {
    M::exp(T::from_f64(-0.5) * r2)
}

fn ard_d1<M: KernelMath, T: KernelScalar>(r2: T) -> T {
    if M::ACCURATE {
        (T::from_f64(-0.5) * r2).exp()
    } else {
        M::jet(T::from_f64(-0.5) * r2).d1
    }
}

fn ard_hess_terms<M: KernelMath, T: KernelScalar>(r2: T, dim_i: T, dim_j: T, same: bool) -> T {
    let two = T::from_f64(2.0);
    if M::ACCURATE {
        let k = ard_value::<M, T>(r2);
        if same {
            k * dim_i * (dim_i - two)
        } else {
            k * dim_i * dim_j
        }
    } else {
        let jet = M::jet(T::from_f64(-0.5) * r2);
        if same {
            dim_i * (jet.d2 * dim_i - two * jet.d1)
        } else {
            jet.d2 * dim_i * dim_j
        }
    }
}

fn rbf_value<M: KernelMath, T: KernelScalar>(t: ArdR2<T>) -> Result<T, GprError> {
    finite_kernel(ard_value::<M, T>(t.r2))
}

fn rbf_grad<M: KernelMath, T: KernelScalar>(t: ArdR2<T>) -> Result<T, GprError> {
    finite_kernel(ard_d1::<M, T>(t.r2) * t.dim_i)
}

fn rbf_hess<M: KernelMath, T: KernelScalar>(t: ArdR2<T>, same: bool) -> Result<T, GprError> {
    finite_kernel(ard_hess_terms::<M, T>(t.r2, t.dim_i, t.dim_j, same))
}

/// Writes `w_d Δ_d²` for every `d` into `terms` and returns `k'(r²)`.
fn ard_grad_terms<M: KernelMath, T: KernelScalar>(
    x: MatRef<'_, T>,
    row: usize,
    xs: MatRef<'_, T>,
    col: usize,
    inv_ell_sq: &[f64],
    terms: &mut [T],
) -> Result<T, GprError> {
    let mut r2 = T::from_f64(0.0);
    for (dim, &w) in inv_ell_sq.iter().enumerate() {
        let diff = x[(row, dim)] - xs[(col, dim)];
        if !diff.is_finite() {
            return Err(GprError::NonFiniteInput);
        }
        let term = diff * diff * T::from_f64(w);
        r2 += term;
        terms[dim] = term;
    }
    if !r2.is_finite() {
        return Err(GprError::NonFiniteKernelValue);
    }
    finite_kernel(ard_d1::<M, T>(r2))
}

fn exp_scaled<M: KernelMath>(src: &[f64], dest: &mut [f64], scale: f64x4) -> Result<(), GprError> {
    let mut i = 0;
    while i + 4 <= src.len() {
        let z = load4(src, i) * scale;
        let v = if M::ACCURATE { z.exp() } else { M::d1_f64x4(z) };
        if !all_finite4(v) {
            return Err(GprError::NonFiniteKernelValue);
        }
        store4(dest, i, v);
        i += 4;
    }
    while i < src.len() {
        let v = ard_d1::<M, f64>(src[i]);
        if !v.is_finite() {
            return Err(GprError::NonFiniteKernelValue);
        }
        dest[i] = v;
        i += 1;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::RbfArdKernel;
    use crate::error::GprError;
    use crate::kernel::{RbfKernel, Triangle};
    use faer::{Mat, MatRef};

    const TOL: f64 = 1e-10;

    use crate::test_check::{assert_close, assert_lower_close, assert_send_sync, fill, points_2d};

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
        assert_close(k[(0, 0)], 1.0, TOL);
        assert_close(k[(1, 1)], 1.0, TOL);
        assert_close(k[(2, 2)], 1.0, TOL);
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
        assert_close(k[(1, 0)], (-0.5_f64).exp(), TOL);
        assert_close(k[(2, 0)], (-0.5_f64).exp(), TOL);
        assert_close(k[(2, 1)], (-0.5_f64 * (1.0 + 1.0)).exp(), TOL);
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
                assert_close(k_ard[(row, col)], k_iso[(row, col)], TOL);
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
        rbf.apply_from_sq_diff::<crate::math::Accurate, _>(
            cache.as_ref(),
            from_cache.as_mut(),
            Triangle::Lower,
        )
        .expect("cache");
        assert_lower_close(from_cache.as_ref(), from_points.as_ref(), TOL);
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
            rbf.grad_from_sq_diff::<crate::math::Accurate, _>(
                cache.as_ref(),
                from_cache.as_mut(),
                param_idx,
                Triangle::Lower,
            )
            .expect("cache");
            assert_lower_close(from_cache.as_ref(), from_points.as_ref(), TOL);
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
                assert_close(k[(row, col)], k[(col, row)], TOL);
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
        assert_lower_close(lower.as_ref(), full.as_ref(), TOL);
        assert_close(lower[(0, 1)], sentinel, TOL);
        assert_close(lower[(0, 2)], sentinel, TOL);
        assert_close(lower[(1, 2)], sentinel, TOL);
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
                assert_close(upper[(row, col)], full[(row, col)], TOL);
            }
        }
        assert_close(upper[(1, 0)], -1.0, TOL);
        assert_close(upper[(2, 0)], -1.0, TOL);
        assert_close(upper[(2, 1)], -1.0, TOL);
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
                        assert_close(d2[(row, col)], fd, TOL);
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
                    assert_close(dk[(row, col)], fd, TOL);
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
        assert_close(dk1[(1, 0)], 0.0, TOL);
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
        assert_lower_close(lower.as_ref(), full.as_ref(), TOL);
        assert_close(lower[(0, 1)], 99.0, TOL);
    }

    #[test]
    fn get_set_params_roundtrip() {
        let mut rbf = RbfArdKernel::new(&[2.0, 0.5]).expect("valid");
        let mut params = [0.0; 2];
        rbf.get_params(&mut params).expect("len 2");
        assert_close(params[0], 2.0_f64.ln(), TOL);
        assert_close(params[1], 0.5_f64.ln(), TOL);
        params[0] = 0.5_f64.ln();
        rbf.set_params(&params).expect("len 2");
        assert_close(rbf.lengthscale(0).expect("dim 0"), 0.5, TOL);
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
        assert_close(cross[(0, 1)], square[(0, 1)], TOL);
        assert_close(cross[(1, 1)], square[(1, 1)], TOL);
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
            Err(GprError::IndexOutOfRange { .. })
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
