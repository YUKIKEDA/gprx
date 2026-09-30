//! Isotropic squared-exponential (RBF) kernel.

use super::lengthscale::{validate_lengthscale, validate_log_lengthscale};
use super::scalar::f64_pair;
use super::simd::{try_apply_rbf, try_apply_rbf_cross, try_grad_rbf};
use super::{
    KernelScalar, Triangle, finite_dist, write_dense, write_rect_from_coords,
    write_square_from_coords, write_triangle,
};
use crate::error::GprError;
use crate::math::{Accurate, ExpJet, KernelMath};
use crate::param::{BoundedParam, Interval};
use faer::reborrow::ReborrowMut;
use faer::{MatMut, MatRef};
use wide::f64x4;

/// Isotropic RBF: `k = exp( -‖x-x'‖² / (2ℓ²) )`.
///
/// The optimizer parameter is `θ = log(ℓ)`. Amplitude is not stored here;
/// compose with [`super::ConstantKernel`] when a signal variance is needed.
/// `dist` is the matrix of squared Euclidean distances. Column-major views
/// with unit row stride use `wide::f64x4` for [`Self::apply`],
/// [`Self::apply_cross`], and [`Self::grad`].
///
/// # Examples
///
/// ```rust
/// use gprx::kernel::RbfKernel;
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let rbf = RbfKernel::new(1.5)?;
/// assert!(rbf.lengthscale() > 0.0);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RbfKernel {
    lengthscale: BoundedParam,
}

impl RbfKernel {
    /// Builds an RBF kernel from a positive finite lengthscale `ℓ`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if `lengthscale` is not
    /// finite or not strictly positive.
    pub fn new(lengthscale: f64) -> Result<Self, GprError> {
        validate_lengthscale(lengthscale)?;
        Ok(Self {
            lengthscale: BoundedParam::default_positive(lengthscale)?,
        })
    }

    /// Builds an RBF kernel from `θ = log(ℓ)`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if `θ` is not finite, if
    /// `exp(θ)` overflows, or if `exp(θ)` underflows to zero.
    pub fn from_log_lengthscale(log_lengthscale: f64) -> Result<Self, GprError> {
        let log_lengthscale = validate_log_lengthscale(log_lengthscale)?;
        Ok(Self {
            lengthscale: BoundedParam::default_positive(log_lengthscale.exp())?,
        })
    }

    /// Returns `ℓ = exp(θ)`.
    pub fn lengthscale(&self) -> f64 {
        self.lengthscale.value()
    }

    /// Returns `θ = log(ℓ)`.
    pub fn log_lengthscale(&self) -> f64 {
        self.lengthscale.ln()
    }

    /// Returns the open interval on `ℓ`.
    pub fn bounds(&self) -> Interval {
        self.lengthscale.interval()
    }

    /// Rebuilds this kernel with a new interval on `ℓ`.
    ///
    /// # Errors
    ///
    /// Returns [`crate::IntervalError`] if the current `ℓ` is not strictly
    /// inside `interval`.
    pub fn with_bounds(self, interval: Interval) -> Result<Self, crate::IntervalError> {
        Ok(Self {
            lengthscale: self.lengthscale.with_interval(interval)?,
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
        crate::data::require_count(out.len(), 1, "RBF parameter")?;
        out[0] = self.lengthscale.ln();
        Ok(())
    }

    /// Replaces `θ` from a length-1 slice.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::LengthMismatch`] if `params` is not length 1,
    /// or [`GprError::InvalidHyperparameter`] if the new `θ` is invalid.
    pub fn set_params(&mut self, params: &[f64]) -> Result<(), GprError> {
        crate::data::require_count(params.len(), 1, "RBF parameter")?;
        let log_lengthscale = validate_log_lengthscale(params[0])?;
        self.lengthscale = BoundedParam::new(log_lengthscale.exp(), self.lengthscale.interval())?;
        Ok(())
    }

    /// Writes `k(dist)` into `out` for the requested triangle.
    ///
    /// The default contract is [`Triangle::Lower`]. Entries outside that
    /// triangle are left unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`GprError`] if the matrices are empty, not square, or size
    /// mismatched, or if `dist` contains a non-finite value.
    pub fn apply<T: KernelScalar>(
        &self,
        dist: MatRef<'_, T>,
        out: MatMut<'_, T>,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        self.apply_math::<Accurate, _>(dist, out, uplo)
    }

    pub(crate) fn apply_math<M: KernelMath, T: KernelScalar>(
        &self,
        dist: MatRef<'_, T>,
        mut out: MatMut<'_, T>,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        let inv_two_ell_sq = 0.5 / (self.lengthscale() * self.lengthscale());
        if let Some((d, o)) = f64_pair(dist, out.rb_mut())
            && try_apply_rbf::<M>(d, o, uplo, inv_two_ell_sq)?
        {
            return Ok(());
        }
        let inv_two_ell_sq = T::from_f64(inv_two_ell_sq);
        write_triangle(dist, out, uplo, |d| {
            rbf_from_sq_dist::<M, _>(d, inv_two_ell_sq)
        })
    }

    /// Writes rectangular `k(dist)` into `out` (train × test).
    ///
    /// # Errors
    ///
    /// Returns [`GprError`] if the matrices are empty, size mismatched, or if
    /// `dist` contains a non-finite value.
    pub fn apply_cross<T: KernelScalar>(
        &self,
        dist: MatRef<'_, T>,
        out: MatMut<'_, T>,
    ) -> Result<(), GprError> {
        self.apply_cross_math::<Accurate, _>(dist, out)
    }

    pub(crate) fn apply_cross_math<M: KernelMath, T: KernelScalar>(
        &self,
        dist: MatRef<'_, T>,
        mut out: MatMut<'_, T>,
    ) -> Result<(), GprError> {
        let inv_two_ell_sq = 0.5 / (self.lengthscale() * self.lengthscale());
        if let Some((d, o)) = f64_pair(dist, out.rb_mut())
            && try_apply_rbf_cross::<M>(d, o, inv_two_ell_sq)?
        {
            return Ok(());
        }
        let inv_two_ell_sq = T::from_f64(inv_two_ell_sq);
        write_dense(dist, out, |d| rbf_from_sq_dist::<M, _>(d, inv_two_ell_sq))
    }

    /// Writes the stationary diagonal `k(x, x) = 1` into `out`.
    pub fn fill_diag<T: KernelScalar>(&self, out: &mut [T]) {
        out.fill(T::from_f64(1.0));
    }

    /// Writes `∂K/∂θ` for `θ = log(ℓ)` into `d_k`.
    ///
    /// `∂k/∂θ = k · ‖x-x'‖² / ℓ²`. This is not `∂k/∂ℓ`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::IndexOutOfRange`] if `param_idx` is not 0, or
    /// the same shape / non-finite errors as [`Self::apply`].
    pub fn grad<T: KernelScalar>(
        &self,
        dist: MatRef<'_, T>,
        d_k: MatMut<'_, T>,
        param_idx: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        self.grad_math::<Accurate, _>(dist, d_k, param_idx, uplo)
    }

    pub(crate) fn grad_math<M: KernelMath, T: KernelScalar>(
        &self,
        dist: MatRef<'_, T>,
        mut d_k: MatMut<'_, T>,
        param_idx: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        require_rbf_param_idx(param_idx)?;
        let (inv_two_ell_sq, inv_ell_sq) = self.inv_scales();
        if let Some((d, o)) = f64_pair(dist, d_k.rb_mut())
            && try_grad_rbf::<M>(d, o, uplo, inv_two_ell_sq, inv_ell_sq)?
        {
            return Ok(());
        }
        let (inv_two_ell_sq, inv_ell_sq) = (T::from_f64(inv_two_ell_sq), T::from_f64(inv_ell_sq));
        write_triangle(dist, d_k, uplo, |d| {
            rbf_grad_from_sq_dist::<M, _>(d, inv_two_ell_sq, inv_ell_sq)
        })
    }

    /// Writes `∂²K/∂θ²` for `θ = log(ℓ)` into `d2_k`.
    ///
    /// `∂²k/∂θ² = k · (s/ℓ²) · (s/ℓ² − 2)` where `s = ‖x-x'‖²`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::IndexOutOfRange`] if `i` or `j` is not 0, or
    /// the same shape / non-finite errors as [`Self::apply`].
    pub fn hess<T: KernelScalar>(
        &self,
        dist: MatRef<'_, T>,
        d2_k: MatMut<'_, T>,
        i: usize,
        j: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        self.hess_math::<Accurate, _>(dist, d2_k, i, j, uplo)
    }

    pub(crate) fn hess_math<M: KernelMath, T: KernelScalar>(
        &self,
        dist: MatRef<'_, T>,
        d2_k: MatMut<'_, T>,
        i: usize,
        j: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        require_rbf_hess_idx(i, j)?;
        let (inv_two_ell_sq, inv_ell_sq) = self.inv_scales_t::<T>();
        write_triangle(dist, d2_k, uplo, |d| {
            rbf_hess_from_sq_dist::<M, _>(d, inv_two_ell_sq, inv_ell_sq)
        })
    }

    pub(crate) fn apply_from_coords<M: KernelMath, T: KernelScalar>(
        &self,
        x: MatRef<'_, T>,
        out: MatMut<'_, T>,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        let (inv_two_ell_sq, _) = self.inv_scales_t::<T>();
        write_square_from_coords(x, out, uplo, |d| {
            rbf_from_sq_dist::<M, _>(d, inv_two_ell_sq)
        })
    }

    /// Rectangular `K(x1, x2)` from coordinates.
    pub(crate) fn apply_cross_from_coords<M: KernelMath, T: KernelScalar>(
        &self,
        x1: MatRef<'_, T>,
        x2: MatRef<'_, T>,
        out: MatMut<'_, T>,
    ) -> Result<(), GprError> {
        let (inv_two_ell_sq, _) = self.inv_scales_t::<T>();
        write_rect_from_coords(x1, x2, out, |d| rbf_from_sq_dist::<M, _>(d, inv_two_ell_sq))
    }

    pub(crate) fn grad_from_coords<M: KernelMath, T: KernelScalar>(
        &self,
        x: MatRef<'_, T>,
        d_k: MatMut<'_, T>,
        param_idx: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        require_rbf_param_idx(param_idx)?;
        let (inv_two_ell_sq, inv_ell_sq) = self.inv_scales_t::<T>();
        write_square_from_coords(x, d_k, uplo, |d| {
            rbf_grad_from_sq_dist::<M, _>(d, inv_two_ell_sq, inv_ell_sq)
        })
    }

    pub(crate) fn hess_from_coords<M: KernelMath, T: KernelScalar>(
        &self,
        x: MatRef<'_, T>,
        d2_k: MatMut<'_, T>,
        i: usize,
        j: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        require_rbf_hess_idx(i, j)?;
        let (inv_two_ell_sq, inv_ell_sq) = self.inv_scales_t::<T>();
        write_square_from_coords(x, d2_k, uplo, |d| {
            rbf_hess_from_sq_dist::<M, _>(d, inv_two_ell_sq, inv_ell_sq)
        })
    }

    /// Writes `∂K(X1, X2)/∂X2[*, dim]` into `d_k`.
    ///
    /// `∂k/∂x2_e = k (x1_e - x2_e) / ℓ²`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::EmptyInput`] or [`GprError::DimensionMismatch`] when
    /// the views are empty or `dim` is out of range, [`GprError::NonFiniteInput`]
    /// when a coordinate is not finite, or [`GprError::ShapeMismatch`]
    /// when `d_k` is the wrong shape.
    pub fn grad_wrt_coord_dim<T: KernelScalar>(
        &self,
        x1: MatRef<'_, T>,
        x2: MatRef<'_, T>,
        d_k: MatMut<'_, T>,
        dim: usize,
    ) -> Result<(), GprError> {
        self.grad_wrt_coord_dim_math::<Accurate, _>(x1, x2, d_k, dim)
    }

    pub(crate) fn grad_wrt_coord_dim_math<M: KernelMath, T: KernelScalar>(
        &self,
        x1: MatRef<'_, T>,
        x2: MatRef<'_, T>,
        d_k: MatMut<'_, T>,
        dim: usize,
    ) -> Result<(), GprError> {
        super::require_coord_grad(x1, x2, d_k.as_ref(), dim)?;
        let (inv_two_ell_sq, inv_ell_sq) = self.inv_scales_t::<T>();
        super::write_rect(d_k, |row, col| {
            let (jet, delta, _) = rbf_pair_with_s::<M, _>(x1, row, x2, col, dim, inv_two_ell_sq)?;
            Ok(jet.d1 * delta * inv_ell_sq)
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
        super::require_coord_grad(x1, x2, d2_k.as_ref(), dim_a)?;
        super::require_coord_grad(x1, x2, d2_k.as_ref(), dim_b)?;
        let (inv_two_ell_sq, inv_ell_sq) = self.inv_scales_t::<T>();
        super::write_rect(d2_k, |row, col| {
            let (jet, _, _) = rbf_pair_with_s::<M, _>(x1, row, x2, col, 0, inv_two_ell_sq)?;
            let da = x1[(row, dim_a)] - x2[(col, dim_a)];
            let db = x1[(row, dim_b)] - x2[(col, dim_b)];
            let mut value = jet.d2 * da * db * inv_ell_sq * inv_ell_sq;
            if dim_a == dim_b {
                value -= jet.d1 * inv_ell_sq;
            }
            Ok(value)
        })
    }

    pub(crate) fn hess_wrt_coord_mixed<M: KernelMath, T: KernelScalar>(
        &self,
        x1: MatRef<'_, T>,
        x2: MatRef<'_, T>,
        d2_k: MatMut<'_, T>,
        dim_x1: usize,
        dim_x2: usize,
    ) -> Result<(), GprError> {
        super::require_coord_grad(x1, x2, d2_k.as_ref(), dim_x1)?;
        super::require_coord_grad(x1, x2, d2_k.as_ref(), dim_x2)?;
        let (inv_two_ell_sq, inv_ell_sq) = self.inv_scales_t::<T>();
        super::write_rect(d2_k, |row, col| {
            let (jet, _, _) = rbf_pair_with_s::<M, _>(x1, row, x2, col, 0, inv_two_ell_sq)?;
            let d1 = x1[(row, dim_x1)] - x2[(col, dim_x1)];
            let d2 = x1[(row, dim_x2)] - x2[(col, dim_x2)];
            let mut value = -jet.d2 * d1 * d2 * inv_ell_sq * inv_ell_sq;
            if dim_x1 == dim_x2 {
                value += jet.d1 * inv_ell_sq;
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
        require_rbf_param_idx(param_idx)?;
        super::require_coord_grad(x1, x2, d2_k.as_ref(), dim)?;
        let (inv_two_ell_sq, inv_ell_sq) = self.inv_scales_t::<T>();
        let two = T::from_f64(2.0);
        super::write_rect(d2_k, |row, col| {
            let (jet, delta, s) = rbf_pair_with_s::<M, _>(x1, row, x2, col, dim, inv_two_ell_sq)?;
            Ok(delta * inv_ell_sq * (jet.d2 * s * inv_ell_sq - two * jet.d1))
        })
    }

    pub(crate) fn grad_cross_from_coords<M: KernelMath, T: KernelScalar>(
        &self,
        x1: MatRef<'_, T>,
        x2: MatRef<'_, T>,
        mut d_k: MatMut<'_, T>,
        param_idx: usize,
    ) -> Result<(), GprError> {
        require_rbf_param_idx(param_idx)?;
        super::require_coord_grad(x1, x2, d_k.as_ref(), 0)?;
        let (inv_two_ell_sq, inv_ell_sq) = self.inv_scales();
        if let (Some(a), Some((b, o))) = (T::as_f64_ref(x1), f64_pair(x2, d_k.rb_mut()))
            && try_grad_rbf_cross::<M>(a, b, o, inv_two_ell_sq, inv_ell_sq)?
        {
            return Ok(());
        }
        let (inv_two_ell_sq, inv_ell_sq) = (T::from_f64(inv_two_ell_sq), T::from_f64(inv_ell_sq));
        super::write_rect(d_k, |row, col| {
            let (jet, _, s) = rbf_pair_with_s::<M, _>(x1, row, x2, col, 0, inv_two_ell_sq)?;
            Ok(jet.d1 * s * inv_ell_sq)
        })
    }

    pub(crate) fn hess_cross_from_coords<M: KernelMath, T: KernelScalar>(
        &self,
        x1: MatRef<'_, T>,
        x2: MatRef<'_, T>,
        d2_k: MatMut<'_, T>,
        i: usize,
        j: usize,
    ) -> Result<(), GprError> {
        require_rbf_hess_idx(i, j)?;
        super::require_coord_grad(x1, x2, d2_k.as_ref(), 0)?;
        let (inv_two_ell_sq, inv_ell_sq) = self.inv_scales_t::<T>();
        super::write_rect(d2_k, |row, col| {
            let (_, _, s) = rbf_pair_with_s::<M, _>(x1, row, x2, col, 0, inv_two_ell_sq)?;
            rbf_hess_from_sq_dist::<M, _>(s, inv_two_ell_sq, inv_ell_sq)
        })
    }

    /// `(1 / (2ℓ²), 1 / ℓ²)` in `f64`.
    fn inv_scales(&self) -> (f64, f64) {
        let inv_ell_sq = 1.0 / (self.lengthscale() * self.lengthscale());
        (0.5 * inv_ell_sq, inv_ell_sq)
    }

    /// [`Self::inv_scales`] rounded into the compute scalar.
    fn inv_scales_t<T: KernelScalar>(&self) -> (T, T) {
        let (half, full) = self.inv_scales();
        (T::from_f64(half), T::from_f64(full))
    }
}

/// `∂k/∂θ = k s / ℓ²` on a rectangular pair. Stays off the square Gram helpers.
fn try_grad_rbf_cross<M: KernelMath>(
    x1: MatRef<'_, f64>,
    x2: MatRef<'_, f64>,
    mut d_k: MatMut<'_, f64>,
    inv_two_ell_sq: f64,
    inv_ell_sq: f64,
) -> Result<bool, GprError> {
    let m = x1.nrows();
    let n = x2.nrows();
    let d = x1.ncols();
    if d == 0 || x2.ncols() != d || d_k.nrows() != m || d_k.ncols() != n {
        return Ok(false);
    }
    if !unit_cols(x1) || !unit_cols(x2) || !unit_cols(d_k.as_ref()) {
        return Ok(false);
    }
    for dim in 0..d {
        finite_slice(col_slice(x1, dim)?)?;
        finite_slice(col_slice(x2, dim)?)?;
    }
    let mut s = vec![0.0; n];
    let mut dk = vec![0.0; n];
    let neg = f64x4::new([-inv_two_ell_sq; 4]);
    let scale = f64x4::new([inv_ell_sq; 4]);
    for row in 0..m {
        s.fill(0.0);
        for dim in 0..d {
            let z = col_slice(x1, dim)?[row];
            add_squared(col_slice(x2, dim)?, z, &mut s);
        }
        let mut i = 0;
        while i + 4 <= n {
            let sv = load4(&s, i);
            let value = M::d1_f64x4(sv * neg) * sv * scale;
            if !all_finite4(value) {
                return Err(GprError::NonFiniteKernelValue);
            }
            store4(&mut dk, i, value);
            i += 4;
        }
        while i < n {
            let d1 = M::jet(-s[i] * inv_two_ell_sq).d1;
            let value = d1 * s[i] * inv_ell_sq;
            if !value.is_finite() {
                return Err(GprError::NonFiniteKernelValue);
            }
            dk[i] = value;
            i += 1;
        }
        for (col, value) in dk.iter().enumerate() {
            d_k[(row, col)] = *value;
        }
    }
    Ok(true)
}

fn unit_cols(mat: MatRef<'_, f64>) -> bool {
    mat.ncols() == 0 || mat.col(0).try_as_col_major().is_some()
}

fn col_slice<'a>(mat: MatRef<'a, f64>, col: usize) -> Result<&'a [f64], GprError> {
    mat.col(col)
        .try_as_col_major()
        .map(|c| c.as_slice())
        .ok_or_else(|| GprError::UnsupportedKernelOperation {
            reason: "expected unit row-stride for RBF cross grad".to_owned(),
        })
}

fn finite_slice(values: &[f64]) -> Result<(), GprError> {
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

fn add_squared(x: &[f64], x0: f64, acc: &mut [f64]) {
    let x0v = f64x4::new([x0; 4]);
    let mut i = 0;
    while i + 4 <= x.len() {
        let d = load4(x, i) - x0v;
        let av = load4(acc, i);
        store4(acc, i, av + d * d);
        i += 4;
    }
    while i < x.len() {
        let d = x[i] - x0;
        acc[i] += d * d;
        i += 1;
    }
}

fn rbf_pair_with_s<M: KernelMath, T: KernelScalar>(
    x1: MatRef<'_, T>,
    i: usize,
    x2: MatRef<'_, T>,
    j: usize,
    dim: usize,
    inv_two_ell_sq: T,
) -> Result<(ExpJet<T>, T, T), GprError> {
    let mut s = T::from_f64(0.0);
    for d in 0..x1.ncols() {
        let a = x1[(i, d)];
        let b = x2[(j, d)];
        if !a.is_finite() || !b.is_finite() {
            return Err(GprError::NonFiniteInput);
        }
        let delta = a - b;
        s += delta * delta;
    }
    let jet = M::jet(-s * inv_two_ell_sq);
    if !jet.v.is_finite() {
        return Err(GprError::NonFiniteKernelValue);
    }
    Ok((jet, x1[(i, dim)] - x2[(j, dim)], s))
}

fn rbf_from_sq_dist<M: KernelMath, T: KernelScalar>(
    d: T,
    inv_two_ell_sq: T,
) -> Result<T, GprError> {
    let d = finite_dist(d)?;
    Ok(M::exp(-d * inv_two_ell_sq))
}

fn rbf_grad_from_sq_dist<M: KernelMath, T: KernelScalar>(
    d: T,
    inv_two_ell_sq: T,
    inv_ell_sq: T,
) -> Result<T, GprError> {
    let d = finite_dist(d)?;
    let dk = M::jet(-d * inv_two_ell_sq).d1;
    Ok(dk * d * inv_ell_sq)
}

fn rbf_hess_from_sq_dist<M: KernelMath, T: KernelScalar>(
    d: T,
    inv_two_ell_sq: T,
    inv_ell_sq: T,
) -> Result<T, GprError> {
    let d = finite_dist(d)?;
    let jet = M::jet(-d * inv_two_ell_sq);
    let u = d * inv_ell_sq;
    let h = u * (jet.d2 * u - T::from_f64(2.0) * jet.d1);
    if h.is_finite() {
        Ok(h)
    } else {
        Err(GprError::NonFiniteKernelValue)
    }
}

fn require_rbf_param_idx(param_idx: usize) -> Result<(), GprError> {
    if param_idx == 0 {
        Ok(())
    } else {
        Err(GprError::IndexOutOfRange {
            reason: "RBF has a single parameter at index 0".to_owned(),
        })
    }
}

fn require_rbf_hess_idx(i: usize, j: usize) -> Result<(), GprError> {
    if i == 0 && j == 0 {
        Ok(())
    } else {
        Err(GprError::IndexOutOfRange {
            reason: format!("RBF has a single parameter; got pair ({i}, {j})"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::RbfKernel;
    use crate::error::GprError;
    use crate::kernel::Triangle;
    use faer::mat;

    const TOL: f64 = 1e-10;

    use crate::test_check::{assert_close, assert_lower_close, assert_send_sync, fill, sq_dist_1d};

    #[test]
    fn is_send_sync() {
        assert_send_sync::<RbfKernel>();
        assert_send_sync::<Triangle>();
    }

    #[test]
    fn diagonal_is_one() {
        let rbf = RbfKernel::new(2.0).expect("valid");
        let dist = sq_dist_1d(&[0.0, 1.0, 3.0]);
        let mut k = fill(3, f64::NAN);
        rbf.apply(dist.as_ref(), k.as_mut(), Triangle::Full)
            .expect("shape");
        assert_close(k[(0, 0)], 1.0, TOL);
        assert_close(k[(1, 1)], 1.0, TOL);
        assert_close(k[(2, 2)], 1.0, TOL);
    }

    #[test]
    fn known_values_at_one_and_two_lengthscales() {
        let ell = 2.0;
        let rbf = RbfKernel::new(ell).expect("valid");
        // ‖x-x'‖² = ℓ² ⇒ k = exp(-1/2);  ‖x-x'‖² = 2ℓ² ⇒ k = exp(-1)
        let dist = mat![
            [0.0, ell * ell, 2.0 * ell * ell],
            [ell * ell, 0.0, 0.0],
            [2.0 * ell * ell, 0.0, 0.0]
        ];
        let mut k = fill(3, 0.0);
        rbf.apply(dist.as_ref(), k.as_mut(), Triangle::Full)
            .expect("shape");
        assert_close(k[(0, 1)], (-0.5_f64).exp(), TOL);
        assert_close(k[(0, 2)], (-1.0_f64).exp(), TOL);
    }

    #[test]
    fn full_is_symmetric() {
        let rbf = RbfKernel::new(1.25).expect("valid");
        let dist = sq_dist_1d(&[0.0, 0.5, 2.0, 2.5]);
        let mut k = fill(4, 0.0);
        rbf.apply(dist.as_ref(), k.as_mut(), Triangle::Full)
            .expect("shape");
        for col in 0..4 {
            for row in 0..4 {
                assert_close(k[(row, col)], k[(col, row)], TOL);
            }
        }
    }

    #[test]
    fn lower_matches_full_and_leaves_upper() {
        let rbf = RbfKernel::new(0.75).expect("valid");
        let dist = sq_dist_1d(&[0.0, 1.0, 2.0]);
        let mut full = fill(3, 0.0);
        rbf.apply(dist.as_ref(), full.as_mut(), Triangle::Full)
            .expect("shape");
        let sentinel = 42.0;
        let mut lower = fill(3, sentinel);
        rbf.apply(dist.as_ref(), lower.as_mut(), Triangle::Lower)
            .expect("shape");
        assert_lower_close(lower.as_ref(), full.as_ref(), TOL);
        assert_close(lower[(0, 1)], sentinel, TOL);
        assert_close(lower[(0, 2)], sentinel, TOL);
        assert_close(lower[(1, 2)], sentinel, TOL);
    }

    #[test]
    fn grad_matches_finite_difference() {
        let rbf = RbfKernel::from_log_lengthscale(-0.3).expect("valid");
        let theta = rbf.log_lengthscale();
        let h = 1e-6;
        let plus = RbfKernel::from_log_lengthscale(theta + h).expect("valid");
        let minus = RbfKernel::from_log_lengthscale(theta - h).expect("valid");
        let dist = sq_dist_1d(&[0.0, 1.2, 2.4]);
        let mut k_plus = fill(3, 0.0);
        let mut k_minus = fill(3, 0.0);
        let mut dk = fill(3, 0.0);
        plus.apply(dist.as_ref(), k_plus.as_mut(), Triangle::Full)
            .expect("shape");
        minus
            .apply(dist.as_ref(), k_minus.as_mut(), Triangle::Full)
            .expect("shape");
        rbf.grad(dist.as_ref(), dk.as_mut(), 0, Triangle::Full)
            .expect("index 0");
        for col in 0..3 {
            for row in 0..3 {
                let fd = (k_plus[(row, col)] - k_minus[(row, col)]) / (2.0 * h);
                assert_close(dk[(row, col)], fd, TOL);
            }
        }
    }

    #[test]
    fn hess_matches_finite_difference_of_grad() {
        let rbf = RbfKernel::from_log_lengthscale(-0.3).expect("valid");
        let theta = rbf.log_lengthscale();
        let h = 1e-6;
        let plus = RbfKernel::from_log_lengthscale(theta + h).expect("valid");
        let minus = RbfKernel::from_log_lengthscale(theta - h).expect("valid");
        let dist = sq_dist_1d(&[0.0, 1.2, 2.4]);
        let mut g_plus = fill(3, 0.0);
        let mut g_minus = fill(3, 0.0);
        let mut d2 = fill(3, 0.0);
        plus.grad(dist.as_ref(), g_plus.as_mut(), 0, Triangle::Full)
            .expect("shape");
        minus
            .grad(dist.as_ref(), g_minus.as_mut(), 0, Triangle::Full)
            .expect("shape");
        rbf.hess(dist.as_ref(), d2.as_mut(), 0, 0, Triangle::Full)
            .expect("index 0");
        for col in 0..3 {
            for row in 0..3 {
                let fd = (g_plus[(row, col)] - g_minus[(row, col)]) / (2.0 * h);
                assert_close(d2[(row, col)], fd, TOL);
            }
        }
    }

    #[test]
    fn grad_lower_matches_full() {
        let rbf = RbfKernel::new(1.0).expect("valid");
        let dist = sq_dist_1d(&[0.0, 0.8, 1.6]);
        let mut full = fill(3, 0.0);
        let mut lower = fill(3, 99.0);
        rbf.grad(dist.as_ref(), full.as_mut(), 0, Triangle::Full)
            .expect("index 0");
        rbf.grad(dist.as_ref(), lower.as_mut(), 0, Triangle::Lower)
            .expect("index 0");
        assert_lower_close(lower.as_ref(), full.as_ref(), TOL);
        assert_close(lower[(0, 1)], 99.0, TOL);
    }

    #[test]
    fn upper_matches_full() {
        let rbf = RbfKernel::new(1.0).expect("valid");
        let dist = sq_dist_1d(&[0.0, 1.0, 2.0]);
        let mut full = fill(3, 0.0);
        let mut upper = fill(3, -1.0);
        rbf.apply(dist.as_ref(), full.as_mut(), Triangle::Full)
            .expect("shape");
        rbf.apply(dist.as_ref(), upper.as_mut(), Triangle::Upper)
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
    fn simd_lower_matches_scalar_on_n_eight() {
        let rbf = RbfKernel::new(1.25).expect("valid");
        let x: Vec<f64> = (0..8).map(|i| i as f64 * 0.37).collect();
        let dist = sq_dist_1d(&x);
        let mut simd = fill(8, 0.0);
        rbf.apply(dist.as_ref(), simd.as_mut(), Triangle::Lower)
            .expect("shape");
        for col in 0..8 {
            for row in col..8 {
                let expected = (-0.5 * dist[(row, col)] / (1.25 * 1.25)).exp();
                assert_close(simd[(row, col)], expected, TOL);
            }
        }
    }

    #[test]
    fn get_set_params_roundtrip() {
        let mut rbf = RbfKernel::new(2.0).expect("valid");
        let mut params = [0.0];
        rbf.get_params(&mut params).expect("len 1");
        assert_close(params[0], 2.0_f64.ln(), TOL);
        params[0] = 0.5_f64.ln();
        rbf.set_params(&params).expect("len 1");
        assert_close(rbf.lengthscale(), 0.5, TOL);
    }

    #[test]
    fn rejects_non_positive_lengthscale_and_bad_index() {
        assert!(matches!(
            RbfKernel::new(0.0),
            Err(GprError::InvalidHyperparameter { .. })
        ));
        assert!(matches!(
            RbfKernel::from_log_lengthscale(f64::INFINITY),
            Err(GprError::InvalidHyperparameter { .. })
        ));
        let rbf = RbfKernel::new(1.0).expect("valid");
        let dist = sq_dist_1d(&[0.0, 1.0]);
        let mut dk = fill(2, 0.0);
        assert!(matches!(
            rbf.grad(dist.as_ref(), dk.as_mut(), 1, Triangle::Lower),
            Err(GprError::IndexOutOfRange { .. })
        ));
    }

    #[test]
    fn apply_rejects_non_finite_dist() {
        let rbf = RbfKernel::new(1.0).expect("valid");
        let dist = mat![[0.0, f64::NAN], [f64::NAN, 0.0]];
        let mut k = fill(2, 0.0);
        assert!(matches!(
            rbf.apply(dist.as_ref(), k.as_mut(), Triangle::Full),
            Err(GprError::NonFiniteInput)
        ));
    }
}
