//! Isotropic squared-exponential (RBF) kernel.

use super::dist::{lower_col, par_lower_fold};
use super::lengthscale::{validate_lengthscale, validate_log_lengthscale};
use super::simd::stationary::{
    RbfScales, try_apply_rbf_cross, try_grad_rbf_cross_from_coords, try_square_rbf,
};
use super::{
    KernelScalar, Triangle, finite_dist, write_dense, write_rect_from_coords,
    write_square_from_coords, write_triangle,
};
use crate::error::GprError;
use crate::math::{Accurate, ExpJet, KernelMath};
use crate::param::{BoundedParam, Interval};
use faer::reborrow::ReborrowMut;
use faer::{MatMut, MatRef};

/// Evaluates the isotropic RBF `k = exp( -‖x-x'‖² / (2ℓ²) )`.
///
/// The optimizer parameter is `θ = log(ℓ)`. Amplitude is not stored here; compose with
/// [`super::ConstantKernel`] when a signal variance is needed. `dist` is the matrix of
/// squared Euclidean distances. Column-major views with unit row stride use four-wide
/// `f64` SIMD for [`Self::apply`], [`Self::apply_cross`], and [`Self::grad`].
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
    ///
    /// See the example on [`RbfKernel`].
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
    ///
    /// See the example on [`RbfKernel`].
    pub fn from_log_lengthscale(log_lengthscale: f64) -> Result<Self, GprError> {
        let log_lengthscale = validate_log_lengthscale(log_lengthscale)?;
        Ok(Self {
            lengthscale: BoundedParam::default_positive(log_lengthscale.exp())?,
        })
    }

    /// Returns `ℓ = exp(θ)`.
    ///
    /// See the example on [`RbfKernel`].
    pub fn lengthscale(&self) -> f64 {
        self.lengthscale.value()
    }

    /// Returns `θ = log(ℓ)`.
    ///
    /// See the example on [`RbfKernel`].
    pub fn log_lengthscale(&self) -> f64 {
        self.lengthscale.ln()
    }

    /// Returns the open interval on `ℓ`.
    ///
    /// See the example on [`RbfKernel`].
    pub fn bounds(&self) -> Interval {
        self.lengthscale.interval()
    }

    /// Rebuilds this kernel with a new interval on `ℓ`.
    ///
    /// # Errors
    ///
    /// Returns [`crate::IntervalError`] if the current `ℓ` is not strictly
    /// inside `interval`.
    ///
    /// See the example on [`RbfKernel`].
    pub fn with_bounds(self, interval: Interval) -> Result<Self, crate::IntervalError> {
        Ok(Self {
            lengthscale: self.lengthscale.with_interval(interval)?,
        })
    }

    /// Returns the number of optimizer parameters (always 1).
    ///
    /// See the example on [`RbfKernel`].
    pub fn num_params(&self) -> usize {
        1
    }

    /// Writes `θ` into a length-1 slice.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::LengthMismatch`] if `out` is not length 1.
    ///
    /// See the example on [`RbfKernel`].
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
    ///
    /// See the example on [`RbfKernel`].
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
    ///
    /// See the example on [`RbfKernel`].
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
        if try_square_rbf::<M, T>(dist, out.rb_mut(), uplo, self.lane_scales(), false)? {
            return Ok(());
        }
        let inv_two_ell_sq = T::from_f64(self.lane_scales().half_inv_ell_sq);
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
    ///
    /// See the example on [`RbfKernel`].
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
        if try_apply_rbf_cross::<M, T>(dist, out.rb_mut(), self.lane_scales())? {
            return Ok(());
        }
        let inv_two_ell_sq = T::from_f64(self.lane_scales().half_inv_ell_sq);
        write_dense(dist, out, |d| rbf_from_sq_dist::<M, _>(d, inv_two_ell_sq))
    }

    /// Writes the stationary diagonal `k(x, x) = 1` into `out`.
    ///
    /// See the example on [`RbfKernel`].
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
    ///
    /// See the example on [`RbfKernel`].
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
        if try_square_rbf::<M, T>(dist, d_k.rb_mut(), uplo, self.lane_scales(), true)? {
            return Ok(());
        }
        let (inv_two_ell_sq, inv_ell_sq) = self.inv_scales_t::<T>();
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
    ///
    /// See the example on [`RbfKernel`].
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

    /// Rectangular `∂K/∂θ` from squared distances.
    pub(crate) fn grad_cross_dist<M: KernelMath, T: KernelScalar>(
        &self,
        dist: MatRef<'_, T>,
        d_k: MatMut<'_, T>,
        param_idx: usize,
    ) -> Result<(), GprError> {
        require_rbf_param_idx(param_idx)?;
        let (inv_two_ell_sq, inv_ell_sq) = self.inv_scales_t::<T>();
        write_dense(dist, d_k, |d| {
            rbf_grad_from_sq_dist::<M, _>(d, inv_two_ell_sq, inv_ell_sq)
        })
    }

    /// Rectangular `∂²K/∂θ_i ∂θ_j` from squared distances.
    pub(crate) fn hess_cross_dist<M: KernelMath, T: KernelScalar>(
        &self,
        dist: MatRef<'_, T>,
        d2_k: MatMut<'_, T>,
        i: usize,
        j: usize,
    ) -> Result<(), GprError> {
        require_rbf_hess_idx(i, j)?;
        let (inv_two_ell_sq, inv_ell_sq) = self.inv_scales_t::<T>();
        write_dense(dist, d2_k, |d| {
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
    ///
    /// See the example on [`RbfKernel`].
    pub fn grad_wrt_coord_dim<T: KernelScalar>(
        &self,
        x1: MatRef<'_, T>,
        x2: MatRef<'_, T>,
        d_k: MatMut<'_, T>,
        dim: usize,
    ) -> Result<(), GprError> {
        super::radial::grad_wrt_coord_dim::<Accurate, _>(self, x1, x2, d_k, dim)
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
        if try_grad_rbf_cross_from_coords::<M, T>(x1, x2, d_k.rb_mut(), self.lane_scales())? {
            return Ok(());
        }
        let (inv_two_ell_sq, inv_ell_sq) = self.inv_scales_t::<T>();
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

    /// `⟨weight, ∂K/∂log ℓ⟩_F` into `out[0]` and `⟨weight, K⟩_F` returned,
    /// from this leaf's Gram `k` at the same `θ` over the lower triangle:
    /// `∂k/∂log ℓ = k s / ℓ²`, so no `exp` is evaluated. Only the `exp`
    /// algebra of [`crate::Accurate`] has `∂k = k ·`; callers check that.
    pub(crate) fn weighted_grads_from_gram<T: KernelScalar>(
        &self,
        dist: MatRef<'_, T>,
        k: MatRef<'_, T>,
        weight: MatRef<'_, T>,
        out: &mut [f64],
    ) -> Result<f64, GprError> {
        let (_, inv_ell_sq) = self.inv_scales();
        let n = dist.nrows();
        let (g_ell, value) = par_lower_fold(
            n,
            &|start, end| {
                let (mut g, mut v) = (0.0, 0.0);
                for col in start..end {
                    let w = weight[(col, col)].to_f64();
                    let wk = w * k[(col, col)].to_f64();
                    v += wk;
                    g += wk * finite_dist(dist[(col, col)])?.to_f64();
                    let (mut g_col, mut v_col) = (0.0, 0.0);
                    if let (Some(d), Some(w), Some(kc)) = (
                        lower_col(dist, col),
                        lower_col(weight, col),
                        lower_col(k, col),
                    ) {
                        for ((&s, &w), &kv) in d[1..].iter().zip(&w[1..]).zip(&kc[1..]) {
                            let wk = w.to_f64() * kv.to_f64();
                            v_col += wk;
                            g_col += wk * s.to_f64();
                        }
                        if !g_col.is_finite() {
                            for &s in &d[1..] {
                                finite_dist(s)?;
                            }
                        }
                    } else {
                        for row in col + 1..n {
                            let s = finite_dist(dist[(row, col)])?.to_f64();
                            let wk = weight[(row, col)].to_f64() * k[(row, col)].to_f64();
                            v_col += wk;
                            g_col += wk * s;
                        }
                    }
                    v += 2.0 * v_col;
                    g += 2.0 * g_col;
                }
                Ok::<_, GprError>((g, v))
            },
            &|a: (f64, f64), b: (f64, f64)| (a.0 + b.0, a.1 + b.1),
        )?;
        out[0] = g_ell * inv_ell_sq;
        Ok(value)
    }

    /// [`Self::inv_scales`] for the lanes of [`super::stationary_simd`].
    fn lane_scales(&self) -> RbfScales {
        let (half_inv_ell_sq, inv_ell_sq) = self.inv_scales();
        RbfScales {
            half_inv_ell_sq,
            inv_ell_sq,
        }
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

    /// `f32` coordinates take the same widened `f64` lanes as `f64`.
    #[test]
    fn f32_cross_grad_from_coords_rounds_the_f64_lanes() {
        use crate::math::{Accurate, FastApprox};
        let x1 = faer::Mat::<f64>::from_fn(5, 2, |i, d| 0.3 * i as f64 - 0.7 * d as f64);
        let x2 = faer::Mat::<f64>::from_fn(7, 2, |i, d| 0.45 * i as f64 + 0.2 * d as f64 - 1.0);
        let x1_32 = faer::Mat::<f32>::from_fn(5, 2, |i, d| x1[(i, d)] as f32);
        let x2_32 = faer::Mat::<f32>::from_fn(7, 2, |i, d| x2[(i, d)] as f32);
        let rbf = RbfKernel::new(1.3).expect("valid");
        let mut want = faer::Mat::<f64>::zeros(5, 7);
        let mut got = faer::Mat::<f32>::zeros(5, 7);
        rbf.grad_cross_from_coords::<Accurate, f64>(x1.as_ref(), x2.as_ref(), want.as_mut(), 0)
            .expect("finite");
        rbf.grad_cross_from_coords::<Accurate, f32>(
            x1_32.as_ref(),
            x2_32.as_ref(),
            got.as_mut(),
            0,
        )
        .expect("finite");
        for j in 0..7 {
            for i in 0..5 {
                assert_close(f64::from(got[(i, j)]), want[(i, j)], 1e-6);
            }
        }
        rbf.grad_cross_from_coords::<FastApprox, f32>(
            x1_32.as_ref(),
            x2_32.as_ref(),
            got.as_mut(),
            0,
        )
        .expect("finite");
        for j in 0..7 {
            for i in 0..5 {
                assert_close(f64::from(got[(i, j)]), want[(i, j)], 1e-5);
            }
        }
    }

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
