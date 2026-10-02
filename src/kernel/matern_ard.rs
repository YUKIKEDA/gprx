//! ARD Matérn kernel for `ν = 1/2`, `3/2`, and `5/2`.

use super::ard::{self, ArdR2, Pick};
use super::dist::ArdSqDiff;
use super::finite_kernel;
use super::matern::{MaternNu, matern_d2k_dtheta_ard, matern_dk_dtheta_ard, matern_from_r};
use super::simd::ard::Profile;
use super::{ArdLengthscales, KernelScalar, Triangle, write_rect};
use crate::error::GprError;
use crate::math::KernelMath;
use faer::{MatMut, MatRef};
use std::marker::PhantomData;
use wide::{CmpGt, f64x4};

/// ARD Matérn: `k` is a function of `r = √(Σ_d (x_d-x'_d)² / ℓ_d²)`.
///
/// Optimizer parameters are `θ_d = log(ℓ_d)` via [`ArdLengthscales`]. `ν` is
/// not an optimizer parameter. When every `ℓ_d` equals a scalar `ℓ`, values
/// match isotropic [`super::MaternKernel`]. `apply` / `grad` take the `n×d`
/// coordinate matrix. Amplitude is not stored here.
///
/// Cloning copies the lengthscale vectors. When
/// [`crate::DistanceCachePolicy::Cached`] is set, [`crate::Gpr`] caches raw
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
    /// Returns [`GprError::LengthMismatch`] if `out` is the wrong length.
    pub fn get_params(&self, out: &mut [f64]) -> Result<(), GprError> {
        self.lengthscales.get_params(out)
    }

    /// Replaces `θ_d` from `params`. The previous values and `ν` are kept on
    /// error.
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
        out: MatMut<'_, T>,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        let w = self.lengthscales.inv_ell_sq();
        let nu = self.nu;
        let profile = MaternValue4::<M>::new(nu);
        ard::write_from_points_simd(
            x,
            out,
            self.num_params(),
            uplo,
            w,
            None,
            &profile,
            |row, col| {
                let t = ard::r2_from_coords(x, row, x, col, w, Pick::NONE)?;
                matern_value::<M, T>(nu, t)
            },
        )
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
        out: MatMut<'_, T>,
    ) -> Result<(), GprError> {
        let w = self.lengthscales.inv_ell_sq();
        let nu = self.nu;
        let profile = MaternValue4::<M>::new(nu);
        ard::write_cross_simd(
            x,
            xs,
            out,
            self.num_params(),
            w,
            None,
            &profile,
            |row, col| {
                let t = ard::r2_from_coords(x, row, xs, col, w, Pick::NONE)?;
                matern_value::<M, T>(nu, t)
            },
        )
    }

    /// Writes the stationary diagonal `k(x, x) = 1` into `out`.
    pub fn fill_diag<T: KernelScalar>(&self, out: &mut [T]) {
        out.fill(T::from_f64(1.0));
    }

    /// Writes `∂K/∂θ_d` for `θ_d = log(ℓ_d)` into `d_k`.
    ///
    /// This is not `∂k/∂ℓ_d`.
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
        d_k: MatMut<'_, T>,
        param_idx: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        ard::require_param(NAME, param_idx, self.num_params())?;
        let w = self.lengthscales.inv_ell_sq();
        let nu = self.nu;
        let profile = MaternGrad4::<M>::new(nu);
        let pick = Some(param_idx);
        ard::write_from_points_simd(
            x,
            d_k,
            self.num_params(),
            uplo,
            w,
            pick,
            &profile,
            |row, col| {
                let t = ard::r2_from_coords(x, row, x, col, w, Pick::one(param_idx))?;
                matern_grad::<M, T>(nu, t)
            },
        )
    }

    pub(crate) fn apply_from_sq_diff<M: KernelMath, T: KernelScalar>(
        &self,
        cache: ArdSqDiff<'_, T>,
        out: MatMut<'_, T>,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        let w = self.lengthscales.inv_ell_sq();
        let nu = self.nu;
        let profile = MaternValue4::<M>::new(nu);
        ard::write_from_cache_simd(
            cache,
            out,
            self.num_params(),
            uplo,
            w,
            None,
            &profile,
            |_, row, col| {
                let t = ard::r2_from_cache(cache, row, col, w, Pick::NONE)?;
                matern_value::<M, T>(nu, t)
            },
        )
    }

    pub(crate) fn grad_from_sq_diff<M: KernelMath, T: KernelScalar>(
        &self,
        cache: ArdSqDiff<'_, T>,
        d_k: MatMut<'_, T>,
        param_idx: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        ard::require_param(NAME, param_idx, self.num_params())?;
        let w = self.lengthscales.inv_ell_sq();
        let nu = self.nu;
        let profile = MaternGrad4::<M>::new(nu);
        let pick = Some(param_idx);
        ard::write_from_cache_simd(
            cache,
            d_k,
            self.num_params(),
            uplo,
            w,
            pick,
            &profile,
            |_, row, col| {
                let t = ard::r2_from_cache(cache, row, col, w, Pick::one(param_idx))?;
                matern_grad::<M, T>(nu, t)
            },
        )
    }

    /// Writes `∂²K/∂θ_i ∂θ_j` for ARD `θ_d = log(ℓ_d)`.
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
        let nu = self.nu;
        ard::write_from_points(x, d2_k, self.num_params(), uplo, |row, col| {
            let t = ard::r2_from_coords(x, row, x, col, w, Pick::pair(i, j))?;
            matern_hess::<M, T>(nu, t, i == j)
        })
    }

    pub(crate) fn hess_from_sq_diff<M: KernelMath, T: KernelScalar>(
        &self,
        cache: ArdSqDiff<'_, T>,
        d2_k: MatMut<'_, T>,
        i: usize,
        j: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        ard::require_param_pair(NAME, i, j, self.num_params())?;
        let w = self.lengthscales.inv_ell_sq();
        let nu = self.nu;
        ard::write_from_cache(cache, d2_k, self.num_params(), uplo, |row, col| {
            let t = ard::r2_from_cache(cache, row, col, w, Pick::pair(i, j))?;
            matern_hess::<M, T>(nu, t, i == j)
        })
    }

    /// `∂K(x1, x2)/∂θ` of a rectangular block, from coordinates.
    pub(crate) fn grad_cross_from_coords<M: KernelMath, T: KernelScalar>(
        &self,
        x1: MatRef<'_, T>,
        x2: MatRef<'_, T>,
        d_k: MatMut<'_, T>,
        param_idx: usize,
    ) -> Result<(), GprError> {
        ard::require_param(NAME, param_idx, self.num_params())?;
        let w = self.lengthscales.inv_ell_sq();
        let nu = self.nu;
        let profile = MaternGrad4::<M>::new(nu);
        let pick = Some(param_idx);
        ard::write_cross_simd(
            x1,
            x2,
            d_k,
            self.num_params(),
            w,
            pick,
            &profile,
            |row, col| {
                let t = ard::r2_from_coords(x1, row, x2, col, w, Pick::one(param_idx))?;
                matern_grad::<M, T>(nu, t)
            },
        )
    }

    /// `∂²K(x1, x2)/∂θ_i ∂θ_j` of a rectangular block, from coordinates.
    pub(crate) fn hess_cross_from_coords<M: KernelMath, T: KernelScalar>(
        &self,
        x1: MatRef<'_, T>,
        x2: MatRef<'_, T>,
        d2_k: MatMut<'_, T>,
        i: usize,
        j: usize,
    ) -> Result<(), GprError> {
        ard::require_param_pair(NAME, i, j, self.num_params())?;
        ard::require_cross(x1, x2, d2_k.as_ref(), self.num_params())?;
        let w = self.lengthscales.inv_ell_sq();
        let nu = self.nu;
        write_rect(d2_k, |row, col| {
            let t = ard::r2_from_coords(x1, row, x2, col, w, Pick::pair(i, j))?;
            matern_hess::<M, T>(nu, t, i == j)
        })
    }
}

const NAME: &str = "Matern";

/// `k` of four pairs from `r²` ([`ard_simd`](super::ard_simd)).
struct MaternValue4<M> {
    nu: MaternNu,
    _math: PhantomData<M>,
}

impl<M> MaternValue4<M> {
    fn new(nu: MaternNu) -> Self {
        Self {
            nu,
            _math: PhantomData,
        }
    }
}

impl<M: KernelMath> Profile for MaternValue4<M> {
    #[inline(always)]
    fn eval(&self, r2: f64x4, _t: f64x4) -> f64x4 {
        let r = r2.max(f64x4::ZERO).sqrt();
        let one = f64x4::ONE;
        match self.nu {
            MaternNu::Half => M::exp_f64x4(-r),
            MaternNu::ThreeHalves => {
                let rho = r * f64x4::splat(3.0_f64.sqrt());
                (one + rho) * M::exp_f64x4(-rho)
            }
            MaternNu::FiveHalves => {
                let rho = r * f64x4::splat(5.0_f64.sqrt());
                (one + rho + rho * rho / f64x4::splat(3.0)) * M::exp_f64x4(-rho)
            }
        }
    }
}

/// `∂k/∂θ_d` of four pairs from `r²` and `t = w_d Δ_d²`, as
/// [`matern_dk_dtheta_ard`] for each mode.
struct MaternGrad4<M> {
    nu: MaternNu,
    _math: PhantomData<M>,
}

impl<M> MaternGrad4<M> {
    fn new(nu: MaternNu) -> Self {
        Self {
            nu,
            _math: PhantomData,
        }
    }
}

impl<M: KernelMath> Profile for MaternGrad4<M> {
    #[inline(always)]
    fn eval(&self, r2: f64x4, t: f64x4) -> f64x4 {
        let r = r2.max(f64x4::ZERO).sqrt();
        let one = f64x4::ONE;
        let positive = r.cmp_gt(f64x4::ZERO);
        // `r = 0` makes every derivative zero; divide by 1 there instead.
        let safe_r = positive.blend(r, one);
        let value = if M::ACCURATE {
            match self.nu {
                MaternNu::Half => (-r).exp() * t / safe_r,
                MaternNu::ThreeHalves => {
                    f64x4::splat(3.0) * t * (-(r * f64x4::splat(3.0_f64.sqrt()))).exp()
                }
                MaternNu::FiveHalves => {
                    let rho = r * f64x4::splat(5.0_f64.sqrt());
                    f64x4::splat(5.0 / 3.0) * (one + rho) * (-rho).exp() * t
                }
            }
        } else {
            match self.nu {
                MaternNu::Half => M::d1_f64x4(-r) * t / safe_r,
                MaternNu::ThreeHalves => {
                    let s3 = f64x4::splat(3.0_f64.sqrt());
                    let rho = r * s3;
                    ((one + rho) * M::d1_f64x4(-rho) - M::exp_f64x4(-rho)) * s3 * t / safe_r
                }
                MaternNu::FiveHalves => {
                    let s5 = f64x4::splat(5.0_f64.sqrt());
                    let rho = r * s5;
                    let a = one + rho + rho * rho / f64x4::splat(3.0);
                    let ap = one + f64x4::splat(2.0) * rho / f64x4::splat(3.0);
                    (a * M::d1_f64x4(-rho) - ap * M::exp_f64x4(-rho)) * s5 * t / safe_r
                }
            }
        };
        positive.blend(value, f64x4::ZERO)
    }
}

fn matern_value<M: KernelMath, T: KernelScalar>(nu: MaternNu, t: ArdR2<T>) -> Result<T, GprError> {
    let r = t.r2.max(T::from_f64(0.0)).sqrt();
    finite_kernel(matern_from_r::<M, T>(nu, r))
}

fn matern_grad<M: KernelMath, T: KernelScalar>(nu: MaternNu, t: ArdR2<T>) -> Result<T, GprError> {
    let r = t.r2.max(T::from_f64(0.0)).sqrt();
    finite_kernel(matern_dk_dtheta_ard::<M, T>(nu, r, t.dim_i))
}

fn matern_hess<M: KernelMath, T: KernelScalar>(
    nu: MaternNu,
    t: ArdR2<T>,
    same: bool,
) -> Result<T, GprError> {
    let r = t.r2.max(T::from_f64(0.0)).sqrt();
    finite_kernel(matern_d2k_dtheta_ard::<M, T>(nu, r, t.dim_i, t.dim_j, same))
}

#[cfg(test)]
mod tests {
    use super::MaternArdKernel;
    use crate::error::GprError;
    use crate::kernel::{MaternKernel, MaternNu, Triangle};
    use faer::{Mat, MatRef};

    const TOL: f64 = 1e-8;

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
        assert_close(k[(0, 0)], 1.0, TOL);
        assert_close(k[(1, 1)], 1.0, TOL);
        assert_close(k[(2, 2)], 1.0, TOL);
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
                    assert_close(k_ard[(row, col)], k_iso[(row, col)], TOL);
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
        assert_close(k[(1, 0)], (-r).exp(), TOL);
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
                assert_close(k[(row, col)], k[(col, row)], TOL);
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
        assert_lower_close(lower.as_ref(), full.as_ref(), TOL);
        assert_close(lower[(0, 1)], 42.0, TOL);
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
                        assert_close(dk[(row, col)], fd, TOL);
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
        assert_close(dk1[(1, 0)], 0.0, TOL);
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
        assert_lower_close(lower.as_ref(), full.as_ref(), TOL);
        assert_close(lower[(0, 1)], 99.0, TOL);
    }

    #[test]
    fn get_set_params_roundtrip() {
        let mut kernel = MaternArdKernel::new(&[2.0, 0.5], MaternNu::ThreeHalves).expect("valid");
        let mut params = [0.0; 2];
        kernel.get_params(&mut params).expect("len 2");
        assert_close(params[0], 2.0_f64.ln(), TOL);
        assert_close(params[1], 0.5_f64.ln(), TOL);
        params[0] = 0.5_f64.ln();
        kernel.set_params(&params).expect("len 2");
        assert_close(kernel.lengthscale(0).expect("dim 0"), 0.5, TOL);
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
        assert_close(cross[(0, 1)], square[(0, 1)], TOL);
        assert_close(cross[(1, 1)], square[(1, 1)], TOL);
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
            Err(GprError::IndexOutOfRange { .. })
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
