//! Isotropic rational quadratic kernel.

use super::lengthscale::{validate_lengthscale, validate_log_lengthscale};
use super::{
    Triangle, finite_dist, finite_kernel, validate_log_positive, validate_positive_finite,
    write_dense, write_rect_from_coords, write_square_from_coords, write_triangle,
};
use crate::error::GprError;
use crate::kernel::KernelScalar;
use crate::param::{BoundedParam, Interval};
use faer::{MatMut, MatRef};

/// Isotropic rational quadratic: `k = (1 + ‖x-x'‖² / (2αℓ²))^(-α)`.
///
/// Optimizer parameters are `θ = [log(ℓ), log(α)]`. Amplitude is not stored
/// here; compose with [`super::ConstantKernel`]. When every ARD lengthscale
/// equals this `ℓ` and `α` matches, values match
/// [`super::RationalQuadraticArdKernel`]. `dist` is squared Euclidean.
///
/// # Examples
///
/// ```rust
/// use gprx::kernel::RationalQuadraticKernel;
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let k = RationalQuadraticKernel::new(1.0, 1.5)?;
/// assert!(k.lengthscale() > 0.0);
/// assert!(k.alpha() > 0.0);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RationalQuadraticKernel {
    lengthscale: BoundedParam,
    alpha: BoundedParam,
}

impl RationalQuadraticKernel {
    /// Builds an isotropic RQ kernel from positive finite `ℓ` and `α`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if either value is not
    /// finite or not strictly positive.
    pub fn new(lengthscale: f64, alpha: f64) -> Result<Self, GprError> {
        validate_lengthscale(lengthscale)?;
        validate_positive_finite(alpha, "alpha")?;
        Ok(Self {
            lengthscale: BoundedParam::default_positive(lengthscale)?,
            alpha: BoundedParam::default_positive(alpha)?,
        })
    }

    /// Builds an isotropic RQ kernel from `θ = [log(ℓ), log(α)]`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if a `θ` is not finite, if
    /// `exp(θ)` overflows, or if `exp(θ)` underflows to zero.
    pub fn from_log(log_lengthscale: f64, log_alpha: f64) -> Result<Self, GprError> {
        Ok(Self {
            lengthscale: BoundedParam::default_positive(
                validate_log_lengthscale(log_lengthscale)?.exp(),
            )?,
            alpha: BoundedParam::default_positive(
                validate_log_positive(log_alpha, "alpha")?.exp(),
            )?,
        })
    }

    /// Returns `ℓ = exp(θ_0)`.
    pub fn lengthscale(&self) -> f64 {
        self.lengthscale.value()
    }

    /// Returns `θ_0 = log(ℓ)`.
    pub fn log_lengthscale(&self) -> f64 {
        self.lengthscale.ln()
    }

    /// Returns `α = exp(θ_1)`.
    pub fn alpha(&self) -> f64 {
        self.alpha.value()
    }

    /// Returns `θ_1 = log(α)`.
    pub fn log_alpha(&self) -> f64 {
        self.alpha.ln()
    }

    /// Returns the open interval on `ℓ`.
    pub fn lengthscale_bounds(&self) -> Interval {
        self.lengthscale.interval()
    }

    /// Returns the open interval on `α`.
    pub fn alpha_bounds(&self) -> Interval {
        self.alpha.interval()
    }

    /// Rebuilds this kernel with new intervals on `ℓ` and `α`.
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
            lengthscale: self.lengthscale.with_interval(lengthscale)?,
            alpha: self.alpha.with_interval(alpha)?,
        })
    }

    /// Returns the number of optimizer parameters (always 2).
    pub fn num_params(&self) -> usize {
        2
    }

    /// Writes `[log(ℓ), log(α)]` into a length-2 slice.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::LengthMismatch`] if `out` is not length 2.
    pub fn get_params(&self, out: &mut [f64]) -> Result<(), GprError> {
        expect_two_params(out.len())?;
        out[0] = self.lengthscale.ln();
        out[1] = self.alpha.ln();
        Ok(())
    }

    /// Replaces `[log(ℓ), log(α)]`. Previous values are kept on error.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::LengthMismatch`] if `params` is not length 2,
    /// or [`GprError::InvalidHyperparameter`] if a `θ` is invalid.
    pub fn set_params(&mut self, params: &[f64]) -> Result<(), GprError> {
        expect_two_params(params.len())?;
        let ell = validate_log_lengthscale(params[0])?.exp();
        let alpha = validate_log_positive(params[1], "alpha")?.exp();
        let lengthscale = BoundedParam::new(ell, self.lengthscale.interval())?;
        let alpha = BoundedParam::new(alpha, self.alpha.interval())?;
        self.lengthscale = lengthscale;
        self.alpha = alpha;
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
        let ell_sq = T::from_f64(self.lengthscale() * self.lengthscale());
        let alpha = T::from_f64(self.alpha());
        write_triangle(dist, out, uplo, |d| rq_from_sq_dist(d, ell_sq, alpha))
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
        let ell_sq = T::from_f64(self.lengthscale() * self.lengthscale());
        let alpha = T::from_f64(self.alpha());
        write_dense(dist, out, |d| rq_from_sq_dist(d, ell_sq, alpha))
    }

    /// Writes the stationary diagonal `k(x, x) = 1` into `out`.
    pub fn fill_diag<T: KernelScalar>(&self, out: &mut [T]) {
        out.fill(T::from_f64(1.0));
    }

    /// Writes `∂K/∂θ` into `d_k`. Index 0 is `log(ℓ)`, index 1 is `log(α)`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::IndexOutOfRange`] if `param_idx` is not 0 or
    /// 1, or the same shape / non-finite errors as [`Self::apply`].
    pub fn grad<T: KernelScalar>(
        &self,
        dist: MatRef<'_, T>,
        d_k: MatMut<'_, T>,
        param_idx: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        if param_idx > 1 {
            return Err(GprError::IndexOutOfRange {
                reason: format!("rational quadratic parameter index {param_idx} is out of range"),
            });
        }
        let ell_sq = T::from_f64(self.lengthscale() * self.lengthscale());
        let alpha = T::from_f64(self.alpha());
        write_triangle(dist, d_k, uplo, |d| {
            let r2 = scaled_r2(d, ell_sq)?;
            finite_kernel(if param_idx == 0 {
                rq_dk_dtheta_lengthscale(r2, alpha)
            } else {
                rq_dk_dtheta_alpha(r2, alpha)
            })
        })
    }

    /// `⟨weight, ∂K/∂θ_p⟩_F` for both parameters over the lower triangle of
    /// `dist`, in one pass, and returns `⟨weight, K⟩_F`. Each entry's `ln`
    /// serves both derivatives, and no `∂K` matrix is written.
    ///
    /// `k` is this leaf's Gram at the same `θ` when the caller kept it; its
    /// entries replace `exp(−α ln u)`.
    pub(crate) fn weighted_grads_dist<T: KernelScalar>(
        &self,
        dist: MatRef<'_, T>,
        k: Option<MatRef<'_, T>>,
        weight: MatRef<'_, T>,
        out: &mut [f64],
    ) -> Result<f64, GprError> {
        let ell_sq = T::from_f64(self.lengthscale() * self.lengthscale());
        let alpha = T::from_f64(self.alpha());
        let one = T::from_f64(1.0);
        let two_alpha = T::from_f64(2.0) * alpha;
        let (mut g_ell, mut g_alpha, mut value) = (0.0, 0.0, 0.0);
        for col in 0..dist.ncols() {
            for row in col..dist.nrows() {
                let r2 = scaled_r2(dist[(row, col)], ell_sq)?;
                let u = one + r2 / two_alpha;
                let ln_u = u.ln();
                let kv = match k {
                    Some(k) => k[(row, col)],
                    None => (-alpha * ln_u).exp(),
                };
                let dk_ell = finite_kernel((kv / u) * r2)?;
                let dk_alpha = finite_kernel(alpha * kv * (-ln_u + one - one / u))?;
                let w = weight[(row, col)].to_f64() * if row == col { 1.0 } else { 2.0 };
                g_ell += w * dk_ell.to_f64();
                g_alpha += w * dk_alpha.to_f64();
                value += w * kv.to_f64();
            }
        }
        out[0] = g_ell;
        out[1] = g_alpha;
        Ok(value)
    }

    pub(crate) fn apply_from_coords<T: KernelScalar>(
        &self,
        x: MatRef<'_, T>,
        out: MatMut<'_, T>,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        let ell_sq = T::from_f64(self.lengthscale() * self.lengthscale());
        let alpha = T::from_f64(self.alpha());
        write_square_from_coords(x, out, uplo, |d| rq_from_sq_dist(d, ell_sq, alpha))
    }

    pub(crate) fn grad_from_coords<T: KernelScalar>(
        &self,
        x: MatRef<'_, T>,
        d_k: MatMut<'_, T>,
        param_idx: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        if param_idx > 1 {
            return Err(GprError::IndexOutOfRange {
                reason: format!("rational quadratic parameter index {param_idx} is out of range"),
            });
        }
        let ell_sq = T::from_f64(self.lengthscale() * self.lengthscale());
        let alpha = T::from_f64(self.alpha());
        write_square_from_coords(x, d_k, uplo, |d| {
            let r2 = scaled_r2(d, ell_sq)?;
            finite_kernel(if param_idx == 0 {
                rq_dk_dtheta_lengthscale(r2, alpha)
            } else {
                rq_dk_dtheta_alpha(r2, alpha)
            })
        })
    }

    /// Writes `∂²K/∂θ_i ∂θ_j`. Index 0 is `log(ℓ)`, index 1 is `log(α)`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::IndexOutOfRange`] if `i` or `j` is out of
    /// range, or the same shape / non-finite errors as [`Self::apply`].
    pub fn hess<T: KernelScalar>(
        &self,
        dist: MatRef<'_, T>,
        d2_k: MatMut<'_, T>,
        i: usize,
        j: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        require_rq_hess_idx(i, j)?;
        let ell_sq = T::from_f64(self.lengthscale() * self.lengthscale());
        let alpha = T::from_f64(self.alpha());
        write_triangle(dist, d2_k, uplo, |d| {
            let r2 = scaled_r2(d, ell_sq)?;
            finite_kernel(rq_d2k(r2, alpha, i, j))
        })
    }

    pub(crate) fn hess_from_coords<T: KernelScalar>(
        &self,
        x: MatRef<'_, T>,
        d2_k: MatMut<'_, T>,
        i: usize,
        j: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        require_rq_hess_idx(i, j)?;
        let ell_sq = T::from_f64(self.lengthscale() * self.lengthscale());
        let alpha = T::from_f64(self.alpha());
        write_square_from_coords(x, d2_k, uplo, |d| {
            let r2 = scaled_r2(d, ell_sq)?;
            finite_kernel(rq_d2k(r2, alpha, i, j))
        })
    }

    /// Rectangular `K(x1, x2)` from coordinates.
    pub(crate) fn apply_cross_from_coords<T: KernelScalar>(
        &self,
        x1: MatRef<'_, T>,
        x2: MatRef<'_, T>,
        out: MatMut<'_, T>,
    ) -> Result<(), GprError> {
        let ell_sq = T::from_f64(self.lengthscale() * self.lengthscale());
        let alpha = T::from_f64(self.alpha());
        write_rect_from_coords(x1, x2, out, |d| rq_from_sq_dist(d, ell_sq, alpha))
    }

    /// `∂K(x1, x2)/∂θ` of a rectangular block, from coordinates.
    pub(crate) fn grad_cross_from_coords<T: KernelScalar>(
        &self,
        x1: MatRef<'_, T>,
        x2: MatRef<'_, T>,
        d_k: MatMut<'_, T>,
        param_idx: usize,
    ) -> Result<(), GprError> {
        if param_idx > 1 {
            return Err(GprError::IndexOutOfRange {
                reason: format!("rational quadratic parameter index {param_idx} is out of range"),
            });
        }
        let ell_sq = T::from_f64(self.lengthscale() * self.lengthscale());
        let alpha = T::from_f64(self.alpha());
        write_rect_from_coords(x1, x2, d_k, |d| {
            let r2 = scaled_r2(d, ell_sq)?;
            finite_kernel(if param_idx == 0 {
                rq_dk_dtheta_lengthscale(r2, alpha)
            } else {
                rq_dk_dtheta_alpha(r2, alpha)
            })
        })
    }

    /// `∂²K(x1, x2)/∂θ_i ∂θ_j` of a rectangular block, from coordinates.
    pub(crate) fn hess_cross_from_coords<T: KernelScalar>(
        &self,
        x1: MatRef<'_, T>,
        x2: MatRef<'_, T>,
        d2_k: MatMut<'_, T>,
        i: usize,
        j: usize,
    ) -> Result<(), GprError> {
        require_rq_hess_idx(i, j)?;
        let ell_sq = T::from_f64(self.lengthscale() * self.lengthscale());
        let alpha = T::from_f64(self.alpha());
        write_rect_from_coords(x1, x2, d2_k, |d| {
            let r2 = scaled_r2(d, ell_sq)?;
            finite_kernel(rq_d2k(r2, alpha, i, j))
        })
    }
}

pub(crate) fn rq_from_r2<T: KernelScalar>(r2: T, alpha: T) -> T {
    let u = T::from_f64(1.0) + r2 / (T::from_f64(2.0) * alpha);
    u.powf(-alpha)
}

pub(crate) fn rq_dk_dtheta_lengthscale<T: KernelScalar>(r2: T, alpha: T) -> T {
    let u = T::from_f64(1.0) + r2 / (T::from_f64(2.0) * alpha);
    let k = u.powf(-alpha);
    (k / u) * r2
}

pub(crate) fn rq_dk_dtheta_alpha<T: KernelScalar>(r2: T, alpha: T) -> T {
    let u = T::from_f64(1.0) + r2 / (T::from_f64(2.0) * alpha);
    let k = u.powf(-alpha);
    alpha * k * (-u.ln() + T::from_f64(1.0) - T::from_f64(1.0) / u)
}

pub(crate) fn rq_dk_dtheta_ard_dim<T: KernelScalar>(r2: T, alpha: T, dim_term: T) -> T {
    let u = T::from_f64(1.0) + r2 / (T::from_f64(2.0) * alpha);
    let k = u.powf(-alpha);
    (k / u) * dim_term
}

pub(crate) fn rq_d2k<T: KernelScalar>(r2: T, alpha: T, i: usize, j: usize) -> T {
    let (a, b) = if i <= j { (i, j) } else { (j, i) };
    let u = T::from_f64(1.0) + r2 / (T::from_f64(2.0) * alpha);
    let k = u.powf(-alpha);
    match (a, b) {
        (0, 0) => {
            -T::from_f64(2.0) * (k / u) * r2
                + (T::from_f64(1.0) + T::from_f64(1.0) / alpha) * r2 * r2 * k / (u * u)
        }
        (0, 1) => {
            (k / u)
                * r2
                * (-alpha * u.ln() + (alpha + T::from_f64(1.0)) * (u - T::from_f64(1.0)) / u)
        }
        (1, 1) => {
            let v = -u.ln() + T::from_f64(1.0) - T::from_f64(1.0) / u;
            let dv = (T::from_f64(1.0) - u) * (T::from_f64(1.0) - u) / (u * u);
            let h = alpha * k * v;
            h + (h * h) / k + alpha * k * dv
        }
        _ => T::from_f64(0.0),
    }
}

pub(crate) fn rq_d2k_ard<T: KernelScalar>(
    r2: T,
    alpha: T,
    dim_i: T,
    dim_j: T,
    i: usize,
    j: usize,
    d: usize,
) -> T {
    let u = T::from_f64(1.0) + r2 / (T::from_f64(2.0) * alpha);
    let k = u.powf(-alpha);
    let alpha_idx = d;
    if i == alpha_idx && j == alpha_idx {
        return rq_d2k(r2, alpha, 1, 1);
    }
    if i == alpha_idx || j == alpha_idx {
        let dim = if i == alpha_idx { dim_j } else { dim_i };
        return (k / u)
            * dim
            * (-alpha * u.ln() + (alpha + T::from_f64(1.0)) * (u - T::from_f64(1.0)) / u);
    }
    if i == j {
        -T::from_f64(2.0) * (k / u) * dim_i
            + (T::from_f64(1.0) + T::from_f64(1.0) / alpha) * dim_i * dim_i * k / (u * u)
    } else {
        (T::from_f64(1.0) + T::from_f64(1.0) / alpha) * dim_i * dim_j * k / (u * u)
    }
}

fn require_rq_hess_idx(i: usize, j: usize) -> Result<(), GprError> {
    if i <= 1 && j <= 1 {
        Ok(())
    } else {
        Err(GprError::IndexOutOfRange {
            reason: format!("rational quadratic parameter pair ({i}, {j}) is out of range"),
        })
    }
}

fn expect_two_params(len: usize) -> Result<(), GprError> {
    if len == 2 {
        Ok(())
    } else {
        Err(GprError::LengthMismatch {
            reason: format!("expected 2 rational quadratic parameters, got {len}"),
        })
    }
}

fn scaled_r2<T: KernelScalar>(sq_dist: T, ell_sq: T) -> Result<T, GprError> {
    let d = finite_dist(sq_dist)?;
    Ok(d.max(T::from_f64(0.0)) / ell_sq)
}

fn rq_from_sq_dist<T: KernelScalar>(sq_dist: T, ell_sq: T, alpha: T) -> Result<T, GprError> {
    let r2 = scaled_r2(sq_dist, ell_sq)?;
    finite_kernel(rq_from_r2(r2, alpha))
}

#[cfg(test)]
mod tests {
    use super::RationalQuadraticKernel;
    use crate::error::GprError;
    use crate::kernel::Triangle;
    use faer::{Mat, mat};

    const TOL: f64 = 1e-8;

    use crate::test_check::{assert_close, assert_lower_close, assert_send_sync, fill, sq_dist_1d};

    #[test]
    fn is_send_sync() {
        assert_send_sync::<RationalQuadraticKernel>();
    }

    #[test]
    fn diagonal_is_one() {
        let kernel = RationalQuadraticKernel::new(1.25, 0.8).expect("valid");
        let dist = sq_dist_1d(&[0.0, 1.0, 3.0]);
        let mut k = fill(3, f64::NAN);
        kernel
            .apply(dist.as_ref(), k.as_mut(), Triangle::Full)
            .expect("shape");
        assert_close(k[(0, 0)], 1.0, TOL);
        assert_close(k[(1, 1)], 1.0, TOL);
        assert_close(k[(2, 2)], 1.0, TOL);
    }

    #[test]
    fn known_value_when_unit_inside() {
        let ell = 2.0;
        let alpha = 1.5;
        let d2 = 2.0 * alpha * ell * ell;
        let kernel = RationalQuadraticKernel::new(ell, alpha).expect("valid");
        let dist = mat![[0.0, d2], [d2, 0.0]];
        let mut k = fill(2, 0.0);
        kernel
            .apply(dist.as_ref(), k.as_mut(), Triangle::Full)
            .expect("shape");
        assert_close(k[(0, 1)], 2.0_f64.powf(-alpha), TOL);
    }

    #[test]
    fn full_is_symmetric() {
        let kernel = RationalQuadraticKernel::new(0.9, 2.0).expect("valid");
        let dist = sq_dist_1d(&[0.0, 0.4, 1.2, 2.0]);
        let mut k = fill(4, 0.0);
        kernel
            .apply(dist.as_ref(), k.as_mut(), Triangle::Full)
            .expect("shape");
        for col in 0..4 {
            for row in 0..4 {
                assert_close(k[(row, col)], k[(col, row)], TOL);
            }
        }
    }

    #[test]
    fn lower_matches_full_and_leaves_upper() {
        let kernel = RationalQuadraticKernel::new(1.0, 1.0).expect("valid");
        let dist = sq_dist_1d(&[0.0, 0.7, 1.4]);
        let mut full = fill(3, 0.0);
        kernel
            .apply(dist.as_ref(), full.as_mut(), Triangle::Full)
            .expect("shape");
        let sentinel = 42.0;
        let mut lower = fill(3, sentinel);
        kernel
            .apply(dist.as_ref(), lower.as_mut(), Triangle::Lower)
            .expect("shape");
        assert_lower_close(lower.as_ref(), full.as_ref(), TOL);
        assert_close(lower[(0, 1)], sentinel, TOL);
    }

    #[test]
    fn upper_matches_full() {
        let kernel = RationalQuadraticKernel::new(1.0, 0.5).expect("valid");
        let dist = sq_dist_1d(&[0.0, 1.0, 1.5]);
        let mut full = fill(3, 0.0);
        let mut upper = fill(3, -1.0);
        kernel
            .apply(dist.as_ref(), full.as_mut(), Triangle::Full)
            .expect("shape");
        kernel
            .apply(dist.as_ref(), upper.as_mut(), Triangle::Upper)
            .expect("shape");
        for col in 0..3 {
            for row in 0..=col {
                assert_close(upper[(row, col)], full[(row, col)], TOL);
            }
        }
        assert_close(upper[(1, 0)], -1.0, TOL);
    }

    #[test]
    fn hess_matches_finite_difference_of_grad() {
        let kernel = RationalQuadraticKernel::from_log(-0.2, 0.3).expect("valid");
        let mut theta = [0.0; 2];
        kernel.get_params(&mut theta).expect("len 2");
        let h = 1e-6;
        let dist = sq_dist_1d(&[0.0, 0.9, 1.7]);
        for i in 0..2 {
            for j in 0..2 {
                let mut plus_th = theta;
                let mut minus_th = theta;
                plus_th[j] += h;
                minus_th[j] -= h;
                let plus =
                    RationalQuadraticKernel::from_log(plus_th[0], plus_th[1]).expect("valid");
                let minus =
                    RationalQuadraticKernel::from_log(minus_th[0], minus_th[1]).expect("valid");
                let mut g_plus = fill(3, 0.0);
                let mut g_minus = fill(3, 0.0);
                let mut d2 = fill(3, 0.0);
                plus.grad(dist.as_ref(), g_plus.as_mut(), i, Triangle::Full)
                    .expect("plus");
                minus
                    .grad(dist.as_ref(), g_minus.as_mut(), i, Triangle::Full)
                    .expect("minus");
                kernel
                    .hess(dist.as_ref(), d2.as_mut(), i, j, Triangle::Full)
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
    fn grad_matches_finite_difference() {
        let kernel = RationalQuadraticKernel::from_log(-0.2, 0.3).expect("valid");
        let mut theta = [0.0; 2];
        kernel.get_params(&mut theta).expect("len 2");
        let h = 1e-6;
        let dist = sq_dist_1d(&[0.0, 0.9, 1.7]);
        for idx in 0..2 {
            let mut plus_th = theta;
            let mut minus_th = theta;
            plus_th[idx] += h;
            minus_th[idx] -= h;
            let plus = RationalQuadraticKernel::from_log(plus_th[0], plus_th[1]).expect("valid");
            let minus = RationalQuadraticKernel::from_log(minus_th[0], minus_th[1]).expect("valid");
            let mut k_plus = fill(3, 0.0);
            let mut k_minus = fill(3, 0.0);
            let mut dk = fill(3, 0.0);
            plus.apply(dist.as_ref(), k_plus.as_mut(), Triangle::Full)
                .expect("plus");
            minus
                .apply(dist.as_ref(), k_minus.as_mut(), Triangle::Full)
                .expect("minus");
            kernel
                .grad(dist.as_ref(), dk.as_mut(), idx, Triangle::Full)
                .expect("idx");
            for col in 0..3 {
                for row in 0..3 {
                    let fd = (k_plus[(row, col)] - k_minus[(row, col)]) / (2.0 * h);
                    assert_close(dk[(row, col)], fd, TOL);
                }
            }
        }
    }

    #[test]
    fn grad_lower_matches_full() {
        let kernel = RationalQuadraticKernel::new(1.0, 1.25).expect("valid");
        let dist = sq_dist_1d(&[0.0, 0.8, 1.6]);
        let mut full = fill(3, 0.0);
        let mut lower = fill(3, 99.0);
        kernel
            .grad(dist.as_ref(), full.as_mut(), 1, Triangle::Full)
            .expect("alpha");
        kernel
            .grad(dist.as_ref(), lower.as_mut(), 1, Triangle::Lower)
            .expect("alpha");
        assert_lower_close(lower.as_ref(), full.as_ref(), TOL);
        assert_close(lower[(0, 1)], 99.0, TOL);
    }

    #[test]
    fn get_set_params_roundtrip_and_atomic() {
        let mut kernel = RationalQuadraticKernel::new(2.0, 3.0).expect("valid");
        let mut params = [0.0; 2];
        kernel.get_params(&mut params).expect("len 2");
        assert_close(params[0], 2.0_f64.ln(), TOL);
        assert_close(params[1], 3.0_f64.ln(), TOL);
        params[0] = 0.5_f64.ln();
        kernel.set_params(&params).expect("len 2");
        assert_close(kernel.lengthscale(), 0.5, TOL);
        let before = kernel;
        assert!(kernel.set_params(&[0.0, f64::INFINITY]).is_err());
        assert_eq!(kernel, before);
    }

    #[test]
    fn apply_cross_and_diag() {
        let kernel = RationalQuadraticKernel::new(1.0, 2.0).expect("valid");
        let dist = mat![[0.0, 1.0], [4.0, 0.0]];
        let mut out = Mat::zeros(2, 2);
        kernel
            .apply_cross(dist.as_ref(), out.as_mut())
            .expect("rect");
        let mut square = fill(2, 0.0);
        kernel
            .apply(dist.as_ref(), square.as_mut(), Triangle::Full)
            .expect("square");
        assert_close(out[(0, 1)], square[(0, 1)], TOL);
        let mut diag = [0.0, 0.0];
        kernel.fill_diag(&mut diag);
        assert_close(diag[0], 1.0, TOL);
    }

    #[test]
    fn rejects_non_positive_and_bad_index() {
        assert!(matches!(
            RationalQuadraticKernel::new(0.0, 1.0),
            Err(GprError::InvalidHyperparameter { .. })
        ));
        assert!(matches!(
            RationalQuadraticKernel::new(1.0, 0.0),
            Err(GprError::InvalidHyperparameter { .. })
        ));
        let kernel = RationalQuadraticKernel::new(1.0, 1.0).expect("valid");
        let dist = sq_dist_1d(&[0.0, 1.0]);
        let mut dk = fill(2, 0.0);
        assert!(matches!(
            kernel.grad(dist.as_ref(), dk.as_mut(), 2, Triangle::Lower),
            Err(GprError::IndexOutOfRange { .. })
        ));
    }

    #[test]
    fn apply_rejects_non_finite_dist() {
        let kernel = RationalQuadraticKernel::new(1.0, 1.0).expect("valid");
        let dist = mat![[0.0, f64::NAN], [f64::NAN, 0.0]];
        let mut k = fill(2, 0.0);
        assert!(matches!(
            kernel.apply(dist.as_ref(), k.as_mut(), Triangle::Full),
            Err(GprError::NonFiniteInput)
        ));
    }
}
