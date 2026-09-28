//! Periodic (exp-sine-squared) kernel.

use super::lengthscale::{validate_lengthscale, validate_log_lengthscale};
use super::{
    Triangle, finite_dist, validate_log_positive, validate_positive_finite, write_dense,
    write_square_from_coords, write_triangle,
};
use crate::error::GprError;
use crate::math::KernelMath;
use crate::param::{BoundedParam, Interval};
use faer::{MatMut, MatRef};

/// Periodic kernel: `k = exp( -2 sin²(π ‖x-x'‖ / p) / ℓ² )`.
///
/// Optimizer parameters are `θ = [log(ℓ), log(p)]`. The input to `sin` is
/// Euclidean distance, not squared Euclidean; `dist` still stores squared
/// distances and this leaf takes the square root. Lengthscale stays scalar
/// (no ARD). Amplitude is not stored here; compose with
/// [`super::ConstantKernel`].
///
/// # Examples
///
/// ```rust
/// use gprx::kernel::PeriodicKernel;
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let k = PeriodicKernel::new(1.0, 2.0)?;
/// assert!(k.lengthscale() > 0.0);
/// assert!(k.period() > 0.0);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PeriodicKernel {
    lengthscale: BoundedParam,
    period: BoundedParam,
}

impl PeriodicKernel {
    /// Builds a periodic kernel from positive finite `ℓ` and period `p`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if either value is not
    /// finite or not strictly positive.
    pub fn new(lengthscale: f64, period: f64) -> Result<Self, GprError> {
        validate_lengthscale(lengthscale)?;
        validate_positive_finite(period, "period")?;
        Ok(Self {
            lengthscale: BoundedParam::default_positive(lengthscale)?,
            period: BoundedParam::default_positive(period)?,
        })
    }

    /// Builds a periodic kernel from `θ = [log(ℓ), log(p)]`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if a `θ` is not finite, if
    /// `exp(θ)` overflows, or if `exp(θ)` underflows to zero.
    pub fn from_log(log_lengthscale: f64, log_period: f64) -> Result<Self, GprError> {
        Ok(Self {
            lengthscale: BoundedParam::default_positive(
                validate_log_lengthscale(log_lengthscale)?.exp(),
            )?,
            period: BoundedParam::default_positive(
                validate_log_positive(log_period, "period")?.exp(),
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

    /// Returns `p = exp(θ_1)`.
    pub fn period(&self) -> f64 {
        self.period.value()
    }

    /// Returns `θ_1 = log(p)`.
    pub fn log_period(&self) -> f64 {
        self.period.ln()
    }

    /// Returns the open interval on `ℓ`.
    pub fn lengthscale_bounds(&self) -> Interval {
        self.lengthscale.interval()
    }

    /// Returns the open interval on `p`.
    pub fn period_bounds(&self) -> Interval {
        self.period.interval()
    }

    /// Rebuilds this kernel with new intervals on `ℓ` and `p`.
    ///
    /// # Errors
    ///
    /// Returns [`crate::IntervalError`] if a current value is not strictly
    /// inside the matching interval.
    pub fn with_bounds(
        self,
        lengthscale: Interval,
        period: Interval,
    ) -> Result<Self, crate::IntervalError> {
        Ok(Self {
            lengthscale: self.lengthscale.with_interval(lengthscale)?,
            period: self.period.with_interval(period)?,
        })
    }

    /// Returns the number of optimizer parameters (always 2).
    pub fn num_params(&self) -> usize {
        2
    }

    /// Writes `[log(ℓ), log(p)]` into a length-2 slice.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::LengthMismatch`] if `out` is not length 2.
    pub fn get_params(&self, out: &mut [f64]) -> Result<(), GprError> {
        expect_two_params(out.len())?;
        out[0] = self.lengthscale.ln();
        out[1] = self.period.ln();
        Ok(())
    }

    /// Replaces `[log(ℓ), log(p)]`. Previous values are kept on error.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::LengthMismatch`] if `params` is not length 2,
    /// or [`GprError::InvalidHyperparameter`] if a `θ` is invalid.
    pub fn set_params(&mut self, params: &[f64]) -> Result<(), GprError> {
        expect_two_params(params.len())?;
        let ell = validate_log_lengthscale(params[0])?.exp();
        let period = validate_log_positive(params[1], "period")?.exp();
        let lengthscale = BoundedParam::new(ell, self.lengthscale.interval())?;
        let period = BoundedParam::new(period, self.period.interval())?;
        self.lengthscale = lengthscale;
        self.period = period;
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
    pub fn apply(
        &self,
        dist: MatRef<'_, f64>,
        out: MatMut<'_, f64>,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        self.apply_math::<crate::math::Accurate>(dist, out, uplo)
    }

    pub(crate) fn apply_math<M: KernelMath>(
        &self,
        dist: MatRef<'_, f64>,
        out: MatMut<'_, f64>,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        let ell = self.lengthscale();
        let period = self.period();
        write_triangle(dist, out, uplo, |d| {
            periodic_from_sq_dist::<M>(d, ell, period)
        })
    }

    /// Writes rectangular `k(dist)` into `out` (train × test).
    ///
    /// # Errors
    ///
    /// Returns [`GprError`] if the matrices are empty, size mismatched, or if
    /// `dist` contains a non-finite value.
    pub fn apply_cross(&self, dist: MatRef<'_, f64>, out: MatMut<'_, f64>) -> Result<(), GprError> {
        self.apply_cross_math::<crate::math::Accurate>(dist, out)
    }

    pub(crate) fn apply_cross_math<M: KernelMath>(
        &self,
        dist: MatRef<'_, f64>,
        out: MatMut<'_, f64>,
    ) -> Result<(), GprError> {
        let ell = self.lengthscale();
        let period = self.period();
        write_dense(dist, out, |d| periodic_from_sq_dist::<M>(d, ell, period))
    }

    /// Writes the stationary diagonal `k(x, x) = 1` into `out`.
    pub fn fill_diag(&self, out: &mut [f64]) {
        out.fill(1.0);
    }

    /// Writes `∂K/∂θ` into `d_k`. Index 0 is `log(ℓ)`, index 1 is `log(p)`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::IndexOutOfRange`] if `param_idx` is not 0 or
    /// 1, or the same shape / non-finite errors as [`Self::apply`].
    pub fn grad(
        &self,
        dist: MatRef<'_, f64>,
        d_k: MatMut<'_, f64>,
        param_idx: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        if param_idx > 1 {
            return Err(GprError::IndexOutOfRange {
                reason: format!("periodic kernel parameter index {param_idx} is out of range"),
            });
        }
        self.grad_math::<crate::math::Accurate>(dist, d_k, param_idx, uplo)
    }

    pub(crate) fn grad_math<M: KernelMath>(
        &self,
        dist: MatRef<'_, f64>,
        d_k: MatMut<'_, f64>,
        param_idx: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        if param_idx > 1 {
            return Err(GprError::IndexOutOfRange {
                reason: format!("periodic kernel parameter index {param_idx} is out of range"),
            });
        }
        let ell = self.lengthscale();
        let period = self.period();
        write_triangle(dist, d_k, uplo, |d| {
            periodic_grad_from_sq_dist::<M>(d, ell, period, param_idx)
        })
    }

    pub(crate) fn apply_from_coords<M: KernelMath>(
        &self,
        x: MatRef<'_, f64>,
        out: MatMut<'_, f64>,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        let ell = self.lengthscale();
        let period = self.period();
        write_square_from_coords(x, out, uplo, |d| periodic_from_sq_dist::<M>(d, ell, period))
    }

    pub(crate) fn grad_from_coords<M: KernelMath>(
        &self,
        x: MatRef<'_, f64>,
        d_k: MatMut<'_, f64>,
        param_idx: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        if param_idx > 1 {
            return Err(GprError::IndexOutOfRange {
                reason: format!("periodic kernel parameter index {param_idx} is out of range"),
            });
        }
        let ell = self.lengthscale();
        let period = self.period();
        write_square_from_coords(x, d_k, uplo, |d| {
            periodic_grad_from_sq_dist::<M>(d, ell, period, param_idx)
        })
    }

    /// Writes `∂²K/∂θ_i ∂θ_j`. Index 0 is `log(ℓ)`, index 1 is `log(p)`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::IndexOutOfRange`] if `i` or `j` is out of
    /// range, or the same shape / non-finite errors as [`Self::apply`].
    pub fn hess(
        &self,
        dist: MatRef<'_, f64>,
        d2_k: MatMut<'_, f64>,
        i: usize,
        j: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        self.hess_math::<crate::math::Accurate>(dist, d2_k, i, j, uplo)
    }

    pub(crate) fn hess_math<M: KernelMath>(
        &self,
        dist: MatRef<'_, f64>,
        d2_k: MatMut<'_, f64>,
        i: usize,
        j: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        require_periodic_hess_idx(i, j)?;
        let ell = self.lengthscale();
        let period = self.period();
        write_triangle(dist, d2_k, uplo, |d| {
            periodic_hess_from_sq_dist::<M>(d, ell, period, i, j)
        })
    }

    pub(crate) fn hess_from_coords<M: KernelMath>(
        &self,
        x: MatRef<'_, f64>,
        d2_k: MatMut<'_, f64>,
        i: usize,
        j: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        require_periodic_hess_idx(i, j)?;
        let ell = self.lengthscale();
        let period = self.period();
        write_square_from_coords(x, d2_k, uplo, |d| {
            periodic_hess_from_sq_dist::<M>(d, ell, period, i, j)
        })
    }
}

fn expect_two_params(len: usize) -> Result<(), GprError> {
    if len == 2 {
        Ok(())
    } else {
        Err(GprError::LengthMismatch {
            reason: format!("expected 2 periodic parameters, got {len}"),
        })
    }
}

fn euclidean_from_sq(sq_dist: f64) -> Result<f64, GprError> {
    let d = finite_dist(sq_dist)?;
    Ok(d.max(0.0).sqrt())
}

fn periodic_from_r<M: KernelMath>(r: f64, ell: f64, period: f64) -> f64 {
    let s = (std::f64::consts::PI * r / period).sin();
    let inv_ell = 1.0 / ell;
    M::exp(-2.0 * s * s * inv_ell * inv_ell)
}

fn periodic_from_sq_dist<M: KernelMath>(
    sq_dist: f64,
    ell: f64,
    period: f64,
) -> Result<f64, GprError> {
    let r = euclidean_from_sq(sq_dist)?;
    finite_kernel(periodic_from_r::<M>(r, ell, period))
}

fn periodic_grad_from_sq_dist<M: KernelMath>(
    sq_dist: f64,
    ell: f64,
    period: f64,
    param_idx: usize,
) -> Result<f64, GprError> {
    let r = euclidean_from_sq(sq_dist)?;
    let alpha = std::f64::consts::PI * r / period;
    let s = alpha.sin();
    let inv_ell_sq = 1.0 / (ell * ell);
    let z = -2.0 * s * s * inv_ell_sq;
    let beta = 4.0 * s * s * inv_ell_sq;
    let gamma = 4.0 * s * alpha.cos() * alpha * inv_ell_sq;
    let dk = if M::ACCURATE {
        let k = z.exp();
        if param_idx == 0 { k * beta } else { k * gamma }
    } else {
        let d1 = M::jet(z).d1;
        if param_idx == 0 {
            d1 * beta
        } else {
            d1 * gamma
        }
    };
    finite_kernel(dk)
}

fn periodic_hess_from_sq_dist<M: KernelMath>(
    sq_dist: f64,
    ell: f64,
    period: f64,
    i: usize,
    j: usize,
) -> Result<f64, GprError> {
    let r = euclidean_from_sq(sq_dist)?;
    let alpha = std::f64::consts::PI * r / period;
    let s = alpha.sin();
    let c = alpha.cos();
    let inv_ell_sq = 1.0 / (ell * ell);
    let z = -2.0 * s * s * inv_ell_sq;
    let beta = 4.0 * s * s * inv_ell_sq;
    let gamma = 4.0 * s * c * alpha * inv_ell_sq;
    let (a, b) = if i <= j { (i, j) } else { (j, i) };
    let h = if M::ACCURATE {
        let k = z.exp();
        match (a, b) {
            (0, 0) => k * beta * (beta - 2.0),
            (0, 1) => k * gamma * (beta - 2.0),
            (1, 1) => {
                let dgamma = 4.0
                    * inv_ell_sq
                    * (-alpha * alpha * c * c + alpha * alpha * s * s - alpha * s * c);
                k * gamma * gamma + k * dgamma
            }
            _ => 0.0,
        }
    } else {
        let jet = M::jet(z);
        match (a, b) {
            (0, 0) => beta * (jet.d2 * beta - 2.0 * jet.d1),
            (0, 1) => gamma * (jet.d2 * beta - 2.0 * jet.d1),
            (1, 1) => {
                let dgamma = 4.0
                    * inv_ell_sq
                    * (-alpha * alpha * c * c + alpha * alpha * s * s - alpha * s * c);
                jet.d2 * gamma * gamma + jet.d1 * dgamma
            }
            _ => 0.0,
        }
    };
    finite_kernel(h)
}

fn require_periodic_hess_idx(i: usize, j: usize) -> Result<(), GprError> {
    if i <= 1 && j <= 1 {
        Ok(())
    } else {
        Err(GprError::IndexOutOfRange {
            reason: format!("periodic kernel parameter pair ({i}, {j}) is out of range"),
        })
    }
}

fn finite_kernel(value: f64) -> Result<f64, GprError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(GprError::NonFiniteKernelValue)
    }
}

#[cfg(test)]
mod tests {
    use super::PeriodicKernel;
    use crate::error::GprError;
    use crate::kernel::Triangle;
    use faer::{Mat, mat};

    const TOL: f64 = 1e-8;

    use crate::test_check::{assert_close, assert_lower_close, assert_send_sync, fill, sq_dist_1d};

    #[test]
    fn is_send_sync() {
        assert_send_sync::<PeriodicKernel>();
    }

    #[test]
    fn diagonal_is_one_and_period_repeats() {
        let period = 2.0;
        let kernel = PeriodicKernel::new(1.25, period).expect("valid");
        let dist = sq_dist_1d(&[0.0, 0.5, period]);
        let mut k = fill(3, f64::NAN);
        kernel
            .apply(dist.as_ref(), k.as_mut(), Triangle::Full)
            .expect("shape");
        assert_close(k[(0, 0)], 1.0, TOL);
        assert_close(k[(1, 1)], 1.0, TOL);
        assert_close(k[(2, 2)], 1.0, TOL);
        assert_close(k[(0, 2)], 1.0, TOL);
    }

    #[test]
    fn known_value_at_half_period() {
        let ell = 1.5;
        let period = 4.0;
        let half = period / 2.0;
        let kernel = PeriodicKernel::new(ell, period).expect("valid");
        let dist = mat![[0.0, half * half], [half * half, 0.0]];
        let mut k = fill(2, 0.0);
        kernel
            .apply(dist.as_ref(), k.as_mut(), Triangle::Full)
            .expect("shape");
        assert_close(k[(0, 1)], (-2.0 / (ell * ell)).exp(), TOL);
    }

    #[test]
    fn full_is_symmetric() {
        let kernel = PeriodicKernel::new(0.8, 3.0).expect("valid");
        let dist = sq_dist_1d(&[0.0, 0.4, 1.1, 2.7]);
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
        let kernel = PeriodicKernel::new(1.0, 2.5).expect("valid");
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
        assert_close(lower[(0, 2)], sentinel, TOL);
        assert_close(lower[(1, 2)], sentinel, TOL);
    }

    #[test]
    fn upper_matches_full() {
        let kernel = PeriodicKernel::new(1.0, 2.0).expect("valid");
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
        let kernel = PeriodicKernel::from_log(-0.15, 0.4).expect("valid");
        let mut theta = [0.0; 2];
        kernel.get_params(&mut theta).expect("len 2");
        let h = 1e-6;
        let dist = sq_dist_1d(&[0.0, 0.7, 1.4]);
        for i in 0..2 {
            for j in 0..2 {
                let mut plus_th = theta;
                let mut minus_th = theta;
                plus_th[j] += h;
                minus_th[j] -= h;
                let plus = PeriodicKernel::from_log(plus_th[0], plus_th[1]).expect("valid");
                let minus = PeriodicKernel::from_log(minus_th[0], minus_th[1]).expect("valid");
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
        let kernel = PeriodicKernel::from_log(-0.2, 0.4).expect("valid");
        let mut theta = [0.0; 2];
        kernel.get_params(&mut theta).expect("len 2");
        let h = 1e-6;
        let dist = sq_dist_1d(&[0.0, 0.9, 1.7]);
        for idx in 0..2 {
            let mut plus_th = theta;
            let mut minus_th = theta;
            plus_th[idx] += h;
            minus_th[idx] -= h;
            let plus = PeriodicKernel::from_log(plus_th[0], plus_th[1]).expect("valid");
            let minus = PeriodicKernel::from_log(minus_th[0], minus_th[1]).expect("valid");
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
        let kernel = PeriodicKernel::new(1.0, 2.0).expect("valid");
        let dist = sq_dist_1d(&[0.0, 0.8, 1.6]);
        let mut full = fill(3, 0.0);
        let mut lower = fill(3, 99.0);
        kernel
            .grad(dist.as_ref(), full.as_mut(), 1, Triangle::Full)
            .expect("period");
        kernel
            .grad(dist.as_ref(), lower.as_mut(), 1, Triangle::Lower)
            .expect("period");
        assert_lower_close(lower.as_ref(), full.as_ref(), TOL);
        assert_close(lower[(0, 1)], 99.0, TOL);
    }

    #[test]
    fn get_set_params_roundtrip_and_atomic() {
        let mut kernel = PeriodicKernel::new(2.0, 3.0).expect("valid");
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
        let kernel = PeriodicKernel::new(1.0, 2.0).expect("valid");
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
            PeriodicKernel::new(0.0, 1.0),
            Err(GprError::InvalidHyperparameter { .. })
        ));
        assert!(matches!(
            PeriodicKernel::new(1.0, 0.0),
            Err(GprError::InvalidHyperparameter { .. })
        ));
        let kernel = PeriodicKernel::new(1.0, 2.0).expect("valid");
        let dist = sq_dist_1d(&[0.0, 1.0]);
        let mut dk = fill(2, 0.0);
        assert!(matches!(
            kernel.grad(dist.as_ref(), dk.as_mut(), 2, Triangle::Lower),
            Err(GprError::IndexOutOfRange { .. })
        ));
    }

    #[test]
    fn apply_rejects_non_finite_dist() {
        let kernel = PeriodicKernel::new(1.0, 2.0).expect("valid");
        let dist = mat![[0.0, f64::NAN], [f64::NAN, 0.0]];
        let mut k = fill(2, 0.0);
        assert!(matches!(
            kernel.apply(dist.as_ref(), k.as_mut(), Triangle::Full),
            Err(GprError::NonFiniteInput)
        ));
    }
}
