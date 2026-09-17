//! Isotropic rational quadratic kernel.

use super::lengthscale::{validate_lengthscale, validate_log_lengthscale};
use super::{
    Triangle, finite_dist, validate_log_positive, validate_positive_finite, write_dense,
    write_triangle,
};
use crate::error::GprError;
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
    log_lengthscale: f64,
    log_alpha: f64,
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
        Self::from_log(lengthscale.ln(), alpha.ln())
    }

    /// Builds an isotropic RQ kernel from `θ = [log(ℓ), log(α)]`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if a `θ` is not finite, if
    /// `exp(θ)` overflows, or if `exp(θ)` underflows to zero.
    pub fn from_log(log_lengthscale: f64, log_alpha: f64) -> Result<Self, GprError> {
        Ok(Self {
            log_lengthscale: validate_log_lengthscale(log_lengthscale)?,
            log_alpha: validate_log_positive(log_alpha, "alpha")?,
        })
    }

    /// Returns `ℓ = exp(θ_0)`.
    pub fn lengthscale(&self) -> f64 {
        self.log_lengthscale.exp()
    }

    /// Returns `θ_0 = log(ℓ)`.
    pub fn log_lengthscale(&self) -> f64 {
        self.log_lengthscale
    }

    /// Returns `α = exp(θ_1)`.
    pub fn alpha(&self) -> f64 {
        self.log_alpha.exp()
    }

    /// Returns `θ_1 = log(α)`.
    pub fn log_alpha(&self) -> f64 {
        self.log_alpha
    }

    /// Returns the number of optimizer parameters (always 2).
    pub fn num_params(&self) -> usize {
        2
    }

    /// Writes `[log(ℓ), log(α)]` into a length-2 slice.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if `out` is not length 2.
    pub fn get_params(&self, out: &mut [f64]) -> Result<(), GprError> {
        expect_two_params(out.len())?;
        out[0] = self.log_lengthscale;
        out[1] = self.log_alpha;
        Ok(())
    }

    /// Replaces `[log(ℓ), log(α)]`. Previous values are kept on error.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if `params` is not length 2
    /// or a `θ` is invalid.
    pub fn set_params(&mut self, params: &[f64]) -> Result<(), GprError> {
        expect_two_params(params.len())?;
        *self = Self::from_log(params[0], params[1])?;
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
        let ell_sq = self.lengthscale() * self.lengthscale();
        let alpha = self.alpha();
        write_triangle(dist, out, uplo, |d| rq_from_sq_dist(d, ell_sq, alpha))
    }

    /// Writes rectangular `k(dist)` into `out` (train × test).
    ///
    /// # Errors
    ///
    /// Returns [`GprError`] if the matrices are empty, size mismatched, or if
    /// `dist` contains a non-finite value.
    pub fn apply_cross(&self, dist: MatRef<'_, f64>, out: MatMut<'_, f64>) -> Result<(), GprError> {
        let ell_sq = self.lengthscale() * self.lengthscale();
        let alpha = self.alpha();
        write_dense(dist, out, |d| rq_from_sq_dist(d, ell_sq, alpha))
    }

    /// Writes the stationary diagonal `k(x, x) = 1` into `out`.
    pub fn fill_diag(&self, out: &mut [f64]) {
        out.fill(1.0);
    }

    /// Writes `∂K/∂θ` into `d_k`. Index 0 is `log(ℓ)`, index 1 is `log(α)`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if `param_idx` is not 0 or
    /// 1, or the same shape / non-finite errors as [`Self::apply`].
    pub fn grad(
        &self,
        dist: MatRef<'_, f64>,
        d_k: MatMut<'_, f64>,
        param_idx: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        if param_idx > 1 {
            return Err(GprError::InvalidHyperparameter {
                reason: format!("rational quadratic parameter index {param_idx} is out of range"),
            });
        }
        let ell_sq = self.lengthscale() * self.lengthscale();
        let alpha = self.alpha();
        write_triangle(dist, d_k, uplo, |d| {
            let r2 = scaled_r2(d, ell_sq)?;
            finite_kernel(if param_idx == 0 {
                rq_dk_dtheta_lengthscale(r2, alpha)
            } else {
                rq_dk_dtheta_alpha(r2, alpha)
            })
        })
    }
}

pub(crate) fn rq_from_r2(r2: f64, alpha: f64) -> f64 {
    let u = 1.0 + r2 / (2.0 * alpha);
    u.powf(-alpha)
}

pub(crate) fn rq_dk_dtheta_lengthscale(r2: f64, alpha: f64) -> f64 {
    let u = 1.0 + r2 / (2.0 * alpha);
    let k = u.powf(-alpha);
    (k / u) * r2
}

pub(crate) fn rq_dk_dtheta_alpha(r2: f64, alpha: f64) -> f64 {
    let u = 1.0 + r2 / (2.0 * alpha);
    let k = u.powf(-alpha);
    alpha * k * (-u.ln() + 1.0 - 1.0 / u)
}

pub(crate) fn rq_dk_dtheta_ard_dim(r2: f64, alpha: f64, dim_term: f64) -> f64 {
    let u = 1.0 + r2 / (2.0 * alpha);
    let k = u.powf(-alpha);
    (k / u) * dim_term
}

pub(crate) fn finite_kernel(value: f64) -> Result<f64, GprError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(GprError::NonFiniteKernelValue)
    }
}

fn expect_two_params(len: usize) -> Result<(), GprError> {
    if len == 2 {
        Ok(())
    } else {
        Err(GprError::InvalidHyperparameter {
            reason: format!("expected 2 rational quadratic parameters, got {len}"),
        })
    }
}

fn scaled_r2(sq_dist: f64, ell_sq: f64) -> Result<f64, GprError> {
    let d = finite_dist(sq_dist)?;
    Ok(d.max(0.0) / ell_sq)
}

fn rq_from_sq_dist(sq_dist: f64, ell_sq: f64, alpha: f64) -> Result<f64, GprError> {
    let r2 = scaled_r2(sq_dist, ell_sq)?;
    finite_kernel(rq_from_r2(r2, alpha))
}

#[cfg(test)]
mod tests {
    use super::RationalQuadraticKernel;
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

    fn assert_send_sync<T: Send + Sync>() {}

    fn fill(n: usize, value: f64) -> Mat<f64> {
        Mat::from_fn(n, n, |_, _| value)
    }

    fn sq_dist_1d(x: &[f64]) -> Mat<f64> {
        let n = x.len();
        Mat::from_fn(n, n, |i, j| {
            let d = x[i] - x[j];
            d * d
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
        assert_close(k[(0, 0)], 1.0);
        assert_close(k[(1, 1)], 1.0);
        assert_close(k[(2, 2)], 1.0);
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
        assert_close(k[(0, 1)], 2.0_f64.powf(-alpha));
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
                assert_close(k[(row, col)], k[(col, row)]);
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
        lower_matches(lower.as_ref(), full.as_ref());
        assert_close(lower[(0, 1)], sentinel);
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
                assert_close(upper[(row, col)], full[(row, col)]);
            }
        }
        assert_close(upper[(1, 0)], -1.0);
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
                    assert_close(dk[(row, col)], fd);
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
        lower_matches(lower.as_ref(), full.as_ref());
        assert_close(lower[(0, 1)], 99.0);
    }

    #[test]
    fn get_set_params_roundtrip_and_atomic() {
        let mut kernel = RationalQuadraticKernel::new(2.0, 3.0).expect("valid");
        let mut params = [0.0; 2];
        kernel.get_params(&mut params).expect("len 2");
        assert_close(params[0], 2.0_f64.ln());
        assert_close(params[1], 3.0_f64.ln());
        params[0] = 0.5_f64.ln();
        kernel.set_params(&params).expect("len 2");
        assert_close(kernel.lengthscale(), 0.5);
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
        assert_close(out[(0, 1)], square[(0, 1)]);
        let mut diag = [0.0, 0.0];
        kernel.fill_diag(&mut diag);
        assert_close(diag[0], 1.0);
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
            Err(GprError::InvalidHyperparameter { .. })
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
