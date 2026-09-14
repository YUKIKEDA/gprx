//! Isotropic squared-exponential (RBF) kernel.

use super::{Triangle, finite_dist, write_dense, write_triangle};
use crate::error::GprError;
use faer::{MatMut, MatRef};

/// Isotropic RBF: `k = exp( -‖x-x'‖² / (2ℓ²) )`.
///
/// The optimizer parameter is `θ = log(ℓ)`. Amplitude is not stored here;
/// compose with a constant kernel in P1A-6 when a signal variance is needed.
/// `dist` is the matrix of squared Euclidean distances.
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
    log_lengthscale: f64,
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
        Self::from_log_lengthscale(lengthscale.ln())
    }

    /// Builds an RBF kernel from `θ = log(ℓ)`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if `θ` is not finite, if
    /// `exp(θ)` overflows, or if `exp(θ)` underflows to zero.
    pub fn from_log_lengthscale(log_lengthscale: f64) -> Result<Self, GprError> {
        Ok(Self {
            log_lengthscale: validate_log_lengthscale(log_lengthscale)?,
        })
    }

    /// Returns `ℓ = exp(θ)`.
    pub fn lengthscale(&self) -> f64 {
        self.log_lengthscale.exp()
    }

    /// Returns `θ = log(ℓ)`.
    pub fn log_lengthscale(&self) -> f64 {
        self.log_lengthscale
    }

    /// Returns the number of optimizer parameters (always 1).
    pub fn num_params(&self) -> usize {
        1
    }

    /// Writes `θ` into a length-1 slice.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if `out` is not length 1.
    pub fn get_params(&self, out: &mut [f64]) -> Result<(), GprError> {
        expect_one_param(out.len())?;
        out[0] = self.log_lengthscale;
        Ok(())
    }

    /// Replaces `θ` from a length-1 slice.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if `params` is not length 1
    /// or if the new `θ` is invalid.
    pub fn set_params(&mut self, params: &[f64]) -> Result<(), GprError> {
        expect_one_param(params.len())?;
        self.log_lengthscale = validate_log_lengthscale(params[0])?;
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
        let inv_two_ell_sq = 0.5 / (self.lengthscale() * self.lengthscale());
        write_triangle(dist, out, uplo, |d| rbf_from_sq_dist(d, inv_two_ell_sq))
    }

    /// Writes rectangular `k(dist)` into `out` (train × test).
    ///
    /// # Errors
    ///
    /// Returns [`GprError`] if the matrices are empty, size mismatched, or if
    /// `dist` contains a non-finite value.
    pub fn apply_cross(&self, dist: MatRef<'_, f64>, out: MatMut<'_, f64>) -> Result<(), GprError> {
        let inv_two_ell_sq = 0.5 / (self.lengthscale() * self.lengthscale());
        write_dense(dist, out, |d| rbf_from_sq_dist(d, inv_two_ell_sq))
    }

    /// Writes the stationary diagonal `k(x, x) = 1` into `out`.
    pub fn fill_diag(&self, out: &mut [f64]) {
        out.fill(1.0);
    }

    /// Writes `∂K/∂θ` for `θ = log(ℓ)` into `d_k`.
    ///
    /// `∂k/∂θ = k · ‖x-x'‖² / ℓ²`. This is not `∂k/∂ℓ`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if `param_idx` is not 0, or
    /// the same shape / non-finite errors as [`Self::apply`].
    pub fn grad(
        &self,
        dist: MatRef<'_, f64>,
        d_k: MatMut<'_, f64>,
        param_idx: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        if param_idx != 0 {
            return Err(GprError::InvalidHyperparameter {
                reason: "RBF has a single parameter at index 0".to_owned(),
            });
        }
        let ell_sq = self.lengthscale() * self.lengthscale();
        let inv_two_ell_sq = 0.5 / ell_sq;
        let inv_ell_sq = 1.0 / ell_sq;
        write_triangle(dist, d_k, uplo, |d| {
            let d = finite_dist(d)?;
            let k = (-d * inv_two_ell_sq).exp();
            Ok(k * d * inv_ell_sq)
        })
    }
}

fn rbf_from_sq_dist(d: f64, inv_two_ell_sq: f64) -> Result<f64, GprError> {
    let d = finite_dist(d)?;
    Ok((-d * inv_two_ell_sq).exp())
}

fn invalid_length(reason: &'static str) -> GprError {
    GprError::InvalidHyperparameter {
        reason: reason.to_owned(),
    }
}

fn validate_lengthscale(lengthscale: f64) -> Result<(), GprError> {
    if !lengthscale.is_finite() {
        return Err(invalid_length("lengthscale must be finite"));
    }
    if lengthscale <= 0.0 {
        return Err(invalid_length("lengthscale must be positive"));
    }
    Ok(())
}

fn validate_log_lengthscale(theta: f64) -> Result<f64, GprError> {
    if !theta.is_finite() {
        return Err(invalid_length("log lengthscale must be finite"));
    }
    let lengthscale = theta.exp();
    if !lengthscale.is_finite() {
        return Err(invalid_length(
            "lengthscale overflowed to a non-finite value",
        ));
    }
    if lengthscale <= 0.0 {
        return Err(invalid_length("lengthscale underflowed to zero"));
    }
    Ok(theta)
}

fn expect_one_param(len: usize) -> Result<(), GprError> {
    if len == 1 {
        Ok(())
    } else {
        Err(GprError::InvalidHyperparameter {
            reason: format!("expected 1 RBF parameter, got {len}"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::RbfKernel;
    use crate::error::GprError;
    use crate::kernel::Triangle;
    use faer::{Mat, MatRef, mat};

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
        assert_close(k[(0, 0)], 1.0);
        assert_close(k[(1, 1)], 1.0);
        assert_close(k[(2, 2)], 1.0);
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
        assert_close(k[(0, 1)], (-0.5_f64).exp());
        assert_close(k[(0, 2)], (-1.0_f64).exp());
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
                assert_close(k[(row, col)], k[(col, row)]);
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
        lower_matches(lower.as_ref(), full.as_ref());
        assert_close(lower[(0, 1)], sentinel);
        assert_close(lower[(0, 2)], sentinel);
        assert_close(lower[(1, 2)], sentinel);
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
                assert_close(dk[(row, col)], fd);
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
        lower_matches(lower.as_ref(), full.as_ref());
        assert_close(lower[(0, 1)], 99.0);
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
                assert_close(upper[(row, col)], full[(row, col)]);
            }
        }
        assert_close(upper[(1, 0)], -1.0);
        assert_close(upper[(2, 0)], -1.0);
        assert_close(upper[(2, 1)], -1.0);
    }

    #[test]
    fn get_set_params_roundtrip() {
        let mut rbf = RbfKernel::new(2.0).expect("valid");
        let mut params = [0.0];
        rbf.get_params(&mut params).expect("len 1");
        assert_close(params[0], 2.0_f64.ln());
        params[0] = 0.5_f64.ln();
        rbf.set_params(&params).expect("len 1");
        assert_close(rbf.lengthscale(), 0.5);
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
            Err(GprError::InvalidHyperparameter { .. })
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
