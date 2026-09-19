//! Isotropic Matérn kernel for `ν = 1/2`, `3/2`, and `5/2`.

use super::lengthscale::{validate_lengthscale, validate_log_lengthscale};
use super::{
    Triangle, expect_one_param, finite_dist, write_dense, write_square_from_coords, write_triangle,
};
use crate::error::GprError;
use crate::param::{BoundedParam, Interval};
use faer::{MatMut, MatRef};

/// Smoothness `ν` for the closed-form Matérn kernels in Phase 1a.
///
/// `ν` is not an optimizer parameter. Amplitude is not stored on the leaf;
/// compose with [`super::ConstantKernel`] for a signal variance.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaternNu {
    /// `ν = 1/2` (exponential).
    Half,
    /// `ν = 3/2`.
    ThreeHalves,
    /// `ν = 5/2`.
    FiveHalves,
}

impl MaternNu {
    /// Returns `ν` as `f64`.
    pub fn value(self) -> f64 {
        match self {
            Self::Half => 0.5,
            Self::ThreeHalves => 1.5,
            Self::FiveHalves => 2.5,
        }
    }
}

/// Isotropic Matérn: `k` is a function of `r = ‖x-x'‖ / ℓ`.
///
/// The optimizer parameter is `θ = log(ℓ)`. `dist` is the matrix of squared
/// Euclidean distances. When every ARD lengthscale equals this `ℓ`, values
/// match [`super::MaternArdKernel`].
///
/// # Examples
///
/// ```rust
/// use gprx::kernel::{MaternKernel, MaternNu};
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let k = MaternKernel::new(1.5, MaternNu::ThreeHalves)?;
/// assert!(k.lengthscale() > 0.0);
/// assert_eq!(k.nu(), MaternNu::ThreeHalves);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaternKernel {
    nu: MaternNu,
    lengthscale: BoundedParam,
}

impl MaternKernel {
    /// Builds an isotropic Matérn kernel from a positive finite `ℓ` and `ν`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if `lengthscale` is not
    /// finite or not strictly positive.
    pub fn new(lengthscale: f64, nu: MaternNu) -> Result<Self, GprError> {
        validate_lengthscale(lengthscale)?;
        Ok(Self {
            nu,
            lengthscale: BoundedParam::default_positive(lengthscale)?,
        })
    }

    /// Builds an isotropic Matérn kernel from `θ = log(ℓ)` and `ν`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if `θ` is not finite, if
    /// `exp(θ)` overflows, or if `exp(θ)` underflows to zero.
    pub fn from_log_lengthscale(log_lengthscale: f64, nu: MaternNu) -> Result<Self, GprError> {
        Ok(Self {
            nu,
            lengthscale: BoundedParam::default_positive(
                validate_log_lengthscale(log_lengthscale)?.exp(),
            )?,
        })
    }

    /// Returns the smoothness `ν`.
    pub fn nu(&self) -> MaternNu {
        self.nu
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
            nu: self.nu,
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
    /// Returns [`GprError::InvalidHyperparameter`] if `out` is not length 1.
    pub fn get_params(&self, out: &mut [f64]) -> Result<(), GprError> {
        expect_one_param(out.len(), "Matern")?;
        out[0] = self.lengthscale.ln();
        Ok(())
    }

    /// Replaces `θ` from a length-1 slice. `ν` is unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if `params` is not length 1
    /// or if the new `θ` is invalid.
    pub fn set_params(&mut self, params: &[f64]) -> Result<(), GprError> {
        expect_one_param(params.len(), "Matern")?;
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
    pub fn apply(
        &self,
        dist: MatRef<'_, f64>,
        out: MatMut<'_, f64>,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        let ell = self.lengthscale();
        let nu = self.nu;
        write_triangle(dist, out, uplo, |d| matern_from_sq_dist(d, ell, nu))
    }

    /// Writes rectangular `k(dist)` into `out` (train × test).
    ///
    /// # Errors
    ///
    /// Returns [`GprError`] if the matrices are empty, size mismatched, or if
    /// `dist` contains a non-finite value.
    pub fn apply_cross(&self, dist: MatRef<'_, f64>, out: MatMut<'_, f64>) -> Result<(), GprError> {
        let ell = self.lengthscale();
        let nu = self.nu;
        write_dense(dist, out, |d| matern_from_sq_dist(d, ell, nu))
    }

    /// Writes the stationary diagonal `k(x, x) = 1` into `out`.
    pub fn fill_diag(&self, out: &mut [f64]) {
        out.fill(1.0);
    }

    /// Writes `∂K/∂θ` for `θ = log(ℓ)` into `d_k`.
    ///
    /// This is not `∂k/∂ℓ`.
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
                reason: "Matern has a single parameter at index 0".to_owned(),
            });
        }
        let ell = self.lengthscale();
        let nu = self.nu;
        write_triangle(dist, d_k, uplo, |d| {
            let r = scaled_distance(d, ell)?;
            finite_kernel(matern_dk_dtheta_iso(nu, r))
        })
    }

    pub(crate) fn apply_from_coords(
        &self,
        x: MatRef<'_, f64>,
        out: MatMut<'_, f64>,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        let ell = self.lengthscale();
        let nu = self.nu;
        write_square_from_coords(x, out, uplo, |d| matern_from_sq_dist(d, ell, nu))
    }

    pub(crate) fn grad_from_coords(
        &self,
        x: MatRef<'_, f64>,
        d_k: MatMut<'_, f64>,
        param_idx: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        if param_idx != 0 {
            return Err(GprError::InvalidHyperparameter {
                reason: "Matern has a single parameter at index 0".to_owned(),
            });
        }
        let ell = self.lengthscale();
        let nu = self.nu;
        write_square_from_coords(x, d_k, uplo, |d| {
            let r = scaled_distance(d, ell)?;
            finite_kernel(matern_dk_dtheta_iso(nu, r))
        })
    }
}

pub(crate) fn matern_from_r(nu: MaternNu, r: f64) -> f64 {
    match nu {
        MaternNu::Half => (-r).exp(),
        MaternNu::ThreeHalves => {
            let rho = 3.0_f64.sqrt() * r;
            (1.0 + rho) * (-rho).exp()
        }
        MaternNu::FiveHalves => {
            let rho = 5.0_f64.sqrt() * r;
            (1.0 + rho + rho * rho / 3.0) * (-rho).exp()
        }
    }
}

/// `∂k/∂θ` for isotropic `θ = log(ℓ)` at scaled distance `r = ‖x-x'‖ / ℓ`.
pub(crate) fn matern_dk_dtheta_iso(nu: MaternNu, r: f64) -> f64 {
    match nu {
        MaternNu::Half => matern_from_r(nu, r) * r,
        MaternNu::ThreeHalves => {
            let rho = 3.0_f64.sqrt() * r;
            rho * rho * (-rho).exp()
        }
        MaternNu::FiveHalves => {
            let rho = 5.0_f64.sqrt() * r;
            (rho * rho / 3.0) * (1.0 + rho) * (-rho).exp()
        }
    }
}

/// `∂k/∂θ_d` for ARD `θ_d = log(ℓ_d)`. `dim_term` is `(x_d-x'_d)² / ℓ_d²`.
pub(crate) fn matern_dk_dtheta_ard(nu: MaternNu, r: f64, dim_term: f64) -> f64 {
    match nu {
        MaternNu::Half => {
            if r <= 0.0 {
                0.0
            } else {
                matern_from_r(nu, r) * dim_term / r
            }
        }
        MaternNu::ThreeHalves => 3.0 * dim_term * (-(3.0_f64.sqrt() * r)).exp(),
        MaternNu::FiveHalves => {
            let rho = 5.0_f64.sqrt() * r;
            (5.0 / 3.0) * (1.0 + rho) * (-rho).exp() * dim_term
        }
    }
}

pub(crate) fn finite_kernel(value: f64) -> Result<f64, GprError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(GprError::NonFiniteKernelValue)
    }
}

fn matern_from_sq_dist(d: f64, ell: f64, nu: MaternNu) -> Result<f64, GprError> {
    let r = scaled_distance(d, ell)?;
    finite_kernel(matern_from_r(nu, r))
}

fn scaled_distance(sq_dist: f64, ell: f64) -> Result<f64, GprError> {
    let d = finite_dist(sq_dist)?;
    Ok(d.max(0.0).sqrt() / ell)
}

#[cfg(test)]
mod tests {
    use super::{MaternKernel, MaternNu};
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

    fn all_nu() -> [MaternNu; 3] {
        [MaternNu::Half, MaternNu::ThreeHalves, MaternNu::FiveHalves]
    }

    #[test]
    fn is_send_sync() {
        assert_send_sync::<MaternKernel>();
        assert_send_sync::<MaternNu>();
    }

    #[test]
    fn nu_values() {
        assert_close(MaternNu::Half.value(), 0.5);
        assert_close(MaternNu::ThreeHalves.value(), 1.5);
        assert_close(MaternNu::FiveHalves.value(), 2.5);
    }

    #[test]
    fn diagonal_is_one() {
        for nu in all_nu() {
            let kernel = MaternKernel::new(2.0, nu).expect("valid");
            let dist = sq_dist_1d(&[0.0, 1.0, 3.0]);
            let mut k = fill(3, f64::NAN);
            kernel
                .apply(dist.as_ref(), k.as_mut(), Triangle::Full)
                .expect("shape");
            assert_close(k[(0, 0)], 1.0);
            assert_close(k[(1, 1)], 1.0);
            assert_close(k[(2, 2)], 1.0);
        }
    }

    #[test]
    fn known_values_at_unit_scaled_distance() {
        let ell = 2.0;
        let dist = mat![[0.0, ell * ell], [ell * ell, 0.0]];
        let r = 1.0_f64;

        let half = MaternKernel::new(ell, MaternNu::Half).expect("valid");
        let mut k = fill(2, 0.0);
        half.apply(dist.as_ref(), k.as_mut(), Triangle::Full)
            .expect("shape");
        assert_close(k[(0, 1)], (-r).exp());

        let three = MaternKernel::new(ell, MaternNu::ThreeHalves).expect("valid");
        three
            .apply(dist.as_ref(), k.as_mut(), Triangle::Full)
            .expect("shape");
        let rho3 = 3.0_f64.sqrt() * r;
        assert_close(k[(0, 1)], (1.0 + rho3) * (-rho3).exp());

        let five = MaternKernel::new(ell, MaternNu::FiveHalves).expect("valid");
        five.apply(dist.as_ref(), k.as_mut(), Triangle::Full)
            .expect("shape");
        let rho5 = 5.0_f64.sqrt() * r;
        assert_close(k[(0, 1)], (1.0 + rho5 + rho5 * rho5 / 3.0) * (-rho5).exp());
    }

    #[test]
    fn full_is_symmetric() {
        let kernel = MaternKernel::new(1.25, MaternNu::FiveHalves).expect("valid");
        let dist = sq_dist_1d(&[0.0, 0.5, 2.0, 2.5]);
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
        let kernel = MaternKernel::new(0.75, MaternNu::ThreeHalves).expect("valid");
        let dist = sq_dist_1d(&[0.0, 1.0, 2.0]);
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
        assert_close(lower[(0, 2)], sentinel);
        assert_close(lower[(1, 2)], sentinel);
    }

    #[test]
    fn upper_matches_full() {
        let kernel = MaternKernel::new(1.0, MaternNu::Half).expect("valid");
        let dist = sq_dist_1d(&[0.0, 1.0, 2.0]);
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
        assert_close(upper[(2, 0)], -1.0);
        assert_close(upper[(2, 1)], -1.0);
    }

    #[test]
    fn grad_matches_finite_difference() {
        for nu in all_nu() {
            let kernel = MaternKernel::from_log_lengthscale(-0.3, nu).expect("valid");
            let theta = kernel.log_lengthscale();
            let h = 1e-6;
            let plus = MaternKernel::from_log_lengthscale(theta + h, nu).expect("valid");
            let minus = MaternKernel::from_log_lengthscale(theta - h, nu).expect("valid");
            let dist = sq_dist_1d(&[0.0, 1.2, 2.4]);
            let mut k_plus = fill(3, 0.0);
            let mut k_minus = fill(3, 0.0);
            let mut dk = fill(3, 0.0);
            plus.apply(dist.as_ref(), k_plus.as_mut(), Triangle::Full)
                .expect("shape");
            minus
                .apply(dist.as_ref(), k_minus.as_mut(), Triangle::Full)
                .expect("shape");
            kernel
                .grad(dist.as_ref(), dk.as_mut(), 0, Triangle::Full)
                .expect("index 0");
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
        let kernel = MaternKernel::new(1.0, MaternNu::FiveHalves).expect("valid");
        let dist = sq_dist_1d(&[0.0, 0.8, 1.6]);
        let mut full = fill(3, 0.0);
        let mut lower = fill(3, 99.0);
        kernel
            .grad(dist.as_ref(), full.as_mut(), 0, Triangle::Full)
            .expect("index 0");
        kernel
            .grad(dist.as_ref(), lower.as_mut(), 0, Triangle::Lower)
            .expect("index 0");
        lower_matches(lower.as_ref(), full.as_ref());
        assert_close(lower[(0, 1)], 99.0);
    }

    #[test]
    fn get_set_params_roundtrip() {
        let mut kernel = MaternKernel::new(2.0, MaternNu::Half).expect("valid");
        let mut params = [0.0];
        kernel.get_params(&mut params).expect("len 1");
        assert_close(params[0], 2.0_f64.ln());
        params[0] = 0.5_f64.ln();
        kernel.set_params(&params).expect("len 1");
        assert_close(kernel.lengthscale(), 0.5);
        assert_eq!(kernel.nu(), MaternNu::Half);
    }

    #[test]
    fn apply_cross_and_diag() {
        let kernel = MaternKernel::new(1.0, MaternNu::ThreeHalves).expect("valid");
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
    fn rejects_non_positive_lengthscale_and_bad_index() {
        assert!(matches!(
            MaternKernel::new(0.0, MaternNu::Half),
            Err(GprError::InvalidHyperparameter { .. })
        ));
        assert!(matches!(
            MaternKernel::from_log_lengthscale(f64::INFINITY, MaternNu::Half),
            Err(GprError::InvalidHyperparameter { .. })
        ));
        let kernel = MaternKernel::new(1.0, MaternNu::Half).expect("valid");
        let dist = sq_dist_1d(&[0.0, 1.0]);
        let mut dk = fill(2, 0.0);
        assert!(matches!(
            kernel.grad(dist.as_ref(), dk.as_mut(), 1, Triangle::Lower),
            Err(GprError::InvalidHyperparameter { .. })
        ));
    }

    #[test]
    fn apply_rejects_non_finite_dist() {
        let kernel = MaternKernel::new(1.0, MaternNu::Half).expect("valid");
        let dist = mat![[0.0, f64::NAN], [f64::NAN, 0.0]];
        let mut k = fill(2, 0.0);
        assert!(matches!(
            kernel.apply(dist.as_ref(), k.as_mut(), Triangle::Full),
            Err(GprError::NonFiniteInput)
        ));
    }
}
