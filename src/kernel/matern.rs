//! Isotropic Matérn kernel for `ν = 1/2`, `3/2`, and `5/2`.

use super::lengthscale::{validate_lengthscale, validate_log_lengthscale};
use super::{Triangle, finite_dist, write_dense, write_square_from_coords, write_triangle};
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
        crate::data::require_count(out.len(), 1, "Matern parameter")?;
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
        crate::data::require_count(params.len(), 1, "Matern parameter")?;
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
        self.apply_math::<crate::math::Accurate>(dist, out, uplo)
    }

    pub(crate) fn apply_math<M: crate::math::KernelMath>(
        &self,
        dist: MatRef<'_, f64>,
        out: MatMut<'_, f64>,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        let ell = self.lengthscale();
        let nu = self.nu;
        write_triangle(dist, out, uplo, |d| matern_from_sq_dist::<M>(d, ell, nu))
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

    pub(crate) fn apply_cross_math<M: crate::math::KernelMath>(
        &self,
        dist: MatRef<'_, f64>,
        out: MatMut<'_, f64>,
    ) -> Result<(), GprError> {
        let ell = self.lengthscale();
        let nu = self.nu;
        write_dense(dist, out, |d| matern_from_sq_dist::<M>(d, ell, nu))
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
        self.grad_math::<crate::math::Accurate>(dist, d_k, param_idx, uplo)
    }

    pub(crate) fn grad_math<M: crate::math::KernelMath>(
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
            finite_kernel(matern_dk_dtheta_iso::<M>(nu, r))
        })
    }

    pub(crate) fn apply_from_coords<M: crate::math::KernelMath>(
        &self,
        x: MatRef<'_, f64>,
        out: MatMut<'_, f64>,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        let ell = self.lengthscale();
        let nu = self.nu;
        write_square_from_coords(x, out, uplo, |d| matern_from_sq_dist::<M>(d, ell, nu))
    }

    pub(crate) fn grad_from_coords<M: crate::math::KernelMath>(
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
            finite_kernel(matern_dk_dtheta_iso::<M>(nu, r))
        })
    }

    /// Writes `∂²K/∂θ²` for `θ = log(ℓ)` into `d2_k`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if `i` or `j` is not 0, or
    /// the same shape / non-finite errors as [`Self::apply`].
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

    pub(crate) fn hess_math<M: crate::math::KernelMath>(
        &self,
        dist: MatRef<'_, f64>,
        d2_k: MatMut<'_, f64>,
        i: usize,
        j: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        require_matern_hess_idx(i, j)?;
        let ell = self.lengthscale();
        let nu = self.nu;
        write_triangle(dist, d2_k, uplo, |d| {
            let r = scaled_distance(d, ell)?;
            finite_kernel(matern_d2k_dtheta2_iso::<M>(nu, r))
        })
    }

    pub(crate) fn hess_from_coords<M: crate::math::KernelMath>(
        &self,
        x: MatRef<'_, f64>,
        d2_k: MatMut<'_, f64>,
        i: usize,
        j: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        require_matern_hess_idx(i, j)?;
        let ell = self.lengthscale();
        let nu = self.nu;
        write_square_from_coords(x, d2_k, uplo, |d| {
            let r = scaled_distance(d, ell)?;
            finite_kernel(matern_d2k_dtheta2_iso::<M>(nu, r))
        })
    }

    /// Writes `∂K(X1, X2)/∂X2[*, dim]` into `d_k`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::CoordGradientUnsupported`] when `ν` is not `3/2`,
    /// or the same shape / non-finite errors as [`RbfKernel::grad_wrt_coord_dim`].
    pub fn grad_wrt_coord_dim(
        &self,
        x1: MatRef<'_, f64>,
        x2: MatRef<'_, f64>,
        d_k: MatMut<'_, f64>,
        dim: usize,
    ) -> Result<(), GprError> {
        self.grad_wrt_coord_dim_math::<crate::math::Accurate>(x1, x2, d_k, dim)
    }

    pub(crate) fn grad_wrt_coord_dim_math<M: crate::math::KernelMath>(
        &self,
        x1: MatRef<'_, f64>,
        x2: MatRef<'_, f64>,
        mut d_k: MatMut<'_, f64>,
        dim: usize,
    ) -> Result<(), GprError> {
        if self.nu != MaternNu::ThreeHalves {
            return Err(GprError::CoordGradientUnsupported);
        }
        super::require_coord_grad(x1, x2, d_k.as_ref(), dim)?;
        let inv_ell_sq = 1.0 / (self.lengthscale() * self.lengthscale());
        let scale = 3.0_f64.sqrt() / self.lengthscale();
        for col in 0..x2.nrows() {
            for row in 0..x1.nrows() {
                let (r, delta) = euclid_pair(x1, row, x2, col, dim)?;
                d_k[(row, col)] = if M::ACCURATE {
                    let psi = (-scale * r).exp();
                    3.0 * inv_ell_sq * psi * delta
                } else if r == 0.0 {
                    0.0
                } else {
                    let rho = scale * r;
                    let jet = M::jet(-rho);
                    let psi = (1.0 + rho) * jet.d1 - jet.v;
                    psi * scale * delta / r
                };
            }
        }
        Ok(())
    }

    pub(crate) fn hess_wrt_coord_dims<M: crate::math::KernelMath>(
        &self,
        x1: MatRef<'_, f64>,
        x2: MatRef<'_, f64>,
        mut d2_k: MatMut<'_, f64>,
        dim_a: usize,
        dim_b: usize,
    ) -> Result<(), GprError> {
        if self.nu != MaternNu::ThreeHalves {
            return Err(GprError::CoordGradientUnsupported);
        }
        super::require_coord_grad(x1, x2, d2_k.as_ref(), dim_a)?;
        super::require_coord_grad(x1, x2, d2_k.as_ref(), dim_b)?;
        let ell = self.lengthscale();
        let inv_ell_sq = 1.0 / (ell * ell);
        let scale = 3.0_f64.sqrt() / ell;
        for col in 0..x2.nrows() {
            for row in 0..x1.nrows() {
                let (r, da, db) = euclid_pair_two(x1, row, x2, col, dim_a, dim_b)?;
                let same = dim_a == dim_b;
                d2_k[(row, col)] = if M::ACCURATE {
                    let psi = (-scale * r).exp();
                    let mut value = if same { -3.0 * inv_ell_sq * psi } else { 0.0 };
                    if r > 0.0 {
                        value = 3.0 * inv_ell_sq * psi * (scale * da * db / r);
                        if same {
                            value -= 3.0 * inv_ell_sq * psi;
                        }
                    }
                    value
                } else {
                    matern_fast_coord_hess(scale, r, da, db, same, false)
                };
            }
        }
        Ok(())
    }

    pub(crate) fn hess_wrt_coord_mixed<M: crate::math::KernelMath>(
        &self,
        x1: MatRef<'_, f64>,
        x2: MatRef<'_, f64>,
        mut d2_k: MatMut<'_, f64>,
        dim_x1: usize,
        dim_x2: usize,
    ) -> Result<(), GprError> {
        if self.nu != MaternNu::ThreeHalves {
            return Err(GprError::CoordGradientUnsupported);
        }
        super::require_coord_grad(x1, x2, d2_k.as_ref(), dim_x1)?;
        super::require_coord_grad(x1, x2, d2_k.as_ref(), dim_x2)?;
        let ell = self.lengthscale();
        let inv_ell_sq = 1.0 / (ell * ell);
        let scale = 3.0_f64.sqrt() / ell;
        for col in 0..x2.nrows() {
            for row in 0..x1.nrows() {
                let (r, d1, d2) = euclid_pair_two(x1, row, x2, col, dim_x1, dim_x2)?;
                let same = dim_x1 == dim_x2;
                d2_k[(row, col)] = if M::ACCURATE {
                    let psi = (-scale * r).exp();
                    let mut value = if same { 3.0 * inv_ell_sq * psi } else { 0.0 };
                    if r > 0.0 {
                        value = 3.0 * inv_ell_sq * psi * (-scale * d1 * d2 / r);
                        if same {
                            value += 3.0 * inv_ell_sq * psi;
                        }
                    }
                    value
                } else {
                    matern_fast_coord_hess(scale, r, d1, d2, same, true)
                };
            }
        }
        Ok(())
    }

    pub(crate) fn hess_theta_coord_dim<M: crate::math::KernelMath>(
        &self,
        x1: MatRef<'_, f64>,
        x2: MatRef<'_, f64>,
        mut d2_k: MatMut<'_, f64>,
        param_idx: usize,
        dim: usize,
    ) -> Result<(), GprError> {
        if param_idx != 0 {
            return Err(GprError::InvalidHyperparameter {
                reason: "Matern has a single parameter at index 0".to_owned(),
            });
        }
        if self.nu != MaternNu::ThreeHalves {
            return Err(GprError::CoordGradientUnsupported);
        }
        super::require_coord_grad(x1, x2, d2_k.as_ref(), dim)?;
        let ell = self.lengthscale();
        let inv_ell_sq = 1.0 / (ell * ell);
        let scale = 3.0_f64.sqrt() / ell;
        for col in 0..x2.nrows() {
            for row in 0..x1.nrows() {
                let (r, delta) = euclid_pair(x1, row, x2, col, dim)?;
                d2_k[(row, col)] = if M::ACCURATE {
                    let psi = (-scale * r).exp();
                    3.0 * inv_ell_sq * psi * delta * (scale * r - 2.0)
                } else if r == 0.0 {
                    0.0
                } else {
                    let rho = scale * r;
                    let jet = M::jet(-rho);
                    let psi = (1.0 + rho) * jet.d1 - jet.v;
                    let dpsi = 2.0 * jet.d1 - (1.0 + rho) * jet.d2;
                    -scale * delta / r * (rho * dpsi + psi)
                };
            }
        }
        Ok(())
    }

    pub(crate) fn grad_cross_from_coords<M: crate::math::KernelMath>(
        &self,
        x1: MatRef<'_, f64>,
        x2: MatRef<'_, f64>,
        mut d_k: MatMut<'_, f64>,
        param_idx: usize,
    ) -> Result<(), GprError> {
        if param_idx != 0 {
            return Err(GprError::InvalidHyperparameter {
                reason: "Matern has a single parameter at index 0".to_owned(),
            });
        }
        super::require_coord_grad(x1, x2, d_k.as_ref(), 0)?;
        let ell = self.lengthscale();
        let nu = self.nu;
        for col in 0..x2.nrows() {
            for row in 0..x1.nrows() {
                let (r, _) = euclid_pair(x1, row, x2, col, 0)?;
                d_k[(row, col)] = finite_kernel(matern_dk_dtheta_iso::<M>(nu, r / ell))?;
            }
        }
        Ok(())
    }

    pub(crate) fn hess_cross_from_coords<M: crate::math::KernelMath>(
        &self,
        x1: MatRef<'_, f64>,
        x2: MatRef<'_, f64>,
        mut d2_k: MatMut<'_, f64>,
        i: usize,
        j: usize,
    ) -> Result<(), GprError> {
        require_matern_hess_idx(i, j)?;
        super::require_coord_grad(x1, x2, d2_k.as_ref(), 0)?;
        let ell = self.lengthscale();
        let nu = self.nu;
        for col in 0..x2.nrows() {
            for row in 0..x1.nrows() {
                let (r, _) = euclid_pair(x1, row, x2, col, 0)?;
                d2_k[(row, col)] = finite_kernel(matern_d2k_dtheta2_iso::<M>(nu, r / ell))?;
            }
        }
        Ok(())
    }
}

/// Second derivative of Matérn 3/2 w.r.t. coordinates, for [`crate::FastApprox`].
///
/// `from_x1` differentiates the `x2` gradient with respect to `x1`.
fn matern_fast_coord_hess(scale: f64, r: f64, da: f64, db: f64, same: bool, from_x1: bool) -> f64 {
    let jet0 = <crate::math::FastApprox as crate::math::KernelMath>::jet(0.0);
    let dpsi0 = 2.0 * jet0.d1 - jet0.d2;
    if r == 0.0 {
        return if same {
            let sign = if from_x1 { 1.0 } else { -1.0 };
            sign * scale * scale * dpsi0
        } else {
            0.0
        };
    }
    let rho = scale * r;
    let jet = <crate::math::FastApprox as crate::math::KernelMath>::jet(-rho);
    let psi = (1.0 + rho) * jet.d1 - jet.v;
    let dpsi = 2.0 * jet.d1 - (1.0 + rho) * jet.d2;
    let sign = if from_x1 { 1.0 } else { -1.0 };
    let same_term = if same {
        if from_x1 { psi / r } else { -psi / r }
    } else {
        0.0
    };
    scale
        * (dpsi * (sign * scale * da / r) * db / r
            + same_term
            + psi * (-sign * da * db) / (r * r * r))
}

fn euclid_pair(
    x1: MatRef<'_, f64>,
    i: usize,
    x2: MatRef<'_, f64>,
    j: usize,
    dim: usize,
) -> Result<(f64, f64), GprError> {
    let (r, da, _) = euclid_pair_two(x1, i, x2, j, dim, dim)?;
    Ok((r, da))
}

fn euclid_pair_two(
    x1: MatRef<'_, f64>,
    i: usize,
    x2: MatRef<'_, f64>,
    j: usize,
    dim_a: usize,
    dim_b: usize,
) -> Result<(f64, f64, f64), GprError> {
    let mut s = 0.0;
    for d in 0..x1.ncols() {
        let a = x1[(i, d)];
        let b = x2[(j, d)];
        if !a.is_finite() || !b.is_finite() {
            return Err(GprError::NonFiniteInput);
        }
        let delta = a - b;
        s += delta * delta;
    }
    Ok((
        s.max(0.0).sqrt(),
        x1[(i, dim_a)] - x2[(j, dim_a)],
        x1[(i, dim_b)] - x2[(j, dim_b)],
    ))
}

pub(crate) fn matern_from_r<M: crate::math::KernelMath>(nu: MaternNu, r: f64) -> f64 {
    match nu {
        MaternNu::Half => M::exp(-r),
        MaternNu::ThreeHalves => {
            let rho = 3.0_f64.sqrt() * r;
            (1.0 + rho) * M::exp(-rho)
        }
        MaternNu::FiveHalves => {
            let rho = 5.0_f64.sqrt() * r;
            (1.0 + rho + rho * rho / 3.0) * M::exp(-rho)
        }
    }
}

/// `∂k/∂θ` for isotropic `θ = log(ℓ)` at scaled distance `r = ‖x-x'‖ / ℓ`.
pub(crate) fn matern_dk_dtheta_iso<M: crate::math::KernelMath>(nu: MaternNu, r: f64) -> f64 {
    if M::ACCURATE {
        return match nu {
            MaternNu::Half => matern_from_r::<M>(nu, r) * r,
            MaternNu::ThreeHalves => {
                let rho = 3.0_f64.sqrt() * r;
                rho * rho * (-rho).exp()
            }
            MaternNu::FiveHalves => {
                let rho = 5.0_f64.sqrt() * r;
                (rho * rho / 3.0) * (1.0 + rho) * (-rho).exp()
            }
        };
    }
    match nu {
        MaternNu::Half => r * M::jet(-r).d1,
        MaternNu::ThreeHalves => {
            let rho = 3.0_f64.sqrt() * r;
            let jet = M::jet(-rho);
            rho * ((1.0 + rho) * jet.d1 - jet.v)
        }
        MaternNu::FiveHalves => {
            let rho = 5.0_f64.sqrt() * r;
            let jet = M::jet(-rho);
            let a = 1.0 + rho + rho * rho / 3.0;
            rho * (a * jet.d1 - (1.0 + 2.0 * rho / 3.0) * jet.v)
        }
    }
}

/// `∂²k/∂θ²` for isotropic `θ = log(ℓ)` at scaled distance `r = ‖x-x'‖ / ℓ`.
pub(crate) fn matern_d2k_dtheta2_iso<M: crate::math::KernelMath>(nu: MaternNu, r: f64) -> f64 {
    if M::ACCURATE {
        return match nu {
            MaternNu::Half => {
                let k = matern_from_r::<M>(nu, r);
                k * r * (r - 1.0)
            }
            MaternNu::ThreeHalves => {
                let rho = 3.0_f64.sqrt() * r;
                rho * rho * (rho - 2.0) * (-rho).exp()
            }
            MaternNu::FiveHalves => {
                let rho = 5.0_f64.sqrt() * r;
                (rho * rho / 3.0) * (rho * rho - 2.0 * rho - 2.0) * (-rho).exp()
            }
        };
    }
    match nu {
        MaternNu::Half => {
            let jet = M::jet(-r);
            r * r * jet.d2 - r * jet.d1
        }
        MaternNu::ThreeHalves => {
            let rho = 3.0_f64.sqrt() * r;
            let jet = M::jet(-rho);
            let u = (1.0 + rho) * jet.d1 - jet.v;
            rho * (-u + rho * ((1.0 + rho) * jet.d2 - 2.0 * jet.d1))
        }
        MaternNu::FiveHalves => {
            let rho = 5.0_f64.sqrt() * r;
            let jet = M::jet(-rho);
            let a = 1.0 + rho + rho * rho / 3.0;
            let ap = 1.0 + 2.0 * rho / 3.0;
            let app = 2.0 / 3.0;
            let dk_drho = ap * jet.v - a * jet.d1;
            let d2_drho = app * jet.v - 2.0 * ap * jet.d1 + a * jet.d2;
            rho * (dk_drho + rho * d2_drho)
        }
    }
}

/// `∂k/∂θ_d` for ARD `θ_d = log(ℓ_d)`. `dim_term` is `(x_d-x'_d)² / ℓ_d²`.
pub(crate) fn matern_dk_dtheta_ard<M: crate::math::KernelMath>(
    nu: MaternNu,
    r: f64,
    dim_term: f64,
) -> f64 {
    if M::ACCURATE {
        return match nu {
            MaternNu::Half => {
                if r <= 0.0 {
                    0.0
                } else {
                    matern_from_r::<M>(nu, r) * dim_term / r
                }
            }
            MaternNu::ThreeHalves => 3.0 * dim_term * (-(3.0_f64.sqrt() * r)).exp(),
            MaternNu::FiveHalves => {
                let rho = 5.0_f64.sqrt() * r;
                (5.0 / 3.0) * (1.0 + rho) * (-rho).exp() * dim_term
            }
        };
    }
    if r <= 0.0 {
        return 0.0;
    }
    match nu {
        MaternNu::Half => M::jet(-r).d1 * dim_term / r,
        MaternNu::ThreeHalves => {
            let rho = 3.0_f64.sqrt() * r;
            let jet = M::jet(-rho);
            ((1.0 + rho) * jet.d1 - jet.v) * 3.0_f64.sqrt() * dim_term / r
        }
        MaternNu::FiveHalves => {
            let rho = 5.0_f64.sqrt() * r;
            let jet = M::jet(-rho);
            let a = 1.0 + rho + rho * rho / 3.0;
            let ap = 1.0 + 2.0 * rho / 3.0;
            (a * jet.d1 - ap * jet.v) * 5.0_f64.sqrt() * dim_term / r
        }
    }
}

/// `∂²k/∂θ_d ∂θ_e` for ARD lengthscales. `same` is `d == e`.
pub(crate) fn matern_d2k_dtheta_ard<M: crate::math::KernelMath>(
    nu: MaternNu,
    r: f64,
    dim_i: f64,
    dim_j: f64,
    same: bool,
) -> f64 {
    if r <= 0.0 {
        return 0.0;
    }
    if M::ACCURATE {
        return match nu {
            MaternNu::Half => {
                let k = matern_from_r::<M>(nu, r);
                if same {
                    k * (dim_i * dim_i / (r * r) - 2.0 * dim_i / r + dim_i * dim_i / (r * r * r))
                } else {
                    k * dim_i * dim_j * (1.0 / (r * r) + 1.0 / (r * r * r))
                }
            }
            MaternNu::ThreeHalves => {
                let rho = 3.0_f64.sqrt() * r;
                let e = (-rho).exp();
                if same {
                    3.0 * e * (rho * dim_i * dim_i / (r * r) - 2.0 * dim_i)
                } else {
                    3.0 * e * (rho * dim_i * dim_j / (r * r))
                }
            }
            MaternNu::FiveHalves => {
                let rho = 5.0_f64.sqrt() * r;
                let e = (-rho).exp();
                if same {
                    (5.0 / 3.0)
                        * e
                        * (rho * rho * dim_i * dim_i / (r * r) - 2.0 * (1.0 + rho) * dim_i)
                } else {
                    (5.0 / 3.0) * e * (rho * rho * dim_i * dim_j / (r * r))
                }
            }
        };
    }
    match nu {
        MaternNu::Half => {
            let jet = M::jet(-r);
            let rr = r * r;
            if same {
                jet.d2 * dim_i * dim_i / rr + jet.d1 * (-2.0 * dim_i / r + dim_i * dim_i / (rr * r))
            } else {
                jet.d2 * dim_i * dim_j / rr + jet.d1 * dim_i * dim_j / (rr * r)
            }
        }
        MaternNu::ThreeHalves => {
            let rho = 3.0_f64.sqrt() * r;
            let jet = M::jet(-rho);
            let phi_p = jet.v - (1.0 + rho) * jet.d1;
            let phi_pp = (1.0 + rho) * jet.d2 - 2.0 * jet.d1;
            matern_ard_hess_from_phi(phi_p, phi_pp, r, dim_i, dim_j, same, 3.0_f64.sqrt())
        }
        MaternNu::FiveHalves => {
            let rho = 5.0_f64.sqrt() * r;
            let jet = M::jet(-rho);
            let a = 1.0 + rho + rho * rho / 3.0;
            let ap = 1.0 + 2.0 * rho / 3.0;
            let app = 2.0 / 3.0;
            let phi_p = ap * jet.v - a * jet.d1;
            let phi_pp = app * jet.v - 2.0 * ap * jet.d1 + a * jet.d2;
            matern_ard_hess_from_phi(phi_p, phi_pp, r, dim_i, dim_j, same, 5.0_f64.sqrt())
        }
    }
}

/// `∂²k/∂θ_i ∂θ_j` from `φ'(ρ)` and `φ''(ρ)`, with `ρ = scale · r`.
fn matern_ard_hess_from_phi(
    phi_p: f64,
    phi_pp: f64,
    r: f64,
    dim_i: f64,
    dim_j: f64,
    same: bool,
    scale: f64,
) -> f64 {
    let rr = r * r;
    if same {
        let g2 = scale * scale * dim_i * dim_i / rr;
        let dg = -scale * (-2.0 * dim_i / r + dim_i * dim_i / (rr * r));
        phi_pp * g2 + phi_p * dg
    } else {
        let g_ij = scale * scale * dim_i * dim_j / rr;
        let dg = -scale * dim_i * dim_j / (rr * r);
        phi_pp * g_ij + phi_p * dg
    }
}

fn require_matern_hess_idx(i: usize, j: usize) -> Result<(), GprError> {
    if i == 0 && j == 0 {
        Ok(())
    } else {
        Err(GprError::InvalidHyperparameter {
            reason: format!("Matern has a single parameter; got pair ({i}, {j})"),
        })
    }
}

pub(crate) fn finite_kernel(value: f64) -> Result<f64, GprError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(GprError::NonFiniteKernelValue)
    }
}

fn matern_from_sq_dist<M: crate::math::KernelMath>(
    d: f64,
    ell: f64,
    nu: MaternNu,
) -> Result<f64, GprError> {
    let r = scaled_distance(d, ell)?;
    finite_kernel(matern_from_r::<M>(nu, r))
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
    fn hess_matches_finite_difference_of_grad() {
        for nu in all_nu() {
            let kernel = MaternKernel::from_log_lengthscale(-0.2, nu).expect("valid");
            let theta = kernel.log_lengthscale();
            let h = 1e-6;
            let plus = MaternKernel::from_log_lengthscale(theta + h, nu).expect("valid");
            let minus = MaternKernel::from_log_lengthscale(theta - h, nu).expect("valid");
            let dist = sq_dist_1d(&[0.0, 1.1, 2.3]);
            let mut g_plus = fill(3, 0.0);
            let mut g_minus = fill(3, 0.0);
            let mut d2 = fill(3, 0.0);
            plus.grad(dist.as_ref(), g_plus.as_mut(), 0, Triangle::Full)
                .expect("shape");
            minus
                .grad(dist.as_ref(), g_minus.as_mut(), 0, Triangle::Full)
                .expect("shape");
            kernel
                .hess(dist.as_ref(), d2.as_mut(), 0, 0, Triangle::Full)
                .expect("index 0");
            for col in 0..3 {
                for row in 0..3 {
                    let fd = (g_plus[(row, col)] - g_minus[(row, col)]) / (2.0 * h);
                    assert_close(d2[(row, col)], fd);
                }
            }
        }
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
