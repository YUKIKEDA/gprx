//! Isotropic Matérn kernel for `ν = 1/2`, `3/2`, and `5/2`.

use super::lengthscale::{validate_lengthscale, validate_log_lengthscale};
use super::{
    Triangle, finite_dist, finite_kernel, write_dense, write_rect_from_coords,
    write_square_from_coords, write_triangle,
};
use crate::error::GprError;
use crate::kernel::KernelScalar;
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
    /// Returns [`GprError::LengthMismatch`] if `out` is not length 1.
    pub fn get_params(&self, out: &mut [f64]) -> Result<(), GprError> {
        crate::data::require_count(out.len(), 1, "Matern parameter")?;
        out[0] = self.lengthscale.ln();
        Ok(())
    }

    /// Replaces `θ` from a length-1 slice. `ν` is unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::LengthMismatch`] if `params` is not length 1,
    /// or [`GprError::InvalidHyperparameter`] if the new `θ` is invalid.
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
    pub fn apply<T: KernelScalar>(
        &self,
        dist: MatRef<'_, T>,
        out: MatMut<'_, T>,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        self.apply_math::<crate::math::Accurate, _>(dist, out, uplo)
    }

    pub(crate) fn apply_math<M: crate::math::KernelMath, T: KernelScalar>(
        &self,
        dist: MatRef<'_, T>,
        out: MatMut<'_, T>,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        let ell = T::from_f64(self.lengthscale());
        let nu = self.nu;
        write_triangle(dist, out, uplo, |d| matern_from_sq_dist::<M, _>(d, ell, nu))
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
        self.apply_cross_math::<crate::math::Accurate, _>(dist, out)
    }

    pub(crate) fn apply_cross_math<M: crate::math::KernelMath, T: KernelScalar>(
        &self,
        dist: MatRef<'_, T>,
        out: MatMut<'_, T>,
    ) -> Result<(), GprError> {
        let ell = T::from_f64(self.lengthscale());
        let nu = self.nu;
        write_dense(dist, out, |d| matern_from_sq_dist::<M, _>(d, ell, nu))
    }

    /// Writes the stationary diagonal `k(x, x) = 1` into `out`.
    pub fn fill_diag<T: KernelScalar>(&self, out: &mut [T]) {
        out.fill(T::from_f64(1.0));
    }

    /// Writes `∂K/∂θ` for `θ = log(ℓ)` into `d_k`.
    ///
    /// This is not `∂k/∂ℓ`.
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
        self.grad_math::<crate::math::Accurate, _>(dist, d_k, param_idx, uplo)
    }

    pub(crate) fn grad_math<M: crate::math::KernelMath, T: KernelScalar>(
        &self,
        dist: MatRef<'_, T>,
        d_k: MatMut<'_, T>,
        param_idx: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        if param_idx != 0 {
            return Err(GprError::IndexOutOfRange {
                reason: "Matern has a single parameter at index 0".to_owned(),
            });
        }
        let ell = T::from_f64(self.lengthscale());
        let nu = self.nu;
        write_triangle(dist, d_k, uplo, |d| {
            let r = scaled_distance(d, ell)?;
            finite_kernel(matern_dk_dtheta_iso::<M, _>(nu, r))
        })
    }

    pub(crate) fn apply_from_coords<M: crate::math::KernelMath, T: KernelScalar>(
        &self,
        x: MatRef<'_, T>,
        out: MatMut<'_, T>,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        let ell = T::from_f64(self.lengthscale());
        let nu = self.nu;
        write_square_from_coords(x, out, uplo, |d| matern_from_sq_dist::<M, _>(d, ell, nu))
    }

    /// Rectangular `K(x1, x2)` from coordinates.
    pub(crate) fn apply_cross_from_coords<M: crate::math::KernelMath, T: KernelScalar>(
        &self,
        x1: MatRef<'_, T>,
        x2: MatRef<'_, T>,
        out: MatMut<'_, T>,
    ) -> Result<(), GprError> {
        let ell = T::from_f64(self.lengthscale());
        let nu = self.nu;
        write_rect_from_coords(x1, x2, out, |d| matern_from_sq_dist::<M, _>(d, ell, nu))
    }

    pub(crate) fn grad_from_coords<M: crate::math::KernelMath, T: KernelScalar>(
        &self,
        x: MatRef<'_, T>,
        d_k: MatMut<'_, T>,
        param_idx: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        if param_idx != 0 {
            return Err(GprError::IndexOutOfRange {
                reason: "Matern has a single parameter at index 0".to_owned(),
            });
        }
        let ell = T::from_f64(self.lengthscale());
        let nu = self.nu;
        write_square_from_coords(x, d_k, uplo, |d| {
            let r = scaled_distance(d, ell)?;
            finite_kernel(matern_dk_dtheta_iso::<M, _>(nu, r))
        })
    }

    /// Writes `∂²K/∂θ²` for `θ = log(ℓ)` into `d2_k`.
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
        self.hess_math::<crate::math::Accurate, _>(dist, d2_k, i, j, uplo)
    }

    pub(crate) fn hess_math<M: crate::math::KernelMath, T: KernelScalar>(
        &self,
        dist: MatRef<'_, T>,
        d2_k: MatMut<'_, T>,
        i: usize,
        j: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        require_matern_hess_idx(i, j)?;
        let ell = T::from_f64(self.lengthscale());
        let nu = self.nu;
        write_triangle(dist, d2_k, uplo, |d| {
            let r = scaled_distance(d, ell)?;
            finite_kernel(matern_d2k_dtheta2_iso::<M, _>(nu, r))
        })
    }

    pub(crate) fn hess_from_coords<M: crate::math::KernelMath, T: KernelScalar>(
        &self,
        x: MatRef<'_, T>,
        d2_k: MatMut<'_, T>,
        i: usize,
        j: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        require_matern_hess_idx(i, j)?;
        let ell = T::from_f64(self.lengthscale());
        let nu = self.nu;
        write_square_from_coords(x, d2_k, uplo, |d| {
            let r = scaled_distance(d, ell)?;
            finite_kernel(matern_d2k_dtheta2_iso::<M, _>(nu, r))
        })
    }

    /// Writes `∂K(X1, X2)/∂X2[*, dim]` into `d_k`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::CoordGradientUnsupported`] when `ν` is `1/2`,
    /// or the same shape / non-finite errors as [`RbfKernel::grad_wrt_coord_dim`](super::RbfKernel::grad_wrt_coord_dim).
    pub fn grad_wrt_coord_dim<T: KernelScalar>(
        &self,
        x1: MatRef<'_, T>,
        x2: MatRef<'_, T>,
        d_k: MatMut<'_, T>,
        dim: usize,
    ) -> Result<(), GprError> {
        super::radial::grad_wrt_coord_dim::<crate::math::Accurate, _>(self, x1, x2, d_k, dim)
    }

    pub(crate) fn grad_cross_from_coords<M: crate::math::KernelMath, T: KernelScalar>(
        &self,
        x1: MatRef<'_, T>,
        x2: MatRef<'_, T>,
        d_k: MatMut<'_, T>,
        param_idx: usize,
    ) -> Result<(), GprError> {
        if param_idx != 0 {
            return Err(GprError::IndexOutOfRange {
                reason: "Matern has a single parameter at index 0".to_owned(),
            });
        }
        super::require_coord_grad(x1, x2, d_k.as_ref(), 0)?;
        let ell = T::from_f64(self.lengthscale());
        let nu = self.nu;
        super::write_rect(d_k, |row, col| {
            let (r, _) = euclid_pair(x1, row, x2, col, 0)?;
            finite_kernel(matern_dk_dtheta_iso::<M, _>(nu, r / ell))
        })
    }

    pub(crate) fn hess_cross_from_coords<M: crate::math::KernelMath, T: KernelScalar>(
        &self,
        x1: MatRef<'_, T>,
        x2: MatRef<'_, T>,
        d2_k: MatMut<'_, T>,
        i: usize,
        j: usize,
    ) -> Result<(), GprError> {
        require_matern_hess_idx(i, j)?;
        super::require_coord_grad(x1, x2, d2_k.as_ref(), 0)?;
        let ell = T::from_f64(self.lengthscale());
        let nu = self.nu;
        super::write_rect(d2_k, |row, col| {
            let (r, _) = euclid_pair(x1, row, x2, col, 0)?;
            finite_kernel(matern_d2k_dtheta2_iso::<M, _>(nu, r / ell))
        })
    }
}

fn euclid_pair<T: KernelScalar>(
    x1: MatRef<'_, T>,
    i: usize,
    x2: MatRef<'_, T>,
    j: usize,
    dim: usize,
) -> Result<(T, T), GprError> {
    let (r, da, _) = euclid_pair_two(x1, i, x2, j, dim, dim)?;
    Ok((r, da))
}

fn euclid_pair_two<T: KernelScalar>(
    x1: MatRef<'_, T>,
    i: usize,
    x2: MatRef<'_, T>,
    j: usize,
    dim_a: usize,
    dim_b: usize,
) -> Result<(T, T, T), GprError> {
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
    Ok((
        s.max(T::from_f64(0.0)).sqrt(),
        x1[(i, dim_a)] - x2[(j, dim_a)],
        x1[(i, dim_b)] - x2[(j, dim_b)],
    ))
}

pub(crate) fn matern_from_r<M: crate::math::KernelMath, T: KernelScalar>(nu: MaternNu, r: T) -> T {
    match nu {
        MaternNu::Half => M::exp(-r),
        MaternNu::ThreeHalves => {
            let rho = T::from_f64(3.0_f64.sqrt()) * r;
            (T::from_f64(1.0) + rho) * M::exp(-rho)
        }
        MaternNu::FiveHalves => {
            let rho = T::from_f64(5.0_f64.sqrt()) * r;
            (T::from_f64(1.0) + rho + rho * rho / T::from_f64(3.0)) * M::exp(-rho)
        }
    }
}

/// `∂k/∂θ` for isotropic `θ = log(ℓ)` at scaled distance `r = ‖x-x'‖ / ℓ`.
pub(crate) fn matern_dk_dtheta_iso<M: crate::math::KernelMath, T: KernelScalar>(
    nu: MaternNu,
    r: T,
) -> T {
    if M::ACCURATE {
        return match nu {
            MaternNu::Half => matern_from_r::<M, _>(nu, r) * r,
            MaternNu::ThreeHalves => {
                let rho = T::from_f64(3.0_f64.sqrt()) * r;
                rho * rho * (-rho).exp()
            }
            MaternNu::FiveHalves => {
                let rho = T::from_f64(5.0_f64.sqrt()) * r;
                (rho * rho / T::from_f64(3.0)) * (T::from_f64(1.0) + rho) * (-rho).exp()
            }
        };
    }
    match nu {
        MaternNu::Half => r * M::jet(-r).d1,
        MaternNu::ThreeHalves => {
            let rho = T::from_f64(3.0_f64.sqrt()) * r;
            let jet = M::jet(-rho);
            rho * ((T::from_f64(1.0) + rho) * jet.d1 - jet.v)
        }
        MaternNu::FiveHalves => {
            let rho = T::from_f64(5.0_f64.sqrt()) * r;
            let jet = M::jet(-rho);
            let a = T::from_f64(1.0) + rho + rho * rho / T::from_f64(3.0);
            rho * (a * jet.d1
                - (T::from_f64(1.0) + T::from_f64(2.0) * rho / T::from_f64(3.0)) * jet.v)
        }
    }
}

/// `∂²k/∂θ²` for isotropic `θ = log(ℓ)` at scaled distance `r = ‖x-x'‖ / ℓ`.
pub(crate) fn matern_d2k_dtheta2_iso<M: crate::math::KernelMath, T: KernelScalar>(
    nu: MaternNu,
    r: T,
) -> T {
    if M::ACCURATE {
        return match nu {
            MaternNu::Half => {
                let k = matern_from_r::<M, _>(nu, r);
                k * r * (r - T::from_f64(1.0))
            }
            MaternNu::ThreeHalves => {
                let rho = T::from_f64(3.0_f64.sqrt()) * r;
                rho * rho * (rho - T::from_f64(2.0)) * (-rho).exp()
            }
            MaternNu::FiveHalves => {
                let rho = T::from_f64(5.0_f64.sqrt()) * r;
                (rho * rho / T::from_f64(3.0))
                    * (rho * rho - T::from_f64(2.0) * rho - T::from_f64(2.0))
                    * (-rho).exp()
            }
        };
    }
    match nu {
        MaternNu::Half => {
            let jet = M::jet(-r);
            r * r * jet.d2 - r * jet.d1
        }
        MaternNu::ThreeHalves => {
            let rho = T::from_f64(3.0_f64.sqrt()) * r;
            let jet = M::jet(-rho);
            let u = (T::from_f64(1.0) + rho) * jet.d1 - jet.v;
            rho * (-u + rho * ((T::from_f64(1.0) + rho) * jet.d2 - T::from_f64(2.0) * jet.d1))
        }
        MaternNu::FiveHalves => {
            let rho = T::from_f64(5.0_f64.sqrt()) * r;
            let jet = M::jet(-rho);
            let a = T::from_f64(1.0) + rho + rho * rho / T::from_f64(3.0);
            let ap = T::from_f64(1.0) + T::from_f64(2.0) * rho / T::from_f64(3.0);
            let app = T::from_f64(2.0) / T::from_f64(3.0);
            let dk_drho = ap * jet.v - a * jet.d1;
            let d2_drho = app * jet.v - T::from_f64(2.0) * ap * jet.d1 + a * jet.d2;
            rho * (dk_drho + rho * d2_drho)
        }
    }
}

/// `∂k/∂θ_d` for ARD `θ_d = log(ℓ_d)`. `dim_term` is `(x_d-x'_d)² / ℓ_d²`.
pub(crate) fn matern_dk_dtheta_ard<M: crate::math::KernelMath, T: KernelScalar>(
    nu: MaternNu,
    r: T,
    dim_term: T,
) -> T {
    if M::ACCURATE {
        return match nu {
            MaternNu::Half => {
                if r <= T::from_f64(0.0) {
                    T::from_f64(0.0)
                } else {
                    matern_from_r::<M, _>(nu, r) * dim_term / r
                }
            }
            MaternNu::ThreeHalves => {
                T::from_f64(3.0) * dim_term * (-(T::from_f64(3.0_f64.sqrt()) * r)).exp()
            }
            MaternNu::FiveHalves => {
                let rho = T::from_f64(5.0_f64.sqrt()) * r;
                (T::from_f64(5.0) / T::from_f64(3.0))
                    * (T::from_f64(1.0) + rho)
                    * (-rho).exp()
                    * dim_term
            }
        };
    }
    if r <= T::from_f64(0.0) {
        return T::from_f64(0.0);
    }
    match nu {
        MaternNu::Half => M::jet(-r).d1 * dim_term / r,
        MaternNu::ThreeHalves => {
            let rho = T::from_f64(3.0_f64.sqrt()) * r;
            let jet = M::jet(-rho);
            ((T::from_f64(1.0) + rho) * jet.d1 - jet.v) * T::from_f64(3.0_f64.sqrt()) * dim_term / r
        }
        MaternNu::FiveHalves => {
            let rho = T::from_f64(5.0_f64.sqrt()) * r;
            let jet = M::jet(-rho);
            let a = T::from_f64(1.0) + rho + rho * rho / T::from_f64(3.0);
            let ap = T::from_f64(1.0) + T::from_f64(2.0) * rho / T::from_f64(3.0);
            (a * jet.d1 - ap * jet.v) * T::from_f64(5.0_f64.sqrt()) * dim_term / r
        }
    }
}

/// `∂²k/∂θ_d ∂θ_e` for ARD lengthscales. `same` is `d == e`.
pub(crate) fn matern_d2k_dtheta_ard<M: crate::math::KernelMath, T: KernelScalar>(
    nu: MaternNu,
    r: T,
    dim_i: T,
    dim_j: T,
    same: bool,
) -> T {
    if r <= T::from_f64(0.0) {
        return T::from_f64(0.0);
    }
    if M::ACCURATE {
        return match nu {
            MaternNu::Half => {
                let k = matern_from_r::<M, _>(nu, r);
                if same {
                    k * (dim_i * dim_i / (r * r) - T::from_f64(2.0) * dim_i / r
                        + dim_i * dim_i / (r * r * r))
                } else {
                    k * dim_i
                        * dim_j
                        * (T::from_f64(1.0) / (r * r) + T::from_f64(1.0) / (r * r * r))
                }
            }
            MaternNu::ThreeHalves => {
                let rho = T::from_f64(3.0_f64.sqrt()) * r;
                let e = (-rho).exp();
                if same {
                    T::from_f64(3.0)
                        * e
                        * (rho * dim_i * dim_i / (r * r) - T::from_f64(2.0) * dim_i)
                } else {
                    T::from_f64(3.0) * e * (rho * dim_i * dim_j / (r * r))
                }
            }
            MaternNu::FiveHalves => {
                let rho = T::from_f64(5.0_f64.sqrt()) * r;
                let e = (-rho).exp();
                if same {
                    (T::from_f64(5.0) / T::from_f64(3.0))
                        * e
                        * (rho * rho * dim_i * dim_i / (r * r)
                            - T::from_f64(2.0) * (T::from_f64(1.0) + rho) * dim_i)
                } else {
                    (T::from_f64(5.0) / T::from_f64(3.0))
                        * e
                        * (rho * rho * dim_i * dim_j / (r * r))
                }
            }
        };
    }
    match nu {
        MaternNu::Half => {
            let jet = M::jet(-r);
            let rr = r * r;
            if same {
                jet.d2 * dim_i * dim_i / rr
                    + jet.d1 * (-T::from_f64(2.0) * dim_i / r + dim_i * dim_i / (rr * r))
            } else {
                jet.d2 * dim_i * dim_j / rr + jet.d1 * dim_i * dim_j / (rr * r)
            }
        }
        MaternNu::ThreeHalves => {
            let rho = T::from_f64(3.0_f64.sqrt()) * r;
            let jet = M::jet(-rho);
            let phi_p = jet.v - (T::from_f64(1.0) + rho) * jet.d1;
            let phi_pp = (T::from_f64(1.0) + rho) * jet.d2 - T::from_f64(2.0) * jet.d1;
            matern_ard_hess_from_phi(
                phi_p,
                phi_pp,
                r,
                dim_i,
                dim_j,
                same,
                T::from_f64(3.0_f64.sqrt()),
            )
        }
        MaternNu::FiveHalves => {
            let rho = T::from_f64(5.0_f64.sqrt()) * r;
            let jet = M::jet(-rho);
            let a = T::from_f64(1.0) + rho + rho * rho / T::from_f64(3.0);
            let ap = T::from_f64(1.0) + T::from_f64(2.0) * rho / T::from_f64(3.0);
            let app = T::from_f64(2.0) / T::from_f64(3.0);
            let phi_p = ap * jet.v - a * jet.d1;
            let phi_pp = app * jet.v - T::from_f64(2.0) * ap * jet.d1 + a * jet.d2;
            matern_ard_hess_from_phi(
                phi_p,
                phi_pp,
                r,
                dim_i,
                dim_j,
                same,
                T::from_f64(5.0_f64.sqrt()),
            )
        }
    }
}

/// `∂²k/∂θ_i ∂θ_j` from `φ'(ρ)` and `φ''(ρ)`, with `ρ = scale · r`.
fn matern_ard_hess_from_phi<T: KernelScalar>(
    phi_p: T,
    phi_pp: T,
    r: T,
    dim_i: T,
    dim_j: T,
    same: bool,
    scale: T,
) -> T {
    let rr = r * r;
    if same {
        let g2 = scale * scale * dim_i * dim_i / rr;
        let dg = -scale * (-T::from_f64(2.0) * dim_i / r + dim_i * dim_i / (rr * r));
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
        Err(GprError::IndexOutOfRange {
            reason: format!("Matern has a single parameter; got pair ({i}, {j})"),
        })
    }
}

fn matern_from_sq_dist<M: crate::math::KernelMath, T: KernelScalar>(
    d: T,
    ell: T,
    nu: MaternNu,
) -> Result<T, GprError> {
    let r = scaled_distance(d, ell)?;
    finite_kernel(matern_from_r::<M, _>(nu, r))
}

fn scaled_distance<T: KernelScalar>(sq_dist: T, ell: T) -> Result<T, GprError> {
    let d = finite_dist(sq_dist)?;
    Ok(d.max(T::from_f64(0.0)).sqrt() / ell)
}

#[cfg(test)]
mod tests {
    use super::{MaternKernel, MaternNu};
    use crate::error::GprError;
    use crate::kernel::Triangle;
    use faer::{Mat, mat};

    const TOL: f64 = 1e-8;

    use crate::test_check::{assert_close, assert_lower_close, assert_send_sync, fill, sq_dist_1d};

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
        assert_close(MaternNu::Half.value(), 0.5, TOL);
        assert_close(MaternNu::ThreeHalves.value(), 1.5, TOL);
        assert_close(MaternNu::FiveHalves.value(), 2.5, TOL);
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
            assert_close(k[(0, 0)], 1.0, TOL);
            assert_close(k[(1, 1)], 1.0, TOL);
            assert_close(k[(2, 2)], 1.0, TOL);
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
        assert_close(k[(0, 1)], (-r).exp(), TOL);

        let three = MaternKernel::new(ell, MaternNu::ThreeHalves).expect("valid");
        three
            .apply(dist.as_ref(), k.as_mut(), Triangle::Full)
            .expect("shape");
        let rho3 = 3.0_f64.sqrt() * r;
        assert_close(k[(0, 1)], (1.0 + rho3) * (-rho3).exp(), TOL);

        let five = MaternKernel::new(ell, MaternNu::FiveHalves).expect("valid");
        five.apply(dist.as_ref(), k.as_mut(), Triangle::Full)
            .expect("shape");
        let rho5 = 5.0_f64.sqrt() * r;
        assert_close(
            k[(0, 1)],
            (1.0 + rho5 + rho5 * rho5 / 3.0) * (-rho5).exp(),
            TOL,
        );
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
                assert_close(k[(row, col)], k[(col, row)], TOL);
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
        assert_lower_close(lower.as_ref(), full.as_ref(), TOL);
        assert_close(lower[(0, 1)], sentinel, TOL);
        assert_close(lower[(0, 2)], sentinel, TOL);
        assert_close(lower[(1, 2)], sentinel, TOL);
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
                assert_close(upper[(row, col)], full[(row, col)], TOL);
            }
        }
        assert_close(upper[(1, 0)], -1.0, TOL);
        assert_close(upper[(2, 0)], -1.0, TOL);
        assert_close(upper[(2, 1)], -1.0, TOL);
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
                    assert_close(d2[(row, col)], fd, TOL);
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
                    assert_close(dk[(row, col)], fd, TOL);
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
        assert_lower_close(lower.as_ref(), full.as_ref(), TOL);
        assert_close(lower[(0, 1)], 99.0, TOL);
    }

    #[test]
    fn get_set_params_roundtrip() {
        let mut kernel = MaternKernel::new(2.0, MaternNu::Half).expect("valid");
        let mut params = [0.0];
        kernel.get_params(&mut params).expect("len 1");
        assert_close(params[0], 2.0_f64.ln(), TOL);
        params[0] = 0.5_f64.ln();
        kernel.set_params(&params).expect("len 1");
        assert_close(kernel.lengthscale(), 0.5, TOL);
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
        assert_close(out[(0, 1)], square[(0, 1)], TOL);
        let mut diag = [0.0, 0.0];
        kernel.fill_diag(&mut diag);
        assert_close(diag[0], 1.0, TOL);
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
            Err(GprError::IndexOutOfRange { .. })
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
