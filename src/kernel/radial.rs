//! Coordinate derivatives of the radial leaves, from one implementation.
//!
//! RBF, Matérn, rational quadratic (isotropic and ARD) and Periodic are
//! `k(x1, x2) = g(q)` with `q = Σ_d w_d Δ_d²` and `Δ = x1 − x2`. A leaf gives
//! `g'(q)`, `g''(q)`, and how its parameters move `q`, `w`, and `g'`; the four
//! coordinate derivatives that `Sgpr<FreeInducing>` needs follow from those:
//!
//! ```text
//! ∂k/∂x2_a            = −2 g' w_a Δ_a
//! ∂²k/∂x2_a ∂x2_b     = 4 g'' w_a w_b Δ_a Δ_b + 2 g' w_a [a = b]
//! ∂²k/∂x1_a ∂x2_b     = −(the line above)
//! ∂²k/∂θ_p ∂x2_a      = −2 Δ_a w_a [ g'' ∂q/∂θ_p + ∂g'/∂θ_p + g' ∂ln w_a/∂θ_p ]
//! ```
//!
//! `∂q/∂θ_p` and `∂ln w_a/∂θ_p` are the change of `q` and `w_a` at fixed
//! coordinates; `∂g'/∂θ_p` is the change of `g'` at fixed `q`.

use super::{KernelScalar, require_coord_grad, write_rect};
use crate::error::GprError;
use crate::math::KernelMath;
use faer::{MatMut, MatRef};

/// The weights `w_d` of `q = Σ_d w_d Δ_d²`.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Weights<'a> {
    Uniform(f64),
    PerDim(&'a [f64]),
}

impl Weights<'_> {
    #[inline]
    fn at(&self, dim: usize) -> f64 {
        match self {
            Self::Uniform(w) => *w,
            Self::PerDim(w) => w[dim],
        }
    }

    fn check(&self, ncols: usize) -> Result<(), GprError> {
        match self {
            Self::PerDim(w) if w.len() != ncols => Err(GprError::DimensionMismatch {
                x_dim: ncols,
                expected_dim: w.len(),
            }),
            _ => Ok(()),
        }
    }
}

/// `g'(q)` and `g''(q)`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Jet<T> {
    pub(crate) g1: T,
    pub(crate) g2: T,
}

/// How one parameter moves `q` (at fixed coordinates) and `g'` (at fixed `q`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct ThetaTerm<T> {
    pub(crate) dq: T,
    pub(crate) dg1: T,
}

/// A leaf of the form `g(q)`.
pub(crate) trait Radial {
    fn weights(&self) -> Weights<'_>;

    /// `g'(q)` and `g''(q)`.
    ///
    /// # Errors
    ///
    /// [`GprError::CoordGradientUnsupported`] when `g` is not twice
    /// differentiable in the coordinates (Matérn ν = 1/2 at `q = 0`).
    fn jet<M: KernelMath, T: KernelScalar>(&self, q: T) -> Result<Jet<T>, GprError>;

    /// The change of `q` and `g'` under parameter `param`. `pick` is
    /// `w_param Δ_param²` when `param` indexes a dimension, else zero.
    fn theta<M: KernelMath, T: KernelScalar>(
        &self,
        q: T,
        param: usize,
        pick: T,
    ) -> Result<ThetaTerm<T>, GprError>;

    /// `∂ ln w_dim / ∂θ_param`.
    fn dlogw(&self, param: usize, dim: usize) -> f64;

    fn num_params(&self) -> usize;
}

fn q_of<T: KernelScalar>(
    x1: MatRef<'_, T>,
    row: usize,
    x2: MatRef<'_, T>,
    col: usize,
    w: Weights<'_>,
) -> Result<T, GprError> {
    let mut q = T::from_f64(0.0);
    for dim in 0..x1.ncols() {
        let diff = x1[(row, dim)] - x2[(col, dim)];
        if !diff.is_finite() {
            return Err(GprError::NonFiniteInput);
        }
        q += diff * diff * T::from_f64(w.at(dim));
    }
    if q.is_finite() {
        Ok(q.max(T::from_f64(0.0)))
    } else {
        Err(GprError::NonFiniteKernelValue)
    }
}

fn delta<T: KernelScalar>(
    x1: MatRef<'_, T>,
    row: usize,
    x2: MatRef<'_, T>,
    col: usize,
    dim: usize,
) -> T {
    x1[(row, dim)] - x2[(col, dim)]
}

fn finite<T: KernelScalar>(value: T) -> Result<T, GprError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(GprError::NonFiniteKernelValue)
    }
}

/// `∂K(X1, X2)/∂X2[*, dim]`.
pub(crate) fn grad_wrt_coord_dim<M: KernelMath, T: KernelScalar>(
    leaf: &impl Radial,
    x1: MatRef<'_, T>,
    x2: MatRef<'_, T>,
    d_k: MatMut<'_, T>,
    dim: usize,
) -> Result<(), GprError> {
    require_coord_grad(x1, x2, d_k.as_ref(), dim)?;
    let w = leaf.weights();
    w.check(x1.ncols())?;
    let scale = T::from_f64(-2.0 * w.at(dim));
    write_rect(d_k, |row, col| {
        let q = q_of(x1, row, x2, col, w)?;
        finite(leaf.jet::<M, T>(q)?.g1 * scale * delta(x1, row, x2, col, dim))
    })
}

/// `∂²K/∂X2[*, a] ∂X2[*, b]` (`mixed = false`) or `∂²K/∂X1[*, a] ∂X2[*, b]`
/// (`mixed = true`).
pub(crate) fn hess_wrt_coord<M: KernelMath, T: KernelScalar>(
    leaf: &impl Radial,
    x1: MatRef<'_, T>,
    x2: MatRef<'_, T>,
    d2_k: MatMut<'_, T>,
    (dim_a, dim_b): (usize, usize),
    mixed: bool,
) -> Result<(), GprError> {
    require_coord_grad(x1, x2, d2_k.as_ref(), dim_a)?;
    require_coord_grad(x1, x2, d2_k.as_ref(), dim_b)?;
    let w = leaf.weights();
    w.check(x1.ncols())?;
    let (wa, wb) = (T::from_f64(w.at(dim_a)), T::from_f64(w.at(dim_b)));
    let sign = T::from_f64(if mixed { -1.0 } else { 1.0 });
    write_rect(d2_k, |row, col| {
        let q = q_of(x1, row, x2, col, w)?;
        let jet = leaf.jet::<M, T>(q)?;
        let mut value = T::from_f64(4.0)
            * jet.g2
            * wa
            * wb
            * delta(x1, row, x2, col, dim_a)
            * delta(x1, row, x2, col, dim_b);
        if dim_a == dim_b {
            value += T::from_f64(2.0) * jet.g1 * wa;
        }
        finite(sign * value)
    })
}

/// `∂²K/∂θ_param ∂X2[*, dim]`.
pub(crate) fn hess_theta_coord_dim<M: KernelMath, T: KernelScalar>(
    leaf: &impl Radial,
    x1: MatRef<'_, T>,
    x2: MatRef<'_, T>,
    d2_k: MatMut<'_, T>,
    param: usize,
    dim: usize,
) -> Result<(), GprError> {
    if param >= leaf.num_params() {
        return Err(GprError::IndexOutOfRange {
            reason: format!("kernel parameter index {param} is out of range"),
        });
    }
    require_coord_grad(x1, x2, d2_k.as_ref(), dim)?;
    let w = leaf.weights();
    w.check(x1.ncols())?;
    let wa = T::from_f64(w.at(dim));
    let dlogw = T::from_f64(leaf.dlogw(param, dim));
    let picks_dim = param < x1.ncols();
    write_rect(d2_k, |row, col| {
        let q = q_of(x1, row, x2, col, w)?;
        let pick = if picks_dim {
            let d = delta(x1, row, x2, col, param);
            d * d * T::from_f64(w.at(param))
        } else {
            T::from_f64(0.0)
        };
        let jet = leaf.jet::<M, T>(q)?;
        let theta = leaf.theta::<M, T>(q, param, pick)?;
        let inner = wa * (jet.g2 * theta.dq + theta.dg1) + jet.g1 * wa * dlogw;
        finite(T::from_f64(-2.0) * delta(x1, row, x2, col, dim) * inner)
    })
}

// ---- the leaves --------------------------------------------------------

use super::{
    MaternArdKernel, MaternKernel, MaternNu, PeriodicKernel, RationalQuadraticArdKernel,
    RationalQuadraticKernel, RbfArdKernel, RbfKernel,
};

fn zero<T: KernelScalar>() -> T {
    T::from_f64(0.0)
}

fn scaled_sq(ell: f64) -> Weights<'static> {
    Weights::Uniform(1.0 / (ell * ell))
}

/// The RBF `g(q) = exp(−q/2)`.
fn rbf_jet<M: KernelMath, T: KernelScalar>(q: T) -> Jet<T> {
    let k = M::exp(T::from_f64(-0.5) * q);
    Jet {
        g1: T::from_f64(-0.5) * k,
        g2: T::from_f64(0.25) * k,
    }
}

/// The Matérn `g(q)` at `q = r²` (already divided by the lengthscale).
fn matern_jet<M: KernelMath, T: KernelScalar>(nu: MaternNu, q: T) -> Result<Jet<T>, GprError> {
    let zero = zero::<T>();
    match nu {
        MaternNu::Half => Err(GprError::CoordGradientUnsupported),
        MaternNu::ThreeHalves => {
            let rho = T::from_f64(3.0_f64.sqrt()) * q.sqrt();
            let e = M::exp(-rho);
            let g2 = if rho > zero {
                T::from_f64(2.25) * e / rho
            } else {
                zero
            };
            Ok(Jet {
                g1: T::from_f64(-1.5) * e,
                g2,
            })
        }
        MaternNu::FiveHalves => {
            let rho = T::from_f64(5.0_f64.sqrt()) * q.sqrt();
            let e = M::exp(-rho);
            Ok(Jet {
                g1: T::from_f64(-5.0 / 6.0) * (T::from_f64(1.0) + rho) * e,
                g2: T::from_f64(25.0 / 12.0) * e,
            })
        }
    }
}

/// `u = 1 + q / (2α)`, `k = u^{−α}`.
fn rq_parts<T: KernelScalar>(q: T, alpha: f64) -> (T, T) {
    let alpha = T::from_f64(alpha);
    let u = T::from_f64(1.0) + q / (T::from_f64(2.0) * alpha);
    (u, u.powf(-alpha))
}

fn rq_jet<T: KernelScalar>(q: T, alpha: f64) -> Jet<T> {
    let (u, k) = rq_parts(q, alpha);
    Jet {
        g1: T::from_f64(-0.5) * k / u,
        g2: (T::from_f64(1.0) + T::from_f64(1.0) / T::from_f64(alpha)) * k
            / (T::from_f64(4.0) * u * u),
    }
}

/// `∂g'/∂ log α` at fixed `q`.
fn rq_dg1_dlog_alpha<T: KernelScalar>(q: T, alpha: f64) -> T {
    let (u, k) = rq_parts(q, alpha);
    let a = T::from_f64(alpha);
    T::from_f64(-0.5) * k / u * (-a * u.ln() + (a + T::from_f64(1.0)) * (u - T::from_f64(1.0)) / u)
}

impl Radial for RbfKernel {
    fn weights(&self) -> Weights<'_> {
        scaled_sq(self.lengthscale())
    }
    fn jet<M: KernelMath, T: KernelScalar>(&self, q: T) -> Result<Jet<T>, GprError> {
        Ok(rbf_jet::<M, T>(q))
    }
    fn theta<M: KernelMath, T: KernelScalar>(
        &self,
        q: T,
        _param: usize,
        _pick: T,
    ) -> Result<ThetaTerm<T>, GprError> {
        Ok(ThetaTerm {
            dq: T::from_f64(-2.0) * q,
            dg1: zero(),
        })
    }
    fn dlogw(&self, _param: usize, _dim: usize) -> f64 {
        -2.0
    }
    fn num_params(&self) -> usize {
        1
    }
}

impl Radial for RbfArdKernel {
    fn weights(&self) -> Weights<'_> {
        Weights::PerDim(self.lengthscales().inv_ell_sq())
    }
    fn jet<M: KernelMath, T: KernelScalar>(&self, q: T) -> Result<Jet<T>, GprError> {
        Ok(rbf_jet::<M, T>(q))
    }
    fn theta<M: KernelMath, T: KernelScalar>(
        &self,
        _q: T,
        _param: usize,
        pick: T,
    ) -> Result<ThetaTerm<T>, GprError> {
        Ok(ThetaTerm {
            dq: T::from_f64(-2.0) * pick,
            dg1: zero(),
        })
    }
    fn dlogw(&self, param: usize, dim: usize) -> f64 {
        if param == dim { -2.0 } else { 0.0 }
    }
    fn num_params(&self) -> usize {
        self.lengthscales().num_params()
    }
}

impl Radial for MaternKernel {
    fn weights(&self) -> Weights<'_> {
        scaled_sq(self.lengthscale())
    }
    fn jet<M: KernelMath, T: KernelScalar>(&self, q: T) -> Result<Jet<T>, GprError> {
        matern_jet::<M, T>(self.nu(), q)
    }
    fn theta<M: KernelMath, T: KernelScalar>(
        &self,
        q: T,
        _param: usize,
        _pick: T,
    ) -> Result<ThetaTerm<T>, GprError> {
        Ok(ThetaTerm {
            dq: T::from_f64(-2.0) * q,
            dg1: zero(),
        })
    }
    fn dlogw(&self, _param: usize, _dim: usize) -> f64 {
        -2.0
    }
    fn num_params(&self) -> usize {
        1
    }
}

impl Radial for MaternArdKernel {
    fn weights(&self) -> Weights<'_> {
        Weights::PerDim(self.lengthscales().inv_ell_sq())
    }
    fn jet<M: KernelMath, T: KernelScalar>(&self, q: T) -> Result<Jet<T>, GprError> {
        matern_jet::<M, T>(self.nu(), q)
    }
    fn theta<M: KernelMath, T: KernelScalar>(
        &self,
        _q: T,
        _param: usize,
        pick: T,
    ) -> Result<ThetaTerm<T>, GprError> {
        Ok(ThetaTerm {
            dq: T::from_f64(-2.0) * pick,
            dg1: zero(),
        })
    }
    fn dlogw(&self, param: usize, dim: usize) -> f64 {
        if param == dim { -2.0 } else { 0.0 }
    }
    fn num_params(&self) -> usize {
        self.lengthscales().num_params()
    }
}

impl Radial for RationalQuadraticKernel {
    fn weights(&self) -> Weights<'_> {
        scaled_sq(self.lengthscale())
    }
    fn jet<M: KernelMath, T: KernelScalar>(&self, q: T) -> Result<Jet<T>, GprError> {
        Ok(rq_jet(q, self.alpha()))
    }
    fn theta<M: KernelMath, T: KernelScalar>(
        &self,
        q: T,
        param: usize,
        _pick: T,
    ) -> Result<ThetaTerm<T>, GprError> {
        Ok(if param == 0 {
            ThetaTerm {
                dq: T::from_f64(-2.0) * q,
                dg1: zero(),
            }
        } else {
            ThetaTerm {
                dq: zero(),
                dg1: rq_dg1_dlog_alpha(q, self.alpha()),
            }
        })
    }
    fn dlogw(&self, param: usize, _dim: usize) -> f64 {
        if param == 0 { -2.0 } else { 0.0 }
    }
    fn num_params(&self) -> usize {
        2
    }
}

impl Radial for RationalQuadraticArdKernel {
    fn weights(&self) -> Weights<'_> {
        Weights::PerDim(self.lengthscales().inv_ell_sq())
    }
    fn jet<M: KernelMath, T: KernelScalar>(&self, q: T) -> Result<Jet<T>, GprError> {
        Ok(rq_jet(q, self.alpha()))
    }
    fn theta<M: KernelMath, T: KernelScalar>(
        &self,
        q: T,
        param: usize,
        pick: T,
    ) -> Result<ThetaTerm<T>, GprError> {
        Ok(if param < self.lengthscales().num_params() {
            ThetaTerm {
                dq: T::from_f64(-2.0) * pick,
                dg1: zero(),
            }
        } else {
            ThetaTerm {
                dq: zero(),
                dg1: rq_dg1_dlog_alpha(q, self.alpha()),
            }
        })
    }
    fn dlogw(&self, param: usize, dim: usize) -> f64 {
        if param == dim { -2.0 } else { 0.0 }
    }
    fn num_params(&self) -> usize {
        self.lengthscales().num_params() + 1
    }
}

/// `sinc(x) = sin x / x` and `sinc'(x) / x = (x cos x − sin x) / x³`.
fn sinc_pair<T: KernelScalar>(x: T) -> (T, T) {
    let x2 = x * x;
    if x2 < T::from_f64(2.5e-3) {
        let one = T::from_f64(1.0);
        let sinc = one - x2 / T::from_f64(6.0) + x2 * x2 / T::from_f64(120.0);
        let s2 = T::from_f64(-1.0 / 3.0) + x2 / T::from_f64(30.0) - x2 * x2 / T::from_f64(840.0);
        (sinc, s2)
    } else {
        let (s, c) = (x.sin(), x.cos());
        (s / x, (x * c - s) / (x2 * x))
    }
}

/// The Periodic leaf is `g(q)` with `q = ‖Δ‖²` (weight 1), `x = 2π r / p`,
/// `u = (1 − cos x) / ℓ²`, `g = e^{−u}`. Then `g' = −B e^{−u} sinc x` with
/// `B = 2π² / (p² ℓ²)`.
struct PeriodicParts<T> {
    x: T,
    b: T,
    e: T,
    sinc: T,
    s2: T,
    inv_ell_sq: T,
}

fn periodic_parts<M: KernelMath, T: KernelScalar>(leaf: &PeriodicKernel, q: T) -> PeriodicParts<T> {
    let (ell, p) = (leaf.lengthscale(), leaf.period());
    let x = T::from_f64(2.0 * std::f64::consts::PI / p) * q.sqrt();
    let inv_ell_sq = T::from_f64(1.0 / (ell * ell));
    let u = (T::from_f64(1.0) - x.cos()) * inv_ell_sq;
    let (sinc, s2) = sinc_pair(x);
    PeriodicParts {
        x,
        b: T::from_f64(2.0 * std::f64::consts::PI.powi(2) / (p * p * ell * ell)),
        e: M::exp(-u),
        sinc,
        s2,
        inv_ell_sq,
    }
}

impl Radial for PeriodicKernel {
    fn weights(&self) -> Weights<'_> {
        Weights::Uniform(1.0)
    }
    fn jet<M: KernelMath, T: KernelScalar>(&self, q: T) -> Result<Jet<T>, GprError> {
        let t = periodic_parts::<M, T>(self, q);
        let two_pi_sq_over_p_sq =
            T::from_f64(2.0 * std::f64::consts::PI.powi(2) / (self.period() * self.period()));
        Ok(Jet {
            g1: -t.b * t.e * t.sinc,
            g2: -t.b * two_pi_sq_over_p_sq * t.e * (t.s2 - t.sinc * t.sinc * t.inv_ell_sq),
        })
    }
    fn theta<M: KernelMath, T: KernelScalar>(
        &self,
        q: T,
        param: usize,
        _pick: T,
    ) -> Result<ThetaTerm<T>, GprError> {
        let t = periodic_parts::<M, T>(self, q);
        let g1 = -t.b * t.e * t.sinc;
        let dg1 = if param == 0 {
            // ∂/∂log ℓ: B → −2B, u → −2u
            g1 * (T::from_f64(-2.0)
                + T::from_f64(2.0) * (T::from_f64(1.0) - t.x.cos()) * t.inv_ell_sq)
        } else {
            // ∂/∂log p: B → −2B, x → −x; x sinc'(x) = cos x − sinc x
            let sin_x = t.x.sin();
            let x_sinc_prime = t.x.cos() - t.sinc;
            -t.b * t.e * ((T::from_f64(-2.0) + t.x * sin_x * t.inv_ell_sq) * t.sinc - x_sinc_prime)
        };
        Ok(ThetaTerm { dq: zero(), dg1 })
    }
    fn dlogw(&self, _param: usize, _dim: usize) -> f64 {
        0.0
    }
    fn num_params(&self) -> usize {
        2
    }
}
