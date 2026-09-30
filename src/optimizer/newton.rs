//! Newton's method with a backtracking line search and a private faer inverse.

use std::num::NonZeroU32;

use faer::Mat;
use faer::linalg::solvers::DenseSolveCore;

use crate::error::GprError;
use crate::objective::{HasBounds, TwiceDifferentiable};
use crate::param::Interval;

use super::logit::{
    LogitMapped, keep_better, log_theta_to_z, sample_log_uniform_z, z_to_log_theta,
};
use super::{OptResult, Optimizer, Restarts};

/// Newton's method with a backtracking line search.
///
/// The Hessian lives in unconstrained logit coordinates. The direction is
/// `−H⁻¹ g` from a private faer LU, or the steepest descent `−g` when `H` is
/// singular or the step does not descend. The step starts at `gamma` and is
/// halved until the Armijo decrease holds; a candidate that cannot be
/// evaluated (out of the bounds, not finite, not positive definite) counts as
/// no decrease. When no step decreases the objective away from a minimum,
/// [`GprError::OptimizationNotConverged`] carries the iteration count.
#[derive(Clone, Debug, PartialEq)]
pub struct Newton {
    max_iterations: u64,
    tolerance: f64,
    gamma: f64,
    restarts: Option<Restarts>,
}

impl Default for Newton {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: f64::EPSILON.sqrt(),
            gamma: 1.0,
            restarts: None,
        }
    }
}

impl Newton {
    /// Builds Newton with 100 iterations, gradient tolerance `sqrt(ε)`, and
    /// step size `gamma = 1`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::num::NonZeroU32;
    /// use gprx::Newton;
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let _newton = Newton::new()
    ///     .with_max_iterations(50)
    ///     .with_tolerance(1e-8)?
    ///     .with_gamma(1.0)?
    ///     .with_restarts(NonZeroU32::MIN, 0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the iteration cap passed to argmin (default 100).
    pub fn with_max_iterations(mut self, max_iterations: u64) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Sets the gradient-norm tolerance (default `sqrt(ε)`).
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidConfig`] if `tolerance` is not finite
    /// or is negative.
    pub fn with_tolerance(mut self, tolerance: f64) -> Result<Self, GprError> {
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(GprError::InvalidConfig {
                reason: "Newton gradient tolerance must be finite and >= 0".to_owned(),
            });
        }
        self.tolerance = tolerance;
        Ok(self)
    }

    /// Sets the Newton step size `gamma ∈ (0, 1]` (default 1).
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidConfig`] if `gamma` is outside
    /// `(0, 1]`.
    pub fn with_gamma(mut self, gamma: f64) -> Result<Self, GprError> {
        if !gamma.is_finite() || gamma <= 0.0 || gamma > 1.0 {
            return Err(GprError::InvalidConfig {
                reason: "Newton gamma must be in (0, 1]".to_owned(),
            });
        }
        self.gamma = gamma;
        Ok(self)
    }

    /// Adds `n` extra log-uniform starts (`n ≥ 1`) and keeps the lowest NLML.
    pub fn with_restarts(mut self, n: NonZeroU32, seed: u64) -> Self {
        self.restarts = Some(Restarts { n, seed });
        self
    }

    #[cfg(test)]
    pub(crate) fn minimize_unconstrained<P: TwiceDifferentiable>(
        &self,
        objective: &mut P,
        init: &[f64],
    ) -> Result<OptResult, GprError> {
        run_newton(self, objective, init)
    }
}

impl<P: TwiceDifferentiable + HasBounds> Optimizer<P> for Newton {
    fn minimize(&self, objective: &mut P, init: &[f64]) -> Result<OptResult, GprError> {
        let n = objective.num_params();
        if init.len() != n {
            return Err(GprError::LengthMismatch {
                reason: format!("expected {n} parameters, got {}", init.len()),
            });
        }
        let mut intervals = vec![Interval::DEFAULT_POSITIVE; n];
        objective.fill_intervals(&mut intervals)?;
        let mut best: Option<OptResult> = None;
        let first_z = log_theta_to_z(init, &intervals)?;
        consider_newton(self, objective, &intervals, &first_z, &mut best)?;
        if let Some(restarts) = self.restarts {
            let mut rng = crate::rng::small_rng(restarts.seed);
            for _ in 0..restarts.n.get() {
                let z = sample_log_uniform_z(&intervals, &mut rng)?;
                let _ = consider_newton(self, objective, &intervals, &z, &mut best);
            }
        }
        best.ok_or(GprError::OptimizationNotConverged { iterations: 0 })
    }
}

fn consider_newton<P: TwiceDifferentiable>(
    newton: &Newton,
    objective: &mut P,
    intervals: &[Interval],
    init_z: &[f64],
    best: &mut Option<OptResult>,
) -> Result<(), GprError> {
    let mut mapped = LogitMapped {
        inner: objective,
        intervals,
        log_scratch: vec![0.0; init_z.len()],
    };
    let run = run_newton(newton, &mut mapped, init_z)?;
    let log_theta = z_to_log_theta(&run.params, intervals)?;
    let mut grad = vec![0.0; log_theta.len()];
    let value = objective.value_and_gradient_into(&log_theta, &mut grad)?;
    keep_better(
        best,
        OptResult {
            params: log_theta,
            value,
            iterations: run.iterations,
        },
    );
    Ok(())
}

/// Backtracking halvings of one step before the search gives up.
const MAX_BACKTRACKS: usize = 30;
/// Armijo constant of the sufficient decrease `f(x + t d) ≤ f(x) + c t g·d`.
const ARMIJO: f64 = 1e-4;

fn run_newton<P: TwiceDifferentiable>(
    newton: &Newton,
    objective: &mut P,
    init: &[f64],
) -> Result<OptResult, GprError> {
    let n = objective.num_params();
    if init.len() != n {
        return Err(GprError::LengthMismatch {
            reason: format!("expected {n} parameters, got {}", init.len()),
        });
    }
    let mut cache = HessCache::new(objective, n);
    let mut x = init.to_vec();
    let mut value = cache.ensure(&x)?;
    let mut iterations = 0_u64;
    let mut candidate = vec![0.0; n];
    while iterations < newton.max_iterations {
        let grad = cache.grad.clone();
        let grad_norm = norm(&grad);
        if grad_norm <= newton.tolerance {
            break;
        }
        iterations += 1;
        cache.fill_hessian(&x)?;
        let direction = descent_direction(&cache.hess, &grad, n);
        let slope: f64 = grad.iter().zip(&direction).map(|(g, d)| g * d).sum();
        let mut step = newton.gamma;
        let mut accepted = false;
        for _ in 0..MAX_BACKTRACKS {
            for ((c, xi), d) in candidate.iter_mut().zip(&x).zip(&direction) {
                *c = xi + step * d;
            }
            // A candidate that cannot be evaluated (out of the bounds, not
            // finite, not positive definite) is rejected like one that does
            // not decrease enough.
            if let Ok(v) = cache.ensure(&candidate) {
                if v <= value + ARMIJO * step * slope {
                    x.copy_from_slice(&candidate);
                    value = v;
                    accepted = true;
                    break;
                }
            }
            step *= 0.5;
        }
        if !accepted {
            // At a minimum to rounding the line search finds no decrease.
            if grad_norm <= 1e-4 * value.abs().max(1.0) {
                break;
            }
            return Err(GprError::OptimizationNotConverged {
                iterations: iterations as usize,
            });
        }
    }
    Ok(OptResult {
        params: x,
        value,
        iterations,
    })
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|a| a * a).sum::<f64>().sqrt()
}

/// `−H⁻¹ g`, or `−g` when `H` is singular, not finite, or the step does not
/// descend (`H` not positive definite).
fn descent_direction(hess: &[f64], grad: &[f64], n: usize) -> Vec<f64> {
    if let Some(step) = newton_step(hess, grad, n) {
        let slope: f64 = grad.iter().zip(&step).map(|(g, d)| g * d).sum();
        if slope < 0.0 {
            return step;
        }
    }
    grad.iter().map(|g| -g).collect()
}

fn newton_step(hess: &[f64], grad: &[f64], n: usize) -> Option<Vec<f64>> {
    if hess.len() != n * n || grad.len() != n {
        return None;
    }
    let a = Mat::<f64>::from_fn(n, n, |row, col| hess[row * n + col]);
    let lu = a.partial_piv_lu();
    let u = lu.U();
    let scale = (0..n).fold(0.0_f64, |m, i| m.max(u[(i, i)].abs()));
    let tol = f64::EPSILON * scale.max(1.0) * n as f64;
    if (0..n).any(|i| !u[(i, i)].is_finite() || u[(i, i)].abs() <= tol) {
        return None;
    }
    let inv = lu.inverse();
    let step: Vec<f64> = (0..n)
        .map(|row| -(0..n).map(|col| inv[(row, col)] * grad[col]).sum::<f64>())
        .collect();
    step.iter().all(|v| v.is_finite()).then_some(step)
}

/// The objective with the buffers of the last evaluation. The Hessian is
/// filled only for a point that was accepted, not for every candidate.
struct HessCache<'a, P: ?Sized> {
    objective: &'a mut P,
    grad: Vec<f64>,
    hess: Vec<f64>,
}

impl<'a, P: TwiceDifferentiable + ?Sized> HessCache<'a, P> {
    fn new(objective: &'a mut P, n: usize) -> Self {
        Self {
            objective,
            grad: vec![0.0; n],
            hess: vec![0.0; n * n],
        }
    }

    /// Value and gradient at `param`, into `self.grad`.
    fn ensure(&mut self, param: &[f64]) -> Result<f64, GprError> {
        let value = self
            .objective
            .value_and_gradient_into(param, &mut self.grad)?;
        if !value.is_finite() || self.grad.iter().any(|g| !g.is_finite()) {
            return Err(GprError::OptimizationNotConverged { iterations: 0 });
        }
        Ok(value)
    }

    /// The Hessian at `param`, into `self.hess`.
    fn fill_hessian(&mut self, param: &[f64]) -> Result<(), GprError> {
        self.objective.hessian_into(param, &mut self.hess)?;
        if self.hess.iter().any(|h| !h.is_finite()) {
            return Err(GprError::OptimizationNotConverged { iterations: 0 });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::Newton;
    use crate::error::GprError;
    use crate::gpr::Gpr;
    use crate::kernel::{KernelSpec, RbfKernel};
    use crate::likelihood::GaussianLikelihood;
    use crate::objective::{Differentiable, Objective, TwiceDifferentiable};
    use crate::optimizer::{Fixed, Optimizer};

    const TOL: f64 = 1e-6;

    use crate::test_check::assert_close;

    struct Quadratic;

    impl Objective for Quadratic {
        fn num_params(&self) -> usize {
            2
        }

        fn value(&mut self, params: &[f64]) -> Result<f64, GprError> {
            let mut dummy = [0.0; 2];
            self.value_and_gradient_into(params, &mut dummy)
        }
    }

    impl Differentiable for Quadratic {
        fn gradient_into(&mut self, params: &[f64], out: &mut [f64]) -> Result<(), GprError> {
            self.value_and_gradient_into(params, out).map(|_| ())
        }

        fn value_and_gradient_into(
            &mut self,
            params: &[f64],
            out: &mut [f64],
        ) -> Result<f64, GprError> {
            if params.len() != 2 || out.len() != 2 {
                return Err(GprError::ShapeMismatch {
                    reason: "quadratic is 2-D".to_owned(),
                });
            }
            out[0] = params[0];
            out[1] = params[1];
            Ok(0.5 * (params[0] * params[0] + params[1] * params[1]))
        }
    }

    impl TwiceDifferentiable for Quadratic {
        fn hessian_into(&mut self, params: &[f64], out: &mut [f64]) -> Result<(), GprError> {
            if params.len() != 2 || out.len() != 4 {
                return Err(GprError::ShapeMismatch {
                    reason: "quadratic Hessian is 2x2".to_owned(),
                });
            }
            out[0] = 1.0;
            out[1] = 0.0;
            out[2] = 0.0;
            out[3] = 1.0;
            Ok(())
        }
    }

    #[test]
    fn newton_minimizes_quadratic_bowl() {
        let mut obj = Quadratic;
        let result = Newton::new()
            .with_max_iterations(5)
            .minimize_unconstrained(&mut obj, &[1.0, -0.5])
            .expect("bowl");
        assert_close(result.params[0], 0.0, TOL);
        assert_close(result.params[1], 0.0, TOL);
        assert!(result.value < 1e-12);
    }

    #[test]
    fn newton_lowers_gpr_nlml() {
        let kernel = KernelSpec::from(RbfKernel::new(1.25).expect("valid"));
        let likelihood = GaussianLikelihood::new(0.16).expect("valid");
        let x = [0.0, 0.8, 1.7];
        let y = [0.4, -0.2, 0.9];
        let mut gpr = Gpr::new(kernel, likelihood)
            .with_optimizer(Fixed)
            .factor(&x, 3, 1, &y)
            .unwrap_or_else(|(_, e)| panic!("{e}"));
        let mut init = [0.0; 2];
        gpr.get_params(&mut init).expect("len 2");
        let start;
        let result;
        {
            let mut obj = gpr.objective();
            start = obj.value(&init).expect("spd");
            result = Newton::new()
                .with_max_iterations(30)
                .minimize(&mut obj, &init)
                .expect("newton");
        }
        assert!(
            result.value < start,
            "start={start}, newton={}",
            result.value
        );
        let fitted = Gpr::new(
            KernelSpec::from(RbfKernel::new(1.25).expect("valid")),
            GaussianLikelihood::new(0.16).expect("valid"),
        )
        .with_optimizer(Newton::new().with_max_iterations(30))
        .fit(&x, 3, 1, &y)
        .unwrap_or_else(|(_, e)| panic!("{e}"));
        assert!(fitted.neg_log_marginal_likelihood().expect("fitted") < start);
    }

    /// `√(1 + x²)`: plain Newton from `|x| > 1` jumps away (`x → −x³`); values
    /// beyond `|x| = 50` cannot be evaluated.
    struct Sqrt1PlusSquare;

    impl Objective for Sqrt1PlusSquare {
        fn num_params(&self) -> usize {
            1
        }

        fn value(&mut self, params: &[f64]) -> Result<f64, GprError> {
            let mut dummy = [0.0; 1];
            self.value_and_gradient_into(params, &mut dummy)
        }
    }

    impl Differentiable for Sqrt1PlusSquare {
        fn gradient_into(&mut self, params: &[f64], out: &mut [f64]) -> Result<(), GprError> {
            self.value_and_gradient_into(params, out).map(|_| ())
        }

        fn value_and_gradient_into(
            &mut self,
            params: &[f64],
            out: &mut [f64],
        ) -> Result<f64, GprError> {
            let x = params[0];
            if x.abs() > 50.0 {
                return Err(GprError::NonFiniteKernelValue);
            }
            let root = (1.0 + x * x).sqrt();
            out[0] = x / root;
            Ok(root)
        }
    }

    impl TwiceDifferentiable for Sqrt1PlusSquare {
        fn hessian_into(&mut self, params: &[f64], out: &mut [f64]) -> Result<(), GprError> {
            let x = params[0];
            out[0] = (1.0 + x * x).powf(-1.5);
            Ok(())
        }
    }

    #[test]
    fn newton_backtracks_where_the_plain_step_diverges() {
        for start in [1.5, 2.0, 4.0, -3.0] {
            let mut obj = Sqrt1PlusSquare;
            let result = Newton::new()
                .with_max_iterations(200)
                .minimize_unconstrained(&mut obj, &[start])
                .expect("safeguarded newton");
            assert_close(result.params[0], 0.0, 1e-6);
        }
    }

    /// The Hessian of `−cos(x)` at `x = 2` is negative: the Newton step
    /// ascends, and the search falls back to the steepest descent.
    struct NegativeCurvature;

    impl Objective for NegativeCurvature {
        fn num_params(&self) -> usize {
            1
        }

        fn value(&mut self, params: &[f64]) -> Result<f64, GprError> {
            Ok(-params[0].cos())
        }
    }

    impl Differentiable for NegativeCurvature {
        fn gradient_into(&mut self, params: &[f64], out: &mut [f64]) -> Result<(), GprError> {
            out[0] = params[0].sin();
            Ok(())
        }

        fn value_and_gradient_into(
            &mut self,
            params: &[f64],
            out: &mut [f64],
        ) -> Result<f64, GprError> {
            out[0] = params[0].sin();
            Ok(-params[0].cos())
        }
    }

    impl TwiceDifferentiable for NegativeCurvature {
        fn hessian_into(&mut self, params: &[f64], out: &mut [f64]) -> Result<(), GprError> {
            out[0] = params[0].cos();
            Ok(())
        }
    }

    #[test]
    fn newton_descends_when_the_hessian_is_not_positive_definite() {
        let mut obj = NegativeCurvature;
        let result = Newton::new()
            .with_max_iterations(200)
            .minimize_unconstrained(&mut obj, &[2.0])
            .expect("falls back to steepest descent");
        assert!(result.value < -0.999, "value {}", result.value);
    }

    #[test]
    fn newton_rejects_bad_gamma() {
        assert!(Newton::new().with_gamma(0.0).is_err());
        assert!(Newton::new().with_gamma(1.1).is_err());
    }
}
