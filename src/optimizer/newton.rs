//! Newton's method via argmin, with a private faer inverse.

use std::cell::RefCell;
use std::num::NonZeroU32;

use argmin::core::{
    CostFunction, Error as ArgminError, Executor, Gradient, Hessian, IterState, KV, Problem,
    Solver, State, TerminationReason, TerminationStatus,
};
use argmin::solver::newton::Newton as ArgminNewton;
use argmin_math::{ArgminDot, ArgminInv};
use faer::Mat;
use faer::linalg::solvers::DenseSolveCore;

use crate::error::GprError;
use crate::objective::{HasBounds, TwiceDifferentiable};
use crate::param::Interval;

use super::logit::{
    LogitMapped, keep_better, log_theta_to_z, map_argmin_error, sample_log_uniform_z,
    z_to_log_theta,
};
use super::{OptResult, Optimizer, Restarts};

/// Newton's method via argmin.
///
/// The Hessian lives in unconstrained logit coordinates. `H⁻¹` is a private
/// faer factorization of the `p×p` matrix. A singular Hessian is
/// [`GprError::OptimizationNotConverged`].
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
    /// Returns [`GprError::InvalidHyperparameter`] if `tolerance` is not finite
    /// or is negative.
    pub fn with_tolerance(mut self, tolerance: f64) -> Result<Self, GprError> {
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(GprError::InvalidHyperparameter {
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
    /// Returns [`GprError::InvalidHyperparameter`] if `gamma` is outside
    /// `(0, 1]`.
    pub fn with_gamma(mut self, gamma: f64) -> Result<Self, GprError> {
        if !gamma.is_finite() || gamma <= 0.0 || gamma > 1.0 {
            return Err(GprError::InvalidHyperparameter {
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
            return Err(GprError::InvalidHyperparameter {
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

fn run_newton<P: TwiceDifferentiable>(
    newton: &Newton,
    objective: &mut P,
    init: &[f64],
) -> Result<OptResult, GprError> {
    let n = objective.num_params();
    if init.len() != n {
        return Err(GprError::InvalidHyperparameter {
            reason: format!("expected {n} parameters, got {}", init.len()),
        });
    }
    let problem = NewtonProblem {
        inner: RefCell::new(HessCache::new(objective, n)),
    };
    let inner = ArgminNewton::new()
        .with_gamma(newton.gamma)
        .map_err(map_argmin_error)?;
    let solver = NewtonTol {
        inner,
        tolerance: newton.tolerance,
    };
    let (params, iterations) = {
        let result = Executor::new(problem, solver)
            .configure(|state| state.param(init.to_vec()).max_iters(newton.max_iterations))
            .ctrlc(false)
            .run()
            .map_err(map_newton_error)?;
        let state = result.state();
        let params = state
            .get_best_param()
            .cloned()
            .ok_or(GprError::OptimizationNotConverged {
                iterations: state.get_iter() as usize,
            })?;
        (params, state.get_iter())
    };
    let mut grad = vec![0.0; n];
    let value = objective.value_and_gradient_into(&params, &mut grad)?;
    Ok(OptResult {
        params,
        value,
        iterations,
    })
}

fn map_newton_error(err: ArgminError) -> GprError {
    if let Some(gpr) = err.downcast_ref::<GprError>() {
        return gpr.clone();
    }
    let text = err.to_string();
    if text.contains("singular") || text.contains("invert") || text.contains("inv") {
        GprError::OptimizationNotConverged { iterations: 0 }
    } else {
        map_argmin_error(err)
    }
}

struct NewtonTol {
    inner: ArgminNewton<f64>,
    tolerance: f64,
}

impl<O> Solver<O, IterState<Vec<f64>, Vec<f64>, (), NewtonHess, (), f64>> for NewtonTol
where
    O: Gradient<Param = Vec<f64>, Gradient = Vec<f64>>
        + Hessian<Param = Vec<f64>, Hessian = NewtonHess>,
{
    fn name(&self) -> &str {
        "Newton method"
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        state: IterState<Vec<f64>, Vec<f64>, (), NewtonHess, (), f64>,
    ) -> Result<
        (
            IterState<Vec<f64>, Vec<f64>, (), NewtonHess, (), f64>,
            Option<KV>,
        ),
        ArgminError,
    > {
        let (state, kv) = self.inner.next_iter(problem, state)?;
        let Some(param) = state.get_param().cloned() else {
            return Ok((state, kv));
        };
        let grad = problem.gradient(&param)?;
        Ok((state.gradient(grad), kv))
    }

    fn terminate(
        &mut self,
        state: &IterState<Vec<f64>, Vec<f64>, (), NewtonHess, (), f64>,
    ) -> TerminationStatus {
        if let Some(grad) = state.get_gradient() {
            let norm = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
            if norm <= self.tolerance {
                return TerminationStatus::Terminated(TerminationReason::SolverConverged);
            }
        }
        TerminationStatus::NotTerminated
    }
}

struct HessCache<'a, P: ?Sized> {
    objective: &'a mut P,
    params: Vec<f64>,
    value: Option<f64>,
    grad: Vec<f64>,
    hess: Vec<f64>,
}

impl<'a, P: TwiceDifferentiable + ?Sized> HessCache<'a, P> {
    fn new(objective: &'a mut P, n: usize) -> Self {
        Self {
            objective,
            params: Vec::new(),
            value: None,
            grad: vec![0.0; n],
            hess: vec![0.0; n * n],
        }
    }

    fn ensure(&mut self, param: &[f64]) -> Result<f64, GprError> {
        if let Some(value) = self.value {
            if same_params(&self.params, param) {
                return Ok(value);
            }
        }
        let n = param.len();
        if self.grad.len() != n {
            self.grad.resize(n, 0.0);
        }
        if self.hess.len() != n * n {
            self.hess.resize(n * n, 0.0);
        }
        let value = self
            .objective
            .value_and_gradient_into(param, &mut self.grad)?;
        self.objective.hessian_into(param, &mut self.hess)?;
        if !value.is_finite()
            || self.grad.iter().any(|g| !g.is_finite())
            || self.hess.iter().any(|h| !h.is_finite())
        {
            return Err(GprError::OptimizationNotConverged { iterations: 0 });
        }
        self.params.clear();
        self.params.extend_from_slice(param);
        self.value = Some(value);
        Ok(value)
    }
}

fn same_params(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

struct NewtonProblem<'a, P: ?Sized> {
    inner: RefCell<HessCache<'a, P>>,
}

impl<P: TwiceDifferentiable + ?Sized> CostFunction for NewtonProblem<'_, P> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, ArgminError> {
        self.inner
            .borrow_mut()
            .ensure(param)
            .map_err(ArgminError::from)
    }
}

impl<P: TwiceDifferentiable + ?Sized> Gradient for NewtonProblem<'_, P> {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, ArgminError> {
        let mut inner = self.inner.borrow_mut();
        inner.ensure(param).map_err(ArgminError::from)?;
        Ok(inner.grad.clone())
    }
}

impl<P: TwiceDifferentiable + ?Sized> Hessian for NewtonProblem<'_, P> {
    type Param = Vec<f64>;
    type Hessian = NewtonHess;

    fn hessian(&self, param: &Self::Param) -> Result<Self::Hessian, ArgminError> {
        let mut inner = self.inner.borrow_mut();
        inner.ensure(param).map_err(ArgminError::from)?;
        Ok(NewtonHess {
            data: inner.hess.clone(),
            n: inner.grad.len(),
        })
    }
}

/// Row-major `p×p` Hessian. Inverse uses faer LU.
#[derive(Clone, Debug)]
struct NewtonHess {
    data: Vec<f64>,
    n: usize,
}

impl ArgminInv<NewtonHess> for NewtonHess {
    fn inv(&self) -> Result<NewtonHess, ArgminError> {
        let n = self.n;
        if self.data.len() != n * n {
            return Err(ArgminError::from(GprError::OptimizationNotConverged {
                iterations: 0,
            }));
        }
        let a = Mat::<f64>::from_fn(n, n, |row, col| self.data[row * n + col]);
        let lu = a.partial_piv_lu();
        let u = lu.U();
        let mut scale = 0.0_f64;
        for i in 0..n {
            scale = scale.max(u[(i, i)].abs());
        }
        let tol = f64::EPSILON * scale.max(1.0) * n as f64;
        for i in 0..n {
            if !u[(i, i)].is_finite() || u[(i, i)].abs() <= tol {
                return Err(ArgminError::from(GprError::OptimizationNotConverged {
                    iterations: 0,
                }));
            }
        }
        let inv = lu.inverse();
        let mut data = vec![0.0; n * n];
        for row in 0..n {
            for col in 0..n {
                data[row * n + col] = inv[(row, col)];
            }
        }
        Ok(NewtonHess { data, n })
    }
}

impl ArgminDot<Vec<f64>, Vec<f64>> for NewtonHess {
    fn dot(&self, other: &Vec<f64>) -> Vec<f64> {
        let n = self.n;
        let mut out = vec![0.0; n];
        if other.len() != n || self.data.len() != n * n {
            return out;
        }
        for (row, slot) in out.iter_mut().enumerate() {
            let start = row * n;
            *slot = self.data[start..start + n]
                .iter()
                .zip(other.iter())
                .map(|(a, b)| a * b)
                .sum();
        }
        out
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

    fn assert_close(actual: f64, expected: f64) {
        let scale = expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= TOL * scale,
            "actual={actual}, expected={expected}"
        );
    }

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
                return Err(GprError::InvalidHyperparameter {
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
                return Err(GprError::InvalidHyperparameter {
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
        assert_close(result.params[0], 0.0);
        assert_close(result.params[1], 0.0);
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

    #[test]
    fn newton_rejects_bad_gamma() {
        assert!(Newton::new().with_gamma(0.0).is_err());
        assert!(Newton::new().with_gamma(1.1).is_err());
    }
}
