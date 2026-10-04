//! The argmin adapters every built-in argmin optimizer shares: one evaluation cache, one argmin problem, one run, one error mapping, and the comparison of runs.
//!
//! An optimizer picks how much an evaluation computes ([`ValueOnly`],
//! [`WithGradient`], [`WithHessian`]); everything else is the same for
//! [`super::Lbfgs`], [`super::NelderMead`], and [`super::TrustRegion`].

use std::cell::RefCell;
use std::marker::PhantomData;

use argmin::core::{
    CostFunction, Error as ArgminError, Executor, Gradient, Hessian, IterState, Solver, State,
};

use crate::error::GprError;
use crate::objective::{Differentiable, Objective, TwiceDifferentiable};
use crate::param::Interval;

use super::OptResult;
use super::logit::{LogitMapped, z_to_log_theta};

/// Cost returned to argmin for a point that cannot be evaluated: the
/// objective fails there, or its value, gradient, or Hessian is not finite
/// (for example outside an open [`Interval`], or a matrix that does not
/// factor). A line search backtracks from it and a trust region shrinks; the
/// solve goes on.
pub(super) const BARRIER_COST: f64 = 1.0e300;

/// What one evaluation computes besides the value.
pub(super) trait Order<P: ?Sized> {
    /// Whether the evaluation writes a gradient.
    const GRADIENT: bool;
    /// Whether the evaluation writes a Hessian.
    const HESSIAN: bool;

    /// The value at `x`, with the gradient and Hessian into `grad` and
    /// `hess` when this order writes them.
    fn eval(
        objective: &mut P,
        x: &[f64],
        grad: &mut [f64],
        hess: &mut [f64],
    ) -> Result<f64, GprError>;
}

/// The value only ([`Objective::value`]).
pub(super) struct ValueOnly;

/// The value and gradient in one call
/// ([`Differentiable::value_and_gradient_into`]).
pub(super) struct WithGradient;

/// The value, gradient, and Hessian in one call
/// ([`TwiceDifferentiable::value_gradient_hessian_into`]).
pub(super) struct WithHessian;

impl<P: Objective + ?Sized> Order<P> for ValueOnly {
    const GRADIENT: bool = false;
    const HESSIAN: bool = false;

    fn eval(objective: &mut P, x: &[f64], _: &mut [f64], _: &mut [f64]) -> Result<f64, GprError> {
        objective.value(x)
    }
}

impl<P: Differentiable + ?Sized> Order<P> for WithGradient {
    const GRADIENT: bool = true;
    const HESSIAN: bool = false;

    fn eval(
        objective: &mut P,
        x: &[f64],
        grad: &mut [f64],
        _: &mut [f64],
    ) -> Result<f64, GprError> {
        objective.value_and_gradient_into(x, grad)
    }
}

impl<P: TwiceDifferentiable + ?Sized> Order<P> for WithHessian {
    const GRADIENT: bool = true;
    const HESSIAN: bool = true;

    fn eval(
        objective: &mut P,
        x: &[f64],
        grad: &mut [f64],
        hess: &mut [f64],
    ) -> Result<f64, GprError> {
        objective.value_gradient_hessian_into(x, grad, hess)
    }
}

/// The objective with the buffers of its last evaluation, so the cost,
/// gradient, and Hessian argmin asks for at one point cost one call.
pub(super) struct EvalCache<'a, P: ?Sized, O> {
    objective: &'a mut P,
    params: Vec<f64>,
    value: Option<f64>,
    grad: Vec<f64>,
    hess: Vec<f64>,
    order: PhantomData<O>,
}

impl<'a, P: ?Sized, O: Order<P>> EvalCache<'a, P, O> {
    pub(super) fn new(objective: &'a mut P, n: usize) -> Self {
        Self {
            objective,
            params: Vec::new(),
            value: None,
            grad: vec![0.0; if O::GRADIENT { n } else { 0 }],
            hess: vec![0.0; if O::HESSIAN { n * n } else { 0 }],
            order: PhantomData,
        }
    }

    /// Evaluates at `x` unless `x` is the cached point.
    ///
    /// A point that cannot be evaluated is cached as the barrier
    /// ([`BARRIER_COST`], gradient and Hessian zero), and its error is
    /// returned: the objective's own error, or
    /// [`GprError::OptimizationNotConverged`] for a value, gradient, or
    /// Hessian that is not finite. A cached point returns its value.
    pub(super) fn try_eval(&mut self, x: &[f64]) -> Result<f64, GprError> {
        if let Some(value) = self.value
            && same_params(&self.params, x)
        {
            return Ok(value);
        }
        if O::GRADIENT && self.grad.len() != x.len() {
            self.grad.resize(x.len(), 0.0);
        }
        if O::HESSIAN && self.hess.len() != x.len() * x.len() {
            self.hess.resize(x.len() * x.len(), 0.0);
        }
        self.params.clear();
        self.params.extend_from_slice(x);
        let result = O::eval(self.objective, x, &mut self.grad, &mut self.hess);
        let failure = match result {
            Ok(value)
                if value.is_finite()
                    && self.grad.iter().all(|g| g.is_finite())
                    && self.hess.iter().all(|h| h.is_finite()) =>
            {
                self.value = Some(value);
                return Ok(value);
            }
            Ok(_) => GprError::OptimizationNotConverged { iterations: 0 },
            Err(err) => err,
        };
        self.grad.fill(0.0);
        self.hess.fill(0.0);
        self.value = Some(BARRIER_COST);
        Err(failure)
    }

    /// [`Self::try_eval`], with the barrier for a point that cannot be
    /// evaluated.
    fn cost(&mut self, x: &[f64]) -> f64 {
        self.try_eval(x).unwrap_or(BARRIER_COST)
    }
}

/// The argmin problem over an [`EvalCache`].
pub(super) struct CachedProblem<'a, P: ?Sized, O> {
    inner: RefCell<EvalCache<'a, P, O>>,
}

impl<'a, P: ?Sized, O: Order<P>> CachedProblem<'a, P, O> {
    pub(super) fn new(cache: EvalCache<'a, P, O>) -> Self {
        Self {
            inner: RefCell::new(cache),
        }
    }
}

impl<P: ?Sized, O: Order<P>> CostFunction for CachedProblem<'_, P, O> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, ArgminError> {
        Ok(self.inner.borrow_mut().cost(param))
    }
}

/// The orders whose evaluation writes a gradient.
pub(super) trait HasGradient {}
impl HasGradient for WithGradient {}
impl HasGradient for WithHessian {}

impl<P: ?Sized, O: Order<P> + HasGradient> Gradient for CachedProblem<'_, P, O> {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, ArgminError> {
        let mut inner = self.inner.borrow_mut();
        inner.cost(param);
        Ok(inner.grad.clone())
    }
}

impl<P: ?Sized> Hessian for CachedProblem<'_, P, WithHessian>
where
    WithHessian: Order<P>,
{
    type Param = Vec<f64>;
    type Hessian = Vec<Vec<f64>>;

    fn hessian(&self, param: &Self::Param) -> Result<Self::Hessian, ArgminError> {
        let mut inner = self.inner.borrow_mut();
        inner.cost(param);
        let n = param.len();
        Ok(inner.hess.chunks(n).map(<[f64]>::to_vec).collect())
    }
}

/// The best point of an argmin run, its recorded cost, and the iterations.
pub(super) struct RunEnd {
    pub(super) params: Vec<f64>,
    pub(super) cost: f64,
    pub(super) iterations: u64,
}

/// Runs `solver` on `problem` from `init` for at most `max_iters` iterations.
///
/// # Errors
///
/// Returns the mapped argmin error ([`map_argmin_error`]), or
/// [`GprError::OptimizationNotConverged`] when the run has no best point.
pub(super) fn run_argmin<O, S, G, J, H>(
    problem: O,
    solver: S,
    init: &[f64],
    max_iters: u64,
) -> Result<RunEnd, GprError>
where
    S: Solver<O, IterState<Vec<f64>, G, J, H, (), f64>>,
    IterState<Vec<f64>, G, J, H, (), f64>: State<Param = Vec<f64>, Float = f64>,
{
    let result = Executor::new(problem, solver)
        .configure(|state| state.param(init.to_vec()).max_iters(max_iters))
        .ctrlc(false)
        .run()
        .map_err(map_argmin_error)?;
    let state = result.state();
    let iterations = state.get_iter();
    let params = state
        .get_best_param()
        .cloned()
        .ok_or(GprError::OptimizationNotConverged {
            iterations: iterations as usize,
        })?;
    Ok(RunEnd {
        params,
        cost: state.get_best_cost(),
        iterations,
    })
}

/// The value of an argmin run at its best point, from the run's own record.
///
/// The solver evaluated that point already, so it is not evaluated again. A
/// run whose best is the barrier (every point it tried failed) has no
/// result: `evaluate` runs at that point once more so the model's own error
/// (for example an unsupported gradient) is returned instead of a generic
/// failure to converge.
pub(super) fn best_value(
    cost: f64,
    iterations: u64,
    evaluate: impl FnOnce() -> Result<f64, GprError>,
) -> Result<f64, GprError> {
    if cost.is_finite() && cost < BARRIER_COST {
        return Ok(cost);
    }
    evaluate()?;
    Err(GprError::OptimizationNotConverged {
        iterations: iterations as usize,
    })
}

/// Runs `run` on `objective` mapped through the logit of `intervals`, from
/// `init_z`, and keeps the result in `best` when it is lower.
pub(super) fn consider_logit_run<P, F>(
    objective: &mut P,
    intervals: &[Interval],
    init_z: &[f64],
    best: &mut Option<OptResult>,
    run: F,
) -> Result<(), GprError>
where
    P: Objective,
    F: FnOnce(&mut LogitMapped<'_, P>, &[f64]) -> Result<OptResult, GprError>,
{
    let mut mapped = LogitMapped {
        inner: objective,
        intervals,
        log_scratch: vec![0.0; init_z.len()],
    };
    let run = run(&mut mapped, init_z)?;
    keep_better(
        best,
        OptResult {
            params: z_to_log_theta(&run.params, intervals)?,
            value: run.value,
            iterations: run.iterations,
        },
    );
    Ok(())
}

/// Keeps the lower of `best` and `candidate`.
///
/// A candidate whose value is not finite is never kept, so one run that ends
/// at `NaN` or `±∞` cannot block a later finite run. When no run is finite,
/// `best` stays `None` and the caller reports no result.
pub(super) fn keep_better(best: &mut Option<OptResult>, candidate: OptResult) {
    if !candidate.value.is_finite() {
        return;
    }
    match best {
        None => *best = Some(candidate),
        Some(current) if candidate.value < current.value => *best = Some(candidate),
        Some(_) => {}
    }
}

fn same_params(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// An argmin error from a run: the objective's own [`GprError`] keeps its
/// type; any other failure of the solver (a line search or subproblem that
/// cannot go on) is a failure to make progress.
pub(super) fn map_argmin_error(err: ArgminError) -> GprError {
    match err.downcast_ref::<GprError>() {
        Some(gpr) => gpr.clone(),
        None => GprError::OptimizationNotConverged { iterations: 0 },
    }
}

/// An argmin error from building a solver: its settings are rejected.
pub(super) fn solver_config_error(err: ArgminError) -> GprError {
    GprError::InvalidConfig {
        reason: err.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BARRIER_COST, CachedProblem, CostFunction, EvalCache, Gradient, Hessian, ValueOnly,
        WithGradient, WithHessian, keep_better,
    };
    use crate::error::GprError;
    use crate::objective::{Differentiable, Objective, TwiceDifferentiable};
    use crate::optimizer::OptResult;
    use crate::test_check::assert_close;

    const TOL: f64 = 1e-6;

    /// `½ ‖x‖²`, counting joint calls; fails where `x[0] > 10`.
    struct Quadratic {
        joint_evals: usize,
    }

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
            self.joint_evals += 1;
            if params[0] > 10.0 {
                return Err(GprError::NonFiniteInput);
            }
            out[0] = params[0];
            out[1] = params[1];
            Ok(0.5 * (params[0] * params[0] + params[1] * params[1]))
        }
    }

    impl TwiceDifferentiable for Quadratic {
        fn hessian_into(&mut self, _params: &[f64], out: &mut [f64]) -> Result<(), GprError> {
            out.copy_from_slice(&[1.0, 0.0, 0.0, 1.0]);
            Ok(())
        }

        fn value_gradient_hessian_into(
            &mut self,
            params: &[f64],
            grad: &mut [f64],
            hess: &mut [f64],
        ) -> Result<f64, GprError> {
            let value = self.value_and_gradient_into(params, grad)?;
            self.hessian_into(params, hess)?;
            Ok(value)
        }
    }

    #[test]
    fn shared_point_uses_one_joint_eval() {
        let mut obj = Quadratic { joint_evals: 0 };
        let param = vec![0.3, -0.2];
        {
            let problem = CachedProblem::new(EvalCache::<_, WithGradient>::new(&mut obj, 2));
            let cost = problem.cost(&param).expect("cost");
            let grad = problem.gradient(&param).expect("grad");
            assert_close(cost, 0.5 * (0.3 * 0.3 + 0.2 * 0.2), TOL);
            assert_close(grad[0], 0.3, TOL);
            assert_close(grad[1], -0.2, TOL);
        }
        assert_eq!(obj.joint_evals, 1);
        {
            let problem = CachedProblem::new(EvalCache::<_, WithGradient>::new(&mut obj, 2));
            let other = vec![-1.0, 0.5];
            let _ = problem.cost(&param).expect("cost");
            let _ = problem.gradient(&param).expect("grad");
            let _ = problem.cost(&other).expect("other");
            let _ = problem.gradient(&other).expect("other grad");
        }
        assert_eq!(obj.joint_evals, 3);
    }

    /// Every order treats a point that cannot be evaluated the same way:
    /// the barrier cost, a zero gradient and Hessian, cached, and the
    /// objective's error from `try_eval`.
    #[test]
    fn a_failed_point_is_the_same_barrier_for_every_order() {
        let bad = vec![11.0, 0.0];
        let mut obj = Quadratic { joint_evals: 0 };
        {
            let problem = CachedProblem::new(EvalCache::<_, ValueOnly>::new(&mut obj, 2));
            assert_eq!(
                problem.cost(&bad).expect("cost").to_bits(),
                BARRIER_COST.to_bits()
            );
        }
        {
            let problem = CachedProblem::new(EvalCache::<_, WithGradient>::new(&mut obj, 2));
            assert_eq!(
                problem.cost(&bad).expect("cost").to_bits(),
                BARRIER_COST.to_bits()
            );
            assert_eq!(problem.gradient(&bad).expect("grad"), vec![0.0, 0.0]);
        }
        {
            let problem = CachedProblem::new(EvalCache::<_, WithHessian>::new(&mut obj, 2));
            assert_eq!(
                problem.cost(&bad).expect("cost").to_bits(),
                BARRIER_COST.to_bits()
            );
            assert_eq!(problem.gradient(&bad).expect("grad"), vec![0.0, 0.0]);
            assert_eq!(
                problem.hessian(&bad).expect("hess"),
                vec![vec![0.0, 0.0], vec![0.0, 0.0]]
            );
        }
        // One call per order: the barrier is cached.
        assert_eq!(obj.joint_evals, 3);
        let mut cache = EvalCache::<_, WithHessian>::new(&mut obj, 2);
        assert_eq!(cache.try_eval(&bad), Err(GprError::NonFiniteInput));
    }

    fn run(value: f64) -> OptResult {
        OptResult {
            params: vec![value],
            value,
            iterations: 1,
        }
    }

    #[test]
    fn keep_better_skips_non_finite_runs() {
        let mut best = None;
        keep_better(&mut best, run(f64::NAN));
        assert!(best.is_none(), "a NaN run must not become the result");
        keep_better(&mut best, run(f64::INFINITY));
        assert!(best.is_none(), "an infinite run must not become the result");
        keep_better(&mut best, run(2.0));
        keep_better(&mut best, run(f64::NAN));
        keep_better(&mut best, run(f64::NEG_INFINITY));
        keep_better(&mut best, run(3.0));
        keep_better(&mut best, run(1.0));
        let kept = best.expect("a finite run");
        assert_close(kept.value, 1.0, TOL);
    }
}
