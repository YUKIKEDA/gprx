//! argmin L-BFGS adapter and optimizer type slot.
//!
//! Does not implement L-BFGS itself. [`Lbfgs::minimize`] maps user-unit
//! [`crate::Interval`] through a logit so argmin stays unconstrained. Positive
//! intervals (`lo > 0`) use a log-uniform logit, matching restart sampling.
//! The unconstrained coordinate is scaled so the Jacobian is 1 at the
//! interval midpoint. When argmin asks for cost and gradient at the same
//! point, one [`crate::Differentiable::value_and_gradient_into`] call fills
//! both.

use std::cell::RefCell;
use std::num::{NonZeroU32, NonZeroUsize};

use argmin::core::{CostFunction, Error as ArgminError, Executor, Gradient, State};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;

use crate::error::GprError;
use crate::objective::{Differentiable, HasBounds, Objective};
use crate::param::Interval;

/// Result of [`Optimizer::minimize`].
#[derive(Clone, Debug)]
pub struct OptResult {
    /// Parameters in the same space as `init` (log-`θ` for [`crate::Gpr`]).
    pub params: Vec<f64>,
    /// Objective value at [`Self::params`].
    pub value: f64,
    /// Optimizer iterations performed.
    pub iterations: u64,
}

/// Hyperparameter optimizer.
///
/// `P` is the objective this algorithm can minimize. [`Lbfgs`] requires
/// [`Differentiable`] plus bounds. A derivative-free solver (P2B-2) requires
/// only [`Objective`].
pub trait Optimizer<P: ?Sized> {
    /// Minimizes `objective` from `init` without taking ownership of `init`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError`] when `init` is the wrong length, the objective
    /// fails, or the solver stops without a best parameter vector.
    fn minimize(&self, objective: &mut P, init: &[f64]) -> Result<OptResult, GprError>;
}

/// Marker for an optimizer that consumes changed-parameter indices.
///
/// [`Lbfgs`] does not implement this. [`crate::Gpr::with_recompute_strategy`]
/// to [`IncrementalRecompute`] exists only when `O: UsesChangeIndices`.
pub trait UsesChangeIndices {}

/// Marker for how kernel matrices are rebuilt during fit.
pub trait RecomputeStrategy:
    Copy + Clone + core::fmt::Debug + Default + Send + Sync + 'static
{
}

/// Bound on [`crate::Gpr::with_recompute_strategy`].
///
/// [`FullRecompute`] is valid for every optimizer. [`IncrementalRecompute`]
/// is valid only when `O: `[`UsesChangeIndices`].
pub trait AcceptsRecompute<O>: RecomputeStrategy {}

/// Always rebuild the full kernel matrix. This is the default.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FullRecompute;

impl RecomputeStrategy for FullRecompute {}

impl<O> AcceptsRecompute<O> for FullRecompute {}

/// Rebuild only leaves touched by changed indices. Body is P2B-18.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct IncrementalRecompute;

impl RecomputeStrategy for IncrementalRecompute {}

impl<O: UsesChangeIndices> AcceptsRecompute<O> for IncrementalRecompute {}

/// Fixed hyperparameters. [`crate::Gpr<Fixed>::factor`] only; not an [`Optimizer`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Fixed;

/// Limited-memory BFGS via argmin.
#[derive(Clone, Debug, PartialEq)]
pub struct Lbfgs {
    max_iterations: u64,
    tolerance: f64,
    history_size: NonZeroUsize,
    restarts: Option<Restarts>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Restarts {
    n: NonZeroU32,
    seed: u64,
}

impl Default for Lbfgs {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: f64::EPSILON.sqrt(),
            history_size: match NonZeroUsize::new(10) {
                Some(n) => n,
                None => NonZeroUsize::MIN,
            },
            restarts: None,
        }
    }
}

impl Lbfgs {
    /// Builds L-BFGS with 100 iterations, gradient tolerance `sqrt(ε)`, and
    /// history 10.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::num::{NonZeroU32, NonZeroUsize};
    /// use gprx::Lbfgs;
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let _lbfgs = Lbfgs::new()
    ///     .with_max_iterations(50)
    ///     .with_tolerance(1e-8)?
    ///     .with_history_size(NonZeroUsize::MIN)
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

    /// Sets the gradient-norm tolerance (`with_tolerance_grad`, default `sqrt(ε)`).
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if `tolerance` is not finite
    /// or is negative.
    pub fn with_tolerance(mut self, tolerance: f64) -> Result<Self, GprError> {
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(GprError::InvalidHyperparameter {
                reason: "L-BFGS gradient tolerance must be finite and >= 0".to_owned(),
            });
        }
        self.tolerance = tolerance;
        Ok(self)
    }

    /// Sets the L-BFGS history size `m` (default 10).
    pub fn with_history_size(mut self, history_size: NonZeroUsize) -> Self {
        self.history_size = history_size;
        self
    }

    /// Adds `n` extra log-uniform starts (`n ≥ 1`) and keeps the lowest NLML.
    ///
    /// The first start is the model `θ`. Failed extra starts are discarded.
    /// The default trainer has no restarts and no seed.
    pub fn with_restarts(mut self, n: NonZeroU32, seed: u64) -> Self {
        self.restarts = Some(Restarts { n, seed });
        self
    }

    #[cfg(test)]
    pub(crate) fn minimize_unconstrained<P: Differentiable>(
        &self,
        objective: &mut P,
        init: &[f64],
    ) -> Result<OptResult, GprError> {
        run_lbfgs(self, objective, init)
    }
}

impl<P: Differentiable + HasBounds> Optimizer<P> for Lbfgs {
    fn minimize(&self, objective: &mut P, init: &[f64]) -> Result<OptResult, GprError> {
        let n = objective.num_params();
        if init.len() != n {
            return Err(GprError::InvalidHyperparameter {
                reason: format!("expected {n} parameters, got {}", init.len()),
            });
        }
        let mut intervals = vec![Interval::SKLEARN_POSITIVE; n];
        objective.fill_intervals(&mut intervals)?;
        let mut best: Option<OptResult> = None;
        let first_z = log_theta_to_z(init, &intervals)?;
        consider_run(self, objective, &intervals, &first_z, &mut best)?;
        if let Some(restarts) = self.restarts {
            let mut rng = restarts.seed;
            for _ in 0..restarts.n.get() {
                let z = sample_log_uniform_z(&intervals, &mut rng)?;
                let _ = consider_run(self, objective, &intervals, &z, &mut best);
            }
        }
        best.ok_or(GprError::OptimizationNotConverged { iterations: 0 })
    }
}

fn consider_run<P: Differentiable>(
    lbfgs: &Lbfgs,
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
    let run = run_lbfgs(lbfgs, &mut mapped, init_z)?;
    let log_theta = z_to_log_theta(&run.params, intervals)?;
    let mut grad = vec![0.0; log_theta.len()];
    let value = objective.value_and_gradient_into(&log_theta, &mut grad)?;
    let candidate = OptResult {
        params: log_theta,
        value,
        iterations: run.iterations,
    };
    match best {
        None => *best = Some(candidate),
        Some(current) if candidate.value < current.value => *best = Some(candidate),
        Some(_) => {}
    }
    Ok(())
}

fn run_lbfgs<P: Differentiable>(
    lbfgs: &Lbfgs,
    objective: &mut P,
    init: &[f64],
) -> Result<OptResult, GprError> {
    let n = objective.num_params();
    if init.len() != n {
        return Err(GprError::InvalidHyperparameter {
            reason: format!("expected {n} parameters, got {}", init.len()),
        });
    }
    let problem = CachedProblem {
        inner: RefCell::new(EvalCache::new(objective, n)),
    };
    let linesearch = MoreThuenteLineSearch::<Vec<f64>, Vec<f64>, f64>::new();
    let solver: LBFGS<_, Vec<f64>, Vec<f64>, f64> =
        LBFGS::new(linesearch, lbfgs.history_size.get())
            .with_tolerance_grad(lbfgs.tolerance)
            .map_err(map_argmin_error)?;
    let (params, iterations) = {
        let result = Executor::new(problem, solver)
            .configure(|state| state.param(init.to_vec()).max_iters(lbfgs.max_iterations))
            .ctrlc(false)
            .run()
            .map_err(map_argmin_error)?;
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

struct LogitMapped<'a, P> {
    inner: &'a mut P,
    intervals: &'a [Interval],
    log_scratch: Vec<f64>,
}

impl<P: Objective> Objective for LogitMapped<'_, P> {
    fn num_params(&self) -> usize {
        self.inner.num_params()
    }

    fn value(&mut self, params: &[f64]) -> Result<f64, GprError> {
        z_to_log_theta_into(params, self.intervals, &mut self.log_scratch)?;
        self.inner.value(&self.log_scratch)
    }
}

impl<P: Differentiable> Differentiable for LogitMapped<'_, P> {
    fn gradient_into(&mut self, params: &[f64], out: &mut [f64]) -> Result<(), GprError> {
        self.value_and_gradient_into(params, out).map(|_| ())
    }

    fn value_and_gradient_into(
        &mut self,
        params: &[f64],
        out: &mut [f64],
    ) -> Result<f64, GprError> {
        z_to_log_theta_into(params, self.intervals, &mut self.log_scratch)?;
        let value = self.inner.value_and_gradient_into(&self.log_scratch, out)?;
        chain_logit_grad(params, self.intervals, &self.log_scratch, out);
        Ok(value)
    }
}

fn log_theta_to_z(log_theta: &[f64], intervals: &[Interval]) -> Result<Vec<f64>, GprError> {
    let mut z = vec![0.0; log_theta.len()];
    for (i, &theta) in log_theta.iter().enumerate() {
        let x = theta.exp();
        z[i] = user_to_z(x, intervals[i])?;
    }
    Ok(z)
}

fn z_to_log_theta(z: &[f64], intervals: &[Interval]) -> Result<Vec<f64>, GprError> {
    let mut log_theta = vec![0.0; z.len()];
    z_to_log_theta_into(z, intervals, &mut log_theta)?;
    Ok(log_theta)
}

fn z_to_log_theta_into(z: &[f64], intervals: &[Interval], out: &mut [f64]) -> Result<(), GprError> {
    if z.len() != intervals.len() || out.len() != z.len() {
        return Err(GprError::InvalidHyperparameter {
            reason: "logit map length mismatch".to_owned(),
        });
    }
    for i in 0..z.len() {
        let x = z_to_user(z[i], intervals[i]);
        out[i] = x.ln();
    }
    Ok(())
}

fn chain_logit_grad(z: &[f64], intervals: &[Interval], log_theta: &[f64], grad: &mut [f64]) {
    for i in 0..z.len() {
        let scale = logit_scale(intervals[i]);
        let t = z[i] / scale;
        let s = sigmoid(t);
        let ds = s * (1.0 - s);
        let dlog_dt = if intervals[i].lo() > 0.0 {
            (intervals[i].hi().ln() - intervals[i].lo().ln()) * ds
        } else {
            let x = log_theta[i].exp();
            intervals[i].width() * ds / x
        };
        grad[i] *= dlog_dt / scale;
    }
}

fn user_to_z(x: f64, interval: Interval) -> Result<f64, GprError> {
    if !interval.contains(x) {
        return Err(GprError::from(crate::param::IntervalError::OutOfRange {
            value: x,
            lo: interval.lo(),
            hi: interval.hi(),
        }));
    }
    let u = unit_from_user(x, interval);
    Ok(logit(u.clamp(f64::EPSILON, 1.0 - f64::EPSILON)) * logit_scale(interval))
}

fn z_to_user(z: f64, interval: Interval) -> f64 {
    let t = z / logit_scale(interval);
    let u = sigmoid(t);
    if interval.lo() > 0.0 {
        let ln_lo = interval.lo().ln();
        let ln_hi = interval.hi().ln();
        (ln_lo + u * (ln_hi - ln_lo)).exp()
    } else {
        interval.lo() + u * interval.width()
    }
}

fn logit_scale(interval: Interval) -> f64 {
    let span = if interval.lo() > 0.0 {
        interval.hi().ln() - interval.lo().ln()
    } else {
        interval.width()
    };
    span / 4.0
}

fn unit_from_user(x: f64, interval: Interval) -> f64 {
    if interval.lo() > 0.0 {
        let ln_lo = interval.lo().ln();
        let ln_hi = interval.hi().ln();
        (x.ln() - ln_lo) / (ln_hi - ln_lo)
    } else {
        (x - interval.lo()) / interval.width()
    }
}

fn logit(u: f64) -> f64 {
    (u / (1.0 - u)).ln()
}

fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        let e = (-z).exp();
        1.0 / (1.0 + e)
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

fn sample_log_uniform_z(intervals: &[Interval], rng: &mut u64) -> Result<Vec<f64>, GprError> {
    let mut z = vec![0.0; intervals.len()];
    for (slot, interval) in z.iter_mut().zip(intervals.iter().copied()) {
        let x = log_uniform_open(*rng, interval);
        *rng = splitmix64_state(*rng);
        *slot = user_to_z(x, interval)?;
    }
    Ok(z)
}

fn log_uniform_open(state: u64, interval: Interval) -> f64 {
    let u = open_unit(state);
    if interval.lo() > 0.0 {
        let ln_lo = interval.lo().ln();
        let ln_hi = interval.hi().ln();
        (ln_lo + u * (ln_hi - ln_lo)).exp()
    } else {
        interval.lo() + u * interval.width()
    }
}

fn open_unit(state: u64) -> f64 {
    let u = splitmix64_f64(state);
    let eps = 1.0 / ((1u64 << 53) as f64);
    if u <= eps {
        eps
    } else if u >= 1.0 - eps {
        1.0 - eps
    } else {
        u
    }
}

fn splitmix64_state(state: u64) -> u64 {
    state.wrapping_add(0x9E3779B97F4A7C15)
}

fn splitmix64_f64(state: u64) -> f64 {
    let mut z = splitmix64_state(state);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^= z >> 31;
    (z >> 11) as f64 / ((1u64 << 53) as f64)
}

/// Cost returned to argmin when a trial point is non-finite or rejected (for
/// example outside an open [`Interval`]). More–Thuente can then backtrack
/// instead of aborting the whole solve.
const BARRIER_COST: f64 = 1.0e300;

struct EvalCache<'a, P: ?Sized> {
    objective: &'a mut P,
    params: Vec<f64>,
    value: Option<f64>,
    grad: Vec<f64>,
}

impl<'a, P: Differentiable + ?Sized> EvalCache<'a, P> {
    fn new(objective: &'a mut P, n: usize) -> Self {
        Self {
            objective,
            params: Vec::new(),
            value: None,
            grad: vec![0.0; n],
        }
    }

    fn eval(&mut self, param: &[f64]) -> Result<f64, GprError> {
        if let Some(value) = self.value {
            if same_params(&self.params, param) {
                return Ok(value);
            }
        }
        if self.grad.len() != param.len() {
            self.grad.resize(param.len(), 0.0);
        }
        match self
            .objective
            .value_and_gradient_into(param, &mut self.grad)
        {
            Ok(value) if value.is_finite() && self.grad.iter().all(|g| g.is_finite()) => {
                self.params.clear();
                self.params.extend_from_slice(param);
                self.value = Some(value);
                Ok(value)
            }
            Ok(_) | Err(_) => {
                self.params.clear();
                self.params.extend_from_slice(param);
                self.grad.fill(0.0);
                self.value = Some(BARRIER_COST);
                Ok(BARRIER_COST)
            }
        }
    }
}

struct CachedProblem<'a, P: ?Sized> {
    inner: RefCell<EvalCache<'a, P>>,
}

impl<P: Differentiable + ?Sized> CostFunction for CachedProblem<'_, P> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, ArgminError> {
        self.inner
            .borrow_mut()
            .eval(param)
            .map_err(ArgminError::from)
    }
}

impl<P: Differentiable + ?Sized> Gradient for CachedProblem<'_, P> {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, ArgminError> {
        let mut inner = self.inner.borrow_mut();
        inner.eval(param).map_err(ArgminError::from)?;
        Ok(inner.grad.clone())
    }
}

fn same_params(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn map_argmin_error(err: ArgminError) -> GprError {
    if let Some(gpr) = err.downcast_ref::<GprError>() {
        return gpr.clone();
    }
    GprError::InvalidHyperparameter {
        reason: err.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CachedProblem, CostFunction, EvalCache, Fixed, Gradient, Lbfgs, Optimizer, logit, sigmoid,
    };
    use crate::error::GprError;
    use crate::gpr::Gpr;
    use crate::kernel::{KernelSpec, RbfArdKernel, RbfKernel};
    use crate::likelihood::GaussianLikelihood;
    use crate::objective::{Differentiable, Objective};
    use crate::param::Interval;
    use std::cell::RefCell;

    const TOL: f64 = 1e-6;

    fn assert_close(actual: f64, expected: f64) {
        let scale = expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= TOL * scale,
            "actual={actual}, expected={expected}"
        );
    }

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
            if params.len() != 2 || out.len() != 2 {
                return Err(GprError::InvalidHyperparameter {
                    reason: "quadratic is 2-D".to_owned(),
                });
            }
            self.joint_evals += 1;
            out[0] = params[0];
            out[1] = params[1];
            Ok(0.5 * (params[0] * params[0] + params[1] * params[1]))
        }
    }

    #[test]
    fn lbfgs_rejects_wrong_init_len() {
        let mut obj = Quadratic { joint_evals: 0 };
        let err = Lbfgs::new()
            .minimize_unconstrained(&mut obj, &[0.0])
            .expect_err("len");
        assert!(matches!(err, GprError::InvalidHyperparameter { .. }));
        assert_eq!(obj.joint_evals, 0);
    }

    #[test]
    fn lbfgs_minimizes_quadratic_bowl() {
        let mut obj = Quadratic { joint_evals: 0 };
        let result = Lbfgs::new()
            .with_max_iterations(50)
            .minimize_unconstrained(&mut obj, &[1.0, -0.5])
            .expect("bowl");
        assert_close(result.params[0], 0.0);
        assert_close(result.params[1], 0.0);
        assert!(result.value < 1e-10);
        assert!(obj.joint_evals > 0);
    }

    #[test]
    fn shared_point_uses_one_joint_eval() {
        let mut obj = Quadratic { joint_evals: 0 };
        let param = vec![0.3, -0.2];
        {
            let problem = CachedProblem {
                inner: RefCell::new(EvalCache::new(&mut obj, 2)),
            };
            let cost = problem.cost(&param).expect("cost");
            let grad = problem.gradient(&param).expect("grad");
            assert_close(cost, 0.5 * (0.3 * 0.3 + 0.2 * 0.2));
            assert_close(grad[0], 0.3);
            assert_close(grad[1], -0.2);
        }
        assert_eq!(obj.joint_evals, 1);
        {
            let problem = CachedProblem {
                inner: RefCell::new(EvalCache::new(&mut obj, 2)),
            };
            let other = vec![-1.0, 0.5];
            let _ = problem.cost(&param).expect("cost");
            let _ = problem.gradient(&param).expect("grad");
            let _ = problem.cost(&other).expect("other");
            let _ = problem.gradient(&other).expect("other grad");
        }
        assert_eq!(obj.joint_evals, 3);
    }

    #[test]
    fn logit_mapped_grad_matches_finite_difference() {
        let kernel = KernelSpec::from(RbfKernel::new(1.0).expect("valid"));
        let likelihood = GaussianLikelihood::new(0.1).expect("valid");
        let mut gpr = Gpr::new(kernel, likelihood)
            .with_optimizer(Fixed)
            .factor(&[0.0, 1.0], 2, 1, &[0.5, -0.25])
            .unwrap_or_else(|(_, e)| panic!("{e}"));
        let mut log_theta = [0.0; 2];
        gpr.get_params(&mut log_theta).expect("len 2");
        let intervals = [Interval::SKLEARN_POSITIVE, Interval::SKLEARN_POSITIVE];
        let z = super::log_theta_to_z(&log_theta, &intervals).expect("z");
        let mut obj = gpr.objective();
        let mut mapped = super::LogitMapped {
            inner: &mut obj,
            intervals: &intervals,
            log_scratch: vec![0.0; 2],
        };
        let mut analytic = [0.0; 2];
        let value = mapped
            .value_and_gradient_into(&z, &mut analytic)
            .expect("grad");
        let h = 1e-6;
        for i in 0..2 {
            let mut plus = z.clone();
            let mut minus = z.clone();
            plus[i] += h;
            minus[i] -= h;
            let vp = mapped.value(&plus).expect("plus");
            let vm = mapped.value(&minus).expect("minus");
            let fd = (vp - vm) / (2.0 * h);
            assert!(
                (analytic[i] - fd).abs() <= 1e-4 * analytic[i].abs().max(1.0),
                "i={i} analytic={} fd={} value={value}",
                analytic[i],
                fd
            );
        }
    }

    #[test]
    fn logit_roundtrip_stays_inside_interval() {
        let interval = Interval::SKLEARN_POSITIVE;
        let x = 2.5;
        let z = super::user_to_z(x, interval).expect("inside");
        let back = super::z_to_user(z, interval);
        assert_close(back, x);
        assert!(interval.contains(back));
        assert_close(sigmoid(logit(0.25)), 0.25);
    }

    #[test]
    fn lbfgs_lowers_gpr_nlml() {
        let kernel = KernelSpec::from(RbfKernel::new(1.0).expect("valid"));
        let likelihood = GaussianLikelihood::new(0.1).expect("valid");
        let mut gpr = Gpr::new(kernel, likelihood)
            .with_optimizer(Fixed)
            .factor(&[0.0, 1.0], 2, 1, &[0.5, -0.25])
            .unwrap_or_else(|(_, e)| panic!("{e}"));
        let mut init = [0.0; 2];
        gpr.get_params(&mut init).expect("len 2");
        init[0] = 2.0_f64.ln();
        let start;
        let result;
        {
            let mut obj = gpr.objective();
            start = obj.value(&init).expect("start");
            result = Lbfgs::new()
                .with_max_iterations(40)
                .minimize(&mut obj, &init)
                .expect("lbfgs");
        }
        assert!(
            result.value <= start + 1e-9,
            "start={start}, best={}",
            result.value
        );
        let mut got = [0.0; 2];
        gpr.get_params(&mut got).expect("len 2");
        assert_close(got[0], result.params[0]);
        assert_close(got[1], result.params[1]);
    }

    struct CountingObj<'a, O, S> {
        inner: crate::objective::GprObjective<'a, O, S>,
        joint_evals: usize,
    }

    impl<O, S> Objective for CountingObj<'_, O, S> {
        fn num_params(&self) -> usize {
            self.inner.num_params()
        }

        fn value(&mut self, params: &[f64]) -> Result<f64, GprError> {
            let mut dummy = vec![0.0; self.num_params()];
            self.value_and_gradient_into(params, &mut dummy)
        }
    }

    impl<O, S> Differentiable for CountingObj<'_, O, S> {
        fn gradient_into(&mut self, params: &[f64], out: &mut [f64]) -> Result<(), GprError> {
            self.value_and_gradient_into(params, out).map(|_| ())
        }

        fn value_and_gradient_into(
            &mut self,
            params: &[f64],
            out: &mut [f64],
        ) -> Result<f64, GprError> {
            self.joint_evals += 1;
            self.inner.value_and_gradient_into(params, out)
        }
    }

    impl<O, S> crate::objective::HasBounds for CountingObj<'_, O, S> {
        fn fill_intervals(&self, out: &mut [crate::param::Interval]) -> Result<(), GprError> {
            self.inner.fill_intervals(out)
        }
    }

    fn splitmix64(state: &mut u64) -> f64 {
        *state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        (z >> 11) as f64 / ((1u64 << 53) as f64)
    }

    fn forrester_bench_xy() -> (Vec<f64>, Vec<f64>) {
        const N: usize = 256;
        let x: Vec<f64> = (0..N).map(|i| i as f64 / (N - 1) as f64).collect();
        let mut state = 0u64;
        let y: Vec<f64> = x
            .iter()
            .map(|&xi| {
                let t = 6.0 * xi - 2.0;
                let u1 = splitmix64(&mut state).max(f64::MIN_POSITIVE);
                let u2 = splitmix64(&mut state);
                let noise = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
                t * t * (12.0 * xi - 4.0).sin() + noise
            })
            .collect();
        (x, y)
    }

    /// The `fit_lbfgs` bench problem must stay a peaked landscape. Independent
    /// random `y` walked a flat ridge and spent ~291 joint evals.
    #[test]
    fn forrester_bench_lbfgs_eval_count_is_bounded() {
        let (x, y) = forrester_bench_xy();
        let kernel = KernelSpec::from(RbfKernel::new(1.0).expect("valid"));
        let likelihood = GaussianLikelihood::new(0.1).expect("valid");
        let mut gpr = Gpr::new(kernel, likelihood)
            .with_target_transform(crate::transform::StandardizeTarget::new())
            .with_optimizer(Fixed)
            .factor(&x, 256, 1, &y)
            .unwrap_or_else(|(_, e)| panic!("{e}"));
        let mut init = vec![0.0; gpr.num_params()];
        gpr.get_params(&mut init).expect("len");
        let mut start_grad = vec![0.0; init.len()];
        let start = gpr
            .value_and_gradient_into(&init, &mut start_grad)
            .expect("start");
        let (result, evals) = {
            let mut obj = CountingObj {
                inner: gpr.objective(),
                joint_evals: 0,
            };
            let result = Lbfgs::new()
                .with_max_iterations(100)
                .minimize(&mut obj, &init)
                .expect("lbfgs");
            (result, obj.joint_evals)
        };
        eprintln!(
            "forrester_bench_lbfgs_eval_count iters={} evals={} start={:.6} value={:.6} init={:?} grad={:?} best={:?}",
            result.iterations, evals, start, result.value, init, start_grad, result.params
        );
        assert!(
            (8..=80).contains(&evals),
            "evals={evals}; Forrester should iterate, not walk a ridge"
        );
        assert!(
            (4..=40).contains(&result.iterations),
            "iters={}",
            result.iterations
        );
        assert!(
            result.value < start - 1.0,
            "start={start}, best={}",
            result.value
        );
    }

    fn sphere_bench_xy() -> (Vec<f64>, Vec<f64>) {
        const N: usize = 256;
        const SIDE: usize = 16;
        let mut x = vec![0.0; N * 2];
        let denom = (SIDE - 1) as f64;
        for row in 0..N {
            let i = row % SIDE;
            let j = row / SIDE;
            x[row] = i as f64 / denom;
            x[N + row] = j as f64 / denom;
        }
        let mut state = 0u64;
        let y: Vec<f64> = (0..N)
            .map(|row| {
                let a = x[row] / 0.25;
                let u1 = splitmix64(&mut state).max(f64::MIN_POSITIVE);
                let u2 = splitmix64(&mut state);
                let noise = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
                a * a + x[N + row] * x[N + row] + noise
            })
            .collect();
        (x, y)
    }

    fn sphere_lbfgs_evals(policy: crate::DistanceCachePolicy) -> (u64, usize, f64, f64) {
        let (x, y) = sphere_bench_xy();
        let kernel = KernelSpec::from(RbfArdKernel::new(&[4.0, 4.0]).expect("valid"));
        let likelihood = GaussianLikelihood::new(0.1).expect("valid");
        let mut gpr = Gpr::new(kernel, likelihood)
            .with_distance_cache_policy(policy)
            .with_target_transform(crate::transform::StandardizeTarget::new())
            .with_optimizer(Fixed)
            .factor(&x, 256, 2, &y)
            .unwrap_or_else(|(_, e)| panic!("{e}"));
        let mut init = vec![0.0; gpr.num_params()];
        gpr.get_params(&mut init).expect("len");
        let start = {
            let mut grad = vec![0.0; init.len()];
            gpr.value_and_gradient_into(&init, &mut grad)
                .expect("start")
        };
        let (result, evals) = {
            let mut obj = CountingObj {
                inner: gpr.objective(),
                joint_evals: 0,
            };
            let result = Lbfgs::new()
                .with_max_iterations(100)
                .minimize(&mut obj, &init)
                .expect("lbfgs");
            (result, obj.joint_evals)
        };
        (result.iterations, evals, start, result.value)
    }

    #[test]
    fn sphere_bench_lbfgs_eval_count_is_bounded() {
        let (iters_a, evals_a, start_a, value_a) =
            sphere_lbfgs_evals(crate::DistanceCachePolicy::Always);
        let (iters_n, evals_n, start_n, value_n) =
            sphere_lbfgs_evals(crate::DistanceCachePolicy::Never);
        eprintln!(
            "sphere_bench_lbfgs_eval_count always iters={iters_a} evals={evals_a} start={start_a:.6} value={value_a:.6}"
        );
        eprintln!(
            "sphere_bench_lbfgs_eval_count never iters={iters_n} evals={evals_n} start={start_n:.6} value={value_n:.6}"
        );
        assert!(
            (8..=90).contains(&evals_a),
            "evals={evals_a}; sphere should iterate, not walk a ridge"
        );
        assert!(
            (8..=90).contains(&evals_n),
            "evals={evals_n}; sphere should iterate, not walk a ridge"
        );
        assert!((4..=40).contains(&iters_a), "iters={iters_a}");
        assert!((4..=40).contains(&iters_n), "iters={iters_n}");
        assert_close(start_a, start_n);
        assert_close(value_a, value_n);
        assert!(value_a < start_a - 1.0, "start={start_a}, best={value_a}");
    }
}
