//! Trust-region optimizer (argmin `TrustRegion` with the Steihaug subproblem):
//! the solver that uses the analytic Hessian.

use std::cell::RefCell;
use std::num::NonZeroU32;

use argmin::core::{
    CostFunction, Error as ArgminError, Executor, Gradient, Hessian, IterState, KV, Problem,
    Solver, State, TerminationReason, TerminationStatus,
};
use argmin::solver::trustregion::{Steihaug, TrustRegion as ArgminTrustRegion};

use crate::error::GprError;
use crate::objective::TwiceDifferentiable;
use crate::param::Interval;

use super::logit::{
    LogitMapped, keep_better, log_theta_to_z, sample_log_uniform_z, z_to_log_theta,
};
use super::{OptResult, Optimizer, Restarts};

/// Cost of a point that cannot be evaluated (out of the bounds, not finite,
/// not positive definite): the ratio of actual to predicted reduction turns
/// negative and the region shrinks. The same barrier the L-BFGS and
/// nonlinear-CG adapters use.
const BARRIER_COST: f64 = 1.0e300;

/// Trust-region method with the Steihaug conjugate-gradient subproblem, via
/// argmin. It uses the analytic Hessian ([`crate::TwiceDifferentiable`]).
///
/// The step is the minimizer of the quadratic model inside a ball whose
/// radius grows and shrinks with how well the model predicted the decrease, so
/// an indefinite or singular Hessian and a step that leaves the bounds are
/// handled by the method: the region shrinks. The Hessian lives in
/// unconstrained logit coordinates. A run that hits the iteration cap returns
/// its best point.
#[derive(Clone, Debug, PartialEq)]
pub struct TrustRegion {
    max_iterations: u64,
    tolerance: f64,
    initial_radius: f64,
    max_radius: f64,
    restarts: Option<Restarts>,
}

impl Default for TrustRegion {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: f64::EPSILON.sqrt(),
            initial_radius: 1.0,
            max_radius: 100.0,
            restarts: None,
        }
    }
}

impl TrustRegion {
    /// Builds a trust-region optimizer with 100 iterations, gradient tolerance
    /// `sqrt(ε)`, initial radius 1, and maximum radius 100 (logit coordinates).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::num::NonZeroU32;
    /// use gprx::TrustRegion;
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let _optimizer = TrustRegion::new()
    ///     .with_max_iterations(50)
    ///     .with_tolerance(1e-8)?
    ///     .with_radii(1.0, 100.0)?
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
                reason: "trust-region gradient tolerance must be finite and >= 0".to_owned(),
            });
        }
        self.tolerance = tolerance;
        Ok(self)
    }

    /// Sets the initial and the maximum trust radius (defaults 1 and 100).
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidConfig`] unless
    /// `0 < initial_radius <= max_radius` and both are finite.
    pub fn with_radii(mut self, initial_radius: f64, max_radius: f64) -> Result<Self, GprError> {
        let ok = initial_radius.is_finite()
            && max_radius.is_finite()
            && initial_radius > 0.0
            && initial_radius <= max_radius;
        if !ok {
            return Err(GprError::InvalidConfig {
                reason: "trust radii must satisfy 0 < initial <= max, both finite".to_owned(),
            });
        }
        self.initial_radius = initial_radius;
        self.max_radius = max_radius;
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
        run_trust_region(self, objective, init)
    }
}

impl<P: TwiceDifferentiable> Optimizer<P> for TrustRegion {
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
        consider(self, objective, &intervals, &first_z, &mut best)?;
        if let Some(restarts) = self.restarts {
            let mut rng = crate::rng::small_rng(restarts.seed);
            for _ in 0..restarts.n.get() {
                let z = sample_log_uniform_z(&intervals, &mut rng)?;
                let _ = consider(self, objective, &intervals, &z, &mut best);
            }
        }
        best.ok_or(GprError::OptimizationNotConverged { iterations: 0 })
    }
}

fn consider<P: TwiceDifferentiable>(
    optimizer: &TrustRegion,
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
    let run = run_trust_region(optimizer, &mut mapped, init_z)?;
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

fn run_trust_region<P: TwiceDifferentiable>(
    optimizer: &TrustRegion,
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
    // The start must be evaluable; an error there is the caller's, not a
    // failure to make progress.
    cache.ensure(init)?;
    let problem = TrustRegionProblem {
        inner: RefCell::new(cache),
    };
    let subproblem = Steihaug::<Vec<f64>, f64>::new().with_max_iters(4 * n as u64 + 10);
    let inner = ArgminTrustRegion::new(subproblem)
        .with_radius(optimizer.initial_radius)
        .and_then(|tr| tr.with_max_radius(optimizer.max_radius))
        .map_err(|e| GprError::InvalidConfig {
            reason: e.to_string(),
        })?;
    let solver = GradNormStop {
        inner,
        tolerance: optimizer.tolerance,
    };
    let result = Executor::new(problem, solver)
        .configure(|state| {
            state
                .param(init.to_vec())
                .max_iters(optimizer.max_iterations)
        })
        .ctrlc(false)
        .run()
        .map_err(map_error)?;
    let state = result.state();
    let iterations = state.get_iter();
    let params = state
        .get_best_param()
        .cloned()
        .ok_or(GprError::OptimizationNotConverged {
            iterations: iterations as usize,
        })?;
    let value = state.get_best_cost();
    if !value.is_finite() {
        return Err(GprError::OptimizationNotConverged {
            iterations: iterations as usize,
        });
    }
    Ok(OptResult {
        params,
        value,
        iterations,
    })
}

/// Failures of the solver (a subproblem or a numerical condition of argmin)
/// are a failure to make progress; the objective's own errors keep their type.
fn map_error(err: ArgminError) -> GprError {
    match err.downcast_ref::<GprError>() {
        Some(gpr) => gpr.clone(),
        None => GprError::OptimizationNotConverged { iterations: 0 },
    }
}

/// Stops on the gradient norm.
struct GradNormStop<S> {
    inner: S,
    tolerance: f64,
}

type TrState = IterState<Vec<f64>, Vec<f64>, (), Vec<Vec<f64>>, (), f64>;

impl<O, S> Solver<O, TrState> for GradNormStop<S>
where
    S: Solver<O, TrState>,
{
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn init(
        &mut self,
        problem: &mut Problem<O>,
        state: TrState,
    ) -> Result<(TrState, Option<KV>), ArgminError> {
        self.inner.init(problem, state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        state: TrState,
    ) -> Result<(TrState, Option<KV>), ArgminError> {
        self.inner.next_iter(problem, state)
    }

    fn terminate(&mut self, state: &TrState) -> TerminationStatus {
        if let Some(grad) = state.get_gradient() {
            let norm = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
            if norm <= self.tolerance {
                return TerminationStatus::Terminated(TerminationReason::SolverConverged);
            }
        }
        self.inner.terminate(state)
    }
}

/// The objective with the buffers of the last evaluation: value, gradient, and
/// Hessian. A trust-region iteration evaluates one candidate, so the Hessian
/// computed there is the one argmin asks for when it accepts the step, and a
/// candidate whose Hessian cannot be computed is a bad step.
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

    /// Value, gradient, and Hessian at `param`, into `self.grad` / `self.hess`.
    fn ensure(&mut self, param: &[f64]) -> Result<f64, GprError> {
        if let Some(value) = self.value {
            if self.params.len() == param.len()
                && self
                    .params
                    .iter()
                    .zip(param)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            {
                return Ok(value);
            }
        }
        let value = self
            .objective
            .value_and_gradient_into(param, &mut self.grad)?;
        let hessian = self.objective.hessian_into(param, &mut self.hess);
        if hessian.is_err()
            || !value.is_finite()
            || self.grad.iter().any(|g| !g.is_finite())
            || self.hess.iter().any(|h| !h.is_finite())
        {
            self.value = None;
            return Err(GprError::OptimizationNotConverged { iterations: 0 });
        }
        self.params.clear();
        self.params.extend_from_slice(param);
        self.value = Some(value);
        Ok(value)
    }
}

struct TrustRegionProblem<'a, P: ?Sized> {
    inner: RefCell<HessCache<'a, P>>,
}

impl<P: TwiceDifferentiable + ?Sized> CostFunction for TrustRegionProblem<'_, P> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, ArgminError> {
        // A point that cannot be evaluated is a bad step, not an error.
        Ok(self
            .inner
            .borrow_mut()
            .ensure(param)
            .unwrap_or(BARRIER_COST))
    }
}

impl<P: TwiceDifferentiable + ?Sized> Gradient for TrustRegionProblem<'_, P> {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, ArgminError> {
        let mut inner = self.inner.borrow_mut();
        inner.ensure(param).map_err(ArgminError::from)?;
        Ok(inner.grad.clone())
    }
}

impl<P: TwiceDifferentiable + ?Sized> Hessian for TrustRegionProblem<'_, P> {
    type Param = Vec<f64>;
    type Hessian = Vec<Vec<f64>>;

    fn hessian(&self, param: &Self::Param) -> Result<Self::Hessian, ArgminError> {
        let mut inner = self.inner.borrow_mut();
        inner.ensure(param).map_err(ArgminError::from)?;
        let n = param.len();
        Ok(inner.hess.chunks(n).map(<[f64]>::to_vec).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::TrustRegion;
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
    fn trust_region_minimizes_quadratic_bowl() {
        let mut obj = Quadratic;
        let result = TrustRegion::new()
            .with_max_iterations(50)
            .minimize_unconstrained(&mut obj, &[1.0, -0.5])
            .expect("bowl");
        assert_close(result.params[0], 0.0, TOL);
        assert_close(result.params[1], 0.0, TOL);
        assert!(result.value < 1e-12);
    }

    #[test]
    fn trust_region_lowers_gpr_nlml() {
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
            result = TrustRegion::new()
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
        .with_optimizer(TrustRegion::new().with_max_iterations(30))
        .fit(&x, 3, 1, &y)
        .unwrap_or_else(|(_, e)| panic!("{e}"));
        assert!(fitted.neg_log_marginal_likelihood().expect("fitted") < start);
    }

    /// `√(1 + x²)`: a plain Newton step from `|x| > 1` jumps away (`x → −x³`); values
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
    fn trust_region_converges_where_the_plain_newton_step_diverges() {
        for start in [1.5, 2.0, 4.0, -3.0] {
            let mut obj = Sqrt1PlusSquare;
            let result = TrustRegion::new()
                .with_max_iterations(200)
                .minimize_unconstrained(&mut obj, &[start])
                .expect("safeguarded newton");
            assert_close(result.params[0], 0.0, 1e-6);
        }
    }

    /// The Hessian of `−cos(x)` at `x = 2` is negative: the Newton step
    /// ascends; the trust-region step still descends.
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
    fn trust_region_descends_when_the_hessian_is_not_positive_definite() {
        let mut obj = NegativeCurvature;
        let result = TrustRegion::new()
            .with_max_iterations(200)
            .minimize_unconstrained(&mut obj, &[2.0])
            .expect("falls back to steepest descent");
        assert!(result.value < -0.999, "value {}", result.value);
    }

    #[test]
    fn trust_region_rejects_bad_radii() {
        assert!(TrustRegion::new().with_radii(0.0, 1.0).is_err());
        assert!(TrustRegion::new().with_radii(2.0, 1.0).is_err());
        assert!(TrustRegion::new().with_radii(1.0, f64::INFINITY).is_err());
    }
}
