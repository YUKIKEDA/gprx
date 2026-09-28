//! Nonlinear conjugate gradient via argmin.

use std::cell::RefCell;
use std::num::NonZeroU32;

use argmin::core::{
    Executor, IterState, KV, Problem, Solver, State, TerminationReason, TerminationStatus,
};
use argmin::solver::conjugategradient::NonlinearConjugateGradient;
use argmin::solver::conjugategradient::beta::PolakRibierePlus;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin_math::ArgminL2Norm;

use crate::error::GprError;
use crate::objective::{Differentiable, HasBounds};
use crate::param::Interval;

use super::logit::{
    CachedProblem, EvalCache, consider_value_run, log_theta_to_z, map_argmin_error,
    sample_log_uniform_z,
};
use super::{OptResult, Optimizer, Restarts};

/// Nonlinear conjugate gradient via argmin.
#[derive(Clone, Debug, PartialEq)]
pub struct NonlinearCg {
    max_iterations: u64,
    tolerance: f64,
    restarts: Option<Restarts>,
}

impl Default for NonlinearCg {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: f64::EPSILON.sqrt(),
            restarts: None,
        }
    }
}

impl NonlinearCg {
    /// Builds nonlinear CG with 100 iterations and gradient tolerance `sqrt(ε)`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::num::NonZeroU32;
    /// use gprx::NonlinearCg;
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let _ncg = NonlinearCg::new()
    ///     .with_max_iterations(50)
    ///     .with_tolerance(1e-8)?
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
                reason: "nonlinear CG gradient tolerance must be finite and >= 0".to_owned(),
            });
        }
        self.tolerance = tolerance;
        Ok(self)
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
        run_ncg(self, objective, init)
    }
}

impl<P: Differentiable + HasBounds> Optimizer<P> for NonlinearCg {
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
        consider_value_run(objective, &intervals, &first_z, &mut best, |mapped, z| {
            run_ncg(self, mapped, z)
        })?;
        if let Some(restarts) = self.restarts {
            let mut rng = crate::rng::small_rng(restarts.seed);
            for _ in 0..restarts.n.get() {
                let z = sample_log_uniform_z(&intervals, &mut rng)?;
                let _ = consider_value_run(objective, &intervals, &z, &mut best, |mapped, z| {
                    run_ncg(self, mapped, z)
                });
            }
        }
        best.ok_or(GprError::OptimizationNotConverged { iterations: 0 })
    }
}

fn run_ncg<P: Differentiable>(
    ncg: &NonlinearCg,
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
    let inner = NonlinearConjugateGradient::new(linesearch, PolakRibierePlus::new())
        .restart_orthogonality(0.1);
    let solver = GradNormStop {
        inner,
        tol_grad: ncg.tolerance,
    };
    let (params, iterations) = {
        let result = Executor::new(problem, solver)
            .configure(|state| state.param(init.to_vec()).max_iters(ncg.max_iterations))
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

struct GradNormStop<S> {
    inner: S,
    tol_grad: f64,
}

impl<O, P, G, S> Solver<O, IterState<P, G, (), (), (), f64>> for GradNormStop<S>
where
    S: Solver<O, IterState<P, G, (), (), (), f64>>,
    P: Clone,
    G: ArgminL2Norm<f64>,
{
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn init(
        &mut self,
        problem: &mut Problem<O>,
        state: IterState<P, G, (), (), (), f64>,
    ) -> Result<(IterState<P, G, (), (), (), f64>, Option<KV>), argmin::core::Error> {
        self.inner.init(problem, state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        state: IterState<P, G, (), (), (), f64>,
    ) -> Result<(IterState<P, G, (), (), (), f64>, Option<KV>), argmin::core::Error> {
        self.inner.next_iter(problem, state)
    }

    fn terminate(&mut self, state: &IterState<P, G, (), (), (), f64>) -> TerminationStatus {
        if let Some(grad) = state.get_gradient() {
            if grad.l2_norm() < self.tol_grad {
                return TerminationStatus::Terminated(TerminationReason::SolverConverged);
            }
        }
        self.inner.terminate(state)
    }
}

#[cfg(test)]
mod tests {
    use super::NonlinearCg;
    use crate::error::GprError;
    use crate::gpr::Gpr;
    use crate::kernel::{KernelSpec, RbfKernel};
    use crate::likelihood::GaussianLikelihood;
    use crate::objective::{Differentiable, Objective};
    use crate::optimizer::{Fixed, Optimizer};

    const TOL: f64 = 1e-6;

    use crate::test_check::assert_close;

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
    fn ncg_minimizes_quadratic_bowl() {
        let mut obj = Quadratic { joint_evals: 0 };
        let result = NonlinearCg::new()
            .with_max_iterations(50)
            .minimize_unconstrained(&mut obj, &[1.0, -0.5])
            .expect("bowl");
        assert_close(result.params[0], 0.0, TOL);
        assert_close(result.params[1], 0.0, TOL);
        assert!(result.value < 1e-10);
        assert!(obj.joint_evals > 0);
    }

    #[test]
    fn ncg_lowers_gpr_nlml() {
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
            result = NonlinearCg::new()
                .with_max_iterations(40)
                .minimize(&mut obj, &init)
                .expect("ncg");
        }
        assert!(
            result.value <= start + 1e-9,
            "start={start}, best={}",
            result.value
        );
        let mut got = [0.0; 2];
        gpr.get_params(&mut got).expect("len 2");
        assert_close(got[0], result.params[0], TOL);
        assert_close(got[1], result.params[1], TOL);
    }
}
