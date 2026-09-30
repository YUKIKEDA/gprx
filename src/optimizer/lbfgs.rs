//! Limited-memory BFGS via argmin.

use std::cell::RefCell;
use std::num::{NonZeroU32, NonZeroUsize};

use argmin::core::{Executor, State};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;

use crate::error::GprError;
use crate::objective::{Differentiable, HasBounds};
use crate::param::Interval;

use super::logit::{
    CachedProblem, EvalCache, LogitMapped, keep_better, log_theta_to_z, map_argmin_error,
    sample_log_uniform_z, z_to_log_theta,
};
use super::{OptResult, Optimizer, Restarts};

/// Limited-memory BFGS via argmin.
#[derive(Clone, Debug, PartialEq)]
pub struct Lbfgs {
    max_iterations: u64,
    tolerance: f64,
    history_size: NonZeroUsize,
    restarts: Option<Restarts>,
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
    /// Returns [`GprError::InvalidConfig`] if `tolerance` is not finite
    /// or is negative.
    pub fn with_tolerance(mut self, tolerance: f64) -> Result<Self, GprError> {
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(GprError::InvalidConfig {
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
            return Err(GprError::LengthMismatch {
                reason: format!("expected {n} parameters, got {}", init.len()),
            });
        }
        let mut intervals = vec![Interval::DEFAULT_POSITIVE; n];
        objective.fill_intervals(&mut intervals)?;
        let mut best: Option<OptResult> = None;
        let first_z = log_theta_to_z(init, &intervals)?;
        consider_run(self, objective, &intervals, &first_z, &mut best)?;
        if let Some(restarts) = self.restarts {
            let mut rng = crate::rng::small_rng(restarts.seed);
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

fn run_lbfgs<P: Differentiable>(
    lbfgs: &Lbfgs,
    objective: &mut P,
    init: &[f64],
) -> Result<OptResult, GprError> {
    let n = objective.num_params();
    if init.len() != n {
        return Err(GprError::LengthMismatch {
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
    let (params, iterations) =
        {
            let result = Executor::new(problem, solver)
                .configure(|state| state.param(init.to_vec()).max_iters(lbfgs.max_iterations))
                .ctrlc(false)
                .run()
                .map_err(map_argmin_error)?;
            let state = result.state();
            let params = state.get_best_param().cloned().ok_or_else(|| {
                GprError::OptimizationNotConverged {
                    iterations: state.get_iter() as usize,
                }
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

#[cfg(test)]
mod tests {
    use super::Lbfgs;
    use crate::error::GprError;
    use crate::gpr::Gpr;
    use crate::kernel::{KernelSpec, RbfArdKernel, RbfKernel};
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
                return Err(GprError::ShapeMismatch {
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
        assert!(matches!(err, GprError::LengthMismatch { .. }));
        assert_eq!(obj.joint_evals, 0);
    }

    #[test]
    fn lbfgs_minimizes_quadratic_bowl() {
        let mut obj = Quadratic { joint_evals: 0 };
        let result = Lbfgs::new()
            .with_max_iterations(50)
            .minimize_unconstrained(&mut obj, &[1.0, -0.5])
            .expect("bowl");
        assert_close(result.params[0], 0.0, TOL);
        assert_close(result.params[1], 0.0, TOL);
        assert!(result.value < 1e-10);
        assert!(obj.joint_evals > 0);
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
        assert_close(got[0], result.params[0], TOL);
        assert_close(got[1], result.params[1], TOL);
    }

    struct CountingObj<'a> {
        inner: crate::objective::GprObjective<'a>,
        joint_evals: usize,
    }

    impl Objective for CountingObj<'_> {
        fn num_params(&self) -> usize {
            self.inner.num_params()
        }

        fn value(&mut self, params: &[f64]) -> Result<f64, GprError> {
            let mut dummy = vec![0.0; self.num_params()];
            self.value_and_gradient_into(params, &mut dummy)
        }
    }

    impl Differentiable for CountingObj<'_> {
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

    impl crate::objective::HasBounds for CountingObj<'_> {
        fn fill_intervals(&self, out: &mut [crate::param::Interval]) -> Result<(), GprError> {
            self.inner.fill_intervals(out)
        }
    }

    fn forrester_bench_xy() -> (Vec<f64>, Vec<f64>) {
        crate::test_problems::forrester_xy(256, 0, 1.0)
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
        // SmallRng seed 0 walks a ridge on DistanceCachePolicy::Uncached (~485 evals).
        sphere_bench_xy_with_seed(9)
    }

    fn sphere_bench_xy_with_seed(seed: u64) -> (Vec<f64>, Vec<f64>) {
        crate::test_problems::sphere_xy(16, seed, 1.0)
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
            sphere_lbfgs_evals(crate::DistanceCachePolicy::Cached);
        let (iters_n, evals_n, start_n, value_n) =
            sphere_lbfgs_evals(crate::DistanceCachePolicy::Uncached);
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
        assert_close(start_a, start_n, TOL);
        assert_close(value_a, value_n, TOL);
        assert!(value_a < start_a - 1.0, "start={start_a}, best={value_a}");
    }
}
