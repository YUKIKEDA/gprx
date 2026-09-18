//! argmin L-BFGS adapter for [`crate::objective::Objective`].
//!
//! Crate-private. Does not implement L-BFGS itself. When argmin asks for cost
//! and gradient at the same `θ`, one [`Objective::value_and_gradient_into`]
//! call fills both.

use std::cell::RefCell;

use argmin::core::{CostFunction, Error as ArgminError, Executor, Gradient, State};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;

use crate::error::GprError;
use crate::objective::Objective;

/// Result of [`Optimizer::minimize`].
#[derive(Debug)]
pub(crate) struct OptResult {
    pub(crate) params: Vec<f64>,
    pub(crate) value: f64,
    pub(crate) iterations: u64,
}

/// Hyperparameter optimizer.
pub(crate) trait Optimizer {
    fn requires_gradient(&self) -> bool;

    fn minimize(&self, objective: &mut dyn Objective, init: &[f64]) -> Result<OptResult, GprError>;
}

/// Limited-memory BFGS via argmin.
pub(crate) struct Lbfgs {
    history: usize,
    max_iters: u64,
}

impl Default for Lbfgs {
    fn default() -> Self {
        Self {
            history: 10,
            max_iters: 100,
        }
    }
}

impl Lbfgs {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn with_max_iters(mut self, max_iters: u64) -> Self {
        self.max_iters = max_iters;
        self
    }
}

impl Optimizer for Lbfgs {
    fn requires_gradient(&self) -> bool {
        true
    }

    fn minimize(&self, objective: &mut dyn Objective, init: &[f64]) -> Result<OptResult, GprError> {
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
        let solver: LBFGS<_, Vec<f64>, Vec<f64>, f64> = LBFGS::new(linesearch, self.history);
        let (params, iterations) = {
            let result = Executor::new(problem, solver)
                .configure(|state| state.param(init.to_vec()).max_iters(self.max_iters))
                .ctrlc(false)
                .run()
                .map_err(map_argmin_error)?;
            let state = result.state();
            let params =
                state
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
}

struct EvalCache<'a> {
    objective: &'a mut dyn Objective,
    params: Vec<f64>,
    value: Option<f64>,
    grad: Vec<f64>,
}

impl<'a> EvalCache<'a> {
    fn new(objective: &'a mut dyn Objective, n: usize) -> Self {
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
            Ok(value) => {
                self.params.clear();
                self.params.extend_from_slice(param);
                self.value = Some(value);
                Ok(value)
            }
            Err(err) => {
                self.value = None;
                Err(err)
            }
        }
    }
}

struct CachedProblem<'a> {
    inner: RefCell<EvalCache<'a>>,
}

impl CostFunction for CachedProblem<'_> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, ArgminError> {
        self.inner
            .borrow_mut()
            .eval(param)
            .map_err(ArgminError::from)
    }
}

impl Gradient for CachedProblem<'_> {
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
    use super::{CachedProblem, CostFunction, EvalCache, Gradient, Lbfgs, Optimizer};
    use crate::error::GprError;
    use crate::gpr::Gpr;
    use crate::kernel::{KernelSpec, RbfArdKernel, RbfKernel};
    use crate::likelihood::GaussianLikelihood;
    use crate::objective::Objective;
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
    fn lbfgs_requires_gradient() {
        assert!(Lbfgs::new().requires_gradient());
    }

    #[test]
    fn lbfgs_rejects_wrong_init_len() {
        let mut obj = Quadratic { joint_evals: 0 };
        let err = Lbfgs::new().minimize(&mut obj, &[0.0]).expect_err("len");
        assert!(matches!(err, GprError::InvalidHyperparameter { .. }));
        assert_eq!(obj.joint_evals, 0);
    }

    #[test]
    fn lbfgs_minimizes_quadratic_bowl() {
        let mut obj = Quadratic { joint_evals: 0 };
        let result = Lbfgs::new()
            .with_max_iters(50)
            .minimize(&mut obj, &[1.0, -0.5])
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
    fn lbfgs_lowers_gpr_nlml() {
        let kernel = KernelSpec::from(RbfKernel::new(1.0).expect("valid"));
        let likelihood = GaussianLikelihood::new(0.1).expect("valid");
        let mut gpr = Gpr::new(kernel, likelihood)
            .fit_with(&[0.0, 1.0], 2, 1, &[0.5, -0.25], crate::FitOptions::FIXED)
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
                .with_max_iters(40)
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
            .fit_with(&x, 256, 1, &y, crate::FitOptions::FIXED)
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
                .with_max_iters(100)
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
            .fit_with(&x, 256, 2, &y, crate::FitOptions::FIXED)
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
                .with_max_iters(100)
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
