//! One gprx cell: `Gpr::fit` (L-BFGS) then `predict` on 100 query points.

use std::cell::RefCell;
use std::env;
use std::fs;
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use argmin::core::{CostFunction, Error as ArgminError, Executor, Gradient, State};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use gprx::kernel::{KernelSpec, RbfArdKernel, RbfKernel};
use gprx::transform::StandardizeTarget;
use gprx::{
    Differentiable, GaussianLikelihood, Gpr, GprError, Objective, OptResult, Optimizer,
};

#[path = "../../case_schema.rs"]
mod case_schema;
#[path = "../../rss_win.rs"]
mod peak_rss;

use case_schema::{Case, ResultRow};

struct CountingLbfgs {
    evals: Arc<AtomicU64>,
}

impl Clone for CountingLbfgs {
    fn clone(&self) -> Self {
        Self {
            evals: Arc::clone(&self.evals),
        }
    }
}

struct Counted<'a, P: ?Sized> {
    inner: &'a mut P,
    evals: &'a AtomicU64,
}

impl<P: Objective + ?Sized> Objective for Counted<'_, P> {
    fn num_params(&self) -> usize {
        self.inner.num_params()
    }

    fn value(&mut self, params: &[f64]) -> Result<f64, GprError> {
        self.inner.value(params)
    }
}

impl<P: Differentiable + ?Sized> Differentiable for Counted<'_, P> {
    fn gradient_into(&mut self, params: &[f64], out: &mut [f64]) -> Result<(), GprError> {
        self.inner.gradient_into(params, out)
    }

    fn value_and_gradient_into(
        &mut self,
        params: &[f64],
        out: &mut [f64],
    ) -> Result<f64, GprError> {
        self.evals.fetch_add(1, Ordering::Relaxed);
        self.inner.value_and_gradient_into(params, out)
    }
}

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
            if self.params == param {
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
                self.value = Some(1.0e300);
                Ok(1.0e300)
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
            .map_err(|e| ArgminError::msg(e.to_string()))
    }
}

impl<P: Differentiable + ?Sized> Gradient for CachedProblem<'_, P> {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, ArgminError> {
        let mut inner = self.inner.borrow_mut();
        inner
            .eval(param)
            .map_err(|e| ArgminError::msg(e.to_string()))?;
        Ok(inner.grad.clone())
    }
}

impl<P: Differentiable> Optimizer<P> for CountingLbfgs {
    fn minimize(&self, objective: &mut P, init: &[f64]) -> Result<OptResult, GprError> {
        let mut counted = Counted {
            inner: objective,
            evals: &self.evals,
        };
        let n = counted.num_params();
        if init.len() != n {
            return Err(GprError::InvalidHyperparameter {
                reason: format!("expected {n} parameters, got {}", init.len()),
            });
        }
        let problem = CachedProblem {
            inner: RefCell::new(EvalCache::new(&mut counted, n)),
        };
        let linesearch = MoreThuenteLineSearch::<Vec<f64>, Vec<f64>, f64>::new();
        let solver: LBFGS<_, Vec<f64>, Vec<f64>, f64> = LBFGS::new(linesearch, 10)
            .with_tolerance_grad(f64::EPSILON.sqrt())
            .map_err(|_| GprError::OptimizationNotConverged { iterations: 0 })?;
        let result = Executor::new(problem, solver)
            .configure(|state| state.param(init.to_vec()).max_iters(100))
            .ctrlc(false)
            .run()
            .map_err(|_| GprError::OptimizationNotConverged { iterations: 0 })?;
        let state = result.state();
        let params = state
            .get_best_param()
            .cloned()
            .ok_or(GprError::OptimizationNotConverged {
                iterations: state.get_iter() as usize,
            })?;
        let iterations = state.get_iter();
        let mut grad = vec![0.0; n];
        let value = counted.value_and_gradient_into(&params, &mut grad)?;
        Ok(OptResult {
            params,
            value,
            iterations,
        })
    }
}

fn na_row(name: &str, note: String) -> ResultRow {
    ResultRow {
        lib: "gprx".to_string(),
        name: name.to_string(),
        status: "na".to_string(),
        fit_s: None,
        predict_s: None,
        joint_evals: None,
        peak_rss_bytes: None,
        note: Some(note),
    }
}

fn run(case: &Case) -> Result<ResultRow, String> {
    let kernel = if case.ard {
        let spec = RbfArdKernel::new(&case.lengthscales_init).map_err(|e| e.to_string())?;
        KernelSpec::from(spec)
    } else {
        let ell = case
            .lengthscales_init
            .first()
            .copied()
            .ok_or_else(|| "missing lengthscale".to_string())?;
        KernelSpec::from(RbfKernel::new(ell).map_err(|e| e.to_string())?)
    };
    let likelihood =
        GaussianLikelihood::new(case.noise_variance_init).map_err(|e| e.to_string())?;
    let evals = Arc::new(AtomicU64::new(0));
    let optimizer = CountingLbfgs {
        evals: Arc::clone(&evals),
    };
    let gpr = Gpr::new(kernel, likelihood)
        .with_target_transform(StandardizeTarget::new())
        .with_optimizer(optimizer);
    let fit_start = Instant::now();
    let fitted = gpr
        .fit(&case.x, case.n_rows, case.n_cols, &case.y)
        .map_err(|(_, e)| e.to_string())?;
    let fit_s = fit_start.elapsed().as_secs_f64();
    let predict_start = Instant::now();
    fitted
        .predict(&case.xs, case.xs_n_rows, case.xs_n_cols)
        .map_err(|e| e.to_string())?;
    let predict_s = predict_start.elapsed().as_secs_f64();
    Ok(ResultRow {
        lib: "gprx".to_string(),
        name: case.name.clone(),
        status: "ok".to_string(),
        fit_s: Some(fit_s),
        predict_s: Some(predict_s),
        joint_evals: Some(evals.load(Ordering::Relaxed)),
        peak_rss_bytes: Some(peak_rss::peak_rss_bytes()?),
        note: Some(
            "argmin L-BFGS on log-theta (HasBounds is crate-private; no logit map)"
                .to_string(),
        ),
    })
}

fn main() -> ExitCode {
    let path = match env::args().nth(1) {
        Some(p) => p,
        None => {
            eprintln!("usage: gprx-perf CASE.json");
            return ExitCode::from(2);
        }
    };
    let text = match fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(1);
        }
    };
    let case: Case = match serde_json::from_str(&text) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(1);
        }
    };
    let row = match run(&case) {
        Ok(row) => row,
        Err(e) => na_row(&case.name, e),
    };
    match serde_json::to_string(&row) {
        Ok(json) => {
            println!("{json}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("{e}");
            ExitCode::from(1)
        }
    }
}
