//! One Sparse cell: `Sgpr<Fixed>::factor` or `Svgp<Fixed>::factor`, then
//! N joint value+grad evals, then predict 100.

use std::env;
use std::fs;
use std::process::ExitCode;
use std::time::Instant;

use gprx::kernel::{KernelSpec, RbfArdKernel, RbfKernel};
use gprx::{FittedSgpr, FittedSvgp, Fixed, GaussianLikelihood, Sgpr, Svgp};

#[allow(dead_code)]
#[path = "../../case_schema.rs"]
mod case_schema;
#[path = "../../rss_win.rs"]
mod peak_rss;
#[path = "../../timing.rs"]
mod timing;

use case_schema::{ResultRow, SparseCase};

fn na_row(lib: &str, name: &str, note: String) -> ResultRow {
    ResultRow {
        lib: lib.to_string(),
        name: name.to_string(),
        status: "na".to_string(),
        factor_s: None,
        factor_min_s: None,
        factor_max_s: None,
        eval_s: None,
        eval_min_s: None,
        eval_max_s: None,
        predict_s: None,
        predict_min_s: None,
        predict_max_s: None,
        joint_evals: None,
        peak_rss_bytes: None,
        warmup: None,
        reps: None,
        kernel_s: None,
        border_s: None,
        rest_s: None,
        note: Some(note),
    }
}

fn make_kernel(case: &SparseCase) -> Result<KernelSpec, String> {
    if case.ard {
        let spec = RbfArdKernel::new(&case.lengthscales_init).map_err(|e| e.to_string())?;
        Ok(KernelSpec::from(spec))
    } else {
        let ell = case
            .lengthscales_init
            .first()
            .copied()
            .ok_or_else(|| "missing lengthscale".to_string())?;
        Ok(KernelSpec::from(RbfKernel::new(ell).map_err(|e| e.to_string())?))
    }
}

fn finish_row(
    case: &SparseCase,
    factor_samples: &[f64],
    eval_scale: &[f64],
    predict_samples: &[f64],
    warmup: usize,
    reps: usize,
) -> Result<ResultRow, String> {
    let (factor_min, factor_max) = timing::min_max(factor_samples);
    let (eval_min, eval_max) = timing::min_max(eval_scale);
    let (predict_min, predict_max) = timing::min_max(predict_samples);
    Ok(ResultRow {
        lib: "gprx".to_string(),
        name: case.name.clone(),
        status: "ok".to_string(),
        factor_s: Some(timing::median(factor_samples)),
        factor_min_s: Some(factor_min),
        factor_max_s: Some(factor_max),
        eval_s: Some(timing::median(eval_scale)),
        eval_min_s: Some(eval_min),
        eval_max_s: Some(eval_max),
        predict_s: Some(timing::median(predict_samples)),
        predict_min_s: Some(predict_min),
        predict_max_s: Some(predict_max),
        joint_evals: Some(case.joint_evals),
        peak_rss_bytes: Some(peak_rss::peak_rss_bytes()?),
        warmup: Some(warmup as u64),
        reps: Some(reps as u64),
        kernel_s: None,
        border_s: None,
        rest_s: None,
        note: Some(format!(
            "{model}<Fixed>::factor + {n}× median of one joint value+grad; discard {warmup} then {reps} timed",
            model = case.model,
            n = case.joint_evals
        )),
    })
}

fn time_eval_predict<F>(
    case: &SparseCase,
    fitted: &mut F,
    warmup: usize,
    reps: usize,
) -> Result<(Vec<f64>, Vec<f64>), String>
where
    F: SparseEvalPredict,
{
    let n_params = fitted.num_params();
    let mut params = vec![0.0; n_params];
    fitted.get_params(&mut params)?;
    let mut grad = vec![0.0; n_params];
    let mut eval_samples = Vec::with_capacity(reps);
    for i in 0..warmup + reps {
        let start = Instant::now();
        fitted.value_and_gradient_into(&params, &mut grad)?;
        let dt = start.elapsed().as_secs_f64();
        if i >= warmup {
            eval_samples.push(dt);
        }
    }
    let n_evals = case.joint_evals as f64;
    let eval_scale: Vec<f64> = eval_samples.iter().map(|s| s * n_evals).collect();

    let mut predict_samples = Vec::with_capacity(reps);
    for i in 0..warmup + reps {
        let start = Instant::now();
        fitted.predict(&case.xs, case.xs_n_rows, case.xs_n_cols)?;
        let dt = start.elapsed().as_secs_f64();
        if i >= warmup {
            predict_samples.push(dt);
        }
    }
    Ok((eval_scale, predict_samples))
}

trait SparseEvalPredict {
    fn num_params(&self) -> usize;
    fn get_params(&self, out: &mut [f64]) -> Result<(), String>;
    fn value_and_gradient_into(&mut self, params: &[f64], out: &mut [f64]) -> Result<f64, String>;
    fn predict(&self, xs: &[f64], n_rows: usize, n_cols: usize) -> Result<(), String>;
}

impl SparseEvalPredict for FittedSgpr<Fixed> {
    fn num_params(&self) -> usize {
        FittedSgpr::num_params(self)
    }

    fn get_params(&self, out: &mut [f64]) -> Result<(), String> {
        FittedSgpr::get_params(self, out).map_err(|e| e.to_string())
    }

    fn value_and_gradient_into(&mut self, params: &[f64], out: &mut [f64]) -> Result<f64, String> {
        FittedSgpr::value_and_gradient_into(self, params, out).map_err(|e| e.to_string())
    }

    fn predict(&self, xs: &[f64], n_rows: usize, n_cols: usize) -> Result<(), String> {
        FittedSgpr::predict(self, xs, n_rows, n_cols)
            .map(|_| ())
            .map_err(|e| e.to_string())
    }
}

impl SparseEvalPredict for FittedSvgp {
    fn num_params(&self) -> usize {
        FittedSvgp::num_params(self)
    }

    fn get_params(&self, out: &mut [f64]) -> Result<(), String> {
        FittedSvgp::get_params(self, out).map_err(|e| e.to_string())
    }

    fn value_and_gradient_into(&mut self, params: &[f64], out: &mut [f64]) -> Result<f64, String> {
        FittedSvgp::value_and_gradient_into(self, params, out).map_err(|e| e.to_string())
    }

    fn predict(&self, xs: &[f64], n_rows: usize, n_cols: usize) -> Result<(), String> {
        FittedSvgp::predict(self, xs, n_rows, n_cols)
            .map(|_| ())
            .map_err(|e| e.to_string())
    }
}

fn run_sgpr(case: &SparseCase) -> Result<ResultRow, String> {
    let warmup = timing::warmup_count();
    let reps = timing::timed_reps(case.n_rows);
    let mut factor_samples = Vec::with_capacity(reps);
    let mut fitted: Option<FittedSgpr<Fixed>> = None;
    for i in 0..warmup + reps {
        drop(fitted.take());
        let likelihood =
            GaussianLikelihood::new(case.noise_variance_init).map_err(|e| e.to_string())?;
        let trainer = Sgpr::new(make_kernel(case)?, likelihood).with_optimizer(Fixed);
        let start = Instant::now();
        let next = trainer
            .factor(
                &case.x,
                case.n_rows,
                case.n_cols,
                &case.y,
                &case.z,
                case.n_inducing,
            )
            .map_err(|(_, e)| e.to_string())?;
        let dt = start.elapsed().as_secs_f64();
        if i >= warmup {
            factor_samples.push(dt);
        }
        fitted = Some(next);
    }
    let mut fitted = fitted.expect("timed_reps is at least 1");
    let (eval_scale, predict_samples) = time_eval_predict(case, &mut fitted, warmup, reps)?;
    finish_row(
        case,
        &factor_samples,
        &eval_scale,
        &predict_samples,
        warmup,
        reps,
    )
}

fn run_svgp(case: &SparseCase) -> Result<ResultRow, String> {
    let warmup = timing::warmup_count();
    let reps = timing::timed_reps(case.n_rows);
    let mut factor_samples = Vec::with_capacity(reps);
    let mut fitted: Option<FittedSvgp> = None;
    for i in 0..warmup + reps {
        drop(fitted.take());
        let likelihood =
            GaussianLikelihood::new(case.noise_variance_init).map_err(|e| e.to_string())?;
        let trainer = Svgp::new(make_kernel(case)?, likelihood);
        let start = Instant::now();
        let next = trainer
            .factor(
                &case.x,
                case.n_rows,
                case.n_cols,
                &case.y,
                &case.z,
                case.n_inducing,
            )
            .map_err(|(_, e)| e.to_string())?;
        let dt = start.elapsed().as_secs_f64();
        if i >= warmup {
            factor_samples.push(dt);
        }
        fitted = Some(next);
    }
    let mut fitted = fitted.expect("timed_reps is at least 1");
    let (eval_scale, predict_samples) = time_eval_predict(case, &mut fitted, warmup, reps)?;
    finish_row(
        case,
        &factor_samples,
        &eval_scale,
        &predict_samples,
        warmup,
        reps,
    )
}

fn main() -> ExitCode {
    let Some(path) = env::args().nth(1) else {
        eprintln!("usage: gprx-sparse-perf CASE.json");
        return ExitCode::from(2);
    };
    let text = match fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(1);
        }
    };
    let case: SparseCase = match serde_json::from_str(&text) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(1);
        }
    };
    let row = match case.model.as_str() {
        "sgpr" => run_sgpr(&case),
        "svgp" => run_svgp(&case),
        other => Err(format!("unknown sparse model {other}")),
    };
    let row = match row {
        Ok(row) => row,
        Err(e) => na_row("gprx", &case.name, e),
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
