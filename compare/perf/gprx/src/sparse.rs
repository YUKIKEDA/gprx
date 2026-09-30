//! One Sparse cell: `Sgpr<Fixed>::factor` or `Svgp<Fixed>::factor`, then
//! N joint value+grad evals, then predict 100.

use std::time::Instant;

use gprx::{FittedSgpr, FittedSvgp, Fixed, GaussianLikelihood, Sgpr, Svgp};

use crate::case::{ResultRow, SparseCase};
use crate::rss::peak_rss_bytes;
use crate::shared::rbf_kernel;
use crate::timing;

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
        peak_rss_bytes: Some(peak_rss_bytes()?),
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
        let trainer = Sgpr::new(rbf_kernel(case.ard, &case.lengthscales_init)?, likelihood)
            .with_optimizer(Fixed);
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
    let mut fitted = fitted.ok_or_else(|| "timed_reps is at least 1".to_string())?;
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
        let trainer = Svgp::new(rbf_kernel(case.ard, &case.lengthscales_init)?, likelihood);
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
    let mut fitted = fitted.ok_or_else(|| "timed_reps is at least 1".to_string())?;
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

/// The case's model (`sgpr` or `svgp`): factor, N joint value+grad evals,
/// predict 100.
pub fn run(case: &SparseCase) -> Result<ResultRow, String> {
    match case.model.as_str() {
        "sgpr" => run_sgpr(case),
        "svgp" => run_svgp(case),
        other => Err(format!("unknown sparse model {other}")),
    }
}
