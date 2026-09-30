//! Exact cell (P2B-16): `Gpr<Fixed>::factor`, then N joint MLL+grad evals,
//! then predict 100.

use std::time::Instant;

use gprx::transform::StandardizeTarget;
use gprx::{FittedGpr, Fixed, GaussianLikelihood, Gpr};

use crate::case::{Case, ResultRow};
use crate::rss::peak_rss_bytes;
use crate::shared::rbf_kernel;
use crate::timing;

fn make_gpr(case: &Case) -> Result<Gpr<Fixed>, String> {
    let likelihood =
        GaussianLikelihood::new(case.noise_variance_init).map_err(|e| e.to_string())?;
    Ok(
        Gpr::new(rbf_kernel(case.ard, &case.lengthscales_init)?, likelihood)
            .with_target_transform(StandardizeTarget::new())
            .with_optimizer(Fixed),
    )
}

fn finish_row(
    case: &Case,
    lib: &str,
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
        lib: lib.to_string(),
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
            "Gpr<Fixed>::factor + {n}× median of one joint MLL+grad; discard {warmup} then {reps} timed",
            n = case.joint_evals
        )),
    })
}

fn time_eval_predict(
    case: &Case,
    fitted: &mut FittedGpr<Fixed>,
    warmup: usize,
    reps: usize,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let n_params = fitted.num_params();
    let mut params = vec![0.0; n_params];
    fitted.get_params(&mut params).map_err(|e| e.to_string())?;
    let mut grad = vec![0.0; n_params];
    let mut eval_samples = Vec::with_capacity(reps);
    for i in 0..warmup + reps {
        let start = Instant::now();
        fitted
            .value_and_gradient_into(&params, &mut grad)
            .map_err(|e| e.to_string())?;
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
        fitted
            .predict(&case.xs, case.xs_n_rows, case.xs_n_cols)
            .map_err(|e| e.to_string())?;
        let dt = start.elapsed().as_secs_f64();
        if i >= warmup {
            predict_samples.push(dt);
        }
    }
    Ok((eval_scale, predict_samples))
}

/// `Gpr<Fixed>::factor` (speed pole, or `with_prefer_memory` with
/// `memory`), then N joint MLL+grad evals, then predict 100.
pub fn run(case: &Case, memory: bool) -> Result<ResultRow, String> {
    let warmup = timing::warmup_count();
    let reps = timing::timed_reps(case.n_rows);
    let mut factor_samples = Vec::with_capacity(reps);
    let mut fitted: Option<FittedGpr<Fixed>> = None;
    for i in 0..warmup + reps {
        drop(fitted.take());
        let gpr = if memory {
            make_gpr(case)?.with_prefer_memory()
        } else {
            make_gpr(case)?
        };
        let start = Instant::now();
        let next = gpr
            .factor(&case.x, case.n_rows, case.n_cols, &case.y)
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
        if memory { "gprx-memory" } else { "gprx" },
        &factor_samples,
        &eval_scale,
        &predict_samples,
        warmup,
        reps,
    )
}
