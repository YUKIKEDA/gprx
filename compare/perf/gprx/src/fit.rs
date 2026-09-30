//! Real-dataset cell (B1-1): ARD RBF fit from a fixed start, then predict the
//! test split and score it. One fit per process (one split of one dataset).

use std::num::NonZeroUsize;
use std::time::Instant;

use gprx::internals as hooks;
use gprx::kernel::{ConstantKernel, KernelSpec, RbfArdKernel};
use gprx::{FittedGpr, Fixed, GaussianLikelihood, Gpr, Lbfgs};

use crate::case::{FitRow, RealCase};
use crate::rss::peak_rss_bytes;

const HISTORY: usize = 10;

fn model_parts(case: &RealCase) -> Result<(KernelSpec, GaussianLikelihood), String> {
    let ard =
        RbfArdKernel::new(&vec![case.lengthscale_init; case.n_cols]).map_err(|e| e.to_string())?;
    let constant = ConstantKernel::new(case.signal_variance_init).map_err(|e| e.to_string())?;
    let kernel = KernelSpec::from(constant) * KernelSpec::from(ard);
    let likelihood =
        GaussianLikelihood::new(case.noise_variance_init).map_err(|e| e.to_string())?;
    Ok((kernel, likelihood))
}

fn make_gpr(case: &RealCase) -> Result<Gpr<Lbfgs>, String> {
    let (kernel, likelihood) = model_parts(case)?;
    let lbfgs = match case.protocol.as_str() {
        "native" => Lbfgs::new(),
        "matched" => Lbfgs::new()
            .with_max_iterations(case.max_iterations)
            .with_tolerance(case.gtol)
            .map_err(|e| e.to_string())?
            .with_history_size(NonZeroUsize::new(HISTORY).ok_or("history")?),
        other => return Err(format!("unknown protocol {other}")),
    };
    Ok(Gpr::new(kernel, likelihood).with_optimizer(lbfgs))
}

/// Untimed warm-up fits first: one below `n = 5000` (the first call pays the
/// thread-pool and allocator start-up), none above, or `PERF_WARMUP`.
fn warmup_fits(n_rows: usize) -> usize {
    match std::env::var("PERF_WARMUP")
        .ok()
        .and_then(|s| s.parse().ok())
    {
        Some(n) => n,
        None if n_rows <= 5000 => 1,
        None => 0,
    }
}

pub fn run(case: &RealCase) -> Result<FitRow, String> {
    if case.protocol != "fixed" {
        for _ in 0..warmup_fits(case.n_rows) {
            eprintln!("PHASE warmup");
            let _ = make_gpr(case)?
                .fit(&case.x, case.n_rows, case.n_cols, &case.y)
                .map_err(|(_, e)| e.to_string())?;
        }
    }
    eprintln!("PHASE fit");
    if case.protocol == "fixed" {
        let (kernel, likelihood) = model_parts(case)?;
        let gpr = Gpr::new(kernel, likelihood).with_optimizer(Fixed);
        let start = Instant::now();
        let fitted = gpr
            .factor(&case.x, case.n_rows, case.n_cols, &case.y)
            .map_err(|(_, e)| e.to_string())?;
        let fit_s = start.elapsed().as_secs_f64();
        return score_row(case, &fitted, fit_s, (0, 0));
    }
    let gpr = make_gpr(case)?;
    hooks::reset_objective_call_counts();
    let start = Instant::now();
    let fitted = gpr
        .fit(&case.x, case.n_rows, case.n_cols, &case.y)
        .map_err(|(_, e)| e.to_string())?;
    let fit_s = start.elapsed().as_secs_f64();
    score_row(case, &fitted, fit_s, hooks::objective_call_counts())
}

fn score_row<O>(
    case: &RealCase,
    fitted: &FittedGpr<O>,
    fit_s: f64,
    (value_evals, joint_evals): (u64, u64),
) -> Result<FitRow, String> {
    eprintln!("PHASE predict");
    let nlml = fitted
        .neg_log_marginal_likelihood()
        .map_err(|e| e.to_string())?;

    let start = Instant::now();
    let pred = fitted
        .predict(&case.xs, case.xs_n_rows, case.n_cols)
        .map_err(|e| e.to_string())?;
    let predict_s = start.elapsed().as_secs_f64();

    let m = case.xs_n_rows as f64;
    let (mut sq, mut nlpd, mut inside) = (0.0, 0.0, 0.0);
    for i in 0..case.xs_n_rows {
        let mu = pred.mean[i] * case.y_std + case.y_mean;
        let var = pred.variance[i] * case.y_std * case.y_std;
        let err = case.ys[i] - mu;
        sq += err * err;
        nlpd += 0.5 * (2.0 * std::f64::consts::PI * var).ln() + 0.5 * err * err / var;
        if err.abs() <= 1.959_963_984_540_054 * var.sqrt() {
            inside += 1.0;
        }
    }
    Ok(FitRow {
        lib: "gprx".to_string(),
        name: case.name.clone(),
        status: "ok".to_string(),
        protocol: case.protocol.clone(),
        fit_s: Some(fit_s),
        predict_s: Some(predict_s),
        joint_evals: Some(joint_evals),
        value_evals: Some(value_evals),
        iterations: None,
        nlml: Some(nlml),
        rmse: Some((sq / m).sqrt()),
        nlpd: Some(nlpd / m),
        coverage95: Some(inside / m),
        peak_rss_bytes: Some(peak_rss_bytes()?),
        note: None,
    })
}
