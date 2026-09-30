//! Real-dataset cell (B1-1): ARD RBF fit from a fixed start, then predict the
//! test split and score it. One fit per process (one split of one dataset).
//! `model` picks `Gpr`, `Sgpr` (fixed inducing points) or `Svgp` (Adam).

use std::num::{NonZeroU64, NonZeroUsize};
use std::time::Instant;

use gprx::internals as hooks;
use gprx::kernel::{ConstantKernel, KernelSpec, RbfArdKernel};
use gprx::{Adam, Fixed, GaussianLikelihood, Gpr, Lbfgs, Sgpr, Svgp};

use crate::case::{FitRow, RealCase};
use crate::rss::peak_rss_bytes;

const HISTORY: usize = 10;

fn model_parts(case: &RealCase) -> Result<(KernelSpec, GaussianLikelihood), String> {
    let ard =
        RbfArdKernel::new(&vec![case.lengthscale_init; case.n_cols]).map_err(|e| e.to_string())?;
    // Sgpr / Svgp have no coordinate derivative for a Product tree, so a
    // `Constant × RBF` kernel cannot be fitted there: sparse cells leave the
    // signal variance out (fixed at 1) in every library.
    let kernel = if case.model == "exact" {
        let constant = ConstantKernel::new(case.signal_variance_init).map_err(|e| e.to_string())?;
        KernelSpec::from(constant) * KernelSpec::from(ard)
    } else {
        KernelSpec::from(ard)
    };
    let likelihood =
        GaussianLikelihood::new(case.noise_variance_init).map_err(|e| e.to_string())?;
    Ok((kernel, likelihood))
}

fn lbfgs(case: &RealCase) -> Result<Lbfgs, String> {
    match case.protocol.as_str() {
        "native" => Ok(Lbfgs::new()),
        "matched" => Ok(Lbfgs::new()
            .with_max_iterations(case.max_iterations)
            .with_tolerance(case.gtol)
            .map_err(|e| e.to_string())?
            .with_history_size(NonZeroUsize::new(HISTORY).ok_or("history")?)),
        other => Err(format!("unknown protocol {other}")),
    }
}

fn adam(case: &RealCase) -> Result<Adam, String> {
    Ok(Adam::new()
        .with_learning_rate(case.adam_lr)
        .map_err(|e| e.to_string())?
        .with_batch_size(NonZeroUsize::new(case.adam_batch_size).ok_or("batch size")?)
        .with_epochs(NonZeroU64::new(case.adam_epochs).ok_or("epochs")?)
        .with_seed(0))
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

/// What one fit yields: predictions and the model's own numbers.
struct Fitted {
    mean: Vec<f64>,
    variance: Vec<f64>,
    predict_s: f64,
    nlml: Option<f64>,
}

fn predict_timed<E: ToString>(
    predict: impl FnOnce() -> Result<gprx::Prediction<f64>, E>,
) -> Result<(Vec<f64>, Vec<f64>, f64), String> {
    eprintln!("PHASE predict");
    let start = Instant::now();
    let pred = predict().map_err(|e| e.to_string())?;
    Ok((pred.mean, pred.variance, start.elapsed().as_secs_f64()))
}

/// One fit and its predictions; returns the fit wall time too.
fn fit_once(case: &RealCase) -> Result<(Fitted, f64), String> {
    let (kernel, likelihood) = model_parts(case)?;
    let (x, n, d, y) = (&case.x, case.n_rows, case.n_cols, &case.y);
    let (xs, m) = (&case.xs, case.xs_n_rows);
    let start = Instant::now();
    match (case.model.as_str(), case.protocol.as_str()) {
        ("exact", "fixed") => {
            let fitted = Gpr::new(kernel, likelihood)
                .with_optimizer(Fixed)
                .factor(x, n, d, y)
                .map_err(|(_, e)| e.to_string())?;
            let fit_s = start.elapsed().as_secs_f64();
            let nlml = fitted
                .neg_log_marginal_likelihood()
                .map_err(|e| e.to_string())?;
            let (mean, variance, predict_s) = predict_timed(|| fitted.predict(xs, m, d))?;
            Ok((
                Fitted {
                    mean,
                    variance,
                    predict_s,
                    nlml: Some(nlml),
                },
                fit_s,
            ))
        }
        ("exact", _) => {
            let fitted = Gpr::new(kernel, likelihood)
                .with_optimizer(lbfgs(case)?)
                .fit(x, n, d, y)
                .map_err(|(_, e)| e.to_string())?;
            let fit_s = start.elapsed().as_secs_f64();
            let nlml = fitted
                .neg_log_marginal_likelihood()
                .map_err(|e| e.to_string())?;
            let (mean, variance, predict_s) = predict_timed(|| fitted.predict(xs, m, d))?;
            Ok((
                Fitted {
                    mean,
                    variance,
                    predict_s,
                    nlml: Some(nlml),
                },
                fit_s,
            ))
        }
        ("sgpr", _) => {
            let fitted = Sgpr::new(kernel, likelihood)
                .with_optimizer(lbfgs(case)?)
                .fit(x, n, d, y, &case.z, case.n_inducing)
                .map_err(|(_, e)| e.to_string())?;
            let fit_s = start.elapsed().as_secs_f64();
            let nlml = fitted
                .neg_log_marginal_likelihood()
                .map_err(|e| e.to_string())?;
            let (mean, variance, predict_s) = predict_timed(|| fitted.predict(xs, m, d))?;
            Ok((
                Fitted {
                    mean,
                    variance,
                    predict_s,
                    nlml: Some(nlml),
                },
                fit_s,
            ))
        }
        ("svgp", _) => {
            let fitted = Svgp::new(kernel, likelihood)
                .with_optimizer(adam(case)?)
                .fit(x, n, d, y, &case.z, case.n_inducing)
                .map_err(|(_, e)| e.to_string())?;
            let fit_s = start.elapsed().as_secs_f64();
            let (mean, variance, predict_s) = predict_timed(|| fitted.predict(xs, m, d))?;
            Ok((
                Fitted {
                    mean,
                    variance,
                    predict_s,
                    nlml: None,
                },
                fit_s,
            ))
        }
        (model, _) => Err(format!("unknown model {model}")),
    }
}

pub fn run(case: &RealCase) -> Result<FitRow, String> {
    if case.protocol != "fixed" {
        for _ in 0..warmup_fits(case.n_rows) {
            eprintln!("PHASE warmup");
            fit_once(case)?;
        }
    }
    eprintln!("PHASE fit");
    hooks::reset_objective_call_counts();
    let (fitted, fit_s) = fit_once(case)?;
    let (value_evals, joint_evals) = hooks::objective_call_counts();

    let m = case.xs_n_rows as f64;
    let (mut sq, mut nlpd, mut inside) = (0.0, 0.0, 0.0);
    for i in 0..case.xs_n_rows {
        let mu = fitted.mean[i] * case.y_std + case.y_mean;
        let var = fitted.variance[i] * case.y_std * case.y_std;
        let err = case.ys[i] - mu;
        sq += err * err;
        nlpd += 0.5 * (2.0 * std::f64::consts::PI * var).ln() + 0.5 * err * err / var;
        if err.abs() <= 1.959_963_984_540_054 * var.sqrt() {
            inside += 1.0;
        }
    }
    // SVGP has no evaluation counts (Adam steps): report the step count.
    let steps = if case.model == "svgp" {
        Some(case.adam_epochs * (case.n_rows.div_ceil(case.adam_batch_size)) as u64)
    } else {
        None
    };
    Ok(FitRow {
        lib: "gprx".to_string(),
        name: case.name.clone(),
        status: "ok".to_string(),
        protocol: case.protocol.clone(),
        fit_s: Some(fit_s),
        predict_s: Some(fitted.predict_s),
        joint_evals: steps.or(Some(joint_evals)),
        value_evals: Some(value_evals),
        iterations: None,
        nlml: fitted.nlml,
        rmse: Some((sq / m).sqrt()),
        nlpd: Some(nlpd / m),
        coverage95: Some(inside / m),
        peak_rss_bytes: Some(peak_rss_bytes()?),
        note: None,
    })
}
