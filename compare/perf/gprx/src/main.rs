//! One gprx cell: `Gpr<Fixed>::factor`, then N joint MLL+grad evals, then predict 100.

use std::env;
use std::fs;
use std::process::ExitCode;
use std::time::Instant;

use gprx::kernel::{KernelSpec, RbfArdKernel, RbfKernel};
use gprx::transform::StandardizeTarget;
use gprx::{
    CachedDistances, DistanceCachePolicy, Fixed, FittedGpr, FullRecompute, GaussianLikelihood, Gpr,
    UncachedDistances,
};

#[path = "../../case_schema.rs"]
mod case_schema;
#[path = "../../rss_win.rs"]
mod peak_rss;
#[path = "../../timing.rs"]
mod timing;

use case_schema::{Case, ResultRow};

fn na_row(name: &str, note: String) -> ResultRow {
    ResultRow {
        lib: "gprx".to_string(),
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
        note: Some(note),
    }
}

fn make_gpr<C: DistanceCachePolicy>(case: &Case, policy: C) -> Result<Gpr<Fixed, FullRecompute, C>, String> {
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
    Ok(Gpr::new(kernel, likelihood)
        .with_target_transform(StandardizeTarget::new())
        .with_distance_cache_policy(policy)
        .with_optimizer(Fixed))
}

fn run<C: DistanceCachePolicy>(case: &Case, policy: C) -> Result<ResultRow, String> {
    let warmup = timing::warmup_count();
    let reps = timing::timed_reps(case.n_rows);
    let mut factor_samples = Vec::with_capacity(reps);
    let mut fitted: Option<FittedGpr<Fixed, FullRecompute, C>> = None;
    for i in 0..warmup + reps {
        drop(fitted.take());
        let gpr = make_gpr(case, policy)?;
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
    let mut fitted = fitted.expect("timed_reps is at least 1");

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

    let (factor_min, factor_max) = timing::min_max(&factor_samples);
    let (eval_min, eval_max) = timing::min_max(&eval_scale);
    let (predict_min, predict_max) = timing::min_max(&predict_samples);
    Ok(ResultRow {
        lib: "gprx".to_string(),
        name: case.name.clone(),
        status: "ok".to_string(),
        factor_s: Some(timing::median(&factor_samples)),
        factor_min_s: Some(factor_min),
        factor_max_s: Some(factor_max),
        eval_s: Some(timing::median(&eval_scale)),
        eval_min_s: Some(eval_min),
        eval_max_s: Some(eval_max),
        predict_s: Some(timing::median(&predict_samples)),
        predict_min_s: Some(predict_min),
        predict_max_s: Some(predict_max),
        joint_evals: Some(case.joint_evals),
        peak_rss_bytes: Some(peak_rss::peak_rss_bytes()?),
        warmup: Some(warmup as u64),
        reps: Some(reps as u64),
        note: Some(format!(
            "Gpr<Fixed>::factor + {n}× median of one joint MLL+grad; discard {warmup} then {reps} timed",
            n = case.joint_evals
        )),
    })
}

fn main() -> ExitCode {
    let mut path = None;
    let mut uncached = false;
    for arg in env::args().skip(1) {
        if arg == "--uncached" {
            uncached = true;
        } else if path.is_none() {
            path = Some(arg);
        }
    }
    let Some(path) = path else {
        eprintln!("usage: gprx-perf CASE.json [--uncached]");
        return ExitCode::from(2);
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
    let row = if uncached {
        run(&case, UncachedDistances)
    } else {
        run(&case, CachedDistances)
    };
    let row = match row {
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
