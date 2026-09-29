//! One friedrich cell. Factor at fixed theta. No public MLL+grad: eval is N/A.

use std::env;
use std::fs;
use std::process::ExitCode;
use std::time::Instant;

use friedrich::gaussian_process::GaussianProcess;
use friedrich::kernel::SquaredExp;

#[allow(dead_code)]
#[path = "../../gprx/src/case.rs"]
mod case_schema;
#[path = "../../gprx/src/rss.rs"]
mod peak_rss;
#[path = "../../gprx/src/timing.rs"]
mod timing;

use case_schema::{Case, ResultRow};

fn unpack_rows(x: &[f64], n_rows: usize, n_cols: usize) -> Vec<Vec<f64>> {
    (0..n_rows)
        .map(|row| (0..n_cols).map(|col| x[col * n_rows + row]).collect())
        .collect()
}

fn zscore(y: &[f64]) -> Vec<f64> {
    let n = y.len() as f64;
    let mean = y.iter().sum::<f64>() / n;
    let var = y.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    let std = var.sqrt();
    let std = if std == 0.0 { 1.0 } else { std };
    y.iter().map(|v| (v - mean) / std).collect()
}

fn na_row(name: &str, note: String) -> ResultRow {
    ResultRow {
        lib: "friedrich".to_string(),
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

fn run(case: &Case) -> ResultRow {
    if case.ard {
        return na_row(
            &case.name,
            "friedrich SquaredExp is isotropic; no ARD lengthscale".to_string(),
        );
    }
    let ell = match case.lengthscales_init.first() {
        Some(v) => *v,
        None => return na_row(&case.name, "missing lengthscale".to_string()),
    };
    let inputs = unpack_rows(&case.x, case.n_rows, case.n_cols);
    let outputs = zscore(&case.y);
    let queries = unpack_rows(&case.xs, case.xs_n_rows, case.xs_n_cols);
    let noise_std = case.noise_variance_init.sqrt();
    let warmup = timing::warmup_count();
    let reps = timing::timed_reps(case.n_rows);
    let mut factor_samples = Vec::with_capacity(reps);
    let mut gp = None;
    for i in 0..warmup + reps {
        drop(gp.take());
        let start = Instant::now();
        let next = GaussianProcess::builder(inputs.clone(), outputs.clone())
            .set_noise(noise_std)
            .set_kernel(SquaredExp::new(ell, 1.0))
            .train();
        let dt = start.elapsed().as_secs_f64();
        if i >= warmup {
            factor_samples.push(dt);
        }
        gp = Some(next);
    }
    let gp = gp.expect("timed_reps is at least 1");
    let mut predict_samples = Vec::with_capacity(reps);
    for i in 0..warmup + reps {
        let start = Instant::now();
        let _ = gp.predict_mean_variance(&queries);
        let dt = start.elapsed().as_secs_f64();
        if i >= warmup {
            predict_samples.push(dt);
        }
    }
    let peak = match peak_rss::peak_rss_bytes() {
        Ok(v) => Some(v),
        Err(e) => {
            return na_row(&case.name, e);
        }
    };
    let (factor_min, factor_max) = timing::min_max(&factor_samples);
    let (predict_min, predict_max) = timing::min_max(&predict_samples);
    ResultRow {
        lib: "friedrich".to_string(),
        name: case.name.clone(),
        status: "ok".to_string(),
        factor_s: Some(timing::median(&factor_samples)),
        factor_min_s: Some(factor_min),
        factor_max_s: Some(factor_max),
        eval_s: None,
        eval_min_s: None,
        eval_max_s: None,
        predict_s: Some(timing::median(&predict_samples)),
        predict_min_s: Some(predict_min),
        predict_max_s: Some(predict_max),
        joint_evals: None,
        peak_rss_bytes: peak,
        warmup: Some(warmup as u64),
        reps: Some(reps as u64),
        kernel_s: None,
        border_s: None,
        rest_s: None,
        note: Some(format!(
            "factor at fixed theta; no public MLL+grad (case asked {} evals); discard {warmup} then {reps} timed",
            case.joint_evals
        )),
    }
}

fn main() -> ExitCode {
    let path = match env::args().nth(1) {
        Some(p) => p,
        None => {
            eprintln!("usage: friedrich-perf CASE.json");
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
    let row = run(&case);
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
