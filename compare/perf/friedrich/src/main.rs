//! One friedrich cell. Factor at fixed theta. No public MLL+grad: eval is N/A.

use std::env;
use std::fs;
use std::process::ExitCode;
use std::time::Instant;

use friedrich::gaussian_process::GaussianProcess;
use friedrich::kernel::SquaredExp;

#[path = "../../case_schema.rs"]
mod case_schema;
#[path = "../../rss_win.rs"]
mod peak_rss;

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
        eval_s: None,
        predict_s: None,
        joint_evals: None,
        peak_rss_bytes: None,
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
    let factor_start = Instant::now();
    let gp = GaussianProcess::builder(inputs, outputs)
        .set_noise(noise_std)
        .set_kernel(SquaredExp::new(ell, 1.0))
        .train();
    let factor_s = factor_start.elapsed().as_secs_f64();
    let predict_start = Instant::now();
    let _ = gp.predict_mean_variance(&queries);
    let predict_s = predict_start.elapsed().as_secs_f64();
    let peak = match peak_rss::peak_rss_bytes() {
        Ok(v) => Some(v),
        Err(e) => {
            return na_row(&case.name, e);
        }
    };
    ResultRow {
        lib: "friedrich".to_string(),
        name: case.name.clone(),
        status: "ok".to_string(),
        factor_s: Some(factor_s),
        eval_s: None,
        predict_s: Some(predict_s),
        joint_evals: None,
        peak_rss_bytes: peak,
        note: Some(format!(
            "factor at fixed theta; no public MLL+grad (case asked {} evals)",
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
