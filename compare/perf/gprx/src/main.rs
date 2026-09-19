//! One gprx cell: `Gpr<Fixed>::factor`, then N joint MLL+grad evals, then predict 100.

use std::env;
use std::fs;
use std::process::ExitCode;
use std::time::Instant;

use gprx::kernel::{KernelSpec, RbfArdKernel, RbfKernel};
use gprx::transform::StandardizeTarget;
use gprx::{Fixed, GaussianLikelihood, Gpr};

#[path = "../../case_schema.rs"]
mod case_schema;
#[path = "../../rss_win.rs"]
mod peak_rss;

use case_schema::{Case, ResultRow};

fn na_row(name: &str, note: String) -> ResultRow {
    ResultRow {
        lib: "gprx".to_string(),
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
    let gpr = Gpr::new(kernel, likelihood)
        .with_target_transform(StandardizeTarget::new())
        .with_optimizer(Fixed);
    let factor_start = Instant::now();
    let mut fitted = gpr
        .factor(&case.x, case.n_rows, case.n_cols, &case.y)
        .map_err(|(_, e)| e.to_string())?;
    let factor_s = factor_start.elapsed().as_secs_f64();

    let n_params = fitted.num_params();
    let mut params = vec![0.0; n_params];
    fitted.get_params(&mut params).map_err(|e| e.to_string())?;
    let mut grad = vec![0.0; n_params];
    let eval_start = Instant::now();
    for _ in 0..case.joint_evals {
        fitted
            .value_and_gradient_into(&params, &mut grad)
            .map_err(|e| e.to_string())?;
    }
    let eval_s = eval_start.elapsed().as_secs_f64();

    let predict_start = Instant::now();
    fitted
        .predict(&case.xs, case.xs_n_rows, case.xs_n_cols)
        .map_err(|e| e.to_string())?;
    let predict_s = predict_start.elapsed().as_secs_f64();
    Ok(ResultRow {
        lib: "gprx".to_string(),
        name: case.name.clone(),
        status: "ok".to_string(),
        factor_s: Some(factor_s),
        eval_s: Some(eval_s),
        predict_s: Some(predict_s),
        joint_evals: Some(case.joint_evals),
        peak_rss_bytes: Some(peak_rss::peak_rss_bytes()?),
        note: Some(format!(
            "Gpr<Fixed>::factor + {} joint MLL+grad at the same theta",
            case.joint_evals
        )),
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
