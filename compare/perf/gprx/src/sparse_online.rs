//! One Sparse-online cell: prefix `factor` + `into_online` untimed, then
//! 32 public ops or 32× `Sgpr<Fixed>::factor` as one wall clock.

use std::env;
use std::fs;
use std::process::ExitCode;
use std::time::Instant;

use gprx::kernel::{KernelSpec, RbfArdKernel, RbfKernel};
use gprx::{Fixed, GaussianLikelihood, OnlineSgpr, Sgpr};

#[allow(dead_code)]
#[path = "../../case_schema.rs"]
mod case_schema;
#[path = "../../rss_win.rs"]
mod peak_rss;
#[path = "../../timing.rs"]
mod timing;

use case_schema::{SparseOnlineCase, SparseOnlineOp, SparseOnlineResult};

fn na_row(lib: &str, name: &str, note: String) -> SparseOnlineResult {
    SparseOnlineResult {
        lib: lib.to_string(),
        name: name.to_string(),
        status: "na".to_string(),
        ops_s: None,
        ops_min_s: None,
        ops_max_s: None,
        peak_rss_bytes: None,
        warmup: None,
        reps: None,
        note: Some(note),
    }
}

fn make_kernel(case: &SparseOnlineCase) -> Result<KernelSpec, String> {
    if case.ard {
        let spec = RbfArdKernel::new(&case.lengthscales_init).map_err(|e| e.to_string())?;
        Ok(KernelSpec::from(spec))
    } else {
        let ell = case
            .lengthscales_init
            .first()
            .copied()
            .ok_or_else(|| "missing lengthscale".to_string())?;
        Ok(KernelSpec::from(
            RbfKernel::new(ell).map_err(|e| e.to_string())?,
        ))
    }
}

fn point_at(values: &[f64], n: usize, d: usize, index: usize) -> Vec<f64> {
    (0..d).map(|feature| values[feature * n + index]).collect()
}

fn prefix_colmajor(values: &[f64], n: usize, d: usize, keep: usize) -> Vec<f64> {
    (0..d)
        .flat_map(|feature| (0..keep).map(move |i| values[feature * n + i]))
        .collect()
}

fn pack_indices(src: &[f64], n_src: usize, d: usize, indices: &[usize]) -> Vec<f64> {
    let keep = indices.len();
    let mut out = vec![0.0; keep * d];
    for (new_i, &old_i) in indices.iter().enumerate() {
        for dim in 0..d {
            out[dim * keep + new_i] = src[dim * n_src + old_i];
        }
    }
    out
}

fn apply_incremental(
    online: &mut OnlineSgpr<Fixed>,
    case: &SparseOnlineCase,
    op: &SparseOnlineOp,
) -> Result<(), String> {
    let d = case.n_cols;
    match op.kind.as_str() {
        "insert" => {
            let index = op.pop_index.ok_or("insert pop_index")?;
            let x_new = point_at(&case.x, case.n_rows, d, index);
            online
                .insert(&x_new, case.y[index])
                .map_err(|e| e.to_string())?;
        }
        "delete" => {
            let slot = op.slot.ok_or("delete slot")?;
            let id = online.point_ids()[slot];
            online.delete(id).map_err(|e| e.to_string())?;
        }
        "insert_inducing" => {
            let index = op.pop_index.ok_or("insert_inducing pop_index")?;
            let z_new = point_at(&case.z, case.n_inducing, d, index);
            online.insert_inducing(&z_new).map_err(|e| e.to_string())?;
        }
        "delete_inducing" => {
            let slot = op.slot.ok_or("delete_inducing slot")?;
            let id = online.inducing_ids()[slot];
            online.delete_inducing(id).map_err(|e| e.to_string())?;
        }
        other => return Err(format!("unknown op {other}")),
    }
    Ok(())
}

fn apply_indices(
    op: &SparseOnlineOp,
    x_in: &mut Vec<usize>,
    z_in: &mut Vec<usize>,
) -> Result<(), String> {
    match op.kind.as_str() {
        "insert" => {
            x_in.push(op.pop_index.ok_or("insert pop_index")?);
        }
        "delete" => {
            let slot = op.slot.ok_or("delete slot")?;
            x_in.remove(slot);
        }
        "insert_inducing" => {
            z_in.push(op.pop_index.ok_or("insert_inducing pop_index")?);
        }
        "delete_inducing" => {
            let slot = op.slot.ok_or("delete_inducing slot")?;
            z_in.remove(slot);
        }
        other => return Err(format!("unknown op {other}")),
    }
    Ok(())
}

fn prefix_online(case: &SparseOnlineCase) -> Result<OnlineSgpr<Fixed>, String> {
    if case.start_n > case.n_max || case.start_m > case.m_max {
        return Err("prefix longer than population".to_string());
    }
    let d = case.n_cols;
    let x0 = prefix_colmajor(&case.x, case.n_rows, d, case.start_n);
    let y0 = case.y[..case.start_n].to_vec();
    let z0 = prefix_colmajor(&case.z, case.n_inducing, d, case.start_m);
    let likelihood =
        GaussianLikelihood::new(case.noise_variance_init).map_err(|e| e.to_string())?;
    let trainer = Sgpr::new(make_kernel(case)?, likelihood).with_optimizer(Fixed);
    trainer
        .factor(&x0, case.start_n, d, &y0, &z0, case.start_m)
        .map(|fitted| fitted.into_online())
        .map_err(|(_, e)| e.to_string())
}

fn factor_current(case: &SparseOnlineCase, x_in: &[usize], z_in: &[usize]) -> Result<(), String> {
    let d = case.n_cols;
    let x = pack_indices(&case.x, case.n_rows, d, x_in);
    let y: Vec<f64> = x_in.iter().map(|&i| case.y[i]).collect();
    let z = pack_indices(&case.z, case.n_inducing, d, z_in);
    let likelihood =
        GaussianLikelihood::new(case.noise_variance_init).map_err(|e| e.to_string())?;
    let trainer = Sgpr::new(make_kernel(case)?, likelihood).with_optimizer(Fixed);
    trainer
        .factor(&x, x_in.len(), d, &y, &z, z_in.len())
        .map(|_| ())
        .map_err(|(_, e)| e.to_string())
}

fn finish_row(
    lib: &str,
    case: &SparseOnlineCase,
    samples: &[f64],
    warmup: usize,
    reps: usize,
    note: String,
) -> Result<SparseOnlineResult, String> {
    let (ops_min, ops_max) = timing::min_max(samples);
    Ok(SparseOnlineResult {
        lib: lib.to_string(),
        name: case.name.clone(),
        status: "ok".to_string(),
        ops_s: Some(timing::median(samples)),
        ops_min_s: Some(ops_min),
        ops_max_s: Some(ops_max),
        peak_rss_bytes: Some(peak_rss::peak_rss_bytes()?),
        warmup: Some(warmup as u64),
        reps: Some(reps as u64),
        note: Some(note),
    })
}

fn run_incremental(case: &SparseOnlineCase) -> Result<SparseOnlineResult, String> {
    let warmup = timing::warmup_count();
    let reps = timing::timed_reps(case.n_rows);
    let mut samples = Vec::with_capacity(reps);
    for i in 0..warmup + reps {
        let mut online = prefix_online(case)?;
        let start = Instant::now();
        for op in &case.ops {
            apply_incremental(&mut online, case, op)?;
        }
        let dt = start.elapsed().as_secs_f64();
        if i >= warmup {
            samples.push(dt);
        }
    }
    finish_row(
        "gprx-inc",
        case,
        &samples,
        warmup,
        reps,
        format!("OnlineSgpr 32 public ops; prefix untimed; discard {warmup} then {reps} timed"),
    )
}

fn run_full(case: &SparseOnlineCase) -> Result<SparseOnlineResult, String> {
    let warmup = timing::warmup_count();
    let reps = timing::timed_reps(case.n_rows);
    let mut samples = Vec::with_capacity(reps);
    for i in 0..warmup + reps {
        let mut x_in: Vec<usize> = (0..case.start_n).collect();
        let mut z_in: Vec<usize> = (0..case.start_m).collect();
        let start = Instant::now();
        for op in &case.ops {
            apply_indices(op, &mut x_in, &mut z_in)?;
            factor_current(case, &x_in, &z_in)?;
        }
        let dt = start.elapsed().as_secs_f64();
        if i >= warmup {
            samples.push(dt);
        }
    }
    finish_row(
        "gprx-full",
        case,
        &samples,
        warmup,
        reps,
        format!(
            "32× Sgpr<Fixed>::factor after each op; prefix untimed; discard {warmup} then {reps} timed"
        ),
    )
}

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: gprx-sparse-online-perf CASE.json incremental|full");
        return ExitCode::from(2);
    };
    let Some(mode) = args.next() else {
        eprintln!("usage: gprx-sparse-online-perf CASE.json incremental|full");
        return ExitCode::from(2);
    };
    let text = match fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(1);
        }
    };
    let case: SparseOnlineCase = match serde_json::from_str(&text) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(1);
        }
    };
    let row = match mode.as_str() {
        "incremental" => run_incremental(&case),
        "full" => run_full(&case),
        other => Err(format!("unknown mode {other}")),
    };
    let row = match row {
        Ok(row) => row,
        Err(e) => na_row(
            match mode.as_str() {
                "full" => "gprx-full",
                _ => "gprx-inc",
            },
            &case.name,
            e,
        ),
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
