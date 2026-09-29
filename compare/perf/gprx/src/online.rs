//! Online Exact cells (P3-6 / P3-7): the insert sequence from `n = 2` to `n`
//! against libgp, its stages, and the delete sequence back to 2.

use std::time::Instant;

use gprx::{Fixed, GaussianLikelihood, Gpr};

use crate::case::{Case, ResultRow};
use crate::rss::peak_rss_bytes;
use crate::shared::{point_at, prefix_colmajor, rbf_kernel};
use crate::timing;

fn make_gpr(case: &Case) -> Result<Gpr<Fixed>, String> {
    let likelihood =
        GaussianLikelihood::new(case.noise_variance_init).map_err(|e| e.to_string())?;
    Ok(Gpr::new(rbf_kernel(case.ard, &case.lengthscales_init)?, likelihood).with_optimizer(Fixed))
}

/// `OnlineGpr::insert` from `n = 2` to `n` (raw `y`); with `stages`, the
/// kernel / bordered LDLT / X·y split (needs `--features insert-stages`).
pub fn run_insert(case: &Case, stages: bool) -> Result<ResultRow, String> {
    if case.n_rows < 2 {
        return Err("n_rows must be at least 2".to_string());
    }
    if stages && !cfg!(feature = "insert-stages") {
        return Err("rebuild gprx-perf with --features insert-stages".to_string());
    }
    let warmup = timing::warmup_count();
    let reps = timing::timed_reps(case.n_rows);
    let d = case.n_cols;
    let n = case.n_rows;
    let x0 = prefix_colmajor(&case.x, n, d, 2);
    let mut insert_samples = Vec::with_capacity(reps);
    let mut kernel_samples = Vec::new();
    let mut border_samples = Vec::new();
    let mut rest_samples = Vec::new();
    if stages {
        kernel_samples.reserve(reps);
        border_samples.reserve(reps);
        rest_samples.reserve(reps);
    }
    for i in 0..warmup + reps {
        let gpr = make_gpr(case)?;
        let fitted = gpr
            .factor(&x0, 2, d, &case.y[..2])
            .map_err(|(_, e)| e.to_string())?;
        let mut online = fitted.into_online().map_err(|e| e.to_string())?;
        if stages {
            let _ = take_insert_stages_or_zero();
        }
        let start = Instant::now();
        for index in 2..n {
            let x_new = point_at(&case.x, n, d, index);
            online
                .insert(&x_new, case.y[index])
                .map_err(|e| e.to_string())?;
        }
        let dt = start.elapsed().as_secs_f64();
        if i >= warmup {
            insert_samples.push(dt);
            if stages {
                let (kernel, border, rest) = take_insert_stages_or_zero();
                kernel_samples.push(kernel);
                border_samples.push(border);
                rest_samples.push(rest);
            }
        }
    }
    let (lo, hi) = timing::min_max(&insert_samples);
    Ok(ResultRow {
        lib: "gprx".to_string(),
        name: case.name.clone(),
        status: "ok".to_string(),
        factor_s: Some(timing::median(&insert_samples)),
        factor_min_s: Some(lo),
        factor_max_s: Some(hi),
        eval_s: None,
        eval_min_s: None,
        eval_max_s: None,
        predict_s: None,
        predict_min_s: None,
        predict_max_s: None,
        joint_evals: None,
        peak_rss_bytes: Some(peak_rss_bytes()?),
        warmup: Some(warmup as u64),
        reps: Some(reps as u64),
        kernel_s: stages.then(|| timing::median(&kernel_samples)),
        border_s: stages.then(|| timing::median(&border_samples)),
        rest_s: stages.then(|| timing::median(&rest_samples)),
        note: Some(if stages {
            "OnlineGpr::insert from n=2 to n; stages kernel / bordered LDLT / X·y".to_string()
        } else {
            "OnlineGpr::insert from n=2 to n; first two points untimed".to_string()
        }),
    })
}

/// `OnlineGpr::delete` from `n` down to 2, last `PointId` first; the
/// inserts before the clock are untimed.
pub fn run_delete(case: &Case) -> Result<ResultRow, String> {
    if case.n_rows < 2 {
        return Err("n_rows must be at least 2".to_string());
    }
    let warmup = timing::warmup_count();
    let reps = timing::timed_reps(case.n_rows);
    let d = case.n_cols;
    let n = case.n_rows;
    let x0 = prefix_colmajor(&case.x, n, d, 2);
    let mut delete_samples = Vec::with_capacity(reps);
    for i in 0..warmup + reps {
        let gpr = make_gpr(case)?;
        let fitted = gpr
            .factor(&x0, 2, d, &case.y[..2])
            .map_err(|(_, e)| e.to_string())?;
        let mut online = fitted.into_online().map_err(|e| e.to_string())?;
        for index in 2..n {
            let x_new = point_at(&case.x, n, d, index);
            online
                .insert(&x_new, case.y[index])
                .map_err(|e| e.to_string())?;
        }
        let start = Instant::now();
        while online.n() > 2 {
            let id = *online
                .point_ids()
                .last()
                .ok_or_else(|| "point_ids empty during delete clock".to_string())?;
            online.delete(id).map_err(|e| e.to_string())?;
        }
        let dt = start.elapsed().as_secs_f64();
        if i >= warmup {
            delete_samples.push(dt);
        }
    }
    let (lo, hi) = timing::min_max(&delete_samples);
    Ok(ResultRow {
        lib: "gprx".to_string(),
        name: case.name.clone(),
        status: "ok".to_string(),
        factor_s: Some(timing::median(&delete_samples)),
        factor_min_s: Some(lo),
        factor_max_s: Some(hi),
        eval_s: None,
        eval_min_s: None,
        eval_max_s: None,
        predict_s: None,
        predict_min_s: None,
        predict_max_s: None,
        joint_evals: None,
        peak_rss_bytes: Some(peak_rss_bytes()?),
        warmup: Some(warmup as u64),
        reps: Some(reps as u64),
        kernel_s: None,
        border_s: None,
        rest_s: None,
        note: Some(
            "OnlineGpr::delete from n to 2; last remaining PointId each step; insert untimed"
                .to_string(),
        ),
    })
}

fn take_insert_stages_or_zero() -> (f64, f64, f64) {
    #[cfg(feature = "insert-stages")]
    {
        gprx::internals::take_insert_stages()
    }
    #[cfg(not(feature = "insert-stages"))]
    {
        (0.0, 0.0, 0.0)
    }
}
