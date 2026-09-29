//! Warmup + timed-rep helpers (the P2B-16 clock).

use std::env;

pub fn warmup_count() -> usize {
    env_usize("PERF_WARMUP", 1)
}

/// Timed reps. `PERF_REPS` wins. Otherwise more samples when `n` is small.
pub fn timed_reps(n_rows: usize) -> usize {
    env_usize("PERF_REPS", default_reps(n_rows)).max(1)
}

pub fn default_reps(n_rows: usize) -> usize {
    if n_rows <= 256 {
        51
    } else if n_rows <= 1024 {
        21
    } else {
        7
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

pub fn median(samples: &[f64]) -> f64 {
    let mut xs = samples.to_vec();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = xs.len();
    assert!(!xs.is_empty(), "median of empty samples");
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        0.5 * (xs[n / 2 - 1] + xs[n / 2])
    }
}

pub fn min_max(samples: &[f64]) -> (f64, f64) {
    let mut iter = samples.iter().copied();
    let first = iter.next().expect("min_max of empty samples");
    iter.fold((first, first), |(lo, hi), x| (lo.min(x), hi.max(x)))
}
