//! Fixed regression problems shared by tests and benches.
//!
//! `y` is the named function plus `noise_std · N(0, 1)` from the crate's
//! seeded `SmallRng` (`src/rng.rs`). The including module provides `rng`.

use super::rng::{small_rng, unit_normal};

/// `n` evenly spaced points on `[lo, hi]`.
pub fn linspace(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![lo];
    }
    let denom = (n - 1) as f64;
    (0..n)
        .map(|i| lo + (hi - lo) * (i as f64) / denom)
        .collect()
}

/// Forrester function `(6x - 2)² sin(12x - 4)`.
pub fn forrester(x: f64) -> f64 {
    let t = 6.0 * x - 2.0;
    t * t * (12.0 * x - 4.0).sin()
}

/// Weighted sphere `(x0 / 0.25)² + x1²`.
pub fn weighted_sphere(x0: f64, x1: f64) -> f64 {
    let a = x0 / 0.25;
    a * a + x1 * x1
}

/// 1-D Forrester on `linspace(0, 1, n)` with noise from `seed`.
pub fn forrester_xy(n: usize, seed: u64, noise_std: f64) -> (Vec<f64>, Vec<f64>) {
    let x = linspace(0.0, 1.0, n);
    let mut rng = small_rng(seed);
    let y = x
        .iter()
        .map(|&xi| forrester(xi) + noise_std * unit_normal(&mut rng))
        .collect();
    (x, y)
}

/// 2-D weighted sphere on a `side × side` grid of `[0, 1]²`, column-major `X`.
pub fn sphere_xy(side: usize, seed: u64, noise_std: f64) -> (Vec<f64>, Vec<f64>) {
    let n = side * side;
    let mut x = vec![0.0; n * 2];
    let denom = (side - 1) as f64;
    for row in 0..n {
        x[row] = (row % side) as f64 / denom;
        x[n + row] = (row / side) as f64 / denom;
    }
    let mut rng = small_rng(seed);
    let y = (0..n)
        .map(|row| weighted_sphere(x[row], x[n + row]) + noise_std * unit_normal(&mut rng))
        .collect();
    (x, y)
}
