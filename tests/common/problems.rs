//! Fixed regression problems shared by tests and benches.
//!
//! `y` is the named function plus `noise_std · N(0, 1)` from the crate's
//! seeded `SeededRng` (`src/rng.rs`). The including module provides `rng`.

use super::rng::{seeded_rng, unit_normal};

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
    let mut rng = seeded_rng(seed);
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
    let mut rng = seeded_rng(seed);
    let y = (0..n)
        .map(|row| weighted_sphere(x[row], x[n + row]) + noise_std * unit_normal(&mut rng))
        .collect();
    (x, y)
}

/// The fixed problem of the supplied-distance baseline (design §5.6, §15):
/// `n` training points, `q` queries, and the first `m` training points as
/// inducing points, uniform on `[0, 1]^d`. Matrices are column-major.
pub struct DistanceBaseline {
    pub n: usize,
    pub d: usize,
    pub q: usize,
    pub m: usize,
    /// `n × d` training coordinates.
    pub x: Vec<f64>,
    /// `n` targets: `Σ_k sin(2π x_k)` plus `0.1 · N(0, 1)`.
    pub y: Vec<f64>,
    /// `q × d` query coordinates.
    pub xq: Vec<f64>,
    /// `m × d` inducing coordinates (rows `0..m` of `x`).
    pub z: Vec<f64>,
    /// One more point (`1 × d`) and its target, for online inserts.
    pub x_new: Vec<f64>,
    pub y_new: f64,
}

/// [`DistanceBaseline`] at `n = 512`, `d = 4`, `q = 100`, `m = 64`, seed 0.
pub fn distance_baseline() -> DistanceBaseline {
    let (n, d, q, m) = (512, 4, 100, 64);
    let mut rng = seeded_rng(0);
    // Column-major `rows × d`: dimension `k` is entries `k * rows .. (k + 1) * rows`.
    let mut points = |rows: usize| -> Vec<f64> { (0..rows * d).map(|_| rng.unit()).collect() };
    let x = points(n + 1);
    let xq = points(q);
    let target = |x: &[f64], rows: usize, i: usize| -> f64 {
        (0..d)
            .map(|k| (2.0 * std::f64::consts::PI * x[i + k * rows]).sin())
            .sum()
    };
    let mut noise = seeded_rng(1);
    let y_all: Vec<f64> = (0..=n)
        .map(|i| target(&x, n + 1, i) + 0.1 * unit_normal(&mut noise))
        .collect();
    let rows_of = |x: &[f64], rows: usize, range: std::ops::Range<usize>| -> Vec<f64> {
        (0..d)
            .flat_map(|k| range.clone().map(move |i| x[i + k * rows]))
            .collect()
    };
    DistanceBaseline {
        n,
        d,
        q,
        m,
        x: rows_of(&x, n + 1, 0..n),
        y: y_all[..n].to_vec(),
        xq,
        z: rows_of(&x, n + 1, 0..m),
        x_new: rows_of(&x, n + 1, n..n + 1),
        y_new: y_all[n],
    }
}

/// [`DistanceBaseline`] as supplied distances: per-dimension `(Δ_k)²`
/// blocks (column-major), training square and train × query.
pub struct Supplied {
    pub train: Vec<Vec<f64>>,
    pub cross: Vec<Vec<f64>>,
    /// `Σ_k` of the blocks: the squared Euclidean distance.
    pub train_sum: Vec<f64>,
    pub cross_sum: Vec<f64>,
}

impl DistanceBaseline {
    /// The training square and the train × query block of the problem.
    pub fn supplied(&self) -> Supplied {
        let block = |a: &[f64], ra: usize, b: &[f64], rb: usize, k: usize| -> Vec<f64> {
            (0..rb)
                .flat_map(|j| (0..ra).map(move |i| (a[i + k * ra] - b[j + k * rb]).powi(2)))
                .collect()
        };
        let train: Vec<Vec<f64>> = (0..self.d)
            .map(|k| block(&self.x, self.n, &self.x, self.n, k))
            .collect();
        let cross: Vec<Vec<f64>> = (0..self.d)
            .map(|k| block(&self.x, self.n, &self.xq, self.q, k))
            .collect();
        let sum = |blocks: &[Vec<f64>]| -> Vec<f64> {
            (0..blocks[0].len())
                .map(|at| blocks.iter().map(|b| b[at]).sum())
                .collect()
        };
        Supplied {
            train_sum: sum(&train),
            cross_sum: sum(&cross),
            train,
            cross,
        }
    }
}
