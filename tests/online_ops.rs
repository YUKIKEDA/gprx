//! Random insert/delete sequences vs [`Gpr<Fixed>::factor`] (P3-5).
//!
//! Each step compares public [`OnlineGpr`] numerics and point identity
//! to a batch factor at the same `θ`. This file uses only the public API.

use std::collections::HashSet;

use gprx::kernel::{KernelSpec, MaternKernel, MaternNu, RbfArdKernel, RbfKernel, WhiteKernel};
use gprx::{Fixed, GaussianLikelihood, Gpr, OnlineGpr, PointId, Prediction};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

const TOL: f64 = 1e-12;
const OPS: usize = 32;
const N_MAX: usize = 16;
const SEEDS: [u64; 3] = [0, 1, 2];
const Y0: [f64; 4] = [0.0, 1.0, 0.5, 0.25];
const X1: [f64; 4] = [0.0, 1.0, 2.0, 3.0];
const X2: [f64; 8] = [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0];

fn assert_close(actual: f64, expected: f64) {
    let scale = expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= TOL * scale,
        "actual={actual}, expected={expected}"
    );
}

fn assert_pred_close(got: &Prediction, want: &Prediction) {
    assert_eq!(got.mean.len(), want.mean.len());
    for (a, b) in got.mean.iter().zip(want.mean.iter()) {
        assert_close(*a, *b);
    }
    for (a, b) in got.variance.iter().zip(want.variance.iter()) {
        assert_close(*a, *b);
    }
}

fn assert_slice_close(got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len());
    for (a, b) in got.iter().zip(want.iter()) {
        assert_close(*a, *b);
    }
}

fn sample_coord(rng: &mut SmallRng) -> f64 {
    4.0 * rng.random::<f64>()
}

fn append_colmajor(x: &mut Vec<f64>, n: usize, d: usize, x_new: &[f64]) {
    let mut next = vec![0.0; (n + 1) * d];
    for feature in 0..d {
        for i in 0..n {
            next[feature * (n + 1) + i] = x[feature * n + i];
        }
        next[feature * (n + 1) + n] = x_new[feature];
    }
    *x = next;
}

fn remove_colmajor(x: &mut Vec<f64>, n: usize, d: usize, index: usize) {
    let mut next = Vec::with_capacity((n - 1) * d);
    for feature in 0..d {
        for i in 0..n {
            if i != index {
                next.push(x[feature * n + i]);
            }
        }
    }
    *x = next;
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn factor_oracle(
    kernel: KernelSpec,
    likelihood: GaussianLikelihood,
    x: &[f64],
    n: usize,
    d: usize,
    y: &[f64],
) -> gprx::FittedGpr<Fixed> {
    Gpr::new(kernel, likelihood)
        .with_optimizer(Fixed)
        .factor(x, n, d, y)
        .map_err(|(_, e)| e)
        .expect("factor")
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
#[allow(clippy::too_many_arguments)]
fn assert_matches_factor(
    online: &OnlineGpr<Fixed>,
    kernel: KernelSpec,
    likelihood: GaussianLikelihood,
    x: &[f64],
    y: &[f64],
    xs: &[f64],
    d: usize,
    deleted: &HashSet<PointId>,
) {
    let n = online.n();
    assert_eq!(n, y.len());
    assert_eq!(online.point_ids().len(), n);
    assert_eq!(online.x().len(), n * d);
    assert_eq!(online.y().len(), n);
    for id in deleted {
        assert!(
            !online.point_ids().contains(id),
            "deleted PointId still present"
        );
    }

    let full = factor_oracle(kernel, likelihood, x, n, d, y);
    assert_eq!(full.n(), n);
    assert_slice_close(online.x(), full.x());
    assert_slice_close(online.y(), full.y());
    assert_slice_close(online.x(), x);
    assert_slice_close(online.y(), y);

    let n_query = xs.len() / d;
    let got = online.predict(xs, n_query, d).expect("online predict");
    let want = full.predict(xs, n_query, d).expect("factor predict");
    assert_pred_close(&got, &want);

    let nlml_online = online.neg_log_marginal_likelihood().expect("online nlml");
    let nlml_full = full.neg_log_marginal_likelihood().expect("factor nlml");
    assert_close(nlml_online, nlml_full);
    assert_slice_close(online.alpha(), full.alpha());
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn run_sequence(kernel: KernelSpec, x0: &[f64], y0: &[f64], d: usize, xs: &[f64], seed: u64) {
    let likelihood = GaussianLikelihood::new(0.1).expect("noise");
    let fitted = factor_oracle(kernel.clone(), likelihood, x0, y0.len(), d, y0);
    let mut online = fitted.into_online().expect("into_online");
    let mut x = x0.to_vec();
    let mut y = y0.to_vec();
    let mut deleted = HashSet::new();
    let mut rng = SmallRng::seed_from_u64(seed);

    assert_matches_factor(&online, kernel.clone(), likelihood, &x, &y, xs, d, &deleted);

    for _ in 0..OPS {
        let n = online.n();
        let insert = n == 2 || (n < N_MAX && rng.random::<bool>());
        if insert {
            let mut x_new = vec![0.0; d];
            for value in &mut x_new {
                *value = sample_coord(&mut rng);
            }
            let y_new = sample_coord(&mut rng);
            let id = online.insert(&x_new, y_new).expect("insert");
            assert!(!deleted.contains(&id), "insert reused a deleted PointId");
            assert_eq!(online.point_ids().last().copied(), Some(id));
            append_colmajor(&mut x, n, d, &x_new);
            y.push(y_new);
        } else {
            let ids = online.point_ids();
            let index = rng.random_range(0..ids.len());
            let id = ids[index];
            online.delete(id).expect("delete");
            deleted.insert(id);
            remove_colmajor(&mut x, n, d, index);
            y.remove(index);
        }
        assert_matches_factor(&online, kernel.clone(), likelihood, &x, &y, xs, d, &deleted);
    }
}

fn run_seeds(kernel: KernelSpec, x0: &[f64], y0: &[f64], d: usize, xs: &[f64]) {
    for seed in SEEDS {
        run_sequence(kernel.clone(), x0, y0, d, xs, seed);
    }
}

#[test]
fn online_ops_rbf_matches_factor() {
    run_seeds(
        KernelSpec::from(RbfKernel::new(1.0).expect("ℓ")),
        &X1,
        &Y0,
        1,
        &[0.5],
    );
}

#[test]
fn online_ops_matern_three_halves_matches_factor() {
    run_seeds(
        KernelSpec::from(MaternKernel::new(1.0, MaternNu::ThreeHalves).expect("ℓ")),
        &X1,
        &Y0,
        1,
        &[0.5],
    );
}

#[test]
fn online_ops_rbf_ard_2d_matches_factor() {
    run_seeds(
        KernelSpec::from(RbfArdKernel::new(&[1.0, 1.5]).expect("ℓ")),
        &X2,
        &Y0,
        2,
        &[0.25, 0.75],
    );
}

#[test]
fn online_ops_rbf_plus_white_matches_factor() {
    run_seeds(
        KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"))
            + KernelSpec::from(WhiteKernel::new(0.05).expect("white")),
        &X1,
        &Y0,
        1,
        &[0.5],
    );
}
