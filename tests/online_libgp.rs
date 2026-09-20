//! libgp `add_pattern` goldens for [`OnlineGpr::insert`] (P3-6).
//!
//! JSON under `compare/goldens/` is produced by `just gen-online-goldens`.
//! This file only reads the committed bytes; `cargo test` must not invoke
//! C++ or Python.

use gprx::kernel::{KernelSpec, RbfArdKernel, RbfKernel};
use gprx::{Fixed, GaussianLikelihood, Gpr, OnlineGpr, Prediction};
use serde::Deserialize;

const TOL: f64 = 1e-8;

#[derive(Debug, Deserialize)]
struct OnlineLibgpGolden {
    ard: bool,
    lengthscales: Vec<f64>,
    noise_variance: f64,
    n_rows: usize,
    n_cols: usize,
    x: Vec<f64>,
    y: Vec<f64>,
    xs_n_rows: usize,
    xs_n_cols: usize,
    xs: Vec<f64>,
    start_n: usize,
    steps: Vec<OnlineLibgpStep>,
}

#[derive(Debug, Deserialize)]
struct OnlineLibgpStep {
    n: usize,
    mean: Vec<f64>,
    observation_variance: Vec<f64>,
    neg_log_marginal_likelihood: f64,
}

fn assert_close(actual: f64, expected: f64) {
    let scale = expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= TOL * scale,
        "actual={actual}, expected={expected}"
    );
}

fn assert_pred_close(got: &Prediction, mean: &[f64], variance: &[f64]) {
    assert_eq!(got.mean.len(), mean.len());
    assert_eq!(got.variance.len(), variance.len());
    for (a, b) in got.mean.iter().zip(mean.iter()) {
        assert_close(*a, *b);
    }
    for (a, b) in got.variance.iter().zip(variance.iter()) {
        assert_close(*a, *b);
    }
}

fn point_at(x: &[f64], n: usize, d: usize, index: usize) -> Vec<f64> {
    (0..d).map(|feature| x[feature * n + index]).collect()
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn replay(golden: &OnlineLibgpGolden) {
    let kernel = if golden.ard {
        KernelSpec::from(RbfArdKernel::new(&golden.lengthscales).expect("ℓ"))
    } else {
        KernelSpec::from(RbfKernel::new(golden.lengthscales[0]).expect("ℓ"))
    };
    let likelihood = GaussianLikelihood::new(golden.noise_variance).expect("noise");
    let n = golden.n_rows;
    let d = golden.n_cols;
    assert_eq!(golden.xs.len(), golden.xs_n_rows * golden.xs_n_cols);
    let start = golden.start_n;
    let x0: Vec<f64> = (0..d)
        .flat_map(|feature| (0..start).map(move |i| golden.x[feature * n + i]))
        .collect();
    let fitted = Gpr::new(kernel, likelihood)
        .with_optimizer(Fixed)
        .factor(&x0, start, d, &golden.y[..start])
        .map_err(|(_, e)| e)
        .expect("factor");
    let mut online = fitted.into_online().expect("into_online");
    let mut step = 0usize;
    check_step(
        &online,
        &golden.steps[step],
        &golden.xs,
        golden.xs_n_rows,
        d,
    );
    step += 1;
    for index in start..n {
        let x_new = point_at(&golden.x, n, d, index);
        online.insert(&x_new, golden.y[index]).expect("insert");
        check_step(
            &online,
            &golden.steps[step],
            &golden.xs,
            golden.xs_n_rows,
            d,
        );
        step += 1;
    }
    assert_eq!(step, golden.steps.len());
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn check_step(
    online: &OnlineGpr<Fixed>,
    step: &OnlineLibgpStep,
    xs: &[f64],
    n_query: usize,
    d: usize,
) {
    assert_eq!(online.n(), step.n);
    let got = online.predict(xs, n_query, d).expect("predict");
    assert_pred_close(&got, &step.mean, &step.observation_variance);
    let nlml = online.neg_log_marginal_likelihood().expect("nlml");
    assert_close(nlml, step.neg_log_marginal_likelihood);
}

#[test]
fn online_insert_forrester_matches_libgp() {
    let golden: OnlineLibgpGolden = serde_json::from_str(include_str!(
        "../compare/goldens/online_libgp_forrester.json"
    ))
    .expect("parse forrester golden");
    replay(&golden);
}

#[test]
fn online_insert_sphere_matches_libgp() {
    let golden: OnlineLibgpGolden =
        serde_json::from_str(include_str!("../compare/goldens/online_libgp_sphere.json"))
            .expect("parse sphere golden");
    replay(&golden);
}
