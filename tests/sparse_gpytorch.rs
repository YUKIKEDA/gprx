//! GPyTorch SGPR / SVGP goldens for [`Sgpr<Fixed>::factor`] and
//! [`Svgp<Fixed>::factor`] (P4-11).
//!
//! JSON under `compare/goldens/` is produced by `just gen-sparse-goldens`.
//! This file only reads the committed bytes; `cargo test` must not invoke
//! Python.

use gprx::kernel::{KernelSpec, MaternKernel, MaternNu, RbfArdKernel, RbfKernel, WhiteKernel};
use gprx::{Fixed, GaussianLikelihood, PredictOptions, Prediction, Sgpr, Svgp, VarianceKind};
use serde::Deserialize;

const TOL: f64 = 1e-8;

#[derive(Debug, Deserialize)]
struct SparseGpytorchGolden {
    n_rows: usize,
    n_cols: usize,
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    m: usize,
    xs_n_rows: usize,
    xs_n_cols: usize,
    xs: Vec<f64>,
    noise_variance: f64,
    kernels: Vec<SparseGpytorchKernel>,
}

#[derive(Debug, Deserialize)]
struct SparseGpytorchKernel {
    name: String,
    lengthscales: Vec<f64>,
    white_variance: Option<f64>,
    #[serde(default)]
    neg_log_marginal_likelihood: Option<f64>,
    #[serde(default)]
    neg_elbo: Option<f64>,
    mean: Vec<f64>,
    observation_variance: Vec<f64>,
    latent_variance: Vec<f64>,
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

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
#[allow(clippy::panic)]
fn kernel_spec(kernel: &SparseGpytorchKernel) -> KernelSpec {
    match kernel.name.as_str() {
        "rbf" => KernelSpec::from(RbfKernel::new(kernel.lengthscales[0]).expect("ℓ")),
        "matern32" => KernelSpec::from(
            MaternKernel::new(kernel.lengthscales[0], MaternNu::ThreeHalves).expect("ℓ"),
        ),
        "rbf_ard" => KernelSpec::from(RbfArdKernel::new(&kernel.lengthscales).expect("ℓ")),
        "rbf_white" => {
            KernelSpec::from(RbfKernel::new(kernel.lengthscales[0]).expect("ℓ"))
                + KernelSpec::from(
                    WhiteKernel::new(kernel.white_variance.expect("white")).expect("white"),
                )
        }
        other => panic!("unknown kernel {other}"),
    }
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn replay_sgpr(golden: &SparseGpytorchGolden) {
    let n = golden.n_rows;
    let d = golden.n_cols;
    assert_eq!(golden.xs.len(), golden.xs_n_rows * golden.xs_n_cols);
    for kernel in &golden.kernels {
        let expected = kernel.neg_log_marginal_likelihood.expect("sgpr nlml");
        let fitted = Sgpr::new(
            kernel_spec(kernel),
            GaussianLikelihood::new(golden.noise_variance).expect("noise"),
        )
        .with_optimizer(Fixed)
        .factor(&golden.x, n, d, &golden.y, &golden.z, golden.m)
        .map_err(|(_, e)| e)
        .expect("sgpr factor");
        let obs = fitted
            .predict(&golden.xs, golden.xs_n_rows, d)
            .expect("predict");
        assert_pred_close(&obs, &kernel.mean, &kernel.observation_variance);
        let latent = fitted
            .predict_with(
                &golden.xs,
                golden.xs_n_rows,
                d,
                PredictOptions {
                    variance_kind: VarianceKind::Latent,
                },
            )
            .expect("predict latent");
        assert_pred_close(&latent, &kernel.mean, &kernel.latent_variance);
        assert_close(
            fitted.neg_log_marginal_likelihood().expect("nlml"),
            expected,
        );
    }
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn replay_svgp(golden: &SparseGpytorchGolden) {
    let n = golden.n_rows;
    let d = golden.n_cols;
    assert_eq!(golden.xs.len(), golden.xs_n_rows * golden.xs_n_cols);
    for kernel in &golden.kernels {
        let expected = kernel.neg_elbo.expect("svgp elbo");
        let fitted = Svgp::new(
            kernel_spec(kernel),
            GaussianLikelihood::new(golden.noise_variance).expect("noise"),
        )
        .factor(&golden.x, n, d, &golden.y, &golden.z, golden.m)
        .map_err(|(_, e)| e)
        .expect("svgp factor");
        let obs = fitted
            .predict(&golden.xs, golden.xs_n_rows, d)
            .expect("predict");
        assert_pred_close(&obs, &kernel.mean, &kernel.observation_variance);
        let latent = fitted
            .predict_with(
                &golden.xs,
                golden.xs_n_rows,
                d,
                PredictOptions {
                    variance_kind: VarianceKind::Latent,
                },
            )
            .expect("predict latent");
        assert_pred_close(&latent, &kernel.mean, &kernel.latent_variance);
        assert_close(fitted.neg_elbo().expect("elbo"), expected);
    }
}

#[test]
fn sgpr_forrester_matches_gpytorch() {
    let golden: SparseGpytorchGolden =
        serde_json::from_str(include_str!("../compare/goldens/sgpr_forrester.json"))
            .expect("parse sgpr forrester");
    replay_sgpr(&golden);
}

#[test]
fn sgpr_sphere_matches_gpytorch() {
    let golden: SparseGpytorchGolden =
        serde_json::from_str(include_str!("../compare/goldens/sgpr_sphere.json"))
            .expect("parse sgpr sphere");
    replay_sgpr(&golden);
}

#[test]
fn svgp_forrester_matches_gpytorch() {
    let golden: SparseGpytorchGolden =
        serde_json::from_str(include_str!("../compare/goldens/svgp_forrester.json"))
            .expect("parse svgp forrester");
    replay_svgp(&golden);
}

#[test]
fn svgp_sphere_matches_gpytorch() {
    let golden: SparseGpytorchGolden =
        serde_json::from_str(include_str!("../compare/goldens/svgp_sphere.json"))
            .expect("parse svgp sphere");
    replay_svgp(&golden);
}
