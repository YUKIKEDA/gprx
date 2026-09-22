//! GPyTorch OnlineSgpr goldens for insert / delete of `X` and `Z` (P4-13).
//!
//! JSON under `compare/goldens/` is produced by `just gen-sparse-online-goldens`.
//! This file only reads the committed bytes; `cargo test` must not invoke
//! Python.

use gprx::kernel::{KernelSpec, MaternKernel, MaternNu, RbfArdKernel, RbfKernel, WhiteKernel};
use gprx::{Fixed, GaussianLikelihood, OnlineSgpr, PredictOptions, Prediction, Sgpr, VarianceKind};
use serde::Deserialize;

const TOL: f64 = 1e-8;

#[derive(Debug, Deserialize)]
struct OnlineSgprGolden {
    n_rows: usize,
    n_cols: usize,
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    m: usize,
    start_n: usize,
    start_m: usize,
    xs_n_rows: usize,
    xs_n_cols: usize,
    xs: Vec<f64>,
    noise_variance: f64,
    ops: Vec<OnlineSgprOp>,
    kernels: Vec<OnlineSgprKernel>,
}

#[derive(Debug, Deserialize)]
struct OnlineSgprOp {
    kind: String,
    #[serde(default)]
    pop_index: Option<usize>,
    #[serde(default)]
    slot: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct OnlineSgprKernel {
    name: String,
    lengthscales: Vec<f64>,
    white_variance: Option<f64>,
    steps: Vec<OnlineSgprStep>,
}

#[derive(Debug, Deserialize)]
struct OnlineSgprStep {
    n: usize,
    m: usize,
    mean: Vec<f64>,
    observation_variance: Vec<f64>,
    latent_variance: Vec<f64>,
    neg_log_marginal_likelihood: f64,
}

fn assert_close(actual: f64, expected: f64, label: &str) {
    let scale = expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= TOL * scale,
        "{label} actual={actual}, expected={expected}, diff={}",
        (actual - expected).abs()
    );
}

fn assert_pred_close(got: &Prediction, mean: &[f64], variance: &[f64], label: &str) {
    assert_eq!(got.mean.len(), mean.len());
    assert_eq!(got.variance.len(), variance.len());
    for (i, (a, b)) in got.mean.iter().zip(mean.iter()).enumerate() {
        assert_close(*a, *b, &format!("{label} mean[{i}]"));
    }
    for (i, (a, b)) in got.variance.iter().zip(variance.iter()).enumerate() {
        assert_close(*a, *b, &format!("{label} var[{i}]"));
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

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
#[allow(clippy::panic)]
fn kernel_spec(kernel: &OnlineSgprKernel) -> KernelSpec {
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
fn check_step(
    online: &OnlineSgpr<Fixed>,
    step: &OnlineSgprStep,
    xs: &[f64],
    n_query: usize,
    d: usize,
    label: &str,
) {
    assert_eq!(online.n(), step.n, "{label} n");
    assert_eq!(online.m(), step.m, "{label} m");
    let obs = online.predict(xs, n_query, d).expect("predict");
    assert_pred_close(&obs, &step.mean, &step.observation_variance, label);
    let latent = online
        .predict_with(
            xs,
            n_query,
            d,
            PredictOptions {
                variance_kind: VarianceKind::Latent,
            },
        )
        .expect("predict latent");
    assert_pred_close(
        &latent,
        &step.mean,
        &step.latent_variance,
        &format!("{label} latent"),
    );
    assert_close(
        online.neg_log_marginal_likelihood().expect("nlml"),
        step.neg_log_marginal_likelihood,
        &format!("{label} nlml"),
    );
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
#[allow(clippy::panic)]
fn apply_op(online: &mut OnlineSgpr<Fixed>, golden: &OnlineSgprGolden, op: &OnlineSgprOp) {
    let d = golden.n_cols;
    match op.kind.as_str() {
        "insert" => {
            let index = op.pop_index.expect("insert pop_index");
            let x_new = point_at(&golden.x, golden.n_rows, d, index);
            online.insert(&x_new, golden.y[index]).expect("insert");
        }
        "delete" => {
            let slot = op.slot.expect("delete slot");
            let id = online.point_ids()[slot];
            online.delete(id).expect("delete");
        }
        "insert_inducing" => {
            let index = op.pop_index.expect("insert_inducing pop_index");
            let z_new = point_at(&golden.z, golden.m, d, index);
            online.insert_inducing(&z_new).expect("insert_inducing");
        }
        "delete_inducing" => {
            let slot = op.slot.expect("delete_inducing slot");
            let id = online.inducing_ids()[slot];
            online.delete_inducing(id).expect("delete_inducing");
        }
        other => panic!("unknown op {other}"),
    }
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn replay(golden: &OnlineSgprGolden) {
    let n = golden.n_rows;
    let d = golden.n_cols;
    assert_eq!(golden.xs.len(), golden.xs_n_rows * golden.xs_n_cols);
    assert_eq!(golden.z.len(), golden.m * d);
    let x0 = prefix_colmajor(&golden.x, n, d, golden.start_n);
    let y0 = golden.y[..golden.start_n].to_vec();
    let z0 = prefix_colmajor(&golden.z, golden.m, d, golden.start_m);
    for kernel in &golden.kernels {
        assert_eq!(kernel.steps.len(), golden.ops.len() + 1);
        let fitted = Sgpr::new(
            kernel_spec(kernel),
            GaussianLikelihood::new(golden.noise_variance).expect("noise"),
        )
        .with_optimizer(Fixed)
        .factor(&x0, golden.start_n, d, &y0, &z0, golden.start_m)
        .map_err(|(_, e)| e)
        .expect("sgpr factor");
        let mut online = fitted.into_online();
        check_step(
            &online,
            &kernel.steps[0],
            &golden.xs,
            golden.xs_n_rows,
            d,
            &format!("{} init", kernel.name),
        );
        for (i, (op, step)) in golden
            .ops
            .iter()
            .zip(kernel.steps.iter().skip(1))
            .enumerate()
        {
            apply_op(&mut online, golden, op);
            check_step(
                &online,
                step,
                &golden.xs,
                golden.xs_n_rows,
                d,
                &format!("{} step {i} {}", kernel.name, op.kind),
            );
        }
    }
}

#[test]
fn online_sgpr_forrester_matches_gpytorch() {
    let golden: OnlineSgprGolden = serde_json::from_str(include_str!(
        "../compare/goldens/online_sgpr_forrester.json"
    ))
    .expect("parse online sgpr forrester");
    replay(&golden);
}

#[test]
fn online_sgpr_sphere_matches_gpytorch() {
    let golden: OnlineSgprGolden =
        serde_json::from_str(include_str!("../compare/goldens/online_sgpr_sphere.json"))
            .expect("parse online sgpr sphere");
    replay(&golden);
}
