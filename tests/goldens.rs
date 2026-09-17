//! sklearn golden checks for fixed-hyperparameter isotropic RBF (P1A-12).
//!
//! JSON under `compare/goldens/` is produced by `just gen-goldens`. This
//! file only reads the committed bytes; `cargo test` must not invoke Python.

use gprx::kernel::{KernelSpec, RbfKernel};
use gprx::{GaussianLikelihood, Gpr, GprError, PredictOptions, VarianceKind};
use serde::Deserialize;

const TOL: f64 = 1e-8;

const GOLDENS: &[(&str, &str)] = &[
    ("rbf_n2", include_str!("../compare/goldens/rbf_n2.json")),
    ("rbf_n3", include_str!("../compare/goldens/rbf_n3.json")),
];

#[derive(Debug, Deserialize)]
struct Golden {
    lengthscale: f64,
    noise_variance: f64,
    n_rows: usize,
    n_cols: usize,
    x: Vec<f64>,
    y: Vec<f64>,
    xs_n_rows: usize,
    xs_n_cols: usize,
    xs: Vec<f64>,
    mean: Vec<f64>,
    latent_variance: Vec<f64>,
    observation_variance: Vec<f64>,
    log_marginal_likelihood: f64,
    grad_log_lengthscale: f64,
}

fn assert_close(actual: f64, expected: f64) {
    let scale = expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= TOL * scale,
        "actual={actual}, expected={expected}"
    );
}

fn check_golden(name: &str, golden: &Golden) -> Result<(), GprError> {
    let mut gpr = Gpr::new(
        KernelSpec::from(RbfKernel::new(golden.lengthscale)?),
        GaussianLikelihood::new(golden.noise_variance)?,
    );
    gpr.fit(&golden.x, golden.n_rows, golden.n_cols, &golden.y)?;
    let pred_lat = gpr.predict_with(
        &golden.xs,
        golden.xs_n_rows,
        golden.xs_n_cols,
        PredictOptions {
            variance_kind: VarianceKind::Latent,
        },
    )?;
    let pred_obs = gpr.predict(&golden.xs, golden.xs_n_rows, golden.xs_n_cols)?;
    assert_eq!(pred_lat.mean.len(), golden.mean.len(), "{name}");
    for i in 0..golden.mean.len() {
        assert_close(pred_lat.mean[i], golden.mean[i]);
        assert_close(pred_obs.mean[i], golden.mean[i]);
        assert_close(pred_lat.variance[i], golden.latent_variance[i]);
        assert_close(pred_obs.variance[i], golden.observation_variance[i]);
        assert_close(
            pred_obs.variance[i],
            pred_lat.variance[i] + golden.noise_variance,
        );
    }
    let nlml = gpr.neg_log_marginal_likelihood()?;
    assert_close(-nlml, golden.log_marginal_likelihood);
    let mut params = vec![0.0; gpr.num_params()];
    gpr.get_params(&mut params)?;
    let mut grad = vec![0.0; params.len()];
    let value = gpr.value_and_gradient_into(&params, &mut grad)?;
    assert_close(value, nlml);
    assert_close(-grad[0], golden.grad_log_lengthscale);
    Ok(())
}

#[test]
fn rbf_fixed_hypers_match_committed_sklearn_json() {
    for &(name, raw) in GOLDENS {
        let golden: Golden = serde_json::from_str(raw).expect("committed JSON parses");
        check_golden(name, &golden).expect(name);
    }
}
