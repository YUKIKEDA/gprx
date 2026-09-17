//! sklearn golden checks for Exact GPR **with** L-BFGS (P1B-6).
//!
//! JSON under `compare/goldens/` is produced by `just gen-goldens`. This
//! file only reads the committed bytes; `cargo test` must not invoke Python.
//! Tolerances are looser than the fixed-hyperparameter 1e-8 cases because
//! sklearn uses scipy L-BFGS-B and gprx uses argmin L-BFGS.

use gprx::kernel::{KernelSpec, RbfKernel};
use gprx::transform::StandardizeTarget;
use gprx::{GaussianLikelihood, Gpr, GprError, PredictOptions, VarianceKind};
use serde::Deserialize;

/// Relative band for NLML and predictive mean / variance.
const REL_TOL: f64 = 0.15;
/// Recovered `ℓ` and `σn²` may differ more than the predictions.
const THETA_REL_TOL: f64 = 0.5;

const FIT_GOLDENS: &[(&str, &str)] = &[(
    "forrester_rbf",
    include_str!("../compare/goldens/forrester_rbf.json"),
)];

#[derive(Debug, Deserialize)]
struct FitGolden {
    lengthscale_init: f64,
    noise_variance_init: f64,
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
}

fn rel_err(actual: f64, expected: f64) -> f64 {
    (actual - expected).abs() / expected.abs().max(1.0)
}

fn theta_rel_err(actual: f64, expected: f64) -> f64 {
    (actual - expected).abs() / expected.abs()
}

fn assert_near(label: &str, actual: f64, expected: f64, tol: f64) {
    let err = rel_err(actual, expected);
    assert!(
        err <= tol,
        "{label}: actual={actual}, expected={expected}, rel_err={err}, tol={tol}"
    );
}

fn assert_theta_near(label: &str, actual: f64, expected: f64) {
    let err = theta_rel_err(actual, expected);
    assert!(
        err <= THETA_REL_TOL,
        "{label}: actual={actual}, expected={expected}, rel_err={err}, tol={THETA_REL_TOL}"
    );
}

fn check_fit_golden(name: &str, golden: &FitGolden) -> Result<(), GprError> {
    let mut gpr = Gpr::new(
        KernelSpec::from(RbfKernel::new(golden.lengthscale_init)?),
        GaussianLikelihood::new(golden.noise_variance_init)?,
    )
    .with_target_transform(StandardizeTarget::new());
    gpr.fit(&golden.x, golden.n_rows, golden.n_cols, &golden.y)?;

    let nlml = gpr.neg_log_marginal_likelihood()?;
    assert_near(
        &format!("{name} nlml"),
        -nlml,
        golden.log_marginal_likelihood,
        REL_TOL,
    );

    let mut params = [0.0; 2];
    gpr.get_params(&mut params)?;
    assert_theta_near(
        &format!("{name} lengthscale"),
        params[0].exp(),
        golden.lengthscale,
    );
    assert_theta_near(
        &format!("{name} noise"),
        params[1].exp(),
        golden.noise_variance,
    );

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
        assert_near(
            &format!("{name} mean[{i}]"),
            pred_lat.mean[i],
            golden.mean[i],
            REL_TOL,
        );
        assert_near(
            &format!("{name} latent_var[{i}]"),
            pred_lat.variance[i],
            golden.latent_variance[i],
            REL_TOL,
        );
        assert_near(
            &format!("{name} obs_var[{i}]"),
            pred_obs.variance[i],
            golden.observation_variance[i],
            REL_TOL,
        );
    }
    Ok(())
}

#[test]
fn forrester_fit_matches_committed_sklearn_json() {
    for (name, raw) in FIT_GOLDENS {
        let golden: FitGolden = serde_json::from_str(raw).expect("committed JSON parses");
        check_fit_golden(name, &golden).expect(name);
    }
}
