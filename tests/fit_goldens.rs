//! sklearn golden checks for Exact GPR **with** L-BFGS (P1B-6), trust-region
//! recovery (P2B-17), and leave-one-out at sklearn's `θ` (P1B-7).
//!
//! JSON under `compare/goldens/` is produced by `just gen-goldens`. This
//! file only reads the committed bytes; `cargo test` must not invoke Python.
//! Query-predict tolerances are looser than the fixed-hyperparameter 1e-8
//! cases because sklearn uses scipy L-BFGS-B and gprx uses argmin L-BFGS.
//! LOO is compared at the committed sklearn `θ` with a tight band.

use gprx::kernel::{KernelSpec, RbfArdKernel, RbfKernel};
use gprx::transform::StandardizeTarget;
use gprx::{Fixed, GaussianLikelihood, Gpr, GprError, PredictOptions, TrustRegion, VarianceKind};
use serde::Deserialize;

/// Relative band for NLML and predictive mean / variance after `Gpr::fit`.
const REL_TOL: f64 = 0.15;
/// `|log actual - log expected|` for `ℓ` and `σn²`. ARD short axes can
/// differ by about a factor of ten between scipy L-BFGS-B and argmin.
const THETA_LOG_ABS_TOL: f64 = 2.5;
/// Relative band for LOO at sklearn's fitted `θ` (`Gpr<Fixed>::factor`).
/// sklearn's `alpha` jitter is `1e-10`; this is not an optimizer comparison.
const LOO_REL_TOL: f64 = 1e-5;

const FIT_GOLDENS: &[(&str, &str)] = &[
    (
        "forrester_rbf",
        include_str!("../compare/goldens/forrester_rbf.json"),
    ),
    (
        "sphere_rbf_ard",
        include_str!("../compare/goldens/sphere_rbf_ard.json"),
    ),
];

#[derive(Debug, Deserialize)]
struct FitGolden {
    kernel: String,
    lengthscales_init: Vec<f64>,
    noise_variance_init: f64,
    lengthscales: Vec<f64>,
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
    loo_mean: Vec<f64>,
    loo_latent_variance: Vec<f64>,
    loo_observation_variance: Vec<f64>,
}

mod common;
use common::assert_close_named;

fn assert_theta_near(label: &str, actual: f64, expected: f64) {
    let err = (actual.ln() - expected.ln()).abs();
    assert!(
        err <= THETA_LOG_ABS_TOL,
        "{label}: actual={actual}, expected={expected}, abs_log_err={err}, tol={THETA_LOG_ABS_TOL}"
    );
}

fn kernel_from_lengthscales(
    golden: &FitGolden,
    lengthscales: &[f64],
) -> Result<KernelSpec, GprError> {
    match golden.kernel.as_str() {
        "rbf" => {
            if lengthscales.len() != 1 {
                return Err(GprError::InvalidHyperparameter {
                    reason: "isotropic RBF golden needs one lengthscale".to_owned(),
                });
            }
            Ok(KernelSpec::from(RbfKernel::new(lengthscales[0])?))
        }
        "rbf_ard" => Ok(KernelSpec::from(RbfArdKernel::new(lengthscales)?)),
        other => Err(GprError::InvalidHyperparameter {
            reason: format!("unsupported fit golden kernel {other}"),
        }),
    }
}

fn kernel_from_golden(golden: &FitGolden) -> Result<KernelSpec, GprError> {
    kernel_from_lengthscales(golden, &golden.lengthscales_init)
}

fn check_fit_golden(name: &str, golden: &FitGolden) -> Result<(), GprError> {
    let gpr = Gpr::new(
        kernel_from_golden(golden)?,
        GaussianLikelihood::new(golden.noise_variance_init)?,
    )
    .with_target_transform(StandardizeTarget::new())
    .fit(&golden.x, golden.n_rows, golden.n_cols, &golden.y)?;

    let nlml = gpr.neg_log_marginal_likelihood()?;
    assert_close_named(
        &format!("{name} nlml"),
        -nlml,
        golden.log_marginal_likelihood,
        REL_TOL,
    );

    let n_params = golden.lengthscales.len() + 1;
    let mut params = vec![0.0; n_params];
    gpr.get_params(&mut params)?;
    for (i, expected) in golden.lengthscales.iter().enumerate() {
        assert_theta_near(
            &format!("{name} lengthscale[{i}]"),
            params[i].exp(),
            *expected,
        );
    }
    if golden.kernel == "rbf_ard" && golden.lengthscales.len() >= 2 {
        let ell0 = params[0].exp();
        let ell1 = params[1].exp();
        assert!(
            ell0 < ell1,
            "{name}: ARD should keep dim 0 shorter than dim 1: {ell0} vs {ell1}"
        );
    }
    assert_theta_near(
        &format!("{name} noise"),
        params[n_params - 1].exp(),
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
        assert_close_named(
            &format!("{name} mean[{i}]"),
            pred_lat.mean[i],
            golden.mean[i],
            REL_TOL,
        );
        assert_close_named(
            &format!("{name} latent_var[{i}]"),
            pred_lat.variance[i],
            golden.latent_variance[i],
            REL_TOL,
        );
        assert_close_named(
            &format!("{name} obs_var[{i}]"),
            pred_obs.variance[i],
            golden.observation_variance[i],
            REL_TOL,
        );
    }
    Ok(())
}

fn check_loo_at_sklearn_theta(name: &str, golden: &FitGolden) -> Result<(), GprError> {
    let gpr = Gpr::new(
        kernel_from_lengthscales(golden, &golden.lengthscales)?,
        GaussianLikelihood::new(golden.noise_variance)?,
    )
    .with_target_transform(StandardizeTarget::new())
    .with_optimizer(Fixed)
    .factor(&golden.x, golden.n_rows, golden.n_cols, &golden.y)?;

    let loo_obs = gpr.loo_predict()?;
    let loo_lat = gpr.loo_predict_with(PredictOptions {
        variance_kind: VarianceKind::Latent,
    })?;
    assert_eq!(loo_obs.mean.len(), golden.loo_mean.len(), "{name}");
    for i in 0..golden.loo_mean.len() {
        assert_close_named(
            &format!("{name} loo_mean[{i}]"),
            loo_obs.mean[i],
            golden.loo_mean[i],
            LOO_REL_TOL,
        );
        assert_close_named(
            &format!("{name} loo_latent_var[{i}]"),
            loo_lat.variance[i],
            golden.loo_latent_variance[i],
            LOO_REL_TOL,
        );
        assert_close_named(
            &format!("{name} loo_obs_var[{i}]"),
            loo_obs.variance[i],
            golden.loo_observation_variance[i],
            LOO_REL_TOL,
        );
    }
    Ok(())
}

#[test]
fn fit_matches_committed_sklearn_json() {
    for (name, raw) in FIT_GOLDENS {
        let golden: FitGolden = serde_json::from_str(raw).expect("committed JSON parses");
        check_fit_golden(name, &golden).expect(name);
        check_loo_at_sklearn_theta(name, &golden).expect(name);
    }
}

fn check_newton_forrester(golden: &FitGolden) -> Result<(), GprError> {
    // The trust region from the L-BFGS start (`ℓ = 1`) may go to another
    // critical point. This start is in the basin of the committed θ.
    let gpr = Gpr::new(
        KernelSpec::from(RbfKernel::new(0.2)?),
        GaussianLikelihood::new(0.03)?,
    )
    .with_target_transform(StandardizeTarget::new())
    .with_optimizer(TrustRegion::new())
    .fit(&golden.x, golden.n_rows, golden.n_cols, &golden.y)?;

    let nlml = gpr.neg_log_marginal_likelihood()?;
    assert_close_named(
        "newton forrester nlml",
        -nlml,
        golden.log_marginal_likelihood,
        REL_TOL,
    );

    let mut params = [0.0; 2];
    gpr.get_params(&mut params)?;
    assert_theta_near(
        "newton forrester lengthscale",
        params[0].exp(),
        golden.lengthscales[0],
    );
    assert_theta_near(
        "newton forrester noise",
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
    for i in 0..golden.mean.len() {
        assert_close_named(
            &format!("newton forrester mean[{i}]"),
            pred_lat.mean[i],
            golden.mean[i],
            REL_TOL,
        );
        assert_close_named(
            &format!("newton forrester latent_var[{i}]"),
            pred_lat.variance[i],
            golden.latent_variance[i],
            REL_TOL,
        );
        assert_close_named(
            &format!("newton forrester obs_var[{i}]"),
            pred_obs.variance[i],
            golden.observation_variance[i],
            REL_TOL,
        );
    }
    Ok(())
}

#[test]
fn newton_forrester_matches_committed_sklearn_json() {
    let raw = include_str!("../compare/goldens/forrester_rbf.json");
    let golden: FitGolden = serde_json::from_str(raw).expect("committed JSON parses");
    check_newton_forrester(&golden).expect("forrester_rbf");
}
