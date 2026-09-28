//! Nelder–Mead golden for Exact GPR (P2B-2).
//!
//! JSON under `compare/goldens/` is produced by `just gen-goldens` via scipy
//! `minimize(method="Nelder-Mead")` on sklearn's log marginal likelihood.
//! This file only reads the committed bytes; `cargo test` must not invoke
//! Python. Not mixed with the L-BFGS `fit_goldens` cases.

use gprx::kernel::{KernelSpec, RbfKernel};
use gprx::{GaussianLikelihood, Gpr, GprError, NelderMead};
use serde::Deserialize;

/// Relative band for NLML after `Gpr<NelderMead>::fit`.
const REL_TOL: f64 = 0.15;
/// `|log actual - log expected|` for `ℓ` and `σn²`.
const THETA_LOG_ABS_TOL: f64 = 2.5;

const GOLDEN: &str = include_str!("../compare/goldens/nelder_mead_rbf_n16.json");

#[derive(Debug, Deserialize)]
struct NelderMeadGolden {
    lengthscale_init: f64,
    noise_variance_init: f64,
    lengthscale: f64,
    noise_variance: f64,
    n_rows: usize,
    n_cols: usize,
    x: Vec<f64>,
    y: Vec<f64>,
    log_marginal_likelihood: f64,
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

fn rbf_gpr(ell: f64, noise: f64) -> Result<Gpr<NelderMead>, GprError> {
    Ok(Gpr::new(
        KernelSpec::from(RbfKernel::new(ell)?),
        GaussianLikelihood::new(noise)?,
    )
    .with_optimizer(NelderMead::new()))
}

#[test]
fn fit_matches_nelder_mead_rbf_n16_golden() {
    let golden: NelderMeadGolden = serde_json::from_str(GOLDEN).expect("json");
    let gpr = rbf_gpr(golden.lengthscale_init, golden.noise_variance_init)
        .expect("valid")
        .fit(&golden.x, golden.n_rows, golden.n_cols, &golden.y)
        .expect("nm");
    let lml = -gpr.neg_log_marginal_likelihood().expect("nlml");
    assert_close_named(
        "log_marginal_likelihood",
        lml,
        golden.log_marginal_likelihood,
        REL_TOL,
    );
    let mut params = [0.0; 2];
    gpr.get_params(&mut params).expect("len 2");
    assert_theta_near("lengthscale", params[0].exp(), golden.lengthscale);
    assert_theta_near("noise", params[1].exp(), golden.noise_variance);
}
