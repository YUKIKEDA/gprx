//! sklearn golden checks for fixed-hyperparameter Exact GPR (P1A-12, P1A-17, #74).
//!
//! JSON under `compare/goldens/` is produced by `just gen-goldens`. This
//! file only reads the committed bytes; `cargo test` must not invoke Python.
//! P1A-12 pins isotropic RBF. P1A-17 adds Sum/Product flatten and extra
//! leaves. Product and mixed trees include `grad_theta`.

use gprx::kernel::{
    ConstantKernel, KernelSpec, MaternKernel, MaternNu, PeriodicKernel, RationalQuadraticKernel,
    RbfKernel,
};
use gprx::{GaussianLikelihood, Gpr, GprError, PredictOptions, VarianceKind};
use serde::Deserialize;

const TOL: f64 = 1e-8;

const RBF_GOLDENS: &[(&str, &str)] = &[
    ("rbf_n2", include_str!("../compare/goldens/rbf_n2.json")),
    ("rbf_n3", include_str!("../compare/goldens/rbf_n3.json")),
];

const COMPOSITE_GOLDENS: &[(&str, &str)] = &[
    (
        "rbf_sum_n2",
        include_str!("../compare/goldens/rbf_sum_n2.json"),
    ),
    (
        "constant_rbf_product_n2",
        include_str!("../compare/goldens/constant_rbf_product_n2.json"),
    ),
    (
        "constant_rbf_plus_rbf_n2",
        include_str!("../compare/goldens/constant_rbf_plus_rbf_n2.json"),
    ),
    (
        "matern32_n2",
        include_str!("../compare/goldens/matern32_n2.json"),
    ),
    ("rq_n2", include_str!("../compare/goldens/rq_n2.json")),
    (
        "periodic_n2",
        include_str!("../compare/goldens/periodic_n2.json"),
    ),
];

#[derive(Debug, Deserialize)]
struct RbfGolden {
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

#[derive(Debug, Deserialize)]
struct CompositeGolden {
    kernel: String,
    leaves: Vec<LeafGolden>,
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
    theta: Vec<f64>,
    #[serde(default)]
    grad_theta: Option<Vec<f64>>,
}

#[derive(Debug, Deserialize)]
struct LeafGolden {
    #[serde(rename = "type")]
    kind: String,
    lengthscale: Option<f64>,
    constant: Option<f64>,
    alpha: Option<f64>,
    period: Option<f64>,
    nu: Option<f64>,
    #[serde(default)]
    leaves: Vec<LeafGolden>,
}

fn assert_close(actual: f64, expected: f64) {
    let scale = expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= TOL * scale,
        "actual={actual}, expected={expected}"
    );
}

fn golden_err(reason: String) -> GprError {
    GprError::InvalidHyperparameter { reason }
}

fn require_field(value: Option<f64>, field: &str) -> Result<f64, GprError> {
    value.ok_or_else(|| golden_err(format!("golden leaf missing {field}")))
}

fn matern_nu(nu: f64) -> Result<MaternNu, GprError> {
    if (nu - 0.5).abs() <= TOL {
        Ok(MaternNu::Half)
    } else if (nu - 1.5).abs() <= TOL {
        Ok(MaternNu::ThreeHalves)
    } else if (nu - 2.5).abs() <= TOL {
        Ok(MaternNu::FiveHalves)
    } else {
        Err(golden_err(format!("unsupported matern nu {nu}")))
    }
}

fn leaf_spec(leaf: &LeafGolden) -> Result<KernelSpec, GprError> {
    match leaf.kind.as_str() {
        "sum" | "product" => {
            let mut terms = Vec::with_capacity(leaf.leaves.len());
            for child in &leaf.leaves {
                terms.push(leaf_spec(child)?);
            }
            let mut iter = terms.into_iter();
            let first = iter
                .next()
                .ok_or_else(|| golden_err(format!("golden {} node has no leaves", leaf.kind)))?;
            if leaf.kind == "sum" {
                Ok(iter.fold(first, |acc, term| acc + term))
            } else {
                Ok(iter.fold(first, |acc, term| acc * term))
            }
        }
        "rbf" => Ok(KernelSpec::from(RbfKernel::new(require_field(
            leaf.lengthscale,
            "lengthscale",
        )?)?)),
        "constant" => Ok(KernelSpec::from(ConstantKernel::new(require_field(
            leaf.constant,
            "constant",
        )?)?)),
        "matern" => Ok(KernelSpec::from(MaternKernel::new(
            require_field(leaf.lengthscale, "lengthscale")?,
            matern_nu(require_field(leaf.nu, "nu")?)?,
        )?)),
        "rq" => Ok(KernelSpec::from(RationalQuadraticKernel::new(
            require_field(leaf.lengthscale, "lengthscale")?,
            require_field(leaf.alpha, "alpha")?,
        )?)),
        "periodic" => Ok(KernelSpec::from(PeriodicKernel::new(
            require_field(leaf.lengthscale, "lengthscale")?,
            require_field(leaf.period, "period")?,
        )?)),
        other => Err(golden_err(format!("unknown golden leaf type {other}"))),
    }
}

fn spec_from_golden(golden: &CompositeGolden) -> Result<KernelSpec, GprError> {
    let mut terms = Vec::with_capacity(golden.leaves.len());
    for leaf in &golden.leaves {
        terms.push(leaf_spec(leaf)?);
    }
    let mut iter = terms.into_iter();
    let first = iter
        .next()
        .ok_or_else(|| golden_err("golden has no leaves".to_owned()))?;
    match golden.kernel.as_str() {
        "leaf" => Ok(first),
        "sum" => Ok(iter.fold(first, |acc, term| acc + term)),
        "product" => Ok(iter.fold(first, |acc, term| acc * term)),
        other => Err(golden_err(format!("unknown golden kernel op {other}"))),
    }
}

struct PredGolden<'a> {
    mean: &'a [f64],
    latent_variance: &'a [f64],
    observation_variance: &'a [f64],
    noise_variance: f64,
    xs: &'a [f64],
    xs_n_rows: usize,
    xs_n_cols: usize,
}

fn check_predictions(name: &str, gpr: &Gpr, golden: PredGolden<'_>) -> Result<(), GprError> {
    let pred_lat = gpr.predict_with(
        golden.xs,
        golden.xs_n_rows,
        golden.xs_n_cols,
        PredictOptions {
            variance_kind: VarianceKind::Latent,
        },
    )?;
    let pred_obs = gpr.predict(golden.xs, golden.xs_n_rows, golden.xs_n_cols)?;
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
    Ok(())
}

fn check_rbf_golden(name: &str, golden: &RbfGolden) -> Result<(), GprError> {
    let mut gpr = Gpr::new(
        KernelSpec::from(RbfKernel::new(golden.lengthscale)?),
        GaussianLikelihood::new(golden.noise_variance)?,
    );
    gpr.fit(&golden.x, golden.n_rows, golden.n_cols, &golden.y)?;
    check_predictions(
        name,
        &gpr,
        PredGolden {
            mean: &golden.mean,
            latent_variance: &golden.latent_variance,
            observation_variance: &golden.observation_variance,
            noise_variance: golden.noise_variance,
            xs: &golden.xs,
            xs_n_rows: golden.xs_n_rows,
            xs_n_cols: golden.xs_n_cols,
        },
    )?;
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

fn check_composite_golden(name: &str, golden: &CompositeGolden) -> Result<(), GprError> {
    let spec = spec_from_golden(golden)?;
    assert_eq!(spec.num_params(), golden.theta.len(), "{name} flatten");
    let mut kernel_params = vec![0.0; spec.num_params()];
    spec.get_params(&mut kernel_params)?;
    for (actual, expected) in kernel_params.iter().zip(&golden.theta) {
        assert_close(*actual, *expected);
    }
    let mut gpr = Gpr::new(spec, GaussianLikelihood::new(golden.noise_variance)?);
    gpr.fit(&golden.x, golden.n_rows, golden.n_cols, &golden.y)?;
    check_predictions(
        name,
        &gpr,
        PredGolden {
            mean: &golden.mean,
            latent_variance: &golden.latent_variance,
            observation_variance: &golden.observation_variance,
            noise_variance: golden.noise_variance,
            xs: &golden.xs,
            xs_n_rows: golden.xs_n_rows,
            xs_n_cols: golden.xs_n_cols,
        },
    )?;
    let nlml = gpr.neg_log_marginal_likelihood()?;
    assert_close(-nlml, golden.log_marginal_likelihood);
    if let Some(grad_theta) = &golden.grad_theta {
        let mut params = vec![0.0; gpr.num_params()];
        gpr.get_params(&mut params)?;
        let mut grad = vec![0.0; params.len()];
        let value = gpr.value_and_gradient_into(&params, &mut grad)?;
        assert_close(value, nlml);
        assert_eq!(grad_theta.len(), kernel_params.len(), "{name} grad flatten");
        for i in 0..grad_theta.len() {
            assert_close(-grad[i], grad_theta[i]);
        }
    }
    Ok(())
}

#[test]
fn rbf_fixed_hypers_match_committed_sklearn_json() {
    for &(name, raw) in RBF_GOLDENS {
        let golden: RbfGolden = serde_json::from_str(raw).expect("committed JSON parses");
        check_rbf_golden(name, &golden).expect(name);
    }
}

#[test]
fn composite_and_extra_leaves_match_committed_sklearn_json() {
    for &(name, raw) in COMPOSITE_GOLDENS {
        let golden: CompositeGolden = serde_json::from_str(raw).expect("committed JSON parses");
        check_composite_golden(name, &golden).expect(name);
    }
}
