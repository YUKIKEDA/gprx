//! Random insert/delete sequences vs [`Sgpr<Fixed>::factor`] (P4-8 / P4-10).
//!
//! Each `X` step compares public [`OnlineSgpr`] numerics and point
//! identity to a batch factor at the same `θ` and `Z`. Each inducing step
//! compares the same public numerics at the same `θ` and `X` after `m`
//! changes. This file uses only the public API.

use std::collections::HashSet;

use gprx::kernel::{KernelSpec, MaternKernel, MaternNu, RbfArdKernel, RbfKernel, WhiteKernel};
use gprx::{
    Fixed, FreeInducing, GaussianLikelihood, GprError, InducingId, OnlineSgpr, PointId, Sgpr,
};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

const TOL: f64 = 1e-12;
const OPS: usize = 32;
const N_MAX: usize = 16;
const M_MAX: usize = 16;
const SEEDS: [u64; 3] = [0, 1, 2];
const Y0: [f64; 4] = [0.0, 1.0, 0.5, 0.25];
const X1: [f64; 4] = [0.0, 1.0, 2.0, 3.0];
const Z1: [f64; 2] = [0.5, 2.5];
const X2: [f64; 8] = [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0];
const Z2: [f64; 4] = [0.25, 0.75, 0.25, 0.75];

mod common;
use common::{assert_close, assert_close_named, assert_mean_var_close, assert_slice_close};

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
#[allow(clippy::too_many_arguments)]
fn factor_oracle(
    kernel: KernelSpec,
    likelihood: GaussianLikelihood,
    x: &[f64],
    n: usize,
    d: usize,
    y: &[f64],
    z: &[f64],
    m: usize,
) -> gprx::FittedSgpr<Fixed> {
    Sgpr::new(kernel, likelihood)
        .with_optimizer(Fixed)
        .factor(x, n, d, y, z, m)
        .map_err(|(_, e)| e)
        .expect("factor")
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
#[allow(clippy::too_many_arguments)]
fn assert_matches_factor(
    online: &OnlineSgpr<Fixed>,
    kernel: KernelSpec,
    likelihood: GaussianLikelihood,
    x: &[f64],
    y: &[f64],
    z: &[f64],
    m: usize,
    xs: &[f64],
    d: usize,
    deleted: &HashSet<PointId>,
) {
    let n = online.n();
    assert_eq!(n, y.len());
    assert_eq!(online.m(), m);
    assert_eq!(online.point_ids().len(), n);
    assert_eq!(online.x().len(), n * d);
    assert_eq!(online.y().len(), n);
    for id in deleted {
        assert!(
            !online.point_ids().contains(id),
            "deleted PointId still present"
        );
    }

    let full = factor_oracle(kernel, likelihood, x, n, d, y, z, m);
    assert_eq!(full.n(), n);
    assert_slice_close(online.x(), full.x(), TOL);
    assert_slice_close(online.y(), full.y(), TOL);
    assert_slice_close(online.z(), full.z(), TOL);
    assert_slice_close(online.x(), x, TOL);
    assert_slice_close(online.y(), y, TOL);

    let n_query = xs.len() / d;
    let got = online.predict(xs, n_query, d).expect("online predict");
    let want = full.predict(xs, n_query, d).expect("factor predict");
    assert_mean_var_close(&got.mean, &got.variance, &want.mean, &want.variance, TOL);

    let nlml_online = online.neg_log_marginal_likelihood().expect("online nlml");
    let nlml_full = full.neg_log_marginal_likelihood().expect("factor nlml");
    assert_close(nlml_online, nlml_full, TOL);
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
#[allow(clippy::too_many_arguments)]
fn run_sequence(
    kernel: KernelSpec,
    x0: &[f64],
    y0: &[f64],
    z: &[f64],
    m: usize,
    d: usize,
    xs: &[f64],
    seed: u64,
) {
    let likelihood = GaussianLikelihood::new(0.1).expect("noise");
    let fitted = factor_oracle(kernel.clone(), likelihood, x0, y0.len(), d, y0, z, m);
    let mut online = fitted.into_online();
    let mut x = x0.to_vec();
    let mut y = y0.to_vec();
    let mut deleted = HashSet::new();
    let mut rng = SmallRng::seed_from_u64(seed);

    assert_matches_factor(
        &online,
        kernel.clone(),
        likelihood,
        &x,
        &y,
        z,
        m,
        xs,
        d,
        &deleted,
    );

    for _ in 0..OPS {
        let n = online.n();
        let insert = n == 1 || (n < N_MAX && rng.random::<bool>());
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
        assert_matches_factor(
            &online,
            kernel.clone(),
            likelihood,
            &x,
            &y,
            z,
            m,
            xs,
            d,
            &deleted,
        );
    }
}

fn run_seeds(
    kernel: KernelSpec,
    x0: &[f64],
    y0: &[f64],
    z: &[f64],
    m: usize,
    d: usize,
    xs: &[f64],
) {
    for seed in SEEDS {
        run_sequence(kernel.clone(), x0, y0, z, m, d, xs, seed);
    }
}

#[test]
fn sgpr_ops_rbf_matches_factor() {
    run_seeds(
        KernelSpec::from(RbfKernel::new(1.0).expect("ℓ")),
        &X1,
        &Y0,
        &Z1,
        2,
        1,
        &[0.5],
    );
}

#[test]
fn sgpr_ops_matern_three_halves_matches_factor() {
    run_seeds(
        KernelSpec::from(MaternKernel::new(1.0, MaternNu::ThreeHalves).expect("ℓ")),
        &X1,
        &Y0,
        &Z1,
        2,
        1,
        &[0.5],
    );
}

#[test]
fn sgpr_ops_rbf_ard_2d_matches_factor() {
    run_seeds(
        KernelSpec::from(RbfArdKernel::new(&[1.0, 1.5]).expect("ℓ")),
        &X2,
        &Y0,
        &Z2,
        2,
        2,
        &[0.25, 0.75],
    );
}

#[test]
fn sgpr_ops_rbf_plus_white_matches_factor() {
    run_seeds(
        KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"))
            + KernelSpec::from(WhiteKernel::new(0.05).expect("white")),
        &X1,
        &Y0,
        &Z1,
        2,
        1,
        &[0.5],
    );
}

#[test]
fn into_online_value_grad_hess_match_fitted() {
    let kernel = KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"));
    let likelihood = GaussianLikelihood::new(0.1).expect("noise");
    let mut fitted = factor_oracle(kernel, likelihood, &X1, 4, 1, &Y0, &Z1, 2);
    let mut params = vec![0.0; fitted.num_params()];
    fitted.get_params(&mut params).expect("params");
    let mut grad_fit = vec![0.0; params.len()];
    let value_fit = fitted
        .value_and_gradient_into(&params, &mut grad_fit)
        .expect("fitted grad");
    let mut hess_fit = vec![0.0; params.len() * params.len()];
    fitted
        .hessian_into(&params, &mut hess_fit)
        .expect("fitted hess");

    let mut online = fitted.into_online();
    assert_eq!(online.num_params(), params.len());
    let mut grad_on = vec![0.0; params.len()];
    let value_on = online
        .value_and_gradient_into(&params, &mut grad_on)
        .expect("online grad");
    let mut hess_on = vec![0.0; params.len() * params.len()];
    online
        .hessian_into(&params, &mut hess_on)
        .expect("online hess");
    assert_close(value_on, value_fit, TOL);
    assert_slice_close(&grad_on, &grad_fit, TOL);
    assert_slice_close(&hess_on, &hess_fit, TOL);
}

#[test]
fn rbf_refit_does_not_raise_nlml() {
    let kernel = KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"));
    let likelihood = GaussianLikelihood::new(0.1).expect("noise");
    let fitted = Sgpr::new(kernel, likelihood)
        .fit(&X1, 4, 1, &Y0, &Z1, 2)
        .map_err(|(_, e)| e)
        .expect("fit");
    let mut online = fitted.into_online();
    let before = online.neg_log_marginal_likelihood().expect("nlml");
    online.refit().expect("refit");
    let after = online.neg_log_marginal_likelihood().expect("nlml");
    assert!(
        after <= before,
        "refit raised NLML: before={before}, after={after}"
    );
}

#[test]
fn delete_unknown_point_id_is_invalid() {
    let kernel = KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"));
    let likelihood = GaussianLikelihood::new(0.1).expect("noise");
    let fitted = factor_oracle(kernel, likelihood, &X1, 4, 1, &Y0, &Z1, 2);
    let mut online = fitted.into_online();
    let id = online.point_ids()[1];
    online.delete(id).expect("delete");
    assert!(matches!(online.delete(id), Err(GprError::InvalidPointId)));
}

#[test]
fn delete_last_point_is_empty_input() {
    let kernel = KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"));
    let likelihood = GaussianLikelihood::new(0.1).expect("noise");
    let fitted = factor_oracle(kernel, likelihood, &X1, 4, 1, &Y0, &Z1, 2);
    let mut online = fitted.into_online();
    while online.n() > 1 {
        let id = online.point_ids()[0];
        online.delete(id).expect("delete");
    }
    let last = online.point_ids()[0];
    assert!(matches!(online.delete(last), Err(GprError::EmptyInput)));
    assert_eq!(online.n(), 1);
}

#[test]
fn free_inducing_into_online_drops_z_params() {
    let kernel = KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"));
    let likelihood = GaussianLikelihood::new(0.1).expect("noise");
    let fitted = Sgpr::new(kernel, likelihood)
        .with_inducing(FreeInducing)
        .fit(&X1, 4, 1, &Y0, &[0.2, 0.4], 2)
        .map_err(|(_, e)| e)
        .expect("fit");
    assert_eq!(fitted.num_params(), 4);
    let online = fitted.into_online();
    assert_eq!(online.num_params(), 2);
    assert_eq!(online.m(), 2);
    let fitted_back = online.into_fitted();
    assert_eq!(fitted_back.num_params(), 2);
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
#[allow(clippy::too_many_arguments)]
fn assert_matches_factor_inducing(
    online: &OnlineSgpr<Fixed>,
    kernel: KernelSpec,
    likelihood: GaussianLikelihood,
    x: &[f64],
    y: &[f64],
    z: &[f64],
    xs: &[f64],
    d: usize,
    deleted: &HashSet<InducingId>,
    ctx: &str,
) {
    let n = online.n();
    let m = online.m();
    assert_eq!(n, y.len());
    assert_eq!(online.inducing_ids().len(), m);
    assert_eq!(online.z().len(), m * d);
    assert_eq!(online.x().len(), n * d);
    for id in deleted {
        assert!(
            !online.inducing_ids().contains(id),
            "deleted InducingId still present"
        );
    }

    let full = factor_oracle(kernel, likelihood, x, n, d, y, z, m);
    assert_eq!(full.n(), n);
    assert_eq!(full.m(), m);
    assert_slice_close(online.x(), full.x(), TOL);
    assert_slice_close(online.y(), full.y(), TOL);
    assert_slice_close(online.z(), full.z(), TOL);
    assert_slice_close(online.z(), z, TOL);

    let n_query = xs.len() / d;
    let got = online.predict(xs, n_query, d).expect("online predict");
    let want = full.predict(xs, n_query, d).expect("factor predict");
    assert_eq!(got.mean.len(), want.mean.len());
    for (i, (a, b)) in got.mean.iter().zip(want.mean.iter()).enumerate() {
        assert_close_named(&format!("{ctx} mean[{i}] m={m}"), *a, *b, TOL);
    }
    for (i, (a, b)) in got.variance.iter().zip(want.variance.iter()).enumerate() {
        assert_close_named(&format!("{ctx} var[{i}] m={m}"), *a, *b, TOL);
    }

    let nlml_online = online.neg_log_marginal_likelihood().expect("online nlml");
    let nlml_full = full.neg_log_marginal_likelihood().expect("factor nlml");
    assert_close_named(&format!("{ctx} nlml m={m}"), nlml_online, nlml_full, TOL);
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
#[allow(clippy::too_many_arguments)]
fn run_inducing_sequence(
    kernel: KernelSpec,
    x0: &[f64],
    y0: &[f64],
    z0: &[f64],
    m0: usize,
    d: usize,
    xs: &[f64],
    seed: u64,
) {
    let likelihood = GaussianLikelihood::new(0.1).expect("noise");
    let fitted = factor_oracle(kernel.clone(), likelihood, x0, y0.len(), d, y0, z0, m0);
    let mut online = fitted.into_online();
    let mut z = z0.to_vec();
    let mut deleted = HashSet::new();
    let mut rng = SmallRng::seed_from_u64(seed);

    assert_matches_factor_inducing(
        &online,
        kernel.clone(),
        likelihood,
        x0,
        y0,
        &z,
        xs,
        d,
        &deleted,
        &format!("seed={seed} start"),
    );

    for step in 0..OPS {
        let m = online.m();
        let insert = m == 1 || (m < M_MAX && rng.random::<bool>());
        if insert {
            let mut z_new = vec![0.0; d];
            for value in &mut z_new {
                *value = sample_coord(&mut rng);
            }
            let id = online.insert_inducing(&z_new).expect("insert inducing");
            assert!(!deleted.contains(&id), "insert reused a deleted InducingId");
            assert_eq!(online.inducing_ids().last().copied(), Some(id));
            append_colmajor(&mut z, m, d, &z_new);
        } else {
            let ids = online.inducing_ids();
            let index = rng.random_range(0..ids.len());
            let id = ids[index];
            online.delete_inducing(id).expect("delete inducing");
            deleted.insert(id);
            remove_colmajor(&mut z, m, d, index);
        }
        assert_matches_factor_inducing(
            &online,
            kernel.clone(),
            likelihood,
            x0,
            y0,
            &z,
            xs,
            d,
            &deleted,
            &format!("seed={seed} step={step}"),
        );
    }
}

fn run_inducing_seeds(
    kernel: KernelSpec,
    x0: &[f64],
    y0: &[f64],
    z0: &[f64],
    m0: usize,
    d: usize,
    xs: &[f64],
) {
    for seed in SEEDS {
        run_inducing_sequence(kernel.clone(), x0, y0, z0, m0, d, xs, seed);
    }
}

#[test]
fn inducing_ops_rbf_matches_factor() {
    run_inducing_seeds(
        KernelSpec::from(RbfKernel::new(1.0).expect("ℓ")),
        &X1,
        &Y0,
        &Z1,
        2,
        1,
        &[0.5],
    );
}

#[test]
fn inducing_ops_matern_three_halves_matches_factor() {
    run_inducing_seeds(
        KernelSpec::from(MaternKernel::new(1.0, MaternNu::ThreeHalves).expect("ℓ")),
        &X1,
        &Y0,
        &Z1,
        2,
        1,
        &[0.5],
    );
}

#[test]
fn inducing_ops_rbf_ard_2d_matches_factor() {
    run_inducing_seeds(
        KernelSpec::from(RbfArdKernel::new(&[1.0, 1.5]).expect("ℓ")),
        &X2,
        &Y0,
        &Z2,
        2,
        2,
        &[0.25, 0.75],
    );
}

#[test]
fn inducing_ops_rbf_plus_white_matches_factor() {
    run_inducing_seeds(
        KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"))
            + KernelSpec::from(WhiteKernel::new(0.05).expect("white")),
        &X1,
        &Y0,
        &Z1,
        2,
        1,
        &[0.5],
    );
}

#[test]
fn delete_unknown_inducing_id_is_invalid() {
    let kernel = KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"));
    let likelihood = GaussianLikelihood::new(0.1).expect("noise");
    let fitted = factor_oracle(kernel, likelihood, &X1, 4, 1, &Y0, &Z1, 2);
    let mut online = fitted.into_online();
    let id = online.inducing_ids()[0];
    online.insert_inducing(&[1.5]).expect("insert inducing");
    online.delete_inducing(id).expect("delete inducing");
    assert!(matches!(
        online.delete_inducing(id),
        Err(GprError::InvalidInducingId)
    ));
}

#[test]
fn delete_last_inducing_is_empty_input() {
    let kernel = KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"));
    let likelihood = GaussianLikelihood::new(0.1).expect("noise");
    let fitted = factor_oracle(kernel, likelihood, &X1, 4, 1, &Y0, &Z1, 2);
    let mut online = fitted.into_online();
    while online.m() > 1 {
        let id = online.inducing_ids()[0];
        online.delete_inducing(id).expect("delete inducing");
    }
    let last = online.inducing_ids()[0];
    assert!(matches!(
        online.delete_inducing(last),
        Err(GprError::EmptyInput)
    ));
    assert_eq!(online.m(), 1);
}
