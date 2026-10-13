//! Random insert/delete sequences vs [`Sgpr<Fixed>::factor`] (P4-8 / P4-10).
//!
//! Each `X` step compares public [`OnlineSgpr`] numerics and point
//! identity to a batch factor at the same `θ` and `Z`. Each inducing step
//! compares the same public numerics at the same `θ` and `X` after `m`
//! changes (incremental, ADR 0005), for coordinate and supplied-distance
//! models, and the reassembly an insert falls back to. This file uses only
//! the public API.

use std::collections::HashSet;

use gprx::kernel::{
    DistanceKernel, KernelSpec, MaternKernel, MaternNu, RbfArdKernel, RbfKernel, ScalarDistance,
    WhiteKernel,
};
use gprx::{
    Fixed, FreeInducing, GaussianLikelihood, GprError, InducingId, OnlineSgpr, PointId, Sgpr,
};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

const TOL: f64 = 1e-12;
/// Relative tolerance of the inducing sequences: an inducing insert or
/// delete updates the factors incrementally (ADR 0005), in another order
/// of operations than the factor it is compared with.
const INDUCING_TOL: f64 = 1e-10;
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

/// Smallest Euclidean distance a new inducing point of the inducing
/// sequences keeps from every inducing point.
//
// Without it, seed 1 of the RBF sequence (ℓ = 1) drew 2.5036 next to
// 2.4448, 2.3742, and 2.2798: the bordered pivot of `K_mm` was 3.0e-9, and
// the batch factor itself was 6e-10 (mean) and 7e-9 (variance) off the
// exact value, so no update in another order of operations can agree with
// it to 1e-10. With this gap the smallest bordered pivot over the four
// kernels and three seeds is 3.4e-3 (RBF), and the largest relative
// difference from the factor is 4e-15. The near-duplicate case is
// `insert_inducing_next_to_an_inducing_point_reassembles`.
const MIN_GAP: f64 = 0.25;
/// Draws a candidate inducing point this many times before the step deletes
/// instead (the gap leaves no room).
const DRAWS: usize = 64;

/// A new inducing point at least [`MIN_GAP`] from each of the `m` points of
/// `z` (column-major `m × d`), or `None` when [`DRAWS`] candidates were all
/// too close.
fn draw_inducing(rng: &mut SmallRng, z: &[f64], m: usize, d: usize) -> Option<Vec<f64>> {
    for _ in 0..DRAWS {
        let candidate: Vec<f64> = (0..d).map(|_| sample_coord(rng)).collect();
        let clear = (0..m).all(|i| {
            let dist2: f64 = (0..d)
                .map(|feature| (z[feature * m + i] - candidate[feature]).powi(2))
                .sum();
            dist2 >= MIN_GAP * MIN_GAP
        });
        if clear {
            return Some(candidate);
        }
    }
    None
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
fn delete_last_point_is_insufficient_data() {
    let kernel = KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"));
    let likelihood = GaussianLikelihood::new(0.1).expect("noise");
    let fitted = factor_oracle(kernel, likelihood, &X1, 4, 1, &Y0, &Z1, 2);
    let mut online = fitted.into_online();
    while online.n() > 1 {
        let id = online.point_ids()[0];
        online.delete(id).expect("delete");
    }
    let last = online.point_ids()[0];
    // The same variant as `OnlineGpr::delete` of its last point.
    assert!(matches!(
        online.delete(last),
        Err(GprError::InsufficientData { n: 1, min: 2 })
    ));
    assert_eq!(online.n(), 1);
    let mut exact = gprx::Gpr::new(
        KernelSpec::from(RbfKernel::new(1.0).expect("ℓ")),
        GaussianLikelihood::new(0.1).expect("noise"),
    )
    .with_optimizer(gprx::Fixed)
    .factor(&[0.0], 1, 1, &[0.5])
    .map_err(|(_, e)| e)
    .expect("factor")
    .into_online()
    .expect("online");
    let only = exact.point_ids()[0];
    assert!(matches!(
        exact.delete(only),
        Err(GprError::InsufficientData { n: 1, min: 2 })
    ));
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
    tol: f64,
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
        assert_close_named(&format!("{ctx} mean[{i}] m={m}"), *a, *b, tol);
    }
    for (i, (a, b)) in got.variance.iter().zip(want.variance.iter()).enumerate() {
        assert_close_named(&format!("{ctx} var[{i}] m={m}"), *a, *b, tol);
    }

    let nlml_online = online.neg_log_marginal_likelihood().expect("online nlml");
    let nlml_full = full.neg_log_marginal_likelihood().expect("factor nlml");
    assert_close_named(&format!("{ctx} nlml m={m}"), nlml_online, nlml_full, tol);
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
        INDUCING_TOL,
    );

    for step in 0..OPS {
        let m = online.m();
        let wants_insert = m == 1 || (m < M_MAX && rng.random::<bool>());
        let drawn = if wants_insert {
            draw_inducing(&mut rng, &z, m, d)
        } else {
            None
        };
        if let Some(z_new) = drawn {
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
            INDUCING_TOL,
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
fn delete_last_inducing_is_insufficient_data() {
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
        Err(GprError::InsufficientData { n: 1, min: 2 })
    ));
    assert_eq!(online.m(), 1);
}

/// An inducing point equal to one the model holds makes the bordered pivot
/// of `K_mm` zero, so the insert assembles the VFE system again (with the
/// jitter the policy gives the singular `K_mm`). The next insert borders
/// that jittered factor with the same offset.
#[test]
fn insert_inducing_next_to_an_inducing_point_reassembles() {
    let kernel = KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"));
    let likelihood = GaussianLikelihood::new(0.1).expect("noise");
    let fitted = factor_oracle(kernel.clone(), likelihood, &X1, 4, 1, &Y0, &Z1, 2);
    let mut online = fitted.into_online();
    let deleted = HashSet::new();
    // `l = L⁻¹ k(Z, z_new)` is row 0 of `L` exactly, so `ℓ² = 1 − 1 = 0`.
    online.insert_inducing(&[0.5]).expect("insert duplicate");
    let pred = online.predict(&[0.5], 1, 1).expect("predict");
    assert!(pred.mean[0].is_finite() && pred.variance[0].is_finite());
    // The reassembly is the factor's own assembly: equal to its rounding.
    assert_matches_factor_inducing(
        &online,
        kernel.clone(),
        likelihood,
        &X1,
        &Y0,
        &[0.5, 2.5, 0.5],
        &[0.5],
        1,
        &deleted,
        "duplicate",
        TOL,
    );
    // The factor of the four points takes the same `j = 1e-8`. `K_mm + j I`
    // has a condition number near `1e8`, so rounding alone could reach
    // `1e8 · ε ≈ 2e-8`; measured, the bordered insert is within `5e-16`.
    // Bordering with `k(z, z)` instead of `k(z, z) + j` is off by `1e-8`
    // (NLML) and `3e-10` (mean), which `INDUCING_TOL` catches.
    online
        .insert_inducing(&[1.5])
        .expect("insert after duplicate");
    assert_matches_factor_inducing(
        &online,
        kernel,
        likelihood,
        &X1,
        &Y0,
        &[0.5, 2.5, 0.5, 1.5],
        &[0.5],
        1,
        &deleted,
        "after duplicate",
        INDUCING_TOL,
    );
}

/// Training points of the supplied-distance inducing sequences, `0.5`
/// apart on `[0, 6]`.
const DIST_N: usize = 13;
/// Largest inducing set of the supplied-distance sequences.
const DIST_M_MAX: usize = 8;
const DIST_QUERY: [f64; 2] = [0.75, 3.1];

fn dist_x(i: usize) -> f64 {
    0.5 * i as f64
}

fn dist_y() -> Vec<f64> {
    (0..DIST_N).map(|i| (0.9 * dist_x(i)).sin()).collect()
}

/// Column-major `rows × cols` squared distances.
fn sq_block(rows: &[f64], cols: &[f64]) -> Vec<f64> {
    cols.iter()
        .flat_map(|&c| rows.iter().map(move |&r| (r - c).powi(2)))
        .collect()
}

fn dist_points() -> Vec<f64> {
    (0..DIST_N).map(dist_x).collect()
}

fn coords_of(rows: &[usize]) -> Vec<f64> {
    rows.iter().map(|&i| dist_x(i)).collect()
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn assert_distance_matches_factor(
    online: &OnlineSgpr<Fixed, gprx::DoublePrecision, DistanceKernel>,
    kernel: &DistanceKernel,
    image: &ScalarDistance,
    inducing: &[usize],
    ctx: &str,
) {
    let likelihood = GaussianLikelihood::new(0.1).expect("noise");
    let points = dist_points();
    let z = coords_of(inducing);
    let full = Sgpr::new(kernel.clone(), likelihood)
        .with_optimizer(Fixed)
        .factor(
            [image.from_vec(sq_block(&points, &z))],
            DIST_N,
            &dist_y(),
            inducing,
        )
        .map_err(|(_, e)| e)
        .expect("factor");
    let ids = online.point_ids();
    let want_points: Vec<_> = inducing.iter().map(|&i| ids[i]).collect();
    assert_eq!(online.inducing_points().collect::<Vec<_>>(), want_points);
    let q = DIST_QUERY.len();
    let got = online
        .predict([image.from_vec(sq_block(&z, &DIST_QUERY))], q)
        .expect("online predict");
    let want = full
        .predict([image.from_vec(sq_block(&z, &DIST_QUERY))], q)
        .expect("factor predict");
    let m = inducing.len();
    for (i, (a, b)) in got.mean.iter().zip(&want.mean).enumerate() {
        assert_close_named(&format!("{ctx} mean[{i}] m={m}"), *a, *b, INDUCING_TOL);
    }
    for (i, (a, b)) in got.variance.iter().zip(&want.variance).enumerate() {
        assert_close_named(&format!("{ctx} var[{i}] m={m}"), *a, *b, INDUCING_TOL);
    }
    assert_close_named(
        &format!("{ctx} nlml m={m}"),
        online.neg_log_marginal_likelihood().expect("online nlml"),
        full.neg_log_marginal_likelihood().expect("factor nlml"),
        INDUCING_TOL,
    );
}

/// The inducing sequence of [`run_inducing_sequence`] on a model of
/// supplied squared distances: each insert makes a training point that is
/// not inducing an inducing point, each delete drops a random one, and
/// every step matches a factor of the same blocks.
#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn run_distance_inducing_sequence(kernel_of: fn(&ScalarDistance) -> DistanceKernel, seed: u64) {
    let image = ScalarDistance::new();
    let kernel = kernel_of(&image);
    let likelihood = GaussianLikelihood::new(0.1).expect("noise");
    let points = dist_points();
    let mut inducing = vec![0, 6];
    let mut online = Sgpr::new(kernel.clone(), likelihood)
        .with_optimizer(Fixed)
        .factor(
            [image.from_vec(sq_block(&points, &coords_of(&inducing)))],
            DIST_N,
            &dist_y(),
            &inducing,
        )
        .map_err(|(_, e)| e)
        .expect("factor")
        .into_online();
    let mut rng = SmallRng::seed_from_u64(seed);
    let start = format!("seed={seed} start");
    assert_distance_matches_factor(&online, &kernel, &image, &inducing, &start);
    for step in 0..OPS {
        let m = inducing.len();
        let insert = m == 1 || (m < DIST_M_MAX && rng.random::<bool>());
        if insert {
            let free: Vec<usize> = (0..DIST_N).filter(|i| !inducing.contains(i)).collect();
            let row = free[rng.random_range(0..free.len())];
            let point = online.point_ids()[row];
            let column = sq_block(&points, &[dist_x(row)]);
            online
                .insert_inducing(point, [image.from_vec(column)])
                .expect("insert inducing");
            inducing.push(row);
        } else {
            let index = rng.random_range(0..m);
            online
                .delete_inducing(online.inducing_ids()[index])
                .expect("delete inducing");
            inducing.remove(index);
        }
        let ctx = format!("seed={seed} step={step}");
        assert_distance_matches_factor(&online, &kernel, &image, &inducing, &ctx);
    }
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn distance_rbf(image: &ScalarDistance) -> DistanceKernel {
    image.kernel(RbfKernel::new(0.7).expect("ℓ"))
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn distance_matern(image: &ScalarDistance) -> DistanceKernel {
    image.kernel(MaternKernel::new(0.7, MaternNu::ThreeHalves).expect("ℓ"))
}

#[test]
fn distance_inducing_ops_rbf_matches_factor() {
    for seed in SEEDS {
        run_distance_inducing_sequence(distance_rbf, seed);
    }
}

#[test]
fn distance_inducing_ops_matern_three_halves_matches_factor() {
    for seed in SEEDS {
        run_distance_inducing_sequence(distance_matern, seed);
    }
}

/// A tidy column whose values at the inducing points differ from the ones
/// the training blocks hold (by `1e-7` relative) is repaired to the mean of
/// each pair, which moves `k(Z, x)` of the new inducing point's own column:
/// the insert updates that column of `A` too, and the model matches one
/// assembled again from the stored blocks at the same `θ`.
#[test]
fn distance_insert_inducing_with_a_repaired_column_matches_a_reassembly() {
    let image = ScalarDistance::new();
    let kernel = distance_rbf(&image);
    let likelihood = GaussianLikelihood::new(0.1).expect("noise");
    let points = dist_points();
    let inducing = [0, 6, 9];
    let mut online = Sgpr::new(kernel, likelihood)
        .with_optimizer(Fixed)
        .factor(
            [image.from_vec(sq_block(&points, &coords_of(&inducing)))],
            DIST_N,
            &dist_y(),
            &inducing,
        )
        .map_err(|(_, e)| e)
        .expect("factor")
        .into_online();
    let row = 3;
    let mut column = sq_block(&points, &[dist_x(row)]);
    for &i in &inducing {
        column[i] *= 1.0 + 1e-7;
    }
    let point = online.point_ids()[row];
    online
        .insert_inducing(point, [image.from_vec(column).tidy(1e-6).expect("tol")])
        .expect("insert inducing");
    let mut reassembled = online.clone();
    let mut theta = vec![0.0; online.num_params()];
    online.get_params(&mut theta).expect("theta");
    // `set_params` at the stored `θ` changes nothing, so it moves away first.
    let away: Vec<f64> = theta.iter().map(|t| t + 0.1).collect();
    reassembled.set_params(&away).expect("move θ");
    reassembled.set_params(&theta).expect("reassemble");
    let z = coords_of(&[0, 6, 9, row]);
    let q = DIST_QUERY.len();
    let got = online
        .predict([image.from_vec(sq_block(&z, &DIST_QUERY))], q)
        .expect("predict");
    let want = reassembled
        .predict([image.from_vec(sq_block(&z, &DIST_QUERY))], q)
        .expect("predict");
    assert_slice_close(&got.mean, &want.mean, INDUCING_TOL);
    assert_slice_close(&got.variance, &want.variance, INDUCING_TOL);
    assert_close(
        online.neg_log_marginal_likelihood().expect("nlml"),
        reassembled.neg_log_marginal_likelihood().expect("nlml"),
        INDUCING_TOL,
    );
}
