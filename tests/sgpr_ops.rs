//! Random insert/delete sequences vs [`Sgpr<Fixed>::factor`] (P4-8 / P4-10).
//!
//! Each `X` step compares public [`OnlineSgpr`] numerics and point
//! identity to a batch factor at the same `θ` and `Z`. Each inducing step
//! compares the same public numerics at the same `θ` and `X` after `m`
//! changes (incremental, ADR 0005), for coordinate and supplied-distance
//! models, and the reassembly an inducing change takes instead (an `f32`
//! storage scalar, a jittered `K_mm`, a pivot below the threshold), the
//! reassembly an `X` step takes from a factor with a pivot below the
//! threshold of its storage scalar (coordinate and supplied-distance
//! models, `f64` and `f32`), mixed sequences of inducing and `X` steps
//! with duplicate and near duplicate inducing points, and, at an `f32`
//! storage scalar, mixed sequences (inducing points at a gap, anywhere, and
//! on or next to one), a long run of `X` steps, supplied-distance models,
//! and saved, loaded, and cloned online models, each step held to a fresh
//! factor of the same precision and to the `f64` factor. This file uses
//! only the public API.

use std::collections::HashSet;

use gprx::kernel::KernelScalar;
use gprx::kernel::{
    DistanceKernel, KernelSpec, MaternKernel, MaternNu, RbfArdKernel, RbfKernel, ScalarDistance,
    WhiteKernel,
};
use gprx::persist::{LoadedSgpr, PersistRegistry};
use gprx::{
    DoublePrecision, Fixed, FreeInducing, GaussianLikelihood, GpScalar, GprError, InducingId,
    MixedPrecision, OnlineSgpr, PointId, Sgpr, SinglePrecision,
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
/// jitter the policy gives the singular `K_mm`). The next insert finds the
/// factor jittered and assembles again too, so `j` is chosen as the factor
/// of those inducing points chooses it.
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
        TOL,
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

/// Training points of the reassembly cases: `x = 0.3 i`, `i = 0..12`.
const LINE_N: usize = 12;
/// Queries of the reassembly cases.
const LINE_QUERY: [f64; 4] = [0.4, 1.3, 2.45, 3.0];

fn line_x() -> Vec<f64> {
    (0..LINE_N).map(|i| 0.3 * i as f64).collect()
}

fn line_y() -> Vec<f64> {
    line_x()
        .iter()
        .map(|&x| (1.3 * x).sin() + 0.1 * x)
        .collect()
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn line_rbf() -> KernelSpec {
    KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"))
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn line_likelihood() -> GaussianLikelihood {
    GaussianLikelihood::new(0.1).expect("noise")
}

/// [`OnlineSgpr`] at precision `P` from a factor of the line data at the
/// 1-D inducing points `z`.
#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn line_online<P: GpScalar>(z: &[f64]) -> OnlineSgpr<Fixed, P> {
    Sgpr::new(line_rbf(), line_likelihood())
        .with_optimizer(Fixed)
        .with_precision::<P>()
        .factor(&line_x(), LINE_N, 1, &line_y(), z, z.len())
        .map_err(|(_, e)| e)
        .expect("factor")
        .into_online()
}

/// The largest relative difference ([`common::rel_err`]) of the mean and
/// the variance at [`LINE_QUERY`] and of the NLML between `online` and a
/// factor of the line data at the same precision and the inducing points
/// `z`, after checking that `online` holds `z`.
#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn line_err_vs_factor<P: GpScalar>(online: &OnlineSgpr<Fixed, P>, z: &[f64]) -> f64 {
    assert_slice_close(online.z(), z, TOL);
    let full = line_online::<P>(z);
    let q = LINE_QUERY.len();
    let got = online.predict(&LINE_QUERY, q, 1).expect("online predict");
    let want = full.predict(&LINE_QUERY, q, 1).expect("factor predict");
    let nlml = common::rel_err(
        online.neg_log_marginal_likelihood().expect("online nlml"),
        full.neg_log_marginal_likelihood().expect("factor nlml"),
    );
    got.mean
        .iter()
        .zip(&want.mean)
        .chain(got.variance.iter().zip(&want.variance))
        .map(|(a, b)| common::rel_err(a.to_f64(), b.to_f64()))
        .fold(nlml, f64::max)
}

/// Relative tolerance of an `f32`-storage model against a factor of the
/// same precision: what `tests/distance_persist.rs` allows an online
/// `SinglePrecision` / `MixedPrecision` model factored again on load.
const F32_FACTOR_TOL: f64 = 1e-5;

/// One inducing step of [`f32_storage_inducing_ops_match_factor`]: an
/// insert at these coordinates, or a delete at this index.
enum Step {
    Insert(f64),
    Delete(usize),
}

const F32_STEPS: [Step; 18] = [
    Step::Insert(1.5),
    Step::Insert(3.2),
    Step::Insert(0.1),
    Step::Insert(1.0),
    Step::Insert(2.0),
    Step::Delete(0),
    Step::Insert(2.9),
    Step::Insert(0.6),
    Step::Delete(3),
    Step::Insert(1.8),
    Step::Insert(2.4),
    Step::Delete(6),
    Step::Insert(0.3),
    Step::Insert(3.0),
    Step::Insert(1.25),
    Step::Delete(1),
    Step::Insert(2.15),
    Step::Delete(0),
];

/// Runs [`F32_STEPS`] from the inducing points `[0.5, 2.5]`, checking
/// [`line_err_vs_factor`] against [`F32_FACTOR_TOL`] after each step.
#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn run_f32_inducing_steps<P: GpScalar>(label: &str) {
    let mut z = vec![0.5, 2.5];
    let mut online = line_online::<P>(&z);
    for (step, op) in F32_STEPS.iter().enumerate() {
        match *op {
            Step::Insert(at) => {
                online.insert_inducing(&[at]).expect("insert inducing");
                z.push(at);
            }
            Step::Delete(index) => {
                online
                    .delete_inducing(online.inducing_ids()[index])
                    .expect("delete inducing");
                z.remove(index);
            }
        }
        let err = line_err_vs_factor(&online, &z);
        assert!(
            err <= F32_FACTOR_TOL,
            "{label} step={step}: rel_err={err} > {F32_FACTOR_TOL}"
        );
    }
}

/// An `f32` storage scalar assembles the VFE system again on every
/// inducing change, so a sequence of inserts and deletes keeps the
/// accuracy of a fresh factor of the same precision. Updated in `f32`, the
/// same steps were up to `5.8e-5` (single) and `4.8e-5` (mixed) off.
#[test]
fn f32_storage_inducing_ops_match_factor() {
    run_f32_inducing_steps::<SinglePrecision>("single");
    run_f32_inducing_steps::<MixedPrecision>("mixed");
}

/// A duplicate inducing point makes the factor take a jitter offset
/// `j = 1e-8`. Deleting the duplicate again assembles the system, which
/// finds `j = 0` as the factor of the remaining points does, instead of
/// keeping the stale `j` (which was `2e-8` off). A delete of another point
/// while `j > 0` takes the offset the factor of the points that stay takes
/// as well.
#[test]
fn delete_inducing_with_a_jittered_factor_matches_factor() {
    let mut online = line_online::<DoublePrecision>(&[0.5, 2.5]);
    let duplicate = online.insert_inducing(&[0.5]).expect("insert duplicate");
    let err = line_err_vs_factor(&online, &[0.5, 2.5, 0.5]);
    assert!(err <= TOL, "duplicate: rel_err={err}");
    let mut other = online.clone();
    online.delete_inducing(duplicate).expect("delete duplicate");
    let err_dup = line_err_vs_factor(&online, &[0.5, 2.5]);
    assert!(err_dup <= TOL, "delete duplicate: rel_err={err_dup}");
    other
        .delete_inducing(other.inducing_ids()[1])
        .expect("delete 2.5");
    let err_other = line_err_vs_factor(&other, &[0.5, 0.5]);
    assert!(err_other <= TOL, "delete 2.5: rel_err={err_other}");
}

/// A near duplicate leaves `K_mm` with a pivot `ℓ² ≈ 1e-10` that the factor
/// takes without jitter; an exact duplicate of another point then leaves
/// `ℓ²` a rounding residue. Both pivots are below the relative threshold
/// (the first in the factor the insert starts from, the second in the
/// border), so the insert assembles the system again. Dividing by that
/// residue was `0.56` off.
#[test]
fn insert_inducing_at_a_rounding_pivot_matches_factor() {
    let z = [0.5, 0.5 + 1e-5, 2.5];
    let mut online = line_online::<DoublePrecision>(&z);
    online.insert_inducing(&[2.5]).expect("insert duplicate");
    let err = line_err_vs_factor(&online, &[0.5, 0.5 + 1e-5, 2.5, 2.5]);
    assert!(err <= TOL, "rel_err={err}");
}

/// A new inducing point `1e-4` from one the model holds has a positive
/// `ℓ² ≈ 1e-8` that the factor takes without jitter, below the relative
/// threshold of `1e-5`: the insert assembles the system again instead of
/// dividing by a pivot with a relative rounding error near `ε / ℓ²` (the
/// update was `8.7e-11` off).
#[test]
fn insert_inducing_below_the_pivot_threshold_matches_factor() {
    let mut online = line_online::<DoublePrecision>(&[0.5, 2.5]);
    online.insert_inducing(&[2.5 + 1e-4]).expect("insert");
    let err = line_err_vs_factor(&online, &[0.5, 2.5, 2.5 + 1e-4]);
    assert!(err <= TOL, "rel_err={err}");
}

/// Deletes down to one inducing point, each by the incremental update.
#[test]
fn delete_inducing_down_to_one_matches_factor() {
    let mut z = vec![0.5, 1.5, 2.5];
    let mut online = line_online::<DoublePrecision>(&z);
    while z.len() > 1 {
        online
            .delete_inducing(online.inducing_ids()[0])
            .expect("delete inducing");
        z.remove(0);
        let err = line_err_vs_factor(&online, &z);
        assert!(err <= INDUCING_TOL, "m={}: rel_err={err}", z.len());
    }
    assert_eq!(online.m(), 1);
}

/// A delete at the first, a middle, and the last index of four inducing
/// points, each from the same model.
#[test]
fn delete_inducing_at_first_middle_last_matches_factor() {
    let z = [0.25, 1.25, 2.25, 3.25];
    let start = line_online::<DoublePrecision>(&z);
    for index in [0, 2, 3] {
        let mut online = start.clone();
        online
            .delete_inducing(online.inducing_ids()[index])
            .expect("delete inducing");
        let mut kept = z.to_vec();
        kept.remove(index);
        let err = line_err_vs_factor(&online, &kept);
        assert!(err <= INDUCING_TOL, "index={index}: rel_err={err}");
    }
}

/// [`delete_inducing_at_first_middle_last_matches_factor`] on supplied
/// distances.
#[test]
fn distance_delete_inducing_at_first_middle_last_matches_factor() {
    let image = ScalarDistance::new();
    let kernel = distance_rbf(&image);
    let likelihood = GaussianLikelihood::new(0.1).expect("noise");
    let points = dist_points();
    let inducing = [0, 4, 8, 12];
    let start = Sgpr::new(kernel.clone(), likelihood)
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
    for index in [0, 2, 3] {
        let mut online = start.clone();
        online
            .delete_inducing(online.inducing_ids()[index])
            .expect("delete inducing");
        let mut kept = inducing.to_vec();
        kept.remove(index);
        let ctx = format!("index={index}");
        assert_distance_matches_factor(&online, &kernel, &image, &kept, &ctx);
    }
}

/// Steps of each mixed sequence.
const MIXED_OPS: usize = 60;
/// Seeds of the mixed sequences.
const MIXED_SEEDS: std::ops::Range<u64> = 0..12;
/// Training points and inducing points a mixed sequence starts from.
const MIXED_N0: usize = 30;
const MIXED_M0: usize = 4;
/// Bounds the mixed sequences keep `n` and `m` within.
const MIXED_N_RANGE: (usize, usize) = (20, 40);
const MIXED_M_MAX: usize = 12;

/// One step of a mixed sequence, named in a failure message.
#[derive(Clone, Copy, Debug)]
enum MixedOp {
    /// Insert of a copy of the inducing point at this index.
    InducingDuplicate(usize),
    /// Insert of a point between `1e-5` and `1e-2` per coordinate from the
    /// inducing point at this index.
    InducingNear(usize),
    InducingRandom,
    InducingDelete(usize),
    Insert,
    Delete(usize),
}

/// A new inducing point of `op` next to the `m` inducing points `z`
/// (column-major `m × d`).
fn mixed_inducing(rng: &mut SmallRng, op: MixedOp, z: &[f64], m: usize, d: usize) -> Vec<f64> {
    match op {
        MixedOp::InducingDuplicate(i) => (0..d).map(|f| z[f * m + i]).collect(),
        MixedOp::InducingNear(i) => (0..d)
            .map(|f| {
                let size = 10f64.powf(-5.0 + 3.0 * rng.random::<f64>());
                let sign = if rng.random::<bool>() { 1.0 } else { -1.0 };
                z[f * m + i] + sign * size
            })
            .collect(),
        _ => (0..d).map(|_| sample_coord(rng)).collect(),
    }
}

/// The step a mixed sequence takes at `n` training and `m` inducing points:
/// an inducing insert (a duplicate, a near duplicate, or a random point), an
/// inducing delete (first, middle, last, or random), or an `X` insert or
/// delete.
fn mixed_op(rng: &mut SmallRng, n: usize, m: usize) -> MixedOp {
    loop {
        match rng.random_range(0..4) {
            0 if m < MIXED_M_MAX => {
                let i = rng.random_range(0..m);
                return match rng.random_range(0..3) {
                    0 => MixedOp::InducingDuplicate(i),
                    1 => MixedOp::InducingNear(i),
                    _ => MixedOp::InducingRandom,
                };
            }
            1 if m > 1 => {
                let index = match rng.random_range(0..4) {
                    0 => 0,
                    1 => m / 2,
                    2 => m - 1,
                    _ => rng.random_range(0..m),
                };
                return MixedOp::InducingDelete(index);
            }
            2 if n < MIXED_N_RANGE.1 => return MixedOp::Insert,
            3 if n > MIXED_N_RANGE.0 => return MixedOp::Delete(rng.random_range(0..n)),
            _ => {}
        }
    }
}

/// The largest relative difference of the mean and the variance at `xs`
/// and of the NLML between `online` and a factor of the same `X`, `y`, and
/// `Z`.
#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
#[allow(clippy::too_many_arguments)]
fn mixed_err_vs_factor<P: GpScalar>(
    online: &OnlineSgpr<Fixed, P>,
    kernel: KernelSpec,
    likelihood: GaussianLikelihood,
    x: &[f64],
    y: &[f64],
    z: &[f64],
    xs: &[f64],
    d: usize,
) -> f64 {
    let (n, m) = (y.len(), z.len() / d);
    assert_eq!((online.n(), online.m()), (n, m));
    assert_slice_close(online.x(), x, TOL);
    assert_slice_close(online.z(), z, TOL);
    let full = Sgpr::new(kernel, likelihood)
        .with_optimizer(Fixed)
        .with_precision::<P>()
        .factor(x, n, d, y, z, m)
        .map_err(|(_, e)| e)
        .expect("factor");
    let q = xs.len() / d;
    let got = online.predict(xs, q, d).expect("online predict");
    let want = full.predict(xs, q, d).expect("factor predict");
    let nlml = common::rel_err(
        online.neg_log_marginal_likelihood().expect("online nlml"),
        full.neg_log_marginal_likelihood().expect("factor nlml"),
    );
    got.mean
        .iter()
        .zip(&want.mean)
        .chain(got.variance.iter().zip(&want.variance))
        .map(|(a, b)| common::rel_err(a.to_f64(), b.to_f64()))
        .fold(nlml, f64::max)
}

/// Runs one mixed sequence of [`MIXED_OPS`] steps from `seed` with the
/// `kernel` of `d` features at precision `P` and returns the largest
/// difference from a factor of the same precision after any step, with
/// the step it was found at.
#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn run_mixed_sequence<P: GpScalar>(kernel: &KernelSpec, d: usize, seed: u64) -> (f64, String) {
    let likelihood = GaussianLikelihood::new(0.1).expect("noise");
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut x: Vec<f64> = (0..MIXED_N0 * d).map(|_| sample_coord(&mut rng)).collect();
    let mut y: Vec<f64> = (0..MIXED_N0).map(|_| sample_coord(&mut rng)).collect();
    let mut z: Vec<f64> = (0..MIXED_M0 * d).map(|_| sample_coord(&mut rng)).collect();
    let xs: Vec<f64> = (0..4 * d).map(|i| 0.3 + 0.9 * (i % 4) as f64).collect();
    let mut online = Sgpr::new(kernel.clone(), likelihood)
        .with_optimizer(Fixed)
        .with_precision::<P>()
        .factor(&x, MIXED_N0, d, &y, &z, MIXED_M0)
        .map_err(|(_, e)| e)
        .expect("factor")
        .into_online();
    let mut worst = (0.0, String::new());
    for step in 0..MIXED_OPS {
        let (n, m) = (online.n(), online.m());
        let op = mixed_op(&mut rng, n, m);
        match op {
            MixedOp::InducingDuplicate(_) | MixedOp::InducingNear(_) | MixedOp::InducingRandom => {
                let z_new = mixed_inducing(&mut rng, op, &z, m, d);
                online.insert_inducing(&z_new).expect("insert inducing");
                append_colmajor(&mut z, m, d, &z_new);
            }
            MixedOp::InducingDelete(index) => {
                online
                    .delete_inducing(online.inducing_ids()[index])
                    .expect("delete inducing");
                remove_colmajor(&mut z, m, d, index);
            }
            MixedOp::Insert => {
                let x_new: Vec<f64> = (0..d).map(|_| sample_coord(&mut rng)).collect();
                let y_new = sample_coord(&mut rng);
                online.insert(&x_new, y_new).expect("insert");
                append_colmajor(&mut x, n, d, &x_new);
                y.push(y_new);
            }
            MixedOp::Delete(index) => {
                online.delete(online.point_ids()[index]).expect("delete");
                remove_colmajor(&mut x, n, d, index);
                y.remove(index);
            }
        }
        let err = mixed_err_vs_factor(&online, kernel.clone(), likelihood, &x, &y, &z, &xs, d);
        assert!(err.is_finite(), "seed={seed} step={step} {op:?}: {err}");
        if err > worst.0 {
            worst = (err, format!("seed={seed} step={step} m={m} {op:?}"));
        }
    }
    worst
}

/// Runs [`run_mixed_sequence`] at precision `P` over [`MIXED_SEEDS`] and
/// checks every step against `tol`.
fn run_mixed_seeds<P: GpScalar>(kernel: &KernelSpec, d: usize, tol: f64, label: &str) {
    let (err, at) = MIXED_SEEDS
        .map(|seed| run_mixed_sequence::<P>(kernel, d, seed))
        .fold((0.0, String::new()), |a, b| if b.0 > a.0 { b } else { a });
    assert!(err <= tol, "{label}: rel_err={err:e} at {at}");
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn mixed_rbf() -> KernelSpec {
    KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"))
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn mixed_rbf_ard() -> KernelSpec {
    KernelSpec::from(RbfArdKernel::new(&[1.0, 1.5]).expect("ℓ"))
}

/// Inducing inserts on, next to (`1e-5` to `1e-2` per coordinate), and
/// away from the inducing points, and inducing deletes at the first, the
/// middle, the last, and a random index, mixed with `X` inserts and
/// deletes, on 1-D RBF data: each step matches a factor of the same `X`,
/// `y`, and `Z`. An inducing change from a factor of `K_mm` with a pivot
/// `L_ii² ≤ 1e-5 (K_ii + j)` assembles the system again, and an `X` step
/// from one with a pivot `L_ii² ≤ 8 ε (K_ii + j)` (the rounding residue of
/// an exact duplicate; between the two an `X` step updates by rank-1, held
/// to the exact system in
/// [`x_steps_next_to_a_near_duplicate_stay_near_the_exact_system`]).
/// Before an inducing change did, a delete there was `0.20` off (seed 2,
/// step 46); before an `X` step did, an `X` step after an exact duplicate
/// was `2.9e-4` off (seed 5, step 8), its rank-1 update of `B` carrying the
/// rounding of a residue pivot. The largest difference is now `1.0e-11`
/// (`1e-13` while an `X` step assembled below `1e-5`).
#[test]
fn mixed_ops_rbf_with_near_duplicates_match_factor() {
    run_mixed_seeds::<DoublePrecision>(&mixed_rbf(), 1, INDUCING_TOL, "rbf");
}

/// [`mixed_ops_rbf_with_near_duplicates_match_factor`] on 2-D ARD RBF
/// data. Before an inducing change from a factor with a pivot below the
/// threshold assembled the system again, a delete of the first inducing
/// point was `0.28` off (seed 7, step 48). The largest difference is
/// `1.6e-12`.
#[test]
fn mixed_ops_rbf_ard_2d_with_near_duplicates_match_factor() {
    run_mixed_seeds::<DoublePrecision>(&mixed_rbf_ard(), 2, INDUCING_TOL, "ard");
}

/// The mixed sequences at an `f32` storage scalar, against a factor of the
/// same precision at [`F32_FACTOR_TOL`]. Every inducing change assembles
/// again; an `X` step does when a pivot of the factor of `K_mm` is at most
/// `8 ε` times its diagonal (`1e-3` until R8-4). Updated by rank-1 in
/// `f32` arithmetic there, an `X` step after an exact duplicate was `2.3e10`
/// off on 1-D RBF data (seed 5, step 8, both precisions) and `30` on 2-D
/// ARD data (seed 5, step 48). The largest difference is now `5.6e-17`
/// (`4.0e-7` while an `X` step assembled below `1e-3`, `1.0e-6` before an
/// `X` step computed in `f64`,
/// [`f32_storage_mixed_ops_with_separated_inducing_points_match_factor`]).
#[test]
fn f32_storage_mixed_ops_with_near_duplicates_match_factor() {
    run_mixed_seeds::<SinglePrecision>(&mixed_rbf(), 1, F32_FACTOR_TOL, "single rbf");
    run_mixed_seeds::<MixedPrecision>(&mixed_rbf(), 1, F32_FACTOR_TOL, "mixed rbf");
    run_mixed_seeds::<SinglePrecision>(&mixed_rbf_ard(), 2, F32_FACTOR_TOL, "single ard");
    run_mixed_seeds::<MixedPrecision>(&mixed_rbf_ard(), 2, F32_FACTOR_TOL, "mixed ard");
}

/// Five `X` inserts, each followed by a delete, from the line data at the
/// inducing points `z`, at precision `P`: the largest difference
/// ([`mixed_err_vs_factor`]) from a factor of the same precision after any
/// step.
#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn line_x_steps_err<P: GpScalar>(z: &[f64]) -> f64 {
    let mut online = line_online::<P>(z);
    let (mut x, mut y) = (line_x(), line_y());
    let mut worst: f64 = 0.0;
    for (i, &x_new) in [0.55, 2.45, 1.1, 3.4, 0.05].iter().enumerate() {
        let y_new = (2.1 * x_new).cos();
        online.insert(&[x_new], y_new).expect("insert");
        x.push(x_new);
        y.push(y_new);
        let err = mixed_err_vs_factor(
            &online,
            line_rbf(),
            line_likelihood(),
            &x,
            &y,
            z,
            &LINE_QUERY,
            1,
        );
        worst = worst.max(err);
        let index = (3 * i + 1) % x.len();
        online.delete(online.point_ids()[index]).expect("delete");
        x.remove(index);
        y.remove(index);
        let err = mixed_err_vs_factor(
            &online,
            line_rbf(),
            line_likelihood(),
            &x,
            &y,
            z,
            &LINE_QUERY,
            1,
        );
        worst = worst.max(err);
    }
    worst
}

/// Two inducing pairs `0.014` and `0.0098` apart leave the factor of
/// `K_mm` a pivot of `1.4e-5` times its diagonal: above the threshold of an
/// `X` step (`8 ε`), so the steps of every precision update by rank-1.
/// Updated by rank-1 in `f32` arithmetic, the steps were `1.6e-5` (single)
/// and `9.7e-6` (mixed) off; they compute in `f64` now (R8-3), and the
/// threshold of an `f32` model was `1e-3` until R8-4.
#[test]
fn x_steps_next_to_a_near_duplicate_match_factor() {
    let gap = 0.014;
    let z = [0.5, 0.5 + gap, 1.7, 2.5, 2.5 + 0.7 * gap];
    let err = line_x_steps_err::<DoublePrecision>(&z);
    assert!(err <= INDUCING_TOL, "double: rel_err={err}");
    let err = line_x_steps_err::<SinglePrecision>(&z);
    assert!(err <= F32_FACTOR_TOL, "single: rel_err={err}");
    let err = line_x_steps_err::<MixedPrecision>(&z);
    assert!(err <= F32_FACTOR_TOL, "mixed: rel_err={err}");
}

/// `X` inserts and deletes on supplied distances from inducing points with
/// a pair `gap` apart (`0` an exact duplicate), at precision `P`: after
/// each step the model matches a factor of the same blocks and precision
/// to `tol`. A distance model inserts and deletes points by the same
/// update as a coordinate model, and the reassembly writes its training
/// blocks.
#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn run_distance_x_steps<P: GpScalar>(gap: f64, tol: f64) {
    let image = ScalarDistance::new();
    let kernel = distance_rbf(&image);
    let likelihood = GaussianLikelihood::new(0.1).expect("noise");
    let mut points = dist_points();
    points.push(dist_x(4) + gap);
    let mut y: Vec<f64> = points.iter().map(|&p| (0.9 * p).sin()).collect();
    let mut inducing = vec![0, 4, 9, DIST_N];
    let factor_of = |points: &[f64], y: &[f64], inducing: &[usize]| {
        let z: Vec<f64> = inducing.iter().map(|&i| points[i]).collect();
        Sgpr::new(kernel.clone(), likelihood)
            .with_optimizer(Fixed)
            .with_precision::<P>()
            .factor(
                [image.from_vec(sq_block(points, &z))],
                points.len(),
                y,
                inducing,
            )
            .map_err(|(_, e)| e)
            .expect("factor")
    };
    let mut online = factor_of(&points, &y, &inducing).into_online();
    let q = DIST_QUERY.len();
    for (step, &at) in [2.2, 4.9, 1.05, 2.0 + gap].iter().enumerate() {
        let z: Vec<f64> = inducing.iter().map(|&i| points[i]).collect();
        let y_new = (0.9 * at).sin();
        online
            .insert([image.from_vec(sq_block(&z, &[at]))], y_new)
            .expect("insert");
        points.push(at);
        y.push(y_new);
        let row = [6, 1, 10, 2][step];
        online.delete(online.point_ids()[row]).expect("delete");
        points.remove(row);
        y.remove(row);
        for i in &mut inducing {
            if *i > row {
                *i -= 1;
            }
        }
        let full = factor_of(&points, &y, &inducing);
        let query = || [image.from_vec(sq_block(&z, &DIST_QUERY))];
        let got = online.predict(query(), q).expect("online predict");
        let want = full.predict(query(), q).expect("factor predict");
        let ctx = format!("gap={gap} step={step}");
        for (a, b) in got.mean.iter().zip(&want.mean) {
            assert_close_named(&format!("{ctx} mean"), a.to_f64(), b.to_f64(), tol);
        }
        for (a, b) in got.variance.iter().zip(&want.variance) {
            assert_close_named(&format!("{ctx} var"), a.to_f64(), b.to_f64(), tol);
        }
        assert_close_named(
            &format!("{ctx} nlml"),
            online.neg_log_marginal_likelihood().expect("online nlml"),
            full.neg_log_marginal_likelihood().expect("factor nlml"),
            tol,
        );
    }
}

/// [`run_distance_x_steps`] at an exact duplicate, near duplicates below
/// both thresholds (`1e-8` and `1e-4` apart), and a pair between them
/// (`1e-2` apart, a pivot of `2e-4` times the diagonal), in `f64` and
/// `f32` storage. Updated by rank-1 from a factor with a pivot below the
/// threshold, the steps were `9.8e-10` off in `f64` at `1e-8` and, in
/// `f32`, `0.69` at `1e-8` and `5.7e-5` at `1e-4`.
#[test]
fn distance_x_steps_next_to_a_duplicate_match_factor() {
    for gap in [0.0, 1e-8, 1e-4, 1e-2] {
        run_distance_x_steps::<DoublePrecision>(gap, INDUCING_TOL);
        run_distance_x_steps::<SinglePrecision>(gap, F32_FACTOR_TOL);
        run_distance_x_steps::<MixedPrecision>(gap, F32_FACTOR_TOL);
    }
}

/// Factor and floor of the bound an `f32` step is held to against the
/// `f64` factor of the same points: within [`TRUTH_FACTOR`] times the
/// error a fresh factor of the same precision has there, plus
/// [`TRUTH_FLOOR`] (eight times the machine epsilon of `f32`), so a step
/// is never much farther from the `f64` system than the rounding of that
/// system is.
const TRUTH_FACTOR: f64 = 2.0;
const TRUTH_FLOOR: f64 = 8.0 * f32::EPSILON as f64;

/// The mean and the variance at `xs` and the NLML of a factor of `X`, `y`,
/// and `Z` at precision `P`.
#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
#[allow(clippy::too_many_arguments)]
fn factor_values<P: GpScalar>(
    kernel: &KernelSpec,
    likelihood: GaussianLikelihood,
    x: &[f64],
    y: &[f64],
    z: &[f64],
    xs: &[f64],
    d: usize,
) -> Vec<f64> {
    let full = Sgpr::new(kernel.clone(), likelihood)
        .with_optimizer(Fixed)
        .with_precision::<P>()
        .factor(x, y.len(), d, y, z, z.len() / d)
        .map_err(|(_, e)| e)
        .expect("factor");
    let pred = full.predict(xs, xs.len() / d, d).expect("factor predict");
    let nlml = full.neg_log_marginal_likelihood().expect("factor nlml");
    prediction_values(&pred.mean, &pred.variance, nlml)
}

/// [`factor_values`] of `online`.
#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn online_values<P: GpScalar>(online: &OnlineSgpr<Fixed, P>, xs: &[f64], d: usize) -> Vec<f64> {
    let pred = online.predict(xs, xs.len() / d, d).expect("online predict");
    let nlml = online.neg_log_marginal_likelihood().expect("online nlml");
    prediction_values(&pred.mean, &pred.variance, nlml)
}

fn prediction_values<R: KernelScalar>(mean: &[R], variance: &[R], nlml: f64) -> Vec<f64> {
    mean.iter()
        .chain(variance)
        .map(|value| value.to_f64())
        .chain([nlml])
        .collect()
}

/// The largest [`common::rel_err`] of `got` against `want`.
fn max_rel_err(got: &[f64], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    got.iter()
        .zip(want)
        .map(|(a, b)| common::rel_err(*a, *b))
        .fold(0.0, f64::max)
}

/// Checks one `f32` step: `online` (the values of the online model) is
/// within [`F32_FACTOR_TOL`] of `fresh` (a factor of the same points at
/// the same precision), and within [`TRUTH_FACTOR`] times the error of
/// `fresh` plus [`TRUTH_FLOOR`] of `truth` (the `f64` factor). Returns
/// the difference from `fresh`.
fn check_f32_step(online: &[f64], fresh: &[f64], truth: &[f64], ctx: &str) -> f64 {
    let online_fresh = max_rel_err(online, fresh);
    let online_truth = max_rel_err(online, truth);
    let fresh_truth = max_rel_err(fresh, truth);
    assert!(
        online_fresh <= F32_FACTOR_TOL,
        "{ctx}: {online_fresh:e} from a fresh factor"
    );
    assert!(
        online_truth <= TRUTH_FACTOR * fresh_truth + TRUTH_FLOOR,
        "{ctx}: {online_truth:e} from the f64 factor, a fresh factor {fresh_truth:e}"
    );
    online_fresh
}

/// [`check_f32_step`] of `online` against factors of `X`, `y`, and `Z`,
/// after checking that `online` holds them.
#[allow(clippy::too_many_arguments)]
fn check_f32_online<P: GpScalar>(
    online: &OnlineSgpr<Fixed, P>,
    kernel: &KernelSpec,
    likelihood: GaussianLikelihood,
    x: &[f64],
    y: &[f64],
    z: &[f64],
    d: usize,
    ctx: &str,
) -> f64 {
    assert_eq!((online.n(), online.m()), (y.len(), z.len() / d), "{ctx}");
    assert_slice_close(online.x(), x, TOL);
    assert_slice_close(online.z(), z, TOL);
    let xs = f32_queries(d);
    check_f32_step(
        &online_values(online, &xs, d),
        &factor_values::<P>(kernel, likelihood, x, y, z, &xs, d),
        &factor_values::<DoublePrecision>(kernel, likelihood, x, y, z, &xs, d),
        ctx,
    )
}

/// Four queries of `d` features across the range of the data.
fn f32_queries(d: usize) -> Vec<f64> {
    (0..4 * d).map(|i| 0.3 + 0.9 * (i % 4) as f64).collect()
}

/// How [`run_f32_sequence`] draws a new inducing point.
#[derive(Clone, Copy, Debug)]
enum InducingDraw {
    /// At least [`MIN_GAP`] from every inducing point ([`draw_inducing`]):
    /// every pivot of the factor of `K_mm` holds the threshold, so every
    /// `X` step updates by rank-1.
    Gap,
    /// Anywhere in the range: near duplicates only by chance.
    Free,
    /// A copy of an inducing point, a point `1e-5` to `1e-2` per
    /// coordinate from one, or a free point ([`mixed_inducing`]): an `X`
    /// step from a factor with a pivot below the threshold assembles again.
    Duplicates,
}

/// One kind of [`run_f32_sequence`]: the `kernel` of `d` features, the
/// noise variance, how inducing points are drawn, and the steps (`X` and
/// inducing steps, or `X` steps only).
#[derive(Clone, Copy)]
struct F32Sequence<'a> {
    kernel: &'a KernelSpec,
    d: usize,
    noise: f64,
    draw: InducingDraw,
    steps: usize,
    x_only: bool,
}

/// The training points, targets, and inducing points a sequence of `d`
/// features starts from: [`MIXED_N0`] and [`MIXED_M0`] points, the
/// inducing points at [`MIN_GAP`] for [`InducingDraw::Gap`].
#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn f32_start(rng: &mut SmallRng, d: usize, draw: InducingDraw) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..MIXED_N0 * d).map(|_| sample_coord(rng)).collect();
    let y: Vec<f64> = (0..MIXED_N0).map(|_| sample_coord(rng)).collect();
    let mut z = Vec::new();
    for m in 0..MIXED_M0 {
        let z_new = match draw {
            InducingDraw::Gap => draw_inducing(rng, &z, m, d).expect("room for the first points"),
            InducingDraw::Free | InducingDraw::Duplicates => {
                (0..d).map(|_| sample_coord(rng)).collect()
            }
        };
        append_colmajor(&mut z, m, d, &z_new);
    }
    (x, y, z)
}

/// An `X` insert or delete that keeps `n` within [`MIXED_N_RANGE`].
fn x_op(rng: &mut SmallRng, n: usize) -> MixedOp {
    match rng.random_range(0..2) {
        0 if n < MIXED_N_RANGE.1 => MixedOp::Insert,
        _ if n > MIXED_N_RANGE.0 => MixedOp::Delete(rng.random_range(0..n)),
        _ => MixedOp::Insert,
    }
}

/// Runs `op` (an `X` insert or delete) on `online` and on `x` and `y`
/// (`d` features).
#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn apply_x_op<P: GpScalar>(
    online: &mut OnlineSgpr<Fixed, P>,
    rng: &mut SmallRng,
    op: MixedOp,
    x: &mut Vec<f64>,
    y: &mut Vec<f64>,
    d: usize,
) {
    let n = online.n();
    match op {
        MixedOp::Insert => {
            let x_new: Vec<f64> = (0..d).map(|_| sample_coord(rng)).collect();
            let y_new = sample_coord(rng);
            online.insert(&x_new, y_new).expect("insert");
            append_colmajor(x, n, d, &x_new);
            y.push(y_new);
        }
        MixedOp::Delete(index) => {
            online.delete(online.point_ids()[index]).expect("delete");
            remove_colmajor(x, n, d, index);
            y.remove(index);
        }
        _ => unreachable!("an X step"),
    }
}

/// One sequence of `seq.steps` steps from `seed` at an `f32` precision
/// `P`: [`MIXED_N0`] training points and [`MIXED_M0`] inducing points,
/// then `X` inserts and deletes and, unless `seq.x_only`, inducing inserts
/// (drawn as `seq.draw` says) and deletes. Every step is checked by
/// [`check_f32_step`]; returns the largest difference from a fresh factor
/// with the step it was found at.
#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn run_f32_sequence<P: GpScalar>(seq: F32Sequence<'_>, seed: u64) -> (f64, String) {
    let F32Sequence {
        kernel,
        d,
        noise,
        draw,
        steps,
        x_only,
    } = seq;
    let likelihood = GaussianLikelihood::new(noise).expect("noise");
    let mut rng = SmallRng::seed_from_u64(seed);
    let (mut x, mut y, mut z) = f32_start(&mut rng, d, draw);
    let mut online = Sgpr::new(kernel.clone(), likelihood)
        .with_optimizer(Fixed)
        .with_precision::<P>()
        .factor(&x, MIXED_N0, d, &y, &z, MIXED_M0)
        .map_err(|(_, e)| e)
        .expect("factor")
        .into_online();
    let mut worst = (0.0, String::new());
    for step in 0..steps {
        let (n, m) = (online.n(), online.m());
        let mut op = if x_only {
            x_op(&mut rng, n)
        } else {
            mixed_op(&mut rng, n, m)
        };
        if let MixedOp::InducingDuplicate(_) | MixedOp::InducingNear(_) | MixedOp::InducingRandom =
            op
        {
            let z_new = match draw {
                InducingDraw::Gap => draw_inducing(&mut rng, &z, m, d),
                InducingDraw::Free => Some((0..d).map(|_| sample_coord(&mut rng)).collect()),
                InducingDraw::Duplicates => Some(mixed_inducing(&mut rng, op, &z, m, d)),
            };
            match z_new {
                Some(z_new) => {
                    online.insert_inducing(&z_new).expect("insert inducing");
                    append_colmajor(&mut z, m, d, &z_new);
                }
                None => op = MixedOp::InducingDelete(rng.random_range(0..m)),
            }
        }
        match op {
            MixedOp::InducingDelete(index) => {
                online
                    .delete_inducing(online.inducing_ids()[index])
                    .expect("delete inducing");
                remove_colmajor(&mut z, m, d, index);
            }
            MixedOp::Insert | MixedOp::Delete(_) => {
                apply_x_op(&mut online, &mut rng, op, &mut x, &mut y, d);
            }
            MixedOp::InducingDuplicate(_) | MixedOp::InducingNear(_) | MixedOp::InducingRandom => {}
        }
        let ctx = format!("{draw:?} seed={seed} step={step} m={m} {op:?}");
        let err = check_f32_online(&online, kernel, likelihood, &x, &y, &z, d, &ctx);
        if err > worst.0 {
            worst = (err, ctx);
        }
    }
    worst
}

/// Runs [`run_f32_sequence`] for each of `seeds`; returns the largest
/// difference from a fresh factor with where it was found.
fn run_f32_seeds<P: GpScalar>(seq: F32Sequence<'_>, seeds: std::ops::Range<u64>) -> (f64, String) {
    seeds
        .map(|seed| run_f32_sequence::<P>(seq, seed))
        .fold((0.0, String::new()), |a, b| if b.0 > a.0 { b } else { a })
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn wide_rbf() -> KernelSpec {
    KernelSpec::from(RbfKernel::new(2.0).expect("ℓ"))
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn wide_rbf_ard() -> KernelSpec {
    KernelSpec::from(RbfArdKernel::new(&[2.0, 3.0]).expect("ℓ"))
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn narrow_rbf() -> KernelSpec {
    KernelSpec::from(RbfKernel::new(0.5).expect("ℓ"))
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn narrow_rbf_ard() -> KernelSpec {
    KernelSpec::from(RbfArdKernel::new(&[0.5, 0.75]).expect("ℓ"))
}

/// Mixed sequences of `X` inserts and deletes and inducing changes whose
/// inducing points stay at least [`MIN_GAP`] apart (every `X` step a
/// rank-1 update), at an `f32` storage scalar with `σn² = 1e-3` and wide
/// lengthscales, on 1-D RBF and 2-D ARD RBF data: each step matches a
/// fresh factor of the same precision to [`F32_FACTOR_TOL`] and the `f64`
/// factor as [`check_f32_step`] says. When an `X` insert evaluated its
/// column of `A` in `f32` and solved it against the rounded factor of
/// `K_mm`, and the factor of `B` was updated in `f32`, a step was up to
/// `3.1e-5` (single, 1-D) and `1.3e-5` (single, 2-D) from a fresh factor.
#[test]
fn f32_storage_mixed_ops_with_separated_inducing_points_match_factor() {
    for (kernel, d) in [(wide_rbf(), 1), (wide_rbf_ard(), 2)] {
        let seq = F32Sequence {
            kernel: &kernel,
            d,
            noise: 1e-3,
            draw: InducingDraw::Gap,
            steps: MIXED_OPS,
            x_only: false,
        };
        run_f32_seeds::<SinglePrecision>(seq, MIXED_SEEDS);
        run_f32_seeds::<MixedPrecision>(seq, MIXED_SEEDS);
    }
}

/// Steps of each sequence of [`f32_storage_long_x_runs_match_factor`].
const LONG_X_OPS: usize = 1500;
/// Seeds of those sequences.
const LONG_X_SEEDS: std::ops::Range<u64> = 0..1;

/// [`LONG_X_OPS`] `X` inserts and deletes in a row at an `f32` storage
/// scalar, with `σn² = 1e-3` and the lengthscales of the mixed sequences
/// ([`mixed_rbf`], [`mixed_rbf_ard`]): no inducing change assembles the
/// system in between, so an error one step leaves in the factor of `B` or
/// in a running sum reaches every later step. Each step matches a fresh
/// factor of the same precision to [`F32_FACTOR_TOL`] and the `f64`
/// factor as [`check_f32_step`] says. When the factor of `B` and the
/// running sums were updated in `f32`, the steps drifted up to `5.8e-5`
/// (single, 1-D) from a fresh factor.
#[test]
fn f32_storage_long_x_runs_match_factor() {
    for (kernel, d) in [(mixed_rbf(), 1), (mixed_rbf_ard(), 2)] {
        let seq = F32Sequence {
            kernel: &kernel,
            d,
            noise: 1e-3,
            draw: InducingDraw::Gap,
            steps: LONG_X_OPS,
            x_only: true,
        };
        run_f32_seeds::<SinglePrecision>(seq, LONG_X_SEEDS);
        run_f32_seeds::<MixedPrecision>(seq, LONG_X_SEEDS);
    }
}

/// Seeds of [`f32_storage_stress_sequences_match_factor`] at
/// [`SinglePrecision`]: these include the seeds that were far from the
/// `f64` factor before the `f64` system started from the assembly's own
/// values (23, 29, 30). A [`MixedPrecision`] model takes the same `X` steps
/// and runs the first [`STRESS_MIXED_SEEDS`] only.
const STRESS_SEEDS: std::ops::Range<u64> = 20..36;
const STRESS_MIXED_SEEDS: std::ops::Range<u64> = 0..8;

/// Mixed sequences of `X` and inducing steps at an `f32` storage scalar
/// with `σn² = 1e-4` and short lengthscales (`0.5`; ARD `0.5`, `0.75`),
/// inducing points drawn at a gap, anywhere, and on or next to an
/// inducing point ([`InducingDraw`]), on 1-D RBF and 2-D ARD RBF data,
/// over [`STRESS_SEEDS`]: each step matches a fresh factor of the same
/// precision to [`F32_FACTOR_TOL`] and the `f64` factor as
/// [`check_f32_step`] says. When the `f64` factor of `B`, `A y`, and
/// `‖A‖_F²` the `X` steps update started from the stored (rounded) factor
/// and `A` and took the rounded columns of `A`, a step was up to `3.8e-5`
/// from a fresh factor (single, 2-D, free draws, seed 29, step 58: `3.8e-5`
/// from the `f64` factor where a fresh factor was `1.1e-7`) and
/// `1.7e-5` (single, 1-D, at a gap, seed 23, step 49).
#[test]
fn f32_storage_stress_sequences_match_factor() {
    for (kernel, d) in [(narrow_rbf(), 1), (narrow_rbf_ard(), 2)] {
        for draw in [
            InducingDraw::Gap,
            InducingDraw::Free,
            InducingDraw::Duplicates,
        ] {
            let seq = F32Sequence {
                kernel: &kernel,
                d,
                noise: 1e-4,
                draw,
                steps: MIXED_OPS,
                x_only: false,
            };
            run_f32_seeds::<SinglePrecision>(seq, STRESS_SEEDS);
            run_f32_seeds::<MixedPrecision>(seq, STRESS_MIXED_SEEDS);
        }
    }
}

/// `steps` random `X` inserts and deletes from `seed` on `online` (`f32`
/// storage, `d` features, `X`, `y`, `Z` its data), each checked by
/// [`check_f32_online`] against factors with `kernel` and `likelihood`.
#[allow(clippy::too_many_arguments)]
fn run_f32_x_steps<P: GpScalar>(
    online: &mut OnlineSgpr<Fixed, P>,
    kernel: &KernelSpec,
    likelihood: GaussianLikelihood,
    x: &mut Vec<f64>,
    y: &mut Vec<f64>,
    z: &[f64],
    d: usize,
    seed: u64,
    steps: usize,
    label: &str,
) {
    let mut rng = SmallRng::seed_from_u64(seed);
    for step in 0..steps {
        let op = x_op(&mut rng, online.n());
        apply_x_op(online, &mut rng, op, x, y, d);
        let ctx = format!("{label} step={step} {op:?}");
        check_f32_online(online, kernel, likelihood, x, y, z, d, &ctx);
    }
}

/// Steps of each run of the saved, loaded, and cloned `f32` models.
const F32_X_STEPS: usize = 40;

/// An online `f32` model of the stress data ([`f32_start`] from `seed`,
/// inducing points at a gap, [`narrow_rbf_ard`], `σn² = 1e-4`) after
/// [`F32_X_STEPS`] checked `X` steps, with its data.
#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn stepped_f32_online<P: GpScalar>(
    seed: u64,
    label: &str,
) -> (OnlineSgpr<Fixed, P>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let (kernel, likelihood) = (narrow_rbf_ard(), stress_likelihood());
    let mut rng = SmallRng::seed_from_u64(seed);
    let (mut x, mut y, z) = f32_start(&mut rng, 2, InducingDraw::Gap);
    let mut online = Sgpr::new(kernel.clone(), likelihood)
        .with_optimizer(Fixed)
        .with_precision::<P>()
        .factor(&x, MIXED_N0, 2, &y, &z, MIXED_M0)
        .map_err(|(_, e)| e)
        .expect("factor")
        .into_online();
    run_f32_x_steps(
        &mut online,
        &kernel,
        likelihood,
        &mut x,
        &mut y,
        &z,
        2,
        seed,
        F32_X_STEPS,
        label,
    );
    (online, x, y, z)
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn stress_likelihood() -> GaussianLikelihood {
    GaussianLikelihood::new(1e-4).expect("noise")
}

/// A directory under the system temporary directory for this process.
fn temp_dir(label: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("gprx-sgpr-ops-{}-{label}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

/// An online `f32` model saved after `X` steps loads with its points and
/// continues: the loaded model (assembled again from its data) and the
/// model it was saved from take the same `X` steps, and each step of both
/// is checked by [`check_f32_online`].
#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn saved_f32_online_continues<P: GpScalar>(
    label: &str,
    unwrap: fn(LoadedSgpr) -> Option<OnlineSgpr<Fixed, P>>,
) {
    let (kernel, likelihood) = (narrow_rbf_ard(), stress_likelihood());
    let (mut online, x, y, z) = stepped_f32_online::<P>(3, label);
    let dir = temp_dir(label);
    online.save(&dir).expect("save");
    let mut loaded = unwrap(LoadedSgpr::load(&dir, &PersistRegistry::new()).expect("load"))
        .expect("the saved precision");
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(loaded.point_ids(), online.point_ids());
    let ctx = format!("{label} loaded");
    check_f32_online(&loaded, &kernel, likelihood, &x, &y, &z, 2, &ctx);
    for (model, name) in [(&mut online, "saved"), (&mut loaded, "loaded")] {
        let (mut x, mut y) = (x.clone(), y.clone());
        let label = format!("{label} {name}");
        run_f32_x_steps(
            model,
            &kernel,
            likelihood,
            &mut x,
            &mut y,
            &z,
            2,
            11,
            F32_X_STEPS,
            &label,
        );
    }
}

/// [`saved_f32_online_continues`] at [`SinglePrecision`] and
/// [`MixedPrecision`].
#[test]
fn saved_f32_online_x_steps_match_factor() {
    saved_f32_online_continues::<SinglePrecision>("single", |loaded| match loaded {
        LoadedSgpr::OnlineSingle(model) => Some(model),
        _ => None,
    });
    saved_f32_online_continues::<MixedPrecision>("mixed", |loaded| match loaded {
        LoadedSgpr::OnlineMixed(model) => Some(model),
        _ => None,
    });
}

/// A fitted `f32` model saved and loaded, then made online: its `X` steps
/// start from the system the load assembled, and each is checked by
/// [`check_f32_online`].
#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn loaded_f32_fitted_goes_online<P: GpScalar>(
    label: &str,
    unwrap: fn(LoadedSgpr) -> Option<gprx::FittedSgpr<Fixed, gprx::FixedInducing, P>>,
) {
    let (kernel, likelihood) = (narrow_rbf_ard(), stress_likelihood());
    let mut rng = SmallRng::seed_from_u64(5);
    let (mut x, mut y, z) = f32_start(&mut rng, 2, InducingDraw::Gap);
    let fitted = Sgpr::new(kernel.clone(), likelihood)
        .with_optimizer(Fixed)
        .with_precision::<P>()
        .factor(&x, MIXED_N0, 2, &y, &z, MIXED_M0)
        .map_err(|(_, e)| e)
        .expect("factor");
    let dir = temp_dir(label);
    fitted.save(&dir).expect("save");
    let loaded = unwrap(LoadedSgpr::load(&dir, &PersistRegistry::new()).expect("load"))
        .expect("the saved precision");
    let _ = std::fs::remove_dir_all(&dir);
    let mut online = loaded.into_online();
    run_f32_x_steps(
        &mut online,
        &kernel,
        likelihood,
        &mut x,
        &mut y,
        &z,
        2,
        5,
        F32_X_STEPS,
        label,
    );
}

/// [`loaded_f32_fitted_goes_online`] at [`SinglePrecision`] and
/// [`MixedPrecision`].
#[test]
fn loaded_f32_fitted_into_online_x_steps_match_factor() {
    loaded_f32_fitted_goes_online::<SinglePrecision>("fitted-single", |loaded| match loaded {
        LoadedSgpr::Single(model) => Some(model),
        _ => None,
    });
    loaded_f32_fitted_goes_online::<MixedPrecision>("fitted-mixed", |loaded| match loaded {
        LoadedSgpr::Mixed(model) => Some(model),
        _ => None,
    });
}

/// A clone of an online `f32` model after `X` steps and the model it was
/// cloned from take different `X` steps: each keeps its own `f64` system,
/// and each step of both is checked by [`check_f32_online`].
fn cloned_f32_online_diverges<P: GpScalar>(label: &str) {
    let (kernel, likelihood) = (narrow_rbf_ard(), stress_likelihood());
    let (mut online, x, y, z) = stepped_f32_online::<P>(7, label);
    let mut clone = online.clone();
    for (model, seed, name) in [(&mut online, 21, "original"), (&mut clone, 22, "clone")] {
        let (mut x, mut y) = (x.clone(), y.clone());
        let label = format!("{label} {name}");
        run_f32_x_steps(
            model,
            &kernel,
            likelihood,
            &mut x,
            &mut y,
            &z,
            2,
            seed,
            F32_X_STEPS,
            &label,
        );
    }
}

/// [`cloned_f32_online_diverges`] at [`SinglePrecision`] and
/// [`MixedPrecision`].
#[test]
fn cloned_f32_online_x_steps_match_factor() {
    cloned_f32_online_diverges::<SinglePrecision>("clone-single");
    cloned_f32_online_diverges::<MixedPrecision>("clone-mixed");
}

/// Steps of each [`run_distance_f32_sequence`].
const DIST_F32_OPS: usize = 40;

/// The mean and the variance at [`DIST_QUERY`] and the NLML of a factor at
/// precision `P` of supplied squared distances among `points` (1-D) with
/// targets `y` and the training rows `inducing` as inducing points.
#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn distance_factor_values<P: GpScalar>(
    kernel: &DistanceKernel,
    image: &ScalarDistance,
    likelihood: GaussianLikelihood,
    points: &[f64],
    y: &[f64],
    inducing: &[usize],
) -> Vec<f64> {
    let z: Vec<f64> = inducing.iter().map(|&i| points[i]).collect();
    let full = Sgpr::new(kernel.clone(), likelihood)
        .with_optimizer(Fixed)
        .with_precision::<P>()
        .factor(
            [image.from_vec(sq_block(points, &z))],
            points.len(),
            y,
            inducing,
        )
        .map_err(|(_, e)| e)
        .expect("factor");
    let pred = full
        .predict(
            [image.from_vec(sq_block(&z, &DIST_QUERY))],
            DIST_QUERY.len(),
        )
        .expect("factor predict");
    let nlml = full.neg_log_marginal_likelihood().expect("factor nlml");
    prediction_values(&pred.mean, &pred.variance, nlml)
}

/// Random `X` inserts and deletes from `seed` on a model of supplied
/// squared distances at an `f32` precision `P` (`σn² = 1e-3`), its
/// inducing points three training points: each step is checked by
/// [`check_f32_step`] against factors of the same blocks.
#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn run_distance_f32_sequence<P: GpScalar>(seed: u64) {
    let image = ScalarDistance::new();
    let kernel = distance_rbf(&image);
    let likelihood = GaussianLikelihood::new(1e-3).expect("noise");
    let mut points = dist_points();
    let mut y: Vec<f64> = points.iter().map(|&p| (0.9 * p).sin()).collect();
    let mut inducing = vec![0, 4, 9];
    let z: Vec<f64> = inducing.iter().map(|&i| points[i]).collect();
    let mut online = Sgpr::new(kernel.clone(), likelihood)
        .with_optimizer(Fixed)
        .with_precision::<P>()
        .factor(
            [image.from_vec(sq_block(&points, &z))],
            points.len(),
            &y,
            &inducing,
        )
        .map_err(|(_, e)| e)
        .expect("factor")
        .into_online();
    let mut rng = SmallRng::seed_from_u64(seed);
    let q = DIST_QUERY.len();
    for step in 0..DIST_F32_OPS {
        let n = points.len();
        let insert = n <= DIST_N || (n < 2 * DIST_N && rng.random::<bool>());
        if insert {
            let at = 6.0 * rng.random::<f64>();
            let y_new = (0.9 * at).sin() + 0.1 * (rng.random::<f64>() - 0.5);
            online
                .insert([image.from_vec(sq_block(&z, &[at]))], y_new)
                .expect("insert");
            points.push(at);
            y.push(y_new);
        } else {
            let free: Vec<usize> = (0..n).filter(|i| !inducing.contains(i)).collect();
            let row = free[rng.random_range(0..free.len())];
            online.delete(online.point_ids()[row]).expect("delete");
            points.remove(row);
            y.remove(row);
            for i in &mut inducing {
                if *i > row {
                    *i -= 1;
                }
            }
        }
        let pred = online
            .predict([image.from_vec(sq_block(&z, &DIST_QUERY))], q)
            .expect("online predict");
        let nlml = online.neg_log_marginal_likelihood().expect("online nlml");
        let ctx = format!("seed={seed} step={step} insert={insert}");
        check_f32_step(
            &prediction_values(&pred.mean, &pred.variance, nlml),
            &distance_factor_values::<P>(&kernel, &image, likelihood, &points, &y, &inducing),
            &distance_factor_values::<DoublePrecision>(
                &kernel, &image, likelihood, &points, &y, &inducing,
            ),
            &ctx,
        );
    }
}

/// [`run_distance_f32_sequence`] at [`SinglePrecision`] and
/// [`MixedPrecision`], seeds `0` to `3`: a model of supplied distances
/// evaluates the new column of `A` from the `f64` row it is given and, for
/// a delete, from the `f64` row its training blocks hold.
#[test]
fn distance_f32_x_steps_match_factor() {
    for seed in 0..4 {
        run_distance_f32_sequence::<SinglePrecision>(seed);
        run_distance_f32_sequence::<MixedPrecision>(seed);
    }
}

/// How far `online` is from the exact system (the mean and the variance at
/// `xs` and the NLML, computed in double-double by
/// [`common::dd::vfe_reference`]) next to how far a factor of the same
/// points at precision `P` is: `(online from exact, factor from exact,
/// online from factor)`, each the largest [`common::rel_err`]. `ls` are the
/// lengthscales of the RBF `kernel`.
#[allow(clippy::too_many_arguments)]
fn distances_from_the_exact_system<P: GpScalar>(
    online: &OnlineSgpr<Fixed, P>,
    kernel: &KernelSpec,
    ls: &[f64],
    likelihood: GaussianLikelihood,
    x: &[f64],
    y: &[f64],
    z: &[f64],
    xs: &[f64],
    d: usize,
) -> (f64, f64, f64) {
    let exact = common::dd::vfe_reference(ls, likelihood.noise_variance(), x, y, z, xs, d);
    let fresh = factor_values::<P>(kernel, likelihood, x, y, z, xs, d);
    let got = online_values(online, xs, d);
    (
        max_rel_err(&got, &exact),
        max_rel_err(&fresh, &exact),
        max_rel_err(&got, &fresh),
    )
}

/// The absolute floor of the bound an `f64` `X` step is held to against
/// the exact system (next to [`TRUTH_FACTOR`] times the distance of a
/// fresh factor): the tolerance of the inducing sequences,
/// [`INDUCING_TOL`].
const F64_TRUTH_FLOOR: f64 = INDUCING_TOL;

/// `n` points of `[0, w]` and a smooth target with noise, from `seed`.
fn smooth_data(n: usize, w: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let x: Vec<f64> = (0..n).map(|_| w * rng.random::<f64>()).collect();
    let y = x
        .iter()
        .map(|&v| (2.5 * v / w).sin() + 0.3 * (rng.random::<f64>() - 0.5))
        .collect();
    (x, y)
}

/// The double-double reference is accurate to `1e-30` in its arithmetic
/// (`exp` against two values computed to 50 digits) and agrees with a
/// factor where the factor is accurate: the values the bounds below are
/// measured against.
#[test]
fn double_double_reference_matches_a_well_conditioned_factor() {
    use common::dd::Dd;
    for (arg, hi, lo) in [
        (1.0, std::f64::consts::E, 1.445_646_891_729_250_2e-16),
        (
            -7.3,
            0.000_675_538_775_193_844_4,
            -2.907_750_493_846_276_6e-20,
        ),
    ] {
        let want = Dd::new(hi) + Dd::new(lo);
        let diff = (Dd::new(arg).exp() - want).to_f64().abs();
        assert!(diff <= 1e-30 * hi, "exp({arg}): {diff:e}");
    }
    let third = Dd::ONE / (Dd::ONE + Dd::ONE + Dd::ONE);
    let diff = (third * (Dd::ONE + Dd::ONE + Dd::ONE) - Dd::ONE)
        .to_f64()
        .abs();
    assert!(diff <= 1e-31, "1/3 * 3 - 1: {diff:e}");
    let two = Dd::ONE + Dd::ONE;
    let diff = (two.sqrt() * two.sqrt() - two).to_f64().abs();
    assert!(diff <= 1e-31, "sqrt(2)^2 - 2: {diff:e}");
    let (x, y) = smooth_data(40, 4.0, 3);
    let z = [0.4, 1.4, 2.4, 3.4];
    let kernel = KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"));
    let likelihood = GaussianLikelihood::new(0.1).expect("noise");
    let xs = [0.3, 1.2, 2.1, 3.0];
    let exact = common::dd::vfe_reference(&[1.0], 0.1, &x, &y, &z, &xs, 1);
    let fresh = factor_values::<DoublePrecision>(&kernel, likelihood, &x, &y, &z, &xs, 1);
    let err = max_rel_err(&fresh, &exact);
    assert!(err < 1e-13, "{err:e}");
}

/// `X` inserts and deletes next to a near duplicate inducing point, on 1-D
/// RBF data at `ℓ = 1`: the inducing set is `0.4, 1.4, 2.4, 3.4` and a
/// point `δ` from `1.4`, so the smallest pivot ratio of `K_mm` runs from
/// about `5e-3` (`δ = 0.1`) down to the rounding residue (`δ ≤ 1e-8`).
/// Every step is within [`TRUTH_FACTOR`] times the distance a factor of
/// the same points has from the exact system, plus [`F64_TRUTH_FLOOR`],
/// of that system: the guarantee of an `X` step whichever path it takes
/// (from a pivot ratio above `8 ε` it updates by rank-1, below it
/// assembles).
#[test]
fn x_steps_next_to_a_near_duplicate_stay_near_the_exact_system() {
    let kernel = KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"));
    let xs = [0.3, 1.2, 2.1, 3.0];
    for noise in [1e-1, 1e-3] {
        let likelihood = GaussianLikelihood::new(noise).expect("noise");
        for exponent in 1..=9 {
            let delta = 10f64.powi(-exponent);
            let z = [0.4, 1.4, 2.4, 3.4, 1.4 + delta];
            let (mut x, mut y) = smooth_data(30, 4.0, 11);
            let mut online = Sgpr::new(kernel.clone(), likelihood)
                .with_optimizer(Fixed)
                .factor(&x, y.len(), 1, &y, &z, z.len())
                .map_err(|(_, e)| e)
                .expect("factor")
                .into_online();
            let mut rng = SmallRng::seed_from_u64(7);
            for step in 0..8 {
                let n = y.len();
                if step % 2 == 0 {
                    let (x_new, y_new) = (4.0 * rng.random::<f64>(), rng.random::<f64>());
                    online.insert(&[x_new], y_new).expect("insert");
                    x.push(x_new);
                    y.push(y_new);
                } else {
                    let index = rng.random_range(0..n);
                    online.delete(online.point_ids()[index]).expect("delete");
                    x.remove(index);
                    y.remove(index);
                }
                let (on_exact, fresh_exact, _) = distances_from_the_exact_system(
                    &online,
                    &kernel,
                    &[1.0],
                    likelihood,
                    &x,
                    &y,
                    &z,
                    &xs,
                    1,
                );
                assert!(
                    on_exact <= TRUTH_FACTOR * fresh_exact + F64_TRUTH_FLOOR,
                    "noise={noise} delta={delta:e} step={step}: {on_exact:e} from the exact \
                     system, a factor {fresh_exact:e}"
                );
            }
        }
    }
}

/// A benchmark-like problem of `compare/perf`: all `n` points and targets
/// (`d` features, column-major), the kernel and its lengthscales, the
/// inducing points, and the queries.
struct SmoothProblem {
    kernel: KernelSpec,
    ls: Vec<f64>,
    d: usize,
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    xs: Vec<f64>,
}

/// The Forrester problem: `n` points of `[0, 1]`, `f(x) = (6x - 2)² sin(12x
/// - 4)` plus noise, RBF at `ℓ = 1`, and the first eight k-means centres of
/// the benchmark as inducing points (rounded to three digits): a smallest
/// pivot ratio of `K_mm` of about `1e-13`.
#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn forrester_problem(n: usize) -> SmoothProblem {
    let mut rng = SmallRng::seed_from_u64(0);
    let x: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
    let y = x
        .iter()
        .map(|&v| (6.0 * v - 2.0).powi(2) * (12.0 * v - 4.0).sin() + (rng.random::<f64>() - 0.5))
        .collect();
    SmoothProblem {
        kernel: KernelSpec::from(RbfKernel::new(1.0).expect("ℓ")),
        ls: vec![1.0],
        d: 1,
        x,
        y,
        z: vec![0.225, 0.606, 0.975, 0.09, 0.42, 0.737, 0.802, 0.543],
        xs: vec![0.05, 0.35, 0.65, 0.95],
    }
}

/// The weighted sphere problem: a `side × side` grid of `[0, 1]²`, `f =
/// (x / 0.25)² + y²` plus noise, ARD RBF at `ℓ_d = 4`, and the first eight
/// k-means centres of the benchmark as inducing points: a smallest pivot
/// ratio of `K_mm` of about `1e-7`.
#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn sphere_problem(side: usize) -> SmoothProblem {
    let mut rng = SmallRng::seed_from_u64(9);
    let n = side * side;
    let grid = |i: usize| i as f64 / (side - 1) as f64;
    let mut x = vec![0.0; 2 * n];
    for i in 0..n {
        x[i] = grid(i % side);
        x[n + i] = grid(i / side);
    }
    let y = (0..n)
        .map(|i| (x[i] / 0.25).powi(2) + x[n + i].powi(2) + (rng.random::<f64>() - 0.5))
        .collect();
    SmoothProblem {
        kernel: KernelSpec::from(RbfArdKernel::new(&[4.0, 4.0]).expect("ℓ")),
        ls: vec![4.0, 4.0],
        d: 2,
        x,
        y,
        z: vec![
            0.45, 0.65, 0.104, 0.9, 0.357, 0.467, 0.933, 0.3, 0.696, 0.233, 0.442, 0.9, 0.914,
            0.058, 0.633, 0.242,
        ],
        xs: vec![0.2, 0.5, 0.8, 0.35, 0.3, 0.6, 0.1, 0.9],
    }
}

/// The `X` steps of the benchmark: from all but the last 16 points, insert
/// those 16 in order, then delete 16 points spread over the model. Calls
/// `check` after the model took each step, with the training points and
/// targets the model now holds.
#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn run_smooth_steps<P: GpScalar>(
    problem: &SmoothProblem,
    noise: f64,
    mut check: impl FnMut(&OnlineSgpr<Fixed, P>, &[f64], &[f64], &str),
) {
    let SmoothProblem {
        kernel, d, x, y, z, ..
    } = problem;
    let d = *d;
    let n_all = y.len();
    let n0 = n_all - 16;
    let likelihood = GaussianLikelihood::new(noise).expect("noise");
    let mut cur_x: Vec<f64> = (0..d)
        .flat_map(|f| x[f * n_all..f * n_all + n0].iter().copied())
        .collect();
    let mut cur_y = y[..n0].to_vec();
    let mut online = Sgpr::new(kernel.clone(), likelihood)
        .with_optimizer(Fixed)
        .with_precision::<P>()
        .factor(&cur_x, n0, d, &cur_y, z, z.len() / d)
        .map_err(|(_, e)| e)
        .expect("factor")
        .into_online();
    for i in n0..n_all {
        let n = cur_y.len();
        let x_new: Vec<f64> = (0..d).map(|f| x[f * n_all + i]).collect();
        online.insert(&x_new, y[i]).expect("insert");
        append_colmajor(&mut cur_x, n, d, &x_new);
        cur_y.push(y[i]);
        check(&online, &cur_x, &cur_y, &format!("insert {i}"));
    }
    for step in 0..16 {
        let n = cur_y.len();
        let index = (n * (2 * step + 1) / 33).min(n - 1);
        online.delete(online.point_ids()[index]).expect("delete");
        remove_colmajor(&mut cur_x, n, d, index);
        cur_y.remove(index);
        check(
            &online,
            &cur_x,
            &cur_y,
            &format!("delete {step} at {index}"),
        );
    }
}

/// `X` steps on the benchmark problems of `compare/perf` (Forrester at
/// `n = 256` and `1024`, the weighted sphere at `16 × 16` and `32 × 32`,
/// `σn² = 0.1`), from a `K_mm` whose smallest pivot ratio is far below the
/// threshold of an inducing change (`1e-13` and `1e-7`) without any
/// duplicate. Each step takes the rank-1 path
/// (`x_steps_stay_rank_1_for_a_smooth_ill_conditioned_k_mm` in
/// `src/sgpr/online/tests.rs`) and is within [`TRUTH_FACTOR`] times the
/// distance a fresh factor has from the exact system, plus
/// [`F64_TRUTH_FLOOR`], of that system, and within `1e-10` of the fresh
/// factor. Before the threshold of an `X` step was lowered each of those
/// steps assembled the system again and equalled the factor.
#[test]
fn x_steps_on_a_smooth_ill_conditioned_k_mm_stay_near_the_exact_system() {
    for problem in [
        forrester_problem(256),
        forrester_problem(1024),
        sphere_problem(16),
        sphere_problem(32),
    ] {
        let likelihood = GaussianLikelihood::new(0.1).expect("noise");
        run_smooth_steps::<DoublePrecision>(&problem, 0.1, |online, x, y, ctx| {
            let (on_exact, fresh_exact, on_fresh) = distances_from_the_exact_system(
                online,
                &problem.kernel,
                &problem.ls,
                likelihood,
                x,
                y,
                &problem.z,
                &problem.xs,
                problem.d,
            );
            assert!(
                on_exact <= TRUTH_FACTOR * fresh_exact + F64_TRUTH_FLOOR,
                "d={} n={} {ctx}: {on_exact:e} from the exact system, a factor {fresh_exact:e}",
                problem.d,
                y.len()
            );
            assert!(
                on_fresh <= INDUCING_TOL,
                "{ctx}: {on_fresh:e} from a factor"
            );
        });
    }
}

/// [`x_steps_on_a_smooth_ill_conditioned_k_mm_stay_near_the_exact_system`]
/// at an `f32` storage scalar: each step is within [`F32_FACTOR_TOL`] of a
/// fresh factor of the same precision and within the bound of
/// [`check_f32_step`] of the `f64` factor (the exact system to `1e-11`
/// here). The step computes in `f64` (R8-3), so its values are the
/// rounding of the `f64` system's.
#[test]
fn f32_x_steps_on_a_smooth_ill_conditioned_k_mm_match_factor() {
    fn run<P: GpScalar>(problem: &SmoothProblem) {
        let likelihood = GaussianLikelihood::new(0.1).expect("noise");
        let SmoothProblem {
            kernel, d, z, xs, ..
        } = problem;
        run_smooth_steps::<P>(problem, 0.1, |online, x, y, ctx| {
            check_f32_step(
                &online_values(online, xs, *d),
                &factor_values::<P>(kernel, likelihood, x, y, z, xs, *d),
                &factor_values::<DoublePrecision>(kernel, likelihood, x, y, z, xs, *d),
                ctx,
            );
        });
    }
    for problem in [forrester_problem(256), sphere_problem(16)] {
        run::<SinglePrecision>(&problem);
        run::<MixedPrecision>(&problem);
    }
}
