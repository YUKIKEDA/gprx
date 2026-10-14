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
//! models, `f64` and `f32`), and mixed sequences of inducing and `X` steps
//! with duplicate and near duplicate inducing points. This file uses only
//! the public API.

use std::collections::HashSet;

use gprx::kernel::KernelScalar;
use gprx::kernel::{
    DistanceKernel, KernelSpec, MaternKernel, MaternNu, RbfArdKernel, RbfKernel, ScalarDistance,
    WhiteKernel,
};
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
/// `y`, and `Z`. An inducing change and an `X` step from a factor of
/// `K_mm` with a pivot `L_ii² ≤ 1e-5 (K_ii + j)` assemble the system
/// again. Before an inducing change did, a delete there was `0.20` off
/// (seed 2, step 46); before an `X` step did, an `X` step after an exact
/// duplicate was `2.9e-4` off (seed 5, step 8), its rank-1 update of `B`
/// carrying the rounding of a residue pivot. The largest difference is now
/// `1e-13`.
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
/// `1e-3` times its diagonal. Updated by rank-1 there, an `X` step after an
/// exact duplicate was `2.3e10` off on 1-D RBF data (seed 5, step 8, both
/// precisions) and `30` on 2-D ARD data (seed 5, step 48). The largest
/// difference is now `1.0e-6`.
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
/// `K_mm` a pivot of `1.4e-5` times its diagonal: above the `f64`
/// threshold of `1e-5`, so `X` steps of a [`DoublePrecision`] model update
/// by rank-1, and below the `f32` one of `1e-3`, so those of a
/// [`SinglePrecision`] or [`MixedPrecision`] model assemble again.
/// Updated by rank-1 in `f32`, the steps were `1.6e-5` (single) and
/// `9.7e-6` (mixed) off.
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
