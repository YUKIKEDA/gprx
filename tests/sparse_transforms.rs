//! Input and target transforms on the sparse models (R5-3 / #281).
//!
//! A sparse model with transforms must match the same model without
//! transforms on data mapped by hand with the same fitted maps, with the
//! predictions mapped back. This file uses only the public API.

use std::num::{NonZeroU64, NonZeroUsize};

use gprx::kernel::{KernelSpec, RbfArdKernel};
use gprx::transform::{
    MinMaxInput, StandardizeInput, StandardizeTarget, TargetTransform, Transform,
};
use gprx::{Adam, Fixed, FreeInducing, GaussianLikelihood, Gpr, Prediction, Sgpr, Svgp};

mod common;
use common::{assert_close, assert_slice_close};

const TOL: f64 = 1e-12;
/// VFE at `Z = X` against Exact, and a rank-1 update against a reassemble.
const REASSEMBLE_TOL: f64 = 1e-9;
const N: usize = 12;
const M: usize = 5;
const D: usize = 2;
const Q: usize = 4;

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn kernel() -> KernelSpec {
    KernelSpec::from(RbfArdKernel::new(&[1.0, 1.0]).expect("valid"))
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn likelihood() -> GaussianLikelihood {
    GaussianLikelihood::new(0.1).expect("valid")
}

/// Column-major points on two features with different offsets and scales.
fn points(rows: usize, shift: f64) -> Vec<f64> {
    let mut x = vec![0.0; rows * D];
    for i in 0..rows {
        let t = i as f64 + shift;
        x[i] = 100.0 + 3.0 * t;
        x[rows + i] = -0.02 * t + 0.005 * (1.7 * t).sin();
    }
    x
}

fn targets(x: &[f64], rows: usize) -> Vec<f64> {
    (0..rows)
        .map(|i| 50.0 + 4.0 * (x[i] / 7.0).sin() + 200.0 * x[rows + i])
        .collect()
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn mapped(map: &dyn Transform, x: &[f64], rows: usize) -> Vec<f64> {
    let mut out = x.to_vec();
    map.apply(&mut out, rows, D).expect("fitted map");
    out
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn mapped_targets(map: &dyn TargetTransform, y: &[f64]) -> Vec<f64> {
    let mut out = y.to_vec();
    map.transform(&mut out).expect("fitted map");
    out
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn unmapped(map: &dyn TargetTransform, mut pred: Prediction) -> Prediction {
    map.inverse_transform_mean(&mut pred.mean)
        .expect("fitted map");
    map.inverse_transform_variance(&mut pred.variance)
        .expect("fitted map");
    pred
}

fn append(x: &[f64], rows: usize, point: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity((rows + 1) * D);
    for feature in 0..D {
        out.extend_from_slice(&x[feature * rows..(feature + 1) * rows]);
        out.push(point[feature]);
    }
    out
}

fn assert_prediction_close(actual: &Prediction, expected: &Prediction, tol: f64) {
    assert_eq!(actual.variance_kind, expected.variance_kind);
    assert_slice_close(&actual.mean, &expected.mean, tol);
    assert_slice_close(&actual.variance, &expected.variance, tol);
}

struct Data {
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    xs: Vec<f64>,
}

fn data() -> Data {
    let x = points(N, 0.0);
    let y = targets(&x, N);
    Data {
        z: points(M, 0.3),
        xs: points(Q, 0.45),
        x,
        y,
    }
}

#[test]
fn sgpr_factor_matches_hand_mapped_data() {
    let Data { x, y, z, xs } = data();
    let x_map = StandardizeInput::new().fit(&x, N, D).expect("valid");
    let y_map = StandardizeTarget::new().fit(&y).expect("valid");
    let fitted = Sgpr::new(kernel(), likelihood())
        .with_optimizer(Fixed)
        .with_input_transform(StandardizeInput::new())
        .with_target_transform(StandardizeTarget::new())
        .factor(&x, N, D, &y, &z, M)
        .map_err(|(_, e)| e)
        .expect("factor");
    let reference = Sgpr::new(kernel(), likelihood())
        .with_optimizer(Fixed)
        .factor(
            &mapped(&x_map, &x, N),
            N,
            D,
            &mapped_targets(&y_map, &y),
            &mapped(&x_map, &z, M),
            M,
        )
        .map_err(|(_, e)| e)
        .expect("factor");
    let expected = unmapped(
        &y_map,
        reference
            .predict(&mapped(&x_map, &xs, Q), Q, D)
            .expect("predict"),
    );
    assert_prediction_close(&fitted.predict(&xs, Q, D).expect("predict"), &expected, TOL);
    assert_close(
        fitted.neg_log_marginal_likelihood().expect("nlml"),
        reference.neg_log_marginal_likelihood().expect("nlml"),
        TOL,
    );
    assert_eq!(fitted.x(), x.as_slice());
    assert_eq!(fitted.y(), y.as_slice());
    assert_eq!(fitted.z(), z.as_slice());
}

#[test]
fn sgpr_at_training_inducing_matches_exact_with_transforms() {
    let Data { x, y, xs, .. } = data();
    let sparse = Sgpr::new(kernel(), likelihood())
        .with_optimizer(Fixed)
        .with_input_transform(MinMaxInput::new())
        .with_target_transform(StandardizeTarget::new())
        .factor(&x, N, D, &y, &x, N)
        .map_err(|(_, e)| e)
        .expect("factor");
    let exact = Gpr::new(kernel(), likelihood())
        .with_optimizer(Fixed)
        .with_input_transform(MinMaxInput::new())
        .with_target_transform(StandardizeTarget::new())
        .factor(&x, N, D, &y)
        .map_err(|(_, e)| e)
        .expect("factor");
    assert_prediction_close(
        &sparse.predict(&xs, Q, D).expect("predict"),
        &exact.predict(&xs, Q, D).expect("predict"),
        REASSEMBLE_TOL,
    );
    assert_close(
        sparse.neg_log_marginal_likelihood().expect("nlml"),
        exact.neg_log_marginal_likelihood().expect("nlml"),
        REASSEMBLE_TOL,
    );
}

#[test]
fn free_inducing_searches_mapped_z_and_reports_original_z() {
    let Data { x, y, z, xs } = data();
    let x_map = MinMaxInput::new().fit(&x, N, D).expect("valid");
    let y_map = StandardizeTarget::new().fit(&y).expect("valid");
    let fitted = Sgpr::new(kernel(), likelihood())
        .with_inducing(FreeInducing)
        .with_input_transform(MinMaxInput::new())
        .with_target_transform(StandardizeTarget::new())
        .fit(&x, N, D, &y, &z, M)
        .map_err(|(_, e)| e)
        .expect("fit");
    let reference = Sgpr::new(kernel(), likelihood())
        .with_inducing(FreeInducing)
        .fit(
            &mapped(&x_map, &x, N),
            N,
            D,
            &mapped_targets(&y_map, &y),
            &mapped(&x_map, &z, M),
            M,
        )
        .map_err(|(_, e)| e)
        .expect("fit");
    let mut params = vec![0.0; fitted.num_params()];
    let mut reference_params = vec![0.0; reference.num_params()];
    fitted.get_params(&mut params).expect("params");
    reference.get_params(&mut reference_params).expect("params");
    assert_slice_close(&params, &reference_params, TOL);
    let mut z_original = reference.z().to_vec();
    x_map
        .inverse_apply(&mut z_original, M, D)
        .expect("fitted map");
    assert_slice_close(fitted.z(), &z_original, TOL);
    assert!(
        fitted
            .z()
            .iter()
            .zip(&z)
            .any(|(moved, start)| (moved - start).abs() > 1e-6),
        "the search moved Z"
    );
    let expected = unmapped(
        &y_map,
        reference
            .predict(&mapped(&x_map, &xs, Q), Q, D)
            .expect("predict"),
    );
    assert_prediction_close(&fitted.predict(&xs, Q, D).expect("predict"), &expected, TOL);
}

#[test]
fn online_updates_map_new_points_with_the_training_fit() {
    let Data { x, y, z, xs } = data();
    let x_map = StandardizeInput::new().fit(&x, N, D).expect("valid");
    let y_map = StandardizeTarget::new().fit(&y).expect("valid");
    let mut online = Sgpr::new(kernel(), likelihood())
        .with_optimizer(Fixed)
        .with_input_transform(StandardizeInput::new())
        .with_target_transform(StandardizeTarget::new())
        .factor(&x, N, D, &y, &z, M)
        .map_err(|(_, e)| e)
        .expect("factor")
        .into_online();
    let x_new = [140.0, -0.3];
    let y_new = -12.0;
    let z_new = [118.0, -0.1];
    online.insert(&x_new, y_new).expect("insert");
    online.insert_inducing(&z_new).expect("insert inducing");
    let x_all = append(&x, N, &x_new);
    let mut y_all = y.clone();
    y_all.push(y_new);
    let z_all = append(&z, M, &z_new);
    assert_eq!(online.x(), x_all.as_slice());
    assert_eq!(online.y(), y_all.as_slice());
    assert_eq!(online.z(), z_all.as_slice());
    let reference = Sgpr::new(kernel(), likelihood())
        .with_optimizer(Fixed)
        .factor(
            &mapped(&x_map, &x_all, N + 1),
            N + 1,
            D,
            &mapped_targets(&y_map, &y_all),
            &mapped(&x_map, &z_all, M + 1),
            M + 1,
        )
        .map_err(|(_, e)| e)
        .expect("factor");
    let expected = unmapped(
        &y_map,
        reference
            .predict(&mapped(&x_map, &xs, Q), Q, D)
            .expect("predict"),
    );
    assert_prediction_close(
        &online.predict(&xs, Q, D).expect("predict"),
        &expected,
        REASSEMBLE_TOL,
    );

    online.delete(online.point_ids()[0]).expect("delete");
    online
        .delete_inducing(online.inducing_ids()[0])
        .expect("delete inducing");
    assert_eq!(online.x(), points_without_first(&x_all, N + 1).as_slice());
    assert_eq!(online.y(), &y_all[1..]);
    assert_eq!(online.z(), points_without_first(&z_all, M + 1).as_slice());
}

fn points_without_first(x: &[f64], rows: usize) -> Vec<f64> {
    (0..D)
        .flat_map(|feature| x[feature * rows + 1..(feature + 1) * rows].iter().copied())
        .collect()
}

#[test]
fn svgp_factor_and_adam_match_hand_mapped_data() {
    let Data { x, y, z, xs } = data();
    let x_map = StandardizeInput::new().fit(&x, N, D).expect("valid");
    let y_map = StandardizeTarget::new().fit(&y).expect("valid");
    let x_t = mapped(&x_map, &x, N);
    let y_t = mapped_targets(&y_map, &y);
    let z_t = mapped(&x_map, &z, M);
    let fitted = Svgp::new(kernel(), likelihood())
        .with_input_transform(StandardizeInput::new())
        .with_target_transform(StandardizeTarget::new())
        .factor(&x, N, D, &y, &z, M)
        .map_err(|(_, e)| e)
        .expect("factor");
    let reference = Svgp::new(kernel(), likelihood())
        .factor(&x_t, N, D, &y_t, &z_t, M)
        .map_err(|(_, e)| e)
        .expect("factor");
    let expected = unmapped(
        &y_map,
        reference
            .predict(&mapped(&x_map, &xs, Q), Q, D)
            .expect("predict"),
    );
    assert_prediction_close(&fitted.predict(&xs, Q, D).expect("predict"), &expected, TOL);
    assert_eq!(fitted.z(), z.as_slice());

    let adam = Adam::new()
        .with_batch_size(NonZeroUsize::new(5).expect("non-zero"))
        .with_epochs(NonZeroU64::new(20).expect("non-zero"))
        .with_seed(7);
    let fitted = Svgp::new(kernel(), likelihood())
        .with_optimizer(adam.clone())
        .with_input_transform(StandardizeInput::new())
        .with_target_transform(StandardizeTarget::new())
        .fit(&x, N, D, &y, &z, M)
        .map_err(|(_, e)| e)
        .expect("fit");
    let reference = Svgp::new(kernel(), likelihood())
        .with_optimizer(adam)
        .fit(&x_t, N, D, &y_t, &z_t, M)
        .map_err(|(_, e)| e)
        .expect("fit");
    let mut params = vec![0.0; fitted.num_params()];
    let mut reference_params = vec![0.0; reference.num_params()];
    fitted.get_params(&mut params).expect("params");
    reference.get_params(&mut reference_params).expect("params");
    assert_slice_close(&params, &reference_params, TOL);
    let expected = unmapped(
        &y_map,
        reference
            .predict(&mapped(&x_map, &xs, Q), Q, D)
            .expect("predict"),
    );
    assert_prediction_close(&fitted.predict(&xs, Q, D).expect("predict"), &expected, TOL);
}
