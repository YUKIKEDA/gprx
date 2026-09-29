//! Leave-one-out prediction of `Sgpr` (R5-7 / #285).
//!
//! The closed form must match removing each point and factoring again at
//! the same `θ` and `Z`, Exact LOO at `Z = X`, the hand-mapped data under
//! transforms, and a refactor after online updates. This file uses only the
//! public API.

use gprx::kernel::{KernelSpec, RbfKernel};
use gprx::transform::{StandardizeInput, StandardizeTarget, TargetTransform, Transform};
use gprx::{
    Fixed, GaussianLikelihood, Gpr, PredictOptions, Prediction, Sgpr, SinglePrecision, VarianceKind,
};

mod common;
use common::{assert_close_named, assert_slice_close};

const N: usize = 11;
const M: usize = 4;
const D: usize = 2;
const TOL: f64 = 1e-10;
/// VFE at `Z = X` against Exact.
const EXACT_TOL: f64 = 1e-9;

fn points(rows: usize, shift: f64) -> Vec<f64> {
    let mut x = vec![0.0; rows * D];
    for i in 0..rows {
        let t = i as f64 + shift;
        x[i] = 0.45 * t;
        x[rows + i] = (0.8 * t).sin();
    }
    x
}

fn targets(x: &[f64], rows: usize) -> Vec<f64> {
    (0..rows)
        .map(|i| (1.1 * x[i]).cos() + 0.3 * x[rows + i])
        .collect()
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn trainer() -> Sgpr<Fixed> {
    Sgpr::new(
        KernelSpec::from(RbfKernel::new(1.1).expect("valid")),
        GaussianLikelihood::new(0.08).expect("valid"),
    )
    .with_optimizer(Fixed)
}

fn options(kind: VarianceKind) -> PredictOptions {
    PredictOptions {
        variance_kind: kind,
    }
}

fn without(x: &[f64], rows: usize, i: usize) -> Vec<f64> {
    (0..D)
        .flat_map(|dim| {
            (0..rows)
                .filter(move |&r| r != i)
                .map(move |r| x[dim * rows + r])
        })
        .collect()
}

fn point(x: &[f64], rows: usize, i: usize) -> Vec<f64> {
    (0..D).map(|dim| x[dim * rows + i]).collect()
}

#[test]
fn loo_matches_refactor_without_each_point() {
    let x = points(N, 0.0);
    let y = targets(&x, N);
    let z = points(M, 0.3);
    let fitted = trainer()
        .factor(&x, N, D, &y, &z, M)
        .map_err(|(_, e)| e)
        .expect("factor");
    for kind in [VarianceKind::Latent, VarianceKind::Observation] {
        let loo = fitted.loo_predict_with(options(kind)).expect("loo");
        assert_eq!(loo.variance_kind, kind);
        for i in 0..N {
            let y_rest: Vec<f64> = y
                .iter()
                .enumerate()
                .filter(|&(r, _)| r != i)
                .map(|(_, v)| *v)
                .collect();
            let rest = trainer()
                .factor(&without(&x, N, i), N - 1, D, &y_rest, &z, M)
                .map_err(|(_, e)| e)
                .expect("factor");
            let pred = rest
                .predict_with(&point(&x, N, i), 1, D, options(kind))
                .expect("predict");
            assert_close_named(&format!("mean {i}"), loo.mean[i], pred.mean[0], TOL);
            assert_close_named(&format!("var {i}"), loo.variance[i], pred.variance[0], TOL);
        }
    }
}

#[test]
fn loo_at_training_inducing_matches_exact() {
    let x = points(N, 0.0);
    let y = targets(&x, N);
    let sparse = trainer()
        .factor(&x, N, D, &y, &x, N)
        .map_err(|(_, e)| e)
        .expect("factor");
    let exact = Gpr::new(
        KernelSpec::from(RbfKernel::new(1.1).expect("valid")),
        GaussianLikelihood::new(0.08).expect("valid"),
    )
    .with_optimizer(Fixed)
    .factor(&x, N, D, &y)
    .map_err(|(_, e)| e)
    .expect("factor");
    for kind in [VarianceKind::Latent, VarianceKind::Observation] {
        let s = sparse.loo_predict_with(options(kind)).expect("sparse");
        let e = exact.loo_predict_with(options(kind)).expect("exact");
        assert_slice_close(&s.mean, &e.mean, EXACT_TOL);
        assert_slice_close(&s.variance, &e.variance, EXACT_TOL);
    }
}

#[test]
fn loo_with_transforms_matches_hand_mapped_data() {
    let x = points(N, 0.0);
    let y: Vec<f64> = targets(&x, N).iter().map(|v| 20.0 - 4.0 * v).collect();
    let z = points(M, 0.3);
    let fitted = trainer()
        .with_input_transform(StandardizeInput::new())
        .with_target_transform(StandardizeTarget::new())
        .factor(&x, N, D, &y, &z, M)
        .map_err(|(_, e)| e)
        .expect("factor");
    let x_map = StandardizeInput::new().fit(&x, N, D).expect("valid");
    let y_map = StandardizeTarget::new().fit(&y).expect("valid");
    let (mut x_t, mut z_t, mut y_t) = (x.clone(), z.clone(), y.clone());
    x_map.apply(&mut x_t, N, D).expect("map");
    x_map.apply(&mut z_t, M, D).expect("map");
    y_map.transform(&mut y_t).expect("map");
    let reference = trainer()
        .factor(&x_t, N, D, &y_t, &z_t, M)
        .map_err(|(_, e)| e)
        .expect("factor");
    let mut expected: Prediction = reference.loo_predict().expect("loo");
    y_map
        .inverse_transform_mean(&mut expected.mean)
        .expect("map");
    y_map
        .inverse_transform_variance(&mut expected.variance)
        .expect("map");
    let loo = fitted.loo_predict().expect("loo");
    assert_slice_close(&loo.mean, &expected.mean, 1e-12);
    assert_slice_close(&loo.variance, &expected.variance, 1e-12);
}

#[test]
fn online_loo_matches_refactor() {
    let x = points(N, 0.0);
    let y = targets(&x, N);
    let z = points(M, 0.3);
    let mut online = trainer()
        .factor(&x, N, D, &y, &z, M)
        .map_err(|(_, e)| e)
        .expect("factor")
        .into_online();
    let x_new = [1.9, -0.5];
    online.insert(&x_new, 0.4).expect("insert");
    online.delete(online.point_ids()[2]).expect("delete");
    let reference = trainer()
        .factor(
            online.x(),
            online.n(),
            D,
            online.y(),
            online.z(),
            online.m(),
        )
        .map_err(|(_, e)| e)
        .expect("factor");
    let loo = online.loo_predict().expect("loo");
    let expected = reference.loo_predict().expect("loo");
    assert_slice_close(&loo.mean, &expected.mean, TOL);
    assert_slice_close(&loo.variance, &expected.variance, TOL);
}

#[test]
fn f32_loo_is_close_to_f64() {
    let x = points(N, 0.0);
    let y = targets(&x, N);
    let z = points(M, 0.3);
    let f64_loo = trainer()
        .factor(&x, N, D, &y, &z, M)
        .map_err(|(_, e)| e)
        .expect("factor")
        .loo_predict()
        .expect("loo");
    let f32_loo = trainer()
        .with_precision::<SinglePrecision>()
        .factor(&x, N, D, &y, &z, M)
        .map_err(|(_, e)| e)
        .expect("factor")
        .loo_predict()
        .expect("loo");
    for i in 0..N {
        assert_close_named(
            &format!("mean {i}"),
            f64::from(f32_loo.mean[i]),
            f64_loo.mean[i],
            1e-5,
        );
        assert_close_named(
            &format!("var {i}"),
            f64::from(f32_loo.variance[i]),
            f64_loo.variance[i],
            1e-5,
        );
    }
}
