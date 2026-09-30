//! Sparse `predict_into` (R5-5 / #283).
//!
//! `predict_into` keeps its buffers on the model; it must return exactly
//! what `predict` returns, also after the query shape, the kernel `θ`, the
//! training set, or the inducing set changes. This file uses only the
//! public API.

use gprx::kernel::{ConstantKernel, KernelSpec, RbfKernel};
use gprx::transform::{StandardizeInput, StandardizeTarget};
use gprx::{
    DoublePrecision, Fixed, GaussianLikelihood, GpScalar, GprError, MixedPrecision, PredictOptions,
    Prediction, PromoteStorage, ReevaluateKernel, Sgpr, SinglePrecision, Svgp, VarianceKind,
};

const N: usize = 12;
const M: usize = 4;
const D: usize = 2;

fn points(rows: usize, shift: f64) -> Vec<f64> {
    let mut x = vec![0.0; rows * D];
    for i in 0..rows {
        let t = i as f64 + shift;
        x[i] = 0.4 * t;
        x[rows + i] = (0.7 * t).sin();
    }
    x
}

fn targets(x: &[f64], rows: usize) -> Vec<f64> {
    (0..rows).map(|i| x[i].cos() + 0.5 * x[rows + i]).collect()
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn kernels() -> [KernelSpec; 2] {
    let rbf = || KernelSpec::from(RbfKernel::new(1.0).expect("valid"));
    let constant = |v: f64| KernelSpec::from(ConstantKernel::new(v).expect("valid"));
    [
        rbf(),
        (rbf() + KernelSpec::from(RbfKernel::new(2.0).expect("valid")))
            * (constant(1.5) * rbf() + constant(0.5)),
    ]
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn likelihood() -> GaussianLikelihood {
    GaussianLikelihood::new(0.1).expect("valid")
}

/// Query shapes in an order that grows, shrinks, and grows the buffers.
const QUERIES: [usize; 4] = [5, 2, 7, 5];

fn options() -> [PredictOptions; 2] {
    [
        PredictOptions::default(),
        PredictOptions {
            variance_kind: VarianceKind::Latent,
        },
    ]
}

/// Checks every query shape and variance kind: `into` must equal `fresh`.
#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn check_same<T: PartialEq + std::fmt::Debug>(
    label: &str,
    out: &mut Prediction<T>,
    mut fresh: impl FnMut(&[f64], usize, PredictOptions) -> Result<Prediction<T>, GprError>,
    mut into: impl FnMut(&[f64], usize, PredictOptions, &mut Prediction<T>) -> Result<(), GprError>,
) {
    for (idx, q) in QUERIES.into_iter().enumerate() {
        let xs = points(q, 0.25 + idx as f64);
        for option in options() {
            into(&xs, q, option, out).expect("predict_into");
            let expected = fresh(&xs, q, option).expect("predict");
            assert_eq!(*out, expected, "{label}: q = {q}, {option:?}");
        }
    }
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn check_precision<P: GpScalar>(label: &str)
where
    P::Refine: PartialEq + std::fmt::Debug,
{
    let x = points(N, 0.0);
    let y = targets(&x, N);
    let z = points(M, 0.5);
    let mut out = Prediction::default();
    for kernel in kernels() {
        let mut fitted = Sgpr::new(kernel.clone(), likelihood())
            .with_optimizer(Fixed)
            .with_precision::<P>()
            .with_input_transform(StandardizeInput::new())
            .with_target_transform(StandardizeTarget::new())
            .factor(&x, N, D, &y, &z, M)
            .map_err(|(_, e)| e)
            .expect("factor");
        let snapshot = fitted.clone();
        check_same(
            &format!("{label} sgpr"),
            &mut out,
            |xs, q, o| snapshot.predict_with(xs, q, D, o),
            |xs, q, o, out| fitted.predict_with_into(xs, q, D, o, out),
        );
        // A new kernel θ must not reuse the compiled kernel of the old one.
        let mut params = vec![0.0; fitted.num_params()];
        fitted.get_params(&mut params).expect("params");
        for value in &mut params {
            *value += 0.3;
        }
        fitted.set_params(&params).expect("set params");
        let snapshot = fitted.clone();
        check_same(
            &format!("{label} sgpr after set_params"),
            &mut out,
            |xs, q, o| snapshot.predict_with(xs, q, D, o),
            |xs, q, o, out| fitted.predict_with_into(xs, q, D, o, out),
        );

        let mut online = fitted.into_online();
        online.insert(&[0.3, -0.2], 0.7).expect("insert");
        online
            .insert_inducing(&[1.1, 0.4])
            .expect("insert inducing");
        online
            .delete_inducing(online.inducing_ids()[0])
            .expect("delete inducing");
        online
            .insert_inducing(&[2.2, -0.6])
            .expect("insert inducing");
        let snapshot = online.clone();
        check_same(
            &format!("{label} online sgpr"),
            &mut out,
            |xs, q, o| snapshot.predict_with(xs, q, D, o),
            |xs, q, o, out| online.predict_with_into(xs, q, D, o, out),
        );

        let mut svgp = Svgp::new(kernel, likelihood())
            .with_precision::<P>()
            .with_input_transform(StandardizeInput::new())
            .with_target_transform(StandardizeTarget::new())
            .factor(&x, N, D, &y, &z, M)
            .map_err(|(_, e)| e)
            .expect("factor");
        let mut params = vec![0.0; svgp.num_params()];
        svgp.get_params(&mut params).expect("params");
        // Move q(u) off the prior so the mean is not zero.
        for (idx, value) in params[svgp.num_params() - M * (M + 3) / 2..]
            .iter_mut()
            .enumerate()
            .take(M)
        {
            *value = 0.2 * (idx as f64 + 1.0);
        }
        svgp.set_params(&params).expect("set params");
        let snapshot = svgp.clone();
        check_same(
            &format!("{label} svgp"),
            &mut out,
            |xs, q, o| snapshot.predict_with(xs, q, D, o),
            |xs, q, o, out| svgp.predict_with_into(xs, q, D, o, out),
        );
    }
}

#[test]
fn predict_into_matches_predict_f64() {
    check_precision::<DoublePrecision>("f64");
}

#[test]
fn predict_into_matches_predict_f32() {
    check_precision::<SinglePrecision>("f32");
}

#[test]
fn predict_into_matches_predict_mixed() {
    check_precision::<MixedPrecision<PromoteStorage>>("mixed promote");
    check_precision::<MixedPrecision<ReevaluateKernel>>("mixed reevaluate");
}

#[test]
fn predict_into_rejects_bad_queries() {
    let x = points(N, 0.0);
    let y = targets(&x, N);
    let z = points(M, 0.5);
    let mut fitted = Sgpr::new(kernels()[0].clone(), likelihood())
        .with_optimizer(Fixed)
        .factor(&x, N, D, &y, &z, M)
        .map_err(|(_, e)| e)
        .expect("factor");
    let mut out = Prediction::default();
    assert!(matches!(
        fitted.predict_into(&[0.0; 3], 1, 3, &mut out),
        Err(GprError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        fitted.predict_into(&[0.0, f64::NAN], 1, D, &mut out),
        Err(GprError::NonFiniteInput)
    ));
    let mut svgp = Svgp::new(kernels()[0].clone(), likelihood())
        .factor(&x, N, D, &y, &z, M)
        .map_err(|(_, e)| e)
        .expect("factor");
    assert!(matches!(
        svgp.predict_into(&[0.0; 3], 1, 3, &mut out),
        Err(GprError::DimensionMismatch { .. })
    ));
}
