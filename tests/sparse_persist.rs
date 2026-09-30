//! Save and load of the sparse models (R5-8 / #286).
//!
//! A saved `FittedSgpr` / `FittedSvgp` loads back with the same predictions
//! to the bit: the factors are rebuilt from the same saved data. An
//! `OnlineSgpr` keeps its identifiers; its factors were updated rank-1 at a
//! time, so the refactor matches to rounding. This file uses only the public
//! API.

use std::num::{NonZeroU64, NonZeroUsize};
use std::path::PathBuf;

use gprx::kernel::{KernelSpec, RbfArdKernel};
use gprx::persist::{LoadedGpr, LoadedSgpr, LoadedSvgp, PersistRegistry};
use gprx::transform::{MinMaxInput, StandardizeInput, StandardizeTarget};
use gprx::{
    Adam, Fixed, FreeInducing, GaussianLikelihood, Gpr, GprError, JitterPolicy, KernelExp,
    MixedPrecision, PromoteStorage, ReevaluateKernel, Sgpr, SinglePrecision, Svgp,
};

mod common;
use common::assert_slice_close;

const N: usize = 14;
const M: usize = 5;
const D: usize = 2;
const Q: usize = 6;
/// A rank-1-updated online model against the refactor on load.
const ONLINE_TOL: f64 = 1e-10;

fn temp_dir(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "gprx-sparse-persist-{}-{label}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn points(rows: usize, shift: f64) -> Vec<f64> {
    let mut x = vec![0.0; rows * D];
    for i in 0..rows {
        let t = i as f64 + shift;
        x[i] = 10.0 + 0.6 * t;
        x[rows + i] = (0.7 * t).sin();
    }
    x
}

fn targets(x: &[f64], rows: usize) -> Vec<f64> {
    (0..rows)
        .map(|i| 30.0 + (0.9 * x[i]).cos() + 2.0 * x[rows + i])
        .collect()
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn kernel() -> KernelSpec {
    KernelSpec::from(RbfArdKernel::new(&[1.3, 0.8]).expect("valid"))
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn likelihood() -> GaussianLikelihood {
    GaussianLikelihood::new(0.05).expect("valid")
}

#[test]
fn fitted_sgpr_round_trips_to_the_bit() {
    let x = points(N, 0.0);
    let y = targets(&x, N);
    let z = points(M, 0.4);
    let xs = points(Q, 0.25);
    let jitter = JitterPolicy::fixed(1e-7).expect("valid");
    let fitted = Sgpr::new(kernel(), likelihood())
        .with_inducing(FreeInducing)
        .with_input_transform(MinMaxInput::new())
        .with_target_transform(StandardizeTarget::new())
        .with_jitter_policy(jitter)
        .with_math(KernelExp::FastApprox)
        .fit(&x, N, D, &y, &z, M)
        .map_err(|(_, e)| e)
        .expect("fit");
    let dir = temp_dir("sgpr");
    fitted.save(&dir).expect("save");
    let loaded = LoadedSgpr::load(&dir, &PersistRegistry::new()).expect("load");
    assert!(!loaded.is_online());
    assert_eq!((loaded.n(), loaded.m(), loaded.d()), (N, M, D));
    let LoadedSgpr::Double(model) = &loaded else {
        panic!("the default precision is double");
    };
    assert_eq!(model.x(), fitted.x());
    assert_eq!(model.y(), fitted.y());
    assert_eq!(model.z(), fitted.z());
    assert_eq!(model.jitter_policy(), jitter);
    assert_eq!(model.math(), KernelExp::FastApprox);
    let mut want = vec![0.0; fitted.num_params() - M * D];
    let mut got = vec![0.0; model.num_params()];
    let mut all = vec![0.0; fitted.num_params()];
    fitted.get_params(&mut all).expect("params");
    want.copy_from_slice(&all[..model.num_params()]);
    model.get_params(&mut got).expect("params");
    assert_eq!(got, want);
    assert_eq!(
        model.predict(&xs, Q, D).expect("predict"),
        fitted.predict(&xs, Q, D).expect("predict")
    );
    assert_eq!(
        loaded.predict(&xs, Q, D).expect("predict"),
        fitted.predict(&xs, Q, D).expect("predict")
    );
    assert_eq!(
        model.neg_log_marginal_likelihood().expect("nlml").to_bits(),
        fitted
            .neg_log_marginal_likelihood()
            .expect("nlml")
            .to_bits()
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn sgpr_precision_round_trip<P: gprx::GpScalar>(
    label: &str,
    unwrap: fn(LoadedSgpr) -> Option<gprx::FittedSgpr<Fixed, gprx::FixedInducing, P>>,
) where
    P::Refine: PartialEq + std::fmt::Debug,
{
    let x = points(N, 0.0);
    let y = targets(&x, N);
    let z = points(M, 0.4);
    let xs = points(Q, 0.25);
    let fitted = Sgpr::new(kernel(), likelihood())
        .with_optimizer(Fixed)
        .with_precision::<P>()
        .with_input_transform(StandardizeInput::new())
        .factor(&x, N, D, &y, &z, M)
        .map_err(|(_, e)| e)
        .expect("factor");
    let dir = temp_dir(label);
    fitted.save(&dir).expect("save");
    let model = unwrap(LoadedSgpr::load(&dir, &PersistRegistry::new()).expect("load"))
        .expect("the saved precision");
    assert_eq!(
        model.predict(&xs, Q, D).expect("predict"),
        fitted.predict(&xs, Q, D).expect("predict"),
        "{label}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn sgpr_round_trips_every_precision() {
    sgpr_precision_round_trip::<SinglePrecision>("sgpr-f32", |loaded| match loaded {
        LoadedSgpr::Single(model) => Some(model),
        _ => None,
    });
    sgpr_precision_round_trip::<MixedPrecision<PromoteStorage>>(
        "sgpr-mixed",
        |loaded| match loaded {
            LoadedSgpr::Mixed(model) => Some(model),
            _ => None,
        },
    );
    sgpr_precision_round_trip::<MixedPrecision<ReevaluateKernel>>("sgpr-reevaluate", |loaded| {
        match loaded {
            LoadedSgpr::Reevaluate(model) => Some(model),
            _ => None,
        }
    });
}

#[test]
fn online_sgpr_keeps_ids_and_predictions() {
    let x = points(N, 0.0);
    let y = targets(&x, N);
    let z = points(M, 0.4);
    let xs = points(Q, 0.25);
    let mut online = Sgpr::new(kernel(), likelihood())
        .with_optimizer(Fixed)
        .with_input_transform(StandardizeInput::new())
        .with_target_transform(StandardizeTarget::new())
        .factor(&x, N, D, &y, &z, M)
        .map_err(|(_, e)| e)
        .expect("factor")
        .into_online();
    online.insert(&[18.0, 0.3], 31.0).expect("insert");
    online.delete(online.point_ids()[1]).expect("delete");
    online
        .insert_inducing(&[14.0, -0.2])
        .expect("insert inducing");
    online
        .delete_inducing(online.inducing_ids()[0])
        .expect("delete inducing");
    let dir = temp_dir("online");
    online.save(&dir).expect("save");
    let loaded = LoadedSgpr::load(&dir, &PersistRegistry::new()).expect("load");
    assert!(loaded.is_online());
    let LoadedSgpr::OnlineDouble(mut model) = loaded else {
        panic!("an online double model");
    };
    assert_eq!(model.point_ids(), online.point_ids());
    assert_eq!(model.inducing_ids(), online.inducing_ids());
    assert_eq!(model.x(), online.x());
    assert_eq!(model.z(), online.z());
    let got = model.predict(&xs, Q, D).expect("predict");
    let want = online.predict(&xs, Q, D).expect("predict");
    assert_slice_close(&got.mean, &want.mean, ONLINE_TOL);
    assert_slice_close(&got.variance, &want.variance, ONLINE_TOL);
    // Both continue with the same identifiers and the fitted transforms of
    // the first training set.
    let id = online.insert(&[21.0, -0.5], 29.0).expect("insert");
    assert_eq!(model.insert(&[21.0, -0.5], 29.0).expect("insert"), id);
    let id = online
        .insert_inducing(&[20.0, 0.1])
        .expect("insert inducing");
    assert_eq!(
        model
            .insert_inducing(&[20.0, 0.1])
            .expect("insert inducing"),
        id
    );
    let got = model.predict(&xs, Q, D).expect("predict");
    let want = online.predict(&xs, Q, D).expect("predict");
    assert_slice_close(&got.mean, &want.mean, ONLINE_TOL);
    assert_slice_close(&got.variance, &want.variance, ONLINE_TOL);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn svgp_round_trips_to_the_bit() {
    let x = points(N, 0.0);
    let y = targets(&x, N);
    let z = points(M, 0.4);
    let xs = points(Q, 0.25);
    let adam = Adam::new()
        .with_batch_size(NonZeroUsize::new(5).expect("non-zero"))
        .with_epochs(NonZeroU64::new(10).expect("non-zero"))
        .with_seed(3);
    let fitted = Svgp::new(kernel(), likelihood())
        .with_optimizer(adam)
        .with_input_transform(StandardizeInput::new())
        .with_target_transform(StandardizeTarget::new())
        .fit(&x, N, D, &y, &z, M)
        .map_err(|(_, e)| e)
        .expect("fit");
    let dir = temp_dir("svgp");
    fitted.save(&dir).expect("save");
    let loaded = LoadedSvgp::load(&dir, &PersistRegistry::new()).expect("load");
    assert_eq!((loaded.n(), loaded.m(), loaded.d()), (N, M, D));
    let LoadedSvgp::Double(model) = &loaded else {
        panic!("the default precision is double");
    };
    let mut want = vec![0.0; fitted.num_params()];
    let mut got = vec![0.0; model.num_params()];
    fitted.get_params(&mut want).expect("params");
    model.get_params(&mut got).expect("params");
    assert_eq!(got, want);
    assert_eq!(
        model.predict(&xs, Q, D).expect("predict"),
        fitted.predict(&xs, Q, D).expect("predict")
    );
    assert_eq!(
        model.neg_elbo().expect("elbo").to_bits(),
        fitted.neg_elbo().expect("elbo").to_bits()
    );
    let single = Svgp::new(kernel(), likelihood())
        .with_precision::<SinglePrecision>()
        .factor(&x, N, D, &y, &z, M)
        .map_err(|(_, e)| e)
        .expect("factor");
    let dir_single = temp_dir("svgp-f32");
    single.save(&dir_single).expect("save");
    let LoadedSvgp::Single(model) =
        LoadedSvgp::load(&dir_single, &PersistRegistry::new()).expect("load")
    else {
        panic!("a single-precision model");
    };
    assert_eq!(
        model.predict(&xs, Q, D).expect("predict"),
        single.predict(&xs, Q, D).expect("predict")
    );
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&dir_single);
}

#[test]
fn each_loader_rejects_the_other_models() {
    let x = points(N, 0.0);
    let y = targets(&x, N);
    let z = points(M, 0.4);
    let sgpr_dir = temp_dir("reject-sgpr");
    Sgpr::new(kernel(), likelihood())
        .with_optimizer(Fixed)
        .factor(&x, N, D, &y, &z, M)
        .map_err(|(_, e)| e)
        .expect("factor")
        .save(&sgpr_dir)
        .expect("save");
    let svgp_dir = temp_dir("reject-svgp");
    Svgp::new(kernel(), likelihood())
        .factor(&x, N, D, &y, &z, M)
        .map_err(|(_, e)| e)
        .expect("factor")
        .save(&svgp_dir)
        .expect("save");
    let exact_dir = temp_dir("reject-exact");
    Gpr::new(kernel(), likelihood())
        .with_optimizer(Fixed)
        .factor(&x, N, D, &y)
        .map_err(|(_, e)| e)
        .expect("factor")
        .save(&exact_dir)
        .expect("save");
    let registry = PersistRegistry::new();
    let names_loader = |result: Result<(), GprError>, loader: &str| match result {
        Err(GprError::PersistFailed { reason }) => {
            assert!(reason.contains(loader), "{reason}");
        }
        other => panic!("expected PersistFailed naming {loader}, got {other:?}"),
    };
    names_loader(
        LoadedGpr::load(&sgpr_dir, &registry).map(drop),
        "LoadedSgpr::load",
    );
    names_loader(
        LoadedSvgp::load(&sgpr_dir, &registry).map(drop),
        "LoadedSgpr::load",
    );
    names_loader(
        LoadedSgpr::load(&svgp_dir, &registry).map(drop),
        "LoadedSvgp::load",
    );
    names_loader(
        LoadedSgpr::load(&exact_dir, &registry).map(drop),
        "LoadedGpr::load",
    );
    assert!(LoadedGpr::load(&exact_dir, &registry).is_ok());
    for dir in [sgpr_dir, svgp_dir, exact_dir] {
        let _ = std::fs::remove_dir_all(dir);
    }
}
