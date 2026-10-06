//! Save and load of models on supplied squared distances. Public API only.

use gprx::kernel::{
    ArdDistance, DistanceKernel, DistanceOnly, DistanceSlot, KernelSpec, RbfArdKernel, RbfKernel,
    ScalarDistance, WithPoints,
};
use gprx::{
    DoublePrecision, FittedGpr, FittedSgpr, FittedSvgp, Fixed, FixedInducing, GaussianLikelihood,
    Gpr, GprError, LoadedGpr, LoadedSgpr, OnlineGpr, PersistErrorKind, PersistRegistry, Sgpr,
    SinglePrecision, Svgp,
};
use std::path::PathBuf;

const N: usize = 5;

fn temp_dir(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "gprx-distance-persist-{}-{label}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn coord(k: usize, rows: usize, offset: f64) -> Vec<f64> {
    (0..rows)
        .map(|i| ((i as f64 + offset) * (0.41 + 0.17 * k as f64)).sin() * (1.0 + 0.5 * k as f64))
        .collect()
}

fn sq(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(a.len() * b.len());
    for bj in b {
        for ai in a {
            out.push((ai - bj) * (ai - bj));
        }
    }
    out
}

fn targets() -> Vec<f64> {
    (0..N).map(|i| (i as f64 * 0.7).cos()).collect()
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn lik() -> GaussianLikelihood {
    GaussianLikelihood::new(0.05).expect("noise")
}

fn reg() -> PersistRegistry {
    PersistRegistry::new()
}

fn wrong_model<T: std::fmt::Debug>(result: Result<T, GprError>) {
    assert!(
        matches!(
            result,
            Err(GprError::PersistFailed {
                kind: PersistErrorKind::WrongModel,
                ..
            })
        ),
        "{result:?}"
    );
}

/// The one scalar slot of a loaded model.
fn scalar(slots: &[DistanceSlot]) -> Option<ScalarDistance> {
    match slots {
        [DistanceSlot::Scalar(slot)] => Some(*slot),
        _ => None,
    }
}

#[test]
fn exact_roundtrip_with_and_without_the_factor() {
    let c = coord(0, N, 0.0);
    let q = coord(0, 2, 0.5);
    let image = ScalarDistance::new();
    let fitted = Gpr::new(image.kernel(RbfKernel::new(0.9).expect("ell")), lik())
        .with_optimizer(Fixed)
        .factor([image.from_vec(sq(&c, &c))], N, &targets())
        .expect("fit");
    let cross = sq(&c, &q);
    let expect = fitted.predict([image.borrow(&cross)], 2).expect("predict");
    for with_factor in [false, true] {
        let dir = temp_dir(&format!("exact-{with_factor}"));
        if with_factor {
            fitted.save_with_factor(&dir).expect("save");
        } else {
            fitted.save(&dir).expect("save");
        }
        type Model = FittedGpr<Fixed, DoublePrecision, DistanceKernel<DistanceOnly>>;
        let loaded = Model::load(&dir, &reg()).expect("load");
        let slot = scalar(&loaded.slots()).expect("one scalar slot");
        let got = loaded.predict([slot.borrow(&cross)], 2).expect("predict");
        assert_eq!(got, expect);
        // The coordinate loader, the other marker, another precision, and
        // the online loader refuse the directory.
        wrong_model(LoadedGpr::load(&dir, &reg()));
        wrong_model(FittedGpr::<
            Fixed,
            DoublePrecision,
            DistanceKernel<WithPoints>,
        >::load(&dir, &reg()));
        wrong_model(FittedGpr::<
            Fixed,
            SinglePrecision,
            DistanceKernel<DistanceOnly>,
        >::load(&dir, &reg()));
        wrong_model(OnlineGpr::<
            Fixed,
            DoublePrecision,
            DistanceKernel<DistanceOnly>,
        >::load(&dir, &reg()));
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[test]
fn a_coordinate_save_is_not_read_as_a_distance_model() {
    let c = coord(0, N, 0.0);
    let fitted = Gpr::new(KernelSpec::from(RbfKernel::new(0.9).expect("ell")), lik())
        .with_optimizer(Fixed)
        .factor(&c, N, 1, &targets())
        .expect("fit");
    let dir = temp_dir("coords");
    fitted.save(&dir).expect("save");
    wrong_model(FittedGpr::<
        Fixed,
        DoublePrecision,
        DistanceKernel<WithPoints>,
    >::load(&dir, &reg()));
    wrong_model(FittedGpr::<
        Fixed,
        DoublePrecision,
        DistanceKernel<DistanceOnly>,
    >::load(&dir, &reg()));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn online_ard_with_points_roundtrip_keeps_ids_and_inserts() {
    let cols: Vec<Vec<f64>> = (0..3).map(|k| coord(k, N + 1, 0.0)).collect();
    let train = |k: usize| cols[k][..N].to_vec();
    let bands = ArdDistance::new(2).expect("dims");
    let kernel = bands
        .kernel(RbfArdKernel::new(&[0.8, 1.4]).expect("ell"))
        .expect("dims")
        * KernelSpec::from(RbfKernel::new(1.1).expect("ell"));
    let y = targets();
    let mut online = Gpr::new(kernel, lik())
        .with_optimizer(Fixed)
        .factor(
            [bands.from_vecs(vec![sq(&train(0), &train(0)), sq(&train(1), &train(1))])],
            N,
            &train(2),
            1,
            &y,
        )
        .expect("fit")
        .into_online()
        .expect("online");
    let dir = temp_dir("online");
    online.save(&dir).expect("save");
    type Model = OnlineGpr<Fixed, DoublePrecision, DistanceKernel<WithPoints>>;
    let mut loaded = Model::load(&dir, &reg()).expect("load");
    let DistanceSlot::Ard(slot) = loaded.slots()[0] else {
        panic!("one ARD slot");
    };
    let new = |k: usize| sq(&train(k), &cols[k][N..]);
    let x_new = [cols[2][N]];
    let a = online
        .insert([bands.from_vecs(vec![new(0), new(1)])], &x_new, 0.3)
        .expect("insert");
    let b = loaded
        .insert([slot.from_vecs(vec![new(0), new(1)])], &x_new, 0.3)
        .expect("insert");
    assert_eq!(a, b);
    let q = (0..3).map(|k| coord(k, 2, 0.5)).collect::<Vec<_>>();
    let all = |k: usize| cols[k].clone();
    let cross = |k: usize| sq(&all(k), &q[k]);
    let expect = online
        .predict([bands.from_vecs(vec![cross(0), cross(1)])], &q[2], 2, 1)
        .expect("predict");
    let got = loaded
        .predict([slot.from_vecs(vec![cross(0), cross(1)])], &q[2], 2, 1)
        .expect("predict");
    for (g, e) in got.mean.iter().zip(&expect.mean) {
        assert!((g - e).abs() < 1e-10, "{g} vs {e}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn sparse_roundtrip_keeps_the_inducing_points() {
    let c = coord(0, N, 0.0);
    let q = coord(0, 2, 0.5);
    let image = ScalarDistance::new();
    let rbf = RbfKernel::new(0.9).expect("ell");
    let cross = sq(&c, &q);

    let sgpr = Sgpr::new(image.kernel(rbf), lik())
        .with_optimizer(Fixed)
        .factor([image.from_vec(sq(&c, &c))], N, &targets(), &[1, 3])
        .expect("sgpr");
    let dir = temp_dir("sgpr");
    sgpr.save(&dir).expect("save");
    type SgprModel =
        FittedSgpr<Fixed, FixedInducing, DoublePrecision, DistanceKernel<DistanceOnly>>;
    let loaded = SgprModel::load(&dir, &reg()).expect("load");
    assert_eq!(loaded.inducing(), &[1, 3]);
    let slot = scalar(&loaded.slots()).expect("one scalar slot");
    assert_eq!(
        loaded.predict([slot.borrow(&cross)], 2).expect("predict"),
        sgpr.predict([image.borrow(&cross)], 2).expect("predict")
    );
    wrong_model(LoadedSgpr::load(&dir, &reg()));
    wrong_model(FittedSvgp::<DoublePrecision, DistanceKernel<DistanceOnly>>::load(&dir, &reg()));
    let _ = std::fs::remove_dir_all(&dir);

    let svgp = Svgp::new(image.kernel(rbf), lik())
        .with_precision::<SinglePrecision>()
        .factor([image.from_vec(sq(&c, &c))], N, &targets(), &[0, 4])
        .expect("svgp");
    let dir = temp_dir("svgp");
    svgp.save(&dir).expect("save");
    let loaded = FittedSvgp::<SinglePrecision, DistanceKernel<DistanceOnly>>::load(&dir, &reg())
        .expect("load");
    let slot = scalar(&loaded.slots()).expect("one scalar slot");
    assert_eq!(
        loaded.predict([slot.borrow(&cross)], 2).expect("predict"),
        svgp.predict([image.borrow(&cross)], 2).expect("predict")
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_distance_model_is_written_as_version_two_and_a_coordinate_model_as_one() {
    let version = |dir: &PathBuf| -> u64 {
        let bytes = std::fs::read(dir.join("config.json")).expect("config");
        let value: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
        value["format_version"].as_u64().expect("version")
    };
    let c = coord(0, N, 0.0);
    let image = ScalarDistance::new();
    let rbf = RbfKernel::new(0.9).expect("ell");
    let dist = Gpr::new(image.kernel(rbf), lik())
        .with_optimizer(Fixed)
        .factor([image.from_vec(sq(&c, &c))], N, &targets())
        .expect("distances");
    let dir = temp_dir("version-dist");
    dist.save(&dir).expect("save");
    assert_eq!(version(&dir), 2);
    let _ = std::fs::remove_dir_all(&dir);
    let coords = Gpr::new(KernelSpec::from(rbf), lik())
        .with_optimizer(Fixed)
        .factor(&c, N, 1, &targets())
        .expect("coords");
    let dir = temp_dir("version-coords");
    coords.save(&dir).expect("save");
    assert_eq!(version(&dir), 1);
    assert!(LoadedGpr::load(&dir, &reg()).is_ok());
    let _ = std::fs::remove_dir_all(&dir);
    let sparse = Sgpr::new(image.kernel(rbf), lik())
        .with_optimizer(Fixed)
        .factor([image.from_vec(sq(&c, &c))], N, &targets(), &[1, 3])
        .expect("sgpr");
    let dir = temp_dir("version-sparse");
    sparse.save(&dir).expect("save");
    assert_eq!(version(&dir), 2);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_sparse_save_whose_z_is_not_the_inducing_rows_is_refused() {
    let (c0, c1) = (coord(0, N, 0.0), coord(1, N, 0.0));
    let x: Vec<f64> = c0.iter().chain(&c1).copied().collect();
    let image = ScalarDistance::new();
    let kernel = image.kernel(RbfKernel::new(0.9).expect("ell"))
        * KernelSpec::from(RbfKernel::new(1.1).expect("ell"));
    let sgpr = Sgpr::new(kernel, lik())
        .with_optimizer(Fixed)
        .factor(
            [image.from_vec(sq(&c0, &c0))],
            N,
            &x,
            2,
            &targets(),
            &[1, 3],
        )
        .expect("sgpr");
    let dir = temp_dir("sgpr-inducing");
    sgpr.save(&dir).expect("save");
    type Model = FittedSgpr<Fixed, FixedInducing, DoublePrecision, DistanceKernel<WithPoints>>;
    assert_eq!(Model::load(&dir, &reg()).expect("load").inducing(), &[1, 3]);
    // Point `inducing` at other rows: `z` no longer matches them.
    let path = dir.join("config.json");
    let mut value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).expect("config")).expect("json");
    value["inducing"] = serde_json::json!([0, 3]);
    std::fs::write(&path, serde_json::to_vec(&value).expect("json")).expect("write");
    assert!(matches!(
        Model::load(&dir, &reg()),
        Err(GprError::PersistFailed {
            kind: PersistErrorKind::Config,
            ..
        })
    ));
    let _ = std::fs::remove_dir_all(&dir);
}
