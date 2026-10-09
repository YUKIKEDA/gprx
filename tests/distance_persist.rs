//! Save and load of models on supplied squared distances: a loaded model
//! predicts as the saved one did, on its own slots, at every precision and
//! factor kind. Public API only.

use gprx::kernel::{
    ArdDistance, ConstantKernel, DistanceKernel, DistanceOnly, DistanceSlot, DistanceSource,
    KernelScalar, KernelSpec, MaternKernel, MaternNu, RationalQuadraticArdKernel, RbfArdKernel,
    RbfKernel, ScalarDistance, WithPoints,
};
use gprx::persist::{
    LoadedDistanceGpr, LoadedDistanceSgpr, LoadedDistanceSvgp, LoadedGpr, LoadedSgpr, LoadedSvgp,
    PersistRegistry,
};
use gprx::{
    DoublePrecision, FittedGpr, Fixed, GaussianLikelihood, GpScalar, Gpr, GprError, MixedPrecision,
    PersistErrorKind, Prediction, ReevaluateKernel, Sgpr, SinglePrecision, Svgp,
};

const N: usize = 7;
const M: usize = 3;
const DIMS: usize = 2;

/// Coordinate `k` of training point `i` (queries are offset by a half).
fn at(k: usize, i: f64) -> f64 {
    (i * (0.41 + 0.17 * k as f64)).sin() * (1.0 + 0.5 * k as f64)
}

fn coords(k: usize, idx: &[f64]) -> Vec<f64> {
    idx.iter().map(|&i| at(k, i)).collect()
}

fn train_idx(n: usize) -> Vec<f64> {
    (0..n).map(|i| i as f64).collect()
}

fn query_idx() -> Vec<f64> {
    (0..M).map(|q| q as f64 + 0.5).collect()
}

fn targets(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i as f64 * 0.7).cos()).collect()
}

/// Column-major `rows × cols` squared differences of dimension `k`.
fn sq(k: usize, rows: &[f64], cols: &[f64]) -> Vec<f64> {
    let (r, c) = (coords(k, rows), coords(k, cols));
    let mut out = Vec::with_capacity(r.len() * c.len());
    for cv in &c {
        for rv in &r {
            out.push((rv - cv) * (rv - cv));
        }
    }
    out
}

fn summed(rows: &[f64], cols: &[f64]) -> Vec<f64> {
    let blocks: Vec<_> = (0..DIMS).map(|k| sq(k, rows, cols)).collect();
    (0..blocks[0].len())
        .map(|i| blocks.iter().map(|b| b[i]).sum())
        .collect()
}

fn ard(rows: &[f64], cols: &[f64]) -> Vec<Vec<f64>> {
    (0..DIMS).map(|k| sq(k, rows, cols)).collect()
}

/// The sources of `slots` between point sets `rows` and `cols`: the summed
/// squares for a scalar slot, the per-dimension blocks for an ARD slot. A
/// slot of another kind gets none, which the model refuses.
fn sources(slots: &[DistanceSlot], rows: &[f64], cols: &[f64]) -> Vec<DistanceSource<'static>> {
    slots
        .iter()
        .filter_map(|slot| match slot {
            DistanceSlot::Scalar(s) => Some(s.from_vec(summed(rows, cols))),
            DistanceSlot::Ard(a) => Some(a.from_vecs(ard(rows, cols))),
            _ => None,
        })
        .collect()
}

/// A kernel of every slot shape: a scalar Matérn times a constant, plus an
/// ARD rational quadratic.
fn two_slots() -> Result<DistanceKernel, GprError> {
    let image = ScalarDistance::new();
    let (_bands, rq) = ArdDistance::from_leaf(RationalQuadraticArdKernel::new(&[0.9, 1.4], 1.5)?);
    Ok(
        ConstantKernel::new(1.3)? * image.kernel(MaternKernel::new(1.1, MaternNu::FiveHalves)?)
            + rq,
    )
}

fn scalar_only() -> Result<DistanceKernel, GprError> {
    Ok(ScalarDistance::new().kernel(RbfKernel::new(1.2)?))
}

fn ard_only() -> Result<DistanceKernel, GprError> {
    Ok(ArdDistance::from_leaf(RbfArdKernel::new(&[0.8, 1.3])?).1)
}

fn temp_dir(label: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "gprx-distance-persist-{label}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

type Widened = (Vec<f64>, Vec<f64>);

fn widen<T: KernelScalar>(pred: Prediction<T>) -> Widened {
    (
        pred.mean.iter().map(|v| v.to_f64()).collect(),
        pred.variance.iter().map(|v| v.to_f64()).collect(),
    )
}

fn assert_same(label: &str, got: &Widened, want: &Widened, tol: f64) {
    for (g, w) in got.0.iter().zip(&want.0).chain(got.1.iter().zip(&want.1)) {
        assert!(
            (g - w).abs() <= tol * (1.0 + w.abs()),
            "{label}: loaded {g}, saved {w}"
        );
    }
}

/// Saves `fitted` (and its online model after an insert and a delete, so an
/// ARD slot is laid out in row runs) with and without the factor, loads
/// each, and checks the predictions at the queries: within `tol`, or within
/// `refactored` for an online model saved without its factor (load factors
/// it afresh, which rounds in the storage scalar unlike the updates did).
fn round_trip<P: GpScalar>(
    label: &str,
    kernel: DistanceKernel,
    tol: f64,
    refactored: f64,
) -> Result<(), GprError> {
    let rows = train_idx(N);
    let y = targets(N);
    let slots = kernel.slots();
    let fitted: FittedGpr<Fixed, P, DistanceKernel> =
        Gpr::new(kernel, GaussianLikelihood::new(0.1)?)
            .with_optimizer(Fixed)
            .with_precision::<P>()
            .factor(sources(&slots, &rows, &rows), N, &y)?;
    let mut online = fitted.clone().into_online()?;
    // One more point, then the first one out: the live points are 1..=N.
    let new = [N as f64];
    online.insert(sources(&slots, &rows, &new), (N as f64 * 0.7).cos())?;
    let first = online.point_ids()[0];
    online.delete(first)?;
    let live: Vec<f64> = (1..=N).map(|i| i as f64).collect();
    let q = query_idx();
    let want_fitted = widen(fitted.predict(sources(&slots, &rows, &q), M)?);
    let want_online = widen(online.predict(sources(&slots, &live, &q), M)?);
    for with_factor in [false, true] {
        for is_online in [false, true] {
            let dir = temp_dir(&format!("{label}-{with_factor}-{is_online}"));
            match (is_online, with_factor) {
                (false, false) => fitted.save(&dir)?,
                (false, true) => fitted.save_with_factor(&dir)?,
                (true, false) => online.save(&dir)?,
                (true, true) => online.save_with_factor(&dir)?,
            }
            let loaded = LoadedDistanceGpr::<DistanceOnly>::load(&dir, &PersistRegistry::new())?;
            assert_eq!(loaded.is_online(), is_online, "{label}");
            let loaded_slots = loaded.slots();
            assert_eq!(loaded_slots.len(), slots.len(), "{label}");
            assert_ne!(loaded_slots, slots, "{label}: the loaded slots are new");
            assert_eq!(loaded.to_kernel().slots(), loaded_slots, "{label}");
            let (rows, want) = if is_online {
                (&live, &want_online)
            } else {
                (&rows, &want_fitted)
            };
            assert_eq!(loaded.n(), rows.len(), "{label}");
            let got = widen(loaded.predict(sources(&loaded_slots, rows, &q), M)?);
            assert_same(
                &format!("{label} factor={with_factor} online={is_online}"),
                &got,
                want,
                if is_online && !with_factor {
                    refactored
                } else {
                    tol
                },
            );
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
    Ok(())
}

type KernelFn = fn() -> Result<DistanceKernel, GprError>;

#[test]
fn every_precision_and_slot_shape_round_trips() -> Result<(), GprError> {
    for (label, kernel) in [
        ("scalar", scalar_only as KernelFn),
        ("ard", ard_only),
        ("two", two_slots),
    ] {
        round_trip::<DoublePrecision>(&format!("{label}-double"), kernel()?, 1e-12, 1e-10)?;
        round_trip::<SinglePrecision>(&format!("{label}-single"), kernel()?, 1e-5, 1e-5)?;
        round_trip::<MixedPrecision>(&format!("{label}-mixed"), kernel()?, 1e-10, 1e-5)?;
        round_trip::<MixedPrecision<ReevaluateKernel>>(
            &format!("{label}-reevaluate"),
            kernel()?,
            1e-10,
            1e-5,
        )?;
    }
    Ok(())
}

/// A double-precision model saved with its factor loads to the same bits.
#[test]
fn saved_factor_predicts_the_same_bits() -> Result<(), GprError> {
    let rows = train_idx(N);
    let kernel = two_slots()?;
    let slots = kernel.slots();
    let fitted = Gpr::new(kernel, GaussianLikelihood::new(0.1)?)
        .with_optimizer(Fixed)
        .factor(sources(&slots, &rows, &rows), N, &targets(N))?;
    let dir = temp_dir("bits");
    fitted.save_with_factor(&dir)?;
    let loaded = LoadedDistanceGpr::<DistanceOnly>::load(&dir, &PersistRegistry::new())?;
    let q = query_idx();
    let want = fitted.predict(sources(&slots, &rows, &q), M)?;
    let got = loaded.predict(sources(&loaded.slots(), &rows, &q), M)?;
    for (g, w) in got
        .mean
        .iter()
        .zip(&want.mean)
        .chain(got.variance.iter().zip(&want.variance))
    {
        assert_eq!(g.to_bits(), w.to_bits());
    }
    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}

fn is_wrong_model<T>(result: &Result<T, GprError>) -> bool {
    matches!(
        result,
        Err(GprError::PersistFailed {
            kind: PersistErrorKind::WrongModel,
            ..
        })
    )
}

#[test]
fn with_points_round_trips() -> Result<(), GprError> {
    let rows = train_idx(N);
    let image = ScalarDistance::new();
    let kernel = image.kernel(RbfKernel::new(1.2)?) * KernelSpec::from(RbfKernel::new(0.7)?);
    let slots = kernel.slots();
    let x = coords(0, &rows);
    let fitted = Gpr::new(kernel, GaussianLikelihood::new(0.1)?)
        .with_optimizer(Fixed)
        .factor(sources(&slots, &rows, &rows), N, &x, 1, &targets(N))?;
    let q = query_idx();
    let xq = coords(0, &q);
    let want = widen(fitted.predict(sources(&slots, &rows, &q), &xq, M, 1)?);
    let dir = temp_dir("points");
    fitted.save(&dir)?;
    let loaded = LoadedDistanceGpr::<WithPoints>::load(&dir, &PersistRegistry::new())?;
    assert_eq!((loaded.n(), loaded.d()), (N, 1));
    let got = widen(loaded.predict(sources(&loaded.slots(), &rows, &q), &xq, M, 1)?);
    assert_same("points", &got, &want, 1e-12);
    // The marker is part of the model: the other loaders refuse the file.
    let registry = PersistRegistry::new();
    assert!(is_wrong_model(&LoadedDistanceGpr::<DistanceOnly>::load(
        &dir, &registry
    )));
    assert!(is_wrong_model(&LoadedGpr::load(&dir, &registry)));
    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}

#[test]
fn coordinate_and_distance_files_refuse_the_other_loader() -> Result<(), GprError> {
    let rows = train_idx(N);
    let kernel = scalar_only()?;
    let slots = kernel.slots();
    let fitted = Gpr::new(kernel, GaussianLikelihood::new(0.1)?)
        .with_optimizer(Fixed)
        .factor(sources(&slots, &rows, &rows), N, &targets(N))?;
    let registry = PersistRegistry::new();
    let dir = temp_dir("mixup-distance");
    fitted.save(&dir)?;
    assert!(is_wrong_model(&LoadedGpr::load(&dir, &registry)));
    assert!(is_wrong_model(&gprx::persist::LoadedSgpr::load(
        &dir, &registry
    )));
    assert!(is_wrong_model(&LoadedDistanceGpr::<WithPoints>::load(
        &dir, &registry
    )));
    let _ = std::fs::remove_dir_all(&dir);

    let coordinate = Gpr::new(
        KernelSpec::from(RbfKernel::new(1.0)?),
        GaussianLikelihood::new(0.1)?,
    )
    .fit(&coords(0, &rows), N, 1, &targets(N))?;
    let dir = temp_dir("mixup-coordinate");
    coordinate.save(&dir)?;
    assert!(is_wrong_model(&LoadedDistanceGpr::<DistanceOnly>::load(
        &dir, &registry
    )));
    assert!(is_wrong_model(&LoadedDistanceGpr::<WithPoints>::load(
        &dir, &registry
    )));
    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}

/// The training points that start as inducing points.
const INDUCING: [usize; 3] = [0, 3, 5];

/// The values of `points` at the indices `at`.
fn pick(points: &[f64], at: &[usize]) -> Vec<f64> {
    at.iter().map(|&i| points[i]).collect()
}

/// Saves an SGPR, its online model after a point insert, an inducing
/// insert, and a point delete, and an SVGP of the same kernel; loads each
/// and checks the predictions at the queries. Load factors afresh, so an
/// online model after updates is checked within `refactored`.
fn sparse_round_trip<P: GpScalar>(
    label: &str,
    kernel: fn() -> Result<DistanceKernel, GprError>,
    tol: f64,
    refactored: f64,
) -> Result<(), GprError> {
    let rows = train_idx(N);
    let y = targets(N);
    let registry = PersistRegistry::new();
    let q = query_idx();
    let first = kernel()?;
    let slots = first.slots();
    let z = pick(&rows, &INDUCING);
    let fitted = Sgpr::new(first, GaussianLikelihood::new(0.1)?)
        .with_optimizer(Fixed)
        .with_precision::<P>()
        .factor(sources(&slots, &rows, &z), N, &y, &INDUCING)
        .map_err(|(_, e)| e)?;
    let want = widen(fitted.predict(sources(&slots, &z, &q), M)?);
    let dir = temp_dir(&format!("sgpr-{label}"));
    fitted.save(&dir)?;
    let loaded = LoadedDistanceSgpr::<DistanceOnly>::load(&dir, &registry)?;
    assert!(!loaded.is_online());
    assert_eq!((loaded.n(), loaded.m()), (N, INDUCING.len()));
    assert_eq!(loaded.inducing(), &INDUCING);
    assert_eq!(loaded.to_kernel().slots(), loaded.slots());
    let got = widen(loaded.predict(sources(&loaded.slots(), &z, &q), M)?);
    assert_same(&format!("sgpr {label}"), &got, &want, tol);
    assert!(is_wrong_model(&LoadedSgpr::load(&dir, &registry)));
    assert!(is_wrong_model(&LoadedDistanceSvgp::<DistanceOnly>::load(
        &dir, &registry
    )));
    assert!(is_wrong_model(&LoadedDistanceGpr::<DistanceOnly>::load(
        &dir, &registry
    )));
    assert!(is_wrong_model(&LoadedDistanceSgpr::<WithPoints>::load(
        &dir, &registry
    )));
    let _ = std::fs::remove_dir_all(&dir);

    let mut online = fitted.into_online();
    let mut live = rows.clone();
    let new = N as f64;
    online.insert(sources(&slots, &z, &[new]), (new * 0.7).cos())?;
    live.push(new);
    let point = online.point_ids()[1];
    online.insert_inducing(point, sources(&slots, &live, &[live[1]]))?;
    let gone = online.point_ids()[2];
    online.delete(gone)?;
    live.remove(2);
    let z_live = pick(&live, online.inducing());
    let want = widen(online.predict(sources(&slots, &z_live, &q), M)?);
    let dir = temp_dir(&format!("online-sgpr-{label}"));
    online.save(&dir)?;
    let loaded = LoadedDistanceSgpr::<DistanceOnly>::load(&dir, &registry)?;
    assert!(loaded.is_online());
    assert_eq!(loaded.n(), live.len());
    assert_eq!(loaded.inducing(), online.inducing());
    let got = widen(loaded.predict(sources(&loaded.slots(), &z_live, &q), M)?);
    assert_same(&format!("online sgpr {label}"), &got, &want, refactored);
    let (LoadedDistanceSgpr::OnlineDouble(_)
    | LoadedDistanceSgpr::OnlineSingle(_)
    | LoadedDistanceSgpr::OnlineMixed(_)
    | LoadedDistanceSgpr::OnlineReevaluate(_)) = loaded
    else {
        return Err(GprError::InvalidConfig {
            reason: format!("{label}: an online file loaded a fitted model"),
        });
    };
    let _ = std::fs::remove_dir_all(&dir);

    let second = kernel()?;
    let slots = second.slots();
    let svgp = Svgp::new(second, GaussianLikelihood::new(0.1)?)
        .with_precision::<P>()
        .factor(sources(&slots, &rows, &z), N, &y, &INDUCING)
        .map_err(|(_, e)| e)?;
    let want = widen(svgp.predict(sources(&slots, &z, &q), M)?);
    let dir = temp_dir(&format!("svgp-{label}"));
    svgp.save(&dir)?;
    let loaded = LoadedDistanceSvgp::<DistanceOnly>::load(&dir, &registry)?;
    assert_eq!((loaded.n(), loaded.m()), (N, INDUCING.len()));
    assert_eq!(loaded.inducing(), &INDUCING);
    assert_eq!(loaded.to_kernel().slots(), loaded.slots());
    let got = widen(loaded.predict(sources(&loaded.slots(), &z, &q), M)?);
    assert_same(&format!("svgp {label}"), &got, &want, tol);
    assert!(is_wrong_model(&LoadedSvgp::load(&dir, &registry)));
    assert!(is_wrong_model(&LoadedDistanceSgpr::<DistanceOnly>::load(
        &dir, &registry
    )));
    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}

#[test]
fn sparse_models_round_trip_at_every_precision() -> Result<(), GprError> {
    for (label, kernel) in [
        ("scalar", scalar_only as KernelFn),
        ("ard", ard_only),
        ("two", two_slots),
    ] {
        sparse_round_trip::<DoublePrecision>(&format!("{label}-double"), kernel, 1e-12, 1e-9)?;
        sparse_round_trip::<SinglePrecision>(&format!("{label}-single"), kernel, 1e-5, 1e-4)?;
        sparse_round_trip::<MixedPrecision>(&format!("{label}-mixed"), kernel, 1e-9, 1e-4)?;
        sparse_round_trip::<MixedPrecision<ReevaluateKernel>>(
            &format!("{label}-reevaluate"),
            kernel,
            1e-9,
            1e-4,
        )?;
    }
    Ok(())
}

#[test]
fn sparse_with_points_round_trips() -> Result<(), GprError> {
    let rows = train_idx(N);
    let z = pick(&rows, &INDUCING);
    let q = query_idx();
    let (x, xq) = (coords(0, &rows), coords(0, &q));
    let kernel = || -> Result<DistanceKernel<WithPoints>, GprError> {
        Ok(ScalarDistance::new().kernel(RbfKernel::new(1.2)?)
            * KernelSpec::from(RbfKernel::new(0.7)?))
    };
    let registry = PersistRegistry::new();
    let first = kernel()?;
    let slots = first.slots();
    let fitted = Sgpr::new(first, GaussianLikelihood::new(0.1)?)
        .with_optimizer(Fixed)
        .factor(sources(&slots, &rows, &z), N, &x, 1, &targets(N), &INDUCING)
        .map_err(|(_, e)| e)?;
    let want = widen(fitted.predict(sources(&slots, &z, &q), &xq, M, 1)?);
    let dir = temp_dir("sgpr-points");
    fitted.save(&dir)?;
    let loaded = LoadedDistanceSgpr::<WithPoints>::load(&dir, &registry)?;
    assert_eq!(loaded.d(), 1);
    let got = widen(loaded.predict(sources(&loaded.slots(), &z, &q), &xq, M, 1)?);
    assert_same("sgpr points", &got, &want, 1e-12);
    assert!(is_wrong_model(&LoadedDistanceSgpr::<DistanceOnly>::load(
        &dir, &registry
    )));
    let _ = std::fs::remove_dir_all(&dir);

    let second = kernel()?;
    let slots = second.slots();
    let svgp = Svgp::new(second, GaussianLikelihood::new(0.1)?)
        .factor(sources(&slots, &rows, &z), N, &x, 1, &targets(N), &INDUCING)
        .map_err(|(_, e)| e)?;
    let want = widen(svgp.predict(sources(&slots, &z, &q), &xq, M, 1)?);
    let dir = temp_dir("svgp-points");
    svgp.save(&dir)?;
    let loaded = LoadedDistanceSvgp::<WithPoints>::load(&dir, &registry)?;
    assert_eq!(loaded.d(), 1);
    let got = widen(loaded.predict(sources(&loaded.slots(), &z, &q), &xq, M, 1)?);
    assert_same("svgp points", &got, &want, 1e-12);
    assert!(is_wrong_model(&LoadedDistanceSvgp::<DistanceOnly>::load(
        &dir, &registry
    )));
    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}
