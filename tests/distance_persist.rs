//! Save and load of models on supplied squared distances: a loaded model
//! predicts as the saved one did, on its own slots, at every precision and
//! factor kind. Public API only.

mod common;

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
    PersistErrorKind, Prediction, ReevaluateKernel, Sgpr, SinglePrecision, SlotErrorKind, Svgp,
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
    assert_eq!(got.0.len(), want.0.len(), "{label}: mean length");
    assert_eq!(got.1.len(), want.1.len(), "{label}: variance length");
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
    // The slots of the model before the save are not the loaded model's.
    assert_eq!(
        loaded.predict(sources(&slots, &rows, &q), M).map(drop),
        Err(GprError::DistanceSlot {
            kind: SlotErrorKind::NotRead,
            slot: None
        })
    );
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
    let z_live = pick(&live, &common::inducing_places(&online));
    let want = widen(online.predict(sources(&slots, &z_live, &q), M)?);
    let dir = temp_dir(&format!("online-sgpr-{label}"));
    online.save(&dir)?;
    let loaded = LoadedDistanceSgpr::<DistanceOnly>::load(&dir, &registry)?;
    assert!(loaded.is_online());
    assert_eq!(loaded.n(), live.len());
    assert_eq!(loaded.inducing(), common::inducing_places(&online));
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

// ---- Files edited by hand: load refuses them with the right error. ----

/// One tensor of a file: its scalar, shape, and bytes.
type Tensor = (safetensors::Dtype, Vec<usize>, Vec<u8>);

/// The tensors of `dir/model.safetensors`, by name.
fn read_tensors(dir: &std::path::Path) -> Vec<(String, Tensor)> {
    let bytes = std::fs::read(dir.join("model.safetensors")).unwrap_or_default();
    let Ok(file) = safetensors::SafeTensors::deserialize(&bytes) else {
        return Vec::new();
    };
    let mut out: Vec<(String, Tensor)> = file
        .tensors()
        .into_iter()
        .map(|(name, view)| {
            (
                name,
                (view.dtype(), view.shape().to_vec(), view.data().to_vec()),
            )
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// Writes `tensors` as `dir/model.safetensors`.
fn write_tensors(dir: &std::path::Path, tensors: &[(String, Tensor)]) -> Result<(), GprError> {
    let views = tensors
        .iter()
        .map(|(name, (dtype, shape, data))| {
            safetensors::tensor::TensorView::new(*dtype, shape.clone(), data)
                .map(|view| (name.clone(), view))
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| GprError::InvalidConfig {
            reason: format!("test tensor: {err}"),
        })?;
    let bytes = safetensors::serialize(views, None).map_err(|err| GprError::InvalidConfig {
        reason: format!("test serialize: {err}"),
    })?;
    std::fs::write(dir.join("model.safetensors"), bytes).map_err(|err| GprError::InvalidConfig {
        reason: format!("test write: {err}"),
    })
}

/// Edits tensor `name` of `dir` with `edit`.
fn edit_tensor(
    dir: &std::path::Path,
    name: &str,
    edit: impl FnOnce(&mut Tensor),
) -> Result<(), GprError> {
    let mut tensors = read_tensors(dir);
    if let Some((_, tensor)) = tensors.iter_mut().find(|(n, _)| n == name) {
        edit(tensor);
    }
    write_tensors(dir, &tensors)
}

/// The `f64` values of a tensor's bytes, and back.
fn f64s(data: &[u8]) -> Vec<f64> {
    data.as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_le_bytes(*c))
        .collect()
}

fn f64_bytes(values: &[f64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Edits `dir/config.json` with `edit`.
fn edit_config(
    dir: &std::path::Path,
    edit: impl FnOnce(&mut serde_json::Value),
) -> Result<(), GprError> {
    let path = dir.join("config.json");
    let text = std::fs::read_to_string(&path).unwrap_or_default();
    let mut value: serde_json::Value =
        serde_json::from_str(&text).map_err(|err| GprError::InvalidConfig {
            reason: format!("test config: {err}"),
        })?;
    edit(&mut value);
    std::fs::write(&path, value.to_string()).map_err(|err| GprError::InvalidConfig {
        reason: format!("test write: {err}"),
    })
}

/// The `kind` of a `PersistFailed` result.
fn persist_kind<T>(result: &Result<T, GprError>) -> Option<PersistErrorKind> {
    match result {
        Err(GprError::PersistFailed { kind, .. }) => Some(*kind),
        _ => None,
    }
}

/// Saves the two-slot Exact model, applies `edit` to the saved directory,
/// and returns the load's result.
fn load_edited(
    label: &str,
    edit: impl FnOnce(&std::path::Path) -> Result<(), GprError>,
) -> Result<Result<LoadedDistanceGpr, GprError>, GprError> {
    let rows = train_idx(N);
    let kernel = two_slots()?;
    let slots = kernel.slots();
    let fitted = Gpr::new(kernel, GaussianLikelihood::new(0.1)?)
        .with_optimizer(Fixed)
        .factor(sources(&slots, &rows, &rows), N, &targets(N))?;
    let dir = temp_dir(&format!("edited-{label}"));
    fitted.save(&dir)?;
    edit(&dir)?;
    let loaded = LoadedDistanceGpr::<DistanceOnly>::load(&dir, &PersistRegistry::new());
    let _ = std::fs::remove_dir_all(&dir);
    Ok(loaded)
}

#[test]
fn edited_exact_tensors_are_refused() -> Result<(), GprError> {
    use PersistErrorKind::Tensor;
    // A file cut short is not a safetensors file.
    let cut = load_edited("cut", |dir| {
        let path = dir.join("model.safetensors");
        let bytes = std::fs::read(&path).unwrap_or_default();
        std::fs::write(&path, &bytes[..bytes.len() / 2]).map_err(|_| GprError::EmptyInput)
    })?;
    assert_eq!(persist_kind(&cut), Some(Tensor), "{cut:?}");
    // A slot's tensor with too few values, or another shape.
    let short = load_edited("short", |dir| {
        edit_tensor(dir, "d2.0", |(_, shape, data)| {
            data.truncate(data.len() - 8);
            shape[0] -= 1;
        })
    })?;
    assert_eq!(persist_kind(&short), Some(Tensor), "{short:?}");
    let shape = load_edited("shape", |dir| {
        edit_tensor(dir, "d2.1", |(_, shape, _)| shape.reverse())
    })?;
    assert_eq!(persist_kind(&shape), Some(Tensor), "{shape:?}");
    // Another scalar.
    let dtype = load_edited("dtype", |dir| {
        edit_tensor(dir, "d2.0", |(dtype, _, data)| {
            *dtype = safetensors::Dtype::F32;
            *data = f64s(data)
                .iter()
                .flat_map(|&v| (v as f32).to_le_bytes())
                .collect();
        })
    })?;
    assert_eq!(persist_kind(&dtype), Some(Tensor), "{dtype:?}");
    // A slot's tensor missing.
    let missing = load_edited("missing", |dir| {
        let tensors: Vec<_> = read_tensors(dir)
            .into_iter()
            .filter(|(name, _)| name != "d2.1")
            .collect();
        write_tensors(dir, &tensors)
    })?;
    assert_eq!(persist_kind(&missing), Some(Tensor), "{missing:?}");
    // A tensor no slot names is not read.
    let extra = load_edited("extra", |dir| {
        let mut tensors = read_tensors(dir);
        let copy = tensors
            .iter()
            .find(|(name, _)| name == "d2.0")
            .map(|(_, tensor)| tensor.clone());
        if let Some(copy) = copy {
            tensors.push(("d2.5".to_owned(), copy));
        }
        write_tensors(dir, &tensors)
    })?;
    assert!(extra.is_ok(), "{extra:?}");
    // A value that is not finite, a negative one, a non-zero diagonal.
    let nan = load_edited("nan", |dir| {
        edit_tensor(dir, "d2.0", |(_, _, data)| {
            let mut values = f64s(data);
            values[1] = f64::NAN;
            *data = f64_bytes(&values);
        })
    })?;
    assert_eq!(persist_kind(&nan), Some(Tensor), "{nan:?}");
    // Column 0 of the packed triangle: rows 0..N, the diagonal first.
    for (label, at, value) in [("negative", 1, -1.0), ("diagonal", 0, 0.5)] {
        let bad = load_edited(label, |dir| {
            edit_tensor(dir, "d2.0", |(_, _, data)| {
                let mut values = f64s(data);
                values[at] = value;
                *data = f64_bytes(&values);
            })
        })?;
        assert!(
            matches!(bad, Err(GprError::InvalidDistance { .. })),
            "{label}: {bad:?}"
        );
    }
    Ok(())
}

/// An edit of `config.json`.
type ConfigEdit = Box<dyn FnOnce(&mut serde_json::Value)>;

#[test]
fn edited_exact_configs_are_refused() -> Result<(), GprError> {
    use PersistErrorKind::Config;
    let cases: Vec<(&str, ConfigEdit)> = vec![
        // A leaf names a slot past the table.
        (
            "slot-past",
            Box::new(|v| {
                v["kernel"]["sum"]["left"]["product"]["right"]["distance"]["slot"] = 5.into()
            }),
        ),
        // The table in another order than the tree reads it: each leaf
        // still fits its slot's shape.
        (
            "order",
            Box::new(|v| {
                if let Some(slots) = v["distance"]["slots"].as_array_mut() {
                    slots.reverse();
                }
                v["kernel"]["sum"]["left"]["product"]["right"]["distance"]["slot"] = 1.into();
                v["kernel"]["sum"]["right"]["distance"]["slot"] = 0.into();
            }),
        ),
        // A slot the tree does not read.
        (
            "unused",
            Box::new(|v| {
                if let Some(slots) = v["distance"]["slots"].as_array_mut() {
                    slots.push(serde_json::json!({ "kind": "scalar" }));
                }
            }),
        ),
        // An ARD slot of other dimensions than its leaf.
        (
            "dims",
            Box::new(|v| v["distance"]["slots"][1]["dims"] = 3.into()),
        ),
        (
            "dims-zero",
            Box::new(|v| v["distance"]["slots"][1]["dims"] = 0.into()),
        ),
        // An empty table.
        (
            "empty",
            Box::new(|v| v["distance"]["slots"] = serde_json::json!([])),
        ),
        // A DistanceOnly file with coordinates.
        ("d", Box::new(|v| v["d"] = 1.into())),
        // A coordinate leaf in a DistanceOnly kernel.
        (
            "coordinate-leaf",
            Box::new(|v| {
                let tree = v["kernel"].take();
                v["kernel"] = serde_json::json!({
                    "product": {
                        "left": tree,
                        "right": { "rbf": { "lengthscale": { "value": 1.0, "lo": 0.00001, "hi": 100000.0 } } }
                    }
                });
            }),
        ),
    ];
    for (label, edit) in cases {
        let loaded = load_edited(label, |dir| edit_config(dir, edit))?;
        assert_eq!(persist_kind(&loaded), Some(Config), "{label}: {loaded:?}");
    }
    // Without the distance part the file is a coordinate one.
    let coordinate = load_edited("no-distance", |dir| {
        edit_config(dir, |v| {
            if let Some(object) = v.as_object_mut() {
                object.remove("distance");
            }
        })
    })?;
    assert!(is_wrong_model(&coordinate), "{coordinate:?}");
    // A scalar leaf on an ARD slot.
    let leaf = load_edited("leaf-shape", |dir| {
        edit_config(dir, |v| {
            v["distance"]["slots"] =
                serde_json::json!([{ "kind": "ard", "dims": 2 }, { "kind": "ard", "dims": 2 }]);
        })
    })?;
    assert_eq!(persist_kind(&leaf), Some(Config), "{leaf:?}");
    Ok(())
}

/// Saves a sparse model, applies `edit`, and returns the load's result.
fn load_edited_sparse(
    label: &str,
    points: bool,
    edit: impl FnOnce(&std::path::Path) -> Result<(), GprError>,
) -> Result<Result<(), GprError>, GprError> {
    let rows = train_idx(N);
    let z = pick(&rows, &INDUCING);
    let image = ScalarDistance::new();
    let dir = temp_dir(&format!("edited-sparse-{label}"));
    let registry = PersistRegistry::new();
    let loaded = if points {
        let kernel = image.kernel(RbfKernel::new(1.2)?) * KernelSpec::from(RbfKernel::new(0.7)?);
        let slots = kernel.slots();
        Sgpr::new(kernel, GaussianLikelihood::new(0.1)?)
            .with_optimizer(Fixed)
            .factor(
                sources(&slots, &rows, &z),
                N,
                &coords(0, &rows),
                1,
                &targets(N),
                &INDUCING,
            )
            .map_err(|(_, e)| e)?
            .save(&dir)?;
        edit(&dir)?;
        LoadedDistanceSgpr::<WithPoints>::load(&dir, &registry).map(|_| ())
    } else {
        let kernel = image.kernel(RbfKernel::new(1.2)?);
        let slots = kernel.slots();
        Sgpr::new(kernel, GaussianLikelihood::new(0.1)?)
            .with_optimizer(Fixed)
            .factor(sources(&slots, &rows, &z), N, &targets(N), &INDUCING)
            .map_err(|(_, e)| e)?
            .save(&dir)?;
        edit(&dir)?;
        LoadedDistanceSgpr::<DistanceOnly>::load(&dir, &registry).map(|_| ())
    };
    let _ = std::fs::remove_dir_all(&dir);
    Ok(loaded)
}

#[test]
fn edited_sparse_files_are_refused() -> Result<(), GprError> {
    // An inducing index past the training points: an error, not a panic.
    let past = load_edited_sparse("past", false, |dir| {
        edit_config(dir, |v| {
            v["distance"]["inducing"] = serde_json::json!([0, 3, 99])
        })
    })?;
    assert!(
        matches!(past, Err(GprError::IndexOutOfRange { .. })),
        "{past:?}"
    );
    // An index listed twice.
    let twice = load_edited_sparse("twice", false, |dir| {
        edit_config(dir, |v| {
            v["distance"]["inducing"] = serde_json::json!([0, 0, 5])
        })
    })?;
    assert!(
        matches!(twice, Err(GprError::InvalidConfig { .. })),
        "{twice:?}"
    );
    // Another number of indices than `m`.
    let count = load_edited_sparse("count", false, |dir| {
        edit_config(dir, |v| {
            v["distance"]["inducing"] = serde_json::json!([0, 3])
        })
    })?;
    assert_eq!(
        persist_kind(&count),
        Some(PersistErrorKind::Config),
        "{count:?}"
    );
    // No indices at all.
    let none = load_edited_sparse("none", false, |dir| {
        edit_config(dir, |v| {
            if let Some(distance) = v["distance"].as_object_mut() {
                distance.remove("inducing");
            }
        })
    })?;
    assert_eq!(
        persist_kind(&none),
        Some(PersistErrorKind::Config),
        "{none:?}"
    );
    // A block whose inducing rows are not a symmetric square.
    let mirror = load_edited_sparse("mirror", false, |dir| {
        edit_tensor(dir, "d2.0", |(_, _, data)| {
            let mut values = f64s(data);
            // Column 1 (inducing point 3), row 0 (inducing point 0).
            values[N] += 1.0;
            *data = f64_bytes(&values);
        })
    })?;
    assert!(
        matches!(mirror, Err(GprError::InvalidDistance { .. })),
        "{mirror:?}"
    );
    // `z` that is not the training rows the indices name.
    for name in ["z", "z_train"] {
        let moved = load_edited_sparse(name, true, |dir| {
            edit_tensor(dir, name, |(_, _, data)| {
                let mut values = f64s(data);
                values[0] += 0.25;
                *data = f64_bytes(&values);
            })
        })?;
        assert_eq!(
            persist_kind(&moved),
            Some(PersistErrorKind::Config),
            "{name}: {moved:?}"
        );
    }
    // A WithPoints file without coordinates.
    let flat = load_edited_sparse("flat", true, |dir| edit_config(dir, |v| v["d"] = 0.into()))?;
    assert!(matches!(flat, Err(GprError::EmptyInput)), "{flat:?}");
    Ok(())
}

/// A saved map that sends the data past `f64` is refused on load, also
/// when the file has no factor and the model is factored again.
#[test]
fn saved_maps_past_f64_are_refused() -> Result<(), GprError> {
    let rows = train_idx(N);
    let fitted = Gpr::new(
        KernelSpec::from(RbfKernel::new(1.0)?),
        GaussianLikelihood::new(0.1)?,
    )
    .with_target_transform(gprx::transform::StandardizeTarget::new())
    .fit(&coords(0, &rows), N, 1, &targets(N))?;
    let dir = temp_dir("past-f64");
    fitted.save(&dir)?;
    edit_config(&dir, |v| {
        v["y_transform"]["standardize"]["std"] = 1e-320.into()
    })?;
    let loaded = LoadedGpr::load(&dir, &PersistRegistry::new());
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        matches!(loaded, Err(GprError::NonFiniteInput)),
        "{loaded:?}"
    );
    Ok(())
}

// ---- A loaded model keeps working as the saved one does. ----

/// A loaded online model has the saved ids, hands out the same next id,
/// and after the same insert, delete, and refit predicts as the saved one.
/// Load lays its ARD slot out again for the changes (a packed store becomes
/// row runs); a model read with its factor reads its `d²` only in refit.
#[test]
fn loaded_online_model_keeps_its_ids_and_updates() -> Result<(), GprError> {
    let rows = train_idx(N);
    let kernel = two_slots()?;
    let slots = kernel.slots();
    let mut online = Gpr::new(kernel, GaussianLikelihood::new(0.1)?)
        .with_optimizer(Fixed)
        .factor(sources(&slots, &rows, &rows), N, &targets(N))?
        .into_online()?;
    let first = online.point_ids()[0];
    online.delete(first)?;
    let live: Vec<f64> = rows[1..].to_vec();
    let q = query_idx();
    for with_factor in [false, true] {
        let dir = temp_dir(&format!("ids-{with_factor}"));
        if with_factor {
            online.save_with_factor(&dir)?;
        } else {
            online.save(&dir)?;
        }
        let loaded = LoadedDistanceGpr::<DistanceOnly>::load(&dir, &PersistRegistry::new())?;
        let _ = std::fs::remove_dir_all(&dir);
        let LoadedDistanceGpr::OnlineDouble(mut loaded) = loaded else {
            return Err(GprError::InvalidConfig {
                reason: "an online double file loads OnlineDouble".into(),
            });
        };
        assert_eq!(loaded.point_ids(), online.point_ids());
        let fresh = loaded.slots();
        let mut saved = online.clone();
        let new = [N as f64 + 1.0];
        let a = saved.insert(sources(&slots, &live, &new), 0.3)?;
        let b = loaded.insert(sources(&fresh, &live, &new), 0.3)?;
        assert_eq!(a, b, "the next id is the saved model's");
        let mut after = live.clone();
        after.extend_from_slice(&new);
        let gone = saved.point_ids()[1];
        saved.delete(gone)?;
        loaded.delete(gone)?;
        after.remove(1);
        saved.refit()?;
        loaded.refit()?;
        assert_eq!(loaded.point_ids(), saved.point_ids());
        let want = widen(saved.predict(sources(&slots, &after, &q), M)?);
        let got = widen(loaded.predict(sources(&fresh, &after, &q), M)?);
        assert_same(&format!("updated factor={with_factor}"), &got, &want, 1e-10);
    }
    Ok(())
}

/// A fitted model read with its factor predicts from that factor; its
/// refit reads the loaded `d²` and must give the saved model's refit.
#[test]
fn loaded_factor_model_refits_from_its_squares() -> Result<(), GprError> {
    let rows = train_idx(N);
    let kernel = two_slots()?;
    let slots = kernel.slots();
    let mut fitted = Gpr::new(kernel, GaussianLikelihood::new(0.1)?)
        .with_optimizer(Fixed)
        .factor(sources(&slots, &rows, &rows), N, &targets(N))?;
    let dir = temp_dir("refit-factor");
    fitted.save_with_factor(&dir)?;
    let loaded = LoadedDistanceGpr::<DistanceOnly>::load(&dir, &PersistRegistry::new())?;
    let _ = std::fs::remove_dir_all(&dir);
    let LoadedDistanceGpr::Double(mut loaded) = loaded else {
        return Err(GprError::InvalidConfig {
            reason: "a fitted double file loads Double".into(),
        });
    };
    let mut theta = vec![0.0; fitted.num_params()];
    fitted.get_params(&mut theta)?;
    for value in &mut theta {
        *value += 0.1;
    }
    fitted.set_params(&theta)?;
    loaded.set_params(&theta)?;
    let q = query_idx();
    let want = widen(fitted.predict(sources(&slots, &rows, &q), M)?);
    let got = widen(loaded.predict(sources(&loaded.slots(), &rows, &q), M)?);
    assert_same("refit at a new theta", &got, &want, 1e-12);
    Ok(())
}

/// The bytes of tensor `name` of a model saved in `dir`.
fn tensor_bytes(dir: &std::path::Path, name: &str) -> Option<Tensor> {
    read_tensors(dir)
        .into_iter()
        .find(|(n, _)| n == name)
        .map(|(_, tensor)| tensor)
}

/// A mixed model writes the caller's `f64` values, the same bytes as a
/// double model of the same data; a single model writes their `f32` casts.
#[test]
fn mixed_writes_the_exact_squares() -> Result<(), GprError> {
    let rows = train_idx(N);
    let y = targets(N);
    let mut saved = Vec::new();
    for label in ["double", "mixed", "single"] {
        let kernel = two_slots()?;
        let slots = kernel.slots();
        let trainer = Gpr::new(kernel, GaussianLikelihood::new(0.1)?).with_optimizer(Fixed);
        let dir = temp_dir(&format!("bytes-{label}"));
        match label {
            "double" => trainer
                .factor(sources(&slots, &rows, &rows), N, &y)?
                .save(&dir)?,
            "mixed" => trainer
                .with_precision::<MixedPrecision>()
                .factor(sources(&slots, &rows, &rows), N, &y)?
                .save(&dir)?,
            _ => trainer
                .with_precision::<SinglePrecision>()
                .factor(sources(&slots, &rows, &rows), N, &y)?
                .save(&dir)?,
        }
        saved.push((tensor_bytes(&dir, "d2.0"), tensor_bytes(&dir, "d2.1")));
        let _ = std::fs::remove_dir_all(&dir);
    }
    let (double, mixed, single) = (&saved[0], &saved[1], &saved[2]);
    assert!(double.0.is_some() && double.1.is_some());
    assert_eq!(mixed, double, "mixed writes the f64 values to the bit");
    for (exact, cast) in [(&double.0, &single.0), (&double.1, &single.1)] {
        let (Some((_, shape, exact)), Some((dtype, cast_shape, cast))) = (exact, cast) else {
            return Err(GprError::EmptyInput);
        };
        assert_eq!(*dtype, safetensors::Dtype::F32);
        assert_eq!(cast_shape, shape);
        let want: Vec<u8> = f64s(exact)
            .iter()
            .flat_map(|&v| (v as f32).to_le_bytes())
            .collect();
        assert_eq!(cast, &want, "single writes its f32 store");
    }
    Ok(())
}

/// A `WithPoints` model of an ARD slot and a coordinate leaf round-trips
/// fitted and online, with and without its factor.
#[test]
fn with_points_ard_online_and_factor_round_trip() -> Result<(), GprError> {
    let rows = train_idx(N);
    let x = coords(0, &rows);
    let kernel = ArdDistance::from_leaf(RbfArdKernel::new(&[0.8, 1.3])?).1
        * KernelSpec::from(RbfKernel::new(0.7)?);
    let slots = kernel.slots();
    let fitted = Gpr::new(kernel, GaussianLikelihood::new(0.1)?)
        .with_optimizer(Fixed)
        .factor(sources(&slots, &rows, &rows), N, &x, 1, &targets(N))?;
    let mut online = fitted.clone().into_online()?;
    let new = [N as f64];
    online.insert(sources(&slots, &rows, &new), &coords(0, &new), 0.3)?;
    let first = online.point_ids()[0];
    online.delete(first)?;
    let live: Vec<f64> = (1..=N).map(|i| i as f64).collect();
    let q = query_idx();
    let xq = coords(0, &q);
    let want_fitted = widen(fitted.predict(sources(&slots, &rows, &q), &xq, M, 1)?);
    let want_online = widen(online.predict(sources(&slots, &live, &q), &xq, M, 1)?);
    for with_factor in [false, true] {
        for is_online in [false, true] {
            let dir = temp_dir(&format!("points-ard-{with_factor}-{is_online}"));
            match (is_online, with_factor) {
                (false, false) => fitted.save(&dir)?,
                (false, true) => fitted.save_with_factor(&dir)?,
                (true, false) => online.save(&dir)?,
                (true, true) => online.save_with_factor(&dir)?,
            }
            let loaded = LoadedDistanceGpr::<WithPoints>::load(&dir, &PersistRegistry::new())?;
            let _ = std::fs::remove_dir_all(&dir);
            assert_eq!(loaded.is_online(), is_online);
            let (rows, want) = if is_online {
                (&live, &want_online)
            } else {
                (&rows, &want_fitted)
            };
            let got = widen(loaded.predict(sources(&loaded.slots(), rows, &q), &xq, M, 1)?);
            assert_same(
                &format!("points ard factor={with_factor} online={is_online}"),
                &got,
                want,
                1e-10,
            );
        }
    }
    Ok(())
}

/// Every pair of a saved model and a loader of another kind is refused.
#[test]
fn every_other_loader_refuses_the_file() -> Result<(), GprError> {
    let rows = train_idx(N);
    let x = coords(0, &rows);
    let y = targets(N);
    let z = pick(&rows, &INDUCING);
    let registry = PersistRegistry::new();
    let refuse = |dir: &std::path::Path, own: &str| {
        let results = [
            ("gpr", LoadedGpr::load(dir, &registry).map(|_| ())),
            ("sgpr", LoadedSgpr::load(dir, &registry).map(|_| ())),
            ("svgp", LoadedSvgp::load(dir, &registry).map(|_| ())),
            (
                "dgpr",
                LoadedDistanceGpr::<DistanceOnly>::load(dir, &registry).map(|_| ()),
            ),
            (
                "dgpr-points",
                LoadedDistanceGpr::<WithPoints>::load(dir, &registry).map(|_| ()),
            ),
            (
                "dsgpr",
                LoadedDistanceSgpr::<DistanceOnly>::load(dir, &registry).map(|_| ()),
            ),
            (
                "dsgpr-points",
                LoadedDistanceSgpr::<WithPoints>::load(dir, &registry).map(|_| ()),
            ),
            (
                "dsvgp",
                LoadedDistanceSvgp::<DistanceOnly>::load(dir, &registry).map(|_| ()),
            ),
            (
                "dsvgp-points",
                LoadedDistanceSvgp::<WithPoints>::load(dir, &registry).map(|_| ()),
            ),
        ];
        for (name, result) in results {
            if name == own {
                assert!(result.is_ok(), "{own}: its own loader: {result:?}");
            } else {
                assert!(is_wrong_model(&result), "{own} read by {name}: {result:?}");
            }
        }
    };
    let coordinate =
        || -> Result<KernelSpec, GprError> { Ok(KernelSpec::from(RbfKernel::new(1.0)?)) };
    let dir = temp_dir("refuse");

    Gpr::new(coordinate()?, GaussianLikelihood::new(0.1)?)
        .fit(&x, N, 1, &y)?
        .save(&dir)?;
    refuse(&dir, "gpr");
    Sgpr::new(coordinate()?, GaussianLikelihood::new(0.1)?)
        .with_optimizer(Fixed)
        .factor(&x, N, 1, &y, &coords(0, &z), INDUCING.len())
        .map_err(|(_, e)| e)?
        .save(&dir)?;
    refuse(&dir, "sgpr");
    Svgp::new(coordinate()?, GaussianLikelihood::new(0.1)?)
        .factor(&x, N, 1, &y, &coords(0, &z), INDUCING.len())
        .map_err(|(_, e)| e)?
        .save(&dir)?;
    refuse(&dir, "svgp");

    let only = scalar_only()?;
    let slots = only.slots();
    Gpr::new(only, GaussianLikelihood::new(0.1)?)
        .with_optimizer(Fixed)
        .factor(sources(&slots, &rows, &rows), N, &y)?
        .save(&dir)?;
    refuse(&dir, "dgpr");
    let only = scalar_only()?;
    let slots = only.slots();
    Sgpr::new(only, GaussianLikelihood::new(0.1)?)
        .with_optimizer(Fixed)
        .factor(sources(&slots, &rows, &z), N, &y, &INDUCING)
        .map_err(|(_, e)| e)?
        .save(&dir)?;
    refuse(&dir, "dsgpr");
    let only = scalar_only()?;
    let slots = only.slots();
    Svgp::new(only, GaussianLikelihood::new(0.1)?)
        .factor(sources(&slots, &rows, &z), N, &y, &INDUCING)
        .map_err(|(_, e)| e)?
        .save(&dir)?;
    refuse(&dir, "dsvgp");

    let points = || -> Result<DistanceKernel<WithPoints>, GprError> {
        Ok(ScalarDistance::new().kernel(RbfKernel::new(1.2)?) * coordinate()?)
    };
    let kernel = points()?;
    let slots = kernel.slots();
    Gpr::new(kernel, GaussianLikelihood::new(0.1)?)
        .with_optimizer(Fixed)
        .factor(sources(&slots, &rows, &rows), N, &x, 1, &y)?
        .save(&dir)?;
    refuse(&dir, "dgpr-points");
    let kernel = points()?;
    let slots = kernel.slots();
    Sgpr::new(kernel, GaussianLikelihood::new(0.1)?)
        .with_optimizer(Fixed)
        .factor(sources(&slots, &rows, &z), N, &x, 1, &y, &INDUCING)
        .map_err(|(_, e)| e)?
        .save(&dir)?;
    refuse(&dir, "dsgpr-points");
    let kernel = points()?;
    let slots = kernel.slots();
    Svgp::new(kernel, GaussianLikelihood::new(0.1)?)
        .factor(sources(&slots, &rows, &z), N, &x, 1, &y, &INDUCING)
        .map_err(|(_, e)| e)?
        .save(&dir)?;
    refuse(&dir, "dsvgp-points");
    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}

/// A loaded online SGPR has the saved point and inducing ids, hands out the
/// same next ids, and after the same updates predicts as the saved one.
#[test]
fn loaded_online_sgpr_keeps_its_ids_and_updates() -> Result<(), GprError> {
    let rows = train_idx(N);
    let z = pick(&rows, &INDUCING);
    let kernel = two_slots()?;
    let slots = kernel.slots();
    let mut online = Sgpr::new(kernel, GaussianLikelihood::new(0.1)?)
        .with_optimizer(Fixed)
        .factor(sources(&slots, &rows, &z), N, &targets(N), &INDUCING)
        .map_err(|(_, e)| e)?
        .into_online();
    let gone = online.point_ids()[1];
    online.delete(gone)?;
    let mut live = rows.clone();
    live.remove(1);
    let dir = temp_dir("sgpr-ids");
    online.save(&dir)?;
    let loaded = LoadedDistanceSgpr::<DistanceOnly>::load(&dir, &PersistRegistry::new())?;
    let _ = std::fs::remove_dir_all(&dir);
    let LoadedDistanceSgpr::OnlineDouble(mut loaded) = loaded else {
        return Err(GprError::InvalidConfig {
            reason: "an online double file loads OnlineDouble".into(),
        });
    };
    assert_eq!(loaded.point_ids(), online.point_ids());
    assert_eq!(loaded.inducing_ids(), online.inducing_ids());
    let fresh = loaded.slots();
    let z_live = pick(&live, &common::inducing_places(&online));
    let new = [N as f64];
    let a = online.insert(sources(&slots, &z_live, &new), 0.3)?;
    let b = loaded.insert(sources(&fresh, &z_live, &new), 0.3)?;
    assert_eq!(a, b, "the next point id is the saved model's");
    live.push(new[0]);
    // Live point 1 (training point 2) is not an inducing point.
    let point = online.point_ids()[1];
    let a = online.insert_inducing(point, sources(&slots, &live, &[live[1]]))?;
    let b = loaded.insert_inducing(point, sources(&fresh, &live, &[live[1]]))?;
    assert_eq!(a, b, "the next inducing id is the saved model's");
    assert!(loaded.inducing_points().eq(online.inducing_points()));
    let z_live = pick(&live, &common::inducing_places(&online));
    let q = query_idx();
    let want = widen(online.predict(sources(&slots, &z_live, &q), M)?);
    let got = widen(loaded.predict(sources(&fresh, &z_live, &q), M)?);
    assert_same("online sgpr after updates", &got, &want, 1e-9);
    Ok(())
}
