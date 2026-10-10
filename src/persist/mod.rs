//! Saves and loads a fitted GPR directory (`config.json` + `model.safetensors`).

/// Runs `$call` in the arm of the precision `$kind`, with `$fitted` (and
/// `$online`) the variants of `$loaded` of that precision. Each arm names
/// its own precision type, so the call stays in the arm.
macro_rules! by_precision {
    ($kind:expr, $loaded:ident, |$fitted:ident, $online:ident| $call:expr) => {
        match $kind {
            $crate::precision::PersistKind::Double => {
                let ($fitted, $online) = ($loaded::Double, $loaded::OnlineDouble);
                $call
            }
            $crate::precision::PersistKind::Single => {
                let ($fitted, $online) = ($loaded::Single, $loaded::OnlineSingle);
                $call
            }
            $crate::precision::PersistKind::MixedPromote => {
                let ($fitted, $online) = ($loaded::Mixed, $loaded::OnlineMixed);
                $call
            }
            $crate::precision::PersistKind::MixedReevaluate => {
                let ($fitted, $online) = ($loaded::Reevaluate, $loaded::OnlineReevaluate);
                $call
            }
        }
    };
    ($kind:expr, $loaded:ident, |$fitted:ident| $call:expr) => {
        match $kind {
            $crate::precision::PersistKind::Double => {
                let $fitted = $loaded::Double;
                $call
            }
            $crate::precision::PersistKind::Single => {
                let $fitted = $loaded::Single;
                $call
            }
            $crate::precision::PersistKind::MixedPromote => {
                let $fitted = $loaded::Mixed;
                $call
            }
            $crate::precision::PersistKind::MixedReevaluate => {
                let $fitted = $loaded::Reevaluate;
                $call
            }
        }
    };
}

mod atomic;
mod config;
mod distance;
mod kernel;
mod registry;
mod sparse;
mod tensors;
mod transform;

use std::path::Path;

use crate::error::GprError;
use crate::error::PersistErrorKind;
use crate::gpr::{FittedGpr, GprCore, OnlineGpr, Policies};
use crate::kernel::{DistanceSlot, KernelSpec, ModelKernel, ModelKernelParts, SourceStore};
use crate::optimizer::Fixed;
use crate::precision::GpScalar;
use crate::transform::{TargetTransform, Transform, UnfittedTarget, UnfittedTransform};
use crate::{GaussianLikelihood, PredictOptions, Prediction};

use crate::kernel::ScalarOps;
use crate::precision::PersistKind;
use config::{
    DistanceCacheJson, DistanceJson, FactorKind, JitterJson, LikelihoodJson, MathJson, ModelConfig,
    PointsJson, PrecisionJson, ResidualJson, d2_tensor,
};
use kernel::KernelJson;
use tensors::{
    FactorBytes, RawTensor, pack_lower, read_matrix, read_scalars, read_xy, scalar_bytes,
    write_tensors,
};

pub use distance::{LoadedDistanceGpr, LoadedDistanceSgpr, LoadedDistanceSvgp};
pub use sparse::{LoadedSgpr, LoadedSvgp};
pub(crate) use sparse::{save_online_sgpr, save_sgpr, save_svgp};
pub(crate) use tensors::MappedTensors;
use transform::{
    encode_fitted_input, encode_fitted_target, encode_unfitted_input, encode_unfitted_target,
};

pub use registry::{
    FittedInputRestore, FittedTargetRestore, KernelRestore, PersistRegistry, UnfittedInputRestore,
    UnfittedTargetRestore,
};

/// Names the `config.json` `format_version` written and read by this crate.
pub const FORMAT_VERSION: u32 = 1;

/// Names the prefix reserved for built-in persist tags.
///
/// Caller `persist_id`s must not use it.
pub const RESERVED_PREFIX: &str = "gprx.";

pub(crate) const CONFIG_FILE: &str = "config.json";
pub(crate) const TENSOR_FILE: &str = "model.safetensors";

/// Represents the prediction-only model loaded from a persist directory.
///
/// One variant per precision and factor kind: `llt` loads a [`FittedGpr`],
/// `ldlt` loads an [`OnlineGpr`]. The distance-cache policy, kernel `exp`,
/// and jitter policy are read back into the model's runtime policies.
///
/// [`Self::predict`], [`Self::predict_with`], [`Self::n`], [`Self::d`], and
/// [`Self::is_online`] work on any variant, so predicting needs no `match`.
/// Match a variant for the typed model: `predict_into`, `insert`, or
/// re-training with [`crate::FittedGpr::with_optimizer`] then
/// [`crate::FittedGpr::refit`]. The file does not store a solver or a
/// Cholesky buffer policy; load is always [`crate::CholeskyBuffer::Retain`].
///
/// # Examples
///
/// ```rust
/// use gprx::kernel::{KernelSpec, RbfKernel};
/// use gprx::persist::{LoadedGpr, PersistRegistry};
/// use gprx::{GaussianLikelihood, Gpr};
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let fitted = Gpr::new(
///     KernelSpec::from(RbfKernel::new(1.0)?),
///     GaussianLikelihood::new(0.1)?,
/// )
/// .fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
/// .map_err(|(_, e)| e)?;
/// let dir = std::env::temp_dir().join(format!(
///     "gprx-doctest-loaded-{}",
///     std::process::id()
/// ));
/// let _ = std::fs::remove_dir_all(&dir);
/// fitted.save(&dir)?;
/// let loaded = LoadedGpr::load(&dir, &PersistRegistry::new())?;
/// let pred = loaded.predict(&[0.5], 1, 1)?;
/// assert_eq!(pred.mean.len(), 1);
/// assert!(!loaded.is_online());
/// let LoadedGpr::Double(model) = loaded else {
///     panic!("default save is a double-precision llt model");
/// };
/// assert_eq!(model.n(), 2);
/// let _ = std::fs::remove_dir_all(&dir);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum LoadedGpr {
    /// Marks a [`crate::DoublePrecision`] model.
    Double(FittedGpr<Fixed>),
    /// Marks a [`crate::SinglePrecision`] model.
    Single(FittedGpr<Fixed, crate::SinglePrecision>),
    /// Marks a promoted-storage [`crate::MixedPrecision`] model.
    Mixed(FittedGpr<Fixed, crate::MixedPrecision>),
    /// Marks a [`crate::MixedPrecision`]`<`[`crate::ReevaluateKernel`]`>` model.
    Reevaluate(FittedGpr<Fixed, crate::MixedPrecision<crate::ReevaluateKernel>>),
    /// Marks a [`crate::DoublePrecision`] online model.
    OnlineDouble(OnlineGpr<Fixed>),
    /// Marks a [`crate::SinglePrecision`] online model.
    OnlineSingle(OnlineGpr<Fixed, crate::SinglePrecision>),
    /// Marks a promoted-storage [`crate::MixedPrecision`] online model.
    OnlineMixed(OnlineGpr<Fixed, crate::MixedPrecision>),
    /// Marks a [`crate::MixedPrecision`]`<`[`crate::ReevaluateKernel`]`>` online model.
    OnlineReevaluate(OnlineGpr<Fixed, crate::MixedPrecision<crate::ReevaluateKernel>>),
}

impl LoadedGpr {
    /// Reads `dir/config.json` and `dir/model.safetensors`.
    ///
    /// Reconstructs fitted transforms from the config and applies them to the
    /// stored original `X` / `y`. `factor_kind` is required: `llt` loads
    /// [`FittedGpr`], `ldlt` loads [`crate::OnlineGpr`] and requires
    /// `point_ids` plus `next_point_id`. When a factor is present, the
    /// safetensors file stays memory-mapped. The buffer policy is
    /// [`crate::CholeskyBuffer::Retain`]. Unknown [`FORMAT_VERSION`] is rejected.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::UnsupportedPersistVersion`] when `format_version`
    /// is not [`FORMAT_VERSION`], or [`GprError::PersistFailed`] when the
    /// directory, JSON, tensors, or registry lookup is invalid, and
    /// [`GprError::NonFiniteInput`] when the saved maps send the training
    /// data past `f64`. Factorization errors from a file written without `L`
    /// use the same variants as [`crate::Gpr<Fixed>::factor`].
    ///
    /// See the example on [`LoadedGpr`].
    pub fn load(dir: impl AsRef<Path>, registry: &PersistRegistry) -> Result<Self, GprError> {
        load_dir(dir.as_ref(), registry)
    }

    /// Holds the number of training points.
    ///
    /// See the example on [`LoadedGpr`].
    pub fn n(&self) -> usize {
        self.view().n()
    }

    /// Holds the number of input features.
    ///
    /// See the example on [`LoadedGpr`].
    pub fn d(&self) -> usize {
        self.view().d()
    }

    /// Returns the `true` for an [`OnlineGpr`] (`ldlt`), `false` for a [`FittedGpr`] (`llt`).
    ///
    /// See the example on [`LoadedGpr`].
    pub fn is_online(&self) -> bool {
        self.view().is_online()
    }

    /// Returns the predictive mean and variance at `xs` (latent variance), in `f64` whatever the stored precision.
    ///
    /// Same as the variant's `predict`. [`crate::SinglePrecision`] results are
    /// widened from `f32`, which does not change their values.
    ///
    /// # Errors
    ///
    /// Same as [`crate::FittedGpr::predict`].
    ///
    /// See the example on [`LoadedGpr`].
    pub fn predict(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
    ) -> Result<Prediction<f64>, GprError> {
        self.predict_with(xs, n_rows, n_cols, PredictOptions::default())
    }

    /// Returns the [`Self::predict`] with [`PredictOptions`].
    ///
    /// # Errors
    ///
    /// Same as [`crate::FittedGpr::predict_with`].
    ///
    /// See the example on [`LoadedGpr`].
    pub fn predict_with(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        options: PredictOptions,
    ) -> Result<Prediction<f64>, GprError> {
        self.view().predict_f64(xs, n_rows, n_cols, options)
    }

    /// The one place that tells the variants apart.
    fn view(&self) -> &dyn LoadedView {
        match self {
            Self::Double(model) => model,
            Self::Single(model) => model,
            Self::Mixed(model) => model,
            Self::Reevaluate(model) => model,
            Self::OnlineDouble(model) => model,
            Self::OnlineSingle(model) => model,
            Self::OnlineMixed(model) => model,
            Self::OnlineReevaluate(model) => model,
        }
    }
}

/// Reads of a loaded model that do not depend on its precision or factor.
trait LoadedView {
    fn n(&self) -> usize;
    fn d(&self) -> usize;
    fn is_online(&self) -> bool;
    fn predict_f64(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        options: PredictOptions,
    ) -> Result<Prediction<f64>, GprError>;
}

impl<P: crate::precision::GpScalar> LoadedView for FittedGpr<Fixed, P> {
    fn n(&self) -> usize {
        FittedGpr::n(self)
    }

    fn d(&self) -> usize {
        FittedGpr::d(self)
    }

    fn is_online(&self) -> bool {
        false
    }

    fn predict_f64(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        options: PredictOptions,
    ) -> Result<Prediction<f64>, GprError> {
        self.predict_with(xs, n_rows, n_cols, options).map(widen)
    }
}

impl<P: crate::precision::GpScalar> LoadedView for OnlineGpr<Fixed, P> {
    fn n(&self) -> usize {
        OnlineGpr::n(self)
    }

    fn d(&self) -> usize {
        OnlineGpr::d(self)
    }

    fn is_online(&self) -> bool {
        true
    }

    fn predict_f64(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        options: PredictOptions,
    ) -> Result<Prediction<f64>, GprError> {
        self.predict_with(xs, n_rows, n_cols, options).map(widen)
    }
}

fn widen<T: crate::kernel::KernelScalar>(pred: Prediction<T>) -> Prediction<f64> {
    let to_f64 = |values: Vec<T>| values.into_iter().map(T::to_f64).collect();
    Prediction {
        mean: to_f64(pred.mean),
        variance: to_f64(pred.variance),
        variance_kind: pred.variance_kind,
    }
}

pub(crate) struct PersistedModel<
    P: crate::precision::GpScalar = crate::precision::DoublePrecision,
    K: crate::kernel::ModelKernel = KernelSpec,
> {
    pub kernel: crate::kernel::SpecOf<K>,
    pub likelihood: GaussianLikelihood,
    pub x_unfitted: Box<dyn UnfittedTransform>,
    pub y_unfitted: Box<dyn UnfittedTarget>,
    pub x_transform: Box<dyn Transform>,
    pub y_transform: Box<dyn TargetTransform>,
    pub policies: Policies,
    pub x_obs: Vec<f64>,
    pub y_obs: Vec<f64>,
    /// `α` with `L` (owned or mapped); `None` when the file holds no factor,
    /// and the model factors again at the saved `θ`.
    pub alpha: Option<Vec<P::Refine>>,
    /// The training `d²` of a kernel on supplied distances.
    pub supplied: crate::kernel::Held<K, crate::gpr::ExactSupplied<P>>,
    pub owned_l: Option<faer::Mat<P::Storage>>,
    pub mapped: Option<MappedTensors>,
    /// Diagonal jitter the saved factor was built with.
    pub factor_jitter: f64,
}

/// Writes `config.json` last, after the tensors it describes, so a save
/// that fails part way never leaves a new config over old tensors.
fn write_config(dir: &Path, json: &[u8]) -> Result<(), GprError> {
    atomic::write_atomic(&dir.join(CONFIG_FILE), json)
}

pub(crate) fn persist_err(kind: PersistErrorKind, reason: impl Into<String>) -> GprError {
    GprError::PersistFailed {
        kind,
        reason: reason.into(),
    }
}

struct PackedFactor {
    l_dtype: safetensors::Dtype,
    l: Vec<u8>,
    alpha_dtype: safetensors::Dtype,
    alpha: Vec<u8>,
}

fn pack_saved_factor<T, A>(l: faer::MatRef<'_, T>, alpha: &[A]) -> Result<PackedFactor, GprError>
where
    T: crate::kernel::KernelScalar,
    A: crate::kernel::KernelScalar,
{
    let n = l.nrows();
    let mut packed_l = vec![T::from_f64(0.0); n * n];
    pack_lower(l, &mut packed_l);
    let l_dtype = <T as crate::kernel::ScalarOps>::DTYPE;
    let alpha_dtype = <A as crate::kernel::ScalarOps>::DTYPE;
    Ok(PackedFactor {
        l_dtype,
        l: scalar_bytes(&packed_l).to_vec(),
        alpha_dtype,
        alpha: scalar_bytes(alpha).to_vec(),
    })
}

/// The scalar a model's training `d²` is saved in.
type SavedScalar<P> = <<P as crate::precision::ModelPrecision>::Sources as SourceStore<
    <P as crate::precision::PrecisionPolicy>::Storage,
>>::Saved;

/// The kernel of a model's core as a save writes it: the tree, the distance
/// part of the config (none for a coordinate kernel), and each slot's
/// training `d²` (tensor `d2.<k>` for slot `k` of the table).
struct KernelSave<'a, T: Clone> {
    kernel: KernelJson,
    distance: Option<DistanceJson>,
    d2: Vec<(String, Vec<usize>, crate::kernel::PackedRuns<'a, T>)>,
}

impl<T: crate::kernel::KernelScalar> KernelSave<'_, T> {
    /// The `d²` tensors as the tensor writer takes them.
    fn tensors(&self) -> Vec<RawTensor<'_>> {
        self.d2
            .iter()
            .map(|(name, shape, values)| RawTensor {
                name: name.as_str(),
                dtype: <T as ScalarOps>::DTYPE,
                shape: shape.clone(),
                runs: values.runs().into_iter().map(scalar_bytes).collect(),
            })
            .collect()
    }
}

/// [`KernelSave`] of `core`'s kernel and training `d²`.
fn kernel_save<P: GpScalar, K: ModelKernel>(
    core: &GprCore<P, K>,
) -> Result<KernelSave<'_, SavedScalar<P>>, GprError> {
    let Some(held) = <K::Supply as crate::kernel::SupplyViews>::held(&core.supplied) else {
        return Ok(KernelSave {
            kernel: KernelJson::encode(&core.kernel, &[])?,
            distance: None,
            d2: Vec::new(),
        });
    };
    let slots = &held.slots;
    let kernel = KernelJson::encode(&core.kernel, slots)?;
    let points = PointsJson::of::<K>();
    let tri = crate::kernel::packed_len(core.n)?;
    let saved = held.sources.saved();
    let d2 = slots
        .iter()
        .enumerate()
        .map(|(k, slot)| {
            let values = saved.packed(slot.id()).ok_or_else(|| {
                persist_err(
                    PersistErrorKind::Tensor,
                    format!("distance slot {k} has no training squared distances"),
                )
            })?;
            Ok((d2_tensor(k), d2_shape(*slot, tri), values))
        })
        .collect::<Result<_, GprError>>()?;
    Ok(KernelSave {
        kernel,
        distance: Some(DistanceJson::encode(points, slots)),
        d2,
    })
}

/// The tensor shape of a slot's `len` values per block: `[len]` for a
/// scalar slot, `[dims, len]` for an ARD slot.
fn d2_shape(slot: DistanceSlot, len: usize) -> Vec<usize> {
    match slot.shape() {
        crate::kernel::SlotShape::Scalar => vec![len],
        crate::kernel::SlotShape::Ard(dims) => vec![dims, len],
    }
}

/// What an Exact save writes, read from either Exact model. The two models
/// differ only in the factor kind, the point ids, and where the factor lives.
struct ExactSave<'a> {
    n: usize,
    d: usize,
    kind: PersistKind,
    factor_kind: FactorKind,
    policies: Policies,
    kernel: KernelJson,
    distance: Option<DistanceJson>,
    d2: Vec<RawTensor<'a>>,
    likelihood: &'a GaussianLikelihood,
    factor_jitter: f64,
    x_unfitted: &'a dyn UnfittedTransform,
    y_unfitted: &'a dyn UnfittedTarget,
    x_transform: &'a dyn Transform,
    y_transform: &'a dyn TargetTransform,
    point_ids: Option<(Vec<u64>, u64)>,
    x: &'a [f64],
    y: &'a [f64],
    factor: Option<PackedFactor>,
}

/// Writes the tensors, then `config.json`, of an Exact model.
fn save_exact(dir: &Path, save: ExactSave<'_>) -> Result<(), GprError> {
    std::fs::create_dir_all(dir)
        .map_err(|err| persist_err(PersistErrorKind::Io, format!("create {dir:?}: {err}")))?;
    let (point_ids, next_point_id) = match save.point_ids {
        Some((ids, next)) => (Some(ids), Some(next)),
        None => (None, None),
    };
    let config = ModelConfig {
        format_version: FORMAT_VERSION,
        n: save.n,
        d: save.d,
        has_factor: save.factor.is_some(),
        factor_kind: save.factor_kind,
        precision: PrecisionJson::from_persist(save.kind),
        residual: ResidualJson::from_persist(save.kind),
        math: MathJson::encode(save.policies.math),
        kernel: save.kernel,
        likelihood: LikelihoodJson::encode(save.likelihood),
        jitter: JitterJson::encode(save.policies.jitter),
        factor_jitter: save.factor_jitter,
        distance_cache: Some(DistanceCacheJson::encode(save.policies.distance_cache)),
        x_unfitted: encode_unfitted_input(save.x_unfitted)?,
        y_unfitted: encode_unfitted_target(save.y_unfitted)?,
        x_transform: encode_fitted_input(save.x_transform)?,
        y_transform: encode_fitted_target(save.y_transform)?,
        point_ids,
        next_point_id,
        distance: save.distance,
    };
    let json = serde_json::to_vec_pretty(&config).map_err(|err| {
        persist_err(
            PersistErrorKind::Config,
            format!("serialize config.json: {err}"),
        )
    })?;
    let factor_refs = save.factor.as_ref().map(|packed| FactorBytes {
        l_dtype: packed.l_dtype,
        l: packed.l.as_slice(),
        alpha_dtype: packed.alpha_dtype,
        alpha: packed.alpha.as_slice(),
    });
    write_tensors(dir, save.x, save.y, save.n, save.d, factor_refs, &save.d2)?;
    write_config(dir, &json)
}

pub(crate) fn save_fitted<O, P, K>(
    model: &FittedGpr<O, P, K>,
    dir: &Path,
    with_factor: bool,
) -> Result<(), GprError>
where
    P: GpScalar,
    K: ModelKernel,
{
    let factor = if with_factor {
        Some(pack_saved_factor(model.chol_l(), model.alpha())?)
    } else {
        None
    };
    let core = model.core();
    let kernel = kernel_save(core)?;
    save_exact(
        dir,
        ExactSave {
            n: core.n,
            d: core.d,
            kind: P::persist_kind(),
            factor_kind: FactorKind::Llt,
            policies: model.policies(),
            kernel: kernel.kernel.clone(),
            distance: kernel.distance.clone(),
            d2: kernel.tensors(),
            likelihood: model.likelihood(),
            factor_jitter: model.factor_jitter(),
            x_unfitted: model.x_unfitted(),
            y_unfitted: model.y_unfitted(),
            x_transform: model.x_transform(),
            y_transform: model.y_transform(),
            point_ids: None,
            x: &core.x_obs,
            y: model.y(),
            factor,
        },
    )
}

pub(crate) fn save_online<O, P, K>(
    model: &OnlineGpr<O, P, K>,
    dir: &Path,
    with_factor: bool,
) -> Result<(), GprError>
where
    P: GpScalar,
    K: ModelKernel,
{
    let factor = if with_factor {
        Some(pack_saved_factor(model.ld_factor(), model.alpha()?)?)
    } else {
        None
    };
    let core = &model.core;
    let kernel = kernel_save(core)?;
    save_exact(
        dir,
        ExactSave {
            n: core.n,
            d: core.d,
            kind: P::persist_kind(),
            factor_kind: FactorKind::Ldlt,
            policies: model.policies(),
            kernel: kernel.kernel.clone(),
            distance: kernel.distance.clone(),
            d2: kernel.tensors(),
            likelihood: model.likelihood(),
            factor_jitter: model.factor_jitter(),
            x_unfitted: model.x_unfitted(),
            y_unfitted: model.y_unfitted(),
            x_transform: model.x_transform(),
            y_transform: model.y_transform(),
            point_ids: Some((model.persist_point_ids(), model.persist_next_point_id())),
            x: &core.x_obs,
            y: model.y(),
            factor,
        },
    )
}

fn apply_online_ids<O, P, K>(
    online: &mut OnlineGpr<O, P, K>,
    ids: &Option<(Vec<u64>, u64)>,
) -> Result<(), GprError>
where
    P: GpScalar,
    K: ModelKernel,
{
    let (ids, next_id) = ids
        .as_ref()
        .ok_or_else(|| persist_err(PersistErrorKind::Config, "ldlt config missing point_ids"))?;
    online.apply_persisted_ids(ids, *next_id)
}

/// The stored tensors of an Exact model, from one open of `model.safetensors`.
struct ExactTensors<P: GpScalar, K: ModelKernel> {
    x_obs: Vec<f64>,
    y_obs: Vec<f64>,
    /// `α` in the precision's refine scalar, when the factor is stored.
    alpha: Option<Vec<P::Refine>>,
    owned_l: Option<faer::Mat<P::Storage>>,
    mapped: Option<MappedTensors>,
    /// The training `d²` of a distance kernel, bound from the tensors
    /// [`kernel_save`] wrote.
    supplied: crate::kernel::Held<K, crate::gpr::ExactSupplied<P>>,
}

/// Reads `x`, `y`, the training `d²` of `slots`, and the factor `α` / `L`
/// in one open of the file.
///
/// An `f64` factor stays memory-mapped, and a file with training `d²` is
/// mapped too: each slot is read in place into its store, one copy. The
/// small tensors are copied out of the map; a file of neither is read once.
fn read_exact_tensors<P: GpScalar, K: ModelKernel>(
    dir: &Path,
    config: &ModelConfig,
    slots: Vec<DistanceSlot>,
) -> Result<ExactTensors<P, K>, GprError> {
    let (n, d, has_factor) = (config.n, config.d, config.has_factor);
    let storage = <P::Storage as ScalarOps>::DTYPE;
    let map_l = has_factor && storage == safetensors::Dtype::F64;
    let file = if map_l || !slots.is_empty() {
        tensors::TensorFile::map(dir)?
    } else {
        tensors::TensorFile::read(dir)?
    };
    let (x_obs, y_obs, alpha, owned_l, supplied) = {
        let tensors = file.tensors()?;
        let (x_obs, y_obs) = read_xy(&tensors, n, d)?;
        let alpha = if has_factor {
            Some(read_scalars::<P::Refine>(
                &tensors,
                tensors::TENSOR_ALPHA,
                &[n],
                <P::Refine as ScalarOps>::DTYPE,
            )?)
        } else {
            None
        };
        let owned_l = if has_factor && !map_l {
            Some(read_matrix::<P::Storage>(&tensors, n, storage)?)
        } else {
            None
        };
        let supplied = <K::Supply as crate::kernel::SupplyViews>::try_hold(|| {
            let tri = crate::kernel::packed_len(n)?;
            // Read in place: the store checks every value as it copies it.
            let d2 = slots
                .iter()
                .enumerate()
                .map(|(k, slot)| {
                    tensors::finite_tensor::<SavedScalar<P>>(
                        &tensors,
                        &d2_tensor(k),
                        &d2_shape(*slot, tri),
                        <SavedScalar<P> as ScalarOps>::DTYPE,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let sources = P::Sources::from_saved(&slots, &d2, n)?;
            Ok::<_, GprError>(crate::gpr::ExactSupplied::new(sources, slots))
        })?;
        (x_obs, y_obs, alpha, owned_l, supplied)
    };
    let mapped = if map_l {
        Some(file.into_mapped_l(n)?)
    } else {
        None
    };
    Ok(ExactTensors {
        x_obs,
        y_obs,
        alpha,
        owned_l,
        mapped,
        supplied,
    })
}

/// The variants of a loaded Exact enum `L` that hold precision `P` and
/// kernel `K`.
struct Variants<P: GpScalar, K: ModelKernel, L> {
    fitted: fn(FittedGpr<Fixed, P, K>) -> L,
    online: fn(OnlineGpr<Fixed, P, K>) -> L,
}

/// Reads and checks `config.json` of an Exact model with distance marker
/// `points` (`None` for a coordinate model).
fn read_exact_config(dir: &Path, points: Option<PointsJson>) -> Result<ModelConfig, GprError> {
    let bytes = read_model_bytes(dir, &[config::ModelJson::Exact], points)?;
    config::parse_config(&bytes)
}

/// The bytes of `dir/config.json`, once its `model` is one of `expected`
/// and its distance marker is `points` ([`config::parse_model`]).
fn read_model_bytes(
    dir: &Path,
    expected: &[config::ModelJson],
    points: Option<PointsJson>,
) -> Result<Vec<u8>, GprError> {
    let config_path = dir.join(CONFIG_FILE);
    let bytes = std::fs::read(&config_path)
        .map_err(|err| persist_err(PersistErrorKind::Io, format!("read {config_path:?}: {err}")))?;
    config::parse_model(&bytes, expected, points)?;
    Ok(bytes)
}

fn load_dir(dir: &Path, registry: &PersistRegistry) -> Result<LoadedGpr, GprError> {
    let config = read_exact_config(dir, None)?;
    by_precision!(config.persist_kind(), LoadedGpr, |fitted, online| {
        load_precision(dir, registry, config, Variants { fitted, online })
    })
}

/// The slots of a config's table, checked against the decoded kernel: the
/// tree reads every slot of the table, in the table's order.
fn decode_kernel<K: ModelKernel>(
    kernel: &KernelJson,
    distance: Option<&DistanceJson>,
    registry: &PersistRegistry,
) -> Result<(crate::kernel::SpecOf<K>, Vec<DistanceSlot>), GprError> {
    let slots = match distance {
        Some(distance) => distance.decode_slots()?,
        None => Vec::new(),
    };
    let spec = kernel.clone().decode_tree::<K::Supply>(registry, &slots)?;
    // Numbers the leaves in their tree, as a kernel built by hand is.
    let spec = <K as ModelKernelParts>::from_spec(spec).into_spec();
    let read = crate::kernel::spec_slots(&spec);
    if read
        .iter()
        .map(|slot| slot.id())
        .ne(slots.iter().map(|slot| slot.id()))
    {
        return Err(persist_err(
            PersistErrorKind::Config,
            "the kernel does not read the slot table's slots in the table's order",
        ));
    }
    // A `DistanceOnly` kernel holds no leaf that reads coordinates: its
    // model has none to give it.
    if !<K as ModelKernelParts>::POINTS && reads_coordinates(&spec) {
        return Err(persist_err(
            PersistErrorKind::Config,
            "a distance_only kernel holds a leaf that reads coordinates",
        ));
    }
    Ok((spec, slots))
}

/// Whether `spec` holds a leaf that reads coordinates: every coordinate
/// leaf but [`crate::kernel::ConstantKernel`] and
/// [`crate::kernel::WhiteKernel`], which a [`crate::kernel::DistanceOnly`]
/// kernel may hold.
fn reads_coordinates<S: crate::kernel::Supply>(spec: &KernelSpec<S>) -> bool {
    match spec {
        KernelSpec::Sum(left, right) | KernelSpec::Product(left, right) => {
            reads_coordinates(left) || reads_coordinates(right)
        }
        KernelSpec::Constant(_) | KernelSpec::White(_) | KernelSpec::Supplied(_) => false,
        _ => true,
    }
}

fn load_precision<P, K, L>(
    dir: &Path,
    registry: &PersistRegistry,
    config: ModelConfig,
    variants: Variants<P, K, L>,
) -> Result<L, GprError>
where
    P: GpScalar,
    K: ModelKernel,
{
    let ldlt_ids = match config.factor_kind {
        FactorKind::Ldlt => {
            let (ids, next_id) = config.online_ids()?;
            Some((ids.to_vec(), next_id))
        }
        FactorKind::Llt => None,
    };
    let (kernel, slots) = decode_kernel::<K>(&config.kernel, config.distance.as_ref(), registry)?;
    let likelihood = config.likelihood.decode()?;
    let policies = Policies {
        distance_cache: config
            .distance_cache
            .map(DistanceCacheJson::decode)
            .unwrap_or_default(),
        cholesky_buffer: crate::CholeskyBuffer::Retain,
        math: config.math.decode(),
        jitter: config.jitter.decode()?,
    };
    let x_unfitted = config.x_unfitted.clone().decode(registry)?;
    let y_unfitted = config.y_unfitted.clone().decode(registry)?;
    let x_transform = config.x_transform.clone().decode(registry)?;
    let y_transform = config.y_transform.clone().decode(registry)?;
    let ExactTensors {
        x_obs,
        y_obs,
        alpha,
        owned_l,
        mapped,
        supplied,
    } = read_exact_tensors::<P, K>(dir, &config, slots)?;
    let parts = PersistedModel {
        kernel,
        likelihood,
        x_unfitted,
        y_unfitted,
        x_transform,
        y_transform,
        policies,
        x_obs,
        y_obs,
        alpha,
        supplied,
        owned_l,
        mapped,
        factor_jitter: config.factor_jitter,
    };
    let saved_factor = parts.alpha.is_some();
    match config.factor_kind {
        FactorKind::Llt => Ok((variants.fitted)(FittedGpr::from_persisted(parts)?)),
        FactorKind::Ldlt => {
            // A saved factor is the packed LDLT; without one the model is
            // factored again as LLT and turned into LDLT.
            let mut online = if saved_factor {
                OnlineGpr::from_persisted(parts)?
            } else {
                FittedGpr::from_persisted(parts)?.into_online()?
            };
            apply_online_ids(&mut online, &ldlt_ids)?;
            Ok((variants.online)(online))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CONFIG_FILE, FORMAT_VERSION, LoadedGpr, PersistRegistry, RESERVED_PREFIX, persist_err,
    };
    use crate::error::PersistErrorKind;
    use crate::kernel::{KernelSpec, KernelTerm, LinearKernel, RbfKernel, Triangle};
    use crate::param::Interval;
    use crate::transform::{StandardizeInput, StandardizeTarget};
    use crate::{Fixed, GaussianLikelihood, Gpr, GprError, Lbfgs};
    use faer::{MatMut, MatRef};
    use std::path::PathBuf;

    const TOL: f64 = 1e-10;

    use crate::test_check::assert_close;

    fn temp_dir(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "gprx-persist-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
            label
        ));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[derive(Clone, Debug)]
    struct PersistUnit;

    impl<T: crate::kernel::KernelScalar> KernelTerm<T> for PersistUnit {
        fn num_params(&self) -> usize {
            0
        }

        fn get_params(&self, out: &mut [f64]) -> Result<(), GprError> {
            if out.is_empty() {
                Ok(())
            } else {
                Err(GprError::IndexOutOfRange {
                    reason: "persist unit kernel has no parameters".to_owned(),
                })
            }
        }

        fn set_params(&mut self, params: &[f64]) -> Result<(), GprError> {
            KernelTerm::<T>::get_params(self, &mut params.to_vec())
        }

        fn bounds_into(&self, out: &mut [Interval]) -> Result<(), GprError> {
            if out.is_empty() {
                Ok(())
            } else {
                Err(GprError::IndexOutOfRange {
                    reason: "persist unit kernel has no parameters".to_owned(),
                })
            }
        }

        fn apply(
            &self,
            dist: MatRef<'_, T>,
            mut out: MatMut<'_, T>,
            _uplo: Triangle,
        ) -> Result<(), GprError> {
            for col in 0..dist.ncols() {
                for row in 0..dist.nrows() {
                    out[(row, col)] = T::from_f64(1.0);
                }
            }
            Ok(())
        }

        fn apply_cross(&self, dist: MatRef<'_, T>, mut out: MatMut<'_, T>) -> Result<(), GprError> {
            for col in 0..dist.ncols() {
                for row in 0..dist.nrows() {
                    out[(row, col)] = T::from_f64(1.0);
                }
            }
            Ok(())
        }

        fn fill_diag(&self, out: &mut [T]) -> Result<(), GprError> {
            out.fill(T::from_f64(1.0));
            Ok(())
        }

        fn grad(
            &self,
            _dist: MatRef<'_, T>,
            _d_k: MatMut<'_, T>,
            param_idx: usize,
            _uplo: Triangle,
        ) -> Result<(), GprError> {
            Err(GprError::IndexOutOfRange {
                reason: format!("persist unit kernel has no parameter {param_idx}"),
            })
        }

        fn hess(
            &self,
            _dist: MatRef<'_, T>,
            _d2_k: MatMut<'_, T>,
            i: usize,
            j: usize,
            _uplo: Triangle,
        ) -> Result<(), GprError> {
            Err(GprError::IndexOutOfRange {
                reason: format!("persist unit kernel has no parameter pair ({i}, {j})"),
            })
        }

        fn hess_points(
            &self,
            _x: MatRef<'_, T>,
            _d2_k: MatMut<'_, T>,
            i: usize,
            j: usize,
            _uplo: Triangle,
        ) -> Result<(), GprError> {
            Err(GprError::IndexOutOfRange {
                reason: format!("persist unit kernel has no parameter pair ({i}, {j})"),
            })
        }

        fn clone_box(&self) -> Box<dyn KernelTerm<T>> {
            Box::new(self.clone())
        }

        fn persist_id(&self) -> &'static str {
            "test.persist_unit"
        }

        fn persist_state(&self) -> Result<serde_json::Value, GprError> {
            Ok(serde_json::json!({}))
        }
    }

    #[test]
    fn persist_err_is_failed_variant() {
        match persist_err(PersistErrorKind::Tensor, "missing l") {
            GprError::PersistFailed { reason, .. } => assert_eq!(reason, "missing l"),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn format_version_is_one() {
        assert_eq!(FORMAT_VERSION, 1);
        assert_eq!(RESERVED_PREFIX, "gprx.");
    }

    #[test]
    fn rbf_save_load_matches_predict() {
        let fitted = Gpr::new(
            KernelSpec::from(RbfKernel::new(1.0).expect("ℓ")),
            GaussianLikelihood::new(0.1).expect("noise"),
        )
        .fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
        .map_err(|(_, e)| e)
        .expect("fit");
        let want = fitted.predict(&[0.5], 1, 1).expect("predict");
        let dir = temp_dir("rbf-save");
        fitted.save(&dir).expect("save");
        let loaded = LoadedGpr::load(&dir, &PersistRegistry::new()).expect("load");
        let LoadedGpr::Double(model) = loaded else {
            panic!("RBF is a distance kernel");
        };
        let got = model.predict(&[0.5], 1, 1).expect("loaded predict");
        assert_close(got.mean[0], want.mean[0], TOL);
        assert_close(got.variance[0], want.variance[0], TOL);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rbf_save_with_factor_matches_predict() {
        let fitted = Gpr::new(
            KernelSpec::from(RbfKernel::new(1.0).expect("ℓ")),
            GaussianLikelihood::new(0.1).expect("noise"),
        )
        .fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
        .map_err(|(_, e)| e)
        .expect("fit");
        let want = fitted.predict(&[0.25], 1, 1).expect("predict");
        let dir = temp_dir("rbf-factor");
        fitted.save_with_factor(&dir).expect("save");
        let loaded = LoadedGpr::load(&dir, &PersistRegistry::new()).expect("load");
        let LoadedGpr::Double(model) = loaded else {
            panic!("RBF is a distance kernel");
        };
        let got = model.predict(&[0.25], 1, 1).expect("loaded predict");
        assert_close(got.mean[0], want.mean[0], TOL);
        assert_close(got.variance[0], want.variance[0], TOL);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Overwrites the first value of the `f64` tensor `name` with `NaN`.
    fn poison_tensor(dir: &std::path::Path, name: &str) {
        let path = dir.join(super::TENSOR_FILE);
        let mut bytes = std::fs::read(&path).expect("read");
        let offset = {
            let tensors = safetensors::SafeTensors::deserialize(&bytes).expect("header");
            let data = tensors.tensor(name).expect("tensor").data();
            data.as_ptr() as usize - bytes.as_ptr() as usize
        };
        bytes[offset..offset + 8].copy_from_slice(&f64::NAN.to_le_bytes());
        std::fs::write(&path, bytes).expect("write");
    }

    #[test]
    fn non_finite_stored_tensors_fail_to_load() {
        for name in ["x", "y", "l", "alpha"] {
            let dir = temp_dir("nan-tensor");
            Gpr::new(
                KernelSpec::from(RbfKernel::new(1.0).expect("ℓ")),
                GaussianLikelihood::new(0.1).expect("noise"),
            )
            .with_optimizer(Fixed)
            .factor(&[0.0, 1.0, 2.0], 3, 1, &[0.5, -1.0, 2.0])
            .map_err(|(_, e)| e)
            .expect("factor")
            .save_with_factor(&dir)
            .expect("save");
            poison_tensor(&dir, name);
            match LoadedGpr::load(&dir, &PersistRegistry::new()) {
                Err(GprError::PersistFailed { kind, reason }) => {
                    assert_eq!(kind, PersistErrorKind::Tensor);
                    assert!(reason.contains("non-finite"), "{name}: {reason}");
                    assert!(
                        reason.contains(&format!("tensor {name} ")),
                        "{name}: {reason}"
                    );
                }
                other => panic!(
                    "{name}: unexpected {:?}",
                    other.err().map(|e| e.to_string())
                ),
            }
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    #[test]
    fn non_finite_stored_sparse_tensor_fails_to_load() {
        let dir = temp_dir("nan-sparse");
        crate::Sgpr::new(
            KernelSpec::from(RbfKernel::new(1.0).expect("ℓ")),
            GaussianLikelihood::new(0.1).expect("noise"),
        )
        .with_optimizer(Fixed)
        .factor(
            &[0.0, 1.0, 2.0, 3.0],
            4,
            1,
            &[0.5, -1.0, 2.0, 0.0],
            &[0.5, 2.5],
            2,
        )
        .map_err(|(_, e)| e)
        .expect("factor")
        .save(&dir)
        .expect("save");
        poison_tensor(&dir, "y");
        match crate::LoadedSgpr::load(&dir, &PersistRegistry::new()) {
            Err(GprError::PersistFailed { kind, reason }) => {
                assert_eq!(kind, PersistErrorKind::Tensor);
                assert!(reason.contains("non-finite"), "{reason}");
            }
            other => panic!("unexpected {:?}", other.err().map(|e| e.to_string())),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Swaps `data_min` and `data_max` in every object of `value` holding both.
    fn swap_extrema(value: &mut serde_json::Value) -> usize {
        let mut swapped = 0;
        match value {
            serde_json::Value::Object(map) => {
                if let (Some(lo), Some(hi)) =
                    (map.get("data_min").cloned(), map.get("data_max").cloned())
                {
                    map.insert("data_min".to_owned(), hi);
                    map.insert("data_max".to_owned(), lo);
                    swapped += 1;
                }
                for child in map.values_mut() {
                    swapped += swap_extrema(child);
                }
            }
            serde_json::Value::Array(items) => {
                for child in items {
                    swapped += swap_extrema(child);
                }
            }
            _ => {}
        }
        swapped
    }

    #[test]
    fn reversed_min_max_extrema_fail_to_load() {
        for key in ["x_transform", "y_transform"] {
            let dir = temp_dir("minmax-reversed");
            Gpr::new(
                KernelSpec::from(RbfKernel::new(1.0).expect("ℓ")),
                GaussianLikelihood::new(0.1).expect("noise"),
            )
            .with_input_transform(crate::transform::MinMaxInput::new())
            .with_target_transform(crate::transform::MinMaxTarget::new())
            .with_optimizer(Fixed)
            .factor(&[0.0, 1.0, 2.0], 3, 1, &[0.5, -1.0, 2.0])
            .map_err(|(_, e)| e)
            .expect("factor")
            .save(&dir)
            .expect("save");
            assert!(LoadedGpr::load(&dir, &PersistRegistry::new()).is_ok());
            let path = dir.join(CONFIG_FILE);
            let mut config: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&path).expect("read")).expect("json");
            assert_eq!(swap_extrema(&mut config[key]), 1, "{key}");
            std::fs::write(&path, serde_json::to_vec(&config).expect("encode")).expect("write");
            match LoadedGpr::load(&dir, &PersistRegistry::new()) {
                Err(GprError::InvalidHyperparameter { reason }) => {
                    assert!(reason.contains("data_min <= data_max"), "{key}: {reason}");
                }
                other => panic!("{key}: unexpected {:?}", other.err()),
            }
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    fn fixed_rbf(x: &[f64], y: &[f64]) -> crate::FittedGpr<Fixed> {
        Gpr::new(
            KernelSpec::from(RbfKernel::new(1.0).expect("ℓ")),
            GaussianLikelihood::new(0.1).expect("noise"),
        )
        .with_optimizer(Fixed)
        .factor(x, x.len(), 1, y)
        .map_err(|(_, e)| e)
        .expect("factor")
    }

    /// `depth` levels of `wrap` around `leaf`.
    fn nested(
        leaf: &serde_json::Value,
        depth: usize,
        wrap: fn(serde_json::Value) -> serde_json::Value,
    ) -> serde_json::Value {
        let mut value = leaf.clone();
        for _ in 0..depth {
            value = wrap(value);
        }
        value
    }

    #[test]
    fn deeply_nested_config_trees_are_rejected_without_overflow() {
        let dir = temp_dir("deep-config");
        fixed_rbf(&[0.0, 1.0], &[0.0, 1.0])
            .save(&dir)
            .expect("save");
        let path = dir.join(CONFIG_FILE);
        let config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).expect("read")).expect("json");
        let kernel = config["kernel"].clone();
        let input = config["x_unfitted"].clone();
        // Well past the parser's limit of 128 nested values, and shallow
        // enough that building and dropping the test value is safe.
        let depth = 300;
        let cases = [
            (
                "kernel",
                nested(&kernel, depth, |inner| {
                    let leaf = serde_json::json!({
                        "rbf": { "lengthscale": { "value": 1.0, "lo": 1e-5, "hi": 1e5 } }
                    });
                    serde_json::json!({ "sum": { "left": inner, "right": leaf } })
                }),
            ),
            (
                "x_unfitted",
                nested(
                    &input,
                    depth,
                    |inner| serde_json::json!({ "pipeline": { "steps": [inner] } }),
                ),
            ),
            (
                "x_unfitted",
                nested(
                    &input,
                    depth,
                    |inner| serde_json::json!({ "columnwise": { "maps": [inner] } }),
                ),
            ),
        ];
        for (key, value) in cases {
            let mut broken = config.clone();
            broken[key] = value;
            std::fs::write(&path, serde_json::to_vec(&broken).expect("encode")).expect("write");
            match LoadedGpr::load(&dir, &PersistRegistry::new()) {
                Err(GprError::PersistFailed { kind, reason }) => {
                    assert_eq!(kind, PersistErrorKind::Config);
                    assert!(reason.contains("recursion limit"), "{key}: {reason}");
                }
                other => panic!("{key}: unexpected {:?}", other.err()),
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn mapped_model_keeps_its_factor_when_its_directory_is_saved_over() {
        let x_big: Vec<f64> = (0..30).map(|i| f64::from(i) / 5.0).collect();
        let y_big: Vec<f64> = x_big.iter().map(|v| v.sin()).collect();
        let dir = temp_dir("mapped-overwrite");
        fixed_rbf(&x_big, &y_big)
            .save_with_factor(&dir)
            .expect("save big");
        let loaded = LoadedGpr::load(&dir, &PersistRegistry::new()).expect("load");
        let before = loaded.predict(&[0.3], 1, 1).expect("predict");
        // A smaller model over the same files: an in-place rewrite would
        // shrink the mapped file under `loaded` and fault on the next read.
        fixed_rbf(&[0.0, 1.0, 2.0], &[0.5, -0.5, 0.25])
            .save_with_factor(&dir)
            .expect("save small");
        let after = loaded.predict(&[0.3], 1, 1).expect("predict");
        assert_close(after.mean[0], before.mean[0], TOL);
        assert_close(after.variance[0], before.variance[0], TOL);
        let reloaded = LoadedGpr::load(&dir, &PersistRegistry::new()).expect("reload");
        assert_eq!(reloaded.n(), 3);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn failed_tensor_write_leaves_the_previous_config() {
        let dir = temp_dir("failed-tensors");
        fixed_rbf(&[0.0, 1.0], &[0.0, 1.0])
            .save(&dir)
            .expect("save");
        let config_before = std::fs::read(dir.join(CONFIG_FILE)).expect("config");
        let tensors = dir.join(super::TENSOR_FILE);
        std::fs::remove_file(&tensors).expect("remove tensors");
        // A non-empty directory where the tensor file goes cannot be replaced.
        std::fs::create_dir_all(tensors.join("blocker")).expect("blocker");
        let err = fixed_rbf(&[0.0, 1.0, 2.0], &[0.0, 1.0, 0.5])
            .save(&dir)
            .expect_err("tensor write must fail");
        assert!(matches!(err, GprError::PersistFailed { .. }), "{err:?}");
        assert_eq!(
            std::fs::read(dir.join(CONFIG_FILE)).expect("config"),
            config_before
        );
        let names: Vec<String> = std::fs::read_dir(&dir)
            .expect("read dir")
            .map(|entry| {
                entry
                    .expect("entry")
                    .file_name()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        assert!(
            names.iter().all(|name| !name.contains(".tmp-")),
            "temporary files left: {names:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn loaded_online_model_refuses_to_reuse_its_last_point_id() {
        let dir = temp_dir("ids-exhausted");
        Gpr::new(
            KernelSpec::from(RbfKernel::new(1.0).expect("ℓ")),
            GaussianLikelihood::new(0.1).expect("noise"),
        )
        .with_optimizer(Fixed)
        .factor(&[0.0, 1.0], 2, 1, &[0.5, -1.0])
        .map_err(|(_, e)| e)
        .expect("factor")
        .into_online()
        .expect("online")
        .save_with_factor(&dir)
        .expect("save");
        let path = dir.join(CONFIG_FILE);
        let mut config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).expect("read")).expect("json");
        config["next_point_id"] = serde_json::json!(u64::MAX - 1);
        std::fs::write(&path, serde_json::to_vec(&config).expect("encode")).expect("write");
        let LoadedGpr::OnlineDouble(mut online) =
            LoadedGpr::load(&dir, &PersistRegistry::new()).expect("load")
        else {
            panic!("an online f64 model");
        };
        let id = online.insert(&[2.0], 0.25).expect("one id left");
        assert_eq!(online.n(), 3);
        let err = online.insert(&[3.0], 0.5).expect_err("ids exhausted");
        assert!(matches!(err, GprError::IndexOutOfRange { .. }), "{err:?}");
        assert_eq!(online.n(), 3);
        assert_eq!(online.point_ids().last().copied(), Some(id));
        online.delete(id).expect("the id still names one point");
        assert_eq!(online.n(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn mapped_factor_survives_failed_set_params() {
        let fitted = Gpr::new(
            KernelSpec::from(RbfKernel::new(1.0).expect("ℓ")),
            GaussianLikelihood::new(0.1)
                .expect("noise")
                .with_bounds(Interval::new(1e-30, 1e5).expect("open"))
                .expect("inside"),
        )
        .with_optimizer(Fixed)
        .factor(&[0.0, 0.0], 2, 1, &[0.5, -0.25])
        .map_err(|(_, e)| e)
        .expect("factor");
        let want = fitted.predict(&[0.25], 1, 1).expect("predict");
        let dir = temp_dir("mapped-failed-set");
        fitted.save_with_factor(&dir).expect("save");
        let loaded = LoadedGpr::load(&dir, &PersistRegistry::new()).expect("load");
        let LoadedGpr::Double(mut model) = loaded else {
            panic!("default save is double precision");
        };
        let mut bad = [0.0; 2];
        model.get_params(&mut bad).expect("len 2");
        bad[0] = 0.5;
        bad[1] = (1e-20_f64).ln();
        assert!(matches!(
            model.set_params(&bad),
            Err(GprError::CholeskyFailed { .. })
        ));
        let got = model.predict(&[0.25], 1, 1).expect("usable after failure");
        assert_close(got.mean[0], want.mean[0], TOL);
        assert_close(got.variance[0], want.variance[0], TOL);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn linear_round_trips_without_distance_cache() {
        let fitted = Gpr::new(
            KernelSpec::from(LinearKernel::new(1.0).expect("σ²")),
            GaussianLikelihood::new(0.1).expect("noise"),
        )
        .fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
        .map_err(|(_, e)| e)
        .expect("fit");
        let want = fitted.predict(&[0.5], 1, 1).expect("predict");
        let dir = temp_dir("linear-points");
        fitted.save(&dir).expect("save");
        let loaded = LoadedGpr::load(&dir, &PersistRegistry::new()).expect("load");
        let LoadedGpr::Double(model) = loaded else {
            panic!("default save is a double-precision llt model");
        };
        let got = model.predict(&[0.5], 1, 1).expect("loaded predict");
        assert_close(got.mean[0], want.mean[0], TOL);
        assert_close(got.variance[0], want.variance[0], TOL);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn standardize_roundtrip_and_refit() {
        let fitted = Gpr::new(
            KernelSpec::from(RbfKernel::new(1.0).expect("ℓ")),
            GaussianLikelihood::new(0.1).expect("noise"),
        )
        .with_input_transform(StandardizeInput::new())
        .with_target_transform(StandardizeTarget::new())
        .fit(&[0.0, 2.0, 4.0], 3, 1, &[1.0, 3.0, 5.0])
        .map_err(|(_, e)| e)
        .expect("fit");
        let want = fitted.predict(&[1.0], 1, 1).expect("predict");
        let dir = temp_dir("std-roundtrip");
        fitted.save(&dir).expect("save");
        let loaded = LoadedGpr::load(&dir, &PersistRegistry::new()).expect("load");
        let LoadedGpr::Double(model) = loaded else {
            panic!("RBF is a distance kernel");
        };
        let got = model.predict(&[1.0], 1, 1).expect("loaded predict");
        assert_close(got.mean[0], want.mean[0], TOL);
        let mut retrained = model.with_optimizer(Lbfgs::new());
        retrained.refit().expect("refit");
        let after = retrained.predict(&[1.0], 1, 1).expect("refit predict");
        assert!(after.mean[0].is_finite());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn unknown_format_version_is_rejected() {
        let fitted = Gpr::new(
            KernelSpec::from(RbfKernel::new(1.0).expect("ℓ")),
            GaussianLikelihood::new(0.1).expect("noise"),
        )
        .fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
        .map_err(|(_, e)| e)
        .expect("fit");
        let dir = temp_dir("bad-version");
        fitted.save(&dir).expect("save");
        let path = dir.join(CONFIG_FILE);
        let mut value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).expect("read")).expect("json");
        value["format_version"] = serde_json::json!(99);
        std::fs::write(&path, serde_json::to_vec_pretty(&value).expect("write")).expect("rewrite");
        match LoadedGpr::load(&dir, &PersistRegistry::new()) {
            Err(GprError::UnsupportedPersistVersion {
                found: 99,
                supported: 1,
            }) => {}
            other => panic!("unexpected {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn reserved_persist_id_is_rejected() {
        let mut registry = PersistRegistry::new();
        let err = registry
            .register_kernel("gprx.rbf", |_state| {
                Ok(crate::kernel::CustomKernel::new(PersistUnit))
            })
            .expect_err("reserved");
        match err {
            GprError::PersistFailed {
                kind: PersistErrorKind::InvalidPersistId,
                reason,
            } => {
                assert!(reason.contains("gprx."), "{reason}");
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn custom_kernel_needs_registry() {
        let fitted = Gpr::new(
            KernelSpec::custom(PersistUnit),
            GaussianLikelihood::new(0.1).expect("noise"),
        )
        .with_optimizer(Fixed)
        .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
        .map_err(|(_, e)| e)
        .expect("factor");
        let dir = temp_dir("custom-kernel");
        fitted.save(&dir).expect("save");
        match LoadedGpr::load(&dir, &PersistRegistry::new()) {
            Err(GprError::PersistFailed {
                kind: PersistErrorKind::UnregisteredId,
                reason,
            }) => {
                assert!(reason.contains("test.persist_unit"), "{reason}");
            }
            other => panic!("unexpected {other:?}"),
        }
        let mut registry = PersistRegistry::new();
        registry
            .register_kernel("test.persist_unit", |_state| {
                Ok(crate::kernel::CustomKernel::new(PersistUnit))
            })
            .expect("register");
        let loaded = LoadedGpr::load(&dir, &registry).expect("load");
        assert!(matches!(loaded, LoadedGpr::Double(_)));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_directory_is_an_io_failure() {
        let dir = temp_dir("missing-dir").join("does-not-exist");
        match LoadedGpr::load(&dir, &PersistRegistry::new()) {
            Err(GprError::PersistFailed {
                kind: PersistErrorKind::Io,
                ..
            }) => {}
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn single_and_mixed_factor_round_trip() {
        fn must<T>(result: Result<T, GprError>) -> T {
            match result {
                Ok(value) => value,
                Err(err) => panic!("persist call failed: {err}"),
            }
        }
        let x = [0.0, 0.5, 1.0];
        let y = [0.0, 1.0, 0.2];
        let kernel = KernelSpec::from(must(RbfKernel::new(1.0)));
        let likelihood = must(GaussianLikelihood::new(0.1));
        let single = must(
            Gpr::new(kernel.clone(), likelihood)
                .with_optimizer(Fixed)
                .with_precision::<crate::SinglePrecision>()
                .factor(&x, 3, 1, &y)
                .map_err(|(_, err)| err),
        );
        let mixed = must(
            Gpr::new(kernel, likelihood)
                .with_optimizer(Fixed)
                .with_precision::<crate::MixedPrecision>()
                .factor(&x, 3, 1, &y)
                .map_err(|(_, err)| err),
        );
        let double = must(
            Gpr::new(
                KernelSpec::from(must(RbfKernel::new(1.0))),
                must(GaussianLikelihood::new(0.1)),
            )
            .with_optimizer(Fixed)
            .factor(&x, 3, 1, &y)
            .map_err(|(_, err)| err),
        );
        let dir_double = temp_dir("prec-double");
        must(double.save_with_factor(&dir_double));
        let double_json = match std::fs::read_to_string(dir_double.join(CONFIG_FILE)) {
            Ok(text) => text,
            Err(err) => panic!("read config failed: {err}"),
        };
        assert!(!double_json.contains("\"precision\""), "{double_json}");
        let _ = std::fs::remove_dir_all(&dir_double);

        let dir_single = temp_dir("prec-single");
        must(single.save_with_factor(&dir_single));
        let single_json = match std::fs::read_to_string(dir_single.join(CONFIG_FILE)) {
            Ok(text) => text,
            Err(err) => panic!("read config failed: {err}"),
        };
        assert!(
            single_json.contains("\"precision\": \"single\""),
            "{single_json}"
        );
        let want = must(single.predict(&[0.25], 1, 1));
        let loaded = must(LoadedGpr::load(&dir_single, &PersistRegistry::new()));
        let LoadedGpr::Single(model) = loaded else {
            panic!("single RBF loads as CachedSingle");
        };
        let got = must(model.predict(&[0.25], 1, 1));
        assert_close(f64::from(got.mean[0]), f64::from(want.mean[0]), TOL);
        assert_close(f64::from(got.variance[0]), f64::from(want.variance[0]), TOL);
        let _ = std::fs::remove_dir_all(&dir_single);

        let online = must(single.into_online());
        let dir_online = temp_dir("prec-online-single");
        must(online.save_with_factor(&dir_online));
        let want = must(online.predict(&[0.25], 1, 1));
        let loaded = must(LoadedGpr::load(&dir_online, &PersistRegistry::new()));
        let LoadedGpr::OnlineSingle(model) = loaded else {
            panic!("single online RBF loads as CachedSingle");
        };
        let got = must(model.predict(&[0.25], 1, 1));
        assert_close(f64::from(got.mean[0]), f64::from(want.mean[0]), TOL);
        assert_close(f64::from(got.variance[0]), f64::from(want.variance[0]), TOL);
        let _ = std::fs::remove_dir_all(&dir_online);

        let dir_mixed = temp_dir("prec-mixed");
        must(mixed.save_with_factor(&dir_mixed));
        let mixed_json = match std::fs::read_to_string(dir_mixed.join(CONFIG_FILE)) {
            Ok(text) => text,
            Err(err) => panic!("read config failed: {err}"),
        };
        assert!(
            mixed_json.contains("\"precision\": \"mixed\""),
            "{mixed_json}"
        );
        assert!(!mixed_json.contains("\"residual\""), "{mixed_json}");
        let want = must(mixed.predict(&[0.25], 1, 1));
        let loaded = must(LoadedGpr::load(&dir_mixed, &PersistRegistry::new()));
        let LoadedGpr::Mixed(model) = loaded else {
            panic!("mixed RBF loads as CachedMixed");
        };
        let got = must(model.predict(&[0.25], 1, 1));
        assert_close(got.mean[0], want.mean[0], TOL);
        assert_close(got.variance[0], want.variance[0], TOL);
        let _ = std::fs::remove_dir_all(&dir_mixed);
    }

    /// Saves `fitted` as `llt` and as `ldlt`, each with and without the
    /// factor, and checks the precision-independent reads of [`LoadedGpr`]
    /// against the typed model.
    fn assert_loaded_reads<P: crate::precision::GpScalar>(
        label: &str,
        fitted: crate::FittedGpr<Fixed, P>,
        is_fitted: fn(&LoadedGpr) -> bool,
        is_online: fn(&LoadedGpr) -> bool,
    ) {
        let xs = [0.25, 0.8];
        let widen = |pred: crate::Prediction<P::Refine>| {
            let mean: Vec<f64> = pred
                .mean
                .iter()
                .map(|v| crate::kernel::KernelScalar::to_f64(*v))
                .collect();
            let variance: Vec<f64> = pred
                .variance
                .iter()
                .map(|v| crate::kernel::KernelScalar::to_f64(*v))
                .collect();
            (mean, variance)
        };
        let online = fitted.clone().into_online().expect("online");
        let want_fitted = widen(fitted.predict(&xs, 2, 1).expect("predict"));
        let want_online = widen(online.predict(&xs, 2, 1).expect("predict"));
        for with_factor in [true, false] {
            for ldlt in [false, true] {
                let dir = temp_dir(&format!("reads-{label}-{with_factor}-{ldlt}"));
                let saved = match (ldlt, with_factor) {
                    (false, true) => fitted.save_with_factor(&dir),
                    (false, false) => fitted.save(&dir),
                    (true, true) => online.save_with_factor(&dir),
                    (true, false) => online.save(&dir),
                };
                saved.expect("save");
                let loaded = LoadedGpr::load(&dir, &PersistRegistry::new()).expect("load");
                let (variant, want) = if ldlt {
                    (is_online(&loaded), &want_online)
                } else {
                    (is_fitted(&loaded), &want_fitted)
                };
                assert!(variant, "{label}: wrong variant {loaded:?}");
                assert_eq!(loaded.is_online(), ldlt, "{label}");
                assert_eq!((loaded.n(), loaded.d()), (3, 1), "{label}");
                let got = loaded.predict(&xs, 2, 1).expect("loaded predict");
                for i in 0..xs.len() {
                    assert_close(got.mean[i], want.0[i], 1e-5);
                    assert_close(got.variance[i], want.1[i], 1e-5);
                }
                let _ = std::fs::remove_dir_all(&dir);
            }
        }
    }

    #[test]
    fn loaded_reads_match_every_precision_and_factor() {
        let x = [0.0, 0.5, 1.0];
        let y = [0.0, 1.0, 0.2];
        let trainer = || {
            Gpr::new(
                KernelSpec::from(RbfKernel::new(1.0).expect("valid")),
                GaussianLikelihood::new(0.1).expect("valid"),
            )
            .with_optimizer(Fixed)
        };
        assert_loaded_reads(
            "double",
            trainer().factor(&x, 3, 1, &y).expect("spd"),
            |l| matches!(l, LoadedGpr::Double(_)),
            |l| matches!(l, LoadedGpr::OnlineDouble(_)),
        );
        assert_loaded_reads(
            "single",
            trainer()
                .with_precision::<crate::SinglePrecision>()
                .factor(&x, 3, 1, &y)
                .expect("spd"),
            |l| matches!(l, LoadedGpr::Single(_)),
            |l| matches!(l, LoadedGpr::OnlineSingle(_)),
        );
        assert_loaded_reads(
            "mixed",
            trainer()
                .with_precision::<crate::MixedPrecision>()
                .factor(&x, 3, 1, &y)
                .expect("spd"),
            |l| matches!(l, LoadedGpr::Mixed(_)),
            |l| matches!(l, LoadedGpr::OnlineMixed(_)),
        );
        assert_loaded_reads(
            "reevaluate",
            trainer()
                .with_precision::<crate::MixedPrecision<crate::ReevaluateKernel>>()
                .factor(&x, 3, 1, &y)
                .expect("spd"),
            |l| matches!(l, LoadedGpr::Reevaluate(_)),
            |l| matches!(l, LoadedGpr::OnlineReevaluate(_)),
        );
    }

    #[test]
    fn fast_approx_round_trip_and_missing_field_is_accurate() {
        fn must<T>(result: Result<T, GprError>) -> T {
            match result {
                Ok(value) => value,
                Err(err) => panic!("persist call failed: {err}"),
            }
        }
        let x = [0.0, 0.5, 1.0];
        let y = [0.0, 1.0, 0.2];
        let fitted = must(
            Gpr::new(
                KernelSpec::from(must(RbfKernel::new(1.0))),
                must(GaussianLikelihood::new(0.1)),
            )
            .with_math(crate::KernelExp::FastApprox)
            .with_optimizer(Fixed)
            .factor(&x, 3, 1, &y)
            .map_err(|(_, err)| err),
        );
        let dir = temp_dir("math-fast");
        must(fitted.save_with_factor(&dir));
        let json = match std::fs::read_to_string(dir.join(CONFIG_FILE)) {
            Ok(text) => text,
            Err(err) => panic!("read config failed: {err}"),
        };
        assert!(json.contains("\"math\": \"fast_approx\""), "{json}");
        let want = must(fitted.predict(&[0.25], 1, 1));
        let loaded = must(LoadedGpr::load(&dir, &PersistRegistry::new()));
        let LoadedGpr::Double(model) = loaded else {
            panic!("fast RBF loads as Double");
        };
        assert_eq!(model.math(), crate::KernelExp::FastApprox);
        let got = must(model.predict(&[0.25], 1, 1));
        assert_close(got.mean[0], want.mean[0], TOL);
        assert_close(got.variance[0], want.variance[0], TOL);
        let online = must(fitted.into_online());
        let dir_online = temp_dir("math-fast-online");
        must(online.save_with_factor(&dir_online));
        let loaded = must(LoadedGpr::load(&dir_online, &PersistRegistry::new()));
        let LoadedGpr::OnlineDouble(model) = loaded else {
            panic!("fast online RBF loads as OnlineDouble");
        };
        assert_eq!(model.math(), crate::KernelExp::FastApprox);
        let broken = json.replace("\"math\": \"fast_approx\"", "\"math\": \"nope\"");
        if broken == json {
            panic!("math field was not replaced");
        }
        std::fs::write(dir.join(CONFIG_FILE), broken).expect("write");
        match LoadedGpr::load(&dir, &PersistRegistry::new()) {
            Err(GprError::PersistFailed { .. }) => {}
            other => panic!("unknown math field should fail, got {other:?}"),
        }
        let accurate = must(
            Gpr::new(
                KernelSpec::from(must(RbfKernel::new(1.0))),
                must(GaussianLikelihood::new(0.1)),
            )
            .with_optimizer(Fixed)
            .factor(&x, 3, 1, &y)
            .map_err(|(_, err)| err),
        );
        let dir_accurate = temp_dir("math-accurate");
        must(accurate.save(&dir_accurate));
        let accurate_json = match std::fs::read_to_string(dir_accurate.join(CONFIG_FILE)) {
            Ok(text) => text,
            Err(err) => panic!("read config failed: {err}"),
        };
        assert!(!accurate_json.contains("\"math\""), "{accurate_json}");
        let loaded = must(LoadedGpr::load(&dir_accurate, &PersistRegistry::new()));
        let LoadedGpr::Double(model) = loaded else {
            panic!("accurate RBF loads as Double");
        };
        assert_eq!(model.math(), crate::KernelExp::Accurate);
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&dir_online);
        let _ = std::fs::remove_dir_all(&dir_accurate);
    }
}
