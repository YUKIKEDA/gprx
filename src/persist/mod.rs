//! Save and load a fitted GPR directory (`config.json` + `model.safetensors`).

mod config;
mod kernel;
mod registry;
mod tensors;
mod transform;

use std::path::Path;

use crate::error::GprError;
use crate::gpr::{DistanceCacheSlot, FittedGpr, OnlineGpr};
use crate::kernel::KernelSpec;
use crate::optimizer::{Fixed, FullRecompute};
use crate::transform::{TargetTransform, Transform, UnfittedTarget, UnfittedTransform};
use crate::{
    CachedDistances, GaussianLikelihood, JitterPolicy, NoDistanceCache, UncachedDistances,
};

use config::{
    DistanceCacheJson, FactorKind, JitterJson, LikelihoodJson, MathJson, ModelConfig,
    PrecisionJson, ResidualJson,
};
use kernel::KernelJson;
use tensors::{
    FactorBytes, pack_lower, read_alpha, read_matrix, read_scalars, read_xy, scalar_bytes,
    write_tensors,
};

pub(crate) use tensors::MappedTensors;
use transform::{
    encode_fitted_input, encode_fitted_target, encode_unfitted_input, encode_unfitted_target,
};

pub use registry::{
    FittedInputRestore, FittedTargetRestore, KernelRestore, PersistRegistry, UnfittedInputRestore,
    UnfittedTargetRestore,
};

/// `config.json` `format_version` written and read by this crate.
pub const FORMAT_VERSION: u32 = 1;

/// Prefix reserved for built-in persist tags. Caller `persist_id`s must not use it.
pub const RESERVED_PREFIX: &str = "gprx.";

pub(crate) const CONFIG_FILE: &str = "config.json";
pub(crate) const TENSOR_FILE: &str = "model.safetensors";

/// Prediction-only model loaded from a persist directory.
///
/// Distance-path models (trainers from [`crate::Gpr::new`]) are
/// [`Self::Distance`]. Standalone Linear / Constant / White models
/// (trainers from [`crate::Gpr::from_points`]) are [`Self::Points`].
/// Re-training is [`crate::FittedGpr::with_optimizer`] then
/// [`crate::FittedGpr::refit`]. The file does not store a solver or a
/// Cholesky buffer policy; load is always [`crate::RetainCholesky`].
///
/// # Examples
///
/// ```rust
/// use gprx::kernel::{KernelSpec, RbfKernel};
/// use gprx::persist::{LoadedDistance, LoadedGpr, PersistRegistry};
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
/// match loaded {
///     LoadedGpr::Distance(LoadedDistance::Cached(model)) => {
///         let pred = model.predict(&[0.5], 1, 1)?;
///         assert_eq!(pred.mean.len(), 1);
///     }
///     LoadedGpr::Distance(LoadedDistance::Uncached(_)) => {
///         panic!("default save is CachedDistances")
///     }
///     LoadedGpr::Points(_) => panic!("RBF is a distance kernel"),
///     LoadedGpr::OnlineDistance(_) | LoadedGpr::OnlinePoints(_) => {
///         panic!("fitted save is llt")
///     }
///     _ => panic!("default save is double precision"),
/// }
/// let _ = std::fs::remove_dir_all(&dir);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
pub enum LoadedGpr {
    /// Model whose trainer stored a [`crate::DistanceCachePolicy`].
    Distance(LoadedDistance),
    /// Model whose trainer was [`crate::Gpr::from_points`].
    Points(FittedGpr<Fixed, FullRecompute, NoDistanceCache>),
    /// [`crate::SinglePrecision`] model from [`crate::Gpr::from_points`].
    PointsSingle(
        FittedGpr<
            Fixed,
            FullRecompute,
            NoDistanceCache,
            crate::RetainCholesky,
            crate::Accurate,
            crate::SinglePrecision,
        >,
    ),
    /// [`crate::MixedPrecision`] model from [`crate::Gpr::from_points`].
    PointsMixed(
        FittedGpr<
            Fixed,
            FullRecompute,
            NoDistanceCache,
            crate::RetainCholesky,
            crate::Accurate,
            crate::MixedPrecision,
        >,
    ),
    /// Reevaluate mixed model from [`crate::Gpr::from_points`].
    PointsReevaluate(
        FittedGpr<
            Fixed,
            FullRecompute,
            NoDistanceCache,
            crate::RetainCholesky,
            crate::Accurate,
            crate::MixedPrecision<crate::ReevaluateKernel>,
        >,
    ),
    /// Online model whose trainer stored a [`crate::DistanceCachePolicy`].
    OnlineDistance(LoadedOnlineDistance),
    /// Online model whose trainer was [`crate::Gpr::from_points`].
    OnlinePoints(OnlineGpr<Fixed, FullRecompute, NoDistanceCache>),
    /// [`crate::SinglePrecision`] online model from [`crate::Gpr::from_points`].
    OnlinePointsSingle(
        OnlineGpr<
            Fixed,
            FullRecompute,
            NoDistanceCache,
            crate::RetainCholesky,
            crate::Accurate,
            crate::SinglePrecision,
        >,
    ),
    /// Promoted-storage mixed online model from [`crate::Gpr::from_points`].
    OnlinePointsMixed(
        OnlineGpr<
            Fixed,
            FullRecompute,
            NoDistanceCache,
            crate::RetainCholesky,
            crate::Accurate,
            crate::MixedPrecision,
        >,
    ),
    /// Reevaluate mixed online model from [`crate::Gpr::from_points`].
    OnlinePointsReevaluate(
        OnlineGpr<
            Fixed,
            FullRecompute,
            NoDistanceCache,
            crate::RetainCholesky,
            crate::Accurate,
            crate::MixedPrecision<crate::ReevaluateKernel>,
        >,
    ),
    /// [`crate::FastApprox`] model from [`crate::Gpr::from_points`].
    PointsFast(
        FittedGpr<Fixed, FullRecompute, NoDistanceCache, crate::RetainCholesky, crate::FastApprox>,
    ),
    /// [`crate::FastApprox`] and [`crate::SinglePrecision`] from [`crate::Gpr::from_points`].
    PointsSingleFast(
        FittedGpr<
            Fixed,
            FullRecompute,
            NoDistanceCache,
            crate::RetainCholesky,
            crate::FastApprox,
            crate::SinglePrecision,
        >,
    ),
    /// [`crate::FastApprox`] and promoted-storage [`crate::MixedPrecision`] from [`crate::Gpr::from_points`].
    PointsMixedFast(
        FittedGpr<
            Fixed,
            FullRecompute,
            NoDistanceCache,
            crate::RetainCholesky,
            crate::FastApprox,
            crate::MixedPrecision,
        >,
    ),
    /// [`crate::FastApprox`] and reevaluate mixed precision from [`crate::Gpr::from_points`].
    PointsReevaluateFast(
        FittedGpr<
            Fixed,
            FullRecompute,
            NoDistanceCache,
            crate::RetainCholesky,
            crate::FastApprox,
            crate::MixedPrecision<crate::ReevaluateKernel>,
        >,
    ),
    /// [`crate::FastApprox`] online model from [`crate::Gpr::from_points`].
    OnlinePointsFast(
        OnlineGpr<Fixed, FullRecompute, NoDistanceCache, crate::RetainCholesky, crate::FastApprox>,
    ),
    /// [`crate::FastApprox`] and [`crate::SinglePrecision`] online model from [`crate::Gpr::from_points`].
    OnlinePointsSingleFast(
        OnlineGpr<
            Fixed,
            FullRecompute,
            NoDistanceCache,
            crate::RetainCholesky,
            crate::FastApprox,
            crate::SinglePrecision,
        >,
    ),
    /// [`crate::FastApprox`] and promoted-storage mixed online model from [`crate::Gpr::from_points`].
    OnlinePointsMixedFast(
        OnlineGpr<
            Fixed,
            FullRecompute,
            NoDistanceCache,
            crate::RetainCholesky,
            crate::FastApprox,
            crate::MixedPrecision,
        >,
    ),
    /// [`crate::FastApprox`] and reevaluate mixed online model from [`crate::Gpr::from_points`].
    OnlinePointsReevaluateFast(
        OnlineGpr<
            Fixed,
            FullRecompute,
            NoDistanceCache,
            crate::RetainCholesky,
            crate::FastApprox,
            crate::MixedPrecision<crate::ReevaluateKernel>,
        >,
    ),
}

/// Distance-path model loaded as [`CachedDistances`] or [`UncachedDistances`].
#[derive(Clone, Debug)]
pub enum LoadedDistance {
    /// Trainer used [`CachedDistances`] (`always` in `config.json`).
    Cached(FittedGpr<Fixed, FullRecompute, CachedDistances>),
    /// Trainer used [`UncachedDistances`] (`never` in `config.json`).
    Uncached(FittedGpr<Fixed, FullRecompute, UncachedDistances>),
    /// [`crate::SinglePrecision`] with [`CachedDistances`].
    CachedSingle(
        FittedGpr<
            Fixed,
            FullRecompute,
            CachedDistances,
            crate::RetainCholesky,
            crate::Accurate,
            crate::SinglePrecision,
        >,
    ),
    /// [`crate::SinglePrecision`] with [`UncachedDistances`].
    UncachedSingle(
        FittedGpr<
            Fixed,
            FullRecompute,
            UncachedDistances,
            crate::RetainCholesky,
            crate::Accurate,
            crate::SinglePrecision,
        >,
    ),
    /// [`crate::MixedPrecision`] with the promoted-storage residual and [`CachedDistances`].
    CachedMixed(
        FittedGpr<
            Fixed,
            FullRecompute,
            CachedDistances,
            crate::RetainCholesky,
            crate::Accurate,
            crate::MixedPrecision,
        >,
    ),
    /// [`crate::MixedPrecision`] with the promoted-storage residual and [`UncachedDistances`].
    UncachedMixed(
        FittedGpr<
            Fixed,
            FullRecompute,
            UncachedDistances,
            crate::RetainCholesky,
            crate::Accurate,
            crate::MixedPrecision,
        >,
    ),
    /// [`crate::MixedPrecision`]`<`[`crate::ReevaluateKernel`]`>` with [`CachedDistances`].
    CachedReevaluate(
        FittedGpr<
            Fixed,
            FullRecompute,
            CachedDistances,
            crate::RetainCholesky,
            crate::Accurate,
            crate::MixedPrecision<crate::ReevaluateKernel>,
        >,
    ),
    /// [`crate::MixedPrecision`]`<`[`crate::ReevaluateKernel`]`>` with [`UncachedDistances`].
    UncachedReevaluate(
        FittedGpr<
            Fixed,
            FullRecompute,
            UncachedDistances,
            crate::RetainCholesky,
            crate::Accurate,
            crate::MixedPrecision<crate::ReevaluateKernel>,
        >,
    ),
    /// [`crate::FastApprox`] with [`CachedDistances`].
    CachedFast(
        FittedGpr<Fixed, FullRecompute, CachedDistances, crate::RetainCholesky, crate::FastApprox>,
    ),
    /// [`crate::FastApprox`] with [`UncachedDistances`].
    UncachedFast(
        FittedGpr<
            Fixed,
            FullRecompute,
            UncachedDistances,
            crate::RetainCholesky,
            crate::FastApprox,
        >,
    ),
    /// [`crate::FastApprox`] and [`crate::SinglePrecision`] with [`CachedDistances`].
    CachedSingleFast(
        FittedGpr<
            Fixed,
            FullRecompute,
            CachedDistances,
            crate::RetainCholesky,
            crate::FastApprox,
            crate::SinglePrecision,
        >,
    ),
    /// [`crate::FastApprox`] and [`crate::SinglePrecision`] with [`UncachedDistances`].
    UncachedSingleFast(
        FittedGpr<
            Fixed,
            FullRecompute,
            UncachedDistances,
            crate::RetainCholesky,
            crate::FastApprox,
            crate::SinglePrecision,
        >,
    ),
    /// [`crate::FastApprox`] and promoted-storage [`crate::MixedPrecision`] with [`CachedDistances`].
    CachedMixedFast(
        FittedGpr<
            Fixed,
            FullRecompute,
            CachedDistances,
            crate::RetainCholesky,
            crate::FastApprox,
            crate::MixedPrecision,
        >,
    ),
    /// [`crate::FastApprox`] and promoted-storage [`crate::MixedPrecision`] with [`UncachedDistances`].
    UncachedMixedFast(
        FittedGpr<
            Fixed,
            FullRecompute,
            UncachedDistances,
            crate::RetainCholesky,
            crate::FastApprox,
            crate::MixedPrecision,
        >,
    ),
    /// [`crate::FastApprox`] and reevaluate mixed precision with [`CachedDistances`].
    CachedReevaluateFast(
        FittedGpr<
            Fixed,
            FullRecompute,
            CachedDistances,
            crate::RetainCholesky,
            crate::FastApprox,
            crate::MixedPrecision<crate::ReevaluateKernel>,
        >,
    ),
    /// [`crate::FastApprox`] and reevaluate mixed precision with [`UncachedDistances`].
    UncachedReevaluateFast(
        FittedGpr<
            Fixed,
            FullRecompute,
            UncachedDistances,
            crate::RetainCholesky,
            crate::FastApprox,
            crate::MixedPrecision<crate::ReevaluateKernel>,
        >,
    ),
}

/// Distance-path [`crate::OnlineGpr`] loaded as [`CachedDistances`] or
/// [`UncachedDistances`].
#[derive(Clone, Debug)]
pub enum LoadedOnlineDistance {
    /// Trainer used [`CachedDistances`] (`always` in `config.json`).
    Cached(OnlineGpr<Fixed, FullRecompute, CachedDistances>),
    /// Trainer used [`UncachedDistances`] (`never` in `config.json`).
    Uncached(OnlineGpr<Fixed, FullRecompute, UncachedDistances>),
    /// [`crate::SinglePrecision`] with [`CachedDistances`].
    CachedSingle(
        OnlineGpr<
            Fixed,
            FullRecompute,
            CachedDistances,
            crate::RetainCholesky,
            crate::Accurate,
            crate::SinglePrecision,
        >,
    ),
    /// [`crate::SinglePrecision`] with [`UncachedDistances`].
    UncachedSingle(
        OnlineGpr<
            Fixed,
            FullRecompute,
            UncachedDistances,
            crate::RetainCholesky,
            crate::Accurate,
            crate::SinglePrecision,
        >,
    ),
    /// Promoted-storage mixed precision with [`CachedDistances`].
    CachedMixed(
        OnlineGpr<
            Fixed,
            FullRecompute,
            CachedDistances,
            crate::RetainCholesky,
            crate::Accurate,
            crate::MixedPrecision,
        >,
    ),
    /// Promoted-storage mixed precision with [`UncachedDistances`].
    UncachedMixed(
        OnlineGpr<
            Fixed,
            FullRecompute,
            UncachedDistances,
            crate::RetainCholesky,
            crate::Accurate,
            crate::MixedPrecision,
        >,
    ),
    /// Reevaluate mixed precision with [`CachedDistances`].
    CachedReevaluate(
        OnlineGpr<
            Fixed,
            FullRecompute,
            CachedDistances,
            crate::RetainCholesky,
            crate::Accurate,
            crate::MixedPrecision<crate::ReevaluateKernel>,
        >,
    ),
    /// Reevaluate mixed precision with [`UncachedDistances`].
    UncachedReevaluate(
        OnlineGpr<
            Fixed,
            FullRecompute,
            UncachedDistances,
            crate::RetainCholesky,
            crate::Accurate,
            crate::MixedPrecision<crate::ReevaluateKernel>,
        >,
    ),
    /// [`crate::FastApprox`] with [`CachedDistances`].
    CachedFast(
        OnlineGpr<Fixed, FullRecompute, CachedDistances, crate::RetainCholesky, crate::FastApprox>,
    ),
    /// [`crate::FastApprox`] with [`UncachedDistances`].
    UncachedFast(
        OnlineGpr<
            Fixed,
            FullRecompute,
            UncachedDistances,
            crate::RetainCholesky,
            crate::FastApprox,
        >,
    ),
    /// [`crate::FastApprox`] and [`crate::SinglePrecision`] with [`CachedDistances`].
    CachedSingleFast(
        OnlineGpr<
            Fixed,
            FullRecompute,
            CachedDistances,
            crate::RetainCholesky,
            crate::FastApprox,
            crate::SinglePrecision,
        >,
    ),
    /// [`crate::FastApprox`] and [`crate::SinglePrecision`] with [`UncachedDistances`].
    UncachedSingleFast(
        OnlineGpr<
            Fixed,
            FullRecompute,
            UncachedDistances,
            crate::RetainCholesky,
            crate::FastApprox,
            crate::SinglePrecision,
        >,
    ),
    /// [`crate::FastApprox`] and promoted-storage mixed precision with [`CachedDistances`].
    CachedMixedFast(
        OnlineGpr<
            Fixed,
            FullRecompute,
            CachedDistances,
            crate::RetainCholesky,
            crate::FastApprox,
            crate::MixedPrecision,
        >,
    ),
    /// [`crate::FastApprox`] and promoted-storage mixed precision with [`UncachedDistances`].
    UncachedMixedFast(
        OnlineGpr<
            Fixed,
            FullRecompute,
            UncachedDistances,
            crate::RetainCholesky,
            crate::FastApprox,
            crate::MixedPrecision,
        >,
    ),
    /// [`crate::FastApprox`] and reevaluate mixed precision with [`CachedDistances`].
    CachedReevaluateFast(
        OnlineGpr<
            Fixed,
            FullRecompute,
            CachedDistances,
            crate::RetainCholesky,
            crate::FastApprox,
            crate::MixedPrecision<crate::ReevaluateKernel>,
        >,
    ),
    /// [`crate::FastApprox`] and reevaluate mixed precision with [`UncachedDistances`].
    UncachedReevaluateFast(
        OnlineGpr<
            Fixed,
            FullRecompute,
            UncachedDistances,
            crate::RetainCholesky,
            crate::FastApprox,
            crate::MixedPrecision<crate::ReevaluateKernel>,
        >,
    ),
}

impl LoadedGpr {
    /// Reads `dir/config.json` and `dir/model.safetensors`.
    ///
    /// Reconstructs fitted transforms from the config and applies them to the
    /// stored original `X` / `y`. `factor_kind` is required: `llt` loads
    /// [`FittedGpr`], `ldlt` loads [`crate::OnlineGpr`] and requires
    /// `point_ids` plus `next_point_id`. When a factor is present, the
    /// safetensors file stays memory-mapped. The buffer policy is
    /// [`crate::RetainCholesky`]. Unknown [`FORMAT_VERSION`] is rejected.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::UnsupportedPersistVersion`] when `format_version`
    /// is not [`FORMAT_VERSION`], or [`GprError::PersistFailed`] when the
    /// directory, JSON, tensors, or registry lookup is invalid. Factorization
    /// errors from a file written without `L` use the same variants as
    /// [`crate::Gpr<Fixed>::factor`].
    pub fn load(dir: impl AsRef<Path>, registry: &PersistRegistry) -> Result<Self, GprError> {
        load_dir(dir.as_ref(), registry)
    }
}

pub(crate) struct PersistedModel<
    C,
    P: crate::precision::GpScalar = crate::precision::DoublePrecision,
> {
    pub kernel: KernelSpec,
    pub likelihood: GaussianLikelihood,
    pub x_unfitted: Box<dyn UnfittedTransform>,
    pub y_unfitted: Box<dyn UnfittedTarget>,
    pub x_transform: Box<dyn Transform>,
    pub y_transform: Box<dyn TargetTransform>,
    pub distance_cache: C,
    pub jitter_policy: JitterPolicy,
    pub x_obs: Vec<f64>,
    pub y_obs: Vec<f64>,
    pub alpha: Vec<P::Refine>,
    pub owned_l: Option<faer::Mat<P::Storage>>,
    pub mapped: Option<MappedTensors>,
}

pub(crate) fn persist_err(reason: impl Into<String>) -> GprError {
    GprError::PersistFailed {
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

pub(crate) fn save_fitted<O, S, C, B, M, P>(
    model: &FittedGpr<O, S, C, B, M, P>,
    dir: &Path,
    with_factor: bool,
) -> Result<(), GprError>
where
    C: DistanceCacheSlot,
    B: crate::gpr::AllocWorkspace,
    M: crate::math::KernelMath,
    P: crate::precision::GpScalar,
    crate::kernel::CompiledKernel<P::Storage>: crate::kernel::GramKernel<T = P::Storage>,
{
    std::fs::create_dir_all(dir).map_err(|err| persist_err(format!("create {dir:?}: {err}")))?;
    let kind = P::persist_kind();
    let config = ModelConfig {
        format_version: FORMAT_VERSION,
        n: model.n(),
        d: model.d(),
        has_factor: with_factor,
        factor_kind: FactorKind::Llt,
        precision: PrecisionJson::from_persist(kind),
        residual: ResidualJson::from_persist(kind),
        math: MathJson::from_math::<M>(),
        kernel: KernelJson::encode(model.kernel())?,
        likelihood: LikelihoodJson::encode(model.likelihood()),
        jitter: JitterJson::encode(model.jitter_policy()),
        distance_cache: model
            .distance_cache_slot()
            .persist()
            .map(DistanceCacheJson::encode),
        x_unfitted: encode_unfitted_input(model.x_unfitted())?,
        y_unfitted: encode_unfitted_target(model.y_unfitted())?,
        x_transform: encode_fitted_input(model.x_transform())?,
        y_transform: encode_fitted_target(model.y_transform())?,
        point_ids: None,
        next_point_id: None,
    };
    let config_path = dir.join(CONFIG_FILE);
    let json = serde_json::to_vec_pretty(&config)
        .map_err(|err| persist_err(format!("serialize config.json: {err}")))?;
    std::fs::write(&config_path, json)
        .map_err(|err| persist_err(format!("write {config_path:?}: {err}")))?;
    let packed = if with_factor {
        Some(pack_saved_factor(model.chol_l(), model.alpha())?)
    } else {
        None
    };
    let factor_refs = packed.as_ref().map(|packed| FactorBytes {
        l_dtype: packed.l_dtype,
        l: packed.l.as_slice(),
        alpha_dtype: packed.alpha_dtype,
        alpha: packed.alpha.as_slice(),
    });
    write_tensors(dir, model.x(), model.y(), model.n(), model.d(), factor_refs)
}

pub(crate) fn save_online<O, S, C, B, M, P>(
    model: &OnlineGpr<O, S, C, B, M, P>,
    dir: &Path,
    with_factor: bool,
) -> Result<(), GprError>
where
    C: DistanceCacheSlot,
    B: crate::gpr::AllocWorkspace,
    M: crate::math::KernelMath,
    P: crate::precision::GpScalar,
    crate::kernel::CompiledKernel<P::Storage>: crate::kernel::GramKernel<T = P::Storage>,
{
    std::fs::create_dir_all(dir).map_err(|err| persist_err(format!("create {dir:?}: {err}")))?;
    let kind = P::persist_kind();
    let config = ModelConfig {
        format_version: FORMAT_VERSION,
        n: model.n(),
        d: model.d(),
        has_factor: with_factor,
        factor_kind: FactorKind::Ldlt,
        precision: PrecisionJson::from_persist(kind),
        residual: ResidualJson::from_persist(kind),
        math: MathJson::from_math::<M>(),
        kernel: KernelJson::encode(model.kernel())?,
        likelihood: LikelihoodJson::encode(model.likelihood()),
        jitter: JitterJson::encode(model.jitter_policy()),
        distance_cache: model
            .distance_cache_slot()
            .persist()
            .map(DistanceCacheJson::encode),
        x_unfitted: encode_unfitted_input(model.x_unfitted())?,
        y_unfitted: encode_unfitted_target(model.y_unfitted())?,
        x_transform: encode_fitted_input(model.x_transform())?,
        y_transform: encode_fitted_target(model.y_transform())?,
        point_ids: Some(model.persist_point_ids()),
        next_point_id: Some(model.persist_next_point_id()),
    };
    let config_path = dir.join(CONFIG_FILE);
    let json = serde_json::to_vec_pretty(&config)
        .map_err(|err| persist_err(format!("serialize config.json: {err}")))?;
    std::fs::write(&config_path, json)
        .map_err(|err| persist_err(format!("write {config_path:?}: {err}")))?;
    let packed = if with_factor {
        Some(pack_saved_factor(model.ld_factor(), model.alpha())?)
    } else {
        None
    };
    let factor_refs = packed.as_ref().map(|packed| FactorBytes {
        l_dtype: packed.l_dtype,
        l: packed.l.as_slice(),
        alpha_dtype: packed.alpha_dtype,
        alpha: packed.alpha.as_slice(),
    });
    write_tensors(dir, model.x(), model.y(), model.n(), model.d(), factor_refs)
}

fn apply_online_ids<O, S, C, B, M, P>(
    online: &mut OnlineGpr<O, S, C, B, M, P>,
    ids: &Option<(Vec<u64>, u64)>,
) -> Result<(), GprError>
where
    C: DistanceCacheSlot,
    B: crate::gpr::AllocWorkspace,
    M: crate::math::KernelMath,
    P: crate::precision::GpScalar,
    crate::kernel::CompiledKernel<P::Storage>: crate::kernel::GramKernel<T = P::Storage>,
{
    let (ids, next_id) = ids
        .as_ref()
        .ok_or_else(|| persist_err("ldlt config missing point_ids"))?;
    online.apply_persisted_ids(ids, *next_id)
}

#[allow(clippy::type_complexity)]
trait PersistLoad: crate::precision::GpScalar {
    fn read_alpha(dir: &Path, n: usize) -> Result<Vec<Self::Refine>, GprError>;
    fn read_factor(
        dir: &Path,
        n: usize,
    ) -> Result<(Option<faer::Mat<Self::Storage>>, Option<MappedTensors>), GprError>;
}

impl PersistLoad for crate::precision::DoublePrecision {
    fn read_alpha(dir: &Path, n: usize) -> Result<Vec<Self::Refine>, GprError> {
        read_alpha(dir, n)
    }
    fn read_factor(
        dir: &Path,
        n: usize,
    ) -> Result<(Option<faer::Mat<Self::Storage>>, Option<MappedTensors>), GprError> {
        Ok((None, Some(MappedTensors::open(dir, n)?)))
    }
}

impl PersistLoad for crate::SinglePrecision {
    fn read_alpha(dir: &Path, n: usize) -> Result<Vec<Self::Refine>, GprError> {
        read_scalars::<f32>(
            dir,
            tensors::TENSOR_ALPHA,
            &[n],
            safetensors::tensor::Dtype::F32,
        )
    }
    fn read_factor(
        dir: &Path,
        n: usize,
    ) -> Result<(Option<faer::Mat<Self::Storage>>, Option<MappedTensors>), GprError> {
        Ok((
            Some(read_matrix::<f32>(dir, n, safetensors::tensor::Dtype::F32)?),
            None,
        ))
    }
}

impl PersistLoad for crate::MixedPrecision<crate::precision::PromoteStorage> {
    fn read_alpha(dir: &Path, n: usize) -> Result<Vec<Self::Refine>, GprError> {
        read_alpha(dir, n)
    }
    fn read_factor(
        dir: &Path,
        n: usize,
    ) -> Result<(Option<faer::Mat<Self::Storage>>, Option<MappedTensors>), GprError> {
        Ok((
            Some(read_matrix::<f32>(dir, n, safetensors::tensor::Dtype::F32)?),
            None,
        ))
    }
}

impl PersistLoad for crate::MixedPrecision<crate::ReevaluateKernel> {
    fn read_alpha(dir: &Path, n: usize) -> Result<Vec<Self::Refine>, GprError> {
        read_alpha(dir, n)
    }
    fn read_factor(
        dir: &Path,
        n: usize,
    ) -> Result<(Option<faer::Mat<Self::Storage>>, Option<MappedTensors>), GprError> {
        Ok((
            Some(read_matrix::<f32>(dir, n, safetensors::tensor::Dtype::F32)?),
            None,
        ))
    }
}

#[allow(private_bounds)]
trait Seal<M: crate::math::KernelMath>: crate::precision::GpScalar {
    fn seal_cached(
        model: FittedGpr<Fixed, FullRecompute, CachedDistances, crate::RetainCholesky, M, Self>,
    ) -> LoadedGpr;
    fn seal_uncached(
        model: FittedGpr<Fixed, FullRecompute, UncachedDistances, crate::RetainCholesky, M, Self>,
    ) -> LoadedGpr;
    fn seal_points(
        model: FittedGpr<Fixed, FullRecompute, NoDistanceCache, crate::RetainCholesky, M, Self>,
    ) -> LoadedGpr;
    fn seal_online_cached(
        model: OnlineGpr<Fixed, FullRecompute, CachedDistances, crate::RetainCholesky, M, Self>,
    ) -> LoadedGpr;
    fn seal_online_uncached(
        model: OnlineGpr<Fixed, FullRecompute, UncachedDistances, crate::RetainCholesky, M, Self>,
    ) -> LoadedGpr;
    fn seal_online_points(
        model: OnlineGpr<Fixed, FullRecompute, NoDistanceCache, crate::RetainCholesky, M, Self>,
    ) -> LoadedGpr;
}

macro_rules! impl_seal {
    (
        $prec:ty, $math:ty,
        $cached:path, $uncached:path, $points:path,
        $online_cached:path, $online_uncached:path, $online_points:path
    ) => {
        impl Seal<$math> for $prec {
            fn seal_cached(
                model: FittedGpr<
                    Fixed,
                    FullRecompute,
                    CachedDistances,
                    crate::RetainCholesky,
                    $math,
                    Self,
                >,
            ) -> LoadedGpr {
                LoadedGpr::Distance($cached(model))
            }
            fn seal_uncached(
                model: FittedGpr<
                    Fixed,
                    FullRecompute,
                    UncachedDistances,
                    crate::RetainCholesky,
                    $math,
                    Self,
                >,
            ) -> LoadedGpr {
                LoadedGpr::Distance($uncached(model))
            }
            fn seal_points(
                model: FittedGpr<
                    Fixed,
                    FullRecompute,
                    NoDistanceCache,
                    crate::RetainCholesky,
                    $math,
                    Self,
                >,
            ) -> LoadedGpr {
                $points(model)
            }
            fn seal_online_cached(
                model: OnlineGpr<
                    Fixed,
                    FullRecompute,
                    CachedDistances,
                    crate::RetainCholesky,
                    $math,
                    Self,
                >,
            ) -> LoadedGpr {
                LoadedGpr::OnlineDistance($online_cached(model))
            }
            fn seal_online_uncached(
                model: OnlineGpr<
                    Fixed,
                    FullRecompute,
                    UncachedDistances,
                    crate::RetainCholesky,
                    $math,
                    Self,
                >,
            ) -> LoadedGpr {
                LoadedGpr::OnlineDistance($online_uncached(model))
            }
            fn seal_online_points(
                model: OnlineGpr<
                    Fixed,
                    FullRecompute,
                    NoDistanceCache,
                    crate::RetainCholesky,
                    $math,
                    Self,
                >,
            ) -> LoadedGpr {
                $online_points(model)
            }
        }
    };
}

impl_seal!(
    crate::precision::DoublePrecision,
    crate::Accurate,
    LoadedDistance::Cached,
    LoadedDistance::Uncached,
    LoadedGpr::Points,
    LoadedOnlineDistance::Cached,
    LoadedOnlineDistance::Uncached,
    LoadedGpr::OnlinePoints
);
impl_seal!(
    crate::SinglePrecision,
    crate::Accurate,
    LoadedDistance::CachedSingle,
    LoadedDistance::UncachedSingle,
    LoadedGpr::PointsSingle,
    LoadedOnlineDistance::CachedSingle,
    LoadedOnlineDistance::UncachedSingle,
    LoadedGpr::OnlinePointsSingle
);
impl_seal!(
    crate::MixedPrecision<crate::precision::PromoteStorage>,
    crate::Accurate,
    LoadedDistance::CachedMixed,
    LoadedDistance::UncachedMixed,
    LoadedGpr::PointsMixed,
    LoadedOnlineDistance::CachedMixed,
    LoadedOnlineDistance::UncachedMixed,
    LoadedGpr::OnlinePointsMixed
);
impl_seal!(
    crate::MixedPrecision<crate::ReevaluateKernel>,
    crate::Accurate,
    LoadedDistance::CachedReevaluate,
    LoadedDistance::UncachedReevaluate,
    LoadedGpr::PointsReevaluate,
    LoadedOnlineDistance::CachedReevaluate,
    LoadedOnlineDistance::UncachedReevaluate,
    LoadedGpr::OnlinePointsReevaluate
);
impl_seal!(
    crate::precision::DoublePrecision,
    crate::FastApprox,
    LoadedDistance::CachedFast,
    LoadedDistance::UncachedFast,
    LoadedGpr::PointsFast,
    LoadedOnlineDistance::CachedFast,
    LoadedOnlineDistance::UncachedFast,
    LoadedGpr::OnlinePointsFast
);
impl_seal!(
    crate::SinglePrecision,
    crate::FastApprox,
    LoadedDistance::CachedSingleFast,
    LoadedDistance::UncachedSingleFast,
    LoadedGpr::PointsSingleFast,
    LoadedOnlineDistance::CachedSingleFast,
    LoadedOnlineDistance::UncachedSingleFast,
    LoadedGpr::OnlinePointsSingleFast
);
impl_seal!(
    crate::MixedPrecision<crate::precision::PromoteStorage>,
    crate::FastApprox,
    LoadedDistance::CachedMixedFast,
    LoadedDistance::UncachedMixedFast,
    LoadedGpr::PointsMixedFast,
    LoadedOnlineDistance::CachedMixedFast,
    LoadedOnlineDistance::UncachedMixedFast,
    LoadedGpr::OnlinePointsMixedFast
);
impl_seal!(
    crate::MixedPrecision<crate::ReevaluateKernel>,
    crate::FastApprox,
    LoadedDistance::CachedReevaluateFast,
    LoadedDistance::UncachedReevaluateFast,
    LoadedGpr::PointsReevaluateFast,
    LoadedOnlineDistance::CachedReevaluateFast,
    LoadedOnlineDistance::UncachedReevaluateFast,
    LoadedGpr::OnlinePointsReevaluateFast
);

fn load_dir(dir: &Path, registry: &PersistRegistry) -> Result<LoadedGpr, GprError> {
    let config_path = dir.join(CONFIG_FILE);
    let bytes = std::fs::read(&config_path)
        .map_err(|err| persist_err(format!("read {config_path:?}: {err}")))?;
    let config = config::parse_config(&bytes)?;
    match (config.precision, config.residual) {
        (PrecisionJson::Double, _) => {
            load_math::<crate::precision::DoublePrecision>(dir, registry, config)
        }
        (PrecisionJson::Single, _) => load_math::<crate::SinglePrecision>(dir, registry, config),
        (PrecisionJson::Mixed, ResidualJson::PromoteStorage) => load_math::<
            crate::MixedPrecision<crate::precision::PromoteStorage>,
        >(dir, registry, config),
        (PrecisionJson::Mixed, ResidualJson::ReevaluateKernel) => {
            load_math::<crate::MixedPrecision<crate::ReevaluateKernel>>(dir, registry, config)
        }
    }
}

fn load_math<P>(
    dir: &Path,
    registry: &PersistRegistry,
    config: ModelConfig,
) -> Result<LoadedGpr, GprError>
where
    P: PersistLoad + Seal<crate::Accurate> + Seal<crate::FastApprox>,
    crate::kernel::CompiledKernel<P::Storage>: crate::kernel::GramKernel<T = P::Storage>,
{
    match config.math {
        MathJson::Accurate => load_precision::<P, crate::Accurate>(dir, registry, config),
        MathJson::FastApprox => load_precision::<P, crate::FastApprox>(dir, registry, config),
    }
}

fn load_precision<P, M>(
    dir: &Path,
    registry: &PersistRegistry,
    config: ModelConfig,
) -> Result<LoadedGpr, GprError>
where
    P: PersistLoad + Seal<M>,
    M: crate::math::KernelMath,
    crate::kernel::CompiledKernel<P::Storage>: crate::kernel::GramKernel<T = P::Storage>,
{
    let ldlt_ids = match config.factor_kind {
        FactorKind::Ldlt => {
            let (ids, next_id) = config.online_ids()?;
            Some((ids.to_vec(), next_id))
        }
        FactorKind::Llt => None,
    };
    let kernel = config.kernel.decode(registry)?;
    let likelihood = config.likelihood.decode()?;
    let jitter = config.jitter.decode()?;
    let x_unfitted = config.x_unfitted.decode(registry)?;
    let y_unfitted = config.y_unfitted.decode(registry)?;
    let x_transform = config.x_transform.decode(registry)?;
    let y_transform = config.y_transform.decode(registry)?;
    let (x_obs, y_obs) = read_xy(dir, config.n, config.d)?;
    let cache = config.distance_cache.map(DistanceCacheJson::decode);
    if config.has_factor {
        let alpha = P::read_alpha(dir, config.n)?;
        let (owned_l, mapped) = P::read_factor(dir, config.n)?;
        match (config.factor_kind, cache) {
            (FactorKind::Llt, Some(crate::gpr::DistanceCachePersist::Cached)) => {
                let model = FittedGpr::from_persisted(PersistedModel {
                    kernel,
                    likelihood,
                    x_unfitted,
                    y_unfitted,
                    x_transform,
                    y_transform,
                    distance_cache: CachedDistances,
                    jitter_policy: jitter,
                    x_obs,
                    y_obs,
                    alpha,
                    owned_l,
                    mapped,
                })?;
                Ok(P::seal_cached(model))
            }
            (FactorKind::Llt, Some(crate::gpr::DistanceCachePersist::Uncached)) => {
                let model = FittedGpr::from_persisted(PersistedModel {
                    kernel,
                    likelihood,
                    x_unfitted,
                    y_unfitted,
                    x_transform,
                    y_transform,
                    distance_cache: UncachedDistances,
                    jitter_policy: jitter,
                    x_obs,
                    y_obs,
                    alpha,
                    owned_l,
                    mapped,
                })?;
                Ok(P::seal_uncached(model))
            }
            (FactorKind::Llt, None) => {
                let model = FittedGpr::from_persisted(PersistedModel {
                    kernel,
                    likelihood,
                    x_unfitted,
                    y_unfitted,
                    x_transform,
                    y_transform,
                    distance_cache: NoDistanceCache,
                    jitter_policy: jitter,
                    x_obs,
                    y_obs,
                    alpha,
                    owned_l,
                    mapped,
                })?;
                Ok(P::seal_points(model))
            }
            (FactorKind::Ldlt, Some(crate::gpr::DistanceCachePersist::Cached)) => {
                let mut online = OnlineGpr::from_persisted(PersistedModel {
                    kernel,
                    likelihood,
                    x_unfitted,
                    y_unfitted,
                    x_transform,
                    y_transform,
                    distance_cache: CachedDistances,
                    jitter_policy: jitter,
                    x_obs,
                    y_obs,
                    alpha,
                    owned_l,
                    mapped,
                })?;
                apply_online_ids(&mut online, &ldlt_ids)?;
                Ok(P::seal_online_cached(online))
            }
            (FactorKind::Ldlt, Some(crate::gpr::DistanceCachePersist::Uncached)) => {
                let mut online = OnlineGpr::from_persisted(PersistedModel {
                    kernel,
                    likelihood,
                    x_unfitted,
                    y_unfitted,
                    x_transform,
                    y_transform,
                    distance_cache: UncachedDistances,
                    jitter_policy: jitter,
                    x_obs,
                    y_obs,
                    alpha,
                    owned_l,
                    mapped,
                })?;
                apply_online_ids(&mut online, &ldlt_ids)?;
                Ok(P::seal_online_uncached(online))
            }
            (FactorKind::Ldlt, None) => {
                let mut online = OnlineGpr::from_persisted(PersistedModel {
                    kernel,
                    likelihood,
                    x_unfitted,
                    y_unfitted,
                    x_transform,
                    y_transform,
                    distance_cache: NoDistanceCache,
                    jitter_policy: jitter,
                    x_obs,
                    y_obs,
                    alpha,
                    owned_l,
                    mapped,
                })?;
                apply_online_ids(&mut online, &ldlt_ids)?;
                Ok(P::seal_online_points(online))
            }
        }
    } else {
        match (config.factor_kind, cache) {
            (FactorKind::Llt, Some(crate::gpr::DistanceCachePersist::Cached)) => {
                let gpr = crate::Gpr::new(kernel, likelihood)
                    .with_optimizer(Fixed)
                    .with_boxed_input_transform(x_unfitted)
                    .with_boxed_target_transform(y_unfitted)
                    .with_jitter_policy(jitter)
                    .with_distance_cache_policy(CachedDistances)
                    .with_math::<M>()
                    .with_precision::<P>();
                Ok(P::seal_cached(
                    gpr.factor(&x_obs, config.n, config.d, &y_obs)
                        .map_err(|(_, err)| err)?,
                ))
            }
            (FactorKind::Llt, Some(crate::gpr::DistanceCachePersist::Uncached)) => {
                let gpr = crate::Gpr::new(kernel, likelihood)
                    .with_optimizer(Fixed)
                    .with_boxed_input_transform(x_unfitted)
                    .with_boxed_target_transform(y_unfitted)
                    .with_jitter_policy(jitter)
                    .with_distance_cache_policy(UncachedDistances)
                    .with_math::<M>()
                    .with_precision::<P>();
                Ok(P::seal_uncached(
                    gpr.factor(&x_obs, config.n, config.d, &y_obs)
                        .map_err(|(_, err)| err)?,
                ))
            }
            (FactorKind::Llt, None) => {
                let gpr = crate::Gpr::from_points(kernel, likelihood)
                    .with_optimizer(Fixed)
                    .with_boxed_input_transform(x_unfitted)
                    .with_boxed_target_transform(y_unfitted)
                    .with_jitter_policy(jitter)
                    .with_math::<M>()
                    .with_precision::<P>();
                Ok(P::seal_points(
                    gpr.factor(&x_obs, config.n, config.d, &y_obs)
                        .map_err(|(_, err)| err)?,
                ))
            }
            (FactorKind::Ldlt, Some(crate::gpr::DistanceCachePersist::Cached)) => {
                let gpr = crate::Gpr::new(kernel, likelihood)
                    .with_optimizer(Fixed)
                    .with_boxed_input_transform(x_unfitted)
                    .with_boxed_target_transform(y_unfitted)
                    .with_jitter_policy(jitter)
                    .with_distance_cache_policy(CachedDistances)
                    .with_math::<M>()
                    .with_precision::<P>();
                let mut online = gpr
                    .factor(&x_obs, config.n, config.d, &y_obs)
                    .map_err(|(_, err)| err)?
                    .into_online()?;
                apply_online_ids(&mut online, &ldlt_ids)?;
                Ok(P::seal_online_cached(online))
            }
            (FactorKind::Ldlt, Some(crate::gpr::DistanceCachePersist::Uncached)) => {
                let gpr = crate::Gpr::new(kernel, likelihood)
                    .with_optimizer(Fixed)
                    .with_boxed_input_transform(x_unfitted)
                    .with_boxed_target_transform(y_unfitted)
                    .with_jitter_policy(jitter)
                    .with_distance_cache_policy(UncachedDistances)
                    .with_math::<M>()
                    .with_precision::<P>();
                let mut online = gpr
                    .factor(&x_obs, config.n, config.d, &y_obs)
                    .map_err(|(_, err)| err)?
                    .into_online()?;
                apply_online_ids(&mut online, &ldlt_ids)?;
                Ok(P::seal_online_uncached(online))
            }
            (FactorKind::Ldlt, None) => {
                let gpr = crate::Gpr::from_points(kernel, likelihood)
                    .with_optimizer(Fixed)
                    .with_boxed_input_transform(x_unfitted)
                    .with_boxed_target_transform(y_unfitted)
                    .with_jitter_policy(jitter)
                    .with_math::<M>()
                    .with_precision::<P>();
                let mut online = gpr
                    .factor(&x_obs, config.n, config.d, &y_obs)
                    .map_err(|(_, err)| err)?
                    .into_online()?;
                apply_online_ids(&mut online, &ldlt_ids)?;
                Ok(P::seal_online_points(online))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CONFIG_FILE, FORMAT_VERSION, LoadedDistance, LoadedGpr, LoadedOnlineDistance,
        PersistRegistry, RESERVED_PREFIX, persist_err,
    };
    use crate::kernel::{KernelSpec, KernelTerm, LinearKernel, RbfKernel, Triangle};
    use crate::param::Interval;
    use crate::transform::{StandardizeInput, StandardizeTarget};
    use crate::{Fixed, GaussianLikelihood, Gpr, GprError, Lbfgs};
    use faer::{MatMut, MatRef};
    use std::path::PathBuf;

    const TOL: f64 = 1e-10;

    fn assert_close(actual: f64, expected: f64) {
        let scale = expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= TOL * scale,
            "actual={actual}, expected={expected}"
        );
    }

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
                Err(GprError::InvalidHyperparameter {
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
                Err(GprError::InvalidHyperparameter {
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
            Err(GprError::InvalidHyperparameter {
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
            Err(GprError::InvalidHyperparameter {
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
            Err(GprError::InvalidHyperparameter {
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
        match persist_err("missing l") {
            GprError::PersistFailed { reason } => assert_eq!(reason, "missing l"),
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
        let LoadedGpr::Distance(LoadedDistance::Cached(model)) = loaded else {
            panic!("RBF is a distance kernel");
        };
        let got = model.predict(&[0.5], 1, 1).expect("loaded predict");
        assert_close(got.mean[0], want.mean[0]);
        assert_close(got.variance[0], want.variance[0]);
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
        let LoadedGpr::Distance(LoadedDistance::Cached(model)) = loaded else {
            panic!("RBF is a distance kernel");
        };
        let got = model.predict(&[0.25], 1, 1).expect("loaded predict");
        assert_close(got.mean[0], want.mean[0]);
        assert_close(got.variance[0], want.variance[0]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn linear_from_points_loads_as_points() {
        let fitted = Gpr::from_points(
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
        let LoadedGpr::Points(model) = loaded else {
            panic!("standalone Linear is a points kernel");
        };
        let got = model.predict(&[0.5], 1, 1).expect("loaded predict");
        assert_close(got.mean[0], want.mean[0]);
        assert_close(got.variance[0], want.variance[0]);
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
        let LoadedGpr::Distance(LoadedDistance::Cached(model)) = loaded else {
            panic!("RBF is a distance kernel");
        };
        let got = model.predict(&[1.0], 1, 1).expect("loaded predict");
        assert_close(got.mean[0], want.mean[0]);
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
            GprError::PersistFailed { reason } => {
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
            Err(GprError::PersistFailed { reason }) => {
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
        assert!(matches!(loaded, LoadedGpr::Distance(_)));
        let _ = std::fs::remove_dir_all(&dir);
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
        let LoadedGpr::Distance(LoadedDistance::CachedSingle(model)) = loaded else {
            panic!("single RBF loads as CachedSingle");
        };
        let got = must(model.predict(&[0.25], 1, 1));
        assert_close(f64::from(got.mean[0]), f64::from(want.mean[0]));
        assert_close(f64::from(got.variance[0]), f64::from(want.variance[0]));
        let _ = std::fs::remove_dir_all(&dir_single);

        let online = must(single.into_online());
        let dir_online = temp_dir("prec-online-single");
        must(online.save_with_factor(&dir_online));
        let want = must(online.predict(&[0.25], 1, 1));
        let loaded = must(LoadedGpr::load(&dir_online, &PersistRegistry::new()));
        let LoadedGpr::OnlineDistance(LoadedOnlineDistance::CachedSingle(model)) = loaded else {
            panic!("single online RBF loads as CachedSingle");
        };
        let got = must(model.predict(&[0.25], 1, 1));
        assert_close(f64::from(got.mean[0]), f64::from(want.mean[0]));
        assert_close(f64::from(got.variance[0]), f64::from(want.variance[0]));
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
        let LoadedGpr::Distance(LoadedDistance::CachedMixed(model)) = loaded else {
            panic!("mixed RBF loads as CachedMixed");
        };
        let got = must(model.predict(&[0.25], 1, 1));
        assert_close(got.mean[0], want.mean[0]);
        assert_close(got.variance[0], want.variance[0]);
        let _ = std::fs::remove_dir_all(&dir_mixed);
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
            .with_math::<crate::FastApprox>()
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
        let LoadedGpr::Distance(LoadedDistance::CachedFast(model)) = loaded else {
            panic!("fast RBF loads as CachedFast");
        };
        let got = must(model.predict(&[0.25], 1, 1));
        assert_close(got.mean[0], want.mean[0]);
        assert_close(got.variance[0], want.variance[0]);
        let online = must(fitted.into_online());
        let dir_online = temp_dir("math-fast-online");
        must(online.save_with_factor(&dir_online));
        let loaded = must(LoadedGpr::load(&dir_online, &PersistRegistry::new()));
        let LoadedGpr::OnlineDistance(LoadedOnlineDistance::CachedFast(_)) = loaded else {
            panic!("fast online RBF loads as CachedFast");
        };
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
        assert!(matches!(
            loaded,
            LoadedGpr::Distance(LoadedDistance::Cached(_))
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&dir_online);
        let _ = std::fs::remove_dir_all(&dir_accurate);
    }
}
