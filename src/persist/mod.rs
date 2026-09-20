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

use config::{DistanceCacheJson, FactorKind, JitterJson, LikelihoodJson, ModelConfig};
use kernel::KernelJson;
use tensors::{pack_lower_l, read_alpha, read_xy, write_tensors};

pub(crate) use tensors::{MappedTensors, copy_l_into};
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
    /// Online model whose trainer stored a [`crate::DistanceCachePolicy`].
    OnlineDistance(LoadedOnlineDistance),
    /// Online model whose trainer was [`crate::Gpr::from_points`].
    OnlinePoints(OnlineGpr<Fixed, FullRecompute, NoDistanceCache>),
}

/// Distance-path model loaded as [`CachedDistances`] or [`UncachedDistances`].
#[derive(Clone, Debug)]
pub enum LoadedDistance {
    /// Trainer used [`CachedDistances`] (`always` in `config.json`).
    Cached(FittedGpr<Fixed, FullRecompute, CachedDistances>),
    /// Trainer used [`UncachedDistances`] (`never` in `config.json`).
    Uncached(FittedGpr<Fixed, FullRecompute, UncachedDistances>),
}

/// Distance-path [`crate::OnlineGpr`] loaded as [`CachedDistances`] or
/// [`UncachedDistances`].
#[derive(Clone, Debug)]
pub enum LoadedOnlineDistance {
    /// Trainer used [`CachedDistances`] (`always` in `config.json`).
    Cached(OnlineGpr<Fixed, FullRecompute, CachedDistances>),
    /// Trainer used [`UncachedDistances`] (`never` in `config.json`).
    Uncached(OnlineGpr<Fixed, FullRecompute, UncachedDistances>),
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

pub(crate) struct PersistedModel<C> {
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
    pub alpha: Vec<f64>,
    pub mapped: Option<MappedTensors>,
}

pub(crate) fn persist_err(reason: impl Into<String>) -> GprError {
    GprError::PersistFailed {
        reason: reason.into(),
    }
}

pub(crate) fn save_fitted<O, S, C, B>(
    model: &FittedGpr<O, S, C, B>,
    dir: &Path,
    with_factor: bool,
) -> Result<(), GprError>
where
    C: DistanceCacheSlot,
    B: crate::gpr::AllocWorkspace,
{
    std::fs::create_dir_all(dir).map_err(|err| persist_err(format!("create {dir:?}: {err}")))?;
    let config = ModelConfig {
        format_version: FORMAT_VERSION,
        n: model.n(),
        d: model.d(),
        has_factor: with_factor,
        factor_kind: FactorKind::Llt,
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
    let factor = if with_factor {
        let n = model.n();
        let mut l = vec![0.0; n * n];
        pack_lower_l(model.chol_l(), &mut l);
        Some((l, model.alpha().to_vec()))
    } else {
        None
    };
    let factor_refs = factor
        .as_ref()
        .map(|(l, alpha)| (l.as_slice(), alpha.as_slice()));
    write_tensors(dir, model.x(), model.y(), model.n(), model.d(), factor_refs)
}

pub(crate) fn save_online<O, S, C, B>(
    model: &OnlineGpr<O, S, C, B>,
    dir: &Path,
    with_factor: bool,
) -> Result<(), GprError>
where
    C: DistanceCacheSlot,
    B: crate::gpr::AllocWorkspace,
{
    std::fs::create_dir_all(dir).map_err(|err| persist_err(format!("create {dir:?}: {err}")))?;
    let config = ModelConfig {
        format_version: FORMAT_VERSION,
        n: model.n(),
        d: model.d(),
        has_factor: with_factor,
        factor_kind: FactorKind::Ldlt,
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
    let factor = if with_factor {
        let n = model.n();
        let mut ld = vec![0.0; n * n];
        pack_lower_l(model.ld_factor(), &mut ld);
        Some((ld, model.alpha().to_vec()))
    } else {
        None
    };
    let factor_refs = factor
        .as_ref()
        .map(|(l, alpha)| (l.as_slice(), alpha.as_slice()));
    write_tensors(dir, model.x(), model.y(), model.n(), model.d(), factor_refs)
}

fn apply_online_ids<O, S, C, B>(
    online: &mut OnlineGpr<O, S, C, B>,
    ids: &Option<(Vec<u64>, u64)>,
) -> Result<(), GprError>
where
    C: DistanceCacheSlot,
    B: crate::gpr::AllocWorkspace,
{
    let (ids, next_id) = ids
        .as_ref()
        .ok_or_else(|| persist_err("ldlt config missing point_ids"))?;
    online.apply_persisted_ids(ids, *next_id)
}

fn load_dir(dir: &Path, registry: &PersistRegistry) -> Result<LoadedGpr, GprError> {
    let config_path = dir.join(CONFIG_FILE);
    let bytes = std::fs::read(&config_path)
        .map_err(|err| persist_err(format!("read {config_path:?}: {err}")))?;
    let config = config::parse_config(&bytes)?;
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
        let alpha = read_alpha(dir, config.n)?;
        let mapped = MappedTensors::open(dir, config.n)?;
        match (config.factor_kind, cache) {
            (FactorKind::Llt, Some(crate::gpr::DistanceCachePersist::Cached)) => {
                Ok(LoadedGpr::Distance(LoadedDistance::Cached(
                    FittedGpr::from_persisted(PersistedModel {
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
                        mapped: Some(mapped),
                    })?,
                )))
            }
            (FactorKind::Llt, Some(crate::gpr::DistanceCachePersist::Uncached)) => {
                Ok(LoadedGpr::Distance(LoadedDistance::Uncached(
                    FittedGpr::from_persisted(PersistedModel {
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
                        mapped: Some(mapped),
                    })?,
                )))
            }
            (FactorKind::Llt, None) => Ok(LoadedGpr::Points(FittedGpr::from_persisted(
                PersistedModel {
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
                    mapped: Some(mapped),
                },
            )?)),
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
                    mapped: Some(mapped),
                })?;
                apply_online_ids(&mut online, &ldlt_ids)?;
                Ok(LoadedGpr::OnlineDistance(LoadedOnlineDistance::Cached(
                    online,
                )))
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
                    mapped: Some(mapped),
                })?;
                apply_online_ids(&mut online, &ldlt_ids)?;
                Ok(LoadedGpr::OnlineDistance(LoadedOnlineDistance::Uncached(
                    online,
                )))
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
                    mapped: Some(mapped),
                })?;
                apply_online_ids(&mut online, &ldlt_ids)?;
                Ok(LoadedGpr::OnlinePoints(online))
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
                    .with_distance_cache_policy(CachedDistances);
                Ok(LoadedGpr::Distance(LoadedDistance::Cached(
                    gpr.factor(&x_obs, config.n, config.d, &y_obs)
                        .map_err(|(_, err)| err)?,
                )))
            }
            (FactorKind::Llt, Some(crate::gpr::DistanceCachePersist::Uncached)) => {
                let gpr = crate::Gpr::new(kernel, likelihood)
                    .with_optimizer(Fixed)
                    .with_boxed_input_transform(x_unfitted)
                    .with_boxed_target_transform(y_unfitted)
                    .with_jitter_policy(jitter)
                    .with_distance_cache_policy(UncachedDistances);
                Ok(LoadedGpr::Distance(LoadedDistance::Uncached(
                    gpr.factor(&x_obs, config.n, config.d, &y_obs)
                        .map_err(|(_, err)| err)?,
                )))
            }
            (FactorKind::Llt, None) => {
                let gpr = crate::Gpr::from_points(kernel, likelihood)
                    .with_optimizer(Fixed)
                    .with_boxed_input_transform(x_unfitted)
                    .with_boxed_target_transform(y_unfitted)
                    .with_jitter_policy(jitter);
                Ok(LoadedGpr::Points(
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
                    .with_distance_cache_policy(CachedDistances);
                let mut online = gpr
                    .factor(&x_obs, config.n, config.d, &y_obs)
                    .map_err(|(_, err)| err)?
                    .into_online()?;
                apply_online_ids(&mut online, &ldlt_ids)?;
                Ok(LoadedGpr::OnlineDistance(LoadedOnlineDistance::Cached(
                    online,
                )))
            }
            (FactorKind::Ldlt, Some(crate::gpr::DistanceCachePersist::Uncached)) => {
                let gpr = crate::Gpr::new(kernel, likelihood)
                    .with_optimizer(Fixed)
                    .with_boxed_input_transform(x_unfitted)
                    .with_boxed_target_transform(y_unfitted)
                    .with_jitter_policy(jitter)
                    .with_distance_cache_policy(UncachedDistances);
                let mut online = gpr
                    .factor(&x_obs, config.n, config.d, &y_obs)
                    .map_err(|(_, err)| err)?
                    .into_online()?;
                apply_online_ids(&mut online, &ldlt_ids)?;
                Ok(LoadedGpr::OnlineDistance(LoadedOnlineDistance::Uncached(
                    online,
                )))
            }
            (FactorKind::Ldlt, None) => {
                let gpr = crate::Gpr::from_points(kernel, likelihood)
                    .with_optimizer(Fixed)
                    .with_boxed_input_transform(x_unfitted)
                    .with_boxed_target_transform(y_unfitted)
                    .with_jitter_policy(jitter);
                let mut online = gpr
                    .factor(&x_obs, config.n, config.d, &y_obs)
                    .map_err(|(_, err)| err)?
                    .into_online()?;
                apply_online_ids(&mut online, &ldlt_ids)?;
                Ok(LoadedGpr::OnlinePoints(online))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CONFIG_FILE, FORMAT_VERSION, LoadedDistance, LoadedGpr, PersistRegistry, RESERVED_PREFIX,
        persist_err,
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

    impl KernelTerm for PersistUnit {
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
            self.get_params(&mut params.to_vec())
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
            dist: MatRef<'_, f64>,
            mut out: MatMut<'_, f64>,
            _uplo: Triangle,
        ) -> Result<(), GprError> {
            for col in 0..dist.ncols() {
                for row in 0..dist.nrows() {
                    out[(row, col)] = 1.0;
                }
            }
            Ok(())
        }

        fn apply_cross(
            &self,
            dist: MatRef<'_, f64>,
            mut out: MatMut<'_, f64>,
        ) -> Result<(), GprError> {
            for col in 0..dist.ncols() {
                for row in 0..dist.nrows() {
                    out[(row, col)] = 1.0;
                }
            }
            Ok(())
        }

        fn fill_diag(&self, out: &mut [f64]) -> Result<(), GprError> {
            out.fill(1.0);
            Ok(())
        }

        fn grad(
            &self,
            _dist: MatRef<'_, f64>,
            _d_k: MatMut<'_, f64>,
            param_idx: usize,
            _uplo: Triangle,
        ) -> Result<(), GprError> {
            Err(GprError::InvalidHyperparameter {
                reason: format!("persist unit kernel has no parameter {param_idx}"),
            })
        }

        fn hess(
            &self,
            _dist: MatRef<'_, f64>,
            _d2_k: MatMut<'_, f64>,
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
            _x: MatRef<'_, f64>,
            _d2_k: MatMut<'_, f64>,
            i: usize,
            j: usize,
            _uplo: Triangle,
        ) -> Result<(), GprError> {
            Err(GprError::InvalidHyperparameter {
                reason: format!("persist unit kernel has no parameter pair ({i}, {j})"),
            })
        }

        fn clone_box(&self) -> Box<dyn KernelTerm> {
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
                Ok(Box::new(PersistUnit) as Box<dyn KernelTerm>)
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
                Ok(Box::new(PersistUnit) as Box<dyn KernelTerm>)
            })
            .expect("register");
        let loaded = LoadedGpr::load(&dir, &registry).expect("load");
        assert!(matches!(loaded, LoadedGpr::Distance(_)));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
