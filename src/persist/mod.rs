//! Save and load a fitted GPR directory (`config.json` + `model.safetensors`).

mod config;
mod kernel;
mod registry;
mod tensors;
mod transform;

use std::path::Path;

use crate::GaussianLikelihood;
use crate::error::GprError;
use crate::gpr::{FittedGpr, OnlineGpr, Policies};
use crate::kernel::KernelSpec;
use crate::optimizer::Fixed;
use crate::transform::{TargetTransform, Transform, UnfittedTarget, UnfittedTransform};

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
/// One variant per precision and factor kind: `llt` loads a [`FittedGpr`],
/// `ldlt` loads an [`OnlineGpr`]. The distance-cache policy, kernel `exp`,
/// and jitter policy are read back into the model's runtime policies.
/// Re-training is [`crate::FittedGpr::with_optimizer`] then
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
/// match loaded {
///     LoadedGpr::Double(model) => {
///         let pred = model.predict(&[0.5], 1, 1)?;
///         assert_eq!(pred.mean.len(), 1);
///     }
///     _ => panic!("default save is a double-precision llt model"),
/// }
/// let _ = std::fs::remove_dir_all(&dir);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
pub enum LoadedGpr {
    /// [`crate::DoublePrecision`] model.
    Double(FittedGpr<Fixed>),
    /// [`crate::SinglePrecision`] model.
    Single(FittedGpr<Fixed, crate::SinglePrecision>),
    /// Promoted-storage [`crate::MixedPrecision`] model.
    Mixed(FittedGpr<Fixed, crate::MixedPrecision>),
    /// [`crate::MixedPrecision`]`<`[`crate::ReevaluateKernel`]`>` model.
    Reevaluate(FittedGpr<Fixed, crate::MixedPrecision<crate::ReevaluateKernel>>),
    /// [`crate::DoublePrecision`] online model.
    OnlineDouble(OnlineGpr<Fixed>),
    /// [`crate::SinglePrecision`] online model.
    OnlineSingle(OnlineGpr<Fixed, crate::SinglePrecision>),
    /// Promoted-storage [`crate::MixedPrecision`] online model.
    OnlineMixed(OnlineGpr<Fixed, crate::MixedPrecision>),
    /// [`crate::MixedPrecision`]`<`[`crate::ReevaluateKernel`]`>` online model.
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
    /// directory, JSON, tensors, or registry lookup is invalid. Factorization
    /// errors from a file written without `L` use the same variants as
    /// [`crate::Gpr<Fixed>::factor`].
    pub fn load(dir: impl AsRef<Path>, registry: &PersistRegistry) -> Result<Self, GprError> {
        load_dir(dir.as_ref(), registry)
    }
}

pub(crate) struct PersistedModel<P: crate::precision::GpScalar = crate::precision::DoublePrecision>
{
    pub kernel: KernelSpec,
    pub likelihood: GaussianLikelihood,
    pub x_unfitted: Box<dyn UnfittedTransform>,
    pub y_unfitted: Box<dyn UnfittedTarget>,
    pub x_transform: Box<dyn Transform>,
    pub y_transform: Box<dyn TargetTransform>,
    pub policies: Policies,
    pub x_obs: Vec<f64>,
    pub y_obs: Vec<f64>,
    pub alpha: Vec<P::Refine>,
    pub owned_l: Option<faer::Mat<P::Storage>>,
    pub mapped: Option<MappedTensors>,
    /// Diagonal jitter the saved factor was built with.
    pub factor_jitter: f64,
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

pub(crate) fn save_fitted<O, P>(
    model: &FittedGpr<O, P>,
    dir: &Path,
    with_factor: bool,
) -> Result<(), GprError>
where
    P: crate::precision::GpScalar,
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
        math: MathJson::encode(model.policies().math),
        kernel: KernelJson::encode(model.kernel())?,
        likelihood: LikelihoodJson::encode(model.likelihood()),
        jitter: JitterJson::encode(model.policies().jitter),
        factor_jitter: model.factor_jitter(),
        distance_cache: Some(DistanceCacheJson::encode(model.policies().distance_cache)),
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

pub(crate) fn save_online<O, P>(
    model: &OnlineGpr<O, P>,
    dir: &Path,
    with_factor: bool,
) -> Result<(), GprError>
where
    P: crate::precision::GpScalar,
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
        math: MathJson::encode(model.policies().math),
        kernel: KernelJson::encode(model.kernel())?,
        likelihood: LikelihoodJson::encode(model.likelihood()),
        jitter: JitterJson::encode(model.policies().jitter),
        factor_jitter: model.factor_jitter(),
        distance_cache: Some(DistanceCacheJson::encode(model.policies().distance_cache)),
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
        Some(pack_saved_factor(model.ld_factor(), model.alpha()?)?)
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

fn apply_online_ids<O, P>(
    online: &mut OnlineGpr<O, P>,
    ids: &Option<(Vec<u64>, u64)>,
) -> Result<(), GprError>
where
    P: crate::precision::GpScalar,
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

trait Seal: crate::precision::GpScalar {
    fn seal(model: FittedGpr<Fixed, Self>) -> LoadedGpr;
    fn seal_online(model: OnlineGpr<Fixed, Self>) -> LoadedGpr;
}

macro_rules! impl_seal {
    ($prec:ty, $fitted:path, $online:path) => {
        impl Seal for $prec {
            fn seal(model: FittedGpr<Fixed, Self>) -> LoadedGpr {
                $fitted(model)
            }
            fn seal_online(model: OnlineGpr<Fixed, Self>) -> LoadedGpr {
                $online(model)
            }
        }
    };
}

impl_seal!(
    crate::precision::DoublePrecision,
    LoadedGpr::Double,
    LoadedGpr::OnlineDouble
);
impl_seal!(
    crate::SinglePrecision,
    LoadedGpr::Single,
    LoadedGpr::OnlineSingle
);
impl_seal!(
    crate::MixedPrecision<crate::precision::PromoteStorage>,
    LoadedGpr::Mixed,
    LoadedGpr::OnlineMixed
);
impl_seal!(
    crate::MixedPrecision<crate::ReevaluateKernel>,
    LoadedGpr::Reevaluate,
    LoadedGpr::OnlineReevaluate
);

fn load_dir(dir: &Path, registry: &PersistRegistry) -> Result<LoadedGpr, GprError> {
    let config_path = dir.join(CONFIG_FILE);
    let bytes = std::fs::read(&config_path)
        .map_err(|err| persist_err(format!("read {config_path:?}: {err}")))?;
    let config = config::parse_config(&bytes)?;
    match (config.precision, config.residual) {
        (PrecisionJson::Double, _) => {
            load_precision::<crate::precision::DoublePrecision>(dir, registry, config)
        }
        (PrecisionJson::Single, _) => {
            load_precision::<crate::SinglePrecision>(dir, registry, config)
        }
        (PrecisionJson::Mixed, ResidualJson::PromoteStorage) => load_precision::<
            crate::MixedPrecision<crate::precision::PromoteStorage>,
        >(dir, registry, config),
        (PrecisionJson::Mixed, ResidualJson::ReevaluateKernel) => {
            load_precision::<crate::MixedPrecision<crate::ReevaluateKernel>>(dir, registry, config)
        }
    }
}

fn load_precision<P>(
    dir: &Path,
    registry: &PersistRegistry,
    config: ModelConfig,
) -> Result<LoadedGpr, GprError>
where
    P: PersistLoad + Seal,
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
    let policies = Policies {
        distance_cache: config
            .distance_cache
            .map(DistanceCacheJson::decode)
            .unwrap_or_default(),
        cholesky_buffer: crate::CholeskyBuffer::Retain,
        math: config.math.decode(),
        jitter: config.jitter.decode()?,
    };
    let x_unfitted = config.x_unfitted.decode(registry)?;
    let y_unfitted = config.y_unfitted.decode(registry)?;
    let x_transform = config.x_transform.decode(registry)?;
    let y_transform = config.y_transform.decode(registry)?;
    let (x_obs, y_obs) = read_xy(dir, config.n, config.d)?;
    if config.has_factor {
        let alpha = P::read_alpha(dir, config.n)?;
        let (owned_l, mapped) = P::read_factor(dir, config.n)?;
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
            owned_l,
            mapped,
            factor_jitter: config.factor_jitter,
        };
        match config.factor_kind {
            FactorKind::Llt => Ok(P::seal(FittedGpr::from_persisted(parts)?)),
            FactorKind::Ldlt => {
                let mut online = OnlineGpr::from_persisted(parts)?;
                apply_online_ids(&mut online, &ldlt_ids)?;
                Ok(P::seal_online(online))
            }
        }
    } else {
        let fitted = crate::Gpr::<Fixed, P>::from_owned(
            kernel, likelihood, x_unfitted, y_unfitted, Fixed, policies,
        )
        .factor(&x_obs, config.n, config.d, &y_obs)
        .map_err(|(_, err)| err)?;
        match config.factor_kind {
            FactorKind::Llt => Ok(P::seal(fitted)),
            FactorKind::Ldlt => {
                let mut online = fitted.into_online()?;
                apply_online_ids(&mut online, &ldlt_ids)?;
                Ok(P::seal_online(online))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CONFIG_FILE, FORMAT_VERSION, LoadedGpr, PersistRegistry, RESERVED_PREFIX, persist_err,
    };
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
        assert!(matches!(loaded, LoadedGpr::Double(_)));
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
