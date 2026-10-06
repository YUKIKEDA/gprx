//! Save and load the sparse models ([`FittedSgpr`], [`OnlineSgpr`],
//! [`FittedSvgp`]) in the directory format of the Exact model.
//!
//! `config.json` holds the model kind, the settings, and the fitted
//! transforms; `model.safetensors` holds the original `X`, `y`, and `Z`, the
//! transformed `Z` (a [`crate::FreeInducing`] search moved it in those
//! coordinates, and mapping it back and forth is not exact), and for SVGP the
//! whitened `q(u)`. The factors are not stored: load factors the system
//! again at the saved `θ` and `Z`.

use crate::kernel::KernelSpec;
use std::path::Path;

use faer::Mat;

use crate::error::GprError;
use crate::error::PersistErrorKind;
use crate::optimizer::Fixed;
use crate::points::{IdRegistry, PointRegistry};
use crate::precision::{GpScalar, PersistKind};
use crate::sgpr::{FittedSgpr, FixedInducing, InducingRegistry, OnlineSgpr};
use crate::sparse::{PersistedSparse, SparseCore, SparseSpec};
use crate::svgp::FittedSvgp;
use crate::{PredictOptions, Prediction};

use super::config::{
    JitterJson, LikelihoodJson, MathJson, ModelJson, PrecisionJson, ResidualJson, SparseConfig,
    parse_model, parse_sparse_config,
};
use super::kernel::KernelJson;
use super::tensors::{TensorFile, read_f64, write_f64_tensors};
use super::transform::{
    encode_fitted_input, encode_fitted_target, encode_unfitted_input, encode_unfitted_target,
};
use super::{CONFIG_FILE, FORMAT_VERSION, PersistRegistry, persist_err, widen};
use safetensors::SafeTensors;

const TENSOR_X: &str = "x";
const TENSOR_Y: &str = "y";
const TENSOR_Z: &str = "z";
const TENSOR_Z_TRAIN: &str = "z_train";
const TENSOR_Q_MEAN: &str = "q_mean";
const TENSOR_Q_L: &str = "q_l";

/// Identifiers of an online model, written only for [`ModelJson::OnlineSgpr`].
struct OnlineIds {
    points: Vec<u64>,
    next_point: u64,
    inducing: Vec<u64>,
    next_inducing: u64,
}

fn write_sparse<P: GpScalar>(
    dir: &Path,
    core: &SparseCore<KernelSpec>,
    model: ModelJson,
    ids: Option<OnlineIds>,
    q: Option<(&[f64], faer::MatRef<'_, f64>)>,
) -> Result<(), GprError> {
    std::fs::create_dir_all(dir)
        .map_err(|err| persist_err(PersistErrorKind::Io, format!("create {dir:?}: {err}")))?;
    let kind = P::persist_kind();
    let (point_ids, next_point_id, inducing_ids, next_inducing_id) = match ids {
        Some(ids) => (
            Some(ids.points),
            Some(ids.next_point),
            Some(ids.inducing),
            Some(ids.next_inducing),
        ),
        None => (None, None, None, None),
    };
    let config = SparseConfig {
        format_version: FORMAT_VERSION,
        model,
        n: core.n,
        m: core.m,
        d: core.d,
        precision: PrecisionJson::from_persist(kind),
        residual: ResidualJson::from_persist(kind),
        math: MathJson::encode(core.math),
        kernel: KernelJson::encode(&core.kernel)?,
        likelihood: LikelihoodJson::encode(&core.likelihood),
        jitter: JitterJson::encode(core.jitter),
        x_unfitted: encode_unfitted_input(core.x_unfitted.as_ref())?,
        y_unfitted: encode_unfitted_target(core.y_unfitted.as_ref())?,
        x_transform: encode_fitted_input(core.x_transform.as_ref())?,
        y_transform: encode_fitted_target(core.y_transform.as_ref())?,
        point_ids,
        next_point_id,
        inducing_ids,
        next_inducing_id,
    };
    let json = serde_json::to_vec_pretty(&config).map_err(|err| {
        persist_err(
            PersistErrorKind::Config,
            format!("serialize config.json: {err}"),
        )
    })?;
    let (n, m, d) = (core.n, core.m, core.d);
    let mut tensors: Vec<(&str, Vec<usize>, &[f64])> = vec![
        (TENSOR_X, vec![n, d], &core.x_obs),
        (TENSOR_Y, vec![n], &core.y_obs),
        (TENSOR_Z, vec![m, d], &core.z_obs),
        (TENSOR_Z_TRAIN, vec![m, d], &core.z_train),
    ];
    let q_l_values: Vec<f64>;
    if let Some((q_mean, q_l)) = q {
        q_l_values = (0..m)
            .flat_map(|col| (0..m).map(move |row| (row, col)))
            .map(|(row, col)| if row < col { 0.0 } else { q_l[(row, col)] })
            .collect();
        tensors.push((TENSOR_Q_MEAN, vec![m], q_mean));
        tensors.push((TENSOR_Q_L, vec![m, m], &q_l_values));
    }
    write_f64_tensors(dir, &tensors)?;
    super::write_config(dir, &json)
}

pub(crate) fn save_sgpr<O, I: crate::sgpr::InducingLayout, P: GpScalar>(
    model: &FittedSgpr<O, I, P>,
    dir: &Path,
) -> Result<(), GprError> {
    write_sparse::<P>(dir, model.core(), ModelJson::Sgpr, None, None)
}

pub(crate) fn save_online_sgpr<O, P: GpScalar>(
    model: &OnlineSgpr<O, P>,
    dir: &Path,
) -> Result<(), GprError> {
    let ids = OnlineIds {
        points: model.point_registry().raw_ids(),
        next_point: model.point_registry().next_id(),
        inducing: model.inducing_registry().raw_ids(),
        next_inducing: model.inducing_registry().next_id(),
    };
    write_sparse::<P>(dir, model.core(), ModelJson::OnlineSgpr, Some(ids), None)
}

pub(crate) fn save_svgp<P: GpScalar>(model: &FittedSvgp<P>, dir: &Path) -> Result<(), GprError> {
    write_sparse::<P>(dir, model.core(), ModelJson::Svgp, None, Some(model.q()))
}

fn read_config(dir: &Path, expected: &[ModelJson]) -> Result<SparseConfig, GprError> {
    let config_path = dir.join(CONFIG_FILE);
    let bytes = std::fs::read(&config_path)
        .map_err(|err| persist_err(PersistErrorKind::Io, format!("read {config_path:?}: {err}")))?;
    parse_model(&bytes, expected)?;
    parse_sparse_config(&bytes)
}

/// The core of a sparse persist directory, with the fitted transforms
/// read back from the config.
fn read_core(
    tensors: &SafeTensors<'_>,
    config: &SparseConfig,
    registry: &PersistRegistry,
) -> Result<SparseCore<KernelSpec>, GprError> {
    let (n, m, d) = (config.n, config.m, config.d);
    let spec = SparseSpec {
        kernel: config.kernel.clone().decode(registry)?,
        likelihood: config.likelihood.decode()?,
        math: config.math.decode(),
        jitter: config.jitter.decode()?,
        x_transform: config.x_unfitted.clone().decode(registry)?,
        y_transform: config.y_unfitted.clone().decode(registry)?,
    };
    SparseCore::from_persisted(PersistedSparse {
        spec,
        x_transform: config.x_transform.clone().decode(registry)?,
        y_transform: config.y_transform.clone().decode(registry)?,
        x_obs: read_f64(tensors, TENSOR_X, &[n, d])?,
        y_obs: read_f64(tensors, TENSOR_Y, &[n])?,
        z_obs: read_f64(tensors, TENSOR_Z, &[m, d])?,
        z_train: read_f64(tensors, TENSOR_Z_TRAIN, &[m, d])?,
        n,
        m,
        d,
    })
}

/// The saved online identifiers.
fn read_ids(config: &SparseConfig) -> Result<(PointRegistry, InducingRegistry), GprError> {
    let missing = |key: &str| {
        persist_err(
            PersistErrorKind::Config,
            format!("online_sgpr config missing {key}"),
        )
    };
    let points = IdRegistry::from_persisted(
        config
            .point_ids
            .as_deref()
            .ok_or_else(|| missing("point_ids"))?,
        config
            .next_point_id
            .ok_or_else(|| missing("next_point_id"))?,
    )?;
    let inducing = IdRegistry::from_persisted(
        config
            .inducing_ids
            .as_deref()
            .ok_or_else(|| missing("inducing_ids"))?,
        config
            .next_inducing_id
            .ok_or_else(|| missing("next_inducing_id"))?,
    )?;
    Ok((points, inducing))
}

/// The saved whitened `q(u)`: a finite mean and a lower `L` with a positive
/// diagonal.
fn read_q(tensors: &SafeTensors<'_>, m: usize) -> Result<(Vec<f64>, Mat<f64>), GprError> {
    let q_mean = read_f64(tensors, TENSOR_Q_MEAN, &[m])?;
    let values = read_f64(tensors, TENSOR_Q_L, &[m, m])?;
    let q_l = Mat::from_fn(m, m, |row, col| values[col * m + row]);
    let finite = q_mean.iter().chain(&values).all(|value| value.is_finite());
    let lower = (0..m).all(|col| (0..col).all(|row| q_l[(row, col)] == 0.0));
    let positive = (0..m).all(|i| q_l[(i, i)] > 0.0);
    if !(finite && lower && positive) {
        return Err(persist_err(
            PersistErrorKind::Tensor,
            "q_l must be lower triangular with a positive diagonal, and q finite",
        ));
    }
    Ok((q_mean, q_l))
}

/// Represents the prediction-only sparse GPR loaded from a persist directory written by [`FittedSgpr::save`] or [`OnlineSgpr::save`].
///
/// One variant per precision and model, as [`super::LoadedGpr`]. The kernel,
/// likelihood, kernel `exp`, `K_mm` jitter policy, and transforms are read
/// back; the VFE system is factored again at the saved `θ` and `Z`. The
/// optimizer is [`Fixed`] and the inducing points are [`FixedInducing`]:
/// the file does not store the search. [`Self::predict`],
/// [`Self::predict_with`], [`Self::n`], [`Self::m`], [`Self::d`], and
/// [`Self::is_online`] work on any variant.
///
/// # Examples
///
/// ```rust
/// use gprx::kernel::{KernelSpec, RbfKernel};
/// use gprx::persist::{LoadedSgpr, PersistRegistry};
/// use gprx::{Fixed, GaussianLikelihood, Sgpr};
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let fitted = Sgpr::new(
///     KernelSpec::from(RbfKernel::new(1.0)?),
///     GaussianLikelihood::new(0.1)?,
/// )
/// .with_optimizer(Fixed)
/// .factor(&[0.0, 1.0, 2.0], 3, 1, &[0.0, 1.0, 0.5], &[0.5, 1.5], 2)
/// .map_err(|(_, e)| e)?;
/// let dir = std::env::temp_dir().join(format!("gprx-doctest-sgpr-{}", std::process::id()));
/// let _ = std::fs::remove_dir_all(&dir);
/// fitted.save(&dir)?;
/// let loaded = LoadedSgpr::load(&dir, &PersistRegistry::new())?;
/// assert_eq!(loaded.predict(&[0.5], 1, 1)?, fitted.predict(&[0.5], 1, 1)?);
/// assert!(!loaded.is_online());
/// let _ = std::fs::remove_dir_all(&dir);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum LoadedSgpr {
    /// Marks a [`crate::DoublePrecision`] model.
    Double(FittedSgpr<Fixed>),
    /// Marks a [`crate::SinglePrecision`] model.
    Single(FittedSgpr<Fixed, FixedInducing, crate::SinglePrecision>),
    /// Marks a promoted-storage [`crate::MixedPrecision`] model.
    Mixed(FittedSgpr<Fixed, FixedInducing, crate::MixedPrecision>),
    /// Marks a [`crate::MixedPrecision`]`<`[`crate::ReevaluateKernel`]`>` model.
    Reevaluate(FittedSgpr<Fixed, FixedInducing, crate::MixedPrecision<crate::ReevaluateKernel>>),
    /// Marks a [`crate::DoublePrecision`] online model.
    OnlineDouble(OnlineSgpr<Fixed>),
    /// Marks a [`crate::SinglePrecision`] online model.
    OnlineSingle(OnlineSgpr<Fixed, crate::SinglePrecision>),
    /// Marks a promoted-storage [`crate::MixedPrecision`] online model.
    OnlineMixed(OnlineSgpr<Fixed, crate::MixedPrecision>),
    /// Marks a [`crate::MixedPrecision`]`<`[`crate::ReevaluateKernel`]`>` online model.
    OnlineReevaluate(OnlineSgpr<Fixed, crate::MixedPrecision<crate::ReevaluateKernel>>),
}

/// The [`LoadedSgpr`] variants that hold precision `P`.
struct SgprVariants<P: GpScalar> {
    fitted: fn(FittedSgpr<Fixed, FixedInducing, P>) -> LoadedSgpr,
    online: fn(OnlineSgpr<Fixed, P>) -> LoadedSgpr,
}

fn load_sgpr_as<P: GpScalar>(
    dir: &Path,
    config: &SparseConfig,
    registry: &PersistRegistry,
    variants: SgprVariants<P>,
) -> Result<LoadedSgpr, GprError> {
    let fitted = FittedSgpr::<Fixed, FixedInducing, P>::from_persisted(read_core(
        &TensorFile::read(dir)?.tensors()?,
        config,
        registry,
    )?)?;
    if config.model == ModelJson::OnlineSgpr {
        let (points, inducing) = read_ids(config)?;
        Ok((variants.online)(OnlineSgpr::from_persisted(
            fitted, points, inducing,
        )?))
    } else {
        Ok((variants.fitted)(fitted))
    }
}

impl LoadedSgpr {
    /// Reads `dir/config.json` and `dir/model.safetensors` written by [`FittedSgpr::save`] (a [`FittedSgpr`] variant) or [`OnlineSgpr::save`] (an [`OnlineSgpr`] variant with the saved identifiers).
    ///
    /// # Errors
    ///
    /// Returns [`GprError::UnsupportedPersistVersion`] when `format_version`
    /// is not [`FORMAT_VERSION`], or [`GprError::PersistFailed`] when the
    /// directory holds another model, or its JSON, tensors, or registry
    /// lookup is invalid. Factorization errors use the same variants as
    /// [`crate::Sgpr<Fixed>::factor`].
    ///
    /// See the example on [`LoadedSgpr`].
    pub fn load(dir: impl AsRef<Path>, registry: &PersistRegistry) -> Result<Self, GprError> {
        let dir = dir.as_ref();
        let config = read_config(dir, &[ModelJson::Sgpr, ModelJson::OnlineSgpr])?;
        match config.persist_kind() {
            PersistKind::Double => load_sgpr_as(
                dir,
                &config,
                registry,
                SgprVariants {
                    fitted: Self::Double,
                    online: Self::OnlineDouble,
                },
            ),
            PersistKind::Single => load_sgpr_as(
                dir,
                &config,
                registry,
                SgprVariants {
                    fitted: Self::Single,
                    online: Self::OnlineSingle,
                },
            ),
            PersistKind::MixedPromote => load_sgpr_as(
                dir,
                &config,
                registry,
                SgprVariants {
                    fitted: Self::Mixed,
                    online: Self::OnlineMixed,
                },
            ),
            PersistKind::MixedReevaluate => load_sgpr_as(
                dir,
                &config,
                registry,
                SgprVariants {
                    fitted: Self::Reevaluate,
                    online: Self::OnlineReevaluate,
                },
            ),
        }
    }

    /// Holds the number of training points.
    ///
    /// See the example on [`LoadedSgpr`].
    pub fn n(&self) -> usize {
        self.view().core().n
    }

    /// Holds the number of inducing points.
    ///
    /// See the example on [`LoadedSgpr`].
    pub fn m(&self) -> usize {
        self.view().core().m
    }

    /// Holds the number of input features.
    ///
    /// See the example on [`LoadedSgpr`].
    pub fn d(&self) -> usize {
        self.view().core().d
    }

    /// Returns the `true` for an [`OnlineSgpr`] variant.
    ///
    /// See the example on [`LoadedSgpr`].
    pub fn is_online(&self) -> bool {
        matches!(
            self,
            Self::OnlineDouble(_)
                | Self::OnlineSingle(_)
                | Self::OnlineMixed(_)
                | Self::OnlineReevaluate(_)
        )
    }

    /// Returns the predictive mean and observation variance at `xs`, in `f64` whatever the stored precision.
    ///
    /// # Errors
    ///
    /// Same as [`FittedSgpr::predict`].
    ///
    /// See the example on [`LoadedSgpr`].
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
    /// Same as [`FittedSgpr::predict_with`].
    ///
    /// See the example on [`LoadedSgpr`].
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
    fn view(&self) -> &dyn SparseView {
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

/// Represents the prediction-only SVGP loaded from a persist directory written by [`FittedSvgp::save`].
///
/// One variant per precision, as [`super::LoadedGpr`]. `K_mm` and `A` are
/// factored again at the saved `θ` and `Z`, with the saved `q(u)`.
///
/// # Examples
///
/// ```rust
/// use gprx::kernel::{KernelSpec, RbfKernel};
/// use gprx::persist::{LoadedSvgp, PersistRegistry};
/// use gprx::{GaussianLikelihood, Svgp};
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let fitted = Svgp::new(
///     KernelSpec::from(RbfKernel::new(1.0)?),
///     GaussianLikelihood::new(0.1)?,
/// )
/// .factor(&[0.0, 1.0, 2.0], 3, 1, &[0.0, 1.0, 0.5], &[0.5, 1.5], 2)
/// .map_err(|(_, e)| e)?;
/// let dir = std::env::temp_dir().join(format!("gprx-doctest-svgp-{}", std::process::id()));
/// let _ = std::fs::remove_dir_all(&dir);
/// fitted.save(&dir)?;
/// let loaded = LoadedSvgp::load(&dir, &PersistRegistry::new())?;
/// assert_eq!(loaded.predict(&[0.5], 1, 1)?, fitted.predict(&[0.5], 1, 1)?);
/// let _ = std::fs::remove_dir_all(&dir);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum LoadedSvgp {
    /// Marks a [`crate::DoublePrecision`] model.
    Double(FittedSvgp),
    /// Marks a [`crate::SinglePrecision`] model.
    Single(FittedSvgp<crate::SinglePrecision>),
    /// Marks a promoted-storage [`crate::MixedPrecision`] model.
    Mixed(FittedSvgp<crate::MixedPrecision>),
    /// Marks a [`crate::MixedPrecision`]`<`[`crate::ReevaluateKernel`]`>` model.
    Reevaluate(FittedSvgp<crate::MixedPrecision<crate::ReevaluateKernel>>),
}

fn load_svgp_as<P: GpScalar>(
    dir: &Path,
    config: &SparseConfig,
    registry: &PersistRegistry,
    variant: fn(FittedSvgp<P>) -> LoadedSvgp,
) -> Result<LoadedSvgp, GprError> {
    let file = TensorFile::read(dir)?;
    let tensors = file.tensors()?;
    let core = read_core(&tensors, config, registry)?;
    let (q_mean, q_l) = read_q(&tensors, config.m)?;
    Ok(variant(FittedSvgp::from_persisted(core, q_mean, q_l)?))
}

impl LoadedSvgp {
    /// Reads `dir/config.json` and `dir/model.safetensors` written by [`FittedSvgp::save`].
    ///
    /// # Errors
    ///
    /// Same as [`LoadedSgpr::load`], plus [`GprError::PersistFailed`] when the
    /// saved `q(u)` is not finite or its `L` is not lower triangular with a
    /// positive diagonal.
    ///
    /// See the example on [`LoadedSvgp`].
    pub fn load(dir: impl AsRef<Path>, registry: &PersistRegistry) -> Result<Self, GprError> {
        let dir = dir.as_ref();
        let config = read_config(dir, &[ModelJson::Svgp])?;
        match config.persist_kind() {
            PersistKind::Double => load_svgp_as(dir, &config, registry, Self::Double),
            PersistKind::Single => load_svgp_as(dir, &config, registry, Self::Single),
            PersistKind::MixedPromote => load_svgp_as(dir, &config, registry, Self::Mixed),
            PersistKind::MixedReevaluate => load_svgp_as(dir, &config, registry, Self::Reevaluate),
        }
    }

    /// Holds the number of training points.
    ///
    /// See the example on [`LoadedSvgp`].
    pub fn n(&self) -> usize {
        self.view().core().n
    }

    /// Holds the number of inducing points.
    ///
    /// See the example on [`LoadedSvgp`].
    pub fn m(&self) -> usize {
        self.view().core().m
    }

    /// Holds the number of input features.
    ///
    /// See the example on [`LoadedSvgp`].
    pub fn d(&self) -> usize {
        self.view().core().d
    }

    /// Returns the predictive mean and observation variance at `xs`, in `f64` whatever the stored precision.
    ///
    /// # Errors
    ///
    /// Same as [`FittedSvgp::predict`].
    ///
    /// See the example on [`LoadedSvgp`].
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
    /// Same as [`FittedSvgp::predict_with`].
    ///
    /// See the example on [`LoadedSvgp`].
    pub fn predict_with(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        options: PredictOptions,
    ) -> Result<Prediction<f64>, GprError> {
        self.view().predict_f64(xs, n_rows, n_cols, options)
    }

    fn view(&self) -> &dyn SparseView {
        match self {
            Self::Double(model) => model,
            Self::Single(model) => model,
            Self::Mixed(model) => model,
            Self::Reevaluate(model) => model,
        }
    }
}

/// Reads of a loaded sparse model that do not depend on its precision.
trait SparseView {
    fn core(&self) -> &SparseCore<KernelSpec>;
    fn predict_f64(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        options: PredictOptions,
    ) -> Result<Prediction<f64>, GprError>;
}

impl<P: GpScalar> SparseView for FittedSgpr<Fixed, FixedInducing, P> {
    fn core(&self) -> &SparseCore<KernelSpec> {
        FittedSgpr::core(self)
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

impl<P: GpScalar> SparseView for OnlineSgpr<Fixed, P> {
    fn core(&self) -> &SparseCore<KernelSpec> {
        OnlineSgpr::core(self)
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

impl<P: GpScalar> SparseView for FittedSvgp<P> {
    fn core(&self) -> &SparseCore<KernelSpec> {
        FittedSvgp::core(self)
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
