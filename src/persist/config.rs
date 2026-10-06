//! `config.json` schema for a fitted model directory.

use serde::{Deserialize, Serialize};

use crate::error::GprError;
use crate::error::PersistErrorKind;
use crate::param::{BoundedParam, Interval};
use crate::policy::{DistanceCachePolicy, KernelExp};
use crate::precision::PersistKind;
use crate::{GaussianLikelihood, JitterPolicy};

use super::kernel::KernelJson;
use super::transform::{FittedInputJson, FittedTargetJson, UnfittedInputJson, UnfittedTargetJson};
use super::{FORMAT_VERSION, persist_err};

/// On-disk model metadata. Integer `format_version` is checked first.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct ModelConfig {
    pub format_version: u32,
    pub n: usize,
    pub d: usize,
    pub has_factor: bool,
    pub factor_kind: FactorKind,
    /// Omitted on disk means [`PrecisionJson::Double`].
    #[serde(default, skip_serializing_if = "PrecisionJson::is_double")]
    pub precision: PrecisionJson,
    /// Omitted on disk means [`ResidualJson::PromoteStorage`]. Read only
    /// through [`Self::persist_kind`].
    #[serde(default, skip_serializing_if = "ResidualJson::is_promote_storage")]
    pub residual: ResidualJson,
    /// Omitted on disk means [`MathJson::Accurate`].
    #[serde(default, skip_serializing_if = "MathJson::is_accurate")]
    pub math: MathJson,
    pub kernel: KernelJson,
    pub likelihood: LikelihoodJson,
    pub jitter: JitterJson,
    /// Diagonal jitter the saved factor was built with. Omitted on disk means `0`.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub factor_jitter: f64,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub distance_cache: Option<DistanceCacheJson>,
    pub x_unfitted: UnfittedInputJson,
    pub y_unfitted: UnfittedTargetJson,
    pub x_transform: FittedInputJson,
    pub y_transform: FittedTargetJson,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub point_ids: Option<Vec<u64>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub next_point_id: Option<u64>,
}

impl ModelConfig {
    /// The precision this directory records ([`decode_precision`]).
    pub(super) fn persist_kind(&self) -> PersistKind {
        decode_precision(self.precision, self.residual)
    }

    pub(super) fn validate_version(&self) -> Result<(), GprError> {
        if self.format_version == FORMAT_VERSION {
            Ok(())
        } else {
            Err(GprError::UnsupportedPersistVersion {
                found: self.format_version,
                supported: FORMAT_VERSION,
            })
        }
    }

    pub(super) fn online_ids(&self) -> Result<(&[u64], u64), GprError> {
        let ids = self.point_ids.as_deref().ok_or_else(|| {
            persist_err(PersistErrorKind::Config, "ldlt config missing point_ids")
        })?;
        let next_id = self.next_point_id.ok_or_else(|| {
            persist_err(
                PersistErrorKind::Config,
                "ldlt config missing next_point_id",
            )
        })?;
        if ids.len() != self.n {
            return Err(persist_err(
                PersistErrorKind::Config,
                format!(
                    "point_ids has {} values, expected n = {}",
                    ids.len(),
                    self.n
                ),
            ));
        }
        Ok((ids, next_id))
    }
}

/// Which triangular factor is stored (or reconstructed) for this directory.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum FactorKind {
    Llt,
    Ldlt,
}

/// Storage and predict scalar recorded in `config.json`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum PrecisionJson {
    #[default]
    Double,
    Single,
    Mixed,
}

impl PrecisionJson {
    fn is_double(kind: &Self) -> bool {
        matches!(kind, Self::Double)
    }

    pub(super) fn from_persist(kind: PersistKind) -> Self {
        match kind {
            PersistKind::Double => Self::Double,
            PersistKind::Single => Self::Single,
            PersistKind::MixedPromote | PersistKind::MixedReevaluate => Self::Mixed,
        }
    }
}

/// The precision a `precision` / `residual` pair records. `residual` only
/// tells the two mixed precisions apart; with `double` or `single` it is not
/// read. The one place the pair is decoded on load.
pub(super) fn decode_precision(precision: PrecisionJson, residual: ResidualJson) -> PersistKind {
    match (precision, residual) {
        (PrecisionJson::Double, _) => PersistKind::Double,
        (PrecisionJson::Single, _) => PersistKind::Single,
        (PrecisionJson::Mixed, ResidualJson::PromoteStorage) => PersistKind::MixedPromote,
        (PrecisionJson::Mixed, ResidualJson::ReevaluateKernel) => PersistKind::MixedReevaluate,
    }
}

/// Kernel `exp`. Omitted on disk means [`MathJson::Accurate`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum MathJson {
    #[default]
    Accurate,
    FastApprox,
}

impl MathJson {
    fn is_accurate(kind: &Self) -> bool {
        matches!(kind, Self::Accurate)
    }

    pub(super) fn encode(math: KernelExp) -> Self {
        match math {
            KernelExp::Accurate => Self::Accurate,
            KernelExp::FastApprox => Self::FastApprox,
        }
    }

    pub(super) fn decode(self) -> KernelExp {
        match self {
            Self::Accurate => KernelExp::Accurate,
            Self::FastApprox => KernelExp::FastApprox,
        }
    }
}

/// Mixed-precision residual. Decoded with `precision` by [`decode_precision`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum ResidualJson {
    #[default]
    PromoteStorage,
    ReevaluateKernel,
}

impl ResidualJson {
    fn is_promote_storage(kind: &Self) -> bool {
        matches!(kind, Self::PromoteStorage)
    }

    pub(super) fn from_persist(kind: PersistKind) -> Self {
        match kind {
            PersistKind::MixedReevaluate => Self::ReevaluateKernel,
            _ => Self::PromoteStorage,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub(super) struct BoundedJson {
    pub value: f64,
    pub lo: f64,
    pub hi: f64,
}

impl BoundedJson {
    pub(super) fn from_param(param: BoundedParam) -> Self {
        let interval = param.interval();
        Self {
            value: param.value(),
            lo: interval.lo(),
            hi: interval.hi(),
        }
    }

    pub(super) fn into_param(self) -> Result<BoundedParam, GprError> {
        let interval = Interval::new(self.lo, self.hi)?;
        Ok(BoundedParam::new(self.value, interval)?)
    }

    pub(super) fn interval(self) -> Result<Interval, GprError> {
        Ok(Interval::new(self.lo, self.hi)?)
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub(super) struct LikelihoodJson {
    pub noise_variance: BoundedJson,
}

impl LikelihoodJson {
    pub(super) fn encode(likelihood: &GaussianLikelihood) -> Self {
        Self {
            noise_variance: BoundedJson {
                value: likelihood.noise_variance(),
                lo: likelihood.bounds().lo(),
                hi: likelihood.bounds().hi(),
            },
        }
    }

    pub(super) fn decode(self) -> Result<GaussianLikelihood, GprError> {
        Ok(GaussianLikelihood::new(self.noise_variance.value)?
            .with_bounds(self.noise_variance.interval()?)?)
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum JitterJson {
    Fixed {
        jitter: f64,
    },
    Adaptive {
        initial: f64,
        multiplier: f64,
        max_retries: usize,
        max_jitter: f64,
    },
}

impl JitterJson {
    pub(super) fn encode(policy: JitterPolicy) -> Self {
        match policy {
            JitterPolicy::Fixed(fixed) => Self::Fixed {
                jitter: fixed.jitter(),
            },
            JitterPolicy::Adaptive(adaptive) => Self::Adaptive {
                initial: adaptive.initial(),
                multiplier: adaptive.multiplier(),
                max_retries: adaptive.max_retries(),
                max_jitter: adaptive.max_jitter(),
            },
        }
    }

    pub(super) fn decode(self) -> Result<JitterPolicy, GprError> {
        match self {
            Self::Fixed { jitter } => JitterPolicy::fixed(jitter),
            Self::Adaptive {
                initial,
                multiplier,
                max_retries,
                max_jitter,
            } => JitterPolicy::adaptive(initial, multiplier, max_retries, max_jitter),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum DistanceCacheJson {
    Always,
    Never,
}

impl DistanceCacheJson {
    pub(super) fn encode(policy: DistanceCachePolicy) -> Self {
        match policy {
            DistanceCachePolicy::Cached => Self::Always,
            DistanceCachePolicy::Uncached => Self::Never,
        }
    }

    pub(super) fn decode(self) -> DistanceCachePolicy {
        match self {
            Self::Always => DistanceCachePolicy::Cached,
            Self::Never => DistanceCachePolicy::Uncached,
        }
    }
}

/// Which model a persist directory holds. Exact files written before the
/// sparse models have no `model` key and read as [`Self::Exact`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum ModelJson {
    #[default]
    Exact,
    Sgpr,
    OnlineSgpr,
    Svgp,
}

impl ModelJson {
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::Sgpr => "sgpr",
            Self::OnlineSgpr => "online_sgpr",
            Self::Svgp => "svgp",
        }
    }

    /// The loader of this model, for the error of a wrong one.
    pub(super) fn loader(self) -> &'static str {
        match self {
            Self::Exact => "LoadedGpr::load",
            Self::Sgpr | Self::OnlineSgpr => "LoadedSgpr::load",
            Self::Svgp => "LoadedSvgp::load",
        }
    }
}

#[derive(Deserialize)]
struct ModelTag {
    #[serde(default)]
    model: ModelJson,
}

/// Reads only the `model` key of `config.json` and checks it is one of
/// `expected`.
///
/// Every parse of `config.json` goes through `serde_json::from_slice`, whose
/// recursion limit (128 nested arrays or objects) rejects a deeply nested
/// `sum` / `product` / `pipeline` / `columnwise` tree as invalid JSON before
/// any recursive decode runs. Keep that limit: do not parse `config.json`
/// with the `unbounded_depth` feature or `disable_recursion_limit`.
pub(super) fn parse_model(bytes: &[u8], expected: &[ModelJson]) -> Result<ModelJson, GprError> {
    let tag: ModelTag = serde_json::from_slice(bytes).map_err(|err| {
        persist_err(
            PersistErrorKind::Config,
            format!("config.json is not valid JSON: {err}"),
        )
    })?;
    if expected.contains(&tag.model) {
        Ok(tag.model)
    } else {
        Err(persist_err(
            PersistErrorKind::WrongModel,
            format!(
                "config.json holds a {} model; load it with {}",
                tag.model.name(),
                tag.model.loader()
            ),
        ))
    }
}

/// `config.json` of a sparse model ([`ModelJson::Sgpr`],
/// [`ModelJson::OnlineSgpr`], [`ModelJson::Svgp`]).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct SparseConfig {
    pub format_version: u32,
    pub model: ModelJson,
    pub n: usize,
    pub m: usize,
    pub d: usize,
    #[serde(default, skip_serializing_if = "PrecisionJson::is_double")]
    pub precision: PrecisionJson,
    /// Read only through [`Self::persist_kind`].
    #[serde(default, skip_serializing_if = "ResidualJson::is_promote_storage")]
    pub residual: ResidualJson,
    #[serde(default, skip_serializing_if = "MathJson::is_accurate")]
    pub math: MathJson,
    pub kernel: KernelJson,
    pub likelihood: LikelihoodJson,
    pub jitter: JitterJson,
    pub x_unfitted: UnfittedInputJson,
    pub y_unfitted: UnfittedTargetJson,
    pub x_transform: FittedInputJson,
    pub y_transform: FittedTargetJson,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub point_ids: Option<Vec<u64>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub next_point_id: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub inducing_ids: Option<Vec<u64>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub next_inducing_id: Option<u64>,
    /// The training points that are the inducing points of a distance
    /// kernel, in order.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub inducing: Option<Vec<usize>>,
}

impl SparseConfig {
    /// The precision this directory records ([`decode_precision`]).
    pub(super) fn persist_kind(&self) -> PersistKind {
        decode_precision(self.precision, self.residual)
    }
}

pub(super) fn parse_sparse_config(bytes: &[u8]) -> Result<SparseConfig, GprError> {
    let config: SparseConfig = serde_json::from_slice(bytes).map_err(|err| {
        persist_err(
            PersistErrorKind::Config,
            format!("config.json is not valid JSON: {err}"),
        )
    })?;
    if config.format_version != FORMAT_VERSION {
        return Err(GprError::UnsupportedPersistVersion {
            found: config.format_version,
            supported: FORMAT_VERSION,
        });
    }
    // A distance kernel without coordinate leaves reads no features.
    if config.n == 0 || config.m == 0 || (config.d == 0 && !config.kernel.reads_distances()) {
        return Err(GprError::EmptyInput);
    }
    Ok(config)
}

pub(super) fn parse_config(bytes: &[u8]) -> Result<ModelConfig, GprError> {
    let config: ModelConfig = serde_json::from_slice(bytes).map_err(|err| {
        persist_err(
            PersistErrorKind::Config,
            format!("config.json is not valid JSON: {err}"),
        )
    })?;
    config.validate_version()?;
    if config.n == 0 || (config.d == 0 && !config.kernel.reads_distances()) {
        return Err(GprError::EmptyInput);
    }
    Ok(config)
}

fn is_zero(value: &f64) -> bool {
    *value == 0.0
}

#[cfg(test)]
mod tests {
    use super::{PrecisionJson, ResidualJson, decode_precision};
    use crate::precision::PersistKind;

    /// Every pair decodes, and the residual only splits `mixed`.
    #[test]
    fn decode_precision_reads_the_residual_only_for_mixed() {
        for residual in [ResidualJson::PromoteStorage, ResidualJson::ReevaluateKernel] {
            assert_eq!(
                decode_precision(PrecisionJson::Double, residual),
                PersistKind::Double
            );
            assert_eq!(
                decode_precision(PrecisionJson::Single, residual),
                PersistKind::Single
            );
        }
        assert_eq!(
            decode_precision(PrecisionJson::Mixed, ResidualJson::PromoteStorage),
            PersistKind::MixedPromote
        );
        assert_eq!(
            decode_precision(PrecisionJson::Mixed, ResidualJson::ReevaluateKernel),
            PersistKind::MixedReevaluate
        );
    }

    /// Decoding what a save wrote gives back the precision it saved.
    #[test]
    fn decode_inverts_the_save_encoding() {
        for kind in [
            PersistKind::Double,
            PersistKind::Single,
            PersistKind::MixedPromote,
            PersistKind::MixedReevaluate,
        ] {
            let pair = (
                PrecisionJson::from_persist(kind),
                ResidualJson::from_persist(kind),
            );
            assert_eq!(decode_precision(pair.0, pair.1), kind);
        }
    }
}
