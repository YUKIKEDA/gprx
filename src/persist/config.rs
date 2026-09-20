//! `config.json` schema for a fitted model directory.

use serde::{Deserialize, Serialize};

use crate::error::GprError;
use crate::gpr::DistanceCachePersist;
use crate::param::{BoundedParam, Interval};
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
    pub kernel: KernelJson,
    pub likelihood: LikelihoodJson,
    pub jitter: JitterJson,
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
        let ids = self
            .point_ids
            .as_deref()
            .ok_or_else(|| persist_err("ldlt config missing point_ids"))?;
        let next_id = self
            .next_point_id
            .ok_or_else(|| persist_err("ldlt config missing next_point_id"))?;
        if ids.len() != self.n {
            return Err(persist_err(format!(
                "point_ids has {} values, expected n = {}",
                ids.len(),
                self.n
            )));
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
    pub(super) fn encode(policy: DistanceCachePersist) -> Self {
        match policy {
            DistanceCachePersist::Cached => Self::Always,
            DistanceCachePersist::Uncached => Self::Never,
        }
    }

    pub(super) fn decode(self) -> DistanceCachePersist {
        match self {
            Self::Always => DistanceCachePersist::Cached,
            Self::Never => DistanceCachePersist::Uncached,
        }
    }
}

pub(super) fn parse_config(bytes: &[u8]) -> Result<ModelConfig, GprError> {
    let config: ModelConfig = serde_json::from_slice(bytes)
        .map_err(|err| persist_err(format!("config.json is not valid JSON: {err}")))?;
    config.validate_version()?;
    if config.n == 0 || config.d == 0 {
        return Err(GprError::EmptyInput);
    }
    Ok(config)
}
