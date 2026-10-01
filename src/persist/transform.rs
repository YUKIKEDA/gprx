//! Transform encoding: closed tags for built-ins, `persist_id` for user maps.

use serde::{Deserialize, Serialize};

use crate::error::GprError;
use crate::error::PersistErrorKind;
use crate::transform::{
    ColumnwiseInput, FittedColumnwiseInput, FittedMinMaxInput, FittedMinMaxTarget, FittedPipeline,
    FittedStandardizeInput, FittedStandardizeTarget, FittedTargetPipeline, IdentityInput,
    IdentityTarget, MinMaxInput, MinMaxTarget, Pipeline, StandardizeInput, StandardizeTarget,
    TargetPipeline, TargetTransform, Transform, UnfittedTarget, UnfittedTransform,
};

use super::RESERVED_PREFIX;
use super::persist_err;
use super::registry::PersistRegistry;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum UnfittedInputJson {
    Identity,
    Standardize,
    MinMax {
        range_lo: f64,
        range_hi: f64,
    },
    Pipeline {
        steps: Vec<UnfittedInputJson>,
    },
    Columnwise {
        maps: Vec<UnfittedInputJson>,
    },
    Custom {
        persist_id: String,
        state: serde_json::Value,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum FittedInputJson {
    Identity,
    Standardize {
        mean: Vec<f64>,
        std: Vec<f64>,
    },
    MinMax {
        data_min: Vec<f64>,
        data_max: Vec<f64>,
        range_lo: f64,
        range_hi: f64,
    },
    Pipeline {
        steps: Vec<FittedInputJson>,
    },
    Columnwise {
        maps: Vec<FittedInputJson>,
    },
    Custom {
        persist_id: String,
        state: serde_json::Value,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum UnfittedTargetJson {
    Identity,
    Standardize,
    MinMax {
        range_lo: f64,
        range_hi: f64,
    },
    Pipeline {
        steps: Vec<UnfittedTargetJson>,
    },
    Custom {
        persist_id: String,
        state: serde_json::Value,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum FittedTargetJson {
    Identity,
    Standardize {
        mean: f64,
        std: f64,
    },
    MinMax {
        data_min: f64,
        data_max: f64,
        range_lo: f64,
        range_hi: f64,
    },
    Pipeline {
        steps: Vec<FittedTargetJson>,
    },
    Custom {
        persist_id: String,
        state: serde_json::Value,
    },
}

pub(super) fn encode_unfitted_input(
    transform: &dyn UnfittedTransform,
) -> Result<UnfittedInputJson, GprError> {
    if transform.as_any().downcast_ref::<IdentityInput>().is_some() {
        return Ok(UnfittedInputJson::Identity);
    }
    if transform
        .as_any()
        .downcast_ref::<StandardizeInput>()
        .is_some()
    {
        return Ok(UnfittedInputJson::Standardize);
    }
    if let Some(t) = transform.as_any().downcast_ref::<MinMaxInput>() {
        let (range_lo, range_hi) = t.feature_range();
        return Ok(UnfittedInputJson::MinMax { range_lo, range_hi });
    }
    if let Some(t) = transform.as_any().downcast_ref::<Pipeline>() {
        let steps = t
            .steps()
            .iter()
            .map(|step| encode_unfitted_input(step.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(UnfittedInputJson::Pipeline { steps });
    }
    if let Some(t) = transform.as_any().downcast_ref::<ColumnwiseInput>() {
        let maps = t
            .maps()
            .iter()
            .map(|map| encode_unfitted_input(map.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(UnfittedInputJson::Columnwise { maps });
    }
    Ok(UnfittedInputJson::Custom {
        persist_id: require_id(transform.persist_id(), "input transform")?,
        state: transform.persist_state()?,
    })
}

pub(super) fn encode_fitted_input(transform: &dyn Transform) -> Result<FittedInputJson, GprError> {
    if transform.as_any().downcast_ref::<IdentityInput>().is_some() {
        return Ok(FittedInputJson::Identity);
    }
    if let Some(t) = transform.as_any().downcast_ref::<FittedStandardizeInput>() {
        return Ok(FittedInputJson::Standardize {
            mean: t.mean().to_vec(),
            std: t.std().to_vec(),
        });
    }
    if let Some(t) = transform.as_any().downcast_ref::<FittedMinMaxInput>() {
        let (range_lo, range_hi) = t.feature_range();
        return Ok(FittedInputJson::MinMax {
            data_min: t.min().to_vec(),
            data_max: t.max().to_vec(),
            range_lo,
            range_hi,
        });
    }
    if let Some(t) = transform.as_any().downcast_ref::<FittedPipeline>() {
        let steps = t
            .steps()
            .iter()
            .map(|step| encode_fitted_input(step.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(FittedInputJson::Pipeline { steps });
    }
    if let Some(t) = transform.as_any().downcast_ref::<FittedColumnwiseInput>() {
        let maps = t
            .maps()
            .iter()
            .map(|map| encode_fitted_input(map.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(FittedInputJson::Columnwise { maps });
    }
    Ok(FittedInputJson::Custom {
        persist_id: require_id(transform.persist_id(), "fitted input transform")?,
        state: transform.persist_state()?,
    })
}

pub(super) fn encode_unfitted_target(
    transform: &dyn UnfittedTarget,
) -> Result<UnfittedTargetJson, GprError> {
    if transform
        .as_any()
        .downcast_ref::<IdentityTarget>()
        .is_some()
    {
        return Ok(UnfittedTargetJson::Identity);
    }
    if transform
        .as_any()
        .downcast_ref::<StandardizeTarget>()
        .is_some()
    {
        return Ok(UnfittedTargetJson::Standardize);
    }
    if let Some(t) = transform.as_any().downcast_ref::<MinMaxTarget>() {
        let (range_lo, range_hi) = t.feature_range();
        return Ok(UnfittedTargetJson::MinMax { range_lo, range_hi });
    }
    if let Some(t) = transform.as_any().downcast_ref::<TargetPipeline>() {
        let steps = t
            .steps()
            .iter()
            .map(|step| encode_unfitted_target(step.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(UnfittedTargetJson::Pipeline { steps });
    }
    Ok(UnfittedTargetJson::Custom {
        persist_id: require_id(transform.persist_id(), "target transform")?,
        state: transform.persist_state()?,
    })
}

pub(super) fn encode_fitted_target(
    transform: &dyn TargetTransform,
) -> Result<FittedTargetJson, GprError> {
    if transform
        .as_any()
        .downcast_ref::<IdentityTarget>()
        .is_some()
    {
        return Ok(FittedTargetJson::Identity);
    }
    if let Some(t) = transform.as_any().downcast_ref::<FittedStandardizeTarget>() {
        return Ok(FittedTargetJson::Standardize {
            mean: t.mean(),
            std: t.std(),
        });
    }
    if let Some(t) = transform.as_any().downcast_ref::<FittedMinMaxTarget>() {
        let (range_lo, range_hi) = t.feature_range();
        return Ok(FittedTargetJson::MinMax {
            data_min: t.min(),
            data_max: t.max(),
            range_lo,
            range_hi,
        });
    }
    if let Some(t) = transform.as_any().downcast_ref::<FittedTargetPipeline>() {
        let steps = t
            .steps()
            .iter()
            .map(|step| encode_fitted_target(step.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(FittedTargetJson::Pipeline { steps });
    }
    Ok(FittedTargetJson::Custom {
        persist_id: require_id(transform.persist_id(), "fitted target transform")?,
        state: transform.persist_state()?,
    })
}

impl UnfittedInputJson {
    pub(super) fn decode(
        self,
        registry: &PersistRegistry,
    ) -> Result<Box<dyn UnfittedTransform>, GprError> {
        match self {
            Self::Identity => Ok(Box::new(IdentityInput)),
            Self::Standardize => Ok(Box::new(StandardizeInput::new())),
            Self::MinMax { range_lo, range_hi } => Ok(Box::new(MinMaxInput::with_feature_range(
                range_lo, range_hi,
            )?)),
            Self::Pipeline { steps } => {
                let steps = steps
                    .into_iter()
                    .map(|step| step.decode(registry))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Box::new(Pipeline::from_steps(steps)))
            }
            Self::Columnwise { maps } => {
                let maps = maps
                    .into_iter()
                    .map(|map| map.decode(registry))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Box::new(ColumnwiseInput::from_maps(maps)))
            }
            Self::Custom { persist_id, state } => {
                registry.restore_unfitted_input(&persist_id, &state)
            }
        }
    }
}

impl FittedInputJson {
    pub(super) fn decode(self, registry: &PersistRegistry) -> Result<Box<dyn Transform>, GprError> {
        match self {
            Self::Identity => Ok(Box::new(IdentityInput)),
            Self::Standardize { mean, std } => {
                Ok(Box::new(FittedStandardizeInput::from_parts(mean, std)?))
            }
            Self::MinMax {
                data_min,
                data_max,
                range_lo,
                range_hi,
            } => Ok(Box::new(FittedMinMaxInput::from_parts(
                data_min, data_max, range_lo, range_hi,
            )?)),
            Self::Pipeline { steps } => {
                let steps = steps
                    .into_iter()
                    .map(|step| step.decode(registry))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Box::new(FittedPipeline::from_steps(steps)))
            }
            Self::Columnwise { maps } => {
                let maps = maps
                    .into_iter()
                    .map(|map| map.decode(registry))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Box::new(FittedColumnwiseInput::from_maps(maps)))
            }
            Self::Custom { persist_id, state } => {
                registry.restore_fitted_input(&persist_id, &state)
            }
        }
    }
}

impl UnfittedTargetJson {
    pub(super) fn decode(
        self,
        registry: &PersistRegistry,
    ) -> Result<Box<dyn UnfittedTarget>, GprError> {
        match self {
            Self::Identity => Ok(Box::new(IdentityTarget)),
            Self::Standardize => Ok(Box::new(StandardizeTarget::new())),
            Self::MinMax { range_lo, range_hi } => Ok(Box::new(MinMaxTarget::with_feature_range(
                range_lo, range_hi,
            )?)),
            Self::Pipeline { steps } => {
                let steps = steps
                    .into_iter()
                    .map(|step| step.decode(registry))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Box::new(TargetPipeline::from_steps(steps)))
            }
            Self::Custom { persist_id, state } => {
                registry.restore_unfitted_target(&persist_id, &state)
            }
        }
    }
}

impl FittedTargetJson {
    pub(super) fn decode(
        self,
        registry: &PersistRegistry,
    ) -> Result<Box<dyn TargetTransform>, GprError> {
        match self {
            Self::Identity => Ok(Box::new(IdentityTarget)),
            Self::Standardize { mean, std } => {
                Ok(Box::new(FittedStandardizeTarget::from_parts(mean, std)?))
            }
            Self::MinMax {
                data_min,
                data_max,
                range_lo,
                range_hi,
            } => Ok(Box::new(FittedMinMaxTarget::from_parts(
                data_min, data_max, range_lo, range_hi,
            )?)),
            Self::Pipeline { steps } => {
                let steps = steps
                    .into_iter()
                    .map(|step| step.decode(registry))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Box::new(FittedTargetPipeline::from_steps(steps)))
            }
            Self::Custom { persist_id, state } => {
                registry.restore_fitted_target(&persist_id, &state)
            }
        }
    }
}

fn require_id(id: Option<&'static str>, what: &str) -> Result<String, GprError> {
    let id = id.ok_or_else(|| {
        persist_err(
            PersistErrorKind::NotPersistable,
            format!("{what} is not a built-in and does not implement persist_id"),
        )
    })?;
    if id.is_empty() {
        return Err(persist_err(
            PersistErrorKind::InvalidPersistId,
            format!("{what} persist_id is empty"),
        ));
    }
    if id.starts_with(RESERVED_PREFIX) {
        return Err(persist_err(
            PersistErrorKind::InvalidPersistId,
            format!("persist_id {id:?} uses the reserved {RESERVED_PREFIX} prefix"),
        ));
    }
    Ok(id.to_owned())
}
