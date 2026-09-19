//! Kernel tree encoding: closed tags for built-ins, `persist_id` for Custom.

use serde::{Deserialize, Serialize};

use crate::error::GprError;
use crate::kernel::ArdLengthscales;
use crate::kernel::{
    ConstantKernel, CustomKernel, KernelSpec, LinearKernel, MaternArdKernel, MaternKernel,
    MaternNu, PeriodicKernel, RationalQuadraticArdKernel, RationalQuadraticKernel, RbfArdKernel,
    RbfKernel, WhiteKernel,
};
use crate::param::BoundedParam;

use super::RESERVED_PREFIX;
use super::config::BoundedJson;
use super::persist_err;
use super::registry::PersistRegistry;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum MaternNuJson {
    Half,
    ThreeHalves,
    FiveHalves,
}

impl MaternNuJson {
    fn encode(nu: MaternNu) -> Self {
        match nu {
            MaternNu::Half => Self::Half,
            MaternNu::ThreeHalves => Self::ThreeHalves,
            MaternNu::FiveHalves => Self::FiveHalves,
        }
    }

    fn decode(self) -> MaternNu {
        match self {
            Self::Half => MaternNu::Half,
            Self::ThreeHalves => MaternNu::ThreeHalves,
            Self::FiveHalves => MaternNu::FiveHalves,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum KernelJson {
    Rbf {
        lengthscale: BoundedJson,
    },
    RbfArd {
        lengthscales: Vec<BoundedJson>,
    },
    Matern {
        lengthscale: BoundedJson,
        nu: MaternNuJson,
    },
    MaternArd {
        lengthscales: Vec<BoundedJson>,
        nu: MaternNuJson,
    },
    Periodic {
        lengthscale: BoundedJson,
        period: BoundedJson,
    },
    RationalQuadratic {
        lengthscale: BoundedJson,
        alpha: BoundedJson,
    },
    RationalQuadraticArd {
        lengthscales: Vec<BoundedJson>,
        alpha: BoundedJson,
    },
    Constant {
        constant: BoundedJson,
    },
    Linear {
        variance: BoundedJson,
    },
    White {
        variance: BoundedJson,
    },
    Custom {
        persist_id: String,
        state: serde_json::Value,
    },
    Sum {
        left: Box<KernelJson>,
        right: Box<KernelJson>,
    },
    Product {
        left: Box<KernelJson>,
        right: Box<KernelJson>,
    },
}

impl KernelJson {
    pub(super) fn encode(spec: &KernelSpec) -> Result<Self, GprError> {
        match spec {
            KernelSpec::Rbf(k) => Ok(Self::Rbf {
                lengthscale: BoundedJson::from_param(bounded_from_value(
                    k.lengthscale(),
                    k.bounds(),
                )?),
            }),
            KernelSpec::RbfArd(k) => Ok(Self::RbfArd {
                lengthscales: encode_ard(k.lengthscales())?,
            }),
            KernelSpec::Matern(k) => Ok(Self::Matern {
                lengthscale: BoundedJson::from_param(bounded_from_value(
                    k.lengthscale(),
                    k.bounds(),
                )?),
                nu: MaternNuJson::encode(k.nu()),
            }),
            KernelSpec::MaternArd(k) => Ok(Self::MaternArd {
                lengthscales: encode_ard(k.lengthscales())?,
                nu: MaternNuJson::encode(k.nu()),
            }),
            KernelSpec::Periodic(k) => Ok(Self::Periodic {
                lengthscale: BoundedJson::from_param(bounded_from_value(
                    k.lengthscale(),
                    k.lengthscale_bounds(),
                )?),
                period: BoundedJson::from_param(bounded_from_value(k.period(), k.period_bounds())?),
            }),
            KernelSpec::RationalQuadratic(k) => Ok(Self::RationalQuadratic {
                lengthscale: BoundedJson::from_param(bounded_from_value(
                    k.lengthscale(),
                    k.lengthscale_bounds(),
                )?),
                alpha: BoundedJson::from_param(bounded_from_value(k.alpha(), k.alpha_bounds())?),
            }),
            KernelSpec::RationalQuadraticArd(k) => Ok(Self::RationalQuadraticArd {
                lengthscales: encode_ard(k.lengthscales())?,
                alpha: BoundedJson::from_param(bounded_from_value(k.alpha(), k.alpha_bounds())?),
            }),
            KernelSpec::Constant(k) => Ok(Self::Constant {
                constant: BoundedJson::from_param(bounded_from_value(k.constant(), k.bounds())?),
            }),
            KernelSpec::Linear(k) => Ok(Self::Linear {
                variance: BoundedJson::from_param(bounded_from_value(k.variance(), k.bounds())?),
            }),
            KernelSpec::White(k) => Ok(Self::White {
                variance: BoundedJson::from_param(bounded_from_value(k.variance(), k.bounds())?),
            }),
            KernelSpec::Custom(k) => encode_custom(k),
            KernelSpec::Sum(left, right) => Ok(Self::Sum {
                left: Box::new(Self::encode(left)?),
                right: Box::new(Self::encode(right)?),
            }),
            KernelSpec::Product(left, right) => Ok(Self::Product {
                left: Box::new(Self::encode(left)?),
                right: Box::new(Self::encode(right)?),
            }),
        }
    }

    pub(super) fn decode(self, registry: &PersistRegistry) -> Result<KernelSpec, GprError> {
        match self {
            Self::Rbf { lengthscale } => {
                let k = RbfKernel::new(lengthscale.value)?.with_bounds(lengthscale.interval()?)?;
                Ok(KernelSpec::from(k))
            }
            Self::RbfArd { lengthscales } => Ok(KernelSpec::from(RbfArdKernel::from_ard(
                decode_ard(lengthscales)?,
            ))),
            Self::Matern { lengthscale, nu } => {
                let k = MaternKernel::new(lengthscale.value, nu.decode())?
                    .with_bounds(lengthscale.interval()?)?;
                Ok(KernelSpec::from(k))
            }
            Self::MaternArd { lengthscales, nu } => Ok(KernelSpec::from(
                MaternArdKernel::from_ard(decode_ard(lengthscales)?, nu.decode()),
            )),
            Self::Periodic {
                lengthscale,
                period,
            } => {
                let k = PeriodicKernel::new(lengthscale.value, period.value)?
                    .with_bounds(lengthscale.interval()?, period.interval()?)?;
                Ok(KernelSpec::from(k))
            }
            Self::RationalQuadratic { lengthscale, alpha } => {
                let k = RationalQuadraticKernel::new(lengthscale.value, alpha.value)?
                    .with_bounds(lengthscale.interval()?, alpha.interval()?)?;
                Ok(KernelSpec::from(k))
            }
            Self::RationalQuadraticArd {
                lengthscales,
                alpha,
            } => Ok(KernelSpec::from(RationalQuadraticArdKernel::from_ard(
                decode_ard(lengthscales)?,
                BoundedParam::new(alpha.value, alpha.interval()?)?,
            ))),
            Self::Constant { constant } => {
                let k = ConstantKernel::new(constant.value)?.with_bounds(constant.interval()?)?;
                Ok(KernelSpec::from(k))
            }
            Self::Linear { variance } => {
                let k = LinearKernel::new(variance.value)?.with_bounds(variance.interval()?)?;
                Ok(KernelSpec::from(k))
            }
            Self::White { variance } => {
                let k = WhiteKernel::new(variance.value)?.with_bounds(variance.interval()?)?;
                Ok(KernelSpec::from(k))
            }
            Self::Custom { persist_id, state } => {
                let term = registry.restore_kernel(&persist_id, &state)?;
                Ok(KernelSpec::from(crate::kernel::CustomKernel::from_box(
                    term,
                )))
            }
            Self::Sum { left, right } => Ok(KernelSpec::Sum(
                Box::new(left.decode(registry)?),
                Box::new(right.decode(registry)?),
            )),
            Self::Product { left, right } => Ok(KernelSpec::Product(
                Box::new(left.decode(registry)?),
                Box::new(right.decode(registry)?),
            )),
        }
    }
}

fn bounded_from_value(value: f64, interval: crate::Interval) -> Result<BoundedParam, GprError> {
    Ok(BoundedParam::new(value, interval)?)
}

fn encode_ard(scales: &ArdLengthscales) -> Result<Vec<BoundedJson>, GprError> {
    Ok(scales
        .bounded_params()
        .iter()
        .copied()
        .map(BoundedJson::from_param)
        .collect())
}

fn decode_ard(values: Vec<BoundedJson>) -> Result<ArdLengthscales, GprError> {
    if values.is_empty() {
        return Err(GprError::EmptyInput);
    }
    let params = values
        .into_iter()
        .map(BoundedJson::into_param)
        .collect::<Result<Vec<_>, _>>()?;
    ArdLengthscales::from_bounded(params)
}

fn encode_custom(kernel: &CustomKernel) -> Result<KernelJson, GprError> {
    let persist_id = kernel.persist_id();
    if persist_id.is_empty() {
        return Err(persist_err(
            "custom kernel persist_id is empty; implement KernelTerm::persist_id",
        ));
    }
    if persist_id.starts_with(RESERVED_PREFIX) {
        return Err(persist_err(format!(
            "persist_id {persist_id:?} uses the reserved {RESERVED_PREFIX} prefix"
        )));
    }
    Ok(KernelJson::Custom {
        persist_id: persist_id.to_owned(),
        state: kernel.persist_state()?,
    })
}
