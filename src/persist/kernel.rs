//! Kernel tree encoding: closed tags for built-ins, `persist_id` for Custom.

use serde::{Deserialize, Serialize};

use crate::error::GprError;
use crate::error::PersistErrorKind;
use crate::kernel::ArdLengthscales;
use crate::kernel::{
    ConstantKernel, CustomKernel, DistanceSlot, KernelSpec, LinearKernel, MaternArdKernel,
    MaternKernel, MaternNu, PeriodicKernel, RationalQuadraticArdKernel, RationalQuadraticKernel,
    RbfArdKernel, RbfKernel, SlotId, SlotShape, SuppliedSpec, WhiteKernel, spec_slots,
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
    /// A leaf on supplied squared distances: the slot's index in the order
    /// of first appearance, its `d²` count per pair (`dims`, absent for a
    /// scalar slot), and the leaf it evaluates.
    Distance {
        slot: usize,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        dims: Option<usize>,
        leaf: Box<KernelJson>,
    },
}

/// Fresh slots a decoded distance kernel gets, one per saved slot index.
#[derive(Default)]
pub(super) struct DecodedSlots {
    slots: Vec<(SlotId, SlotShape)>,
}

impl DecodedSlots {
    fn get(&mut self, index: usize, shape: SlotShape) -> Result<SlotId, GprError> {
        match self.slots.get(index) {
            Some(&(id, saved)) if saved == shape => Ok(id),
            Some(_) => Err(persist_err(
                PersistErrorKind::Config,
                format!("distance slot {index} is saved with two shapes"),
            )),
            None if index == self.slots.len() => {
                let id = SlotId::fresh();
                self.slots.push((id, shape));
                Ok(id)
            }
            None => Err(persist_err(
                PersistErrorKind::Config,
                format!("distance slot {index} appears before slot {}", self.slots.len()),
            )),
        }
    }
}

impl KernelJson {
    pub(super) fn encode(spec: &KernelSpec) -> Result<Self, GprError> {
        let slots = spec_slots(spec);
        Self::encode_in(spec, &slots)
    }

    fn encode_in(spec: &KernelSpec, slots: &[DistanceSlot]) -> Result<Self, GprError> {
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
                left: Box::new(Self::encode_in(left, slots)?),
                right: Box::new(Self::encode_in(right, slots)?),
            }),
            KernelSpec::Product(left, right) => Ok(Self::Product {
                left: Box::new(Self::encode_in(left, slots)?),
                right: Box::new(Self::encode_in(right, slots)?),
            }),
            KernelSpec::Supplied(leaf) => {
                let slot = slots
                    .iter()
                    .position(|s| s.id() == leaf.slot)
                    .ok_or_else(|| {
                        persist_err(PersistErrorKind::Config, "distance slot is not listed")
                    })?;
                Ok(Self::Distance {
                    slot,
                    dims: match leaf.shape {
                        SlotShape::Scalar => None,
                        SlotShape::Ard(dims) => Some(dims),
                    },
                    leaf: Box::new(Self::encode_in(&leaf.leaf, slots)?),
                })
            }
        }
    }

    /// Decodes the kernel of a coordinate model. A saved distance kernel is
    /// another model and is not read as this one.
    pub(super) fn decode_points(self, registry: &PersistRegistry) -> Result<KernelSpec, GprError> {
        let mut slots = DecodedSlots::default();
        let spec = self.decode(registry, &mut slots)?;
        if slots.slots.is_empty() {
            Ok(spec)
        } else {
            Err(persist_err(
                PersistErrorKind::WrongModel,
                "the saved model reads supplied distances; load it as a distance model",
            ))
        }
    }

    /// Decodes a kernel; distance leaves get fresh slots in `slots`.
    pub(super) fn decode(
        self,
        registry: &PersistRegistry,
        slots: &mut DecodedSlots,
    ) -> Result<KernelSpec, GprError> {
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
            Self::Custom { persist_id, state } => Ok(KernelSpec::from(
                registry.restore_kernel(&persist_id, &state)?,
            )),
            Self::Sum { left, right } => Ok(KernelSpec::Sum(
                Box::new(left.decode(registry, slots)?),
                Box::new(right.decode(registry, slots)?),
            )),
            Self::Product { left, right } => Ok(KernelSpec::Product(
                Box::new(left.decode(registry, slots)?),
                Box::new(right.decode(registry, slots)?),
            )),
            Self::Distance { slot, dims, leaf } => {
                let shape = match dims {
                    None => SlotShape::Scalar,
                    Some(dims) => SlotShape::Ard(dims),
                };
                let id = slots.get(slot, shape)?;
                let leaf = leaf.decode(registry, slots)?;
                check_distance_leaf(&leaf, shape)?;
                Ok(KernelSpec::Supplied(SuppliedSpec {
                    slot: id,
                    shape,
                    leaf: Box::new(leaf),
                }))
            }
        }
    }
}

/// A saved distance leaf is one a slot of `shape` accepts.
fn check_distance_leaf(leaf: &KernelSpec, shape: SlotShape) -> Result<(), GprError> {
    let ok = match (leaf, shape) {
        (
            KernelSpec::Rbf(_)
            | KernelSpec::Matern(_)
            | KernelSpec::Periodic(_)
            | KernelSpec::RationalQuadratic(_)
            | KernelSpec::Custom(_),
            SlotShape::Scalar,
        ) => true,
        (KernelSpec::RbfArd(k), SlotShape::Ard(d)) => k.num_params() == d,
        (KernelSpec::MaternArd(k), SlotShape::Ard(d)) => k.num_params() == d,
        (KernelSpec::RationalQuadraticArd(k), SlotShape::Ard(d)) => {
            k.lengthscales().num_params() == d
        }
        _ => false,
    };
    if ok {
        Ok(())
    } else {
        Err(persist_err(
            PersistErrorKind::Config,
            "a distance leaf does not match its slot",
        ))
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
            PersistErrorKind::InvalidPersistId,
            "custom kernel persist_id is empty; implement KernelTerm::persist_id",
        ));
    }
    if persist_id.starts_with(RESERVED_PREFIX) {
        return Err(persist_err(
            PersistErrorKind::InvalidPersistId,
            format!("persist_id {persist_id:?} uses the reserved {RESERVED_PREFIX} prefix"),
        ));
    }
    Ok(KernelJson::Custom {
        persist_id: persist_id.to_owned(),
        state: kernel.persist_state()?,
    })
}
