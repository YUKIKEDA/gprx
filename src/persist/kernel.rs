//! Kernel tree encoding: closed tags for built-ins, `persist_id` for Custom,
//! and a leaf on supplied distances as its slot's number in the config's
//! slot table.

use serde::{Deserialize, Serialize};

use crate::error::GprError;
use crate::error::PersistErrorKind;
use crate::kernel::ArdLengthscales;
use crate::kernel::{
    ArdLeafSpec, ConstantKernel, CustomKernel, DistanceSlot, KernelSpec, LinearKernel,
    MaternArdKernel, MaternKernel, MaternNu, PeriodicKernel, RationalQuadraticArdKernel,
    RationalQuadraticKernel, RbfArdKernel, RbfKernel, ScalarLeafSpec, SlotShape, SuppliedLeafSpec,
    SuppliedSpec, Supply, WhiteKernel,
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
    /// A leaf on supplied distances: `leaf` (an isotropic or custom leaf
    /// for a scalar slot, an ARD leaf for an ARD slot) reads the slot
    /// numbered `slot` in the config's slot table.
    Distance {
        slot: usize,
        leaf: Box<KernelJson>,
    },
}

impl KernelJson {
    /// The JSON of `spec`; a leaf on supplied distances names its slot by
    /// its place in `slots` (the tree's [`crate::kernel::spec_slots`]).
    pub(super) fn encode<S: Supply>(
        spec: &KernelSpec<S>,
        slots: &[DistanceSlot],
    ) -> Result<Self, GprError> {
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
                left: Box::new(Self::encode(left, slots)?),
                right: Box::new(Self::encode(right, slots)?),
            }),
            KernelSpec::Supplied(leaf) => encode_supplied(S::spec(leaf), slots),
            KernelSpec::Product(left, right) => Ok(Self::Product {
                left: Box::new(Self::encode(left, slots)?),
                right: Box::new(Self::encode(right, slots)?),
            }),
        }
    }

    /// The coordinate tree of this JSON.
    ///
    /// # Errors
    ///
    /// As [`Self::decode_tree`]; a leaf on supplied distances is
    /// [`PersistErrorKind::WrongModel`].
    pub(super) fn decode(self, registry: &PersistRegistry) -> Result<KernelSpec, GprError> {
        self.decode_tree(registry, &[])
    }

    /// The tree of kind `S` of this JSON; a leaf on supplied distances reads
    /// `slots[slot]` (the decoded slot table).
    ///
    /// # Errors
    ///
    /// Returns [`GprError::PersistFailed`] with [`PersistErrorKind::WrongModel`]
    /// for a leaf on supplied distances in a coordinate tree, and with
    /// [`PersistErrorKind::Config`] for a slot number past the table or a
    /// leaf that does not fit its slot's shape, plus the errors of the
    /// leaves' constructors and the registry.
    pub(super) fn decode_tree<S: Supply>(
        self,
        registry: &PersistRegistry,
        slots: &[DistanceSlot],
    ) -> Result<KernelSpec<S>, GprError> {
        match self {
            Self::Sum { left, right } => Ok(KernelSpec::Sum(
                Box::new(left.decode_tree(registry, slots)?),
                Box::new(right.decode_tree(registry, slots)?),
            )),
            Self::Product { left, right } => Ok(KernelSpec::Product(
                Box::new(left.decode_tree(registry, slots)?),
                Box::new(right.decode_tree(registry, slots)?),
            )),
            Self::Distance { slot, leaf } => {
                let supplied = decode_supplied(slot, *leaf, registry, slots)?;
                S::from_spec(supplied)
                    .map(KernelSpec::Supplied)
                    .ok_or_else(|| {
                        persist_err(
                            PersistErrorKind::WrongModel,
                            "the kernel reads supplied distances; load it as a distance model",
                        )
                    })
            }
            leaf => Ok(leaf.decode_leaf(registry)?.widen()),
        }
    }

    /// One coordinate leaf.
    fn decode_leaf(self, registry: &PersistRegistry) -> Result<KernelSpec, GprError> {
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
            Self::Sum { .. } | Self::Product { .. } | Self::Distance { .. } => {
                self.decode_tree(registry, &[])
            }
        }
    }
}

/// The JSON of a leaf on supplied distances.
fn encode_supplied(leaf: &SuppliedSpec, slots: &[DistanceSlot]) -> Result<KernelJson, GprError> {
    let slot = slots
        .iter()
        .position(|slot| slot.id() == leaf.slot)
        .ok_or_else(|| {
            persist_err(
                PersistErrorKind::Config,
                "a distance leaf reads a slot the kernel does not list",
            )
        })?;
    let inner: KernelSpec = match &leaf.leaf {
        SuppliedLeafSpec::Scalar(ScalarLeafSpec::Rbf(k)) => KernelSpec::Rbf(*k),
        SuppliedLeafSpec::Scalar(ScalarLeafSpec::Matern(k)) => KernelSpec::Matern(*k),
        SuppliedLeafSpec::Scalar(ScalarLeafSpec::Periodic(k)) => KernelSpec::Periodic(*k),
        SuppliedLeafSpec::Scalar(ScalarLeafSpec::RationalQuadratic(k)) => {
            KernelSpec::RationalQuadratic(*k)
        }
        SuppliedLeafSpec::Scalar(ScalarLeafSpec::Custom(k)) => KernelSpec::Custom(k.clone()),
        SuppliedLeafSpec::Ard(ArdLeafSpec::Rbf(k)) => KernelSpec::RbfArd(k.clone()),
        SuppliedLeafSpec::Ard(ArdLeafSpec::Matern(k)) => KernelSpec::MaternArd(k.clone()),
        SuppliedLeafSpec::Ard(ArdLeafSpec::RationalQuadratic(k)) => {
            KernelSpec::RationalQuadraticArd(k.clone())
        }
    };
    Ok(KernelJson::Distance {
        slot,
        leaf: Box::new(KernelJson::encode(&inner, &[])?),
    })
}

/// The leaf on supplied distances of slot number `slot`: `leaf` must be
/// one its slot's shape holds (an ARD leaf with one lengthscale per
/// dimension for an ARD slot).
fn decode_supplied(
    slot: usize,
    leaf: KernelJson,
    registry: &PersistRegistry,
    slots: &[DistanceSlot],
) -> Result<SuppliedSpec, GprError> {
    let config = |reason: String| persist_err(PersistErrorKind::Config, reason);
    let table = slots.get(slot).copied().ok_or_else(|| {
        config(format!(
            "distance leaf names slot {slot}, but the slot table has {}",
            slots.len()
        ))
    })?;
    let inner = leaf.decode_leaf(registry)?;
    let leaf = match (table.shape(), inner) {
        (SlotShape::Scalar, KernelSpec::Rbf(k)) => SuppliedLeafSpec::Scalar(ScalarLeafSpec::Rbf(k)),
        (SlotShape::Scalar, KernelSpec::Matern(k)) => {
            SuppliedLeafSpec::Scalar(ScalarLeafSpec::Matern(k))
        }
        (SlotShape::Scalar, KernelSpec::Periodic(k)) => {
            SuppliedLeafSpec::Scalar(ScalarLeafSpec::Periodic(k))
        }
        (SlotShape::Scalar, KernelSpec::RationalQuadratic(k)) => {
            SuppliedLeafSpec::Scalar(ScalarLeafSpec::RationalQuadratic(k))
        }
        (SlotShape::Scalar, KernelSpec::Custom(k)) => {
            SuppliedLeafSpec::Scalar(ScalarLeafSpec::Custom(k))
        }
        (SlotShape::Ard(_), KernelSpec::RbfArd(k)) => SuppliedLeafSpec::Ard(ArdLeafSpec::Rbf(k)),
        (SlotShape::Ard(_), KernelSpec::MaternArd(k)) => {
            SuppliedLeafSpec::Ard(ArdLeafSpec::Matern(k))
        }
        (SlotShape::Ard(_), KernelSpec::RationalQuadraticArd(k)) => {
            SuppliedLeafSpec::Ard(ArdLeafSpec::RationalQuadratic(k))
        }
        (shape, _) => {
            return Err(config(format!(
                "distance leaf of slot {slot} does not fit a {shape:?} slot"
            )));
        }
    };
    if let (SlotShape::Ard(dims), SuppliedLeafSpec::Ard(ard)) = (table.shape(), &leaf)
        && ard.dims() != dims
    {
        return Err(config(format!(
            "ARD leaf of slot {slot} has {} lengthscales, the slot {dims} dimensions",
            ard.dims()
        )));
    }
    Ok(SuppliedSpec {
        slot: table.id(),
        at: 0,
        leaf,
    })
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
