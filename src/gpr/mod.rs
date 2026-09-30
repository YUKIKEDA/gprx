//! Exact Gaussian process regression: `A = K + σn² I`, LLT or LDLT, and `α`.

mod exact_fit;
pub(crate) mod factor;
mod factor_store;
mod fitted;
pub(crate) mod online;
mod points;
mod policy;
mod prediction;
mod shared;
mod trainer;

#[cfg(test)]
mod tests;

pub(crate) use crate::workspace::FitBuffers;
pub(crate) use exact_fit::{ExactFit, LeafCache, fit_buffers};
pub(crate) use factor_store::{LdltStore, LltStore};
pub use fitted::FittedGpr;
pub use online::OnlineGpr;
#[cfg(feature = "insert-stages")]
pub use online::take_insert_stages;
pub use points::PointId;
pub(crate) use points::PointRegistry;
pub use policy::{
    AdaptiveJitter, CholeskyBuffer, DistanceCachePolicy, FixedJitter, JitterPolicy, KernelExp,
};
pub(crate) use policy::{Policies, with_kernel_exp};
pub use prediction::{PredictOptions, Prediction, PredictiveCovariance, VarianceKind};
pub(crate) use shared::GprCore;
pub use trainer::Gpr;
