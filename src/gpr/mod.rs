//! Batch Gaussian process regression: `A = K + σn² I`, LLT, and `α`.

pub(crate) mod factor;
mod model;
pub(crate) mod online;
mod types;

pub(crate) use crate::workspace::FitBuffers;
pub(crate) use model::Policies;
pub use model::{FittedGpr, Gpr};
pub use online::OnlineGpr;
#[cfg(feature = "insert-stages")]
pub use online::take_insert_stages;
pub(crate) use types::with_kernel_exp;
pub use types::{
    AdaptiveJitter, CholeskyBuffer, DistanceCachePolicy, FixedJitter, JitterPolicy, KernelExp,
    PointId, PredictOptions, Prediction, PredictiveCovariance, VarianceKind,
};
