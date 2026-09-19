//! Batch Gaussian process regression: `A = K + σn² I`, LLT, and `α`.

pub(crate) mod factor;
mod model;
mod types;

pub use model::{FittedGpr, Gpr};
pub(crate) use types::DistanceCacheSlot;
pub use types::{
    AdaptiveJitter, DistanceCachePolicy, FixedJitter, JitterPolicy, NoDistanceCache,
    PredictOptions, Prediction, PredictiveCovariance, VarianceKind,
};
