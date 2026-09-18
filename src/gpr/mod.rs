//! Batch Gaussian process regression: `A = K + σn² I`, LLT, and `α`.

pub(crate) mod factor;
mod model;
mod types;

pub use model::{FittedGpr, Gpr};
pub use types::{
    AdaptiveJitter, DistanceCachePolicy, FixedJitter, JitterPolicy, PredictOptions, Prediction,
    VarianceKind,
};
