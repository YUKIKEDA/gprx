//! Batch Gaussian process regression: `A = K + σn² I`, LLT, and `α`.

pub(crate) mod factor;
mod model;
mod types;

pub use model::{FittedGpr, Gpr};
pub(crate) use types::AllocWorkspace;
pub(crate) use types::DistanceCacheSlot;
pub use types::{
    AdaptiveJitter, CholeskyBuffer, DistanceCachePolicy, FixedJitter, JitterPolicy,
    NoDistanceCache, PredictOptions, Prediction, PredictiveCovariance, RetainCholesky,
    ReuseCholesky, VarianceKind,
};
