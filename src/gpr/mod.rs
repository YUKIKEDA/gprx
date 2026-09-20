//! Batch Gaussian process regression: `A = K + σn² I`, LLT, and `α`.

pub(crate) mod factor;
mod model;
mod online;
mod types;

pub use model::{FittedGpr, Gpr};
pub use online::OnlineGpr;
pub(crate) use types::AllocWorkspace;
pub use types::{
    AdaptiveJitter, CachedDistances, CholeskyBuffer, DistanceCachePolicy, FixedJitter,
    JitterPolicy, NoDistanceCache, PredictOptions, Prediction, PredictiveCovariance,
    RetainCholesky, ReuseCholesky, UncachedDistances, VarianceKind,
};
pub(crate) use types::{DistanceCachePersist, DistanceCacheSlot, FitBuffers};
