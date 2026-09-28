//! Batch Gaussian process regression: `A = K + σn² I`, LLT, and `α`.

pub(crate) mod factor;
mod model;
pub(crate) mod online;
mod types;

pub use model::{FittedGpr, Gpr};
pub use online::OnlineGpr;
#[cfg(feature = "insert-stages")]
pub use online::take_insert_stages;
pub(crate) use types::AllocWorkspace;
pub use types::{
    AdaptiveJitter, CachedDistances, CholeskyBuffer, DistanceCachePolicy, FixedJitter,
    JitterPolicy, NoDistanceCache, PointId, PredictOptions, Prediction, PredictiveCovariance,
    RetainCholesky, ReuseCholesky, UncachedDistances, VarianceKind,
};
pub(crate) use types::{DistanceCachePersist, DistanceCacheSlot, FitBuffers};
