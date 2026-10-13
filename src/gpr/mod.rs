//! Exact Gaussian process regression: `A = K + σn² I`, LLT or LDLT, and `α`.

mod exact_fit;
pub(crate) mod factor;
mod factor_store;
mod fitted;
mod objective;
pub(crate) mod online;
mod shared;
mod trainer;

// Declared after the coordinate modules: rustdoc renders inherent impls in
// declaration order, so an intra-doc link to a method name shared by the
// coordinate and the supplied-distance impls lands on the coordinate one.
mod distance;

#[cfg(test)]
mod tests;

pub(crate) use crate::workspace::FitBuffers;
pub(crate) use exact_fit::{ExactFit, LeafCache, fit_buffers};
pub(crate) use factor_store::{LdltStore, LltStore};
pub use fitted::FittedGpr;
pub(crate) use fitted::TrainInput;
pub(crate) use objective::GprObjective;
pub use online::OnlineGpr;
#[cfg(feature = "insert-stages")]
pub use online::take_insert_stages;
pub(crate) use shared::{ExactSupplied, GprCore, Policies};
pub use trainer::Gpr;
