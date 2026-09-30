//! Exact Gaussian process regression: `A = K + σn² I`, LLT or LDLT, and `α`.

mod exact_fit;
pub(crate) mod factor;
mod factor_store;
mod fitted;
mod objective;
pub(crate) mod online;
mod shared;
mod trainer;

#[cfg(test)]
mod tests;

pub(crate) use crate::workspace::FitBuffers;
pub(crate) use exact_fit::{ExactFit, LeafCache, fit_buffers};
pub(crate) use factor_store::{LdltStore, LltStore};
pub use fitted::FittedGpr;
pub(crate) use objective::GprObjective;
pub use online::OnlineGpr;
#[cfg(feature = "insert-stages")]
pub use online::take_insert_stages;
pub(crate) use shared::{GprCore, Policies};
pub use trainer::Gpr;
