//! Flexible high-performance Gaussian process regression.
//!
//! This crate is under active development. The public API will grow in
//! Phase 1 (`Gpr`, kernels, and `fit` / `predict`). See the repository
//! `AGENTS.md` and `.dev/roadmap.md` for the current milestone.

mod error;
mod gpr;
pub mod kernel;
mod likelihood;
mod objective;
mod precision;
pub mod transform;
mod workspace;

pub use error::{CholeskyStage, GprError};
pub use gpr::{Gpr, PredictOptions, Prediction, VarianceKind};
pub use likelihood::GaussianLikelihood;
pub use precision::{DoublePrecision, PrecisionPolicy};

#[cfg(test)]
mod tests {
    #[test]
    fn crate_compiles() {}
}
