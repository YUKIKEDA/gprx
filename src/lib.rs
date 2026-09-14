//! Flexible high-performance Gaussian process regression.
//!
//! This crate is under active development. The public API will grow in
//! Phase 1 (`ExactGP`, kernels, and `fit` / `predict`). See the repository
//! `AGENTS.md` and `.dev/roadmap.md` for the current milestone.

mod error;
mod likelihood;
mod precision;
mod workspace;

pub use error::{CholeskyStage, GpError};
pub use likelihood::GaussianLikelihood;
pub use precision::{DoublePrecision, PrecisionPolicy};

#[cfg(test)]
mod tests {
    #[test]
    fn crate_compiles() {}
}
