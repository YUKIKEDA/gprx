//! Exact Gaussian process regression with L-BFGS hyperparameter fitting.
//!
//! Training `X` is column-major: `n` points and `d` features packed as
//! feature 0 for all rows, then feature 1, and so on. Observation noise
//! lives in [`GaussianLikelihood`]. Do not also add a large
//! [`kernel::WhiteKernel`]. This crate is not published to crates.io
//! (`publish = false`).
//!
//! # Examples
//!
//! ```rust
//! use gprx::kernel::{KernelSpec, RbfKernel};
//! use gprx::{GaussianLikelihood, Gpr};
//!
//! # fn main() -> Result<(), gprx::GprError> {
//! let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
//! let likelihood = GaussianLikelihood::new(0.1)?;
//! let mut gpr = Gpr::new(kernel, likelihood);
//! gpr.fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0])?;
//! let pred = gpr.predict(&[0.5], 1, 1)?;
//! assert_eq!(pred.mean.len(), 1);
//! # Ok(())
//! # }
//! ```

mod error;
mod gpr;
pub mod kernel;
mod likelihood;
mod objective;
mod optimizer;
mod precision;
pub mod transform;
mod workspace;

pub use error::{CholeskyStage, GprError};
pub use gpr::{DistanceCachePolicy, FitOptions, Gpr, PredictOptions, Prediction, VarianceKind};
pub use likelihood::GaussianLikelihood;
pub use precision::{DoublePrecision, PrecisionPolicy};

#[cfg(test)]
mod tests {
    #[test]
    fn crate_compiles() {}
}
