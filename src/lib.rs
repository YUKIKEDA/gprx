//! Exact Gaussian process regression with L-BFGS hyperparameter fitting.
//!
//! [`Gpr`] is the trainer. [`Gpr::fit`] consumes it and returns
//! [`FittedGpr`]. Training `X` is column-major: `n` points and `d`
//! features packed as feature 0 for all rows, then feature 1, and so on.
//! Observation noise lives in [`GaussianLikelihood`]. Do not also add a
//! large [`kernel::WhiteKernel`]. This crate is not published to crates.io
//! (`publish = false`).
//!
//! Distance fills and lower-triangle kernel writes use the process-wide
//! Rayon pool (shared with faer). There is no parallel on/off flag and no
//! `n_jobs` setter on [`Gpr`]. Set `RAYON_NUM_THREADS` before the process
//! starts, or call
//! `rayon::ThreadPoolBuilder::new().num_threads(n).build_global()` before
//! the first [`Gpr::fit`] / [`FittedGpr::predict`]. One worker
//! (`RAYON_NUM_THREADS=1`) is sequential. The global pool can be
//! initialized only once.
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
//! let gpr = Gpr::new(kernel, likelihood);
//! let fitted = gpr.fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0]).map_err(|(_, e)| e)?;
//! let pred = fitted.predict(&[0.5], 1, 1)?;
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
pub use gpr::{
    DistanceCachePolicy, FitOptions, FittedGpr, Gpr, PredictOptions, Prediction, VarianceKind,
};
pub use likelihood::GaussianLikelihood;
pub use precision::{DoublePrecision, PrecisionPolicy};

#[cfg(test)]
mod tests {
    #[test]
    fn crate_compiles() {}
}
