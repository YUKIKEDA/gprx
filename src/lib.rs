//! Exact Gaussian process regression with L-BFGS hyperparameter fitting.
//!
//! [`Gpr`] is the trainer. [`Gpr::fit`] consumes [`Gpr<Lbfgs>`] and returns
//! [`FittedGpr`]. [`Gpr::with_optimizer`] swaps in [`NonlinearCg`],
//! [`NelderMead`], [`Newton`], or [`FastSimulatedAnnealing`]. [`Gpr<Fixed>::factor`] factors at the current `θ` without
//! a search. Training `X` is column-major: `n` points and `d` features
//! packed as feature 0 for all rows, then feature 1, and so on.
//! Observation noise lives in [`GaussianLikelihood`].
//! [`kernel::WhiteKernel`] is opt-in composition; using both at large
//! values double-counts noise. This crate is not published to crates.io
//! (`publish = false`).
//!
//! Distance fills and lower-triangle kernel writes use the process-wide
//! Rayon pool (shared with faer). There is no parallel on/off flag and no
//! `n_jobs` setter on [`Gpr`]. Set `RAYON_NUM_THREADS` before the process
//! starts, or call
//! `rayon::ThreadPoolBuilder::new().num_threads(n).build_global()` before
//! the first [`Gpr::fit`] / [`FittedGpr::predict`]. One worker
//! (`RAYON_NUM_THREADS=1`) is sequential. The global pool can be
//! initialized only once. Training Cholesky and the `W` n-RHS solve cap
//! faer at `min(pool, n / 64)`. Predict / covariance triangular solves
//! also cap at `n · m / 16384` and `m / 12` so a 1024×100 `L⁻¹ k_*`
//! does not start 16 workers. Kernel fills still use the full pool.
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

mod data;
mod error;
mod gpr;
#[cfg(any(feature = "bench-internals", feature = "insert-stages"))]
pub mod internals;
pub mod kernel;
mod likelihood;
mod linalg;
mod math;
mod objective;
mod online;
mod optimizer;
mod param;
pub mod persist;
mod precision;
mod rng;
mod sgpr;
mod svgp;
#[cfg(test)]
#[path = "../tests/common/check.rs"]
#[allow(dead_code)]
mod test_check;
#[cfg(test)]
#[path = "../tests/common/problems.rs"]
#[allow(dead_code)]
mod test_problems;
pub mod transform;
mod workspace;

pub use error::{CholeskyStage, GprError};
pub use gpr::{
    AdaptiveJitter, CachedDistances, CholeskyBuffer, DistanceCachePolicy, FittedGpr, FixedJitter,
    Gpr, JitterPolicy, NoDistanceCache, OnlineGpr, PointId, PredictOptions, Prediction,
    PredictiveCovariance, RetainCholesky, ReuseCholesky, UncachedDistances, VarianceKind,
};
pub use likelihood::GaussianLikelihood;
pub use math::{Accurate, FastApprox, KernelMath};
pub use objective::{Differentiable, IncrementalObjective, Objective, TwiceDifferentiable};
pub use optimizer::{
    AcceptsRecompute, Adam, BoundaryPolicy, FastSimulatedAnnealing, Fixed, FullRecompute,
    IncrementalRecompute, Lbfgs, NelderMead, Newton, NonlinearCg, OptResult, Optimizer,
    PoleRecompute, RecomputeStrategy, UsesChangeIndices,
};
pub use param::{BoundedParam, Interval, IntervalError};
pub use persist::{
    FORMAT_VERSION, LoadedDistance, LoadedGpr, LoadedOnlineDistance, PersistRegistry,
};
pub use precision::{
    DoublePrecision, GpScalar, MixedPrecision, PrecisionPolicy, PromoteStorage, ReevaluateKernel,
    ResidualFormula, SinglePrecision,
};
pub use sgpr::{FittedSgpr, FixedInducing, FreeInducing, InducingId, OnlineSgpr, Sgpr};
pub use svgp::{FittedSvgp, Svgp};

#[cfg(test)]
mod tests {
    #[test]
    fn crate_compiles() {}
}
