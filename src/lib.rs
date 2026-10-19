//! Gaussian process regression in Rust: exact GPR, sparse GPR (VFE), and SVGP.
//!
//! [`Gpr`] is the trainer. [`Gpr::fit`] consumes [`Gpr<Lbfgs>`] and returns
//! [`FittedGpr`]. [`Gpr::with_optimizer`] swaps in [`NelderMead`],
//! [`TrustRegion`], or [`FastSimulatedAnnealing`]. [`Gpr<Fixed>::factor`] factors at the current `θ` without
//! a search. Training `X` is column-major: `n` points and `d` features
//! packed as feature 0 for all rows, then feature 1, and so on.
//! Observation noise lives in [`GaussianLikelihood`].
//! [`kernel::WhiteKernel`] is opt-in composition; using both at large
//! values double-counts noise.
//!
//! `fit` and `factor` return `Result<Fitted, (Trainer, GprError)>`: a failure
//! hands the trainer back with the error, so a caller can match
//! `Err((trainer, err))` and try again with the same trainer. On [`Gpr`], `?`
//! drops the trainer (`From<(Gpr<O, P, K>, GprError)> for GprError`).
//! [`Sgpr`] and [`Svgp`] have no such conversion: write
//! `.map_err(|(_, e)| e)?`.
//!
//! Call a model's methods on the value (`fitted.predict(..)`). The models of
//! a [`kernel::KernelSpec`] and of a [`kernel::DistanceKernel`] share method
//! names (`fit`, `factor`, the `predict` / covariance / `sample` family,
//! `insert`, `insert_inducing`, and the sparse `into_online` and `save`), so
//! a call by the type's path without its parameters, such as
//! `FittedGpr::predict(&fitted, ..)`, does not compile (`E0034`): the path
//! does not say which of the same-named methods it means. Name the
//! parameters the value has: `<FittedGpr>::predict(&fitted, ..)` for a
//! [`FittedGpr`] with the default parameters, `<FittedGpr<Fixed>>::predict(&frozen, ..)`
//! for one from `factor`.
//!
//! Depend on crates.io with `gprx = "0.2"`. The 0.2.0 contract (tag
//! `v0.2.0`) is the default-feature public API: [`Gpr`], [`Sgpr`], and
//! [`Svgp`], online updates, supplied squared distances
//! ([`kernel::ScalarDistance`], [`kernel::ArdDistance`]), and directory
//! save/load. A 0.x minor may break that API; the
//! [changelog](https://github.com/YUKIKEDA/gprx/blob/main/CHANGELOG.md)
//! lists what 0.2.0 changed from 0.1.0. The MSRV is 1.88. The `internals`
//! module (`bench-internals`, `insert-stages`) is outside semantic
//! versioning.
//!
//! Distance fills and lower-triangle kernel writes run on the process-wide Rayon thread
//! pool, which the linear-algebra backend shares. There is no parallel on/off flag and no
//! `n_jobs` setter on [`Gpr`]. Set `RAYON_NUM_THREADS` before the process starts, or call
//! `rayon::ThreadPoolBuilder::new().num_threads(n).build_global()` before the first
//! [`Gpr::fit`] / [`FittedGpr::predict`]. One worker (`RAYON_NUM_THREADS=1`) is
//! sequential. The global pool can be initialized only once. A factor or solve of `n` rows
//! and `k` columns (the training Cholesky factor and the `n`-column solve that forms the
//! gradient matrix `ααᵀ − A⁻¹` have `k = n`; a predict or covariance solve has one column per
//! query point) runs the linear-algebra backend on `min(pool, n / 64, n · k / 16384, k / 12)`
//! workers, at least one, so a 1024 × 100 solve does not start 16 workers. Kernel fills
//! still use the full pool.
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
//! // The same call by path, with the type's default parameters.
//! assert_eq!(<gprx::FittedGpr>::predict(&fitted, &[0.5], 1, 1)?, pred);
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
mod optimizer;
mod param;
pub mod persist;
mod points;
mod policy;
mod precision;
mod prediction;
mod rng;
mod sgpr;
mod sparse;
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

pub use error::{CholeskyStage, GprError, PersistErrorKind, SlotErrorKind};
pub use gpr::{FittedGpr, Gpr, OnlineGpr};
pub use likelihood::GaussianLikelihood;
pub use math::{Accurate, FastApprox, KernelMath};
pub use objective::{Differentiable, IncrementalObjective, Objective, TwiceDifferentiable};
pub use optimizer::{
    Adam, BoundaryPolicy, FastSimulatedAnnealing, Fixed, Lbfgs, NelderMead, OptResult, Optimizer,
    TrustRegion,
};
pub use param::{BoundedParam, Interval, IntervalError};
pub use persist::{FORMAT_VERSION, LoadedGpr, LoadedSgpr, LoadedSvgp, PersistRegistry};
pub use points::PointId;
pub use policy::{
    AdaptiveJitter, CholeskyBuffer, DistanceCachePolicy, FixedJitter, JitterPolicy, KernelExp,
};
pub use precision::{
    DoublePrecision, GpScalar, MixedPrecision, PrecisionPolicy, PromoteStorage, ReevaluateKernel,
    ResidualFormula, SinglePrecision,
};
pub use prediction::{PredictOptions, Prediction, PredictiveCovariance, VarianceKind};
pub use sgpr::{FittedSgpr, FixedInducing, FreeInducing, InducingId, OnlineSgpr, Sgpr};
pub use svgp::{FittedSvgp, Svgp};

#[cfg(test)]
mod tests {
    #[test]
    fn crate_compiles() {}
}

/// The code blocks of `README.md`, run as doctests so the crates.io page
/// stays true to the API.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;

/// The code blocks of `README.ja.md`, run as doctests (the same blocks as
/// `README.md`).
#[cfg(doctest)]
#[doc = include_str!("../README.ja.md")]
pub struct ReadmeJaDoctests;

/// The code blocks of `CHANGELOG.md`, run as doctests so each migration
/// builds.
#[cfg(doctest)]
#[doc = include_str!("../CHANGELOG.md")]
pub struct ChangelogDoctests;
