//! argmin optimizer adapters and the [`Optimizer`] type slot.
//!
//! Does not implement L-BFGS, nonlinear CG, Nelder–Mead, or Newton. [`Lbfgs`],
//! [`NonlinearCg`], [`NelderMead`], and [`Newton`] map user-unit [`crate::Interval`]
//! through a logit so argmin stays unconstrained. Positive intervals
//! (`lo > 0`) use a log-uniform logit, matching restart sampling. The
//! unconstrained coordinate is scaled so the Jacobian is 1 at the interval
//! midpoint. When a gradient solver asks for cost and gradient at the same
//! point, one [`crate::Differentiable::value_and_gradient_into`] call fills
//! both. [`Newton`] also maps the analytic Hessian to logit coordinates.
//! [`NelderMead`] evaluates [`crate::Objective::value`] only.
//! [`Adam`] is a mini-batch loop for [`crate::Svgp`] and does not implement
//! [`Optimizer`].
//! [`FastSimulatedAnnealing`] is a homemade value-only solver that walks
//! log-`θ` with Cauchy / Metropolis steps instead of a logit map. The first
//! evaluation and each restart use [`crate::Objective::value`]; each
//! coordinate step uses [`crate::Objective::value_at_changes`].

mod adam;
mod fsa;
mod lbfgs;
mod logit;
mod ncg;
mod neldermead;
mod newton;

use std::num::NonZeroU32;

pub use adam::Adam;
pub use fsa::{BoundaryPolicy, FastSimulatedAnnealing};
pub use lbfgs::Lbfgs;
pub(crate) use logit::{chain_logit_grad, log_theta_to_z, z_to_log_theta};
pub use ncg::NonlinearCg;
pub use neldermead::NelderMead;
pub use newton::Newton;

use crate::error::GprError;

/// Result of [`Optimizer::minimize`].
#[derive(Clone, Debug)]
pub struct OptResult {
    /// Parameters in the same space as `init` (log-`θ` for [`crate::Gpr`]).
    pub params: Vec<f64>,
    /// Objective value at [`Self::params`].
    pub value: f64,
    /// Optimizer iterations performed.
    pub iterations: u64,
}

/// Hyperparameter optimizer.
///
/// `P` is the objective this algorithm can minimize. [`Lbfgs`] and
/// [`NonlinearCg`] require [`crate::Differentiable`] plus bounds. [`Newton`]
/// requires [`crate::TwiceDifferentiable`] plus bounds. [`NelderMead`]
/// and [`FastSimulatedAnnealing`] require only [`crate::Objective`] plus bounds.
pub trait Optimizer<P: ?Sized> {
    /// Minimizes `objective` from `init` without taking ownership of `init`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError`] when `init` is the wrong length, the objective
    /// fails, or the solver stops without a best parameter vector.
    fn minimize(&self, objective: &mut P, init: &[f64]) -> Result<OptResult, GprError>;
}

/// Marker for an optimizer that consumes changed-parameter indices.
///
/// [`Lbfgs`] does not implement this. At [`crate::RetainCholesky`] (the
/// speed pole) [`crate::Gpr::with_optimizer`] / [`crate::Gpr::with_prefer_speed`]
/// select [`IncrementalRecompute`] only when `O: UsesChangeIndices`.
pub trait UsesChangeIndices {}

/// Marker for how kernel matrices are rebuilt during fit.
pub trait RecomputeStrategy:
    Copy + Clone + core::fmt::Debug + Default + Send + Sync + 'static
{
}

/// Bound that keeps [`IncrementalRecompute`] off optimizers without
/// [`UsesChangeIndices`].
///
/// [`FullRecompute`] is valid for every optimizer. [`IncrementalRecompute`]
/// is valid only when `O: `[`UsesChangeIndices`].
pub trait AcceptsRecompute<O>: RecomputeStrategy {}

/// Selects [`FullRecompute`] or [`IncrementalRecompute`] from the Cholesky pole.
///
/// [`ReuseCholesky`] (memory pole) is always [`FullRecompute`].
/// [`crate::RetainCholesky`] is [`IncrementalRecompute`] when `O`
/// implements [`UsesChangeIndices`], and [`FullRecompute`] for the
/// built-in solvers that do not. A custom [`Optimizer`] that does not
/// implement [`UsesChangeIndices`] implements this trait for
/// [`crate::RetainCholesky`] with [`FullRecompute`].
///
/// # Examples
///
/// ```rust
/// use gprx::{
///     FastSimulatedAnnealing, FullRecompute, IncrementalRecompute, Lbfgs,
///     PoleRecompute, RetainCholesky, ReuseCholesky,
/// };
///
/// fn assert_full<T: PoleRecompute<B, Strategy = FullRecompute>, B>() {}
/// fn assert_incr<T: PoleRecompute<B, Strategy = IncrementalRecompute>, B>() {}
///
/// assert_full::<Lbfgs, RetainCholesky>();
/// assert_full::<Lbfgs, ReuseCholesky>();
/// assert_incr::<FastSimulatedAnnealing, RetainCholesky>();
/// assert_full::<FastSimulatedAnnealing, ReuseCholesky>();
/// ```
pub trait PoleRecompute<B> {
    /// Strategy at pole `B`.
    type Strategy: RecomputeStrategy;
}

impl<O> PoleRecompute<crate::ReuseCholesky> for O {
    type Strategy = FullRecompute;
}

impl<O: UsesChangeIndices> PoleRecompute<crate::RetainCholesky> for O {
    type Strategy = IncrementalRecompute;
}

impl PoleRecompute<crate::RetainCholesky> for Lbfgs {
    type Strategy = FullRecompute;
}

impl PoleRecompute<crate::RetainCholesky> for NonlinearCg {
    type Strategy = FullRecompute;
}

impl PoleRecompute<crate::RetainCholesky> for NelderMead {
    type Strategy = FullRecompute;
}

impl PoleRecompute<crate::RetainCholesky> for Newton {
    type Strategy = FullRecompute;
}

impl PoleRecompute<crate::RetainCholesky> for Fixed {
    type Strategy = FullRecompute;
}

/// Always rebuild the full kernel matrix. This is the default.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FullRecompute;

impl RecomputeStrategy for FullRecompute {}

impl<O> AcceptsRecompute<O> for FullRecompute {}

/// Rebuild only compiled leaves touched by changed parameter indices.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct IncrementalRecompute;

impl RecomputeStrategy for IncrementalRecompute {}

impl<O: UsesChangeIndices> AcceptsRecompute<O> for IncrementalRecompute {}

/// Fixed hyperparameters. [`crate::Gpr<Fixed>::factor`] only; not an [`Optimizer`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Fixed;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Restarts {
    n: NonZeroU32,
    seed: u64,
}
