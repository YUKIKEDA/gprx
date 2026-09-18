//! argmin optimizer adapters and the [`Optimizer`] type slot.
//!
//! Does not implement L-BFGS, nonlinear CG, or Nelder–Mead. [`Lbfgs`],
//! [`NonlinearCg`], and [`NelderMead`] map user-unit [`crate::Interval`]
//! through a logit so argmin stays unconstrained. Positive intervals
//! (`lo > 0`) use a log-uniform logit, matching restart sampling. The
//! unconstrained coordinate is scaled so the Jacobian is 1 at the interval
//! midpoint. When a gradient solver asks for cost and gradient at the same
//! point, one [`crate::Differentiable::value_and_gradient_into`] call fills
//! both. [`NelderMead`] evaluates [`crate::Objective::value`] only.

mod lbfgs;
mod logit;
mod ncg;
mod neldermead;

use std::num::NonZeroU32;

pub use lbfgs::Lbfgs;
pub use ncg::NonlinearCg;
pub use neldermead::NelderMead;

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
/// [`NonlinearCg`] require [`crate::Differentiable`] plus bounds. [`NelderMead`]
/// requires only [`crate::Objective`] plus bounds.
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
/// [`Lbfgs`] does not implement this. [`crate::Gpr::with_recompute_strategy`]
/// to [`IncrementalRecompute`] exists only when `O: UsesChangeIndices`.
pub trait UsesChangeIndices {}

/// Marker for how kernel matrices are rebuilt during fit.
pub trait RecomputeStrategy:
    Copy + Clone + core::fmt::Debug + Default + Send + Sync + 'static
{
}

/// Bound on [`crate::Gpr::with_recompute_strategy`].
///
/// [`FullRecompute`] is valid for every optimizer. [`IncrementalRecompute`]
/// is valid only when `O: `[`UsesChangeIndices`].
pub trait AcceptsRecompute<O>: RecomputeStrategy {}

/// Always rebuild the full kernel matrix. This is the default.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FullRecompute;

impl RecomputeStrategy for FullRecompute {}

impl<O> AcceptsRecompute<O> for FullRecompute {}

/// Rebuild only leaves touched by changed indices. Body is P2B-18.
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
