//! Optimizer-facing objective traits.
//!
//! Each model implements them on its own adapter (`GprObjective` in
//! `src/gpr/`, `SgprObjective` in `src/sgpr/`), which borrows the fitted
//! model during `fit` / `refit` and forwards concatenated kernel-then-
//! likelihood `θ` to it. Capability is split so a derivative-free solver
//! can require only [`Objective`], L-BFGS can require [`Differentiable`],
//! and a Newton solver can require [`TwiceDifferentiable`].

use crate::error::GprError;
use crate::param::Interval;

/// Optimizer-facing scalar objective (`value` only).
///
/// Default [`Differentiable::value_and_gradient_into`] is not on this trait.
/// A solver that needs derivatives takes [`Differentiable`], not a runtime
/// flag.
pub trait Objective {
    /// Returns the concatenated parameter count.
    fn num_params(&self) -> usize;

    /// Returns the objective at `params`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError`] when `params` is the wrong length or the model
    /// cannot evaluate at that point.
    fn value(&mut self, params: &[f64]) -> Result<f64, GprError>;

    /// Returns the objective after the coordinates in `indices` changed.
    ///
    /// `indices` lists **every** coordinate of `params` that differs from
    /// the `params` of the previous evaluation on this objective, not from
    /// the optimizer's accepted point. After a rejected proposal, the next
    /// step lists the coordinate it reverted as well.
    ///
    /// The default rebuilds everything through [`Self::value`]. The GPR fit
    /// objective rebuilds only the touched kernel leaves when the optimizer
    /// sets [`crate::Optimizer::USES_CHANGE_INDICES`] and the fit keeps a
    /// dedicated `W` ([`crate::CholeskyBuffer::Retain`]).
    ///
    /// # Errors
    ///
    /// Same as [`Self::value`]. An incremental implementation also rejects
    /// an empty, duplicate, or out-of-range `indices`, and a step that
    /// changed a coordinate `indices` does not list.
    fn value_at_changes(&mut self, params: &[f64], indices: &[usize]) -> Result<f64, GprError> {
        let _ = indices;
        self.value(params)
    }
}

/// First-order objective. Supertrait of [`Objective`].
///
/// The GPR fit objective overrides [`Self::value_and_gradient_into`] so one
/// Cholesky produces `L`, `α`, and `W`.
pub trait Differentiable: Objective {
    /// Writes `∂L/∂θ` into `out`.
    ///
    /// # Errors
    ///
    /// Same as [`Objective::value`], plus a length mismatch on `out`.
    fn gradient_into(&mut self, params: &[f64], out: &mut [f64]) -> Result<(), GprError>;

    /// Returns the value and writes the gradient in one evaluation.
    ///
    /// The default calls [`Objective::value`] then [`Self::gradient_into`].
    ///
    /// # Errors
    ///
    /// Same as [`Self::gradient_into`].
    fn value_and_gradient_into(
        &mut self,
        params: &[f64],
        out: &mut [f64],
    ) -> Result<f64, GprError> {
        let value = self.value(params)?;
        self.gradient_into(params, out)?;
        Ok(value)
    }
}

/// Second-order objective. Supertrait of [`Differentiable`].
///
/// The GPR fit objective implements this by forwarding to
/// [`crate::FittedGpr::hessian_into`]. There is no runtime `NotImplemented`.
pub trait TwiceDifferentiable: Differentiable {
    /// Writes the Hessian (row-major `n×n`) into `out`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError`] when a slice length is wrong or the model cannot
    /// evaluate at `params`.
    fn hessian_into(&mut self, params: &[f64], out: &mut [f64]) -> Result<(), GprError>;
}

/// Partial kernel rebuild from changed parameter indices.
///
/// `indices` is every flat `θ` position that changed since the previous
/// evaluation on this objective (see [`Objective::value_at_changes`]). Only
/// those positions decide which leaves are rebuilt; changes are never
/// inferred from the numbers. Empty, duplicate, or out-of-range indices, and
/// an unlisted change, are a [`GprError`] at this boundary. Full rebuilds use
/// [`Objective::value`]. The GPR fit objective implements this for every
/// optimizer and buffer policy.
pub trait IncrementalObjective: Objective {
    /// Returns the objective after rebuilding only the leaves that `indices`
    /// touch.
    ///
    /// # Errors
    ///
    /// Returns [`GprError`] when `params` is the wrong length, `indices` is
    /// empty, contains a duplicate, contains `i >= n_params`, or leaves out a
    /// coordinate that changed, or when the model cannot evaluate.
    fn value_with_changes(&mut self, params: &[f64], indices: &[usize]) -> Result<f64, GprError>;
}

/// Supplies per-parameter [`Interval`] in user units (crate-private).
pub(crate) trait HasBounds {
    fn fill_intervals(&self, out: &mut [Interval]) -> Result<(), GprError>;
}
