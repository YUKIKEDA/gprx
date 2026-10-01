//! Optimizer-facing objective traits.
//!
//! Each model implements them on its own adapter (`GprObjective` in
//! `src/gpr/`, `SgprObjective` in `src/sgpr/`), which borrows the fitted
//! model during `fit` / `refit` and forwards concatenated kernel-then-
//! likelihood `θ` to it. Capability is split so a derivative-free solver
//! can require only [`Objective`], L-BFGS can require [`Differentiable`],
//! and a Hessian-based solver ([`crate::TrustRegion`]) can require [`TwiceDifferentiable`].

use crate::error::GprError;
use crate::param::Interval;

/// Calls the model adapters make, for the `compare/perf` optimizer tables.
///
/// Present only with the non-default feature `bench-internals`. One global
/// pair of counters: fits must not run concurrently while it is read.
#[cfg(feature = "bench-internals")]
pub(crate) mod call_counts {
    use std::sync::atomic::{AtomicU64, Ordering};

    static VALUE_CALLS: AtomicU64 = AtomicU64::new(0);
    static JOINT_CALLS: AtomicU64 = AtomicU64::new(0);

    pub(crate) fn count_value() {
        VALUE_CALLS.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn count_joint() {
        JOINT_CALLS.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn reset() {
        VALUE_CALLS.store(0, Ordering::Relaxed);
        JOINT_CALLS.store(0, Ordering::Relaxed);
    }

    pub(crate) fn read() -> (u64, u64) {
        (
            VALUE_CALLS.load(Ordering::Relaxed),
            JOINT_CALLS.load(Ordering::Relaxed),
        )
    }
}

#[cfg(feature = "bench-internals")]
macro_rules! count_value_call {
    () => {
        $crate::objective::call_counts::count_value()
    };
}
#[cfg(not(feature = "bench-internals"))]
macro_rules! count_value_call {
    () => {};
}
#[cfg(feature = "bench-internals")]
macro_rules! count_joint_call {
    () => {
        $crate::objective::call_counts::count_joint()
    };
}
#[cfg(not(feature = "bench-internals"))]
macro_rules! count_joint_call {
    () => {};
}
pub(crate) use {count_joint_call, count_value_call};

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

    /// Returns the value at `params` and writes the gradient into `grad` and
    /// the Hessian (row-major `n×n`) into `hess`.
    ///
    /// A second-order solver evaluates all three at each candidate. The
    /// default calls [`Differentiable::value_and_gradient_into`] then
    /// [`Self::hessian_into`]. Override it when the three share work, as
    /// the GPR fit objective does: one factorization serves all three.
    ///
    /// # Errors
    ///
    /// Same as [`Differentiable::value_and_gradient_into`] and
    /// [`Self::hessian_into`].
    fn value_gradient_hessian_into(
        &mut self,
        params: &[f64],
        grad: &mut [f64],
        hess: &mut [f64],
    ) -> Result<f64, GprError> {
        let value = self.value_and_gradient_into(params, grad)?;
        self.hessian_into(params, hess)?;
        Ok(value)
    }
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
