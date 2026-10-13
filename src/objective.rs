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

/// Represents the optimizer-facing scalar objective (`value` only).
///
/// Default [`Differentiable::value_and_gradient_into`] is not on this trait.
/// A solver that needs derivatives takes [`Differentiable`], not a runtime
/// flag.
///
/// The built-in optimizers read each parameter on a positive interval as
/// `log` of a user-unit value inside that [`Interval`] (log-`θ` for the model
/// fits), and any other parameter as the user-unit value itself.
///
/// # Examples
///
/// The smallest objective:
///
/// ```rust
/// use gprx::Objective;
///
/// struct Zero;
///
/// impl Objective for Zero {
///     fn num_params(&self) -> usize {
///         0
///     }
///
///     fn value(&mut self, _params: &[f64]) -> Result<f64, gprx::GprError> {
///         Ok(0.0)
///     }
/// }
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let mut objective = Zero;
/// assert_eq!(objective.value(&[])?, 0.0);
/// # Ok(())
/// # }
/// ```
///
/// A bowl `f(θ) = Σ (θ_i − c_i)²` with every capability, minimized by each
/// built-in optimizer:
///
/// ```rust
/// use gprx::{
///     Differentiable, FastSimulatedAnnealing, GprError, IncrementalObjective, Interval, Lbfgs,
///     NelderMead, Objective, Optimizer, TrustRegion, TwiceDifferentiable,
/// };
///
/// struct Bowl {
///     center: Vec<f64>,
/// }
///
/// impl Bowl {
///     fn check(&self, params: &[f64]) -> Result<(), GprError> {
///         if params.len() == self.center.len() {
///             Ok(())
///         } else {
///             Err(GprError::LengthMismatch {
///                 reason: format!("expected {} parameters", self.center.len()),
///             })
///         }
///     }
/// }
///
/// impl Objective for Bowl {
///     fn num_params(&self) -> usize {
///         self.center.len()
///     }
///
///     fn value(&mut self, params: &[f64]) -> Result<f64, GprError> {
///         self.check(params)?;
///         Ok(params.iter().zip(&self.center).map(|(p, c)| (p - c).powi(2)).sum())
///     }
///
///     // The default `fill_intervals` writes `Interval::DEFAULT_POSITIVE`, and
///     // the default `value_at_changes` calls `value`.
/// }
///
/// impl Differentiable for Bowl {
///     fn gradient_into(&mut self, params: &[f64], out: &mut [f64]) -> Result<(), GprError> {
///         self.check(params)?;
///         for ((g, p), c) in out.iter_mut().zip(params).zip(&self.center) {
///             *g = 2.0 * (p - c);
///         }
///         Ok(())
///     }
/// }
///
/// impl TwiceDifferentiable for Bowl {
///     fn hessian_into(&mut self, params: &[f64], out: &mut [f64]) -> Result<(), GprError> {
///         self.check(params)?;
///         let n = params.len();
///         for (k, h) in out.iter_mut().enumerate() {
///             *h = if k / n == k % n { 2.0 } else { 0.0 };
///         }
///         Ok(())
///     }
/// }
///
/// impl IncrementalObjective for Bowl {
///     // Every term depends on one coordinate, so nothing is cached to skip.
///     fn value_with_changes(&mut self, params: &[f64], _indices: &[usize]) -> Result<f64, GprError> {
///         self.value(params)
///     }
/// }
///
/// # fn main() -> Result<(), GprError> {
/// // The minimum is at user values 2 and 0.5, so at their logs.
/// let center = vec![2.0_f64.ln(), 0.5_f64.ln()];
/// let mut bowl = Bowl { center: center.clone() };
/// let init = [0.0, 0.0];
///
/// let mut intervals = [Interval::DEFAULT_POSITIVE; 2];
/// bowl.fill_intervals(&mut intervals)?;
/// assert_eq!(intervals, [Interval::DEFAULT_POSITIVE; 2]);
/// let start = bowl.value(&init)?;
/// assert!((bowl.value_at_changes(&[center[0], 0.0], &[0])? - center[1].powi(2)).abs() < 1e-12);
/// assert!((bowl.value_with_changes(&init, &[0])? - start).abs() < 1e-12);
///
/// let close = |params: &[f64]| params.iter().zip(&center).all(|(p, c)| (p - c).abs() < 1e-3);
/// assert!(close(&NelderMead::new().minimize(&mut bowl, &init)?.params));
/// assert!(close(&Lbfgs::new().minimize(&mut bowl, &init)?.params));
/// assert!(close(&TrustRegion::new().minimize(&mut bowl, &init)?.params));
/// // Annealing is a random search: it keeps the best point it visited.
/// let annealed = FastSimulatedAnnealing::new().minimize(&mut bowl, &init)?;
/// assert!(annealed.value <= start);
/// # Ok(())
/// # }
/// ```
pub trait Objective {
    /// Returns the concatenated parameter count.
    ///
    /// See the example on [`Objective`].
    fn num_params(&self) -> usize;

    /// Returns the objective at `params`.
    ///
    /// # Errors
    ///
    /// An implementation chooses its errors. The exact GPR fit objective
    /// returns [`GprError::LengthMismatch`] when `params` is not
    /// [`Self::num_params`] long, [`GprError::InvalidHyperparameter`] or
    /// [`GprError::InvalidNoiseVariance`] when a coordinate is not finite or
    /// its `exp` overflows or underflows, [`GprError::InvalidInterval`] when a
    /// value leaves its [`Interval`], and [`GprError::CholeskyFailed`] when
    /// `A = K + σn² I` does not factor at `params`.
    ///
    /// See the example on [`Objective`].
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
    /// Same as [`Self::value`]. The exact GPR fit objective, when it
    /// rebuilds only the touched leaves, also returns
    /// [`GprError::IndexOutOfRange`] for an empty, duplicate, or out-of-range
    /// `indices`, and for a step that changed a coordinate `indices` does not
    /// list.
    ///
    /// See the example on [`Objective`].
    fn value_at_changes(&mut self, params: &[f64], indices: &[usize]) -> Result<f64, GprError> {
        let _ = indices;
        self.value(params)
    }

    /// Writes the [`Interval`] of each parameter, in user units, into `out` (length [`Self::num_params`]).
    ///
    /// The built-in optimizers search inside these intervals, so a user
    /// optimizer can read them too and call a built-in one on the same
    /// objective. The GPR fit objectives write the intervals the kernel and
    /// likelihood declare. The default writes [`Interval::DEFAULT_POSITIVE`]
    /// for every parameter.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::LengthMismatch`] when `out.len()` is not
    /// [`Self::num_params`].
    ///
    /// See the example on [`Objective`].
    fn fill_intervals(&self, out: &mut [Interval]) -> Result<(), GprError> {
        crate::data::require_count(out.len(), self.num_params(), "parameters")?;
        out.fill(Interval::DEFAULT_POSITIVE);
        Ok(())
    }
}

/// Represents the first-order objective.
///
/// [`Objective`] is its supertrait: an implementation also implements
/// [`Objective`].
///
/// The GPR fit objective overrides [`Self::value_and_gradient_into`] so one
/// Cholesky produces `L`, `α`, and `W`.
///
/// # Examples
///
/// ```rust
/// use gprx::{Differentiable, Objective};
///
/// struct Zero;
///
/// impl Objective for Zero {
///     fn num_params(&self) -> usize {
///         1
///     }
///
///     fn value(&mut self, _params: &[f64]) -> Result<f64, gprx::GprError> {
///         Ok(0.0)
///     }
/// }
///
/// impl Differentiable for Zero {
///     fn gradient_into(
///         &mut self,
///         _params: &[f64],
///         out: &mut [f64],
///     ) -> Result<(), gprx::GprError> {
///         out.fill(0.0);
///         Ok(())
///     }
/// }
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let mut objective = Zero;
/// let mut grad = [1.0];
/// objective.gradient_into(&[0.0], &mut grad)?;
/// assert_eq!(grad, [0.0]);
/// // The default joint call runs `value`, then `gradient_into`.
/// let value = objective.value_and_gradient_into(&[0.0], &mut grad)?;
/// assert_eq!((value, grad), (0.0, [0.0]));
/// # Ok(())
/// # }
/// ```
pub trait Differentiable: Objective {
    /// Writes `∂L/∂θ` into `out`.
    ///
    /// # Errors
    ///
    /// Same as [`Objective::value`]. The exact GPR fit objective also
    /// returns [`GprError::LengthMismatch`] when `out` is not
    /// [`Objective::num_params`] long.
    ///
    /// See the example on [`Differentiable`].
    fn gradient_into(&mut self, params: &[f64], out: &mut [f64]) -> Result<(), GprError>;

    /// Returns the value and writes the gradient in one evaluation.
    ///
    /// The default calls [`Objective::value`] then [`Self::gradient_into`].
    ///
    /// # Errors
    ///
    /// Same as [`Self::gradient_into`].
    ///
    /// See the example on [`Differentiable`].
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

/// Represents the second-order objective.
///
/// [`Differentiable`] is its supertrait: an implementation also implements
/// [`Differentiable`] and [`Objective`].
///
/// The exact GPR fit objective implements it with the analytic Hessian that
/// [`crate::FittedGpr::hessian_into`] also returns, so [`crate::TrustRegion`]
/// takes it at compile time; there is no runtime "not implemented" error.
///
/// The second example on [`Objective`] minimizes an implementation with
/// [`crate::TrustRegion`].
///
/// # Examples
///
/// ```rust
/// use gprx::{Differentiable, Objective, TwiceDifferentiable};
///
/// struct Zero;
///
/// impl Objective for Zero {
///     fn num_params(&self) -> usize {
///         1
///     }
///
///     fn value(&mut self, _params: &[f64]) -> Result<f64, gprx::GprError> {
///         Ok(0.0)
///     }
/// }
///
/// impl Differentiable for Zero {
///     fn gradient_into(
///         &mut self,
///         _params: &[f64],
///         out: &mut [f64],
///     ) -> Result<(), gprx::GprError> {
///         out.fill(0.0);
///         Ok(())
///     }
/// }
///
/// impl TwiceDifferentiable for Zero {
///     fn hessian_into(
///         &mut self,
///         _params: &[f64],
///         out: &mut [f64],
///     ) -> Result<(), gprx::GprError> {
///         out.fill(0.0);
///         Ok(())
///     }
/// }
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let mut objective = Zero;
/// let mut hess = [1.0];
/// objective.hessian_into(&[0.0], &mut hess)?;
/// assert_eq!(hess, [0.0]);
/// // The default joint call runs the value, gradient, and Hessian in turn.
/// let mut grad = [1.0];
/// let value = objective.value_gradient_hessian_into(&[0.0], &mut grad, &mut hess)?;
/// assert_eq!((value, grad, hess), (0.0, [0.0], [0.0]));
/// # Ok(())
/// # }
/// ```
pub trait TwiceDifferentiable: Differentiable {
    /// Writes the Hessian (row-major `n×n`) into `out`.
    ///
    /// # Errors
    ///
    /// Same as [`Objective::value`]. The exact GPR fit objective also
    /// returns [`GprError::LengthMismatch`] when `out` is not `n²` long.
    ///
    /// See the example on [`TwiceDifferentiable`].
    fn hessian_into(&mut self, params: &[f64], out: &mut [f64]) -> Result<(), GprError>;

    /// Returns the value at `params` and writes the gradient into `grad` and the Hessian (row-major `n×n`) into `hess`.
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
    ///
    /// See the example on [`TwiceDifferentiable`].
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

/// Represents the partial kernel rebuild from changed parameter indices.
///
/// `indices` is every flat `θ` position that changed since the previous
/// evaluation on this objective (see [`Objective::value_at_changes`]). Only
/// those positions decide which leaves are rebuilt; changes are never
/// inferred from the numbers. Empty, duplicate, or out-of-range indices, and
/// an unlisted change, are a [`GprError`] at this boundary. Full rebuilds use
/// [`Objective::value`]. The GPR fit objective implements this for every
/// optimizer and buffer policy.
///
/// # Examples
///
/// ```rust
/// use gprx::{IncrementalObjective, Objective};
///
/// struct Zero;
///
/// impl Objective for Zero {
///     fn num_params(&self) -> usize {
///         1
///     }
///
///     fn value(&mut self, _params: &[f64]) -> Result<f64, gprx::GprError> {
///         Ok(0.0)
///     }
/// }
///
/// impl IncrementalObjective for Zero {
///     fn value_with_changes(
///         &mut self,
///         _params: &[f64],
///         _indices: &[usize],
///     ) -> Result<f64, gprx::GprError> {
///         Ok(0.0)
///     }
/// }
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let mut objective = Zero;
/// assert_eq!(objective.value_with_changes(&[0.0], &[0])?, 0.0);
/// # Ok(())
/// # }
/// ```
pub trait IncrementalObjective: Objective {
    /// Returns the objective after rebuilding only the leaves that `indices` touch.
    ///
    /// # Errors
    ///
    /// Same as [`Objective::value`]. The exact GPR fit objective also
    /// returns [`GprError::IndexOutOfRange`] when `indices` is empty,
    /// contains a duplicate or an `i >= n_params`, or leaves out a coordinate
    /// that changed since the previous evaluation.
    ///
    /// See the example on [`IncrementalObjective`].
    fn value_with_changes(&mut self, params: &[f64], indices: &[usize]) -> Result<f64, GprError>;
}

#[cfg(test)]
mod interval_tests {
    use super::Objective;
    use crate::error::GprError;
    use crate::param::Interval;

    struct Plain;

    impl Objective for Plain {
        fn num_params(&self) -> usize {
            2
        }

        fn value(&mut self, params: &[f64]) -> Result<f64, GprError> {
            Ok(params.iter().map(|p| p * p).sum())
        }
    }

    #[test]
    fn default_intervals_are_the_default_positive_range() {
        let mut out = [Interval::new(1.0, 2.0).expect("valid"); 2];
        Plain.fill_intervals(&mut out).expect("len 2");
        assert_eq!(out, [Interval::DEFAULT_POSITIVE; 2]);
        let mut short = [Interval::DEFAULT_POSITIVE; 1];
        assert!(matches!(
            Plain.fill_intervals(&mut short),
            Err(GprError::LengthMismatch { .. })
        ));
    }
}
