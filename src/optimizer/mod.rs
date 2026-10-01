//! argmin optimizer adapters and the [`Optimizer`] type slot.
//!
//! Does not implement L-BFGS, Nelder–Mead, or a trust-region method. [`Lbfgs`],
//! [`NelderMead`], and [`TrustRegion`] map user-unit [`crate::Interval`]
//! through a logit so argmin stays unconstrained. Positive intervals
//! (`lo > 0`) use a log-uniform logit, matching restart sampling. The
//! unconstrained coordinate is scaled so the Jacobian is 1 at the interval
//! midpoint. When a gradient solver asks for cost and gradient at the same
//! point, one [`crate::Differentiable::value_and_gradient_into`] call fills
//! both. [`TrustRegion`] also maps the analytic Hessian to logit coordinates.
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
mod neldermead;
mod trust_region;

use std::num::NonZeroU32;

pub use adam::Adam;
pub use fsa::{BoundaryPolicy, FastSimulatedAnnealing};
pub use lbfgs::Lbfgs;
pub(crate) use logit::{chain_logit_grad, log_theta_to_z, z_to_log_theta};
pub use neldermead::NelderMead;
pub use trust_region::TrustRegion;

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
/// `P` is the objective this algorithm can minimize. [`Lbfgs`]
/// requires [`crate::Differentiable`], [`TrustRegion`]
/// [`crate::TwiceDifferentiable`], and [`NelderMead`] and
/// [`FastSimulatedAnnealing`] only [`crate::Objective`]. Every one searches
/// inside [`crate::Objective::fill_intervals`].
///
/// A user optimizer implements this generically over the capability it
/// needs. It can read the intervals and call a built-in optimizer on the
/// same objective:
///
/// ```rust
/// use gprx::kernel::{KernelSpec, RbfKernel};
/// use gprx::{
///     Differentiable, GaussianLikelihood, Gpr, GprError, Interval, Lbfgs, NelderMead,
///     OptResult, Optimizer,
/// };
///
/// /// A short Nelder–Mead pass, then L-BFGS from its result.
/// #[derive(Clone, Debug)]
/// struct Polish;
///
/// impl<P: Differentiable> Optimizer<P> for Polish {
///     fn minimize(&self, objective: &mut P, init: &[f64]) -> Result<OptResult, GprError> {
///         let mut intervals = vec![Interval::DEFAULT_POSITIVE; objective.num_params()];
///         objective.fill_intervals(&mut intervals)?;
///         let coarse = NelderMead::new()
///             .with_max_iterations(20)
///             .minimize(objective, init)?;
///         Lbfgs::new().minimize(objective, &coarse.params)
///     }
/// }
///
/// # fn main() -> Result<(), GprError> {
/// let fitted = Gpr::new(
///     KernelSpec::from(RbfKernel::new(1.0)?),
///     GaussianLikelihood::new(0.1)?,
/// )
/// .with_optimizer(Polish)
/// .fit(&[0.0, 1.0, 2.0, 3.0], 4, 1, &[0.0, 0.8, 0.9, 0.1])
/// .map_err(|(_, e)| e)?;
/// assert!(fitted.neg_log_marginal_likelihood()?.is_finite());
/// # Ok(())
/// # }
/// ```
pub trait Optimizer<P: ?Sized> {
    /// Minimizes `objective` from `init` without taking ownership of `init`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError`] when `init` is the wrong length, the objective
    /// fails, or the solver stops without a best parameter vector.
    fn minimize(&self, objective: &mut P, init: &[f64]) -> Result<OptResult, GprError>;

    /// Whether this optimizer reports changed coordinates through
    /// [`crate::Objective::value_at_changes`].
    ///
    /// When `true` and the trainer keeps a dedicated gradient buffer
    /// ([`crate::CholeskyBuffer::Retain`]), a fit rebuilds only the kernel
    /// leaves a step touches. The default is `false`.
    const USES_CHANGE_INDICES: bool = false;
}

/// Fixed hyperparameters. [`crate::Gpr<Fixed>::factor`] only; not an [`Optimizer`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Fixed;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Restarts {
    n: NonZeroU32,
    seed: u64,
}
