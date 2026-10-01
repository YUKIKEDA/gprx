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
/// requires [`crate::Differentiable`] plus bounds. [`TrustRegion`]
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

/// The restart loop every built-in optimizer shares.
///
/// Checks `init`'s length, reads the intervals, runs once from
/// `first_start(init)`, then once from each start `sample` draws with the
/// restart seed. `run` adds its result to the best so far (it may skip a run
/// that has none). The first run's error is returned; a restart's error only
/// drops that restart. Fails when no run produced a result.
fn minimize_with_restarts<P: crate::objective::Objective + crate::objective::HasBounds + ?Sized>(
    objective: &mut P,
    init: &[f64],
    restarts: Option<Restarts>,
    first_start: impl FnOnce(&[f64], &[crate::param::Interval]) -> Result<Vec<f64>, GprError>,
    mut sample: impl FnMut(
        &[crate::param::Interval],
        &mut rand::rngs::SmallRng,
    ) -> Result<Vec<f64>, GprError>,
    mut run: impl FnMut(
        &mut P,
        &[crate::param::Interval],
        &[f64],
        &mut Option<OptResult>,
    ) -> Result<(), GprError>,
) -> Result<OptResult, GprError> {
    let n = objective.num_params();
    if init.len() != n {
        return Err(GprError::LengthMismatch {
            reason: format!("expected {n} parameters, got {}", init.len()),
        });
    }
    let mut intervals = vec![crate::param::Interval::DEFAULT_POSITIVE; n];
    objective.fill_intervals(&mut intervals)?;
    let mut best: Option<OptResult> = None;
    let start = first_start(init, &intervals)?;
    run(objective, &intervals, &start, &mut best)?;
    if let Some(restarts) = restarts {
        let mut rng = crate::rng::small_rng(restarts.seed);
        for _ in 0..restarts.n.get() {
            let start = sample(&intervals, &mut rng)?;
            let _ = run(objective, &intervals, &start, &mut best);
        }
    }
    best.ok_or(GprError::OptimizationNotConverged { iterations: 0 })
}
