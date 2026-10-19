//! Adapts unconstrained solvers to the [`Optimizer`] type slot.
//!
//! Does not implement L-BFGS, Nelder–Mead, or a trust-region method. [`Lbfgs`],
//! [`NelderMead`], and [`TrustRegion`] map user-unit [`crate::Interval`] through a logit so
//! the solver stays unconstrained. Positive intervals (`lo > 0`) use a log-uniform logit,
//! matching restart sampling. The unconstrained coordinate is scaled so the Jacobian is 1
//! at the interval midpoint. When a gradient solver asks for cost and gradient at the same
//! point, one [`crate::Differentiable::value_and_gradient_into`] call fills both.
//! [`TrustRegion`] also maps the analytic Hessian to logit coordinates. [`NelderMead`]
//! evaluates [`crate::Objective::value`] only. [`Adam`] is a mini-batch loop for
//! [`crate::Svgp`] and does not implement [`Optimizer`]. [`FastSimulatedAnnealing`] is a
//! homemade value-only solver that walks log-`θ` with Cauchy / Metropolis steps instead of
//! a logit map. The first evaluation and each restart use [`crate::Objective::value`]; each
//! coordinate step uses [`crate::Objective::value_at_changes`].

mod adam;
mod adapter;
mod fsa;
mod lbfgs;
mod logit;
mod neldermead;
mod trust_region;

use std::num::NonZeroU32;

pub use adam::Adam;
pub use fsa::{BoundaryPolicy, FastSimulatedAnnealing};
pub use lbfgs::Lbfgs;
pub(crate) use logit::{chain_logit_grad, log_theta_to_z, z_to_log_theta_into};
pub use neldermead::NelderMead;
pub use trust_region::TrustRegion;

use crate::error::GprError;

/// Represents the result of [`Optimizer::minimize`].
///
/// See the example on [`Optimizer`].
#[derive(Clone, Debug)]
pub struct OptResult {
    /// Holds the parameters in the same space as `init` (log-`θ` for [`crate::Gpr`]).
    pub params: Vec<f64>,
    /// Holds the objective value at [`Self::params`].
    pub value: f64,
    /// Holds the optimizer iterations performed.
    pub iterations: u64,
}

/// Represents the hyperparameter optimizer.
///
/// `P` is the objective this algorithm can minimize. [`Lbfgs`]
/// requires [`crate::Differentiable`], [`TrustRegion`]
/// [`crate::TwiceDifferentiable`], and [`NelderMead`] and
/// [`FastSimulatedAnnealing`] only [`crate::Objective`]. Every one searches
/// inside [`crate::Objective::fill_intervals`].
///
/// A trainer holds one optimizer as its type parameter, and
/// `with_optimizer` replaces it:
///
/// | Type | Search | Setters |
/// | --- | --- | --- |
/// | [`Lbfgs`] | L-BFGS, Moré–Thuente line search. Needs [`crate::Differentiable`] | `new`: 100 iterations, tolerance `sqrt(ε)`, history 10. `with_max_iterations`, `with_tolerance`, `with_history_size`, `with_restarts(n, seed)` |
/// | [`NelderMead`] | derivative-free. Needs [`crate::Objective`] | `with_max_iterations`, `with_tolerance`, `with_restarts` |
/// | [`TrustRegion`] | uses the Hessian. Needs [`crate::TwiceDifferentiable`] | `with_max_iterations`, `with_tolerance`, `with_restarts`, `with_radii(initial, max)` |
/// | [`FastSimulatedAnnealing`] | Cauchy proposals, Metropolis, Ingber cooling. Needs [`crate::Objective`] | `with_max_iterations`, `with_restarts`, `with_initial_temperature`, `with_cooling_rate`, `with_seed`, `with_boundary` |
/// | [`Fixed`] | no search: `factor` only. Not an `Optimizer` | unit struct |
/// | [`Adam`] | mini-batch, [`crate::Svgp`] only. Not an `Optimizer` | `with_learning_rate`, `with_beta1`, `with_beta2`, `with_epsilon`, `with_batch_size`, `with_epochs`, `with_seed` |
///
/// `with_restarts(n, seed)` adds `n` extra starts, drawn log-uniform in each
/// interval with `lo > 0` and uniform in any other, and keeps the
/// lowest value; the first start is the model's `θ`.
///
/// A user optimizer implements this generically over the capability it
/// needs. It can read the intervals and call a built-in optimizer on the
/// same objective.
///
/// # Examples
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
///     // Neither stage reports changed coordinates, so the fit rebuilds the
///     // whole kernel at each step (the default, written out here).
///     const USES_CHANGE_INDICES: bool = false;
///
///     fn minimize(&self, objective: &mut P, init: &[f64]) -> Result<OptResult, GprError> {
///         let mut intervals = vec![Interval::DEFAULT_POSITIVE; objective.num_params()];
///         objective.fill_intervals(&mut intervals)?;
///         let coarse = NelderMead::new()
///             .with_max_iterations(20)
///             .minimize(objective, init)?;
///         let fine = Lbfgs::new().minimize(objective, &coarse.params)?;
///         // Keep the better point; count the iterations of both stages.
///         let best = if fine.value <= coarse.value { fine.params } else { coarse.params };
///         Ok(OptResult {
///             params: best,
///             value: fine.value.min(coarse.value),
///             iterations: coarse.iterations + fine.iterations,
///         })
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
    /// An implementation chooses its errors. The built-in optimizers return
    /// [`GprError::LengthMismatch`] when `init` is not
    /// [`crate::Objective::num_params`] long (and [`FastSimulatedAnnealing`]
    /// when there are no parameters), the objective's own error when
    /// [`crate::Objective::fill_intervals`] fails, when the objective fails
    /// at the initial point, or when the best point a run ends on is one
    /// where it failed (a run treats a failed point as a barrier and goes on;
    /// [`FastSimulatedAnnealing`] skips proposals whose factorization fails),
    /// [`GprError::InvalidConfig`] when the solver rejects its settings or
    /// (for [`FastSimulatedAnnealing`]) the first value is not finite, and
    /// [`GprError::OptimizationNotConverged`] in the cases that variant
    /// lists. A run from an extra start (`with_restarts`) that fails is
    /// dropped, not returned.
    ///
    /// See the example on [`Optimizer`].
    fn minimize(&self, objective: &mut P, init: &[f64]) -> Result<OptResult, GprError>;

    /// Declares whether this optimizer reports changed coordinates through [`crate::Objective::value_at_changes`].
    ///
    /// When `true` and the trainer keeps a dedicated gradient buffer
    /// ([`crate::CholeskyBuffer::Retain`]), a fit rebuilds only the kernel
    /// leaves a step touches. The default is `false`;
    /// [`FastSimulatedAnnealing`] sets it.
    ///
    /// See the example on [`Optimizer`].
    const USES_CHANGE_INDICES: bool = false;
}

/// Represents no hyperparameter search: the model keeps the `θ` it was built with.
///
/// It is not an [`Optimizer`]. A trainer whose optimizer slot is `Fixed`
/// factors instead of fitting: [`crate::Gpr<Fixed>::factor`],
/// [`crate::Sgpr<Fixed>::factor`], and [`crate::Svgp<Fixed>::factor`]
/// (`Fixed` is the default slot of [`crate::Svgp`]). On the exact models,
/// [`crate::FittedGpr`] and [`crate::OnlineGpr`] with this slot refactor at
/// their current `θ` on `refit`; the sparse models have no `refit` for
/// `Fixed`. A loaded model carries this slot.
///
/// See the example on [`crate::Gpr::factor`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Fixed;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Restarts {
    n: NonZeroU32,
    seed: u64,
}

/// Checks that `init` has the objective's `n` parameters.
fn require_params(init: &[f64], n: usize) -> Result<(), GprError> {
    if init.len() == n {
        Ok(())
    } else {
        Err(GprError::LengthMismatch {
            reason: format!("expected {n} parameters, got {}", init.len()),
        })
    }
}

/// The restart loop every built-in optimizer shares.
///
/// Checks `init`'s length, reads the intervals, runs once from
/// `first_start(init)`, then once from each start `sample` draws with the
/// restart seed. `run` receives `None` for the first run and `Some(k)` for
/// restart `k` (1-based), and adds its result to the best so far. The first
/// run's error is returned; a restart's error only drops that restart. Fails
/// when no run produced a result.
fn minimize_with_restarts<P: crate::objective::Objective + ?Sized>(
    objective: &mut P,
    init: &[f64],
    restarts: Option<Restarts>,
    first_start: impl FnOnce(&[f64], &[crate::param::Interval]) -> Result<Vec<f64>, GprError>,
    mut sample: impl FnMut(
        &[crate::param::Interval],
        &mut crate::rng::SeededRng,
    ) -> Result<Vec<f64>, GprError>,
    mut run: impl FnMut(
        &mut P,
        &[crate::param::Interval],
        &[f64],
        Option<u64>,
        &mut Option<OptResult>,
    ) -> Result<(), GprError>,
) -> Result<OptResult, GprError> {
    let n = objective.num_params();
    require_params(init, n)?;
    let mut intervals = vec![crate::param::Interval::DEFAULT_POSITIVE; n];
    objective.fill_intervals(&mut intervals)?;
    let mut best: Option<OptResult> = None;
    let start = first_start(init, &intervals)?;
    run(objective, &intervals, &start, None, &mut best)?;
    if let Some(restarts) = restarts {
        let mut rng = crate::rng::seeded_rng(restarts.seed);
        for restart in 1..=u64::from(restarts.n.get()) {
            let start = sample(&intervals, &mut rng)?;
            let _ = run(objective, &intervals, &start, Some(restart), &mut best);
        }
    }
    best.ok_or(GprError::OptimizationNotConverged { iterations: 0 })
}
