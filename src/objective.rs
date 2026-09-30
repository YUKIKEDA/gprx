//! Negative log marginal likelihood as an optimizer objective.
//!
//! The GPR fit objective borrows [`crate::FittedGpr`] during `fit` / `refit` and forwards
//! concatenated kernel-then-likelihood `θ` to the model, which owns the
//! source of truth. [`SgprObjective`] does the same for
//! [`crate::FittedSgpr`] and the VFE evidence lower bound. Capability
//! is split so a derivative-free solver can require only [`Objective`],
//! L-BFGS can require [`Differentiable`], and a Newton solver can require
//! [`TwiceDifferentiable`].

use crate::error::GprError;
use crate::gpr::{ExactFit, LeafCache};
use crate::param::Interval;
use crate::sgpr::{FittedSgpr, InducingLayout};

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

/// Exact GPR objective. Parameters are kernel `θ` followed by likelihood `θ`.
///
/// Does not own hyperparameters. After a successful evaluation, the fitted
/// model's kernel and likelihood match `params`.
pub struct GprObjective<'a, P: crate::precision::GpScalar = crate::precision::DoublePrecision> {
    model: ExactFit<'a, P>,
    scratch: Vec<f64>,
    leaves: LeafCache<P::Storage>,
    /// Rebuild only dirty leaves in [`Objective::value`] /
    /// [`Objective::value_at_changes`]. Set when the optimizer reports changed
    /// coordinates and the fit keeps a dedicated `W`.
    incremental: bool,
}

impl<'a, P> GprObjective<'a, P>
where
    P: crate::precision::GpScalar,
{
    pub(crate) fn new(model: ExactFit<'a, P>) -> Self {
        let scratch = vec![0.0; model.num_params()];
        Self {
            model,
            scratch,
            leaves: LeafCache::new(),
            incremental: false,
        }
    }

    /// Enables leaf-level rebuilds for an optimizer with
    /// [`crate::Optimizer::USES_CHANGE_INDICES`], unless a gradient
    /// overwrites `L`.
    pub(crate) fn with_change_indices(mut self, uses_change_indices: bool) -> Self {
        self.incremental = uses_change_indices && !self.model.overwrites_cholesky();
        self
    }

    #[cfg(test)]
    pub(crate) fn is_incremental(&self) -> bool {
        self.incremental
    }

    fn full_value(&mut self, params: &[f64]) -> Result<f64, GprError> {
        let n = self.model.num_params();
        if self.scratch.len() != n {
            self.scratch.resize(n, 0.0);
        }
        self.model
            .value_and_gradient_into_fit(params, &mut self.scratch)
    }

    fn leaf_value(&mut self, params: &[f64], indices: Option<&[usize]>) -> Result<f64, GprError> {
        self.model
            .value_from_leaf_grams(params, indices, &mut self.leaves)
    }
}

impl<P> Objective for GprObjective<'_, P>
where
    P: crate::precision::GpScalar,
{
    fn num_params(&self) -> usize {
        self.model.num_params()
    }

    fn value(&mut self, params: &[f64]) -> Result<f64, GprError> {
        if self.incremental {
            self.leaf_value(params, None)
        } else {
            self.full_value(params)
        }
    }

    fn value_at_changes(&mut self, params: &[f64], indices: &[usize]) -> Result<f64, GprError> {
        if self.incremental {
            self.leaf_value(params, Some(indices))
        } else {
            self.full_value(params)
        }
    }
}

impl<P> IncrementalObjective for GprObjective<'_, P>
where
    P: crate::precision::GpScalar,
{
    fn value_with_changes(&mut self, params: &[f64], indices: &[usize]) -> Result<f64, GprError> {
        self.leaf_value(params, Some(indices))
    }
}

impl<P> Differentiable for GprObjective<'_, P>
where
    P: crate::precision::GpScalar,
{
    fn gradient_into(&mut self, params: &[f64], out: &mut [f64]) -> Result<(), GprError> {
        self.model
            .value_and_gradient_into_fit(params, out)
            .map(|_| ())
    }

    fn value_and_gradient_into(
        &mut self,
        params: &[f64],
        out: &mut [f64],
    ) -> Result<f64, GprError> {
        self.model.value_and_gradient_into_fit(params, out)
    }
}

impl<P> TwiceDifferentiable for GprObjective<'_, P>
where
    P: crate::precision::GpScalar,
{
    fn hessian_into(&mut self, params: &[f64], out: &mut [f64]) -> Result<(), GprError> {
        self.model.hessian_into_fit(params, out)
    }
}

impl<P> HasBounds for GprObjective<'_, P>
where
    P: crate::precision::GpScalar,
{
    fn fill_intervals(&self, out: &mut [Interval]) -> Result<(), GprError> {
        self.model.fill_intervals(out)
    }
}

/// Sparse VFE objective. Parameters are kernel `θ` followed by likelihood `θ`.
///
/// Does not own hyperparameters. After a successful evaluation,
/// [`FittedSgpr`]'s kernel and likelihood match `params`. [`FreeInducing`]
/// also treats column-major `Z` as parameters.
pub struct SgprObjective<
    'a,
    O,
    I = crate::FixedInducing,
    P: crate::precision::GpScalar = crate::precision::DoublePrecision,
> {
    model: &'a mut FittedSgpr<O, I, P>,
}

impl<'a, O, I, P> SgprObjective<'a, O, I, P>
where
    P: crate::precision::GpScalar,
{
    pub(crate) fn new(model: &'a mut FittedSgpr<O, I, P>) -> Self {
        Self { model }
    }
}

impl<O, I: InducingLayout, P> Objective for SgprObjective<'_, O, I, P>
where
    P: crate::precision::GpScalar,
{
    fn num_params(&self) -> usize {
        self.model.num_params()
    }

    fn value(&mut self, params: &[f64]) -> Result<f64, GprError> {
        match self.model.set_params(params) {
            Ok(()) => self.model.neg_log_marginal_likelihood(),
            Err(GprError::CholeskyFailed { .. }) => Ok(1.0e300),
            Err(err) => Err(err),
        }
    }
}

impl<O, I: InducingLayout, P> Differentiable for SgprObjective<'_, O, I, P>
where
    P: crate::precision::GpScalar,
{
    fn gradient_into(&mut self, params: &[f64], out: &mut [f64]) -> Result<(), GprError> {
        self.model.value_and_gradient_into(params, out).map(|_| ())
    }

    fn value_and_gradient_into(
        &mut self,
        params: &[f64],
        out: &mut [f64],
    ) -> Result<f64, GprError> {
        self.model.value_and_gradient_into(params, out)
    }
}

impl<O, I: InducingLayout, P> TwiceDifferentiable for SgprObjective<'_, O, I, P>
where
    P: crate::precision::GpScalar,
{
    fn hessian_into(&mut self, params: &[f64], out: &mut [f64]) -> Result<(), GprError> {
        self.model.hessian_into(params, out)
    }
}

impl<O, I: InducingLayout, P> HasBounds for SgprObjective<'_, O, I, P>
where
    P: crate::precision::GpScalar,
{
    fn fill_intervals(&self, out: &mut [Interval]) -> Result<(), GprError> {
        self.model.fill_intervals(out)
    }
}

#[cfg(test)]
mod tests {
    use super::{Differentiable, Objective};
    use crate::error::GprError;
    use crate::gpr::{FittedGpr, Gpr};
    use crate::kernel::{KernelSpec, RbfKernel};
    use crate::likelihood::GaussianLikelihood;
    use crate::optimizer::Fixed;

    const TOL: f64 = 1e-12;

    use crate::test_check::assert_close;

    fn fitted_rbf() -> FittedGpr<Fixed> {
        let kernel = KernelSpec::from(RbfKernel::new(1.0).expect("valid"));
        let likelihood = GaussianLikelihood::new(0.1).expect("valid");
        Gpr::new(kernel, likelihood)
            .with_optimizer(Fixed)
            .factor(&[0.0, 1.0], 2, 1, &[0.5, -0.25])
            .unwrap_or_else(|(_, e)| panic!("{e}"))
    }

    struct CountingObjective {
        n: usize,
        values: usize,
        grads: usize,
        last_params: Vec<f64>,
    }

    impl Objective for CountingObjective {
        fn num_params(&self) -> usize {
            self.n
        }

        fn value(&mut self, params: &[f64]) -> Result<f64, GprError> {
            self.values += 1;
            self.last_params = params.to_vec();
            Ok(1.5)
        }
    }

    impl Differentiable for CountingObjective {
        fn gradient_into(&mut self, params: &[f64], out: &mut [f64]) -> Result<(), GprError> {
            self.grads += 1;
            self.last_params = params.to_vec();
            out.fill(0.25);
            Ok(())
        }
    }

    #[test]
    fn default_value_and_gradient_into_calls_value_then_gradient() {
        let mut obj = CountingObjective {
            n: 2,
            values: 0,
            grads: 0,
            last_params: Vec::new(),
        };
        let params = [0.5, -0.25];
        let mut grad = [0.0; 2];
        let value = obj
            .value_and_gradient_into(&params, &mut grad)
            .expect("dummy");
        assert_close(value, 1.5, TOL);
        assert_eq!(obj.values, 1);
        assert_eq!(obj.grads, 1);
        assert_eq!(obj.last_params, params);
        assert_close(grad[0], 0.25, TOL);
        assert_close(grad[1], 0.25, TOL);
    }

    #[test]
    fn concatenated_params_reach_kernel_and_likelihood() {
        let mut gpr = fitted_rbf();
        let n = gpr.num_params();
        assert_eq!(n, gpr.kernel().num_params() + gpr.likelihood().num_params());
        let ell = 1.25_f64;
        let noise = 0.16_f64;
        let params = [ell.ln(), noise.ln()];
        let nlml;
        {
            let mut obj = gpr.objective();
            assert_eq!(obj.num_params(), n);
            let mut grad = [0.0; 2];
            nlml = obj
                .value_and_gradient_into(&params, &mut grad)
                .expect("spd");
            assert!(nlml.is_finite());
            assert!(grad.iter().all(|g| g.is_finite()));
        }
        let mut got = [0.0; 2];
        gpr.get_params(&mut got).expect("len 2");
        assert_close(got[0], params[0], TOL);
        assert_close(got[1], params[1], TOL);
        let mut kernel_theta = [0.0; 1];
        gpr.kernel().get_params(&mut kernel_theta).expect("len 1");
        assert_close(kernel_theta[0], params[0], TOL);
        let mut lik_theta = [0.0; 1];
        gpr.likelihood().get_params(&mut lik_theta).expect("len 1");
        assert_close(lik_theta[0], params[1], TOL);
        assert_close(
            nlml,
            gpr.neg_log_marginal_likelihood().expect("fitted"),
            TOL,
        );
    }

    #[test]
    fn value_matches_value_and_gradient() {
        let mut gpr = fitted_rbf();
        let mut params = [0.0; 2];
        gpr.get_params(&mut params).expect("len 2");
        params[0] = 0.8_f64.ln();
        let via_value;
        let via_joint;
        {
            let mut obj = gpr.objective();
            via_value = obj.value(&params).expect("spd");
            let mut grad = [0.0; 2];
            via_joint = obj
                .value_and_gradient_into(&params, &mut grad)
                .expect("spd");
        }
        assert_close(via_value, via_joint, TOL);
    }

    #[test]
    fn rejected_params_do_not_reach_the_model() {
        let mut gpr = fitted_rbf();
        let mut before = [0.0; 2];
        gpr.get_params(&mut before).expect("len 2");
        let bad = [before[0], f64::INFINITY];
        {
            let mut obj = gpr.objective();
            let mut grad = [0.0; 2];
            assert!(matches!(
                obj.value_and_gradient_into(&bad, &mut grad),
                Err(GprError::InvalidNoiseVariance { .. })
            ));
        }
        let mut after = [0.0; 2];
        gpr.get_params(&mut after).expect("len 2");
        assert_close(after[0], before[0], TOL);
        assert_close(after[1], before[1], TOL);
    }
}
