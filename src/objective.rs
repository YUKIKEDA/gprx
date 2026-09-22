//! Negative log marginal likelihood as an optimizer objective.
//!
//! [`GprObjective`] borrows [`FittedGpr`] during `fit` / `refit` and forwards
//! concatenated kernel-then-likelihood `θ` to the model, which owns the
//! source of truth. [`SgprObjective`] does the same for
//! [`crate::FittedSgpr`] and the VFE evidence lower bound. Capability
//! is split so a derivative-free solver can require only [`Objective`],
//! L-BFGS can require [`Differentiable`], and a Newton solver can require
//! [`TwiceDifferentiable`].

use crate::error::GprError;
use crate::gpr::{DistanceCacheSlot, FittedGpr};
use crate::optimizer::{FullRecompute, IncrementalRecompute};
use crate::param::Interval;
use crate::sgpr::{FittedSgpr, InducingLayout};
use faer::Mat;

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
    /// The default rebuilds everything through [`Self::value`].
    /// [`crate::IncrementalRecompute`] forwards to
    /// [`IncrementalObjective::value_with_changes`].
    ///
    /// # Errors
    ///
    /// Same as [`Self::value`]. An incremental implementation also rejects
    /// an empty, duplicate, or out-of-range `indices`.
    fn value_at_changes(&mut self, params: &[f64], indices: &[usize]) -> Result<f64, GprError> {
        let _ = indices;
        self.value(params)
    }
}

/// First-order objective. Supertrait of [`Objective`].
///
/// [`GprObjective`] overrides [`Self::value_and_gradient_into`] so one
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
/// [`GprObjective`] implements this by forwarding to
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
/// `indices` is the list of flat `θ` positions that changed. Empty, duplicate,
/// or out-of-range indices are a [`GprError`] at this boundary. Full rebuilds
/// use [`Objective::value`]. [`crate::FullRecompute`] does not implement this
/// trait. [`GprObjective`]`<`[`crate::IncrementalRecompute`]`>` does.
pub trait IncrementalObjective: Objective {
    /// Returns the objective after rebuilding only the leaves that `indices`
    /// touch.
    ///
    /// # Errors
    ///
    /// Returns [`GprError`] when `params` is the wrong length, `indices` is
    /// empty, contains a duplicate, or contains `i >= n_params`, or when the
    /// model cannot evaluate.
    fn value_with_changes(&mut self, params: &[f64], indices: &[usize]) -> Result<f64, GprError>;
}

/// Supplies per-parameter [`Interval`] in user units (crate-private).
pub(crate) trait HasBounds {
    fn fill_intervals(&self, out: &mut [Interval]) -> Result<(), GprError>;
}

/// Exact GPR objective. Parameters are kernel `θ` followed by likelihood `θ`.
///
/// Does not own hyperparameters. After a successful evaluation, [`FittedGpr`]'s
/// kernel and likelihood match `params`.
pub(crate) struct GprObjective<
    'a,
    O,
    S,
    C: crate::gpr::DistanceCacheSlot = crate::CachedDistances,
    B: crate::gpr::AllocWorkspace = crate::RetainCholesky,
> {
    model: &'a mut FittedGpr<O, S, C, B>,
    scratch: Vec<f64>,
    leaf_grams: Vec<Mat<f64>>,
    leaves_primed: bool,
}

impl<'a, O, S, C, B> GprObjective<'a, O, S, C, B>
where
    C: DistanceCacheSlot,
    B: crate::gpr::AllocWorkspace,
{
    pub(crate) fn new(model: &'a mut FittedGpr<O, S, C, B>) -> Self {
        let scratch = vec![0.0; model.num_params()];
        Self {
            model,
            scratch,
            leaf_grams: Vec::new(),
            leaves_primed: false,
        }
    }
}

pub(crate) trait EvalObjective: Sized {
    fn eval_value<O, C, B>(
        obj: &mut GprObjective<'_, O, Self, C, B>,
        params: &[f64],
    ) -> Result<f64, GprError>
    where
        C: DistanceCacheSlot,
        B: crate::gpr::AllocWorkspace;

    fn eval_at_changes<O, C, B>(
        obj: &mut GprObjective<'_, O, Self, C, B>,
        params: &[f64],
        indices: &[usize],
    ) -> Result<f64, GprError>
    where
        C: DistanceCacheSlot,
        B: crate::gpr::AllocWorkspace,
    {
        let _ = indices;
        Self::eval_value(obj, params)
    }
}

impl EvalObjective for FullRecompute {
    fn eval_value<O, C, B>(
        obj: &mut GprObjective<'_, O, Self, C, B>,
        params: &[f64],
    ) -> Result<f64, GprError>
    where
        C: DistanceCacheSlot,
        B: crate::gpr::AllocWorkspace,
    {
        let n = obj.model.num_params();
        if obj.scratch.len() != n {
            obj.scratch.resize(n, 0.0);
        }
        obj.model
            .value_and_gradient_into_fit(params, &mut obj.scratch)
    }
}

impl EvalObjective for IncrementalRecompute {
    fn eval_value<O, C, B>(
        obj: &mut GprObjective<'_, O, Self, C, B>,
        params: &[f64],
    ) -> Result<f64, GprError>
    where
        C: DistanceCacheSlot,
        B: crate::gpr::AllocWorkspace,
    {
        obj.model
            .value_from_leaf_grams(params, None, &mut obj.leaf_grams, &mut obj.leaves_primed)
    }

    fn eval_at_changes<O, C, B>(
        obj: &mut GprObjective<'_, O, Self, C, B>,
        params: &[f64],
        indices: &[usize],
    ) -> Result<f64, GprError>
    where
        C: DistanceCacheSlot,
        B: crate::gpr::AllocWorkspace,
    {
        obj.model.value_from_leaf_grams(
            params,
            Some(indices),
            &mut obj.leaf_grams,
            &mut obj.leaves_primed,
        )
    }
}

impl<O, S, C, B> Objective for GprObjective<'_, O, S, C, B>
where
    S: EvalObjective,
    C: DistanceCacheSlot,
    B: crate::gpr::AllocWorkspace,
{
    fn num_params(&self) -> usize {
        self.model.num_params()
    }

    fn value(&mut self, params: &[f64]) -> Result<f64, GprError> {
        S::eval_value(self, params)
    }

    fn value_at_changes(&mut self, params: &[f64], indices: &[usize]) -> Result<f64, GprError> {
        S::eval_at_changes(self, params, indices)
    }
}

impl<O, C, B> IncrementalObjective for GprObjective<'_, O, IncrementalRecompute, C, B>
where
    C: DistanceCacheSlot,
    B: crate::gpr::AllocWorkspace,
{
    fn value_with_changes(&mut self, params: &[f64], indices: &[usize]) -> Result<f64, GprError> {
        self.model.value_from_leaf_grams(
            params,
            Some(indices),
            &mut self.leaf_grams,
            &mut self.leaves_primed,
        )
    }
}

impl<O, S, C, B> Differentiable for GprObjective<'_, O, S, C, B>
where
    S: EvalObjective,
    C: DistanceCacheSlot,
    B: crate::gpr::AllocWorkspace,
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

impl<O, S, C, B> TwiceDifferentiable for GprObjective<'_, O, S, C, B>
where
    S: EvalObjective,
    C: DistanceCacheSlot,
    B: crate::gpr::AllocWorkspace,
{
    fn hessian_into(&mut self, params: &[f64], out: &mut [f64]) -> Result<(), GprError> {
        self.model.hessian_into_fit(params, out)
    }
}

impl<O, S, C, B> HasBounds for GprObjective<'_, O, S, C, B>
where
    C: DistanceCacheSlot,
    B: crate::gpr::AllocWorkspace,
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
pub(crate) struct SgprObjective<'a, O, I = crate::FixedInducing> {
    model: &'a mut FittedSgpr<O, I>,
}

impl<'a, O, I> SgprObjective<'a, O, I> {
    pub(crate) fn new(model: &'a mut FittedSgpr<O, I>) -> Self {
        Self { model }
    }
}

impl<O, I: InducingLayout> Objective for SgprObjective<'_, O, I> {
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

impl<O, I: InducingLayout> Differentiable for SgprObjective<'_, O, I> {
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

impl<O, I: InducingLayout> TwiceDifferentiable for SgprObjective<'_, O, I> {
    fn hessian_into(&mut self, params: &[f64], out: &mut [f64]) -> Result<(), GprError> {
        self.model.hessian_into(params, out)
    }
}

impl<O, I: InducingLayout> HasBounds for SgprObjective<'_, O, I> {
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

    fn assert_close(actual: f64, expected: f64) {
        let scale = expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= TOL * scale,
            "actual={actual}, expected={expected}"
        );
    }

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
        assert_close(value, 1.5);
        assert_eq!(obj.values, 1);
        assert_eq!(obj.grads, 1);
        assert_eq!(obj.last_params, params);
        assert_close(grad[0], 0.25);
        assert_close(grad[1], 0.25);
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
        assert_close(got[0], params[0]);
        assert_close(got[1], params[1]);
        let mut kernel_theta = [0.0; 1];
        gpr.kernel().get_params(&mut kernel_theta).expect("len 1");
        assert_close(kernel_theta[0], params[0]);
        let mut lik_theta = [0.0; 1];
        gpr.likelihood().get_params(&mut lik_theta).expect("len 1");
        assert_close(lik_theta[0], params[1]);
        assert_close(nlml, gpr.neg_log_marginal_likelihood().expect("fitted"));
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
        assert_close(via_value, via_joint);
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
        assert_close(after[0], before[0]);
        assert_close(after[1], before[1]);
    }
}
