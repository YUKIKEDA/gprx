//! Negative log marginal likelihood as an optimizer objective.
//!
//! The GPR fit objective borrows [`FittedGpr`] during `fit` / `refit` and forwards
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
/// `indices` is the list of flat `θ` positions that changed. Empty, duplicate,
/// or out-of-range indices are a [`GprError`] at this boundary. Full rebuilds
/// use [`Objective::value`]. [`crate::FullRecompute`] does not implement this
/// trait. The GPR fit objective with [`crate::IncrementalRecompute`] does.
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
pub struct GprObjective<
    'a,
    O,
    S,
    C: crate::gpr::DistanceCacheSlot = crate::CachedDistances,
    B: crate::gpr::AllocWorkspace = crate::RetainCholesky,
    M = crate::math::Accurate,
    P: crate::precision::GpScalar = crate::precision::DoublePrecision,
> {
    model: &'a mut FittedGpr<O, S, C, B, M, P>,
    scratch: Vec<f64>,
    leaf_grams: Vec<Mat<P::Storage>>,
    leaves_primed: bool,
}

impl<'a, O, S, C, B, M, P> GprObjective<'a, O, S, C, B, M, P>
where
    C: DistanceCacheSlot,
    B: crate::gpr::AllocWorkspace,
    P: crate::precision::GpScalar,
    M: crate::math::KernelMath,
{
    pub(crate) fn new(model: &'a mut FittedGpr<O, S, C, B, M, P>) -> Self {
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
    fn eval_value<O, C, B, M, P>(
        obj: &mut GprObjective<'_, O, Self, C, B, M, P>,
        params: &[f64],
    ) -> Result<f64, GprError>
    where
        C: DistanceCacheSlot,
        B: crate::gpr::AllocWorkspace,
        P: crate::precision::GpScalar,
        M: crate::math::KernelMath;

    fn eval_at_changes<O, C, B, M, P>(
        obj: &mut GprObjective<'_, O, Self, C, B, M, P>,
        params: &[f64],
        indices: &[usize],
    ) -> Result<f64, GprError>
    where
        C: DistanceCacheSlot,
        B: crate::gpr::AllocWorkspace,
        P: crate::precision::GpScalar,
        M: crate::math::KernelMath,
    {
        let _ = indices;
        Self::eval_value(obj, params)
    }
}

impl EvalObjective for FullRecompute {
    fn eval_value<O, C, B, M, P>(
        obj: &mut GprObjective<'_, O, Self, C, B, M, P>,
        params: &[f64],
    ) -> Result<f64, GprError>
    where
        C: DistanceCacheSlot,
        B: crate::gpr::AllocWorkspace,
        P: crate::precision::GpScalar,
        M: crate::math::KernelMath,
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
    fn eval_value<O, C, B, M, P>(
        obj: &mut GprObjective<'_, O, Self, C, B, M, P>,
        params: &[f64],
    ) -> Result<f64, GprError>
    where
        C: DistanceCacheSlot,
        B: crate::gpr::AllocWorkspace,
        P: crate::precision::GpScalar,
        M: crate::math::KernelMath,
    {
        obj.model
            .value_from_leaf_grams(params, None, &mut obj.leaf_grams, &mut obj.leaves_primed)
    }

    fn eval_at_changes<O, C, B, M, P>(
        obj: &mut GprObjective<'_, O, Self, C, B, M, P>,
        params: &[f64],
        indices: &[usize],
    ) -> Result<f64, GprError>
    where
        C: DistanceCacheSlot,
        B: crate::gpr::AllocWorkspace,
        P: crate::precision::GpScalar,
        M: crate::math::KernelMath,
    {
        obj.model.value_from_leaf_grams(
            params,
            Some(indices),
            &mut obj.leaf_grams,
            &mut obj.leaves_primed,
        )
    }
}

impl<O, S, C, B, M, P> Objective for GprObjective<'_, O, S, C, B, M, P>
where
    S: EvalObjective,
    C: DistanceCacheSlot,
    B: crate::gpr::AllocWorkspace,
    P: crate::precision::GpScalar,
    M: crate::math::KernelMath,
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

impl<O, C, B, M, P> IncrementalObjective for GprObjective<'_, O, IncrementalRecompute, C, B, M, P>
where
    C: DistanceCacheSlot,
    B: crate::gpr::AllocWorkspace,
    P: crate::precision::GpScalar,
    M: crate::math::KernelMath,
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

impl<O, S, C, B, M, P> Differentiable for GprObjective<'_, O, S, C, B, M, P>
where
    S: EvalObjective,
    C: DistanceCacheSlot,
    B: crate::gpr::AllocWorkspace,
    P: crate::precision::GpScalar,
    M: crate::math::KernelMath,
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

impl<O, S, C, B, M, P> TwiceDifferentiable for GprObjective<'_, O, S, C, B, M, P>
where
    S: EvalObjective,
    C: DistanceCacheSlot,
    B: crate::gpr::AllocWorkspace,
    P: crate::precision::GpScalar,
    M: crate::math::KernelMath,
{
    fn hessian_into(&mut self, params: &[f64], out: &mut [f64]) -> Result<(), GprError> {
        self.model.hessian_into_fit(params, out)
    }
}

impl<O, S, C, B, M, P> HasBounds for GprObjective<'_, O, S, C, B, M, P>
where
    C: DistanceCacheSlot,
    B: crate::gpr::AllocWorkspace,
    P: crate::precision::GpScalar,
    M: crate::math::KernelMath,
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
    M = crate::math::Accurate,
    P: crate::precision::GpScalar = crate::precision::DoublePrecision,
> {
    model: &'a mut FittedSgpr<O, I, M, P>,
}

impl<'a, O, I, M, P> SgprObjective<'a, O, I, M, P>
where
    M: crate::math::KernelMath,
    P: crate::precision::GpScalar,
{
    pub(crate) fn new(model: &'a mut FittedSgpr<O, I, M, P>) -> Self {
        Self { model }
    }
}

impl<O, I: InducingLayout, M, P> Objective for SgprObjective<'_, O, I, M, P>
where
    M: crate::math::KernelMath,
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

impl<O, I: InducingLayout, M, P> Differentiable for SgprObjective<'_, O, I, M, P>
where
    M: crate::math::KernelMath,
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

impl<O, I: InducingLayout, M, P> TwiceDifferentiable for SgprObjective<'_, O, I, M, P>
where
    M: crate::math::KernelMath,
    P: crate::precision::GpScalar,
{
    fn hessian_into(&mut self, params: &[f64], out: &mut [f64]) -> Result<(), GprError> {
        self.model.hessian_into(params, out)
    }
}

impl<O, I: InducingLayout, M, P> HasBounds for SgprObjective<'_, O, I, M, P>
where
    M: crate::math::KernelMath,
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
