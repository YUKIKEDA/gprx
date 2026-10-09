//! [`SgprObjective`]: the negative VFE bound as an optimizer objective.

use crate::error::GprError;
use crate::kernel::ModelKernel;
use crate::objective::{Differentiable, Objective, TwiceDifferentiable};
use crate::param::Interval;

use super::{FittedSgpr, InducingLayout};

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
    K: ModelKernel = crate::kernel::KernelSpec,
> {
    model: &'a mut FittedSgpr<O, I, P, K>,
}

impl<'a, O, I, P, K: ModelKernel> SgprObjective<'a, O, I, P, K>
where
    P: crate::precision::GpScalar,
{
    pub(crate) fn new(model: &'a mut FittedSgpr<O, I, P, K>) -> Self {
        Self { model }
    }
}

impl<O, I: InducingLayout, P, K: ModelKernel> Objective for SgprObjective<'_, O, I, P, K>
where
    P: crate::precision::GpScalar,
{
    fn num_params(&self) -> usize {
        self.model.num_params()
    }

    fn fill_intervals(&self, out: &mut [Interval]) -> Result<(), GprError> {
        self.model.fill_intervals(out)
    }

    fn value(&mut self, params: &[f64]) -> Result<f64, GprError> {
        crate::objective::count_value_call!();
        match self.model.set_params(params) {
            Ok(()) => self.model.neg_log_marginal_likelihood(),
            Err(GprError::CholeskyFailed { .. }) => Ok(1.0e300),
            Err(err) => Err(err),
        }
    }
}

impl<O, I: InducingLayout, P, K: ModelKernel> Differentiable for SgprObjective<'_, O, I, P, K>
where
    P: crate::precision::GpScalar,
{
    fn gradient_into(&mut self, params: &[f64], out: &mut [f64]) -> Result<(), GprError> {
        crate::objective::count_joint_call!();
        self.model.value_and_gradient_into(params, out).map(|_| ())
    }

    fn value_and_gradient_into(
        &mut self,
        params: &[f64],
        out: &mut [f64],
    ) -> Result<f64, GprError> {
        crate::objective::count_joint_call!();
        self.model.value_and_gradient_into(params, out)
    }
}

impl<O, I: InducingLayout, P, K: ModelKernel> TwiceDifferentiable for SgprObjective<'_, O, I, P, K>
where
    P: crate::precision::GpScalar,
{
    fn hessian_into(&mut self, params: &[f64], out: &mut [f64]) -> Result<(), GprError> {
        self.model.hessian_into(params, out)
    }
}
