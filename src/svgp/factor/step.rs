//! The buffers an Adam fit keeps between mini-batch steps.

use super::assemble::unpack_q_into;
use super::gradient::GradBuffers;
use crate::data::pack_points;
use crate::error::{CholeskyStage, GprError};
use crate::kernel::{CompiledKernel, GramInputs, KernelScalar, Triangle};
use crate::linalg::{cholesky_lower_with_backup, llt_scratch};
use crate::precision::GpScalar;
use crate::sparse::KernelScratch;
use crate::svgp::FittedSvgp;
use dyn_stack::MemBuffer;
use faer::Mat;

/// Everything one Adam step reads and writes besides the model: the kernel
/// compiled at both scalars and updated in place, the packed `Z`, the next
/// factor of `K_mm` and `q` (swapped with the model's on commit), and the
/// gradient's [`GradBuffers`]. Built once per fit, so a step allocates
/// nothing after the first one (see [`GradBuffers`] for the one exception).
pub(crate) struct AdamStep<S: KernelScalar> {
    theta: Vec<f64>,
    theta_prev: Vec<f64>,
    compiled_storage: CompiledKernel<S>,
    pub(crate) compiled: CompiledKernel<f64>,
    z: Mat<S>,
    k_mm: Mat<S>,
    backup: Mat<S>,
    chol: MemBuffer,
    q_mean: Vec<f64>,
    q_l: Mat<f64>,
    ks: KernelScratch<S>,
    pub(crate) grad: GradBuffers,
}

impl<S: KernelScalar> AdamStep<S> {
    pub(crate) fn new<P>(model: &FittedSvgp<P>) -> Self
    where
        P: GpScalar<Storage = S>,
    {
        let core = &model.core;
        let m = core.m;
        let z64 = pack_points(&core.z_train, m, core.d);
        let mut cast = S::empty_cols();
        let z = S::storage_cols(z64.as_ref(), &mut cast).to_owned();
        Self {
            theta: vec![0.0; core.theta_len()],
            theta_prev: vec![0.0; core.theta_len()],
            compiled_storage: core.kernel.compile_as::<S>(),
            compiled: core.kernel.compile(),
            z,
            k_mm: Mat::zeros(m, m),
            backup: Mat::zeros(m, m),
            chol: llt_scratch::<S>(m),
            q_mean: vec![0.0; m],
            q_l: Mat::zeros(m, m),
            ks: KernelScratch::new(),
            grad: GradBuffers::default(),
        }
    }
}

impl<P> FittedSvgp<P>
where
    P: GpScalar,
{
    /// [`Self::set_params_light`] on the buffers of `step`: `θ` is written in
    /// place into the spec and both compiled kernels, and the new factor of
    /// `K_mm` and `q` are swapped in. Committed together only after `K_mm`
    /// factors; on error the model and `step.compiled` are unchanged.
    pub(crate) fn set_params_step<M: crate::math::KernelMath>(
        &mut self,
        params: &[f64],
        step: &mut AdamStep<P::Storage>,
    ) -> Result<(), GprError> {
        crate::data::require_count(params.len(), self.num_params(), "parameters")?;
        let core = &mut self.core;
        let (m, n_theta) = (core.m, core.theta_len());
        let n_kernel = core.kernel.num_params();
        let (theta, q) = params.split_at(n_theta);
        unpack_q_into(q, m, &mut step.q_mean, &mut step.q_l)?;
        let mut likelihood = core.likelihood;
        likelihood.set_params(&theta[n_kernel..])?;
        core.kernel.get_params(&mut step.theta_prev[..n_kernel])?;
        step.theta[..n_kernel].copy_from_slice(&theta[..n_kernel]);
        let (new_k, prev_k) = (&step.theta[..n_kernel], &step.theta_prev[..n_kernel]);
        core.kernel.set_params_in_place(new_k, prev_k)?;
        let factored = (|| {
            step.compiled_storage.set_params_in_place(new_k, prev_k)?;
            step.ks.gram::<M>(
                &step.compiled_storage,
                GramInputs::points(step.z.as_ref()),
                step.k_mm.as_mut(),
                Triangle::Lower,
            )?;
            cholesky_lower_with_backup(
                &mut step.k_mm,
                &mut step.backup,
                &mut step.chol,
                core.jitter.retry_jitters(),
                CholeskyStage::Fit,
            )
        })();
        if let Err(err) = factored {
            let _ = core.kernel.set_params_in_place(prev_k, new_k);
            let _ = step.compiled_storage.set_params_in_place(prev_k, new_k);
            return Err(err);
        }
        step.compiled.set_params_in_place(new_k, prev_k)?;
        core.likelihood = likelihood;
        std::mem::swap(&mut self.k_mm_l, &mut step.k_mm);
        std::mem::swap(&mut self.q_mean, &mut step.q_mean);
        std::mem::swap(&mut self.q_l, &mut step.q_l);
        Ok(())
    }
}
