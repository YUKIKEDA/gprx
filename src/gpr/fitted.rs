//! [`FittedGpr`] factorization, prediction, and refit.

use dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt;
use faer::{Mat, MatMut, MatRef};

use crate::data::{pack_points, validate_training};
use crate::error::{CholeskyStage, GprError};
use crate::kernel::ScalarOps;
use crate::kernel::{CompiledKernel, KernelScalar, KernelSpec, Triangle};
use crate::likelihood::GaussianLikelihood;
use crate::linalg::{
    faer_par, faer_par_dims, frobenius_lower, gemv_full, gemv_sym_lower, solve_llt_in_place,
    symmetrize_lower, trace_product,
};
use crate::objective::GprObjective;
use crate::online::OnlineWorkspace;
use crate::optimizer::{Fixed, OptResult, Optimizer};
use crate::param::Interval;
use crate::persist::{self, PersistedModel};
use crate::precision::{GpScalar, StoredFactor};
use crate::transform::{TargetTransform, Transform, UnfittedTarget, UnfittedTransform};
use crate::workspace::{FitWorkspace, HessianScratch, QueryWorkspace, WorkspaceCore};
use crate::{PredictOptions, Prediction, PredictiveCovariance};

use super::super::online::OnlineGpr;

use super::super::factor::{
    FactorPolicy, apply_compiled_to, factor_train_with_policy, factor_written_k_with_policy,
    fill_cached_inputs, neg_mll_from_factor,
};
use super::{DistanceCachePolicy, FitBuffers, with_kernel_exp};
use super::{FittedGpr, Gpr, GprCore, Policies};
use crate::persist::MappedTensors;

/// [`Optimizer::USES_CHANGE_INDICES`] of `O` for the objective `obj`.
fn uses_change_indices<Obj, O: Optimizer<Obj>>(_obj: &Obj) -> bool {
    O::USES_CHANGE_INDICES
}

/// Allocates fit buffers for `n` points under `policies`.
///
/// A kernel that reads neither pairwise distances nor the ARD `(Δx_d)²`
/// tensor gets no distance cache, whatever the policy says.
pub(crate) fn fit_buffers<P: GpScalar>(
    n: usize,
    policies: Policies,
    compiled: &CompiledKernel<P::Storage>,
) -> Result<FitBuffers<P>, GprError> {
    let cache = if compiled.reads_distances()? || compiled.needs_ard_sq_diff() {
        policies.distance_cache
    } else {
        DistanceCachePolicy::Uncached
    };
    FitBuffers::<P>::new(n, cache, policies.cholesky_buffer)
}

/// Borrowed fit state: the shared core plus the LLT buffers.
///
/// Every hyperparameter write (`set_params`, gradient, Hessian, `fit`,
/// `refit`) runs here. [`FittedGpr`] lends its own buffers.
/// [`OnlineGpr`] lends temporary ones filled from its LDLT.
pub(crate) struct ExactFit<'a, P: GpScalar> {
    pub(crate) core: &'a mut GprCore<P>,
    pub(crate) store: &'a mut LltStore<P>,
}

/// Per-leaf Gram matrices an incremental objective keeps during `fit` /
/// `refit` (`L · n²`, outside the fit buffers), plus the reused bookkeeping
/// for one coordinate step.
pub(crate) struct LeafCache<S> {
    grams: Vec<Mat<S>>,
    dirty: Vec<bool>,
    /// `params` of the last evaluation that factored.
    last: Vec<f64>,
    /// `grams` match `last`.
    primed: bool,
}

impl<S: KernelScalar> LeafCache<S> {
    pub(crate) fn new() -> Self {
        Self {
            grams: Vec::new(),
            dirty: Vec::new(),
            last: Vec::new(),
            primed: false,
        }
    }

    /// Sizes the Grams for `n_leaves` leaves of order `n`. A resize drops them.
    fn fit(&mut self, n_leaves: usize, n: usize) {
        if self.grams.len() != n_leaves || self.grams.first().is_some_and(|m| m.nrows() != n) {
            self.grams = (0..n_leaves).map(|_| Mat::<S>::zeros(n, n)).collect();
            self.primed = false;
        }
        self.dirty.resize(n_leaves, true);
    }

    /// Rejects a step that changed a coordinate `changed` does not list.
    fn require_listed(&self, params: &[f64], changed: &[usize]) -> Result<(), GprError> {
        for (j, (prev, next)) in self.last.iter().zip(params).enumerate() {
            if prev.to_bits() != next.to_bits() && !changed.contains(&j) {
                return Err(GprError::IndexOutOfRange {
                    reason: format!(
                        "coordinate {j} changed since the previous evaluation but is not in the change indices"
                    ),
                });
            }
        }
        Ok(())
    }

    fn record(&mut self, params: &[f64]) {
        self.last.clear();
        self.last.extend_from_slice(params);
        self.primed = true;
    }
}

/// The LLT factor of a batch fit: the fit buffers, plus a memory-mapped
/// `f64` `L` while a loaded model has not been written to.
pub(crate) struct LltStore<P: GpScalar> {
    pub(crate) buffers: FitBuffers<P>,
    /// Loaded `L`. Every factor write drops it first and lands in `buffers`.
    mapped: Option<MappedTensors>,
}

impl<P: GpScalar> LltStore<P> {
    pub(crate) fn new(buffers: FitBuffers<P>) -> Self {
        Self {
            buffers,
            mapped: None,
        }
    }

    pub(crate) fn with_mapped(buffers: FitBuffers<P>, mapped: Option<MappedTensors>) -> Self {
        Self { buffers, mapped }
    }

    /// `L` of the current training system.
    pub(crate) fn l(&self) -> MatRef<'_, P::Storage> {
        let mapped = self.mapped.as_ref().map(|mapped| mapped.l_view());
        P::view_factor(mapped, self.buffers.core().k_matrix.as_ref())
    }

    /// `L`, and the per-thread kernel scratch, borrowed together.
    pub(crate) fn l_and_thread_scratch(
        &mut self,
    ) -> (MatRef<'_, P::Storage>, &mut Vec<Mat<P::Storage>>) {
        let mapped = self.mapped.as_ref().map(|mapped| mapped.l_view());
        let WorkspaceCore {
            k_matrix,
            thread_scratch,
            ..
        } = self.buffers.core_mut();
        (P::view_factor(mapped, k_matrix.as_ref()), thread_scratch)
    }

    /// Drops the mapped `L` before the buffers are written.
    fn release_mapped(&mut self) {
        self.mapped = None;
    }
}

impl<P: GpScalar> Clone for LltStore<P> {
    /// Copies a mapped `L` into the clone's own buffers.
    fn clone(&self) -> Self {
        let mut buffers = self.buffers.clone();
        if let Some(mapped) = &self.mapped {
            P::copy_mapped_l(mapped.l_view(), buffers.core_mut().k_matrix.as_mut());
        }
        Self {
            buffers,
            mapped: None,
        }
    }
}

impl<P: GpScalar> ExactFit<'_, P> {
    pub(crate) fn reborrow(&mut self) -> ExactFit<'_, P> {
        ExactFit {
            core: &mut *self.core,
            store: &mut *self.store,
        }
    }

    pub(crate) fn num_params(&self) -> usize {
        self.core.num_params()
    }

    fn get_params(&self, out: &mut [f64]) -> Result<(), GprError> {
        self.core.get_params(out)
    }

    /// Runs `optimizer` from the current `θ`, then leaves `L` / `α` at the result.
    pub(crate) fn optimize<O>(&mut self, optimizer: &O) -> Result<(), GprError>
    where
        O: for<'b> Optimizer<GprObjective<'b, P>>,
    {
        let mut init = vec![0.0; self.num_params()];
        self.get_params(&mut init)?;
        let kernel_before = self.core.kernel.clone();
        let likelihood_before = self.core.likelihood;
        let result = {
            let obj = GprObjective::new(self.reborrow());
            let uses_change_indices = uses_change_indices::<_, O>(&obj);
            let mut obj = obj.with_change_indices(uses_change_indices);
            optimizer.minimize(&mut obj, &init)
        };
        self.commit_or_revert_optimize(kernel_before, likelihood_before, result)?;
        self.finish()
    }

    /// Rebuilds `L` at the current `θ` and publishes the predict `α`.
    pub(crate) fn refactor(&mut self) -> Result<(), GprError> {
        self.factorize_current()?;
        self.finish()
    }

    /// Ends every public write: `L` back in place of a reuse `W`, then the
    /// predict `α` for that `L`. Between public calls the predict `α`
    /// always matches the stored factor.
    fn finish(&mut self) -> Result<(), GprError> {
        self.restore_cholesky_if_overwritten()?;
        self.publish_predict_alpha()
    }

    pub(crate) fn publish_predict_alpha(&mut self) -> Result<(), GprError> {
        let jitter = self.store.buffers.core().factor_jitter;
        self.core.publish_predict_alpha(
            StoredFactor::Llt(self.store.l()),
            jitter,
            CholeskyStage::Fit,
        )
    }

    pub(crate) fn set_params(&mut self, params: &[f64]) -> Result<(), GprError> {
        crate::data::require_count(params.len(), self.num_params(), "parameters")?;
        self.factor_at(params)?;
        if let Err(err) = self.finish() {
            self.restore_theta();
            let _ = self.factorize_current();
            let _ = self.publish_predict_alpha();
            return Err(err);
        }
        Ok(())
    }

    pub(crate) fn value_and_gradient_into(
        &mut self,
        params: &[f64],
        out: &mut [f64],
    ) -> Result<f64, GprError> {
        let nlml = self.value_and_gradient_into_fit(params, out)?;
        self.finish()?;
        Ok(nlml)
    }

    /// Writes `θ = params` and factors `A` at it.
    ///
    /// The one rollback rule for hyperparameter writes: `θ` is written in
    /// place (no clone of the kernel trees), with the previous `θ` kept in a
    /// reused buffer. When `A` does not factor, that `θ` is written back and
    /// `L` / `α` are rebuilt at it, so nothing is copied up front.
    fn factor_at(&mut self, params: &[f64]) -> Result<(), GprError> {
        self.write_theta(params)?;
        if let Err(err) = self.factor() {
            self.restore_theta();
            let _ = self.factorize_current();
            return Err(err);
        }
        Ok(())
    }

    /// Writes `params` into the stored kernel, compiled kernel, and likelihood.
    ///
    /// The previous `θ` goes to the fit buffers' `θ` scratch for
    /// [`Self::restore_theta`]. A rejected slice changes nothing.
    fn write_theta(&mut self, params: &[f64]) -> Result<(), GprError> {
        let n_kernel = self.core.kernel.num_params();
        let mut likelihood = self.core.likelihood;
        likelihood.set_params(&params[n_kernel..])?;
        let prev = &mut self.store.buffers.core_mut().theta;
        prev.resize(params.len(), 0.0);
        self.core.get_params(prev)?;
        let (kernel_prev, _) = prev.split_at(n_kernel);
        self.core
            .kernel
            .set_params_in_place(&params[..n_kernel], kernel_prev)?;
        if let Err(err) = self
            .core
            .compiled
            .set_params_in_place(&params[..n_kernel], kernel_prev)
        {
            let _ = self
                .core
                .kernel
                .set_params_in_place(kernel_prev, kernel_prev);
            return Err(err);
        }
        self.core.likelihood = likelihood;
        Ok(())
    }

    /// Writes back the `θ` the last [`Self::write_theta`] replaced.
    fn restore_theta(&mut self) {
        let n_kernel = self.core.kernel.num_params();
        let prev = &self.store.buffers.core().theta;
        let (kernel_prev, likelihood_prev) = prev.split_at(n_kernel);
        let _ = self
            .core
            .kernel
            .set_params_in_place(kernel_prev, kernel_prev);
        let _ = self
            .core
            .compiled
            .set_params_in_place(kernel_prev, kernel_prev);
        let _ = self.core.likelihood.set_params(likelihood_prev);
    }

    /// Factors `A` at the stored `θ` into the buffers.
    fn factor(&mut self) -> Result<(), GprError> {
        self.store.release_mapped();
        let x = P::Storage::storage_cols(
            self.core
                .x
                .as_ref()
                .submatrix(0, 0, self.core.n, self.core.d),
            &mut self.core.x_cast,
        );
        with_kernel_exp!(self.core.policies.math, M => factor_train_with_policy::<_, _, M>(
            &self.core.compiled,
            x,
            &mut self.store.buffers,
            &self.core.y_train,
            self.core.likelihood.noise_variance(),
            FactorPolicy {
                jitter: self.core.policies.jitter,
                stage: CholeskyStage::Fit,
            },
        ))?;
        self.commit_factor();
        Ok(())
    }

    /// Joint MLL+grad used during `fit`. Does not restore `L` when the buffer
    /// overwrites the factor; the optimizer's next step rebuilds `A`.
    pub(crate) fn value_and_gradient_into_fit(
        &mut self,
        params: &[f64],
        out: &mut [f64],
    ) -> Result<f64, GprError> {
        let n_kernel = self.core.kernel.num_params();
        let n_params = self.num_params();
        crate::data::require_count(params.len(), n_params, "parameters")?;
        crate::data::require_count(out.len(), n_params, "parameters")?;
        self.factor_at(params)?;
        let n = self.core.n;
        let mut rows = P::Storage::empty_rows();
        let y = P::Storage::storage_rows(&self.core.y_train, &mut rows);
        let nlml = neg_mll_from_factor(
            self.store.buffers.core().k_matrix.as_ref(),
            y,
            &self.core.factor_alpha,
            n,
        )
        .to_f64();
        self.fill_gradient_from_factor(n_kernel, n, out)?;
        Ok(nlml)
    }

    /// Rebuilds dirty compiled leaves, recombines the tree, and factors.
    ///
    /// `indices` lists every coordinate that differs from the previous
    /// evaluation through `cache`; only the leaves they touch are rebuilt.
    /// `None`, or a cache that holds nothing yet, rebuilds every leaf.
    pub(crate) fn value_from_leaf_grams(
        &mut self,
        params: &[f64],
        indices: Option<&[usize]>,
        cache: &mut LeafCache<P::Storage>,
    ) -> Result<f64, GprError> {
        let n_kernel = self.core.kernel.num_params();
        let n_params = self.num_params();
        crate::data::require_count(params.len(), n_params, "parameters")?;
        if let Some(changed) = indices {
            require_change_indices(changed, n_params)?;
        }
        let n = self.core.n;
        let n_leaves = self.core.compiled.leaf_count();
        cache.fit(n_leaves, n);
        match (cache.primed, indices) {
            (true, Some(changed)) => {
                cache.require_listed(params, changed)?;
                cache.dirty.fill(false);
                for &j in changed {
                    if j < n_kernel {
                        cache.dirty[self.core.compiled.leaf_index_for_param(j)?] = true;
                    }
                }
            }
            _ => cache.dirty.fill(true),
        }
        self.write_theta(params)?;
        // Grams are only trusted again once this evaluation factors.
        cache.primed = false;
        if let Err(err) = self.rebuild_dirty_leaves(cache) {
            self.restore_theta();
            return Err(err);
        }
        self.store.release_mapped();
        let compiled = &self.core.compiled;
        let grams = &cache.grams;
        if let Err(err) = factor_written_k_with_policy(
            &mut self.store.buffers,
            &self.core.y_train,
            self.core.likelihood.noise_variance(),
            FactorPolicy {
                jitter: self.core.policies.jitter,
                stage: CholeskyStage::Fit,
            },
            |ws| {
                let core = ws.core_mut();
                compiled.combine_from_leaf_grams(
                    grams,
                    core.k_matrix.as_mut(),
                    core.exp_buf.as_mut(),
                    &mut core.nested,
                    Triangle::Lower,
                )
            },
        ) {
            self.restore_theta();
            let _ = self.factorize_current();
            return Err(err);
        }
        cache.record(params);
        self.commit_factor();
        let mut rows = P::Storage::empty_rows();
        let y = P::Storage::storage_rows(&self.core.y_train, &mut rows);
        Ok(neg_mll_from_factor(
            self.store.buffers.core().k_matrix.as_ref(),
            y,
            &self.core.factor_alpha,
            n,
        )
        .to_f64())
    }

    /// Re-evaluates the leaves `cache.dirty` marks at the stored `θ`.
    fn rebuild_dirty_leaves(&mut self, cache: &mut LeafCache<P::Storage>) -> Result<(), GprError> {
        let LeafCache { grams, dirty, .. } = cache;
        for (i, slot) in grams.iter_mut().enumerate() {
            if dirty[i] {
                let x = P::Storage::storage_cols(
                    self.core
                        .x
                        .as_ref()
                        .submatrix(0, 0, self.core.n, self.core.d),
                    &mut self.core.x_cast,
                );
                with_kernel_exp!(self.core.policies.math, M => apply_compiled_to::<_, _, M>(
                    self.core.compiled.leaf_at(i)?,
                    x,
                    &mut self.store.buffers,
                    slot.as_mut(),
                ))?;
            }
        }
        Ok(())
    }

    pub(crate) fn hessian_into(&mut self, params: &[f64], out: &mut [f64]) -> Result<(), GprError> {
        self.hessian_into_fit(params, out)?;
        self.finish()
    }

    pub(crate) fn hessian_into_fit(
        &mut self,
        params: &[f64],
        out: &mut [f64],
    ) -> Result<(), GprError> {
        let n_kernel = self.core.kernel.num_params();
        let n_params = self.num_params();
        crate::data::require_count(params.len(), n_params, "parameters")?;
        crate::data::require_count(out.len(), n_params * n_params, "parameters")?;
        self.factor_at(params)?;
        let n = self.core.n;
        self.fill_hessian_from_factor(n_kernel, n, out)
    }

    fn fill_hessian_from_factor(
        &mut self,
        n_kernel: usize,
        n: usize,
        out: &mut [f64],
    ) -> Result<(), GprError> {
        self.store.buffers.core_mut().ensure_kernel_scratch(n)?;
        if self.core.compiled.needs_grad_scratch() {
            self.store.buffers.core_mut().ensure_kernel_scratch(n)?;
        }
        self.store
            .buffers
            .form_gradient_w(&self.core.factor_alpha, n);
        out.fill(0.0);
        let n_params = n_kernel + 1;
        let noise = self.core.likelihood.noise_variance();
        let thread_scratch = std::mem::take(&mut self.store.buffers.core_mut().thread_scratch);
        let second = (|| {
            for i in 0..n_params {
                for j in i..n_params {
                    self.write_second_deriv(n_kernel, i, j, n)?;
                    let inner = frobenius_lower(
                        self.store.buffers.gradient_w(),
                        self.store.buffers.core().exp_buf.as_ref(),
                        n,
                    );
                    let hij = -0.5 * inner.to_f64();
                    out[i * n_params + j] = hij;
                    out[j * n_params + i] = hij;
                }
            }
            Ok::<(), GprError>(())
        })();
        self.store.buffers.core_mut().thread_scratch = thread_scratch;
        second?;
        let mut hess = std::mem::take(&mut self.store.buffers.core_mut().hessian);
        hess.ensure(n);
        let first = self.add_first_order(n_kernel, n, noise, out, &mut hess);
        self.store.buffers.core_mut().hessian = hess;
        first
    }

    fn add_first_order(
        &mut self,
        n_kernel: usize,
        n: usize,
        noise: f64,
        out: &mut [f64],
        hess: &mut HessianScratch<P::Storage>,
    ) -> Result<(), GprError> {
        self.add_noise_first_order(n_kernel, n, noise, out, hess)?;
        if !self.store.buffers.has_dedicated_w() {
            self.factorize_current()?;
        }
        self.add_kernel_first_order(n_kernel, n, out, hess)
    }

    fn write_second_deriv(
        &mut self,
        n_kernel: usize,
        i: usize,
        j: usize,
        n: usize,
    ) -> Result<(), GprError> {
        if i >= n_kernel || j >= n_kernel {
            zero_and_maybe_noise(
                self.store.buffers.core_mut().exp_buf.as_mut(),
                n,
                i == n_kernel && j == n_kernel,
                self.core.likelihood.noise_variance(),
            );
            return Ok(());
        }
        let x = P::Storage::storage_cols(
            self.core
                .x
                .as_ref()
                .submatrix(0, 0, self.core.n, self.core.d),
            &mut self.core.x_cast,
        );
        let (core, dist) = self.store.buffers.split_fit();
        let WorkspaceCore {
            exp_buf,
            kernel_scratch,
            nested,
            thread_scratch,
            ..
        } = core;
        let inputs = fill_cached_inputs(&self.core.compiled, x, dist, thread_scratch)?;
        with_kernel_exp!(self.core.policies.math, M => self.core.compiled.hess_gram::<M>(
            inputs,
            exp_buf.as_mut(),
            (i, j),
            Triangle::Lower,
            kernel_scratch.as_mut(),
            nested,
        ))
    }

    fn write_first_deriv(&mut self, idx: usize) -> Result<(), GprError> {
        let x = P::Storage::storage_cols(
            self.core
                .x
                .as_ref()
                .submatrix(0, 0, self.core.n, self.core.d),
            &mut self.core.x_cast,
        );
        let (core, dist) = self.store.buffers.split_fit();
        let WorkspaceCore {
            exp_buf,
            kernel_scratch,
            nested,
            thread_scratch,
            ..
        } = core;
        let inputs = fill_cached_inputs(&self.core.compiled, x, dist, thread_scratch)?;
        with_kernel_exp!(self.core.policies.math, M => self.core.compiled.grad_gram::<M>(
            inputs,
            exp_buf.as_mut(),
            idx,
            Triangle::Lower,
            kernel_scratch.as_mut(),
            nested,
        ))
    }

    fn add_noise_first_order(
        &mut self,
        n_kernel: usize,
        n: usize,
        noise: f64,
        out: &mut [f64],
        hess: &mut HessianScratch<P::Storage>,
    ) -> Result<(), GprError> {
        let n_params = n_kernel + 1;
        let zero = P::Storage::from_f64(0.0);
        let two = P::Storage::from_f64(2.0);
        let noise_s = P::Storage::from_f64(noise);
        let alpha = &self.core.factor_alpha;
        let tr_kinv2 = {
            let w = self.store.buffers.gradient_w();
            // `w_noise` holds `W α`, then `K⁻¹ α = α (αᵀα) - W α`, then `σn² K⁻¹ α`.
            gemv_sym_lower(w, alpha, &mut hess.w_noise, n);
            let mut alpha_dot = zero;
            for a in alpha {
                alpha_dot += *a * *a;
            }
            for (w_n, a) in hess.w_noise.iter_mut().zip(alpha) {
                *w_n = noise_s * (*a * alpha_dot - *w_n);
            }
            let mut tr_kinv2 = zero;
            for col in 0..n {
                let kinv_cc = alpha[col] * alpha[col] - w[(col, col)];
                tr_kinv2 += kinv_cc * kinv_cc;
                for row in col + 1..n {
                    let kinv_rc = alpha[row] * alpha[col] - w[(row, col)];
                    tr_kinv2 += two * kinv_rc * kinv_rc;
                }
            }
            tr_kinv2
        };
        let mut un_wn = zero;
        for (a, w_n) in alpha.iter().zip(&hess.w_noise) {
            un_wn += noise_s * *a * *w_n;
        }
        let nn = n_kernel;
        out[nn * n_params + nn] += -0.5 * noise * noise * tr_kinv2.to_f64() + un_wn.to_f64();

        self.store.buffers.core_mut().ensure_kernel_scratch(n)?;
        let thread_scratch = std::mem::take(&mut self.store.buffers.core_mut().thread_scratch);
        let cross = (|| {
            for i in 0..n_kernel {
                self.write_first_deriv(i)?;
                let ki = self.store.buffers.core().exp_buf.as_ref();
                let alpha = &self.core.factor_alpha;
                let tr = trace_ki_kinv2(ki, self.store.buffers.gradient_w(), alpha, n);
                gemv_sym_lower(ki, alpha, &mut hess.u_i, n);
                let mut ui_wn = zero;
                for (u, w_n) in hess.u_i.iter().zip(&hess.w_noise) {
                    ui_wn += *u * *w_n;
                }
                let hij = -0.5 * noise * tr.to_f64() + ui_wn.to_f64();
                out[i * n_params + nn] += hij;
                out[nn * n_params + i] += hij;
            }
            Ok::<(), GprError>(())
        })();
        self.store.buffers.core_mut().thread_scratch = thread_scratch;
        cross
    }

    fn add_kernel_first_order(
        &mut self,
        n_kernel: usize,
        n: usize,
        out: &mut [f64],
        hess: &mut HessianScratch<P::Storage>,
    ) -> Result<(), GprError> {
        if n_kernel == 0 {
            return Ok(());
        }
        self.store.buffers.core_mut().ensure_kernel_scratch(n)?;
        let n_params = n_kernel + 1;
        let zero = P::Storage::from_f64(0.0);
        let thread_scratch = std::mem::take(&mut self.store.buffers.core_mut().thread_scratch);
        let result = (|| {
            for j in 0..n_kernel {
                self.write_first_deriv(j)?;
                gemv_sym_lower(
                    self.store.buffers.core().exp_buf.as_ref(),
                    &self.core.factor_alpha,
                    &mut hess.u_j,
                    n,
                );
                symmetrize_lower(self.store.buffers.core_mut().exp_buf.as_mut(), n);
                self.solve_exp_against_l(n);
                // `write_first_deriv(i)` below overwrites `exp_buf`, so keep `Q_j`.
                hess.q.copy_from(self.store.buffers.core().exp_buf.as_ref());
                gemv_full(hess.q.as_ref(), &self.core.factor_alpha, &mut hess.w_j, n);
                for i in 0..=j {
                    let tr;
                    let mut ui_wj = zero;
                    if i == j {
                        tr = trace_product(hess.q.as_ref(), hess.q.as_ref(), n);
                        for (u, w) in hess.u_j.iter().zip(&hess.w_j) {
                            ui_wj += *u * *w;
                        }
                    } else {
                        self.write_first_deriv(i)?;
                        gemv_sym_lower(
                            self.store.buffers.core().exp_buf.as_ref(),
                            &self.core.factor_alpha,
                            &mut hess.u_i,
                            n,
                        );
                        symmetrize_lower(self.store.buffers.core_mut().exp_buf.as_mut(), n);
                        self.solve_exp_against_l(n);
                        tr = trace_product(
                            self.store.buffers.core().exp_buf.as_ref(),
                            hess.q.as_ref(),
                            n,
                        );
                        for (u, w) in hess.u_i.iter().zip(&hess.w_j) {
                            ui_wj += *u * *w;
                        }
                    }
                    let add = -0.5 * tr.to_f64() + ui_wj.to_f64();
                    out[i * n_params + j] += add;
                    if i != j {
                        out[j * n_params + i] += add;
                    }
                }
            }
            Ok::<(), GprError>(())
        })();
        self.store.buffers.core_mut().thread_scratch = thread_scratch;
        result
    }

    fn solve_exp_against_l(&mut self, n: usize) {
        let core = self.store.buffers.core_mut();
        let stack = MemStack::new(&mut core.faer_scratch);
        llt::solve::solve_in_place(
            core.k_matrix.as_ref(),
            core.exp_buf.as_mut(),
            faer_par(n),
            stack,
        );
    }

    fn fill_gradient_from_factor(
        &mut self,
        n_kernel: usize,
        n: usize,
        out: &mut [f64],
    ) -> Result<(), GprError> {
        if self.core.compiled.needs_grad_scratch() {
            self.store.buffers.core_mut().ensure_kernel_scratch(n)?;
        }
        self.store
            .buffers
            .form_gradient_w(&self.core.factor_alpha, n);
        let thread_scratch = std::mem::take(&mut self.store.buffers.core_mut().thread_scratch);
        let result = (|| {
            for (i, slot) in out.iter_mut().enumerate().take(n_kernel) {
                self.write_first_deriv(i)?;
                let inner = frobenius_lower(
                    self.store.buffers.gradient_w(),
                    self.store.buffers.core().exp_buf.as_ref(),
                    n,
                );
                *slot = -0.5 * inner.to_f64();
            }
            Ok::<(), GprError>(())
        })();
        self.store.buffers.core_mut().thread_scratch = thread_scratch;
        result?;
        let mut noise_inner = 0.0;
        let d_noise = self.core.likelihood.noise_variance();
        let w = self.store.buffers.gradient_w();
        for i in 0..n {
            noise_inner += w[(i, i)].to_f64() * d_noise;
        }
        out[n_kernel] = -0.5 * noise_inner;
        Ok(())
    }

    /// Whether a gradient writes `W` over `L` ([`crate::CholeskyBuffer::Reuse`]).
    pub(crate) fn overwrites_cholesky(&self) -> bool {
        self.store.buffers.overwrites_cholesky()
    }

    pub(crate) fn restore_cholesky_if_overwritten(&mut self) -> Result<(), GprError> {
        if self.overwrites_cholesky() {
            self.factorize_current()?;
        }
        Ok(())
    }

    pub(crate) fn commit_or_revert_optimize(
        &mut self,
        kernel_before: KernelSpec,
        likelihood_before: GaussianLikelihood,
        result: Result<OptResult, GprError>,
    ) -> Result<(), GprError> {
        match result {
            Ok(opt) => {
                if opt.params.len() != self.num_params() || !opt.value.is_finite() {
                    self.revert_theta(kernel_before, likelihood_before);
                    return Err(GprError::OptimizationNotConverged {
                        iterations: opt.iterations as usize,
                    });
                }
                Ok(())
            }
            Err(err) => {
                self.revert_theta(kernel_before, likelihood_before);
                Err(err)
            }
        }
    }

    fn revert_theta(&mut self, kernel: KernelSpec, likelihood: GaussianLikelihood) {
        self.core.kernel = kernel;
        self.core.likelihood = likelihood;
        self.core.compiled = self.core.kernel.compile_as::<P::Storage>();
        let _ = self.factorize_current();
    }

    pub(crate) fn factorize_current(&mut self) -> Result<(), GprError> {
        self.factor()
    }

    /// The one writer of the factor `α`: `A⁻¹ y` from the solve that just ran.
    fn commit_factor(&mut self) {
        let n = self.core.n;
        if self.core.factor_alpha.len() != n {
            self.core.factor_alpha.resize(n, P::Storage::from_f64(0.0));
        }
        for (i, slot) in self.core.factor_alpha.iter_mut().enumerate() {
            *slot = self.store.buffers.core().rhs[(i, 0)];
        }
    }

    pub(crate) fn fill_intervals(&self, out: &mut [Interval]) -> Result<(), GprError> {
        let n = self.num_params();
        if out.len() != n {
            return Err(GprError::LengthMismatch {
                reason: format!("expected {n} intervals, got {}", out.len()),
            });
        }
        let n_kernel = self.core.kernel.num_params();
        let mut offset = 0;
        self.core
            .kernel
            .write_intervals(&mut out[..n_kernel], &mut offset)?;
        out[n_kernel] = self.core.likelihood.bounds();
        Ok(())
    }
}

impl<O, P> FittedGpr<O, P>
where
    P: GpScalar,
{
    #[allow(clippy::result_large_err, clippy::type_complexity)] // failure returns the trainer so the caller can retry
    #[allow(clippy::result_large_err)] // failure returns the trainer so the caller can retry
    pub(crate) fn prepare(
        gpr: Gpr<O, P>,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
        y: &[f64],
    ) -> Result<Self, (Gpr<O, P>, GprError)> {
        if let Err(err) = validate_training(x, n_rows, n_cols, y) {
            return Err((gpr, err));
        }
        let mut x_buf = x.to_vec();
        let x_fitted = match gpr.x_transform.clone_box().fit(&x_buf, n_rows, n_cols) {
            Ok(t) => t,
            Err(err) => return Err((gpr, err)),
        };
        if let Err(err) = x_fitted.apply(&mut x_buf, n_rows, n_cols) {
            return Err((gpr, err));
        }
        let mut y_buf = y.to_vec();
        let y_fitted = match gpr.y_transform.clone_box().fit(&y_buf) {
            Ok(t) => t,
            Err(err) => return Err((gpr, err)),
        };
        if let Err(err) = y_fitted.transform(&mut y_buf) {
            return Err((gpr, err));
        }
        let compiled = gpr.kernel.compile_as::<P::Storage>();
        let workspace = match fit_buffers::<P>(n_rows, gpr.policies, &compiled) {
            Ok(ws) => ws,
            Err(err) => return Err((gpr, err)),
        };
        Ok(Self {
            core: GprCore {
                kernel: gpr.kernel,
                compiled,
                likelihood: gpr.likelihood,
                x_unfitted: gpr.x_transform,
                y_unfitted: gpr.y_transform,
                x_transform: x_fitted,
                y_transform: y_fitted,
                policies: gpr.policies,
                query: QueryWorkspace::new(),
                x_obs: x.to_vec(),
                y_obs: y.to_vec(),
                x: pack_points(&x_buf, n_rows, n_cols),
                y_train: y_buf,
                factor_alpha: vec![P::Storage::from_f64(0.0); n_rows],
                alpha: vec![P::Refine::from_f64(0.0); n_rows],
                x_cast: P::Storage::empty_cols(),
                y_cast: P::Storage::empty_rows(),
                n: n_rows,
                d: n_cols,
            },
            optimizer: gpr.optimizer,
            store: LltStore::new(workspace),
        })
    }

    /// Drops `L` / `α` / training data and returns a trainer with the current
    /// kernel, likelihood, transforms, optimizer, and policies.
    pub fn into_trainer(self) -> Gpr<O, P> {
        self.core.into_trainer(self.optimizer)
    }

    /// Converts this LLT factorization into an [`OnlineGpr`] for tail inserts.
    ///
    /// Writes `D[j] = L_jj²` and `L_ldlt[i,j] = L_llt[i,j] / L_jj`, then
    /// rebuilds `A = K + σn² I` on the online workspace. [`OnlineGpr::insert`]
    /// updates that LDLT in place.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::EmptyInput`] if `n` is zero, or
    /// [`GprError::CholeskyFailed`] if a diagonal of `L` is not positive.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Fixed, GaussianLikelihood, Gpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let fitted = Gpr::new(
    ///     KernelSpec::from(RbfKernel::new(1.0)?),
    ///     GaussianLikelihood::new(0.1)?,
    /// )
    /// .with_optimizer(Fixed)
    /// .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
    /// .map_err(|(_, e)| e)?;
    /// let mut online = fitted.into_online()?;
    /// online.insert(&[1.5], 0.5)?;
    /// let pred = online.predict(&[0.5], 1, 1)?;
    /// assert_eq!(pred.mean.len(), 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn into_online(self) -> Result<OnlineGpr<O, P>, GprError> {
        let n = self.core.n;
        let mut workspace = OnlineWorkspace::from_active(n)?;
        workspace.fill_ld_from_llt(self.chol_l(), n)?;
        workspace.factor_jitter = self.store.buffers.core().factor_jitter;
        OnlineWorkspace::set_f64_prefix(&mut workspace.y, &self.core.y_train);
        OnlineWorkspace::set_vector_prefix(&mut workspace.alpha, &self.core.factor_alpha);
        Ok(OnlineGpr::from_core(self.core, self.optimizer, workspace))
    }

    /// Returns the number of training points.
    pub fn n(&self) -> usize {
        self.core.n
    }

    /// Returns the feature dimension from the last successful fit.
    pub fn d(&self) -> usize {
        self.core.d
    }

    /// Returns the kernel whose hyperparameters this model owns.
    pub fn kernel(&self) -> &KernelSpec {
        &self.core.kernel
    }

    /// Returns the observation-noise model.
    pub fn likelihood(&self) -> &GaussianLikelihood {
        &self.core.likelihood
    }

    /// Returns `α = A⁻¹ y` from the last successful fit.
    pub fn alpha(&self) -> &[P::Refine] {
        &self.core.alpha
    }

    /// Returns the original training features in column-major order.
    ///
    /// Same packing as [`Gpr::fit`] / [`Gpr<Fixed>::factor`]: `n` points by
    /// `d` features. Values are on the scale passed to fit, before the input
    /// transform.
    pub fn x(&self) -> &[f64] {
        &self.core.x_obs
    }

    /// Returns the original training targets.
    ///
    /// Values are on the scale passed to fit, before the target transform.
    pub fn y(&self) -> &[f64] {
        &self.core.y_obs
    }

    /// Writes this fitted model to `dir/config.json` and `dir/model.safetensors`.
    ///
    /// Omits `L` and `α`. [`crate::persist::LoadedGpr::load`] rebuilds them
    /// by factorizing. The Cholesky buffer policy is not written; load
    /// reconstructs [`crate::CholeskyBuffer::Retain`].
    ///
    /// # Errors
    ///
    /// Returns [`GprError::PersistFailed`] when the directory cannot be
    /// created or a Custom leaf / caller transform has no persist form.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{GaussianLikelihood, Gpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let fitted = Gpr::new(
    ///     KernelSpec::from(RbfKernel::new(1.0)?),
    ///     GaussianLikelihood::new(0.1)?,
    /// )
    /// .fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
    /// .map_err(|(_, e)| e)?;
    /// let dir = std::env::temp_dir().join(format!(
    ///     "gprx-doctest-save-{}",
    ///     std::process::id()
    /// ));
    /// let _ = std::fs::remove_dir_all(&dir);
    /// fitted.save(&dir)?;
    /// let _ = std::fs::remove_dir_all(&dir);
    /// # Ok(())
    /// # }
    /// ```
    pub fn save(&self, dir: impl AsRef<std::path::Path>) -> Result<(), GprError> {
        persist::save_fitted(self, dir.as_ref(), false)
    }

    /// Writes this fitted model including the Cholesky factor `L` and `α`.
    ///
    /// `L` is stored column-major. Its dtype is `F64` when storage is `f64`
    /// and `F32` when storage is `f32`. `α` uses the predict scalar: `F32`
    /// for [`crate::SinglePrecision`], `F64` for [`crate::DoublePrecision`]
    /// and [`crate::MixedPrecision`]. [`crate::persist::LoadedGpr::load`]
    /// keeps an `f64` factor memory-mapped.
    ///
    /// # Errors
    ///
    /// Same as [`Self::save`].
    pub fn save_with_factor(&self, dir: impl AsRef<std::path::Path>) -> Result<(), GprError> {
        persist::save_fitted(self, dir.as_ref(), true)
    }

    /// Replaces the optimizer used by a later [`Self::refit`].
    ///
    /// Does not write a solver into a persist directory. A model loaded as
    /// [`crate::persist::LoadedGpr`] is [`Fixed`]; call this before `refit`
    /// to search again. Incremental rebuilds follow the same rule as
    /// [`Gpr::with_optimizer`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::persist::{LoadedGpr, PersistRegistry};
    /// use gprx::{GaussianLikelihood, Gpr, Lbfgs};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let fitted = Gpr::new(
    ///     KernelSpec::from(RbfKernel::new(1.0)?),
    ///     GaussianLikelihood::new(0.1)?,
    /// )
    /// .fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
    /// .map_err(|(_, e)| e)?;
    /// let dir = std::env::temp_dir().join(format!(
    ///     "gprx-doctest-refit-{}",
    ///     std::process::id()
    /// ));
    /// let _ = std::fs::remove_dir_all(&dir);
    /// fitted.save(&dir)?;
    /// let LoadedGpr::Double(model) = LoadedGpr::load(&dir, &PersistRegistry::new())? else {
    ///     return Ok(());
    /// };
    /// let mut model = model.with_optimizer(Lbfgs::new());
    /// model.refit()?;
    /// let _ = std::fs::remove_dir_all(&dir);
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_optimizer<O2>(self, optimizer: O2) -> FittedGpr<O2, P> {
        FittedGpr {
            core: self.core,
            optimizer,
            store: self.store,
        }
    }

    /// Diagonal jitter the current factor was built with.
    pub(crate) fn factor_jitter(&self) -> f64 {
        self.store.buffers.core().factor_jitter
    }

    pub(crate) fn policies(&self) -> Policies {
        self.core.policies
    }

    /// Returns the distance-cache policy carried from the trainer.
    pub fn distance_cache_policy(&self) -> crate::DistanceCachePolicy {
        self.core.policies.distance_cache
    }

    /// Returns the Cholesky buffer policy carried from the trainer.
    pub fn cholesky_buffer(&self) -> crate::CholeskyBuffer {
        self.core.policies.cholesky_buffer
    }

    /// Returns the kernel `exp` used by fit and predict.
    pub fn math(&self) -> crate::KernelExp {
        self.core.policies.math
    }

    pub(crate) fn x_unfitted(&self) -> &dyn UnfittedTransform {
        self.core.x_unfitted.as_ref()
    }

    pub(crate) fn y_unfitted(&self) -> &dyn UnfittedTarget {
        self.core.y_unfitted.as_ref()
    }

    pub(crate) fn x_transform(&self) -> &dyn Transform {
        self.core.x_transform.as_ref()
    }

    pub(crate) fn y_transform(&self) -> &dyn TargetTransform {
        self.core.y_transform.as_ref()
    }

    pub(crate) fn chol_l(&self) -> MatRef<'_, P::Storage> {
        self.store.l()
    }

    pub(crate) fn factor(&self) -> StoredFactor<'_, P::Storage> {
        StoredFactor::Llt(self.chol_l())
    }

    /// Lends the core and the LLT buffers to the fit code.
    pub(crate) fn fit_view(&mut self) -> ExactFit<'_, P> {
        ExactFit {
            core: &mut self.core,
            store: &mut self.store,
        }
    }

    /// Returns the negative log marginal likelihood of the last successful fit.
    ///
    /// Evaluates `½ yᵀ A⁻¹ y + ½ log|A| + (n/2) log(2π)` from the stored
    /// `α` and the Cholesky factor `L` in the workspace, using
    /// `log|A| = 2 Σ log(L_ii)`. `y` is the target after the target
    /// transform.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Gpr, GaussianLikelihood};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    /// let likelihood = GaussianLikelihood::new(0.1)?;
    /// let gpr = Gpr::new(kernel, likelihood);
    /// let fitted = gpr.fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0]).map_err(|(_, e)| e)?;
    /// let nlml = fitted.neg_log_marginal_likelihood()?;
    /// assert!(nlml.is_finite());
    /// # Ok(())
    /// # }
    /// ```
    pub fn neg_log_marginal_likelihood(&self) -> Result<f64, GprError> {
        Ok(self
            .core
            .neg_log_marginal_likelihood(self.factor(), &self.core.factor_alpha))
    }

    /// Returns the concatenated kernel and likelihood parameter count.
    pub fn num_params(&self) -> usize {
        self.core.num_params()
    }

    /// Writes kernel `θ` then likelihood `θ` into `out`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::LengthMismatch`] if `out` is the wrong length
    /// or a custom leaf rejects the write.
    pub fn get_params(&self, out: &mut [f64]) -> Result<(), GprError> {
        self.core.get_params(out)
    }

    /// Sets kernel then likelihood `θ` and rebuilds `L` / `α`.
    ///
    /// `params` is kernel parameters followed by the likelihood parameter,
    /// matching [`Self::get_params`]. Transforms and training `X` / `y` are
    /// not changed. [`Self::kernel`] stays a shared reference; this is the
    /// write path. After success, [`Gpr<Fixed>::factor`] on
    /// [`Self::into_trainer`] with [`Self::x`] / [`Self::y`] rebuilds the
    /// same factorization from the stored observations.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::LengthMismatch`] if `params` is the wrong
    /// length, [`GprError::InvalidNoiseVariance`] if the likelihood `θ` is
    /// invalid, or [`GprError::CholeskyFailed`] if `A` cannot be factored.
    /// Kernel and likelihood `θ` are committed together only after `A`
    /// factors. A rejected slice or a Cholesky failure leaves stored `θ`
    /// and `L` / `α` unchanged.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Fixed, GaussianLikelihood, Gpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let mut fitted = Gpr::new(
    ///     KernelSpec::from(RbfKernel::new(1.0)?),
    ///     GaussianLikelihood::new(0.1)?,
    /// )
    /// .fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
    /// .map_err(|(_, e)| e)?;
    /// let mut params = [0.0; 2];
    /// fitted.get_params(&mut params)?;
    /// params[0] = 0.5_f64.ln();
    /// fitted.set_params(&params)?;
    /// let x = fitted.x().to_vec();
    /// let y = fitted.y().to_vec();
    /// let n = fitted.n();
    /// let d = fitted.d();
    /// let _fitted = fitted
    ///     .into_trainer()
    ///     .with_optimizer(Fixed)
    ///     .factor(&x, n, d, &y)
    ///     .map_err(|(_, e)| e)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_params(&mut self, params: &[f64]) -> Result<(), GprError> {
        self.fit_view().set_params(params)
    }

    #[cfg(test)]
    pub(crate) fn objective(&mut self) -> GprObjective<'_, P> {
        GprObjective::new(self.fit_view())
    }

    /// Sets kernel and likelihood `θ`, rebuilds `L` / `α` / `W`, and writes `∂L/∂θ`.
    ///
    /// `params` and `out` are kernel parameters followed by the likelihood
    /// parameter. One Cholesky produces `L` and `α`; `W = ααᵀ - A⁻¹` is
    /// formed from that factor. [`crate::CholeskyBuffer::Retain`] keeps `W` in a dedicated
    /// buffer. [`crate::CholeskyBuffer::Reuse`] writes `W` over `L` and this method
    /// refactors afterwards so [`Self::predict`] still sees `L`. Kernel
    /// `∂A/∂θ` goes through `exp_buf`. Product trees also use
    /// `kernel_scratch`. The returned value is the same as
    /// [`Self::neg_log_marginal_likelihood`] after a successful call.
    ///
    /// Training `X` / `y` come from [`Gpr::fit`]. Transforms
    /// are not re-fit.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::LengthMismatch`] if a slice length is wrong,
    /// [`GprError::InvalidNoiseVariance`] if the likelihood `θ` is invalid,
    /// [`GprError::CholeskyFailed`] if `A` cannot be factored, or
    /// [`GprError::UnsupportedKernelOperation`] if the compiled tree cannot
    /// evaluate at this `θ`. Distance-mode and points-mode product trees are
    /// supported. Kernel and likelihood `θ` are committed together only after
    /// `A` factors. A rejected slice or a Cholesky failure leaves stored `θ`
    /// unchanged.
    /// Cholesky failure restores `L` and `α` at the previous `θ` so this
    /// value stays a usable [`FittedGpr`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Gpr, GaussianLikelihood};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    /// let likelihood = GaussianLikelihood::new(0.1)?;
    /// let gpr = Gpr::new(kernel, likelihood);
    /// let mut fitted = gpr.fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0]).map_err(|(_, e)| e)?;
    /// let mut params = [0.0; 2];
    /// fitted.get_params(&mut params)?;
    /// let mut grad = [0.0; 2];
    /// let nlml = fitted.value_and_gradient_into(&params, &mut grad)?;
    /// assert!(nlml.is_finite());
    /// # Ok(())
    /// # }
    /// ```
    pub fn value_and_gradient_into(
        &mut self,
        params: &[f64],
        out: &mut [f64],
    ) -> Result<f64, GprError> {
        self.fit_view().value_and_gradient_into(params, out)
    }

    /// Writes the analytic NLML Hessian (row-major `p×p`) at `params`.
    ///
    /// `params` is kernel `θ` followed by likelihood `θ`. After a successful
    /// call the stored kernel and likelihood match `params`. [`crate::CholeskyBuffer::Reuse`]
    /// rebuilds `L` before return, matching [`Self::value_and_gradient_into`].
    ///
    /// # Errors
    ///
    /// Returns [`GprError`] when a slice length is wrong, `params` is
    /// rejected, or the Gram matrix does not factor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Gpr, GaussianLikelihood};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    /// let likelihood = GaussianLikelihood::new(0.1)?;
    /// let mut fitted = Gpr::new(kernel, likelihood)
    ///     .fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
    ///     .map_err(|(_, e)| e)?;
    /// let mut params = [0.0; 2];
    /// fitted.get_params(&mut params)?;
    /// let mut hess = [0.0; 4];
    /// fitted.hessian_into(&params, &mut hess)?;
    /// assert!(hess.iter().all(|h| h.is_finite()));
    /// # Ok(())
    /// # }
    /// ```
    pub fn hessian_into(&mut self, params: &[f64], out: &mut [f64]) -> Result<(), GprError> {
        self.fit_view().hessian_into(params, out)
    }

    /// Predicts at `xs` with [`PredictOptions::default`] (observation variance).
    ///
    /// `xs` is column-major with `n_rows` query points and `n_cols` features.
    /// Allocates query buffers for this call. Reuse [`Self::predict_into`]
    /// after a warmup call for a zero-allocation path. See [`Gpr`] for a
    /// complete fit→predict example.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::DimensionMismatch`] if `n_cols` differs from the
    /// training features, [`GprError::EmptyInput`] if a dimension is zero, or
    /// [`GprError::LengthMismatch`] / [`GprError::NonFiniteInput`] for a
    /// badly packed or non-finite `xs`.
    pub fn predict(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
    ) -> Result<Prediction<P::Refine>, GprError> {
        self.predict_with(xs, n_rows, n_cols, PredictOptions::default())
    }

    /// Writes [`Self::predict`] into `out`, reusing `mean` / `variance`
    /// capacity when the query length matches.
    ///
    /// # Errors
    ///
    /// Same as [`Self::predict`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Gpr, GaussianLikelihood, Prediction};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    /// let likelihood = GaussianLikelihood::new(0.1)?;
    /// let gpr = Gpr::new(kernel, likelihood);
    /// let mut fitted = gpr.fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0]).map_err(|(_, e)| e)?;
    /// let mut pred = Prediction::default();
    /// fitted.predict_into(&[0.5], 1, 1, &mut pred)?;
    /// assert_eq!(pred.mean.len(), 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn predict_into(
        &mut self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        out: &mut Prediction<P::Refine>,
    ) -> Result<(), GprError> {
        self.predict_with_into(xs, n_rows, n_cols, PredictOptions::default(), out)
    }

    /// Predicts at `xs` with an explicit variance kind.
    ///
    /// Latent variance is `k(x*, x*) - ‖L⁻¹ k_*‖²`. Observation variance adds
    /// `σn²` in the transformed space, then both mean and variance are mapped
    /// back by the target transform.
    ///
    /// # Errors
    ///
    /// Same as [`Self::predict`].
    pub fn predict_with(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        options: PredictOptions,
    ) -> Result<Prediction<P::Refine>, GprError> {
        let mut out = Prediction::default();
        self.core.write_prediction(
            self.factor(),
            &self.core.alpha,
            xs,
            n_rows,
            n_cols,
            options,
            &mut out,
        )?;
        Ok(out)
    }

    /// Writes [`Self::predict_with`] into `out`, reusing `mean` / `variance`
    /// capacity when the query length matches.
    ///
    /// # Errors
    ///
    /// Same as [`Self::predict`].
    pub fn predict_with_into(
        &mut self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        options: PredictOptions,
        out: &mut Prediction<P::Refine>,
    ) -> Result<(), GprError> {
        let (l, thread_scratch) = self.store.l_and_thread_scratch();
        let factor = StoredFactor::Llt(l);
        self.core
            .predict_with_into(factor, thread_scratch, xs, n_rows, n_cols, options, out)
    }

    /// Returns the predictive mean and query–query covariance at `xs`.
    ///
    /// Default [`PredictOptions`] uses [`crate::VarianceKind::Observation`]: `σn²`
    /// is added on the diagonal in the transformed space. The diagonal
    /// matches [`Self::predict`] for the same query. This path allocates
    /// an `m × m` matrix; the default [`Self::predict`] stays diagonal-only.
    ///
    /// # Errors
    ///
    /// Same as [`Self::predict`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Gpr, GaussianLikelihood};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    /// let likelihood = GaussianLikelihood::new(0.1)?;
    /// let gpr = Gpr::new(kernel, likelihood);
    /// let fitted = gpr.fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0]).map_err(|(_, e)| e)?;
    /// let cov = fitted.predict_covariance(&[0.25, 0.75], 2, 1)?;
    /// assert_eq!(cov.mean.len(), 2);
    /// assert_eq!(cov.covariance.len(), 4);
    /// # Ok(())
    /// # }
    /// ```
    pub fn predict_covariance(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
    ) -> Result<PredictiveCovariance<P::Refine>, GprError> {
        self.predict_covariance_with(xs, n_rows, n_cols, PredictOptions::default())
    }

    /// Returns query–query covariance with an explicit variance kind.
    ///
    /// Posterior covariance is `K** − VᵀV` with `V = L⁻¹ K_*`. Latent
    /// diagonals are clipped at 0. Observation adds `σn²` on the diagonal
    /// in the transformed space, then the target transform scales the
    /// whole matrix.
    ///
    /// # Errors
    ///
    /// Same as [`Self::predict`].
    pub fn predict_covariance_with(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        options: PredictOptions,
    ) -> Result<PredictiveCovariance<P::Refine>, GprError> {
        self.core
            .write_covariance(self.factor(), &self.core.alpha, xs, n_rows, n_cols, options)
    }

    /// Draws posterior samples at `xs` from [`Self::predict_covariance`].
    ///
    /// Each column of the returned column-major `m × n_draws` matrix is
    /// `μ + L z` with `z ∼ N(0, I)` and `L` the Cholesky factor of the
    /// posterior covariance. `seed` is the crate [`rand::rngs::SmallRng`]
    /// start state. Zero draws returns an empty vector after the covariance
    /// is formed.
    ///
    /// # Errors
    ///
    /// Same as [`Self::predict`], plus [`GprError::CholeskyFailed`] with
    /// [`CholeskyStage::Predict`] if the posterior covariance cannot be
    /// factored after [`crate::JitterPolicy`] retries.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Gpr, GaussianLikelihood};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    /// let likelihood = GaussianLikelihood::new(0.1)?;
    /// let gpr = Gpr::new(kernel, likelihood);
    /// let fitted = gpr.fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0]).map_err(|(_, e)| e)?;
    /// let draws = fitted.sample(&[0.25, 0.75], 2, 1, 4, 1)?;
    /// assert_eq!(draws.len(), 8);
    /// # Ok(())
    /// # }
    /// ```
    pub fn sample(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        n_draws: usize,
        seed: u64,
    ) -> Result<Vec<P::Refine>, GprError> {
        self.sample_with(xs, n_rows, n_cols, PredictOptions::default(), n_draws, seed)
    }

    /// Draws posterior samples with an explicit variance kind.
    ///
    /// # Errors
    ///
    /// Same as [`Self::sample`].
    pub fn sample_with(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        options: PredictOptions,
        n_draws: usize,
        seed: u64,
    ) -> Result<Vec<P::Refine>, GprError> {
        self.core.sample_with(
            self.factor(),
            &self.core.alpha,
            xs,
            n_rows,
            n_cols,
            options,
            n_draws,
            seed,
        )
    }

    /// Returns leave-one-out mean and observation variance at every training
    /// point.
    ///
    /// Uses the GPML identities `μ_i = y_i - α_i / Q_ii` and
    /// `σ_i² = 1 / Q_ii` with `Q = A⁻¹` and `A = K + σn² I`. This is
    /// `p(y_i | X, y_{-i}, θ)`, not a query at a new `x*`. Mean and
    /// variance are inverse-transformed like [`Self::predict`].
    ///
    /// # Errors
    ///
    /// Returns [`GprError::NonPositiveDefiniteMatrix`] if a diagonal of `A⁻¹`
    /// is not positive and finite.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Gpr, GaussianLikelihood};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    /// let likelihood = GaussianLikelihood::new(0.1)?;
    /// let gpr = Gpr::new(kernel, likelihood);
    /// let fitted = gpr.fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0]).map_err(|(_, e)| e)?;
    /// let loo = fitted.loo_predict()?;
    /// assert_eq!(loo.mean.len(), 2);
    /// # Ok(())
    /// # }
    /// ```
    pub fn loo_predict(&self) -> Result<Prediction<P::Refine>, GprError> {
        self.loo_predict_with(PredictOptions::default())
    }

    /// Returns leave-one-out mean and variance with an explicit variance kind.
    ///
    /// Observation variance is `1 / Q_ii`. Latent variance is
    /// `max(0, 1 / Q_ii - σn²)` in the transformed space, then both mean
    /// and variance are mapped back by the target transform.
    ///
    /// # Errors
    ///
    /// Same as [`Self::loo_predict`].
    pub fn loo_predict_with(
        &self,
        options: PredictOptions,
    ) -> Result<Prediction<P::Refine>, GprError> {
        self.core
            .loo_predict_with(self.factor(), &self.core.alpha, options)
    }
}

impl<O, P> FittedGpr<O, P>
where
    P: GpScalar,
    O: for<'a> Optimizer<GprObjective<'a, P>>,
{
    /// Re-runs the stored optimizer on the stored training data from the current `θ`.
    ///
    /// This is the same `O` that [`Gpr::with_optimizer`] installed. Transforms
    /// are not re-fit. `n` and `d` stay the same.
    ///
    /// # Errors
    ///
    /// Same as [`Gpr::fit`].
    pub fn refit(&mut self) -> Result<(), GprError> {
        let mut view = ExactFit {
            core: &mut self.core,
            store: &mut self.store,
        };
        view.optimize(&self.optimizer)
    }
}

impl<P> FittedGpr<Fixed, P>
where
    P: GpScalar,
{
    pub(crate) fn into_online_preserving_factor(self) -> Result<OnlineGpr<Fixed, P>, GprError> {
        let n = self.core.n;
        let mut workspace = OnlineWorkspace::<P::Storage>::from_active(n)?;
        workspace.copy_ld_from(self.chol_l(), n)?;
        workspace.factor_jitter = self.store.buffers.core().factor_jitter;
        OnlineWorkspace::set_f64_prefix(&mut workspace.y, &self.core.y_train);
        OnlineWorkspace::set_vector_prefix(&mut workspace.alpha, &self.core.factor_alpha);
        Ok(OnlineGpr::from_core(self.core, self.optimizer, workspace))
    }

    pub(crate) fn from_persisted(mut parts: PersistedModel<P>) -> Result<Self, GprError> {
        let n = parts.y_obs.len();
        if n == 0 {
            return Err(GprError::EmptyInput);
        }
        if parts.x_obs.len() % n != 0 {
            return Err(persist::persist_err("persisted x length is not n * d"));
        }
        let d = parts.x_obs.len() / n;
        if parts.alpha.len() != n {
            return Err(persist::persist_err(format!(
                "alpha has {} values, expected n = {n}",
                parts.alpha.len()
            )));
        }
        let mut x_buf = parts.x_obs.clone();
        parts.x_transform.apply(&mut x_buf, n, d)?;
        let mut y_buf = parts.y_obs.clone();
        parts.y_transform.transform(&mut y_buf)?;
        let compiled = parts.kernel.compile_as::<P::Storage>();
        let mut workspace = fit_buffers::<P>(n, parts.policies, &compiled)?;
        workspace.core_mut().factor_jitter = parts.factor_jitter;
        if let Some(l) = parts.owned_l.take() {
            let mut dest = workspace.core_mut().k_matrix.as_mut();
            for col in 0..n {
                for row in 0..n {
                    dest[(row, col)] = l[(row, col)];
                }
            }
        }
        let factor_alpha = storage_alpha_from_saved::<P>(
            workspace.core().k_matrix.as_ref(),
            &y_buf,
            &parts.alpha,
        )?;
        Ok(Self {
            core: GprCore {
                kernel: parts.kernel,
                compiled,
                likelihood: parts.likelihood,
                x_unfitted: parts.x_unfitted,
                y_unfitted: parts.y_unfitted,
                x_transform: parts.x_transform,
                y_transform: parts.y_transform,
                policies: parts.policies,
                query: QueryWorkspace::new(),
                x_obs: parts.x_obs,
                y_obs: parts.y_obs,
                x: pack_points(&x_buf, n, d),
                y_train: y_buf,
                factor_alpha,
                alpha: parts.alpha,
                x_cast: P::Storage::empty_cols(),
                y_cast: P::Storage::empty_rows(),
                n,
                d,
            },
            optimizer: Fixed,
            store: LltStore::with_mapped(workspace, parts.mapped),
        })
    }

    /// Rebuilds `L` and `α` at the current `θ` without a search.
    ///
    /// Transforms are not re-fit. `n` and `d` stay the same.
    ///
    /// # Errors
    ///
    /// Same as [`Gpr<Fixed>::factor`].
    pub fn refit(&mut self) -> Result<(), GprError> {
        self.fit_view().refactor()
    }
}

fn storage_alpha_from_saved<P: GpScalar>(
    l: MatRef<'_, P::Storage>,
    y: &[f64],
    saved: &[P::Refine],
) -> Result<Vec<P::Storage>, GprError> {
    let mixed = P::REFINES_IN_F64;
    if !mixed {
        return Ok(saved
            .iter()
            .map(|weight| P::Storage::from_f64(weight.to_f64()))
            .collect());
    }
    let n = y.len();
    let mut rhs = Mat::<P::Storage>::zeros(n, 1);
    for (i, &yi) in y.iter().enumerate() {
        rhs[(i, 0)] = P::Storage::from_f64(yi);
    }
    let par = faer_par_dims(n, 1);
    let req = llt::solve::solve_in_place_scratch::<P::Storage>(n, 1, par);
    let mut scratch = MemBuffer::new(req);
    solve_llt_in_place(l, rhs.as_mut(), &mut scratch);
    Ok((0..n).map(|i| rhs[(i, 0)]).collect())
}

fn require_change_indices(indices: &[usize], n_params: usize) -> Result<(), GprError> {
    if indices.is_empty() {
        return Err(GprError::IndexOutOfRange {
            reason: "change indices must not be empty".to_owned(),
        });
    }
    for (k, &i) in indices.iter().enumerate() {
        if i >= n_params {
            return Err(GprError::IndexOutOfRange {
                reason: format!("change index {i} is out of range (n_params={n_params})"),
            });
        }
        if indices[..k].contains(&i) {
            return Err(GprError::IndexOutOfRange {
                reason: format!("change index {i} is duplicated"),
            });
        }
    }
    Ok(())
}

fn zero_and_maybe_noise<T: KernelScalar>(
    mut out: MatMut<'_, T>,
    n: usize,
    noise_diag: bool,
    noise: f64,
) {
    let noise_s = T::from_f64(noise);
    let zero = T::from_f64(0.0);
    for col in 0..n {
        for row in col..n {
            out[(row, col)] = if noise_diag && row == col {
                noise_s
            } else {
                zero
            };
        }
    }
}

fn kinv_from_w<T: KernelScalar>(alpha: &[T], w: MatRef<'_, T>, row: usize, col: usize) -> T {
    let (r, c) = if row >= col { (row, col) } else { (col, row) };
    alpha[row] * alpha[col] - w[(r, c)]
}

fn ki_sym<T: Copy>(ki: MatRef<'_, T>, row: usize, col: usize) -> T {
    if row >= col {
        ki[(row, col)]
    } else {
        ki[(col, row)]
    }
}

fn trace_ki_kinv2<T: KernelScalar>(
    ki: MatRef<'_, T>,
    w: MatRef<'_, T>,
    alpha: &[T],
    n: usize,
) -> T {
    let mut tr = T::from_f64(0.0);
    for c in 0..n {
        for b in 0..n {
            let mut m_bc = T::from_f64(0.0);
            for k in 0..n {
                m_bc += ki_sym(ki, b, k) * kinv_from_w(alpha, w, k, c);
            }
            tr += kinv_from_w(alpha, w, b, c) * m_bc;
        }
    }
    tr
}
