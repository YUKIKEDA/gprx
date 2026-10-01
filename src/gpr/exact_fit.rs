//! [`ExactFit`]: every hyperparameter write, gradient, and Hessian of an Exact GPR.

use dyn_stack::MemStack;
use faer::linalg::cholesky::llt;
use faer::{Mat, MatMut, MatRef};

use crate::error::{CholeskyStage, GprError};
use crate::gpr::GprObjective;
use crate::kernel::ScalarOps;
use crate::kernel::{CompiledKernel, KernelScalar, KernelSpec, Triangle};
use crate::likelihood::GaussianLikelihood;
use crate::linalg::{
    faer_par, frobenius_lower, gemv_full, gemv_sym_lower, symmetrize_lower, trace_product,
};
use crate::optimizer::{OptResult, Optimizer};
use crate::param::Interval;
use crate::precision::{GpScalar, StoredFactor};
use crate::workspace::{FitWorkspace, HessianScratch, WorkspaceCore};

use super::factor::{
    FactorPolicy, apply_compiled_to, factor_train_with_policy, factor_written_k_with_policy,
    fill_cached_inputs, neg_mll_from_factor,
};
use super::{FitBuffers, GprCore, LltStore, Policies};
use crate::policy::{DistanceCachePolicy, with_kernel_exp};

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

    /// Joint value, gradient, and Hessian during `fit` from one
    /// factorization of `A`.
    ///
    /// The gradient forms `W` from that factor. A dedicated `W` keeps `L`, so
    /// the Hessian reads the same factor; [`crate::CholeskyBuffer::Reuse`]
    /// wrote `W` over `L`, so `L` is rebuilt at the same `θ` first.
    pub(crate) fn value_gradient_hessian_into_fit(
        &mut self,
        params: &[f64],
        grad: &mut [f64],
        hess: &mut [f64],
    ) -> Result<f64, GprError> {
        let n_params = self.num_params();
        crate::data::require_count(hess.len(), n_params * n_params, "parameters")?;
        let nlml = self.value_and_gradient_into_fit(params, grad)?;
        self.restore_cholesky_if_overwritten()?;
        let n_kernel = self.core.kernel.num_params();
        let n = self.core.n;
        self.fill_hessian_from_factor(n_kernel, n, hess)?;
        Ok(nlml)
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
                // The model holds the last θ the optimizer evaluated, which is
                // not the result after restarts, a rejected annealing step, or
                // a caller optimizer that searched past its best point.
                if !self.holds_params(&opt.params)? {
                    if let Err(err) = self.factor_at(&opt.params) {
                        self.revert_theta(kernel_before, likelihood_before);
                        return Err(err);
                    }
                }
                Ok(())
            }
            Err(err) => {
                self.revert_theta(kernel_before, likelihood_before);
                Err(err)
            }
        }
    }

    /// Whether the stored `θ` is bit-for-bit `params`.
    fn holds_params(&mut self, params: &[f64]) -> Result<bool, GprError> {
        let current = &mut self.store.buffers.core_mut().theta;
        current.resize(params.len(), 0.0);
        self.core.get_params(current)?;
        Ok(current
            .iter()
            .zip(params)
            .all(|(a, b)| a.to_bits() == b.to_bits()))
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
