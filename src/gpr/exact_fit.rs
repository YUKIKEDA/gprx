//! [`ExactFit`]: every hyperparameter write, gradient, and Hessian of an Exact GPR.

use faer::{Mat, MatMut, MatRef};

use crate::error::{CholeskyStage, GprError};
use crate::gpr::GprObjective;
use crate::kernel::ScalarOps;
use crate::kernel::{CompiledKernel, KernelScalar, KernelSpec, Triangle};
use crate::likelihood::GaussianLikelihood;
use crate::linalg::{frobenius_lower, gemv_sym_lower, solve_lower};
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

    fn fill_hessian_from_factor(
        &mut self,
        n_kernel: usize,
        n: usize,
        out: &mut [f64],
    ) -> Result<(), GprError> {
        self.store.buffers.core_mut().ensure_kernel_scratch(n)?;
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
        hess.ensure(n, n_params);
        let first = self.add_first_order(n_kernel, n, noise, out, &mut hess);
        self.store.buffers.core_mut().hessian = hess;
        first
    }

    /// Adds the first-order part of the Hessian of every pair:
    /// `−½ tr(A⁻¹ A_i A⁻¹ A_j) + αᵀ A_i A⁻¹ A_j α`, with `A_i = ∂A/∂θ_i`
    /// (the kernel's `∂K/∂θ_i`, or `σn² I` for the noise).
    ///
    /// With `A = L Lᵀ`, `S_i = L⁻¹ A_i L⁻ᵀ` and `v_i = L⁻¹ A_i α` give
    /// `tr(A⁻¹ A_i A⁻¹ A_j) = ⟨S_i, S_j⟩_F` and `αᵀ A_i A⁻¹ A_j α = v_iᵀ v_j`.
    /// Each `S_i` takes two triangular solves, so the pairs cost
    /// `O(p n³ + p² n²)` and `hess` keeps `p` matrices of `n × n`.
    fn add_first_order(
        &mut self,
        n_kernel: usize,
        n: usize,
        noise: f64,
        out: &mut [f64],
        hess: &mut HessianScratch<P::Storage>,
    ) -> Result<(), GprError> {
        if !self.store.buffers.has_dedicated_w() {
            // `W` was written over `L` for the second-order part.
            self.factorize_current()?;
        }
        let n_params = n_kernel + 1;
        let thread_scratch = std::mem::take(&mut self.store.buffers.core_mut().thread_scratch);
        let halves = (|| {
            for i in 0..n_params {
                if i < n_kernel {
                    self.write_first_deriv(i)?;
                } else {
                    zero_and_maybe_noise(
                        self.store.buffers.core_mut().exp_buf.as_mut(),
                        n,
                        true,
                        noise,
                    );
                }
                let l = self.store.buffers.core().k_matrix.as_ref();
                let d_a = self.store.buffers.core().exp_buf.as_ref();
                gemv_sym_lower(d_a, &self.core.factor_alpha, &mut hess.u, n);
                let mut v_i = hess.v.as_mut().col_mut(i);
                for (row, value) in hess.u.iter().enumerate() {
                    v_i[row] = *value;
                }
                solve_lower(l, hess.v.as_mut().subcols_mut(i, 1));
                let s_i = &mut hess.s[i];
                for col in 0..n {
                    for row in col..n {
                        s_i[(row, col)] = d_a[(row, col)];
                        s_i[(col, row)] = d_a[(row, col)];
                    }
                }
                // `L⁻¹ A_i`, then `L⁻¹ (L⁻¹ A_i)ᵀ = L⁻¹ A_i L⁻ᵀ`.
                solve_lower(l, s_i.as_mut());
                transpose_square_in_place(s_i.as_mut(), n);
                solve_lower(l, s_i.as_mut());
            }
            Ok::<(), GprError>(())
        })();
        self.store.buffers.core_mut().thread_scratch = thread_scratch;
        halves?;
        for i in 0..n_params {
            for j in 0..=i {
                let tr = frobenius_full(hess.s[i].as_ref(), hess.s[j].as_ref(), n);
                let mut v_dot = 0.0;
                for row in 0..n {
                    v_dot += hess.v[(row, i)].to_f64() * hess.v[(row, j)].to_f64();
                }
                let add = -0.5 * tr + v_dot;
                out[i * n_params + j] += add;
                if i != j {
                    out[j * n_params + i] += add;
                }
            }
        }
        Ok(())
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

/// `⟨a, b⟩_F` over the full `n × n` matrices, accumulated in `f64`.
fn frobenius_full<T: KernelScalar>(a: MatRef<'_, T>, b: MatRef<'_, T>, n: usize) -> f64 {
    let mut sum = 0.0;
    for col in 0..n {
        for row in 0..n {
            sum += a[(row, col)].to_f64() * b[(row, col)].to_f64();
        }
    }
    sum
}

fn transpose_square_in_place<T: KernelScalar>(mut a: MatMut<'_, T>, n: usize) {
    for col in 0..n {
        for row in col + 1..n {
            let upper = a[(col, row)];
            a[(col, row)] = a[(row, col)];
            a[(row, col)] = upper;
        }
    }
}
