//! [`FittedGpr`] factorization, prediction, and refit.

use std::marker::PhantomData;

use dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt;
use faer::{Mat, MatMut, MatRef};

use crate::error::{CholeskyStage, GprError};
use crate::kernel::{
    CompiledKernel, CoordMode, KernelSpec, MixedKernelViews, Triangle, fill_squared_euclidean,
    fill_squared_euclidean_cross,
};
use crate::likelihood::GaussianLikelihood;
use crate::objective::GprObjective;
use crate::online::OnlineWorkspace;
use crate::optimizer::{Fixed, FullRecompute, OptResult, Optimizer, PoleRecompute};
use crate::param::Interval;
use crate::persist::{self, PersistedModel};
use crate::transform::{TargetTransform, Transform, UnfittedTarget, UnfittedTransform};
use crate::workspace::{
    FitWorkspace, QueryWorkspace, empty_thread_scratch, faer_par, faer_par_dims,
};
use crate::{PredictOptions, Prediction, PredictiveCovariance, VarianceKind};

use super::super::online::OnlineGpr;

use super::super::factor::{
    FactorPolicy, apply_compiled_to, cholesky_lower_with_policy, factor_train_with_policy,
    factor_written_k_with_policy, frobenius_lower, gemv_full, gemv_sym_lower, inv_diag_from_chol_l,
    neg_mll_from_factor, pack_points, pack_points_into, require_param_len, symmetrize_lower,
    trace_product, validate_query, validate_training, write_kernel_grad,
    write_kernel_grad_from_coords, write_kernel_hess, write_kernel_hess_from_coords, write_params,
};
use super::{AllocWorkspace, DistanceCacheSlot, FitBuffers, JitterPolicy, RetainCholesky};
use super::{FittedGpr, Gpr};

#[allow(private_bounds)] // `DistanceCacheSlot` is crate-private; factorization reads it.
impl<O, S, C: DistanceCacheSlot, B: AllocWorkspace> FittedGpr<O, S, C, B> {
    #[allow(clippy::result_large_err, clippy::type_complexity)] // failure returns the trainer so the caller can retry
    pub(crate) fn prepare(
        gpr: Gpr<O, S, C, B>,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
        y: &[f64],
    ) -> Result<Self, (Gpr<O, S, C, B>, GprError)> {
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
        let mut workspace = match FitBuffers::<C, B>::new(n_rows) {
            Ok(ws) => ws,
            Err(err) => return Err((gpr, err)),
        };
        let compiled = gpr.kernel.compile();
        if C::CACHES_DISTANCES && compiled.needs_ard_sq_diff() {
            if let Err(err) = workspace.ensure_ard_if_cached(n_rows, n_cols) {
                return Err((gpr, err));
            }
        }
        Ok(Self {
            kernel: gpr.kernel,
            compiled,
            likelihood: gpr.likelihood,
            x_unfitted: gpr.x_transform,
            y_unfitted: gpr.y_transform,
            x_transform: x_fitted,
            y_transform: y_fitted,
            optimizer: gpr.optimizer,
            distance_cache: gpr.distance_cache,
            jitter_policy: gpr.jitter_policy,
            workspace,
            query: QueryWorkspace::new(),
            x_obs: x.to_vec(),
            y_obs: y.to_vec(),
            x: pack_points(&x_buf, n_rows, n_cols),
            y_train: y_buf,
            alpha: vec![0.0; n_rows],
            n: n_rows,
            d: n_cols,
            mapped_factor: None,
            _recompute: PhantomData,
        })
    }

    /// Drops `L` / `α` / training data and returns a trainer with the current
    /// kernel, likelihood, transforms, optimizer, distance-cache slot, and
    /// jitter policy.
    pub fn into_trainer(self) -> Gpr<O, S, C, B> {
        Gpr::from_owned(
            self.kernel,
            self.likelihood,
            self.x_unfitted,
            self.y_unfitted,
            self.optimizer,
            self.distance_cache,
            self.jitter_policy,
        )
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
    pub fn into_online(self) -> Result<OnlineGpr<O, S, C, B>, GprError> {
        let n = self.n;
        let mut workspace = OnlineWorkspace::from_active(n)?;
        workspace.fill_ld_from_llt(self.chol_l(), n)?;
        OnlineWorkspace::set_vector_prefix(&mut workspace.y, &self.y_train);
        OnlineWorkspace::set_vector_prefix(&mut workspace.alpha, &self.alpha);
        Ok(OnlineGpr::from_parts(
            self.kernel,
            self.compiled,
            self.likelihood,
            self.x_unfitted,
            self.y_unfitted,
            self.x_transform,
            self.y_transform,
            self.optimizer,
            self.distance_cache,
            self.jitter_policy,
            workspace,
            self.query,
            self.x_obs,
            self.y_obs,
            self.x,
            self.y_train,
            self.alpha,
            self.n,
            self.d,
        ))
    }

    pub(crate) fn from_online_snapshot(online: &OnlineGpr<O, S, C, B>) -> Result<Self, GprError>
    where
        O: Clone,
        C: Copy,
    {
        let n = online.n;
        let d = online.d;
        let mut workspace = FitBuffers::<C, B>::new(n)?;
        let compiled = online.compiled.clone();
        if C::CACHES_DISTANCES && compiled.needs_ard_sq_diff() {
            workspace.ensure_ard_if_cached(n, d)?;
        }
        let mut fitted = Self {
            kernel: online.kernel.clone(),
            compiled,
            likelihood: online.likelihood,
            x_unfitted: online.x_unfitted.clone_box(),
            y_unfitted: online.y_unfitted.clone_box(),
            x_transform: online.x_transform.clone_box(),
            y_transform: online.y_transform.clone_box(),
            optimizer: online.optimizer.clone(),
            distance_cache: online.distance_cache,
            jitter_policy: online.jitter_policy,
            workspace,
            query: online.query.clone(),
            x_obs: online.x_obs.clone(),
            y_obs: online.y_obs.clone(),
            x: compact_train_x(&online.x, n, d),
            y_train: online.y_train.clone(),
            alpha: online.alpha().to_vec(),
            n,
            d,
            mapped_factor: None,
            _recompute: PhantomData,
        };
        fitted.factorize_current()?;
        Ok(fitted)
    }

    /// Returns the number of training points.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Returns the feature dimension from the last successful fit.
    pub fn d(&self) -> usize {
        self.d
    }

    /// Returns the kernel whose hyperparameters this model owns.
    pub fn kernel(&self) -> &KernelSpec {
        &self.kernel
    }

    /// Returns the observation-noise model.
    pub fn likelihood(&self) -> &GaussianLikelihood {
        &self.likelihood
    }

    /// Returns `α = A⁻¹ y` from the last successful fit.
    pub fn alpha(&self) -> &[f64] {
        &self.alpha
    }

    /// Returns the original training features in column-major order.
    ///
    /// Same packing as [`Gpr::fit`] / [`Gpr<Fixed>::factor`]: `n` points by
    /// `d` features. Values are on the scale passed to fit, before the input
    /// transform.
    pub fn x(&self) -> &[f64] {
        &self.x_obs
    }

    /// Returns the original training targets.
    ///
    /// Values are on the scale passed to fit, before the target transform.
    pub fn y(&self) -> &[f64] {
        &self.y_obs
    }

    /// Writes this fitted model to `dir/config.json` and `dir/model.safetensors`.
    ///
    /// Omits `L` and `α`. [`crate::persist::LoadedGpr::load`] rebuilds them
    /// by factorizing. The Cholesky buffer policy is not written; load
    /// reconstructs [`RetainCholesky`].
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
    /// `L` is stored as a column-major `n×n` `f64` tensor; the lower triangle
    /// is canonical. [`crate::persist::LoadedGpr::load`] keeps the safetensors
    /// file mapped for `L`.
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
    /// to search again. `S` follows the same Cholesky-pole rule as
    /// [`Gpr::with_optimizer`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::persist::{LoadedDistance, LoadedGpr, PersistRegistry};
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
    /// let LoadedGpr::Distance(LoadedDistance::Cached(model)) =
    ///     LoadedGpr::load(&dir, &PersistRegistry::new())?
    /// else {
    ///     return Ok(());
    /// };
    /// let mut model = model.with_optimizer(Lbfgs::new());
    /// model.refit()?;
    /// let _ = std::fs::remove_dir_all(&dir);
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_optimizer<O2: PoleRecompute<B>>(
        self,
        optimizer: O2,
    ) -> FittedGpr<O2, O2::Strategy, C, B> {
        FittedGpr {
            kernel: self.kernel,
            compiled: self.compiled,
            likelihood: self.likelihood,
            x_unfitted: self.x_unfitted,
            y_unfitted: self.y_unfitted,
            x_transform: self.x_transform,
            y_transform: self.y_transform,
            optimizer,
            distance_cache: self.distance_cache,
            jitter_policy: self.jitter_policy,
            workspace: self.workspace,
            query: self.query,
            x_obs: self.x_obs,
            y_obs: self.y_obs,
            x: self.x,
            y_train: self.y_train,
            alpha: self.alpha,
            n: self.n,
            d: self.d,
            mapped_factor: self.mapped_factor,
            _recompute: PhantomData,
        }
    }

    pub(crate) fn jitter_policy(&self) -> JitterPolicy {
        self.jitter_policy
    }

    pub(crate) fn distance_cache_slot(&self) -> C {
        self.distance_cache
    }

    pub(crate) fn x_unfitted(&self) -> &dyn UnfittedTransform {
        self.x_unfitted.as_ref()
    }

    pub(crate) fn y_unfitted(&self) -> &dyn UnfittedTarget {
        self.y_unfitted.as_ref()
    }

    pub(crate) fn x_transform(&self) -> &dyn Transform {
        self.x_transform.as_ref()
    }

    pub(crate) fn y_transform(&self) -> &dyn TargetTransform {
        self.y_transform.as_ref()
    }

    pub(crate) fn chol_l(&self) -> MatRef<'_, f64> {
        match &self.mapped_factor {
            Some(mapped) => mapped.l_view(),
            None => self.workspace.core().k_matrix.as_ref(),
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
        Ok(neg_mll_from_factor(
            self.chol_l(),
            &self.y_train,
            &self.alpha,
            self.n,
        ))
    }

    /// Returns the concatenated kernel and likelihood parameter count.
    pub fn num_params(&self) -> usize {
        self.kernel.num_params() + self.likelihood.num_params()
    }

    /// Writes kernel `θ` then likelihood `θ` into `out`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if `out` is the wrong length
    /// or a custom leaf rejects the write.
    pub fn get_params(&self, out: &mut [f64]) -> Result<(), GprError> {
        let n_kernel = self.kernel.num_params();
        require_param_len(out.len(), self.num_params())?;
        self.kernel.get_params(&mut out[..n_kernel])?;
        self.likelihood.get_params(&mut out[n_kernel..])
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
    /// Returns [`GprError::InvalidHyperparameter`] if `params` is the wrong
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
        let n_kernel = self.kernel.num_params();
        require_param_len(params.len(), self.num_params())?;
        let (kernel, compiled, likelihood) = self.prepared_params(params, n_kernel)?;
        let workspace = self.workspace.clone();
        let alpha = self.alpha.clone();
        if let Err(err) = factor_train_with_policy(
            &compiled,
            self.x.as_ref(),
            &mut self.workspace,
            &self.y_train,
            likelihood.noise_variance(),
            FactorPolicy {
                jitter: self.jitter_policy,
                stage: CholeskyStage::Fit,
            },
        ) {
            self.workspace = workspace;
            self.alpha = alpha;
            return Err(err);
        }
        self.copy_alpha_from_rhs();
        self.kernel = kernel;
        self.compiled = compiled;
        self.likelihood = likelihood;
        self.mapped_factor = None;
        Ok(())
    }

    pub(crate) fn objective(&mut self) -> GprObjective<'_, O, S, C, B> {
        GprObjective::new(self)
    }

    pub(crate) fn fill_intervals(&self, out: &mut [Interval]) -> Result<(), GprError> {
        let n = self.num_params();
        if out.len() != n {
            return Err(GprError::InvalidHyperparameter {
                reason: format!("expected {n} intervals, got {}", out.len()),
            });
        }
        let n_kernel = self.kernel.num_params();
        let mut offset = 0;
        self.kernel
            .write_intervals(&mut out[..n_kernel], &mut offset)?;
        out[n_kernel] = self.likelihood.bounds();
        Ok(())
    }

    /// Sets kernel and likelihood `θ`, rebuilds `L` / `α` / `W`, and writes `∂L/∂θ`.
    ///
    /// `params` and `out` are kernel parameters followed by the likelihood
    /// parameter. One Cholesky produces `L` and `α`; `W = ααᵀ - A⁻¹` is
    /// formed from that factor. [`RetainCholesky`] keeps `W` in a dedicated
    /// buffer. [`crate::ReuseCholesky`] writes `W` over `L` and this method
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
    /// Returns [`GprError::InvalidHyperparameter`] if a slice length is wrong,
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
        let nlml = self.value_and_gradient_into_fit(params, out)?;
        self.restore_cholesky_if_overwritten()?;
        Ok(nlml)
    }

    /// Joint MLL+grad used during `fit`. Does not restore `L` when `B`
    /// overwrites the factor; the optimizer's next step rebuilds `A`.
    pub(crate) fn value_and_gradient_into_fit(
        &mut self,
        params: &[f64],
        out: &mut [f64],
    ) -> Result<f64, GprError> {
        let n_kernel = self.kernel.num_params();
        let n_params = self.num_params();
        require_param_len(params.len(), n_params)?;
        require_param_len(out.len(), n_params)?;
        let (kernel, compiled, likelihood) = self.prepared_params(params, n_kernel)?;
        let n = self.n;
        if let Err(err) = factor_train_with_policy(
            &compiled,
            self.x.as_ref(),
            &mut self.workspace,
            &self.y_train,
            likelihood.noise_variance(),
            FactorPolicy {
                jitter: self.jitter_policy,
                stage: CholeskyStage::Fit,
            },
        ) {
            let _ = self.factorize_current();
            return Err(err);
        }
        self.copy_alpha_from_rhs();
        self.kernel = kernel;
        self.likelihood = likelihood;
        self.compiled = compiled;
        self.mapped_factor = None;
        let nlml = neg_mll_from_factor(
            self.workspace.core().k_matrix.as_ref(),
            &self.y_train,
            &self.alpha,
            n,
        );
        self.fill_gradient_from_factor(n_kernel, n, out)?;
        Ok(nlml)
    }

    /// Rebuilds dirty compiled leaves, recombines the tree, and factors.
    pub(crate) fn value_from_leaf_grams(
        &mut self,
        params: &[f64],
        indices: Option<&[usize]>,
        leaf_grams: &mut Vec<Mat<f64>>,
        primed: &mut bool,
    ) -> Result<f64, GprError> {
        let n_kernel = self.kernel.num_params();
        let n_params = self.num_params();
        require_param_len(params.len(), n_params)?;
        if let Some(changed) = indices {
            require_change_indices(changed, n_params)?;
        }
        let (kernel, compiled, likelihood) = self.prepared_params(params, n_kernel)?;
        let n = self.n;
        let n_leaves = compiled.leaf_count();
        if leaf_grams.len() != n_leaves || leaf_grams.first().is_none_or(|m| m.nrows() != n) {
            *leaf_grams = (0..n_leaves).map(|_| Mat::zeros(n, n)).collect();
            *primed = false;
        }
        let mut dirty = vec![true; n_leaves];
        if *primed && indices.is_some() {
            dirty.fill(false);
            let mut last = vec![0.0; n_params];
            write_params(&self.kernel, &self.likelihood, &mut last)?;
            for (j, (&prev, &next)) in last.iter().zip(params.iter()).enumerate() {
                if prev.to_bits() != next.to_bits() && j < n_kernel {
                    dirty[compiled.leaf_index_for_param(j)?] = true;
                }
            }
        }
        for (i, slot) in leaf_grams.iter_mut().enumerate() {
            if dirty[i] {
                apply_compiled_to(
                    compiled.leaf_at(i)?,
                    self.x.as_ref(),
                    &mut self.workspace,
                    slot.as_mut(),
                )?;
            }
        }
        if let Err(err) = factor_written_k_with_policy(
            &mut self.workspace,
            &self.y_train,
            likelihood.noise_variance(),
            FactorPolicy {
                jitter: self.jitter_policy,
                stage: CholeskyStage::Fit,
            },
            |ws| {
                let core = ws.core_mut();
                compiled.combine_from_leaf_grams(
                    leaf_grams,
                    core.k_matrix.as_mut(),
                    core.exp_buf.as_mut(),
                    Triangle::Lower,
                )
            },
        ) {
            *primed = false;
            let _ = self.factorize_current();
            return Err(err);
        }
        *primed = true;
        self.copy_alpha_from_rhs();
        self.kernel = kernel;
        self.likelihood = likelihood;
        self.compiled = compiled;
        self.mapped_factor = None;
        Ok(neg_mll_from_factor(
            self.workspace.core().k_matrix.as_ref(),
            &self.y_train,
            &self.alpha,
            n,
        ))
    }

    /// Writes the analytic NLML Hessian (row-major `p×p`) at `params`.
    ///
    /// `params` is kernel `θ` followed by likelihood `θ`. After a successful
    /// call the stored kernel and likelihood match `params`. `ReuseCholesky`
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
        self.hessian_into_fit(params, out)?;
        self.restore_cholesky_if_overwritten()?;
        Ok(())
    }

    pub(crate) fn hessian_into_fit(
        &mut self,
        params: &[f64],
        out: &mut [f64],
    ) -> Result<(), GprError> {
        let n_kernel = self.kernel.num_params();
        let n_params = self.num_params();
        require_param_len(params.len(), n_params)?;
        require_param_len(out.len(), n_params * n_params)?;
        let (kernel, compiled, likelihood) = self.prepared_params(params, n_kernel)?;
        let n = self.n;
        if let Err(err) = factor_train_with_policy(
            &compiled,
            self.x.as_ref(),
            &mut self.workspace,
            &self.y_train,
            likelihood.noise_variance(),
            FactorPolicy {
                jitter: self.jitter_policy,
                stage: CholeskyStage::Fit,
            },
        ) {
            let _ = self.factorize_current();
            return Err(err);
        }
        self.copy_alpha_from_rhs();
        self.kernel = kernel;
        self.likelihood = likelihood;
        self.compiled = compiled;
        self.mapped_factor = None;
        self.fill_hessian_from_factor(n_kernel, n, out)
    }

    fn fill_hessian_from_factor(
        &mut self,
        n_kernel: usize,
        n: usize,
        out: &mut [f64],
    ) -> Result<(), GprError> {
        self.workspace.core_mut().ensure_kernel_scratch(n)?;
        if self.compiled.needs_product_grad_scratch() {
            self.workspace.core_mut().ensure_kernel_scratch(n)?;
        }
        self.workspace.form_gradient_w(&self.alpha, n);
        out.fill(0.0);
        let n_params = n_kernel + 1;
        let noise = self.likelihood.noise_variance();
        let thread_scratch = std::mem::take(&mut self.workspace.core_mut().thread_scratch);
        let second = (|| {
            for i in 0..n_params {
                for j in i..n_params {
                    self.write_second_deriv(n_kernel, i, j, n)?;
                    let inner = frobenius_lower(
                        self.workspace.gradient_w(),
                        self.workspace.core().exp_buf.as_ref(),
                        n,
                    );
                    let hij = -0.5 * inner;
                    out[i * n_params + j] = hij;
                    out[j * n_params + i] = hij;
                }
            }
            Ok::<(), GprError>(())
        })();
        self.workspace.core_mut().thread_scratch = thread_scratch;
        second?;
        self.add_noise_first_order(n_kernel, n, noise, out)?;
        if !self.workspace.has_dedicated_w() {
            self.factorize_current()?;
        }
        self.add_kernel_first_order(n_kernel, n, out)
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
                self.workspace.core_mut().exp_buf.as_mut(),
                n,
                i == n_kernel && j == n_kernel,
                self.likelihood.noise_variance(),
            );
            return Ok(());
        }
        let (core, dist) = self.workspace.split_fit();
        if let Some(d) = dist {
            let ard_cache = if self.compiled.needs_ard_sq_diff() && *d.ard_sq_diff_ready {
                Some(d.ard_sq_diff.as_ref())
            } else {
                None
            };
            write_kernel_hess(
                &self.compiled,
                d.dist_cache.as_ref(),
                self.x.as_ref(),
                ard_cache,
                core.exp_buf.as_mut(),
                core.kernel_scratch.as_mut(),
                (i, j),
            )
        } else {
            write_kernel_hess_from_coords(
                &self.compiled,
                self.x.as_ref(),
                core.exp_buf.as_mut(),
                core.kernel_scratch.as_mut(),
                i,
                j,
            )
        }
    }

    fn write_first_deriv(&mut self, idx: usize) -> Result<(), GprError> {
        let (core, dist) = self.workspace.split_fit();
        if let Some(d) = dist {
            let ard_cache = if self.compiled.needs_ard_sq_diff() && *d.ard_sq_diff_ready {
                Some(d.ard_sq_diff.as_ref())
            } else {
                None
            };
            write_kernel_grad(
                &self.compiled,
                d.dist_cache.as_ref(),
                self.x.as_ref(),
                ard_cache,
                core.exp_buf.as_mut(),
                core.kernel_scratch.as_mut(),
                idx,
            )
        } else {
            write_kernel_grad_from_coords(
                &self.compiled,
                self.x.as_ref(),
                core.exp_buf.as_mut(),
                core.kernel_scratch.as_mut(),
                idx,
            )
        }
    }

    fn add_noise_first_order(
        &mut self,
        n_kernel: usize,
        n: usize,
        noise: f64,
        out: &mut [f64],
    ) -> Result<(), GprError> {
        let n_params = n_kernel + 1;
        let (kinv_alpha, tr_kinv2) = {
            let w = self.workspace.gradient_w();
            let mut w_alpha = vec![0.0; n];
            gemv_sym_lower(w, &self.alpha, &mut w_alpha, n);
            let alpha_dot: f64 = self.alpha.iter().map(|a| a * a).sum();
            let mut kinv_alpha = vec![0.0; n];
            for i in 0..n {
                kinv_alpha[i] = self.alpha[i] * alpha_dot - w_alpha[i];
            }
            let mut tr_kinv2 = 0.0;
            for col in 0..n {
                let kinv_cc = self.alpha[col] * self.alpha[col] - w[(col, col)];
                tr_kinv2 += kinv_cc * kinv_cc;
                for row in col + 1..n {
                    let kinv_rc = self.alpha[row] * self.alpha[col] - w[(row, col)];
                    tr_kinv2 += 2.0 * kinv_rc * kinv_rc;
                }
            }
            (kinv_alpha, tr_kinv2)
        };
        let u_n: Vec<f64> = self.alpha.iter().map(|a| noise * a).collect();
        let w_n: Vec<f64> = kinv_alpha.iter().map(|a| noise * a).collect();
        let mut un_wn = 0.0;
        for i in 0..n {
            un_wn += u_n[i] * w_n[i];
        }
        let nn = n_kernel;
        out[nn * n_params + nn] += -0.5 * noise * noise * tr_kinv2 + un_wn;

        self.workspace.core_mut().ensure_kernel_scratch(n)?;
        let thread_scratch = std::mem::take(&mut self.workspace.core_mut().thread_scratch);
        let cross = (|| {
            for i in 0..n_kernel {
                self.write_first_deriv(i)?;
                let tr = trace_ki_kinv2(
                    self.workspace.core().exp_buf.as_ref(),
                    self.workspace.gradient_w(),
                    &self.alpha,
                    n,
                );
                let mut u_i = vec![0.0; n];
                gemv_sym_lower(
                    self.workspace.core().exp_buf.as_ref(),
                    &self.alpha,
                    &mut u_i,
                    n,
                );
                let mut ui_wn = 0.0;
                for k in 0..n {
                    ui_wn += u_i[k] * w_n[k];
                }
                let hij = -0.5 * noise * tr + ui_wn;
                out[i * n_params + nn] += hij;
                out[nn * n_params + i] += hij;
            }
            Ok::<(), GprError>(())
        })();
        self.workspace.core_mut().thread_scratch = thread_scratch;
        cross
    }

    fn add_kernel_first_order(
        &mut self,
        n_kernel: usize,
        n: usize,
        out: &mut [f64],
    ) -> Result<(), GprError> {
        if n_kernel == 0 {
            return Ok(());
        }
        self.workspace.core_mut().ensure_kernel_scratch(n)?;
        let n_params = n_kernel + 1;
        let thread_scratch = std::mem::take(&mut self.workspace.core_mut().thread_scratch);
        let result = (|| {
            for j in 0..n_kernel {
                self.write_first_deriv(j)?;
                let mut u_j = vec![0.0; n];
                gemv_sym_lower(
                    self.workspace.core().exp_buf.as_ref(),
                    &self.alpha,
                    &mut u_j,
                    n,
                );
                symmetrize_lower(self.workspace.core_mut().exp_buf.as_mut(), n);
                self.solve_exp_against_l(n);
                // `write_first_deriv` for a product reuses `kernel_scratch`.
                let mut q_j = Mat::zeros(n, n);
                {
                    let core = self.workspace.core();
                    for col in 0..n {
                        for row in 0..n {
                            q_j[(row, col)] = core.exp_buf[(row, col)];
                        }
                    }
                }
                let mut w_j = vec![0.0; n];
                gemv_full(q_j.as_ref(), &self.alpha, &mut w_j, n);
                for i in 0..=j {
                    let tr;
                    let mut ui_wj = 0.0;
                    if i == j {
                        tr = trace_product(q_j.as_ref(), q_j.as_ref(), n);
                        for k in 0..n {
                            ui_wj += u_j[k] * w_j[k];
                        }
                    } else {
                        self.write_first_deriv(i)?;
                        let mut u_i = vec![0.0; n];
                        gemv_sym_lower(
                            self.workspace.core().exp_buf.as_ref(),
                            &self.alpha,
                            &mut u_i,
                            n,
                        );
                        symmetrize_lower(self.workspace.core_mut().exp_buf.as_mut(), n);
                        self.solve_exp_against_l(n);
                        tr = trace_product(self.workspace.core().exp_buf.as_ref(), q_j.as_ref(), n);
                        for k in 0..n {
                            ui_wj += u_i[k] * w_j[k];
                        }
                    }
                    let add = -0.5 * tr + ui_wj;
                    out[i * n_params + j] += add;
                    if i != j {
                        out[j * n_params + i] += add;
                    }
                }
            }
            Ok::<(), GprError>(())
        })();
        self.workspace.core_mut().thread_scratch = thread_scratch;
        result
    }

    fn solve_exp_against_l(&mut self, n: usize) {
        let core = self.workspace.core_mut();
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
        if self.compiled.needs_product_grad_scratch() {
            self.workspace.core_mut().ensure_kernel_scratch(n)?;
        }
        self.workspace.form_gradient_w(&self.alpha, n);
        let thread_scratch = std::mem::take(&mut self.workspace.core_mut().thread_scratch);
        let result = (|| {
            for (i, slot) in out.iter_mut().enumerate().take(n_kernel) {
                {
                    let (core, dist) = self.workspace.split_fit();
                    if let Some(d) = dist {
                        let ard_cache = if self.compiled.needs_ard_sq_diff() && *d.ard_sq_diff_ready
                        {
                            Some(d.ard_sq_diff.as_ref())
                        } else {
                            None
                        };
                        write_kernel_grad(
                            &self.compiled,
                            d.dist_cache.as_ref(),
                            self.x.as_ref(),
                            ard_cache,
                            core.exp_buf.as_mut(),
                            core.kernel_scratch.as_mut(),
                            i,
                        )?;
                    } else {
                        write_kernel_grad_from_coords(
                            &self.compiled,
                            self.x.as_ref(),
                            core.exp_buf.as_mut(),
                            core.kernel_scratch.as_mut(),
                            i,
                        )?;
                    }
                }
                let inner = frobenius_lower(
                    self.workspace.gradient_w(),
                    self.workspace.core().exp_buf.as_ref(),
                    n,
                );
                *slot = -0.5 * inner;
            }
            Ok::<(), GprError>(())
        })();
        self.workspace.core_mut().thread_scratch = thread_scratch;
        result?;
        let mut noise_inner = 0.0;
        let d_noise = self.likelihood.noise_variance();
        let w = self.workspace.gradient_w();
        for i in 0..n {
            noise_inner += w[(i, i)] * d_noise;
        }
        out[n_kernel] = -0.5 * noise_inner;
        Ok(())
    }

    pub(super) fn restore_cholesky_if_overwritten(&mut self) -> Result<(), GprError> {
        if B::OVERWRITES_CHOLESKY {
            self.factorize_current()?;
        }
        Ok(())
    }

    /// Builds kernel, compiled kernel, and likelihood `θ` without storing them.
    ///
    /// Each `set_params` is atomic on its own type. The caller commits the
    /// triple only after `A` factors, so a later Cholesky failure cannot
    /// leave stored kernel and likelihood `θ` mixed or half-applied.
    fn prepared_params(
        &self,
        params: &[f64],
        n_kernel: usize,
    ) -> Result<(KernelSpec, CompiledKernel, GaussianLikelihood), GprError> {
        let mut likelihood = self.likelihood;
        likelihood.set_params(&params[n_kernel..])?;
        let mut kernel = self.kernel.clone();
        kernel.set_params(&params[..n_kernel])?;
        let mut compiled = self.compiled.clone();
        compiled.set_params(&params[..n_kernel])?;
        Ok((kernel, compiled, likelihood))
    }

    pub(super) fn optimize_hyperparameters(&mut self) -> Result<(), GprError>
    where
        O: Clone + for<'a> Optimizer<GprObjective<'a, O, S, C, B>>,
    {
        let mut init = vec![0.0; self.num_params()];
        self.get_params(&mut init)?;
        let kernel_before = self.kernel.clone();
        let likelihood_before = self.likelihood;
        let optimizer = self.optimizer.clone();
        let result = {
            let mut obj = self.objective();
            optimizer.minimize(&mut obj, &init)
        };
        self.commit_or_revert_optimize(kernel_before, likelihood_before, result)
    }

    pub(super) fn commit_or_revert_optimize(
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
        self.kernel = kernel;
        self.likelihood = likelihood;
        self.compiled = self.kernel.compile();
        let _ = self.factorize_current();
    }

    pub(crate) fn factorize_current(&mut self) -> Result<(), GprError> {
        self.mapped_factor = None;
        factor_train_with_policy(
            &self.compiled,
            self.x.as_ref(),
            &mut self.workspace,
            &self.y_train,
            self.likelihood.noise_variance(),
            FactorPolicy {
                jitter: self.jitter_policy,
                stage: CholeskyStage::Fit,
            },
        )?;
        self.copy_alpha_from_rhs();
        Ok(())
    }

    fn copy_alpha_from_rhs(&mut self) {
        let n = self.n;
        if self.alpha.len() != n {
            self.alpha.resize(n, 0.0);
        }
        for (i, slot) in self.alpha.iter_mut().enumerate() {
            *slot = self.workspace.core().rhs[(i, 0)];
        }
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
    /// [`GprError::InvalidHyperparameter`] / [`GprError::NonFiniteInput`] for a
    /// badly packed or non-finite `xs`.
    pub fn predict(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
    ) -> Result<Prediction, GprError> {
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
        out: &mut Prediction,
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
    ) -> Result<Prediction, GprError> {
        let mut out = Prediction::default();
        self.write_prediction(xs, n_rows, n_cols, options, &mut out)?;
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
        out: &mut Prediction,
    ) -> Result<(), GprError> {
        if n_cols != self.d {
            return Err(GprError::DimensionMismatch {
                x_dim: n_cols,
                expected_dim: self.d,
            });
        }
        validate_query(xs, n_rows, n_cols)?;
        let n = self.n;
        let m = n_rows;
        self.query.ensure(n, m, n_cols)?;
        self.query.query_xs.copy_from_slice(xs);
        self.x_transform
            .apply(&mut self.query.query_xs, n_rows, n_cols)?;
        pack_points_into(
            &self.query.query_xs,
            n_rows,
            n_cols,
            self.query.query_x.as_mut(),
        );
        let compiled = &self.compiled;
        let x_train = self.x.as_ref();
        let alpha = self.alpha.as_slice();
        let query = &mut self.query;
        match compiled.coord_mode()? {
            CoordMode::Dist | CoordMode::Either => {
                let mut thread_scratch =
                    std::mem::take(&mut self.workspace.core_mut().thread_scratch);
                fill_squared_euclidean_cross(
                    x_train.as_ref(),
                    query.query_x.as_ref(),
                    query.query_dist.as_mut(),
                    &mut thread_scratch,
                );
                self.workspace.core_mut().thread_scratch = thread_scratch;
                compiled.apply_cross(
                    query.query_dist.as_ref(),
                    query.query_k_star.as_mut(),
                    query.query_scratch.as_mut(),
                )?;
            }
            CoordMode::Points => {
                compiled.apply_cross_points(
                    x_train.as_ref(),
                    query.query_x.as_ref(),
                    query.query_k_star.as_mut(),
                    query.query_scratch.as_mut(),
                )?;
            }
            CoordMode::Mixed => {
                let mut thread_scratch =
                    std::mem::take(&mut self.workspace.core_mut().thread_scratch);
                fill_squared_euclidean_cross(
                    x_train.as_ref(),
                    query.query_x.as_ref(),
                    query.query_dist.as_mut(),
                    &mut thread_scratch,
                );
                self.workspace.core_mut().thread_scratch = thread_scratch;
                compiled.apply_cross_mixed(
                    query.query_dist.as_ref(),
                    x_train.as_ref(),
                    query.query_x.as_ref(),
                    query.query_k_star.as_mut(),
                    query.query_scratch.as_mut(),
                )?;
            }
        }
        if out.mean.len() != m {
            out.mean.resize(m, 0.0);
        }
        if out.variance.len() != m {
            out.variance.resize(m, 0.0);
        }
        for (col, mean) in out.mean.iter_mut().enumerate() {
            let mut sum = 0.0;
            for (row, &a) in alpha.iter().enumerate() {
                sum += query.query_k_star[(row, col)] * a;
            }
            *mean = sum;
        }
        let l = match &self.mapped_factor {
            Some(mapped) => mapped.l_view(),
            None => self.workspace.core().k_matrix.as_ref(),
        };
        faer::linalg::triangular_solve::solve_lower_triangular_in_place(
            l,
            query.query_k_star.as_mut(),
            faer_par_dims(n, m),
        );
        match compiled.coord_mode()? {
            CoordMode::Dist | CoordMode::Either => compiled.fill_diag(&mut query.query_kss)?,
            CoordMode::Points | CoordMode::Mixed => {
                compiled.fill_diag_points(query.query_x.as_ref(), &mut query.query_kss)?
            }
        }
        let noise = self.likelihood.noise_variance();
        for col in 0..m {
            let mut vnorm = 0.0;
            for row in 0..n {
                let v = query.query_k_star[(row, col)];
                vnorm += v * v;
            }
            let mut latent = query.query_kss[col] - vnorm;
            if latent < 0.0 {
                latent = 0.0;
            }
            out.variance[col] = match options.variance_kind {
                VarianceKind::Latent => latent,
                VarianceKind::Observation => latent + noise,
            };
        }
        self.y_transform.inverse_transform_mean(&mut out.mean)?;
        self.y_transform
            .inverse_transform_variance(&mut out.variance)?;
        out.variance_kind = options.variance_kind;
        Ok(())
    }

    fn write_prediction(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        options: PredictOptions,
        out: &mut Prediction,
    ) -> Result<(), GprError> {
        if n_cols != self.d {
            return Err(GprError::DimensionMismatch {
                x_dim: n_cols,
                expected_dim: self.d,
            });
        }
        validate_query(xs, n_rows, n_cols)?;
        let compiled = &self.compiled;
        let x_train = self.x.as_ref();
        let alpha = self.alpha.as_slice();
        let n = self.n;
        let m = n_rows;
        let mut query_xs = xs.to_vec();
        self.x_transform.apply(&mut query_xs, n_rows, n_cols)?;
        let mut query_x = Mat::zeros(m, n_cols);
        pack_points_into(&query_xs, n_rows, n_cols, query_x.as_mut());
        let mut query_dist = Mat::zeros(n, m);
        let mut query_k_star = Mat::zeros(n, m);
        let mut query_scratch = Mat::zeros(n, m);
        let mut query_kss = vec![0.0; m];
        let mut thread_scratch = empty_thread_scratch();
        match compiled.coord_mode()? {
            CoordMode::Dist | CoordMode::Either => {
                fill_squared_euclidean_cross(
                    x_train.as_ref(),
                    query_x.as_ref(),
                    query_dist.as_mut(),
                    &mut thread_scratch,
                );
                compiled.apply_cross(
                    query_dist.as_ref(),
                    query_k_star.as_mut(),
                    query_scratch.as_mut(),
                )?;
            }
            CoordMode::Points => {
                compiled.apply_cross_points(
                    x_train.as_ref(),
                    query_x.as_ref(),
                    query_k_star.as_mut(),
                    query_scratch.as_mut(),
                )?;
            }
            CoordMode::Mixed => {
                fill_squared_euclidean_cross(
                    x_train.as_ref(),
                    query_x.as_ref(),
                    query_dist.as_mut(),
                    &mut thread_scratch,
                );
                compiled.apply_cross_mixed(
                    query_dist.as_ref(),
                    x_train.as_ref(),
                    query_x.as_ref(),
                    query_k_star.as_mut(),
                    query_scratch.as_mut(),
                )?;
            }
        }
        if out.mean.len() != m {
            out.mean.resize(m, 0.0);
        }
        if out.variance.len() != m {
            out.variance.resize(m, 0.0);
        }
        for (col, mean) in out.mean.iter_mut().enumerate() {
            let mut sum = 0.0;
            for (row, &a) in alpha.iter().enumerate() {
                sum += query_k_star[(row, col)] * a;
            }
            *mean = sum;
        }
        faer::linalg::triangular_solve::solve_lower_triangular_in_place(
            self.chol_l(),
            query_k_star.as_mut(),
            faer_par_dims(n, m),
        );
        match compiled.coord_mode()? {
            CoordMode::Dist | CoordMode::Either => compiled.fill_diag(&mut query_kss)?,
            CoordMode::Points | CoordMode::Mixed => {
                compiled.fill_diag_points(query_x.as_ref(), &mut query_kss)?
            }
        }
        let noise = self.likelihood.noise_variance();
        for col in 0..m {
            let mut vnorm = 0.0;
            for row in 0..n {
                let v = query_k_star[(row, col)];
                vnorm += v * v;
            }
            let mut latent = query_kss[col] - vnorm;
            if latent < 0.0 {
                latent = 0.0;
            }
            out.variance[col] = match options.variance_kind {
                VarianceKind::Latent => latent,
                VarianceKind::Observation => latent + noise,
            };
        }
        self.y_transform.inverse_transform_mean(&mut out.mean)?;
        self.y_transform
            .inverse_transform_variance(&mut out.variance)?;
        out.variance_kind = options.variance_kind;
        Ok(())
    }

    /// Returns the predictive mean and query–query covariance at `xs`.
    ///
    /// Default [`PredictOptions`] uses [`VarianceKind::Observation`]: `σn²`
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
    ) -> Result<PredictiveCovariance, GprError> {
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
    ) -> Result<PredictiveCovariance, GprError> {
        self.write_covariance(xs, n_rows, n_cols, options)
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
    /// factored after [`JitterPolicy`] retries.
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
    ) -> Result<Vec<f64>, GprError> {
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
    ) -> Result<Vec<f64>, GprError> {
        let cov = self.write_covariance(xs, n_rows, n_cols, options)?;
        if n_draws == 0 {
            return Ok(Vec::new());
        }
        let m = cov.mean.len();
        let mut a = Mat::zeros(m, m);
        for col in 0..m {
            for row in 0..m {
                a[(row, col)] = cov.covariance[col * m + row];
            }
        }
        let req = llt::factor::cholesky_in_place_scratch::<f64>(m, faer_par(m), Default::default());
        let mut scratch = MemBuffer::new(req);
        cholesky_lower_with_policy(
            &mut a,
            &mut scratch,
            self.jitter_policy,
            CholeskyStage::Predict,
        )?;
        let mut rng = crate::rng::small_rng(seed);
        let mut out = vec![0.0; m * n_draws];
        let mut z = vec![0.0; m];
        let mut lz = vec![0.0; m];
        for draw in 0..n_draws {
            for slot in &mut z {
                *slot = crate::rng::unit_normal(&mut rng);
            }
            mul_lower_chol(a.as_ref(), &z, &mut lz);
            let col = &mut out[draw * m..(draw + 1) * m];
            for i in 0..m {
                col[i] = cov.mean[i] + lz[i];
            }
        }
        Ok(out)
    }

    fn write_covariance(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        options: PredictOptions,
    ) -> Result<PredictiveCovariance, GprError> {
        if n_cols != self.d {
            return Err(GprError::DimensionMismatch {
                x_dim: n_cols,
                expected_dim: self.d,
            });
        }
        validate_query(xs, n_rows, n_cols)?;
        let compiled = &self.compiled;
        let x_train = self.x.as_ref();
        let alpha = self.alpha.as_slice();
        let n = self.n;
        let m = n_rows;
        let mut query_xs = xs.to_vec();
        self.x_transform.apply(&mut query_xs, n_rows, n_cols)?;
        let mut query_x = Mat::zeros(m, n_cols);
        pack_points_into(&query_xs, n_rows, n_cols, query_x.as_mut());
        let mut query_dist = Mat::zeros(n, m);
        let mut query_k_star = Mat::zeros(n, m);
        let mut query_scratch = Mat::zeros(n, m);
        let mut thread_scratch = empty_thread_scratch();
        match compiled.coord_mode()? {
            CoordMode::Dist | CoordMode::Either => {
                fill_squared_euclidean_cross(
                    x_train.as_ref(),
                    query_x.as_ref(),
                    query_dist.as_mut(),
                    &mut thread_scratch,
                );
                compiled.apply_cross(
                    query_dist.as_ref(),
                    query_k_star.as_mut(),
                    query_scratch.as_mut(),
                )?;
            }
            CoordMode::Points => {
                compiled.apply_cross_points(
                    x_train.as_ref(),
                    query_x.as_ref(),
                    query_k_star.as_mut(),
                    query_scratch.as_mut(),
                )?;
            }
            CoordMode::Mixed => {
                fill_squared_euclidean_cross(
                    x_train.as_ref(),
                    query_x.as_ref(),
                    query_dist.as_mut(),
                    &mut thread_scratch,
                );
                compiled.apply_cross_mixed(
                    query_dist.as_ref(),
                    x_train.as_ref(),
                    query_x.as_ref(),
                    query_k_star.as_mut(),
                    query_scratch.as_mut(),
                )?;
            }
        }
        let mut mean = vec![0.0; m];
        for (col, slot) in mean.iter_mut().enumerate() {
            let mut sum = 0.0;
            for (row, &a) in alpha.iter().enumerate() {
                sum += query_k_star[(row, col)] * a;
            }
            *slot = sum;
        }
        faer::linalg::triangular_solve::solve_lower_triangular_in_place(
            self.chol_l(),
            query_k_star.as_mut(),
            faer_par_dims(n, m),
        );
        let mut kss = Mat::zeros(m, m);
        let mut kss_scratch = Mat::zeros(m, m);
        fill_query_query_kernel(
            compiled,
            query_x.as_ref(),
            kss.as_mut(),
            kss_scratch.as_mut(),
            &mut thread_scratch,
        )?;
        for col in 0..m {
            for row in 0..m {
                let mut dot = 0.0;
                for k in 0..n {
                    dot += query_k_star[(k, row)] * query_k_star[(k, col)];
                }
                kss[(row, col)] -= dot;
            }
        }
        let noise = self.likelihood.noise_variance();
        for i in 0..m {
            let mut latent = kss[(i, i)];
            if latent < 0.0 {
                latent = 0.0;
            }
            kss[(i, i)] = match options.variance_kind {
                VarianceKind::Latent => latent,
                VarianceKind::Observation => latent + noise,
            };
        }
        self.y_transform.inverse_transform_mean(&mut mean)?;
        let mut covariance = vec![0.0; m * m];
        for col in 0..m {
            for row in 0..m {
                covariance[col * m + row] = kss[(row, col)];
            }
        }
        self.y_transform
            .inverse_transform_covariance(&mut covariance)?;
        Ok(PredictiveCovariance {
            mean,
            covariance,
            variance_kind: options.variance_kind,
        })
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
    pub fn loo_predict(&self) -> Result<Prediction, GprError> {
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
    pub fn loo_predict_with(&self, options: PredictOptions) -> Result<Prediction, GprError> {
        let y = self.y_train.as_slice();
        let alpha = self.alpha.as_slice();
        let n = self.n;
        let mut q_diag = vec![0.0; n];
        inv_diag_from_chol_l(self.chol_l(), &mut q_diag);
        let noise = self.likelihood.noise_variance();
        let mut mean = vec![0.0; n];
        let mut variance = vec![0.0; n];
        for i in 0..n {
            let qii = q_diag[i];
            if !qii.is_finite() || qii <= 0.0 {
                return Err(GprError::NonPositiveDefiniteMatrix);
            }
            mean[i] = y[i] - alpha[i] / qii;
            let obs = 1.0 / qii;
            variance[i] = match options.variance_kind {
                VarianceKind::Observation => obs,
                VarianceKind::Latent => (obs - noise).max(0.0),
            };
        }
        self.y_transform.inverse_transform_mean(&mut mean)?;
        self.y_transform.inverse_transform_variance(&mut variance)?;
        Ok(Prediction {
            mean,
            variance,
            variance_kind: options.variance_kind,
        })
    }
}

#[allow(private_bounds)] // `GprObjective` is crate-private; `refit` still needs `O: Optimizer` for it.
impl<O, S, C, B> FittedGpr<O, S, C, B>
where
    C: DistanceCacheSlot,
    B: AllocWorkspace,
    O: Clone + for<'a> Optimizer<GprObjective<'a, O, S, C, B>>,
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
        self.optimize_hyperparameters()?;
        self.restore_cholesky_if_overwritten()
    }
}

#[allow(private_bounds)] // `DistanceCacheSlot` is crate-private; `refit` still needs it.
impl<C: DistanceCacheSlot> FittedGpr<Fixed, FullRecompute, C, RetainCholesky> {
    pub(crate) fn into_online_preserving_factor(
        self,
    ) -> Result<OnlineGpr<Fixed, FullRecompute, C, RetainCholesky>, GprError> {
        if self.mapped_factor.is_none() {
            return self.into_online();
        }
        let n = self.n;
        let mut workspace = OnlineWorkspace::from_active(n)?;
        workspace.copy_ld_from(self.chol_l(), n)?;
        OnlineWorkspace::set_vector_prefix(&mut workspace.y, &self.y_train);
        OnlineWorkspace::set_vector_prefix(&mut workspace.alpha, &self.alpha);
        Ok(OnlineGpr::from_parts(
            self.kernel,
            self.compiled,
            self.likelihood,
            self.x_unfitted,
            self.y_unfitted,
            self.x_transform,
            self.y_transform,
            self.optimizer,
            self.distance_cache,
            self.jitter_policy,
            workspace,
            self.query,
            self.x_obs,
            self.y_obs,
            self.x,
            self.y_train,
            self.alpha,
            self.n,
            self.d,
        ))
    }

    pub(crate) fn from_persisted(parts: PersistedModel<C>) -> Result<Self, GprError> {
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
        let mut workspace = FitBuffers::<C, RetainCholesky>::new(n)?;
        let compiled = parts.kernel.compile();
        if C::CACHES_DISTANCES && compiled.needs_ard_sq_diff() {
            workspace.ensure_ard_if_cached(n, d)?;
        }
        Ok(Self {
            kernel: parts.kernel,
            compiled,
            likelihood: parts.likelihood,
            x_unfitted: parts.x_unfitted,
            y_unfitted: parts.y_unfitted,
            x_transform: parts.x_transform,
            y_transform: parts.y_transform,
            optimizer: Fixed,
            distance_cache: parts.distance_cache,
            jitter_policy: parts.jitter_policy,
            workspace,
            query: QueryWorkspace::new(),
            x: pack_points(&x_buf, n, d),
            y_train: y_buf,
            x_obs: parts.x_obs,
            y_obs: parts.y_obs,
            alpha: parts.alpha,
            n,
            d,
            mapped_factor: parts.mapped,
            _recompute: PhantomData,
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
        self.factorize_current()
    }
}

fn fill_query_query_kernel(
    compiled: &CompiledKernel,
    query_x: MatRef<'_, f64>,
    kss: MatMut<'_, f64>,
    scratch: MatMut<'_, f64>,
    thread_scratch: &mut [Mat<f64>],
) -> Result<(), GprError> {
    match compiled.coord_mode()? {
        CoordMode::Dist | CoordMode::Either => {
            let m = query_x.nrows();
            let mut dist_ss = Mat::zeros(m, m);
            fill_squared_euclidean(query_x, dist_ss.as_mut(), thread_scratch);
            compiled.apply(dist_ss.as_ref(), kss, Triangle::Full, scratch)
        }
        CoordMode::Points => compiled.apply_points(query_x, kss, Triangle::Full, scratch),
        CoordMode::Mixed => {
            let m = query_x.nrows();
            let mut dist_ss = Mat::zeros(m, m);
            fill_squared_euclidean(query_x, dist_ss.as_mut(), thread_scratch);
            compiled.apply_mixed(
                MixedKernelViews::new(dist_ss.as_ref(), query_x),
                kss,
                Triangle::Full,
                scratch,
            )
        }
    }
}

fn require_change_indices(indices: &[usize], n_params: usize) -> Result<(), GprError> {
    if indices.is_empty() {
        return Err(GprError::InvalidHyperparameter {
            reason: "change indices must not be empty".to_owned(),
        });
    }
    let mut seen = vec![false; n_params];
    for &i in indices {
        if i >= n_params {
            return Err(GprError::InvalidHyperparameter {
                reason: format!("change index {i} is out of range (n_params={n_params})"),
            });
        }
        if seen[i] {
            return Err(GprError::InvalidHyperparameter {
                reason: format!("change index {i} is duplicated"),
            });
        }
        seen[i] = true;
    }
    Ok(())
}

fn zero_and_maybe_noise(mut out: MatMut<'_, f64>, n: usize, noise_diag: bool, noise: f64) {
    for col in 0..n {
        for row in col..n {
            out[(row, col)] = if noise_diag && row == col { noise } else { 0.0 };
        }
    }
}

fn kinv_from_w(alpha: &[f64], w: MatRef<'_, f64>, row: usize, col: usize) -> f64 {
    let (r, c) = if row >= col { (row, col) } else { (col, row) };
    alpha[row] * alpha[col] - w[(r, c)]
}

fn ki_sym(ki: MatRef<'_, f64>, row: usize, col: usize) -> f64 {
    if row >= col {
        ki[(row, col)]
    } else {
        ki[(col, row)]
    }
}

fn trace_ki_kinv2(ki: MatRef<'_, f64>, w: MatRef<'_, f64>, alpha: &[f64], n: usize) -> f64 {
    let mut tr = 0.0;
    for c in 0..n {
        for b in 0..n {
            let mut m_bc = 0.0;
            for k in 0..n {
                m_bc += ki_sym(ki, b, k) * kinv_from_w(alpha, w, k, c);
            }
            tr += kinv_from_w(alpha, w, b, c) * m_bc;
        }
    }
    tr
}

fn compact_train_x(x: &Mat<f64>, n: usize, d: usize) -> Mat<f64> {
    if x.nrows() == n && x.ncols() == d {
        return x.clone();
    }
    Mat::from_fn(n, d, |i, j| x[(i, j)])
}

fn mul_lower_chol(l: MatRef<'_, f64>, z: &[f64], out: &mut [f64]) {
    let m = l.nrows();
    debug_assert_eq!(z.len(), m);
    debug_assert_eq!(out.len(), m);
    for i in 0..m {
        let mut s = 0.0;
        for (j, &zj) in z.iter().enumerate().take(i + 1) {
            s += l[(i, j)] * zj;
        }
        out[i] = s;
    }
}
