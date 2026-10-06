//! Factored collapsed variational SGPR.

use crate::sparse::QueryDist;
use std::marker::PhantomData;

use faer::Mat;
#[cfg(test)]
use faer::MatRef;

use crate::error::GprError;
use crate::policy::with_kernel_exp;
use crate::sparse::{
    PredictScratch, SparseCore, SparseScratch, sparse_core_accessors, sparse_distance_accessors,
    sparse_kernel_accessor, sparse_point_accessors,
};

use crate::kernel::{DistanceKernel, KernelScalar, KernelSpec, ModelKernel, PointKernel, PointUse};
use crate::optimizer::{Fixed, Lbfgs, OptResult, Optimizer};
use crate::param::Interval;
use crate::precision::{DoublePrecision, ModelPrecision};
use crate::sgpr::SgprObjective;
use crate::{PredictOptions, Prediction, PredictiveCovariance};

use super::factor::{
    VfeState, VfeSystem, analytic_gradient, analytic_hessian, assemble_vfe, fill_z_intervals,
    predict_vfe_covariance, predict_vfe_into, publish_sgpr_weights, vfe_loo,
    vfe_neg_log_marginal_likelihood,
};
use super::model::Sgpr;
use super::online::OnlineSgpr;
use super::{FixedInducing, InducingLayout};

/// Represents the factored collapsed variational SGPR at the `θ` used by [`Sgpr::fit`] or [`Sgpr<Fixed>::factor`].
///
/// Stores the LLT of `K_mm = k(Z, Z)` and the VFE factors used by
/// [`Self::predict`] and [`Self::neg_log_marginal_likelihood`]. Observation
/// noise is not added to `K_mm`. Hyperparameters are kernel `θ` then
/// likelihood `θ`. [`FreeInducing`](crate::FreeInducing) then appends column-major `Z`.
/// [`Self::into_online`] yields [`OnlineSgpr`] for training-point and
/// inducing-point updates.
///
/// See the example on [`Self::predict`].
#[derive(Clone, Debug)]
pub struct FittedSgpr<
    O = Lbfgs,
    I = FixedInducing,
    P: ModelPrecision = DoublePrecision,
    K: ModelKernel = KernelSpec,
> {
    pub(super) core: SparseCore<K>,
    pub(super) _kernel: PhantomData<K>,
    /// Kernel scratch kept between `&mut self` calls.
    pub(super) scratch: SparseScratch<P::Storage, K::Supply>,
    pub(super) optimizer: O,
    pub(super) inducing: PhantomData<I>,
    /// Lower `L` from `K_mm = L Lᵀ`.
    pub(super) k_mm_l: Mat<P::Storage>,
    /// `A = L_mm⁻¹ K(Z, X)` (`m × n`).
    pub(super) a: Mat<P::Storage>,
    /// Lower `L_B` from `B = σn² I + A Aᵀ`.
    pub(super) b_l: Mat<P::Storage>,
    /// Storage solve `B w = A y`. Marginal likelihood uses this.
    pub(super) w: Vec<P::Storage>,
    /// Predict weights. [`DoublePrecision`] and [`SinglePrecision`] promote `w`.
    /// [`MixedPrecision`] stores the refined `f64` weights.
    pub(super) predict_w: Vec<P::Refine>,
    pub(super) k_diag_sum: P::Storage,
    pub(super) a_frobenius2: P::Storage,
}

impl<O, I: InducingLayout, P, K: ModelKernel> FittedSgpr<O, I, P, K>
where
    P: crate::precision::GpScalar,
{
    sparse_core_accessors!();

    /// The `predict_with` of a query with the supplied distances `qd`.
    pub(crate) fn query(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        qd: &QueryDist,
        options: PredictOptions,
    ) -> Result<Prediction<P::Refine>, GprError> {
        let mut out = Prediction::default();
        predict_vfe_into::<P, _>(
            &self.core,
            &VfeSystem::new(
                &self.core,
                self.k_mm_l.as_ref(),
                self.b_l.as_ref(),
                &self.predict_w,
            ),
            xs,
            n_rows,
            n_cols,
            qd,
            options,
            &mut PredictScratch::default(),
            &mut out,
        )?;
        Ok(out)
    }

    /// The `predict_covariance_with` of a query with the supplied distances `qd`.
    pub(crate) fn query_covariance(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        qd: &QueryDist,
        options: PredictOptions,
    ) -> Result<PredictiveCovariance<P::Refine>, GprError> {
        predict_vfe_covariance::<P, _>(
            &self.core,
            &VfeSystem::new(
                &self.core,
                self.k_mm_l.as_ref(),
                self.b_l.as_ref(),
                &self.predict_w,
            ),
            xs,
            n_rows,
            n_cols,
            qd,
            options,
        )
    }

    /// The `predict_with_into` of a query with the supplied distances `qd`.
    pub(crate) fn query_into(
        &mut self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        qd: &QueryDist,
        options: PredictOptions,
        out: &mut Prediction<P::Refine>,
    ) -> Result<(), GprError> {
        predict_vfe_into::<P, _>(
            &self.core,
            &VfeSystem::new(
                &self.core,
                self.k_mm_l.as_ref(),
                self.b_l.as_ref(),
                &self.predict_w,
            ),
            xs,
            n_rows,
            n_cols,
            qd,
            options,
            &mut self.scratch.predict,
            out,
        )
    }

    /// Returns the concatenated parameter count.
    ///
    /// Kernel `θ` then likelihood `θ`. [`FreeInducing`](crate::FreeInducing) also counts
    /// column-major `Z` (`m × d`).
    ///
    /// See the example on [`Self::predict`].
    pub fn num_params(&self) -> usize {
        self.core.theta_len() + I::z_params(self.core.m, self.core.d)
    }

    /// Writes kernel `θ`, likelihood `θ`, and (when free) column-major `Z`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::LengthMismatch`] if `out` is the wrong length
    /// or a custom leaf rejects the write.
    ///
    /// See the example on [`Self::predict`].
    pub fn get_params(&self, out: &mut [f64]) -> Result<(), GprError> {
        crate::data::require_count(out.len(), self.num_params(), "parameters")?;
        let n_theta = self.core.theta_len();
        self.core.read_theta(&mut out[..n_theta])?;
        if I::z_params(self.core.m, self.core.d) > 0 {
            out[n_theta..].copy_from_slice(&self.core.z_train);
        }
        Ok(())
    }

    fn same_stored_params(&self, params: &[f64]) -> Result<bool, GprError> {
        let mut current = vec![0.0; params.len()];
        self.get_params(&mut current)?;
        Ok(current == params)
    }

    /// Sets kernel then likelihood `θ` and rebuilds the VFE factors.
    ///
    /// `params` matches [`Self::get_params`]. [`FreeInducing`](crate::FreeInducing) also writes
    /// column-major `Z` from the tail of the slice. Training `X` / `y` are
    /// not changed. Values are committed together only after the VFE system
    /// factors.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::LengthMismatch`] if `params` is the wrong
    /// length, [`GprError::InvalidNoiseVariance`] if the likelihood `θ` is
    /// invalid, or [`GprError::CholeskyFailed`] if `K_mm` or `B` cannot be
    /// factored. A rejected slice or a Cholesky failure leaves stored `θ`
    /// and the VFE factors unchanged.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Fixed, GaussianLikelihood, Sgpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let mut fitted = Sgpr::new(
    ///     KernelSpec::from(RbfKernel::new(1.0)?),
    ///     GaussianLikelihood::new(0.1)?,
    /// )
    /// .with_optimizer(Fixed)
    /// .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0], &[0.0, 1.0], 2)
    /// .map_err(|(_, e)| e)?;
    /// let mut params = [0.0; 2];
    /// fitted.get_params(&mut params)?;
    /// params[0] = 0.5_f64.ln();
    /// fitted.set_params(&params)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_params(&mut self, params: &[f64]) -> Result<(), GprError> {
        let n_theta = self.core.theta_len();
        crate::data::require_count(params.len(), self.num_params(), "parameters")?;
        if self.same_stored_params(params)? {
            return Ok(());
        }
        let (kernel, likelihood) = self.core.stage_theta(&params[..n_theta])?;
        let free_z = I::z_params(self.core.m, self.core.d) > 0;
        let z = if free_z {
            params[n_theta..].to_vec()
        } else {
            self.core.z_train.clone()
        };
        let z_obs = if free_z {
            self.core.inducing_obs(&z, self.core.m)?
        } else {
            self.core.z_obs.clone()
        };
        let state = with_kernel_exp!(self.core.math, M => assemble_vfe::<M, _, _>(
            &kernel,
            self.core.jitter,
            likelihood,
            &self.core.x_train,
            self.core.n,
            self.core.d,
            &self.core.y_train,
            &z,
            self.core.m,
            self.core.dist.as_ref(),
            &mut self.scratch.storage,
            &mut self.scratch.f64,
        ))?;
        self.core.kernel = kernel;
        self.core.likelihood = likelihood;
        self.core.z_train = z;
        self.core.z_obs = z_obs;
        self.apply_vfe(state);
        Ok(())
    }

    pub(crate) fn fill_intervals(&self, out: &mut [Interval]) -> Result<(), GprError> {
        crate::data::require_count(out.len(), self.num_params(), "intervals")?;
        let n_theta = self.core.theta_len();
        self.core.theta_intervals(&mut out[..n_theta])?;
        if I::z_params(self.core.m, self.core.d) > 0 {
            fill_z_intervals(
                &self.core.x_train,
                self.core.n,
                self.core.d,
                &mut out[n_theta..],
            )?;
        }
        Ok(())
    }

    /// Sets parameters, rebuilds the VFE system, and writes `∂L/∂θ` of the negative ELBO.
    ///
    /// `params` and `out` match [`Self::get_params`]. When inducing points
    /// are fixed and `Z = X` the gradient matches
    /// [`crate::FittedGpr::value_and_gradient_into`]. The returned value is
    /// the same as [`Self::neg_log_marginal_likelihood`] after a successful
    /// call.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::LengthMismatch`] if a slice length is wrong,
    /// [`GprError::InvalidNoiseVariance`] if the likelihood `θ` is invalid,
    /// or [`GprError::CholeskyFailed`] if the VFE system cannot be factored.
    /// Kernel and likelihood `θ` are committed together only after the
    /// system factors.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Fixed, GaussianLikelihood, Sgpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let mut fitted = Sgpr::new(
    ///     KernelSpec::from(RbfKernel::new(1.0)?),
    ///     GaussianLikelihood::new(0.1)?,
    /// )
    /// .with_optimizer(Fixed)
    /// .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0], &[0.0, 1.0], 2)
    /// .map_err(|(_, e)| e)?;
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
    ) -> Result<f64, GprError>
    where
        P: crate::precision::GpScalar,
    {
        let n_params = self.num_params();
        crate::data::require_count(params.len(), n_params, "parameters")?;
        crate::data::require_count(out.len(), n_params, "parameters")?;
        self.set_params(params)?;
        let value = self.neg_log_marginal_likelihood()?;
        let include_z = I::z_params(self.core.m, self.core.d) > 0;
        let mut ks = std::mem::take(&mut self.scratch.storage);
        let result = with_kernel_exp!(self.core.math, M => analytic_gradient::<M, _, _, _, _>(
            self, out, include_z, &mut ks
        ));
        self.scratch.storage = ks;
        result?;
        Ok(value)
    }

    /// Writes the Hessian of the negative ELBO (row-major `p×p`) into `out`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::LengthMismatch`] if a slice length is wrong,
    /// [`GprError::InvalidNoiseVariance`] if the likelihood `θ` is invalid,
    /// or [`GprError::CholeskyFailed`] if the VFE system cannot be factored.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Fixed, GaussianLikelihood, Sgpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let mut fitted = Sgpr::new(
    ///     KernelSpec::from(RbfKernel::new(1.0)?),
    ///     GaussianLikelihood::new(0.1)?,
    /// )
    /// .with_optimizer(Fixed)
    /// .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0], &[0.0, 1.0], 2)
    /// .map_err(|(_, e)| e)?;
    /// let mut params = [0.0; 2];
    /// fitted.get_params(&mut params)?;
    /// let mut hess = [0.0; 4];
    /// fitted.hessian_into(&params, &mut hess)?;
    /// assert!(hess.iter().all(|h| h.is_finite()));
    /// # Ok(())
    /// # }
    /// ```
    pub fn hessian_into(&mut self, params: &[f64], out: &mut [f64]) -> Result<(), GprError>
    where
        P: crate::precision::GpScalar,
    {
        let n_params = self.num_params();
        crate::data::require_count(params.len(), n_params, "parameters")?;
        crate::data::require_count(out.len(), n_params * n_params, "parameters")?;
        self.set_params(params)?;
        let include_z = I::z_params(self.core.m, self.core.d) > 0;
        let mut ks = std::mem::take(&mut self.scratch.storage);
        let result = with_kernel_exp!(self.core.math, M => analytic_hessian::<M, _, _, _, _>(
            self, out, include_z, &mut ks
        ));
        self.scratch.storage = ks;
        result?;
        Ok(())
    }

    /// Writes this model to `dir` as `config.json` and `model.safetensors`.
    ///
    /// Stores the kernel, likelihood, kernel `exp`, `K_mm` jitter policy,
    /// precision, transforms (unfitted and fitted), the original `X`, `y`,
    /// and `Z`, and `Z` in transformed coordinates. The factors are
    /// not stored; [`crate::LoadedSgpr::load`] factors the system again at the saved `θ`
    /// and `Z`. Caller-defined kernels and transforms need their
    /// `persist_id` / `persist_state` and a [`crate::PersistRegistry`] entry.
    /// The optimizer and the inducing-point search are not stored.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::PersistFailed`] when the directory cannot be
    /// written or a kernel or transform has no persist form.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Fixed, GaussianLikelihood, Sgpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let model = Sgpr::new(KernelSpec::from(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?)
    ///     .with_optimizer(Fixed)
    ///     .factor(&[0.0, 1.0, 2.0], 3, 1, &[0.0, 1.0, 0.5], &[0.5, 1.5], 2)
    ///     .map_err(|(_, e)| e)?;
    /// let dir = std::env::temp_dir().join(format!("gprx-doctest-save-sgpr-{}", std::process::id()));
    /// let _ = std::fs::remove_dir_all(&dir);
    /// model.save(&dir)?;
    /// let loaded = gprx::LoadedSgpr::load(&dir, &gprx::PersistRegistry::new())?;
    /// assert_eq!(loaded.n(), model.n());
    /// let _ = std::fs::remove_dir_all(&dir);
    /// # Ok(())
    /// # }
    /// ```
    pub fn save(&self, dir: impl AsRef<std::path::Path>) -> Result<(), GprError> {
        crate::persist::save_sgpr(self, dir.as_ref())
    }

    pub(crate) fn refresh_predict_w(&mut self) -> Result<(), GprError> {
        self.predict_w = with_kernel_exp!(self.core.math, M => publish_sgpr_weights::<M, P, _>(
            &self.core.kernel,
            self.core.jitter,
            self.a.as_ref(),
            self.b_l.as_ref(),
            &self.w,
            &self.core.x_train,
            &self.core.y_train,
            &self.core.z_train,
            self.core.likelihood.noise_variance(),
            self.core.n,
            self.core.m,
            self.core.d,
            self.core.dist.as_ref(),
        ))?;
        Ok(())
    }

    pub(crate) fn apply_vfe(&mut self, state: VfeState<P::Storage>) {
        self.predict_w = promote_predict_w::<P>(&state.w);
        self.k_mm_l = state.k_mm_l;
        self.a = state.a;
        self.b_l = state.b_l;
        self.w = state.w;
        self.k_diag_sum = state.k_diag_sum;
        self.a_frobenius2 = state.a_frobenius2;
    }

    /// `w` and the lower `L_B` of `B`, for the SVGP-at-Titsias checks.
    #[cfg(test)]
    pub(crate) fn vfe_w_and_b_l(&self) -> (&[P::Storage], MatRef<'_, P::Storage>) {
        (&self.w, self.b_l.as_ref())
    }

    /// The training data, settings, and fitted transforms.
    pub(crate) fn core(&self) -> &SparseCore<K> {
        &self.core
    }

    pub(crate) fn into_trainer(self) -> Sgpr<O, I, P, K> {
        Sgpr {
            spec: self.core.spec(),
            optimizer: self.optimizer,
            inducing: PhantomData,
            _precision: PhantomData,
            _kernel: PhantomData,
        }
    }

    pub(crate) fn optimize_hyperparameters(&mut self) -> Result<(), GprError>
    where
        O: Clone + for<'a> Optimizer<SgprObjective<'a, O, I, P, K>>,
    {
        let mut init = vec![0.0; self.num_params()];
        self.get_params(&mut init)?;
        let optimizer = self.optimizer.clone();
        let result = {
            let mut obj = SgprObjective::new(self);
            optimizer.minimize(&mut obj, &init)
        };
        self.commit_or_revert_optimize(&init, result)
    }

    /// Takes the optimizer's result, or rebuilds the model at `init` (the
    /// parameters before the search) when the search failed.
    ///
    /// Only the parameters are kept for the way back, not a copy of the
    /// model: the VFE system at `init` factored before the search, so it is
    /// assembled again from them.
    fn commit_or_revert_optimize(
        &mut self,
        init: &[f64],
        result: Result<OptResult, GprError>,
    ) -> Result<(), GprError> {
        let committed = match result {
            Ok(opt) if opt.params.len() != self.num_params() || !opt.value.is_finite() => {
                Err(GprError::OptimizationNotConverged {
                    iterations: opt.iterations as usize,
                })
            }
            Ok(opt) => self
                .set_params(&opt.params)
                .and_then(|()| self.refresh_predict_w()),
            Err(err) => Err(err),
        };
        if let Err(err) = committed {
            self.revert_to(init);
            return Err(err);
        }
        Ok(())
    }

    fn revert_to(&mut self, init: &[f64]) {
        // `init` factored before the search; a failure here would leave the
        // model at the last factored point, which is still consistent.
        if self.set_params(init).is_ok() {
            let _ = self.refresh_predict_w();
        }
    }

    /// Returns the negative VFE evidence lower bound (the sparse NLML).
    ///
    /// When `Z = X` and the kernel has no White leaf, this matches
    /// [`crate::FittedGpr::neg_log_marginal_likelihood`] of
    /// [`crate::Gpr<Fixed>::factor`] on the same data. `K(Z, X)` never holds
    /// a White leaf's diagonal, so the bound does not jump when `Z` moves off
    /// `X`.
    ///
    /// # Errors
    ///
    /// The stored factors are already valid, so this returns `Ok`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Fixed, GaussianLikelihood, Sgpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    /// let likelihood = GaussianLikelihood::new(0.1)?;
    /// let fitted = Sgpr::new(kernel, likelihood)
    ///     .with_optimizer(Fixed)
    ///     .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0], &[0.0, 1.0], 2)
    ///     .map_err(|(_, e)| e)?;
    /// let nlml = fitted.neg_log_marginal_likelihood()?;
    /// assert!(nlml.is_finite());
    /// # Ok(())
    /// # }
    /// ```
    pub fn neg_log_marginal_likelihood(&self) -> Result<f64, GprError> {
        vfe_neg_log_marginal_likelihood(
            self.a.as_ref(),
            self.b_l.as_ref(),
            &self.w,
            &self.core.y_train,
            self.k_diag_sum,
            self.a_frobenius2,
            self.core.likelihood.noise_variance(),
            self.core.n,
            self.core.m,
        )
    }

    /// Returns the leave-one-out mean and observation variance at every training point.
    ///
    /// `p(y_i | X, y_{-i}, θ, Z)` of the collapsed VFE posterior: the
    /// optimal `q(u)` without point `i` at fixed `θ` and `Z`, predicted at
    /// `x_i`. It is a rank-1 downdate of `B = σn² I + A Aᵀ` per point
    /// (Sherman–Morrison), `O(n m²)` in all. At `Z = X` it matches
    /// [`crate::FittedGpr::loo_predict`]. Mean and variance are
    /// inverse-transformed like [`Self::predict`].
    ///
    /// # Errors
    ///
    /// Returns [`GprError::NonPositiveDefiniteMatrix`] if a downdated `B` is
    /// not positive definite, or [`GprError::CholeskyFailed`] if an `f32` storage cannot factor `K_mm` again in `f64`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Fixed, GaussianLikelihood, Sgpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let fitted = Sgpr::new(
    ///     KernelSpec::from(RbfKernel::new(1.0)?),
    ///     GaussianLikelihood::new(0.1)?,
    /// )
    /// .with_optimizer(Fixed)
    /// .factor(&[0.0, 1.0, 2.0], 3, 1, &[0.0, 1.0, 0.5], &[0.5, 1.5], 2)
    /// .map_err(|(_, e)| e)?;
    /// let loo = fitted.loo_predict()?;
    /// assert_eq!(loo.mean.len(), fitted.n());
    /// # Ok(())
    /// # }
    /// ```
    pub fn loo_predict(&self) -> Result<Prediction<P::Refine>, GprError> {
        self.loo_predict_with(PredictOptions::default())
    }

    /// Returns leave-one-out mean and variance with an explicit variance kind.
    ///
    /// Latent variance is the VFE variance of `f(x_i)` without point `i`;
    /// observation variance adds `σn²`.
    ///
    /// # Errors
    ///
    /// Same as [`Self::loo_predict`].
    ///
    /// See the example on [`Self::predict`].
    pub fn loo_predict_with(
        &self,
        options: PredictOptions,
    ) -> Result<Prediction<P::Refine>, GprError> {
        vfe_loo::<P, _>(
            &self.core,
            self.a.as_ref(),
            self.b_l.as_ref(),
            &self.w,
            options,
        )
    }

    #[cfg(test)]
    pub(crate) fn k_mm_l(&self) -> MatRef<'_, P::Storage> {
        self.k_mm_l.as_ref()
    }
}

impl<O, I: InducingLayout, P: crate::precision::GpScalar> FittedSgpr<O, I, P> {
    sparse_kernel_accessor!();

    /// Converts this model into an online sparse GPR.
    ///
    /// [`OnlineSgpr`] can append or drop training points and inducing
    /// points. [`FixedInducing`] and [`FreeInducing`](crate::FreeInducing) both produce
    /// [`OnlineSgpr<O>`] whose parameters are kernel then likelihood
    /// `θ`. The stored VFE factors are reused.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Fixed, GaussianLikelihood, Sgpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let fitted = Sgpr::new(
    ///     KernelSpec::from(RbfKernel::new(1.0)?),
    ///     GaussianLikelihood::new(0.1)?,
    /// )
    /// .with_optimizer(Fixed)
    /// .factor(
    ///     &[0.0, 1.0, 2.0, 3.0],
    ///     4,
    ///     1,
    ///     &[0.0, 1.0, 0.5, 0.25],
    ///     &[0.5, 2.5],
    ///     2,
    /// )
    /// .map_err(|(_, e)| e)?;
    /// let mut online = fitted.into_online();
    /// online.insert(&[4.0], 0.1)?;
    /// assert_eq!(online.n(), 5);
    /// # Ok(())
    /// # }
    /// ```
    pub fn into_online(self) -> OnlineSgpr<O, P> {
        OnlineSgpr::from_fitted(self)
    }

    /// Predicts at `xs` with [`PredictOptions::default`] (observation variance).
    ///
    /// `xs` is column-major with `n_rows` query points and `n_cols` features.
    /// Returns the diagonal VFE predictive mean and variance.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::DimensionMismatch`] if `n_cols` differs from the
    /// training features, [`GprError::EmptyInput`] if a dimension is zero, or
    /// [`GprError::LengthMismatch`] / [`GprError::NonFiniteInput`] for a
    /// badly packed or non-finite `xs`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Fixed, GaussianLikelihood, Sgpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    /// let likelihood = GaussianLikelihood::new(0.1)?;
    /// let fitted = Sgpr::new(kernel, likelihood)
    ///     .with_optimizer(Fixed)
    ///     .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0], &[0.0, 1.0], 2)
    ///     .map_err(|(_, e)| e)?;
    /// let pred = fitted.predict(&[0.5], 1, 1)?;
    /// assert_eq!(pred.mean.len(), 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn predict(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
    ) -> Result<Prediction<P::Refine>, GprError> {
        self.predict_with(xs, n_rows, n_cols, PredictOptions::default())
    }

    /// Predicts at `xs` with an explicit variance kind.
    ///
    /// Latent variance is the VFE predictive variance of `f*`. Observation
    /// variance adds `σn²`.
    ///
    /// # Errors
    ///
    /// Same as [`Self::predict`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Fixed, GaussianLikelihood, PredictOptions, Sgpr, VarianceKind};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    /// let likelihood = GaussianLikelihood::new(0.1)?;
    /// let fitted = Sgpr::new(kernel, likelihood)
    ///     .with_optimizer(Fixed)
    ///     .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0], &[0.0, 1.0], 2)
    ///     .map_err(|(_, e)| e)?;
    /// let pred = fitted.predict_with(
    ///     &[0.5],
    ///     1,
    ///     1,
    ///     PredictOptions {
    ///         variance_kind: VarianceKind::Latent,
    ///     },
    /// )?;
    /// assert_eq!(pred.variance_kind, VarianceKind::Latent);
    /// # Ok(())
    /// # }
    /// ```
    pub fn predict_with(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        options: PredictOptions,
    ) -> Result<Prediction<P::Refine>, GprError> {
        self.query(xs, n_rows, n_cols, &QueryDist::default(), options)
    }

    /// Returns the predictive mean and query–query covariance at `xs`.
    ///
    /// Default [`PredictOptions`] uses [`crate::VarianceKind::Observation`]:
    /// `σn²` is added on the diagonal in the transformed space. The diagonal
    /// is exactly the variance of [`Self::predict`] for the same query. This
    /// path allocates an `m × m` matrix; [`Self::predict`] stays diagonal.
    ///
    /// # Errors
    ///
    /// Same as [`Self::predict`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Fixed, GaussianLikelihood, Sgpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let fitted = Sgpr::new(
    ///     KernelSpec::from(RbfKernel::new(1.0)?),
    ///     GaussianLikelihood::new(0.1)?,
    /// )
    /// .with_optimizer(Fixed)
    /// .factor(&[0.0, 1.0, 2.0], 3, 1, &[0.0, 1.0, 0.5], &[0.5, 1.5], 2)
    /// .map_err(|(_, e)| e)?;
    /// let cov = fitted.predict_covariance(&[0.25, 0.75], 2, 1)?;
    /// assert_eq!(cov.covariance.len(), 4);
    /// assert_eq!(cov.covariance[0], fitted.predict(&[0.25, 0.75], 2, 1)?.variance[0]);
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
    /// The VFE posterior covariance is `K** − Q** + K*m Σ Km*` with
    /// `Q** = K*m K_mm⁻¹ Km*` and `Σ = (K_mm + σn⁻² K_mn K_nm)⁻¹`.
    /// Latent diagonals are clipped at 0. Observation adds `σn²` on the
    /// diagonal in the transformed space, then the target transform scales
    /// the whole matrix.
    ///
    /// # Errors
    ///
    /// Same as [`Self::predict`].
    ///
    /// See the example on [`Self::predict`].
    pub fn predict_covariance_with(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        options: PredictOptions,
    ) -> Result<PredictiveCovariance<P::Refine>, GprError> {
        self.query_covariance(xs, n_rows, n_cols, &QueryDist::default(), options)
    }

    /// Draws posterior samples at `xs` from [`Self::predict_covariance`].
    ///
    /// Each column of the returned column-major `m × n_draws` matrix is
    /// `μ + L z` with `z ∼ N(0, I)` and `L` the Cholesky factor of the
    /// posterior covariance, the same draw as [`crate::FittedGpr::sample`].
    /// `seed` is the start state of gprx's seeded generator (Xoshiro256++; the same seed gives the same draws on every platform). Zero draws
    /// returns an empty vector after the covariance is formed.
    ///
    /// # Errors
    ///
    /// Same as [`Self::predict`], plus [`GprError::CholeskyFailed`] with
    /// [`CholeskyStage::Predict`](crate::CholeskyStage::Predict) if the
    /// posterior covariance cannot be factored after the retries of
    /// [`Self::jitter_policy`].
    ///
    /// See the example on [`Self::predict`].
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
    ///
    /// See the example on [`Self::predict`].
    pub fn sample_with(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        options: PredictOptions,
        n_draws: usize,
        seed: u64,
    ) -> Result<Vec<P::Refine>, GprError> {
        self.predict_covariance_with(xs, n_rows, n_cols, options)?
            .draw(n_draws, seed, self.core.jitter)
    }

    /// Predicts at `xs` with [`PredictOptions::default`] into `out`.
    ///
    /// Same values as [`Self::predict`]. The buffers are kept on the model
    /// and `out` keeps its capacity, so a call after one with the same
    /// shapes allocates nothing.
    ///
    /// # Errors
    ///
    /// Same as [`Self::predict`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Fixed, GaussianLikelihood, Prediction, Sgpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let mut fitted = Sgpr::new(
    ///     KernelSpec::from(RbfKernel::new(1.0)?),
    ///     GaussianLikelihood::new(0.1)?,
    /// )
    /// .with_optimizer(Fixed)
    /// .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0], &[0.0, 1.0], 2)
    /// .map_err(|(_, e)| e)?;
    /// let mut pred = Prediction::default();
    /// fitted.predict_into(&[0.5], 1, 1, &mut pred)?;
    /// assert_eq!(pred, fitted.predict(&[0.5], 1, 1)?);
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

    /// Predicts at `xs` with an explicit variance kind into `out`.
    ///
    /// Same values as [`Self::predict_with`], with the buffers of
    /// [`Self::predict_into`].
    ///
    /// # Errors
    ///
    /// Same as [`Self::predict`].
    ///
    /// See the example on [`Self::predict`].
    pub fn predict_with_into(
        &mut self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        options: PredictOptions,
        out: &mut Prediction<P::Refine>,
    ) -> Result<(), GprError> {
        self.query_into(xs, n_rows, n_cols, &QueryDist::default(), options, out)
    }
}

impl<O, I: InducingLayout, P: crate::precision::GpScalar, K: PointKernel> FittedSgpr<O, I, P, K> {
    sparse_point_accessors!();
}

impl<O, I: InducingLayout, P: crate::precision::GpScalar, C: PointUse>
    FittedSgpr<O, I, P, DistanceKernel<C>>
{
    sparse_distance_accessors!();
}

impl<P: crate::precision::GpScalar, K: ModelKernel> FittedSgpr<Fixed, FixedInducing, P, K> {
    /// The model of a persist directory: the VFE system factored at the
    /// saved `θ` and `Z`, as [`Sgpr<Fixed>::factor`] does.
    ///
    /// # Errors
    ///
    /// Same as [`Sgpr<Fixed>::factor`].
    pub(crate) fn from_persisted(core: SparseCore<K>) -> Result<Self, GprError> {
        with_kernel_exp!(core.math, M => super::factor::assemble_fitted::<Fixed, FixedInducing, M, P, K>(
            core, Fixed
        ))
    }
}

pub(crate) fn promote_predict_w<P: ModelPrecision>(w: &[P::Storage]) -> Vec<P::Refine> {
    w.iter()
        .map(|value| P::Refine::from_f64(value.to_f64()))
        .collect()
}
