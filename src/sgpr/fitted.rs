//! Factored collapsed variational SGPR.

use std::marker::PhantomData;

use faer::Mat;
#[cfg(test)]
use faer::MatRef;

use crate::error::GprError;
use crate::param::write_params;

use crate::kernel::{KernelScalar, KernelSpec};
use crate::likelihood::GaussianLikelihood;
use crate::objective::SgprObjective;
use crate::optimizer::{Fixed, Lbfgs, OptResult, Optimizer};
use crate::param::Interval;
use crate::precision::{DoublePrecision, ModelPrecision};
use crate::{PredictOptions, Prediction};

use super::factor::{
    MeanDot, PublishSgprWeights, VfeState, analytic_gradient, analytic_hessian, assemble_vfe,
    fill_z_intervals, publish_sgpr_weights, vfe_neg_log_marginal_likelihood, vfe_predict,
};
use super::model::Sgpr;
use super::online::OnlineSgpr;
use super::{FixedInducing, InducingLayout};

/// Factored collapsed variational SGPR at the `θ` used by [`Sgpr::fit`] or
/// [`Sgpr<Fixed>::factor`].
///
/// Stores the LLT of `K_mm = k(Z, Z)` and the VFE factors used by
/// [`Self::predict`] and [`Self::neg_log_marginal_likelihood`]. Observation
/// noise is not added to `K_mm`. Hyperparameters are kernel `θ` then
/// likelihood `θ`. [`FreeInducing`] then appends column-major `Z`.
/// [`Self::into_online`] yields [`OnlineSgpr`] for training-point and
/// inducing-point updates.
#[allow(private_bounds)]
#[derive(Clone, Debug)]
pub struct FittedSgpr<
    O = Lbfgs,
    I = FixedInducing,
    M = crate::math::Accurate,
    P: ModelPrecision = DoublePrecision,
> {
    pub(crate) kernel: KernelSpec,
    pub(crate) likelihood: GaussianLikelihood,
    pub(crate) optimizer: O,
    pub(crate) inducing: PhantomData<I>,
    pub(crate) _math: PhantomData<M>,
    pub(crate) x_obs: Vec<f64>,
    pub(crate) z_obs: Vec<f64>,
    pub(crate) y: Vec<f64>,
    /// Lower `L` from `K_mm = L Lᵀ`.
    pub(crate) k_mm_l: Mat<P::Storage>,
    /// `A = L_mm⁻¹ K(Z, X)` (`m × n`).
    pub(crate) a: Mat<P::Storage>,
    /// Lower `L_B` from `B = σn² I + A Aᵀ`.
    pub(crate) b_l: Mat<P::Storage>,
    /// Storage solve `B w = A y`. Marginal likelihood uses this.
    pub(crate) w: Vec<P::Storage>,
    /// Predict weights. [`DoublePrecision`] and [`SinglePrecision`] promote `w`.
    /// [`MixedPrecision`] stores the refined `f64` weights.
    pub(crate) predict_w: Vec<P::Refine>,
    pub(crate) k_diag_sum: P::Storage,
    pub(crate) a_frobenius2: P::Storage,
    pub(crate) n: usize,
    pub(crate) m: usize,
    pub(crate) d: usize,
}

#[allow(private_bounds)]
impl<O, I: InducingLayout, M, P> FittedSgpr<O, I, M, P>
where
    M: crate::math::KernelMath,
    P: crate::precision::GpScalar + MeanDot + PublishSgprWeights,
{
    /// Returns the number of training points.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Returns the number of inducing points.
    pub fn m(&self) -> usize {
        self.m
    }

    /// Returns the feature dimension.
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

    /// Returns the original training features in column-major order.
    pub fn x(&self) -> &[f64] {
        &self.x_obs
    }

    /// Returns the inducing features in column-major order.
    pub fn z(&self) -> &[f64] {
        &self.z_obs
    }

    /// Returns the original training targets.
    pub fn y(&self) -> &[f64] {
        &self.y
    }

    /// Returns the concatenated parameter count.
    ///
    /// Kernel `θ` then likelihood `θ`. [`FreeInducing`] also counts
    /// column-major `Z` (`m × d`).
    pub fn num_params(&self) -> usize {
        self.kernel.num_params() + self.likelihood.num_params() + I::z_params(self.m, self.d)
    }

    /// Writes kernel `θ`, likelihood `θ`, and (when free) column-major `Z`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::LengthMismatch`] if `out` is the wrong length
    /// or a custom leaf rejects the write.
    pub fn get_params(&self, out: &mut [f64]) -> Result<(), GprError> {
        crate::data::require_count(out.len(), self.num_params(), "parameters")?;
        let n_theta = self.kernel.num_params() + self.likelihood.num_params();
        write_params(&self.kernel, &self.likelihood, &mut out[..n_theta])?;
        if I::z_params(self.m, self.d) > 0 {
            out[n_theta..].copy_from_slice(&self.z_obs);
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
    /// `params` matches [`Self::get_params`]. [`FreeInducing`] also writes
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
        let n_kernel = self.kernel.num_params();
        let n_theta = n_kernel + self.likelihood.num_params();
        crate::data::require_count(params.len(), self.num_params(), "parameters")?;
        if self.same_stored_params(params)? {
            return Ok(());
        }
        let mut kernel = self.kernel.clone();
        kernel.set_params(&params[..n_kernel])?;
        let mut likelihood = self.likelihood;
        likelihood.set_params(&params[n_kernel..n_theta])?;
        let z_obs = if I::z_params(self.m, self.d) > 0 {
            params[n_theta..].to_vec()
        } else {
            self.z_obs.clone()
        };
        let state = assemble_vfe::<M, _>(
            &kernel,
            likelihood,
            &self.x_obs,
            self.n,
            self.d,
            &self.y,
            &z_obs,
            self.m,
        )?;
        self.kernel = kernel;
        self.likelihood = likelihood;
        self.z_obs = z_obs;
        self.apply_vfe(state);
        Ok(())
    }

    pub(crate) fn fill_intervals(&self, out: &mut [Interval]) -> Result<(), GprError> {
        let n = self.num_params();
        if out.len() != n {
            return Err(GprError::LengthMismatch {
                reason: format!("expected {n} intervals, got {}", out.len()),
            });
        }
        let n_kernel = self.kernel.num_params();
        let n_theta = n_kernel + self.likelihood.num_params();
        let mut offset = 0;
        self.kernel
            .write_intervals(&mut out[..n_kernel], &mut offset)?;
        out[n_kernel] = self.likelihood.bounds();
        if I::z_params(self.m, self.d) > 0 {
            fill_z_intervals(&self.x_obs, self.n, self.d, &mut out[n_theta..])?;
        }
        Ok(())
    }

    /// Sets parameters, rebuilds the VFE system, and writes `∂L/∂θ` of the
    /// negative ELBO.
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
        let include_z = I::z_params(self.m, self.d) > 0;
        if !include_z && self.inducing_equals_training() {
            let mut exact = self.exact_fitted()?;
            exact.value_and_gradient_into(params, out)?;
        } else {
            analytic_gradient(self, out, include_z)?;
        }
        Ok(value)
    }

    /// Writes the Hessian of the negative ELBO (row-major `p×p`) into `out`.
    ///
    /// When `Z = X` this matches [`crate::FittedGpr::hessian_into`].
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
        let include_z = I::z_params(self.m, self.d) > 0;
        if !include_z && self.inducing_equals_training() {
            let mut exact = self.exact_fitted()?;
            exact.hessian_into(params, out)?;
        } else {
            analytic_hessian(self, out, include_z)?;
        }
        Ok(())
    }

    /// Converts this model into an online sparse GPR.
    ///
    /// [`OnlineSgpr`] can append or drop training points and inducing
    /// points. [`FixedInducing`] and [`FreeInducing`] both produce
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
    pub fn into_online(self) -> OnlineSgpr<O, M, P> {
        OnlineSgpr::from_fitted(self)
    }

    fn inducing_equals_training(&self) -> bool {
        self.x_obs == self.z_obs
    }

    fn exact_fitted(
        &self,
    ) -> Result<
        crate::FittedGpr<
            Fixed,
            crate::FullRecompute,
            crate::CachedDistances,
            crate::RetainCholesky,
            crate::Accurate,
            P,
        >,
        GprError,
    >
    where
        P: crate::precision::GpScalar,
    {
        crate::Gpr::new(self.kernel.clone(), self.likelihood)
            .with_optimizer(Fixed)
            .with_precision::<P>()
            .factor(&self.x_obs, self.n, self.d, &self.y)
            .map_err(|(_, e)| e)
    }

    pub(crate) fn refresh_predict_w(&mut self) -> Result<(), GprError> {
        self.predict_w = publish_sgpr_weights::<M, P>(
            &self.kernel,
            self.a.as_ref(),
            self.b_l.as_ref(),
            &self.w,
            &self.x_obs,
            &self.y,
            &self.z_obs,
            self.likelihood.noise_variance(),
            self.n,
            self.m,
            self.d,
        )?;
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

    pub(crate) fn into_trainer(self) -> Sgpr<O, I, M, P> {
        Sgpr {
            kernel: self.kernel,
            likelihood: self.likelihood,
            optimizer: self.optimizer,
            inducing: PhantomData,
            _math: PhantomData,
            _precision: PhantomData,
        }
    }

    pub(crate) fn optimize_hyperparameters(&mut self) -> Result<(), GprError>
    where
        O: Clone + for<'a> Optimizer<SgprObjective<'a, O, I, M, P>>,
    {
        let mut init = vec![0.0; self.num_params()];
        self.get_params(&mut init)?;
        let before = self.clone();
        let optimizer = self.optimizer.clone();
        let result = {
            let mut obj = SgprObjective::new(self);
            optimizer.minimize(&mut obj, &init)
        };
        self.commit_or_revert_optimize(before, result)
    }

    fn commit_or_revert_optimize(
        &mut self,
        before: Self,
        result: Result<OptResult, GprError>,
    ) -> Result<(), GprError> {
        match result {
            Ok(opt) => {
                if opt.params.len() != self.num_params() || !opt.value.is_finite() {
                    *self = before;
                    return Err(GprError::OptimizationNotConverged {
                        iterations: opt.iterations as usize,
                    });
                }
                if let Err(err) = self.set_params(&opt.params) {
                    *self = before;
                    return Err(err);
                }
                if let Err(err) = self.refresh_predict_w() {
                    *self = before;
                    return Err(err);
                }
                Ok(())
            }
            Err(err) => {
                *self = before;
                Err(err)
            }
        }
    }

    /// Returns the negative VFE evidence lower bound (the sparse NLML).
    ///
    /// When `Z = X` this matches [`crate::FittedGpr::neg_log_marginal_likelihood`]
    /// of [`crate::Gpr<Fixed>::factor`] on the same data.
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
            &self.y,
            self.k_diag_sum,
            self.a_frobenius2,
            self.likelihood.noise_variance(),
            self.n,
            self.m,
        )
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
        if n_cols != self.d {
            return Err(GprError::DimensionMismatch {
                x_dim: n_cols,
                expected_dim: self.d,
            });
        }
        vfe_predict::<M, P>(
            &self.kernel,
            &self.z_obs,
            self.k_mm_l.as_ref(),
            self.b_l.as_ref(),
            &self.predict_w,
            self.likelihood.noise_variance(),
            self.m,
            self.d,
            xs,
            n_rows,
            n_cols,
            options,
        )
    }

    #[cfg(test)]
    pub(crate) fn k_mm_l(&self) -> MatRef<'_, P::Storage> {
        self.k_mm_l.as_ref()
    }
}

pub(crate) fn promote_predict_w<P: ModelPrecision>(w: &[P::Storage]) -> Vec<P::Refine> {
    w.iter()
        .map(|value| P::Refine::from_f64(value.to_f64()))
        .collect()
}
