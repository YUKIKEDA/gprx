//! Factored stochastic variational GPR.

use faer::Mat;

use crate::error::GprError;
use crate::gpr::{KernelExp, with_kernel_exp};
use crate::param::write_params;

use crate::kernel::KernelSpec;
use crate::likelihood::GaussianLikelihood;
use crate::precision::{DoublePrecision, GpScalar, ModelPrecision};
use crate::{PredictOptions, Prediction};

use super::factor::{
    assemble_svgp, pack_q, q_param_len, svgp_neg_elbo, svgp_predict, svgp_value_and_gradient,
    unpack_q,
};

/// Factored stochastic variational GPR at the `θ` used by [`crate::Svgp<Fixed>::factor`]
/// or [`crate::Svgp<crate::Adam>::fit`].
///
/// Stores the LLT of `K_mm = k(Z, Z)` and a whitened variational posterior
/// `q(v) = N(m, L Lᵀ)` used by [`Self::predict`] and [`Self::neg_elbo`].
/// Observation noise is not added to `K_mm`. Hyperparameters are kernel `θ`,
/// likelihood `θ`, the whitened mean vector, then the packed column-major
/// lower triangle of `L`.
#[derive(Clone, Debug)]
pub struct FittedSvgp<P: ModelPrecision = DoublePrecision> {
    pub(crate) kernel: KernelSpec,
    pub(crate) likelihood: GaussianLikelihood,
    pub(crate) x_obs: Vec<f64>,
    pub(crate) z_obs: Vec<f64>,
    pub(crate) y: Vec<f64>,
    /// Lower `L_mm` from `K_mm = L_mm L_mmᵀ`.
    pub(crate) k_mm_l: Mat<P::Storage>,
    /// `A = L_mm⁻¹ K(Z, X)` (`m × n`).
    pub(crate) a: Mat<P::Storage>,
    pub(crate) q_mean: Vec<f64>,
    /// Lower `L` from the whitened `S = L Lᵀ`.
    pub(crate) q_l: Mat<f64>,
    pub(crate) k_diag: Vec<P::Storage>,
    pub(crate) n: usize,
    pub(crate) m: usize,
    pub(crate) d: usize,
    pub(crate) math: KernelExp,
}

impl<P> FittedSvgp<P>
where
    P: GpScalar,
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

    /// Returns the kernel `exp` mode the trainer set with `with_math`.
    pub fn math(&self) -> KernelExp {
        self.math
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
    /// Kernel `θ`, likelihood `θ`, the whitened mean (`m` scalars), then the
    /// packed lower triangle of `L` (`m(m + 1) / 2` scalars).
    pub fn num_params(&self) -> usize {
        self.kernel.num_params() + self.likelihood.num_params() + q_param_len(self.m)
    }

    /// Writes kernel `θ`, likelihood `θ`, the whitened mean, and packed `L`.
    ///
    /// `L` is packed column-major lower triangle: column 0 (`L_{00}`,
    /// `L_{10}`, …), then column 1 (`L_{11}`, `L_{21}`, …), and so on.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::LengthMismatch`] if `out` is the wrong length
    /// or a custom leaf rejects the write.
    pub fn get_params(&self, out: &mut [f64]) -> Result<(), GprError> {
        crate::data::require_count(out.len(), self.num_params(), "parameters")?;
        let n_theta = self.kernel.num_params() + self.likelihood.num_params();
        write_params(&self.kernel, &self.likelihood, &mut out[..n_theta])?;
        pack_q(&self.q_mean, self.q_l.as_ref(), &mut out[n_theta..]);
        Ok(())
    }

    fn same_stored_params(&self, params: &[f64]) -> Result<bool, GprError> {
        let mut current = vec![0.0; params.len()];
        self.get_params(&mut current)?;
        Ok(current == params)
    }

    /// Sets kernel then likelihood `θ` and the whitened `q`, then rebuilds.
    ///
    /// `params` matches [`Self::get_params`]. Training `X` / `y` and inducing
    /// `Z` are not changed. The diagonal of `L` must be strictly positive.
    /// Values are committed together only after `K_mm` factors.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::LengthMismatch`] if `params` is the wrong
    /// length or an `L` diagonal entry is not positive,
    /// [`GprError::NonFiniteInput`] if a variational value is not finite,
    /// [`GprError::InvalidNoiseVariance`] if the likelihood `θ` is invalid, or
    /// [`GprError::CholeskyFailed`] if `K_mm` cannot be factored. A rejected
    /// slice or a Cholesky failure leaves stored `θ` and `q` unchanged.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{GaussianLikelihood, Svgp};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let mut fitted = Svgp::new(
    ///     KernelSpec::from(RbfKernel::new(1.0)?),
    ///     GaussianLikelihood::new(0.1)?,
    /// )
    /// .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0], &[0.0, 1.0], 2)
    /// .map_err(|(_, e)| e)?;
    /// let mut params = [0.0; 7];
    /// fitted.get_params(&mut params)?;
    /// params[2] = 0.1;
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
        let q = unpack_q(&params[n_theta..], self.m)?;
        let state = with_kernel_exp!(self.math, M => assemble_svgp::<M, P::Storage>(
            &kernel,
            &self.x_obs,
            self.n,
            self.d,
            &self.y,
            &self.z_obs,
            self.m,
            Some(q),
        ))?;
        self.kernel = kernel;
        self.likelihood = likelihood;
        self.k_mm_l = state.k_mm_l;
        self.a = state.a;
        self.q_mean = state.q_mean;
        self.q_l = state.q_l;
        self.k_diag = state.k_diag;
        Ok(())
    }

    /// Returns the negative variational ELBO on the full training set.
    ///
    /// This is the un-collapsed bound. At the Titsias-optimal whitened `q`
    /// it matches [`crate::FittedSgpr::neg_log_marginal_likelihood`] on
    /// the same `θ`, `X`, and `Z`.
    ///
    /// # Errors
    ///
    /// The stored factors are already valid, so this returns `Ok`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{GaussianLikelihood, Svgp};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    /// let likelihood = GaussianLikelihood::new(0.1)?;
    /// let fitted = Svgp::new(kernel, likelihood)
    ///     .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0], &[0.0, 1.0], 2)
    ///     .map_err(|(_, e)| e)?;
    /// let value = fitted.neg_elbo()?;
    /// assert!(value.is_finite());
    /// # Ok(())
    /// # }
    /// ```
    pub fn neg_elbo(&self) -> Result<f64, GprError> {
        Ok(svgp_neg_elbo(
            self.a.as_ref(),
            &self.q_mean,
            self.q_l.as_ref(),
            &self.y,
            &self.k_diag,
            self.likelihood.noise_variance(),
            self.n,
            self.m,
        ))
    }

    /// Sets parameters, rebuilds `K_mm` / `q`, and writes `∂/∂θ` of the
    /// full-data negative ELBO.
    ///
    /// `params` and `out` match [`Self::get_params`]. The returned value is
    /// the same as [`Self::neg_elbo`] after a successful call. Mini-batch
    /// scaling lives in [`crate::Svgp<crate::Adam>::fit`], not here.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::LengthMismatch`] if a slice length is wrong
    /// or an `L` diagonal entry is not positive,
    /// [`GprError::NonFiniteInput`] if a variational value is not finite,
    /// [`GprError::InvalidNoiseVariance`] if the likelihood `θ` is invalid,
    /// or [`GprError::CholeskyFailed`] if `K_mm` cannot be factored.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{GaussianLikelihood, Svgp};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let mut fitted = Svgp::new(
    ///     KernelSpec::from(RbfKernel::new(1.0)?),
    ///     GaussianLikelihood::new(0.1)?,
    /// )
    /// .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0], &[0.0, 1.0], 2)
    /// .map_err(|(_, e)| e)?;
    /// let mut params = [0.0; 7];
    /// fitted.get_params(&mut params)?;
    /// let mut grad = [0.0; 7];
    /// let value = fitted.value_and_gradient_into(&params, &mut grad)?;
    /// assert!(value.is_finite());
    /// # Ok(())
    /// # }
    /// ```
    pub fn value_and_gradient_into(
        &mut self,
        params: &[f64],
        out: &mut [f64],
    ) -> Result<f64, GprError> {
        let n_params = self.num_params();
        crate::data::require_count(params.len(), n_params, "parameters")?;
        crate::data::require_count(out.len(), n_params, "parameters")?;
        self.set_params(params)?;
        let batch: Vec<usize> = (0..self.n).collect();
        with_kernel_exp!(self.math, M => svgp_value_and_gradient::<M, _>(self, out, &batch))
    }

    /// Predicts at `xs` with [`PredictOptions::default`] (observation variance).
    ///
    /// `xs` is column-major with `n_rows` query points and `n_cols` features.
    /// Returns the diagonal SVGP predictive mean and variance.
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
    /// use gprx::{GaussianLikelihood, Svgp};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    /// let likelihood = GaussianLikelihood::new(0.1)?;
    /// let fitted = Svgp::new(kernel, likelihood)
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
    /// Latent variance is the SVGP predictive variance of `f*`. Observation
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
    /// use gprx::{GaussianLikelihood, PredictOptions, Svgp, VarianceKind};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    /// let likelihood = GaussianLikelihood::new(0.1)?;
    /// let fitted = Svgp::new(kernel, likelihood)
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
        with_kernel_exp!(self.math, M => svgp_predict::<M, P>(
            &self.kernel,
            &self.z_obs,
            self.k_mm_l.as_ref(),
            &self.q_mean,
            self.q_l.as_ref(),
            self.likelihood.noise_variance(),
            self.m,
            self.d,
            xs,
            n_rows,
            n_cols,
            options,
        ))
    }
}
