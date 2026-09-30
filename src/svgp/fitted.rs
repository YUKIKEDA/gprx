//! Factored stochastic variational GPR.

use faer::Mat;

use crate::error::GprError;
use crate::policy::with_kernel_exp;
use crate::sparse::{SparseCore, SparseScratch, sparse_core_accessors};

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
    pub(super) core: SparseCore,
    /// Kernel scratch kept between `&mut self` calls.
    pub(super) scratch: SparseScratch<P::Storage>,
    /// Lower `L_mm` from `K_mm = L_mm L_mmᵀ`.
    pub(super) k_mm_l: Mat<P::Storage>,
    /// `A = L_mm⁻¹ K(Z, X)` (`m × n`).
    pub(super) a: Mat<P::Storage>,
    pub(super) q_mean: Vec<f64>,
    /// Lower `L` from the whitened `S = L Lᵀ`.
    pub(super) q_l: Mat<f64>,
    pub(super) k_diag: Vec<P::Storage>,
}

impl<P> FittedSvgp<P>
where
    P: GpScalar,
{
    sparse_core_accessors!();

    /// Returns the concatenated parameter count.
    ///
    /// Kernel `θ`, likelihood `θ`, the whitened mean (`m` scalars), then the
    /// packed lower triangle of `L` (`m(m + 1) / 2` scalars).
    pub fn num_params(&self) -> usize {
        self.core.theta_len() + q_param_len(self.core.m)
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
        let n_theta = self.core.theta_len();
        self.core.read_theta(&mut out[..n_theta])?;
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
        let n_theta = self.core.theta_len();
        crate::data::require_count(params.len(), self.num_params(), "parameters")?;
        if self.same_stored_params(params)? {
            return Ok(());
        }
        let (kernel, likelihood) = self.core.stage_theta(&params[..n_theta])?;
        let q = unpack_q(&params[n_theta..], self.core.m)?;
        let state = with_kernel_exp!(self.core.math, M => assemble_svgp::<M, P::Storage>(
            &kernel,
            &self.core.x_obs,
            self.core.n,
            self.core.d,
            &self.core.y,
            &self.core.z_obs,
            self.core.m,
            Some(q),
            &mut self.scratch.storage,
        ))?;
        self.core.kernel = kernel;
        self.core.likelihood = likelihood;
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
            &self.core.y,
            &self.k_diag,
            self.core.likelihood.noise_variance(),
            self.core.n,
            self.core.m,
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
        let batch: Vec<usize> = (0..self.core.n).collect();
        let mut scratch = std::mem::take(&mut self.scratch);
        let result = with_kernel_exp!(
            self.core.math,
            M => svgp_value_and_gradient::<M, _>(self, out, &batch, &mut scratch)
        );
        self.scratch = scratch;
        result
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
        if n_cols != self.core.d {
            return Err(GprError::DimensionMismatch {
                x_dim: n_cols,
                expected_dim: self.core.d,
            });
        }
        with_kernel_exp!(self.core.math, M => svgp_predict::<M, P>(
            &self.core.kernel,
            &self.core.z_obs,
            self.k_mm_l.as_ref(),
            &self.q_mean,
            self.q_l.as_ref(),
            self.core.likelihood.noise_variance(),
            self.core.m,
            self.core.d,
            xs,
            n_rows,
            n_cols,
            options,
        ))
    }
}
