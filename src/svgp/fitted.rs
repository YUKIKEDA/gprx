//! Factored stochastic variational GPR.

use faer::Mat;

use crate::error::GprError;
use crate::policy::with_kernel_exp;
use crate::sparse::{PredictScratch, SparseCore, SparseScratch, sparse_core_accessors};

use crate::precision::{DoublePrecision, GpScalar, ModelPrecision};
use crate::{PredictOptions, Prediction, PredictiveCovariance};

use super::factor::{
    SvgpSystem, assemble_data_terms, assemble_kmm, assemble_svgp, pack_q, predict_svgp_covariance,
    predict_svgp_into, q_param_len, svgp_neg_elbo, svgp_value_and_gradient, unpack_q,
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

    /// The model of a persist directory: `K_mm` and `A` factored at the
    /// saved `θ` and `Z`, with the saved whitened `q(u)`.
    ///
    /// # Errors
    ///
    /// Same as [`crate::Svgp<crate::Fixed>::factor`].
    pub(crate) fn from_persisted(
        core: SparseCore,
        q_mean: Vec<f64>,
        q_l: Mat<f64>,
    ) -> Result<Self, GprError> {
        with_kernel_exp!(core.math, M => super::factor::assemble_fitted::<M, P>(
            core,
            Some((q_mean, q_l))
        ))
    }

    /// Writes this model to `dir` as `config.json` and `model.safetensors`.
    ///
    /// Stores the kernel, likelihood, kernel `exp`, `K_mm` jitter policy,
    /// precision, transforms (unfitted and fitted), the original `X`, `y`,
    /// and `Z`, and `Z` in transformed coordinates, and the whitened `q(u)`. The factors are
    /// not stored; [`crate::LoadedSvgp::load`] factors the system again at the saved `θ`
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
    /// use gprx::{GaussianLikelihood, Svgp};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let model = Svgp::new(KernelSpec::from(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?)
    ///     .factor(&[0.0, 1.0, 2.0], 3, 1, &[0.0, 1.0, 0.5], &[0.5, 1.5], 2)
    ///     .map_err(|(_, e)| e)?;
    /// let dir = std::env::temp_dir().join(format!("gprx-doctest-save-svgp-{}", std::process::id()));
    /// let _ = std::fs::remove_dir_all(&dir);
    /// model.save(&dir)?;
    /// let loaded = gprx::LoadedSvgp::load(&dir, &gprx::PersistRegistry::new())?;
    /// assert_eq!(loaded.n(), model.n());
    /// let _ = std::fs::remove_dir_all(&dir);
    /// # Ok(())
    /// # }
    /// ```
    pub fn save(&self, dir: impl AsRef<std::path::Path>) -> Result<(), GprError> {
        crate::persist::save_svgp(self, dir.as_ref())
    }

    /// The training data, settings, and fitted transforms.
    pub(crate) fn core(&self) -> &SparseCore {
        &self.core
    }

    /// The whitened variational mean and lower `L` of `q(u)`.
    pub(crate) fn q(&self) -> (&[f64], faer::MatRef<'_, f64>) {
        (&self.q_mean, self.q_l.as_ref())
    }

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
            self.core.jitter,
            &self.core.x_train,
            self.core.n,
            self.core.d,
            &self.core.y_train,
            &self.core.z_train,
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

    /// One mini-batch step's update: `θ`, the factor of `K_mm`, and `q`.
    ///
    /// Leaves `A` and `k_diag`, which cost `O(n)`, stale until
    /// [`Self::rebuild_data_terms`]: a mini-batch gradient forms the terms of
    /// its own points and never reads them. Committed together only after
    /// `K_mm` factors, like [`Self::set_params`].
    pub(crate) fn set_params_light(&mut self, params: &[f64]) -> Result<(), GprError> {
        let n_theta = self.core.theta_len();
        crate::data::require_count(params.len(), self.num_params(), "parameters")?;
        let (kernel, likelihood) = self.core.stage_theta(&params[..n_theta])?;
        let (q_mean, q_l) = unpack_q(&params[n_theta..], self.core.m)?;
        let k_mm_l = with_kernel_exp!(self.core.math, M => assemble_kmm::<M, P::Storage>(
            &kernel,
            self.core.jitter,
            &self.core.z_train,
            self.core.m,
            self.core.d,
            &mut self.scratch.storage,
        ))?;
        self.core.kernel = kernel;
        self.core.likelihood = likelihood;
        self.k_mm_l = k_mm_l;
        self.q_mean = q_mean;
        self.q_l = q_l;
        Ok(())
    }

    /// `A` and `k_diag` for every training point at the stored `θ`, `Z`, and
    /// `K_mm` factor (after [`Self::set_params_light`] steps).
    pub(crate) fn rebuild_data_terms(&mut self) -> Result<(), GprError> {
        let (a, k_diag) = with_kernel_exp!(self.core.math, M => assemble_data_terms::<M, P::Storage>(
            &self.core.kernel,
            &self.core.x_train,
            self.core.n,
            self.core.d,
            &self.core.z_train,
            self.core.m,
            self.k_mm_l.as_ref(),
            &mut self.scratch.storage,
        ))?;
        self.a = a;
        self.k_diag = k_diag;
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
            &self.core.y_train,
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
        let mut out = Prediction::default();
        predict_svgp_into::<P>(
            &self.core,
            &SvgpSystem::new(
                &self.core,
                self.k_mm_l.as_ref(),
                &self.q_mean,
                self.q_l.as_ref(),
            ),
            xs,
            n_rows,
            n_cols,
            options,
            &mut PredictScratch::default(),
            &mut out,
        )?;
        Ok(out)
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
    /// use gprx::{GaussianLikelihood, Svgp};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let fitted = Svgp::new(
    ///     KernelSpec::from(RbfKernel::new(1.0)?),
    ///     GaussianLikelihood::new(0.1)?,
    /// )
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
    /// The SVGP posterior covariance is `K** − K*m K_mm⁻¹ Km* + K*m L⁻ᵀ S L⁻¹ Km*`
    /// with the whitened `q(u) = N(m, S)` and `K_mm = L Lᵀ`.
    /// Latent diagonals are clipped at 0. Observation adds `σn²` on the
    /// diagonal in the transformed space, then the target transform scales
    /// the whole matrix.
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
        predict_svgp_covariance::<P>(
            &self.core,
            &SvgpSystem::new(
                &self.core,
                self.k_mm_l.as_ref(),
                &self.q_mean,
                self.q_l.as_ref(),
            ),
            xs,
            n_rows,
            n_cols,
            options,
        )
    }

    /// Draws posterior samples at `xs` from [`Self::predict_covariance`].
    ///
    /// Each column of the returned column-major `m × n_draws` matrix is
    /// `μ + L z` with `z ∼ N(0, I)` and `L` the Cholesky factor of the
    /// posterior covariance, the same draw as [`crate::FittedGpr::sample`].
    /// `seed` is the crate [`rand::rngs::SmallRng`] start state. Zero draws
    /// returns an empty vector after the covariance is formed.
    ///
    /// # Errors
    ///
    /// Same as [`Self::predict`], plus [`GprError::CholeskyFailed`] with
    /// [`CholeskyStage::Predict`](crate::CholeskyStage::Predict) if the
    /// posterior covariance cannot be factored after the retries of
    /// [`Self::jitter_policy`].
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
        self.predict_covariance_with(xs, n_rows, n_cols, options)?
            .draw(n_draws, seed, self.core.jitter)
    }

    /// Predicts at `xs` with [`PredictOptions::default`] into `out`.
    ///
    /// Same values as [`Self::predict`]. The buffers are kept on the model
    /// and `out` keeps its capacity, so a call after one with the same
    /// shapes allocates nothing, except under [`crate::MixedPrecision`],
    /// which refines each mean in `f64`.
    ///
    /// # Errors
    ///
    /// Same as [`Self::predict`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{GaussianLikelihood, Prediction, Svgp};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let mut fitted = Svgp::new(
    ///     KernelSpec::from(RbfKernel::new(1.0)?),
    ///     GaussianLikelihood::new(0.1)?,
    /// )
    /// .factor(&[0.0, 1.0, 2.0, 3.0], 4, 1, &[0.0, 1.0, 0.5, 0.25], &[0.5, 2.5], 2)
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
    pub fn predict_with_into(
        &mut self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        options: PredictOptions,
        out: &mut Prediction<P::Refine>,
    ) -> Result<(), GprError> {
        predict_svgp_into::<P>(
            &self.core,
            &SvgpSystem::new(
                &self.core,
                self.k_mm_l.as_ref(),
                &self.q_mean,
                self.q_l.as_ref(),
            ),
            xs,
            n_rows,
            n_cols,
            options,
            &mut self.scratch.predict,
            out,
        )
    }
}
