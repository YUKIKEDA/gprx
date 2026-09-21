//! Variational sparse GPR with caller-supplied inducing points.

use std::marker::PhantomData;

use dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt;
use faer::{Mat, MatMut, MatRef};

use crate::error::{CholeskyStage, GprError};
use crate::gpr::JitterPolicy;
use crate::gpr::factor::{
    cholesky_lower_with_policy, pack_points, pack_points_into, require_param_len, symmetrize_lower,
    validate_query, validate_training, write_params,
};
use crate::kernel::{
    CompiledKernel, CoordMode, KernelSpec, Triangle, fill_squared_euclidean_cross,
};
use crate::likelihood::GaussianLikelihood;
use crate::objective::SparseGprObjective;
use crate::optimizer::{Fixed, Lbfgs, OptResult, Optimizer};
use crate::param::Interval;
use crate::workspace::{faer_par, faer_par_dims};
use crate::{PredictOptions, Prediction, VarianceKind};

/// Keeps inducing coordinates fixed during [`SparseGpr::fit`].
///
/// `Z` is an argument of `fit` / `factor` and is not a parameter.
#[derive(Clone, Copy, Debug, Default)]
pub struct FixedInducing;

/// Optimizes inducing coordinates jointly with kernel and likelihood `θ`.
///
/// Switch with [`SparseGpr::with_inducing`]. Params append column-major `Z`
/// after kernel and likelihood `θ`.
#[derive(Clone, Copy, Debug, Default)]
pub struct FreeInducing;

pub(crate) trait InducingLayout: Clone {
    fn z_params(m: usize, d: usize) -> usize;
}

impl InducingLayout for FixedInducing {
    fn z_params(_m: usize, _d: usize) -> usize {
        0
    }
}

impl InducingLayout for FreeInducing {
    fn z_params(m: usize, d: usize) -> usize {
        m * d
    }
}

/// Trainer for variational sparse GPR at a caller-supplied inducing set `Z`.
///
/// The default [`FixedInducing`] searches kernel and likelihood `θ` only.
/// [`Self::with_inducing`]`(`[`FreeInducing`]`)` searches `θ` and `Z`
/// together. [`SparseGpr<Fixed, I>::factor`] prepares the VFE system at the
/// current `θ` with no search.
///
/// # Examples
///
/// ```rust
/// use gprx::kernel::{KernelSpec, RbfKernel};
/// use gprx::{GaussianLikelihood, SparseGpr};
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
/// let likelihood = GaussianLikelihood::new(0.1)?;
/// let fitted = SparseGpr::new(kernel, likelihood)
///     .fit(&[0.0, 1.0, 2.0, 3.0], 4, 1, &[0.0, 1.0, 0.5, 0.25], &[0.5, 2.5], 2)
///     .map_err(|(_, e)| e)?;
/// assert_eq!(fitted.n(), 4);
/// assert_eq!(fitted.m(), 2);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct SparseGpr<O = Lbfgs, I = FixedInducing> {
    kernel: KernelSpec,
    likelihood: GaussianLikelihood,
    optimizer: O,
    inducing: PhantomData<I>,
}

/// Factored variational sparse GPR at the `θ` used by [`SparseGpr::fit`] or
/// [`SparseGpr<Fixed>::factor`].
///
/// Stores the LLT of `K_mm = k(Z, Z)` and the VFE factors used by
/// [`Self::predict`] and [`Self::neg_log_marginal_likelihood`]. Observation
/// noise is not added to `K_mm`. Hyperparameters are kernel `θ` then
/// likelihood `θ`. [`FreeInducing`] then appends column-major `Z`.
#[derive(Clone, Debug)]
pub struct FittedSparseGpr<O = Lbfgs, I = FixedInducing> {
    kernel: KernelSpec,
    likelihood: GaussianLikelihood,
    optimizer: O,
    inducing: PhantomData<I>,
    x_obs: Vec<f64>,
    z_obs: Vec<f64>,
    y: Vec<f64>,
    /// Lower `L` from `K_mm = L Lᵀ`.
    k_mm_l: Mat<f64>,
    /// `A = L_mm⁻¹ K(Z, X)` (`m × n`).
    a: Mat<f64>,
    /// Lower `L_B` from `B = σn² I + A Aᵀ`.
    b_l: Mat<f64>,
    /// `B⁻¹ A y`.
    w: Vec<f64>,
    k_diag_sum: f64,
    a_frobenius2: f64,
    n: usize,
    m: usize,
    d: usize,
}

impl SparseGpr {
    /// Builds a trainer with identity transforms, the current kernel `θ`, and
    /// [`Lbfgs`].
    ///
    /// Inducing coordinates are an argument of [`SparseGpr::fit`] /
    /// [`SparseGpr<Fixed>::factor`], not of this constructor. Call
    /// [`Self::with_optimizer`] to switch to [`Fixed`] or another
    /// [`Optimizer`].
    pub fn new(kernel: KernelSpec, likelihood: GaussianLikelihood) -> Self {
        Self {
            kernel,
            likelihood,
            optimizer: Lbfgs::new(),
            inducing: PhantomData,
        }
    }
}

impl<O, I> SparseGpr<O, I> {
    /// Replaces the optimizer type parameter.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Fixed, GaussianLikelihood, SparseGpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    /// let likelihood = GaussianLikelihood::new(0.1)?;
    /// let fitted = SparseGpr::new(kernel, likelihood)
    ///     .with_optimizer(Fixed)
    ///     .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0], &[0.0, 1.0], 2)
    ///     .map_err(|(_, e)| e)?;
    /// assert_eq!(fitted.m(), fitted.n());
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_optimizer<O2>(self, optimizer: O2) -> SparseGpr<O2, I> {
        SparseGpr {
            kernel: self.kernel,
            likelihood: self.likelihood,
            optimizer,
            inducing: PhantomData,
        }
    }

    /// Replaces the inducing-point type parameter.
    ///
    /// [`FixedInducing`] (the default) keeps `Z` fixed. [`FreeInducing`]
    /// appends column-major `Z` to the parameter vector and searches it with
    /// kernel and likelihood `θ`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{FreeInducing, GaussianLikelihood, SparseGpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    /// let likelihood = GaussianLikelihood::new(0.1)?;
    /// let fitted = SparseGpr::new(kernel, likelihood)
    ///     .with_inducing(FreeInducing)
    ///     .fit(&[0.0, 1.0, 2.0, 3.0], 4, 1, &[0.0, 1.0, 0.5, 0.25], &[0.5, 2.5], 2)
    ///     .map_err(|(_, e)| e)?;
    /// assert_eq!(fitted.num_params(), 4);
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_inducing<I2>(self, _inducing: I2) -> SparseGpr<O, I2> {
        SparseGpr {
            kernel: self.kernel,
            likelihood: self.likelihood,
            optimizer: self.optimizer,
            inducing: PhantomData,
        }
    }

    /// Returns the kernel whose hyperparameters this trainer owns.
    pub fn kernel(&self) -> &KernelSpec {
        &self.kernel
    }

    /// Returns the observation-noise model.
    pub fn likelihood(&self) -> &GaussianLikelihood {
        &self.likelihood
    }

    /// Returns the concatenated kernel and likelihood parameter count.
    ///
    /// Inducing coordinates are not counted.
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
        write_params(&self.kernel, &self.likelihood, out)
    }

    /// Sets kernel then likelihood `θ` without forming the VFE system.
    ///
    /// `params` is kernel parameters followed by the likelihood parameter,
    /// matching [`Self::get_params`]. Inducing coordinates are not in this
    /// slice.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if `params` is the wrong
    /// length, or [`GprError::InvalidNoiseVariance`] if the likelihood `θ`
    /// is invalid. Kernel and likelihood `θ` are committed together only
    /// after both writes succeed.
    pub fn set_params(&mut self, params: &[f64]) -> Result<(), GprError> {
        let n_kernel = self.kernel.num_params();
        require_param_len(params.len(), self.num_params())?;
        let mut kernel = self.kernel.clone();
        kernel.set_params(&params[..n_kernel])?;
        let mut likelihood = self.likelihood;
        likelihood.set_params(&params[n_kernel..])?;
        self.kernel = kernel;
        self.likelihood = likelihood;
        Ok(())
    }
}

#[allow(private_bounds)] // `SparseGprObjective` is crate-private; `fit` still needs `O: Optimizer` for it.
impl<O> SparseGpr<O, FixedInducing>
where
    O: Clone + for<'a> Optimizer<SparseGprObjective<'a, O, FixedInducing>>,
{
    /// Factors the VFE system and searches kernel and likelihood `θ`.
    ///
    /// `x` and `z` are column-major (`n` / `m` points by `d` features). `Z`
    /// is supplied by the caller and is not moved. Likelihood noise is not
    /// added to `K_mm`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::EmptyInput`] when `n`, `d`, or `m` is zero.
    /// Returns [`GprError::DimensionMismatch`] when `z` is packed with a
    /// different feature count than `x` (the same `n_cols` is required).
    /// Length and finiteness errors match [`crate::Gpr::fit`].
    /// [`GprError::CholeskyFailed`] when `K_mm` or the VFE `B` matrix cannot
    /// be factored. [`GprError::OptimizationNotConverged`] when the solver
    /// stops without a finite best vector.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{GaussianLikelihood, SparseGpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    /// let likelihood = GaussianLikelihood::new(0.1)?;
    /// let fitted = SparseGpr::new(kernel, likelihood)
    ///     .fit(&[0.0, 1.0, 2.0, 3.0], 4, 1, &[0.0, 1.0, 0.5, 0.25], &[0.5, 2.5], 2)
    ///     .map_err(|(_, e)| e)?;
    /// let nlml = fitted.neg_log_marginal_likelihood()?;
    /// assert!(nlml.is_finite());
    /// # Ok(())
    /// # }
    /// ```
    #[allow(clippy::result_large_err)] // failure returns the trainer so the caller can retry
    pub fn fit(
        self,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
        y: &[f64],
        z: &[f64],
        n_inducing: usize,
    ) -> Result<FittedSparseGpr<O>, (Self, GprError)> {
        match assemble_fitted(
            self.kernel.clone(),
            self.likelihood,
            self.optimizer.clone(),
            x,
            n_rows,
            n_cols,
            y,
            z,
            n_inducing,
        ) {
            Ok(mut fitted) => match fitted.optimize_hyperparameters() {
                Ok(()) => Ok(fitted),
                Err(err) => Err((fitted.into_trainer(), err)),
            },
            Err(err) => Err((self, err)),
        }
    }
}

#[allow(private_bounds)]
impl<O> SparseGpr<O, FreeInducing>
where
    O: Clone + for<'a> Optimizer<SparseGprObjective<'a, O, FreeInducing>>,
{
    /// Factors the VFE system and searches kernel `θ`, likelihood `θ`, and `Z`.
    ///
    /// `x` and `z` are column-major. The initial `Z` is the start of the
    /// joint search. Params are kernel `θ`, likelihood `θ`, then column-major
    /// `Z`. Coordinate intervals are the training box opened by a slack of
    /// 10% of each feature range, at least `0.1`.
    ///
    /// # Errors
    ///
    /// Same input errors as [`SparseGpr<O, FixedInducing>::fit`], plus
    /// [`GprError::CoordGradientUnsupported`] when the kernel has no
    /// coordinate derivative.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{FreeInducing, GaussianLikelihood, SparseGpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    /// let likelihood = GaussianLikelihood::new(0.1)?;
    /// let fitted = SparseGpr::new(kernel, likelihood)
    ///     .with_inducing(FreeInducing)
    ///     .fit(&[0.0, 1.0, 2.0, 3.0], 4, 1, &[0.0, 1.0, 0.5, 0.25], &[0.2, 0.4], 2)
    ///     .map_err(|(_, e)| e)?;
    /// assert_eq!(fitted.num_params(), 4);
    /// # Ok(())
    /// # }
    /// ```
    #[allow(clippy::result_large_err)]
    pub fn fit(
        self,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
        y: &[f64],
        z: &[f64],
        n_inducing: usize,
    ) -> Result<FittedSparseGpr<O, FreeInducing>, (Self, GprError)> {
        match assemble_fitted(
            self.kernel.clone(),
            self.likelihood,
            self.optimizer.clone(),
            x,
            n_rows,
            n_cols,
            y,
            z,
            n_inducing,
        ) {
            Ok(mut fitted) => match fitted.optimize_hyperparameters() {
                Ok(()) => Ok(fitted),
                Err(err) => Err((fitted.into_trainer(), err)),
            },
            Err(err) => Err((self, err)),
        }
    }
}

#[allow(private_bounds)]
impl<I: InducingLayout> SparseGpr<Fixed, I> {
    /// Factors `K_mm = k(Z, Z)` at the current `θ` without a search.
    ///
    /// `x` and `z` are column-major (`n` / `m` points by `d` features). `Z`
    /// is supplied by the caller and is not moved. Likelihood noise is not
    /// added to `K_mm`. Also forms the VFE factors used by
    /// [`FittedSparseGpr::predict`] and
    /// [`FittedSparseGpr::neg_log_marginal_likelihood`].
    ///
    /// # Errors
    ///
    /// Returns [`GprError::EmptyInput`] when `n`, `d`, or `m` is zero.
    /// Returns [`GprError::DimensionMismatch`] when `z` is packed with a
    /// different feature count than `x` (the same `n_cols` is required).
    /// Length and finiteness errors match [`crate::Gpr<Fixed>::factor`].
    /// [`GprError::CholeskyFailed`] when `K_mm` or the VFE `B` matrix cannot
    /// be factored.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Fixed, GaussianLikelihood, SparseGpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    /// let likelihood = GaussianLikelihood::new(0.1)?;
    /// let fitted = SparseGpr::new(kernel, likelihood)
    ///     .with_optimizer(Fixed)
    ///     .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0], &[0.0, 1.0], 2)
    ///     .map_err(|(_, e)| e)?;
    /// assert_eq!(fitted.m(), fitted.n());
    /// # Ok(())
    /// # }
    /// ```
    #[allow(clippy::result_large_err)] // failure returns the trainer so the caller can retry
    pub fn factor(
        self,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
        y: &[f64],
        z: &[f64],
        n_inducing: usize,
    ) -> Result<FittedSparseGpr<Fixed, I>, (Self, GprError)> {
        match assemble_fitted(
            self.kernel.clone(),
            self.likelihood,
            Fixed,
            x,
            n_rows,
            n_cols,
            y,
            z,
            n_inducing,
        ) {
            Ok(fitted) => Ok(fitted),
            Err(err) => Err((self, err)),
        }
    }
}

#[allow(private_bounds)]
impl<O, I: InducingLayout> FittedSparseGpr<O, I> {
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
    /// Returns [`GprError::InvalidHyperparameter`] if `out` is the wrong length
    /// or a custom leaf rejects the write.
    pub fn get_params(&self, out: &mut [f64]) -> Result<(), GprError> {
        require_param_len(out.len(), self.num_params())?;
        let n_theta = self.kernel.num_params() + self.likelihood.num_params();
        write_params(&self.kernel, &self.likelihood, &mut out[..n_theta])?;
        if I::z_params(self.m, self.d) > 0 {
            out[n_theta..].copy_from_slice(&self.z_obs);
        }
        Ok(())
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
    /// Returns [`GprError::InvalidHyperparameter`] if `params` is the wrong
    /// length, [`GprError::InvalidNoiseVariance`] if the likelihood `θ` is
    /// invalid, or [`GprError::CholeskyFailed`] if `K_mm` or `B` cannot be
    /// factored. A rejected slice or a Cholesky failure leaves stored `θ`
    /// and the VFE factors unchanged.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Fixed, GaussianLikelihood, SparseGpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let mut fitted = SparseGpr::new(
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
        require_param_len(params.len(), self.num_params())?;
        let mut kernel = self.kernel.clone();
        kernel.set_params(&params[..n_kernel])?;
        let mut likelihood = self.likelihood;
        likelihood.set_params(&params[n_kernel..n_theta])?;
        let z_obs = if I::z_params(self.m, self.d) > 0 {
            params[n_theta..].to_vec()
        } else {
            self.z_obs.clone()
        };
        let state = assemble_vfe(
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
            return Err(GprError::InvalidHyperparameter {
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
    /// Returns [`GprError::InvalidHyperparameter`] if a slice length is wrong,
    /// [`GprError::InvalidNoiseVariance`] if the likelihood `θ` is invalid,
    /// or [`GprError::CholeskyFailed`] if the VFE system cannot be factored.
    /// Kernel and likelihood `θ` are committed together only after the
    /// system factors.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Fixed, GaussianLikelihood, SparseGpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let mut fitted = SparseGpr::new(
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
    ) -> Result<f64, GprError> {
        let n_params = self.num_params();
        require_param_len(params.len(), n_params)?;
        require_param_len(out.len(), n_params)?;
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
    /// Returns [`GprError::InvalidHyperparameter`] if a slice length is wrong,
    /// [`GprError::InvalidNoiseVariance`] if the likelihood `θ` is invalid,
    /// or [`GprError::CholeskyFailed`] if the VFE system cannot be factored.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Fixed, GaussianLikelihood, SparseGpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let mut fitted = SparseGpr::new(
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
    pub fn hessian_into(&mut self, params: &[f64], out: &mut [f64]) -> Result<(), GprError> {
        let n_params = self.num_params();
        require_param_len(params.len(), n_params)?;
        require_param_len(out.len(), n_params * n_params)?;
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

    fn inducing_equals_training(&self) -> bool {
        self.x_obs == self.z_obs
    }

    fn exact_fitted(&self) -> Result<crate::FittedGpr<Fixed>, GprError> {
        crate::Gpr::new(self.kernel.clone(), self.likelihood)
            .with_optimizer(Fixed)
            .factor(&self.x_obs, self.n, self.d, &self.y)
            .map_err(|(_, e)| e)
    }

    fn apply_vfe(&mut self, state: VfeState) {
        self.k_mm_l = state.k_mm_l;
        self.a = state.a;
        self.b_l = state.b_l;
        self.w = state.w;
        self.k_diag_sum = state.k_diag_sum;
        self.a_frobenius2 = state.a_frobenius2;
    }

    fn into_trainer(self) -> SparseGpr<O, I> {
        SparseGpr {
            kernel: self.kernel,
            likelihood: self.likelihood,
            optimizer: self.optimizer,
            inducing: PhantomData,
        }
    }

    fn optimize_hyperparameters(&mut self) -> Result<(), GprError>
    where
        O: Clone + for<'a> Optimizer<SparseGprObjective<'a, O, I>>,
    {
        let mut init = vec![0.0; self.num_params()];
        self.get_params(&mut init)?;
        let before = self.clone();
        let optimizer = self.optimizer.clone();
        let result = {
            let mut obj = SparseGprObjective::new(self);
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
    /// use gprx::{Fixed, GaussianLikelihood, SparseGpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    /// let likelihood = GaussianLikelihood::new(0.1)?;
    /// let fitted = SparseGpr::new(kernel, likelihood)
    ///     .with_optimizer(Fixed)
    ///     .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0], &[0.0, 1.0], 2)
    ///     .map_err(|(_, e)| e)?;
    /// let nlml = fitted.neg_log_marginal_likelihood()?;
    /// assert!(nlml.is_finite());
    /// # Ok(())
    /// # }
    /// ```
    pub fn neg_log_marginal_likelihood(&self) -> Result<f64, GprError> {
        let noise = self.likelihood.noise_variance();
        let mut log_det_b = 0.0;
        for i in 0..self.m {
            log_det_b += self.b_l[(i, i)].ln();
        }
        log_det_b *= 2.0;
        let n_minus_m = (self.n - self.m) as f64;
        let log_det = n_minus_m * noise.ln() + log_det_b;
        let y_norm2: f64 = self.y.iter().map(|v| v * v).sum();
        let mut ay_dot_w = 0.0;
        for i in 0..self.m {
            let mut ay_i = 0.0;
            for j in 0..self.n {
                ay_i += self.a[(i, j)] * self.y[j];
            }
            ay_dot_w += ay_i * self.w[i];
        }
        let quad = (y_norm2 - ay_dot_w) / noise;
        let trace = (self.k_diag_sum - self.a_frobenius2) / (2.0 * noise);
        let log_two_pi = (2.0 * std::f64::consts::PI).ln();
        Ok(0.5 * ((self.n as f64) * log_two_pi + log_det + quad) + trace)
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
    /// [`GprError::InvalidHyperparameter`] / [`GprError::NonFiniteInput`] for a
    /// badly packed or non-finite `xs`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Fixed, GaussianLikelihood, SparseGpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    /// let likelihood = GaussianLikelihood::new(0.1)?;
    /// let fitted = SparseGpr::new(kernel, likelihood)
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
    ) -> Result<Prediction, GprError> {
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
    /// use gprx::{Fixed, GaussianLikelihood, PredictOptions, SparseGpr, VarianceKind};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    /// let likelihood = GaussianLikelihood::new(0.1)?;
    /// let fitted = SparseGpr::new(kernel, likelihood)
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
    ) -> Result<Prediction, GprError> {
        if n_cols != self.d {
            return Err(GprError::DimensionMismatch {
                x_dim: n_cols,
                expected_dim: self.d,
            });
        }
        validate_query(xs, n_rows, n_cols)?;
        let compiled = self.kernel.compile();
        let z_mat = pack_points(&self.z_obs, self.m, self.d);
        let mut query_x = Mat::zeros(n_rows, n_cols);
        pack_points_into(xs, n_rows, n_cols, query_x.as_mut());
        let mut k_sz = kernel_cross(&compiled, z_mat.as_ref(), query_x.as_ref())?;
        faer::linalg::triangular_solve::solve_lower_triangular_in_place(
            self.k_mm_l.as_ref(),
            k_sz.as_mut(),
            faer_par_dims(self.m, n_rows),
        );
        let mut kss = vec![0.0; n_rows];
        compiled.fill_diag_points(query_x.as_ref(), &mut kss)?;
        let mut binv_astar = k_sz.clone();
        solve_llt_in_place(self.b_l.as_ref(), binv_astar.as_mut());
        let noise = self.likelihood.noise_variance();
        let mut out = Prediction {
            mean: vec![0.0; n_rows],
            variance: vec![0.0; n_rows],
            variance_kind: options.variance_kind,
        };
        for col in 0..n_rows {
            let mut mean = 0.0;
            let mut a_norm = 0.0;
            let mut binv_norm = 0.0;
            for row in 0..self.m {
                let a_star = k_sz[(row, col)];
                mean += a_star * self.w[row];
                a_norm += a_star * a_star;
                let solved = binv_astar[(row, col)];
                binv_norm += a_star * solved;
            }
            let mut latent = kss[col] - a_norm + noise * binv_norm;
            if latent < 0.0 {
                latent = 0.0;
            }
            out.mean[col] = mean;
            out.variance[col] = match options.variance_kind {
                VarianceKind::Latent => latent,
                VarianceKind::Observation => latent + noise,
            };
        }
        Ok(out)
    }

    #[cfg(test)]
    fn k_mm_l(&self) -> MatRef<'_, f64> {
        self.k_mm_l.as_ref()
    }
}

struct VfeState {
    k_mm_l: Mat<f64>,
    a: Mat<f64>,
    b_l: Mat<f64>,
    w: Vec<f64>,
    k_diag_sum: f64,
    a_frobenius2: f64,
}

#[allow(clippy::too_many_arguments)]
fn assemble_fitted<O, I: InducingLayout>(
    kernel: KernelSpec,
    likelihood: GaussianLikelihood,
    optimizer: O,
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    y: &[f64],
    z: &[f64],
    n_inducing: usize,
) -> Result<FittedSparseGpr<O, I>, GprError> {
    let state = assemble_vfe(&kernel, likelihood, x, n_rows, n_cols, y, z, n_inducing)?;
    Ok(FittedSparseGpr {
        kernel,
        likelihood,
        optimizer,
        inducing: PhantomData,
        x_obs: x.to_vec(),
        z_obs: z.to_vec(),
        y: y.to_vec(),
        k_mm_l: state.k_mm_l,
        a: state.a,
        b_l: state.b_l,
        w: state.w,
        k_diag_sum: state.k_diag_sum,
        a_frobenius2: state.a_frobenius2,
        n: n_rows,
        m: n_inducing,
        d: n_cols,
    })
}

#[allow(clippy::too_many_arguments)]
fn assemble_vfe(
    kernel: &KernelSpec,
    likelihood: GaussianLikelihood,
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    y: &[f64],
    z: &[f64],
    n_inducing: usize,
) -> Result<VfeState, GprError> {
    validate_training(x, n_rows, n_cols, y)?;
    validate_inducing(z, n_inducing, n_cols)?;
    let compiled = kernel.compile();
    let x_mat = pack_points(x, n_rows, n_cols);
    let z_mat = pack_points(z, n_inducing, n_cols);
    let mut k_mm = Mat::zeros(n_inducing, n_inducing);
    let mut scratch = Mat::zeros(n_inducing, n_inducing);
    compiled.apply_points(
        z_mat.as_ref(),
        k_mm.as_mut(),
        Triangle::Lower,
        scratch.as_mut(),
    )?;
    let req = llt::factor::cholesky_in_place_scratch::<f64>(
        n_inducing,
        faer_par(n_inducing),
        Default::default(),
    );
    let mut chol_scratch = MemBuffer::new(req);
    cholesky_lower_with_policy(
        &mut k_mm,
        &mut chol_scratch,
        JitterPolicy::default(),
        CholeskyStage::Fit,
    )?;
    // Same packed `X` and `Z` share a training White diagonal. Rectangular
    // `apply_cross` leaves White at zero.
    let mut a = if x == z {
        let mut gram = Mat::zeros(n_rows, n_rows);
        let mut gram_scratch = Mat::zeros(n_rows, n_rows);
        compiled.apply_points(
            x_mat.as_ref(),
            gram.as_mut(),
            Triangle::Lower,
            gram_scratch.as_mut(),
        )?;
        symmetrize_lower(gram.as_mut(), n_rows);
        gram
    } else {
        kernel_cross(&compiled, z_mat.as_ref(), x_mat.as_ref())?
    };
    faer::linalg::triangular_solve::solve_lower_triangular_in_place(
        k_mm.as_ref(),
        a.as_mut(),
        faer_par_dims(n_inducing, n_rows),
    );
    let noise = likelihood.noise_variance();
    let mut b = gram_aat_plus_noise(a.as_ref(), noise);
    let b_req = llt::factor::cholesky_in_place_scratch::<f64>(
        n_inducing,
        faer_par(n_inducing),
        Default::default(),
    );
    let mut b_scratch = MemBuffer::new(b_req);
    cholesky_lower_with_policy(
        &mut b,
        &mut b_scratch,
        JitterPolicy::default(),
        CholeskyStage::Fit,
    )?;
    let mut k_diag = vec![0.0; n_rows];
    compiled.fill_diag_points(x_mat.as_ref(), &mut k_diag)?;
    let k_diag_sum = k_diag.iter().sum();
    let a_frobenius2 = frobenius2(a.as_ref());
    let mut ay = Mat::zeros(n_inducing, 1);
    for i in 0..n_inducing {
        let mut sum = 0.0;
        for j in 0..n_rows {
            sum += a[(i, j)] * y[j];
        }
        ay[(i, 0)] = sum;
    }
    solve_llt_in_place(b.as_ref(), ay.as_mut());
    let mut w = vec![0.0; n_inducing];
    for i in 0..n_inducing {
        w[i] = ay[(i, 0)];
    }
    Ok(VfeState {
        k_mm_l: k_mm,
        a,
        b_l: b,
        w,
        k_diag_sum,
        a_frobenius2,
    })
}

fn fill_z_intervals(x: &[f64], n: usize, d: usize, out: &mut [Interval]) -> Result<(), GprError> {
    if d == 0 || out.len() % d != 0 {
        return Err(GprError::InvalidHyperparameter {
            reason: "inducing interval length is not a multiple of d".to_owned(),
        });
    }
    let m = out.len() / d;
    for dim in 0..d {
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        for i in 0..n {
            let v = x[i + dim * n];
            min_x = min_x.min(v);
            max_x = max_x.max(v);
        }
        let range = (max_x - min_x).max(0.0);
        let slack = (0.1 * range).max(0.1);
        let mut lo = min_x - slack;
        let hi = max_x + slack;
        if lo > 0.0 {
            lo = 0.0;
        }
        let interval = Interval::new(lo, hi)?;
        for p in 0..m {
            out[p + dim * m] = interval;
        }
    }
    Ok(())
}

struct KernelVar {
    d_kmm: Mat<f64>,
    d_kmn: Mat<f64>,
    d_kdiag: f64,
    d_noise: f64,
}

struct VfeEngine {
    l: Mat<f64>,
    a: Mat<f64>,
    b_l: Mat<f64>,
    w: Vec<f64>,
    noise: f64,
    k_diag_sum: f64,
    a_fro: f64,
    y: Vec<f64>,
    n: usize,
    m: usize,
}

impl VfeEngine {
    fn from_model<O, I>(model: &FittedSparseGpr<O, I>) -> Self {
        Self {
            l: model.k_mm_l.clone(),
            a: model.a.clone(),
            b_l: model.b_l.clone(),
            w: model.w.clone(),
            noise: model.likelihood.noise_variance(),
            k_diag_sum: model.k_diag_sum,
            a_fro: model.a_frobenius2,
            y: model.y.clone(),
            n: model.n,
            m: model.m,
        }
    }

    fn quad_and_trace(&self) -> (f64, f64) {
        let y_norm2: f64 = self.y.iter().map(|v| v * v).sum();
        let mut ay_dot_w = 0.0;
        for i in 0..self.m {
            let mut ay_i = 0.0;
            for j in 0..self.n {
                ay_i += self.a[(i, j)] * self.y[j];
            }
            ay_dot_w += ay_i * self.w[i];
        }
        let quad = (y_norm2 - ay_dot_w) / self.noise;
        let trace = (self.k_diag_sum - self.a_fro) / (2.0 * self.noise);
        (quad, trace)
    }

    fn first_tangent(&self, var: &KernelVar) -> VfeTangent {
        let phi = chol_phi(self.l.as_ref(), var.d_kmm.as_ref());
        let phi_l = tril_half(phi.as_ref());
        let mut da = var.d_kmn.clone();
        solve_lower(self.l.as_ref(), da.as_mut());
        mat_sub_mul(&mut da, phi_l.as_ref(), self.a.as_ref());
        let db = noise_plus_sym_prod(da.as_ref(), self.a.as_ref(), var.d_noise);
        let mut u = vec![0.0; self.m];
        for i in 0..self.m {
            let mut sum = 0.0;
            for j in 0..self.n {
                sum += da[(i, j)] * self.y[j];
            }
            u[i] = sum;
        }
        VfeTangent {
            phi,
            phi_l,
            da,
            db,
            u,
            d_kdiag: var.d_kdiag,
            d_noise: var.d_noise,
        }
    }

    fn directional(&self, var: &KernelVar) -> f64 {
        let t = self.first_tangent(var);
        self.directional_from_tangent(&t)
    }

    fn directional_from_tangent(&self, t: &VfeTangent) -> f64 {
        let (quad, trace) = self.quad_and_trace();
        let d_logdet_b = trace_solve(self.b_l.as_ref(), t.db.as_ref());
        let d_q = 2.0 * dot(&self.w, &t.u) - quad_form(&self.w, t.db.as_ref());
        let d_af = 2.0 * frobenius_dot(self.a.as_ref(), t.da.as_ref());
        let d_quad = -d_q / self.noise - quad * t.d_noise / self.noise;
        let d_trace = (t.d_kdiag - d_af) / (2.0 * self.noise) - trace * t.d_noise / self.noise;
        let n_minus_m = (self.n - self.m) as f64;
        0.5 * (n_minus_m * t.d_noise / self.noise + d_logdet_b + d_quad) + d_trace
    }

    fn second_directional(&self, ti: &VfeTangent, tj: &VfeTangent, dd: &KernelVar) -> f64 {
        let (quad, trace) = self.quad_and_trace();
        let phi_dd = chol_phi(self.l.as_ref(), dd.d_kmm.as_ref());
        let dphi_j_on_i = dphi_from(ti.phi.as_ref(), tj.phi_l.as_ref(), phi_dd.as_ref());
        let dphi_l = tril_half(dphi_j_on_i.as_ref());
        let mut linv_di_kmn = ti.da.clone();
        mat_add_mul(&mut linv_di_kmn, ti.phi_l.as_ref(), self.a.as_ref());
        let mut dda = dd.d_kmn.clone();
        solve_lower(self.l.as_ref(), dda.as_mut());
        mat_sub_mul(&mut dda, tj.phi_l.as_ref(), linv_di_kmn.as_ref());
        mat_sub_mul(&mut dda, dphi_l.as_ref(), self.a.as_ref());
        mat_sub_mul(&mut dda, ti.phi_l.as_ref(), tj.da.as_ref());
        let ddb = second_db(
            ti.da.as_ref(),
            tj.da.as_ref(),
            dda.as_ref(),
            self.a.as_ref(),
            dd.d_noise,
        );
        let mut ddu = vec![0.0; self.m];
        for i in 0..self.m {
            let mut sum = 0.0;
            for j in 0..self.n {
                sum += dda[(i, j)] * self.y[j];
            }
            ddu[i] = sum;
        }
        let d_logdet_b_i = trace_solve(self.b_l.as_ref(), ti.db.as_ref());
        let _ = d_logdet_b_i;
        let d2_logdet_b = second_logdet_b(
            self.b_l.as_ref(),
            ti.db.as_ref(),
            tj.db.as_ref(),
            ddb.as_ref(),
        );
        let dwi = dw_from(self.b_l.as_ref(), &self.w, ti.db.as_ref(), &ti.u);
        let dwj = dw_from(self.b_l.as_ref(), &self.w, tj.db.as_ref(), &tj.u);
        let d_q_i = 2.0 * dot(&self.w, &ti.u) - quad_form(&self.w, ti.db.as_ref());
        let d_q_j = 2.0 * dot(&self.w, &tj.u) - quad_form(&self.w, tj.db.as_ref());
        let d2_q = 2.0 * dot(&dwj, &ti.u) + 2.0 * dot(&self.w, &ddu)
            - dot(&dwj, &mat_vec(ti.db.as_ref(), &self.w))
            - quad_form(&self.w, ddb.as_ref())
            - dot(&self.w, &mat_vec(ti.db.as_ref(), &dwj));
        let d_af_i = 2.0 * frobenius_dot(self.a.as_ref(), ti.da.as_ref());
        let d_af_j = 2.0 * frobenius_dot(self.a.as_ref(), tj.da.as_ref());
        let d2_af = 2.0 * frobenius_dot(tj.da.as_ref(), ti.da.as_ref())
            + 2.0 * frobenius_dot(self.a.as_ref(), dda.as_ref());
        let d_quad_j = -d_q_j / self.noise - quad * tj.d_noise / self.noise;
        let d2_quad = -d2_q / self.noise + d_q_i * tj.d_noise / (self.noise * self.noise)
            - d_quad_j * ti.d_noise / self.noise
            - quad * dd.d_noise / self.noise
            + quad * ti.d_noise * tj.d_noise / (self.noise * self.noise);
        let d_trace_j =
            (tj.d_kdiag - d_af_j) / (2.0 * self.noise) - trace * tj.d_noise / self.noise;
        let d2_trace = (dd.d_kdiag - d2_af) / (2.0 * self.noise)
            - (ti.d_kdiag - d_af_i) * tj.d_noise / (2.0 * self.noise * self.noise)
            - d_trace_j * ti.d_noise / self.noise
            - trace * dd.d_noise / self.noise
            + trace * ti.d_noise * tj.d_noise / (self.noise * self.noise);
        let n_minus_m = (self.n - self.m) as f64;
        let d2_log_noise =
            dd.d_noise / self.noise - ti.d_noise * tj.d_noise / (self.noise * self.noise);
        let _ = dwi;
        0.5 * (n_minus_m * d2_log_noise + d2_logdet_b + d2_quad) + d2_trace
    }
}

struct VfeTangent {
    phi: Mat<f64>,
    phi_l: Mat<f64>,
    da: Mat<f64>,
    db: Mat<f64>,
    u: Vec<f64>,
    d_kdiag: f64,
    d_noise: f64,
}

fn analytic_gradient<O, I>(
    model: &FittedSparseGpr<O, I>,
    out: &mut [f64],
    include_z: bool,
) -> Result<(), GprError> {
    let engine = VfeEngine::from_model(model);
    let vars = collect_first_vars(model, include_z)?;
    for (i, var) in vars.iter().enumerate() {
        out[i] = engine.directional(var);
    }
    Ok(())
}

fn analytic_hessian<O, I>(
    model: &FittedSparseGpr<O, I>,
    out: &mut [f64],
    include_z: bool,
) -> Result<(), GprError> {
    let engine = VfeEngine::from_model(model);
    let vars = collect_first_vars(model, include_z)?;
    let tangents: Vec<VfeTangent> = vars.iter().map(|v| engine.first_tangent(v)).collect();
    let p = vars.len();
    for j in 0..p {
        for i in j..p {
            let dd = second_var(model, i, j, include_z)?;
            let hij = engine.second_directional(&tangents[i], &tangents[j], &dd);
            out[i * p + j] = hij;
            out[j * p + i] = hij;
        }
    }
    Ok(())
}

fn collect_first_vars<O, I>(
    model: &FittedSparseGpr<O, I>,
    include_z: bool,
) -> Result<Vec<KernelVar>, GprError> {
    let compiled = model.kernel.compile();
    let x = pack_points(&model.x_obs, model.n, model.d);
    let z = pack_points(&model.z_obs, model.m, model.d);
    let n_kernel = model.kernel.num_params();
    let n_theta = n_kernel + model.likelihood.num_params();
    let mut vars = Vec::with_capacity(n_theta + if include_z { model.m * model.d } else { 0 });
    for i in 0..n_kernel {
        vars.push(kernel_theta_var(
            &compiled,
            x.as_ref(),
            z.as_ref(),
            model.n,
            i,
        )?);
    }
    vars.push(likelihood_var(
        model.m,
        model.n,
        model.likelihood.noise_variance(),
    ));
    if include_z {
        for dim in 0..model.d {
            for p in 0..model.m {
                vars.push(z_coord_var(&compiled, x.as_ref(), z.as_ref(), p, dim)?);
            }
        }
    }
    Ok(vars)
}

fn kernel_theta_var(
    compiled: &CompiledKernel,
    x: MatRef<'_, f64>,
    z: MatRef<'_, f64>,
    n: usize,
    param_idx: usize,
) -> Result<KernelVar, GprError> {
    let m = z.nrows();
    let mut d_kmm = Mat::zeros(m, m);
    let mut scratch_mm = Mat::zeros(m, m);
    compiled.grad_points(
        z,
        d_kmm.as_mut(),
        param_idx,
        Triangle::Full,
        scratch_mm.as_mut(),
    )?;
    let mut d_kmn = Mat::zeros(m, n);
    let mut scratch_mn = Mat::zeros(m, n);
    compiled.grad_cross_points(z, x, d_kmn.as_mut(), param_idx, scratch_mn.as_mut())?;
    let mut d_xx = Mat::zeros(n, n);
    let mut scratch_xx = Mat::zeros(n, n);
    compiled.grad_points(
        x,
        d_xx.as_mut(),
        param_idx,
        Triangle::Lower,
        scratch_xx.as_mut(),
    )?;
    let mut d_kdiag = 0.0;
    for i in 0..n {
        d_kdiag += d_xx[(i, i)];
    }
    Ok(KernelVar {
        d_kmm,
        d_kmn,
        d_kdiag,
        d_noise: 0.0,
    })
}

fn likelihood_var(m: usize, n: usize, noise: f64) -> KernelVar {
    let _ = n;
    KernelVar {
        d_kmm: Mat::zeros(m, m),
        d_kmn: Mat::zeros(m, n),
        d_kdiag: 0.0,
        d_noise: noise,
    }
}

fn z_coord_var(
    compiled: &CompiledKernel,
    x: MatRef<'_, f64>,
    z: MatRef<'_, f64>,
    point: usize,
    dim: usize,
) -> Result<KernelVar, GprError> {
    let m = z.nrows();
    let n = x.nrows();
    let mut g2 = Mat::zeros(m, m);
    compiled.grad_wrt_coord_dim(z, z, g2.as_mut(), dim)?;
    let mut g_xz = Mat::zeros(n, m);
    compiled.grad_wrt_coord_dim(x, z, g_xz.as_mut(), dim)?;
    let mut d_kmm = Mat::zeros(m, m);
    for i in 0..m {
        d_kmm[(i, point)] += g2[(i, point)];
        d_kmm[(point, i)] += g2[(i, point)];
    }
    let mut d_kmn = Mat::zeros(m, n);
    for col in 0..n {
        d_kmn[(point, col)] = g_xz[(col, point)];
    }
    Ok(KernelVar {
        d_kmm,
        d_kmn,
        d_kdiag: 0.0,
        d_noise: 0.0,
    })
}

fn second_var<O, I>(
    model: &FittedSparseGpr<O, I>,
    i: usize,
    j: usize,
    include_z: bool,
) -> Result<KernelVar, GprError> {
    let n_kernel = model.kernel.num_params();
    let n_theta = n_kernel + model.likelihood.num_params();
    let compiled = model.kernel.compile();
    let x = pack_points(&model.x_obs, model.n, model.d);
    let z = pack_points(&model.z_obs, model.m, model.d);
    let m = model.m;
    let n = model.n;
    let z_index = |idx: usize| -> Option<(usize, usize)> {
        if !include_z || idx < n_theta {
            None
        } else {
            let local = idx - n_theta;
            Some((local % m, local / m))
        }
    };
    if i < n_kernel && j < n_kernel {
        return kernel_theta_second(&compiled, x.as_ref(), z.as_ref(), n, i, j);
    }
    if i == n_kernel && j == n_kernel {
        return Ok(likelihood_var(m, n, model.likelihood.noise_variance()));
    }
    if i < n_theta && j < n_theta {
        return Ok(KernelVar {
            d_kmm: Mat::zeros(m, m),
            d_kmn: Mat::zeros(m, n),
            d_kdiag: 0.0,
            d_noise: 0.0,
        });
    }
    if let (Some((pi, ei)), Some((pj, ej))) = (z_index(i), z_index(j)) {
        return z_z_second(&compiled, x.as_ref(), z.as_ref(), pi, ei, pj, ej);
    }
    let (theta, (p, e)) = if i < n_theta {
        (
            i,
            z_index(j).ok_or_else(|| GprError::InvalidHyperparameter {
                reason: "expected a free inducing coordinate".to_owned(),
            })?,
        )
    } else {
        (
            j,
            z_index(i).ok_or_else(|| GprError::InvalidHyperparameter {
                reason: "expected a free inducing coordinate".to_owned(),
            })?,
        )
    };
    if theta == n_kernel {
        return Ok(KernelVar {
            d_kmm: Mat::zeros(m, m),
            d_kmn: Mat::zeros(m, n),
            d_kdiag: 0.0,
            d_noise: 0.0,
        });
    }
    theta_z_second(&compiled, x.as_ref(), z.as_ref(), theta, p, e)
}

fn kernel_theta_second(
    compiled: &CompiledKernel,
    x: MatRef<'_, f64>,
    z: MatRef<'_, f64>,
    n: usize,
    i: usize,
    j: usize,
) -> Result<KernelVar, GprError> {
    let m = z.nrows();
    let mut d_kmm = Mat::zeros(m, m);
    let mut scratch_mm = Mat::zeros(m, m);
    compiled.hess_points(z, d_kmm.as_mut(), i, j, Triangle::Full, scratch_mm.as_mut())?;
    let mut d_kmn = Mat::zeros(m, n);
    let mut scratch_mn = Mat::zeros(m, n);
    compiled.hess_cross_points(z, x, d_kmn.as_mut(), i, j, scratch_mn.as_mut())?;
    let mut d_xx = Mat::zeros(n, n);
    let mut scratch_xx = Mat::zeros(n, n);
    compiled.hess_points(x, d_xx.as_mut(), i, j, Triangle::Lower, scratch_xx.as_mut())?;
    let mut d_kdiag = 0.0;
    for r in 0..n {
        d_kdiag += d_xx[(r, r)];
    }
    Ok(KernelVar {
        d_kmm,
        d_kmn,
        d_kdiag,
        d_noise: 0.0,
    })
}

fn z_z_second(
    compiled: &CompiledKernel,
    x: MatRef<'_, f64>,
    z: MatRef<'_, f64>,
    p: usize,
    e: usize,
    q: usize,
    f: usize,
) -> Result<KernelVar, GprError> {
    let m = z.nrows();
    let n = x.nrows();
    let mut h22 = Mat::zeros(m, m);
    compiled.hess_wrt_coord_dims(z, z, h22.as_mut(), e, f)?;
    let mut d_kmm = Mat::zeros(m, m);
    if p == q {
        for i in 0..m {
            if i != p {
                d_kmm[(i, p)] += h22[(i, p)];
                d_kmm[(p, i)] += h22[(i, p)];
            }
        }
    } else {
        let mut h12 = Mat::zeros(m, m);
        compiled.hess_wrt_coord_mixed(z, z, h12.as_mut(), e, f)?;
        let mut h21 = Mat::zeros(m, m);
        compiled.hess_wrt_coord_mixed(z, z, h21.as_mut(), f, e)?;
        d_kmm[(p, q)] = h12[(p, q)];
        d_kmm[(q, p)] = h21[(q, p)];
    }
    let mut d_kmn = Mat::zeros(m, n);
    if p == q {
        let mut h_xz = Mat::zeros(n, m);
        compiled.hess_wrt_coord_dims(x, z, h_xz.as_mut(), e, f)?;
        for col in 0..n {
            d_kmn[(p, col)] = h_xz[(col, p)];
        }
    }
    Ok(KernelVar {
        d_kmm,
        d_kmn,
        d_kdiag: 0.0,
        d_noise: 0.0,
    })
}

fn theta_z_second(
    compiled: &CompiledKernel,
    x: MatRef<'_, f64>,
    z: MatRef<'_, f64>,
    theta: usize,
    p: usize,
    e: usize,
) -> Result<KernelVar, GprError> {
    let m = z.nrows();
    let n = x.nrows();
    let mut g2 = Mat::zeros(m, m);
    compiled.hess_theta_coord_dim(z, z, g2.as_mut(), theta, e)?;
    let mut d_kmm = Mat::zeros(m, m);
    for i in 0..m {
        d_kmm[(i, p)] += g2[(i, p)];
        d_kmm[(p, i)] += g2[(i, p)];
    }
    let mut g_xz = Mat::zeros(n, m);
    compiled.hess_theta_coord_dim(x, z, g_xz.as_mut(), theta, e)?;
    let mut d_kmn = Mat::zeros(m, n);
    for col in 0..n {
        d_kmn[(p, col)] = g_xz[(col, p)];
    }
    Ok(KernelVar {
        d_kmm,
        d_kmn,
        d_kdiag: 0.0,
        d_noise: 0.0,
    })
}

fn chol_phi(l: MatRef<'_, f64>, dk: MatRef<'_, f64>) -> Mat<f64> {
    let m = l.nrows();
    let mut tmp = Mat::zeros(m, m);
    copy_mat(dk, tmp.as_mut());
    solve_lower(l, tmp.as_mut());
    let mut u = Mat::zeros(m, m);
    for i in 0..m {
        for j in 0..m {
            u[(i, j)] = tmp[(j, i)];
        }
    }
    solve_lower(l, u.as_mut());
    let mut phi = Mat::zeros(m, m);
    for i in 0..m {
        for j in 0..m {
            phi[(i, j)] = u[(j, i)];
        }
    }
    phi
}

fn tril_half(phi: MatRef<'_, f64>) -> Mat<f64> {
    let m = phi.nrows();
    let mut out = Mat::zeros(m, m);
    for j in 0..m {
        for i in j..m {
            out[(i, j)] = if i == j {
                0.5 * phi[(i, j)]
            } else {
                phi[(i, j)]
            };
        }
    }
    out
}

fn dphi_from(
    phi_i: MatRef<'_, f64>,
    phi_l_j: MatRef<'_, f64>,
    phi_dd: MatRef<'_, f64>,
) -> Mat<f64> {
    let m = phi_i.nrows();
    let mut out = Mat::zeros(m, m);
    for i in 0..m {
        for j in 0..m {
            let mut sum = phi_dd[(i, j)];
            for k in 0..m {
                sum -= phi_l_j[(i, k)] * phi_i[(k, j)];
                sum -= phi_i[(i, k)] * phi_l_j[(j, k)];
            }
            out[(i, j)] = sum;
        }
    }
    out
}

fn noise_plus_sym_prod(da: MatRef<'_, f64>, a: MatRef<'_, f64>, d_noise: f64) -> Mat<f64> {
    let m = da.nrows();
    let n = da.ncols();
    let mut db = Mat::zeros(m, m);
    for j in 0..m {
        for i in j..m {
            let mut sum = 0.0;
            for k in 0..n {
                sum += da[(i, k)] * a[(j, k)] + a[(i, k)] * da[(j, k)];
            }
            db[(i, j)] = sum;
            db[(j, i)] = sum;
        }
        db[(j, j)] += d_noise;
    }
    db
}

fn second_db(
    dai: MatRef<'_, f64>,
    daj: MatRef<'_, f64>,
    dda: MatRef<'_, f64>,
    a: MatRef<'_, f64>,
    dd_noise: f64,
) -> Mat<f64> {
    let m = a.nrows();
    let n = a.ncols();
    let mut db = Mat::zeros(m, m);
    for col in 0..m {
        for row in col..m {
            let mut sum = 0.0;
            for k in 0..n {
                sum += dda[(row, k)] * a[(col, k)]
                    + a[(row, k)] * dda[(col, k)]
                    + dai[(row, k)] * daj[(col, k)]
                    + daj[(row, k)] * dai[(col, k)];
            }
            db[(row, col)] = sum;
            db[(col, row)] = sum;
        }
        db[(col, col)] += dd_noise;
    }
    db
}

fn trace_solve(b_l: MatRef<'_, f64>, db: MatRef<'_, f64>) -> f64 {
    let mut solved = Mat::zeros(db.nrows(), db.ncols());
    copy_mat(db, solved.as_mut());
    solve_llt_in_place(b_l, solved.as_mut());
    let mut tr = 0.0;
    for i in 0..solved.nrows() {
        tr += solved[(i, i)];
    }
    tr
}

fn second_logdet_b(
    b_l: MatRef<'_, f64>,
    dbi: MatRef<'_, f64>,
    dbj: MatRef<'_, f64>,
    ddb: MatRef<'_, f64>,
) -> f64 {
    let mut si = Mat::zeros(dbi.nrows(), dbi.ncols());
    copy_mat(dbi, si.as_mut());
    solve_llt_in_place(b_l, si.as_mut());
    let mut sj = Mat::zeros(dbj.nrows(), dbj.ncols());
    copy_mat(dbj, sj.as_mut());
    solve_llt_in_place(b_l, sj.as_mut());
    let mut tr = 0.0;
    for i in 0..si.nrows() {
        for k in 0..si.ncols() {
            tr -= si[(k, i)] * sj[(i, k)];
        }
    }
    tr + trace_solve(b_l, ddb)
}

fn dw_from(b_l: MatRef<'_, f64>, w: &[f64], db: MatRef<'_, f64>, u: &[f64]) -> Vec<f64> {
    let m = w.len();
    let mut rhs = Mat::zeros(m, 1);
    let dbw = mat_vec(db, w);
    for i in 0..m {
        rhs[(i, 0)] = u[i] - dbw[i];
    }
    solve_llt_in_place(b_l, rhs.as_mut());
    let mut out = vec![0.0; m];
    for i in 0..m {
        out[i] = rhs[(i, 0)];
    }
    out
}

fn solve_lower(l: MatRef<'_, f64>, rhs: MatMut<'_, f64>) {
    let n = l.nrows();
    let n_rhs = rhs.ncols();
    faer::linalg::triangular_solve::solve_lower_triangular_in_place(
        l,
        rhs,
        faer_par_dims(n, n_rhs),
    );
}

fn copy_mat(src: MatRef<'_, f64>, mut dest: MatMut<'_, f64>) {
    for j in 0..src.ncols() {
        for i in 0..src.nrows() {
            dest[(i, j)] = src[(i, j)];
        }
    }
}

fn mat_sub_mul(dest: &mut Mat<f64>, left: MatRef<'_, f64>, right: MatRef<'_, f64>) {
    let m = dest.nrows();
    let n = dest.ncols();
    let k = left.ncols();
    for col in 0..n {
        for row in 0..m {
            let mut sum = 0.0;
            for t in 0..k {
                sum += left[(row, t)] * right[(t, col)];
            }
            dest[(row, col)] -= sum;
        }
    }
}

fn mat_add_mul(dest: &mut Mat<f64>, left: MatRef<'_, f64>, right: MatRef<'_, f64>) {
    let m = dest.nrows();
    let n = dest.ncols();
    let k = left.ncols();
    for col in 0..n {
        for row in 0..m {
            let mut sum = 0.0;
            for t in 0..k {
                sum += left[(row, t)] * right[(t, col)];
            }
            dest[(row, col)] += sum;
        }
    }
}

fn frobenius_dot(a: MatRef<'_, f64>, b: MatRef<'_, f64>) -> f64 {
    let mut sum = 0.0;
    for col in 0..a.ncols() {
        for row in 0..a.nrows() {
            sum += a[(row, col)] * b[(row, col)];
        }
    }
    sum
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn quad_form(w: &[f64], m: MatRef<'_, f64>) -> f64 {
    let mw = mat_vec(m, w);
    dot(w, &mw)
}

fn mat_vec(m: MatRef<'_, f64>, v: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; m.nrows()];
    for i in 0..m.nrows() {
        let mut sum = 0.0;
        for j in 0..m.ncols() {
            sum += m[(i, j)] * v[j];
        }
        out[i] = sum;
    }
    out
}

fn kernel_cross(
    compiled: &CompiledKernel,
    x: MatRef<'_, f64>,
    xs: MatRef<'_, f64>,
) -> Result<Mat<f64>, GprError> {
    let n = x.nrows();
    let q = xs.nrows();
    let mut out = Mat::zeros(n, q);
    let mut scratch = Mat::zeros(n, q);
    match compiled.coord_mode()? {
        CoordMode::Dist | CoordMode::Either => {
            let mut dist = Mat::zeros(n, q);
            let mut thread_scratch = Vec::new();
            fill_squared_euclidean_cross(x, xs, dist.as_mut(), &mut thread_scratch);
            compiled.apply_cross(dist.as_ref(), out.as_mut(), scratch.as_mut())?;
        }
        CoordMode::Points => {
            compiled.apply_cross_points(x, xs, out.as_mut(), scratch.as_mut())?;
        }
        CoordMode::Mixed => {
            let mut dist = Mat::zeros(n, q);
            let mut thread_scratch = Vec::new();
            fill_squared_euclidean_cross(x, xs, dist.as_mut(), &mut thread_scratch);
            compiled.apply_cross_mixed(dist.as_ref(), x, xs, out.as_mut(), scratch.as_mut())?;
        }
    }
    Ok(out)
}

fn gram_aat_plus_noise(a: MatRef<'_, f64>, noise: f64) -> Mat<f64> {
    let m = a.nrows();
    let n = a.ncols();
    let mut b = Mat::zeros(m, m);
    for j in 0..m {
        for i in j..m {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a[(i, k)] * a[(j, k)];
            }
            b[(i, j)] = sum;
        }
        b[(j, j)] += noise;
    }
    b
}

fn frobenius2(a: MatRef<'_, f64>) -> f64 {
    let mut sum = 0.0;
    for col in 0..a.ncols() {
        for row in 0..a.nrows() {
            let v = a[(row, col)];
            sum += v * v;
        }
    }
    sum
}

fn solve_llt_in_place(l: MatRef<'_, f64>, mut rhs: MatMut<'_, f64>) {
    let n = l.nrows();
    let n_rhs = rhs.ncols();
    let par = faer_par_dims(n, n_rhs);
    let req = llt::solve::solve_in_place_scratch::<f64>(n, n_rhs, par);
    let mut buf = MemBuffer::new(req);
    let stack = MemStack::new(&mut buf);
    llt::solve::solve_in_place(l, rhs.as_mut(), par, stack);
}

fn validate_inducing(z: &[f64], m: usize, d: usize) -> Result<(), GprError> {
    if m == 0 || d == 0 {
        return Err(GprError::EmptyInput);
    }
    if z.len() % m == 0 {
        let z_dim = z.len() / m;
        if z_dim != d {
            return Err(GprError::DimensionMismatch {
                x_dim: z_dim,
                expected_dim: d,
            });
        }
    }
    let expected = m.checked_mul(d).ok_or(GprError::EmptyInput)?;
    if z.len() != expected {
        return Err(GprError::InvalidHyperparameter {
            reason: format!(
                "expected {expected} inducing feature values, got {}",
                z.len()
            ),
        });
    }
    if z.iter().any(|v| !v.is_finite()) {
        return Err(GprError::NonFiniteInput);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{MaternKernel, MaternNu, RbfArdKernel, RbfKernel, WhiteKernel};
    use crate::objective::SparseGprObjective;
    use crate::{
        FastSimulatedAnnealing, Fixed, FreeInducing, Gpr, Lbfgs, NelderMead, Newton, NonlinearCg,
        Optimizer,
    };

    const TOL: f64 = 1e-12;
    const GRAD_FD: f64 = 1e-5;
    const HESS_FD: f64 = 1e-4;

    fn assert_close(actual: f64, expected: f64) {
        let scale = expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= TOL * scale,
            "actual={actual}, expected={expected}"
        );
    }

    fn factor_sparse(
        kernel: KernelSpec,
        x: &[f64],
        n: usize,
        d: usize,
        y: &[f64],
        z: &[f64],
        m: usize,
    ) -> FittedSparseGpr<Fixed> {
        let likelihood = GaussianLikelihood::new(0.1).expect("noise");
        SparseGpr::new(kernel, likelihood)
            .with_optimizer(Fixed)
            .factor(x, n, d, y, z, m)
            .map_err(|(_, e)| e)
            .expect("factor")
    }

    fn factor_ok(
        kernel: KernelSpec,
        x: &[f64],
        n: usize,
        d: usize,
        y: &[f64],
        z: &[f64],
        m: usize,
    ) {
        let fitted = factor_sparse(kernel, x, n, d, y, z, m);
        assert_eq!(fitted.n(), n);
        assert_eq!(fitted.m(), m);
        assert_eq!(fitted.d(), d);
        assert_eq!(fitted.x(), x);
        assert_eq!(fitted.y(), y);
        assert_eq!(fitted.z(), z);
    }

    fn assert_matches_exact(
        kernel: KernelSpec,
        x: &[f64],
        n: usize,
        d: usize,
        y: &[f64],
        xs_extra: &[f64],
        n_extra: usize,
    ) {
        let likelihood = GaussianLikelihood::new(0.1).expect("noise");
        let sparse = SparseGpr::new(kernel.clone(), likelihood)
            .with_optimizer(Fixed)
            .factor(x, n, d, y, x, n)
            .map_err(|(_, e)| e)
            .expect("sparse factor");
        let exact = Gpr::new(kernel, likelihood)
            .with_optimizer(Fixed)
            .factor(x, n, d, y)
            .map_err(|(_, e)| e)
            .expect("exact factor");
        assert_close(
            sparse.neg_log_marginal_likelihood().expect("sparse nlml"),
            exact.neg_log_marginal_likelihood().expect("exact nlml"),
        );
        assert_pred_close(&sparse, &exact, x, n, d);
        assert_pred_close(&sparse, &exact, xs_extra, n_extra, d);
    }

    fn assert_pred_close(
        sparse: &FittedSparseGpr<Fixed>,
        exact: &crate::FittedGpr<Fixed>,
        xs: &[f64],
        n_rows: usize,
        d: usize,
    ) {
        let kinds = [VarianceKind::Observation, VarianceKind::Latent];
        for kind in kinds {
            let options = PredictOptions {
                variance_kind: kind,
            };
            let got = sparse
                .predict_with(xs, n_rows, d, options)
                .expect("sparse predict");
            let want = exact
                .predict_with(xs, n_rows, d, options)
                .expect("exact predict");
            assert_eq!(got.variance_kind, kind);
            assert_eq!(got.mean.len(), n_rows);
            for i in 0..n_rows {
                assert_close(got.mean[i], want.mean[i]);
                assert_close(got.variance[i], want.variance[i]);
            }
        }
    }

    #[test]
    fn factor_rbf_n4_m2() {
        factor_ok(
            KernelSpec::from(RbfKernel::new(1.0).expect("ℓ")),
            &[0.0, 1.0, 2.0, 3.0],
            4,
            1,
            &[0.0, 1.0, 0.5, 0.25],
            &[0.5, 2.5],
            2,
        );
    }

    #[test]
    fn factor_matern_three_halves_n4_m2() {
        factor_ok(
            KernelSpec::from(MaternKernel::new(1.0, MaternNu::ThreeHalves).expect("ℓ")),
            &[0.0, 1.0, 2.0, 3.0],
            4,
            1,
            &[0.0, 1.0, 0.5, 0.25],
            &[0.5, 2.5],
            2,
        );
    }

    #[test]
    fn factor_rbf_ard_2d_n4_m2() {
        factor_ok(
            KernelSpec::from(RbfArdKernel::new(&[1.0, 1.5]).expect("ℓ")),
            &[0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            4,
            2,
            &[0.0, 1.0, 0.5, 0.25],
            &[0.25, 0.75, 0.25, 0.75],
            2,
        );
    }

    #[test]
    fn factor_rbf_plus_white_n4_m2() {
        factor_ok(
            KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"))
                + KernelSpec::from(WhiteKernel::new(0.05).expect("white")),
            &[0.0, 1.0, 2.0, 3.0],
            4,
            1,
            &[0.0, 1.0, 0.5, 0.25],
            &[0.5, 2.5],
            2,
        );
    }

    #[test]
    fn rbf_n2_z_eq_x_matches_analytic_gram() {
        let x = [0.0, 1.0];
        let y = [0.0, 1.0];
        let likelihood = GaussianLikelihood::new(0.1).expect("noise");
        let fitted = SparseGpr::new(
            KernelSpec::from(RbfKernel::new(1.0).expect("ℓ")),
            likelihood,
        )
        .with_optimizer(Fixed)
        .factor(&x, 2, 1, &y, &x, 2)
        .map_err(|(_, e)| e)
        .expect("factor");
        let l = fitted.k_mm_l();
        let reconstructed = |i: usize, j: usize| {
            let k_max = i.min(j);
            let mut sum = 0.0;
            for k in 0..=k_max {
                sum += l[(i, k)] * l[(j, k)];
            }
            sum
        };
        let off = (-0.5_f64).exp();
        assert_close(reconstructed(0, 0), 1.0);
        assert_close(reconstructed(1, 1), 1.0);
        assert_close(reconstructed(1, 0), off);
    }

    #[test]
    fn rbf_n4_z_eq_x_matches_exact() {
        assert_matches_exact(
            KernelSpec::from(RbfKernel::new(1.0).expect("ℓ")),
            &[0.0, 1.0, 2.0, 3.0],
            4,
            1,
            &[0.0, 1.0, 0.5, 0.25],
            &[0.25, 3.5],
            2,
        );
    }

    #[test]
    fn matern_n4_z_eq_x_matches_exact() {
        assert_matches_exact(
            KernelSpec::from(MaternKernel::new(1.0, MaternNu::ThreeHalves).expect("ℓ")),
            &[0.0, 1.0, 2.0, 3.0],
            4,
            1,
            &[0.0, 1.0, 0.5, 0.25],
            &[0.25, 3.5],
            2,
        );
    }

    #[test]
    fn rbf_ard_n4_z_eq_x_matches_exact() {
        assert_matches_exact(
            KernelSpec::from(RbfArdKernel::new(&[1.0, 1.5]).expect("ℓ")),
            &[0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            4,
            2,
            &[0.0, 1.0, 0.5, 0.25],
            &[0.25, 0.75, 0.25, 0.75],
            2,
        );
    }

    #[test]
    fn rbf_plus_white_n4_z_eq_x_matches_exact() {
        assert_matches_exact(
            KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"))
                + KernelSpec::from(WhiteKernel::new(0.05).expect("white")),
            &[0.0, 1.0, 2.0, 3.0],
            4,
            1,
            &[0.0, 1.0, 0.5, 0.25],
            &[0.25, 3.5],
            2,
        );
    }

    #[test]
    fn rbf_n4_m2_predict_and_nlml_succeed() {
        let likelihood = GaussianLikelihood::new(0.1).expect("noise");
        let fitted = SparseGpr::new(
            KernelSpec::from(RbfKernel::new(1.0).expect("ℓ")),
            likelihood,
        )
        .with_optimizer(Fixed)
        .factor(
            &[0.0, 1.0, 2.0, 3.0],
            4,
            1,
            &[0.0, 1.0, 0.5, 0.25],
            &[0.5, 2.5],
            2,
        )
        .map_err(|(_, e)| e)
        .expect("factor");
        let pred = fitted.predict(&[0.25, 3.5], 2, 1).expect("predict");
        assert_eq!(pred.mean.len(), 2);
        assert!(pred.mean.iter().all(|v| v.is_finite()));
        assert!(pred.variance.iter().all(|v| v.is_finite()));
        let nlml = fitted.neg_log_marginal_likelihood().expect("nlml");
        assert!(nlml.is_finite());
    }

    #[test]
    fn empty_training_is_empty_input() {
        let kernel = KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"));
        let likelihood = GaussianLikelihood::new(0.1).expect("noise");
        let err = SparseGpr::new(kernel, likelihood)
            .with_optimizer(Fixed)
            .factor(&[], 0, 1, &[], &[0.0], 1)
            .map_err(|(_, e)| e)
            .expect_err("empty n");
        assert_eq!(err, GprError::EmptyInput);
    }

    #[test]
    fn zero_inducing_is_empty_input() {
        let kernel = KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"));
        let likelihood = GaussianLikelihood::new(0.1).expect("noise");
        let err = SparseGpr::new(kernel, likelihood)
            .with_optimizer(Fixed)
            .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0], &[], 0)
            .map_err(|(_, e)| e)
            .expect_err("m = 0");
        assert_eq!(err, GprError::EmptyInput);
    }

    #[test]
    fn inducing_feature_mismatch_is_dimension_mismatch() {
        let kernel = KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"));
        let likelihood = GaussianLikelihood::new(0.1).expect("noise");
        let err = SparseGpr::new(kernel, likelihood)
            .with_optimizer(Fixed)
            .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0], &[0.0, 1.0, 0.0, 1.0], 2)
            .map_err(|(_, e)| e)
            .expect_err("z d = 2");
        assert_eq!(
            err,
            GprError::DimensionMismatch {
                x_dim: 2,
                expected_dim: 1,
            }
        );
    }

    #[test]
    fn inducing_length_mismatch_is_invalid_hyperparameter() {
        let kernel = KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"));
        let likelihood = GaussianLikelihood::new(0.1).expect("noise");
        let err = SparseGpr::new(kernel, likelihood)
            .with_optimizer(Fixed)
            .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0], &[0.0], 2)
            .map_err(|(_, e)| e)
            .expect_err("short z");
        assert!(matches!(err, GprError::InvalidHyperparameter { .. }));
    }

    #[test]
    fn empty_query_is_empty_input() {
        let kernel = KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"));
        let likelihood = GaussianLikelihood::new(0.1).expect("noise");
        let fitted = SparseGpr::new(kernel, likelihood)
            .with_optimizer(Fixed)
            .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0], &[0.0, 1.0], 2)
            .map_err(|(_, e)| e)
            .expect("factor");
        let err = fitted.predict(&[], 0, 1).expect_err("empty query");
        assert_eq!(err, GprError::EmptyInput);
    }

    #[test]
    fn query_feature_mismatch_is_dimension_mismatch() {
        let kernel = KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"));
        let likelihood = GaussianLikelihood::new(0.1).expect("noise");
        let fitted = SparseGpr::new(kernel, likelihood)
            .with_optimizer(Fixed)
            .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0], &[0.0, 1.0], 2)
            .map_err(|(_, e)| e)
            .expect("factor");
        let err = fitted.predict(&[0.0, 0.0], 1, 2).expect_err("query d = 2");
        assert_eq!(
            err,
            GprError::DimensionMismatch {
                x_dim: 2,
                expected_dim: 1,
            }
        );
    }

    fn kernel_ard() -> KernelSpec {
        KernelSpec::from(RbfArdKernel::new(&[1.0, 1.5]).expect("ℓ"))
    }

    fn assert_slice_close(actual: &[f64], expected: &[f64], tol: f64) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            let scale = e.abs().max(1.0);
            assert!(
                (a - e).abs() <= tol * scale,
                "actual={a}, expected={e}, tol={tol}"
            );
        }
    }

    fn fd_grad_from_value<I: InducingLayout>(
        model: &mut FittedSparseGpr<Fixed, I>,
        params: &[f64],
    ) -> Vec<f64> {
        let mut out = vec![0.0; params.len()];
        let mut plus = params.to_vec();
        let mut minus = params.to_vec();
        for i in 0..params.len() {
            plus.copy_from_slice(params);
            minus.copy_from_slice(params);
            plus[i] += GRAD_FD;
            minus[i] -= GRAD_FD;
            model.set_params(&plus).expect("plus");
            let fp = model.neg_log_marginal_likelihood().expect("fp");
            model.set_params(&minus).expect("minus");
            let fm = model.neg_log_marginal_likelihood().expect("fm");
            out[i] = (fp - fm) / (2.0 * GRAD_FD);
        }
        model.set_params(params).expect("restore");
        out
    }

    fn fd_hess_from_grad<I: InducingLayout>(
        model: &mut FittedSparseGpr<Fixed, I>,
        params: &[f64],
    ) -> Vec<f64> {
        let p = params.len();
        let mut out = vec![0.0; p * p];
        let mut plus = params.to_vec();
        let mut minus = params.to_vec();
        let mut gp = vec![0.0; p];
        let mut gm = vec![0.0; p];
        for j in 0..p {
            plus.copy_from_slice(params);
            minus.copy_from_slice(params);
            plus[j] += HESS_FD;
            minus[j] -= HESS_FD;
            model
                .value_and_gradient_into(&plus, &mut gp)
                .expect("grad+");
            model
                .value_and_gradient_into(&minus, &mut gm)
                .expect("grad-");
            for i in 0..p {
                out[i * p + j] = (gp[i] - gm[i]) / (2.0 * HESS_FD);
            }
        }
        model.set_params(params).expect("restore");
        out
    }

    fn assert_z_eq_x_matches_exact_derivs(
        kernel: KernelSpec,
        x: &[f64],
        n: usize,
        d: usize,
        y: &[f64],
    ) {
        let likelihood = GaussianLikelihood::new(0.1).expect("noise");
        let mut sparse = SparseGpr::new(kernel.clone(), likelihood)
            .with_optimizer(Fixed)
            .factor(x, n, d, y, x, n)
            .map_err(|(_, e)| e)
            .expect("sparse");
        let mut exact = Gpr::new(kernel, likelihood)
            .with_optimizer(Fixed)
            .factor(x, n, d, y)
            .map_err(|(_, e)| e)
            .expect("exact");
        let p = sparse.num_params();
        let mut params = vec![0.0; p];
        sparse.get_params(&mut params).expect("params");
        let mut g_s = vec![0.0; p];
        let mut g_e = vec![0.0; p];
        let vs = sparse
            .value_and_gradient_into(&params, &mut g_s)
            .expect("sparse vg");
        let ve = exact
            .value_and_gradient_into(&params, &mut g_e)
            .expect("exact vg");
        assert_close(vs, ve);
        assert_slice_close(&g_s, &g_e, TOL);
        let mut h_s = vec![0.0; p * p];
        let mut h_e = vec![0.0; p * p];
        sparse.hessian_into(&params, &mut h_s).expect("sparse hess");
        exact.hessian_into(&params, &mut h_e).expect("exact hess");
        assert_slice_close(&h_s, &h_e, TOL);
        let fd_g = fd_grad_from_value(&mut sparse, &params);
        assert_slice_close(&g_s, &fd_g, 1e-5);
    }

    #[test]
    fn rbf_n4_z_eq_x_matches_exact_value_grad_hess() {
        assert_z_eq_x_matches_exact_derivs(
            KernelSpec::from(RbfKernel::new(1.0).expect("ℓ")),
            &[0.0, 1.0, 2.0, 3.0],
            4,
            1,
            &[0.0, 1.0, 0.5, 0.25],
        );
    }

    #[test]
    fn matern_n4_z_eq_x_matches_exact_value_grad_hess() {
        assert_z_eq_x_matches_exact_derivs(
            KernelSpec::from(MaternKernel::new(1.0, MaternNu::ThreeHalves).expect("ℓ")),
            &[0.0, 1.0, 2.0, 3.0],
            4,
            1,
            &[0.0, 1.0, 0.5, 0.25],
        );
    }

    #[test]
    fn rbf_ard_n4_z_eq_x_matches_exact_value_grad_hess() {
        assert_z_eq_x_matches_exact_derivs(
            kernel_ard(),
            &[0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            4,
            2,
            &[0.0, 1.0, 0.5, 0.25],
        );
    }

    #[test]
    fn rbf_plus_white_n4_z_eq_x_matches_exact_value_grad_hess() {
        assert_z_eq_x_matches_exact_derivs(
            KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"))
                + KernelSpec::from(WhiteKernel::new(0.05).expect("white")),
            &[0.0, 1.0, 2.0, 3.0],
            4,
            1,
            &[0.0, 1.0, 0.5, 0.25],
        );
    }

    #[test]
    fn rbf_n4_z_eq_x_hessian_matches_grad_fd() {
        let kernel = KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"));
        let mut sparse = factor_sparse(
            kernel,
            &[0.0, 1.0, 2.0, 3.0],
            4,
            1,
            &[0.0, 1.0, 0.5, 0.25],
            &[0.0, 1.0, 2.0, 3.0],
            4,
        );
        let p = sparse.num_params();
        let mut params = vec![0.0; p];
        sparse.get_params(&mut params).expect("params");
        let mut hess = vec![0.0; p * p];
        sparse.hessian_into(&params, &mut hess).expect("hess");
        let fd = fd_hess_from_grad(&mut sparse, &params);
        assert_slice_close(&hess, &fd, 2e-4);
    }

    fn assert_fit_finishes_and_nlml_drops<O>(optimizer: O)
    where
        O: Clone + for<'a> Optimizer<SparseGprObjective<'a, O, FixedInducing>>,
    {
        let x = [0.0, 1.0, 2.0, 3.0];
        let y = [0.0, 1.0, 0.5, 0.25];
        let z = [0.5, 2.5];
        let kernel = KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"));
        let likelihood = GaussianLikelihood::new(0.1).expect("noise");
        let start = SparseGpr::new(kernel.clone(), likelihood)
            .with_optimizer(Fixed)
            .factor(&x, 4, 1, &y, &z, 2)
            .map_err(|(_, e)| e)
            .expect("start")
            .neg_log_marginal_likelihood()
            .expect("start nlml");
        let fitted = SparseGpr::new(kernel, likelihood)
            .with_optimizer(optimizer)
            .fit(&x, 4, 1, &y, &z, 2)
            .map_err(|(_, e)| e)
            .expect("fit");
        let end = fitted.neg_log_marginal_likelihood().expect("end nlml");
        assert!(end.is_finite(), "nlml={end}");
        assert!(end <= start, "end={end} start={start}");
    }

    #[test]
    fn rbf_n4_m2_fit_lbfgs_drops_nlml() {
        assert_fit_finishes_and_nlml_drops(Lbfgs::new());
    }

    #[test]
    fn rbf_n4_m2_fit_ncg_drops_nlml() {
        assert_fit_finishes_and_nlml_drops(NonlinearCg::new());
    }

    #[test]
    fn rbf_n4_m2_fit_nelder_mead_drops_nlml() {
        assert_fit_finishes_and_nlml_drops(NelderMead::new());
    }

    #[test]
    fn rbf_n4_m2_fit_newton_drops_nlml() {
        assert_fit_finishes_and_nlml_drops(Newton::new());
    }

    #[test]
    fn rbf_n4_m2_fit_fsa_drops_nlml() {
        assert_fit_finishes_and_nlml_drops(FastSimulatedAnnealing::new());
    }

    #[test]
    fn is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SparseGpr>();
        assert_send_sync::<FittedSparseGpr>();
        assert_send_sync::<SparseGpr<Lbfgs, FreeInducing>>();
        assert_send_sync::<FittedSparseGpr<Lbfgs, FreeInducing>>();
    }

    fn free_rbf_n8() -> (KernelSpec, [f64; 8], [f64; 8], [f64; 2]) {
        let x = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0];
        let y = [0.0, 0.35, 0.1, -0.4, 0.2, 0.55, -0.15, 0.3];
        (
            KernelSpec::from(RbfKernel::new(1.0).expect("ℓ")),
            x,
            y,
            [0.05, 0.12],
        )
    }

    fn free_fit_ok(
        kernel: KernelSpec,
        x: &[f64],
        n: usize,
        d: usize,
        y: &[f64],
        z: &[f64],
        m: usize,
    ) {
        let likelihood = GaussianLikelihood::new(0.1).expect("noise");
        let fitted = SparseGpr::new(kernel, likelihood)
            .with_inducing(FreeInducing)
            .fit(x, n, d, y, z, m)
            .map_err(|(_, e)| e)
            .expect("free fit");
        assert!(
            fitted
                .neg_log_marginal_likelihood()
                .expect("nlml")
                .is_finite()
        );
    }

    #[test]
    fn rbf_n8_m2_free_lbfgs_beats_fixed_factor() {
        let (kernel, x, y, z) = free_rbf_n8();
        let likelihood = GaussianLikelihood::new(0.1).expect("noise");
        let fixed = SparseGpr::new(kernel.clone(), likelihood)
            .with_optimizer(Fixed)
            .factor(&x, 8, 1, &y, &z, 2)
            .map_err(|(_, e)| e)
            .expect("factor")
            .neg_log_marginal_likelihood()
            .expect("fixed nlml");
        let free = SparseGpr::new(kernel, likelihood)
            .with_inducing(FreeInducing)
            .fit(&x, 8, 1, &y, &z, 2)
            .map_err(|(_, e)| e)
            .expect("free fit")
            .neg_log_marginal_likelihood()
            .expect("free nlml");
        assert!(free.is_finite(), "free={free}");
        assert!(free < fixed, "free={free} fixed={fixed}");
    }

    #[test]
    fn matern_n8_m2_free_fit_succeeds() {
        let (_, x, y, z) = free_rbf_n8();
        free_fit_ok(
            KernelSpec::from(MaternKernel::new(1.0, MaternNu::ThreeHalves).expect("ℓ")),
            &x,
            8,
            1,
            &y,
            &z,
            2,
        );
    }

    #[test]
    fn rbf_ard_n8_m2_free_fit_succeeds() {
        let x = [
            0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0, 0.0, 0.2, 0.1, 0.8, 0.4, 0.6, 0.3, 0.9,
        ];
        let y = [0.0, 0.35, 0.1, -0.4, 0.2, 0.55, -0.15, 0.3];
        let z = [0.05, 0.12, 0.08, 0.18];
        free_fit_ok(kernel_ard(), &x, 8, 2, &y, &z, 2);
    }

    #[test]
    fn rbf_plus_white_n8_m2_free_fit_succeeds() {
        let (_, x, y, z) = free_rbf_n8();
        free_fit_ok(
            KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"))
                + KernelSpec::from(WhiteKernel::new(0.05).expect("white")),
            &x,
            8,
            1,
            &y,
            &z,
            2,
        );
    }

    fn assert_free_solver_nlml_drops<O>(optimizer: O)
    where
        O: Clone + for<'a> Optimizer<SparseGprObjective<'a, O, FreeInducing>>,
    {
        let (kernel, x, y, z) = free_rbf_n8();
        let likelihood = GaussianLikelihood::new(0.1).expect("noise");
        let start = SparseGpr::new(kernel.clone(), likelihood)
            .with_optimizer(Fixed)
            .with_inducing(FreeInducing)
            .factor(&x, 8, 1, &y, &z, 2)
            .map_err(|(_, e)| e)
            .expect("start")
            .neg_log_marginal_likelihood()
            .expect("start nlml");
        let fitted = SparseGpr::new(kernel, likelihood)
            .with_inducing(FreeInducing)
            .with_optimizer(optimizer)
            .fit(&x, 8, 1, &y, &z, 2)
            .map_err(|(_, e)| e)
            .expect("fit");
        let end = fitted.neg_log_marginal_likelihood().expect("end nlml");
        assert!(end.is_finite(), "nlml={end}");
        assert!(end <= start, "end={end} start={start}");
    }

    #[test]
    fn rbf_n8_m2_free_lbfgs_drops_nlml() {
        assert_free_solver_nlml_drops(Lbfgs::new());
    }

    #[test]
    fn rbf_n8_m2_free_ncg_drops_nlml() {
        assert_free_solver_nlml_drops(NonlinearCg::new());
    }

    #[test]
    fn rbf_n8_m2_free_nelder_mead_drops_nlml() {
        assert_free_solver_nlml_drops(NelderMead::new());
    }

    #[test]
    fn rbf_n8_m2_free_newton_drops_nlml() {
        assert_free_solver_nlml_drops(Newton::new());
    }

    #[test]
    fn rbf_n8_m2_free_fsa_drops_nlml() {
        assert_free_solver_nlml_drops(FastSimulatedAnnealing::new());
    }

    fn assert_free_derivs(
        kernel: KernelSpec,
        x: &[f64],
        n: usize,
        d: usize,
        y: &[f64],
        z: &[f64],
        m: usize,
    ) {
        let likelihood = GaussianLikelihood::new(0.1).expect("noise");
        let mut fitted = SparseGpr::new(kernel, likelihood)
            .with_optimizer(Fixed)
            .with_inducing(FreeInducing)
            .factor(x, n, d, y, z, m)
            .map_err(|(_, e)| e)
            .expect("factor");
        let p = fitted.num_params();
        let mut params = vec![0.0; p];
        fitted.get_params(&mut params).expect("params");
        let mut grad = vec![0.0; p];
        fitted
            .value_and_gradient_into(&params, &mut grad)
            .expect("grad");
        let fd_g = fd_grad_from_value(&mut fitted, &params);
        assert_slice_close(&grad, &fd_g, 1e-5);
        let mut hess = vec![0.0; p * p];
        fitted.hessian_into(&params, &mut hess).expect("hess");
        let fd_h = fd_hess_from_grad(&mut fitted, &params);
        assert_slice_close(&hess, &fd_h, 2e-4);
    }

    #[test]
    fn rbf_n8_m2_free_grad_hess_match_fd() {
        let (kernel, x, y, z) = free_rbf_n8();
        assert_free_derivs(kernel, &x, 8, 1, &y, &z, 2);
    }

    #[test]
    fn matern_n8_m2_free_grad_hess_match_fd() {
        let (_, x, y, z) = free_rbf_n8();
        assert_free_derivs(
            KernelSpec::from(MaternKernel::new(1.0, MaternNu::ThreeHalves).expect("ℓ")),
            &x,
            8,
            1,
            &y,
            &z,
            2,
        );
    }

    #[test]
    fn rbf_ard_n8_m2_free_grad_hess_match_fd() {
        let x = [
            0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0, 0.0, 0.2, 0.1, 0.8, 0.4, 0.6, 0.3, 0.9,
        ];
        let y = [0.0, 0.35, 0.1, -0.4, 0.2, 0.55, -0.15, 0.3];
        let z = [0.05, 0.12, 0.08, 0.18];
        assert_free_derivs(kernel_ard(), &x, 8, 2, &y, &z, 2);
    }

    #[test]
    fn rbf_plus_white_n8_m2_free_grad_hess_match_fd() {
        let (_, x, y, z) = free_rbf_n8();
        assert_free_derivs(
            KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"))
                + KernelSpec::from(WhiteKernel::new(0.05).expect("white")),
            &x,
            8,
            1,
            &y,
            &z,
            2,
        );
    }
}
