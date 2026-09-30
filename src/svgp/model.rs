//! Trainer for stochastic variational GPR.

use std::marker::PhantomData;

use super::factor::SvgpMean;
use crate::error::GprError;
use crate::param::write_params;

use crate::kernel::KernelSpec;
use crate::kernel::{CompiledKernel, GramKernel};
use crate::likelihood::GaussianLikelihood;
use crate::optimizer::{Adam, Fixed};
use crate::precision::{DoublePrecision, GpScalar};

use super::factor::{assemble_fitted, run_adam_fit};
use super::fitted::FittedSvgp;

/// Trainer for stochastic variational GPR at a caller-supplied inducing set `Z`.
///
/// [`Svgp<Fixed>::factor`] prepares `K_mm` and a whitened prior `q(u)`
/// (`mean = 0`, `L = I`) at the current kernel and likelihood `θ`.
/// [`Svgp<Adam>::fit`] starts from that prior and runs mini-batch Adam.
/// [`Self::with_optimizer`] switches the type parameter; there is no fit
/// flag.
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
///     .factor(&[0.0, 1.0, 2.0, 3.0], 4, 1, &[0.0, 1.0, 0.5, 0.25], &[0.5, 2.5], 2)
///     .map_err(|(_, e)| e)?;
/// assert_eq!(fitted.n(), 4);
/// assert_eq!(fitted.m(), 2);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct Svgp<O = Fixed, M = crate::math::Accurate, P = DoublePrecision> {
    pub(crate) kernel: KernelSpec,
    pub(crate) likelihood: GaussianLikelihood,
    pub(crate) optimizer: O,
    pub(crate) _math: PhantomData<M>,
    pub(crate) _precision: PhantomData<P>,
}

impl Svgp {
    /// Builds a trainer with the current kernel `θ` and [`Fixed`].
    ///
    /// Inducing coordinates are an argument of [`Svgp<Fixed>::factor`], not
    /// of this constructor. The whitened prior is allocated at `factor` from
    /// the inducing count.
    pub fn new(kernel: KernelSpec, likelihood: GaussianLikelihood) -> Self {
        Self {
            kernel,
            likelihood,
            optimizer: Fixed,
            _math: PhantomData,
            _precision: PhantomData,
        }
    }
}

impl<O, M, P> Svgp<O, M, P> {
    /// Replaces the optimizer type parameter.
    ///
    /// [`Fixed`] keeps [`Svgp<Fixed>::factor`]. [`Adam`] enables
    /// [`Svgp<Adam>::fit`]. [`Adam`] is not an [`crate::Optimizer`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Adam, GaussianLikelihood, Svgp};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    /// let likelihood = GaussianLikelihood::new(0.1)?;
    /// let fitted = Svgp::new(kernel, likelihood)
    ///     .with_optimizer(Adam::new())
    ///     .fit(&[0.0, 1.0, 2.0, 3.0], 4, 1, &[0.0, 1.0, 0.5, 0.25], &[0.5, 2.5], 2)
    ///     .map_err(|(_, e)| e)?;
    /// assert!(fitted.neg_elbo()?.is_finite());
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_optimizer<O2>(self, optimizer: O2) -> Svgp<O2, M, P> {
        Svgp {
            kernel: self.kernel,
            likelihood: self.likelihood,
            optimizer,
            _math: PhantomData,
            _precision: PhantomData,
        }
    }

    /// Selects the storage precision. Omitting it leaves [`DoublePrecision`].
    #[allow(private_bounds)]
    pub fn with_precision<P2: GpScalar + SvgpMean>(self) -> Svgp<O, M, P2>
    where
        CompiledKernel<P2::Storage>: GramKernel<T = P2::Storage>,
    {
        Svgp {
            kernel: self.kernel,
            likelihood: self.likelihood,
            optimizer: self.optimizer,
            _math: PhantomData,
            _precision: PhantomData,
        }
    }

    /// Selects the kernel `exp`. Omitting it leaves [`crate::Accurate`].
    ///
    /// `factor`, `fit`, and predict use the same polynomial. Hyperparameter
    /// `exp(θ)` is unchanged.
    #[allow(private_bounds)]
    pub fn with_math<M2>(self) -> Svgp<O, M2, P>
    where
        M2: crate::math::KernelMath,
    {
        Svgp {
            kernel: self.kernel,
            likelihood: self.likelihood,
            optimizer: self.optimizer,
            _math: PhantomData,
            _precision: PhantomData,
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
    /// Inducing coordinates and the variational posterior are not counted.
    pub fn num_params(&self) -> usize {
        self.kernel.num_params() + self.likelihood.num_params()
    }

    /// Writes kernel `θ` then likelihood `θ` into `out`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::LengthMismatch`] if `out` is the wrong length
    /// or a custom leaf rejects the write.
    pub fn get_params(&self, out: &mut [f64]) -> Result<(), GprError> {
        write_params(&self.kernel, &self.likelihood, out)
    }

    /// Sets kernel then likelihood `θ` without forming the SVGP system.
    ///
    /// `params` is kernel parameters followed by the likelihood parameter,
    /// matching [`Self::get_params`]. Inducing coordinates are not in this
    /// slice.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::LengthMismatch`] if `params` is the wrong
    /// length, or [`GprError::InvalidNoiseVariance`] if the likelihood `θ`
    /// is invalid. Kernel and likelihood `θ` are committed together only
    /// after both writes succeed.
    pub fn set_params(&mut self, params: &[f64]) -> Result<(), GprError> {
        let n_kernel = self.kernel.num_params();
        crate::data::require_count(params.len(), self.num_params(), "parameters")?;
        let mut kernel = self.kernel.clone();
        kernel.set_params(&params[..n_kernel])?;
        let mut likelihood = self.likelihood;
        likelihood.set_params(&params[n_kernel..])?;
        self.kernel = kernel;
        self.likelihood = likelihood;
        Ok(())
    }
}

#[allow(private_bounds)]
impl<M, P> Svgp<Fixed, M, P>
where
    M: crate::math::KernelMath,
    P: GpScalar + SvgpMean,
    CompiledKernel<P::Storage>: GramKernel<T = P::Storage>,
{
    /// Factors `K_mm` and installs a whitened prior `q(u)` at the current `θ`.
    ///
    /// `x` and `z` are column-major (`n` / `m` points by `d` features). `Z`
    /// is supplied by the caller and is not a parameter. Likelihood noise is
    /// not added to `K_mm`. The variational mean is zero and the whitened
    /// Cholesky factor is the identity of order `m`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::EmptyInput`] when `n`, `d`, or `m` is zero.
    /// Returns [`GprError::DimensionMismatch`] when `z` is packed with a
    /// different feature count than `x` (the same `n_cols` is required).
    /// Length and finiteness errors match [`crate::Gpr::fit`].
    /// [`GprError::CholeskyFailed`] when `K_mm` cannot be factored.
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
    ///     .factor(&[0.0, 1.0, 2.0, 3.0], 4, 1, &[0.0, 1.0, 0.5, 0.25], &[0.5, 2.5], 2)
    ///     .map_err(|(_, e)| e)?;
    /// assert_eq!(fitted.num_params(), 7);
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
    ) -> Result<FittedSvgp<M, P>, (Self, GprError)> {
        match assemble_fitted(
            self.kernel.clone(),
            self.likelihood,
            x,
            n_rows,
            n_cols,
            y,
            z,
            n_inducing,
            None,
        ) {
            Ok(fitted) => Ok(fitted),
            Err(err) => Err((self, err)),
        }
    }
}

#[allow(private_bounds)]
impl<M, P> Svgp<Adam, M, P>
where
    M: crate::math::KernelMath,
    P: GpScalar + SvgpMean,
    CompiledKernel<P::Storage>: GramKernel<T = P::Storage>,
{
    /// Factors a whitened prior `q` and runs mini-batch Adam on `θ` and `q`.
    ///
    /// Starts from the same prior as [`Svgp<Fixed>::factor`]. Kernel and
    /// likelihood `θ` move in the existing [`crate::Interval`] logit.
    /// The diagonal of `L` moves in log space without that interval. The
    /// variational mean and the off-diagonal of `L` stay in user units.
    /// Each epoch shuffles the training indices. A remainder batch is kept.
    /// When `batch_size ≥ n` the epoch is one full-data step. Public
    /// [`FittedSvgp::value_and_gradient_into`] stays a full-data sum; this
    /// loop scales the data term by `n / b_actual` and leaves the KL full.
    ///
    /// # Errors
    ///
    /// Same input errors as [`Svgp<Fixed>::factor`].
    /// [`GprError::CholeskyFailed`] when `K_mm` cannot be factored.
    /// [`GprError::InvalidHyperparameter`] / [`GprError::NonFiniteInput`]
    /// when a trial `θ` or `L` is rejected.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Adam, GaussianLikelihood, Svgp};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    /// let likelihood = GaussianLikelihood::new(0.1)?;
    /// let fitted = Svgp::new(kernel, likelihood)
    ///     .with_optimizer(Adam::new())
    ///     .fit(&[0.0, 1.0, 2.0, 3.0], 4, 1, &[0.0, 1.0, 0.5, 0.25], &[0.5, 2.5], 2)
    ///     .map_err(|(_, e)| e)?;
    /// assert_eq!(fitted.n(), 4);
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
    ) -> Result<FittedSvgp<M, P>, (Self, GprError)> {
        match assemble_fitted(
            self.kernel.clone(),
            self.likelihood,
            x,
            n_rows,
            n_cols,
            y,
            z,
            n_inducing,
            None,
        ) {
            Ok(mut fitted) => match run_adam_fit(&mut fitted, &self.optimizer) {
                Ok(()) => Ok(fitted),
                Err(err) => Err((self, err)),
            },
            Err(err) => Err((self, err)),
        }
    }
}
