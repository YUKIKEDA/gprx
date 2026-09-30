//! Trainer for collapsed variational SGPR.

use std::marker::PhantomData;

use crate::error::GprError;
use crate::gpr::{KernelExp, with_kernel_exp};
use crate::sparse::SparseSpec;

use crate::kernel::KernelSpec;
use crate::likelihood::GaussianLikelihood;
use crate::objective::SgprObjective;
use crate::optimizer::{Fixed, Lbfgs, Optimizer};
use crate::precision::{DoublePrecision, GpScalar};

use super::factor::assemble_fitted;
use super::fitted::FittedSgpr;
use super::{FixedInducing, FreeInducing, InducingLayout};

/// Trainer for collapsed variational SGPR at a caller-supplied inducing set `Z`.
///
/// The default [`FixedInducing`] searches kernel and likelihood `θ` only.
/// [`Self::with_inducing`]`(`[`FreeInducing`]`)` searches `θ` and `Z`
/// together. [`Sgpr<Fixed, I>::factor`] prepares the VFE system at the
/// current `θ` with no search.
///
/// # Examples
///
/// ```rust
/// use gprx::kernel::{KernelSpec, RbfKernel};
/// use gprx::{GaussianLikelihood, Sgpr};
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
/// let likelihood = GaussianLikelihood::new(0.1)?;
/// let fitted = Sgpr::new(kernel, likelihood)
///     .fit(&[0.0, 1.0, 2.0, 3.0], 4, 1, &[0.0, 1.0, 0.5, 0.25], &[0.5, 2.5], 2)
///     .map_err(|(_, e)| e)?;
/// assert_eq!(fitted.n(), 4);
/// assert_eq!(fitted.m(), 2);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct Sgpr<O = Lbfgs, I = FixedInducing, P = DoublePrecision> {
    pub(super) spec: SparseSpec,
    pub(super) optimizer: O,
    pub(super) inducing: PhantomData<I>,
    pub(super) _precision: PhantomData<P>,
}
impl Sgpr {
    /// Builds a trainer with identity transforms, the current kernel `θ`, and
    /// [`Lbfgs`].
    ///
    /// Inducing coordinates are an argument of [`Sgpr::fit`] /
    /// [`Sgpr<Fixed>::factor`], not of this constructor. Call
    /// [`Self::with_optimizer`] to switch to [`Fixed`] or another
    /// [`Optimizer`].
    pub fn new(kernel: KernelSpec, likelihood: GaussianLikelihood) -> Self {
        Self {
            spec: SparseSpec::new(kernel, likelihood),
            optimizer: Lbfgs::new(),
            inducing: PhantomData,
            _precision: PhantomData,
        }
    }
}

impl<O, I, P> Sgpr<O, I, P> {
    /// The same settings under new type parameters, with `map` applied to
    /// the optimizer.
    fn retype<O2, I2, P2>(self, map: impl FnOnce(O) -> O2) -> Sgpr<O2, I2, P2> {
        Sgpr {
            spec: self.spec,
            optimizer: map(self.optimizer),
            inducing: PhantomData,
            _precision: PhantomData,
        }
    }

    /// Replaces the optimizer type parameter.
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
    /// assert_eq!(fitted.m(), fitted.n());
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_optimizer<O2>(self, optimizer: O2) -> Sgpr<O2, I, P> {
        self.retype(|_| optimizer)
    }

    /// Selects the storage precision. Omitting it leaves [`DoublePrecision`].
    pub fn with_precision<P2: GpScalar>(self) -> Sgpr<O, I, P2> {
        self.retype(|optimizer| optimizer)
    }

    /// Selects the kernel `exp`. Omitting it leaves [`KernelExp::Accurate`].
    ///
    /// `fit` and predict use the same polynomial. Hyperparameter `exp(θ)` is
    /// unchanged.
    pub fn with_math(mut self, math: KernelExp) -> Self {
        self.spec.math = math;
        self
    }

    /// Returns the kernel `exp` mode.
    pub fn math(&self) -> KernelExp {
        self.spec.math
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
    /// use gprx::{FreeInducing, GaussianLikelihood, Sgpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    /// let likelihood = GaussianLikelihood::new(0.1)?;
    /// let fitted = Sgpr::new(kernel, likelihood)
    ///     .with_inducing(FreeInducing)
    ///     .fit(&[0.0, 1.0, 2.0, 3.0], 4, 1, &[0.0, 1.0, 0.5, 0.25], &[0.5, 2.5], 2)
    ///     .map_err(|(_, e)| e)?;
    /// assert_eq!(fitted.num_params(), 4);
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_inducing<I2>(self, _inducing: I2) -> Sgpr<O, I2, P> {
        self.retype(|optimizer| optimizer)
    }

    /// Returns the kernel whose hyperparameters this trainer owns.
    pub fn kernel(&self) -> &KernelSpec {
        &self.spec.kernel
    }

    /// Returns the observation-noise model.
    pub fn likelihood(&self) -> &GaussianLikelihood {
        &self.spec.likelihood
    }

    /// Returns the concatenated kernel and likelihood parameter count.
    ///
    /// Inducing coordinates are not counted.
    pub fn num_params(&self) -> usize {
        self.spec.theta_len()
    }

    /// Writes kernel `θ` then likelihood `θ` into `out`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::LengthMismatch`] if `out` is the wrong length
    /// or a custom leaf rejects the write.
    pub fn get_params(&self, out: &mut [f64]) -> Result<(), GprError> {
        self.spec.read_theta(out)
    }

    /// Sets kernel then likelihood `θ` without forming the VFE system.
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
        self.spec.write_theta(params)
    }
}

impl<O, P> Sgpr<O, FixedInducing, P>
where
    P: GpScalar,
    O: Clone + for<'a> Optimizer<SgprObjective<'a, O, FixedInducing, P>>,
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
    /// use gprx::{GaussianLikelihood, Sgpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    /// let likelihood = GaussianLikelihood::new(0.1)?;
    /// let fitted = Sgpr::new(kernel, likelihood)
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
    ) -> Result<FittedSgpr<O, FixedInducing, P>, (Self, GprError)> {
        match with_kernel_exp!(self.spec.math, M => assemble_fitted::<_, _, M, _>(
            self.spec.kernel.clone(),
            self.spec.likelihood,
            self.optimizer.clone(),
            x,
            n_rows,
            n_cols,
            y,
            z,
            n_inducing,
        )) {
            Ok(mut fitted) => match fitted.optimize_hyperparameters() {
                Ok(()) => Ok(fitted),
                Err(err) => Err((fitted.into_trainer(), err)),
            },
            Err(err) => Err((self, err)),
        }
    }
}

impl<O, P> Sgpr<O, FreeInducing, P>
where
    P: GpScalar,
    O: Clone + for<'a> Optimizer<SgprObjective<'a, O, FreeInducing, P>>,
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
    /// Same input errors as [`Sgpr<O, FixedInducing>::fit`], plus
    /// [`GprError::CoordGradientUnsupported`] when the kernel has no
    /// coordinate derivative.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{FreeInducing, GaussianLikelihood, Sgpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    /// let likelihood = GaussianLikelihood::new(0.1)?;
    /// let fitted = Sgpr::new(kernel, likelihood)
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
    ) -> Result<FittedSgpr<O, FreeInducing, P>, (Self, GprError)> {
        match with_kernel_exp!(self.spec.math, M => assemble_fitted::<_, _, M, _>(
            self.spec.kernel.clone(),
            self.spec.likelihood,
            self.optimizer.clone(),
            x,
            n_rows,
            n_cols,
            y,
            z,
            n_inducing,
        )) {
            Ok(mut fitted) => match fitted.optimize_hyperparameters() {
                Ok(()) => Ok(fitted),
                Err(err) => Err((fitted.into_trainer(), err)),
            },
            Err(err) => Err((self, err)),
        }
    }
}

impl<I: InducingLayout, P> Sgpr<Fixed, I, P>
where
    P: GpScalar,
{
    /// Factors `K_mm = k(Z, Z)` at the current `θ` without a search.
    ///
    /// `x` and `z` are column-major (`n` / `m` points by `d` features). `Z`
    /// is supplied by the caller and is not moved. Likelihood noise is not
    /// added to `K_mm`. Also forms the VFE factors used by
    /// [`FittedSgpr::predict`] and
    /// [`FittedSgpr::neg_log_marginal_likelihood`].
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
    /// use gprx::{Fixed, GaussianLikelihood, Sgpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    /// let likelihood = GaussianLikelihood::new(0.1)?;
    /// let fitted = Sgpr::new(kernel, likelihood)
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
    ) -> Result<FittedSgpr<Fixed, I, P>, (Self, GprError)> {
        match with_kernel_exp!(self.spec.math, M => assemble_fitted::<_, _, M, _>(
            self.spec.kernel.clone(),
            self.spec.likelihood,
            Fixed,
            x,
            n_rows,
            n_cols,
            y,
            z,
            n_inducing,
        )) {
            Ok(fitted) => Ok(fitted),
            Err(err) => Err((self, err)),
        }
    }
}
