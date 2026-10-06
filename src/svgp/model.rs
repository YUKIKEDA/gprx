//! Trainer for stochastic variational GPR.

use std::marker::PhantomData;

use crate::error::GprError;
use crate::policy::{JitterPolicy, KernelExp, with_kernel_exp};
use crate::sparse::{SparseCore, SparseSpec};
use crate::transform::{UnfittedTarget, UnfittedTransform};

use crate::kernel::{
    DistanceKernel, KernelSpec, ModelKernel, ModelKernelParts, PointKernel, PointUse,
};
use crate::likelihood::GaussianLikelihood;
use crate::optimizer::{Adam, Fixed};
use crate::precision::{DoublePrecision, GpScalar};

use super::factor::{assemble_fitted, run_adam_fit};
use super::fitted::FittedSvgp;

/// Represents the trainer for stochastic variational GPR at a caller-supplied inducing set `Z`.
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
pub struct Svgp<O = Fixed, P = DoublePrecision, K: ModelKernel = KernelSpec> {
    pub(super) spec: SparseSpec<K>,
    pub(super) optimizer: O,
    pub(super) _precision: PhantomData<P>,
    pub(super) _kernel: PhantomData<K>,
}

impl<K: ModelKernel> Svgp<Fixed, DoublePrecision, K> {
    /// Builds a trainer with the current kernel `θ` and [`Fixed`].
    ///
    /// Inducing coordinates are an argument of [`Svgp<Fixed>::factor`], not
    /// of this constructor. The whitened prior is allocated at `factor` from
    /// the inducing count.
    ///
    /// See the example on [`Svgp`].
    ///
    /// A [`DistanceKernel`](crate::kernel::DistanceKernel) is read from
    /// supplied squared distances; its `factor` and `fit` take the training
    /// indices of the inducing points (see [`crate::kernel::ScalarDistance`]).
    pub fn new(kernel: K, likelihood: GaussianLikelihood) -> Self {
        Self {
            spec: SparseSpec::new(<K as ModelKernelParts>::into_spec(kernel), likelihood),
            optimizer: Fixed,
            _precision: PhantomData,
            _kernel: PhantomData,
        }
    }
}

impl<O, P, K: ModelKernel> Svgp<O, P, K> {
    /// The same settings under new type parameters, with `map` applied to
    /// the optimizer.
    fn retype<O2, P2>(self, map: impl FnOnce(O) -> O2) -> Svgp<O2, P2, K> {
        Svgp {
            spec: self.spec,
            optimizer: map(self.optimizer),
            _precision: PhantomData,
            _kernel: PhantomData,
        }
    }

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
    pub fn with_optimizer<O2>(self, optimizer: O2) -> Svgp<O2, P, K> {
        self.retype(|_| optimizer)
    }

    /// Selects the storage precision.
    ///
    /// Omitting it leaves [`DoublePrecision`].
    ///
    /// See the example on [`Svgp`].
    pub fn with_precision<P2: GpScalar>(self) -> Svgp<O, P2, K> {
        self.retype(|optimizer| optimizer)
    }

    /// Selects the kernel `exp`.
    ///
    /// Omitting it leaves [`KernelExp::Accurate`].
    ///
    /// `factor`, `fit`, and predict use the same polynomial. Hyperparameter
    /// `exp(θ)` is unchanged.
    ///
    /// See the example on [`Svgp`].
    pub fn with_math(mut self, math: KernelExp) -> Self {
        self.spec.math = math;
        self
    }

    /// Returns the kernel `exp` mode.
    ///
    /// See the example on [`Svgp`].
    pub fn math(&self) -> KernelExp {
        self.spec.math
    }

    /// Replaces the jitter retries for factoring `K_mm = k(Z, Z)`.
    ///
    /// The default is `JitterPolicy::adaptive(1e-8, 10.0, 5, 1e-3)`, not
    /// the [`JitterPolicy::default`] (no retry) of [`crate::Gpr`]: the
    /// Exact system `K + σn² I` carries the observation noise, `K_mm`
    /// does not, so close inducing points leave it singular in floating
    /// point. The policy applies wherever `K_mm` is factored: fit, factor,
    /// `set_params`, predict, and the online updates.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{GaussianLikelihood, JitterPolicy, Svgp};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let policy = JitterPolicy::fixed(1e-6)?;
    /// let fitted = Svgp::new(
    ///     KernelSpec::from(RbfKernel::new(1.0)?),
    ///     GaussianLikelihood::new(0.1)?,
    /// )
    /// .with_jitter_policy(policy)
    /// .factor(&[0.0, 1.0, 2.0, 3.0], 4, 1, &[0.0, 1.0, 0.5, 0.25], &[0.5, 2.5], 2)
    /// .map_err(|(_, e)| e)?;
    /// assert_eq!(fitted.jitter_policy(), policy);
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_jitter_policy(mut self, policy: JitterPolicy) -> Self {
        self.spec.jitter = policy;
        self
    }

    /// Returns the jitter retries for factoring `K_mm`.
    ///
    /// See the example on [`Svgp`].
    pub fn jitter_policy(&self) -> JitterPolicy {
        self.spec.jitter
    }

    /// Replaces the target (`y`) transform.
    ///
    /// Omitting it leaves identity.
    ///
    /// The map is fitted on training `y`. Predictions are mapped back to the
    /// original scale. The negative marginal likelihood and its gradient are
    /// those of the transformed `y`. A single map or a
    /// [`crate::transform::TargetPipeline`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::transform::StandardizeTarget;
    /// use gprx::{GaussianLikelihood, Svgp};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let fitted = Svgp::new(
    ///     KernelSpec::from(RbfKernel::new(1.0)?),
    ///     GaussianLikelihood::new(0.1)?,
    /// )
    /// .with_target_transform(StandardizeTarget::new())
    /// .factor(&[0.0, 1.0, 2.0, 3.0], 4, 1, &[100.0, 101.0, 100.5, 100.25], &[0.5, 2.5], 2)
    /// .map_err(|(_, e)| e)?;
    /// let pred = fitted.predict(&[1.0], 1, 1)?;
    /// assert!(pred.mean[0] > 99.0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_target_transform(mut self, transform: impl UnfittedTarget + 'static) -> Self {
        self.spec.y_transform = Box::new(transform);
        self
    }

    /// Returns the observation-noise model.
    ///
    /// See the example on [`Svgp`].
    pub fn likelihood(&self) -> &GaussianLikelihood {
        &self.spec.likelihood
    }

    /// Returns the concatenated kernel and likelihood parameter count.
    ///
    /// Inducing coordinates and the variational posterior are not counted.
    ///
    /// See the example on [`Svgp`].
    pub fn num_params(&self) -> usize {
        self.spec.theta_len()
    }

    /// Writes kernel `θ` then likelihood `θ` into `out`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::LengthMismatch`] if `out` is the wrong length
    /// or a custom leaf rejects the write.
    ///
    /// See the example on [`Svgp`].
    pub fn get_params(&self, out: &mut [f64]) -> Result<(), GprError> {
        self.spec.read_theta(out)
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
    ///
    /// See the example on [`Svgp`].
    pub fn set_params(&mut self, params: &[f64]) -> Result<(), GprError> {
        self.spec.write_theta(params)
    }
}

impl<O, P, K: PointKernel> Svgp<O, P, K> {
    /// Replaces the input (`X`) transform.
    ///
    /// Omitting it leaves identity.
    ///
    /// The map is fitted on training `X`. `X`, the inducing points `Z`, and
    /// every later query or inserted point go through it, so `Z` is passed
    /// in the same coordinates as `X`. [`crate::FreeInducing`] searches `Z`
    /// in the transformed coordinates; the fitted model reports `Z` in the
    /// original ones. A single map, a [`crate::transform::Pipeline`], or
    /// [`crate::transform::ColumnwiseInput`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::transform::StandardizeInput;
    /// use gprx::{GaussianLikelihood, Svgp};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let fitted = Svgp::new(
    ///     KernelSpec::from(RbfKernel::new(1.0)?),
    ///     GaussianLikelihood::new(0.1)?,
    /// )
    /// .with_input_transform(StandardizeInput::new())
    /// .factor(&[0.0, 10.0, 20.0, 30.0], 4, 1, &[0.0, 1.0, 0.5, 0.25], &[5.0, 25.0], 2)
    /// .map_err(|(_, e)| e)?;
    /// assert_eq!(fitted.z(), &[5.0, 25.0]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_input_transform(mut self, transform: impl UnfittedTransform + 'static) -> Self {
        self.spec.x_transform = Box::new(transform);
        self
    }
}

impl<O, P> Svgp<O, P> {
    /// Returns the kernel whose hyperparameters this trainer owns.
    ///
    /// See the example on [`Svgp`].
    pub fn kernel(&self) -> &KernelSpec {
        &self.spec.kernel
    }
}

impl<O, P, C: PointUse> Svgp<O, P, DistanceKernel<C>> {
    /// Returns a copy of the kernel whose hyperparameters this trainer owns.
    ///
    /// See the example on [`crate::kernel::ScalarDistance`].
    pub fn to_kernel(&self) -> DistanceKernel<C> {
        <DistanceKernel<C> as ModelKernelParts>::from_spec(self.spec.kernel.clone())
    }
}

impl<P> Svgp<Fixed, P>
where
    P: GpScalar,
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
    ) -> Result<FittedSvgp<P>, (Self, GprError)> {
        let core = match SparseCore::prepare(&self.spec, x, n_rows, n_cols, y, z, n_inducing) {
            Ok(core) => core,
            Err(err) => return Err((self, err)),
        };
        match with_kernel_exp!(self.spec.math, M => assemble_fitted::<M, _, _>(core, None)) {
            Ok(fitted) => Ok(fitted),
            Err(err) => Err((self, err)),
        }
    }
}

impl<P> Svgp<Adam, P>
where
    P: GpScalar,
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
    ) -> Result<FittedSvgp<P>, (Self, GprError)> {
        let core = match SparseCore::prepare(&self.spec, x, n_rows, n_cols, y, z, n_inducing) {
            Ok(core) => core,
            Err(err) => return Err((self, err)),
        };
        match with_kernel_exp!(self.spec.math, M => assemble_fitted::<M, _, _>(core, None)) {
            Ok(mut fitted) => match with_kernel_exp!(
                self.spec.math,
                M => run_adam_fit::<M, _, _>(&mut fitted, &self.optimizer)
            ) {
                Ok(()) => Ok(fitted),
                Err(err) => Err((self, err)),
            },
            Err(err) => Err((self, err)),
        }
    }
}
