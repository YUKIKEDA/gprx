//! Trainer for collapsed variational SGPR.

use std::marker::PhantomData;

use crate::error::GprError;
use crate::policy::{JitterPolicy, KernelExp, with_kernel_exp};
use crate::sparse::{SparseCore, SparseSpec};
use crate::transform::{UnfittedTarget, UnfittedTransform};

use crate::kernel::{KernelSpec, ModelKernel, ModelKernelParts, PointKernel};
use crate::likelihood::GaussianLikelihood;
use crate::optimizer::{Fixed, Lbfgs, Optimizer};
use crate::precision::{DoublePrecision, GpScalar};
use crate::sgpr::SgprObjective;

use super::factor::assemble_fitted;
use super::fitted::FittedSgpr;
use super::{FixedInducing, FreeInducing, InducingLayout};
use crate::kernel::NoSupply;

/// Represents the trainer for collapsed variational SGPR at an inducing set the caller chooses.
///
/// The type parameters are the optimizer `O` (default [`Lbfgs`]; [`Fixed`]
/// for [`Sgpr<Fixed, I>::factor`]), the inducing layout `I`, the precision
/// `P` (default [`DoublePrecision`]), and the kernel `K` (default
/// [`KernelSpec`]). The default [`FixedInducing`] searches kernel and
/// likelihood `θ` only. [`Self::with_inducing`]`(`[`FreeInducing`]`)`
/// searches `θ` and `Z` together. [`Sgpr<Fixed, I>::factor`] prepares the
/// VFE system at the current `θ` with no search.
///
/// A failed `fit` or `factor` returns the trainer with the [`GprError`], so
/// a caller can match `Err((trainer, err))` and try again. There is no
/// `From` conversion of that pair into [`GprError`]: write
/// `.map_err(|(_, e)| e)?`.
///
/// `K` selects how the inducing points are given. A [`KernelSpec`] model
/// takes their coordinates `Z` (column-major, `m` points by `d` features)
/// next to the training `X`. A [`crate::kernel::DistanceKernel`] model has
/// no coordinates to place them at: its inducing points are training
/// points, named by their indices, and the caller supplies per slot the
/// `n × m` squared distances from the training points to them (`K_mm`
/// reads the inducing points' rows of that block). A prediction then takes
/// the `m × q` squared distances from the inducing points to the queries.
/// Such a model keeps its inducing points fixed; [`FreeInducing`] needs
/// coordinates to move.
///
/// # Examples
///
/// ```rust
/// use gprx::kernel::{KernelSpec, RbfKernel};
/// use gprx::{
///     Fixed, FixedInducing, FreeInducing, GaussianLikelihood, JitterPolicy, KernelExp, Lbfgs,
///     Sgpr, SinglePrecision,
/// };
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
/// let likelihood = GaussianLikelihood::new(0.1)?;
/// let mut sgpr: Sgpr<Lbfgs, FixedInducing> =
///     Sgpr::new(kernel, likelihood).with_math(KernelExp::FastApprox);
/// assert_eq!(sgpr.math(), KernelExp::FastApprox);
/// assert!(matches!(sgpr.jitter_policy(), JitterPolicy::Adaptive(_)));
/// // θ is the kernel's log-lengthscale, then the likelihood's log σn²; `Z` is not in it.
/// assert_eq!(sgpr.num_params(), sgpr.kernel().num_params() + 1);
/// let mut theta = [0.0; 2];
/// sgpr.get_params(&mut theta)?;
/// theta[0] = 0.5_f64.ln();
/// sgpr.set_params(&theta)?;
/// assert!((sgpr.likelihood().log_noise_variance() - theta[1]).abs() < 1e-12);
///
/// // Column-major `X` (n = 4, d = 1) and the inducing coordinates `Z` (m = 2).
/// let (x, y, z) = ([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 0.5, 0.25], [0.5, 2.5]);
/// let fitted = sgpr.clone().fit(&x, 4, 1, &y, &z, 2).map_err(|(_, e)| e)?;
/// assert_eq!(fitted.n(), 4);
/// assert_eq!(fitted.m(), 2);
///
/// // `Z` searched with θ: two more parameters.
/// let free = sgpr.clone().with_inducing(FreeInducing).fit(&x, 4, 1, &y, &z, 2).map_err(|(_, e)| e)?;
/// assert_eq!(free.num_params(), 4);
///
/// // `f32` storage, factored at the current θ.
/// let single = sgpr
///     .with_precision::<SinglePrecision>()
///     .with_optimizer(Fixed)
///     .factor(&x, 4, 1, &y, &z, 2)
///     .map_err(|(_, e)| e)?;
/// assert_eq!(single.predict(&[1.5], 1, 1)?.mean.len(), 1);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct Sgpr<O = Lbfgs, I = FixedInducing, P = DoublePrecision, K: ModelKernel = KernelSpec> {
    pub(super) spec: SparseSpec<K::Supply>,
    pub(super) optimizer: O,
    pub(super) inducing: PhantomData<I>,
    pub(super) _precision: PhantomData<P>,
    pub(super) _kernel: PhantomData<K>,
}

impl<K: ModelKernel> Sgpr<Lbfgs, FixedInducing, DoublePrecision, K> {
    /// Builds a trainer with identity transforms, the current kernel `θ`, and [`Lbfgs`].
    ///
    /// Inducing coordinates are an argument of [`Sgpr::fit`] /
    /// [`Sgpr<Fixed>::factor`], not of this constructor. Call
    /// [`Self::with_optimizer`] to switch to [`Fixed`] or another
    /// [`Optimizer`].
    ///
    /// See the example on [`Sgpr`].
    pub fn new(kernel: K, likelihood: GaussianLikelihood) -> Self {
        Self {
            spec: SparseSpec::new(<K as ModelKernelParts>::into_spec(kernel), likelihood),
            optimizer: Lbfgs::new(),
            inducing: PhantomData,
            _precision: PhantomData,
            _kernel: PhantomData,
        }
    }
}

impl<O, I, P, K: ModelKernel> Sgpr<O, I, P, K> {
    /// The same settings under new type parameters, with `map` applied to
    /// the optimizer.
    fn retype<O2, I2, P2>(self, map: impl FnOnce(O) -> O2) -> Sgpr<O2, I2, P2, K> {
        Sgpr {
            spec: self.spec,
            optimizer: map(self.optimizer),
            inducing: PhantomData,
            _precision: PhantomData,
            _kernel: PhantomData,
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
    pub fn with_optimizer<O2>(self, optimizer: O2) -> Sgpr<O2, I, P, K> {
        self.retype(|_| optimizer)
    }

    /// Selects the storage precision.
    ///
    /// Omitting it leaves [`DoublePrecision`].
    ///
    /// See the example on [`Sgpr`].
    pub fn with_precision<P2: GpScalar>(self) -> Sgpr<O, I, P2, K> {
        self.retype(|optimizer| optimizer)
    }

    /// Selects the kernel `exp`.
    ///
    /// Omitting it leaves [`KernelExp::Accurate`].
    ///
    /// `fit`, `factor`, and predict use the same `exp`. Hyperparameter
    /// `exp(θ)` is unchanged.
    ///
    /// See the example on [`Sgpr`].
    pub fn with_math(mut self, math: KernelExp) -> Self {
        self.spec.math = math;
        self
    }

    /// Returns the kernel `exp` mode.
    ///
    /// See the example on [`Sgpr`].
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
    /// use gprx::{Fixed, GaussianLikelihood, JitterPolicy, Sgpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let policy = JitterPolicy::fixed(1e-6)?;
    /// let fitted = Sgpr::new(
    ///     KernelSpec::from(RbfKernel::new(1.0)?),
    ///     GaussianLikelihood::new(0.1)?,
    /// )
    /// .with_jitter_policy(policy)
    /// .with_optimizer(Fixed)
    ///     .factor(&[0.0, 1.0, 2.0, 3.0], 4, 1, &[0.0, 1.0, 0.5, 0.25], &[0.5, 2.5], 2)
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
    /// See the example on [`Sgpr`].
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
    /// use gprx::{Fixed, GaussianLikelihood, Sgpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let fitted = Sgpr::new(
    ///     KernelSpec::from(RbfKernel::new(1.0)?),
    ///     GaussianLikelihood::new(0.1)?,
    /// )
    /// .with_target_transform(StandardizeTarget::new())
    /// .with_optimizer(Fixed)
    ///     .factor(&[0.0, 1.0, 2.0, 3.0], 4, 1, &[100.0, 101.0, 100.5, 100.25], &[0.5, 2.5], 2)
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
    /// See the example on [`Sgpr`].
    pub fn likelihood(&self) -> &GaussianLikelihood {
        &self.spec.likelihood
    }

    /// Returns the concatenated kernel and likelihood parameter count.
    ///
    /// Inducing coordinates are not counted.
    ///
    /// See the example on [`Sgpr`].
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
    /// See the example on [`Sgpr`].
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
    /// length, [`GprError::InvalidHyperparameter`] or
    /// [`GprError::InvalidNoiseVariance`] if a kernel or likelihood `θ` is
    /// not finite or its `exp` is not a positive finite value, or
    /// [`GprError::InvalidInterval`] if a value is outside the interval of
    /// its parameter (see [`crate::BoundedParam`] and
    /// [`GaussianLikelihood::bounds`]). A custom kernel leaf returns its own
    /// error. Kernel and likelihood `θ` are committed together only after
    /// both writes succeed.
    ///
    /// See the example on [`Sgpr`].
    pub fn set_params(&mut self, params: &[f64]) -> Result<(), GprError> {
        self.spec.write_theta(params)
    }
}

impl<O, I, P, K: PointKernel> Sgpr<O, I, P, K> {
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
    /// use gprx::{Fixed, GaussianLikelihood, Sgpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let fitted = Sgpr::new(
    ///     KernelSpec::from(RbfKernel::new(1.0)?),
    ///     GaussianLikelihood::new(0.1)?,
    /// )
    /// .with_input_transform(StandardizeInput::new())
    /// .with_optimizer(Fixed)
    ///     .factor(&[0.0, 10.0, 20.0, 30.0], 4, 1, &[0.0, 1.0, 0.5, 0.25], &[5.0, 25.0], 2)
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

impl<O, I, P> Sgpr<O, I, P> {
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
    ///
    /// See the example on [`Sgpr`].
    pub fn kernel(&self) -> &KernelSpec {
        &self.spec.kernel
    }
}

impl<O, I, P, C: crate::kernel::PointUse> Sgpr<O, I, P, crate::kernel::DistanceKernel<C>> {
    /// Returns a copy of the kernel whose hyperparameters this trainer owns.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{RbfKernel, ScalarDistance};
    /// use gprx::{GaussianLikelihood, Sgpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let image = ScalarDistance::new();
    /// let sgpr = Sgpr::new(image.kernel(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?);
    /// assert_eq!(sgpr.to_kernel().num_params(), 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn to_kernel(&self) -> crate::kernel::DistanceKernel<C> {
        <crate::kernel::DistanceKernel<C> as ModelKernelParts>::from_spec(self.spec.kernel.clone())
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
    /// Returns [`GprError::LengthMismatch`] when `x`, `y`, or `z` has the
    /// wrong length, and [`GprError::NonFiniteInput`] when a value is `NaN`
    /// or `Inf`. [`GprError::CholeskyFailed`] when `K_mm` or the VFE `B`
    /// matrix cannot be factored. [`GprError::OptimizationNotConverged`] when the solver
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
        let core = match SparseCore::prepare(&self.spec, x, n_rows, n_cols, y, z, n_inducing) {
            Ok(core) => core,
            Err(err) => return Err((self, err)),
        };
        match with_kernel_exp!(self.spec.math, M => assemble_fitted::<_, _, M, _, KernelSpec>(
            core,
            self.optimizer.clone(),
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
    /// `Z`. Each coordinate of `Z` stays inside an interval built from the
    /// training inputs after the input transform: their range in that
    /// feature, opened on each side by 10% of the range and at least `0.1`,
    /// with the lower end moved down to `0` when it is above `0`.
    ///
    /// # Errors
    ///
    /// Same input errors as `fit` with [`FixedInducing`], plus
    /// [`GprError::CoordGradientUnsupported`] for Matérn with `ν = 1/2` or a
    /// `Custom` leaf that does not implement the squared-distance derivatives
    /// of [`KernelTerm`](crate::kernel::KernelTerm).
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
        let core = match SparseCore::prepare(&self.spec, x, n_rows, n_cols, y, z, n_inducing) {
            Ok(core) => core,
            Err(err) => return Err((self, err)),
        };
        match with_kernel_exp!(self.spec.math, M => assemble_fitted::<_, _, M, _, KernelSpec>(
            core,
            self.optimizer.clone(),
        )) {
            Ok(mut fitted) => match fitted.optimize_hyperparameters() {
                Ok(()) => Ok(fitted),
                Err(err) => Err((fitted.into_trainer(), err)),
            },
            Err(err) => Err((self, err)),
        }
    }
}

impl<I: InducingLayout<NoSupply>, P> Sgpr<Fixed, I, P>
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
    /// Returns [`GprError::LengthMismatch`] when `x`, `y`, or `z` has the
    /// wrong length, and [`GprError::NonFiniteInput`] when a value is `NaN`
    /// or `Inf`. [`GprError::CholeskyFailed`] when `K_mm` or the VFE `B`
    /// matrix cannot be factored.
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
        let core = match SparseCore::prepare(&self.spec, x, n_rows, n_cols, y, z, n_inducing) {
            Ok(core) => core,
            Err(err) => return Err((self, err)),
        };
        match with_kernel_exp!(self.spec.math, M => assemble_fitted::<_, _, M, _, KernelSpec>(
            core,
            Fixed,
        )) {
            Ok(fitted) => Ok(fitted),
            Err(err) => Err((self, err)),
        }
    }
}
