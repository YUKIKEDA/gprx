//! SVGP on supplied squared distances: the `factor`, `fit`, and predict of
//! [`Svgp`] and [`FittedSvgp`] for a [`DistanceKernel`].
//!
//! The inducing points are training points, named by index. Every other
//! method is the one of the coordinate model.

use crate::error::GprError;
use crate::kernel::{
    DistanceKernel, DistanceOnly, DistanceSource, ModelKernel, PointUse, WithPoints,
};
use crate::optimizer::{Adam, Fixed};
use crate::policy::with_kernel_exp;
use crate::precision::GpScalar;
use crate::prediction::{QueryPoints, distance_predict};
use crate::sparse::{QueryDist, SparseCore, sparse_query};
use crate::{PredictOptions, Prediction, PredictiveCovariance};

use super::factor::{assemble_fitted, run_adam_fit};
use super::{FittedSvgp, Svgp};

impl<P: GpScalar, K: ModelKernel> Svgp<Fixed, P, K> {
    /// Factors `K_mm` of `core` and installs the whitened prior.
    #[allow(clippy::result_large_err)]
    fn assemble(
        self,
        core: Result<SparseCore, GprError>,
    ) -> Result<FittedSvgp<P, K>, (Self, GprError)> {
        let core = match core {
            Ok(core) => core,
            Err(err) => return Err((self, err)),
        };
        match with_kernel_exp!(self.spec.math, M => assemble_fitted::<M, _, _>(core, None)) {
            Ok(fitted) => Ok(fitted),
            Err(err) => Err((self, err)),
        }
    }
}

impl<P: GpScalar, K: ModelKernel> Svgp<Adam, P, K> {
    /// Factors `K_mm` of `core` and runs mini-batch Adam from the prior.
    #[allow(clippy::result_large_err)]
    fn train(
        self,
        core: Result<SparseCore, GprError>,
    ) -> Result<FittedSvgp<P, K>, (Self, GprError)> {
        let core = match core {
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

impl<P: GpScalar> Svgp<Fixed, P, DistanceKernel<DistanceOnly>> {
    /// Factors `K_mm` on supplied squared distances and installs a whitened
    /// prior `q(u)` at the current `θ`.
    ///
    /// `sources` holds one `n × n` training square per slot (zero diagonal,
    /// symmetric); the model keeps a copy. `inducing` names the training
    /// points that are the inducing points: `K_mm` and `K_mn` read their
    /// rows of the squares.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::EmptyInput`] if `n` or `inducing` is empty,
    /// [`GprError::IndexOutOfRange`] for an index `≥ n`,
    /// [`GprError::LengthMismatch`] if a table has the wrong length or a slot
    /// has no source or two, [`GprError::ShapeMismatch`] for a non-zero
    /// diagonal or an asymmetric square, and [`GprError::CholeskyFailed`]
    /// when `K_mm` cannot be factored.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{RbfKernel, ScalarDistance};
    /// use gprx::{GaussianLikelihood, Svgp};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// // Four points 0, 1, 2, 3 on a line: d²[i + j·4] = (i − j)².
    /// let image = ScalarDistance::new();
    /// let d2 = vec![0.0, 1.0, 4.0, 9.0, 1.0, 0.0, 1.0, 4.0, 4.0, 1.0, 0.0, 1.0, 9.0, 4.0, 1.0, 0.0];
    /// let fitted = Svgp::new(image.kernel(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?)
    ///     .factor([image.from_vec(d2)], 4, &[0.0, 1.0, 0.5, 0.25], &[0, 2])
    ///     .map_err(|(_, e)| e)?;
    /// assert_eq!(fitted.inducing(), &[0, 2]);
    /// assert_eq!(fitted.slots().len(), 1);
    /// let _kernel = fitted.to_kernel();
    /// let pred = fitted.predict([image.borrow(&[0.25, 0.25, 2.25, 6.25])], 1)?;
    /// assert!(pred.mean[0].is_finite());
    /// # Ok(())
    /// # }
    /// ```
    #[allow(clippy::result_large_err)]
    pub fn factor<'s>(
        self,
        sources: impl IntoIterator<Item = DistanceSource<'s>>,
        n: usize,
        y: &[f64],
        inducing: &[usize],
    ) -> Result<FittedSvgp<P, DistanceKernel<DistanceOnly>>, (Self, GprError)> {
        let core = SparseCore::prepare_with_distances(
            &self.spec,
            None,
            n,
            y,
            sources.into_iter().collect(),
            inducing,
        );
        self.assemble(core)
    }
}

impl<P: GpScalar> Svgp<Adam, P, DistanceKernel<DistanceOnly>> {
    /// Factors a whitened prior on supplied squared distances and runs
    /// mini-batch Adam on `θ` and `q`.
    ///
    /// # Errors
    ///
    /// The input errors of [`Svgp<Fixed>::factor`] of a
    /// [`DistanceKernel<DistanceOnly>`], and the step errors of the
    /// coordinate [`Svgp<Adam>::fit`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{RbfKernel, ScalarDistance};
    /// use gprx::{Adam, GaussianLikelihood, Svgp};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let image = ScalarDistance::new();
    /// let d2 = vec![0.0, 1.0, 4.0, 9.0, 1.0, 0.0, 1.0, 4.0, 4.0, 1.0, 0.0, 1.0, 9.0, 4.0, 1.0, 0.0];
    /// let fitted = Svgp::new(image.kernel(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?)
    ///     .with_optimizer(Adam::new())
    ///     .fit([image.from_vec(d2)], 4, &[0.0, 1.0, 0.5, 0.25], &[0, 2])
    ///     .map_err(|(_, e)| e)?;
    /// assert!(fitted.neg_elbo()?.is_finite());
    /// # Ok(())
    /// # }
    /// ```
    #[allow(clippy::result_large_err)]
    pub fn fit<'s>(
        self,
        sources: impl IntoIterator<Item = DistanceSource<'s>>,
        n: usize,
        y: &[f64],
        inducing: &[usize],
    ) -> Result<FittedSvgp<P, DistanceKernel<DistanceOnly>>, (Self, GprError)> {
        let core = SparseCore::prepare_with_distances(
            &self.spec,
            None,
            n,
            y,
            sources.into_iter().collect(),
            inducing,
        );
        self.train(core)
    }
}

impl<P: GpScalar> Svgp<Fixed, P, DistanceKernel<WithPoints>> {
    /// Factors `K_mm` on supplied squared distances and the column-major `x`
    /// (`n × n_cols`) of the coordinate leaves, and installs a whitened
    /// prior.
    ///
    /// The inducing points are the rows `inducing` of `x` and of the
    /// training squares.
    ///
    /// # Errors
    ///
    /// Same as the [`DistanceKernel<DistanceOnly>`] factor, plus the
    /// coordinate errors of [`Svgp<Fixed>::factor`] for `x`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel, ScalarDistance};
    /// use gprx::{GaussianLikelihood, Svgp};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let image = ScalarDistance::new();
    /// let kernel = image.kernel(RbfKernel::new(1.0)?) * KernelSpec::from(RbfKernel::new(2.0)?);
    /// let d2 = vec![0.0, 1.0, 4.0, 9.0, 1.0, 0.0, 1.0, 4.0, 4.0, 1.0, 0.0, 1.0, 9.0, 4.0, 1.0, 0.0];
    /// let fitted = Svgp::new(kernel, GaussianLikelihood::new(0.1)?)
    ///     .factor([image.from_vec(d2)], 4, &[0.0, 1.0, 2.0, 3.0], 1, &[0.0, 1.0, 0.5, 0.25], &[0, 2])
    ///     .map_err(|(_, e)| e)?;
    /// assert_eq!(fitted.z(), &[0.0, 2.0]);
    /// let pred = fitted.predict([image.borrow(&[0.25, 0.25, 2.25, 6.25])], &[0.5], 1, 1)?;
    /// assert!(pred.mean[0].is_finite());
    /// # Ok(())
    /// # }
    /// ```
    #[allow(clippy::result_large_err, clippy::too_many_arguments)]
    pub fn factor<'s>(
        self,
        sources: impl IntoIterator<Item = DistanceSource<'s>>,
        n: usize,
        x: &[f64],
        n_cols: usize,
        y: &[f64],
        inducing: &[usize],
    ) -> Result<FittedSvgp<P, DistanceKernel<WithPoints>>, (Self, GprError)> {
        let core = SparseCore::prepare_with_distances(
            &self.spec,
            Some((x, n_cols)),
            n,
            y,
            sources.into_iter().collect(),
            inducing,
        );
        self.assemble(core)
    }
}

impl<P: GpScalar> Svgp<Adam, P, DistanceKernel<WithPoints>> {
    /// Factors a whitened prior on supplied squared distances and `x`, and
    /// runs mini-batch Adam on `θ` and `q`.
    ///
    /// # Errors
    ///
    /// The input errors of [`Svgp<Fixed>::factor`] of a
    /// [`DistanceKernel<WithPoints>`], and the step errors of the coordinate
    /// [`Svgp<Adam>::fit`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel, ScalarDistance};
    /// use gprx::{Adam, GaussianLikelihood, Svgp};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let image = ScalarDistance::new();
    /// let kernel = image.kernel(RbfKernel::new(1.0)?) + KernelSpec::from(RbfKernel::new(2.0)?);
    /// let d2 = vec![0.0, 1.0, 4.0, 9.0, 1.0, 0.0, 1.0, 4.0, 4.0, 1.0, 0.0, 1.0, 9.0, 4.0, 1.0, 0.0];
    /// let fitted = Svgp::new(kernel, GaussianLikelihood::new(0.1)?)
    ///     .with_optimizer(Adam::new())
    ///     .fit([image.from_vec(d2)], 4, &[0.0, 1.0, 2.0, 3.0], 1, &[0.0, 1.0, 0.5, 0.25], &[1, 3])
    ///     .map_err(|(_, e)| e)?;
    /// assert_eq!(fitted.inducing(), &[1, 3]);
    /// # Ok(())
    /// # }
    /// ```
    #[allow(clippy::result_large_err, clippy::too_many_arguments)]
    pub fn fit<'s>(
        self,
        sources: impl IntoIterator<Item = DistanceSource<'s>>,
        n: usize,
        x: &[f64],
        n_cols: usize,
        y: &[f64],
        inducing: &[usize],
    ) -> Result<FittedSvgp<P, DistanceKernel<WithPoints>>, (Self, GprError)> {
        let core = SparseCore::prepare_with_distances(
            &self.spec,
            Some((x, n_cols)),
            n,
            y,
            sources.into_iter().collect(),
            inducing,
        );
        self.train(core)
    }
}

sparse_query!(impl [P: GpScalar, C: PointUse] FittedSvgp<P, DistanceKernel<C>>);

distance_predict!(
    impl [P: GpScalar] FittedSvgp<P, DistanceKernel<DistanceOnly>>,
    refine = P::Refine,
    args = (),
    tail = (),
    points = QueryPoints::NONE,
    reads = {
        /// The model keeps the rows of its inducing points for this call.
    },
    predict_doc = {
        /// See the example on [`Svgp<Fixed>::factor`] of a [`DistanceKernel<DistanceOnly>`].
    },
    covariance_doc = {
        /// # Examples
        ///
        /// ```rust
        /// use gprx::kernel::{RbfKernel, ScalarDistance};
        /// use gprx::{GaussianLikelihood, PredictOptions, Prediction, Svgp};
        ///
        /// # fn main() -> Result<(), gprx::GprError> {
        /// let image = ScalarDistance::new();
        /// let d2 = vec![0.0, 1.0, 4.0, 9.0, 1.0, 0.0, 1.0, 4.0, 4.0, 1.0, 0.0, 1.0, 9.0, 4.0, 1.0, 0.0];
        /// let mut fitted = Svgp::new(image.kernel(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?)
        ///     .factor([image.from_vec(d2)], 4, &[0.0, 1.0, 0.5, 0.25], &[0, 2])
        ///     .map_err(|(_, e)| e)?;
        /// // Queries at 0.5 and 1.5: train × query, then query × query.
        /// let cross = [0.25, 0.25, 2.25, 6.25, 2.25, 0.25, 0.25, 2.25];
        /// let query = [0.0, 1.0, 1.0, 0.0];
        /// let options = PredictOptions::default();
        /// let mut out = Prediction::default();
        /// fitted.predict_into([image.borrow(&cross)], 2, &mut out)?;
        /// fitted.predict_with_into([image.borrow(&cross)], 2, options, &mut out)?;
        /// let _ = fitted.predict_with([image.borrow(&cross)], 2, options)?;
        /// let cov = fitted.predict_covariance([image.borrow(&cross)], [image.borrow(&query)], 2)?;
        /// assert_eq!(cov.covariance.len(), 4);
        /// let _ = fitted.predict_covariance_with([image.borrow(&cross)], [image.borrow(&query)], 2, options)?;
        /// let draws = fitted.sample([image.borrow(&cross)], [image.borrow(&query)], 2, 3, 7)?;
        /// assert_eq!(draws.len(), 6);
        /// let _ = fitted.sample_with([image.borrow(&cross)], [image.borrow(&query)], 2, options, 3, 7)?;
        /// # Ok(())
        /// # }
        /// ```
    },
);

distance_predict!(
    impl [P: GpScalar] FittedSvgp<P, DistanceKernel<WithPoints>>,
    refine = P::Refine,
    args = (xs: &[f64]),
    tail = (n_cols: usize),
    points = QueryPoints { xs, n_cols },
    reads = {
        /// The model keeps the rows of its inducing points for this call.
    },
    predict_doc = {
        /// See the example on [`Svgp<Fixed>::factor`] of a [`DistanceKernel<WithPoints>`].
    },
    covariance_doc = {
        /// # Examples
        ///
        /// ```rust
        /// use gprx::kernel::{KernelSpec, RbfKernel, ScalarDistance};
        /// use gprx::{GaussianLikelihood, PredictOptions, Prediction, Svgp};
        ///
        /// # fn main() -> Result<(), gprx::GprError> {
        /// let image = ScalarDistance::new();
        /// let kernel = image.kernel(RbfKernel::new(1.0)?) * KernelSpec::from(RbfKernel::new(2.0)?);
        /// let d2 = vec![0.0, 1.0, 4.0, 9.0, 1.0, 0.0, 1.0, 4.0, 4.0, 1.0, 0.0, 1.0, 9.0, 4.0, 1.0, 0.0];
        /// let mut fitted = Svgp::new(kernel, GaussianLikelihood::new(0.1)?)
        ///     .factor([image.from_vec(d2)], 4, &[0.0, 1.0, 2.0, 3.0], 1, &[0.0, 1.0, 0.5, 0.25], &[0, 2])
        ///     .map_err(|(_, e)| e)?;
        /// let cross = [0.25, 0.25, 2.25, 6.25, 2.25, 0.25, 0.25, 2.25];
        /// let (query, xs) = ([0.0, 1.0, 1.0, 0.0], [0.5, 1.5]);
        /// let options = PredictOptions::default();
        /// let mut out = Prediction::default();
        /// fitted.predict_into([image.borrow(&cross)], &xs, 2, 1, &mut out)?;
        /// fitted.predict_with_into([image.borrow(&cross)], &xs, 2, 1, options, &mut out)?;
        /// let _ = fitted.predict_with([image.borrow(&cross)], &xs, 2, 1, options)?;
        /// let cov = fitted.predict_covariance([image.borrow(&cross)], [image.borrow(&query)], &xs, 2, 1)?;
        /// assert_eq!(cov.mean.len(), 2);
        /// let _ = fitted.predict_covariance_with([image.borrow(&cross)], [image.borrow(&query)], &xs, 2, 1, options)?;
        /// let _ = fitted.sample([image.borrow(&cross)], [image.borrow(&query)], &xs, 2, 1, 2, 0)?;
        /// let _ = fitted.sample_with([image.borrow(&cross)], [image.borrow(&query)], &xs, 2, 1, options, 2, 0)?;
        /// # Ok(())
        /// # }
        /// ```
    },
);
