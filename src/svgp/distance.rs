//! Stochastic variational GPR on supplied squared distances: the `fit`,
//! `factor`, and predict of [`Svgp`] and [`FittedSvgp`] for a
//! [`DistanceKernel`].
//!
//! The inducing points are training points, named by their indices, as in
//! the SGPR on supplied distances: a model takes, per slot, the `n × m`
//! block from the `n` training points to the `m` inducing points; a
//! prediction the `m × q` block from the inducing points to the `q`
//! queries, and a covariance the `q × q` square among the queries. A
//! mini-batch step reads the batch's rows of the stored blocks.

use crate::error::GprError;
use crate::kernel::{
    DistanceKernel, DistanceOnly, DistanceSlot, DistanceSource, PointUse, SuppliedSpec, WithPoints,
};
use crate::optimizer::{Adam, Fixed};
use crate::policy::{JitterPolicy, with_kernel_exp};
use crate::precision::GpScalar;
use crate::prediction::{DistanceQuery, QueryPoints, distance_predict};
use crate::sparse::{PredictScratch, SparseCore};
use crate::{PredictOptions, Prediction, PredictiveCovariance};

use super::factor::{
    SvgpSystem, assemble_fitted, predict_svgp_covariance, predict_svgp_into, run_adam_fit,
};
use super::{FittedSvgp, Svgp};

impl<O, P: GpScalar, C: PointUse> Svgp<O, P, DistanceKernel<C>> {
    /// `K_mm`, `A`, and the whitened prior `q(u)` at the current `θ`: the
    /// shared part of `fit` and `factor`.
    fn factor_supplied<'s>(
        &self,
        sources: impl IntoIterator<Item = DistanceSource<'s>>,
        n: usize,
        points: (&[f64], usize),
        y: &[f64],
        inducing: &[usize],
    ) -> Result<FittedSvgp<P, DistanceKernel<C>>, GprError> {
        let core = SparseCore::prepare_supplied::<P::Storage>(
            &self.spec, sources, n, points, y, inducing,
        )?;
        with_kernel_exp!(self.spec.math, M => assemble_fitted::<M, _, DistanceKernel<C>>(core, None))
    }
}

impl<P: GpScalar, C: PointUse> Svgp<Adam, P, DistanceKernel<C>> {
    /// [`Self::factor_supplied`], then mini-batch Adam.
    fn fit_supplied<'s>(
        &self,
        sources: impl IntoIterator<Item = DistanceSource<'s>>,
        n: usize,
        points: (&[f64], usize),
        y: &[f64],
        inducing: &[usize],
    ) -> Result<FittedSvgp<P, DistanceKernel<C>>, GprError> {
        let mut fitted = self.factor_supplied(sources, n, points, y, inducing)?;
        with_kernel_exp!(
            self.spec.math,
            M => run_adam_fit::<M, _, DistanceKernel<C>>(&mut fitted, &self.optimizer)
        )?;
        Ok(fitted)
    }
}

impl<P: GpScalar> Svgp<Fixed, P, DistanceKernel<DistanceOnly>> {
    /// Factors `K_mm` on the supplied distances and installs a whitened
    /// prior `q(u)` at the current `θ`.
    ///
    /// `sources` holds one source per slot of the kernel, in any order: the
    /// `n × m` squared distances from the `n` training samples to the `m`
    /// inducing points. The inducing points are the training samples
    /// `inducing`, in that order: column `a` of a block is the training
    /// sample `inducing[a]`, and `K_mm` reads their rows of it. The model
    /// keeps the blocks. `y` has `n` targets.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::EmptyInput`] if `n` or `inducing` is empty,
    /// [`GprError::IndexOutOfRange`] for an inducing index not below `n`,
    /// [`GprError::InvalidConfig`] for an inducing index listed twice,
    /// [`GprError::LengthMismatch`] if a table or `y` has the wrong length,
    /// [`GprError::DistanceSlot`] if a source names a slot the kernel does
    /// not read, two name one slot, or a slot has none,
    /// [`GprError::InvalidDistance`] for a value that is not
    /// finite or is negative, or for inducing rows that are not a square
    /// with a zero diagonal and equal mirror entries (see
    /// [`crate::kernel::DistanceSource::tidy`]), and the errors of the
    /// coordinate [`Svgp::factor`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{RbfKernel, ScalarDistance};
    /// use gprx::{GaussianLikelihood, Svgp};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let image = ScalarDistance::new();
    /// // Three samples on a line at 0, 1, 2; the inducing points are samples 0 and 2.
    /// // Column-major 3 × 2: the squared distances to sample 0, then to sample 2.
    /// let train = [0.0, 1.0, 4.0, 4.0, 1.0, 0.0];
    /// let fitted = Svgp::new(image.kernel(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?)
    ///     .factor([image.borrow(&train)], 3, &[0.0, 1.0, 0.5], &[0, 2])
    ///     .map_err(|(_, e)| e)?;
    /// assert_eq!(fitted.inducing(), &[0, 2]);
    /// // One query at 1.5: its squared distances to the two inducing points (2 × 1).
    /// let pred = fitted.predict([image.borrow(&[2.25, 0.25])], 1)?;
    /// assert_eq!(pred.mean.len(), 1);
    /// # Ok(())
    /// # }
    /// ```
    #[allow(clippy::result_large_err)] // failure returns the trainer so the caller can retry
    pub fn factor<'s>(
        self,
        sources: impl IntoIterator<Item = DistanceSource<'s>>,
        n: usize,
        y: &[f64],
        inducing: &[usize],
    ) -> Result<FittedSvgp<P, DistanceKernel>, (Self, GprError)> {
        match self.factor_supplied(sources, n, (&[], 0), y, inducing) {
            Ok(fitted) => Ok(fitted),
            Err(err) => Err((self, err)),
        }
    }
}

impl<P: GpScalar> Svgp<Adam, P, DistanceKernel<DistanceOnly>> {
    /// Factors on the supplied distances, then runs mini-batch Adam from
    /// the whitened prior. Same data contract as [`Svgp::factor`] of this
    /// kernel; a mini-batch step reads the batch's rows of the stored
    /// blocks.
    ///
    /// # Errors
    ///
    /// Same as [`Svgp::factor`] of this kernel, plus the errors of the
    /// coordinate [`Svgp::fit`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{RbfKernel, ScalarDistance};
    /// use gprx::{Adam, GaussianLikelihood, Svgp};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let image = ScalarDistance::new();
    /// let train = [0.0, 1.0, 4.0, 4.0, 1.0, 0.0];
    /// let fitted = Svgp::new(image.kernel(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?)
    ///     .with_optimizer(Adam::new())
    ///     .fit([image.borrow(&train)], 3, &[0.0, 1.0, 0.5], &[0, 2])
    ///     .map_err(|(_, e)| e)?;
    /// assert_eq!(fitted.m(), 2);
    /// # Ok(())
    /// # }
    /// ```
    #[allow(clippy::result_large_err)] // failure returns the trainer so the caller can retry
    pub fn fit<'s>(
        self,
        sources: impl IntoIterator<Item = DistanceSource<'s>>,
        n: usize,
        y: &[f64],
        inducing: &[usize],
    ) -> Result<FittedSvgp<P, DistanceKernel>, (Self, GprError)> {
        match self.fit_supplied(sources, n, (&[], 0), y, inducing) {
            Ok(fitted) => Ok(fitted),
            Err(err) => Err((self, err)),
        }
    }
}

impl<P: GpScalar> Svgp<Fixed, P, DistanceKernel<WithPoints>> {
    /// Factors `K_mm` on the supplied distances and the coordinates `x`,
    /// and installs a whitened prior `q(u)` at the current `θ`.
    ///
    /// `x` is column-major, `n` points by `n_cols` features, and goes
    /// through the input transform; the distances do not. The inducing
    /// points' coordinates are their rows of `x`.
    ///
    /// # Errors
    ///
    /// Same as [`Svgp::factor`] of a [`DistanceKernel<DistanceOnly>`], plus
    /// [`GprError::EmptyInput`] if `x` is empty, [`GprError::LengthMismatch`]
    /// if `x` is not `n × n_cols` (`n_cols` zero included), [`GprError::NonFiniteInput`] for a
    /// coordinate that is not finite, and the error of the input transform.
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
    /// let train = [0.0, 1.0, 4.0, 4.0, 1.0, 0.0];
    /// let fitted = Svgp::new(kernel, GaussianLikelihood::new(0.1)?)
    ///     .factor([image.borrow(&train)], 3, &[0.0, 0.5, 1.0], 1, &[0.0, 1.0, 0.5], &[0, 2])
    ///     .map_err(|(_, e)| e)?;
    /// assert_eq!(fitted.z(), &[0.0, 1.0]);
    /// let pred = fitted.predict([image.borrow(&[2.25, 0.25])], &[0.75], 1, 1)?;
    /// assert_eq!(pred.mean.len(), 1);
    /// # Ok(())
    /// # }
    /// ```
    #[allow(clippy::result_large_err)] // failure returns the trainer so the caller can retry
    #[allow(clippy::too_many_arguments)]
    pub fn factor<'s>(
        self,
        sources: impl IntoIterator<Item = DistanceSource<'s>>,
        n: usize,
        x: &[f64],
        n_cols: usize,
        y: &[f64],
        inducing: &[usize],
    ) -> Result<FittedSvgp<P, DistanceKernel<WithPoints>>, (Self, GprError)> {
        match self.factor_supplied(sources, n, (x, n_cols), y, inducing) {
            Ok(fitted) => Ok(fitted),
            Err(err) => Err((self, err)),
        }
    }
}

impl<P: GpScalar> Svgp<Adam, P, DistanceKernel<WithPoints>> {
    /// Factors on the supplied distances and the coordinates `x`, then runs
    /// mini-batch Adam from the whitened prior. Same data contract as
    /// [`Svgp::factor`] of this kernel.
    ///
    /// # Errors
    ///
    /// Same as [`Svgp::factor`] of this kernel, plus the errors of the
    /// coordinate [`Svgp::fit`].
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
    /// let train = [0.0, 1.0, 4.0, 4.0, 1.0, 0.0];
    /// let fitted = Svgp::new(kernel, GaussianLikelihood::new(0.1)?)
    ///     .with_optimizer(Adam::new())
    ///     .fit([image.borrow(&train)], 3, &[0.0, 0.5, 1.0], 1, &[0.0, 1.0, 0.5], &[0, 2])
    ///     .map_err(|(_, e)| e)?;
    /// assert_eq!(fitted.d(), 1);
    /// # Ok(())
    /// # }
    /// ```
    #[allow(clippy::result_large_err)] // failure returns the trainer so the caller can retry
    #[allow(clippy::too_many_arguments)]
    pub fn fit<'s>(
        self,
        sources: impl IntoIterator<Item = DistanceSource<'s>>,
        n: usize,
        x: &[f64],
        n_cols: usize,
        y: &[f64],
        inducing: &[usize],
    ) -> Result<FittedSvgp<P, DistanceKernel<WithPoints>>, (Self, GprError)> {
        match self.fit_supplied(sources, n, (x, n_cols), y, inducing) {
            Ok(fitted) => Ok(fitted),
            Err(err) => Err((self, err)),
        }
    }
}

impl<P: GpScalar, C: PointUse> FittedSvgp<P, DistanceKernel<C>> {
    /// Writes this model to `dir` as `config.json` and `model.safetensors`.
    ///
    /// Stores what [`FittedSvgp::save`] of a coordinate kernel stores, with
    /// the kernel's slot table, each slot's `n × m` training blocks (in
    /// `f64` at every precision), and the inducing indices
    /// ([`Self::inducing`]). [`crate::persist::LoadedDistanceSvgp::load`]
    /// binds the blocks to new slots and factors `K_mm` again at the saved
    /// `θ`, with the saved `q(u)`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::PersistFailed`] when the directory cannot be
    /// written or a kernel or transform has no persist form.
    ///
    /// See the example on [`crate::persist::LoadedDistanceSvgp`].
    pub fn save(&self, dir: impl AsRef<std::path::Path>) -> Result<(), GprError> {
        crate::persist::save_svgp(self, dir.as_ref())
    }

    /// Returns a copy of the kernel whose hyperparameters this model owns.
    ///
    /// See the example on [`DistanceKernel`].
    pub fn to_kernel(&self) -> DistanceKernel<C> {
        <DistanceKernel<C> as crate::kernel::ModelKernelParts>::from_spec(self.core.kernel.clone())
    }

    /// Returns the slots of the kernel, in the order of
    /// [`DistanceKernel::slots`]; bind supplies to these.
    ///
    /// See the example on [`DistanceKernel`].
    pub fn slots(&self) -> Vec<DistanceSlot> {
        crate::kernel::spec_slots(&self.core.kernel)
    }

    /// Returns the training samples that are the inducing points, in the
    /// order of the columns of the training blocks and the rows of a
    /// prediction's blocks.
    ///
    /// See the example on [`Svgp::factor`] of a [`DistanceKernel<DistanceOnly>`].
    pub fn inducing(&self) -> &[usize] {
        &self.core.supply.inducing
    }

    /// The factors and `q(u)` a prediction reads.
    fn svgp_system(&self) -> SvgpSystem<'_, P, SuppliedSpec> {
        SvgpSystem::new(
            &self.core,
            self.k_mm_l.as_ref(),
            &self.q_mean,
            self.q_l.as_ref(),
        )
    }
}

/// The query of an SVGP on supplied distances.
impl<P: GpScalar, C: PointUse> DistanceQuery for FittedSvgp<P, DistanceKernel<C>> {
    type Refine = P::Refine;

    fn query_distances<'s>(
        &self,
        cross: impl IntoIterator<Item = DistanceSource<'s>>,
        points: QueryPoints<'_>,
        q: usize,
        options: PredictOptions,
    ) -> Result<Prediction<P::Refine>, GprError> {
        let mut out = Prediction::default();
        predict_svgp_into::<P, SuppliedSpec>(
            &self.core,
            &self.svgp_system(),
            points.xs,
            q,
            points.n_cols,
            cross,
            options,
            &mut PredictScratch::default(),
            &mut out,
        )?;
        Ok(out)
    }

    fn query_distances_into<'s>(
        &mut self,
        cross: impl IntoIterator<Item = DistanceSource<'s>>,
        points: QueryPoints<'_>,
        q: usize,
        options: PredictOptions,
        out: &mut Prediction<P::Refine>,
    ) -> Result<(), GprError> {
        let sys = SvgpSystem::new(
            &self.core,
            self.k_mm_l.as_ref(),
            &self.q_mean,
            self.q_l.as_ref(),
        );
        predict_svgp_into::<P, SuppliedSpec>(
            &self.core,
            &sys,
            points.xs,
            q,
            points.n_cols,
            cross,
            options,
            &mut self.scratch.predict,
            out,
        )
    }

    fn query_distance_covariance<'s>(
        &self,
        cross: impl IntoIterator<Item = DistanceSource<'s>>,
        square: impl IntoIterator<Item = DistanceSource<'s>>,
        points: QueryPoints<'_>,
        q: usize,
        options: PredictOptions,
    ) -> Result<PredictiveCovariance<P::Refine>, GprError> {
        predict_svgp_covariance::<P, SuppliedSpec>(
            &self.core,
            &self.svgp_system(),
            points.xs,
            q,
            points.n_cols,
            cross,
            square,
            options,
        )
    }

    fn draw_jitter(&self) -> JitterPolicy {
        self.core.jitter
    }
}

distance_predict!(
    impl [P: GpScalar] FittedSvgp<P, DistanceKernel<DistanceOnly>>,
    refine = P::Refine,
    args = (),
    tail = (),
    points = QueryPoints::NONE,
    count = q,
    cross = {
        /// the `m × q` squared distances from the `m` inducing points (in
        /// [`Self::inducing`] order) to the `q` queries.
    },
    reads = {
        /// A table is read in place for this call (an `f32` model reads it
        /// in `f64`, as it predicts); a fill writes scratch once.
    },
    predict_doc = {
        /// See the example on [`Svgp::factor`] of a [`DistanceKernel<DistanceOnly>`].
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
        /// let train = [0.0, 1.0, 4.0, 4.0, 1.0, 0.0];
        /// let mut fitted = Svgp::new(image.kernel(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?)
        ///     .factor([image.borrow(&train)], 3, &[0.0, 1.0, 0.5], &[0, 2])
        ///     .map_err(|(_, e)| e)?;
        /// // Two queries at 0.5 and 1.5: inducing × query (2 × 2), then query × query.
        /// let cross = [0.25, 2.25, 2.25, 0.25];
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
    count = q,
    cross = {
        /// the `m × q` squared distances from the `m` inducing points (in
        /// [`Self::inducing`] order) to the `q` queries.
    },
    reads = {
        /// A table is read in place for this call (an `f32` model reads it
        /// in `f64`, as it predicts); a fill writes scratch once.
    },
    predict_doc = {
        /// See the example on [`Svgp::factor`] of a [`DistanceKernel<WithPoints>`].
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
        /// let train = [0.0, 1.0, 4.0, 4.0, 1.0, 0.0];
        /// let mut fitted = Svgp::new(kernel, GaussianLikelihood::new(0.1)?)
        ///     .factor([image.borrow(&train)], 3, &[0.0, 0.5, 1.0], 1, &[0.0, 1.0, 0.5], &[0, 2])
        ///     .map_err(|(_, e)| e)?;
        /// let (cross, query, xs) = ([0.25, 2.25, 2.25, 0.25], [0.0, 1.0, 1.0, 0.0], [0.25, 0.75]);
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
