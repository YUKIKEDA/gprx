//! Exact GPR on supplied squared distances: the `fit`, `factor`, and
//! predict of [`Gpr`] and [`FittedGpr`] for a [`DistanceKernel`]. Online
//! insert and delete on supplied distances are not here yet (#473).
//!
//! A [`DistanceKernel<DistanceOnly>`] model takes no coordinates; a
//! [`DistanceKernel<WithPoints>`] model takes the column-major `x` of its
//! coordinate leaves next to the supplied distances. Every other method is
//! the one of the coordinate model.

use crate::error::GprError;
use crate::gpr::GprObjective;
use crate::kernel::{
    BlockKind, DistanceKernel, DistanceOnly, DistanceSlot, DistanceSource, KernelScalar, PointUse,
    QueryScratch, QuerySources, RectSlots, SquareSlots, TrainSources, WithPoints,
};
use crate::optimizer::{Fixed, Optimizer};
use crate::policy::JitterPolicy;
use crate::precision::GpScalar;
use crate::prediction::{DistanceQuery, QueryPoints, distance_predict};
use crate::{PredictOptions, Prediction, PredictiveCovariance};

use super::shared::Query;
use super::{FittedGpr, Gpr, TrainInput};

/// The checked blocks of one query: the `n × m` train × query blocks, and
/// for a covariance the `m × m` query squares as a store a Gram reads.
struct BoundQuery<'a, T: KernelScalar> {
    cross: QuerySources<'a, T>,
    square: Option<TrainSources<T>>,
}

/// Binds the blocks of `cross` on `scratch` and, for a covariance, the
/// query squares of `square`. The result borrows `scratch` and the caller's
/// tables, not `slots`, so the model can be borrowed again to run it.
fn bind_query<'a, 's: 'a, T: KernelScalar>(
    slots: &[DistanceSlot],
    (n, m): (usize, usize),
    cross: impl IntoIterator<Item = DistanceSource<'s>>,
    square: Option<Vec<DistanceSource<'s>>>,
    scratch: &'a mut QueryScratch<T>,
) -> Result<BoundQuery<'a, T>, GprError> {
    crate::data::require_nonempty(m)?;
    let cross = QuerySources::<T>::bind(slots, cross, n, m, BlockKind::Rect, scratch)?;
    // The query squares are one set: a Gram reads them, as it reads the
    // training squares.
    let square = match square {
        Some(square) => {
            let mut local = QueryScratch::new();
            let bound =
                QuerySources::<T>::bind(slots, square, m, m, BlockKind::Square, &mut local)?;
            Some(bound.to_square(m)?)
        }
        None => None,
    };
    Ok(BoundQuery { cross, square })
}

impl<T: KernelScalar> BoundQuery<'_, T> {
    /// Runs `f` on the query; the `f64` view of the blocks only when
    /// `refines`.
    fn run<R>(
        &self,
        points: QueryPoints<'_>,
        m: usize,
        refines: bool,
        f: impl FnOnce(Query<'_, T>) -> Result<R, GprError>,
    ) -> Result<R, GprError> {
        let cross64 = refines.then(|| self.cross.f64_view());
        f(Query {
            xs: points.xs,
            m,
            n_cols: points.n_cols,
            cross: Some(&self.cross),
            cross64: cross64.as_ref().map(|t| t as &dyn RectSlots<f64>),
            square: self.square.as_ref().map(|s| s as &dyn SquareSlots<T>),
        })
    }
}

impl<O, P> Gpr<O, P, DistanceKernel<DistanceOnly>>
where
    P: GpScalar,
    O: for<'a> Optimizer<GprObjective<'a, P, DistanceKernel<DistanceOnly>>>,
{
    /// Factors `A = K + σn² I` on the supplied training distances and
    /// updates `θ` with `O`.
    ///
    /// `sources` holds one source per slot of the kernel, in any order:
    /// the `n × n` squared distances between the training samples. The
    /// model keeps them (a moved or copied table, or what a fill wrote).
    /// `y` has `n` targets.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::EmptyInput`] if `n` is zero,
    /// [`GprError::LengthMismatch`] if a table or `y` has the wrong length,
    /// a slot has no source or two, or a source names a slot the kernel does
    /// not read, [`GprError::InvalidDistance`] for a value that is not
    /// finite or is negative, or a training square whose diagonal is not
    /// zero or that is not symmetric (see
    /// [`crate::kernel::DistanceSource::tidy`]), and the errors of the
    /// coordinate [`Gpr::fit`].
    ///
    /// See the example on [`crate::kernel::ScalarDistance`].
    #[allow(clippy::result_large_err)] // failure returns the trainer so the caller can retry
    pub fn fit<'s>(
        self,
        sources: impl IntoIterator<Item = DistanceSource<'s>>,
        n: usize,
        y: &[f64],
    ) -> Result<FittedGpr<O, P, DistanceKernel>, (Self, GprError)> {
        self.fit_input(train_input(sources, n, (&[], 0), y))
    }
}

impl<P: GpScalar> Gpr<Fixed, P, DistanceKernel<DistanceOnly>> {
    /// Factors at the current `θ` without a search. Same data contract as
    /// [`Gpr::fit`] of this kernel.
    ///
    /// # Errors
    ///
    /// Same as [`Gpr::fit`] of this kernel.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{RbfKernel, ScalarDistance};
    /// use gprx::{Fixed, GaussianLikelihood, Gpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let image = ScalarDistance::new();
    /// let train = [0.0, 1.0, 1.0, 0.0];
    /// let fitted = Gpr::new(image.kernel(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?)
    ///     .with_optimizer(Fixed)
    ///     .factor([image.from_slice(&train)], 2, &[0.0, 1.0])
    ///     .map_err(|(_, e)| e)?;
    /// assert_eq!(fitted.n(), 2);
    /// # Ok(())
    /// # }
    /// ```
    #[allow(clippy::result_large_err)] // failure returns the trainer so the caller can retry
    pub fn factor<'s>(
        self,
        sources: impl IntoIterator<Item = DistanceSource<'s>>,
        n: usize,
        y: &[f64],
    ) -> Result<FittedGpr<Fixed, P, DistanceKernel>, (Self, GprError)> {
        self.factor_input(train_input(sources, n, (&[], 0), y))
    }
}

impl<O, P> Gpr<O, P, DistanceKernel<WithPoints>>
where
    P: GpScalar,
    O: for<'a> Optimizer<GprObjective<'a, P, DistanceKernel<WithPoints>>>,
{
    /// Factors `A = K + σn² I` on the supplied training distances and the
    /// coordinates `x`, and updates `θ` with `O`.
    ///
    /// `x` is column-major, `n` points by `n_cols` features, and goes
    /// through the input transform; the distances do not.
    ///
    /// # Errors
    ///
    /// Same as [`Gpr::fit`] of a [`DistanceKernel<DistanceOnly>`], plus the
    /// coordinate errors of the coordinate [`Gpr::fit`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel, ScalarDistance};
    /// use gprx::{GaussianLikelihood, Gpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let image = ScalarDistance::new();
    /// let kernel = image.kernel(RbfKernel::new(1.0)?) * KernelSpec::from(RbfKernel::new(0.5)?);
    /// let train = [0.0, 1.0, 1.0, 0.0];
    /// let fitted = Gpr::new(kernel, GaussianLikelihood::new(0.1)?)
    ///     .fit([image.from_slice(&train)], 2, &[0.0, 1.0], 1, &[0.0, 1.0])
    ///     .map_err(|(_, e)| e)?;
    /// let pred = fitted.predict([image.borrow(&[0.25, 0.25])], &[0.5], 1, 1)?;
    /// assert_eq!(pred.mean.len(), 1);
    /// # Ok(())
    /// # }
    /// ```
    #[allow(clippy::result_large_err)] // failure returns the trainer so the caller can retry
    pub fn fit<'s>(
        self,
        sources: impl IntoIterator<Item = DistanceSource<'s>>,
        n: usize,
        x: &[f64],
        n_cols: usize,
        y: &[f64],
    ) -> Result<FittedGpr<O, P, DistanceKernel<WithPoints>>, (Self, GprError)> {
        self.fit_input(train_input(sources, n, (x, n_cols), y))
    }
}

impl<P: GpScalar> Gpr<Fixed, P, DistanceKernel<WithPoints>> {
    /// Factors at the current `θ` without a search. Same data contract as
    /// [`Gpr::fit`] of this kernel.
    ///
    /// # Errors
    ///
    /// Same as [`Gpr::fit`] of this kernel.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel, ScalarDistance};
    /// use gprx::{Fixed, GaussianLikelihood, Gpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let image = ScalarDistance::new();
    /// let kernel = image.kernel(RbfKernel::new(1.0)?) + KernelSpec::from(RbfKernel::new(0.5)?);
    /// let train = [0.0, 1.0, 1.0, 0.0];
    /// let fitted = Gpr::new(kernel, GaussianLikelihood::new(0.1)?)
    ///     .with_optimizer(Fixed)
    ///     .factor([image.from_slice(&train)], 2, &[0.0, 1.0], 1, &[0.0, 1.0])
    ///     .map_err(|(_, e)| e)?;
    /// assert_eq!(fitted.d(), 1);
    /// # Ok(())
    /// # }
    /// ```
    #[allow(clippy::result_large_err)] // failure returns the trainer so the caller can retry
    pub fn factor<'s>(
        self,
        sources: impl IntoIterator<Item = DistanceSource<'s>>,
        n: usize,
        x: &[f64],
        n_cols: usize,
        y: &[f64],
    ) -> Result<FittedGpr<Fixed, P, DistanceKernel<WithPoints>>, (Self, GprError)> {
        self.factor_input(train_input(sources, n, (x, n_cols), y))
    }
}

/// A source of a longer borrow, read for a shorter call.
fn shorten<'s: 'a, 'a>(source: DistanceSource<'s>) -> DistanceSource<'a> {
    source
}

/// The training input of a distance model: `x` (`n × n_cols`) is empty for
/// a [`DistanceOnly`] kernel.
fn train_input<'s: 'a, 'a>(
    sources: impl IntoIterator<Item = DistanceSource<'s>>,
    n: usize,
    (x, n_cols): (&'a [f64], usize),
    y: &'a [f64],
) -> TrainInput<'a> {
    TrainInput {
        x,
        n_rows: n,
        n_cols,
        y,
        sources: sources.into_iter().map(shorten).collect(),
    }
}

/// The query of an Exact model on supplied distances.
impl<O, P: GpScalar, C: PointUse> DistanceQuery for FittedGpr<O, P, DistanceKernel<C>> {
    type Refine = P::Refine;

    fn query_distances<'s>(
        &self,
        cross: impl IntoIterator<Item = DistanceSource<'s>>,
        points: QueryPoints<'_>,
        m: usize,
        options: PredictOptions,
    ) -> Result<Prediction<P::Refine>, GprError> {
        let slots = &self.core.slots;
        let alpha = &self.core.alpha[..];
        let mut out = Prediction::default();
        let mut scratch = QueryScratch::new();
        bind_query(slots, (self.core.n, m), cross, None, &mut scratch)?.run(
            points,
            m,
            P::REFINES_IN_F64,
            |q| {
                self.core
                    .write_prediction(self.factor(), alpha, q, options, &mut out)
            },
        )?;
        Ok(out)
    }

    fn query_distances_into<'s>(
        &mut self,
        cross: impl IntoIterator<Item = DistanceSource<'s>>,
        points: QueryPoints<'_>,
        m: usize,
        options: PredictOptions,
        out: &mut Prediction<P::Refine>,
    ) -> Result<(), GprError> {
        // The model's buffers, taken for the call: the blocks borrow
        // them while the predict borrows the model. They hold no
        // state, so a panic that loses them loses only capacity.
        let mut scratch = std::mem::take(&mut self.core.query_sources);
        let result = bind_query(
            &self.core.slots,
            (self.core.n, m),
            cross,
            None,
            &mut scratch,
        )
        .and_then(|bound| {
            bound.run(points, m, P::REFINES_IN_F64, |q| {
                self.predict_query_into(q, options, out)
            })
        });
        self.core.query_sources = scratch;
        result
    }

    fn query_distance_covariance<'s>(
        &self,
        cross: impl IntoIterator<Item = DistanceSource<'s>>,
        square: impl IntoIterator<Item = DistanceSource<'s>>,
        points: QueryPoints<'_>,
        m: usize,
        options: PredictOptions,
    ) -> Result<PredictiveCovariance<P::Refine>, GprError> {
        let slots = &self.core.slots;
        let alpha = &self.core.alpha[..];
        let square = Some(square.into_iter().collect());
        let mut scratch = QueryScratch::new();
        bind_query(slots, (self.core.n, m), cross, square, &mut scratch)?.run(
            points,
            m,
            P::REFINES_IN_F64,
            |q| self.core.write_covariance(self.factor(), alpha, q, options),
        )
    }

    fn draw_jitter(&self) -> JitterPolicy {
        self.core.policies.jitter
    }
}

distance_predict!(
    impl [O, P: GpScalar] FittedGpr<O, P, DistanceKernel<DistanceOnly>>,
    refine = P::Refine,
    args = (),
    tail = (),
    points = QueryPoints::NONE,
    reads = {
        /// A table is read in place for this call (an `f32` model reads it
        /// through a cast); a fill writes scratch once.
    },
    predict_doc = {
        /// See the example on [`crate::kernel::ScalarDistance`].
    },
    covariance_doc = {
        /// # Examples
        ///
        /// ```rust
        /// use gprx::kernel::{RbfKernel, ScalarDistance};
        /// use gprx::{GaussianLikelihood, Gpr, PredictOptions, Prediction, VarianceKind};
        ///
        /// # fn main() -> Result<(), gprx::GprError> {
        /// let image = ScalarDistance::new();
        /// let mut fitted = Gpr::new(image.kernel(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?)
        ///     .fit([image.from_vec(vec![0.0, 1.0, 1.0, 0.0])], 2, &[0.0, 1.0])
        ///     .map_err(|(_, e)| e)?;
        /// // Two queries: train × query, then query × query.
        /// let cross = [0.25, 0.25, 0.25, 2.25];
        /// let query = [0.0, 1.0, 1.0, 0.0];
        /// let latent = PredictOptions { variance_kind: VarianceKind::Latent };
        /// let mut out = Prediction::default();
        /// fitted.predict_into([image.borrow(&cross)], 2, &mut out)?;
        /// fitted.predict_with_into([image.borrow(&cross)], 2, latent, &mut out)?;
        /// let _ = fitted.predict_with([image.borrow(&cross)], 2, latent)?;
        /// let cov = fitted.predict_covariance([image.borrow(&cross)], [image.borrow(&query)], 2)?;
        /// assert_eq!(cov.covariance.len(), 4);
        /// let _ = fitted.predict_covariance_with([image.borrow(&cross)], [image.borrow(&query)], 2, latent)?;
        /// let draws = fitted.sample([image.borrow(&cross)], [image.borrow(&query)], 2, 3, 7)?;
        /// assert_eq!(draws.len(), 6);
        /// let _ = fitted.sample_with([image.borrow(&cross)], [image.borrow(&query)], 2, latent, 3, 7)?;
        /// # Ok(())
        /// # }
        /// ```
    },
);

distance_predict!(
    impl [O, P: GpScalar] FittedGpr<O, P, DistanceKernel<WithPoints>>,
    refine = P::Refine,
    args = (xs: &[f64]),
    tail = (n_cols: usize),
    points = QueryPoints { xs, n_cols },
    reads = {
        /// A table is read in place for this call (an `f32` model reads it
        /// through a cast); a fill writes scratch once.
    },
    predict_doc = {
        /// See the example on [`Gpr::fit`] of a [`DistanceKernel<WithPoints>`].
    },
    covariance_doc = {
        /// # Examples
        ///
        /// ```rust
        /// use gprx::kernel::{KernelSpec, RbfKernel, ScalarDistance};
        /// use gprx::{GaussianLikelihood, Gpr, PredictOptions, Prediction};
        ///
        /// # fn main() -> Result<(), gprx::GprError> {
        /// let image = ScalarDistance::new();
        /// let kernel = image.kernel(RbfKernel::new(1.0)?) * KernelSpec::from(RbfKernel::new(0.5)?);
        /// let mut fitted = Gpr::new(kernel, GaussianLikelihood::new(0.1)?)
        ///     .fit([image.from_vec(vec![0.0, 1.0, 1.0, 0.0])], 2, &[0.0, 1.0], 1, &[0.0, 1.0])
        ///     .map_err(|(_, e)| e)?;
        /// let (cross, query, xs) = ([0.25, 0.25, 0.25, 2.25], [0.0, 1.0, 1.0, 0.0], [0.5, 1.5]);
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
