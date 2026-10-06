//! Exact GPR on supplied squared distances: the `fit`, `factor`, predict,
//! and `insert` of [`Gpr`], [`FittedGpr`], and [`OnlineGpr`] for a
//! [`DistanceKernel`].
//!
//! A [`DistanceKernel<DistanceOnly>`] model takes no coordinates; a
//! [`DistanceKernel<WithPoints>`] model takes the column-major `x` of its
//! coordinate leaves next to the supplied distances. Every other method is
//! the one of the coordinate model.

use crate::error::GprError;
use crate::gpr::GprObjective;
use crate::kernel::{
    BlockKind, DistanceKernel, DistanceOnly, DistanceSlot, DistanceSource, KernelScalar, PointUse,
    QuerySources, RectSlots, SquareSlots, TrainSources, WithPoints,
};
use crate::optimizer::{Fixed, Optimizer};
use crate::points::PointId;
use crate::policy::JitterPolicy;
use crate::precision::GpScalar;
use crate::prediction::{DistanceQuery, QueryPoints, distance_predict};
use crate::{PredictOptions, Prediction, PredictiveCovariance};

use super::shared::Query;
use super::{FittedGpr, Gpr, OnlineGpr, TrainInput};

/// The checked blocks of one query: train × query (`n × m`) and, for a
/// covariance, the query × query squares (`m × m`) as a store a Gram reads.
struct BoundQuery<'s, T: KernelScalar> {
    cross: QuerySources<'s, T>,
    square: Option<TrainSources<T>>,
}

impl<'s, T: KernelScalar> BoundQuery<'s, T> {
    fn bind(
        slots: &[DistanceSlot],
        n: usize,
        m: usize,
        cross: Vec<DistanceSource<'s>>,
        square: Option<Vec<DistanceSource<'s>>>,
    ) -> Result<Self, GprError> {
        crate::data::require_nonempty(m)?;
        let cross = QuerySources::<T>::bind(slots, cross, n, m, BlockKind::Rect)?;
        // The query squares are one set: a Gram reads them, as it reads the
        // training squares.
        let square = square
            .map(|square| {
                QuerySources::<T>::bind(slots, square, m, m, BlockKind::Square)
                    .and_then(|square| square.into_square(m))
            })
            .transpose()?;
        Ok(Self { cross, square })
    }

    /// Runs `f` on the query; the `f64` tables only for a refining model.
    fn run<R>(
        &mut self,
        points: QueryPoints<'_>,
        m: usize,
        refines: bool,
        f: impl FnOnce(Query<'_, T>) -> Result<R, GprError>,
    ) -> Result<R, GprError> {
        let (table, table64) = self.cross.tables(refines);
        f(Query {
            xs: points.xs,
            m,
            n_cols: points.n_cols,
            cross: Some(&table),
            cross64: table64.as_ref().map(|t| t as &dyn RectSlots<f64>),
            square: self.square.as_ref().map(|s| s as &dyn SquareSlots<T>),
        })
    }
}

/// [`BoundQuery::bind`], then [`BoundQuery::run`].
fn with_query<'s, P: GpScalar, R>(
    slots: &[DistanceSlot],
    n: usize,
    points: QueryPoints<'_>,
    m: usize,
    cross: Vec<DistanceSource<'s>>,
    square: Option<Vec<DistanceSource<'s>>>,
    f: impl FnOnce(Query<'_, P::Storage>) -> Result<R, GprError>,
) -> Result<R, GprError> {
    BoundQuery::<P::Storage>::bind(slots, n, m, cross, square)?.run(points, m, P::REFINES_IN_F64, f)
}

impl<O, P> Gpr<O, P, DistanceKernel<DistanceOnly>>
where
    P: GpScalar,
    O: for<'a> Optimizer<GprObjective<'a, P>>,
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
    /// not read, [`GprError::NonFiniteInput`] for a non-finite value,
    /// [`GprError::ShapeMismatch`] for a negative value, or a training
    /// square whose diagonal or symmetry is off past rounding (see
    /// [`crate::kernel::ScalarDistance`]), and the errors
    /// of the coordinate
    /// [`Gpr::fit`].
    ///
    /// See the example on [`crate::kernel::ScalarDistance`].
    #[allow(clippy::result_large_err)] // failure returns the trainer so the caller can retry
    pub fn fit<'s>(
        self,
        sources: impl IntoIterator<Item = DistanceSource<'s>>,
        n: usize,
        y: &[f64],
    ) -> Result<FittedGpr<O, P, DistanceKernel>, (Self, GprError)> {
        self.fit_input(distances_only(sources, n, y))
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
        self.factor_input(distances_only(sources, n, y))
    }
}

impl<O, P> Gpr<O, P, DistanceKernel<WithPoints>>
where
    P: GpScalar,
    O: for<'a> Optimizer<GprObjective<'a, P>>,
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
        self.fit_input(with_points(sources, n, x, n_cols, y))
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
        self.factor_input(with_points(sources, n, x, n_cols, y))
    }
}

/// A source of a longer borrow, read for a shorter call.
fn shorten<'s: 'a, 'a>(source: DistanceSource<'s>) -> DistanceSource<'a> {
    source
}

fn distances_only<'s: 'a, 'a>(
    sources: impl IntoIterator<Item = DistanceSource<'s>>,
    n: usize,
    y: &'a [f64],
) -> TrainInput<'a> {
    TrainInput {
        x: &[],
        n_rows: n,
        n_cols: 0,
        y,
        sources: sources.into_iter().map(shorten).collect(),
    }
}

fn with_points<'s: 'a, 'a>(
    sources: impl IntoIterator<Item = DistanceSource<'s>>,
    n: usize,
    x: &'a [f64],
    n_cols: usize,
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

/// The [`DistanceQuery`] of an Exact model; `$alpha` reads its predict `α`.
macro_rules! exact_query {
    ($model:ident, alpha = |$this:ident| $alpha:expr) => {
        impl<O, P: GpScalar, C: PointUse> DistanceQuery for $model<O, P, DistanceKernel<C>> {
            type Refine = P::Refine;

            fn query_distances(
                &self,
                cross: Vec<DistanceSource<'_>>,
                points: QueryPoints<'_>,
                m: usize,
                options: PredictOptions,
            ) -> Result<Prediction<P::Refine>, GprError> {
                let slots = &self.core.slots;
                let $this = self;
                let alpha = $alpha?;
                let mut out = Prediction::default();
                with_query::<P, ()>(slots, self.core.n, points, m, cross, None, |q| {
                    self.core
                        .write_prediction(self.factor(), alpha, q, options, &mut out)
                })?;
                Ok(out)
            }

            fn query_distances_into(
                &mut self,
                cross: Vec<DistanceSource<'_>>,
                points: QueryPoints<'_>,
                m: usize,
                options: PredictOptions,
                out: &mut Prediction<P::Refine>,
            ) -> Result<(), GprError> {
                let mut bound =
                    BoundQuery::<P::Storage>::bind(&self.core.slots, self.core.n, m, cross, None)?;
                bound.run(points, m, P::REFINES_IN_F64, |q| {
                    self.predict_query_into(q, options, out)
                })
            }

            fn query_distance_covariance(
                &self,
                cross: Vec<DistanceSource<'_>>,
                square: Vec<DistanceSource<'_>>,
                points: QueryPoints<'_>,
                m: usize,
                options: PredictOptions,
            ) -> Result<PredictiveCovariance<P::Refine>, GprError> {
                let slots = &self.core.slots;
                let $this = self;
                let alpha = $alpha?;
                with_query::<P, _>(slots, self.core.n, points, m, cross, Some(square), |q| {
                    self.core.write_covariance(self.factor(), alpha, q, options)
                })
            }

            fn draw_jitter(&self) -> JitterPolicy {
                self.core.policies.jitter
            }
        }
    };
}

exact_query!(
    FittedGpr,
    alpha = |model| Ok::<_, GprError>(&model.core.alpha[..])
);
exact_query!(OnlineGpr, alpha = |model| model.predict_alpha());

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

distance_predict!(
    impl [O, P: GpScalar] OnlineGpr<O, P, DistanceKernel<DistanceOnly>>,
    refine = P::Refine,
    args = (),
    tail = (),
    points = QueryPoints::NONE,
    reads = {
        /// A table is read in place for this call (an `f32` model reads it
        /// through a cast); a fill writes scratch once.
    },
    predict_doc = {
        /// See the example on [`OnlineGpr::insert`] of a [`DistanceKernel<DistanceOnly>`].
    },
    covariance_doc = {
        /// See the example on [`OnlineGpr::insert`] of a [`DistanceKernel<DistanceOnly>`].
    },
);

distance_predict!(
    impl [O, P: GpScalar] OnlineGpr<O, P, DistanceKernel<WithPoints>>,
    refine = P::Refine,
    args = (xs: &[f64]),
    tail = (n_cols: usize),
    points = QueryPoints { xs, n_cols },
    reads = {
        /// A table is read in place for this call (an `f32` model reads it
        /// through a cast); a fill writes scratch once.
    },
    predict_doc = {
        /// See the example on [`OnlineGpr::insert`] of a [`DistanceKernel<WithPoints>`].
    },
    covariance_doc = {
        /// See the example on [`OnlineGpr::insert`] of a [`DistanceKernel<WithPoints>`].
    },
);

impl<O, P: GpScalar> OnlineGpr<O, P, DistanceKernel<DistanceOnly>> {
    /// Appends one training point with a bordered LDLT update.
    ///
    /// `sources` holds one source per slot: the squared distances from the
    /// `n` live training points to the new one (`n × 1`). The model adds
    /// them to its training distances; the new diagonal is zero. `α` is
    /// left stale, as in the coordinate model's `insert`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::LengthMismatch`] if a column is not `n` long or a
    /// slot has no source or two, [`GprError::NonFiniteInput`] for a
    /// non-finite value, and the errors of the coordinate model's `insert`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{RbfKernel, ScalarDistance};
    /// use gprx::{GaussianLikelihood, Gpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let image = ScalarDistance::new();
    /// let fitted = Gpr::new(image.kernel(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?)
    ///     .fit([image.from_vec(vec![0.0, 1.0, 1.0, 0.0])], 2, &[0.0, 1.0])
    ///     .map_err(|(_, e)| e)?;
    /// let mut online = fitted.into_online()?;
    /// // The new point is 4 and 1 (squared) away from the two points.
    /// online.insert([image.from_slice(&[4.0, 1.0])], 0.5)?;
    /// assert_eq!(online.n(), 3);
    /// let cross = [0.25, 0.25, 2.25];
    /// let query = [0.0];
    /// let pred = online.predict([image.borrow(&cross)], 1)?;
    /// let mut out = gprx::Prediction::default();
    /// online.predict_into([image.borrow(&cross)], 1, &mut out)?;
    /// let options = gprx::PredictOptions::default();
    /// online.predict_with_into([image.borrow(&cross)], 1, options, &mut out)?;
    /// let _ = online.predict_with([image.borrow(&cross)], 1, options)?;
    /// let _ = online.predict_covariance([image.borrow(&cross)], [image.borrow(&query)], 1)?;
    /// let _ = online.predict_covariance_with(
    ///     [image.borrow(&cross)],
    ///     [image.borrow(&query)],
    ///     1,
    ///     options,
    /// )?;
    /// let _ = online.sample([image.borrow(&cross)], [image.borrow(&query)], 1, 2, 0)?;
    /// let _ = online.sample_with([image.borrow(&cross)], [image.borrow(&query)], 1, options, 2, 0)?;
    /// assert_eq!(pred.mean.len(), 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn insert<'s>(
        &mut self,
        sources: impl IntoIterator<Item = DistanceSource<'s>>,
        y_new: f64,
    ) -> Result<PointId, GprError> {
        self.insert_point(&[], sources.into_iter().collect(), y_new)
    }
}

impl<O, P: GpScalar> OnlineGpr<O, P, DistanceKernel<WithPoints>> {
    /// Appends one training point: its squared distances to the `n` live
    /// points (one source per slot, `n × 1`), its coordinates `x_new`
    /// (length [`OnlineGpr::d`]), and its target.
    ///
    /// # Errors
    ///
    /// Same as [`OnlineGpr::insert`] of a [`DistanceKernel<DistanceOnly>`],
    /// plus [`GprError::DimensionMismatch`] if `x_new` is the wrong length.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel, ScalarDistance};
    /// use gprx::{GaussianLikelihood, Gpr, PredictOptions, Prediction};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let image = ScalarDistance::new();
    /// let kernel = image.kernel(RbfKernel::new(1.0)?) * KernelSpec::from(RbfKernel::new(0.5)?);
    /// let fitted = Gpr::new(kernel, GaussianLikelihood::new(0.1)?)
    ///     .fit([image.from_vec(vec![0.0, 1.0, 1.0, 0.0])], 2, &[0.0, 1.0], 1, &[0.0, 1.0])
    ///     .map_err(|(_, e)| e)?;
    /// let mut online = fitted.into_online()?;
    /// online.insert([image.from_slice(&[4.0, 1.0])], &[2.0], 0.5)?;
    /// let (cross, query, xs) = ([0.25, 0.25, 2.25], [0.0], [0.5]);
    /// let options = PredictOptions::default();
    /// let mut out = Prediction::default();
    /// let _ = online.predict([image.borrow(&cross)], &xs, 1, 1)?;
    /// online.predict_into([image.borrow(&cross)], &xs, 1, 1, &mut out)?;
    /// online.predict_with_into([image.borrow(&cross)], &xs, 1, 1, options, &mut out)?;
    /// let _ = online.predict_with([image.borrow(&cross)], &xs, 1, 1, options)?;
    /// let _ = online.predict_covariance([image.borrow(&cross)], [image.borrow(&query)], &xs, 1, 1)?;
    /// let _ = online.predict_covariance_with(
    ///     [image.borrow(&cross)],
    ///     [image.borrow(&query)],
    ///     &xs,
    ///     1,
    ///     1,
    ///     options,
    /// )?;
    /// let _ = online.sample([image.borrow(&cross)], [image.borrow(&query)], &xs, 1, 1, 2, 0)?;
    /// let _ = online.sample_with(
    ///     [image.borrow(&cross)],
    ///     [image.borrow(&query)],
    ///     &xs,
    ///     1,
    ///     1,
    ///     options,
    ///     2,
    ///     0,
    /// )?;
    /// assert_eq!(online.n(), 3);
    /// # Ok(())
    /// # }
    /// ```
    pub fn insert<'s>(
        &mut self,
        sources: impl IntoIterator<Item = DistanceSource<'s>>,
        x_new: &[f64],
        y_new: f64,
    ) -> Result<PointId, GprError> {
        self.insert_point(x_new, sources.into_iter().collect(), y_new)
    }
}
