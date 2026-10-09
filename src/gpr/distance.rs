//! Exact GPR on supplied squared distances: the `fit`, `factor`, and
//! predict of [`Gpr`] and [`FittedGpr`] for a [`DistanceKernel`], and the
//! insert and predict of its [`OnlineGpr`].
//!
//! A [`DistanceKernel<DistanceOnly>`] model takes no coordinates; a
//! [`DistanceKernel<WithPoints>`] model takes the column-major `x` of its
//! coordinate leaves next to the supplied distances. Every other method is
//! the one of the coordinate model.

use crate::error::GprError;
use crate::gpr::GprObjective;
use crate::kernel::{
    DistanceKernel, DistanceOnly, DistanceSlot, DistanceSource, KernelScalar, PointUse,
    QueryScratch, QuerySources, SourceStore, SuppliedSpec, WithPoints,
};
use crate::optimizer::{Fixed, Optimizer};
use crate::policy::JitterPolicy;
use crate::precision::GpScalar;
use crate::prediction::{DistanceQuery, QueryPoints, distance_predict};
use crate::{PredictOptions, Prediction, PredictiveCovariance};

use super::shared::Query;
use crate::points::PointId;

use super::{FittedGpr, Gpr, OnlineGpr, TrainInput};

/// Binds the `n × m` train × query blocks of `cross` on `scratch`. The
/// result borrows the scratch and the caller's tables, not `slots`, so the
/// model can be borrowed again to run it.
fn bind_cross<'a, 's: 'a, T: KernelScalar>(
    slots: &[DistanceSlot],
    (n, m): (usize, usize),
    cross: impl IntoIterator<Item = DistanceSource<'s>>,
    scratch: &'a mut QueryScratch<T>,
) -> Result<QuerySources<'a, T>, GprError> {
    crate::data::require_nonempty(m)?;
    QuerySources::<T>::bind_rect(slots, cross, (n, m), scratch)
}

/// Runs `f` on the query of the bound blocks `cross`, with their `f64`
/// view for a model that refines in `f64`.
fn run<T: KernelScalar, R>(
    cross: &QuerySources<'_, T>,
    points: QueryPoints<'_>,
    m: usize,
    f: impl FnOnce(Query<'_, T, SuppliedSpec>) -> Result<R, GprError>,
) -> Result<R, GprError> {
    let cross64 = cross.f64_view();
    f(Query {
        xs: points.xs,
        m,
        n_cols: points.n_cols,
        cross,
        cross64: &cross64,
    })
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
        let cross = bind_cross(slots, (self.core.n, m), cross, &mut scratch)?;
        run(&cross, points, m, |q| {
            self.core
                .write_prediction(self.factor(), alpha, q, options, &mut out)
        })?;
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
        let result =
            bind_cross(&self.core.slots, (self.core.n, m), cross, &mut scratch).and_then(|cross| {
                run(&cross, points, m, |q| {
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
        let mut scratch = QueryScratch::new();
        let mut square_scratch = QueryScratch::new();
        let cross = bind_cross(slots, (self.core.n, m), cross, &mut scratch)?;
        // The query squares are one set: a Gram reads them, as it reads the
        // training squares, checked in full when bound.
        let square = QuerySources::bind_square(slots, square, m, &mut square_scratch)?;
        run(&cross, points, m, |q| {
            self.core
                .write_covariance(self.factor(), alpha, q, &square, options)
        })
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
    count = m,
    cross = {
        /// the `n × m` squared distances from the training samples to the
        /// `m` queries.
    },
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
    count = m,
    cross = {
        /// the `n × m` squared distances from the training samples to the
        /// `m` queries.
    },
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

impl<O, P: GpScalar, C: PointUse> OnlineGpr<O, P, DistanceKernel<C>> {
    /// Appends one point from its supplied columns: each binds its slot's
    /// `n × 1` squared distances from the `n` live points, in
    /// [`Self::point_ids`] order, to the new point (`d` such columns for an
    /// ARD slot). The columns are checked in full, as training squares
    /// are, then kept.
    fn insert_sources<'s>(
        &mut self,
        sources: impl IntoIterator<Item = DistanceSource<'s>>,
        x_new: &[f64],
        y_new: f64,
    ) -> Result<PointId, GprError> {
        // The model's buffers, taken for the call as a predict takes them.
        let mut scratch = std::mem::take(&mut self.core.query_sources);
        let result = self.insert_bound(sources, x_new, y_new, &mut scratch);
        self.core.query_sources = scratch;
        result
    }

    fn insert_bound<'s>(
        &mut self,
        sources: impl IntoIterator<Item = DistanceSource<'s>>,
        x_new: &[f64],
        y_new: f64,
        scratch: &mut QueryScratch<P::Storage>,
    ) -> Result<PointId, GprError> {
        // Every check of the point and its columns runs before the store
        // makes room; a pivot the factor refuses after that leaves only the
        // room, which reads nothing.
        self.check_new_point(x_new, y_new)?;
        let cols = QuerySources::bind_column(&self.core.slots, sources, self.core.n, scratch)?;
        let exact = cols.f64_view();
        self.core.sources.reserve_point()?;
        self.insert_with(
            x_new,
            y_new,
            &cols,
            |store| store.check_push(&cols, &exact),
            |store| store.write_point(&cols, &exact),
        )
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
}

impl<O, P: GpScalar> OnlineGpr<O, P, DistanceKernel<DistanceOnly>> {
    /// Appends one training point at the current `θ` with a bordered LDLT
    /// update, from its squared distances to the live points.
    ///
    /// `sources` binds, per slot, the `n × 1` column from the `n` live
    /// points (in [`Self::point_ids`] order) to the new point; an ARD slot
    /// binds one such column per dimension. A table may be borrowed, owned,
    /// or filled. Every value is checked (finite, `≥ 0`) and the column is
    /// kept: the store grows its capacity by doubling, so most inserts copy
    /// only the column. `α` is not solved here; the first later read solves
    /// it. The returned [`PointId`] is never reused after a later
    /// [`Self::delete`].
    ///
    /// # Errors
    ///
    /// Returns [`GprError::NonFiniteInput`] if `y_new` is `NaN` or `Inf`,
    /// [`GprError::InvalidDistance`] for a negative or non-finite distance,
    /// [`GprError::LengthMismatch`] for a column whose length is not `n`, a
    /// source of a slot the kernel does not read, a slot without a source,
    /// or two sources of one slot, [`GprError::IndexOutOfRange`] if no new
    /// [`PointId`] is left, [`GprError::SizeOverflow`] if the kept squares
    /// cannot grow by a point (their memory cannot be reserved), or
    /// [`GprError::CholeskyFailed`] if the new pivot is not positive. On an
    /// error the model holds the same points.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{RbfKernel, ScalarDistance};
    /// use gprx::{Fixed, GaussianLikelihood, Gpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let image = ScalarDistance::new();
    /// let fitted = Gpr::new(image.kernel(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?)
    ///     .with_optimizer(Fixed)
    ///     .factor([image.from_vec(vec![0.0, 1.0, 1.0, 0.0])], 2, &[0.0, 1.0])
    ///     .map_err(|(_, e)| e)?;
    /// let mut online = fitted.into_online()?;
    /// // The new point's squared distances to the two live points.
    /// let id = online.insert([image.from_vec(vec![4.0, 1.0])], 0.5)?;
    /// let pred = online.predict([image.from_vec(vec![1.0, 0.0, 1.0])], 1)?;
    /// assert_eq!(pred.mean.len(), 1);
    /// online.delete(id)?;
    /// assert_eq!(online.n(), 2);
    /// # Ok(())
    /// # }
    /// ```
    pub fn insert<'s>(
        &mut self,
        sources: impl IntoIterator<Item = DistanceSource<'s>>,
        y_new: f64,
    ) -> Result<PointId, GprError> {
        self.insert_sources(sources, &[], y_new)
    }
}

impl<O, P: GpScalar> OnlineGpr<O, P, DistanceKernel<WithPoints>> {
    /// Appends one training point at the current `θ` with a bordered LDLT
    /// update, from its squared distances to the live points and its
    /// coordinates `x_new` (length [`Self::d`]).
    ///
    /// The columns are as in the insert of a [`DistanceOnly`] model; `x_new`
    /// goes through the stored input transform, which is not re-fit.
    ///
    /// # Errors
    ///
    /// Those of the insert of a [`DistanceOnly`] model, and
    /// [`GprError::DimensionMismatch`] if `x_new` is the wrong length or
    /// [`GprError::NonFiniteInput`] if one of its values is `NaN` or `Inf`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel, ScalarDistance};
    /// use gprx::{Fixed, GaussianLikelihood, Gpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let image = ScalarDistance::new();
    /// let kernel = image.kernel(RbfKernel::new(1.0)?) * KernelSpec::from(RbfKernel::new(0.5)?);
    /// let fitted = Gpr::new(kernel, GaussianLikelihood::new(0.1)?)
    ///     .with_optimizer(Fixed)
    ///     .factor([image.from_vec(vec![0.0, 1.0, 1.0, 0.0])], 2, &[0.0, 1.0], 1, &[0.0, 1.0])
    ///     .map_err(|(_, e)| e)?;
    /// let mut online = fitted.into_online()?;
    /// online.insert([image.from_vec(vec![4.0, 1.0])], &[2.0], 0.5)?;
    /// assert_eq!(online.n(), 3);
    /// let pred = online.predict([image.from_vec(vec![1.0, 0.0, 1.0])], &[1.0], 1, 1)?;
    /// assert_eq!(pred.mean.len(), 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn insert<'s>(
        &mut self,
        sources: impl IntoIterator<Item = DistanceSource<'s>>,
        x_new: &[f64],
        y_new: f64,
    ) -> Result<PointId, GprError> {
        self.insert_sources(sources, x_new, y_new)
    }
}

/// The query of an online model on supplied distances.
impl<O, P: GpScalar, C: PointUse> DistanceQuery for OnlineGpr<O, P, DistanceKernel<C>> {
    type Refine = P::Refine;

    fn query_distances<'s>(
        &self,
        cross: impl IntoIterator<Item = DistanceSource<'s>>,
        points: QueryPoints<'_>,
        m: usize,
        options: PredictOptions,
    ) -> Result<Prediction<P::Refine>, GprError> {
        let alpha = self.alpha()?;
        let mut out = Prediction::default();
        let mut scratch = QueryScratch::new();
        let cross = bind_cross(&self.core.slots, (self.core.n, m), cross, &mut scratch)?;
        run(&cross, points, m, |q| {
            self.core
                .write_prediction(self.factor(), alpha, q, options, &mut out)
        })?;
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
        let mut scratch = std::mem::take(&mut self.core.query_sources);
        let result =
            bind_cross(&self.core.slots, (self.core.n, m), cross, &mut scratch).and_then(|cross| {
                run(&cross, points, m, |q| {
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
        let alpha = self.alpha()?;
        let slots = &self.core.slots;
        let mut scratch = QueryScratch::new();
        let mut square_scratch = QueryScratch::new();
        let cross = bind_cross(slots, (self.core.n, m), cross, &mut scratch)?;
        let square = QuerySources::bind_square(slots, square, m, &mut square_scratch)?;
        run(&cross, points, m, |q| {
            self.core
                .write_covariance(self.factor(), alpha, q, &square, options)
        })
    }

    fn draw_jitter(&self) -> JitterPolicy {
        self.core.policies.jitter
    }
}

distance_predict!(
    impl [O, P: GpScalar] OnlineGpr<O, P, DistanceKernel<DistanceOnly>>,
    refine = P::Refine,
    args = (),
    tail = (),
    points = QueryPoints::NONE,
    count = m,
    cross = {
        /// the `n × m` squared distances from the training samples to the
        /// `m` queries.
    },
    reads = {
        /// A table is read in place for this call (an `f32` model reads it
        /// through a cast); a fill writes scratch once. After an insert or
        /// delete the first read solves `α`.
    },
    predict_doc = {
        /// See the example on [`OnlineGpr::insert`].
    },
    covariance_doc = {
        /// # Examples
        ///
        /// ```rust
        /// use gprx::kernel::{RbfKernel, ScalarDistance};
        /// use gprx::{GaussianLikelihood, Gpr, PredictOptions, Prediction};
        ///
        /// # fn main() -> Result<(), gprx::GprError> {
        /// let image = ScalarDistance::new();
        /// let fitted = Gpr::new(image.kernel(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?)
        ///     .fit([image.from_vec(vec![0.0, 1.0, 1.0, 0.0])], 2, &[0.0, 1.0])
        ///     .map_err(|(_, e)| e)?;
        /// let mut online = fitted.into_online()?;
        /// online.insert([image.from_vec(vec![4.0, 1.0])], 0.5)?;
        /// // Two queries: train × query (3 × 2), then query × query.
        /// let cross = [0.25, 0.25, 2.25, 0.25, 2.25, 0.25];
        /// let query = [0.0, 1.0, 1.0, 0.0];
        /// let options = PredictOptions::default();
        /// let mut out = Prediction::default();
        /// online.predict_into([image.borrow(&cross)], 2, &mut out)?;
        /// online.predict_with_into([image.borrow(&cross)], 2, options, &mut out)?;
        /// let _ = online.predict_with([image.borrow(&cross)], 2, options)?;
        /// let cov = online.predict_covariance([image.borrow(&cross)], [image.borrow(&query)], 2)?;
        /// assert_eq!(cov.covariance.len(), 4);
        /// let _ = online.predict_covariance_with([image.borrow(&cross)], [image.borrow(&query)], 2, options)?;
        /// let draws = online.sample([image.borrow(&cross)], [image.borrow(&query)], 2, 3, 7)?;
        /// assert_eq!(draws.len(), 6);
        /// let _ = online.sample_with([image.borrow(&cross)], [image.borrow(&query)], 2, options, 3, 7)?;
        /// # Ok(())
        /// # }
        /// ```
    },
);

distance_predict!(
    impl [O, P: GpScalar] OnlineGpr<O, P, DistanceKernel<WithPoints>>,
    refine = P::Refine,
    args = (xs: &[f64]),
    tail = (n_cols: usize),
    points = QueryPoints { xs, n_cols },
    count = m,
    cross = {
        /// the `n × m` squared distances from the training samples to the
        /// `m` queries.
    },
    reads = {
        /// A table is read in place for this call (an `f32` model reads it
        /// through a cast); a fill writes scratch once. After an insert or
        /// delete the first read solves `α`.
    },
    predict_doc = {
        /// See the example on [`OnlineGpr::insert`].
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
        /// let fitted = Gpr::new(kernel, GaussianLikelihood::new(0.1)?)
        ///     .fit([image.from_vec(vec![0.0, 1.0, 1.0, 0.0])], 2, &[0.0, 1.0], 1, &[0.0, 1.0])
        ///     .map_err(|(_, e)| e)?;
        /// let mut online = fitted.into_online()?;
        /// online.insert([image.from_vec(vec![4.0, 1.0])], &[2.0], 0.5)?;
        /// let (cross, query, xs) = ([0.25, 0.25, 2.25, 0.25, 2.25, 0.25], [0.0, 1.0, 1.0, 0.0], [0.5, 1.5]);
        /// let options = PredictOptions::default();
        /// let mut out = Prediction::default();
        /// online.predict_into([image.borrow(&cross)], &xs, 2, 1, &mut out)?;
        /// online.predict_with_into([image.borrow(&cross)], &xs, 2, 1, options, &mut out)?;
        /// let _ = online.predict_with([image.borrow(&cross)], &xs, 2, 1, options)?;
        /// let cov = online.predict_covariance([image.borrow(&cross)], [image.borrow(&query)], &xs, 2, 1)?;
        /// assert_eq!(cov.mean.len(), 2);
        /// let _ = online.predict_covariance_with([image.borrow(&cross)], [image.borrow(&query)], &xs, 2, 1, options)?;
        /// let _ = online.sample([image.borrow(&cross)], [image.borrow(&query)], &xs, 2, 1, 2, 0)?;
        /// let _ = online.sample_with([image.borrow(&cross)], [image.borrow(&query)], &xs, 2, 1, options, 2, 0)?;
        /// # Ok(())
        /// # }
        /// ```
    },
);
