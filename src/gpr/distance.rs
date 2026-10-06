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
    BlockKind, DistanceKernel, DistanceOnly, DistanceSlot, DistanceSource, QuerySources, RectSlots,
    WithPoints,
};
use crate::optimizer::{Fixed, Optimizer};
use crate::points::PointId;
use crate::precision::GpScalar;
use crate::{PredictOptions, Prediction, PredictiveCovariance};

use super::shared::Query;
use super::{FittedGpr, Gpr, OnlineGpr, TrainInput};

/// Binds the train × query blocks (`n × m`) and, for a covariance, the
/// query × query squares (`m × m`), then runs `f` on the query.
#[allow(clippy::too_many_arguments)]
fn with_query<'s, P: GpScalar, R>(
    slots: &[DistanceSlot],
    n: usize,
    xs: &[f64],
    m: usize,
    n_cols: usize,
    cross: Vec<DistanceSource<'s>>,
    square: Option<Vec<DistanceSource<'s>>>,
    f: impl FnOnce(Query<'_, P::Storage>) -> Result<R, GprError>,
) -> Result<R, GprError> {
    crate::data::require_nonempty(m)?;
    let mut cross = QuerySources::<P::Storage>::bind(slots, cross, n, m, BlockKind::Rect)?;
    let mut square = square
        .map(|square| QuerySources::<P::Storage>::bind(slots, square, m, m, BlockKind::Square))
        .transpose()?;
    let (table, table64) = cross.tables();
    let square_table = square.as_mut().map(QuerySources::table);
    f(Query {
        xs,
        m,
        n_cols,
        cross: Some(&table),
        cross64: Some(&table64),
        square: square_table
            .as_ref()
            .map(|t| t as &dyn RectSlots<P::Storage>),
    })
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
    /// [`GprError::ShapeMismatch`] if a training square has a non-zero
    /// diagonal or is not symmetric, and the errors of the coordinate
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

/// The predict family of a distance model. `$points` are the coordinate
/// arguments a [`WithPoints`] model adds (none for [`DistanceOnly`]).
macro_rules! distance_predict {
    (
        model = $model:ident,
        marker = $marker:ty,
        args = ($($arg:ident: $ty:ty),*),
        tail = ($($tail:ident: $tty:ty),*),
        xs = $xs:expr,
        n_cols = $n_cols:expr,
        alpha = |$this:ident| $alpha:expr,
        predict_doc = $predict_doc:literal,
        covariance_doc = $cov_doc:literal,
    ) => {
        impl<O, P: GpScalar> $model<O, P, DistanceKernel<$marker>> {
            /// Predicts at `m` queries with [`PredictOptions::default`]
            /// (observation variance).
            ///
            /// `sources` holds one source per slot: the `n × m` squared
            /// distances from the training samples to the queries. A table
            /// is read in place for this call (an `f32` model reads it
            /// through a cast); a fill writes scratch once.
            ///
            /// # Errors
            ///
            /// Returns [`GprError::EmptyInput`] if `m` is zero,
            /// [`GprError::LengthMismatch`] if a table has the wrong length
            /// or a slot has no source or two, [`GprError::NonFiniteInput`]
            /// for a non-finite value, and the query errors of the
            /// coordinate model's `predict`.
            ///
            #[doc = $predict_doc]
            pub fn predict<'s>(
                &self,
                sources: impl IntoIterator<Item = DistanceSource<'s>>,
                $($arg: $ty,)*
                m: usize,
                $($tail: $tty,)*
            ) -> Result<Prediction<P::Refine>, GprError> {
                self.predict_with(sources, $($arg,)* m, $($tail,)* PredictOptions::default())
            }

            /// Writes [`Self::predict`] into `out`, reusing its capacity.
            ///
            /// # Errors
            ///
            /// Same as [`Self::predict`].
            ///
            #[doc = $predict_doc]
            pub fn predict_into<'s>(
                &mut self,
                sources: impl IntoIterator<Item = DistanceSource<'s>>,
                $($arg: $ty,)*
                m: usize,
                $($tail: $tty,)*
                out: &mut Prediction<P::Refine>,
            ) -> Result<(), GprError> {
                self.predict_with_into(sources, $($arg,)* m, $($tail,)* PredictOptions::default(), out)
            }

            /// Predicts with an explicit variance kind.
            ///
            /// # Errors
            ///
            /// Same as [`Self::predict`].
            ///
            #[doc = $predict_doc]
            pub fn predict_with<'s>(
                &self,
                sources: impl IntoIterator<Item = DistanceSource<'s>>,
                $($arg: $ty,)*
                m: usize,
                $($tail: $tty,)*
                options: PredictOptions,
            ) -> Result<Prediction<P::Refine>, GprError> {
                let slots = crate::kernel::spec_slots(&self.core.kernel);
                let $this = self;
                let alpha = $alpha?;
                let mut out = Prediction::default();
                with_query::<P, ()>(
                    &slots,
                    self.core.n,
                    $xs,
                    m,
                    $n_cols,
                    sources.into_iter().collect(),
                    None,
                    |q| {
                        self.core
                            .write_prediction(self.factor(), alpha, q, options, &mut out)
                    },
                )?;
                Ok(out)
            }

            /// Writes [`Self::predict_with`] into `out`, reusing its capacity.
            ///
            /// # Errors
            ///
            /// Same as [`Self::predict`].
            ///
            #[doc = $predict_doc]
            pub fn predict_with_into<'s>(
                &mut self,
                sources: impl IntoIterator<Item = DistanceSource<'s>>,
                $($arg: $ty,)*
                m: usize,
                $($tail: $tty,)*
                options: PredictOptions,
                out: &mut Prediction<P::Refine>,
            ) -> Result<(), GprError> {
                let slots = crate::kernel::spec_slots(&self.core.kernel);
                let n = self.core.n;
                with_query::<P, ()>(
                    &slots,
                    n,
                    $xs,
                    m,
                    $n_cols,
                    sources.into_iter().collect(),
                    None,
                    |q| self.predict_query_into(q, options, out),
                )
            }

            /// Returns the predictive mean and query–query covariance.
            ///
            /// `cross` holds the `n × m` squared distances from the training
            /// samples to the queries; `square` the `m × m` ones between the
            /// queries (zero diagonal, symmetric). Both are read in place for
            /// this call.
            ///
            /// # Errors
            ///
            /// Same as [`Self::predict`], plus [`GprError::ShapeMismatch`] if
            /// a query square has a non-zero diagonal or is not symmetric.
            ///
            #[doc = $cov_doc]
            pub fn predict_covariance<'s>(
                &self,
                cross: impl IntoIterator<Item = DistanceSource<'s>>,
                square: impl IntoIterator<Item = DistanceSource<'s>>,
                $($arg: $ty,)*
                m: usize,
                $($tail: $tty,)*
            ) -> Result<PredictiveCovariance<P::Refine>, GprError> {
                self.predict_covariance_with(
                    cross,
                    square,
                    $($arg,)*
                    m,
                    $($tail,)*
                    PredictOptions::default(),
                )
            }

            /// Returns query–query covariance with an explicit variance kind.
            ///
            /// # Errors
            ///
            /// Same as [`Self::predict_covariance`].
            ///
            #[doc = $cov_doc]
            pub fn predict_covariance_with<'s>(
                &self,
                cross: impl IntoIterator<Item = DistanceSource<'s>>,
                square: impl IntoIterator<Item = DistanceSource<'s>>,
                $($arg: $ty,)*
                m: usize,
                $($tail: $tty,)*
                options: PredictOptions,
            ) -> Result<PredictiveCovariance<P::Refine>, GprError> {
                let slots = crate::kernel::spec_slots(&self.core.kernel);
                let $this = self;
                let alpha = $alpha?;
                with_query::<P, _>(
                    &slots,
                    self.core.n,
                    $xs,
                    m,
                    $n_cols,
                    cross.into_iter().collect(),
                    Some(square.into_iter().collect()),
                    |q| self.core.write_covariance(self.factor(), alpha, q, options),
                )
            }

            /// Draws posterior samples from [`Self::predict_covariance`].
            ///
            /// `seed` starts gprx's seeded generator, as in the coordinate
            /// model's `sample`.
            ///
            /// # Errors
            ///
            /// Same as [`Self::predict_covariance`], plus
            /// [`GprError::CholeskyFailed`] if the posterior covariance does
            /// not factor.
            ///
            #[doc = $cov_doc]
            #[allow(clippy::too_many_arguments)]
            pub fn sample<'s>(
                &self,
                cross: impl IntoIterator<Item = DistanceSource<'s>>,
                square: impl IntoIterator<Item = DistanceSource<'s>>,
                $($arg: $ty,)*
                m: usize,
                $($tail: $tty,)*
                n_draws: usize,
                seed: u64,
            ) -> Result<Vec<P::Refine>, GprError> {
                self.sample_with(
                    cross,
                    square,
                    $($arg,)*
                    m,
                    $($tail,)*
                    PredictOptions::default(),
                    n_draws,
                    seed,
                )
            }

            /// Draws posterior samples with an explicit variance kind.
            ///
            /// # Errors
            ///
            /// Same as [`Self::sample`].
            ///
            #[doc = $cov_doc]
            #[allow(clippy::too_many_arguments)]
            pub fn sample_with<'s>(
                &self,
                cross: impl IntoIterator<Item = DistanceSource<'s>>,
                square: impl IntoIterator<Item = DistanceSource<'s>>,
                $($arg: $ty,)*
                m: usize,
                $($tail: $tty,)*
                options: PredictOptions,
                n_draws: usize,
                seed: u64,
            ) -> Result<Vec<P::Refine>, GprError> {
                self.predict_covariance_with(cross, square, $($arg,)* m, $($tail,)* options)?
                    .draw(n_draws, seed, self.core.policies.jitter)
            }
        }
    };
}

distance_predict!(
    model = FittedGpr,
    marker = DistanceOnly,
    args = (),
    tail = (),
    xs = &[],
    n_cols = 0,
    alpha = |model| Ok::<_, GprError>(&model.core.alpha[..]),
    predict_doc = "See the example on [`crate::kernel::ScalarDistance`].",
    covariance_doc = "# Examples\n\n```rust\nuse gprx::kernel::{RbfKernel, ScalarDistance};\nuse gprx::{GaussianLikelihood, Gpr, PredictOptions, Prediction, VarianceKind};\n\n# fn main() -> Result<(), gprx::GprError> {\nlet image = ScalarDistance::new();\nlet mut fitted = Gpr::new(image.kernel(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?)\n    .fit([image.from_vec(vec![0.0, 1.0, 1.0, 0.0])], 2, &[0.0, 1.0])\n    .map_err(|(_, e)| e)?;\n// Two queries: train × query, then query × query.\nlet cross = [0.25, 0.25, 0.25, 2.25];\nlet query = [0.0, 1.0, 1.0, 0.0];\nlet latent = PredictOptions { variance_kind: VarianceKind::Latent };\nlet mut out = Prediction::default();\nfitted.predict_into([image.borrow(&cross)], 2, &mut out)?;\nfitted.predict_with_into([image.borrow(&cross)], 2, latent, &mut out)?;\nlet _ = fitted.predict_with([image.borrow(&cross)], 2, latent)?;\nlet cov = fitted.predict_covariance([image.borrow(&cross)], [image.borrow(&query)], 2)?;\nassert_eq!(cov.covariance.len(), 4);\nlet _ = fitted.predict_covariance_with([image.borrow(&cross)], [image.borrow(&query)], 2, latent)?;\nlet draws = fitted.sample([image.borrow(&cross)], [image.borrow(&query)], 2, 3, 7)?;\nassert_eq!(draws.len(), 6);\nlet _ = fitted.sample_with([image.borrow(&cross)], [image.borrow(&query)], 2, latent, 3, 7)?;\n# Ok(())\n# }\n```",
);

distance_predict!(
    model = FittedGpr,
    marker = WithPoints,
    args = (xs: &[f64]),
    tail = (n_cols: usize),
    xs = xs,
    n_cols = n_cols,
    alpha = |model| Ok::<_, GprError>(&model.core.alpha[..]),
    predict_doc = "See the example on [`Gpr::fit`] of a [`DistanceKernel<WithPoints>`].",
    covariance_doc = "# Examples\n\n```rust\nuse gprx::kernel::{KernelSpec, RbfKernel, ScalarDistance};\nuse gprx::{GaussianLikelihood, Gpr, PredictOptions, Prediction};\n\n# fn main() -> Result<(), gprx::GprError> {\nlet image = ScalarDistance::new();\nlet kernel = image.kernel(RbfKernel::new(1.0)?) * KernelSpec::from(RbfKernel::new(0.5)?);\nlet mut fitted = Gpr::new(kernel, GaussianLikelihood::new(0.1)?)\n    .fit([image.from_vec(vec![0.0, 1.0, 1.0, 0.0])], 2, &[0.0, 1.0], 1, &[0.0, 1.0])\n    .map_err(|(_, e)| e)?;\nlet (cross, query, xs) = ([0.25, 0.25, 0.25, 2.25], [0.0, 1.0, 1.0, 0.0], [0.5, 1.5]);\nlet options = PredictOptions::default();\nlet mut out = Prediction::default();\nfitted.predict_into([image.borrow(&cross)], &xs, 2, 1, &mut out)?;\nfitted.predict_with_into([image.borrow(&cross)], &xs, 2, 1, options, &mut out)?;\nlet _ = fitted.predict_with([image.borrow(&cross)], &xs, 2, 1, options)?;\nlet cov = fitted.predict_covariance([image.borrow(&cross)], [image.borrow(&query)], &xs, 2, 1)?;\nassert_eq!(cov.mean.len(), 2);\nlet _ = fitted.predict_covariance_with([image.borrow(&cross)], [image.borrow(&query)], &xs, 2, 1, options)?;\nlet _ = fitted.sample([image.borrow(&cross)], [image.borrow(&query)], &xs, 2, 1, 2, 0)?;\nlet _ = fitted.sample_with([image.borrow(&cross)], [image.borrow(&query)], &xs, 2, 1, options, 2, 0)?;\n# Ok(())\n# }\n```",
);

distance_predict!(
    model = OnlineGpr,
    marker = DistanceOnly,
    args = (),
    tail = (),
    xs = &[],
    n_cols = 0,
    alpha = |model| model.predict_alpha(),
    predict_doc = "See the example on [`OnlineGpr::insert`] of a [`DistanceKernel<DistanceOnly>`].",
    covariance_doc =
        "See the example on [`OnlineGpr::insert`] of a [`DistanceKernel<DistanceOnly>`].",
);

distance_predict!(
    model = OnlineGpr,
    marker = WithPoints,
    args = (xs: &[f64]),
    tail = (n_cols: usize),
    xs = xs,
    n_cols = n_cols,
    alpha = |model| model.predict_alpha(),
    predict_doc = "See the example on [`OnlineGpr::insert`] of a [`DistanceKernel<WithPoints>`].",
    covariance_doc = "See the example on [`OnlineGpr::insert`] of a [`DistanceKernel<WithPoints>`].",
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
