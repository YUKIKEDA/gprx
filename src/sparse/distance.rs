//! The supplied distances of a sparse prediction, and the predict family of
//! a sparse model of a [`DistanceKernel`](crate::kernel::DistanceKernel).

use crate::error::GprError;
use crate::kernel::{BlockKind, DistanceSource, QuerySources, spec_slots};

use super::{QueryDist, SparseCore};

impl QueryDist {
    /// Binds the train × query blocks (`n × q`) and, for a covariance, the
    /// query × query squares (`q × q`) of `core`, and keeps their inducing
    /// rows.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::EmptyInput`] if `q` is zero, and the errors of
    /// binding the sources.
    pub(crate) fn bind<'s>(
        core: &SparseCore,
        cross: Vec<DistanceSource<'s>>,
        square: Option<Vec<DistanceSource<'s>>>,
        q: usize,
    ) -> Result<Self, GprError> {
        crate::data::require_nonempty(q)?;
        let inducing = core
            .dist
            .as_ref()
            .map_or(&[][..], |dist| &dist.inducing[..]);
        let slots = spec_slots(&core.kernel);
        let zq = QuerySources::<f64>::bind(&slots, cross, core.n, q, BlockKind::Rect)?
            .gather_rows(inducing);
        let qq = square
            .map(|square| {
                let all: Vec<usize> = (0..q).collect();
                QuerySources::<f64>::bind(&slots, square, q, q, BlockKind::Square)
                    .map(|square| square.gather_rows(&all))
            })
            .transpose()?;
        Ok(Self { zq: Some(zq), qq })
    }
}

/// The predict family of a sparse distance model: `impl [generics] Model`,
/// with the coordinate arguments a [`WithPoints`](crate::kernel::WithPoints)
/// model adds (`args` before the query count, `tail` after it) and the
/// coordinates and feature count the query reads.
macro_rules! sparse_distance_predict {
    (
        impl [$($gen:tt)*] $model:ty,
        args = ($($arg:ident: $ty:ty),*),
        tail = ($($tail:ident: $tty:ty),*),
        xs = $xs:expr,
        n_cols = $n_cols:expr,
        predict_doc = $predict_doc:literal,
        covariance_doc = $cov_doc:literal,
    ) => {
        impl<$($gen)*> $model {
            /// Predicts at `q` queries with [`PredictOptions::default`]
            /// (observation variance).
            ///
            /// `sources` holds one source per slot: the `n × q` squared
            /// distances from the training samples to the queries. The model
            /// keeps the rows of its inducing points for this call.
            ///
            /// # Errors
            ///
            /// Returns [`GprError::EmptyInput`] if `q` is zero,
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
                q: usize,
                $($tail: $tty,)*
            ) -> Result<Prediction<P::Refine>, GprError> {
                self.predict_with(sources, $($arg,)* q, $($tail,)* PredictOptions::default())
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
                q: usize,
                $($tail: $tty,)*
                out: &mut Prediction<P::Refine>,
            ) -> Result<(), GprError> {
                self.predict_with_into(sources, $($arg,)* q, $($tail,)* PredictOptions::default(), out)
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
                q: usize,
                $($tail: $tty,)*
                options: PredictOptions,
            ) -> Result<Prediction<P::Refine>, GprError> {
                let qd = QueryDist::bind(&self.core, sources.into_iter().collect(), None, q)?;
                self.query($xs, q, $n_cols, &qd, options)
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
                q: usize,
                $($tail: $tty,)*
                options: PredictOptions,
                out: &mut Prediction<P::Refine>,
            ) -> Result<(), GprError> {
                let qd = QueryDist::bind(&self.core, sources.into_iter().collect(), None, q)?;
                self.query_into($xs, q, $n_cols, &qd, options, out)
            }

            /// Returns the predictive mean and query–query covariance.
            ///
            /// `cross` holds the `n × q` squared distances from the training
            /// samples to the queries; `square` the `q × q` ones between the
            /// queries (zero diagonal, symmetric).
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
                q: usize,
                $($tail: $tty,)*
            ) -> Result<PredictiveCovariance<P::Refine>, GprError> {
                self.predict_covariance_with(
                    cross,
                    square,
                    $($arg,)*
                    q,
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
                q: usize,
                $($tail: $tty,)*
                options: PredictOptions,
            ) -> Result<PredictiveCovariance<P::Refine>, GprError> {
                let qd = QueryDist::bind(
                    &self.core,
                    cross.into_iter().collect(),
                    Some(square.into_iter().collect()),
                    q,
                )?;
                self.query_covariance($xs, q, $n_cols, &qd, options)
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
                q: usize,
                $($tail: $tty,)*
                n_draws: usize,
                seed: u64,
            ) -> Result<Vec<P::Refine>, GprError> {
                self.sample_with(
                    cross,
                    square,
                    $($arg,)*
                    q,
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
                q: usize,
                $($tail: $tty,)*
                options: PredictOptions,
                n_draws: usize,
                seed: u64,
            ) -> Result<Vec<P::Refine>, GprError> {
                self.predict_covariance_with(cross, square, $($arg,)* q, $($tail,)* options)?
                    .draw(n_draws, seed, self.core.jitter)
            }
        }
    };
}

pub(crate) use sparse_distance_predict;
