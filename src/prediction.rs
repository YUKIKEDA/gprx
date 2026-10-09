//! Prediction results and options of an Exact GPR, and the predict family
//! every model of a [`DistanceKernel`](crate::kernel::DistanceKernel)
//! shares ([`distance_predict`]).

use faer::Mat;

use crate::error::{CholeskyStage, GprError};
use crate::kernel::{DistanceSource, KernelScalar};
use crate::linalg::{cholesky_lower_with_retries, llt_scratch, mul_lower_vec};
use crate::policy::JitterPolicy;

/// Records which predictive variance [`Prediction`] reports.
///
/// # Examples
///
/// ```rust
/// use gprx::VarianceKind;
///
/// assert_eq!(VarianceKind::default(), VarianceKind::Observation);
/// ```
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum VarianceKind {
    /// Marks a variance of the latent function `f*`, without observation noise.
    Latent,
    /// Marks a variance of a new observation `y*`, including `σn²`.
    ///
    /// This is the default.
    #[default]
    Observation,
}

/// Represents the options for [`crate::FittedGpr::predict`], [`crate::FittedGpr::predict_covariance`], and [`crate::FittedGpr::sample`].
///
/// # Examples
///
/// ```rust
/// use gprx::{PredictOptions, VarianceKind};
///
/// let opts = PredictOptions {
///     variance_kind: VarianceKind::Latent,
/// };
/// assert_eq!(opts.variance_kind, VarianceKind::Latent);
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PredictOptions {
    /// Records which variance to return.
    ///
    /// Defaults to [`VarianceKind::Observation`].
    pub variance_kind: VarianceKind,
}

impl Default for PredictOptions {
    fn default() -> Self {
        Self {
            variance_kind: VarianceKind::Observation,
        }
    }
}

/// Represents the predictive mean and (diagonal) variance at the query points.
///
/// [`FittedGpr::predict_into`](crate::FittedGpr::predict_into) reuses `mean` / `variance` capacity when the
/// query length matches a previous call. Query–query covariance is
/// [`PredictiveCovariance`], not a field here.
///
/// # Examples
///
/// ```rust
/// use gprx::{Prediction, VarianceKind};
///
/// let pred = Prediction {
///     mean: vec![0.0],
///     variance: vec![1.0],
///     variance_kind: VarianceKind::Observation,
/// };
/// assert_eq!(pred.mean.len(), pred.variance.len());
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Prediction<T = f64> {
    /// Holds the predictive mean on the original target scale.
    pub mean: Vec<T>,
    /// Holds the predictive variance on the original target scale.
    pub variance: Vec<T>,
    /// Records whether [`Self::variance`] is latent or observation variance.
    pub variance_kind: VarianceKind,
}

/// Represents the predictive mean and query–query covariance at the query points.
///
/// [`crate::FittedGpr::predict_covariance`] returns this type. `covariance` is
/// packed column-major as an `m × m` matrix: index `col * m + row`. The
/// diagonal matches [`Prediction::variance`] for the same query and
/// [`PredictOptions`]. Observation covariance adds `σn²` on the diagonal in
/// the transformed space, then the target transform scales every entry by
/// the same `s²` as variance.
///
/// # Examples
///
/// ```rust
/// use gprx::kernel::{KernelSpec, RbfKernel};
/// use gprx::{Gpr, GaussianLikelihood};
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
/// let likelihood = GaussianLikelihood::new(0.1)?;
/// let gpr = Gpr::new(kernel, likelihood);
/// let fitted = gpr.fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0]).map_err(|(_, e)| e)?;
/// let cov = fitted.predict_covariance(&[0.25, 0.75], 2, 1)?;
/// assert_eq!(cov.covariance.len(), cov.mean.len() * cov.mean.len());
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct PredictiveCovariance<T = f64> {
    /// Holds the predictive mean on the original target scale.
    pub mean: Vec<T>,
    /// Holds the predictive covariance, packed column-major `m × m`.
    pub covariance: Vec<T>,
    /// Records whether the diagonal of [`Self::covariance`] is latent or observation.
    pub variance_kind: VarianceKind,
}

impl<T> Default for Prediction<T> {
    fn default() -> Self {
        Self {
            mean: Vec::new(),
            variance: Vec::new(),
            variance_kind: VarianceKind::default(),
        }
    }
}

impl<T> Default for PredictiveCovariance<T> {
    fn default() -> Self {
        Self {
            mean: Vec::new(),
            covariance: Vec::new(),
            variance_kind: VarianceKind::default(),
        }
    }
}

impl<T: KernelScalar> PredictiveCovariance<T> {
    /// `n_draws` posterior draws, column-major `m × n_draws`: each column is
    /// `μ + L z` with `z ∼ N(0, I)` from gprx's seeded generator at `seed` (Xoshiro256++, the same on every platform), and
    /// `L` the Cholesky factor of [`Self::covariance`], retried with `jitter`.
    /// Zero draws return an empty vector without factoring.
    pub(crate) fn draw(
        &self,
        n_draws: usize,
        seed: u64,
        jitter: JitterPolicy,
    ) -> Result<Vec<T>, GprError> {
        if n_draws == 0 {
            return Ok(Vec::new());
        }
        let m = self.mean.len();
        let mut a = Mat::<T>::zeros(m, m);
        for col in 0..m {
            for row in 0..m {
                a[(row, col)] = self.covariance[col * m + row];
            }
        }
        let mut scratch = llt_scratch::<T>(m);
        cholesky_lower_with_retries(
            &mut a,
            &mut scratch,
            jitter.retry_jitters(),
            CholeskyStage::Predict,
        )?;
        let mut rng = crate::rng::seeded_rng(seed);
        let zero = T::from_f64(0.0);
        let mut out = vec![zero; m * n_draws];
        let mut z = vec![zero; m];
        let mut lz = vec![zero; m];
        for draw in 0..n_draws {
            for slot in &mut z {
                *slot = T::from_f64(crate::rng::unit_normal(&mut rng));
            }
            mul_lower_vec(a.as_ref(), &z, &mut lz);
            let col = &mut out[draw * m..(draw + 1) * m];
            for i in 0..m {
                col[i] = self.mean[i] + lz[i];
            }
        }
        Ok(out)
    }
}

/// The query coordinates of a distance model: none for a
/// [`DistanceOnly`](crate::kernel::DistanceOnly) kernel, the column-major
/// `m × n_cols` coordinates of the coordinate leaves for a
/// [`WithPoints`](crate::kernel::WithPoints) kernel.
#[derive(Clone, Copy, Debug)]
pub(crate) struct QueryPoints<'a> {
    pub(crate) xs: &'a [f64],
    pub(crate) n_cols: usize,
}

impl QueryPoints<'static> {
    /// A kernel without coordinate leaves.
    pub(crate) const NONE: Self = Self { xs: &[], n_cols: 0 };
}

/// The query of a fitted model whose kernel reads supplied distances. The
/// public predict family [`distance_predict`] writes calls these, so every
/// model checks and reads a query in one place.
pub(crate) trait DistanceQuery {
    /// The predict scalar.
    type Refine;

    /// Mean and variance at `m` queries: `cross` is one source per slot of
    /// the squared distances from the model's points to the queries (`n × m`
    /// from the training points for an Exact model, from the inducing
    /// points for a sparse one).
    fn query_distances<'s>(
        &self,
        cross: impl IntoIterator<Item = DistanceSource<'s>>,
        points: QueryPoints<'_>,
        m: usize,
        options: PredictOptions,
    ) -> Result<Prediction<Self::Refine>, GprError>;

    /// [`Self::query_distances`] into `out`, through the model's buffers.
    fn query_distances_into<'s>(
        &mut self,
        cross: impl IntoIterator<Item = DistanceSource<'s>>,
        points: QueryPoints<'_>,
        m: usize,
        options: PredictOptions,
        out: &mut Prediction<Self::Refine>,
    ) -> Result<(), GprError>;

    /// Mean and query × query covariance: `square` is one source per slot
    /// of the `m × m` query squares.
    fn query_distance_covariance<'s>(
        &self,
        cross: impl IntoIterator<Item = DistanceSource<'s>>,
        square: impl IntoIterator<Item = DistanceSource<'s>>,
        points: QueryPoints<'_>,
        m: usize,
        options: PredictOptions,
    ) -> Result<PredictiveCovariance<Self::Refine>, GprError>;

    /// The jitter retries of a posterior draw.
    fn draw_jitter(&self) -> JitterPolicy;
}

/// The public predict family of a model of a distance kernel:
/// `predict`, `predict_into`, `predict_with`, `predict_with_into`,
/// `predict_covariance`, `predict_covariance_with`, `sample`, and
/// `sample_with`, over [`DistanceQuery`].
///
/// `refine` is the model's predict scalar. `args` are the coordinate
/// arguments before the query count `count` and `tail` those after it
/// (none for a `DistanceOnly` kernel); `points` is the [`QueryPoints`]
/// they make. `cross` names the block of one slot, from the model's points
/// to the queries. `reads` says how the model reads a
/// table; `predict_doc` / `covariance_doc` end the docs of the `predict`
/// and the covariance methods.
macro_rules! distance_predict {
    (
        impl [$($gen:tt)*] $model:ty,
        refine = $refine:ty,
        args = ($($arg:ident: $ty:ty),*),
        tail = ($($tail:ident: $tty:ty),*),
        points = $points:expr,
        count = $count:ident,
        cross = { $(#[$cross:meta])* },
        reads = { $(#[$reads:meta])* },
        predict_doc = { $(#[$pdoc:meta])* },
        covariance_doc = { $(#[$cdoc:meta])* },
    ) => {
        impl<$($gen)*> $model {
            /// Predicts at the queries with [`PredictOptions::default`]
            /// (observation variance).
            ///
            /// `sources` holds one source per slot:
            $(#[$cross])*
            $(#[$reads])*
            ///
            /// # Errors
            ///
            #[doc = concat!("Returns [`GprError::EmptyInput`] if `", stringify!($count), "` is zero,")]
            /// [`GprError::LengthMismatch`] if a table has the wrong length
            /// or a slot has no source or two,
            /// [`GprError::InvalidDistance`] for a value that is not finite or
            /// is negative (see [`crate::kernel::DistanceSource::tidy`]), and
            /// the query errors of the coordinate model's `predict`.
            ///
            $(#[$pdoc])*
            pub fn predict<'s>(
                &self,
                sources: impl IntoIterator<Item = DistanceSource<'s>>,
                $($arg: $ty,)*
                $count: usize,
                $($tail: $tty,)*
            ) -> Result<Prediction<$refine>, GprError> {
                self.predict_with(sources, $($arg,)* $count, $($tail,)* PredictOptions::default())
            }

            /// Writes [`Self::predict`] into `out`, reusing its capacity.
            ///
            /// # Errors
            ///
            /// Same as [`Self::predict`]. On an error the contents of `out`
            /// are unspecified: a value a table fails on can be found after
            /// part of the kernel block is written.
            ///
            $(#[$pdoc])*
            pub fn predict_into<'s>(
                &mut self,
                sources: impl IntoIterator<Item = DistanceSource<'s>>,
                $($arg: $ty,)*
                $count: usize,
                $($tail: $tty,)*
                out: &mut Prediction<$refine>,
            ) -> Result<(), GprError> {
                self.predict_with_into(sources, $($arg,)* $count, $($tail,)* PredictOptions::default(), out)
            }

            /// Predicts with an explicit variance kind.
            ///
            /// # Errors
            ///
            /// Same as [`Self::predict`].
            ///
            $(#[$pdoc])*
            pub fn predict_with<'s>(
                &self,
                sources: impl IntoIterator<Item = DistanceSource<'s>>,
                $($arg: $ty,)*
                $count: usize,
                $($tail: $tty,)*
                options: PredictOptions,
            ) -> Result<Prediction<$refine>, GprError> {
                $crate::prediction::DistanceQuery::query_distances(
                    self,
                    sources,
                    $points,
                    $count,
                    options,
                )
            }

            /// Writes [`Self::predict_with`] into `out`, reusing its capacity.
            ///
            /// # Errors
            ///
            /// Same as [`Self::predict_into`], `out` included.
            ///
            $(#[$pdoc])*
            pub fn predict_with_into<'s>(
                &mut self,
                sources: impl IntoIterator<Item = DistanceSource<'s>>,
                $($arg: $ty,)*
                $count: usize,
                $($tail: $tty,)*
                options: PredictOptions,
                out: &mut Prediction<$refine>,
            ) -> Result<(), GprError> {
                $crate::prediction::DistanceQuery::query_distances_into(
                    self,
                    sources,
                    $points,
                    $count,
                    options,
                    out,
                )
            }

            /// Returns the predictive mean and query–query covariance.
            ///
            /// `cross` holds one source per slot:
            $(#[$cross])*
            /// `square` holds the squared distances between the queries
            /// (square, zero diagonal, symmetric).
            $(#[$reads])*
            ///
            /// # Errors
            ///
            /// Same as [`Self::predict`], plus [`GprError::InvalidDistance`]
            /// if a query square's diagonal is not zero or it is not
            /// symmetric (see [`crate::kernel::DistanceSource::tidy`]).
            ///
            $(#[$cdoc])*
            pub fn predict_covariance<'s>(
                &self,
                cross: impl IntoIterator<Item = DistanceSource<'s>>,
                square: impl IntoIterator<Item = DistanceSource<'s>>,
                $($arg: $ty,)*
                $count: usize,
                $($tail: $tty,)*
            ) -> Result<PredictiveCovariance<$refine>, GprError> {
                self.predict_covariance_with(
                    cross,
                    square,
                    $($arg,)*
                    $count,
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
            $(#[$cdoc])*
            pub fn predict_covariance_with<'s>(
                &self,
                cross: impl IntoIterator<Item = DistanceSource<'s>>,
                square: impl IntoIterator<Item = DistanceSource<'s>>,
                $($arg: $ty,)*
                $count: usize,
                $($tail: $tty,)*
                options: PredictOptions,
            ) -> Result<PredictiveCovariance<$refine>, GprError> {
                $crate::prediction::DistanceQuery::query_distance_covariance(
                    self,
                    cross,
                    square,
                    $points,
                    $count,
                    options,
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
            $(#[$cdoc])*
            #[allow(clippy::too_many_arguments)]
            pub fn sample<'s>(
                &self,
                cross: impl IntoIterator<Item = DistanceSource<'s>>,
                square: impl IntoIterator<Item = DistanceSource<'s>>,
                $($arg: $ty,)*
                $count: usize,
                $($tail: $tty,)*
                n_draws: usize,
                seed: u64,
            ) -> Result<Vec<$refine>, GprError> {
                self.sample_with(
                    cross,
                    square,
                    $($arg,)*
                    $count,
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
            $(#[$cdoc])*
            #[allow(clippy::too_many_arguments)]
            pub fn sample_with<'s>(
                &self,
                cross: impl IntoIterator<Item = DistanceSource<'s>>,
                square: impl IntoIterator<Item = DistanceSource<'s>>,
                $($arg: $ty,)*
                $count: usize,
                $($tail: $tty,)*
                options: PredictOptions,
                n_draws: usize,
                seed: u64,
            ) -> Result<Vec<$refine>, GprError> {
                let jitter = $crate::prediction::DistanceQuery::draw_jitter(self);
                self.predict_covariance_with(cross, square, $($arg,)* $count, $($tail,)* options)?
                    .draw(n_draws, seed, jitter)
            }
        }
    };
}

pub(crate) use distance_predict;
