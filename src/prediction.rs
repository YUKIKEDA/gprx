//! Prediction results and options of an Exact GPR.

use faer::Mat;

use crate::error::{CholeskyStage, GprError};
use crate::kernel::KernelScalar;
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
