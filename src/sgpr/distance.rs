//! SGPR on supplied squared distances: the `fit`, `factor`, and predict of
//! [`Sgpr`] and [`FittedSgpr`] for a [`DistanceKernel`].
//!
//! The inducing points are training points, named by index. Every other
//! method is the one of the coordinate model.

use crate::error::GprError;
use crate::kernel::{DistanceKernel, DistanceOnly, DistanceSource, ModelKernel, WithPoints};
use crate::optimizer::{Fixed, Optimizer};
use crate::policy::with_kernel_exp;
use crate::precision::GpScalar;
use crate::sparse::{QueryDist, SparseCore, sparse_distance_predict};
use crate::{PredictOptions, Prediction, PredictiveCovariance};

use super::factor::assemble_fitted;
use super::{FittedSgpr, FixedInducing, InducingLayout, Sgpr, SgprObjective};

impl<O, P: GpScalar, K: ModelKernel> Sgpr<O, FixedInducing, P, K> {
    /// Assembles the VFE system of `core` and searches `θ`.
    #[allow(clippy::result_large_err)]
    fn search(
        self,
        core: Result<SparseCore, GprError>,
    ) -> Result<FittedSgpr<O, FixedInducing, P, K>, (Self, GprError)>
    where
        O: Clone + for<'a> Optimizer<SgprObjective<'a, O, FixedInducing, P, K>>,
    {
        let core = match core {
            Ok(core) => core,
            Err(err) => return Err((self, err)),
        };
        match with_kernel_exp!(self.spec.math, M => assemble_fitted::<_, _, M, _, _>(
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

impl<P: GpScalar, K: ModelKernel> Sgpr<Fixed, FixedInducing, P, K> {
    /// Assembles the VFE system of `core` at the current `θ`.
    #[allow(clippy::result_large_err)]
    fn assemble(
        self,
        core: Result<SparseCore, GprError>,
    ) -> Result<FittedSgpr<Fixed, FixedInducing, P, K>, (Self, GprError)> {
        let core = match core {
            Ok(core) => core,
            Err(err) => return Err((self, err)),
        };
        match with_kernel_exp!(self.spec.math, M => assemble_fitted::<_, _, M, _, _>(core, Fixed)) {
            Ok(fitted) => Ok(fitted),
            Err(err) => Err((self, err)),
        }
    }
}

impl<O, P> Sgpr<O, FixedInducing, P, DistanceKernel<DistanceOnly>>
where
    P: GpScalar,
    O: Clone
        + for<'a> Optimizer<SgprObjective<'a, O, FixedInducing, P, DistanceKernel<DistanceOnly>>>,
{
    /// Factors the VFE system on supplied squared distances and searches
    /// kernel and likelihood `θ`.
    ///
    /// `sources` holds one `n × n` training square per slot (zero diagonal,
    /// symmetric); the model keeps a copy. `inducing` names the training
    /// points that are the inducing points: `K_mm` and `K_mn` read their
    /// rows of the squares. Duplicates are allowed, as repeated rows of `Z`
    /// are.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::EmptyInput`] if `n` or `inducing` is empty,
    /// [`GprError::IndexOutOfRange`] for an index `≥ n`,
    /// [`GprError::LengthMismatch`] if a table has the wrong length or a slot
    /// has no source or two, [`GprError::ShapeMismatch`] for a non-zero
    /// diagonal or an asymmetric square, and the factor and search errors of
    /// the coordinate [`Sgpr::fit`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{RbfKernel, ScalarDistance};
    /// use gprx::{GaussianLikelihood, Sgpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// // Four points 0, 1, 2, 3 on a line: d²[i + j·4] = (i − j)².
    /// let image = ScalarDistance::new();
    /// let d2 = vec![0.0, 1.0, 4.0, 9.0, 1.0, 0.0, 1.0, 4.0, 4.0, 1.0, 0.0, 1.0, 9.0, 4.0, 1.0, 0.0];
    /// let mut fitted = Sgpr::new(image.kernel(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?)
    ///     .fit([image.from_vec(d2)], 4, &[0.0, 1.0, 0.5, 0.25], &[0, 2])
    ///     .map_err(|(_, e)| e)?;
    /// assert_eq!(fitted.inducing(), &[0, 2]);
    /// assert_eq!(fitted.slots(), image.kernel(RbfKernel::new(1.0)?).slots());
    /// let _kernel = fitted.to_kernel();
    /// // One query at 0.5: its squared distances to the four training points.
    /// let pred = fitted.predict([image.borrow(&[0.25, 0.25, 2.25, 6.25])], 1)?;
    /// assert!(pred.mean[0].is_finite());
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
    ) -> Result<FittedSgpr<O, FixedInducing, P, DistanceKernel<DistanceOnly>>, (Self, GprError)>
    {
        let core = SparseCore::prepare_with_distances(
            &self.spec,
            &[],
            n,
            0,
            y,
            sources.into_iter().collect(),
            inducing,
        );
        self.search(core)
    }
}

impl<P: GpScalar> Sgpr<Fixed, FixedInducing, P, DistanceKernel<DistanceOnly>> {
    /// Factors the VFE system on supplied squared distances at the current
    /// `θ` without a search.
    ///
    /// # Errors
    ///
    /// The input and factor errors of [`Self::fit`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{RbfKernel, ScalarDistance};
    /// use gprx::{Fixed, GaussianLikelihood, Sgpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let image = ScalarDistance::new();
    /// let d2 = vec![0.0, 1.0, 4.0, 9.0, 1.0, 0.0, 1.0, 4.0, 4.0, 1.0, 0.0, 1.0, 9.0, 4.0, 1.0, 0.0];
    /// let fitted = Sgpr::new(image.kernel(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?)
    ///     .with_optimizer(Fixed)
    ///     .factor([image.from_vec(d2)], 4, &[0.0, 1.0, 0.5, 0.25], &[0, 2])
    ///     .map_err(|(_, e)| e)?;
    /// assert_eq!(fitted.m(), 2);
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
    ) -> Result<FittedSgpr<Fixed, FixedInducing, P, DistanceKernel<DistanceOnly>>, (Self, GprError)>
    {
        let core = SparseCore::prepare_with_distances(
            &self.spec,
            &[],
            n,
            0,
            y,
            sources.into_iter().collect(),
            inducing,
        );
        self.assemble(core)
    }
}

impl<O, P> Sgpr<O, FixedInducing, P, DistanceKernel<WithPoints>>
where
    P: GpScalar,
    O: Clone + for<'a> Optimizer<SgprObjective<'a, O, FixedInducing, P, DistanceKernel<WithPoints>>>,
{
    /// Factors the VFE system on supplied squared distances and the
    /// column-major `x` (`n × n_cols`) of the coordinate leaves, and searches
    /// kernel and likelihood `θ`.
    ///
    /// The inducing points are the rows `inducing` of `x` and of the
    /// training squares.
    ///
    /// # Errors
    ///
    /// Same as the [`DistanceKernel<DistanceOnly>`] fit, plus the coordinate
    /// errors of [`Sgpr::fit`] for `x`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel, ScalarDistance};
    /// use gprx::{GaussianLikelihood, Sgpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let image = ScalarDistance::new();
    /// let kernel = image.kernel(RbfKernel::new(1.0)?) * KernelSpec::from(RbfKernel::new(2.0)?);
    /// let d2 = vec![0.0, 1.0, 4.0, 9.0, 1.0, 0.0, 1.0, 4.0, 4.0, 1.0, 0.0, 1.0, 9.0, 4.0, 1.0, 0.0];
    /// let x = [0.0, 1.0, 2.0, 3.0];
    /// let fitted = Sgpr::new(kernel, GaussianLikelihood::new(0.1)?)
    ///     .fit([image.from_vec(d2)], 4, &x, 1, &[0.0, 1.0, 0.5, 0.25], &[0, 2])
    ///     .map_err(|(_, e)| e)?;
    /// assert_eq!(fitted.z(), &[0.0, 2.0]);
    /// let pred = fitted.predict([image.borrow(&[0.25, 0.25, 2.25, 6.25])], &[0.5], 1, 1)?;
    /// assert!(pred.mean[0].is_finite());
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
    ) -> Result<FittedSgpr<O, FixedInducing, P, DistanceKernel<WithPoints>>, (Self, GprError)> {
        let core = SparseCore::prepare_with_distances(
            &self.spec,
            x,
            n,
            n_cols,
            y,
            sources.into_iter().collect(),
            inducing,
        );
        self.search(core)
    }
}

impl<P: GpScalar> Sgpr<Fixed, FixedInducing, P, DistanceKernel<WithPoints>> {
    /// Factors the VFE system on supplied squared distances and `x` at the
    /// current `θ` without a search.
    ///
    /// # Errors
    ///
    /// The input and factor errors of [`Self::fit`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel, ScalarDistance};
    /// use gprx::{Fixed, GaussianLikelihood, Sgpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let image = ScalarDistance::new();
    /// let kernel = image.kernel(RbfKernel::new(1.0)?) + KernelSpec::from(RbfKernel::new(2.0)?);
    /// let d2 = vec![0.0, 1.0, 4.0, 9.0, 1.0, 0.0, 1.0, 4.0, 4.0, 1.0, 0.0, 1.0, 9.0, 4.0, 1.0, 0.0];
    /// let fitted = Sgpr::new(kernel, GaussianLikelihood::new(0.1)?)
    ///     .with_optimizer(Fixed)
    ///     .factor([image.from_vec(d2)], 4, &[0.0, 1.0, 2.0, 3.0], 1, &[0.0, 1.0, 0.5, 0.25], &[1, 3])
    ///     .map_err(|(_, e)| e)?;
    /// assert_eq!(fitted.inducing(), &[1, 3]);
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
    ) -> Result<FittedSgpr<Fixed, FixedInducing, P, DistanceKernel<WithPoints>>, (Self, GprError)>
    {
        let core = SparseCore::prepare_with_distances(
            &self.spec,
            x,
            n,
            n_cols,
            y,
            sources.into_iter().collect(),
            inducing,
        );
        self.assemble(core)
    }
}

sparse_distance_predict!(
    impl [O, I: InducingLayout, P: GpScalar] FittedSgpr<O, I, P, DistanceKernel<DistanceOnly>>,
    args = (),
    tail = (),
    xs = &[],
    n_cols = 0,
    predict_doc = "See the example on [`Sgpr::fit`] of a [`DistanceKernel<DistanceOnly>`].",
    covariance_doc = "# Examples\n\n```rust\nuse gprx::kernel::{RbfKernel, ScalarDistance};\nuse gprx::{Fixed, GaussianLikelihood, PredictOptions, Prediction, Sgpr};\n\n# fn main() -> Result<(), gprx::GprError> {\nlet image = ScalarDistance::new();\nlet d2 = vec![0.0, 1.0, 4.0, 9.0, 1.0, 0.0, 1.0, 4.0, 4.0, 1.0, 0.0, 1.0, 9.0, 4.0, 1.0, 0.0];\nlet mut fitted = Sgpr::new(image.kernel(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?)\n    .with_optimizer(Fixed)\n    .factor([image.from_vec(d2)], 4, &[0.0, 1.0, 0.5, 0.25], &[0, 2])\n    .map_err(|(_, e)| e)?;\n// Queries at 0.5 and 1.5: train × query, then query × query.\nlet cross = [0.25, 0.25, 2.25, 6.25, 2.25, 0.25, 0.25, 2.25];\nlet query = [0.0, 1.0, 1.0, 0.0];\nlet options = PredictOptions::default();\nlet mut out = Prediction::default();\nfitted.predict_into([image.borrow(&cross)], 2, &mut out)?;\nfitted.predict_with_into([image.borrow(&cross)], 2, options, &mut out)?;\nlet _ = fitted.predict_with([image.borrow(&cross)], 2, options)?;\nlet cov = fitted.predict_covariance([image.borrow(&cross)], [image.borrow(&query)], 2)?;\nassert_eq!(cov.covariance.len(), 4);\nlet _ = fitted.predict_covariance_with([image.borrow(&cross)], [image.borrow(&query)], 2, options)?;\nlet draws = fitted.sample([image.borrow(&cross)], [image.borrow(&query)], 2, 3, 7)?;\nassert_eq!(draws.len(), 6);\nlet _ = fitted.sample_with([image.borrow(&cross)], [image.borrow(&query)], 2, options, 3, 7)?;\n# Ok(())\n# }\n```",
);

sparse_distance_predict!(
    impl [O, I: InducingLayout, P: GpScalar] FittedSgpr<O, I, P, DistanceKernel<WithPoints>>,
    args = (xs: &[f64]),
    tail = (n_cols: usize),
    xs = xs,
    n_cols = n_cols,
    predict_doc = "See the example on [`Sgpr::fit`] of a [`DistanceKernel<WithPoints>`].",
    covariance_doc = "# Examples\n\n```rust\nuse gprx::kernel::{KernelSpec, RbfKernel, ScalarDistance};\nuse gprx::{Fixed, GaussianLikelihood, PredictOptions, Prediction, Sgpr};\n\n# fn main() -> Result<(), gprx::GprError> {\nlet image = ScalarDistance::new();\nlet kernel = image.kernel(RbfKernel::new(1.0)?) * KernelSpec::from(RbfKernel::new(2.0)?);\nlet d2 = vec![0.0, 1.0, 4.0, 9.0, 1.0, 0.0, 1.0, 4.0, 4.0, 1.0, 0.0, 1.0, 9.0, 4.0, 1.0, 0.0];\nlet mut fitted = Sgpr::new(kernel, GaussianLikelihood::new(0.1)?)\n    .with_optimizer(Fixed)\n    .factor([image.from_vec(d2)], 4, &[0.0, 1.0, 2.0, 3.0], 1, &[0.0, 1.0, 0.5, 0.25], &[0, 2])\n    .map_err(|(_, e)| e)?;\nlet cross = [0.25, 0.25, 2.25, 6.25, 2.25, 0.25, 0.25, 2.25];\nlet (query, xs) = ([0.0, 1.0, 1.0, 0.0], [0.5, 1.5]);\nlet options = PredictOptions::default();\nlet mut out = Prediction::default();\nfitted.predict_into([image.borrow(&cross)], &xs, 2, 1, &mut out)?;\nfitted.predict_with_into([image.borrow(&cross)], &xs, 2, 1, options, &mut out)?;\nlet _ = fitted.predict_with([image.borrow(&cross)], &xs, 2, 1, options)?;\nlet cov = fitted.predict_covariance([image.borrow(&cross)], [image.borrow(&query)], &xs, 2, 1)?;\nassert_eq!(cov.mean.len(), 2);\nlet _ = fitted.predict_covariance_with([image.borrow(&cross)], [image.borrow(&query)], &xs, 2, 1, options)?;\nlet _ = fitted.sample([image.borrow(&cross)], [image.borrow(&query)], &xs, 2, 1, 2, 0)?;\nlet _ = fitted.sample_with([image.borrow(&cross)], [image.borrow(&query)], &xs, 2, 1, options, 2, 0)?;\n# Ok(())\n# }\n```",
);
