//! The loaded models of a [`DistanceKernel`]: [`LoadedDistanceGpr`],
//! [`LoadedDistanceSgpr`], and [`LoadedDistanceSvgp`].

use std::path::Path;

use crate::error::GprError;
use crate::gpr::{FittedGpr, OnlineGpr};
use crate::kernel::{
    DistanceKernel, DistanceOnly, DistanceSlot, DistanceSource, ModelKernelParts, PointUse,
    WithPoints,
};
use crate::optimizer::Fixed;
use crate::precision::PersistKind;
use crate::sgpr::{FittedSgpr, FixedInducing, OnlineSgpr};
use crate::svgp::FittedSvgp;
use crate::{
    DoublePrecision, MixedPrecision, PointId, PredictOptions, Prediction, ReevaluateKernel,
    SinglePrecision,
};

use super::config::{ModelJson, PointsJson};
use super::sparse::{self, SgprVariants};
use super::{PersistRegistry, Variants, load_precision, read_exact_config, widen};

/// Represents the prediction-only Exact model of a [`DistanceKernel`]
/// loaded from a persist directory written by [`FittedGpr::save`] or
/// [`OnlineGpr::save`].
///
/// One variant per precision and factor kind, as [`super::LoadedGpr`]. `C`
/// is the kernel's [`PointUse`]: loading a directory of the other marker,
/// or of a coordinate model, returns [`GprError::PersistFailed`] with
/// [`crate::PersistErrorKind::WrongModel`].
///
/// The loaded model owns the training `d²` it was saved with, and its
/// kernel's slots are new: a [`crate::kernel::ScalarDistance`] or
/// [`crate::kernel::ArdDistance`] from before the save names none of them.
/// Take the slots from [`Self::slots`] (in the order of
/// [`DistanceKernel::slots`] of the saved kernel) and bind each query's
/// sources to them. [`Self::predict`], [`Self::predict_with`], [`Self::n`],
/// [`Self::is_online`], [`Self::slots`], and [`Self::to_kernel`] work on
/// any variant; match a variant for the typed model.
///
/// # Examples
///
/// ```rust
/// use gprx::kernel::{DistanceOnly, DistanceSlot, RbfKernel, ScalarDistance};
/// use gprx::persist::{LoadedDistanceGpr, PersistRegistry};
/// use gprx::{GaussianLikelihood, Gpr};
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let image = ScalarDistance::new();
/// let train = vec![0.0, 1.0, 4.0, 1.0, 0.0, 1.0, 4.0, 1.0, 0.0];
/// let fitted = Gpr::new(image.kernel(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?)
///     .fit([image.from_vec(train)], 3, &[0.0, 1.0, 0.5])
///     .map_err(|(_, e)| e)?;
/// let dir = std::env::temp_dir().join(format!("gprx-doctest-distance-{}", std::process::id()));
/// let _ = std::fs::remove_dir_all(&dir);
/// fitted.save(&dir)?;
/// let loaded = LoadedDistanceGpr::<DistanceOnly>::load(&dir, &PersistRegistry::new())?;
/// // The loaded kernel has slots of its own: bind the query to them.
/// let [DistanceSlot::Scalar(slot)] = loaded.slots()[..] else {
///     panic!("one scalar slot");
/// };
/// let cross = [0.25, 0.25, 2.25];
/// let got = loaded.predict([slot.borrow(&cross)], 1)?;
/// let want = fitted.predict([image.borrow(&cross)], 1)?;
/// assert_eq!(got.mean, want.mean);
/// let _ = std::fs::remove_dir_all(&dir);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum LoadedDistanceGpr<C: PointUse = DistanceOnly> {
    /// Marks a [`DoublePrecision`] model.
    Double(FittedGpr<Fixed, DoublePrecision, DistanceKernel<C>>),
    /// Marks a [`SinglePrecision`] model.
    Single(FittedGpr<Fixed, SinglePrecision, DistanceKernel<C>>),
    /// Marks a promoted-storage [`MixedPrecision`] model.
    Mixed(FittedGpr<Fixed, MixedPrecision, DistanceKernel<C>>),
    /// Marks a [`MixedPrecision`]`<`[`ReevaluateKernel`]`>` model.
    Reevaluate(FittedGpr<Fixed, MixedPrecision<ReevaluateKernel>, DistanceKernel<C>>),
    /// Marks a [`DoublePrecision`] online model.
    OnlineDouble(OnlineGpr<Fixed, DoublePrecision, DistanceKernel<C>>),
    /// Marks a [`SinglePrecision`] online model.
    OnlineSingle(OnlineGpr<Fixed, SinglePrecision, DistanceKernel<C>>),
    /// Marks a promoted-storage [`MixedPrecision`] online model.
    OnlineMixed(OnlineGpr<Fixed, MixedPrecision, DistanceKernel<C>>),
    /// Marks a [`MixedPrecision`]`<`[`ReevaluateKernel`]`>` online model.
    OnlineReevaluate(OnlineGpr<Fixed, MixedPrecision<ReevaluateKernel>, DistanceKernel<C>>),
}

/// Runs `$body` on the model of any variant, bound to `$model`.
macro_rules! each_model {
    ($value:expr, $model:ident => $body:expr) => {
        match $value {
            LoadedDistanceGpr::Double($model) => $body,
            LoadedDistanceGpr::Single($model) => $body,
            LoadedDistanceGpr::Mixed($model) => $body,
            LoadedDistanceGpr::Reevaluate($model) => $body,
            LoadedDistanceGpr::OnlineDouble($model) => $body,
            LoadedDistanceGpr::OnlineSingle($model) => $body,
            LoadedDistanceGpr::OnlineMixed($model) => $body,
            LoadedDistanceGpr::OnlineReevaluate($model) => $body,
        }
    };
}

/// The distance marker the config of a model of `C` records.
pub(super) fn points_of<C: PointUse>() -> PointsJson {
    if <DistanceKernel<C> as ModelKernelParts>::POINTS {
        PointsJson::WithPoints
    } else {
        PointsJson::DistanceOnly
    }
}

impl<C: PointUse> LoadedDistanceGpr<C> {
    /// Reads `dir/config.json` and `dir/model.safetensors` written by the
    /// `save` of an Exact model of a [`DistanceKernel<C>`].
    ///
    /// Reads the training `d²` back into the store a fit would have made,
    /// with new slots; the factor is read as [`super::LoadedGpr::load`]
    /// reads it, or factored again at the saved `θ` when the file holds
    /// none.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::UnsupportedPersistVersion`] when `format_version`
    /// is not [`super::FORMAT_VERSION`], [`GprError::PersistFailed`] with
    /// [`crate::PersistErrorKind::WrongModel`] when the directory holds
    /// another model (a sparse one, a coordinate one, or one of the other
    /// [`PointUse`]), and [`GprError::PersistFailed`] when the JSON, the
    /// tensors, or a registry lookup is invalid, or when a `DistanceOnly`
    /// file's kernel holds a leaf that reads coordinates. A stored `d²` that
    /// is not finite, negative, or a non-zero diagonal is
    /// [`GprError::InvalidDistance`]. Saved maps that send the training data
    /// past `f64` are [`GprError::NonFiniteInput`]. Factorization errors use
    /// the same variants as the model's `factor`.
    ///
    /// See the example on [`LoadedDistanceGpr`].
    pub fn load(dir: impl AsRef<Path>, registry: &PersistRegistry) -> Result<Self, GprError> {
        let dir = dir.as_ref();
        let config = read_exact_config(dir, Some(points_of::<C>()))?;
        // Each arm names its own precision type, so the call stays in the arm.
        match config.persist_kind() {
            PersistKind::Double => load_precision(
                dir,
                registry,
                config,
                Variants {
                    fitted: Self::Double,
                    online: Self::OnlineDouble,
                },
            ),
            PersistKind::Single => load_precision(
                dir,
                registry,
                config,
                Variants {
                    fitted: Self::Single,
                    online: Self::OnlineSingle,
                },
            ),
            PersistKind::MixedPromote => load_precision(
                dir,
                registry,
                config,
                Variants {
                    fitted: Self::Mixed,
                    online: Self::OnlineMixed,
                },
            ),
            PersistKind::MixedReevaluate => load_precision(
                dir,
                registry,
                config,
                Variants {
                    fitted: Self::Reevaluate,
                    online: Self::OnlineReevaluate,
                },
            ),
        }
    }

    /// Returns the number of training points.
    ///
    /// See the example on [`LoadedDistanceGpr`].
    pub fn n(&self) -> usize {
        each_model!(self, model => model.n())
    }

    /// Returns `true` for an [`OnlineGpr`] (`ldlt`), `false` for a
    /// [`FittedGpr`] (`llt`).
    ///
    /// See the example on [`LoadedDistanceGpr`].
    pub fn is_online(&self) -> bool {
        matches!(
            self,
            Self::OnlineDouble(_)
                | Self::OnlineSingle(_)
                | Self::OnlineMixed(_)
                | Self::OnlineReevaluate(_)
        )
    }

    /// Returns the slots of the loaded kernel, in the order of the saved
    /// kernel's [`DistanceKernel::slots`]. Bind the sources of a query to
    /// these.
    ///
    /// See the example on [`LoadedDistanceGpr`].
    pub fn slots(&self) -> Vec<DistanceSlot> {
        each_model!(self, model => model.slots())
    }

    /// Returns a copy of the loaded kernel, on the slots of [`Self::slots`].
    ///
    /// See the example on [`LoadedDistanceGpr`].
    pub fn to_kernel(&self) -> DistanceKernel<C> {
        each_model!(self, model => model.to_kernel())
    }
}

impl LoadedDistanceGpr<DistanceOnly> {
    /// Returns the predictive mean and variance at `q` queries (observation
    /// variance), in `f64` whatever the stored precision. `sources` holds
    /// one source per slot of [`Self::slots`]: the `n × q` squared
    /// distances from the training points to the queries.
    ///
    /// # Errors
    ///
    /// Same as the variant's `predict`.
    ///
    /// See the example on [`LoadedDistanceGpr`].
    pub fn predict<'s>(
        &self,
        sources: impl IntoIterator<Item = DistanceSource<'s>>,
        q: usize,
    ) -> Result<Prediction<f64>, GprError> {
        self.predict_with(sources, q, PredictOptions::default())
    }

    /// Returns [`Self::predict`] with [`PredictOptions`].
    ///
    /// # Errors
    ///
    /// Same as the variant's `predict_with`.
    ///
    /// See the example on [`LoadedDistanceGpr`].
    pub fn predict_with<'s>(
        &self,
        sources: impl IntoIterator<Item = DistanceSource<'s>>,
        q: usize,
        options: PredictOptions,
    ) -> Result<Prediction<f64>, GprError> {
        each_model!(self, model => model.predict_with(sources, q, options).map(widen))
    }
}

impl LoadedDistanceGpr<WithPoints> {
    /// Returns the number of input features of the coordinate leaves.
    ///
    /// See the example on [`LoadedDistanceGpr`].
    pub fn d(&self) -> usize {
        each_model!(self, model => model.d())
    }

    /// Returns the predictive mean and variance at the `q` queries `xs`
    /// (column-major `q × n_cols`), in `f64` whatever the stored precision.
    /// `sources` holds one source per slot of [`Self::slots`]: the `n × q`
    /// squared distances from the training points to the queries.
    ///
    /// # Errors
    ///
    /// Same as the variant's `predict`.
    ///
    /// See the example on [`LoadedDistanceGpr`].
    pub fn predict<'s>(
        &self,
        sources: impl IntoIterator<Item = DistanceSource<'s>>,
        xs: &[f64],
        q: usize,
        n_cols: usize,
    ) -> Result<Prediction<f64>, GprError> {
        self.predict_with(sources, xs, q, n_cols, PredictOptions::default())
    }

    /// Returns [`Self::predict`] with [`PredictOptions`].
    ///
    /// # Errors
    ///
    /// Same as the variant's `predict_with`.
    ///
    /// See the example on [`LoadedDistanceGpr`].
    pub fn predict_with<'s>(
        &self,
        sources: impl IntoIterator<Item = DistanceSource<'s>>,
        xs: &[f64],
        q: usize,
        n_cols: usize,
        options: PredictOptions,
    ) -> Result<Prediction<f64>, GprError> {
        each_model!(self, model => model.predict_with(sources, xs, q, n_cols, options).map(widen))
    }
}

/// Represents the prediction-only SGPR of a [`DistanceKernel`] loaded from
/// a persist directory written by [`crate::FittedSgpr::save`] or
/// [`crate::OnlineSgpr::save`].
///
/// One variant per precision and model, as [`super::LoadedSgpr`]. The
/// training blocks are bound to new slots, as [`LoadedDistanceGpr`] binds
/// its `d²`: take them from [`Self::slots`]. The VFE system is factored
/// again at the saved `θ`. Loading a directory of the other [`PointUse`],
/// or of a coordinate model, returns [`GprError::PersistFailed`] with
/// [`crate::PersistErrorKind::WrongModel`].
///
/// # Examples
///
/// ```rust
/// use gprx::kernel::{DistanceOnly, DistanceSlot, RbfKernel, ScalarDistance};
/// use gprx::persist::{LoadedDistanceSgpr, PersistRegistry};
/// use gprx::{Fixed, GaussianLikelihood, Sgpr};
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let image = ScalarDistance::new();
/// // Three samples at 0, 1, 2; the inducing points are samples 0 and 2.
/// let train = [0.0, 1.0, 4.0, 4.0, 1.0, 0.0];
/// let fitted = Sgpr::new(image.kernel(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?)
///     .with_optimizer(Fixed)
///     .factor([image.borrow(&train)], 3, &[0.0, 1.0, 0.5], &[0, 2])
///     .map_err(|(_, e)| e)?;
/// let dir = std::env::temp_dir().join(format!("gprx-doctest-dsgpr-{}", std::process::id()));
/// let _ = std::fs::remove_dir_all(&dir);
/// fitted.save(&dir)?;
/// let loaded = LoadedDistanceSgpr::<DistanceOnly>::load(&dir, &PersistRegistry::new())?;
/// assert_eq!(loaded.inducing(), Some(&[0, 2][..]));
/// assert!(loaded.inducing_points().is_none());
/// let [DistanceSlot::Scalar(slot)] = loaded.slots()[..] else {
///     panic!("one scalar slot");
/// };
/// // One query at 1.5: its squared distances to the two inducing points.
/// let cross = [2.25, 0.25];
/// let got = loaded.predict([slot.borrow(&cross)], 1)?;
/// assert_eq!(got.mean, fitted.predict([image.borrow(&cross)], 1)?.mean);
/// // An online model names its inducing points by `PointId`.
/// let online = fitted.into_online();
/// let ids = online.point_ids().to_vec();
/// online.save(&dir)?;
/// let loaded = LoadedDistanceSgpr::<DistanceOnly>::load(&dir, &PersistRegistry::new())?;
/// assert_eq!(loaded.inducing(), None);
/// let points: Option<Vec<_>> = loaded.inducing_points().map(Iterator::collect);
/// assert_eq!(points, Some(vec![ids[0], ids[2]]));
/// let _ = std::fs::remove_dir_all(&dir);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum LoadedDistanceSgpr<C: PointUse = DistanceOnly> {
    /// Marks a [`DoublePrecision`] model.
    Double(FittedSgpr<Fixed, FixedInducing, DoublePrecision, DistanceKernel<C>>),
    /// Marks a [`SinglePrecision`] model.
    Single(FittedSgpr<Fixed, FixedInducing, SinglePrecision, DistanceKernel<C>>),
    /// Marks a promoted-storage [`MixedPrecision`] model.
    Mixed(FittedSgpr<Fixed, FixedInducing, MixedPrecision, DistanceKernel<C>>),
    /// Marks a [`MixedPrecision`]`<`[`ReevaluateKernel`]`>` model.
    Reevaluate(
        FittedSgpr<Fixed, FixedInducing, MixedPrecision<ReevaluateKernel>, DistanceKernel<C>>,
    ),
    /// Marks a [`DoublePrecision`] online model.
    OnlineDouble(OnlineSgpr<Fixed, DoublePrecision, DistanceKernel<C>>),
    /// Marks a [`SinglePrecision`] online model.
    OnlineSingle(OnlineSgpr<Fixed, SinglePrecision, DistanceKernel<C>>),
    /// Marks a promoted-storage [`MixedPrecision`] online model.
    OnlineMixed(OnlineSgpr<Fixed, MixedPrecision, DistanceKernel<C>>),
    /// Marks a [`MixedPrecision`]`<`[`ReevaluateKernel`]`>` online model.
    OnlineReevaluate(OnlineSgpr<Fixed, MixedPrecision<ReevaluateKernel>, DistanceKernel<C>>),
}

/// Runs `$body` on the model of any [`LoadedDistanceSgpr`] variant.
macro_rules! each_sgpr {
    ($value:expr, $model:ident => $body:expr) => {
        match $value {
            LoadedDistanceSgpr::Double($model) => $body,
            LoadedDistanceSgpr::Single($model) => $body,
            LoadedDistanceSgpr::Mixed($model) => $body,
            LoadedDistanceSgpr::Reevaluate($model) => $body,
            LoadedDistanceSgpr::OnlineDouble($model) => $body,
            LoadedDistanceSgpr::OnlineSingle($model) => $body,
            LoadedDistanceSgpr::OnlineMixed($model) => $body,
            LoadedDistanceSgpr::OnlineReevaluate($model) => $body,
        }
    };
}

impl<C: PointUse> LoadedDistanceSgpr<C> {
    /// Reads `dir/config.json` and `dir/model.safetensors` written by the
    /// `save` of an SGPR (or online SGPR) of a [`DistanceKernel<C>`].
    ///
    /// # Errors
    ///
    /// Returns [`GprError::UnsupportedPersistVersion`] when `format_version`
    /// is not [`super::FORMAT_VERSION`], [`GprError::PersistFailed`] with
    /// [`crate::PersistErrorKind::WrongModel`] when the directory holds
    /// another model, and [`GprError::PersistFailed`] when the JSON, the
    /// tensors, or a registry lookup is invalid, when a `DistanceOnly`
    /// file's kernel holds a leaf that reads coordinates, when the number of
    /// inducing indices is not `m`, or when a `WithPoints` file's `z` or
    /// `z_train` is not the training rows its inducing indices name. The
    /// inducing indices and the stored blocks are checked as a fit checks
    /// them: [`GprError::IndexOutOfRange`] for an index not below `n`,
    /// [`GprError::InvalidConfig`] for an index listed twice, and
    /// [`GprError::InvalidDistance`] for a value. Saved maps that send the
    /// training data past `f64` are [`GprError::NonFiniteInput`].
    /// Factorization errors use the same variants as the model's `factor`.
    ///
    /// See the example on [`LoadedDistanceSgpr`].
    pub fn load(dir: impl AsRef<Path>, registry: &PersistRegistry) -> Result<Self, GprError> {
        let dir = dir.as_ref();
        let config = sparse::read_config(
            dir,
            &[ModelJson::Sgpr, ModelJson::OnlineSgpr],
            Some(points_of::<C>()),
        )?;
        // Each arm names its own precision type, so the call stays in the arm.
        match config.persist_kind() {
            PersistKind::Double => sparse::load_sgpr_as(
                dir,
                &config,
                registry,
                SgprVariants {
                    fitted: Self::Double,
                    online: Self::OnlineDouble,
                },
            ),
            PersistKind::Single => sparse::load_sgpr_as(
                dir,
                &config,
                registry,
                SgprVariants {
                    fitted: Self::Single,
                    online: Self::OnlineSingle,
                },
            ),
            PersistKind::MixedPromote => sparse::load_sgpr_as(
                dir,
                &config,
                registry,
                SgprVariants {
                    fitted: Self::Mixed,
                    online: Self::OnlineMixed,
                },
            ),
            PersistKind::MixedReevaluate => sparse::load_sgpr_as(
                dir,
                &config,
                registry,
                SgprVariants {
                    fitted: Self::Reevaluate,
                    online: Self::OnlineReevaluate,
                },
            ),
        }
    }

    /// Returns the number of training points.
    ///
    /// See the example on [`LoadedDistanceSgpr`].
    pub fn n(&self) -> usize {
        each_sgpr!(self, model => model.n())
    }

    /// Returns the number of inducing points.
    ///
    /// See the example on [`LoadedDistanceSgpr`].
    pub fn m(&self) -> usize {
        each_sgpr!(self, model => model.m())
    }

    /// Returns the training samples that are the inducing points of a
    /// fitted variant, in the order of the rows of a prediction's blocks,
    /// or `None` for an [`OnlineSgpr`] variant: its points are named by
    /// [`Self::inducing_points`], since a delete shifts their places.
    ///
    /// See the example on [`LoadedDistanceSgpr`].
    pub fn inducing(&self) -> Option<&[usize]> {
        match self {
            Self::Double(model) => Some(model.inducing()),
            Self::Single(model) => Some(model.inducing()),
            Self::Mixed(model) => Some(model.inducing()),
            Self::Reevaluate(model) => Some(model.inducing()),
            Self::OnlineDouble(_)
            | Self::OnlineSingle(_)
            | Self::OnlineMixed(_)
            | Self::OnlineReevaluate(_) => None,
        }
    }

    /// Returns the [`PointId`]s of the inducing points of an
    /// [`OnlineSgpr`] variant, as [`OnlineSgpr::inducing_points`], or
    /// `None` for a fitted variant (see [`Self::inducing`]).
    ///
    /// See the example on [`LoadedDistanceSgpr`].
    pub fn inducing_points(&self) -> Option<impl Iterator<Item = PointId> + '_> {
        match self {
            Self::OnlineDouble(model) => Some(model.inducing_point_iter()),
            Self::OnlineSingle(model) => Some(model.inducing_point_iter()),
            Self::OnlineMixed(model) => Some(model.inducing_point_iter()),
            Self::OnlineReevaluate(model) => Some(model.inducing_point_iter()),
            Self::Double(_) | Self::Single(_) | Self::Mixed(_) | Self::Reevaluate(_) => None,
        }
    }

    /// Returns `true` for an [`OnlineSgpr`] variant.
    ///
    /// See the example on [`LoadedDistanceSgpr`].
    pub fn is_online(&self) -> bool {
        matches!(
            self,
            Self::OnlineDouble(_)
                | Self::OnlineSingle(_)
                | Self::OnlineMixed(_)
                | Self::OnlineReevaluate(_)
        )
    }

    /// Returns the slots of the loaded kernel, in the order of the saved
    /// kernel's [`DistanceKernel::slots`].
    ///
    /// See the example on [`LoadedDistanceSgpr`].
    pub fn slots(&self) -> Vec<DistanceSlot> {
        each_sgpr!(self, model => model.slots())
    }

    /// Returns a copy of the loaded kernel, on the slots of [`Self::slots`].
    ///
    /// See the example on [`LoadedDistanceSgpr`].
    pub fn to_kernel(&self) -> DistanceKernel<C> {
        each_sgpr!(self, model => model.to_kernel())
    }
}

impl LoadedDistanceSgpr<DistanceOnly> {
    /// Returns the predictive mean and observation variance at `q` queries,
    /// in `f64` whatever the stored precision. `sources` holds one source
    /// per slot of [`Self::slots`]: the `m × q` squared distances
    /// from the inducing points ([`Self::inducing`]) to the queries.
    ///
    /// # Errors
    ///
    /// Same as the variant's `predict`.
    ///
    /// See the example on [`LoadedDistanceSgpr`].
    pub fn predict<'s>(
        &self,
        sources: impl IntoIterator<Item = DistanceSource<'s>>,
        q: usize,
    ) -> Result<Prediction<f64>, GprError> {
        self.predict_with(sources, q, PredictOptions::default())
    }

    /// Returns [`Self::predict`] with [`PredictOptions`].
    ///
    /// # Errors
    ///
    /// Same as the variant's `predict_with`.
    ///
    /// See the example on [`LoadedDistanceSgpr`].
    pub fn predict_with<'s>(
        &self,
        sources: impl IntoIterator<Item = DistanceSource<'s>>,
        q: usize,
        options: PredictOptions,
    ) -> Result<Prediction<f64>, GprError> {
        each_sgpr!(self, model => model.predict_with(sources, q, options).map(widen))
    }
}

impl LoadedDistanceSgpr<WithPoints> {
    /// Returns the number of input features of the coordinate leaves.
    ///
    /// See the example on [`LoadedDistanceSgpr`].
    pub fn d(&self) -> usize {
        each_sgpr!(self, model => model.d())
    }

    /// Returns the predictive mean and observation variance at the `q`
    /// queries `xs` (column-major `q × n_cols`), in `f64` whatever the
    /// stored precision. `sources` holds one source per slot of
    /// [`Self::slots`], from the inducing points to the queries.
    ///
    /// # Errors
    ///
    /// Same as the variant's `predict`.
    ///
    /// See the example on [`LoadedDistanceSgpr`].
    pub fn predict<'s>(
        &self,
        sources: impl IntoIterator<Item = DistanceSource<'s>>,
        xs: &[f64],
        q: usize,
        n_cols: usize,
    ) -> Result<Prediction<f64>, GprError> {
        self.predict_with(sources, xs, q, n_cols, PredictOptions::default())
    }

    /// Returns [`Self::predict`] with [`PredictOptions`].
    ///
    /// # Errors
    ///
    /// Same as the variant's `predict_with`.
    ///
    /// See the example on [`LoadedDistanceSgpr`].
    pub fn predict_with<'s>(
        &self,
        sources: impl IntoIterator<Item = DistanceSource<'s>>,
        xs: &[f64],
        q: usize,
        n_cols: usize,
        options: PredictOptions,
    ) -> Result<Prediction<f64>, GprError> {
        each_sgpr!(self, model => model.predict_with(sources, xs, q, n_cols, options).map(widen))
    }
}

/// Represents the prediction-only SVGP of a [`DistanceKernel`] loaded from
/// a persist directory written by [`crate::FittedSvgp::save`].
///
/// One variant per precision, as [`super::LoadedSvgp`]. The training blocks
/// are bound to new slots ([`Self::slots`]); `K_mm` is factored again at
/// the saved `θ`, with the saved `q(u)`. Loading a directory of the other
/// [`PointUse`], or of a coordinate model, returns
/// [`GprError::PersistFailed`] with [`crate::PersistErrorKind::WrongModel`].
///
/// # Examples
///
/// ```rust
/// use gprx::kernel::{DistanceOnly, DistanceSlot, RbfKernel, ScalarDistance};
/// use gprx::persist::{LoadedDistanceSvgp, PersistRegistry};
/// use gprx::{GaussianLikelihood, Svgp};
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let image = ScalarDistance::new();
/// let train = [0.0, 1.0, 4.0, 4.0, 1.0, 0.0];
/// let fitted = Svgp::new(image.kernel(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?)
///     .factor([image.borrow(&train)], 3, &[0.0, 1.0, 0.5], &[0, 2])
///     .map_err(|(_, e)| e)?;
/// let dir = std::env::temp_dir().join(format!("gprx-doctest-dsvgp-{}", std::process::id()));
/// let _ = std::fs::remove_dir_all(&dir);
/// fitted.save(&dir)?;
/// let loaded = LoadedDistanceSvgp::<DistanceOnly>::load(&dir, &PersistRegistry::new())?;
/// let [DistanceSlot::Scalar(slot)] = loaded.slots()[..] else {
///     panic!("one scalar slot");
/// };
/// let cross = [2.25, 0.25];
/// let got = loaded.predict([slot.borrow(&cross)], 1)?;
/// assert_eq!(got.mean, fitted.predict([image.borrow(&cross)], 1)?.mean);
/// let _ = std::fs::remove_dir_all(&dir);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum LoadedDistanceSvgp<C: PointUse = DistanceOnly> {
    /// Marks a [`DoublePrecision`] model.
    Double(FittedSvgp<DoublePrecision, DistanceKernel<C>>),
    /// Marks a [`SinglePrecision`] model.
    Single(FittedSvgp<SinglePrecision, DistanceKernel<C>>),
    /// Marks a promoted-storage [`MixedPrecision`] model.
    Mixed(FittedSvgp<MixedPrecision, DistanceKernel<C>>),
    /// Marks a [`MixedPrecision`]`<`[`ReevaluateKernel`]`>` model.
    Reevaluate(FittedSvgp<MixedPrecision<ReevaluateKernel>, DistanceKernel<C>>),
}

/// Runs `$body` on the model of any [`LoadedDistanceSvgp`] variant.
macro_rules! each_svgp {
    ($value:expr, $model:ident => $body:expr) => {
        match $value {
            LoadedDistanceSvgp::Double($model) => $body,
            LoadedDistanceSvgp::Single($model) => $body,
            LoadedDistanceSvgp::Mixed($model) => $body,
            LoadedDistanceSvgp::Reevaluate($model) => $body,
        }
    };
}

impl<C: PointUse> LoadedDistanceSvgp<C> {
    /// Reads `dir/config.json` and `dir/model.safetensors` written by the
    /// `save` of an SVGP of a [`DistanceKernel<C>`].
    ///
    /// # Errors
    ///
    /// Same as [`LoadedDistanceSgpr::load`], plus
    /// [`GprError::PersistFailed`] when the saved `q(u)` is not finite or
    /// its `L` is not lower triangular with a positive diagonal.
    ///
    /// See the example on [`LoadedDistanceSvgp`].
    pub fn load(dir: impl AsRef<Path>, registry: &PersistRegistry) -> Result<Self, GprError> {
        let dir = dir.as_ref();
        let config = sparse::read_config(dir, &[ModelJson::Svgp], Some(points_of::<C>()))?;
        match config.persist_kind() {
            PersistKind::Double => sparse::load_svgp_as(dir, &config, registry, Self::Double),
            PersistKind::Single => sparse::load_svgp_as(dir, &config, registry, Self::Single),
            PersistKind::MixedPromote => sparse::load_svgp_as(dir, &config, registry, Self::Mixed),
            PersistKind::MixedReevaluate => {
                sparse::load_svgp_as(dir, &config, registry, Self::Reevaluate)
            }
        }
    }

    /// Returns the number of training points.
    ///
    /// See the example on [`LoadedDistanceSvgp`].
    pub fn n(&self) -> usize {
        each_svgp!(self, model => model.n())
    }

    /// Returns the number of inducing points.
    ///
    /// See the example on [`LoadedDistanceSvgp`].
    pub fn m(&self) -> usize {
        each_svgp!(self, model => model.m())
    }

    /// Returns the training samples that are the inducing points.
    ///
    /// See the example on [`LoadedDistanceSvgp`].
    pub fn inducing(&self) -> &[usize] {
        each_svgp!(self, model => model.inducing())
    }

    /// Returns the slots of the loaded kernel, in the order of the saved
    /// kernel's [`DistanceKernel::slots`].
    ///
    /// See the example on [`LoadedDistanceSvgp`].
    pub fn slots(&self) -> Vec<DistanceSlot> {
        each_svgp!(self, model => model.slots())
    }

    /// Returns a copy of the loaded kernel, on the slots of [`Self::slots`].
    ///
    /// See the example on [`LoadedDistanceSvgp`].
    pub fn to_kernel(&self) -> DistanceKernel<C> {
        each_svgp!(self, model => model.to_kernel())
    }
}

impl LoadedDistanceSvgp<DistanceOnly> {
    /// Returns the predictive mean and observation variance at `q` queries,
    /// in `f64` whatever the stored precision; `sources` as
    /// [`LoadedDistanceSgpr::predict`] takes them.
    ///
    /// # Errors
    ///
    /// Same as the variant's `predict`.
    ///
    /// See the example on [`LoadedDistanceSvgp`].
    pub fn predict<'s>(
        &self,
        sources: impl IntoIterator<Item = DistanceSource<'s>>,
        q: usize,
    ) -> Result<Prediction<f64>, GprError> {
        self.predict_with(sources, q, PredictOptions::default())
    }

    /// Returns [`Self::predict`] with [`PredictOptions`].
    ///
    /// # Errors
    ///
    /// Same as the variant's `predict_with`.
    ///
    /// See the example on [`LoadedDistanceSvgp`].
    pub fn predict_with<'s>(
        &self,
        sources: impl IntoIterator<Item = DistanceSource<'s>>,
        q: usize,
        options: PredictOptions,
    ) -> Result<Prediction<f64>, GprError> {
        each_svgp!(self, model => model.predict_with(sources, q, options).map(widen))
    }
}

impl LoadedDistanceSvgp<WithPoints> {
    /// Returns the number of input features of the coordinate leaves.
    ///
    /// See the example on [`LoadedDistanceSvgp`].
    pub fn d(&self) -> usize {
        each_svgp!(self, model => model.d())
    }

    /// Returns the predictive mean and observation variance at the `q`
    /// queries `xs` (column-major `q × n_cols`); `sources` as
    /// [`LoadedDistanceSgpr::predict`] takes them.
    ///
    /// # Errors
    ///
    /// Same as the variant's `predict`.
    ///
    /// See the example on [`LoadedDistanceSvgp`].
    pub fn predict<'s>(
        &self,
        sources: impl IntoIterator<Item = DistanceSource<'s>>,
        xs: &[f64],
        q: usize,
        n_cols: usize,
    ) -> Result<Prediction<f64>, GprError> {
        self.predict_with(sources, xs, q, n_cols, PredictOptions::default())
    }

    /// Returns [`Self::predict`] with [`PredictOptions`].
    ///
    /// # Errors
    ///
    /// Same as the variant's `predict_with`.
    ///
    /// See the example on [`LoadedDistanceSvgp`].
    pub fn predict_with<'s>(
        &self,
        sources: impl IntoIterator<Item = DistanceSource<'s>>,
        xs: &[f64],
        q: usize,
        n_cols: usize,
        options: PredictOptions,
    ) -> Result<Prediction<f64>, GprError> {
        each_svgp!(self, model => model.predict_with(sources, xs, q, n_cols, options).map(widen))
    }
}
