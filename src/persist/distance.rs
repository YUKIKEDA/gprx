//! The loaded Exact model of a [`DistanceKernel`]: [`LoadedDistanceGpr`].

use std::path::Path;

use crate::error::GprError;
use crate::gpr::{FittedGpr, OnlineGpr};
use crate::kernel::{
    DistanceKernel, DistanceOnly, DistanceSlot, DistanceSource, ModelKernelParts, PointUse,
    WithPoints,
};
use crate::optimizer::Fixed;
use crate::precision::PersistKind;
use crate::{
    DoublePrecision, MixedPrecision, PredictOptions, Prediction, ReevaluateKernel, SinglePrecision,
};

use super::config::PointsJson;
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
/// use gprx::kernel::{DistanceSlot, RbfKernel, ScalarDistance};
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
/// let loaded = LoadedDistanceGpr::load(&dir, &PersistRegistry::new())?;
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
    /// tensors, or a registry lookup is invalid. A stored `d²` that is not
    /// finite, negative, or a non-zero diagonal is
    /// [`GprError::InvalidDistance`]. Factorization errors use the same
    /// variants as the model's `factor`.
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
    /// Returns the predictive mean and variance at `m` queries (observation
    /// variance), in `f64` whatever the stored precision. `sources` holds
    /// one source per slot of [`Self::slots`]: the `n × m` squared
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
        m: usize,
    ) -> Result<Prediction<f64>, GprError> {
        self.predict_with(sources, m, PredictOptions::default())
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
        m: usize,
        options: PredictOptions,
    ) -> Result<Prediction<f64>, GprError> {
        each_model!(self, model => model.predict_with(sources, m, options).map(widen))
    }
}

impl LoadedDistanceGpr<WithPoints> {
    /// Returns the number of input features of the coordinate leaves.
    ///
    /// See the example on [`LoadedDistanceGpr`].
    pub fn d(&self) -> usize {
        each_model!(self, model => model.d())
    }

    /// Returns the predictive mean and variance at the `m` queries `xs`
    /// (column-major `m × n_cols`), in `f64` whatever the stored precision.
    /// `sources` holds one source per slot of [`Self::slots`]: the `n × m`
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
        m: usize,
        n_cols: usize,
    ) -> Result<Prediction<f64>, GprError> {
        self.predict_with(sources, xs, m, n_cols, PredictOptions::default())
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
        m: usize,
        n_cols: usize,
        options: PredictOptions,
    ) -> Result<Prediction<f64>, GprError> {
        each_model!(self, model => model.predict_with(sources, xs, m, n_cols, options).map(widen))
    }
}
