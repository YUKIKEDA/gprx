//! The supplied squared distances of a saved distance model: the `d2.<k>`
//! tensors (one per slot, in the slot order of the kernel) and the checks
//! that a directory holds the distance model a typed loader asks for.

use std::path::Path;

use safetensors::SafeTensors;

use crate::error::{GprError, PersistErrorKind};
use crate::gpr::{FittedGpr, OnlineGpr};
use crate::kernel::{
    DistanceKernel, DistanceSlot, DistanceSource, KernelScalar, PointUse, SlotShape, SourceStore,
    TrainSources, reads_points,
};
use crate::optimizer::Fixed;
use crate::precision::GpScalar;
use crate::sgpr::{FittedSgpr, FixedInducing};
use crate::svgp::FittedSvgp;

use super::config::ModelJson;
use super::sparse::load_sparse_dir;
use super::tensors::read_f64;
use super::{DistanceLoad, ExactModel, PersistRegistry, load_exact_dir, persist_err};

/// A named `f64` tensor: `(name, shape, column-major values)`.
pub(super) type OwnedTensor = (String, Vec<usize>, Vec<f64>);

/// The tensor name of slot `k`.
fn name(k: usize) -> String {
    format!("d2.{k}")
}

/// The training squares of `store` as tensors: `[n, n]` for a scalar
/// slot, `[dims, n, n]` for an ARD slot. The `f64` values when the store
/// keeps them; otherwise widened to `f64`, which an `f32` model reads back
/// exactly.
pub(super) fn tensors<S: KernelScalar, Store: SourceStore<S>>(store: &Store) -> Vec<OwnedTensor> {
    match store.exact() {
        Some(exact) => dense_tensors(exact),
        None => dense_tensors(store.storage()),
    }
}

fn dense_tensors<T: KernelScalar>(sources: &TrainSources<T>) -> Vec<OwnedTensor> {
    let n = sources.n();
    sources
        .dense_f64()
        .into_iter()
        .enumerate()
        .map(|(k, (shape, blocks))| {
            let dims = match shape {
                SlotShape::Ard(dims) => vec![dims, n, n],
                SlotShape::Scalar => vec![n, n],
            };
            (name(k), dims, blocks.concat())
        })
        .collect()
}

/// The saved training squares of `slots`, as sources of a fit of `n` points.
pub(super) fn read_sources(
    tensors: &SafeTensors<'_>,
    slots: &[DistanceSlot],
    n: usize,
) -> Result<Vec<DistanceSource<'static>>, GprError> {
    slots
        .iter()
        .enumerate()
        .map(|(k, slot)| match slot {
            DistanceSlot::Ard(ard) => {
                let dims = ard.dims();
                let values = read_f64(tensors, &name(k), &[dims, n, n])?;
                let blocks = values.chunks_exact(n * n).map(<[f64]>::to_vec).collect();
                Ok(ard.from_vecs(blocks))
            }
            DistanceSlot::Scalar(scalar) => {
                Ok(scalar.from_vec(read_f64(tensors, &name(k), &[n, n])?))
            }
        })
        .collect()
}

/// `slots` is the kernel of a distance model whose coordinate leaves read
/// `d` features, and `points` says whether the loader's marker reads any.
pub(super) fn check_model(
    slots: &[DistanceSlot],
    d: usize,
    points: bool,
    loader: &str,
) -> Result<(), GprError> {
    if slots.is_empty() {
        return Err(persist_err(
            PersistErrorKind::WrongModel,
            format!("the saved model reads coordinates only; load it with {loader}::load"),
        ));
    }
    if points != (d > 0) {
        let saved = if d > 0 { "WithPoints" } else { "DistanceOnly" };
        return Err(persist_err(
            PersistErrorKind::WrongModel,
            format!("the saved distance kernel is DistanceKernel<{saved}>"),
        ));
    }
    Ok(())
}

/// The [`DistanceLoad`] of marker `C`, naming `coordinate_loader` for a
/// coordinate save.
fn load_of<C: PointUse>(coordinate_loader: &'static str) -> DistanceLoad {
    DistanceLoad {
        points: reads_points::<C>(),
        coordinate_loader,
    }
}

impl<P: GpScalar, C: PointUse> FittedGpr<Fixed, P, DistanceKernel<C>> {
    /// Reads a directory written by [`FittedGpr::save`] or
    /// [`FittedGpr::save_with_factor`] of a model of the same precision and
    /// kernel marker.
    ///
    /// The kernel's slots are new handles: [`Self::slots`] returns them in
    /// the order of [`DistanceKernel::slots`], and the predict calls take
    /// sources of those. The training squares are read back from the file.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::PersistFailed`] with
    /// [`PersistErrorKind::WrongModel`] when the directory holds another
    /// model kind, another precision, a coordinate kernel (load it with
    /// [`crate::LoadedGpr::load`]), the other kernel marker, or an
    /// [`OnlineGpr`]; and the errors of [`crate::LoadedGpr::load`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{DistanceKernel, DistanceOnly, DistanceSlot, RbfKernel, ScalarDistance};
    /// use gprx::{FittedGpr, Fixed, GaussianLikelihood, Gpr, PersistRegistry};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let image = ScalarDistance::new();
    /// let fitted = Gpr::new(image.kernel(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?)
    ///     .with_optimizer(Fixed)
    ///     .factor([image.from_vec(vec![0.0, 1.0, 1.0, 0.0])], 2, &[0.0, 1.0])
    ///     .map_err(|(_, e)| e)?;
    /// let dir = std::env::temp_dir().join(format!("gprx-doctest-dist-{}", std::process::id()));
    /// let _ = std::fs::remove_dir_all(&dir);
    /// fitted.save(&dir)?;
    /// let loaded =
    ///     FittedGpr::<Fixed, gprx::DoublePrecision, DistanceKernel<DistanceOnly>>::load(
    ///         &dir,
    ///         &PersistRegistry::new(),
    ///     )?;
    /// // The loaded kernel has new slot handles.
    /// let [DistanceSlot::Scalar(slot)] = loaded.slots()[..] else {
    ///     panic!("one scalar slot");
    /// };
    /// let cross = [0.25, 0.25];
    /// assert_eq!(
    ///     loaded.predict([slot.borrow(&cross)], 1)?,
    ///     fitted.predict([image.borrow(&cross)], 1)?
    /// );
    /// let _ = std::fs::remove_dir_all(&dir);
    /// # Ok(())
    /// # }
    /// ```
    pub fn load(dir: impl AsRef<Path>, registry: &PersistRegistry) -> Result<Self, GprError> {
        match load_exact_dir::<P, DistanceKernel<C>>(
            dir.as_ref(),
            registry,
            load_of::<C>("LoadedGpr"),
        )? {
            ExactModel::Fitted(model) => Ok(*model),
            ExactModel::Online(_) => Err(persist_err(
                PersistErrorKind::WrongModel,
                "config.json holds an online model; load it with OnlineGpr::load",
            )),
        }
    }
}

impl<P: GpScalar, C: PointUse> OnlineGpr<Fixed, P, DistanceKernel<C>> {
    /// Reads a directory written by [`OnlineGpr::save`] or
    /// [`OnlineGpr::save_with_factor`] of a model of the same precision and
    /// kernel marker, with its point identifiers.
    ///
    /// # Errors
    ///
    /// Same as [`FittedGpr::load`] of a distance model, with a
    /// [`FittedGpr`] directory the wrong model.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{DistanceKernel, DistanceOnly, RbfKernel, ScalarDistance};
    /// use gprx::{Fixed, GaussianLikelihood, Gpr, OnlineGpr, PersistRegistry, SinglePrecision};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let image = ScalarDistance::new();
    /// let online = Gpr::new(image.kernel(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?)
    ///     .with_optimizer(Fixed)
    ///     .with_precision::<SinglePrecision>()
    ///     .factor([image.from_vec(vec![0.0, 1.0, 1.0, 0.0])], 2, &[0.0, 1.0])
    ///     .map_err(|(_, e)| e)?
    ///     .into_online()?;
    /// let dir = std::env::temp_dir().join(format!("gprx-doctest-dist-online-{}", std::process::id()));
    /// let _ = std::fs::remove_dir_all(&dir);
    /// online.save_with_factor(&dir)?;
    /// let loaded = OnlineGpr::<Fixed, SinglePrecision, DistanceKernel<DistanceOnly>>::load(
    ///     &dir,
    ///     &PersistRegistry::new(),
    /// )?;
    /// assert_eq!(loaded.n(), 2);
    /// let _ = std::fs::remove_dir_all(&dir);
    /// # Ok(())
    /// # }
    /// ```
    pub fn load(dir: impl AsRef<Path>, registry: &PersistRegistry) -> Result<Self, GprError> {
        match load_exact_dir::<P, DistanceKernel<C>>(
            dir.as_ref(),
            registry,
            load_of::<C>("LoadedGpr"),
        )? {
            ExactModel::Online(model) => Ok(*model),
            ExactModel::Fitted(_) => Err(persist_err(
                PersistErrorKind::WrongModel,
                "config.json holds a fitted (llt) model; load it with FittedGpr::load",
            )),
        }
    }
}

impl<P: GpScalar, C: PointUse> FittedSgpr<Fixed, FixedInducing, P, DistanceKernel<C>> {
    /// Reads a directory written by [`FittedSgpr::save`] of a model of the
    /// same precision and kernel marker. The VFE system is factored again
    /// at the saved `θ` on the saved training squares and inducing indices.
    ///
    /// # Errors
    ///
    /// Same as [`FittedGpr::load`] of a distance model (a coordinate save
    /// is read with [`crate::LoadedSgpr::load`]), plus the factor errors of
    /// [`crate::Sgpr<Fixed>::factor`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{DistanceKernel, DistanceOnly, RbfKernel, ScalarDistance};
    /// use gprx::{DoublePrecision, FittedSgpr, Fixed, FixedInducing, GaussianLikelihood};
    /// use gprx::{PersistRegistry, Sgpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let image = ScalarDistance::new();
    /// let d2 = vec![0.0, 1.0, 4.0, 1.0, 0.0, 1.0, 4.0, 1.0, 0.0];
    /// let fitted = Sgpr::new(image.kernel(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?)
    ///     .with_optimizer(Fixed)
    ///     .factor([image.from_vec(d2)], 3, &[0.0, 1.0, 0.5], &[0, 2])
    ///     .map_err(|(_, e)| e)?;
    /// let dir = std::env::temp_dir().join(format!("gprx-doctest-dist-sgpr-{}", std::process::id()));
    /// let _ = std::fs::remove_dir_all(&dir);
    /// fitted.save(&dir)?;
    /// let loaded =
    ///     FittedSgpr::<Fixed, FixedInducing, DoublePrecision, DistanceKernel<DistanceOnly>>::load(
    ///         &dir,
    ///         &PersistRegistry::new(),
    ///     )?;
    /// assert_eq!(loaded.inducing(), &[0, 2]);
    /// let _ = std::fs::remove_dir_all(&dir);
    /// # Ok(())
    /// # }
    /// ```
    pub fn load(dir: impl AsRef<Path>, registry: &PersistRegistry) -> Result<Self, GprError> {
        let (_, core) = load_sparse_dir::<P>(
            dir.as_ref(),
            registry,
            ModelJson::Sgpr,
            load_of::<C>("LoadedSgpr"),
        )?;
        Self::from_persisted(core)
    }
}

impl<P: GpScalar, C: PointUse> FittedSvgp<P, DistanceKernel<C>> {
    /// Reads a directory written by [`FittedSvgp::save`] of a model of the
    /// same precision and kernel marker, with the saved whitened `q(u)`.
    ///
    /// # Errors
    ///
    /// Same as [`FittedSgpr::load`] of a distance model (a coordinate save
    /// is read with [`crate::LoadedSvgp::load`]), plus the `q(u)` errors of
    /// [`crate::LoadedSvgp::load`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{DistanceKernel, KernelSpec, RbfKernel, ScalarDistance, WithPoints};
    /// use gprx::{DoublePrecision, FittedSvgp, GaussianLikelihood, PersistRegistry, Svgp};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let image = ScalarDistance::new();
    /// let kernel = image.kernel(RbfKernel::new(1.0)?) * KernelSpec::from(RbfKernel::new(2.0)?);
    /// let d2 = vec![0.0, 1.0, 4.0, 1.0, 0.0, 1.0, 4.0, 1.0, 0.0];
    /// let fitted = Svgp::new(kernel, GaussianLikelihood::new(0.1)?)
    ///     .factor([image.from_vec(d2)], 3, &[0.0, 1.0, 2.0], 1, &[0.0, 1.0, 0.5], &[0, 2])
    ///     .map_err(|(_, e)| e)?;
    /// let dir = std::env::temp_dir().join(format!("gprx-doctest-dist-svgp-{}", std::process::id()));
    /// let _ = std::fs::remove_dir_all(&dir);
    /// fitted.save(&dir)?;
    /// let loaded = FittedSvgp::<DoublePrecision, DistanceKernel<WithPoints>>::load(
    ///     &dir,
    ///     &PersistRegistry::new(),
    /// )?;
    /// assert_eq!(loaded.z(), fitted.z());
    /// let _ = std::fs::remove_dir_all(&dir);
    /// # Ok(())
    /// # }
    /// ```
    pub fn load(dir: impl AsRef<Path>, registry: &PersistRegistry) -> Result<Self, GprError> {
        let dir = dir.as_ref();
        let (q, core) =
            load_sparse_dir::<P>(dir, registry, ModelJson::Svgp, load_of::<C>("LoadedSvgp"))?;
        let (q_mean, q_l) = q.ok_or_else(|| {
            persist_err(
                PersistErrorKind::Tensor,
                "an svgp directory is missing q(u)",
            )
        })?;
        Self::from_persisted(core, q_mean, q_l)
    }
}
