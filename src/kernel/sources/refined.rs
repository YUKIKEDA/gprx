//! The training `d²` of one precision ([`SourceStore`]): the storage
//! scalar's, and for a model that refines in `f64` the caller's values next
//! to it ([`RefinedSources`]).

#[allow(
    unused_imports,
    reason = "a split file takes its parent's imports whole; each uses some"
)]
use super::*;

/// The training `d²` a model of one precision keeps.
///
/// [`TrainSources`] in the storage scalar for a model that factors and
/// predicts in it; [`RefinedSources`] for a model that also refines in
/// `f64` and so keeps the caller's `f64` values next to the storage copy.
pub trait SourceStore<S: KernelScalar>: Clone + fmt::Debug + Send + Sync + 'static {
    /// The scalar a save writes: the storage scalar, or `f64` for a store
    /// that keeps the caller's values.
    type Saved: KernelScalar;

    /// A coordinate model's: no slots.
    fn empty() -> Self;

    /// The copy a save writes, in [`Self::Saved`].
    fn saved(&self) -> &TrainSources<Self::Saved>;

    /// The store of a loaded model from what [`Self::saved`] wrote
    /// ([`TrainSources::from_packed`]).
    ///
    /// # Errors
    ///
    /// As [`TrainSources::from_packed`], and [`GprError::InvalidDistance`]
    /// when a value does not fit the storage scalar.
    fn from_saved(
        slots: &[DistanceSlot],
        values: Vec<Vec<Self::Saved>>,
        n: usize,
    ) -> Result<Self, GprError>;

    /// [`TrainSources::bind`].
    fn bind<'a>(
        slots: &[DistanceSlot],
        sources: impl IntoIterator<Item = DistanceSource<'a>>,
        n: usize,
    ) -> Result<Self, GprError>;

    /// The squares in the storage scalar.
    fn storage(&self) -> &TrainSources<S>;

    /// The squares at `f64` without rounding, when the store keeps them.
    fn exact(&self) -> Option<&TrainSources<f64>>;

    /// The squares at `f64`: [`Self::exact`], or the storage values widened.
    fn to_f64(&self) -> Result<Cow<'_, TrainSources<f64>>, GprError> {
        widened(self.exact(), self.storage())
    }

    /// [`TrainSources::reserve_point`] on every copy the store keeps.
    ///
    /// # Errors
    ///
    /// As [`TrainSources::reserve_point`].
    fn reserve_point(&mut self) -> Result<(), GprError>;

    /// [`TrainSources::check_push`] on every copy the store keeps: `cols`
    /// in the storage scalar, `exact` the same columns at `f64`.
    ///
    /// # Errors
    ///
    /// As [`TrainSources::check_push`].
    fn check_push(
        &self,
        cols: &dyn RectSlots<S>,
        exact: &dyn RectSlots<f64>,
    ) -> Result<(), GprError>;

    /// [`TrainSources::write_point`] on every copy, once
    /// [`Self::check_push`] passed on the same columns. Nothing in it fails,
    /// so every copy is written or none.
    fn write_point(&mut self, cols: &dyn RectSlots<S>, exact: &dyn RectSlots<f64>);

    /// About how many values [`Self::remove_point`] moves for `index`.
    fn remove_work(&self, index: usize) -> usize;

    /// Lays every copy out for an insert or a delete
    /// ([`TrainSources::ready_to_change`]), so neither fails after.
    ///
    /// # Errors
    ///
    /// As [`TrainSources::ready_to_change`].
    fn ready_to_change(&mut self) -> Result<(), GprError>;

    /// [`TrainSources::check_remove`] on every copy the store keeps.
    ///
    /// # Errors
    ///
    /// As [`TrainSources::check_remove`].
    fn check_remove(&self, index: usize) -> Result<(), GprError>;

    /// [`TrainSources::remove_point`] on every copy the store keeps, once
    /// [`Self::check_remove`] passed. Nothing in it fails.
    fn remove_point(&mut self, index: usize);
}

/// The squares at `f64`: `exact` when a store keeps them, else `storage`
/// widened. The one place that picks between the two.
pub(crate) fn widened<'a, S: KernelScalar>(
    exact: Option<&'a TrainSources<f64>>,
    storage: &TrainSources<S>,
) -> Result<Cow<'a, TrainSources<f64>>, GprError> {
    match exact {
        Some(exact) => Ok(Cow::Borrowed(exact)),
        None => storage.to_f64().map(Cow::Owned),
    }
}

impl<S: KernelScalar> SourceStore<S> for TrainSources<S> {
    type Saved = S;

    fn empty() -> Self {
        Self::empty()
    }

    fn saved(&self) -> &TrainSources<S> {
        self
    }

    fn from_saved(slots: &[DistanceSlot], values: Vec<Vec<S>>, n: usize) -> Result<Self, GprError> {
        Self::from_packed(slots, values, n)
    }

    fn bind<'a>(
        slots: &[DistanceSlot],
        sources: impl IntoIterator<Item = DistanceSource<'a>>,
        n: usize,
    ) -> Result<Self, GprError> {
        Self::bind(slots, sources, n)
    }

    fn storage(&self) -> &TrainSources<S> {
        self
    }

    fn exact(&self) -> Option<&TrainSources<f64>> {
        (self as &dyn Any).downcast_ref::<TrainSources<f64>>()
    }

    fn reserve_point(&mut self) -> Result<(), GprError> {
        Self::reserve_point(self)
    }

    fn check_push(
        &self,
        cols: &dyn RectSlots<S>,
        _exact: &dyn RectSlots<f64>,
    ) -> Result<(), GprError> {
        Self::check_push(self, cols)
    }

    fn write_point(&mut self, cols: &dyn RectSlots<S>, _exact: &dyn RectSlots<f64>) {
        Self::write_point(self, cols);
    }

    fn check_remove(&self, index: usize) -> Result<(), GprError> {
        Self::check_remove(self, index)
    }

    fn remove_point(&mut self, index: usize) {
        Self::remove_point(self, index);
    }

    fn remove_work(&self, index: usize) -> usize {
        Self::remove_work(self, index)
    }

    fn ready_to_change(&mut self) -> Result<(), GprError> {
        Self::ready_to_change(self)
    }
}

/// The training `d²` of a model that factors in `f32` and refines in
/// `f64`: the caller's values at `f64` and the `f32` copy the factor reads.
#[derive(Clone, Debug, Default)]
pub struct RefinedSources {
    storage: TrainSources<f32>,
    exact: TrainSources<f64>,
}

impl SourceStore<f32> for RefinedSources {
    type Saved = f64;

    fn empty() -> Self {
        Self::default()
    }

    fn saved(&self) -> &TrainSources<f64> {
        &self.exact
    }

    fn from_saved(
        slots: &[DistanceSlot],
        values: Vec<Vec<f64>>,
        n: usize,
    ) -> Result<Self, GprError> {
        let exact = TrainSources::from_packed(slots, values, n)?;
        Ok(Self {
            storage: exact.cast(slots)?,
            exact,
        })
    }

    fn bind<'a>(
        slots: &[DistanceSlot],
        sources: impl IntoIterator<Item = DistanceSource<'a>>,
        n: usize,
    ) -> Result<Self, GprError> {
        let exact = TrainSources::<f64>::bind(slots, sources, n)?;
        Ok(Self {
            storage: exact.cast(slots)?,
            exact,
        })
    }

    fn storage(&self) -> &TrainSources<f32> {
        &self.storage
    }

    fn exact(&self) -> Option<&TrainSources<f64>> {
        Some(&self.exact)
    }

    fn reserve_point(&mut self) -> Result<(), GprError> {
        self.storage.reserve_point()?;
        self.exact.reserve_point()
    }

    fn check_push(
        &self,
        cols: &dyn RectSlots<f32>,
        exact: &dyn RectSlots<f64>,
    ) -> Result<(), GprError> {
        self.exact.check_push(exact)?;
        self.storage.check_push(cols)
    }

    fn write_point(&mut self, cols: &dyn RectSlots<f32>, exact: &dyn RectSlots<f64>) {
        self.exact.write_point(exact);
        self.storage.write_point(cols);
    }

    fn check_remove(&self, index: usize) -> Result<(), GprError> {
        self.storage.check_remove(index)?;
        self.exact.check_remove(index)
    }

    fn remove_point(&mut self, index: usize) {
        self.storage.remove_point(index);
        self.exact.remove_point(index);
    }

    fn remove_work(&self, index: usize) -> usize {
        self.storage
            .remove_work(index)
            .saturating_add(self.exact.remove_work(index))
    }

    fn ready_to_change(&mut self) -> Result<(), GprError> {
        self.storage.ready_to_change()?;
        self.exact.ready_to_change()
    }
}
