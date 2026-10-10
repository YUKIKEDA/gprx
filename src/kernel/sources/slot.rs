//! Where a source's slot is among the kernel's slots, and the errors that
//! name it.

#[allow(
    unused_imports,
    reason = "a split file takes its parent's imports whole; each uses some"
)]
use super::*;

/// Where a source's slot is among the kernel's slots.
#[derive(Clone, Copy)]
pub(super) struct SlotPlace<'k> {
    pub(super) slot: &'k DistanceSlot,
    /// Its place in the kernel's slots ([`crate::kernel::DistanceKernel::slots`]),
    /// which an error names.
    pub(super) place: usize,
    /// Its place among the slots of its shape: the entry a compiled leaf
    /// numbers ([`crate::kernel::SuppliedSpec::at`]) and a store keeps it at.
    pub(super) in_shape: usize,
}

impl SlotPlace<'_> {
    pub(super) fn is_ard(self) -> bool {
        matches!(self.slot.shape(), SlotShape::Ard(_))
    }
}

/// Each slot of `slots` (the kernel's), in order: whether it is an ARD
/// slot, and its place among the slots of its shape.
pub(super) fn shape_places(slots: &[DistanceSlot]) -> impl Iterator<Item = (bool, usize)> + '_ {
    let mut seen = [0usize; 2];
    slots.iter().map(move |slot| {
        let ard = matches!(slot.shape(), SlotShape::Ard(_));
        let at = seen[usize::from(ard)];
        seen[usize::from(ard)] += 1;
        (ard, at)
    })
}

/// The place of `slots[place]` among the slots of its shape (`0` past the
/// end).
pub(crate) fn place_in_shape(slots: &[DistanceSlot], place: usize) -> usize {
    shape_places(slots).nth(place).map_or(0, |(_, at)| at)
}

/// The number of ARD slots in `slots`; the others are scalar.
pub(super) fn ard_count(slots: &[DistanceSlot]) -> usize {
    shape_places(slots).filter(|&(ard, _)| ard).count()
}

/// The slot of `slots` (the kernel's) that `source` is for.
///
/// # Errors
///
/// Returns [`GprError::DistanceSlot`] for a source of a slot the kernel
/// does not read.
pub(super) fn slot_of<'k>(
    slots: &'k [DistanceSlot],
    source: &DistanceSource<'_>,
) -> Result<SlotPlace<'k>, GprError> {
    let Some(place) = slots.iter().position(|slot| slot.id() == source.slot) else {
        return Err(GprError::DistanceSlot {
            kind: SlotErrorKind::NotRead,
            slot: None,
        });
    };
    let slot = &slots[place];
    // A source is made by the slot it names (`ScalarDistance`,
    // `ArdDistance`), which gives it data of its own shape, so the two
    // cannot disagree.
    debug_assert_eq!(slot.shape(), source.data.shape());
    Ok(SlotPlace {
        slot,
        place,
        in_shape: place_in_shape(slots, place),
    })
}

/// The error of a second source of the slot at `place`.
pub(super) fn duplicate(place: usize) -> GprError {
    GprError::DistanceSlot {
        kind: SlotErrorKind::Duplicate,
        slot: Some(place),
    }
}

/// The error of a slot without a source, at `place` in the kernel's slots.
pub(super) fn missing(place: Option<usize>) -> GprError {
    GprError::DistanceSlot {
        kind: SlotErrorKind::Missing,
        slot: place,
    }
}

/// `err` of block `k` of a slot of shape `shape`: the dimension of an ARD
/// slot is named; a scalar slot has one block and no dimension.
pub(super) fn in_block(err: GprError, shape: SlotShape, k: usize) -> GprError {
    match shape {
        SlotShape::Ard(_) => err.in_dim(k),
        SlotShape::Scalar => err,
    }
}

/// `err` of the block `at` of a store in the order of `slots` (the
/// kernel's): its ARD dimension, and its slot when `slots` has it.
pub(super) fn locate_block(slots: &[DistanceSlot], at: BlockAt, err: GprError) -> GprError {
    let (ard, nth, err) = match at {
        BlockAt::Scalar(nth) => (false, nth, err),
        BlockAt::Ard(nth, k) => (true, nth, err.in_dim(k)),
    };
    match shape_places(slots).position(|place| place == (ard, nth)) {
        Some(place) => err.in_slot(place),
        None => err,
    }
}

/// `err` in the slot `id` of `slots` (the kernel's), when `slots` has it.
pub(super) fn in_slot_of(err: GprError, slots: &[DistanceSlot], id: SlotId) -> GprError {
    match slots.iter().position(|slot| slot.id() == id) {
        Some(place) => err.in_slot(place),
        None => err,
    }
}

/// A store changed before it was laid out for the change
/// ([`TrainSources::reserve_point`], [`TrainSources::ready_to_change`]):
/// refused before anything is written.
pub(super) fn no_room() -> GprError {
    GprError::UnsupportedKernelOperation {
        reason: "the training store was not laid out for the change".to_owned(),
    }
}
