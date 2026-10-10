//! The per-call blocks of a prediction or an online change: the caller's
//! sources bound on a model's [`QueryScratch`], checked or left to be checked
//! as the kernel reads them.

#[allow(
    unused_imports,
    reason = "a split file takes its parent's imports whole; each uses some"
)]
use super::*;

/// A source's `d²`, checked: `shape.blocks()` dense blocks of `rows × cols`.
pub(super) struct RawSlot<'a> {
    /// The slot's place in the kernel's slots, which an error names.
    place: usize,
    pub(crate) shape: SlotShape,
    data: RawData<'a>,
    rows: usize,
    cols: usize,
    /// Where the blocks start in [`QueryScratch`]'s casts (an `f32` model).
    cast_at: usize,
    /// Whether the values are left to be checked as they are read.
    unchecked: bool,
    /// How the source asked its values to be checked.
    tidy: Tidy,
}

pub(super) enum RawData<'a> {
    /// One table (a scalar slot).
    Values(Cow<'a, [f64]>),
    /// One owned table per block.
    Blocks(Vec<Vec<f64>>),
    /// One borrowed table per block.
    Slices(&'a [&'a [f64]]),
    /// Every block in [`QueryScratch`]'s written values from this offset,
    /// one after another: what a fill wrote, or a repaired copy.
    Written(usize),
}

impl RawSlot<'_> {
    /// Block `k` (`rows × cols`, column-major); `written` is the call's
    /// [`QueryScratch`] values.
    pub(crate) fn block<'b>(&'b self, k: usize, written: &'b [f64]) -> &'b [f64] {
        let len = self.rows * self.cols;
        match &self.data {
            RawData::Values(values) => values,
            RawData::Blocks(blocks) => blocks.get(k).map_or(&[], Vec::as_slice),
            RawData::Slices(blocks) => blocks.get(k).copied().unwrap_or(&[]),
            RawData::Written(at) => written.get(at + k * len..at + (k + 1) * len).unwrap_or(&[]),
        }
    }
}

/// Buffers of [`QuerySources`] kept by a model from call to call: the bound
/// slots (empty between calls), what a fill or a repair writes, and the
/// casts an `f32` model reads. A call that fits the capacity of an earlier
/// one allocates nothing.
pub(crate) struct QueryScratch<T> {
    raw: Vec<RawSlot<'static>>,
    written: Vec<f64>,
    column: Vec<f64>,
    pub(super) cast: Vec<T>,
}

impl<T> QueryScratch<T> {
    /// No buffers yet.
    pub(crate) const fn new() -> Self {
        Self {
            raw: Vec::new(),
            written: Vec::new(),
            column: Vec::new(),
            cast: Vec::new(),
        }
    }
}

impl<T> Default for QueryScratch<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// A clone starts without buffers: they hold no state between calls.
impl<T> Clone for QueryScratch<T> {
    fn clone(&self) -> Self {
        Self::new()
    }
}

impl<T> fmt::Debug for QueryScratch<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QueryScratch")
            .field("written", &self.written.capacity())
            .field("cast", &self.cast.capacity())
            .finish_non_exhaustive()
    }
}

/// The empty `raw` of another borrow, on the same allocation: an in-place
/// collect of an empty `Vec` into one of the same layout keeps its buffer.
// `filter_map` changes the element's lifetime, which `filter` cannot.
#[allow(clippy::unnecessary_filter_map)]
pub(super) fn recycle<'x, 'y>(mut raw: Vec<RawSlot<'x>>) -> Vec<RawSlot<'y>> {
    raw.clear();
    raw.into_iter().filter_map(|_| None).collect()
}

/// The checked `d²` blocks of one prediction, bound on a model's
/// [`QueryScratch`]: a caller's table is read in place (an `f64` model) or
/// through one cast (`f32`), and a fill or a repair writes the scratch.
pub(crate) struct QuerySources<'a, T: KernelScalar> {
    /// One entry per slot of the kernel: the scalar slots, then the ARD
    /// slots, each in the kernel's slot order (entry `at` of a shape is the
    /// slot a compiled leaf numbers `at`).
    raw: Vec<RawSlot<'a>>,
    /// Number of scalar slots: where the ARD slots start in `raw`.
    scalars: usize,
    scratch: &'a mut QueryScratch<T>,
}

/// The query squares of a covariance, bound by [`QuerySources::bind_square`]
/// and so checked in full: a Gram of one set reads them where they were
/// bound, in place for an `f64` model, from the one cast otherwise.
pub(crate) struct QuerySquares<'a, T: KernelScalar>(pub(super) QuerySources<'a, T>);

impl<T: KernelScalar> Drop for QuerySources<'_, T> {
    fn drop(&mut self) {
        self.scratch.raw = recycle(std::mem::take(&mut self.raw));
    }
}

impl<'a, T: KernelScalar> QuerySources<'a, T> {
    /// Binds `sources` to `slots` (the kernel's slots) and checks each
    /// block of `rows × cols` as its source asks ([`check_block`]).
    ///
    /// # Errors
    ///
    /// Returns [`GprError::DistanceSlot`] for a source of a slot the kernel
    /// does not read, a slot without a source, or two sources of one slot;
    /// [`GprError::LengthMismatch`] for a block of the wrong length or
    /// count;
    /// [`GprError::EmptyInput`] when `rows` or `cols` is zero;
    /// [`GprError::InvalidDistance`] for a value the source's check refuses.
    pub(crate) fn bind_rect<'s: 'a>(
        slots: &[DistanceSlot],
        sources: impl IntoIterator<Item = DistanceSource<'s>>,
        (rows, cols): (usize, usize),
        scratch: &'a mut QueryScratch<T>,
    ) -> Result<Self, GprError> {
        Self::bind(slots, sources, rows, cols, BlockKind::Rect, true, scratch)
    }

    /// Binds the `n × 1` columns of one new point to the `n` training
    /// points, checked in full when bound (they are kept in the store,
    /// not only read by the kernel).
    ///
    /// # Errors
    ///
    /// The errors of [`Self::bind_rect`].
    pub(crate) fn bind_column<'s: 'a>(
        slots: &[DistanceSlot],
        sources: impl IntoIterator<Item = DistanceSource<'s>>,
        n: usize,
        scratch: &'a mut QueryScratch<T>,
    ) -> Result<Self, GprError> {
        Self::bind(slots, sources, n, 1, BlockKind::Rect, false, scratch)
    }

    /// Binds the `n × n` squares of one set, as [`Self::bind_rect`] binds
    /// blocks, checking each in full ([`BlockKind::Square`]).
    ///
    /// # Errors
    ///
    /// The errors of [`Self::bind_rect`], and [`GprError::InvalidDistance`]
    /// for a square whose diagonal is not zero or that is not symmetric.
    pub(crate) fn bind_square<'s: 'a>(
        slots: &[DistanceSlot],
        sources: impl IntoIterator<Item = DistanceSource<'s>>,
        n: usize,
        scratch: &'a mut QueryScratch<T>,
    ) -> Result<QuerySquares<'a, T>, GprError> {
        Self::bind(slots, sources, n, n, BlockKind::Square, false, scratch).map(QuerySquares)
    }

    /// `defer_check` lets an ARD block of a rectangle be checked as the
    /// kernel reads it rather than when bound.
    pub(super) fn bind<'s: 'a>(
        slots: &[DistanceSlot],
        sources: impl IntoIterator<Item = DistanceSource<'s>>,
        rows: usize,
        cols: usize,
        kind: BlockKind,
        defer_check: bool,
        scratch: &'a mut QueryScratch<T>,
    ) -> Result<Self, GprError> {
        crate::data::require_nonempty(rows)?;
        crate::data::require_nonempty(cols)?;
        let len = rows.checked_mul(cols).ok_or(GprError::SizeOverflow)?;
        if kind == BlockKind::Square && rows != cols {
            return Err(GprError::ShapeMismatch {
                reason: format!("a square of squared distances is {rows}x{cols}"),
            });
        }
        let raw = recycle(std::mem::take(&mut scratch.raw));
        let mut this = Self {
            raw,
            scalars: 0,
            scratch,
        };
        this.scratch.written.clear();
        for source in sources {
            let SlotPlace { slot, place, .. } = slot_of(slots, &source)?;
            if this.raw.iter().any(|raw| raw.place == place) {
                return Err(duplicate(place));
            }
            let shape = slot.shape();
            let blocks = shape.blocks();
            let tidy = source.tidy;
            let QueryScratch {
                written, column, ..
            } = &mut *this.scratch;
            let data = match source.data {
                SourceData::Scalar(ScalarData::Values(values)) => RawData::Values(values),
                SourceData::Ard(_, ArdData::Blocks(tables)) => RawData::Blocks(tables),
                SourceData::Ard(_, ArdData::Slices(tables)) => RawData::Slices(tables),
                SourceData::Scalar(ScalarData::Fill(filler))
                | SourceData::Ard(_, ArdData::Fill(filler)) => {
                    let at = written.len();
                    fill_dense(filler, (rows, cols), blocks, kind, written, column)?;
                    RawData::Written(at)
                }
            };
            let mut raw_slot = RawSlot {
                place,
                shape,
                data,
                rows,
                cols,
                cast_at: 0,
                unchecked: false,
                tidy,
            };
            // An ARD block of pairs of two sets is checked as it is read:
            // an `f64` model's by the kernel ([`crate::kernel::ard::r2_from_blocks`]
            // and the ARD RBF lanes), any other's by the cast below. One
            // pass over the caller's values either way.
            let read_checked = defer_check
                && kind == BlockKind::Rect
                && tidy == Tidy::Exact
                && matches!(shape, SlotShape::Ard(_));
            if read_checked {
                check_counts(&raw_slot, blocks, len, written)?;
                raw_slot.unchecked = true;
            } else {
                check_slot(&mut raw_slot, blocks, len, kind, tidy, written)
                    .map_err(|err| err.in_slot(place))?;
            }
            this.raw.push(raw_slot);
        }
        if this.raw.len() != slots.len() {
            let raw = &this.raw;
            let bound = |place: &usize| raw.iter().any(|raw| raw.place == *place);
            return Err(missing((0..slots.len()).find(|place| !bound(place))));
        }
        // The scalar slots, then the ARD slots, each in the kernel's order.
        this.raw
            .sort_unstable_by_key(|raw| (matches!(raw.shape, SlotShape::Ard(_)), raw.place));
        this.scalars = this
            .raw
            .iter()
            .filter(|raw| raw.shape == SlotShape::Scalar)
            .count();
        if !reads_in_place::<T>() {
            let Self { raw, scratch, .. } = &mut this;
            let QueryScratch { written, cast, .. } = &mut **scratch;
            cast.clear();
            for slot in raw.iter_mut() {
                slot.cast_at = cast.len();
                for k in 0..slot.shape.blocks() {
                    let block = slot.block(k, written);
                    let place = |err| in_block(err, slot.shape, k).in_slot(slot.place);
                    // An unchecked block is checked a tile at a time as it
                    // is cast, while the tile is in cache.
                    for tile in block.chunks(CAST_TILE) {
                        if slot.unchecked && !crate::kernel::simd::all_valid_distances(tile) {
                            return Err(place(first_invalid_from(block, slot.rows, 0)));
                        }
                        let at = cast.len();
                        cast.extend(tile.iter().map(|&v| T::from_f64(v)));
                        if let Some(i) = cast[at..].iter().position(|v| !v.is_finite()) {
                            let pos = (at - slot.cast_at) % len + i;
                            return Err(place(out_of_range(pos % slot.rows, pos / slot.rows)));
                        }
                    }
                }
                // The cast checked every value: an `f64` view of the
                // caller's block (the refinement's) reads it as checked.
                slot.unchecked = false;
            }
        }
        Ok(this)
    }

    /// The checked blocks, by shape in slot order.
    fn blocks(&self) -> BoundBlocks<'_> {
        BoundBlocks {
            raw: &self.raw,
            scalars: self.scalars,
            written: &self.scratch.written,
        }
    }

    /// The same blocks in `f64`, read in place.
    pub(crate) fn f64_view(&self) -> F64Blocks<'_> {
        F64Blocks(self.blocks())
    }

    /// How the source of `block`'s slot asked its values to be checked.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::UnsupportedKernelOperation`] for a slot these
    /// sources do not bind.
    pub(crate) fn tidy(&self, block: BlockAt) -> Result<Tidy, GprError> {
        self.raw_of(block).map(|raw| raw.tidy)
    }

    /// `err` of `block`, named by its slot (its place in the kernel's
    /// slots) and, for an ARD slot, its dimension.
    pub(crate) fn locate(&self, block: BlockAt, err: GprError) -> GprError {
        let err = match block {
            BlockAt::Scalar(_) => err,
            BlockAt::Ard(_, k) => err.in_dim(k),
        };
        match self.raw_of(block) {
            Ok(raw) => err.in_slot(raw.place),
            Err(_) => err,
        }
    }

    /// The bound slot of `block`.
    fn raw_of(&self, block: BlockAt) -> Result<&RawSlot<'a>, GprError> {
        let at = match block {
            BlockAt::Scalar(at) => at,
            BlockAt::Ard(at, _) => self.scalars + at,
        };
        self.raw.get(at).ok_or_else(unbound)
    }
}

impl<T: KernelScalar> RectSlots<T> for QuerySources<'_, T> {
    fn scalar(&self, at: usize) -> Result<MatRef<'_, T>, GprError> {
        self.blocks().scalar(&self.scratch.cast, at)
    }

    fn ard(&self, at: usize) -> Result<ArdRect<'_, T>, GprError> {
        self.blocks().ard(&self.scratch.cast, at)
    }
}

/// A scalar slot is its dense square; an ARD slot its dense blocks, which
/// [`QuerySources::bind_square`] checked in full.
impl<T: KernelScalar> SquareSlots<T> for QuerySquares<'_, T> {
    fn scalar(&self, at: usize) -> Result<MatRef<'_, T>, GprError> {
        self.0.blocks().scalar(&self.0.scratch.cast, at)
    }

    fn ard(&self, at: usize) -> Result<ArdSquare<'_, T>, GprError> {
        let blocks = self.0.blocks();
        let raw = blocks.ard_slot(at)?;
        // A square is checked in full when bound ([`QuerySources::bind_square`]
        // binds no slot to be checked as it is read); one that was not is
        // refused rather than read as checked.
        if raw.unchecked {
            return Err(GprError::UnsupportedKernelOperation {
                reason: "a square of squared distances was bound unchecked".to_owned(),
            });
        }
        let (list, _) = block_list(blocks, &self.0.scratch.cast, raw);
        Ok(ArdSquare::Dense(ArdBlocks::new(
            list, raw.rows, raw.cols, 0,
        )))
    }
}

/// The checked blocks of a [`QuerySources`]: the scalar slots, then the
/// ARD slots, each in slot order.
#[derive(Clone, Copy)]
pub(super) struct BoundBlocks<'v> {
    raw: &'v [RawSlot<'v>],
    scalars: usize,
    written: &'v [f64],
}

impl<'v> BoundBlocks<'v> {
    /// Scalar slot `at` as the scalar `U`.
    fn scalar<U: KernelScalar>(self, cast: &'v [U], at: usize) -> Result<MatRef<'v, U>, GprError> {
        let raw = self
            .raw
            .get(at)
            .filter(|_| at < self.scalars)
            .ok_or_else(unbound)?;
        let (list, _) = block_list(self, cast, raw);
        let checked = ArdBlocks::<U, Checked>::new(list, raw.rows, raw.cols, 0);
        Ok(MatRef::from_column_major_slice(
            &checked.block(0)[..raw.rows * raw.cols],
            raw.rows,
            raw.cols,
        ))
    }

    /// The bound ARD slot `at`.
    fn ard_slot(self, at: usize) -> Result<&'v RawSlot<'v>, GprError> {
        self.raw
            .get(self.scalars.saturating_add(at))
            .ok_or_else(unbound)
    }

    /// ARD slot `at` as the scalar `U`. A caller's block read in place
    /// and not checked when bound is checked as the kernel reads it; a
    /// cast checked it already.
    fn ard<U: KernelScalar>(self, cast: &'v [U], at: usize) -> Result<ArdRect<'v, U>, GprError> {
        let raw = self.ard_slot(at)?;
        let (list, in_place) = block_list(self, cast, raw);
        Ok(if raw.unchecked && in_place {
            ArdRect::Unchecked(ArdBlocks::new(list, raw.rows, raw.cols, 0).of_slot(raw.place))
        } else {
            ArdRect::Checked(ArdBlocks::new(list, raw.rows, raw.cols, 0))
        })
    }
}

/// [`BoundBlocks`] read as `f64` in place.
pub(crate) struct F64Blocks<'v>(BoundBlocks<'v>);

impl RectSlots<f64> for F64Blocks<'_> {
    fn scalar(&self, at: usize) -> Result<MatRef<'_, f64>, GprError> {
        self.0.scalar(&[], at)
    }

    fn ard(&self, at: usize) -> Result<ArdRect<'_, f64>, GprError> {
        self.0.ard(&[], at)
    }
}

/// The blocks of `raw` as the scalar `U`, and whether they are read in
/// place: in place when `U` is `f64`, else from `cast` (what
/// [`QuerySources::bind`] cast).
pub(super) fn block_list<'v, U: KernelScalar>(
    blocks: BoundBlocks<'v>,
    cast: &'v [U],
    raw: &'v RawSlot<'v>,
) -> (BlockList<'v, U>, bool) {
    let len = raw.rows * raw.cols;
    let dims = raw.shape.blocks();
    let in_place = match &raw.data {
        RawData::Values(values) => U::from_f64_slice(values).map(|v| BlockList::Packed(v, len, 1)),
        RawData::Blocks(tables) => U::from_f64_vecs(tables).map(BlockList::Vecs),
        RawData::Slices(tables) => U::from_f64_slices(tables).map(BlockList::Slices),
        RawData::Written(at) => {
            U::from_f64_slice(&blocks.written[*at..]).map(|v| BlockList::Packed(v, len, dims))
        }
    };
    match in_place {
        Some(list) => (list, true),
        None => (BlockList::Packed(&cast[raw.cast_at..], len, dims), false),
    }
}

/// Appends the `blocks` dense `rows × cols` blocks a fill writes to `all`,
/// one after another. A square asks only the lower triangle and mirrors
/// it. `column` is scratch for one column of every block.
pub(super) fn fill_dense(
    filler: &dyn DistanceFill,
    (rows, cols): (usize, usize),
    blocks: usize,
    kind: BlockKind,
    all: &mut Vec<f64>,
    column: &mut Vec<f64>,
) -> Result<(), GprError> {
    let len = rows.checked_mul(cols).ok_or(GprError::SizeOverflow)?;
    let total = len.checked_mul(blocks).ok_or(GprError::SizeOverflow)?;
    let at = all.len();
    all.resize(at.checked_add(total).ok_or(GprError::SizeOverflow)?, 0.0);
    let all = &mut all[at..];
    column.clear();
    column.resize(rows.checked_mul(blocks).ok_or(GprError::SizeOverflow)?, 0.0);
    for col in 0..cols {
        let first = if kind == BlockKind::Square { col } else { 0 };
        let run = rows - first;
        let column = &mut column[..run * blocks];
        filler.fill_column(col, first..rows, column);
        for k in 0..blocks {
            let src = &column[k * run..(k + 1) * run];
            let block = &mut all[k * len..(k + 1) * len];
            block[col * rows + first..(col + 1) * rows].copy_from_slice(src);
        }
    }
    if kind == BlockKind::Square {
        for block in all.chunks_exact_mut(len.max(1)) {
            mirror_lower(block, rows);
        }
    }
    Ok(())
}

/// Checks that `slot` has `blocks` blocks of `len` values each.
pub(super) fn check_counts(
    slot: &RawSlot<'_>,
    blocks: usize,
    len: usize,
    written: &[f64],
) -> Result<(), GprError> {
    let count = match &slot.data {
        RawData::Values(_) => 1,
        RawData::Blocks(tables) => tables.len(),
        RawData::Slices(tables) => tables.len(),
        RawData::Written(_) => blocks,
    };
    require_blocks(
        (0..count).map(|k| slot.block(k, written).len()),
        blocks,
        len,
    )
}

/// Values per tile of the cast of [`QuerySources::bind`]: checked and cast
/// while in cache.
pub(super) const CAST_TILE: usize = 4096;

/// Checks the `blocks` blocks of `slot` (`len` values each) as `tidy`
/// asks. A repair changes an owned table in place; a borrowed one is first
/// copied into `written`.
pub(super) fn check_slot(
    slot: &mut RawSlot<'_>,
    blocks: usize,
    len: usize,
    kind: BlockKind,
    tidy: Tidy,
    written: &mut Vec<f64>,
) -> Result<(), GprError> {
    let (rows, cols) = (slot.rows, slot.cols);
    // Every length first: a repaired copy packs the blocks one after another.
    check_counts(slot, blocks, len, written)?;
    for k in 0..blocks {
        let checked = check_block(slot.block(k, written), rows, cols, kind, tidy);
        if !checked.map_err(|err| in_block(err, slot.shape, k))? {
            continue;
        }
        // An owned table is repaired in place. A borrowed one is copied
        // only when its repair changes it: every block of the slot, one
        // after another, so the slot reads them all from `written`.
        let at = match &mut slot.data {
            RawData::Values(Cow::Owned(values)) => {
                repair_block(values, rows, cols, kind);
                continue;
            }
            RawData::Blocks(tables) => {
                repair_block(&mut tables[k], rows, cols, kind);
                continue;
            }
            RawData::Written(at) => *at,
            RawData::Values(Cow::Borrowed(values)) => {
                let at = written.len();
                written.extend_from_slice(values);
                at
            }
            RawData::Slices(tables) => {
                let at = written.len();
                for table in tables.iter() {
                    written.extend_from_slice(table);
                }
                at
            }
        };
        slot.data = RawData::Written(at);
        repair_block(
            &mut written[at + k * len..at + (k + 1) * len],
            rows,
            cols,
            kind,
        );
    }
    Ok(())
}
