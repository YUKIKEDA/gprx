//! The `n × m` training blocks and `m × m` squares of a sparse model
//! ([`BlockStore`]): bound from the caller's sources, and edited as an online
//! model inserts and deletes points and inducing points.

#[allow(
    unused_imports,
    reason = "a split file takes its parent's imports whole; each uses some"
)]
use super::*;

/// Training `d²` a sparse model keeps: per slot, a checked column-major
/// `rows × cols` block, one per dimension of an ARD slot. The blocks from
/// the training points (rows) to the inducing points (columns), as the
/// caller laid them out, or the squares among the inducing points. Empty
/// for a coordinate kernel.
///
/// Column `c` of a block starts at `c · ld`, and a block has room for
/// `cap` columns. A store is bound with `ld = rows` and `cap = cols`; an
/// online model's changes make room for more rows ([`Self::reserve_row`])
/// or columns ([`Self::reserve_col`]), so most of them write in place. The
/// values past `rows` in a column, and past `cols` in a block, are zero.
#[derive(Clone, Debug)]
pub(crate) struct BlockStore<T> {
    rows: usize,
    cols: usize,
    ld: usize,
    cap: usize,
    /// Values from the start of one dimension of a packed ARD slot to the
    /// next: `ld · cap`, or a little more once laid out again, so the `d`
    /// blocks a kernel reads side by side do not start at addresses that
    /// share their cache sets.
    stride: usize,
    scalar: Vec<Vec<T>>,
    ard: Vec<SlotBlocks<T>>,
}

/// The `d` blocks of an ARD slot of a [`BlockStore`].
#[derive(Clone, Debug)]
pub(super) enum SlotBlocks<T> {
    /// One after another in one buffer, `stride` values apart.
    Flat(Vec<T>),
    /// The tables a caller moved in, kept as they are.
    Tables(Vec<Vec<T>>),
}

impl<T> SlotBlocks<T> {
    /// Block `k`, from its start to the next block's.
    fn block(&self, k: usize, stride: usize) -> &[T] {
        match self {
            Self::Flat(all) => &all[k * stride..(k + 1) * stride],
            Self::Tables(tables) => &tables[k],
        }
    }

    /// Number of blocks `stride` values apart.
    fn count(&self, stride: usize) -> usize {
        match self {
            Self::Flat(all) => all.len().checked_div(stride).unwrap_or(0),
            Self::Tables(tables) => tables.len(),
        }
    }

    /// The blocks as a kernel reads them.
    fn list(&self, stride: usize) -> BlockList<'_, T> {
        match self {
            Self::Flat(all) => BlockList::Packed(all, stride, self.count(stride)),
            Self::Tables(tables) => BlockList::Vecs(tables),
        }
    }
}

impl<T> Default for BlockStore<T> {
    fn default() -> Self {
        Self {
            rows: 0,
            cols: 0,
            ld: 0,
            cap: 0,
            stride: 0,
            scalar: Vec::new(),
            ard: Vec::new(),
        }
    }
}

/// One block of a [`BlockStore`]: a scalar slot's, or dimension `k` of an
/// ARD slot's (`Ard(at, k)`), each slot numbered within its shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BlockAt {
    Scalar(usize),
    Ard(usize, usize),
}

impl<T: KernelScalar> BlockStore<T> {
    /// A store of `rows × cols` blocks laid out back to back.
    pub(super) fn packed(
        rows: usize,
        cols: usize,
        scalar: Vec<Vec<T>>,
        ard: Vec<SlotBlocks<T>>,
    ) -> Self {
        Self {
            rows,
            cols,
            ld: rows,
            cap: cols,
            stride: rows * cols,
            scalar,
            ard,
        }
    }

    /// Runs `f` on every block, in slot order.
    fn each_block_mut(&mut self, mut f: impl FnMut(BlockAt, &mut [T])) {
        let stride = self.stride;
        for (at, block) in self.scalar.iter_mut().enumerate() {
            f(BlockAt::Scalar(at), block);
        }
        for (at, slot) in self.ard.iter_mut().enumerate() {
            match slot {
                SlotBlocks::Flat(all) => {
                    for (k, block) in all.chunks_exact_mut(stride.max(1)).enumerate() {
                        f(BlockAt::Ard(at, k), block);
                    }
                }
                SlotBlocks::Tables(tables) => {
                    for (k, block) in tables.iter_mut().enumerate() {
                        f(BlockAt::Ard(at, k), block);
                    }
                }
            }
        }
    }

    /// Appends the values of block `at` to `out`: `rows × cols`,
    /// column-major, without the room past them.
    pub(crate) fn block_into(&self, at: BlockAt, out: &mut Vec<T>) {
        let (rows, cols, ld, stride) = (self.rows, self.cols, self.ld, self.stride);
        let block: &[T] = match at {
            BlockAt::Scalar(place) => &self.scalar[place],
            BlockAt::Ard(place, k) => self.ard[place].block(k, stride),
        };
        for c in 0..cols {
            out.extend_from_slice(&block[c * ld..c * ld + rows]);
        }
    }

    /// The same blocks with column stride `ld` and room for `cap` columns,
    /// `cols` of them: column `c` holds the rows of this store's column
    /// `from(c)`, or zeros.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::SizeOverflow`] if the new blocks cannot be
    /// allocated.
    fn relaid(
        &self,
        ld: usize,
        cap: usize,
        cols: usize,
        from: impl Fn(usize) -> Option<usize>,
    ) -> Result<Self, GprError> {
        let len = ld.checked_mul(cap).ok_or(GprError::SizeOverflow)?;
        // A whole number of 4 KiB pages and one 64-byte cache line more: the
        // blocks of one slot start a line apart within a page.
        let size = std::mem::size_of::<T>().max(1);
        let (page, line) = (4096 / size, 64 / size);
        let stride = len
            .div_ceil(page)
            .checked_mul(page)
            .and_then(|v| v.checked_add(line))
            .ok_or(GprError::SizeOverflow)?;
        let (rows, old_ld, old_stride) = (self.rows, self.ld, self.stride);
        let zero = T::from_f64(0.0);
        let copy = |block: &[T], out: &mut Vec<T>, extent: usize| {
            let start = out.len();
            for c in 0..cols {
                match from(c) {
                    Some(old) => out.extend_from_slice(&block[old * old_ld..old * old_ld + rows]),
                    None => out.resize(out.len() + rows, zero),
                }
                out.resize(out.len() + (ld - rows), zero);
            }
            out.resize(start + extent, zero);
        };
        let fresh = |count: usize| -> Result<Vec<T>, GprError> {
            let mut out = Vec::new();
            out.try_reserve_exact(count)
                .map_err(|_| GprError::SizeOverflow)?;
            Ok(out)
        };
        let mut scalar = Vec::with_capacity(self.scalar.len());
        for block in &self.scalar {
            let mut out = fresh(len)?;
            copy(block, &mut out, len);
            scalar.push(out);
        }
        let mut ard = Vec::with_capacity(self.ard.len());
        for slot in &self.ard {
            let dims = slot.count(old_stride);
            let mut out = fresh(stride.checked_mul(dims).ok_or(GprError::SizeOverflow)?)?;
            for k in 0..dims {
                copy(slot.block(k, old_stride), &mut out, stride);
            }
            ard.push(SlotBlocks::Flat(out));
        }
        Ok(Self {
            rows,
            cols,
            ld,
            cap,
            stride,
            scalar,
            ard,
        })
    }

    /// Makes room for one more row, so [`Self::push_row`] writes in place:
    /// a full store is laid out again with a quarter more rows, copying its
    /// values once per `rows / 4` inserts.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::SizeOverflow`] if the larger blocks cannot be
    /// allocated; the store is then unchanged.
    pub(crate) fn reserve_row(&mut self) -> Result<(), GprError> {
        if self.rows < self.ld {
            return Ok(());
        }
        let ld = (self.rows + 1).max(self.rows + self.rows / 4);
        *self = self.relaid(ld, self.cap, self.cols, Some)?;
        Ok(())
    }

    /// Appends row `rows`, `value(b, col)` in each column of block `b`
    /// (in [`Self::block_ids`] order), once [`Self::reserve_row`] made
    /// room: nothing in it fails or allocates.
    pub(crate) fn push_row(&mut self, mut value: impl FnMut(usize, usize) -> T) {
        debug_assert!(self.rows < self.ld, "push_row before reserve_row");
        let (row, ld, cols) = (self.rows, self.ld, self.cols);
        let mut b = 0;
        self.each_block_mut(|_, block| {
            for c in 0..cols {
                block[row + c * ld] = value(b, c);
            }
            b += 1;
        });
        self.rows += 1;
    }

    /// Removes the last row (the undo of [`Self::push_row`]).
    pub(crate) fn pop_row(&mut self) {
        debug_assert!(self.rows > 0, "pop_row of no rows");
        let (row, ld, cols) = (self.rows - 1, self.ld, self.cols);
        self.each_block_mut(|_, block| {
            for c in 0..cols {
                block[row + c * ld] = T::from_f64(0.0);
            }
        });
        self.rows = row;
    }

    /// Removes row `index` (`< rows`), moving the rows after it up by one in
    /// every column: `(rows − index) · cols` values per block.
    pub(crate) fn remove_row(&mut self, index: usize) {
        debug_assert!(index < self.rows, "remove_row past the rows");
        let (rows, ld, cols) = (self.rows, self.ld, self.cols);
        self.each_block_mut(|_, block| {
            for c in 0..cols {
                let start = c * ld;
                block.copy_within(start + index + 1..start + rows, start + index);
                block[start + rows - 1] = T::from_f64(0.0);
            }
        });
        self.rows -= 1;
    }

    /// Appends row `index`'s values, block by block in [`Self::block_ids`]
    /// order (`cols` each), to `out`.
    pub(crate) fn row_into(&self, index: usize, out: &mut Vec<f64>) {
        let (ld, cols, stride) = (self.ld, self.cols, self.stride);
        let mut push = |block: &[T]| {
            out.extend((0..cols).map(|c| block[index + c * ld].to_f64()));
        };
        for block in &self.scalar {
            push(block);
        }
        for slot in &self.ard {
            for k in 0..slot.count(stride) {
                push(slot.block(k, stride));
            }
        }
    }

    /// Puts back row `index` with `saved` ([`Self::row_into`]), moving the
    /// rows from `index` down by one: the undo of [`Self::remove_row`],
    /// which left room for it.
    pub(crate) fn insert_row(&mut self, index: usize, saved: &[f64]) {
        debug_assert!(self.rows < self.ld, "insert_row without room");
        let (rows, ld, cols) = (self.rows, self.ld, self.cols);
        let mut at = 0;
        self.each_block_mut(|_, block| {
            for c in 0..cols {
                let start = c * ld;
                block.copy_within(start + index..start + rows, start + index + 1);
                block[start + index] = T::from_f64(saved[at + c]);
            }
            at += cols;
        });
        self.rows += 1;
    }

    /// Makes room for one more column, so [`Self::push_col`] writes in
    /// place: a full store is laid out again with a quarter more columns.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::SizeOverflow`] if the wider blocks cannot be
    /// allocated; the store is then unchanged.
    pub(crate) fn reserve_col(&mut self) -> Result<(), GprError> {
        if self.cols < self.cap {
            return Ok(());
        }
        let cap = (self.cols + 1).max(self.cols + self.cols / 4);
        *self = self.relaid(self.ld, cap, self.cols, Some)?;
        Ok(())
    }

    /// Appends a column, `value(b, row)` in each row of block `b` (in
    /// [`Self::block_ids`] order), once [`Self::reserve_col`] made room:
    /// nothing in it fails or allocates.
    pub(crate) fn push_col(&mut self, mut value: impl FnMut(usize, usize) -> T) {
        debug_assert!(self.cols < self.cap, "push_col before reserve_col");
        let (rows, ld, col) = (self.rows, self.ld, self.cols);
        let mut b = 0;
        self.each_block_mut(|_, block| {
            for (row, slot) in block[col * ld..col * ld + rows].iter_mut().enumerate() {
                *slot = value(b, row);
            }
            b += 1;
        });
        self.cols += 1;
    }

    /// Removes the last column (the undo of [`Self::push_col`]).
    pub(crate) fn pop_col(&mut self) {
        let (ld, col) = (self.ld, self.cols - 1);
        self.each_block_mut(|_, block| block[col * ld..(col + 1) * ld].fill(T::from_f64(0.0)));
        self.cols = col;
    }

    /// Removes column `index` in place; with `saved`, its `rows` values are
    /// appended there block by block for [`Self::insert_col`].
    pub(crate) fn remove_col(&mut self, index: usize, mut saved: Option<&mut Vec<f64>>) {
        debug_assert!(index < self.cols, "remove_col past the columns");
        let (rows, ld, cols) = (self.rows, self.ld, self.cols);
        self.each_block_mut(|_, block| {
            let start = index * ld;
            if let Some(saved) = saved.as_deref_mut() {
                saved.extend(block[start..start + rows].iter().map(|v| v.to_f64()));
            }
            block.copy_within(start + ld..cols * ld, start);
            block[(cols - 1) * ld..cols * ld].fill(T::from_f64(0.0));
        });
        self.cols -= 1;
    }

    /// Puts back column `index` with `saved` ([`Self::remove_col`]), in the
    /// room the removal left: nothing allocates.
    pub(crate) fn insert_col(&mut self, index: usize, saved: &[f64]) {
        debug_assert!(self.cols < self.cap, "insert_col without room");
        let (rows, ld, cols) = (self.rows, self.ld, self.cols);
        let mut at = 0;
        self.each_block_mut(|_, block| {
            let start = index * ld;
            block.copy_within(start..cols * ld, start + ld);
            for (slot, &v) in block[start..start + rows]
                .iter_mut()
                .zip(&saved[at..at + rows])
            {
                *slot = T::from_f64(v);
            }
            block[start + rows..start + ld].fill(T::from_f64(0.0));
            at += rows;
        });
        self.cols += 1;
    }

    /// The value of `block` at `(row, col)`.
    pub(crate) fn get(&self, at: BlockAt, row: usize, col: usize) -> T {
        let i = row + col * self.ld;
        match at {
            BlockAt::Scalar(slot) => self.scalar[slot][i],
            BlockAt::Ard(slot, k) => self.ard[slot].block(k, self.stride)[i],
        }
    }

    /// Sets the value of `block` at `(row, col)`.
    pub(crate) fn set(&mut self, at: BlockAt, row: usize, col: usize, v: T) {
        let (i, stride) = (row + col * self.ld, self.stride);
        match at {
            BlockAt::Scalar(slot) => self.scalar[slot][i] = v,
            BlockAt::Ard(slot, k) => match &mut self.ard[slot] {
                SlotBlocks::Flat(all) => all[k * stride + i] = v,
                SlotBlocks::Tables(tables) => tables[k][i] = v,
            },
        }
    }

    /// Every block, as [`Self::push_row`] and [`Self::push_col`] name them.
    pub(crate) fn block_ids(&self) -> Vec<BlockAt> {
        self.block_at().collect()
    }

    /// [`Self::block_ids`] without collecting them.
    fn block_at(&self) -> impl Iterator<Item = BlockAt> + '_ {
        let stride = self.stride;
        (0..self.scalar.len()).map(BlockAt::Scalar).chain(
            self.ard.iter().enumerate().flat_map(move |(at, slot)| {
                (0..slot.count(stride)).map(move |k| BlockAt::Ard(at, k))
            }),
        )
    }

    /// Every value `(block, row, col)` in [`Self::block_ids`] order, with
    /// the shape: what a change and its undo must give back.
    #[cfg(test)]
    pub(crate) fn values(&self) -> (usize, usize, Vec<f64>) {
        let mut out = Vec::new();
        for at in self.block_ids() {
            for col in 0..self.cols {
                for row in 0..self.rows {
                    out.push(self.get(at, row, col).to_f64());
                }
            }
        }
        (self.rows, self.cols, out)
    }

    /// Every block's buffer, in slot order.
    fn blocks(&self) -> impl Iterator<Item = &[T]> {
        let stride = self.stride;
        self.scalar.iter().map(Vec::as_slice).chain(
            self.ard
                .iter()
                .flat_map(move |slot| (0..slot.count(stride)).map(move |k| slot.block(k, stride))),
        )
    }

    /// The same blocks at the scalar `U`: checked when they were bound, so
    /// a value past the range of `U` is reported where it is.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidDistance`] for a value that is not
    /// finite at `U`.
    pub(crate) fn cast<U: KernelScalar>(&self) -> Result<BlockStore<U>, GprError> {
        self.require_in_range::<U>(&[])?;
        let stride = self.stride;
        let cast =
            |block: &[T]| -> Vec<U> { block.iter().map(|v| U::from_f64(v.to_f64())).collect() };
        Ok(BlockStore {
            rows: self.rows,
            cols: self.cols,
            ld: self.ld,
            cap: self.cap,
            stride,
            scalar: self.scalar.iter().map(|b| cast(b)).collect(),
            ard: self
                .ard
                .iter()
                .map(|slot| {
                    SlotBlocks::Flat(
                        (0..slot.count(stride))
                            .flat_map(|k| cast(slot.block(k, stride)))
                            .collect(),
                    )
                })
                .collect(),
        })
    }

    /// Checks that every value is finite at the scalar `U`, without a copy.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidDistance`] at the first value past the
    /// range of `U`, located in its block and, when `slots` (the kernel's)
    /// has it, in its slot.
    pub(crate) fn require_in_range<U: KernelScalar>(
        &self,
        slots: &[DistanceSlot],
    ) -> Result<(), GprError> {
        if !U::ROUNDS_FROM_F64 {
            return Ok(());
        }
        let ld = self.ld.max(1);
        for (at, block) in self.block_at().zip(self.blocks()) {
            if let Some(i) = block
                .iter()
                .position(|v| !U::from_f64(v.to_f64()).to_f64().is_finite())
            {
                return Err(locate_block(slots, at, out_of_range(i % ld, i / ld)));
            }
        }
        Ok(())
    }

    /// Rows `rows` of every block (a minibatch, or the inducing rows) into
    /// `out`, laid out back to back and reusing its buffers: once `out` has
    /// held a batch this large, nothing allocates.
    pub(crate) fn rows_into(&self, rows: &[usize], out: &mut Self) {
        let (ld, cols, stride) = (self.ld, self.cols, self.stride);
        out.rows = rows.len();
        out.cols = cols;
        out.ld = rows.len();
        out.cap = cols;
        out.stride = rows.len() * cols;
        // Column by column of every block: the picked blocks lie one after
        // another.
        let pick = |block: &[T], picked: &mut Vec<T>| {
            for c in 0..cols {
                let column = &block[c * ld..];
                picked.extend(rows.iter().map(|&i| column[i]));
            }
        };
        out.scalar.resize_with(self.scalar.len(), Vec::new);
        for (block, picked) in self.scalar.iter().zip(&mut out.scalar) {
            picked.clear();
            pick(block, picked);
        }
        out.ard
            .resize_with(self.ard.len(), || SlotBlocks::Flat(Vec::new()));
        for (slot, picked) in self.ard.iter().zip(&mut out.ard) {
            if !matches!(picked, SlotBlocks::Flat(_)) {
                *picked = SlotBlocks::Flat(Vec::new());
            }
            if let SlotBlocks::Flat(picked) = picked {
                picked.clear();
                for k in 0..slot.count(stride) {
                    pick(slot.block(k, stride), picked);
                }
            }
        }
    }
}

/// The squares read in place (a store of `rows = cols`, checked as
/// squares when bound, one buffer per ARD slot).
impl<T: KernelScalar> SquareSlots<T> for BlockStore<T> {
    fn scalar(&self, at: usize) -> Result<MatRef<'_, T>, GprError> {
        RectSlots::scalar(self, at)
    }

    fn ard(&self, at: usize) -> Result<ArdSquare<'_, T>, GprError> {
        debug_assert_eq!(
            (self.ld, self.stride),
            (self.rows, self.rows * self.cols),
            "a square store is laid out back to back"
        );
        match self.ard.get(at).ok_or_else(unbound)? {
            SlotBlocks::Flat(all) => Ok(ArdSquare::Packed(ArdSqDiff::flat(all, self.rows))),
            SlotBlocks::Tables(_) => Ok(ArdSquare::Dense(ArdBlocks::new(
                self.ard[at].list(self.stride),
                self.rows,
                self.cols,
                0,
            ))),
        }
    }
}

/// The blocks read in place.
impl<T: KernelScalar> RectSlots<T> for BlockStore<T> {
    fn scalar(&self, at: usize) -> Result<MatRef<'_, T>, GprError> {
        let block = self.scalar.get(at).ok_or_else(unbound)?;
        Ok(MatRef::from_column_major_slice_with_stride(
            block, self.rows, self.cols, self.ld,
        ))
    }

    fn ard(&self, at: usize) -> Result<ArdRect<'_, T>, GprError> {
        let slot = self.ard.get(at).ok_or_else(unbound)?;
        Ok(ArdRect::Checked(ArdBlocks::strided(
            slot.list(self.stride),
            self.rows,
            self.cols,
            0,
            self.ld,
        )))
    }
}

/// The training `d²` of a sparse model on supplied distances: `sources`
/// bind each slot's `n × m` block from the `n` training points to the `m`
/// inducing points (training points `inducing`, in that order), checked as
/// a training block is. Returns the `m × m` squares among the inducing
/// points (their rows of the block: checked, with the source's repair, as
/// a training square is) and the `n × m` blocks themselves. A block the
/// caller moved in is kept as it is; a borrowed one is copied once; a fill
/// is written once.
///
/// # Errors
///
/// Returns [`GprError::EmptyInput`] when `n` or `inducing` is empty,
/// [`GprError::DistanceSlot`] for a source of a slot the kernel does not
/// read, a slot without a source, or two sources of one slot,
/// [`GprError::LengthMismatch`] for a block of the wrong length or count, and [`GprError::InvalidDistance`] for a value
/// the source's check refuses, or inducing rows that are not a square with
/// a zero diagonal and equal mirror entries (located in the caller's
/// block).
pub(crate) fn bind_inducing<'s>(
    slots: &[DistanceSlot],
    sources: impl IntoIterator<Item = DistanceSource<'s>>,
    n: usize,
    inducing: &[usize],
) -> Result<(BlockStore<f64>, BlockStore<f64>), GprError> {
    let m = inducing.len();
    crate::data::require_nonempty(n)?;
    crate::data::require_nonempty(m)?;
    let ards = ard_count(slots);
    let store = |rows, cols| {
        BlockStore::packed(
            rows,
            cols,
            vec![Vec::new(); slots.len() - ards],
            (0..ards).map(|_| SlotBlocks::Flat(Vec::new())).collect(),
        )
    };
    let (mut zz, mut xz) = (store(m, m), store(n, m));
    let mut column = Vec::new();
    // Each store in the kernel's slot order: a slot's entry is its place
    // among the slots of its shape; an entry not yet bound is empty.
    let is_bound = |xz: &BlockStore<f64>, ard: bool, at: usize| {
        if ard {
            !matches!(&xz.ard[at], SlotBlocks::Flat(all) if all.is_empty())
        } else {
            !xz.scalar[at].is_empty()
        }
    };
    for source in sources {
        let at = slot_of(slots, &source)?;
        if is_bound(&xz, at.is_ard(), at.in_shape) {
            return Err(duplicate(at.place));
        }
        bind_inducing_slot(
            source,
            (n, m),
            inducing,
            &mut column,
            (&mut xz, &mut zz),
            at.in_shape,
        )
        .map_err(|err| err.in_slot(at.place))?;
    }
    let unbound_place = shape_places(slots).position(|(ard, at)| !is_bound(&xz, ard, at));
    if unbound_place.is_some() {
        return Err(missing(unbound_place));
    }
    Ok((zz, xz))
}

/// Binds the `n × m` block `source` of the slot at `place` among the slots
/// of its shape into `xz`, and the `m × m` square of its inducing rows into
/// `zz` ([`bind_inducing`]).
pub(super) fn bind_inducing_slot(
    source: DistanceSource<'_>,
    (n, m): (usize, usize),
    inducing: &[usize],
    column: &mut Vec<f64>,
    (xz, zz): (&mut BlockStore<f64>, &mut BlockStore<f64>),
    place: usize,
) -> Result<(), GprError> {
    let len = n.checked_mul(m).ok_or(GprError::SizeOverflow)?;
    let square_len = m.checked_mul(m).ok_or(GprError::SizeOverflow)?;
    {
        let tidy = source.tidy;
        match source.data {
            SourceData::Scalar(data) => {
                let mut block = match data {
                    ScalarData::Values(values) => {
                        crate::data::require_count(values.len(), len, "squared distances")?;
                        values
                    }
                    ScalarData::Fill(filler) => {
                        let mut filled = Vec::new();
                        fill_dense(filler, (n, m), 1, BlockKind::Rect, &mut filled, column)?;
                        Cow::Owned(filled)
                    }
                };
                let mut square = vec![0.0; square_len];
                inducing_parts(&mut block, n, inducing, tidy, &mut square)?;
                xz.scalar[place] = block.into_owned();
                zz.scalar[place] = square;
            }
            SourceData::Ard(d, data) => {
                let mut squares =
                    vec![0.0; square_len.checked_mul(d).ok_or(GprError::SizeOverflow)?];
                require_ard_data(&data, d, len)?;
                let blocks = match data {
                    ArdData::Blocks(mut tables) => {
                        for (k, table) in tables.iter_mut().enumerate() {
                            let square = &mut squares[k * square_len..(k + 1) * square_len];
                            let mut block = Cow::Borrowed(table.as_slice());
                            inducing_parts(&mut block, n, inducing, tidy, square)
                                .map_err(|err| err.in_dim(k))?;
                            if let Cow::Owned(repaired) = block {
                                *table = repaired;
                            }
                        }
                        SlotBlocks::Tables(tables)
                    }
                    ArdData::Slices(tables) => {
                        let mut all =
                            Vec::with_capacity(len.checked_mul(d).ok_or(GprError::SizeOverflow)?);
                        for (k, table) in tables.iter().enumerate() {
                            let square = &mut squares[k * square_len..(k + 1) * square_len];
                            let mut block = Cow::Borrowed(*table);
                            inducing_parts(&mut block, n, inducing, tidy, square)
                                .map_err(|err| err.in_dim(k))?;
                            all.extend_from_slice(&block);
                        }
                        SlotBlocks::Flat(all)
                    }
                    ArdData::Fill(filler) => {
                        let mut all = Vec::new();
                        fill_dense(filler, (n, m), d, BlockKind::Rect, &mut all, column)?;
                        for k in 0..d {
                            let square = &mut squares[k * square_len..(k + 1) * square_len];
                            let mut block = Cow::Borrowed(&all[k * len..(k + 1) * len]);
                            inducing_parts(&mut block, n, inducing, tidy, square)
                                .map_err(|err| err.in_dim(k))?;
                            if let Cow::Owned(repaired) = block {
                                all[k * len..(k + 1) * len].copy_from_slice(&repaired);
                            }
                        }
                        SlotBlocks::Flat(all)
                    }
                };
                xz.ard[place] = blocks;
                zz.ard[place] = SlotBlocks::Flat(squares);
            }
        }
    }
    Ok(())
}

/// One `n × m` block (training points × inducing points `inducing`),
/// checked as `tidy` asks and repaired where it allows (a borrowed block is
/// copied only to be repaired), and the `m × m` square of its inducing rows
/// written to `square`, checked and repaired as a training square is; the
/// repaired pairs are written back to those rows. A violation of the square
/// is located in `block`: row `inducing[a]`, column `b`.
pub(super) fn inducing_parts(
    block: &mut Cow<'_, [f64]>,
    n: usize,
    inducing: &[usize],
    tidy: Tidy,
    square: &mut [f64],
) -> Result<(), GprError> {
    let m = inducing.len();
    if check_block(block, n, m, BlockKind::Rect, tidy)? {
        repair_block(block.to_mut(), n, m, BlockKind::Rect);
    }
    for (b, column) in square.chunks_exact_mut(m).enumerate() {
        let source = &block[b * n..(b + 1) * n];
        for (v, &i) in column.iter_mut().zip(inducing) {
            *v = source[i];
        }
    }
    match check_block(square, m, m, BlockKind::Square, tidy) {
        Ok(true) => {
            // The repaired pairs go back to the block's inducing rows, so
            // `K_mm` and `K(Z, X)` read the same values.
            repair_block(square, m, m, BlockKind::Square);
            let block = block.to_mut();
            for (b, column) in square.chunks_exact(m).enumerate() {
                for (&v, &i) in column.iter().zip(inducing) {
                    block[i + b * n] = v;
                }
            }
        }
        Ok(false) => {}
        Err(GprError::InvalidDistance {
            slot,
            dim,
            pair,
            reason,
        }) => {
            return Err(GprError::InvalidDistance {
                slot,
                dim,
                pair: pair.map(|(row, col)| (inducing.get(row).copied().unwrap_or(row), col)),
                reason,
            });
        }
        Err(err) => return Err(err),
    }
    Ok(())
}

/// The `rows` values of the bound `rows × 1` column of `block` in `cols`,
/// appended to `out`.
///
/// # Errors
///
/// Returns the error of reading the block: unbound, or an unchecked value
/// that is invalid.
pub(crate) fn column_into(
    cols: &dyn RectSlots<f64>,
    block: BlockAt,
    rows: usize,
    out: &mut Vec<f64>,
) -> Result<(), GprError> {
    fn from_blocks<S: crate::kernel::dist::BlockState>(
        blocks: ArdBlocks<'_, f64, S>,
        k: usize,
        rows: usize,
        out: &mut Vec<f64>,
    ) -> Result<(), GprError> {
        match blocks.column(k, 0)? {
            Some(run) => out.extend_from_slice(run),
            None => {
                for row in 0..rows {
                    out.push(blocks.read(k, row, 0)?);
                }
            }
        }
        Ok(())
    }
    match block {
        BlockAt::Scalar(at) => {
            let column = cols.scalar(at)?;
            out.extend((0..rows).map(|row| column[(row, 0)]));
            Ok(())
        }
        BlockAt::Ard(at, k) => match cols.ard(at)? {
            ArdRect::Checked(blocks) => from_blocks(blocks, k, rows, out),
            ArdRect::Unchecked(blocks) => from_blocks(blocks, k, rows, out),
        },
    }
}

/// The column of training point `point` (`n × 1` per block in `cols`, its
/// squared distances to the `n` training points) once it becomes inducing
/// point `m` of the training blocks `xz` and the squares `zz` of
/// `inducing`. Its `(m + 1)²` square among the inducing points is checked
/// as a training square is: its row of `xz` and the new column's inducing
/// rows are the mirror pairs, and its own row is the diagonal. A tidy
/// source repairs that square as a training square is repaired.
///
/// Returns, block by block in [`BlockStore::block_ids`] order, the new
/// column (`n` values each) and the row of `xz` the point holds after the
/// repair (`m` values each).
///
/// # Errors
///
/// Returns [`GprError::InvalidDistance`] for a square the source's check
/// refuses, located in the caller's column, and the errors of reading
/// `cols`. `locate` names the slot and dimension of a block's error
/// ([`QuerySources::locate`]).
pub(crate) fn new_inducing_column(
    xz: &BlockStore<f64>,
    zz: &BlockStore<f64>,
    inducing: &[usize],
    point: usize,
    cols: &dyn RectSlots<f64>,
    tidy: impl Fn(BlockAt) -> Result<Tidy, GprError>,
    locate: impl Fn(BlockAt, GprError) -> GprError,
) -> Result<(Vec<f64>, Vec<f64>), GprError> {
    let (n, m) = (xz.rows, inducing.len());
    let side = m + 1;
    let blocks = xz.block_ids();
    let mut column = Vec::with_capacity(n * blocks.len());
    let mut mirror = Vec::with_capacity(m * blocks.len());
    let mut square = vec![0.0; side * side];
    for &at in &blocks {
        let start = column.len();
        column_into(cols, at, n, &mut column).map_err(|err| locate(at, err))?;
        let new = &mut column[start..];
        for b in 0..m {
            for a in 0..m {
                square[a + b * side] = zz.get(at, a, b);
            }
            square[m + b * side] = xz.get(at, point, b);
            square[b + m * side] = new[inducing[b]];
        }
        square[m + m * side] = new[point];
        // A violation is located in the caller's column: the row of an
        // inducing point, or of the point itself.
        let row_of = |i: usize| inducing.get(i).copied().unwrap_or(point);
        match check_block(&square, side, side, BlockKind::Square, tidy(at)?) {
            Ok(true) => repair_block(&mut square, side, side, BlockKind::Square),
            Ok(false) => {}
            Err(GprError::InvalidDistance {
                slot,
                dim,
                pair,
                reason,
            }) => {
                let pair = pair.map(|(row, col)| {
                    let row = if row == m { row_of(col) } else { row_of(row) };
                    (row, 0)
                });
                let err = GprError::InvalidDistance {
                    slot,
                    dim,
                    pair,
                    reason,
                };
                return Err(locate(at, err));
            }
            Err(err) => return Err(err),
        }
        for b in 0..m {
            new[inducing[b]] = square[b + m * side];
            mirror.push(square[m + b * side]);
        }
        new[point] = square[m + m * side];
    }
    Ok((column, mirror))
}
