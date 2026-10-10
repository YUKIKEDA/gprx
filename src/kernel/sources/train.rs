//! The training `d²` a model owns ([`TrainSources`]): bound from the
//! caller's sources, kept in the storage scalar, grown and shrunk by an
//! online model, and saved and loaded packed.

#[allow(unused_imports)]
use super::*;

/// The packed training triangles of an ARD slot of `d` dimensions from its
/// `n × n` tables, checked as `tidy` asks.
pub(super) fn train_ard<T: KernelScalar>(
    data: ArdData<'_>,
    n: usize,
    d: usize,
    tidy: Tidy,
) -> Result<ArdSqDiffBuf<T>, GprError> {
    let len = n.checked_mul(n).ok_or(GprError::SizeOverflow)?;
    require_ard_data(&data, d, len)?;
    match data {
        ArdData::Blocks(tables) if tidy == Tidy::Exact => keep_exact_ard(tables, n, d),
        ArdData::Slices(tables) if tidy == Tidy::Exact => pack_exact_ard(n, d, |k| tables[k]),
        ArdData::Blocks(mut tables) => {
            for (k, table) in tables.iter_mut().enumerate() {
                if check_block(table, n, n, BlockKind::Square, tidy).map_err(|err| err.in_dim(k))? {
                    repair_block(table, n, n, BlockKind::Square);
                }
            }
            ArdSqDiffBuf::from_dense(n, d, |k| &tables[k])
        }
        ArdData::Slices(tables) => {
            let mut repaired: Vec<Option<Vec<f64>>> = (0..d).map(|_| None).collect();
            for (k, table) in tables.iter().enumerate() {
                if check_block(table, n, n, BlockKind::Square, tidy).map_err(|err| err.in_dim(k))? {
                    let mut copy = table.to_vec();
                    repair_block(&mut copy, n, n, BlockKind::Square);
                    repaired[k] = Some(copy);
                }
            }
            ArdSqDiffBuf::from_dense(n, d, |k| repaired[k].as_deref().unwrap_or(tables[k]))
        }
        ArdData::Fill(filler) => fill_ard_square(filler, n, d, tidy),
    }
}

/// A scalar training square of `n` points from a fill: the lower triangle,
/// column by column, mirrored into the dense square.
pub(super) fn fill_scalar_square(
    filler: &dyn DistanceFill,
    n: usize,
    tidy: Tidy,
) -> Result<Vec<f64>, GprError> {
    let len = n.checked_mul(n).ok_or(GprError::SizeOverflow)?;
    let mut square = vec![0.0; len];
    let mut run = vec![0.0; n];
    let mut rounding = FillRounding::default();
    for col in 0..n {
        let run = &mut run[..n - col];
        filler.fill_column(col, col..n, run);
        let column = &mut square[col * n + col..(col + 1) * n];
        fill_run(run, col, tidy, &mut rounding, |at, v| column[at] = v)?;
    }
    if let Tidy::Within(rel) = tidy {
        rounding.judge(rel)?;
    }
    mirror_lower(&mut square, n);
    Ok(square)
}

/// Side of the tiles [`mirror_lower`] copies: a source tile and its
/// destination tile fit in L1 together.
pub(super) const MIRROR_TILE: usize = 32;

/// Copies the strict lower triangle of the column-major `n × n` `square`
/// onto its upper triangle, a tile at a time: each destination row of a
/// tile is written contiguously while the tile's source columns are in
/// cache, not one strided store per pair.
pub(super) fn mirror_lower(square: &mut [f64], n: usize) {
    for j0 in (0..n).step_by(MIRROR_TILE) {
        let j1 = (j0 + MIRROR_TILE).min(n);
        for i0 in (j0..n).step_by(MIRROR_TILE) {
            let i1 = (i0 + MIRROR_TILE).min(n);
            for i in i0..i1 {
                // Row `i` of the upper triangle, columns `j0..min(j1, i)`:
                // `square[j + i * n]` for consecutive `j`.
                for j in j0..j1.min(i) {
                    square[j + i * n] = square[i + j * n];
                }
            }
        }
    }
}

/// An ARD training square of `n` points and `d` dimensions from a fill:
/// each column run of the lower triangle goes straight into its packed
/// place through one reused `d · n` buffer.
pub(super) fn fill_ard_square<T: KernelScalar>(
    filler: &dyn DistanceFill,
    n: usize,
    d: usize,
    tidy: Tidy,
) -> Result<ArdSqDiffBuf<T>, GprError> {
    let len = packed_len(n)?
        .checked_mul(d)
        .ok_or(GprError::SizeOverflow)?;
    let mut packed = vec![T::from_f64(0.0); len];
    let mut buffer = vec![0.0; n.checked_mul(d).ok_or(GprError::SizeOverflow)?];
    let mut rounding: Vec<FillRounding> = (0..d).map(|_| FillRounding::default()).collect();
    for col in 0..n {
        let len = n - col;
        let runs = &mut buffer[..len * d];
        filler.fill_column(col, col..n, runs);
        for (k, rounding) in rounding.iter_mut().enumerate() {
            let column = &mut packed[packed_run(n, k, col)];
            fill_run(
                &runs[k * len..(k + 1) * len],
                col,
                tidy,
                rounding,
                |at, v| {
                    column[at] = T::from_f64(v);
                },
            )
            .map_err(|err| err.in_dim(k))?;
        }
    }
    if let Tidy::Within(rel) = tidy {
        for (k, rounding) in rounding.iter().enumerate() {
            rounding.judge(rel).map_err(|err| err.in_dim(k))?;
        }
    }
    Ok(ArdSqDiffBuf::from_packed(packed, n, d))
}

/// Checks one column run of a training square a fill wrote (rows
/// `col..n`): [`Tidy::Exact`] refuses at once; [`Tidy::Within`] notes into
/// `rounding` and stores the repaired values in `out`.
pub(super) fn fill_run(
    run: &[f64],
    col: usize,
    tidy: Tidy,
    rounding: &mut FillRounding,
    mut store: impl FnMut(usize, f64),
) -> Result<(), GprError> {
    match tidy {
        Tidy::Exact => {
            check_lower_run(run, col)?;
            for (at, &v) in run.iter().enumerate() {
                store(at, v);
            }
        }
        Tidy::Within(_) => {
            if let Some(at) = run.iter().position(|v| !v.is_finite()) {
                return Err(invalid_value(run[at], col + at, col));
            }
            for (at, &v) in run.iter().enumerate() {
                store(at, rounding.note(v, col + at, col));
            }
        }
    }
    Ok(())
}

/// The training `d²` a model owns, one entry per slot of its kernel, by
/// shape in the kernel's slot order: entry `at` of a shape is the slot a
/// compiled leaf numbers `at` ([`crate::kernel::SuppliedSpec::at`]).
#[derive(Clone, Debug)]
pub struct TrainSources<T> {
    n: usize,
    /// Leading dimension of the scalar squares (`≥ n`; online growth).
    cap: usize,
    /// Dense squares of the scalar slots.
    scalar: Vec<(SlotId, Vec<T>)>,
    /// Packed lower triangles of the ARD slots.
    ard: Vec<(SlotId, ArdSqDiffBuf<T>)>,
}

/// Columns `start..start + len` of a [`TrainSources`], every row, as
/// rectangular blocks ([`TrainSources::columns`]).
pub(crate) struct TrainColumns<'s, T> {
    store: &'s TrainSources<T>,
    start: usize,
    len: usize,
}

impl<T: KernelScalar> RectSlots<T> for TrainColumns<'_, T> {
    fn scalar(&self, at: usize) -> Result<MatRef<'_, T>, GprError> {
        let store = self.store;
        let cap = store.cap.max(1);
        let (_, square) = store.scalar.get(at).ok_or_else(unbound)?;
        Ok(MatRef::from_column_major_slice_with_stride(
            &square[self.start * cap..],
            store.n,
            self.len,
            cap,
        ))
    }

    fn ard(&self, at: usize) -> Result<ArdRect<'_, T>, GprError> {
        let store = self.store;
        let (_, cache) = store.ard.get(at).ok_or_else(unbound)?;
        Ok(ArdRect::Checked(ArdBlocks::new(
            BlockList::Triangles(cache.view()),
            store.n,
            self.len,
            self.start,
        )))
    }
}

impl<T: KernelScalar> Default for TrainSources<T> {
    fn default() -> Self {
        Self::empty()
    }
}

impl<T: KernelScalar> TrainSources<T> {
    /// A coordinate model's: no slots.
    pub(crate) fn empty() -> Self {
        Self {
            n: 0,
            cap: 0,
            scalar: Vec::new(),
            ard: Vec::new(),
        }
    }

    /// Whether the store has no slot (a coordinate model's).
    pub(crate) fn is_empty(&self) -> bool {
        self.scalar.is_empty() && self.ard.is_empty()
    }

    /// About how many values [`Self::remove_point`] moves for `index`:
    /// the columns past it of each scalar square, and the rows past it of
    /// each dimension of each ARD slot.
    pub(crate) fn remove_work(&self, index: usize) -> usize {
        let n = self.n;
        let later = n.saturating_sub(index + 1);
        let scalar = self.scalar.len().saturating_mul(later.saturating_mul(n));
        let dims: usize = self.ard.iter().map(|(_, cache)| cache.view().d()).sum();
        let rows = (n * (n + 1) / 2).saturating_sub(index * (index + 1) / 2);
        scalar.saturating_add(dims.saturating_mul(rows))
    }

    /// Lays every ARD slot out as row runs (once), so an insert or a
    /// delete after does not fail.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::SizeOverflow`] when a slot cannot be laid out.
    pub(crate) fn ready_to_change(&mut self) -> Result<(), GprError> {
        for (_, cache) in &mut self.ard {
            cache.ready_to_change()?;
        }
        Ok(())
    }

    /// Makes room for one more point, so [`Self::check_push`] passes and
    /// [`Self::write_point`] writes in place: a full scalar square grows its leading dimension by a
    /// quarter, and each
    /// ARD slot reserves its own ([`ArdSqDiffBuf::reserve_point`]). The
    /// values read stay the same.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::SizeOverflow`] when a grown slot does not fit.
    pub(crate) fn reserve_point(&mut self) -> Result<(), GprError> {
        if self.is_empty() {
            return Ok(());
        }
        let n = self.n;
        if self.cap <= n {
            // A quarter more room: a re-layout copies the n² values once
            // per n/4 inserts, and the square stays near n² in memory.
            let cap = (n + 1).max(n + n / 4);
            let len = cap.checked_mul(cap).ok_or(GprError::SizeOverflow)?;
            for (_, square) in &mut self.scalar {
                let mut wider = vec![T::from_f64(0.0); len];
                for j in 0..n {
                    wider[j * cap..j * cap + n]
                        .copy_from_slice(&square[j * self.cap..j * self.cap + n]);
                }
                *square = wider;
            }
            self.cap = cap;
        }
        for (_, cache) in &mut self.ard {
            cache.reserve_point()?;
        }
        Ok(())
    }

    /// Every check of appending a point (`n × 1` columns `cols` to the
    /// points `0..n`), with nothing written: the room
    /// [`Self::reserve_point`] made, and a checked column for every slot.
    /// [`Self::write_point`] then writes it.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::UnsupportedKernelOperation`] when no room was
    /// reserved, or when `cols` lacks a slot of the store or holds an
    /// unchecked ARD block.
    pub(crate) fn check_push(&self, cols: &dyn RectSlots<T>) -> Result<(), GprError> {
        if self.is_empty() {
            return Ok(());
        }
        if self.cap <= self.n || self.ard.iter().any(|(_, cache)| !cache.can_push()) {
            return Err(no_room());
        }
        for at in 0..self.scalar.len() {
            cols.scalar(at)?;
        }
        for at in 0..self.ard.len() {
            if let ArdRect::Unchecked(_) = cols.ard(at)? {
                return Err(unbound());
            }
        }
        Ok(())
    }

    /// Appends the point once [`Self::check_push`] passed on the same
    /// `cols`: nothing in it fails, so a store of two copies writes both or
    /// neither.
    pub(crate) fn write_point(&mut self, cols: &dyn RectSlots<T>) {
        debug_assert!(
            self.check_push(cols).is_ok(),
            "write_point before check_push"
        );
        if self.is_empty() {
            return;
        }
        let (n, cap) = (self.n, self.cap);
        for (at, (_, square)) in self.scalar.iter_mut().enumerate() {
            if let Ok(column) = cols.scalar(at) {
                for i in 0..n {
                    let v = column[(i, 0)];
                    square[i + n * cap] = v;
                    square[n + i * cap] = v;
                }
                square[n + n * cap] = T::from_f64(0.0);
            }
        }
        for (at, (_, cache)) in self.ard.iter_mut().enumerate() {
            if let Ok(ArdRect::Checked(blocks)) = cols.ard(at) {
                cache.push_point(|k| &blocks.block(k)[..n]);
            }
        }
        self.n = n + 1;
    }

    /// Every check of [`Self::remove_point`], with nothing changed: `index`
    /// below `n`, and the ARD slots laid out ([`Self::ready_to_change`]).
    ///
    /// # Errors
    ///
    /// Returns [`GprError::IndexOutOfRange`] when `index ≥ n`, and
    /// [`GprError::UnsupportedKernelOperation`] when the store was not laid
    /// out for the change.
    pub(crate) fn check_remove(&self, index: usize) -> Result<(), GprError> {
        if self.is_empty() {
            return Ok(());
        }
        let n = self.n;
        if index >= n {
            return Err(GprError::IndexOutOfRange {
                reason: format!("point index {index} is out of range for n={n}"),
            });
        }
        if self.ard.iter().any(|(_, cache)| !cache.can_remove()) {
            return Err(no_room());
        }
        Ok(())
    }

    /// Removes point `index` in place, once [`Self::check_remove`] passed:
    /// its row and column leave every slot, and the later points move up
    /// one. Nothing in it fails.
    pub(crate) fn remove_point(&mut self, index: usize) {
        if self.is_empty() {
            return;
        }
        let (n, cap) = (self.n, self.cap);
        debug_assert!(index < n, "remove_point past the points");
        for (_, cache) in &mut self.ard {
            cache.remove_point(index);
        }
        for (_, square) in &mut self.scalar {
            // Columns before `index` keep their rows above it and move the
            // rows below it up one; each later column `j` takes column
            // `j + 1` without row `index`, going forward, so it reads a
            // column not yet written. The two parts share no column, and
            // from inside the pool (a delete beside the factor's update)
            // they move side by side.
            let split = (index * cap).min(square.len());
            let (before, after) = square.split_at_mut(split);
            let head = |before: &mut [T]| {
                for j in 0..index {
                    let at = j * cap;
                    before.copy_within(at + index + 1..at + n, at + index);
                }
            };
            let tail = |after: &mut [T]| {
                for j in 0..(n - 1).saturating_sub(index) {
                    let (from, to) = ((j + 1) * cap, j * cap);
                    after.copy_within(from..from + index, to);
                    after.copy_within(from + index + 1..from + n, to + index);
                }
            };
            if rayon::current_thread_index().is_some() {
                rayon::join(|| head(before), || tail(after));
            } else {
                head(before);
                tail(after);
            }
        }
        self.n = n - 1;
    }

    /// The store of the `n × n` training squares of `slots` (the kernel's
    /// slots, in order) from `sources`, checked as each source asks
    /// ([`check_block`]). A scalar table the caller moved in is kept without
    /// a copy (`f64`); an ARD table is packed into its lower triangles one
    /// column run at a time; a fill writes the lower triangle column by
    /// column into the store, so no dense `d · n²` buffer is made.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::EmptyInput`] when `n` is zero,
    /// [`GprError::DistanceSlot`] for a source of a slot the kernel does
    /// not read, two sources of one slot, or a slot without a source,
    /// [`GprError::LengthMismatch`] for a table of the wrong length or
    /// count, [`GprError::SizeOverflow`] when
    /// a store does not fit, and [`GprError::InvalidDistance`] for a value
    /// the source's check refuses.
    pub(crate) fn bind<'a>(
        slots: &[DistanceSlot],
        sources: impl IntoIterator<Item = DistanceSource<'a>>,
        n: usize,
    ) -> Result<Self, GprError> {
        crate::data::require_nonempty(n)?;
        let len = n.checked_mul(n).ok_or(GprError::SizeOverflow)?;
        let ards = ard_count(slots);
        // One entry per slot of each shape, at its place among them.
        let mut scalar: Vec<Option<(SlotId, Vec<T>)>> =
            (0..slots.len() - ards).map(|_| None).collect();
        let mut ard: Vec<Option<(SlotId, ArdSqDiffBuf<T>)>> = (0..ards).map(|_| None).collect();
        for source in sources {
            let at = slot_of(slots, &source)?;
            let taken = if at.is_ard() {
                ard[at.in_shape].is_some()
            } else {
                scalar[at.in_shape].is_some()
            };
            if taken {
                return Err(duplicate(at.place));
            }
            let tidy = source.tidy;
            let id = at.slot.id();
            let at_slot = |err: GprError| err.in_slot(at.place);
            match source.data {
                SourceData::Scalar(ScalarData::Values(values)) => {
                    crate::data::require_count(values.len(), len, "squared distances")?;
                    let mut values = values;
                    if check_block(&values, n, n, BlockKind::Square, tidy).map_err(at_slot)? {
                        repair_block(values.to_mut(), n, n, BlockKind::Square);
                    }
                    scalar[at.in_shape] = Some((id, T::vec_from_f64(values.into_owned())));
                }
                SourceData::Scalar(ScalarData::Fill(filler)) => {
                    let square = fill_scalar_square(filler, n, tidy).map_err(at_slot)?;
                    scalar[at.in_shape] = Some((id, T::vec_from_f64(square)));
                }
                SourceData::Ard(d, data) => {
                    let cache = train_ard(data, n, d, tidy).map_err(at_slot)?;
                    ard[at.in_shape] = Some((id, cache));
                }
            }
        }
        let unbound_place = shape_places(slots).position(|(is_ard, at)| {
            if is_ard {
                ard[at].is_none()
            } else {
                scalar[at].is_none()
            }
        });
        if unbound_place.is_some() {
            return Err(missing(unbound_place));
        }
        // Every entry is bound: in slot order, collected in place.
        let scalar: Option<Vec<_>> = scalar.into_iter().collect();
        let ard: Option<Vec<_>> = ard.into_iter().collect();
        let (Some(scalar), Some(ard)) = (scalar, ard) else {
            return Err(unbound());
        };
        let store = Self {
            n,
            cap: n,
            scalar,
            ard,
        };
        store.require_in_range(slots)?;
        Ok(store)
    }

    /// Checks that the store holds every value in its scalar: an `f64` value
    /// past the range of a narrower storage scalar rounds to infinity when
    /// it is cast. Nothing to check for an `f64` store.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidDistance`] at the first value that did
    /// not fit.
    fn require_in_range(&self, slots: &[DistanceSlot]) -> Result<(), GprError> {
        if reads_in_place::<T>() {
            return Ok(());
        }
        let (n, cap) = (self.n, self.cap.max(1));
        for (id, square) in &self.scalar {
            for j in 0..n {
                let column = &square[j * cap..j * cap + n];
                if let Some(i) = column.iter().position(|v| !v.is_finite()) {
                    return Err(in_slot_of(out_of_range(i, j), slots, *id));
                }
            }
        }
        for (id, cache) in &self.ard {
            let view = cache.view();
            for dim in 0..view.d() {
                if let Some((row, col)) = view.position(dim, |v| !v.is_finite()) {
                    return Err(in_slot_of(out_of_range(row, col).in_dim(dim), slots, *id));
                }
            }
        }
        Ok(())
    }

    /// The training `d²` of `slot` in the persist layout: the lower
    /// triangle column by column (column `col` holds rows `col..n`), an ARD
    /// slot dimension after dimension. `None` when the store has no such
    /// slot. A packed ARD slot is read in place.
    pub(crate) fn packed(&self, slot: SlotId) -> Option<Cow<'_, [T]>> {
        let (n, cap) = (self.n, self.cap.max(1));
        if let Some((_, square)) = self.scalar.iter().find(|(id, _)| *id == slot) {
            return Some(Cow::Owned(
                (0..n)
                    .flat_map(|col| &square[col * cap + col..col * cap + n])
                    .copied()
                    .collect(),
            ));
        }
        self.ard
            .iter()
            .find(|(id, _)| *id == slot)
            .map(|(_, cache)| cache.packed())
    }

    /// The store of `n` points of `slots` (the kernel's slots, in order)
    /// from their [`Self::packed`] values, one entry per slot. A scalar
    /// slot is unpacked into its square; an ARD slot keeps its values as
    /// they are. Every value must be finite and non-negative and every
    /// diagonal zero: a persisted store was checked when it was bound.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::EmptyInput`] when `n` is zero,
    /// [`GprError::LengthMismatch`] when the entries do not match the slots
    /// or their lengths their shapes, [`GprError::SizeOverflow`] when a
    /// slot does not fit, and [`GprError::InvalidDistance`] at the first
    /// value that is not valid.
    pub(crate) fn from_packed(
        slots: &[DistanceSlot],
        values: Vec<Vec<T>>,
        n: usize,
    ) -> Result<Self, GprError> {
        crate::data::require_nonempty(n)?;
        crate::data::require_count(values.len(), slots.len(), "persisted distance slots")?;
        let tri = packed_len(n)?;
        let mut scalar = Vec::new();
        let mut ard = Vec::new();
        for (place, (slot, values)) in slots.iter().zip(values).enumerate() {
            let len = tri
                .checked_mul(slot.shape().blocks())
                .ok_or(GprError::SizeOverflow)?;
            crate::data::require_count(values.len(), len, "persisted squared distances")?;
            check_packed(&values, n, slot.shape()).map_err(|err| err.in_slot(place))?;
            match slot.shape() {
                SlotShape::Scalar => {
                    let mut square =
                        vec![T::from_f64(0.0); n.checked_mul(n).ok_or(GprError::SizeOverflow)?];
                    let mut at = 0;
                    for col in 0..n {
                        for row in col..n {
                            let v = values[at];
                            square[row + col * n] = v;
                            square[col + row * n] = v;
                            at += 1;
                        }
                    }
                    scalar.push((slot.id(), square));
                }
                SlotShape::Ard(dims) => {
                    ard.push((slot.id(), ArdSqDiffBuf::from_packed(values, n, dims)));
                }
            }
        }
        Ok(Self {
            n,
            cap: n,
            scalar,
            ard,
        })
    }

    /// The same squares in `f64`: every value fits, so no slot is named.
    pub(crate) fn to_f64(&self) -> Result<TrainSources<f64>, GprError> {
        self.cast(&[])
    }

    #[cfg(test)]
    /// Each slot's blocks in `f64`, dense `n × n` and one after another in
    /// one buffer: the scalar slots, then the ARD slots, each in slot order.
    pub(crate) fn dense_f64(&self) -> Vec<(SlotShape, Vec<f64>)> {
        let n = self.n;
        let scalar = self.scalar.iter().map(|(_, square)| {
            (
                SlotShape::Scalar,
                (0..n * n)
                    .map(|at| square[at % n + (at / n) * self.cap].to_f64())
                    .collect(),
            )
        });
        let ard = self.ard.iter().map(|(_, cache)| {
            let view = cache.view();
            let values = (0..view.d())
                .flat_map(|k| (0..n * n).map(move |at| view.get(k, at % n, at / n).to_f64()))
                .collect();
            (SlotShape::Ard(view.d()), values)
        });
        scalar.chain(ard).collect()
    }

    /// The training columns `cols` (every row) as rectangular blocks, read
    /// in place: a scalar slot through its leading dimension, an ARD slot
    /// from its packed triangles.
    pub(crate) fn columns(&self, cols: std::ops::Range<usize>) -> TrainColumns<'_, T> {
        TrainColumns {
            store: self,
            start: cols.start,
            len: cols.len(),
        }
    }

    /// The same squares at the scalar `U`: a scalar square column by
    /// column (contiguous runs past the leading dimension), an ARD cache in
    /// one pass over its packed values. `slots` (the kernel's) name the
    /// slot of a value past the range of `U`.
    pub(crate) fn cast<U: KernelScalar>(
        &self,
        slots: &[DistanceSlot],
    ) -> Result<TrainSources<U>, GprError> {
        let (n, cap) = (self.n, self.cap.max(1));
        let cast = |v: T| U::from_f64(v.to_f64());
        let scalar = self
            .scalar
            .iter()
            .map(|(id, square)| {
                let mut out = Vec::with_capacity(n * n);
                for j in 0..n {
                    out.extend(square[j * cap..j * cap + n].iter().map(|&v| cast(v)));
                }
                (*id, out)
            })
            .collect();
        let ard = self
            .ard
            .iter()
            .map(|(id, cache)| (*id, cache.map(cast)))
            .collect();
        let store = TrainSources {
            n,
            cap: n,
            scalar,
            ard,
        };
        store.require_in_range(slots)?;
        Ok(store)
    }
}

/// Checks persisted lower triangles of order `n` of a slot of `shape`,
/// block after block: every value finite and non-negative, every diagonal
/// zero.
pub(super) fn check_packed<T: KernelScalar>(
    values: &[T],
    n: usize,
    shape: SlotShape,
) -> Result<(), GprError> {
    let mut at = 0;
    let mut k = 0;
    while at < values.len() {
        for col in 0..n {
            let run = &values[at..at + (n - col)];
            check_lower_run(run, col).map_err(|err| in_block(err, shape, k))?;
            at += n - col;
        }
        k += 1;
    }
    Ok(())
}

impl<T: KernelScalar> SquareSlots<T> for TrainSources<T> {
    fn scalar(&self, at: usize) -> Result<MatRef<'_, T>, GprError> {
        let (_, square) = self.scalar.get(at).ok_or_else(unbound)?;
        Ok(MatRef::from_column_major_slice_with_stride(
            square,
            self.n,
            self.n,
            self.cap.max(1),
        ))
    }

    fn ard(&self, at: usize) -> Result<ArdSquare<'_, T>, GprError> {
        let (_, cache) = self.ard.get(at).ok_or_else(unbound)?;
        Ok(ArdSquare::Packed(cache.view()))
    }
}
