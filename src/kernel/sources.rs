//! The supplied squared distances a model reads: binding and checking the
//! caller's sources, the training `d²` a model owns, and the per-call blocks
//! of a prediction or an insert.
//!
//! A source arrives as `f64`. The training store keeps it in the model's
//! storage scalar: a scalar slot as a dense square, an ARD slot as the
//! packed lower triangles of [`ArdSqDiffBuf`]. A prediction block of an
//! `f64` model is read in place.

use std::any::Any;
use std::borrow::Cow;
use std::fmt;

use faer::MatRef;

use super::compiled::supplied::{RectEntry, RectTable, SquareSlot, SquareSlots};
use super::dist::ArdSqDiffBuf;
use super::{DistanceSlot, DistanceSource, KernelScalar, SlotId, SlotShape};
use super::{ScalarOps, SourceData};
use crate::error::GprError;

/// A source's `d²`, checked: `shape.blocks()` dense blocks of `rows × cols`.
pub struct RawSlot<'a> {
    pub(crate) id: SlotId,
    pub(crate) shape: SlotShape,
    data: RawData<'a>,
    rows: usize,
    cols: usize,
}

enum RawData<'a> {
    /// One table per block.
    Blocks(Vec<Cow<'a, [f64]>>),
    /// Every block in one buffer, one after another (a fill).
    Packed(Vec<f64>),
}

impl RawSlot<'_> {
    /// Block `k` (`rows × cols`, column-major).
    pub(crate) fn block(&self, k: usize) -> &[f64] {
        let len = self.rows * self.cols;
        match &self.data {
            RawData::Blocks(blocks) => &blocks[k],
            RawData::Packed(all) => &all[k * len..(k + 1) * len],
        }
    }

    /// Takes block 0 as an owned buffer (moved when the source owned it).
    fn into_first(self) -> Vec<f64> {
        match self.data {
            RawData::Blocks(mut blocks) if !blocks.is_empty() => blocks.swap_remove(0).into_owned(),
            RawData::Blocks(_) => Vec::new(),
            RawData::Packed(all) => all,
        }
    }
}

/// What a block must satisfy beyond its length and finite values.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BlockKind {
    /// Pairs of one set: zero diagonal, symmetric.
    Square,
    /// Pairs of two sets.
    Rect,
}

/// Binds `sources` to `slots` (the kernel's slots, in order) and checks
/// each block of `rows × cols`. The result follows the order of `slots`.
///
/// # Errors
///
/// Returns [`GprError::LengthMismatch`] for a source of a slot the kernel
/// does not read, a slot without a source, two sources of one slot, or a
/// block of the wrong length or count; [`GprError::EmptyInput`] when `rows`
/// or `cols` is zero; [`GprError::NonFiniteInput`] for a non-finite value;
/// [`GprError::ShapeMismatch`] for a value, a diagonal, or a mirror pair
/// past what rounding leaves ([`check_block`]). Within that, a block is
/// tidied ([`tidy_block`]): a borrowed table is then copied.
pub(crate) fn bind<'a>(
    slots: &[DistanceSlot],
    sources: impl IntoIterator<Item = DistanceSource<'a>>,
    rows: usize,
    cols: usize,
    kind: BlockKind,
) -> Result<Vec<RawSlot<'a>>, GprError> {
    crate::data::require_nonempty(rows)?;
    crate::data::require_nonempty(cols)?;
    let len = rows.checked_mul(cols).ok_or(GprError::SizeOverflow)?;
    let mut bound: Vec<Option<DistanceSource<'a>>> = slots.iter().map(|_| None).collect();
    for source in sources {
        let Some(at) = slots.iter().position(|slot| slot.id() == source.slot) else {
            return Err(GprError::LengthMismatch {
                reason: "squared distances were supplied for a slot the kernel does not read"
                    .to_owned(),
            });
        };
        if bound[at].is_some() {
            return Err(GprError::LengthMismatch {
                reason: "two sources were supplied for one distance slot".to_owned(),
            });
        }
        bound[at] = Some(source);
    }
    let mut raw = Vec::with_capacity(slots.len());
    for (slot, source) in slots.iter().zip(bound) {
        let Some(source) = source else {
            return Err(GprError::LengthMismatch {
                reason: "a distance slot of the kernel has no source".to_owned(),
            });
        };
        let shape = slot.shape();
        let blocks = shape.blocks();
        let data = match source.data {
            SourceData::Values(values) if blocks == 1 => RawData::Blocks(vec![values]),
            SourceData::Blocks(tables) => RawData::Blocks(tables),
            SourceData::Values(_) => {
                return Err(GprError::LengthMismatch {
                    reason: format!("expected {blocks} tables of squared distances, got 1"),
                });
            }
            SourceData::Fill(filler) => {
                let total = len.checked_mul(blocks).ok_or(GprError::SizeOverflow)?;
                let mut all = vec![0.0; total];
                filler.fill(rows, cols, &mut all);
                RawData::Packed(all)
            }
        };
        let mut raw_slot = RawSlot {
            id: slot.id(),
            shape,
            data,
            rows,
            cols,
        };
        check_slot(&mut raw_slot, blocks, len, kind)?;
        raw.push(raw_slot);
    }
    Ok(raw)
}

fn check_slot(
    slot: &mut RawSlot<'_>,
    blocks: usize,
    len: usize,
    kind: BlockKind,
) -> Result<(), GprError> {
    let (rows, cols) = (slot.rows, slot.cols);
    match &mut slot.data {
        RawData::Blocks(tables) => {
            if tables.len() != blocks {
                return Err(GprError::LengthMismatch {
                    reason: format!(
                        "expected {blocks} tables of squared distances, got {}",
                        tables.len()
                    ),
                });
            }
            for table in tables.iter_mut() {
                // A borrowed table is copied only when rounding needs a fix.
                if check_block(table, rows, cols, kind)? {
                    tidy_block(table.to_mut(), rows, cols, kind);
                }
            }
        }
        RawData::Packed(all) => {
            for block in all.chunks_exact_mut(len.max(1)).take(blocks) {
                if check_block(block, rows, cols, kind)? {
                    tidy_block(block, rows, cols, kind);
                }
            }
        }
    }
    Ok(())
}

/// Relative tolerance, against the largest `|d²|` of a block, for what
/// floating point leaves in a table of squared distances: a diagonal that
/// is not exactly zero, two mirror entries of a square that differ, or a
/// slightly negative value. Within it the block is tidied; past it the
/// table is not one of squared distances.
const ROUNDING_TOL: f64 = 1e-6;

/// The rounding tolerance of `block`.
fn rounding_tol(block: &[f64]) -> f64 {
    ROUNDING_TOL * block.iter().fold(0.0f64, |acc, v| acc.max(v.abs()))
}

/// Checks one `rows × cols` block of `d²` and returns whether rounding left
/// something [`tidy_block`] fixes (within [`ROUNDING_TOL`]).
///
/// # Errors
///
/// Returns [`GprError::LengthMismatch`] for the wrong length,
/// [`GprError::NonFiniteInput`] for a non-finite value, and
/// [`GprError::ShapeMismatch`] for a value below `−tol`, a square that is
/// not square, a diagonal past `tol`, or mirror entries further apart.
pub(crate) fn check_block(
    block: &[f64],
    rows: usize,
    cols: usize,
    kind: BlockKind,
) -> Result<bool, GprError> {
    crate::data::require_count(block.len(), rows * cols, "squared distances")?;
    crate::data::require_finite(block)?;
    let tol = rounding_tol(block);
    let mut tidy = false;
    for (at, &v) in block.iter().enumerate() {
        if v < -tol {
            return Err(GprError::ShapeMismatch {
                reason: format!(
                    "squared distance ({}, {}) is negative",
                    at % rows.max(1),
                    at / rows.max(1)
                ),
            });
        }
        tidy |= v < 0.0;
    }
    if kind == BlockKind::Square {
        if rows != cols {
            return Err(GprError::ShapeMismatch {
                reason: format!("a square of squared distances is {rows}x{cols}"),
            });
        }
        for j in 0..cols {
            let diag = block[j + j * rows];
            if diag.abs() > tol {
                return Err(GprError::ShapeMismatch {
                    reason: format!("squared distance ({j}, {j}) is not zero"),
                });
            }
            tidy |= diag.abs() > 0.0;
            for i in (j + 1)..rows {
                let gap = (block[i + j * rows] - block[j + i * rows]).abs();
                if gap > tol {
                    return Err(GprError::ShapeMismatch {
                        reason: format!("squared distances ({i}, {j}) and ({j}, {i}) differ"),
                    });
                }
                tidy |= gap > 0.0;
            }
        }
    }
    Ok(tidy)
}

/// Fixes what [`check_block`] accepted as rounding: negative values to
/// zero and, for a square, a zero diagonal and each mirror pair set to its
/// mean.
fn tidy_block(block: &mut [f64], rows: usize, cols: usize, kind: BlockKind) {
    for v in block.iter_mut() {
        *v = v.max(0.0);
    }
    if kind == BlockKind::Square {
        for j in 0..cols {
            block[j + j * rows] = 0.0;
            for i in (j + 1)..rows {
                let mean = 0.5 * (block[i + j * rows] + block[j + i * rows]);
                block[i + j * rows] = mean;
                block[j + i * rows] = mean;
            }
        }
    }
}

/// The training `d²` of one slot, in the storage scalar.
#[derive(Clone, Debug)]
pub enum TrainData<T> {
    /// Dense square, leading dimension `TrainSources::cap`.
    Scalar(Vec<T>),
    /// Packed lower triangles of every dimension.
    Ard(ArdSqDiffBuf<T>),
}

/// The training `d²` a model owns, one entry per slot of its kernel.
#[derive(Clone, Debug)]
pub struct TrainSources<T> {
    n: usize,
    /// Leading dimension of the scalar squares (`≥ n`; online growth).
    cap: usize,
    slots: Vec<(SlotId, TrainData<T>)>,
}

/// A change to a [`TrainSources`] computed by [`TrainSources::stage_append`]
/// or [`TrainSources::stage_delete`], applied by [`TrainSources::commit`].
#[derive(Debug)]
pub struct Staged<T> {
    n: usize,
    cap: usize,
    changes: Vec<SlotChange<T>>,
}

#[derive(Debug)]
enum SlotChange<T> {
    /// A rebuilt slot.
    Replace(TrainData<T>),
    /// The new column of a scalar square, written in place.
    Column(Vec<T>),
    /// Removes a point of a scalar square in place.
    Remove(usize),
}

/// Writes point `n`'s column and row (`column`, then a zero diagonal) into
/// a square of leading dimension `cap`.
fn write_column<T: KernelScalar>(square: &mut [T], cap: usize, n: usize, column: &[T]) {
    for (i, &v) in column.iter().enumerate() {
        square[i + n * cap] = v;
        square[n + i * cap] = v;
    }
    square[n + n * cap] = T::from_f64(0.0);
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
            slots: Vec::new(),
        }
    }

    /// Whether the kernel reads any supplied distances.
    pub(crate) fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// The training points the squares cover.
    pub(crate) fn n(&self) -> usize {
        self.n
    }

    /// The store for the checked training squares `raw` of `n` points.
    pub(crate) fn from_raw(raw: Vec<RawSlot<'_>>, n: usize) -> Result<Self, GprError> {
        let mut slots = Vec::with_capacity(raw.len());
        for slot in raw {
            let id = slot.id;
            let data = match slot.shape {
                SlotShape::Scalar => TrainData::Scalar(T::vec_from_f64(slot.into_first())),
                SlotShape::Ard(d) => TrainData::Ard(ArdSqDiffBuf::from_pairs(n, d, |k, i, j| {
                    T::from_f64(slot.block(k)[i + j * n])
                })?),
            };
            slots.push((id, data));
        }
        Ok(Self { n, cap: n, slots })
    }

    /// Computes the append of one point without changing `self`: `cols`
    /// holds each slot's `n × 1` column to the existing points, in slot
    /// order. The new diagonal is zero. [`Self::commit`] applies it.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::SizeOverflow`] when the grown store does not fit.
    pub(crate) fn stage_append(&self, cols: &[RawSlot<'_>]) -> Result<Staged<T>, GprError> {
        let n = self.n;
        if self.slots.is_empty() {
            return Ok(self.unchanged());
        }
        if cols.len() != self.slots.len()
            || cols
                .iter()
                .zip(&self.slots)
                .any(|(col, (id, _))| col.id != *id)
        {
            return Err(GprError::LengthMismatch {
                reason: "the new point's squared distances do not match the model's slots"
                    .to_owned(),
            });
        }
        let grow = self.cap < n + 1;
        let new_cap = if grow {
            (n + 1).max(self.cap.max(1).saturating_mul(2))
        } else {
            self.cap
        };
        let mut changes = Vec::with_capacity(self.slots.len());
        for ((_, data), col) in self.slots.iter().zip(cols) {
            let change = match data {
                TrainData::Scalar(square) => {
                    let column: Vec<T> = col.block(0).iter().map(|&v| T::from_f64(v)).collect();
                    if grow {
                        let len = new_cap.checked_mul(new_cap).ok_or(GprError::SizeOverflow)?;
                        let mut wider = vec![T::from_f64(0.0); len];
                        for j in 0..n {
                            for i in 0..n {
                                wider[i + j * new_cap] = square[i + j * self.cap];
                            }
                        }
                        write_column(&mut wider, new_cap, n, &column);
                        SlotChange::Replace(TrainData::Scalar(wider))
                    } else {
                        SlotChange::Column(column)
                    }
                }
                TrainData::Ard(cache) => {
                    let old = cache.view();
                    SlotChange::Replace(TrainData::Ard(ArdSqDiffBuf::from_pairs(
                        n + 1,
                        old.d(),
                        |k, i, j| {
                            if i == n && j == n {
                                T::from_f64(0.0)
                            } else if i == n {
                                T::from_f64(col.block(k)[j])
                            } else {
                                old.get(k, i, j)
                            }
                        },
                    )?))
                }
            };
            changes.push(change);
        }
        Ok(Staged {
            n: n + 1,
            cap: new_cap,
            changes,
        })
    }

    /// Computes the removal of point `index` without changing `self`.
    /// [`Self::commit`] applies it.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::IndexOutOfRange`] when `index ≥ n`.
    pub(crate) fn stage_delete(&self, index: usize) -> Result<Staged<T>, GprError> {
        if self.slots.is_empty() {
            return Ok(self.unchanged());
        }
        let n = self.n;
        if index >= n {
            return Err(GprError::IndexOutOfRange {
                reason: format!("point index {index} is out of range for n={n}"),
            });
        }
        let skip = |i: usize| if i >= index { i + 1 } else { i };
        let mut changes = Vec::with_capacity(self.slots.len());
        for (_, data) in &self.slots {
            changes.push(match data {
                TrainData::Scalar(_) => SlotChange::Remove(index),
                TrainData::Ard(cache) => {
                    let old = cache.view();
                    SlotChange::Replace(TrainData::Ard(ArdSqDiffBuf::from_pairs(
                        n - 1,
                        old.d(),
                        |k, i, j| old.get(k, skip(i), skip(j)),
                    )?))
                }
            });
        }
        Ok(Staged {
            n: n - 1,
            cap: self.cap,
            changes,
        })
    }

    fn unchanged(&self) -> Staged<T> {
        Staged {
            n: self.n,
            cap: self.cap,
            changes: Vec::new(),
        }
    }

    /// Applies a change staged on this store. Cannot fail.
    pub(crate) fn commit(&mut self, staged: Staged<T>) {
        let (n, cap) = (self.n, self.cap);
        // Staged on this store: one change per slot, in slot order, and an
        // in-place change only for a scalar slot.
        debug_assert!(staged.changes.is_empty() || staged.changes.len() == self.slots.len());
        for ((_, data), change) in self.slots.iter_mut().zip(staged.changes) {
            match (data, change) {
                (data, SlotChange::Replace(fresh)) => *data = fresh,
                (TrainData::Scalar(square), SlotChange::Column(column)) => {
                    write_column(square, cap, n, &column);
                }
                (TrainData::Scalar(square), SlotChange::Remove(index)) => {
                    let skip = |i: usize| if i >= index { i + 1 } else { i };
                    // Forward walk: every read is at or past its write.
                    for j in 0..n - 1 {
                        for i in 0..n - 1 {
                            square[i + j * cap] = square[skip(i) + skip(j) * cap];
                        }
                    }
                }
                (TrainData::Ard(_), SlotChange::Column(_) | SlotChange::Remove(_)) => {
                    debug_assert!(false, "an in-place change staged for an ARD slot");
                }
            }
        }
        self.n = staged.n;
        self.cap = staged.cap;
    }

    /// The same squares in `f64`.
    pub(crate) fn to_f64(&self) -> Result<TrainSources<f64>, GprError> {
        let n = self.n;
        let mut slots = Vec::with_capacity(self.slots.len());
        for (id, data) in &self.slots {
            let data = match data {
                TrainData::Scalar(square) => TrainData::Scalar(
                    (0..n * n)
                        .map(|at| square[at % n + (at / n) * self.cap].to_f64())
                        .collect(),
                ),
                TrainData::Ard(cache) => {
                    let view = cache.view();
                    TrainData::Ard(ArdSqDiffBuf::from_pairs(n, view.d(), |k, i, j| {
                        view.get(k, i, j).to_f64()
                    })?)
                }
            };
            slots.push((*id, data));
        }
        Ok(TrainSources { n, cap: n, slots })
    }

    /// Each slot's blocks in `f64`, dense `n × n`, in slot order (saving).
    pub(crate) fn dense_f64(&self) -> Vec<(SlotShape, Vec<Vec<f64>>)> {
        let n = self.n;
        self.slots
            .iter()
            .map(|(_, data)| match data {
                TrainData::Scalar(square) => (
                    SlotShape::Scalar,
                    vec![
                        (0..n * n)
                            .map(|at| square[at % n + (at / n) * self.cap].to_f64())
                            .collect(),
                    ],
                ),
                TrainData::Ard(cache) => {
                    let view = cache.view();
                    let blocks = (0..view.d())
                        .map(|k| {
                            (0..n * n)
                                .map(|at| view.get(k, at % n, at / n).to_f64())
                                .collect()
                        })
                        .collect();
                    (SlotShape::Ard(view.d()), blocks)
                }
            })
            .collect()
    }

    /// The `d²` of `rows × cols` training pairs, gathered per slot.
    pub(crate) fn gather(&self, rows: &[usize], cols: &[usize]) -> GatheredRect<T> {
        let slots = self
            .slots
            .iter()
            .map(|(id, data)| {
                let blocks = match data {
                    TrainData::Scalar(square) => vec![
                        cols.iter()
                            .flat_map(|&j| rows.iter().map(move |&i| square[i + j * self.cap]))
                            .collect(),
                    ],
                    TrainData::Ard(cache) => {
                        let view = cache.view();
                        (0..view.d())
                            .map(|k| {
                                cols.iter()
                                    .flat_map(|&j| rows.iter().map(move |&i| view.get(k, i, j)))
                                    .collect()
                            })
                            .collect()
                    }
                };
                (*id, matches!(data, TrainData::Ard(_)), blocks)
            })
            .collect();
        GatheredRect {
            rows: rows.len(),
            cols: cols.len(),
            slots,
        }
    }

    /// The training columns `cols` (every row) as a rectangular table. A
    /// scalar slot is read in place; an ARD slot's packed triangles are
    /// unpacked into `ard`, whose buffers are reused from call to call.
    pub(crate) fn column_table<'s>(
        &'s self,
        cols: std::ops::Range<usize>,
        ard: &'s mut Vec<Vec<Vec<T>>>,
    ) -> RectTable<'s, T> {
        let (n, cap) = (self.n, self.cap.max(1));
        let (start, len) = (cols.start, cols.len());
        ard.resize_with(self.slots.len(), Vec::new);
        for ((_, data), buffers) in self.slots.iter().zip(ard.iter_mut()) {
            if let TrainData::Ard(cache) = data {
                let view = cache.view();
                buffers.resize_with(view.d(), Vec::new);
                for (k, block) in buffers.iter_mut().enumerate() {
                    block.clear();
                    block.extend(
                        (0..len)
                            .flat_map(|jj| (0..n).map(move |i| (i, start + jj)))
                            .map(|(i, j)| view.get(k, i, j)),
                    );
                }
            }
        }
        let ard: &'s Vec<Vec<Vec<T>>> = ard;
        RectTable(
            self.slots
                .iter()
                .zip(ard)
                .map(|((id, data), blocks)| {
                    let entry = match data {
                        TrainData::Scalar(square) => {
                            RectEntry::Scalar(MatRef::from_column_major_slice_with_stride(
                                &square[start * cap..],
                                n,
                                len,
                                cap,
                            ))
                        }
                        TrainData::Ard(_) => RectEntry::Ard {
                            blocks: blocks.iter().map(Vec::as_slice).collect(),
                            rows: n,
                            cols: len,
                        },
                    };
                    (*id, entry)
                })
                .collect(),
        )
    }

    /// The same squares at the scalar `U`.
    pub(crate) fn cast<U: KernelScalar>(&self) -> Result<TrainSources<U>, GprError> {
        let n = self.n;
        let mut slots = Vec::with_capacity(self.slots.len());
        for (id, data) in &self.slots {
            let data = match data {
                TrainData::Scalar(square) => TrainData::Scalar(
                    (0..n * n)
                        .map(|at| U::from_f64(square[at % n + (at / n) * self.cap].to_f64()))
                        .collect(),
                ),
                TrainData::Ard(cache) => {
                    let view = cache.view();
                    TrainData::Ard(ArdSqDiffBuf::from_pairs(n, view.d(), |k, i, j| {
                        U::from_f64(view.get(k, i, j).to_f64())
                    })?)
                }
            };
            slots.push((*id, data));
        }
        Ok(TrainSources { n, cap: n, slots })
    }

    /// The training squares of the points `index` (a subset, in that order).
    pub(crate) fn subset(&self, index: &[usize]) -> Result<Self, GprError> {
        let m = index.len();
        let mut slots = Vec::with_capacity(self.slots.len());
        for (id, data) in &self.slots {
            let data = match data {
                TrainData::Scalar(square) => TrainData::Scalar(
                    (0..m * m)
                        .map(|at| square[index[at % m] + index[at / m] * self.cap])
                        .collect(),
                ),
                TrainData::Ard(cache) => {
                    let view = cache.view();
                    TrainData::Ard(ArdSqDiffBuf::from_pairs(m, view.d(), |k, i, j| {
                        view.get(k, index[i], index[j])
                    })?)
                }
            };
            slots.push((*id, data));
        }
        Ok(Self {
            n: m,
            cap: m,
            slots,
        })
    }
}

/// The training `d²` a model of one precision keeps.
///
/// [`TrainSources`] in the storage scalar for a model that factors and
/// predicts in it; [`RefinedSources`] for a model that also refines in
/// `f64` and so keeps the caller's `f64` values next to the storage copy.
pub trait SourceStore<S: KernelScalar>: Clone + fmt::Debug + Send + Sync + 'static {
    /// A change computed before it is applied.
    type Staged;

    /// A coordinate model's: no slots.
    fn empty() -> Self;

    /// The store for the checked training squares `raw` of `n` points.
    fn from_raw(raw: Vec<RawSlot<'_>>, n: usize) -> Result<Self, GprError>;

    /// The squares in the storage scalar.
    fn storage(&self) -> &TrainSources<S>;

    /// The squares at `f64` without rounding, when the store keeps them.
    fn exact(&self) -> Option<&TrainSources<f64>>;

    /// [`TrainSources::stage_append`].
    fn stage_append(&self, cols: &[RawSlot<'_>]) -> Result<Self::Staged, GprError>;

    /// [`TrainSources::stage_delete`].
    fn stage_delete(&self, index: usize) -> Result<Self::Staged, GprError>;

    /// [`TrainSources::commit`].
    fn commit(&mut self, staged: Self::Staged);

    /// The squares at `f64`: [`Self::exact`], or the storage values widened.
    fn to_f64(&self) -> Result<Cow<'_, TrainSources<f64>>, GprError> {
        match self.exact() {
            Some(exact) => Ok(Cow::Borrowed(exact)),
            None => self.storage().to_f64().map(Cow::Owned),
        }
    }
}

impl<S: KernelScalar> SourceStore<S> for TrainSources<S> {
    type Staged = Staged<S>;

    fn empty() -> Self {
        Self::empty()
    }

    fn from_raw(raw: Vec<RawSlot<'_>>, n: usize) -> Result<Self, GprError> {
        Self::from_raw(raw, n)
    }

    fn storage(&self) -> &TrainSources<S> {
        self
    }

    fn exact(&self) -> Option<&TrainSources<f64>> {
        (self as &dyn Any).downcast_ref::<TrainSources<f64>>()
    }

    fn stage_append(&self, cols: &[RawSlot<'_>]) -> Result<Staged<S>, GprError> {
        Self::stage_append(self, cols)
    }

    fn stage_delete(&self, index: usize) -> Result<Staged<S>, GprError> {
        Self::stage_delete(self, index)
    }

    fn commit(&mut self, staged: Staged<S>) {
        Self::commit(self, staged);
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
    type Staged = (Staged<f32>, Staged<f64>);

    fn empty() -> Self {
        Self::default()
    }

    fn from_raw(raw: Vec<RawSlot<'_>>, n: usize) -> Result<Self, GprError> {
        let exact = TrainSources::<f64>::from_raw(raw, n)?;
        Ok(Self {
            storage: exact.cast()?,
            exact,
        })
    }

    fn storage(&self) -> &TrainSources<f32> {
        &self.storage
    }

    fn exact(&self) -> Option<&TrainSources<f64>> {
        Some(&self.exact)
    }

    fn stage_append(&self, cols: &[RawSlot<'_>]) -> Result<Self::Staged, GprError> {
        Ok((
            self.storage.stage_append(cols)?,
            self.exact.stage_append(cols)?,
        ))
    }

    fn stage_delete(&self, index: usize) -> Result<Self::Staged, GprError> {
        Ok((
            self.storage.stage_delete(index)?,
            self.exact.stage_delete(index)?,
        ))
    }

    fn commit(&mut self, (storage, exact): Self::Staged) {
        self.storage.commit(storage);
        self.exact.commit(exact);
    }
}

impl<T: KernelScalar> SquareSlots<T> for TrainSources<T> {
    fn square(&self, slot: SlotId) -> Option<SquareSlot<'_, T>> {
        let (_, data) = self.slots.iter().find(|(id, _)| *id == slot)?;
        Some(match data {
            TrainData::Scalar(square) => {
                SquareSlot::Scalar(MatRef::from_column_major_slice_with_stride(
                    square,
                    self.n,
                    self.n,
                    self.cap.max(1),
                ))
            }
            TrainData::Ard(cache) => SquareSlot::Ard(cache.view()),
        })
    }
}

/// Training `d²` gathered into rectangular blocks (rows × cols), per slot:
/// `(slot, is_ard, blocks)`.
#[derive(Clone, Debug)]
pub(crate) struct GatheredRect<T> {
    rows: usize,
    cols: usize,
    slots: Vec<(SlotId, bool, Vec<Vec<T>>)>,
}

impl<T: KernelScalar> GatheredRect<T> {
    /// The blocks as a table.
    pub(crate) fn table(&self) -> RectTable<'_, T> {
        RectTable(
            self.slots
                .iter()
                .map(|(id, ard, blocks)| {
                    let entry = if *ard {
                        RectEntry::Ard {
                            blocks: blocks.iter().map(Vec::as_slice).collect(),
                            rows: self.rows,
                            cols: self.cols,
                        }
                    } else {
                        RectEntry::Scalar(MatRef::from_column_major_slice(
                            &blocks[0], self.rows, self.cols,
                        ))
                    };
                    (*id, entry)
                })
                .collect(),
        )
    }

    /// The same blocks at the scalar `U`.
    pub(crate) fn cast<U: KernelScalar>(&self) -> GatheredRect<U> {
        GatheredRect {
            rows: self.rows,
            cols: self.cols,
            slots: self
                .slots
                .iter()
                .map(|(id, ard, blocks)| {
                    let blocks = blocks
                        .iter()
                        .map(|block| block.iter().map(|v| U::from_f64(v.to_f64())).collect())
                        .collect();
                    (*id, *ard, blocks)
                })
                .collect(),
        }
    }

    /// Columns `cols` of these blocks (a minibatch).
    pub(crate) fn columns(&self, cols: &[usize]) -> Self {
        let rows = self.rows;
        Self {
            rows,
            cols: cols.len(),
            slots: self
                .slots
                .iter()
                .map(|(id, ard, blocks)| {
                    let picked = blocks
                        .iter()
                        .map(|block| {
                            cols.iter()
                                .flat_map(|&j| block[j * rows..(j + 1) * rows].iter().copied())
                                .collect()
                        })
                        .collect();
                    (*id, *ard, picked)
                })
                .collect(),
        }
    }
}

/// The checked `d²` blocks of one prediction or insert, and the casts an
/// `f32` model reads them through.
pub(crate) struct QuerySources<'a, T: ScalarOps> {
    raw: Vec<RawSlot<'a>>,
    casts: Vec<Vec<T::RowCast>>,
}

impl<'a, T: KernelScalar> QuerySources<'a, T> {
    /// Binds and checks the blocks of `rows × cols` pairs.
    ///
    /// # Errors
    ///
    /// The errors of [`bind`].
    pub(crate) fn bind(
        slots: &[DistanceSlot],
        sources: impl IntoIterator<Item = DistanceSource<'a>>,
        rows: usize,
        cols: usize,
        kind: BlockKind,
    ) -> Result<Self, GprError> {
        let raw = bind(slots, sources, rows, cols, kind)?;
        let casts = raw
            .iter()
            .map(|slot| (0..slot.shape.blocks()).map(|_| T::empty_rows()).collect())
            .collect();
        Ok(Self { raw, casts })
    }

    /// Rows `rows` of every block, in `f64` (a sparse model's inducing rows).
    pub(crate) fn gather_rows(&self, rows: &[usize]) -> GatheredRect<f64> {
        let cols = self.raw.first().map_or(0, |slot| slot.cols);
        GatheredRect {
            rows: rows.len(),
            cols,
            slots: self
                .raw
                .iter()
                .map(|slot| {
                    let blocks = (0..slot.shape.blocks())
                        .map(|k| {
                            let block = slot.block(k);
                            (0..cols)
                                .flat_map(|j| rows.iter().map(move |&i| block[i + j * slot.rows]))
                                .collect()
                        })
                        .collect();
                    (slot.id, matches!(slot.shape, SlotShape::Ard(_)), blocks)
                })
                .collect(),
        }
    }

    /// The checked blocks, in slot order.
    pub(crate) fn raw(&self) -> &[RawSlot<'a>] {
        &self.raw
    }

    /// [`Self::table`] and, when `with_f64`, the same blocks in `f64`, read
    /// in place.
    pub(crate) fn tables(
        &mut self,
        with_f64: bool,
    ) -> (RectTable<'_, T>, Option<RectTable<'_, f64>>) {
        let Self { raw, casts } = self;
        (storage_table(raw, casts), with_f64.then(|| f64_table(raw)))
    }

    /// The checked `m × m` squares of a query as a store a Gram reads (a
    /// copy in the storage scalar).
    ///
    /// # Errors
    ///
    /// Returns [`GprError::SizeOverflow`] when an ARD square does not fit.
    pub(crate) fn into_square(self, m: usize) -> Result<TrainSources<T>, GprError> {
        TrainSources::from_raw(self.raw, m)
    }

    /// The blocks as a table of views in the storage scalar. An `f64`
    /// model reads the caller's tables in place.
    pub(crate) fn table(&mut self) -> RectTable<'_, T> {
        storage_table(&self.raw, &mut self.casts)
    }
}

fn f64_table<'s>(raw: &'s [RawSlot<'_>]) -> RectTable<'s, f64> {
    RectTable(
        raw.iter()
            .map(|slot| {
                let entry = match slot.shape {
                    SlotShape::Scalar => RectEntry::Scalar(MatRef::from_column_major_slice(
                        slot.block(0),
                        slot.rows,
                        slot.cols,
                    )),
                    SlotShape::Ard(d) => RectEntry::Ard {
                        blocks: (0..d).map(|k| slot.block(k)).collect(),
                        rows: slot.rows,
                        cols: slot.cols,
                    },
                };
                (slot.id, entry)
            })
            .collect(),
    )
}

fn storage_table<'s, T: KernelScalar>(
    raw: &'s [RawSlot<'_>],
    casts: &'s mut [Vec<T::RowCast>],
) -> RectTable<'s, T> {
    let mut entries = Vec::with_capacity(raw.len());
    for (slot, casts) in raw.iter().zip(casts.iter_mut()) {
        let (rows, cols) = (slot.rows, slot.cols);
        let mut views = Vec::with_capacity(casts.len());
        for (k, cast) in casts.iter_mut().enumerate() {
            views.push(T::storage_rows(slot.block(k), cast));
        }
        let entry = match slot.shape {
            SlotShape::Scalar => {
                RectEntry::Scalar(MatRef::from_column_major_slice(views[0], rows, cols))
            }
            SlotShape::Ard(_) => RectEntry::Ard {
                blocks: views,
                rows,
                cols,
            },
        };
        entries.push((slot.id, entry));
    }
    RectTable(entries)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::compiled::supplied::RectSlot;
    use crate::kernel::{ArdDistance, RectSlots, ScalarDistance};

    fn line(scale: f64, rows: std::ops::Range<usize>, cols: std::ops::Range<usize>) -> Vec<f64> {
        cols.flat_map(|j| {
            rows.clone()
                .map(move |i| scale * (i as f64 - j as f64).powi(2))
        })
        .collect()
    }

    /// Two scalar slots and one ARD slot of two points.
    fn store() -> (Vec<DistanceSlot>, TrainSources<f64>) {
        let slots = vec![
            DistanceSlot::Scalar(ScalarDistance::new()),
            DistanceSlot::Scalar(ScalarDistance::new()),
            DistanceSlot::Ard(ArdDistance::new(2).expect("dims")),
        ];
        let sources = slots.iter().enumerate().map(|(k, slot)| match *slot {
            DistanceSlot::Scalar(s) => s.from_vec(line(k as f64 + 1.0, 0..2, 0..2)),
            DistanceSlot::Ard(a) => a.from_vecs(vec![line(4.0, 0..2, 0..2), line(5.0, 0..2, 0..2)]),
        });
        let raw = bind(&slots, sources, 2, 2, BlockKind::Square).expect("bind");
        (slots, TrainSources::from_raw(raw, 2).expect("store"))
    }

    /// The column of the new point `n` to the points `0..n`.
    fn column<'a>(slots: &[DistanceSlot], n: usize) -> QuerySources<'a, f64> {
        let sources = slots.iter().enumerate().map(|(k, slot)| match *slot {
            DistanceSlot::Scalar(s) => s.from_vec(line(k as f64 + 1.0, 0..n, n..n + 1)),
            DistanceSlot::Ard(a) => {
                a.from_vecs(vec![line(4.0, 0..n, n..n + 1), line(5.0, 0..n, n..n + 1)])
            }
        });
        QuerySources::bind(slots, sources, n, 1, BlockKind::Rect).expect("column")
    }

    fn expected(n: usize) -> Vec<(SlotShape, Vec<Vec<f64>>)> {
        vec![
            (SlotShape::Scalar, vec![line(1.0, 0..n, 0..n)]),
            (SlotShape::Scalar, vec![line(2.0, 0..n, 0..n)]),
            (
                SlotShape::Ard(2),
                vec![line(4.0, 0..n, 0..n), line(5.0, 0..n, 0..n)],
            ),
        ]
    }

    #[test]
    fn a_staged_change_applies_only_on_commit() {
        let (slots, mut store) = store();
        let staged = store.stage_append(column(&slots, 2).raw()).expect("stage");
        assert_eq!(store.n(), 2);
        assert_eq!(store.dense_f64(), expected(2));
        store.commit(staged);
        assert_eq!(store.dense_f64(), expected(3));
        let staged = store.stage_append(column(&slots, 3).raw()).expect("stage");
        store.commit(staged);
        assert_eq!(store.dense_f64(), expected(4));
        assert!(store.stage_delete(4).is_err());
        let staged = store.stage_delete(3).expect("stage");
        assert_eq!(store.dense_f64(), expected(4));
        store.commit(staged);
        assert_eq!(store.dense_f64(), expected(3));
    }

    #[test]
    fn a_borrowed_table_is_read_in_place_by_an_f64_model() {
        let image = ScalarDistance::new();
        let cross = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let slots = [DistanceSlot::Scalar(image)];
        let mut bound =
            QuerySources::<f64>::bind(&slots, [image.borrow(&cross)], 3, 2, BlockKind::Rect)
                .expect("bind");
        let table = bound.table();
        let Some(RectSlot::Scalar(view)) = table.rect(slots[0].id()) else {
            panic!("scalar slot");
        };
        assert_eq!(view.as_ptr(), cross.as_ptr());
    }

    #[test]
    fn an_owned_table_moves_into_an_f64_store() {
        let image = ScalarDistance::new();
        let train = vec![0.0, 1.0, 1.0, 0.0];
        let ptr = train.as_ptr();
        let slots = [DistanceSlot::Scalar(image)];
        let raw = bind(&slots, [image.from_vec(train)], 2, 2, BlockKind::Square).expect("bind");
        let store = TrainSources::<f64>::from_raw(raw, 2).expect("store");
        let Some(SquareSlot::Scalar(view)) = store.square(slots[0].id()) else {
            panic!("scalar slot");
        };
        assert_eq!(view.as_ptr(), ptr);
    }
}
