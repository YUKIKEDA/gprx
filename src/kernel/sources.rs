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

use faer::MatRef;

use super::compiled::supplied::{RectEntry, RectTable, SquareSlot, SquareSlots};
use super::dist::ArdSqDiffBuf;
use super::{DistanceFill, DistanceSlot, DistanceSource, KernelScalar, SlotId, SlotShape};
use super::{ScalarOps, SourceData};
use crate::error::GprError;

/// A source's `d²`, checked: `shape.blocks()` dense blocks of `rows × cols`.
pub(crate) struct RawSlot<'a> {
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
            RawData::Blocks(mut blocks) if !blocks.is_empty() => {
                blocks.swap_remove(0).into_owned()
            }
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

/// The fills of a training bind, kept for [`crate::DistanceCachePolicy::Uncached`].
pub(crate) type Fills<'a> = Vec<(SlotId, &'a dyn DistanceFill)>;

/// Binds `sources` to `slots` (the kernel's slots, in order) and checks
/// each block of `rows × cols`. The result follows the order of `slots`.
///
/// # Errors
///
/// Returns [`GprError::LengthMismatch`] for a source of a slot the kernel
/// does not read, a slot without a source, two sources of one slot, or a
/// block of the wrong length or count; [`GprError::EmptyInput`] when `rows`
/// or `cols` is zero; [`GprError::NonFiniteInput`] for a non-finite value;
/// [`GprError::ShapeMismatch`] for a square block with a non-zero diagonal
/// or that is not symmetric.
pub(crate) fn bind<'a>(
    slots: &[DistanceSlot],
    sources: impl IntoIterator<Item = DistanceSource<'a>>,
    rows: usize,
    cols: usize,
    kind: BlockKind,
) -> Result<(Vec<RawSlot<'a>>, Fills<'a>), GprError> {
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
    let mut fills = Vec::new();
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
                fills.push((slot.id(), filler));
                RawData::Packed(all)
            }
        };
        let raw_slot = RawSlot {
            id: slot.id(),
            shape,
            data,
            rows,
            cols,
        };
        check_slot(&raw_slot, blocks, len, kind)?;
        raw.push(raw_slot);
    }
    Ok((raw, fills))
}

fn check_slot(slot: &RawSlot<'_>, blocks: usize, len: usize, kind: BlockKind) -> Result<(), GprError> {
    if let RawData::Blocks(tables) = &slot.data
        && tables.len() != blocks
    {
        return Err(GprError::LengthMismatch {
            reason: format!(
                "expected {blocks} tables of squared distances, got {}",
                tables.len()
            ),
        });
    }
    for k in 0..blocks {
        let block = match &slot.data {
            RawData::Blocks(tables) => &tables[k][..],
            RawData::Packed(all) => &all[k * len..(k + 1) * len],
        };
        check_block(block, slot.rows, slot.cols, kind)?;
    }
    Ok(())
}

/// Checks one `rows × cols` block of `d²`.
pub(crate) fn check_block(
    block: &[f64],
    rows: usize,
    cols: usize,
    kind: BlockKind,
) -> Result<(), GprError> {
    crate::data::require_count(block.len(), rows * cols, "squared distances")?;
    crate::data::require_finite(block)?;
    if kind == BlockKind::Square {
        if rows != cols {
            return Err(GprError::ShapeMismatch {
                reason: format!("a square of squared distances is {rows}x{cols}"),
            });
        }
        for j in 0..cols {
            if block[j + j * rows].abs() > 0.0 {
                return Err(GprError::ShapeMismatch {
                    reason: format!("squared distance ({j}, {j}) is not zero"),
                });
            }
            for i in (j + 1)..rows {
                if (block[i + j * rows] - block[j + i * rows]).abs() > 0.0 {
                    return Err(GprError::ShapeMismatch {
                        reason: format!("squared distances ({i}, {j}) and ({j}, {i}) differ"),
                    });
                }
            }
        }
    }
    Ok(())
}

/// `values` in the storage scalar; an `f64` model takes the buffer as is.
fn into_storage<T: KernelScalar>(values: Vec<f64>) -> Vec<T> {
    let mut slot = Some(values);
    if let Some(same) = (&mut slot as &mut dyn Any).downcast_mut::<Option<Vec<T>>>()
        && let Some(values) = same.take()
    {
        return values;
    }
    slot.map(|values| values.into_iter().map(T::from_f64).collect())
        .unwrap_or_default()
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
                SlotShape::Scalar => TrainData::Scalar(into_storage(slot.into_first())),
                SlotShape::Ard(d) => TrainData::Ard(ArdSqDiffBuf::from_pairs(n, d, |k, i, j| {
                    T::from_f64(slot.block(k)[i + j * n])
                })?),
            };
            slots.push((id, data));
        }
        Ok(Self { n, cap: n, slots })
    }

    /// Writes the training squares of `fills` again (each fill is called once).
    pub(crate) fn refill(&mut self, fills: &[(SlotId, &dyn DistanceFill)]) -> Result<(), GprError> {
        let n = self.n;
        for (id, filler) in fills {
            let Some(at) = self.slots.iter().position(|(slot, _)| slot == id) else {
                continue;
            };
            let blocks = match &self.slots[at].1 {
                TrainData::Scalar(_) => 1,
                TrainData::Ard(cache) => cache.view().d(),
            };
            let shape = if blocks == 1 && matches!(self.slots[at].1, TrainData::Scalar(_)) {
                SlotShape::Scalar
            } else {
                SlotShape::Ard(blocks)
            };
            let mut all = vec![0.0; n * n * blocks];
            filler.fill(n, n, &mut all);
            let raw = RawSlot {
                id: *id,
                shape,
                data: RawData::Packed(all),
                rows: n,
                cols: n,
            };
            check_slot(&raw, blocks, n * n, BlockKind::Square)?;
            let fresh = Self::from_raw(vec![raw], n)?;
            if let Some((_, data)) = fresh.slots.into_iter().next() {
                self.slots[at].1 = data;
            }
        }
        self.cap = n;
        Ok(())
    }

    /// Appends one point: `cols` holds each slot's `n × 1` column to the
    /// existing points, in slot order. The new diagonal is zero.
    pub(crate) fn append(&mut self, cols: &[RawSlot<'_>]) -> Result<(), GprError> {
        if self.slots.is_empty() {
            return Ok(());
        }
        let n = self.n;
        let grow = self.cap < n + 1;
        let new_cap = if grow { (n + 1).max(self.cap.max(1) * 2) } else { self.cap };
        for ((_, data), col) in self.slots.iter_mut().zip(cols) {
            match data {
                TrainData::Scalar(square) => {
                    if grow {
                        let mut wider = vec![T::from_f64(0.0); new_cap * new_cap];
                        for j in 0..n {
                            for i in 0..n {
                                wider[i + j * new_cap] = square[i + j * self.cap];
                            }
                        }
                        *square = wider;
                    }
                    let block = col.block(0);
                    for (i, &value) in block.iter().enumerate() {
                        let v = T::from_f64(value);
                        square[i + n * new_cap] = v;
                        square[n + i * new_cap] = v;
                    }
                    square[n + n * new_cap] = T::from_f64(0.0);
                }
                TrainData::Ard(cache) => {
                    let old = cache.view();
                    let d = old.d();
                    let next = ArdSqDiffBuf::from_pairs(n + 1, d, |k, i, j| {
                        if i == n && j == n {
                            T::from_f64(0.0)
                        } else if i == n {
                            T::from_f64(col.block(k)[j])
                        } else {
                            old.get(k, i, j)
                        }
                    })?;
                    *cache = next;
                }
            }
        }
        self.cap = new_cap;
        self.n = n + 1;
        Ok(())
    }

    /// Removes point `index`.
    pub(crate) fn delete(&mut self, index: usize) -> Result<(), GprError> {
        if self.slots.is_empty() {
            return Ok(());
        }
        let n = self.n;
        if index >= n {
            return Err(GprError::IndexOutOfRange {
                reason: format!("point index {index} is out of range for n={n}"),
            });
        }
        let skip = |i: usize| if i >= index { i + 1 } else { i };
        for (_, data) in &mut self.slots {
            match data {
                TrainData::Scalar(square) => {
                    for j in 0..n - 1 {
                        for i in 0..n - 1 {
                            square[i + j * self.cap] = square[skip(i) + skip(j) * self.cap];
                        }
                    }
                }
                TrainData::Ard(cache) => {
                    let old = cache.view();
                    *cache = ArdSqDiffBuf::from_pairs(n - 1, old.d(), |k, i, j| {
                        old.get(k, skip(i), skip(j))
                    })?;
                }
            }
        }
        self.n = n - 1;
        Ok(())
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
    pub(crate) fn dense_f64(&self) -> Vec<(SlotId, Vec<Vec<f64>>)> {
        let n = self.n;
        self.slots
            .iter()
            .map(|(id, data)| {
                let blocks = match data {
                    TrainData::Scalar(square) => vec![
                        (0..n * n)
                            .map(|at| square[at % n + (at / n) * self.cap].to_f64())
                            .collect(),
                    ],
                    TrainData::Ard(cache) => {
                        let view = cache.view();
                        (0..view.d())
                            .map(|k| {
                                (0..n * n)
                                    .map(|at| view.get(k, at % n, at / n).to_f64())
                                    .collect()
                            })
                            .collect()
                    }
                };
                (*id, blocks)
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

    /// [`Self::gather`] of the training columns `cols` (every row), in `f64`.
    pub(crate) fn columns_f64(&self, cols: std::ops::Range<usize>) -> GatheredRect<f64> {
        let rows: Vec<usize> = (0..self.n).collect();
        let cols: Vec<usize> = cols.collect();
        let gathered = self.gather(&rows, &cols);
        GatheredRect {
            rows: gathered.rows,
            cols: gathered.cols,
            slots: gathered
                .slots
                .into_iter()
                .map(|(id, ard, blocks)| {
                    let blocks = blocks
                        .into_iter()
                        .map(|block| block.into_iter().map(|v| v.to_f64()).collect())
                        .collect();
                    (id, ard, blocks)
                })
                .collect(),
        }
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

impl<T: KernelScalar> SquareSlots<T> for TrainSources<T> {
    fn square(&self, slot: SlotId) -> Option<SquareSlot<'_, T>> {
        let (_, data) = self.slots.iter().find(|(id, _)| *id == slot)?;
        Some(match data {
            TrainData::Scalar(square) => SquareSlot::Scalar(
                MatRef::from_column_major_slice_with_stride(square, self.n, self.n, self.cap.max(1)),
            ),
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
        let (raw, _) = bind(slots, sources, rows, cols, kind)?;
        let casts = raw
            .iter()
            .map(|slot| (0..slot.shape.blocks()).map(|_| T::empty_rows()).collect())
            .collect();
        Ok(Self { raw, casts })
    }

    /// The checked blocks, in slot order.
    pub(crate) fn raw(&self) -> &[RawSlot<'a>] {
        &self.raw
    }

    /// The blocks in `f64`, read in place.
    pub(crate) fn table_f64(&self) -> RectTable<'_, f64> {
        f64_table(&self.raw)
    }

    /// [`Self::table`] and [`Self::table_f64`] together.
    pub(crate) fn tables(&mut self) -> (RectTable<'_, T>, RectTable<'_, f64>) {
        let Self { raw, casts } = self;
        (storage_table(raw, casts), f64_table(raw))
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
    use crate::kernel::{RectSlots, ScalarDistance};
    use crate::kernel::compiled::supplied::RectSlot;

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
        let (raw, _) =
            bind(&slots, [image.from_vec(train)], 2, 2, BlockKind::Square).expect("bind");
        let store = TrainSources::<f64>::from_raw(raw, 2).expect("store");
        let Some(SquareSlot::Scalar(view)) = store.square(slots[0].id()) else {
            panic!("scalar slot");
        };
        assert_eq!(view.as_ptr(), ptr);
    }
}
