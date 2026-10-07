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
use rayon::prelude::*;

use super::compiled::supplied::{RectEntry, RectTable, SquareSlot, SquareSlots};
use super::dist::ArdSqDiffBuf;
use super::{DistanceFill, ScalarOps, SourceData, Tidy};
use super::{DistanceSlot, DistanceSource, KernelScalar, SlotId, SlotShape};
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
    /// One table (a scalar slot).
    Values(Cow<'a, [f64]>),
    /// One owned table per block.
    Blocks(Vec<Vec<f64>>),
    /// One borrowed table per block.
    Slices(&'a [&'a [f64]]),
    /// Every block in one buffer, one after another (a fill).
    Packed(Vec<f64>),
}

impl RawSlot<'_> {
    /// Block `k` (`rows × cols`, column-major).
    pub(crate) fn block(&self, k: usize) -> &[f64] {
        let len = self.rows * self.cols;
        match &self.data {
            RawData::Values(values) => values,
            RawData::Blocks(blocks) => &blocks[k],
            RawData::Slices(blocks) => blocks[k],
            RawData::Packed(all) => &all[k * len..(k + 1) * len],
        }
    }

    /// Takes block 0 as an owned buffer (moved when the source owned it).
    fn into_first(self) -> Vec<f64> {
        match self.data {
            RawData::Values(values) => values.into_owned(),
            RawData::Blocks(mut blocks) if !blocks.is_empty() => blocks.swap_remove(0),
            RawData::Slices(blocks) if !blocks.is_empty() => blocks[0].to_vec(),
            RawData::Packed(all) => all,
            RawData::Blocks(_) | RawData::Slices(_) => {
                // `bind` gives every slot at least one block.
                debug_assert!(false, "a slot bound with no block");
                Vec::new()
            }
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

/// The source of each of `slots` (the kernel's slots, in order), taken
/// from `sources` in any order.
///
/// # Errors
///
/// Returns [`GprError::LengthMismatch`] for a source of a slot the kernel
/// does not read, a slot without a source, or two sources of one slot.
fn by_slot<'a>(
    slots: &[DistanceSlot],
    sources: impl IntoIterator<Item = DistanceSource<'a>>,
) -> Result<Vec<DistanceSource<'a>>, GprError> {
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
    bound
        .into_iter()
        .map(|source| {
            source.ok_or_else(|| GprError::LengthMismatch {
                reason: "a distance slot of the kernel has no source".to_owned(),
            })
        })
        .collect()
}

/// Binds `sources` to `slots` (the kernel's slots, in order) and checks
/// each block of `rows × cols`. The result follows the order of `slots`.
///
/// # Errors
///
/// Returns [`GprError::LengthMismatch`] for a source of a slot the kernel
/// does not read, a slot without a source, two sources of one slot, or a
/// block of the wrong length or count; [`GprError::EmptyInput`] when `rows`
/// or `cols` is zero; [`GprError::InvalidDistance`] for a value the
/// source's check refuses ([`check_block`]). A borrowed table that its
/// source's repair changes is copied.
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
    if kind == BlockKind::Square && rows != cols {
        return Err(GprError::ShapeMismatch {
            reason: format!("a square of squared distances is {rows}x{cols}"),
        });
    }
    let mut raw = Vec::with_capacity(slots.len());
    for (slot, source) in slots.iter().zip(by_slot(slots, sources)?) {
        let shape = slot.shape();
        let blocks = shape.blocks();
        let tidy = source.tidy;
        let data = match source.data {
            SourceData::Values(values) if blocks == 1 => RawData::Values(values),
            SourceData::Blocks(tables) => RawData::Blocks(tables),
            SourceData::Slices(tables) => RawData::Slices(tables),
            SourceData::Values(_) => {
                return Err(GprError::LengthMismatch {
                    reason: format!("expected {blocks} tables of squared distances, got 1"),
                });
            }
            SourceData::Fill(filler) => {
                RawData::Packed(fill_dense(filler, rows, cols, blocks, kind)?)
            }
        };
        let mut raw_slot = RawSlot {
            id: slot.id(),
            shape,
            data,
            rows,
            cols,
        };
        check_slot(&mut raw_slot, blocks, len, kind, tidy)?;
        raw.push(raw_slot);
    }
    Ok(raw)
}

/// The `blocks` dense `rows × cols` blocks a fill writes, one after
/// another. A square asks only the lower triangle and mirrors it.
fn fill_dense(
    filler: &dyn DistanceFill,
    rows: usize,
    cols: usize,
    blocks: usize,
    kind: BlockKind,
) -> Result<Vec<f64>, GprError> {
    let len = rows.checked_mul(cols).ok_or(GprError::SizeOverflow)?;
    let total = len.checked_mul(blocks).ok_or(GprError::SizeOverflow)?;
    let mut all = vec![0.0; total];
    let mut column = vec![0.0; rows.checked_mul(blocks).ok_or(GprError::SizeOverflow)?];
    for col in 0..cols {
        let first = if kind == BlockKind::Square { col } else { 0 };
        let run = rows - first;
        let column = &mut column[..run * blocks];
        filler.fill_column(col, first..rows, column);
        for k in 0..blocks {
            let src = &column[k * run..(k + 1) * run];
            let block = &mut all[k * len..(k + 1) * len];
            block[col * rows + first..(col + 1) * rows].copy_from_slice(src);
            if kind == BlockKind::Square {
                for (i, &v) in (first..rows).zip(src) {
                    block[col + i * rows] = v;
                }
            }
        }
    }
    Ok(all)
}

fn check_slot(
    slot: &mut RawSlot<'_>,
    blocks: usize,
    len: usize,
    kind: BlockKind,
    tidy: Tidy,
) -> Result<(), GprError> {
    let (rows, cols) = (slot.rows, slot.cols);
    let count = match &slot.data {
        RawData::Values(_) => 1,
        RawData::Blocks(tables) => tables.len(),
        RawData::Slices(tables) => tables.len(),
        RawData::Packed(_) => blocks,
    };
    if count != blocks {
        return Err(GprError::LengthMismatch {
            reason: format!("expected {blocks} tables of squared distances, got {count}"),
        });
    }
    for k in 0..blocks {
        crate::data::require_count(slot.block(k).len(), len, "squared distances")?;
        if check_block(slot.block(k), rows, cols, kind, tidy)? {
            // A borrowed table is copied only when its repair changes it.
            let block: &mut [f64] = match &mut slot.data {
                RawData::Values(values) => values.to_mut(),
                RawData::Blocks(tables) => &mut tables[k],
                RawData::Slices(tables) => {
                    let owned: Vec<Vec<f64>> = tables.iter().map(|t| t.to_vec()).collect();
                    slot.data = RawData::Blocks(owned);
                    let RawData::Blocks(tables) = &mut slot.data else {
                        return Err(GprError::LengthMismatch {
                            reason: "internal: a copied table was lost".to_owned(),
                        });
                    };
                    &mut tables[k]
                }
                RawData::Packed(all) => &mut all[k * len..(k + 1) * len],
            };
            repair_block(block, rows, cols, kind);
        }
    }
    Ok(())
}

/// An invalid pair `(row, col)` of a table.
fn invalid(row: usize, col: usize, reason: impl Into<String>) -> GprError {
    GprError::InvalidDistance {
        row,
        col,
        reason: reason.into(),
    }
}

/// Whether `v` is a valid squared distance: finite and non-negative.
/// Both bounds are tested without a branch (`&`, not `contains`), so a fold
/// of it stays vectorized.
#[inline]
#[allow(clippy::manual_range_contains)]
fn valid(v: f64) -> bool {
    (v >= 0.0) & (v <= f64::MAX)
}

/// Whether two supplied values are the same number: their difference is
/// exactly zero, which for finite values holds only when they are equal
/// (`0.0` and `-0.0` included). A `NaN` is never the same as anything.
#[inline]
fn same(a: f64, b: f64) -> bool {
    a - b == 0.0
}

/// The first value of a `rows`-row block that is not [`valid`].
fn first_invalid(block: &[f64], rows: usize) -> GprError {
    let rows = rows.max(1);
    let at = block.iter().position(|&v| !valid(v)).unwrap_or(0);
    let v = block.get(at).copied().unwrap_or(0.0);
    let reason = if v.is_finite() {
        format!("{v} is negative")
    } else {
        format!("{v} is not finite")
    };
    invalid(at % rows, at / rows, reason)
}

/// Checks one `rows × cols` block of `d²` against its source's check, and
/// returns whether [`repair_block`] has values to fix.
///
/// [`Tidy::Exact`] asks every value to be finite and non-negative and, for
/// a square, a zero diagonal and equal mirror entries, and never repairs.
/// [`Tidy::Within`] allows a negative value, a non-zero diagonal, and a
/// mirror gap up to its tolerance times the largest value of the block,
/// and reports them for repair.
///
/// # Errors
///
/// Returns [`GprError::InvalidDistance`] for the first violation past what
/// the check allows.
pub(crate) fn check_block(
    block: &[f64],
    rows: usize,
    cols: usize,
    kind: BlockKind,
    tidy: Tidy,
) -> Result<bool, GprError> {
    match tidy {
        Tidy::Exact => exact_block(block, rows, cols, kind).map(|()| false),
        Tidy::Within(rel) => within_block(block, rows, cols, kind, rel),
    }
}

/// [`check_block`] for [`Tidy::Exact`]. The values are folded without a
/// branch; a violation is located only once the fold has found one.
fn exact_block(block: &[f64], rows: usize, cols: usize, kind: BlockKind) -> Result<(), GprError> {
    if !block.iter().fold(true, |ok, &v| ok & valid(v)) {
        return Err(first_invalid(block, rows));
    }
    if kind == BlockKind::Square {
        for j in 0..cols {
            let diag = block[j + j * rows];
            if diag != 0.0 {
                return Err(invalid(j, j, format!("the diagonal is {diag}, not zero")));
            }
        }
        let bands: Vec<usize> = (0..rows).step_by(BAND).collect();
        let symmetric = bands
            .into_par_iter()
            .all(|j0| check_band(block, rows, j0, (j0 + BAND).min(rows), |_| {}));
        if !symmetric {
            let mut found = Ok(());
            let _ = for_each_lower_pair(rows, |i, j| {
                let (a, b) = (block[i + j * rows], block[j + i * rows]);
                if !same(a, b) {
                    found = Err(invalid(
                        i,
                        j,
                        format!("{a} differs from its mirror ({j}, {i}), {b}"),
                    ));
                    return Err(());
                }
                Ok(())
            });
            found?;
        }
    }
    Ok(())
}

/// Columns of one band of [`check_band`]: the mirror of the band is read
/// one row at a time, `BAND` contiguous values (a cache line of `f64`).
const BAND: usize = 8;

/// Checks the columns `j0..j1` (`j1 − j0 ≤` [`BAND`]) of the `n × n`
/// square `block` exactly, below and on the diagonal, and hands each lower
/// run (rows `j..n` of column `j`) to `run`: values finite and
/// non-negative, a zero diagonal, and each entry equal to its mirror. The
/// band's mirror is read row by row, contiguous, while its `BAND` columns
/// are streamed, so no pair is a lone strided load.
fn check_band(block: &[f64], n: usize, j0: usize, j1: usize, mut run: impl FnMut(&[f64])) -> bool {
    let mut ok = true;
    for j in j0..j1 {
        let lower = &block[j * n + j..(j + 1) * n];
        ok &= lower.iter().fold(lower[0] == 0.0, |ok, &v| ok & valid(v));
        run(lower);
        for i in j + 1..j1 {
            ok &= same(block[i + j * n], block[j + i * n]);
        }
    }
    if j1 - j0 == BAND {
        let cols: [&[f64]; BAND] = std::array::from_fn(|c| &block[(j0 + c) * n..(j0 + c + 1) * n]);
        for i in j1..n {
            let row = &block[i * n + j0..i * n + j1];
            ok &= (0..BAND).fold(true, |ok, c| ok & same(cols[c][i], row[c]));
        }
    } else {
        for j in j0..j1 {
            for i in j1..n {
                ok &= same(block[i + j * n], block[j + i * n]);
            }
        }
    }
    ok
}

/// Checks the `d` dense `n × n` training squares of an ARD slot exactly
/// (`block(k)` is dimension `k`) and packs their lower triangles, reading
/// each square once, band by band ([`check_band`]) on the Rayon pool, as
/// the coordinate path fills its `(Δx_d)²` cache. A band's columns are one
/// contiguous range of the packed triangle. A violation is located with
/// [`exact_block`].
///
/// # Errors
///
/// Returns [`GprError::InvalidDistance`] for the first violation and
/// [`GprError::SizeOverflow`] when the packed size does not fit.
fn pack_exact_ard<'b, T: KernelScalar>(
    n: usize,
    d: usize,
    block: impl Fn(usize) -> &'b [f64] + Sync,
) -> Result<ArdSqDiffBuf<T>, GprError> {
    let per_dim = n
        .checked_add(1)
        .and_then(|n1| n.checked_mul(n1))
        .map(|cells| cells / 2)
        .ok_or(GprError::SizeOverflow)?;
    let len = per_dim.checked_mul(d).ok_or(GprError::SizeOverflow)?;
    let mut data = vec![T::from_f64(0.0); len];
    let mut bands: Vec<(usize, usize, usize, &mut [T])> = Vec::new();
    let mut rest = data.as_mut_slice();
    for k in 0..d {
        for j0 in (0..n).step_by(BAND) {
            let j1 = (j0 + BAND).min(n);
            let size = (j0..j1).map(|j| n - j).sum();
            let (head, tail) = rest.split_at_mut(size);
            bands.push((k, j0, j1, head));
            rest = tail;
        }
    }
    let ok = bands.into_par_iter().all(|(k, j0, j1, dest)| {
        let mut at = 0;
        check_band(block(k), n, j0, j1, |lower| {
            for (slot, &v) in dest[at..at + lower.len()].iter_mut().zip(lower) {
                *slot = T::from_f64(v);
            }
            at += lower.len();
        })
    });
    if !ok {
        for k in 0..d {
            exact_block(block(k), n, n, BlockKind::Square)?;
        }
        return Err(invalid(0, 0, "a band of the square failed its check"));
    }
    Ok(ArdSqDiffBuf::from_packed(data, n, d))
}

/// [`check_block`] for [`Tidy::Within`].
fn within_block(
    block: &[f64],
    rows: usize,
    cols: usize,
    kind: BlockKind,
    rel: f64,
) -> Result<bool, GprError> {
    if !block.iter().fold(true, |ok, &v| ok & v.is_finite()) {
        let at = block.iter().position(|v| !v.is_finite()).unwrap_or(0);
        let rows = rows.max(1);
        return Err(invalid(at % rows, at / rows, "is not finite"));
    }
    let tol = rel * block.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
    let mut repair = false;
    for (at, &v) in block.iter().enumerate() {
        if v < -tol {
            let rows = rows.max(1);
            return Err(invalid(
                at % rows,
                at / rows,
                format!("{v} is negative past the tolerance {tol}"),
            ));
        }
        repair |= v < 0.0;
    }
    if kind == BlockKind::Square {
        for j in 0..cols {
            let diag = block[j + j * rows];
            if diag.abs() > tol {
                return Err(invalid(
                    j,
                    j,
                    format!("the diagonal is {diag}, past the tolerance {tol}"),
                ));
            }
            repair |= diag != 0.0;
        }
        for_each_lower_pair(rows, |i, j| {
            let (a, b) = (block[i + j * rows], block[j + i * rows]);
            let gap = (a - b).abs();
            if gap > tol {
                return Err(invalid(
                    i,
                    j,
                    format!(
                        "{a} differs from its mirror ({j}, {i}), {b}, past the tolerance {tol}"
                    ),
                ));
            }
            repair |= gap > 0.0;
            Ok(())
        })?;
    }
    Ok(repair)
}

/// Side of the square tiles [`for_each_lower_pair`] walks.
const PAIR_TILE: usize = 64;

/// Visits each pair `i > j` of an `n × n` column-major square tile by tile,
/// so the mirror `(j, i)` (a row of the square) is read while its tile is
/// still in cache, not one strided load per pair.
fn for_each_lower_pair<E>(
    n: usize,
    mut visit: impl FnMut(usize, usize) -> Result<(), E>,
) -> Result<(), E> {
    for j0 in (0..n).step_by(PAIR_TILE) {
        let j1 = (j0 + PAIR_TILE).min(n);
        for i0 in (j0..n).step_by(PAIR_TILE) {
            let i1 = (i0 + PAIR_TILE).min(n);
            for j in j0..j1 {
                for i in i0.max(j + 1)..i1 {
                    visit(i, j)?;
                }
            }
        }
    }
    Ok(())
}

/// Fixes what [`within_block`] accepted: negative values to zero and, for a
/// square, a zero diagonal and each mirror pair set to its mean.
fn repair_block(block: &mut [f64], rows: usize, cols: usize, kind: BlockKind) {
    for v in block.iter_mut() {
        *v = v.max(0.0);
    }
    if kind == BlockKind::Square {
        for j in 0..cols {
            block[j + j * rows] = 0.0;
        }
        let walked = for_each_lower_pair::<std::convert::Infallible>(rows, |i, j| {
            let mean = 0.5 * (block[i + j * rows] + block[j + i * rows]);
            block[i + j * rows] = mean;
            block[j + i * rows] = mean;
            Ok(())
        });
        if let Err(never) = walked {
            match never {}
        }
    }
}

/// What a fill of a training square has met, for [`Tidy::Within`]: the
/// largest value, and the worst negative value and diagonal with their
/// pairs. Judged once the whole triangle is in, against its largest value.
#[derive(Default)]
struct FillRounding {
    max: f64,
    negative: Option<(f64, usize, usize)>,
    diagonal: Option<(f64, usize, usize)>,
}

impl FillRounding {
    /// Notes `v` at `(row, col)` and returns what is stored: `v`, or `0.0`
    /// for a negative value or a diagonal.
    fn note(&mut self, v: f64, row: usize, col: usize) -> f64 {
        self.max = self.max.max(v.abs());
        if row == col {
            if self.diagonal.is_none_or(|(worst, _, _)| v.abs() > worst) && v != 0.0 {
                self.diagonal = Some((v.abs(), row, col));
            }
            return 0.0;
        }
        if v < 0.0 {
            if self.negative.is_none_or(|(worst, _, _)| -v > worst) {
                self.negative = Some((-v, row, col));
            }
            return 0.0;
        }
        v
    }

    fn judge(&self, rel: f64) -> Result<(), GprError> {
        let tol = rel * self.max;
        if let Some((v, row, col)) = self.negative.filter(|(v, _, _)| *v > tol) {
            return Err(invalid(
                row,
                col,
                format!("{} is negative past the tolerance {tol}", -v),
            ));
        }
        if let Some((v, row, col)) = self.diagonal.filter(|(v, _, _)| *v > tol) {
            return Err(invalid(
                row,
                col,
                format!("the diagonal is {v} in size, past the tolerance {tol}"),
            ));
        }
        Ok(())
    }
}

fn tables_mismatch(expected: usize, got: usize) -> GprError {
    GprError::LengthMismatch {
        reason: format!("expected {expected} tables of squared distances, got {got}"),
    }
}

fn require_tables(got: usize, expected: usize) -> Result<(), GprError> {
    if got == expected {
        Ok(())
    } else {
        Err(tables_mismatch(expected, got))
    }
}

/// A scalar training square of `n` points from a fill: the lower triangle,
/// column by column, mirrored into the dense square.
fn fill_scalar_square(
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
    for col in 0..n {
        for row in col + 1..n {
            square[col + row * n] = square[row + col * n];
        }
    }
    Ok(square)
}

/// An ARD training square of `n` points and `d` dimensions from a fill:
/// each column run of the lower triangle goes straight into its packed
/// place through one reused `d · n` buffer.
fn fill_ard_square<T: KernelScalar>(
    filler: &dyn DistanceFill,
    n: usize,
    d: usize,
    tidy: Tidy,
) -> Result<ArdSqDiffBuf<T>, GprError> {
    let mut packed = ArdSqDiffBuf::<T>::zeros(n, d)?;
    let mut buffer = vec![0.0; n.checked_mul(d).ok_or(GprError::SizeOverflow)?];
    let mut rounding: Vec<FillRounding> = (0..d).map(|_| FillRounding::default()).collect();
    for col in 0..n {
        let len = n - col;
        let runs = &mut buffer[..len * d];
        filler.fill_column(col, col..n, runs);
        for (k, rounding) in rounding.iter_mut().enumerate() {
            let column = packed.column_mut(k, col);
            fill_run(
                &runs[k * len..(k + 1) * len],
                col,
                tidy,
                rounding,
                |at, v| {
                    column[at] = T::from_f64(v);
                },
            )?;
        }
    }
    if let Tidy::Within(rel) = tidy {
        for rounding in &rounding {
            rounding.judge(rel)?;
        }
    }
    Ok(packed)
}

/// Checks one column run of a training square a fill wrote (rows
/// `col..n`): [`Tidy::Exact`] refuses at once; [`Tidy::Within`] notes into
/// `rounding` and stores the repaired values in `out`.
fn fill_run(
    run: &[f64],
    col: usize,
    tidy: Tidy,
    rounding: &mut FillRounding,
    mut store: impl FnMut(usize, f64),
) -> Result<(), GprError> {
    match tidy {
        Tidy::Exact => {
            if !run.iter().fold(true, |ok, &v| ok & valid(v)) {
                let at = run.iter().position(|&v| !valid(v)).unwrap_or(0);
                let v = run[at];
                let reason = if v.is_finite() {
                    format!("{v} is negative")
                } else {
                    format!("{v} is not finite")
                };
                return Err(invalid(col + at, col, reason));
            }
            if let Some(&diag) = run.first()
                && diag != 0.0
            {
                return Err(invalid(
                    col,
                    col,
                    format!("the diagonal is {diag}, not zero"),
                ));
            }
            for (at, &v) in run.iter().enumerate() {
                store(at, v);
            }
        }
        Tidy::Within(_) => {
            if let Some(at) = run.iter().position(|v| !v.is_finite()) {
                return Err(invalid(col + at, col, "is not finite"));
            }
            for (at, &v) in run.iter().enumerate() {
                store(at, rounding.note(v, col + at, col));
            }
        }
    }
    Ok(())
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

    #[cfg(test)]
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
                SlotShape::Ard(d) => {
                    TrainData::Ard(ArdSqDiffBuf::from_dense(n, d, |k| slot.block(k))?)
                }
            };
            slots.push((id, data));
        }
        Ok(Self { n, cap: n, slots })
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
    /// The errors of [`bind`] for a square.
    pub(crate) fn bind<'a>(
        slots: &[DistanceSlot],
        sources: impl IntoIterator<Item = DistanceSource<'a>>,
        n: usize,
    ) -> Result<Self, GprError> {
        crate::data::require_nonempty(n)?;
        let len = n.checked_mul(n).ok_or(GprError::SizeOverflow)?;
        let mut out = Vec::with_capacity(slots.len());
        for (slot, source) in slots.iter().zip(by_slot(slots, sources)?) {
            let tidy = source.tidy;
            let blocks = slot.shape().blocks();
            let data = match (slot.shape(), source.data) {
                (SlotShape::Scalar, SourceData::Values(values)) => {
                    crate::data::require_count(values.len(), len, "squared distances")?;
                    let mut values = values;
                    if check_block(&values, n, n, BlockKind::Square, tidy)? {
                        repair_block(values.to_mut(), n, n, BlockKind::Square);
                    }
                    TrainData::Scalar(T::vec_from_f64(values.into_owned()))
                }
                (SlotShape::Scalar, SourceData::Fill(filler)) => {
                    TrainData::Scalar(T::vec_from_f64(fill_scalar_square(filler, n, tidy)?))
                }
                (SlotShape::Ard(d), SourceData::Blocks(tables)) if tidy == Tidy::Exact => {
                    require_tables(tables.len(), d)?;
                    for table in &tables {
                        crate::data::require_count(table.len(), len, "squared distances")?;
                    }
                    TrainData::Ard(pack_exact_ard(n, d, |k| &tables[k])?)
                }
                (SlotShape::Ard(d), SourceData::Slices(tables)) if tidy == Tidy::Exact => {
                    require_tables(tables.len(), d)?;
                    for table in tables {
                        crate::data::require_count(table.len(), len, "squared distances")?;
                    }
                    TrainData::Ard(pack_exact_ard(n, d, |k| tables[k])?)
                }
                (SlotShape::Ard(d), SourceData::Blocks(mut tables)) => {
                    require_tables(tables.len(), d)?;
                    for table in &mut tables {
                        crate::data::require_count(table.len(), len, "squared distances")?;
                        if check_block(table, n, n, BlockKind::Square, tidy)? {
                            repair_block(table, n, n, BlockKind::Square);
                        }
                    }
                    TrainData::Ard(ArdSqDiffBuf::from_dense(n, d, |k| &tables[k])?)
                }
                (SlotShape::Ard(d), SourceData::Slices(tables)) => {
                    require_tables(tables.len(), d)?;
                    let mut repaired: Vec<Option<Vec<f64>>> = (0..d).map(|_| None).collect();
                    for (k, table) in tables.iter().enumerate() {
                        crate::data::require_count(table.len(), len, "squared distances")?;
                        if check_block(table, n, n, BlockKind::Square, tidy)? {
                            let mut copy = table.to_vec();
                            repair_block(&mut copy, n, n, BlockKind::Square);
                            repaired[k] = Some(copy);
                        }
                    }
                    TrainData::Ard(ArdSqDiffBuf::from_dense(n, d, |k| {
                        repaired[k].as_deref().unwrap_or(tables[k])
                    })?)
                }
                (SlotShape::Ard(d), SourceData::Fill(filler)) => {
                    TrainData::Ard(fill_ard_square(filler, n, d, tidy)?)
                }
                (_, SourceData::Values(_)) => return Err(tables_mismatch(blocks, 1)),
                (_, SourceData::Blocks(tables)) => {
                    return Err(tables_mismatch(blocks, tables.len()));
                }
                (_, SourceData::Slices(tables)) => {
                    return Err(tables_mismatch(blocks, tables.len()));
                }
            };
            out.push((slot.id(), data));
        }
        Ok(Self {
            n,
            cap: n,
            slots: out,
        })
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
            // The caller binds the columns against the store's own slots,
            // so this is a crate bug, not the caller's data.
            debug_assert!(
                false,
                "the bound columns are not the store's slots in order"
            );
            return Err(GprError::LengthMismatch {
                reason: "internal: the bound columns are not the store's slots in order".to_owned(),
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

    #[cfg(test)]
    /// Each slot's blocks in `f64`, dense `n × n` and one after another in
    /// one buffer, in slot order (saving).
    pub(crate) fn dense_f64(&self) -> Vec<(SlotShape, Vec<f64>)> {
        let n = self.n;
        self.slots
            .iter()
            .map(|(_, data)| match data {
                TrainData::Scalar(square) => (
                    SlotShape::Scalar,
                    (0..n * n)
                        .map(|at| square[at % n + (at / n) * self.cap].to_f64())
                        .collect(),
                ),
                TrainData::Ard(cache) => {
                    let view = cache.view();
                    let values = (0..view.d())
                        .flat_map(|k| {
                            (0..n * n).map(move |at| view.get(k, at % n, at / n).to_f64())
                        })
                        .collect();
                    (SlotShape::Ard(view.d()), values)
                }
            })
            .collect()
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

    fn bind<'a>(
        slots: &[DistanceSlot],
        sources: impl IntoIterator<Item = DistanceSource<'a>>,
        n: usize,
    ) -> Result<Self, GprError> {
        let exact = TrainSources::<f64>::bind(slots, sources, n)?;
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

    #[test]
    fn the_tiled_walk_visits_every_pair_below_the_diagonal_once() {
        for n in [0, 1, 63, 64, 65, 150] {
            let mut seen = vec![0_u8; n * n];
            let walked = for_each_lower_pair::<()>(n, |i, j| {
                seen[i + j * n] += 1;
                Ok(())
            });
            assert!(walked.is_ok());
            for j in 0..n {
                for i in 0..n {
                    assert_eq!(seen[i + j * n], u8::from(i > j), "n={n} ({i}, {j})");
                }
            }
        }
    }

    #[test]
    fn a_mirror_pair_far_from_the_first_tile_is_checked_and_repaired() {
        let n = 150;
        let mut block = line(1.0, 0..n, 0..n);
        assert_eq!(
            check_block(&block, n, n, BlockKind::Square, Tidy::Exact),
            Ok(false)
        );
        // Any gap is refused by the exact check, at the pair in a tile off
        // the diagonal.
        block[140 + 3 * n] += 1e-9;
        let refused = check_block(&block, n, n, BlockKind::Square, Tidy::Exact);
        assert!(
            matches!(
                &refused,
                Err(GprError::InvalidDistance {
                    row: 140,
                    col: 3,
                    ..
                })
            ),
            "{refused:?}"
        );
        // Within a repair's tolerance it is set to the pair's mean.
        let within = Tidy::Within(1e-6);
        assert_eq!(
            check_block(&block, n, n, BlockKind::Square, within),
            Ok(true)
        );
        repair_block(&mut block, n, n, BlockKind::Square);
        assert_eq!(block[140 + 3 * n].to_bits(), block[3 + 140 * n].to_bits());
        // Past it, refused.
        block[140 + 3 * n] += 1.0;
        assert!(matches!(
            check_block(&block, n, n, BlockKind::Square, within),
            Err(GprError::InvalidDistance {
                row: 140,
                col: 3,
                ..
            })
        ));
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

    fn expected(n: usize) -> Vec<(SlotShape, Vec<f64>)> {
        vec![
            (SlotShape::Scalar, line(1.0, 0..n, 0..n)),
            (SlotShape::Scalar, line(2.0, 0..n, 0..n)),
            (
                SlotShape::Ard(2),
                [line(4.0, 0..n, 0..n), line(5.0, 0..n, 0..n)].concat(),
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

    /// Every slot's `d²` of the points at `at` on the line, as `expected`.
    fn expected_at(at: &[f64]) -> Vec<(SlotShape, Vec<f64>)> {
        let sq = |scale: f64| -> Vec<f64> {
            at.iter()
                .flat_map(|&b| at.iter().map(move |&a| scale * (a - b).powi(2)))
                .collect()
        };
        vec![
            (SlotShape::Scalar, sq(1.0)),
            (SlotShape::Scalar, sq(2.0)),
            (SlotShape::Ard(2), [sq(4.0), sq(5.0)].concat()),
        ]
    }

    /// The column of a new point at `p` to the points at `at`.
    fn column_at<'a>(slots: &[DistanceSlot], at: &[f64], p: f64) -> QuerySources<'a, f64> {
        let col =
            |scale: f64| -> Vec<f64> { at.iter().map(|&a| scale * (a - p).powi(2)).collect() };
        let sources = slots.iter().enumerate().map(|(k, slot)| match *slot {
            DistanceSlot::Scalar(s) => s.from_vec(col(k as f64 + 1.0)),
            DistanceSlot::Ard(a) => a.from_vecs(vec![col(4.0), col(5.0)]),
        });
        QuerySources::bind(slots, sources, at.len(), 1, BlockKind::Rect).expect("column")
    }

    #[test]
    fn deleting_the_first_a_middle_or_the_last_point_then_appending_matches_a_rebuild() {
        for index in [0, 2, 4] {
            let (slots, mut store) = store();
            let mut at = vec![0.0, 1.0];
            // Five points: the leading dimension grows past `n`.
            for p in [2.0, 3.0, 4.0] {
                let staged = store
                    .stage_append(column_at(&slots, &at, p).raw())
                    .expect("stage");
                store.commit(staged);
                at.push(p);
            }
            let staged = store.stage_delete(index).expect("stage");
            store.commit(staged);
            at.remove(index);
            assert_eq!(store.dense_f64(), expected_at(&at), "delete {index}");
            // Appending writes over the rows the delete left behind.
            for p in [7.5, 9.0] {
                let staged = store
                    .stage_append(column_at(&slots, &at, p).raw())
                    .expect("stage");
                store.commit(staged);
                at.push(p);
                assert_eq!(store.dense_f64(), expected_at(&at), "append after {index}");
            }
        }
    }

    #[test]
    fn deleting_down_to_one_point_then_appending_matches_a_rebuild() {
        let (slots, mut store) = store();
        let staged = store.stage_delete(0).expect("stage");
        store.commit(staged);
        assert_eq!(store.dense_f64(), expected_at(&[1.0]));
        let staged = store
            .stage_append(column_at(&slots, &[1.0], 3.0).raw())
            .expect("stage");
        store.commit(staged);
        assert_eq!(store.dense_f64(), expected_at(&[1.0, 3.0]));
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
