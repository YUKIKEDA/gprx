//! The supplied squared distances a model reads: binding and checking the
//! caller's sources, the training `d²` a model owns, and the per-call blocks
//! of a prediction.
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

use super::compiled::supplied::{ArdRect, RectSlot, RectSlots, SquareSlot, SquareSlots};
use super::dist::{ArdBlocks, ArdSqDiffBuf, BlockList, Checked};
use super::{DistanceFill, ScalarOps, SourceData, Tidy};
use super::{DistanceSlot, DistanceSource, KernelScalar, SlotId, SlotShape};
use crate::error::GprError;

/// A source's `d²`, checked: `shape.blocks()` dense blocks of `rows × cols`.
pub(crate) struct RawSlot<'a> {
    pub(crate) id: SlotId,
    pub(crate) shape: SlotShape,
    data: RawData<'a>,
    rows: usize,
    cols: usize,
    /// Where the blocks start in [`QueryScratch`]'s casts (an `f32` model).
    cast_at: usize,
    /// Whether the values are left to be checked as they are read.
    unchecked: bool,
}

enum RawData<'a> {
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

/// What a block must satisfy beyond its length and finite values.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BlockKind {
    /// Pairs of one set: zero diagonal, symmetric.
    Square,
    /// Pairs of two sets.
    Rect,
}

/// The slot of `slots` (the kernel's) that `source` is for; `bound` are
/// the slots that already have a source.
///
/// # Errors
///
/// Returns [`GprError::LengthMismatch`] for a source of a slot the kernel
/// does not read, or a second source of one slot.
fn slot_of<'k>(
    slots: &'k [DistanceSlot],
    source: &DistanceSource<'_>,
    mut bound: impl Iterator<Item = SlotId>,
) -> Result<&'k DistanceSlot, GprError> {
    let Some(slot) = slots.iter().find(|slot| slot.id() == source.slot) else {
        return Err(GprError::LengthMismatch {
            reason: "squared distances were supplied for a slot the kernel does not read"
                .to_owned(),
        });
    };
    if bound.any(|id| id == source.slot) {
        return Err(GprError::LengthMismatch {
            reason: "two sources were supplied for one distance slot".to_owned(),
        });
    }
    Ok(slot)
}

/// A slot of the kernel was given no source (every source names a distinct
/// slot of the kernel, and fewer sources than slots arrived).
fn no_source() -> GprError {
    GprError::LengthMismatch {
        reason: "a distance slot of the kernel has no source".to_owned(),
    }
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
pub(crate) fn valid(v: f64) -> bool {
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
    first_invalid_from(block, rows, 0)
}

/// [`first_invalid`] of a block whose first column is column `col0` of
/// the caller's table.
pub(crate) fn first_invalid_from(block: &[f64], rows: usize, col0: usize) -> GprError {
    let rows = rows.max(1);
    let at = block.iter().position(|&v| !valid(v)).unwrap_or(0);
    let v = block.get(at).copied().unwrap_or(0.0);
    invalid_value(v, at % rows, col0 + at / rows)
}

/// The error of a value at `(row, col)` that is finite in `f64` but past
/// the range of the model's narrower storage scalar.
fn out_of_range(row: usize, col: usize) -> GprError {
    invalid(row, col, "is past the range of the model's storage scalar")
}

/// The error of a negative value `v` at `(row, col)` past a repair's
/// tolerance `tol`.
fn negative_past(v: f64, tol: f64, row: usize, col: usize) -> GprError {
    invalid(
        row,
        col,
        format!("{v} is negative past the tolerance {tol}"),
    )
}

/// The error of a value `v` at `(row, col)` that is not [`valid`].
pub(crate) fn invalid_value(v: f64, row: usize, col: usize) -> GprError {
    let reason = if v.is_finite() {
        format!("{v} is negative")
    } else {
        format!("{v} is not finite")
    };
    invalid(row, col, reason)
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
    if kind == BlockKind::Rect {
        return if super::simd::all_valid_distances(block) {
            Ok(())
        } else {
            Err(first_invalid(block, rows))
        };
    }
    // A square: [`check_band`] reads each value once and checks the lower
    // triangle's values, the diagonal, and each mirror, so an invalid value
    // above the diagonal fails its mirror. The violation is located only
    // once a band has failed.
    let bands = rows.div_ceil(BAND);
    let band = |band: usize| {
        let j0 = band * BAND;
        check_band(block, rows, j0, (j0 + BAND).min(rows), |_| {})
    };
    let ok = if rows < PAR_ROWS {
        (0..bands).all(band)
    } else {
        (0..bands).into_par_iter().all(band)
    };
    if ok {
        return Ok(());
    }
    if !super::simd::all_valid_distances(block) {
        return Err(first_invalid(block, rows));
    }
    for j in 0..cols {
        let diag = block[j + j * rows];
        if diag != 0.0 {
            return Err(invalid(j, j, format!("the diagonal is {diag}, not zero")));
        }
    }
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
    // A band failed, yet the scan that locates violations found none: the
    // two checks disagree. Refuse the table rather than accept it.
    Err(invalid(0, 0, "a band of the square failed its check"))
}

/// Rows below which a square is checked on the calling thread: smaller
/// squares cost less than handing their bands to the pool.
const PAR_ROWS: usize = 256;

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
    let ok = data
        .par_chunks_mut(per_dim.max(1))
        .enumerate()
        .all(|(k, dest)| pack_bands(block(k), n, 0..n.div_ceil(BAND), dest));
    if !ok {
        for k in 0..d {
            exact_block(block(k), n, n, BlockKind::Square)?;
        }
        return Err(invalid(0, 0, "a band of the square failed its check"));
    }
    Ok(ArdSqDiffBuf::from_packed(data, n, d))
}

/// Checks and packs the bands `bands` of one square into `dest` (their
/// columns of the packed triangle, in order), halving the range on the
/// Rayon pool down to one band, so no list of bands is made.
fn pack_bands<T: KernelScalar>(
    block: &[f64],
    n: usize,
    bands: std::ops::Range<usize>,
    dest: &mut [T],
) -> bool {
    let (j0, j1) = (bands.start * BAND, (bands.end * BAND).min(n));
    if bands.len() <= 1 {
        let mut at = 0;
        return j0 >= j1
            || check_band(block, n, j0, j1, |lower| {
                for (slot, &v) in dest[at..at + lower.len()].iter_mut().zip(lower) {
                    *slot = T::from_f64(v);
                }
                at += lower.len();
            });
    }
    let mid = bands.start + bands.len() / 2;
    // Columns `j0..mid · BAND` hold `n − j` values each.
    let split = (j0..mid * BAND).map(|j| n - j).sum();
    let (head, tail) = dest.split_at_mut(split);
    let (a, b) = rayon::join(
        || pack_bands(block, n, bands.start..mid, head),
        || pack_bands(block, n, mid..bands.end, tail),
    );
    a & b
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
        return Err(invalid_value(block[at], at % rows, at / rows));
    }
    let tol = rel * block.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
    let mut repair = false;
    for (at, &v) in block.iter().enumerate() {
        if v < -tol {
            let rows = rows.max(1);
            return Err(negative_past(v, tol, at % rows, at / rows));
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
            return Err(negative_past(-v, tol, row, col));
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
    mirror_lower(&mut square, n);
    Ok(square)
}

/// Side of the tiles [`mirror_lower`] copies: a source tile and its
/// destination tile fit in L1 together.
const MIRROR_TILE: usize = 32;

/// Copies the strict lower triangle of the column-major `n × n` `square`
/// onto its upper triangle, a tile at a time: each destination row of a
/// tile is written contiguously while the tile's source columns are in
/// cache, not one strided store per pair.
fn mirror_lower(square: &mut [f64], n: usize) {
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
                return Err(invalid_value(run[at], col + at, col));
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
                return Err(invalid_value(run[at], col + at, col));
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

/// Columns `start..start + len` of a [`TrainSources`], every row, as
/// rectangular blocks ([`TrainSources::columns`]).
pub(crate) struct TrainColumns<'s, T> {
    store: &'s TrainSources<T>,
    start: usize,
    len: usize,
}

impl<T: KernelScalar> RectSlots<T> for TrainColumns<'_, T> {
    fn rect(&self, slot: SlotId) -> Option<RectSlot<'_, T>> {
        let store = self.store;
        let (_, data) = store.slots.iter().find(|(id, _)| *id == slot)?;
        let (n, cap) = (store.n, store.cap.max(1));
        Some(match data {
            TrainData::Scalar(square) => {
                RectSlot::Scalar(MatRef::from_column_major_slice_with_stride(
                    square.get(self.start * cap..)?,
                    n,
                    self.len,
                    cap,
                ))
            }
            TrainData::Ard(cache) => RectSlot::Ard(ArdRect::Checked(ArdBlocks::new(
                BlockList::Triangles(cache.view()),
                n,
                self.len,
                self.start,
            ))),
        })
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
            slots: Vec::new(),
        }
    }

    /// Whether the kernel reads any supplied distances.
    pub(crate) fn is_empty(&self) -> bool {
        self.slots.is_empty()
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
        let mut out: Vec<(SlotId, TrainData<T>)> = Vec::with_capacity(slots.len());
        for source in sources {
            let slot = slot_of(slots, &source, out.iter().map(|(id, _)| *id))?;
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
        if out.len() != slots.len() {
            return Err(no_source());
        }
        let store = Self {
            n,
            cap: n,
            slots: out,
        };
        store.require_in_range()?;
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
    fn require_in_range(&self) -> Result<(), GprError> {
        if reads_in_place::<T>() {
            return Ok(());
        }
        let (n, cap) = (self.n, self.cap.max(1));
        for (_, data) in &self.slots {
            match data {
                TrainData::Scalar(square) => {
                    for j in 0..n {
                        let column = &square[j * cap..j * cap + n];
                        if let Some(i) = column.iter().position(|v| !v.is_finite()) {
                            return Err(out_of_range(i, j));
                        }
                    }
                }
                TrainData::Ard(cache) => {
                    let view = cache.view();
                    for dim in 0..view.d() {
                        for j in 0..n {
                            let run = view.column(dim, j);
                            if let Some(k) = run.iter().position(|v| !v.is_finite()) {
                                return Err(out_of_range(j + k, j));
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// The same squares in `f64`.
    pub(crate) fn to_f64(&self) -> Result<TrainSources<f64>, GprError> {
        self.cast()
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
    /// one pass over its packed values.
    pub(crate) fn cast<U: KernelScalar>(&self) -> Result<TrainSources<U>, GprError> {
        let (n, cap) = (self.n, self.cap.max(1));
        let cast = |v: T| U::from_f64(v.to_f64());
        let mut slots = Vec::with_capacity(self.slots.len());
        for (id, data) in &self.slots {
            let data = match data {
                TrainData::Scalar(square) => {
                    let mut out = Vec::with_capacity(n * n);
                    for j in 0..n {
                        out.extend(square[j * cap..j * cap + n].iter().map(|&v| cast(v)));
                    }
                    TrainData::Scalar(out)
                }
                TrainData::Ard(cache) => TrainData::Ard(cache.map(cast)),
            };
            slots.push((*id, data));
        }
        let store = TrainSources { n, cap: n, slots };
        store.require_in_range()?;
        Ok(store)
    }
}

/// The training `d²` a model of one precision keeps.
///
/// [`TrainSources`] in the storage scalar for a model that factors and
/// predicts in it; [`RefinedSources`] for a model that also refines in
/// `f64` and so keeps the caller's `f64` values next to the storage copy.
pub trait SourceStore<S: KernelScalar>: Clone + fmt::Debug + Send + Sync + 'static {
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

    /// The squares at `f64`: [`Self::exact`], or the storage values widened.
    fn to_f64(&self) -> Result<Cow<'_, TrainSources<f64>>, GprError> {
        match self.exact() {
            Some(exact) => Ok(Cow::Borrowed(exact)),
            None => self.storage().to_f64().map(Cow::Owned),
        }
    }
}

impl<S: KernelScalar> SourceStore<S> for TrainSources<S> {
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
}

/// The training `d²` of a model that factors in `f32` and refines in
/// `f64`: the caller's values at `f64` and the `f32` copy the factor reads.
#[derive(Clone, Debug, Default)]
pub struct RefinedSources {
    storage: TrainSources<f32>,
    exact: TrainSources<f64>,
}

impl SourceStore<f32> for RefinedSources {
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

/// Buffers of [`QuerySources`] kept by a model from call to call: the bound
/// slots (empty between calls), what a fill or a repair writes, and the
/// casts an `f32` model reads. A call that fits the capacity of an earlier
/// one allocates nothing.
pub(crate) struct QueryScratch<T> {
    raw: Vec<RawSlot<'static>>,
    written: Vec<f64>,
    column: Vec<f64>,
    cast: Vec<T>,
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
fn recycle<'x, 'y>(mut raw: Vec<RawSlot<'x>>) -> Vec<RawSlot<'y>> {
    raw.clear();
    raw.into_iter().filter_map(|_| None).collect()
}

/// Whether `T` reads `f64` tables in place.
fn reads_in_place<T: ScalarOps>() -> bool {
    T::from_f64_slice(&[]).is_some()
}

/// The checked `d²` blocks of one prediction, bound on a model's
/// [`QueryScratch`]: a caller's table is read in place (an `f64` model) or
/// through one cast (`f32`), and a fill or a repair writes the scratch.
pub(crate) struct QuerySources<'a, T: KernelScalar> {
    /// One entry per slot of the kernel, in the order the caller gave them.
    raw: Vec<RawSlot<'a>>,
    scratch: &'a mut QueryScratch<T>,
}

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
    /// Returns [`GprError::LengthMismatch`] for a source of a slot the
    /// kernel does not read, a slot without a source, two sources of one
    /// slot, or a block of the wrong length or count;
    /// [`GprError::EmptyInput`] when `rows` or `cols` is zero;
    /// [`GprError::InvalidDistance`] for a value the source's check refuses.
    pub(crate) fn bind<'s: 'a>(
        slots: &[DistanceSlot],
        sources: impl IntoIterator<Item = DistanceSource<'s>>,
        rows: usize,
        cols: usize,
        kind: BlockKind,
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
        let mut this = Self { raw, scratch };
        this.scratch.written.clear();
        for source in sources {
            let slot = slot_of(slots, &source, this.raw.iter().map(|raw| raw.id))?;
            let shape = slot.shape();
            let blocks = shape.blocks();
            let tidy = source.tidy;
            let QueryScratch {
                written, column, ..
            } = &mut *this.scratch;
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
                    let at = written.len();
                    fill_dense(filler, (rows, cols), blocks, kind, written, column)?;
                    RawData::Written(at)
                }
            };
            let mut raw_slot = RawSlot {
                id: slot.id(),
                shape,
                data,
                rows,
                cols,
                cast_at: 0,
                unchecked: false,
            };
            // An ARD block of pairs of two sets is checked as it is read:
            // an `f64` model's by the kernel ([`crate::kernel::ard::r2_from_blocks`]
            // and the ARD RBF lanes), any other's by the cast below. One
            // pass over the caller's values either way.
            let read_checked = kind == BlockKind::Rect
                && tidy == Tidy::Exact
                && matches!(shape, SlotShape::Ard(_));
            if read_checked {
                check_counts(&raw_slot, blocks, len, written)?;
                raw_slot.unchecked = true;
            } else {
                check_slot(&mut raw_slot, blocks, len, kind, tidy, written)?;
            }
            this.raw.push(raw_slot);
        }
        if this.raw.len() != slots.len() {
            return Err(no_source());
        }
        if !reads_in_place::<T>() {
            let Self { raw, scratch } = &mut this;
            let QueryScratch { written, cast, .. } = &mut **scratch;
            cast.clear();
            for slot in raw.iter_mut() {
                slot.cast_at = cast.len();
                for k in 0..slot.shape.blocks() {
                    let block = slot.block(k, written);
                    // An unchecked block is checked a tile at a time as it
                    // is cast, while the tile is in cache.
                    for tile in block.chunks(CAST_TILE) {
                        if slot.unchecked && !super::simd::all_valid_distances(tile) {
                            return Err(first_invalid_from(block, slot.rows, 0));
                        }
                        let at = cast.len();
                        cast.extend(tile.iter().map(|&v| T::from_f64(v)));
                        if let Some(i) = cast[at..].iter().position(|v| !v.is_finite()) {
                            let pos = (at - slot.cast_at) % len + i;
                            return Err(out_of_range(pos % slot.rows, pos / slot.rows));
                        }
                    }
                }
            }
        }
        Ok(this)
    }

    /// The checked blocks, by slot.
    pub(crate) fn blocks(&self) -> BoundBlocks<'_> {
        BoundBlocks {
            raw: &self.raw,
            written: &self.scratch.written,
        }
    }

    /// The same blocks in `f64`, read in place.
    pub(crate) fn f64_view(&self) -> F64Blocks<'_> {
        F64Blocks(self.blocks())
    }
}

impl<T: KernelScalar> RectSlots<T> for QuerySources<'_, T> {
    fn rect(&self, slot: SlotId) -> Option<RectSlot<'_, T>> {
        rect_view(self.blocks(), &self.scratch.cast, slot)
    }
}

/// The query squares of a covariance (bound as [`BlockKind::Square`], so
/// checked in full when bound) read where they were bound: in place for an
/// `f64` model, from the one cast otherwise. A scalar slot is its dense
/// square; an ARD slot its dense blocks.
impl<T: KernelScalar> SquareSlots<T> for QuerySources<'_, T> {
    fn square(&self, slot: SlotId) -> Option<SquareSlot<'_, T>> {
        Some(match rect_view(self.blocks(), &self.scratch.cast, slot)? {
            RectSlot::Scalar(view) => SquareSlot::Scalar(view),
            RectSlot::Ard(ArdRect::Checked(blocks)) => SquareSlot::ArdDense(blocks),
            // A square is never left to be checked as it is read.
            RectSlot::Ard(ArdRect::Unchecked(_)) => return None,
        })
    }
}

/// The checked blocks of a [`QuerySources`], looked up by slot.
#[derive(Clone, Copy)]
pub struct BoundBlocks<'v> {
    raw: &'v [RawSlot<'v>],
    written: &'v [f64],
}

impl<'v> BoundBlocks<'v> {
    /// The blocks of `slot`.
    fn find(&self, slot: SlotId) -> Option<&'v RawSlot<'v>> {
        self.raw.iter().find(|raw| raw.id == slot)
    }
}

/// [`BoundBlocks`] read as `f64` in place.
pub(crate) struct F64Blocks<'v>(BoundBlocks<'v>);

impl RectSlots<f64> for F64Blocks<'_> {
    fn rect(&self, slot: SlotId) -> Option<RectSlot<'_, f64>> {
        rect_view(self.0, &[], slot)
    }
}

/// The blocks of `slot` as the scalar `U`: in place when `U` is `f64`,
/// else from `cast` (what [`QuerySources::bind`] cast).
fn rect_view<'v, U: KernelScalar>(
    blocks: BoundBlocks<'v>,
    cast: &'v [U],
    slot: SlotId,
) -> Option<RectSlot<'v, U>> {
    let raw = blocks.find(slot)?;
    let (rows, cols) = (raw.rows, raw.cols);
    let len = rows * cols;
    let dims = raw.shape.blocks();
    let list = if reads_in_place::<U>() {
        match &raw.data {
            RawData::Values(values) => BlockList::Packed(U::from_f64_slice(values)?, len, 1),
            RawData::Blocks(tables) => BlockList::Vecs(U::from_f64_vecs(tables)?),
            RawData::Slices(tables) => BlockList::Slices(U::from_f64_slices(tables)?),
            RawData::Written(at) => {
                BlockList::Packed(U::from_f64_slice(blocks.written.get(*at..)?)?, len, dims)
            }
        }
    } else {
        BlockList::Packed(cast.get(raw.cast_at..)?, len, dims)
    };
    let checked = ArdBlocks::<U, Checked>::new(list, rows, cols, 0);
    Some(match raw.shape {
        SlotShape::Scalar => RectSlot::Scalar(MatRef::from_column_major_slice(
            checked.block(0).get(..len)?,
            rows,
            cols,
        )),
        // A caller's block read in place and not checked when bound is
        // checked as the kernel reads it; a cast checked it already.
        SlotShape::Ard(_) if raw.unchecked && reads_in_place::<U>() => {
            RectSlot::Ard(ArdRect::Unchecked(ArdBlocks::new(list, rows, cols, 0)))
        }
        SlotShape::Ard(_) => RectSlot::Ard(ArdRect::Checked(checked)),
    })
}

/// Appends the `blocks` dense `rows × cols` blocks a fill writes to `all`,
/// one after another. A square asks only the lower triangle and mirrors
/// it. `column` is scratch for one column of every block.
fn fill_dense(
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
fn check_counts(
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
    if count != blocks {
        return Err(GprError::LengthMismatch {
            reason: format!("expected {blocks} tables of squared distances, got {count}"),
        });
    }
    for k in 0..blocks {
        crate::data::require_count(slot.block(k, written).len(), len, "squared distances")?;
    }
    Ok(())
}

/// Values per tile of the cast of [`QuerySources::bind`]: checked and cast
/// while in cache.
const CAST_TILE: usize = 4096;

/// Checks the `blocks` blocks of `slot` (`len` values each) as `tidy`
/// asks. A repair changes an owned table in place; a borrowed one is first
/// copied into `written`.
fn check_slot(
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
        if !check_block(slot.block(k, written), rows, cols, kind, tidy)? {
            continue;
        }
        // A borrowed table is copied only when its repair changes it.
        if let RawData::Values(Cow::Borrowed(_)) | RawData::Slices(_) = slot.data {
            let at = written.len();
            for j in 0..blocks {
                written.extend_from_slice(slot.block(j, &[]));
            }
            slot.data = RawData::Written(at);
        }
        let block: &mut [f64] = match &mut slot.data {
            RawData::Values(values) => values.to_mut(),
            RawData::Blocks(tables) => &mut tables[k],
            RawData::Slices(_) => &mut [],
            RawData::Written(at) => &mut written[*at + k * len..*at + (k + 1) * len],
        };
        repair_block(block, rows, cols, kind);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{ArdDistance, ScalarDistance};

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

    /// The buffers show their capacity, and a clone starts without them.
    #[test]
    fn query_scratch_holds_no_state_to_clone() {
        let image = ScalarDistance::new();
        let slots = [DistanceSlot::Scalar(image)];
        let mut scratch = QueryScratch::<f32>::new();
        let cross = [0.5, 1.0, 1.5, 2.0];
        let bound = QuerySources::bind(
            &slots,
            [image.borrow(&cross)],
            2,
            2,
            BlockKind::Rect,
            &mut scratch,
        )
        .expect("bind");
        drop(bound);
        assert!(scratch.cast.capacity() >= 4);
        assert!(format!("{scratch:?}").starts_with("QueryScratch"));
        assert_eq!(scratch.clone().cast.capacity(), 0);
    }

    #[test]
    fn two_scalar_slots_and_an_ard_slot_bind_into_one_store() {
        let slots = vec![
            DistanceSlot::Scalar(ScalarDistance::new()),
            DistanceSlot::Scalar(ScalarDistance::new()),
            DistanceSlot::Ard(ArdDistance::new(2).expect("dims")),
        ];
        let sources = slots.iter().enumerate().map(|(k, slot)| match *slot {
            DistanceSlot::Scalar(s) => s.from_vec(line(k as f64 + 1.0, 0..3, 0..3)),
            DistanceSlot::Ard(a) => a.from_vecs(vec![line(4.0, 0..3, 0..3), line(5.0, 0..3, 0..3)]),
        });
        let store = TrainSources::<f64>::bind(&slots, sources, 3).expect("store");
        assert_eq!(
            store.dense_f64(),
            vec![
                (SlotShape::Scalar, line(1.0, 0..3, 0..3)),
                (SlotShape::Scalar, line(2.0, 0..3, 0..3)),
                (
                    SlotShape::Ard(2),
                    [line(4.0, 0..3, 0..3), line(5.0, 0..3, 0..3)].concat(),
                ),
            ]
        );
    }

    #[test]
    fn a_borrowed_table_is_read_in_place_by_an_f64_model() {
        let image = ScalarDistance::new();
        let cross = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let slots = [DistanceSlot::Scalar(image)];
        let mut scratch = QueryScratch::new();
        let bound = QuerySources::<f64>::bind(
            &slots,
            [image.borrow(&cross)],
            3,
            2,
            BlockKind::Rect,
            &mut scratch,
        )
        .expect("bind");
        let Some(RectSlot::Scalar(view)) = bound.rect(slots[0].id()) else {
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
        let store = TrainSources::<f64>::bind(&slots, [image.from_vec(train)], 2).expect("store");
        let Some(SquareSlot::Scalar(view)) = store.square(slots[0].id()) else {
            panic!("scalar slot");
        };
        assert_eq!(view.as_ptr(), ptr);
    }
}
