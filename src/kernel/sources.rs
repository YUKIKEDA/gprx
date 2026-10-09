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

use super::compiled::supplied::{ArdRect, ArdSquare, RectSlots, SquareSlots, unbound};
use super::dist::{ArdBlocks, ArdSqDiff, ArdSqDiffBuf, BlockList, Checked, packed_len, packed_run};
use super::simd::SquareOut;
use super::{ArdData, DistanceFill, ScalarData, ScalarOps, SourceData, Tidy};
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
    /// How the source asked its values to be checked.
    tidy: Tidy,
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
    // A source is made by the slot it names (`ScalarDistance`,
    // `ArdDistance`), which gives it data of its own shape, so the two
    // cannot disagree.
    debug_assert_eq!(slot.shape(), source.data.shape());
    Ok(slot)
}

/// Where `id` sorts among the bound supplies: its place in `slots` (the
/// kernel's, in the order its compiled leaves number them per shape).
fn slot_rank(slots: &[DistanceSlot], id: SlotId) -> usize {
    slots
        .iter()
        .position(|slot| slot.id() == id)
        .unwrap_or(slots.len())
}

/// A store changed before it was laid out for the change
/// ([`TrainSources::reserve_point`], [`TrainSources::ready_to_change`]):
/// refused before anything is written.
fn no_room() -> GprError {
    GprError::UnsupportedKernelOperation {
        reason: "the training store was not laid out for the change".to_owned(),
    }
}

/// A slot of the kernel was given no source (every source names a distinct
/// slot of the kernel, and fewer sources than slots arrived).
fn no_source() -> GprError {
    GprError::LengthMismatch {
        reason: "a distance slot of the kernel has no source".to_owned(),
    }
}

/// A check of a table failed, yet the scan that locates a violation found
/// none (the two checks disagree). The table is refused; `(0, 0)` stands
/// for an unknown place, as the reason says.
pub(crate) fn unlocated() -> GprError {
    invalid(
        0,
        0,
        "the table failed its check, but no value could be located \
         (the position (0, 0) is a placeholder)",
    )
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
pub(crate) fn same(a: f64, b: f64) -> bool {
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
    // A square: [`super::simd::square_band`] checks the lower triangle's
    // values, the diagonal, and each mirror, so an invalid value above the
    // diagonal fails its mirror. The violation is located only
    // once a band has failed.
    let bands = rows.div_ceil(BAND);
    let band = |band: usize| {
        let j0 = band * BAND;
        super::simd::square_band(block, rows, (j0, (j0 + BAND).min(rows)))
    };
    let ok = if rows < PAR_ROWS || rayon::current_num_threads() == 1 {
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
    Err(unlocated())
}

/// Rows below which a square is checked on the calling thread: smaller
/// squares cost less than handing their bands to the pool. A pool of one
/// thread is never handed the bands.
const PAR_ROWS: usize = 256;

/// Columns of one band of a square check ([`super::simd::square_band`],
/// [`super::simd::pack_columns`]).
const BAND: usize = super::simd::SQUARE_BAND;

/// Checks the `d` dense `n × n` training squares of an ARD slot exactly
/// (`block(k)` is dimension `k`) and packs their lower triangles, band by
/// band ([`super::simd::pack_columns`]), as the coordinate path fills its
/// `(Δx_d)²` cache. On one thread the runs are appended in order, so the
/// buffer is written once and never zeroed; on the Rayon pool each square
/// is split into ranges of bands ([`pack_bands`]). A violation is located
/// with [`exact_block`].
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
    let (data, ok) = if rayon::current_num_threads() > 1 {
        let mut data = vec![T::from_f64(0.0); len];
        let ok = data
            .par_chunks_mut(per_dim.max(1))
            .enumerate()
            .all(|(k, dest)| pack_bands(block(k), n, 0..n.div_ceil(BAND), dest));
        (data, ok)
    } else {
        let mut data = Vec::with_capacity(len);
        let ok = (0..d).fold(true, |ok, k| {
            ok & super::simd::pack_columns(block(k), n, (0, n), SquareOut::Push(&mut data))
        });
        (data, ok)
    };
    if !ok {
        for k in 0..d {
            exact_block(block(k), n, n, BlockKind::Square)?;
        }
        return Err(unlocated());
    }
    Ok(ArdSqDiffBuf::from_packed(data, n, d))
}

/// Checks the `d` owned dense `n × n` training squares of an ARD slot
/// exactly ([`exact_block`]) and keeps them as the store, when the model
/// reads `f64`: a fit then reads each square once and copies nothing. An
/// `f32` model packs them ([`pack_exact_ard`]).
///
/// # Errors
///
/// As [`pack_exact_ard`].
fn keep_exact_ard<T: KernelScalar>(
    tables: Vec<Vec<f64>>,
    n: usize,
    d: usize,
) -> Result<ArdSqDiffBuf<T>, GprError> {
    if !reads_in_place::<T>() {
        return pack_exact_ard(n, d, |k| &tables[k]);
    }
    for table in &tables {
        exact_block(table, n, n, BlockKind::Square)?;
    }
    match T::vecs_from_f64(tables) {
        Ok(tables) => Ok(ArdSqDiffBuf::from_tables(tables, n)),
        Err(tables) => pack_exact_ard(n, d, |k| &tables[k]),
    }
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
        return j0 >= j1 || super::simd::pack_columns(block, n, (j0, j1), SquareOut::Over(dest));
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
            let (a, b) = (block[i + j * rows], block[j + i * rows]);
            // Only a pair that differs is rewritten, and its mean is taken
            // as `a + (b − a) / 2`, which stays finite for finite values
            // past `f64::MAX / 2`, where `(a + b) / 2` overflows.
            let diff = b - a;
            if diff != 0.0 {
                let mean = a + 0.5 * diff;
                block[i + j * rows] = mean;
                block[j + i * rows] = mean;
            }
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

/// The packed training triangles of an ARD slot of `d` dimensions from its
/// `n × n` tables, checked as `tidy` asks.
fn train_ard<T: KernelScalar>(
    data: ArdData<'_>,
    n: usize,
    d: usize,
    tidy: Tidy,
) -> Result<ArdSqDiffBuf<T>, GprError> {
    let len = n.checked_mul(n).ok_or(GprError::SizeOverflow)?;
    match data {
        ArdData::Blocks(tables) if tidy == Tidy::Exact => {
            require_tables(tables.len(), d)?;
            for table in &tables {
                crate::data::require_count(table.len(), len, "squared distances")?;
            }
            keep_exact_ard(tables, n, d)
        }
        ArdData::Slices(tables) if tidy == Tidy::Exact => {
            require_tables(tables.len(), d)?;
            for table in tables {
                crate::data::require_count(table.len(), len, "squared distances")?;
            }
            pack_exact_ard(n, d, |k| tables[k])
        }
        ArdData::Blocks(mut tables) => {
            require_tables(tables.len(), d)?;
            for table in &mut tables {
                crate::data::require_count(table.len(), len, "squared distances")?;
                if check_block(table, n, n, BlockKind::Square, tidy)? {
                    repair_block(table, n, n, BlockKind::Square);
                }
            }
            ArdSqDiffBuf::from_dense(n, d, |k| &tables[k])
        }
        ArdData::Slices(tables) => {
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
            ArdSqDiffBuf::from_dense(n, d, |k| repaired[k].as_deref().unwrap_or(tables[k]))
        }
        ArdData::Fill(filler) => fill_ard_square(filler, n, d, tidy),
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
    let len = n
        .checked_add(1)
        .and_then(|n1| n.checked_mul(n1))
        .map(|cells| cells / 2)
        .and_then(|per_dim| per_dim.checked_mul(d))
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
            )?;
        }
    }
    if let Tidy::Within(rel) = tidy {
        for rounding in &rounding {
            rounding.judge(rel)?;
        }
    }
    Ok(ArdSqDiffBuf::from_packed(packed, n, d))
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

/// The training `d²` a model owns, one entry per slot of its kernel, by
/// shape in the kernel's slot order: entry `at` of a shape is the slot a
/// compiled leaf numbers `at` ([`super::SuppliedSpec::at`]).
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

    /// Makes room for one more point, so [`Self::push_point`] writes in
    /// place: a full scalar square grows its leading dimension by a
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
    /// The errors of [`bind`] for a square.
    pub(crate) fn bind<'a>(
        slots: &[DistanceSlot],
        sources: impl IntoIterator<Item = DistanceSource<'a>>,
        n: usize,
    ) -> Result<Self, GprError> {
        crate::data::require_nonempty(n)?;
        let len = n.checked_mul(n).ok_or(GprError::SizeOverflow)?;
        let ards = slots
            .iter()
            .filter(|slot| matches!(slot.shape(), SlotShape::Ard(_)))
            .count();
        let mut scalar = Vec::with_capacity(slots.len() - ards);
        let mut ard = Vec::with_capacity(ards);
        for source in sources {
            let bound = scalar.iter().map(|(id, _)| *id);
            let slot = slot_of(slots, &source, bound.chain(ard.iter().map(|(id, _)| *id)))?;
            let tidy = source.tidy;
            let id = slot.id();
            match source.data {
                SourceData::Scalar(ScalarData::Values(values)) => {
                    crate::data::require_count(values.len(), len, "squared distances")?;
                    let mut values = values;
                    if check_block(&values, n, n, BlockKind::Square, tidy)? {
                        repair_block(values.to_mut(), n, n, BlockKind::Square);
                    }
                    scalar.push((id, T::vec_from_f64(values.into_owned())));
                }
                SourceData::Scalar(ScalarData::Fill(filler)) => {
                    scalar.push((id, T::vec_from_f64(fill_scalar_square(filler, n, tidy)?)));
                }
                SourceData::Ard(d, data) => ard.push((id, train_ard(data, n, d, tidy)?)),
            }
        }
        if scalar.len() + ard.len() != slots.len() {
            return Err(no_source());
        }
        scalar.sort_unstable_by_key(|(id, _)| slot_rank(slots, *id));
        ard.sort_unstable_by_key(|(id, _)| slot_rank(slots, *id));
        let store = Self {
            n,
            cap: n,
            scalar,
            ard,
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
        for (_, square) in &self.scalar {
            for j in 0..n {
                let column = &square[j * cap..j * cap + n];
                if let Some(i) = column.iter().position(|v| !v.is_finite()) {
                    return Err(out_of_range(i, j));
                }
            }
        }
        for (_, cache) in &self.ard {
            let view = cache.view();
            for dim in 0..view.d() {
                if let Some((row, col)) = view.position(dim, |v| !v.is_finite()) {
                    return Err(out_of_range(row, col));
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
        for (slot, values) in slots.iter().zip(values) {
            let len = tri
                .checked_mul(slot.shape().blocks())
                .ok_or(GprError::SizeOverflow)?;
            crate::data::require_count(values.len(), len, "persisted squared distances")?;
            check_packed(&values, n)?;
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

    /// The same squares in `f64`.
    pub(crate) fn to_f64(&self) -> Result<TrainSources<f64>, GprError> {
        self.cast()
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
    /// one pass over its packed values.
    pub(crate) fn cast<U: KernelScalar>(&self) -> Result<TrainSources<U>, GprError> {
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
        store.require_in_range()?;
        Ok(store)
    }
}

/// Checks persisted lower triangles of order `n`, block after block:
/// every value finite and non-negative, every diagonal zero.
fn check_packed<T: KernelScalar>(values: &[T], n: usize) -> Result<(), GprError> {
    let mut at = 0;
    while at < values.len() {
        for col in 0..n {
            for row in col..n {
                let v = values[at].to_f64();
                if !valid(v) {
                    return Err(invalid_value(v, row, col));
                }
                if row == col && v != 0.0 {
                    return Err(invalid(row, col, format!("the diagonal is {v}, not zero")));
                }
                at += 1;
            }
        }
    }
    Ok(())
}

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
            storage: exact.cast()?,
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
enum SlotBlocks<T> {
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
    fn packed(rows: usize, cols: usize, scalar: Vec<Vec<T>>, ard: Vec<SlotBlocks<T>>) -> Self {
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
        let stride = self.stride;
        (0..self.scalar.len())
            .map(BlockAt::Scalar)
            .chain(self.ard.iter().enumerate().flat_map(move |(at, slot)| {
                (0..slot.count(stride)).map(move |k| BlockAt::Ard(at, k))
            }))
            .collect()
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
        self.require_in_range::<U>()?;
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
    /// range of `U`, located in its block.
    pub(crate) fn require_in_range<U: KernelScalar>(&self) -> Result<(), GprError> {
        if !U::ROUNDS_FROM_F64 {
            return Ok(());
        }
        let ld = self.ld.max(1);
        for block in self.blocks() {
            if let Some(at) = block
                .iter()
                .position(|v| !U::from_f64(v.to_f64()).to_f64().is_finite())
            {
                return Err(out_of_range(at % ld, at / ld));
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
/// [`GprError::LengthMismatch`] for a source of a slot the kernel does not
/// read, a slot without a source, two sources of one slot, or a block of
/// the wrong length or count, and [`GprError::InvalidDistance`] for a value
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
    let len = n.checked_mul(m).ok_or(GprError::SizeOverflow)?;
    let square_len = m.checked_mul(m).ok_or(GprError::SizeOverflow)?;
    let is_ard = |slot: &DistanceSlot| matches!(slot.shape(), SlotShape::Ard(_));
    let ards = slots.iter().filter(|slot| is_ard(slot)).count();
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
    for source in sources {
        let slot = slot_of(slots, &source, std::iter::empty())?;
        let Some(at) = slots.iter().position(|s| s.id() == slot.id()) else {
            return Err(unbound());
        };
        // Each store in the kernel's slot order: a slot's place among the
        // slots of its shape.
        let place = slots[..at]
            .iter()
            .filter(|other| is_ard(other) == is_ard(slot))
            .count();
        let bound = if is_ard(slot) {
            !matches!(&xz.ard[place], SlotBlocks::Flat(all) if all.is_empty())
        } else {
            !xz.scalar[place].is_empty()
        };
        if bound {
            return Err(GprError::LengthMismatch {
                reason: "two sources were supplied for one distance slot".to_owned(),
            });
        }
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
                        fill_dense(filler, (n, m), 1, BlockKind::Rect, &mut filled, &mut column)?;
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
                let blocks = match data {
                    ArdData::Blocks(mut tables) => {
                        require_tables(tables.len(), d)?;
                        for (k, table) in tables.iter_mut().enumerate() {
                            crate::data::require_count(table.len(), len, "squared distances")?;
                            let square = &mut squares[k * square_len..(k + 1) * square_len];
                            let mut block = Cow::Borrowed(table.as_slice());
                            inducing_parts(&mut block, n, inducing, tidy, square)?;
                            if let Cow::Owned(repaired) = block {
                                *table = repaired;
                            }
                        }
                        SlotBlocks::Tables(tables)
                    }
                    ArdData::Slices(tables) => {
                        require_tables(tables.len(), d)?;
                        let mut all =
                            Vec::with_capacity(len.checked_mul(d).ok_or(GprError::SizeOverflow)?);
                        for (k, table) in tables.iter().enumerate() {
                            crate::data::require_count(table.len(), len, "squared distances")?;
                            let square = &mut squares[k * square_len..(k + 1) * square_len];
                            let mut block = Cow::Borrowed(*table);
                            inducing_parts(&mut block, n, inducing, tidy, square)?;
                            all.extend_from_slice(&block);
                        }
                        SlotBlocks::Flat(all)
                    }
                    ArdData::Fill(filler) => {
                        let mut all = Vec::new();
                        fill_dense(filler, (n, m), d, BlockKind::Rect, &mut all, &mut column)?;
                        for k in 0..d {
                            let square = &mut squares[k * square_len..(k + 1) * square_len];
                            let mut block = Cow::Borrowed(&all[k * len..(k + 1) * len]);
                            inducing_parts(&mut block, n, inducing, tidy, square)?;
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
    let unbound_slot = xz.scalar.iter().any(Vec::is_empty)
        || xz
            .ard
            .iter()
            .any(|slot| matches!(slot, SlotBlocks::Flat(all) if all.is_empty()));
    if unbound_slot {
        return Err(no_source());
    }
    Ok((zz, xz))
}

/// One `n × m` block (training points × inducing points `inducing`),
/// checked as `tidy` asks and repaired where it allows (a borrowed block is
/// copied only to be repaired), and the `m × m` square of its inducing rows
/// written to `square`, checked and repaired as a training square is; the
/// repaired pairs are written back to those rows. A violation of the square
/// is located in `block`: row `inducing[a]`, column `b`.
fn inducing_parts(
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
        Err(GprError::InvalidDistance { row, col, reason }) => {
            return Err(GprError::InvalidDistance {
                row: inducing.get(row).copied().unwrap_or(row),
                col,
                reason,
            });
        }
        Err(err) => return Err(err),
    }
    Ok(())
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
pub(crate) struct QuerySquares<'a, T: KernelScalar>(QuerySources<'a, T>);

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
    fn bind<'s: 'a>(
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
            let slot = slot_of(slots, &source, this.raw.iter().map(|raw| raw.id))?;
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
                id: slot.id(),
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
                check_slot(&mut raw_slot, blocks, len, kind, tidy, written)?;
            }
            this.raw.push(raw_slot);
        }
        if this.raw.len() != slots.len() {
            return Err(no_source());
        }
        this.raw.sort_unstable_by_key(|raw| {
            let ard = matches!(raw.shape, SlotShape::Ard(_));
            (ard, slot_rank(slots, raw.id))
        });
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
        let at = match block {
            BlockAt::Scalar(at) => at,
            BlockAt::Ard(at, _) => self.scalars + at,
        };
        self.raw.get(at).map(|raw| raw.tidy).ok_or_else(unbound)
    }
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
    fn from_blocks<S: super::dist::BlockState>(
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
/// `cols`.
pub(crate) fn new_inducing_column(
    xz: &BlockStore<f64>,
    zz: &BlockStore<f64>,
    inducing: &[usize],
    point: usize,
    cols: &dyn RectSlots<f64>,
    tidy: impl Fn(BlockAt) -> Result<Tidy, GprError>,
) -> Result<(Vec<f64>, Vec<f64>), GprError> {
    let (n, m) = (xz.rows, inducing.len());
    let side = m + 1;
    let blocks = xz.block_ids();
    let mut column = Vec::with_capacity(n * blocks.len());
    let mut mirror = Vec::with_capacity(m * blocks.len());
    let mut square = vec![0.0; side * side];
    for &at in &blocks {
        let start = column.len();
        column_into(cols, at, n, &mut column)?;
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
            Err(GprError::InvalidDistance { row, col, reason }) => {
                let row = if row == m { row_of(col) } else { row_of(row) };
                return Err(GprError::InvalidDistance {
                    row,
                    col: 0,
                    reason,
                });
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
struct BoundBlocks<'v> {
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
            ArdRect::Unchecked(ArdBlocks::new(list, raw.rows, raw.cols, 0))
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
fn block_list<'v, U: KernelScalar>(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{ArdDistance, ScalarDistance};

    /// A store of one scalar slot and two ARD slots (packed, and moved-in
    /// tables), `rows × cols`, every value distinct.
    fn store(rows: usize, cols: usize) -> BlockStore<f64> {
        let block =
            |seed: f64| -> Vec<f64> { (0..rows * cols).map(|i| seed + i as f64 * 0.25).collect() };
        let packed: Vec<f64> = [block(100.0), block(200.0)].concat();
        BlockStore::packed(
            rows,
            cols,
            vec![block(0.0)],
            vec![
                SlotBlocks::Flat(packed),
                SlotBlocks::Tables(vec![block(300.0), block(400.0), block(500.0)]),
            ],
        )
    }

    /// What a kernel reads from `store`: every value through
    /// [`RectSlots`], in [`BlockStore::values`] order.
    fn read(store: &BlockStore<f64>) -> Vec<f64> {
        let (rows, cols, _) = store.values();
        let mut out = Vec::new();
        let scalar = RectSlots::scalar(store, 0).expect("scalar");
        for c in 0..cols {
            out.extend((0..rows).map(|r| scalar[(r, c)]));
        }
        for at in 0..2 {
            let ArdRect::Checked(blocks) = RectSlots::ard(store, at).expect("ard") else {
                panic!("a stored block is checked");
            };
            for k in 0..blocks.d() {
                for c in 0..cols {
                    for r in 0..rows {
                        out.push(blocks.read(k, r, c).expect("read"));
                    }
                }
            }
        }
        out
    }

    /// Each change of a store and its undo give back every value, and a
    /// kernel reads what the store holds, laid out again or not.
    #[test]
    fn block_store_changes_undo_to_the_same_values() {
        let mut s = store(5, 3);
        let original = s.values();
        assert_eq!(read(&s), original.2);
        // A row: room (a new layout), then in place.
        for _ in 0..2 {
            s.reserve_row().expect("room");
            s.push_row(|b, c| 1000.0 + (b * 10 + c) as f64);
            assert_eq!(s.values().0, 6);
            assert_eq!(read(&s), s.values().2);
            assert_eq!(
                s.get(BlockAt::Ard(1, 2), 5, 1).to_bits(),
                1051.0f64.to_bits()
            );
            s.pop_row();
            assert_eq!(s.values(), original);
        }
        for index in [0, 2, 4] {
            let mut saved = Vec::new();
            s.row_into(index, &mut saved);
            s.remove_row(index);
            assert_eq!(read(&s), s.values().2);
            s.insert_row(index, &saved);
            assert_eq!(s.values(), original);
        }
        // A column: room, then in place.
        for _ in 0..2 {
            s.reserve_col().expect("room");
            s.push_col(|b, r| 2000.0 + (b * 10 + r) as f64);
            assert_eq!(read(&s), s.values().2);
            assert_eq!(
                s.get(BlockAt::Scalar(0), 4, 3).to_bits(),
                2004.0f64.to_bits()
            );
            s.pop_col();
            assert_eq!(s.values(), original);
        }
        for index in [0, 1, 2] {
            let mut saved = Vec::new();
            s.remove_col(index, Some(&mut saved));
            assert_eq!(read(&s), s.values().2);
            s.insert_col(index, &saved);
            assert_eq!(s.values(), original);
            assert_eq!(read(&s), original.2);
        }
        // The same changes on the store as bound (moved-in tables, no room).
        let mut fresh = store(5, 3);
        let mut saved = Vec::new();
        fresh.remove_col(1, Some(&mut saved));
        fresh.insert_col(1, &saved);
        assert_eq!(fresh.values(), original);
        let mut saved = Vec::new();
        fresh.row_into(3, &mut saved);
        fresh.remove_row(3);
        fresh.insert_row(3, &saved);
        assert_eq!(fresh.values(), original);
        // A cast keeps the layout; the inducing rows are read back to back.
        let cast = s.cast::<f32>().expect("cast");
        assert_eq!(cast.values().2, original.2);
        let mut square = BlockStore::default();
        s.rows_into(&[4, 0], &mut square);
        assert_eq!(
            square.values().2[..2],
            [
                s.get(BlockAt::Scalar(0), 4, 0),
                s.get(BlockAt::Scalar(0), 0, 0)
            ]
        );
    }

    fn line(scale: f64, rows: std::ops::Range<usize>, cols: std::ops::Range<usize>) -> Vec<f64> {
        cols.flat_map(|j| {
            rows.clone()
                .map(move |i| scale * (i as f64 - j as f64).powi(2))
        })
        .collect()
    }

    /// A repair keeps huge equal pairs as they are and averages a huge
    /// unequal pair without overflowing to infinity.
    #[test]
    fn a_repair_of_huge_values_stays_finite() {
        let big = 1.0e308;
        let mut block = vec![0.0, big, -1.0e-300, big, 0.0, big, 0.0, 0.9 * big, 0.0];
        repair_block(&mut block, 3, 3, BlockKind::Square);
        assert!(block.iter().all(|v| v.is_finite()));
        assert_eq!(block[1].to_bits(), big.to_bits());
        assert_eq!(block[2].to_bits(), 0.0_f64.to_bits());
        assert_eq!(block[5].to_bits(), block[7].to_bits());
        assert!(block[5] > 0.9 * big && block[5] < big);
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
        let bound = QuerySources::bind_rect(&slots, [image.borrow(&cross)], (2, 2), &mut scratch)
            .expect("bind");
        drop(bound);
        assert!(scratch.cast.capacity() >= 4);
        assert!(format!("{scratch:?}").starts_with("QueryScratch"));
        assert_eq!(scratch.clone().cast.capacity(), 0);
    }

    /// An `f32` model checks an ARD block as it casts it, so the `f64`
    /// view of the caller's block (the refinement's) reads it as checked,
    /// not again.
    #[test]
    fn a_cast_ard_block_is_read_as_checked_in_f64() {
        let bands = ArdDistance::of_dims(2);
        let slots = [DistanceSlot::Ard(bands)];
        let (b0, b1) = ([0.5, 1.0, 1.5, 2.0], [0.25, 0.5, 0.75, 1.0]);
        let tables: [&[f64]; 2] = [&b0, &b1];
        let mut scratch = QueryScratch::<f32>::new();
        let bound = QuerySources::bind(
            &slots,
            [bands.borrow(&tables)],
            2,
            2,
            BlockKind::Rect,
            true,
            &mut scratch,
        )
        .expect("bind");
        let view = bound.f64_view();
        assert!(matches!(view.ard(0), Ok(ArdRect::Checked(_))));
    }

    #[test]
    fn two_scalar_slots_and_an_ard_slot_bind_into_one_store() {
        let slots = vec![
            DistanceSlot::Scalar(ScalarDistance::new()),
            DistanceSlot::Scalar(ScalarDistance::new()),
            DistanceSlot::Ard(ArdDistance::of_dims(2)),
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

    /// Sources in any order land at the number a compiled leaf reads: by
    /// shape, in the kernel's slot order.
    #[test]
    fn sources_bind_by_shape_in_the_kernels_slot_order() {
        let (s1, s2) = (ScalarDistance::new(), ScalarDistance::new());
        let a = ArdDistance::of_dims(2);
        let slots = [
            DistanceSlot::Scalar(s1),
            DistanceSlot::Ard(a),
            DistanceSlot::Scalar(s2),
        ];
        let first = |sources: &dyn SquareSlots<f64>| sources.scalar(0).expect("slot")[(1, 0)];
        use crate::test_check::assert_close;
        let store = TrainSources::<f64>::bind(
            &slots,
            [
                s2.from_vec(line(2.0, 0..3, 0..3)),
                a.from_vecs(vec![line(4.0, 0..3, 0..3), line(5.0, 0..3, 0..3)]),
                s1.from_vec(line(1.0, 0..3, 0..3)),
            ],
            3,
        )
        .expect("store");
        assert_close(first(&store), 1.0, 0.0);
        assert_close(store.scalar(1).expect("slot")[(1, 0)], 2.0, 0.0);
        let Ok(ArdSquare::Packed(triangles)) = store.ard(0) else {
            panic!("packed");
        };
        assert_close(triangles.get(1, 1, 0), 5.0, 0.0);
        let cross = [line(10.0, 0..3, 0..2), line(20.0, 0..3, 0..2)];
        let bands = [line(40.0, 0..3, 0..2), line(50.0, 0..3, 0..2)];
        let band_refs: Vec<&[f64]> = bands.iter().map(Vec::as_slice).collect();
        let mut scratch = QueryScratch::new();
        let bound = QuerySources::<f64>::bind_rect(
            &slots,
            [
                a.borrow(&band_refs),
                s2.borrow(&cross[1]),
                s1.borrow(&cross[0]),
            ],
            (3, 2),
            &mut scratch,
        )
        .expect("bind");
        assert_close(bound.scalar(0).expect("slot")[(1, 0)], 10.0, 0.0);
        assert_close(bound.scalar(1).expect("slot")[(1, 0)], 20.0, 0.0);
        let Ok(ArdRect::Unchecked(blocks)) = bound.ard(0) else {
            panic!("read in place");
        };
        let column = blocks.column(1, 0).expect("checked").expect("dense");
        assert_close(column[1], 50.0, 0.0);
        // Blocks bound to be checked as they are read are not a square.
        let squares = QuerySquares(bound);
        assert!(matches!(
            squares.ard(0),
            Err(GprError::UnsupportedKernelOperation { .. })
        ));
        assert!(matches!(
            squares.ard(1),
            Err(GprError::UnsupportedKernelOperation { .. })
        ));
    }

    #[test]
    fn a_borrowed_table_is_read_in_place_by_an_f64_model() {
        let image = ScalarDistance::new();
        let cross = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let slots = [DistanceSlot::Scalar(image)];
        let mut scratch = QueryScratch::new();
        let bound =
            QuerySources::<f64>::bind_rect(&slots, [image.borrow(&cross)], (3, 2), &mut scratch)
                .expect("bind");
        let view = bound.scalar(0).expect("slot");
        assert_eq!(view.as_ptr(), cross.as_ptr());
    }

    #[test]
    fn an_owned_table_moves_into_an_f64_store() {
        let image = ScalarDistance::new();
        let train = vec![0.0, 1.0, 1.0, 0.0];
        let ptr = train.as_ptr();
        let slots = [DistanceSlot::Scalar(image)];
        let store = TrainSources::<f64>::bind(&slots, [image.from_vec(train)], 2).expect("store");
        let view = store.scalar(0).expect("slot");
        assert_eq!(view.as_ptr(), ptr);
    }

    /// An `f64` store keeps owned ARD tables as they are, after the exact
    /// check; an `f32` store packs them. Both read the same pairs.
    #[test]
    fn owned_ard_tables_move_into_an_f64_store() {
        use crate::test_check::assert_close;
        let bands = ArdDistance::of_dims(2);
        let slots = [DistanceSlot::Ard(bands)];
        let tables = vec![line(1.0, 0..3, 0..3), line(2.0, 0..3, 0..3)];
        let kept = tables.clone();
        let ptr = kept[1].as_ptr();
        let store = TrainSources::<f64>::bind(&slots, [bands.from_vecs(kept)], 3).expect("f64");
        let Ok(ArdSquare::Packed(view)) = store.ard(0) else {
            panic!("ard slot");
        };
        assert_eq!(view.lower().expect("lower").column(1, 0).as_ptr(), ptr);
        assert!(view.lower().expect("lower").packed_block(0).is_none());
        let narrow =
            TrainSources::<f32>::bind(&slots, [bands.from_vecs(tables.clone())], 3).expect("f32");
        let Ok(ArdSquare::Packed(packed)) = narrow.ard(0) else {
            panic!("ard slot");
        };
        assert!(packed.lower().expect("lower").packed_block(0).is_some());
        for (k, table) in tables.iter().enumerate() {
            for j in 0..3 {
                for i in 0..3 {
                    assert_close(view.get(k, i, j), table[i + j * 3], 0.0);
                    assert_close(f64::from(packed.get(k, i, j)), table[i + j * 3], 0.0);
                }
            }
        }
        // An asymmetric table is refused before it is kept.
        let mut skewed = tables;
        skewed[0][1] += 1.0;
        assert!(matches!(
            TrainSources::<f64>::bind(&slots, [bands.from_vecs(skewed)], 3),
            Err(GprError::InvalidDistance { .. })
        ));
    }
}
