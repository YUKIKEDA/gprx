//! Checks and repairs of a table of squared distances: finite and
//! non-negative values, and for a square a zero diagonal and equal mirror
//! entries, exactly or within a source's tolerance.

#[allow(
    unused_imports,
    reason = "a split file takes its parent's imports whole; each uses some"
)]
use super::*;

/// What a block must satisfy beyond its length and finite values.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum BlockKind {
    /// Pairs of one set: zero diagonal, symmetric.
    Square,
    /// Pairs of two sets.
    Rect,
}

/// A check of a table failed, yet the scan that locates a violation found
/// none (the two checks disagree). The table is refused with no pair.
pub(crate) fn unlocated() -> GprError {
    GprError::InvalidDistance {
        slot: None,
        dim: None,
        pair: None,
        reason: "the table failed its check, but no value could be located".to_owned(),
    }
}

/// An invalid pair `(row, col)` of a table; the caller names its slot and
/// dimension ([`GprError::in_slot`], [`GprError::in_dim`]).
pub(super) fn invalid(row: usize, col: usize, reason: impl Into<String>) -> GprError {
    GprError::InvalidDistance {
        slot: None,
        dim: None,
        pair: Some((row, col)),
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
pub(super) fn first_invalid(block: &[f64], rows: usize) -> GprError {
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
pub(super) fn out_of_range(row: usize, col: usize) -> GprError {
    invalid(row, col, "is past the range of the model's storage scalar")
}

/// The error of a negative value `v` at `(row, col)` past a repair's
/// tolerance `tol`.
pub(super) fn negative_past(v: f64, tol: f64, row: usize, col: usize) -> GprError {
    invalid(
        row,
        col,
        format!("{v} is negative past the tolerance {tol}"),
    )
}

/// The error of a diagonal `v` of column `j` that is not zero.
pub(super) fn nonzero_diagonal(v: f64, j: usize) -> GprError {
    invalid(j, j, format!("the diagonal is {v}, not zero"))
}

/// Checks one column run of a lower triangle exactly: rows `col..` of
/// column `col`, its diagonal first. The diagonal must be zero and every
/// value [`valid`]; the values are folded without a branch, and a
/// violation is located only once the fold has found one.
///
/// # Errors
///
/// Returns [`GprError::InvalidDistance`] at the diagonal if it is valid
/// and not zero, else at the first value that is not valid.
pub(super) fn check_lower_run<T: KernelScalar>(run: &[T], col: usize) -> Result<(), GprError> {
    if let Some(diag) = run.first().map(|v| v.to_f64())
        && valid(diag)
        && diag != 0.0
    {
        return Err(nonzero_diagonal(diag, col));
    }
    if !run.iter().fold(true, |ok, v| ok & valid(v.to_f64())) {
        let at = run.iter().position(|v| !valid(v.to_f64())).unwrap_or(0);
        return Err(invalid_value(run[at].to_f64(), col + at, col));
    }
    Ok(())
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
pub(super) fn check_block(
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
pub(super) fn exact_block(
    block: &[f64],
    rows: usize,
    cols: usize,
    kind: BlockKind,
) -> Result<(), GprError> {
    if kind == BlockKind::Rect {
        return if crate::kernel::simd::all_valid_distances(block) {
            Ok(())
        } else {
            Err(first_invalid(block, rows))
        };
    }
    // A square: [`crate::kernel::simd::square_band`] checks the lower triangle's
    // values, the diagonal, and each mirror, so an invalid value above the
    // diagonal fails its mirror. The violation is located only
    // once a band has failed.
    let bands = rows.div_ceil(BAND);
    let band = |band: usize| {
        let j0 = band * BAND;
        crate::kernel::simd::square_band(block, rows, (j0, (j0 + BAND).min(rows)))
    };
    let ok = if rows < PAR_ROWS || rayon::current_num_threads() == 1 {
        (0..bands).all(band)
    } else {
        (0..bands).into_par_iter().all(band)
    };
    if ok {
        return Ok(());
    }
    if !crate::kernel::simd::all_valid_distances(block) {
        return Err(first_invalid(block, rows));
    }
    for j in 0..cols {
        let diag = block[j + j * rows];
        if diag != 0.0 {
            return Err(nonzero_diagonal(diag, j));
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
pub(super) const PAR_ROWS: usize = 256;

/// Columns of one band of a square check ([`crate::kernel::simd::square_band`],
/// [`crate::kernel::simd::pack_columns`]).
pub(super) const BAND: usize = crate::kernel::simd::SQUARE_BAND;

/// Checks the `d` dense `n × n` training squares of an ARD slot exactly
/// (`block(k)` is dimension `k`) and packs their lower triangles, band by
/// band ([`crate::kernel::simd::pack_columns`]), as the coordinate path fills its
/// `(Δx_d)²` cache. On one thread the runs are appended in order, so the
/// buffer is written once and never zeroed; on the Rayon pool each square
/// is split into ranges of bands ([`pack_bands`]). A violation is located
/// with [`exact_block`].
///
/// # Errors
///
/// Returns [`GprError::InvalidDistance`] for the first violation and
/// [`GprError::SizeOverflow`] when the packed size does not fit.
pub(super) fn pack_exact_ard<'b, T: KernelScalar>(
    n: usize,
    d: usize,
    block: impl Fn(usize) -> &'b [f64] + Sync,
) -> Result<ArdSqDiffBuf<T>, GprError> {
    let per_dim = packed_len(n)?;
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
            ok & crate::kernel::simd::pack_columns(block(k), n, (0, n), SquareOut::Push(&mut data))
        });
        (data, ok)
    };
    if !ok {
        for k in 0..d {
            exact_block(block(k), n, n, BlockKind::Square).map_err(|err| err.in_dim(k))?;
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
pub(super) fn keep_exact_ard<T: KernelScalar>(
    tables: Vec<Vec<f64>>,
    n: usize,
    d: usize,
) -> Result<ArdSqDiffBuf<T>, GprError> {
    if !reads_in_place::<T>() {
        return pack_exact_ard(n, d, |k| &tables[k]);
    }
    for (k, table) in tables.iter().enumerate() {
        exact_block(table, n, n, BlockKind::Square).map_err(|err| err.in_dim(k))?;
    }
    match T::vecs_from_f64(tables) {
        Ok(tables) => Ok(ArdSqDiffBuf::from_tables(tables, n)),
        Err(tables) => pack_exact_ard(n, d, |k| &tables[k]),
    }
}

/// Checks and packs the bands `bands` of one square into `dest` (their
/// columns of the packed triangle, in order), halving the range on the
/// Rayon pool down to one band, so no list of bands is made.
pub(super) fn pack_bands<T: KernelScalar>(
    block: &[f64],
    n: usize,
    bands: std::ops::Range<usize>,
    dest: &mut [T],
) -> bool {
    let (j0, j1) = (bands.start * BAND, (bands.end * BAND).min(n));
    if bands.len() <= 1 {
        return j0 >= j1
            || crate::kernel::simd::pack_columns(block, n, (j0, j1), SquareOut::Over(dest));
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
pub(super) fn within_block(
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
pub(super) const PAIR_TILE: usize = 64;

/// Visits each pair `i > j` of an `n × n` column-major square tile by tile,
/// so the mirror `(j, i)` (a row of the square) is read while its tile is
/// still in cache, not one strided load per pair.
pub(super) fn for_each_lower_pair<E>(
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
pub(super) fn repair_block(block: &mut [f64], rows: usize, cols: usize, kind: BlockKind) {
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
pub(super) struct FillRounding {
    max: f64,
    negative: Option<(f64, usize, usize)>,
    diagonal: Option<(f64, usize, usize)>,
}

impl FillRounding {
    /// Notes `v` at `(row, col)` and returns what is stored: `v`, or `0.0`
    /// for a negative value or a diagonal.
    pub(super) fn note(&mut self, v: f64, row: usize, col: usize) -> f64 {
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

    pub(super) fn judge(&self, rel: f64) -> Result<(), GprError> {
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

/// Checks the lengths `lens` of a slot's tables, one per table: `blocks`
/// tables of `len` values each.
///
/// # Errors
///
/// Returns [`GprError::LengthMismatch`] for another number of tables or a
/// table of another length.
pub(super) fn require_blocks(
    lens: impl ExactSizeIterator<Item = usize>,
    blocks: usize,
    len: usize,
) -> Result<(), GprError> {
    if lens.len() != blocks {
        return Err(GprError::LengthMismatch {
            reason: format!(
                "expected {blocks} tables of squared distances, got {}",
                lens.len()
            ),
        });
    }
    for got in lens {
        crate::data::require_count(got, len, "squared distances")?;
    }
    Ok(())
}

/// [`require_blocks`] of the tables `data` holds; a fill writes its own.
pub(super) fn require_ard_data(data: &ArdData<'_>, d: usize, len: usize) -> Result<(), GprError> {
    match data {
        ArdData::Blocks(tables) => require_blocks(tables.iter().map(Vec::len), d, len),
        ArdData::Slices(tables) => require_blocks(tables.iter().map(|table| table.len()), d, len),
        ArdData::Fill(_) => Ok(()),
    }
}
