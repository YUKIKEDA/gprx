//! Describes the pairwise squared-Euclidean distances, filled by the process-wide thread pool column partitions.

use super::KernelScalar;
use super::simd::dist::{try_fill_ard_column, try_fill_cross_chunk, try_fill_lower_col};
use crate::error::GprError;
use faer::reborrow::ReborrowMut;
use faer::{ColMut, Mat, MatMut, MatRef};
use rayon::prelude::*;
use std::convert::Infallible;

/// Returns the Rayon pool size, at least 1.
pub(crate) fn worker_count() -> usize {
    rayon::current_num_threads().max(1)
}

/// Column range `(start, len)` for chunk `idx` of `n_chunks` covering `n` columns.
///
/// Matches faer's `par_col_partition` split so each view owns a disjoint column block.
pub(crate) fn col_chunk(n: usize, idx: usize, n_chunks: usize) -> (usize, usize) {
    let chunk_size = n / n_chunks;
    let rem = n % n_chunks;
    let start = |i: usize| {
        if i < rem {
            i * (chunk_size + 1)
        } else {
            rem + i * chunk_size
        }
    };
    let begin = start(idx);
    (begin, start(idx + 1) - begin)
}

/// Entries of the lower triangle (diagonal included) in columns `[0, c)` of
/// an `n × n` matrix: `c n − c (c − 1) / 2`.
fn lower_area(n: usize, c: usize) -> u128 {
    let (n, c) = (n as u128, c as u128);
    c * n - c * c.saturating_sub(1) / 2
}

/// First column of block `idx` of `n_blocks` whose lower triangles hold
/// about equal entries: the `c` whose `lower_area(c) · n_blocks` is nearest
/// `idx · lower_area(n)`, so two blocks differ by at most one column's
/// entries. Block `n_blocks` starts at `n`.
pub(crate) fn lower_block_start(n: usize, idx: usize, n_blocks: usize) -> usize {
    let n_blocks = n_blocks.max(1);
    if idx >= n_blocks {
        return n;
    }
    let target = idx as u128 * lower_area(n, n);
    let (mut lo, mut hi) = (0, n);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if lower_area(n, mid) * n_blocks as u128 >= target {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    // `lo` is the first column at or past the target; the one before may be nearer.
    let k = n_blocks as u128;
    if lo > 0 && target - lower_area(n, lo - 1) * k < lower_area(n, lo) * k - target {
        lo - 1
    } else {
        lo
    }
}

/// Runs `f(first_col, block)` on the Rayon pool over `n_blocks` column
/// blocks of the square `out` whose lower triangles hold about equal
/// entries ([`lower_block_start`]). Splits by halving, so nothing is
/// allocated. Returns the first error of the left-most failing block.
pub(crate) fn par_lower_blocks<T, E, F>(out: MatMut<'_, T>, n_blocks: usize, f: &F) -> Result<(), E>
where
    T: Send,
    E: Send,
    F: Fn(usize, MatMut<'_, T>) -> Result<(), E> + Sync,
{
    let n = out.ncols();
    let n_blocks = n_blocks.max(1);
    split_lower_blocks(out, n, (0, n_blocks), n_blocks, f)
}

fn split_lower_blocks<T, E, F>(
    block: MatMut<'_, T>,
    n: usize,
    (lo, hi): (usize, usize),
    n_blocks: usize,
    f: &F,
) -> Result<(), E>
where
    T: Send,
    E: Send,
    F: Fn(usize, MatMut<'_, T>) -> Result<(), E> + Sync,
{
    let start = lower_block_start(n, lo, n_blocks);
    if hi - lo <= 1 {
        return f(start, block);
    }
    let mid = lo + (hi - lo) / 2;
    let (left, right) = block.split_at_col_mut(lower_block_start(n, mid, n_blocks) - start);
    let (a, b) = rayon::join(
        || split_lower_blocks(left, n, (lo, mid), n_blocks, f),
        || split_lower_blocks(right, n, (mid, hi), n_blocks, f),
    );
    a.and(b)
}

/// Runs `f(col, rows)` for every column `col` of the lower triangle of the
/// square `out`, on the Rayon pool in [`par_lower_blocks`] of
/// [`worker_count`]: `rows` is rows `col..n` of that column. Returns the
/// first error of the left-most failing block.
pub(crate) fn par_lower_cols<T, E, F>(out: MatMut<'_, T>, f: &F) -> Result<(), E>
where
    T: Send,
    E: Send,
    F: Fn(usize, ColMut<'_, T>) -> Result<(), E> + Sync,
{
    par_lower_cols_in(out, worker_count(), f)
}

/// [`par_lower_cols`] over `n_blocks` blocks.
pub(crate) fn par_lower_cols_in<T, E, F>(
    out: MatMut<'_, T>,
    n_blocks: usize,
    f: &F,
) -> Result<(), E>
where
    T: Send,
    E: Send,
    F: Fn(usize, ColMut<'_, T>) -> Result<(), E> + Sync,
{
    let n = out.nrows();
    par_lower_blocks(out, n_blocks, &|start, mut part: MatMut<'_, T>| {
        for local in 0..part.ncols() {
            let col = start + local;
            f(col, part.rb_mut().col_mut(local).subrows_mut(col, n - col))?;
        }
        Ok(())
    })
}

/// [`par_lower_cols`] for an `f` that cannot fail.
pub(crate) fn for_each_lower_col<T, F>(out: MatMut<'_, T>, f: &F)
where
    T: Send,
    F: Fn(usize, ColMut<'_, T>) + Sync,
{
    let done: Result<(), Infallible> = par_lower_cols(out, &|col, rows| {
        f(col, rows);
        Ok(())
    });
    match done {
        Ok(()) => {}
        Err(never) => match never {},
    }
}

/// [`par_lower_fold`] for an `f` that cannot fail.
pub(crate) fn lower_fold_infallible<R, F, J>(n: usize, f: &F, join: &J) -> R
where
    R: Send,
    F: Fn(usize, usize) -> R + Sync,
    J: Fn(R, R) -> R + Sync,
{
    let folded: Result<R, Infallible> = par_lower_fold(n, &|start, end| Ok(f(start, end)), join);
    match folded {
        Ok(value) => value,
        Err(never) => match never {},
    }
}

/// Rows `col..` of column `col` of `m` as a slice, when `m` is column-major.
#[inline(always)]
pub(crate) fn lower_col<T>(m: MatRef<'_, T>, col: usize) -> Option<&[T]> {
    m.col(col).try_as_col_major().map(|c| &c.as_slice()[col..])
}

/// Columns evaluated together before their results are folded in order.
/// The fold itself is one column at a time, so this width does not change
/// the sum.
const COL_CHUNK: usize = 16;

/// Folds `f(col, col + 1)` from column `0` to column `n`, with `join`, in
/// that column order.
///
/// A group of [`COL_CHUNK`] columns is evaluated on the Rayon pool, then
/// folded on this thread. The association is the serial scan
/// `acc = join(acc, f(col, col + 1))`, so the result depends neither on
/// scheduling nor on the pool size. Joining area blocks is a different
/// association: near an optimum the gradient is mostly cancellation, and
/// that rounding steers L-BFGS. `n = 0` calls `f(0, 0)` once. Returns the
/// first error of the left-most failing column.
pub(crate) fn par_lower_fold<R, E, F, J>(n: usize, f: &F, join: &J) -> Result<R, E>
where
    R: Send,
    E: Send,
    F: Fn(usize, usize) -> Result<R, E> + Sync,
    J: Fn(R, R) -> R + Sync,
{
    if n == 0 {
        return f(0, 0);
    }
    let mut acc: Option<R> = None;
    let mut start = 0;
    while start < n {
        let len = COL_CHUNK.min(n - start);
        let mut slots: [Option<R>; COL_CHUNK] = std::array::from_fn(|_| None);
        fill_col_sums(&mut slots[..len], start, f)?;
        for (i, slot) in slots[..len].iter_mut().enumerate() {
            let part = match slot.take() {
                Some(part) => part,
                None => {
                    debug_assert!(false, "par_lower_fold wrote every column");
                    f(start + i, start + i + 1)?
                }
            };
            acc = Some(match acc {
                None => part,
                Some(prev) => join(prev, part),
            });
        }
        start += len;
    }
    // `n > 0` and every column was written, so `acc` holds the fold.
    match acc {
        Some(acc) => Ok(acc),
        None => f(0, 0),
    }
}

/// Writes `f(col, col + 1)` into `slots[0..]`, column `col0` first.
fn fill_col_sums<R, E, F>(slots: &mut [Option<R>], col0: usize, f: &F) -> Result<(), E>
where
    R: Send,
    E: Send,
    F: Fn(usize, usize) -> Result<R, E> + Sync,
{
    let n = slots.len();
    if n == 0 {
        return Ok(());
    }
    if n == 1 {
        slots[0] = Some(f(col0, col0 + 1)?);
        return Ok(());
    }
    let mid = n / 2;
    let (left, right) = slots.split_at_mut(mid);
    // One worker: the same columns, in order, without `rayon::join`. A join
    // from outside the pool queues a job, and the queue allocates a block
    // every few dozen jobs, which would break the zero-allocation hot path.
    let (a, b) = if rayon::current_num_threads() > 1 {
        rayon::join(
            || fill_col_sums(left, col0, f),
            || fill_col_sums(right, col0 + mid, f),
        )
    } else {
        let a = fill_col_sums(left, col0, f);
        let b = fill_col_sums(right, col0 + mid, f);
        (a, b)
    };
    a.and(b)
}

fn partition_count(thread_scratch: &[Mat<f64>]) -> usize {
    if thread_scratch.is_empty() {
        worker_count()
    } else {
        thread_scratch.len()
    }
    .max(1)
}

/// Writes squared Euclidean distances for every pair of rows of `x`.
///
/// Lower triangle is filled in parallel. The upper triangle is copied afterwards
/// so [`crate::kernel::Triangle::Full`] readers stay valid. `thread_scratch` is
/// the detached per-worker slice from [`crate::workspace::Workspace`]; an empty
/// slice still parallelizes with [`worker_count`].
pub(crate) fn fill_squared_euclidean(
    x: MatRef<'_, f64>,
    mut dist: MatMut<'_, f64>,
    thread_scratch: &mut [Mat<f64>],
) {
    let n = x.nrows();
    if n == 0 {
        return;
    }
    let n_parts = partition_count(thread_scratch);
    let done: Result<(), Infallible> = par_lower_cols_in(dist.rb_mut(), n_parts, &|col, rows| {
        fill_lower_col(x, col, rows);
        Ok(())
    });
    match done {
        Ok(()) => {}
        Err(never) => match never {},
    }
    copy_lower_to_upper(dist);
}

/// Number of entries in the lower triangle (diagonal included) of an
/// `n × n` matrix.
fn packed_len(n: usize) -> Result<usize, GprError> {
    n.checked_add(1)
        .and_then(|n1| n.checked_mul(n1))
        .map(|cells| cells / 2)
        .ok_or(GprError::SizeOverflow)
}

/// Offset of column `col` in a column-packed lower triangle of order `n`.
#[inline]
fn packed_col_offset(n: usize, col: usize) -> usize {
    // Columns 0..col hold n, n-1, …, n-col+1 entries.
    col * (2 * n - col + 1) / 2
}

/// Raw `(Δx_d)²` for every pair of rows of `x`, owned.
///
/// Only the lower triangle (diagonal included) of each dimension is stored,
/// column by column, so the cache holds `d · n(n+1)/2` values instead of
/// `d · n²`. Read it through [`Self::view`].
#[derive(Clone, Debug)]
pub(crate) struct ArdSqDiffBuf<T> {
    data: Vec<T>,
    n: usize,
    d: usize,
}

impl<T: KernelScalar> ArdSqDiffBuf<T> {
    /// Fills the cache for the rows of `x`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::SizeOverflow`] when `d · n(n+1)/2` overflows.
    pub(crate) fn new(x: MatRef<'_, T>) -> Result<Self, GprError> {
        let n = x.nrows();
        let d = x.ncols();
        let len = packed_len(n)?
            .checked_mul(d)
            .ok_or(GprError::SizeOverflow)?;
        let mut data = vec![T::from_f64(0.0); len];
        T::write_ard(x, &mut data);
        Ok(Self { data, n, d })
    }

    /// Packs the lower triangles of `d` dense `n × n` blocks; `pair(k, i, j)`
    /// is `(Δ_k)²` of the pair `(i, j)`, `i ≥ j`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::SizeOverflow`] when `d · n(n+1)/2` overflows.
    pub(crate) fn from_pairs(
        n: usize,
        d: usize,
        pair: impl Fn(usize, usize, usize) -> T,
    ) -> Result<Self, GprError> {
        let len = packed_len(n)?
            .checked_mul(d)
            .ok_or(GprError::SizeOverflow)?;
        let mut data = Vec::with_capacity(len);
        for k in 0..d {
            for col in 0..n {
                for row in col..n {
                    data.push(pair(k, row, col));
                }
            }
        }
        Ok(Self { data, n, d })
    }

    /// `(points, dimensions)` the cache was filled for.
    #[cfg(test)]
    pub(crate) fn shape(&self) -> (usize, usize) {
        (self.n, self.d)
    }

    /// Number of stored values.
    #[cfg(test)]
    pub(crate) fn stored_len(&self) -> usize {
        self.data.len()
    }

    /// Overwrites every cached value, to show that a reader uses the cache.
    #[cfg(test)]
    pub(crate) fn poison(&mut self, value: T) {
        self.data.fill(value);
    }

    pub(crate) fn view(&self) -> ArdSqDiff<'_, T> {
        ArdSqDiff {
            data: &self.data,
            n: self.n,
            d: self.d,
            block: self.data.len().checked_div(self.d).unwrap_or(0),
        }
    }
}

/// Borrowed raw `(Δx_d)²` cache of [`ArdSqDiffBuf`].
///
/// [`Self::get`] reads any pair, in either order. [`Self::column`] is the
/// contiguous stored part of one column: rows `col..n`.
#[derive(Clone, Copy, Debug)]
pub struct ArdSqDiff<'a, T> {
    data: &'a [T],
    n: usize,
    d: usize,
    /// Entries per dimension, `n(n+1)/2`.
    block: usize,
}

impl<'a, T: KernelScalar> ArdSqDiff<'a, T> {
    /// Number of points.
    pub(crate) fn n(&self) -> usize {
        self.n
    }

    /// Number of dimensions.
    pub(crate) fn d(&self) -> usize {
        self.d
    }

    /// `(x_row,dim − x_col,dim)²` for rows `col..n`, in row order.
    #[inline]
    pub(crate) fn column(&self, dim: usize, col: usize) -> &'a [T] {
        let start = dim * self.block + packed_col_offset(self.n, col);
        &self.data[start..start + (self.n - col)]
    }

    /// `(x_row,dim − x_col,dim)²` for any pair.
    #[inline]
    pub(crate) fn get(&self, dim: usize, row: usize, col: usize) -> T {
        let (row, col) = if row >= col { (row, col) } else { (col, row) };
        self.column(dim, col)[row - col]
    }

    /// The same cache as `f64` when `T` is `f64`, for the SIMD paths.
    pub(crate) fn as_f64(self) -> Option<ArdSqDiff<'a, f64>> {
        Some(ArdSqDiff {
            data: T::as_f64_slice(self.data)?,
            n: self.n,
            d: self.d,
            block: self.block,
        })
    }
}

/// Raw `(Δ_d)²` of a rectangular block (`rows × cols`), one dense
/// column-major block per dimension: `(row, col)` of dimension `k` is
/// `block(k)[row + col * rows]`.
#[derive(Clone, Copy, Debug)]
pub struct ArdBlocks<'a, T> {
    /// One block per dimension.
    pub(crate) blocks: BlockList<'a, T>,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    /// First column of the stored blocks this view starts at.
    pub(crate) col0: usize,
}

/// The per-dimension blocks of an [`ArdBlocks`]: borrowed slices (a
/// caller's blocks), or one block repeated.
#[derive(Clone, Copy, Debug)]
pub(crate) enum BlockList<'a, T> {
    Slices(&'a [&'a [T]]),
    /// One block for each of `dims` dimensions.
    Repeat(&'a [T], usize),
}

impl<'a, T: KernelScalar> ArdBlocks<'a, T> {
    /// Number of dimensions.
    pub(crate) fn d(&self) -> usize {
        match self.blocks {
            BlockList::Slices(blocks) => blocks.len(),
            BlockList::Repeat(_, dims) => dims,
        }
    }

    /// The block of dimension `dim`.
    pub(crate) fn block(&self, dim: usize) -> &'a [T] {
        match self.blocks {
            BlockList::Slices(blocks) => blocks[dim],
            BlockList::Repeat(block, _) => block,
        }
    }

    /// `(Δ_dim)²` of the pair `(row, col)`.
    #[inline]
    pub(crate) fn get(&self, dim: usize, row: usize, col: usize) -> T {
        self.block(dim)[row + (col + self.col0) * self.rows]
    }
}

/// Writes raw `(Δx_d)²` into `cache`, packed as [`ArdSqDiffBuf`] stores it.
/// Columns are filled in parallel.
pub(crate) fn fill_ard_squared_diff(x: MatRef<'_, f64>, cache: &mut [f64]) {
    let n = x.nrows();
    let d = x.ncols();
    if n == 0 || d == 0 {
        return;
    }
    debug_assert_eq!(cache.len(), d * n * (n + 1) / 2);
    let mut columns: Vec<(usize, usize, &mut [f64])> = Vec::with_capacity(n * d);
    let mut rest = cache;
    for dim in 0..d {
        for col in 0..n {
            let (head, tail) = rest.split_at_mut(n - col);
            columns.push((dim, col, head));
            rest = tail;
        }
    }
    columns.into_par_iter().for_each(|(dim, col, dest)| {
        if !try_fill_ard_column(x, dim, col, dest) {
            for (offset, slot) in dest.iter_mut().enumerate() {
                let diff = x[(col + offset, dim)] - x[(col, dim)];
                *slot = diff * diff;
            }
        }
    });
}

/// Requires `cache` to hold `n` points in `d` dimensions.
pub(crate) fn require_ard_sq_diff_shape<T: KernelScalar>(
    cache: ArdSqDiff<'_, T>,
    n: usize,
    d: usize,
) -> Result<(), GprError> {
    if cache.n() == n && cache.d() == d {
        Ok(())
    } else {
        Err(GprError::ShapeMismatch {
            reason: format!(
                "ARD cache holds {} points in {} dimensions, expected {n} in {d}",
                cache.n(),
                cache.d()
            ),
        })
    }
}

/// Writes rectangular squared distances `k(x_train, x_test)`.
pub(crate) fn fill_squared_euclidean_cross(
    x_train: MatRef<'_, f64>,
    x_test: MatRef<'_, f64>,
    mut dist: MatMut<'_, f64>,
    thread_scratch: &mut [Mat<f64>],
) {
    let m = x_test.nrows();
    if x_train.nrows() == 0 || m == 0 {
        return;
    }
    if m == 1 {
        fill_cross_chunk(x_train, x_test, dist.rb_mut(), 0);
        return;
    }
    let n_parts = partition_count(thread_scratch);
    dist.rb_mut()
        .par_col_partition_mut(n_parts)
        .enumerate()
        .for_each(|(chunk_idx, part)| {
            let (start, _) = col_chunk(m, chunk_idx, n_parts);
            fill_cross_chunk(x_train, x_test, part, start);
        });
}

/// Rows `col..n` of column `col` of the squared distances of `x`.
fn fill_lower_col(x: MatRef<'_, f64>, col: usize, mut rows: ColMut<'_, f64>) {
    if let Some(dest) = rows.rb_mut().try_as_col_major_mut()
        && try_fill_lower_col(x, col, dest.as_slice_mut())
    {
        return;
    }
    for (i, row) in (col..x.nrows()).enumerate() {
        let mut sum = 0.0;
        for dim in 0..x.ncols() {
            let diff = x[(row, dim)] - x[(col, dim)];
            sum += diff * diff;
        }
        rows[i] = sum;
    }
}

/// The train–test squared distances of the test points
/// `start..start + dist_chunk.ncols()`.
fn fill_cross_chunk(
    x_train: MatRef<'_, f64>,
    x_test: MatRef<'_, f64>,
    mut dist_chunk: MatMut<'_, f64>,
    start: usize,
) {
    if try_fill_cross_chunk(x_train, x_test, dist_chunk.rb_mut(), start) {
        return;
    }
    for local in 0..dist_chunk.ncols() {
        let col = start + local;
        for row in 0..x_train.nrows() {
            let mut sum = 0.0;
            for dim in 0..x_train.ncols() {
                let diff = x_train[(row, dim)] - x_test[(col, dim)];
                sum += diff * diff;
            }
            dist_chunk[(row, local)] = sum;
        }
    }
}

fn copy_lower_to_upper(mut dist: MatMut<'_, f64>) {
    let n = dist.nrows();
    for col in 1..n {
        for row in 0..col {
            dist[(row, col)] = dist[(col, row)];
        }
    }
}

/// `f32` squared distances: the same formula as the `f64` fill, in scalar.
pub(crate) fn fill_squared_scalar(x: MatRef<'_, f32>, mut dist: MatMut<'_, f32>) {
    let n = x.nrows();
    let d = x.ncols();
    for col in 0..n {
        for row in 0..n {
            let mut sum = 0.0f32;
            for dim in 0..d {
                let diff = x[(row, dim)] - x[(col, dim)];
                sum += diff * diff;
            }
            dist[(row, col)] = sum;
        }
    }
}

pub(crate) fn fill_cross_scalar(
    x_train: MatRef<'_, f32>,
    x_test: MatRef<'_, f32>,
    mut dist: MatMut<'_, f32>,
) {
    let n = x_train.nrows();
    let m = x_test.nrows();
    let d = x_train.ncols();
    for col in 0..m {
        for row in 0..n {
            let mut sum = 0.0f32;
            for dim in 0..d {
                let diff = x_train[(row, dim)] - x_test[(col, dim)];
                sum += diff * diff;
            }
            dist[(row, col)] = sum;
        }
    }
}

pub(crate) fn fill_ard_scalar(x: MatRef<'_, f32>, cache: &mut [f32]) {
    let n = x.nrows();
    let mut slots = cache.iter_mut();
    for dim in 0..x.ncols() {
        for col in 0..n {
            for row in col..n {
                let diff = x[(row, dim)] - x[(col, dim)];
                if let Some(slot) = slots.next() {
                    *slot = diff * diff;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ArdSqDiffBuf, col_chunk, fill_squared_euclidean, fill_squared_euclidean_cross,
        par_lower_fold, worker_count,
    };
    use faer::Mat;

    fn sequential_sq(x: faer::MatRef<'_, f64>) -> Mat<f64> {
        let n = x.nrows();
        let d = x.ncols();
        let mut dist = Mat::zeros(n, n);
        for col in 0..n {
            for row in col..n {
                let mut sum = 0.0;
                for dim in 0..d {
                    let diff = x[(row, dim)] - x[(col, dim)];
                    sum += diff * diff;
                }
                dist[(row, col)] = sum;
                dist[(col, row)] = sum;
            }
        }
        dist
    }

    #[test]
    fn col_chunks_cover_n() {
        for n in [1, 2, 5, 7, 256] {
            for chunks in [1, 2, 3, 4, 8, worker_count()] {
                let chunks = chunks.max(1);
                let mut covered = 0;
                for idx in 0..chunks {
                    let (_start, len) = col_chunk(n, idx, chunks);
                    covered += len;
                }
                assert_eq!(covered, n, "n={n} chunks={chunks}");
            }
        }
    }

    #[test]
    fn lower_blocks_cover_every_column_once_with_balanced_area() {
        for n in [0, 1, 2, 3, 7, 64, 550, 1001] {
            for blocks in [1, 2, 3, 4, 8, 16, 37] {
                let starts: Vec<usize> = (0..=blocks)
                    .map(|i| super::lower_block_start(n, i, blocks))
                    .collect();
                assert_eq!(starts[0], 0);
                assert_eq!(starts[blocks], n);
                assert!(
                    starts.windows(2).all(|w| w[0] <= w[1]),
                    "n={n} blocks={blocks}"
                );
                let areas: Vec<u128> = starts
                    .windows(2)
                    .map(|w| super::lower_area(n, w[1]) - super::lower_area(n, w[0]))
                    .collect();
                // Each boundary is the column nearest its share, so every
                // block is within one column (≤ n entries) of the even share.
                let (k, total) = (blocks as u128, super::lower_area(n, n));
                for area in &areas {
                    assert!(
                        (area * k).abs_diff(total) <= n as u128 * k,
                        "n={n} blocks={blocks} areas={areas:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn par_lower_blocks_hands_every_column_to_one_block() {
        for n in [1, 5, 9, 130] {
            for blocks in [1, 3, 4, 16] {
                let mut out = Mat::<f64>::zeros(n, n);
                super::par_lower_blocks(out.as_mut(), blocks, &|start,
                                                                mut part: faer::MatMut<
                    '_,
                    f64,
                >| {
                    for local in 0..part.ncols() {
                        for row in start + local..n {
                            part[(row, local)] += (start + local) as f64 + 1.0;
                        }
                    }
                    Ok::<(), std::convert::Infallible>(())
                })
                .expect("infallible");
                for col in 0..n {
                    for row in 0..n {
                        let want = if row >= col { col as f64 + 1.0 } else { 0.0 };
                        assert_eq!(out[(row, col)].to_bits(), want.to_bits());
                    }
                }
            }
        }
    }

    /// Column sums added left to right, including pairs that do not commute
    /// under rounding. A balanced join of area blocks fails this.
    #[test]
    fn par_lower_fold_matches_a_serial_column_sum() {
        fn term(col: usize) -> f64 {
            if col.is_multiple_of(2) {
                1.0e16
            } else {
                -1.0e16 + col as f64
            }
        }
        for n in [0, 1, 2, 15, 16, 17, 64, 256] {
            let got = par_lower_fold(
                n,
                &|start, end| {
                    let mut sum = 0.0;
                    for col in start..end {
                        sum += term(col);
                    }
                    Ok::<f64, std::convert::Infallible>(sum)
                },
                &|a, b| a + b,
            )
            .expect("infallible");
            let mut expect = 0.0;
            for col in 0..n {
                expect += term(col);
            }
            assert_eq!(got.to_bits(), expect.to_bits(), "n={n}");
        }
    }

    #[test]
    fn col_chunk_five_by_four_covers_remainder() {
        // Remainder split: first `rem` chunks get `chunk_size + 1` columns.
        // n=5, n_chunks=4 → sizes 2, 1, 1, 1 (not four length-1 chunks).
        assert_eq!(col_chunk(5, 0, 4), (0, 2));
        assert_eq!(col_chunk(5, 1, 4), (2, 1));
        assert_eq!(col_chunk(5, 2, 4), (3, 1));
        assert_eq!(col_chunk(5, 3, 4), (4, 1));
        let covered: usize = (0..4).map(|idx| col_chunk(5, idx, 4).1).sum();
        assert_eq!(covered, 5);
    }

    #[test]
    fn fill_when_more_chunks_than_columns_matches_sequential() {
        let x = Mat::from_fn(5, 3, |r, c| (r as f64) * 0.1 + (c as f64) * 0.3);
        let expected = sequential_sq(x.as_ref());
        let mut dist = Mat::zeros(5, 5);
        let mut scratches = vec![Mat::<f64>::zeros(0, 0); 8];
        fill_squared_euclidean(x.as_ref(), dist.as_mut(), &mut scratches);
        for col in 0..5 {
            for row in 0..5 {
                assert!((dist[(row, col)] - expected[(row, col)]).abs() <= 1e-15);
            }
        }
    }

    #[test]
    fn parallel_fill_matches_sequential() {
        let x = Mat::from_fn(5, 3, |r, c| (r as f64) * 0.1 + (c as f64) * 0.3);
        let expected = sequential_sq(x.as_ref());
        let mut dist = Mat::zeros(5, 5);
        let mut scratches = vec![Mat::<f64>::zeros(0, 0); worker_count()];
        fill_squared_euclidean(x.as_ref(), dist.as_mut(), &mut scratches);
        for col in 0..5 {
            for row in 0..5 {
                assert!((dist[(row, col)] - expected[(row, col)]).abs() <= 1e-15);
            }
        }
    }

    #[test]
    fn parallel_cross_matches_sequential() {
        let x = Mat::from_fn(4, 2, |r, c| r as f64 + c as f64);
        let xs = Mat::from_fn(3, 2, |r, c| (r as f64) * 0.5 + c as f64);
        let mut expected = Mat::zeros(4, 3);
        for col in 0..3 {
            for row in 0..4 {
                let mut sum = 0.0;
                for dim in 0..2 {
                    let diff = x[(row, dim)] - xs[(col, dim)];
                    sum += diff * diff;
                }
                expected[(row, col)] = sum;
            }
        }
        let mut dist = Mat::zeros(4, 3);
        fill_squared_euclidean_cross(x.as_ref(), xs.as_ref(), dist.as_mut(), &mut []);
        for col in 0..3 {
            for row in 0..4 {
                assert!((dist[(row, col)] - expected[(row, col)]).abs() <= 1e-15);
            }
        }
    }

    #[test]
    fn ard_fill_matches_per_dim_squared_diff() {
        let x = Mat::from_fn(5, 3, |r, c| {
            (r as f64) * 0.1 + (c as f64) * 0.3 + (r * c) as f64
        });
        let n = 5;
        let d = 3;
        let cache = ArdSqDiffBuf::new(x.as_ref()).expect("size");
        let view = cache.view();
        assert_eq!((view.n(), view.d()), (n, d));
        // The lower triangle of each dimension only: d · n(n+1)/2, not d · n².
        assert_eq!(cache.stored_len(), d * n * (n + 1) / 2);
        for dim in 0..d {
            for col in 0..n {
                assert_eq!(view.column(dim, col).len(), n - col);
                for row in 0..n {
                    let diff = x[(row, dim)] - x[(col, dim)];
                    assert!((view.get(dim, row, col) - diff * diff).abs() <= 1e-15);
                }
            }
        }
        let x32 = Mat::from_fn(5, 3, |r, c| x[(r, c)] as f32);
        let cache32 = ArdSqDiffBuf::new(x32.as_ref()).expect("size");
        for dim in 0..d {
            for col in 0..n {
                for row in 0..n {
                    let diff = x32[(row, dim)] - x32[(col, dim)];
                    assert!((cache32.view().get(dim, row, col) - diff * diff).abs() <= 1e-6);
                }
            }
        }
    }
}
