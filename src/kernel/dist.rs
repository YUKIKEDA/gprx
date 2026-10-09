//! Describes the pairwise squared-Euclidean distances, filled by the process-wide thread pool column partitions.

use super::KernelScalar;
use super::simd::dist::{try_fill_ard_column, try_fill_cross_chunk, try_fill_lower_col};
use crate::error::GprError;
use faer::reborrow::ReborrowMut;
use faer::{ColMut, Mat, MatMut, MatRef};
use rayon::prelude::*;
use std::convert::Infallible;
use std::fmt;
use std::marker::PhantomData;
use wide::f64x4;

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
/// Each dimension keeps one triangle (diagonal included) in one of three
/// layouts ([`ArdStore`]): the lower triangle packed as column runs
/// (column `col` holds rows `col..n`), the dense `n × n` tables a caller
/// handed over (kept as they are so a fit copies nothing), or, once an
/// online model grows or shrinks the cache, row runs (row `i` holds
/// columns `0..=i`), to which a new point is one contiguous run. Read it
/// through [`Self::view`].
#[derive(Clone, Debug)]
pub(crate) struct ArdSqDiffBuf<T> {
    data: ArdStore<T>,
    n: usize,
    d: usize,
}

/// The layout of an [`ArdSqDiffBuf`].
#[derive(Debug)]
enum ArdStore<T> {
    /// The lower triangles, dimension after dimension, column by column:
    /// column `col` holds rows `col..n`.
    Packed(Vec<T>),
    /// One dense column-major `n × n` table per dimension; only the lower
    /// triangle is read.
    Dense(Vec<Vec<T>>),
    /// One buffer per dimension, row by row: row `i` holds columns
    /// `0..=i` and starts at `i(i+1)/2`. A buffer's spare capacity is the
    /// room for later points.
    Rows(Vec<Vec<T>>),
}

/// A clone keeps the room the row runs reserved, so a cloned online model
/// grows as the original does.
impl<T: Clone> Clone for ArdStore<T> {
    fn clone(&self) -> Self {
        match self {
            Self::Packed(data) => Self::Packed(data.clone()),
            Self::Dense(tables) => Self::Dense(tables.clone()),
            Self::Rows(rows) => Self::Rows(
                rows.iter()
                    .map(|dim| {
                        let mut copy = Vec::with_capacity(dim.capacity());
                        copy.extend_from_slice(dim);
                        copy
                    })
                    .collect(),
            ),
        }
    }
}

/// Offset of row `i` in a row-packed lower triangle.
#[inline]
fn row_offset(i: usize) -> usize {
    i * (i + 1) / 2
}

/// Columns transposed together when column runs become row runs.
const TRANSPOSE_TILE: usize = 16;

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
        Ok(Self::from_packed(data, n, d))
    }

    /// Packs the lower triangles of `d` dense `n × n` blocks; `pair(k, i, j)`
    /// is `(Δ_k)²` of the pair `(i, j)`, `i ≥ j`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::SizeOverflow`] when `d · n(n+1)/2` overflows.
    #[cfg(test)]
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
        Ok(Self::from_packed(data, n, d))
    }

    /// Packs the lower triangles of `d` dense, column-major `n × n` blocks
    /// (`block(k)` is dimension `k`), one contiguous run per column.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::SizeOverflow`] when `d · n(n+1)/2` overflows.
    pub(crate) fn from_dense<'b>(
        n: usize,
        d: usize,
        block: impl Fn(usize) -> &'b [f64],
    ) -> Result<Self, GprError> {
        let len = packed_len(n)?
            .checked_mul(d)
            .ok_or(GprError::SizeOverflow)?;
        let mut data = Vec::with_capacity(len);
        for k in 0..d {
            let block = block(k);
            for col in 0..n {
                data.extend(
                    block[col * n + col..(col + 1) * n]
                        .iter()
                        .map(|&v| T::from_f64(v)),
                );
            }
        }
        Ok(Self::from_packed(data, n, d))
    }

    /// A cache of `n` points and `d` dimensions from its packed values:
    /// dimension after dimension, each the lower triangle column by column.
    pub(crate) fn from_packed(data: Vec<T>, n: usize, d: usize) -> Self {
        debug_assert_eq!(
            Some(data.len()),
            packed_len(n).ok().and_then(|l| l.checked_mul(d))
        );
        Self {
            data: ArdStore::Packed(data),
            n,
            d,
        }
    }

    /// A cache of `n` points from `d` dense column-major `n × n` tables,
    /// kept as they are (one per dimension).
    pub(crate) fn from_tables(tables: Vec<Vec<T>>, n: usize) -> Self {
        debug_assert!(tables.iter().all(|t| Some(t.len()) == n.checked_mul(n)));
        Self {
            d: tables.len(),
            data: ArdStore::Dense(tables),
            n,
        }
    }

    /// The same cache with every value mapped by `f` (a cast), in one pass
    /// over the stored triangles: row runs stay row runs, and the other
    /// layouts are packed as column runs.
    pub(crate) fn map<U>(&self, f: impl Fn(T) -> U) -> ArdSqDiffBuf<U>
    where
        T: Copy,
    {
        let data = match &self.data {
            ArdStore::Packed(data) => ArdStore::Packed(data.iter().map(|&v| f(v)).collect()),
            ArdStore::Rows(rows) => ArdStore::Rows(
                rows.iter()
                    .map(|dim| dim.iter().map(|&v| f(v)).collect())
                    .collect(),
            ),
            ArdStore::Dense(tables) => {
                let n = self.n;
                ArdStore::Packed(
                    tables
                        .iter()
                        .flat_map(|t| (0..n).flat_map(move |col| &t[col * n + col..(col + 1) * n]))
                        .map(|&v| f(v))
                        .collect(),
                )
            }
        };
        ArdSqDiffBuf {
            data,
            n: self.n,
            d: self.d,
        }
    }

    /// A cache of `n` points and `d` dimensions holding zeros, to be
    /// written column by column through [`Self::column_mut`].
    ///
    /// # Errors
    ///
    /// Returns [`GprError::SizeOverflow`] when `d · n(n+1)/2` overflows.
    pub(crate) fn zeros(n: usize, d: usize) -> Result<Self, GprError> {
        let len = packed_len(n)?
            .checked_mul(d)
            .ok_or(GprError::SizeOverflow)?;
        Ok(Self::from_packed(vec![T::from_f64(0.0); len], n, d))
    }

    /// The stored rows `col..n` of column `col` of dimension `dim`, of a
    /// cache [`Self::zeros`] made (column runs).
    pub(crate) fn column_mut(&mut self, dim: usize, col: usize) -> &mut [T] {
        let n = self.n;
        match &mut self.data {
            ArdStore::Packed(data) => {
                let block = data.len().checked_div(self.d).unwrap_or(0);
                let start = dim * block + packed_col_offset(n, col);
                &mut data[start..start + (n - col)]
            }
            ArdStore::Dense(tables) => &mut tables[dim][col * n + col..(col + 1) * n],
            // Only a cache of column runs is written column by column.
            ArdStore::Rows(_) => &mut [],
        }
    }

    /// Makes room for one more point: the cache becomes row runs (once,
    /// for a cache of column runs or dense tables), and each dimension's
    /// buffer reserves the new row, so [`Self::push_point`] does not
    /// allocate. A buffer grows as a `Vec` does, by doubling.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::SizeOverflow`] when the grown cache does not fit.
    pub(crate) fn reserve_point(&mut self) -> Result<(), GprError> {
        self.lay_out_rows()?;
        let n = self.n;
        if let ArdStore::Rows(rows) = &mut self.data {
            let more = n.checked_add(1).ok_or(GprError::SizeOverflow)?;
            for dim in rows {
                dim.try_reserve(more).map_err(|_| GprError::SizeOverflow)?;
            }
        }
        Ok(())
    }

    /// Lays the cache out as row runs, unless it is already, so a later
    /// [`Self::remove_point`] does not fail.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::SizeOverflow`] when the row runs do not fit.
    pub(crate) fn ready_to_change(&mut self) -> Result<(), GprError> {
        self.lay_out_rows()
    }

    /// Lays the cache out as row runs, unless it is already.
    fn lay_out_rows(&mut self) -> Result<(), GprError> {
        if matches!(self.data, ArdStore::Rows(_)) {
            return Ok(());
        }
        let n = self.n;
        let len = packed_len(n)?;
        let view = self.view();
        let Some(lower) = view.lower() else {
            return Ok(());
        };
        let mut rows = Vec::with_capacity(self.d);
        for dim in 0..self.d {
            let mut out = Vec::new();
            out.try_reserve_exact(packed_len(n + 1)?)
                .map_err(|_| GprError::SizeOverflow)?;
            out.resize(len, T::from_f64(0.0));
            // Columns `j0..j1` at a time: each row takes a contiguous run
            // of them, read from as many column runs.
            let mut j0 = 0;
            while j0 < n {
                let j1 = (j0 + TRANSPOSE_TILE).min(n);
                for i in j0..n {
                    let base = row_offset(i);
                    for j in j0..j1.min(i + 1) {
                        out[base + j] = lower.column(dim, j)[i - j];
                    }
                }
                j0 = j1;
            }
            rows.push(out);
        }
        self.data = ArdStore::Rows(rows);
        Ok(())
    }

    /// Appends point `n`: `column(dim)` holds its `(Δ_dim)²` to the points
    /// `0..n` (at least `n` values); its own diagonal is zero. Each
    /// dimension's buffer takes one contiguous run, in the room
    /// [`Self::reserve_point`] made.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::SizeOverflow`] when no room was reserved (a
    /// crate bug: the store is left as it was).
    pub(crate) fn push_point<'c>(
        &mut self,
        column: impl Fn(usize) -> &'c [T],
    ) -> Result<(), GprError>
    where
        T: 'c,
    {
        let n = self.n;
        let ArdStore::Rows(rows) = &mut self.data else {
            return Err(GprError::SizeOverflow);
        };
        if rows.iter().any(|dim| dim.capacity() - dim.len() < n + 1) {
            return Err(GprError::SizeOverflow);
        }
        for (dim, buffer) in rows.iter_mut().enumerate() {
            buffer.extend_from_slice(&column(dim)[..n]);
            buffer.push(T::from_f64(0.0));
        }
        self.n = n + 1;
        Ok(())
    }

    /// Removes point `index` in place: row `index` goes, each later row
    /// loses its column `index`, and the later points move up one. One
    /// forward pass over the rows past `index`; rows before it stay put.
    /// A cache of column runs or dense tables becomes row runs first.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::IndexOutOfRange`] when `index ≥ n`, and
    /// [`GprError::SizeOverflow`] when the cache cannot be laid out as row
    /// runs.
    pub(crate) fn remove_point(&mut self, index: usize) -> Result<(), GprError> {
        let n = self.n;
        if index >= n {
            return Err(GprError::IndexOutOfRange {
                reason: format!("point index {index} is out of range for n={n}"),
            });
        }
        self.lay_out_rows()?;
        let ArdStore::Rows(rows) = &mut self.data else {
            return Err(GprError::SizeOverflow);
        };
        let compact = |buffer: &mut Vec<T>| {
            let mut write = row_offset(index);
            for i in index + 1..n {
                let start = row_offset(i);
                // Columns `0..index`, then `index + 1..=i`.
                buffer.copy_within(start..start + index, write);
                write += index;
                buffer.copy_within(start + index + 1..start + i + 1, write);
                write += i - index;
            }
            buffer.truncate(write);
        };
        // From inside the pool (a delete beside the factor's update), the
        // dimensions' buffers move side by side.
        if rows.len() > 1 && rayon::current_thread_index().is_some() {
            rows.par_iter_mut().for_each(compact);
        } else {
            rows.iter_mut().for_each(compact);
        }
        self.n = n - 1;
        Ok(())
    }

    /// `(points, dimensions)` the cache was filled for.
    #[cfg(test)]
    pub(crate) fn shape(&self) -> (usize, usize) {
        (self.n, self.d)
    }

    /// Number of stored values.
    #[cfg(test)]
    pub(crate) fn stored_len(&self) -> usize {
        match &self.data {
            ArdStore::Packed(data) => data.len(),
            ArdStore::Dense(tables) | ArdStore::Rows(tables) => tables.iter().map(Vec::len).sum(),
        }
    }

    /// Whether the cache keeps the caller's dense tables.
    #[cfg(test)]
    pub(crate) fn is_dense(&self) -> bool {
        matches!(self.data, ArdStore::Dense(_))
    }

    /// Overwrites every cached value, to show that a reader uses the cache.
    #[cfg(test)]
    pub(crate) fn poison(&mut self, value: T) {
        match &mut self.data {
            ArdStore::Packed(data) => data.fill(value),
            ArdStore::Dense(tables) | ArdStore::Rows(tables) => {
                tables.iter_mut().for_each(|t| t.fill(value));
            }
        }
    }

    pub(crate) fn view(&self) -> ArdSqDiff<'_, T> {
        let data = match &self.data {
            ArdStore::Packed(data) => {
                StoreRef::Packed(data, data.len().checked_div(self.d).unwrap_or(0))
            }
            ArdStore::Dense(tables) => StoreRef::Dense(tables),
            ArdStore::Rows(rows) => StoreRef::Rows(rows),
        };
        ArdSqDiff {
            data,
            n: self.n,
            d: self.d,
        }
    }
}

/// Borrowed raw `(Δx_d)²` cache of [`ArdSqDiffBuf`].
///
/// [`Self::get`] reads any pair, in either order. A cache of column runs
/// hands out [`Self::lower`], one of row runs [`Self::rows`]: the loops
/// that read whole runs take the layout's own.
#[derive(Clone, Copy, Debug)]
pub struct ArdSqDiff<'a, T> {
    data: StoreRef<'a, T>,
    n: usize,
    d: usize,
}

/// The borrowed layout of an [`ArdSqDiff`].
#[derive(Clone, Copy, Debug)]
enum StoreRef<'a, T> {
    /// Packed lower triangles, with the entries per dimension.
    Packed(&'a [T], usize),
    /// Dense `n × n` tables, one per dimension.
    Dense(&'a [Vec<T>]),
    /// Dense `n × n` tables, one per dimension, one after another.
    Flat(&'a [T]),
    /// Row runs, one buffer per dimension.
    Rows(&'a [Vec<T>]),
}

/// The column runs of a cache: column `col` holds rows `col..n`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LowerRuns<'a, T> {
    data: StoreRef<'a, T>,
    n: usize,
}

/// The row runs of a cache: row `i` holds columns `0..=i`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RowRuns<'a, T> {
    rows: &'a [Vec<T>],
    n: usize,
}

impl<'a, T> LowerRuns<'a, T> {
    /// `(x_row,dim − x_col,dim)²` for rows `col..n`, in row order.
    #[inline]
    pub(crate) fn column(&self, dim: usize, col: usize) -> &'a [T] {
        let n = self.n;
        match self.data {
            StoreRef::Packed(data, block) => {
                let start = dim * block + packed_col_offset(n, col);
                &data[start..start + (n - col)]
            }
            StoreRef::Dense(tables) => &tables[dim][col * n + col..(col + 1) * n],
            StoreRef::Flat(data) => &data[dim * n * n + col * n + col..dim * n * n + (col + 1) * n],
            StoreRef::Rows(_) => &[],
        }
    }

    /// Every stored value of dimension `dim`, when the cache is packed:
    /// the lower triangle, column by column.
    #[inline]
    pub(crate) fn packed_block(&self, dim: usize) -> Option<&'a [T]> {
        match self.data {
            StoreRef::Packed(data, block) => Some(&data[dim * block..(dim + 1) * block]),
            StoreRef::Dense(_) | StoreRef::Flat(_) | StoreRef::Rows(_) => None,
        }
    }
}

impl<'a, T> RowRuns<'a, T> {
    /// Number of points.
    pub(crate) fn n(&self) -> usize {
        self.n
    }

    /// Every stored value of dimension `dim`: row after row.
    #[inline]
    pub(crate) fn buffer(&self, dim: usize) -> &'a [T] {
        &self.rows[dim][..row_offset(self.n)]
    }

    /// `(x_i,dim − x_col,dim)²` for columns `0..=i`, in column order.
    #[inline]
    pub(crate) fn row(&self, dim: usize, i: usize) -> &'a [T] {
        let start = row_offset(i);
        &self.rows[dim][start..start + i + 1]
    }
}

impl<'a, T: KernelScalar> ArdSqDiff<'a, T> {
    /// The `d` dense checked `n × n` squares of `data`, one after another
    /// (only the lower triangles are read).
    pub(crate) fn flat(data: &'a [T], n: usize) -> Self {
        let d = data.len().checked_div(n * n).unwrap_or(0);
        Self {
            data: StoreRef::Flat(data),
            n,
            d,
        }
    }

    /// Number of points.
    pub(crate) fn n(&self) -> usize {
        self.n
    }

    /// Number of dimensions.
    pub(crate) fn d(&self) -> usize {
        self.d
    }

    /// The column runs, when the cache holds them (packed or dense).
    #[inline]
    pub(crate) fn lower(&self) -> Option<LowerRuns<'a, T>> {
        match self.data {
            StoreRef::Packed(..) | StoreRef::Dense(_) | StoreRef::Flat(_) => Some(LowerRuns {
                data: self.data,
                n: self.n,
            }),
            StoreRef::Rows(_) => None,
        }
    }

    /// The row runs, when the cache holds them.
    #[inline]
    pub(crate) fn rows(&self) -> Option<RowRuns<'a, T>> {
        match self.data {
            StoreRef::Rows(rows) => Some(RowRuns { rows, n: self.n }),
            StoreRef::Packed(..) | StoreRef::Dense(_) | StoreRef::Flat(_) => None,
        }
    }

    /// `(x_row,dim − x_col,dim)²` for any pair.
    #[inline]
    pub(crate) fn get(&self, dim: usize, row: usize, col: usize) -> T {
        let (row, col) = if row >= col { (row, col) } else { (col, row) };
        match self.data {
            StoreRef::Rows(rows) => rows[dim][row_offset(row) + col],
            StoreRef::Packed(..) | StoreRef::Dense(_) | StoreRef::Flat(_) => LowerRuns {
                data: self.data,
                n: self.n,
            }
            .column(dim, col)[row - col],
        }
    }

    /// The first stored pair `(row, col)`, `row ≥ col`, of dimension `dim`
    /// whose value `bad` picks, in the layout's own order.
    pub(crate) fn position(&self, dim: usize, bad: impl Fn(T) -> bool) -> Option<(usize, usize)> {
        if let Some(rows) = self.rows() {
            return (0..self.n).find_map(|i| {
                rows.row(dim, i)
                    .iter()
                    .position(|&v| bad(v))
                    .map(|j| (i, j))
            });
        }
        let lower = self.lower()?;
        (0..self.n).find_map(|j| {
            lower
                .column(dim, j)
                .iter()
                .position(|&v| bad(v))
                .map(|k| (j + k, j))
        })
    }

    /// The same cache as `f64` when `T` is `f64`, for the SIMD paths.
    pub(crate) fn as_f64(self) -> Option<ArdSqDiff<'a, f64>> {
        let data = match self.data {
            StoreRef::Packed(data, block) => StoreRef::Packed(T::as_f64_slice(data)?, block),
            StoreRef::Dense(tables) => StoreRef::Dense(T::as_f64_vecs(tables)?),
            StoreRef::Flat(data) => StoreRef::Flat(T::as_f64_slice(data)?),
            StoreRef::Rows(rows) => StoreRef::Rows(T::as_f64_vecs(rows)?),
        };
        Some(ArdSqDiff {
            data,
            n: self.n,
            d: self.d,
        })
    }
}

/// Whether the values of an [`ArdBlocks`] were checked when they were bound.
///
/// [`Checked`] blocks (the training triangles, a cast, a repaired or
/// filled table) are read as they are. [`Unchecked`] blocks (a caller's
/// prediction block that an `f64` model reads in place) are checked as
/// they are read, so the caller's values are read once: their values come
/// out only through [`ArdBlocks::read`], [`ArdBlocks::column`], and
/// [`ArdBlocks::gates`], each of which checks what it hands out.
pub trait BlockState: Copy + fmt::Debug + Send + Sync + 'static + sealed::Sealed {
    /// Whether the values were checked when bound.
    const CHECKED: bool;
}

/// Blocks whose values were checked when bound.
#[derive(Clone, Copy, Debug)]
pub enum Checked {}

/// Blocks whose values are checked as they are read.
#[derive(Clone, Copy, Debug)]
pub enum Unchecked {}

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::Checked {}
    impl Sealed for super::Unchecked {}
}

impl BlockState for Checked {
    const CHECKED: bool = true;
}

impl BlockState for Unchecked {
    const CHECKED: bool = false;
}

/// Raw `(Δ_d)²` of a rectangular block (`rows × cols`), one dense
/// column-major block per dimension: `(row, col)` of dimension `k` is
/// `block(k)[row + col * rows]`. `S` says whether the values were checked
/// when bound ([`BlockState`]).
#[derive(Clone, Copy, Debug)]
pub struct ArdBlocks<'a, T, S = Checked> {
    /// One block per dimension.
    blocks: BlockList<'a, T>,
    rows: usize,
    cols: usize,
    /// First column of the stored blocks this view starts at.
    col0: usize,
    state: PhantomData<S>,
}

/// The per-dimension blocks of an [`ArdBlocks`]: a caller's borrowed or
/// moved blocks, blocks packed one after another (a fill, a repair, or a
/// cast), or one block repeated.
#[derive(Clone, Copy, Debug)]
pub(crate) enum BlockList<'a, T> {
    Slices(&'a [&'a [T]]),
    /// Owned blocks a caller moved in.
    Vecs(&'a [Vec<T>]),
    /// `dims` blocks of `len` values each, one after another.
    Packed(&'a [T], usize, usize),
    /// One block for each of `dims` dimensions.
    Repeat(&'a [T], usize),
    /// The packed training triangles: pair `(row, col)` of the square,
    /// read in either order. No dense block exists.
    Triangles(ArdSqDiff<'a, T>),
}

impl<'a, T: KernelScalar, S: BlockState> ArdBlocks<'a, T, S> {
    /// `blocks` of `rows × cols` pairs in the state `S`. The type does not
    /// prove the state: only binding code makes a `Checked` block, from
    /// values it checked, cast, or packed.
    pub(crate) fn new(blocks: BlockList<'a, T>, rows: usize, cols: usize, col0: usize) -> Self {
        Self {
            blocks,
            rows,
            cols,
            col0,
            state: PhantomData,
        }
    }

    /// Columns `start..start + len` of these blocks.
    pub(crate) fn subcols(self, start: usize, len: usize) -> Self {
        Self {
            cols: len,
            col0: self.col0 + start,
            ..self
        }
    }

    /// Number of dimensions.
    pub(crate) fn d(&self) -> usize {
        match self.blocks {
            BlockList::Slices(blocks) => blocks.len(),
            BlockList::Vecs(blocks) => blocks.len(),
            BlockList::Packed(_, _, dims) | BlockList::Repeat(_, dims) => dims,
            BlockList::Triangles(cache) => cache.d(),
        }
    }

    /// Rows of the view.
    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    /// Columns of the view.
    pub(crate) fn cols(&self) -> usize {
        self.cols
    }

    /// The dense block of dimension `dim`, or an empty slice when the
    /// blocks are packed triangles. Raw: only [`Checked`] blocks expose it.
    fn raw_block(&self, dim: usize) -> &'a [T] {
        match self.blocks {
            BlockList::Slices(blocks) => blocks[dim],
            BlockList::Vecs(blocks) => &blocks[dim],
            BlockList::Packed(all, len, _) => &all[dim * len..(dim + 1) * len],
            BlockList::Repeat(block, _) => block,
            BlockList::Triangles(_) => &[],
        }
    }

    fn raw_get(&self, dim: usize, row: usize, col: usize) -> T {
        match self.blocks {
            BlockList::Triangles(cache) => cache.get(dim, row, col + self.col0),
            _ => self.raw_block(dim)[row + (col + self.col0) * self.rows],
        }
    }

    /// `(Δ_dim)²` of the pair `(row, col)`, checked unless the blocks were.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidDistance`] if the value is not finite or
    /// is negative, at its place in the caller's table.
    #[inline]
    pub(crate) fn read(&self, dim: usize, row: usize, col: usize) -> Result<T, GprError> {
        let v = self.raw_get(dim, row, col);
        if S::CHECKED || super::sources::valid(v.to_f64()) {
            Ok(v)
        } else {
            Err(super::sources::invalid_value(
                v.to_f64(),
                row,
                col + self.col0,
            ))
        }
    }

    /// Column `col` of dimension `dim` (every row) as an `f64` slice,
    /// checked unless the blocks were; `None` when the blocks are not dense
    /// `f64` columns.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidDistance`] at the first invalid value.
    pub(crate) fn column(&self, dim: usize, col: usize) -> Result<Option<&'a [f64]>, GprError> {
        let Some(run) = self.f64_column(dim, col) else {
            return Ok(None);
        };
        if !S::CHECKED && !super::simd::all_valid_distances(run) {
            return Err(super::sources::first_invalid_from(
                run,
                self.rows,
                self.col0 + col,
            ));
        }
        Ok(Some(run))
    }

    /// Whether every block is a dense `f64` block holding all the view's
    /// columns, as [`Self::column`] and [`Self::gate`] read them.
    pub(crate) fn dense_f64(&self) -> bool {
        let end = (self.col0 + self.cols) * self.rows;
        !matches!(self.blocks, BlockList::Triangles(_))
            && T::as_f64_slice(&[]).is_some()
            && (0..self.d()).all(|dim| self.raw_block(dim).len() >= end)
    }

    fn f64_column(&self, dim: usize, col: usize) -> Option<&'a [f64]> {
        if matches!(self.blocks, BlockList::Triangles(_)) {
            return None;
        }
        let start = (self.col0 + col) * self.rows;
        T::as_f64_slice(self.raw_block(dim).get(start..start + self.rows)?)
    }

    /// The dense `f64` blocks of every dimension, from which a SIMD loop
    /// takes one [`Gate`] per column; `None` when the blocks are not dense
    /// `f64` columns or have more than [`GATE_DIMS`] dimensions.
    pub(crate) fn gates(&self) -> Option<Gates<'a, S>> {
        let d = self.d();
        if d > GATE_DIMS || !self.dense_f64() {
            return None;
        }
        let mut blocks: [&'a [f64]; GATE_DIMS] = [&[]; GATE_DIMS];
        for (dim, block) in blocks.iter_mut().enumerate().take(d) {
            *block = T::as_f64_slice(self.raw_block(dim))?;
        }
        Some(Gates {
            blocks,
            d,
            rows: self.rows,
            col0: self.col0,
            state: PhantomData,
        })
    }
}

impl<'a, T: KernelScalar> ArdBlocks<'a, T, Checked> {
    /// The dense block of dimension `dim` (`rows` rows per column), or an
    /// empty slice when the blocks are packed triangles.
    pub(crate) fn block(&self, dim: usize) -> &'a [T] {
        self.raw_block(dim)
    }
}

/// The dense `f64` blocks of an [`ArdBlocks`] ([`ArdBlocks::gates`]).
#[derive(Clone, Copy)]
pub(crate) struct Gates<'a, S> {
    blocks: [&'a [f64]; GATE_DIMS],
    d: usize,
    rows: usize,
    col0: usize,
    state: PhantomData<S>,
}

impl<'a, S: BlockState> Gates<'a, S> {
    /// The column `col` of every dimension, for one pass of a SIMD loop that
    /// reads each value once: a [`Gate`] folds the check of every value it
    /// sums (unless the blocks were checked), and [`Gate::verdict`] gives
    /// the result.
    #[inline]
    pub(crate) fn gate(&self, col: usize) -> Gate<'a, S> {
        let start = (self.col0 + col) * self.rows;
        let mut runs: [&'a [f64]; GATE_DIMS] = [&[]; GATE_DIMS];
        for (run, block) in runs.iter_mut().zip(&self.blocks[..self.d]) {
            *run = &block[start..start + self.rows];
        }
        Gate {
            runs,
            d: self.d,
            nonfinite: f64x4::ZERO,
            least: f64x4::ZERO,
            rows: self.rows,
            col: self.col0 + col,
            state: PhantomData,
        }
    }
}

/// Dimensions a [`Gate`] holds.
pub(crate) const GATE_DIMS: usize = 16;

/// One column of every dimension of an [`ArdBlocks`], handed to a SIMD loop
/// that reads each value once. The loop gets only weighted sums of the
/// values ([`Self::weighted4`], [`Self::weighted1`]); for [`Unchecked`]
/// blocks each value is folded into the check as it is summed (two lane
/// sums without a branch: `v · 0`, which stays `0` unless a value is `NaN`
/// or infinite, and the least value, which stays `≥ 0` unless one is
/// negative), and [`Self::verdict`] reports the result.
#[must_use = "a gate's sums are valid only once its verdict is Ok"]
pub(crate) struct Gate<'a, S> {
    runs: [&'a [f64]; GATE_DIMS],
    d: usize,
    nonfinite: f64x4,
    least: f64x4,
    rows: usize,
    col: usize,
    state: PhantomData<S>,
}

impl<S: BlockState> Gate<'_, S> {
    /// `Σ_d w_d src_d` of rows `i..i + 4`.
    #[inline(always)]
    pub(crate) fn weighted4(&mut self, w: &[f64], i: usize) -> f64x4 {
        let mut r2 = f64x4::ZERO;
        for (run, &wd) in self.runs[..self.d].iter().zip(w) {
            let v = super::simd::load4(run, i);
            if !S::CHECKED {
                self.nonfinite += v * f64x4::ZERO;
                self.least = self.least.fast_min(v);
            }
            r2 += v * f64x4::splat(wd);
        }
        r2
    }

    /// `Σ_d w_d src_d` of row `i`.
    #[inline(always)]
    pub(crate) fn weighted1(&mut self, w: &[f64], i: usize) -> f64 {
        let mut r2 = 0.0;
        for (run, &wd) in self.runs[..self.d].iter().zip(w) {
            let v = run[i];
            if !S::CHECKED {
                self.nonfinite += f64x4::splat(v * 0.0);
                self.least = self.least.fast_min(f64x4::splat(v));
            }
            r2 += v * wd;
        }
        r2
    }

    /// Whether every value summed was valid.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidDistance`] at the first invalid value of
    /// the column, when one was summed.
    pub(crate) fn verdict(self) -> Result<(), GprError> {
        if S::CHECKED {
            return Ok(());
        }
        // Exact, not a tolerance: a sum of `v · 0` is `0` or `NaN`.
        let valid =
            self.nonfinite.reduce_add() == 0.0 && self.least.to_array().iter().all(|&l| l >= 0.0);
        if valid {
            return Ok(());
        }
        for run in &self.runs[..self.d] {
            if !super::simd::all_valid_distances(run) {
                return Err(super::sources::first_invalid_from(run, self.rows, self.col));
            }
        }
        // The lanes failed, yet no run holds an invalid value: the two
        // checks disagree. Refuse without claiming a place.
        Err(super::sources::unlocated())
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
    use crate::error::GprError;
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
                assert_eq!(view.lower().expect("lower").column(dim, col).len(), n - col);
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

    /// Coordinate `k` of pool point `p`, and the pair of two pool points.
    fn pool_pair(k: usize, a: usize, b: usize) -> f64 {
        let at = |p: usize| (p as f64 * (0.41 + 0.17 * k as f64)).sin() * (1.0 + k as f64);
        let diff = at(a) - at(b);
        diff * diff
    }

    /// Every pair of `cache` is the pair of the pool points `live` names.
    fn assert_holds(cache: &ArdSqDiffBuf<f64>, live: &[usize], d: usize) {
        let view = cache.view();
        assert_eq!((view.n(), view.d()), (live.len(), d));
        for k in 0..d {
            for (i, &a) in live.iter().enumerate() {
                for (j, &b) in live.iter().enumerate() {
                    assert_eq!(
                        view.get(k, i, j).to_bits(),
                        pool_pair(k, a, b).to_bits(),
                        "{k} ({i}, {j})"
                    );
                }
            }
        }
    }

    /// Inserts grow the row runs in their reserved room, and deletes at
    /// the head, the middle, and the tail compact them in place, from a
    /// cache of column runs and from dense tables.
    #[test]
    fn row_runs_grow_and_shrink_in_place() {
        let d = 3;
        let start: Vec<usize> = (0..5).collect();
        let packed = ArdSqDiffBuf::<f64>::from_pairs(5, d, pool_pair).expect("cache");
        let tables: Vec<Vec<f64>> = (0..d)
            .map(|k| (0..25).map(|at| pool_pair(k, at % 5, at / 5)).collect())
            .collect();
        let dense = ArdSqDiffBuf::<f64>::from_tables(tables, 5);
        for mut cache in [packed, dense] {
            let mut live = start.clone();
            assert_holds(&cache, &live, d);
            let mut next = 5;
            for step in 0..12 {
                if step % 3 == 2 {
                    let index = [0, live.len() / 2, live.len() - 1][step % 9 / 3];
                    cache.remove_point(index).expect("remove");
                    live.remove(index);
                } else {
                    cache.reserve_point().expect("room");
                    let columns: Vec<Vec<f64>> = (0..d)
                        .map(|k| live.iter().map(|&a| pool_pair(k, a, next)).collect())
                        .collect();
                    cache.push_point(|k| &columns[k]).expect("push");
                    live.push(next);
                    next += 1;
                }
                assert!(cache.view().rows().is_some());
                assert_holds(&cache, &live, d);
            }
        }
    }

    /// A clone keeps the room its original reserved; a push without room
    /// and a delete out of range are refused, leaving the cache as it was.
    #[test]
    fn row_runs_refuse_a_push_without_room_and_clone_their_room() {
        let mut cache = ArdSqDiffBuf::<f64>::from_pairs(4, 2, pool_pair).expect("cache");
        let column: Vec<f64> = (0..4).map(|a| pool_pair(0, a, 4)).collect();
        assert!(matches!(
            cache.push_point(|_| &column),
            Err(GprError::SizeOverflow)
        ));
        assert!(matches!(
            cache.remove_point(4),
            Err(GprError::IndexOutOfRange { .. })
        ));
        cache.reserve_point().expect("room");
        let mut copy = cache.clone();
        let columns: Vec<Vec<f64>> = (0..2)
            .map(|k| (0..4).map(|a| pool_pair(k, a, 4)).collect())
            .collect();
        copy.push_point(|k| &columns[k])
            .expect("the clone has room");
        assert_holds(&copy, &[0, 1, 2, 3, 4], 2);
        assert_holds(&cache, &[0, 1, 2, 3], 2);
        // A cast keeps the row runs.
        let cast = copy.map(|v| v as f32);
        assert!(cast.view().rows().is_some());
        assert_eq!(
            cast.view().get(1, 4, 2).to_bits(),
            (pool_pair(1, 4, 2) as f32).to_bits()
        );
        assert_eq!(
            copy.view()
                .position(1, |v| v.to_bits() == pool_pair(1, 4, 2).to_bits()),
            Some((4, 2))
        );
    }
}
