//! Pairwise squared-Euclidean distances, filled by Rayon column partitions.

use super::KernelScalar;
use super::simd::{try_fill_ard_column, try_fill_cross_chunk, try_fill_lower_chunk};
use crate::error::GprError;
use faer::reborrow::ReborrowMut;
use faer::{Mat, MatMut, MatRef};
use rayon::prelude::*;

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
    if thread_scratch.is_empty() {
        dist.rb_mut()
            .par_col_partition_mut(n_parts)
            .enumerate()
            .for_each(|(chunk_idx, part)| {
                fill_lower_chunk(x, part, chunk_idx, n_parts);
            });
    } else {
        dist.rb_mut()
            .par_col_partition_mut(n_parts)
            .zip(thread_scratch.par_iter_mut())
            .enumerate()
            .for_each(|(chunk_idx, (part, _scratch))| {
                fill_lower_chunk(x, part, chunk_idx, n_parts);
            });
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
pub(crate) struct ArdSqDiff<'a, T> {
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
        fill_cross_chunk(x_train, x_test, dist.rb_mut(), 0, 1);
        return;
    }
    let n_parts = partition_count(thread_scratch);
    if thread_scratch.is_empty() {
        dist.rb_mut()
            .par_col_partition_mut(n_parts)
            .enumerate()
            .for_each(|(chunk_idx, part)| {
                fill_cross_chunk(x_train, x_test, part, chunk_idx, n_parts);
            });
    } else {
        dist.rb_mut()
            .par_col_partition_mut(n_parts)
            .zip(thread_scratch.par_iter_mut())
            .enumerate()
            .for_each(|(chunk_idx, (part, _scratch))| {
                fill_cross_chunk(x_train, x_test, part, chunk_idx, n_parts);
            });
    }
}

fn fill_lower_chunk(
    x: MatRef<'_, f64>,
    mut dist_chunk: MatMut<'_, f64>,
    chunk_idx: usize,
    n_chunks: usize,
) {
    let n = x.nrows();
    let d = x.ncols();
    let (start, len) = col_chunk(n, chunk_idx, n_chunks);
    debug_assert_eq!(dist_chunk.ncols(), len);
    if try_fill_lower_chunk(x, dist_chunk.rb_mut(), chunk_idx, n_chunks) {
        return;
    }
    for local in 0..len {
        let col = start + local;
        for row in col..n {
            let mut sum = 0.0;
            for dim in 0..d {
                let diff = x[(row, dim)] - x[(col, dim)];
                sum += diff * diff;
            }
            dist_chunk[(row, local)] = sum;
        }
    }
}

fn fill_cross_chunk(
    x_train: MatRef<'_, f64>,
    x_test: MatRef<'_, f64>,
    mut dist_chunk: MatMut<'_, f64>,
    chunk_idx: usize,
    n_chunks: usize,
) {
    let n = x_train.nrows();
    let d = x_train.ncols();
    let m = x_test.nrows();
    let (start, len) = col_chunk(m, chunk_idx, n_chunks);
    debug_assert_eq!(dist_chunk.ncols(), len);
    if try_fill_cross_chunk(x_train, x_test, dist_chunk.rb_mut(), chunk_idx, n_chunks) {
        return;
    }
    for local in 0..len {
        let col = start + local;
        for row in 0..n {
            let mut sum = 0.0;
            for dim in 0..d {
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
        ArdSqDiffBuf, col_chunk, fill_squared_euclidean, fill_squared_euclidean_cross, worker_count,
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
