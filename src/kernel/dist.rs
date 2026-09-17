//! Pairwise squared-Euclidean distances, filled by Rayon column partitions.

use super::simd::{try_fill_cross_chunk, try_fill_lower_chunk};
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

#[cfg(test)]
mod tests {
    use super::{col_chunk, fill_squared_euclidean, fill_squared_euclidean_cross, worker_count};
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
}
