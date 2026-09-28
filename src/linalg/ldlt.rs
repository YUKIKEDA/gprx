//! Unit-lower LDLT: forward solve and row / column delete.

use dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::ldlt;
use faer::linalg::triangular_solve::solve_unit_lower_triangular_in_place;
use faer::{Mat, MatMut, MatRef, Par};

/// Solves `L w = v` in place for unit-lower `L` (`f64`, faer).
pub(crate) fn solve_unit_lower_faer(ld: MatRef<'_, f64>, v: MatMut<'_, f64>) {
    solve_unit_lower_triangular_in_place(ld, v, Par::Seq);
}

/// Solves `L w = v` in place for unit-lower `L`, accumulating in `f64`.
#[allow(clippy::needless_range_loop)]
pub(crate) fn solve_unit_lower_f64_accum(ld: MatRef<'_, f32>, mut v: MatMut<'_, f32>) {
    let n = ld.nrows();
    let mut solved = vec![0.0f64; n];
    for i in 0..n {
        solved[i] = f64::from(v[(i, 0)]);
    }
    for i in 0..n {
        let mut sum = solved[i];
        for j in 0..i {
            sum -= f64::from(ld[(i, j)]) * solved[j];
        }
        solved[i] = sum;
    }
    for i in 0..n {
        v[(i, 0)] = solved[i] as f32;
    }
}

/// Deletes row and column `index` from the `n×n` LDLT in `ld` (`f64`, faer).
pub(crate) fn ldlt_delete_faer(
    ld: MatMut<'_, f64>,
    index: usize,
    _n: usize,
    scratch: &mut MemBuffer,
) {
    let mut indices = [index];
    let stack = MemStack::new(scratch);
    ldlt::update::delete_rows_and_cols_clobber(ld, &mut indices, Par::Seq, stack);
}

/// Deletes row and column `index` from the `n×n` LDLT in `ld` through an
/// `f64` copy, then rounds back to `f32`.
pub(crate) fn ldlt_delete_via_f64(mut ld: MatMut<'_, f32>, index: usize, n: usize) {
    let mut ld64 = Mat::<f64>::zeros(n, n);
    for col in 0..n {
        for row in col..n {
            ld64[(row, col)] = f64::from(ld[(row, col)]);
        }
    }
    let scratch_req = ldlt::update::delete_rows_and_cols_clobber_scratch::<f64>(n.max(1), 1);
    let mut scratch = MemBuffer::new(scratch_req);
    let stack = MemStack::new(&mut scratch);
    let mut indices = [index];
    ldlt::update::delete_rows_and_cols_clobber(ld64.as_mut(), &mut indices, Par::Seq, stack);
    for col in 0..n {
        for row in col..n {
            ld[(row, col)] = ld64[(row, col)] as f32;
        }
    }
}
