//! Unit-lower LDLT: solves and row / column delete.

use dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::ldlt;
use faer::linalg::triangular_solve::{
    solve_unit_lower_triangular_in_place, solve_unit_upper_triangular_in_place,
};

use crate::kernel::KernelScalar;
use faer::{Mat, MatMut, MatRef, Par};
use wide::f64x4;

/// Below this order the append solve runs one contiguous dot product per
/// row; from here faer's blocked solve on the transposed view is faster
/// (measured on the bordered append for `n` = 256 … 4096, #267). At 512 the
/// lower triangle is 1 MiB of `f64`.
const ROW_DOT_MAX: usize = 512;

/// Solves `L w = v` in place for unit-lower `L`, where row `i` of `L` is
/// the head `lt[0..i, i]` of column `i` of the stored `Lᵀ` (`f64`).
pub(crate) fn solve_unit_lower_rows_f64(lt: &Mat<f64>, w: &mut [f64]) {
    let n = w.len();
    if n >= ROW_DOT_MAX {
        let ld = lt.as_ref().submatrix(0, 0, n, n).transpose();
        solve_unit_lower_triangular_in_place(
            ld,
            MatMut::from_column_major_slice_mut(w, n, 1),
            Par::Seq,
        );
        return;
    }
    for i in 1..n {
        let (head, rest) = w.split_at_mut(i);
        rest[0] -= dot_f64(&lt.col_as_slice(i)[..i], head);
    }
}

/// `aᵀ b` with four `f64x4` accumulators.
fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
    const LANES: usize = 4;
    const STEP: usize = 4 * LANES;
    let mut acc = [f64x4::ZERO; 4];
    let body = a.len() - a.len() % STEP;
    for (ca, cb) in a[..body]
        .chunks_exact(STEP)
        .zip(b[..body].chunks_exact(STEP))
    {
        for (k, slot) in acc.iter_mut().enumerate() {
            let at = k * LANES;
            let va = f64x4::from([ca[at], ca[at + 1], ca[at + 2], ca[at + 3]]);
            let vb = f64x4::from([cb[at], cb[at + 1], cb[at + 2], cb[at + 3]]);
            *slot = va.mul_add(vb, *slot);
        }
    }
    let total = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    let mut sum = total.reduce_add();
    for (x, y) in a[body..].iter().zip(&b[body..]) {
        sum += x * y;
    }
    sum
}

/// [`solve_unit_lower_rows_f64`] for `f32` storage. The solution stays in
/// `f64` until every row is solved, then rounds once.
pub(crate) fn solve_unit_lower_rows_f32(lt: &Mat<f32>, w: &mut [f32]) {
    let mut solved: Vec<f64> = w.iter().map(|&v| f64::from(v)).collect();
    for i in 1..solved.len() {
        let (head, rest) = solved.split_at_mut(i);
        let mut sum = rest[0];
        for (l, x) in lt.col_as_slice(i)[..i].iter().zip(head.iter()) {
            sum -= f64::from(*l) * *x;
        }
        rest[0] = sum;
    }
    for (out, v) in w.iter_mut().zip(&solved) {
        *out = *v as f32;
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

/// Solves `L D Lᵀ x = b` for the leading `n` (overwrites the first column of `rhs`).
pub(crate) fn solve_ldlt_in_place<T: KernelScalar>(
    ld: MatRef<'_, T>,
    rhs: MatMut<'_, T>,
    n: usize,
) {
    T::solve_ldlt_in_place(ld, rhs, n);
}

/// [`solve_ldlt_in_place`] for `f32` storage: each column is solved in
/// `f64` from the `f32` entries and rounded once at the end.
///
/// The stored `Lᵀ` makes `ld` a row-major view; an `f32` solve over it rounds
/// in row (dot-product) order, whose error cancels worse in `k*ᵀ α` than the
/// column order did. Accumulating in `f64` removes that dependence.
pub(crate) fn solve_ldlt_f64_accum(ld: MatRef<'_, f32>, mut rhs: MatMut<'_, f32>, n: usize) {
    let mut x = vec![0.0f64; n];
    for col in 0..rhs.ncols() {
        for (i, v) in x.iter_mut().enumerate() {
            *v = f64::from(rhs[(i, col)]);
        }
        for i in 1..n {
            let mut sum = x[i];
            for j in 0..i {
                sum -= f64::from(ld[(i, j)]) * x[j];
            }
            x[i] = sum;
        }
        for (i, v) in x.iter_mut().enumerate() {
            *v /= f64::from(ld[(i, i)]);
        }
        for i in (0..n).rev() {
            let mut sum = x[i];
            for j in (i + 1)..n {
                sum -= f64::from(ld[(j, i)]) * x[j];
            }
            x[i] = sum;
        }
        for (i, v) in x.iter().enumerate() {
            rhs[(i, col)] = *v as f32;
        }
    }
}

/// [`solve_ldlt_in_place`] with faer (`f64`).
pub(crate) fn solve_ldlt_faer<T: KernelScalar>(
    ld: MatRef<'_, T>,
    mut rhs: MatMut<'_, T>,
    n: usize,
) {
    if n == 0 {
        return;
    }
    let ld_n = ld.submatrix(0, 0, n, n);
    solve_unit_lower_triangular_in_place(ld_n, rhs.as_mut(), Par::Seq);
    for i in 0..n {
        rhs[(i, 0)] /= ld[(i, i)];
    }
    solve_unit_upper_triangular_in_place(ld_n.transpose(), rhs, Par::Seq);
}

/// Overwrites each column of `rhs` (`n×m`) with `L⁻¹` of that column.
pub(crate) fn apply_ldlt_inv_l<T: KernelScalar>(ld: MatRef<'_, T>, rhs: MatMut<'_, T>, n: usize) {
    if n == 0 {
        return;
    }
    let ld_n = ld.submatrix(0, 0, n, n);
    solve_unit_lower_triangular_in_place(ld_n, rhs, Par::Seq);
}
