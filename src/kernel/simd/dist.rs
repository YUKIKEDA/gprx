//! The row loops of squared distances and of the packed `(Δx_d)²` cache.

use super::{add_squared_diff, col_slice, col_slice_mut, unit_row_stride};
use faer::reborrow::ReborrowMut;
use faer::{MatMut, MatRef};

/// Writes `‖x_row − x_col‖²` into `dest`, the rows `col..n` of column `col`.
/// Returns `false` when `x` is not column-major.
pub(crate) fn try_fill_lower_col(x: MatRef<'_, f64>, col: usize, dest: &mut [f64]) -> bool {
    if !unit_row_stride(x) {
        return false;
    }
    dest.fill(0.0);
    for dim in 0..x.ncols() {
        let Some(xdim) = col_slice(x, dim) else {
            return false;
        };
        add_squared_diff(&xdim[col..], xdim[col], dest);
    }
    true
}

/// Writes the train–test squared distances of the test points
/// `start..start + dist_chunk.ncols()` into `dist_chunk`. Returns `false`
/// when a view is not column-major.
pub(crate) fn try_fill_cross_chunk(
    x_train: MatRef<'_, f64>,
    x_test: MatRef<'_, f64>,
    mut dist_chunk: MatMut<'_, f64>,
    start: usize,
) -> bool {
    if !unit_row_stride(x_train) || !unit_row_stride(x_test) {
        return false;
    }
    for local in 0..dist_chunk.ncols() {
        let Some(dest) = col_slice_mut(dist_chunk.rb_mut(), local) else {
            return false;
        };
        dest.fill(0.0);
        for dim in 0..x_train.ncols() {
            let (Some(x_tr), Some(x_te)) = (col_slice(x_train, dim), col_slice(x_test, dim)) else {
                return false;
            };
            add_squared_diff(x_tr, x_te[start + local], dest);
        }
    }
    true
}

/// Fills the packed column `col` of dimension `dim` of a raw `(Δx_d)²`
/// cache: rows `col..n`. Returns `false` when `x` is not column-major.
pub(crate) fn try_fill_ard_column(
    x: MatRef<'_, f64>,
    dim: usize,
    col: usize,
    dest: &mut [f64],
) -> bool {
    let Some(xdim) = col_slice(x, dim) else {
        return false;
    };
    dest.fill(0.0);
    add_squared_diff(&xdim[col..], xdim[col], dest);
    true
}
