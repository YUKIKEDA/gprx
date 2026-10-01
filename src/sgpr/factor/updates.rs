//! Rank-1 training-point updates and inducing-point insert / delete.

use super::lit;
use super::vfe::{VfeState, refresh_w};
use crate::data::{pack_points, validate_inducing};
use crate::error::{CholeskyStage, GprError};
use crate::kernel::{KernelScalar, KernelSpec};
use crate::linalg::{
    append_chol_border, cholesky_lower_owned, delete_chol_row, frobenius2, gram_aat_plus_noise,
    mul_lower_left, solve_lower,
};
use crate::sparse::KernelScratch;
use faer::{Mat, MatMut, MatRef};

/// Appends `col` (`m × 1`) as a new last column of `a` in place.
///
/// The column capacity doubles when it runs out, so a run of appends costs
/// `O(m)` each, amortized, instead of copying all of `a` every time.
pub(crate) fn push_column<T: KernelScalar>(a: &mut Mat<T>, col: MatRef<'_, T>) {
    let m = a.nrows();
    let n = a.ncols();
    a.reserve(m, (n + 1).next_power_of_two());
    a.resize_with(m, n + 1, |row, _| col[(row, 0)]);
}

/// Removes column `idx` of `a` in place, shifting the later columns left.
pub(crate) fn remove_column_in_place<T: KernelScalar>(a: &mut Mat<T>, idx: usize) {
    let m = a.nrows();
    let n = a.ncols();
    for col in idx..n - 1 {
        for row in 0..m {
            a[(row, col)] = a[(row, col + 1)];
        }
    }
    a.truncate(m, n - 1);
}

/// `A y` in `f64` (`m`), whatever the storage scalar of `A` (`m × n`).
pub(crate) fn a_times_y<T: KernelScalar>(a: MatRef<'_, T>, y: &[f64]) -> Vec<f64> {
    let mut ay = vec![0.0; a.nrows()];
    for (col, &y_col) in y.iter().enumerate().take(a.ncols()) {
        for (row, slot) in ay.iter_mut().enumerate() {
            *slot += a[(row, col)].to_f64() * y_col;
        }
    }
    ay
}

/// Appends one point to column-major `x` (`n × d`) in place. Growth is
/// amortized by the `Vec`.
pub(crate) fn append_point(x: &mut Vec<f64>, n: usize, d: usize, x_new: &[f64]) {
    x.resize((n + 1) * d, 0.0);
    // Last column first, so no column is overwritten before it moves.
    for dim in (0..d).rev() {
        x.copy_within(dim * n..(dim + 1) * n, dim * (n + 1));
        x[dim * (n + 1) + n] = x_new[dim];
    }
}

pub(crate) fn remove_point(x: &[f64], n: usize, d: usize, idx: usize) -> Vec<f64> {
    let mut out = vec![0.0; (n - 1) * d];
    for dim in 0..d {
        let mut dest = 0;
        for i in 0..n {
            if i == idx {
                continue;
            }
            out[dest + (n - 1) * dim] = x[i + n * dim];
            dest += 1;
        }
    }
    out
}

pub(crate) fn point_at(x: &[f64], n: usize, d: usize, idx: usize) -> Vec<f64> {
    let mut out = vec![0.0; d];
    for dim in 0..d {
        out[dim] = x[idx + n * dim];
    }
    out
}

pub(crate) fn kernel_column<M: crate::math::KernelMath, T>(
    kernel: &KernelSpec,
    z: &[f64],
    m: usize,
    x_pt: &[f64],
    d: usize,
    ks: &mut KernelScratch<T>,
) -> Result<Mat<T>, GprError>
where
    T: KernelScalar,
{
    let compiled = kernel.compile_as::<T>();
    let z64 = pack_points(z, m, d);
    let x64 = pack_points(x_pt, 1, d);
    let mut z_cast = T::empty_cols();
    let mut x_cast = T::empty_cols();
    let z_mat = T::storage_cols(z64.as_ref(), &mut z_cast);
    let x_mat = T::storage_cols(x64.as_ref(), &mut x_cast);
    ks.cross::<M>(&compiled, z_mat, x_mat)
}

pub(crate) fn kernel_diag_at<T>(kernel: &KernelSpec, x_pt: &[f64], d: usize) -> Result<T, GprError>
where
    T: KernelScalar,
{
    let compiled = kernel.compile_as::<T>();
    let x64 = pack_points(x_pt, 1, d);
    let mut x_cast = T::empty_cols();
    let x_mat = T::storage_cols(x64.as_ref(), &mut x_cast);
    let mut diag = vec![lit::<T>(0.0); 1];
    compiled.fill_diag_points(x_mat, &mut diag)?;
    Ok(diag[0])
}

pub(crate) fn solve_lmm<T: KernelScalar>(k_mm_l: MatRef<'_, T>, col: MatMut<'_, T>) {
    solve_lower(k_mm_l, col);
}

/// Appends one inducing point at the end by a bordered LLT of `K_mm` and `B`.
///
/// `A` gains a row. `k_diag_sum` is unchanged. `w` is solved from the new `B`.
#[allow(clippy::too_many_arguments)] // kernel, data, and new `Z` stay explicit
pub(crate) fn inducing_insert<M: crate::math::KernelMath, T>(
    state: &mut VfeState<T>,
    kernel: &KernelSpec,
    noise: f64,
    x: &[f64],
    n: usize,
    d: usize,
    y: &[f64],
    z: &[f64],
    m: usize,
    z_new: &[f64],
    ks: &mut KernelScratch<T>,
) -> Result<(), GprError>
where
    T: KernelScalar,
{
    validate_inducing(z, m, d)?;
    validate_inducing(z_new, 1, d)?;
    if n == 0 {
        return Err(GprError::EmptyInput);
    }
    let compiled = kernel.compile_as::<T>();
    let z64 = pack_points(z, m, d);
    let z_new64 = pack_points(z_new, 1, d);
    let x64 = pack_points(x, n, d);
    let mut z_cast = T::empty_cols();
    let mut zn_cast = T::empty_cols();
    let mut x_cast = T::empty_cols();
    let mut y_cast = T::empty_rows();
    let z_mat = T::storage_cols(z64.as_ref(), &mut z_cast);
    let z_new_mat = T::storage_cols(z_new64.as_ref(), &mut zn_cast);
    let x_mat = T::storage_cols(x64.as_ref(), &mut x_cast);
    let y_s = T::storage_rows(y, &mut y_cast);
    let mut k_zz = ks.cross::<M>(&compiled, z_mat, z_new_mat)?;
    let k_nn = kernel_diag_at::<T>(kernel, z_new, d)?;
    let k_zx = ks.cross::<M>(&compiled, z_new_mat, x_mat)?;
    solve_lmm(state.k_mm_l.as_ref(), k_zz.as_mut());
    let mut ell2 = k_nn;
    for i in 0..m {
        let li = k_zz[(i, 0)];
        ell2 -= li * li;
    }
    if ell2.to_f64() <= 0.0 || !ell2.to_f64().is_finite() {
        return Err(GprError::CholeskyFailed {
            jitter: 0.0,
            matrix_size: m + 1,
            stage: CholeskyStage::OnlineInsert,
        });
    }
    let ell = ell2.sqrt();
    let mut a_new = vec![lit::<T>(0.0); n];
    let mut a_new_norm2 = lit::<T>(0.0);
    for j in 0..n {
        let mut dot = lit::<T>(0.0);
        for i in 0..m {
            dot += k_zz[(i, 0)] * state.a[(i, j)];
        }
        let value = (k_zx[(0, j)] - dot) / ell;
        a_new[j] = value;
        a_new_norm2 += value * value;
    }
    let mut v = vec![lit::<T>(0.0); m];
    for (i, slot) in v.iter_mut().enumerate() {
        let mut sum = lit::<T>(0.0);
        for (j, a_val) in a_new.iter().enumerate() {
            sum += state.a[(i, j)] * a_val;
        }
        *slot = sum;
    }
    let mut b_border = Mat::zeros(m, 1);
    for i in 0..m {
        b_border[(i, 0)] = v[i];
    }
    solve_lmm(state.b_l.as_ref(), b_border.as_mut());
    let mut beta2 = lit::<T>(noise) + a_new_norm2;
    for i in 0..m {
        let bi = b_border[(i, 0)];
        beta2 -= bi * bi;
    }
    if beta2.to_f64() <= 0.0 || !beta2.to_f64().is_finite() {
        return Err(GprError::CholeskyFailed {
            jitter: 0.0,
            matrix_size: m + 1,
            stage: CholeskyStage::OnlineInsert,
        });
    }
    let mut l_col = vec![lit::<T>(0.0); m];
    for i in 0..m {
        l_col[i] = k_zz[(i, 0)];
    }
    let mut b_col = vec![lit::<T>(0.0); m];
    for i in 0..m {
        b_col[i] = b_border[(i, 0)];
    }
    state.k_mm_l = append_chol_border(&state.k_mm_l, &l_col, ell);
    state.b_l = append_chol_border(&state.b_l, &b_col, beta2.sqrt());
    state.a = append_row(&state.a, &a_new);
    state.a_frobenius2 += a_new_norm2;
    state.w = refresh_w(state.a.as_ref(), state.b_l.as_ref(), y_s);
    Ok(())
}

/// Drops inducing row `idx` by a trailing cholupdate of `L_mm`.
///
/// Reuses `K(Z, X) = L A`, drops that row, and solves the reduced `A`.
/// `B` is formed again from the new `A`. `k_diag_sum` is unchanged.
pub(crate) fn inducing_delete<T: KernelScalar>(
    state: &mut VfeState<T>,
    noise: f64,
    y: &[f64],
    idx: usize,
) -> Result<(), GprError> {
    let m = state.a.nrows();
    if m <= 1 {
        return Err(GprError::EmptyInput);
    }
    if idx >= m {
        return Err(GprError::IndexOutOfRange {
            reason: "inducing index is out of range".to_owned(),
        });
    }
    let k_zx = mul_lower_left(state.k_mm_l.as_ref(), state.a.as_ref());
    let k_zx = remove_row(&k_zx, idx);
    state.k_mm_l = delete_chol_row(&state.k_mm_l, idx);
    let mut a = k_zx;
    solve_lower(state.k_mm_l.as_ref(), a.as_mut());
    let mut b = gram_aat_plus_noise(a.as_ref(), noise);
    cholesky_lower_owned(&mut b, std::iter::empty(), CholeskyStage::OnlineDelete)?;
    state.a = a;
    state.b_l = b;
    state.a_frobenius2 = frobenius2(state.a.as_ref());
    let mut y_cast = T::empty_rows();
    let y_s = T::storage_rows(y, &mut y_cast);
    state.w = refresh_w(state.a.as_ref(), state.b_l.as_ref(), y_s);
    Ok(())
}

pub(super) fn append_row<T: KernelScalar>(a: &Mat<T>, row: &[T]) -> Mat<T> {
    let m = a.nrows();
    let n = a.ncols();
    let mut out = Mat::zeros(m + 1, n);
    for j in 0..n {
        for i in 0..m {
            out[(i, j)] = a[(i, j)];
        }
        out[(m, j)] = row[j];
    }
    out
}

pub(super) fn remove_row<T: KernelScalar>(a: &Mat<T>, idx: usize) -> Mat<T> {
    let m = a.nrows();
    let n = a.ncols();
    let mut out = Mat::zeros(m - 1, n);
    let mut dest = 0;
    for i in 0..m {
        if i == idx {
            continue;
        }
        for j in 0..n {
            out[(dest, j)] = a[(i, j)];
        }
        dest += 1;
    }
    out
}
