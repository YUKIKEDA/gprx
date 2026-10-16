//! Rank-1 training-point updates and inducing-point insert / delete.

use super::lit;
use super::vfe::{VfeState, refresh_w};
use crate::data::{pack_points, validate_inducing};
use crate::error::{CholeskyStage, GprError};
use crate::kernel::{CrossViews, KernelScalar, KernelSpec, NoSupply};
use crate::linalg::{
    append_chol_border, chol_rank1_downdate, chol_rank1_update, cholesky_lower_owned,
    delete_chol_row, frobenius2, gram_aat_plus_noise, mat_vec, mul_lower_left, solve_lower,
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
    ks.cross::<M, NoSupply>(&compiled, CrossViews::points(z_mat, x_mat))
}

pub(crate) fn kernel_diag_at<T, U: crate::kernel::Supply>(
    kernel: &KernelSpec<U>,
    x_pt: &[f64],
    d: usize,
) -> Result<T, GprError>
where
    T: KernelScalar,
{
    let compiled = kernel.compile_as::<T>();
    let x64 = pack_points(x_pt, 1, d);
    let mut x_cast = T::empty_cols();
    let x_mat = T::storage_cols(x64.as_ref(), &mut x_cast);
    let mut diag = vec![lit::<T>(0.0); 1];
    match U::coordinates(&compiled) {
        Some(coords) => coords.fill_diag_points(x_mat, &mut diag)?,
        None => compiled.eval_diag(x_mat, &mut diag)?,
    }
    Ok(diag[0])
}

pub(crate) fn solve_lmm<T: KernelScalar>(k_mm_l: MatRef<'_, T>, col: MatMut<'_, T>) {
    solve_lower(k_mm_l, col);
}

/// The kernel values a new inducing point `z_new` brings to the VFE system
/// of `m` inducing points `Z` and `n` training points `X`.
pub(crate) struct InducingBorder<T> {
    /// `k(Z, z_new)` (`m × 1`).
    pub(crate) k_z: Mat<T>,
    /// `k(z_new, z_new)`.
    pub(crate) k_self: T,
    /// `k(z_new, X)` (`n` values, in the order of `X`).
    pub(crate) k_x: Vec<T>,
}

/// [`InducingBorder`] of the coordinates `z_new` for a coordinate kernel,
/// with `X` (`n × d`) and `Z` (`m × d`) column-major.
///
/// # Errors
///
/// Returns the errors of [`validate_inducing`] for `Z` or `z_new`,
/// [`GprError::EmptyInput`] when `n` is zero, and the errors of the kernel
/// evaluation.
#[allow(clippy::too_many_arguments)] // kernel, data, and new `Z` stay explicit
pub(crate) fn coordinate_border<M: crate::math::KernelMath, T>(
    kernel: &KernelSpec,
    x: &[f64],
    n: usize,
    d: usize,
    z: &[f64],
    m: usize,
    z_new: &[f64],
    ks: &mut KernelScratch<T>,
) -> Result<InducingBorder<T>, GprError>
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
    let z_mat = T::storage_cols(z64.as_ref(), &mut z_cast);
    let z_new_mat = T::storage_cols(z_new64.as_ref(), &mut zn_cast);
    let x_mat = T::storage_cols(x64.as_ref(), &mut x_cast);
    let k_z = ks.cross::<M, NoSupply>(&compiled, CrossViews::points(z_mat, z_new_mat))?;
    let k_self = kernel_diag_at::<T, NoSupply>(kernel, z_new, d)?;
    let k_zx = ks.cross::<M, NoSupply>(&compiled, CrossViews::points(z_new_mat, x_mat))?;
    let k_x = (0..n).map(|j| k_zx[(0, j)]).collect();
    Ok(InducingBorder { k_z, k_self, k_x })
}

/// Smallest bordered pivot, relative to the diagonal it is cancelled from,
/// that [`inducing_append`] takes: `ℓ² > τ (k(z_new, z_new) + j)` for
/// `K_mm + j I` and `β² > τ (σ² + ‖a_new‖²)` for `B`. A smaller pivot
/// returns [`GprError::CholeskyFailed`], and the caller assembles the
/// system again as the factor would.
//
// `ℓ² = (k(z_new, z_new) + j) − ‖l‖²` cancels two values of size up to
// `k(z_new, z_new) + j`, so it carries an absolute rounding error of about
// `ε (k(z_new, z_new) + j)` (`ε = 2.2e-16`), plus the error of `l` itself;
// `β²` cancels `‖b‖²` from `σ² + ‖a_new‖²` the same way. At a pivot of
// `τ` times the diagonal that is a relative error of `ε / τ` in `ℓ²` and
// half of it in `ℓ`, which divides the whole new row of `A`. `τ = 1e-5`
// keeps that at `2.2e-11` (`1.1e-11` in `ℓ`), an order below the `1e-10`
// the inducing sequences compare the update with the factor at, with room
// for the `O(m)` growth of the error of `‖l‖²`. A pivot that small is an
// inducing point within about `0.003` lengthscales of the inducing set
// (`ℓ² ≈ r² / ℓ_k²` for a stationary kernel), where the factor itself is
// that ill-conditioned; assembling again there costs one `O(n·m²)` factor.
const PIVOT_RTOL: f64 = 1e-5;

/// [`PIVOT_RTOL`] for an `f32` storage scalar ([`crate::SinglePrecision`],
/// [`crate::MixedPrecision`]), whose factor of `K_mm` an `X` insert or
/// delete starts from ([`factor_pivots_hold`]); their inducing changes
/// always assemble again, so no bordered pivot is checked against it.
//
// The assembly of an `f32` model computes in `f64` and rounds, so the
// stored factor of `K_mm` keeps the relative accuracy of `f32` in every
// entry, small pivots included, and the ratio [`factor_pivots_hold`] reads
// from it is that of the `f64` factor to about `1e-7`. An `X` step then
// solves `L⁻¹ k(Z, x)` and updates `B` in `f32`: a pivot `r = L_ii² /
// (K_ii + j)` divides row `i` of the new column, so the rounding error of
// the step grows like `ε / r` with the `ε = 1.2e-7` of `f32`, where `f64`
// has `2.2e-16`. Measured on 1-D RBF data (two near duplicate pairs among
// five inducing points, `σ² = 0.1`), five `X` inserts and deletes were
// within `1.3e-6` to `2.7e-6` of a fresh `f32` factor for `r ≥ 7e-5` (the
// difference an `f32` update has from the factor anyway), `1.6e-5` at
// `r = 2.3e-5`, `3.8e-5` at `7e-6`, `1.5e-3` at `7e-9`, and `0.25` at
// `7e-15`; in `f64` the same steps were within `1e-14` down to
// `r = 2.3e-5`. `1e-3` keeps the part the pivot adds (`≈ 4e-10 / r`)
// below that floor, so the `1e-5` a model factored again on load is held
// to keeps a margin of about `4`; the `f64` value `1e-5` would let
// `1.6e-5` through. A pivot that small is an inducing point within about
// `0.03` lengthscales of the inducing set.
const F32_PIVOT_RTOL: f64 = 1e-3;

/// The relative pivot threshold of storage scalar `T`: [`PIVOT_RTOL`] for
/// `f64`, [`F32_PIVOT_RTOL`] for `f32` (the scalar the assembly computes
/// in `f64` and rounds into).
fn pivot_rtol<T: KernelScalar>() -> f64 {
    if T::ROUNDS_FROM_F64 {
        F32_PIVOT_RTOL
    } else {
        PIVOT_RTOL
    }
}

/// Appends one inducing point at the end of `state` from its kernel values
/// `border` (ADR 0005): a bordered LLT of `K_mm + j I`, with the offset
/// `j` the factor of `K_mm` already took, and of `B`.
///
/// With `l = L⁻¹ k(Z, z_new)` and `ℓ² = k(z_new, z_new) + j − ‖l‖²`, `A`
/// gains the row `(k(z_new, X) − lᵀ A) / ℓ` (`O(n·m)`) and `B` the border
/// that row makes (`O(m²)` past `A` times the row). `k_diag_sum` is
/// unchanged, `‖A‖_F²` adds the norm of the row, and `w` is solved again
/// from the new `B`.
///
/// `replace`, when it is `Some(col)`, names the training point that is
/// `z_new` itself and whose `k(Z, x_col)` changed with this insert (a
/// supplied distance repaired to the mean of its mirror pair): column `col`
/// of `A` becomes `l` first, and `B` follows by a rank-1 update and
/// downdate.
///
/// The caller takes this path only from a factor whose pivots hold
/// ([`factor_pivots_hold`]); the bordered pivot is checked here.
///
/// # Errors
///
/// Returns [`GprError::CholeskyFailed`] when the bordered pivot of
/// `K_mm + j I` or of `B` is at most [`PIVOT_RTOL`] times the diagonal it
/// is cancelled from (or not finite), or the downdate of `B` for `replace`
/// fails. `state` is then partly written: callers pass a copy.
pub(crate) fn inducing_append<T: KernelScalar>(
    state: &mut VfeState<T>,
    border: InducingBorder<T>,
    replace: Option<usize>,
    noise: f64,
    y: &[f64],
) -> Result<(), GprError> {
    let InducingBorder {
        k_z: mut l,
        k_self,
        k_x,
    } = border;
    let (m, n) = (state.a.nrows(), state.a.ncols());
    let failed = |jitter| GprError::CholeskyFailed {
        jitter,
        matrix_size: m + 1,
        stage: CholeskyStage::OnlineInsert,
    };
    solve_lmm(state.k_mm_l.as_ref(), l.as_mut());
    let k_diag = k_self + lit::<T>(state.k_mm_jitter);
    let mut ell2 = k_diag;
    for i in 0..m {
        ell2 -= l[(i, 0)] * l[(i, 0)];
    }
    if !pivot_holds(ell2, k_diag) {
        return Err(failed(state.k_mm_jitter));
    }
    let l_col: Vec<T> = (0..m).map(|i| l[(i, 0)]).collect();
    if let Some(col) = replace {
        let mut old: Vec<T> = (0..m).map(|i| state.a[(i, col)]).collect();
        let mut new = l_col.clone();
        for (i, (&o, &v)) in old.iter().zip(&new).enumerate() {
            state.a_frobenius2 += v * v - o * o;
            state.a[(i, col)] = v;
        }
        chol_rank1_update(&mut state.b_l, &mut new);
        if !chol_rank1_downdate(&mut state.b_l, &mut old) {
            return Err(failed(0.0));
        }
    }
    let ell = ell2.sqrt();
    let mut a_new = vec![lit::<T>(0.0); n];
    let mut a_new_norm2 = lit::<T>(0.0);
    for j in 0..n {
        let mut dot = lit::<T>(0.0);
        for (i, &li) in l_col.iter().enumerate() {
            dot += li * state.a[(i, j)];
        }
        let value = (k_x[j] - dot) / ell;
        a_new[j] = value;
        a_new_norm2 += value * value;
    }
    let v = mat_vec(state.a.as_ref(), &a_new);
    let mut b_border = Mat::zeros(m, 1);
    for i in 0..m {
        b_border[(i, 0)] = v[i];
    }
    solve_lmm(state.b_l.as_ref(), b_border.as_mut());
    let b_diag = lit::<T>(noise) + a_new_norm2;
    let mut beta2 = b_diag;
    for i in 0..m {
        let bi = b_border[(i, 0)];
        beta2 -= bi * bi;
    }
    if !pivot_holds(beta2, b_diag) {
        return Err(failed(0.0));
    }
    let b_col: Vec<T> = (0..m).map(|i| b_border[(i, 0)]).collect();
    state.k_mm_l = append_chol_border(&state.k_mm_l, &l_col, ell);
    state.b_l = append_chol_border(&state.b_l, &b_col, beta2.sqrt());
    state.a = append_row(&state.a, &a_new);
    state.a_frobenius2 += a_new_norm2;
    let mut y_cast = T::empty_rows();
    let y_s = T::storage_rows(y, &mut y_cast);
    state.w = refresh_w(state.a.as_ref(), state.b_l.as_ref(), y_s);
    Ok(())
}

/// Whether the bordered pivot `pivot`, cancelled from `diag`, is finite and
/// above the threshold of `T` ([`pivot_rtol`]) times `diag`.
fn pivot_holds<T: KernelScalar>(pivot: T, diag: T) -> bool {
    let (pivot, diag) = (pivot.to_f64(), diag.to_f64());
    pivot.is_finite() && pivot > pivot_rtol::<T>() * diag
}

/// Whether every pivot of `l`, the lower factor of `K_mm + j I`, holds
/// the threshold an inserted pivot is held to: `L_ii² > τ (K_ii + j)`
/// with `τ` the threshold of `T` ([`pivot_rtol`]: [`PIVOT_RTOL`] for
/// `f64`, [`F32_PIVOT_RTOL`] for `f32`), where `K_ii + j = Σ_k L_ik²` is
/// the squared norm of row `i` (`O(m²)`, summed in `f64`).
///
/// [`inducing_append`] and [`inducing_delete`] start from `l`, so a pivot
/// below the threshold anywhere in it (one an assembly took: the factor
/// accepts any positive pivot, down to the rounding residue of an exact
/// duplicate) puts its relative rounding error into every row the update
/// computes after it, as the inserted pivot would. The rank-1 update of
/// `B` an `X` insert or delete makes (ADR 0004) solves `L⁻¹ k(Z, x)` the
/// same way. The caller assembles the system again instead, as the factor
/// does. In exact arithmetic an append leaves the pivots of `l` as they
/// are and a delete only raises those after the row it drops, so a factor
/// whose pivots hold keeps them through updates; only an assembly brings
/// in a smaller one, and an `X` step does not change `l`.
pub(crate) fn factor_pivots_hold<T: KernelScalar>(l: MatRef<'_, T>) -> bool {
    (0..l.nrows()).all(|i| {
        let mut row2 = 0.0;
        for k in 0..=i {
            let value = l[(i, k)].to_f64();
            row2 += value * value;
        }
        let pivot = l[(i, i)].to_f64();
        pivot.is_finite() && pivot * pivot > pivot_rtol::<T>() * row2
    })
}

/// Drops inducing row `idx` by a trailing cholupdate of `L_mm` (ADR 0005).
///
/// Reuses `K(Z, X) = L A`, drops that row, and solves the reduced `A`
/// (`O(n·m²)`, no kernel evaluation). `B` is formed again from the new
/// `A`. `k_diag_sum` and the offset `j` of `K_mm + j I` are unchanged.
/// The caller takes this path only from a factor whose pivots hold
/// ([`factor_pivots_hold`]): the remaining pivots then hold as well.
///
/// # Errors
///
/// Returns [`GprError::EmptyInput`] when `m ≤ 1`,
/// [`GprError::IndexOutOfRange`] when `idx ≥ m`, and
/// [`GprError::CholeskyFailed`] when the new `B` does not factor.
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
    for j in 0..n {
        for i in 0..idx {
            out[(i, j)] = a[(i, j)];
        }
        for i in (idx + 1)..m {
            out[(i - 1, j)] = a[(i, j)];
        }
    }
    out
}
