//! Cholesky (LLT): factor, solves, rank-1 updates, and jitter retries.

use dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt;
use faer::linalg::cholesky::llt::factor::{LltError, LltRegularization};
use faer::{Mat, MatMut, MatRef};

use crate::error::{CholeskyStage, GprError};
use crate::kernel::KernelScalar;

use super::dense::add_to_diag;
use super::par::{faer_par, faer_par_dims};

/// Tries `attempt(0)`, then `attempt(j)` for each retry offset `j`, and
/// returns the `j` that succeeded.
///
/// `attempt(j)` factors `A + j I`. Only [`GprError::CholeskyFailed`] moves to
/// the next offset; any other error returns at once. After the last offset,
/// the error reports the last `j` tried (`0.0` when there was no retry).
pub(crate) fn retry_with_jitter(
    retries: impl IntoIterator<Item = f64>,
    matrix_size: usize,
    stage: CholeskyStage,
    mut attempt: impl FnMut(f64) -> Result<(), GprError>,
) -> Result<f64, GprError> {
    match attempt(0.0) {
        Err(GprError::CholeskyFailed { .. }) => {}
        other => return other.map(|()| 0.0),
    }
    let mut last_j = 0.0;
    for j in retries {
        last_j = j;
        match attempt(j) {
            Err(GprError::CholeskyFailed { .. }) => {}
            other => return other.map(|()| j),
        }
    }
    Err(GprError::CholeskyFailed {
        jitter: last_j,
        matrix_size,
        stage,
    })
}

/// Factors `A` in place, retrying from a snapshot of `A` with each offset.
///
/// The first attempt uses `A` as given. Each retry restores the snapshot and
/// adds `j` to the diagonal.
pub(crate) fn cholesky_lower_with_retries<T: KernelScalar>(
    a: &mut Mat<T>,
    scratch: &mut MemBuffer,
    retries: impl IntoIterator<Item = f64>,
    stage: CholeskyStage,
) -> Result<(), GprError> {
    cholesky_lower_with_backup(a, &mut Mat::new(), scratch, retries, stage)
}

/// [`cholesky_lower_with_retries`] keeping the copy of `a` a retry starts
/// from in `backup`, which is reused when it already has the shape of `a`.
pub(crate) fn cholesky_lower_with_backup<T: KernelScalar>(
    a: &mut Mat<T>,
    backup: &mut Mat<T>,
    scratch: &mut MemBuffer,
    retries: impl IntoIterator<Item = f64>,
    stage: CholeskyStage,
) -> Result<(), GprError> {
    if backup.nrows() == a.nrows() && backup.ncols() == a.ncols() {
        backup.copy_from(&*a);
    } else {
        *backup = a.clone();
    }
    let n = a.nrows();
    retry_with_jitter(retries, n, stage, |j| {
        if j != 0.0 {
            a.copy_from(&*backup);
            add_to_diag(a.as_mut(), j);
        }
        cholesky_lower(a, scratch, 0.0, stage)
    })
    .map(|_| ())
}

/// [`cholesky_lower_with_retries`] with a scratch buffer sized for this call.
pub(crate) fn cholesky_lower_owned<T: KernelScalar>(
    a: &mut Mat<T>,
    retries: impl IntoIterator<Item = f64>,
    stage: CholeskyStage,
) -> Result<(), GprError> {
    let mut scratch = llt_scratch::<T>(a.nrows());
    cholesky_lower_with_retries(a, &mut scratch, retries, stage)
}

/// faer scratch for one `n × n` LLT factor.
pub(crate) fn llt_scratch<T: KernelScalar>(n: usize) -> MemBuffer {
    MemBuffer::new(llt::factor::cholesky_in_place_scratch::<T>(
        n,
        faer_par(n),
        Default::default(),
    ))
}

/// Factors `A` in place with faer, for any scalar. `jitter` is faer's
/// dynamic regularization delta.
pub(crate) fn cholesky_lower_faer<T: KernelScalar>(
    a: &mut Mat<T>,
    scratch: &mut MemBuffer,
    jitter: f64,
    stage: CholeskyStage,
) -> Result<(), GprError> {
    let n = a.nrows();
    let regularization = LltRegularization {
        dynamic_regularization_delta: T::from_f64(jitter),
        dynamic_regularization_epsilon: T::from_f64(0.0),
    };
    let stack = MemStack::new(scratch);
    match llt::factor::cholesky_in_place(
        a.as_mut(),
        regularization,
        faer_par(n),
        stack,
        Default::default(),
    ) {
        Ok(_) => Ok(()),
        Err(LltError::NonPositivePivot { .. }) => Err(GprError::CholeskyFailed {
            jitter,
            matrix_size: n,
            stage,
        }),
    }
}

/// Overwrites `rhs` with `(L Lᵀ)⁻¹ rhs` through faer, for any scalar, with a
/// scratch buffer sized for this call.
pub(crate) fn solve_llt_faer_owned<T: KernelScalar>(l: MatRef<'_, T>, rhs: MatMut<'_, T>) {
    let n = l.nrows();
    let n_rhs = rhs.ncols();
    let par = faer_par_dims(n, n_rhs);
    let req = llt::solve::solve_in_place_scratch::<T>(n, n_rhs, par);
    let mut buf = MemBuffer::new(req);
    llt::solve::solve_in_place(l, rhs, par, MemStack::new(&mut buf));
}

/// Forward substitution `L x = b` in this scalar. A zero pivot yields `0`.
pub(crate) fn forward_substitute<T: KernelScalar>(l: MatRef<'_, T>, b: &[T]) -> Vec<T> {
    let m = b.len();
    let mut x = vec![T::from_f64(0.0); m];
    for i in 0..m {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[(i, j)] * x[j];
        }
        let diag = l[(i, i)];
        x[i] = if diag.abs().to_f64() > 0.0 {
            sum / diag
        } else {
            T::from_f64(0.0)
        };
    }
    x
}

pub(crate) fn log_det_from_l<T: KernelScalar>(l: MatRef<'_, T>, n: usize) -> T {
    let mut log_diag = T::from_f64(0.0);
    for i in 0..n {
        log_diag += KernelScalar::ln(l[(i, i)]);
    }
    T::from_f64(2.0) * log_diag
}

/// Writes `diag(A⁻¹)` given the lower Cholesky factor `L` of `A = L Lᵀ`.
///
/// `A⁻¹ = L^{-T} L^{-1}`, so entry `i` is the squared Euclidean norm of
/// column `i` of `L⁻¹`.
pub(crate) fn inv_diag_from_chol_l<T: KernelScalar>(l: MatRef<'_, T>, q_diag: &mut [T]) {
    T::inv_diag_from_chol_l(l, q_diag);
}

/// [`inv_diag_from_chol_l`] for `f32`: each column of `L⁻¹` in `f64`.
#[allow(clippy::needless_range_loop)]
pub(crate) fn inv_diag_from_chol_l_f64_accum(l: MatRef<'_, f32>, q_diag: &mut [f32]) {
    let n = l.nrows();
    debug_assert_eq!(q_diag.len(), n);
    let mut col = vec![0.0f64; n];
    for i in 0..n {
        for (row, slot) in col.iter_mut().enumerate() {
            *slot = if row == i { 1.0 } else { 0.0 };
        }
        for row in 0..n {
            let mut sum = col[row];
            for k in 0..row {
                sum -= f64::from(l[(row, k)]) * col[k];
            }
            col[row] = sum / f64::from(l[(row, row)]);
        }
        let mut q = 0.0f64;
        for v in &col {
            q += v * v;
        }
        q_diag[i] = q as f32;
    }
}

/// [`inv_diag_from_chol_l`] for `f64` through faer's triangular solve.
pub(crate) fn inv_diag_from_chol_l_faer(l: MatRef<'_, f64>, q_diag: &mut [f64]) {
    let n = l.nrows();
    debug_assert_eq!(q_diag.len(), n);
    let mut inv_l = Mat::from_fn(n, n, |row, col| if row == col { 1.0 } else { 0.0 });
    faer::linalg::triangular_solve::solve_lower_triangular_in_place(l, inv_l.as_mut(), faer_par(n));
    for (i, qi) in q_diag.iter_mut().enumerate() {
        let mut q = 0.0;
        for k in 0..n {
            let v = inv_l[(k, i)];
            q += v * v;
        }
        *qi = q;
    }
}

/// Factors `A` in place as `L Lᵀ`. The strictly upper triangle is unspecified.
pub(crate) fn cholesky_lower<T: KernelScalar>(
    a: &mut Mat<T>,
    scratch: &mut MemBuffer,
    jitter: f64,
    stage: CholeskyStage,
) -> Result<(), GprError> {
    T::cholesky_lower(a, scratch, jitter, stage)
}

/// [`cholesky_lower`] for `f32`, accumulating each dot product in `f64`.
pub(crate) fn cholesky_lower_f64_accum(
    a: &mut Mat<f32>,
    jitter: f64,
    stage: CholeskyStage,
) -> Result<(), GprError> {
    let n = a.nrows();
    for j in 0..n {
        for i in j..n {
            let mut sum = f64::from(a[(i, j)]);
            for k in 0..j {
                sum -= f64::from(a[(i, k)]) * f64::from(a[(j, k)]);
            }
            if i == j {
                if sum.is_nan() || sum <= 0.0 {
                    return Err(GprError::CholeskyFailed {
                        jitter,
                        matrix_size: n,
                        stage,
                    });
                }
                a[(j, j)] = sum.sqrt() as f32;
            } else {
                let diag = f64::from(a[(j, j)]);
                a[(i, j)] = (sum / diag) as f32;
            }
        }
    }
    Ok(())
}

/// Factors `A` in place as `L Lᵀ` and overwrites `rhs` with `A⁻¹ rhs`.
///
/// P1A-18 can call this on the same `Workspace` buffers as [`crate::Gpr::fit`].
pub(crate) fn cholesky_and_solve<T: KernelScalar>(
    a: &mut Mat<T>,
    rhs: &mut Mat<T>,
    scratch: &mut MemBuffer,
    jitter: f64,
    stage: CholeskyStage,
) -> Result<(), GprError> {
    cholesky_lower(a, scratch, jitter, stage)?;
    solve_llt_in_place(a.as_ref(), rhs.as_mut(), scratch);
    Ok(())
}

pub(crate) fn solve_llt_in_place<T: KernelScalar>(
    l: MatRef<'_, T>,
    rhs: MatMut<'_, T>,
    scratch: &mut MemBuffer,
) {
    T::solve_llt_in_place(l, rhs, scratch);
}

/// [`solve_llt_in_place`] for `f64` through faer.
pub(crate) fn solve_llt_faer(l: MatRef<'_, f64>, rhs: MatMut<'_, f64>, scratch: &mut MemBuffer) {
    let n = l.nrows();
    let n_rhs = rhs.ncols();
    let stack = MemStack::new(scratch);
    llt::solve::solve_in_place(l, rhs, faer_par_dims(n, n_rhs), stack);
}

/// [`solve_llt_in_place`] for `f32`: both triangular sweeps in `f64`.
pub(crate) fn solve_llt_f64_accum(l: MatRef<'_, f32>, mut rhs: MatMut<'_, f32>) {
    let n = l.nrows();
    let n_rhs = rhs.ncols();
    let mut y = vec![0.0f64; n];
    let mut x = vec![0.0f64; n];
    for col in 0..n_rhs {
        for i in 0..n {
            let mut sum = f64::from(rhs[(i, col)]);
            for j in 0..i {
                sum -= f64::from(l[(i, j)]) * y[j];
            }
            y[i] = sum / f64::from(l[(i, i)]);
        }
        for i in (0..n).rev() {
            let mut sum = y[i];
            for j in (i + 1)..n {
                sum -= f64::from(l[(j, i)]) * x[j];
            }
            x[i] = sum / f64::from(l[(i, i)]);
        }
        for i in 0..n {
            rhs[(i, col)] = x[i] as f32;
        }
    }
}

/// Solves `Lᵀ X = B` in place for lower `L`.
pub(crate) fn solve_lower_transpose<T: KernelScalar>(l: MatRef<'_, T>, rhs: MatMut<'_, T>) {
    let n = l.nrows();
    let n_rhs = rhs.ncols();
    faer::linalg::triangular_solve::solve_upper_triangular_in_place(
        l.transpose(),
        rhs,
        faer_par_dims(n, n_rhs),
    );
}

pub(crate) fn solve_lower<T: KernelScalar>(l: MatRef<'_, T>, rhs: MatMut<'_, T>) {
    let n = l.nrows();
    let n_rhs = rhs.ncols();
    faer::linalg::triangular_solve::solve_lower_triangular_in_place(
        l,
        rhs,
        faer_par_dims(n, n_rhs),
    );
}

pub(crate) fn append_chol_border<T: KernelScalar>(l: &Mat<T>, row: &[T], ell: T) -> Mat<T> {
    let m = l.nrows();
    let mut out = Mat::zeros(m + 1, m + 1);
    for j in 0..m {
        for i in j..m {
            out[(i, j)] = l[(i, j)];
        }
        out[(m, j)] = row[j];
    }
    out[(m, m)] = ell;
    out
}

pub(crate) fn delete_chol_row<T: KernelScalar>(l: &Mat<T>, idx: usize) -> Mat<T> {
    let m = l.nrows();
    let trail = m - idx - 1;
    let mut work = l.clone();
    if trail > 0 {
        let mut l22 = Mat::zeros(trail, trail);
        let mut v = vec![T::from_f64(0.0); trail];
        for j in 0..trail {
            for i in j..trail {
                l22[(i, j)] = work[(idx + 1 + i, idx + 1 + j)];
            }
            v[j] = work[(idx + 1 + j, idx)];
        }
        chol_rank1_update(&mut l22, &mut v);
        for j in 0..trail {
            for i in j..trail {
                work[(idx + 1 + i, idx + 1 + j)] = l22[(i, j)];
            }
        }
    }
    let mut out = Mat::zeros(m - 1, m - 1);
    let mut jo = 0;
    for j in 0..m {
        if j == idx {
            continue;
        }
        let mut io = 0;
        for i in 0..m {
            if i == idx {
                continue;
            }
            if io >= jo {
                out[(io, jo)] = work[(i, j)];
            }
            io += 1;
        }
        jo += 1;
    }
    out
}

pub(crate) fn solve_llt<T: KernelScalar>(l: MatRef<'_, T>, rhs: MatMut<'_, T>) {
    T::solve_llt_owned_scratch(l, rhs);
}

pub(crate) fn chol_rank1_update<T: KernelScalar>(l: &mut Mat<T>, v: &mut [T]) {
    let n = l.nrows();
    for k in 0..n {
        let lkk = l[(k, k)];
        let vk = v[k];
        let r = (lkk * lkk + vk * vk).sqrt();
        let c = r / lkk;
        let s = vk / lkk;
        l[(k, k)] = r;
        for i in (k + 1)..n {
            let li = l[(i, k)];
            let vi = v[i];
            l[(i, k)] = (li + s * vi) / c;
            v[i] = c * vi - s * l[(i, k)];
        }
    }
}

pub(crate) fn chol_rank1_downdate<T: KernelScalar>(l: &mut Mat<T>, v: &mut [T]) -> bool {
    let n = l.nrows();
    for k in 0..n {
        let lkk = l[(k, k)];
        let vk = v[k];
        let r2 = lkk * lkk - vk * vk;
        let res = {
            let r2_f = r2.to_f64();
            r2_f <= 0.0 || !r2_f.is_finite()
        };
        if res {
            return false;
        }
        let r = r2.sqrt();
        let c = r / lkk;
        let s = vk / lkk;
        l[(k, k)] = r;
        for i in (k + 1)..n {
            let li = l[(i, k)];
            let vi = v[i];
            l[(i, k)] = (li - s * vi) / c;
            v[i] = c * vi - s * l[(i, k)];
        }
    }
    true
}
