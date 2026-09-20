//! Training Gram assembly, Cholesky, and MLL helpers.

use dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt;
use faer::linalg::cholesky::llt::factor::{LltError, LltRegularization};
use faer::{Mat, MatMut, MatRef};

use crate::error::{CholeskyStage, GprError};
use crate::kernel::{
    CompiledKernel, CoordMode, KernelSpec, MixedKernelViews, Triangle, fill_ard_squared_diff,
    fill_squared_euclidean,
};
use crate::likelihood::GaussianLikelihood;
use crate::workspace::{FitWorkspace, faer_par, faer_par_dims};

use super::JitterPolicy;

/// Writes the training Gram matrix.
///
/// [`crate::CachedDistances`] fills `dist_cache` (and ARD `ard_sq_diff`) once
/// and reuses them. [`crate::UncachedDistances`] has no those tensors;
/// isotropic and mixed trees compute distances from `X`.
fn apply_train_kernel<W: FitWorkspace>(
    compiled: &CompiledKernel,
    x: MatRef<'_, f64>,
    ws: &mut W,
) -> Result<(), GprError> {
    let (core, dist) = ws.split_fit();
    match compiled.coord_mode()? {
        CoordMode::Dist | CoordMode::Either => {
            if let Some(d) = dist {
                if !*d.dist_ready {
                    let mut thread_scratch = std::mem::take(&mut core.thread_scratch);
                    fill_squared_euclidean(x, d.dist_cache.as_mut(), &mut thread_scratch);
                    core.thread_scratch = thread_scratch;
                    *d.dist_ready = true;
                }
                compiled.apply(
                    d.dist_cache.as_ref(),
                    core.k_matrix.as_mut(),
                    Triangle::Lower,
                    core.exp_buf.as_mut(),
                )
            } else {
                compiled.apply_points(
                    x,
                    core.k_matrix.as_mut(),
                    Triangle::Lower,
                    core.exp_buf.as_mut(),
                )
            }
        }
        CoordMode::Points => {
            if compiled.needs_ard_sq_diff()
                && let Some(d) = dist
                && d.ard_sq_diff.ncols() > 0
            {
                if !*d.ard_sq_diff_ready {
                    let mut thread_scratch = std::mem::take(&mut core.thread_scratch);
                    fill_ard_squared_diff(x, d.ard_sq_diff.as_mut(), &mut thread_scratch);
                    core.thread_scratch = thread_scratch;
                    *d.ard_sq_diff_ready = true;
                }
                compiled.apply_from_ard_cache(
                    d.ard_sq_diff.as_ref(),
                    x,
                    core.k_matrix.as_mut(),
                    Triangle::Lower,
                    core.exp_buf.as_mut(),
                )
            } else {
                compiled.apply_points(
                    x,
                    core.k_matrix.as_mut(),
                    Triangle::Lower,
                    core.exp_buf.as_mut(),
                )
            }
        }
        CoordMode::Mixed => {
            if let Some(d) = dist {
                if !*d.dist_ready {
                    let mut thread_scratch = std::mem::take(&mut core.thread_scratch);
                    fill_squared_euclidean(x, d.dist_cache.as_mut(), &mut thread_scratch);
                    core.thread_scratch = thread_scratch;
                    *d.dist_ready = true;
                }
                let mut views = MixedKernelViews::new(d.dist_cache.as_ref(), x);
                if compiled.needs_ard_sq_diff() && d.ard_sq_diff.ncols() > 0 {
                    if !*d.ard_sq_diff_ready {
                        let mut thread_scratch = std::mem::take(&mut core.thread_scratch);
                        fill_ard_squared_diff(x, d.ard_sq_diff.as_mut(), &mut thread_scratch);
                        core.thread_scratch = thread_scratch;
                        *d.ard_sq_diff_ready = true;
                    }
                    views.ard_cache = Some(d.ard_sq_diff.as_ref());
                }
                compiled.apply_mixed(
                    views,
                    core.k_matrix.as_mut(),
                    Triangle::Lower,
                    core.exp_buf.as_mut(),
                )
            } else {
                compiled.apply_points(
                    x,
                    core.k_matrix.as_mut(),
                    Triangle::Lower,
                    core.exp_buf.as_mut(),
                )
            }
        }
    }
}

pub(crate) fn validate_training(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    y: &[f64],
) -> Result<(), GprError> {
    if n_rows == 0 || n_cols == 0 {
        return Err(GprError::EmptyInput);
    }
    let expected_x = n_rows.checked_mul(n_cols).ok_or(GprError::EmptyInput)?;
    if x.len() != expected_x {
        return Err(GprError::InvalidHyperparameter {
            reason: format!("expected {expected_x} feature values, got {}", x.len()),
        });
    }
    if y.len() != n_rows {
        return Err(GprError::InvalidHyperparameter {
            reason: format!("expected {n_rows} targets, got {}", y.len()),
        });
    }
    if x.iter().any(|v| !v.is_finite()) || y.iter().any(|v| !v.is_finite()) {
        return Err(GprError::NonFiniteInput);
    }
    Ok(())
}

pub(crate) fn validate_query(xs: &[f64], n_rows: usize, n_cols: usize) -> Result<(), GprError> {
    if n_rows == 0 || n_cols == 0 {
        return Err(GprError::EmptyInput);
    }
    let expected = n_rows.checked_mul(n_cols).ok_or(GprError::EmptyInput)?;
    if xs.len() != expected {
        return Err(GprError::InvalidHyperparameter {
            reason: format!("expected {expected} feature values, got {}", xs.len()),
        });
    }
    if xs.iter().any(|v| !v.is_finite()) {
        return Err(GprError::NonFiniteInput);
    }
    Ok(())
}

pub(crate) fn pack_points(x: &[f64], n_rows: usize, n_cols: usize) -> Mat<f64> {
    let mut dest = Mat::zeros(n_rows, n_cols);
    pack_points_into(x, n_rows, n_cols, dest.as_mut());
    dest
}

pub(crate) fn pack_points_into(x: &[f64], n_rows: usize, n_cols: usize, mut dest: MatMut<'_, f64>) {
    debug_assert_eq!(dest.nrows(), n_rows);
    debug_assert_eq!(dest.ncols(), n_cols);
    for col in 0..n_cols {
        for row in 0..n_rows {
            dest[(row, col)] = x[col * n_rows + row];
        }
    }
}

pub(crate) fn add_noise_to_diag(mut k: MatMut<'_, f64>, noise: f64) {
    let n = k.nrows();
    for i in 0..n {
        k[(i, i)] += noise;
    }
}

fn assemble_train_system<W: FitWorkspace>(
    compiled: &CompiledKernel,
    x: MatRef<'_, f64>,
    ws: &mut W,
    y: &[f64],
    noise: f64,
    extra_diag: f64,
) -> Result<(), GprError> {
    apply_train_kernel(compiled, x, ws)?;
    let core = ws.core_mut();
    add_noise_to_diag(core.k_matrix.as_mut(), noise);
    if extra_diag != 0.0 {
        add_noise_to_diag(core.k_matrix.as_mut(), extra_diag);
    }
    for (i, &yi) in y.iter().enumerate() {
        core.rhs[(i, 0)] = yi;
    }
    Ok(())
}

pub(crate) struct FactorPolicy {
    pub(crate) jitter: JitterPolicy,
    pub(crate) stage: CholeskyStage,
}

fn map_cholesky_jitter(err: GprError, jitter: f64) -> GprError {
    match err {
        GprError::CholeskyFailed {
            matrix_size, stage, ..
        } => GprError::CholeskyFailed {
            jitter,
            matrix_size,
            stage,
        },
        other => other,
    }
}

pub(crate) fn factor_train_with_policy<W: FitWorkspace>(
    compiled: &CompiledKernel,
    x: MatRef<'_, f64>,
    ws: &mut W,
    y: &[f64],
    noise: f64,
    policy: FactorPolicy,
) -> Result<(), GprError> {
    assemble_train_system(compiled, x, ws, y, noise, 0.0)?;
    {
        let core = ws.core_mut();
        match cholesky_and_solve(
            &mut core.k_matrix,
            &mut core.rhs,
            &mut core.faer_scratch,
            0.0,
            policy.stage,
        ) {
            Ok(()) => return Ok(()),
            Err(GprError::CholeskyFailed { .. }) => {}
            Err(err) => return Err(err),
        }
    }
    let mut last_j = 0.0;
    for j in policy.jitter.retry_jitters() {
        last_j = j;
        assemble_train_system(compiled, x, ws, y, noise, j)?;
        let core = ws.core_mut();
        match cholesky_and_solve(
            &mut core.k_matrix,
            &mut core.rhs,
            &mut core.faer_scratch,
            0.0,
            policy.stage,
        ) {
            Ok(()) => return Ok(()),
            Err(GprError::CholeskyFailed { .. }) => {}
            Err(err) => return Err(map_cholesky_jitter(err, j)),
        }
    }
    let n = ws.core().k_matrix.nrows();
    Err(GprError::CholeskyFailed {
        jitter: last_j,
        matrix_size: n,
        stage: policy.stage,
    })
}

pub(crate) fn log_det_from_l(l: MatRef<'_, f64>, n: usize) -> f64 {
    let mut log_diag = 0.0;
    for i in 0..n {
        log_diag += l[(i, i)].ln();
    }
    2.0 * log_diag
}

pub(crate) fn neg_mll_from_factor(l: MatRef<'_, f64>, y: &[f64], alpha: &[f64], n: usize) -> f64 {
    let mut quad = 0.0;
    for i in 0..n {
        quad += y[i] * alpha[i];
    }
    let log_det = log_det_from_l(l, n);
    let log_two_pi = (2.0 * std::f64::consts::PI).ln();
    0.5 * (quad + log_det + n as f64 * log_two_pi)
}

pub(crate) fn write_params(
    kernel: &KernelSpec,
    likelihood: &GaussianLikelihood,
    out: &mut [f64],
) -> Result<(), GprError> {
    let n_kernel = kernel.num_params();
    require_param_len(out.len(), n_kernel + likelihood.num_params())?;
    kernel.get_params(&mut out[..n_kernel])?;
    likelihood.get_params(&mut out[n_kernel..])
}

pub(crate) fn require_param_len(actual: usize, expected: usize) -> Result<(), GprError> {
    if actual == expected {
        Ok(())
    } else {
        Err(GprError::InvalidHyperparameter {
            reason: format!("expected {expected} parameters, got {actual}"),
        })
    }
}

pub(crate) fn frobenius_lower(w: MatRef<'_, f64>, d_k: MatRef<'_, f64>, n: usize) -> f64 {
    let mut inner = 0.0;
    for col in 0..n {
        inner += w[(col, col)] * d_k[(col, col)];
        for row in col + 1..n {
            inner += 2.0 * w[(row, col)] * d_k[(row, col)];
        }
    }
    inner
}

pub(crate) fn write_kernel_grad(
    compiled: &CompiledKernel,
    dist: MatRef<'_, f64>,
    x: MatRef<'_, f64>,
    ard_cache: Option<MatRef<'_, f64>>,
    d_k: MatMut<'_, f64>,
    scratch: MatMut<'_, f64>,
    param_idx: usize,
) -> Result<(), GprError> {
    match compiled.coord_mode()? {
        CoordMode::Dist | CoordMode::Either => {
            compiled.grad(dist, d_k, param_idx, Triangle::Lower, scratch)
        }
        CoordMode::Points => {
            if let Some(cache) = ard_cache {
                compiled.grad_from_ard_cache(cache, x, d_k, param_idx, Triangle::Lower, scratch)
            } else {
                compiled.grad_points(x, d_k, param_idx, Triangle::Lower, scratch)
            }
        }
        CoordMode::Mixed => compiled.grad_mixed(
            MixedKernelViews::new(dist, x),
            d_k,
            param_idx,
            Triangle::Lower,
            scratch,
        ),
    }
}

pub(crate) fn write_kernel_grad_from_coords(
    compiled: &CompiledKernel,
    x: MatRef<'_, f64>,
    d_k: MatMut<'_, f64>,
    scratch: MatMut<'_, f64>,
    param_idx: usize,
) -> Result<(), GprError> {
    compiled.grad_points(x, d_k, param_idx, Triangle::Lower, scratch)
}

pub(crate) fn write_kernel_hess(
    compiled: &CompiledKernel,
    dist: MatRef<'_, f64>,
    x: MatRef<'_, f64>,
    ard_cache: Option<MatRef<'_, f64>>,
    d2_k: MatMut<'_, f64>,
    scratch: MatMut<'_, f64>,
    pair: (usize, usize),
) -> Result<(), GprError> {
    let (i, j) = pair;
    match compiled.coord_mode()? {
        CoordMode::Dist | CoordMode::Either => {
            compiled.hess(dist, d2_k, i, j, Triangle::Lower, scratch)
        }
        CoordMode::Points => {
            if let Some(cache) = ard_cache {
                compiled.hess_from_ard_cache(cache, x, d2_k, pair, Triangle::Lower, scratch)
            } else {
                compiled.hess_points(x, d2_k, i, j, Triangle::Lower, scratch)
            }
        }
        CoordMode::Mixed => compiled.hess_mixed(
            MixedKernelViews::new(dist, x),
            d2_k,
            i,
            j,
            Triangle::Lower,
            scratch,
        ),
    }
}

pub(crate) fn write_kernel_hess_from_coords(
    compiled: &CompiledKernel,
    x: MatRef<'_, f64>,
    d2_k: MatMut<'_, f64>,
    scratch: MatMut<'_, f64>,
    i: usize,
    j: usize,
) -> Result<(), GprError> {
    compiled.hess_points(x, d2_k, i, j, Triangle::Lower, scratch)
}

pub(crate) fn symmetrize_lower(mut a: MatMut<'_, f64>, n: usize) {
    for col in 0..n {
        for row in col + 1..n {
            a[(col, row)] = a[(row, col)];
        }
    }
}

pub(crate) fn gemv_sym_lower(a: MatRef<'_, f64>, x: &[f64], y: &mut [f64], n: usize) {
    for i in 0..n {
        let mut s = a[(i, i)] * x[i];
        for j in 0..i {
            s += a[(i, j)] * x[j];
        }
        for j in i + 1..n {
            s += a[(j, i)] * x[j];
        }
        y[i] = s;
    }
}

pub(crate) fn gemv_full(a: MatRef<'_, f64>, x: &[f64], y: &mut [f64], n: usize) {
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..n {
            s += a[(i, j)] * x[j];
        }
        y[i] = s;
    }
}

pub(crate) fn trace_product(a: MatRef<'_, f64>, b: MatRef<'_, f64>, n: usize) -> f64 {
    let mut tr = 0.0;
    for col in 0..n {
        for row in 0..n {
            tr += a[(row, col)] * b[(col, row)];
        }
    }
    tr
}

/// Writes `diag(A⁻¹)` given the lower Cholesky factor `L` of `A = L Lᵀ`.
///
/// `A⁻¹ = L^{-T} L^{-1}`, so entry `i` is the squared Euclidean norm of
/// column `i` of `L⁻¹`.
pub(crate) fn inv_diag_from_chol_l(l: MatRef<'_, f64>, q_diag: &mut [f64]) {
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
pub(crate) fn cholesky_lower(
    a: &mut Mat<f64>,
    scratch: &mut MemBuffer,
    jitter: f64,
    stage: CholeskyStage,
) -> Result<(), GprError> {
    let n = a.nrows();
    let regularization = LltRegularization {
        dynamic_regularization_delta: jitter,
        dynamic_regularization_epsilon: 0.0,
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

/// Factors `A` in place as `L Lᵀ`, retrying with [`JitterPolicy`] on failure.
///
/// The first attempt uses `A` as given. Each retry restores that snapshot and
/// adds `j` to the diagonal. Used for the posterior covariance in
/// [`crate::FittedGpr::sample`].
pub(crate) fn cholesky_lower_with_policy(
    a: &mut Mat<f64>,
    scratch: &mut MemBuffer,
    policy: JitterPolicy,
    stage: CholeskyStage,
) -> Result<(), GprError> {
    let backup = a.clone();
    match cholesky_lower(a, scratch, 0.0, stage) {
        Ok(()) => return Ok(()),
        Err(GprError::CholeskyFailed { .. }) => {}
        Err(err) => return Err(err),
    }
    let mut last_j = 0.0;
    for j in policy.retry_jitters() {
        last_j = j;
        *a = backup.clone();
        add_noise_to_diag(a.as_mut(), j);
        match cholesky_lower(a, scratch, 0.0, stage) {
            Ok(()) => return Ok(()),
            Err(GprError::CholeskyFailed { .. }) => {}
            Err(err) => return Err(map_cholesky_jitter(err, j)),
        }
    }
    Err(GprError::CholeskyFailed {
        jitter: last_j,
        matrix_size: a.nrows(),
        stage,
    })
}

/// Factors `A` in place as `L Lᵀ` and overwrites `rhs` with `A⁻¹ rhs`.
///
/// P1A-18 can call this on the same `Workspace` buffers as [`crate::Gpr::fit`].
pub(crate) fn cholesky_and_solve(
    a: &mut Mat<f64>,
    rhs: &mut Mat<f64>,
    scratch: &mut MemBuffer,
    jitter: f64,
    stage: CholeskyStage,
) -> Result<(), GprError> {
    cholesky_lower(a, scratch, jitter, stage)?;
    let n = a.nrows();
    let n_rhs = rhs.ncols();
    let stack = MemStack::new(scratch);
    llt::solve::solve_in_place(a.as_ref(), rhs.as_mut(), faer_par_dims(n, n_rhs), stack);
    Ok(())
}
