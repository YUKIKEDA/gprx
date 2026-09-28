//! Training Gram assembly, Cholesky, and MLL helpers.

use faer::{Mat, MatMut, MatRef};

use crate::error::{CholeskyStage, GprError};
use crate::kernel::{
    CoordMode, FillDistances, GramKernel, KernelScalar, KernelSpec, MixedKernelViews, Triangle,
};
use crate::likelihood::GaussianLikelihood;
use crate::linalg::{add_to_diag, cholesky_and_solve, log_det_from_l, retry_with_jitter};
use crate::precision::PrecisionPolicy;
use crate::workspace::FitWorkspace;

use super::JitterPolicy;

/// Writes the training Gram matrix.
///
/// [`crate::CachedDistances`] fills `dist_cache` (and ARD `ard_sq_diff`) once
/// and reuses them. [`crate::UncachedDistances`] has no those tensors;
/// isotropic and mixed trees compute distances from `X`.
fn apply_train_kernel<K, W, M: crate::math::KernelMath>(
    compiled: &K,
    x: MatRef<'_, K::T>,
    ws: &mut W,
) -> Result<(), GprError>
where
    K: GramKernel,
    K::T: FillDistances + KernelScalar,
    W: FitWorkspace<Policy: PrecisionPolicy<Storage = K::T>>,
{
    let (core, dist) = ws.split_fit();
    apply_compiled_views::<K, M>(
        compiled,
        x,
        dist,
        core.k_matrix.as_mut(),
        core.exp_buf.as_mut(),
        &mut core.thread_scratch,
    )
}

/// Writes a compiled tree (or a single leaf) into `dest` from the fit views.
pub(crate) fn apply_compiled_to<K, W, M: crate::math::KernelMath>(
    compiled: &K,
    x: MatRef<'_, K::T>,
    ws: &mut W,
    dest: MatMut<'_, K::T>,
) -> Result<(), GprError>
where
    K: GramKernel,
    K::T: FillDistances + KernelScalar,
    W: FitWorkspace<Policy: PrecisionPolicy<Storage = K::T>>,
{
    let (core, dist) = ws.split_fit();
    apply_compiled_views::<K, M>(
        compiled,
        x,
        dist,
        dest,
        core.exp_buf.as_mut(),
        &mut core.thread_scratch,
    )
}

fn apply_compiled_views<K: GramKernel, M: crate::math::KernelMath>(
    compiled: &K,
    x: MatRef<'_, K::T>,
    dist: Option<crate::workspace::DistBufs<'_, K::T>>,
    dest: MatMut<'_, K::T>,
    scratch: MatMut<'_, K::T>,
    thread_scratch: &mut Vec<Mat<K::T>>,
) -> Result<(), GprError>
where
    K::T: FillDistances,
{
    let reads_ard = <K::T as FillDistances>::READS_ARD_CACHE;
    match compiled.coord_mode()? {
        CoordMode::Dist | CoordMode::Either => {
            if let Some(d) = dist {
                if !*d.dist_ready {
                    let mut pool = std::mem::take(thread_scratch);
                    K::T::write_squared(x, d.dist_cache.as_mut(), &mut pool);
                    *thread_scratch = pool;
                    *d.dist_ready = true;
                }
                compiled.apply::<M>(d.dist_cache.as_ref(), dest, Triangle::Lower, scratch)
            } else {
                compiled.apply_points::<M>(x, dest, Triangle::Lower, scratch)
            }
        }
        CoordMode::Points => {
            if reads_ard
                && compiled.needs_ard_sq_diff()
                && let Some(d) = dist
                && d.ard_sq_diff.ncols() > 0
            {
                if !*d.ard_sq_diff_ready {
                    let mut pool = std::mem::take(thread_scratch);
                    K::T::write_ard(x, d.ard_sq_diff.as_mut(), &mut pool);
                    *thread_scratch = pool;
                    *d.ard_sq_diff_ready = true;
                }
                compiled.apply_from_ard_cache::<M>(
                    d.ard_sq_diff.as_ref(),
                    x,
                    dest,
                    Triangle::Lower,
                    scratch,
                )
            } else {
                compiled.apply_points::<M>(x, dest, Triangle::Lower, scratch)
            }
        }
        CoordMode::Mixed => {
            if let Some(d) = dist {
                if !*d.dist_ready {
                    let mut pool = std::mem::take(thread_scratch);
                    K::T::write_squared(x, d.dist_cache.as_mut(), &mut pool);
                    *thread_scratch = pool;
                    *d.dist_ready = true;
                }
                let mut views = MixedKernelViews::new(d.dist_cache.as_ref(), x);
                if reads_ard && compiled.needs_ard_sq_diff() && d.ard_sq_diff.ncols() > 0 {
                    if !*d.ard_sq_diff_ready {
                        let mut pool = std::mem::take(thread_scratch);
                        K::T::write_ard(x, d.ard_sq_diff.as_mut(), &mut pool);
                        *thread_scratch = pool;
                        *d.ard_sq_diff_ready = true;
                    }
                    views.ard_cache = Some(d.ard_sq_diff.as_ref());
                }
                compiled.apply_mixed::<M>(views, dest, Triangle::Lower, scratch)
            } else {
                compiled.apply_points::<M>(x, dest, Triangle::Lower, scratch)
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

pub(crate) fn finish_train_system<W>(ws: &mut W, y: &[f64], noise: f64, extra_diag: f64)
where
    W: FitWorkspace,
{
    let core = ws.core_mut();
    add_to_diag(core.k_matrix.as_mut(), noise);
    if extra_diag != 0.0 {
        add_to_diag(core.k_matrix.as_mut(), extra_diag);
    }
    for (i, &yi) in y.iter().enumerate() {
        core.rhs[(i, 0)] = <W::Policy as PrecisionPolicy>::Storage::from_f64(yi);
    }
}

pub(crate) struct FactorPolicy {
    pub(crate) jitter: JitterPolicy,
    pub(crate) stage: CholeskyStage,
}

pub(crate) fn factor_train_with_policy<K, W, M: crate::math::KernelMath>(
    compiled: &K,
    x: MatRef<'_, K::T>,
    ws: &mut W,
    y: &[f64],
    noise: f64,
    policy: FactorPolicy,
) -> Result<(), GprError>
where
    K: GramKernel,
    K::T: FillDistances + KernelScalar,
    W: FitWorkspace<Policy: PrecisionPolicy<Storage = K::T>>,
{
    factor_written_k_with_policy(ws, y, noise, policy, |ws| {
        apply_train_kernel::<K, W, M>(compiled, x, ws)
    })
}

/// Factors after `write_k` fills the lower training Gram (no noise).
///
/// Clears `k_matrix` before every `write_k`. In-place Cholesky overwrites
/// that buffer with `L` (or a partial factor on failure). A later
/// `write_k` that accumulates — Sum [`crate::kernel::CompiledKernel`]
/// combine uses `add_triangle` — must not add into that leftover.
pub(crate) fn factor_written_k_with_policy<W, F>(
    ws: &mut W,
    y: &[f64],
    noise: f64,
    policy: FactorPolicy,
    mut write_k: F,
) -> Result<(), GprError>
where
    W: FitWorkspace,
    F: FnMut(&mut W) -> Result<(), GprError>,
{
    let n = ws.core().k_matrix.nrows();
    retry_with_jitter(policy.jitter.retry_jitters(), n, policy.stage, |j| {
        clear_train_gram(ws);
        write_k(ws)?;
        finish_train_system(ws, y, noise, j);
        let core = ws.core_mut();
        cholesky_and_solve(
            &mut core.k_matrix,
            &mut core.rhs,
            &mut core.faer_scratch,
            0.0,
            policy.stage,
        )
    })
}

fn clear_train_gram<W>(ws: &mut W)
where
    W: FitWorkspace,
{
    let zero = <W::Policy as PrecisionPolicy>::Storage::from_f64(0.0);
    ws.core_mut().k_matrix.fill(zero);
}

pub(crate) fn neg_mll_from_factor<T: KernelScalar>(
    l: MatRef<'_, T>,
    y: &[T],
    alpha: &[T],
    n: usize,
) -> T {
    let mut quad = T::from_f64(0.0);
    for i in 0..n {
        quad += y[i] * alpha[i];
    }
    let log_det = log_det_from_l(l, n);
    let log_two_pi = T::from_f64((2.0 * std::f64::consts::PI).ln());
    T::from_f64(0.5) * (quad + log_det + T::from_f64(n as f64) * log_two_pi)
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

pub(crate) fn write_kernel_grad<K: GramKernel, M: crate::math::KernelMath>(
    compiled: &K,
    dist: MatRef<'_, K::T>,
    x: MatRef<'_, K::T>,
    ard_cache: Option<MatRef<'_, K::T>>,
    d_k: MatMut<'_, K::T>,
    scratch: MatMut<'_, K::T>,
    param_idx: usize,
) -> Result<(), GprError>
where
    K::T: FillDistances,
{
    let reads_ard = <K::T as FillDistances>::READS_ARD_CACHE;
    match compiled.coord_mode()? {
        CoordMode::Dist | CoordMode::Either => {
            compiled.grad::<M>(dist, d_k, param_idx, Triangle::Lower, scratch)
        }
        CoordMode::Points => {
            if reads_ard && let Some(cache) = ard_cache {
                compiled.grad_from_ard_cache::<M>(
                    cache,
                    x,
                    d_k,
                    param_idx,
                    Triangle::Lower,
                    scratch,
                )
            } else {
                compiled.grad_points::<M>(x, d_k, param_idx, Triangle::Lower, scratch)
            }
        }
        CoordMode::Mixed => compiled.grad_mixed::<M>(
            MixedKernelViews::new(dist, x),
            d_k,
            param_idx,
            Triangle::Lower,
            scratch,
        ),
    }
}

pub(crate) fn write_kernel_grad_from_coords<K: GramKernel, M: crate::math::KernelMath>(
    compiled: &K,
    x: MatRef<'_, K::T>,
    d_k: MatMut<'_, K::T>,
    scratch: MatMut<'_, K::T>,
    param_idx: usize,
) -> Result<(), GprError> {
    compiled.grad_points::<M>(x, d_k, param_idx, Triangle::Lower, scratch)
}

pub(crate) fn write_kernel_hess<K: GramKernel, M: crate::math::KernelMath>(
    compiled: &K,
    dist: MatRef<'_, K::T>,
    x: MatRef<'_, K::T>,
    ard_cache: Option<MatRef<'_, K::T>>,
    d2_k: MatMut<'_, K::T>,
    scratch: MatMut<'_, K::T>,
    pair: (usize, usize),
) -> Result<(), GprError>
where
    K::T: FillDistances,
{
    let (i, j) = pair;
    let reads_ard = <K::T as FillDistances>::READS_ARD_CACHE;
    match compiled.coord_mode()? {
        CoordMode::Dist | CoordMode::Either => {
            compiled.hess::<M>(dist, d2_k, i, j, Triangle::Lower, scratch)
        }
        CoordMode::Points => {
            if reads_ard && let Some(cache) = ard_cache {
                compiled.hess_from_ard_cache::<M>(cache, x, d2_k, pair, Triangle::Lower, scratch)
            } else {
                compiled.hess_points::<M>(x, d2_k, i, j, Triangle::Lower, scratch)
            }
        }
        CoordMode::Mixed => compiled.hess_mixed::<M>(
            MixedKernelViews::new(dist, x),
            d2_k,
            i,
            j,
            Triangle::Lower,
            scratch,
        ),
    }
}

pub(crate) fn write_kernel_hess_from_coords<K: GramKernel, M: crate::math::KernelMath>(
    compiled: &K,
    x: MatRef<'_, K::T>,
    d2_k: MatMut<'_, K::T>,
    scratch: MatMut<'_, K::T>,
    i: usize,
    j: usize,
) -> Result<(), GprError> {
    compiled.hess_points::<M>(x, d2_k, i, j, Triangle::Lower, scratch)
}

pub(crate) fn pack_storage<T: KernelScalar>(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    mut dest: MatMut<'_, T>,
) {
    debug_assert_eq!(dest.nrows(), n_rows);
    debug_assert_eq!(dest.ncols(), n_cols);
    for col in 0..n_cols {
        for row in 0..n_rows {
            dest[(row, col)] = T::from_f64(x[col * n_rows + row]);
        }
    }
}
