//! Training Gram assembly, Cholesky, and MLL helpers.

use dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt;
use faer::linalg::cholesky::llt::factor::{LltError, LltRegularization};
use faer::{Mat, MatMut, MatRef};

use crate::error::{CholeskyStage, GprError};
use crate::kernel::{
    CoordMode, FillDistances, GramKernel, KernelScalar, KernelSpec, MixedKernelViews, Triangle,
};
use crate::likelihood::GaussianLikelihood;
use crate::precision::{PrecisionPolicy, StorageScalar};
use crate::workspace::{FitWorkspace, faer_par, faer_par_dims};

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
    K::T: FillDistances + StorageScalar,
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
    K::T: FillDistances + StorageScalar,
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

pub(crate) fn add_noise_to_diag<T: StorageScalar>(mut k: MatMut<'_, T>, noise: f64) {
    let n = k.nrows();
    let noise = T::from_f64(noise);
    for i in 0..n {
        k[(i, i)] += noise;
    }
}

pub(crate) fn finish_train_system<W>(ws: &mut W, y: &[f64], noise: f64, extra_diag: f64)
where
    W: FitWorkspace,
    <W::Policy as PrecisionPolicy>::Storage: StorageScalar,
{
    let core = ws.core_mut();
    add_noise_to_diag(core.k_matrix.as_mut(), noise);
    if extra_diag != 0.0 {
        add_noise_to_diag(core.k_matrix.as_mut(), extra_diag);
    }
    for (i, &yi) in y.iter().enumerate() {
        core.rhs[(i, 0)] = <W::Policy as PrecisionPolicy>::Storage::from_f64(yi);
    }
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
    K::T: FillDistances + StorageScalar,
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
    <W::Policy as PrecisionPolicy>::Storage: StorageScalar,
    F: FnMut(&mut W) -> Result<(), GprError>,
{
    clear_train_gram(ws);
    write_k(ws)?;
    finish_train_system(ws, y, noise, 0.0);
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
        clear_train_gram(ws);
        write_k(ws)?;
        finish_train_system(ws, y, noise, j);
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

fn clear_train_gram<W>(ws: &mut W)
where
    W: FitWorkspace,
    <W::Policy as PrecisionPolicy>::Storage: StorageScalar,
{
    let zero = <W::Policy as PrecisionPolicy>::Storage::from_f64(0.0);
    ws.core_mut().k_matrix.fill(zero);
}

pub(crate) fn log_det_from_l<T: StorageScalar>(l: MatRef<'_, T>, n: usize) -> T {
    let mut log_diag = T::from_f64(0.0);
    for i in 0..n {
        log_diag += StorageScalar::ln(l[(i, i)]);
    }
    T::from_f64(2.0) * log_diag
}

pub(crate) fn neg_mll_from_factor<T: StorageScalar>(
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

pub(crate) fn frobenius_lower<T: StorageScalar>(
    w: MatRef<'_, T>,
    d_k: MatRef<'_, T>,
    n: usize,
) -> T {
    let mut inner = T::from_f64(0.0);
    let two = T::from_f64(2.0);
    for col in 0..n {
        inner += w[(col, col)] * d_k[(col, col)];
        for row in col + 1..n {
            inner += two * w[(row, col)] * d_k[(row, col)];
        }
    }
    inner
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

pub(crate) fn pack_storage<T: StorageScalar>(
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

pub(crate) fn symmetrize_lower<T: StorageScalar>(mut a: MatMut<'_, T>, n: usize) {
    for col in 0..n {
        for row in col + 1..n {
            a[(col, row)] = a[(row, col)];
        }
    }
}

pub(crate) fn gemv_sym_lower<T: StorageScalar>(a: MatRef<'_, T>, x: &[T], y: &mut [T], n: usize) {
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

pub(crate) fn gemv_full<T: StorageScalar>(a: MatRef<'_, T>, x: &[T], y: &mut [T], n: usize) {
    let zero = T::from_f64(0.0);
    for i in 0..n {
        let mut s = zero;
        for j in 0..n {
            s += a[(i, j)] * x[j];
        }
        y[i] = s;
    }
}

pub(crate) fn trace_product<T: StorageScalar>(a: MatRef<'_, T>, b: MatRef<'_, T>, n: usize) -> T {
    let mut tr = T::from_f64(0.0);
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
#[allow(clippy::needless_range_loop)]
pub(crate) fn inv_diag_from_chol_l<T: StorageScalar>(l: MatRef<'_, T>, q_diag: &mut [T]) {
    let n = l.nrows();
    debug_assert_eq!(q_diag.len(), n);
    if std::mem::size_of::<T>() == std::mem::size_of::<f32>() {
        for i in 0..n {
            let mut col = vec![0.0f64; n];
            for row in 0..n {
                col[row] = if row == i { 1.0 } else { 0.0 };
            }
            for row in 0..n {
                let mut sum = col[row];
                for k in 0..row {
                    sum -= l[(row, k)].to_f64() * col[k];
                }
                col[row] = sum / l[(row, row)].to_f64();
            }
            let mut q = 0.0f64;
            for v in &col {
                q += v * v;
            }
            q_diag[i] = T::from_f64(q);
        }
        return;
    }
    let one = T::from_f64(1.0);
    let zero = T::from_f64(0.0);
    let mut inv_l = Mat::from_fn(n, n, |row, col| if row == col { one } else { zero });
    faer::linalg::triangular_solve::solve_lower_triangular_in_place(l, inv_l.as_mut(), faer_par(n));
    for (i, qi) in q_diag.iter_mut().enumerate() {
        let mut q = zero;
        for k in 0..n {
            let v = inv_l[(k, i)];
            q += v * v;
        }
        *qi = q;
    }
}

/// Factors `A` in place as `L Lᵀ`. The strictly upper triangle is unspecified.
pub(crate) fn cholesky_lower<T: StorageScalar>(
    a: &mut Mat<T>,
    scratch: &mut MemBuffer,
    jitter: f64,
    stage: CholeskyStage,
) -> Result<(), GprError> {
    let n = a.nrows();
    if std::mem::size_of::<T>() == std::mem::size_of::<f32>() {
        return cholesky_lower_f32_accum(a, jitter, stage);
    }
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

fn cholesky_lower_f32_accum<T: StorageScalar>(
    a: &mut Mat<T>,
    jitter: f64,
    stage: CholeskyStage,
) -> Result<(), GprError> {
    let n = a.nrows();
    for j in 0..n {
        for i in j..n {
            let mut sum = a[(i, j)].to_f64();
            for k in 0..j {
                sum -= a[(i, k)].to_f64() * a[(j, k)].to_f64();
            }
            if i == j {
                if sum.is_nan() || sum <= 0.0 {
                    return Err(GprError::CholeskyFailed {
                        jitter,
                        matrix_size: n,
                        stage,
                    });
                }
                a[(j, j)] = T::from_f64(sum.sqrt());
            } else {
                let diag = a[(j, j)].to_f64();
                a[(i, j)] = T::from_f64(sum / diag);
            }
        }
    }
    Ok(())
}

/// Factors `A` in place as `L Lᵀ`, retrying with [`JitterPolicy`] on failure.
///
/// The first attempt uses `A` as given. Each retry restores that snapshot and
/// adds `j` to the diagonal. Used for the posterior covariance in
/// [`crate::FittedGpr::sample`].
pub(crate) fn cholesky_lower_with_policy<T: StorageScalar>(
    a: &mut Mat<T>,
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
pub(crate) fn cholesky_and_solve<T: StorageScalar>(
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

pub(crate) fn solve_llt_in_place<T: StorageScalar>(
    l: MatRef<'_, T>,
    mut rhs: MatMut<'_, T>,
    scratch: &mut MemBuffer,
) {
    let n = l.nrows();
    let n_rhs = rhs.ncols();
    if std::mem::size_of::<T>() != std::mem::size_of::<f32>() {
        let stack = MemStack::new(scratch);
        llt::solve::solve_in_place(l, rhs.as_mut(), faer_par_dims(n, n_rhs), stack);
        return;
    }
    for col in 0..n_rhs {
        let mut y = vec![0.0f64; n];
        let mut x = vec![0.0f64; n];
        for i in 0..n {
            let mut sum = rhs[(i, col)].to_f64();
            for j in 0..i {
                sum -= l[(i, j)].to_f64() * y[j];
            }
            y[i] = sum / l[(i, i)].to_f64();
        }
        for i in (0..n).rev() {
            let mut sum = y[i];
            for j in (i + 1)..n {
                sum -= l[(j, i)].to_f64() * x[j];
            }
            x[i] = sum / l[(i, i)].to_f64();
        }
        for i in 0..n {
            rhs[(i, col)] = T::from_f64(x[i]);
        }
    }
}
