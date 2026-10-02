//! Training Gram assembly, Cholesky, and MLL helpers.

use faer::{Mat, MatMut, MatRef};

use crate::error::{CholeskyStage, GprError};
use crate::kernel::ArdSqDiffBuf;
use crate::kernel::{CompiledKernel, GramInputs, KernelScalar, Triangle};
use crate::linalg::{add_to_diag, cholesky_and_solve, log_det_from_l, retry_with_jitter};
use crate::precision::PrecisionPolicy;
use crate::workspace::FitWorkspace;

use crate::policy::JitterPolicy;

/// Writes the training Gram matrix.
///
/// [`crate::DistanceCachePolicy::Cached`] fills `dist_cache` (and ARD
/// `ard_sq_diff`) once and reuses them. [`crate::DistanceCachePolicy::Uncached`]
/// has no such tensors; isotropic and mixed trees compute distances from `X`.
fn apply_train_kernel<T, W, M: crate::math::KernelMath>(
    compiled: &CompiledKernel<T>,
    x: MatRef<'_, T>,
    ws: &mut W,
) -> Result<(), GprError>
where
    T: KernelScalar,
    W: FitWorkspace<Policy: PrecisionPolicy<Storage = T>>,
{
    let (core, dist) = ws.split_fit();
    apply_compiled_views::<T, M>(
        compiled,
        x,
        dist,
        core.k_matrix.as_mut(),
        core.exp_buf.as_mut(),
        &mut core.nested,
        &mut core.thread_scratch,
    )
}

/// Writes a compiled tree (or a single leaf) into `dest` from the fit views.
pub(crate) fn apply_compiled_to<T, W, M: crate::math::KernelMath>(
    compiled: &CompiledKernel<T>,
    x: MatRef<'_, T>,
    ws: &mut W,
    dest: MatMut<'_, T>,
) -> Result<(), GprError>
where
    T: KernelScalar,
    W: FitWorkspace<Policy: PrecisionPolicy<Storage = T>>,
{
    let (core, dist) = ws.split_fit();
    apply_compiled_views::<T, M>(
        compiled,
        x,
        dist,
        dest,
        core.exp_buf.as_mut(),
        &mut core.nested,
        &mut core.thread_scratch,
    )
}

fn apply_compiled_views<T: KernelScalar, M: crate::math::KernelMath>(
    compiled: &CompiledKernel<T>,
    x: MatRef<'_, T>,
    dist: Option<&mut crate::workspace::DistCache<T>>,
    dest: MatMut<'_, T>,
    scratch: MatMut<'_, T>,
    nested: &mut Vec<Mat<T>>,
    thread_scratch: &mut Vec<Mat<T>>,
) -> Result<(), GprError> {
    let inputs = fill_cached_inputs(compiled, x, dist, thread_scratch)?;
    compiled.eval_gram::<M>(inputs, dest, Triangle::Lower, scratch, nested)
}

/// Fills the training distance caches the tree reads (once per `X`) and
/// returns the views for a Gram evaluation. Without caches, only `x`.
pub(crate) fn fill_cached_inputs<'a, T: KernelScalar>(
    compiled: &CompiledKernel<T>,
    x: MatRef<'a, T>,
    dist: Option<&'a mut crate::workspace::DistCache<T>>,
    thread_scratch: &mut Vec<Mat<T>>,
) -> Result<GramInputs<'a, T>, GprError> {
    let Some(d) = dist else {
        return Ok(GramInputs::points(x));
    };
    let reads_dist = compiled.reads_distances()?;
    if reads_dist && d.dist.is_none() {
        let n = x.nrows();
        let mut dist = Mat::<T>::zeros(n, n);
        let mut pool = std::mem::take(thread_scratch);
        T::write_squared(x, dist.as_mut(), &mut pool);
        *thread_scratch = pool;
        d.dist = Some(dist);
    }
    let reads_ard = T::READS_ARD_CACHE && compiled.needs_ard_sq_diff();
    if reads_ard && d.ard_sq_diff.is_none() {
        d.ard_sq_diff = Some(ArdSqDiffBuf::new(x)?);
    }
    let d: &'a crate::workspace::DistCache<T> = d;
    Ok(GramInputs {
        x,
        dist: if reads_dist {
            d.dist.as_ref().map(Mat::as_ref)
        } else {
            None
        },
        ard: if reads_ard {
            d.ard_sq_diff.as_ref().map(ArdSqDiffBuf::view)
        } else {
            None
        },
    })
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

pub(crate) fn factor_train_with_policy<T, W, M: crate::math::KernelMath>(
    compiled: &CompiledKernel<T>,
    x: MatRef<'_, T>,
    ws: &mut W,
    y: &[f64],
    noise: f64,
    policy: FactorPolicy,
) -> Result<(), GprError>
where
    T: KernelScalar,
    W: FitWorkspace<Policy: PrecisionPolicy<Storage = T>>,
{
    factor_written_k_with_policy(ws, y, noise, policy, |ws| {
        apply_train_kernel::<T, W, M>(compiled, x, ws)
    })
}

/// [`factor_train_with_policy`] for the joint gradient: the factor Grams
/// of the first `products` products stay in the leading
/// [`CompiledKernel::kept_buffers`] of `weighted` for the gradient walk.
/// `weighted` and `kernel_scratch` must already be `n×n`.
pub(crate) fn factor_train_keeping_with_policy<T, W, M: crate::math::KernelMath>(
    compiled: &CompiledKernel<T>,
    x: MatRef<'_, T>,
    ws: &mut W,
    y: &[f64],
    noise: f64,
    policy: FactorPolicy,
    products: usize,
) -> Result<(), GprError>
where
    T: KernelScalar,
    W: FitWorkspace<Policy: PrecisionPolicy<Storage = T>>,
{
    let kept = compiled.kept_buffers();
    factor_written_k_with_policy(ws, y, noise, policy, |ws| {
        let (core, dist) = ws.split_fit();
        let inputs = fill_cached_inputs(compiled, x, dist, &mut core.thread_scratch)?;
        let grams = core
            .weighted
            .get_mut(..kept)
            .ok_or(GprError::WorkspaceTooSmall)?;
        compiled.eval_gram_keeping::<M>(
            inputs,
            core.k_matrix.as_mut(),
            core.exp_buf.as_mut(),
            core.kernel_scratch.as_mut(),
            &mut core.nested,
            grams,
            products,
        )
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
    let jitter = retry_with_jitter(policy.jitter.retry_jitters(), n, policy.stage, |j| {
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
    })?;
    ws.core_mut().factor_jitter = jitter;
    Ok(())
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
