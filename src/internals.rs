//! Measurement hooks for this repository's benches and `compare/perf`.
//!
//! Present only with the non-default features `bench-internals` or
//! `insert-stages`. Nothing here is part of the model API, and this module
//! is not covered by semantic versioning.

#[cfg(feature = "bench-internals")]
use faer::{MatMut, MatRef};

/// Writes the full pairwise squared Euclidean distances of the rows of `x`
/// with the library's parallel fill (the `kernel_rbf` bench input).
#[cfg(feature = "bench-internals")]
pub fn fill_pairwise_sq_euclidean(x: MatRef<'_, f64>, dist: MatMut<'_, f64>) {
    crate::kernel::fill_squared_euclidean(x, dist, &mut []);
}

/// The raw `(Δx_d)²` cache that ARD fits read: the lower triangle of each
/// dimension, packed by column (`d · n(n+1)/2` values).
#[cfg(feature = "bench-internals")]
pub struct ArdCache(crate::kernel::ArdSqDiffBuf<f64>);

/// Fills the [`ArdCache`] of the rows of `x` with the library's parallel fill.
///
/// # Errors
///
/// Returns [`crate::GprError::SizeOverflow`] when the cache size overflows.
#[cfg(feature = "bench-internals")]
pub fn fill_ard_squared_diff(x: MatRef<'_, f64>) -> Result<ArdCache, crate::GprError> {
    crate::kernel::ArdSqDiffBuf::new(x).map(ArdCache)
}

/// [`CompiledKernel::apply_points`](crate::kernel::CompiledKernel::apply_points)
/// for an ARD tree, read from the `(Δx_d)²` cache that
/// [`fill_ard_squared_diff`] writes.
///
/// # Errors
///
/// Same as `apply_points`, or a shape error when `cache` was filled for
/// another `n` or `d`.
#[cfg(feature = "bench-internals")]
pub fn apply_from_ard_cache<M: crate::KernelMath>(
    kernel: &crate::kernel::CompiledKernel,
    cache: &ArdCache,
    x: MatRef<'_, f64>,
    out: MatMut<'_, f64>,
    uplo: crate::kernel::Triangle,
    scratch: MatMut<'_, f64>,
) -> Result<(), crate::GprError> {
    let mut nested = kernel.nested_buffers(out.nrows(), out.ncols());
    kernel.apply_from_ard_cache::<M>(cache.0.view(), x, out, uplo, scratch, &mut nested)
}

/// [`CompiledKernel::grad_points`](crate::kernel::CompiledKernel::grad_points)
/// for an ARD tree, read from the `(Δx_d)²` cache.
///
/// # Errors
///
/// Same as [`apply_from_ard_cache`], or an index error for `param_idx`.
#[cfg(feature = "bench-internals")]
pub fn grad_from_ard_cache<M: crate::KernelMath>(
    kernel: &crate::kernel::CompiledKernel,
    cache: &ArdCache,
    x: MatRef<'_, f64>,
    d_k: MatMut<'_, f64>,
    param_idx: usize,
    uplo: crate::kernel::Triangle,
    scratch: MatMut<'_, f64>,
) -> Result<(), crate::GprError> {
    let mut nested = kernel.nested_buffers(d_k.nrows(), d_k.ncols());
    kernel.grad_from_ard_cache::<M>(
        cache.0.view(),
        x,
        d_k,
        param_idx,
        uplo,
        scratch,
        &mut nested,
    )
}

#[cfg(feature = "insert-stages")]
pub use crate::gpr::take_insert_stages;

/// Zeroes the counters of [`objective_call_counts`].
#[cfg(feature = "bench-internals")]
pub fn reset_objective_call_counts() {
    crate::objective::call_counts::reset();
}

/// `(value-only calls, joint value-and-gradient calls)` the `Gpr` and `Sgpr`
/// fit objectives received since [`reset_objective_call_counts`]. A
/// gradient-only call counts as joint. The counters are global: read them
/// after a single fit, with no other fit running.
#[cfg(feature = "bench-internals")]
pub fn objective_call_counts() -> (u64, u64) {
    crate::objective::call_counts::read()
}
