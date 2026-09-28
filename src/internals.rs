//! Measurement hooks for this repository's benches and `compare/perf`.
//!
//! Present only with the non-default features `bench-internals` or
//! `insert-stages`. Nothing here is part of the model API.

#[cfg(feature = "bench-internals")]
use faer::{Mat, MatMut, MatRef};

/// Writes the full pairwise squared Euclidean distances of the rows of `x`
/// with the library's parallel fill (the `kernel_rbf` bench input).
#[cfg(feature = "bench-internals")]
pub fn fill_pairwise_sq_euclidean(x: MatRef<'_, f64>, dist: MatMut<'_, f64>) {
    crate::kernel::fill_squared_euclidean(x, dist, &mut []);
}

/// Writes the raw `(Δx_d)²` cache (`n × (n·d)`, dimension `k` in columns
/// `[k n, (k+1) n)`, lower triangle of each block) that ARD fits read.
#[cfg(feature = "bench-internals")]
pub fn fill_ard_squared_diff(x: MatRef<'_, f64>, cache: MatMut<'_, f64>, scratch: &mut [Mat<f64>]) {
    crate::kernel::fill_ard_squared_diff(x, cache, scratch);
}

/// [`CompiledKernel::apply_points`](crate::kernel::CompiledKernel::apply_points)
/// for an ARD tree, read from the `(Δx_d)²` cache that
/// [`fill_ard_squared_diff`] writes.
///
/// # Errors
///
/// Same as `apply_points`, or a shape error when `cache` is not `n × (n·d)`.
#[cfg(feature = "bench-internals")]
pub fn apply_from_ard_cache<M: crate::KernelMath>(
    kernel: &crate::kernel::CompiledKernel,
    cache: MatRef<'_, f64>,
    x: MatRef<'_, f64>,
    out: MatMut<'_, f64>,
    uplo: crate::kernel::Triangle,
    scratch: MatMut<'_, f64>,
) -> Result<(), crate::GprError> {
    kernel.apply_from_ard_cache::<M>(cache, x, out, uplo, scratch)
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
    cache: MatRef<'_, f64>,
    x: MatRef<'_, f64>,
    d_k: MatMut<'_, f64>,
    param_idx: usize,
    uplo: crate::kernel::Triangle,
    scratch: MatMut<'_, f64>,
) -> Result<(), crate::GprError> {
    kernel.grad_from_ard_cache::<M>(cache, x, d_k, param_idx, uplo, scratch)
}

#[cfg(feature = "insert-stages")]
pub use crate::gpr::take_insert_stages;
