//! Column-major SIMD helpers for squared distance, isotropic RBF, and ARD RBF.
//!
//! Uses [`wide::f64x4`]. When a view is not unit row-stride, callers keep the
//! scalar path. `wide::exp` may differ from scalar `f64::exp` by a few ULP.
//! ARD caches store raw `(Δx_d)²` as packed lower triangles
//! ([`super::dist::ArdSqDiff`]); each cached column holds rows `col..n`.

use super::dist::{ArdSqDiff, col_chunk, worker_count};
use super::{Triangle, finite_dist, require_same_shape, require_square_pair};
use crate::error::GprError;
use crate::math::{FastApprox, KernelMath, MathOps, f64x4_all_finite};
use faer::reborrow::ReborrowMut;
use faer::{MatMut, MatRef};
use rayon::prelude::*;
use wide::f64x4;

const LANES: usize = 4;

const _: () = assert!(std::mem::size_of::<f64x4>() == 32);

fn load4(src: &[f64], i: usize) -> f64x4 {
    f64x4::new([src[i], src[i + 1], src[i + 2], src[i + 3]])
}

fn store4(dest: &mut [f64], i: usize, v: f64x4) {
    let a = v.to_array();
    dest[i] = a[0];
    dest[i + 1] = a[1];
    dest[i + 2] = a[2];
    dest[i + 3] = a[3];
}

fn all_finite4(v: f64x4) -> bool {
    let a = v.to_array();
    a[0].is_finite() && a[1].is_finite() && a[2].is_finite() && a[3].is_finite()
}

fn col_slice(mat: MatRef<'_, f64>, col: usize) -> Option<&[f64]> {
    mat.col(col).try_as_col_major().map(|c| c.as_slice())
}

fn col_slice_mut(mat: MatMut<'_, f64>, col: usize) -> Option<&mut [f64]> {
    mat.col_mut(col)
        .try_as_col_major_mut()
        .map(|c| c.as_slice_mut())
}

fn unit_row_stride(mat: MatRef<'_, f64>) -> bool {
    mat.ncols() == 0 || col_slice(mat, 0).is_some()
}

/// Adds `(x[i] - x0)²` into `acc[i]` with `f64x4` lanes.
pub(crate) fn add_squared_diff(x: &[f64], x0: f64, acc: &mut [f64]) {
    debug_assert_eq!(x.len(), acc.len());
    let x0v = f64x4::new([x0; LANES]);
    let mut i = 0;
    while i + LANES <= x.len() {
        let xv = load4(x, i);
        let av = load4(acc, i);
        let d = xv - x0v;
        store4(acc, i, av + d * d);
        i += LANES;
    }
    while i < x.len() {
        let d = x[i] - x0;
        acc[i] += d * d;
        i += 1;
    }
}

fn load4_raw(src: &[f64], i: usize) -> f64x4 {
    let mut v = f64x4::ZERO;
    // SAFETY: caller keeps `i + 4 <= src.len()`. `f64x4` is four contiguous lanes;
    // the source is only required to be 8-byte aligned.
    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr().add(i), (&mut v as *mut f64x4).cast::<f64>(), 4);
    }
    v
}

fn store4_raw(dest: &mut [f64], i: usize, v: f64x4) {
    // SAFETY: caller keeps `i + 4 <= dest.len()`. `f64x4` is four contiguous lanes.
    unsafe {
        std::ptr::copy_nonoverlapping(
            (&v as *const f64x4).cast::<f64>(),
            dest.as_mut_ptr().add(i),
            4,
        );
    }
}

fn rbf_exp_slice<M: KernelMath>(
    dist: &[f64],
    out: &mut [f64],
    inv_two_ell_sq: f64,
) -> Result<(), GprError> {
    if M::ACCURATE {
        rbf_exp_slice_lanes::<M>(dist, out, inv_two_ell_sq)
    } else {
        rbf_exp_slice_fast(dist, out, inv_two_ell_sq)
    }
}

fn rbf_exp_slice_lanes<M: KernelMath>(
    dist: &[f64],
    out: &mut [f64],
    inv_two_ell_sq: f64,
) -> Result<(), GprError> {
    debug_assert_eq!(dist.len(), out.len());
    let scale = f64x4::new([-inv_two_ell_sq; LANES]);
    let mut i = 0;
    while i + LANES <= dist.len() {
        let d = load4(dist, i);
        if !all_finite4(d) {
            return Err(GprError::NonFiniteInput);
        }
        store4(out, i, M::exp_f64x4(d * scale));
        i += LANES;
    }
    while i < dist.len() {
        let d = finite_dist(dist[i])?;
        out[i] = M::exp(-d * inv_two_ell_sq);
        i += 1;
    }
    Ok(())
}

fn rbf_exp_slice_fast(dist: &[f64], out: &mut [f64], inv_two_ell_sq: f64) -> Result<(), GprError> {
    debug_assert_eq!(dist.len(), out.len());
    let scale = f64x4::splat(-inv_two_ell_sq);
    let mut i = 0;
    while i + LANES <= dist.len() {
        let d = load4_raw(dist, i);
        if !f64x4_all_finite(d) {
            return Err(GprError::NonFiniteInput);
        }
        store4_raw(out, i, FastApprox::exp_f64x4(d * scale));
        i += LANES;
    }
    while i < dist.len() {
        let d = finite_dist(dist[i])?;
        out[i] = FastApprox::exp(-d * inv_two_ell_sq);
        i += 1;
    }
    Ok(())
}

fn rbf_grad_slice<M: KernelMath>(
    dist: &[f64],
    out: &mut [f64],
    inv_two_ell_sq: f64,
    inv_ell_sq: f64,
) -> Result<(), GprError> {
    if M::ACCURATE {
        rbf_grad_slice_lanes::<M>(dist, out, inv_two_ell_sq, inv_ell_sq)
    } else {
        rbf_grad_slice_fast(dist, out, inv_two_ell_sq, inv_ell_sq)
    }
}

fn rbf_grad_slice_lanes<M: KernelMath>(
    dist: &[f64],
    out: &mut [f64],
    inv_two_ell_sq: f64,
    inv_ell_sq: f64,
) -> Result<(), GprError> {
    debug_assert_eq!(dist.len(), out.len());
    let scale = f64x4::new([-inv_two_ell_sq; LANES]);
    let inv = f64x4::new([inv_ell_sq; LANES]);
    let mut i = 0;
    while i + LANES <= dist.len() {
        let d = load4(dist, i);
        if !all_finite4(d) {
            return Err(GprError::NonFiniteInput);
        }
        let dk = M::d1_f64x4(d * scale);
        store4(out, i, dk * d * inv);
        i += LANES;
    }
    while i < dist.len() {
        let d = finite_dist(dist[i])?;
        let dk = M::jet(-d * inv_two_ell_sq).d1;
        out[i] = dk * d * inv_ell_sq;
        i += 1;
    }
    Ok(())
}

fn rbf_grad_slice_fast(
    dist: &[f64],
    out: &mut [f64],
    inv_two_ell_sq: f64,
    inv_ell_sq: f64,
) -> Result<(), GprError> {
    debug_assert_eq!(dist.len(), out.len());
    let scale = f64x4::splat(-inv_two_ell_sq);
    let inv = f64x4::splat(inv_ell_sq);
    let mut i = 0;
    while i + LANES <= dist.len() {
        let d = load4_raw(dist, i);
        if !f64x4_all_finite(d) {
            return Err(GprError::NonFiniteInput);
        }
        let dk = FastApprox::d1_f64x4(d * scale);
        store4_raw(out, i, dk * d * inv);
        i += LANES;
    }
    while i < dist.len() {
        let d = finite_dist(dist[i])?;
        let dk = FastApprox::jet(-d * inv_two_ell_sq).d1;
        out[i] = dk * d * inv_ell_sq;
        i += 1;
    }
    Ok(())
}

/// Fills a lower-triangle distance chunk when `x` and `dist` are column-major.
///
/// Returns `false` if a view is not unit row-stride so the caller can use
/// the scalar loop.
pub(crate) fn try_fill_lower_chunk(
    x: MatRef<'_, f64>,
    mut dist_chunk: MatMut<'_, f64>,
    chunk_idx: usize,
    n_chunks: usize,
) -> bool {
    if !unit_row_stride(x) {
        return false;
    }
    let n = x.nrows();
    let d = x.ncols();
    let (start, len) = col_chunk(n, chunk_idx, n_chunks);
    if len > 0 && col_slice_mut(dist_chunk.rb_mut(), 0).is_none() {
        return false;
    }
    for local in 0..len {
        let col = start + local;
        let Some(dest) = col_slice_mut(dist_chunk.rb_mut(), local) else {
            return false;
        };
        let dest = &mut dest[col..];
        dest.fill(0.0);
        for dim in 0..d {
            let Some(xdim) = col_slice(x, dim) else {
                return false;
            };
            add_squared_diff(&xdim[col..], xdim[col], dest);
        }
    }
    true
}

/// Fills a rectangular train–test distance chunk when views are column-major.
pub(crate) fn try_fill_cross_chunk(
    x_train: MatRef<'_, f64>,
    x_test: MatRef<'_, f64>,
    mut dist_chunk: MatMut<'_, f64>,
    chunk_idx: usize,
    n_chunks: usize,
) -> bool {
    if !unit_row_stride(x_train) || !unit_row_stride(x_test) {
        return false;
    }
    let n = x_train.nrows();
    let d = x_train.ncols();
    let m = x_test.nrows();
    let (start, len) = col_chunk(m, chunk_idx, n_chunks);
    if len > 0 && col_slice_mut(dist_chunk.rb_mut(), 0).is_none() {
        return false;
    }
    for local in 0..len {
        let col = start + local;
        let Some(dest) = col_slice_mut(dist_chunk.rb_mut(), local) else {
            return false;
        };
        dest.fill(0.0);
        for dim in 0..d {
            let Some(x_tr) = col_slice(x_train, dim) else {
                return false;
            };
            let Some(x_te) = col_slice(x_test, dim) else {
                return false;
            };
            add_squared_diff(x_tr, x_te[col], dest);
        }
        debug_assert_eq!(dest.len(), n);
    }
    true
}

struct ColWindow {
    dist_col: usize,
    out_col: usize,
    row_start: usize,
    row_end: usize,
}

fn map_column_range(
    dist: MatRef<'_, f64>,
    mut out: MatMut<'_, f64>,
    window: ColWindow,
    f: impl Fn(&[f64], &mut [f64]) -> Result<(), GprError>,
) -> Result<(), GprError> {
    let Some(src) = col_slice(dist, window.dist_col) else {
        return Err(GprError::UnsupportedKernelOperation {
            reason: "expected unit row-stride for SIMD RBF".to_owned(),
        });
    };
    let Some(dest) = col_slice_mut(out.rb_mut(), window.out_col) else {
        return Err(GprError::UnsupportedKernelOperation {
            reason: "expected unit row-stride for SIMD RBF".to_owned(),
        });
    };
    f(
        &src[window.row_start..window.row_end],
        &mut dest[window.row_start..window.row_end],
    )
}

fn apply_rbf_range<M: KernelMath>(
    dist: MatRef<'_, f64>,
    out: MatMut<'_, f64>,
    window: ColWindow,
    inv_two_ell_sq: f64,
) -> Result<(), GprError> {
    map_column_range(dist, out, window, |src, dest| {
        rbf_exp_slice::<M>(src, dest, inv_two_ell_sq)
    })
}

fn grad_rbf_range<M: KernelMath>(
    dist: MatRef<'_, f64>,
    out: MatMut<'_, f64>,
    window: ColWindow,
    inv_two_ell_sq: f64,
    inv_ell_sq: f64,
) -> Result<(), GprError> {
    map_column_range(dist, out, window, |src, dest| {
        rbf_grad_slice::<M>(src, dest, inv_two_ell_sq, inv_ell_sq)
    })
}

fn rbf_lower_parallel<M: KernelMath>(
    dist: MatRef<'_, f64>,
    out: MatMut<'_, f64>,
    inv_two_ell_sq: f64,
    inv_ell_sq: Option<f64>,
) -> Result<(), GprError> {
    let n = dist.nrows();
    let n_parts = worker_count();
    out.par_col_partition_mut(n_parts)
        .enumerate()
        .try_for_each(|(chunk_idx, mut part)| {
            let (start, len) = col_chunk(n, chunk_idx, n_parts);
            for local in 0..len {
                let col = start + local;
                match inv_ell_sq {
                    None => apply_rbf_range::<M>(
                        dist,
                        part.rb_mut(),
                        ColWindow {
                            dist_col: col,
                            out_col: local,
                            row_start: col,
                            row_end: n,
                        },
                        inv_two_ell_sq,
                    )?,
                    Some(inv_ell) => grad_rbf_range::<M>(
                        dist,
                        part.rb_mut(),
                        ColWindow {
                            dist_col: col,
                            out_col: local,
                            row_start: col,
                            row_end: n,
                        },
                        inv_two_ell_sq,
                        inv_ell,
                    )?,
                }
            }
            Ok::<(), GprError>(())
        })
}

/// Writes isotropic RBF `exp(-d / (2ℓ²))` when `dist`/`out` are column-major.
///
/// Returns `Ok(false)` if SIMD cannot run so the caller uses the scalar
/// [`super::write_triangle`] path.
pub(crate) fn try_apply_rbf<M: KernelMath>(
    dist: MatRef<'_, f64>,
    mut out: MatMut<'_, f64>,
    uplo: Triangle,
    inv_two_ell_sq: f64,
) -> Result<bool, GprError> {
    let n = require_square_pair(dist, out.as_ref())?;
    if !unit_row_stride(dist) || !unit_row_stride(out.as_ref()) {
        return Ok(false);
    }
    match uplo {
        Triangle::Lower => rbf_lower_parallel::<M>(dist, out, inv_two_ell_sq, None)?,
        Triangle::Full => {
            for col in 0..n {
                apply_rbf_range::<M>(
                    dist,
                    out.rb_mut(),
                    ColWindow {
                        dist_col: col,
                        out_col: col,
                        row_start: 0,
                        row_end: n,
                    },
                    inv_two_ell_sq,
                )?;
            }
        }
        Triangle::Upper => {
            for col in 0..n {
                apply_rbf_range::<M>(
                    dist,
                    out.rb_mut(),
                    ColWindow {
                        dist_col: col,
                        out_col: col,
                        row_start: 0,
                        row_end: col + 1,
                    },
                    inv_two_ell_sq,
                )?;
            }
        }
    }
    Ok(true)
}

/// Writes rectangular RBF `k(X, X*)` when views are column-major.
pub(crate) fn try_apply_rbf_cross<M: KernelMath>(
    dist: MatRef<'_, f64>,
    out: MatMut<'_, f64>,
    inv_two_ell_sq: f64,
) -> Result<bool, GprError> {
    require_same_shape(dist, out.as_ref())?;
    if !unit_row_stride(dist) || !unit_row_stride(out.as_ref()) {
        return Ok(false);
    }
    let n = dist.nrows();
    let m = dist.ncols();
    if m == 1 {
        apply_rbf_range::<M>(
            dist,
            out,
            ColWindow {
                dist_col: 0,
                out_col: 0,
                row_start: 0,
                row_end: n,
            },
            inv_two_ell_sq,
        )?;
        return Ok(true);
    }
    let n_parts = worker_count();
    out.par_col_partition_mut(n_parts)
        .enumerate()
        .try_for_each(|(chunk_idx, mut part)| {
            let (start, len) = col_chunk(m, chunk_idx, n_parts);
            for local in 0..len {
                let col = start + local;
                apply_rbf_range::<M>(
                    dist,
                    part.rb_mut(),
                    ColWindow {
                        dist_col: col,
                        out_col: local,
                        row_start: 0,
                        row_end: n,
                    },
                    inv_two_ell_sq,
                )?;
            }
            Ok::<(), GprError>(())
        })?;
    Ok(true)
}

/// Writes rectangular ARD RBF `k(X, X*)` when views are column-major.
pub(crate) fn try_apply_rbf_ard_cross<M: KernelMath>(
    x: MatRef<'_, f64>,
    xs: MatRef<'_, f64>,
    out: MatMut<'_, f64>,
    inv_ell_sq: &[f64],
) -> Result<bool, GprError> {
    let n = x.nrows();
    let m = xs.nrows();
    let d = inv_ell_sq.len();
    if x.ncols() != d || xs.ncols() != d || out.nrows() != n || out.ncols() != m {
        return Ok(false);
    }
    if !unit_row_stride(x) || !unit_row_stride(xs) || !unit_row_stride(out.as_ref()) {
        return Ok(false);
    }
    if m == 1 {
        map_ard_cross_column::<M>(x, xs, out, 0, 0, inv_ell_sq)?;
        return Ok(true);
    }
    let n_parts = worker_count();
    out.par_col_partition_mut(n_parts)
        .enumerate()
        .try_for_each(|(chunk_idx, mut part)| {
            let (start, len) = col_chunk(m, chunk_idx, n_parts);
            for local in 0..len {
                let col = start + local;
                map_ard_cross_column::<M>(x, xs, part.rb_mut(), local, col, inv_ell_sq)?;
            }
            Ok::<(), GprError>(())
        })?;
    Ok(true)
}

fn map_ard_cross_column<M: KernelMath>(
    x: MatRef<'_, f64>,
    xs: MatRef<'_, f64>,
    mut out: MatMut<'_, f64>,
    out_col: usize,
    query_col: usize,
    inv_ell_sq: &[f64],
) -> Result<(), GprError> {
    let Some(dest) = col_slice_mut(out.rb_mut(), out_col) else {
        return Err(GprError::UnsupportedKernelOperation {
            reason: "expected unit row-stride for SIMD ARD".to_owned(),
        });
    };
    dest.fill(0.0);
    for (dim, &w) in inv_ell_sq.iter().enumerate() {
        let Some(xdim) = col_slice(x, dim) else {
            return Err(GprError::UnsupportedKernelOperation {
                reason: "expected unit row-stride for SIMD ARD".to_owned(),
            });
        };
        let Some(xs_dim) = col_slice(xs, dim) else {
            return Err(GprError::UnsupportedKernelOperation {
                reason: "expected unit row-stride for SIMD ARD".to_owned(),
            });
        };
        add_squared_diff_scaled(xdim, xs_dim[query_col], w, dest);
    }
    rbf_exp_in_place::<M>(dest, 0.5)
}

/// Writes `∂k/∂θ = k · d / ℓ²` when views are column-major.
pub(crate) fn try_grad_rbf<M: KernelMath>(
    dist: MatRef<'_, f64>,
    mut d_k: MatMut<'_, f64>,
    uplo: Triangle,
    inv_two_ell_sq: f64,
    inv_ell_sq: f64,
) -> Result<bool, GprError> {
    let n = require_square_pair(dist, d_k.as_ref())?;
    if !unit_row_stride(dist) || !unit_row_stride(d_k.as_ref()) {
        return Ok(false);
    }
    match uplo {
        Triangle::Lower => {
            rbf_lower_parallel::<M>(dist, d_k, inv_two_ell_sq, Some(inv_ell_sq))?;
        }
        Triangle::Full => {
            for col in 0..n {
                grad_rbf_range::<M>(
                    dist,
                    d_k.rb_mut(),
                    ColWindow {
                        dist_col: col,
                        out_col: col,
                        row_start: 0,
                        row_end: n,
                    },
                    inv_two_ell_sq,
                    inv_ell_sq,
                )?;
            }
        }
        Triangle::Upper => {
            for col in 0..n {
                grad_rbf_range::<M>(
                    dist,
                    d_k.rb_mut(),
                    ColWindow {
                        dist_col: col,
                        out_col: col,
                        row_start: 0,
                        row_end: col + 1,
                    },
                    inv_two_ell_sq,
                    inv_ell_sq,
                )?;
            }
        }
    }
    Ok(true)
}

/// Offset of row `row_start` in the cached column `col`, which stores rows
/// `col..n` only. Rows above the diagonal are not cached.
fn cached_rows_offset(row_start: usize, col: usize) -> Result<usize, GprError> {
    row_start
        .checked_sub(col)
        .ok_or_else(|| GprError::UnsupportedKernelOperation {
            reason: "the ARD cache holds the lower triangle only".to_owned(),
        })
}

fn scale_add(src: &[f64], scale: f64, acc: &mut [f64]) -> Result<(), GprError> {
    debug_assert_eq!(src.len(), acc.len());
    let sv = f64x4::new([scale; LANES]);
    let mut i = 0;
    while i + LANES <= src.len() {
        let s = load4(src, i);
        if !all_finite4(s) {
            return Err(GprError::NonFiniteInput);
        }
        store4(acc, i, load4(acc, i) + s * sv);
        i += LANES;
    }
    while i < src.len() {
        let v = finite_dist(src[i])?;
        acc[i] += v * scale;
        i += 1;
    }
    Ok(())
}

/// Adds `scale · (x[i] - x0)²` into `acc[i]` with `f64x4` lanes.
pub(crate) fn add_squared_diff_scaled(x: &[f64], x0: f64, scale: f64, acc: &mut [f64]) {
    debug_assert_eq!(x.len(), acc.len());
    let x0v = f64x4::new([x0; LANES]);
    let sv = f64x4::new([scale; LANES]);
    let mut i = 0;
    while i + LANES <= x.len() {
        let d = load4(x, i) - x0v;
        store4(acc, i, load4(acc, i) + d * d * sv);
        i += LANES;
    }
    while i < x.len() {
        let d = x[i] - x0;
        acc[i] += d * d * scale;
        i += 1;
    }
}

fn rbf_exp_in_place<M: KernelMath>(buf: &mut [f64], inv_two: f64) -> Result<(), GprError> {
    let scale = f64x4::new([-inv_two; LANES]);
    let mut i = 0;
    while i + LANES <= buf.len() {
        let d = load4(buf, i);
        if !all_finite4(d) {
            return Err(GprError::NonFiniteInput);
        }
        store4(buf, i, M::exp_f64x4(d * scale));
        i += LANES;
    }
    while i < buf.len() {
        let d = finite_dist(buf[i])?;
        buf[i] = M::exp(-d * inv_two);
        i += 1;
    }
    Ok(())
}

fn rbf_ard_grad_from_points<M: KernelMath>(
    r2: &mut [f64],
    xdim: &[f64],
    x0: f64,
    inv_dim: f64,
) -> Result<(), GprError> {
    debug_assert_eq!(r2.len(), xdim.len());
    let half = f64x4::new([-0.5; LANES]);
    let inv = f64x4::new([inv_dim; LANES]);
    let x0v = f64x4::new([x0; LANES]);
    let mut i = 0;
    while i + LANES <= r2.len() {
        let d = load4(r2, i);
        if !all_finite4(d) {
            return Err(GprError::NonFiniteInput);
        }
        let delta = load4(xdim, i) - x0v;
        let dk = M::d1_f64x4(d * half);
        store4(r2, i, dk * delta * delta * inv);
        i += LANES;
    }
    while i < r2.len() {
        let d = finite_dist(r2[i])?;
        let delta = xdim[i] - x0;
        r2[i] = M::jet(-0.5 * d).d1 * delta * delta * inv_dim;
        i += 1;
    }
    Ok(())
}

fn rbf_ard_grad_in_place<M: KernelMath>(
    r2: &mut [f64],
    dim_sq: &[f64],
    inv_dim: f64,
) -> Result<(), GprError> {
    debug_assert_eq!(r2.len(), dim_sq.len());
    let half = f64x4::new([-0.5; LANES]);
    let inv = f64x4::new([inv_dim; LANES]);
    let mut i = 0;
    while i + LANES <= r2.len() {
        let d = load4(r2, i);
        let sq = load4(dim_sq, i);
        if !all_finite4(d) || !all_finite4(sq) {
            return Err(GprError::NonFiniteInput);
        }
        let dk = M::d1_f64x4(d * half);
        store4(r2, i, dk * sq * inv);
        i += LANES;
    }
    while i < r2.len() {
        let d = finite_dist(r2[i])?;
        let sq = finite_dist(dim_sq[i])?;
        r2[i] = M::jet(-0.5 * d).d1 * sq * inv_dim;
        i += 1;
    }
    Ok(())
}

fn accumulate_ard_r2(
    cache: Option<ArdSqDiff<'_, f64>>,
    x: Option<MatRef<'_, f64>>,
    inv_ell_sq: &[f64],
    pair_col: usize,
    dest: &mut [f64],
    row_start: usize,
) -> Result<(), GprError> {
    dest.fill(0.0);
    let n_rows = dest.len();
    if let Some(cache) = cache {
        let offset = cached_rows_offset(row_start, pair_col)?;
        for (dim, &w) in inv_ell_sq.iter().enumerate() {
            let src = cache.column(dim, pair_col);
            scale_add(&src[offset..offset + n_rows], w, dest)?;
        }
        return Ok(());
    }
    let x = x.ok_or_else(|| GprError::UnsupportedKernelOperation {
        reason: "ARD SIMD needs coordinates or a squared-diff cache".to_owned(),
    })?;
    for (dim, &w) in inv_ell_sq.iter().enumerate() {
        let Some(xdim) = col_slice(x, dim) else {
            return Err(GprError::UnsupportedKernelOperation {
                reason: "expected unit row-stride for SIMD ARD".to_owned(),
            });
        };
        add_squared_diff_scaled(
            &xdim[row_start..row_start + n_rows],
            xdim[pair_col],
            w,
            dest,
        );
    }
    Ok(())
}

fn map_ard_column<M: KernelMath>(
    cache: Option<ArdSqDiff<'_, f64>>,
    x: Option<MatRef<'_, f64>>,
    mut out: MatMut<'_, f64>,
    window: ColWindow,
    inv_ell_sq: &[f64],
    param_idx: Option<usize>,
) -> Result<(), GprError> {
    let Some(dest) = col_slice_mut(out.rb_mut(), window.out_col) else {
        return Err(GprError::UnsupportedKernelOperation {
            reason: "expected unit row-stride for SIMD ARD".to_owned(),
        });
    };
    let dest = &mut dest[window.row_start..window.row_end];
    accumulate_ard_r2(
        cache,
        x,
        inv_ell_sq,
        window.dist_col,
        dest,
        window.row_start,
    )?;
    match param_idx {
        None => rbf_exp_in_place::<M>(dest, 0.5),
        Some(dim) => {
            if let Some(cache) = cache {
                let offset = cached_rows_offset(window.row_start, window.dist_col)?;
                let src = cache.column(dim, window.dist_col);
                let len = window.row_end - window.row_start;
                rbf_ard_grad_in_place::<M>(dest, &src[offset..offset + len], inv_ell_sq[dim])
            } else {
                let x = x.ok_or_else(|| GprError::UnsupportedKernelOperation {
                    reason: "ARD SIMD needs coordinates or a squared-diff cache".to_owned(),
                })?;
                let Some(xdim) = col_slice(x, dim) else {
                    return Err(GprError::UnsupportedKernelOperation {
                        reason: "expected unit row-stride for SIMD ARD".to_owned(),
                    });
                };
                rbf_ard_grad_from_points::<M>(
                    dest,
                    &xdim[window.row_start..window.row_end],
                    xdim[window.dist_col],
                    inv_ell_sq[dim],
                )
            }
        }
    }
}

fn rbf_ard_lower_parallel<M: KernelMath>(
    cache: Option<ArdSqDiff<'_, f64>>,
    x: Option<MatRef<'_, f64>>,
    out: MatMut<'_, f64>,
    inv_ell_sq: &[f64],
    param_idx: Option<usize>,
) -> Result<(), GprError> {
    let n = out.nrows();
    let n_parts = worker_count();
    out.par_col_partition_mut(n_parts)
        .enumerate()
        .try_for_each(|(chunk_idx, mut part)| {
            let (start, len) = col_chunk(n, chunk_idx, n_parts);
            for local in 0..len {
                let col = start + local;
                map_ard_column::<M>(
                    cache,
                    x,
                    part.rb_mut(),
                    ColWindow {
                        dist_col: col,
                        out_col: local,
                        row_start: col,
                        row_end: n,
                    },
                    inv_ell_sq,
                    param_idx,
                )?;
            }
            Ok::<(), GprError>(())
        })
}

fn rbf_ard_serial_uplo<M: KernelMath>(
    cache: Option<ArdSqDiff<'_, f64>>,
    x: Option<MatRef<'_, f64>>,
    mut out: MatMut<'_, f64>,
    uplo: Triangle,
    inv_ell_sq: &[f64],
    param_idx: Option<usize>,
) -> Result<(), GprError> {
    let n = out.nrows();
    match uplo {
        Triangle::Lower => rbf_ard_lower_parallel::<M>(cache, x, out, inv_ell_sq, param_idx),
        Triangle::Full => {
            for col in 0..n {
                map_ard_column::<M>(
                    cache,
                    x,
                    out.rb_mut(),
                    ColWindow {
                        dist_col: col,
                        out_col: col,
                        row_start: 0,
                        row_end: n,
                    },
                    inv_ell_sq,
                    param_idx,
                )?;
            }
            Ok(())
        }
        Triangle::Upper => {
            for col in 0..n {
                map_ard_column::<M>(
                    cache,
                    x,
                    out.rb_mut(),
                    ColWindow {
                        dist_col: col,
                        out_col: col,
                        row_start: 0,
                        row_end: col + 1,
                    },
                    inv_ell_sq,
                    param_idx,
                )?;
            }
            Ok(())
        }
    }
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

/// Writes ARD RBF from a raw `(Δx_d)²` cache when views are column-major.
pub(crate) fn try_apply_rbf_ard_cache<M: KernelMath>(
    cache: ArdSqDiff<'_, f64>,
    mut out: MatMut<'_, f64>,
    uplo: Triangle,
    inv_ell_sq: &[f64],
) -> Result<bool, GprError> {
    let n = out.nrows();
    if out.ncols() != n {
        return Ok(false);
    }
    super::dist::require_ard_sq_diff_shape(cache, n, inv_ell_sq.len())?;
    // The cache stores the lower triangle; the scalar path reads any pair.
    if uplo != Triangle::Lower || !unit_row_stride(out.as_ref()) {
        return Ok(false);
    }
    rbf_ard_serial_uplo::<M>(Some(cache), None, out.rb_mut(), uplo, inv_ell_sq, None)?;
    Ok(true)
}

/// Writes ARD RBF from coordinates when views are column-major.
pub(crate) fn try_apply_rbf_ard_points<M: KernelMath>(
    x: MatRef<'_, f64>,
    mut out: MatMut<'_, f64>,
    uplo: Triangle,
    inv_ell_sq: &[f64],
) -> Result<bool, GprError> {
    let n = out.nrows();
    if out.ncols() != n || x.nrows() != n || x.ncols() != inv_ell_sq.len() {
        return Ok(false);
    }
    if !unit_row_stride(x) || !unit_row_stride(out.as_ref()) {
        return Ok(false);
    }
    rbf_ard_serial_uplo::<M>(None, Some(x), out.rb_mut(), uplo, inv_ell_sq, None)?;
    Ok(true)
}

/// Writes ARD RBF `∂k/∂θ_d` from a raw `(Δx_d)²` cache.
pub(crate) fn try_grad_rbf_ard_cache<M: KernelMath>(
    cache: ArdSqDiff<'_, f64>,
    mut d_k: MatMut<'_, f64>,
    uplo: Triangle,
    inv_ell_sq: &[f64],
    param_idx: usize,
) -> Result<bool, GprError> {
    let n = d_k.nrows();
    if d_k.ncols() != n || param_idx >= inv_ell_sq.len() {
        return Ok(false);
    }
    super::dist::require_ard_sq_diff_shape(cache, n, inv_ell_sq.len())?;
    // The cache stores the lower triangle; the scalar path reads any pair.
    if uplo != Triangle::Lower || !unit_row_stride(d_k.as_ref()) {
        return Ok(false);
    }
    rbf_ard_serial_uplo::<M>(
        Some(cache),
        None,
        d_k.rb_mut(),
        uplo,
        inv_ell_sq,
        Some(param_idx),
    )?;
    Ok(true)
}

/// Writes ARD RBF `∂k/∂θ_d` from coordinates.
pub(crate) fn try_grad_rbf_ard_points<M: KernelMath>(
    x: MatRef<'_, f64>,
    mut d_k: MatMut<'_, f64>,
    uplo: Triangle,
    inv_ell_sq: &[f64],
    param_idx: usize,
) -> Result<bool, GprError> {
    let n = d_k.nrows();
    if d_k.ncols() != n
        || x.nrows() != n
        || x.ncols() != inv_ell_sq.len()
        || param_idx >= inv_ell_sq.len()
    {
        return Ok(false);
    }
    if !unit_row_stride(x) || !unit_row_stride(d_k.as_ref()) {
        return Ok(false);
    }
    rbf_ard_serial_uplo::<M>(
        None,
        Some(x),
        d_k.rb_mut(),
        uplo,
        inv_ell_sq,
        Some(param_idx),
    )?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::{add_squared_diff, add_squared_diff_scaled, rbf_exp_slice, rbf_grad_slice};
    use crate::error::GprError;
    use crate::math::{Accurate, FastApprox, MathOps};

    const TOL: f64 = 1e-12;

    use crate::test_check::assert_close;

    #[test]
    fn add_squared_diff_matches_scalar() {
        let x: Vec<f64> = (0..11).map(|i| i as f64 * 0.3).collect();
        let mut acc = vec![1.0; 11];
        let mut expected = acc.clone();
        let x0 = 1.25;
        add_squared_diff(&x, x0, &mut acc);
        for i in 0..11 {
            let d = x[i] - x0;
            expected[i] += d * d;
            assert_close(acc[i], expected[i], TOL);
        }
    }

    #[test]
    fn add_squared_diff_scaled_matches_scalar() {
        let x: Vec<f64> = (0..11).map(|i| i as f64 * 0.3).collect();
        let mut acc = vec![0.25; 11];
        let mut expected = acc.clone();
        let x0 = 1.25;
        let w = 0.4;
        add_squared_diff_scaled(&x, x0, w, &mut acc);
        for i in 0..11 {
            let d = x[i] - x0;
            expected[i] += w * d * d;
            assert_close(acc[i], expected[i], TOL);
        }
    }

    #[test]
    fn rbf_exp_slice_matches_scalar_within_tol() {
        let dist: Vec<f64> = (0..10).map(|i| (i as f64) * 0.4).collect();
        let inv = 0.5;
        let mut out = vec![0.0; 10];
        rbf_exp_slice::<Accurate>(&dist, &mut out, inv).expect("finite");
        for i in 0..10 {
            let expected = (-dist[i] * inv).exp();
            assert_close(out[i], expected, TOL);
        }
    }

    #[test]
    fn rbf_grad_slice_matches_scalar_within_tol() {
        let dist: Vec<f64> = (0..9).map(|i| 0.2 + i as f64).collect();
        let inv_two = 0.25;
        let inv_ell = 0.5;
        let mut out = vec![0.0; 9];
        rbf_grad_slice::<Accurate>(&dist, &mut out, inv_two, inv_ell).expect("finite");
        for i in 0..9 {
            let k = (-dist[i] * inv_two).exp();
            assert_close(out[i], k * dist[i] * inv_ell, TOL);
        }
    }

    #[test]
    fn rbf_exp_slice_rejects_nan() {
        let dist = [0.0, 1.0, f64::NAN, 3.0, 4.0];
        let mut out = [0.0; 5];
        assert!(matches!(
            rbf_exp_slice::<Accurate>(&dist, &mut out, 0.5),
            Err(GprError::NonFiniteInput)
        ));
        assert!(matches!(
            rbf_exp_slice::<FastApprox>(&dist, &mut out, 0.5),
            Err(GprError::NonFiniteInput)
        ));
    }

    #[test]
    fn rbf_fast_slice_matches_scalar_polynomial() {
        let dist: Vec<f64> = (0..11).map(|i| (i as f64) * 0.35).collect();
        let inv_two = 0.5;
        let inv_ell = 1.0;
        let mut value = vec![0.0; dist.len()];
        let mut grad = vec![0.0; dist.len()];
        rbf_exp_slice::<FastApprox>(&dist, &mut value, inv_two).expect("finite");
        rbf_grad_slice::<FastApprox>(&dist, &mut grad, inv_two, inv_ell).expect("finite");
        for (i, &d) in dist.iter().enumerate() {
            let jet = FastApprox::jet(-d * inv_two);
            assert_eq!(value[i].to_bits(), jet.v.to_bits());
            assert_eq!(grad[i].to_bits(), (jet.d1 * d * inv_ell).to_bits());
        }
    }
}
