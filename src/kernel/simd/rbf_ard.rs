//! `f64x4` paths of the ARD RBF leaf: the value and `∂K/∂θ_d` of a square
//! output from coordinates or the packed `(Δx_d)²` cache, the rectangle, and
//! the rectangular `∂K/∂θ_d` from coordinates. The rectangular contraction
//! forms every lengthscale from one `exp`, then folds `Σ (W ∘ k) (Δ_d)²`
//! as one matrix product.
//!
//! ARD caches store raw `(Δx_d)²` as packed lower triangles
//! ([`crate::kernel::dist::ArdSqDiff`]); each cached column holds rows
//! `col..n`.

use super::{
    LANES, add_squared_diff_scaled, all_finite, all_valid_distances, col_slice_checked,
    col_slice_mut_checked, finite_slice, load4, rows_checked, store4, unit_row_stride,
};
use crate::error::GprError;
use crate::kernel::dist::{ArdSqDiff, col_chunk, par_lower_cols, worker_count};
use crate::kernel::sources::first_invalid_from;
use crate::kernel::{Triangle, finite_dist};
use crate::math::KernelMath;
use faer::linalg::matmul::matmul;
use faer::reborrow::ReborrowMut;
use faer::{Accum, MatMut, MatRef, Par};
use rayon::prelude::*;
use wide::f64x4;

/// Writes rectangular ARD RBF `k(X, X*)` when views are column-major.
pub(crate) fn try_apply_cross<M: KernelMath>(
    x: MatRef<'_, f64>,
    xs: MatRef<'_, f64>,
    mut out: MatMut<'_, f64>,
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
    let column = |query: usize, dest: &mut [f64]| {
        dest.fill(0.0);
        for (dim, &w) in inv_ell_sq.iter().enumerate() {
            let x0 = col_slice_checked(xs, dim)?[query];
            add_squared_diff_scaled(col_slice_checked(x, dim)?, x0, w, dest);
        }
        exp_half_in_place::<M>(dest)
    };
    if m == 1 {
        column(0, col_slice_mut_checked(out, 0)?)?;
        return Ok(true);
    }
    let n_parts = worker_count();
    out.rb_mut()
        .par_col_partition_mut(n_parts)
        .enumerate()
        .try_for_each(|(chunk_idx, mut part)| {
            let (start, len) = col_chunk(m, chunk_idx, n_parts);
            for local in 0..len {
                column(start + local, col_slice_mut_checked(part.rb_mut(), local)?)?;
            }
            Ok::<(), GprError>(())
        })?;
    Ok(true)
}

/// Writes rectangular ARD RBF `k` from raw `(Δ_d)²` blocks when `out` is
/// column-major. `block(dim)` is dimension `dim`'s column-major block with
/// `out.nrows()` rows; `out` column `j` reads block column `col0 + j`.
///
/// The blocks of an `f64` model are not checked when bound: each value is
/// checked as it is read here. Up to [`FUSED_DIMS`] dimensions, the loop
/// is software-pipelined over columns: one pass writes `exp(−r²/2)` of
/// column `j` while it reads, checks, and sums column `j + 1` into `r²`,
/// so the reads of the caller's values overlap the `exp` of the column
/// before.
///
/// # Errors
///
/// Returns [`GprError::InvalidDistance`] for a value that is not finite or
/// is negative, at its place in the caller's table, and
/// [`GprError::NonFiniteKernelValue`] when an `r²` overflows.
pub(crate) fn try_apply_cross_from_blocks<'b, M: KernelMath>(
    block: impl Fn(usize) -> Option<&'b [f64]> + Sync,
    col0: usize,
    mut out: MatMut<'_, f64>,
    inv_ell_sq: &[f64],
) -> Result<bool, GprError> {
    let n = out.nrows();
    let m = out.ncols();
    let d = inv_ell_sq.len();
    if !unit_row_stride(out.as_ref()) {
        return Ok(false);
    }
    let end = (col0 + m) * n;
    for dim in 0..d {
        match block(dim) {
            Some(b) if b.len() >= end => {}
            _ => return Ok(false),
        }
    }
    // Column `query` of dimension `dim`.
    let run = |dim: usize, query: usize| -> Result<&'b [f64], GprError> {
        let start = (col0 + query) * n;
        Ok(&block(dim).ok_or_else(changed_scalar)?[start..start + n])
    };
    // The first invalid value of column `query`.
    let invalid = |query: usize| -> GprError {
        for dim in 0..d {
            match run(dim, query) {
                Ok(src) if !all_valid_distances(src) => {
                    return first_invalid_from(src, n, col0 + query);
                }
                Ok(_) => {}
                Err(e) => return e,
            }
        }
        GprError::NonFiniteInput
    };
    let runs = |query: usize, dims: &mut [&'b [f64]; FUSED_DIMS]| -> Result<(), GprError> {
        for (dim, slot) in dims.iter_mut().enumerate().take(d) {
            *slot = run(dim, query)?;
        }
        Ok(())
    };
    let chunk = |start: usize, mut part: MatMut<'_, f64>| -> Result<(), GprError> {
        let len = part.ncols();
        if len == 0 {
            return Ok(());
        }
        if d > FUSED_DIMS {
            for local in 0..len {
                let query = start + local;
                let dest = col_slice_mut_checked(part.rb_mut(), local)?;
                dest.fill(0.0);
                for (dim, &w) in inv_ell_sq.iter().enumerate() {
                    let src = run(dim, query)?;
                    if !all_valid_distances(src) {
                        return Err(first_invalid_from(src, n, col0 + query));
                    }
                    scale_add_checked(src, w, dest);
                }
                exp_half_in_place::<M>(dest)?;
            }
            return Ok(());
        }
        let mut dims: [&'b [f64]; FUSED_DIMS] = [&[]; FUSED_DIMS];
        runs(start, &mut dims)?;
        if !weighted_sum(
            &dims[..d],
            inv_ell_sq,
            col_slice_mut_checked(part.rb_mut(), 0)?,
        ) {
            return Err(invalid(start));
        }
        for local in 0..len {
            let finite = if local + 1 < len {
                runs(start + local + 1, &mut dims)?;
                let (left, right) = part.rb_mut().split_at_col_mut(local + 1);
                let (valid, finite) = exp_half_then_sum::<M>(
                    col_slice_mut_checked(left, local)?,
                    &dims[..d],
                    inv_ell_sq,
                    col_slice_mut_checked(right, 0)?,
                );
                if !valid {
                    return Err(invalid(start + local + 1));
                }
                finite
            } else {
                exp_half_finite::<M>(col_slice_mut_checked(part.rb_mut(), local)?)
            };
            if !finite {
                return Err(GprError::NonFiniteKernelValue);
            }
        }
        Ok(())
    };
    if m == 1 {
        chunk(0, out)?;
        return Ok(true);
    }
    let n_parts = worker_count();
    out.rb_mut()
        .par_col_partition_mut(n_parts)
        .enumerate()
        .try_for_each(|(chunk_idx, part)| chunk(col_chunk(m, chunk_idx, n_parts).0, part))?;
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

/// `acc += scale · src`, checking that `src` is finite.
fn scale_add(src: &[f64], scale: f64, acc: &mut [f64]) -> Result<(), GprError> {
    debug_assert_eq!(src.len(), acc.len());
    let sv = f64x4::splat(scale);
    let mut i = 0;
    while i + LANES <= src.len() {
        let s = load4(src, i);
        if !all_finite(s) {
            return Err(GprError::NonFiniteInput);
        }
        store4(acc, i, load4(acc, i) + s * sv);
        i += LANES;
    }
    while i < src.len() {
        acc[i] += finite_dist(src[i])? * scale;
        i += 1;
    }
    Ok(())
}

/// Dimensions up to which [`try_apply_cross_from_blocks`] keeps the column
/// runs of every dimension at hand and pipelines its columns.
const FUSED_DIMS: usize = 16;

fn changed_scalar() -> GprError {
    GprError::UnsupportedKernelOperation {
        reason: "an ARD block changed scalar type".to_owned(),
    }
}

/// The validity of the supplied values one pass reads: `v · 0` stays `0`
/// unless a value is `NaN` or infinite, and the least value stays `≥ 0`
/// unless one is negative.
struct Validity {
    nonfinite: f64x4,
    least: f64x4,
}

impl Validity {
    fn new() -> Self {
        Self {
            nonfinite: f64x4::ZERO,
            least: f64x4::ZERO,
        }
    }

    #[inline(always)]
    fn read(&mut self, v: f64x4) -> f64x4 {
        self.nonfinite += v * f64x4::ZERO;
        self.least = self.least.fast_min(v);
        v
    }

    fn valid(&self) -> bool {
        // Exact, as [`super::all_valid_distances`]: `0` or `NaN`, nothing between.
        self.nonfinite.reduce_add() == 0.0 && self.least.to_array().iter().all(|&l| l >= 0.0)
    }
}

/// `dest = Σ_d w_d src_d` (`r²`) of supplied values; whether every value
/// read is valid (finite, not negative).
fn weighted_sum(srcs: &[&[f64]], w: &[f64], dest: &mut [f64]) -> bool {
    let mut validity = Validity::new();
    let mut i = 0;
    while i + LANES <= dest.len() {
        let mut r2 = f64x4::ZERO;
        for (src, &wd) in srcs.iter().zip(w) {
            r2 += validity.read(load4(src, i)) * f64x4::splat(wd);
        }
        store4(dest, i, r2);
        i += LANES;
    }
    let mut valid = validity.valid();
    while i < dest.len() {
        dest[i] = 0.0;
        for (src, &wd) in srcs.iter().zip(w) {
            valid &= super::super::sources::valid(src[i]);
            dest[i] += src[i] * wd;
        }
        i += 1;
    }
    valid
}

/// `cur = exp(−cur / 2)` and `next = Σ_d w_d src_d` in one pass, so the
/// reads of `srcs` overlap the `exp`. Returns whether every value of `srcs`
/// is valid and whether every `cur` was finite.
fn exp_half_then_sum<M: KernelMath>(
    cur: &mut [f64],
    srcs: &[&[f64]],
    w: &[f64],
    next: &mut [f64],
) -> (bool, bool) {
    let half = f64x4::splat(-0.5);
    let mut validity = Validity::new();
    // `r² · 0` stays `0` unless an `r²` is infinite.
    let mut overflow = f64x4::ZERO;
    let mut i = 0;
    while i + LANES <= cur.len() {
        let mut r2 = f64x4::ZERO;
        for (src, &wd) in srcs.iter().zip(w) {
            r2 += validity.read(load4(src, i)) * f64x4::splat(wd);
        }
        store4(next, i, r2);
        let c = load4(cur, i);
        overflow += c * f64x4::ZERO;
        store4(cur, i, M::exp_f64x4(c * half));
        i += LANES;
    }
    let mut valid = validity.valid();
    let mut tail = 0.0;
    while i < cur.len() {
        next[i] = 0.0;
        for (src, &wd) in srcs.iter().zip(w) {
            valid &= super::super::sources::valid(src[i]);
            next[i] += src[i] * wd;
        }
        tail += cur[i] * 0.0;
        cur[i] = M::exp(-cur[i] * 0.5);
        i += 1;
    }
    (valid, overflow.reduce_add() + tail == 0.0)
}

/// `cur = exp(−cur / 2)`; whether every `cur` was finite.
fn exp_half_finite<M: KernelMath>(cur: &mut [f64]) -> bool {
    let half = f64x4::splat(-0.5);
    let mut overflow = f64x4::ZERO;
    let mut i = 0;
    while i + LANES <= cur.len() {
        let c = load4(cur, i);
        overflow += c * f64x4::ZERO;
        store4(cur, i, M::exp_f64x4(c * half));
        i += LANES;
    }
    let mut tail = 0.0;
    while i < cur.len() {
        tail += cur[i] * 0.0;
        cur[i] = M::exp(-cur[i] * 0.5);
        i += 1;
    }
    overflow.reduce_add() + tail == 0.0
}

/// `acc += scale · src` for values already checked (finite, not
/// negative): supplied blocks. The sum is checked by the `exp` pass.
fn scale_add_checked(src: &[f64], scale: f64, acc: &mut [f64]) {
    debug_assert_eq!(src.len(), acc.len());
    let sv = f64x4::splat(scale);
    let mut i = 0;
    while i + LANES <= src.len() {
        store4(acc, i, load4(acc, i) + load4(src, i) * sv);
        i += LANES;
    }
    while i < src.len() {
        acc[i] += src[i] * scale;
        i += 1;
    }
}

/// `buf = exp(−buf / 2)`, checking that `buf` is finite.
fn exp_half_in_place<M: KernelMath>(buf: &mut [f64]) -> Result<(), GprError> {
    let scale = f64x4::splat(-0.5);
    let mut i = 0;
    while i + LANES <= buf.len() {
        let d = load4(buf, i);
        if !all_finite(d) {
            return Err(GprError::NonFiniteInput);
        }
        store4(buf, i, M::exp_f64x4(d * scale));
        i += LANES;
    }
    while i < buf.len() {
        buf[i] = M::exp(-finite_dist(buf[i])? * 0.5);
        i += 1;
    }
    Ok(())
}

/// `r2 = k'(r2) · w (x − x0)²` for the picked dimension, from coordinates.
fn grad_from_points_in_place<M: KernelMath>(
    r2: &mut [f64],
    xdim: &[f64],
    x0: f64,
    inv_dim: f64,
) -> Result<(), GprError> {
    debug_assert_eq!(r2.len(), xdim.len());
    let half = f64x4::splat(-0.5);
    let inv = f64x4::splat(inv_dim);
    let x0v = f64x4::splat(x0);
    let mut i = 0;
    while i + LANES <= r2.len() {
        let d = load4(r2, i);
        if !all_finite(d) {
            return Err(GprError::NonFiniteInput);
        }
        let delta = load4(xdim, i) - x0v;
        store4(r2, i, M::d1_f64x4(d * half) * delta * delta * inv);
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

/// `r2 = k'(r2) · w (Δx_d)²` for the picked dimension, from the cache.
fn grad_from_cache_in_place<M: KernelMath>(
    r2: &mut [f64],
    dim_sq: &[f64],
    inv_dim: f64,
) -> Result<(), GprError> {
    debug_assert_eq!(r2.len(), dim_sq.len());
    let half = f64x4::splat(-0.5);
    let inv = f64x4::splat(inv_dim);
    let mut i = 0;
    while i + LANES <= r2.len() {
        let d = load4(r2, i);
        let sq = load4(dim_sq, i);
        if !all_finite(d) || !all_finite(sq) {
            return Err(GprError::NonFiniteInput);
        }
        store4(r2, i, M::d1_f64x4(d * half) * sq * inv);
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

/// Where `(Δx_d)²` of the square output comes from.
#[derive(Clone, Copy)]
enum Square<'a> {
    Cache(ArdSqDiff<'a, f64>),
    Points(MatRef<'a, f64>),
}

/// `dest = Σ_d w_d (Δx_d)²` for the rows `row_start..` of column `col`.
fn accumulate_r2(
    source: Square<'_>,
    inv_ell_sq: &[f64],
    col: usize,
    row_start: usize,
    dest: &mut [f64],
) -> Result<(), GprError> {
    dest.fill(0.0);
    let n_rows = dest.len();
    match source {
        Square::Cache(cache) => {
            let offset = cached_rows_offset(row_start, col)?;
            for (dim, &w) in inv_ell_sq.iter().enumerate() {
                let src = cache.column(dim, col);
                scale_add(&src[offset..offset + n_rows], w, dest)?;
            }
        }
        Square::Points(x) => {
            for (dim, &w) in inv_ell_sq.iter().enumerate() {
                let xdim = col_slice_checked(x, dim)?;
                add_squared_diff_scaled(&xdim[row_start..row_start + n_rows], xdim[col], w, dest);
            }
        }
    }
    Ok(())
}

/// The value (`param_idx` `None`) or `∂k/∂θ_d` of the rows `row_start..` of
/// column `col` into `dest`.
fn fill_column<M: KernelMath>(
    source: Square<'_>,
    inv_ell_sq: &[f64],
    param_idx: Option<usize>,
    col: usize,
    row_start: usize,
    dest: &mut [f64],
) -> Result<(), GprError> {
    accumulate_r2(source, inv_ell_sq, col, row_start, dest)?;
    let Some(dim) = param_idx else {
        return exp_half_in_place::<M>(dest);
    };
    let rows = row_start..row_start + dest.len();
    match source {
        Square::Cache(cache) => {
            let offset = cached_rows_offset(row_start, col)?;
            let src = cache.column(dim, col);
            grad_from_cache_in_place::<M>(dest, &src[offset..offset + rows.len()], inv_ell_sq[dim])
        }
        Square::Points(x) => {
            let xdim = col_slice_checked(x, dim)?;
            grad_from_points_in_place::<M>(dest, &xdim[rows], xdim[col], inv_ell_sq[dim])
        }
    }
}

/// The square output for `uplo`; the lower triangle runs on the Rayon pool.
fn fill_square<M: KernelMath>(
    source: Square<'_>,
    mut out: MatMut<'_, f64>,
    uplo: Triangle,
    inv_ell_sq: &[f64],
    param_idx: Option<usize>,
) -> Result<(), GprError> {
    let n = out.nrows();
    if matches!(uplo, Triangle::Lower) {
        return par_lower_cols(out, &|col, rows| {
            fill_column::<M>(source, inv_ell_sq, param_idx, col, col, rows_checked(rows)?)
        });
    }
    for col in 0..n {
        let end = if matches!(uplo, Triangle::Upper) {
            col + 1
        } else {
            n
        };
        let dest = &mut col_slice_mut_checked(out.rb_mut(), col)?[..end];
        fill_column::<M>(source, inv_ell_sq, param_idx, col, 0, dest)?;
    }
    Ok(())
}

/// Writes ARD RBF from a raw `(Δx_d)²` cache when views are column-major.
pub(crate) fn try_apply_cache<M: KernelMath>(
    cache: ArdSqDiff<'_, f64>,
    out: MatMut<'_, f64>,
    uplo: Triangle,
    inv_ell_sq: &[f64],
) -> Result<bool, GprError> {
    let n = out.nrows();
    if out.ncols() != n {
        return Ok(false);
    }
    crate::kernel::dist::require_ard_sq_diff_shape(cache, n, inv_ell_sq.len())?;
    // The cache stores the lower triangle; the scalar path reads any pair.
    if uplo != Triangle::Lower || !unit_row_stride(out.as_ref()) {
        return Ok(false);
    }
    fill_square::<M>(Square::Cache(cache), out, uplo, inv_ell_sq, None)?;
    Ok(true)
}

/// Writes ARD RBF from coordinates when views are column-major.
pub(crate) fn try_apply_points<M: KernelMath>(
    x: MatRef<'_, f64>,
    out: MatMut<'_, f64>,
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
    fill_square::<M>(Square::Points(x), out, uplo, inv_ell_sq, None)?;
    Ok(true)
}

/// Writes ARD RBF `∂k/∂θ_d` from a raw `(Δx_d)²` cache.
pub(crate) fn try_grad_cache<M: KernelMath>(
    cache: ArdSqDiff<'_, f64>,
    d_k: MatMut<'_, f64>,
    uplo: Triangle,
    inv_ell_sq: &[f64],
    param_idx: usize,
) -> Result<bool, GprError> {
    let n = d_k.nrows();
    if d_k.ncols() != n || param_idx >= inv_ell_sq.len() {
        return Ok(false);
    }
    crate::kernel::dist::require_ard_sq_diff_shape(cache, n, inv_ell_sq.len())?;
    // The cache stores the lower triangle; the scalar path reads any pair.
    if uplo != Triangle::Lower || !unit_row_stride(d_k.as_ref()) {
        return Ok(false);
    }
    fill_square::<M>(Square::Cache(cache), d_k, uplo, inv_ell_sq, Some(param_idx))?;
    Ok(true)
}

/// Writes ARD RBF `∂k/∂θ_d` from coordinates.
pub(crate) fn try_grad_points<M: KernelMath>(
    x: MatRef<'_, f64>,
    d_k: MatMut<'_, f64>,
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
    fill_square::<M>(Square::Points(x), d_k, uplo, inv_ell_sq, Some(param_idx))?;
    Ok(true)
}

/// The lengthscale a cross pass writes `∂k/∂θ_d` for.
#[derive(Clone, Copy)]
pub(crate) enum Which {
    One(usize),
}

impl Which {
    fn includes(self, dim: usize) -> bool {
        match self {
            Self::One(d) => d == dim,
        }
    }
}

/// Whether both inputs have unit row stride and `d` columns, checking that
/// they are finite.
pub(crate) fn cross_inputs(
    x1: MatRef<'_, f64>,
    x2: MatRef<'_, f64>,
    d: usize,
) -> Result<bool, GprError> {
    if d == 0 || x1.ncols() != d || x2.ncols() != d {
        return Ok(false);
    }
    if !unit_row_stride(x1) || !unit_row_stride(x2) {
        return Ok(false);
    }
    for dim in 0..d {
        finite_slice(col_slice_checked(x1, dim)?)?;
        finite_slice(col_slice_checked(x2, dim)?)?;
    }
    Ok(true)
}

/// `∂k/∂θ_d = k · w_d (Δ_d)²` with one `exp` for every lengthscale.
/// `out` holds the one matrix of [`Which::One`]. `Ok(false)` when a view is
/// not column-major or a shape does not match.
pub(crate) fn try_grad_cross<M: KernelMath>(
    x1: MatRef<'_, f64>,
    x2: MatRef<'_, f64>,
    out: &mut [MatMut<'_, f64>],
    inv_ell_sq: &[f64],
    which: Which,
) -> Result<bool, GprError> {
    let m = x1.nrows();
    let n = x2.nrows();
    let d = inv_ell_sq.len();
    let expected = match which {
        Which::One(dim) if dim < d => 1,
        Which::One(_) => return Ok(false),
    };
    if out.len() != expected || !cross_inputs(x1, x2, d)? {
        return Ok(false);
    }
    for dest in out.iter() {
        if dest.nrows() != m || dest.ncols() != n || !unit_row_stride(dest.as_ref()) {
            return Ok(false);
        }
    }
    if n <= 1024 {
        write_cross_rows::<M>(
            x1,
            x2,
            inv_ell_sq,
            which,
            0,
            n,
            0,
            &mut |_dim, row, col, v| {
                out[0][(row, col)] = v;
            },
        )?;
        return Ok(true);
    }
    let n_parts = worker_count();
    let shared = ShareBases(out.iter_mut().map(packed_mut).collect());
    let results: Vec<Result<(), GprError>> = (0..n_parts)
        .into_par_iter()
        .map(|idx| {
            let (start, len) = col_chunk(n, idx, n_parts);
            write_cross_rows::<M>(
                x1,
                x2,
                inv_ell_sq,
                which,
                start,
                len,
                start,
                &mut |_dim, row, col, v| {
                    // SAFETY: `shared` holds the column-major matrices of
                    // `out`, each `m × n` with row stride 1 (checked above);
                    // `row < m`, and `col` is in this worker's own column
                    // range of `n`, which no other worker writes.
                    unsafe { store_packed(&shared.slots()[0], row, col, v) }
                },
            )
        })
        .collect();
    for result in results {
        result?;
    }
    Ok(true)
}

/// Writes `∂k/∂θ_d` of the lengthscales in `which` for every row of `x1`
/// and the `len` points of `x2` from `x_begin`, at destination columns from
/// `dest_col`, through `store(dim, row, col, value)`. Columns go in stack
/// blocks; each lengthscale's term is formed again after the `exp` rather
/// than kept, so nothing is allocated.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_cross_rows<M: KernelMath>(
    x1: MatRef<'_, f64>,
    x2: MatRef<'_, f64>,
    inv_ell_sq: &[f64],
    which: Which,
    x_begin: usize,
    len: usize,
    dest_col: usize,
    store: &mut dyn FnMut(usize, usize, usize, f64),
) -> Result<(), GprError> {
    const BLOCK: usize = 256;
    let mut r2_buf = [0.0f64; BLOCK];
    let mut k_buf = [0.0f64; BLOCK];
    let mut term_buf = [0.0f64; BLOCK];
    let mut offset = 0;
    while offset < len {
        let block = BLOCK.min(len - offset);
        let begin = x_begin + offset;
        let (r2, k, term) = (
            &mut r2_buf[..block],
            &mut k_buf[..block],
            &mut term_buf[..block],
        );
        for row in 0..x1.nrows() {
            r2.fill(0.0);
            for (dim, &w) in inv_ell_sq.iter().enumerate() {
                let z = col_slice_checked(x1, dim)?[row];
                add_squared_diff_scaled(
                    &col_slice_checked(x2, dim)?[begin..begin + block],
                    z,
                    w,
                    r2,
                );
            }
            d1_half::<M>(r2, k)?;
            for (dim, &w) in inv_ell_sq.iter().enumerate() {
                if !which.includes(dim) {
                    continue;
                }
                let z = col_slice_checked(x1, dim)?[row];
                term.fill(0.0);
                add_squared_diff_scaled(
                    &col_slice_checked(x2, dim)?[begin..begin + block],
                    z,
                    w,
                    term,
                );
                for (i, (&kv, &tv)) in k.iter().zip(term.iter()).enumerate() {
                    let dk = kv * tv;
                    if !dk.is_finite() {
                        return Err(GprError::NonFiniteKernelValue);
                    }
                    store(dim, row, dest_col + offset + i, dk);
                }
            }
        }
        offset += block;
    }
    Ok(())
}

/// `dest = k'(src)`: the derivative of `exp(−r² / 2)` along `−r² / 2`
/// (with [`crate::Accurate`] the value itself).
fn d1_half<M: KernelMath>(src: &[f64], dest: &mut [f64]) -> Result<(), GprError> {
    let scale = f64x4::splat(-0.5);
    let mut i = 0;
    while i + LANES <= src.len() {
        let z = load4(src, i) * scale;
        let v = if M::ACCURATE { z.exp() } else { M::d1_f64x4(z) };
        if !all_finite(v) {
            return Err(GprError::NonFiniteKernelValue);
        }
        store4(dest, i, v);
        i += LANES;
    }
    while i < src.len() {
        let z = -0.5 * src[i];
        let v = if M::ACCURATE { z.exp() } else { M::jet(z).d1 };
        if !v.is_finite() {
            return Err(GprError::NonFiniteKernelValue);
        }
        dest[i] = v;
        i += 1;
    }
    Ok(())
}

/// Columns of one rectangular contraction whose `exp` is evaluated together.
/// Same batch as [`crate::kernel::dist::par_lower_fold`].
const COLUMN_BATCH: usize = 16;

/// `⟨weight, ∂K/∂θ_d⟩` for every lengthscale, and `⟨weight, K⟩` when
/// `want_value` is set. One `exp` per pair, then one matrix product for
/// `Σ (W ∘ k) (Δ_d)²`. `Ok(None)` when a view is not column-major.
///
/// `jobs` holds the column scratch, `S = W ∘ k`, and the product. It grows
/// on the first call and is kept by the caller.
pub(crate) fn try_contract_cross<M: KernelMath>(
    x1: MatRef<'_, f64>,
    x2: MatRef<'_, f64>,
    weight: MatRef<'_, f64>,
    inv_ell_sq: &[f64],
    out: &mut [f64],
    jobs: &mut Vec<f64>,
    want_value: bool,
) -> Result<Option<f64>, GprError> {
    let m = x1.nrows();
    let n = x2.nrows();
    let d = inv_ell_sq.len();
    if out.len() != d || weight.nrows() != m || weight.ncols() != n || !cross_inputs(x1, x2, d)? {
        return Ok(None);
    }
    if !unit_row_stride(weight) {
        return Ok(None);
    }
    let batch = COLUMN_BATCH.min(n);
    let scratch_len = m
        .checked_mul(2)
        .and_then(|rows| rows.checked_mul(batch))
        .ok_or(GprError::SizeOverflow)?;
    let s_len = m.checked_mul(n).ok_or(GprError::SizeOverflow)?;
    let prod_len = n.checked_mul(d).ok_or(GprError::SizeOverflow)?;
    let need = scratch_len
        .checked_add(s_len)
        .and_then(|sum| sum.checked_add(m))
        .and_then(|sum| sum.checked_add(n))
        .and_then(|sum| sum.checked_add(prod_len))
        .ok_or(GprError::SizeOverflow)?;
    if jobs.len() < need {
        jobs.resize(need, 0.0);
    }
    let (scratch, rest) = jobs.split_at_mut(scratch_len);
    let (s, rest) = rest.split_at_mut(s_len);
    let (row_sum, rest) = rest.split_at_mut(m);
    let (col_sum, prod) = rest.split_at_mut(n);
    let prod = &mut prod[..prod_len];
    let mut values = [0.0; COLUMN_BATCH];
    let mut value = 0.0;
    let mut start = 0;
    while start < n {
        let len = batch.min(n - start);
        fill_contract_batch::<M>(
            x1,
            x2,
            weight,
            inv_ell_sq,
            start,
            &mut scratch[..len * 2 * m],
            &mut s[start * m..(start + len) * m],
            &mut values[..len],
            want_value,
        )?;
        if want_value {
            for part in values.iter().take(len) {
                value += part;
            }
        }
        start += len;
    }
    fold_rect_lengthscales(x1, x2, s, inv_ell_sq, row_sum, col_sum, prod, out)?;
    if !value.is_finite() {
        return Err(GprError::NonFiniteKernelValue);
    }
    Ok(Some(value))
}

/// Column `col0` and the `values.len()` columns after it. `s` holds those
/// columns of `W ∘ k`, packed column-major.
#[allow(clippy::too_many_arguments)]
fn fill_contract_batch<M: KernelMath>(
    x1: MatRef<'_, f64>,
    x2: MatRef<'_, f64>,
    weight: MatRef<'_, f64>,
    inv_ell_sq: &[f64],
    col0: usize,
    scratch: &mut [f64],
    s: &mut [f64],
    values: &mut [f64],
    want_value: bool,
) -> Result<(), GprError> {
    let n = values.len();
    let m = x1.nrows();
    if n == 0 {
        return Ok(());
    }
    if n == 1 {
        let (r2, k) = scratch.split_at_mut(m);
        values[0] = contract_column::<M>(x1, x2, weight, inv_ell_sq, col0, r2, k, s, want_value)?;
        return Ok(());
    }
    let mid = n / 2;
    let (left_scratch, right_scratch) = scratch.split_at_mut(mid * 2 * m);
    let (left_s, right_s) = s.split_at_mut(mid * m);
    let (left_values, right_values) = values.split_at_mut(mid);
    let (left, right) = if rayon::current_num_threads() > 1 {
        rayon::join(
            || {
                fill_contract_batch::<M>(
                    x1,
                    x2,
                    weight,
                    inv_ell_sq,
                    col0,
                    left_scratch,
                    left_s,
                    left_values,
                    want_value,
                )
            },
            || {
                fill_contract_batch::<M>(
                    x1,
                    x2,
                    weight,
                    inv_ell_sq,
                    col0 + mid,
                    right_scratch,
                    right_s,
                    right_values,
                    want_value,
                )
            },
        )
    } else {
        let left = fill_contract_batch::<M>(
            x1,
            x2,
            weight,
            inv_ell_sq,
            col0,
            left_scratch,
            left_s,
            left_values,
            want_value,
        );
        let right = fill_contract_batch::<M>(
            x1,
            x2,
            weight,
            inv_ell_sq,
            col0 + mid,
            right_scratch,
            right_s,
            right_values,
            want_value,
        );
        (left, right)
    };
    left.and(right)
}

/// One column of `⟨W, K⟩` and of `S = W ∘ k`. `r2` and `k` each hold `m`
/// entries. `s_col` holds `m`.
#[allow(clippy::too_many_arguments)]
fn contract_column<M: KernelMath>(
    x1: MatRef<'_, f64>,
    x2: MatRef<'_, f64>,
    weight: MatRef<'_, f64>,
    inv_ell_sq: &[f64],
    col: usize,
    r2: &mut [f64],
    k: &mut [f64],
    s_col: &mut [f64],
    want_value: bool,
) -> Result<f64, GprError> {
    let m = x1.nrows();
    r2.fill(0.0);
    for (dim, &w) in inv_ell_sq.iter().enumerate() {
        let x0 = col_slice_checked(x2, dim)?[col];
        add_squared_diff_scaled(col_slice_checked(x1, dim)?, x0, w, r2);
    }
    d1_half::<M>(r2, k)?;
    let wcol = col_slice_checked(weight, col)?;
    let mut value = 0.0;
    for row in 0..m {
        let s = wcol[row] * k[row];
        s_col[row] = s;
        if want_value {
            value += s;
        }
    }
    Ok(value)
}

/// `Σ_{ij} S_ij (x1_i − x2_j)²`, times each `inv_ell_sq`, from one product
/// `Sᵀ X1`.
///
/// `S` is `m × n` column-major. `prod` holds `n × d`. One thread: a parallel
/// faer product allocates a scratch buffer on every call, and this product
/// replaces a scalar pass.
#[allow(clippy::too_many_arguments)]
fn fold_rect_lengthscales(
    x1: MatRef<'_, f64>,
    x2: MatRef<'_, f64>,
    s: &[f64],
    inv_ell_sq: &[f64],
    row_sum: &mut [f64],
    col_sum: &mut [f64],
    prod: &mut [f64],
    out: &mut [f64],
) -> Result<(), GprError> {
    let m = x1.nrows();
    let n = x2.nrows();
    let d = inv_ell_sq.len();
    row_sum.fill(0.0);
    for col in 0..n {
        let column = &s[col * m..(col + 1) * m];
        let mut sum = 0.0;
        for (row, &value) in column.iter().enumerate() {
            row_sum[row] += value;
            sum += value;
        }
        col_sum[col] = sum;
    }
    let s_ref = MatRef::from_column_major_slice(s, m, n);
    let mut prod_mat = MatMut::from_column_major_slice_mut(prod, n, d);
    matmul(
        prod_mat.rb_mut(),
        Accum::Replace,
        s_ref.transpose(),
        x1,
        1.0,
        Par::Seq,
    );
    for (dim, &wd) in inv_ell_sq.iter().enumerate() {
        let x1_col = col_slice_checked(x1, dim)?;
        let x2_col = col_slice_checked(x2, dim)?;
        let prod_col = &prod[dim * n..(dim + 1) * n];
        let mut left = 0.0;
        for row in 0..m {
            left += x1_col[row] * x1_col[row] * row_sum[row];
        }
        let mut right = 0.0;
        let mut cross = 0.0;
        for col in 0..n {
            right += x2_col[col] * x2_col[col] * col_sum[col];
            cross += prod_col[col] * x2_col[col];
        }
        let total = wd * (left + right - 2.0 * cross);
        if !total.is_finite() {
            return Err(GprError::NonFiniteKernelValue);
        }
        out[dim] = total;
    }
    Ok(())
}

/// `2 Σ_{i>j} S_ij (x_i − x_j)²` for every lengthscale, from one product
/// `S X`. `x` and `prod` are `n × d` column-major. `s` is symmetric `n × n`
/// with a zero diagonal. One thread, for the same reason as
/// [`fold_rect_lengthscales`].
pub(crate) fn fold_square_lengthscales(
    x: &[f64],
    s: &[f64],
    inv_ell_sq: &[f64],
    row_sum: &mut [f64],
    prod: &mut [f64],
    out: &mut [f64],
) -> Result<(), GprError> {
    let n = row_sum.len();
    let d = inv_ell_sq.len();
    row_sum.fill(0.0);
    for col in 0..n {
        let column = &s[col * n..(col + 1) * n];
        for (row, &value) in column.iter().enumerate() {
            row_sum[row] += value;
        }
    }
    let s_ref = MatRef::from_column_major_slice(s, n, n);
    let x_ref = MatRef::from_column_major_slice(x, n, d);
    let mut prod_mat = MatMut::from_column_major_slice_mut(prod, n, d);
    matmul(
        prod_mat.rb_mut(),
        Accum::Replace,
        s_ref,
        x_ref,
        1.0,
        Par::Seq,
    );
    for (dim, &wd) in inv_ell_sq.iter().enumerate() {
        let x_col = &x[dim * n..(dim + 1) * n];
        let prod_col = &prod[dim * n..(dim + 1) * n];
        let mut xx = 0.0;
        let mut quad = 0.0;
        for row in 0..n {
            xx += x_col[row] * x_col[row] * row_sum[row];
            quad += prod_col[row] * x_col[row];
        }
        let total = 2.0 * wd * (xx - quad);
        if !total.is_finite() {
            return Err(GprError::NonFiniteKernelValue);
        }
        out[dim] = total;
    }
    Ok(())
}

/// The base pointer and column stride of a column-major matrix.
struct PackedMut {
    ptr: *mut f64,
    stride: isize,
}

struct ShareBases(Vec<PackedMut>);

impl ShareBases {
    fn slots(&self) -> &[PackedMut] {
        &self.0
    }
}

/// # Safety
///
/// Each pointer is a column-major matrix. Parallel callers write disjoint
/// columns of those matrices and do not read the columns they write.
unsafe impl Send for ShareBases {}

/// # Safety
///
/// Each pointer is a column-major matrix. Parallel callers write disjoint
/// columns of those matrices and do not read the columns they write.
unsafe impl Sync for ShareBases {}

fn packed_mut(mat: &mut MatMut<'_, f64>) -> PackedMut {
    let view = mat.rb_mut();
    debug_assert!(view.ncols() == 0 || view.row_stride() == 1);
    PackedMut {
        ptr: view.as_ptr_mut(),
        stride: view.col_stride(),
    }
}

/// Writes `value` at `(row, col)` of the matrix `slot` addresses.
///
/// # Safety
///
/// `slot` addresses a live column-major matrix whose row stride is `+1`,
/// `row` is inside that matrix and `col` is inside its column count, and no
/// other thread reads or writes `(row, col)` meanwhile.
#[inline(always)]
unsafe fn store_packed(slot: &PackedMut, row: usize, col: usize, value: f64) {
    // SAFETY: the caller keeps `(row, col)` inside the matrix, whose row
    // stride is +1, and owns that entry for the write.
    unsafe {
        *slot.ptr.offset(row as isize + col as isize * slot.stride) = value;
    }
}
