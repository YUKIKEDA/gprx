//! `f64x4` paths of the isotropic stationary leaves over any storage: the
//! RBF, Periodic, and rational-quadratic square Gram from cached squared
//! distances, the RBF `∂K/∂θ` and rectangle, the rectangular RBF `∂K/∂θ`
//! from coordinates, and the one-pass weighted gradient of Periodic and RQ.
//!
//! `f32` storage is widened to `f64` lanes and rounded once on the store;
//! `f64` takes the same lanes. A column's last partial lane is padded, so it
//! uses the same lane functions as the rest. The rational quadratic
//! `u^{-α}` is `exp(−α ln u)` here.

use super::{
    LANES, add_squared_diff, checked_input, checked_kernel, col_slice, col_slice_checked,
    col_slice_mut_checked, finite_slice, load, load4, rows_checked, store, store4, sum_lanes,
    unit_row_stride,
};
use crate::error::GprError;
use crate::kernel::dist::{col_chunk, par_lower_cols, par_lower_fold, worker_count};
use crate::kernel::{KernelScalar, Triangle, require_same_shape, require_square_pair};
use crate::math::KernelMath;
use faer::reborrow::ReborrowMut;
use faer::{MatMut, MatRef};
use rayon::prelude::*;
use wide::f64x4;

/// Periodic constants: `π / p` and `2 / ℓ²`.
#[derive(Clone, Copy)]
pub(crate) struct PeriodicScales {
    pub(crate) pi_over_period: f64,
    pub(crate) two_inv_ell_sq: f64,
}

/// Rational-quadratic constants: `1 / ℓ²` and `α`.
#[derive(Clone, Copy)]
pub(crate) struct RqScales {
    pub(crate) inv_ell_sq: f64,
    pub(crate) alpha: f64,
}

/// `exp(−2 sin²(π √d / p) / ℓ²)`, with `sin(π √d / p)` and `π √d / p`.
#[inline(always)]
fn periodic_lanes<M: KernelMath>(d: f64x4, s: PeriodicScales) -> (f64x4, f64x4, f64x4, f64x4) {
    let alpha = d.max(f64x4::ZERO).sqrt() * f64x4::splat(s.pi_over_period);
    let (sin, cos) = alpha.sin_cos();
    let z = -(sin * sin) * f64x4::splat(s.two_inv_ell_sq);
    (M::exp_f64x4(z), sin, cos, alpha)
}

/// `(r², u, ln u)` with `r² = d / ℓ²` and `u = 1 + r² / (2α)`.
#[inline(always)]
fn rq_lanes(d: f64x4, s: RqScales) -> (f64x4, f64x4, f64x4) {
    let r2 = d.max(f64x4::ZERO) * f64x4::splat(s.inv_ell_sq);
    let u = f64x4::ONE + r2 / f64x4::splat(2.0 * s.alpha);
    (r2, u, u.ln())
}

fn periodic_slice<M: KernelMath, T: KernelScalar>(
    dist: &[T],
    out: &mut [T],
    s: PeriodicScales,
) -> Result<(), GprError> {
    let mut i = 0;
    while i < dist.len() {
        let d = checked_input(load(dist, i, 0.0))?;
        store(out, i, checked_kernel(periodic_lanes::<M>(d, s).0)?);
        i += LANES;
    }
    Ok(())
}

fn rq_slice<T: KernelScalar>(dist: &[T], out: &mut [T], s: RqScales) -> Result<(), GprError> {
    let mut i = 0;
    while i < dist.len() {
        let d = checked_input(load(dist, i, 0.0))?;
        let (_, _, ln_u) = rq_lanes(d, s);
        store(
            out,
            i,
            checked_kernel((-f64x4::splat(s.alpha) * ln_u).exp())?,
        );
        i += LANES;
    }
    Ok(())
}

/// Maps each column's rows of `uplo` through `f`; the lower triangle runs
/// on the Rayon pool. `Ok(false)` when a view is not unit row-stride.
fn try_map_square<T: KernelScalar>(
    dist: MatRef<'_, T>,
    mut out: MatMut<'_, T>,
    uplo: Triangle,
    f: impl Fn(&[T], &mut [T]) -> Result<(), GprError> + Sync,
) -> Result<bool, GprError> {
    let n = require_square_pair(dist, out.as_ref())?;
    if !unit_row_stride(dist) || !unit_row_stride(out.as_ref()) {
        return Ok(false);
    }
    if matches!(uplo, Triangle::Lower) {
        par_lower_cols(out, &|col, rows| {
            f(&col_slice_checked(dist, col)?[col..], rows_checked(rows)?)
        })?;
        return Ok(true);
    }
    for col in 0..n {
        let end = if matches!(uplo, Triangle::Upper) {
            col + 1
        } else {
            n
        };
        let src = &col_slice_checked(dist, col)?[..end];
        f(src, &mut col_slice_mut_checked(out.rb_mut(), col)?[..end])?;
    }
    Ok(true)
}

/// RBF constants: `1 / (2ℓ²)` and `1 / ℓ²`.
#[derive(Clone, Copy)]
pub(crate) struct RbfScales {
    pub(crate) half_inv_ell_sq: f64,
    pub(crate) inv_ell_sq: f64,
}

/// `exp(−d / (2ℓ²))`, or with `grad` its `∂/∂log ℓ = k d / ℓ²` (the
/// [`KernelMath`] derivative, as the `f64` lanes take it). Like the scalar
/// path, only `d` is checked: both are finite for a finite `d`.
fn rbf_slice<M: KernelMath, T: KernelScalar>(
    dist: &[T],
    out: &mut [T],
    s: RbfScales,
    grad: bool,
) -> Result<(), GprError> {
    let neg_half = f64x4::splat(-s.half_inv_ell_sq);
    let inv = f64x4::splat(s.inv_ell_sq);
    // Full lanes first, then one padded lane for the tail.
    let full = dist.len() - dist.len() % LANES;
    let mut i = 0;
    if grad {
        let lane = |d: f64x4| M::d1_f64x4(d * neg_half) * d * inv;
        while i < full {
            store4(out, i, lane(checked_input(load4(dist, i))?));
            i += LANES;
        }
        if i < dist.len() {
            store(out, i, lane(checked_input(load(dist, i, 0.0))?));
        }
    } else {
        let lane = |d: f64x4| M::exp_f64x4(d * neg_half);
        while i < full {
            store4(out, i, lane(checked_input(load4(dist, i))?));
            i += LANES;
        }
        if i < dist.len() {
            store(out, i, lane(checked_input(load(dist, i, 0.0))?));
        }
    }
    Ok(())
}

/// Writes the RBF Gram (or with `grad` its `∂K/∂log ℓ`) for `uplo` from
/// squared distances.
pub(crate) fn try_square_rbf<M: KernelMath, T: KernelScalar>(
    dist: MatRef<'_, T>,
    out: MatMut<'_, T>,
    uplo: Triangle,
    s: RbfScales,
    grad: bool,
) -> Result<bool, GprError> {
    try_map_square(dist, out, uplo, |src, dest| {
        rbf_slice::<M, T>(src, dest, s, grad)
    })
}

/// Writes the rectangular RBF `k(dist)` (train × test) from squared distances.
pub(crate) fn try_apply_rbf_cross<M: KernelMath, T: KernelScalar>(
    dist: MatRef<'_, T>,
    mut out: MatMut<'_, T>,
    s: RbfScales,
) -> Result<bool, GprError> {
    require_same_shape(dist, out.as_ref())?;
    if !unit_row_stride(dist) || !unit_row_stride(out.as_ref()) {
        return Ok(false);
    }
    let m = dist.ncols();
    let column = |col: usize, dest: &mut [T]| {
        rbf_slice::<M, T>(col_slice_checked(dist, col)?, dest, s, false)
    };
    if m == 1 {
        column(0, col_slice_mut_checked(out, 0)?)?;
        return Ok(true);
    }
    let n_parts = worker_count().clamp(1, m.max(1));
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

/// Writes the Periodic Gram for `uplo` from squared distances.
pub(crate) fn try_apply_periodic<M: KernelMath, T: KernelScalar>(
    dist: MatRef<'_, T>,
    out: MatMut<'_, T>,
    uplo: Triangle,
    s: PeriodicScales,
) -> Result<bool, GprError> {
    try_map_square(dist, out, uplo, |src, dest| {
        periodic_slice::<M, T>(src, dest, s)
    })
}

/// Writes the rational-quadratic Gram for `uplo` from squared distances.
pub(crate) fn try_apply_rq<T: KernelScalar>(
    dist: MatRef<'_, T>,
    out: MatMut<'_, T>,
    uplo: Triangle,
    s: RqScales,
) -> Result<bool, GprError> {
    try_map_square(dist, out, uplo, |src, dest| rq_slice(src, dest, s))
}

/// Column slices `rows col..n` of the lower-triangle views a weighted pass reads.
struct LowerColumns<'a, T> {
    dist: MatRef<'a, T>,
    weight: MatRef<'a, T>,
    k: Option<MatRef<'a, T>>,
}

impl<'a, T: KernelScalar> LowerColumns<'a, T> {
    fn new(dist: MatRef<'a, T>, weight: MatRef<'a, T>, k: Option<MatRef<'a, T>>) -> Option<Self> {
        let unit = unit_row_stride(dist)
            && unit_row_stride(weight)
            && k.is_none_or(unit_row_stride)
            && weight.nrows() == dist.nrows()
            && weight.ncols() == dist.ncols()
            && k.is_none_or(|k| k.nrows() == dist.nrows() && k.ncols() == dist.ncols());
        unit.then_some(Self { dist, weight, k })
    }

    /// `(dist, weight, k)` of column `col`, rows `col..n`.
    #[allow(clippy::type_complexity)]
    fn column(&self, col: usize) -> Option<(&'a [T], &'a [T], Option<&'a [T]>)> {
        let d = &col_slice(self.dist, col)?[col..];
        let w = &col_slice(self.weight, col)?[col..];
        let k = match self.k {
            Some(k) => Some(&col_slice(k, col)?[col..]),
            None => None,
        };
        Some((d, w, k))
    }
}

/// `w` lanes of a column slice: the diagonal (`i = 0`) once, the rest twice.
#[inline(always)]
fn sym_weight<T: KernelScalar>(w: &[T], i: usize) -> f64x4 {
    let v = load(w, i, 0.0) * f64x4::splat(2.0);
    if i == 0 {
        let mut a = v.to_array();
        a[0] *= 0.5;
        f64x4::new(a)
    } else {
        v
    }
}

/// Per-block `[∂ first, ∂ second, value]` of a weighted pass; `None` when
/// a column is not unit row-stride.
type Sums = [f64; 3];

fn join_sums(a: Option<Sums>, b: Option<Sums>) -> Option<Sums> {
    let (a, b) = (a?, b?);
    Some([a[0] + b[0], a[1] + b[1], a[2] + b[2]])
}

/// One-pass `⟨weight, ∂K/∂log ℓ⟩`, `⟨weight, ∂K/∂log p⟩`, and the returned
/// `⟨weight, K⟩` of the Periodic leaf, or `None` when a view is not unit
/// row-stride. `k` is the leaf's own Gram; with [`crate::Accurate`] it
/// replaces `exp`.
pub(crate) fn try_weighted_periodic<M: KernelMath, T: KernelScalar>(
    dist: MatRef<'_, T>,
    k: Option<MatRef<'_, T>>,
    weight: MatRef<'_, T>,
    s: PeriodicScales,
    out: &mut [f64],
) -> Result<Option<f64>, GprError> {
    let Some(cols) = LowerColumns::new(dist, weight, k) else {
        return Ok(None);
    };
    let two_inv = f64x4::splat(2.0 * s.two_inv_ell_sq);
    let block = |start: usize, end: usize| -> Result<Option<Sums>, GprError> {
        let (mut g_ell, mut g_period, mut value) = (f64x4::ZERO, f64x4::ZERO, f64x4::ZERO);
        for col in start..end {
            let Some((d, w, kc)) = cols.column(col) else {
                return Ok(None);
            };
            let mut i = 0;
            while i < d.len() {
                let dv = checked_input(load(d, i, 0.0))?;
                let wv = sym_weight(w, i);
                let alpha = dv.max(f64x4::ZERO).sqrt() * f64x4::splat(s.pi_over_period);
                let (sin, cos) = alpha.sin_cos();
                let z = -(sin * sin) * f64x4::splat(s.two_inv_ell_sq);
                let (kv, e) = match (M::ACCURATE, kc) {
                    (true, Some(kc)) => {
                        let kv = load(kc, i, 0.0);
                        (kv, kv)
                    }
                    (true, None) => {
                        let kv = z.exp();
                        (kv, kv)
                    }
                    (false, _) => (M::exp_f64x4(z), M::d1_f64x4(z)),
                };
                // `∂k/∂log ℓ = 4 s² / ℓ² · e`, `∂k/∂log p = 4 s c α / ℓ² · e`.
                let dk_ell = checked_kernel(e * two_inv * sin * sin)?;
                let dk_period = checked_kernel(e * two_inv * sin * cos * alpha)?;
                g_ell += wv * dk_ell;
                g_period += wv * dk_period;
                value += wv * kv;
                i += LANES;
            }
        }
        Ok(Some([
            sum_lanes(g_ell),
            sum_lanes(g_period),
            sum_lanes(value),
        ]))
    };
    let Some([g_ell, g_period, value]) = par_lower_fold(dist.ncols(), &block, &join_sums)? else {
        return Ok(None);
    };
    out[0] = g_ell;
    out[1] = g_period;
    Ok(Some(value))
}

/// One-pass `⟨weight, ∂K/∂log ℓ⟩`, `⟨weight, ∂K/∂log α⟩`, and the returned
/// `⟨weight, K⟩` of the rational-quadratic leaf, or `None` when a view is
/// not unit row-stride. `k` is the leaf's own Gram; it replaces `exp`.
pub(crate) fn try_weighted_rq<T: KernelScalar>(
    dist: MatRef<'_, T>,
    k: Option<MatRef<'_, T>>,
    weight: MatRef<'_, T>,
    s: RqScales,
    out: &mut [f64],
) -> Result<Option<f64>, GprError> {
    let Some(cols) = LowerColumns::new(dist, weight, k) else {
        return Ok(None);
    };
    let alpha = f64x4::splat(s.alpha);
    let block = |start: usize, end: usize| -> Result<Option<Sums>, GprError> {
        let (mut g_ell, mut g_alpha, mut value) = (f64x4::ZERO, f64x4::ZERO, f64x4::ZERO);
        for col in start..end {
            let Some((d, w, kc)) = cols.column(col) else {
                return Ok(None);
            };
            let mut i = 0;
            while i < d.len() {
                let dv = checked_input(load(d, i, 0.0))?;
                let wv = sym_weight(w, i);
                let (r2, u, ln_u) = rq_lanes(dv, s);
                let kv = match kc {
                    Some(kc) => load(kc, i, 0.0),
                    None => (-alpha * ln_u).exp(),
                };
                let dk_ell = checked_kernel(kv / u * r2)?;
                let dk_alpha = checked_kernel(alpha * kv * (f64x4::ONE - ln_u - f64x4::ONE / u))?;
                g_ell += wv * dk_ell;
                g_alpha += wv * dk_alpha;
                value += wv * kv;
                i += LANES;
            }
        }
        Ok(Some([
            sum_lanes(g_ell),
            sum_lanes(g_alpha),
            sum_lanes(value),
        ]))
    };
    let Some([g_ell, g_alpha, value]) = par_lower_fold(dist.ncols(), &block, &join_sums)? else {
        return Ok(None);
    };
    out[0] = g_ell;
    out[1] = g_alpha;
    Ok(Some(value))
}

/// `∂k/∂log ℓ = k s / ℓ²` of the RBF on a rectangular pair, from
/// coordinates (`x1` rows × `x2` rows), or `Ok(false)` when a view is not
/// column-major or the shapes do not match. Columns go in stack chunks, so
/// the gradient of a mini-batch step does not allocate.
pub(crate) fn try_grad_rbf_cross_from_coords<M: KernelMath, T: KernelScalar>(
    x1: MatRef<'_, T>,
    x2: MatRef<'_, T>,
    mut d_k: MatMut<'_, T>,
    s: RbfScales,
) -> Result<bool, GprError> {
    const CHUNK: usize = 256;
    let (m, n, d) = (x1.nrows(), x2.nrows(), x1.ncols());
    if d == 0 || x2.ncols() != d || d_k.nrows() != m || d_k.ncols() != n {
        return Ok(false);
    }
    if !unit_row_stride(x1) || !unit_row_stride(x2) || !unit_row_stride(d_k.as_ref()) {
        return Ok(false);
    }
    for dim in 0..d {
        finite_slice(col_slice_checked(x1, dim)?)?;
        finite_slice(col_slice_checked(x2, dim)?)?;
    }
    let mut s_buf = [0.0f64; CHUNK];
    let mut dk_buf = [0.0f64; CHUNK];
    let neg = f64x4::splat(-s.half_inv_ell_sq);
    let scale = f64x4::splat(s.inv_ell_sq);
    let mut start = 0;
    while start < n {
        let len = CHUNK.min(n - start);
        let (sq, dk) = (&mut s_buf[..len], &mut dk_buf[..len]);
        for row in 0..m {
            sq.fill(0.0);
            for dim in 0..d {
                let z = col_slice_checked(x1, dim)?[row].to_f64();
                add_squared_diff(&col_slice_checked(x2, dim)?[start..start + len], z, sq);
            }
            let mut i = 0;
            while i < len {
                let sv = load(sq, i, 0.0);
                store(dk, i, checked_kernel(M::d1_f64x4(sv * neg) * sv * scale)?);
                i += LANES;
            }
            for (col, value) in dk.iter().enumerate() {
                d_k[(row, start + col)] = T::from_f64(*value);
            }
        }
        start += len;
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::{
        PeriodicScales, RqScales, try_apply_periodic, try_apply_rq, try_weighted_periodic,
        try_weighted_rq,
    };
    use crate::error::GprError;
    use crate::kernel::Triangle;
    use crate::math::{Accurate, FastApprox, KernelMath};
    use faer::Mat;

    /// Squared distances of 7 points (one partial lane per column) spread
    /// past several periods.
    fn dist() -> Mat<f64> {
        let x = [0.0, 0.31, 1.7, 4.2, 9.9, 23.5, 41.0];
        Mat::from_fn(7, 7, |i, j| (x[i] - x[j]) * (x[i] - x[j]))
    }

    fn weight() -> Mat<f64> {
        Mat::from_fn(7, 7, |i, j| {
            let (a, b) = (i.max(j) as f64, i.min(j) as f64);
            0.3 * a - 0.7 * b + 0.05 * a * b - 0.4
        })
    }

    const PER: PeriodicScales = PeriodicScales {
        pi_over_period: std::f64::consts::PI / 1.3,
        two_inv_ell_sq: 2.0 / (0.9 * 0.9),
    };
    const RQ: RqScales = RqScales {
        inv_ell_sq: 1.0 / (1.2 * 1.2),
        alpha: 0.78,
    };

    fn periodic_ref<M: KernelMath>(d: f64) -> (f64, f64, f64) {
        let a = d.sqrt() * PER.pi_over_period;
        let (s, c) = (a.sin(), a.cos());
        let z = -s * s * PER.two_inv_ell_sq;
        let jet = M::jet(z);
        let four = 2.0 * PER.two_inv_ell_sq;
        (jet.v, jet.d1 * four * s * s, jet.d1 * four * s * c * a)
    }

    fn rq_ref(d: f64) -> (f64, f64, f64) {
        let r2 = d * RQ.inv_ell_sq;
        let u = 1.0 + r2 / (2.0 * RQ.alpha);
        let k = u.powf(-RQ.alpha);
        (k, k / u * r2, RQ.alpha * k * (1.0 - u.ln() - 1.0 / u))
    }

    fn assert_rel(got: f64, want: f64, tol: f64) {
        assert!(
            (got - want).abs() <= tol * want.abs().max(1.0),
            "got={got} want={want}"
        );
    }

    fn gram_matches<M: KernelMath>(tol: f64) {
        let d = dist();
        for uplo in [Triangle::Lower, Triangle::Upper, Triangle::Full] {
            let mut per = Mat::<f64>::zeros(7, 7);
            let mut rq = Mat::<f64>::zeros(7, 7);
            assert!(
                try_apply_periodic::<M, f64>(d.as_ref(), per.as_mut(), uplo, PER).expect("finite")
            );
            assert!(try_apply_rq(d.as_ref(), rq.as_mut(), uplo, RQ).expect("finite"));
            for j in 0..7 {
                for i in 0..7 {
                    let inside = match uplo {
                        Triangle::Lower => i >= j,
                        Triangle::Upper => i <= j,
                        Triangle::Full => true,
                    };
                    if inside {
                        assert_rel(per[(i, j)], periodic_ref::<M>(d[(i, j)]).0, tol);
                        assert_rel(rq[(i, j)], rq_ref(d[(i, j)]).0, tol);
                    } else {
                        // Untouched: still the zero the buffer started with.
                        assert_eq!(per[(i, j)].to_bits(), 0);
                        assert_eq!(rq[(i, j)].to_bits(), 0);
                    }
                }
            }
        }
    }

    #[test]
    fn grams_match_the_scalar_formulas() {
        gram_matches::<Accurate>(1e-13);
        gram_matches::<FastApprox>(1e-13);
    }

    fn weighted_matches<M: KernelMath>(with_k: bool) {
        let (d, w) = (dist(), weight());
        let mut want = [0.0; 6];
        for j in 0..7 {
            for i in j..7 {
                let s = if i == j { 1.0 } else { 2.0 } * w[(i, j)];
                let (k, a, b) = periodic_ref::<M>(d[(i, j)]);
                want[0] += s * a;
                want[1] += s * b;
                want[2] += s * k;
                let (k, a, b) = rq_ref(d[(i, j)]);
                want[3] += s * a;
                want[4] += s * b;
                want[5] += s * k;
            }
        }
        let mut per_k = Mat::<f64>::zeros(7, 7);
        let mut rq_k = Mat::<f64>::zeros(7, 7);
        try_apply_periodic::<M, f64>(d.as_ref(), per_k.as_mut(), Triangle::Lower, PER)
            .expect("finite");
        try_apply_rq(d.as_ref(), rq_k.as_mut(), Triangle::Lower, RQ).expect("finite");
        let (per_k, rq_k) = if with_k {
            (Some(per_k.as_ref()), Some(rq_k.as_ref()))
        } else {
            (None, None)
        };
        let mut got = [0.0; 6];
        let v = try_weighted_periodic::<M, f64>(d.as_ref(), per_k, w.as_ref(), PER, &mut got[..2])
            .expect("finite")
            .expect("unit stride");
        got[2] = v;
        let v = try_weighted_rq(d.as_ref(), rq_k, w.as_ref(), RQ, &mut got[3..5])
            .expect("finite")
            .expect("unit stride");
        got[5] = v;
        for (g, e) in got.iter().zip(&want) {
            assert_rel(*g, *e, 1e-12);
        }
    }

    #[test]
    fn weighted_passes_match_the_scalar_formulas() {
        for with_k in [false, true] {
            weighted_matches::<Accurate>(with_k);
        }
        weighted_matches::<FastApprox>(false);
    }

    #[test]
    fn non_finite_distance_is_an_input_error() {
        let mut d = dist();
        d[(5, 2)] = f64::NAN;
        let mut out = Mat::<f64>::zeros(7, 7);
        assert_eq!(
            try_apply_periodic::<Accurate, f64>(d.as_ref(), out.as_mut(), Triangle::Lower, PER),
            Err(GprError::NonFiniteInput)
        );
        assert_eq!(
            try_apply_rq(d.as_ref(), out.as_mut(), Triangle::Lower, RQ),
            Err(GprError::NonFiniteInput)
        );
        let mut g = [0.0; 2];
        assert_eq!(
            try_weighted_rq(d.as_ref(), None, weight().as_ref(), RQ, &mut g),
            Err(GprError::NonFiniteInput)
        );
    }

    /// `f32` storage takes the same lanes, rounded once on the store.
    #[test]
    fn f32_grams_round_the_f64_lanes() {
        let d = dist();
        let d32 = Mat::<f32>::from_fn(7, 7, |i, j| d[(i, j)] as f32);
        let mut per = Mat::<f32>::zeros(7, 7);
        let mut rq = Mat::<f32>::zeros(7, 7);
        assert!(
            try_apply_periodic::<Accurate, f32>(d32.as_ref(), per.as_mut(), Triangle::Lower, PER)
                .expect("finite")
        );
        assert!(try_apply_rq(d32.as_ref(), rq.as_mut(), Triangle::Lower, RQ).expect("finite"));
        for j in 0..7 {
            for i in j..7 {
                let d = f64::from(d32[(i, j)]);
                assert_rel(f64::from(per[(i, j)]), periodic_ref::<Accurate>(d).0, 1e-6);
                assert_rel(f64::from(rq[(i, j)]), rq_ref(d).0, 1e-6);
            }
        }
    }

    const RBF: super::RbfScales = super::RbfScales {
        half_inv_ell_sq: 0.5 / (1.6 * 1.6),
        inv_ell_sq: 1.0 / (1.6 * 1.6),
    };

    fn rbf_ref<M: KernelMath>(d: f64, grad: bool) -> f64 {
        let jet = M::jet(-d * RBF.half_inv_ell_sq);
        if grad {
            jet.d1 * d * RBF.inv_ell_sq
        } else {
            jet.v
        }
    }

    /// `f32` RBF lanes are the `f64` formula rounded once.
    fn f32_rbf_matches<M: KernelMath>() {
        let d = dist();
        let d32 = Mat::<f32>::from_fn(7, 7, |i, j| d[(i, j)] as f32);
        for grad in [false, true] {
            for uplo in [Triangle::Lower, Triangle::Upper, Triangle::Full] {
                let mut out = Mat::<f32>::zeros(7, 7);
                assert!(
                    super::try_square_rbf::<M, f32>(d32.as_ref(), out.as_mut(), uplo, RBF, grad)
                        .expect("finite")
                );
                for j in 0..7 {
                    for i in 0..7 {
                        let inside = match uplo {
                            Triangle::Lower => i >= j,
                            Triangle::Upper => i <= j,
                            Triangle::Full => true,
                        };
                        let want = if inside {
                            rbf_ref::<M>(f64::from(d32[(i, j)]), grad) as f32
                        } else {
                            0.0
                        };
                        assert_rel(f64::from(out[(i, j)]), f64::from(want), 1e-6);
                    }
                }
            }
        }
        let rect = Mat::<f32>::from_fn(7, 5, |i, j| d32[(i, j + 2)]);
        let mut out = Mat::<f32>::zeros(7, 5);
        assert!(
            super::try_apply_rbf_cross::<M, f32>(rect.as_ref(), out.as_mut(), RBF).expect("finite")
        );
        for j in 0..5 {
            for i in 0..7 {
                let want = rbf_ref::<M>(f64::from(rect[(i, j)]), false);
                assert_rel(f64::from(out[(i, j)]), want, 1e-6);
            }
        }
    }

    #[test]
    fn f32_rbf_lanes_round_the_f64_formula() {
        f32_rbf_matches::<Accurate>();
        f32_rbf_matches::<FastApprox>();
    }

    #[test]
    fn f32_rbf_non_finite_distance_is_an_input_error() {
        let mut d = Mat::<f32>::from_fn(5, 5, |i, j| (i as f32 - j as f32).powi(2));
        d[(3, 1)] = f32::INFINITY;
        let mut out = Mat::<f32>::zeros(5, 5);
        assert_eq!(
            super::try_square_rbf::<Accurate, f32>(
                d.as_ref(),
                out.as_mut(),
                Triangle::Lower,
                RBF,
                false
            ),
            Err(GprError::NonFiniteInput)
        );
    }

    /// `f64` takes the same RBF lanes as `f32`, padded tail included.
    #[test]
    fn f64_rbf_slice_matches_scalar_within_tol() {
        use crate::math::MathOps;
        let dist: Vec<f64> = (0..10).map(|i| (i as f64) * 0.4).collect();
        let s = super::RbfScales {
            half_inv_ell_sq: 0.5,
            inv_ell_sq: 1.0,
        };
        let mut value = vec![0.0; 10];
        let mut grad = vec![0.0; 10];
        super::rbf_slice::<Accurate, f64>(&dist, &mut value, s, false).expect("finite");
        super::rbf_slice::<Accurate, f64>(&dist, &mut grad, s, true).expect("finite");
        for (i, &d) in dist.iter().enumerate() {
            let k = (-d * 0.5).exp();
            assert_rel(value[i], k, 1e-12);
            assert_rel(grad[i], k * d, 1e-12);
        }
        // `FastApprox` lanes are its scalar polynomial, bit for bit.
        super::rbf_slice::<FastApprox, f64>(&dist, &mut value, s, false).expect("finite");
        super::rbf_slice::<FastApprox, f64>(&dist, &mut grad, s, true).expect("finite");
        for (i, &d) in dist.iter().enumerate() {
            let jet = FastApprox::jet(-d * 0.5);
            assert_eq!(value[i].to_bits(), jet.v.to_bits());
            assert_eq!(grad[i].to_bits(), (jet.d1 * d).to_bits());
        }
    }

    #[test]
    fn f64_rbf_slice_rejects_nan() {
        let dist = [0.0, 1.0, f64::NAN, 3.0, 4.0];
        let mut out = [0.0; 5];
        let s = super::RbfScales {
            half_inv_ell_sq: 0.5,
            inv_ell_sq: 1.0,
        };
        for grad in [false, true] {
            assert_eq!(
                super::rbf_slice::<Accurate, f64>(&dist, &mut out, s, grad),
                Err(GprError::NonFiniteInput)
            );
            assert_eq!(
                super::rbf_slice::<FastApprox, f64>(&dist, &mut out, s, grad),
                Err(GprError::NonFiniteInput)
            );
        }
    }
}
