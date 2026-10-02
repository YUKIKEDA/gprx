//! `f64x4` paths of the Periodic and rational-quadratic leaves: the square
//! Gram from cached squared distances, and the one-pass weighted gradient.
//!
//! Both read column-major views with unit row stride; anything else returns
//! `Ok(false)` / `Ok(None)` and the caller keeps its scalar loop. `f32`
//! storage is widened to `f64` lanes and rounded once on the store.
//! `wide`'s `sin`, `cos`, `ln`, and `exp` may differ from libm by a few ULP.
//! The rational quadratic `u^{-α}` is `exp(−α ln u)` here.

use super::dist::{par_lower_blocks, par_lower_fold, worker_count};
use super::{KernelScalar, Triangle, require_square_pair};
use crate::error::GprError;
use crate::math::{KernelMath, f64x4_all_finite};
use faer::reborrow::ReborrowMut;
use faer::{MatMut, MatRef};
use wide::f64x4;

const LANES: usize = 4;

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

fn col_slice<T: KernelScalar>(mat: MatRef<'_, T>, col: usize) -> Option<&[T]> {
    mat.col(col).try_as_col_major().map(|c| c.as_slice())
}

fn col_slice_mut<T: KernelScalar>(mat: MatMut<'_, T>, col: usize) -> Option<&mut [T]> {
    mat.col_mut(col)
        .try_as_col_major_mut()
        .map(|c| c.as_slice_mut())
}

fn unit_row_stride<T: KernelScalar>(mat: MatRef<'_, T>) -> bool {
    mat.ncols() == 0 || col_slice(mat, 0).is_some()
}

/// Four lanes from `src[i..]`, padded with `pad` past the end.
#[inline(always)]
fn load<T: KernelScalar>(src: &[T], i: usize, pad: f64) -> f64x4 {
    let mut lanes = [pad; LANES];
    for (lane, value) in lanes.iter_mut().zip(&src[i..]) {
        *lane = value.to_f64();
    }
    f64x4::new(lanes)
}

#[inline(always)]
fn store<T: KernelScalar>(dest: &mut [T], i: usize, v: f64x4) {
    for (slot, value) in dest[i..].iter_mut().zip(v.to_array()) {
        *slot = T::from_f64(value);
    }
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

#[inline(always)]
fn checked_input(d: f64x4) -> Result<f64x4, GprError> {
    if f64x4_all_finite(d) {
        Ok(d)
    } else {
        Err(GprError::NonFiniteInput)
    }
}

#[inline(always)]
fn checked_kernel(v: f64x4) -> Result<f64x4, GprError> {
    if f64x4_all_finite(v) {
        Ok(v)
    } else {
        Err(GprError::NonFiniteKernelValue)
    }
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
    let rows = |col: usize| match uplo {
        Triangle::Lower => (col, n),
        Triangle::Upper => (0, col + 1),
        Triangle::Full => (0, n),
    };
    let column = |dist_col: usize, dest: Option<&mut [T]>| -> Result<(), GprError> {
        let (Some(src), Some(dest)) = (col_slice(dist, dist_col), dest) else {
            return Err(GprError::UnsupportedKernelOperation {
                reason: "expected unit row-stride for SIMD kernel".to_owned(),
            });
        };
        let (start, end) = rows(dist_col);
        f(&src[start..end], &mut dest[start..end])
    };
    if matches!(uplo, Triangle::Lower) {
        par_lower_blocks(out, worker_count(), &|start, mut part: MatMut<'_, T>| {
            for local in 0..part.ncols() {
                column(start + local, col_slice_mut(part.rb_mut(), local))?;
            }
            Ok::<(), GprError>(())
        })?;
    } else {
        for col in 0..n {
            column(col, col_slice_mut(out.rb_mut(), col))?;
        }
    }
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

#[inline(always)]
fn sum_lanes(v: f64x4) -> f64 {
    v.to_array().iter().sum()
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
}
