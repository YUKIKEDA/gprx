//! Defines the four-wide `f64` SIMD lanes of the kernel leaves, and the lane helpers they share.
//!
//! - [`stationary`]: the isotropic RBF, Periodic, and rational quadratic
//!   leaves from cached squared distances (square, every `uplo`, and the
//!   RBF rectangle and its rectangular `∂K/∂θ` from coordinates), and the
//!   one-pass weighted gradient of Periodic and RQ. Any storage: `f32` is
//!   widened to `f64` lanes and rounded once on the store.
//! - [`rbf_ard`]: the ARD RBF value and `∂K/∂θ_d`, from coordinates or the
//!   packed `(Δx_d)²` cache, square and rectangular.
//! - [`ard`]: the ARD Matérn and ARD RQ value and `θ` derivative, through a
//!   [`ard::Profile`] that turns `r²` into four values.
//! - [`dist`]: the row loops of squared distances and `(Δx_d)²`.
//!
//! The coordinate derivatives of the radial leaves are scalar
//! ([`super::radial::Radial`]); the isotropic Matérn and the rectangular and
//! coordinate paths of Periodic / RQ are scalar too.
//!
//! Every loop reads column-major views with unit row stride. A view whose
//! stride is not 1 makes it report that it did not run (`Ok(false)` /
//! `false` / `Ok(None)`), and the caller runs its scalar loop, which also
//! names the error of a value that is not finite. `wide`'s `exp`, `ln`,
//! `sin`, and `cos` may differ from the language math library by a few ULP.

pub(crate) mod ard;
pub(crate) mod dist;
pub(crate) mod rbf_ard;
pub(crate) mod rows;
pub(crate) mod stationary;

use super::KernelScalar;
use crate::error::GprError;
use faer::{ColMut, MatMut, MatRef};
use wide::f64x4;

/// Lanes of [`f64x4`].
pub(crate) const LANES: usize = 4;

pub(crate) use crate::math::f64x4_all_finite as all_finite;

/// Whether every value is a valid squared distance: finite and not
/// negative. Eight lanes keep two sums without a branch: `v · 0`, which
/// stays `0` unless a value is `NaN` or infinite, and the least value,
/// which stays `≥ 0` (`-0.0` included) unless one is negative. The loop is
/// compiled for the widest SIMD the CPU has (`pulp`'s dispatch, as faer's
/// kernels are), since it reads every supplied value of a call.
pub(crate) fn all_valid_distances(values: &[f64]) -> bool {
    faer_traits::pulp::Arch::new().dispatch(ValidLanes(values))
}

/// [`valid_lanes`] as a `pulp` op: its body is inlined into the function
/// `pulp` compiles for the CPU's features.
struct ValidLanes<'a>(&'a [f64]);

impl faer_traits::pulp::WithSimd for ValidLanes<'_> {
    type Output = bool;

    #[inline(always)]
    fn with_simd<S: faer_traits::pulp::Simd>(self, _simd: S) -> bool {
        valid_lanes(self.0)
    }
}

#[inline(always)]
fn valid_lanes(values: &[f64]) -> bool {
    let mut zero = [0.0f64; 8];
    let mut least = [0.0f64; 8];
    let (chunks, rest) = values.as_chunks::<8>();
    for chunk in chunks {
        for ((z, l), &v) in zero.iter_mut().zip(&mut least).zip(chunk) {
            *z += v * 0.0;
            *l = if v < *l { v } else { *l };
        }
    }
    let rest = rest
        .iter()
        .fold(true, |ok, &v| ok & super::sources::valid(v));
    let sum: f64 = zero.iter().sum();
    let min = least.iter().fold(0.0f64, |m, &l| if l < m { l } else { m });
    // An exact test, not a tolerance: a sum of `v · 0` is `0` exactly for
    // finite values and `NaN` otherwise (`float_cmp` allows a literal `0`).
    rest & (sum == 0.0) & (min >= 0.0)
}

/// Columns of one band of a square check ([`square_band`], [`pack_columns`]).
pub(crate) const SQUARE_BAND: usize = 64;

/// Rows ahead whose mirrors [`mirrors_v3`] fetches into cache.
#[cfg(target_arch = "x86_64")]
const MIRROR_AHEAD: usize = 8;

/// Columns compared at a time inside a band ([`band_mirrors`]).
const SUB_BAND: usize = 8;

/// Checks the columns `j0..j1` of the `n × n` column-major square `block`
/// exactly: each lower run (rows `j..n` of column `j`) finite and
/// non-negative, a zero diagonal, each entry equal to its mirror. An
/// invalid value above the diagonal fails its mirror ([`band_mirrors`]).
/// Compiled for the widest SIMD the CPU has (`pulp`'s dispatch).
pub(crate) fn square_band(block: &[f64], n: usize, (j0, j1): (usize, usize)) -> bool {
    faer_traits::pulp::Arch::new().dispatch(SquareBand {
        block,
        n,
        cols: (j0, j1),
    })
}

/// [`square_band`] as a `pulp` op.
struct SquareBand<'a> {
    block: &'a [f64],
    n: usize,
    cols: (usize, usize),
}

impl faer_traits::pulp::WithSimd for SquareBand<'_> {
    type Output = bool;

    #[inline(always)]
    fn with_simd<S: faer_traits::pulp::Simd>(self, _simd: S) -> bool {
        let Self {
            block,
            n,
            cols: (j0, j1),
        } = self;
        let mut ok = true;
        for j in j0..j1 {
            let lower = &block[j * n + j..(j + 1) * n];
            // The diagonal is exactly zero (`float_cmp` allows a literal `0`).
            ok &= lower[0] == 0.0;
            ok &= valid_lanes(lower);
        }
        ok & band_mirrors(block, n, (j0, j1))
    }
}

/// Where [`pack_columns`] puts the lower runs of its columns.
pub(crate) enum SquareOut<'a, T> {
    /// Over a slice of their exact length.
    Over(&'a mut [T]),
    /// At the end of a vector whose capacity holds it, so no buffer is
    /// zeroed first.
    Push(&'a mut Vec<T>),
}

/// Checks the columns `j0..j1` of the `n × n` column-major square `block`
/// exactly, as [`square_band`] does, and packs their lower runs into `out`
/// in order, [`SQUARE_BAND`] columns at a time: a band's runs are copied
/// (and checked in the same pass), then compared with their mirrors while
/// they are still in cache.
pub(crate) fn pack_columns<T: KernelScalar>(
    block: &[f64],
    n: usize,
    (j0, j1): (usize, usize),
    out: SquareOut<'_, T>,
) -> bool {
    faer_traits::pulp::Arch::new().dispatch(PackColumns {
        block,
        n,
        cols: (j0, j1),
        out,
    })
}

/// [`pack_columns`] as a `pulp` op.
struct PackColumns<'a, T> {
    block: &'a [f64],
    n: usize,
    cols: (usize, usize),
    out: SquareOut<'a, T>,
}

impl<T: KernelScalar> faer_traits::pulp::WithSimd for PackColumns<'_, T> {
    type Output = bool;

    #[inline(always)]
    fn with_simd<S: faer_traits::pulp::Simd>(self, _simd: S) -> bool {
        let Self {
            block,
            n,
            cols: (j0, j1),
            mut out,
        } = self;
        let mut at = 0;
        let mut ok = true;
        for k0 in (j0..j1).step_by(SQUARE_BAND) {
            let k1 = (k0 + SQUARE_BAND).min(j1);
            for j in k0..k1 {
                let lower = &block[j * n + j..(j + 1) * n];
                // The diagonal is exactly zero (`float_cmp` allows a literal `0`).
                ok &= lower[0] == 0.0;
                ok &= match &mut out {
                    SquareOut::Over(dest) => copy_valid(lower, &mut dest[at..at + lower.len()]),
                    SquareOut::Push(dest) => {
                        dest.extend(lower.iter().map(|&v| T::from_f64(v)));
                        valid_lanes(lower)
                    }
                };
                at += lower.len();
            }
            ok &= band_mirrors(block, n, (k0, k1));
        }
        ok
    }
}

/// Whether every entry below the diagonal in the columns `k0..k1` equals
/// its mirror. Inside the band, [`SUB_BAND`] columns at a time against the
/// rows of the band below them, and each small diagonal block pair by
/// pair; below the band, all its columns against the rows `k1..n`
/// ([`mirrors`]).
#[inline(always)]
fn band_mirrors(block: &[f64], n: usize, (k0, k1): (usize, usize)) -> bool {
    let mut ok = true;
    for s0 in (k0..k1).step_by(SUB_BAND) {
        let s1 = (s0 + SUB_BAND).min(k1);
        for j in s0..s1 {
            for i in j + 1..s1 {
                ok &= super::sources::same(block[i + j * n], block[j + i * n]);
            }
        }
        ok &= mirrors(block, n, (s0, s1), (s1, k1));
    }
    ok & mirrors(block, n, (k0, k1), (k1, n))
}

/// Whether the rows `r0..r1` of the columns `j0..j1` (`r0 ≥ j1`) equal
/// their mirrors, the rows `j0..j1` of the columns `r0..r1`. Four rows by
/// four columns at a time: the lower values of a row quad are four short
/// contiguous runs, its mirrors four contiguous runs of the band's width
/// (one column of the square each), transposed in registers. Going down
/// the rows a quad at a time keeps the band's few lines per column in
/// cache. `(a − b) · 0` stays `0` unless a mirror is `NaN` or infinite,
/// and `|a − b|` stays `0` unless a pair differs. On x86-64 with AVX2 a
/// quad is transposed with eight shuffles ([`mirrors_v3`]); elsewhere the
/// compiler lowers the transposes of [`mirrors_lanes`].
#[inline(always)]
fn mirrors(block: &[f64], n: usize, cols: (usize, usize), rows: (usize, usize)) -> bool {
    #[cfg(target_arch = "x86_64")]
    if let Some(simd) = faer_traits::pulp::x86::V3::try_new() {
        return simd.vectorize(|| mirrors_v3(simd, block, n, cols, rows));
    }
    mirrors_lanes(block, n, cols, rows)
}

/// [`mirrors`] with AVX2 shuffles, inlined into `pulp`'s AVX2 function.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn mirrors_v3(
    simd: faer_traits::pulp::x86::V3,
    block: &[f64],
    n: usize,
    (j0, j1): (usize, usize),
    (r0, r1): (usize, usize),
) -> bool {
    use core::arch::x86_64::__m256d;
    let avx = simd.avx;
    let load = |q: &[f64; 4]| -> __m256d { faer_traits::pulp::cast(*q) };
    let zeros = avx._mm256_setzero_pd();
    let sign = avx._mm256_set1_pd(-0.0);
    let mut zero = zeros;
    let mut most = zeros;
    let c4 = j0 + (j1 - j0) / 4 * 4;
    let i4 = r0 + r1.saturating_sub(r0) / 4 * 4;
    let mut i = r0;
    while i < i4 {
        // The mirrors of a row quad a few quads ahead are fetched into
        // cache while this one is compared: each is a short run in its own
        // page, which the hardware does not prefetch.
        for k in i + MIRROR_AHEAD..(i + MIRROR_AHEAD + 4).min(r1) {
            for at in (k * n + j0..k * n + c4).step_by(8) {
                if let Some(v) = block.get(at) {
                    simd.sse
                        ._mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(
                            std::ptr::from_ref(v).cast(),
                        );
                }
            }
        }
        // The mirrors of the row quad: rows `j0..c4` of the columns
        // `i..i + 4`, one contiguous run each.
        let run = |k: usize| block.get(k * n + j0..k * n + c4);
        let (Some(m0), Some(m1), Some(m2), Some(m3)) = (run(i), run(i + 1), run(i + 2), run(i + 3))
        else {
            return false;
        };
        let quads = m0
            .as_chunks::<4>()
            .0
            .iter()
            .zip(m1.as_chunks::<4>().0)
            .zip(m2.as_chunks::<4>().0)
            .zip(m3.as_chunks::<4>().0);
        for (q, (((a, b), c), d)) in quads.enumerate() {
            let (a, b, c, d) = (load(a), load(b), load(c), load(d));
            let ab_lo = avx._mm256_unpacklo_pd(a, b);
            let ab_hi = avx._mm256_unpackhi_pd(a, b);
            let cd_lo = avx._mm256_unpacklo_pd(c, d);
            let cd_hi = avx._mm256_unpackhi_pd(c, d);
            let mirror = [
                avx._mm256_permute2f128_pd::<0x20>(ab_lo, cd_lo),
                avx._mm256_permute2f128_pd::<0x20>(ab_hi, cd_hi),
                avx._mm256_permute2f128_pd::<0x31>(ab_lo, cd_lo),
                avx._mm256_permute2f128_pd::<0x31>(ab_hi, cd_hi),
            ];
            let col = j0 + 4 * q;
            for (m, mirror) in mirror.into_iter().enumerate() {
                let at = (col + m) * n + i;
                let Some(low) = block.get(at..at + 4).and_then(|s| s.first_chunk::<4>()) else {
                    return false;
                };
                let diff = avx._mm256_sub_pd(load(low), mirror);
                zero = avx._mm256_add_pd(zero, avx._mm256_mul_pd(diff, zeros));
                most = avx._mm256_max_pd(most, avx._mm256_andnot_pd(sign, diff));
            }
        }
        i += 4;
    }
    let zero: [f64; 4] = faer_traits::pulp::cast(zero);
    let most: [f64; 4] = faer_traits::pulp::cast(most);
    lanes_hold(&zero, &most) & mirror_tails(block, n, (j0, j1), (r0, r1), (c4, i4))
}

/// [`mirrors`] on four-value arrays, for the compiler to vectorize.
#[inline(always)]
fn mirrors_lanes(
    block: &[f64],
    n: usize,
    (j0, j1): (usize, usize),
    (r0, r1): (usize, usize),
) -> bool {
    let quad = |at: usize| -> [f64; 4] {
        block
            .get(at..at + 4)
            .and_then(|s| s.first_chunk::<4>())
            .copied()
            .unwrap_or([f64::NAN; 4])
    };
    let mut zero = [0.0f64; 4];
    let mut most = [0.0f64; 4];
    let c4 = j0 + (j1 - j0) / 4 * 4;
    let i4 = r0 + r1.saturating_sub(r0) / 4 * 4;
    let mut i = r0;
    while i < i4 {
        let mut c = j0;
        while c < c4 {
            let r = [
                quad(i * n + c),
                quad((i + 1) * n + c),
                quad((i + 2) * n + c),
                quad((i + 3) * n + c),
            ];
            for m in 0..4 {
                let low = quad((c + m) * n + i);
                let mirror = r.map(|row| row[m]);
                for (((z, top), &l), &v) in zero.iter_mut().zip(&mut most).zip(&low).zip(&mirror) {
                    let d = l - v;
                    *z += d * 0.0;
                    let d = d.abs();
                    *top = if d > *top { d } else { *top };
                }
            }
            c += 4;
        }
        i += 4;
    }
    lanes_hold(&zero, &most) & mirror_tails(block, n, (j0, j1), (r0, r1), (c4, i4))
}

/// Whether the folded lanes of [`mirrors`] say every pair was equal: each
/// `(a − b) · 0` summed to `0`, each `|a − b|` at most `0`.
#[inline(always)]
fn lanes_hold(zero: &[f64; 4], most: &[f64; 4]) -> bool {
    let sum: f64 = zero.iter().sum();
    let most = most.iter().fold(0.0f64, |m, &v| if v > m { v } else { m });
    // Exact tests, not tolerances (`float_cmp` allows a literal `0`).
    (sum == 0.0) & (most == 0.0)
}

/// The pairs [`mirrors`] leaves outside its quads, one by one: the
/// columns `c4..j1` of the rows `r0..i4`, and every column of the rows
/// `i4..r1`.
#[inline(always)]
fn mirror_tails(
    block: &[f64],
    n: usize,
    (j0, j1): (usize, usize),
    (r0, r1): (usize, usize),
    (c4, i4): (usize, usize),
) -> bool {
    let mut ok = true;
    for k in r0..r1 {
        let from = if k < i4 { c4 } else { j0 };
        for j in from..j1 {
            ok &= super::sources::same(block[k + j * n], block[j + k * n]);
        }
    }
    ok
}

/// Copies `src` into `dest` and reports whether every value is
/// [`super::sources::valid`], in one pass: the lanes of [`valid_lanes`]
/// fold while the values are stored.
#[inline(always)]
fn copy_valid<T: KernelScalar>(src: &[f64], dest: &mut [T]) -> bool {
    let mut zero = [0.0f64; 8];
    let mut least = [0.0f64; 8];
    let (chunks, rest) = src.as_chunks::<8>();
    let (out, out_rest) = dest.split_at_mut(chunks.len() * 8);
    for (chunk, out) in chunks.iter().zip(out.as_chunks_mut::<8>().0) {
        for (((z, l), o), &v) in zero.iter_mut().zip(&mut least).zip(out).zip(chunk) {
            *o = T::from_f64(v);
            *z += v * 0.0;
            *l = if v < *l { v } else { *l };
        }
    }
    let mut ok = true;
    for (o, &v) in out_rest.iter_mut().zip(rest) {
        *o = T::from_f64(v);
        ok &= super::sources::valid(v);
    }
    let sum: f64 = zero.iter().sum();
    let min = least.iter().fold(0.0f64, |m, &l| if l < m { l } else { m });
    // Exact tests, as in [`valid_lanes`].
    ok & (sum == 0.0) & (min >= 0.0)
}

/// Column `col` of `mat` as a slice, when `mat` is column-major.
#[inline(always)]
pub(crate) fn col_slice<T>(mat: MatRef<'_, T>, col: usize) -> Option<&[T]> {
    mat.col(col).try_as_col_major().map(|c| c.as_slice())
}

/// Column `col` of `mat` as a mutable slice, when `mat` is column-major.
#[inline(always)]
pub(crate) fn col_slice_mut<T>(mat: MatMut<'_, T>, col: usize) -> Option<&mut [T]> {
    mat.col_mut(col)
        .try_as_col_major_mut()
        .map(|c| c.as_slice_mut())
}

/// [`col_slice`] of a view the caller checked with [`unit_row_stride`].
pub(crate) fn col_slice_checked<T>(mat: MatRef<'_, T>, col: usize) -> Result<&[T], GprError> {
    col_slice(mat, col).ok_or_else(not_unit_stride)
}

/// [`col_slice_mut`] of a view the caller checked with [`unit_row_stride`].
pub(crate) fn col_slice_mut_checked<T>(
    mat: MatMut<'_, T>,
    col: usize,
) -> Result<&mut [T], GprError> {
    col_slice_mut(mat, col).ok_or_else(not_unit_stride)
}

/// The rows of a column view as a slice, when they are contiguous (always,
/// for a column of a view the caller checked with [`unit_row_stride`]).
pub(crate) fn rows_checked<T>(rows: ColMut<'_, T>) -> Result<&mut [T], GprError> {
    rows.try_as_col_major_mut()
        .map(|c| c.as_slice_mut())
        .ok_or_else(not_unit_stride)
}

fn not_unit_stride() -> GprError {
    GprError::UnsupportedKernelOperation {
        reason: "expected unit row-stride for SIMD kernel".to_owned(),
    }
}

/// Whether `mat`'s columns are contiguous (an empty view counts).
pub(crate) fn unit_row_stride<T>(mat: MatRef<'_, T>) -> bool {
    mat.ncols() == 0 || col_slice(mat, 0).is_some()
}

/// Four lanes from `src[i..i + 4]`. `f64` is one array load.
#[inline(always)]
pub(crate) fn load4<T: KernelScalar>(src: &[T], i: usize) -> f64x4 {
    let l = &src[i..i + LANES];
    if let Some(Ok(lanes)) = T::as_f64_slice(l).map(<[f64; LANES]>::try_from) {
        return f64x4::from(lanes);
    }
    f64x4::new([l[0].to_f64(), l[1].to_f64(), l[2].to_f64(), l[3].to_f64()])
}

/// `v` into `dest[i..i + 4]`, each lane rounded once into `T`. `f64` is one
/// array store.
#[inline(always)]
pub(crate) fn store4<T: KernelScalar>(dest: &mut [T], i: usize, v: f64x4) {
    let a = v.to_array();
    let l = &mut dest[i..i + LANES];
    if let Some(f) = T::as_f64_slice_mut(l) {
        f.copy_from_slice(&a);
        return;
    }
    l[0] = T::from_f64(a[0]);
    l[1] = T::from_f64(a[1]);
    l[2] = T::from_f64(a[2]);
    l[3] = T::from_f64(a[3]);
}

/// Four lanes from `src[i..]`, padded with `pad` past the end.
#[inline(always)]
pub(crate) fn load<T: KernelScalar>(src: &[T], i: usize, pad: f64) -> f64x4 {
    if let Some(l) = src.get(i..i + LANES) {
        return f64x4::new([l[0].to_f64(), l[1].to_f64(), l[2].to_f64(), l[3].to_f64()]);
    }
    let mut lanes = [pad; LANES];
    for (lane, value) in lanes.iter_mut().zip(&src[i..]) {
        *lane = value.to_f64();
    }
    f64x4::new(lanes)
}

/// The lanes of `v` that fall inside `dest[i..]`.
#[inline(always)]
pub(crate) fn store<T: KernelScalar>(dest: &mut [T], i: usize, v: f64x4) {
    let a = v.to_array();
    if let Some(l) = dest.get_mut(i..i + LANES) {
        l[0] = T::from_f64(a[0]);
        l[1] = T::from_f64(a[1]);
        l[2] = T::from_f64(a[2]);
        l[3] = T::from_f64(a[3]);
        return;
    }
    for (slot, value) in dest[i..].iter_mut().zip(a) {
        *slot = T::from_f64(value);
    }
}

/// `d` when every lane is finite, else [`GprError::NonFiniteInput`].
#[inline(always)]
pub(crate) fn checked_input(d: f64x4) -> Result<f64x4, GprError> {
    if all_finite(d) {
        Ok(d)
    } else {
        Err(GprError::NonFiniteInput)
    }
}

/// `v` when every lane is finite, else [`GprError::NonFiniteKernelValue`].
#[inline(always)]
pub(crate) fn checked_kernel(v: f64x4) -> Result<f64x4, GprError> {
    if all_finite(v) {
        Ok(v)
    } else {
        Err(GprError::NonFiniteKernelValue)
    }
}

/// [`GprError::NonFiniteInput`] when a value of `values` is not finite.
pub(crate) fn finite_slice<T: KernelScalar>(values: &[T]) -> Result<(), GprError> {
    if values.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(GprError::NonFiniteInput)
    }
}

/// The sum of the four lanes.
#[inline(always)]
pub(crate) fn sum_lanes(v: f64x4) -> f64 {
    v.to_array().iter().sum()
}

/// Adds `(x[i] − x0)²` into `acc[i]`, in `f64`.
pub(crate) fn add_squared_diff<T: KernelScalar>(x: &[T], x0: f64, acc: &mut [f64]) {
    debug_assert_eq!(x.len(), acc.len());
    let x0v = f64x4::splat(x0);
    let mut i = 0;
    while i + LANES <= x.len() {
        let d = load4(x, i) - x0v;
        store4(acc, i, load4(acc, i) + d * d);
        i += LANES;
    }
    while i < x.len() {
        let d = x[i].to_f64() - x0;
        acc[i] += d * d;
        i += 1;
    }
}

/// Adds `scale · (x[i] − x0)²` into `acc[i]`, in `f64`.
pub(crate) fn add_squared_diff_scaled<T: KernelScalar>(
    x: &[T],
    x0: f64,
    scale: f64,
    acc: &mut [f64],
) {
    debug_assert_eq!(x.len(), acc.len());
    let x0v = f64x4::splat(x0);
    let sv = f64x4::splat(scale);
    let mut i = 0;
    while i + LANES <= x.len() {
        let d = load4(x, i) - x0v;
        store4(acc, i, load4(acc, i) + d * d * sv);
        i += LANES;
    }
    while i < x.len() {
        let d = x[i].to_f64() - x0;
        acc[i] += d * d * scale;
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::{
        SQUARE_BAND, SquareOut, add_squared_diff, add_squared_diff_scaled, load, mirrors_lanes,
        pack_columns, square_band, store,
    };
    use crate::kernel::sources::{same, valid};
    use crate::test_check::assert_close;

    const TOL: f64 = 1e-12;

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

    /// A partial tail is padded on load and only its own lanes are stored.
    #[test]
    fn padded_lanes_touch_only_the_tail() {
        let src = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let v = load(&src, 4, -9.0).to_array();
        assert_eq!(bits(&v), bits(&[5.0, 6.0, -9.0, -9.0]));
        let mut dest = [0.0f32; 6];
        store(&mut dest, 4, wide::f64x4::new([7.0, 8.0, 99.0, 99.0]));
        let dest: Vec<f64> = dest.iter().map(|&x| f64::from(x)).collect();
        assert_eq!(bits(&dest), bits(&[0.0, 0.0, 0.0, 0.0, 7.0, 8.0]));
    }

    /// A symmetric `n × n` square of squared differences, zero diagonal.
    fn square(n: usize) -> Vec<f64> {
        let x: Vec<f64> = (0..n).map(|i| ((i * 37) % 101) as f64 / 13.0).collect();
        let mut block = vec![0.0; n * n];
        for j in 0..n {
            for i in 0..n {
                let t = x[i] - x[j];
                block[i + j * n] = t * t;
            }
        }
        block
    }

    /// The exact check, pair by pair.
    fn reference(block: &[f64], n: usize) -> bool {
        (0..n).all(|j| {
            block[j + j * n] == 0.0
                && (j..n)
                    .all(|i| valid(block[i + j * n]) && same(block[i + j * n], block[j + i * n]))
        })
    }

    /// Every way a square is read: band by band, packed onto the end of a
    /// vector (`f64` and `f32`), and packed over slices band range by band
    /// range. Each agrees with [`reference`], and a valid square's packed
    /// runs are its lower triangle.
    fn check_all(block: &[f64], n: usize) {
        let expected = reference(block, n);
        let bands: Vec<(usize, usize)> = (0..n)
            .step_by(SQUARE_BAND)
            .map(|j0| (j0, (j0 + SQUARE_BAND).min(n)))
            .collect();
        let by_band = bands.iter().all(|&cols| square_band(block, n, cols));
        assert_eq!(by_band, expected, "square_band, n = {n}");
        let lower: Vec<f64> = (0..n)
            .flat_map(|j| block[j * n + j..(j + 1) * n].iter().copied())
            .collect();
        let mut pushed: Vec<f64> = vec![-1.0];
        assert_eq!(
            pack_columns(block, n, (0, n), SquareOut::Push(&mut pushed)),
            expected,
            "push, n = {n}"
        );
        let mut narrow: Vec<f32> = Vec::new();
        assert_eq!(
            pack_columns(block, n, (0, n), SquareOut::Push(&mut narrow)),
            expected,
            "push f32, n = {n}"
        );
        let mut over = vec![0.0f64; lower.len()];
        let mut at = 0;
        let mut ok = true;
        for &(j0, j1) in &bands {
            let len: usize = (j0..j1).map(|j| n - j).sum();
            ok &= pack_columns(block, n, (j0, j1), SquareOut::Over(&mut over[at..at + len]));
            at += len;
        }
        assert_eq!(ok, expected, "over, n = {n}");
        if expected {
            assert_eq!(pushed[1..], lower[..], "push, n = {n}");
            assert_eq!(over, lower, "over, n = {n}");
            let narrow: Vec<f64> = narrow.iter().map(|&v| f64::from(v)).collect();
            let rounded: Vec<f64> = lower.iter().map(|&v| f64::from(v as f32)).collect();
            assert_eq!(narrow, rounded, "push f32, n = {n}");
        }
    }

    /// Valid squares of every shape the bands and quads split unevenly,
    /// and each one broken at one entry: below the diagonal (a different
    /// value, `NaN`, `∞`, a negative mirror pair), above it (a different
    /// value, `NaN`, `-∞`), and on it.
    #[test]
    fn square_checks_match_the_pairwise_check() {
        let sizes = [1, 2, 3, 4, 5, 7, 8, 9, 12, 13, 63, 64, 65, 67, 71, 129, 130];
        for n in sizes {
            let block = square(n);
            check_all(&block, n);
            let mut picks: Vec<(usize, usize)> = vec![(n - 1, 0), (n / 2, n / 3), (n - 1, n - 1)];
            for j in [0, 3, 7, 8, 9, 63, 64, 65, n.saturating_sub(2)] {
                for i in [j + 1, j + 4, j + 8, j + 64, n - 1] {
                    if j < n && i < n && i > j {
                        picks.push((i, j));
                    }
                }
            }
            for (i, j) in picks {
                let base = block[i + j * n];
                let lower = [base + 0.5, f64::NAN, f64::INFINITY];
                for v in lower {
                    let mut broken = block.clone();
                    broken[i + j * n] = v;
                    check_all(&broken, n);
                    if i != j {
                        let mut broken = block.clone();
                        broken[j + i * n] = v;
                        check_all(&broken, n);
                    }
                }
                if i != j {
                    let mut broken = block.clone();
                    broken[j + i * n] = f64::NEG_INFINITY;
                    check_all(&broken, n);
                    // A negative pair agrees with its mirror and still fails.
                    let mut broken = block.clone();
                    broken[i + j * n] = -1.0;
                    broken[j + i * n] = -1.0;
                    check_all(&broken, n);
                }
            }
            let mut broken = block.clone();
            broken[(n / 2) * (n + 1)] = 0.25;
            check_all(&broken, n);
        }
    }

    /// The array fallback of the mirror check agrees with the pairwise
    /// check, quads and tails.
    #[test]
    fn mirror_lanes_match_the_pairwise_check() {
        for n in [9, 13, 67] {
            let block = square(n);
            let pairs = |b: &[f64], cols: (usize, usize), rows: (usize, usize)| {
                (cols.0..cols.1).all(|j| (rows.0..rows.1).all(|i| same(b[i + j * n], b[j + i * n])))
            };
            for (cols, rows) in [((0, 4), (4, n)), ((0, 3), (3, n)), ((1, 5), (6, n - 1))] {
                assert!(mirrors_lanes(&block, n, cols, rows));
                let mut broken = block.clone();
                broken[(rows.1 - 1) + cols.0 * n] += 1.0;
                assert_eq!(
                    mirrors_lanes(&broken, n, cols, rows),
                    pairs(&broken, cols, rows)
                );
                let mut broken = block.clone();
                broken[cols.1 - 1 + rows.0 * n] = f64::NAN;
                assert_eq!(
                    mirrors_lanes(&broken, n, cols, rows),
                    pairs(&broken, cols, rows)
                );
                assert!(!mirrors_lanes(&broken, n, cols, rows));
            }
        }
    }
}
