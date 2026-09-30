//! Kernel `exp` for [`Accurate`] and [`FastApprox`].
//!
//! [`Accurate`] is `f64::exp` / `f32::exp` / `wide::exp`. [`FastApprox`]
//! evaluates one Taylor polynomial in the storage scalar. Hyperparameter
//! `exp(θ)` does not use this module.

use wide::{CmpEq, CmpGt, CmpLe, CmpLt, f64x4, i64x4};

/// Value and the first two derivatives of the kernel `exp` approximation.
#[derive(Clone, Copy, Debug)]
pub struct ExpJet<T> {
    pub v: T,
    pub d1: T,
    pub d2: T,
}

/// Exact libm / SIMD `exp`. The omitted math-mode parameter.
///
/// # Examples
///
/// ```rust
/// use gprx::Accurate;
///
/// let _mode = Accurate;
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct Accurate;

/// Polynomial `exp` for kernel evaluation, including `fit`.
///
/// Coefficients are the degree-7 Taylor terms of `exp` after range reduction.
/// f64 nonzero values stay within a relative `2^{-23}` of `f64::exp`. f32
/// evaluates those coefficients in `f32` and stays within a relative
/// `8 · u_f32`.
///
/// # Examples
///
/// ```rust
/// use gprx::FastApprox;
///
/// let _mode = FastApprox;
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct FastApprox;

/// Kernel `exp` mode: [`Accurate`] or [`FastApprox`].
///
/// This is the `M` parameter of the [`crate::kernel::CompiledKernel`]
/// evaluators. Models take the runtime [`crate::KernelExp`] instead
/// (`with_math`) and dispatch to one of these once per kernel call. It is sealed: only those two
/// types implement it, and its operations are crate-private.
///
/// # Examples
///
/// ```rust
/// use gprx::{Accurate, FastApprox, KernelMath};
///
/// fn mode_name<M: KernelMath>() -> &'static str {
///     std::any::type_name::<M>()
/// }
///
/// assert!(mode_name::<Accurate>().ends_with("Accurate"));
/// assert!(mode_name::<FastApprox>().ends_with("FastApprox"));
/// ```
pub trait KernelMath: MathOps {}

impl KernelMath for Accurate {}
impl KernelMath for FastApprox {}

/// Kernel `exp` and its derivatives with respect to the exponent.
///
/// `pub` in a private module: nameable only inside the crate, so
/// [`KernelMath`] stays sealed.
pub trait MathOps: Copy + Send + Sync + 'static {
    /// `true` keeps the libm `exp` algebra. `false` differentiates the polynomial.
    const ACCURATE: bool;

    /// `exp(x)` in the compute scalar.
    fn exp<T: crate::kernel::KernelScalar>(x: T) -> T;

    /// `exp(x)`, `d/dx`, and `d²/dx²` in the compute scalar.
    fn jet<T: crate::kernel::KernelScalar>(x: T) -> ExpJet<T>;

    /// Four `f64` exponents. Value only.
    fn exp_f64x4(x: f64x4) -> f64x4;

    /// Four derivatives of the kernel `exp` approximation.
    fn d1_f64x4(x: f64x4) -> f64x4;
}

impl MathOps for Accurate {
    const ACCURATE: bool = true;

    #[inline(always)]
    fn exp<T: crate::kernel::KernelScalar>(x: T) -> T {
        x.exp()
    }

    #[inline(always)]
    fn jet<T: crate::kernel::KernelScalar>(x: T) -> ExpJet<T> {
        let e = x.exp();
        ExpJet { v: e, d1: e, d2: e }
    }

    #[inline(always)]
    fn exp_f64x4(x: f64x4) -> f64x4 {
        x.exp()
    }

    #[inline(always)]
    fn d1_f64x4(x: f64x4) -> f64x4 {
        x.exp()
    }
}

impl MathOps for FastApprox {
    const ACCURATE: bool = false;

    #[inline(always)]
    fn exp<T: crate::kernel::KernelScalar>(x: T) -> T {
        x.fast_exp()
    }

    #[inline(always)]
    fn jet<T: crate::kernel::KernelScalar>(x: T) -> ExpJet<T> {
        x.fast_jet()
    }

    #[inline(always)]
    fn exp_f64x4(x: f64x4) -> f64x4 {
        fast_exp_f64x4(x)
    }

    #[inline(always)]
    fn d1_f64x4(x: f64x4) -> f64x4 {
        fast_d1_f64x4(x)
    }
}

const DEGREE: usize = 7;

const fn taylor_f64() -> [f64; DEGREE + 1] {
    let mut c = [0.0; DEGREE + 1];
    c[0] = 1.0;
    let mut fact = 1.0;
    let mut k = 1;
    while k <= DEGREE {
        fact *= k as f64;
        c[k] = 1.0 / fact;
        k += 1;
    }
    c
}

const fn taylor_f32() -> [f32; DEGREE + 1] {
    let mut c = [0.0; DEGREE + 1];
    c[0] = 1.0;
    let mut fact = 1.0_f32;
    let mut k = 1;
    while k <= DEGREE {
        fact *= k as f32;
        c[k] = 1.0 / fact;
        k += 1;
    }
    c
}

const TAYLOR_F64: [f64; DEGREE + 1] = taylor_f64();
const TAYLOR_F32: [f32; DEGREE + 1] = taylor_f32();

// Extra digits select the Cody–Waite split. Truncating changes the rounded `f64`.
#[allow(clippy::excessive_precision)]
const LN2_HI: f64 = 6.931_471_803_691_238_164_90e-1;
#[allow(clippy::excessive_precision)]
const LN2_LO: f64 = 1.908_214_929_270_587_700_02e-10;

/// `ln(2)` split used by fdlibm `expf`, exact `f32` values.
const LN2_HI_F32: f32 = f32::from_bits(0x3f31_7180);
const LN2_LO_F32: f32 = f32::from_bits(0x3717_f7d1);
const LOG2_E_F32: f32 = f32::from_bits(0x3fb8_aa3b);

#[inline(always)]
fn horner<T>(g: T, c: &[T]) -> T
where
    T: Copy + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    let mut acc = c[c.len() - 1];
    let mut i = c.len() - 1;
    while i > 0 {
        i -= 1;
        acc = acc * g + c[i];
    }
    acc
}

#[inline(always)]
fn pow2_f64(n: i32) -> f64 {
    if n <= -1075 {
        return 0.0;
    }
    if n >= 1024 {
        return f64::INFINITY;
    }
    if n >= -1022 {
        return f64::from_bits(((n as i64 + 1023) as u64) << 52);
    }
    f64::from_bits(1_u64 << (n + 1074))
}

#[inline(always)]
fn pow2_f32(n: i32) -> f32 {
    if n <= -150 {
        return 0.0;
    }
    if n >= 128 {
        return f32::INFINITY;
    }
    if n >= -126 {
        return f32::from_bits(((n + 127) as u32) << 23);
    }
    f32::from_bits(1_u32 << (n + 149))
}

#[inline(always)]
fn reduce_f64(x: f64) -> (f64, f64) {
    let n = (x * std::f64::consts::LOG2_E).round();
    let g = (x - n * LN2_HI) - n * LN2_LO;
    (g, pow2_f64(n as i32))
}

#[inline(always)]
fn reduce_f32(x: f32) -> (f32, f32) {
    let n = (x * LOG2_E_F32).round();
    let g = (x - n * LN2_HI_F32) - n * LN2_LO_F32;
    (g, pow2_f32(n as i32))
}

#[inline(always)]
pub(crate) fn fast_exp_f64(x: f64) -> f64 {
    if !x.is_finite() {
        return x.exp();
    }
    let (g, scale) = reduce_f64(x);
    horner(g, &TAYLOR_F64) * scale
}

#[inline(always)]
pub(crate) fn fast_jet_f64(x: f64) -> ExpJet<f64> {
    if !x.is_finite() {
        let e = x.exp();
        return ExpJet { v: e, d1: e, d2: e };
    }
    let (g, scale) = reduce_f64(x);
    ExpJet {
        v: horner(g, &TAYLOR_F64) * scale,
        d1: horner(g, &TAYLOR_F64[..=DEGREE - 1]) * scale,
        d2: horner(g, &TAYLOR_F64[..=DEGREE - 2]) * scale,
    }
}

#[inline(always)]
pub(crate) fn fast_exp_f32(x: f32) -> f32 {
    if !x.is_finite() {
        return x.exp();
    }
    let (g, scale) = reduce_f32(x);
    horner(g, &TAYLOR_F32) * scale
}

#[inline(always)]
pub(crate) fn fast_jet_f32(x: f32) -> ExpJet<f32> {
    if !x.is_finite() {
        let e = x.exp();
        return ExpJet { v: e, d1: e, d2: e };
    }
    let (g, scale) = reduce_f32(x);
    ExpJet {
        v: horner(g, &TAYLOR_F32) * scale,
        d1: horner(g, &TAYLOR_F32[..=DEGREE - 1]) * scale,
        d2: horner(g, &TAYLOR_F32[..=DEGREE - 2]) * scale,
    }
}

#[inline(always)]
fn bitcast_f64x4_to_i64x4(v: f64x4) -> i64x4 {
    // SAFETY: both are 32-byte, 32-aligned vectors of four lanes.
    unsafe { std::mem::transmute(v) }
}

#[inline(always)]
fn bitcast_i64x4_to_f64x4(v: i64x4) -> f64x4 {
    // SAFETY: both are 32-byte, 32-aligned vectors of four lanes.
    unsafe { std::mem::transmute(v) }
}

/// True when every lane is finite.
#[inline(always)]
pub(crate) fn f64x4_all_finite(v: f64x4) -> bool {
    // `NaN` fails equality. `±Inf` is the only value whose magnitude exceeds `f64::MAX`.
    v.cmp_eq(v).all() && v.abs().cmp_le(f64x4::splat(f64::MAX)).all()
}

/// `2^n` for an integer-valued `n`, matching [`pow2_f64`] on normal exponents.
#[inline(always)]
fn pow2_f64x4(n: f64x4) -> f64x4 {
    let magic = n + f64x4::splat(1023.0 + 4_503_599_627_370_496.0);
    let normal = bitcast_i64x4_to_f64x4(bitcast_f64x4_to_i64x4(magic) << 52);
    // Subnormals flush to zero. Their magnitude is below the `2^{-23}` absolute bound.
    let too_small = n.cmp_lt(f64x4::splat(-1022.0));
    let too_big = n.cmp_gt(f64x4::splat(1023.0));
    let scale = too_small.blend(f64x4::ZERO, normal);
    too_big.blend(f64x4::splat(f64::INFINITY), scale)
}

#[inline(always)]
fn reduce_f64x4(x: f64x4) -> (f64x4, f64x4) {
    let n = (x * f64x4::splat(std::f64::consts::LOG2_E)).round();
    let g = (x - n * f64x4::splat(LN2_HI)) - n * f64x4::splat(LN2_LO);
    (g, pow2_f64x4(n))
}

#[inline(always)]
fn horner_value(g: f64x4) -> f64x4 {
    let mut acc = f64x4::splat(TAYLOR_F64[7]);
    acc = acc.mul_add(g, f64x4::splat(TAYLOR_F64[6]));
    acc = acc.mul_add(g, f64x4::splat(TAYLOR_F64[5]));
    acc = acc.mul_add(g, f64x4::splat(TAYLOR_F64[4]));
    acc = acc.mul_add(g, f64x4::splat(TAYLOR_F64[3]));
    acc = acc.mul_add(g, f64x4::splat(TAYLOR_F64[2]));
    acc = acc.mul_add(g, f64x4::splat(TAYLOR_F64[1]));
    acc.mul_add(g, f64x4::splat(TAYLOR_F64[0]))
}

#[inline(always)]
fn horner_d1(g: f64x4) -> f64x4 {
    let mut acc = f64x4::splat(TAYLOR_F64[6]);
    acc = acc.mul_add(g, f64x4::splat(TAYLOR_F64[5]));
    acc = acc.mul_add(g, f64x4::splat(TAYLOR_F64[4]));
    acc = acc.mul_add(g, f64x4::splat(TAYLOR_F64[3]));
    acc = acc.mul_add(g, f64x4::splat(TAYLOR_F64[2]));
    acc = acc.mul_add(g, f64x4::splat(TAYLOR_F64[1]));
    acc.mul_add(g, f64x4::splat(TAYLOR_F64[0]))
}

#[inline(always)]
fn fast_exp_f64x4(x: f64x4) -> f64x4 {
    let (g, scale) = reduce_f64x4(x);
    horner_value(g) * scale
}

#[inline(always)]
fn fast_d1_f64x4(x: f64x4) -> f64x4 {
    let (g, scale) = reduce_f64x4(x);
    horner_d1(g) * scale
}

#[cfg(test)]
mod tests {
    use super::FastApprox;
    use super::MathOps;
    use wide::f64x4;

    fn assert_exp_f64(x: f64) {
        let value = FastApprox::exp(x);
        let truth = x.exp();
        let tol = 2.0_f64.powi(-23);
        if truth.abs() < tol {
            assert!(
                (value - truth).abs() < tol,
                "x={x} value={value} truth={truth}"
            );
            return;
        }
        let rel = (value - truth).abs() / truth.abs();
        assert!(rel < tol, "x={x} rel={rel} value={value} truth={truth}");
    }

    fn assert_exp_f32(x: f32) {
        let value = FastApprox::exp(x);
        let truth = x.exp();
        let bound = 8.0 * f32::EPSILON;
        if truth.abs() < bound {
            assert!(
                (value - truth).abs() < bound,
                "x={x} value={value} truth={truth}"
            );
            return;
        }
        let rel = (value - truth).abs() / truth.abs();
        assert!(rel < bound, "x={x} rel={rel} value={value} truth={truth}");
    }

    #[test]
    fn f64_polynomial_matches_exp_on_kernel_arguments() {
        let mut x = -80.0;
        while x <= 5.0 {
            assert_exp_f64(x);
            x += 1.0e-2;
        }
        assert_exp_f64(0.0);
        assert_exp_f64(-745.0);
        assert_exp_f64(-1000.0);
    }

    #[test]
    fn f32_polynomial_matches_exp_on_kernel_arguments() {
        let mut x = -40.0_f32;
        while x <= 5.0 {
            assert_exp_f32(x);
            x += 1.0e-1;
        }
        assert_exp_f32(0.0);
        assert_exp_f32(-80.0);
        assert_exp_f32(-100.0);
    }

    #[test]
    fn f64_derivative_matches_finite_difference_of_the_polynomial() {
        let h = 1.0e-4_f64;
        for x in [-2.0_f64, -0.5, -1.0e-3, 0.0, 0.3] {
            let mid = FastApprox::jet(x);
            let hi = FastApprox::exp(x + h);
            let lo = FastApprox::exp(x - h);
            let fd = (hi - lo) / (2.0 * h);
            let scale = mid.d1.abs().max(1.0);
            assert!(
                (fd - mid.d1).abs() <= 1.0e-6 * scale,
                "x={x} fd={fd} d1={}",
                mid.d1
            );
            let hi2 = FastApprox::jet(x + h).d1;
            let lo2 = FastApprox::jet(x - h).d1;
            let fd2 = (hi2 - lo2) / (2.0 * h);
            let scale2 = mid.d2.abs().max(1.0);
            assert!(
                (fd2 - mid.d2).abs() <= 1.0e-4 * scale2,
                "x={x} fd2={fd2} d2={}",
                mid.d2
            );
        }
    }

    #[test]
    fn f64x4_polynomial_matches_scalar_exp() {
        let x = f64x4::new([-2.0, -0.3, 0.0, 0.4]);
        let lanes = [-2.0_f64, -0.3, 0.0, 0.4];
        let got = FastApprox::exp_f64x4(x).to_array();
        for (lane, value) in got.iter().enumerate() {
            let truth = lanes[lane].exp();
            let rel = (value - truth).abs() / truth.abs().max(1.0);
            assert!(rel < 2.0_f64.powi(-23), "lane={lane} rel={rel}");
        }
        let d1 = FastApprox::d1_f64x4(x).to_array();
        for (lane, value) in d1.iter().enumerate() {
            let scalar = FastApprox::jet(lanes[lane]).d1;
            let scale = scalar.abs().max(1.0);
            assert!(
                (value - scalar).abs() <= 1.0e-12 * scale,
                "lane={lane} simd={value} scalar={scalar}"
            );
        }
        let extreme = [-40.0, -713.9, -800.0, 800.0];
        let got = FastApprox::exp_f64x4(f64x4::new(extreme)).to_array();
        for (lane, &x) in extreme.iter().enumerate() {
            let scalar = FastApprox::exp(x);
            if scalar.is_infinite() {
                assert!(got[lane].is_infinite(), "lane={lane} x={x}");
            } else if scalar.abs() < 2.0_f64.powi(-23) {
                assert!(
                    (got[lane] - scalar).abs() < 2.0_f64.powi(-23),
                    "lane={lane} x={x} simd={} scalar={scalar}",
                    got[lane]
                );
            } else {
                assert_eq!(
                    got[lane].to_bits(),
                    scalar.to_bits(),
                    "lane={lane} x={x} simd={} scalar={scalar}",
                    got[lane]
                );
            }
        }
    }
}
