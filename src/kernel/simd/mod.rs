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
pub(crate) mod stationary;

use super::KernelScalar;
use crate::error::GprError;
use faer::{ColMut, MatMut, MatRef};
use wide::f64x4;

/// Lanes of [`f64x4`].
pub(crate) const LANES: usize = 4;

pub(crate) use crate::math::f64x4_all_finite as all_finite;

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
    use super::{add_squared_diff, add_squared_diff_scaled, load, store};
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
}
