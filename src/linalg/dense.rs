//! Dense helpers: products, norms, symmetrization, and diagonal updates.

use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, MatMut, MatRef};
use wide::f64x4;

use crate::kernel::KernelScalar;

use super::par::faer_par_dims;

/// Adds `value` to every diagonal entry of the square `k`.
pub(crate) fn add_to_diag<T: KernelScalar>(mut k: MatMut<'_, T>, value: f64) {
    let n = k.nrows();
    let value = T::from_f64(value);
    for i in 0..n {
        k[(i, i)] += value;
    }
}

pub(crate) fn frobenius_lower<T: KernelScalar>(
    w: MatRef<'_, T>,
    d_k: MatRef<'_, T>,
    n: usize,
) -> T {
    let mut inner = T::from_f64(0.0);
    let two = T::from_f64(2.0);
    for col in 0..n {
        inner += w[(col, col)] * d_k[(col, col)];
        for row in col + 1..n {
            inner += two * w[(row, col)] * d_k[(row, col)];
        }
    }
    inner
}

pub(crate) fn symmetrize_lower<T: KernelScalar>(mut a: MatMut<'_, T>, n: usize) {
    for col in 0..n {
        for row in col + 1..n {
            a[(col, row)] = a[(row, col)];
        }
    }
}

pub(crate) fn gemv_sym_lower<T: KernelScalar>(a: MatRef<'_, T>, x: &[T], y: &mut [T], n: usize) {
    for i in 0..n {
        let mut s = a[(i, i)] * x[i];
        for j in 0..i {
            s += a[(i, j)] * x[j];
        }
        for j in i + 1..n {
            s += a[(j, i)] * x[j];
        }
        y[i] = s;
    }
}

pub(crate) fn gemv_full<T: KernelScalar>(a: MatRef<'_, T>, x: &[T], y: &mut [T], n: usize) {
    let zero = T::from_f64(0.0);
    for i in 0..n {
        let mut s = zero;
        for j in 0..n {
            s += a[(i, j)] * x[j];
        }
        y[i] = s;
    }
}

pub(crate) fn trace_product<T: KernelScalar>(a: MatRef<'_, T>, b: MatRef<'_, T>, n: usize) -> T {
    let mut tr = T::from_f64(0.0);
    for col in 0..n {
        for row in 0..n {
            tr += a[(row, col)] * b[(col, row)];
        }
    }
    tr
}

pub(crate) fn matvec_columns<T: KernelScalar>(a: MatRef<'_, T>, y: &[T], ay: MatMut<'_, T>) {
    T::matvec_columns(a, y, ay);
}

/// [`matvec_columns`] for `f32`: each row sum in `f64`.
pub(crate) fn matvec_columns_f64_accum(a: MatRef<'_, f32>, y: &[f32], mut ay: MatMut<'_, f32>) {
    for i in 0..a.nrows() {
        let mut sum = 0.0f64;
        for j in 0..a.ncols() {
            sum += f64::from(a[(i, j)]) * f64::from(y[j]);
        }
        ay[(i, 0)] = sum as f32;
    }
}

/// [`matvec_columns`] for `f64`.
pub(crate) fn matvec_columns_native(a: MatRef<'_, f64>, y: &[f64], mut ay: MatMut<'_, f64>) {
    for i in 0..a.nrows() {
        let mut sum = 0.0;
        for j in 0..a.ncols() {
            sum += a[(i, j)] * y[j];
        }
        ay[(i, 0)] = sum;
    }
}

pub(crate) fn copy_mat<T: KernelScalar>(src: MatRef<'_, T>, mut dest: MatMut<'_, T>) {
    for j in 0..src.ncols() {
        for i in 0..src.nrows() {
            dest[(i, j)] = src[(i, j)];
        }
    }
}

pub(crate) fn mat_sub_mul<T: KernelScalar>(
    dest: &mut Mat<T>,
    left: MatRef<'_, T>,
    right: MatRef<'_, T>,
) {
    gemm(dest.as_mut(), Accum::Add, left, right, T::from_f64(-1.0));
}

pub(crate) fn mat_add_mul<T: KernelScalar>(
    dest: &mut Mat<T>,
    left: MatRef<'_, T>,
    right: MatRef<'_, T>,
) {
    gemm(dest.as_mut(), Accum::Add, left, right, T::from_f64(1.0));
}

pub(crate) fn mat_mul_into<T: KernelScalar>(
    dest: &mut Mat<T>,
    left: MatRef<'_, T>,
    right: MatRef<'_, T>,
) {
    gemm(dest.as_mut(), Accum::Replace, left, right, T::from_f64(1.0));
}

pub(crate) fn gemm<T: KernelScalar>(
    dest: MatMut<'_, T>,
    accum: Accum,
    lhs: MatRef<'_, T>,
    rhs: MatRef<'_, T>,
    alpha: T,
) {
    let par = faer_par_dims(dest.nrows(), dest.ncols());
    matmul(dest, accum, lhs, rhs, alpha, par);
}

pub(crate) fn frobenius_dot<T: KernelScalar>(a: MatRef<'_, T>, b: MatRef<'_, T>) -> T {
    let mut sum = T::from_f64(0.0);
    for col in 0..a.ncols() {
        for row in 0..a.nrows() {
            sum += a[(row, col)] * b[(row, col)];
        }
    }
    sum
}

pub(crate) fn dot<T: KernelScalar>(a: &[T], b: &[T]) -> T {
    let mut sum = T::from_f64(0.0);
    for (x, y) in a.iter().zip(b.iter()) {
        sum += *x * *y;
    }
    sum
}

pub(crate) fn quad_form<T: KernelScalar>(w: &[T], m: MatRef<'_, T>) -> T {
    let mw = mat_vec(m, w);
    dot(w, &mw)
}

pub(crate) fn mat_vec<T: KernelScalar>(m: MatRef<'_, T>, v: &[T]) -> Vec<T> {
    let mut out = vec![T::from_f64(0.0); m.nrows()];
    for i in 0..m.nrows() {
        let mut sum = T::from_f64(0.0);
        for j in 0..m.ncols() {
            sum += m[(i, j)] * v[j];
        }
        out[i] = sum;
    }
    out
}

pub(crate) fn round_mat<T: KernelScalar>(src: MatRef<'_, f64>) -> Mat<T> {
    let mut out = Mat::<T>::zeros(src.nrows(), src.ncols());
    for col in 0..src.ncols() {
        for row in 0..src.nrows() {
            out[(row, col)] = T::from_f64(src[(row, col)]);
        }
    }
    out
}

pub(crate) fn gram_aat_plus_noise<T: KernelScalar>(a: MatRef<'_, T>, noise: f64) -> Mat<T> {
    let m = a.nrows();
    let mut b = Mat::zeros(m, m);
    T::gram_aat(a, b.as_mut());
    for j in 0..m {
        b[(j, j)] += T::from_f64(noise);
    }
    b
}

/// `a aᵀ` for `f32`: each entry in `f64`.
pub(crate) fn gram_aat_f64_accum(a: MatRef<'_, f32>, mut b: MatMut<'_, f32>) {
    let m = a.nrows();
    let n = a.ncols();
    for i in 0..m {
        for j in 0..m {
            let mut sum = 0.0f64;
            for t in 0..n {
                sum += f64::from(a[(i, t)]) * f64::from(a[(j, t)]);
            }
            b[(i, j)] = sum as f32;
        }
    }
}

/// `a aᵀ` for `f64` through faer's matmul.
pub(crate) fn gram_aat_faer(a: MatRef<'_, f64>, b: MatMut<'_, f64>) {
    gemm(b, Accum::Replace, a, a.transpose(), 1.0);
}

pub(crate) fn frobenius2<T: KernelScalar>(a: MatRef<'_, T>) -> T {
    let mut sum = T::from_f64(0.0);
    for col in 0..a.ncols() {
        for row in 0..a.nrows() {
            let v = a[(row, col)];
            sum += v * v;
        }
    }
    sum
}

pub(crate) fn mul_lower_left<T: KernelScalar>(l: MatRef<'_, T>, a: MatRef<'_, T>) -> Mat<T> {
    let m = l.nrows();
    let n = a.ncols();
    let mut out = Mat::zeros(m, n);
    for j in 0..n {
        for i in 0..m {
            let mut sum = T::from_f64(0.0);
            for t in 0..=i {
                sum += l[(i, t)] * a[(t, j)];
            }
            out[(i, j)] = sum;
        }
    }
    out
}

/// Largest absolute value (`0` for an empty slice).
pub(crate) fn inf_norm(values: &[f64]) -> f64 {
    values.iter().fold(0.0, |acc, v| acc.max(v.abs()))
}

/// `f64` dot product in four SIMD lanes. The summation order differs from [`dot`].
pub(crate) fn dot_f64x4(left: &[f64], right: &[f64]) -> f64 {
    let n = left.len().min(right.len());
    let mut acc = f64x4::new([0.0; 4]);
    let mut i = 0;
    while i + 4 <= n {
        acc += load4(left, i) * load4(right, i);
        i += 4;
    }
    let parts = acc.to_array();
    let mut sum = parts[0] + parts[1] + parts[2] + parts[3];
    while i < n {
        sum += left[i] * right[i];
        i += 1;
    }
    sum
}

/// Squared Euclidean norm through [`dot_f64x4`].
pub(crate) fn norm2_f64x4(values: &[f64]) -> f64 {
    dot_f64x4(values, values)
}

pub(crate) fn load4(src: &[f64], i: usize) -> f64x4 {
    f64x4::new([src[i], src[i + 1], src[i + 2], src[i + 3]])
}

pub(crate) fn fill_identity<T: KernelScalar>(mut a: faer::MatMut<'_, T>) {
    let n = a.nrows();
    for col in 0..n {
        for row in 0..n {
            a[(row, col)] = if row == col {
                T::from_f64(1.0)
            } else {
                T::from_f64(0.0)
            };
        }
    }
}

/// `a aᵀ + noise I`, summed in this scalar (no `f64` accumulation).
pub(crate) fn gram_aat_plus_noise_in_scalar<T: KernelScalar>(
    a: MatRef<'_, T>,
    noise: f64,
) -> Mat<T> {
    let m = a.nrows();
    let n = a.ncols();
    let noise = T::from_f64(noise);
    let mut b = Mat::<T>::zeros(m, m);
    for i in 0..m {
        for j in 0..m {
            let mut sum = T::from_f64(0.0);
            for k in 0..n {
                sum += a[(i, k)] * a[(j, k)];
            }
            if i == j {
                sum += noise;
            }
            b[(i, j)] = sum;
        }
    }
    b
}

/// `a y` with `a` promoted to `f64`.
pub(crate) fn matvec_promoted<T: KernelScalar>(a: MatRef<'_, T>, y: &[f64]) -> Vec<f64> {
    let m = a.nrows();
    let mut out = vec![0.0; m];
    for (i, slot) in out.iter_mut().enumerate() {
        let mut sum = 0.0;
        for (j, &yj) in y.iter().enumerate().take(a.ncols()) {
            sum += a[(i, j)].to_f64() * yj;
        }
        *slot = sum;
    }
    out
}

/// `L z` for lower-triangular `L`, written into `out`.
pub(crate) fn mul_lower_vec<T: KernelScalar>(l: MatRef<'_, T>, z: &[T], out: &mut [T]) {
    let m = l.nrows();
    debug_assert_eq!(z.len(), m);
    debug_assert_eq!(out.len(), m);
    for i in 0..m {
        let mut s = T::from_f64(0.0);
        for (j, &zj) in z.iter().enumerate().take(i + 1) {
            s += l[(i, j)] * zj;
        }
        out[i] = s;
    }
}
