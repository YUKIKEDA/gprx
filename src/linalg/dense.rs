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
    // The strictly lower part is accumulated down each column. Adding that sum
    // to the diagonal in one step differs in the last bits from folding each
    // lower term onto the diagonal. The upper part stays in ascending column
    // order, which is already contiguous.
    const BLOCK: usize = 64;
    let mut start = 0;
    while start < n {
        let len = BLOCK.min(n - start);
        let mut lower = [T::from_f64(0.0); BLOCK];
        for j in 0..n {
            let xj = x[j];
            let i_begin = (j + 1).max(start);
            let i_end = start + len;
            if i_begin >= i_end {
                continue;
            }
            for i in i_begin..i_end {
                lower[i - start] += a[(i, j)] * xj;
            }
        }
        for (t, &low) in lower.iter().enumerate().take(len) {
            let i = start + t;
            let mut s = a[(i, i)] * x[i] + low;
            for j in (i + 1)..n {
                s += a[(j, i)] * x[j];
            }
            y[i] = s;
        }
        start += len;
    }
}

pub(crate) fn matvec_columns<T: KernelScalar>(a: MatRef<'_, T>, y: &[T], ay: MatMut<'_, T>) {
    T::matvec_columns(a, y, ay);
}

/// [`matvec_columns`] for `f32`: each row sum in `f64`.
///
/// Rows are visited in blocks so the `f64` totals stay on the stack. Within a
/// block every row still adds columns `0..n` in order, matching the
/// one-row dot product bit for bit.
pub(crate) fn matvec_columns_f64_accum(a: MatRef<'_, f32>, y: &[f32], mut ay: MatMut<'_, f32>) {
    const BLOCK: usize = 64;
    let m = a.nrows();
    let n = a.ncols();
    let mut start = 0;
    while start < m {
        let len = BLOCK.min(m - start);
        let mut acc = [0.0f64; BLOCK];
        for j in 0..n {
            let yj = f64::from(y[j]);
            for t in 0..len {
                acc[t] += f64::from(a[(start + t, j)]) * yj;
            }
        }
        for (t, &sum) in acc.iter().enumerate().take(len) {
            ay[(start + t, 0)] = sum as f32;
        }
        start += len;
    }
}

/// [`matvec_columns`] for `f64`.
///
/// Each entry of `ay` is still `Σ_j A[(i, j)] y[j]` with `j` ascending. The
/// inner loop walks down a column.
pub(crate) fn matvec_columns_native(a: MatRef<'_, f64>, y: &[f64], mut ay: MatMut<'_, f64>) {
    let m = a.nrows();
    let n = a.ncols();
    for i in 0..m {
        ay[(i, 0)] = 0.0;
    }
    for j in 0..n {
        let yj = y[j];
        for i in 0..m {
            ay[(i, 0)] += a[(i, j)] * yj;
        }
    }
}

/// `wᵀ A y`. Each row of `A` is summed in ascending column order, then scaled
/// by `w`, so the value matches a per-row dot product.
pub(crate) fn dot_ay<T: KernelScalar>(a: MatRef<'_, T>, y: &[T], w: &[T]) -> T {
    const BLOCK: usize = 64;
    let m = a.nrows();
    let n = a.ncols();
    let mut total = T::from_f64(0.0);
    let mut start = 0;
    while start < m {
        let len = BLOCK.min(m - start);
        let mut acc = [T::from_f64(0.0); BLOCK];
        for j in 0..n {
            let yj = y[j];
            for t in 0..len {
                acc[t] += a[(start + t, j)] * yj;
            }
        }
        for t in 0..len {
            total += acc[t] * w[start + t];
        }
        start += len;
    }
    total
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
    let rows = m.nrows();
    let cols = m.ncols();
    let mut out = vec![T::from_f64(0.0); rows];
    for j in 0..cols {
        let vj = v[j];
        for i in 0..rows {
            out[i] += m[(i, j)] * vj;
        }
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
///
/// Tiles of `b` are accumulated with `t` ascending, so each entry matches the
/// row-wise dot product. A tile is read down the columns of `a`.
pub(crate) fn gram_aat_f64_accum(a: MatRef<'_, f32>, mut b: MatMut<'_, f32>) {
    const TILE: usize = 32;
    let m = a.nrows();
    let n = a.ncols();
    let mut i0 = 0;
    while i0 < m {
        let ni = TILE.min(m - i0);
        let mut j0 = 0;
        while j0 < m {
            let nj = TILE.min(m - j0);
            let mut acc = [0.0f64; TILE * TILE];
            for t in 0..n {
                for jj in 0..nj {
                    let aj = f64::from(a[(j0 + jj, t)]);
                    for ii in 0..ni {
                        acc[jj * TILE + ii] += f64::from(a[(i0 + ii, t)]) * aj;
                    }
                }
            }
            for jj in 0..nj {
                for ii in 0..ni {
                    b[(i0 + ii, j0 + jj)] = acc[jj * TILE + ii] as f32;
                }
            }
            j0 += TILE;
        }
        i0 += TILE;
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
        for t in 0..m {
            let at = a[(t, j)];
            for i in t..m {
                out[(i, j)] += l[(i, t)] * at;
            }
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

/// Four lanes of `src` for [`dot_f64x4`]. `linalg` sits below `kernel`, so it does
/// not read `kernel::simd`.
fn load4(src: &[f64], i: usize) -> f64x4 {
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
///
/// Same tile order as [`gram_aat_f64_accum`]: `k` ascends inside each entry.
pub(crate) fn gram_aat_plus_noise_in_scalar<T: KernelScalar>(
    a: MatRef<'_, T>,
    noise: f64,
) -> Mat<T> {
    const TILE: usize = 32;
    let m = a.nrows();
    let n = a.ncols();
    let noise = T::from_f64(noise);
    let mut b = Mat::<T>::zeros(m, m);
    let mut i0 = 0;
    while i0 < m {
        let ni = TILE.min(m - i0);
        let mut j0 = 0;
        while j0 < m {
            let nj = TILE.min(m - j0);
            let mut acc = [T::from_f64(0.0); TILE * TILE];
            for k in 0..n {
                for jj in 0..nj {
                    let aj = a[(j0 + jj, k)];
                    for ii in 0..ni {
                        acc[jj * TILE + ii] += a[(i0 + ii, k)] * aj;
                    }
                }
            }
            for jj in 0..nj {
                for ii in 0..ni {
                    let mut sum = acc[jj * TILE + ii];
                    if i0 + ii == j0 + jj {
                        sum += noise;
                    }
                    b[(i0 + ii, j0 + jj)] = sum;
                }
            }
            j0 += TILE;
        }
        i0 += TILE;
    }
    b
}

/// `a y` with `a` promoted to `f64`.
pub(crate) fn matvec_promoted<T: KernelScalar>(a: MatRef<'_, T>, y: &[f64]) -> Vec<f64> {
    let m = a.nrows();
    let mut out = vec![0.0; m];
    for (j, &yj) in y.iter().enumerate().take(a.ncols()) {
        for i in 0..m {
            out[i] += a[(i, j)].to_f64() * yj;
        }
    }
    out
}

/// `L z` for lower-triangular `L`, written into `out`.
pub(crate) fn mul_lower_vec<T: KernelScalar>(l: MatRef<'_, T>, z: &[T], out: &mut [T]) {
    let m = l.nrows();
    debug_assert_eq!(z.len(), m);
    debug_assert_eq!(out.len(), m);
    for slot in out.iter_mut().take(m) {
        *slot = T::from_f64(0.0);
    }
    for j in 0..m {
        let zj = z[j];
        for i in j..m {
            out[i] += l[(i, j)] * zj;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{dot_ay, gram_aat_f64_accum, matvec_columns_f64_accum, matvec_columns_native};
    use faer::Mat;

    #[test]
    fn column_matvec_matches_row_dot_bits() {
        let (m, n) = (70_usize, 40_usize);
        let mut a = Mat::<f64>::zeros(m, n);
        let mut y = vec![0.0; n];
        for j in 0..n {
            y[j] = (j as f64) * 0.01 - 0.2;
            for i in 0..m {
                a[(i, j)] = ((i * 3 + j * 5) % 17) as f64 * 0.1 - 0.8;
            }
        }
        let mut ay = Mat::<f64>::zeros(m, 1);
        matvec_columns_native(a.as_ref(), &y, ay.as_mut());
        for i in 0..m {
            let mut sum = 0.0;
            for j in 0..n {
                sum += a[(i, j)] * y[j];
            }
            assert_eq!(ay[(i, 0)].to_bits(), sum.to_bits());
        }
        let w: Vec<f64> = (0..m).map(|i| (i as f64) * 0.02).collect();
        let mut dot = 0.0;
        for (i, &wi) in w.iter().enumerate() {
            let mut sum = 0.0;
            for j in 0..n {
                sum += a[(i, j)] * y[j];
            }
            dot += sum * wi;
        }
        assert_eq!(dot_ay(a.as_ref(), &y, &w).to_bits(), dot.to_bits());
    }

    #[test]
    fn f32_column_products_match_row_dots_bits() {
        let (m, n) = (70_usize, 35_usize);
        let mut a = Mat::<f32>::zeros(m, n);
        let mut y = vec![0.0f32; n];
        for j in 0..n {
            y[j] = (j as f32) * 0.01;
            for i in 0..m {
                a[(i, j)] = ((i + j * 3) % 11) as f32 * 0.05;
            }
        }
        let mut ay = Mat::<f32>::zeros(m, 1);
        matvec_columns_f64_accum(a.as_ref(), &y, ay.as_mut());
        for i in 0..m {
            let mut sum = 0.0f64;
            for j in 0..n {
                sum += f64::from(a[(i, j)]) * f64::from(y[j]);
            }
            assert_eq!(ay[(i, 0)].to_bits(), (sum as f32).to_bits());
        }
        let mut b = Mat::<f32>::zeros(m, m);
        gram_aat_f64_accum(a.as_ref(), b.as_mut());
        for i in 0..m {
            for j in 0..m {
                let mut sum = 0.0f64;
                for t in 0..n {
                    sum += f64::from(a[(i, t)]) * f64::from(a[(j, t)]);
                }
                assert_eq!(b[(i, j)].to_bits(), (sum as f32).to_bits());
            }
        }
    }
}
