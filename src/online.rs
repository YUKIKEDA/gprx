//! Growable buffers for online insert and delete on a fitted GPR.
//!
//! Crate-private. [`crate::OnlineGpr`] owns training `X` and calls
//! [`OnlineWorkspace::ensure_capacity`] before a tail insert.

use std::fmt;

use dyn_stack::{MemBuffer, StackReq};
use faer::linalg::cholesky::ldlt;
use faer::linalg::triangular_solve::{
    solve_unit_lower_triangular_in_place, solve_unit_upper_triangular_in_place,
};
use faer::{Col, Mat, MatMut, MatRef, Par};

use crate::error::{CholeskyStage, GprError};
use crate::kernel::KernelScalar;

/// Capacity-backed LDLT, targets, and a one-column solve buffer.
///
/// `ld_factor` is `n_capacity × n_capacity`. Vectors are length
/// `n_capacity`. The live prefix is `n_active`. Insert and predict read
/// the factor only; there is no live Gram or distance cache.
pub(crate) struct OnlineWorkspace<T: KernelScalar = f64> {
    pub(crate) ld_factor: Mat<T>,
    pub(crate) y: Col<T>,
    pub(crate) alpha: Col<T>,
    pub(crate) v_buf: Col<T>,
    delete_scratch: MemBuffer,
    pub(crate) n_active: usize,
    pub(crate) n_capacity: usize,
}

impl<T: KernelScalar> OnlineWorkspace<T> {
    /// Allocates a full workspace of order `n` (`n_active == n_capacity`).
    pub(crate) fn from_active(n: usize) -> Result<Self, GprError> {
        if n == 0 {
            return Err(GprError::EmptyInput);
        }
        Ok(Self {
            ld_factor: Mat::zeros(n, n),
            y: Col::zeros(n),
            alpha: Col::zeros(n),
            v_buf: Col::zeros(n),
            delete_scratch: MemBuffer::new(delete_scratch_req::<T>(n)),
            n_active: n,
            n_capacity: n,
        })
    }

    /// Grows every buffer to the same capacity when `needed` does not fit.
    ///
    /// The new capacity is `max(needed, max(n_capacity, 1) * 2)`. Copies the
    /// leading `n_active × n_active` blocks and the leading `n_active` of
    /// each vector. No-op when `n_capacity >= needed`.
    pub(crate) fn ensure_capacity(&mut self, needed: usize) {
        if self.n_capacity >= needed {
            return;
        }
        let doubled = self.n_capacity.max(1).saturating_mul(2);
        let new_cap = needed.max(doubled);
        let n = self.n_active;

        let mut ld_factor = Mat::<T>::zeros(new_cap, new_cap);
        copy_leading_lower(&self.ld_factor, &mut ld_factor, n);

        let mut y = Col::<T>::zeros(new_cap);
        let mut alpha = Col::<T>::zeros(new_cap);
        let mut v_buf = Col::<T>::zeros(new_cap);
        copy_leading_col(&self.y, &mut y, n);
        copy_leading_col(&self.alpha, &mut alpha, n);
        copy_leading_col(&self.v_buf, &mut v_buf, n);

        self.ld_factor = ld_factor;
        self.y = y;
        self.alpha = alpha;
        self.v_buf = v_buf;
        ensure_delete_scratch::<T>(&mut self.delete_scratch, new_cap);
        self.n_capacity = new_cap;
    }

    /// Fills the leading `n` of `ld_factor` from an LLT factor (`L Lᵀ`).
    pub(crate) fn fill_ld_from_llt(&mut self, l: MatRef<'_, T>, n: usize) -> Result<(), GprError> {
        if n == 0 || n > self.n_capacity || l.nrows() < n || l.ncols() < n {
            return Err(GprError::EmptyInput);
        }
        for j in 0..n {
            let ljj = l[(j, j)];
            let ljj_f = ljj.to_f64();
            if !ljj_f.is_finite() || ljj_f <= 0.0 {
                return Err(GprError::CholeskyFailed {
                    jitter: 0.0,
                    matrix_size: n,
                    stage: CholeskyStage::OnlineInsert,
                });
            }
            self.ld_factor[(j, j)] = ljj * ljj;
            let inv = T::from_f64(1.0) / ljj;
            for i in (j + 1)..n {
                self.ld_factor[(i, j)] = l[(i, j)] * inv;
            }
        }
        self.n_active = n;
        Ok(())
    }

    /// Copies packed LDLT from `ld` into the leading `n`.
    pub(crate) fn copy_ld_from(&mut self, ld: MatRef<'_, T>, n: usize) -> Result<(), GprError> {
        if n == 0 || n > self.n_capacity || ld.nrows() < n || ld.ncols() < n {
            return Err(GprError::EmptyInput);
        }
        for j in 0..n {
            for i in j..n {
                self.ld_factor[(i, j)] = ld[(i, j)];
            }
        }
        self.n_active = n;
        Ok(())
    }

    /// Appends one bordered row: `L D v = k`, `δ = k_new - vᵀ D v`.
    ///
    /// The leading `n_active` of [`Self::v_buf`] must already hold `k`.
    #[allow(clippy::needless_range_loop)]
    pub(crate) fn append_border(&mut self, k_new: T) -> Result<(), GprError> {
        let n = self.n_active;
        self.ensure_capacity(n + 1);
        if n > 0 {
            let ld = self.ld_factor.as_ref().submatrix(0, 0, n, n);
            let w = self.v_buf.as_mat_mut().submatrix_mut(0, 0, n, 1);
            T::solve_unit_lower_in_place(ld, w);
        }
        let mut vtdv = 0.0f64;
        for i in 0..n {
            let d = self.ld_factor[(i, i)].to_f64();
            let vi = self.v_buf[i].to_f64() / d;
            self.ld_factor[(n, i)] = T::from_f64(vi);
            vtdv += vi * vi * d;
        }
        let delta = T::from_f64(k_new.to_f64() - vtdv);
        let delta_f = delta.to_f64();
        if !delta_f.is_finite() || delta_f <= 0.0 {
            return Err(GprError::CholeskyFailed {
                jitter: 0.0,
                matrix_size: n + 1,
                stage: CholeskyStage::OnlineInsert,
            });
        }
        self.ld_factor[(n, n)] = delta;
        self.n_active = n + 1;
        Ok(())
    }

    /// Drops row and column `index` from every live buffer. Capacity is unchanged.
    pub(crate) fn delete_index(&mut self, index: usize) -> Result<(), GprError> {
        let n = self.n_active;
        if n <= 1 || index >= n {
            return Err(GprError::EmptyInput);
        }
        compact_leading_col(&mut self.y, n, index);
        compact_leading_col(&mut self.alpha, n, index);
        compact_leading_col(&mut self.v_buf, n, index);

        ensure_delete_scratch::<T>(&mut self.delete_scratch, n);
        let ld = self.ld_factor.as_mut().submatrix_mut(0, 0, n, n);
        T::ldlt_delete_row_col(ld, index, n, &mut self.delete_scratch);
        zero_trailing_row_col(&mut self.ld_factor, n);
        self.n_active = n - 1;
        Ok(())
    }

    pub(crate) fn set_vector_prefix(col: &mut Col<T>, values: &[T]) {
        for (i, &v) in values.iter().enumerate() {
            col[i] = v;
        }
    }

    pub(crate) fn set_f64_prefix(col: &mut Col<T>, values: &[f64]) {
        for (i, &v) in values.iter().enumerate() {
            col[i] = T::from_f64(v);
        }
    }

    /// Solves `L D Lᵀ x = b` for the leading `n` (overwrites the first column of `rhs`).
    pub(crate) fn solve_ldlt_in_place(ld: MatRef<'_, T>, mut rhs: MatMut<'_, T>, n: usize) {
        if n == 0 {
            return;
        }
        let ld_n = ld.submatrix(0, 0, n, n);
        solve_unit_lower_triangular_in_place(ld_n, rhs.as_mut(), Par::Seq);
        for i in 0..n {
            rhs[(i, 0)] /= ld[(i, i)];
        }
        solve_unit_upper_triangular_in_place(ld_n.transpose(), rhs, Par::Seq);
    }

    /// Overwrites each column of `rhs` (`n×m`) with `L⁻¹` of that column.
    pub(crate) fn apply_inv_l(ld: MatRef<'_, T>, rhs: MatMut<'_, T>, n: usize) {
        if n == 0 {
            return;
        }
        let ld_n = ld.submatrix(0, 0, n, n);
        solve_unit_lower_triangular_in_place(ld_n, rhs, Par::Seq);
    }
}

impl<T: KernelScalar> Clone for OnlineWorkspace<T> {
    fn clone(&self) -> Self {
        Self {
            ld_factor: self.ld_factor.clone(),
            y: self.y.clone(),
            alpha: self.alpha.clone(),
            v_buf: self.v_buf.clone(),
            delete_scratch: MemBuffer::new(delete_scratch_req::<T>(self.n_capacity)),
            n_active: self.n_active,
            n_capacity: self.n_capacity,
        }
    }
}

impl<T: KernelScalar> fmt::Debug for OnlineWorkspace<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OnlineWorkspace")
            .field("n_active", &self.n_active)
            .field("n_capacity", &self.n_capacity)
            .finish_non_exhaustive()
    }
}

fn delete_scratch_req<T: KernelScalar>(n: usize) -> StackReq {
    ldlt::update::delete_rows_and_cols_clobber_scratch::<T>(n.max(1), 1)
}

fn ensure_delete_scratch<T: KernelScalar>(buf: &mut MemBuffer, n: usize) {
    let req = delete_scratch_req::<T>(n);
    if buf.len() < req.size_bytes() {
        *buf = MemBuffer::new(req);
    }
}

fn copy_leading_lower<T: KernelScalar>(src: &Mat<T>, dest: &mut Mat<T>, n: usize) {
    for j in 0..n {
        for i in j..n {
            dest[(i, j)] = src[(i, j)];
        }
    }
}

fn copy_leading_col<T: KernelScalar>(src: &Col<T>, dest: &mut Col<T>, n: usize) {
    for i in 0..n {
        dest[i] = src[i];
    }
}

fn compact_leading_col<T: KernelScalar>(col: &mut Col<T>, n: usize, index: usize) {
    for i in index..(n - 1) {
        col[i] = col[i + 1];
    }
    col[n - 1] = T::from_f64(0.0);
}

fn zero_trailing_row_col<T: KernelScalar>(mat: &mut Mat<T>, n: usize) {
    let last = n - 1;
    let zero = T::from_f64(0.0);
    for i in 0..n {
        mat[(i, last)] = zero;
        mat[(last, i)] = zero;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;

    fn assert_close(actual: f64, expected: f64) {
        let scale = expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= TOL * scale,
            "actual={actual}, expected={expected}"
        );
    }

    fn mark(ws: &mut OnlineWorkspace) {
        let n = ws.n_active;
        for j in 0..n {
            for i in j..n {
                let base = (i * n + j) as f64;
                ws.ld_factor[(i, j)] = 20.0 + base;
            }
            ws.y[j] = 40.0 + j as f64;
            ws.alpha[j] = 50.0 + j as f64;
            ws.v_buf[j] = 60.0 + j as f64;
        }
    }

    fn assert_leading_marks(ws: &OnlineWorkspace, n: usize) {
        for j in 0..n {
            for i in j..n {
                let base = (i * n + j) as f64;
                assert_close(ws.ld_factor[(i, j)], 20.0 + base);
            }
            assert_close(ws.y[j], 40.0 + j as f64);
            assert_close(ws.alpha[j], 50.0 + j as f64);
            assert_close(ws.v_buf[j], 60.0 + j as f64);
        }
    }

    fn assert_tail_zero(ws: &OnlineWorkspace, n: usize) {
        let cap = ws.n_capacity;
        for j in 0..cap {
            for i in 0..cap {
                if i < n && j < n && i >= j {
                    continue;
                }
                assert_close(ws.ld_factor[(i, j)], 0.0);
            }
        }
        for i in n..cap {
            assert_close(ws.y[i], 0.0);
            assert_close(ws.alpha[i], 0.0);
            assert_close(ws.v_buf[i], 0.0);
        }
    }

    fn assert_same_capacity(ws: &OnlineWorkspace, cap: usize) {
        assert_eq!(ws.n_capacity, cap);
        assert_eq!(ws.ld_factor.nrows(), cap);
        assert_eq!(ws.ld_factor.ncols(), cap);
        assert_eq!(ws.y.nrows(), cap);
        assert_eq!(ws.alpha.nrows(), cap);
        assert_eq!(ws.v_buf.nrows(), cap);
    }

    fn grow_preserves_marks(n: usize) {
        let mut ws = OnlineWorkspace::from_active(n).unwrap();
        mark(&mut ws);
        ws.ensure_capacity(n);
        assert_eq!(ws.n_active, n);
        assert_same_capacity(&ws, n);
        assert_leading_marks(&ws, n);

        ws.ensure_capacity(n + 1);
        let expected_cap = (n + 1).max(n * 2);
        assert_eq!(ws.n_active, n);
        assert_same_capacity(&ws, expected_cap);
        assert_leading_marks(&ws, n);
        assert_tail_zero(&ws, n);
    }

    #[test]
    fn from_active_rejects_empty() {
        assert_eq!(
            OnlineWorkspace::<f64>::from_active(0).unwrap_err(),
            GprError::EmptyInput
        );
    }

    #[test]
    fn grow_n_two_preserves_all_buffers() {
        grow_preserves_marks(2);
    }

    #[test]
    fn grow_n_three_preserves_all_buffers() {
        grow_preserves_marks(3);
    }
}
