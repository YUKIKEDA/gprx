//! Factor stores of an Exact GPR: [`LltStore`] for [`crate::FittedGpr`]
//! and the growable [`LdltStore`] for [`crate::OnlineGpr`].
//!
//! Crate-private. [`crate::OnlineGpr`] owns training `X` and calls
//! [`LdltStore::ensure_capacity`] before a tail insert.

use std::fmt;

use dyn_stack::{MemBuffer, StackReq};
use faer::linalg::cholesky::ldlt;
use faer::{Col, Mat, MatMut, MatRef};

use crate::error::{CholeskyStage, GprError};
use crate::kernel::KernelScalar;
use crate::persist::MappedTensors;
use crate::precision::GpScalar;
use crate::workspace::{FitBuffers, FitWorkspace, WorkspaceCore};

/// The LLT factor of a batch fit: the fit buffers, plus a memory-mapped
/// `f64` `L` while a loaded model has not been written to.
pub(crate) struct LltStore<P: GpScalar> {
    pub(crate) buffers: FitBuffers<P>,
    /// Loaded `L`. Every factor write drops it first and lands in `buffers`.
    mapped: Option<MappedTensors>,
}

impl<P: GpScalar> LltStore<P> {
    pub(crate) fn new(buffers: FitBuffers<P>) -> Self {
        Self {
            buffers,
            mapped: None,
        }
    }

    pub(crate) fn with_mapped(buffers: FitBuffers<P>, mapped: Option<MappedTensors>) -> Self {
        Self { buffers, mapped }
    }

    /// `L` of the current training system.
    pub(crate) fn l(&self) -> MatRef<'_, P::Storage> {
        let mapped = self.mapped.as_ref().map(|mapped| mapped.l_view());
        P::view_factor(mapped, self.buffers.core().k_matrix.as_ref())
    }

    /// `L`, and the per-thread kernel scratch, borrowed together.
    pub(crate) fn l_and_thread_scratch(
        &mut self,
    ) -> (MatRef<'_, P::Storage>, &mut Vec<Mat<P::Storage>>) {
        let mapped = self.mapped.as_ref().map(|mapped| mapped.l_view());
        let WorkspaceCore {
            k_matrix,
            thread_scratch,
            ..
        } = self.buffers.core_mut();
        (P::view_factor(mapped, k_matrix.as_ref()), thread_scratch)
    }

    /// Drops the mapped `L` before the buffers are written.
    pub(super) fn release_mapped(&mut self) {
        self.mapped = None;
    }
}

impl<P: GpScalar> Clone for LltStore<P> {
    /// Copies a mapped `L` into the clone's own buffers.
    fn clone(&self) -> Self {
        let mut buffers = self.buffers.clone();
        if let Some(mapped) = &self.mapped {
            P::copy_mapped_l(mapped.l_view(), buffers.core_mut().k_matrix.as_mut());
        }
        Self {
            buffers,
            mapped: None,
        }
    }
}

/// Capacity-backed LDLT, targets, and a one-column solve buffer.
///
/// The factor is stored transposed: `lt` is `n_capacity × n_capacity`,
/// column-major, and holds `Lᵀ` with `D` on the diagonal, so row `i` of `L`
/// is the contiguous head of column `i`. An appended row is one contiguous
/// write, and the forward solve of an append reads each row contiguously.
/// [`Self::ld`] is the lower view the solves and the persist format read.
/// Vectors are length `n_capacity`. The live prefix is `n_active`. Insert
/// and predict read the factor only; there is no live Gram or distance cache.
pub(crate) struct LdltStore<T: KernelScalar = f64> {
    lt: Mat<T>,
    pub(crate) y: Col<T>,
    pub(crate) alpha: Col<T>,
    pub(crate) v_buf: Vec<T>,
    delete_scratch: MemBuffer,
    pub(crate) n_active: usize,
    pub(crate) n_capacity: usize,
    /// Diagonal jitter `j` of the batch factor this workspace came from. Every
    /// row, inserted ones included, factors `A + (σn² + j) I`.
    pub(crate) factor_jitter: f64,
}

impl<T: KernelScalar> LdltStore<T> {
    /// Allocates a full workspace of order `n` (`n_active == n_capacity`).
    pub(crate) fn from_active(n: usize) -> Result<Self, GprError> {
        if n == 0 {
            return Err(GprError::EmptyInput);
        }
        Ok(Self {
            lt: Mat::zeros(n, n),
            y: Col::zeros(n),
            alpha: Col::zeros(n),
            v_buf: vec![T::from_f64(0.0); n],
            delete_scratch: MemBuffer::new(delete_scratch_req::<T>(n)),
            n_active: n,
            n_capacity: n,
            factor_jitter: 0.0,
        })
    }

    /// The unit-lower `L` with `D` on the diagonal, leading `n_active`
    /// (a row-major view of the stored `Lᵀ`).
    pub(crate) fn ld(&self) -> MatRef<'_, T> {
        self.lt
            .as_ref()
            .submatrix(0, 0, self.n_active, self.n_active)
            .transpose()
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

        let mut lt = Mat::<T>::zeros(new_cap, new_cap);
        copy_leading_upper(&self.lt, &mut lt, n);

        let mut y = Col::<T>::zeros(new_cap);
        let mut alpha = Col::<T>::zeros(new_cap);
        copy_leading_col(&self.y, &mut y, n);
        copy_leading_col(&self.alpha, &mut alpha, n);
        self.v_buf.resize(new_cap, T::from_f64(0.0));

        self.lt = lt;
        self.y = y;
        self.alpha = alpha;
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
            self.lt[(j, j)] = ljj * ljj;
            let inv = T::from_f64(1.0) / ljj;
            for i in (j + 1)..n {
                self.lt[(j, i)] = l[(i, j)] * inv;
            }
        }
        self.n_active = n;
        Ok(())
    }

    /// Writes the LLT factor `L √D` of the leading `n` into `dest` (lower triangle).
    pub(crate) fn fill_llt_into(&self, mut dest: MatMut<'_, T>, n: usize) {
        for j in 0..n {
            let root = self.lt[(j, j)].sqrt();
            dest[(j, j)] = root;
            for i in (j + 1)..n {
                dest[(i, j)] = self.lt[(j, i)] * root;
            }
        }
    }

    /// Copies packed LDLT from `ld` into the leading `n`.
    pub(crate) fn copy_ld_from(&mut self, ld: MatRef<'_, T>, n: usize) -> Result<(), GprError> {
        if n == 0 || n > self.n_capacity || ld.nrows() < n || ld.ncols() < n {
            return Err(GprError::EmptyInput);
        }
        for j in 0..n {
            for i in j..n {
                self.lt[(j, i)] = ld[(i, j)];
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
            let w = &mut self.v_buf[..n];
            T::solve_unit_lower_rows(&self.lt, w);
        }
        let mut vtdv = 0.0f64;
        for i in 0..n {
            let d = self.lt[(i, i)].to_f64();
            let vi = self.v_buf[i].to_f64() / d;
            self.lt[(i, n)] = T::from_f64(vi);
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
        self.lt[(n, n)] = delta;
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
        self.v_buf.copy_within(index + 1..n, index);
        self.v_buf[n - 1] = T::from_f64(0.0);

        ensure_delete_scratch::<T>(&mut self.delete_scratch, n);
        let ld = self.lt.as_mut().submatrix_mut(0, 0, n, n).transpose_mut();
        T::ldlt_delete_row_col(ld, index, n, &mut self.delete_scratch);
        zero_trailing_row_col(&mut self.lt, n);
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
}

impl<T: KernelScalar> Clone for LdltStore<T> {
    fn clone(&self) -> Self {
        Self {
            lt: self.lt.clone(),
            y: self.y.clone(),
            alpha: self.alpha.clone(),
            v_buf: self.v_buf.clone(),
            delete_scratch: MemBuffer::new(delete_scratch_req::<T>(self.n_capacity)),
            n_active: self.n_active,
            n_capacity: self.n_capacity,
            factor_jitter: self.factor_jitter,
        }
    }
}

impl<T: KernelScalar> fmt::Debug for LdltStore<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LdltStore")
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

fn copy_leading_upper<T: KernelScalar>(src: &Mat<T>, dest: &mut Mat<T>, n: usize) {
    for j in 0..n {
        for i in 0..=j {
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

    use crate::test_check::assert_close;

    fn mark(ws: &mut LdltStore) {
        let n = ws.n_active;
        for j in 0..n {
            for i in j..n {
                let base = (i * n + j) as f64;
                ws.lt[(j, i)] = 20.0 + base;
            }
            ws.y[j] = 40.0 + j as f64;
            ws.alpha[j] = 50.0 + j as f64;
            ws.v_buf[j] = 60.0 + j as f64;
        }
    }

    fn assert_leading_marks(ws: &LdltStore, n: usize) {
        for j in 0..n {
            for i in j..n {
                let base = (i * n + j) as f64;
                assert_close(ws.lt[(j, i)], 20.0 + base, TOL);
            }
            assert_close(ws.y[j], 40.0 + j as f64, TOL);
            assert_close(ws.alpha[j], 50.0 + j as f64, TOL);
            assert_close(ws.v_buf[j], 60.0 + j as f64, TOL);
        }
    }

    fn assert_tail_zero(ws: &LdltStore, n: usize) {
        let cap = ws.n_capacity;
        for j in 0..cap {
            for i in 0..cap {
                if i < n && j < n && i <= j {
                    continue;
                }
                assert_close(ws.lt[(i, j)], 0.0, TOL);
            }
        }
        for i in n..cap {
            assert_close(ws.y[i], 0.0, TOL);
            assert_close(ws.alpha[i], 0.0, TOL);
            assert_close(ws.v_buf[i], 0.0, TOL);
        }
    }

    fn assert_same_capacity(ws: &LdltStore, cap: usize) {
        assert_eq!(ws.n_capacity, cap);
        assert_eq!(ws.lt.nrows(), cap);
        assert_eq!(ws.lt.ncols(), cap);
        assert_eq!(ws.y.nrows(), cap);
        assert_eq!(ws.alpha.nrows(), cap);
        assert_eq!(ws.v_buf.len(), cap);
    }

    fn grow_preserves_marks(n: usize) {
        let mut ws = LdltStore::from_active(n).unwrap();
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
            LdltStore::<f64>::from_active(0).unwrap_err(),
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
