//! Growable buffers for online insert on a fitted GPR.
//!
//! Crate-private. [`crate::OnlineGpr`] owns training `X` and calls
//! [`OnlineWorkspace::ensure_capacity`] before a tail insert.

use faer::linalg::triangular_solve::{
    solve_unit_lower_triangular_in_place, solve_unit_upper_triangular_in_place,
};
use faer::{Col, Mat, MatMut, MatRef, Par};

use crate::error::{CholeskyStage, GprError};

/// Capacity-backed K, LDLT, targets, and an isotropic distance cache.
///
/// All matrices are `n_capacity × n_capacity`. Vectors are length
/// `n_capacity`. The live prefix is `n_active`.
#[derive(Clone, Debug)]
pub(crate) struct OnlineWorkspace {
    pub(crate) k_matrix: Mat<f64>,
    pub(crate) ld_factor: Mat<f64>,
    pub(crate) dist_cache: Mat<f64>,
    pub(crate) y: Col<f64>,
    pub(crate) alpha: Col<f64>,
    pub(crate) v_buf: Col<f64>,
    pub(crate) n_active: usize,
    pub(crate) n_capacity: usize,
}

impl OnlineWorkspace {
    /// Allocates a full workspace of order `n` (`n_active == n_capacity`).
    pub(crate) fn from_active(n: usize) -> Result<Self, GprError> {
        if n == 0 {
            return Err(GprError::EmptyInput);
        }
        Ok(Self {
            k_matrix: Mat::zeros(n, n),
            ld_factor: Mat::zeros(n, n),
            dist_cache: Mat::zeros(n, n),
            y: Col::zeros(n),
            alpha: Col::zeros(n),
            v_buf: Col::zeros(n),
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

        let mut k_matrix = Mat::zeros(new_cap, new_cap);
        let mut ld_factor = Mat::zeros(new_cap, new_cap);
        let mut dist_cache = Mat::zeros(new_cap, new_cap);
        copy_leading_mat(&self.k_matrix, &mut k_matrix, n);
        copy_leading_mat(&self.ld_factor, &mut ld_factor, n);
        copy_leading_mat(&self.dist_cache, &mut dist_cache, n);

        let mut y = Col::zeros(new_cap);
        let mut alpha = Col::zeros(new_cap);
        let mut v_buf = Col::zeros(new_cap);
        copy_leading_col(&self.y, &mut y, n);
        copy_leading_col(&self.alpha, &mut alpha, n);
        copy_leading_col(&self.v_buf, &mut v_buf, n);

        self.k_matrix = k_matrix;
        self.ld_factor = ld_factor;
        self.dist_cache = dist_cache;
        self.y = y;
        self.alpha = alpha;
        self.v_buf = v_buf;
        self.n_capacity = new_cap;
    }

    /// Fills the leading `n` of `ld_factor` from an LLT factor (`L Lᵀ`).
    pub(crate) fn fill_ld_from_llt(
        &mut self,
        l: MatRef<'_, f64>,
        n: usize,
    ) -> Result<(), GprError> {
        if n == 0 || n > self.n_capacity || l.nrows() < n || l.ncols() < n {
            return Err(GprError::EmptyInput);
        }
        for j in 0..n {
            let ljj = l[(j, j)];
            if !ljj.is_finite() || ljj <= 0.0 {
                return Err(GprError::CholeskyFailed {
                    jitter: 0.0,
                    matrix_size: n,
                    stage: CholeskyStage::OnlineInsert,
                });
            }
            self.ld_factor[(j, j)] = ljj * ljj;
            let inv = 1.0 / ljj;
            for i in (j + 1)..n {
                self.ld_factor[(i, j)] = l[(i, j)] * inv;
            }
        }
        self.n_active = n;
        Ok(())
    }

    /// Copies packed LDLT from `ld` into the leading `n`.
    pub(crate) fn copy_ld_from(&mut self, ld: MatRef<'_, f64>, n: usize) -> Result<(), GprError> {
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
    pub(crate) fn append_border(&mut self, k: &[f64], k_new: f64) -> Result<(), GprError> {
        let n = self.n_active;
        if k.len() != n {
            return Err(GprError::EmptyInput);
        }
        self.ensure_capacity(n + 1);
        let mut w = Mat::zeros(n, 1);
        for i in 0..n {
            w[(i, 0)] = k[i];
        }
        if n > 0 {
            let ld = self.ld_factor.as_ref().submatrix(0, 0, n, n);
            solve_unit_lower_triangular_in_place(ld, w.as_mut(), Par::Seq);
        }
        let mut vtdv = 0.0;
        for i in 0..n {
            let d = self.ld_factor[(i, i)];
            if !d.is_finite() || d <= 0.0 {
                return Err(GprError::CholeskyFailed {
                    jitter: 0.0,
                    matrix_size: n + 1,
                    stage: CholeskyStage::OnlineInsert,
                });
            }
            let vi = w[(i, 0)] / d;
            self.ld_factor[(n, i)] = vi;
            vtdv += vi * d * vi;
        }
        let delta = k_new - vtdv;
        if !delta.is_finite() || delta <= 0.0 {
            return Err(GprError::CholeskyFailed {
                jitter: 0.0,
                matrix_size: n + 1,
                stage: CholeskyStage::OnlineInsert,
            });
        }
        self.ld_factor[(n, n)] = delta;
        for (i, &ki) in k.iter().enumerate() {
            self.k_matrix[(n, i)] = ki;
            self.k_matrix[(i, n)] = ki;
        }
        self.k_matrix[(n, n)] = k_new;
        self.n_active = n + 1;
        Ok(())
    }

    pub(crate) fn set_vector_prefix(col: &mut Col<f64>, values: &[f64]) {
        for (i, &v) in values.iter().enumerate() {
            col[i] = v;
        }
    }

    /// Solves `L D Lᵀ x = b` for the leading `n` (overwrites the first column of `rhs`).
    pub(crate) fn solve_ldlt_in_place(ld: MatRef<'_, f64>, mut rhs: MatMut<'_, f64>, n: usize) {
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
    pub(crate) fn apply_inv_l(ld: MatRef<'_, f64>, rhs: MatMut<'_, f64>, n: usize) {
        if n == 0 {
            return;
        }
        let ld_n = ld.submatrix(0, 0, n, n);
        solve_unit_lower_triangular_in_place(ld_n, rhs, Par::Seq);
    }
}

fn copy_leading_mat(src: &Mat<f64>, dest: &mut Mat<f64>, n: usize) {
    for j in 0..n {
        for i in 0..n {
            dest[(i, j)] = src[(i, j)];
        }
    }
}

fn copy_leading_col(src: &Col<f64>, dest: &mut Col<f64>, n: usize) {
    for i in 0..n {
        dest[i] = src[i];
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
            for i in 0..n {
                let base = (i * n + j) as f64;
                ws.k_matrix[(i, j)] = 10.0 + base;
                ws.ld_factor[(i, j)] = 20.0 + base;
                ws.dist_cache[(i, j)] = 30.0 + base;
            }
            ws.y[j] = 40.0 + j as f64;
            ws.alpha[j] = 50.0 + j as f64;
            ws.v_buf[j] = 60.0 + j as f64;
        }
    }

    fn assert_leading_marks(ws: &OnlineWorkspace, n: usize) {
        for j in 0..n {
            for i in 0..n {
                let base = (i * n + j) as f64;
                assert_close(ws.k_matrix[(i, j)], 10.0 + base);
                assert_close(ws.ld_factor[(i, j)], 20.0 + base);
                assert_close(ws.dist_cache[(i, j)], 30.0 + base);
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
                if i < n && j < n {
                    continue;
                }
                assert_close(ws.k_matrix[(i, j)], 0.0);
                assert_close(ws.ld_factor[(i, j)], 0.0);
                assert_close(ws.dist_cache[(i, j)], 0.0);
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
        assert_eq!(ws.k_matrix.nrows(), cap);
        assert_eq!(ws.k_matrix.ncols(), cap);
        assert_eq!(ws.ld_factor.nrows(), cap);
        assert_eq!(ws.ld_factor.ncols(), cap);
        assert_eq!(ws.dist_cache.nrows(), cap);
        assert_eq!(ws.dist_cache.ncols(), cap);
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
            OnlineWorkspace::from_active(0).unwrap_err(),
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
