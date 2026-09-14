//! Reusable buffers for one Exact GP fit of size `n`.
//!
//! Allocated once when fit starts. Later optimizer iterations overwrite the
//! same storage. Crate-private; faer types are not re-exported.

use dyn_stack::{MemBuffer, StackReq};
use faer::linalg::cholesky::llt;
use faer::{Mat, Par};

use crate::error::GpError;
use crate::precision::{DoublePrecision, PrecisionPolicy};

/// Dense buffers for batch Exact GP of a fixed `n`.
pub(crate) struct Workspace<P: PrecisionPolicy> {
    /// `A = K + σn² I`, then the LLT factor `L` after Cholesky.
    pub(crate) k_matrix: Mat<P::Storage>,
    /// `W = ααᵀ - K⁻¹` for the MLL gradient trace term.
    #[allow(dead_code)]
    pub(crate) w_matrix: Mat<P::Storage>,
    /// Cached pairwise distances (squared Euclidean for Phase 1 RBF).
    pub(crate) dist_cache: Mat<P::Storage>,
    /// Kernel values and `∂K/∂θ` scratch.
    pub(crate) exp_buf: Mat<P::Storage>,
    /// Residual buffer for mixed-precision refinement. `None` in Phase 1.
    #[allow(dead_code)]
    pub(crate) refine_buf: Option<Mat<P::Refine>>,
    /// Scratch for faer `cholesky_in_place` / `solve_in_place`.
    pub(crate) faer_scratch: MemBuffer,
}

fn faer_scratch_req(n: usize) -> StackReq {
    let chol = llt::factor::cholesky_in_place_scratch::<f64>(n, Par::Seq, Default::default());
    let solve_vec = llt::solve::solve_in_place_scratch::<f64>(n, 1, Par::Seq);
    let solve_mat = llt::solve::solve_in_place_scratch::<f64>(n, n, Par::Seq);
    chol.or(solve_vec).or(solve_mat)
}

impl Workspace<DoublePrecision> {
    /// Allocates `n×n` buffers and faer scratch for an `n`-point batch fit.
    ///
    /// # Errors
    ///
    /// Returns [`GpError::EmptyInput`] if `n` is zero.
    pub(crate) fn new(n: usize) -> Result<Self, GpError> {
        if n == 0 {
            return Err(GpError::EmptyInput);
        }
        Ok(Self {
            k_matrix: Mat::<f64>::zeros(n, n),
            w_matrix: Mat::<f64>::zeros(n, n),
            dist_cache: Mat::<f64>::zeros(n, n),
            exp_buf: Mat::<f64>::zeros(n, n),
            refine_buf: None,
            faer_scratch: MemBuffer::new(faer_scratch_req(n)),
        })
    }

    /// Returns the number of training points this workspace was sized for.
    pub(crate) fn n(&self) -> usize {
        self.k_matrix.nrows()
    }

    /// Reuses the existing allocation when `n` matches, otherwise reallocates.
    ///
    /// # Errors
    ///
    /// Returns [`GpError::EmptyInput`] if `n` is zero.
    pub(crate) fn ensure(&mut self, n: usize) -> Result<(), GpError> {
        if n == self.n() {
            return Ok(());
        }
        *self = Self::new(n)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{Workspace, faer_scratch_req};
    use crate::error::GpError;
    use crate::precision::DoublePrecision;

    fn assert_send_sync<T: Send + Sync>() {}

    fn assert_square(mat: &faer::Mat<f64>, n: usize) {
        assert_eq!(mat.nrows(), n);
        assert_eq!(mat.ncols(), n);
    }

    #[test]
    fn new_rejects_empty() {
        assert_eq!(
            Workspace::<DoublePrecision>::new(0).err(),
            Some(GpError::EmptyInput)
        );
    }

    #[test]
    fn new_allocates_n_by_n_buffers_and_scratch() {
        let n = 8;
        let ws = Workspace::<DoublePrecision>::new(n).expect("n > 0");
        assert_eq!(ws.n(), n);
        assert_square(&ws.k_matrix, n);
        assert_square(&ws.w_matrix, n);
        assert_square(&ws.dist_cache, n);
        assert_square(&ws.exp_buf, n);
        assert!(ws.refine_buf.is_none());
        assert_eq!(ws.faer_scratch.len(), faer_scratch_req(n).size_bytes());
        assert_send_sync::<Workspace<DoublePrecision>>();
    }

    #[test]
    fn ensure_keeps_size_then_grows() {
        let mut ws = Workspace::<DoublePrecision>::new(4).expect("n > 0");
        ws.ensure(4).expect("same n");
        assert_eq!(ws.n(), 4);
        ws.ensure(6).expect("grow");
        assert_eq!(ws.n(), 6);
        assert_square(&ws.k_matrix, 6);
        assert_eq!(ws.ensure(0).err(), Some(GpError::EmptyInput));
    }
}
