//! Reusable buffers for one batch GPR fit of size `n`.
//!
//! Allocated once when fit starts. Later optimizer iterations overwrite the
//! same storage. Crate-private; faer types are not re-exported.

use dyn_stack::{MemBuffer, StackReq};
use faer::linalg::cholesky::llt;
use faer::{Mat, Par};

use crate::error::GprError;
use crate::precision::{DoublePrecision, PrecisionPolicy};

/// Dense buffers for a batch GPR of a fixed `n`.
pub(crate) struct Workspace<P: PrecisionPolicy> {
    /// `A = K + σn² I`, then the LLT factor `L` after Cholesky.
    pub(crate) k_matrix: Mat<P::Storage>,
    /// `W = ααᵀ - K⁻¹` for the MLL gradient trace term.
    pub(crate) w_matrix: Mat<P::Storage>,
    /// Cached pairwise squared distances for isotropic (distance-mode) leaves.
    /// `Always` reuses this across optimizer steps; `Never` refills it every
    /// kernel build.
    pub(crate) dist_cache: Mat<P::Storage>,
    /// Whether `dist_cache` matches the current training `X`.
    pub(crate) dist_ready: bool,
    /// Kernel values and `∂K/∂θ` output.
    pub(crate) exp_buf: Mat<P::Storage>,
    /// Distinct `n×n` scratch for product `∂K/∂θ`. Empty until a product tree
    /// needs a gradient, so isotropic RBF does not carry an extra matrix.
    pub(crate) kernel_scratch: Mat<P::Storage>,
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
    /// Returns [`GprError::EmptyInput`] if `n` is zero.
    pub(crate) fn new(n: usize) -> Result<Self, GprError> {
        if n == 0 {
            return Err(GprError::EmptyInput);
        }
        Ok(Self {
            k_matrix: Mat::<f64>::zeros(n, n),
            w_matrix: Mat::<f64>::zeros(n, n),
            dist_cache: Mat::<f64>::zeros(n, n),
            dist_ready: false,
            exp_buf: Mat::<f64>::zeros(n, n),
            kernel_scratch: Mat::<f64>::zeros(0, 0),
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
    /// Returns [`GprError::EmptyInput`] if `n` is zero.
    pub(crate) fn ensure(&mut self, n: usize) -> Result<(), GprError> {
        if n == self.n() {
            return Ok(());
        }
        *self = Self::new(n)?;
        Ok(())
    }

    /// Ensures product `∂K/∂θ` scratch is `n×n`. No-op when already sized.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::EmptyInput`] if `n` is zero.
    pub(crate) fn ensure_kernel_scratch(&mut self, n: usize) -> Result<(), GprError> {
        if n == 0 {
            return Err(GprError::EmptyInput);
        }
        if self.kernel_scratch.nrows() == n && self.kernel_scratch.ncols() == n {
            return Ok(());
        }
        self.kernel_scratch = Mat::<f64>::zeros(n, n);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{Workspace, faer_scratch_req};
    use crate::error::GprError;
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
            Some(GprError::EmptyInput)
        );
    }

    #[test]
    fn new_allocates_n_by_n_buffers_and_scratch() {
        let n = 8;
        let ws = Workspace::<DoublePrecision>::new(n).expect("n > 0");
        assert_eq!(ws.n(), n);
        assert!(!ws.dist_ready);
        assert_square(&ws.k_matrix, n);
        assert_square(&ws.w_matrix, n);
        assert_square(&ws.dist_cache, n);
        assert_square(&ws.exp_buf, n);
        assert_eq!(ws.kernel_scratch.nrows(), 0);
        assert_eq!(ws.kernel_scratch.ncols(), 0);
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
        assert_eq!(ws.ensure(0).err(), Some(GprError::EmptyInput));
    }

    #[test]
    fn ensure_kernel_scratch_allocates_when_needed() {
        let mut ws = Workspace::<DoublePrecision>::new(4).expect("n > 0");
        assert_eq!(ws.kernel_scratch.nrows(), 0);
        ws.ensure_kernel_scratch(4).expect("n > 0");
        assert_square(&ws.kernel_scratch, 4);
        ws.ensure_kernel_scratch(4).expect("same n");
        assert_square(&ws.kernel_scratch, 4);
        assert_eq!(
            ws.ensure_kernel_scratch(0).err(),
            Some(GprError::EmptyInput)
        );
    }
}
