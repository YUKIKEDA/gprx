//! Reusable buffers for one batch GPR fit of size `n`, and query buffers
//! for [`crate::FittedGpr::predict_into`].
//!
//! Fit buffers are allocated once when fit starts. Later optimizer iterations
//! overwrite the same storage. Query buffers live on [`QueryWorkspace`], not
//! on [`Workspace`]. Crate-private; faer types are not re-exported.

use dyn_stack::{MemBuffer, StackReq};
use faer::linalg::cholesky::llt;
use faer::{Mat, Par};

use crate::error::GprError;
use crate::precision::{DoublePrecision, PrecisionPolicy};

/// Dense fit buffers for a batch GPR of a fixed `n`.
///
/// Owns `L`, `W`, distance caches, and Cholesky scratch. Does not own
/// train–test predict buffers; those are [`QueryWorkspace`].
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
    /// Raw `(Δx_d)²` for ARD leaves: `n × (n·d)`, dimension `k` in columns
    /// `[k n, (k+1) n)`. Empty (`0×0`) for isotropic kernels and for
    /// [`crate::DistanceCachePolicy::Never`].
    pub(crate) ard_sq_diff: Mat<P::Storage>,
    /// Whether `ard_sq_diff` matches the current training `X`.
    pub(crate) ard_sq_diff_ready: bool,
    /// One empty `0×0` matrix per Rayon worker. Detached with `mem::take`
    /// before a parallel kernel fill so closures never borrow `&mut Workspace`.
    pub(crate) thread_scratch: Vec<Mat<P::Storage>>,
    /// Right-hand side `y` then `α` for the training Cholesky solve (`n×1`).
    pub(crate) rhs: Mat<P::Storage>,
    /// Residual buffer for mixed-precision refinement. `None` until P5-2.
    #[allow(dead_code)]
    pub(crate) refine_buf: Option<Mat<P::Refine>>,
    /// Scratch for faer `cholesky_in_place` / `solve_in_place`.
    pub(crate) faer_scratch: MemBuffer,
}

/// Predict-into buffers owned by [`crate::FittedGpr`].
///
/// Sized on the first `predict_into` for `(n, m, d)`. The same query length
/// reuses this storage. [`crate::FittedGpr::predict`] allocates locally and
/// does not touch these fields.
#[derive(Clone)]
pub(crate) struct QueryWorkspace<P: PrecisionPolicy> {
    /// Transformed query features, packed column-major.
    pub(crate) query_xs: Vec<f64>,
    /// Query points `m×d`.
    pub(crate) query_x: Mat<P::Storage>,
    /// `k(X, X*)` then `L⁻¹ k_*` (`n×m`).
    pub(crate) query_k_star: Mat<P::Storage>,
    /// Scratch for `apply_cross` (`n×m`).
    pub(crate) query_scratch: Mat<P::Storage>,
    /// Train–test squared distances (`n×m`).
    pub(crate) query_dist: Mat<P::Storage>,
    /// `k(x*_j, x*_j)` for each query column.
    pub(crate) query_kss: Vec<f64>,
}

pub(crate) fn empty_thread_scratch() -> Vec<Mat<f64>> {
    let n = rayon::current_num_threads().max(1);
    (0..n).map(|_| Mat::<f64>::zeros(0, 0)).collect()
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
            ard_sq_diff: Mat::<f64>::zeros(0, 0),
            ard_sq_diff_ready: false,
            thread_scratch: empty_thread_scratch(),
            rhs: Mat::<f64>::zeros(n, 1),
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
    #[cfg(test)]
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

    /// Ensures the ARD `(Δx_d)²` tensor is `n × (n·d)`. No-op when already sized.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::EmptyInput`] if `n` or `d` is zero.
    pub(crate) fn ensure_ard_sq_diff(&mut self, n: usize, d: usize) -> Result<(), GprError> {
        if n == 0 || d == 0 {
            return Err(GprError::EmptyInput);
        }
        let cols = n.checked_mul(d).ok_or(GprError::EmptyInput)?;
        if self.ard_sq_diff.nrows() == n && self.ard_sq_diff.ncols() == cols {
            return Ok(());
        }
        self.ard_sq_diff = Mat::<f64>::zeros(n, cols);
        self.ard_sq_diff_ready = false;
        Ok(())
    }

    /// Drops the ARD tensor so isotropic / `Never` fits do not keep `n×n×d`.
    pub(crate) fn clear_ard_sq_diff(&mut self) {
        if self.ard_sq_diff.nrows() != 0 || self.ard_sq_diff.ncols() != 0 {
            self.ard_sq_diff = Mat::<f64>::zeros(0, 0);
        }
        self.ard_sq_diff_ready = false;
    }
}

impl Clone for Workspace<DoublePrecision> {
    fn clone(&self) -> Self {
        Self {
            k_matrix: self.k_matrix.clone(),
            w_matrix: self.w_matrix.clone(),
            dist_cache: self.dist_cache.clone(),
            dist_ready: self.dist_ready,
            exp_buf: self.exp_buf.clone(),
            kernel_scratch: self.kernel_scratch.clone(),
            ard_sq_diff: self.ard_sq_diff.clone(),
            ard_sq_diff_ready: self.ard_sq_diff_ready,
            thread_scratch: self.thread_scratch.clone(),
            rhs: self.rhs.clone(),
            refine_buf: self.refine_buf.clone(),
            faer_scratch: MemBuffer::new(faer_scratch_req(self.n())),
        }
    }
}

impl QueryWorkspace<DoublePrecision> {
    /// Builds empty query buffers. [`Self::ensure`] sizes them on first use.
    pub(crate) fn new() -> Self {
        Self {
            query_xs: Vec::new(),
            query_x: Mat::<f64>::zeros(0, 0),
            query_k_star: Mat::<f64>::zeros(0, 0),
            query_scratch: Mat::<f64>::zeros(0, 0),
            query_dist: Mat::<f64>::zeros(0, 0),
            query_kss: Vec::new(),
        }
    }

    /// Sizes buffers for an `n×m` predict. No-op when already sized.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::EmptyInput`] if `n`, `m`, or `d` is zero.
    pub(crate) fn ensure(&mut self, n: usize, m: usize, d: usize) -> Result<(), GprError> {
        if n == 0 || m == 0 || d == 0 {
            return Err(GprError::EmptyInput);
        }
        if self.query_k_star.nrows() == n
            && self.query_k_star.ncols() == m
            && self.query_x.ncols() == d
            && self.query_x.nrows() == m
        {
            return Ok(());
        }
        self.query_xs
            .resize(m.checked_mul(d).ok_or(GprError::EmptyInput)?, 0.0);
        self.query_x = Mat::<f64>::zeros(m, d);
        self.query_k_star = Mat::<f64>::zeros(n, m);
        self.query_scratch = Mat::<f64>::zeros(n, m);
        self.query_dist = Mat::<f64>::zeros(n, m);
        self.query_kss.resize(m, 0.0);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{QueryWorkspace, Workspace, faer_scratch_req};
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
    fn ensure_ard_sq_diff_allocates_n_by_n_d() {
        let mut ws = Workspace::<DoublePrecision>::new(4).expect("n > 0");
        ws.ensure_ard_sq_diff(4, 3).expect("n,d > 0");
        assert_eq!(ws.ard_sq_diff.nrows(), 4);
        assert_eq!(ws.ard_sq_diff.ncols(), 12);
        ws.ensure_ard_sq_diff(4, 3).expect("same");
        assert_eq!(ws.ard_sq_diff.ncols(), 12);
        ws.ensure_ard_sq_diff(4, 2).expect("retile d");
        assert_eq!(ws.ard_sq_diff.ncols(), 8);
        ws.clear_ard_sq_diff();
        assert_eq!(ws.ard_sq_diff.nrows(), 0);
        assert_eq!(
            ws.ensure_ard_sq_diff(0, 2).err(),
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
        assert_eq!(ws.ard_sq_diff.nrows(), 0);
        assert_eq!(ws.ard_sq_diff.ncols(), 0);
        assert!(!ws.ard_sq_diff_ready);
        assert_eq!(ws.rhs.nrows(), n);
        assert_eq!(ws.rhs.ncols(), 1);
        assert_eq!(ws.thread_scratch.len(), rayon::current_num_threads().max(1));
        assert!(
            ws.thread_scratch
                .iter()
                .all(|m| m.nrows() == 0 && m.ncols() == 0)
        );
        assert!(ws.refine_buf.is_none());
        assert_eq!(ws.faer_scratch.len(), faer_scratch_req(n).size_bytes());
        assert_send_sync::<Workspace<DoublePrecision>>();
        assert_send_sync::<QueryWorkspace<DoublePrecision>>();
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

    #[test]
    fn query_ensure_allocates_when_needed() {
        let mut query = QueryWorkspace::<DoublePrecision>::new();
        assert_eq!(query.query_k_star.ncols(), 0);
        query.ensure(4, 3, 2).expect("m,d > 0");
        assert_eq!(query.query_x.nrows(), 3);
        assert_eq!(query.query_x.ncols(), 2);
        assert_eq!(query.query_k_star.nrows(), 4);
        assert_eq!(query.query_k_star.ncols(), 3);
        assert_eq!(query.query_xs.len(), 6);
        assert_eq!(query.query_kss.len(), 3);
        query.ensure(4, 3, 2).expect("same size");
        assert_eq!(query.query_k_star.ncols(), 3);
        assert_eq!(query.ensure(4, 0, 2).err(), Some(GprError::EmptyInput));
    }
}
