//! Reusable buffers for one batch GPR fit of size `n`, and query buffers
//! for [`crate::FittedGpr::predict_into`].
//!
//! Fit buffers are allocated once when fit starts. Later optimizer iterations
//! overwrite the same storage. Distance caches sit on [`WithDist`]; the
//! dedicated `W` matrix sits on [`WithW`]. Query buffers live on
//! [`QueryWorkspace`]. Crate-private; faer types are not re-exported.

use std::ops::{Deref, DerefMut};

use dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::cholesky::llt;
use faer::{Mat, MatRef, Par};

use crate::error::GprError;
use crate::precision::{DoublePrecision, PrecisionPolicy};

/// Shared fit buffers: `L` (or `W` while a reuse gradient is in progress)
/// and Cholesky scratch. No distance cache.
pub(crate) struct WorkspaceCore<P: PrecisionPolicy> {
    /// `A = K + σn² I`, then the LLT factor `L` after Cholesky.
    pub(crate) k_matrix: Mat<P::Storage>,
    /// Kernel values and `∂K/∂θ` output. Reuse n-RHS solve also uses this.
    pub(crate) exp_buf: Mat<P::Storage>,
    /// Distinct `n×n` scratch for product `∂K/∂θ`. Empty until a product tree
    /// needs a gradient, so isotropic RBF does not carry an extra matrix.
    pub(crate) kernel_scratch: Mat<P::Storage>,
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

/// Training-distance cache wrapping an inner workspace ([`crate::CachedDistances`]).
pub(crate) struct WithDist<W> {
    pub(crate) inner: W,
    /// Pairwise squared distances for isotropic (distance-mode) leaves.
    pub(crate) dist_cache: Mat<f64>,
    /// Whether `dist_cache` matches the current training `X`.
    pub(crate) dist_ready: bool,
    /// Raw `(Δx_d)²` for ARD leaves: `n × (n·d)`. Empty for isotropic.
    pub(crate) ard_sq_diff: Mat<f64>,
    /// Whether `ard_sq_diff` matches the current training `X`.
    pub(crate) ard_sq_diff_ready: bool,
}

/// Dedicated `W = ααᵀ - K⁻¹` wrapping an inner workspace ([`crate::RetainCholesky`]).
pub(crate) struct WithW<W> {
    pub(crate) inner: W,
    pub(crate) w_matrix: Mat<f64>,
}

/// Mutable view of the distance tensors on [`WithDist`].
pub(crate) struct DistBufs<'a> {
    pub dist_cache: &'a mut Mat<f64>,
    pub dist_ready: &'a mut bool,
    pub ard_sq_diff: &'a mut Mat<f64>,
    pub ard_sq_diff_ready: &'a mut bool,
}

impl DistBufs<'_> {
    pub(crate) fn ensure_ard_sq_diff(&mut self, n: usize, d: usize) -> Result<(), GprError> {
        if n == 0 || d == 0 {
            return Err(GprError::EmptyInput);
        }
        let cols = n.checked_mul(d).ok_or(GprError::EmptyInput)?;
        if self.ard_sq_diff.nrows() == n && self.ard_sq_diff.ncols() == cols {
            return Ok(());
        }
        *self.ard_sq_diff = Mat::<f64>::zeros(n, cols);
        *self.ard_sq_diff_ready = false;
        Ok(())
    }
}

/// Construction and core access for composed fit buffers.
pub(crate) trait FitWorkspace: Clone + Send + Sync + 'static {
    fn new(n: usize) -> Result<Self, GprError>
    where
        Self: Sized;

    fn core(&self) -> &WorkspaceCore<DoublePrecision>;

    fn core_mut(&mut self) -> &mut WorkspaceCore<DoublePrecision>;

    /// Splits core buffers from an optional distance cache.
    fn split_fit(&mut self) -> (&mut WorkspaceCore<DoublePrecision>, Option<DistBufs<'_>>);

    /// Forms `W = ααᵀ - K⁻¹` after `k_matrix` holds `L`.
    fn form_gradient_w(&mut self, alpha: &[f64], n: usize);

    /// `W` after [`Self::form_gradient_w`].
    fn gradient_w(&self) -> MatRef<'_, f64>;

    /// Sizes ARD `(Δx_d)²` when this workspace has a distance cache.
    fn ensure_ard_if_cached(&mut self, n: usize, d: usize) -> Result<(), GprError> {
        if let (_, Some(mut bufs)) = self.split_fit() {
            bufs.ensure_ard_sq_diff(n, d)?;
        }
        Ok(())
    }

    /// Whether this workspace stores `dist_cache` / `ard_sq_diff`.
    #[allow(dead_code)] // used by unit tests on `FittedGpr::workspace`
    fn has_distance_cache(&self) -> bool {
        false
    }
}

/// Default cached + retain layout used by unit tests that still name `Workspace`.
#[cfg(test)]
pub(crate) type Workspace<P> = WithDist<WithW<WorkspaceCore<P>>>;

/// Cached + reuse layout (distance tensors, no dedicated `W`).
#[cfg(test)]
pub(crate) type ReuseWorkspace<P> = WithDist<WorkspaceCore<P>>;

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

/// Caps a square `n×n` faer kernel at `min(pool, n/64, n²/16384)`.
///
/// For square `n` this matches `min(pool, n/64)` used by Cholesky and the
/// `W` n-RHS solve. See [`faer_par_dims`] and
/// `.dev/adr/0001-faer-parallel-degree.md`.
#[inline]
pub(crate) fn faer_par(n: usize) -> Par {
    faer_par_dims(n, n)
}

/// Caps faer workers by `n/64`, `n·k/16384`, and `k/12` (`k` = RHS columns).
///
/// Predict `L⁻¹ k_*` is `n×m` with `m = 100` in `compare/perf`. Using
/// [`faer_par`] (`k = n`) starts 16 workers; n=1024 jumps 1.5–22 ms and
/// n=4096 can spike above 100 ms. Kernel Rayon is unchanged.
#[inline]
pub(crate) fn faer_par_dims(nrows: usize, ncols: usize) -> Par {
    Par::rayon(faer_degree(nrows, ncols, rayon::current_num_threads()))
}

/// Worker cap: `min(pool, n/64, n·k/16384, max(1, k/12))`.
pub(crate) fn faer_degree(nrows: usize, ncols: usize, pool: usize) -> usize {
    let pool = pool.max(1);
    let by_n = (nrows / 64).max(1);
    let by_work = (nrows.saturating_mul(ncols) / 16_384).max(1);
    let by_rhs = (ncols / 12).max(1);
    pool.min(by_n).min(by_work).min(by_rhs)
}

fn faer_scratch_req(n: usize) -> StackReq {
    let par = faer_par(n);
    let chol = llt::factor::cholesky_in_place_scratch::<f64>(n, par, Default::default());
    let solve_vec = llt::solve::solve_in_place_scratch::<f64>(n, 1, par);
    let solve_mat = llt::solve::solve_in_place_scratch::<f64>(n, n, par);
    chol.or(solve_vec).or(solve_mat)
}

impl WorkspaceCore<DoublePrecision> {
    fn new(n: usize) -> Result<Self, GprError> {
        if n == 0 {
            return Err(GprError::EmptyInput);
        }
        Ok(Self {
            k_matrix: Mat::<f64>::zeros(n, n),
            exp_buf: Mat::<f64>::zeros(n, n),
            kernel_scratch: Mat::<f64>::zeros(0, 0),
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

impl Clone for WorkspaceCore<DoublePrecision> {
    fn clone(&self) -> Self {
        Self {
            k_matrix: self.k_matrix.clone(),
            exp_buf: self.exp_buf.clone(),
            kernel_scratch: self.kernel_scratch.clone(),
            thread_scratch: self.thread_scratch.clone(),
            rhs: self.rhs.clone(),
            refine_buf: self.refine_buf.clone(),
            faer_scratch: MemBuffer::new(faer_scratch_req(self.n())),
        }
    }
}

impl<W: FitWorkspace> WithDist<W> {
    #[cfg(test)]
    pub(crate) fn ensure_ard_sq_diff(&mut self, n: usize, d: usize) -> Result<(), GprError> {
        let mut bufs = DistBufs {
            dist_cache: &mut self.dist_cache,
            dist_ready: &mut self.dist_ready,
            ard_sq_diff: &mut self.ard_sq_diff,
            ard_sq_diff_ready: &mut self.ard_sq_diff_ready,
        };
        bufs.ensure_ard_sq_diff(n, d)
    }

    #[cfg(test)]
    pub(crate) fn clear_ard_sq_diff(&mut self) {
        if self.ard_sq_diff.nrows() != 0 || self.ard_sq_diff.ncols() != 0 {
            self.ard_sq_diff = Mat::<f64>::zeros(0, 0);
        }
        self.ard_sq_diff_ready = false;
    }

    /// Reuses the existing allocation when `n` matches, otherwise reallocates.
    #[cfg(test)]
    pub(crate) fn ensure(&mut self, n: usize) -> Result<(), GprError> {
        if n == self.core().n() {
            return Ok(());
        }
        *self = Self::new(n)?;
        Ok(())
    }
}

impl<W> Deref for WithDist<W> {
    type Target = W;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<W> DerefMut for WithDist<W> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<W: Clone> Clone for WithDist<W> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            dist_cache: self.dist_cache.clone(),
            dist_ready: self.dist_ready,
            ard_sq_diff: self.ard_sq_diff.clone(),
            ard_sq_diff_ready: self.ard_sq_diff_ready,
        }
    }
}

impl<W> Deref for WithW<W> {
    type Target = W;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<W> DerefMut for WithW<W> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<W: Clone> Clone for WithW<W> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            w_matrix: self.w_matrix.clone(),
        }
    }
}

fn fill_identity(mut a: faer::MatMut<'_, f64>) {
    let n = a.nrows();
    for col in 0..n {
        for row in 0..n {
            a[(row, col)] = if row == col { 1.0 } else { 0.0 };
        }
    }
}

fn form_w_lower(mut w: faer::MatMut<'_, f64>, alpha: &[f64], n: usize) {
    for col in 0..n {
        for row in col..n {
            w[(row, col)] = alpha[row] * alpha[col] - w[(row, col)];
        }
    }
}

fn form_w_from_inverse(
    mut dest: faer::MatMut<'_, f64>,
    k_inv: MatRef<'_, f64>,
    alpha: &[f64],
    n: usize,
) {
    for col in 0..n {
        for row in col..n {
            dest[(row, col)] = alpha[row] * alpha[col] - k_inv[(row, col)];
        }
    }
}

impl FitWorkspace for WorkspaceCore<DoublePrecision> {
    fn new(n: usize) -> Result<Self, GprError> {
        Self::new(n)
    }

    fn core(&self) -> &WorkspaceCore<DoublePrecision> {
        self
    }

    fn core_mut(&mut self) -> &mut WorkspaceCore<DoublePrecision> {
        self
    }

    fn split_fit(&mut self) -> (&mut WorkspaceCore<DoublePrecision>, Option<DistBufs<'_>>) {
        (self, None)
    }

    fn form_gradient_w(&mut self, alpha: &[f64], n: usize) {
        fill_identity(self.exp_buf.as_mut());
        {
            let stack = MemStack::new(&mut self.faer_scratch);
            llt::solve::solve_in_place(
                self.k_matrix.as_ref(),
                self.exp_buf.as_mut(),
                faer_par(n),
                stack,
            );
        }
        form_w_from_inverse(self.k_matrix.as_mut(), self.exp_buf.as_ref(), alpha, n);
    }

    fn gradient_w(&self) -> MatRef<'_, f64> {
        self.k_matrix.as_ref()
    }
}

impl<W: FitWorkspace> FitWorkspace for WithW<W> {
    fn new(n: usize) -> Result<Self, GprError> {
        Ok(Self {
            inner: W::new(n)?,
            w_matrix: Mat::<f64>::zeros(n, n),
        })
    }

    fn core(&self) -> &WorkspaceCore<DoublePrecision> {
        self.inner.core()
    }

    fn core_mut(&mut self) -> &mut WorkspaceCore<DoublePrecision> {
        self.inner.core_mut()
    }

    fn split_fit(&mut self) -> (&mut WorkspaceCore<DoublePrecision>, Option<DistBufs<'_>>) {
        self.inner.split_fit()
    }

    fn form_gradient_w(&mut self, alpha: &[f64], n: usize) {
        fill_identity(self.w_matrix.as_mut());
        {
            let WithW { inner, w_matrix } = self;
            let core = inner.core_mut();
            let stack = MemStack::new(&mut core.faer_scratch);
            llt::solve::solve_in_place(
                core.k_matrix.as_ref(),
                w_matrix.as_mut(),
                faer_par(n),
                stack,
            );
        }
        form_w_lower(self.w_matrix.as_mut(), alpha, n);
    }

    fn gradient_w(&self) -> MatRef<'_, f64> {
        self.w_matrix.as_ref()
    }
}

impl<W: FitWorkspace> FitWorkspace for WithDist<W> {
    fn new(n: usize) -> Result<Self, GprError> {
        Ok(Self {
            inner: W::new(n)?,
            dist_cache: Mat::<f64>::zeros(n, n),
            dist_ready: false,
            ard_sq_diff: Mat::<f64>::zeros(0, 0),
            ard_sq_diff_ready: false,
        })
    }

    fn core(&self) -> &WorkspaceCore<DoublePrecision> {
        self.inner.core()
    }

    fn core_mut(&mut self) -> &mut WorkspaceCore<DoublePrecision> {
        self.inner.core_mut()
    }

    fn split_fit(&mut self) -> (&mut WorkspaceCore<DoublePrecision>, Option<DistBufs<'_>>) {
        (
            self.inner.core_mut(),
            Some(DistBufs {
                dist_cache: &mut self.dist_cache,
                dist_ready: &mut self.dist_ready,
                ard_sq_diff: &mut self.ard_sq_diff,
                ard_sq_diff_ready: &mut self.ard_sq_diff_ready,
            }),
        )
    }

    fn form_gradient_w(&mut self, alpha: &[f64], n: usize) {
        self.inner.form_gradient_w(alpha, n);
    }

    fn gradient_w(&self) -> MatRef<'_, f64> {
        self.inner.gradient_w()
    }

    fn has_distance_cache(&self) -> bool {
        true
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
    use super::{
        FitWorkspace, QueryWorkspace, ReuseWorkspace, Workspace, WorkspaceCore, faer_degree,
        faer_scratch_req,
    };
    use crate::error::GprError;
    use crate::precision::DoublePrecision;

    fn assert_send_sync<T: Send + Sync>() {}

    fn assert_square(mat: &faer::Mat<f64>, n: usize) {
        assert_eq!(mat.nrows(), n);
        assert_eq!(mat.ncols(), n);
    }

    #[test]
    fn faer_degree_keeps_square_n_over_64() {
        assert_eq!(faer_degree(256, 256, 16), 4);
        assert_eq!(faer_degree(1024, 1024, 16), 16);
        assert_eq!(faer_degree(4096, 4096, 16), 16);
    }

    #[test]
    fn faer_degree_caps_skinny_predict_rhs() {
        assert_eq!(faer_degree(256, 100, 16), 1);
        assert_eq!(faer_degree(1024, 100, 16), 6);
        assert_eq!(faer_degree(4096, 100, 16), 8);
        assert_eq!(faer_degree(1024, 1, 16), 1);
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
    fn reuse_workspace_has_no_second_n_by_n_w() {
        let n = 8;
        let ws = ReuseWorkspace::<DoublePrecision>::new(n).expect("n > 0");
        assert_eq!(ws.core().n(), n);
        assert_square(&ws.core().k_matrix, n);
        assert_square(&ws.dist_cache, n);
        assert_square(&ws.core().exp_buf, n);
        assert_eq!(ws.core().kernel_scratch.nrows(), 0);
        assert_eq!(ws.core().kernel_scratch.ncols(), 0);
        let uncached_reuse = WorkspaceCore::<DoublePrecision>::new(n).expect("n > 0");
        assert_eq!(
            std::mem::size_of_val(&uncached_reuse),
            std::mem::size_of::<WorkspaceCore<DoublePrecision>>()
        );
        let _no_w: &WorkspaceCore<DoublePrecision> = &ws.inner;
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
