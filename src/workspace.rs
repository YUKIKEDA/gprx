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
use faer::{Mat, MatRef};

use crate::error::GprError;
use crate::kernel::KernelScalar;
use crate::linalg::{faer_par, fill_identity};
use crate::precision::{DoublePrecision, PrecisionPolicy};

/// Shared fit buffers: `L` (or `W` while a reuse gradient is in progress)
/// and Cholesky scratch. No distance cache.
pub struct WorkspaceCore<P: PrecisionPolicy> {
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
    /// Scratch for faer `cholesky_in_place` / `solve_in_place`.
    pub(crate) faer_scratch: MemBuffer,
    /// Diagonal jitter `j` the last successful factor of `A + σn² I` added
    /// (`0` without a retry). `k_matrix` then holds the factor of `A + (σn² + j) I`.
    pub(crate) factor_jitter: f64,
}

/// Training-distance cache wrapping an inner workspace ([`crate::CachedDistances`]).
pub struct WithDist<W, S = f64> {
    pub(crate) inner: W,
    /// Pairwise squared distances for isotropic (distance-mode) leaves.
    pub(crate) dist_cache: Mat<S>,
    /// Whether `dist_cache` matches the current training `X`.
    pub(crate) dist_ready: bool,
    /// Raw `(Δx_d)²` for ARD leaves: `n × (n·d)`. Empty for isotropic.
    pub(crate) ard_sq_diff: Mat<S>,
    /// Whether `ard_sq_diff` matches the current training `X`.
    pub(crate) ard_sq_diff_ready: bool,
}

/// Dedicated `W = ααᵀ - K⁻¹` wrapping an inner workspace ([`crate::RetainCholesky`]).
pub struct WithW<W, S = f64> {
    pub(crate) inner: W,
    pub(crate) w_matrix: Mat<S>,
}

/// Mutable view of the distance tensors on [`WithDist`].
pub struct DistBufs<'a, S = f64> {
    pub dist_cache: &'a mut Mat<S>,
    pub dist_ready: &'a mut bool,
    pub ard_sq_diff: &'a mut Mat<S>,
    pub ard_sq_diff_ready: &'a mut bool,
}

impl<S: KernelScalar> DistBufs<'_, S> {
    pub(crate) fn ensure_ard_sq_diff(&mut self, n: usize, d: usize) -> Result<(), GprError> {
        if n == 0 || d == 0 {
            return Err(GprError::EmptyInput);
        }
        let cols = n.checked_mul(d).ok_or(GprError::SizeOverflow)?;
        if self.ard_sq_diff.nrows() == n && self.ard_sq_diff.ncols() == cols {
            return Ok(());
        }
        *self.ard_sq_diff = Mat::<S>::zeros(n, cols);
        *self.ard_sq_diff_ready = false;
        Ok(())
    }
}

/// Construction and core access for composed fit buffers.
pub trait FitWorkspace: Clone + Send + Sync + 'static {
    type Policy: PrecisionPolicy;

    fn new(n: usize) -> Result<Self, GprError>
    where
        Self: Sized;

    fn core(&self) -> &WorkspaceCore<Self::Policy>;

    fn core_mut(&mut self) -> &mut WorkspaceCore<Self::Policy>;

    /// Splits core buffers from an optional distance cache.
    #[allow(clippy::type_complexity)]
    fn split_fit(
        &mut self,
    ) -> (
        &mut WorkspaceCore<Self::Policy>,
        Option<DistBufs<'_, <Self::Policy as PrecisionPolicy>::Storage>>,
    );

    /// Forms `W = ααᵀ - K⁻¹` after `k_matrix` holds `L`.
    fn form_gradient_w(&mut self, alpha: &[<Self::Policy as PrecisionPolicy>::Storage], n: usize);

    /// `W` after [`Self::form_gradient_w`].
    fn gradient_w(&self) -> MatRef<'_, <Self::Policy as PrecisionPolicy>::Storage>;

    /// Sizes ARD `(Δx_d)²` when this workspace has a distance cache.
    fn ensure_ard_if_cached(&mut self, _n: usize, _d: usize) -> Result<(), GprError> {
        Ok(())
    }

    /// Whether this workspace stores `dist_cache` / `ard_sq_diff`.
    #[allow(dead_code)] // used by unit tests on `FittedGpr::workspace`
    fn has_distance_cache(&self) -> bool {
        false
    }

    /// Whether this workspace stores a dedicated `W` matrix.
    #[allow(dead_code)] // used by unit tests on `FittedGpr::workspace`
    fn has_dedicated_w(&self) -> bool {
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
    pub(crate) query_kss: Vec<P::Storage>,
}

pub(crate) fn empty_thread_scratch<T: KernelScalar>() -> Vec<Mat<T>> {
    let n = rayon::current_num_threads().max(1);
    (0..n).map(|_| Mat::<T>::zeros(0, 0)).collect()
}

fn faer_scratch_req<T: faer_traits::ComplexField>(n: usize) -> StackReq {
    let par = faer_par(n);
    let chol = llt::factor::cholesky_in_place_scratch::<T>(n, par, Default::default());
    let solve_vec = llt::solve::solve_in_place_scratch::<T>(n, 1, par);
    let solve_mat = llt::solve::solve_in_place_scratch::<T>(n, n, par);
    chol.or(solve_vec).or(solve_mat)
}

impl<P> WorkspaceCore<P>
where
    P: PrecisionPolicy,
{
    fn new(n: usize) -> Result<Self, GprError> {
        if n == 0 {
            return Err(GprError::EmptyInput);
        }
        Ok(Self {
            k_matrix: Mat::<P::Storage>::zeros(n, n),
            exp_buf: Mat::<P::Storage>::zeros(n, n),
            kernel_scratch: Mat::<P::Storage>::zeros(0, 0),
            thread_scratch: empty_thread_scratch::<P::Storage>(),
            rhs: Mat::<P::Storage>::zeros(n, 1),
            faer_scratch: MemBuffer::new(faer_scratch_req::<P::Storage>(n)),
            factor_jitter: 0.0,
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
        self.kernel_scratch = Mat::<P::Storage>::zeros(n, n);
        Ok(())
    }
}

impl<P> Clone for WorkspaceCore<P>
where
    P: PrecisionPolicy,
{
    fn clone(&self) -> Self {
        Self {
            k_matrix: self.k_matrix.clone(),
            exp_buf: self.exp_buf.clone(),
            kernel_scratch: self.kernel_scratch.clone(),
            thread_scratch: self.thread_scratch.clone(),
            rhs: self.rhs.clone(),
            faer_scratch: MemBuffer::new(faer_scratch_req::<P::Storage>(self.n())),
            factor_jitter: self.factor_jitter,
        }
    }
}

impl<W: FitWorkspace<Policy = DoublePrecision>> WithDist<W> {
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

impl<W, S> Deref for WithDist<W, S> {
    type Target = W;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<W, S> DerefMut for WithDist<W, S> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<W: Clone, S: KernelScalar> Clone for WithDist<W, S> {
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

impl<W, S> Deref for WithW<W, S> {
    type Target = W;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<W, S> DerefMut for WithW<W, S> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<W: Clone, S: KernelScalar> Clone for WithW<W, S> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            w_matrix: self.w_matrix.clone(),
        }
    }
}

fn form_w_lower<T>(mut w: faer::MatMut<'_, T>, alpha: &[T], n: usize)
where
    T: KernelScalar,
{
    for col in 0..n {
        for row in col..n {
            w[(row, col)] = alpha[row] * alpha[col] - w[(row, col)];
        }
    }
}

fn form_w_from_inverse<T>(
    mut dest: faer::MatMut<'_, T>,
    k_inv: MatRef<'_, T>,
    alpha: &[T],
    n: usize,
) where
    T: KernelScalar,
{
    for col in 0..n {
        for row in col..n {
            dest[(row, col)] = alpha[row] * alpha[col] - k_inv[(row, col)];
        }
    }
}

impl<P> FitWorkspace for WorkspaceCore<P>
where
    P: PrecisionPolicy + 'static,
{
    type Policy = P;

    fn new(n: usize) -> Result<Self, GprError> {
        Self::new(n)
    }

    fn core(&self) -> &WorkspaceCore<P> {
        self
    }

    fn core_mut(&mut self) -> &mut WorkspaceCore<P> {
        self
    }

    fn split_fit(&mut self) -> (&mut WorkspaceCore<P>, Option<DistBufs<'_, P::Storage>>) {
        (self, None)
    }

    fn form_gradient_w(&mut self, alpha: &[P::Storage], n: usize) {
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

    fn gradient_w(&self) -> MatRef<'_, P::Storage> {
        self.k_matrix.as_ref()
    }
}

impl<W, S> FitWorkspace for WithW<W, S>
where
    W: FitWorkspace<Policy: PrecisionPolicy<Storage = S>> + 'static,
    S: KernelScalar,
{
    type Policy = W::Policy;

    fn new(n: usize) -> Result<Self, GprError> {
        Ok(Self {
            inner: W::new(n)?,
            w_matrix: Mat::<S>::zeros(n, n),
        })
    }

    fn core(&self) -> &WorkspaceCore<W::Policy> {
        self.inner.core()
    }

    fn core_mut(&mut self) -> &mut WorkspaceCore<W::Policy> {
        self.inner.core_mut()
    }

    fn split_fit(&mut self) -> (&mut WorkspaceCore<W::Policy>, Option<DistBufs<'_, S>>) {
        self.inner.split_fit()
    }

    fn form_gradient_w(&mut self, alpha: &[S], n: usize) {
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

    fn gradient_w(&self) -> MatRef<'_, S> {
        self.w_matrix.as_ref()
    }

    fn has_dedicated_w(&self) -> bool {
        true
    }
}

impl<W, S> FitWorkspace for WithDist<W, S>
where
    W: FitWorkspace<Policy: PrecisionPolicy<Storage = S>> + 'static,
    S: KernelScalar,
{
    type Policy = W::Policy;

    fn new(n: usize) -> Result<Self, GprError> {
        Ok(Self {
            inner: W::new(n)?,
            dist_cache: Mat::<S>::zeros(n, n),
            dist_ready: false,
            ard_sq_diff: Mat::<S>::zeros(0, 0),
            ard_sq_diff_ready: false,
        })
    }

    fn core(&self) -> &WorkspaceCore<W::Policy> {
        self.inner.core()
    }

    fn core_mut(&mut self) -> &mut WorkspaceCore<W::Policy> {
        self.inner.core_mut()
    }

    fn split_fit(&mut self) -> (&mut WorkspaceCore<W::Policy>, Option<DistBufs<'_, S>>) {
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

    fn form_gradient_w(&mut self, alpha: &[S], n: usize) {
        self.inner.form_gradient_w(alpha, n);
    }

    fn gradient_w(&self) -> MatRef<'_, S> {
        self.inner.gradient_w()
    }

    fn ensure_ard_if_cached(&mut self, n: usize, d: usize) -> Result<(), GprError> {
        let mut bufs = DistBufs {
            dist_cache: &mut self.dist_cache,
            dist_ready: &mut self.dist_ready,
            ard_sq_diff: &mut self.ard_sq_diff,
            ard_sq_diff_ready: &mut self.ard_sq_diff_ready,
        };
        bufs.ensure_ard_sq_diff(n, d)
    }

    fn has_distance_cache(&self) -> bool {
        true
    }

    fn has_dedicated_w(&self) -> bool {
        self.inner.has_dedicated_w()
    }
}

impl<P> QueryWorkspace<P>
where
    P: PrecisionPolicy + 'static,
{
    /// Builds empty query buffers. [`Self::ensure`] sizes them on first use.
    pub(crate) fn new() -> Self {
        Self {
            query_xs: Vec::new(),
            query_x: Mat::<P::Storage>::zeros(0, 0),
            query_k_star: Mat::<P::Storage>::zeros(0, 0),
            query_scratch: Mat::<P::Storage>::zeros(0, 0),
            query_dist: Mat::<P::Storage>::zeros(0, 0),
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
            .resize(m.checked_mul(d).ok_or(GprError::SizeOverflow)?, 0.0);
        self.query_x = Mat::<P::Storage>::zeros(m, d);
        self.query_k_star = Mat::<P::Storage>::zeros(n, m);
        self.query_scratch = Mat::<P::Storage>::zeros(n, m);
        self.query_dist = Mat::<P::Storage>::zeros(n, m);
        self.query_kss.resize(m, P::Storage::from_f64(0.0));
        Ok(())
    }

    /// Grows query buffers so an `n×m` fill with feature count `d` fits.
    ///
    /// New sides are `max(needed, max(current, 1) * 2)` when a side is short.
    /// No-op when every buffer already fits. Does not shrink.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::EmptyInput`] if `n`, `m`, or `d` is zero.
    pub(crate) fn ensure_at_least(&mut self, n: usize, m: usize, d: usize) -> Result<(), GprError> {
        if n == 0 || m == 0 || d == 0 {
            return Err(GprError::EmptyInput);
        }
        let have_n = self.query_k_star.nrows();
        let have_m = self.query_k_star.ncols();
        let have_d = self.query_x.ncols();
        let have_xq = self.query_x.nrows();
        if have_n >= n && have_m >= m && have_d == d && have_xq >= m {
            if self.query_xs.len() < m.saturating_mul(d) {
                self.query_xs.resize(m * d, 0.0);
            }
            if self.query_kss.len() < m {
                self.query_kss.resize(m, P::Storage::from_f64(0.0));
            }
            return Ok(());
        }
        let new_n = if have_n >= n {
            have_n
        } else {
            n.max(have_n.max(1).saturating_mul(2))
        };
        let new_m = if have_m >= m && have_xq >= m {
            have_m.max(have_xq)
        } else {
            m.max(have_m.max(have_xq).max(1).saturating_mul(2))
        };
        self.query_xs
            .resize(new_m.checked_mul(d).ok_or(GprError::SizeOverflow)?, 0.0);
        self.query_x = Mat::<P::Storage>::zeros(new_m, d);
        self.query_k_star = Mat::<P::Storage>::zeros(new_n, new_m);
        self.query_scratch = Mat::<P::Storage>::zeros(new_n, new_m);
        self.query_dist = Mat::<P::Storage>::zeros(new_n, new_m);
        self.query_kss.resize(new_m, P::Storage::from_f64(0.0));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        FitWorkspace, QueryWorkspace, ReuseWorkspace, Workspace, WorkspaceCore, faer_scratch_req,
    };
    use crate::error::GprError;
    use crate::precision::DoublePrecision;
    use crate::test_check::assert_send_sync;

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
        assert_eq!(
            ws.faer_scratch.len(),
            faer_scratch_req::<f64>(n).size_bytes()
        );
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
