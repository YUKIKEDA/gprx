//! Reusable buffers for one batch GPR fit of size `n`, and query buffers
//! for [`crate::FittedGpr::predict_into`].
//!
//! Fit buffers are allocated once when fit starts. Later optimizer iterations
//! overwrite the same storage. [`FitBuffers`] holds the shared core, the
//! distance cache when [`crate::DistanceCachePolicy::Cached`], and the
//! dedicated `W` matrix when [`crate::CholeskyBuffer::Retain`]. Query buffers
//! live on [`QueryWorkspace`]. Crate-private; faer types are not re-exported.

use dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::cholesky::llt;
use faer::{Mat, MatRef};

use crate::error::GprError;
use crate::kernel::KernelScalar;
use crate::linalg::{faer_par, fill_identity};
use crate::precision::PrecisionPolicy;

/// Shared fit buffers: `L` (or `W` while a reuse gradient is in progress)
/// and Cholesky scratch. No distance cache.
pub struct WorkspaceCore<P: PrecisionPolicy> {
    /// `A = K + σn² I`, then the LLT factor `L` after Cholesky.
    pub(crate) k_matrix: Mat<P::Storage>,
    /// Kernel values and `∂K/∂θ` output. Reuse n-RHS solve also uses this.
    pub(crate) exp_buf: Mat<P::Storage>,
    /// Distinct `n×n` scratch for `∂K/∂θ` of a product or custom leaf. Empty
    /// until such a tree needs a gradient, so isotropic RBF does not carry an
    /// extra matrix.
    pub(crate) kernel_scratch: Mat<P::Storage>,
    /// One empty `0×0` matrix per Rayon worker. Detached with `mem::take`
    /// before a parallel kernel fill so closures never borrow `&mut Workspace`.
    pub(crate) thread_scratch: Vec<Mat<P::Storage>>,
    /// Right-hand side `y` then `α` for the training Cholesky solve (`n×1`).
    pub(crate) rhs: Mat<P::Storage>,
    /// Scratch for faer `cholesky_in_place` / `solve_in_place`.
    pub(crate) faer_scratch: MemBuffer,
    /// `θ` before the current hyperparameter write, to write back when `A`
    /// does not factor. Scratch: meaningless between writes.
    pub(crate) theta: Vec<f64>,
    /// Buffers for sum / product terms nested in another sum / product, one
    /// per nesting level. Empty until such a tree is evaluated. Scratch.
    pub(crate) nested: Vec<Mat<P::Storage>>,
    /// Buffers of the exact Hessian. Empty until the first Hessian. Scratch.
    pub(crate) hessian: HessianScratch<P::Storage>,
    /// Diagonal jitter `j` the last successful factor of `A + σn² I` added
    /// (`0` without a retry). `k_matrix` then holds the factor of `A + (σn² + j) I`.
    pub(crate) factor_jitter: f64,
}

/// Training-distance tensors ([`crate::DistanceCachePolicy::Cached`]).
///
/// Buffers here are caches: their contents stay valid across calls, so a
/// slot is `None` until it is filled from the training `X` (which never
/// changes for one set of fit buffers). Scratch buffers on
/// [`WorkspaceCore`] are the opposite: empty `Mat`s grown on demand whose
/// contents mean nothing between calls.
#[derive(Clone, Default)]
pub struct DistCache<S> {
    /// Pairwise squared distances for isotropic (distance-mode) leaves (`n × n`).
    pub(crate) dist: Option<Mat<S>>,
    /// Raw `(Δx_d)²` for ARD leaves (`d · n(n+1)/2`, lower triangles).
    pub(crate) ard_sq_diff: Option<crate::kernel::ArdSqDiffBuf<S>>,
}

/// Fit buffers for one training size: the shared core, plus the distance
/// cache and the dedicated `W` when the trainer's policies ask for them.
pub struct FitBuffers<P: PrecisionPolicy> {
    pub(crate) core: WorkspaceCore<P>,
    /// `Some` for [`crate::DistanceCachePolicy::Cached`].
    pub(crate) dist: Option<DistCache<P::Storage>>,
    /// Dedicated `W = ααᵀ - K⁻¹` for [`crate::CholeskyBuffer::Retain`].
    /// `None` reuses `k_matrix` as `W` and refactors afterwards.
    pub(crate) w_matrix: Option<Mat<P::Storage>>,
}

impl<P: PrecisionPolicy> Clone for FitBuffers<P> {
    fn clone(&self) -> Self {
        Self {
            core: self.core.clone(),
            dist: self.dist.clone(),
            w_matrix: self.w_matrix.clone(),
        }
    }
}

impl<P: PrecisionPolicy> FitBuffers<P> {
    /// Allocates the core, the cache for `cache`, and `W` for `buffer`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::EmptyInput`] if `n` is zero.
    pub(crate) fn new(
        n: usize,
        cache: crate::policy::DistanceCachePolicy,
        buffer: crate::policy::CholeskyBuffer,
    ) -> Result<Self, GprError> {
        let core = WorkspaceCore::new(n)?;
        let dist = match cache {
            crate::policy::DistanceCachePolicy::Cached => Some(DistCache::default()),
            crate::policy::DistanceCachePolicy::Uncached => None,
        };
        let w_matrix = match buffer {
            crate::policy::CholeskyBuffer::Retain => Some(Mat::<P::Storage>::zeros(n, n)),
            crate::policy::CholeskyBuffer::Reuse => None,
        };
        Ok(Self {
            core,
            dist,
            w_matrix,
        })
    }

    /// Whether a gradient overwrites `L` with `W` (no dedicated `W`).
    pub(crate) fn overwrites_cholesky(&self) -> bool {
        self.w_matrix.is_none()
    }
}

/// Construction and core access for composed fit buffers.
pub trait FitWorkspace: Clone + Send + Sync + 'static {
    type Policy: PrecisionPolicy;

    fn core(&self) -> &WorkspaceCore<Self::Policy>;

    fn core_mut(&mut self) -> &mut WorkspaceCore<Self::Policy>;

    /// Splits core buffers from an optional distance cache.
    #[allow(clippy::type_complexity)]
    fn split_fit(
        &mut self,
    ) -> (
        &mut WorkspaceCore<Self::Policy>,
        Option<&mut DistCache<<Self::Policy as PrecisionPolicy>::Storage>>,
    );

    /// Forms `W = ααᵀ - K⁻¹` after `k_matrix` holds `L`.
    fn form_gradient_w(&mut self, alpha: &[<Self::Policy as PrecisionPolicy>::Storage], n: usize);

    /// `W` after [`Self::form_gradient_w`].
    fn gradient_w(&self) -> MatRef<'_, <Self::Policy as PrecisionPolicy>::Storage>;

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
    /// Nested sum / product buffers for the train–query block. Scratch.
    pub(crate) query_nested: Vec<Mat<P::Storage>>,
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
            theta: Vec::new(),
            nested: Vec::new(),
            hessian: HessianScratch::default(),
            factor_jitter: 0.0,
        })
    }

    /// Returns the number of training points this workspace was sized for.
    pub(crate) fn n(&self) -> usize {
        self.k_matrix.nrows()
    }

    /// Ensures the `∂K/∂θ` scratch is `n×n`. No-op when already sized.
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

/// Buffers of the exact Hessian (`hessian_into`): the half-solved
/// derivatives `S_i` and `v_i` of every parameter. Scratch.
pub(crate) struct HessianScratch<S> {
    /// `S_i = L⁻¹ ∂A/∂θ_i L⁻ᵀ`, one `n×n` per parameter.
    pub(crate) s: Vec<Mat<S>>,
    /// `v_i = L⁻¹ ∂A/∂θ_i α` in column `i` (`n × p`).
    pub(crate) v: Mat<S>,
    /// `∂A/∂θ_i α` before its solve.
    pub(crate) u: Vec<S>,
}

impl<S> Default for HessianScratch<S> {
    fn default() -> Self {
        Self {
            s: Vec::new(),
            v: Mat::new(),
            u: Vec::new(),
        }
    }
}

impl<S: KernelScalar> HessianScratch<S> {
    /// Sizes every buffer for `n` training points and `p` parameters.
    /// No-op when already sized.
    pub(crate) fn ensure(&mut self, n: usize, p: usize) {
        if self.s.len() != p || self.s.first().is_some_and(|m| m.nrows() != n) {
            self.s = (0..p).map(|_| Mat::zeros(n, n)).collect();
        }
        if self.v.nrows() != n || self.v.ncols() != p {
            self.v = Mat::zeros(n, p);
        }
        self.u.resize(n, S::from_f64(0.0));
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
            theta: self.theta.clone(),
            nested: Vec::new(),
            hessian: HessianScratch::default(),
            factor_jitter: self.factor_jitter,
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

impl<P> FitWorkspace for FitBuffers<P>
where
    P: PrecisionPolicy + 'static,
{
    type Policy = P;

    fn core(&self) -> &WorkspaceCore<P> {
        &self.core
    }

    fn core_mut(&mut self) -> &mut WorkspaceCore<P> {
        &mut self.core
    }

    fn split_fit(&mut self) -> (&mut WorkspaceCore<P>, Option<&mut DistCache<P::Storage>>) {
        (&mut self.core, self.dist.as_mut())
    }

    fn form_gradient_w(&mut self, alpha: &[P::Storage], n: usize) {
        let core = &mut self.core;
        match &mut self.w_matrix {
            Some(w_matrix) => {
                fill_identity(w_matrix.as_mut());
                {
                    let stack = MemStack::new(&mut core.faer_scratch);
                    llt::solve::solve_in_place(
                        core.k_matrix.as_ref(),
                        w_matrix.as_mut(),
                        faer_par(n),
                        stack,
                    );
                }
                form_w_lower(w_matrix.as_mut(), alpha, n);
            }
            None => {
                fill_identity(core.exp_buf.as_mut());
                {
                    let stack = MemStack::new(&mut core.faer_scratch);
                    llt::solve::solve_in_place(
                        core.k_matrix.as_ref(),
                        core.exp_buf.as_mut(),
                        faer_par(n),
                        stack,
                    );
                }
                form_w_from_inverse(core.k_matrix.as_mut(), core.exp_buf.as_ref(), alpha, n);
            }
        }
    }

    fn gradient_w(&self) -> MatRef<'_, P::Storage> {
        match &self.w_matrix {
            Some(w_matrix) => w_matrix.as_ref(),
            None => self.core.k_matrix.as_ref(),
        }
    }

    fn has_distance_cache(&self) -> bool {
        self.dist.is_some()
    }

    fn has_dedicated_w(&self) -> bool {
        self.w_matrix.is_some()
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
            query_nested: Vec::new(),
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
    use super::{FitBuffers, FitWorkspace, QueryWorkspace, WorkspaceCore, faer_scratch_req};
    use crate::error::GprError;
    use crate::policy::{CholeskyBuffer, DistanceCachePolicy};
    use crate::precision::DoublePrecision;
    use crate::test_check::assert_send_sync;

    fn assert_square(mat: &faer::Mat<f64>, n: usize) {
        assert_eq!(mat.nrows(), n);
        assert_eq!(mat.ncols(), n);
    }

    fn speed(n: usize) -> Result<FitBuffers<DoublePrecision>, GprError> {
        FitBuffers::new(n, DistanceCachePolicy::Cached, CholeskyBuffer::Retain)
    }

    #[test]
    fn new_rejects_empty() {
        assert_eq!(speed(0).err(), Some(GprError::EmptyInput));
    }

    #[test]
    fn reuse_workspace_has_no_second_n_by_n_w() {
        let n = 8;
        let ws = FitBuffers::<DoublePrecision>::new(
            n,
            DistanceCachePolicy::Cached,
            CholeskyBuffer::Reuse,
        )
        .expect("n > 0");
        assert_eq!(ws.core().n(), n);
        assert_square(&ws.core().k_matrix, n);
        assert!(ws.dist.as_ref().expect("cached").dist.is_none());
        assert_square(&ws.core().exp_buf, n);
        assert!(ws.w_matrix.is_none());
        assert!(ws.overwrites_cholesky());
        let uncached = FitBuffers::<DoublePrecision>::new(
            n,
            DistanceCachePolicy::Uncached,
            CholeskyBuffer::Reuse,
        )
        .expect("n > 0");
        assert!(uncached.dist.is_none());
        assert!(!uncached.has_distance_cache());
    }

    #[test]
    fn new_allocates_n_by_n_buffers_and_scratch() {
        let n = 8;
        let ws = speed(n).expect("n > 0");
        let core = ws.core();
        assert_eq!(core.n(), n);
        let dist = ws.dist.as_ref().expect("cached");
        assert!(dist.dist.is_none());
        assert_square(&core.k_matrix, n);
        assert_square(ws.w_matrix.as_ref().expect("retain"), n);
        assert_square(&core.exp_buf, n);
        assert_eq!(core.kernel_scratch.nrows(), 0);
        assert!(dist.ard_sq_diff.is_none());
        assert_eq!(core.rhs.nrows(), n);
        assert_eq!(core.rhs.ncols(), 1);
        assert_eq!(
            core.thread_scratch.len(),
            rayon::current_num_threads().max(1)
        );
        assert!(
            core.thread_scratch
                .iter()
                .all(|m| m.nrows() == 0 && m.ncols() == 0)
        );
        assert_eq!(
            core.faer_scratch.len(),
            faer_scratch_req::<f64>(n).size_bytes()
        );
        assert_send_sync::<FitBuffers<DoublePrecision>>();
        assert_send_sync::<QueryWorkspace<DoublePrecision>>();
        let _core: &WorkspaceCore<DoublePrecision> = core;
    }

    #[test]
    fn ensure_kernel_scratch_allocates_when_needed() {
        let mut ws = speed(4).expect("n > 0");
        let core = ws.core_mut();
        assert_eq!(core.kernel_scratch.nrows(), 0);
        core.ensure_kernel_scratch(4).expect("n > 0");
        assert_square(&core.kernel_scratch, 4);
        core.ensure_kernel_scratch(4).expect("same n");
        assert_square(&core.kernel_scratch, 4);
        assert_eq!(
            core.ensure_kernel_scratch(0).err(),
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
