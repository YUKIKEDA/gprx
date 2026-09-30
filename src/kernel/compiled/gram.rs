//! One evaluation surface for [`CompiledKernel<f32>`](super::CompiledKernel) and
//! [`CompiledKernel<f64>`](super::CompiledKernel).

use faer::{Mat, MatMut, MatRef};

use crate::error::GprError;
use crate::kernel::{FillDistances, KernelScalar, Triangle};

use super::{CompiledKernel, CoordMode, MixedKernelViews};

/// Views that a square (training) Gram evaluation can read.
///
/// `x` is always present. `dist` is the filled squared-Euclidean matrix and
/// `ard` the filled raw `(Δx_d)²` cache, when the caller keeps them. A kernel
/// reads what its leaves need and falls back to `x` for the rest.
#[derive(Clone, Copy)]
pub(crate) struct GramInputs<'a, T> {
    pub(crate) x: MatRef<'a, T>,
    pub(crate) dist: Option<MatRef<'a, T>>,
    pub(crate) ard: Option<MatRef<'a, T>>,
}

impl<'a, T> GramInputs<'a, T> {
    /// Coordinates only. Distance leaves compute `‖x_i − x_j‖²` per pair.
    pub(crate) fn points(x: MatRef<'a, T>) -> Self {
        Self {
            x,
            dist: None,
            ard: None,
        }
    }
}

/// Kernel apply / gradient / Hessian used by fit and predict.
///
/// Callers use the `eval_*` / `grad_gram` / `hess_gram` entry points, which
/// pick the distance, coordinate, ARD-cache, or mixed path from
/// [`Self::coord_mode`] and the inputs at hand. The per-path methods below
/// are what each [`CompiledKernel`] scalar implements.
pub(crate) trait GramKernel: Clone {
    type T: KernelScalar + FillDistances;

    fn coord_mode(&self) -> Result<CoordMode, GprError>;

    fn needs_ard_sq_diff(&self) -> bool;

    fn apply<M: crate::math::KernelMath>(
        &self,
        dist: MatRef<'_, Self::T>,
        out: MatMut<'_, Self::T>,
        uplo: Triangle,
        scratch: MatMut<'_, Self::T>,
    ) -> Result<(), GprError>;

    fn apply_points<M: crate::math::KernelMath>(
        &self,
        x: MatRef<'_, Self::T>,
        out: MatMut<'_, Self::T>,
        uplo: Triangle,
        scratch: MatMut<'_, Self::T>,
    ) -> Result<(), GprError>;

    fn apply_from_ard_cache<M: crate::math::KernelMath>(
        &self,
        cache: MatRef<'_, Self::T>,
        x: MatRef<'_, Self::T>,
        out: MatMut<'_, Self::T>,
        uplo: Triangle,
        scratch: MatMut<'_, Self::T>,
    ) -> Result<(), GprError>;

    fn apply_mixed<M: crate::math::KernelMath>(
        &self,
        views: MixedKernelViews<'_, Self::T>,
        out: MatMut<'_, Self::T>,
        uplo: Triangle,
        scratch: MatMut<'_, Self::T>,
    ) -> Result<(), GprError>;

    fn apply_cross<M: crate::math::KernelMath>(
        &self,
        dist: MatRef<'_, Self::T>,
        out: MatMut<'_, Self::T>,
        scratch: MatMut<'_, Self::T>,
    ) -> Result<(), GprError>;

    fn apply_cross_points<M: crate::math::KernelMath>(
        &self,
        x: MatRef<'_, Self::T>,
        xs: MatRef<'_, Self::T>,
        out: MatMut<'_, Self::T>,
        scratch: MatMut<'_, Self::T>,
    ) -> Result<(), GprError>;

    fn apply_cross_mixed<M: crate::math::KernelMath>(
        &self,
        dist: MatRef<'_, Self::T>,
        x: MatRef<'_, Self::T>,
        xs: MatRef<'_, Self::T>,
        out: MatMut<'_, Self::T>,
        scratch: MatMut<'_, Self::T>,
    ) -> Result<(), GprError>;

    fn fill_diag(&self, out: &mut [Self::T]) -> Result<(), GprError>;

    fn fill_diag_points(&self, x: MatRef<'_, Self::T>, out: &mut [Self::T])
    -> Result<(), GprError>;

    fn grad<M: crate::math::KernelMath>(
        &self,
        dist: MatRef<'_, Self::T>,
        d_k: MatMut<'_, Self::T>,
        param_idx: usize,
        uplo: Triangle,
        scratch: MatMut<'_, Self::T>,
    ) -> Result<(), GprError>;

    fn grad_points<M: crate::math::KernelMath>(
        &self,
        x: MatRef<'_, Self::T>,
        d_k: MatMut<'_, Self::T>,
        param_idx: usize,
        uplo: Triangle,
        scratch: MatMut<'_, Self::T>,
    ) -> Result<(), GprError>;

    fn grad_from_ard_cache<M: crate::math::KernelMath>(
        &self,
        cache: MatRef<'_, Self::T>,
        x: MatRef<'_, Self::T>,
        d_k: MatMut<'_, Self::T>,
        param_idx: usize,
        uplo: Triangle,
        scratch: MatMut<'_, Self::T>,
    ) -> Result<(), GprError>;

    fn grad_mixed<M: crate::math::KernelMath>(
        &self,
        views: MixedKernelViews<'_, Self::T>,
        d_k: MatMut<'_, Self::T>,
        param_idx: usize,
        uplo: Triangle,
        scratch: MatMut<'_, Self::T>,
    ) -> Result<(), GprError>;

    fn grad_cross_points<M: crate::math::KernelMath>(
        &self,
        x1: MatRef<'_, Self::T>,
        x2: MatRef<'_, Self::T>,
        d_k: MatMut<'_, Self::T>,
        param_idx: usize,
        scratch: MatMut<'_, Self::T>,
    ) -> Result<(), GprError>;

    fn grad_diag_points<M: crate::math::KernelMath>(
        &self,
        x: MatRef<'_, Self::T>,
        out: &mut [Self::T],
        param_idx: usize,
    ) -> Result<(), GprError>;

    fn grad_wrt_coord_dim<M: crate::math::KernelMath>(
        &self,
        x1: MatRef<'_, Self::T>,
        x2: MatRef<'_, Self::T>,
        d_k: MatMut<'_, Self::T>,
        dim: usize,
    ) -> Result<(), GprError>;

    fn hess<M: crate::math::KernelMath>(
        &self,
        dist: MatRef<'_, Self::T>,
        d2_k: MatMut<'_, Self::T>,
        i: usize,
        j: usize,
        uplo: Triangle,
        scratch: MatMut<'_, Self::T>,
    ) -> Result<(), GprError>;

    fn hess_points<M: crate::math::KernelMath>(
        &self,
        x: MatRef<'_, Self::T>,
        d2_k: MatMut<'_, Self::T>,
        i: usize,
        j: usize,
        uplo: Triangle,
        scratch: MatMut<'_, Self::T>,
    ) -> Result<(), GprError>;

    fn hess_from_ard_cache<M: crate::math::KernelMath>(
        &self,
        cache: MatRef<'_, Self::T>,
        x: MatRef<'_, Self::T>,
        d2_k: MatMut<'_, Self::T>,
        pair: (usize, usize),
        uplo: Triangle,
        scratch: MatMut<'_, Self::T>,
    ) -> Result<(), GprError>;

    fn hess_mixed<M: crate::math::KernelMath>(
        &self,
        views: MixedKernelViews<'_, Self::T>,
        d2_k: MatMut<'_, Self::T>,
        i: usize,
        j: usize,
        uplo: Triangle,
        scratch: MatMut<'_, Self::T>,
    ) -> Result<(), GprError>;

    fn hess_cross_points<M: crate::math::KernelMath>(
        &self,
        x1: MatRef<'_, Self::T>,
        x2: MatRef<'_, Self::T>,
        d2_k: MatMut<'_, Self::T>,
        i: usize,
        j: usize,
        scratch: MatMut<'_, Self::T>,
    ) -> Result<(), GprError>;

    fn hess_diag_points<M: crate::math::KernelMath>(
        &self,
        x: MatRef<'_, Self::T>,
        out: &mut [Self::T],
        i: usize,
        j: usize,
    ) -> Result<(), GprError>;

    fn hess_wrt_coord_dims<M: crate::math::KernelMath>(
        &self,
        x1: MatRef<'_, Self::T>,
        x2: MatRef<'_, Self::T>,
        d2_k: MatMut<'_, Self::T>,
        dim_a: usize,
        dim_b: usize,
    ) -> Result<(), GprError>;

    fn hess_wrt_coord_mixed<M: crate::math::KernelMath>(
        &self,
        x1: MatRef<'_, Self::T>,
        x2: MatRef<'_, Self::T>,
        d2_k: MatMut<'_, Self::T>,
        dim_x1: usize,
        dim_x2: usize,
    ) -> Result<(), GprError>;

    fn hess_theta_coord_dim<M: crate::math::KernelMath>(
        &self,
        x1: MatRef<'_, Self::T>,
        x2: MatRef<'_, Self::T>,
        d2_k: MatMut<'_, Self::T>,
        param_idx: usize,
        dim: usize,
    ) -> Result<(), GprError>;

    /// Whether this tree reads a squared-Euclidean distance matrix.
    fn reads_distances(&self) -> Result<bool, GprError> {
        Ok(!matches!(self.coord_mode()?, CoordMode::Points))
    }

    /// Writes `K` for `uplo` from whichever views `inputs` holds.
    fn eval_gram<M: crate::math::KernelMath>(
        &self,
        inputs: GramInputs<'_, Self::T>,
        out: MatMut<'_, Self::T>,
        uplo: Triangle,
        scratch: MatMut<'_, Self::T>,
    ) -> Result<(), GprError> {
        let ard = ard_view(self, inputs.ard);
        match (self.coord_mode()?, inputs.dist) {
            (CoordMode::Dist | CoordMode::Either, Some(dist)) => {
                self.apply::<M>(dist, out, uplo, scratch)
            }
            (CoordMode::Points, _) => match ard {
                Some(cache) => self.apply_from_ard_cache::<M>(cache, inputs.x, out, uplo, scratch),
                None => self.apply_points::<M>(inputs.x, out, uplo, scratch),
            },
            (CoordMode::Mixed, Some(dist)) => {
                let mut views = MixedKernelViews::new(dist, inputs.x);
                views.ard_cache = ard;
                self.apply_mixed::<M>(views, out, uplo, scratch)
            }
            (_, None) => self.apply_points::<M>(inputs.x, out, uplo, scratch),
        }
    }

    /// Writes `K` for `uplo` from coordinates, filling a temporary distance
    /// matrix first when the tree reads one.
    fn eval_gram_from_points<M: crate::math::KernelMath>(
        &self,
        x: MatRef<'_, Self::T>,
        out: MatMut<'_, Self::T>,
        uplo: Triangle,
        scratch: MatMut<'_, Self::T>,
        thread_scratch: &mut [Mat<Self::T>],
    ) -> Result<(), GprError> {
        if !self.reads_distances()? {
            return self.eval_gram::<M>(GramInputs::points(x), out, uplo, scratch);
        }
        let m = x.nrows();
        let mut dist = Mat::<Self::T>::zeros(m, m);
        Self::T::write_squared(x, dist.as_mut(), thread_scratch);
        let inputs = GramInputs {
            x,
            dist: Some(dist.as_ref()),
            ard: None,
        };
        self.eval_gram::<M>(inputs, out, uplo, scratch)
    }

    /// Writes `∂K/∂θ_{param_idx}` for `uplo`.
    fn grad_gram<M: crate::math::KernelMath>(
        &self,
        inputs: GramInputs<'_, Self::T>,
        d_k: MatMut<'_, Self::T>,
        param_idx: usize,
        uplo: Triangle,
        scratch: MatMut<'_, Self::T>,
    ) -> Result<(), GprError> {
        let ard = ard_view(self, inputs.ard);
        match (self.coord_mode()?, inputs.dist) {
            (CoordMode::Dist | CoordMode::Either, Some(dist)) => {
                self.grad::<M>(dist, d_k, param_idx, uplo, scratch)
            }
            (CoordMode::Points, _) => match ard {
                Some(cache) => {
                    self.grad_from_ard_cache::<M>(cache, inputs.x, d_k, param_idx, uplo, scratch)
                }
                None => self.grad_points::<M>(inputs.x, d_k, param_idx, uplo, scratch),
            },
            (CoordMode::Mixed, Some(dist)) => self.grad_mixed::<M>(
                MixedKernelViews::new(dist, inputs.x),
                d_k,
                param_idx,
                uplo,
                scratch,
            ),
            (_, None) => self.grad_points::<M>(inputs.x, d_k, param_idx, uplo, scratch),
        }
    }

    /// Writes `∂²K/∂θ_i ∂θ_j` for `uplo`.
    fn hess_gram<M: crate::math::KernelMath>(
        &self,
        inputs: GramInputs<'_, Self::T>,
        d2_k: MatMut<'_, Self::T>,
        pair: (usize, usize),
        uplo: Triangle,
        scratch: MatMut<'_, Self::T>,
    ) -> Result<(), GprError> {
        let (i, j) = pair;
        let ard = ard_view(self, inputs.ard);
        match (self.coord_mode()?, inputs.dist) {
            (CoordMode::Dist | CoordMode::Either, Some(dist)) => {
                self.hess::<M>(dist, d2_k, i, j, uplo, scratch)
            }
            (CoordMode::Points, _) => match ard {
                Some(cache) => {
                    self.hess_from_ard_cache::<M>(cache, inputs.x, d2_k, pair, uplo, scratch)
                }
                None => self.hess_points::<M>(inputs.x, d2_k, i, j, uplo, scratch),
            },
            (CoordMode::Mixed, Some(dist)) => self.hess_mixed::<M>(
                MixedKernelViews::new(dist, inputs.x),
                d2_k,
                i,
                j,
                uplo,
                scratch,
            ),
            (_, None) => self.hess_points::<M>(inputs.x, d2_k, i, j, uplo, scratch),
        }
    }

    /// Writes the rectangular `K(x, xs)` into `out`.
    ///
    /// `dist` receives the train–query squared distances when the tree reads
    /// them; `None` allocates that buffer for this call.
    fn eval_cross<M: crate::math::KernelMath>(
        &self,
        x: MatRef<'_, Self::T>,
        xs: MatRef<'_, Self::T>,
        dist: Option<MatMut<'_, Self::T>>,
        out: MatMut<'_, Self::T>,
        scratch: MatMut<'_, Self::T>,
        thread_scratch: &mut [Mat<Self::T>],
    ) -> Result<(), GprError> {
        let mode = self.coord_mode()?;
        if matches!(mode, CoordMode::Points) {
            return self.apply_cross_points::<M>(x, xs, out, scratch);
        }
        let mut owned;
        let mut dist = match dist {
            Some(dist) => dist,
            None => {
                owned = Mat::<Self::T>::zeros(x.nrows(), xs.nrows());
                owned.as_mut()
            }
        };
        Self::T::write_cross(x, xs, dist.as_mut(), thread_scratch);
        match mode {
            CoordMode::Mixed => self.apply_cross_mixed::<M>(dist.as_ref(), x, xs, out, scratch),
            _ => self.apply_cross::<M>(dist.as_ref(), out, scratch),
        }
    }

    /// Writes the diagonal `k(x_i, x_i)` into `out`.
    fn eval_diag(&self, x: MatRef<'_, Self::T>, out: &mut [Self::T]) -> Result<(), GprError> {
        match self.coord_mode()? {
            CoordMode::Dist | CoordMode::Either => self.fill_diag(out),
            CoordMode::Points | CoordMode::Mixed => self.fill_diag_points(x, out),
        }
    }
}

/// The ARD cache when this scalar reads it and the tree has ARD leaves.
fn ard_view<'a, K: GramKernel>(
    kernel: &K,
    ard: Option<MatRef<'a, K::T>>,
) -> Option<MatRef<'a, K::T>> {
    if <K::T as FillDistances>::READS_ARD_CACHE && kernel.needs_ard_sq_diff() {
        ard.filter(|cache| cache.ncols() > 0)
    } else {
        None
    }
}

macro_rules! forward_gram {
    ($t:ty) => {
        impl GramKernel for CompiledKernel<$t> {
            type T = $t;

            fn coord_mode(&self) -> Result<CoordMode, GprError> {
                CompiledKernel::<$t>::coord_mode(self)
            }

            fn needs_ard_sq_diff(&self) -> bool {
                CompiledKernel::<$t>::needs_ard_sq_diff(self)
            }

            fn apply<M: crate::math::KernelMath>(
                &self,
                dist: MatRef<'_, Self::T>,
                out: MatMut<'_, Self::T>,
                uplo: Triangle,
                scratch: MatMut<'_, Self::T>,
            ) -> Result<(), GprError> {
                CompiledKernel::<$t>::apply::<M>(self, dist, out, uplo, scratch)
            }

            fn apply_points<M: crate::math::KernelMath>(
                &self,
                x: MatRef<'_, Self::T>,
                out: MatMut<'_, Self::T>,
                uplo: Triangle,
                scratch: MatMut<'_, Self::T>,
            ) -> Result<(), GprError> {
                CompiledKernel::<$t>::apply_points::<M>(self, x, out, uplo, scratch)
            }

            fn apply_from_ard_cache<M: crate::math::KernelMath>(
                &self,
                cache: MatRef<'_, Self::T>,
                x: MatRef<'_, Self::T>,
                out: MatMut<'_, Self::T>,
                uplo: Triangle,
                scratch: MatMut<'_, Self::T>,
            ) -> Result<(), GprError> {
                CompiledKernel::<$t>::apply_from_ard_cache::<M>(self, cache, x, out, uplo, scratch)
            }

            fn apply_mixed<M: crate::math::KernelMath>(
                &self,
                views: MixedKernelViews<'_, Self::T>,
                out: MatMut<'_, Self::T>,
                uplo: Triangle,
                scratch: MatMut<'_, Self::T>,
            ) -> Result<(), GprError> {
                CompiledKernel::<$t>::apply_mixed::<M>(self, views, out, uplo, scratch)
            }

            fn apply_cross<M: crate::math::KernelMath>(
                &self,
                dist: MatRef<'_, Self::T>,
                out: MatMut<'_, Self::T>,
                scratch: MatMut<'_, Self::T>,
            ) -> Result<(), GprError> {
                CompiledKernel::<$t>::apply_cross::<M>(self, dist, out, scratch)
            }

            fn apply_cross_points<M: crate::math::KernelMath>(
                &self,
                x: MatRef<'_, Self::T>,
                xs: MatRef<'_, Self::T>,
                out: MatMut<'_, Self::T>,
                scratch: MatMut<'_, Self::T>,
            ) -> Result<(), GprError> {
                CompiledKernel::<$t>::apply_cross_points::<M>(self, x, xs, out, scratch)
            }

            fn apply_cross_mixed<M: crate::math::KernelMath>(
                &self,
                dist: MatRef<'_, Self::T>,
                x: MatRef<'_, Self::T>,
                xs: MatRef<'_, Self::T>,
                out: MatMut<'_, Self::T>,
                scratch: MatMut<'_, Self::T>,
            ) -> Result<(), GprError> {
                CompiledKernel::<$t>::apply_cross_mixed::<M>(self, dist, x, xs, out, scratch)
            }

            fn fill_diag(&self, out: &mut [Self::T]) -> Result<(), GprError> {
                CompiledKernel::<$t>::fill_diag(self, out)
            }

            fn fill_diag_points(
                &self,
                x: MatRef<'_, Self::T>,
                out: &mut [Self::T],
            ) -> Result<(), GprError> {
                CompiledKernel::<$t>::fill_diag_points(self, x, out)
            }

            fn grad<M: crate::math::KernelMath>(
                &self,
                dist: MatRef<'_, Self::T>,
                d_k: MatMut<'_, Self::T>,
                param_idx: usize,
                uplo: Triangle,
                scratch: MatMut<'_, Self::T>,
            ) -> Result<(), GprError> {
                CompiledKernel::<$t>::grad::<M>(self, dist, d_k, param_idx, uplo, scratch)
            }

            fn grad_points<M: crate::math::KernelMath>(
                &self,
                x: MatRef<'_, Self::T>,
                d_k: MatMut<'_, Self::T>,
                param_idx: usize,
                uplo: Triangle,
                scratch: MatMut<'_, Self::T>,
            ) -> Result<(), GprError> {
                CompiledKernel::<$t>::grad_points::<M>(self, x, d_k, param_idx, uplo, scratch)
            }

            fn grad_from_ard_cache<M: crate::math::KernelMath>(
                &self,
                cache: MatRef<'_, Self::T>,
                x: MatRef<'_, Self::T>,
                d_k: MatMut<'_, Self::T>,
                param_idx: usize,
                uplo: Triangle,
                scratch: MatMut<'_, Self::T>,
            ) -> Result<(), GprError> {
                CompiledKernel::<$t>::grad_from_ard_cache::<M>(
                    self, cache, x, d_k, param_idx, uplo, scratch,
                )
            }

            fn grad_mixed<M: crate::math::KernelMath>(
                &self,
                views: MixedKernelViews<'_, Self::T>,
                d_k: MatMut<'_, Self::T>,
                param_idx: usize,
                uplo: Triangle,
                scratch: MatMut<'_, Self::T>,
            ) -> Result<(), GprError> {
                CompiledKernel::<$t>::grad_mixed::<M>(self, views, d_k, param_idx, uplo, scratch)
            }

            fn grad_cross_points<M: crate::math::KernelMath>(
                &self,
                x1: MatRef<'_, Self::T>,
                x2: MatRef<'_, Self::T>,
                d_k: MatMut<'_, Self::T>,
                param_idx: usize,
                scratch: MatMut<'_, Self::T>,
            ) -> Result<(), GprError> {
                CompiledKernel::<$t>::grad_cross_points::<M>(self, x1, x2, d_k, param_idx, scratch)
            }

            fn grad_diag_points<M: crate::math::KernelMath>(
                &self,
                x: MatRef<'_, Self::T>,
                out: &mut [Self::T],
                param_idx: usize,
            ) -> Result<(), GprError> {
                CompiledKernel::<$t>::grad_diag_points::<M>(self, x, out, param_idx)
            }

            fn grad_wrt_coord_dim<M: crate::math::KernelMath>(
                &self,
                x1: MatRef<'_, Self::T>,
                x2: MatRef<'_, Self::T>,
                d_k: MatMut<'_, Self::T>,
                dim: usize,
            ) -> Result<(), GprError> {
                CompiledKernel::<$t>::grad_wrt_coord_dim::<M>(self, x1, x2, d_k, dim)
            }

            fn hess<M: crate::math::KernelMath>(
                &self,
                dist: MatRef<'_, Self::T>,
                d2_k: MatMut<'_, Self::T>,
                i: usize,
                j: usize,
                uplo: Triangle,
                scratch: MatMut<'_, Self::T>,
            ) -> Result<(), GprError> {
                CompiledKernel::<$t>::hess::<M>(self, dist, d2_k, i, j, uplo, scratch)
            }

            fn hess_points<M: crate::math::KernelMath>(
                &self,
                x: MatRef<'_, Self::T>,
                d2_k: MatMut<'_, Self::T>,
                i: usize,
                j: usize,
                uplo: Triangle,
                scratch: MatMut<'_, Self::T>,
            ) -> Result<(), GprError> {
                CompiledKernel::<$t>::hess_points::<M>(self, x, d2_k, i, j, uplo, scratch)
            }

            fn hess_from_ard_cache<M: crate::math::KernelMath>(
                &self,
                cache: MatRef<'_, Self::T>,
                x: MatRef<'_, Self::T>,
                d2_k: MatMut<'_, Self::T>,
                pair: (usize, usize),
                uplo: Triangle,
                scratch: MatMut<'_, Self::T>,
            ) -> Result<(), GprError> {
                CompiledKernel::<$t>::hess_from_ard_cache::<M>(
                    self, cache, x, d2_k, pair, uplo, scratch,
                )
            }

            fn hess_mixed<M: crate::math::KernelMath>(
                &self,
                views: MixedKernelViews<'_, Self::T>,
                d2_k: MatMut<'_, Self::T>,
                i: usize,
                j: usize,
                uplo: Triangle,
                scratch: MatMut<'_, Self::T>,
            ) -> Result<(), GprError> {
                CompiledKernel::<$t>::hess_mixed::<M>(self, views, d2_k, i, j, uplo, scratch)
            }

            fn hess_cross_points<M: crate::math::KernelMath>(
                &self,
                x1: MatRef<'_, Self::T>,
                x2: MatRef<'_, Self::T>,
                d2_k: MatMut<'_, Self::T>,
                i: usize,
                j: usize,
                scratch: MatMut<'_, Self::T>,
            ) -> Result<(), GprError> {
                CompiledKernel::<$t>::hess_cross_points::<M>(self, x1, x2, d2_k, i, j, scratch)
            }

            fn hess_diag_points<M: crate::math::KernelMath>(
                &self,
                x: MatRef<'_, Self::T>,
                out: &mut [Self::T],
                i: usize,
                j: usize,
            ) -> Result<(), GprError> {
                CompiledKernel::<$t>::hess_diag_points::<M>(self, x, out, i, j)
            }

            fn hess_wrt_coord_dims<M: crate::math::KernelMath>(
                &self,
                x1: MatRef<'_, Self::T>,
                x2: MatRef<'_, Self::T>,
                d2_k: MatMut<'_, Self::T>,
                dim_a: usize,
                dim_b: usize,
            ) -> Result<(), GprError> {
                CompiledKernel::<$t>::hess_wrt_coord_dims::<M>(self, x1, x2, d2_k, dim_a, dim_b)
            }

            fn hess_wrt_coord_mixed<M: crate::math::KernelMath>(
                &self,
                x1: MatRef<'_, Self::T>,
                x2: MatRef<'_, Self::T>,
                d2_k: MatMut<'_, Self::T>,
                dim_x1: usize,
                dim_x2: usize,
            ) -> Result<(), GprError> {
                CompiledKernel::<$t>::hess_wrt_coord_mixed::<M>(self, x1, x2, d2_k, dim_x1, dim_x2)
            }

            fn hess_theta_coord_dim<M: crate::math::KernelMath>(
                &self,
                x1: MatRef<'_, Self::T>,
                x2: MatRef<'_, Self::T>,
                d2_k: MatMut<'_, Self::T>,
                param_idx: usize,
                dim: usize,
            ) -> Result<(), GprError> {
                CompiledKernel::<$t>::hess_theta_coord_dim::<M>(self, x1, x2, d2_k, param_idx, dim)
            }
        }
    };
}

forward_gram!(f64);
forward_gram!(f32);
