//! One evaluation surface for [`CompiledKernel<f32>`](super::CompiledKernel) and
//! [`CompiledKernel<f64>`](super::CompiledKernel).

use faer::{MatMut, MatRef};

use crate::error::GprError;
use crate::kernel::Triangle;

use super::{CompiledKernel, CoordMode, MixedKernelViews};

/// Kernel apply / gradient / Hessian used by fit and predict.
///
/// `f64` keeps the distance-cache readers. `f32` evaluates the same formulas
/// in scalar arithmetic.
pub(crate) trait GramKernel: Clone {
    type T: Copy;

    fn coord_mode(&self) -> Result<CoordMode, GprError>;

    fn needs_ard_sq_diff(&self) -> bool;

    fn apply(
        &self,
        dist: MatRef<'_, Self::T>,
        out: MatMut<'_, Self::T>,
        uplo: Triangle,
        scratch: MatMut<'_, Self::T>,
    ) -> Result<(), GprError>;

    fn apply_points(
        &self,
        x: MatRef<'_, Self::T>,
        out: MatMut<'_, Self::T>,
        uplo: Triangle,
        scratch: MatMut<'_, Self::T>,
    ) -> Result<(), GprError>;

    fn apply_from_ard_cache(
        &self,
        cache: MatRef<'_, Self::T>,
        x: MatRef<'_, Self::T>,
        out: MatMut<'_, Self::T>,
        uplo: Triangle,
        scratch: MatMut<'_, Self::T>,
    ) -> Result<(), GprError>;

    fn apply_mixed(
        &self,
        views: MixedKernelViews<'_, Self::T>,
        out: MatMut<'_, Self::T>,
        uplo: Triangle,
        scratch: MatMut<'_, Self::T>,
    ) -> Result<(), GprError>;

    fn apply_cross(
        &self,
        dist: MatRef<'_, Self::T>,
        out: MatMut<'_, Self::T>,
        scratch: MatMut<'_, Self::T>,
    ) -> Result<(), GprError>;

    fn apply_cross_points(
        &self,
        x: MatRef<'_, Self::T>,
        xs: MatRef<'_, Self::T>,
        out: MatMut<'_, Self::T>,
        scratch: MatMut<'_, Self::T>,
    ) -> Result<(), GprError>;

    fn apply_cross_mixed(
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

    fn grad(
        &self,
        dist: MatRef<'_, Self::T>,
        d_k: MatMut<'_, Self::T>,
        param_idx: usize,
        uplo: Triangle,
        scratch: MatMut<'_, Self::T>,
    ) -> Result<(), GprError>;

    fn grad_points(
        &self,
        x: MatRef<'_, Self::T>,
        d_k: MatMut<'_, Self::T>,
        param_idx: usize,
        uplo: Triangle,
        scratch: MatMut<'_, Self::T>,
    ) -> Result<(), GprError>;

    fn grad_from_ard_cache(
        &self,
        cache: MatRef<'_, Self::T>,
        x: MatRef<'_, Self::T>,
        d_k: MatMut<'_, Self::T>,
        param_idx: usize,
        uplo: Triangle,
        scratch: MatMut<'_, Self::T>,
    ) -> Result<(), GprError>;

    fn grad_mixed(
        &self,
        views: MixedKernelViews<'_, Self::T>,
        d_k: MatMut<'_, Self::T>,
        param_idx: usize,
        uplo: Triangle,
        scratch: MatMut<'_, Self::T>,
    ) -> Result<(), GprError>;

    fn grad_cross_points(
        &self,
        x1: MatRef<'_, Self::T>,
        x2: MatRef<'_, Self::T>,
        d_k: MatMut<'_, Self::T>,
        param_idx: usize,
        scratch: MatMut<'_, Self::T>,
    ) -> Result<(), GprError>;

    fn grad_diag_points(
        &self,
        x: MatRef<'_, Self::T>,
        out: &mut [Self::T],
        param_idx: usize,
    ) -> Result<(), GprError>;

    fn grad_wrt_coord_dim(
        &self,
        x1: MatRef<'_, Self::T>,
        x2: MatRef<'_, Self::T>,
        d_k: MatMut<'_, Self::T>,
        dim: usize,
    ) -> Result<(), GprError>;

    fn hess(
        &self,
        dist: MatRef<'_, Self::T>,
        d2_k: MatMut<'_, Self::T>,
        i: usize,
        j: usize,
        uplo: Triangle,
        scratch: MatMut<'_, Self::T>,
    ) -> Result<(), GprError>;

    fn hess_points(
        &self,
        x: MatRef<'_, Self::T>,
        d2_k: MatMut<'_, Self::T>,
        i: usize,
        j: usize,
        uplo: Triangle,
        scratch: MatMut<'_, Self::T>,
    ) -> Result<(), GprError>;

    fn hess_from_ard_cache(
        &self,
        cache: MatRef<'_, Self::T>,
        x: MatRef<'_, Self::T>,
        d2_k: MatMut<'_, Self::T>,
        pair: (usize, usize),
        uplo: Triangle,
        scratch: MatMut<'_, Self::T>,
    ) -> Result<(), GprError>;

    fn hess_mixed(
        &self,
        views: MixedKernelViews<'_, Self::T>,
        d2_k: MatMut<'_, Self::T>,
        i: usize,
        j: usize,
        uplo: Triangle,
        scratch: MatMut<'_, Self::T>,
    ) -> Result<(), GprError>;

    fn hess_cross_points(
        &self,
        x1: MatRef<'_, Self::T>,
        x2: MatRef<'_, Self::T>,
        d2_k: MatMut<'_, Self::T>,
        i: usize,
        j: usize,
        scratch: MatMut<'_, Self::T>,
    ) -> Result<(), GprError>;

    fn hess_diag_points(
        &self,
        x: MatRef<'_, Self::T>,
        out: &mut [Self::T],
        i: usize,
        j: usize,
    ) -> Result<(), GprError>;

    fn hess_wrt_coord_dims(
        &self,
        x1: MatRef<'_, Self::T>,
        x2: MatRef<'_, Self::T>,
        d2_k: MatMut<'_, Self::T>,
        dim_a: usize,
        dim_b: usize,
    ) -> Result<(), GprError>;

    fn hess_wrt_coord_mixed(
        &self,
        x1: MatRef<'_, Self::T>,
        x2: MatRef<'_, Self::T>,
        d2_k: MatMut<'_, Self::T>,
        dim_x1: usize,
        dim_x2: usize,
    ) -> Result<(), GprError>;

    fn hess_theta_coord_dim(
        &self,
        x1: MatRef<'_, Self::T>,
        x2: MatRef<'_, Self::T>,
        d2_k: MatMut<'_, Self::T>,
        param_idx: usize,
        dim: usize,
    ) -> Result<(), GprError>;
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

            fn apply(
                &self,
                dist: MatRef<'_, Self::T>,
                out: MatMut<'_, Self::T>,
                uplo: Triangle,
                scratch: MatMut<'_, Self::T>,
            ) -> Result<(), GprError> {
                CompiledKernel::<$t>::apply(self, dist, out, uplo, scratch)
            }

            fn apply_points(
                &self,
                x: MatRef<'_, Self::T>,
                out: MatMut<'_, Self::T>,
                uplo: Triangle,
                scratch: MatMut<'_, Self::T>,
            ) -> Result<(), GprError> {
                CompiledKernel::<$t>::apply_points(self, x, out, uplo, scratch)
            }

            fn apply_from_ard_cache(
                &self,
                cache: MatRef<'_, Self::T>,
                x: MatRef<'_, Self::T>,
                out: MatMut<'_, Self::T>,
                uplo: Triangle,
                scratch: MatMut<'_, Self::T>,
            ) -> Result<(), GprError> {
                CompiledKernel::<$t>::apply_from_ard_cache(self, cache, x, out, uplo, scratch)
            }

            fn apply_mixed(
                &self,
                views: MixedKernelViews<'_, Self::T>,
                out: MatMut<'_, Self::T>,
                uplo: Triangle,
                scratch: MatMut<'_, Self::T>,
            ) -> Result<(), GprError> {
                CompiledKernel::<$t>::apply_mixed(self, views, out, uplo, scratch)
            }

            fn apply_cross(
                &self,
                dist: MatRef<'_, Self::T>,
                out: MatMut<'_, Self::T>,
                scratch: MatMut<'_, Self::T>,
            ) -> Result<(), GprError> {
                CompiledKernel::<$t>::apply_cross(self, dist, out, scratch)
            }

            fn apply_cross_points(
                &self,
                x: MatRef<'_, Self::T>,
                xs: MatRef<'_, Self::T>,
                out: MatMut<'_, Self::T>,
                scratch: MatMut<'_, Self::T>,
            ) -> Result<(), GprError> {
                CompiledKernel::<$t>::apply_cross_points(self, x, xs, out, scratch)
            }

            fn apply_cross_mixed(
                &self,
                dist: MatRef<'_, Self::T>,
                x: MatRef<'_, Self::T>,
                xs: MatRef<'_, Self::T>,
                out: MatMut<'_, Self::T>,
                scratch: MatMut<'_, Self::T>,
            ) -> Result<(), GprError> {
                CompiledKernel::<$t>::apply_cross_mixed(self, dist, x, xs, out, scratch)
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

            fn grad(
                &self,
                dist: MatRef<'_, Self::T>,
                d_k: MatMut<'_, Self::T>,
                param_idx: usize,
                uplo: Triangle,
                scratch: MatMut<'_, Self::T>,
            ) -> Result<(), GprError> {
                CompiledKernel::<$t>::grad(self, dist, d_k, param_idx, uplo, scratch)
            }

            fn grad_points(
                &self,
                x: MatRef<'_, Self::T>,
                d_k: MatMut<'_, Self::T>,
                param_idx: usize,
                uplo: Triangle,
                scratch: MatMut<'_, Self::T>,
            ) -> Result<(), GprError> {
                CompiledKernel::<$t>::grad_points(self, x, d_k, param_idx, uplo, scratch)
            }

            fn grad_from_ard_cache(
                &self,
                cache: MatRef<'_, Self::T>,
                x: MatRef<'_, Self::T>,
                d_k: MatMut<'_, Self::T>,
                param_idx: usize,
                uplo: Triangle,
                scratch: MatMut<'_, Self::T>,
            ) -> Result<(), GprError> {
                CompiledKernel::<$t>::grad_from_ard_cache(
                    self, cache, x, d_k, param_idx, uplo, scratch,
                )
            }

            fn grad_mixed(
                &self,
                views: MixedKernelViews<'_, Self::T>,
                d_k: MatMut<'_, Self::T>,
                param_idx: usize,
                uplo: Triangle,
                scratch: MatMut<'_, Self::T>,
            ) -> Result<(), GprError> {
                CompiledKernel::<$t>::grad_mixed(self, views, d_k, param_idx, uplo, scratch)
            }

            fn grad_cross_points(
                &self,
                x1: MatRef<'_, Self::T>,
                x2: MatRef<'_, Self::T>,
                d_k: MatMut<'_, Self::T>,
                param_idx: usize,
                scratch: MatMut<'_, Self::T>,
            ) -> Result<(), GprError> {
                CompiledKernel::<$t>::grad_cross_points(self, x1, x2, d_k, param_idx, scratch)
            }

            fn grad_diag_points(
                &self,
                x: MatRef<'_, Self::T>,
                out: &mut [Self::T],
                param_idx: usize,
            ) -> Result<(), GprError> {
                CompiledKernel::<$t>::grad_diag_points(self, x, out, param_idx)
            }

            fn grad_wrt_coord_dim(
                &self,
                x1: MatRef<'_, Self::T>,
                x2: MatRef<'_, Self::T>,
                d_k: MatMut<'_, Self::T>,
                dim: usize,
            ) -> Result<(), GprError> {
                CompiledKernel::<$t>::grad_wrt_coord_dim(self, x1, x2, d_k, dim)
            }

            fn hess(
                &self,
                dist: MatRef<'_, Self::T>,
                d2_k: MatMut<'_, Self::T>,
                i: usize,
                j: usize,
                uplo: Triangle,
                scratch: MatMut<'_, Self::T>,
            ) -> Result<(), GprError> {
                CompiledKernel::<$t>::hess(self, dist, d2_k, i, j, uplo, scratch)
            }

            fn hess_points(
                &self,
                x: MatRef<'_, Self::T>,
                d2_k: MatMut<'_, Self::T>,
                i: usize,
                j: usize,
                uplo: Triangle,
                scratch: MatMut<'_, Self::T>,
            ) -> Result<(), GprError> {
                CompiledKernel::<$t>::hess_points(self, x, d2_k, i, j, uplo, scratch)
            }

            fn hess_from_ard_cache(
                &self,
                cache: MatRef<'_, Self::T>,
                x: MatRef<'_, Self::T>,
                d2_k: MatMut<'_, Self::T>,
                pair: (usize, usize),
                uplo: Triangle,
                scratch: MatMut<'_, Self::T>,
            ) -> Result<(), GprError> {
                CompiledKernel::<$t>::hess_from_ard_cache(self, cache, x, d2_k, pair, uplo, scratch)
            }

            fn hess_mixed(
                &self,
                views: MixedKernelViews<'_, Self::T>,
                d2_k: MatMut<'_, Self::T>,
                i: usize,
                j: usize,
                uplo: Triangle,
                scratch: MatMut<'_, Self::T>,
            ) -> Result<(), GprError> {
                CompiledKernel::<$t>::hess_mixed(self, views, d2_k, i, j, uplo, scratch)
            }

            fn hess_cross_points(
                &self,
                x1: MatRef<'_, Self::T>,
                x2: MatRef<'_, Self::T>,
                d2_k: MatMut<'_, Self::T>,
                i: usize,
                j: usize,
                scratch: MatMut<'_, Self::T>,
            ) -> Result<(), GprError> {
                CompiledKernel::<$t>::hess_cross_points(self, x1, x2, d2_k, i, j, scratch)
            }

            fn hess_diag_points(
                &self,
                x: MatRef<'_, Self::T>,
                out: &mut [Self::T],
                i: usize,
                j: usize,
            ) -> Result<(), GprError> {
                CompiledKernel::<$t>::hess_diag_points(self, x, out, i, j)
            }

            fn hess_wrt_coord_dims(
                &self,
                x1: MatRef<'_, Self::T>,
                x2: MatRef<'_, Self::T>,
                d2_k: MatMut<'_, Self::T>,
                dim_a: usize,
                dim_b: usize,
            ) -> Result<(), GprError> {
                CompiledKernel::<$t>::hess_wrt_coord_dims(self, x1, x2, d2_k, dim_a, dim_b)
            }

            fn hess_wrt_coord_mixed(
                &self,
                x1: MatRef<'_, Self::T>,
                x2: MatRef<'_, Self::T>,
                d2_k: MatMut<'_, Self::T>,
                dim_x1: usize,
                dim_x2: usize,
            ) -> Result<(), GprError> {
                CompiledKernel::<$t>::hess_wrt_coord_mixed(self, x1, x2, d2_k, dim_x1, dim_x2)
            }

            fn hess_theta_coord_dim(
                &self,
                x1: MatRef<'_, Self::T>,
                x2: MatRef<'_, Self::T>,
                d2_k: MatMut<'_, Self::T>,
                param_idx: usize,
                dim: usize,
            ) -> Result<(), GprError> {
                CompiledKernel::<$t>::hess_theta_coord_dim(self, x1, x2, d2_k, param_idx, dim)
            }
        }
    };
}

forward_gram!(f64);
forward_gram!(f32);
