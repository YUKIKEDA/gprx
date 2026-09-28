//! Fit / predict entry points of [`CompiledKernel`](super::CompiledKernel).

use faer::{Mat, MatMut, MatRef};

use crate::error::GprError;
use crate::kernel::{KernelScalar, Triangle};

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

/// Fit / predict entry points. Each picks the distance, coordinate,
/// ARD-cache, or mixed path from [`CompiledKernel::coord_mode`] and the
/// inputs at hand.
impl<T: KernelScalar> CompiledKernel<T> {
    /// Whether this tree reads a squared-Euclidean distance matrix.
    pub(crate) fn reads_distances(&self) -> Result<bool, GprError> {
        Ok(!matches!(self.coord_mode()?, CoordMode::Points))
    }

    /// Writes `K` for `uplo` from whichever views `inputs` holds.
    pub(crate) fn eval_gram<M: crate::math::KernelMath>(
        &self,
        inputs: GramInputs<'_, T>,
        out: MatMut<'_, T>,
        uplo: Triangle,
        scratch: MatMut<'_, T>,
    ) -> Result<(), GprError> {
        let ard = self.ard_view(inputs.ard);
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
    pub(crate) fn eval_gram_from_points<M: crate::math::KernelMath>(
        &self,
        x: MatRef<'_, T>,
        out: MatMut<'_, T>,
        uplo: Triangle,
        scratch: MatMut<'_, T>,
        thread_scratch: &mut [Mat<T>],
    ) -> Result<(), GprError> {
        if !self.reads_distances()? {
            return self.eval_gram::<M>(GramInputs::points(x), out, uplo, scratch);
        }
        let m = x.nrows();
        let mut dist = Mat::<T>::zeros(m, m);
        T::write_squared(x, dist.as_mut(), thread_scratch);
        let inputs = GramInputs {
            x,
            dist: Some(dist.as_ref()),
            ard: None,
        };
        self.eval_gram::<M>(inputs, out, uplo, scratch)
    }

    /// Writes `∂K/∂θ_{param_idx}` for `uplo`.
    pub(crate) fn grad_gram<M: crate::math::KernelMath>(
        &self,
        inputs: GramInputs<'_, T>,
        d_k: MatMut<'_, T>,
        param_idx: usize,
        uplo: Triangle,
        scratch: MatMut<'_, T>,
    ) -> Result<(), GprError> {
        let ard = self.ard_view(inputs.ard);
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
    pub(crate) fn hess_gram<M: crate::math::KernelMath>(
        &self,
        inputs: GramInputs<'_, T>,
        d2_k: MatMut<'_, T>,
        pair: (usize, usize),
        uplo: Triangle,
        scratch: MatMut<'_, T>,
    ) -> Result<(), GprError> {
        let (i, j) = pair;
        let ard = self.ard_view(inputs.ard);
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
    pub(crate) fn eval_cross<M: crate::math::KernelMath>(
        &self,
        x: MatRef<'_, T>,
        xs: MatRef<'_, T>,
        dist: Option<MatMut<'_, T>>,
        out: MatMut<'_, T>,
        scratch: MatMut<'_, T>,
        thread_scratch: &mut [Mat<T>],
    ) -> Result<(), GprError> {
        let mode = self.coord_mode()?;
        if matches!(mode, CoordMode::Points) {
            return self.apply_cross_points::<M>(x, xs, out, scratch);
        }
        let mut owned;
        let mut dist = match dist {
            Some(dist) => dist,
            None => {
                owned = Mat::<T>::zeros(x.nrows(), xs.nrows());
                owned.as_mut()
            }
        };
        T::write_cross(x, xs, dist.as_mut(), thread_scratch);
        match mode {
            CoordMode::Mixed => self.apply_cross_mixed::<M>(dist.as_ref(), x, xs, out, scratch),
            _ => self.apply_cross::<M>(dist.as_ref(), out, scratch),
        }
    }

    /// Writes the diagonal `k(x_i, x_i)` into `out`.
    pub(crate) fn eval_diag(&self, x: MatRef<'_, T>, out: &mut [T]) -> Result<(), GprError> {
        match self.coord_mode()? {
            CoordMode::Dist | CoordMode::Either => self.fill_diag(out),
            CoordMode::Points | CoordMode::Mixed => self.fill_diag_points(x, out),
        }
    }

    /// The ARD cache when this scalar reads it and the tree has ARD leaves.
    fn ard_view<'a>(&self, ard: Option<MatRef<'a, T>>) -> Option<MatRef<'a, T>> {
        if T::READS_ARD_CACHE && self.needs_ard_sq_diff() {
            ard.filter(|cache| cache.ncols() > 0)
        } else {
            None
        }
    }
}
