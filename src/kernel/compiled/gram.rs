//! Fit / predict entry points of [`CompiledKernel`](super::CompiledKernel).

use crate::kernel::dist::ArdSqDiff;
use faer::{Mat, MatMut, MatRef};

use crate::error::GprError;
use crate::kernel::{KernelScalar, Triangle};

use super::supplied::{RectSlots, SquareSlots};
use super::{CompiledKernel, CoordMode, CrossViews, MixedKernelViews, ensure_nested};

/// Views that a square (training) Gram evaluation can read.
///
/// `x` is always present. `dist` is the filled squared-Euclidean matrix and
/// `ard` the filled raw `(Δx_d)²` cache, when the caller keeps them. A kernel
/// reads what its leaves need and falls back to `x` for the rest. `slots`
/// holds the supplied distances of a distance model.
#[derive(Clone, Copy)]
pub(crate) struct GramInputs<'a, T> {
    pub(crate) x: MatRef<'a, T>,
    pub(crate) dist: Option<MatRef<'a, T>>,
    pub(crate) ard: Option<ArdSqDiff<'a, T>>,
    pub(crate) slots: Option<&'a dyn SquareSlots<T>>,
}

impl<'a, T> GramInputs<'a, T> {
    /// Coordinates only. Distance leaves compute `‖x_i − x_j‖²` per pair.
    pub(crate) fn points(x: MatRef<'a, T>) -> Self {
        Self {
            x,
            dist: None,
            ard: None,
            slots: None,
        }
    }

    fn mixed(self, ard: Option<ArdSqDiff<'a, T>>) -> MixedKernelViews<'a, T> {
        MixedKernelViews {
            dist: self.dist,
            x: self.x,
            ard_cache: ard,
            slots: self.slots,
        }
    }
}

/// Fit / predict entry points. Each picks the distance, coordinate,
/// ARD-cache, or mixed path from [`CompiledKernel::coord_mode`] and the
/// inputs at hand.
///
/// `nested` holds the [`super::Nested`] levels. Each entry point grows it to
/// this tree's depth and the output's shape, so a caller that keeps it
/// allocates only on the first call.
impl<T: KernelScalar> CompiledKernel<T> {
    /// Whether this tree reads a squared-Euclidean distance matrix of its
    /// coordinates. Supplied distances are not one: a tree of supplied
    /// leaves (and Constant / White) reads none.
    pub(crate) fn reads_distances(&self) -> Result<bool, GprError> {
        if self.has_supplied() {
            return Ok(self.has_coord_dist_leaf());
        }
        Ok(!matches!(self.coord_mode()?, CoordMode::Points))
    }

    /// Writes `K` for `uplo` from whichever views `inputs` holds.
    pub(crate) fn eval_gram<M: crate::math::KernelMath>(
        &self,
        inputs: GramInputs<'_, T>,
        out: MatMut<'_, T>,
        uplo: Triangle,
        scratch: MatMut<'_, T>,
        nested: &mut Vec<Mat<T>>,
    ) -> Result<(), GprError> {
        ensure_nested(nested, self.nested_depth(), out.nrows(), out.ncols());
        let ard = self.ard_view(inputs.ard);
        match (self.coord_mode()?, inputs.dist) {
            (CoordMode::Dist | CoordMode::Either, Some(dist)) => {
                self.apply_with::<M>(dist, out, uplo, scratch, nested)
            }
            (CoordMode::Points, _) => match ard {
                Some(cache) => {
                    self.apply_from_ard_cache::<M>(cache, inputs.x, out, uplo, scratch, nested)
                }
                None => self.apply_points_with::<M>(inputs.x, out, uplo, scratch, nested),
            },
            (CoordMode::Mixed, _) => {
                self.apply_mixed::<M>(inputs.mixed(ard), out, uplo, scratch, nested)
            }
            (_, None) => self.apply_points_with::<M>(inputs.x, out, uplo, scratch, nested),
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
        nested: &mut Vec<Mat<T>>,
        thread_scratch: &mut [Mat<T>],
    ) -> Result<(), GprError> {
        if !self.reads_distances()? {
            return self.eval_gram::<M>(GramInputs::points(x), out, uplo, scratch, nested);
        }
        let m = x.nrows();
        let mut dist = Mat::<T>::zeros(m, m);
        T::write_squared(x, dist.as_mut(), thread_scratch);
        let inputs = GramInputs {
            x,
            dist: Some(dist.as_ref()),
            ard: None,
            slots: None,
        };
        self.eval_gram::<M>(inputs, out, uplo, scratch, nested)
    }

    /// Writes `∂K/∂θ_{param_idx}` for `uplo`.
    pub(crate) fn grad_gram<M: crate::math::KernelMath>(
        &self,
        inputs: GramInputs<'_, T>,
        d_k: MatMut<'_, T>,
        param_idx: usize,
        uplo: Triangle,
        scratch: MatMut<'_, T>,
        nested: &mut Vec<Mat<T>>,
    ) -> Result<(), GprError> {
        ensure_nested(nested, self.nested_depth(), d_k.nrows(), d_k.ncols());
        let ard = self.ard_view(inputs.ard);
        match (self.coord_mode()?, inputs.dist) {
            (CoordMode::Dist | CoordMode::Either, Some(dist)) => {
                self.grad_with::<M>(dist, d_k, param_idx, uplo, scratch, nested)
            }
            (CoordMode::Points, _) => match ard {
                Some(cache) => self.grad_from_ard_cache::<M>(
                    cache, inputs.x, d_k, param_idx, uplo, scratch, nested,
                ),
                None => self.grad_points_with::<M>(inputs.x, d_k, param_idx, uplo, scratch, nested),
            },
            (CoordMode::Mixed, _) => {
                self.grad_mixed::<M>(inputs.mixed(ard), d_k, param_idx, uplo, scratch, nested)
            }
            (_, None) => {
                self.grad_points_with::<M>(inputs.x, d_k, param_idx, uplo, scratch, nested)
            }
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
        nested: &mut Vec<Mat<T>>,
    ) -> Result<(), GprError> {
        ensure_nested(nested, self.nested_depth(), d2_k.nrows(), d2_k.ncols());
        let ard = self.ard_view(inputs.ard);
        match (self.coord_mode()?, inputs.dist) {
            (CoordMode::Dist | CoordMode::Either, Some(dist)) => {
                self.hess_with::<M>(dist, d2_k, pair, uplo, scratch, nested)
            }
            (CoordMode::Points, _) => match ard {
                Some(cache) => self
                    .hess_from_ard_cache::<M>(cache, inputs.x, d2_k, pair, uplo, scratch, nested),
                None => self.hess_points_with::<M>(inputs.x, d2_k, pair, uplo, scratch, nested),
            },
            (CoordMode::Mixed, _) => {
                self.hess_mixed::<M>(inputs.mixed(ard), d2_k, pair, uplo, scratch, nested)
            }
            (_, None) => self.hess_points_with::<M>(inputs.x, d2_k, pair, uplo, scratch, nested),
        }
    }

    /// [`Self::eval_cross_slots`] of a coordinate tree.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn eval_cross<M: crate::math::KernelMath>(
        &self,
        x: MatRef<'_, T>,
        xs: MatRef<'_, T>,
        dist: Option<MatMut<'_, T>>,
        out: MatMut<'_, T>,
        scratch: MatMut<'_, T>,
        nested: &mut Vec<Mat<T>>,
        thread_scratch: &mut [Mat<T>],
    ) -> Result<(), GprError> {
        self.eval_cross_slots::<M>(x, xs, None, dist, out, scratch, nested, thread_scratch)
    }

    /// Writes the rectangular `K(x, xs)` into `out`.
    ///
    /// `dist` receives the train–query squared distances when the tree reads
    /// them; `None` allocates that buffer for this call. `slots` holds the
    /// supplied distances of the block for a distance model.
    // Both views, the supplies, the distance and output blocks, and three scratch kinds.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn eval_cross_slots<M: crate::math::KernelMath>(
        &self,
        x: MatRef<'_, T>,
        xs: MatRef<'_, T>,
        slots: Option<&dyn RectSlots<T>>,
        dist: Option<MatMut<'_, T>>,
        out: MatMut<'_, T>,
        scratch: MatMut<'_, T>,
        nested: &mut Vec<Mat<T>>,
        thread_scratch: &mut [Mat<T>],
    ) -> Result<(), GprError> {
        ensure_nested(nested, self.nested_depth(), out.nrows(), out.ncols());
        let mode = self.coord_mode()?;
        if matches!(mode, CoordMode::Points) {
            return self.apply_cross_points_with::<M>(x, xs, out, scratch, nested);
        }
        if !self.reads_distances()? {
            let views = CrossViews {
                x1: x,
                x2: xs,
                dist: None,
                slots,
            };
            return self.apply_cross_mixed::<M>(views, out, scratch, nested);
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
            CoordMode::Mixed => {
                let views = CrossViews {
                    x1: x,
                    x2: xs,
                    dist: Some(dist.as_ref()),
                    slots,
                };
                self.apply_cross_mixed::<M>(views, out, scratch, nested)
            }
            _ => self.apply_cross_with::<M>(dist.as_ref(), out, scratch, nested),
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
    fn ard_view<'a>(&self, ard: Option<ArdSqDiff<'a, T>>) -> Option<ArdSqDiff<'a, T>> {
        if T::READS_ARD_CACHE && self.needs_ard_sq_diff() {
            ard.filter(|cache| cache.n() > 0 && cache.d() > 0)
        } else {
            None
        }
    }
}
