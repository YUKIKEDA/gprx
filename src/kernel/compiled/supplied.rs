//! Leaves that read supplied squared distances ([`SuppliedLeaf`]), and the
//! views of those supplies a kernel evaluation reads.
//!
//! A supplied leaf wraps a built-in or custom leaf and names its slot. The
//! square views hold a slot's `d²` between the points of one set (a dense
//! square for a scalar slot, the packed `(Δ_d)²` lower triangles of
//! [`ArdSqDiff`] for an ARD slot); the rectangular views hold the block
//! between two sets (dense, or one dense block per dimension). Each walk
//! hands the wrapped leaf the view of its own slot in place of the
//! coordinate distances.

use faer::{MatMut, MatRef};

use super::CompiledKernel;
use super::grad::eval_cell;
use crate::error::GprError;
use crate::kernel::dist::{ArdBlocks, ArdSqDiff};
use crate::kernel::{KernelScalar, SlotId, Triangle};

/// A compiled leaf that reads the supplied `d²` of one slot.
#[derive(Clone, Debug, PartialEq)]
pub struct SuppliedLeaf<T: KernelScalar> {
    pub(crate) slot: SlotId,
    pub(crate) leaf: Box<CompiledKernel<T>>,
}

/// One slot's `d²` between the points of one set.
#[derive(Clone, Copy, Debug)]
pub enum SquareSlot<'a, T> {
    /// Dense symmetric `n × n`.
    Scalar(MatRef<'a, T>),
    /// Packed lower triangles, one per dimension.
    Ard(ArdSqDiff<'a, T>),
}

/// One slot's `d²` between the points of two sets.
#[derive(Clone, Copy, Debug)]
pub enum RectSlot<'a, T> {
    /// Dense `rows × cols`.
    Scalar(MatRef<'a, T>),
    /// One dense `rows × cols` block per dimension.
    Ard(ArdBlocks<'a, T>),
}

/// The square supplies of every slot of a tree.
pub trait SquareSlots<T>: Sync {
    /// The supply of `slot`, or `None` when this set does not hold it.
    fn square(&self, slot: SlotId) -> Option<SquareSlot<'_, T>>;
}

/// The rectangular supplies of every slot of a tree.
pub trait RectSlots<T>: Sync {
    /// The supply of `slot`, or `None` when this set does not hold it.
    fn rect(&self, slot: SlotId) -> Option<RectSlot<'_, T>>;
}

fn missing() -> GprError {
    GprError::UnsupportedKernelOperation {
        reason: "a distance leaf has no supplied squared distances here".to_owned(),
    }
}

pub(super) fn needs_supply() -> GprError {
    GprError::UnsupportedKernelOperation {
        reason: "a distance leaf reads supplied squared distances, not coordinates".to_owned(),
    }
}

pub(super) fn square_slot<'a, T>(
    slots: Option<&'a dyn SquareSlots<T>>,
    slot: SlotId,
) -> Result<SquareSlot<'a, T>, GprError> {
    slots.and_then(|s| s.square(slot)).ok_or_else(missing)
}

pub(super) fn rect_slot<'a, T>(
    slots: Option<&'a dyn RectSlots<T>>,
    slot: SlotId,
) -> Result<RectSlot<'a, T>, GprError> {
    slots.and_then(|s| s.rect(slot)).ok_or_else(missing)
}

impl<T: KernelScalar> SuppliedLeaf<T> {
    /// `K` for `uplo` from the square supply.
    pub(super) fn apply<M: crate::math::KernelMath>(
        &self,
        slot: SquareSlot<'_, T>,
        x: MatRef<'_, T>,
        out: MatMut<'_, T>,
        uplo: Triangle,
        scratch: MatMut<'_, T>,
    ) -> Result<(), GprError> {
        match slot {
            SquareSlot::Scalar(d) => self.leaf.apply_with::<M>(d, out, uplo, scratch, &mut []),
            SquareSlot::Ard(c) => {
                self.leaf
                    .apply_from_ard_cache::<M>(c, x, out, uplo, scratch, &mut [])
            }
        }
    }

    /// `∂K/∂θ_p` for `uplo` from the square supply.
    pub(super) fn grad<M: crate::math::KernelMath>(
        &self,
        slot: SquareSlot<'_, T>,
        x: MatRef<'_, T>,
        d_k: MatMut<'_, T>,
        param_idx: usize,
        uplo: Triangle,
        scratch: MatMut<'_, T>,
    ) -> Result<(), GprError> {
        match slot {
            SquareSlot::Scalar(d) => {
                self.leaf
                    .grad_with::<M>(d, d_k, param_idx, uplo, scratch, &mut [])
            }
            SquareSlot::Ard(c) => {
                self.leaf
                    .grad_from_ard_cache::<M>(c, x, d_k, param_idx, uplo, scratch, &mut [])
            }
        }
    }

    /// `∂²K/∂θ_i ∂θ_j` for `uplo` from the square supply.
    pub(super) fn hess<M: crate::math::KernelMath>(
        &self,
        slot: SquareSlot<'_, T>,
        x: MatRef<'_, T>,
        d2_k: MatMut<'_, T>,
        pair: (usize, usize),
        uplo: Triangle,
        scratch: MatMut<'_, T>,
    ) -> Result<(), GprError> {
        match slot {
            SquareSlot::Scalar(d) => {
                self.leaf
                    .hess_with::<M>(d, d2_k, pair, uplo, scratch, &mut [])
            }
            SquareSlot::Ard(c) => {
                self.leaf
                    .hess_from_ard_cache::<M>(c, x, d2_k, pair, uplo, scratch, &mut [])
            }
        }
    }

    /// The rectangular `K` from the rectangular supply.
    pub(super) fn apply_cross<M: crate::math::KernelMath>(
        &self,
        slot: RectSlot<'_, T>,
        out: MatMut<'_, T>,
        scratch: MatMut<'_, T>,
    ) -> Result<(), GprError> {
        match slot {
            RectSlot::Scalar(d) => self.leaf.apply_cross_with::<M>(d, out, scratch, &mut []),
            RectSlot::Ard(b) => match self.leaf.as_ref() {
                CompiledKernel::RbfArd(leaf) => leaf.apply_cross_from_blocks::<M, T>(b, out),
                CompiledKernel::MaternArd(leaf) => leaf.apply_cross_from_blocks::<M, T>(b, out),
                CompiledKernel::RationalQuadraticArd(leaf) => leaf.apply_cross_from_blocks(b, out),
                _ => Err(not_ard()),
            },
        }
    }

    /// The rectangular `∂K/∂θ_p` from the rectangular supply.
    pub(super) fn grad_cross<M: crate::math::KernelMath>(
        &self,
        slot: RectSlot<'_, T>,
        d_k: MatMut<'_, T>,
        param_idx: usize,
    ) -> Result<(), GprError> {
        match slot {
            RectSlot::Scalar(d) => match self.leaf.as_ref() {
                CompiledKernel::Rbf(leaf) => leaf.grad_cross_dist::<M, T>(d, d_k, param_idx),
                CompiledKernel::Matern(leaf) => leaf.grad_cross_dist::<M, T>(d, d_k, param_idx),
                CompiledKernel::Periodic(leaf) => leaf.grad_cross_dist::<M, T>(d, d_k, param_idx),
                CompiledKernel::RationalQuadratic(leaf) => leaf.grad_cross_dist(d, d_k, param_idx),
                CompiledKernel::Custom(leaf) => leaf.grad_cross(d, d_k, param_idx),
                _ => Err(not_scalar()),
            },
            RectSlot::Ard(b) => match self.leaf.as_ref() {
                CompiledKernel::RbfArd(leaf) => {
                    leaf.grad_cross_from_blocks::<M, T>(b, d_k, param_idx)
                }
                CompiledKernel::MaternArd(leaf) => {
                    leaf.grad_cross_from_blocks::<M, T>(b, d_k, param_idx)
                }
                CompiledKernel::RationalQuadraticArd(leaf) => {
                    leaf.grad_cross_from_blocks(b, d_k, param_idx)
                }
                _ => Err(not_ard()),
            },
        }
    }

    /// The rectangular `∂²K/∂θ_i ∂θ_j` from the rectangular supply.
    pub(super) fn hess_cross<M: crate::math::KernelMath>(
        &self,
        slot: RectSlot<'_, T>,
        d2_k: MatMut<'_, T>,
        (i, j): (usize, usize),
    ) -> Result<(), GprError> {
        match slot {
            RectSlot::Scalar(d) => match self.leaf.as_ref() {
                CompiledKernel::Rbf(leaf) => leaf.hess_cross_dist::<M, T>(d, d2_k, i, j),
                CompiledKernel::Matern(leaf) => leaf.hess_cross_dist::<M, T>(d, d2_k, i, j),
                CompiledKernel::Periodic(leaf) => leaf.hess_cross_dist::<M, T>(d, d2_k, i, j),
                CompiledKernel::RationalQuadratic(leaf) => leaf.hess_cross_dist(d, d2_k, i, j),
                CompiledKernel::Custom(leaf) => leaf.hess_cross(d, d2_k, i, j),
                _ => Err(not_scalar()),
            },
            RectSlot::Ard(b) => match self.leaf.as_ref() {
                CompiledKernel::RbfArd(leaf) => leaf.hess_cross_from_blocks::<M, T>(b, d2_k, i, j),
                CompiledKernel::MaternArd(leaf) => {
                    leaf.hess_cross_from_blocks::<M, T>(b, d2_k, i, j)
                }
                CompiledKernel::RationalQuadraticArd(leaf) => {
                    leaf.hess_cross_from_blocks(b, d2_k, i, j)
                }
                _ => Err(not_ard()),
            },
        }
    }

    /// `k(x, x)`: the leaf at `d² = 0`.
    pub(super) fn fill_diag(&self, out: &mut [T]) -> Result<(), GprError> {
        self.leaf.fill_diag(out)
    }

    /// `∂k(x, x)/∂θ_p` broadcast over `out`.
    pub(super) fn grad_diag<M: crate::math::KernelMath>(
        &self,
        out: &mut [T],
        param_idx: usize,
    ) -> Result<(), GprError> {
        let value = if self.is_ard() {
            self.require_param(param_idx)?;
            T::from_f64(0.0)
        } else {
            self.at_zero(|leaf, slot, x, cell, scratch| {
                leaf.grad::<M>(slot, x, cell, param_idx, Triangle::Lower, scratch)
            })?
        };
        out.fill(value);
        Ok(())
    }

    /// `∂²k(x, x)/∂θ_i ∂θ_j` broadcast over `out`.
    pub(super) fn hess_diag<M: crate::math::KernelMath>(
        &self,
        out: &mut [T],
        pair: (usize, usize),
    ) -> Result<(), GprError> {
        let value = if self.is_ard() {
            self.require_param(pair.0)?;
            self.require_param(pair.1)?;
            T::from_f64(0.0)
        } else {
            self.at_zero(|leaf, slot, x, cell, scratch| {
                leaf.hess::<M>(slot, x, cell, pair, Triangle::Lower, scratch)
            })?
        };
        out.fill(value);
        Ok(())
    }

    /// Evaluates `eval` on one pair at zero distance of a scalar slot.
    fn at_zero(
        &self,
        eval: impl FnOnce(
            &Self,
            SquareSlot<'_, T>,
            MatRef<'_, T>,
            MatMut<'_, T>,
            MatMut<'_, T>,
        ) -> Result<(), GprError>,
    ) -> Result<T, GprError> {
        let mut scratch = [T::from_f64(0.0)];
        let scratch = MatMut::from_column_major_slice_mut(&mut scratch, 1, 1);
        let zero = [T::from_f64(0.0)];
        let dist = MatRef::from_column_major_slice(&zero, 1, 1);
        let x = MatRef::from_column_major_slice(&[], 1, 0);
        eval_cell(|cell| eval(self, SquareSlot::Scalar(dist), x, cell, scratch))
    }

    /// Whether the wrapped leaf is an ARD leaf. Each one is `1` at `d² = 0`
    /// whatever its parameters, so every derivative of `k(x, x)` is `0`.
    fn is_ard(&self) -> bool {
        matches!(
            self.leaf.as_ref(),
            CompiledKernel::RbfArd(_)
                | CompiledKernel::MaternArd(_)
                | CompiledKernel::RationalQuadraticArd(_)
        )
    }

    /// Rejects a parameter index past the wrapped leaf's parameters.
    fn require_param(&self, param_idx: usize) -> Result<(), GprError> {
        let count = self.leaf.num_params();
        if param_idx < count {
            Ok(())
        } else {
            Err(GprError::IndexOutOfRange {
                reason: format!("parameter index {param_idx} is out of range for {count}"),
            })
        }
    }
}

fn not_ard() -> GprError {
    GprError::UnsupportedKernelOperation {
        reason: "an ARD distance slot needs an ARD leaf".to_owned(),
    }
}

fn not_scalar() -> GprError {
    GprError::UnsupportedKernelOperation {
        reason: "a scalar distance slot needs an isotropic or custom leaf".to_owned(),
    }
}

/// Square supplies of one call, looked up by slot.
#[cfg(test)]
pub(crate) struct SquareTable<'a, T>(pub(crate) Vec<(SlotId, SquareSlot<'a, T>)>);

#[cfg(test)]
impl<T: Sync + Copy> SquareSlots<T> for SquareTable<'_, T> {
    fn square(&self, slot: SlotId) -> Option<SquareSlot<'_, T>> {
        self.0
            .iter()
            .find(|(id, _)| *id == slot)
            .map(|(_, view)| *view)
    }
}

/// One rectangular supply of a [`RectTable`].
pub(crate) enum RectEntry<'a, T> {
    /// Dense `rows × cols`.
    Scalar(MatRef<'a, T>),
    /// One dense `rows × cols` block per dimension.
    Ard {
        blocks: Vec<&'a [T]>,
        rows: usize,
        cols: usize,
    },
}

/// Rectangular supplies of one call, looked up by slot.
pub(crate) struct RectTable<'a, T>(pub(crate) Vec<(SlotId, RectEntry<'a, T>)>);

impl<T: Sync + Copy> RectSlots<T> for RectTable<'_, T> {
    fn rect(&self, slot: SlotId) -> Option<RectSlot<'_, T>> {
        let (_, entry) = self.0.iter().find(|(id, _)| *id == slot)?;
        Some(match entry {
            RectEntry::Scalar(view) => RectSlot::Scalar(*view),
            RectEntry::Ard { blocks, rows, cols } => RectSlot::Ard(ArdBlocks {
                blocks,
                rows: *rows,
                cols: *cols,
                col0: 0,
            }),
        })
    }
}

/// Columns `start..start + len` of every block of `inner`.
pub(crate) struct ColRange<'a, T> {
    pub(crate) inner: &'a dyn RectSlots<T>,
    pub(crate) start: usize,
    pub(crate) len: usize,
}

impl<T: Sync> RectSlots<T> for ColRange<'_, T> {
    fn rect(&self, slot: SlotId) -> Option<RectSlot<'_, T>> {
        Some(match self.inner.rect(slot)? {
            RectSlot::Scalar(view) => RectSlot::Scalar(view.subcols(self.start, self.len)),
            RectSlot::Ard(blocks) => RectSlot::Ard(ArdBlocks {
                cols: self.len,
                col0: blocks.col0 + self.start,
                ..blocks
            }),
        })
    }
}
