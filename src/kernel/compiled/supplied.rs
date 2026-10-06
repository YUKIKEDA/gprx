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

use crate::error::GprError;
use crate::kernel::dist::{ArdBlocks, ArdSqDiff};
use crate::kernel::leaf_params::LeafParams;
use crate::kernel::supply::{ArdLeafSpec, ScalarLeafSpec, SuppliedLeafSpec};
use crate::kernel::{
    CustomKernel, KernelScalar, MaternArdKernel, MaternKernel, PeriodicKernel,
    RationalQuadraticArdKernel, RationalQuadraticKernel, RbfArdKernel, RbfKernel, SlotId,
    SuppliedSpec, Triangle,
};
use crate::param::Interval;

/// A compiled leaf that reads the supplied `d²` of one slot.
#[derive(Clone, Debug, PartialEq)]
pub struct SuppliedLeaf<T: KernelScalar> {
    pub(crate) slot: SlotId,
    pub(crate) leaf: SuppliedCompiled<T>,
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

/// The square supply of a scalar slot.
pub(super) fn scalar_square<'a, T>(
    slots: Option<&'a dyn SquareSlots<T>>,
    slot: SlotId,
) -> Result<MatRef<'a, T>, GprError> {
    match slots.and_then(|s| s.square(slot)) {
        Some(SquareSlot::Scalar(dist)) => Ok(dist),
        _ => Err(missing()),
    }
}

/// The square supply of an ARD slot.
fn ard_square<'a, T>(
    slots: Option<&'a dyn SquareSlots<T>>,
    slot: SlotId,
) -> Result<ArdSqDiff<'a, T>, GprError> {
    match slots.and_then(|s| s.square(slot)) {
        Some(SquareSlot::Ard(cache)) => Ok(cache),
        _ => Err(missing()),
    }
}

/// The rectangular supply of a scalar slot.
fn scalar_rect<'a, T>(
    slots: Option<&'a dyn RectSlots<T>>,
    slot: SlotId,
) -> Result<MatRef<'a, T>, GprError> {
    match slots.and_then(|s| s.rect(slot)) {
        Some(RectSlot::Scalar(dist)) => Ok(dist),
        _ => Err(missing()),
    }
}

/// The rectangular supply of an ARD slot.
fn ard_rect<'a, T>(
    slots: Option<&'a dyn RectSlots<T>>,
    slot: SlotId,
) -> Result<ArdBlocks<'a, T>, GprError> {
    match slots.and_then(|s| s.rect(slot)) {
        Some(RectSlot::Ard(blocks)) => Ok(blocks),
        _ => Err(missing()),
    }
}

/// The compiled leaf of a [`SuppliedLeaf`], typed by its slot's shape.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum SuppliedCompiled<T: KernelScalar> {
    Scalar(ScalarLeaf<T>),
    Ard(ArdLeaf),
}

/// A compiled leaf on one `d²` per pair.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ScalarLeaf<T: KernelScalar> {
    Rbf(RbfKernel),
    Matern(MaternKernel),
    Periodic(PeriodicKernel),
    RationalQuadratic(RationalQuadraticKernel),
    Custom(CustomKernel<T>),
}

/// A compiled leaf on one `d²` per dimension per pair.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ArdLeaf {
    Rbf(RbfArdKernel),
    Matern(MaternArdKernel),
    RationalQuadratic(RationalQuadraticArdKernel),
}

impl<T: KernelScalar> SuppliedLeaf<T> {
    /// Compiles `spec` for the scalar `T`.
    pub(super) fn compile(spec: &SuppliedSpec) -> Self {
        let leaf = match &spec.leaf {
            SuppliedLeafSpec::Scalar(leaf) => SuppliedCompiled::Scalar(match leaf {
                ScalarLeafSpec::Rbf(k) => ScalarLeaf::Rbf(*k),
                ScalarLeafSpec::Matern(k) => ScalarLeaf::Matern(*k),
                ScalarLeafSpec::Periodic(k) => ScalarLeaf::Periodic(*k),
                ScalarLeafSpec::RationalQuadratic(k) => ScalarLeaf::RationalQuadratic(*k),
                ScalarLeafSpec::Custom(k) => ScalarLeaf::Custom(k.with_scalar()),
            }),
            SuppliedLeafSpec::Ard(leaf) => SuppliedCompiled::Ard(match leaf {
                ArdLeafSpec::Rbf(k) => ArdLeaf::Rbf(k.clone()),
                ArdLeafSpec::Matern(k) => ArdLeaf::Matern(k.clone()),
                ArdLeafSpec::RationalQuadratic(k) => ArdLeaf::RationalQuadratic(k.clone()),
            }),
        };
        Self {
            slot: spec.slot,
            leaf,
        }
    }

    /// Whether `∂K/∂θ` reads an output-shaped scratch (a custom leaf).
    pub(super) fn needs_grad_scratch(&self) -> bool {
        matches!(self.leaf, SuppliedCompiled::Scalar(ScalarLeaf::Custom(_)))
    }

    /// `K` for `uplo` from the square supply of the slot.
    pub(super) fn apply<M: crate::math::KernelMath>(
        &self,
        slots: Option<&dyn SquareSlots<T>>,
        out: MatMut<'_, T>,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        match &self.leaf {
            SuppliedCompiled::Scalar(leaf) => {
                let d = scalar_square(slots, self.slot)?;
                match leaf {
                    ScalarLeaf::Rbf(k) => k.apply_math::<M, _>(d, out, uplo),
                    ScalarLeaf::Matern(k) => k.apply_math::<M, _>(d, out, uplo),
                    ScalarLeaf::Periodic(k) => k.apply_math::<M, _>(d, out, uplo),
                    ScalarLeaf::RationalQuadratic(k) => k.apply(d, out, uplo),
                    ScalarLeaf::Custom(k) => k.apply(d, out, uplo),
                }
            }
            SuppliedCompiled::Ard(leaf) => {
                let c = ard_square(slots, self.slot)?;
                match leaf {
                    ArdLeaf::Rbf(k) => k.apply_from_sq_diff::<M, _>(c, out, uplo),
                    ArdLeaf::Matern(k) => k.apply_from_sq_diff::<M, _>(c, out, uplo),
                    ArdLeaf::RationalQuadratic(k) => k.apply_from_sq_diff(c, out, uplo),
                }
            }
        }
    }

    /// `∂K/∂θ_p` for `uplo` from the square supply of the slot.
    pub(super) fn grad<M: crate::math::KernelMath>(
        &self,
        slots: Option<&dyn SquareSlots<T>>,
        d_k: MatMut<'_, T>,
        p: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        match &self.leaf {
            SuppliedCompiled::Scalar(leaf) => {
                let d = scalar_square(slots, self.slot)?;
                match leaf {
                    ScalarLeaf::Rbf(k) => k.grad_math::<M, _>(d, d_k, p, uplo),
                    ScalarLeaf::Matern(k) => k.grad_math::<M, _>(d, d_k, p, uplo),
                    ScalarLeaf::Periodic(k) => k.grad_math::<M, _>(d, d_k, p, uplo),
                    ScalarLeaf::RationalQuadratic(k) => k.grad(d, d_k, p, uplo),
                    ScalarLeaf::Custom(k) => k.grad(d, d_k, p, uplo),
                }
            }
            SuppliedCompiled::Ard(leaf) => {
                let c = ard_square(slots, self.slot)?;
                match leaf {
                    ArdLeaf::Rbf(k) => k.grad_from_sq_diff::<M, _>(c, d_k, p, uplo),
                    ArdLeaf::Matern(k) => k.grad_from_sq_diff::<M, _>(c, d_k, p, uplo),
                    ArdLeaf::RationalQuadratic(k) => k.grad_from_sq_diff(c, d_k, p, uplo),
                }
            }
        }
    }

    /// `∂²K/∂θ_i ∂θ_j` for `uplo` from the square supply of the slot.
    pub(super) fn hess<M: crate::math::KernelMath>(
        &self,
        slots: Option<&dyn SquareSlots<T>>,
        d2_k: MatMut<'_, T>,
        (i, j): (usize, usize),
        uplo: Triangle,
    ) -> Result<(), GprError> {
        match &self.leaf {
            SuppliedCompiled::Scalar(leaf) => {
                let d = scalar_square(slots, self.slot)?;
                match leaf {
                    ScalarLeaf::Rbf(k) => k.hess_math::<M, _>(d, d2_k, i, j, uplo),
                    ScalarLeaf::Matern(k) => k.hess_math::<M, _>(d, d2_k, i, j, uplo),
                    ScalarLeaf::Periodic(k) => k.hess_math::<M, _>(d, d2_k, i, j, uplo),
                    ScalarLeaf::RationalQuadratic(k) => k.hess(d, d2_k, i, j, uplo),
                    ScalarLeaf::Custom(k) => k.hess(d, d2_k, i, j, uplo),
                }
            }
            SuppliedCompiled::Ard(leaf) => {
                let c = ard_square(slots, self.slot)?;
                match leaf {
                    ArdLeaf::Rbf(k) => k.hess_from_sq_diff::<M, _>(c, d2_k, i, j, uplo),
                    ArdLeaf::Matern(k) => k.hess_from_sq_diff::<M, _>(c, d2_k, i, j, uplo),
                    ArdLeaf::RationalQuadratic(k) => k.hess_from_sq_diff(c, d2_k, i, j, uplo),
                }
            }
        }
    }

    /// The rectangular `K` from the rectangular supply of the slot.
    pub(super) fn apply_cross<M: crate::math::KernelMath>(
        &self,
        slots: Option<&dyn RectSlots<T>>,
        out: MatMut<'_, T>,
    ) -> Result<(), GprError> {
        match &self.leaf {
            SuppliedCompiled::Scalar(leaf) => {
                let d = scalar_rect(slots, self.slot)?;
                match leaf {
                    ScalarLeaf::Rbf(k) => k.apply_cross_math::<M, _>(d, out),
                    ScalarLeaf::Matern(k) => k.apply_cross_math::<M, _>(d, out),
                    ScalarLeaf::Periodic(k) => k.apply_cross_math::<M, _>(d, out),
                    ScalarLeaf::RationalQuadratic(k) => k.apply_cross(d, out),
                    ScalarLeaf::Custom(k) => k.apply_cross(d, out),
                }
            }
            SuppliedCompiled::Ard(leaf) => {
                let b = ard_rect(slots, self.slot)?;
                ard_cross::<M, T>(leaf, b, out)
            }
        }
    }

    /// The rectangular `∂K/∂θ_p` from the rectangular supply of the slot.
    pub(super) fn grad_cross<M: crate::math::KernelMath>(
        &self,
        slots: Option<&dyn RectSlots<T>>,
        d_k: MatMut<'_, T>,
        p: usize,
    ) -> Result<(), GprError> {
        match &self.leaf {
            SuppliedCompiled::Scalar(leaf) => {
                let d = scalar_rect(slots, self.slot)?;
                match leaf {
                    ScalarLeaf::Rbf(k) => k.grad_cross_dist::<M, T>(d, d_k, p),
                    ScalarLeaf::Matern(k) => k.grad_cross_dist::<M, T>(d, d_k, p),
                    ScalarLeaf::Periodic(k) => k.grad_cross_dist::<M, T>(d, d_k, p),
                    ScalarLeaf::RationalQuadratic(k) => k.grad_cross_dist(d, d_k, p),
                    ScalarLeaf::Custom(k) => k.grad_cross(d, d_k, p),
                }
            }
            SuppliedCompiled::Ard(leaf) => {
                let b = ard_rect(slots, self.slot)?;
                ard_grad_cross::<M, T>(leaf, b, d_k, p)
            }
        }
    }

    /// The rectangular `∂²K/∂θ_i ∂θ_j` from the rectangular supply of the slot.
    pub(super) fn hess_cross<M: crate::math::KernelMath>(
        &self,
        slots: Option<&dyn RectSlots<T>>,
        d2_k: MatMut<'_, T>,
        pair: (usize, usize),
    ) -> Result<(), GprError> {
        match &self.leaf {
            SuppliedCompiled::Scalar(leaf) => {
                let d = scalar_rect(slots, self.slot)?;
                scalar_hess_cross::<M, T>(leaf, d, d2_k, pair)
            }
            SuppliedCompiled::Ard(leaf) => {
                let b = ard_rect(slots, self.slot)?;
                ard_hess_cross::<M, T>(leaf, b, d2_k, pair)
            }
        }
    }

    /// `k(x, x)`: the leaf at `d² = 0`.
    pub(super) fn fill_diag(&self, out: &mut [T]) -> Result<(), GprError> {
        match &self.leaf {
            SuppliedCompiled::Scalar(ScalarLeaf::Rbf(k)) => k.fill_diag(out),
            SuppliedCompiled::Scalar(ScalarLeaf::Matern(k)) => k.fill_diag(out),
            SuppliedCompiled::Scalar(ScalarLeaf::Periodic(k)) => k.fill_diag(out),
            SuppliedCompiled::Scalar(ScalarLeaf::RationalQuadratic(k)) => k.fill_diag(out),
            SuppliedCompiled::Scalar(ScalarLeaf::Custom(k)) => return k.fill_diag(out),
            SuppliedCompiled::Ard(ArdLeaf::Rbf(k)) => k.fill_diag(out),
            SuppliedCompiled::Ard(ArdLeaf::Matern(k)) => k.fill_diag(out),
            SuppliedCompiled::Ard(ArdLeaf::RationalQuadratic(k)) => k.fill_diag(out),
        }
        Ok(())
    }

    /// `∂k(x, x)/∂θ_p` broadcast over `out`: the leaf at `d² = 0`.
    pub(super) fn grad_diag<M: crate::math::KernelMath>(
        &self,
        out: &mut [T],
        p: usize,
    ) -> Result<(), GprError> {
        let value = self.at_zero(
            |leaf, d, cell| match leaf {
                ScalarLeaf::Rbf(k) => k.grad_math::<M, _>(d, cell, p, Triangle::Lower),
                ScalarLeaf::Matern(k) => k.grad_math::<M, _>(d, cell, p, Triangle::Lower),
                ScalarLeaf::Periodic(k) => k.grad_math::<M, _>(d, cell, p, Triangle::Lower),
                ScalarLeaf::RationalQuadratic(k) => k.grad(d, cell, p, Triangle::Lower),
                ScalarLeaf::Custom(k) => k.grad(d, cell, p, Triangle::Lower),
            },
            |leaf, b, cell| ard_grad_cross::<M, T>(leaf, b, cell, p),
        )?;
        out.fill(value);
        Ok(())
    }

    /// `∂²k(x, x)/∂θ_i ∂θ_j` broadcast over `out`: the leaf at `d² = 0`.
    pub(super) fn hess_diag<M: crate::math::KernelMath>(
        &self,
        out: &mut [T],
        (i, j): (usize, usize),
    ) -> Result<(), GprError> {
        let value = self.at_zero(
            |leaf, d, cell| match leaf {
                ScalarLeaf::Rbf(k) => k.hess_math::<M, _>(d, cell, i, j, Triangle::Lower),
                ScalarLeaf::Matern(k) => k.hess_math::<M, _>(d, cell, i, j, Triangle::Lower),
                ScalarLeaf::Periodic(k) => k.hess_math::<M, _>(d, cell, i, j, Triangle::Lower),
                ScalarLeaf::RationalQuadratic(k) => k.hess(d, cell, i, j, Triangle::Lower),
                ScalarLeaf::Custom(k) => k.hess(d, cell, i, j, Triangle::Lower),
            },
            |leaf, b, cell| ard_hess_cross::<M, T>(leaf, b, cell, (i, j)),
        )?;
        out.fill(value);
        Ok(())
    }

    /// Evaluates the leaf on one pair at `d² = 0`: a scalar leaf on a
    /// `1 × 1` square, an ARD leaf on one `1 × 1` block per dimension (only
    /// the list of blocks is allocated).
    fn at_zero(
        &self,
        scalar: impl FnOnce(&ScalarLeaf<T>, MatRef<'_, T>, MatMut<'_, T>) -> Result<(), GprError>,
        ard: impl FnOnce(&ArdLeaf, ArdBlocks<'_, T>, MatMut<'_, T>) -> Result<(), GprError>,
    ) -> Result<T, GprError> {
        let zero = [T::from_f64(0.0)];
        let mut cell = [T::from_f64(0.0)];
        let out = MatMut::from_column_major_slice_mut(&mut cell, 1, 1);
        match &self.leaf {
            SuppliedCompiled::Scalar(leaf) => {
                scalar(leaf, MatRef::from_column_major_slice(&zero, 1, 1), out)?;
            }
            SuppliedCompiled::Ard(leaf) => {
                let blocks = vec![&zero[..]; leaf.dims()];
                let b = ArdBlocks {
                    blocks: &blocks,
                    rows: 1,
                    cols: 1,
                    col0: 0,
                };
                ard(leaf, b, out)?;
            }
        }
        Ok(cell[0])
    }
}

impl ArdLeaf {
    /// Number of lengthscales, one per dimension of the slot.
    fn dims(&self) -> usize {
        match self {
            Self::Rbf(k) => k.lengthscales().num_params(),
            Self::Matern(k) => k.lengthscales().num_params(),
            Self::RationalQuadratic(k) => k.lengthscales().num_params(),
        }
    }
}

fn ard_cross<M: crate::math::KernelMath, T: KernelScalar>(
    leaf: &ArdLeaf,
    b: ArdBlocks<'_, T>,
    out: MatMut<'_, T>,
) -> Result<(), GprError> {
    match leaf {
        ArdLeaf::Rbf(k) => k.apply_cross_from_blocks::<M, T>(b, out),
        ArdLeaf::Matern(k) => k.apply_cross_from_blocks::<M, T>(b, out),
        ArdLeaf::RationalQuadratic(k) => k.apply_cross_from_blocks(b, out),
    }
}

fn ard_grad_cross<M: crate::math::KernelMath, T: KernelScalar>(
    leaf: &ArdLeaf,
    b: ArdBlocks<'_, T>,
    d_k: MatMut<'_, T>,
    p: usize,
) -> Result<(), GprError> {
    match leaf {
        ArdLeaf::Rbf(k) => k.grad_cross_from_blocks::<M, T>(b, d_k, p),
        ArdLeaf::Matern(k) => k.grad_cross_from_blocks::<M, T>(b, d_k, p),
        ArdLeaf::RationalQuadratic(k) => k.grad_cross_from_blocks(b, d_k, p),
    }
}

fn ard_hess_cross<M: crate::math::KernelMath, T: KernelScalar>(
    leaf: &ArdLeaf,
    b: ArdBlocks<'_, T>,
    d2_k: MatMut<'_, T>,
    (i, j): (usize, usize),
) -> Result<(), GprError> {
    match leaf {
        ArdLeaf::Rbf(k) => k.hess_cross_from_blocks::<M, T>(b, d2_k, i, j),
        ArdLeaf::Matern(k) => k.hess_cross_from_blocks::<M, T>(b, d2_k, i, j),
        ArdLeaf::RationalQuadratic(k) => k.hess_cross_from_blocks(b, d2_k, i, j),
    }
}

fn scalar_hess_cross<M: crate::math::KernelMath, T: KernelScalar>(
    leaf: &ScalarLeaf<T>,
    d: MatRef<'_, T>,
    d2_k: MatMut<'_, T>,
    (i, j): (usize, usize),
) -> Result<(), GprError> {
    match leaf {
        ScalarLeaf::Rbf(k) => k.hess_cross_dist::<M, T>(d, d2_k, i, j),
        ScalarLeaf::Matern(k) => k.hess_cross_dist::<M, T>(d, d2_k, i, j),
        ScalarLeaf::Periodic(k) => k.hess_cross_dist::<M, T>(d, d2_k, i, j),
        ScalarLeaf::RationalQuadratic(k) => k.hess_cross_dist(d, d2_k, i, j),
        ScalarLeaf::Custom(k) => k.hess_cross(d, d2_k, i, j),
    }
}

/// Every variant of a compiled supplied leaf, with the leaf bound to `$leaf`.
macro_rules! each_compiled {
    ($value:expr, $leaf:ident => $body:expr) => {
        match $value {
            SuppliedCompiled::Scalar(ScalarLeaf::Rbf($leaf)) => $body,
            SuppliedCompiled::Scalar(ScalarLeaf::Matern($leaf)) => $body,
            SuppliedCompiled::Scalar(ScalarLeaf::Periodic($leaf)) => $body,
            SuppliedCompiled::Scalar(ScalarLeaf::RationalQuadratic($leaf)) => $body,
            SuppliedCompiled::Scalar(ScalarLeaf::Custom($leaf)) => $body,
            SuppliedCompiled::Ard(ArdLeaf::Rbf($leaf)) => $body,
            SuppliedCompiled::Ard(ArdLeaf::Matern($leaf)) => $body,
            SuppliedCompiled::Ard(ArdLeaf::RationalQuadratic($leaf)) => $body,
        }
    };
}

impl<T: KernelScalar> LeafParams for SuppliedCompiled<T> {
    fn leaf_num_params(&self) -> usize {
        each_compiled!(self, leaf => leaf.leaf_num_params())
    }

    fn write_leaf_params(&self, out: &mut [f64], offset: &mut usize) -> Result<(), GprError> {
        each_compiled!(self, leaf => leaf.write_leaf_params(out, offset))
    }

    fn write_leaf_intervals(
        &self,
        out: &mut [Interval],
        offset: &mut usize,
    ) -> Result<(), GprError> {
        each_compiled!(self, leaf => leaf.write_leaf_intervals(out, offset))
    }

    fn apply_leaf_params(&mut self, params: &[f64], offset: &mut usize) -> Result<(), GprError> {
        each_compiled!(self, leaf => leaf.apply_leaf_params(params, offset))
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
