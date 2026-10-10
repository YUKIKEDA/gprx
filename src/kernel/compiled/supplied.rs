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
use crate::kernel::dist::{ArdBlocks, ArdSqDiff, BlockList, Checked, Unchecked};
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
    /// The slot's number among the slots of its shape in the tree
    /// ([`SuppliedSpec::at`]): where every supply of the tree keeps it.
    pub(crate) at: usize,
    pub(crate) leaf: SuppliedCompiled<T>,
}

/// The square supply of an ARD slot: packed lower triangles (the training
/// store), or dense checked blocks (a query square read where it was bound).
#[derive(Clone, Copy, Debug)]
pub enum ArdSquare<'a, T> {
    /// The training store: per dimension, the lower triangle as column
    /// runs (packed, or the dense tables an `f64` model kept).
    Packed(ArdSqDiff<'a, T>),
    /// Dense checked `n × n` blocks, one per dimension.
    Dense(ArdBlocks<'a, T, Checked>),
}

/// The blocks of an ARD slot, checked when bound or to be checked as they
/// are read ([`crate::kernel::dist::BlockState`]).
#[derive(Clone, Copy, Debug)]
pub enum ArdRect<'a, T> {
    /// Checked when bound: the training triangles, a cast, a repaired or
    /// filled table.
    Checked(ArdBlocks<'a, T, Checked>),
    /// A caller's block read in place, checked as it is read.
    Unchecked(ArdBlocks<'a, T, Unchecked>),
}

impl<T: KernelScalar> ArdRect<'_, T> {
    /// Columns `start..start + len`.
    pub(crate) fn subcols(self, start: usize, len: usize) -> Self {
        match self {
            Self::Checked(b) => Self::Checked(b.subcols(start, len)),
            Self::Unchecked(b) => Self::Unchecked(b.subcols(start, len)),
        }
    }
}

/// The `d²` of every slot of a tree between the points of one set, by the
/// slot's number in its shape ([`SuppliedLeaf::at`]). A supply bound for
/// the tree holds every slot it numbers; one bound for another kernel is
/// reported on the lookup ([`unbound`]), never read out of range.
pub trait SquareSlots<T>: Sync {
    /// The dense symmetric `n × n` square of scalar slot `at`.
    ///
    /// # Errors
    ///
    /// [`unbound`] when the supply holds no scalar slot `at`.
    fn scalar(&self, at: usize) -> Result<MatRef<'_, T>, GprError>;

    /// The square of ARD slot `at`.
    ///
    /// # Errors
    ///
    /// [`unbound`] when the supply holds no ARD slot `at`.
    fn ard(&self, at: usize) -> Result<ArdSquare<'_, T>, GprError>;
}

/// The `d²` of every slot of a tree between the points of two sets, by the
/// slot's number in its shape ([`SuppliedLeaf::at`]). A supply bound for
/// the tree holds every slot it numbers; one bound for another kernel is
/// reported on the lookup ([`unbound`]), never read out of range.
pub trait RectSlots<T>: Sync {
    /// The dense `rows × cols` block of scalar slot `at`.
    ///
    /// # Errors
    ///
    /// [`unbound`] when the supply holds no scalar slot `at`.
    fn scalar(&self, at: usize) -> Result<MatRef<'_, T>, GprError>;

    /// One dense `rows × cols` block per dimension of ARD slot `at`.
    ///
    /// # Errors
    ///
    /// [`unbound`] when the supply holds no ARD slot `at`.
    fn ard(&self, at: usize) -> Result<ArdRect<'_, T>, GprError>;
}

/// A supply was paired with a tree whose leaf numbers it does not hold
/// (bound for another kernel). Reported, never read out of range.
pub(crate) fn unbound() -> GprError {
    GprError::UnsupportedKernelOperation {
        reason: "a distance leaf has no supplied squared distances here (the supply was bound \
                 for another kernel)"
            .to_owned(),
    }
}

/// No supplied squared distances: what the view of a model reads whose
/// kernel reads none. Every read is [`unbound`].
pub(crate) struct NoSlots;

/// The one [`NoSlots`].
pub(crate) static NO_SLOTS: NoSlots = NoSlots;

impl<T> SquareSlots<T> for NoSlots {
    fn scalar(&self, _at: usize) -> Result<MatRef<'_, T>, GprError> {
        Err(unbound())
    }

    fn ard(&self, _at: usize) -> Result<ArdSquare<'_, T>, GprError> {
        Err(unbound())
    }
}

impl<T> RectSlots<T> for NoSlots {
    fn scalar(&self, _at: usize) -> Result<MatRef<'_, T>, GprError> {
        Err(unbound())
    }

    fn ard(&self, _at: usize) -> Result<ArdRect<'_, T>, GprError> {
        Err(unbound())
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

/// `$body` with `$k` the kernel of the scalar leaf `$leaf`, or `$custom`
/// with `$c` a custom leaf's term. The built-in leaves share their method
/// names, so one body serves them all.
macro_rules! each_scalar_leaf {
    ($leaf:expr, |$k:ident| $body:expr; custom |$c:ident| $custom:expr) => {
        match $leaf {
            ScalarLeaf::Rbf($k) => $body,
            ScalarLeaf::Matern($k) => $body,
            ScalarLeaf::Periodic($k) => $body,
            ScalarLeaf::RationalQuadratic($k) => $body,
            ScalarLeaf::Custom($c) => $custom,
        }
    };
}

/// `$body` with `$k` the kernel of the ARD leaf `$leaf`.
macro_rules! each_ard_leaf {
    ($leaf:expr, |$k:ident| $body:expr) => {
        match $leaf {
            ArdLeaf::Rbf($k) => $body,
            ArdLeaf::Matern($k) => $body,
            ArdLeaf::RationalQuadratic($k) => $body,
        }
    };
}

impl<T: KernelScalar> SuppliedLeaf<T> {
    /// Compiles `spec` for the scalar `T`.
    pub(crate) fn compile(spec: &SuppliedSpec) -> Self {
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
            at: spec.at,
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
        slots: &dyn SquareSlots<T>,
        out: MatMut<'_, T>,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        match &self.leaf {
            SuppliedCompiled::Scalar(leaf) => {
                let d = slots.scalar(self.at)?;
                each_scalar_leaf!(
                    leaf,
                    |k| k.apply_math::<M, _>(d, out, uplo);
                    custom |k| k.apply(d, out, uplo)
                )
            }
            SuppliedCompiled::Ard(leaf) => match slots.ard(self.at)? {
                ArdSquare::Packed(c) => {
                    each_ard_leaf!(leaf, |k| k.apply_from_sq_diff::<M, _>(c, out, uplo))
                }
                // Every entry of a dense square, whatever `uplo` asks.
                ArdSquare::Dense(b) => ard_cross::<M, T>(leaf, ArdRect::Checked(b), out),
            },
        }
    }

    /// `∂K/∂θ_p` for `uplo` from the square supply of the slot.
    pub(super) fn grad<M: crate::math::KernelMath>(
        &self,
        slots: &dyn SquareSlots<T>,
        d_k: MatMut<'_, T>,
        p: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        match &self.leaf {
            SuppliedCompiled::Scalar(leaf) => {
                let d = slots.scalar(self.at)?;
                each_scalar_leaf!(
                    leaf,
                    |k| k.grad_math::<M, _>(d, d_k, p, uplo);
                    custom |k| k.grad(d, d_k, p, uplo)
                )
            }
            SuppliedCompiled::Ard(leaf) => match slots.ard(self.at)? {
                ArdSquare::Packed(c) => {
                    each_ard_leaf!(leaf, |k| k.grad_from_sq_diff::<M, _>(c, d_k, p, uplo))
                }
                ArdSquare::Dense(b) => ard_grad_cross::<M, T>(leaf, ArdRect::Checked(b), d_k, p),
            },
        }
    }

    /// `∂²K/∂θ_i ∂θ_j` for `uplo` from the square supply of the slot.
    pub(super) fn hess<M: crate::math::KernelMath>(
        &self,
        slots: &dyn SquareSlots<T>,
        d2_k: MatMut<'_, T>,
        (i, j): (usize, usize),
        uplo: Triangle,
    ) -> Result<(), GprError> {
        match &self.leaf {
            SuppliedCompiled::Scalar(leaf) => {
                let d = slots.scalar(self.at)?;
                each_scalar_leaf!(
                    leaf,
                    |k| k.hess_math::<M, _>(d, d2_k, i, j, uplo);
                    custom |k| k.hess(d, d2_k, i, j, uplo)
                )
            }
            SuppliedCompiled::Ard(leaf) => match slots.ard(self.at)? {
                ArdSquare::Packed(c) => {
                    each_ard_leaf!(leaf, |k| k.hess_from_sq_diff::<M, _>(c, d2_k, i, j, uplo))
                }
                ArdSquare::Dense(b) => {
                    ard_hess_cross::<M, T>(leaf, ArdRect::Checked(b), d2_k, (i, j))
                }
            },
        }
    }

    /// The rectangular `K` from the rectangular supply of the slot.
    pub(super) fn apply_cross<M: crate::math::KernelMath>(
        &self,
        slots: &dyn RectSlots<T>,
        out: MatMut<'_, T>,
    ) -> Result<(), GprError> {
        match &self.leaf {
            SuppliedCompiled::Scalar(leaf) => {
                let d = slots.scalar(self.at)?;
                each_scalar_leaf!(
                    leaf,
                    |k| k.apply_cross_math::<M, _>(d, out);
                    custom |k| k.apply_cross(d, out)
                )
            }
            SuppliedCompiled::Ard(leaf) => {
                let b = slots.ard(self.at)?;
                ard_cross::<M, T>(leaf, b, out)
            }
        }
    }

    /// The rectangular `∂K/∂θ_p` from the rectangular supply of the slot.
    pub(super) fn grad_cross<M: crate::math::KernelMath>(
        &self,
        slots: &dyn RectSlots<T>,
        d_k: MatMut<'_, T>,
        p: usize,
    ) -> Result<(), GprError> {
        match &self.leaf {
            SuppliedCompiled::Scalar(leaf) => {
                let d = slots.scalar(self.at)?;
                each_scalar_leaf!(
                    leaf,
                    |k| k.grad_cross_dist::<M, T>(d, d_k, p);
                    custom |k| k.grad_cross(d, d_k, p)
                )
            }
            SuppliedCompiled::Ard(leaf) => {
                let b = slots.ard(self.at)?;
                ard_grad_cross::<M, T>(leaf, b, d_k, p)
            }
        }
    }

    /// The rectangular `∂²K/∂θ_i ∂θ_j` from the rectangular supply of the slot.
    pub(super) fn hess_cross<M: crate::math::KernelMath>(
        &self,
        slots: &dyn RectSlots<T>,
        d2_k: MatMut<'_, T>,
        pair: (usize, usize),
    ) -> Result<(), GprError> {
        match &self.leaf {
            SuppliedCompiled::Scalar(leaf) => {
                let d = slots.scalar(self.at)?;
                scalar_hess_cross::<M, T>(leaf, d, d2_k, pair)
            }
            SuppliedCompiled::Ard(leaf) => {
                let b = slots.ard(self.at)?;
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
            |leaf, d, cell| {
                each_scalar_leaf!(
                    leaf,
                    |k| k.grad_math::<M, _>(d, cell, p, Triangle::Lower);
                    custom |k| k.grad(d, cell, p, Triangle::Lower)
                )
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
            |leaf, d, cell| {
                each_scalar_leaf!(
                    leaf,
                    |k| k.hess_math::<M, _>(d, cell, i, j, Triangle::Lower);
                    custom |k| k.hess(d, cell, i, j, Triangle::Lower)
                )
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
        ard: impl FnOnce(&ArdLeaf, ArdRect<'_, T>, MatMut<'_, T>) -> Result<(), GprError>,
    ) -> Result<T, GprError> {
        let zero = [T::from_f64(0.0)];
        let mut cell = [T::from_f64(0.0)];
        let out = MatMut::from_column_major_slice_mut(&mut cell, 1, 1);
        match &self.leaf {
            SuppliedCompiled::Scalar(leaf) => {
                scalar(leaf, MatRef::from_column_major_slice(&zero, 1, 1), out)?;
            }
            SuppliedCompiled::Ard(leaf) => {
                let b = ArdRect::Checked(ArdBlocks::new(
                    BlockList::Repeat(&zero, leaf.dims()),
                    1,
                    1,
                    0,
                ));
                ard(leaf, b, out)?;
            }
        }
        Ok(cell[0])
    }
}

impl ArdLeaf {
    /// Number of lengthscales, one per dimension of the slot.
    fn dims(&self) -> usize {
        each_ard_leaf!(self, |k| k.lengthscales().num_params())
    }
}

/// Calls `$f` on the blocks of `$rect` in their state.
macro_rules! on_state {
    ($rect:expr, |$b:ident| $f:expr) => {
        match $rect {
            ArdRect::Checked($b) => $f,
            ArdRect::Unchecked($b) => $f,
        }
    };
}

fn ard_cross<M: crate::math::KernelMath, T: KernelScalar>(
    leaf: &ArdLeaf,
    b: ArdRect<'_, T>,
    out: MatMut<'_, T>,
) -> Result<(), GprError> {
    on_state!(b, |b| each_ard_leaf!(leaf, |k| k
        .apply_cross_from_blocks::<M, T, _>(b, out)))
}

fn ard_grad_cross<M: crate::math::KernelMath, T: KernelScalar>(
    leaf: &ArdLeaf,
    b: ArdRect<'_, T>,
    d_k: MatMut<'_, T>,
    p: usize,
) -> Result<(), GprError> {
    on_state!(b, |b| each_ard_leaf!(leaf, |k| k
        .grad_cross_from_blocks::<M, T, _>(b, d_k, p)))
}

fn ard_hess_cross<M: crate::math::KernelMath, T: KernelScalar>(
    leaf: &ArdLeaf,
    b: ArdRect<'_, T>,
    d2_k: MatMut<'_, T>,
    (i, j): (usize, usize),
) -> Result<(), GprError> {
    on_state!(b, |b| each_ard_leaf!(leaf, |k| k
        .hess_cross_from_blocks::<M, T, _>(b, d2_k, i, j)))
}

fn scalar_hess_cross<M: crate::math::KernelMath, T: KernelScalar>(
    leaf: &ScalarLeaf<T>,
    d: MatRef<'_, T>,
    d2_k: MatMut<'_, T>,
    (i, j): (usize, usize),
) -> Result<(), GprError> {
    each_scalar_leaf!(
        leaf,
        |k| k.hess_cross_dist::<M, T>(d, d2_k, i, j);
        custom |k| k.hess_cross(d, d2_k, i, j)
    )
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

/// Square supplies of one call, by shape in slot order.
#[cfg(test)]
pub(crate) struct SquareTable<'a, T> {
    pub(crate) scalar: Vec<MatRef<'a, T>>,
    pub(crate) ard: Vec<ArdSquare<'a, T>>,
}

#[cfg(test)]
impl<T: Sync + Copy> SquareSlots<T> for SquareTable<'_, T> {
    fn scalar(&self, at: usize) -> Result<MatRef<'_, T>, GprError> {
        self.scalar.get(at).copied().ok_or_else(unbound)
    }

    fn ard(&self, at: usize) -> Result<ArdSquare<'_, T>, GprError> {
        self.ard.get(at).copied().ok_or_else(unbound)
    }
}

/// Columns `start..start + len` of every block of `inner`.
pub(crate) struct ColRange<'a, T> {
    pub(crate) inner: &'a dyn RectSlots<T>,
    pub(crate) start: usize,
    pub(crate) len: usize,
}

impl<T: KernelScalar> RectSlots<T> for ColRange<'_, T> {
    fn scalar(&self, at: usize) -> Result<MatRef<'_, T>, GprError> {
        Ok(self.inner.scalar(at)?.subcols(self.start, self.len))
    }

    fn ard(&self, at: usize) -> Result<ArdRect<'_, T>, GprError> {
        Ok(self.inner.ard(at)?.subcols(self.start, self.len))
    }
}
