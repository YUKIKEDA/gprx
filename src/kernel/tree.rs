//! What a kernel tree holds besides coordinate leaves: nothing
//! ([`NoSupply`], a coordinate tree) or leaves that read supplied squared
//! distances (a [`super::DistanceKernel`]).

use std::fmt;

use super::compiled::supplied::{ColRange, RectSlots, SquareSlots, SuppliedLeaf};
use super::{CompiledKernel, KernelScalar, SuppliedSpec};

/// Represents the leaves a kernel tree holds besides coordinate leaves.
/// Sealed.
///
/// A [`super::KernelSpec`] and a [`CompiledKernel`] take it as their last
/// type parameter. The default, [`NoSupply`], is a coordinate tree: it has
/// no value, so a coordinate tree cannot hold a leaf that reads supplied
/// distances. A [`super::DistanceKernel`] holds the other kind, which only
/// this crate names.
///
/// # Examples
///
/// ```rust
/// use gprx::kernel::{CompiledKernel, KernelSpec, NoSupply, RbfKernel};
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let spec: KernelSpec<NoSupply> = KernelSpec::from(RbfKernel::new(1.0)?);
/// let compiled: CompiledKernel<f64, NoSupply> = spec.compile();
/// assert_eq!(compiled.num_params(), 1);
/// # Ok(())
/// # }
/// ```
pub trait Supply: sealed::Supply + Clone + fmt::Debug + PartialEq + Send + Sync + 'static {}

/// Represents a coordinate kernel tree: one with no leaf that reads supplied
/// distances. It has no value.
///
/// See the example on [`Supply`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NoSupply {}

impl Supply for NoSupply {}
impl Supply for SuppliedSpec {}

pub(crate) mod sealed {
    use super::{
        ColRange, CompiledKernel, KernelScalar, NoSupply, RectSlots, SquareSlots, SuppliedLeaf,
        SuppliedSpec,
    };
    use std::fmt;

    pub trait Supply: Sized {
        /// The compiled leaf for compute scalar `T`.
        type Compiled<T: KernelScalar>: Clone + fmt::Debug + PartialEq + Send + Sync;

        /// The leaf, as the crate reads it.
        fn spec(leaf: &Self) -> &SuppliedSpec;

        /// The leaf, as the crate writes it.
        fn spec_mut(leaf: &mut Self) -> &mut SuppliedSpec;

        /// The compiled leaf, as the crate reads it.
        fn compiled<T: KernelScalar>(leaf: &Self::Compiled<T>) -> &SuppliedLeaf<T>;

        /// The compiled leaf, as the crate writes it.
        fn compiled_mut<T: KernelScalar>(leaf: &mut Self::Compiled<T>) -> &mut SuppliedLeaf<T>;

        /// Compiles the leaf for `T`, numbering its slot in `order`.
        fn compile<T: KernelScalar>(leaf: &Self) -> Self::Compiled<T>;

        /// The squared distances of one set a tree of this kind reads:
        /// nothing for a coordinate tree, every slot's otherwise.
        type Squares<'a, T: KernelScalar>: Copy;

        /// The squared distances between two sets a tree of this kind
        /// reads: nothing for a coordinate tree, every slot's otherwise.
        type Rects<'a, T: KernelScalar>: Copy;

        /// The supplies of `slots` as this kind reads them.
        fn squares<'a, T: KernelScalar>(slots: &'a dyn SquareSlots<T>) -> Self::Squares<'a, T>;

        /// The supplies of `slots` as this kind reads them.
        fn rects<'a, T: KernelScalar>(slots: &'a dyn RectSlots<T>) -> Self::Rects<'a, T>;

        /// `slots` borrowed for the shorter `'b`: a projection is invariant
        /// in its lifetime, so the views that hold one shorten through here.
        fn shorter_squares<'a: 'b, 'b, T: KernelScalar>(
            slots: Self::Squares<'a, T>,
        ) -> Self::Squares<'b, T>;

        /// [`Self::shorter_squares`] of the rectangular supplies.
        fn shorter_rects<'a: 'b, 'b, T: KernelScalar>(
            slots: Self::Rects<'a, T>,
        ) -> Self::Rects<'b, T>;

        /// A compiled leaf and the square supplies it reads.
        fn square_leaf<'l, 'a, T: KernelScalar>(
            leaf: &'l Self::Compiled<T>,
            slots: Self::Squares<'a, T>,
        ) -> (&'l SuppliedLeaf<T>, &'a dyn SquareSlots<T>);

        /// A compiled leaf and the rectangular supplies it reads.
        fn rect_leaf<'l, 'a, T: KernelScalar>(
            leaf: &'l Self::Compiled<T>,
            slots: Self::Rects<'a, T>,
        ) -> (&'l SuppliedLeaf<T>, &'a dyn RectSlots<T>);

        /// Calls `f` on columns `start..start + len` of `slots`.
        fn with_cols<T: KernelScalar, R>(
            slots: Self::Rects<'_, T>,
            start: usize,
            len: usize,
            f: impl FnOnce(Self::Rects<'_, T>) -> R,
        ) -> R;

        /// The tree as a coordinate tree: itself for [`NoSupply`], `None` for
        /// a tree of the other kind, which reads its leaves one by one.
        fn coordinates<T: KernelScalar>(
            tree: &CompiledKernel<T, Self>,
        ) -> Option<&CompiledKernel<T, NoSupply>>
        where
            Self: super::Supply;

        /// A decoded supplied leaf in this kind of tree; `None` when this
        /// kind holds none.
        fn from_spec(leaf: SuppliedSpec) -> Option<Self>;
    }

    impl Supply for NoSupply {
        type Compiled<T: KernelScalar> = NoSupply;

        fn spec(leaf: &Self) -> &SuppliedSpec {
            match *leaf {}
        }

        fn spec_mut(leaf: &mut Self) -> &mut SuppliedSpec {
            match *leaf {}
        }

        fn compiled<T: KernelScalar>(leaf: &NoSupply) -> &SuppliedLeaf<T> {
            match *leaf {}
        }

        fn compiled_mut<T: KernelScalar>(leaf: &mut NoSupply) -> &mut SuppliedLeaf<T> {
            match *leaf {}
        }

        fn compile<T: KernelScalar>(leaf: &Self) -> NoSupply {
            match *leaf {}
        }

        type Squares<'a, T: KernelScalar> = ();

        type Rects<'a, T: KernelScalar> = ();

        fn squares<'a, T: KernelScalar>(_slots: &'a dyn SquareSlots<T>) -> Self::Squares<'a, T> {}

        fn rects<'a, T: KernelScalar>(_slots: &'a dyn RectSlots<T>) -> Self::Rects<'a, T> {}

        fn shorter_squares<'a: 'b, 'b, T: KernelScalar>(
            (): Self::Squares<'a, T>,
        ) -> Self::Squares<'b, T> {
        }

        fn shorter_rects<'a: 'b, 'b, T: KernelScalar>(
            (): Self::Rects<'a, T>,
        ) -> Self::Rects<'b, T> {
        }

        fn square_leaf<'l, 'a, T: KernelScalar>(
            leaf: &'l Self::Compiled<T>,
            _slots: Self::Squares<'a, T>,
        ) -> (&'l SuppliedLeaf<T>, &'a dyn SquareSlots<T>) {
            match *leaf {}
        }

        fn rect_leaf<'l, 'a, T: KernelScalar>(
            leaf: &'l Self::Compiled<T>,
            _slots: Self::Rects<'a, T>,
        ) -> (&'l SuppliedLeaf<T>, &'a dyn RectSlots<T>) {
            match *leaf {}
        }

        fn with_cols<T: KernelScalar, R>(
            (): (),
            _start: usize,
            _len: usize,
            f: impl FnOnce(()) -> R,
        ) -> R {
            f(())
        }

        fn coordinates<T: KernelScalar>(
            tree: &CompiledKernel<T, Self>,
        ) -> Option<&CompiledKernel<T, NoSupply>> {
            Some(tree)
        }

        fn from_spec(_leaf: SuppliedSpec) -> Option<Self> {
            None
        }
    }

    impl Supply for SuppliedSpec {
        type Compiled<T: KernelScalar> = SuppliedLeaf<T>;

        fn spec(leaf: &Self) -> &SuppliedSpec {
            leaf
        }

        fn spec_mut(leaf: &mut Self) -> &mut SuppliedSpec {
            leaf
        }

        fn compiled<T: KernelScalar>(leaf: &SuppliedLeaf<T>) -> &SuppliedLeaf<T> {
            leaf
        }

        fn compiled_mut<T: KernelScalar>(leaf: &mut SuppliedLeaf<T>) -> &mut SuppliedLeaf<T> {
            leaf
        }

        fn compile<T: KernelScalar>(leaf: &Self) -> SuppliedLeaf<T> {
            SuppliedLeaf::compile(leaf)
        }

        type Squares<'a, T: KernelScalar> = &'a dyn SquareSlots<T>;

        type Rects<'a, T: KernelScalar> = &'a dyn RectSlots<T>;

        fn squares<'a, T: KernelScalar>(slots: &'a dyn SquareSlots<T>) -> Self::Squares<'a, T> {
            slots
        }

        fn rects<'a, T: KernelScalar>(slots: &'a dyn RectSlots<T>) -> Self::Rects<'a, T> {
            slots
        }

        fn shorter_squares<'a: 'b, 'b, T: KernelScalar>(
            slots: Self::Squares<'a, T>,
        ) -> Self::Squares<'b, T> {
            slots
        }

        fn shorter_rects<'a: 'b, 'b, T: KernelScalar>(
            slots: Self::Rects<'a, T>,
        ) -> Self::Rects<'b, T> {
            slots
        }

        fn square_leaf<'l, 'a, T: KernelScalar>(
            leaf: &'l Self::Compiled<T>,
            slots: Self::Squares<'a, T>,
        ) -> (&'l SuppliedLeaf<T>, &'a dyn SquareSlots<T>) {
            (leaf, slots)
        }

        fn rect_leaf<'l, 'a, T: KernelScalar>(
            leaf: &'l Self::Compiled<T>,
            slots: Self::Rects<'a, T>,
        ) -> (&'l SuppliedLeaf<T>, &'a dyn RectSlots<T>) {
            (leaf, slots)
        }

        fn with_cols<T: KernelScalar, R>(
            slots: &dyn RectSlots<T>,
            start: usize,
            len: usize,
            f: impl FnOnce(&dyn RectSlots<T>) -> R,
        ) -> R {
            f(&ColRange {
                inner: slots,
                start,
                len,
            })
        }

        fn coordinates<T: KernelScalar>(
            _tree: &CompiledKernel<T, Self>,
        ) -> Option<&CompiledKernel<T, NoSupply>> {
            None
        }

        fn from_spec(leaf: SuppliedSpec) -> Option<Self> {
            Some(leaf)
        }
    }
}
