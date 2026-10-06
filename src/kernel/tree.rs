//! What a kernel tree holds besides coordinate leaves: nothing
//! ([`NoSupply`], a coordinate tree) or leaves that read supplied squared
//! distances (a [`super::DistanceKernel`]).

use std::fmt;

use super::compiled::supplied::SuppliedLeaf;
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

mod sealed {
    use super::{CompiledKernel, KernelScalar, NoSupply, SuppliedLeaf, SuppliedSpec};
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

        /// Compiles the leaf for `T`.
        fn compile<T: KernelScalar>(leaf: &Self) -> Self::Compiled<T>;

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
