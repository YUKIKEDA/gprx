//! Kernels on squared distances the caller supplies.
//!
//! A slot ([`ScalarDistance`] or [`ArdDistance`]) names one supply of
//! squared distances `d²` between samples the caller owns. Leaves made from
//! a slot ([`ScalarDistance::kernel`], [`ArdDistance::kernel`]) read that
//! supply; every leaf of one slot reads the same supply. A
//! [`DistanceKernel`] is the expression those leaves form, with
//! [`ConstantKernel`] / [`WhiteKernel`] and, as [`DistanceKernel<WithPoints>`],
//! coordinate leaves of a [`KernelSpec`]. A [`DistanceSource`] binds a supply
//! to its slot for one call.

use std::borrow::Cow;
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Add, Mul};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::GprError;
use crate::kernel::leaf_params::LeafParams;
use crate::kernel::{
    ConstantKernel, CustomKernel, KernelSpec, KernelTerm, MaternArdKernel, MaternKernel,
    ParameterBinding, PeriodicKernel, RationalQuadraticArdKernel, RationalQuadraticKernel,
    RbfArdKernel, RbfKernel, WhiteKernel,
};
use crate::kernel::{NoSupply, Supply};
use crate::param::Interval;

static NEXT_SLOT: AtomicU64 = AtomicU64::new(1);

/// Identity of one distance slot. Unique within the process.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SlotId(u64);

impl SlotId {
    pub(crate) fn fresh() -> Self {
        Self(NEXT_SLOT.fetch_add(1, Ordering::Relaxed))
    }
}

/// What one slot supplies: one `d²`, or one `d²` per dimension.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SlotShape {
    Scalar,
    Ard(usize),
}

impl SlotShape {
    /// Number of `d²` blocks one fill writes.
    pub(crate) fn blocks(self) -> usize {
        match self {
            Self::Scalar => 1,
            Self::Ard(d) => d,
        }
    }
}

/// A leaf of a [`DistanceKernel`] that reads the supply of one slot.
///
/// Crate-private payload of the hidden [`KernelSpec`] variant, in a tree of
/// this [`Supply`] kind. A coordinate tree ([`NoSupply`]) cannot hold one.
/// The slot's shape is the leaf's: a scalar slot holds an isotropic or
/// custom leaf, an ARD slot an ARD leaf.
#[derive(Clone, Debug, PartialEq)]
pub struct SuppliedSpec {
    pub(crate) slot: SlotId,
    /// The slot's number among the slots of its shape in the tree that
    /// holds the leaf: depth first, in order of first appearance, the
    /// order of [`DistanceKernel::slots`] split by shape. Every supply of
    /// the tree keeps its slots in this order. Set when the tree is built
    /// ([`DistanceKernel::from_spec`]).
    pub(crate) at: usize,
    pub(crate) leaf: SuppliedLeafSpec,
}

impl SuppliedSpec {
    /// What the slot supplies, from the leaf.
    pub(crate) fn shape(&self) -> SlotShape {
        match &self.leaf {
            SuppliedLeafSpec::Scalar(_) => SlotShape::Scalar,
            SuppliedLeafSpec::Ard(leaf) => SlotShape::Ard(leaf.dims()),
        }
    }
}

/// The leaf of a [`SuppliedSpec`], typed by the shape of its slot.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum SuppliedLeafSpec {
    Scalar(ScalarLeafSpec),
    Ard(ArdLeafSpec),
}

/// A leaf on one `d²` per pair.
#[derive(Clone, Debug, PartialEq)]
pub enum ScalarLeafSpec {
    Rbf(RbfKernel),
    Matern(MaternKernel),
    Periodic(PeriodicKernel),
    RationalQuadratic(RationalQuadraticKernel),
    Custom(CustomKernel),
}

/// A leaf on one `d²` per dimension per pair.
#[derive(Clone, Debug, PartialEq)]
pub enum ArdLeafSpec {
    Rbf(RbfArdKernel),
    Matern(MaternArdKernel),
    RationalQuadratic(RationalQuadraticArdKernel),
}

impl ArdLeafSpec {
    /// Number of lengthscales, one per dimension of the slot.
    pub(crate) fn dims(&self) -> usize {
        match self {
            Self::Rbf(leaf) => leaf.lengthscales().num_params(),
            Self::Matern(leaf) => leaf.lengthscales().num_params(),
            Self::RationalQuadratic(leaf) => leaf.lengthscales().num_params(),
        }
    }
}

/// Every variant of a typed leaf enum, with the leaf bound to `$leaf`.
macro_rules! each_leaf {
    ($value:expr, $leaf:ident => $body:expr) => {
        match $value {
            SuppliedLeafSpec::Scalar(ScalarLeafSpec::Rbf($leaf)) => $body,
            SuppliedLeafSpec::Scalar(ScalarLeafSpec::Matern($leaf)) => $body,
            SuppliedLeafSpec::Scalar(ScalarLeafSpec::Periodic($leaf)) => $body,
            SuppliedLeafSpec::Scalar(ScalarLeafSpec::RationalQuadratic($leaf)) => $body,
            SuppliedLeafSpec::Scalar(ScalarLeafSpec::Custom($leaf)) => $body,
            SuppliedLeafSpec::Ard(ArdLeafSpec::Rbf($leaf)) => $body,
            SuppliedLeafSpec::Ard(ArdLeafSpec::Matern($leaf)) => $body,
            SuppliedLeafSpec::Ard(ArdLeafSpec::RationalQuadratic($leaf)) => $body,
        }
    };
}

impl LeafParams for SuppliedLeafSpec {
    fn leaf_num_params(&self) -> usize {
        each_leaf!(self, leaf => leaf.leaf_num_params())
    }

    fn write_leaf_params(&self, out: &mut [f64], offset: &mut usize) -> Result<(), GprError> {
        each_leaf!(self, leaf => leaf.write_leaf_params(out, offset))
    }

    fn write_leaf_intervals(
        &self,
        out: &mut [Interval],
        offset: &mut usize,
    ) -> Result<(), GprError> {
        each_leaf!(self, leaf => leaf.write_leaf_intervals(out, offset))
    }

    fn apply_leaf_params(&mut self, params: &[f64], offset: &mut usize) -> Result<(), GprError> {
        each_leaf!(self, leaf => leaf.apply_leaf_params(params, offset))
    }
}

/// Names a supply of one squared distance per pair of samples.
///
/// [`Self::kernel`] makes a leaf that reads this supply. Every leaf made
/// from the same slot reads one supply, filled once. The supply itself is
/// bound to the slot for one call: [`Self::from_vec`] (moves),
/// [`Self::from_slice`] (copies), [`Self::borrow`] (reads in place for the
/// call), or [`Self::fill`] (a function writes it). A squared distance is
/// `d²`, column-major: the pair `(i, j)` is at `i + j * n_rows`.
///
/// Every `d²` must be finite and non-negative, and a square of the pairs of
/// one set (the training square, or the query square of a covariance) must
/// have a zero diagonal and be symmetric, exactly; otherwise the model
/// returns [`GprError::InvalidDistance`] at the first value that is not.
/// [`DistanceSource::tidy`] repairs what rounding leaves, within a
/// tolerance the caller names.
///
/// # Examples
///
/// ```rust
/// use gprx::kernel::{RbfKernel, ScalarDistance};
/// use gprx::{GaussianLikelihood, Gpr};
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let image = ScalarDistance::new();
/// let kernel = image.kernel(RbfKernel::new(1.0)?);
/// // Squared distances between three training samples.
/// let train = vec![0.0, 1.0, 4.0, 1.0, 0.0, 1.0, 4.0, 1.0, 0.0];
/// let fitted = Gpr::new(kernel, GaussianLikelihood::new(0.1)?)
///     .fit([image.from_vec(train)], 3, &[0.0, 1.0, 0.5])
///     .map_err(|(_, e)| e)?;
/// // Squared distances from the three training samples to one query.
/// let cross = [0.25, 0.25, 2.25];
/// let pred = fitted.predict([image.borrow(&cross)], 1)?;
/// assert_eq!(pred.mean.len(), 1);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScalarDistance {
    slot: SlotId,
}

impl Default for ScalarDistance {
    fn default() -> Self {
        Self::new()
    }
}

impl ScalarDistance {
    /// Returns a new slot, distinct from every other slot.
    ///
    /// See the example on [`ScalarDistance`].
    pub fn new() -> Self {
        Self {
            slot: SlotId::fresh(),
        }
    }

    pub(crate) fn with_slot(slot: SlotId) -> Self {
        Self { slot }
    }

    /// Returns a leaf that evaluates `leaf` on this slot's `d²`.
    ///
    /// The leaf is an isotropic RBF, Matérn, periodic, or rational
    /// quadratic, or a user [`KernelTerm`] (or a [`CustomKernel`]).
    ///
    /// See the example on [`ScalarDistance`].
    pub fn kernel(&self, leaf: impl ScalarDistanceLeaf) -> DistanceKernel {
        DistanceKernel::leaf(SuppliedSpec {
            slot: self.slot,
            at: 0,
            leaf: SuppliedLeafSpec::Scalar(leaf.into_leaf()),
        })
    }

    /// Binds an owned table of `d²` to this slot. An `f64` model keeps the
    /// buffer as its store, without a copy; an `f32` model casts it into a
    /// buffer of its own, and a `MixedPrecision` model keeps it and adds
    /// that cast.
    ///
    /// See the example on [`ScalarDistance`].
    pub fn from_vec(&self, d2: Vec<f64>) -> DistanceSource<'static> {
        self.source(ScalarData::Values(Cow::Owned(d2)))
    }

    /// Binds a copy of `d2` to this slot.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::ScalarDistance;
    ///
    /// let image = ScalarDistance::new();
    /// let train = [0.0, 1.0, 1.0, 0.0];
    /// let _source = image.from_slice(&train);
    /// ```
    pub fn from_slice(&self, d2: &[f64]) -> DistanceSource<'static> {
        self.source(ScalarData::Values(Cow::Owned(d2.to_vec())))
    }

    /// Binds `d2` to this slot for one call. Prediction reads it in place;
    /// a fit copies it into the model.
    ///
    /// See the example on [`ScalarDistance`].
    pub fn borrow<'a>(&self, d2: &'a [f64]) -> DistanceSource<'a> {
        self.source(ScalarData::Values(Cow::Borrowed(d2)))
    }

    /// Binds a function that writes this slot's `d²`.
    ///
    /// A fit calls it once into the buffer the model keeps, whatever the
    /// [`crate::DistanceCachePolicy`]; a prediction calls it once into
    /// scratch.
    /// See [`DistanceFill`].
    ///
    /// See the example on [`DistanceFill`].
    pub fn fill<'a>(&self, filler: &'a dyn DistanceFill) -> DistanceSource<'a> {
        self.source(ScalarData::Fill(filler))
    }

    fn source<'a>(&self, data: ScalarData<'a>) -> DistanceSource<'a> {
        DistanceSource {
            slot: self.slot,
            data: SourceData::Scalar(data),
            tidy: Tidy::Exact,
        }
    }
}

/// Names a supply of `d` squared distances per pair of samples, one per
/// dimension, for the ARD leaves.
///
/// [`Self::from_leaf`] makes the slot from its first leaf, whose
/// lengthscale `ℓ_k` scales dimension `k`: `r² = Σ_k d_k² / ℓ_k²`; the slot
/// has one dimension per lengthscale. [`Self::kernel`] puts further leaves
/// on the same slot. Bind the supply with [`Self::from_vecs`]
/// (moves), [`Self::from_slices`] (copies), [`Self::borrow`] (reads in place
/// for the call), or [`Self::fill`]. Each dimension is a column-major block
/// of `d²`, laid out and checked as for [`ScalarDistance`].
///
/// # Examples
///
/// ```rust
/// use gprx::kernel::{ArdDistance, RbfArdKernel};
/// use gprx::{GaussianLikelihood, Gpr};
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let (bands, kernel) = ArdDistance::from_leaf(RbfArdKernel::new(&[1.0, 2.0])?);
/// let b0 = vec![0.0, 1.0, 1.0, 0.0];
/// let b1 = vec![0.0, 4.0, 4.0, 0.0];
/// let fitted = Gpr::new(kernel, GaussianLikelihood::new(0.1)?)
///     .fit([bands.from_vecs(vec![b0, b1])], 2, &[0.0, 1.0])
///     .map_err(|(_, e)| e)?;
/// let (c0, c1) = ([0.25, 0.25], [1.0, 1.0]);
/// let pred = fitted.predict([bands.borrow(&[&c0, &c1])], 1)?;
/// assert_eq!(pred.mean.len(), 1);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArdDistance {
    slot: SlotId,
    dims: usize,
}

impl ArdDistance {
    /// Returns a new slot of as many squared distances per pair as `leaf`
    /// has lengthscales, and the leaf that evaluates `leaf` on it. The
    /// slot's dimensions are the leaf's, so they cannot disagree.
    ///
    /// See the example on [`ArdDistance`].
    pub fn from_leaf(leaf: impl ArdDistanceLeaf) -> (Self, DistanceKernel) {
        let bands = Self {
            slot: SlotId::fresh(),
            dims: leaf.lengthscale_count(),
        };
        let kernel = bands.leaf(leaf);
        (bands, kernel)
    }

    /// A slot of `dims` squared distances per pair with no leaf, for the
    /// tests of the stores.
    #[cfg(test)]
    pub(crate) fn of_dims(dims: usize) -> Self {
        Self {
            slot: SlotId::fresh(),
            dims,
        }
    }

    pub(crate) fn with_slot(slot: SlotId, dims: usize) -> Self {
        Self { slot, dims }
    }

    /// Returns the number of squared distances per pair.
    ///
    /// See the example on [`ArdDistance`].
    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Returns another leaf that evaluates the ARD `leaf` on this slot's
    /// `d²` (the first comes from [`Self::from_leaf`]): every leaf of the
    /// slot reads the same supply.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::DimensionMismatch`] if the leaf's lengthscale
    /// count is not [`Self::dims`]: `x_dim` is [`Self::dims`], `expected_dim`
    /// the lengthscale count.
    ///
    /// See the example on [`ArdDistance`].
    pub fn kernel(&self, leaf: impl ArdDistanceLeaf) -> Result<DistanceKernel, GprError> {
        let lengthscales = leaf.lengthscale_count();
        if lengthscales != self.dims {
            // The slot's `d²` per pair are the data's dimensions; the
            // leaf's lengthscales are the ones the kernel expects.
            return Err(GprError::DimensionMismatch {
                x_dim: self.dims,
                expected_dim: lengthscales,
            });
        }
        Ok(self.leaf(leaf))
    }

    /// The leaf of `leaf` on this slot, its lengthscale count already
    /// [`Self::dims`].
    fn leaf(&self, leaf: impl ArdDistanceLeaf) -> DistanceKernel {
        DistanceKernel::leaf(SuppliedSpec {
            slot: self.slot,
            at: 0,
            leaf: SuppliedLeafSpec::Ard(leaf.into_leaf()),
        })
    }

    /// Binds owned tables of `d²`, one per dimension, to this slot.
    ///
    /// An `f64` model checks the tables and keeps them as they are, so a
    /// fit copies nothing; it then holds `d · n²` values for the slot
    /// rather than the `d · n(n+1)/2` of the packed lower triangles it
    /// makes from [`Self::borrow`] (or when `tidy` repairs a table). An
    /// `f32` model packs them.
    ///
    /// See the example on [`ArdDistance`].
    pub fn from_vecs(&self, d2: Vec<Vec<f64>>) -> DistanceSource<'static> {
        self.source(ArdData::Blocks(d2))
    }

    /// Binds copies of `d2`, one table per dimension, to this slot: the
    /// copies are then handled as [`Self::from_vecs`], so an `f64` model
    /// keeps `d · n²` values. [`Self::borrow`] fits without the copy.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{ArdDistance, RbfArdKernel};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let (bands, _kernel) = ArdDistance::from_leaf(RbfArdKernel::new(&[1.0, 2.0])?);
    /// let (b0, b1) = ([0.0, 1.0, 1.0, 0.0], [0.0, 4.0, 4.0, 0.0]);
    /// let _source = bands.from_slices(&[&b0, &b1]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_slices(&self, d2: &[&[f64]]) -> DistanceSource<'static> {
        self.source(ArdData::Blocks(
            d2.iter().map(|block| block.to_vec()).collect(),
        ))
    }

    /// Binds `d2`, one table per dimension, for one call. Prediction reads
    /// the tables in place; a fit copies them into the model.
    ///
    /// See the example on [`ArdDistance`].
    pub fn borrow<'a>(&self, d2: &'a [&'a [f64]]) -> DistanceSource<'a> {
        self.source(ArdData::Slices(d2))
    }

    /// Binds a function that writes this slot's `d²`, every dimension in
    /// one call. See [`DistanceFill`].
    ///
    /// See the example on [`DistanceFill`].
    pub fn fill<'a>(&self, filler: &'a dyn DistanceFill) -> DistanceSource<'a> {
        self.source(ArdData::Fill(filler))
    }

    fn source<'a>(&self, data: ArdData<'a>) -> DistanceSource<'a> {
        DistanceSource {
            slot: self.slot,
            data: SourceData::Ard(self.dims, data),
            tidy: Tidy::Exact,
        }
    }
}

/// One slot of a [`DistanceKernel`].
///
/// # Examples
///
/// ```rust
/// use gprx::kernel::{DistanceSlot, RbfKernel, ScalarDistance};
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let image = ScalarDistance::new();
/// let kernel = image.kernel(RbfKernel::new(1.0)?);
/// assert_eq!(kernel.slots(), vec![DistanceSlot::Scalar(image)]);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum DistanceSlot {
    /// Marks a slot of one `d²` per pair.
    Scalar(ScalarDistance),
    /// Marks a slot of one `d²` per dimension per pair.
    Ard(ArdDistance),
}

impl DistanceSlot {
    pub(crate) fn id(self) -> SlotId {
        match self {
            Self::Scalar(slot) => slot.slot,
            Self::Ard(slot) => slot.slot,
        }
    }

    pub(crate) fn shape(self) -> SlotShape {
        match self {
            Self::Scalar(_) => SlotShape::Scalar,
            Self::Ard(slot) => SlotShape::Ard(slot.dims),
        }
    }

    pub(crate) fn from_parts(slot: SlotId, shape: SlotShape) -> Self {
        match shape {
            SlotShape::Scalar => Self::Scalar(ScalarDistance::with_slot(slot)),
            SlotShape::Ard(dims) => Self::Ard(ArdDistance::with_slot(slot, dims)),
        }
    }
}

mod sealed {
    use crate::kernel::KernelSpec;

    pub trait Leaf {
        fn into_leaf(self) -> super::ScalarLeafSpec;
    }

    pub trait ArdLeaf {
        fn into_leaf(self) -> super::ArdLeafSpec;
        fn lengthscale_count(&self) -> usize;
    }

    pub trait Points {
        /// Whether the kernel has coordinate leaves that read `x`.
        const POINTS: bool;
    }

    pub trait Model: Sized {
        /// Whether the kernel has coordinate leaves that read `x`.
        const POINTS: bool;
        /// What the tree holds besides coordinate leaves.
        type Supply: crate::kernel::Supply;
        fn into_spec(self) -> KernelSpec<Self::Supply>;
        fn from_spec(spec: KernelSpec<Self::Supply>) -> Self;
    }
}

/// Represents a leaf [`ScalarDistance::kernel`] accepts: [`RbfKernel`],
/// [`MaternKernel`], [`PeriodicKernel`], [`RationalQuadraticKernel`],
/// [`CustomKernel`], or a user [`KernelTerm`].
///
/// Sealed: the crate implements it.
///
/// See the example on [`ScalarDistance`].
pub trait ScalarDistanceLeaf: sealed::Leaf {}

/// Represents a leaf [`ArdDistance::kernel`] accepts: [`RbfArdKernel`],
/// [`MaternArdKernel`], or [`RationalQuadraticArdKernel`].
///
/// Sealed: the crate implements it.
///
/// See the example on [`ArdDistance`].
pub trait ArdDistanceLeaf: sealed::ArdLeaf {}

macro_rules! scalar_leaf {
    ($($leaf:ty => $variant:ident),*) => {$(
        impl sealed::Leaf for $leaf {
            fn into_leaf(self) -> ScalarLeafSpec {
                ScalarLeafSpec::$variant(self)
            }
        }
        impl ScalarDistanceLeaf for $leaf {}
    )*};
}

scalar_leaf!(
    RbfKernel => Rbf,
    MaternKernel => Matern,
    PeriodicKernel => Periodic,
    RationalQuadraticKernel => RationalQuadratic,
    CustomKernel => Custom
);

impl<K> sealed::Leaf for K
where
    K: KernelTerm<f64> + KernelTerm<f32> + Clone + fmt::Debug + Send + Sync + 'static,
{
    fn into_leaf(self) -> ScalarLeafSpec {
        ScalarLeafSpec::Custom(CustomKernel::new(self))
    }
}

impl<K> ScalarDistanceLeaf for K where
    K: KernelTerm<f64> + KernelTerm<f32> + Clone + fmt::Debug + Send + Sync + 'static
{
}

macro_rules! ard_leaf {
    ($($leaf:ty => $variant:ident),*) => {$(
        impl sealed::ArdLeaf for $leaf {
            fn into_leaf(self) -> ArdLeafSpec {
                ArdLeafSpec::$variant(self)
            }
            fn lengthscale_count(&self) -> usize {
                self.lengthscales().num_params()
            }
        }
        impl ArdDistanceLeaf for $leaf {}
    )*};
}

ard_leaf!(
    RbfArdKernel => Rbf,
    MaternArdKernel => Matern,
    RationalQuadraticArdKernel => RationalQuadratic
);

/// Marks a [`DistanceKernel`] whose leaves read only supplied distances
/// (and [`ConstantKernel`] / [`WhiteKernel`]). A model of it takes no
/// coordinates.
///
/// See the example on [`DistanceKernel`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DistanceOnly;

/// Marks a [`DistanceKernel`] that also holds coordinate leaves of a
/// [`KernelSpec`]. A model of it takes the column-major coordinates `x`
/// with the supplied distances.
///
/// See the example on [`DistanceKernel`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WithPoints;

/// Represents whether a [`DistanceKernel`] reads coordinates:
/// [`DistanceOnly`] or [`WithPoints`]. Sealed.
///
/// See the example on [`DistanceKernel`].
pub trait PointUse: sealed::Points + Send + Sync + 'static {}

impl sealed::Points for DistanceOnly {
    const POINTS: bool = false;
}
impl sealed::Points for WithPoints {
    const POINTS: bool = true;
}

impl PointUse for DistanceOnly {}
impl PointUse for WithPoints {}

/// Represents the [`PointUse`] of a sum or product of two kernels.
///
/// Sealed: [`DistanceOnly`] with [`DistanceOnly`] stays [`DistanceOnly`];
/// anything with [`WithPoints`] is [`WithPoints`].
///
/// See the example on [`DistanceKernel`].
pub trait JoinPoints<Rhs: PointUse>: PointUse {
    /// The joined marker.
    type Output: PointUse;
}

impl JoinPoints<DistanceOnly> for DistanceOnly {
    type Output = DistanceOnly;
}
impl JoinPoints<WithPoints> for DistanceOnly {
    type Output = WithPoints;
}
impl JoinPoints<DistanceOnly> for WithPoints {
    type Output = WithPoints;
}
impl JoinPoints<WithPoints> for WithPoints {
    type Output = WithPoints;
}

/// Represents a kernel expression whose leaves read supplied squared
/// distances.
///
/// Leaves come from [`ScalarDistance::kernel`] and [`ArdDistance::kernel`].
/// `+` and `*` combine them with each other, with [`ConstantKernel`] and
/// [`WhiteKernel`] (still [`DistanceOnly`]), and with a coordinate
/// [`KernelSpec`] (then [`WithPoints`]). Parameters follow the leaves in
/// depth-first, left-to-right order, as in [`KernelSpec`].
///
/// # Examples
///
/// ```rust
/// use gprx::kernel::{
///     ArdDistance, ConstantKernel, DistanceKernel, DistanceOnly, KernelSpec, RbfArdKernel,
///     RbfKernel, ScalarDistance, WithPoints,
/// };
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let image = ScalarDistance::new();
/// let (_bands, ard) = ArdDistance::from_leaf(RbfArdKernel::new(&[1.0, 0.5])?);
/// let only: DistanceKernel<DistanceOnly> =
///     ConstantKernel::new(2.0)? * image.kernel(RbfKernel::new(1.0)?) * ard;
/// assert_eq!(only.num_params(), 4);
/// let mixed: DistanceKernel<WithPoints> = only * KernelSpec::from(RbfKernel::new(0.5)?);
/// assert_eq!(mixed.slots().len(), 2);
/// # Ok(())
/// # }
/// ```
pub struct DistanceKernel<C: PointUse = DistanceOnly> {
    spec: KernelSpec<SuppliedSpec>,
    _points: PhantomData<C>,
}

impl<C: PointUse> Clone for DistanceKernel<C> {
    fn clone(&self) -> Self {
        // The leaves keep their numbers: the tree is the same.
        Self {
            spec: self.spec.clone(),
            _points: PhantomData,
        }
    }
}

impl<C: PointUse> PartialEq for DistanceKernel<C> {
    fn eq(&self, other: &Self) -> bool {
        self.spec == other.spec
    }
}

impl<C: PointUse> fmt::Debug for DistanceKernel<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DistanceKernel")
            .field("spec", &self.spec)
            .finish()
    }
}

impl DistanceKernel {
    fn leaf(spec: SuppliedSpec) -> Self {
        Self::from_spec(KernelSpec::Supplied(spec))
    }
}

impl<C: PointUse> DistanceKernel<C> {
    /// The kernel of `spec`, its leaves numbered in their tree
    /// ([`SuppliedSpec::at`]).
    pub(crate) fn from_spec(mut spec: KernelSpec<SuppliedSpec>) -> Self {
        number_leaves(&mut spec, &mut [Vec::new(), Vec::new()]);
        Self {
            spec,
            _points: PhantomData,
        }
    }

    /// Returns the number of flattened parameters.
    ///
    /// See the example on [`DistanceKernel`].
    pub fn num_params(&self) -> usize {
        self.spec.num_params()
    }

    /// Writes flattened `θ` in depth-first, left-to-right leaf order.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::LengthMismatch`] if `out` is the wrong length.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{RbfKernel, ScalarDistance};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let mut kernel = ScalarDistance::new().kernel(RbfKernel::new(1.0)?);
    /// let mut theta = [0.0];
    /// kernel.get_params(&mut theta)?;
    /// kernel.set_params(&[2.0_f64.ln()])?;
    /// assert_eq!(kernel.parameter_bindings().len(), 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_params(&self, out: &mut [f64]) -> Result<(), GprError> {
        self.spec.get_params(out)
    }

    /// Replaces flattened `θ`. All leaves are updated or none are.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::LengthMismatch`] if `params` is the wrong length,
    /// or the error of a leaf that rejects its slice.
    ///
    /// See the example on [`Self::get_params`].
    pub fn set_params(&mut self, params: &[f64]) -> Result<(), GprError> {
        self.spec.set_params(params)
    }

    /// Returns the mapping from flat indices to leaves.
    ///
    /// See the example on [`Self::get_params`].
    pub fn parameter_bindings(&self) -> Vec<ParameterBinding> {
        self.spec.parameter_bindings()
    }

    /// Returns the slots this expression reads, in depth-first,
    /// left-to-right order of their first leaf.
    ///
    /// Each source a fit or a prediction takes names one of these slots.
    ///
    /// See the example on [`DistanceSlot`].
    pub fn slots(&self) -> Vec<DistanceSlot> {
        spec_slots(&self.spec)
    }

    #[cfg(test)]
    pub(crate) fn spec(&self) -> &KernelSpec<SuppliedSpec> {
        &self.spec
    }
}

/// The slots of `spec` in depth-first order of first appearance; none for
/// a coordinate tree.
pub(crate) fn spec_slots<S: Supply>(spec: &KernelSpec<S>) -> Vec<DistanceSlot> {
    let mut out = Vec::new();
    collect_slots(spec, &mut out);
    out
}

fn collect_slots<S: Supply>(spec: &KernelSpec<S>, out: &mut Vec<DistanceSlot>) {
    match spec {
        KernelSpec::Supplied(leaf) => {
            let leaf = S::spec(leaf);
            // A slot id is minted by one constructor with its shape, and a
            // leaf takes its shape from the slot it was made on, so every
            // leaf of an id has the same shape.
            match out.iter().find(|slot| slot.id() == leaf.slot) {
                Some(seen) => debug_assert_eq!(seen.shape(), leaf.shape()),
                None => out.push(DistanceSlot::from_parts(leaf.slot, leaf.shape())),
            }
        }
        KernelSpec::Sum(left, right) | KernelSpec::Product(left, right) => {
            collect_slots(left, out);
            collect_slots(right, out);
        }
        _ => {}
    }
}

/// Sets each leaf's [`SuppliedSpec::at`] in one depth-first walk: `seen`
/// lists the slots met so far, scalar and ARD, in order, and a leaf's
/// number is its slot's place in the list of its shape, the slot added
/// when first met.
fn number_leaves(spec: &mut KernelSpec<SuppliedSpec>, seen: &mut [Vec<SlotId>; 2]) {
    match spec {
        KernelSpec::Supplied(leaf) => {
            let list = &mut seen[usize::from(matches!(leaf.shape(), SlotShape::Ard(_)))];
            leaf.at = match list.iter().position(|&slot| slot == leaf.slot) {
                Some(at) => at,
                None => {
                    list.push(leaf.slot);
                    list.len() - 1
                }
            };
        }
        KernelSpec::Sum(left, right) | KernelSpec::Product(left, right) => {
            number_leaves(left, seen);
            number_leaves(right, seen);
        }
        _ => {}
    }
}

macro_rules! distance_ops {
    ($trait:ident, $method:ident, $variant:ident) => {
        impl<C1, C2> $trait<DistanceKernel<C2>> for DistanceKernel<C1>
        where
            C1: JoinPoints<C2>,
            C2: PointUse,
        {
            type Output = DistanceKernel<<C1 as JoinPoints<C2>>::Output>;

            fn $method(self, rhs: DistanceKernel<C2>) -> Self::Output {
                DistanceKernel::from_spec(KernelSpec::$variant(
                    Box::new(self.spec),
                    Box::new(rhs.spec),
                ))
            }
        }

        impl<C: PointUse> $trait<KernelSpec> for DistanceKernel<C> {
            type Output = DistanceKernel<WithPoints>;

            fn $method(self, rhs: KernelSpec) -> Self::Output {
                DistanceKernel::from_spec(KernelSpec::$variant(
                    Box::new(self.spec),
                    Box::new(rhs.widen()),
                ))
            }
        }

        impl<C: PointUse> $trait<DistanceKernel<C>> for KernelSpec {
            type Output = DistanceKernel<WithPoints>;

            fn $method(self, rhs: DistanceKernel<C>) -> Self::Output {
                DistanceKernel::from_spec(KernelSpec::$variant(
                    Box::new(self.widen()),
                    Box::new(rhs.spec),
                ))
            }
        }

        distance_ops!(@leaf $trait, $method, $variant, ConstantKernel);
        distance_ops!(@leaf $trait, $method, $variant, WhiteKernel);
    };
    (@leaf $trait:ident, $method:ident, $variant:ident, $leaf:ty) => {
        impl<C: PointUse> $trait<$leaf> for DistanceKernel<C> {
            type Output = DistanceKernel<C>;

            fn $method(self, rhs: $leaf) -> Self::Output {
                DistanceKernel::from_spec(KernelSpec::$variant(
                    Box::new(self.spec),
                    Box::new(KernelSpec::from(rhs).widen()),
                ))
            }
        }

        impl<C: PointUse> $trait<DistanceKernel<C>> for $leaf {
            type Output = DistanceKernel<C>;

            fn $method(self, rhs: DistanceKernel<C>) -> Self::Output {
                DistanceKernel::from_spec(KernelSpec::$variant(
                    Box::new(KernelSpec::from(self).widen()),
                    Box::new(rhs.spec),
                ))
            }
        }
    };
}

distance_ops!(Add, add, Sum);
distance_ops!(Mul, mul, Product);

/// Represents the kernel type a model is built from: a coordinate
/// [`KernelSpec`] or a [`DistanceKernel`]. Sealed.
///
/// See the example on [`DistanceKernel`].
pub trait ModelKernel: sealed::Model + Clone + fmt::Debug + Send + Sync + 'static {}

impl sealed::Model for KernelSpec {
    const POINTS: bool = true;
    type Supply = NoSupply;

    fn into_spec(self) -> KernelSpec {
        self
    }

    fn from_spec(spec: KernelSpec) -> Self {
        spec
    }
}

impl ModelKernel for KernelSpec {}

impl<C: PointUse> sealed::Model for DistanceKernel<C> {
    const POINTS: bool = <C as sealed::Points>::POINTS;
    type Supply = SuppliedSpec;

    fn into_spec(self) -> KernelSpec<SuppliedSpec> {
        self.spec
    }

    fn from_spec(spec: KernelSpec<SuppliedSpec>) -> Self {
        Self::from_spec(spec)
    }
}

impl<C: PointUse> ModelKernel for DistanceKernel<C> {}

/// Represents a model kernel that reads coordinates: [`KernelSpec`] or
/// [`DistanceKernel<WithPoints>`]. A model of one takes `x`, an input
/// transform, and reports its feature count. Sealed.
///
/// See the example on [`DistanceKernel`].
pub trait PointKernel: ModelKernel {}

impl PointKernel for KernelSpec {}
impl PointKernel for DistanceKernel<WithPoints> {}

pub(crate) use sealed::Model as ModelKernelParts;

/// The declared tree of model kernel `K`.
pub(crate) type SpecOf<K> = KernelSpec<<K as sealed::Model>::Supply>;

/// The compiled tree of model kernel `K` for compute scalar `T`.
pub(crate) type CompiledOf<T, K> = crate::kernel::CompiledKernel<T, <K as sealed::Model>::Supply>;

/// Writes squared distances one column of pairs at a time.
///
/// The crate asks for the columns it needs, into a buffer it reuses, so a
/// fill never makes the crate hold a dense table it would not keep. For a
/// training square it asks, for each column `col`, only the rows
/// `col..n` (the lower triangle, diagonal included): the square is
/// symmetric by construction. For a block of two sets (train × query) it
/// asks every row. A fit calls it once per column and keeps what it wrote,
/// whatever the [`crate::DistanceCachePolicy`]; a prediction calls it into
/// scratch. The caller knows which samples the rows and columns are: it
/// binds a filler to a slot for one call with [`ScalarDistance::fill`] or
/// [`ArdDistance::fill`].
///
/// # Examples
///
/// ```rust
/// use gprx::kernel::{DistanceFill, RbfKernel, ScalarDistance};
/// use gprx::{GaussianLikelihood, Gpr};
/// use std::ops::Range;
///
/// /// Squared distances between two sets of one-dimensional samples.
/// struct Pairs<'a> {
///     rows: &'a [f64],
///     cols: &'a [f64],
/// }
///
/// impl DistanceFill for Pairs<'_> {
///     fn fill_column(&self, col: usize, rows: Range<usize>, out: &mut [f64]) {
///         for (slot, i) in out.iter_mut().zip(rows) {
///             let diff = self.rows[i] - self.cols[col];
///             *slot = diff * diff;
///         }
///     }
/// }
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let samples = [0.0, 1.0, 2.0];
/// let queries = [0.5];
/// let image = ScalarDistance::new();
/// let fitted = Gpr::new(image.kernel(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?)
///     .fit([image.fill(&Pairs { rows: &samples, cols: &samples })], 3, &[0.0, 1.0, 0.5])
///     .map_err(|(_, e)| e)?;
/// let pred = fitted.predict([image.fill(&Pairs { rows: &samples, cols: &queries })], 1)?;
/// assert_eq!(pred.mean.len(), 1);
/// # Ok(())
/// # }
/// ```
pub trait DistanceFill: Send + Sync {
    /// Writes `d²(i, col)` for each row `i` of `rows` into `out`, in row
    /// order. A slot of [`ArdDistance::dims`] `d` writes `d` runs one after
    /// another: dimension `k` at `k * rows.len()`.
    fn fill_column(&self, col: usize, rows: std::ops::Range<usize>, out: &mut [f64]);
}

/// Binds a supply of squared distances to one slot for one call.
///
/// Made by [`ScalarDistance`] and [`ArdDistance`]. A model call takes one
/// source per slot of its kernel, in any order.
///
/// See the example on [`ScalarDistance`].
pub struct DistanceSource<'a> {
    pub(crate) slot: SlotId,
    pub(crate) data: SourceData<'a>,
    pub(crate) tidy: Tidy,
}

impl DistanceSource<'_> {
    /// Repairs what rounding leaves in this source's tables, within
    /// `rel_tol` times the largest value of each table: a negative value or
    /// a non-zero diagonal becomes `0.0`, and two mirror entries of a
    /// square that differ become their mean. Past it the table is refused.
    ///
    /// Without it, a table must be exact: finite, non-negative, and, for a
    /// square, a zero diagonal and equal mirror entries. Ask for the repair
    /// when the table is built in a way that rounds (`‖a‖² + ‖b‖² − 2a·b`);
    /// a table computed pair by pair needs none. A repaired table that was
    /// borrowed is copied.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidConfig`] when `rel_tol` is not in
    /// `[0, 1)` (negative, `NaN`, or so large that any table would pass).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{RbfKernel, ScalarDistance};
    /// use gprx::{GaussianLikelihood, Gpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let image = ScalarDistance::new();
    /// // The mirror entries differ in the last digit.
    /// let train = [0.0, 1.0, 1.0 + 1e-15, 0.0];
    /// let fitted = Gpr::new(image.kernel(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?)
    ///     .fit([image.borrow(&train).tidy(1e-12)?], 2, &[0.0, 1.0])
    ///     .map_err(|(_, e)| e)?;
    /// assert_eq!(fitted.n(), 2);
    /// # Ok(())
    /// # }
    /// ```
    pub fn tidy(mut self, rel_tol: f64) -> Result<Self, GprError> {
        // Past `1` the tolerance reaches the largest value itself, so every
        // negative value, diagonal, and mirror gap would pass as rounding.
        if !(0.0..1.0).contains(&rel_tol) {
            return Err(GprError::InvalidConfig {
                reason: format!("tidy tolerance must be in [0, 1), got {rel_tol}"),
            });
        }
        self.tidy = Tidy::Within(rel_tol);
        Ok(self)
    }
}

/// How the crate checks the tables of one source.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Tidy {
    /// Exactly: finite, non-negative, a zero diagonal, equal mirror entries.
    Exact,
    /// Repaired within this tolerance, relative to the largest value.
    Within(f64),
}

impl fmt::Debug for DistanceSource<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let data = match &self.data {
            SourceData::Scalar(ScalarData::Values(v)) => format!("{} values", v.len()),
            SourceData::Ard(_, ArdData::Blocks(b)) => format!("{} blocks", b.len()),
            SourceData::Ard(_, ArdData::Slices(b)) => format!("{} borrowed blocks", b.len()),
            SourceData::Scalar(ScalarData::Fill(_)) | SourceData::Ard(_, ArdData::Fill(_)) => {
                "fill".to_owned()
            }
        };
        f.debug_struct("DistanceSource")
            .field("slot", &self.slot)
            .field("shape", &self.data.shape())
            .field("data", &data)
            .finish()
    }
}

/// The data of a [`DistanceSource`], typed by the shape of its slot: a
/// scalar slot's source holds one table, an ARD slot's one per dimension.
pub(crate) enum SourceData<'a> {
    /// A scalar slot's.
    Scalar(ScalarData<'a>),
    /// An ARD slot's with its number of dimensions.
    Ard(usize, ArdData<'a>),
}

/// The data of a scalar slot's source.
pub(crate) enum ScalarData<'a> {
    /// One table.
    Values(Cow<'a, [f64]>),
    /// A function that writes it.
    Fill(&'a dyn DistanceFill),
}

/// The data of an ARD slot's source.
pub(crate) enum ArdData<'a> {
    /// One owned table per dimension.
    Blocks(Vec<Vec<f64>>),
    /// One borrowed table per dimension.
    Slices(&'a [&'a [f64]]),
    /// A function that writes them.
    Fill(&'a dyn DistanceFill),
}

impl SourceData<'_> {
    /// The shape of the slot the data is for.
    pub(crate) fn shape(&self) -> SlotShape {
        match self {
            Self::Scalar(_) => SlotShape::Scalar,
            Self::Ard(dims, _) => SlotShape::Ard(*dims),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slots_are_distinct_and_ordered() {
        let a = ScalarDistance::new();
        let (b, ard) = ArdDistance::from_leaf(RbfArdKernel::new(&[1.0, 2.0]).expect("ell"));
        assert_eq!(b.dims(), 2);
        assert_ne!(a, ScalarDistance::new());
        let k = a.kernel(RbfKernel::new(1.0).expect("ell")) * ard
            + a.kernel(MaternKernel::new(1.0, crate::kernel::MaternNu::FiveHalves).expect("ell"))
            + b.kernel(RbfArdKernel::new(&[3.0, 4.0]).expect("ell"))
                .expect("dims");
        // A slot read by two leaves is listed once, with its one shape.
        assert_eq!(
            k.slots(),
            vec![DistanceSlot::Scalar(a), DistanceSlot::Ard(b)]
        );
        assert_eq!(k.num_params(), 6);
    }

    #[test]
    fn a_further_ard_leaf_must_match_the_slot() {
        let (b, _) = ArdDistance::from_leaf(RbfArdKernel::new(&[1.0, 2.0, 3.0]).expect("ell"));
        assert!(matches!(
            b.kernel(RbfArdKernel::new(&[1.0, 2.0]).expect("ell")),
            Err(GprError::DimensionMismatch {
                x_dim: 3,
                expected_dim: 2
            })
        ));
    }

    struct Zeros;

    impl DistanceFill for Zeros {
        fn fill_column(&self, _col: usize, _rows: std::ops::Range<usize>, out: &mut [f64]) {
            out.fill(0.0);
        }
    }

    #[test]
    fn sources_and_kernels_describe_themselves() {
        let a = ScalarDistance::default();
        let b = ArdDistance::of_dims(2);
        let (b0, b1) = ([0.0, 1.0], [0.0, 4.0]);
        let shown = [
            format!("{:?}", a.from_slice(&[0.0])),
            format!("{:?}", b.from_slices(&[&b0, &b1])),
            format!("{:?}", b.fill(&Zeros)),
        ];
        assert!(shown[0].contains("1 values"));
        assert!(shown[1].contains("2 blocks"));
        assert!(shown[2].contains("fill"));
        let rbf = RbfKernel::new(1.0).expect("ell");
        let k = a.kernel(rbf);
        assert_eq!(k.clone(), k);
        assert!(format!("{k:?}").starts_with("DistanceKernel"));
        assert_eq!(k.parameter_bindings().len(), k.num_params());
        // A coordinate kernel on the left makes a `WithPoints` kernel too.
        let mixed: DistanceKernel<WithPoints> = KernelSpec::from(rbf) * k.clone();
        assert_eq!(mixed.slots(), k.slots());
        let summed: DistanceKernel<WithPoints> = KernelSpec::from(rbf) + k;
        assert_eq!(summed.num_params(), 2);
    }
}
