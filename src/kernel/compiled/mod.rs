//! Execution-layer kernel: static dispatch over built-in leaves.

use super::{
    ConstantKernel, CustomKernel, LinearKernel, MaternArdKernel, MaternKernel, PeriodicKernel,
    RationalQuadraticArdKernel, RationalQuadraticKernel, RbfArdKernel, RbfKernel, Triangle,
    WhiteKernel, visit_triangle,
};
use crate::error::GprError;
use crate::kernel::dist::{ArdSqDiff, for_each_lower_col, lower_col};
use crate::kernel::leaf_params::LeafParams;
use crate::kernel::tree::{NoSupply, Supply};
use crate::kernel::{KernelScalar, KernelSpec};
use faer::reborrow::ReborrowMut;
use faer::{Mat, MatMut, MatRef};

mod apply;
mod coord;
mod grad;
pub(crate) mod gram;
mod hess;
pub(crate) mod supplied;
pub(crate) mod weighted;

use supplied::{RectSlots, SquareSlots};

#[cfg(test)]
mod leaf_table;
#[cfg(test)]
mod tests;

/// Whether a compiled tree evaluates from a distance matrix or from coordinates.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CoordMode {
    /// Isotropic leaves: squared Euclidean `dist`.
    Dist,
    /// ARD / linear leaves: point coordinates.
    Points,
    /// Constant and White: either a distance matrix (shape) or coordinates.
    Either,
    /// Dist leaves and Points leaves in one tree. Each leaf keeps its mode.
    Mixed,
}

/// Distance matrix, coordinates, optional ARD `(Δx_d)²`, and the supplied
/// distances of a Mixed tree. Without `dist`, a distance leaf computes its
/// distances from `x`.
#[derive(Clone, Copy)]
pub(crate) struct MixedKernelViews<'a, T = f64> {
    pub(crate) dist: Option<MatRef<'a, T>>,
    pub(crate) x: MatRef<'a, T>,
    pub(crate) ard_cache: Option<ArdSqDiff<'a, T>>,
    pub(crate) slots: Option<&'a dyn SquareSlots<T>>,
}

#[cfg(test)]
impl<'a, T> MixedKernelViews<'a, T> {
    pub(crate) fn new(dist: MatRef<'a, T>, x: MatRef<'a, T>) -> Self {
        Self {
            dist: Some(dist),
            x,
            ard_cache: None,
            slots: None,
        }
    }
}

/// The views of a rectangular block `K(x1, x2)`: coordinates, the
/// coordinate distances when the caller filled them, and the supplied
/// distances.
#[derive(Clone, Copy)]
pub(crate) struct CrossViews<'a, T = f64> {
    pub(crate) x1: MatRef<'a, T>,
    pub(crate) x2: MatRef<'a, T>,
    pub(crate) dist: Option<MatRef<'a, T>>,
    pub(crate) slots: Option<&'a dyn RectSlots<T>>,
}

impl<'a, T> CrossViews<'a, T> {
    /// Coordinates only.
    pub(crate) fn points(x1: MatRef<'a, T>, x2: MatRef<'a, T>) -> Self {
        Self {
            x1,
            x2,
            dist: None,
            slots: None,
        }
    }
}

/// Represents the compiled kernel.
///
/// Built-ins are enum arms; Sum/Product are flattened lists.
///
/// [`Self::apply`] and [`Self::grad`] take a scratch buffer of the same size
/// as `out`. A lone RBF does not write it. Nested rest terms that are
/// themselves sums or products may allocate one extra `n×n` buffer. Fit
/// will pass Workspace storage for the caller scratch. Cloning copies the
/// whole tree.
///
/// # Examples
///
/// ```rust
/// use gprx::kernel::{KernelSpec, RbfKernel};
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let spec = KernelSpec::from(RbfKernel::new(1.0)?)
///     + KernelSpec::from(RbfKernel::new(2.0)?);
/// let compiled = spec.compile();
/// assert_eq!(compiled.num_params(), 2);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum CompiledKernel<T: KernelScalar = f64, S: Supply = NoSupply> {
    /// Marks an isotropic RBF.
    Rbf(RbfKernel),
    /// Marks an ARD RBF (`θ_d = log(ℓ_d)`).
    RbfArd(RbfArdKernel),
    /// Marks an isotropic Matérn (`ν = 1/2`, `3/2`, or `5/2`).
    Matern(MaternKernel),
    /// Marks an ARD Matérn (`θ_d = log(ℓ_d)`).
    MaternArd(MaternArdKernel),
    /// Marks a periodic (exp-sine-squared).
    Periodic(PeriodicKernel),
    /// Marks an isotropic rational quadratic (`θ = [log(ℓ), log(α)]`).
    RationalQuadratic(RationalQuadraticKernel),
    /// Marks an ARD rational quadratic (`θ_d = log(ℓ_d)`, then `log(α)`).
    RationalQuadraticArd(RationalQuadraticArdKernel),
    /// Marks a constant `k = c`.
    Constant(ConstantKernel),
    /// Marks a linear `k = σ² xᵀ x'`.
    Linear(LinearKernel),
    /// Marks a white nugget.
    White(WhiteKernel),
    /// Marks a user-defined distance leaf ([`super::KernelTerm`]).
    Custom(CustomKernel<T>),
    /// Marks a flattened sum of compiled terms.
    Sum(Vec<CompiledKernel<T, S>>),
    /// Marks a flattened Hadamard product of compiled terms.
    Product(Vec<CompiledKernel<T, S>>),
    /// A leaf that reads supplied squared distances. A coordinate tree
    /// ([`NoSupply`]) cannot hold one.
    #[doc(hidden)]
    Supplied(S::Compiled<T>),
}

/// A coordinate leaf of a compiled tree, borrowed.
///
/// The leaf arms of the coordinate paths live here, so a path that picks
/// a mode per leaf (the mixed and supplied-distance paths) and a whole
/// coordinate tree share them.
#[derive(Clone, Copy)]
pub(crate) enum LeafRef<'a, T: KernelScalar> {
    Rbf(&'a RbfKernel),
    RbfArd(&'a RbfArdKernel),
    Matern(&'a MaternKernel),
    MaternArd(&'a MaternArdKernel),
    Periodic(&'a PeriodicKernel),
    RationalQuadratic(&'a RationalQuadraticKernel),
    RationalQuadraticArd(&'a RationalQuadraticArdKernel),
    Constant(&'a ConstantKernel),
    Linear(&'a LinearKernel),
    White(&'a WhiteKernel),
    Custom(&'a CustomKernel<T>),
}

/// One node of a compiled tree, borrowed: a coordinate leaf, a supplied
/// leaf, or a sum / product of terms.
pub(crate) enum Term<'a, T: KernelScalar, S: Supply> {
    Leaf(LeafRef<'a, T>),
    Supplied(&'a S::Compiled<T>),
    Sum(&'a [CompiledKernel<T, S>]),
    Product(&'a [CompiledKernel<T, S>]),
}

impl<T: KernelScalar, S: Supply> CompiledKernel<T, S> {
    /// This node, with its coordinate leaves in one arm.
    pub(crate) fn term(&self) -> Term<'_, T, S> {
        match self {
            Self::Rbf(leaf) => Term::Leaf(LeafRef::Rbf(leaf)),
            Self::RbfArd(leaf) => Term::Leaf(LeafRef::RbfArd(leaf)),
            Self::Matern(leaf) => Term::Leaf(LeafRef::Matern(leaf)),
            Self::MaternArd(leaf) => Term::Leaf(LeafRef::MaternArd(leaf)),
            Self::Periodic(leaf) => Term::Leaf(LeafRef::Periodic(leaf)),
            Self::RationalQuadratic(leaf) => Term::Leaf(LeafRef::RationalQuadratic(leaf)),
            Self::RationalQuadraticArd(leaf) => Term::Leaf(LeafRef::RationalQuadraticArd(leaf)),
            Self::Constant(leaf) => Term::Leaf(LeafRef::Constant(leaf)),
            Self::Linear(leaf) => Term::Leaf(LeafRef::Linear(leaf)),
            Self::White(leaf) => Term::Leaf(LeafRef::White(leaf)),
            Self::Custom(leaf) => Term::Leaf(LeafRef::Custom(leaf)),
            Self::Supplied(leaf) => Term::Supplied(leaf),
            Self::Sum(terms) => Term::Sum(terms),
            Self::Product(terms) => Term::Product(terms),
        }
    }

    pub(crate) fn from_spec(spec: &KernelSpec<S>) -> Self {
        match spec {
            KernelSpec::Rbf(leaf) => Self::Rbf(*leaf),
            KernelSpec::RbfArd(leaf) => Self::RbfArd(leaf.clone()),
            KernelSpec::Matern(leaf) => Self::Matern(*leaf),
            KernelSpec::MaternArd(leaf) => Self::MaternArd(leaf.clone()),
            KernelSpec::Periodic(leaf) => Self::Periodic(*leaf),
            KernelSpec::RationalQuadratic(leaf) => Self::RationalQuadratic(*leaf),
            KernelSpec::RationalQuadraticArd(leaf) => Self::RationalQuadraticArd(leaf.clone()),
            KernelSpec::Constant(leaf) => Self::Constant(*leaf),
            KernelSpec::Linear(leaf) => Self::Linear(*leaf),
            KernelSpec::White(leaf) => Self::White(*leaf),
            KernelSpec::Custom(leaf) => Self::Custom(leaf.with_scalar()),
            KernelSpec::Supplied(leaf) => Self::Supplied(S::compile(leaf)),
            KernelSpec::Sum(left, right) => {
                let mut terms = Vec::new();
                flatten_sum(left, &mut terms);
                flatten_sum(right, &mut terms);
                Self::Sum(terms)
            }
            KernelSpec::Product(left, right) => {
                let mut terms = Vec::new();
                flatten_product(left, &mut terms);
                flatten_product(right, &mut terms);
                Self::Product(terms)
            }
        }
    }

    /// Returns the number of flattened parameters.
    ///
    /// See the example on [`CompiledKernel`].
    pub fn num_params(&self) -> usize {
        match self {
            Self::Rbf(leaf) => leaf.num_params(),
            Self::RbfArd(leaf) => leaf.num_params(),
            Self::Matern(leaf) => leaf.num_params(),
            Self::MaternArd(leaf) => leaf.num_params(),
            Self::Periodic(leaf) => leaf.num_params(),
            Self::RationalQuadratic(leaf) => leaf.num_params(),
            Self::RationalQuadraticArd(leaf) => leaf.num_params(),
            Self::Constant(leaf) => leaf.num_params(),
            Self::Linear(leaf) => leaf.num_params(),
            Self::White(leaf) => leaf.num_params(),
            Self::Custom(leaf) => leaf.num_params(),
            Self::Supplied(leaf) => S::compiled(leaf).leaf.leaf_num_params(),
            Self::Sum(terms) | Self::Product(terms) => terms.iter().map(Self::num_params).sum(),
        }
    }

    /// Writes flattened `θ` in depth-first, left-to-right leaf order.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::LengthMismatch`] if `out` is the wrong length
    /// or a custom leaf rejects the write.
    ///
    /// See the example on [`CompiledKernel`].
    pub fn get_params(&self, out: &mut [f64]) -> Result<(), GprError> {
        crate::data::require_count(out.len(), self.num_params(), "kernel parameters")?;
        let mut offset = 0;
        self.write_params(out, &mut offset)
    }

    /// Replaces flattened `θ`.
    ///
    /// All leaves are updated or none are.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::LengthMismatch`] if `params` is the wrong
    /// length or a leaf rejects its slice.
    ///
    /// See the example on [`CompiledKernel`].
    pub fn set_params(&mut self, params: &[f64]) -> Result<(), GprError> {
        crate::data::require_count(params.len(), self.num_params(), "kernel parameters")?;
        let mut next = self.clone();
        let mut offset = 0;
        next.apply_params(params, &mut offset)?;
        *self = next;
        Ok(())
    }

    /// Writes flattened `θ` in place without cloning the tree.
    ///
    /// `prev` is the current `θ`. When a leaf rejects its slice, the leaves
    /// already written are put back from `prev`, so the tree is unchanged on
    /// error. Built-in leaves validate before they write; a custom leaf that
    /// rejects `prev` on the way back leaves its own value as it chose.
    pub(crate) fn set_params_in_place(
        &mut self,
        params: &[f64],
        prev: &[f64],
    ) -> Result<(), GprError> {
        crate::data::require_count(params.len(), self.num_params(), "kernel parameters")?;
        let mut offset = 0;
        if let Err(err) = self.apply_params(params, &mut offset) {
            let mut offset = 0;
            let _ = self.apply_params(prev, &mut offset);
            return Err(err);
        }
        Ok(())
    }

    /// Returns the number of compiled leaves in this tree.
    pub(crate) fn leaf_count(&self) -> usize {
        match self {
            Self::Sum(terms) | Self::Product(terms) => terms.iter().map(Self::leaf_count).sum(),
            _ => 1,
        }
    }

    /// Returns the leaf that owns kernel parameter `param_idx`.
    pub(crate) fn leaf_index_for_param(&self, param_idx: usize) -> Result<usize, GprError> {
        if param_idx >= self.num_params() {
            return Err(GprError::IndexOutOfRange {
                reason: format!("kernel parameter index {param_idx} is out of range"),
            });
        }
        let mut offset = 0;
        let mut leaf = 0;
        if self.locate_leaf(param_idx, &mut offset, &mut leaf) {
            Ok(leaf)
        } else {
            Err(GprError::IndexOutOfRange {
                reason: format!("kernel parameter index {param_idx} is out of range"),
            })
        }
    }

    fn locate_leaf(&self, param_idx: usize, offset: &mut usize, leaf: &mut usize) -> bool {
        match self {
            Self::Sum(terms) | Self::Product(terms) => terms
                .iter()
                .any(|term| term.locate_leaf(param_idx, offset, leaf)),
            _ => {
                let start = *offset;
                *offset += self.num_params();
                if param_idx >= start && param_idx < *offset {
                    true
                } else {
                    *leaf += 1;
                    false
                }
            }
        }
    }

    /// Returns the compiled leaf at depth-first index `leaf`.
    pub(crate) fn leaf_at(&self, leaf: usize) -> Result<&Self, GprError> {
        let mut remaining = leaf;
        self.find_leaf_at(&mut remaining)
            .ok_or_else(|| GprError::IndexOutOfRange {
                reason: format!("leaf index {leaf} is out of range"),
            })
    }

    fn find_leaf_at(&self, remaining: &mut usize) -> Option<&Self> {
        match self {
            Self::Sum(terms) | Self::Product(terms) => {
                for term in terms {
                    if let Some(found) = term.find_leaf_at(remaining) {
                        return Some(found);
                    }
                }
                None
            }
            _ => {
                if *remaining == 0 {
                    Some(self)
                } else {
                    *remaining -= 1;
                    None
                }
            }
        }
    }

    /// Whether `∂K/∂θ` reads an output-shaped `scratch`: a product, or a
    /// custom leaf that may hold its distances there.
    pub(crate) fn needs_grad_scratch(&self) -> bool {
        // Every leaf is listed, so a new leaf is a compile error here until
        // its answer is chosen (docs/architecture.md, adding a leaf).
        match self {
            Self::Product(_) | Self::Custom(_) => true,
            Self::Supplied(leaf) => S::compiled(leaf).needs_grad_scratch(),
            Self::Sum(terms) => terms.iter().any(Self::needs_grad_scratch),
            Self::Rbf(_)
            | Self::RbfArd(_)
            | Self::Matern(_)
            | Self::MaternArd(_)
            | Self::Periodic(_)
            | Self::RationalQuadratic(_)
            | Self::RationalQuadraticArd(_)
            | Self::Constant(_)
            | Self::Linear(_)
            | Self::White(_) => false,
        }
    }

    /// Combines cached leaf Grams into `out` (sum / product tree).
    ///
    /// `scratch` must match `out`. `nested` grows to the levels a nested
    /// sum / product reads ([`Self::nested_depth`]).
    pub(crate) fn combine_from_leaf_grams(
        &self,
        grams: &[Mat<T>],
        mut out: MatMut<'_, T>,
        mut scratch: MatMut<'_, T>,
        nested: &mut Vec<Mat<T>>,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        if grams.len() != self.leaf_count() {
            return Err(GprError::LengthMismatch {
                reason: format!(
                    "expected {} leaf Grams, got {}",
                    self.leaf_count(),
                    grams.len()
                ),
            });
        }
        require_scratch_shape(out.as_ref(), scratch.as_ref())?;
        visit_triangle(out.nrows(), uplo, |row, col| {
            out[(row, col)] = T::from_f64(0.0);
        });
        ensure_nested(nested, self.nested_depth(), out.nrows(), out.ncols());
        let mut index = 0;
        self.write_from_leaf_grams(
            grams,
            &mut index,
            out.as_mut(),
            scratch.as_mut(),
            nested,
            uplo,
        )?;
        if index != grams.len() {
            return Err(GprError::LengthMismatch {
                reason: "leaf Gram walk did not consume every leaf".to_owned(),
            });
        }
        Ok(())
    }

    fn write_from_leaf_grams(
        &self,
        grams: &[Mat<T>],
        index: &mut usize,
        dest: MatMut<'_, T>,
        scratch: MatMut<'_, T>,
        nested: &mut Nested<T>,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        let fold = CachedFold {
            grams,
            index,
            nested,
            uplo,
        };
        match self {
            Self::Sum(terms) => fold.run(terms, dest, scratch, add_triangle),
            Self::Product(terms) => fold.run(terms, dest, scratch, mul_triangle),
            _ => {
                if *index >= grams.len() {
                    return Err(GprError::LengthMismatch {
                        reason: "leaf Gram walk ran past the cache".to_owned(),
                    });
                }
                copy_triangle(dest, grams[*index].as_ref(), uplo);
                *index += 1;
                Ok(())
            }
        }
    }

    pub(crate) fn coord_mode(&self) -> Result<CoordMode, GprError> {
        match self {
            Self::Rbf(_)
            | Self::Matern(_)
            | Self::Periodic(_)
            | Self::RationalQuadratic(_)
            | Self::Custom(_) => Ok(CoordMode::Dist),
            Self::RbfArd(_)
            | Self::Linear(_)
            | Self::MaternArd(_)
            | Self::RationalQuadraticArd(_) => Ok(CoordMode::Points),
            Self::Constant(_) | Self::White(_) => Ok(CoordMode::Either),
            Self::Supplied(_) => Ok(CoordMode::Mixed),
            Self::Sum(terms) | Self::Product(terms) => {
                let (first, rest) = split_terms(terms)?;
                let mut mode = first.coord_mode()?;
                for term in rest {
                    mode = merge_coord_mode(mode, term.coord_mode()?);
                }
                Ok(mode)
            }
        }
    }

    pub(crate) fn needs_ard_sq_diff(&self) -> bool {
        // Every leaf is listed (see `needs_grad_scratch`).
        match self {
            Self::RbfArd(_) | Self::MaternArd(_) | Self::RationalQuadraticArd(_) => true,
            Self::Sum(terms) | Self::Product(terms) => terms.iter().any(Self::needs_ard_sq_diff),
            Self::Rbf(_)
            | Self::Matern(_)
            | Self::Periodic(_)
            | Self::RationalQuadratic(_)
            | Self::Constant(_)
            | Self::Linear(_)
            | Self::White(_)
            | Self::Custom(_)
            | Self::Supplied(_) => false,
        }
    }

    /// Whether a leaf outside the supplied ones reads coordinate distances.
    pub(crate) fn has_coord_dist_leaf(&self) -> bool {
        match self {
            Self::Rbf(_)
            | Self::Matern(_)
            | Self::Periodic(_)
            | Self::RationalQuadratic(_)
            | Self::Custom(_) => true,
            Self::Sum(terms) | Self::Product(terms) => terms.iter().any(Self::has_coord_dist_leaf),
            Self::RbfArd(_)
            | Self::MaternArd(_)
            | Self::RationalQuadraticArd(_)
            | Self::Constant(_)
            | Self::Linear(_)
            | Self::White(_)
            | Self::Supplied(_) => false,
        }
    }

    fn write_params(&self, out: &mut [f64], offset: &mut usize) -> Result<(), GprError> {
        match self {
            Self::Rbf(leaf) => leaf.write_leaf_params(out, offset),
            Self::RbfArd(leaf) => leaf.write_leaf_params(out, offset),
            Self::Matern(leaf) => leaf.write_leaf_params(out, offset),
            Self::MaternArd(leaf) => leaf.write_leaf_params(out, offset),
            Self::Periodic(leaf) => leaf.write_leaf_params(out, offset),
            Self::RationalQuadratic(leaf) => leaf.write_leaf_params(out, offset),
            Self::RationalQuadraticArd(leaf) => leaf.write_leaf_params(out, offset),
            Self::Constant(leaf) => leaf.write_leaf_params(out, offset),
            Self::Linear(leaf) => leaf.write_leaf_params(out, offset),
            Self::White(leaf) => leaf.write_leaf_params(out, offset),
            Self::Custom(leaf) => leaf.write_leaf_params(out, offset),
            Self::Supplied(leaf) => S::compiled(leaf).leaf.write_leaf_params(out, offset),
            Self::Sum(terms) | Self::Product(terms) => {
                for term in terms {
                    term.write_params(out, offset)?;
                }
                Ok(())
            }
        }
    }

    fn apply_params(&mut self, params: &[f64], offset: &mut usize) -> Result<(), GprError> {
        match self {
            Self::Rbf(leaf) => leaf.apply_leaf_params(params, offset),
            Self::RbfArd(leaf) => leaf.apply_leaf_params(params, offset),
            Self::Matern(leaf) => leaf.apply_leaf_params(params, offset),
            Self::MaternArd(leaf) => leaf.apply_leaf_params(params, offset),
            Self::Periodic(leaf) => leaf.apply_leaf_params(params, offset),
            Self::RationalQuadratic(leaf) => leaf.apply_leaf_params(params, offset),
            Self::RationalQuadraticArd(leaf) => leaf.apply_leaf_params(params, offset),
            Self::Constant(leaf) => leaf.apply_leaf_params(params, offset),
            Self::Linear(leaf) => leaf.apply_leaf_params(params, offset),
            Self::White(leaf) => leaf.apply_leaf_params(params, offset),
            Self::Custom(leaf) => leaf.apply_leaf_params(params, offset),
            Self::Supplied(leaf) => S::compiled_mut(leaf).leaf.apply_leaf_params(params, offset),
            Self::Sum(terms) | Self::Product(terms) => {
                for term in terms {
                    term.apply_params(params, offset)?;
                }
                Ok(())
            }
        }
    }

    /// Levels of [`Nested`] buffers this tree reads: one for each nesting
    /// level whose sum / product term needs a buffer of its own.
    pub(crate) fn nested_depth(&self) -> usize {
        match self {
            Self::Sum(terms) | Self::Product(terms) => {
                let deepest = terms.iter().map(Self::nested_depth).max().unwrap_or(0);
                deepest + usize::from(terms.iter().any(Self::needs_internal_scratch))
            }
            _ => 0,
        }
    }

    /// Fresh [`Nested`] buffers for one `rows × cols` call. Empty (no
    /// allocation) unless a sum / product nests another multi-term one.
    pub(crate) fn nested_buffers(&self, rows: usize, cols: usize) -> Vec<Mat<T>> {
        (0..self.nested_depth())
            .map(|_| Mat::zeros(rows, cols))
            .collect()
    }

    fn needs_internal_scratch(&self) -> bool {
        match self {
            Self::Rbf(_)
            | Self::RbfArd(_)
            | Self::Matern(_)
            | Self::MaternArd(_)
            | Self::Periodic(_)
            | Self::RationalQuadratic(_)
            | Self::RationalQuadraticArd(_)
            | Self::Constant(_)
            | Self::Linear(_)
            | Self::White(_)
            | Self::Custom(_)
            | Self::Supplied(_) => false,
            Self::Sum(terms) | Self::Product(terms) => {
                terms.len() > 1 || terms.iter().any(Self::needs_internal_scratch)
            }
        }
    }
}

fn flatten_sum<T: KernelScalar, S: Supply>(
    spec: &KernelSpec<S>,
    out: &mut Vec<CompiledKernel<T, S>>,
) {
    match spec {
        KernelSpec::Sum(left, right) => {
            flatten_sum(left, out);
            flatten_sum(right, out);
        }
        other => out.push(CompiledKernel::<T, S>::from_spec(other)),
    }
}

fn flatten_product<T: KernelScalar, S: Supply>(
    spec: &KernelSpec<S>,
    out: &mut Vec<CompiledKernel<T, S>>,
) {
    match spec {
        KernelSpec::Product(left, right) => {
            flatten_product(left, out);
            flatten_product(right, out);
        }
        other => out.push(CompiledKernel::<T, S>::from_spec(other)),
    }
}

/// Buffers for sum / product terms nested inside another sum / product, one
/// per level (see [`CompiledKernel::nested_depth`]), each at least the
/// output's shape. Scratch: contents mean nothing between calls.
pub(crate) type Nested<T> = [Mat<T>];

/// Grows `levels` to `depth` buffers of at least `rows × cols`. Allocates
/// only when a level is missing or too small.
/// [`ensure_nested`] at the depth `compiled` needs.
pub(crate) fn ensure_nested_levels<T: KernelScalar, S: Supply>(
    levels: &mut Vec<Mat<T>>,
    compiled: &CompiledKernel<T, S>,
    rows: usize,
    cols: usize,
) {
    ensure_nested(levels, compiled.nested_depth(), rows, cols);
}

pub(crate) fn ensure_nested<T: KernelScalar>(
    levels: &mut Vec<Mat<T>>,
    depth: usize,
    rows: usize,
    cols: usize,
) {
    if levels.len() < depth {
        levels.resize_with(depth, || Mat::zeros(0, 0));
    }
    for level in levels.iter_mut().take(depth) {
        if level.nrows() < rows || level.ncols() < cols {
            *level = Mat::zeros(rows.max(level.nrows()), cols.max(level.ncols()));
        }
    }
}

/// The scratch and deeper levels for `term` writing a `rows × cols` block.
///
/// A multi-term sum / product takes the first [`Nested`] level as its
/// scratch; any other term reads `fallback` (distinct from its output).
fn term_scratch<'a, T: KernelScalar, S: Supply>(
    term: &CompiledKernel<T, S>,
    rows: usize,
    cols: usize,
    fallback: MatMut<'a, T>,
    nested: &'a mut Nested<T>,
) -> Result<(MatMut<'a, T>, &'a mut Nested<T>), GprError> {
    if !term.needs_internal_scratch() {
        return Ok((fallback, nested));
    }
    let (level, deeper) = nested
        .split_first_mut()
        .ok_or(GprError::WorkspaceTooSmall)?;
    if level.nrows() < rows || level.ncols() < cols {
        return Err(GprError::WorkspaceTooSmall);
    }
    Ok((level.as_mut().submatrix_mut(0, 0, rows, cols), deeper))
}

impl<T: KernelScalar, S: Supply> CompiledKernel<T, S> {
    /// [`CompiledKernel::require_columns`] for a coordinate tree; a tree of
    /// supplied distances may read no coordinates. The leaves read only the
    /// rows of `x` (Constant and White), so the entry points check here.
    pub(crate) fn require_tree_columns(&self, x: MatRef<'_, T>) -> Result<(), GprError> {
        match S::coordinates(self) {
            Some(tree) => tree.require_columns(x),
            None => Ok(()),
        }
    }
}

impl<T: KernelScalar> CompiledKernel<T> {
    /// Rejects coordinates without a column ([`GprError::EmptyInput`]): a
    /// coordinate tree reads at least one.
    pub(crate) fn require_columns(&self, x: MatRef<'_, T>) -> Result<(), GprError> {
        if x.ncols() == 0 {
            return Err(GprError::EmptyInput);
        }
        Ok(())
    }
}

fn require_scratch_shape<T>(out: MatRef<'_, T>, scratch: MatRef<'_, T>) -> Result<(), GprError> {
    if scratch.nrows() == out.nrows() && scratch.ncols() == out.ncols() {
        Ok(())
    } else {
        Err(GprError::WorkspaceTooSmall)
    }
}

fn ard_needs_coords() -> GprError {
    GprError::UnsupportedKernelOperation {
        reason: "ARD RBF evaluates from coordinates, not a scalar distance matrix".to_owned(),
    }
}

fn merge_coord_mode(a: CoordMode, b: CoordMode) -> CoordMode {
    use CoordMode::{Dist, Either, Mixed, Points};
    match (a, b) {
        (Either, other) | (other, Either) => other,
        (Dist, Dist) => Dist,
        (Points, Points) => Points,
        (Mixed, _) | (_, Mixed) | (Dist, Points) | (Points, Dist) => Mixed,
    }
}

/// The first term of a sum / product and the rest.
type Split<'a, T, S> = (&'a CompiledKernel<T, S>, &'a [CompiledKernel<T, S>]);

fn split_terms<T: KernelScalar, S: Supply>(
    terms: &[CompiledKernel<T, S>],
) -> Result<Split<'_, T, S>, GprError> {
    terms
        .split_first()
        .ok_or_else(|| GprError::UnsupportedKernelOperation {
            reason: "sum/product has no terms".to_owned(),
        })
}

/// One fold over cached leaf Grams: the walk position and the nested levels.
struct CachedFold<'g, 'n, T> {
    grams: &'g [Mat<T>],
    index: &'g mut usize,
    nested: &'n mut Nested<T>,
    uplo: Triangle,
}

impl<T: KernelScalar> CachedFold<'_, '_, T> {
    fn run<S: Supply>(
        self,
        terms: &[CompiledKernel<T, S>],
        mut dest: MatMut<'_, T>,
        mut scratch: MatMut<'_, T>,
        combine: fn(MatMut<'_, T>, MatRef<'_, T>, Triangle),
    ) -> Result<(), GprError> {
        let Self {
            grams,
            index,
            nested,
            uplo,
        } = self;
        let (first, rest) = split_terms(terms)?;
        first.write_from_leaf_grams(grams, index, dest.as_mut(), scratch.as_mut(), nested, uplo)?;
        let (rows, cols) = (dest.nrows(), dest.ncols());
        for term in rest {
            let (own, deeper) = term_scratch(term, rows, cols, dest.as_mut(), &mut *nested)?;
            term.write_from_leaf_grams(grams, index, scratch.as_mut(), own, deeper, uplo)?;
            combine(dest.as_mut(), scratch.as_ref(), uplo);
        }
        Ok(())
    }
}

fn copy_triangle<T: Copy>(mut dest: MatMut<'_, T>, src: MatRef<'_, T>, uplo: Triangle) {
    visit_triangle(dest.nrows(), uplo, |row, col| {
        dest[(row, col)] = src[(row, col)];
    });
}

fn add_triangle<T: KernelScalar>(acc: MatMut<'_, T>, src: MatRef<'_, T>, uplo: Triangle) {
    zip_triangle(acc, src, uplo, |a, s| a + s);
}

fn mul_triangle<T: KernelScalar>(acc: MatMut<'_, T>, src: MatRef<'_, T>, uplo: Triangle) {
    zip_triangle(acc, src, uplo, |a, s| a * s);
}

/// `acc = op(acc, src)` on `uplo`; the lower triangle runs on the Rayon pool.
fn zip_triangle<T: KernelScalar>(
    mut acc: MatMut<'_, T>,
    src: MatRef<'_, T>,
    uplo: Triangle,
    op: impl Fn(T, T) -> T + Sync,
) {
    let n = acc.nrows();
    if matches!(uplo, Triangle::Lower) {
        for_each_lower_col(acc, &|col, mut rows| {
            if let (Some(dest), Some(from)) =
                (rows.rb_mut().try_as_col_major_mut(), lower_col(src, col))
            {
                for (a, &b) in dest.as_slice_mut().iter_mut().zip(from) {
                    *a = op(*a, b);
                }
            } else {
                for i in 0..rows.nrows() {
                    rows[i] = op(rows[i], src[(col + i, col)]);
                }
            }
        });
        return;
    }
    visit_triangle(n, uplo, |row, col| {
        acc[(row, col)] = op(acc[(row, col)], src[(row, col)]);
    });
}
