//! Execution-layer kernel: static dispatch over built-in leaves.

use super::{
    ConstantKernel, CustomKernel, LinearKernel, MaternArdKernel, MaternKernel, PeriodicKernel,
    RationalQuadraticArdKernel, RationalQuadraticKernel, RbfArdKernel, RbfKernel, Triangle,
    WhiteKernel, visit_triangle,
};
use crate::error::GprError;
use crate::kernel::{KernelScalar, KernelSpec};
use faer::{Mat, MatMut, MatRef};

mod apply;
mod f32_eval;
mod grad;
pub(crate) mod gram;
mod hess;

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

/// Distance matrix, coordinates, and optional ARD `(Δx_d)²` for a Mixed tree.
#[derive(Clone, Copy)]
pub(crate) struct MixedKernelViews<'a, T = f64> {
    pub(crate) dist: MatRef<'a, T>,
    pub(crate) x: MatRef<'a, T>,
    pub(crate) ard_cache: Option<MatRef<'a, T>>,
}

impl<'a, T> MixedKernelViews<'a, T> {
    pub(crate) fn new(dist: MatRef<'a, T>, x: MatRef<'a, T>) -> Self {
        Self {
            dist,
            x,
            ard_cache: None,
        }
    }
}

/// Compiled kernel. Built-ins are enum arms; Sum/Product are flattened lists.
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
pub enum CompiledKernel<T: KernelScalar = f64> {
    /// Isotropic RBF.
    Rbf(RbfKernel),
    /// ARD RBF (`θ_d = log(ℓ_d)`).
    RbfArd(RbfArdKernel),
    /// Isotropic Matérn (`ν = 1/2`, `3/2`, or `5/2`).
    Matern(MaternKernel),
    /// ARD Matérn (`θ_d = log(ℓ_d)`).
    MaternArd(MaternArdKernel),
    /// Periodic (exp-sine-squared).
    Periodic(PeriodicKernel),
    /// Isotropic rational quadratic (`θ = [log(ℓ), log(α)]`).
    RationalQuadratic(RationalQuadraticKernel),
    /// ARD rational quadratic (`θ_d = log(ℓ_d)`, then `log(α)`).
    RationalQuadraticArd(RationalQuadraticArdKernel),
    /// Constant `k = c`.
    Constant(ConstantKernel),
    /// Linear `k = σ² xᵀ x'`.
    Linear(LinearKernel),
    /// White nugget.
    White(WhiteKernel),
    /// User-defined distance leaf ([`super::KernelTerm`]).
    Custom(CustomKernel<T>),
    /// Flattened sum of compiled terms.
    Sum(Vec<CompiledKernel<T>>),
    /// Flattened Hadamard product of compiled terms.
    Product(Vec<CompiledKernel<T>>),
}

impl<T: KernelScalar> CompiledKernel<T> {
    pub(crate) fn from_spec(spec: &KernelSpec) -> Self {
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
            Self::Sum(terms) | Self::Product(terms) => terms.iter().map(Self::num_params).sum(),
        }
    }

    /// Writes flattened `θ` in depth-first, left-to-right leaf order.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if `out` is the wrong length
    /// or a custom leaf rejects the write.
    pub fn get_params(&self, out: &mut [f64]) -> Result<(), GprError> {
        crate::data::require_count(out.len(), self.num_params(), "kernel parameters")?;
        let mut offset = 0;
        self.write_params(out, &mut offset)
    }

    /// Replaces flattened `θ`. All leaves are updated or none are.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if `params` is the wrong
    /// length or a leaf rejects its slice.
    pub fn set_params(&mut self, params: &[f64]) -> Result<(), GprError> {
        crate::data::require_count(params.len(), self.num_params(), "kernel parameters")?;
        let mut next = self.clone();
        let mut offset = 0;
        next.apply_params(params, &mut offset)?;
        *self = next;
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
            return Err(GprError::InvalidHyperparameter {
                reason: format!("kernel parameter index {param_idx} is out of range"),
            });
        }
        let mut offset = 0;
        let mut leaf = 0;
        if self.locate_leaf(param_idx, &mut offset, &mut leaf) {
            Ok(leaf)
        } else {
            Err(GprError::InvalidHyperparameter {
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
            .ok_or(GprError::InvalidHyperparameter {
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

    /// Combines cached leaf Grams into `out` (sum / product tree).
    ///
    /// `scratch` must match `out`. Nested products may allocate one extra
    /// `n×n` buffer.
    pub(crate) fn needs_product_grad_scratch(&self) -> bool {
        match self {
            Self::Product(_) => true,
            Self::Sum(terms) => terms.iter().any(Self::needs_product_grad_scratch),
            _ => false,
        }
    }

    pub(crate) fn combine_from_leaf_grams(
        &self,
        grams: &[Mat<T>],
        mut out: MatMut<'_, T>,
        mut scratch: MatMut<'_, T>,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        if grams.len() != self.leaf_count() {
            return Err(GprError::InvalidHyperparameter {
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
        let mut index = 0;
        self.write_from_leaf_grams(grams, &mut index, out.as_mut(), scratch.as_mut(), uplo)?;
        if index != grams.len() {
            return Err(GprError::InvalidHyperparameter {
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
        uplo: Triangle,
    ) -> Result<(), GprError> {
        match self {
            Self::Sum(terms) => {
                fold_cached_leaves(terms, grams, index, dest, scratch, uplo, add_triangle)
            }
            Self::Product(terms) => {
                fold_cached_leaves(terms, grams, index, dest, scratch, uplo, mul_triangle)
            }
            _ => {
                if *index >= grams.len() {
                    return Err(GprError::InvalidHyperparameter {
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
        match self {
            Self::RbfArd(_) | Self::MaternArd(_) | Self::RationalQuadraticArd(_) => true,
            Self::Sum(terms) | Self::Product(terms) => terms.iter().any(Self::needs_ard_sq_diff),
            _ => false,
        }
    }

    fn write_params(&self, out: &mut [f64], offset: &mut usize) -> Result<(), GprError> {
        match self {
            Self::Rbf(leaf) => {
                out[*offset] = leaf.log_lengthscale();
                *offset += 1;
                Ok(())
            }
            Self::RbfArd(leaf) => {
                let n = leaf.num_params();
                out[*offset..*offset + n].copy_from_slice(leaf.log_lengthscales());
                *offset += n;
                Ok(())
            }
            Self::Matern(leaf) => {
                out[*offset] = leaf.log_lengthscale();
                *offset += 1;
                Ok(())
            }
            Self::MaternArd(leaf) => {
                let n = leaf.num_params();
                out[*offset..*offset + n].copy_from_slice(leaf.log_lengthscales());
                *offset += n;
                Ok(())
            }
            Self::Periodic(leaf) => {
                out[*offset] = leaf.log_lengthscale();
                out[*offset + 1] = leaf.log_period();
                *offset += 2;
                Ok(())
            }
            Self::RationalQuadratic(leaf) => {
                out[*offset] = leaf.log_lengthscale();
                out[*offset + 1] = leaf.log_alpha();
                *offset += 2;
                Ok(())
            }
            Self::RationalQuadraticArd(leaf) => {
                let n = leaf.lengthscales().num_params();
                out[*offset..*offset + n].copy_from_slice(leaf.log_lengthscales());
                out[*offset + n] = leaf.log_alpha();
                *offset += n + 1;
                Ok(())
            }
            Self::Constant(leaf) => {
                out[*offset] = leaf.log_constant();
                *offset += 1;
                Ok(())
            }
            Self::Linear(leaf) => {
                out[*offset] = leaf.log_variance();
                *offset += 1;
                Ok(())
            }
            Self::White(leaf) => {
                out[*offset] = leaf.log_variance();
                *offset += 1;
                Ok(())
            }
            Self::Custom(leaf) => leaf.write_params(out, offset),
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
            Self::Rbf(leaf) => {
                let n = leaf.num_params();
                leaf.set_params(&params[*offset..*offset + n])?;
                *offset += n;
                Ok(())
            }
            Self::RbfArd(leaf) => {
                let n = leaf.num_params();
                leaf.set_params(&params[*offset..*offset + n])?;
                *offset += n;
                Ok(())
            }
            Self::Matern(leaf) => {
                let n = leaf.num_params();
                leaf.set_params(&params[*offset..*offset + n])?;
                *offset += n;
                Ok(())
            }
            Self::MaternArd(leaf) => {
                let n = leaf.num_params();
                leaf.set_params(&params[*offset..*offset + n])?;
                *offset += n;
                Ok(())
            }
            Self::Periodic(leaf) => {
                let n = leaf.num_params();
                leaf.set_params(&params[*offset..*offset + n])?;
                *offset += n;
                Ok(())
            }
            Self::RationalQuadratic(leaf) => {
                let n = leaf.num_params();
                leaf.set_params(&params[*offset..*offset + n])?;
                *offset += n;
                Ok(())
            }
            Self::RationalQuadraticArd(leaf) => {
                let n = leaf.num_params();
                leaf.set_params(&params[*offset..*offset + n])?;
                *offset += n;
                Ok(())
            }
            Self::Constant(leaf) => {
                let n = leaf.num_params();
                leaf.set_params(&params[*offset..*offset + n])?;
                *offset += n;
                Ok(())
            }
            Self::Linear(leaf) => {
                let n = leaf.num_params();
                leaf.set_params(&params[*offset..*offset + n])?;
                *offset += n;
                Ok(())
            }
            Self::White(leaf) => {
                let n = leaf.num_params();
                leaf.set_params(&params[*offset..*offset + n])?;
                *offset += n;
                Ok(())
            }
            Self::Custom(leaf) => leaf.apply_params(params, offset),
            Self::Sum(terms) | Self::Product(terms) => {
                for term in terms {
                    term.apply_params(params, offset)?;
                }
                Ok(())
            }
        }
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
            | Self::Custom(_) => false,
            Self::Sum(terms) | Self::Product(terms) => {
                terms.len() > 1 || terms.iter().any(Self::needs_internal_scratch)
            }
        }
    }
}

fn flatten_sum<T: KernelScalar>(spec: &KernelSpec, out: &mut Vec<CompiledKernel<T>>) {
    match spec {
        KernelSpec::Sum(left, right) => {
            flatten_sum(left, out);
            flatten_sum(right, out);
        }
        other => out.push(CompiledKernel::<T>::from_spec(other)),
    }
}

fn flatten_product<T: KernelScalar>(spec: &KernelSpec, out: &mut Vec<CompiledKernel<T>>) {
    match spec {
        KernelSpec::Product(left, right) => {
            flatten_product(left, out);
            flatten_product(right, out);
        }
        other => out.push(CompiledKernel::<T>::from_spec(other)),
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

fn iso_needs_dist() -> GprError {
    GprError::UnsupportedKernelOperation {
        reason: "isotropic RBF evaluates from a squared-distance matrix".to_owned(),
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

fn split_terms<T: KernelScalar>(
    terms: &[CompiledKernel<T>],
) -> Result<(&CompiledKernel<T>, &[CompiledKernel<T>]), GprError> {
    terms
        .split_first()
        .ok_or(GprError::UnsupportedKernelOperation {
            reason: "sum/product has no terms".to_owned(),
        })
}

fn fold_cached_leaves<T: KernelScalar>(
    terms: &[CompiledKernel<T>],
    grams: &[Mat<T>],
    index: &mut usize,
    mut dest: MatMut<'_, T>,
    mut scratch: MatMut<'_, T>,
    uplo: Triangle,
    combine: fn(MatMut<'_, T>, MatRef<'_, T>, Triangle),
) -> Result<(), GprError> {
    let (first, rest) = split_terms(terms)?;
    first.write_from_leaf_grams(grams, index, dest.as_mut(), scratch.as_mut(), uplo)?;
    let n = dest.nrows();
    let mut extra = None;
    for term in rest {
        if term.needs_internal_scratch() {
            let buf = extra.get_or_insert_with(|| Mat::zeros(n, n));
            term.write_from_leaf_grams(grams, index, scratch.as_mut(), buf.as_mut(), uplo)?;
        } else {
            term.write_from_leaf_grams(grams, index, scratch.as_mut(), dest.as_mut(), uplo)?;
        }
        combine(dest.as_mut(), scratch.as_ref(), uplo);
    }
    Ok(())
}

fn copy_triangle<T: Copy>(mut dest: MatMut<'_, T>, src: MatRef<'_, T>, uplo: Triangle) {
    visit_triangle(dest.nrows(), uplo, |row, col| {
        dest[(row, col)] = src[(row, col)];
    });
}

fn add_triangle<T: Copy + std::ops::Add<Output = T>>(
    mut acc: MatMut<'_, T>,
    src: MatRef<'_, T>,
    uplo: Triangle,
) {
    visit_triangle(acc.nrows(), uplo, |row, col| {
        acc[(row, col)] = acc[(row, col)] + src[(row, col)];
    });
}

fn mul_triangle<T: Copy + std::ops::Mul<Output = T>>(
    mut acc: MatMut<'_, T>,
    src: MatRef<'_, T>,
    uplo: Triangle,
) {
    visit_triangle(acc.nrows(), uplo, |row, col| {
        acc[(row, col)] = acc[(row, col)] * src[(row, col)];
    });
}
