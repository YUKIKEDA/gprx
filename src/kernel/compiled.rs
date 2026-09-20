//! Execution-layer kernel: static dispatch over built-in leaves.

use super::{
    ConstantKernel, CustomKernel, LinearKernel, MaternArdKernel, MaternKernel, PeriodicKernel,
    RationalQuadraticArdKernel, RationalQuadraticKernel, RbfArdKernel, RbfKernel, Triangle,
    WhiteKernel, visit_triangle, write_square_from_coords,
};
use crate::error::GprError;
use crate::kernel::KernelSpec;
use faer::{Mat, MatMut, MatRef};

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
pub(crate) struct MixedKernelViews<'a> {
    pub(crate) dist: MatRef<'a, f64>,
    pub(crate) x: MatRef<'a, f64>,
    pub(crate) ard_cache: Option<MatRef<'a, f64>>,
}

impl<'a> MixedKernelViews<'a> {
    pub(crate) fn new(dist: MatRef<'a, f64>, x: MatRef<'a, f64>) -> Self {
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
pub enum CompiledKernel {
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
    Custom(CustomKernel),
    /// Flattened sum of compiled terms.
    Sum(Vec<CompiledKernel>),
    /// Flattened Hadamard product of compiled terms.
    Product(Vec<CompiledKernel>),
}

impl CompiledKernel {
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
            KernelSpec::Custom(leaf) => Self::Custom(leaf.clone()),
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
        require_len(out.len(), self.num_params())?;
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
        require_len(params.len(), self.num_params())?;
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
    pub(crate) fn combine_from_leaf_grams(
        &self,
        grams: &[Mat<f64>],
        mut out: MatMut<'_, f64>,
        mut scratch: MatMut<'_, f64>,
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
        grams: &[Mat<f64>],
        index: &mut usize,
        dest: MatMut<'_, f64>,
        scratch: MatMut<'_, f64>,
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

    /// Writes `k` into `out` for `uplo`. `scratch` must match `out`.
    ///
    /// Entries outside the requested triangle are left unchanged. `scratch`
    /// must be a distinct buffer from `out`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError`] if shapes mismatch, `scratch` is the wrong size, a
    /// leaf fails, or a sum/product has no terms.
    pub fn apply(
        &self,
        dist: MatRef<'_, f64>,
        mut out: MatMut<'_, f64>,
        uplo: Triangle,
        mut scratch: MatMut<'_, f64>,
    ) -> Result<(), GprError> {
        require_scratch_shape(out.as_ref(), scratch.as_ref())?;
        match self {
            Self::Rbf(leaf) => leaf.apply(dist, out, uplo),
            Self::RbfArd(_)
            | Self::Linear(_)
            | Self::MaternArd(_)
            | Self::RationalQuadraticArd(_) => Err(ard_needs_coords()),
            Self::Matern(leaf) => leaf.apply(dist, out, uplo),
            Self::Periodic(leaf) => leaf.apply(dist, out, uplo),
            Self::RationalQuadratic(leaf) => leaf.apply(dist, out, uplo),
            Self::Constant(leaf) => leaf.apply(dist, out, uplo),
            Self::White(leaf) => leaf.apply(dist, out, uplo),
            Self::Custom(leaf) => leaf.apply(dist, out, uplo),
            Self::Sum(terms) => fold_terms(
                terms,
                dist,
                out.as_mut(),
                uplo,
                scratch.as_mut(),
                add_triangle,
            ),
            Self::Product(terms) => fold_terms(
                terms,
                dist,
                out.as_mut(),
                uplo,
                scratch.as_mut(),
                mul_triangle,
            ),
        }
    }

    /// Writes rectangular `k(dist)` (train × test) into `out`.
    ///
    /// `scratch` must match `out` and be a distinct buffer.
    ///
    /// # Errors
    ///
    /// Returns the same shape errors as [`Self::apply`].
    pub fn apply_cross(
        &self,
        dist: MatRef<'_, f64>,
        mut out: MatMut<'_, f64>,
        mut scratch: MatMut<'_, f64>,
    ) -> Result<(), GprError> {
        require_scratch_shape(out.as_ref(), scratch.as_ref())?;
        match self {
            Self::Rbf(leaf) => leaf.apply_cross(dist, out),
            Self::RbfArd(_)
            | Self::Linear(_)
            | Self::MaternArd(_)
            | Self::RationalQuadraticArd(_) => Err(ard_needs_coords()),
            Self::Matern(leaf) => leaf.apply_cross(dist, out),
            Self::Periodic(leaf) => leaf.apply_cross(dist, out),
            Self::RationalQuadratic(leaf) => leaf.apply_cross(dist, out),
            Self::Constant(leaf) => leaf.apply_cross(dist, out),
            Self::White(leaf) => leaf.apply_cross(dist, out),
            Self::Custom(leaf) => leaf.apply_cross(dist, out),
            Self::Sum(terms) => fold_rect(terms, dist, out.as_mut(), scratch.as_mut(), add_rect),
            Self::Product(terms) => {
                fold_rect(terms, dist, out.as_mut(), scratch.as_mut(), mul_rect)
            }
        }
    }

    /// Writes the diagonal `k(x, x)` into `out`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::UnsupportedKernelOperation`] if a sum/product has no
    /// terms.
    pub fn fill_diag(&self, out: &mut [f64]) -> Result<(), GprError> {
        match self {
            Self::Rbf(leaf) => {
                leaf.fill_diag(out);
                Ok(())
            }
            Self::RbfArd(leaf) => {
                leaf.fill_diag(out);
                Ok(())
            }
            Self::Matern(leaf) => {
                leaf.fill_diag(out);
                Ok(())
            }
            Self::MaternArd(leaf) => {
                leaf.fill_diag(out);
                Ok(())
            }
            Self::Periodic(leaf) => {
                leaf.fill_diag(out);
                Ok(())
            }
            Self::RationalQuadratic(leaf) => {
                leaf.fill_diag(out);
                Ok(())
            }
            Self::RationalQuadraticArd(leaf) => {
                leaf.fill_diag(out);
                Ok(())
            }
            Self::Constant(leaf) => {
                leaf.fill_diag(out);
                Ok(())
            }
            Self::White(leaf) => {
                leaf.fill_diag(out);
                Ok(())
            }
            Self::Custom(leaf) => leaf.fill_diag(out),
            Self::Linear(_) => Err(GprError::UnsupportedKernelOperation {
                reason: "linear kernel diagonal needs coordinates".to_owned(),
            }),
            Self::Sum(terms) => {
                let (first, rest) = split_terms(terms)?;
                first.fill_diag(out)?;
                let mut tmp = vec![0.0; out.len()];
                for term in rest {
                    term.fill_diag(&mut tmp)?;
                    for (dst, src) in out.iter_mut().zip(&tmp) {
                        *dst += *src;
                    }
                }
                Ok(())
            }
            Self::Product(terms) => {
                let (first, rest) = split_terms(terms)?;
                first.fill_diag(out)?;
                let mut tmp = vec![0.0; out.len()];
                for term in rest {
                    term.fill_diag(&mut tmp)?;
                    for (dst, src) in out.iter_mut().zip(&tmp) {
                        *dst *= *src;
                    }
                }
                Ok(())
            }
        }
    }

    /// Writes the diagonal `k(x, x)` using coordinates when the leaf needs them.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::UnsupportedKernelOperation`] if a sum/product has no
    /// terms, or the same shape errors as the leaf.
    pub fn fill_diag_points(&self, x: MatRef<'_, f64>, out: &mut [f64]) -> Result<(), GprError> {
        match self {
            Self::Rbf(leaf) => {
                leaf.fill_diag(out);
                Ok(())
            }
            Self::RbfArd(leaf) => {
                leaf.fill_diag(out);
                Ok(())
            }
            Self::Matern(leaf) => {
                leaf.fill_diag(out);
                Ok(())
            }
            Self::MaternArd(leaf) => {
                leaf.fill_diag(out);
                Ok(())
            }
            Self::Periodic(leaf) => {
                leaf.fill_diag(out);
                Ok(())
            }
            Self::RationalQuadratic(leaf) => {
                leaf.fill_diag(out);
                Ok(())
            }
            Self::RationalQuadraticArd(leaf) => {
                leaf.fill_diag(out);
                Ok(())
            }
            Self::Constant(leaf) => {
                leaf.fill_diag(out);
                Ok(())
            }
            Self::White(leaf) => {
                leaf.fill_diag(out);
                Ok(())
            }
            Self::Custom(leaf) => leaf.fill_diag(out),
            Self::Linear(leaf) => leaf.fill_diag_points(x, out),
            Self::Sum(terms) => {
                let (first, rest) = split_terms(terms)?;
                first.fill_diag_points(x, out)?;
                let mut tmp = vec![0.0; out.len()];
                for term in rest {
                    term.fill_diag_points(x, &mut tmp)?;
                    for (dst, src) in out.iter_mut().zip(&tmp) {
                        *dst += *src;
                    }
                }
                Ok(())
            }
            Self::Product(terms) => {
                let (first, rest) = split_terms(terms)?;
                first.fill_diag_points(x, out)?;
                let mut tmp = vec![0.0; out.len()];
                for term in rest {
                    term.fill_diag_points(x, &mut tmp)?;
                    for (dst, src) in out.iter_mut().zip(&tmp) {
                        *dst *= *src;
                    }
                }
                Ok(())
            }
        }
    }

    /// Writes `∂K/∂θ_{param_idx}` into `d_k`.
    ///
    /// Product trees need `scratch` the same shape as `d_k` and distinct from
    /// it. Leaves ignore `scratch`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if `param_idx` is out of
    /// range, [`GprError::WorkspaceTooSmall`] if a product tree's `scratch` is
    /// the wrong size, or the same shape errors as [`Self::apply`].
    pub fn grad(
        &self,
        dist: MatRef<'_, f64>,
        mut d_k: MatMut<'_, f64>,
        param_idx: usize,
        uplo: Triangle,
        mut scratch: MatMut<'_, f64>,
    ) -> Result<(), GprError> {
        match self {
            Self::Rbf(leaf) => leaf.grad(dist, d_k, param_idx, uplo),
            Self::RbfArd(_)
            | Self::Linear(_)
            | Self::MaternArd(_)
            | Self::RationalQuadraticArd(_) => Err(ard_needs_coords()),
            Self::Matern(leaf) => leaf.grad(dist, d_k, param_idx, uplo),
            Self::Periodic(leaf) => leaf.grad(dist, d_k, param_idx, uplo),
            Self::RationalQuadratic(leaf) => leaf.grad(dist, d_k, param_idx, uplo),
            Self::Constant(leaf) => leaf.grad(dist, d_k, param_idx, uplo),
            Self::White(leaf) => leaf.grad(dist, d_k, param_idx, uplo),
            Self::Custom(leaf) => leaf.grad(dist, d_k, param_idx, uplo),
            Self::Sum(terms) => {
                let (term, local) = term_for_param(terms, param_idx)?;
                term.grad(dist, d_k, local, uplo, scratch)
            }
            Self::Product(terms) => {
                require_scratch_shape(d_k.as_ref(), scratch.as_ref())?;
                product_grad(terms, dist, d_k.as_mut(), param_idx, uplo, scratch.as_mut())
            }
        }
    }

    /// Writes `k` from point coordinates.
    ///
    /// Isotropic leaves compute `‖x_i-x_j‖²` from `X` on each pair. ARD and
    /// Linear evaluate from coordinates. Custom distance leaves fill the
    /// triangle with those distances, then apply in place.
    ///
    /// # Errors
    ///
    /// Same shape errors as [`Self::apply`].
    pub fn apply_points(
        &self,
        x: MatRef<'_, f64>,
        mut out: MatMut<'_, f64>,
        uplo: Triangle,
        mut scratch: MatMut<'_, f64>,
    ) -> Result<(), GprError> {
        require_scratch_shape(out.as_ref(), scratch.as_ref())?;
        match self {
            Self::Rbf(leaf) => leaf.apply_from_coords(x, out, uplo),
            Self::Matern(leaf) => leaf.apply_from_coords(x, out, uplo),
            Self::Periodic(leaf) => leaf.apply_from_coords(x, out, uplo),
            Self::RationalQuadratic(leaf) => leaf.apply_from_coords(x, out, uplo),
            Self::Custom(leaf) => apply_custom_from_coords(leaf, x, out, uplo),
            Self::RbfArd(leaf) => leaf.apply(x, out, uplo),
            Self::Linear(leaf) => leaf.apply(x, out, uplo),
            Self::MaternArd(leaf) => leaf.apply(x, out, uplo),
            Self::RationalQuadraticArd(leaf) => leaf.apply(x, out, uplo),
            Self::Constant(leaf) => leaf.apply_points(x, out, uplo),
            Self::White(leaf) => leaf.apply_points(x, out, uplo),
            Self::Sum(terms) => {
                fold_terms_points(terms, x, out.as_mut(), uplo, scratch.as_mut(), add_triangle)
            }
            Self::Product(terms) => {
                fold_terms_points(terms, x, out.as_mut(), uplo, scratch.as_mut(), mul_triangle)
            }
        }
    }

    /// Writes rectangular `k(x, xs)` from coordinates.
    ///
    /// # Errors
    ///
    /// Same as [`Self::apply_points`].
    pub fn apply_cross_points(
        &self,
        x: MatRef<'_, f64>,
        xs: MatRef<'_, f64>,
        mut out: MatMut<'_, f64>,
        mut scratch: MatMut<'_, f64>,
    ) -> Result<(), GprError> {
        require_scratch_shape(out.as_ref(), scratch.as_ref())?;
        match self {
            Self::Rbf(_)
            | Self::Matern(_)
            | Self::Periodic(_)
            | Self::RationalQuadratic(_)
            | Self::Custom(_) => Err(iso_needs_dist()),
            Self::RbfArd(leaf) => leaf.apply_cross(x, xs, out),
            Self::Linear(leaf) => leaf.apply_cross(x, xs, out),
            Self::MaternArd(leaf) => leaf.apply_cross(x, xs, out),
            Self::RationalQuadraticArd(leaf) => leaf.apply_cross(x, xs, out),
            Self::Constant(leaf) => leaf.apply_cross_points(x, xs, out),
            Self::White(leaf) => leaf.apply_cross_points(x, xs, out),
            Self::Sum(terms) => {
                fold_rect_points(terms, x, xs, out.as_mut(), scratch.as_mut(), add_rect)
            }
            Self::Product(terms) => {
                fold_rect_points(terms, x, xs, out.as_mut(), scratch.as_mut(), mul_rect)
            }
        }
    }

    /// Writes `∂K/∂θ_{param_idx}` from point coordinates.
    ///
    /// Product trees need `scratch` the same shape as `d_k` and distinct from
    /// it. Leaves ignore `scratch`.
    ///
    /// # Errors
    ///
    /// [`GprError::InvalidHyperparameter`] if `param_idx` is out of range,
    /// [`GprError::WorkspaceTooSmall`] if a product tree's `scratch` is the
    /// wrong size, or the same shape errors as [`Self::apply_points`].
    #[allow(clippy::only_used_in_recursion)] // leaves ignore scratch; Sum forwards it
    pub fn grad_points(
        &self,
        x: MatRef<'_, f64>,
        mut d_k: MatMut<'_, f64>,
        param_idx: usize,
        uplo: Triangle,
        mut scratch: MatMut<'_, f64>,
    ) -> Result<(), GprError> {
        match self {
            Self::Rbf(leaf) => leaf.grad_from_coords(x, d_k, param_idx, uplo),
            Self::Matern(leaf) => leaf.grad_from_coords(x, d_k, param_idx, uplo),
            Self::Periodic(leaf) => leaf.grad_from_coords(x, d_k, param_idx, uplo),
            Self::RationalQuadratic(leaf) => leaf.grad_from_coords(x, d_k, param_idx, uplo),
            Self::Custom(leaf) => grad_custom_from_coords(leaf, x, d_k, param_idx, uplo),
            Self::RbfArd(leaf) => leaf.grad(x, d_k, param_idx, uplo),
            Self::Linear(leaf) => leaf.grad(x, d_k, param_idx, uplo),
            Self::MaternArd(leaf) => leaf.grad(x, d_k, param_idx, uplo),
            Self::RationalQuadraticArd(leaf) => leaf.grad(x, d_k, param_idx, uplo),
            Self::Constant(leaf) => leaf.grad_points(x, d_k, param_idx, uplo),
            Self::White(leaf) => leaf.grad_points(x, d_k, param_idx, uplo),
            Self::Sum(terms) => {
                let (term, local) = term_for_param(terms, param_idx)?;
                term.grad_points(x, d_k, local, uplo, scratch)
            }
            Self::Product(terms) => {
                require_scratch_shape(d_k.as_ref(), scratch.as_ref())?;
                product_grad_points(terms, x, d_k.as_mut(), param_idx, uplo, scratch.as_mut())
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

    pub(crate) fn apply_from_ard_cache(
        &self,
        cache: MatRef<'_, f64>,
        x: MatRef<'_, f64>,
        mut out: MatMut<'_, f64>,
        uplo: Triangle,
        mut scratch: MatMut<'_, f64>,
    ) -> Result<(), GprError> {
        require_scratch_shape(out.as_ref(), scratch.as_ref())?;
        match self {
            Self::RbfArd(leaf) => leaf.apply_from_sq_diff(cache, out, uplo),
            Self::MaternArd(leaf) => leaf.apply_from_sq_diff(cache, out, uplo),
            Self::RationalQuadraticArd(leaf) => leaf.apply_from_sq_diff(cache, out, uplo),
            Self::Constant(leaf) => leaf.apply_points(x, out, uplo),
            Self::White(leaf) => leaf.apply_points(x, out, uplo),
            Self::Sum(terms) => {
                fold_terms_ard_cache(terms, cache, x, out.as_mut(), uplo, scratch.as_mut())
            }
            _ => self.apply_points(x, out, uplo, scratch),
        }
    }

    pub(crate) fn grad_from_ard_cache(
        &self,
        cache: MatRef<'_, f64>,
        x: MatRef<'_, f64>,
        d_k: MatMut<'_, f64>,
        param_idx: usize,
        uplo: Triangle,
        scratch: MatMut<'_, f64>,
    ) -> Result<(), GprError> {
        match self {
            Self::RbfArd(leaf) => leaf.grad_from_sq_diff(cache, d_k, param_idx, uplo),
            Self::MaternArd(leaf) => leaf.grad_from_sq_diff(cache, d_k, param_idx, uplo),
            Self::RationalQuadraticArd(leaf) => leaf.grad_from_sq_diff(cache, d_k, param_idx, uplo),
            Self::Constant(leaf) => leaf.grad_points(x, d_k, param_idx, uplo),
            Self::White(leaf) => leaf.grad_points(x, d_k, param_idx, uplo),
            Self::Sum(terms) => {
                let (term, local) = term_for_param(terms, param_idx)?;
                term.grad_from_ard_cache(cache, x, d_k, local, uplo, scratch)
            }
            _ => self.grad_points(x, d_k, param_idx, uplo, scratch),
        }
    }

    /// Writes `K` from a distance matrix and coordinates, one mode per leaf.
    pub(crate) fn apply_mixed(
        &self,
        views: MixedKernelViews<'_>,
        mut out: MatMut<'_, f64>,
        uplo: Triangle,
        mut scratch: MatMut<'_, f64>,
    ) -> Result<(), GprError> {
        require_scratch_shape(out.as_ref(), scratch.as_ref())?;
        match self {
            Self::Rbf(_)
            | Self::Matern(_)
            | Self::Periodic(_)
            | Self::RationalQuadratic(_)
            | Self::Custom(_)
            | Self::Constant(_)
            | Self::White(_) => self.apply(views.dist, out, uplo, scratch),
            Self::RbfArd(_)
            | Self::Linear(_)
            | Self::MaternArd(_)
            | Self::RationalQuadraticArd(_) => {
                if let Some(cache) = views.ard_cache.filter(|_| self.needs_ard_sq_diff()) {
                    self.apply_from_ard_cache(cache, views.x, out, uplo, scratch)
                } else {
                    self.apply_points(views.x, out, uplo, scratch)
                }
            }
            Self::Sum(terms) => fold_terms_mixed(
                terms,
                views,
                out.as_mut(),
                uplo,
                scratch.as_mut(),
                add_triangle,
            ),
            Self::Product(terms) => fold_terms_mixed(
                terms,
                views,
                out.as_mut(),
                uplo,
                scratch.as_mut(),
                mul_triangle,
            ),
        }
    }

    /// Writes rectangular `k` from train–query distances and coordinates.
    pub(crate) fn apply_cross_mixed(
        &self,
        dist: MatRef<'_, f64>,
        x: MatRef<'_, f64>,
        xs: MatRef<'_, f64>,
        mut out: MatMut<'_, f64>,
        mut scratch: MatMut<'_, f64>,
    ) -> Result<(), GprError> {
        require_scratch_shape(out.as_ref(), scratch.as_ref())?;
        match self {
            Self::Rbf(_)
            | Self::Matern(_)
            | Self::Periodic(_)
            | Self::RationalQuadratic(_)
            | Self::Custom(_)
            | Self::Constant(_)
            | Self::White(_) => self.apply_cross(dist, out, scratch),
            Self::RbfArd(_)
            | Self::Linear(_)
            | Self::MaternArd(_)
            | Self::RationalQuadraticArd(_) => self.apply_cross_points(x, xs, out, scratch),
            Self::Sum(terms) => {
                fold_rect_mixed(terms, dist, x, xs, out.as_mut(), scratch.as_mut(), add_rect)
            }
            Self::Product(terms) => {
                fold_rect_mixed(terms, dist, x, xs, out.as_mut(), scratch.as_mut(), mul_rect)
            }
        }
    }

    /// Writes `∂K/∂θ` from a distance matrix and coordinates, one mode per leaf.
    pub(crate) fn grad_mixed(
        &self,
        views: MixedKernelViews<'_>,
        mut d_k: MatMut<'_, f64>,
        param_idx: usize,
        uplo: Triangle,
        mut scratch: MatMut<'_, f64>,
    ) -> Result<(), GprError> {
        match self {
            Self::Rbf(_)
            | Self::Matern(_)
            | Self::Periodic(_)
            | Self::RationalQuadratic(_)
            | Self::Custom(_)
            | Self::Constant(_)
            | Self::White(_) => self.grad(views.dist, d_k, param_idx, uplo, scratch),
            Self::RbfArd(_)
            | Self::Linear(_)
            | Self::MaternArd(_)
            | Self::RationalQuadraticArd(_) => {
                if let Some(cache) = views.ard_cache.filter(|_| self.needs_ard_sq_diff()) {
                    self.grad_from_ard_cache(cache, views.x, d_k, param_idx, uplo, scratch)
                } else {
                    self.grad_points(views.x, d_k, param_idx, uplo, scratch)
                }
            }
            Self::Sum(terms) => {
                let (term, local) = term_for_param(terms, param_idx)?;
                term.grad_mixed(views, d_k, local, uplo, scratch)
            }
            Self::Product(terms) => {
                require_scratch_shape(d_k.as_ref(), scratch.as_ref())?;
                product_grad_mixed(
                    terms,
                    views,
                    d_k.as_mut(),
                    param_idx,
                    uplo,
                    scratch.as_mut(),
                )
            }
        }
    }

    /// Writes `∂²K/∂θ_i ∂θ_j` from squared distances into `d2_k`.
    ///
    /// Product trees need `scratch` the same shape as `d2_k`. Leaves ignore it.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if `i` or `j` is out of
    /// range, [`GprError::WorkspaceTooSmall`] if a product tree's `scratch` is
    /// the wrong size, or the same shape errors as [`Self::apply`].
    pub fn hess(
        &self,
        dist: MatRef<'_, f64>,
        mut d2_k: MatMut<'_, f64>,
        i: usize,
        j: usize,
        uplo: Triangle,
        mut scratch: MatMut<'_, f64>,
    ) -> Result<(), GprError> {
        match self {
            Self::Rbf(leaf) => leaf.hess(dist, d2_k, i, j, uplo),
            Self::RbfArd(_)
            | Self::Linear(_)
            | Self::MaternArd(_)
            | Self::RationalQuadraticArd(_) => Err(ard_needs_coords()),
            Self::Matern(leaf) => leaf.hess(dist, d2_k, i, j, uplo),
            Self::Periodic(leaf) => leaf.hess(dist, d2_k, i, j, uplo),
            Self::RationalQuadratic(leaf) => leaf.hess(dist, d2_k, i, j, uplo),
            Self::Constant(leaf) => leaf.hess(dist, d2_k, i, j, uplo),
            Self::White(leaf) => leaf.hess(dist, d2_k, i, j, uplo),
            Self::Custom(leaf) => leaf.hess(dist, d2_k, i, j, uplo),
            Self::Sum(terms) => match owners_for_pair(terms, i, j)? {
                PairOwners::Same {
                    term,
                    local_i,
                    local_j,
                } => term.hess(dist, d2_k, local_i, local_j, uplo, scratch),
                PairOwners::Distinct { .. } => {
                    zero_triangle(d2_k.as_mut(), uplo);
                    Ok(())
                }
            },
            Self::Product(terms) => {
                require_scratch_shape(d2_k.as_ref(), scratch.as_ref())?;
                product_hess(terms, dist, d2_k.as_mut(), i, j, uplo, scratch.as_mut())
            }
        }
    }

    /// Writes `∂²K/∂θ_i ∂θ_j` from point coordinates.
    ///
    /// # Errors
    ///
    /// Same as [`Self::hess`], with coordinates in place of distances.
    #[allow(clippy::only_used_in_recursion)]
    pub fn hess_points(
        &self,
        x: MatRef<'_, f64>,
        mut d2_k: MatMut<'_, f64>,
        i: usize,
        j: usize,
        uplo: Triangle,
        mut scratch: MatMut<'_, f64>,
    ) -> Result<(), GprError> {
        match self {
            Self::Rbf(leaf) => leaf.hess_from_coords(x, d2_k, i, j, uplo),
            Self::Matern(leaf) => leaf.hess_from_coords(x, d2_k, i, j, uplo),
            Self::Periodic(leaf) => leaf.hess_from_coords(x, d2_k, i, j, uplo),
            Self::RationalQuadratic(leaf) => leaf.hess_from_coords(x, d2_k, i, j, uplo),
            Self::Custom(leaf) => leaf.hess_points(x, d2_k, i, j, uplo),
            Self::RbfArd(leaf) => leaf.hess(x, d2_k, i, j, uplo),
            Self::Linear(leaf) => leaf.hess(x, d2_k, i, j, uplo),
            Self::MaternArd(leaf) => leaf.hess(x, d2_k, i, j, uplo),
            Self::RationalQuadraticArd(leaf) => leaf.hess(x, d2_k, i, j, uplo),
            Self::Constant(leaf) => leaf.hess_points(x, d2_k, i, j, uplo),
            Self::White(leaf) => leaf.hess_points(x, d2_k, i, j, uplo),
            Self::Sum(terms) => match owners_for_pair(terms, i, j)? {
                PairOwners::Same {
                    term,
                    local_i,
                    local_j,
                } => term.hess_points(x, d2_k, local_i, local_j, uplo, scratch),
                PairOwners::Distinct { .. } => {
                    zero_triangle(d2_k.as_mut(), uplo);
                    Ok(())
                }
            },
            Self::Product(terms) => {
                require_scratch_shape(d2_k.as_ref(), scratch.as_ref())?;
                product_hess_points(terms, x, d2_k.as_mut(), i, j, uplo, scratch.as_mut())
            }
        }
    }

    pub(crate) fn hess_from_ard_cache(
        &self,
        cache: MatRef<'_, f64>,
        x: MatRef<'_, f64>,
        d2_k: MatMut<'_, f64>,
        pair: (usize, usize),
        uplo: Triangle,
        scratch: MatMut<'_, f64>,
    ) -> Result<(), GprError> {
        let (i, j) = pair;
        match self {
            Self::RbfArd(leaf) => leaf.hess_from_sq_diff(cache, d2_k, i, j, uplo),
            Self::MaternArd(leaf) => leaf.hess_from_sq_diff(cache, d2_k, i, j, uplo),
            Self::RationalQuadraticArd(leaf) => leaf.hess_from_sq_diff(cache, d2_k, i, j, uplo),
            Self::Constant(leaf) => leaf.hess_points(x, d2_k, i, j, uplo),
            Self::White(leaf) => leaf.hess_points(x, d2_k, i, j, uplo),
            Self::Sum(terms) => match owners_for_pair(terms, i, j)? {
                PairOwners::Same {
                    term,
                    local_i,
                    local_j,
                } => term.hess_from_ard_cache(cache, x, d2_k, (local_i, local_j), uplo, scratch),
                PairOwners::Distinct { .. } => {
                    zero_triangle(d2_k, uplo);
                    Ok(())
                }
            },
            _ => self.hess_points(x, d2_k, i, j, uplo, scratch),
        }
    }

    pub(crate) fn hess_mixed(
        &self,
        views: MixedKernelViews<'_>,
        mut d2_k: MatMut<'_, f64>,
        i: usize,
        j: usize,
        uplo: Triangle,
        mut scratch: MatMut<'_, f64>,
    ) -> Result<(), GprError> {
        match self {
            Self::Rbf(_)
            | Self::Matern(_)
            | Self::Periodic(_)
            | Self::RationalQuadratic(_)
            | Self::Custom(_)
            | Self::Constant(_)
            | Self::White(_) => self.hess(views.dist, d2_k, i, j, uplo, scratch),
            Self::RbfArd(_)
            | Self::Linear(_)
            | Self::MaternArd(_)
            | Self::RationalQuadraticArd(_) => {
                if let Some(cache) = views.ard_cache.filter(|_| self.needs_ard_sq_diff()) {
                    self.hess_from_ard_cache(cache, views.x, d2_k, (i, j), uplo, scratch)
                } else {
                    self.hess_points(views.x, d2_k, i, j, uplo, scratch)
                }
            }
            Self::Sum(terms) => match owners_for_pair(terms, i, j)? {
                PairOwners::Same {
                    term,
                    local_i,
                    local_j,
                } => term.hess_mixed(views, d2_k, local_i, local_j, uplo, scratch),
                PairOwners::Distinct { .. } => {
                    zero_triangle(d2_k.as_mut(), uplo);
                    Ok(())
                }
            },
            Self::Product(terms) => {
                require_scratch_shape(d2_k.as_ref(), scratch.as_ref())?;
                product_hess_mixed(terms, views, d2_k.as_mut(), i, j, uplo, scratch.as_mut())
            }
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

    pub(crate) fn needs_product_grad_scratch(&self) -> bool {
        match self {
            Self::Product(_) => true,
            Self::Sum(terms) => terms.iter().any(Self::needs_product_grad_scratch),
            _ => false,
        }
    }
}

fn flatten_sum(spec: &KernelSpec, out: &mut Vec<CompiledKernel>) {
    match spec {
        KernelSpec::Sum(left, right) => {
            flatten_sum(left, out);
            flatten_sum(right, out);
        }
        other => out.push(CompiledKernel::from_spec(other)),
    }
}

fn flatten_product(spec: &KernelSpec, out: &mut Vec<CompiledKernel>) {
    match spec {
        KernelSpec::Product(left, right) => {
            flatten_product(left, out);
            flatten_product(right, out);
        }
        other => out.push(CompiledKernel::from_spec(other)),
    }
}

fn require_len(actual: usize, expected: usize) -> Result<(), GprError> {
    if actual == expected {
        Ok(())
    } else {
        Err(GprError::InvalidHyperparameter {
            reason: format!("expected {expected} kernel parameters, got {actual}"),
        })
    }
}

fn require_scratch_shape(out: MatRef<'_, f64>, scratch: MatRef<'_, f64>) -> Result<(), GprError> {
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

fn apply_custom_from_coords(
    leaf: &CustomKernel,
    x: MatRef<'_, f64>,
    mut out: MatMut<'_, f64>,
    uplo: Triangle,
) -> Result<(), GprError> {
    let n = out.nrows();
    let mut dist = Mat::zeros(n, n);
    write_square_from_coords(x, dist.as_mut(), uplo, Ok)?;
    leaf.apply(dist.as_ref(), out.as_mut(), uplo)
}

fn grad_custom_from_coords(
    leaf: &CustomKernel,
    x: MatRef<'_, f64>,
    mut d_k: MatMut<'_, f64>,
    param_idx: usize,
    uplo: Triangle,
) -> Result<(), GprError> {
    let n = d_k.nrows();
    let mut dist = Mat::zeros(n, n);
    write_square_from_coords(x, dist.as_mut(), uplo, Ok)?;
    leaf.grad(dist.as_ref(), d_k.as_mut(), param_idx, uplo)
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

fn split_terms(terms: &[CompiledKernel]) -> Result<(&CompiledKernel, &[CompiledKernel]), GprError> {
    terms
        .split_first()
        .ok_or(GprError::UnsupportedKernelOperation {
            reason: "sum/product has no terms".to_owned(),
        })
}

fn term_for_param(
    terms: &[CompiledKernel],
    param_idx: usize,
) -> Result<(&CompiledKernel, usize), GprError> {
    let mut offset = 0;
    for term in terms {
        let n = term.num_params();
        if param_idx < offset + n {
            return Ok((term, param_idx - offset));
        }
        offset += n;
    }
    Err(GprError::InvalidHyperparameter {
        reason: format!("kernel parameter index {param_idx} is out of range"),
    })
}

fn fold_cached_leaves(
    terms: &[CompiledKernel],
    grams: &[Mat<f64>],
    index: &mut usize,
    mut dest: MatMut<'_, f64>,
    mut scratch: MatMut<'_, f64>,
    uplo: Triangle,
    combine: fn(MatMut<'_, f64>, MatRef<'_, f64>, Triangle),
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

fn copy_triangle(mut dest: MatMut<'_, f64>, src: MatRef<'_, f64>, uplo: Triangle) {
    visit_triangle(dest.nrows(), uplo, |row, col| {
        dest[(row, col)] = src[(row, col)];
    });
}

fn add_triangle(mut acc: MatMut<'_, f64>, src: MatRef<'_, f64>, uplo: Triangle) {
    visit_triangle(acc.nrows(), uplo, |row, col| {
        acc[(row, col)] += src[(row, col)];
    });
}

fn mul_triangle(mut acc: MatMut<'_, f64>, src: MatRef<'_, f64>, uplo: Triangle) {
    visit_triangle(acc.nrows(), uplo, |row, col| {
        acc[(row, col)] *= src[(row, col)];
    });
}

fn fold_terms(
    terms: &[CompiledKernel],
    dist: MatRef<'_, f64>,
    mut out: MatMut<'_, f64>,
    uplo: Triangle,
    mut scratch: MatMut<'_, f64>,
    combine: fn(MatMut<'_, f64>, MatRef<'_, f64>, Triangle),
) -> Result<(), GprError> {
    let (first, rest) = split_terms(terms)?;
    first.apply(dist, out.as_mut(), uplo, scratch.as_mut())?;
    let n = out.nrows();
    let mut extra = None;
    for term in rest {
        apply_into(
            term,
            dist,
            scratch.as_mut(),
            out.as_mut(),
            uplo,
            &mut extra,
            n,
        )?;
        combine(out.as_mut(), scratch.as_ref(), uplo);
    }
    Ok(())
}

fn apply_into(
    term: &CompiledKernel,
    dist: MatRef<'_, f64>,
    dest: MatMut<'_, f64>,
    fallback_scratch: MatMut<'_, f64>,
    uplo: Triangle,
    extra: &mut Option<Mat<f64>>,
    n: usize,
) -> Result<(), GprError> {
    if term.needs_internal_scratch() {
        let buf = extra.get_or_insert_with(|| Mat::zeros(n, n));
        term.apply(dist, dest, uplo, buf.as_mut())
    } else {
        term.apply(dist, dest, uplo, fallback_scratch)
    }
}

fn fold_terms_points(
    terms: &[CompiledKernel],
    x: MatRef<'_, f64>,
    mut out: MatMut<'_, f64>,
    uplo: Triangle,
    mut scratch: MatMut<'_, f64>,
    combine: fn(MatMut<'_, f64>, MatRef<'_, f64>, Triangle),
) -> Result<(), GprError> {
    let (first, rest) = split_terms(terms)?;
    first.apply_points(x, out.as_mut(), uplo, scratch.as_mut())?;
    let n = out.nrows();
    let mut extra = None;
    for term in rest {
        apply_into_points(term, x, scratch.as_mut(), out.as_mut(), uplo, &mut extra, n)?;
        combine(out.as_mut(), scratch.as_ref(), uplo);
    }
    Ok(())
}

fn apply_into_points(
    term: &CompiledKernel,
    x: MatRef<'_, f64>,
    dest: MatMut<'_, f64>,
    fallback_scratch: MatMut<'_, f64>,
    uplo: Triangle,
    extra: &mut Option<Mat<f64>>,
    n: usize,
) -> Result<(), GprError> {
    if term.needs_internal_scratch() {
        let buf = extra.get_or_insert_with(|| Mat::zeros(n, n));
        term.apply_points(x, dest, uplo, buf.as_mut())
    } else {
        term.apply_points(x, dest, uplo, fallback_scratch)
    }
}

fn fold_terms_ard_cache(
    terms: &[CompiledKernel],
    cache: MatRef<'_, f64>,
    x: MatRef<'_, f64>,
    mut out: MatMut<'_, f64>,
    uplo: Triangle,
    mut scratch: MatMut<'_, f64>,
) -> Result<(), GprError> {
    let (first, rest) = split_terms(terms)?;
    first.apply_from_ard_cache(cache, x, out.as_mut(), uplo, scratch.as_mut())?;
    let n = out.nrows();
    let mut extra = None;
    for term in rest {
        if term.needs_internal_scratch() {
            let buf = extra.get_or_insert_with(|| Mat::zeros(n, n));
            term.apply_from_ard_cache(cache, x, scratch.as_mut(), uplo, buf.as_mut())?;
        } else {
            term.apply_from_ard_cache(cache, x, scratch.as_mut(), uplo, out.as_mut())?;
        }
        add_triangle(out.as_mut(), scratch.as_ref(), uplo);
    }
    Ok(())
}

fn add_rect(mut acc: MatMut<'_, f64>, src: MatRef<'_, f64>) {
    for col in 0..acc.ncols() {
        for row in 0..acc.nrows() {
            acc[(row, col)] += src[(row, col)];
        }
    }
}

fn mul_rect(mut acc: MatMut<'_, f64>, src: MatRef<'_, f64>) {
    for col in 0..acc.ncols() {
        for row in 0..acc.nrows() {
            acc[(row, col)] *= src[(row, col)];
        }
    }
}

fn fold_rect(
    terms: &[CompiledKernel],
    dist: MatRef<'_, f64>,
    mut out: MatMut<'_, f64>,
    mut scratch: MatMut<'_, f64>,
    combine: fn(MatMut<'_, f64>, MatRef<'_, f64>),
) -> Result<(), GprError> {
    let (first, rest) = split_terms(terms)?;
    first.apply_cross(dist, out.as_mut(), scratch.as_mut())?;
    let nrows = out.nrows();
    let ncols = out.ncols();
    let mut extra = None;
    for term in rest {
        apply_into_cross(
            term,
            dist,
            scratch.as_mut(),
            out.as_mut(),
            &mut extra,
            nrows,
            ncols,
        )?;
        combine(out.as_mut(), scratch.as_ref());
    }
    Ok(())
}

fn apply_into_cross(
    term: &CompiledKernel,
    dist: MatRef<'_, f64>,
    dest: MatMut<'_, f64>,
    fallback_scratch: MatMut<'_, f64>,
    extra: &mut Option<Mat<f64>>,
    nrows: usize,
    ncols: usize,
) -> Result<(), GprError> {
    if term.needs_internal_scratch() {
        let buf = extra.get_or_insert_with(|| Mat::zeros(nrows, ncols));
        term.apply_cross(dist, dest, buf.as_mut())
    } else {
        term.apply_cross(dist, dest, fallback_scratch)
    }
}

fn fold_rect_points(
    terms: &[CompiledKernel],
    x: MatRef<'_, f64>,
    xs: MatRef<'_, f64>,
    mut out: MatMut<'_, f64>,
    mut scratch: MatMut<'_, f64>,
    combine: fn(MatMut<'_, f64>, MatRef<'_, f64>),
) -> Result<(), GprError> {
    let (first, rest) = split_terms(terms)?;
    first.apply_cross_points(x, xs, out.as_mut(), scratch.as_mut())?;
    let mut extra = None;
    for term in rest {
        apply_into_cross_points(term, x, xs, scratch.as_mut(), out.as_mut(), &mut extra)?;
        combine(out.as_mut(), scratch.as_ref());
    }
    Ok(())
}

fn apply_into_cross_points(
    term: &CompiledKernel,
    x: MatRef<'_, f64>,
    xs: MatRef<'_, f64>,
    dest: MatMut<'_, f64>,
    fallback_scratch: MatMut<'_, f64>,
    extra: &mut Option<Mat<f64>>,
) -> Result<(), GprError> {
    if term.needs_internal_scratch() {
        let buf = extra.get_or_insert_with(|| Mat::zeros(dest.nrows(), dest.ncols()));
        term.apply_cross_points(x, xs, dest, buf.as_mut())
    } else {
        term.apply_cross_points(x, xs, dest, fallback_scratch)
    }
}

fn product_grad(
    terms: &[CompiledKernel],
    dist: MatRef<'_, f64>,
    mut d_k: MatMut<'_, f64>,
    param_idx: usize,
    uplo: Triangle,
    mut scratch: MatMut<'_, f64>,
) -> Result<(), GprError> {
    let mut offset = 0;
    let mut owner = None;
    for (i, term) in terms.iter().enumerate() {
        let n = term.num_params();
        if param_idx < offset + n {
            owner = Some((i, param_idx - offset));
            break;
        }
        offset += n;
    }
    let (owner_i, local) = owner.ok_or_else(|| GprError::InvalidHyperparameter {
        reason: format!("kernel parameter index {param_idx} is out of range"),
    })?;

    let n = d_k.nrows();
    let mut extra = None;
    let mut started = false;
    for (j, term) in terms.iter().enumerate() {
        if j == owner_i {
            continue;
        }
        if !started {
            term.apply(dist, d_k.as_mut(), uplo, scratch.as_mut())?;
            started = true;
        } else {
            apply_into(
                term,
                dist,
                scratch.as_mut(),
                d_k.as_mut(),
                uplo,
                &mut extra,
                n,
            )?;
            mul_triangle(d_k.as_mut(), scratch.as_ref(), uplo);
        }
    }
    if started {
        if terms[owner_i].needs_internal_scratch() {
            let buf = extra.get_or_insert_with(|| Mat::zeros(n, n));
            terms[owner_i].grad(dist, scratch.as_mut(), local, uplo, buf.as_mut())?;
        } else {
            terms[owner_i].grad(dist, scratch.as_mut(), local, uplo, d_k.as_mut())?;
        }
        mul_triangle(d_k.as_mut(), scratch.as_ref(), uplo);
    } else {
        terms[owner_i].grad(dist, d_k.as_mut(), local, uplo, scratch.as_mut())?;
    }
    Ok(())
}

fn product_grad_points(
    terms: &[CompiledKernel],
    x: MatRef<'_, f64>,
    mut d_k: MatMut<'_, f64>,
    param_idx: usize,
    uplo: Triangle,
    mut scratch: MatMut<'_, f64>,
) -> Result<(), GprError> {
    let mut offset = 0;
    let mut owner = None;
    for (i, term) in terms.iter().enumerate() {
        let n = term.num_params();
        if param_idx < offset + n {
            owner = Some((i, param_idx - offset));
            break;
        }
        offset += n;
    }
    let (owner_i, local) = owner.ok_or_else(|| GprError::InvalidHyperparameter {
        reason: format!("kernel parameter index {param_idx} is out of range"),
    })?;

    let n = d_k.nrows();
    let mut extra = None;
    let mut started = false;
    for (j, term) in terms.iter().enumerate() {
        if j == owner_i {
            continue;
        }
        if !started {
            term.apply_points(x, d_k.as_mut(), uplo, scratch.as_mut())?;
            started = true;
        } else {
            apply_into_points(term, x, scratch.as_mut(), d_k.as_mut(), uplo, &mut extra, n)?;
            mul_triangle(d_k.as_mut(), scratch.as_ref(), uplo);
        }
    }
    if started {
        if terms[owner_i].needs_internal_scratch() {
            let buf = extra.get_or_insert_with(|| Mat::zeros(n, n));
            terms[owner_i].grad_points(x, scratch.as_mut(), local, uplo, buf.as_mut())?;
        } else {
            terms[owner_i].grad_points(x, scratch.as_mut(), local, uplo, d_k.as_mut())?;
        }
        mul_triangle(d_k.as_mut(), scratch.as_ref(), uplo);
    } else {
        terms[owner_i].grad_points(x, d_k.as_mut(), local, uplo, scratch.as_mut())?;
    }
    Ok(())
}

fn fold_terms_mixed(
    terms: &[CompiledKernel],
    views: MixedKernelViews<'_>,
    mut out: MatMut<'_, f64>,
    uplo: Triangle,
    mut scratch: MatMut<'_, f64>,
    combine: fn(MatMut<'_, f64>, MatRef<'_, f64>, Triangle),
) -> Result<(), GprError> {
    let (first, rest) = split_terms(terms)?;
    first.apply_mixed(views, out.as_mut(), uplo, scratch.as_mut())?;
    let n = out.nrows();
    let mut extra = None;
    for term in rest {
        apply_into_mixed(
            term,
            views,
            scratch.as_mut(),
            out.as_mut(),
            uplo,
            &mut extra,
            n,
        )?;
        combine(out.as_mut(), scratch.as_ref(), uplo);
    }
    Ok(())
}

fn apply_into_mixed(
    term: &CompiledKernel,
    views: MixedKernelViews<'_>,
    dest: MatMut<'_, f64>,
    fallback_scratch: MatMut<'_, f64>,
    uplo: Triangle,
    extra: &mut Option<Mat<f64>>,
    n: usize,
) -> Result<(), GprError> {
    if term.needs_internal_scratch() {
        let buf = extra.get_or_insert_with(|| Mat::zeros(n, n));
        term.apply_mixed(views, dest, uplo, buf.as_mut())
    } else {
        term.apply_mixed(views, dest, uplo, fallback_scratch)
    }
}

fn fold_rect_mixed(
    terms: &[CompiledKernel],
    dist: MatRef<'_, f64>,
    x: MatRef<'_, f64>,
    xs: MatRef<'_, f64>,
    mut out: MatMut<'_, f64>,
    mut scratch: MatMut<'_, f64>,
    combine: fn(MatMut<'_, f64>, MatRef<'_, f64>),
) -> Result<(), GprError> {
    let (first, rest) = split_terms(terms)?;
    first.apply_cross_mixed(dist, x, xs, out.as_mut(), scratch.as_mut())?;
    let mut extra = None;
    for term in rest {
        apply_into_cross_mixed(
            term,
            dist,
            x,
            xs,
            scratch.as_mut(),
            out.as_mut(),
            &mut extra,
        )?;
        combine(out.as_mut(), scratch.as_ref());
    }
    Ok(())
}

fn apply_into_cross_mixed(
    term: &CompiledKernel,
    dist: MatRef<'_, f64>,
    x: MatRef<'_, f64>,
    xs: MatRef<'_, f64>,
    dest: MatMut<'_, f64>,
    fallback_scratch: MatMut<'_, f64>,
    extra: &mut Option<Mat<f64>>,
) -> Result<(), GprError> {
    if term.needs_internal_scratch() {
        let buf = extra.get_or_insert_with(|| Mat::zeros(dest.nrows(), dest.ncols()));
        term.apply_cross_mixed(dist, x, xs, dest, buf.as_mut())
    } else {
        term.apply_cross_mixed(dist, x, xs, dest, fallback_scratch)
    }
}

fn product_grad_mixed(
    terms: &[CompiledKernel],
    views: MixedKernelViews<'_>,
    mut d_k: MatMut<'_, f64>,
    param_idx: usize,
    uplo: Triangle,
    mut scratch: MatMut<'_, f64>,
) -> Result<(), GprError> {
    let mut offset = 0;
    let mut owner = None;
    for (i, term) in terms.iter().enumerate() {
        let n = term.num_params();
        if param_idx < offset + n {
            owner = Some((i, param_idx - offset));
            break;
        }
        offset += n;
    }
    let (owner_i, local) = owner.ok_or_else(|| GprError::InvalidHyperparameter {
        reason: format!("kernel parameter index {param_idx} is out of range"),
    })?;

    let n = d_k.nrows();
    let mut extra = None;
    let mut started = false;
    for (j, term) in terms.iter().enumerate() {
        if j == owner_i {
            continue;
        }
        if !started {
            term.apply_mixed(views, d_k.as_mut(), uplo, scratch.as_mut())?;
            started = true;
        } else {
            apply_into_mixed(
                term,
                views,
                scratch.as_mut(),
                d_k.as_mut(),
                uplo,
                &mut extra,
                n,
            )?;
            mul_triangle(d_k.as_mut(), scratch.as_ref(), uplo);
        }
    }
    if started {
        if terms[owner_i].needs_internal_scratch() {
            let buf = extra.get_or_insert_with(|| Mat::zeros(n, n));
            terms[owner_i].grad_mixed(views, scratch.as_mut(), local, uplo, buf.as_mut())?;
        } else {
            terms[owner_i].grad_mixed(views, scratch.as_mut(), local, uplo, d_k.as_mut())?;
        }
        mul_triangle(d_k.as_mut(), scratch.as_ref(), uplo);
    } else {
        terms[owner_i].grad_mixed(views, d_k.as_mut(), local, uplo, scratch.as_mut())?;
    }
    Ok(())
}

enum PairOwners<'a> {
    Same {
        term: &'a CompiledKernel,
        local_i: usize,
        local_j: usize,
    },
    Distinct {
        owner_i: usize,
        local_i: usize,
        owner_j: usize,
        local_j: usize,
    },
}

fn owners_for_pair(
    terms: &[CompiledKernel],
    i: usize,
    j: usize,
) -> Result<PairOwners<'_>, GprError> {
    let (owner_i, local_i) = term_index_for_param(terms, i)?;
    let (owner_j, local_j) = term_index_for_param(terms, j)?;
    if owner_i == owner_j {
        Ok(PairOwners::Same {
            term: &terms[owner_i],
            local_i,
            local_j,
        })
    } else {
        Ok(PairOwners::Distinct {
            owner_i,
            local_i,
            owner_j,
            local_j,
        })
    }
}

fn term_index_for_param(
    terms: &[CompiledKernel],
    param_idx: usize,
) -> Result<(usize, usize), GprError> {
    let mut offset = 0;
    for (idx, term) in terms.iter().enumerate() {
        let n = term.num_params();
        if param_idx < offset + n {
            return Ok((idx, param_idx - offset));
        }
        offset += n;
    }
    Err(GprError::InvalidHyperparameter {
        reason: format!("kernel parameter index {param_idx} is out of range"),
    })
}

fn zero_triangle(mut out: MatMut<'_, f64>, uplo: Triangle) {
    visit_triangle(out.nrows(), uplo, |row, col| {
        out[(row, col)] = 0.0;
    });
}

fn product_hess(
    terms: &[CompiledKernel],
    dist: MatRef<'_, f64>,
    d2_k: MatMut<'_, f64>,
    i: usize,
    j: usize,
    uplo: Triangle,
    scratch: MatMut<'_, f64>,
) -> Result<(), GprError> {
    match owners_for_pair(terms, i, j)? {
        PairOwners::Same {
            term: _,
            local_i,
            local_j,
        } => {
            let (owner, _) = term_index_for_param(terms, i)?;
            product_same_leaf(
                terms,
                owner,
                d2_k,
                scratch,
                uplo,
                |term, dest, scratch| term.apply(dist, dest, uplo, scratch),
                |term, dest, scratch| term.hess(dist, dest, local_i, local_j, uplo, scratch),
            )
        }
        PairOwners::Distinct {
            owner_i,
            local_i,
            owner_j,
            local_j,
        } => product_cross_leaf(
            terms,
            owner_i,
            owner_j,
            d2_k,
            scratch,
            uplo,
            |term, dest, scratch| term.apply(dist, dest, uplo, scratch),
            |term, dest, scratch| term.grad(dist, dest, local_i, uplo, scratch),
            |term, dest, scratch| term.grad(dist, dest, local_j, uplo, scratch),
        ),
    }
}

fn product_hess_points(
    terms: &[CompiledKernel],
    x: MatRef<'_, f64>,
    d2_k: MatMut<'_, f64>,
    i: usize,
    j: usize,
    uplo: Triangle,
    scratch: MatMut<'_, f64>,
) -> Result<(), GprError> {
    match owners_for_pair(terms, i, j)? {
        PairOwners::Same {
            local_i, local_j, ..
        } => {
            let (owner, _) = term_index_for_param(terms, i)?;
            product_same_leaf(
                terms,
                owner,
                d2_k,
                scratch,
                uplo,
                |term, dest, scratch| term.apply_points(x, dest, uplo, scratch),
                |term, dest, scratch| term.hess_points(x, dest, local_i, local_j, uplo, scratch),
            )
        }
        PairOwners::Distinct {
            owner_i,
            local_i,
            owner_j,
            local_j,
        } => product_cross_leaf(
            terms,
            owner_i,
            owner_j,
            d2_k,
            scratch,
            uplo,
            |term, dest, scratch| term.apply_points(x, dest, uplo, scratch),
            |term, dest, scratch| term.grad_points(x, dest, local_i, uplo, scratch),
            |term, dest, scratch| term.grad_points(x, dest, local_j, uplo, scratch),
        ),
    }
}

fn product_hess_mixed(
    terms: &[CompiledKernel],
    views: MixedKernelViews<'_>,
    d2_k: MatMut<'_, f64>,
    i: usize,
    j: usize,
    uplo: Triangle,
    scratch: MatMut<'_, f64>,
) -> Result<(), GprError> {
    match owners_for_pair(terms, i, j)? {
        PairOwners::Same {
            local_i, local_j, ..
        } => {
            let (owner, _) = term_index_for_param(terms, i)?;
            product_same_leaf(
                terms,
                owner,
                d2_k,
                scratch,
                uplo,
                |term, dest, scratch| term.apply_mixed(views, dest, uplo, scratch),
                |term, dest, scratch| term.hess_mixed(views, dest, local_i, local_j, uplo, scratch),
            )
        }
        PairOwners::Distinct {
            owner_i,
            local_i,
            owner_j,
            local_j,
        } => product_cross_leaf(
            terms,
            owner_i,
            owner_j,
            d2_k,
            scratch,
            uplo,
            |term, dest, scratch| term.apply_mixed(views, dest, uplo, scratch),
            |term, dest, scratch| term.grad_mixed(views, dest, local_i, uplo, scratch),
            |term, dest, scratch| term.grad_mixed(views, dest, local_j, uplo, scratch),
        ),
    }
}

fn product_same_leaf(
    terms: &[CompiledKernel],
    owner: usize,
    mut d2_k: MatMut<'_, f64>,
    mut scratch: MatMut<'_, f64>,
    uplo: Triangle,
    mut apply: impl FnMut(&CompiledKernel, MatMut<'_, f64>, MatMut<'_, f64>) -> Result<(), GprError>,
    mut hess: impl FnMut(&CompiledKernel, MatMut<'_, f64>, MatMut<'_, f64>) -> Result<(), GprError>,
) -> Result<(), GprError> {
    let n = d2_k.nrows();
    let mut extra = None;
    let mut started = false;
    for (k, term) in terms.iter().enumerate() {
        if k == owner {
            continue;
        }
        if !started {
            apply(term, d2_k.as_mut(), scratch.as_mut())?;
            started = true;
        } else if term.needs_internal_scratch() {
            let buf = extra.get_or_insert_with(|| Mat::zeros(n, n));
            apply(term, scratch.as_mut(), buf.as_mut())?;
            mul_triangle(d2_k.as_mut(), scratch.as_ref(), uplo);
        } else {
            apply(term, scratch.as_mut(), d2_k.as_mut())?;
            mul_triangle(d2_k.as_mut(), scratch.as_ref(), uplo);
        }
    }
    if started {
        if terms[owner].needs_internal_scratch() {
            let buf = extra.get_or_insert_with(|| Mat::zeros(n, n));
            hess(&terms[owner], scratch.as_mut(), buf.as_mut())?;
        } else {
            hess(&terms[owner], scratch.as_mut(), d2_k.as_mut())?;
        }
        mul_triangle(d2_k.as_mut(), scratch.as_ref(), uplo);
    } else {
        hess(&terms[owner], d2_k.as_mut(), scratch.as_mut())?;
    }
    Ok(())
}

// Owners, dest/scratch, apply, and both leaf grads do not fold without a new type.
#[allow(clippy::too_many_arguments)]
fn product_cross_leaf(
    terms: &[CompiledKernel],
    owner_i: usize,
    owner_j: usize,
    mut d2_k: MatMut<'_, f64>,
    mut scratch: MatMut<'_, f64>,
    uplo: Triangle,
    mut apply: impl FnMut(&CompiledKernel, MatMut<'_, f64>, MatMut<'_, f64>) -> Result<(), GprError>,
    mut grad_i: impl FnMut(&CompiledKernel, MatMut<'_, f64>, MatMut<'_, f64>) -> Result<(), GprError>,
    mut grad_j: impl FnMut(&CompiledKernel, MatMut<'_, f64>, MatMut<'_, f64>) -> Result<(), GprError>,
) -> Result<(), GprError> {
    let n = d2_k.nrows();
    let mut extra = None;
    let mut started = false;
    for (k, term) in terms.iter().enumerate() {
        if k == owner_i || k == owner_j {
            continue;
        }
        if !started {
            apply(term, d2_k.as_mut(), scratch.as_mut())?;
            started = true;
        } else if term.needs_internal_scratch() {
            let buf = extra.get_or_insert_with(|| Mat::zeros(n, n));
            apply(term, scratch.as_mut(), buf.as_mut())?;
            mul_triangle(d2_k.as_mut(), scratch.as_ref(), uplo);
        } else {
            apply(term, scratch.as_mut(), d2_k.as_mut())?;
            mul_triangle(d2_k.as_mut(), scratch.as_ref(), uplo);
        }
    }
    if started {
        write_product_grad(
            &terms[owner_i],
            scratch.as_mut(),
            d2_k.as_mut(),
            &mut extra,
            n,
            &mut grad_i,
        )?;
        mul_triangle(d2_k.as_mut(), scratch.as_ref(), uplo);
        write_product_grad(
            &terms[owner_j],
            scratch.as_mut(),
            d2_k.as_mut(),
            &mut extra,
            n,
            &mut grad_j,
        )?;
        mul_triangle(d2_k.as_mut(), scratch.as_ref(), uplo);
    } else {
        write_product_grad(
            &terms[owner_i],
            d2_k.as_mut(),
            scratch.as_mut(),
            &mut extra,
            n,
            &mut grad_i,
        )?;
        write_product_grad(
            &terms[owner_j],
            scratch.as_mut(),
            d2_k.as_mut(),
            &mut extra,
            n,
            &mut grad_j,
        )?;
        mul_triangle(d2_k.as_mut(), scratch.as_ref(), uplo);
    }
    Ok(())
}

fn write_product_grad(
    term: &CompiledKernel,
    dest: MatMut<'_, f64>,
    fallback: MatMut<'_, f64>,
    extra: &mut Option<Mat<f64>>,
    n: usize,
    mut grad: impl FnMut(&CompiledKernel, MatMut<'_, f64>, MatMut<'_, f64>) -> Result<(), GprError>,
) -> Result<(), GprError> {
    if term.needs_internal_scratch() {
        let buf = extra.get_or_insert_with(|| Mat::zeros(n, n));
        grad(term, dest, buf.as_mut())
    } else {
        grad(term, dest, fallback)
    }
}

#[cfg(test)]
mod tests {
    use super::{CompiledKernel, MixedKernelViews};
    use crate::kernel::{
        ConstantKernel, KernelSpec, KernelTerm, LinearKernel, MaternArdKernel, MaternKernel,
        MaternNu, PeriodicKernel, RationalQuadraticArdKernel, RationalQuadraticKernel,
        RbfArdKernel, RbfKernel, Triangle, WhiteKernel,
    };
    use crate::param::Interval;
    use faer::{Mat, MatRef, mat};

    const TOL: f64 = 1e-9;

    fn assert_close(actual: f64, expected: f64) {
        let scale = expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= TOL * scale,
            "actual={actual}, expected={expected}"
        );
    }

    fn assert_send_sync<T: Send + Sync>() {}

    fn rbf(ell: f64) -> KernelSpec {
        KernelSpec::from(RbfKernel::new(ell).expect("valid"))
    }

    #[derive(Clone, Debug)]
    struct RbfAsTerm(RbfKernel);

    impl KernelTerm for RbfAsTerm {
        fn num_params(&self) -> usize {
            self.0.num_params()
        }

        fn get_params(&self, out: &mut [f64]) -> Result<(), crate::GprError> {
            self.0.get_params(out)
        }

        fn set_params(&mut self, params: &[f64]) -> Result<(), crate::GprError> {
            self.0.set_params(params)
        }

        fn bounds_into(&self, out: &mut [Interval]) -> Result<(), crate::GprError> {
            if out.len() != 1 {
                return Err(crate::GprError::InvalidHyperparameter {
                    reason: format!("expected 1 bound, got {}", out.len()),
                });
            }
            out[0] = self.0.bounds();
            Ok(())
        }

        fn apply(
            &self,
            dist: MatRef<'_, f64>,
            out: faer::MatMut<'_, f64>,
            uplo: Triangle,
        ) -> Result<(), crate::GprError> {
            self.0.apply(dist, out, uplo)
        }

        fn apply_cross(
            &self,
            dist: MatRef<'_, f64>,
            out: faer::MatMut<'_, f64>,
        ) -> Result<(), crate::GprError> {
            self.0.apply_cross(dist, out)
        }

        fn fill_diag(&self, out: &mut [f64]) -> Result<(), crate::GprError> {
            self.0.fill_diag(out);
            Ok(())
        }

        fn grad(
            &self,
            dist: MatRef<'_, f64>,
            d_k: faer::MatMut<'_, f64>,
            param_idx: usize,
            uplo: Triangle,
        ) -> Result<(), crate::GprError> {
            self.0.grad(dist, d_k, param_idx, uplo)
        }

        fn hess(
            &self,
            dist: MatRef<'_, f64>,
            d2_k: faer::MatMut<'_, f64>,
            i: usize,
            j: usize,
            uplo: Triangle,
        ) -> Result<(), crate::GprError> {
            self.0.hess(dist, d2_k, i, j, uplo)
        }

        fn hess_points(
            &self,
            x: MatRef<'_, f64>,
            d2_k: faer::MatMut<'_, f64>,
            i: usize,
            j: usize,
            uplo: Triangle,
        ) -> Result<(), crate::GprError> {
            self.0.hess_from_coords(x, d2_k, i, j, uplo)
        }

        fn clone_box(&self) -> Box<dyn KernelTerm> {
            Box::new(self.clone())
        }
    }

    fn custom_rbf(ell: f64) -> KernelSpec {
        KernelSpec::custom(RbfAsTerm(RbfKernel::new(ell).expect("valid")))
    }

    fn sq_dist_1d(x: &[f64]) -> Mat<f64> {
        let n = x.len();
        Mat::from_fn(n, n, |i, j| {
            let d = x[i] - x[j];
            d * d
        })
    }

    fn fill(n: usize, value: f64) -> Mat<f64> {
        Mat::from_fn(n, n, |_, _| value)
    }

    fn apply_compiled(compiled: &CompiledKernel, dist: MatRef<'_, f64>) -> Mat<f64> {
        let n = dist.nrows();
        let mut out = fill(n, 0.0);
        let mut scratch = fill(n, 0.0);
        compiled
            .apply(dist, out.as_mut(), Triangle::Full, scratch.as_mut())
            .expect("shape");
        out
    }

    fn apply_compiled_points(compiled: &CompiledKernel, x: MatRef<'_, f64>) -> Mat<f64> {
        let n = x.nrows();
        let mut out = fill(n, 0.0);
        let mut scratch = fill(n, 0.0);
        compiled
            .apply_points(x, out.as_mut(), Triangle::Full, scratch.as_mut())
            .expect("shape");
        out
    }

    fn apply_rbf(ell: f64, dist: MatRef<'_, f64>) -> Mat<f64> {
        let n = dist.nrows();
        let mut out = fill(n, 0.0);
        RbfKernel::new(ell)
            .expect("valid")
            .apply(dist, out.as_mut(), Triangle::Full)
            .expect("shape");
        out
    }

    fn lower_matches(actual: MatRef<'_, f64>, expected: MatRef<'_, f64>) {
        let n = actual.nrows();
        for col in 0..n {
            for row in col..n {
                assert_close(actual[(row, col)], expected[(row, col)]);
            }
        }
    }

    #[test]
    fn is_send_sync() {
        assert_send_sync::<CompiledKernel>();
    }

    #[test]
    fn custom_plus_rbf_apply_matches_two_rbf() {
        let compiled = (custom_rbf(1.0) + rbf(2.0)).compile();
        let builtin = (rbf(1.0) + rbf(2.0)).compile();
        let dist = sq_dist_1d(&[0.0, 1.0, 2.0]);
        let out = apply_compiled(&compiled, dist.as_ref());
        let expected = apply_compiled(&builtin, dist.as_ref());
        for col in 0..3 {
            for row in 0..3 {
                assert_close(out[(row, col)], expected[(row, col)]);
            }
        }
    }

    #[test]
    fn custom_times_rbf_apply_matches_two_rbf() {
        let compiled = (custom_rbf(1.0) * rbf(0.5)).compile();
        let builtin = (rbf(1.0) * rbf(0.5)).compile();
        let dist = sq_dist_1d(&[0.0, 1.2]);
        let out = apply_compiled(&compiled, dist.as_ref());
        let expected = apply_compiled(&builtin, dist.as_ref());
        assert_close(out[(0, 1)], expected[(0, 1)]);
        assert_close(out[(0, 0)], expected[(0, 0)]);
    }

    #[test]
    fn custom_sum_grad_matches_finite_difference() {
        let spec = custom_rbf(1.0) + rbf(2.0);
        let compiled = spec.compile();
        let dist = sq_dist_1d(&[0.0, 0.8, 1.5]);
        let mut params = [0.0; 2];
        spec.get_params(&mut params).expect("len 2");
        let h = 1e-6;
        let mut spec_plus = spec.clone();
        let mut spec_minus = spec.clone();
        params[1] += h;
        spec_plus.set_params(&params).expect("valid");
        params[1] -= 2.0 * h;
        spec_minus.set_params(&params).expect("valid");
        let kp = apply_compiled(&spec_plus.compile(), dist.as_ref());
        let km = apply_compiled(&spec_minus.compile(), dist.as_ref());
        let mut dk = fill(3, 0.0);
        let mut scratch = fill(3, 0.0);
        compiled
            .grad(
                dist.as_ref(),
                dk.as_mut(),
                1,
                Triangle::Full,
                scratch.as_mut(),
            )
            .expect("idx 1");
        for col in 0..3 {
            for row in 0..3 {
                let fd = (kp[(row, col)] - km[(row, col)]) / (2.0 * h);
                assert!(
                    (dk[(row, col)] - fd).abs() <= 1e-4 * fd.abs().max(1.0),
                    "row={row} col={col} analytic={} fd={fd}",
                    dk[(row, col)]
                );
            }
        }
    }

    #[test]
    fn sum_apply_adds_leaves() {
        let compiled = (rbf(1.0) + rbf(2.0)).compile();
        let dist = sq_dist_1d(&[0.0, 1.0, 2.0]);
        let out = apply_compiled(&compiled, dist.as_ref());
        let k1 = apply_rbf(1.0, dist.as_ref());
        let k2 = apply_rbf(2.0, dist.as_ref());
        for col in 0..3 {
            for row in 0..3 {
                assert_close(out[(row, col)], k1[(row, col)] + k2[(row, col)]);
            }
        }
    }

    #[test]
    fn product_apply_multiplies_leaves() {
        let compiled = (rbf(1.0) * rbf(0.5)).compile();
        let dist = sq_dist_1d(&[0.0, 1.2]);
        let out = apply_compiled(&compiled, dist.as_ref());
        let k1 = apply_rbf(1.0, dist.as_ref());
        let k2 = apply_rbf(0.5, dist.as_ref());
        assert_close(out[(0, 1)], k1[(0, 1)] * k2[(0, 1)]);
        assert_close(out[(0, 0)], 1.0);
    }

    #[test]
    fn mixed_product_of_sum_matches_leaves() {
        let spec = rbf(1.0) * (rbf(2.0) + rbf(3.0));
        let compiled = spec.compile();
        let dist = sq_dist_1d(&[0.0, 0.7, 1.4]);
        let out = apply_compiled(&compiled, dist.as_ref());
        let k1 = apply_rbf(1.0, dist.as_ref());
        let k2 = apply_rbf(2.0, dist.as_ref());
        let k3 = apply_rbf(3.0, dist.as_ref());
        for col in 0..3 {
            for row in 0..3 {
                assert_close(
                    out[(row, col)],
                    k1[(row, col)] * (k2[(row, col)] + k3[(row, col)]),
                );
            }
        }
    }

    #[test]
    fn product_of_two_sums_matches_leaves() {
        let spec = (rbf(1.0) + rbf(2.0)) * (rbf(0.5) + rbf(1.5));
        let compiled = spec.compile();
        let dist = sq_dist_1d(&[0.0, 1.0]);
        let out = apply_compiled(&compiled, dist.as_ref());
        let a = apply_rbf(1.0, dist.as_ref());
        let b = apply_rbf(2.0, dist.as_ref());
        let c = apply_rbf(0.5, dist.as_ref());
        let d = apply_rbf(1.5, dist.as_ref());
        for col in 0..2 {
            for row in 0..2 {
                assert_close(
                    out[(row, col)],
                    (a[(row, col)] + b[(row, col)]) * (c[(row, col)] + d[(row, col)]),
                );
            }
        }
    }

    #[test]
    fn sum_lower_matches_full_and_leaves_upper() {
        let compiled = (rbf(1.0) + rbf(2.0)).compile();
        let dist = sq_dist_1d(&[0.0, 1.0, 2.0]);
        let full = apply_compiled(&compiled, dist.as_ref());
        let sentinel = 42.0;
        let mut lower = fill(3, sentinel);
        let mut scratch = fill(3, 0.0);
        compiled
            .apply(
                dist.as_ref(),
                lower.as_mut(),
                Triangle::Lower,
                scratch.as_mut(),
            )
            .expect("shape");
        lower_matches(lower.as_ref(), full.as_ref());
        assert_close(lower[(0, 1)], sentinel);
        assert_close(lower[(0, 2)], sentinel);
        assert_close(lower[(1, 2)], sentinel);
    }

    #[test]
    fn sum_grad_matches_finite_difference() {
        let spec = rbf(1.0) + rbf(2.0);
        let compiled = spec.compile();
        let dist = sq_dist_1d(&[0.0, 0.8, 1.5]);
        let mut params = [0.0; 2];
        spec.get_params(&mut params).expect("len 2");
        let h = 1e-6;
        let mut spec_plus = spec.clone();
        let mut spec_minus = spec.clone();
        params[1] += h;
        spec_plus.set_params(&params).expect("valid");
        params[1] -= 2.0 * h;
        spec_minus.set_params(&params).expect("valid");
        let kp = apply_compiled(&spec_plus.compile(), dist.as_ref());
        let km = apply_compiled(&spec_minus.compile(), dist.as_ref());
        let mut dk = fill(3, 0.0);
        let mut scratch = fill(3, 0.0);
        compiled
            .grad(
                dist.as_ref(),
                dk.as_mut(),
                1,
                Triangle::Full,
                scratch.as_mut(),
            )
            .expect("idx 1");
        for col in 0..3 {
            for row in 0..3 {
                let fd = (kp[(row, col)] - km[(row, col)]) / (2.0 * h);
                assert_close(dk[(row, col)], fd);
            }
        }
    }

    #[test]
    fn product_hess_matches_finite_difference_of_grad() {
        let spec = rbf(1.25) * KernelSpec::from(ConstantKernel::new(1.4).expect("valid"));
        let compiled = spec.compile();
        let dist = mat![[0.0, 0.64, 2.89], [0.64, 0.0, 0.81], [2.89, 0.81, 0.0]];
        let mut params = [0.0; 2];
        spec.get_params(&mut params).expect("len 2");
        let h = 1e-6;
        for j in 0..2 {
            let mut plus = params;
            let mut minus = params;
            plus[j] += h;
            minus[j] -= h;
            let mut spec_plus = spec.clone();
            let mut spec_minus = spec.clone();
            spec_plus.set_params(&plus).expect("valid");
            spec_minus.set_params(&minus).expect("valid");
            let mut gp = fill(3, 0.0);
            let mut gm = fill(3, 0.0);
            let mut scratch = fill(3, 0.0);
            spec_plus
                .compile()
                .grad(
                    dist.as_ref(),
                    gp.as_mut(),
                    0,
                    Triangle::Full,
                    scratch.as_mut(),
                )
                .expect("plus");
            spec_minus
                .compile()
                .grad(
                    dist.as_ref(),
                    gm.as_mut(),
                    0,
                    Triangle::Full,
                    scratch.as_mut(),
                )
                .expect("minus");
            let mut d2 = fill(3, 0.0);
            compiled
                .hess(
                    dist.as_ref(),
                    d2.as_mut(),
                    0,
                    j,
                    Triangle::Full,
                    scratch.as_mut(),
                )
                .expect("hess");
            let fd = (gp[(0, 1)] - gm[(0, 1)]) / (2.0 * h);
            assert_close(d2[(0, 1)], fd);
        }
    }

    #[test]
    fn product_grad_matches_finite_difference() {
        let spec = rbf(1.0) * rbf(2.0);
        let compiled = spec.compile();
        let dist = mat![[0.0, 1.0], [1.0, 0.0]];
        let mut params = [0.0; 2];
        spec.get_params(&mut params).expect("len 2");
        let h = 1e-6;
        let mut spec_plus = spec.clone();
        let mut spec_minus = spec.clone();
        params[0] += h;
        spec_plus.set_params(&params).expect("valid");
        params[0] -= 2.0 * h;
        spec_minus.set_params(&params).expect("valid");
        let kp = apply_compiled(&spec_plus.compile(), dist.as_ref());
        let km = apply_compiled(&spec_minus.compile(), dist.as_ref());
        let mut dk = fill(2, 0.0);
        let mut scratch = fill(2, 0.0);
        compiled
            .grad(
                dist.as_ref(),
                dk.as_mut(),
                0,
                Triangle::Full,
                scratch.as_mut(),
            )
            .expect("idx 0");
        let fd = (kp[(0, 1)] - km[(0, 1)]) / (2.0 * h);
        assert_close(dk[(0, 1)], fd);
    }

    #[test]
    fn nested_product_grad_matches_finite_difference() {
        let spec = rbf(1.0) * (rbf(2.0) + rbf(3.0));
        let compiled = spec.compile();
        let dist = sq_dist_1d(&[0.0, 0.9]);
        let mut params = [0.0; 3];
        spec.get_params(&mut params).expect("len 3");
        let h = 1e-6;
        let mut spec_plus = spec.clone();
        let mut spec_minus = spec.clone();
        params[2] += h;
        spec_plus.set_params(&params).expect("valid");
        params[2] -= 2.0 * h;
        spec_minus.set_params(&params).expect("valid");
        let kp = apply_compiled(&spec_plus.compile(), dist.as_ref());
        let km = apply_compiled(&spec_minus.compile(), dist.as_ref());
        let mut dk = fill(2, 0.0);
        let mut scratch = fill(2, 0.0);
        compiled
            .grad(
                dist.as_ref(),
                dk.as_mut(),
                2,
                Triangle::Full,
                scratch.as_mut(),
            )
            .expect("idx 2");
        let fd = (kp[(0, 1)] - km[(0, 1)]) / (2.0 * h);
        assert_close(dk[(0, 1)], fd);
    }

    fn assert_points_product_grad_fd(spec: KernelSpec, x: Mat<f64>, param_idx: usize) {
        let compiled = spec.compile();
        let mut params = vec![0.0; spec.num_params()];
        spec.get_params(&mut params).expect("len");
        let h = 1e-6;
        let mut spec_plus = spec.clone();
        let mut spec_minus = spec.clone();
        params[param_idx] += h;
        spec_plus.set_params(&params).expect("valid");
        params[param_idx] -= 2.0 * h;
        spec_minus.set_params(&params).expect("valid");
        let kp = apply_compiled_points(&spec_plus.compile(), x.as_ref());
        let km = apply_compiled_points(&spec_minus.compile(), x.as_ref());
        let mut dk = fill(x.nrows(), 0.0);
        let mut scratch = fill(x.nrows(), 0.0);
        compiled
            .grad_points(
                x.as_ref(),
                dk.as_mut(),
                param_idx,
                Triangle::Full,
                scratch.as_mut(),
            )
            .expect("grad");
        let fd = (kp[(0, 1)] - km[(0, 1)]) / (2.0 * h);
        assert_close(dk[(0, 1)], fd);
    }

    #[test]
    fn linear_times_linear_grad_matches_finite_difference() {
        let spec = KernelSpec::from(LinearKernel::new(1.0).expect("valid"))
            * KernelSpec::from(LinearKernel::new(0.5).expect("valid"));
        let x = Mat::from_fn(2, 1, |i, _| 0.5 + i as f64);
        assert_points_product_grad_fd(spec, x, 0);
    }

    #[test]
    fn linear_times_constant_grad_matches_finite_difference() {
        let spec = KernelSpec::from(LinearKernel::new(1.0).expect("valid"))
            * KernelSpec::from(ConstantKernel::new(1.5).expect("valid"));
        let x = Mat::from_fn(2, 1, |i, _| 0.5 + i as f64);
        assert_points_product_grad_fd(spec, x, 1);
    }

    #[test]
    fn linear_times_ard_grad_matches_finite_difference() {
        let spec = KernelSpec::from(LinearKernel::new(1.0).expect("valid"))
            * KernelSpec::from(RbfArdKernel::new(&[1.2, 0.8]).expect("valid"));
        let x = points_2d(&[[0.0, 0.0], [1.0, 0.4], [0.2, 1.1]]);
        assert_points_product_grad_fd(spec, x, 2);
    }

    #[test]
    fn nested_points_product_grad_matches_finite_difference() {
        let spec = KernelSpec::from(LinearKernel::new(1.0).expect("valid"))
            * (KernelSpec::from(LinearKernel::new(0.8).expect("valid"))
                + KernelSpec::from(ConstantKernel::new(0.5).expect("valid")));
        let x = Mat::from_fn(2, 1, |i, _| 0.5 + i as f64);
        assert_points_product_grad_fd(spec, x, 2);
    }

    #[test]
    fn get_set_params_roundtrip_and_atomic() {
        let mut compiled = (rbf(1.0) + rbf(2.0)).compile();
        let mut params = [0.0; 2];
        compiled.get_params(&mut params).expect("len 2");
        assert_close(params[0], 1.0_f64.ln());
        assert_close(params[1], 2.0_f64.ln());
        params[0] = 0.5_f64.ln();
        compiled.set_params(&params).expect("valid");
        compiled.get_params(&mut params).expect("len 2");
        assert_close(params[0], 0.5_f64.ln());
        let before = compiled.clone();
        assert!(compiled.set_params(&[0.0, f64::INFINITY]).is_err());
        assert_eq!(compiled, before);
    }

    #[test]
    fn rejects_bad_index_and_scratch_shape() {
        let compiled = (rbf(1.0) + rbf(2.0)).compile();
        let dist = sq_dist_1d(&[0.0, 1.0]);
        let mut dk = fill(2, 0.0);
        let mut scratch = fill(2, 0.0);
        assert!(matches!(
            compiled.grad(
                dist.as_ref(),
                dk.as_mut(),
                2,
                Triangle::Lower,
                scratch.as_mut()
            ),
            Err(crate::error::GprError::InvalidHyperparameter { .. })
        ));
        let mut small = fill(1, 0.0);
        assert!(matches!(
            compiled.apply(dist.as_ref(), dk.as_mut(), Triangle::Full, small.as_mut()),
            Err(crate::error::GprError::WorkspaceTooSmall)
        ));
    }

    #[test]
    fn empty_sum_is_unsupported() {
        let compiled = CompiledKernel::Sum(Vec::new());
        let dist = sq_dist_1d(&[0.0, 1.0]);
        let mut out = fill(2, 0.0);
        let mut scratch = fill(2, 0.0);
        assert!(matches!(
            compiled.apply(
                dist.as_ref(),
                out.as_mut(),
                Triangle::Full,
                scratch.as_mut()
            ),
            Err(crate::error::GprError::UnsupportedKernelOperation { .. })
        ));
    }

    #[test]
    fn fill_diag_adds_rbf_leaves() {
        let compiled = (rbf(1.0) + rbf(2.0)).compile();
        let mut diag = [0.0, 0.0];
        compiled.fill_diag(&mut diag).expect("two terms");
        assert_close(diag[0], 2.0);
        assert_close(diag[1], 2.0);
    }

    #[test]
    fn apply_cross_matches_full_block() {
        let compiled = rbf(1.0).compile();
        let train = sq_dist_1d(&[0.0, 1.0]);
        let mut k_nn = fill(2, 0.0);
        let mut scratch = fill(2, 0.0);
        compiled
            .apply(
                train.as_ref(),
                k_nn.as_mut(),
                Triangle::Full,
                scratch.as_mut(),
            )
            .expect("square");
        let dist_cross = faer::mat![[0.0, 1.0], [1.0, 0.0]];
        let mut k_cross = fill(2, 0.0);
        let mut scratch_cross = fill(2, 0.0);
        compiled
            .apply_cross(
                dist_cross.as_ref(),
                k_cross.as_mut(),
                scratch_cross.as_mut(),
            )
            .expect("rect");
        assert_close(k_cross[(0, 0)], k_nn[(0, 0)]);
        assert_close(k_cross[(0, 1)], k_nn[(0, 1)]);
    }

    fn points_2d(rows: &[[f64; 2]]) -> Mat<f64> {
        Mat::from_fn(rows.len(), 2, |i, j| rows[i][j])
    }

    #[test]
    fn ard_apply_dist_is_unsupported_points_match_isotropic() {
        let ell = 1.3;
        let compiled = KernelSpec::from(RbfArdKernel::new(&[ell, ell]).expect("valid")).compile();
        let x = points_2d(&[[0.0, 0.0], [1.0, 0.4], [0.2, 1.1]]);
        let dist = {
            let n = x.nrows();
            Mat::from_fn(n, n, |row, col| {
                let mut sum = 0.0;
                for dim in 0..2 {
                    let diff = x[(row, dim)] - x[(col, dim)];
                    sum += diff * diff;
                }
                sum
            })
        };
        let mut out = fill(3, 0.0);
        let mut scratch = fill(3, 0.0);
        assert!(matches!(
            compiled.apply(
                dist.as_ref(),
                out.as_mut(),
                Triangle::Full,
                scratch.as_mut()
            ),
            Err(crate::error::GprError::UnsupportedKernelOperation { .. })
        ));
        compiled
            .apply_points(x.as_ref(), out.as_mut(), Triangle::Full, scratch.as_mut())
            .expect("points");
        let iso = apply_rbf(ell, dist.as_ref());
        for col in 0..3 {
            for row in 0..3 {
                assert_close(out[(row, col)], iso[(row, col)]);
            }
        }
    }

    #[test]
    fn custom_plus_linear_is_mixed() {
        let spec = custom_rbf(1.0) + KernelSpec::from(LinearKernel::new(1.0).expect("valid"));
        let compiled = spec.compile();
        assert_eq!(compiled.coord_mode().expect("mix"), super::CoordMode::Mixed);
    }

    #[test]
    fn mixed_isotropic_and_ard_is_mixed() {
        let spec = rbf(1.0) + KernelSpec::from(RbfArdKernel::new(&[1.0, 2.0]).expect("valid"));
        let compiled = spec.compile();
        assert_eq!(compiled.coord_mode().expect("mix"), super::CoordMode::Mixed);
    }

    #[test]
    fn rbf_plus_linear_apply_adds_leaves() {
        let spec = rbf(1.0) + KernelSpec::from(LinearKernel::new(1.0).expect("valid"));
        let compiled = spec.compile();
        assert_eq!(compiled.coord_mode().expect("mix"), super::CoordMode::Mixed);
        let xs = [0.5, 1.5];
        let x = Mat::from_fn(2, 1, |i, _| xs[i]);
        let dist = sq_dist_1d(&xs);
        let mut out = fill(2, 0.0);
        let mut scratch = fill(2, 0.0);
        compiled
            .apply_mixed(
                MixedKernelViews::new(dist.as_ref(), x.as_ref()),
                out.as_mut(),
                Triangle::Full,
                scratch.as_mut(),
            )
            .expect("mix");
        let kr = apply_compiled(&rbf(1.0).compile(), dist.as_ref());
        let mut kl = fill(2, 0.0);
        LinearKernel::new(1.0)
            .expect("valid")
            .apply(x.as_ref(), kl.as_mut(), Triangle::Full)
            .expect("linear");
        for col in 0..2 {
            for row in 0..2 {
                assert_close(out[(row, col)], kr[(row, col)] + kl[(row, col)]);
            }
        }
    }

    #[test]
    fn rbf_times_linear_apply_multiplies_leaves() {
        let spec = rbf(1.0) * KernelSpec::from(LinearKernel::new(1.0).expect("valid"));
        let compiled = spec.compile();
        assert_eq!(compiled.coord_mode().expect("mix"), super::CoordMode::Mixed);
        let xs = [0.5, 1.5];
        let x = Mat::from_fn(2, 1, |i, _| xs[i]);
        let dist = sq_dist_1d(&xs);
        let mut out = fill(2, 0.0);
        let mut scratch = fill(2, 0.0);
        compiled
            .apply_mixed(
                MixedKernelViews::new(dist.as_ref(), x.as_ref()),
                out.as_mut(),
                Triangle::Full,
                scratch.as_mut(),
            )
            .expect("mix");
        let kr = apply_compiled(&rbf(1.0).compile(), dist.as_ref());
        let mut kl = fill(2, 0.0);
        LinearKernel::new(1.0)
            .expect("valid")
            .apply(x.as_ref(), kl.as_mut(), Triangle::Full)
            .expect("linear");
        for col in 0..2 {
            for row in 0..2 {
                assert_close(out[(row, col)], kr[(row, col)] * kl[(row, col)]);
            }
        }
    }

    #[test]
    fn rbf_plus_linear_grad_matches_finite_difference() {
        let spec = rbf(1.0) + KernelSpec::from(LinearKernel::new(0.8).expect("valid"));
        let compiled = spec.compile();
        let xs = [0.5, 1.5];
        let x = Mat::from_fn(2, 1, |i, _| xs[i]);
        let dist = sq_dist_1d(&xs);
        let mut params = [0.0; 2];
        spec.get_params(&mut params).expect("len 2");
        let h = 1e-6;
        let mut spec_plus = spec.clone();
        let mut spec_minus = spec.clone();
        params[1] += h;
        spec_plus.set_params(&params).expect("valid");
        params[1] -= 2.0 * h;
        spec_minus.set_params(&params).expect("valid");
        let mut kp = fill(2, 0.0);
        let mut km = fill(2, 0.0);
        let mut scratch = fill(2, 0.0);
        spec_plus
            .compile()
            .apply_mixed(
                MixedKernelViews::new(dist.as_ref(), x.as_ref()),
                kp.as_mut(),
                Triangle::Full,
                scratch.as_mut(),
            )
            .expect("plus");
        spec_minus
            .compile()
            .apply_mixed(
                MixedKernelViews::new(dist.as_ref(), x.as_ref()),
                km.as_mut(),
                Triangle::Full,
                scratch.as_mut(),
            )
            .expect("minus");
        let mut dk = fill(2, 0.0);
        compiled
            .grad_mixed(
                MixedKernelViews::new(dist.as_ref(), x.as_ref()),
                dk.as_mut(),
                1,
                Triangle::Full,
                scratch.as_mut(),
            )
            .expect("grad");
        let fd = (kp[(0, 1)] - km[(0, 1)]) / (2.0 * h);
        assert_close(dk[(0, 1)], fd);
    }

    #[test]
    fn rbf_plus_white_is_distance_mode() {
        let spec = rbf(1.0) + KernelSpec::from(WhiteKernel::new(0.1).expect("valid"));
        let compiled = spec.compile();
        assert_eq!(
            compiled.coord_mode().expect("compat"),
            super::CoordMode::Dist
        );
        let dist = sq_dist_1d(&[0.0, 1.0]);
        let mut out = fill(2, 0.0);
        let mut scratch = fill(2, 0.0);
        compiled
            .apply(
                dist.as_ref(),
                out.as_mut(),
                Triangle::Full,
                scratch.as_mut(),
            )
            .expect("shape");
        assert_close(out[(0, 0)], 1.0 + 0.1);
        assert_close(out[(0, 1)], apply_rbf(1.0, dist.as_ref())[(0, 1)]);
    }

    #[test]
    fn linear_plus_constant_is_points_mode() {
        let spec = KernelSpec::from(LinearKernel::new(1.0).expect("valid"))
            + KernelSpec::from(ConstantKernel::new(0.5).expect("valid"));
        let compiled = spec.compile();
        assert_eq!(
            compiled.coord_mode().expect("compat"),
            super::CoordMode::Points
        );
        let x = Mat::from_fn(2, 1, |i, _| i as f64);
        let mut out = fill(2, 0.0);
        let mut scratch = fill(2, 0.0);
        compiled
            .apply_points(x.as_ref(), out.as_mut(), Triangle::Full, scratch.as_mut())
            .expect("points");
        assert_close(out[(0, 0)], 0.5);
        assert_close(out[(1, 1)], 1.0 + 0.5);
        assert_close(out[(1, 0)], 0.5);
    }

    #[test]
    fn matern_plus_white_is_distance_mode() {
        let spec = KernelSpec::from(MaternKernel::new(1.0, MaternNu::ThreeHalves).expect("valid"))
            + KernelSpec::from(WhiteKernel::new(0.1).expect("valid"));
        let compiled = spec.compile();
        assert_eq!(
            compiled.coord_mode().expect("compat"),
            super::CoordMode::Dist
        );
        let dist = sq_dist_1d(&[0.0, 1.0]);
        let mut out = fill(2, 0.0);
        let mut scratch = fill(2, 0.0);
        compiled
            .apply(
                dist.as_ref(),
                out.as_mut(),
                Triangle::Full,
                scratch.as_mut(),
            )
            .expect("shape");
        let rho = 3.0_f64.sqrt();
        let k01 = (1.0 + rho) * (-rho).exp();
        assert_close(out[(0, 0)], 1.0 + 0.1);
        assert_close(out[(0, 1)], k01);
    }

    #[test]
    fn matern_ard_plus_constant_is_points_mode() {
        let spec = KernelSpec::from(MaternArdKernel::new(&[1.0], MaternNu::Half).expect("valid"))
            + KernelSpec::from(ConstantKernel::new(0.5).expect("valid"));
        let compiled = spec.compile();
        assert_eq!(
            compiled.coord_mode().expect("compat"),
            super::CoordMode::Points
        );
        let x = Mat::from_fn(2, 1, |i, _| i as f64);
        let mut out = fill(2, 0.0);
        let mut scratch = fill(2, 0.0);
        compiled
            .apply_points(x.as_ref(), out.as_mut(), Triangle::Full, scratch.as_mut())
            .expect("points");
        assert_close(out[(0, 0)], 1.5);
        assert_close(out[(1, 1)], 1.5);
        assert_close(out[(1, 0)], (-1.0_f64).exp() + 0.5);
    }

    #[test]
    fn periodic_plus_white_is_distance_mode() {
        let spec = KernelSpec::from(PeriodicKernel::new(1.0, 2.0).expect("valid"))
            + KernelSpec::from(WhiteKernel::new(0.1).expect("valid"));
        let compiled = spec.compile();
        assert_eq!(
            compiled.coord_mode().expect("compat"),
            super::CoordMode::Dist
        );
        let dist = sq_dist_1d(&[0.0, 1.0]);
        let mut out = fill(2, 0.0);
        let mut scratch = fill(2, 0.0);
        compiled
            .apply(
                dist.as_ref(),
                out.as_mut(),
                Triangle::Full,
                scratch.as_mut(),
            )
            .expect("shape");
        let s = (std::f64::consts::PI * 0.5).sin();
        let k01 = (-2.0 * s * s).exp();
        assert_close(out[(0, 0)], 1.0 + 0.1);
        assert_close(out[(0, 1)], k01);
    }

    #[test]
    fn rational_quadratic_plus_white_is_distance_mode() {
        let spec = KernelSpec::from(RationalQuadraticKernel::new(1.0, 1.0).expect("valid"))
            + KernelSpec::from(WhiteKernel::new(0.1).expect("valid"));
        let compiled = spec.compile();
        assert_eq!(
            compiled.coord_mode().expect("compat"),
            super::CoordMode::Dist
        );
        let dist = sq_dist_1d(&[0.0, 1.0]);
        let mut out = fill(2, 0.0);
        let mut scratch = fill(2, 0.0);
        compiled
            .apply(
                dist.as_ref(),
                out.as_mut(),
                Triangle::Full,
                scratch.as_mut(),
            )
            .expect("shape");
        assert_close(out[(0, 0)], 1.1);
        assert_close(out[(0, 1)], 2.0 / 3.0);
    }

    #[test]
    fn rational_quadratic_ard_plus_constant_is_points_mode() {
        let spec = KernelSpec::from(RationalQuadraticArdKernel::new(&[1.0], 1.0).expect("valid"))
            + KernelSpec::from(ConstantKernel::new(0.5).expect("valid"));
        let compiled = spec.compile();
        assert_eq!(
            compiled.coord_mode().expect("compat"),
            super::CoordMode::Points
        );
        let x = Mat::from_fn(2, 1, |i, _| i as f64);
        let mut out = fill(2, 0.0);
        let mut scratch = fill(2, 0.0);
        compiled
            .apply_points(x.as_ref(), out.as_mut(), Triangle::Full, scratch.as_mut())
            .expect("points");
        assert_close(out[(0, 0)], 1.5);
        assert_close(out[(1, 1)], 1.5);
        assert_close(out[(1, 0)], 2.0 / 3.0 + 0.5);
    }
}
