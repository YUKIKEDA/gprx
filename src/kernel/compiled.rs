//! Execution-layer kernel: static dispatch over built-in leaves.

use super::{
    ConstantKernel, LinearKernel, MaternArdKernel, MaternKernel, RbfArdKernel, RbfKernel, Triangle,
    WhiteKernel, visit_triangle,
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
    /// Constant `k = c`.
    Constant(ConstantKernel),
    /// Linear `k = σ² xᵀ x'`.
    Linear(LinearKernel),
    /// White nugget.
    White(WhiteKernel),
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
            KernelSpec::Constant(leaf) => Self::Constant(*leaf),
            KernelSpec::Linear(leaf) => Self::Linear(*leaf),
            KernelSpec::White(leaf) => Self::White(*leaf),
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
            Self::Constant(leaf) => leaf.num_params(),
            Self::Linear(leaf) => leaf.num_params(),
            Self::White(leaf) => leaf.num_params(),
            Self::Sum(terms) | Self::Product(terms) => terms.iter().map(Self::num_params).sum(),
        }
    }

    /// Writes flattened `θ` in depth-first, left-to-right leaf order.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if `out` is the wrong length.
    pub fn get_params(&self, out: &mut [f64]) -> Result<(), GprError> {
        require_len(out.len(), self.num_params())?;
        let mut offset = 0;
        self.write_params(out, &mut offset);
        Ok(())
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
            Self::RbfArd(_) | Self::Linear(_) | Self::MaternArd(_) => Err(ard_needs_coords()),
            Self::Matern(leaf) => leaf.apply(dist, out, uplo),
            Self::Constant(leaf) => leaf.apply(dist, out, uplo),
            Self::White(leaf) => leaf.apply(dist, out, uplo),
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
            Self::RbfArd(_) | Self::Linear(_) | Self::MaternArd(_) => Err(ard_needs_coords()),
            Self::Matern(leaf) => leaf.apply_cross(dist, out),
            Self::Constant(leaf) => leaf.apply_cross(dist, out),
            Self::White(leaf) => leaf.apply_cross(dist, out),
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
            Self::Constant(leaf) => {
                leaf.fill_diag(out);
                Ok(())
            }
            Self::White(leaf) => {
                leaf.fill_diag(out);
                Ok(())
            }
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
            Self::Constant(leaf) => {
                leaf.fill_diag(out);
                Ok(())
            }
            Self::White(leaf) => {
                leaf.fill_diag(out);
                Ok(())
            }
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

    /// Writes `∂K/∂θ_{param_idx}` into `d_k`. `scratch` must match `d_k`.
    ///
    /// `scratch` must be a distinct buffer from `d_k`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if `param_idx` is out of
    /// range, [`GprError::WorkspaceTooSmall`] if `scratch` is the wrong size, or
    /// the same shape errors as [`Self::apply`].
    pub fn grad(
        &self,
        dist: MatRef<'_, f64>,
        mut d_k: MatMut<'_, f64>,
        param_idx: usize,
        uplo: Triangle,
        mut scratch: MatMut<'_, f64>,
    ) -> Result<(), GprError> {
        require_scratch_shape(d_k.as_ref(), scratch.as_ref())?;
        match self {
            Self::Rbf(leaf) => leaf.grad(dist, d_k, param_idx, uplo),
            Self::RbfArd(_) | Self::Linear(_) | Self::MaternArd(_) => Err(ard_needs_coords()),
            Self::Matern(leaf) => leaf.grad(dist, d_k, param_idx, uplo),
            Self::Constant(leaf) => leaf.grad(dist, d_k, param_idx, uplo),
            Self::White(leaf) => leaf.grad(dist, d_k, param_idx, uplo),
            Self::Sum(terms) => {
                let (term, local) = term_for_param(terms, param_idx)?;
                term.grad(dist, d_k, local, uplo, scratch)
            }
            Self::Product(terms) => {
                product_grad(terms, dist, d_k.as_mut(), param_idx, uplo, scratch.as_mut())
            }
        }
    }

    /// Writes `k` from point coordinates. Used by ARD leaves.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::UnsupportedKernelOperation`] for isotropic leaves,
    /// or the same shape errors as [`Self::apply`].
    pub fn apply_points(
        &self,
        x: MatRef<'_, f64>,
        mut out: MatMut<'_, f64>,
        uplo: Triangle,
        mut scratch: MatMut<'_, f64>,
    ) -> Result<(), GprError> {
        require_scratch_shape(out.as_ref(), scratch.as_ref())?;
        match self {
            Self::Rbf(_) | Self::Matern(_) => Err(iso_needs_dist()),
            Self::RbfArd(leaf) => leaf.apply(x, out, uplo),
            Self::Linear(leaf) => leaf.apply(x, out, uplo),
            Self::MaternArd(leaf) => leaf.apply(x, out, uplo),
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
            Self::Rbf(_) | Self::Matern(_) => Err(iso_needs_dist()),
            Self::RbfArd(leaf) => leaf.apply_cross(x, xs, out),
            Self::Linear(leaf) => leaf.apply_cross(x, xs, out),
            Self::MaternArd(leaf) => leaf.apply_cross(x, xs, out),
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
    /// # Errors
    ///
    /// Returns [`GprError::UnsupportedKernelOperation`] for isotropic leaves
    /// or product trees, or the same index / shape errors as [`Self::grad`].
    pub fn grad_points(
        &self,
        x: MatRef<'_, f64>,
        d_k: MatMut<'_, f64>,
        param_idx: usize,
        uplo: Triangle,
        scratch: MatMut<'_, f64>,
    ) -> Result<(), GprError> {
        require_scratch_shape(d_k.as_ref(), scratch.as_ref())?;
        match self {
            Self::Rbf(_) | Self::Matern(_) => Err(iso_needs_dist()),
            Self::RbfArd(leaf) => leaf.grad(x, d_k, param_idx, uplo),
            Self::Linear(leaf) => leaf.grad(x, d_k, param_idx, uplo),
            Self::MaternArd(leaf) => leaf.grad(x, d_k, param_idx, uplo),
            Self::Constant(leaf) => leaf.grad_points(x, d_k, param_idx, uplo),
            Self::White(leaf) => leaf.grad_points(x, d_k, param_idx, uplo),
            Self::Sum(terms) => {
                let (term, local) = term_for_param(terms, param_idx)?;
                term.grad_points(x, d_k, local, uplo, scratch)
            }
            Self::Product(_) => Err(GprError::UnsupportedKernelOperation {
                reason: "product kernel gradient needs a dedicated scratch buffer".to_owned(),
            }),
        }
    }

    pub(crate) fn coord_mode(&self) -> Result<CoordMode, GprError> {
        match self {
            Self::Rbf(_) | Self::Matern(_) => Ok(CoordMode::Dist),
            Self::RbfArd(_) | Self::Linear(_) | Self::MaternArd(_) => Ok(CoordMode::Points),
            Self::Constant(_) | Self::White(_) => Ok(CoordMode::Either),
            Self::Sum(terms) | Self::Product(terms) => {
                let (first, rest) = split_terms(terms)?;
                let mut mode = first.coord_mode()?;
                for term in rest {
                    mode = merge_coord_mode(mode, term.coord_mode()?)?;
                }
                Ok(mode)
            }
        }
    }

    fn write_params(&self, out: &mut [f64], offset: &mut usize) {
        match self {
            Self::Rbf(leaf) => {
                out[*offset] = leaf.log_lengthscale();
                *offset += 1;
            }
            Self::RbfArd(leaf) => {
                let n = leaf.num_params();
                out[*offset..*offset + n].copy_from_slice(leaf.log_lengthscales());
                *offset += n;
            }
            Self::Matern(leaf) => {
                out[*offset] = leaf.log_lengthscale();
                *offset += 1;
            }
            Self::MaternArd(leaf) => {
                let n = leaf.num_params();
                out[*offset..*offset + n].copy_from_slice(leaf.log_lengthscales());
                *offset += n;
            }
            Self::Constant(leaf) => {
                out[*offset] = leaf.log_constant();
                *offset += 1;
            }
            Self::Linear(leaf) => {
                out[*offset] = leaf.log_variance();
                *offset += 1;
            }
            Self::White(leaf) => {
                out[*offset] = leaf.log_variance();
                *offset += 1;
            }
            Self::Sum(terms) | Self::Product(terms) => {
                for term in terms {
                    term.write_params(out, offset);
                }
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
            | Self::Constant(_)
            | Self::Linear(_)
            | Self::White(_) => false,
            Self::Sum(terms) | Self::Product(terms) => {
                terms.len() > 1 || terms.iter().any(Self::needs_internal_scratch)
            }
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

fn merge_coord_mode(a: CoordMode, b: CoordMode) -> Result<CoordMode, GprError> {
    use CoordMode::{Dist, Either, Points};
    match (a, b) {
        (Either, other) | (other, Either) => Ok(other),
        (Dist, Dist) => Ok(Dist),
        (Points, Points) => Ok(Points),
        _ => Err(GprError::UnsupportedKernelOperation {
            reason: "cannot mix isotropic distance kernels with coordinate kernels".to_owned(),
        }),
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

#[cfg(test)]
mod tests {
    use super::CompiledKernel;
    use crate::kernel::{
        ConstantKernel, KernelSpec, LinearKernel, MaternArdKernel, MaternKernel, MaternNu,
        RbfArdKernel, RbfKernel, Triangle, WhiteKernel,
    };
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
    fn mixed_isotropic_and_ard_is_unsupported() {
        let spec = rbf(1.0) + KernelSpec::from(RbfArdKernel::new(&[1.0, 2.0]).expect("valid"));
        let compiled = spec.compile();
        assert!(matches!(
            compiled.coord_mode(),
            Err(crate::error::GprError::UnsupportedKernelOperation { .. })
        ));
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
}
