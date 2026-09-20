use super::apply::{apply_into, apply_into_mixed, apply_into_points};
use super::{
    CompiledKernel, MixedKernelViews, ard_needs_coords, mul_triangle, require_scratch_shape,
};
use crate::error::GprError;
use crate::kernel::{CustomKernel, Triangle, write_square_from_coords};
use faer::{Mat, MatMut, MatRef};

impl CompiledKernel {
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

    pub(crate) fn needs_product_grad_scratch(&self) -> bool {
        match self {
            Self::Product(_) => true,
            Self::Sum(terms) => terms.iter().any(Self::needs_product_grad_scratch),
            _ => false,
        }
    }
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

pub(super) fn write_product_grad(
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
