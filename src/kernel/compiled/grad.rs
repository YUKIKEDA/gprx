use super::apply::{apply_into, apply_into_mixed, apply_into_points};
use super::{
    CompiledKernel, MixedKernelViews, ard_needs_coords, mul_triangle, require_scratch_shape,
};
use crate::error::GprError;
use crate::kernel::{CustomKernel, Triangle, write_square_from_coords};
use faer::{Mat, MatMut, MatRef};

impl CompiledKernel<f64> {
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

    /// Writes `∂k(x_i, x_i)/∂θ_{param_idx}` into `out[i]`.
    ///
    /// The off-diagonal Gram derivative is not formed.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::EmptyInput`] if `x` is empty,
    /// [`GprError::InvalidHyperparameter`] if `param_idx` is out of range or
    /// `out.len()` is not `x.nrows()`, or the leaf error for that diagonal entry.
    pub(crate) fn grad_diag_points(
        &self,
        x: MatRef<'_, f64>,
        out: &mut [f64],
        param_idx: usize,
    ) -> Result<(), GprError> {
        match self {
            Self::Linear(leaf) => {
                require_diag_len(x, out)?;
                let one = x.submatrix(0, 0, 1, x.ncols());
                let mut cell = Mat::zeros(1, 1);
                leaf.grad(one.as_ref(), cell.as_mut(), param_idx, Triangle::Lower)?;
                leaf.fill_diag_points(x, out)
            }
            Self::Rbf(leaf) => broadcast_self_diag(x, out, |one, cell| {
                leaf.grad_from_coords(one, cell, param_idx, Triangle::Lower)
            }),
            Self::Matern(leaf) => broadcast_self_diag(x, out, |one, cell| {
                leaf.grad_from_coords(one, cell, param_idx, Triangle::Lower)
            }),
            Self::Periodic(leaf) => broadcast_self_diag(x, out, |one, cell| {
                leaf.grad_from_coords(one, cell, param_idx, Triangle::Lower)
            }),
            Self::RationalQuadratic(leaf) => broadcast_self_diag(x, out, |one, cell| {
                leaf.grad_from_coords(one, cell, param_idx, Triangle::Lower)
            }),
            Self::Custom(leaf) => broadcast_self_diag(x, out, |one, cell| {
                grad_custom_from_coords(leaf, one, cell, param_idx, Triangle::Lower)
            }),
            Self::RbfArd(leaf) => broadcast_self_diag(x, out, |one, cell| {
                leaf.grad(one, cell, param_idx, Triangle::Lower)
            }),
            Self::MaternArd(leaf) => broadcast_self_diag(x, out, |one, cell| {
                leaf.grad(one, cell, param_idx, Triangle::Lower)
            }),
            Self::RationalQuadraticArd(leaf) => broadcast_self_diag(x, out, |one, cell| {
                leaf.grad(one, cell, param_idx, Triangle::Lower)
            }),
            Self::Constant(leaf) => broadcast_self_diag(x, out, |one, cell| {
                leaf.grad_points(one, cell, param_idx, Triangle::Lower)
            }),
            Self::White(leaf) => broadcast_self_diag(x, out, |one, cell| {
                leaf.grad_points(one, cell, param_idx, Triangle::Lower)
            }),
            Self::Sum(terms) => {
                let (term, local) = term_for_param(terms, param_idx)?;
                term.grad_diag_points(x, out, local)
            }
            Self::Product(terms) => product_grad_diag(terms, x, out, param_idx),
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

    /// Writes `∂K(X1, X2)/∂X2[*, dim]` into `d_k`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::CoordGradientUnsupported`] when a leaf does not
    /// implement coordinate derivatives (including Product trees).
    pub fn grad_wrt_coord_dim(
        &self,
        x1: MatRef<'_, f64>,
        x2: MatRef<'_, f64>,
        mut d_k: MatMut<'_, f64>,
        dim: usize,
    ) -> Result<(), GprError> {
        match self {
            Self::Rbf(leaf) => leaf.grad_wrt_coord_dim(x1, x2, d_k, dim),
            Self::Matern(leaf) => leaf.grad_wrt_coord_dim(x1, x2, d_k, dim),
            Self::RbfArd(leaf) => leaf.grad_wrt_coord_dim(x1, x2, d_k, dim),
            Self::White(leaf) => leaf.grad_wrt_coord_dim(x1, x2, d_k, dim),
            Self::Custom(leaf) => leaf.grad_wrt_coord_dim(x1, x2, d_k, dim),
            Self::Sum(terms) => {
                let (first, rest) = terms
                    .split_first()
                    .ok_or(GprError::CoordGradientUnsupported)?;
                first.grad_wrt_coord_dim(x1, x2, d_k.as_mut(), dim)?;
                if rest.is_empty() {
                    return Ok(());
                }
                let mut scratch = Mat::zeros(d_k.nrows(), d_k.ncols());
                for term in rest {
                    term.grad_wrt_coord_dim(x1, x2, scratch.as_mut(), dim)?;
                    super::apply::add_rect(d_k.as_mut(), scratch.as_ref());
                }
                Ok(())
            }
            _ => Err(GprError::CoordGradientUnsupported),
        }
    }

    pub(crate) fn hess_wrt_coord_dims(
        &self,
        x1: MatRef<'_, f64>,
        x2: MatRef<'_, f64>,
        mut d2_k: MatMut<'_, f64>,
        dim_a: usize,
        dim_b: usize,
    ) -> Result<(), GprError> {
        match self {
            Self::Rbf(leaf) => leaf.hess_wrt_coord_dims(x1, x2, d2_k, dim_a, dim_b),
            Self::Matern(leaf) => leaf.hess_wrt_coord_dims(x1, x2, d2_k, dim_a, dim_b),
            Self::RbfArd(leaf) => leaf.hess_wrt_coord_dims(x1, x2, d2_k, dim_a, dim_b),
            Self::White(leaf) => leaf.hess_wrt_coord_dims(x1, x2, d2_k, dim_a, dim_b),
            Self::Sum(terms) => fold_coord_sum(terms, d2_k.as_mut(), |term, dest| {
                term.hess_wrt_coord_dims(x1, x2, dest, dim_a, dim_b)
            }),
            _ => Err(GprError::CoordGradientUnsupported),
        }
    }

    pub(crate) fn hess_wrt_coord_mixed(
        &self,
        x1: MatRef<'_, f64>,
        x2: MatRef<'_, f64>,
        mut d2_k: MatMut<'_, f64>,
        dim_x1: usize,
        dim_x2: usize,
    ) -> Result<(), GprError> {
        match self {
            Self::Rbf(leaf) => leaf.hess_wrt_coord_mixed(x1, x2, d2_k, dim_x1, dim_x2),
            Self::Matern(leaf) => leaf.hess_wrt_coord_mixed(x1, x2, d2_k, dim_x1, dim_x2),
            Self::RbfArd(leaf) => leaf.hess_wrt_coord_mixed(x1, x2, d2_k, dim_x1, dim_x2),
            Self::White(leaf) => leaf.hess_wrt_coord_mixed(x1, x2, d2_k, dim_x1, dim_x2),
            Self::Sum(terms) => fold_coord_sum(terms, d2_k.as_mut(), |term, dest| {
                term.hess_wrt_coord_mixed(x1, x2, dest, dim_x1, dim_x2)
            }),
            _ => Err(GprError::CoordGradientUnsupported),
        }
    }

    pub(crate) fn hess_theta_coord_dim(
        &self,
        x1: MatRef<'_, f64>,
        x2: MatRef<'_, f64>,
        d2_k: MatMut<'_, f64>,
        param_idx: usize,
        dim: usize,
    ) -> Result<(), GprError> {
        match self {
            Self::Rbf(leaf) => leaf.hess_theta_coord_dim(x1, x2, d2_k, param_idx, dim),
            Self::Matern(leaf) => leaf.hess_theta_coord_dim(x1, x2, d2_k, param_idx, dim),
            Self::RbfArd(leaf) => leaf.hess_theta_coord_dim(x1, x2, d2_k, param_idx, dim),
            Self::White(leaf) => leaf.hess_theta_coord_dim(x1, x2, d2_k, param_idx, dim),
            Self::Sum(terms) => {
                let (term, local) = term_for_param(terms, param_idx)?;
                term.hess_theta_coord_dim(x1, x2, d2_k, local, dim)
            }
            _ => Err(GprError::CoordGradientUnsupported),
        }
    }

    pub(crate) fn grad_cross_points(
        &self,
        x1: MatRef<'_, f64>,
        x2: MatRef<'_, f64>,
        d_k: MatMut<'_, f64>,
        param_idx: usize,
        mut scratch: MatMut<'_, f64>,
    ) -> Result<(), GprError> {
        match self {
            Self::Rbf(leaf) => leaf.grad_cross_from_coords(x1, x2, d_k, param_idx),
            Self::Matern(leaf) => leaf.grad_cross_from_coords(x1, x2, d_k, param_idx),
            Self::RbfArd(leaf) => leaf.grad_cross_from_coords(x1, x2, d_k, param_idx),
            Self::White(leaf) => {
                let _ = param_idx;
                if x1.ncols() == 0 {
                    return Err(GprError::EmptyInput);
                }
                leaf.grad_wrt_coord_dim(x1, x2, d_k, 0)
            }
            Self::Sum(terms) => {
                let (term, local) = term_for_param(terms, param_idx)?;
                term.grad_cross_points(x1, x2, d_k, local, scratch.as_mut())
            }
            _ => Err(GprError::CoordGradientUnsupported),
        }
    }

    pub(crate) fn hess_cross_points(
        &self,
        x1: MatRef<'_, f64>,
        x2: MatRef<'_, f64>,
        mut d2_k: MatMut<'_, f64>,
        i: usize,
        j: usize,
        mut scratch: MatMut<'_, f64>,
    ) -> Result<(), GprError> {
        match self {
            Self::Rbf(leaf) => leaf.hess_cross_from_coords(x1, x2, d2_k, i, j),
            Self::Matern(leaf) => leaf.hess_cross_from_coords(x1, x2, d2_k, i, j),
            Self::RbfArd(leaf) => leaf.hess_cross_from_coords(x1, x2, d2_k, i, j),
            Self::White(leaf) => {
                let _ = (i, j);
                if x1.ncols() == 0 {
                    return Err(GprError::EmptyInput);
                }
                leaf.grad_wrt_coord_dim(x1, x2, d2_k, 0)
            }
            Self::Sum(terms) => {
                let (term_i, li) = term_for_param(terms, i)?;
                let (term_j, lj) = term_for_param(terms, j)?;
                if std::ptr::eq(term_i, term_j) {
                    term_i.hess_cross_points(x1, x2, d2_k, li, lj, scratch.as_mut())
                } else {
                    for col in 0..d2_k.ncols() {
                        for row in 0..d2_k.nrows() {
                            d2_k[(row, col)] = 0.0;
                        }
                    }
                    Ok(())
                }
            }
            _ => Err(GprError::CoordGradientUnsupported),
        }
    }
}

fn fold_coord_sum(
    terms: &[CompiledKernel],
    mut out: MatMut<'_, f64>,
    mut eval: impl FnMut(&CompiledKernel, MatMut<'_, f64>) -> Result<(), GprError>,
) -> Result<(), GprError> {
    let (first, rest) = terms
        .split_first()
        .ok_or(GprError::CoordGradientUnsupported)?;
    eval(first, out.as_mut())?;
    if rest.is_empty() {
        return Ok(());
    }
    let mut scratch = Mat::zeros(out.nrows(), out.ncols());
    for term in rest {
        eval(term, scratch.as_mut())?;
        super::apply::add_rect(out.as_mut(), scratch.as_ref());
    }
    Ok(())
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

pub(super) fn require_diag_len(x: MatRef<'_, f64>, out: &[f64]) -> Result<(), GprError> {
    if x.nrows() == 0 || x.ncols() == 0 {
        return Err(GprError::EmptyInput);
    }
    if out.len() != x.nrows() {
        return Err(GprError::InvalidHyperparameter {
            reason: format!("expected {} diagonal entries, got {}", x.nrows(), out.len()),
        });
    }
    Ok(())
}

pub(super) fn broadcast_self_diag(
    x: MatRef<'_, f64>,
    out: &mut [f64],
    eval: impl FnOnce(MatRef<'_, f64>, MatMut<'_, f64>) -> Result<(), GprError>,
) -> Result<(), GprError> {
    require_diag_len(x, out)?;
    let one = x.submatrix(0, 0, 1, x.ncols());
    let mut cell = Mat::zeros(1, 1);
    eval(one.as_ref(), cell.as_mut())?;
    out.fill(cell[(0, 0)]);
    Ok(())
}

pub(super) fn scale_by_other_diags(
    terms: &[CompiledKernel],
    skip_a: usize,
    skip_b: Option<usize>,
    x: MatRef<'_, f64>,
    out: &mut [f64],
) -> Result<(), GprError> {
    let mut tmp = vec![0.0; out.len()];
    for (index, term) in terms.iter().enumerate() {
        if index == skip_a || Some(index) == skip_b {
            continue;
        }
        term.fill_diag_points(x, &mut tmp)?;
        for (dst, src) in out.iter_mut().zip(tmp.iter()) {
            *dst *= *src;
        }
    }
    Ok(())
}

fn product_grad_diag(
    terms: &[CompiledKernel],
    x: MatRef<'_, f64>,
    out: &mut [f64],
    param_idx: usize,
) -> Result<(), GprError> {
    let mut offset = 0;
    let mut found = None;
    for (index, term) in terms.iter().enumerate() {
        let count = term.num_params();
        if param_idx < offset + count {
            found = Some((index, param_idx - offset));
            break;
        }
        offset += count;
    }
    let (owner, local) = found.ok_or_else(|| GprError::InvalidHyperparameter {
        reason: format!("kernel parameter index {param_idx} is out of range"),
    })?;
    terms[owner].grad_diag_points(x, out, local)?;
    scale_by_other_diags(terms, owner, None, x, out)
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
