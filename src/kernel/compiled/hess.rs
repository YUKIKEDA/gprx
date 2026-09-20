use super::grad::write_product_grad;
use super::{
    CompiledKernel, MixedKernelViews, ard_needs_coords, mul_triangle, require_scratch_shape,
};
use crate::error::GprError;
use crate::kernel::{Triangle, visit_triangle};
use faer::{Mat, MatMut, MatRef};

impl CompiledKernel {
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
