use super::apply::{combine_diag, mul_assign};
use super::coord;
use super::grad::{
    ProductBuffers, broadcast_self_diag, eval_cell, mul_fold, product_with_owner, require_diag_len,
    scale_by_other_diags, term_index_for_param,
};
use super::{
    CompiledKernel, CrossViews, LeafRef, MixedKernelViews, Nested, Term, ard_needs_coords,
    require_scratch_shape, term_scratch,
};
use crate::error::GprError;
use crate::kernel::KernelScalar;
use crate::kernel::dist::ArdSqDiff;
use crate::kernel::tree::Supply;
use crate::kernel::{Triangle, visit_triangle};
use faer::{MatMut, MatRef};

impl<T: KernelScalar> CompiledKernel<T> {
    /// Writes `∂²K/∂θ_i ∂θ_j` from squared distances into `d2_k`.
    ///
    /// Product trees need `scratch` the same shape as `d2_k`. Leaves ignore it.
    ///
    /// A sum / product nested in another allocates one output-shaped buffer
    /// per nesting level for the call.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::IndexOutOfRange`] if `i` or `j` is out of
    /// range, [`GprError::WorkspaceTooSmall`] if a product tree's `scratch` is
    /// the wrong size, or the same shape errors as [`Self::apply`].
    ///
    /// See the example on [`CompiledKernel`].
    pub fn hess<M: crate::math::KernelMath>(
        &self,
        dist: MatRef<'_, T>,
        d2_k: MatMut<'_, T>,
        i: usize,
        j: usize,
        uplo: Triangle,
        scratch: MatMut<'_, T>,
    ) -> Result<(), GprError> {
        let mut nested = self.nested_buffers(d2_k.nrows(), d2_k.ncols());
        self.hess_with::<M>(dist, d2_k, (i, j), uplo, scratch, &mut nested)
    }

    /// [`Self::hess`] with caller-owned [`Nested`] levels.
    pub(crate) fn hess_with<M: crate::math::KernelMath>(
        &self,
        dist: MatRef<'_, T>,
        mut d2_k: MatMut<'_, T>,
        pair: (usize, usize),
        uplo: Triangle,
        scratch: MatMut<'_, T>,
        nested: &mut Nested<T>,
    ) -> Result<(), GprError> {
        let (i, j) = pair;
        match self.term() {
            Term::Leaf(leaf) => leaf.hess_dist::<M>(dist, d2_k, pair, uplo),
            Term::Supplied(never) => match *never {},
            Term::Sum(terms) => match owners_for_pair(terms, i, j)? {
                PairOwners::Same {
                    term,
                    local_i,
                    local_j,
                } => term.hess_with::<M>(dist, d2_k, (local_i, local_j), uplo, scratch, nested),
                PairOwners::Distinct { .. } => {
                    zero_triangle(d2_k.as_mut(), uplo);
                    Ok(())
                }
            },
            Term::Product(terms) => {
                require_scratch_shape(d2_k.as_ref(), scratch.as_ref())?;
                let buffers = ProductBuffers::new(d2_k, scratch, nested, uplo);
                product_hess(
                    terms,
                    (i, j),
                    buffers,
                    |term, out, scratch, nested| {
                        term.apply_with::<M>(dist, out, uplo, scratch, nested)
                    },
                    |term, out, pair, scratch, nested| {
                        term.hess_with::<M>(dist, out, pair, uplo, scratch, nested)
                    },
                    |term, out, param, scratch, nested| {
                        term.grad_with::<M>(dist, out, param, uplo, scratch, nested)
                    },
                )
            }
        }
    }

    /// Writes `∂²K/∂θ_i ∂θ_j` from point coordinates.
    ///
    /// A sum / product nested in another allocates one output-shaped buffer
    /// per nesting level for the call.
    ///
    /// # Errors
    ///
    /// Same as [`Self::hess`], with coordinates in place of distances.
    ///
    /// See the example on [`CompiledKernel`].
    pub fn hess_points<M: crate::math::KernelMath>(
        &self,
        x: MatRef<'_, T>,
        d2_k: MatMut<'_, T>,
        i: usize,
        j: usize,
        uplo: Triangle,
        scratch: MatMut<'_, T>,
    ) -> Result<(), GprError> {
        self.require_columns(x)?;
        let mut nested = self.nested_buffers(d2_k.nrows(), d2_k.ncols());
        self.hess_points_with::<M>(x, d2_k, (i, j), uplo, scratch, &mut nested)
    }

    /// [`Self::hess_points`] with caller-owned [`Nested`] levels.
    pub(crate) fn hess_points_with<M: crate::math::KernelMath>(
        &self,
        x: MatRef<'_, T>,
        mut d2_k: MatMut<'_, T>,
        pair: (usize, usize),
        uplo: Triangle,
        scratch: MatMut<'_, T>,
        nested: &mut Nested<T>,
    ) -> Result<(), GprError> {
        let (i, j) = pair;
        match self.term() {
            Term::Leaf(leaf) => leaf.hess_points::<M>(x, d2_k, pair, uplo),
            Term::Supplied(never) => match *never {},
            Term::Sum(terms) => match owners_for_pair(terms, i, j)? {
                PairOwners::Same {
                    term,
                    local_i,
                    local_j,
                } => term.hess_points_with::<M>(x, d2_k, (local_i, local_j), uplo, scratch, nested),
                PairOwners::Distinct { .. } => {
                    zero_triangle(d2_k.as_mut(), uplo);
                    Ok(())
                }
            },
            Term::Product(terms) => {
                require_scratch_shape(d2_k.as_ref(), scratch.as_ref())?;
                let buffers = ProductBuffers::new(d2_k, scratch, nested, uplo);
                product_hess(
                    terms,
                    (i, j),
                    buffers,
                    |term, out, scratch, nested| {
                        term.apply_points_with::<M>(x, out, uplo, scratch, nested)
                    },
                    |term, out, pair, scratch, nested| {
                        term.hess_points_with::<M>(x, out, pair, uplo, scratch, nested)
                    },
                    |term, out, param, scratch, nested| {
                        term.grad_points_with::<M>(x, out, param, uplo, scratch, nested)
                    },
                )
            }
        }
    }

    // The cache and `x` views, output, pair, triangle, scratch, and levels.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn hess_from_ard_cache<M: crate::math::KernelMath>(
        &self,
        cache: ArdSqDiff<'_, T>,
        x: MatRef<'_, T>,
        d2_k: MatMut<'_, T>,
        pair: (usize, usize),
        uplo: Triangle,
        scratch: MatMut<'_, T>,
        nested: &mut Nested<T>,
    ) -> Result<(), GprError> {
        let (i, j) = pair;
        match self.term() {
            Term::Leaf(leaf) => leaf.hess_ard_cache::<M>(cache, x, d2_k, pair, uplo),
            Term::Supplied(never) => match *never {},
            Term::Sum(terms) => match owners_for_pair(terms, i, j)? {
                PairOwners::Same {
                    term,
                    local_i,
                    local_j,
                } => term.hess_from_ard_cache::<M>(
                    cache,
                    x,
                    d2_k,
                    (local_i, local_j),
                    uplo,
                    scratch,
                    nested,
                ),
                PairOwners::Distinct { .. } => {
                    zero_triangle(d2_k, uplo);
                    Ok(())
                }
            },
            Term::Product(_) => self.hess_points_with::<M>(x, d2_k, pair, uplo, scratch, nested),
        }
    }
}

impl<T: KernelScalar, S: Supply> CompiledKernel<T, S> {
    /// Writes `∂²k(x_r, x_r)/∂θ_i ∂θ_j` into `out[r]`.
    ///
    /// The off-diagonal Gram Hessian is not formed.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::EmptyInput`] if `x` is empty,
    /// [`GprError::IndexOutOfRange`] if `i` or `j` is out of range or
    /// `out.len()` is not `x.nrows()`, or the leaf error for that diagonal entry.
    pub(crate) fn hess_diag_points<M: crate::math::KernelMath>(
        &self,
        x: MatRef<'_, T>,
        out: &mut [T],
        i: usize,
        j: usize,
    ) -> Result<(), GprError> {
        match self {
            Self::Linear(leaf) => {
                require_diag_len(x, out)?;
                let one = x.submatrix(0, 0, 1, x.ncols());
                eval_cell(|cell| leaf.hess(one, cell, i, j, Triangle::Lower))?;
                leaf.fill_diag_points(x, out)
            }
            Self::Rbf(leaf) => broadcast_self_diag(x, out, |one, cell| {
                leaf.hess_from_coords::<M, _>(one, cell, i, j, Triangle::Lower)
            }),
            Self::Matern(leaf) => broadcast_self_diag(x, out, |one, cell| {
                leaf.hess_from_coords::<M, _>(one, cell, i, j, Triangle::Lower)
            }),
            Self::Periodic(leaf) => broadcast_self_diag(x, out, |one, cell| {
                leaf.hess_from_coords::<M, _>(one, cell, i, j, Triangle::Lower)
            }),
            Self::RationalQuadratic(leaf) => broadcast_self_diag(x, out, |one, cell| {
                leaf.hess_from_coords(one, cell, i, j, Triangle::Lower)
            }),
            Self::Custom(leaf) => custom_hess_diag(leaf, x, out, i, j),
            Self::RbfArd(leaf) => broadcast_self_diag(x, out, |one, cell| {
                leaf.hess_math::<M, _>(one, cell, i, j, Triangle::Lower)
            }),
            Self::MaternArd(leaf) => broadcast_self_diag(x, out, |one, cell| {
                leaf.hess_math::<M, _>(one, cell, i, j, Triangle::Lower)
            }),
            Self::RationalQuadraticArd(leaf) => broadcast_self_diag(x, out, |one, cell| {
                leaf.hess(one, cell, i, j, Triangle::Lower)
            }),
            Self::Constant(leaf) => broadcast_self_diag(x, out, |one, cell| {
                leaf.hess_rows(one, cell, i, j, Triangle::Lower)
            }),
            Self::White(leaf) => broadcast_self_diag(x, out, |one, cell| {
                leaf.hess_rows(one, cell, i, j, Triangle::Lower)
            }),
            Self::Supplied(leaf) => {
                require_diag_len(x, out)?;
                S::compiled(leaf).hess_diag::<M>(out, (i, j))
            }
            Self::Sum(terms) => match owners_for_pair(terms, i, j)? {
                PairOwners::Same {
                    term,
                    local_i,
                    local_j,
                } => term.hess_diag_points::<M>(x, out, local_i, local_j),
                PairOwners::Distinct { .. } => {
                    require_diag_len(x, out)?;
                    out.fill(T::from_f64(0.0));
                    Ok(())
                }
            },
            Self::Product(terms) => product_hess_diag::<M, _, _>(terms, x, out, i, j),
        }
    }

    pub(crate) fn hess_mixed<M: crate::math::KernelMath>(
        &self,
        views: MixedKernelViews<'_, T>,
        mut d2_k: MatMut<'_, T>,
        pair: (usize, usize),
        uplo: Triangle,
        scratch: MatMut<'_, T>,
        nested: &mut Nested<T>,
    ) -> Result<(), GprError> {
        let (i, j) = pair;
        match self.term() {
            Term::Leaf(leaf) => leaf.hess_mixed::<M>(views, d2_k, pair, uplo),
            Term::Supplied(leaf) => S::compiled(leaf).hess::<M>(views.slots, d2_k, pair, uplo),
            Term::Sum(terms) => match owners_for_pair(terms, i, j)? {
                PairOwners::Same {
                    term,
                    local_i,
                    local_j,
                } => term.hess_mixed::<M>(views, d2_k, (local_i, local_j), uplo, scratch, nested),
                PairOwners::Distinct { .. } => {
                    zero_triangle(d2_k.as_mut(), uplo);
                    Ok(())
                }
            },
            Term::Product(terms) => {
                require_scratch_shape(d2_k.as_ref(), scratch.as_ref())?;
                let buffers = ProductBuffers::new(d2_k, scratch, nested, uplo);
                product_hess(
                    terms,
                    pair,
                    buffers,
                    |term, out, scratch, nested| {
                        term.apply_mixed::<M>(views, out, uplo, scratch, nested)
                    },
                    |term, out, pair, scratch, nested| {
                        term.hess_mixed::<M>(views, out, pair, uplo, scratch, nested)
                    },
                    |term, out, param, scratch, nested| {
                        term.grad_mixed::<M>(views, out, param, uplo, scratch, nested)
                    },
                )
            }
        }
    }

    /// `∂²K(x1, x2)/∂θ_i ∂θ_j` of a rectangular block
    /// (`x1.nrows() × x2.nrows()`) from coordinates: every built-in leaf, and
    /// Sum / Product trees of them. A `Custom` leaf has no rectangular
    /// derivative and returns [`GprError::CoordGradientUnsupported`].
    ///
    /// Product trees need `scratch` the same shape as `d2_k` and distinct
    /// from it; leaves ignore it.
    #[cfg(test)]
    pub(crate) fn hess_cross_points<M: crate::math::KernelMath>(
        &self,
        x1: MatRef<'_, T>,
        x2: MatRef<'_, T>,
        d2_k: MatMut<'_, T>,
        i: usize,
        j: usize,
        scratch: MatMut<'_, T>,
    ) -> Result<(), GprError> {
        let mut nested = self.nested_buffers(d2_k.nrows(), d2_k.ncols());
        self.hess_cross_points_with::<M>(x1, x2, d2_k, (i, j), scratch, &mut nested)
    }

    /// [`Self::hess_cross_points`] with caller-owned [`Nested`] levels.
    #[cfg(test)]
    pub(crate) fn hess_cross_points_with<M: crate::math::KernelMath>(
        &self,
        x1: MatRef<'_, T>,
        x2: MatRef<'_, T>,
        d2_k: MatMut<'_, T>,
        pair: (usize, usize),
        scratch: MatMut<'_, T>,
        nested: &mut Nested<T>,
    ) -> Result<(), GprError> {
        self.hess_cross_views::<M>(CrossViews::points(x1, x2), d2_k, pair, scratch, nested)
    }

    /// The rectangular `∂²K/∂θ_i ∂θ_j` of the block `views` describes:
    /// coordinate leaves from coordinates, supplied leaves from their block.
    pub(crate) fn hess_cross_views<M: crate::math::KernelMath>(
        &self,
        views: CrossViews<'_, T>,
        mut d2_k: MatMut<'_, T>,
        pair: (usize, usize),
        scratch: MatMut<'_, T>,
        nested: &mut Nested<T>,
    ) -> Result<(), GprError> {
        let (i, j) = pair;
        let CrossViews { x1, x2, .. } = views;
        match self {
            Self::Rbf(leaf) => leaf.hess_cross_from_coords::<M, _>(x1, x2, d2_k, i, j),
            Self::Matern(leaf) => leaf.hess_cross_from_coords::<M, _>(x1, x2, d2_k, i, j),
            Self::RbfArd(leaf) => leaf.hess_cross_from_coords::<M, _>(x1, x2, d2_k, i, j),
            Self::MaternArd(leaf) => leaf.hess_cross_from_coords::<M, _>(x1, x2, d2_k, i, j),
            Self::Periodic(leaf) => leaf.hess_cross_from_coords::<M, _>(x1, x2, d2_k, i, j),
            Self::RationalQuadratic(leaf) => leaf.hess_cross_from_coords(x1, x2, d2_k, i, j),
            Self::RationalQuadraticArd(leaf) => leaf.hess_cross_from_coords(x1, x2, d2_k, i, j),
            Self::Linear(leaf) => leaf.hess_cross(x1, x2, d2_k, i, j),
            Self::Constant(leaf) => leaf.hess_cross_points(x1, x2, d2_k, i, j),
            Self::White(_) => super::grad::white_cross_zero(x1, x2, d2_k),
            Self::Custom(leaf) => coord::custom_cross_hess(leaf, x1, x2, d2_k, (i, j)),
            Self::Supplied(leaf) => S::compiled(leaf).hess_cross::<M>(views.slots, d2_k, (i, j)),
            Self::Sum(terms) => match owners_for_pair(terms, i, j)? {
                PairOwners::Same {
                    term,
                    local_i,
                    local_j,
                } => term.hess_cross_views::<M>(views, d2_k, (local_i, local_j), scratch, nested),
                PairOwners::Distinct { .. } => {
                    for col in 0..d2_k.ncols() {
                        for row in 0..d2_k.nrows() {
                            d2_k[(row, col)] = T::from_f64(0.0);
                        }
                    }
                    Ok(())
                }
            },
            Self::Product(terms) => {
                require_scratch_shape(d2_k.as_ref(), scratch.as_ref())?;
                let buffers = ProductBuffers::rect(d2_k, scratch, nested);
                product_hess(
                    terms,
                    (i, j),
                    buffers,
                    |term, out, scratch, nested| {
                        term.apply_cross_mixed::<M>(views, out, scratch, nested)
                    },
                    |term, out, pair, scratch, nested| {
                        term.hess_cross_views::<M>(views, out, pair, scratch, nested)
                    },
                    |term, out, param, scratch, nested| {
                        term.grad_cross_views::<M>(views, out, param, scratch, nested)
                    },
                )
            }
        }
    }
}

/// The leaf arms of the coordinate `∂²K/∂θ_i ∂θ_j` paths.
impl<T: KernelScalar> LeafRef<'_, T> {
    fn hess_dist<M: crate::math::KernelMath>(
        self,
        dist: MatRef<'_, T>,
        d2_k: MatMut<'_, T>,
        (i, j): (usize, usize),
        uplo: Triangle,
    ) -> Result<(), GprError> {
        match self {
            Self::Rbf(leaf) => leaf.hess_math::<M, _>(dist, d2_k, i, j, uplo),
            Self::RbfArd(_)
            | Self::Linear(_)
            | Self::MaternArd(_)
            | Self::RationalQuadraticArd(_) => Err(ard_needs_coords()),
            Self::Matern(leaf) => leaf.hess_math::<M, _>(dist, d2_k, i, j, uplo),
            Self::Periodic(leaf) => leaf.hess_math::<M, _>(dist, d2_k, i, j, uplo),
            Self::RationalQuadratic(leaf) => leaf.hess(dist, d2_k, i, j, uplo),
            Self::Constant(leaf) => leaf.hess(dist, d2_k, i, j, uplo),
            Self::White(leaf) => leaf.hess(dist, d2_k, i, j, uplo),
            Self::Custom(leaf) => leaf.hess(dist, d2_k, i, j, uplo),
        }
    }

    fn hess_points<M: crate::math::KernelMath>(
        self,
        x: MatRef<'_, T>,
        d2_k: MatMut<'_, T>,
        (i, j): (usize, usize),
        uplo: Triangle,
    ) -> Result<(), GprError> {
        match self {
            Self::Rbf(leaf) => leaf.hess_from_coords::<M, _>(x, d2_k, i, j, uplo),
            Self::Matern(leaf) => leaf.hess_from_coords::<M, _>(x, d2_k, i, j, uplo),
            Self::Periodic(leaf) => leaf.hess_from_coords::<M, _>(x, d2_k, i, j, uplo),
            Self::RationalQuadratic(leaf) => leaf.hess_from_coords(x, d2_k, i, j, uplo),
            Self::Custom(leaf) => leaf.hess_points(x, d2_k, i, j, uplo),
            Self::RbfArd(leaf) => leaf.hess_math::<M, _>(x, d2_k, i, j, uplo),
            Self::Linear(leaf) => leaf.hess(x, d2_k, i, j, uplo),
            Self::MaternArd(leaf) => leaf.hess_math::<M, _>(x, d2_k, i, j, uplo),
            Self::RationalQuadraticArd(leaf) => leaf.hess(x, d2_k, i, j, uplo),
            Self::Constant(leaf) => leaf.hess_rows(x, d2_k, i, j, uplo),
            Self::White(leaf) => leaf.hess_rows(x, d2_k, i, j, uplo),
        }
    }

    /// From the raw `(Δx_d)²` cache for an ARD leaf; any other leaf from
    /// coordinates.
    fn hess_ard_cache<M: crate::math::KernelMath>(
        self,
        cache: ArdSqDiff<'_, T>,
        x: MatRef<'_, T>,
        d2_k: MatMut<'_, T>,
        pair: (usize, usize),
        uplo: Triangle,
    ) -> Result<(), GprError> {
        let (i, j) = pair;
        match self {
            Self::RbfArd(leaf) => leaf.hess_from_sq_diff::<M, _>(cache, d2_k, i, j, uplo),
            Self::MaternArd(leaf) => leaf.hess_from_sq_diff::<M, _>(cache, d2_k, i, j, uplo),
            Self::RationalQuadraticArd(leaf) => leaf.hess_from_sq_diff(cache, d2_k, i, j, uplo),
            Self::Constant(leaf) => leaf.hess_rows(x, d2_k, i, j, uplo),
            Self::White(leaf) => leaf.hess_rows(x, d2_k, i, j, uplo),
            _ => self.hess_points::<M>(x, d2_k, pair, uplo),
        }
    }

    /// In the leaf's own mode (see `apply_mixed`).
    fn hess_mixed<M: crate::math::KernelMath>(
        self,
        views: MixedKernelViews<'_, T>,
        d2_k: MatMut<'_, T>,
        pair: (usize, usize),
        uplo: Triangle,
    ) -> Result<(), GprError> {
        if self.reads_dist() {
            return match views.dist {
                Some(dist) => self.hess_dist::<M>(dist, d2_k, pair, uplo),
                None => self.hess_points::<M>(views.x, d2_k, pair, uplo),
            };
        }
        match views.ard_cache.filter(|_| self.needs_ard_sq_diff()) {
            Some(cache) => self.hess_ard_cache::<M>(cache, views.x, d2_k, pair, uplo),
            None => self.hess_points::<M>(views.x, d2_k, pair, uplo),
        }
    }
}

enum PairOwners<'a, T: KernelScalar, S: Supply> {
    Same {
        term: &'a CompiledKernel<T, S>,
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

fn owners_for_pair<T: KernelScalar, S: Supply>(
    terms: &[CompiledKernel<T, S>],
    i: usize,
    j: usize,
) -> Result<PairOwners<'_, T, S>, GprError> {
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

fn zero_triangle<T: KernelScalar>(mut out: MatMut<'_, T>, uplo: Triangle) {
    visit_triangle(out.nrows(), uplo, |row, col| {
        out[(row, col)] = T::from_f64(0.0);
    });
}

fn custom_hess_diag<T: KernelScalar>(
    leaf: &crate::kernel::CustomKernel<T>,
    x: MatRef<'_, T>,
    out: &mut [T],
    i: usize,
    j: usize,
) -> Result<(), GprError> {
    require_diag_len(x, out)?;
    let width = x.ncols();
    for (row, slot) in out.iter_mut().enumerate() {
        let one = x.submatrix(row, 0, 1, width);
        *slot = eval_cell(|cell| leaf.hess_points(one, cell, i, j, Triangle::Lower))?;
    }
    Ok(())
}

fn product_hess_diag<M: crate::math::KernelMath, T: KernelScalar, S: Supply>(
    terms: &[CompiledKernel<T, S>],
    x: MatRef<'_, T>,
    out: &mut [T],
    i: usize,
    j: usize,
) -> Result<(), GprError> {
    match owners_for_pair(terms, i, j)? {
        PairOwners::Same {
            local_i, local_j, ..
        } => {
            let (owner, _) = term_index_for_param(terms, i)?;
            terms[owner].hess_diag_points::<M>(x, out, local_i, local_j)?;
            scale_by_other_diags(terms, owner, None, x, out)
        }
        PairOwners::Distinct {
            owner_i,
            local_i,
            owner_j,
            local_j,
        } => {
            terms[owner_i].grad_diag_points::<M>(x, out, local_i)?;
            combine_diag(out, mul_assign, |start, block| {
                let rows = x.subrows(start, block.len());
                terms[owner_j].grad_diag_points::<M>(rows, block, local_j)
            })?;
            scale_by_other_diags(terms, owner_i, Some(owner_j), x, out)
        }
    }
}

/// `∂²/∂θ_i ∂θ_j` of a product: one term's Hessian, or two terms' gradients,
/// times the other terms. The writers get `(term, out, …, scratch, nested)`.
fn product_hess<T: KernelScalar, S: Supply>(
    terms: &[CompiledKernel<T, S>],
    pair: (usize, usize),
    buffers: ProductBuffers<'_, '_, T>,
    apply: impl FnMut(
        &CompiledKernel<T, S>,
        MatMut<'_, T>,
        MatMut<'_, T>,
        &mut Nested<T>,
    ) -> Result<(), GprError>,
    mut hess: impl FnMut(
        &CompiledKernel<T, S>,
        MatMut<'_, T>,
        (usize, usize),
        MatMut<'_, T>,
        &mut Nested<T>,
    ) -> Result<(), GprError>,
    grad: impl FnMut(
        &CompiledKernel<T, S>,
        MatMut<'_, T>,
        usize,
        MatMut<'_, T>,
        &mut Nested<T>,
    ) -> Result<(), GprError>,
) -> Result<(), GprError> {
    let (i, j) = pair;
    match owners_for_pair(terms, i, j)? {
        PairOwners::Same {
            local_i, local_j, ..
        } => {
            let (owner, _) = term_index_for_param(terms, i)?;
            product_with_owner(
                terms,
                owner,
                buffers,
                apply,
                |term, out, scratch, nested| hess(term, out, (local_i, local_j), scratch, nested),
            )
        }
        PairOwners::Distinct {
            owner_i,
            local_i,
            owner_j,
            local_j,
        } => product_cross_leaf(
            terms,
            (owner_i, local_i),
            (owner_j, local_j),
            buffers,
            apply,
            grad,
        ),
    }
}

/// `∂K_a/∂θ_i · ∂K_b/∂θ_j · ∏_{k ∉ {a, b}} K_k` for owners `(a, i)` and `(b, j)`.
fn product_cross_leaf<T: KernelScalar, S: Supply>(
    terms: &[CompiledKernel<T, S>],
    (owner_i, local_i): (usize, usize),
    (owner_j, local_j): (usize, usize),
    buffers: ProductBuffers<'_, '_, T>,
    mut apply: impl FnMut(
        &CompiledKernel<T, S>,
        MatMut<'_, T>,
        MatMut<'_, T>,
        &mut Nested<T>,
    ) -> Result<(), GprError>,
    mut grad: impl FnMut(
        &CompiledKernel<T, S>,
        MatMut<'_, T>,
        usize,
        MatMut<'_, T>,
        &mut Nested<T>,
    ) -> Result<(), GprError>,
) -> Result<(), GprError> {
    let ProductBuffers {
        mut out,
        mut scratch,
        nested,
        fold,
    } = buffers;
    let (rows, cols) = (out.nrows(), out.ncols());
    let mut started = false;
    for (k, term) in terms.iter().enumerate() {
        if k == owner_i || k == owner_j {
            continue;
        }
        if started {
            let (own, deeper) = term_scratch(term, rows, cols, out.as_mut(), &mut *nested)?;
            apply(term, scratch.as_mut(), own, deeper)?;
            mul_fold(out.as_mut(), scratch.as_ref(), fold);
        } else {
            apply(term, out.as_mut(), scratch.as_mut(), &mut *nested)?;
            started = true;
        }
    }
    let term_i = &terms[owner_i];
    if started {
        let (own, deeper) = term_scratch(term_i, rows, cols, out.as_mut(), &mut *nested)?;
        grad(term_i, scratch.as_mut(), local_i, own, deeper)?;
        mul_fold(out.as_mut(), scratch.as_ref(), fold);
    } else {
        grad(
            term_i,
            out.as_mut(),
            local_i,
            scratch.as_mut(),
            &mut *nested,
        )?;
    }
    let term_j = &terms[owner_j];
    let (own, deeper) = term_scratch(term_j, rows, cols, out.as_mut(), nested)?;
    grad(term_j, scratch.as_mut(), local_j, own, deeper)?;
    mul_fold(out.as_mut(), scratch.as_ref(), fold);
    Ok(())
}
