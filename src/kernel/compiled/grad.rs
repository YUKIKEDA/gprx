use super::apply::{combine_diag, mul_assign};
use super::coord::{self, Dir};
use super::{
    CompiledKernel, CrossViews, LeafRef, MixedKernelViews, Nested, Term, ard_needs_coords,
    mul_triangle, require_scratch_shape, term_scratch,
};
use crate::error::GprError;
use crate::kernel::KernelScalar;
use crate::kernel::dist::ArdSqDiff;
use crate::kernel::radial;
use crate::kernel::tree::Supply;
use crate::kernel::{CustomKernel, Triangle, write_square_from_coords};
use faer::{Mat, MatMut, MatRef};

impl<T: KernelScalar> CompiledKernel<T> {
    /// Writes `∂K/∂θ_{param_idx}` into `d_k`.
    ///
    /// Product trees need `scratch` the same shape as `d_k` and distinct from
    /// it. Leaves ignore `scratch`.
    ///
    /// A sum / product nested in another allocates one output-shaped buffer
    /// per nesting level for the call.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::IndexOutOfRange`] if `param_idx` is out of
    /// range, [`GprError::WorkspaceTooSmall`] if a product tree's `scratch` is
    /// the wrong size, or the same shape errors as [`Self::apply`].
    ///
    /// See the example on [`CompiledKernel`].
    pub fn grad<M: crate::math::KernelMath>(
        &self,
        dist: MatRef<'_, T>,
        d_k: MatMut<'_, T>,
        param_idx: usize,
        uplo: Triangle,
        scratch: MatMut<'_, T>,
    ) -> Result<(), GprError> {
        let mut nested = self.nested_buffers(d_k.nrows(), d_k.ncols());
        self.grad_with::<M>(dist, d_k, param_idx, uplo, scratch, &mut nested)
    }

    /// [`Self::grad`] with caller-owned [`Nested`] levels.
    pub(crate) fn grad_with<M: crate::math::KernelMath>(
        &self,
        dist: MatRef<'_, T>,
        d_k: MatMut<'_, T>,
        param_idx: usize,
        uplo: Triangle,
        scratch: MatMut<'_, T>,
        nested: &mut Nested<T>,
    ) -> Result<(), GprError> {
        match self.term() {
            Term::Leaf(leaf) => leaf.grad_dist::<M>(dist, d_k, param_idx, uplo),
            Term::Supplied(never) => match *never {},
            Term::Sum(terms) => {
                let (term, local) = term_for_param(terms, param_idx)?;
                term.grad_with::<M>(dist, d_k, local, uplo, scratch, nested)
            }
            Term::Product(terms) => {
                require_scratch_shape(d_k.as_ref(), scratch.as_ref())?;
                let (owner, local) = term_index_for_param(terms, param_idx)?;
                product_with_owner(
                    terms,
                    owner,
                    ProductBuffers::new(d_k, scratch, nested, uplo),
                    |term, out, scratch, nested| {
                        term.apply_with::<M>(dist, out, uplo, scratch, nested)
                    },
                    |term, out, scratch, nested| {
                        term.grad_with::<M>(dist, out, local, uplo, scratch, nested)
                    },
                )
            }
        }
    }

    /// Writes `∂K/∂θ_{param_idx}` from point coordinates.
    ///
    /// Product trees and custom distance leaves need `scratch` the same shape
    /// as `d_k` and distinct from it; a custom leaf writes its distances
    /// there. Other leaves ignore `scratch`.
    ///
    /// A sum / product nested in another allocates one output-shaped buffer
    /// per nesting level for the call.
    ///
    /// # Errors
    ///
    /// [`GprError::IndexOutOfRange`] if `param_idx` is out of range,
    /// [`GprError::WorkspaceTooSmall`] if a product tree's or custom leaf's
    /// `scratch` is the wrong size, or the same shape errors as
    /// [`Self::apply_points`].
    ///
    /// See the example on [`CompiledKernel`].
    pub fn grad_points<M: crate::math::KernelMath>(
        &self,
        x: MatRef<'_, T>,
        d_k: MatMut<'_, T>,
        param_idx: usize,
        uplo: Triangle,
        scratch: MatMut<'_, T>,
    ) -> Result<(), GprError> {
        self.require_columns(x)?;
        let mut nested = self.nested_buffers(d_k.nrows(), d_k.ncols());
        self.grad_points_with::<M>(x, d_k, param_idx, uplo, scratch, &mut nested)
    }

    /// [`Self::grad_points`] with caller-owned [`Nested`] levels.
    pub(crate) fn grad_points_with<M: crate::math::KernelMath>(
        &self,
        x: MatRef<'_, T>,
        d_k: MatMut<'_, T>,
        param_idx: usize,
        uplo: Triangle,
        scratch: MatMut<'_, T>,
        nested: &mut Nested<T>,
    ) -> Result<(), GprError> {
        match self.term() {
            Term::Leaf(leaf) => leaf.grad_points::<M>(x, d_k, param_idx, uplo, scratch),
            Term::Supplied(never) => match *never {},
            Term::Sum(terms) => {
                let (term, local) = term_for_param(terms, param_idx)?;
                term.grad_points_with::<M>(x, d_k, local, uplo, scratch, nested)
            }
            Term::Product(terms) => {
                require_scratch_shape(d_k.as_ref(), scratch.as_ref())?;
                let (owner, local) = term_index_for_param(terms, param_idx)?;
                product_with_owner(
                    terms,
                    owner,
                    ProductBuffers::new(d_k, scratch, nested, uplo),
                    |term, out, scratch, nested| {
                        term.apply_points_with::<M>(x, out, uplo, scratch, nested)
                    },
                    |term, out, scratch, nested| {
                        term.grad_points_with::<M>(x, out, local, uplo, scratch, nested)
                    },
                )
            }
        }
    }

    // The cache and `x` views, output, index, triangle, scratch, and levels.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn grad_from_ard_cache<M: crate::math::KernelMath>(
        &self,
        cache: ArdSqDiff<'_, T>,
        x: MatRef<'_, T>,
        d_k: MatMut<'_, T>,
        param_idx: usize,
        uplo: Triangle,
        scratch: MatMut<'_, T>,
        nested: &mut Nested<T>,
    ) -> Result<(), GprError> {
        match self.term() {
            Term::Leaf(leaf) => leaf.grad_ard_cache::<M>(cache, x, d_k, param_idx, uplo, scratch),
            Term::Supplied(never) => match *never {},
            Term::Sum(terms) => {
                let (term, local) = term_for_param(terms, param_idx)?;
                term.grad_from_ard_cache::<M>(cache, x, d_k, local, uplo, scratch, nested)
            }
            Term::Product(_) => {
                self.grad_points_with::<M>(x, d_k, param_idx, uplo, scratch, nested)
            }
        }
    }

    /// Writes `∂K(X1, X2)/∂X2[*, dim]` into `d_k`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::CoordGradientUnsupported`] when a leaf does not
    /// implement coordinate derivatives (including Product trees).
    ///
    /// See the example on [`CompiledKernel`].
    pub fn grad_wrt_coord_dim<M: crate::math::KernelMath>(
        &self,
        x1: MatRef<'_, T>,
        x2: MatRef<'_, T>,
        d_k: MatMut<'_, T>,
        dim: usize,
    ) -> Result<(), GprError> {
        let mut scratch = Mat::zeros(d_k.nrows(), d_k.ncols());
        self.grad_wrt_coord_dim_with::<M>(x1, x2, d_k, dim, scratch.as_mut())
    }

    /// [`Self::grad_wrt_coord_dim`] with a caller-owned `scratch` (the shape
    /// of `d_k`, distinct from it) for the terms of a sum.
    pub(crate) fn grad_wrt_coord_dim_with<M: crate::math::KernelMath>(
        &self,
        x1: MatRef<'_, T>,
        x2: MatRef<'_, T>,
        mut d_k: MatMut<'_, T>,
        dim: usize,
        scratch: MatMut<'_, T>,
    ) -> Result<(), GprError> {
        match self {
            Self::Rbf(leaf) => radial::grad_wrt_coord_dim::<M, _>(leaf, x1, x2, d_k, dim),
            Self::Matern(leaf) => radial::grad_wrt_coord_dim::<M, _>(leaf, x1, x2, d_k, dim),
            Self::RbfArd(leaf) => radial::grad_wrt_coord_dim::<M, _>(leaf, x1, x2, d_k, dim),
            Self::MaternArd(leaf) => radial::grad_wrt_coord_dim::<M, _>(leaf, x1, x2, d_k, dim),
            Self::Periodic(leaf) => radial::grad_wrt_coord_dim::<M, _>(leaf, x1, x2, d_k, dim),
            Self::RationalQuadratic(leaf) => {
                radial::grad_wrt_coord_dim::<M, _>(leaf, x1, x2, d_k, dim)
            }
            Self::RationalQuadraticArd(leaf) => {
                radial::grad_wrt_coord_dim::<M, _>(leaf, x1, x2, d_k, dim)
            }
            Self::White(leaf) => leaf.grad_wrt_coord_dim(x1, x2, d_k, dim),
            Self::Constant(_) => coord::zero_block(x1, x2, d_k, &[dim]),
            Self::Linear(leaf) => coord::linear_grad_dim(leaf, x1, x2, d_k, dim),
            Self::Custom(leaf) => coord::custom_grad_dim(leaf, x1, x2, d_k, dim),
            Self::Supplied(never) => match *never {},
            Self::Sum(terms) => fold_coord_sum(terms, d_k.as_mut(), scratch, |term, dest| {
                term.grad_wrt_coord_dim_with::<M>(x1, x2, dest, dim, Mat::new().as_mut())
            }),
            Self::Product(terms) => {
                Self::product_coord::<M>(terms, x1, x2, d_k, (Dir::X2(dim), None))
            }
        }
    }

    /// `scratch` (the shape of `d2_k`, distinct from it) holds each term of a sum.
    pub(crate) fn hess_wrt_coord_dims<M: crate::math::KernelMath>(
        &self,
        x1: MatRef<'_, T>,
        x2: MatRef<'_, T>,
        mut d2_k: MatMut<'_, T>,
        dim_a: usize,
        dim_b: usize,
        scratch: MatMut<'_, T>,
    ) -> Result<(), GprError> {
        match self {
            Self::Rbf(leaf) => {
                radial::hess_wrt_coord::<M, _>(leaf, x1, x2, d2_k, (dim_a, dim_b), false)
            }
            Self::Matern(leaf) => {
                radial::hess_wrt_coord::<M, _>(leaf, x1, x2, d2_k, (dim_a, dim_b), false)
            }
            Self::RbfArd(leaf) => {
                radial::hess_wrt_coord::<M, _>(leaf, x1, x2, d2_k, (dim_a, dim_b), false)
            }
            Self::MaternArd(leaf) => {
                radial::hess_wrt_coord::<M, _>(leaf, x1, x2, d2_k, (dim_a, dim_b), false)
            }
            Self::Periodic(leaf) => {
                radial::hess_wrt_coord::<M, _>(leaf, x1, x2, d2_k, (dim_a, dim_b), false)
            }
            Self::RationalQuadratic(leaf) => {
                radial::hess_wrt_coord::<M, _>(leaf, x1, x2, d2_k, (dim_a, dim_b), false)
            }
            Self::RationalQuadraticArd(leaf) => {
                radial::hess_wrt_coord::<M, _>(leaf, x1, x2, d2_k, (dim_a, dim_b), false)
            }
            Self::White(leaf) => leaf.hess_wrt_coord_dims(x1, x2, d2_k, dim_a, dim_b),
            Self::Constant(_) => coord::zero_block(x1, x2, d2_k, &[dim_a, dim_b]),
            Self::Linear(_) => coord::zero_block(x1, x2, d2_k, &[dim_a, dim_b]),
            Self::Custom(leaf) => {
                coord::custom_hess_dims(leaf, x1, x2, d2_k, (dim_a, dim_b), false)
            }
            Self::Supplied(never) => match *never {},
            Self::Sum(terms) => fold_coord_sum(terms, d2_k.as_mut(), scratch, |term, dest| {
                term.hess_wrt_coord_dims::<M>(x1, x2, dest, dim_a, dim_b, Mat::new().as_mut())
            }),
            Self::Product(terms) => Self::product_coord::<M>(
                terms,
                x1,
                x2,
                d2_k,
                (Dir::X2(dim_a), Some(Dir::X2(dim_b))),
            ),
        }
    }

    /// `scratch` (the shape of `d2_k`, distinct from it) holds each term of a sum.
    pub(crate) fn hess_wrt_coord_mixed<M: crate::math::KernelMath>(
        &self,
        x1: MatRef<'_, T>,
        x2: MatRef<'_, T>,
        mut d2_k: MatMut<'_, T>,
        dim_x1: usize,
        dim_x2: usize,
        scratch: MatMut<'_, T>,
    ) -> Result<(), GprError> {
        match self {
            Self::Rbf(leaf) => {
                radial::hess_wrt_coord::<M, _>(leaf, x1, x2, d2_k, (dim_x1, dim_x2), true)
            }
            Self::Matern(leaf) => {
                radial::hess_wrt_coord::<M, _>(leaf, x1, x2, d2_k, (dim_x1, dim_x2), true)
            }
            Self::RbfArd(leaf) => {
                radial::hess_wrt_coord::<M, _>(leaf, x1, x2, d2_k, (dim_x1, dim_x2), true)
            }
            Self::MaternArd(leaf) => {
                radial::hess_wrt_coord::<M, _>(leaf, x1, x2, d2_k, (dim_x1, dim_x2), true)
            }
            Self::Periodic(leaf) => {
                radial::hess_wrt_coord::<M, _>(leaf, x1, x2, d2_k, (dim_x1, dim_x2), true)
            }
            Self::RationalQuadratic(leaf) => {
                radial::hess_wrt_coord::<M, _>(leaf, x1, x2, d2_k, (dim_x1, dim_x2), true)
            }
            Self::RationalQuadraticArd(leaf) => {
                radial::hess_wrt_coord::<M, _>(leaf, x1, x2, d2_k, (dim_x1, dim_x2), true)
            }
            Self::White(leaf) => leaf.hess_wrt_coord_mixed(x1, x2, d2_k, dim_x1, dim_x2),
            Self::Constant(_) => coord::zero_block(x1, x2, d2_k, &[dim_x1, dim_x2]),
            Self::Linear(leaf) => coord::linear_mixed(leaf, x1, x2, d2_k, (dim_x1, dim_x2)),
            Self::Custom(leaf) => {
                coord::custom_hess_dims(leaf, x1, x2, d2_k, (dim_x1, dim_x2), true)
            }
            Self::Supplied(never) => match *never {},
            Self::Sum(terms) => fold_coord_sum(terms, d2_k.as_mut(), scratch, |term, dest| {
                term.hess_wrt_coord_mixed::<M>(x1, x2, dest, dim_x1, dim_x2, Mat::new().as_mut())
            }),
            Self::Product(terms) => Self::product_coord::<M>(
                terms,
                x1,
                x2,
                d2_k,
                (Dir::X1(dim_x1), Some(Dir::X2(dim_x2))),
            ),
        }
    }

    pub(crate) fn hess_theta_coord_dim<M: crate::math::KernelMath>(
        &self,
        x1: MatRef<'_, T>,
        x2: MatRef<'_, T>,
        d2_k: MatMut<'_, T>,
        param_idx: usize,
        dim: usize,
    ) -> Result<(), GprError> {
        match self {
            Self::Rbf(leaf) => {
                radial::hess_theta_coord_dim::<M, _>(leaf, x1, x2, d2_k, param_idx, dim)
            }
            Self::Matern(leaf) => {
                radial::hess_theta_coord_dim::<M, _>(leaf, x1, x2, d2_k, param_idx, dim)
            }
            Self::RbfArd(leaf) => {
                radial::hess_theta_coord_dim::<M, _>(leaf, x1, x2, d2_k, param_idx, dim)
            }
            Self::MaternArd(leaf) => {
                radial::hess_theta_coord_dim::<M, _>(leaf, x1, x2, d2_k, param_idx, dim)
            }
            Self::Periodic(leaf) => {
                radial::hess_theta_coord_dim::<M, _>(leaf, x1, x2, d2_k, param_idx, dim)
            }
            Self::RationalQuadratic(leaf) => {
                radial::hess_theta_coord_dim::<M, _>(leaf, x1, x2, d2_k, param_idx, dim)
            }
            Self::RationalQuadraticArd(leaf) => {
                radial::hess_theta_coord_dim::<M, _>(leaf, x1, x2, d2_k, param_idx, dim)
            }
            Self::White(leaf) => leaf.hess_theta_coord_dim(x1, x2, d2_k, param_idx, dim),
            Self::Constant(leaf) => {
                coord::constant_param(leaf, param_idx)?;
                coord::zero_block(x1, x2, d2_k, &[dim])
            }
            Self::Linear(leaf) => {
                coord::linear_param(param_idx)?;
                coord::linear_grad_dim(leaf, x1, x2, d2_k, dim)
            }
            Self::Custom(leaf) => coord::custom_theta_dim(leaf, x1, x2, d2_k, param_idx, dim),
            Self::Supplied(never) => match *never {},
            Self::Sum(terms) => {
                let (term, local) = term_for_param(terms, param_idx)?;
                term.hess_theta_coord_dim::<M>(x1, x2, d2_k, local, dim)
            }
            Self::Product(terms) => Self::product_coord::<M>(
                terms,
                x1,
                x2,
                d2_k,
                (Dir::Theta(param_idx), Some(Dir::X2(dim))),
            ),
        }
    }
}

impl<T: KernelScalar, S: Supply> CompiledKernel<T, S> {
    /// Writes `∂k(x_i, x_i)/∂θ_{param_idx}` into `out[i]`.
    ///
    /// The off-diagonal Gram derivative is not formed.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::EmptyInput`] if `x` is empty,
    /// [`GprError::IndexOutOfRange`] if `param_idx` is out of range or
    /// `out.len()` is not `x.nrows()`, or the leaf error for that diagonal entry.
    pub(crate) fn grad_diag_points<M: crate::math::KernelMath>(
        &self,
        x: MatRef<'_, T>,
        out: &mut [T],
        param_idx: usize,
    ) -> Result<(), GprError> {
        match self {
            Self::Linear(leaf) => {
                require_diag_len(x, out)?;
                let one = x.submatrix(0, 0, 1, x.ncols());
                eval_cell(|cell| leaf.grad(one, cell, param_idx, Triangle::Lower))?;
                leaf.fill_diag_points(x, out)
            }
            Self::Rbf(leaf) => broadcast_self_diag(x, out, |one, cell| {
                leaf.grad_from_coords::<M, _>(one, cell, param_idx, Triangle::Lower)
            }),
            Self::Matern(leaf) => broadcast_self_diag(x, out, |one, cell| {
                leaf.grad_from_coords::<M, _>(one, cell, param_idx, Triangle::Lower)
            }),
            Self::Periodic(leaf) => broadcast_self_diag(x, out, |one, cell| {
                leaf.grad_from_coords::<M, _>(one, cell, param_idx, Triangle::Lower)
            }),
            Self::RationalQuadratic(leaf) => broadcast_self_diag(x, out, |one, cell| {
                leaf.grad_from_coords(one, cell, param_idx, Triangle::Lower)
            }),
            Self::Custom(leaf) => broadcast_self_diag(x, out, |one, cell| {
                let mut dist = [T::from_f64(0.0)];
                let dist = MatMut::from_column_major_slice_mut(&mut dist, 1, 1);
                grad_custom_from_coords(leaf, one, cell, param_idx, Triangle::Lower, dist)
            }),
            Self::RbfArd(leaf) => broadcast_self_diag(x, out, |one, cell| {
                leaf.grad_math::<M, _>(one, cell, param_idx, Triangle::Lower)
            }),
            Self::MaternArd(leaf) => broadcast_self_diag(x, out, |one, cell| {
                leaf.grad_math::<M, _>(one, cell, param_idx, Triangle::Lower)
            }),
            Self::RationalQuadraticArd(leaf) => broadcast_self_diag(x, out, |one, cell| {
                leaf.grad(one, cell, param_idx, Triangle::Lower)
            }),
            Self::Constant(leaf) => broadcast_self_diag(x, out, |one, cell| {
                leaf.grad_rows(one, cell, param_idx, Triangle::Lower)
            }),
            Self::White(leaf) => broadcast_self_diag(x, out, |one, cell| {
                leaf.grad_rows(one, cell, param_idx, Triangle::Lower)
            }),
            Self::Supplied(leaf) => {
                require_count_diag(x, out)?;
                S::compiled(leaf).grad_diag::<M>(out, param_idx)
            }
            Self::Sum(terms) => {
                let (term, local) = term_for_param(terms, param_idx)?;
                term.grad_diag_points::<M>(x, out, local)
            }
            Self::Product(terms) => product_grad_diag::<M, _, _>(terms, x, out, param_idx),
        }
    }

    /// Writes `∂K/∂θ` from a distance matrix and coordinates, one mode per leaf.
    pub(crate) fn grad_mixed<M: crate::math::KernelMath>(
        &self,
        views: MixedKernelViews<'_, T, S>,
        d_k: MatMut<'_, T>,
        param_idx: usize,
        uplo: Triangle,
        scratch: MatMut<'_, T>,
        nested: &mut Nested<T>,
    ) -> Result<(), GprError> {
        match self.term() {
            Term::Leaf(leaf) => leaf.grad_mixed::<M, _>(views, d_k, param_idx, uplo, scratch),
            Term::Supplied(leaf) => {
                let (leaf, slots) = S::square_leaf(leaf, views.slots);
                leaf.grad::<M>(slots, d_k, param_idx, uplo)
            }
            Term::Sum(terms) => {
                let (term, local) = term_for_param(terms, param_idx)?;
                term.grad_mixed::<M>(views, d_k, local, uplo, scratch, nested)
            }
            Term::Product(terms) => {
                require_scratch_shape(d_k.as_ref(), scratch.as_ref())?;
                let (owner, local) = term_index_for_param(terms, param_idx)?;
                product_with_owner(
                    terms,
                    owner,
                    ProductBuffers::new(d_k, scratch, nested, uplo),
                    |term, out, scratch, nested| {
                        term.apply_mixed::<M>(views, out, uplo, scratch, nested)
                    },
                    |term, out, scratch, nested| {
                        term.grad_mixed::<M>(views, out, local, uplo, scratch, nested)
                    },
                )
            }
        }
    }

    /// The rectangular `∂K/∂θ_{param_idx}` of the block `views` describes:
    /// coordinate leaves from coordinates, supplied leaves from their block.
    pub(crate) fn grad_cross_views<M: crate::math::KernelMath>(
        &self,
        views: CrossViews<'_, T, S>,
        d_k: MatMut<'_, T>,
        param_idx: usize,
        scratch: MatMut<'_, T>,
        nested: &mut Nested<T>,
    ) -> Result<(), GprError> {
        let CrossViews { x1, x2, .. } = views;
        match self {
            Self::Rbf(leaf) => leaf.grad_cross_from_coords::<M, _>(x1, x2, d_k, param_idx),
            Self::Matern(leaf) => leaf.grad_cross_from_coords::<M, _>(x1, x2, d_k, param_idx),
            Self::RbfArd(leaf) => leaf.grad_cross_from_coords::<M, _>(x1, x2, d_k, param_idx),
            Self::MaternArd(leaf) => leaf.grad_cross_from_coords::<M, _>(x1, x2, d_k, param_idx),
            Self::Periodic(leaf) => leaf.grad_cross_from_coords::<M, _>(x1, x2, d_k, param_idx),
            Self::RationalQuadratic(leaf) => leaf.grad_cross_from_coords(x1, x2, d_k, param_idx),
            Self::RationalQuadraticArd(leaf) => leaf.grad_cross_from_coords(x1, x2, d_k, param_idx),
            Self::Linear(leaf) => leaf.grad_cross(x1, x2, d_k, param_idx),
            Self::Constant(leaf) => leaf.grad_cross_points(x1, x2, d_k, param_idx),
            Self::White(_) => {
                let _ = param_idx;
                white_cross_zero(x1, x2, d_k)
            }
            Self::Custom(leaf) => coord::custom_cross_grad(leaf, x1, x2, d_k, param_idx),
            Self::Supplied(leaf) => {
                let (leaf, slots) = S::rect_leaf(leaf, views.slots);
                leaf.grad_cross::<M>(slots, d_k, param_idx)
            }
            Self::Sum(terms) => {
                let (term, local) = term_for_param(terms, param_idx)?;
                term.grad_cross_views::<M>(views, d_k, local, scratch, nested)
            }
            Self::Product(terms) => {
                require_scratch_shape(d_k.as_ref(), scratch.as_ref())?;
                let (owner, local) = term_index_for_param(terms, param_idx)?;
                product_with_owner(
                    terms,
                    owner,
                    ProductBuffers::rect(d_k, scratch, nested),
                    |term, out, scratch, nested| {
                        term.apply_cross_mixed::<M>(views, out, scratch, nested)
                    },
                    |term, out, scratch, nested| {
                        term.grad_cross_views::<M>(views, out, local, scratch, nested)
                    },
                )
            }
        }
    }
}

/// The leaf arms of the coordinate `∂K/∂θ` paths.
impl<T: KernelScalar> LeafRef<'_, T> {
    /// `∂k(dist)/∂θ_{param_idx}` for `uplo`.
    fn grad_dist<M: crate::math::KernelMath>(
        self,
        dist: MatRef<'_, T>,
        d_k: MatMut<'_, T>,
        param_idx: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        match self {
            Self::Rbf(leaf) => leaf.grad_math::<M, _>(dist, d_k, param_idx, uplo),
            Self::RbfArd(_)
            | Self::Linear(_)
            | Self::MaternArd(_)
            | Self::RationalQuadraticArd(_) => Err(ard_needs_coords()),
            Self::Matern(leaf) => leaf.grad_math::<M, _>(dist, d_k, param_idx, uplo),
            Self::Periodic(leaf) => leaf.grad_math::<M, _>(dist, d_k, param_idx, uplo),
            Self::RationalQuadratic(leaf) => leaf.grad(dist, d_k, param_idx, uplo),
            Self::Constant(leaf) => leaf.grad(dist, d_k, param_idx, uplo),
            Self::White(leaf) => leaf.grad(dist, d_k, param_idx, uplo),
            Self::Custom(leaf) => leaf.grad(dist, d_k, param_idx, uplo),
        }
    }

    /// `∂K/∂θ_{param_idx}` from coordinates. A custom leaf writes its
    /// distances into `scratch` first.
    fn grad_points<M: crate::math::KernelMath>(
        self,
        x: MatRef<'_, T>,
        d_k: MatMut<'_, T>,
        param_idx: usize,
        uplo: Triangle,
        scratch: MatMut<'_, T>,
    ) -> Result<(), GprError> {
        match self {
            Self::Rbf(leaf) => leaf.grad_from_coords::<M, _>(x, d_k, param_idx, uplo),
            Self::Matern(leaf) => leaf.grad_from_coords::<M, _>(x, d_k, param_idx, uplo),
            Self::Periodic(leaf) => leaf.grad_from_coords::<M, _>(x, d_k, param_idx, uplo),
            Self::RationalQuadratic(leaf) => leaf.grad_from_coords(x, d_k, param_idx, uplo),
            Self::Custom(leaf) => {
                require_scratch_shape(d_k.as_ref(), scratch.as_ref())?;
                grad_custom_from_coords(leaf, x, d_k, param_idx, uplo, scratch)
            }
            Self::RbfArd(leaf) => leaf.grad_math::<M, _>(x, d_k, param_idx, uplo),
            Self::Linear(leaf) => leaf.grad(x, d_k, param_idx, uplo),
            Self::MaternArd(leaf) => leaf.grad_math::<M, _>(x, d_k, param_idx, uplo),
            Self::RationalQuadraticArd(leaf) => leaf.grad(x, d_k, param_idx, uplo),
            Self::Constant(leaf) => leaf.grad_rows(x, d_k, param_idx, uplo),
            Self::White(leaf) => leaf.grad_rows(x, d_k, param_idx, uplo),
        }
    }

    /// `∂K/∂θ_{param_idx}` from the raw `(Δx_d)²` cache for an ARD leaf;
    /// any other leaf from coordinates.
    fn grad_ard_cache<M: crate::math::KernelMath>(
        self,
        cache: ArdSqDiff<'_, T>,
        x: MatRef<'_, T>,
        d_k: MatMut<'_, T>,
        param_idx: usize,
        uplo: Triangle,
        scratch: MatMut<'_, T>,
    ) -> Result<(), GprError> {
        match self {
            Self::RbfArd(leaf) => leaf.grad_from_sq_diff::<M, _>(cache, d_k, param_idx, uplo),
            Self::MaternArd(leaf) => leaf.grad_from_sq_diff::<M, _>(cache, d_k, param_idx, uplo),
            Self::RationalQuadraticArd(leaf) => leaf.grad_from_sq_diff(cache, d_k, param_idx, uplo),
            Self::Constant(leaf) => leaf.grad_rows(x, d_k, param_idx, uplo),
            Self::White(leaf) => leaf.grad_rows(x, d_k, param_idx, uplo),
            _ => self.grad_points::<M>(x, d_k, param_idx, uplo, scratch),
        }
    }

    /// `∂K/∂θ_{param_idx}` in the leaf's own mode (see `apply_mixed`).
    fn grad_mixed<M: crate::math::KernelMath, S: Supply>(
        self,
        views: MixedKernelViews<'_, T, S>,
        d_k: MatMut<'_, T>,
        param_idx: usize,
        uplo: Triangle,
        scratch: MatMut<'_, T>,
    ) -> Result<(), GprError> {
        if self.reads_dist() {
            return match views.dist {
                Some(dist) => self.grad_dist::<M>(dist, d_k, param_idx, uplo),
                None => self.grad_points::<M>(views.x, d_k, param_idx, uplo, scratch),
            };
        }
        match views.ard_cache.filter(|_| self.needs_ard_sq_diff()) {
            Some(cache) => self.grad_ard_cache::<M>(cache, views.x, d_k, param_idx, uplo, scratch),
            None => self.grad_points::<M>(views.x, d_k, param_idx, uplo, scratch),
        }
    }
}

/// The rectangular White block (and each of its derivatives): zero, since
/// `K(x1, x2)` is always the cross covariance. Needs no coordinate column.
pub(super) fn white_cross_zero<T: KernelScalar>(
    x1: MatRef<'_, T>,
    x2: MatRef<'_, T>,
    mut d_k: MatMut<'_, T>,
) -> Result<(), GprError> {
    if x1.nrows() == 0 || x2.nrows() == 0 {
        return Err(GprError::EmptyInput);
    }
    if d_k.nrows() != x1.nrows() || d_k.ncols() != x2.nrows() {
        return Err(GprError::ShapeMismatch {
            reason: format!(
                "output is {}x{}, expected {}x{}",
                d_k.nrows(),
                d_k.ncols(),
                x1.nrows(),
                x2.nrows()
            ),
        });
    }
    d_k.fill(T::from_f64(0.0));
    Ok(())
}

/// `out` has one entry per row of `x`.
fn require_count_diag<T>(x: MatRef<'_, T>, out: &[T]) -> Result<(), GprError> {
    if out.len() == x.nrows() {
        Ok(())
    } else {
        Err(GprError::LengthMismatch {
            reason: format!("expected {} diagonal entries, got {}", x.nrows(), out.len()),
        })
    }
}

/// Folds `eval` over the terms of a sum into `out`, each later term through
/// `scratch` (the shape of `out`).
///
/// A sum's terms are never sums (the tree flattens them), so `eval` hands
/// each term an empty scratch, which a leaf does not read.
fn fold_coord_sum<T: KernelScalar>(
    terms: &[CompiledKernel<T>],
    mut out: MatMut<'_, T>,
    mut scratch: MatMut<'_, T>,
    mut eval: impl FnMut(&CompiledKernel<T>, MatMut<'_, T>) -> Result<(), GprError>,
) -> Result<(), GprError> {
    let (first, rest) = terms
        .split_first()
        .ok_or(GprError::CoordGradientUnsupported)?;
    eval(first, out.as_mut())?;
    if !rest.is_empty() {
        super::require_scratch_shape(out.as_ref(), scratch.as_ref())?;
    }
    for term in rest {
        eval(term, scratch.as_mut())?;
        super::apply::add_rect(out.as_mut(), scratch.as_ref());
    }
    Ok(())
}

/// A custom distance leaf's `∂K/∂θ` from coordinates: `scratch` (the shape
/// of `d_k`, distinct from it) holds the pairwise squared distances.
fn grad_custom_from_coords<T: KernelScalar>(
    leaf: &CustomKernel<T>,
    x: MatRef<'_, T>,
    mut d_k: MatMut<'_, T>,
    param_idx: usize,
    uplo: Triangle,
    mut scratch: MatMut<'_, T>,
) -> Result<(), GprError> {
    write_square_from_coords(x, scratch.as_mut(), uplo, Ok)?;
    leaf.grad(scratch.as_ref(), d_k.as_mut(), param_idx, uplo)
}

fn term_for_param<T: KernelScalar, S: Supply>(
    terms: &[CompiledKernel<T, S>],
    param_idx: usize,
) -> Result<(&CompiledKernel<T, S>, usize), GprError> {
    let (index, local) = term_index_for_param(terms, param_idx)?;
    Ok((&terms[index], local))
}

/// The index of the term that owns `param_idx`, and the index within it.
pub(super) fn term_index_for_param<T: KernelScalar, S: Supply>(
    terms: &[CompiledKernel<T, S>],
    param_idx: usize,
) -> Result<(usize, usize), GprError> {
    let mut offset = 0;
    for (index, term) in terms.iter().enumerate() {
        let count = term.num_params();
        if param_idx < offset + count {
            return Ok((index, param_idx - offset));
        }
        offset += count;
    }
    Err(GprError::IndexOutOfRange {
        reason: format!("kernel parameter index {param_idx} is out of range"),
    })
}

/// How a product multiplies the running block by the next factor: the
/// triangle of a square block, or every entry of a rectangular one.
#[derive(Clone, Copy)]
pub(super) enum Fold {
    Triangle(Triangle),
    Rect,
}

/// `acc *= src` over the part of the block `fold` names.
pub(super) fn mul_fold<T: KernelScalar>(acc: MatMut<'_, T>, src: MatRef<'_, T>, fold: Fold) {
    match fold {
        Fold::Triangle(uplo) => mul_triangle(acc, src, uplo),
        Fold::Rect => super::apply::mul_rect(acc, src),
    }
}

/// Output, scratch, nested levels, and fold of one product.
pub(super) struct ProductBuffers<'a, 'n, T> {
    pub(super) out: MatMut<'a, T>,
    pub(super) scratch: MatMut<'a, T>,
    pub(super) nested: &'n mut Nested<T>,
    pub(super) fold: Fold,
}

impl<'a, 'n, T> ProductBuffers<'a, 'n, T> {
    /// A square block, folded over `uplo`.
    pub(super) fn new(
        out: MatMut<'a, T>,
        scratch: MatMut<'a, T>,
        nested: &'n mut Nested<T>,
        uplo: Triangle,
    ) -> Self {
        Self {
            out,
            scratch,
            nested,
            fold: Fold::Triangle(uplo),
        }
    }

    /// A rectangular block, folded over every entry.
    pub(super) fn rect(
        out: MatMut<'a, T>,
        scratch: MatMut<'a, T>,
        nested: &'n mut Nested<T>,
    ) -> Self {
        Self {
            out,
            scratch,
            nested,
            fold: Fold::Rect,
        }
    }
}

/// Writes `deriv(terms[owner]) · ∏_{k ≠ owner} apply(terms[k])` into `out`.
///
/// Each writer gets `(term, out, scratch, nested levels below it)`.
pub(super) fn product_with_owner<T: KernelScalar, S: Supply>(
    terms: &[CompiledKernel<T, S>],
    owner: usize,
    buffers: ProductBuffers<'_, '_, T>,
    mut apply: impl FnMut(
        &CompiledKernel<T, S>,
        MatMut<'_, T>,
        MatMut<'_, T>,
        &mut Nested<T>,
    ) -> Result<(), GprError>,
    mut deriv: impl FnMut(
        &CompiledKernel<T, S>,
        MatMut<'_, T>,
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
        if k == owner {
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
    let owner = &terms[owner];
    if started {
        let (own, deeper) = term_scratch(owner, rows, cols, out.as_mut(), &mut *nested)?;
        deriv(owner, scratch.as_mut(), own, deeper)?;
        mul_fold(out.as_mut(), scratch.as_ref(), fold);
    } else {
        deriv(owner, out, scratch, nested)?;
    }
    Ok(())
}

pub(super) fn require_diag_len<T: KernelScalar>(
    x: MatRef<'_, T>,
    out: &[T],
) -> Result<(), GprError> {
    // Columns may be zero: Constant, White, and supplied leaves read none.
    if x.nrows() == 0 {
        return Err(GprError::EmptyInput);
    }
    if out.len() != x.nrows() {
        return Err(GprError::LengthMismatch {
            reason: format!("expected {} diagonal entries, got {}", x.nrows(), out.len()),
        });
    }
    Ok(())
}

pub(super) fn broadcast_self_diag<T: KernelScalar>(
    x: MatRef<'_, T>,
    out: &mut [T],
    eval: impl FnOnce(MatRef<'_, T>, MatMut<'_, T>) -> Result<(), GprError>,
) -> Result<(), GprError> {
    require_diag_len(x, out)?;
    let one = x.submatrix(0, 0, 1, x.ncols());
    out.fill(eval_cell(|cell| eval(one, cell))?);
    Ok(())
}

/// Runs `eval` on a stack `1 × 1` matrix and returns its entry.
pub(super) fn eval_cell<T: KernelScalar>(
    eval: impl FnOnce(MatMut<'_, T>) -> Result<(), GprError>,
) -> Result<T, GprError> {
    let mut cell = [T::from_f64(0.0)];
    eval(MatMut::from_column_major_slice_mut(&mut cell, 1, 1))?;
    Ok(cell[0])
}

pub(super) fn scale_by_other_diags<T: KernelScalar, S: Supply>(
    terms: &[CompiledKernel<T, S>],
    skip_a: usize,
    skip_b: Option<usize>,
    x: MatRef<'_, T>,
    out: &mut [T],
) -> Result<(), GprError> {
    for (index, term) in terms.iter().enumerate() {
        if index == skip_a || Some(index) == skip_b {
            continue;
        }
        combine_diag(out, mul_assign, |start, block| {
            term.fill_diag_rows(x.subrows(start, block.len()), block)
        })?;
    }
    Ok(())
}

fn product_grad_diag<M: crate::math::KernelMath, T: KernelScalar, S: Supply>(
    terms: &[CompiledKernel<T, S>],
    x: MatRef<'_, T>,
    out: &mut [T],
    param_idx: usize,
) -> Result<(), GprError> {
    let (owner, local) = term_index_for_param(terms, param_idx)?;
    terms[owner].grad_diag_points::<M>(x, out, local)?;
    scale_by_other_diags(terms, owner, None, x, out)
}

/// Coordinate entry points of a coordinate tree.
impl<T: KernelScalar> CompiledKernel<T> {
    /// `∂K(x1, x2)/∂θ_{param_idx}` of a rectangular block
    /// (`x1.nrows() × x2.nrows()`) from coordinates: every built-in leaf, and
    /// Sum / Product trees of them. A `Custom` leaf has no rectangular
    /// derivative and returns [`GprError::CoordGradientUnsupported`].
    ///
    /// Product trees need `scratch` the same shape as `d_k` and distinct from
    /// it; leaves ignore it.
    pub(crate) fn grad_cross_points<M: crate::math::KernelMath>(
        &self,
        x1: MatRef<'_, T>,
        x2: MatRef<'_, T>,
        d_k: MatMut<'_, T>,
        param_idx: usize,
        scratch: MatMut<'_, T>,
    ) -> Result<(), GprError> {
        let mut nested = self.nested_buffers(d_k.nrows(), d_k.ncols());
        self.grad_cross_points_with::<M>(x1, x2, d_k, param_idx, scratch, &mut nested)
    }

    /// [`Self::grad_cross_points`] with caller-owned [`Nested`] levels.
    pub(crate) fn grad_cross_points_with<M: crate::math::KernelMath>(
        &self,
        x1: MatRef<'_, T>,
        x2: MatRef<'_, T>,
        d_k: MatMut<'_, T>,
        param_idx: usize,
        scratch: MatMut<'_, T>,
        nested: &mut Nested<T>,
    ) -> Result<(), GprError> {
        self.grad_cross_views::<M>(CrossViews::points(x1, x2), d_k, param_idx, scratch, nested)
    }
}
