use super::supplied::needs_supply;
use super::{
    CompiledKernel, CrossViews, MixedKernelViews, Nested, add_triangle, ard_needs_coords,
    mul_triangle, require_scratch_shape, split_terms, term_scratch,
};
use crate::error::GprError;
use crate::kernel::KernelScalar;
use crate::kernel::dist::ArdSqDiff;
use crate::kernel::{CustomKernel, Triangle, write_rect_from_coords, write_square_from_coords};
use faer::{MatMut, MatRef};

impl<T: KernelScalar> CompiledKernel<T> {
    /// Writes `k` into `out` for `uplo`. `scratch` must match `out`.
    ///
    /// Entries outside the requested triangle are left unchanged. `scratch`
    /// must be a distinct buffer from `out`.
    ///
    /// A sum / product nested in another allocates one output-shaped buffer
    /// per nesting level for the call.
    ///
    /// # Errors
    ///
    /// Returns [`GprError`] if shapes mismatch, `scratch` is the wrong size, a
    /// leaf fails, or a sum/product has no terms.
    ///
    /// See the example on [`CompiledKernel`].
    pub fn apply<M: crate::math::KernelMath>(
        &self,
        dist: MatRef<'_, T>,
        out: MatMut<'_, T>,
        uplo: Triangle,
        scratch: MatMut<'_, T>,
    ) -> Result<(), GprError> {
        let mut nested = self.nested_buffers(out.nrows(), out.ncols());
        self.apply_with::<M>(dist, out, uplo, scratch, &mut nested)
    }

    /// [`Self::apply`] with caller-owned [`Nested`] levels.
    pub(crate) fn apply_with<M: crate::math::KernelMath>(
        &self,
        dist: MatRef<'_, T>,
        mut out: MatMut<'_, T>,
        uplo: Triangle,
        mut scratch: MatMut<'_, T>,
        nested: &mut Nested<T>,
    ) -> Result<(), GprError> {
        require_scratch_shape(out.as_ref(), scratch.as_ref())?;
        match self {
            Self::Rbf(leaf) => leaf.apply_math::<M, _>(dist, out, uplo),
            Self::RbfArd(_)
            | Self::Linear(_)
            | Self::MaternArd(_)
            | Self::RationalQuadraticArd(_) => Err(ard_needs_coords()),
            Self::Matern(leaf) => leaf.apply_math::<M, _>(dist, out, uplo),
            Self::Periodic(leaf) => leaf.apply_math::<M, _>(dist, out, uplo),
            Self::RationalQuadratic(leaf) => leaf.apply(dist, out, uplo),
            Self::Constant(leaf) => leaf.apply(dist, out, uplo),
            Self::White(leaf) => leaf.apply(dist, out, uplo),
            Self::Custom(leaf) => leaf.apply(dist, out, uplo),
            Self::Supplied(_) => Err(needs_supply()),
            Self::Sum(terms) => fold_terms::<M, _>(
                terms,
                dist,
                out.as_mut(),
                uplo,
                scratch.as_mut(),
                nested,
                add_triangle,
            ),
            Self::Product(terms) => fold_terms::<M, _>(
                terms,
                dist,
                out.as_mut(),
                uplo,
                scratch.as_mut(),
                nested,
                mul_triangle,
            ),
        }
    }

    /// Writes rectangular `k(dist)` (train × test) into `out`.
    ///
    /// `scratch` must match `out` and be a distinct buffer.
    ///
    /// A sum / product nested in another allocates one output-shaped buffer
    /// per nesting level for the call.
    ///
    /// # Errors
    ///
    /// Returns the same shape errors as [`Self::apply`].
    ///
    /// See the example on [`CompiledKernel`].
    pub fn apply_cross<M: crate::math::KernelMath>(
        &self,
        dist: MatRef<'_, T>,
        out: MatMut<'_, T>,
        scratch: MatMut<'_, T>,
    ) -> Result<(), GprError> {
        let mut nested = self.nested_buffers(out.nrows(), out.ncols());
        self.apply_cross_with::<M>(dist, out, scratch, &mut nested)
    }

    /// [`Self::apply_cross`] with caller-owned [`Nested`] levels.
    pub(crate) fn apply_cross_with<M: crate::math::KernelMath>(
        &self,
        dist: MatRef<'_, T>,
        mut out: MatMut<'_, T>,
        mut scratch: MatMut<'_, T>,
        nested: &mut Nested<T>,
    ) -> Result<(), GprError> {
        require_scratch_shape(out.as_ref(), scratch.as_ref())?;
        match self {
            Self::Rbf(leaf) => leaf.apply_cross_math::<M, _>(dist, out),
            Self::RbfArd(_)
            | Self::Linear(_)
            | Self::MaternArd(_)
            | Self::RationalQuadraticArd(_) => Err(ard_needs_coords()),
            Self::Matern(leaf) => leaf.apply_cross_math::<M, _>(dist, out),
            Self::Periodic(leaf) => leaf.apply_cross_math::<M, _>(dist, out),
            Self::RationalQuadratic(leaf) => leaf.apply_cross(dist, out),
            Self::Constant(leaf) => leaf.apply_cross(dist, out),
            Self::White(leaf) => leaf.apply_cross(dist, out),
            Self::Custom(leaf) => leaf.apply_cross(dist, out),
            Self::Supplied(_) => Err(needs_supply()),
            Self::Sum(terms) => fold_rect::<M, _>(
                terms,
                dist,
                out.as_mut(),
                scratch.as_mut(),
                nested,
                add_rect,
            ),
            Self::Product(terms) => fold_rect::<M, _>(
                terms,
                dist,
                out.as_mut(),
                scratch.as_mut(),
                nested,
                mul_rect,
            ),
        }
    }

    /// Writes the diagonal `k(x, x)` into `out`.
    ///
    /// Leaf `fill_diag` methods cannot fail and return `()`. This one returns
    /// a `Result` because a tree may hold a leaf whose diagonal needs the
    /// coordinates ([`super::super::LinearKernel`]); use
    /// [`Self::fill_diag_points`] for such trees.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::UnsupportedKernelOperation`] if a sum/product has no
    /// terms or a leaf needs coordinates.
    ///
    /// See the example on [`CompiledKernel`].
    pub fn fill_diag(&self, out: &mut [T]) -> Result<(), GprError> {
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
            Self::Supplied(leaf) => leaf.fill_diag(out),
            Self::Linear(_) => Err(GprError::UnsupportedKernelOperation {
                reason: "linear kernel diagonal needs coordinates".to_owned(),
            }),
            Self::Sum(terms) => fold_diag(terms, out, add_assign),
            Self::Product(terms) => fold_diag(terms, out, mul_assign),
        }
    }

    /// Writes the diagonal `k(x, x)` using coordinates when the leaf needs them.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::UnsupportedKernelOperation`] if a sum/product has no
    /// terms, [`GprError::LengthMismatch`] if a sum/product's `out` is not
    /// `x.nrows()` long, or the same shape errors as the leaf.
    ///
    /// See the example on [`CompiledKernel`].
    pub fn fill_diag_points(&self, x: MatRef<'_, T>, out: &mut [T]) -> Result<(), GprError> {
        self.require_columns(x)?;
        self.fill_diag_rows(x, out)
    }

    /// [`Self::fill_diag_points`] without the column check, for a subtree
    /// of a tree that already passed it.
    pub(crate) fn fill_diag_rows(&self, x: MatRef<'_, T>, out: &mut [T]) -> Result<(), GprError> {
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
            Self::Supplied(leaf) => leaf.fill_diag(out),
            Self::Linear(leaf) => leaf.fill_diag_points(x, out),
            Self::Sum(terms) => fold_diag_points(terms, x, out, add_assign),
            Self::Product(terms) => fold_diag_points(terms, x, out, mul_assign),
        }
    }

    /// Writes `k` from point coordinates.
    ///
    /// Isotropic leaves compute `‖x_i-x_j‖²` from `X` on each pair. ARD and
    /// Linear evaluate from coordinates. Custom distance leaves write those
    /// distances into `scratch`, then apply.
    ///
    /// A sum / product nested in another allocates one output-shaped buffer
    /// per nesting level for the call.
    ///
    /// # Errors
    ///
    /// Same shape errors as [`Self::apply`].
    ///
    /// See the example on [`CompiledKernel`].
    pub fn apply_points<M: crate::math::KernelMath>(
        &self,
        x: MatRef<'_, T>,
        out: MatMut<'_, T>,
        uplo: Triangle,
        scratch: MatMut<'_, T>,
    ) -> Result<(), GprError> {
        self.require_columns(x)?;
        let mut nested = self.nested_buffers(out.nrows(), out.ncols());
        self.apply_points_with::<M>(x, out, uplo, scratch, &mut nested)
    }

    /// [`Self::apply_points`] with caller-owned [`Nested`] levels.
    pub(crate) fn apply_points_with<M: crate::math::KernelMath>(
        &self,
        x: MatRef<'_, T>,
        mut out: MatMut<'_, T>,
        uplo: Triangle,
        mut scratch: MatMut<'_, T>,
        nested: &mut Nested<T>,
    ) -> Result<(), GprError> {
        require_scratch_shape(out.as_ref(), scratch.as_ref())?;
        match self {
            Self::Rbf(leaf) => leaf.apply_from_coords::<M, _>(x, out, uplo),
            Self::Matern(leaf) => leaf.apply_from_coords::<M, _>(x, out, uplo),
            Self::Periodic(leaf) => leaf.apply_from_coords::<M, _>(x, out, uplo),
            Self::RationalQuadratic(leaf) => leaf.apply_from_coords(x, out, uplo),
            Self::Custom(leaf) => apply_custom_from_coords(leaf, x, out, uplo, scratch),
            Self::RbfArd(leaf) => leaf.apply_math::<M, _>(x, out, uplo),
            Self::Linear(leaf) => leaf.apply(x, out, uplo),
            Self::MaternArd(leaf) => leaf.apply_math::<M, _>(x, out, uplo),
            Self::RationalQuadraticArd(leaf) => leaf.apply(x, out, uplo),
            Self::Constant(leaf) => leaf.apply_rows(x, out, uplo),
            Self::White(leaf) => leaf.apply_rows(x, out, uplo),
            Self::Supplied(_) => Err(needs_supply()),
            Self::Sum(terms) => fold_terms_points::<M, _>(
                terms,
                x,
                out.as_mut(),
                uplo,
                scratch.as_mut(),
                nested,
                add_triangle,
            ),
            Self::Product(terms) => fold_terms_points::<M, _>(
                terms,
                x,
                out.as_mut(),
                uplo,
                scratch.as_mut(),
                nested,
                mul_triangle,
            ),
        }
    }

    /// Writes rectangular `k(x, xs)` from coordinates.
    ///
    /// A sum / product nested in another allocates one output-shaped buffer
    /// per nesting level for the call.
    ///
    /// # Errors
    ///
    /// Same as [`Self::apply_points`].
    ///
    /// See the example on [`CompiledKernel`].
    pub fn apply_cross_points<M: crate::math::KernelMath>(
        &self,
        x: MatRef<'_, T>,
        xs: MatRef<'_, T>,
        out: MatMut<'_, T>,
        scratch: MatMut<'_, T>,
    ) -> Result<(), GprError> {
        self.require_columns(x)?;
        self.require_columns(xs)?;
        self.apply_cross_rows::<M>(x, xs, out, scratch)
    }

    /// [`Self::apply_cross_points`] without the column check, for a subtree
    /// of a tree that already passed it.
    pub(crate) fn apply_cross_rows<M: crate::math::KernelMath>(
        &self,
        x: MatRef<'_, T>,
        xs: MatRef<'_, T>,
        out: MatMut<'_, T>,
        scratch: MatMut<'_, T>,
    ) -> Result<(), GprError> {
        let mut nested = self.nested_buffers(out.nrows(), out.ncols());
        self.apply_cross_points_with::<M>(x, xs, out, scratch, &mut nested)
    }

    /// [`Self::apply_cross_points`] with caller-owned [`Nested`] levels.
    pub(crate) fn apply_cross_points_with<M: crate::math::KernelMath>(
        &self,
        x: MatRef<'_, T>,
        xs: MatRef<'_, T>,
        mut out: MatMut<'_, T>,
        mut scratch: MatMut<'_, T>,
        nested: &mut Nested<T>,
    ) -> Result<(), GprError> {
        require_scratch_shape(out.as_ref(), scratch.as_ref())?;
        match self {
            Self::Rbf(leaf) => leaf.apply_cross_from_coords::<M, _>(x, xs, out),
            Self::Matern(leaf) => leaf.apply_cross_from_coords::<M, _>(x, xs, out),
            Self::Periodic(leaf) => leaf.apply_cross_from_coords::<M, _>(x, xs, out),
            Self::RationalQuadratic(leaf) => leaf.apply_cross_from_coords(x, xs, out),
            Self::Custom(leaf) => {
                write_rect_from_coords(x, xs, scratch.as_mut(), Ok)?;
                leaf.apply_cross(scratch.as_ref(), out)
            }
            Self::RbfArd(leaf) => leaf.apply_cross_math::<M, _>(x, xs, out),
            Self::Linear(leaf) => leaf.apply_cross(x, xs, out),
            Self::MaternArd(leaf) => leaf.apply_cross_math::<M, _>(x, xs, out),
            Self::RationalQuadraticArd(leaf) => leaf.apply_cross(x, xs, out),
            Self::Constant(leaf) => leaf.apply_cross_rows(x, xs, out),
            Self::White(leaf) => leaf.apply_cross_rows(x, xs, out),
            Self::Supplied(_) => Err(needs_supply()),
            Self::Sum(terms) => fold_rect_points::<M, _>(
                terms,
                x,
                xs,
                out.as_mut(),
                scratch.as_mut(),
                nested,
                add_rect,
            ),
            Self::Product(terms) => fold_rect_points::<M, _>(
                terms,
                x,
                xs,
                out.as_mut(),
                scratch.as_mut(),
                nested,
                mul_rect,
            ),
        }
    }

    pub(crate) fn apply_from_ard_cache<M: crate::math::KernelMath>(
        &self,
        cache: ArdSqDiff<'_, T>,
        x: MatRef<'_, T>,
        mut out: MatMut<'_, T>,
        uplo: Triangle,
        mut scratch: MatMut<'_, T>,
        nested: &mut Nested<T>,
    ) -> Result<(), GprError> {
        require_scratch_shape(out.as_ref(), scratch.as_ref())?;
        match self {
            Self::RbfArd(leaf) => leaf.apply_from_sq_diff::<M, _>(cache, out, uplo),
            Self::MaternArd(leaf) => leaf.apply_from_sq_diff::<M, _>(cache, out, uplo),
            Self::RationalQuadraticArd(leaf) => leaf.apply_from_sq_diff(cache, out, uplo),
            Self::Constant(leaf) => leaf.apply_rows(x, out, uplo),
            Self::White(leaf) => leaf.apply_rows(x, out, uplo),
            Self::Sum(terms) => fold_terms_ard_cache::<M, _>(
                terms,
                cache,
                x,
                out.as_mut(),
                uplo,
                scratch.as_mut(),
                nested,
            ),
            _ => self.apply_points_with::<M>(x, out, uplo, scratch, nested),
        }
    }

    /// Writes `K` from a distance matrix and coordinates, one mode per leaf.
    pub(crate) fn apply_mixed<M: crate::math::KernelMath>(
        &self,
        views: MixedKernelViews<'_, T>,
        mut out: MatMut<'_, T>,
        uplo: Triangle,
        mut scratch: MatMut<'_, T>,
        nested: &mut Nested<T>,
    ) -> Result<(), GprError> {
        require_scratch_shape(out.as_ref(), scratch.as_ref())?;
        match self {
            Self::Rbf(_)
            | Self::Matern(_)
            | Self::Periodic(_)
            | Self::RationalQuadratic(_)
            | Self::Custom(_)
            | Self::Constant(_)
            | Self::White(_) => match views.dist {
                Some(dist) => self.apply_with::<M>(dist, out, uplo, scratch, nested),
                None => self.apply_points_with::<M>(views.x, out, uplo, scratch, nested),
            },
            Self::RbfArd(_)
            | Self::Linear(_)
            | Self::MaternArd(_)
            | Self::RationalQuadraticArd(_) => {
                if let Some(cache) = views.ard_cache.filter(|_| self.needs_ard_sq_diff()) {
                    self.apply_from_ard_cache::<M>(cache, views.x, out, uplo, scratch, nested)
                } else {
                    self.apply_points_with::<M>(views.x, out, uplo, scratch, nested)
                }
            }
            Self::Supplied(leaf) => leaf.apply::<M>(views.slots, out, uplo),
            Self::Sum(terms) => fold_terms_mixed::<M, _>(
                terms,
                views,
                out.as_mut(),
                uplo,
                scratch.as_mut(),
                nested,
                add_triangle,
            ),
            Self::Product(terms) => fold_terms_mixed::<M, _>(
                terms,
                views,
                out.as_mut(),
                uplo,
                scratch.as_mut(),
                nested,
                mul_triangle,
            ),
        }
    }

    /// Writes rectangular `k` from the views of the block, one mode per
    /// leaf: coordinate distances (or coordinates without them),
    /// coordinates, or the supplied distances.
    pub(crate) fn apply_cross_mixed<M: crate::math::KernelMath>(
        &self,
        views: CrossViews<'_, T>,
        mut out: MatMut<'_, T>,
        mut scratch: MatMut<'_, T>,
        nested: &mut Nested<T>,
    ) -> Result<(), GprError> {
        require_scratch_shape(out.as_ref(), scratch.as_ref())?;
        let CrossViews { x1, x2, dist, .. } = views;
        match self {
            Self::Rbf(_)
            | Self::Matern(_)
            | Self::Periodic(_)
            | Self::RationalQuadratic(_)
            | Self::Custom(_)
            | Self::Constant(_)
            | Self::White(_) => match dist {
                Some(dist) => self.apply_cross_with::<M>(dist, out, scratch, nested),
                None => self.apply_cross_points_with::<M>(x1, x2, out, scratch, nested),
            },
            Self::RbfArd(_)
            | Self::Linear(_)
            | Self::MaternArd(_)
            | Self::RationalQuadraticArd(_) => {
                self.apply_cross_points_with::<M>(x1, x2, out, scratch, nested)
            }
            Self::Supplied(leaf) => leaf.apply_cross::<M>(views.slots, out),
            Self::Sum(terms) => fold_rect_mixed::<M, _>(
                terms,
                views,
                out.as_mut(),
                scratch.as_mut(),
                nested,
                add_rect,
            ),
            Self::Product(terms) => fold_rect_mixed::<M, _>(
                terms,
                views,
                out.as_mut(),
                scratch.as_mut(),
                nested,
                mul_rect,
            ),
        }
    }
}

/// Rows per stack block in the diagonal folds, which keep no heap temporary.
const DIAG_BLOCK: usize = 64;

/// Combines `eval(start, block)` into `out` one stack block of rows at a
/// time. `start` is the block's first row.
pub(super) fn combine_diag<T: KernelScalar>(
    out: &mut [T],
    combine: fn(&mut T, T),
    mut eval: impl FnMut(usize, &mut [T]) -> Result<(), GprError>,
) -> Result<(), GprError> {
    let mut buffer = [T::from_f64(0.0); DIAG_BLOCK];
    for (index, chunk) in out.chunks_mut(DIAG_BLOCK).enumerate() {
        let block = &mut buffer[..chunk.len()];
        eval(index * DIAG_BLOCK, block)?;
        for (dst, src) in chunk.iter_mut().zip(block.iter()) {
            combine(dst, *src);
        }
    }
    Ok(())
}

fn fold_diag<T: KernelScalar>(
    terms: &[CompiledKernel<T>],
    out: &mut [T],
    combine: fn(&mut T, T),
) -> Result<(), GprError> {
    let (first, rest) = split_terms(terms)?;
    first.fill_diag(out)?;
    for term in rest {
        combine_diag(out, combine, |_, block| term.fill_diag(block))?;
    }
    Ok(())
}

fn fold_diag_points<T: KernelScalar>(
    terms: &[CompiledKernel<T>],
    x: MatRef<'_, T>,
    out: &mut [T],
    combine: fn(&mut T, T),
) -> Result<(), GprError> {
    if out.len() != x.nrows() {
        return Err(GprError::LengthMismatch {
            reason: format!("expected {} diagonal entries, got {}", x.nrows(), out.len()),
        });
    }
    let (first, rest) = split_terms(terms)?;
    first.fill_diag_rows(x, out)?;
    for term in rest {
        combine_diag(out, combine, |start, block| {
            term.fill_diag_rows(x.subrows(start, block.len()), block)
        })?;
    }
    Ok(())
}

pub(super) fn add_assign<T: KernelScalar>(dst: &mut T, src: T) {
    *dst += src;
}

pub(super) fn mul_assign<T: KernelScalar>(dst: &mut T, src: T) {
    *dst *= src;
}

/// A custom distance leaf from coordinates: `scratch` (the shape of `out`,
/// distinct from it) holds the pairwise squared distances.
fn apply_custom_from_coords<T: KernelScalar>(
    leaf: &CustomKernel<T>,
    x: MatRef<'_, T>,
    mut out: MatMut<'_, T>,
    uplo: Triangle,
    mut scratch: MatMut<'_, T>,
) -> Result<(), GprError> {
    write_square_from_coords(x, scratch.as_mut(), uplo, Ok)?;
    leaf.apply(scratch.as_ref(), out.as_mut(), uplo)
}

fn fold_terms<M: crate::math::KernelMath, T: KernelScalar>(
    terms: &[CompiledKernel<T>],
    dist: MatRef<'_, T>,
    mut out: MatMut<'_, T>,
    uplo: Triangle,
    mut scratch: MatMut<'_, T>,
    nested: &mut Nested<T>,
    combine: fn(MatMut<'_, T>, MatRef<'_, T>, Triangle),
) -> Result<(), GprError> {
    let (first, rest) = split_terms(terms)?;
    first.apply_with::<M>(dist, out.as_mut(), uplo, scratch.as_mut(), nested)?;
    let (rows, cols) = (out.nrows(), out.ncols());
    for term in rest {
        let (own, deeper) = term_scratch(term, rows, cols, out.as_mut(), &mut *nested)?;
        term.apply_with::<M>(dist, scratch.as_mut(), uplo, own, deeper)?;
        combine(out.as_mut(), scratch.as_ref(), uplo);
    }
    Ok(())
}

fn fold_terms_points<M: crate::math::KernelMath, T: KernelScalar>(
    terms: &[CompiledKernel<T>],
    x: MatRef<'_, T>,
    mut out: MatMut<'_, T>,
    uplo: Triangle,
    mut scratch: MatMut<'_, T>,
    nested: &mut Nested<T>,
    combine: fn(MatMut<'_, T>, MatRef<'_, T>, Triangle),
) -> Result<(), GprError> {
    let (first, rest) = split_terms(terms)?;
    first.apply_points_with::<M>(x, out.as_mut(), uplo, scratch.as_mut(), nested)?;
    let (rows, cols) = (out.nrows(), out.ncols());
    for term in rest {
        let (own, deeper) = term_scratch(term, rows, cols, out.as_mut(), &mut *nested)?;
        term.apply_points_with::<M>(x, scratch.as_mut(), uplo, own, deeper)?;
        combine(out.as_mut(), scratch.as_ref(), uplo);
    }
    Ok(())
}

fn fold_terms_ard_cache<M: crate::math::KernelMath, T: KernelScalar>(
    terms: &[CompiledKernel<T>],
    cache: ArdSqDiff<'_, T>,
    x: MatRef<'_, T>,
    mut out: MatMut<'_, T>,
    uplo: Triangle,
    mut scratch: MatMut<'_, T>,
    nested: &mut Nested<T>,
) -> Result<(), GprError> {
    let (first, rest) = split_terms(terms)?;
    first.apply_from_ard_cache::<M>(cache, x, out.as_mut(), uplo, scratch.as_mut(), nested)?;
    let (rows, cols) = (out.nrows(), out.ncols());
    for term in rest {
        let (own, deeper) = term_scratch(term, rows, cols, out.as_mut(), &mut *nested)?;
        term.apply_from_ard_cache::<M>(cache, x, scratch.as_mut(), uplo, own, deeper)?;
        add_triangle(out.as_mut(), scratch.as_ref(), uplo);
    }
    Ok(())
}

fn fold_terms_mixed<M: crate::math::KernelMath, T: KernelScalar>(
    terms: &[CompiledKernel<T>],
    views: MixedKernelViews<'_, T>,
    mut out: MatMut<'_, T>,
    uplo: Triangle,
    mut scratch: MatMut<'_, T>,
    nested: &mut Nested<T>,
    combine: fn(MatMut<'_, T>, MatRef<'_, T>, Triangle),
) -> Result<(), GprError> {
    let (first, rest) = split_terms(terms)?;
    first.apply_mixed::<M>(views, out.as_mut(), uplo, scratch.as_mut(), nested)?;
    let (rows, cols) = (out.nrows(), out.ncols());
    for term in rest {
        let (own, deeper) = term_scratch(term, rows, cols, out.as_mut(), &mut *nested)?;
        term.apply_mixed::<M>(views, scratch.as_mut(), uplo, own, deeper)?;
        combine(out.as_mut(), scratch.as_ref(), uplo);
    }
    Ok(())
}

pub(super) fn add_rect<T: KernelScalar>(mut acc: MatMut<'_, T>, src: MatRef<'_, T>) {
    for col in 0..acc.ncols() {
        for row in 0..acc.nrows() {
            acc[(row, col)] += src[(row, col)];
        }
    }
}

pub(super) fn mul_rect<T: KernelScalar>(mut acc: MatMut<'_, T>, src: MatRef<'_, T>) {
    for col in 0..acc.ncols() {
        for row in 0..acc.nrows() {
            acc[(row, col)] *= src[(row, col)];
        }
    }
}

fn fold_rect<M: crate::math::KernelMath, T: KernelScalar>(
    terms: &[CompiledKernel<T>],
    dist: MatRef<'_, T>,
    mut out: MatMut<'_, T>,
    mut scratch: MatMut<'_, T>,
    nested: &mut Nested<T>,
    combine: fn(MatMut<'_, T>, MatRef<'_, T>),
) -> Result<(), GprError> {
    let (first, rest) = split_terms(terms)?;
    first.apply_cross_with::<M>(dist, out.as_mut(), scratch.as_mut(), nested)?;
    let (rows, cols) = (out.nrows(), out.ncols());
    for term in rest {
        let (own, deeper) = term_scratch(term, rows, cols, out.as_mut(), &mut *nested)?;
        term.apply_cross_with::<M>(dist, scratch.as_mut(), own, deeper)?;
        combine(out.as_mut(), scratch.as_ref());
    }
    Ok(())
}

fn fold_rect_points<M: crate::math::KernelMath, T: KernelScalar>(
    terms: &[CompiledKernel<T>],
    x: MatRef<'_, T>,
    xs: MatRef<'_, T>,
    mut out: MatMut<'_, T>,
    mut scratch: MatMut<'_, T>,
    nested: &mut Nested<T>,
    combine: fn(MatMut<'_, T>, MatRef<'_, T>),
) -> Result<(), GprError> {
    let (first, rest) = split_terms(terms)?;
    first.apply_cross_points_with::<M>(x, xs, out.as_mut(), scratch.as_mut(), nested)?;
    let (rows, cols) = (out.nrows(), out.ncols());
    for term in rest {
        let (own, deeper) = term_scratch(term, rows, cols, out.as_mut(), &mut *nested)?;
        term.apply_cross_points_with::<M>(x, xs, scratch.as_mut(), own, deeper)?;
        combine(out.as_mut(), scratch.as_ref());
    }
    Ok(())
}

fn fold_rect_mixed<M: crate::math::KernelMath, T: KernelScalar>(
    terms: &[CompiledKernel<T>],
    views: CrossViews<'_, T>,
    mut out: MatMut<'_, T>,
    mut scratch: MatMut<'_, T>,
    nested: &mut Nested<T>,
    combine: fn(MatMut<'_, T>, MatRef<'_, T>),
) -> Result<(), GprError> {
    let (first, rest) = split_terms(terms)?;
    first.apply_cross_mixed::<M>(views, out.as_mut(), scratch.as_mut(), nested)?;
    let (rows, cols) = (out.nrows(), out.ncols());
    for term in rest {
        let (own, deeper) = term_scratch(term, rows, cols, out.as_mut(), &mut *nested)?;
        term.apply_cross_mixed::<M>(views, scratch.as_mut(), own, deeper)?;
        combine(out.as_mut(), scratch.as_ref());
    }
    Ok(())
}
