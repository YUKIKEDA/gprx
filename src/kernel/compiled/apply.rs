use super::{
    CompiledKernel, MixedKernelViews, add_triangle, ard_needs_coords, iso_needs_dist, mul_triangle,
    require_scratch_shape, split_terms,
};
use crate::error::GprError;
use crate::kernel::{CustomKernel, Triangle, write_square_from_coords};
use faer::{Mat, MatMut, MatRef};

impl CompiledKernel<f64> {
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

pub(super) fn apply_into(
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

pub(super) fn apply_into_points(
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

pub(super) fn add_rect(mut acc: MatMut<'_, f64>, src: MatRef<'_, f64>) {
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

pub(super) fn apply_into_mixed(
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
