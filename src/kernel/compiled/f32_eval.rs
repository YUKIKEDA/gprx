//! Scalar `f32` evaluation of [`super::CompiledKernel`].
//!
//! Parameters stay `f64` and are cast once at the start of each leaf. The `f64`
//! distance cache and SIMD paths are not used here.

use super::{CompiledKernel, ard_needs_coords, iso_needs_dist, split_terms};
use crate::error::GprError;
use crate::kernel::{MaternNu, Triangle, visit_triangle};
use faer::{Mat, MatMut, MatRef};

impl CompiledKernel<f32> {
    /// Writes `k` into `out` for `uplo`. `scratch` must match `out`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError`] if shapes mismatch, `scratch` is the wrong size, a
    /// leaf fails, or a sum/product has no terms.
    pub fn apply(
        &self,
        dist: MatRef<'_, f32>,
        mut out: MatMut<'_, f32>,
        uplo: Triangle,
        mut scratch: MatMut<'_, f32>,
    ) -> Result<(), GprError> {
        scratch_ok(out.as_ref(), scratch.as_ref())?;
        match self {
            Self::Rbf(leaf) => rbf_apply(leaf.lengthscale(), dist, out, uplo),
            Self::Matern(leaf) => matern_apply(leaf.nu(), leaf.lengthscale(), dist, out, uplo),
            Self::Periodic(leaf) => {
                periodic_apply(leaf.lengthscale(), leaf.period(), dist, out, uplo)
            }
            Self::RationalQuadratic(leaf) => {
                rq_apply(leaf.lengthscale(), leaf.alpha(), dist, out, uplo)
            }
            Self::Constant(leaf) => constant_apply(leaf.constant(), dist, out, uplo),
            Self::White(leaf) => white_apply(leaf.variance(), dist, out, uplo),
            Self::Custom(leaf) => leaf.apply_f32(dist, out, uplo),
            Self::RbfArd(_)
            | Self::Linear(_)
            | Self::MaternArd(_)
            | Self::RationalQuadraticArd(_) => Err(ard_needs_coords()),
            Self::Sum(terms) => {
                fold_apply(terms, dist, out.as_mut(), uplo, scratch.as_mut(), add_tri)
            }
            Self::Product(terms) => {
                fold_apply(terms, dist, out.as_mut(), uplo, scratch.as_mut(), mul_tri)
            }
        }
    }

    /// Writes rectangular `k(dist)` into `out`.
    ///
    /// # Errors
    ///
    /// Returns the same shape errors as [`Self::apply`].
    pub fn apply_cross(
        &self,
        dist: MatRef<'_, f32>,
        mut out: MatMut<'_, f32>,
        mut scratch: MatMut<'_, f32>,
    ) -> Result<(), GprError> {
        scratch_ok(out.as_ref(), scratch.as_ref())?;
        match self {
            Self::Rbf(leaf) => rbf_cross(leaf.lengthscale(), dist, out),
            Self::Matern(leaf) => matern_cross(leaf.nu(), leaf.lengthscale(), dist, out),
            Self::Periodic(leaf) => periodic_cross(leaf.lengthscale(), leaf.period(), dist, out),
            Self::RationalQuadratic(leaf) => rq_cross(leaf.lengthscale(), leaf.alpha(), dist, out),
            Self::Constant(leaf) => constant_cross(leaf.constant(), dist, out),
            Self::White(_) => white_cross(dist, out),
            Self::Custom(leaf) => leaf.apply_cross_f32(dist, out),
            Self::RbfArd(_)
            | Self::Linear(_)
            | Self::MaternArd(_)
            | Self::RationalQuadraticArd(_) => Err(ard_needs_coords()),
            Self::Sum(terms) => fold_cross(terms, dist, out.as_mut(), scratch.as_mut(), add_rect),
            Self::Product(terms) => {
                fold_cross(terms, dist, out.as_mut(), scratch.as_mut(), mul_rect)
            }
        }
    }

    /// Writes the diagonal `k(x, x)` into `out`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::UnsupportedKernelOperation`] if a sum/product has no
    /// terms, or if a linear leaf is asked for a distance diagonal.
    pub fn fill_diag(&self, out: &mut [f32]) -> Result<(), GprError> {
        match self {
            Self::Rbf(_)
            | Self::RbfArd(_)
            | Self::Matern(_)
            | Self::MaternArd(_)
            | Self::Periodic(_)
            | Self::RationalQuadratic(_)
            | Self::RationalQuadraticArd(_) => {
                out.fill(1.0);
                Ok(())
            }
            Self::Constant(leaf) => {
                out.fill(f32_of(leaf.constant()));
                Ok(())
            }
            Self::White(leaf) => {
                out.fill(f32_of(leaf.variance()));
                Ok(())
            }
            Self::Custom(leaf) => leaf.fill_diag_f32(out),
            Self::Linear(_) => Err(GprError::UnsupportedKernelOperation {
                reason: "linear kernel diagonal needs coordinates".to_owned(),
            }),
            Self::Sum(terms) => fold_diag(terms, out, |a, b| a + b),
            Self::Product(terms) => fold_diag(terms, out, |a, b| a * b),
        }
    }

    /// Writes the diagonal `k(x, x)` using coordinates when the leaf needs them.
    ///
    /// # Errors
    ///
    /// Same as [`Self::fill_diag`], plus an empty coordinate matrix.
    pub fn fill_diag_points(&self, x: MatRef<'_, f32>, out: &mut [f32]) -> Result<(), GprError> {
        match self {
            Self::Linear(leaf) => linear_diag(leaf.variance(), x, out),
            Self::Sum(terms) => fold_diag_points(terms, x, out, |a, b| a + b),
            Self::Product(terms) => fold_diag_points(terms, x, out, |a, b| a * b),
            _ => self.fill_diag(out),
        }
    }

    /// Writes `k` from point coordinates.
    ///
    /// # Errors
    ///
    /// Same shape errors as [`Self::apply`].
    pub fn apply_points(
        &self,
        x: MatRef<'_, f32>,
        mut out: MatMut<'_, f32>,
        uplo: Triangle,
        mut scratch: MatMut<'_, f32>,
    ) -> Result<(), GprError> {
        scratch_ok(out.as_ref(), scratch.as_ref())?;
        match self {
            Self::Rbf(leaf) => iso_points(x, out, uplo, |s| rbf_k(s, leaf.lengthscale())),
            Self::Matern(leaf) => {
                iso_points(x, out, uplo, |s| matern_k(s, leaf.lengthscale(), leaf.nu()))
            }
            Self::Periodic(leaf) => iso_points(x, out, uplo, |s| {
                periodic_k(s, leaf.lengthscale(), leaf.period())
            }),
            Self::RationalQuadratic(leaf) => {
                iso_points(x, out, uplo, |s| rq_k(s, leaf.lengthscale(), leaf.alpha()))
            }
            Self::Custom(leaf) => custom_from_coords(leaf, x, out, uplo),
            Self::RbfArd(leaf) => rbf_ard_apply(leaf, x, out, uplo),
            Self::Linear(leaf) => linear_apply(leaf.variance(), x, out, uplo),
            Self::MaternArd(leaf) => matern_ard_apply(leaf, x, out, uplo),
            Self::RationalQuadraticArd(leaf) => rq_ard_apply(leaf, x, out, uplo),
            Self::Constant(leaf) => {
                require_points_square(x, out.as_ref())?;
                fill_const(out, uplo, f32_of(leaf.constant()))
            }
            Self::White(leaf) => {
                require_points_square(x, out.as_ref())?;
                fill_white(out, uplo, f32_of(leaf.variance()))
            }
            Self::Sum(terms) => {
                fold_points(terms, x, out.as_mut(), uplo, scratch.as_mut(), add_tri)
            }
            Self::Product(terms) => {
                fold_points(terms, x, out.as_mut(), uplo, scratch.as_mut(), mul_tri)
            }
        }
    }

    /// Writes rectangular `k(x, xs)` from coordinates.
    ///
    /// # Errors
    ///
    /// Same as [`Self::apply_points`]. Isotropic leaves still need a distance
    /// matrix, matching the `f64` path.
    pub fn apply_cross_points(
        &self,
        x: MatRef<'_, f32>,
        xs: MatRef<'_, f32>,
        mut out: MatMut<'_, f32>,
        mut scratch: MatMut<'_, f32>,
    ) -> Result<(), GprError> {
        scratch_ok(out.as_ref(), scratch.as_ref())?;
        match self {
            Self::Rbf(_)
            | Self::Matern(_)
            | Self::Periodic(_)
            | Self::RationalQuadratic(_)
            | Self::Custom(_) => Err(iso_needs_dist()),
            Self::RbfArd(leaf) => rbf_ard_cross(leaf, x, xs, out),
            Self::Linear(leaf) => linear_cross(leaf.variance(), x, xs, out),
            Self::MaternArd(leaf) => matern_ard_cross(leaf, x, xs, out),
            Self::RationalQuadraticArd(leaf) => rq_ard_cross(leaf, x, xs, out),
            Self::Constant(leaf) => {
                require_cross_points(x, xs, out.as_ref())?;
                out.fill(f32_of(leaf.constant()));
                Ok(())
            }
            Self::White(leaf) => {
                let _ = leaf;
                require_cross_points(x, xs, out.as_ref())?;
                out.fill(0.0);
                Ok(())
            }
            Self::Sum(terms) => {
                fold_cross_points(terms, x, xs, out.as_mut(), scratch.as_mut(), add_rect)
            }
            Self::Product(terms) => {
                fold_cross_points(terms, x, xs, out.as_mut(), scratch.as_mut(), mul_rect)
            }
        }
    }

    /// Writes `∂K/∂θ_{param_idx}` into `d_k`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if `param_idx` is out of
    /// range, or the same shape errors as [`Self::apply`].
    pub fn grad(
        &self,
        dist: MatRef<'_, f32>,
        mut d_k: MatMut<'_, f32>,
        param_idx: usize,
        uplo: Triangle,
        mut scratch: MatMut<'_, f32>,
    ) -> Result<(), GprError> {
        match self {
            Self::Rbf(leaf) => rbf_grad(leaf.lengthscale(), dist, d_k, param_idx, uplo),
            Self::Matern(leaf) => {
                matern_grad(leaf.nu(), leaf.lengthscale(), dist, d_k, param_idx, uplo)
            }
            Self::Periodic(leaf) => periodic_grad(
                leaf.lengthscale(),
                leaf.period(),
                dist,
                d_k,
                param_idx,
                uplo,
            ),
            Self::RationalQuadratic(leaf) => {
                rq_grad(leaf.lengthscale(), leaf.alpha(), dist, d_k, param_idx, uplo)
            }
            Self::Constant(leaf) => constant_grad(leaf.constant(), dist, d_k, param_idx, uplo),
            Self::White(leaf) => white_grad(leaf.variance(), dist, d_k, param_idx, uplo),
            Self::Custom(leaf) => leaf.grad_f32(dist, d_k, param_idx, uplo),
            Self::RbfArd(_)
            | Self::Linear(_)
            | Self::MaternArd(_)
            | Self::RationalQuadraticArd(_) => Err(ard_needs_coords()),
            Self::Sum(terms) => {
                let (term, local) = term_for_param(terms, param_idx)?;
                term.grad(dist, d_k, local, uplo, scratch)
            }
            Self::Product(terms) => {
                scratch_ok(d_k.as_ref(), scratch.as_ref())?;
                product_grad(terms, dist, d_k.as_mut(), param_idx, uplo, scratch.as_mut())
            }
        }
    }

    /// Writes `∂K/∂θ_{param_idx}` from point coordinates.
    ///
    /// # Errors
    ///
    /// Same as [`Self::grad`], with coordinates in place of distances.
    pub fn grad_points(
        &self,
        x: MatRef<'_, f32>,
        mut d_k: MatMut<'_, f32>,
        param_idx: usize,
        uplo: Triangle,
        mut scratch: MatMut<'_, f32>,
    ) -> Result<(), GprError> {
        match self {
            Self::Rbf(leaf) => iso_points(x, d_k.as_mut(), uplo, |s| {
                rbf_dk(s, leaf.lengthscale(), param_idx)
            }),
            Self::Matern(leaf) => iso_points(x, d_k.as_mut(), uplo, |s| {
                matern_dk(s, leaf.lengthscale(), leaf.nu(), param_idx)
            }),
            Self::Periodic(leaf) => iso_points(x, d_k.as_mut(), uplo, |s| {
                periodic_dk(s, leaf.lengthscale(), leaf.period(), param_idx)
            }),
            Self::RationalQuadratic(leaf) => iso_points(x, d_k.as_mut(), uplo, |s| {
                rq_dk(s, leaf.lengthscale(), leaf.alpha(), param_idx)
            }),
            Self::Custom(leaf) => custom_grad_from_coords(leaf, x, d_k, param_idx, uplo),
            Self::RbfArd(leaf) => rbf_ard_grad(leaf, x, d_k, param_idx, uplo),
            Self::Linear(leaf) => linear_grad(leaf.variance(), x, d_k, param_idx, uplo),
            Self::MaternArd(leaf) => matern_ard_grad(leaf, x, d_k, param_idx, uplo),
            Self::RationalQuadraticArd(leaf) => rq_ard_grad(leaf, x, d_k, param_idx, uplo),
            Self::Constant(leaf) => {
                require_points_square(x, d_k.as_ref())?;
                one_index(param_idx, "constant")?;
                fill_const(d_k, uplo, f32_of(leaf.constant()))
            }
            Self::White(leaf) => {
                require_points_square(x, d_k.as_ref())?;
                one_index(param_idx, "white kernel")?;
                fill_white(d_k, uplo, f32_of(leaf.variance()))
            }
            Self::Sum(terms) => {
                let (term, local) = term_for_param(terms, param_idx)?;
                term.grad_points(x, d_k, local, uplo, scratch)
            }
            Self::Product(terms) => {
                scratch_ok(d_k.as_ref(), scratch.as_ref())?;
                product_grad_points(terms, x, d_k.as_mut(), param_idx, uplo, scratch.as_mut())
            }
        }
    }

    /// Writes `∂²K/∂θ_i ∂θ_j` from squared distances into `d2_k`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if `i` or `j` is out of
    /// range, or the same shape errors as [`Self::apply`].
    pub fn hess(
        &self,
        dist: MatRef<'_, f32>,
        mut d2_k: MatMut<'_, f32>,
        i: usize,
        j: usize,
        uplo: Triangle,
        mut scratch: MatMut<'_, f32>,
    ) -> Result<(), GprError> {
        match self {
            Self::Rbf(leaf) => rbf_hess(leaf.lengthscale(), dist, d2_k, i, j, uplo),
            Self::Matern(leaf) => {
                matern_hess(leaf.nu(), leaf.lengthscale(), dist, d2_k, i, j, uplo)
            }
            Self::Periodic(leaf) => {
                periodic_hess(leaf.lengthscale(), leaf.period(), dist, d2_k, i, j, uplo)
            }
            Self::RationalQuadratic(leaf) => {
                rq_hess(leaf.lengthscale(), leaf.alpha(), dist, d2_k, i, j, uplo)
            }
            Self::Constant(leaf) => constant_hess(leaf.constant(), dist, d2_k, i, j, uplo),
            Self::White(leaf) => white_hess(leaf.variance(), dist, d2_k, i, j, uplo),
            Self::Custom(leaf) => leaf.hess_f32(dist, d2_k, i, j, uplo),
            Self::RbfArd(_)
            | Self::Linear(_)
            | Self::MaternArd(_)
            | Self::RationalQuadraticArd(_) => Err(ard_needs_coords()),
            Self::Sum(terms) => match owners(terms, i, j)? {
                Owners::Same {
                    term,
                    local_i,
                    local_j,
                } => term.hess(dist, d2_k, local_i, local_j, uplo, scratch),
                Owners::Distinct { .. } => {
                    zero_tri(d2_k.as_mut(), uplo);
                    Ok(())
                }
            },
            Self::Product(terms) => {
                scratch_ok(d2_k.as_ref(), scratch.as_ref())?;
                product_hess(terms, dist, d2_k.as_mut(), i, j, uplo, scratch.as_mut())
            }
        }
    }

    /// Writes `∂²K/∂θ_i ∂θ_j` from point coordinates.
    ///
    /// # Errors
    ///
    /// Same as [`Self::hess`].
    pub fn hess_points(
        &self,
        x: MatRef<'_, f32>,
        mut d2_k: MatMut<'_, f32>,
        i: usize,
        j: usize,
        uplo: Triangle,
        mut scratch: MatMut<'_, f32>,
    ) -> Result<(), GprError> {
        match self {
            Self::Rbf(leaf) => iso_points(x, d2_k.as_mut(), uplo, |s| {
                rbf_d2(s, leaf.lengthscale(), i, j)
            }),
            Self::Matern(leaf) => iso_points(x, d2_k.as_mut(), uplo, |s| {
                matern_d2(s, leaf.lengthscale(), leaf.nu(), i, j)
            }),
            Self::Periodic(leaf) => iso_points(x, d2_k.as_mut(), uplo, |s| {
                periodic_d2(s, leaf.lengthscale(), leaf.period(), i, j)
            }),
            Self::RationalQuadratic(leaf) => iso_points(x, d2_k.as_mut(), uplo, |s| {
                rq_d2(s, leaf.lengthscale(), leaf.alpha(), i, j)
            }),
            Self::Custom(leaf) => leaf.hess_points_f32(x, d2_k, i, j, uplo),
            Self::RbfArd(leaf) => rbf_ard_hess(leaf, x, d2_k, i, j, uplo),
            Self::Linear(leaf) => linear_hess(leaf.variance(), x, d2_k, i, j, uplo),
            Self::MaternArd(leaf) => matern_ard_hess(leaf, x, d2_k, i, j, uplo),
            Self::RationalQuadraticArd(leaf) => rq_ard_hess(leaf, x, d2_k, i, j, uplo),
            Self::Constant(leaf) => {
                require_points_square(x, d2_k.as_ref())?;
                pair_index(i, j, 1, "constant")?;
                fill_const(d2_k, uplo, f32_of(leaf.constant()))
            }
            Self::White(leaf) => {
                require_points_square(x, d2_k.as_ref())?;
                pair_index(i, j, 1, "white kernel")?;
                fill_white(d2_k, uplo, f32_of(leaf.variance()))
            }
            Self::Sum(terms) => match owners(terms, i, j)? {
                Owners::Same {
                    term,
                    local_i,
                    local_j,
                } => term.hess_points(x, d2_k, local_i, local_j, uplo, scratch),
                Owners::Distinct { .. } => {
                    zero_tri(d2_k.as_mut(), uplo);
                    Ok(())
                }
            },
            Self::Product(terms) => {
                scratch_ok(d2_k.as_ref(), scratch.as_ref())?;
                product_hess_points(terms, x, d2_k.as_mut(), i, j, uplo, scratch.as_mut())
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
        x1: MatRef<'_, f32>,
        x2: MatRef<'_, f32>,
        mut d_k: MatMut<'_, f32>,
        dim: usize,
    ) -> Result<(), GprError> {
        match self {
            Self::Rbf(leaf) => rbf_coord(leaf.lengthscale(), x1, x2, d_k, dim),
            Self::Matern(leaf) => matern_coord(leaf.nu(), leaf.lengthscale(), x1, x2, d_k, dim),
            Self::RbfArd(leaf) => rbf_ard_coord(leaf, x1, x2, d_k, dim),
            Self::White(_) => {
                require_coord(x1, x2, d_k.as_ref(), dim)?;
                d_k.fill(0.0);
                Ok(())
            }
            Self::Custom(leaf) => leaf.grad_wrt_coord_dim_f32(x1, x2, d_k, dim),
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
                    add_rect(d_k.as_mut(), scratch.as_ref());
                }
                Ok(())
            }
            _ => Err(GprError::CoordGradientUnsupported),
        }
    }
}

fn f32_of(value: f64) -> f32 {
    value as f32
}

fn finite(value: f32) -> Result<f32, GprError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(GprError::NonFiniteKernelValue)
    }
}

fn finite_dist(value: f32) -> Result<f32, GprError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(GprError::NonFiniteInput)
    }
}

fn one_index(idx: usize, name: &str) -> Result<(), GprError> {
    if idx == 0 {
        Ok(())
    } else {
        Err(GprError::InvalidHyperparameter {
            reason: format!("{name} has a single parameter at index 0"),
        })
    }
}

fn pair_index(i: usize, j: usize, n: usize, name: &str) -> Result<(), GprError> {
    if i < n && j < n {
        Ok(())
    } else {
        Err(GprError::InvalidHyperparameter {
            reason: format!("{name} parameter pair ({i}, {j}) is out of range"),
        })
    }
}

fn scratch_ok(out: MatRef<'_, f32>, scratch: MatRef<'_, f32>) -> Result<(), GprError> {
    if scratch.nrows() == out.nrows() && scratch.ncols() == out.ncols() {
        Ok(())
    } else {
        Err(GprError::WorkspaceTooSmall)
    }
}

fn square_pair(dist: MatRef<'_, f32>, out: MatRef<'_, f32>) -> Result<usize, GprError> {
    if dist.nrows() != dist.ncols() {
        return Err(GprError::InvalidHyperparameter {
            reason: format!(
                "distance matrix must be square, got {}x{}",
                dist.nrows(),
                dist.ncols()
            ),
        });
    }
    if out.nrows() != dist.nrows() || out.ncols() != dist.ncols() {
        return Err(GprError::InvalidHyperparameter {
            reason: format!(
                "output is {}x{}, expected {}x{}",
                out.nrows(),
                out.ncols(),
                dist.nrows(),
                dist.ncols()
            ),
        });
    }
    if dist.nrows() == 0 {
        return Err(GprError::EmptyInput);
    }
    Ok(dist.nrows())
}

fn same_shape(dist: MatRef<'_, f32>, out: MatRef<'_, f32>) -> Result<(), GprError> {
    if out.nrows() != dist.nrows() || out.ncols() != dist.ncols() {
        return Err(GprError::InvalidHyperparameter {
            reason: format!(
                "output is {}x{}, expected {}x{}",
                out.nrows(),
                out.ncols(),
                dist.nrows(),
                dist.ncols()
            ),
        });
    }
    if dist.nrows() == 0 || dist.ncols() == 0 {
        return Err(GprError::EmptyInput);
    }
    Ok(())
}

fn require_points_square(x: MatRef<'_, f32>, out: MatRef<'_, f32>) -> Result<usize, GprError> {
    let n = out.nrows();
    if out.ncols() != n {
        return Err(GprError::InvalidHyperparameter {
            reason: format!("output must be square, got {}x{}", out.nrows(), out.ncols()),
        });
    }
    if n == 0 || x.nrows() != n || x.ncols() == 0 {
        return Err(GprError::EmptyInput);
    }
    Ok(n)
}

fn require_feature(x: MatRef<'_, f32>, expected: usize) -> Result<(), GprError> {
    if x.nrows() == 0 || x.ncols() == 0 {
        return Err(GprError::EmptyInput);
    }
    if x.ncols() != expected {
        return Err(GprError::DimensionMismatch {
            x_dim: x.ncols(),
            expected_dim: expected,
        });
    }
    Ok(())
}

fn require_cross_points(
    x: MatRef<'_, f32>,
    xs: MatRef<'_, f32>,
    out: MatRef<'_, f32>,
) -> Result<(), GprError> {
    if x.nrows() == 0 || xs.nrows() == 0 || x.ncols() == 0 {
        return Err(GprError::EmptyInput);
    }
    if x.ncols() != xs.ncols() {
        return Err(GprError::DimensionMismatch {
            x_dim: xs.ncols(),
            expected_dim: x.ncols(),
        });
    }
    if out.nrows() != x.nrows() || out.ncols() != xs.nrows() {
        return Err(GprError::InvalidHyperparameter {
            reason: format!(
                "output is {}x{}, expected {}x{}",
                out.nrows(),
                out.ncols(),
                x.nrows(),
                xs.nrows()
            ),
        });
    }
    Ok(())
}

fn require_coord(
    x1: MatRef<'_, f32>,
    x2: MatRef<'_, f32>,
    d_k: MatRef<'_, f32>,
    dim: usize,
) -> Result<(), GprError> {
    if x1.nrows() == 0 || x2.nrows() == 0 || x1.ncols() == 0 {
        return Err(GprError::EmptyInput);
    }
    if x1.ncols() != x2.ncols() {
        return Err(GprError::DimensionMismatch {
            x_dim: x2.ncols(),
            expected_dim: x1.ncols(),
        });
    }
    if dim >= x1.ncols() {
        return Err(GprError::InvalidHyperparameter {
            reason: format!(
                "coordinate dimension {dim} is out of range for d={}",
                x1.ncols()
            ),
        });
    }
    if d_k.nrows() != x1.nrows() || d_k.ncols() != x2.nrows() {
        return Err(GprError::InvalidHyperparameter {
            reason: format!(
                "output is {}x{}, expected {}x{}",
                d_k.nrows(),
                d_k.ncols(),
                x1.nrows(),
                x2.nrows()
            ),
        });
    }
    Ok(())
}

fn walk_tri(
    dist: MatRef<'_, f32>,
    mut out: MatMut<'_, f32>,
    uplo: Triangle,
    mut kernel: impl FnMut(f32) -> Result<f32, GprError>,
) -> Result<(), GprError> {
    let n = square_pair(dist, out.as_ref())?;
    let mut err = None;
    visit_triangle(n, uplo, |row, col| {
        if err.is_some() {
            return;
        }
        match finite_dist(dist[(row, col)]).and_then(&mut kernel) {
            Ok(value) => out[(row, col)] = value,
            Err(e) => err = Some(e),
        }
    });
    match err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

fn walk_dense(
    dist: MatRef<'_, f32>,
    mut out: MatMut<'_, f32>,
    mut kernel: impl FnMut(f32) -> Result<f32, GprError>,
) -> Result<(), GprError> {
    same_shape(dist, out.as_ref())?;
    for col in 0..dist.ncols() {
        for row in 0..dist.nrows() {
            out[(row, col)] = kernel(finite_dist(dist[(row, col)])?)?;
        }
    }
    Ok(())
}

fn iso_points(
    x: MatRef<'_, f32>,
    mut out: MatMut<'_, f32>,
    uplo: Triangle,
    mut kernel: impl FnMut(f32) -> Result<f32, GprError>,
) -> Result<(), GprError> {
    let n = require_points_square(x, out.as_ref())?;
    let mut err = None;
    visit_triangle(n, uplo, |row, col| {
        if err.is_some() {
            return;
        }
        match sq_pair(x, row, x, col).and_then(&mut kernel) {
            Ok(value) => out[(row, col)] = value,
            Err(e) => err = Some(e),
        }
    });
    match err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

fn sq_pair(
    x: MatRef<'_, f32>,
    row: usize,
    xs: MatRef<'_, f32>,
    col: usize,
) -> Result<f32, GprError> {
    let mut s = 0.0f32;
    for dim in 0..x.ncols() {
        let a = x[(row, dim)];
        let b = xs[(col, dim)];
        if !a.is_finite() || !b.is_finite() {
            return Err(GprError::NonFiniteInput);
        }
        let d = a - b;
        s += d * d;
    }
    finite(s)
}

fn dot_pair(
    x: MatRef<'_, f32>,
    row: usize,
    xs: MatRef<'_, f32>,
    col: usize,
) -> Result<f32, GprError> {
    let mut s = 0.0f32;
    for dim in 0..x.ncols() {
        let a = x[(row, dim)];
        let b = xs[(col, dim)];
        if !a.is_finite() || !b.is_finite() {
            return Err(GprError::NonFiniteInput);
        }
        s += a * b;
    }
    finite(s)
}

fn fill_const(mut out: MatMut<'_, f32>, uplo: Triangle, value: f32) -> Result<(), GprError> {
    visit_triangle(out.nrows(), uplo, |row, col| out[(row, col)] = value);
    Ok(())
}

fn fill_white(mut out: MatMut<'_, f32>, uplo: Triangle, variance: f32) -> Result<(), GprError> {
    visit_triangle(out.nrows(), uplo, |row, col| {
        out[(row, col)] = if row == col { variance } else { 0.0 };
    });
    Ok(())
}

fn zero_tri(mut out: MatMut<'_, f32>, uplo: Triangle) {
    visit_triangle(out.nrows(), uplo, |row, col| out[(row, col)] = 0.0);
}

fn add_tri(mut acc: MatMut<'_, f32>, src: MatRef<'_, f32>, uplo: Triangle) {
    visit_triangle(acc.nrows(), uplo, |row, col| {
        acc[(row, col)] += src[(row, col)];
    });
}

fn mul_tri(mut acc: MatMut<'_, f32>, src: MatRef<'_, f32>, uplo: Triangle) {
    visit_triangle(acc.nrows(), uplo, |row, col| {
        acc[(row, col)] *= src[(row, col)];
    });
}

fn add_rect(mut acc: MatMut<'_, f32>, src: MatRef<'_, f32>) {
    for col in 0..acc.ncols() {
        for row in 0..acc.nrows() {
            acc[(row, col)] += src[(row, col)];
        }
    }
}

fn mul_rect(mut acc: MatMut<'_, f32>, src: MatRef<'_, f32>) {
    for col in 0..acc.ncols() {
        for row in 0..acc.nrows() {
            acc[(row, col)] *= src[(row, col)];
        }
    }
}

fn rbf_k(s: f32, ell: f64) -> Result<f32, GprError> {
    let ell = f32_of(ell);
    let inv_two = 0.5 / (ell * ell);
    finite((-s * inv_two).exp())
}

fn rbf_dk(s: f32, ell: f64, idx: usize) -> Result<f32, GprError> {
    one_index(idx, "RBF")?;
    let ell = f32_of(ell);
    let inv_ell_sq = 1.0 / (ell * ell);
    let k = finite((-s * 0.5 * inv_ell_sq).exp())?;
    finite(k * s * inv_ell_sq)
}

fn rbf_d2(s: f32, ell: f64, i: usize, j: usize) -> Result<f32, GprError> {
    pair_index(i, j, 1, "RBF")?;
    let ell = f32_of(ell);
    let inv_ell_sq = 1.0 / (ell * ell);
    let k = finite((-s * 0.5 * inv_ell_sq).exp())?;
    let u = s * inv_ell_sq;
    finite(k * u * (u - 2.0))
}

fn rbf_apply(
    ell: f64,
    dist: MatRef<'_, f32>,
    out: MatMut<'_, f32>,
    uplo: Triangle,
) -> Result<(), GprError> {
    walk_tri(dist, out, uplo, |s| rbf_k(s, ell))
}

fn rbf_cross(ell: f64, dist: MatRef<'_, f32>, out: MatMut<'_, f32>) -> Result<(), GprError> {
    walk_dense(dist, out, |s| rbf_k(s, ell))
}

fn rbf_grad(
    ell: f64,
    dist: MatRef<'_, f32>,
    out: MatMut<'_, f32>,
    idx: usize,
    uplo: Triangle,
) -> Result<(), GprError> {
    walk_tri(dist, out, uplo, |s| rbf_dk(s, ell, idx))
}

fn rbf_hess(
    ell: f64,
    dist: MatRef<'_, f32>,
    out: MatMut<'_, f32>,
    i: usize,
    j: usize,
    uplo: Triangle,
) -> Result<(), GprError> {
    walk_tri(dist, out, uplo, |s| rbf_d2(s, ell, i, j))
}

fn matern_from_r(nu: MaternNu, r: f32) -> f32 {
    match nu {
        MaternNu::Half => (-r).exp(),
        MaternNu::ThreeHalves => {
            let rho = 3.0f32.sqrt() * r;
            (1.0 + rho) * (-rho).exp()
        }
        MaternNu::FiveHalves => {
            let rho = 5.0f32.sqrt() * r;
            (1.0 + rho + rho * rho / 3.0) * (-rho).exp()
        }
    }
}

fn matern_dk_iso(nu: MaternNu, r: f32) -> f32 {
    match nu {
        MaternNu::Half => matern_from_r(nu, r) * r,
        MaternNu::ThreeHalves => {
            let rho = 3.0f32.sqrt() * r;
            rho * rho * (-rho).exp()
        }
        MaternNu::FiveHalves => {
            let rho = 5.0f32.sqrt() * r;
            (rho * rho / 3.0) * (1.0 + rho) * (-rho).exp()
        }
    }
}

fn matern_d2_iso(nu: MaternNu, r: f32) -> f32 {
    match nu {
        MaternNu::Half => {
            let k = matern_from_r(nu, r);
            k * r * (r - 1.0)
        }
        MaternNu::ThreeHalves => {
            let rho = 3.0f32.sqrt() * r;
            rho * rho * (rho - 2.0) * (-rho).exp()
        }
        MaternNu::FiveHalves => {
            let rho = 5.0f32.sqrt() * r;
            (rho * rho / 3.0) * (rho * rho - 2.0 * rho - 2.0) * (-rho).exp()
        }
    }
}

fn matern_dk_ard(nu: MaternNu, r: f32, dim_term: f32) -> f32 {
    match nu {
        MaternNu::Half => {
            if r <= 0.0 {
                0.0
            } else {
                matern_from_r(nu, r) * dim_term / r
            }
        }
        MaternNu::ThreeHalves => 3.0 * dim_term * (-(3.0f32.sqrt() * r)).exp(),
        MaternNu::FiveHalves => {
            let rho = 5.0f32.sqrt() * r;
            (5.0 / 3.0) * (1.0 + rho) * (-rho).exp() * dim_term
        }
    }
}

fn matern_d2_ard(nu: MaternNu, r: f32, dim_i: f32, dim_j: f32, same: bool) -> f32 {
    if r <= 0.0 {
        return 0.0;
    }
    match nu {
        MaternNu::Half => {
            let k = matern_from_r(nu, r);
            if same {
                k * (dim_i * dim_i / (r * r) - 2.0 * dim_i / r + dim_i * dim_i / (r * r * r))
            } else {
                k * dim_i * dim_j * (1.0 / (r * r) + 1.0 / (r * r * r))
            }
        }
        MaternNu::ThreeHalves => {
            let rho = 3.0f32.sqrt() * r;
            let e = (-rho).exp();
            if same {
                3.0 * e * (rho * dim_i * dim_i / (r * r) - 2.0 * dim_i)
            } else {
                3.0 * e * (rho * dim_i * dim_j / (r * r))
            }
        }
        MaternNu::FiveHalves => {
            let rho = 5.0f32.sqrt() * r;
            let e = (-rho).exp();
            if same {
                (5.0 / 3.0) * e * (rho * rho * dim_i * dim_i / (r * r) - 2.0 * (1.0 + rho) * dim_i)
            } else {
                (5.0 / 3.0) * e * (rho * rho * dim_i * dim_j / (r * r))
            }
        }
    }
}

fn scaled_r(s: f32, ell: f64) -> Result<f32, GprError> {
    let ell = f32_of(ell);
    finite(s.max(0.0).sqrt() / ell)
}

fn matern_k(s: f32, ell: f64, nu: MaternNu) -> Result<f32, GprError> {
    finite(matern_from_r(nu, scaled_r(s, ell)?))
}

fn matern_dk(s: f32, ell: f64, nu: MaternNu, idx: usize) -> Result<f32, GprError> {
    one_index(idx, "Matern")?;
    finite(matern_dk_iso(nu, scaled_r(s, ell)?))
}

fn matern_d2(s: f32, ell: f64, nu: MaternNu, i: usize, j: usize) -> Result<f32, GprError> {
    pair_index(i, j, 1, "Matern")?;
    finite(matern_d2_iso(nu, scaled_r(s, ell)?))
}

fn matern_apply(
    nu: MaternNu,
    ell: f64,
    dist: MatRef<'_, f32>,
    out: MatMut<'_, f32>,
    uplo: Triangle,
) -> Result<(), GprError> {
    walk_tri(dist, out, uplo, |s| matern_k(s, ell, nu))
}

fn matern_cross(
    nu: MaternNu,
    ell: f64,
    dist: MatRef<'_, f32>,
    out: MatMut<'_, f32>,
) -> Result<(), GprError> {
    walk_dense(dist, out, |s| matern_k(s, ell, nu))
}

fn matern_grad(
    nu: MaternNu,
    ell: f64,
    dist: MatRef<'_, f32>,
    out: MatMut<'_, f32>,
    idx: usize,
    uplo: Triangle,
) -> Result<(), GprError> {
    walk_tri(dist, out, uplo, |s| matern_dk(s, ell, nu, idx))
}

fn matern_hess(
    nu: MaternNu,
    ell: f64,
    dist: MatRef<'_, f32>,
    out: MatMut<'_, f32>,
    i: usize,
    j: usize,
    uplo: Triangle,
) -> Result<(), GprError> {
    walk_tri(dist, out, uplo, |s| matern_d2(s, ell, nu, i, j))
}

fn periodic_k(s: f32, ell: f64, period: f64) -> Result<f32, GprError> {
    let r = finite(s.max(0.0).sqrt())?;
    let ell = f32_of(ell);
    let period = f32_of(period);
    let sine = (std::f32::consts::PI * r / period).sin();
    let inv = 1.0 / ell;
    finite((-2.0 * sine * sine * inv * inv).exp())
}

fn periodic_dk(s: f32, ell: f64, period: f64, idx: usize) -> Result<f32, GprError> {
    pair_index(idx, 0, 2, "periodic")?;
    let r = finite(s.max(0.0).sqrt())?;
    let ell = f32_of(ell);
    let period = f32_of(period);
    let alpha = std::f32::consts::PI * r / period;
    let sine = alpha.sin();
    let inv_ell_sq = 1.0 / (ell * ell);
    let k = finite((-2.0 * sine * sine * inv_ell_sq).exp())?;
    let dk = if idx == 0 {
        k * 4.0 * sine * sine * inv_ell_sq
    } else {
        k * 4.0 * sine * alpha.cos() * alpha * inv_ell_sq
    };
    finite(dk)
}

fn periodic_d2(s: f32, ell: f64, period: f64, i: usize, j: usize) -> Result<f32, GprError> {
    pair_index(i, j, 2, "periodic")?;
    let r = finite(s.max(0.0).sqrt())?;
    let ell = f32_of(ell);
    let period = f32_of(period);
    let alpha = std::f32::consts::PI * r / period;
    let sine = alpha.sin();
    let cosine = alpha.cos();
    let inv_ell_sq = 1.0 / (ell * ell);
    let k = finite((-2.0 * sine * sine * inv_ell_sq).exp())?;
    let beta = 4.0 * sine * sine * inv_ell_sq;
    let gamma = 4.0 * sine * cosine * alpha * inv_ell_sq;
    let (a, b) = if i <= j { (i, j) } else { (j, i) };
    let h = match (a, b) {
        (0, 0) => k * beta * (beta - 2.0),
        (0, 1) => k * gamma * (beta - 2.0),
        (1, 1) => {
            let dgamma = 4.0
                * inv_ell_sq
                * (-alpha * alpha * cosine * cosine + alpha * alpha * sine * sine
                    - alpha * sine * cosine);
            k * gamma * gamma + k * dgamma
        }
        _ => 0.0,
    };
    finite(h)
}

fn periodic_apply(
    ell: f64,
    period: f64,
    dist: MatRef<'_, f32>,
    out: MatMut<'_, f32>,
    uplo: Triangle,
) -> Result<(), GprError> {
    walk_tri(dist, out, uplo, |s| periodic_k(s, ell, period))
}

fn periodic_cross(
    ell: f64,
    period: f64,
    dist: MatRef<'_, f32>,
    out: MatMut<'_, f32>,
) -> Result<(), GprError> {
    walk_dense(dist, out, |s| periodic_k(s, ell, period))
}

fn periodic_grad(
    ell: f64,
    period: f64,
    dist: MatRef<'_, f32>,
    out: MatMut<'_, f32>,
    idx: usize,
    uplo: Triangle,
) -> Result<(), GprError> {
    walk_tri(dist, out, uplo, |s| periodic_dk(s, ell, period, idx))
}

fn periodic_hess(
    ell: f64,
    period: f64,
    dist: MatRef<'_, f32>,
    out: MatMut<'_, f32>,
    i: usize,
    j: usize,
    uplo: Triangle,
) -> Result<(), GprError> {
    walk_tri(dist, out, uplo, |s| periodic_d2(s, ell, period, i, j))
}

fn rq_from(r2: f32, alpha: f32) -> Result<f32, GprError> {
    let u = 1.0 + r2 / (2.0 * alpha);
    finite(u.powf(-alpha))
}

fn rq_k(s: f32, ell: f64, alpha: f64) -> Result<f32, GprError> {
    let ell = f32_of(ell);
    let alpha = f32_of(alpha);
    rq_from(s / (ell * ell), alpha)
}

fn rq_dk(s: f32, ell: f64, alpha: f64, idx: usize) -> Result<f32, GprError> {
    pair_index(idx, 0, 2, "rational quadratic")?;
    let ell = f32_of(ell);
    let alpha = f32_of(alpha);
    let r2 = s / (ell * ell);
    let u = 1.0 + r2 / (2.0 * alpha);
    let k = rq_from(r2, alpha)?;
    if idx == 0 {
        finite((k / u) * r2)
    } else {
        finite(alpha * k * (-u.ln() + 1.0 - 1.0 / u))
    }
}

fn rq_d2(s: f32, ell: f64, alpha: f64, i: usize, j: usize) -> Result<f32, GprError> {
    pair_index(i, j, 2, "rational quadratic")?;
    let ell = f32_of(ell);
    let alpha = f32_of(alpha);
    let r2 = s / (ell * ell);
    let u = 1.0 + r2 / (2.0 * alpha);
    let k = rq_from(r2, alpha)?;
    let (a, b) = if i <= j { (i, j) } else { (j, i) };
    let h = match (a, b) {
        (0, 0) => -2.0 * (k / u) * r2 + (1.0 + 1.0 / alpha) * r2 * r2 * k / (u * u),
        (0, 1) => (k / u) * r2 * (-alpha * u.ln() + (alpha + 1.0) * (u - 1.0) / u),
        (1, 1) => {
            let v = -u.ln() + 1.0 - 1.0 / u;
            let dv = (1.0 - u) * (1.0 - u) / (u * u);
            let h = alpha * k * v;
            h + (h * h) / k + alpha * k * dv
        }
        _ => 0.0,
    };
    finite(h)
}

fn rq_apply(
    ell: f64,
    alpha: f64,
    dist: MatRef<'_, f32>,
    out: MatMut<'_, f32>,
    uplo: Triangle,
) -> Result<(), GprError> {
    walk_tri(dist, out, uplo, |s| rq_k(s, ell, alpha))
}

fn rq_cross(
    ell: f64,
    alpha: f64,
    dist: MatRef<'_, f32>,
    out: MatMut<'_, f32>,
) -> Result<(), GprError> {
    walk_dense(dist, out, |s| rq_k(s, ell, alpha))
}

fn rq_grad(
    ell: f64,
    alpha: f64,
    dist: MatRef<'_, f32>,
    out: MatMut<'_, f32>,
    idx: usize,
    uplo: Triangle,
) -> Result<(), GprError> {
    walk_tri(dist, out, uplo, |s| rq_dk(s, ell, alpha, idx))
}

fn rq_hess(
    ell: f64,
    alpha: f64,
    dist: MatRef<'_, f32>,
    out: MatMut<'_, f32>,
    i: usize,
    j: usize,
    uplo: Triangle,
) -> Result<(), GprError> {
    walk_tri(dist, out, uplo, |s| rq_d2(s, ell, alpha, i, j))
}

fn constant_apply(
    c: f64,
    dist: MatRef<'_, f32>,
    out: MatMut<'_, f32>,
    uplo: Triangle,
) -> Result<(), GprError> {
    square_pair(dist, out.as_ref())?;
    fill_const(out, uplo, f32_of(c))
}

fn constant_cross(c: f64, dist: MatRef<'_, f32>, mut out: MatMut<'_, f32>) -> Result<(), GprError> {
    same_shape(dist, out.as_ref())?;
    out.fill(f32_of(c));
    Ok(())
}

fn constant_grad(
    c: f64,
    dist: MatRef<'_, f32>,
    out: MatMut<'_, f32>,
    idx: usize,
    uplo: Triangle,
) -> Result<(), GprError> {
    one_index(idx, "constant")?;
    constant_apply(c, dist, out, uplo)
}

fn constant_hess(
    c: f64,
    dist: MatRef<'_, f32>,
    out: MatMut<'_, f32>,
    i: usize,
    j: usize,
    uplo: Triangle,
) -> Result<(), GprError> {
    pair_index(i, j, 1, "constant")?;
    constant_apply(c, dist, out, uplo)
}

fn white_apply(
    variance: f64,
    dist: MatRef<'_, f32>,
    out: MatMut<'_, f32>,
    uplo: Triangle,
) -> Result<(), GprError> {
    square_pair(dist, out.as_ref())?;
    fill_white(out, uplo, f32_of(variance))
}

fn white_cross(dist: MatRef<'_, f32>, mut out: MatMut<'_, f32>) -> Result<(), GprError> {
    same_shape(dist, out.as_ref())?;
    out.fill(0.0);
    Ok(())
}

fn white_grad(
    variance: f64,
    dist: MatRef<'_, f32>,
    out: MatMut<'_, f32>,
    idx: usize,
    uplo: Triangle,
) -> Result<(), GprError> {
    one_index(idx, "white kernel")?;
    white_apply(variance, dist, out, uplo)
}

fn white_hess(
    variance: f64,
    dist: MatRef<'_, f32>,
    out: MatMut<'_, f32>,
    i: usize,
    j: usize,
    uplo: Triangle,
) -> Result<(), GprError> {
    pair_index(i, j, 1, "white kernel")?;
    white_apply(variance, dist, out, uplo)
}

fn linear_apply(
    variance: f64,
    x: MatRef<'_, f32>,
    mut out: MatMut<'_, f32>,
    uplo: Triangle,
) -> Result<(), GprError> {
    let n = require_points_square(x, out.as_ref())?;
    let var = f32_of(variance);
    let mut err = None;
    visit_triangle(n, uplo, |row, col| {
        if err.is_some() {
            return;
        }
        match dot_pair(x, row, x, col) {
            Ok(dot) => out[(row, col)] = var * dot,
            Err(e) => err = Some(e),
        }
    });
    match err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

fn linear_cross(
    variance: f64,
    x: MatRef<'_, f32>,
    xs: MatRef<'_, f32>,
    mut out: MatMut<'_, f32>,
) -> Result<(), GprError> {
    require_cross_points(x, xs, out.as_ref())?;
    let var = f32_of(variance);
    for col in 0..xs.nrows() {
        for row in 0..x.nrows() {
            out[(row, col)] = var * dot_pair(x, row, xs, col)?;
        }
    }
    Ok(())
}

fn linear_diag(variance: f64, x: MatRef<'_, f32>, out: &mut [f32]) -> Result<(), GprError> {
    if x.nrows() == 0 || x.ncols() == 0 {
        return Err(GprError::EmptyInput);
    }
    if out.len() != x.nrows() {
        return Err(GprError::InvalidHyperparameter {
            reason: format!("expected {} diagonal entries, got {}", x.nrows(), out.len()),
        });
    }
    let var = f32_of(variance);
    for (row, slot) in out.iter_mut().enumerate() {
        *slot = var * dot_pair(x, row, x, row)?;
    }
    Ok(())
}

fn linear_grad(
    variance: f64,
    x: MatRef<'_, f32>,
    out: MatMut<'_, f32>,
    idx: usize,
    uplo: Triangle,
) -> Result<(), GprError> {
    one_index(idx, "linear kernel")?;
    linear_apply(variance, x, out, uplo)
}

fn linear_hess(
    variance: f64,
    x: MatRef<'_, f32>,
    out: MatMut<'_, f32>,
    i: usize,
    j: usize,
    uplo: Triangle,
) -> Result<(), GprError> {
    pair_index(i, j, 1, "linear kernel")?;
    linear_apply(variance, x, out, uplo)
}

fn weights(leaf: &crate::kernel::ArdLengthscales) -> Vec<f32> {
    leaf.inv_ell_sq().iter().copied().map(f32_of).collect()
}

fn ard_terms(
    x: MatRef<'_, f32>,
    row: usize,
    xs: MatRef<'_, f32>,
    col: usize,
    w: &[f32],
) -> Result<(f32, Vec<f32>), GprError> {
    let mut r2 = 0.0f32;
    let mut terms = vec![0.0; w.len()];
    for (dim, &weight) in w.iter().enumerate() {
        let a = x[(row, dim)];
        let b = xs[(col, dim)];
        if !a.is_finite() || !b.is_finite() {
            return Err(GprError::NonFiniteInput);
        }
        let term = (a - b) * (a - b) * weight;
        terms[dim] = term;
        r2 += term;
    }
    finite(r2)?;
    Ok((r2, terms))
}

fn rbf_ard_apply(
    leaf: &crate::kernel::RbfArdKernel,
    x: MatRef<'_, f32>,
    mut out: MatMut<'_, f32>,
    uplo: Triangle,
) -> Result<(), GprError> {
    require_feature(x, leaf.num_params())?;
    let n = require_points_square(x, out.as_ref())?;
    let w = weights(leaf.lengthscales());
    let mut err = None;
    visit_triangle(n, uplo, |row, col| {
        if err.is_some() {
            return;
        }
        match ard_terms(x, row, x, col, &w).and_then(|(r2, _)| finite((-0.5 * r2).exp())) {
            Ok(k) => out[(row, col)] = k,
            Err(e) => err = Some(e),
        }
    });
    match err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

fn rbf_ard_cross(
    leaf: &crate::kernel::RbfArdKernel,
    x: MatRef<'_, f32>,
    xs: MatRef<'_, f32>,
    mut out: MatMut<'_, f32>,
) -> Result<(), GprError> {
    require_feature(x, leaf.num_params())?;
    require_cross_points(x, xs, out.as_ref())?;
    let w = weights(leaf.lengthscales());
    for col in 0..xs.nrows() {
        for row in 0..x.nrows() {
            let (r2, _) = ard_terms(x, row, xs, col, &w)?;
            out[(row, col)] = finite((-0.5 * r2).exp())?;
        }
    }
    Ok(())
}

fn rbf_ard_grad(
    leaf: &crate::kernel::RbfArdKernel,
    x: MatRef<'_, f32>,
    mut out: MatMut<'_, f32>,
    idx: usize,
    uplo: Triangle,
) -> Result<(), GprError> {
    if idx >= leaf.num_params() {
        return Err(GprError::InvalidHyperparameter {
            reason: format!(
                "ARD RBF parameter {idx} is out of range (d={})",
                leaf.num_params()
            ),
        });
    }
    require_feature(x, leaf.num_params())?;
    let n = require_points_square(x, out.as_ref())?;
    let w = weights(leaf.lengthscales());
    let mut err = None;
    visit_triangle(n, uplo, |row, col| {
        if err.is_some() {
            return;
        }
        let value = ard_terms(x, row, x, col, &w).and_then(|(r2, terms)| {
            let k = finite((-0.5 * r2).exp())?;
            finite(k * terms[idx])
        });
        match value {
            Ok(v) => out[(row, col)] = v,
            Err(e) => err = Some(e),
        }
    });
    match err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

fn rbf_ard_hess(
    leaf: &crate::kernel::RbfArdKernel,
    x: MatRef<'_, f32>,
    mut out: MatMut<'_, f32>,
    i: usize,
    j: usize,
    uplo: Triangle,
) -> Result<(), GprError> {
    let d = leaf.num_params();
    pair_index(i, j, d, "ARD RBF")?;
    require_feature(x, d)?;
    let n = require_points_square(x, out.as_ref())?;
    let w = weights(leaf.lengthscales());
    let same = i == j;
    let mut err = None;
    visit_triangle(n, uplo, |row, col| {
        if err.is_some() {
            return;
        }
        let value = ard_terms(x, row, x, col, &w).and_then(|(r2, terms)| {
            let k = finite((-0.5 * r2).exp())?;
            let h = if same {
                k * terms[i] * (terms[i] - 2.0)
            } else {
                k * terms[i] * terms[j]
            };
            finite(h)
        });
        match value {
            Ok(v) => out[(row, col)] = v,
            Err(e) => err = Some(e),
        }
    });
    match err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

fn matern_ard_apply(
    leaf: &crate::kernel::MaternArdKernel,
    x: MatRef<'_, f32>,
    mut out: MatMut<'_, f32>,
    uplo: Triangle,
) -> Result<(), GprError> {
    require_feature(x, leaf.num_params())?;
    let n = require_points_square(x, out.as_ref())?;
    let w = weights(leaf.lengthscales());
    let nu = leaf.nu();
    let mut err = None;
    visit_triangle(n, uplo, |row, col| {
        if err.is_some() {
            return;
        }
        match ard_terms(x, row, x, col, &w)
            .and_then(|(r2, _)| finite(matern_from_r(nu, r2.max(0.0).sqrt())))
        {
            Ok(k) => out[(row, col)] = k,
            Err(e) => err = Some(e),
        }
    });
    match err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

fn matern_ard_cross(
    leaf: &crate::kernel::MaternArdKernel,
    x: MatRef<'_, f32>,
    xs: MatRef<'_, f32>,
    mut out: MatMut<'_, f32>,
) -> Result<(), GprError> {
    require_feature(x, leaf.num_params())?;
    require_cross_points(x, xs, out.as_ref())?;
    let w = weights(leaf.lengthscales());
    let nu = leaf.nu();
    for col in 0..xs.nrows() {
        for row in 0..x.nrows() {
            let (r2, _) = ard_terms(x, row, xs, col, &w)?;
            out[(row, col)] = finite(matern_from_r(nu, r2.max(0.0).sqrt()))?;
        }
    }
    Ok(())
}

fn matern_ard_grad(
    leaf: &crate::kernel::MaternArdKernel,
    x: MatRef<'_, f32>,
    mut out: MatMut<'_, f32>,
    idx: usize,
    uplo: Triangle,
) -> Result<(), GprError> {
    if idx >= leaf.num_params() {
        return Err(GprError::InvalidHyperparameter {
            reason: format!(
                "ARD Matern parameter index {idx} is out of range (d={})",
                leaf.num_params()
            ),
        });
    }
    require_feature(x, leaf.num_params())?;
    let n = require_points_square(x, out.as_ref())?;
    let w = weights(leaf.lengthscales());
    let nu = leaf.nu();
    let mut err = None;
    visit_triangle(n, uplo, |row, col| {
        if err.is_some() {
            return;
        }
        let value = ard_terms(x, row, x, col, &w)
            .and_then(|(r2, terms)| finite(matern_dk_ard(nu, r2.max(0.0).sqrt(), terms[idx])));
        match value {
            Ok(v) => out[(row, col)] = v,
            Err(e) => err = Some(e),
        }
    });
    match err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

fn matern_ard_hess(
    leaf: &crate::kernel::MaternArdKernel,
    x: MatRef<'_, f32>,
    mut out: MatMut<'_, f32>,
    i: usize,
    j: usize,
    uplo: Triangle,
) -> Result<(), GprError> {
    let d = leaf.num_params();
    pair_index(i, j, d, "ARD Matern")?;
    require_feature(x, d)?;
    let n = require_points_square(x, out.as_ref())?;
    let w = weights(leaf.lengthscales());
    let nu = leaf.nu();
    let same = i == j;
    let mut err = None;
    visit_triangle(n, uplo, |row, col| {
        if err.is_some() {
            return;
        }
        let value = ard_terms(x, row, x, col, &w).and_then(|(r2, terms)| {
            finite(matern_d2_ard(
                nu,
                r2.max(0.0).sqrt(),
                terms[i],
                terms[j],
                same,
            ))
        });
        match value {
            Ok(v) => out[(row, col)] = v,
            Err(e) => err = Some(e),
        }
    });
    match err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

fn rq_ard_parts(
    leaf: &crate::kernel::RationalQuadraticArdKernel,
) -> Result<(Vec<f32>, f32), GprError> {
    Ok((weights(leaf.lengthscales()), f32_of(leaf.alpha())))
}

fn rq_ard_apply(
    leaf: &crate::kernel::RationalQuadraticArdKernel,
    x: MatRef<'_, f32>,
    mut out: MatMut<'_, f32>,
    uplo: Triangle,
) -> Result<(), GprError> {
    let d = leaf.lengthscales().num_params();
    require_feature(x, d)?;
    let n = require_points_square(x, out.as_ref())?;
    let (w, alpha) = rq_ard_parts(leaf)?;
    let mut err = None;
    visit_triangle(n, uplo, |row, col| {
        if err.is_some() {
            return;
        }
        match ard_terms(x, row, x, col, &w).and_then(|(r2, _)| rq_from(r2, alpha)) {
            Ok(k) => out[(row, col)] = k,
            Err(e) => err = Some(e),
        }
    });
    match err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

fn rq_ard_cross(
    leaf: &crate::kernel::RationalQuadraticArdKernel,
    x: MatRef<'_, f32>,
    xs: MatRef<'_, f32>,
    mut out: MatMut<'_, f32>,
) -> Result<(), GprError> {
    let d = leaf.lengthscales().num_params();
    require_feature(x, d)?;
    require_cross_points(x, xs, out.as_ref())?;
    let (w, alpha) = rq_ard_parts(leaf)?;
    for col in 0..xs.nrows() {
        for row in 0..x.nrows() {
            let (r2, _) = ard_terms(x, row, xs, col, &w)?;
            out[(row, col)] = rq_from(r2, alpha)?;
        }
    }
    Ok(())
}

fn rq_ard_grad(
    leaf: &crate::kernel::RationalQuadraticArdKernel,
    x: MatRef<'_, f32>,
    mut out: MatMut<'_, f32>,
    idx: usize,
    uplo: Triangle,
) -> Result<(), GprError> {
    let d = leaf.lengthscales().num_params();
    if idx > d {
        return Err(GprError::InvalidHyperparameter {
            reason: format!("rational quadratic ARD parameter {idx} is out of range"),
        });
    }
    require_feature(x, d)?;
    let n = require_points_square(x, out.as_ref())?;
    let (w, alpha) = rq_ard_parts(leaf)?;
    let mut err = None;
    visit_triangle(n, uplo, |row, col| {
        if err.is_some() {
            return;
        }
        let value = ard_terms(x, row, x, col, &w).and_then(|(r2, terms)| {
            let u = 1.0 + r2 / (2.0 * alpha);
            let k = rq_from(r2, alpha)?;
            if idx == d {
                finite(alpha * k * (-u.ln() + 1.0 - 1.0 / u))
            } else {
                finite((k / u) * terms[idx])
            }
        });
        match value {
            Ok(v) => out[(row, col)] = v,
            Err(e) => err = Some(e),
        }
    });
    match err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

fn rq_ard_hess(
    leaf: &crate::kernel::RationalQuadraticArdKernel,
    x: MatRef<'_, f32>,
    mut out: MatMut<'_, f32>,
    i: usize,
    j: usize,
    uplo: Triangle,
) -> Result<(), GprError> {
    let d = leaf.lengthscales().num_params();
    pair_index(i, j, d + 1, "rational quadratic ARD")?;
    require_feature(x, d)?;
    let n = require_points_square(x, out.as_ref())?;
    let (w, alpha) = rq_ard_parts(leaf)?;
    let mut err = None;
    visit_triangle(n, uplo, |row, col| {
        if err.is_some() {
            return;
        }
        let value = ard_terms(x, row, x, col, &w).and_then(|(r2, terms)| {
            let u = 1.0 + r2 / (2.0 * alpha);
            let k = rq_from(r2, alpha)?;
            let (a, b) = if i <= j { (i, j) } else { (j, i) };
            let h = if a == d && b == d {
                let v = -u.ln() + 1.0 - 1.0 / u;
                let dv = (1.0 - u) * (1.0 - u) / (u * u);
                let h = alpha * k * v;
                h + (h * h) / k + alpha * k * dv
            } else if b == d {
                (k / u) * terms[a] * (-alpha * u.ln() + (alpha + 1.0) * (u - 1.0) / u)
            } else if a == b {
                -2.0 * (k / u) * terms[a] + (1.0 + 1.0 / alpha) * terms[a] * terms[a] * k / (u * u)
            } else {
                (1.0 + 1.0 / alpha) * terms[a] * terms[b] * k / (u * u)
            };
            finite(h)
        });
        match value {
            Ok(v) => out[(row, col)] = v,
            Err(e) => err = Some(e),
        }
    });
    match err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

fn rbf_coord(
    ell: f64,
    x1: MatRef<'_, f32>,
    x2: MatRef<'_, f32>,
    mut d_k: MatMut<'_, f32>,
    dim: usize,
) -> Result<(), GprError> {
    require_coord(x1, x2, d_k.as_ref(), dim)?;
    let ell = f32_of(ell);
    let inv_ell_sq = 1.0 / (ell * ell);
    let inv_two = 0.5 * inv_ell_sq;
    for col in 0..x2.nrows() {
        for row in 0..x1.nrows() {
            let s = sq_pair(x1, row, x2, col)?;
            let k = finite((-s * inv_two).exp())?;
            let delta = x1[(row, dim)] - x2[(col, dim)];
            d_k[(row, col)] = finite(k * delta * inv_ell_sq)?;
        }
    }
    Ok(())
}

fn matern_coord(
    nu: MaternNu,
    ell: f64,
    x1: MatRef<'_, f32>,
    x2: MatRef<'_, f32>,
    mut d_k: MatMut<'_, f32>,
    dim: usize,
) -> Result<(), GprError> {
    if nu != MaternNu::ThreeHalves {
        return Err(GprError::CoordGradientUnsupported);
    }
    require_coord(x1, x2, d_k.as_ref(), dim)?;
    let ell = f32_of(ell);
    let inv_ell_sq = 1.0 / (ell * ell);
    let scale = 3.0f32.sqrt() / ell;
    for col in 0..x2.nrows() {
        for row in 0..x1.nrows() {
            let s = sq_pair(x1, row, x2, col)?;
            let r = finite(s.max(0.0).sqrt())?;
            let delta = x1[(row, dim)] - x2[(col, dim)];
            let psi = (-scale * r).exp();
            d_k[(row, col)] = finite(3.0 * inv_ell_sq * psi * delta)?;
        }
    }
    Ok(())
}

fn rbf_ard_coord(
    leaf: &crate::kernel::RbfArdKernel,
    x1: MatRef<'_, f32>,
    x2: MatRef<'_, f32>,
    mut d_k: MatMut<'_, f32>,
    dim: usize,
) -> Result<(), GprError> {
    require_coord(x1, x2, d_k.as_ref(), dim)?;
    if dim >= leaf.num_params() {
        return Err(GprError::InvalidHyperparameter {
            reason: format!(
                "coordinate dimension {dim} is out of range for d={}",
                leaf.num_params()
            ),
        });
    }
    let w = weights(leaf.lengthscales());
    for col in 0..x2.nrows() {
        for row in 0..x1.nrows() {
            let (r2, _) = ard_terms(x1, row, x2, col, &w)?;
            let k = finite((-0.5 * r2).exp())?;
            let delta = x1[(row, dim)] - x2[(col, dim)];
            d_k[(row, col)] = finite(k * delta * w[dim])?;
        }
    }
    Ok(())
}

fn custom_from_coords(
    leaf: &crate::kernel::CustomKernel<f32>,
    x: MatRef<'_, f32>,
    out: MatMut<'_, f32>,
    uplo: Triangle,
) -> Result<(), GprError> {
    let n = require_points_square(x, out.as_ref())?;
    let mut dist = Mat::zeros(n, n);
    let mut err = None;
    visit_triangle(n, uplo, |row, col| {
        if err.is_some() {
            return;
        }
        match sq_pair(x, row, x, col) {
            Ok(s) => dist[(row, col)] = s,
            Err(e) => err = Some(e),
        }
    });
    if let Some(e) = err {
        return Err(e);
    }
    leaf.apply_f32(dist.as_ref(), out, uplo)
}

fn custom_grad_from_coords(
    leaf: &crate::kernel::CustomKernel<f32>,
    x: MatRef<'_, f32>,
    out: MatMut<'_, f32>,
    param_idx: usize,
    uplo: Triangle,
) -> Result<(), GprError> {
    let n = require_points_square(x, out.as_ref())?;
    let mut dist = Mat::zeros(n, n);
    let mut err = None;
    visit_triangle(n, uplo, |row, col| {
        if err.is_some() {
            return;
        }
        match sq_pair(x, row, x, col) {
            Ok(s) => dist[(row, col)] = s,
            Err(e) => err = Some(e),
        }
    });
    if let Some(e) = err {
        return Err(e);
    }
    leaf.grad_f32(dist.as_ref(), out, param_idx, uplo)
}

fn fold_apply(
    terms: &[CompiledKernel<f32>],
    dist: MatRef<'_, f32>,
    mut out: MatMut<'_, f32>,
    uplo: Triangle,
    mut scratch: MatMut<'_, f32>,
    combine: fn(MatMut<'_, f32>, MatRef<'_, f32>, Triangle),
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
    term: &CompiledKernel<f32>,
    dist: MatRef<'_, f32>,
    dest: MatMut<'_, f32>,
    fallback: MatMut<'_, f32>,
    uplo: Triangle,
    extra: &mut Option<Mat<f32>>,
    n: usize,
) -> Result<(), GprError> {
    if term.needs_internal_scratch() {
        let buf = extra.get_or_insert_with(|| Mat::zeros(n, n));
        term.apply(dist, dest, uplo, buf.as_mut())
    } else {
        term.apply(dist, dest, uplo, fallback)
    }
}

fn fold_points(
    terms: &[CompiledKernel<f32>],
    x: MatRef<'_, f32>,
    mut out: MatMut<'_, f32>,
    uplo: Triangle,
    mut scratch: MatMut<'_, f32>,
    combine: fn(MatMut<'_, f32>, MatRef<'_, f32>, Triangle),
) -> Result<(), GprError> {
    let (first, rest) = split_terms(terms)?;
    first.apply_points(x, out.as_mut(), uplo, scratch.as_mut())?;
    let n = out.nrows();
    let mut extra = None;
    for term in rest {
        if term.needs_internal_scratch() {
            let buf = extra.get_or_insert_with(|| Mat::zeros(n, n));
            term.apply_points(x, scratch.as_mut(), uplo, buf.as_mut())?;
        } else {
            term.apply_points(x, scratch.as_mut(), uplo, out.as_mut())?;
        }
        combine(out.as_mut(), scratch.as_ref(), uplo);
    }
    Ok(())
}

fn fold_cross(
    terms: &[CompiledKernel<f32>],
    dist: MatRef<'_, f32>,
    mut out: MatMut<'_, f32>,
    mut scratch: MatMut<'_, f32>,
    combine: fn(MatMut<'_, f32>, MatRef<'_, f32>),
) -> Result<(), GprError> {
    let (first, rest) = split_terms(terms)?;
    first.apply_cross(dist, out.as_mut(), scratch.as_mut())?;
    let rows = out.nrows();
    let cols = out.ncols();
    let mut extra = None;
    for term in rest {
        if term.needs_internal_scratch() {
            let buf = extra.get_or_insert_with(|| Mat::zeros(rows, cols));
            term.apply_cross(dist, scratch.as_mut(), buf.as_mut())?;
        } else {
            term.apply_cross(dist, scratch.as_mut(), out.as_mut())?;
        }
        combine(out.as_mut(), scratch.as_ref());
    }
    Ok(())
}

fn fold_cross_points(
    terms: &[CompiledKernel<f32>],
    x: MatRef<'_, f32>,
    xs: MatRef<'_, f32>,
    mut out: MatMut<'_, f32>,
    mut scratch: MatMut<'_, f32>,
    combine: fn(MatMut<'_, f32>, MatRef<'_, f32>),
) -> Result<(), GprError> {
    let (first, rest) = split_terms(terms)?;
    first.apply_cross_points(x, xs, out.as_mut(), scratch.as_mut())?;
    let rows = out.nrows();
    let cols = out.ncols();
    let mut extra = None;
    for term in rest {
        if term.needs_internal_scratch() {
            let buf = extra.get_or_insert_with(|| Mat::zeros(rows, cols));
            term.apply_cross_points(x, xs, scratch.as_mut(), buf.as_mut())?;
        } else {
            term.apply_cross_points(x, xs, scratch.as_mut(), out.as_mut())?;
        }
        combine(out.as_mut(), scratch.as_ref());
    }
    Ok(())
}

fn fold_diag(
    terms: &[CompiledKernel<f32>],
    out: &mut [f32],
    combine: fn(f32, f32) -> f32,
) -> Result<(), GprError> {
    let (first, rest) = split_terms(terms)?;
    first.fill_diag(out)?;
    let mut tmp = vec![0.0; out.len()];
    for term in rest {
        term.fill_diag(&mut tmp)?;
        for (dst, src) in out.iter_mut().zip(&tmp) {
            *dst = combine(*dst, *src);
        }
    }
    Ok(())
}

fn fold_diag_points(
    terms: &[CompiledKernel<f32>],
    x: MatRef<'_, f32>,
    out: &mut [f32],
    combine: fn(f32, f32) -> f32,
) -> Result<(), GprError> {
    let (first, rest) = split_terms(terms)?;
    first.fill_diag_points(x, out)?;
    let mut tmp = vec![0.0; out.len()];
    for term in rest {
        term.fill_diag_points(x, &mut tmp)?;
        for (dst, src) in out.iter_mut().zip(&tmp) {
            *dst = combine(*dst, *src);
        }
    }
    Ok(())
}

fn term_for_param(
    terms: &[CompiledKernel<f32>],
    param_idx: usize,
) -> Result<(&CompiledKernel<f32>, usize), GprError> {
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

fn term_index(terms: &[CompiledKernel<f32>], param_idx: usize) -> Result<(usize, usize), GprError> {
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

enum Owners<'a> {
    Same {
        term: &'a CompiledKernel<f32>,
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

fn owners(terms: &[CompiledKernel<f32>], i: usize, j: usize) -> Result<Owners<'_>, GprError> {
    let (owner_i, local_i) = term_index(terms, i)?;
    let (owner_j, local_j) = term_index(terms, j)?;
    if owner_i == owner_j {
        Ok(Owners::Same {
            term: &terms[owner_i],
            local_i,
            local_j,
        })
    } else {
        Ok(Owners::Distinct {
            owner_i,
            local_i,
            owner_j,
            local_j,
        })
    }
}

fn product_grad(
    terms: &[CompiledKernel<f32>],
    dist: MatRef<'_, f32>,
    mut d_k: MatMut<'_, f32>,
    param_idx: usize,
    uplo: Triangle,
    mut scratch: MatMut<'_, f32>,
) -> Result<(), GprError> {
    let (owner_i, local) = term_index(terms, param_idx)?;
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
            mul_tri(d_k.as_mut(), scratch.as_ref(), uplo);
        }
    }
    if started {
        if terms[owner_i].needs_internal_scratch() {
            let buf = extra.get_or_insert_with(|| Mat::zeros(n, n));
            terms[owner_i].grad(dist, scratch.as_mut(), local, uplo, buf.as_mut())?;
        } else {
            terms[owner_i].grad(dist, scratch.as_mut(), local, uplo, d_k.as_mut())?;
        }
        mul_tri(d_k.as_mut(), scratch.as_ref(), uplo);
    } else {
        terms[owner_i].grad(dist, d_k.as_mut(), local, uplo, scratch.as_mut())?;
    }
    Ok(())
}

fn product_grad_points(
    terms: &[CompiledKernel<f32>],
    x: MatRef<'_, f32>,
    mut d_k: MatMut<'_, f32>,
    param_idx: usize,
    uplo: Triangle,
    mut scratch: MatMut<'_, f32>,
) -> Result<(), GprError> {
    let (owner_i, local) = term_index(terms, param_idx)?;
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
        } else if term.needs_internal_scratch() {
            let buf = extra.get_or_insert_with(|| Mat::zeros(n, n));
            term.apply_points(x, scratch.as_mut(), uplo, buf.as_mut())?;
            mul_tri(d_k.as_mut(), scratch.as_ref(), uplo);
        } else {
            term.apply_points(x, scratch.as_mut(), uplo, d_k.as_mut())?;
            mul_tri(d_k.as_mut(), scratch.as_ref(), uplo);
        }
    }
    if started {
        if terms[owner_i].needs_internal_scratch() {
            let buf = extra.get_or_insert_with(|| Mat::zeros(n, n));
            terms[owner_i].grad_points(x, scratch.as_mut(), local, uplo, buf.as_mut())?;
        } else {
            terms[owner_i].grad_points(x, scratch.as_mut(), local, uplo, d_k.as_mut())?;
        }
        mul_tri(d_k.as_mut(), scratch.as_ref(), uplo);
    } else {
        terms[owner_i].grad_points(x, d_k.as_mut(), local, uplo, scratch.as_mut())?;
    }
    Ok(())
}

fn product_hess(
    terms: &[CompiledKernel<f32>],
    dist: MatRef<'_, f32>,
    d2: MatMut<'_, f32>,
    i: usize,
    j: usize,
    uplo: Triangle,
    scratch: MatMut<'_, f32>,
) -> Result<(), GprError> {
    match owners(terms, i, j)? {
        Owners::Same {
            local_i, local_j, ..
        } => {
            let (owner, _) = term_index(terms, i)?;
            product_same(
                terms,
                owner,
                d2,
                scratch,
                uplo,
                |term, dest, scratch| term.apply(dist, dest, uplo, scratch),
                |term, dest, scratch| term.hess(dist, dest, local_i, local_j, uplo, scratch),
            )
        }
        Owners::Distinct {
            owner_i,
            local_i,
            owner_j,
            local_j,
        } => product_cross_leaf(
            terms,
            owner_i,
            owner_j,
            d2,
            scratch,
            uplo,
            |term, dest, scratch| term.apply(dist, dest, uplo, scratch),
            |term, dest, scratch| term.grad(dist, dest, local_i, uplo, scratch),
            |term, dest, scratch| term.grad(dist, dest, local_j, uplo, scratch),
        ),
    }
}

fn product_hess_points(
    terms: &[CompiledKernel<f32>],
    x: MatRef<'_, f32>,
    d2: MatMut<'_, f32>,
    i: usize,
    j: usize,
    uplo: Triangle,
    scratch: MatMut<'_, f32>,
) -> Result<(), GprError> {
    match owners(terms, i, j)? {
        Owners::Same {
            local_i, local_j, ..
        } => {
            let (owner, _) = term_index(terms, i)?;
            product_same(
                terms,
                owner,
                d2,
                scratch,
                uplo,
                |term, dest, scratch| term.apply_points(x, dest, uplo, scratch),
                |term, dest, scratch| term.hess_points(x, dest, local_i, local_j, uplo, scratch),
            )
        }
        Owners::Distinct {
            owner_i,
            local_i,
            owner_j,
            local_j,
        } => product_cross_leaf(
            terms,
            owner_i,
            owner_j,
            d2,
            scratch,
            uplo,
            |term, dest, scratch| term.apply_points(x, dest, uplo, scratch),
            |term, dest, scratch| term.grad_points(x, dest, local_i, uplo, scratch),
            |term, dest, scratch| term.grad_points(x, dest, local_j, uplo, scratch),
        ),
    }
}

fn product_same(
    terms: &[CompiledKernel<f32>],
    owner: usize,
    mut d2: MatMut<'_, f32>,
    mut scratch: MatMut<'_, f32>,
    uplo: Triangle,
    mut apply: impl FnMut(
        &CompiledKernel<f32>,
        MatMut<'_, f32>,
        MatMut<'_, f32>,
    ) -> Result<(), GprError>,
    mut hess: impl FnMut(&CompiledKernel<f32>, MatMut<'_, f32>, MatMut<'_, f32>) -> Result<(), GprError>,
) -> Result<(), GprError> {
    let n = d2.nrows();
    let mut extra = None;
    let mut started = false;
    for (k, term) in terms.iter().enumerate() {
        if k == owner {
            continue;
        }
        if !started {
            apply(term, d2.as_mut(), scratch.as_mut())?;
            started = true;
        } else if term.needs_internal_scratch() {
            let buf = extra.get_or_insert_with(|| Mat::zeros(n, n));
            apply(term, scratch.as_mut(), buf.as_mut())?;
            mul_tri(d2.as_mut(), scratch.as_ref(), uplo);
        } else {
            apply(term, scratch.as_mut(), d2.as_mut())?;
            mul_tri(d2.as_mut(), scratch.as_ref(), uplo);
        }
    }
    if started {
        if terms[owner].needs_internal_scratch() {
            let buf = extra.get_or_insert_with(|| Mat::zeros(n, n));
            hess(&terms[owner], scratch.as_mut(), buf.as_mut())?;
        } else {
            hess(&terms[owner], scratch.as_mut(), d2.as_mut())?;
        }
        mul_tri(d2.as_mut(), scratch.as_ref(), uplo);
    } else {
        hess(&terms[owner], d2.as_mut(), scratch.as_mut())?;
    }
    Ok(())
}

// Owners, dest/scratch, apply, and both leaf grads do not fold without a new type.
#[allow(clippy::too_many_arguments)]
fn product_cross_leaf(
    terms: &[CompiledKernel<f32>],
    owner_i: usize,
    owner_j: usize,
    mut d2: MatMut<'_, f32>,
    mut scratch: MatMut<'_, f32>,
    uplo: Triangle,
    mut apply: impl FnMut(
        &CompiledKernel<f32>,
        MatMut<'_, f32>,
        MatMut<'_, f32>,
    ) -> Result<(), GprError>,
    mut grad_i: impl FnMut(
        &CompiledKernel<f32>,
        MatMut<'_, f32>,
        MatMut<'_, f32>,
    ) -> Result<(), GprError>,
    mut grad_j: impl FnMut(
        &CompiledKernel<f32>,
        MatMut<'_, f32>,
        MatMut<'_, f32>,
    ) -> Result<(), GprError>,
) -> Result<(), GprError> {
    let n = d2.nrows();
    let mut extra = None;
    let mut started = false;
    for (k, term) in terms.iter().enumerate() {
        if k == owner_i || k == owner_j {
            continue;
        }
        if !started {
            apply(term, d2.as_mut(), scratch.as_mut())?;
            started = true;
        } else if term.needs_internal_scratch() {
            let buf = extra.get_or_insert_with(|| Mat::zeros(n, n));
            apply(term, scratch.as_mut(), buf.as_mut())?;
            mul_tri(d2.as_mut(), scratch.as_ref(), uplo);
        } else {
            apply(term, scratch.as_mut(), d2.as_mut())?;
            mul_tri(d2.as_mut(), scratch.as_ref(), uplo);
        }
    }
    if started {
        write_owned(
            &terms[owner_i],
            scratch.as_mut(),
            d2.as_mut(),
            &mut extra,
            n,
            &mut grad_i,
        )?;
        mul_tri(d2.as_mut(), scratch.as_ref(), uplo);
        write_owned(
            &terms[owner_j],
            scratch.as_mut(),
            d2.as_mut(),
            &mut extra,
            n,
            &mut grad_j,
        )?;
        mul_tri(d2.as_mut(), scratch.as_ref(), uplo);
    } else {
        write_owned(
            &terms[owner_i],
            d2.as_mut(),
            scratch.as_mut(),
            &mut extra,
            n,
            &mut grad_i,
        )?;
        write_owned(
            &terms[owner_j],
            scratch.as_mut(),
            d2.as_mut(),
            &mut extra,
            n,
            &mut grad_j,
        )?;
        mul_tri(d2.as_mut(), scratch.as_ref(), uplo);
    }
    Ok(())
}

fn write_owned(
    term: &CompiledKernel<f32>,
    dest: MatMut<'_, f32>,
    fallback: MatMut<'_, f32>,
    extra: &mut Option<Mat<f32>>,
    n: usize,
    grad: &mut impl FnMut(
        &CompiledKernel<f32>,
        MatMut<'_, f32>,
        MatMut<'_, f32>,
    ) -> Result<(), GprError>,
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
    use super::visit_triangle;
    use crate::error::GprError;
    use crate::kernel::{
        ConstantKernel, KernelScalar, KernelSpec, KernelTerm, LinearKernel, MaternArdKernel,
        MaternKernel, MaternNu, PeriodicKernel, RationalQuadraticArdKernel,
        RationalQuadraticKernel, RbfArdKernel, RbfKernel, Triangle, WhiteKernel,
    };
    use crate::param::Interval;
    use faer::{Mat, MatMut, MatRef};

    fn close(what: &str, got: f32, expect: f64) {
        let tol = 10.0 * f64::from(f32::EPSILON);
        let got = f64::from(got);
        let diff = (got - expect).abs();
        if expect.abs() < tol {
            assert!(diff < tol, "{what} absolute {diff} tol {tol}");
        } else {
            let rel = diff / expect.abs();
            assert!(rel < tol, "{what} relative {rel} got {got} expect {expect}");
        }
    }

    fn cmp_lower(what: &str, got: MatRef<'_, f32>, expect: MatRef<'_, f64>) {
        let n = expect.nrows();
        visit_triangle(n, Triangle::Lower, |row, col| {
            close(
                &format!("{what} ({row},{col})"),
                got[(row, col)],
                expect[(row, col)],
            );
        });
    }

    fn points() -> Mat<f64> {
        Mat::from_fn(2, 2, |row, col| [[0.0, 0.0], [1.0, 0.0]][row][col])
    }

    fn sq_pair(x: MatRef<'_, f64>, i: usize, j: usize) -> (f64, f32) {
        let mut s64 = 0.0;
        let mut s32 = 0.0f32;
        for col in 0..x.ncols() {
            let a = x[(i, col)];
            let b = x[(j, col)];
            s64 += (a - b) * (a - b);
            let a32 = a as f32;
            let b32 = b as f32;
            s32 += (a32 - b32) * (a32 - b32);
        }
        (s64, s32)
    }

    fn agree(name: &str, spec: &KernelSpec, x: &Mat<f64>, on_dist: bool) {
        let n = x.nrows();
        let k64 = spec.compile();
        let k32 = spec.compile_as::<f32>();
        let mut x32 = Mat::<f32>::zeros(n, x.ncols());
        for col in 0..x.ncols() {
            for row in 0..n {
                x32[(row, col)] = x[(row, col)] as f32;
            }
        }
        let mut o64 = Mat::<f64>::zeros(n, n);
        let mut o32 = Mat::<f32>::zeros(n, n);
        let mut s64 = Mat::<f64>::zeros(n, n);
        let mut s32 = Mat::<f32>::zeros(n, n);
        k64.apply_points(x.as_ref(), o64.as_mut(), Triangle::Lower, s64.as_mut())
            .expect("f64 points");
        k32.apply_points(x32.as_ref(), o32.as_mut(), Triangle::Lower, s32.as_mut())
            .expect("f32 points");
        cmp_lower(&format!("{name} points"), o32.as_ref(), o64.as_ref());
        let p = k64.num_params();
        for idx in 0..p {
            k64.grad_points(x.as_ref(), o64.as_mut(), idx, Triangle::Lower, s64.as_mut())
                .expect("f64 grad");
            k32.grad_points(
                x32.as_ref(),
                o32.as_mut(),
                idx,
                Triangle::Lower,
                s32.as_mut(),
            )
            .expect("f32 grad");
            cmp_lower(&format!("{name} grad {idx}"), o32.as_ref(), o64.as_ref());
            for j in 0..=idx {
                k64.hess_points(
                    x.as_ref(),
                    o64.as_mut(),
                    idx,
                    j,
                    Triangle::Lower,
                    s64.as_mut(),
                )
                .expect("f64 hess");
                k32.hess_points(
                    x32.as_ref(),
                    o32.as_mut(),
                    idx,
                    j,
                    Triangle::Lower,
                    s32.as_mut(),
                )
                .expect("f32 hess");
                cmp_lower(
                    &format!("{name} hess {idx},{j}"),
                    o32.as_ref(),
                    o64.as_ref(),
                );
            }
        }
        if !on_dist {
            return;
        }
        let mut d64 = Mat::<f64>::zeros(n, n);
        let mut d32 = Mat::<f32>::zeros(n, n);
        for col in 0..n {
            for row in col..n {
                let (s64v, s32v) = sq_pair(x.as_ref(), row, col);
                d64[(row, col)] = s64v;
                d32[(row, col)] = s32v;
            }
        }
        k64.apply(d64.as_ref(), o64.as_mut(), Triangle::Lower, s64.as_mut())
            .expect("f64 dist");
        k32.apply(d32.as_ref(), o32.as_mut(), Triangle::Lower, s32.as_mut())
            .expect("f32 dist");
        cmp_lower(&format!("{name} dist"), o32.as_ref(), o64.as_ref());
    }

    #[derive(Clone, Debug)]
    struct ScaleLeaf(f64);

    impl<T: KernelScalar> KernelTerm<T> for ScaleLeaf {
        fn clone_box(&self) -> Box<dyn KernelTerm<T>> {
            Box::new(self.clone())
        }
        fn num_params(&self) -> usize {
            1
        }
        fn get_params(&self, out: &mut [f64]) -> Result<(), GprError> {
            out[0] = self.0.ln();
            Ok(())
        }
        fn set_params(&mut self, params: &[f64]) -> Result<(), GprError> {
            self.0 = params[0].exp();
            Ok(())
        }
        fn bounds_into(&self, out: &mut [Interval]) -> Result<(), GprError> {
            out[0] = Interval::new(1e-6, 1e6).expect("interval");
            Ok(())
        }
        fn apply(
            &self,
            dist: MatRef<'_, T>,
            out: MatMut<'_, T>,
            uplo: Triangle,
        ) -> Result<(), GprError> {
            let _ = dist;
            paint(out, uplo, T::from_f64(self.0));
            Ok(())
        }
        fn apply_cross(&self, dist: MatRef<'_, T>, mut out: MatMut<'_, T>) -> Result<(), GprError> {
            let _ = dist;
            let value = T::from_f64(self.0);
            for col in 0..out.ncols() {
                for row in 0..out.nrows() {
                    out[(row, col)] = value;
                }
            }
            Ok(())
        }
        fn fill_diag(&self, out: &mut [T]) -> Result<(), GprError> {
            out.fill(T::from_f64(self.0));
            Ok(())
        }
        fn grad(
            &self,
            dist: MatRef<'_, T>,
            out: MatMut<'_, T>,
            param_idx: usize,
            uplo: Triangle,
        ) -> Result<(), GprError> {
            if param_idx != 0 {
                return Err(GprError::InvalidHyperparameter {
                    reason: "scale leaf has one parameter".to_owned(),
                });
            }
            self.apply(dist, out, uplo)
        }
        fn hess(
            &self,
            dist: MatRef<'_, T>,
            out: MatMut<'_, T>,
            i: usize,
            j: usize,
            uplo: Triangle,
        ) -> Result<(), GprError> {
            if i != 0 || j != 0 {
                return Err(GprError::InvalidHyperparameter {
                    reason: "scale leaf has one parameter".to_owned(),
                });
            }
            self.apply(dist, out, uplo)
        }
        fn hess_points(
            &self,
            x: MatRef<'_, T>,
            out: MatMut<'_, T>,
            i: usize,
            j: usize,
            uplo: Triangle,
        ) -> Result<(), GprError> {
            let _ = x;
            self.hess(x, out, i, j, uplo)
        }
    }

    fn paint<T: KernelScalar>(mut out: MatMut<'_, T>, uplo: Triangle, value: T) {
        visit_triangle(out.nrows(), uplo, |row, col| out[(row, col)] = value);
    }

    #[test]
    fn f32_leaves_match_f64() {
        let x = points();
        let rbf = KernelSpec::from(RbfKernel::new(1.0).expect("rbf"));
        agree("rbf", &rbf, &x, true);
        agree(
            "rbf ard",
            &KernelSpec::from(RbfArdKernel::new(&[1.0, 2.0]).expect("ard")),
            &x,
            false,
        );
        for nu in [MaternNu::Half, MaternNu::ThreeHalves, MaternNu::FiveHalves] {
            agree(
                "matern",
                &KernelSpec::from(MaternKernel::new(1.0, nu).expect("matern")),
                &x,
                true,
            );
        }
        agree(
            "matern ard",
            &KernelSpec::from(
                MaternArdKernel::new(&[1.0, 2.0], MaternNu::ThreeHalves).expect("matern ard"),
            ),
            &x,
            false,
        );
        agree(
            "periodic",
            &KernelSpec::from(PeriodicKernel::new(1.0, 4.0).expect("periodic")),
            &x,
            true,
        );
        agree(
            "rq",
            &KernelSpec::from(RationalQuadraticKernel::new(1.0, 1.0).expect("rq")),
            &x,
            true,
        );
        agree(
            "rq ard",
            &KernelSpec::from(RationalQuadraticArdKernel::new(&[1.0, 1.0], 1.0).expect("rq ard")),
            &x,
            false,
        );
        agree(
            "constant",
            &KernelSpec::from(ConstantKernel::new(1.5).expect("constant")),
            &x,
            true,
        );
        agree(
            "linear",
            &KernelSpec::from(LinearKernel::new(1.0).expect("linear")),
            &x,
            false,
        );
        agree(
            "white",
            &KernelSpec::from(WhiteKernel::new(0.0625).expect("white")),
            &x,
            true,
        );
        let other = KernelSpec::from(RbfKernel::new(2.0).expect("rbf"));
        agree("sum", &(rbf.clone() + other.clone()), &x, true);
        agree("product", &(rbf * other), &x, true);
        agree("custom", &KernelSpec::custom(ScaleLeaf(1.25)), &x, true);
    }
}
