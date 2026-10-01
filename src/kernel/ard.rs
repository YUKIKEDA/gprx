//! Shared checks and pair loops for the ARD leaves.
//!
//! RBF-ARD, Matérn-ARD, and RQ-ARD differ only in the scalar formula of
//! `r² = Σ_d w_d Δ_d²` (`w_d = 1/ℓ_d²`). The shape checks, the `r²` sums from
//! coordinates or from the `(Δx_d)²` cache, and the matrix loops live here.

use super::dist::ArdSqDiff;
use super::{KernelScalar, Triangle, write_square};
use crate::error::GprError;
use faer::reborrow::ReborrowMut;
use faer::{MatMut, MatRef};

/// `r²` and the terms `w_i Δ_i²`, `w_j Δ_j²` of up to two picked dimensions.
///
/// A term whose dimension was not picked is zero.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ArdR2<T> {
    pub(crate) r2: T,
    pub(crate) dim_i: T,
    pub(crate) dim_j: T,
}

/// Dimensions whose terms [`ArdR2`] keeps.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Pick {
    i: Option<usize>,
    j: Option<usize>,
}

impl Pick {
    pub(crate) const NONE: Self = Self { i: None, j: None };

    pub(crate) fn one(i: usize) -> Self {
        Self {
            i: Some(i),
            j: None,
        }
    }

    pub(crate) fn pair(i: usize, j: usize) -> Self {
        Self {
            i: Some(i),
            j: Some(j),
        }
    }
}

/// Sums `r²` from `sq(dim)`, the squared difference in `dim`.
///
/// # Errors
///
/// Propagates the error of `sq`, or returns
/// [`GprError::NonFiniteKernelValue`] if `r²` is not finite.
#[inline(always)]
fn sum_r2<T: KernelScalar>(
    inv_ell_sq: &[f64],
    pick: Pick,
    mut sq: impl FnMut(usize) -> Result<T, GprError>,
) -> Result<ArdR2<T>, GprError> {
    let zero = T::from_f64(0.0);
    let mut out = ArdR2 {
        r2: zero,
        dim_i: zero,
        dim_j: zero,
    };
    for (dim, &w) in inv_ell_sq.iter().enumerate() {
        let term = sq(dim)? * T::from_f64(w);
        out.r2 += term;
        if pick.i == Some(dim) {
            out.dim_i = term;
        }
        if pick.j == Some(dim) {
            out.dim_j = term;
        }
    }
    if out.r2.is_finite() {
        Ok(out)
    } else {
        Err(GprError::NonFiniteKernelValue)
    }
}

/// [`ArdR2`] of `x1[row, :]` and `x2[col, :]`.
///
/// # Errors
///
/// Returns [`GprError::NonFiniteInput`] if a difference is not finite, or
/// [`GprError::NonFiniteKernelValue`] if `r²` is not finite.
#[inline]
pub(crate) fn r2_from_coords<T: KernelScalar>(
    x1: MatRef<'_, T>,
    row: usize,
    x2: MatRef<'_, T>,
    col: usize,
    inv_ell_sq: &[f64],
    pick: Pick,
) -> Result<ArdR2<T>, GprError> {
    sum_r2(inv_ell_sq, pick, |dim| {
        let diff = x1[(row, dim)] - x2[(col, dim)];
        if diff.is_finite() {
            Ok(diff * diff)
        } else {
            Err(GprError::NonFiniteInput)
        }
    })
}

/// [`ArdR2`] of the pair `(row, col)`, in either order, from the `(Δx_d)²` cache.
///
/// # Errors
///
/// Returns [`GprError::NonFiniteInput`] if a cached value is not finite, or
/// [`GprError::NonFiniteKernelValue`] if `r²` is not finite.
#[inline]
pub(crate) fn r2_from_cache<T: KernelScalar>(
    cache: ArdSqDiff<'_, T>,
    row: usize,
    col: usize,
    inv_ell_sq: &[f64],
    pick: Pick,
) -> Result<ArdR2<T>, GprError> {
    sum_r2(inv_ell_sq, pick, |dim| {
        let v = cache.get(dim, row, col);
        if v.is_finite() {
            Ok(v)
        } else {
            Err(GprError::NonFiniteInput)
        }
    })
}

/// Rejects an empty `x` or one whose column count is not `expected_d`.
pub(crate) fn require_feature_dim<T>(x: MatRef<'_, T>, expected_d: usize) -> Result<(), GprError> {
    if x.nrows() == 0 || x.ncols() == 0 {
        return Err(GprError::EmptyInput);
    }
    if x.ncols() != expected_d {
        return Err(GprError::DimensionMismatch {
            x_dim: x.ncols(),
            expected_dim: expected_d,
        });
    }
    Ok(())
}

/// Checks `x` (`n × d`, finite) against an `n × n` output and returns `n`.
pub(crate) fn require_square_points<T: KernelScalar>(
    x: MatRef<'_, T>,
    out: MatRef<'_, T>,
    expected_d: usize,
) -> Result<usize, GprError> {
    require_feature_dim(x, expected_d)?;
    if out.nrows() != x.nrows() || out.ncols() != x.nrows() {
        return Err(GprError::ShapeMismatch {
            reason: format!(
                "output is {}x{}, expected {}x{}",
                out.nrows(),
                out.ncols(),
                x.nrows(),
                x.nrows()
            ),
        });
    }
    crate::data::require_finite_points(x)?;
    Ok(x.nrows())
}

/// Checks a non-empty square output and returns `n`.
pub(crate) fn require_square_out<T>(out: MatRef<'_, T>) -> Result<usize, GprError> {
    if out.nrows() == 0 || out.ncols() == 0 {
        return Err(GprError::EmptyInput);
    }
    if out.nrows() != out.ncols() {
        return Err(GprError::ShapeMismatch {
            reason: format!("output is {}x{}, expected square", out.nrows(), out.ncols()),
        });
    }
    Ok(out.nrows())
}

/// Rejects `param_idx >= count` for the kernel `name`.
pub(crate) fn require_param(name: &str, param_idx: usize, count: usize) -> Result<(), GprError> {
    if param_idx < count {
        Ok(())
    } else {
        Err(GprError::IndexOutOfRange {
            reason: format!(
                "ARD {name} parameter index {param_idx} is out of range ({count} parameters)"
            ),
        })
    }
}

/// Rejects a pair with an index `>= count` for the kernel `name`.
pub(crate) fn require_param_pair(
    name: &str,
    i: usize,
    j: usize,
    count: usize,
) -> Result<(), GprError> {
    if i < count && j < count {
        Ok(())
    } else {
        Err(GprError::IndexOutOfRange {
            reason: format!(
                "ARD {name} parameter pair ({i}, {j}) is out of range ({count} parameters)"
            ),
        })
    }
}

/// Writes `uplo` of `out` from the coordinates `x`. `pair(row, col)` is the value.
pub(crate) fn write_from_points<T: KernelScalar>(
    x: MatRef<'_, T>,
    out: MatMut<'_, T>,
    d: usize,
    uplo: Triangle,
    pair: impl FnMut(usize, usize) -> Result<T, GprError>,
) -> Result<(), GprError> {
    require_square_points(x, out.as_ref(), d)?;
    write_square(out, uplo, pair)
}

/// Writes `uplo` of `out` from the `(Δx_d)²` cache. `pair(row, col)` is the value.
pub(crate) fn write_from_cache<T: KernelScalar>(
    cache: ArdSqDiff<'_, T>,
    out: MatMut<'_, T>,
    d: usize,
    uplo: Triangle,
    pair: impl FnMut(usize, usize) -> Result<T, GprError>,
) -> Result<(), GprError> {
    let n = require_square_out(out.as_ref())?;
    super::dist::require_ard_sq_diff_shape(cache, n, d)?;
    write_square(out, uplo, pair)
}

/// [`write_from_points`] through the vectorized loop of `profile` when the
/// scalar is `f64` and the layout allows it; `pair` otherwise.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_from_points_simd<T: KernelScalar, P: super::ard_simd::Profile>(
    x: MatRef<'_, T>,
    mut out: MatMut<'_, T>,
    d: usize,
    uplo: Triangle,
    inv_ell_sq: &[f64],
    pick: Option<usize>,
    profile: &P,
    pair: impl FnMut(usize, usize) -> Result<T, GprError>,
) -> Result<(), GprError> {
    require_square_points(x, out.as_ref(), d)?;
    if let (Some(x), Some(o)) = (T::as_f64_ref(x), T::as_f64_mut(out.rb_mut()))
        && super::ard_simd::try_fill(
            super::ard_simd::Source::Points { x, y: x },
            o,
            inv_ell_sq,
            pick,
            super::ard_simd::Rows::Square(uplo),
            profile,
        )?
    {
        return Ok(());
    }
    write_square(out, uplo, pair)
}

/// [`write_from_cache`] through the vectorized loop of `profile` when the
/// scalar is `f64` and the layout allows it; `pair` otherwise.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_from_cache_simd<T: KernelScalar, P: super::ard_simd::Profile>(
    cache: ArdSqDiff<'_, T>,
    mut out: MatMut<'_, T>,
    d: usize,
    uplo: Triangle,
    inv_ell_sq: &[f64],
    pick: Option<usize>,
    profile: &P,
    mut pair: impl FnMut(usize, usize, usize) -> Result<T, GprError>,
) -> Result<(), GprError> {
    let n = require_square_out(out.as_ref())?;
    super::dist::require_ard_sq_diff_shape(cache, n, d)?;
    if let (Some(cache), Some(o)) = (cache.as_f64(), T::as_f64_mut(out.rb_mut()))
        && super::ard_simd::try_fill(
            super::ard_simd::Source::Cache { cache },
            o,
            inv_ell_sq,
            pick,
            super::ard_simd::Rows::Square(uplo),
            profile,
        )?
    {
        return Ok(());
    }
    write_square(out, uplo, |row, col| pair(n, row, col))
}

/// A rectangular `out` (train × test, checked by [`require_cross`]) through
/// the vectorized loop of `profile` when the scalar is `f64` and the layout
/// allows it; `pair` otherwise.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_cross_simd<T: KernelScalar, P: super::ard_simd::Profile>(
    x: MatRef<'_, T>,
    xs: MatRef<'_, T>,
    mut out: MatMut<'_, T>,
    d: usize,
    inv_ell_sq: &[f64],
    pick: Option<usize>,
    profile: &P,
    pair: impl FnMut(usize, usize) -> Result<T, GprError>,
) -> Result<(), GprError> {
    require_cross(x, xs, out.as_ref(), d)?;
    if let (Some(x), Some(xs), Some(o)) = (
        T::as_f64_ref(x),
        T::as_f64_ref(xs),
        T::as_f64_mut(out.rb_mut()),
    ) && super::ard_simd::try_fill(
        super::ard_simd::Source::Points { x, y: xs },
        o,
        inv_ell_sq,
        pick,
        super::ard_simd::Rows::All,
        profile,
    )? {
        return Ok(());
    }
    super::write_rect(out, pair)
}

/// Checks a train × test pair of `d`-column inputs against `out`.
pub(crate) fn require_cross<T: KernelScalar>(
    x: MatRef<'_, T>,
    xs: MatRef<'_, T>,
    out: MatRef<'_, T>,
    d: usize,
) -> Result<(), GprError> {
    require_feature_dim(x, d)?;
    require_feature_dim(xs, d)?;
    if out.nrows() != x.nrows() || out.ncols() != xs.nrows() {
        return Err(GprError::ShapeMismatch {
            reason: format!(
                "output is {}x{}, expected {}x{}",
                out.nrows(),
                out.ncols(),
                x.nrows(),
                xs.nrows()
            ),
        });
    }
    crate::data::require_finite_points(x)?;
    crate::data::require_finite_points(xs)
}
