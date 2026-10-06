//! Describes kernel leaves and composition ([`KernelSpec`] / [`CompiledKernel`]).
//!
//! Built-in leaves are enum arms. User distance leaves implement [`KernelTerm`]
//! and enter the tree as [`KernelSpec::Custom`].
//!
//! Isotropic RBF, Matérn, Periodic, and rational quadratic evaluate from a
//! squared-Euclidean distance matrix (Periodic then takes the square root). ARD RBF, ARD
//! Matérn, and ARD rational quadratic evaluate from coordinates via [`ArdLengthscales`]
//! (`θ_d = log(ℓ_d)`). Callers pass column-major matrix views; this module does not
//! re-export matrix-library types.
//!
//! Squared-Euclidean fills and [`Triangle::Lower`] writes run on the process-wide thread
//! pool. Isotropic RBF, ARD RBF, and the distance / `(Δx_d)²` row loops use four-wide
//! `f64` SIMD when the matrix view is column-major with unit row stride. There is no
//! parallel on/off flag; `RAYON_NUM_THREADS=1` is sequential. Limit threads with
//! `RAYON_NUM_THREADS` or the global thread-pool builder before the first fill. Linear and
//! other points-mode leaves stay sequential. See the [crate-level parallelism
//! notes](crate).

mod ard;
mod compiled;
mod constant;
mod dist;
mod lengthscale;
mod linear;
mod matern;
mod matern_ard;
mod periodic;
mod radial;
mod rbf;
mod rbf_ard;
mod rq;
mod rq_ard;
mod scalar;
mod simd;
mod sources;
mod spec;
mod supply;
mod term;
mod white;

pub use compiled::CompiledKernel;
pub(crate) use compiled::ensure_nested_levels;
pub(crate) use compiled::gram::GramInputs;
pub(crate) use compiled::supplied::{ColRange, RectSlots, RectTable, SquareSlots};
pub(crate) use compiled::CrossViews;
pub(crate) use sources::{
    BlockKind, Fills, GatheredRect, QuerySources, TrainSources, bind as bind_sources,
};
pub(crate) use compiled::weighted::{DiagAccum, WeightedWalk};
pub use constant::ConstantKernel;
pub(crate) use dist::ArdSqDiffBuf;
#[cfg(any(test, feature = "bench-internals"))]
pub(crate) use dist::fill_squared_euclidean;
pub use lengthscale::ArdLengthscales;
pub use linear::LinearKernel;
pub use matern::{MaternKernel, MaternNu};
pub use matern_ard::MaternArdKernel;
pub use periodic::PeriodicKernel;
pub use rbf::RbfKernel;
pub use rbf_ard::RbfArdKernel;
pub use rq::RationalQuadraticKernel;
pub use rq_ard::RationalQuadraticArdKernel;
pub use scalar::KernelScalar;
pub(crate) use scalar::sealed::ScalarOps;
pub use spec::{KernelSpec, ParameterBinding};
pub use supply::{
    ArdDistance, ArdDistanceLeaf, DistanceFill, DistanceKernel, DistanceOnly, DistanceSlot,
    DistanceSource, JoinPoints, ModelKernel, PointKernel, PointUse, ScalarDistance, ScalarDistanceLeaf,
    WithPoints,
};
pub(crate) use supply::{
    ModelKernelParts, SlotId, SlotShape, SourceData, SuppliedSpec, remap_slots, spec_slots,
};
pub use term::{CustomKernel, KernelTerm};
pub use white::WhiteKernel;

use crate::error::GprError;
use dist::par_lower_cols;
pub(crate) use dist::worker_count;
use faer::{MatMut, MatRef};

/// Records which triangle of a symmetric kernel matrix to write.
///
/// # Examples
///
/// ```rust
/// use gprx::kernel::Triangle;
///
/// assert_ne!(Triangle::Lower, Triangle::Upper);
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Triangle {
    /// Marks entries with `row >= col`.
    ///
    /// This is the default for Cholesky.
    Lower,
    /// Marks entries with `row <= col`.
    Upper,
    /// Marks every entry.
    ///
    /// Upper and lower must match when `dist` is symmetric.
    Full,
}

pub(crate) fn visit_triangle(n: usize, uplo: Triangle, mut visit: impl FnMut(usize, usize)) {
    match uplo {
        Triangle::Lower => {
            for col in 0..n {
                for row in col..n {
                    visit(row, col);
                }
            }
        }
        Triangle::Upper => {
            for col in 0..n {
                for row in 0..=col {
                    visit(row, col);
                }
            }
        }
        Triangle::Full => {
            for col in 0..n {
                for row in 0..n {
                    visit(row, col);
                }
            }
        }
    }
}

fn require_square_pair<T>(dist: MatRef<'_, T>, out: MatRef<'_, T>) -> Result<usize, GprError> {
    if dist.nrows() != dist.ncols() {
        return Err(GprError::ShapeMismatch {
            reason: format!(
                "distance matrix must be square, got {}x{}",
                dist.nrows(),
                dist.ncols()
            ),
        });
    }
    if out.nrows() != dist.nrows() || out.ncols() != dist.ncols() {
        return Err(GprError::ShapeMismatch {
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

pub(crate) fn require_coord_grad<T>(
    x1: MatRef<'_, T>,
    x2: MatRef<'_, T>,
    d_k: MatRef<'_, T>,
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
        return Err(GprError::IndexOutOfRange {
            reason: format!(
                "coordinate dimension {dim} is out of range for d={}",
                x1.ncols()
            ),
        });
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
    Ok(())
}

fn require_same_shape<T>(dist: MatRef<'_, T>, out: MatRef<'_, T>) -> Result<(), GprError> {
    if out.nrows() != dist.nrows() || out.ncols() != dist.ncols() {
        return Err(GprError::ShapeMismatch {
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

fn write_dense<T: KernelScalar>(
    dist: MatRef<'_, T>,
    mut out: MatMut<'_, T>,
    mut kernel: impl FnMut(T) -> Result<T, GprError>,
) -> Result<(), GprError> {
    require_same_shape(dist, out.as_ref())?;
    let mut err = None;
    for col in 0..dist.ncols() {
        for row in 0..dist.nrows() {
            if err.is_some() {
                continue;
            }
            match kernel(dist[(row, col)]) {
                Ok(value) => out[(row, col)] = value,
                Err(e) => err = Some(e),
            }
        }
    }
    match err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

fn write_triangle<T: KernelScalar>(
    dist: MatRef<'_, T>,
    mut out: MatMut<'_, T>,
    uplo: Triangle,
    kernel: impl Fn(T) -> Result<T, GprError> + Sync,
) -> Result<(), GprError> {
    let n = require_square_pair(dist, out.as_ref())?;
    if n > 0 && matches!(uplo, Triangle::Lower) {
        return write_lower_parallel(dist, out, kernel);
    }
    let mut err = None;
    visit_triangle(n, uplo, |row, col| {
        if err.is_some() {
            return;
        }
        match kernel(dist[(row, col)]) {
            Ok(value) => out[(row, col)] = value,
            Err(e) => err = Some(e),
        }
    });
    match err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

fn write_lower_parallel<T: KernelScalar>(
    dist: MatRef<'_, T>,
    out: MatMut<'_, T>,
    kernel: impl Fn(T) -> Result<T, GprError> + Sync,
) -> Result<(), GprError> {
    par_lower_cols(out, &|col, mut rows| {
        for i in 0..rows.nrows() {
            rows[i] = kernel(dist[(col + i, col)])?;
        }
        Ok(())
    })
}

/// Returns `value` if it is finite, else [`GprError::NonFiniteKernelValue`].
#[inline(always)]
pub(crate) fn finite_kernel<T: KernelScalar>(value: T) -> Result<T, GprError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(GprError::NonFiniteKernelValue)
    }
}

fn finite_dist<T: KernelScalar>(d: T) -> Result<T, GprError> {
    if d.is_finite() {
        Ok(d)
    } else {
        Err(GprError::NonFiniteInput)
    }
}

pub(crate) fn pair_squared_euclidean<T: KernelScalar>(x: MatRef<'_, T>, i: usize, j: usize) -> T {
    let d = x.ncols();
    let mut sum = T::from_f64(0.0);
    for dim in 0..d {
        let diff = x[(i, dim)] - x[(j, dim)];
        sum += diff * diff;
    }
    sum
}

/// `‖x1_row − x2_col‖²` of two point sets.
pub(crate) fn cross_squared_euclidean<T: KernelScalar>(
    x1: MatRef<'_, T>,
    row: usize,
    x2: MatRef<'_, T>,
    col: usize,
) -> T {
    let mut sum = T::from_f64(0.0);
    for dim in 0..x1.ncols() {
        let diff = x1[(row, dim)] - x2[(col, dim)];
        sum += diff * diff;
    }
    sum
}

/// Writes a rectangular kernel block (`x1.nrows() × x2.nrows()`) from
/// coordinates: `kernel(‖x1_i − x2_j‖²)` per entry, without a distance matrix.
pub(crate) fn write_rect_from_coords<T: KernelScalar>(
    x1: MatRef<'_, T>,
    x2: MatRef<'_, T>,
    out: MatMut<'_, T>,
    kernel: impl Fn(T) -> Result<T, GprError>,
) -> Result<(), GprError> {
    require_coord_grad(x1, x2, out.as_ref(), 0)?;
    write_rect(out, |row, col| {
        kernel(finite_dist(cross_squared_euclidean(x1, row, x2, col))?)
    })
}

/// Writes a kernel triangle from coordinates. Each pair computes `‖x_i-x_j‖²`
/// without a distance matrix.
pub(crate) fn write_square_from_coords<T: KernelScalar>(
    x: MatRef<'_, T>,
    out: MatMut<'_, T>,
    uplo: Triangle,
    kernel: impl Fn(T) -> Result<T, GprError> + Sync,
) -> Result<(), GprError> {
    let n = out.nrows();
    if out.ncols() != n {
        return Err(GprError::ShapeMismatch {
            reason: format!("output must be square, got {}x{}", out.nrows(), out.ncols()),
        });
    }
    if n == 0 || x.nrows() != n || x.ncols() == 0 {
        return Err(GprError::EmptyInput);
    }
    if matches!(uplo, Triangle::Lower) {
        return write_lower_from_coords_parallel(x, out, kernel);
    }
    write_square(out, uplo, |row, col| {
        kernel(finite_dist(pair_squared_euclidean(x, row, col))?)
    })
}

fn write_lower_from_coords_parallel<T: KernelScalar>(
    x: MatRef<'_, T>,
    out: MatMut<'_, T>,
    kernel: impl Fn(T) -> Result<T, GprError> + Sync,
) -> Result<(), GprError> {
    par_lower_cols(out, &|col, mut rows| {
        for i in 0..rows.nrows() {
            rows[i] = kernel(finite_dist(pair_squared_euclidean(x, col + i, col))?)?;
        }
        Ok(())
    })
}

/// Writes every entry of a rectangular `out`. `pair(row, col)` is the value.
pub(crate) fn write_rect<T: KernelScalar>(
    mut out: MatMut<'_, T>,
    mut pair: impl FnMut(usize, usize) -> Result<T, GprError>,
) -> Result<(), GprError> {
    for col in 0..out.ncols() {
        for row in 0..out.nrows() {
            out[(row, col)] = pair(row, col)?;
        }
    }
    Ok(())
}

fn write_square<T: KernelScalar>(
    mut out: MatMut<'_, T>,
    uplo: Triangle,
    mut kernel: impl FnMut(usize, usize) -> Result<T, GprError>,
) -> Result<(), GprError> {
    if out.nrows() != out.ncols() {
        return Err(GprError::ShapeMismatch {
            reason: format!("output must be square, got {}x{}", out.nrows(), out.ncols()),
        });
    }
    if out.nrows() == 0 {
        return Err(GprError::EmptyInput);
    }
    let n = out.nrows();
    let mut err = None;
    visit_triangle(n, uplo, |row, col| {
        if err.is_some() {
            return;
        }
        match kernel(row, col) {
            Ok(value) => out[(row, col)] = value,
            Err(e) => err = Some(e),
        }
    });
    match err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

fn validate_positive_finite(value: f64, what: &str) -> Result<(), GprError> {
    if !value.is_finite() {
        return Err(GprError::InvalidHyperparameter {
            reason: format!("{what} must be finite"),
        });
    }
    if value <= 0.0 {
        return Err(GprError::InvalidHyperparameter {
            reason: format!("{what} must be positive"),
        });
    }
    Ok(())
}

fn validate_log_positive(theta: f64, what: &str) -> Result<f64, GprError> {
    if !theta.is_finite() {
        return Err(GprError::InvalidHyperparameter {
            reason: format!("log {what} must be finite"),
        });
    }
    let value = theta.exp();
    if !value.is_finite() {
        return Err(GprError::InvalidHyperparameter {
            reason: format!("{what} overflowed to a non-finite value"),
        });
    }
    if value <= 0.0 {
        return Err(GprError::InvalidHyperparameter {
            reason: format!("{what} underflowed to zero"),
        });
    }
    Ok(theta)
}
