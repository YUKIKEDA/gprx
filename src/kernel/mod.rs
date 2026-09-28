//! Kernel leaves and composition ([`KernelSpec`] / [`CompiledKernel`]).
//!
//! Built-in leaves are enum arms. User distance leaves implement [`KernelTerm`]
//! and enter the tree as [`KernelSpec::Custom`].
//!
//! Isotropic RBF, Matérn, Periodic, and rational quadratic evaluate from a
//! squared-Euclidean distance matrix (Periodic then takes the square root).
//! ARD RBF, ARD Matérn, and ARD rational quadratic evaluate from coordinates
//! via [`ArdLengthscales`] (`θ_d = log(ℓ_d)`). Callers pass faer views; this
//! module does not re-export faer types.
//!
//! Squared-Euclidean fills and [`Triangle::Lower`] writes run on Rayon's
//! global pool. Isotropic RBF, ARD RBF, and the distance / `(Δx_d)²` row loops
//! use `wide::f64x4` when the faer view is column-major with unit row stride.
//! There is no parallel on/off flag; `RAYON_NUM_THREADS=1` is sequential. Limit
//! threads with `RAYON_NUM_THREADS` or
//! `rayon::ThreadPoolBuilder::build_global` before the first fill. Linear and
//! other points-mode leaves stay sequential. See the [crate-level parallelism
//! notes](crate).

mod compiled;
mod constant;
mod dist;
mod lengthscale;
mod linear;
mod matern;
mod matern_ard;
mod periodic;
mod rbf;
mod rbf_ard;
mod rq;
mod rq_ard;
mod scalar;
mod simd;
mod spec;
mod term;
mod white;

pub use compiled::CompiledKernel;
pub(crate) use compiled::gram::GramKernel;
pub(crate) use compiled::{CoordMode, MixedKernelViews};
pub use constant::ConstantKernel;
#[doc(hidden)]
pub use dist::fill_ard_squared_diff;
pub(crate) use dist::{FillDistances, fill_squared_euclidean};
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
pub use spec::{KernelSpec, ParameterBinding};
pub use term::{CustomKernel, KernelTerm};
pub use white::WhiteKernel;

use crate::error::GprError;
use dist::{col_chunk, worker_count};
use faer::{MatMut, MatRef};
use rayon::prelude::*;

/// Which triangle of a symmetric kernel matrix to write.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Triangle {
    /// Entries with `row >= col`. This is the default for Cholesky.
    Lower,
    /// Entries with `row <= col`.
    Upper,
    /// Every entry. Upper and lower must match when `dist` is symmetric.
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

fn require_square_pair(dist: MatRef<'_, f64>, out: MatRef<'_, f64>) -> Result<usize, GprError> {
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

pub(crate) fn require_coord_grad(
    x1: MatRef<'_, f64>,
    x2: MatRef<'_, f64>,
    d_k: MatRef<'_, f64>,
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

fn require_same_shape(dist: MatRef<'_, f64>, out: MatRef<'_, f64>) -> Result<(), GprError> {
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

fn write_dense(
    dist: MatRef<'_, f64>,
    mut out: MatMut<'_, f64>,
    mut kernel: impl FnMut(f64) -> Result<f64, GprError>,
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

fn write_triangle(
    dist: MatRef<'_, f64>,
    mut out: MatMut<'_, f64>,
    uplo: Triangle,
    kernel: impl Fn(f64) -> Result<f64, GprError> + Sync,
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

fn write_lower_parallel(
    dist: MatRef<'_, f64>,
    out: MatMut<'_, f64>,
    kernel: impl Fn(f64) -> Result<f64, GprError> + Sync,
) -> Result<(), GprError> {
    let n = dist.nrows();
    let n_parts = worker_count();
    out.par_col_partition_mut(n_parts)
        .enumerate()
        .try_for_each(|(chunk_idx, mut part)| {
            let (start, len) = col_chunk(n, chunk_idx, n_parts);
            for local in 0..len {
                let col = start + local;
                for row in col..n {
                    part[(row, local)] = kernel(dist[(row, col)])?;
                }
            }
            Ok::<(), GprError>(())
        })
}

/// Fills pairwise squared Euclidean distances for criterion's `kernel_rbf`.
///
/// Hidden so benches can share the library fill without duplicating the
/// Rayon partition. Not part of the documented public API.
#[doc(hidden)]
pub fn fill_pairwise_sq_euclidean(x: MatRef<'_, f64>, dist: MatMut<'_, f64>) {
    fill_squared_euclidean(x, dist, &mut []);
}

fn finite_dist(d: f64) -> Result<f64, GprError> {
    if d.is_finite() {
        Ok(d)
    } else {
        Err(GprError::NonFiniteInput)
    }
}

pub(crate) fn pair_squared_euclidean(x: MatRef<'_, f64>, i: usize, j: usize) -> f64 {
    let d = x.ncols();
    let mut sum = 0.0;
    for dim in 0..d {
        let diff = x[(i, dim)] - x[(j, dim)];
        sum += diff * diff;
    }
    sum
}

/// Writes a kernel triangle from coordinates. Each pair computes `‖x_i-x_j‖²`
/// without a distance matrix.
pub(crate) fn write_square_from_coords(
    x: MatRef<'_, f64>,
    out: MatMut<'_, f64>,
    uplo: Triangle,
    kernel: impl Fn(f64) -> Result<f64, GprError> + Sync,
) -> Result<(), GprError> {
    let n = out.nrows();
    if out.ncols() != n {
        return Err(GprError::InvalidHyperparameter {
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

fn write_lower_from_coords_parallel(
    x: MatRef<'_, f64>,
    out: MatMut<'_, f64>,
    kernel: impl Fn(f64) -> Result<f64, GprError> + Sync,
) -> Result<(), GprError> {
    let n = x.nrows();
    let n_parts = worker_count();
    out.par_col_partition_mut(n_parts)
        .enumerate()
        .try_for_each(|(chunk_idx, mut part)| {
            let (start, len) = col_chunk(n, chunk_idx, n_parts);
            for local in 0..len {
                let col = start + local;
                for row in col..n {
                    let d = finite_dist(pair_squared_euclidean(x, row, col))?;
                    part[(row, local)] = kernel(d)?;
                }
            }
            Ok::<(), GprError>(())
        })
}

fn write_square(
    mut out: MatMut<'_, f64>,
    uplo: Triangle,
    mut kernel: impl FnMut(usize, usize) -> Result<f64, GprError>,
) -> Result<(), GprError> {
    if out.nrows() != out.ncols() {
        return Err(GprError::InvalidHyperparameter {
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

fn expect_one_param(len: usize, what: &str) -> Result<(), GprError> {
    if len == 1 {
        Ok(())
    } else {
        Err(GprError::InvalidHyperparameter {
            reason: format!("expected 1 {what} parameter, got {len}"),
        })
    }
}
