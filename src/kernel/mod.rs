//! Kernel leaves and composition ([`KernelSpec`] / [`CompiledKernel`]).
//!
//! Phase 1 evaluates isotropic RBF from a squared-Euclidean distance matrix.
//! Callers pass faer views; this module does not re-export faer types.

mod compiled;
mod rbf;
mod spec;

pub use compiled::CompiledKernel;
pub use rbf::RbfKernel;
pub use spec::{KernelSpec, ParameterBinding};

use crate::error::GprError;
use faer::{MatMut, MatRef};

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
    mut kernel: impl FnMut(f64) -> Result<f64, GprError>,
) -> Result<(), GprError> {
    let n = require_square_pair(dist, out.as_ref())?;
    let mut err = None;
    visit_triangle(n, uplo, |row, col| {
        if err.is_some() {
            return;
        }
        let d = dist[(row, col)];
        match kernel(d) {
            Ok(value) => out[(row, col)] = value,
            Err(e) => err = Some(e),
        }
    });
    match err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

fn finite_dist(d: f64) -> Result<f64, GprError> {
    if d.is_finite() {
        Ok(d)
    } else {
        Err(GprError::NonFiniteInput)
    }
}
