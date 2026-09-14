//! Input (`X`) and target (`y`) transforms.
//!
//! Phase 1 provides [`IdentityInput`] / [`StandardizeInput`] for features and
//! [`IdentityTarget`] / [`StandardizeTarget`] for observations. A pipeline of
//! extra maps (MinMax, …) waits until a later issue.

mod input;
mod target;

pub use input::{IdentityInput, StandardizeInput, Transform};
pub use target::{IdentityTarget, StandardizeTarget, TargetTransform};

use crate::error::GpError;

fn require_nonempty(n: usize) -> Result<(), GpError> {
    if n == 0 {
        Err(GpError::EmptyInput)
    } else {
        Ok(())
    }
}

fn require_finite(values: &[f64]) -> Result<(), GpError> {
    if values.iter().any(|value| !value.is_finite()) {
        Err(GpError::NonFiniteInput)
    } else {
        Ok(())
    }
}

fn require_len(values: &[f64], expected: usize) -> Result<(), GpError> {
    if values.len() == expected {
        Ok(())
    } else {
        Err(GpError::InvalidHyperparameter {
            reason: format!("expected {expected} values, got {}", values.len()),
        })
    }
}

fn population_std(values: &[f64], mean: f64) -> f64 {
    let n = values.len() as f64;
    let var = values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / n;
    let std = var.sqrt();
    if std > 0.0 && std.is_finite() {
        std
    } else {
        1.0
    }
}

fn column_major_len(n_rows: usize, n_cols: usize) -> Result<usize, GpError> {
    require_nonempty(n_rows)?;
    require_nonempty(n_cols)?;
    Ok(n_rows * n_cols)
}
