//! Input (`X`) and target (`y`) transforms.
//!
//! Input maps: [`IdentityInput`], [`StandardizeInput`], [`MinMaxInput`].
//! Target maps: [`IdentityTarget`], [`StandardizeTarget`], [`MinMaxTarget`].
//! Stack several maps with [`Pipeline`] (`X`) or [`TargetPipeline`] (`y`).

mod input;
mod pipeline;
mod target;

pub use input::{
    FittedMinMaxInput, FittedStandardizeInput, IdentityInput, MinMaxInput, StandardizeInput,
    Transform, UnfittedTransform,
};
pub use pipeline::{FittedPipeline, FittedTargetPipeline, Pipeline, TargetPipeline};
pub use target::{
    FittedMinMaxTarget, FittedStandardizeTarget, IdentityTarget, MinMaxTarget, StandardizeTarget,
    TargetTransform, UnfittedTarget,
};

use crate::error::GprError;

fn require_nonempty(n: usize) -> Result<(), GprError> {
    if n == 0 {
        Err(GprError::EmptyInput)
    } else {
        Ok(())
    }
}

fn require_finite(values: &[f64]) -> Result<(), GprError> {
    if values.iter().any(|value| !value.is_finite()) {
        Err(GprError::NonFiniteInput)
    } else {
        Ok(())
    }
}

fn require_len(values: &[f64], expected: usize) -> Result<(), GprError> {
    if values.len() == expected {
        Ok(())
    } else {
        Err(GprError::InvalidHyperparameter {
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

fn column_major_len(n_rows: usize, n_cols: usize) -> Result<usize, GprError> {
    require_nonempty(n_rows)?;
    require_nonempty(n_cols)?;
    Ok(n_rows * n_cols)
}
