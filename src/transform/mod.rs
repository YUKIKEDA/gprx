//! Input (`X`) and target (`y`) transforms.
//!
//! Input maps: [`IdentityInput`], [`StandardizeInput`], [`MinMaxInput`].
//! Target maps: [`IdentityTarget`], [`StandardizeTarget`], [`MinMaxTarget`].
//! Stack several maps with [`Pipeline`] (`X`) or [`TargetPipeline`] (`y`).
//! Assign a map per feature with [`ColumnwiseInput`].

mod columnwise;
mod input;
mod pipeline;
mod target;

pub use columnwise::{ColumnwiseInput, FittedColumnwiseInput};
pub use input::{
    FittedMinMaxInput, FittedStandardizeInput, IdentityInput, MinMaxInput, StandardizeInput,
    Transform, UnfittedTransform,
};
pub use pipeline::{FittedPipeline, FittedTargetPipeline, Pipeline, TargetPipeline};
pub use target::{
    FittedMinMaxTarget, FittedStandardizeTarget, IdentityTarget, MinMaxTarget, StandardizeTarget,
    TargetTransform, UnfittedTarget,
};

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
