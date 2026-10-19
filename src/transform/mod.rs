//! Transforms input (`X`) and target (`y`).
//!
//! Input maps: [`IdentityInput`], [`StandardizeInput`], [`MinMaxInput`].
//! Target maps: [`IdentityTarget`], [`StandardizeTarget`], [`MinMaxTarget`].
//! Stack several maps with [`Pipeline`] (`X`) or [`TargetPipeline`] (`y`).
//! Assign a map per feature with [`ColumnwiseInput`].
//!
//! Pass an unfitted map to `with_input_transform` or `with_target_transform`
//! before `fit`; the model fits the map on the training data. The defaults
//! are [`IdentityInput`] and [`IdentityTarget`].
//!
//! | Type | Role |
//! | --- | --- |
//! | [`IdentityInput`], [`IdentityTarget`] | no change; `fit` returns the same map |
//! | [`StandardizeInput`] | per-feature mean and standard deviation; fits to [`FittedStandardizeInput`] (`mean`, `std`) |
//! | [`StandardizeTarget`] | one mean and standard deviation for `y`; fits to [`FittedStandardizeTarget`] (`mean`, `std`) |
//! | [`MinMaxInput`] | per-feature map into a range: `new` is `[0, 1]`, `with_feature_range(lo, hi)` sets another; fits to [`FittedMinMaxInput`] (`min`, `max`, `feature_range`) |
//! | [`MinMaxTarget`] | the same for `y`; fits to [`FittedMinMaxTarget`] |
//! | [`Pipeline`] | `Pipeline::new().then(step)`, each map fitted on the output of the one before; fits to [`FittedPipeline`] |
//! | [`TargetPipeline`] | the same for `y`; fits to [`FittedTargetPipeline`] |
//! | [`ColumnwiseInput`] | `new().then(map)` assigns the next feature; fits to [`FittedColumnwiseInput`] |
//!
//! The concrete maps also have an inherent `fit`: input maps take
//! `(x, n_rows, n_cols)` and target maps take `y`.
//!
//! A map of your own implements [`UnfittedTransform`] and [`Transform`]
//! (`X`), or [`UnfittedTarget`] and [`TargetTransform`] (`y`); the examples
//! on [`UnfittedTransform`] and [`UnfittedTarget`] show one each, used on a
//! model that is saved and loaded.

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
