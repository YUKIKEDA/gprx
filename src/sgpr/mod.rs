//! Collapsed variational SGPR (Titsias / VFE) with caller-supplied inducing points.

mod distance;
pub(crate) mod factor;
mod fitted;
mod model;
mod objective;
mod online;

#[cfg(test)]
mod tests;

pub use fitted::FittedSgpr;
pub use model::Sgpr;
pub(crate) use objective::SgprObjective;
pub(crate) use online::InducingRegistry;
pub use online::{InducingId, OnlineSgpr};

/// Keeps inducing coordinates fixed during [`Sgpr::fit`].
///
/// `Z` is an argument of `fit` / `factor` and is not a parameter.
///
/// See the example on [`Sgpr`].
#[derive(Clone, Copy, Debug, Default)]
pub struct FixedInducing;

/// Optimizes inducing coordinates jointly with kernel and likelihood `θ`.
///
/// Switch with [`Sgpr::with_inducing`]. Params append column-major `Z`
/// after kernel and likelihood `θ`.
///
/// See the example on [`Sgpr`].
#[derive(Clone, Copy, Debug, Default)]
pub struct FreeInducing;

pub trait InducingLayout: Clone {
    fn z_params(m: usize, d: usize) -> usize;
}

impl InducingLayout for FixedInducing {
    fn z_params(_m: usize, _d: usize) -> usize {
        0
    }
}

impl InducingLayout for FreeInducing {
    fn z_params(m: usize, d: usize) -> usize {
        m * d
    }
}
