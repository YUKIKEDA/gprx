//! Variational sparse GPR with caller-supplied inducing points.

mod factor;
mod fitted;
mod model;
mod online;

#[cfg(test)]
mod tests;

pub use fitted::FittedSparseGpr;
pub use model::SparseGpr;
pub use online::OnlineSparseGpr;

/// Keeps inducing coordinates fixed during [`SparseGpr::fit`].
///
/// `Z` is an argument of `fit` / `factor` and is not a parameter.
#[derive(Clone, Copy, Debug, Default)]
pub struct FixedInducing;

/// Optimizes inducing coordinates jointly with kernel and likelihood `θ`.
///
/// Switch with [`SparseGpr::with_inducing`]. Params append column-major `Z`
/// after kernel and likelihood `θ`.
#[derive(Clone, Copy, Debug, Default)]
pub struct FreeInducing;

pub(crate) trait InducingLayout: Clone {
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
