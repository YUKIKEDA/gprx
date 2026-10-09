//! Stochastic variational GPR with an explicit whitened `q(u)`.
//!
//! [`Svgp<Fixed>::factor`] installs a prior `q`. [`Svgp<Adam>::fit`] runs
//! mini-batch Adam from that prior. [`FittedSvgp::value_and_gradient_into`]
//! is the full-data negative ELBO.

mod distance;
pub(crate) mod factor;
mod fitted;
mod model;

#[cfg(test)]
mod tests;

pub use fitted::FittedSvgp;
pub use model::Svgp;
