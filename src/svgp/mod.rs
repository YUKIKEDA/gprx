//! Stochastic variational GPR with an explicit whitened `q(u)`.
//!
//! [`Svgp<Fixed>::factor`] installs a prior `q`. [`Svgp<Adam>::fit`] runs
//! mini-batch Adam from that prior. [`FittedSvgp::value_and_gradient_into`]
//! is the full-data negative ELBO.

pub(crate) mod factor;
mod fitted;
mod model;

// Declared after the coordinate modules: rustdoc renders inherent impls in
// declaration order, so an intra-doc link to a method name shared by the
// coordinate and the supplied-distance impls lands on the coordinate one.
mod distance;

#[cfg(test)]
mod tests;

pub use fitted::FittedSvgp;
pub use model::Svgp;
