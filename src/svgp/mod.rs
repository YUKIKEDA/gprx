//! Stochastic variational GPR with an explicit whitened `q(u)`.

mod factor;
mod fitted;
mod model;

#[cfg(test)]
mod tests;

pub use fitted::FittedSvgp;
pub use model::Svgp;
