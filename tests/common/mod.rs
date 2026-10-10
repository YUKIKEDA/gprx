//! Helpers shared by the integration tests. Each test crate uses a subset.
#![allow(dead_code)]

pub mod check;
pub mod problems;
#[path = "../../src/rng.rs"]
pub mod rng;

#[allow(unused_imports)]
pub use check::*;

/// The places of the inducing points of `online` among its training
/// points ([`gprx::OnlineSgpr::point_ids`]).
pub fn inducing_places<O, P: gprx::GpScalar, C: gprx::kernel::PointUse>(
    online: &gprx::OnlineSgpr<O, P, gprx::kernel::DistanceKernel<C>>,
) -> Vec<usize> {
    let ids = online.point_ids();
    online
        .inducing_points()
        .filter_map(|id| ids.iter().position(|&p| p == id))
        .collect()
}
