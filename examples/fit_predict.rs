//! Shortest Exact GPR path: fit, then predict.

use gprx::kernel::{KernelSpec, RbfKernel};
use gprx::{GaussianLikelihood, Gpr};

fn main() -> Result<(), gprx::GprError> {
    let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    let likelihood = GaussianLikelihood::new(0.1)?;
    let mut gpr = Gpr::new(kernel, likelihood);
    // Column-major `X`: n = 2 points, d = 1 feature.
    gpr.fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0])?;
    let pred = gpr.predict(&[0.5], 1, 1)?;
    println!("mean = {}, variance = {}", pred.mean[0], pred.variance[0]);
    Ok(())
}
