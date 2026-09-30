//! The comparison regression, gprx: ARD RBF, learned from a fixed start.
//! Run: `cargo run --release --example gprx_ard` in `compare/perf/gprx`.
use gprx::kernel::{ConstantKernel, KernelSpec, RbfArdKernel};
use gprx::{GaussianLikelihood, Gpr};

fn main() -> Result<(), gprx::GprError> {
    // A small deterministic problem: column-major x (n × d), y = sin(Σ x) + noise.
    let (n, m, d) = (200, 50, 3);
    let mut state = 12345_u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0
    };
    let x: Vec<f64> = (0..n * d).map(|_| next()).collect();
    let xs: Vec<f64> = (0..m * d).map(|_| next()).collect();
    let sum = |v: &[f64], rows: usize, i: usize| (0..d).map(|j| v[j * rows + i]).sum::<f64>();
    let y: Vec<f64> = (0..n).map(|i| sum(&x, n, i).sin() + 0.1 * next()).collect();
    let ys: Vec<f64> = (0..m).map(|i| sum(&xs, m, i).sin()).collect();

    // snippet:begin
    let kernel = KernelSpec::from(ConstantKernel::new(1.0)?)
        * KernelSpec::from(RbfArdKernel::new(&vec![1.0; d])?);
    let fitted = Gpr::new(kernel, GaussianLikelihood::new(0.1)?)
        .fit(&x, n, d, &y) // L-BFGS on the negative log marginal likelihood
        .map_err(|(_, e)| e)?;
    let pred = fitted.predict(&xs, m, d)?; // variance includes the noise
    let nlpd = (0..m)
        .map(|i| {
            let (v, e) = (pred.variance[i], ys[i] - pred.mean[i]);
            0.5 * (2.0 * std::f64::consts::PI * v).ln() + 0.5 * e * e / v
        })
        .sum::<f64>()
        / m as f64;
    // snippet:end
    println!(
        "gprx     mean[0]={:.4} std[0]={:.4} nlpd={nlpd:.4}",
        pred.mean[0],
        pred.variance[0].sqrt()
    );
    Ok(())
}
