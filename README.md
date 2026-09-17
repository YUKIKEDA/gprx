# gprx

Exact Gaussian process regression in Rust. `Gpr::fit` runs argmin L-BFGS on the negative log marginal likelihood. The crate is **not** published to crates.io (`publish = false` in `Cargo.toml`).

## Status

Local **0.1.0** quality: `Gpr`, kernels, `fit` / `predict` / leave-one-out, English rustdoc, and `examples/`. Depend on git or a path, not crates.io.

Design: [`.dev/gprx-design.md`](.dev/gprx-design.md). Tasks: [`.dev/roadmap.md`](.dev/roadmap.md). Agent rules: [`AGENTS.md`](AGENTS.md).

## Example

`X` is column-major (`n` points × `d` features: all rows of feature 0, then feature 1, …).

```rust
use gprx::kernel::{KernelSpec, RbfKernel};
use gprx::{GaussianLikelihood, Gpr};

fn main() -> Result<(), gprx::GprError> {
    let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    let likelihood = GaussianLikelihood::new(0.1)?;
    let mut gpr = Gpr::new(kernel, likelihood);
    gpr.fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0])?;
    let pred = gpr.predict(&[0.5], 1, 1)?;
    println!("mean = {}, variance = {}", pred.mean[0], pred.variance[0]);
    Ok(())
}
```

Same program: `cargo run --example fit_predict`.

Default `predict` variance is observation (`latent + σn²`). Use `predict_with` and `VarianceKind::Latent` for the latent function. After fit, `loo_predict` is the GPML leave-one-out at every training point.

Transforms default to identity. Call `with_target_transform(StandardizeTarget::new())` before `fit` when the mean function is zero. Observation noise belongs in `GaussianLikelihood`; do not also enable a large `WhiteKernel`.

`FitOptions::FIXED` skips L-BFGS and factors at the kernel and likelihood `θ` already on the model.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
