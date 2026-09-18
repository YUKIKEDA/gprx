# gprx

Exact Gaussian process regression in Rust. `Gpr` is the unfitted trainer. `Gpr::fit` consumes it, runs argmin L-BFGS on the negative log marginal likelihood, and returns `FittedGpr`. The crate is **not** published to crates.io (`publish = false` in `Cargo.toml`).

## Status

Local **0.1.0** quality: `Gpr` / `FittedGpr`, kernels, `fit` / `predict` / `predict_into` / leave-one-out, English rustdoc, and `examples/`. Depend on git or a path, not crates.io.

Design: [`.dev/gprx-design.md`](.dev/gprx-design.md). Tasks: [`.dev/roadmap.md`](.dev/roadmap.md). Agent rules: [`AGENTS.md`](AGENTS.md).

## Example

`X` is column-major (`n` points × `d` features: all rows of feature 0, then feature 1, …).

```rust
use gprx::kernel::{KernelSpec, RbfKernel};
use gprx::{GaussianLikelihood, Gpr};

fn main() -> Result<(), gprx::GprError> {
    let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    let likelihood = GaussianLikelihood::new(0.1)?;
    let gpr = Gpr::new(kernel, likelihood);
    let fitted = gpr.fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0])?;
    let pred = fitted.predict(&[0.5], 1, 1)?;
    println!("mean = {}, variance = {}", pred.mean[0], pred.variance[0]);
    Ok(())
}
```

Same program: `cargo run --example fit_predict`. `?` on `fit` drops the trainer (`From<(Gpr, GprError)> for GprError`). Use `.map_err(|(_, e)| e)` when you want only the error, or match `Err((gpr, err))` to retry with the same trainer.

Default `predict` variance is observation (`latent + σn²`). Use `predict_with` and `VarianceKind::Latent` for the latent function. After fit, `loo_predict` is the GPML leave-one-out at every training point. `predict` allocates query buffers; `predict_into` reuses them after a warmup call. `FittedGpr::refit` re-factors or re-optimizes on the stored training data.

Transforms default to identity. Call `with_target_transform(StandardizeTarget::new())` before `fit` when the mean function is zero. Features can use `MinMaxInput` (default `[0, 1]`). Observation noise lives in `GaussianLikelihood`. `WhiteKernel` is opt-in composition; using both at large values double-counts noise.

`Gpr<Fixed>::factor` (after `with_optimizer(Fixed)`) factors at the kernel and likelihood `θ` already on the trainer. L-BFGS knobs live on `Lbfgs` (`with_max_iterations`, `with_tolerance`, `with_history_size`, `with_restarts`). Nonlinear CG and Nelder–Mead share the first three knobs except `history_size` (`NonlinearCg`, `NelderMead`).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
