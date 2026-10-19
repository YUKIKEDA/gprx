English | [日本語](README.ja.md)

# gprx

Gaussian process regression in Rust: exact GPR, sparse GPR (VFE), and SVGP. `Gpr` is the unfitted trainer. `Gpr::fit` consumes it, minimizes the negative log marginal likelihood with the default `Lbfgs`, and returns `FittedGpr`. The same blocks build `Sgpr` and `Svgp`. Exact and sparse GPR models also update online, and every model saves to and loads from a directory. A kernel reads coordinates, or squared distances you supply.

`X` is column-major: `n` points by `d` features, feature 0 for every row, then feature 1. `fit` consumes the trainer. Observation noise lives in `GaussianLikelihood`. **0.2.0** (tag `v0.2.0`) is the default-feature public API. The MSRV is 1.88. A 0.x minor may break that API: what 0.2.0 changed from 0.1.0, and how to move, is in the [changelog](https://github.com/YUKIKEDA/gprx/blob/main/CHANGELOG.md). `internals` (`bench-internals` and `insert-stages`) is outside that contract.

```toml
[dependencies]
gprx = "0.2"
```

## Example

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

Same program: `cargo run --example fit_predict`.

## What to know

`fit` and `factor` return `Result<Fitted, (Trainer, GprError)>`: a failure hands the trainer back with the error. On `Gpr`, `?` drops the trainer and keeps the `GprError`. `Sgpr` and `Svgp` have no such conversion, so write `.map_err(|(_, e)| e)?`, as the examples below do.

`x` is column-major `f64` with `n_rows` points and `n_cols` features, and `y` has `n_rows` values. Empty input, a slice whose length is not `n_rows * n_cols`, and `NaN` or `Inf` are a `GprError`, not a panic.

Call a model's methods on the value (`fitted.predict(..)`). A call by the type's path without its parameters, such as `FittedGpr::predict(&fitted, ..)`, does not compile (`E0034`), because the models on supplied distances have methods of the same names. Write `<FittedGpr>::predict(&fitted, ..)` instead.

## Choose a model

| Model | Use it when | Training cost |
| --- | --- | --- |
| `Gpr` | the `n × n` kernel matrix fits in memory. Exact predictions | `O(n³)` time, `O(n²)` memory |
| `Sgpr` | `n` is large. `m` inducing points summarize the data (VFE) | `O(n m²)` per evaluation |
| `Svgp` | `n` is large and you train on minibatches with `Adam` | `O(b (m² + m d) + m³)` per step of batch `b` |

`FittedGpr::into_online` and `FittedSgpr::into_online` return `OnlineGpr` and `OnlineSgpr`, which add and remove points without a new fit. `Svgp` has no online form.

## How-to

Each block below is a complete program. It needs only `gprx = "0.2"` unless it says otherwise.

### Choose and combine kernels

`+` adds two kernels and `*` multiplies them. `*` binds tighter than `+`. An ARD leaf such as `RbfArdKernel` has one lengthscale per feature. The leaves, their parameters, and their bounds are listed in the rustdoc of [`gprx::kernel`](https://docs.rs/gprx/latest/gprx/kernel/).

```rust
use gprx::kernel::{ConstantKernel, KernelSpec, PeriodicKernel, RbfArdKernel, RbfKernel};
use gprx::{GaussianLikelihood, Gpr};

fn main() -> Result<(), gprx::GprError> {
    // A signal variance times a smooth trend, plus a periodic term.
    let trend = KernelSpec::from(ConstantKernel::new(2.0)?) * KernelSpec::from(RbfKernel::new(5.0)?);
    let seasonal = trend + KernelSpec::from(PeriodicKernel::new(1.0, 0.5)?);
    assert_eq!(seasonal.num_params(), 4);

    // ARD: three points with two features, column-major.
    let ard = KernelSpec::from(RbfArdKernel::new(&[1.0, 10.0])?);
    let x = [0.0, 1.0, 2.0, 5.0, 3.0, 8.0];
    let fitted = Gpr::new(ard, GaussianLikelihood::new(0.1)?).fit(&x, 3, 2, &[0.0, 1.0, 0.5])?;
    let pred = fitted.predict(&[1.5, 4.0], 1, 2)?;
    println!("mean = {}", pred.mean[0]);
    Ok(())
}
```

### Use your own distance matrix

A `ScalarDistance` slot lets a kernel leaf read squared distances you computed, such as a geodesic or graph distance, instead of coordinates. Tables are column-major. Every value must be finite and non-negative, and a training table symmetric with a zero diagonal, or the call returns `GprError::InvalidDistance`. `tidy(rel_tol)` on a source repairs a table that rounding left slightly off. The rustdoc of [`ScalarDistance`](https://docs.rs/gprx/latest/gprx/kernel/struct.ScalarDistance.html) covers one table per feature (`ArdDistance`), mixing distances with coordinates (`WithPoints`), the sparse models, and filling columns on demand (`fill`).

```rust
use gprx::kernel::{RbfKernel, ScalarDistance};
use gprx::{GaussianLikelihood, Gpr};

fn main() -> Result<(), gprx::GprError> {
    let slot = ScalarDistance::new();
    let kernel = slot.kernel(RbfKernel::new(1.0)?);
    // Three training points: their 3 × 3 squared distances.
    let d2 = vec![0.0, 1.0, 4.0, 1.0, 0.0, 1.0, 4.0, 1.0, 0.0];
    let y = [0.0, 1.0, 0.5];
    let fitted = Gpr::new(kernel, GaussianLikelihood::new(0.1)?).fit([slot.from_vec(d2)], 3, &y)?;
    // One query: its squared distances to the three training points (3 × 1).
    let cross = [0.25, 0.25, 2.25];
    let pred = fitted.predict([slot.borrow(&cross)], 1)?;
    println!("mean = {}", pred.mean[0]);
    Ok(())
}
```

### Inducing points for large data

`Sgpr` takes inducing inputs `z`, column-major like `x`, after `y`. They stay where you put them. `with_inducing(FreeInducing)` moves them together with the hyperparameters.

```rust
use gprx::kernel::{KernelSpec, RbfKernel};
use gprx::{GaussianLikelihood, Sgpr};

fn main() -> Result<(), gprx::GprError> {
    // 500 points on [0, 25), one feature.
    let x: Vec<f64> = (0..500).map(|i| f64::from(i) * 0.05).collect();
    let y: Vec<f64> = x.iter().map(|v| v.sin()).collect();
    // 20 inducing points.
    let z: Vec<f64> = (0..20).map(|i| f64::from(i) * 1.25).collect();
    let fitted = Sgpr::new(KernelSpec::from(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?)
        .fit(&x, 500, 1, &y, &z, 20)
        .map_err(|(_, e)| e)?;
    let pred = fitted.predict(&[3.3], 1, 1)?;
    println!("mean = {}, variance = {}", pred.mean[0], pred.variance[0]);
    Ok(())
}
```

### Minibatch training

`Svgp::new` does not train: `factor` places the variational posterior at the prior. `with_optimizer(Adam::new())` makes `fit` run minibatch Adam on the hyperparameters and that posterior. `Adam` defaults to learning rate `1e-3`, batch 32, 100 epochs, and seed 0. `Adam` works only with `Svgp`.

```rust
use gprx::kernel::{KernelSpec, RbfKernel};
use gprx::{Adam, GaussianLikelihood, Svgp};

fn main() -> Result<(), gprx::GprError> {
    let x: Vec<f64> = (0..200).map(|i| f64::from(i) * 0.05).collect();
    let y: Vec<f64> = x.iter().map(|v| v.sin()).collect();
    let z: Vec<f64> = (0..10).map(f64::from).collect();
    let adam = Adam::new().with_learning_rate(1e-2)?.with_seed(7);
    let fitted = Svgp::new(KernelSpec::from(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?)
        .with_optimizer(adam)
        .fit(&x, 200, 1, &y, &z, 10)
        .map_err(|(_, e)| e)?;
    println!("negative ELBO = {}", fitted.neg_elbo()?);
    let pred = fitted.predict(&[2.5], 1, 1)?;
    println!("mean = {}", pred.mean[0]);
    Ok(())
}
```

### Add and remove points later

`into_online` keeps the hyperparameters and updates the factor by one point at a time, in `O(n²)` for `OnlineGpr`. `insert` returns a `PointId`. `delete` takes it back. The last point cannot be deleted. `refit` searches the hyperparameters again on the current points. `OnlineSgpr` also adds and removes inducing points (`insert_inducing`, `delete_inducing`).

```rust
use gprx::kernel::{KernelSpec, RbfKernel};
use gprx::{GaussianLikelihood, Gpr, OnlineGpr, PointId};

fn main() -> Result<(), gprx::GprError> {
    let fitted = Gpr::new(KernelSpec::from(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?)
        .fit(&[0.0, 1.0, 2.0], 3, 1, &[0.0, 1.0, 0.5])?;
    let mut online: OnlineGpr = fitted.into_online()?;
    // A new point at x = 3 with y = 0.25.
    let id: PointId = online.insert(&[3.0], 0.25)?;
    let pred = online.predict(&[2.5], 1, 1)?;
    online.delete(id)?;
    online.refit()?;
    println!("mean = {}, points = {}", pred.mean[0], online.n());
    Ok(())
}
```

### Fix hyperparameters or choose an optimizer

`Gpr::new` and `Sgpr::new` search with `Lbfgs`. `with_optimizer` swaps in `NelderMead`, `TrustRegion`, or `FastSimulatedAnnealing`, each with its own setters. `with_optimizer(Fixed)` replaces `fit` with `factor`, which keeps the hyperparameters you gave. `with_bounds` on a kernel leaf or on `GaussianLikelihood` limits where a search may go. The rustdoc of [`Optimizer`](https://docs.rs/gprx/latest/gprx/trait.Optimizer.html) compares the optimizers and shows how to write your own.

```rust
use std::num::NonZeroU32;

use gprx::kernel::{KernelSpec, RbfKernel};
use gprx::{Fixed, GaussianLikelihood, Gpr, Interval, Lbfgs, NelderMead};

fn main() -> Result<(), gprx::GprError> {
    let (x, y) = ([0.0, 1.0, 2.0], [0.0, 1.0, 0.5]);
    // Keep the lengthscale inside (0.1, 10).
    let rbf = RbfKernel::new(1.0)?.with_bounds(Interval::new(0.1, 10.0)?)?;
    let kernel = KernelSpec::from(rbf);
    let likelihood = GaussianLikelihood::new(0.1)?;

    let fixed = Gpr::new(kernel.clone(), likelihood.clone()).with_optimizer(Fixed).factor(&x, 3, 1, &y)?;
    // 200 iterations and one extra start drawn with seed 42.
    let lbfgs = Lbfgs::new().with_max_iterations(200).with_restarts(NonZeroU32::MIN, 42);
    let searched = Gpr::new(kernel.clone(), likelihood.clone()).with_optimizer(lbfgs).fit(&x, 3, 1, &y)?;
    let simplex = Gpr::new(kernel, likelihood).with_optimizer(NelderMead::new()).fit(&x, 3, 1, &y)?;
    for nlml in [fixed.neg_log_marginal_likelihood()?, searched.neg_log_marginal_likelihood()?, simplex.neg_log_marginal_likelihood()?] {
        println!("negative log marginal likelihood = {nlml}");
    }
    Ok(())
}
```

### Transform inputs and targets

Pass an unfitted map before `fit`. The model fits it on the training data, applies it to every query, and returns predictions in the original units of `y`. The prior mean is zero, so use `StandardizeTarget` when `y` is not centred near zero. [`gprx::transform`](https://docs.rs/gprx/latest/gprx/transform/) also has min-max scaling, pipelines, per-column maps, and the traits for a map of your own.

```rust
use gprx::kernel::{KernelSpec, RbfKernel};
use gprx::transform::{StandardizeInput, StandardizeTarget};
use gprx::{GaussianLikelihood, Gpr};

fn main() -> Result<(), gprx::GprError> {
    // Inputs in the thousands, targets far from zero.
    let x = [1000.0, 2000.0, 3000.0, 4000.0];
    let y = [250.0, 260.0, 255.0, 270.0];
    let fitted = Gpr::new(KernelSpec::from(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?)
        .with_input_transform(StandardizeInput::new())
        .with_target_transform(StandardizeTarget::new())
        .fit(&x, 4, 1, &y)?;
    // The query and the prediction are in the original units.
    let pred = fitted.predict(&[2500.0], 1, 1)?;
    println!("mean = {}", pred.mean[0]);
    Ok(())
}
```

### Save and load

`save(dir)` writes `config.json` and `model.safetensors`. `LoadedGpr::load` returns an enum with one variant per precision and online state. `predict` works on any variant and returns `f64`. Match a variant to get the typed model back. `FittedGpr::save_with_factor` also writes the Cholesky factor, so loading does not factor again. `LoadedSgpr`, `LoadedSvgp`, and the loaders for models on supplied distances are in [`gprx::persist`](https://docs.rs/gprx/latest/gprx/persist/), with `PersistRegistry` for kernels and transforms of your own. The file format is documented in [persist-format.md](https://github.com/YUKIKEDA/gprx/blob/main/docs/persist-format.md).

```rust
use gprx::kernel::{KernelSpec, RbfKernel};
use gprx::persist::{LoadedGpr, PersistRegistry};
use gprx::{GaussianLikelihood, Gpr};

fn main() -> Result<(), gprx::GprError> {
    let fitted = Gpr::new(KernelSpec::from(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?)
        .fit(&[0.0, 1.0, 2.0], 3, 1, &[0.0, 1.0, 0.5])?;
    let dir = std::env::temp_dir().join(format!("gprx-readme-save-en-{}", std::process::id()));
    fitted.save(&dir)?;
    // Built-in kernels and transforms need no registration.
    let loaded = LoadedGpr::load(&dir, &PersistRegistry::new())?;
    assert_eq!(loaded.predict(&[0.5], 1, 1)?, fitted.predict(&[0.5], 1, 1)?);
    if let LoadedGpr::Double(model) = loaded {
        println!("loaded {} points", model.n());
    }
    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}
```

### Speed, memory, and precision

`with_prefer_speed` (the default) keeps the training distances between kernel builds and a separate `n × n` gradient matrix. `with_prefer_memory` recomputes the distances and writes the gradient matrix over the Cholesky factor, which saves one `n × n` matrix and factors again afterwards. `with_math(KernelExp::FastApprox)` evaluates the kernel `exp` with a polynomial, within a relative `2^{-23}` of `f64::exp` on `f64`. `SinglePrecision` stores and factors in `f32` and predicts `f32`. `MixedPrecision` factors in `f32`, refines in `f64`, and predicts `f64`. `Sgpr` and `Svgp` take `with_precision` and `with_math` too. The rustdoc of [`GpScalar`](https://docs.rs/gprx/latest/gprx/trait.GpScalar.html) compares the precisions.

```rust
use gprx::kernel::{KernelSpec, RbfKernel};
use gprx::{GaussianLikelihood, Gpr, KernelExp, MixedPrecision, Prediction, SinglePrecision};

fn main() -> Result<(), gprx::GprError> {
    let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    let likelihood = GaussianLikelihood::new(0.1)?;
    let (x, y) = ([0.0, 1.0, 2.0], [0.0, 1.0, 0.5]);

    let small = Gpr::new(kernel.clone(), likelihood.clone())
        .with_prefer_memory()
        .with_math(KernelExp::FastApprox)
        .with_precision::<SinglePrecision>()
        .fit(&x, 3, 1, &y)?;
    let single: Prediction<f32> = small.predict(&[0.5], 1, 1)?;

    let mixed = Gpr::new(kernel, likelihood).with_precision::<MixedPrecision>().fit(&x, 3, 1, &y)?;
    let refined: Prediction<f64> = mixed.predict(&[0.5], 1, 1)?;
    println!("f32 mean = {}, mixed mean = {}", single.mean[0], refined.mean[0]);
    Ok(())
}
```

### Write your own kernel

Implement `KernelTerm<T>` for every `T: KernelScalar` on a type that is `Clone + Debug + Send + Sync + 'static`, then insert it with `KernelSpec::custom`. A leaf reads squared distances as `faer` matrices, so add `faer = "0.24"`. The sparse models call further derivative methods (`grad_cross`, `hess_cross`, and for `FreeInducing` the `*_wrt_sq_dist` methods), whose defaults return `GprError::CoordGradientUnsupported`. Saving the leaf needs `persist_id` and `persist_state` (add `serde_json = "1"`). The rustdoc of [`KernelTerm`](https://docs.rs/gprx/latest/gprx/kernel/trait.KernelTerm.html) has a leaf with a parameter, every derivative, and save and load.

```rust
use faer::{MatMut, MatRef};
use gprx::kernel::{ConstantKernel, KernelScalar, KernelSpec, KernelTerm, Triangle};
use gprx::{GaussianLikelihood, Gpr, GprError, Interval};

/// `k(d²) = exp(−d² / 2)`, with no parameter.
#[derive(Clone, Debug)]
struct UnitRbf;

fn none(len: usize) -> Result<(), GprError> {
    match len {
        0 => Ok(()),
        _ => Err(GprError::LengthMismatch { reason: format!("UnitRbf has no parameters, got {len}") }),
    }
}

impl<T: KernelScalar> KernelTerm<T> for UnitRbf {
    fn num_params(&self) -> usize { 0 }
    fn get_params(&self, out: &mut [f64]) -> Result<(), GprError> { none(out.len()) }
    fn set_params(&mut self, params: &[f64]) -> Result<(), GprError> { none(params.len()) }
    fn bounds_into(&self, out: &mut [Interval]) -> Result<(), GprError> { none(out.len()) }
    fn apply(&self, d2: MatRef<'_, T>, mut out: MatMut<'_, T>, uplo: Triangle) -> Result<(), GprError> {
        for col in 0..d2.ncols() {
            let rows = match uplo {
                Triangle::Lower => col..d2.nrows(),
                Triangle::Upper => 0..col + 1,
                Triangle::Full => 0..d2.nrows(),
            };
            for row in rows {
                out[(row, col)] = (T::from_f64(-0.5) * d2[(row, col)]).exp();
            }
        }
        Ok(())
    }
    fn apply_cross(&self, d2: MatRef<'_, T>, out: MatMut<'_, T>) -> Result<(), GprError> { self.apply(d2, out, Triangle::Full) }
    fn fill_diag(&self, out: &mut [T]) -> Result<(), GprError> { out.fill(T::from_f64(1.0)); Ok(()) }
    fn grad(&self, _: MatRef<'_, T>, _: MatMut<'_, T>, i: usize, _: Triangle) -> Result<(), GprError> { none(i + 1) }
    fn hess(&self, _: MatRef<'_, T>, _: MatMut<'_, T>, i: usize, _: usize, _: Triangle) -> Result<(), GprError> { none(i + 1) }
    fn hess_points(&self, _: MatRef<'_, T>, _: MatMut<'_, T>, i: usize, _: usize, _: Triangle) -> Result<(), GprError> { none(i + 1) }
}

fn main() -> Result<(), GprError> {
    // A signal variance times the custom leaf.
    let kernel = KernelSpec::from(ConstantKernel::new(1.0)?) * KernelSpec::custom(UnitRbf);
    let fitted = Gpr::new(kernel, GaussianLikelihood::new(0.1)?).fit(&[0.0, 1.0, 2.0], 3, 1, &[0.0, 1.0, 0.5])?;
    println!("mean = {}", fitted.predict(&[0.5], 1, 1)?.mean[0]);
    Ok(())
}
```

### Handle errors

`GprError` is non-exhaustive, so a `match` on it ends with a catch-all arm. When `fit` fails, the trainer comes back with the error, and you can change it and try again. Branch on the variant and its `kind` field, not on the display text.

```rust
use gprx::kernel::{KernelSpec, RbfKernel};
use gprx::{GaussianLikelihood, Gpr, GprError, JitterPolicy};

fn main() -> Result<(), GprError> {
    let (x, y) = ([0.0, 1.0, 2.0], [0.0, 1.0, 0.5]);
    let trainer = Gpr::new(KernelSpec::from(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?);
    let fitted = match trainer.fit(&x, 3, 1, &y) {
        Ok(fitted) => fitted,
        // Retry with jitter on the diagonal.
        Err((trainer, GprError::CholeskyFailed { .. })) => {
            trainer.with_jitter_policy(JitterPolicy::fixed(1e-6)?).fit(&x, 3, 1, &y)?
        }
        Err((_, other)) => return Err(other),
    };
    // Two features in the query, one in training.
    match fitted.predict(&[0.5, 1.0], 1, 2) {
        Err(GprError::DimensionMismatch { x_dim, expected_dim }) => {
            println!("query has {x_dim} features, the model {expected_dim}");
        }
        other => println!("{other:?}"),
    }
    Ok(())
}
```

### Set the thread count

gprx has no `n_jobs` setting. It runs on the process-wide Rayon pool. Set `RAYON_NUM_THREADS` before the process starts, or build the global pool before the first fit or predict, which needs `rayon = "1"`. The global pool can be built only once, and one thread runs everything sequentially. The linear-algebra backend runs on as many of those threads as the problem size warrants, as the [crate rustdoc](https://docs.rs/gprx/latest/gprx/) explains.

```rust
use gprx::kernel::{KernelSpec, RbfKernel};
use gprx::{GaussianLikelihood, Gpr};

fn main() -> Result<(), gprx::GprError> {
    // Fails if the global pool was already built in this process.
    if let Err(err) = rayon::ThreadPoolBuilder::new().num_threads(4).build_global() {
        eprintln!("thread pool not changed: {err}");
    }
    let fitted = Gpr::new(KernelSpec::from(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?)
        .fit(&[0.0, 1.0, 2.0], 3, 1, &[0.0, 1.0, 0.5])?;
    println!("threads = {}, mean = {}", rayon::current_num_threads(), fitted.predict(&[0.5], 1, 1)?.mean[0]);
    Ok(())
}
```

## API reference

Every public item has a usage example in the API reference on [docs.rs](https://docs.rs/gprx). Start from:

- [`gprx`](https://docs.rs/gprx/latest/gprx/): the models (`Gpr`, `Sgpr`, `Svgp` and their fitted and online forms), `GaussianLikelihood`, the optimizers, the precisions, and `GprError`.
- [`gprx::kernel`](https://docs.rs/gprx/latest/gprx/kernel/): kernel leaves, `KernelSpec`, supplied distances, and `KernelTerm`.
- [`gprx::transform`](https://docs.rs/gprx/latest/gprx/transform/): input and target maps.
- [`gprx::persist`](https://docs.rs/gprx/latest/gprx/persist/): loaders and `PersistRegistry`.

How the crate is built: [`docs/architecture.md`](https://github.com/YUKIKEDA/gprx/blob/main/docs/architecture.md) maps every module and the way its imports point. [`docs/persist-format.md`](https://github.com/YUKIKEDA/gprx/blob/main/docs/persist-format.md) describes what `save` writes. [`docs/design.md`](https://github.com/YUKIKEDA/gprx/blob/main/docs/design.md) records the algorithms and the decisions behind them.

## Comparison

Accuracy, training time, and memory against other libraries: [comparison](https://github.com/YUKIKEDA/gprx/blob/main/docs/comparison.md) | [日本語](https://github.com/YUKIKEDA/gprx/blob/main/docs/comparison.ja.md).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
