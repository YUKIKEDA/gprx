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

## Usage

`fit` and `factor` return `Result<Fitted, (Trainer, GprError)>`. On `Gpr`, `?` drops the trainer (`From<(Gpr<O, P, K>, GprError)> for GprError`). `Sgpr` and `Svgp` have no such conversion: write `.map_err(|(_, e)| e)?`, as the examples below do. Match `Err((trainer, err))` to retry with the same trainer.

Call a model's methods on the value (`fitted.predict(..)`). A method that the distance models also have (`fit`, `factor`, the `predict` / covariance / `sample` family, `insert`, `insert_inducing`, and the sparse `save`) does not compile when called by its type's path without the type's parameters, such as `FittedGpr::predict(&fitted, ..)` (`E0034`): the path does not say which of the same-named methods it means. Name the parameters the value has: `<FittedGpr>::predict(&fitted, ..)` for a `FittedGpr` with the defaults, `<FittedGpr<Fixed>>::predict(&frozen, ..)` for one from `factor`.

`X` and inducing inputs `Z` are column-major `f64`: feature 0 for every row, then feature 1. `n_rows` is the point count, `n_cols` the feature count. `y` has length `n_rows`. Empty input, a length that is not `n_rows * n_cols`, or `NaN` / `Inf` is `GprError`.

scikit-learn's `n_jobs` knob is not on this crate. Distance fills use the process-wide Rayon pool, and that pool is the thread count. Set `RAYON_NUM_THREADS` before the process starts, or call `rayon::ThreadPoolBuilder::new().num_threads(n).build_global()` before the first fit or predict. The pool can be initialized only once. One worker is sequential. A factor or solve with `n` rows and `k` columns runs the linear-algebra backend on `min(pool, n / 64, n · k / 16384, k / 12)` workers, at least one. The training Cholesky factor and the solve with `n` right-hand sides that the gradient needs have `k = n`; a predict or covariance solve has one column per query point. Kernel fills still use the full pool.

`gprx::internals` (`bench-internals`, `insert-stages`) is outside semantic versioning. Do not depend on it.

### Exact: `Gpr`, `FittedGpr`, `OnlineGpr`

`Gpr::new(kernel, likelihood)` with a `KernelSpec` is `Gpr<Lbfgs, DoublePrecision, KernelSpec>`. `fit(x, n_rows, n_cols, y)` consumes it, runs the optimizer on the negative log marginal likelihood, and returns `FittedGpr` with the same parameters.

```rust
use gprx::kernel::{KernelSpec, RbfKernel};
use gprx::{
    DoublePrecision, Fixed, FittedGpr, GaussianLikelihood, Gpr, Lbfgs, PredictOptions, Prediction,
    VarianceKind,
};

fn main() -> Result<(), gprx::GprError> {
    let trainer: Gpr<Lbfgs, DoublePrecision, KernelSpec> = Gpr::new(
        KernelSpec::from(RbfKernel::new(1.0)?),
        GaussianLikelihood::new(0.1)?,
    );
    let fitted: FittedGpr = trainer.fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0])?;
    // The same call by path, with the type's defaults.
    let _ = <FittedGpr>::predict(&fitted, &[0.5], 1, 1)?;

    let pred = fitted.predict(&[0.5], 1, 1)?;
    let latent = fitted.predict_with(
        &[0.5],
        1,
        1,
        PredictOptions {
            variance_kind: VarianceKind::Latent,
        },
    )?;
    let _ = (pred.mean[0], pred.variance[0], latent.variance_kind);

    let mut fitted = fitted;
    let mut reused = Prediction::default();
    fitted.predict_into(&[0.5], 1, 1, &mut reused)?;
    fitted.predict_with_into(&[0.5], 1, 1, PredictOptions::default(), &mut reused)?;

    let frozen = fitted
        .into_trainer()
        .with_optimizer(Fixed)
        .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0])?;
    let _ = frozen.n();
    Ok(())
}
```

`predict` variance is `VarianceKind::Observation` (`latent + σn²`) on the original `y` scale. `VarianceKind::Latent` drops `σn²`. `predict_with` takes `PredictOptions` (`variance_kind`). `predict` allocates. `predict_into` and `predict_with_into` write into a `Prediction` (`mean`, `variance`, `variance_kind`) and reuse `mean` / `variance` when the query length matches.

`predict_covariance` returns `PredictiveCovariance` (`mean`, `covariance`, `variance_kind`). `covariance` is column-major `m × m` (`col * m + row`). The diagonal matches `predict` for the same query and options. `predict_covariance_with` takes `PredictOptions`.

`sample(xs, n_rows, n_cols, n_draws, seed)` draws `μ + Lz` from that covariance. The result is column-major `m × n_draws`. `seed` starts gprx's Xoshiro256++ (the same sequence on every platform). `sample_with` takes `PredictOptions`.

`loo_predict` is the GPML leave-one-out mean and variance at every training point, not a query at a new `x`. `loo_predict_with` takes `PredictOptions`.

```rust
use gprx::kernel::{KernelSpec, RbfKernel};
use gprx::{GaussianLikelihood, Gpr, PredictOptions, PredictiveCovariance, VarianceKind};

fn main() -> Result<(), gprx::GprError> {
    let mut fitted = Gpr::new(
        KernelSpec::from(RbfKernel::new(1.0)?),
        GaussianLikelihood::new(0.1)?,
    )
    .fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0])?;
    let cov: PredictiveCovariance = fitted.predict_covariance(&[0.25, 0.75], 2, 1)?;
    let cov_latent = fitted.predict_covariance_with(
        &[0.25, 0.75],
        2,
        1,
        PredictOptions {
            variance_kind: VarianceKind::Latent,
        },
    )?;
    let draws = fitted.sample(&[0.25, 0.75], 2, 1, 2, 1)?;
    let _draws_with = fitted.sample_with(&[0.25], 1, 1, PredictOptions::default(), 1, 2)?;
    let loo = fitted.loo_predict()?;
    let _loo_with = fitted.loo_predict_with(PredictOptions::default())?;
    let nll = fitted.neg_log_marginal_likelihood()?;
    let mut theta = vec![0.0; fitted.num_params()];
    fitted.get_params(&mut theta)?;
    let value = fitted.value_and_gradient_into(&theta, &mut theta.clone())?;
    let mut hess = vec![0.0; fitted.num_params() * fitted.num_params()];
    fitted.hessian_into(&theta, &mut hess)?;
    fitted.set_params(&theta)?;
    let _ = (
        cov.covariance[0],
        cov_latent.mean[0],
        draws[0],
        loo.mean[0],
        nll,
        value,
    );
    Ok(())
}
```

`neg_log_marginal_likelihood` is the objective at the current `θ`. `num_params`, `get_params`, and `set_params` are the flat log-`θ` vector: kernel parameters, then the likelihood parameter. `value_and_gradient_into` and `hessian_into` evaluate that objective. `set_params` updates `θ` and the factorization.

`n`, `d`, `kernel`, `likelihood`, `x`, `y`, and `alpha` read the fitted model. `distance_cache_policy`, `cholesky_buffer`, `math`, and `jitter_policy` read the policies. Before `fit`, `Gpr` reads `kernel`, `likelihood`, `num_params`, `get_params`, and those four policy getters. `set_params` starts after `fit`. `FittedGpr::alpha` is a slice from the last factor. `OnlineGpr::alpha` returns `Result`: insert and delete leave `α` stale, and the first call after them solves it.

`into_trainer` returns the unfitted `Gpr` with the current `θ`. `with_optimizer` on a fitted model only changes a later `refit`. `OnlineGpr::with_optimizer` does the same. `refit` on `FittedGpr<O: Optimizer>` searches again from the current `θ` on the stored data. `refit` on `FittedGpr<Fixed>` rebuilds `L` and `α` and does not search. Transforms are not fit again.

`into_online` returns `OnlineGpr`. `insert(x_new, y_new)` appends one point and returns a `PointId`. `delete(id)` removes that point. The last remaining point cannot be deleted (`GprError::InsufficientData`, `min` 2). `PointId` has no public constructor. `into_online` assigns `0 .. n-1`. Later inserts increase and are never reused. `point_ids` is the current list. `InvalidPointId` means the id is not in the model. `into_trainer` on the online model returns `Gpr`. `refit` on `OnlineGpr` follows the same split: an `Optimizer` searches, and `Fixed` rebuilds the factor.

```rust
use gprx::kernel::{KernelSpec, RbfKernel};
use gprx::{GaussianLikelihood, Gpr, OnlineGpr, PointId};

fn main() -> Result<(), gprx::GprError> {
    let fitted = Gpr::new(
        KernelSpec::from(RbfKernel::new(1.0)?),
        GaussianLikelihood::new(0.1)?,
    )
    .fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0])?;
    let mut online: OnlineGpr = fitted.into_online()?;
    let id: PointId = online.insert(&[0.5], 0.25)?;
    assert!(online.point_ids().contains(&id));
    let _ = online.alpha()?;
    online.delete(id)?;
    let trainer = online.into_trainer();
    let _ = trainer.kernel();
    Ok(())
}
```

`save(dir)` writes `config.json` and `model.safetensors` without the factor. `save_with_factor` also writes the factor and `α`: column-major `L` from a `FittedGpr`, the packed LDLT factor from an `OnlineGpr`.

Before `fit`, on `Gpr`:

| Method | Effect |
| --- | --- |
| `with_optimizer(solver)` | Replaces `Lbfgs`. `Fixed` removes `fit` and adds `factor`. |
| `with_precision::<P>()` | `DoublePrecision` (default), `SinglePrecision`, or `MixedPrecision`. |
| `with_math(KernelExp::FastApprox)` | Polynomial kernel `exp`. Default is `KernelExp::Accurate`. |
| `with_input_transform(map)` | Default is identity. |
| `with_target_transform(map)` | Default is identity. The prior mean is zero, so use `StandardizeTarget::new()` when `y` is not centred near zero. |
| `with_jitter_policy(policy)` | Default is `JitterPolicy::fixed(0.0)`. |
| `with_prefer_speed` | `DistanceCachePolicy::Cached` and `CholeskyBuffer::Retain` (the defaults). Replaces both. |
| `with_prefer_memory` | `DistanceCachePolicy::Uncached` and `CholeskyBuffer::Reuse`. Replaces both. |

`with_prefer_speed` and `with_prefer_memory` exist on `Gpr` and set the distance cache and the Cholesky buffer together. `DistanceCachePolicy` (`Cached` is the default, `Uncached` recomputes) and `KernelExp` are non-exhaustive. `CholeskyBuffer` is `Retain` or `Reuse`. `Sgpr` and `Svgp` take `with_optimizer`, `with_precision`, `with_math`, `with_jitter_policy`, `with_input_transform`, and `with_target_transform`. `Sgpr` also takes `with_inducing`. Their `K_mm` jitter starts at `JitterPolicy::adaptive(1e-8, 10.0, 5, 1e-3)`. Before `fit`, they read `kernel`, `likelihood`, `math`, `jitter_policy`, `num_params`, and `get_params`. `set_params` updates `θ`.

```rust
use gprx::kernel::{KernelSpec, RbfKernel};
use gprx::transform::StandardizeTarget;
use gprx::{
    CholeskyBuffer, DistanceCachePolicy, GaussianLikelihood, Gpr, JitterPolicy, KernelExp,
    Prediction, SinglePrecision,
};

fn main() -> Result<(), gprx::GprError> {
    let gpr = Gpr::new(
        KernelSpec::from(RbfKernel::new(1.0)?),
        GaussianLikelihood::new(0.1)?,
    )
    .with_precision::<SinglePrecision>()
    .with_math(KernelExp::FastApprox)
    .with_input_transform(gprx::transform::IdentityInput)
    .with_target_transform(StandardizeTarget::new())
    .with_jitter_policy(JitterPolicy::fixed(0.0)?)
    .with_prefer_memory()
    .with_prefer_speed();
    assert_eq!(gpr.distance_cache_policy(), DistanceCachePolicy::Cached);
    assert_eq!(gpr.cholesky_buffer(), CholeskyBuffer::Retain);
    let fitted = gpr.fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0]).map_err(|(_, e)| e)?;
    // A single-precision model predicts in `f32`.
    let pred: Prediction<f32> = fitted.predict(&[0.5], 1, 1)?;
    let _ = pred.mean[0];
    Ok(())
}
```

### Sparse: `Sgpr`, `FittedSgpr`, `OnlineSgpr`

`Sgpr::new` with a `KernelSpec` is `Sgpr<Lbfgs, FixedInducing, DoublePrecision, KernelSpec>`. Inducing points stay where you put them. `with_inducing(FreeInducing)` also searches `Z`.

```rust
use gprx::kernel::{KernelSpec, RbfKernel};
use gprx::{
    Fixed, FittedSgpr, FixedInducing, FreeInducing, GaussianLikelihood, InducingId, OnlineSgpr,
    Sgpr,
};

fn main() -> Result<(), gprx::GprError> {
    let x = &[0.0, 1.0, 2.0, 3.0];
    let y = &[0.0, 1.0, 0.5, 0.25];
    let z = &[0.5, 2.5];
    let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    let likelihood = GaussianLikelihood::new(0.1)?;

    let fitted: FittedSgpr = Sgpr::new(kernel.clone(), likelihood.clone())
        .with_inducing(FixedInducing)
        .fit(x, 4, 1, y, z, 2)
        .map_err(|(_, e)| e)?;
    let _moved = Sgpr::new(kernel.clone(), likelihood.clone())
        .with_inducing(FreeInducing)
        .fit(x, 4, 1, y, z, 2)
        .map_err(|(_, e)| e)?;
    let frozen = Sgpr::new(kernel, likelihood)
        .with_optimizer(Fixed)
        .factor(x, 4, 1, y, z, 2)
        .map_err(|(_, e)| e)?;
    assert_eq!(fitted.neg_log_marginal_likelihood()?.is_finite(), true);
    let _ = frozen.m();

    // Online: add a point and an inducing point, then return to a fitted model.
    let mut online: OnlineSgpr = fitted.into_online();
    let point = online.insert(&[1.5], 0.75)?;
    let inducing: InducingId = online.insert_inducing(&[1.5])?;
    online.delete_inducing(inducing)?;
    online.delete(point)?;
    let back = online.into_fitted();
    assert_eq!(back.m(), 2);
    Ok(())
}
```

`fit` and `factor` take `(x, n_rows, n_cols, y, z, n_inducing)`. `z` is column-major with `n_inducing` points and the same `n_cols`. `factor` does not move `θ` or `Z`. Free inducing appends column-major `Z` after kernel `θ` and likelihood `θ`. Each coordinate is limited to an interval built from the training inputs after the input transform: their range, opened by 10% of it and at least `0.1` on each side, with the lower end moved down to `0` when it is above `0`. Matérn `ν = 1/2` has no coordinate derivative for free `Z` (`GprError::CoordGradientUnsupported`).

`FittedSgpr` has the same predict, covariance, sample, leave-one-out, and objective methods as `FittedGpr` (`neg_log_marginal_likelihood`, `num_params`, `get_params`, `set_params`, `value_and_gradient_into`, `hessian_into`). It reads `n`, `m`, `d`, `kernel`, `likelihood`, `math`, `jitter_policy`, `x`, `y`, and `z` (inducing inputs, original coordinates). `into_online` returns `OnlineSgpr` with those readers, those objective methods, `point_ids`, and `inducing_ids`. `insert` / `delete` use `PointId`. `insert_inducing` / `delete_inducing` use `InducingId` (no public constructor, never reused). `InvalidInducingId` means the id is absent. `OnlineSgpr<O: Optimizer>::refit` searches again and keeps `Z` fixed. `OnlineSgpr<Fixed>` has no `refit`. `into_fitted` returns `FittedSgpr<_, FixedInducing, _>`. `save` writes the directory. Sparse models have no `save_with_factor`.

### Minibatch: `Svgp`, `FittedSvgp`

`Svgp::new` is `Svgp<Fixed>`. `factor` builds a whitened variational posterior `q` at the current `θ` and does not search. `with_optimizer(Adam::new()).fit(...)` runs mini-batch Adam on `θ` and `q`. `Adam` does not implement `Optimizer`. `Gpr` and `Sgpr` cannot take it.

```rust
use gprx::kernel::{KernelSpec, RbfKernel};
use gprx::{Adam, FittedSvgp, GaussianLikelihood, Svgp};

fn main() -> Result<(), gprx::GprError> {
    let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    let likelihood = GaussianLikelihood::new(0.1)?;
    let x = &[0.0, 1.0, 2.0, 3.0];
    let y = &[0.0, 1.0, 0.5, 0.25];
    let z = &[0.5, 2.5];
    let factored: FittedSvgp = Svgp::new(kernel.clone(), likelihood.clone())
        .factor(x, 4, 1, y, z, 2)
        .map_err(|(_, e)| e)?;
    let trained = Svgp::new(kernel, likelihood)
        .with_optimizer(Adam::new())
        .fit(x, 4, 1, y, z, 2)
        .map_err(|(_, e)| e)?;
    let _ = (factored.neg_elbo()?, trained.n());
    Ok(())
}
```

`FittedSvgp::neg_elbo` is the evidence lower bound. `num_params`, `get_params`, `set_params`, and `value_and_gradient_into` are the full-data objective. There is no `hessian_into`. The Adam loop scales the data term by `n / batch` and leaves the KL whole. Predict, covariance, and sample match the other families: `predict`, `predict_with`, `predict_into`, `predict_with_into`, `predict_covariance`, `predict_covariance_with`, `sample`, `sample_with`. Readers are `n`, `m`, `d`, `kernel`, `likelihood`, `math`, `jitter_policy`, `x`, `y`, and `z`. There is no leave-one-out, no online type, and no `refit`. `save` writes the directory.

`Adam::new` is learning rate `1e-3`, `β1 = 0.9`, `β2 = 0.999`, `ε = 1e-8`, batch 32, 100 epochs, seed 0. Setters: `with_learning_rate`, `with_beta1`, `with_beta2`, `with_epsilon`, `with_batch_size` (`NonZeroUsize`), `with_epochs` (`NonZeroU64`), `with_seed`. The rate, betas, and epsilon return `Result`.

```rust
use std::num::{NonZeroU64, NonZeroUsize};

use gprx::kernel::{KernelSpec, RbfKernel};
use gprx::{Adam, GaussianLikelihood, Svgp};

fn main() -> Result<(), gprx::GprError> {
    let adam = Adam::new()
        .with_learning_rate(1e-3)?
        .with_beta1(0.9)?
        .with_beta2(0.999)?
        .with_epsilon(1e-8)?
        .with_batch_size(NonZeroUsize::MIN)
        .with_epochs(NonZeroU64::MIN)
        .with_seed(0);
    let _ = Svgp::new(
        KernelSpec::from(RbfKernel::new(1.0)?),
        GaussianLikelihood::new(0.1)?,
    )
    .with_optimizer(adam);
    Ok(())
}
```

### Kernels

Build a `KernelSpec` with `KernelSpec::from(leaf)` or `KernelSpec::custom(term)`. `+` is a sum. `*` is a product. `*` binds tighter than `+`, so `c * k + k2` is a scaled kernel plus another kernel. `num_params`, `get_params`, and `set_params` are the flattened log-`θ` in depth-first leaf order. `parameter_bindings` returns `Vec<ParameterBinding>`. The fields are `index`, `leaf_id`, and `local_index`. `compile` makes a `CompiledKernel<f64>`. `compile_as::<T>()` picks the storage scalar (`KernelScalar`, implemented for `f32` and `f64` only). The trait methods are `from_f64`, `to_f64`, `exp`, `ln`, `sqrt`, `abs`, `is_finite`, `powf`, `sin`, `cos`, `max`, and `min`. `CompiledKernel` has the same `num_params`, `get_params`, and `set_params`.

`KernelSpec` and `CompiledKernel` are non-exhaustive. Match either on `Rbf`, `RbfArd`, `Matern`, `MaternArd`, `Periodic`, `RationalQuadratic`, `RationalQuadraticArd`, `Constant`, `Linear`, `White`, `Custom`, `Sum`, or `Product`. On `KernelSpec`, `Sum` and `Product` are a pair of boxes. On `CompiledKernel`, they are flattened vectors.

```rust
use gprx::kernel::{ConstantKernel, KernelSpec, RbfKernel};

fn main() -> Result<(), gprx::GprError> {
    let scaled = KernelSpec::from(ConstantKernel::new(1.5)?)
        * KernelSpec::from(RbfKernel::new(1.0)?);
    let kernel = scaled + KernelSpec::from(RbfKernel::new(2.0)?);
    assert_eq!(kernel.num_params(), 3);
    Ok(())
}
```

Each built-in leaf stores positive parameters as `log`. `new` takes user units (`ℓ`, variance, period, `α`). `from_log_*` takes the optimizer coordinate. `bounds` / `with_bounds` use an open `Interval` (default `(1e-5, 1e5)`, `Interval::DEFAULT_POSITIVE`). `with_bounds` returns `IntervalError` when the current value is outside.

| Leaf | Construct | Parameters |
| --- | --- | --- |
| `RbfKernel` | `new(ℓ)` | isotropic lengthscale |
| `RbfArdKernel` | `new(&[ℓ_d])` | one lengthscale per feature (`ArdLengthscales`) |
| `MaternKernel` | `new(ℓ, MaternNu)` | `MaternNu::Half`, `ThreeHalves`, `FiveHalves` (`value` is 0.5, 1.5, 2.5). `ν` is not optimized |
| `MaternArdKernel` | `new(&[ℓ_d], nu)` | ARD lengthscales, fixed `ν` |
| `PeriodicKernel` | `new(ℓ, period)` | lengthscale and period |
| `RationalQuadraticKernel` | `new(ℓ, alpha)` | lengthscale and `α` |
| `RationalQuadraticArdKernel` | `new(&[ℓ_d], alpha)` | ARD lengthscales, then `α` |
| `ConstantKernel` | `new(c)` | signal variance. Multiply by an RBF for `c * k` |
| `LinearKernel` | `new(variance)` | `σ² xᵀ x'` |
| `WhiteKernel` | `new(variance)` | diagonal nugget |

Observation noise belongs in `GaussianLikelihood`. `WhiteKernel` is an extra kernel term. A large likelihood and a large white term count the noise twice.

Isotropic leaves read `lengthscale` and `log_lengthscale`. Matérn also reads `nu`. Periodic reads `period` and `log_period`. Rational quadratic reads `alpha` and `log_alpha`. Linear and white read `variance` and `log_variance`. Constant reads `constant` and `log_constant`. Constructors from the optimizer coordinate are `from_log_lengthscale`, `from_log_lengthscales`, `from_log` (periodic: log lengthscale and log period; rational quadratic: log lengthscale and log `α`), `from_log_variance`, and `from_log_constant`. A single `bounds` covers the isotropic RBF and Matérn leaves, constant, linear, and white; the ARD leaves bound their lengthscales through `ArdLengthscales`. Periodic uses `lengthscale_bounds` and `period_bounds`; `with_bounds` takes both intervals. Rational quadratic uses `lengthscale_bounds` and `alpha_bounds`; `with_bounds` takes both. `lengthscale(dim)` on an ARD leaf returns one `ℓ_d`. `log_lengthscales` is the stored vector. `lengthscales()` returns `ArdLengthscales` (`new`, `from_log_lengthscales`, `with_bounds`, `lengthscale(dim)`, `log_lengthscales`, `num_params`, `get_params`, `set_params`). ARD `with_bounds` takes one interval for every `ℓ_d`. Rational-quadratic ARD `with_bounds` takes a lengthscale interval and an `α` interval.

A leaf and a `CompiledKernel` evaluate on `faer` matrices (`MatRef` / `MatMut`; add `faer = "0.24"` to call them) with `apply`, `apply_cross`, `grad`, and `hess`. Every leaf but `LinearKernel` reads squared distances there and writes its diagonal with `fill_diag`; `LinearKernel` reads coordinates in those four methods, and its diagonal is `fill_diag_points`. `ConstantKernel` and `WhiteKernel` also have `apply_points`, `apply_cross_points`, `grad_points`, and `hess_points`, and `RbfKernel`, `RbfArdKernel`, the isotropic `MaternKernel`, and `WhiteKernel` have `grad_wrt_coord_dim`. `Triangle::Lower` (Cholesky), `Upper`, or `Full` selects which entries are written. A leaf evaluates `exp` with `Accurate`. `CompiledKernel::apply::<M>` and its other evaluating methods (not `fill_diag` / `fill_diag_points`) take the `KernelMath` type `M`: `Accurate` (libm / SIMD `exp`) or `FastApprox` (degree-7 polynomial), plus a scratch matrix of the output's shape. `FastApprox` on `f64` stays within a relative `2^{-23}` of `f64::exp`. Hyperparameter `exp(θ)` does not use this choice.

`KernelTerm<T>` is the trait for a leaf of your own that reads squared distances: `num_params`, `get_params`, `set_params`, `bounds_into`, `apply`, `apply_cross`, `fill_diag`, `grad`, `hess`, `hess_points`, and the derivative methods a sparse model needs (`grad_cross` / `hess_cross`, and `grad_wrt_sq_dist`, `hess_wrt_sq_dist`, `grad_wrt_sq_dist_theta` for `FreeInducing`). `persist_id` and `persist_state` save it (they return a `serde_json::Value`, so add `serde_json = "1"`). Implement it once for every `T: KernelScalar`: a model needs it at `f64` and `f32`, `Clone`, `Debug`, `Send`, `Sync`, and `'static`. `CustomKernel::new(term)` boxes it, and `KernelSpec::from(custom)` or `KernelSpec::custom(term)` inserts it. A custom leaf that omits a derivative a sparse model needs returns `CoordGradientUnsupported`. The save-and-load example below implements one.

```rust
use gprx::kernel::{
    ArdLengthscales, CompiledKernel, ConstantKernel, KernelSpec, LinearKernel, MaternArdKernel,
    MaternKernel, MaternNu, PeriodicKernel, RationalQuadraticArdKernel, RationalQuadraticKernel,
    RbfArdKernel, RbfKernel, Triangle, WhiteKernel,
};
use gprx::Interval;

fn main() -> Result<(), gprx::GprError> {
    let rbf = RbfKernel::from_log_lengthscale(0.0)?;
    let _ = (rbf.lengthscale(), rbf.log_lengthscale(), rbf.bounds());
    let rbf = rbf.with_bounds(Interval::new(1e-3, 1e2)?)?;
    let ard = RbfArdKernel::new(&[0.5, 2.0])?;
    let _ = ard.lengthscale(0)?;
    let scales: &ArdLengthscales = ard.lengthscales();
    let _ = scales.log_lengthscales();
    let matern = MaternKernel::new(1.0, MaternNu::FiveHalves)?;
    let _ = matern.nu().value();
    let _ = MaternArdKernel::new(&[1.0, 1.0], MaternNu::Half)?;
    let periodic = PeriodicKernel::new(1.0, 0.5)?;
    let _ = (periodic.period(), periodic.log_period());
    let rq = RationalQuadraticKernel::new(1.0, 1.5)?;
    let _ = (rq.alpha(), rq.log_alpha());
    let _ = RationalQuadraticArdKernel::new(&[1.0], 1.5)?;
    let constant = ConstantKernel::new(1.5)?;
    let _ = (constant.constant(), constant.log_constant());
    let linear = LinearKernel::new(0.2)?;
    let _ = (linear.variance(), linear.log_variance());
    let white = WhiteKernel::new(0.01)?;
    let _ = white.log_variance();
    let spec = KernelSpec::from(rbf) + KernelSpec::from(white);
    let bindings = spec.parameter_bindings();
    let _ = (bindings[0].index, bindings[0].leaf_id, bindings[0].local_index);
    let compiled: CompiledKernel = spec.compile();
    let compiled_f32 = spec.compile_as::<f32>();
    let mut theta = vec![0.0; compiled.num_params()];
    compiled.get_params(&mut theta)?;
    let _ = (compiled_f32.num_params(), Triangle::Lower);
    Ok(())
}
```

Evaluating a leaf and a compiled tree on squared distances, the parameter constructors, and a function generic over the scalar:

```rust
use faer::Mat;
use gprx::kernel::{
    ConstantKernel, KernelScalar, KernelSpec, LinearKernel, ParameterBinding, PeriodicKernel,
    RationalQuadraticKernel, RbfKernel, Triangle, WhiteKernel,
};
use gprx::{Accurate, FastApprox, KernelMath};

/// `exp(−d² / 2)` at `f32` or `f64`.
fn unit_rbf<T: KernelScalar>(d2: f64) -> f64 {
    (T::from_f64(-0.5) * T::from_f64(d2)).exp().to_f64()
}

/// The `(0, 1)` entry of `k` for the `KernelMath` mode `M`.
fn off_diagonal<M: KernelMath>(spec: &KernelSpec, d2: &Mat<f64>) -> Result<f64, gprx::GprError> {
    let mut k = Mat::<f64>::zeros(2, 2);
    let mut scratch = Mat::<f64>::zeros(2, 2);
    spec.compile()
        .apply::<M>(d2.as_ref(), k.as_mut(), Triangle::Full, scratch.as_mut())?;
    Ok(k[(0, 1)])
}

fn main() -> Result<(), gprx::GprError> {
    // Two points at 0 and 1: d² is [[0, 1], [1, 0]].
    let d2 = Mat::from_fn(2, 2, |i, j| (i as f64 - j as f64).powi(2));
    let rbf = RbfKernel::new(1.0)?;
    let mut k = Mat::<f64>::zeros(2, 2);
    rbf.apply(d2.as_ref(), k.as_mut(), Triangle::Lower)?;
    assert!((k[(1, 0)] - unit_rbf::<f64>(1.0)).abs() < 1e-12);
    assert!((unit_rbf::<f32>(1.0) - unit_rbf::<f64>(1.0)).abs() < 1e-6);
    rbf.apply_cross(d2.as_ref(), k.as_mut())?;
    let mut diag = [0.0; 2];
    rbf.fill_diag(&mut diag);
    rbf.grad(d2.as_ref(), k.as_mut(), 0, Triangle::Full)?;
    rbf.hess(d2.as_ref(), k.as_mut(), 0, 0, Triangle::Full)?;
    let x = Mat::from_fn(2, 1, |i, _| i as f64);
    LinearKernel::from_log_variance(0.0)?.fill_diag_points(x.as_ref(), &mut diag)?;

    let _ = (WhiteKernel::from_log_variance(-2.0)?, ConstantKernel::from_log_constant(0.0)?);
    let periodic = PeriodicKernel::from_log(0.0, 0.5_f64.ln())?;
    let _ = (periodic.lengthscale_bounds(), periodic.period_bounds());
    let rq = RationalQuadraticKernel::from_log(0.0, 0.0)?;
    let _ = (rq.lengthscale_bounds(), rq.alpha_bounds());

    let spec = KernelSpec::from(rbf);
    let bindings: Vec<ParameterBinding> = spec.parameter_bindings();
    assert_eq!(bindings.len(), 1);
    let accurate = off_diagonal::<Accurate>(&spec, &d2)?;
    let fast = off_diagonal::<FastApprox>(&spec, &d2)?;
    assert!((accurate - fast).abs() < 1e-6);
    Ok(())
}
```

### Supplied distances (`ScalarDistance`, `ArdDistance`)

A kernel leaf can read squared distances you supply instead of coordinates: a geodesic or graph distance, or a distance from another program. `ScalarDistance::new()` is a slot of one `d²` per pair; `kernel(leaf)` takes an `RbfKernel`, `MaternKernel`, `PeriodicKernel`, `RationalQuadraticKernel`, or a leaf of your own that implements `KernelTerm` at `f64` and `f32` and is `Clone + Debug + Send + Sync + 'static` (the `ScalarDistanceLeaf` bound). `ArdDistance::from_leaf(leaf)` takes an `RbfArdKernel`, `MaternArdKernel`, or `RationalQuadraticArdKernel` (the `ArdDistanceLeaf` bound) and returns a slot of `d` blocks, one `(Δ_k)²` per dimension, where `d` is the leaf's lengthscale count, with the leaf on it. A further leaf on the same slot comes from `kernel(leaf)`, which returns `DimensionMismatch` when its lengthscale count is not `d`. Every leaf of one slot reads the same supply. `ConstantKernel`, `WhiteKernel`, and `LinearKernel` stay `KernelSpec` leaves.

The result is a `DistanceKernel<C>`, a separate type from `KernelSpec`. `C` is `DistanceOnly` (no coordinates) or `WithPoints` (the kernel also has coordinate leaves). `DistanceKernel + DistanceKernel` and `*` join the markers (`JoinPoints`). `DistanceKernel` with a `ConstantKernel` or `WhiteKernel`, in either order, keeps `C`. `DistanceKernel` with a `KernelSpec` on its right is `WithPoints`. For a coordinate term on the left, write `DistanceKernel::from(spec)` (a `DistanceKernel<WithPoints>` with no slot): `KernelSpec + DistanceKernel` is not implemented, so `KernelSpec + leaf.into()` still infers a coordinate sum. `num_params`, `get_params`, `set_params`, and `parameter_bindings` match `KernelSpec`. `slots()` returns the `DistanceSlot`s in depth-first order of first use. `KernelSpec<S>` and `CompiledKernel<T, S>` take a sealed `Supply` kind `S` as their last type parameter. The default, `NoSupply`, is a coordinate tree: it has no value, so `KernelSpec` and `CompiledKernel<T>` are the coordinate types, and only a `DistanceKernel` holds the other kind.

`Gpr::new` takes any `ModelKernel` (`KernelSpec` or `DistanceKernel<C>`). `with_input_transform` is on the `PointKernel` models only (`KernelSpec` and `DistanceKernel<WithPoints>`). Each slot gets one `DistanceSource` per call:

| Source | Copy |
| --- | --- |
| `from_vec(d2)` / `from_vecs(blocks)` | moved into the call. An `f64` model keeps the tables as its store, so the fit copies nothing, and an ARD slot then holds `d · n²` values instead of its packed `d · n(n+1)/2` (an ARD table that `tidy` repairs is packed instead). An `f32` model casts them into a buffer of its own, packed for ARD; a `MixedPrecision` model keeps the `f64` tables and adds that cast |
| `from_slice(d2)` / `from_slices(blocks)` | copied at the call, then handled as `from_vec` / `from_vecs`: an `f64` model keeps the copy (`d · n²` for an ARD slot). `borrow` fits without that copy |
| `borrow(d2)` / `borrow(blocks)` | read in place by the `predict` of an `f64` model, cast once by an `f32` model, and copied by `fit` (an ARD square straight into its packed triangles) |
| `fill(&filler)` | `DistanceFill::fill_column(col, rows, out)` writes `d²(i, col)` for each row `i` of `rows`; an ARD fill writes its `d` runs one after another. A square asks only for the rows `col..n` of each column |

Tables are column-major `dist[i + j * n_rows]`. Every value must be finite and non-negative, and a training square (and a query square) must have a zero diagonal and be symmetric, exactly. Otherwise the call returns `InvalidDistance { slot, dim, pair, reason }` at the first value that is not: the slot's place in `slots()`, the ARD dimension, and the `(row, col)` in the block passed. A table that rounding leaves slightly off (one from `‖a‖² + ‖b‖² − 2a·b`) is accepted when its source asks for it: `source.tidy(rel_tol)` sets a negative value or a diagonal within `rel_tol` times the table's largest value to `0.0` and a mirror pair within it to its mean, and refuses anything past it (a borrowed table that the repair changes is copied, not written). A slot with no source, two sources, or a source of a slot the kernel does not have is `DistanceSlot { kind, slot }` (`SlotErrorKind::Missing`, `Duplicate`, `NotRead`); a loaded model reads new slots, so a slot from before the save is `NotRead`. A training fill is called once per fit, whatever the `DistanceCachePolicy`: the model keeps the squares it wrote. A `MixedPrecision` model keeps the training `d²` in `f64` next to its `f32` copy, so the `f64` refinement reads the values the caller gave. `predict_into` binds the sources on buffers the model keeps, so a call allocates nothing once a call of the same shape has run.

| Model | `fit` / `factor` | `predict` family | Covariance and `sample` |
| --- | --- | --- | --- |
| `Gpr`, `DistanceOnly` | `(sources, n, y)` | `(sources, q)` | `(cross, square, q)` |
| `Gpr`, `WithPoints` | `(sources, n, x, n_cols, y)` | `(sources, xs, q, n_cols)` | `(cross, square, xs, q, n_cols)` |
| `Sgpr` / `Svgp`, `DistanceOnly` | `(sources, n, y, inducing)` | `(sources, q)` | `(cross, square, q)` |
| `Sgpr` / `Svgp`, `WithPoints` | `(sources, n, x, n_cols, y, inducing)` | `(sources, xs, q, n_cols)` | `(cross, square, xs, q, n_cols)` |

`cross` is the `n × q` block from the training points to the queries and `square` the `q × q` block between the queries. On a distance model, `to_kernel()` returns a copy of the `DistanceKernel` and `slots()` its slots; `d` and `x` exist only for `WithPoints`. `into_online` converts a distance model too. Its `insert` takes, per slot, the `n × 1` column of squared distances from the current points, in `point_ids` order, to the new point (`d` such columns for an ARD slot), from any source above: `insert(sources, y_new)` for `DistanceOnly`, `insert(sources, x_new, y_new)` for `WithPoints`. The column is checked as a training square is (or repaired by `tidy`) and kept, and `delete(id)` removes the point from the kept squares in place. The online model predicts with the same arguments as the fitted one. Neither allocates once the model has grown once (a scalar square grows by a quarter, an ARD slot by doubling), except that a large delete on a Rayon pool of more than one worker queues one job beside the factor's update.

A sparse model on supplied distances names its inducing points by training index: `inducing` lists them, without repeats, and each source of `fit` / `factor` is the `n × m` block from the training points to them (column `a` is the training point `inducing[a]`); `K_mm` reads their rows of it. The model keeps the blocks as they were handed over (a moved table without a copy, a borrowed one copied once). A prediction's `cross` is the `m × q` block from the inducing points, in `inducing()` order (on `OnlineSgpr`, `inducing_points()`: their `PointId`s, which a delete does not shift), to the queries; `square` is as above. `FreeInducing` needs coordinates to move, so a distance model keeps its inducing points. `into_online` keeps the blocks: `insert` takes, per slot, the `m × 1` squared distances from the inducing points to the new point; `insert_inducing` names a training point by its `PointId` and takes its `n × 1` squared distances to the training points, checked against the stored pairs; an inducing point cannot be `delete`d until `delete_inducing` removes it. A factor reads the `d · n · m` supplied values a coordinate model computes from `n · d` coordinates, so it costs more for an ARD slot than the coordinate factor; the gradient and the predictions cost no more ([design §5.6](https://github.com/YUKIKEDA/gprx/blob/main/docs/design.md#56-supplied-squared-distances-requirements-470)).

```rust
use gprx::kernel::{
    ArdDistance, ArdDistanceLeaf, ConstantKernel, DistanceFill, DistanceKernel, DistanceOnly,
    DistanceSlot, DistanceSource, JoinPoints, KernelSpec, ModelKernel, NoSupply, PointKernel,
    PointUse, RbfArdKernel, RbfKernel, ScalarDistance, ScalarDistanceLeaf, Supply, WithPoints,
};
use gprx::{Fixed, GaussianLikelihood, Gpr, Sgpr, Svgp};

/// Squared distances of points 0, 1, 2, … on a line.
struct Line;

impl DistanceFill for Line {
    fn fill_column(&self, col: usize, rows: std::ops::Range<usize>, out: &mut [f64]) {
        for (slot, i) in out.iter_mut().zip(rows) {
            *slot = (i as f64 - col as f64).powi(2);
        }
    }
}

fn scalar_leaf(leaf: impl ScalarDistanceLeaf, slot: ScalarDistance) -> DistanceKernel {
    slot.kernel(leaf)
}

fn takes<K: ModelKernel>(_: &K) {}
fn supply_of<S: Supply>(_: &KernelSpec<S>) {}
fn takes_points<K: PointKernel>(_: &K) {}
fn joined<A: JoinPoints<B>, B: PointUse>(_: DistanceKernel<A>, _: DistanceKernel<B>) {}

fn main() -> Result<(), gprx::GprError> {
    // A coordinate tree is `KernelSpec<NoSupply>`, the default.
    let coords: KernelSpec<NoSupply> = KernelSpec::from(RbfKernel::new(1.0)?);
    supply_of(&coords);
    let image = ScalarDistance::new();
    let other = ScalarDistance::new();
    let kernel = scalar_leaf(RbfKernel::new(1.0)?, image) * ConstantKernel::new(0.8)?;
    assert_eq!(kernel.slots(), vec![DistanceSlot::Scalar(image)]);
    let mut theta = vec![0.0; kernel.num_params()];
    kernel.get_params(&mut theta)?;
    let _ = kernel.parameter_bindings();
    takes(&kernel);
    joined(image.kernel(RbfKernel::new(1.0)?), other.kernel(RbfKernel::new(2.0)?));

    // Four points 0, 1, 2, 3: d²[i + j·4] = (i − j)².
    let d2 = vec![0.0, 1.0, 4.0, 9.0, 1.0, 0.0, 1.0, 4.0, 4.0, 1.0, 0.0, 1.0, 9.0, 4.0, 1.0, 0.0];
    let y = [0.0, 1.0, 0.5, 0.25];
    let fitted = Gpr::new(kernel.clone(), GaussianLikelihood::new(0.1)?)
        .with_optimizer(Fixed)
        .factor([image.from_vec(d2.clone())], 4, &y)?;
    // Two queries at 0.5 and 1.5.
    let cross = [0.25, 0.25, 2.25, 6.25, 2.25, 0.25, 0.25, 2.25];
    let square = [0.0, 1.0, 1.0, 0.0];
    let pred = fitted.predict([image.borrow(&cross)], 2)?;
    let cov = fitted.predict_covariance([image.borrow(&cross)], [image.borrow(&square)], 2)?;
    assert_eq!((pred.mean.len(), cov.covariance.len()), (2, 4));
    let by_fill = Gpr::new(kernel.clone(), GaussianLikelihood::new(0.1)?)
        .with_optimizer(Fixed)
        .factor([image.fill(&Line)], 4, &y)?;
    let source: DistanceSource<'_> = image.from_slice(&cross);
    assert_eq!(by_fill.predict([source], 2)?, pred);
    // A table the Gram trick rounded: repaired on request, refused without.
    let rounded = [0.25, 0.25, 2.25, 6.25, 2.25, 0.25, 0.25, -1e-15];
    assert!(fitted.predict([image.borrow(&rounded)], 2).is_err());
    let _ = fitted.predict([image.borrow(&rounded).tidy(1e-12)?], 2)?;

    // ARD: one block per dimension, here two copies of the line. The slot
    // takes its dimensions from its first leaf.
    let (bands, ard) = ArdDistance::from_leaf(RbfArdKernel::new(&[1.0, 2.0])?);
    assert_eq!(bands.dims(), 2);
    let ard: DistanceKernel<DistanceOnly> = ard;
    let blocks: [&[f64]; 2] = [&d2, &d2];
    let fitted = Gpr::new(ard, GaussianLikelihood::new(0.1)?)
        .with_optimizer(Fixed)
        .factor([bands.from_slices(&blocks)], 4, &y)?;
    let cross_blocks: [&[f64]; 2] = [&cross, &cross];
    let _ = fitted.predict([bands.borrow(&cross_blocks)], 2)?;
    let _ = (bands.from_vecs(vec![d2.clone(), d2.clone()]), bands.fill(&Line));

    // A distance leaf times a coordinate leaf.
    let mixed: DistanceKernel<WithPoints> =
        image.kernel(RbfKernel::new(1.0)?) * KernelSpec::from(RbfKernel::new(2.0)?);
    takes_points(&mixed);
    let x = [0.0, 1.0, 2.0, 3.0];
    let fitted = Gpr::new(mixed, GaussianLikelihood::new(0.1)?)
        .with_optimizer(Fixed)
        .factor([image.from_vec(d2)], 4, &x, 1, &y)?;
    assert_eq!((fitted.d(), fitted.x()), (1, &x[..]));
    let _ = fitted.predict([image.borrow(&cross)], &[0.5, 1.5], 2, 1)?;
    let _ = (fitted.to_kernel(), fitted.slots());

    // Sgpr and Svgp: the inducing points are training points 0 and 2, and a
    // source is the 4 × 2 block from the training points to them.
    let train_z = [0.0, 1.0, 4.0, 9.0, 4.0, 1.0, 0.0, 1.0];
    let sgpr = Sgpr::new(kernel.clone(), GaussianLikelihood::new(0.1)?)
        .with_optimizer(Fixed)
        .factor([image.borrow(&train_z)], 4, &y, &[0, 2])
        .map_err(|(_, e)| e)?;
    assert_eq!(sgpr.inducing(), &[0, 2]);
    // The 2 × 2 block from the inducing points to the queries at 0.5 and 1.5.
    let cross_z = [0.25, 2.25, 2.25, 0.25];
    let _ = sgpr.predict([image.borrow(&cross_z)], 2)?;
    // OnlineSgpr: a point at 4 (its squared distances to the inducing
    // points), then training point 1 as a third inducing point (its squared
    // distances to the five training points).
    let mut online = sgpr.clone().into_online();
    online.insert([image.from_vec(vec![16.0, 4.0])], 0.1)?;
    let point = online.point_ids()[1];
    online.insert_inducing(point, [image.from_vec(vec![1.0, 0.0, 1.0, 4.0, 9.0])])?;
    let ids = online.point_ids();
    assert_eq!(online.inducing_points().collect::<Vec<_>>(), [ids[0], ids[2], point]);
    let _ = online.predict([image.from_vec(vec![0.25, 2.25, 0.25])], 1)?;
    let _ = (online.to_kernel(), online.slots());
    online.delete_inducing(online.inducing_ids()[2])?;
    let svgp = Svgp::new(kernel, GaussianLikelihood::new(0.1)?)
        .factor([image.from_vec(train_z.to_vec())], 4, &y, &[0, 2])
        .map_err(|(_, e)| e)?;
    let cov = svgp.predict_covariance([image.borrow(&cross_z)], [image.borrow(&square)], 2)?;
    assert_eq!(cov.covariance.len(), 4);
    Ok(())
}

fn _ard_bound(leaf: impl ArdDistanceLeaf, slot: ArdDistance) -> Result<DistanceKernel, gprx::GprError> {
    slot.kernel(leaf)
}
```

The rest of a distance model's surface, on the same four points: `fit`, the `predict` family, `sample`, the online `insert` / `delete`, an `InvalidDistance` match, an ARD slot in `slots()`, an input transform on a model with points, and save and load of the sparse models.

```rust
use gprx::kernel::{
    ArdDistance, DistanceOnly, DistanceSlot, KernelSpec, RbfArdKernel, RbfKernel, ScalarDistance,
};
use gprx::persist::{LoadedDistanceSgpr, LoadedDistanceSvgp, PersistRegistry};
use gprx::transform::MinMaxInput;
use gprx::{
    Fixed, GaussianLikelihood, Gpr, GprError, PredictOptions, Prediction, Sgpr, Svgp,
    VarianceKind,
};

fn main() -> Result<(), GprError> {
    let image = ScalarDistance::new();
    let kernel = image.kernel(RbfKernel::new(1.0)?);
    // Four points 0, 1, 2, 3 and two queries at 0.5 and 1.5, as above.
    let d2 = vec![0.0, 1.0, 4.0, 9.0, 1.0, 0.0, 1.0, 4.0, 4.0, 1.0, 0.0, 1.0, 9.0, 4.0, 1.0, 0.0];
    let y = [0.0, 1.0, 0.5, 0.25];
    let cross = [0.25, 0.25, 2.25, 6.25, 2.25, 0.25, 0.25, 2.25];
    let square = [0.0, 1.0, 1.0, 0.0];

    // `fit` searches θ (L-BFGS by default) on the supplied training square.
    let trainer = Gpr::new(kernel.clone(), GaussianLikelihood::new(0.1)?);
    assert_eq!(trainer.to_kernel().slots(), vec![DistanceSlot::Scalar(image)]);
    let mut fitted = trainer
        .fit([image.from_vec(d2.clone())], 4, &y)
        .map_err(|(_, e)| e)?;
    let latent = PredictOptions {
        variance_kind: VarianceKind::Latent,
    };
    let _ = fitted.predict_with([image.borrow(&cross)], 2, latent)?;
    let mut out = Prediction::default();
    fitted.predict_into([image.borrow(&cross)], 2, &mut out)?;
    let _ = fitted.predict_covariance_with([image.borrow(&cross)], [image.borrow(&square)], 2, latent)?;
    // Three draws, column-major 2 × 3, from seed 7.
    let draws = fitted.sample([image.borrow(&cross)], [image.borrow(&square)], 2, 3, 7)?;
    assert_eq!(draws.len(), 6);

    // A value that is not a squared distance names its slot and pair.
    let negative = [0.25, 0.25, 2.25, 6.25, 2.25, 0.25, 0.25, -2.25];
    match fitted.predict([image.borrow(&negative)], 2) {
        Err(GprError::InvalidDistance { slot, pair, .. }) => {
            assert_eq!((slot, pair), (Some(0), Some((3, 1))));
        }
        other => panic!("{other:?}"),
    }

    // OnlineGpr: a point at 4 (its squared distances to the live points, in
    // `point_ids` order), then its delete.
    let mut online = fitted.into_online()?;
    let id = online.insert([image.from_vec(vec![16.0, 9.0, 4.0, 1.0])], 0.3)?;
    online.delete(id)?;
    let _ = online.predict([image.borrow(&cross)], 2)?;

    // An ARD kernel lists its slot as `DistanceSlot::Ard`.
    let (bands, ard) = ArdDistance::from_leaf(RbfArdKernel::new(&[1.0, 2.0])?);
    assert_eq!(ard.slots(), vec![DistanceSlot::Ard(bands)]);

    // A kernel with points maps its coordinates with an input transform.
    let mixed = image.kernel(RbfKernel::new(1.0)?) * KernelSpec::from(RbfKernel::new(2.0)?);
    let x = [0.0, 10.0, 20.0, 30.0];
    let _ = Gpr::new(mixed, GaussianLikelihood::new(0.1)?)
        .with_optimizer(Fixed)
        .with_input_transform(MinMaxInput::new())
        .factor([image.from_vec(d2)], 4, &x, 1, &y)?;

    // Svgp and OnlineSgpr save; a loaded model binds its new slots.
    let train_z = vec![0.0, 1.0, 4.0, 9.0, 4.0, 1.0, 0.0, 1.0];
    let cross_z = [0.25, 2.25, 2.25, 0.25];
    let svgp = Svgp::new(kernel.clone(), GaussianLikelihood::new(0.1)?)
        .factor([image.from_vec(train_z.clone())], 4, &y, &[0, 2])
        .map_err(|(_, e)| e)?;
    assert_eq!((svgp.inducing(), svgp.slots()), (&[0, 2][..], vec![DistanceSlot::Scalar(image)]));
    let _ = svgp.to_kernel();
    let dir = std::env::temp_dir().join(format!("gprx-readme-distance-svgp-en-{}", std::process::id()));
    svgp.save(&dir)?;
    let loaded = LoadedDistanceSvgp::<DistanceOnly>::load(&dir, &PersistRegistry::new())?;
    assert_eq!((loaded.n(), loaded.inducing()), (4, &[0, 2][..]));
    let _ = loaded.to_kernel();
    let [DistanceSlot::Scalar(slot)] = loaded.slots()[..] else {
        return Err(GprError::EmptyInput);
    };
    let _ = loaded.predict([slot.borrow(&cross_z)], 2)?;
    let _ = std::fs::remove_dir_all(&dir);
    let online = Sgpr::new(kernel, GaussianLikelihood::new(0.1)?)
        .with_optimizer(Fixed)
        .factor([image.from_vec(train_z)], 4, &y, &[0, 2])
        .map_err(|(_, e)| e)?
        .into_online();
    let dir = std::env::temp_dir().join(format!("gprx-readme-distance-online-sgpr-en-{}", std::process::id()));
    online.save(&dir)?;
    let _ = LoadedDistanceSgpr::<DistanceOnly>::load(&dir, &PersistRegistry::new())?;
    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}
```

### Likelihood

`GaussianLikelihood::new(noise_variance)` stores `σn²` as a log parameter. `from_log_noise_variance`, `noise_variance`, `log_noise_variance`, `bounds`, `with_bounds`, `num_params`, `get_params`, `set_params`. `add_noise_diag` adds `σn²` to a kernel diagonal. `noise_grad_diag` is the derivative of that diagonal with respect to one parameter. `InvalidNoiseVariance` is a noise variance that is not finite or not positive. A positive one outside the noise interval (default `(1e-5, 1e5)`) is `InvalidInterval`.

```rust
use gprx::{GaussianLikelihood, Interval};

fn main() -> Result<(), gprx::GprError> {
    let like = GaussianLikelihood::from_log_noise_variance(0.1_f64.ln())?;
    let _ = (like.noise_variance(), like.log_noise_variance(), like.bounds());
    let like = like.with_bounds(Interval::new(1e-4, 10.0)?)?;
    let mut diag = [1.0, 1.0];
    like.add_noise_diag(&mut diag);
    let mut grad = [0.0, 0.0];
    like.noise_grad_diag(&mut grad, 0)?;
    let mut theta = [0.0; 1];
    like.get_params(&mut theta)?;
    let _ = like.num_params();
    Ok(())
}
```

### Transforms (`gprx::transform`)

Pass an unfitted map to `with_input_transform` or `with_target_transform` before `fit`. The model fits the map on the training data. Defaults are `IdentityInput` and `IdentityTarget`.

| Type | Role |
| --- | --- |
| `IdentityInput`, `IdentityTarget` | no change. `fit` returns the same map |
| `StandardizeInput` | per-feature mean and standard deviation. `FittedStandardizeInput::mean` and `std` |
| `StandardizeTarget` | one mean and standard deviation for `y`. `FittedStandardizeTarget::mean` and `std` |
| `MinMaxInput` | per-feature map into a range. `new` is `[0, 1]`. `with_feature_range(lo, hi)`. `feature_range` reads it. Fits to `FittedMinMaxInput` (`min`, `max`, `feature_range`) |
| `MinMaxTarget` | the same for `y`. Fits to `FittedMinMaxTarget` |
| `Pipeline` | `Pipeline::new().then(step)`. `len`, `is_empty`. Fits to `FittedPipeline` |
| `TargetPipeline` | the same for `y`, fits to `FittedTargetPipeline` |
| `ColumnwiseInput` | `new().then(map)` assigns the next feature. Fits to `FittedColumnwiseInput` |

Concrete maps also have an inherent `fit`. Input maps take `(x, n_rows, n_cols)`. Target maps take `y`. The trait methods `UnfittedTransform::fit` and `UnfittedTarget::fit` take `Box<Self>`. A caller-defined map implements `clone_box` and `as_any`. Saving it also implements `persist_id` and `persist_state`. The save-and-load example below defines one input map and one target map.

`Transform` is the fitted input trait: `apply` and `inverse_apply`, in place, column-major. `TargetTransform` is the fitted target trait: `transform`, `inverse_transform_mean`, `inverse_transform_variance`, and `inverse_transform_covariance` (the default scales every entry the same way as the variance).

```rust
use gprx::transform::{
    ColumnwiseInput, FittedColumnwiseInput, FittedMinMaxInput, FittedMinMaxTarget, FittedPipeline,
    FittedStandardizeInput, FittedStandardizeTarget, FittedTargetPipeline, IdentityInput,
    IdentityTarget, MinMaxInput, MinMaxTarget, Pipeline, StandardizeInput, StandardizeTarget,
    TargetPipeline, TargetTransform, Transform,
};

fn main() -> Result<(), gprx::GprError> {
    let x = [0.0, 2.0, 10.0, 30.0];
    let input: FittedStandardizeInput = StandardizeInput::new().fit(&x, 2, 2)?;
    let _ = (input.mean(), input.std());
    let mut applied = x;
    input.apply(&mut applied, 2, 2)?;
    input.inverse_apply(&mut applied, 2, 2)?;
    let y = [0.0, 1.0, 3.0];
    let target: FittedStandardizeTarget = StandardizeTarget::new().fit(&y)?;
    let _ = (target.mean(), target.std());
    let mut mean = [0.0];
    target.inverse_transform_mean(&mut mean)?;
    let mut variance = [1.0];
    target.inverse_transform_variance(&mut variance)?;
    let mut cov = [1.0];
    target.inverse_transform_covariance(&mut cov)?;
    let unfitted = MinMaxInput::with_feature_range(0.0, 1.0)?;
    let _ = unfitted.feature_range();
    let minmax: FittedMinMaxInput = unfitted.fit(&x, 2, 2)?;
    let _ = (minmax.feature_range(), minmax.min(), minmax.max());
    let target_range: FittedMinMaxTarget = MinMaxTarget::new().fit(&y)?;
    let _ = (target_range.min(), target_range.max(), target_range.feature_range());
    let pipeline = Pipeline::new()
        .then(IdentityInput)
        .then(StandardizeInput::new());
    assert!(!pipeline.is_empty());
    let _ = pipeline.len();
    let fitted_pipeline: FittedPipeline = pipeline.fit(&x, 2, 2)?;
    assert_eq!(fitted_pipeline.len(), 2);
    let targets = TargetPipeline::new()
        .then(IdentityTarget)
        .then(StandardizeTarget::new());
    let fitted_targets: FittedTargetPipeline = targets.fit(&y)?;
    assert!(!fitted_targets.is_empty());
    let columns = ColumnwiseInput::new()
        .then(StandardizeInput::new())
        .then(IdentityInput);
    let fitted_columns: FittedColumnwiseInput = columns.fit(&x, 2, 2)?;
    assert_eq!(fitted_columns.len(), 2);
    Ok(())
}
```

### Optimizers

One optimizer is the type parameter. `Gpr::new` is `Lbfgs`. `with_optimizer` replaces it.

| Type | Search | Setters |
| --- | --- | --- |
| `Lbfgs` | L-BFGS, More–Thuente line search. Needs `Differentiable` | `new`: 100 iterations, tolerance `sqrt(ε)`, history 10. `with_max_iterations`, `with_tolerance`, `with_history_size` (`NonZeroUsize`), `with_restarts(n, seed)` |
| `NelderMead` | derivative-free. Needs `Objective` | `with_max_iterations`, `with_tolerance`, `with_restarts` |
| `TrustRegion` | uses the Hessian. Needs `TwiceDifferentiable` | `with_max_iterations`, `with_tolerance`, `with_restarts`, `with_radii(initial, max)` |
| `FastSimulatedAnnealing` | Cauchy proposals, Metropolis, Ingber cooling. Needs `Objective` | `with_max_iterations`, `with_restarts`, `with_initial_temperature`, `with_cooling_rate`, `with_seed`, `with_boundary` |
| `Fixed` | no search. `factor` only. Does not implement `Optimizer` | unit struct |
| `Adam` | mini-batch, `Svgp` only | see above |

`with_restarts(n, seed)` adds `n` extra log-uniform starts (`NonZeroU32`) and keeps the lowest value. The first start is the model's `θ`. `BoundaryPolicy::Clamp` (default) projects a proposal just inside the open interval. `BoundaryPolicy::Periodic` wraps to the other side.

`Optimizer::minimize` returns `OptResult { params, value, iterations }`. `USES_CHANGE_INDICES` defaults to `false`. `FastSimulatedAnnealing` sets it `true`, and a custom solver does the same when it reports which coordinates changed. With `CholeskyBuffer::Retain`, a fit then rebuilds only the touched kernel leaves. Implement `Optimizer<P>` for a solver of your own. `P` is `Objective`, `Differentiable`, or `TwiceDifferentiable`. `IncrementalObjective::value_with_changes` rebuilds only the leaves those indices touch. Empty, duplicate, or out-of-range indices are `GprError`.

`Objective::num_params` is the length. `Objective::value` is the scalar. `value_at_changes` lists every coordinate that differs from the previous evaluation on that objective. `fill_intervals` writes each parameter's open interval in user units. The built-in solvers stay inside those intervals. `Differentiable` adds `gradient_into` and `value_and_gradient_into`. `TwiceDifferentiable` adds `hessian_into` and `value_gradient_hessian_into`.

```rust
use std::num::{NonZeroU32, NonZeroUsize};

use gprx::kernel::{KernelSpec, RbfKernel};
use gprx::{
    BoundaryPolicy, FastSimulatedAnnealing, Fixed, GaussianLikelihood, Gpr, Lbfgs, NelderMead,
    TrustRegion,
};

fn main() -> Result<(), gprx::GprError> {
    let lbfgs = Lbfgs::new()
        .with_max_iterations(40)
        .with_tolerance(1e-6)?
        .with_history_size(NonZeroUsize::MIN)
        .with_restarts(NonZeroU32::MIN, 1);
    let nm = NelderMead::new()
        .with_max_iterations(20)
        .with_tolerance(1e-6)?;
    let tr = TrustRegion::new()
        .with_max_iterations(20)
        .with_tolerance(1e-6)?
        .with_radii(1.0, 10.0)?;
    let fsa = FastSimulatedAnnealing::new()
        .with_max_iterations(20)
        .with_initial_temperature(1.0)?
        .with_cooling_rate(0.95)?
        .with_seed(1)
        .with_boundary(BoundaryPolicy::Periodic);
    let _clamp = BoundaryPolicy::Clamp;
    let _ = Gpr::new(
        KernelSpec::from(RbfKernel::new(1.0)?),
        GaussianLikelihood::new(0.1)?,
    )
    .with_optimizer(lbfgs)
    .with_optimizer(nm)
    .with_optimizer(tr)
    .with_optimizer(fsa)
    .with_optimizer(Fixed);
    Ok(())
}
```

An objective and a solver of your own:

```rust
use gprx::{
    Differentiable, GprError, IncrementalObjective, Interval, Lbfgs, Objective, OptResult,
    Optimizer, TrustRegion, TwiceDifferentiable,
};

/// `f(θ) = (θ − ln 2)²` on one log-parameter: its minimum is at `ℓ = 2`.
struct Bowl;

impl Objective for Bowl {
    fn num_params(&self) -> usize {
        1
    }

    fn value(&mut self, params: &[f64]) -> Result<f64, GprError> {
        Ok((params[0] - 2.0_f64.ln()).powi(2))
    }
}

impl Differentiable for Bowl {
    fn gradient_into(&mut self, params: &[f64], out: &mut [f64]) -> Result<(), GprError> {
        out[0] = 2.0 * (params[0] - 2.0_f64.ln());
        Ok(())
    }
}

impl TwiceDifferentiable for Bowl {
    fn hessian_into(&mut self, _params: &[f64], out: &mut [f64]) -> Result<(), GprError> {
        out[0] = 2.0;
        Ok(())
    }
}

impl IncrementalObjective for Bowl {
    fn value_with_changes(&mut self, params: &[f64], indices: &[usize]) -> Result<f64, GprError> {
        // One parameter: every change is a full evaluation.
        let _ = indices;
        self.value(params)
    }
}

/// Starts at the middle of each interval, then runs L-BFGS.
struct FromMiddle;

impl<P: Differentiable> Optimizer<P> for FromMiddle {
    const USES_CHANGE_INDICES: bool = false;

    fn minimize(&self, objective: &mut P, _init: &[f64]) -> Result<OptResult, GprError> {
        let mut intervals = vec![Interval::DEFAULT_POSITIVE; objective.num_params()];
        objective.fill_intervals(&mut intervals)?;
        let start: Vec<f64> = intervals.iter().map(|i| (i.lo() * i.hi()).sqrt().ln()).collect();
        Lbfgs::new().minimize(objective, &start)
    }
}

fn main() -> Result<(), GprError> {
    let mut bowl = Bowl;
    let _ = (bowl.value_at_changes(&[0.0], &[0])?, bowl.value_with_changes(&[0.0], &[0])?);
    let mut grad = [0.0];
    let value = bowl.value_and_gradient_into(&[0.0], &mut grad)?;
    let mut hess = [0.0];
    let _ = bowl.value_gradient_hessian_into(&[0.0], &mut grad, &mut hess)?;
    let OptResult { params, value: best, iterations } = FromMiddle.minimize(&mut bowl, &[0.0])?;
    assert!((params[0] - 2.0_f64.ln()).abs() < 1e-4 && best <= value);
    let _ = iterations;
    let _ = TrustRegion::new().minimize(&mut bowl, &[0.0])?;
    Ok(())
}
```

### Precision, math, jitter

`DoublePrecision` stores and solves in `f64` (`PrecisionPolicy::Storage` and `Refine`). `SinglePrecision` uses `f32` and keeps that factorization; its predictions are `Prediction<f32>`. `MixedPrecision` defaults to `MixedPrecision<PromoteStorage>`. It factors in `f32`, refines the predict weights in `f64`, and predicts `f64`. `MixedPrecision<ReevaluateKernel>` is the other `ResidualFormula`. `GpScalar` is the bound on a fitted model's precision parameter `P`; the three policies implement it. Select with `with_precision::<SinglePrecision>()`.

`KernelExp::Accurate` and `FastApprox` are the runtime switch (`with_math`). `Accurate` and `FastApprox` in `gprx` are the corresponding `KernelMath` types for `CompiledKernel::apply::<M>` (see Kernels).

`JitterPolicy::fixed(j)` retries a failed Cholesky once with `j ≥ 0` on the diagonal (`JitterPolicy::Fixed`, `FixedJitter`). `adaptive(initial, multiplier, max_retries, max_jitter)` grows the offset after the unregularized factor fails (`JitterPolicy::Adaptive`, `AdaptiveJitter`): `initial > 0`, `multiplier > 1`, `max_retries ≥ 1`, `max_jitter ≥ initial`. Exact models default to `fixed(0.0)`. `Sgpr` and `Svgp` default to `adaptive(1e-8, 10.0, 5, 1e-3)` on `K_mm`. Read the stored numbers with `jitter`, or `initial`, `multiplier`, `max_retries`, and `max_jitter`. The enum is non-exhaustive.

```rust
use gprx::kernel::{KernelSpec, RbfKernel};
use gprx::{
    AdaptiveJitter, DoublePrecision, FittedGpr, FixedJitter, GaussianLikelihood, GpScalar, Gpr,
    JitterPolicy, KernelExp, MixedPrecision, PrecisionPolicy, PromoteStorage, ReevaluateKernel,
    ResidualFormula,
};

/// The storage scalar of a precision policy, by name.
fn storage<P: PrecisionPolicy>() -> &'static str {
    std::any::type_name::<P::Storage>()
}

/// Reads the observation count of any fitted model on a `KernelSpec`.
fn points<P: GpScalar>(model: &FittedGpr<gprx::Lbfgs, P>) -> usize {
    model.n()
}

fn formula<R: ResidualFormula>(_: MixedPrecision<R>) {}

fn main() -> Result<(), gprx::GprError> {
    assert_eq!(storage::<DoublePrecision>(), "f64");
    assert_eq!(storage::<MixedPrecision<PromoteStorage>>(), "f32");
    let _ = formula::<ReevaluateKernel>;
    let mixed = Gpr::new(
        KernelSpec::from(RbfKernel::new(1.0)?),
        GaussianLikelihood::new(0.1)?,
    )
    .with_precision::<MixedPrecision<ReevaluateKernel>>()
    .with_math(KernelExp::Accurate)
    .with_jitter_policy(JitterPolicy::adaptive(1e-8, 10.0, 5, 1e-3)?)
    .fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
    .map_err(|(_, e)| e)?;
    assert_eq!(points(&mixed), 2);

    match JitterPolicy::adaptive(1e-8, 10.0, 5, 1e-3)? {
        JitterPolicy::Adaptive(adaptive) => {
            let adaptive: AdaptiveJitter = adaptive;
            let _ = (adaptive.initial(), adaptive.multiplier());
            let _ = (adaptive.max_retries(), adaptive.max_jitter());
        }
        JitterPolicy::Fixed(fixed) => {
            let _ = fixed.jitter();
        }
        _ => {}
    }
    if let JitterPolicy::Fixed(fixed) = JitterPolicy::fixed(1e-6)? {
        let fixed: FixedJitter = fixed;
        assert_eq!(fixed.jitter(), 1e-6);
    }
    Ok(())
}
```

### Parameters

`Interval::new(lo, hi)` is a finite open interval, `lo < hi`. `lo`, `hi`, `contains`. `IntervalError::InvalidBounds` and `OutOfRange`. `GprError::InvalidInterval` wraps that error. `BoundedParam::new(value, interval)` stores a user-unit value strictly inside the interval. `default_positive(value)` uses `Interval::DEFAULT_POSITIVE`. `value` reads the value, `interval` the interval, `ln` is `log(value)`. `with_value` keeps the interval. `with_interval` keeps the value. Leaves and `GaussianLikelihood` hold a `BoundedParam` internally. Callers usually go through `with_bounds`.

```rust
use gprx::{BoundedParam, Interval, IntervalError};

fn main() -> Result<(), gprx::GprError> {
    assert!(matches!(
        Interval::new(2.0, 1.0),
        Err(IntervalError::InvalidBounds { .. })
    ));
    let interval = Interval::new(1e-3, 1e2)?;
    assert!(interval.contains(1.0));
    let _ = (interval.lo(), interval.hi(), Interval::DEFAULT_POSITIVE);
    let param = BoundedParam::new(1.0, interval)?;
    let _ = (param.value(), param.interval(), param.ln());
    let param = param.with_value(2.0)?.with_interval(interval)?;
    let _ = BoundedParam::default_positive(0.1)?;
    assert!(matches!(
        param.with_value(1e3),
        Err(IntervalError::OutOfRange { .. })
    ));
    Ok(())
}
```

### Save and load (`gprx::persist`)

`FORMAT_VERSION` is `1`. `RESERVED_PREFIX` is `"gprx."`. A caller `persist_id` must not use that prefix.

`LoadedGpr::load(dir, registry)`, `LoadedSgpr::load`, and `LoadedSvgp::load` read the directory of a coordinate model; `LoadedDistanceGpr::<C>::load`, `LoadedDistanceSgpr::<C>::load`, and `LoadedDistanceSvgp::<C>::load` read that of a model on supplied distances (below). `PersistRegistry::new` is empty. Built-ins need no registration. Register a custom kernel or transform before load:

- `register_kernel`
- `register_unfitted_input`, `register_fitted_input`
- `register_unfitted_target`, `register_fitted_target`

The restore function types are `KernelRestore`, `UnfittedInputRestore`, `FittedInputRestore`, `UnfittedTargetRestore`, and `FittedTargetRestore`.

```rust
use gprx::kernel::{KernelSpec, RbfKernel};
use gprx::persist::{
    LoadedGpr, LoadedSgpr, LoadedSvgp, PersistRegistry, FORMAT_VERSION, RESERVED_PREFIX,
};
use gprx::{GaussianLikelihood, Gpr, Lbfgs, PersistErrorKind};

fn main() -> Result<(), gprx::GprError> {
    let fitted = Gpr::new(
        KernelSpec::from(RbfKernel::new(1.0)?),
        GaussianLikelihood::new(0.1)?,
    )
    .fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0])?;
    let dir = std::env::temp_dir().join(format!("gprx-readme-save-en-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    fitted.save(&dir)?;
    assert_eq!(FORMAT_VERSION, 1);
    assert!(!"mine.kernel".starts_with(RESERVED_PREFIX));

    let loaded = LoadedGpr::load(&dir, &PersistRegistry::new())?;
    let pred = loaded.predict(&[0.5], 1, 1)?;
    assert_eq!(loaded.n(), 2);
    assert_eq!(loaded.d(), 1);
    assert!(!loaded.is_online());
    let LoadedGpr::Double(model) = loaded else {
        return Err(gprx::GprError::PersistFailed {
            kind: PersistErrorKind::WrongModel,
            reason: "expected Double".into(),
        });
    };
    let mut model = model.with_optimizer(Lbfgs::new());
    model.refit()?;
    let factor_dir = std::env::temp_dir().join(format!("gprx-readme-factor-en-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&factor_dir);
    model.save_with_factor(&factor_dir)?;
    let _ = pred.mean[0];
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&factor_dir);

    let x = &[0.0, 1.0, 2.0, 3.0];
    let y = &[0.0, 1.0, 0.5, 0.25];
    let z = &[0.5, 2.5];
    let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    let likelihood = GaussianLikelihood::new(0.1)?;
    let sparse = gprx::Sgpr::new(kernel.clone(), likelihood.clone())
        .fit(x, 4, 1, y, z, 2)
        .map_err(|(_, e)| e)?;
    let sparse_dir = std::env::temp_dir().join(format!("gprx-readme-sgpr-en-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&sparse_dir);
    sparse.save(&sparse_dir)?;
    let loaded = LoadedSgpr::load(&sparse_dir, &PersistRegistry::new())?;
    let _ = loaded.m();
    let LoadedSgpr::Double(model) = loaded else {
        return Err(gprx::GprError::PersistFailed {
            kind: PersistErrorKind::WrongModel,
            reason: "expected Double".into(),
        });
    };
    let mut online = model.into_online();
    let id = online.insert_inducing(&[1.5])?;
    assert!(online.inducing_ids().contains(&id));
    online.delete_inducing(id)?;
    let _ = std::fs::remove_dir_all(&sparse_dir);

    let svgp = gprx::Svgp::new(kernel, likelihood)
        .factor(x, 4, 1, y, z, 2)
        .map_err(|(_, e)| e)?;
    let svgp_dir = std::env::temp_dir().join(format!("gprx-readme-svgp-en-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&svgp_dir);
    svgp.save(&svgp_dir)?;
    let loaded = LoadedSvgp::load(&svgp_dir, &PersistRegistry::new())?;
    let LoadedSvgp::Double(model) = loaded else {
        return Err(gprx::GprError::PersistFailed {
            kind: PersistErrorKind::WrongModel,
            reason: "expected Double".into(),
        });
    };
    let _ = model.neg_elbo()?;
    let _ = std::fs::remove_dir_all(&svgp_dir);
    Ok(())
}
```

A kernel leaf, an input map, and a target map of your own, saved and loaded through the registry:

```rust
use std::any::Any;
use std::sync::Arc;

use faer::{MatMut, MatRef};
use gprx::kernel::{CustomKernel, KernelScalar, KernelSpec, KernelTerm, RbfKernel, Triangle};
use gprx::persist::{
    FittedInputRestore, FittedTargetRestore, KernelRestore, LoadedGpr, PersistRegistry,
    UnfittedInputRestore, UnfittedTargetRestore,
};
use gprx::transform::{TargetTransform, Transform, UnfittedTarget, UnfittedTransform};
use gprx::{Fixed, GaussianLikelihood, Gpr, GprError, Interval};

/// `k(d²) = exp(−d² / 2)`: an RBF with `ℓ = 1` and no parameter.
#[derive(Clone, Debug)]
struct UnitRbf;

fn no_params(len: usize) -> Result<(), GprError> {
    if len == 0 {
        Ok(())
    } else {
        Err(GprError::LengthMismatch {
            reason: format!("UnitRbf has no parameters, got {len}"),
        })
    }
}

fn no_param(i: usize) -> GprError {
    GprError::IndexOutOfRange {
        reason: format!("UnitRbf has no parameter {i}"),
    }
}

impl<T: KernelScalar> KernelTerm<T> for UnitRbf {
    fn num_params(&self) -> usize {
        0
    }

    fn get_params(&self, out: &mut [f64]) -> Result<(), GprError> {
        no_params(out.len())
    }

    fn set_params(&mut self, params: &[f64]) -> Result<(), GprError> {
        no_params(params.len())
    }

    fn bounds_into(&self, out: &mut [Interval]) -> Result<(), GprError> {
        no_params(out.len())
    }

    fn apply(&self, dist: MatRef<'_, T>, mut out: MatMut<'_, T>, uplo: Triangle) -> Result<(), GprError> {
        for col in 0..dist.ncols() {
            let rows = match uplo {
                Triangle::Lower => col..dist.nrows(),
                Triangle::Upper => 0..col + 1,
                Triangle::Full => 0..dist.nrows(),
            };
            for row in rows {
                out[(row, col)] = (T::from_f64(-0.5) * dist[(row, col)]).exp();
            }
        }
        Ok(())
    }

    fn apply_cross(&self, dist: MatRef<'_, T>, out: MatMut<'_, T>) -> Result<(), GprError> {
        self.apply(dist, out, Triangle::Full)
    }

    fn fill_diag(&self, out: &mut [T]) -> Result<(), GprError> {
        out.fill(T::from_f64(1.0));
        Ok(())
    }

    fn grad(&self, _: MatRef<'_, T>, _: MatMut<'_, T>, i: usize, _: Triangle) -> Result<(), GprError> {
        Err(no_param(i))
    }

    fn hess(&self, _: MatRef<'_, T>, _: MatMut<'_, T>, i: usize, _: usize, _: Triangle) -> Result<(), GprError> {
        Err(no_param(i))
    }

    fn hess_points(&self, _: MatRef<'_, T>, _: MatMut<'_, T>, i: usize, _: usize, _: Triangle) -> Result<(), GprError> {
        Err(no_param(i))
    }

    fn persist_id(&self) -> &'static str {
        "mine.unit_rbf"
    }

    fn persist_state(&self) -> Result<serde_json::Value, GprError> {
        Ok(serde_json::json!({}))
    }
}

/// Moves every input by `−1`. Fitting learns nothing, so one type is both forms.
#[derive(Clone, Debug)]
struct Shift;

impl UnfittedTransform for Shift {
    fn fit(self: Box<Self>, _x: &[f64], _n_rows: usize, _n_cols: usize) -> Result<Box<dyn Transform>, GprError> {
        Ok(self)
    }

    fn clone_box(&self) -> Box<dyn UnfittedTransform> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn persist_id(&self) -> Option<&'static str> {
        Some("mine.shift")
    }

    fn persist_state(&self) -> Result<serde_json::Value, GprError> {
        Ok(serde_json::json!({}))
    }
}

impl Transform for Shift {
    fn apply(&self, x: &mut [f64], _n_rows: usize, _n_cols: usize) -> Result<(), GprError> {
        x.iter_mut().for_each(|v| *v -= 1.0);
        Ok(())
    }

    fn inverse_apply(&self, x: &mut [f64], _n_rows: usize, _n_cols: usize) -> Result<(), GprError> {
        x.iter_mut().for_each(|v| *v += 1.0);
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn Transform> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn persist_id(&self) -> Option<&'static str> {
        Some("mine.shift")
    }

    fn persist_state(&self) -> Result<serde_json::Value, GprError> {
        Ok(serde_json::json!({}))
    }
}

/// Halves `y` for the model and doubles the predictions back.
#[derive(Clone, Debug)]
struct Half;

impl UnfittedTarget for Half {
    fn fit(self: Box<Self>, _y: &[f64]) -> Result<Box<dyn TargetTransform>, GprError> {
        Ok(self)
    }

    fn clone_box(&self) -> Box<dyn UnfittedTarget> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn persist_id(&self) -> Option<&'static str> {
        Some("mine.half")
    }

    fn persist_state(&self) -> Result<serde_json::Value, GprError> {
        Ok(serde_json::json!({}))
    }
}

impl TargetTransform for Half {
    fn transform(&self, y: &mut [f64]) -> Result<(), GprError> {
        y.iter_mut().for_each(|v| *v *= 0.5);
        Ok(())
    }

    fn inverse_transform_mean(&self, mean: &mut [f64]) -> Result<(), GprError> {
        mean.iter_mut().for_each(|v| *v *= 2.0);
        Ok(())
    }

    fn inverse_transform_variance(&self, var: &mut [f64]) -> Result<(), GprError> {
        var.iter_mut().for_each(|v| *v *= 4.0);
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn TargetTransform> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn persist_id(&self) -> Option<&'static str> {
        Some("mine.half")
    }

    fn persist_state(&self) -> Result<serde_json::Value, GprError> {
        Ok(serde_json::json!({}))
    }
}

fn main() -> Result<(), GprError> {
    // Two copies of the custom leaf, summed, times a built-in leaf.
    let kernel = KernelSpec::from(CustomKernel::new(UnitRbf)) + KernelSpec::custom(UnitRbf);
    let kernel = kernel * KernelSpec::from(RbfKernel::new(2.0)?);
    let fitted = Gpr::new(kernel, GaussianLikelihood::new(0.1)?)
        .with_optimizer(Fixed)
        .with_input_transform(Shift)
        .with_target_transform(Half)
        .factor(&[0.0, 1.0, 2.0], 3, 1, &[0.0, 1.0, 0.5])?;
    let dir = std::env::temp_dir().join(format!("gprx-readme-custom-en-{}", std::process::id()));
    fitted.save(&dir)?;

    // One restore function per persist id.
    let kernel: KernelRestore = Arc::new(|_state| Ok(CustomKernel::new(UnitRbf)));
    let unfitted_input: UnfittedInputRestore = Arc::new(|_state| Ok(Box::new(Shift)));
    let fitted_input: FittedInputRestore = Arc::new(|_state| Ok(Box::new(Shift)));
    let unfitted_target: UnfittedTargetRestore = Arc::new(|_state| Ok(Box::new(Half)));
    let fitted_target: FittedTargetRestore = Arc::new(|_state| Ok(Box::new(Half)));
    let mut registry = PersistRegistry::new();
    registry.register_kernel("mine.unit_rbf", move |state| kernel(state))?;
    registry.register_unfitted_input("mine.shift", move |state| unfitted_input(state))?;
    registry.register_fitted_input("mine.shift", move |state| fitted_input(state))?;
    registry.register_unfitted_target("mine.half", move |state| unfitted_target(state))?;
    registry.register_fitted_target("mine.half", move |state| fitted_target(state))?;

    let loaded = LoadedGpr::load(&dir, &registry)?;
    let query = [0.5];
    assert_eq!(loaded.predict(&query, 1, 1)?, fitted.predict(&query, 1, 1)?);
    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}
```

A model on supplied distances saves with the same `save` / `save_with_factor`. The file also holds the training `d²` the model owns (an Exact model's lower triangles, a sparse model's `n × m` blocks and its inducing indices) and the kernel's slot table. The loader is typed by the kernel's marker (`DistanceOnly` or `WithPoints`); a coordinate file, or one of the other marker, is `WrongModel`. A loaded kernel has new slots: a `ScalarDistance` or `ArdDistance` from before the save names none of them, so take them from `slots()` (in the saved kernel's order) and bind the queries to those. The format: [persist-format.md §10](https://github.com/YUKIKEDA/gprx/blob/main/docs/persist-format.md#10-models-on-supplied-distances).

```rust
use gprx::kernel::{
    DistanceOnly, DistanceSlot, KernelSpec, RbfKernel, ScalarDistance, WithPoints,
};
use gprx::persist::{LoadedDistanceGpr, LoadedDistanceSgpr, LoadedDistanceSvgp, PersistRegistry};
use gprx::{Fixed, GaussianLikelihood, GprError, Gpr, Sgpr, Svgp};

/// The one scalar slot of a loaded kernel.
fn scalar(slots: &[DistanceSlot]) -> Result<ScalarDistance, GprError> {
    match slots {
        [DistanceSlot::Scalar(slot)] => Ok(*slot),
        _ => Err(GprError::InvalidConfig {
            reason: "expected one scalar slot".into(),
        }),
    }
}

fn main() -> Result<(), GprError> {
    let registry = PersistRegistry::new();
    let dir = std::env::temp_dir().join(format!("gprx-readme-distance-en-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let image = ScalarDistance::new();
    // Three samples at 0, 1, 2: their 3 × 3 squared distances.
    let train = [0.0, 1.0, 4.0, 1.0, 0.0, 1.0, 4.0, 1.0, 0.0];
    let y = [0.0, 1.0, 0.5];
    // The three samples × one query at 0.5.
    let cross = [0.25, 0.25, 2.25];

    let fitted = Gpr::new(image.kernel(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?)
        .with_optimizer(Fixed)
        .factor([image.borrow(&train)], 3, &y)
        .map_err(|(_, e)| e)?;
    fitted.save(&dir)?;
    // The marker is part of the loaded type: name it.
    let loaded = LoadedDistanceGpr::<DistanceOnly>::load(&dir, &registry)?;
    assert_eq!((loaded.n(), loaded.is_online()), (3, false));
    // The loaded kernel has new slots: bind the query to them.
    let slot = scalar(&loaded.slots())?;
    assert_eq!(loaded.to_kernel().slots(), loaded.slots());
    let pred = loaded.predict([slot.borrow(&cross)], 1)?;
    assert_eq!(pred.mean, fitted.predict([image.borrow(&cross)], 1)?.mean);

    // A kernel with coordinate leaves is `WithPoints`: the query adds its coordinates.
    let kernel = image.kernel(RbfKernel::new(1.0)?) * KernelSpec::from(RbfKernel::new(0.5)?);
    Gpr::new(kernel, GaussianLikelihood::new(0.1)?)
        .with_optimizer(Fixed)
        .factor([image.borrow(&train)], 3, &[0.0, 1.0, 2.0], 1, &y)
        .map_err(|(_, e)| e)?
        .save(&dir)?;
    let loaded = LoadedDistanceGpr::<WithPoints>::load(&dir, &registry)?;
    assert_eq!(loaded.d(), 1);
    let slot = scalar(&loaded.slots())?;
    let _ = loaded.predict_with([slot.borrow(&cross)], &[0.5], 1, 1, gprx::PredictOptions::default())?;

    // Sparse: the blocks from the three samples to the inducing samples 0 and 2.
    let blocks = [0.0, 1.0, 4.0, 4.0, 1.0, 0.0];
    Sgpr::new(image.kernel(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?)
        .with_optimizer(Fixed)
        .factor([image.borrow(&blocks)], 3, &y, &[0, 2])
        .map_err(|(_, e)| e)?
        .save(&dir)?;
    let loaded = LoadedDistanceSgpr::<DistanceOnly>::load(&dir, &registry)?;
    assert_eq!((loaded.m(), loaded.inducing()), (2, Some(&[0, 2][..])));
    // A loaded online model names its inducing points by `PointId` instead.
    assert!(loaded.inducing_points().is_none());
    let slot = scalar(&loaded.slots())?;
    // The inducing samples × the query at 0.5.
    let _ = loaded.predict([slot.borrow(&[2.25, 0.25])], 1)?;
    let LoadedDistanceSgpr::Double(model) = loaded else {
        return Err(GprError::InvalidConfig {
            reason: "expected Double".into(),
        });
    };
    let _online = model.into_online();

    Svgp::new(image.kernel(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?)
        .factor([image.borrow(&blocks)], 3, &y, &[0, 2])
        .map_err(|(_, e)| e)?
        .save(&dir)?;
    let loaded = LoadedDistanceSvgp::<DistanceOnly>::load(&dir, &registry)?;
    let slot = scalar(&loaded.slots())?;
    let _ = loaded.predict([slot.borrow(&[2.25, 0.25])], 1)?;
    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}
```

`predict` and `predict_with` on a loaded model return `f64`, including when the file was `f32`. `n`, `d`, and (sparse) `m`. `is_online` is true for a file an `OnlineGpr` or `OnlineSgpr` wrote. Match the variant for the typed model:

| Enum | Variants |
| --- | --- |
| `LoadedGpr` | `Double`, `Single`, `Mixed`, `Reevaluate`, `OnlineDouble`, `OnlineSingle`, `OnlineMixed`, `OnlineReevaluate` |
| `LoadedSgpr` | the same eight names. `Double` is `FittedSgpr<Fixed>`. Online variants are `OnlineSgpr<Fixed, _>` |
| `LoadedSvgp` | `Double`, `Single`, `Mixed`, `Reevaluate`. No online variant |
| `LoadedDistanceGpr<C>` | the eight names of `LoadedGpr`, over `FittedGpr<Fixed, _, DistanceKernel<C>>` / `OnlineGpr<Fixed, _, DistanceKernel<C>>`. `C` is `DistanceOnly` or `WithPoints`, named at `load` |
| `LoadedDistanceSgpr<C>` | the eight names of `LoadedSgpr`, over the models of `DistanceKernel<C>` |
| `LoadedDistanceSvgp<C>` | the four names of `LoadedSvgp`, over `FittedSvgp<_, DistanceKernel<C>>` |

A loaded exact model is `Fixed` and `CholeskyBuffer::Retain`. The file does not store a solver. Call `with_optimizer` on the matched model, then `refit`, to search again. A file written with `save` and no factor is factored on load. A file from `save_with_factor` is not: an `f64` model keeps `L` memory-mapped, and an `f32` model copies it.

### Errors

`GprError` is non-exhaustive. Display text is English.

| Variant | When |
| --- | --- |
| `DimensionMismatch { x_dim, expected_dim }` | query features differ from training |
| `InsufficientData { n, min }` | too few points |
| `EmptyInput` | a dimension is zero |
| `NonFiniteInput` | `NaN` or `Inf` in caller data |
| `NonFiniteKernelValue` | a kernel evaluation was not finite |
| `CholeskyFailed { jitter, matrix_size, stage }` | factorization failed after jitter |
| `NonPositiveDefiniteMatrix` | a matrix that must be a covariance is not |
| `CoordGradientUnsupported` | the kernel has no derivative the sparse model needs |
| `OptimizationNotConverged { iterations }` | the solver stopped short of its test |
| `InvalidHyperparameter { reason }` | a kernel parameter is outside its domain |
| `ShapeMismatch { reason }` | a matrix has the wrong shape |
| `InvalidDistance { slot, dim, pair, reason }` | a supplied squared distance is not finite, is negative, or breaks a square's zero diagonal or symmetry. `slot` is the table's place in the kernel's `slots()`, `dim` its ARD dimension, `pair` the `(row, col)` in the block passed; each is `None` when unknown |
| `DistanceSlot { kind, slot }` | the sources do not match the kernel's slots: `SlotErrorKind::NotRead` (a slot the kernel does not read, such as one from before a save), `Duplicate`, or `Missing`; `slot` is the place in `slots()` of the slot with two sources or none, and `None` for `NotRead` |
| `LengthMismatch { reason }` | a slice has the wrong length |
| `IndexOutOfRange { reason }` | a parameter, leaf, or dimension index |
| `InvalidConfig { reason }` | an optimizer, jitter, or transform setting |
| `SizeOverflow` | `n_rows * n_cols` does not fit in `usize` |
| `InvalidInterval` | an `Interval` could not be built, or a value is outside its interval (a noise variance outside `(1e-5, 1e5)` by default) |
| `InvalidNoiseVariance { reason }` | observation noise is outside its domain |
| `UnsupportedKernelOperation { reason }` | that leaf does not implement the operation |
| `WorkspaceTooSmall` | a buffer is shorter than the problem |
| `InvalidPointId` | `PointId` is not in the model |
| `InvalidInducingId` | `InducingId` is not in the model |
| `PersistFailed { kind, reason }` | save or load failed |
| `UnsupportedPersistVersion { found, supported }` | `format_version` is not `FORMAT_VERSION` |

`CholeskyStage` is `Fit`, `Predict`, `OnlineInsert`, `OnlineDelete`. `PersistErrorKind` is `Io`, `Config`, `Tensor`, `InvalidPersistId`, `NotPersistable`, `UnregisteredId`, `WrongModel`. Branch on `kind`. `reason` is for a person to read.

```rust
use gprx::{CholeskyStage, GprError, PersistErrorKind, SlotErrorKind};

fn main() -> Result<(), GprError> {
    let missing = GprError::DistanceSlot {
        kind: SlotErrorKind::Missing,
        slot: Some(0),
    };
    assert!(matches!(missing, GprError::DistanceSlot { kind: SlotErrorKind::Missing, .. }));
    let err = GprError::DimensionMismatch {
        x_dim: 2,
        expected_dim: 1,
    };
    match err {
        GprError::DimensionMismatch { x_dim, expected_dim } => {
            let _ = (x_dim, expected_dim);
        }
        GprError::CholeskyFailed {
            jitter,
            matrix_size,
            stage,
        } => {
            let _ = (jitter, matrix_size, stage);
        }
        GprError::PersistFailed { kind, reason } => {
            let _ = (kind, reason);
        }
        GprError::UnsupportedPersistVersion { found, supported } => {
            let _ = (found, supported);
        }
        other => {
            let _ = other.to_string();
        }
    }
    let _ = (CholeskyStage::Fit, PersistErrorKind::Io);
    Ok(())
}
```

## Architecture and the saved format

Three model families (`Gpr`, `Sgpr`, `Svgp`) are built from the same blocks and never import one another. The crate map is [Architecture §1](https://github.com/YUKIKEDA/gprx/blob/main/docs/architecture.md#1-at-a-glance).

- [`docs/architecture.md`](https://github.com/YUKIKEDA/gprx/blob/main/docs/architecture.md): every module, what it is responsible for, which way its imports point, the public types by family, and where to change what.
- [`docs/persist-format.md`](https://github.com/YUKIKEDA/gprx/blob/main/docs/persist-format.md): what `save` writes. The keys of `config.json`, the tensors of `model.safetensors` (names, shapes, dtypes, column-major layout), the JSON form of kernels and transforms, `Custom` restore, versions, and errors.

## Comparison

Accuracy, training time, and memory against other libraries: [comparison](https://github.com/YUKIKEDA/gprx/blob/main/docs/comparison.md) | [日本語](https://github.com/YUKIKEDA/gprx/blob/main/docs/comparison.ja.md).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
