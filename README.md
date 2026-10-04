English | [日本語](README.ja.md)

# gprx

Exact Gaussian process regression in Rust. `Gpr` is the unfitted trainer. `Gpr::fit` consumes it, runs argmin L-BFGS on the negative log marginal likelihood, and returns `FittedGpr`. The same blocks build `Sgpr` and `Svgp`, including online updates and directory save/load. The crate is **not** published to crates.io (`publish = false` in `Cargo.toml`).

## Status

**0.1.0** is the default-feature public API: `Gpr`, `Sgpr`, and `Svgp`, online updates, and save/load, together with kernels, `fit` / `predict` / `predict_into` / leave-one-out, English rustdoc, and `examples/`. That sentence names families, not every type. The MSRV is 1.85 (`rust-version` in `Cargo.toml`). A 0.x minor may break the public API. `internals` (`bench-internals` and `insert-stages`) is outside semantic versioning. Depend on git or a path, not crates.io.

Design: [`docs/design.md`](https://github.com/YUKIKEDA/gprx/blob/main/docs/design.md). Architecture: [`docs/architecture.md`](https://github.com/YUKIKEDA/gprx/blob/main/docs/architecture.md). Saved format: [`docs/persist-format.md`](https://github.com/YUKIKEDA/gprx/blob/main/docs/persist-format.md). Tasks: [`docs/roadmap.md`](https://github.com/YUKIKEDA/gprx/blob/main/docs/roadmap.md). Agent rules: [`AGENTS.md`](https://github.com/YUKIKEDA/gprx/blob/main/AGENTS.md). Cross-library wall time and peak RSS: [`compare/perf/`](https://github.com/YUKIKEDA/gprx/blob/main/compare/perf/) (P2B-16 Exact `just perf`; P4-12 Sparse `just perf-sparse`; P4-14 Sparse online `just perf-sparse-online`; not criterion).

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

`Gpr<Fixed>::factor` (after `with_optimizer(Fixed)`) factors at the kernel and likelihood `θ` already on the trainer. L-BFGS knobs live on `Lbfgs` (`with_max_iterations`, `with_tolerance`, `with_history_size`, `with_restarts`). Nelder–Mead has `with_max_iterations`, `with_tolerance`, and `with_restarts` (`NelderMead`). The Hessian solver is `TrustRegion` (`with_max_iterations`, `with_tolerance`, `with_restarts`, `with_radii`). Homemade Fast Simulated Annealing is `FastSimulatedAnnealing` (`with_max_iterations`, `with_restarts`, `with_initial_temperature`, `with_cooling_rate`, `with_seed`, `with_boundary`).

## Architecture and the saved format

Three model families (`Gpr`, `Sgpr`, `Svgp`) are built from the same blocks and never import one another. The map below is the whole crate; each box is a module under `src/`.

```mermaid
flowchart TB
    api["<b>Public API</b><br/>lib.rs re-exports; pub mods kernel, transform, persist"]
    subgraph models["Models — one directory per family"]
        direction LR
        gpr["<b>gpr</b><br/>Exact GPR"]
        sgpr["<b>sgpr</b><br/>Sparse GPR (VFE)"]
        svgp["<b>svgp</b><br/>SVGP (minibatch)"]
    end
    sparse["<b>sparse</b><br/>crate-private core shared by sgpr and svgp"]
    persist["<b>persist</b><br/>save / load directories"]
    subgraph services["Building blocks the models compose"]
        direction LR
        kernel["<b>kernel</b><br/>spec, compiled, leaves"]
        likelihood["<b>likelihood</b>"]
        transform["<b>transform</b><br/>input / target maps"]
        precision["<b>precision</b><br/>f32 / f64 / mixed"]
        optimizer["<b>optimizer</b><br/>+ objective traits"]
        workspace["<b>workspace</b><br/>+ prediction"]
    end
    subgraph foundation["Foundation — scalars, numerics, checks"]
        direction LR
        f1["linalg · math · policy"]
        f2["param · data · error · rng · points"]
    end
    api --> models
    gpr --> services
    sgpr --> sparse --> services
    svgp --> sparse
    services --> foundation
    persist -.->|"reads and rebuilds"| models
    models -.->|"save, persist_err"| persist
```

- [`docs/architecture.md`](https://github.com/YUKIKEDA/gprx/blob/main/docs/architecture.md): every module, what it is responsible for, which way its imports point, the public types by family, and where to change what.
- [`docs/persist-format.md`](https://github.com/YUKIKEDA/gprx/blob/main/docs/persist-format.md): what `save` writes. The keys of `config.json`, the tensors of `model.safetensors` (names, shapes, dtypes, column-major layout), the JSON form of kernels and transforms, `Custom` restore, versions, and errors.

## Comparison with other libraries

gprx is compared with scikit-learn, GPyTorch, GPy, libgp, and friedrich on the regression problems used in Gaussian-process papers ([#298](https://github.com/YUKIKEDA/gprx/issues/298)). The order is whether the arithmetic agrees, whether the predictions are good, and whether the model runs on large data. This is a report, not a pass/fail test. A row where another library is better stays in the table.

| tier | data | what it shows | gprx model |
| --- | --- | --- | --- |
| T0 | Snelson 1-D (200 points), Mauna Loa CO₂ | the fit, by eye | `Gpr` |
| T1 | UCI, Hernández-Lobato & Adams splits (20 each; Boston left out): yacht, energy, concrete, wine (red), power plant, kin8nm, naval | accuracy and calibrated uncertainty | `Gpr` |
| T2 | Kin40k, Protein (5 splits) | mid-size scale | `Gpr` where `K` fits in memory, `Sgpr` / `Svgp` |
| T3 | 3DRoad, Song, Buzz, HouseElectric (`treforevans/uci_datasets`, 10 splits of 90 / 10) | large scale | `Sgpr` / `Svgp` |

Every model uses an RBF kernel with one length per input dimension, and Gaussian noise. Inputs and targets are shifted and scaled with the training mean and variance, and the start is the same (length 1, signal variance 1, noise variance 0.1).

There are three scores. RMSE is the prediction error. NLPD is the log loss of the predictive distribution. The third is the fraction of test points that fall inside the 95% predictive interval. Units are the original units of `y`, written as the mean and standard error over the data splits.

Before the comparison, every library is evaluated at one fixed set of hyperparameters, with no training (`just perf-real-check`). The negative log marginal likelihood, RMSE, and NLPD agree to 1e-6. A difference after training is a difference in the optimizer, not in the objective. The same check on the inducing-point model (`just perf-real-check yacht 0 sgpr`) finds the lower bound of the marginal likelihood equal in gprx, GPyTorch, and GPy. GPyTorch alone computes the test variance with its own low-rank formula, so at the same parameters and inducing points its RMSE and NLPD differ from the other two by less than one percent.

### Optimizers matter

Training time depends on the optimizer as much as on the matrix arithmetic. Each row records which setup trained the model, and how many times the likelihood and the gradient were computed together.

There are two setups.

- The library default. Each library trains with the settings it ships with.
- A shared setup. Each library keeps its own objective and gradient. The iteration limit is 100, the gradient tolerance is √ε, and the history length is 10. The programs are not the same. gprx uses argmin's L-BFGS with a More–Thuente line search, and those limits are already its default. scikit-learn, GPyTorch, and GPy pass the same limits to scipy's L-BFGS-B. libgp has only Rprop, which cannot take a gradient tolerance. The minibatch model has no default Adam setting that finishes on hundreds of thousands of points, so every library uses learning rate 0.01, batch size 1024, and three passes over the data.

A difference in fit time is not a difference in speed when the evaluation counts differ. Peak memory is the high point of the resident set of the whole process. Memory over time is in the figures in the results.

### Interface

The same regression in each library: ARD RBF, learn the hyperparameters, predict the mean and the observation variance, score the NLPD. Runnable files: [`compare/perf/real/snippets/`](https://github.com/YUKIKEDA/gprx/blob/main/compare/perf/real/snippets/).

<!-- snippets:begin -->
<details><summary>gprx</summary>

```rust
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
```

</details>

<details><summary>scikit-learn</summary>

```python
kernel = ConstantKernel(1.0) * RBF([1.0] * d) + WhiteKernel(0.1)
gp = GaussianProcessRegressor(kernel).fit(x, y)
mean, std = gp.predict(xs, return_std=True)  # std includes the noise
nlpd = np.mean(0.5 * np.log(2 * np.pi * std**2) + 0.5 * (ys - mean) ** 2 / std**2)
```

</details>

<details><summary>GPyTorch</summary>

```python
class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, x, y, likelihood):
        super().__init__(x, y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=d))

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
model = ExactGP(x, y, likelihood).double()
model.train(); likelihood.train()
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # there is no default optimizer
for _ in range(50):
    optimizer.zero_grad()
    loss = -mll(model(x), y)
    loss.backward()
    optimizer.step()
model.eval(); likelihood.eval()
with torch.no_grad():
    pred = likelihood(model(xs))  # the observation distribution
    mean, var = pred.mean, pred.variance
nlpd = torch.mean(0.5 * torch.log(2 * torch.pi * var) + 0.5 * (ys - mean) ** 2 / var)
```

</details>

<details><summary>GPy</summary>

```python
kernel = GPy.kern.RBF(d, variance=1.0, lengthscale=[1.0] * d, ARD=True)
model = GPy.models.GPRegression(x, y[:, None], kernel)
model.Gaussian_noise.variance = 0.1
model.optimize()
mean, var = model.predict(xs)  # includes the noise
nlpd = np.mean(0.5 * np.log(2 * np.pi * var) + 0.5 * (ys[:, None] - mean) ** 2 / var)
```

</details>

<details><summary>libgp</summary>

```cpp
libgp::GaussianProcess gp(d, "CovSum ( CovSEard, CovNoise)");
Eigen::VectorXd loghyper(d + 2);  // log ell (d of them), log sf, log sn
loghyper << 0.0, 0.0, 0.0, 0.0, std::log(std::sqrt(0.1));
gp.covf().set_loghyper(loghyper);
for (int i = 0; i < n; ++i) {  // add_patterns(x, y) would read strided rows
    std::vector<double> row(d);
    for (int j = 0; j < d; ++j) row[j] = x(i, j);
    gp.add_pattern(row.data(), y(i));
}
libgp::RProp rprop;  // libgp's optimizer: resilient backpropagation
rprop.init();
rprop.maximize(&gp, 100, false);
const Eigen::MatrixXd pred = gp.predict(xs, true);  // mean, latent variance
const double noise = std::exp(2.0 * gp.covf().get_loghyper()(d + 1));
double nlpd = 0.0;
for (int i = 0; i < m; ++i) {
    const double v = pred(i, 1) + noise, e = ys(i) - pred(i, 0);
    nlpd += 0.5 * std::log(2.0 * M_PI * v) + 0.5 * e * e / v;
}
nlpd /= m;
```

</details>
<!-- snippets:end -->

libgp has two behaviors the comparison code avoids. The variance from `predict` leaves out the noise. The bulk `add_patterns(x, y)` reads a column-major matrix as rows, which is correct only when the dimension is 1.

### Features

`✓` means found in the listed version's source or documentation; `—` means not found there (it says nothing about extensions). Versions: scikit-learn 1.6.1, GPyTorch 1.15.2, GPy 1.14.2, libgp `f4a2fb7`, friedrich 0.6.0.

| | gprx | scikit-learn | GPyTorch | GPy | libgp | friedrich |
| --- | --- | --- | --- | --- | --- | --- |
| language | Rust | Python | Python (PyTorch) | Python | C++ (Eigen) | Rust |
| ARD RBF kernel | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| kernel sum / product | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| sparse GP with inducing points | ✓ `Sgpr` | — | ✓ `InducingPointKernel` | ✓ `SparseGPRegression` | — | — |
| SVGP with minibatches | ✓ `Svgp` (Adam) | — | ✓ | class only, no minibatch loop | — | — |
| add training points without a full refit | ✓ `OnlineGpr` | — | ✓ `get_fantasy_model` | — | ✓ `add_pattern` | ✓ `add_samples` |
| delete training points | ✓ `OnlineGpr` | — | — | — | — | — |
| posterior samples | ✓ `sample` | ✓ `sample_y` | ✓ | ✓ `posterior_samples` | — | ✓ `sample_at` |
| f32 / mixed precision | ✓ | — | ✓ | — | — | — |
| save and load a fitted model | ✓ | ✓ pickle / joblib | ✓ `state_dict` | ✓ `save_model` | ✓ `write` | ✓ serde |
| swap the optimizer | ✓ `Optimizer` | ✓ callable | ✓ any torch optimizer | ✓ `optimizer=` | ✓ `RProp` / `CG` | — |

### Results

The numbers in this section are one comparison, on one PC, with the same training setup in every library.

Exact GP and SGPR (512 inducing points) use L-BFGS for at most 100 iterations. The minibatch model uses Adam, learning rate 0.01, 1024 points at a time, and three passes over the data. None of the libraries is trained with the optimizer settings it ships with.

Exact GP is Snelson, Mauna Loa, yacht, and energy. SGPR is wine, power plant, and naval (20 splits), kin40k (5 splits), and 3droad and song (1 split). The minibatch model is kin40k, 3droad, song, and HouseElectric (1 split each).

RMSE is the prediction error and NLPD is the log loss of the predictive distribution; lower is better. The 95% column is the fraction of test points inside that interval. Fit seconds are the timed training. Evaluations count a combined likelihood-and-gradient call. The iteration column is the optimizer's own count, and the gprx rows are blank. Milliseconds per evaluation are the median of fit time divided by the evaluation count. NLML is the negative log marginal likelihood at the end of training. Memory is the high point of the whole process tree's resident set.

Treat a gap in fit seconds as a difference in speed only where the evaluation counts match.

On song with inducing points, gprx's NLPD differs from GPyTorch and GPy. On power plant, GPy's predictions broke on some splits, so its RMSE and NLPD averages are not a measure of fit.

The figures follow the tables. The sentence above each figure says what it shows.

<!-- bench:begin -->
Measured on Intel64 Family 6 Model 191 Stepping 2, GenuineIntel (16 logical CPUs, 47.8 GiB, Windows-11-10.0.26200-SP0). scikit-learn 1.6.1, gpytorch 1.15.2, GPy 1.14.2, torch 2.14.0, scipy 1.18.1, argmin 0.11.0; libgp f4a2fb7d.

| library | default optimizer | shared optimizer | search space | bounds |
| --- | --- | --- | --- | --- |
| gprx | argmin 0.11 LBFGS + MoreThuente line search; history 10, max 100 iterations, gradient-norm tolerance sqrt(eps) | same call: the gprx default already equals the shared setting | logit of log θ inside each interval, so the search is unconstrained | (1e-5, 1e5) on ℓ, signal variance and noise variance |
| sklearn | scipy minimize L-BFGS-B via optimizer='fmin_l_bfgs_b' with scipy defaults (maxiter 15000, ftol 2.2e-9, gtol 1e-5, maxcor 10, maxls 20) | scipy L-BFGS-B, maxiter 100, gtol sqrt(eps), ftol 0, maxcor 10; the library's own objective and gradient | log θ | (1e-5, 1e5) on ℓ, constant value and noise level (kernel defaults) |
| gpytorch | torch.optim.Adam, lr 0.1, 50 steps (the exact-GP tutorial setting; GPyTorch has no default optimizer) | scipy L-BFGS-B, maxiter 100, gtol sqrt(eps), ftol 0, maxcor 10; the library's own objective and gradient; gradient by autograd on -mll × n | raw parameters behind softplus | noise ≥ 1e-5 (set here; GPyTorch's default is 1e-4); ℓ and outputscale positive only |
| gpy | model.optimize(): paramz opt_lbfgsb = scipy fmin_l_bfgs_b with maxfun = maxiter = 1000, factr 1e7, pgtol 1e-5 | scipy L-BFGS-B, maxiter 100, gtol sqrt(eps), ftol 0, maxcor 10; the library's own objective and gradient | softplus (Logexp) of each parameter | positive only, no upper bound |
| libgp | RProp (resilient backpropagation), 100 iterations, eps_stop 0, Delta0 0.1, Deltamin 1e-6, Deltamax 50, eta- 0.5, eta+ 1.2; keeps the best likelihood seen | N/A: libgp offers RProp and CG only, and RProp has no gradient tolerance | log ℓ, log sf, log sn (amplitude and std, not variances) | none |
| friedrich | N/A: no ARD kernel | N/A: no ARD kernel | - | - |

#### Exact GP

| dataset | library | splits | RMSE | NLPD | 95% interval | fit [s] | evaluations | iterations | ms / evaluation | NLML | memory [MiB] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| energy | gprx | 20/20 | 0.4796 ± 0.014 | 0.6993 ± 0.033 | 0.923 ± 0.0067 | 1.283 ± 0.013 | 116 ± 0.58 | N/A | 10.94 | -1013 ± 2.5 | 47.3 |
| energy | sklearn | 20/20 | 0.4773 ± 0.013 | 0.692 ± 0.029 | 0.924 ± 0.0077 | 10.26 ± 0.79 | 94.3 ± 7.1 | 53 ± 5.6 | 108.1 | -973.3 ± 6.1 | 225.1 |
| energy | gpytorch | 20/20 | 0.4773 ± 0.012 | 0.6894 ± 0.031 | 0.921 ± 0.0082 | 1.614 ± 0.036 | 111 ± 1.8 | 96.8 ± 1.6 | 14.2 | -968.5 ± 7.4 | 316.2 |
| energy | gpy | 20/20 | 0.4756 ± 0.013 | 0.6859 ± 0.031 | 0.923 ± 0.0075 | 10.24 ± 0.23 | 119 ± 2.7 | 99.2 ± 0.8 | 85.45 | -972 ± 7.6 | 245.9 |
| maunaloa | gprx | 1/1 | 8.749 | 4.838 | 0.358 | 1.484 | 130 | N/A | 11.41 | -1479 | 70.6 |
| maunaloa | sklearn | 1/1 | 8.495 | 4.856 | 0.369 | 16.54 | 120 | 100 | 137.8 | -1479 | 176.5 |
| maunaloa | gpytorch | 1/1 | 8.649 | 4.98 | 0.339 | 2.733 | 118 | 100 | 23.16 | -1479 | 522.9 |
| maunaloa | gpy | 1/1 | 8.717 | 5.072 | 0.332 | 17.98 | 116 | 100 | 155 | -1479 | 408.0 |
| snelson | gprx | 1/1 | N/A | N/A | N/A | 0.02802 | 23 | N/A | 1.218 | 89.81 | 11.5 |
| snelson | sklearn | 1/1 | N/A | N/A | N/A | 0.3259 | 126 | 18 | 2.587 | 89.81 | 109.8 |
| snelson | gpytorch | 1/1 | N/A | N/A | N/A | 0.1444 | 55 | 20 | 2.626 | 89.81 | 282.3 |
| snelson | gpy | 1/1 | N/A | N/A | N/A | 0.3382 | 50 | 19 | 6.765 | 89.81 | 142.4 |
| yacht | gprx | 20/20 | 0.7522 ± 0.081 | 1.01 ± 0.14 | 0.927 ± 0.009 | 0.8274 ± 0.11 | 327 ± 51 | N/A | 2.655 | -338.2 ± 20 | 13.9 |
| yacht | sklearn | 20/20 | 0.936 ± 0.074 | 1.411 ± 0.059 | 0.945 ± 0.0094 | 1.093 ± 0.079 | 89 ± 5.3 | 44.5 ± 1.6 | 11.58 | -270.2 ± 1.7 | 124.4 |
| yacht | gpytorch | 20/20 | 0.4157 ± 0.056 | 0.2 ± 0.099 | 0.916 ± 0.013 | 0.4499 ± 0.03 | 108 ± 6.5 | 68 ± 5.9 | 3.835 | -451.2 ± 27 | 287.3 |
| yacht | gpy | 20/20 | 0.3931 ± 0.055 | 0.1971 ± 0.1 | 0.91 ± 0.014 | 2.575 ± 0.14 | 120 ± 6 | 78.1 ± 4.7 | 21.12 | -460.6 ± 27 | 151.4 |

#### SGPR, 512 inducing points

| dataset | library | splits | RMSE | NLPD | 95% interval | fit [s] | evaluations | iterations | ms / evaluation | NLML | memory [MiB] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3droad | gprx | 1/1 | 8.823 | 3.597 | N/A | 5230 | 788 | N/A | 6637 | N/A | N/A |
| 3droad | gpytorch | 1/1 | 8.823 | 3.597 | N/A | 1694 | 129 | N/A | 1.314e+04 | N/A | N/A |
| 3droad | gpy | 1/1 | 8.823 | 3.597 | N/A | 3970 | 135 | N/A | 2.941e+04 | N/A | N/A |
| kin40k | gprx | 5/5 | 0.2863 ± 0.0029 | 0.1634 ± 0.0083 | 0.964 ± 0.0028 | 273.4 ± 96 | 394 ± 1.3e+02 | N/A | 681.8 | 1.039e+04 ± 59 | 890.7 |
| kin40k | gpytorch | 5/5 | 0.2865 ± 0.0028 | 0.1634 ± 0.0079 | 0.964 ± 0.0028 | 73.99 ± 4.9 | 86.4 ± 5.6 | 43 ± 2.3 | 854.9 | 1.039e+04 ± 59 | 1746.8 |
| kin40k | gpy | 5/5 | 0.2863 ± 0.0029 | 0.1634 ± 0.0083 | 0.964 ± 0.0028 | 231.8 ± 42 | 88 ± 16 | 42.4 ± 2.8 | 2643 | 1.039e+04 ± 59 | 1820.1 |
| naval | gprx | 20/20 | 1.582e-05 ± 1.1e-07 | -8.994 ± 0.00076 | 1 | 56.39 ± 8.8 | 252 ± 38 | N/A | 220.7 | -5.084e+04 ± 1.4 | 288.1 |
| naval | gpytorch | 20/20 | 1.707e-05 ± 4.4e-07 | 3784 ± 5.9e+02 | 0.419 ± 0.052 | 22.54 ± 1.3 | 82.8 ± 5 | 37.9 ± 3.8 | 272.7 | -5.075e+04 ± 20 | 745.1 |
| naval | gpy | 20/20 | 1.571e-05 ± 6e-07 | -9.681 ± 0.0095 | 0.94 ± 0.0036 | 72.17 ± 5.3 | 71.5 ± 5.3 | 7.65 ± 0.99 | 1021 | -5.624e+04 ± 63 | 675.9 |
| power_plant | gprx | 20/20 | 3.775 ± 0.039 | 2.752 ± 0.01 | 0.957 ± 0.0016 | 55.88 ± 8.6 | 309 ± 46 | N/A | 178 | -377 ± 11 | 232.3 |
| power_plant | gpytorch | 20/20 | 3.777 ± 0.04 | 2.753 ± 0.011 | 0.957 ± 0.0016 | 18.94 ± 1.4 | 82.8 ± 5 | 33.5 ± 0.87 | 222.3 | -377 ± 11 | 652.0 |
| power_plant | gpy | 20/20 | 3.097e+13 ± 3.1e+13 | 3.312e+40 ± 3.3e+40 | 0.958 ± 0.0026 | 54.58 ± 2.5 | 79.8 ± 3.5 | 33.6 ± 1.3 | 666.6 | -7.044e+45 ± 7e+45 | 570.0 |
| song | gprx | 1/1 | 0.5741 | 6.122 | N/A | 31.96 | 26 | N/A | 1229 | N/A | N/A |
| song | gpytorch | 1/1 | 0.5741 | 0.8639 | N/A | 579.9 | 47 | N/A | 1.234e+04 | N/A | N/A |
| song | gpy | 1/1 | 0.5741 | 0.8639 | N/A | 6355 | 47 | N/A | 1.352e+05 | N/A | N/A |
| wine_red | gprx | 20/20 | 0.6279 ± 0.0082 | 0.951 ± 0.014 | 0.939 ± 0.0048 | 10.78 ± 2.9 | 226 ± 56 | N/A | 45.09 | 1687 ± 2 | 63.5 |
| wine_red | gpytorch | 20/20 | 0.6276 ± 0.0082 | 0.9506 ± 0.014 | 0.939 ± 0.0047 | 6.03 ± 0.13 | 114 ± 0.82 | 100 | 51.29 | 1688 ± 2 | 376.5 |
| wine_red | gpy | 20/20 | 0.6275 ± 0.0082 | 0.9504 ± 0.014 | 0.94 ± 0.0048 | 22.8 ± 1 | 112 ± 3.1 | 97.3 ± 2.7 | 196.3 | 1688 ± 2 | 299.8 |

#### SVGP

| dataset | library | splits | RMSE | NLPD | 95% interval | fit [s] | evaluations | iterations | ms / evaluation | NLML | memory [MiB] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3droad | gprx | 1/1 | 11.18 | 3.834 | 0.948 | 34.49 | 1.15e+03 | N/A | 30.02 | N/A | 6236.7 |
| 3droad | gpytorch | 1/1 | 11.16 | 3.832 | 0.944 | 42.93 | 1.15e+03 | 1.15e+03 | 37.36 | N/A | 1284.3 |
| houseelectric | gprx | 1/1 | 0.05622 | -1.457 | 0.946 | 176.5 | 5.41e+03 | N/A | 32.65 | N/A | 29590.8 |
| houseelectric | gpytorch | 1/1 | 0.05872 | -1.419 | 0.945 | 203 | 5.41e+03 | 5.41e+03 | 37.55 | N/A | 5508.9 |
| kin40k | gprx | 1/1 | 0.6087 | 0.9093 | 0.921 | 3.181 | 108 | N/A | 29.45 | N/A | 638.8 |
| kin40k | gpytorch | 1/1 | 0.5975 | 0.8712 | 0.939 | 4.912 | 108 | 108 | 45.48 | N/A | 457.7 |
| song | gprx | 1/1 | 0.4669 | 0.6575 | 0.947 | 57.79 | 1.36e+03 | N/A | 42.53 | N/A | 8650.4 |
| song | gpytorch | 1/1 | 0.4709 | 0.666 | 0.954 | 55.96 | 1.36e+03 | 1.36e+03 | 41.17 | N/A | 3769.8 |

The marker is the same library in every figure: a blue circle is gprx, an orange square is scikit-learn, a green triangle is GPyTorch, and a yellow diamond is GPy.

**Prediction error, Exact GP**

Each column is a dataset. Top is RMSE, bottom is NLPD; lower is better. A marker is a library and the bar is the standard error across splits. Snelson has no test points, so that column is empty.

![Prediction error, Exact GP](https://raw.githubusercontent.com/YUKIKEDA/gprx/main/docs/bench/accuracy_matched.svg)

**Training time, Exact GP**

Top is the seconds spent training, bottom is how many times the library evaluated the likelihood and its gradient together. Both axes are logarithmic. Compare the seconds only where the counts match.

![Training time, Exact GP](https://raw.githubusercontent.com/YUKIKEDA/gprx/main/docs/bench/fit_time_matched.svg)

**Prediction error, SGPR**

Same reading as the Exact GP error figure. 512 inducing points. RMSE on top, NLPD below.

![Prediction error, SGPR](https://raw.githubusercontent.com/YUKIKEDA/gprx/main/docs/bench/accuracy_sgpr_matched.svg)

**Training time, SGPR**

Same reading as the Exact GP time figure. Seconds on top, likelihood-and-gradient counts below.

![Training time, SGPR](https://raw.githubusercontent.com/YUKIKEDA/gprx/main/docs/bench/fit_time_sgpr_matched.svg)

**Prediction error, SVGP**

Adam, learning rate 0.01, batch 1024, three passes over the data, in both libraries. GPy has no minibatch trainer, so it is absent. RMSE on top, NLPD below.

![Prediction error, SVGP](https://raw.githubusercontent.com/YUKIKEDA/gprx/main/docs/bench/accuracy_svgp_matched.svg)

**Training time, SVGP**

Seconds on top, Adam updates below. The update count matches, so the seconds are the speed.

![Training time, SVGP](https://raw.githubusercontent.com/YUKIKEDA/gprx/main/docs/bench/fit_time_svgp_matched.svg)

**Memory over time, energy**

The line is the resident memory of the whole process. The horizontal axis is seconds since the process started. A dotted line, in that library's color, is when training or prediction starts. Split 0.

![Memory over time, energy](https://raw.githubusercontent.com/YUKIKEDA/gprx/main/docs/bench/rss_timeline_energy_exact_s0_matched.svg)

**Memory over time, kin40k**

Same reading as the energy memory figure. SGPR with 512 inducing points, split 0.

![Memory over time, kin40k](https://raw.githubusercontent.com/YUKIKEDA/gprx/main/docs/bench/rss_timeline_kin40k_sgpr_s0_matched.svg)

**Mauna Loa predictions**

One panel per library. The line is the predictive mean, the band is the 95% interval, filled points are training data, and hollow points are held out.

![Mauna Loa predictions](https://raw.githubusercontent.com/YUKIKEDA/gprx/main/docs/bench/curve_maunaloa_matched.svg)

**Snelson predictions**

One panel per library. The line is the predictive mean and the band is the 95% interval. The points are the training data. Nothing is held out.

![Snelson predictions](https://raw.githubusercontent.com/YUKIKEDA/gprx/main/docs/bench/curve_snelson_matched.svg)
<!-- bench:end -->

### Reproduce

```text
just perf-real-full                                            # the comparison above, then this section
```

`--timeline` records the resident memory of the whole process every 10 ms. The raw output stays in `compare/perf/out/real/` and is not committed. `docs/bench/summary.json` holds the table numbers, the machine, the library versions, and the optimizer settings. Details: [`compare/perf/README.md`](https://github.com/YUKIKEDA/gprx/blob/main/compare/perf/README.md).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
