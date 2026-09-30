# Conventions

Where things live, for humans. Procedure: [CONTRIBUTING.md](../CONTRIBUTING.md). Public surface and meaning: [design.md](design.md). Enforcement: `.cursor/rules/`.

## Repository

Single crate `gprx`. Not a Cargo workspace. Do not create an empty file before the Issue that needs it.

Do not copy the public surface here or into `.cursor/rules/`. The current specification is [design.md](design.md).

```text
src/lib.rs                 public re-exports
src/data.rs                crate-private boundary checks and column-major packing for caller data
src/error.rs
src/internals.rs           bench / compare-perf hooks. Only with the bench-internals or insert-stages feature
src/likelihood.rs
src/linalg/                crate-private Cholesky, LDLT, solves, dense helpers, faer worker caps. Models do not define these
src/math.rs                Accurate / FastApprox
src/objective.rs           optimizer-facing traits only. Each model keeps its own adapter (gpr/objective.rs, sgpr/objective.rs)
src/param.rs
src/points.rs              PointId and the crate-private PointRegistry
src/policy.rs              runtime policies (distance cache, Cholesky buffer, KernelExp, jitter)
src/precision/             precision policies and refinement. Does not import a model: models pass their f64 reference as a closure
src/prediction.rs          Prediction, PredictiveCovariance, PredictOptions, VarianceKind
src/rng.rs                 crate-private SmallRng
src/workspace.rs           Workspace and QueryWorkspace. Do not pack them into one struct
src/gpr/                   one responsibility per file, no #[path]. Models import the shared layers above, never each other
  trainer.rs fitted.rs online.rs    Gpr, FittedGpr, OnlineGpr
  exact_fit.rs             ExactFit: every hyperparameter write, gradient, and Hessian
  objective.rs             GprObjective
  factor_store.rs          LltStore (FittedGpr) and LdltStore (OnlineGpr)
  shared.rs factor.rs      GprCore, Policies, and the shared predict body; Gram assembly and Cholesky
  tests.rs
src/kernel/                leaves live here, not in src/*.rs
  compiled/                mod.rs, apply.rs, grad.rs, hess.rs, gram.rs. One dispatch over T: KernelScalar
    tests/                 one file per concern (compose, coord_mode, custom, fast_math, params, scalar)
  constant.rs linear.rs white.rs
  rbf.rs rbf_ard.rs matern.rs matern_ard.rs periodic.rs rq.rs rq_ard.rs    leaves: formulas over T, scans in mod.rs / ard.rs
  ard.rs                   shared checks and r² sums of the ARD leaves
  spec.rs term.rs dist.rs simd.rs lengthscale.rs scalar.rs
src/optimizer/             mod.rs, logit.rs. lbfgs.rs neldermead.rs trust_region.rs fsa.rs. adam.rs does not implement Optimizer
src/persist/               config.rs kernel.rs registry.rs tensors.rs transform.rs sparse.rs (save / load of Sgpr, OnlineSgpr, Svgp)
src/sparse/                crate-private: SparseSpec / SparseCore and the inducing-point helpers Sgpr and Svgp share. Sgpr and Svgp do not import each other
src/sgpr/                  model.rs fitted.rs online.rs objective.rs tests.rs
  factor/                  vfe.rs (assembly, weights, bound) derivatives.rs predict.rs loo.rs updates.rs (rank-1, inducing)
src/svgp/                  model.rs fitted.rs tests.rs
  factor/                  assemble.rs (K_mm, A, q, ELBO) gradient.rs predict.rs adam.rs
src/transform/             target.rs input.rs pipeline.rs columnwise.rs. Do not split into leaves
tests/                     integration tests. Goldens only under compare/goldens/
tests/common/               check.rs (tolerance asserts) and problems.rs (Forrester / sphere). Unit tests and benches include them with #[path]
benches/exact.rs           criterion
compare/                   one Python environment (pyproject.toml, uv.lock; group `perf`). Run from here
  common/                  problems.py ops.py harness.py records.py timing.py rss.py, shared by the generators and perf/
  generate*.py             sklearn / GPyTorch / libgp goldens into goldens/
  perf/                    manual wall time and RSS: run*.py (python -m perf.run*), runners.py, the Python runners
  perf/gprx/               one runner binary gprx-perf (exact / online / sparse / sparse-online)
examples/fit_predict.rs
docs/                      design, roadmap, this file, adr/
```

`.dev/` is local scratch (measurement logs, reviews, drafts). It is not committed. Decisions do not stay there.

`Workspace`, `QueryWorkspace`, `LltStore`, `LdltStore`, and faer types are crate-private.

Unit tests are `#[cfg(test)]` in the module. Integration tests are `tests/`.
