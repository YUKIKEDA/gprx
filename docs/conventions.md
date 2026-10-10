# Conventions

Where things live, for humans. Procedure: [CONTRIBUTING.md](../CONTRIBUTING.md). Public surface and meaning: [design.md](design.md). Enforcement: `.cursor/rules/`.

## Repository

Single crate `gprx`. Not a Cargo workspace. Do not create an empty file before the Issue that needs it.

Which module owns what, and which way imports point: [architecture.md](architecture.md). The saved directory: [persist-format.md](persist-format.md).

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
src/rng.rs                 crate-private SeededRng (Xoshiro256++)
src/workspace.rs           Workspace and QueryWorkspace. Do not pack them into one struct
src/gpr/                   one responsibility per file, no #[path]. Models import the shared layers above, never each other
  trainer.rs fitted.rs online.rs    Gpr, FittedGpr, OnlineGpr
  exact_fit.rs             ExactFit: every hyperparameter write, gradient, and Hessian
  objective.rs             GprObjective
  factor_store.rs          LltStore (FittedGpr) and LdltStore (OnlineGpr)
  shared.rs factor.rs      GprCore, Policies, and the shared predict body; Gram assembly and Cholesky
  distance.rs              fit, factor, predict, and online insert / delete of a DistanceKernel
  tests.rs
src/kernel/                leaves live here, not in src/*.rs
  compiled/                mod.rs, apply.rs, grad.rs, hess.rs, gram.rs, weighted.rs, coord.rs, leaf_table.rs. One dispatch over T: KernelScalar
                           supplied.rs: the compiled leaves on supplied distances
    tests/                 one file per concern (columns, compose, coord_deriv, coord_mode, cross, custom, fast_math, params, scalar, supplied)
  constant.rs linear.rs white.rs
  rbf.rs rbf_ard.rs matern.rs matern_ard.rs periodic.rs rq.rs rq_ard.rs    leaves: formulas over T, scans in mod.rs / ard.rs
  ard.rs                   shared checks and r² sums of the ARD leaves
  spec.rs term.rs dist.rs lengthscale.rs scalar.rs radial.rs leaf_params.rs
  tree.rs                  the Supply kind of a tree: NoSupply (coordinates) or supplied leaves
  supply.rs                slots, DistanceKernel, DistanceSource, the model kernels (ModelKernel, PointKernel)
  sources.rs               binding and checking supplied tables; the training d² stores; the sparse blocks
  simd/                    mod.rs (the lane helpers), stationary.rs, rbf_ard.rs, ard.rs, dist.rs, rows.rs. Every f64x4 loop of the leaves
src/optimizer/             mod.rs, logit.rs. lbfgs.rs neldermead.rs trust_region.rs fsa.rs. adam.rs does not implement Optimizer
src/persist/               config.rs kernel.rs registry.rs tensors.rs transform.rs sparse.rs (save / load of Sgpr, OnlineSgpr, Svgp)
                           distance.rs (the loaded models of a DistanceKernel)
src/sparse/                crate-private: SparseSpec / SparseCore and the inducing-point helpers Sgpr and Svgp share. Sgpr and Svgp do not import each other
src/sgpr/                  model.rs fitted.rs online.rs objective.rs distance.rs tests.rs
  factor/                  vfe.rs (assembly, weights, bound) derivatives.rs predict.rs loo.rs updates.rs (rank-1, inducing)
src/svgp/                  model.rs fitted.rs distance.rs tests.rs
  factor/                  assemble.rs (K_mm, A, q, ELBO) gradient.rs predict.rs adam.rs
src/transform/             target.rs input.rs pipeline.rs columnwise.rs. Do not split into leaves
tests/                     integration tests. Goldens only under compare/goldens/
tests/common/               check.rs (tolerance asserts) and problems.rs (Forrester / sphere, the distance baseline). Unit tests and benches include them with #[path]
tests/alloc.rs             harness = false: its checks run one after another (#494); a new check goes in CHECKS
benches/exact.rs           criterion
benches/distance.rs        supplied distances beside the coordinate path, at the same problem
compare/                   one Python environment (pyproject.toml, uv.lock; group `perf`). Run from here
  common/                  problems.py ops.py harness.py records.py timing.py rss.py, shared by the generators and perf/
  generate*.py             sklearn / GPyTorch / libgp goldens into goldens/
  perf/                    manual wall time and RSS: run*.py (python -m perf.run*), runners.py, the Python runners
  perf/gprx/               one runner binary gprx-perf (exact / online / sparse / sparse-online)
examples/fit_predict.rs
docs/                      design, architecture, persist-format (each with a .ja.md), roadmap, this file, adr/
docs/bench/                B1-1 summary.json and SVG figures (`just perf-real-report`), plus the Snelson sparse curve and online GIF (`just perf-visual`). Raw output stays in compare/perf/out/ and is not committed
```

`.dev/bench-log.md` is the local measurement log. It is not committed. Any other file an agent writes is deleted when that step no longer needs it (`.cursor/rules/scratch.mdc`). Decisions do not stay in `.dev/`.

`Workspace`, `QueryWorkspace`, `LltStore`, `LdltStore`, and faer types are crate-private.

Unit tests are `#[cfg(test)]` in the module. Integration tests are `tests/`.
