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
src/objective.rs
src/param.rs
src/precision.rs
src/rng.rs                 crate-private SmallRng
src/workspace.rs           Workspace and QueryWorkspace. Do not pack them into one struct
src/gpr/                   one responsibility per file, no #[path]
  trainer.rs fitted.rs online.rs    Gpr, FittedGpr, OnlineGpr
  exact_fit.rs             ExactFit: every hyperparameter write, gradient, and Hessian
  factor_store.rs          LltStore (FittedGpr) and LdltStore (OnlineGpr)
  shared.rs factor.rs      GprCore and the shared predict body; Gram assembly and Cholesky
  points.rs prediction.rs policy.rs   PointId, prediction types, runtime policies
  tests.rs
src/kernel/                leaves live here, not in src/*.rs
  compiled/                mod.rs, apply.rs, grad.rs, hess.rs, gram.rs. One dispatch over T: KernelScalar
    tests/                 one file per concern (compose, coord_mode, custom, fast_math, params, scalar)
  constant.rs linear.rs white.rs
  rbf.rs rbf_ard.rs matern.rs matern_ard.rs periodic.rs rq.rs rq_ard.rs    leaves: formulas over T, scans in mod.rs / ard.rs
  ard.rs                   shared checks and r² sums of the ARD leaves
  spec.rs term.rs dist.rs simd.rs lengthscale.rs scalar.rs
src/optimizer/             mod.rs, logit.rs. lbfgs.rs ncg.rs neldermead.rs fsa.rs newton.rs. adam.rs does not implement Optimizer
src/persist/               config.rs kernel.rs registry.rs tensors.rs transform.rs
src/sgpr/                  model.rs, fitted.rs, factor.rs, online.rs, tests.rs
src/svgp/                  model.rs, fitted.rs, factor.rs, tests.rs
src/transform/             target.rs input.rs pipeline.rs columnwise.rs. Do not split into leaves
tests/                     integration tests. Goldens only under compare/goldens/
tests/common/               check.rs (tolerance asserts) and problems.rs (Forrester / sphere). Unit tests and benches include them with #[path]
benches/exact.rs           criterion
compare/                   sklearn and GPyTorch generation. perf/ is manual wall time and RSS
examples/fit_predict.rs
docs/                      design, roadmap, this file, adr/
```

`.dev/` is local scratch (measurement logs, reviews, drafts). It is not committed. Decisions do not stay there.

`Workspace`, `QueryWorkspace`, `LltStore`, `LdltStore`, and faer types are crate-private.

Unit tests are `#[cfg(test)]` in the module. Integration tests are `tests/`.
