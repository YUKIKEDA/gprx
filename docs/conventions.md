# Conventions

Where things live, for humans. Procedure: [CONTRIBUTING.md](../CONTRIBUTING.md). Public surface and meaning: [design.md](design.md). Enforcement: `.cursor/rules/`.

## Repository

Single crate `gprx`. Not a Cargo workspace. Do not create an empty file before the Issue that needs it.

Do not copy the public surface here or into `.cursor/rules/`. The current specification is [design.md](design.md).

```text
src/lib.rs                 public re-exports
src/error.rs
src/likelihood.rs
src/math.rs                Accurate / FastApprox
src/objective.rs
src/online.rs              crate-private OnlineWorkspace
src/param.rs
src/precision.rs
src/rng.rs                 crate-private SmallRng
src/workspace.rs           Workspace and QueryWorkspace. Do not pack them into one struct
src/gpr/                   Gpr and FittedGpr in model.rs. fitted.rs, factor.rs, online.rs, types.rs, tests.rs
src/kernel/                leaves live here, not in src/*.rs
  compiled/                mod.rs, apply.rs, grad.rs, hess.rs, gram.rs, f32_eval.rs, tests.rs
  constant.rs linear.rs white.rs
  rbf.rs rbf_ard.rs matern.rs matern_ard.rs periodic.rs rq.rs rq_ard.rs
  spec.rs term.rs dist.rs simd.rs lengthscale.rs scalar.rs
src/optimizer/             mod.rs, logit.rs. lbfgs.rs ncg.rs neldermead.rs fsa.rs newton.rs. adam.rs does not implement Optimizer
src/persist/               config.rs kernel.rs registry.rs tensors.rs transform.rs
src/sgpr/                  model.rs, fitted.rs, factor.rs, online.rs, tests.rs
src/svgp/                  model.rs, fitted.rs, factor.rs, tests.rs
src/transform/             target.rs input.rs pipeline.rs columnwise.rs. Do not split into leaves
tests/                     integration tests. Goldens only under compare/goldens/
benches/exact.rs           criterion
compare/                   sklearn and GPyTorch generation. perf/ is manual wall time and RSS
examples/fit_predict.rs
docs/                      design, roadmap, this file, adr/
.dev/                      bench-log.md, reviews/, issue-seed.json. Decisions do not stay here
```

`Workspace`, `QueryWorkspace`, `OnlineWorkspace`, and faer types are crate-private.

Unit tests are `#[cfg(test)]` in the module. Integration tests are `tests/`.
