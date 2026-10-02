# gprx implementation roadmap

Procedure: [CONTRIBUTING.md](../CONTRIBUTING.md). Design: [design.md](design.md).

Acceptance text stays on each Issue. This file keeps ID, title, Issue, and status.

## Current work

B1-1 ([#298](https://github.com/YUKIKEDA/gprx/issues/298), real-dataset benchmark and library comparison in the README) stays acceptance on #298: the measured tables and figures are not in yet. B1-6 is done. P5-5 ([#43](https://github.com/YUKIKEDA/gprx/issues/43), threshold for `DistanceCachePolicy::Auto`) stays set after Grill. It follows R4-1.

## Dependencies

```
M0 → 1a → 1b → 2 → 2b → 3
                     → 4
                after the phase-2 measurement → 5
R1 → R2 → R3 → R4 → R5 → R6
R4-1 → P5-5
```

Phase 3 follows 2b. Early phase 4 can run in parallel after 1b. Phase 5 cannot claim a speedup without `phase-2`. Each R row lists its own dependencies on its Issue.

## M0

| ID | Kind | Title | Issue | Status |
| --- | --- | --- | --- | --- |
| M0-1 | Task | Crate bootstrap (Cargo.toml, MIT OR Apache-2.0, justfile, CI yaml, gitignore, README stub) | [#1](https://github.com/YUKIKEDA/gprx/issues/1) | done |
| M0-2 | Spike | faer 0.24 `cholesky_in_place` round trip | [#2](https://github.com/YUKIKEDA/gprx/issues/2) | done |

## 1a

| ID | Kind | Title | Issue | Status |
| --- | --- | --- | --- | --- |
| P1A-1 | Task | `GprError` and `DoublePrecision`-only types | [#3](https://github.com/YUKIKEDA/gprx/issues/3) | done |
| P1A-2 | Task | `Workspace` (f64, `k_matrix` / `w_matrix` / `dist_cache` / `exp_buf` / faer scratch) | [#4](https://github.com/YUKIKEDA/gprx/issues/4) | done |
| P1A-3 | Feat | `GaussianLikelihood` (`θ=log(σn²)`, `∂K/∂θ=σn² I`) | [#5](https://github.com/YUKIKEDA/gprx/issues/5) | done |
| P1A-4 | Feat | `TargetTransform` and an X `Transform` (Identity / Standardize) | [#6](https://github.com/YUKIKEDA/gprx/issues/6) | done |
| P1A-5 | Feat | RBF kernel (`apply`/`grad`, `uplo=Lower`) | [#7](https://github.com/YUKIKEDA/gprx/issues/7) | done |
| P1A-6 | Feat | `KernelSpec` / `CompiledKernel` (RBF + Sum/Product + param flatten) | [#8](https://github.com/YUKIKEDA/gprx/issues/8) | done |
| P1A-7 | Feat | `Gpr`: `A=K+σn²I`, LLT, `α` | [#9](https://github.com/YUKIKEDA/gprx/issues/9) | done |
| P1A-8 | Feat | `predict` (mean, `VarianceKind::{Latent,Observation}`) | [#10](https://github.com/YUKIKEDA/gprx/issues/10) | done |
| P1A-9 | Feat | Negative MLL | [#11](https://github.com/YUKIKEDA/gprx/issues/11) | done |
| P1A-10 | Feat | Gradient via `W=ααᵀ-K⁻¹` | [#12](https://github.com/YUKIKEDA/gprx/issues/12) | done |
| P1A-11 | Task | Analytic integration tests (n=2/3 RBF) | [#13](https://github.com/YUKIKEDA/gprx/issues/13) | done |
| P1A-12 | Task | sklearn goldens (`compare/` + `just gen-goldens` + JSON) | [#14](https://github.com/YUKIKEDA/gprx/issues/14) | done |
| P1A-13 | Feat | Constant, Linear, White | [#15](https://github.com/YUKIKEDA/gprx/issues/15) | done |
| P1A-14 | Feat | Matern ν=1/2, 3/2, 5/2 | [#16](https://github.com/YUKIKEDA/gprx/issues/16) | done |
| P1A-15 | Feat | Periodic | [#17](https://github.com/YUKIKEDA/gprx/issues/17) | done |
| P1A-16 | Feat | Rational Quadratic | [#18](https://github.com/YUKIKEDA/gprx/issues/18) | done |
| P1A-17 | Task | Composite kernels and extra goldens | [#19](https://github.com/YUKIKEDA/gprx/issues/19) | done |
| P1A-18 | Task | criterion harness (`benches/exact.rs`, `just bench`) | [#44](https://github.com/YUKIKEDA/gprx/issues/44) | done |
| P1A-19 | Task | Allocation ratchet (`tests/alloc.rs`) | [#45](https://github.com/YUKIKEDA/gprx/issues/45) | done |
| P1A-20 | Feat | ARD lengthscale (RBF first) | [#54](https://github.com/YUKIKEDA/gprx/issues/54) | done |

## 1b

| ID | Kind | Title | Issue | Status |
| --- | --- | --- | --- | --- |
| P1B-1 | Feat | `Objective` and `GprObjective` | [#20](https://github.com/YUKIKEDA/gprx/issues/20) | done |
| P1B-2 | Feat | argmin L-BFGS adapter | [#21](https://github.com/YUKIKEDA/gprx/issues/21) | done |
| P1B-3 | Feat | `Gpr::fit` optimizes | [#22](https://github.com/YUKIKEDA/gprx/issues/22) | done |
| P1B-4 | Task | Parameter recovery tests | [#23](https://github.com/YUKIKEDA/gprx/issues/23) | done |
| P1B-5 | Docs | README, rustdoc, `examples/` | [#24](https://github.com/YUKIKEDA/gprx/issues/24) | done |
| P1B-6 | Task | sklearn goldens that include fit (Forrester / ARD) | [#80](https://github.com/YUKIKEDA/gprx/issues/80) | done |
| P1B-7 | Feat | Leave-one-out prediction | [#83](https://github.com/YUKIKEDA/gprx/issues/83) | done |

## 2

| ID | Kind | Title | Issue | Status |
| --- | --- | --- | --- | --- |
| P2-1 | Task | Read `phase-1b` and order the bottlenecks | [#25](https://github.com/YUKIKEDA/gprx/issues/25) | done |
| P2-2 | Feat | Distance cache | [#26](https://github.com/YUKIKEDA/gprx/issues/26) | done |
| P2-3 | Feat | Build kernels with Rayon. Split `thread_scratch` off before the parallel region | [#27](https://github.com/YUKIKEDA/gprx/issues/27) | done |
| P2-4 | Task | Lower the allocation ratchet to 0 on the hot path | [#28](https://github.com/YUKIKEDA/gprx/issues/28) | done |
| P2-5 | Feat | SIMD for isotropic RBF and distances | [#29](https://github.com/YUKIKEDA/gprx/issues/29) | done |
| P2-6 | Spike | Speed contribution of the NLML constant `(n/2) log(2π)` | [#60](https://github.com/YUKIKEDA/gprx/issues/60) | done |
| P2-7 | Feat | ARD distance cache | [#88](https://github.com/YUKIKEDA/gprx/issues/88) | done |
| P2-8 | Feat | `Gpr` / `FittedGpr` typestate | [#98](https://github.com/YUKIKEDA/gprx/issues/98) | done |
| P2-9 | Task | Close Phase 2 | [#97](https://github.com/YUKIKEDA/gprx/issues/97) | done |

## 2b

| ID | Kind | Title | Issue | Status |
| --- | --- | --- | --- | --- |
| P2B-1 | Feat | Optimizer knobs and `Gpr<Fixed>` | [#108](https://github.com/YUKIKEDA/gprx/issues/108) | done |
| P2B-2 | Feat | Choosing an argmin solver | [#114](https://github.com/YUKIKEDA/gprx/issues/114) | done |
| P2B-3 | Feat | `KernelTerm` and `KernelSpec::Custom` | [#117](https://github.com/YUKIKEDA/gprx/issues/117) | done |
| P2B-4 | Feat | `JitterPolicy` | [#119](https://github.com/YUKIKEDA/gprx/issues/119) | done |
| P2B-5 | Feat | Read, write, and Clone a fitted model | [#121](https://github.com/YUKIKEDA/gprx/issues/121) | done |
| P2B-6 | Feat | Arbitrary predictive covariance and posterior samples | [#123](https://github.com/YUKIKEDA/gprx/issues/123) | done |
| P2B-7 | Feat | Transform pipeline | [#127](https://github.com/YUKIKEDA/gprx/issues/127) | done |
| P2B-8 | Feat | Per-column input transforms | [#129](https://github.com/YUKIKEDA/gprx/issues/129) | done |
| P2B-9 | Feat | Swap in a user `Optimizer` | [#131](https://github.com/YUKIKEDA/gprx/issues/131) | done |
| P2B-10 | Feat | Fitted transforms as types | [#125](https://github.com/YUKIKEDA/gprx/issues/125) | done |
| P2B-11 | Feat | Distance cache only on the distance path | [#133](https://github.com/YUKIKEDA/gprx/issues/133) | done |
| P2B-12 | Feat | Product gradient with respect to points | [#135](https://github.com/YUKIKEDA/gprx/issues/135) | done |
| P2B-13 | Feat | Sum/Product of Dist and Points leaves | [#137](https://github.com/YUKIKEDA/gprx/issues/137) | done |
| P2B-14 | Feat | Save and load a fitted model | [#63](https://github.com/YUKIKEDA/gprx/issues/63) | done |
| P2B-15 | Feat | Example of a custom Optimizer | [#106](https://github.com/YUKIKEDA/gprx/issues/106) | done |
| P2B-16 | Spike | Wall time and RSS versus other libraries | [#103](https://github.com/YUKIKEDA/gprx/issues/103) | done |
| P2B-17 | Feat | NLML Hessian of `GprObjective` | [#109](https://github.com/YUKIKEDA/gprx/issues/109) | done |
| P2B-18 | Feat | `IncrementalRecompute` implementation | [#110](https://github.com/YUKIKEDA/gprx/issues/110) | done |
| P2B-19 | Feat | Share the `L`/`W` buffers during fit | [#111](https://github.com/YUKIKEDA/gprx/issues/111) | done |
| P2B-20 | Task | Split modules (keep transform; split compiled / FittedGpr) | [#116](https://github.com/YUKIKEDA/gprx/issues/116) | done |
| P2B-21 | Feat | Peak RSS versus libgp | [#142](https://github.com/YUKIKEDA/gprx/issues/142) | done |
| P2B-22 | Feat | MLL+grad at n=4096 (versus sklearn) | [#143](https://github.com/YUKIKEDA/gprx/issues/143) | done |
| P2B-23 | Feat | Speed / memory presets | [#148](https://github.com/YUKIKEDA/gprx/issues/148) | done |

## 3

| ID | Kind | Title | Issue | Status |
| --- | --- | --- | --- | --- |
| P3-1 | Spike | Measure `ldlt::delete_rows_and_cols_clobber` | [#30](https://github.com/YUKIKEDA/gprx/issues/30) | done |
| P3-2 | Feat | `OnlineWorkspace` and capacity growth | [#31](https://github.com/YUKIKEDA/gprx/issues/31) | done |
| P3-3 | Feat | Append insert (hand-rolled bordered LDLT) | [#32](https://github.com/YUKIKEDA/gprx/issues/32) | done |
| P3-4 | Feat | delete + `PointId` / `PointRegistry` | [#33](https://github.com/YUKIKEDA/gprx/issues/33) | done |
| P3-5 | Task | Property tests for random insert/delete | [#34](https://github.com/YUKIKEDA/gprx/issues/34) | done |
| P3-6 | Spike | Online insert check and timing versus libgp | [#176](https://github.com/YUKIKEDA/gprx/issues/176) | done |
| P3-7 | Feat | Make online insert faster than libgp | [#177](https://github.com/YUKIKEDA/gprx/issues/177) | done |

## 4

| ID | Kind | Title | Issue | Status |
| --- | --- | --- | --- | --- |
| P4-1 | Spike | Pick one of VFE or FITC | [#35](https://github.com/YUKIKEDA/gprx/issues/35) | done |
| P4-2 | Feat | `SparseGpr`, fixed inducing points | [#36](https://github.com/YUKIKEDA/gprx/issues/36) | done |
| P4-3 | Feat | Diagonal prediction and MLL | [#37](https://github.com/YUKIKEDA/gprx/issues/37) | done |
| P4-4 | Feat | Hyperparameter optimization (Z stays out of params) | [#38](https://github.com/YUKIKEDA/gprx/issues/38) | done |
| P4-5 | Spike | How to optimize inducing locations Z | [#184](https://github.com/YUKIKEDA/gprx/issues/184) | done |
| P4-6 | Feat | Make Z an optimization target | [#186](https://github.com/YUKIKEDA/gprx/issues/186) | done |
| P4-7 | Spike | Rank-1 update of the VFE factor | [#188](https://github.com/YUKIKEDA/gprx/issues/188) | done |
| P4-8 | Feat | Online learning for Sparse | [#190](https://github.com/YUKIKEDA/gprx/issues/190) | done |
| P4-9 | Spike | Incremental insert / delete of inducing points | [#192](https://github.com/YUKIKEDA/gprx/issues/192) | done |
| P4-10 | Feat | Adding and removing inducing points | [#193](https://github.com/YUKIKEDA/gprx/issues/193) | done |
| P4-11 | Spike | Sparse check against GPyTorch | [#196](https://github.com/YUKIKEDA/gprx/issues/196) | done |
| P4-12 | Spike | Sparse wall time and RSS | [#197](https://github.com/YUKIKEDA/gprx/issues/197) | done |
| P4-13 | Spike | Sparse online check against GPyTorch | [#198](https://github.com/YUKIKEDA/gprx/issues/198) | done |
| P4-14 | Spike | Sparse online wall time | [#199](https://github.com/YUKIKEDA/gprx/issues/199) | done |
| P4-15 | Feat | SVGP factor / ELBO / diagonal predict | [#201](https://github.com/YUKIKEDA/gprx/issues/201) | done |
| P4-16 | Feat | SVGP Adam / minibatch fit | [#202](https://github.com/YUKIKEDA/gprx/issues/202) | done |
| P4-17 | Feat | Rename `SparseGpr` to `Sgpr` | [#205](https://github.com/YUKIKEDA/gprx/issues/205) | done |
| P4-18 | Spike | Sparse wall time (versus GPy / GPyTorch) | [#209](https://github.com/YUKIKEDA/gprx/issues/209) | done |
| P4-19 | Spike | Sparse peak RSS (n=4096) | [#210](https://github.com/YUKIKEDA/gprx/issues/210) | done |
| P4-20 | Spike | Remaining Sparse joint (versus GPy) | [#214](https://github.com/YUKIKEDA/gprx/issues/214) | done |
| P4-21 | Spike | Remaining SVGP joint (versus GPy) | [#217](https://github.com/YUKIKEDA/gprx/issues/217) | done |

## 5

| ID | Kind | Title | Issue | Status |
| --- | --- | --- | --- | --- |
| P5-1 | Spike | Mixed-precision residual (`PromoteStorage` vs `ReevaluateKernel`) | [#39](https://github.com/YUKIKEDA/gprx/issues/39) | done |
| P5-2 | Feat | f32, f64, and mixed precision | [#40](https://github.com/YUKIKEDA/gprx/issues/40) | done |
| P5-4 | Feat | Opt-in `FastApprox` | [#42](https://github.com/YUKIKEDA/gprx/issues/42) | done |
| P5-5 | Task | Threshold for `DistanceCachePolicy::Auto` | [#43](https://github.com/YUKIKEDA/gprx/issues/43) | Set after Grill on #43 |

## R (codebase cleanup, parent [#226](https://github.com/YUKIKEDA/gprx/issues/226))

| ID | Kind | Title | Issue | Status |
| --- | --- | --- | --- | --- |
| R1-1 | Task | Shared numeric scalar trait for f32 / f64; remove `size_of` type checks | [#227](https://github.com/YUKIKEDA/gprx/issues/227) | done |
| R1-2 | Task | Collect linear algebra into a `linalg` module | [#228](https://github.com/YUKIKEDA/gprx/issues/228) | done |
| R1-3 | Task | Collect input validation and column-major packing into a `data` module | [#229](https://github.com/YUKIKEDA/gprx/issues/229) | done |
| R1-4 | Task | One place for shared test helpers and problem generators | [#230](https://github.com/YUKIKEDA/gprx/issues/230) | done |
| R1-5 | Task | Reclassify `GprError` (shape, length, overflow, unused variants) | [#231](https://github.com/YUKIKEDA/gprx/issues/231) | done |
| R2-1 | Task | One input struct for kernel evaluation entry points | [#232](https://github.com/YUKIKEDA/gprx/issues/232) | done |
| R2-2 | Task | Split stationary kernels into profile and driver; remove `f32_eval.rs` | [#233](https://github.com/YUKIKEDA/gprx/issues/233) | done |
| R2-3 | Feat | Custom kernels written once for f32 and f64 | [#234](https://github.com/YUKIKEDA/gprx/issues/234) | done |
| R2-4 | Task | Fix inconsistencies and leaks in the kernel and crate public surface | [#235](https://github.com/YUKIKEDA/gprx/issues/235) | done |
| R3-1 | Bug | `MixedPrecision` predict re-runs refinement and recompiles the kernel on every call | [#236](https://github.com/YUKIKEDA/gprx/issues/236) | done |
| R3-2 | Bug | Refinement ignores the stored factor and `JitterPolicy` | [#237](https://github.com/YUKIKEDA/gprx/issues/237) | done |
| R3-3 | Task | One precision-policy trait; refinement as a `Refiner` | [#238](https://github.com/YUKIKEDA/gprx/issues/238) | done |
| R4-1 | Task | Distance cache, Cholesky buffer, exp mode, and recompute strategy as runtime values | [#239](https://github.com/YUKIKEDA/gprx/issues/239) | done |
| R4-2 | Task | Factor abstraction (LLT / LDLT) and a shared model core for `FittedGpr` and `OnlineGpr` | [#240](https://github.com/YUKIKEDA/gprx/issues/240) | done |
| R4-3 | Task | One predict path | [#241](https://github.com/YUKIKEDA/gprx/issues/241) | done |
| R4-4 | Task | Uniform model state, rollback, and optional buffers | [#242](https://github.com/YUKIKEDA/gprx/issues/242) | done |
| R4-5 | Bug | `IncrementalRecompute` infers changed leaves from bit differences and allocates per call | [#243](https://github.com/YUKIKEDA/gprx/issues/243) | done |
| R4-6 | Task | Shrink the 48 `Loaded*` persist types | [#244](https://github.com/YUKIKEDA/gprx/issues/244) | done |
| R4-7 | Task | Reorganize `src/gpr` files | [#245](https://github.com/YUKIKEDA/gprx/issues/245) | done |
| R5-1 | Task | Put `Sgpr` / `Svgp` on the shared core, linalg, and precision; fix module dependencies | [#246](https://github.com/YUKIKEDA/gprx/issues/246) | done |
| R5-2 | Spike | Decide which Exact features `Sgpr` / `Svgp` should match | [#247](https://github.com/YUKIKEDA/gprx/issues/247) | done |
| R6-1 | Task | Deduplicate the `compare/` Python harness and problem definitions | [#248](https://github.com/YUKIKEDA/gprx/issues/248) | done |
| R6-2 | Task | Align `design.md` pseudo-code with the implementation | [#249](https://github.com/YUKIKEDA/gprx/issues/249) | done |

## R7 (codebase cleanup after #299–#386, parent [#387](https://github.com/YUKIKEDA/gprx/issues/387))

| ID | Kind | Title | Issue | Status |
| --- | --- | --- | --- | --- |
| R7-1 | Task | One SIMD layer and one parallel lower-triangle walk in `kernel` | [#388](https://github.com/YUKIKEDA/gprx/issues/388) | acceptance on #388 |
| R7-2 | Task | One plan for the joint gradient's kept Grams, tied to `θ` by type | [#389](https://github.com/YUKIKEDA/gprx/issues/389) | acceptance on #389 |
| R7-3 | Bug | f32-storage prediction: `Sgpr` recomputes in f64, `Svgp` stays in f32 | [#390](https://github.com/YUKIKEDA/gprx/issues/390) | acceptance on #390 |
| R7-4 | Task | One set of argmin adapters; one error kind for argmin failures | [#391](https://github.com/YUKIKEDA/gprx/issues/391) | acceptance on #391 |
| R7-5 | Task | `OnlineSgpr` undoes a failed update from one state value, not a field list | [#392](https://github.com/YUKIKEDA/gprx/issues/392) | acceptance on #392 |
| R7-6 | Task | Decode `precision` and `residual` into one type once when loading | [#393](https://github.com/YUKIKEDA/gprx/issues/393) | done |

## B (real-dataset comparison)

| ID | Kind | Title | Issue | Status |
| --- | --- | --- | --- | --- |
| B1-1 | Task | Real-dataset benchmark and library comparison in the README | [#298](https://github.com/YUKIKEDA/gprx/issues/298) | acceptance on #298 |
| B1-2 | Bug | `Svgp` Adam step costs O(n·m²), not O(batch·m²) | [#300](https://github.com/YUKIKEDA/gprx/issues/300) | done |
| B1-3 | Feat | `Sgpr` / `Svgp` fit with every built-in kernel (rectangular `∂K(Z,X)/∂θ` and its Hessian) | [#301](https://github.com/YUKIKEDA/gprx/issues/301) | done |
| B1-4 | Feat | `FreeInducing` coordinate derivatives and `Custom` cross derivatives for every kernel | [#302](https://github.com/YUKIKEDA/gprx/issues/302) | done |
| B1-5 | Bug | The Hessian solver (plain Newton) fails with different errors when a step leaves the bounds; the Hessian solver becomes `TrustRegion` | [#306](https://github.com/YUKIKEDA/gprx/issues/306) | done |
| B1-6 | Task | Extend the path matrix to Exact, Svgp, and the online models | [#308](https://github.com/YUKIKEDA/gprx/issues/308) | done |

## Intentionally out of scope

Adding or removing a bullet here is Grill → Issue (`.cursor/rules/workflow.mdc`). An agent does not add a row without that agreement.

- A homemade L-BFGS / quasi-Newton (call an argmin solver)
- Treating cloud CI as a completion condition while runners are limited
- Publishing to crates.io, promising an MSRV, or requiring coverage
