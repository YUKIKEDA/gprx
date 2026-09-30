English | [日本語](design.ja.md)

# gprx design

## 1. Purpose and scope

Build the most flexible and the fastest Gaussian process regression library in Rust. Flexible means user-defined kernels, preprocessing, exact and sparse inference, a swappable optimizer, and **adding and removing data points one at a time (online learning)**. Fast means minimizing allocations, using SIMD and multiple threads, and trading compute against memory by switching precision.

**Revision history**:

- Round 1: reviews from ChatGPT and Gemini. Fixed the mixed-precision residual, the confusion of jitter with observation noise, the allocation policy, and precision generics.
- Round 2: a second review. (1) The trace term of the MLL gradient and the `W` buffer. (2) The faer 0.24.4 Cholesky update API (LLT has no insert/delete). (3) `GaussianLikelihood` parameterization matches the gradient. (4) Flattened kernel parameters. (5) A `TargetTransform` for `y`. (6) The mixed-precision residual matrix and the jitter fallback. (7) Static dispatch for built-in kernels. (8) The meaning of predictive variance. Implementation order follows the roadmap in §13.
- Round 3: the public surface splits into `Gpr` (trainer) and `FittedGpr` (fitted). sklearn JSON is a numerical check only. Implemented in P2-8.

## 2. Architecture

```
input X, y
  → Transform pipeline (preprocess X: MinMax, Standardize, …)
  → TargetTransform (standardize y, and invert mean/variance at predict)
  → Likelihood (observation noise σn², a model parameter of its own)
  → CompiledKernel<T> (the plan compiled from a KernelSpec, plus a Workspace)
  → Gpr (trainer: kernel, likelihood, transforms, fit options)
       → Objective (likelihood and gradient; only during fit)
       → Optimizer (type parameter. Default `Lbfgs`. The slot is P2B-1. argmin solvers are P2B-2. A user `O` is `minimize`d through the same slot. The example is P2B-15)
       → fit(self) → FittedGpr | (Gpr, GprError)
  → FittedGpr (L, α, X. predict / predict_into / refit / loo / save)
       → persist: one directory (`config.json` + `model.safetensors`). `format_version` 1. `factor_kind` is required (`llt` / `ldlt`). `load` of `llt` is `FittedGpr<Fixed>`. `ldlt` is `OnlineGpr<Fixed>`. mmap when a factor is present. Retrain is `with_optimizer` → `refit`
       → OnlineGpr: `FittedGpr::into_online(self)` converts LLT→LDLT. An append `insert` exists only on `OnlineGpr`
       → Phase 4: Sgpr returns a fitted type the same way
```

Principles:

- **Identifiers name gprx / GPR concepts** (kernel, likelihood, θ, factorization, an interval on a positive parameter, …). Do not name another product, a test harness, or an unrelated domain
- **Static dispatch by default. `dyn` only at the extension point (a user-defined kernel)**
- **The hot path inside gprx does not allocate** (the phrase "zero allocations during fit" cannot be enforced inside a user kernel, so the rule is this one)
- **Precision is a compile-time generic**
- **Numerical stabilization (jitter) is separate from the model parameter (observation noise)**

## 3. Linear algebra: faer

Depend on **faer 0.24.x** (0.24.4 was current when this was written). Stride and view constraints of `Mat<T>` follow the pinned API. This document does not freeze a layout.

Pure Rust, at or above OpenBLAS / LAPACK / Eigen, with Rayon parallelism in the same class as OpenMP / TBB.

- `Mat<T>` is column-major. **Kernel SIMD that assumes a contiguous stride is written only after checking the real `MatRef` / `MatMut` stride**
- Batch-fit Cholesky is `llt::factor::cholesky_in_place` (lower-triangular LLT, in place)
- Dynamic regularization (jitter) is built in as `LltRegularization`. **It is numerical stabilization only, and it is not the GPR observation noise (a model parameter)** (§4.0)
- `Mat` supports capacity-based reallocation (used by online learning in §11)
- **Cholesky update API, checked on faer 0.24.4**:
  - `llt::update` has only `rank_r_update_clobber`. **LLT has no high-level row/column insert/delete**
  - `ldlt::update::delete_rows_and_cols_clobber(LD, indices: &mut [usize], ...)` exists and deletes several rows at arbitrary indices
  - `ldlt::update::insert_rows_and_cols_clobber` is not public (only `insert_rows_and_cols_clobber_scratch`. The body is private)
  - Online learning follows §11 (append is hand-rolled, delete uses the LDLT API). `delete_rows_and_cols_clobber` matches a full LDLT rebuild in P3-1
- `llt::update::rank_r_update_clobber` / `ldlt::update::rank_r_update_clobber`: rank-r update, only when ΔK is low rank (§5.4.1)

## 4. Precision, and separating noise from jitter

### 4.0 Observation noise is not jitter

Separate "observation noise σn²" (a GPR model parameter, optimized) from "jitter" (a numerical offset that keeps Cholesky positive definite).

```rust
/// Model likelihood. Observation noise lives here and is optimized.
/// Parameters are get/set on the same flat array as the optimizer.
trait Likelihood<T: Scalar>: Send + Sync {
    fn num_params(&self) -> usize;
    fn get_params(&self, out: &mut [T]);
    fn set_params(&mut self, params: &[T]);
    fn add_noise_diag(&self, k_diag: &mut [T]); // K += σn² I (add to the diagonal)
    /// Write the diagonal of ∂K/∂θ_{param_idx} into dK_diag.
    /// θ uses the same parameterization as get/set_params.
    fn noise_grad_diag(&self, dK_diag: &mut [T], param_idx: usize);
}

/// θ = log(σn²). The positive constraint is the log parameterization.
/// σn² = exp(θ), so ∂K/∂θ = exp(θ) I = σn² I.
/// ∂K/∂σn = 2σn I is the formula when the standard deviation σn is the parameter. This crate does not use it.
struct GaussianLikelihood<T: Scalar> {
    log_noise_variance: T,
}

impl<T: Scalar> GaussianLikelihood<T> {
    fn noise_variance(&self) -> T { self.log_noise_variance.exp() }
}

/// Numerical stabilization only. It does not touch the model parameter (observation noise).
struct NumericalStability {
    policy: JitterPolicy,
}

enum JitterPolicy {
    Fixed(f64),
    Adaptive {
        initial: f64,
        multiplier: f64,  // each retry does jitter *= multiplier
        max_retries: usize,
        max_jitter: f64,
    },
}
```

`GaussianLikelihood::num_params()` is 1. `get_params` / `set_params` read and write `log_noise_variance` on a length-1 slice. `noise_grad_diag` fills the diagonal with `exp(θ)`.

`A = K + Likelihood.noise_diag` is **the matrix of the linear system that is actually solved** (the GPR model).

**Where jitter applies**:

- `JitterPolicy` is used **only when Cholesky itself fails**. A factor that succeeded is the factor of `A + j I`, and the solution in hand is `(A + j I)^{-1} y`. The `j` that was used is logged and kept on `CholeskyFailed` / `FitResult`.
- Do not iteratively refine that factor back onto the original `A`. The gap between the preconditioner `LLᵀ ≈ A + jI` and the target `A` grows, and the contraction `||I - (LLᵀ)^{-1} A||` can exceed 1 and diverge (§4.2).

### 4.1 Precision: f32 / f64 / mixed

The goals are both "less memory" and "more speed". Use mixed-precision iterative refinement, and **split where it applies between fit and predict**.

```rust
trait PrecisionPolicy {
    type Storage: Scalar;
    type Refine: Scalar;
}
struct MixedPrecision;  // Storage=f32, Refine=f64
struct SinglePrecision; // Storage=f32, Refine=f32
struct DoublePrecision; // Storage=f64, Refine=f64
```

**Limit of the split**: `log|K| = 2Σlog(L_ii)` in the marginal log likelihood, and the trace term `Tr(K⁻¹∂K/∂θ)` in the gradient, are not made more accurate by refining `α = K⁻¹y` (they depend on the f32 diagonal of `L` itself). The default for fit (the hyperparameter loop), which includes those terms, is **`DoublePrecision`**. `MixedPrecision` mainly targets predict, where the hyperparameters are fixed and the linear solve for `α` is the whole job. Using `MixedPrecision` during fit assumes a separate accuracy check of `log|K|` and the trace term (§14).

Steps (predict, or a solve at a fixed kernel):

1. Factor `A = K + Likelihood.noise_diag` in place with `cholesky_in_place::<f32>`, still in f32 (only jitter regularization inside)
2. `alpha_0 = solve(L, y)` with the f32 `L`
3. Compute the residual in f64. **Two ways to build the residual matrix `A_resid`**. Memory and accuracy trade off:
   - **`PromoteStorage` (default)**: promote the stored f32 `A` to f64 and set `r = y_f64 - A_f32→f64 @ alpha`. This refines the solution of "the linear system held in f32". It is not iterative refinement against the true f64 kernel matrix. It does not keep a separate f64 `A`, so it matches the memory goal.
   - **`ReevaluateKernel`**: reevaluate the kernel in f64 on every residual matvec. `A_f64` is not stored. Each iteration pays an O(n²) kernel evaluation, and the system is closer to the true f64 system.
   - Refinement runs on the factor that fit (or an online update) left behind: `α₀` is that factor's solve of `y`, and each correction solves through it. The system is `A + (σn² + j) I` with the jitter `j` the factor retry added (`0` without a retry); fit records `j`, online inserts add the same `j`, and a saved factor stores it. There is no second f32 factorization. A converged `PromoteStorage` α is checked once against the f64 system (kernel evaluated in column blocks, no `n×n` f64 matrix); when `κ(A) u_f32` is large and the check fails, α falls back to the f64 Cholesky solution of the same system. The f64 fallback retries with the model's `JitterPolicy` and reports the caller's stage (R3-2, [#237](https://github.com/YUKIKEDA/gprx/issues/237)).
   - Storing the whole f64 `A` contradicts the memory goal, so it is not used.
4. `delta = solve(L, r)` with the f32 `L`, then `alpha_1 = alpha_0 + delta`
5. Repeat a few times until convergence

Implementation priority: the omitted precision is `DoublePrecision` (Storage = f64, Refine = f64, the current f64 path). `SinglePrecision` is Storage = f32, Refine = f32, runs the same steps in f32, and uses the factor as-is. It has no residual type parameter. `MixedPrecision<R = PromoteStorage>` is Storage = f32, Refine = f64. It factors in f32 and iteratively refines only the predictive α. MLL and the gradient during training use that precision's factor. Iterative refinement does not run inside the training loop. The residual type parameter exists only on `MixedPrecision`. `PromoteStorage` subtracts with the stored f32 matrix. `ReevaluateKernel` recomputes the kernel in f64. Both stay. The omission is `PromoteStorage`. There is no flag and no in-code alias. On Forrester `n=1024`, the release median is 22.40 ms for the type that subtracts the stored f32 matrix and 48.77 ms for the type that recomputes the kernel in f64, which is outside 5%. The targets are Exact, `Sgpr`, `Svgp`, and the online path that f64 already has. Optimization covers every optimizer that exists in f64. P5-2 ([#40](https://github.com/YUKIKEDA/gprx/issues/40)).

### 4.2 Convergence parameters for mixed-precision refinement

From classical iterative refinement (Higham), with factorization precision u_f (f32 ≈ 1.19×10⁻⁷) and refinement precision u_r (f64 ≈ 2.22×10⁻¹⁶), the rate depends on κ(A)·u_f. **The actual stopping test uses the measured residual, not the theoretical value.**

The parameters are fixed crate constants in `src/precision/refine.rs`, not a public configuration:

| Parameter | Value |
| --- | --- |
| Most corrections | 10 |
| Relative tolerance | `10 · dim · u_r` (`u_r = f64::EPSILON`), on the measured residual |
| Stagnation | two consecutive residual-norm ratios above `0.9` |
| Not converged | the `f64` solution of the same system (never an error) |

Stopping test: `||r_k||∞ / (||B||∞ ||w_k||∞ + ||b||∞) < 10 · dim · u_r`. One loop (`refine` over a `RefineSystem`) serves Exact `α`, Sgpr weights, and the Svgp triangular solve; each system supplies its residual, the solve through its stored factor, and its `f64` fallback (R3-3, [#238](https://github.com/YUKIKEDA/gprx/issues/238)). Refinement never returns a convergence error: the fallback is always the `f64` solve.

**Do not raise jitter when IR fails to converge.** Raising only the factorization jitter widens the gap between the preconditioner `LLᵀ` and the target `A`, and IR can diverge. A failed IR falls back to the `f64` solution. Adaptive jitter stays reserved for Cholesky failure, as in §4.0.

**Standing**: the theory holds, and checking the parameters on a real workload is still open (§14).

## 5. Kernels

### 5.1 Spec versus evaluator, and precision generics

`KernelSpec` (the declaration) is a precision-independent type-erased form. Parameters are always `f64` (values a user writes and reads should not depend on precision). `CompiledKernel<T>` (the evaluator) is compiled per `PrecisionPolicy::Storage`, and the inner arithmetic is `T`.

The optimizer sees only a flat `params: &[T]`. A composite kernel needs a map back to the leaves.

```rust
struct ParameterId(usize);
struct LeafId(usize);

struct ParameterBinding {
    id: ParameterId,
    leaf_id: LeafId,
    local_index: usize,
}

enum KernelSpec {
    Leaf(Box<dyn KernelTermSpec>),
    Sum(Box<KernelSpec>, Box<KernelSpec>),
    Product(Box<KernelSpec>, Box<KernelSpec>),
}

trait KernelTermSpec: Send + Sync {
    fn num_params(&self) -> usize;
    fn get_params(&self, out: &mut [f64]);
    fn set_params(&mut self, params: &[f64]);
    fn compile<T: Scalar>(&self) -> CompiledTerm<T>;
}

impl KernelSpec {
    fn num_params(&self) -> usize { /* sum over leaves */ }
    fn get_params(&self, out: &mut [f64]);
    fn set_params(&mut self, params: &[f64]);
    fn parameter_bindings(&self) -> Vec<ParameterBinding>;
    fn compile<T: Scalar>(&self) -> CompiledKernel<T>;
}
```

What P5-1 shipped: leaf parameters stay f64. The public compute scalar is `CompiledKernel<T = f64>`. `compile()` with the type omitted is f64. `compile_as::<T>()` makes apply, gradient, Hessian, built-in leaves, sum, product, and user leaves the same operation in f32 and f64. The f64 distance cache and SIMD stay on the f64 side. f32 is the same formula, scalar. There is no conversion that swaps an f32 value with an f64 value.

Since R2-2 / R2-3 every built-in leaf is one implementation over `T: KernelScalar`, and `CompiledKernel<T>` has one dispatch. f64 SIMD paths are reached through a scalar hook that returns an f64 view only when `T = f64`. A user leaf is one `impl<T: KernelScalar> KernelTerm<T>`: `KernelScalar` carries the arithmetic and `exp` / `ln` / `sqrt` / `powf` / `sin` / `cos` a formula needs, and the generic built-in leaves can be called from it. `CustomKernel::new` keeps the bound `KernelTerm<f64> + KernelTerm<f32>`, which one generic impl satisfies.

```rust
/// Evaluator. Built-ins are an enum (static dispatch). Only a user term is dyn.
enum CompiledKernel<T: Scalar> {
    Rbf(RbfKernel<T>),
    Matern(MaternKernel<T>),
    Periodic(PeriodicKernel<T>),
    Sum(Vec<CompiledKernel<T>>),
    Product(Vec<CompiledKernel<T>>),
    Custom(Box<dyn KernelTerm<T>>),
}

enum Triangle { Lower, Upper, Full }

trait KernelTerm<T: Scalar>: Send + Sync {
    fn distance_kind(&self) -> DistanceKind;
    /// `uplo` selects the triangle to write. The default contract is Lower.
    /// faer `cholesky_in_place` reads only the lower triangle, so filling Full about doubles the kernel evaluation.
    fn apply(&self, dist: MatRef<T>, out: MatMut<T>, uplo: Triangle);
    fn grad(&self, dist: MatRef<T>, dK: MatMut<T>, param_idx: usize, uplo: Triangle);
    fn rank_structure(&self) -> KRankStructure { KRankStructure::Dense }
    /// ∂K(X1, X2)/∂(X2_{*, dim}) for one dimension, in bulk.
    /// A per-point vtable call (m×d times) blocks SIMD, so the unit is a dimension, not one coordinate.
    fn grad_wrt_coord_dim(&self, x1: MatRef<T>, x2: MatRef<T>, dK: MatMut<T>, dim: usize) -> Result<(), GprError> {
        Err(GprError::CoordGradientUnsupported)
    }
}
```

`KernelSpec` composes through operator overloads. `KernelTermSpec` is object-safe, so a user kernel plugs in by implementing it. Conversion to `CompiledKernel<T>` happens once, at the start of fit.

**Lengthscale**: isotropic is a scalar `ℓ` (`θ=log(ℓ)`). ARD is a per-dimension `ℓ_d` (`θ_d=log(ℓ_d)`). The targets are stationary kernels that have a lengthscale (RBF / Matern / RQ). P1A-20 fixed the slot on RBF, and P1A-14 / P1A-16 use the same slot. The Periodic lengthscale stays a scalar.

The ARD squared distance is `r² = Σ_d (x_d - x'_d)² / ℓ_d²`. It matches isotropic when every `ℓ_d` is equal. `∂K/∂θ_d` needs the per-dimension difference, so the isotropic squared-distance matrix is not enough. The `n×n×d` cache is §5.2 / P2-7 (raw `(Δx_d)²`. An `r²` that already includes ℓ is not stored). P1A-20 builds it from coordinates every time. P2-2 is the isotropic n×n.

A user kernel (`Custom`) is encouraged not to allocate on the hot path, and that is not enforced (§2). Phase 1 does not pass `Workspace` into a user kernel. There is no pair of a safe API and an unsafe fast API.

On the hot path (the double loop of distances and kernel evaluation) `CompiledKernel` is dispatched with `match`. Only `Custom` goes through the vtable. That matches §2: static dispatch by default, `dyn` only at the extension point.

The optimizer's parameter array is concatenated in this order:

```
[kernel_params | likelihood_params]
```

Inducing locations Z of Sparse GPR are not an optimization target in the early Phase 4 (§6.1).

### 5.2 Distance cache and cache policy

Isotropic kernels have raw coordinate differences that do not change during fit, so the distance tensor is computed once and reused.

```rust
enum DistanceKind { SqEuclidean, SqEuclideanARD, Periodic { period: usize } }

/// Cached intermediate. This, not DistanceKind, is the stored value.
enum DistanceCache<T: Scalar> {
    None,
    SquaredEuclidean(Mat<T>),     // n×n, isotropic RBF/Matern, …
    SquaredEuclideanArd(Mat<T>),  // stands in for n×n×d. Memory is about d times K
    Periodic(Mat<T>),             // already through the periodic map, such as sin²(π|x-x'|/p). Not a raw squared distance
}
```

Periodic is not squared Euclidean. ARD needs a per-dimension difference. The cache unit is the intermediate above, per kernel kind. Do not grow this enum on every new kernel. **The concrete layout is reconsidered when Phase 2 is implemented.**

**The cache policy is a benchmark-based policy, not a hard rule.**

```rust
enum DistanceCachePolicy {
    Cached,   // default: fill once per fit and reuse
    Uncached, // recompute from X on every kernel build
}
```

A theoretical reference, not a decision: an `(n,n,d)` tensor is `n²×d×sizeof(T)` bytes. `K` itself is also n²×sizeof(T) (about 200MB at n=5000, f64), and an ARD cache is `d` times that. In a typical GPR with d≪n the cache often does not pay for itself. The concrete `Auto` threshold is decided by a benchmark after implementation (§14). `Auto` is not a variant yet.

P2-2 ([#26](https://github.com/YUKIKEDA/gprx/issues/26)): `Cached` / `Uncached` sit on the existing `Workspace.dist_cache` (isotropic Dist/Either, n×n). The default is `Cached`. P5-5 adds `Auto`.

P2-7 ([#88](https://github.com/YUKIKEDA/gprx/issues/88)): the same `DistanceCachePolicy` covers the raw `(Δx_d)²` of ARD leaves. An `r²` that already includes ℓ is not stored. The public policy is not extended. `Workspace` looks at `n` and `d`. An ARD fit with Always (`Cached`) allocates once. Isotropic / Never (`Uncached`) stays empty (same as `kernel_scratch`). Layout is column-major `n × (n·d)`. Dimension `k` is columns `[k n, (k+1) n)`. Each block is lower triangular. The fill and RBF ARD `apply` / `grad` are Rayon + `wide::f64x4` (unit row stride). Matérn / RQ ARD read the same cache in scalar code. The required numbers are ARD RBF on the same fixed problem (`mll_and_grad_ard` / `fit_lbfgs_ard`, Always versus Never). `Auto` is P5-5. A train×test / LOO cache is outside P2-7. Evaluating a composite of Dist leaves and Points leaves is P2B-13 (P2-7 does not mix that into the cache path).

### 5.3 Building a CompiledKernel plan

Sum/Product are associative and commutative, so flatten-and-fold is enough. Public `KernelSpec *` reaches `grad` for both Dist leaves and Points leaves. A Sum/Product of a Dist leaf and a Points leaf (for example `RBF + Linear`) is evaluated mixed. `coord_mode` returns `Mixed`. Mixing is not a runtime error and not a type ban. A Dist leaf stays on distances. A Points leaf stays on coordinates.

1. Deduplicate distance caches: walk the tree and build the set of `DistanceKind`
2. Flatten: normalize `(A+B)+C` to `Sum(vec![A,B,C])`
3. Build the plan: a `BufAllocator` (free list) tracks `alloc()` / `free()`, and **the maximum live count is computed from the real plan**, which sets the Workspace size

**Correction**: "the buffer count only grows with nesting depth, and in practice never exceeds 3" is false. A composite such as `(A*B)*(C*D)` cannot reuse a buffer across siblings, so the count grows. Do not assume a fixed cap. Measure `WorkspacePlan { max_buffers, max_bytes }` while building the plan.

```rust
enum PlanOp {
    EvalLeafInto { term_id: usize, dist_id: usize, dst: BufId },
    AddLeafInto  { term_id: usize, dist_id: usize, dst: BufId },
    MulLeafInto  { term_id: usize, dist_id: usize, dst: BufId },
    AddBufInto   { src: BufId, dst: BufId },
    MulBufInto   { src: BufId, dst: BufId },
}
struct WorkspacePlan { max_buffers: usize, max_bytes: usize }
```

While running the plan, a built-in leaf calls the `CompiledKernel` enum arm directly. Only `Custom` delegates to `dyn KernelTerm`.

### 5.4 Partial updates (coordinate optimizers)

**Policy**: whether a fit rebuilds only the touched kernel leaves is not a type. It is derived at run time (R4-1 / [#239](https://github.com/YUKIKEDA/gprx/issues/239)): `O::USES_CHANGE_INDICES && buffer == CholeskyBuffer::Retain`. The leaf rebuild itself is P2B-18 ([#110](https://github.com/YUKIKEDA/gprx/issues/110)). In Exact GPR, Cholesky is O(n³), so a partial update only helps while building the kernel matrix.

```rust
trait Optimizer<P> {
    const USES_CHANGE_INDICES: bool = false; // FSA sets true
    // ...
}

trait IncrementalObjective: Objective {
    fn value_with_changes(&mut self, params: &[T], indices: &[usize]) -> Result<T, GprError>;
}
```

Changed indices are the `&[usize]` of `IncrementalObjective::value_with_changes`. They list every coordinate that differs from the previous evaluation on that objective, not from the optimizer's accepted point: after a rejected proposal, FSA passes the coordinate it reverted together with the new one (R4-5 / [#243](https://github.com/YUKIKEDA/gprx/issues/243)). The indices alone decide which leaves are rebuilt. The old sketch `ChangeSet { Vec<usize> }`, and guessing changes by a numeric difference of θ, are not shipped; an unlisted change is rejected with `IndexOutOfRange` instead of silently giving a wrong value. Empty, duplicate, and `i >= n_params` are `GprError` at the boundary. A full recompute is `Objective::value`. The default of `Objective::value_at_changes` is `value`. `GprObjective` implements `IncrementalObjective` for every optimizer and buffer. `value` / `value_at_changes` take the leaf path only when the flag above is set; otherwise they run the full joint evaluation.

There is no `with_recompute_strategy`. `CholeskyBuffer::Reuse` always rebuilds everything. `CholeskyBuffer::Retain` rebuilds leaves when the optimizer sets `USES_CHANGE_INDICES`. `with_optimizer` / `refit` re-derive the flag from the new optimizer. `Gpr<Fixed>::factor` is one full pass. L-BFGS / NCG / Nelder–Mead / Newton keep the default `false`. FSA sets `true`. The first evaluation and a restart use `value`. One coordinate step uses `value_at_changes`.

The leaf rebuild caches only compiled leaves, and reapplies only the leaves a changed index touches. The per-leaf Grams (`L · n²` for `L` leaves) and the reused dirty flags and previous `θ` live on `GprObjective` for the length of one `fit` / `refit`, not on the Workspace. The tree combination and **Cholesky are full every time**. There is no low-rank update. No new `n×n` is added to Workspace. There is no runtime NotImplemented.

#### 5.4.1 Leaf rebuilds and the faer update API

Row/column insert/delete is for adding and removing data points (§11). It cannot express a hyperparameter change. `rank_r_update_clobber` is usable only when the hyperparameter change makes a low-rank `ΔK` (an amplitude change of a linear kernel term, a global scale change, and similar).

```rust
enum KRankStructure { Scalar, LowRank(usize), Dense }
```

The default `Dense` puts a user kernel on the safe side.

### 5.5 Preprocessing pipeline

Split X and y. When a GPR has no mean function, **standardizing y to mean 0 and variance 1 is the basic numerical step**. Predictions are mapped back to the original scale.

```rust
trait Transform {
    fn fit(&mut self, x: MatRef<f64>);
    fn apply(&self, x: MatMut<f64>);
}
struct Pipeline(Vec<Box<dyn Transform>>);

trait TargetTransform<T: Scalar>: Send + Sync {
    fn fit(&mut self, y: &[T]);
    fn transform(&self, y: &mut [T]);
    fn inverse_transform_mean(&self, mean: &mut [T]);
    fn inverse_transform_variance(&self, var: &mut [T]);
}

struct IdentityTarget<T>(PhantomData<T>);
struct StandardizeTarget<T: Scalar> { mean: T, std: T }
struct MinMaxInput { /* per-column min/max, default range [0, 1] */ }
struct MinMaxTarget { /* y min/max, default range [0, 1] */ }
/// P2B-8: length d. Per column: Identity / Standardize / MinMax / user
struct ColumnwiseInput { maps: Vec<Box<dyn Transform>> }
```

The default `Gpr` is Identity. When the mean function is zero, `StandardizeTarget` is the basic numerical step. `MinMaxInput` / `MinMaxTarget` scale to an interval (default `[0, 1]`). An unfitted `transform` / `apply` cannot happen: the types do not allow it. A series of maps is `Pipeline` (`X`) and `TargetPipeline` (`y`). A one-step `with_*` stays as it is. Inputs can be per column with `ColumnwiseInput` (a uniform column is MinMax, a near-normal column is Standardize. A length other than `d` is an error). `src/transform/` is `input.rs` / `target.rs` / `pipeline.rs` / `columnwise.rs`. Whether to split into leaf files is P2B-20 ([#116](https://github.com/YUKIKEDA/gprx/issues/116)). Judge by whether the adapter is independent, not by line count. `predict` computes latent or observation variance internally, then returns through `inverse_transform_mean` / `inverse_transform_variance`. For an affine `y' = (y - a)/s`, the inverse variance is `Var(y) = s² Var(y')`.

## 6. GP model: swapping exact and sparse

Training and inference are different types. An unfitted `predict` is not on the public API. sklearn's same-object `fit` / `predict` is a numerical-check target, not the public contract. `Objective` borrows `Gpr` only during `fit` and is not coupled to the fitted values.

```rust
/// Kernel, likelihood, transforms, optimizer settings. Unfitted.
struct Gpr { /* FitOptions, DistanceCachePolicy, transforms */ }

impl Gpr {
    fn fit(self, x: &[f64], n_rows: usize, n_cols: usize, y: &[f64])
        -> Result<FittedGpr, (Self, GprError)>;
}

/// Fitted. L, α, X, kernel, likelihood, transforms. No W and no L-BFGS state.
struct FittedGpr { /* … */ }

impl FittedGpr {
    fn predict(&self, xs: &[f64], n_rows: usize, n_cols: usize)
        -> Result<Prediction, GprError>;
    fn predict_into(
        &mut self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        out: &mut Prediction,
    ) -> Result<(), GprError>;
    fn refit(&mut self) -> Result<(), GprError>;
    fn log_marginal_likelihood(&self) -> Result<f64, GprError>;
    fn loo_predict(&self) -> Result<Prediction, GprError>;
}
```

`Gpr` (Exact) and `Sgpr` (P4-2) each return their own fitted type. Hyperparameter optimization goes through `Objective` (§9) and only during `fit`.

```rust
enum VarianceKind {
    Latent,       // variance of the latent f* (no noise)
    Observation,  // variance of the observation y* (includes σn²). Default
}

struct Prediction<T: Scalar> {
    mean: Vec<T>,
    variance: Vec<T>,
    variance_kind: VarianceKind,
}

struct PredictOptions {
    variance_kind: VarianceKind, // default Observation
}
```

Diagonal variance is the default from Phase 1. Covariance between queries and a posterior sample are the P2B-6 opt-in (not computed by default). `predict` switches the meaning of variance with `PredictOptions`. Unspecified is `Observation` (what a user usually wants is the noisy predictive variance).

### 6.1 Inducing-point cache for Sparse GPR

The Sparse approximation is VFE. The reason is [ADR 0002](adr/0002-sparse-vfe.md). FITC is not shipped. SVGP is a separate public type (`Svgp` / `FittedSvgp`). The reason is [ADR 0006](adr/0006-sparse-svgp.md). `Svgp<Fixed>::factor` LLTs `K_mm` at the caller's `Z` and places a whitened `q(u)` at the prior (mean 0, `L = I`). `Svgp<Adam>::fit` starts from that prior and moves kernel `θ`, likelihood `θ`, and the whitened `q` with minibatch Adam. `Adam` is not an `Optimizer`. `FittedSvgp` returns diagonal `predict` / `predict_with`, `neg_elbo`, and a full-data `value_and_gradient_into`. At the optimal `q` (Titsias) it matches `FittedSgpr` at the same `θ`, `X`, and `Z`. The public types are `Sgpr` / `FittedSgpr`. The default is `Sgpr<Lbfgs, FixedInducing>`. `fit` searches kernel and likelihood `θ`. `Sgpr<Fixed, I>::factor` LLTs `K_mm = k(Z, Z)` at the caller's inducing locations `Z`. By default `Z` is not in params. `fit` after `with_inducing(FreeInducing)` moves kernel `θ`, likelihood `θ`, and column-major `Z` together in the same `Optimizer`. `FittedSgpr` returns diagonal `predict` / `predict_with`, `neg_log_marginal_likelihood` (the negative VFE ELBO), `value_and_gradient_into`, and `hessian_into` (row-major `p×p`). At `Z = X` it matches Exact `Gpr<Fixed>::factor`. There is no k-means. The batch external check is P4-11 (collapsed GPyTorch SGPR / whitened-prior SVGP at the same initial θ, relative `1e-8`). The online external check is P4-13 (collapsed GPyTorch SGPR at the same initial θ, relative `1e-8`. `OnlineSgpr` insert / delete / insert_inducing / delete_inducing. Each stage is a full reassemble). Batch wall time and RSS are P4-12 (`just perf-sparse`. GPyTorch / GPy. CPU. No correctness gate). Online wall time is P4-14 (`just perf-sparse-online`. Our `Sgpr<Fixed>::factor` and a GPyTorch Titsias assemble. CPU. No correctness gate).

The diagonal of `K(X,X)` is invariant, so it is computed once and reused. `K(X,Z)` and `K(Z,Z)` must be recomputed whenever Z moves, but `m` (the number of inducing points) is small, so that cost is negligible next to the O(nm²) Cholesky and is not cached. The joint gradient and Hessian of `K(X,X)` sum the diagonal `∂k(x_i, x_i)/∂θ` in `O(n)`. Gradients of `K(Z,Z)` and `K(Z,X)` stay dense.

The gradient of inducing coordinates is `grad_wrt_coord_dim` (§5.1). An unsupported kernel returns `GprError::CoordGradientUnsupported`, not a panic. The default `FixedInducing` `fit` does not call this API. `FreeInducing` calls it once per dimension during joint optimization. The reason is [ADR 0003](adr/0003-sparse-z-joint.md).

**The default is: the caller passes Z, and the optimization targets are kernel hyperparameters and noise only.** Free Z switches with `FixedInducing` / `FreeInducing`. The same `Optimizer` moves kernel `θ`, likelihood `θ`, and column-major `Z` together. The interval is the raw coordinates of the training-`X` box, opened a little. L-BFGS history length is `p = p_θ + m×d`, and the extra storage is `history_size × m × d` values of `f64` (small next to the VFE `O(nm²)`, because `m` is small). Alternating is not shipped.

Online can add and remove both X and inducing points. `FittedSgpr::into_online` returns `OnlineSgpr<O>` (no inducing typestate). `insert` / `delete` update the VFE factor by the rank-1 of ADR 0004. `insert_inducing` / `delete_inducing` are [ADR 0005](adr/0005-sparse-inducing-update.md) (insert is a bordered LLT, delete is a trailing cholupdate). The identifier is `InducingId`. Coordinates come from the caller. `Z` is not in params. `set_params` and `refit` fully reassemble.

**Parity with Exact** (R5-2 / [#247](https://github.com/YUKIKEDA/gprx/issues/247)). The sparse models follow `Gpr` for the features below. Each row that is not shipped yet has its own row and Issue.

| Feature | `FittedSgpr` / `OnlineSgpr` | `FittedSvgp` | Row |
| --- | --- | --- | --- |
| Input / target transforms | yes | yes | R5-3 ([#281](https://github.com/YUKIKEDA/gprx/issues/281)) |
| `JitterPolicy` of `K_mm` (default stays `adaptive(1e-8, 10, 5, 1e-3)`, because `K_mm` has no noise, §4.0) | yes | yes | R5-4 ([#282](https://github.com/YUKIKEDA/gprx/issues/282)) |
| `predict_into` with no allocation after warmup | yes | yes | R5-5 ([#283](https://github.com/YUKIKEDA/gprx/issues/283)) |
| Predictive covariance and posterior sample | yes | yes | R5-6 ([#284](https://github.com/YUKIKEDA/gprx/issues/284)) |
| Leave-one-out | yes | no | R5-7 ([#285](https://github.com/YUKIKEDA/gprx/issues/285)) |
| Save and load | yes | yes | R5-8 ([#286](https://github.com/YUKIKEDA/gprx/issues/286)) |

SVGP has no leave-one-out. The collapsed VFE `q(u)` is the closed-form optimum, so leaving out point `i` is a rank-1 downdate of `A = K_mm + σ⁻² K_mn K_nm` (`O(m²)` per point at fixed `θ` and `Z`). The SVGP `q(u)` is a variational parameter that minibatch Adam fitted to all points. Leaving out a point means fitting `q(u)` again, and there is no closed form. Reusing the fitted `q(u)` is not a leave-one-out prediction, so it is not offered under that name.

### 6.2 MLL and gradient of `Gpr`

Without an algorithm and a memory plan for the hyperparameter gradient, the gradient loop either allocates temporary matrices (breaking the allocation policy) or repeats a linear solve per parameter (O(p n³)).

Negative marginal log likelihood (the quantity that is minimized):

```
L(θ) = ½ yᵀ K⁻¹ y + ½ log|K| + (n/2) log(2π)
∂L/∂θ_i = -½ αᵀ (∂K/∂θ_i) α + ½ Tr(K⁻¹ ∂K/∂θ_i)
        = -½ ⟨W, ∂K/∂θ_i⟩_F
where α = K⁻¹ y and W = ααᵀ - K⁻¹
```

`(n/2) log(2π)` does not depend on θ. In P2-6 an isolated add was about 650 ps, and the with/without difference on `mll_and_grad` was inside the noise of the baseline. The public NLML and the optimization `Objective` stay the same `L(θ)`. The API is not split.

Standard algorithm (Rasmussen & Williams / the GPy family):

1. Build `A = K + σn² I` into `k_matrix` (lower triangle only, `uplo=Lower` from §5.1)
2. In-place Cholesky. `k_matrix` becomes L
3. `log|K| = 2 Σ log(L_ii)` from the diagonal of L
4. Solve `L Lᵀ α = y` by forward and back substitution (O(n²))
5. Compute `K⁻¹` from `L` (triangular solves of `L Lᵀ X = I`, one O(n³))
6. `W[i,j] ← α[i] α[j] - K⁻¹[i,j]` (symmetric, so lower triangle only)
7. For each θ_i, evaluate `∂K/∂θ_i` into `exp_buf` and accumulate `⟨W, ∂K/∂θ_i⟩_F` in O(n²). Kernel parameters use `KernelTerm::grad`. Noise uses `Likelihood::noise_grad_diag` (diagonal only)

Total cost is O(n³ + p n²). `K⁻¹` is not rebuilt per parameter.

Analytic NLML Hessian (P2B-17 / [#109](https://github.com/YUKIKEDA/gprx/issues/109)):

```
H_ij = -½ ⟨W, ∂²K/∂θ_i∂θ_j⟩ - ½ Tr(K⁻¹ K_i K⁻¹ K_j) + αᵀ K_i K⁻¹ K_j α
```

`KernelTerm::hess` / `hess_points` write `∂²K` for one pair `(i, j)`. Custom, Sum, and Product are analytic. `FittedGpr::hessian_into` is the public entry, and `GprObjective` forwards to `TwiceDifferentiable`. `Q_j` (one `n×n`) and four length-`n` vectors live in `WorkspaceCore::hessian`: empty until the first Hessian, reused after it, so a Hessian after the first allocates nothing (R4-5c / [#272](https://github.com/YUKIKEDA/gprx/issues/272)). `CholeskyBuffer::Reuse` Chols again after ⟨W, K_ij⟩ and solves the first-order term `Q_i = K⁻¹ K_i`.

`value_and_gradient_into` runs this once and shares L, α, and `exp_buf` between the likelihood and the gradient. The default two-step `value` then `gradient_into` does not share them.

The default `CholeskyBuffer` is `Retain`. `K⁻¹` → `W` is written into a dedicated `w_matrix`, and `L` stays in `k_matrix`. Speed does not change. The public memory pole (`with_prefer_memory`) selects `CholeskyBuffer::Reuse`. `with_cholesky_buffer` sets it alone. `Reuse` solves `K⁻¹` in `exp_buf` and writes `W` into the Cholesky region. It does not restore `L` in the middle of the optimization loop. It Chols again at the end of `fit` and at the end of a standalone `value_and_gradient_into`. persist does not write this policy. `load` is `Retain`.

### 6.3 Exact GPR (`Gpr` / `FittedGpr`)

The public surface splits the trainer from the fitted model (P2-8).

`Gpr<O = Lbfgs, P = DoublePrecision>` holds a `KernelSpec`, a `GaussianLikelihood`, transforms, an optimizer `O`, and three runtime policies (R4-1 / [#239](https://github.com/YUKIKEDA/gprx/issues/239)): `DistanceCachePolicy { Cached, Uncached }`, `CholeskyBuffer { Retain, Reuse }`, and `KernelExp { Accurate, FastApprox }`. They are plain enums. No combination is illegal, so none is a type parameter. The `Gpr::new` default is the speed pole (`Cached` + `Retain`) with `Accurate`. The public switch is `with_prefer_memory` / `with_prefer_speed`. The memory pole is `Uncached` + `Reuse`. `with_distance_cache_policy` / `with_cholesky_buffer` / `with_math` set one policy each. A kernel that never reads pairwise distances (standalone Linear / Constant / White) allocates no distance cache whatever the policy says; there is no `from_points`. `FittedGpr` has no `with_prefer_*` (`into_trainer` → prefer → `refit`); it exposes the three policies through getters. The types stay at the crate root. `Gpr<O: Optimizer>::fit(self, …)` moves hyperparameters with `O` and, on success, returns `FittedGpr<O, P>`. Fixed hyperparameters are `Gpr<Fixed>::factor` (the old `FitOptions::FIXED`). There is no `optimize: bool`. On failure the consumed `Gpr<O, P>` is returned with the error. There is no `fitted: bool` and no `GprError::NotFitted`. An unfitted `transform` / `apply` cannot happen (`StandardizeTarget::fit(self)` returns `FittedStandardizeTarget`). On the public `FittedGpr`, `L` / `α` / `X` / compiled are not `Option` (P2B-5). A missing piece does not return `EmptyInput`.

`FittedGpr` holds what inference needs: `L`, `α`, training `X`, the kernel, the likelihood, and the transforms. `W`, `∂K`, and argmin state live only during `fit` and are not kept on the fitted value. Calling `predict` in the same process immediately after `fit` is treated as the minority path. The main path hands over a fitted model, so the inference object is `FittedGpr`.

`FittedGpr` and `OnlineGpr` share one crate-private `GprCore` (kernel spec, compiled kernel, likelihood, transforms, policies, training data, `α`, query buffers) and differ only in the factor (R4-2 / [#240](https://github.com/YUKIKEDA/gprx/issues/240)): `FittedGpr` holds the LLT buffers (`FitBuffers`, or a memory-mapped `L`), `OnlineGpr` holds the LDLT `LdltStore` and the `PointId` table. `StoredFactor { Llt, Ldlt }` is the factor view: `solve`, `L⁻¹` on columns, the per-pivot weight (`1` or `1/Dᵢ`), `log|A|`, and `diag(A⁻¹)`. Predict, covariance, sampling, LOO, NLML, and the predict `α` are written once on `GprCore` against that view. Every hyperparameter write (`set_params`, gradient, Hessian, `fit`, `refit`) runs on one borrowed `ExactFit` view (core + LLT buffers). `OnlineGpr` lends it temporary LLT buffers filled with `L √D` in O(n²) and writes the new factor back; the training data is not copied, and the only O(n³) work is the refactor the new `θ` needs.

The default `Gpr` is `Gpr<Lbfgs>`. `with_optimizer` replaces `O` (P2B-1). argmin `NonlinearCg` / `NelderMead` are P2B-2. argmin `Newton` is P2B-17. Leaf rebuilds follow §5.4 (P2B-18). There is no `with_recompute_strategy`. `Gpr<Fixed>::factor` only factors. The default `FittedGpr::predict` is a diagonal variance. Covariance between queries is a separate P2B-6 path (not a flag on diagonal `predict`). `loo_predict` returns per-training-point LOO from `L` and `α` as in GPML 5.4.2. Refactoring the same data at new hyperparameters is `FittedGpr::refit` (the fitted value keeps its `O`). `with_optimizer` / `factor` / `into_trainer` / `refit` keep the policies.

```rust
struct Gpr<O = Lbfgs, P = DoublePrecision> {
    kernel: KernelSpec,
    likelihood: GaussianLikelihood,
    x_transform: Box<dyn InputTransform>,
    y_transform: Box<dyn TargetTransform>,
    optimizer: O,
    policies: Policies,
    _precision: PhantomData<P>,
}

struct Policies {
    distance_cache: DistanceCachePolicy, // Cached (default) | Uncached
    cholesky_buffer: CholeskyBuffer,     // Retain (default) | Reuse
    math: KernelExp,                     // Accurate (default) | FastApprox
    jitter: JitterPolicy,
}

struct Fixed;

struct Lbfgs {
    max_iterations: u64,   // default 100
    tolerance: f64,
    history_size: usize,   // default 10. L-BFGS only
    n_restarts: u32,       // default 0
}

struct Newton {
    max_iterations: u64,
    tolerance: f64,
    gamma: f64,            // default 1. Newton only
    n_restarts: u32,
}

struct NonlinearCg {
    max_iterations: u64,
    tolerance: f64,
    n_restarts: u32,
}

struct NelderMead {
    max_iterations: u64,
    tolerance: f64,
    n_restarts: u32,
}

enum DistanceCachePolicy { Cached, Uncached } // distance-mode paths only (P2B-11)
enum CholeskyBuffer { Retain, Reuse }
enum KernelExp { Accurate, FastApprox }

struct FittedGpr<O = Lbfgs, P = DoublePrecision> {
    kernel: KernelSpec,
    likelihood: GaussianLikelihood,
    x_transform: Box<dyn InputTransform>,
    y_transform: Box<dyn TargetTransform>,
    optimizer: O,
    policies: Policies,
    workspace: FitBuffers<P>, // L. dist / W per policy
    query: QueryWorkspace<DoublePrecision>, // for predict_into
    compiled: CompiledKernel,
    alpha: Vec<f64>,
    x: Mat<f64>,
    y: Vec<f64>,
    n: usize,
    d: usize,
}

/// Objective borrows `Gpr` as &mut only during fit, and forwards set_params → MLL/gradient.
/// The parameter source of truth is Gpr.kernel / Gpr.likelihood.
struct GprObjective<'a, O, P = DoublePrecision> {
    model: &'a mut FittedGpr<O, P>,
    incremental: bool, // §5.4
}
```

`x` is received as column-major `&[f64]` and packed into an `n×d` `Mat`.

| Method | Receiver | Allocation |
| --- | --- | --- |
| `FittedGpr::predict` | `&self` | The output `Prediction`, and a temporary query buffer when one is needed |
| `FittedGpr::predict_into` | `&mut self` | 0 after warmup. Reuses the capacity of `mean` / `variance` |

Preconditions:

- An unfitted `predict` cannot happen. An unfitted `transform` / `apply` cannot happen either
- `FittedGpr::refit` replaces L and `α` at the same `n` / `d`
- The query input dimension `d` is fixed. A mismatch is `DimensionMismatch`
- n=0 is `EmptyInput`. n below the kernel's minimum point count is `InsufficientData`
- NaN/Inf in the input is `NonFiniteInput`
- Cholesky failure is `Err((gpr, err))`. A half-built `FittedGpr` is not returned

The default distance-cache policy is `Cached`. At the start of `fit` the squared distances of the training points are filled once, and later hyperparameter iterations rewrite only the kernel. Isotropic is `n×n`. ARD stores raw `(Δx_d)²` as `n × (n·d)` (P2-7). `Uncached` does not put those tensors on the Workspace, and both isotropic and ARD compute distances from `X`. The public memory pole is `with_prefer_memory` (`Uncached` + `Reuse`). The speed pole stays the default (`with_prefer_speed`). The P2B-21 libgp RSS pass/fail is `Uncached` + `Retain` (`with_distance_cache_policy` alone). The cache is allocated only when the compiled kernel reads distances: `RBF + White` and `Constant * RBF` do; standalone Linear / Constant / White do not, and their policy is kept but unused. persist tags are `always` / `never` (a missing tag loads as `Cached`). `LoadedGpr` has one variant per precision and factor kind (8), because both are type parameters of the model. `predict` / `predict_with` (widened to `f64`), `n`, `d`, and `is_online` work on any variant without a `match`; a variant is matched only for the typed model (`predict_into`, `insert`, `refit`) (R4-6 / [#244](https://github.com/YUKIKEDA/gprx/issues/244)). `load` is `Retain`.

### 6.4 Leave-one-out (P1B-7)

Leave-one-out for Exact GPR is a closed form from the fitted `L` and `α` (Rasmussen & Williams, GPML §5.4.2). With `A = K + σn² I`, `Q = A⁻¹`, and `α = A⁻¹ y`:

```
μ_i = y_i - α_i / Q_ii
σ_i² = 1 / Q_ii
```

This is the observation `p(y_i | X, y_{-i}, θ)`. The LOO variance of the latent `f_i` is `max(0, 1/Q_ii - σn²)`. `Q_ii` is the column norm of `L⁻¹` from the lower-triangular `L` (`A⁻¹ = L^{-T} L^{-1}`). The cost is the same order as Cholesky, O(n³), and the extra memory is one temporary `n×n`. At the Phase 1b sizes n=16 / 36 that is not a problem.

`FittedGpr::loo_predict` returns a `Prediction` of the same length as the training points. The default is `VarianceKind::Observation`. Mean and variance are mapped back to the original scale by `TargetTransform`, as in `predict`. A White leaf is not used. Noise is `GaussianLikelihood` only.

sklearn has no LOO API. `just gen-goldens` applies the same GPML formula to `L_` / `alpha_` after fit and writes JSON. The Rust side factors at the `θ` sklearn chose with `FitOptions::FIXED` (optimizer differences are not mixed into LOO).

## 7. Workspace and memory

### 7.1 Separate buffers

The buffer count is small and fixed, so they are separate fields. Storage and Refine of the precision policy are explicit.

```rust
struct WorkspaceCore<P: PrecisionPolicy> {
    k_matrix: Mat<P::Storage>,       // K; L after Cholesky. W during a Reuse gradient
    exp_buf: Mat<P::Storage>,        // kernel evaluation, ∂K/∂θ. Reuse n-RHS lives here
    kernel_scratch: Mat<P::Storage>, // product / custom ∂K/∂θ. Empty for isotropic RBF
    thread_scratch: Vec<Mat<P::Storage>>, // split ahead of time, one per Rayon thread
    rhs: Mat<P::Storage>,            // n×1, training Cholesky right-hand side y → α
    faer_scratch: MemBuffer,         // faer's own scratch, used as-is
    nested: Vec<Mat<P::Storage>>,    // one n×n per nesting level of a sum / product in another. Empty otherwise
    hessian: HessianScratch<P::Storage>, // Q_j and four n-vectors. Empty until the first Hessian (§6.2)
}

struct FitBuffers<P: PrecisionPolicy> {
    core: WorkspaceCore<P>,
    dist: Option<DistCache<P::Storage>>, // Some for Cached on a distance kernel
    w_matrix: Option<Mat<P::Storage>>,   // Some for Retain. W = ααᵀ - K⁻¹ (§6.2)
}

struct DistCache<S> {
    dist_cache: Mat<S>,
    ard_sq_diff: Mat<S>,             // 0×0 when isotropic
}

// Held by FittedGpr. predict_into warmup sizes it to (n, m, d)
struct QueryWorkspace<P: PrecisionPolicy> {
    query_xs: Vec<f64>,              // transformed query (column-major)
    query_x: Mat<P::Storage>,        // m×d
    query_k_star: Mat<P::Storage>,   // n×m
    query_scratch: Mat<P::Storage>,
    query_nested: Vec<Mat<P::Storage>>, // nested sum / product levels of the n×m block
    query_dist: Mat<P::Storage>,
    query_kss: Vec<f64>,
}
```

A sum or product whose term is itself a multi-term sum or product needs one more output-shaped buffer per nesting level (`CompiledKernel::nested_depth`). The crate-internal fit / predict entry points take those levels from `nested` / `query_nested`, which grow on the first call and are reused after (R4-5c / [#272](https://github.com/YUKIKEDA/gprx/issues/272)). The public `CompiledKernel::apply` / `grad` / `hess` family keeps its signature and builds the levels for that one call. The diagonal folds (`fill_diag`, `fill_diag_points`, and their gradients and Hessians) combine terms in fixed-size stack blocks of rows and allocate nothing. The sparse models (`FittedSgpr`, `OnlineSgpr`, `FittedSvgp`) keep their kernel scratch (output-shaped scratch, nested levels, train–query distances) in a crate-private `SparseScratch` between `&mut self` calls (`set_params`, gradient, Hessian, online updates); `&self` calls (`predict`, NLML) build it once per call. They still return new matrices for their factors and results, so `tests/alloc.rs` records their counts as a ratchet rather than zero (R5-1d / [#246](https://github.com/YUKIKEDA/gprx/issues/246)); a sparse `predict_into` is R5-5 ([#283](https://github.com/YUKIKEDA/gprx/issues/283)).

Fit buffers have a known size at the start of `fit`, so they are allocated once with `reserve_exact` (or built once with `Mat::zeros`) and overwritten on later iterations. Query buffers live on `FittedGpr`'s `QueryWorkspace`. The first `predict_into` sizes them to `(n, m, d)`, and the same query length reuses them. `predict(&self)` may allocate the output `Vec` every call. faer's `PodStack` / `MemStack` is the scratch manager. A hand-rolled scratch arena is not used.

Allocating inside a Rayon parallel closure is forbidden. `thread_scratch` is split ahead of time and **detached from `Workspace` immediately before entering the parallel region**. Do not pass `&mut self` (`Objective` / `Gpr`) into a Rayon closure.

```rust
// Before the parallel region:
let scratches = &mut self.workspace.thread_scratch[..];
// Hand them to workers with par_chunks_mut / zip.
// Bind k_matrix and the rest with as_mut locally, then parallelize.
```

### 7.2 Memory layout

faer `Mat` is column-major. Check `MatRef` / `MatMut` strides of the pinned version at implementation time.

- Distance and kernel matrices are scanned column-major, and symmetry is used to **compute only the lower triangle** (`uplo=Lower` on `KernelTerm::apply`)
- Input `X(n×d)` is stored as one point = one column = contiguous memory (column-major `d×n`)

### 7.3 Lifecycle during an iteration

```
fit() starts → n, d known → each Mat<T> allocated once → distance cache (once)
  → optimization loop:
       build A into the lower triangle of k_matrix
       in-place Cholesky (the same region becomes L)
       α, log|K|
       Retain: K⁻¹ → W in w_matrix. Reuse: K⁻¹ in exp_buf, W into k_matrix
       write ∂K/∂θ into exp_buf in turn and accumulate ⟨W, dK⟩
fit() ends → FittedGpr keeps L, α, X (Reuse Chols again here). W / ∂K / L-BFGS may be dropped
  → predict(&self): allocate the output
  → predict_into(&mut self): overwrite query_*, reuse Prediction capacity
```

A batch-fit Workspace has fixed n. Capacity growth for online learning belongs to `LdltStore` (§11), and its memory policy is separate from the batch Workspace. `Workspace`, `QueryWorkspace`, `LdltStore`, and faer types are crate-private.

## 8. Parallelism, SIMD, and the math backend

- The inner kernel loop is vectorized with `wide::f64x4` (P2-5 / P2-7). The targets are column-major, unit-row-stride isotropic RBF `apply` / `grad` / `apply_cross`, ARD RBF `apply` / `grad`, and the row loops of squared distance and `(Δx_d)²`. A view whose stride is not 1 falls back to scalar. `std::simd` is not used until it is stable. The inner loops of Matérn / Periodic / RQ are not vectorized yet.
- Distance-matrix and kernel-matrix construction is block-parallel with Rayon
- faer is itself Rayon-parallel, so do not nest a second pool. Share one `rayon::ThreadPool`. faer's thread count is `min(pool, n/64, n·k/16384, k/12)` ([ADR 0001](adr/0001-faer-parallel-degree.md)). `k` is the number of right-hand-side columns. Kernel fill uses the whole pool

**`MathBackend` starts from a minimal API, and the default is an accurate implementation, not an approximation.** An approximation error in the kernel matrix reaches positive-definiteness, Cholesky stability, the likelihood, the gradient, and the prediction.

```rust
trait MathBackend<T: Scalar>: Send + Sync {
    fn exp_inplace(&self, buf: &mut [T]); // exp only at first. erf is added when a kernel that needs it (a probit likelihood, and similar) actually appears
}
enum MathMode { Accurate, FastApprox }
```

The default is `Accurate` (`f64::exp` / `f32::exp` / `wide::exp`). `FastApprox` replaces `exp` in kernel evaluation, including during `fit` (P5-4 / [#42](https://github.com/YUKIKEDA/gprx/issues/42)). `exp(θ)` that maps a lengthscale back, and the `KernelTerm` formulas, stay on the accurate `exp`. On every model (`Gpr` / `FittedGpr` / `OnlineGpr`, `Sgpr` / `FittedSgpr` / `OnlineSgpr`, `Svgp` / `FittedSvgp`) the mode is the runtime enum `KernelExp`, set by `with_math(KernelExp::FastApprox)`; each kernel call dispatches once to the sealed `KernelMath` marker (R4-1 / [#239](https://github.com/YUKIKEDA/gprx/issues/239), R5-1a / [#246](https://github.com/YUKIKEDA/gprx/issues/246)). The `MathMode` enum above is not shipped. `FittedGpr` and `OnlineGpr` save records the mode. A file with no field loads as `Accurate`.

## 9. Optimizer

**Allocation-free, and wrapped in `Result`.**

```rust
trait Objective<T: Scalar> {
    fn num_params(&self) -> usize;
    fn value(&mut self, params: &[T]) -> Result<T, GprError>;
    /// Write the gradient into out. Err(GprError::UnsupportedKernelOperation) when gradients are not supported.
    fn gradient_into(&mut self, params: &[T], out: &mut [T]) -> Result<(), GprError>;
    /// Implement this so the inner work (Cholesky, W, exp_buf) is actually shared.
    fn value_and_gradient_into(&mut self, params: &[T], out: &mut [T]) -> Result<T, GprError> {
        let v = self.value(params)?;
        self.gradient_into(params, out)?;
        Ok(v)
    }
}
trait Optimizer<T: Scalar> {
    fn minimize(&self, objective: &mut dyn Objective<T>, init: &[T]) -> Result<OptResult<T>, GprError>;
    fn requires_gradient(&self) -> bool;
}
```

`init` is a slice (it does not consume the caller's `Vec`). `Gpr`'s `GprObjective` overrides `value_and_gradient_into` and shares L, α, W, and `exp_buf` by the §6.2 steps. `GprObjective` implements `TwiceDifferentiable`, and `hessian_into` forwards to `FittedGpr`. The public surface is `Gpr<O: Optimizer>`. The default is `Lbfgs`. Other argmin solvers and a user implementation replace the same type parameter through `with_optimizer`. The public `Newton` is argmin's `Newton` (`H⁻¹` is a private faer type. The logit matches L-BFGS and `H_z` is the analytic chain. Knobs are the shared three plus `with_gamma`). The homemade example is `FastSimulatedAnnealing` (Cauchy / Metropolis. P2B-15 / [#106](https://github.com/YUKIKEDA/gprx/issues/106)). It does not use the logit. `minimize` walks the log-`θ` it receives. Do not put `FitOptions::solver` next to a custom optimizer and ignore one of them (`.cursor/rules/types.mdc`). gprx does not implement its own quasi-Newton. Objective capability is `Objective` (value) ⊂ `Differentiable` ⊂ `TwiceDifferentiable`. There is no runtime NotImplemented. A partial update is `IncrementalObjective::value_with_changes(params, indices: &[usize])`. `GprObjective` implements it (P2B-18 / [#110](https://github.com/YUKIKEDA/gprx/issues/110)); an optimizer opts into leaf rebuilds with `Optimizer::USES_CHANGE_INDICES` (§5.4). The default of `Objective::value_at_changes` is `value`. One FSA coordinate step calls it. There is no `ChangeSet` struct.

## 10. `GprError`

Cover the failures that are specific to numerical work.

```rust
#[derive(Debug, thiserror::Error)]
pub enum GprError {
    #[error("input dimension mismatch: X.ncols()={x_dim}, expected={expected_dim}")]
    DimensionMismatch { x_dim: usize, expected_dim: usize },
    #[error("not enough data points: n={n}, at least {min} required")]
    InsufficientData { n: usize, min: usize },
    #[error("input is empty")]
    EmptyInput,
    #[error("input contains a non-finite value (NaN/Inf)")]
    NonFiniteInput,
    #[error("a kernel evaluation contains a non-finite value")]
    NonFiniteKernelValue,
    #[error("Cholesky failed (stage={stage:?}, size={matrix_size}, jitter={jitter} already applied)")]
    CholeskyFailed { jitter: f64, matrix_size: usize, stage: CholeskyStage },
    #[error("matrix is not positive semidefinite")]
    NonPositiveDefiniteMatrix,
    #[error("this kernel term does not implement the Sparse GPR coordinate derivative (grad_wrt_coord_dim)")]
    CoordGradientUnsupported,
    #[error("optimization did not converge (after {iterations} iterations)")]
    OptimizationNotConverged { iterations: usize },
    #[error("invalid hyperparameter: {reason}")]
    InvalidHyperparameter { reason: String },
    #[error("shape mismatch: {reason}")]
    ShapeMismatch { reason: String },
    #[error("length mismatch: {reason}")]
    LengthMismatch { reason: String },
    #[error("index out of range: {reason}")]
    IndexOutOfRange { reason: String },
    #[error("invalid configuration: {reason}")]
    InvalidConfig { reason: String },
    #[error("size overflows usize")]
    SizeOverflow,
    #[error("invalid observation-noise variance: {reason}")]
    InvalidNoiseVariance { reason: String },
    #[error("unsupported kernel operation: {reason}")]
    UnsupportedKernelOperation { reason: String },
    #[error("Workspace capacity is too small")]
    WorkspaceTooSmall,
    #[error("the PointId does not exist")]
    InvalidPointId,
}

#[derive(Debug)]
pub enum CholeskyStage { Fit, Predict, OnlineInsert, OnlineDelete }
```

`InvalidHyperparameter` is only for a hyperparameter value outside its domain. Matrix shape, slice length, and index errors are `ShapeMismatch`, `LengthMismatch`, and `IndexOutOfRange`. Optimizer, jitter-policy, and transform settings are `InvalidConfig`. A size product that overflows `usize` is `SizeOverflow`, not `EmptyInput`.

**Error versus panic**: failures caused by user input (`DimensionMismatch` and similar) and by the model or the data (`CholeskyFailed` and similar) return `Result` and stay recoverable. `CoordGradientUnsupported` is not an internal panic, so it returns this error instead of `unimplemented!()`. There is no `NotFitted` variant. An unfitted call cannot be formed.

## 11. Online learning (adding and removing points)

GPR cost grows as O(n³) with n, so adding and removing points one at a time is in scope. Beside the batch-fit Workspace (fixed n), there is a crate-private `LdltStore` and a public `OnlineGpr`. `FittedGpr::into_online(self)` converts. `insert` exists only on `OnlineGpr`.

### Cost

| Operation | Full refit | Incremental |
| --- | --- | --- |
| Add one point | O(n³) | O(n²) |
| Delete one point | O(n³) | O(n²) |

### Following the faer API

As in §3, **LLT has no insert/delete API**. The online path is:

1. **Append**: a hand-rolled bordered update. O(n²)
2. **Delete at any index**: `LdltStore` holds an **LDLT factor** and uses `ldlt::update::delete_rows_and_cols_clobber`. On handwritten SPD matrices of size `2×2` / `5×5`, the reconstructed `A = L D Lᵀ` after delete matches a full LDLT (P3-1 / [#30](https://github.com/YUKIKEDA/gprx/issues/30)). There is no Givens downdate

Batch fit stays LLT. `FittedGpr::into_online` converts LLT→LDLT in O(n²):

- `D[j] = L_llt[j,j]²`
- `L_ldlt[:, j] = L_llt[:, j] / L_llt[j, j]` (the diagonal is 1)

### Why the incremental append works

For a lower-triangular LLT, the matrix with a new point is `K_new = [[K, k], [kᵀ, k_new]]`. Against the existing factor `L`, set `L_new = [[L, 0], [vᵀ, d]]`. Then `L_new L_newᵀ = K_new` requires:

- `L v = k` (forward substitution. `v = L⁻¹ k`)
- `d = √(k_new - vᵀ v)`

The online path is LDLT, so the matching bordered update is:

When `A = L D Lᵀ`, `A_new = [[L, 0], [vᵀ, 1]] [[D, 0], [0, δ]] [[Lᵀ, v], [0, 1]]`

- `L D v = k`, that is `L w = k` and then `v = D⁻¹ w`
- `δ = k_new - vᵀ D v`

At implementation time, check agreement with a full factorization on a small matrix (for example 2×2 and 5×5) (§12).

### Capacity

Batch fit and online differ (fixed n versus n that grows and shrinks). On growth, **reallocate and copy `LD`, `y`, `α`, and `v_buf` by the same steps**. faer scratch for delete grows to the same capacity. predict, NLML, and insert do not read the Gram `K` or the distance cache, so `LdltStore` does not hold them.

Crate-private. `from_active(n)` sets `n_active = n_capacity = n`. Training `X` is held by `OnlineGpr` and is not on this struct. Before an append insert, `OnlineGpr` calls `ensure_capacity`. There is no growth-factor field.

```rust
struct LdltStore {
    ld_factor: Mat<f64>,    // LDLT factor (diagonal = D, strict lower triangle = L)
    alpha: Col<f64>,
    y: Col<f64>,
    v_buf: Col<f64>,        // forward-substitution scratch for predictive variance (O(n²) per test point)
    delete_scratch: MemBuffer, // faer delete_rows_and_cols. Grown with capacity
    n_active: usize,
    n_capacity: usize,
}
```

**Growth** (`ensure_capacity(needed)`, when `n_capacity < needed`):

1. `new_cap = max(needed, max(n_capacity, 1) * 2)`
2. Reallocate `ld_factor`, `alpha`, `y`, and `v_buf` at `new_cap`. Grow the delete scratch for `new_cap` too
3. Copy the existing `n_active × n_active` lower triangle and the length-`n_active` vectors
4. `PointRegistry` indices stay below `n_active`, so they do not need to be rewritten
5. Run the insert after growth. Do not reallocate in the middle of the update

**Variance at predict**: the predictive mean is O(n), but the predictive variance `σ*² = k(x*,x*) - vᵀ D v` (LDLT, the rewritten `L v = k*`) is O(n²) per test point. `v_buf` is allocated up front.

### Steps and the invariant

**Append**: (1) if capacity is short, `LdltStore::ensure_capacity` (factor 2). Training `X` / `y` on `OnlineGpr` grow by the same factor. Query buffers use `ensure_at_least`. (2) Distances from the new point to the existing n points (O(n). One column is sequential, and `k` is written directly into `v_buf`). (3) Add only the kernel diagonal `k_new` (insert does not write a new row or column of `K`. predict and NLML read only LD). (4) Bordered LDLT update (O(n²). The triangular solve reuses `v_buf`). (5) `α` is not solved on insert (libgp `alpha_needs_update`); insert only marks it stale in O(1). The first read resolves the LDLT again: a `&mut self` read (`predict_into`, a hyperparameter write) stores `α` on the model, and a `&self` read (`predict`, covariance, sample, LOO, NLML, `alpha()`, `save_with_factor`) fills a `OnceLock` cache that the next insert / delete clears. A failed solve (a `MixedPrecision` `f64` fallback that does not factor) is that read's `Err`, so `OnlineGpr::alpha()` returns `Result` (R4-2b / [#265](https://github.com/YUKIKEDA/gprx/issues/265)). (6) `PointRegistry` issues a new `PointId`.

**Delete**: (1) update LD with `ldlt::update::delete_rows_and_cols_clobber` (O(n²). Scratch lives on `LdltStore` and is reused). (2) Remove the matching entries from `y` and `X` on `OnlineGpr` and pack the later rows/columns (O(n)). (3) Shift `PointRegistry` indices in the same order. (4) `α` is not solved on delete either; the stale mark and the first-read solve are the same as append step (5). `n_capacity` stays. The last point is not deleted (`InsufficientData`, `min = 2`). An unknown or already-deleted `PointId` is `InvalidPointId`.

**Invariant**: when a delete shifts internal indices, `LD` / `y` / `alpha` on the workspace, `X` on `OnlineGpr`, and `PointRegistry` **must stay in the same order**. One of them drifting produces the wrong solution. Tests (§12) check this invariant explicitly.

```rust
struct PointRegistry {
    id_to_index: HashMap<PointId, usize>,
    index_to_id: Vec<PointId>,
    next_id: u64,
}
```

### API

**Separate insert/delete from reoptimizing hyperparameters.**

An unfitted `Gpr` does not gain points. A batch `FittedGpr` has no `insert`.

```rust
impl FittedGpr<O, P> {
    fn into_online(self) -> Result<OnlineGpr<O, P>, GprError>;
}

impl OnlineGpr<O, P> {
    fn insert(&mut self, x_new: &[f64], y_new: f64) -> Result<PointId, GprError>;
    fn delete(&mut self, id: PointId) -> Result<(), GprError>;
    fn point_ids(&self) -> &[PointId];
}
```

`insert` / `delete` update LD and alpha at the current kernel and hyperparameters. Hyperparameters are reoptimized only when `OnlineGpr::refit` / `set_params` is called explicitly. Those run the batch fit code on temporary LLT buffers (§6.3) and keep `PointId` values and workspace capacity. `into_online` assigns `0 .. n-1` to the existing `n` points. Later `insert` ids increase and are not reused. `PointId` has no public constructor. `PointRegistry` is crate-private and owned by `OnlineGpr`. persist keeps `FORMAT_VERSION` 1 and requires `factor_kind` (`llt` / `ldlt`). `load` of `llt` is `FittedGpr`. `ldlt` is `OnlineGpr`, and `point_ids` plus `next_point_id` are also required. Sparse online is `OnlineSgpr` (§6).

## 12. Test plan

Put correctness tests into each phase before speed.

1. **Kernel mathematics**: known values for RBF/Matern/Periodic, symmetry, the diagonal, numerical derivatives against analytic gradients, and agreement of `uplo=Lower` with `Full`
2. **Cholesky**: reconstruction error of `K=LLᵀ`, with and without jitter, and behavior on ill-conditioned or duplicate data
3. **MLL and gradient** (separate from inference tests):
   - Analytic MLL on a known small problem
   - Numerical derivatives of the MLL against the analytic gradient
   - Gradient of each kernel parameter
   - Gradient of the noise parameter (`log_noise_variance`). `∂K/∂θ = σn² I`
   - Gradient stability on an ill-conditioned matrix
4. **Online updates**: one-point add/delete matches a full refit, delete at an arbitrary index, repeated add and delete, and agreement of PointId with the internal index (the §11 invariant)
5. **Online property tests**: at every stage of a random insert/delete sequence, incremental == `Gpr<Fixed>::factor` (mean, variance, LML, alpha). Delete order is randomized with `SmallRng`
5b. **External check of online insert** (P3-6): libgp `add_pattern` at the same θ, against predict (mean and observation variance) and NLML, relative `1e-8`. There is no external delete API. `cargo test` reads committed JSON (it does not call C++)
5c. **External check of Sparse** (P4-11): `Sgpr<Fixed>::factor` against collapsed GPyTorch SGPR, and `Svgp<Fixed>::factor` (prior `q`) against whitened SVGP, at the same initial θ, relative `1e-8` (mean, Observation, Latent, NLML / ELBO). `cargo test` reads committed JSON (it does not call Python). Batch wall time and RSS are P4-12 (`just perf-sparse`. GPyTorch / GPy. Manual, not CI)
5d. **External check of Sparse online** (P4-13): `OnlineSgpr` `insert` / `delete` / `insert_inducing` / `delete_inducing` against a collapsed GPyTorch SGPR (full reassemble) at each stage, same initial θ, relative `1e-8` (mean, Observation, Latent, NLML). `cargo test` reads committed JSON (it does not call Python)
5e. **Sparse online wall time** (P4-14): time `OnlineSgpr` `insert` / `delete` / `insert_inducing` / `delete_inducing` against a full reassemble of our `Sgpr<Fixed>::factor` and a GPyTorch Titsias assemble (no queries), at the same initial θ. Correctness is P4-13. The prefix is outside the timer. 32 moves are one wall-clock sample (1 discarded + the median. The count follows the P2B-16 tier rule). `cargo test` does not run this. `just perf-sparse-online` (manual, not CI)
6. **Precision**: compare f32/f64/mixed, ill-conditioned matrices, and the f64 fallback when refinement does not converge
7. **Inference**: compare with a known small GPR (mean, latent variance, observation variance, log marginal likelihood, gradient). sklearn JSON is the second numerical check, not the public-API contract. The algorithm source of truth is GPML / Rasmussen
8. **Preprocessing**: `predict` after `StandardizeTarget` matches the unstandardized model on the original scale (the closed form of the affine map)
9. **Inference after optimization** (P1B-6): 1-D Forrester and the 2-D weighted sphere (ARD), sklearn L-BFGS against `Gpr::fit`, at a loose tolerance. Separate from the fixed-hyperparameter JSON (1e-8). `cargo test` does not call Python
10. **Leave-one-out** (P1B-7): the GPML analytic formula at n=2, a real leave-one-out `fit`+`predict` at n=3, and the LOO fields of the P1B-6 JSON at sklearn's `θ`. `cargo test` does not call Python

## 13. Implementation roadmap

Order and status are [roadmap.md](roadmap.md). Acceptance text stays on each Issue.

## 14. Open items

1. **Checking mixed-precision refinement parameters**: the §4.2 defaults have a theoretical basis, and they have not been checked on a real workload. That includes the accuracy gap between `PromoteStorage` and `ReevaluateKernel`, and `log|K|` plus the trace term when MixedPrecision is used during fit
2. **The `DistanceCachePolicy::Auto` threshold**: it needs a measurement that accounts for kernel kind, SIMD efficiency, and memory bandwidth (P5-5. Acceptance is set after Grill)

## 15. Benchmark strategy

"Fastest" and "fewest allocations" do not become a picture if measurement starts suddenly in Phase 2. **After correctness, measure the same path while building it.** Phase 2 is the optimization phase, not the moment measurement starts. Day-to-day rules are `.cursor/rules/bench.mdc`.

### 15.1 Two tracks

| Track | Tool | When | What is read |
|---|---|---|---|
| Time | criterion, `benches/exact.rs` | `just bench` (local). Not in default CI (noise) | Wall time. Groups are measured separately |
| Allocations | `tests/alloc.rs` | `just test` (required) | New allocations **after** Workspace setup. The cap is a ratchet (it may fall, and it does not rise without an Issue) |

Do not mix time and allocations into one number. Do not mix a full L-BFGS with "one MLL+gradient".

### 15.2 Fixed problems (the regression unit)

The input has to be the same every time, or a faster run cannot be told from a different dataset.

- `n = 256` is required from P1A-18. Add `512` / `1024` once they finish in seconds
- Isotropic: 1-D Forrester `f(x)=(6x-2)² sin(12x-4)`, `x ∈ [0, 1]`, RBF + `GaussianLikelihood` + `StandardizeTarget`. Initial hyperparameters `ℓ = 1`, `σn² = 0.1`
- ARD: 2-D weighted sphere `f=(x/0.25)²+(y/1)²`, a 16×16 grid on `[0, 1]²`. Initial `ℓ_d = 4` (`ℓ_d = 1` dies on the first line search)
- `y` is that function plus `N(0, 1)` (`SmallRng`. Forrester seed `0`, ARD sphere seed `9`. Seed `0` walks a ridge on Never). It is not an independent random series (L-BFGS eval counts move with the landscape)
- Some of the historical `phase-1a` / `phase-1b` log used `d = 8` and an independent random `y`. The Forrester remeasure of `phase-1b` is P2-9 (`.dev/bench-log.md` (local, not committed)). Do not mix those times with the d = 8 times
- Groups (only paths that exist. Do not write a group that is not there yet):
  1. `kernel_rbf` — lower-triangle build of K
  2. `cholesky_alpha` — LLT of `A` and `α`
  3. `mll_and_grad` — one §6.2 evaluation (from P1A-10)
  4. `predict_100` — 100 test points (from P1A-8)
  5. `fit_lbfgs` — the whole optimization loop (from 1b. Do not mix it with 1). Record the L-BFGS eval count next to the wall time. A difference at a different eval count is not a speed difference
  6. `mll_and_grad_ard` / `fit_lbfgs_ard` — ARD RBF on the weighted sphere (P2-7). Always versus Never. Do not compare with isotropic. `fit_lbfgs_ard` also records the eval count
  7. `kernel_exp` / `kernel_exp_ard` — `apply` and the θ `grad` after the distance is filled once (P5-4). `FastApprox` versus `Accurate`. Do not mix with `mll_and_grad`
  8. `online_insert` / `online_delete` — Phase 3

### 15.3 What is added when

| When | What |
|---|---|
| M0 | The crate only. Do not add an empty `benches/` |
| Right after P1A-7 (P1A-18) | criterion and `just bench`. `kernel_rbf` and `cholesky_alpha` |
| P1A-8 / P1A-10 | Add `predict_100` / `mll_and_grad` to the same file. P1A-19 adds the allocation ratchet |
| End of 1a | Take the named baseline `phase-1a` and record the machine and the numbers in `.dev/bench-log.md` (local, not committed) |
| End of 1b | Add `fit_lbfgs` and the baseline `phase-1b` |
| Phase 2 | **No new harness.** Read `phase-1b` and optimize in bottleneck order. P2-5: SIMD for isotropic RBF and distances. Judge it on `kernel_rbf` / `predict` / `FIXED`, not on the gradient term of `mll_and_grad` alone. The NLML constant is measured in P2-6, the difference is noise, and `L(θ)` stays one formula. The ARD distance cache is P2-7, Always versus Never on `mll_and_grad_ard` / `fit_lbfgs_ard`. The fill and RBF ARD are Rayon + SIMD |
| End of 2 (P2-9) | Take the named baseline `phase-2` and record the machine and the numbers in `.dev/bench-log.md` (local, not committed). Isotropic is compared with `phase-1b`. ARD is Always versus Never. `just test` and alloc 0 on the `FittedGpr` path |
| Phase 3+ | Add insert/delete and the rest on the same problem definition. The comparison baseline is `phase-2`. Sparse wall time and RSS are `just perf-sparse` (P4-12). Online time is `just perf-sparse-online` (P4-14). Do not add a Sparse group to criterion |

A PR that touches a hot path (`src/kernel/`, `workspace`, `exact`, `objective`, `online`) pastes criterion against the previous baseline in Verification. If the change cannot affect speed, say why.

### 15.4 Metrics

Target ratios are set after `phase-1b` exists. After Phase 2 the baseline is `phase-2`. Until then the gate is "not worse than before".

| Metric | Contents |
|---|---|
| One MLL+grad | By n and kernel. Separate from the optimization loop |
| Fit (L-BFGS) | Includes iterations. From 1b |
| Predict | By test-point count. Latent / observation |
| Peak memory | Including Workspace. Includes `w_matrix` |
| Allocations | Count after setup. Ratchet, and finally 0 on the hot path |
| Parallel | By thread count. Watch double parallelism with faer. Phase 2 |
| f32/f64 | Accuracy and speed. Phase 5 |
| Online insert/delete | One point versus a full refit. Phase 3 |

### 15.5 Not done

- Adding Rayon / SIMD / an approximate exp because it "should be faster", without a measurement
- Making criterion in CI a red/green gate (it flakes across machines)
- Requiring zero allocations in a test on day one of 1a (count first, then lower the cap in steps)

