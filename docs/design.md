English | [日本語](design.ja.md)

# gprx design

## 1. Purpose and scope

Build the most flexible and the fastest Gaussian process regression library in Rust. Flexible means user-defined kernels, preprocessing, exact and sparse inference, a swappable optimizer, and **adding and removing data points one at a time (online learning)**. Fast means minimizing allocations, using SIMD and multiple threads, and trading compute against memory by switching precision.

This document states the current design only. Why a decision was made is in [adr/](adr/); order and status are in [roadmap.md](roadmap.md).

## 2. Architecture

```
input X, y
  → UnfittedTransform / Transform (preprocess X: MinMax, Standardize, per column, pipeline)
  → UnfittedTarget / TargetTransform (standardize y, and invert mean/variance at predict)
  → GaussianLikelihood (observation noise σn², a model parameter of its own)
  → KernelSpec → CompiledKernel<T> (flattened tree, static dispatch over built-in leaves)
  → Gpr<O, P> (trainer: kernel, likelihood, transforms, optimizer, policies)
       → GprObjective (NLML and its derivatives; only during fit / refit)
       → O: Optimizer<GprObjective> (default `Lbfgs`; argmin solvers and a user `O` use the same slot)
       → fit(self) → FittedGpr | (Gpr, GprError);  Gpr<Fixed>::factor for fixed θ
  → FittedGpr (LLT, α, X. predict / predict_into / covariance / sample / loo / refit / save)
       → OnlineGpr: `FittedGpr::into_online(self)` converts LLT→LDLT. `insert` / `delete` exist only here
  → Sgpr / Svgp: sparse trainers with their own fitted types (FittedSgpr, OnlineSgpr, FittedSvgp) (§6.1)
  → persist: one directory (`config.json` + `model.safetensors`) per model (§6.3, §11)
```

Persist writes one directory. `format_version` is 1. Exact models require `factor_kind` (`llt` loads as `FittedGpr`, `ldlt` as `OnlineGpr`), and a stored factor is memory-mapped. Sparse models add a `model` key (`sgpr` / `online_sgpr` / `svgp`; an Exact file has none) and load through `LoadedSgpr` / `LoadedSvgp`. Their tensors are the original `X` / `y` / `Z`, the transformed `Z`, and SVGP's `q(u)`; the factors are rebuilt, so a loaded model predicts the same values to the bit. `config.json` floats round-trip exactly (serde_json `float_roundtrip`). Retraining a loaded model is `with_optimizer` → `refit`.

Principles:

- **Identifiers name gprx / GPR concepts** (kernel, likelihood, θ, factorization, an interval on a positive parameter, …). Do not name another product, a test harness, or an unrelated domain
- **Static dispatch by default. `dyn` only at the extension point (a user-defined kernel)**
- **The hot path inside gprx does not allocate** (the phrase "zero allocations during fit" cannot be enforced inside a user kernel, so the rule is this one)
- **Precision is a compile-time generic**
- **Numerical stabilization (jitter) is separate from the model parameter (observation noise)**

## 3. Linear algebra: faer

Depend on **faer 0.24.x**. Stride and view constraints of `Mat<T>` follow the pinned API. This document does not freeze a layout.

Pure Rust, at or above OpenBLAS / LAPACK / Eigen, with Rayon parallelism in the same class as OpenMP / TBB.

- `Mat<T>` is column-major. **Kernel SIMD that assumes a contiguous stride is written only after checking the real `MatRef` / `MatMut` stride**
- Batch-fit Cholesky is `llt::factor::cholesky_in_place` (lower-triangular LLT, in place)
- Jitter is gprx's retry loop around that factor (`JitterPolicy`, §4.0); each attempt passes its `j` to faer's `LltRegularization`. **It is numerical stabilization only, and it is not the GPR observation noise (a model parameter)**
- `Mat` supports capacity-based reallocation (used by online learning in §11)
- **Cholesky update API of faer 0.24**:
  - `llt::update` has only `rank_r_update_clobber`. **LLT has no high-level row/column insert/delete**
  - `ldlt::update::delete_rows_and_cols_clobber(LD, indices: &mut [usize], ...)` exists and deletes several rows at arbitrary indices
  - `ldlt::update::insert_rows_and_cols_clobber` is not public (only `insert_rows_and_cols_clobber_scratch`. The body is private)
  - Online learning follows §11 (append is hand-rolled, delete uses the LDLT API)
- `llt::update::rank_r_update_clobber` / `ldlt::update::rank_r_update_clobber` are not used. The sparse online updates hand-roll their rank-1 cholupdates (ADR 0004 / 0005), and a hyperparameter change is always a full refactor (§5.4)

## 4. Precision, and separating noise from jitter

### 4.0 Observation noise is not jitter

Separate "observation noise σn²" (a GPR model parameter, optimized) from "jitter" (a numerical offset that keeps Cholesky positive definite).

```rust
/// Observation noise σn², stored as θ = log(σn²) on an open interval of σn².
/// σn² = exp(θ), so ∂K/∂θ = σn² I (not the 2σn I of a σn parameterization).
pub struct GaussianLikelihood {
    noise_variance: BoundedParam,
}

impl GaussianLikelihood {
    pub fn new(noise_variance: f64) -> Result<Self, GprError>;
    pub fn from_log_noise_variance(log_noise_variance: f64) -> Result<Self, GprError>;
    pub fn noise_variance(&self) -> f64;
    pub fn log_noise_variance(&self) -> f64;
    pub fn bounds(&self) -> Interval;
    pub fn with_bounds(self, interval: Interval) -> Result<Self, IntervalError>;
    pub fn num_params(&self) -> usize;                                   // always 1
    pub fn get_params(&self, out: &mut [f64]) -> Result<(), GprError>;   // θ
    pub fn set_params(&mut self, params: &[f64]) -> Result<(), GprError>;
    pub fn add_noise_diag(&self, k_diag: &mut [f64]);                    // K += σn² I
    pub fn noise_grad_diag(&self, dk_diag: &mut [f64], param_idx: usize) -> Result<(), GprError>;
}

/// Numerical Cholesky stabilizer. It never touches σn².
pub enum JitterPolicy {
    Fixed(FixedJitter),       // JitterPolicy::fixed(j): one retry with j ≥ 0
    Adaptive(AdaptiveJitter), // JitterPolicy::adaptive(initial, multiplier, max_retries, max_jitter)
}
```

The likelihood is a concrete type, not a trait: Gaussian noise is the only likelihood. `get_params` / `set_params` read and write `θ` on a length-1 slice. `noise_grad_diag` fills the diagonal with `σn²`. A model takes its jitter policy through `with_jitter_policy`; there is no separate stabilizer object.

`A = K + σn² I` is **the matrix of the linear system that is actually solved** (the GPR model).

**Where jitter applies**:

- The first factorization always tries `A` with no extra diagonal. `JitterPolicy` is used **only when that Cholesky fails**. `Fixed` retries once with its `j`. `Adaptive` retries with `initial`, then multiplies by `multiplier`, and stops at `max_retries` or when `j` would exceed `max_jitter`. A factor that succeeded is the factor of `A + j I`, and the solution in hand is `(A + j I)^{-1} y`. The model keeps that `j` (crate-private `factor_jitter`, also saved with a stored factor), and `CholeskyFailed::jitter` is the last `j` tried. The Exact default is `fixed(0.0)`: no retry.
- Do not iteratively refine that factor back onto the original `A`. The gap between the preconditioner `LLᵀ ≈ A + jI` and the target `A` grows, and the contraction `||I - (LLᵀ)^{-1} A||` can exceed 1 and diverge (§4.2).
- The sparse models factor `K_mm = k(Z, Z)`, which carries no observation noise. `Sgpr` / `Svgp` take their own `with_jitter_policy` for `K_mm`, and its default is `adaptive(1e-8, 10, 5, 1e-3)` rather than the Exact default: close inducing points leave `K_mm` singular in floating point. Fit, factor, `set_params`, predict, and the online updates all use that policy.

### 4.1 Precision: f32 / f64 / mixed

The goals are both "less memory" and "more speed". Use mixed-precision iterative refinement, and **split where it applies between fit and predict**.

```rust
pub trait PrecisionPolicy {
    type Storage: KernelScalar; // factor and kernel matrices
    type Refine: KernelScalar;  // predictive α and the returned Prediction<Refine>
}
pub struct DoublePrecision;                                     // Storage=f64, Refine=f64 (default)
pub struct SinglePrecision;                                     // Storage=f32, Refine=f32
pub struct MixedPrecision<R: ResidualFormula = PromoteStorage>; // Storage=f32, Refine=f64
pub struct PromoteStorage;   // residual from the stored f32 matrix
pub struct ReevaluateKernel; // residual from the kernel re-evaluated in f64
```

A model's precision is its type parameter `P` (`GpScalar` for Exact, `ModelPrecision` for the sparse models; both are implemented for the three types above), switched with `with_precision::<P2>()`. `ModelPrecision` carries the per-precision behavior the models share.

**Limit of the split**: `log|K| = 2Σlog(L_ii)` in the marginal log likelihood, and the trace term `Tr(K⁻¹∂K/∂θ)` in the gradient, are not made more accurate by refining `α = K⁻¹y` (they depend on the f32 diagonal of `L` itself). The default for fit (the hyperparameter loop), which includes those terms, is **`DoublePrecision`**. `MixedPrecision` mainly targets predict, where the hyperparameters are fixed and the linear solve for `α` is the whole job. Using `MixedPrecision` during fit assumes a separate accuracy check of `log|K|` and the trace term (§14).

Steps (predict, or a solve at a fixed kernel):

1. Factor `A = K + σn² I` in place with `cholesky_in_place::<f32>`, still in f32 (only jitter regularization inside)
2. `alpha_0 = solve(L, y)` with the f32 `L`
3. Compute the residual in f64. **Two ways to build the residual matrix `A_resid`**. Memory and accuracy trade off:
   - **`PromoteStorage` (default)**: promote the stored f32 `A` to f64 and set `r = y_f64 - A_f32→f64 @ alpha`. This refines the solution of "the linear system held in f32". It is not iterative refinement against the true f64 kernel matrix. It does not keep a separate f64 `A`, so it matches the memory goal.
   - **`ReevaluateKernel`**: reevaluate the kernel in f64 on every residual matvec. `A_f64` is not stored. Each iteration pays an O(n²) kernel evaluation, and the system is closer to the true f64 system.
   - Refinement runs on the factor that fit (or an online update) left behind: `α₀` is that factor's solve of `y`, and each correction solves through it. The system is `A + (σn² + j) I` with the jitter `j` the factor retry added (`0` without a retry); fit records `j`, online inserts add the same `j`, and a saved factor stores it. There is no second f32 factorization. A converged `PromoteStorage` α is checked once against the f64 system (kernel evaluated in column blocks, no `n×n` f64 matrix); when `κ(A) u_f32` is large and the check fails, α falls back to the f64 Cholesky solution of the same system. The f64 fallback retries with the model's `JitterPolicy` and reports the caller's stage.
   - Storing the whole f64 `A` contradicts the memory goal, so it is not used.
4. `delta = solve(L, r)` with the f32 `L`, then `alpha_1 = alpha_0 + delta`
5. Repeat a few times until convergence

The omitted precision is `DoublePrecision`. `SinglePrecision` runs the same steps in f32 and uses the factor as-is; it has no residual type parameter. `MixedPrecision<R>` factors in f32 and iteratively refines only the predictive α. MLL and the gradient during training use that precision's factor; iterative refinement does not run inside the training loop. The residual type parameter exists only on `MixedPrecision`, and its default is `PromoteStorage`: on Forrester `n=1024` the release median is 22.40 ms for `PromoteStorage` and 48.77 ms for `ReevaluateKernel`. There is no flag and no in-code alias. Every model (Exact, online, `Sgpr`, `Svgp`) and every optimizer takes all three precisions.

### 4.2 Convergence parameters for mixed-precision refinement

From classical iterative refinement (Higham), with factorization precision u_f (f32 ≈ 1.19×10⁻⁷) and refinement precision u_r (f64 ≈ 2.22×10⁻¹⁶), the rate depends on κ(A)·u_f. **The actual stopping test uses the measured residual, not the theoretical value.**

The parameters are fixed crate constants in `src/precision/refine.rs`, not a public configuration:

| Parameter | Value |
| --- | --- |
| Most corrections | 10 |
| Relative tolerance | `10 · dim · u_r` (`u_r = f64::EPSILON`), on the measured residual |
| Stagnation | two consecutive residual-norm ratios above `0.9` |
| Not converged | the `f64` solution of the same system (never an error) |

Stopping test: `||r_k||∞ / (||B||∞ ||w_k||∞ + ||b||∞) < 10 · dim · u_r`. One loop (`refine` over a `RefineSystem`) serves Exact `α`, Sgpr weights, and the Svgp triangular solve; each system supplies its residual, the solve through its stored factor, and its `f64` fallback. Refinement never returns a convergence error: the fallback is always the `f64` solve.

**Do not raise jitter when IR fails to converge.** Raising only the factorization jitter widens the gap between the preconditioner `LLᵀ` and the target `A`, and IR can diverge. A failed IR falls back to the `f64` solution. Adaptive jitter stays reserved for Cholesky failure, as in §4.0.

**Standing**: the theory holds, and checking the parameters on a real workload is still open (§14).

## 5. Kernels

### 5.1 Spec versus evaluator, and precision generics

`KernelSpec` (the declaration) is precision-independent. Parameters are always `f64` (values a user writes and reads should not depend on precision). `CompiledKernel<T>` (the evaluator) is compiled per `PrecisionPolicy::Storage`, and the inner arithmetic is `T`.

The optimizer sees only a flat `params: &[f64]` of log-`θ`. A composite kernel maps it back to the leaves in depth-first, left-to-right order.

```rust
pub struct ParameterBinding {
    pub index: usize,       // position in the concatenated kernel θ
    pub leaf_id: usize,     // leaf in depth-first, left-to-right order
    pub local_index: usize, // parameter inside that leaf
}

/// Declaration. Built-in leaves are stored directly; `+` / `*` build Sum / Product.
pub enum KernelSpec {
    Rbf(RbfKernel),
    RbfArd(RbfArdKernel),
    Matern(MaternKernel),
    MaternArd(MaternArdKernel),
    Periodic(PeriodicKernel),
    RationalQuadratic(RationalQuadraticKernel),
    RationalQuadraticArd(RationalQuadraticArdKernel),
    Constant(ConstantKernel),
    Linear(LinearKernel),
    White(WhiteKernel),
    Custom(CustomKernel), // a user KernelTerm, boxed for f64 and f32
    Sum(Box<KernelSpec>, Box<KernelSpec>),
    Product(Box<KernelSpec>, Box<KernelSpec>),
}

impl KernelSpec {
    pub fn custom<K>(term: K) -> Self; // K: KernelTerm<f64> + KernelTerm<f32> + Clone
    pub fn num_params(&self) -> usize;
    pub fn get_params(&self, out: &mut [f64]) -> Result<(), GprError>;
    pub fn set_params(&mut self, params: &[f64]) -> Result<(), GprError>; // atomic
    pub fn parameter_bindings(&self) -> Vec<ParameterBinding>;
    pub fn compile(&self) -> CompiledKernel<f64>;
    pub fn compile_as<T: KernelScalar>(&self) -> CompiledKernel<T>;
}

/// Evaluator. The same leaves as enum arms (static dispatch); Sum / Product
/// are flattened lists. Only `Custom` goes through a vtable.
pub enum CompiledKernel<T: KernelScalar = f64> {
    Rbf(RbfKernel),
    // … one arm per built-in leaf, as in KernelSpec …
    Custom(CustomKernel<T>),
    Sum(Vec<CompiledKernel<T>>),
    Product(Vec<CompiledKernel<T>>),
}

pub enum Triangle { Lower, Upper, Full }

/// A user leaf. It reads squared Euclidean distances (or coordinates for
/// `hess_points`) and writes the triangle `uplo` asks for.
pub trait KernelTerm<T: KernelScalar = f64>: Send + Sync + Debug + 'static {
    fn num_params(&self) -> usize;
    fn get_params(&self, out: &mut [f64]) -> Result<(), GprError>;
    fn set_params(&mut self, params: &[f64]) -> Result<(), GprError>;
    fn bounds_into(&self, out: &mut [Interval]) -> Result<(), GprError>;
    fn apply(&self, dist: MatRef<'_, T>, out: MatMut<'_, T>, uplo: Triangle) -> Result<(), GprError>;
    fn apply_cross(&self, dist: MatRef<'_, T>, out: MatMut<'_, T>) -> Result<(), GprError>;
    fn fill_diag(&self, out: &mut [T]) -> Result<(), GprError>;
    fn grad(&self, dist: MatRef<'_, T>, d_k: MatMut<'_, T>, param_idx: usize, uplo: Triangle)
        -> Result<(), GprError>;
    fn hess(&self, dist: MatRef<'_, T>, d2_k: MatMut<'_, T>, i: usize, j: usize, uplo: Triangle)
        -> Result<(), GprError>;
    fn hess_points(&self, x: MatRef<'_, T>, d2_k: MatMut<'_, T>, i: usize, j: usize, uplo: Triangle)
        -> Result<(), GprError>;
    /// Rectangular ∂K/∂θ and ∂²K/∂θ∂θ from squared distances (train × test).
    /// Default: `CoordGradientUnsupported`.
    fn grad_cross(&self, dist: MatRef<'_, T>, d_k: MatMut<'_, T>, param_idx: usize)
        -> Result<(), GprError> { Err(GprError::CoordGradientUnsupported) }
    fn hess_cross(&self, dist: MatRef<'_, T>, d2_k: MatMut<'_, T>, i: usize, j: usize)
        -> Result<(), GprError> { Err(GprError::CoordGradientUnsupported) }
    /// ∂k/∂(d²), ∂²k/∂(d²)², ∂²k/∂θ∂(d²): with these the coordinate derivatives of
    /// `FreeInducing` follow (k is a function of the squared distance). Default:
    /// `CoordGradientUnsupported`.
    fn grad_wrt_sq_dist(&self, dist: MatRef<'_, T>, out: MatMut<'_, T>) -> Result<(), GprError>
        { Err(GprError::CoordGradientUnsupported) }
    fn hess_wrt_sq_dist(&self, dist: MatRef<'_, T>, out: MatMut<'_, T>) -> Result<(), GprError>
        { Err(GprError::CoordGradientUnsupported) }
    fn grad_wrt_sq_dist_theta(&self, dist: MatRef<'_, T>, out: MatMut<'_, T>, param_idx: usize)
        -> Result<(), GprError> { Err(GprError::CoordGradientUnsupported) }
    fn clone_box(&self) -> Box<dyn KernelTerm<T>>;
    fn persist_id(&self) -> &'static str { "" }                          // registry key for save / load
    fn persist_state(&self) -> Result<serde_json::Value, GprError>;      // default: PersistFailed
}
```

`uplo` defaults to `Lower`: faer `cholesky_in_place` reads only the lower triangle, so filling `Full` would about double the kernel evaluation. `CompiledKernel` exposes the same operations (`apply`, `apply_cross`, `apply_points`, `apply_cross_points`, `fill_diag`, `fill_diag_points`, `grad`, `grad_points`, `grad_wrt_coord_dim`, `hess_wrt_coord_dims`, `hess_wrt_coord_mixed`, `hess_theta_coord_dim`, `hess`, `hess_points`), each generic over the `KernelMath` of §8.

Leaf parameters stay f64. `compile()` is f64; `compile_as::<T>()` gives the same apply, gradient, and Hessian in f32. Every built-in leaf is one implementation over `T: KernelScalar`, and `CompiledKernel<T>` has one dispatch. f64 SIMD paths are reached through a scalar hook that returns an f64 view only when `T = f64`; f32 is the same formula, scalar. A user leaf is one `impl<T: KernelScalar> KernelTerm<T>`: `KernelScalar` carries the arithmetic and `exp` / `ln` / `sqrt` / `powf` / `sin` / `cos` a formula needs, and the generic built-in leaves can be called from it. `CustomKernel::new` requires `KernelTerm<f64> + KernelTerm<f32>`, which one generic impl satisfies.

The model keeps the compiled tree next to its `KernelSpec` and recompiles it after a hyperparameter write.

**Lengthscale**: isotropic is a scalar `ℓ` (`θ=log(ℓ)`). ARD is a per-dimension `ℓ_d` (`θ_d=log(ℓ_d)`, `ArdLengthscales`) on RBF, Matérn, and RQ. The Periodic lengthscale is a scalar.

The ARD squared distance is `r² = Σ_d (x_d - x'_d)² / ℓ_d²`. It matches isotropic when every `ℓ_d` is equal. `∂K/∂θ_d` needs the per-dimension difference, so the isotropic squared-distance matrix is not enough: ARD leaves read coordinates, or the raw `(Δx_d)²` cache of §5.2.

A user kernel (`Custom`) is encouraged not to allocate on the hot path, and that is not enforced (§2). A user kernel does not receive the Workspace. There is no pair of a safe API and an unsafe fast API.

On the hot path (the double loop of distances and kernel evaluation) `CompiledKernel` is dispatched with `match`. Only `Custom` goes through the vtable. That matches §2: static dispatch by default, `dyn` only at the extension point.

The optimizer's parameter array is concatenated in this order:

```
[kernel_params | likelihood_params]
```

`Sgpr` with `FreeInducing` appends the column-major inducing coordinates `Z` after these (§6.1).

### 5.2 Distance cache and cache policy

Training coordinates do not change during a fit, so the pairwise distances are computed once and reused while the kernel is rebuilt at each `θ`.

```rust
pub enum DistanceCachePolicy {
    Cached,   // default: fill once per fit, reuse (the speed pole)
    Uncached, // recompute from X on every kernel build (the memory pole)
}

/// Crate-private. What Cached stores (§7.1).
struct DistCache<S> {
    dist: Option<Mat<S>>,        // n×n squared Euclidean, for distance-mode leaves
    ard_sq_diff: Option<Mat<S>>, // raw (Δx_d)² as n × (n·d), for ARD leaves
}
```

The stored intermediates are the squared Euclidean distance (isotropic RBF / Matérn / RQ / Periodic / a user leaf) and the raw per-dimension `(Δx_d)²` (ARD leaves). An `r²` that already includes `ℓ` is not stored. The ARD layout is column-major `n × (n·d)`: dimension `k` is columns `[k n, (k+1) n)`, and each block is lower triangular. Both slots are filled on first use and only when the compiled kernel reads them: `RBF + White` and `Constant * RBF` fill `dist`; standalone Linear / Constant / White fill nothing, and their policy is kept but unused. A train × query or LOO cache does not exist.

The policy is a runtime enum because no combination with the other policies is illegal (§6.3). An `(n,n,d)` tensor is `n²×d×sizeof(T)` bytes; `K` itself is `n²×sizeof(T)` (about 200MB at n=5000, f64), and an ARD cache is `d` times that. Choosing the policy from `n`, `d`, and a memory budget is open (§14).

### 5.3 Evaluating a composite kernel

`KernelSpec::compile` flattens associative chains: `(A+B)+C` becomes `Sum(vec![A, B, C])`, and a product likewise. A product nested in a sum (or the reverse) stays nested.

Each compiled tree has a coordinate mode (crate-private `CoordMode`): `Dist` (isotropic leaves read squared distances), `Points` (ARD and Linear leaves read coordinates), `Either` (Constant and White read only the shape), or `Mixed` (a Sum / Product of Dist and Points leaves, for example `RBF + Linear`). Mixing is not a runtime error and not a type ban: a Dist leaf stays on distances and a Points leaf stays on coordinates.

Evaluation walks the flattened list. A Sum writes its first term into `out` and adds each later term through one output-shaped `scratch`; a Product multiplies likewise. A term that is itself a multi-term Sum / Product needs one more output-shaped buffer per nesting level (`CompiledKernel::nested_depth`); the fit and predict paths take those levels from the Workspace (§7.1). The count is measured from the tree, not assumed: `(A*B)*(C*D)` flattens to one product, while `A*(B+C*D)` needs a level for `B+C*D` and another for `C*D`. A built-in leaf is a `match` arm; only `Custom` calls `dyn KernelTerm`.

### 5.4 Partial updates (coordinate optimizers)

Whether a fit rebuilds only the touched kernel leaves is not a type. It is derived at run time: `O::USES_CHANGE_INDICES && cholesky_buffer == CholeskyBuffer::Retain`. In Exact GPR, Cholesky is O(n³), so a partial update only helps while building the kernel matrix.

```rust
pub trait Optimizer<P: ?Sized> {
    fn minimize(&self, objective: &mut P, init: &[f64]) -> Result<OptResult, GprError>;
    const USES_CHANGE_INDICES: bool = false; // FastSimulatedAnnealing sets true
}

pub trait Objective {
    fn num_params(&self) -> usize;
    fn value(&mut self, params: &[f64]) -> Result<f64, GprError>;
    fn value_at_changes(&mut self, params: &[f64], indices: &[usize]) -> Result<f64, GprError> {
        self.value(params) // default: full rebuild
    }
}

pub trait IncrementalObjective: Objective {
    fn value_with_changes(&mut self, params: &[f64], indices: &[usize]) -> Result<f64, GprError>;
}
```

`indices` lists every coordinate that differs from the `params` of the previous evaluation on that objective, not from the optimizer's accepted point: after a rejected proposal, FSA passes the coordinate it reverted together with the new one. The indices alone decide which leaves are rebuilt. `GprObjective` also compares the previous and new `θ` bit for bit (`f64::to_bits`), and a changed coordinate that `indices` does not list is rejected with `IndexOutOfRange` instead of silently giving a wrong value. Empty, duplicate, and `i >= n_params` indices are `GprError` at the boundary. There is no `ChangeSet` type.

`GprObjective` implements `IncrementalObjective` for every optimizer and buffer. `value` / `value_at_changes` take the leaf path only when the flag above is set; otherwise they run the full joint evaluation. `CholeskyBuffer::Reuse` always rebuilds everything. `with_optimizer` / `refit` re-derive the flag from the new optimizer. `Gpr<Fixed>::factor` is one full pass. `Lbfgs` / `NelderMead` / `TrustRegion` keep the default `false`. The first evaluation and a restart of FSA use `value`; one coordinate step uses `value_at_changes`.

The leaf rebuild caches compiled leaves and reapplies only the leaves a changed index touches. The per-leaf Grams (`L · n²` for `L` leaves), the dirty flags, and the previous `θ` live on `GprObjective` (crate-private `LeafCache`) for one `fit` / `refit`, not on the Workspace. The tree combination and **Cholesky are full every time**: a hyperparameter change is not a low-rank update of `K`, and faer's `rank_r_update_clobber` is not used for it.

### 5.5 Preprocessing pipeline

Split X and y. When a GPR has no mean function, **standardizing y to mean 0 and variance 1 is the basic numerical step**. Predictions are mapped back to the original scale.

```rust
/// Unfitted input map. `fit` consumes it and returns the fitted map, so an
/// unfitted `apply` cannot be written. `x` is column-major `n_rows × n_cols`.
pub trait UnfittedTransform: Send + Sync {
    fn fit(self: Box<Self>, x: &[f64], n_rows: usize, n_cols: usize)
        -> Result<Box<dyn Transform>, GprError>;
    fn clone_box(&self) -> Box<dyn UnfittedTransform>;
    fn as_any(&self) -> &dyn Any;
    fn persist_id(&self) -> Option<&'static str> { None }   // Some for a user map
    fn persist_state(&self) -> Result<serde_json::Value, GprError>;
}

/// Fitted input map. Every map is invertible.
pub trait Transform: Send + Sync {
    fn apply(&self, x: &mut [f64], n_rows: usize, n_cols: usize) -> Result<(), GprError>;
    fn inverse_apply(&self, x: &mut [f64], n_rows: usize, n_cols: usize) -> Result<(), GprError>;
    fn clone_box(&self) -> Box<dyn Transform>;
    // as_any / persist_id / persist_state as above
}

/// Unfitted target map, fitted the same way.
pub trait UnfittedTarget: Send + Sync {
    fn fit(self: Box<Self>, y: &[f64]) -> Result<Box<dyn TargetTransform>, GprError>;
    // clone_box / as_any / persist_id / persist_state
}

pub trait TargetTransform: Send + Sync {
    fn transform(&self, y: &mut [f64]) -> Result<(), GprError>;
    fn inverse_transform_mean(&self, mean: &mut [f64]) -> Result<(), GprError>;
    fn inverse_transform_variance(&self, var: &mut [f64]) -> Result<(), GprError>;
    fn inverse_transform_covariance(&self, cov: &mut [f64]) -> Result<(), GprError> {
        self.inverse_transform_variance(cov) // affine maps scale every entry by s²
    }
    // clone_box / as_any / persist_id / persist_state
}
```

The built-in maps are unfitted / fitted pairs: `IdentityInput`, `StandardizeInput` / `FittedStandardizeInput`, `MinMaxInput` / `FittedMinMaxInput`, `ColumnwiseInput` / `FittedColumnwiseInput`, `Pipeline` / `FittedPipeline` for `X`; `IdentityTarget`, `StandardizeTarget` / `FittedStandardizeTarget`, `MinMaxTarget` / `FittedMinMaxTarget`, `TargetPipeline` / `FittedTargetPipeline` for `y`. Transforms work in `f64` whatever the model precision.

The default `Gpr` is Identity on both sides. When the mean function is zero, `StandardizeTarget` is the basic numerical step. `MinMaxInput` / `MinMaxTarget` scale to an interval (default `[0, 1]`). A series of maps is `Pipeline` (`X`) or `TargetPipeline` (`y`); a single `with_input_transform` / `with_target_transform` takes one map. `ColumnwiseInput` gives each column its own map (a uniform column is MinMax, a near-normal column is Standardize); a length other than `d` is an error. `predict` computes latent or observation variance in the transformed space, then returns through `inverse_transform_mean` / `inverse_transform_variance`. For an affine `y' = (y - a)/s`, the inverse variance is `Var(y) = s² Var(y')`. A user map is saved through `persist_id` / `persist_state` and restored through `PersistRegistry`.

`Sgpr` / `Svgp` take the same transforms with the same identity default. The inducing points `Z` are passed in the coordinates of `X` and go through the fitted input map with `X`. Queries and points inserted into `OnlineSgpr` go through the maps fitted at training. `FreeInducing` searches `Z` in the mapped coordinates. The fitted model reports `Z` in the original coordinates through `Transform::inverse_apply`.

## 6. GP model: swapping exact and sparse

Training and inference are different types. An unfitted `predict` is not on the public API. sklearn's same-object `fit` / `predict` is a numerical-check target, not the public contract. The fit objective borrows the model only during `fit` / `refit` and is not kept on the fitted value.

```rust
impl<O, P: GpScalar> Gpr<O, P> where O: for<'a> Optimizer<GprObjective<'a, P>> {
    pub fn fit(self, x: &[f64], n_rows: usize, n_cols: usize, y: &[f64])
        -> Result<FittedGpr<O, P>, (Self, GprError)>;
}
impl<P: GpScalar> Gpr<Fixed, P> {
    pub fn factor(self, x: &[f64], n_rows: usize, n_cols: usize, y: &[f64])
        -> Result<FittedGpr<Fixed, P>, (Self, GprError)>;
}

/// Fitted: L, α, X, kernel, likelihood, transforms. No W and no optimizer state.
impl<O, P: GpScalar> FittedGpr<O, P> {
    pub fn predict(&self, xs: &[f64], n_rows: usize, n_cols: usize)
        -> Result<Prediction<P::Refine>, GprError>;
    pub fn predict_into(&mut self, xs: &[f64], n_rows: usize, n_cols: usize,
        out: &mut Prediction<P::Refine>) -> Result<(), GprError>;
    pub fn predict_covariance(&self, xs: &[f64], n_rows: usize, n_cols: usize)
        -> Result<PredictiveCovariance<P::Refine>, GprError>;
    pub fn sample(&self, xs: &[f64], n_rows: usize, n_cols: usize, n_draws: usize, seed: u64)
        -> Result<Vec<P::Refine>, GprError>;
    pub fn loo_predict(&self) -> Result<Prediction<P::Refine>, GprError>;
    pub fn neg_log_marginal_likelihood(&self) -> Result<f64, GprError>;
    pub fn refit(&mut self) -> Result<(), GprError>; // re-optimize (O) or refactor (Fixed) on the same data
    // *_with variants take PredictOptions; get_params / set_params /
    // value_and_gradient_into / hessian_into; into_online / into_trainer / save
}

pub enum VarianceKind {
    Latent,      // variance of the latent f* (no noise)
    Observation, // variance of the observation y* (includes σn²). Default
}

pub struct Prediction<T = f64> {
    pub mean: Vec<T>,
    pub variance: Vec<T>,
    pub variance_kind: VarianceKind,
}

pub struct PredictiveCovariance<T = f64> {
    pub mean: Vec<T>,
    pub covariance: Vec<T>, // column-major m × m; its diagonal is Prediction::variance
    pub variance_kind: VarianceKind,
}

pub struct PredictOptions {
    pub variance_kind: VarianceKind, // default Observation
}
```

`Gpr` (Exact), `Sgpr`, and `Svgp` each return their own fitted type. Hyperparameter optimization goes through the objective traits (§9) and only during `fit` / `refit`.

`predict` returns the diagonal variance. Covariance between queries and posterior draws are separate calls (`predict_covariance`, `sample`), never a flag on `predict`. The `*_with` variants take `PredictOptions`; the others use `Observation` (what a user usually wants is the noisy predictive variance). `sample` returns `n_draws` draws of length `m` in one `Vec` (`n_draws × m` values).

### 6.1 Inducing-point cache for Sparse GPR

The Sparse approximation is VFE. The reason is [ADR 0002](adr/0002-sparse-vfe.md). FITC is not shipped. SVGP is a separate public type (`Svgp` / `FittedSvgp`). The reason is [ADR 0006](adr/0006-sparse-svgp.md). `Svgp<Fixed>::factor` LLTs `K_mm` at the caller's `Z` and places a whitened `q(u)` at the prior (mean 0, `L = I`). `Svgp<Adam>::fit` starts from that prior and moves kernel `θ`, likelihood `θ`, and the whitened `q` with minibatch Adam. A step forms `A_b = L⁻¹ K(Z, X_b)`, `k_diag`, and `∂K(Z, X_b)/∂θ` for its own points only and refactors `K_mm`, so it costs `O(b (m² + m d) + m³)` and nothing in it scales with `n`; `A` and `k_diag` of all `n` points are built once before the first step and once after the last. The gradient is computed in `f64` whatever the storage scalar is. `Adam` is not an `Optimizer`. `FittedSvgp` returns diagonal `predict` / `predict_with` (and `predict_into` / `predict_with_into`), `neg_elbo`, and a full-data `value_and_gradient_into`. At the optimal `q` (Titsias) it matches `FittedSgpr` at the same `θ`, `X`, and `Z`. The public types are `Sgpr` / `FittedSgpr`. The default is `Sgpr<Lbfgs, FixedInducing>`. `fit` searches kernel and likelihood `θ`. `Sgpr<Fixed, I>::factor` LLTs `K_mm = k(Z, Z)` at the caller's inducing locations `Z`. By default `Z` is not in params. `fit` after `with_inducing(FreeInducing)` moves kernel `θ`, likelihood `θ`, and column-major `Z` together in the same `Optimizer`. `FittedSgpr` returns diagonal `predict` / `predict_with` (and `predict_into` / `predict_with_into`), `neg_log_marginal_likelihood` (the negative VFE ELBO), `value_and_gradient_into`, and `hessian_into` (row-major `p×p`). At `Z = X` it matches Exact `Gpr<Fixed>::factor`. There is no k-means. External checks are §12 (5c, 5d); wall time and RSS are §15.

The diagonal of `K(X,X)` is invariant, so it is computed once and reused. `K(X,Z)` and `K(Z,Z)` must be recomputed whenever Z moves, but `m` (the number of inducing points) is small, so that cost is negligible next to the O(nm²) Cholesky and is not cached. The joint gradient and Hessian of `K(X,X)` sum the diagonal `∂k(x_i, x_i)/∂θ` in `O(n)`. Gradients of `K(Z,Z)` and `K(Z,X)` stay dense.

The gradient of inducing coordinates is `grad_wrt_coord_dim` and the Hessian is `hess_wrt_coord_dims` / `hess_wrt_coord_mixed` / `hess_theta_coord_dim` (§5.1). Every built-in leaf has them, Sum and Product trees compose them (the radial leaves from `g'(q)`, `g''(q)` of `k = g(q)`, `q = Σ w_d Δ_d²`, one implementation; a Product by the product rule over each term's value and first and second derivative). Matérn with `ν = 1/2` returns `GprError::CoordGradientUnsupported`, not a panic: its coordinate derivative is undefined where two points coincide, and `Z ⊂ X` starts there. A `Custom` leaf provides them through `grad_wrt_sq_dist`, `hess_wrt_sq_dist`, and `grad_wrt_sq_dist_theta`, and the rectangular `∂K/∂θ` through `grad_cross` / `hess_cross`; a leaf that leaves the defaults returns `CoordGradientUnsupported`. The default `FixedInducing` `fit` does not call this API: it needs the rectangular `∂K(Z, X)/∂θ` and `∂²K(Z, X)/∂θ∂θ` (`grad_cross_points` / `hess_cross_points`), which every built-in leaf and every Sum / Product tree of them provides, so a `Constant × RBF` signal variance fits in `Sgpr` and `Svgp` as in Exact. A `Custom` leaf needs `KernelTerm::grad_cross` / `hess_cross` (see above). `FreeInducing` calls the coordinate API once per dimension during joint optimization. The reason is [ADR 0003](adr/0003-sparse-z-joint.md).

**The default is: the caller passes Z, and the optimization targets are kernel hyperparameters and noise only.** Free Z switches with `FixedInducing` / `FreeInducing`. The same `Optimizer` moves kernel `θ`, likelihood `θ`, and column-major `Z` together. The interval is the raw coordinates of the training-`X` box, opened a little. L-BFGS history length is `p = p_θ + m×d`, and the extra storage is `history_size × m × d` values of `f64` (small next to the VFE `O(nm²)`, because `m` is small). Alternating is not shipped.

Online can add and remove both X and inducing points. `FittedSgpr::into_online` returns `OnlineSgpr<O>` (no inducing typestate). `insert` / `delete` update the VFE factor by the rank-1 of ADR 0004. `insert_inducing` / `delete_inducing` are [ADR 0005](adr/0005-sparse-inducing-update.md) (insert is a bordered LLT, delete is a trailing cholupdate). The identifier is `InducingId`. Coordinates come from the caller. `Z` is not in params. `set_params` and `refit` fully reassemble.

**Parity with Exact.** The sparse models follow `Gpr` for the features below.

| Feature | `FittedSgpr` / `OnlineSgpr` | `FittedSvgp` |
| --- | --- | --- |
| Input / target transforms | yes | yes |
| `JitterPolicy` of `K_mm` (default `adaptive(1e-8, 10, 5, 1e-3)`, because `K_mm` has no noise, §4.0) | yes | yes |
| `predict_into` with no allocation after warmup | yes | yes |
| Predictive covariance and posterior sample | yes | yes |
| Leave-one-out | yes | no |
| Save and load | yes | yes |

The sparse `predict_covariance` / `sample` run the `predict` path and build the off-diagonal from its buffers: VFE `K** − A*ᵀA* + σn² S*ᵀS*` with `A* = L_mm⁻¹ K_m*` and `S* = L_B⁻¹ A*`, SVGP `K** − AᵀA + UᵀU` with `U = L_qᵀ A`. The diagonal is `predict`'s variance, bit for bit. The draws are the Exact draw (`μ + L z`, the model's `JitterPolicy` on the factor).

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

`(n/2) log(2π)` does not depend on θ, and adding it costs nothing measurable, so the public NLML and the optimization objective are the same `L(θ)`. The API is not split.

Standard algorithm (Rasmussen & Williams / the GPy family):

1. Build `A = K + σn² I` into `k_matrix` (lower triangle only, `uplo=Lower` from §5.1)
2. In-place Cholesky. `k_matrix` becomes L
3. `log|K| = 2 Σ log(L_ii)` from the diagonal of L
4. Solve `L Lᵀ α = y` by forward and back substitution (O(n²))
5. Compute `K⁻¹` from `L` (triangular solves of `L Lᵀ X = I`, one O(n³))
6. `W[i,j] ← α[i] α[j] - K⁻¹[i,j]` (symmetric, so lower triangle only)
7. For each θ_i, evaluate `∂K/∂θ_i` into `exp_buf` and accumulate `⟨W, ∂K/∂θ_i⟩_F` in O(n²). Kernel parameters use `KernelTerm::grad`. Noise uses `Likelihood::noise_grad_diag` (diagonal only)

Total cost is O(n³ + p n²). `K⁻¹` is not rebuilt per parameter.

Analytic NLML Hessian:

```
H_ij = -½ ⟨W, ∂²K/∂θ_i∂θ_j⟩ - ½ Tr(K⁻¹ K_i K⁻¹ K_j) + αᵀ K_i K⁻¹ K_j α
```

`KernelTerm::hess` / `hess_points` write `∂²K` for one pair `(i, j)`. Custom, Sum, and Product are analytic. `FittedGpr::hessian_into` is the public entry, and `GprObjective` forwards to `TwiceDifferentiable`. `Q_j` (one `n×n`) and four length-`n` vectors live in `WorkspaceCore::hessian`: empty until the first Hessian, reused after it, so a Hessian after the first allocates nothing. `CholeskyBuffer::Reuse` Chols again after ⟨W, K_ij⟩ and solves the first-order term `Q_i = K⁻¹ K_i`.

`value_and_gradient_into` runs this once and shares L, α, and `exp_buf` between the likelihood and the gradient. The default two-step `value` then `gradient_into` does not share them.

The default `CholeskyBuffer` is `Retain`. `K⁻¹` → `W` is written into a dedicated `w_matrix`, and `L` stays in `k_matrix`. Speed does not change. The public memory pole (`with_prefer_memory`) selects `CholeskyBuffer::Reuse`. `with_cholesky_buffer` sets it alone. `Reuse` solves `K⁻¹` in `exp_buf` and writes `W` into the Cholesky region. It does not restore `L` in the middle of the optimization loop. It Chols again at the end of `fit` and at the end of a standalone `value_and_gradient_into`. persist does not write this policy. `load` is `Retain`.

### 6.3 Exact GPR (`Gpr` / `FittedGpr`)

The public surface splits the trainer from the fitted model.

`Gpr<O = Lbfgs, P = DoublePrecision>` holds a `KernelSpec`, a `GaussianLikelihood`, transforms, an optimizer `O`, and four runtime policies: `DistanceCachePolicy { Cached, Uncached }`, `CholeskyBuffer { Retain, Reuse }`, `KernelExp { Accurate, FastApprox }`, and `JitterPolicy` (§4.0). They are plain enums. No combination is illegal, so none is a type parameter. The `Gpr::new` default is the speed pole (`Cached` + `Retain`) with `Accurate`. The public switch is `with_prefer_memory` / `with_prefer_speed`. The memory pole is `Uncached` + `Reuse`. `with_distance_cache_policy` / `with_cholesky_buffer` / `with_math` / `with_jitter_policy` set one policy each. A kernel that never reads pairwise distances (standalone Linear / Constant / White) allocates no distance cache whatever the policy says; there is no `from_points`. `FittedGpr` has no `with_prefer_*` (`into_trainer` → prefer → `refit`); it exposes the policies through getters. The types stay at the crate root. `Gpr<O: Optimizer>::fit(self, …)` moves hyperparameters with `O` and, on success, returns `FittedGpr<O, P>`. Fixed hyperparameters are `Gpr<Fixed>::factor`. There is no `optimize: bool`. On failure the consumed `Gpr<O, P>` is returned with the error. There is no `fitted: bool` and no `GprError::NotFitted`. An unfitted `transform` / `apply` cannot happen (`StandardizeTarget::fit(self)` returns `FittedStandardizeTarget`). On `FittedGpr`, `L` / `α` / `X` / the compiled kernel are not `Option`, so no call can find a missing piece.

`FittedGpr` holds what inference needs: `L`, `α`, training `X`, the kernel, the likelihood, and the transforms. `W`, `∂K`, and argmin state live only during `fit` and are not kept on the fitted value. Calling `predict` in the same process immediately after `fit` is treated as the minority path. The main path hands over a fitted model, so the inference object is `FittedGpr`.

`FittedGpr` and `OnlineGpr` share one crate-private `GprCore` (kernel spec, compiled kernel, likelihood, transforms, policies, training data, `α`, query buffers) and differ only in the factor: `FittedGpr` holds the LLT buffers (crate-private `LltStore`: `FitBuffers`, or a memory-mapped `L`), `OnlineGpr` holds the LDLT `LdltStore` and the `PointId` table. `StoredFactor { Llt, Ldlt }` is the factor view: `solve`, `L⁻¹` on columns, the per-pivot weight (`1` or `1/Dᵢ`), `log|A|`, and `diag(A⁻¹)`. Predict, covariance, sampling, LOO, NLML, and the predict `α` are written once on `GprCore` against that view. Every hyperparameter write (`set_params`, gradient, Hessian, `fit`, `refit`) runs on one borrowed `ExactFit` view (core + LLT buffers). `OnlineGpr` lends it temporary LLT buffers filled with `L √D` in O(n²) and writes the new factor back; the training data is not copied, and the only O(n³) work is the refactor the new `θ` needs.

The default `Gpr` is `Gpr<Lbfgs>`. `with_optimizer` replaces `O` with another argmin solver (`NelderMead`, `TrustRegion`), `FastSimulatedAnnealing`, or a user optimizer. Leaf rebuilds follow §5.4; there is no `with_recompute_strategy`. `Gpr<Fixed>::factor` only factors. `FittedGpr::predict` is a diagonal variance, and covariance between queries is `predict_covariance` (§6). `loo_predict` returns per-training-point LOO from `L` and `α` as in GPML 5.4.2. Refactoring the same data at new hyperparameters is `FittedGpr::refit` (the fitted value keeps its `O`). `with_optimizer` / `factor` / `into_trainer` / `refit` keep the policies.

```rust
pub struct Gpr<O = Lbfgs, P = DoublePrecision> {
    kernel: KernelSpec,
    likelihood: GaussianLikelihood,
    x_transform: Box<dyn UnfittedTransform>,
    y_transform: Box<dyn UnfittedTarget>,
    optimizer: O,
    policies: Policies,
    _precision: PhantomData<P>,
}

/// Crate-private. Shared by the trainer and its fitted models.
struct Policies {
    distance_cache: DistanceCachePolicy, // Cached (default) | Uncached
    cholesky_buffer: CholeskyBuffer,     // Retain (default) | Reuse
    math: KernelExp,                     // Accurate (default) | FastApprox
    jitter: JitterPolicy,                // fixed(0.0) (default)
}

pub struct Fixed; // Gpr<Fixed>::factor only; not an Optimizer

// Optimizers. Fields are private; each has `Default` and `with_*` setters.
pub struct Lbfgs       { max_iterations: u64, tolerance: f64, history_size: NonZeroUsize, restarts: Option<Restarts> }
pub struct TrustRegion { max_iterations: u64, tolerance: f64, initial_radius: f64, max_radius: f64, restarts: Option<Restarts> }
pub struct NelderMead  { max_iterations: u64, tolerance: f64, restarts: Option<Restarts> }
pub struct FastSimulatedAnnealing {
    max_iterations: u64, restarts: Option<Restarts>,
    initial_temperature: f64, cooling_rate: f64, seed: u64, boundary: BoundaryPolicy,
}
// Defaults: max_iterations 100, tolerance √ε, history_size 10, gamma 1, no restarts;
// FSA initial_temperature 1, cooling_rate 3, seed 0, BoundaryPolicy::Clamp.

pub struct FittedGpr<O = Lbfgs, P: GpScalar = DoublePrecision> {
    core: GprCore<P>,  // crate-private: everything but the factor
    optimizer: O,
    store: LltStore<P>, // FitBuffers<P> (L; dist / W per policy), or a mapped L
}

/// Crate-private. Shared by FittedGpr and OnlineGpr.
struct GprCore<P: GpScalar> {
    kernel: KernelSpec,
    compiled: CompiledKernel<P::Storage>,
    likelihood: GaussianLikelihood,
    x_unfitted: Box<dyn UnfittedTransform>, // kept for into_trainer / refit
    y_unfitted: Box<dyn UnfittedTarget>,
    x_transform: Box<dyn Transform>,
    y_transform: Box<dyn TargetTransform>,
    policies: Policies,
    query: QueryWorkspace<P>,  // predict_into buffers (§7.1)
    x_obs: Vec<f64>,           // caller-scale X, column-major n × d
    y_obs: Vec<f64>,
    x: Mat<f64>,               // transformed X (rows past n are online capacity)
    y_train: Vec<f64>,         // transformed y
    factor_alpha: Vec<P::Storage>, // the factor's solve; the NLML reads it
    alpha: Vec<P::Refine>,     // predict weights (refined for MixedPrecision)
    n: usize,
    d: usize,
    // plus the f32 cast scratch of the storage scalar
}

/// Crate-private fit objective. Borrows the model only during fit / refit and
/// writes θ through to its kernel and likelihood.
struct GprObjective<'a, P: GpScalar = DoublePrecision> {
    model: ExactFit<'a, P>,       // crate-private view: &mut GprCore + &mut LltStore
    scratch: Vec<f64>,
    leaves: LeafCache<P::Storage>, // §5.4
    incremental: bool,             // §5.4
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

The default distance-cache policy is `Cached`. At the start of `fit` the squared distances of the training points are filled once, and later hyperparameter iterations rewrite only the kernel. Isotropic is `n×n`. ARD stores raw `(Δx_d)²` as `n × (n·d)`. `Uncached` does not put those tensors on the Workspace, and both isotropic and ARD compute distances from `X`. The public memory pole is `with_prefer_memory` (`Uncached` + `Reuse`). The speed pole stays the default (`with_prefer_speed`). `Uncached` + `Retain` is the lowest-RSS setting that keeps leaf rebuilds (`with_distance_cache_policy` alone). The cache is allocated only when the compiled kernel reads distances: `RBF + White` and `Constant * RBF` do; standalone Linear / Constant / White do not, and their policy is kept but unused. persist tags are `always` / `never` (a missing tag loads as `Cached`). `LoadedGpr` has one variant per precision and factor kind (8), because both are type parameters of the model. `predict` / `predict_with` (widened to `f64`), `n`, `d`, and `is_online` work on any variant without a `match`; a variant is matched only for the typed model (`predict_into`, `insert`, `refit`). `load` is `Retain`.

### 6.4 Leave-one-out

Leave-one-out for Exact GPR is a closed form from the fitted `L` and `α` (Rasmussen & Williams, GPML §5.4.2). With `A = K + σn² I`, `Q = A⁻¹`, and `α = A⁻¹ y`:

```
μ_i = y_i - α_i / Q_ii
σ_i² = 1 / Q_ii
```

This is the observation `p(y_i | X, y_{-i}, θ)`. The LOO variance of the latent `f_i` is `max(0, 1/Q_ii - σn²)`. `Q_ii` is the column norm of `L⁻¹` from the lower-triangular `L` (`A⁻¹ = L^{-T} L^{-1}`). The cost is the same order as Cholesky, O(n³), and the extra memory is one temporary `n×n`.

`FittedGpr::loo_predict` returns a `Prediction` of the same length as the training points. The default is `VarianceKind::Observation`. Mean and variance are mapped back to the original scale by `TargetTransform`, as in `predict`. A White leaf is not used. Noise is `GaussianLikelihood` only.

sklearn has no LOO API. `just gen-goldens` applies the same GPML formula to `L_` / `alpha_` after fit and writes JSON. The Rust side factors at the `θ` sklearn chose with `Gpr<Fixed>::factor` (optimizer differences are not mixed into LOO).

`FittedSgpr` / `OnlineSgpr::loo_predict` is the LOO of the collapsed VFE posterior at fixed `θ` and `Z`: the optimal `q(u)` without point `i`, predicted at `x_i`. With `A = L_mm⁻¹ K_mn`, `B = σn² I + A Aᵀ`, and `w = B⁻¹ A y`, leaving out `i` subtracts `a_i a_iᵀ` from `B`. Sherman–Morrison with `h = a_iᵀ B⁻¹ a_i` and `g = a_iᵀ w` gives

```
μ_i = (g - h y_i) / (1 - h)
latent σ_i² = k(x_i, x_i) - ‖a_i‖² + σn² h / (1 - h)
```

One triangular solve `L_B⁻¹ A` makes the pass `O(n m²)`. At `Z = X` it is the Exact LOO. An `f32` storage assembles the VFE system again in `f64`, as its prediction does. SVGP has no LOO (§6.1).

## 7. Workspace and memory

### 7.1 Separate buffers

The buffer count is small and fixed, so they are separate fields. Storage and Refine of the precision policy are explicit.

```rust
/// Crate-private (as are all types in this block).
struct WorkspaceCore<P: PrecisionPolicy> {
    k_matrix: Mat<P::Storage>,       // A = K + σn² I, then L. W during a Reuse gradient
    exp_buf: Mat<P::Storage>,        // kernel evaluation, ∂K/∂θ. Reuse n-RHS lives here
    kernel_scratch: Mat<P::Storage>, // product / custom ∂K/∂θ. Empty until a tree needs it
    thread_scratch: Vec<Mat<P::Storage>>, // one per Rayon worker, detached before a parallel fill
    rhs: Mat<P::Storage>,            // n×1, training Cholesky right-hand side y → α
    faer_scratch: MemBuffer,         // faer's own scratch, used as-is
    theta: Vec<f64>,                 // θ before the current write, restored when A does not factor
    nested: Vec<Mat<P::Storage>>,    // one n×n per nesting level of a sum / product (§5.3)
    hessian: HessianScratch<P::Storage>, // Q_j and four n-vectors. Empty until the first Hessian (§6.2)
    factor_jitter: f64,              // j of the last successful factor (§4.0)
}

struct FitBuffers<P: PrecisionPolicy> {
    core: WorkspaceCore<P>,
    dist: Option<DistCache<P::Storage>>, // Some for Cached (§5.2)
    w_matrix: Option<Mat<P::Storage>>,   // Some for Retain. W = ααᵀ - K⁻¹ (§6.2)
}

/// Held by FittedGpr / OnlineGpr. The first predict_into sizes it to (n, m, d).
struct QueryWorkspace<P: PrecisionPolicy> {
    query_xs: Vec<f64>,                 // transformed query (column-major)
    query_x: Mat<P::Storage>,           // m×d
    query_k_star: Mat<P::Storage>,      // k(X, X*), then L⁻¹ k_* (n×m)
    query_scratch: Mat<P::Storage>,     // apply_cross scratch (n×m)
    query_nested: Vec<Mat<P::Storage>>, // nested sum / product levels of the n×m block
    query_dist: Mat<P::Storage>,        // train–query squared distances (n×m)
    query_kss: Vec<P::Storage>,         // k(x*_j, x*_j)
}
```

A sum or product whose term is itself a multi-term sum or product needs one more output-shaped buffer per nesting level (`CompiledKernel::nested_depth`). The crate-internal fit / predict entry points take those levels from `nested` / `query_nested`, which grow on the first call and are reused after. The public `CompiledKernel::apply` / `grad` / `hess` family keeps its signature and builds the levels for that one call. The diagonal folds (`fill_diag`, `fill_diag_points`, and their gradients and Hessians) combine terms in fixed-size stack blocks of rows and allocate nothing. The sparse models (`FittedSgpr`, `OnlineSgpr`, `FittedSvgp`) keep their kernel scratch (output-shaped scratch, nested levels, train–query distances) in a crate-private `SparseScratch` between `&mut self` calls (`set_params`, gradient, Hessian, online updates). Their fits still return new matrices for their factors, so `tests/alloc.rs` records those counts as a ratchet rather than zero. Their `predict_into` keeps the prediction buffers there too: the mapped query, the packed `Z` and queries, `K(Z, X*)` and its solves, and the compiled kernel, rebuilt only when the kernel changes. A rounding storage predicts through `f64` buffers (`K_mm` factored in `f64`, `B` promoted). After a warmup call with the same shapes it allocates nothing, except the mixed-precision SVGP mean, whose per-query refinement keeps three `f64` vectors of length `m`; its `f64` reference is built once per call. `predict` (`&self`) runs the same path with buffers of its own, so both return the same values.

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

A batch-fit Workspace has fixed n. Capacity growth for online learning belongs to `LdltStore` (§11), and its memory policy is separate from the batch Workspace. `FitBuffers`, `QueryWorkspace`, `LdltStore`, and faer types are crate-private.

## 8. Parallelism, SIMD, and the math backend

- The inner kernel loop is vectorized with `wide::f64x4`. The targets are column-major, unit-row-stride isotropic RBF `apply` / `grad` / `apply_cross`, ARD RBF `apply` / `grad`, and the row loops of squared distance and `(Δx_d)²`. A view whose stride is not 1 falls back to scalar. `std::simd` is not used until it is stable. The inner loops of Matérn / Periodic / RQ are not vectorized yet.
- Distance-matrix and kernel-matrix construction is block-parallel with Rayon
- faer is itself Rayon-parallel, so do not nest a second pool. Share one `rayon::ThreadPool`. faer's thread count is `min(pool, n/64, n·k/16384, k/12)` ([ADR 0001](adr/0001-faer-parallel-degree.md)). `k` is the number of right-hand-side columns. Kernel fill uses the whole pool

**The kernel `exp` starts from a minimal API, and the default is an accurate implementation, not an approximation.** An approximation error in the kernel matrix reaches positive-definiteness, Cholesky stability, the likelihood, the gradient, and the prediction.

```rust
/// Runtime choice on every model, set by with_math.
pub enum KernelExp { Accurate, FastApprox }

/// Zero-sized markers the kernel code is monomorphized over. `KernelMath` is
/// sealed through the crate-private `MathOps` (exp, its jet, and the f64x4 forms).
pub struct Accurate;   // f64::exp / f32::exp / wide::exp
pub struct FastApprox; // degree-7 polynomial exp in the storage scalar
pub trait KernelMath: MathOps {}
```

The default is `Accurate`. `FastApprox` replaces `exp` in kernel evaluation, fit and predict alike. `exp(θ)` that maps a lengthscale back, and the `KernelTerm` formulas, stay on the accurate `exp`. On every model (`Gpr` / `FittedGpr` / `OnlineGpr`, `Sgpr` / `FittedSgpr` / `OnlineSgpr`, `Svgp` / `FittedSvgp`) the mode is the runtime enum `KernelExp`, set by `with_math(KernelExp::FastApprox)`; each kernel call dispatches once from it to the `KernelMath` marker. `erf` and other functions are added only when a kernel or likelihood needs them. `FittedGpr` and `OnlineGpr` save the mode; a file with no field loads as `Accurate`.

## 9. Optimizer

**Allocation-free, and wrapped in `Result`.** Objective capability is layered, so a solver states what it needs as a bound, not a runtime flag: `Objective` (value, §5.4) ⊂ `Differentiable` ⊂ `TwiceDifferentiable`.

```rust
pub trait Differentiable: Objective {
    fn gradient_into(&mut self, params: &[f64], out: &mut [f64]) -> Result<(), GprError>;
    /// Override this so one evaluation shares its inner work (Cholesky, W, exp_buf).
    fn value_and_gradient_into(&mut self, params: &[f64], out: &mut [f64]) -> Result<f64, GprError> {
        let value = self.value(params)?;
        self.gradient_into(params, out)?;
        Ok(value)
    }
}

pub trait TwiceDifferentiable: Differentiable {
    fn hessian_into(&mut self, params: &[f64], out: &mut [f64]) -> Result<(), GprError>; // row-major p×p
}

pub struct OptResult {
    pub params: Vec<f64>, // same space as init (log-θ for the GP models)
    pub value: f64,
    pub iterations: u64,
}

// Optimizer<P> is in §5.4: `P` is the objective type the algorithm can minimize.
```

`init` is a slice (it does not consume the caller's `Vec`). `GprObjective` overrides `value_and_gradient_into` and shares L, α, W, and `exp_buf` by the §6.2 steps. `GprObjective` implements `TwiceDifferentiable`, and `hessian_into` forwards to `FittedGpr`. `SgprObjective` is the same crate-private adapter for `Sgpr`. Bounds come from each parameter's `Interval` (crate-private `HasBounds`).

The trainer bound is `O: for<'a> Optimizer<GprObjective<'a, P>>`. The default is `Lbfgs`. `Lbfgs` needs `Differentiable`, `TrustRegion` needs `TwiceDifferentiable`, and `NelderMead` / `FastSimulatedAnnealing` need only `Objective`. The argmin adapters map each user-unit interval through a logit so argmin stays unconstrained (log-uniform for positive intervals, scaled so the Jacobian is 1 at the midpoint); `TrustRegion` also maps the analytic Hessian. `TrustRegion` is argmin's trust-region method with the Steihaug subproblem, the solver that uses the Hessian: an indefinite or singular Hessian and a step out of the bounds are handled by the region shrinking (a candidate that cannot be evaluated, Hessian included, costs a barrier value), and a solver failure is `OptimizationNotConverged`. `FastSimulatedAnnealing` is gprx's own value-only solver (Cauchy / Metropolis): it walks the log-`θ` it receives, with no logit, and is the example of a user optimizer. The objective types are crate-private, so a user optimizer implements `Optimizer<P>` generically over the capability it needs (`impl<P: Objective> Optimizer<P> for Mine`) and replaces the same type parameter through `with_optimizer`; there is no second solver setting next to it to ignore (`.cursor/rules/types.mdc`). gprx does not implement its own quasi-Newton. `Adam` is the minibatch loop of `Svgp` and is not an `Optimizer`. There is no runtime NotImplemented.

## 10. `GprError`

Cover the failures that are specific to numerical work.

```rust
#[derive(Clone, Debug, thiserror::Error, PartialEq)]
pub enum GprError {
    DimensionMismatch { x_dim: usize, expected_dim: usize },
    InsufficientData { n: usize, min: usize },
    EmptyInput,
    NonFiniteInput,
    NonFiniteKernelValue,
    CholeskyFailed { jitter: f64, matrix_size: usize, stage: CholeskyStage },
    NonPositiveDefiniteMatrix,
    CoordGradientUnsupported,
    OptimizationNotConverged { iterations: usize },
    InvalidHyperparameter { reason: String },
    ShapeMismatch { reason: String },
    LengthMismatch { reason: String },
    IndexOutOfRange { reason: String },
    InvalidConfig { reason: String },
    SizeOverflow,
    InvalidInterval(IntervalError), // #[from]
    InvalidNoiseVariance { reason: String },
    UnsupportedKernelOperation { reason: String },
    WorkspaceTooSmall,
    InvalidPointId,
    InvalidInducingId,
    PersistFailed { reason: String },
    UnsupportedPersistVersion { found: u32, supported: u32 },
}

pub enum CholeskyStage { Fit, Predict, OnlineInsert, OnlineDelete }
```

Display text is English (see `src/error.rs`).

`InvalidHyperparameter` is only for a hyperparameter value outside its domain. Matrix shape, slice length, and index errors are `ShapeMismatch`, `LengthMismatch`, and `IndexOutOfRange`. Optimizer, jitter-policy, and transform settings are `InvalidConfig`. A size product that overflows `usize` is `SizeOverflow`, not `EmptyInput`. An interval that does not contain its value is `InvalidInterval`. Save / load failures are `PersistFailed`, and a file from another format version is `UnsupportedPersistVersion`.

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
2. **Delete at any index**: `LdltStore` holds an **LDLT factor** and uses `ldlt::update::delete_rows_and_cols_clobber`. On handwritten SPD matrices of size `2×2` / `5×5`, the reconstructed `A = L D Lᵀ` after delete matches a full LDLT. There is no Givens downdate

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

The factor is stored transposed, as `Lᵀ` in a column-major `n_capacity × n_capacity` matrix, so each row of `L` is contiguous: an append writes one contiguous column, and its forward solve `L w = k` reads each row contiguously (rows in panels of 8: the panel's rows are dotted with the solved head of `w` in one pass, then the 8×8 triangle is solved; one flat loop, unlike faer's recursive solve). The solves and persist read the lower view `ld()` (a row-major view of `Lᵀ`); the saved packed LDLT is unchanged. With `f32` storage the LDLT solve accumulates in `f64` and rounds once, because the row order of an `f32` solve over that view cancels worse in `k*ᵀ α`.

Crate-private. `from_active(n)` sets `n_active = n_capacity = n`. Training `X` is held by `OnlineGpr` and is not on this struct. Before an append insert, `OnlineGpr` calls `ensure_capacity`. There is no growth-factor field.

```rust
struct LdltStore<T: KernelScalar = f64> { // T is the precision's Storage
    lt: Mat<T>,                // Lᵀ with D on the diagonal: row i of L is the head of column i
    y: Col<T>,
    alpha: Col<T>,
    v_buf: Vec<T>,             // k of an appended point, then its forward solve
    delete_scratch: MemBuffer, // faer delete_rows_and_cols. Grown with capacity
    n_active: usize,
    n_capacity: usize,
    factor_jitter: f64,        // j of the batch factor; every row factors A + (σn² + j) I
}
```

**Growth** (`ensure_capacity(needed)`, when `n_capacity < needed`):

1. `new_cap = max(needed, max(n_capacity, 1) * 2)`
2. Reallocate `lt`, `alpha`, `y`, and `v_buf` at `new_cap`. Grow the delete scratch for `new_cap` too
3. Copy the existing `n_active × n_active` lower triangle and the length-`n_active` vectors
4. `PointRegistry` indices stay below `n_active`, so they do not need to be rewritten
5. Run the insert after growth. Do not reallocate in the middle of the update

**Variance at predict**: the predictive mean is O(n), but the predictive variance `σ*² = k(x*,x*) - vᵀ D v` (LDLT, the rewritten `L v = k*`) is O(n²) per test point. `v_buf` is allocated up front.

### Steps and the invariant

**Append**: (1) if capacity is short, `LdltStore::ensure_capacity` (factor 2). Training `X` / `y` on `OnlineGpr` grow by the same factor. Query buffers use `ensure_at_least`. (2) Distances from the new point to the existing n points (O(n). One column is sequential, and `k` is written directly into `v_buf`). (3) Add only the kernel diagonal `k_new` (insert does not write a new row or column of `K`. predict and NLML read only LD). (4) Bordered LDLT update (O(n²). The triangular solve reuses `v_buf`). (5) `α` is not solved on insert (libgp `alpha_needs_update`); insert only marks it stale in O(1). The first read resolves the LDLT again: a `&mut self` read (`predict_into`, a hyperparameter write) stores `α` on the model, and a `&self` read (`predict`, covariance, sample, LOO, NLML, `alpha()`, `save_with_factor`) fills a `OnceLock` cache that the next insert / delete clears. A failed solve (a `MixedPrecision` `f64` fallback that does not factor) is that read's `Err`, so `OnlineGpr::alpha()` returns `Result`. (6) `PointRegistry` issues a new `PointId`.

**Delete**: (1) update LD with `ldlt::update::delete_rows_and_cols_clobber` (O(n²). Scratch lives on `LdltStore` and is reused). (2) Remove the matching entries from `y` and `X` on `OnlineGpr` and pack the later rows/columns (O(n)). (3) Shift `PointRegistry` indices in the same order. (4) `α` is not solved on delete either; the stale mark and the first-read solve are the same as append step (5). `n_capacity` stays. The last point is not deleted (`InsufficientData`, `min = 2`). An unknown or already-deleted `PointId` is `InvalidPointId`.

**Invariant**: when a delete shifts internal indices, `LD` / `y` / `alpha` on the workspace, `X` on `OnlineGpr`, and `PointRegistry` **must stay in the same order**. One of them drifting produces the wrong solution. Tests (§12) check this invariant explicitly.

```rust
/// Crate-private. One registry type for data points and inducing points.
struct IdRegistry<I: RegistryId> {
    id_to_index: HashMap<I, usize>,
    index_to_id: Vec<I>,
    next_id: u64,
}
type PointRegistry = IdRegistry<PointId>; // OnlineGpr, OnlineSgpr
type InducingRegistry = IdRegistry<InducingId>; // OnlineSgpr's inducing points
```

### API

**Separate insert/delete from reoptimizing hyperparameters.**

An unfitted `Gpr` does not gain points. A batch `FittedGpr` has no `insert`.

```rust
impl<O, P: GpScalar> FittedGpr<O, P> {
    pub fn into_online(self) -> Result<OnlineGpr<O, P>, GprError>;
}

impl<O, P: GpScalar> OnlineGpr<O, P> {
    pub fn insert(&mut self, x_new: &[f64], y_new: f64) -> Result<PointId, GprError>;
    pub fn delete(&mut self, id: PointId) -> Result<(), GprError>;
    pub fn point_ids(&self) -> &[PointId];
    pub fn alpha(&self) -> Result<&[P::Refine], GprError>; // solves a stale α (above)
    // predict / covariance / sample / loo / NLML / set_params / refit / save as on FittedGpr
}

pub struct OnlineGpr<O = Lbfgs, P: GpScalar = DoublePrecision> {
    core: GprCore<P>,             // §6.3
    optimizer: O,
    workspace: LdltStore<P::Storage>,
    registry: PointRegistry,
    alpha: AlphaState<P>,         // stale flag, stored α, and the OnceLock cache
}
```

`insert` / `delete` update LD at the current kernel and hyperparameters and mark `α` stale. Hyperparameters are reoptimized only when `OnlineGpr::refit` / `set_params` is called explicitly. Those run the batch fit code on temporary LLT buffers (§6.3) and keep `PointId` values and workspace capacity. `into_online` assigns `0 .. n-1` to the existing `n` points. Later `insert` ids increase and are not reused. `PointId` has no public constructor. `PointRegistry` is crate-private and owned by the online model. persist keeps `FORMAT_VERSION` 1 and requires `factor_kind` (`llt` / `ldlt`). `load` of `llt` is `FittedGpr`. `ldlt` is `OnlineGpr`, and `point_ids` plus `next_point_id` are also required. Sparse online is `OnlineSgpr` (§6).

## 12. Test plan

Correctness comes before speed: every path has its correctness tests before it is optimized.

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
5b. **External check of online insert**: libgp `add_pattern` at the same θ, against predict (mean and observation variance) and NLML, relative `1e-8`. There is no external delete API. `cargo test` reads committed JSON (it does not call C++)
5c. **External check of Sparse**: `Sgpr<Fixed>::factor` against collapsed GPyTorch SGPR, and `Svgp<Fixed>::factor` (prior `q`) against whitened SVGP, at the same initial θ, relative `1e-8` (mean, Observation, Latent, NLML / ELBO). `cargo test` reads committed JSON (it does not call Python)
5d. **External check of Sparse online**: `OnlineSgpr` `insert` / `delete` / `insert_inducing` / `delete_inducing` against a collapsed GPyTorch SGPR (full reassemble) at each stage, same initial θ, relative `1e-8` (mean, Observation, Latent, NLML). `cargo test` reads committed JSON (it does not call Python)
6. **Precision**: compare f32/f64/mixed, ill-conditioned matrices, and the f64 fallback when refinement does not converge
7. **Inference**: compare with a known small GPR (mean, latent variance, observation variance, log marginal likelihood, gradient). sklearn JSON is the second numerical check, not the public-API contract. The algorithm source of truth is GPML / Rasmussen
8. **Preprocessing**: `predict` after `StandardizeTarget` matches the unstandardized model on the original scale (the closed form of the affine map)
9. **Inference after optimization**: 1-D Forrester and the 2-D weighted sphere (ARD), sklearn L-BFGS against `Gpr::fit`, at a loose tolerance. Separate from the fixed-hyperparameter JSON (1e-8). `cargo test` does not call Python
10. **Leave-one-out**: the GPML analytic formula at n=2, a real leave-one-out `fit`+`predict` at n=3, and the LOO fields of the optimization JSON (item 9) at sklearn's `θ`. `cargo test` does not call Python
11. **Persist**: a save → load round trip of every model type predicts the same values
12. **Allocations**: `tests/alloc.rs` (§15.1)

The goldens live under `compare/goldens/` and are regenerated by `just gen-goldens`, `gen-online-goldens`, `gen-sparse-goldens`, and `gen-sparse-online-goldens`; `cargo test` only reads them.

## 13. Implementation roadmap

Order and status are [roadmap.md](roadmap.md). Acceptance text stays on each Issue.

## 14. Open items

1. **Checking mixed-precision refinement parameters**: the §4.2 defaults have a theoretical basis, and they have not been checked on a real workload. That includes the accuracy gap between `PromoteStorage` and `ReevaluateKernel`, and `log|K|` plus the trace term when MixedPrecision is used during fit
2. **Choosing the distance-cache policy automatically** (a `DistanceCachePolicy::Auto` from `n`, `d`, and a memory budget, §5.2): it needs a measurement that accounts for kernel kind, SIMD efficiency, and memory bandwidth. Tracked as P5-5 ([#43](https://github.com/YUKIKEDA/gprx/issues/43)); acceptance is set after Grill

## 15. Benchmark strategy

**After correctness, measure the same path while building it.** Optimization reads a measured baseline; it does not start from a guess. Day-to-day rules are `.cursor/rules/bench.mdc`.

### 15.1 Tracks

| Track | Tool | When | What is read |
|---|---|---|---|
| Time | criterion, `benches/exact.rs` | `just bench` (local). Not in default CI (noise) | Wall time. Groups are measured separately |
| Allocations | `tests/alloc.rs` | `just test` (required) | New allocations **after** Workspace setup. The cap is a ratchet (it may fall, and it does not rise without an Issue) |
| Cross-library time and RSS | `compare/perf/` | `just perf`, `perf-online`, `perf-online-stages`, `perf-online-delete`, `perf-sparse`, `perf-sparse-online` (manual, not CI) | gprx against sklearn / libgp / friedrich (Exact), libgp (online insert), GPyTorch / GPy (sparse), GPyTorch (sparse online). No correctness gate |

Do not mix time and allocations into one number. Do not mix a full L-BFGS with "one MLL+gradient". Do not add a sparse or online group to criterion; those are measured by `compare/perf/`.

### 15.2 Fixed problems (the regression unit)

The input has to be the same every time, or a faster run cannot be told from a different dataset.

- Criterion uses `n = 256`. The cross-library harness uses `n = 256 / 1024 / 4096` (Forrester) and `16×16 / 32×32 / 64×64` (sphere)
- Isotropic: 1-D Forrester `f(x)=(6x-2)² sin(12x-4)`, `x ∈ [0, 1]`, RBF + `GaussianLikelihood` + `StandardizeTarget`. Initial hyperparameters `ℓ = 1`, `σn² = 0.1`
- ARD: 2-D weighted sphere `f=(x/0.25)²+(y/1)²`, a 16×16 grid on `[0, 1]²`. Initial `ℓ_d = 4` (`ℓ_d = 1` dies on the first line search)
- `y` is that function plus `N(0, 1)` (`SmallRng`. Forrester seed `0`, ARD sphere seed `9`. Seed `0` walks a ridge on `Uncached`). It is not an independent random series (L-BFGS eval counts move with the landscape)
- Criterion groups in `benches/exact.rs` (only paths that exist):
  1. `kernel_rbf` — lower-triangle build of K
  2. `cholesky_alpha` — LLT of `A` and `α`
  3. `mll_and_grad` — one §6.2 evaluation
  4. `predict_100` / `predict_100_mixed` — 100 test points, `DoublePrecision` and `MixedPrecision`
  5. `fit_lbfgs` — the whole optimization loop. Record the L-BFGS eval count next to the wall time. A difference at a different eval count is not a speed difference
  6. `fit_fsa` — `FastSimulatedAnnealing` on a sum of two leaves (the leaf-rebuild path of §5.4)
  7. `mll_and_grad_ard` / `fit_lbfgs_ard` — ARD RBF on the weighted sphere, `Cached` versus `Uncached` (bench ids `always` / `never`). Do not compare with isotropic. `fit_lbfgs_ard` also records the eval count
  8. `kernel_exp` / `kernel_exp_ard` — `apply` and the θ `grad` after the distance is filled once. `FastApprox` versus `Accurate`. Do not mix with `mll_and_grad`

### 15.3 Baselines

Named criterion baselines and the machine they ran on are recorded in `.dev/bench-log.md` (local, not committed). The current comparison baseline is `phase-2`. A PR that touches a hot path (`src/kernel/`, `workspace`, `gpr`, `objective`, `sgpr`, `svgp`, `precision`) pastes criterion against that baseline in Verification. If the change cannot affect speed, say why.

### 15.4 Metrics

The gate is "not worse than the baseline".

| Metric | Contents |
|---|---|
| One MLL+grad | By n and kernel. Separate from the optimization loop |
| Fit (L-BFGS) | Includes iterations |
| Predict | By test-point count. Latent / observation |
| Peak memory | Including Workspace. Includes `w_matrix` |
| Allocations | Count after setup. Ratchet, and 0 on the Exact predict / fit hot path |
| Parallel | By thread count. Watch double parallelism with faer |
| f32/f64 | Accuracy and speed |
| Online insert/delete | One point versus a full refit |

### 15.5 Not done

- Adding Rayon / SIMD / an approximate exp because it "should be faster", without a measurement
- Making criterion in CI a red/green gate (it flakes across machines)
- Requiring zero allocations before counting them (count first, then lower the cap in steps)
