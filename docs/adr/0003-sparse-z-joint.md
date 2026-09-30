# ADR 0003: Sparse GPR optimizes inducing locations Z jointly

- Status: accepted
- Date: 2026-09-21
- Issue: [#184](https://github.com/YUKIKEDA/gprx/issues/184) (P4-5)

## Context

`SparseGpr::fit` in P4-4 moves only the kernel and likelihood `θ`. The caller passes inducing locations `Z`, and they are not in params. When P4-6 moves `Z`, decide first whether one `Optimizer` moves `θ` and `Z` together, or whether they alternate.

Joint optimization adds `m×d` parameters. L-BFGS history holds `history_size` vectors of that length. Alternating makes the inner count and the order a combination of settings. Type names and the implementation are P4-6. This note fixes only how `Z` moves.

## Decision

- `fit` with free `Z` moves kernel `θ`, likelihood `θ`, and `Z` together. Alternating is not shipped. An implementation of both is not shipped either
- The `fit` that moves only `θ` and keeps `Z` fixed stays. Fixed and free are different types. There is no flag. Identifiers are P4-6
- `factor` moves neither `θ` nor `Z`
- Params for free `Z` are kernel `θ`, likelihood `θ`, then column-major `Z` (`m×d`)
- `Z` is raw coordinates. The interval is the open box of per-dimension min/max of training `X`, widened a little
- The `Z` gradient is `grad_wrt_coord_dim`, and its Hessian is `hess_wrt_coord_dims` / `hess_wrt_coord_mixed` / `hess_theta_coord_dim`. One call per dimension, not per point. Every built-in kernel and Sum / Product tree has them. Matérn `ν = 1/2` returns `GprError::CoordGradientUnsupported`: its coordinate derivative is undefined at coincident points, and `Z ⊂ X` starts there (a type-level exclusion would need a separate kernel type for `FreeInducing`, which the row rejected). A `Custom` leaf provides them through the squared-distance derivatives of `KernelTerm`

## Rationale

Joint keeps `Optimizer` as one slot. Alternating needs "which one first, and how many times", which becomes an ignored field or a runtime error. gprx represents impossible states as types.

GPyTorch SGPR and Titsias (2009) normally move `Z` and the hyperparameters together. `m` is small for Sparse, and the VFE cost is `O(nm²)`.

The L-BFGS history grows by `history_size × m × d` values of `f64`. The free-`Z` length is `p = p_θ + m×d` (`p_θ` is kernel plus likelihood). Example: `m = 64`, `d = 8`, `history_size = 10` adds 5120 values (about 40 KiB), which is small next to the `K_mn` / `B` workspace. The earlier §6.1 claim that "`m×d` breaks L-BFGS" does not hold at this size.

Removing the `θ`-only path would remove P4-4's "keep the caller's `Z` fixed and search `θ`". Splitting the type leaves the current `fit` as the default.

Keeping params as a `θ` prefix and appending `Z` means kernel and likelihood indices do not shift between the fixed type and the free type. Column-major matches training `X` / `Z`.

Putting raw coordinates on an open interval of the training box reuses the existing logit. Finite-differencing `Z` runs VFE `m×d` times and empties the point of joint optimization. The coordinate derivative is already `grad_wrt_coord_dim` in §5.1.

## Rejected

- **Alternating optimization**: each step is lower dimensional. It needs an inner count, an order, and a stopping rule. Two objectives and a schedule become the public surface of the same row
- **`fit` always moves `θ+Z`**: one path. The path that receives a good `Z` and searches only `θ` disappears
- **`optimize_z: bool`**: one field is ignored. That breaks the type rule
- **Finite-difference gradient of `Z`**: no new leaves. The number of evaluations grows by `m×d`

## Consequences

- P4-6 adds the free-`Z` type and ships the joint `fit`. The fixed-`Z` `fit` stays
- No later row adds alternating. Reversing this is a new Grill → Issue
- Type names and numerical experiments are not written in this ADR
