# ADR 0004: The online Sparse VFE factor is rank-1

- Status: accepted
- Date: 2026-09-21
- Issue: [#188](https://github.com/YUKIKEDA/gprx/issues/188) (P4-7)

## Context

VFE costs `O(nm²)`. When `Z` and `m` stay fixed and only training points `X` are added or removed, calling `assemble_vfe` every time redoes the LLT of `K_mm` and every column of `A`. `B = σn²I + AAᵀ` is `m×m`, so adding or removing one column should be a rank-1 cholupdate / choldown. There is no public online type yet. This note fixes only the factor update.

## Decision

- The target is one-point insert / delete of `X` only. `Z` and `m` do not move
- `K_mm` and `L_mm` stay
- insert solves `k(Z, x_new)` against `L_mm`, appends that column to `A`, and cholupdates `B`
- delete removes that column and choldowns `B`
- `w = B⁻¹ Ay` is resolved by LLT after `B`. Sherman–Morrison is not used
- `k_diag_sum` and `‖A‖_F²` grow and shrink by the diagonal and the column norm
- The check reconstructs `B = LLᵀ`. The sign of `L` does not matter
- No public online type and no insert API (P4-8)
- Changing `m` is not in this row (P4-9)

On RBF / Matern ν=3/2 / RBF ARD (2-D) / RBF+White, each with `n = 4` and `m = 2`, after one insert and one delete of the middle point, `A` / reconstructed `B` / `w` / `k_diag_sum` / `‖A‖_F²` matched `Sgpr<Fixed>::factor` at the same `θ` and `Z` to relative `1e-12`. Both insert and delete passed as rank-1.

## Amendment: reassembly from a factor with a small pivot (R8-2, [#525](https://github.com/YUKIKEDA/gprx/issues/525))

The public `insert` / `delete` of `OnlineSgpr` (coordinates and supplied distances share them) take the rank-1 update above unless one of these holds; then they assemble the VFE system of the changed points again (`O(n·m²)`, the factor a batch model of those points takes, the same assembly an inducing change of [ADR 0005](0005-sparse-inducing-update.md) falls back to):

- **A pivot of the factor of `K_mm` is at most `τ` times its diagonal**: `L_ii² ≤ τ (K_ii + j)` in `K_mm + j I = L Lᵀ`, with `K_ii + j = Σ_k L_ik²` read from row `i` of `L` in `f64`. `τ = 1e-5` for `f64` storage (the threshold of ADR 0005) and `τ = 1e-3` for `f32` storage (`SinglePrecision`, `MixedPrecision`)
- **The downdate of `B` for a delete fails** (the fallback the decision above already named)

Cause. The factor accepts any positive pivot of `K_mm` without jitter, down to the rounding residue an exact duplicate inducing point leaves (`L_ii² / K_ii ≈ 1e-22`). The new column `L⁻¹ k(Z, x)` divides by that pivot, so its row there is a residue divided by another, far above `1`; `B` then carries `ε ‖A‖² / σ²`, and the rank-1 update of `B` and the product `A Aᵀ` the factor forms round that differently. Before this amendment, in the mixed sequences of ADR 0005 (40 seeds), an `X` step was up to `0.18` (2-D ARD RBF) and `7.3e-4` (1-D RBF) off the factor in `f64`. In `f32` the stored `L` is the `f64` factor rounded, the update runs in `f32`, and the steps were `9e13` (1-D) and `1.3e15` (2-D) off a fresh `f32` factor (relative to `max(1, |value|)`: a variance far from the factor's). An exact fix in the update would have to reproduce the factor's own rounding of an ill-posed product, which only the assembly does: it is bitwise the factor.

`f32` threshold. The assembly of an `f32` model computes in `f64` and rounds, so the stored `L` keeps `f32` relative accuracy in every entry, small pivots included, and the ratio read from it is the `f64` ratio to about `1e-7`; the check is meaningful on it. An `X` step updates in `f32`, and the error the pivot adds grows like `ε / r` with the `ε` of `f32`. Measured on 1-D RBF data (two near duplicate pairs among five inducing points, five inserts and deletes), the steps were within `1.3e-6` to `2.7e-6` of a fresh `f32` factor for `r ≥ 7e-5`, `1.6e-5` at `r = 2.3e-5`, `3.8e-5` at `7e-6`, and `0.25` at `7e-15`. `τ = 1e-5` would let `1.6e-5` through the `1e-5` a model factored again on load is held to; `τ = 1e-3` keeps the pivot's part below the `f32` floor. A pivot that small is an inducing point within about `0.03` lengthscales of the inducing set. Inducing changes of an `f32` model always assemble (ADR 0005), so the `f32` value applies only to `X` steps.

Cost. The check is `O(m²)`. An `X` step does not change `L`, so the result is computed when `L` is replaced (factor, inducing change, `θ`, load) and kept with it; an `X` step reads a flag. Checking at every step had cost about a third of an `f64` insert at `m = 64` (`12.7` → `17.0 µs` at `n = 1024`). With the flag, an insert or delete from a factor whose pivots hold takes the time it took before (`f64` and `MixedPrecision`, `n = 1024` and `4096`, `m = 16` to `64`, well-separated inducing points: within the run-to-run noise of `origin/main`). A reassembly costs one assembly: `f64` at `n = 4096`, `m = 64`, `19 ms` per insert against `14 µs`. A `MixedPrecision` step already assembles its `f64` predict weights from all points; the reassembly takes the rounded factor and those weights from the same `f64` assembly, so its step costs about what the rank-1 one does (`18` to `20 ms` against `17` to `18 ms` at `n = 4096`, `m = 64`).

Tolerance. The mixed sequences of ADR 0005 compare every step, `X` steps from a factor with a pivot below `τ` included, at `1e-10` in `f64` (largest `1.6e-12` over 12 seeds, `6e-12` over 40) and, run at `SinglePrecision` and `MixedPrecision` against a factor of the same precision, at `1e-5` (largest `1.0e-6` over 12 seeds, `1.2e-6` over 40). Supplied-distance models are checked from an exact duplicate, near duplicates `1e-8` and `1e-4` apart, and a pair `1e-2` apart (a pivot between the two thresholds), at both storages; updated by rank-1, the `1e-8` case was `9.8e-10` off in `f64` and `0.69` in `f32`.

## Rationale

If `Z` is still, `K_mm` is invariant. For a new column `a`, `B ← B + aaᵀ`. For a removal, `B ← B − aaᵀ`. `m` is small for Sparse. cholupdate is `O(m²)`. The column kernel is `O(md)`. That is cheaper than a full refactor at `O(nm²)`.

An incremental fix of `w` would keep the packing of `y` and a sign on the side. Solving `Ay` with the existing LLT of `B` reuses the same workspace.

A downdate can fail when `r² ≤ 0`. It did not fail on this small problem. A path that fails falls back to refactoring the delete (option C). The current tests do not require that branch.

## Rejected

- **`assemble_vfe` every time**: correct. Pays `O(nm²)` every time `n` grows
- **Sherman–Morrison for `w`**: counts the `B` update twice. More implementation and more checks
- **A public `insert` on this row**: factor agreement and the public surface would be the same PR. The public API is P4-8
- **Growing `m` on this row**: `K_mm` changes. That is not an X-only update

## Consequences

- The public online path in P4-8 uses this rank-1 update for adding and removing `X`
- If a downdate fails on a large problem, the same ADR drops only delete back to a refactor. A new Grill is not required
- An `X` insert or delete from a factor of `K_mm` with a pivot at most `τ` times its diagonal pays `O(n·m²)` (the amendment) until an inducing change or new `θ` leaves a factor whose pivots hold
- Updates that move `Z` or `m` are outside this ADR
