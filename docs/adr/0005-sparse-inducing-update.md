# ADR 0005: Inducing-point growth is a bordered insert and a trailing delete

- Status: accepted
- Date: 2026-09-21
- Issue: [#192](https://github.com/YUKIKEDA/gprx/issues/192) (P4-9)

## Context

`OnlineSgpr` adds and removes only `X`, by rank-1. The reason is [ADR 0004](0004-sparse-online-rank1.md). As `n` grows, approximation quality wants `m` to grow too. Calling `assemble_vfe` every time redoes the LLT of `K_mm` and every column of `A` at `O(nm²)`. There is no public `InducingId` / `insert_inducing` yet. This note fixes only the factor update.

## Decision

- The target is an append insert of one inducing point and a delete at any index. `X` and `θ` do not move
- insert solves `k(z_new, Z)` against `L_mm` and bordered-LLTs `K_mm`. The new row of `A` is `(k(z_new, X) − lᵀ A) / ℓ`. `B` is also a bordered LLT
- delete drops that row of `L_mm` and cholupdates the trailing triangle. It reuses `K(Z, X) = L A` to remove the row, then solves `A` against the new `L_mm`. `B` is rebuilt from the new `A`
- `w = B⁻¹ Ay` is resolved by LLT after `B`
- `k_diag_sum` is the diagonal of `X`, so it stays. `‖A‖_F²` adds the row norm on insert and is rebuilt on delete
- The check reconstructs `K_mm = LLᵀ` and `B = LLᵀ`. The sign of `L` does not matter
- No public `InducingId` / `insert_inducing` / `delete_inducing` (P4-10)
- Coordinates come from the caller. k-means is not in this row

On RBF / Matern ν=3/2 / RBF ARD (2-D) / RBF+White, each with `n = 4` and `m = 2`, after one append insert and one delete of the first point (not the tail, because `m = 2`), reconstructed `K_mm` / `A` / reconstructed `B` / `w` / `k_diag_sum` / `‖A‖_F²` matched `Sgpr<Fixed>::factor` at the same `θ`, `X`, and `Z` to relative `1e-12`. Both insert and delete passed as incremental updates.

## Amendment: jitter, tolerance, and reassembly (R8-1, [#520](https://github.com/YUKIKEDA/gprx/issues/520))

The public `insert_inducing` / `delete_inducing` of `OnlineSgpr` (P4-10) first reassembled on every change. R8-1 moves them to the update above and fixes what the update keeps and when it does not apply.

- **Jitter.** The factor of `K_mm` may take an offset `j` under the jitter policy (`K_mm + j I = L Lᵀ`). The model keeps that `j` (`k_mm_jitter`) with `L`, and the border is the border of `K_mm + j I`: `ℓ² = k(z_new, z_new) + j − ‖l‖²`. A delete keeps `j`. Since the update runs only at `j = 0` (below), the border adds `j = 0` in practice; it stays in the formula so the factor it borders is the one the model holds
- **Reassembly.** The change assembles the VFE system of the new inducing set again (`O(n·m²)`, the factor a batch model of that set takes) instead of the update when any of these holds:
  - the storage scalar is `f32` (`SinglePrecision`, `MixedPrecision`). Their assembly computes in `f64` and rounds into `f32`; an update in `f32` arithmetic was `3e-4` (mixed) to `1.7e-3` (single) off the `f64` factor, where the assembly keeps the accuracy of a fresh factor of that precision
  - the factor of `K_mm` took `j > 0`. The factor of the changed set chooses its own `j` (zero once the near duplicate that needed it is gone); an update keeps the stale one (`2e-8` off in the NLML after deleting a duplicate)
  - a bordered pivot of an insert is at most `τ = 1e-5` times the diagonal it is cancelled from: `ℓ² ≤ τ (k(z_new, z_new) + j)`, or `β² ≤ τ (σ² + ‖a_new‖²)` for `B`. The cancellation leaves an absolute error of about `ε` times that diagonal, so the relative error of the pivot is about `ε / τ ≈ 2e-11` at the threshold (half that in `ℓ`, which divides the new row of `A`), an order below the tolerance below. Without the threshold an exact duplicate next to a near duplicate left `ℓ²` a rounding residue and the update was `0.5` off
  - the update does not factor: a non-positive or non-finite pivot, a failed downdate of `B` for a repaired supplied column, or a `B` of a delete that does not factor
- **Tolerance.** Random insert / delete sequences (coordinates: RBF, Matérn ν=3/2, RBF ARD 2-D, RBF+White; supplied distances: RBF, Matérn ν=3/2) match `Sgpr<Fixed>::factor` at the same `θ`, `X`, and `Z` to relative `1e-10` (mean, variance, NLML) after every step. The update and the factor round in a different order, so they are not compared at `1e-12`; drawn inducing points keep a gap of `0.25` so the factor itself is accurate to that level (a pivot of `3e-9` left the factor `7e-9` off the exact value). Measured, the largest difference is `4e-15`. A reassembly is the factor's own assembly and is compared at `1e-12`; an `f32` storage at `1e-5`, the tolerance of a model factored again on load

## Rationale

An append insert attaches the new row and column at the edge of `K_mm`. The existing `L_mm` and `A` stay usable. The new row is `O(nm)`. The bordered `B` is `O(m²)`. That is cheaper than a full reassemble at `O(nm²)`.

A delete in the middle removes a row and a column of `K_mm`. The leading block of `L` stays, and a rank-1 update of the trailing part is `O(m²)`. The remaining rows of `A` cannot be reused because `L` changed. Reusing `K(Z, X) = L A` avoids reevaluating the kernel. The triangular solve stays `O(nm²)`. `B` is rebuilt because every row of `A` moved.

An incremental fix of `w` would keep the packing of `y` and a sign on the side. Solving `Ay` with the existing LLT of `B` reuses the same workspace.

## Rejected

- **`assemble_vfe` every time**: correct. Pays `O(nm²)` and a reevaluation of `k(Z, X)` every time `m` grows
- **Delete as only the inverse of the bordered step**: any index other than the tail needs a permutation first. A trailing cholupdate avoids that permutation
- **A public `insert_inducing` on this row**: factor agreement and the public surface would be the same PR. The public API is P4-10
- **k-means on this row**: choosing coordinates is separate from updating the factor

## Consequences

- The public inducing API in P4-10 uses this incremental update to grow and shrink `m`
- If the trailing update or the refactor of `B` fails on a large problem, the same ADR drops only delete back to a reassemble. A new Grill is not required
- If the bordered Schur complement of an insert is non-positive or below the relative threshold (nearby `Z` in 1-D, large ℓ), the same ADR reassembles that one point. A new Grill is not required
- `f32` storage and a jittered `K_mm` always reassemble (see the amendment), so they pay `O(n·m²)` per inducing change as before R8-1
- Adding and removing `X` is outside this ADR ([ADR 0004](0004-sparse-online-rank1.md))
