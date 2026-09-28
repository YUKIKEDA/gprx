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
- Updates that move `Z` or `m` are outside this ADR
