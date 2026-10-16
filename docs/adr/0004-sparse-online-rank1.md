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

- **A pivot of the factor of `K_mm` is at most `τ` times its diagonal**: `L_ii² ≤ τ (K_ii + j)` in `K_mm + j I = L Lᵀ`, with `K_ii + j = Σ_k L_ik²` read from row `i` of `L` in `f64`. `τ = 1e-5` for `f64` storage (the threshold of ADR 0005) and `τ = 1e-3` for `f32` storage (`SinglePrecision`, `MixedPrecision`). R8-4 lowered the threshold of an `X` step to `8 ε` for every storage (the amendment below); the text of this amendment is the reasoning that led to `1e-5`, and the numbers in it were measured against it
- **The downdate of `B` for a delete fails** (the fallback the decision above already named)

Cause. The factor accepts any positive pivot of `K_mm` without jitter, down to the rounding residue an exact duplicate inducing point leaves (`L_ii² / K_ii ≈ 1e-22`). The new column `L⁻¹ k(Z, x)` divides by that pivot, so its row there is a residue divided by another, far above `1`; `B` then carries `ε ‖A‖² / σ²`, and the rank-1 update of `B` and the product `A Aᵀ` the factor forms round that differently. Before this amendment, in the mixed sequences of ADR 0005 (40 seeds), an `X` step was up to `0.18` (2-D ARD RBF) and `7.3e-4` (1-D RBF) off the factor in `f64`. In `f32` the stored `L` is the `f64` factor rounded, the update runs in `f32`, and the steps were `9e13` (1-D) and `1.3e15` (2-D) off a fresh `f32` factor (relative to `max(1, |value|)`: a variance far from the factor's). An exact fix in the update would have to reproduce the factor's own rounding of an ill-posed product, which only the assembly does: it is bitwise the factor.

`f32` threshold. The assembly of an `f32` model computes in `f64` and rounds, so the stored `L` keeps `f32` relative accuracy in every entry, small pivots included, and the ratio read from it is the `f64` ratio to about `1e-7`; the check is meaningful on it. An `X` step updates in `f32`, and the error the pivot adds grows like `ε / r` with the `ε` of `f32`. Measured on 1-D RBF data (two near duplicate pairs among five inducing points, five inserts and deletes), the steps were within `1.3e-6` to `2.7e-6` of a fresh `f32` factor for `r ≥ 7e-5`, `1.6e-5` at `r = 2.3e-5`, `3.8e-5` at `7e-6`, and `0.25` at `7e-15`. `τ = 1e-5` would let `1.6e-5` through the `1e-5` a model factored again on load is held to; `τ = 1e-3` keeps the pivot's part below the `f32` floor. A pivot that small is an inducing point within about `0.03` lengthscales of the inducing set. Inducing changes of an `f32` model always assemble (ADR 0005), so the `f32` value applies only to `X` steps.

Cost. The check is `O(m²)`. An `X` step does not change `L`, so the result is computed when `L` is replaced (factor, inducing change, `θ`, load) and kept with it; an `X` step reads a flag. Checking at every step had cost about a third of an `f64` insert at `m = 64` (`12.7` → `17.0 µs` at `n = 1024`). With the flag, an insert or delete from a factor whose pivots hold takes the time it took before (`f64` and `MixedPrecision`, `n = 1024` and `4096`, `m = 16` to `64`, well-separated inducing points: within the run-to-run noise of `origin/main`). A reassembly costs one assembly: `f64` at `n = 4096`, `m = 64`, `19 ms` per insert against `14 µs`. A `MixedPrecision` step already assembles its `f64` predict weights from all points; the reassembly takes the rounded factor and those weights from the same `f64` assembly, so its step costs about what the rank-1 one does (`18` to `20 ms` against `17` to `18 ms` at `n = 4096`, `m = 64`).

Tolerance. The mixed sequences of ADR 0005 compare every step, `X` steps from a factor with a pivot below `τ` included, at `1e-10` in `f64` (largest `1.6e-12` over 12 seeds, `6e-12` over 40) and, run at `SinglePrecision` and `MixedPrecision` against a factor of the same precision, at `1e-5` (largest `1.0e-6` over 12 seeds, `1.2e-6` over 40). Supplied-distance models are checked from an exact duplicate, near duplicates `1e-8` and `1e-4` apart, and a pair `1e-2` apart (a pivot between the two thresholds), at both storages; updated by rank-1, the `1e-8` case was `9.8e-10` off in `f64` and `0.69` in `f32`.

## Amendment: `f32` storage steps compute in `f64` (R8-3, [#527](https://github.com/YUKIKEDA/gprx/issues/527))

An `X` insert or delete that updates by rank-1 computes in `f64` whatever the storage scalar, and rounds once into the stored `A`, the factor of `B`, and `w`:

- the new column `k(Z, x)` and `k(x, x)` are evaluated in `f64`, and `L⁻¹ k(Z, x)` is solved against the `f64` factor of `K_mm` the last assembly formed. An `f32` model keeps that factor next to its rounded one (in the fitted and the online model, `m × m` in `f64`); an `f64` model's stored factor is that factor
- an online `f32` model keeps the factor of `B` in `f64` next to the stored one, and `A y` in `f64`. The rank-1 update and downdate run on the `f64` factor, the stored factor is its rounding after every step, and `w = B⁻¹ A y` is solved with the `f64` factor from the `f64` `A y`. Whenever the system is assembled, the `f64` factor of `B`, `A y` (as `B w`, from the `f64` `w` of the assembly), `Σ k(x_i, x_i)` and `‖A‖_F²` are the assembly's own `f64` values before rounding (`VfeState::exact`), not the stored values promoted
- `Σ k(x_i, x_i)` and `‖A‖_F²` are running sums in `f64`, rounded where the NLML reads them. The new column enters them, `A y`, and the update of `B` in `f64`, unrounded; the stored `A` holds its rounding. A delete takes the column of the removed point the same way: evaluated again in `f64` against the `f64` factor of `K_mm` (the stored `A` column is a rounding)

Cause. Before this amendment the step ran in the storage scalar, while the assembly of an `f32` model computes in `f64` and rounds, and its predictions solve against a factor of `K_mm` formed again in `f64`. Three parts of the step differed from that, with `ε ≈ 6e-8` the unit roundoff of `f32`:

- the new column was evaluated in `f32` and solved in `f32` against the rounded factor of `K_mm`. Its relative error is about `κ(L_mm) ε`, and `κ(L_mm) = κ(K_mm)^½` reaches `1e3` and more for inducing points `0.25` apart at `ℓ = 2` (pivots near the `1e-3` threshold), so the column is `1e-4` off where a fresh factor's is `ε`. In the mixed sequences below at `σn² = 1e-3`, `ℓ = 2`, solving the column against the rounded factor promoted to `f64` brought the largest step from `3.6e-5` to `6e-6`; in the sequences of `tests/sgpr_ops.rs` at the same `σn²` and `ℓ`, with `B` already updated in `f64`, solving it against the `f64` factor of the assembly instead of the promoted one brought the largest step from `1.3e-5` to `1.2e-6`
- the factor of `B` was updated in `f32`. Each sweep leaves a backward error of about `m ε ‖B‖`, the next step starts from it, and `w` carries it amplified by up to `κ(B) ≤ 1 + ‖A‖² / σn²`. The steps grew with the run: `1e-6` after one step, `2.7e-5` after `50` (`σn² = 1e-3`), `5e-5` to `6e-5` after `1000` (single, 1-D and 2-D). Promoting the stored factor, updating in `f64`, and rounding each step still adds `ε` per step and drifted to `2e-5` to `9e-5` after `1000`; keeping the `f64` factor does not drift
- the running sums were added in `f32`: each step rounds `‖A‖_F²` by about `ε n`, the NLML divides `Σ k − ‖A‖_F²` by `2 σn²`, and the error grows with the run

What was left in the first version of this amendment, and its cause. That version started the `f64` factor of `B`, `A y`, and the sums from the stored (rounded) `A` and factor, and took the rounded column of `A` into them. The stored `A` is the `f32` rounding of the assembly's, so the `f64` system the steps updated was the system of the rounded `A`, not of the points: a perturbation of relative size `ε` in `A` that each step then carried (the rounded column entered `B`, `A y`, and the sums, and a delete removed the rounded column it had stored), amplified in `w` by up to `κ(B) ≤ 1 + ‖A‖² / σn²`. It was not a one-time perturbation: a step that inserts the column rounded and a later step that deletes a different rounding of it leave a residue, and the residue grows with `κ(B)`. A stress run (seeds `0` to `39`, `60` steps of `X` and inducing inserts and deletes, `σn²` `1e-4`, `ℓ = 0.5`) was `1.7e-5` (single, 1-D, at a gap, seed 23, step 49, a delete) and `3.8e-5` (single, 2-D, inducing points drawn anywhere, seed 29, step 58, a delete) from a fresh `f32` factor, where the fresh factor was `2.6e-6` and `1.1e-7` from the `f64` factor of the same points: 8 to 38 times farther from the `f64` system than the rounding of that system is. The steps now start from the assembly's `f64` values and take the unrounded column, so the `f64` system is the `f64` system of the points to `f64` rounding, and the stored values are its rounding: in every setting swept below the online model's values equal the fresh factor's, or differ by one rounding of an `f32` entry (the `f64` rounding residue `1e-13` of the two routes flipping the rounding of an entry). The `3.2e-4` of the review of #528 (single, 2-D ARD, inducing points at a gap, an insert) was not reproduced with the generator of the tests, whose random stream is not the review's; sequences of its shape reproduced the mechanism above (`8.0e-6`, 8 times the fresh factor's distance, at `σn² = 1e-3`, `ℓ = 1`).

Measured (relative to `max(1, |value|)`, the largest over the mean and variance at four queries and the NLML after every step; mixed sequences of `X` and inducing steps as in ADR 0005, inducing points at least `0.25` apart, `30` training and up to `12` inducing points, `60` steps, `12` seeds; and runs of `1500` `X` steps from `5` inducing points, four at least `0.25` apart and one `0.2` from the first, `3` seeds). Before → after:

| Sequence | Single, RBF | Mixed, RBF | Single, ARD | Mixed, ARD |
| --- | --- | --- | --- | --- |
| mixed, `σn² = 0.1`, `ℓ = 1` | `1.7e-6` → `8.4e-7` | `1.7e-6` → `8.4e-7` | `1.5e-6` → `1.1e-6` | `1.1e-6` → `1.1e-6` |
| mixed, `σn² = 1e-2`, `ℓ = 2` | `1.6e-5` → `1.4e-6` | `3.0e-6` → `7.3e-7` | `5.4e-6` → `1.6e-6` | `2.1e-6` → `1.6e-6` |
| mixed, `σn² = 1e-3`, `ℓ = 2` | `3.6e-5` → `1.5e-6` | `9.9e-6` → `9.7e-7` | `3.2e-5` → `1.9e-6` | `2.7e-6` → `1.5e-6` |
| mixed, `σn² = 1e-4`, `ℓ = 2` | `4.6e-5` → `2.4e-6` | `1.3e-5` → `8.1e-7` | `5.4e-5` → `1.4e-6` | `5.3e-6` → `1.4e-6` |
| `1500` `X` steps, `σn² = 0.1`, `ℓ = 1` | `1.7e-5` → `1.7e-6` | | `2.9e-5` → `1.1e-6` | |
| `1500` `X` steps, `σn² = 1e-3`, `ℓ = 1` | `5.8e-5` → `2.2e-6` | | `5.2e-5` → `1.3e-6` | |

(ARD: lengthscales `ℓ` and `1.5 ℓ`.) A fresh `f32` factor is itself up to `3.8e-7` (mean) and `6e-6` (NLML) from an `f64` factor of the same points. `MixedPrecision` predicts its mean from `f64` weights it assembles again at every step, so its mean matches exactly; its variance and NLML read the stored factors. `f64` steps are unchanged (the same operations on the same factors). The `4.0e-4` (single, 2-D ARD) the review of #526 reported came from a stress test that is not in the repository; sequences of the same shape reproduced the mechanisms above but not that value.

Guarantee. An `f32` `X` step of an online model is held to two bounds, against a factor of the same points at the same precision (a fresh factor) and against the `f64` factor of the same points (the truth), for the mean and the variance at four queries and the NLML, relative to `max(1, |value|)`:

- the online values are within `1e-5` of the fresh factor's, the tolerance a model factored again on load is held to (`F32_FACTOR_TOL`), and
- the online values' distance from the truth is at most `2` times the fresh factor's distance from the truth plus `8 ε` of `f32` (`9.5e-7`): a step is never much farther from the `f64` system than the rounding of that system is. Where a fresh factor is itself far from the truth (an ill-conditioned `K_mm`: `8.4e-4` for inducing points `1e-5` to `1e-2` apart at `ℓ = 4`), the online model is as far, and not farther.

The second bound is the one that says what the model guarantees; the first is kept because it holds in every setting swept. The bound is for the settings the tests run: `ℓ` from `0.5` to `4` (isotropic, and ARD with `ℓ` and `1.5 ℓ`), `σn²` from `1e-4` to `1e-1`, `n` between `20` and `40` and `m` up to `12`, inducing points at least `0.25` apart, anywhere in `[0, 4]ᵈ`, and on or next to one another (`1e-5` to `1e-2` per coordinate), both for the `X` steps that update by rank-1 and for those that assemble again from a factor with a pivot below the threshold; long runs (`1500` steps at `σn² = 1e-3`, `ℓ = 1`); supplied-distance models; a loaded online model; an online model made from a loaded fitted model; and a clone that takes different steps than the original. It is not shown for a `κ(K_mm)` beyond that (the thresholds `τ` above send those to the assembly, which is a fresh factor), nor for `n` much larger than `40` (the sums grow, the bound is relative to them). `f64` storage is unchanged and keeps its `1e-10`.

Measured over the sweep (single, 40 seeds of `60` steps; mixed the same values for the `X` steps, since its stored values are those of single; largest over the steps, `ℓ = 0.5` isotropic and ARD, `σn² = 1e-4`, the setting that separated the two most): distance of the online model from the truth / of a fresh factor from the truth / of the online model from a fresh factor, before → after:

| Setting | online vs truth | fresh vs truth | online vs fresh |
| --- | --- | --- | --- |
| single, RBF, at a gap | `1.7e-5` → `2.6e-6` | `2.6e-6` | `1.7e-5` → `0` |
| single, ARD, at a gap | `1.7e-5` → `1.5e-6` | `1.5e-6` | `1.7e-5` → `0` |
| single, RBF, anywhere | `6.9e-6` | `6.9e-6` | `1.8e-6` → `0` |
| single, ARD, anywhere | `3.9e-5` → `1.8e-6` | `1.8e-6` | `3.9e-5` → `0` |
| single, RBF/ARD, duplicates | `2.9e-6`, `1.1e-6` | same | `0` |

(the largest ratio of the online distance to the fresh factor's over every setting of the sweep, `ℓ` `0.5` to `4`, `σn²` `1e-4` to `1e-1`: `38` before, `1.0` after; the largest online-to-fresh difference over the same sweep: `3.9e-5` before, `0` after.) `MixedPrecision`'s mean and variance are read from the same stored factors and its NLML from them, so its values are those of single; its mean comes from `f64` weights it assembles again at every step.

Tolerance. `tests/sgpr_ops.rs` holds an `f32` step to the two bounds above: the mixed sequences at `σn² = 1e-3`, `ℓ = 2` with inducing points at a gap; `1500` `X` steps in a row at `σn² = 1e-3`, `ℓ = 1` (1-D and 2-D ARD, single and mixed; the ADR's worst case); mixed sequences at `σn² = 1e-4`, `ℓ = 0.5` with inducing points at a gap, anywhere, and on or next to one another (1-D RBF and 2-D ARD, single over `16` seeds that include the ones that failed before this change and mixed over `8`); `X` steps of supplied-distance models; online models saved, loaded, and continued; fitted models loaded and made online; and a clone that diverges from the original. The stress sequences fail on the first version of this amendment.

Cost. The step does the same `O(m² + n·d)` work in `f64`. Insert and delete of `X` (2-D ARD RBF, inducing points on a grid, the smallest of several runs of `10000` steps, against the R8-2 head; the first version of this amendment), `SinglePrecision`: `n = 1024`, `m = 64`, insert `6.3` → `6.6 µs`, delete `8.1` → `9.2 µs`; `n = 4096`, `m = 64`, `9.9` → `11.0 µs` and `23.4` → `25.0 µs`; `m = 16` within the run-to-run noise. The added `O(m²)` is the rounding of the factor of `B` after each step and, for a delete, the `f64` copy the downdate runs on. `f64` and `MixedPrecision` (`300 µs` to `3 ms`, the assembly of its `f64` weights) are within the run-to-run noise. A fitted `f32` model holds two more `m × m` `f64` matrices (the factor of `K_mm`, and the factor of `B` the assembly formed, which `into_online` starts from), an online one two (the factor of `K_mm` and the `f64` factor of `B`, which the steps update). Making an online model from a fitted one moves them, and `set_vfe` (an assembly) moves the assembly's `f64` factor of `B` into the online state, so it no longer copies an `m × m` `f64` matrix (the first version of this amendment promoted the stored factor into a new one: one allocation per assembly). Allocations of one call on an online `f32` model after a warmup call (`n = 200`, `m = 16`, 2-D ARD RBF, `RAYON_NUM_THREADS=1`; `SinglePrecision` / `MixedPrecision`): `X` insert `16` / `51` (the first version and now), `X` delete `14` / `49` → `21` / `56` (the delete evaluates the removed point's column in `f64` again: seven allocations), inducing insert `30` / `50` → `29` / `50`. A `MixedPrecision` step runs in `atomically`, which clones the state to undo a failed weights assembly: that includes both `m × m` `f64` matrices, so a Mixed `X` step allocates the `35` more than a single one, as before. The ratchet of `tests/alloc.rs` counts `f64` models only and is unchanged. Time, `SinglePrecision`, 2-D ARD RBF, inducing points apart (so every step is a rank-1 update), the smallest of five runs of `2000` steps, against the first version of this amendment: `n = 1024`, `m = 16`, insert `2.5` → `2.3 µs`, delete `30.7` → `29.8 µs`; `m = 64`, insert `8.6` → `8.4 µs`, delete `42.4` → `40.8 µs`; `n = 4096`, `m = 64`, insert `9.8` → `9.4 µs`, delete `63.8` → `59.8 µs`: within the run-to-run noise.

## Amendment: the threshold of an `X` step (R8-4, [#530](https://github.com/YUKIKEDA/gprx/issues/530))

An `X` insert or delete assembles the system of the changed points again when a pivot of the factor of `K_mm` is at most `8 ε` times its diagonal (`L_ii² ≤ 8 ε (K_ii + j)`, `ε = 2.2e-16` the machine epsilon of `f64`, so `1.8e-15`), for every storage scalar. The R8-2 amendment above set `τ = 1e-5` (`f64`) and `1e-3` (`f32`); those are no longer the thresholds of an `X` step. `1e-5` stays the threshold of an inducing change from that factor ([ADR 0005](0005-sparse-inducing-update.md)), and the code keeps the two apart by type: `PivotGate::XStep` and `PivotGate::InducingStep` (`src/sgpr/factor/updates.rs`) compare the smallest pivot ratio of the factor, which the model keeps with the factor instead of a flag, with the threshold of the step.

Cause. R8-2 compared an `X` step with a fresh factor of the same points at `1e-10` and chose `τ` so that every step passed. A smooth kernel on a few inducing points has a `K_mm` far more ill-conditioned than `1e-5` allows without any duplicate in it, and for such a `K_mm` the fresh factor is itself far from the exact system, so "equal to the fresh factor" forced the assembly in a wide range. The benchmark of `compare/perf` (the sparse-online cases, `X` steps only: 32 inserts and deletes from `n - 16` points, the first 8 of the 16 k-means centres as inducing points; Forrester `ℓ = 1`, weighted sphere ARD `ℓ = 4`) has smallest pivot ratios of `1.2e-13` (`1.0e-13` at `n = 4096`) and `1.8e-7` (`1.7e-7`). With more inducing points the Forrester `K_mm` has no factor without jitter (12 of its points are not positive definite in `f64`), takes the retry jitter `j = 1e-8`, and then has `r ≈ 1e-8`. All of these are below `1e-5`, so every `X` step assembled again, at the cost of one assembly where 0.1.0 did none.

| ms per 32 `X` steps | `n = 256` | `1024` | `4096` |
| --- | --- | --- | --- |
| Forrester, R8-3 head | `5.6` | `11.1` | `20.7` |
| Forrester, this amendment | `0.10` | `0.28` | `1.1` |
| Forrester, 0.1.0 | `0.11` | `0.27` | `1.1` |
| sphere, R8-3 head | `5.6` | `10.5` | `19.6` |
| sphere, this amendment | `0.13` | `0.37` | `1.9` |
| sphere, 0.1.0 | `0.18` | `0.36` | `1.6` |

(The smallest of two or three runs of `compare/perf/gprx sparse-online incremental`, whose clock is the median of 51 / 21 / 7 repetitions; the sub-millisecond cells move by a third between runs.) The harness's own 32 ops, mixed with inducing changes (Forrester `n = 256` / `1024` / `4096`): `6.2` / `5.3` / `7.6` ms before, `3.3` / `3.6` / `4.6` after, `3.8` / `4.5` / `6.2` for 0.1.0; sphere `5.1` / `5.4` / `7.9`, `2.7` / `3.7` / `4.2`, `3.7` / `3.9` / `5.7`. Inducing changes alone (32 inserts and deletes of inducing points at `n` points) take the same time before and after (Forrester `6.2` / `10.6` / `23.1` and `6.3` / `10.5` / `20.3`; sphere `6.0` / `9.9` / `22.3` and `6.2` / `10.5` / `21.4`): that path is unchanged.

Yardstick. The comparison is with the exact system, as in the R8-3 amendment: an `X` step is held to being within `2` times the distance a fresh factor of the same points has from the exact system, plus a floor. The exact system is computed in extended precision (`tests/common/dd.rs`, double-double; the sweeps below used 600-bit balls), and "distance" is the largest error, relative to `max(1, |value|)`, of the mean and the variance at four queries and of the NLML. The floor is `1e-10` for `f64` (the tolerance of R8-2) and `8` `ε` of `f32` for `f32`, which is also held to `1e-5` of a fresh factor (R8-3). A step that is as far from the exact system as the fresh factor is, but not close to the fresh factor, is as good a model as the fresh factor.

Measured with the gate off for `X` steps (inducing changes unchanged), `f64`: 4,000 mixed sequences of 60 steps as in ADR 0005 (copies and near copies of inducing points; `n` 20 to 40, `m` up to 12; 1-D RBF and 2-D ARD RBF with `ℓ` and `1.5 ℓ`; `(ℓ, σn²)` of `(0.5, 0.1)`, `(2, 0.1)`, `(1, 1e-3)`, `(2, 1e-3)`, `(0.5, 1e-4)`; 400 seeds each), 123,000 `X` steps of which 72,000 are from a factor without jitter. By the smallest pivot ratio `r` of the factor the step started from (largest over the steps of the row; the last column counts the steps farther from the exact system than `2 ×` the fresh factor plus `1e-10`):

| `r` | steps | online from exact | fresh from exact | online from fresh | beyond the bound |
| --- | --- | --- | --- | --- | --- |
| `1e-2` to `1` | `34875` | `6.3e-14` | `4.4e-14` | `6.3e-14` | `0` |
| `1e-4` to `1e-2` | `3798` | `5.9e-12` | `4.1e-12` | `3.1e-12` | `0` |
| `1e-6` to `1e-4` | `6793` | `1.8e-8` | `1.8e-8` | `5.0e-10` | `0` |
| `1e-8` to `1e-6` | `7580` | `7.0e-6` | `7.0e-6` | `2.4e-11` | `0` |
| `1e-10` to `1e-8` | `5352` | `6.1e-3` | `6.1e-3` | `2.9e-9` | `0` |
| `1e-12` to `1e-10` | `1994` | `9.2e-2` | `9.2e-2` | `2.1e-8` | `0` |
| `1e-14` to `1e-12` | `1263` | `0.45` | `0.45` | `2.7e-8` | `0` |
| `1e-16` to `1e-14` | `2640` | `2.5` | `2.5` | `1.1e-7` | `0` |
| `1e-20` to `1e-16` | `4919` | `2.1` | `2.1` | `6.8e-7` | `0` |
| `1e-24` to `1e-20` | `1568` | `1.2` | `1.2` | `6.3e-6` | `9` |
| `1e-30` to `1e-24` | `766` | `0.72` | `0.72` | `6.1e-2` | `8` |
| below `1e-30` | `853` | | | `1.4` | `11` |

Above `8 ε` the largest distance from the fresh factor is `2.7e-8`. The fresh factor's distance from the exact system grows by a factor of `20` to `30` per decade of `r` from `1e-4` to `1e-9` (`1.8e-8`, `7e-6`, `6e-3`) and is `0.1` or more below. The distance of the step from the fresh factor is `ε / r` for `r ≥ 1e-5` (`3e-12` for `r` in `1e-4` to `1e-2`, `5e-10` for `1e-6` to `1e-4`: what the rank-1 update of `B` adds to the rounding the fresh factor's `A` already carries) and does not grow that way below `1e-6`. The step and the fresh factor share `L` and solve the new column against it as the assembly solves every column, so the rounding the pivot puts into the column is in both; what separates them is the rounding of the rank-1 update of `B` against that of `A Aᵀ`. Below `r ≈ 1e-22` the bound fails: the column divides a numerator rounded to `ε` by `√r`, so its entry carries `ε / √r` (`2e-4` at `1e-24`, above `1` at `1e-32`).

The `f32` storages, 150 seeds per setting and kernel (`(ℓ, σn²)` of `(0.5, 0.1)`, `(2, 1e-3)`, `(1, 1e-4)`), 16,850 `X` steps each from a factor without jitter, gate off. By `r`: the largest distance from a fresh factor (single / mixed), the steps beyond `1e-5` from the fresh factor (single / mixed), and the steps beyond `2 ×` the fresh distance from the `f64` system plus `8 ε` of `f32` (`9.5e-7`; single / mixed):

| `r` | steps | online from fresh | beyond `1e-5` | beyond the `f64` bound |
| --- | --- | --- | --- | --- |
| `1e-4` to `1` | `8826` | `0` | `0` / `0` | `0` / `0` |
| `1e-8` to `1e-4` | `3315` | `0` | `0` / `0` | `0` / `0` |
| `1e-12` to `1e-8` | `1922` | `2.1e-6` | `0` / `0` | `0` / `0` |
| `1e-15` to `1e-12` | `421` | `9.6e-7` | `0` / `0` | `0` / `0` |
| `1e-16` to `1e-15` | `475` | `6.6e-7` | `0` / `0` | `0` / `0` |
| `1e-20` to `1e-16` | `1132` | `1.0e-6` | `0` / `0` | `0` / `0` |
| `1e-24` to `1e-20` | `321` | `2.1e-4` / `1.9e-5` | `5` / `2` | `3` / `0` |
| `1e-30` to `1e-24` | `228` | `4.2e-2` / `8.4e-3` | `36` / `32` | `4` / `0` |
| below `1e-30` | `210` | `1.4` | `88` / `76` | `0` / `0` |

Down to `r = 1e-20` the step equals the fresh factor's values or differs by the rounding of one `f32` entry (the `f64` systems of the two routes differ by `1e-13`); the first step beyond a bound was at `r = 3.6e-24`.

Why `8 ε`. The bound holds down to `r = 1e-20` (`f64`) and `r = 3.6e-24` (`f32`) and fails below `r = 9e-23`, so the bound alone would put the threshold anywhere above that; it does not fix `8 ε`. What R8-2 fixed in earnest was the factor of an exact duplicate, whose positive pivot is the rounding residue of the original's pivot: `r ≤ 1.7 ε` in 480 duplicates (largest `3.8e-16`; in the other two thirds of the copies the factor took the retry jitter, `r ≈ 1e-8`, and the step updates). The step from there is only as close to the fresh factor as the fresh factor is meaningful (seed 5 of the R8-2 sequences, `r = 2e-35`: the step was `2.9e-4` from the fresh factor and both were `0.16` from the exact system), and the assembly of the exact duplicate is the value a user can compare with a batch model. `8 ε` is a margin of four above the largest residue seen, rounded to a multiple of `ε`, and `65` times below the benchmark's `1.2e-13`. Lowering it further would trade that agreement for nothing the benchmark needs; raising it to the old `1e-5` is the regression.

Not changed: the inducing threshold. An inducing change rebuilds `L` and every row of `A` from the factor it starts from. With the gate off for inducing changes only, the 24 sequences of the R8-2 tests (12 seeds, 1-D RBF and 2-D ARD RBF) had 6 of 731 inducing steps farther from the exact system than `2 ×` a fresh factor plus `1e-10`, up to `0.28` from the factor (a delete: the fresh factor `0.02` from the exact system, the update `0.26`); the same sequences with the gate on had none. The benchmark's inducing cells assemble at that threshold or at the jitter rule, as before, and take the time they took (above).

Tolerance. `tests/sgpr_ops.rs`: `x_steps_on_a_smooth_ill_conditioned_k_mm_stay_near_the_exact_system` runs the 32 `X` steps of the benchmark (Forrester `n = 256` and `1024`, sphere `16 × 16` and `32 × 32`) and holds each to the bound above and to `1e-10` of the fresh factor (largest: online `6.8e-12` from the exact system, the fresh factor `6.8e-12`, online from fresh `1.6e-13`); `f32_x_steps_on_a_smooth_ill_conditioned_k_mm_match_factor` holds the `f32` storages to the two bounds of the R8-3 amendment; `x_steps_next_to_a_near_duplicate_stay_near_the_exact_system` runs `X` steps with a point `10^-1` to `10^-9` from an inducing point (smallest ratio from `5e-3` down to the rounding residue) at `σn²` `0.1` and `1e-3`. `src/sgpr/online/tests.rs`: `x_steps_stay_rank_1_for_a_smooth_ill_conditioned_k_mm` checks that every `X` step of a benchmark-shaped model (`f64`, `f32` single and mixed; smallest ratio `1e-13` and `1.8e-7`) reads the rank-1 gate; it fails with the R8-2 thresholds at the first insert. The mixed sequences of ADR 0005 keep their `1e-10` against the fresh factor (largest `1.0e-11` over the 12 seeds of the tests, `9.6e-11` over 40).

Cost. The smallest pivot ratio is read from `L` when `L` is replaced (factor, inducing change, `θ`, load), `O(m²)`, as the flag was; an `X` step compares one number.

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
- An `X` insert or delete from a factor of `K_mm` with a pivot at most `8 ε` times its diagonal (the rounding residue of an exact duplicate inducing point) pays `O(n·m²)` (the R8-4 amendment) until an inducing change or new `θ` leaves a factor whose pivots hold; a smaller-looking `K_mm` of a smooth kernel does not
- An `X` step of an `f32` model computes in `f64` against `f64` factors of `K_mm` and `B` kept beside the stored ones (the R8-3 amendment), so a run of steps stays as close to a fresh factor as one step
- Updates that move `Z` or `m` are outside this ADR
