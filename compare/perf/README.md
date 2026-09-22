# Cross-library wall time and peak RSS (P2B-16)

Manual. Not a CI gate. `just test` must not run this.

```text
uv run --directory compare/perf python run.py
```

or `just perf`.

Same JSON cases for gprx, sklearn, libgp, and friedrich:

- isotropic Forrester, `n = 256 / 1024 / 4096`, `ℓ = 1`, `σn² = 0.1`
- ARD weighted sphere, `16×16 / 32×32 / 64×64`, `ℓ_d = 4`
- `y = f(x) + N(0, 1)` (Forrester seed `0`, sphere seed `9`, NumPy Generator)
- factor at the same initial `θ` (no hyperparameter search)
- then joint MLL+grad at that `θ`, reported as `joint_evals` × median of one call
- then predict 100 points, then peak RSS
- times: discard `PERF_WARMUP` (default 1) then the median of `PERF_REPS` (default 51 if `n ≤ 256`, 21 if `n ≤ 1024`, else 7); the table also shows min–max
- RSS is still the peak of that one process

gprx uses `Gpr<Fixed>::factor` and `value_and_gradient_into` with `StandardizeTarget`. sklearn uses `optimizer=None` + `normalize_y=True`, then `log_marginal_likelihood(..., eval_gradient=True)`. libgp is the native C++ library (`compare/perf/libgp/`): `add_patterns` then `log_likelihood_gradient`. friedrich / libgp z-score `y` in the runner. friedrich has no ARD and no public MLL+grad: those cells are N/A. Python bindings are not used.

`just perf` prints two tables: the speed pole (default `CachedDistances` + `RetainCholesky`), then the memory pole (`Gpr::with_prefer_memory` = `UncachedDistances` + `ReuseCholesky`). P2B-16 time / RSS gates use the first table. The second table is recorded only; it does not add an RSS gate. The P2B-21 Uncached+Retain table stays in `.dev/bench-log.md`. `python run.py --reprint` rebuilds the tables from `out/results.json` without rerunning.

Results: `compare/perf/out/results.json`. Pass / fail is recorded in `.dev/bench-log.md`. criterion is not used for these gates.

## Online insert (P3-6)

`just perf-online` times `OnlineGpr::insert` vs libgp `add_pattern` from `n = 2` to `n` on the same Forrester / sphere cases. Raw `y` (no `StandardizeTarget`, no z-score). Gate: `n = 256 / 1024` median ≤ libgp (5% inconclusive). `4096` and RSS are recorded only. Goldens for `just test` are `just gen-online-goldens` → `compare/goldens/online_libgp_*.json`.

`just perf-online-stages` rebuilds gprx-perf with `--features insert-stages` and prints kernel / bordered LDLT / X·y medians for Forrester `n = 256 / 1024`. It does not change the gate clock.

`just perf-online-delete` times gprx `OnlineGpr::delete` from `n` down to 2 (last remaining `PointId` each step; insert is untimed). No libgp cell. Used to record before / after of the P3-7 delete path.

## Sparse SGPR / SVGP (P4-12)

`just perf-sparse` times `Sgpr<Fixed>::factor` and `Svgp<Fixed>::factor` (prior `q`) against GPyTorch / GPy. Same Forrester / sphere `n` as P2B-16, `m = 16` (k-means seed `0` at generation time), population-standardized `y`, CPU, no `fit`. Clock matches P2B-16. Gate: factor / joint / predict and peak RSS each smaller than every opponent that wrote the cell (5% inconclusive). Unwritable cells are N/A. Results: `compare/perf/out/sparse_results.json`. Pass / fail is recorded in `.dev/bench-log.md`. `just perf` / `just perf-online` stay Exact / online. `cargo test` must not run this.
