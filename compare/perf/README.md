# Cross-library wall time and peak RSS (P2B-16)

Manual. Not a CI gate. `just test` must not run this.

```text
uv run --directory compare --group perf python -X utf8 -m perf.run
```

or `just perf`. Every harness runs from `compare/` (one Python environment, `compare/pyproject.toml`; the perf-only packages are the `perf` dependency group) and shares `compare/common/`: the problems, the runner process, the verdicts, and the tables. `python -m perf.<harness> --reprint` rebuilds the tables from the saved results without rerunning.

The gprx side is one binary, `gprx/` (`gprx-perf`), with one subcommand per harness:

```text
gprx-perf exact CASE.json [--memory]
gprx-perf online CASE.json [--stages | --delete]
gprx-perf sparse CASE.json
gprx-perf sparse-online CASE.json incremental|full
```

`gprx/src/case.rs` (JSON records), `rss.rs`, and `timing.rs` do not use gprx; `friedrich/` includes them by path.

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

`just perf` prints two tables: the speed pole (default `DistanceCachePolicy::Cached` + `CholeskyBuffer::Retain`), then the memory pole (`Gpr::with_prefer_memory` = `Uncached` + `Reuse`). P2B-16 time / RSS gates use the first table. The second table is recorded only; it does not add an RSS gate. The P2B-21 Uncached+Retain table stays in `.dev/bench-log.md` (local, not committed). `python -m perf.run --reprint` rebuilds the tables from `out/results.json` without rerunning.

Results: `compare/perf/out/results.json`. Pass / fail is recorded in `.dev/bench-log.md` (local, not committed). criterion is not used for these gates.

## Online insert (P3-6)

`just perf-online` times `OnlineGpr::insert` vs libgp `add_pattern` from `n = 2` to `n` on the same Forrester / sphere cases. Raw `y` (no `StandardizeTarget`, no z-score). Gate: `n = 256 / 1024` median ≤ libgp (5% inconclusive). `4096` and RSS are recorded only. Goldens for `just test` are `just gen-online-goldens` → `compare/goldens/online_libgp_*.json`.

`just perf-online-stages` rebuilds gprx-perf with `--features insert-stages` and prints kernel / bordered LDLT / X·y medians for Forrester `n = 256 / 1024`. It does not change the gate clock.

`just perf-online-delete` times gprx `OnlineGpr::delete` from `n` down to 2 (last remaining `PointId` each step; insert is untimed). No libgp cell. Used to record before / after of the P3-7 delete path.

## Sparse SGPR / SVGP (P4-12)

`just perf-sparse` times `Sgpr<Fixed>::factor` and `Svgp<Fixed>::factor` (prior `q`) against GPyTorch / GPy. Same Forrester / sphere `n` as P2B-16, `m = 16` (k-means seed `0` at generation time), population-standardized `y`, CPU, no `fit`. Clock matches P2B-16. Gate: factor / joint / predict and peak RSS each smaller than every opponent that wrote the cell (5% inconclusive). Unwritable cells are N/A. Results: `compare/perf/out/sparse_results.json`. Pass / fail is recorded in `.dev/bench-log.md` (local, not committed). `just perf` / `just perf-online` stay Exact / online. `cargo test` must not run this.

## Sparse online (P4-14)

`just perf-sparse-online` times `OnlineSgpr` `insert` / `delete` / `insert_inducing` / `delete_inducing` against self `Sgpr<Fixed>::factor` and GPyTorch Titsias assemble (no query). Same Forrester / sphere `n` as P2B-16, `m_max = 16` (k-means seed `0` at generation time), prefix `start_n = 32/128/512` and `start_m = 8`, raw `y`, CPU, no `fit`. The harness draws 32 ops (`ops_seed = 0`, RBF probe PD filter) and does not read P4-13 goldens. Prefix is untimed. Clock is one 32-op wall (discard 1 + median; reps follow P2B-16). Gate: incremental median smaller than self full and GPyTorch full (5% inconclusive). RSS is recorded only. Results: `compare/perf/out/sparse_online_results.json`. Pass / fail is recorded in `.dev/bench-log.md` (local, not committed). `just perf` / `just perf-online` / `just perf-sparse` stay Exact / online / batch Sparse. `cargo test` must not run this.

## Real datasets (B1-1)

`just perf-real` fits every library on the benchmark data of Gaussian-process papers, scores it (RMSE, NLPD, 95% coverage in the original units of `y`), and reports fit time, joint evaluation counts and peak RSS. `just perf-real-report` turns the raw output into `docs/bench/summary.json`, the SVG figures and the README tables. Manual. Not a CI gate. `just test` must not run this and needs no network.

```text
just perf-real-data                       # fetch every dataset once, pin the SHA-256 in real/checksums.json
just perf-real-check                      # fixed-θ agreement of all libraries (NLML, RMSE, NLPD to 1e-6)
just perf-real --datasets yacht,energy --splits 2 --protocol native --timeline
just perf-real --datasets kin40k --model sgpr --m 512 --protocol matched
just perf-real-report
```

- Data: `yaringal/DropoutUncertaintyExps` (T1 and Protein, the Hernández-Lobato & Adams splits), `treforevans/uci_datasets` (Kin40k and T3, 10 splits of 90 / 10), the NOAA Mauna Loa monthly means through `datasets/co2-ppm`, and Snelson's archive (the author's page is gone; the harness fetches the 2022-03-31 Internet Archive snapshot, and `SNELSON_ZIP` may point at a local copy). Files land in `out/real/data/`; a checksum mismatch stops the run.
- Cases: one JSON per (dataset, split, protocol, model) in `out/real/cases/`. `x`, `y` are standardized with the training statistics; metrics are converted back. One cell is one process: one split, one library.
- Protocols: `native` (each library's own optimizer), `matched` (scipy L-BFGS-B, 100 iterations, gradient tolerance √ε, history 10), `fixed` (no optimizer, for `perf-real-check`). The comparison run is `matched` only. `real/optimizers.py` records every optimizer with the source it was read from; `out/real/meta.json` records the machine and library versions.
- Exact fits are Snelson, Mauna Loa, yacht, and energy: one fit stays under about 5 minutes. Concrete is the same size class as energy and is not in this comparison. Wine and every larger set run as `Sgpr` (`just perf-real-full`). kin8nm and protein match kin40k, and buzz matches song, so those three are not repeated. GPyTorch's exact cell factors `K` with Cholesky up to that `n`. The library default switches to conjugate gradients above 800 rows, which is a different objective from the other libraries.
- Sparse models: `--model sgpr` (fixed inducing points, k-means with a fixed seed on at most 100 000 training rows) or `--model svgp` (Adam, one shared setting).
- Timing: one timed fit per cell after an untimed warm-up fit when n ≤ 5000 (`PERF_WARMUP` overrides); the split-to-split spread is the standard error. Joint evaluations are counted through `gprx::internals` (feature `bench-internals`, off by default) for gprx and through wrappers for the Python libraries; libgp's RProp counts its 100 iterations. A time difference between cells with different counts is not a speed difference.
- `--timeline`: the RSS of the whole process tree every 10 ms, with the start of load / warm-up / fit / predict marked (`out/real/timeline/`).
- Do not run two cells at once: they share the CPU and the timings mix.
- friedrich has no ARD kernel, so its cells are N/A. libgp's fit is Rprop only, so its `matched` cells are N/A.

### Real datasets: every path, and what is known not to work

`just perf-real-smoke` runs the whole matrix (dataset kind × model × protocol × library, with and without `--timeline`) on tiny problems, and with `--data` loads two splits of every dataset and checks the shapes against the sources' tables. A cell either runs or is N/A with a reason the script knows (`EXPECTED_NA`); anything else is a FAIL. Run it after touching a runner. `just perf-real-full` starts with it.

N/A on purpose: friedrich (no ARD kernel, no sparse model), libgp (Rprop only, so no `matched`; no sparse model; the composite Mauna Loa kernel is not wired), scikit-learn (no sparse model), GPy's SVGP (no minibatch fit in its API), SVGP `native` (no library default suits a large n) and SVGP `fixed`. An exact fit whose `K` and factor (2 n² f64) do not fit in the available memory is N/A with the sizes (`--force-exact` tries anyway). An SGPR fit whose cross kernel `K(X, Z)` needs six float64 copies that do not fit in physical memory is N/A the same way. On this machine that is HouseElectric (`n` ≈ 1.8e6, `m` = 512, one matrix ≈ 7.0 GiB).

Known limits of what is measured:

- gprx's argmin L-BFGS uses several function evaluations per iteration (Sgpr, n = 1000, m = 128, d = 8: about 720 for 100 iterations); scipy's L-BFGS-B uses about one. Compare times only with the evaluation counts beside them.
- At the same θ and Z, GPyTorch's sparse model has the same marginal likelihood as gprx and GPy but predicts with its own low-rank test covariance (RMSE / NLPD differ by a fraction of a percent).
- The case of a large dataset is one JSON file (HouseElectric: 600 MB) that each runner parses whole: allow a few GiB of memory and a minute of start-up per cell.
- The cloud VM this was developed on (4 vCPUs, 15 GiB) is too small for a final run: exact Kin40k / Protein do not fit, and an `Sgpr` fit at m = 512 takes hours.
