# ADR 0001: faer parallelism

- Status: accepted
- Date: 2026-09-19
- Issue: [#143](https://github.com/YUKIKEDA/gprx/issues/143) (P2B-22)

## Context

At n=4096, Cholesky and the n triangular solves that form `W` are more than 97% of the wall time of one joint MLL+grad. sklearn also builds A⁻¹ with `cho_solve(L, I)`. gprx used `Par::Seq`. Always using `Par::rayon(0)` on an unset Rayon pool (16 threads on this machine) makes an n=256 eval several times slower.

## Decision

- Cholesky of the factor and `W` (n right-hand sides) use `faer_par(n) = Par::rayon(min(pool, n/64))`
- α (one right-hand side) and `L⁻¹ k_*` in predict / `predict_covariance` (`n×m`) use `faer_par_dims(n, k) = Par::rayon(min(pool, n/64, n·k/16384, k/12))`. When `k = n` this matches the square formula (`k/12` is the loose term)
- Kernel fill stays on the process-wide pool
- There is no public `n_jobs` and no public parallel on/off. `RAYON_NUM_THREADS=1` is one thread
- faer scratch on `Workspace` is taken with the same `Par`
- Cholesky in `benches/exact.rs` stays Seq

## Evidence (this machine, 16 logical processors, 5 runs each)

Mean of eval 10. Ratio against one thread.

| n | Problem | 1 thread | 4 | 8 | 16 |
|---:|---|---:|---:|---:|---:|
| 256 | Forrester | 12.3 ms | **10.5 ms** | 18.5 ms | 105 ms |
| 256 | sphere | 14.5 ms | **10.2 ms** | 31.2 ms | 32.8 ms |
| 1024 | Forrester | 546 ms | 209 ms | **168 ms** | 193 ms |
| 1024 | sphere | 585 ms | 233 ms | **189 ms** | 250 ms |
| 2048 | Forrester | 4.54 s | 1.54 s | **1.06 s** | 1.07 s |
| 2025 | sphere | 3.90 s | 1.41 s | **0.98 s** | 1.01 s |
| 4096 | Forrester | 38.8 s | 11.7 s | 8.68 s | **8.34 s** |
| 4096 | sphere | 37.8 s | 12.1 s | 9.09 s | **8.68 s** |

`n/64` is 256→4, 1024→16, 4096→16. Mixing faer at 4 threads with the kernel at 16 avoids the 16-thread accident at 256 and is a few milliseconds slower than both at 4. For n≥1024 the formula uses the whole pool, so narrowing the kernel does not change the time.

predict 100 has `k = 100`. Leaving `faer_par(n)` in place uses 16 threads at n=1024, and the triangular solve jumps between 1.5 and 22 ms (allocation and the kernel stay at 0.2–0.8 ms). At n=4096, still on 16 threads, one of five runs was 103 ms. In the same cell Seq / 1 thread is 2.4–3.0 ms, 4 threads is 1.6–2.1 ms, and 8 threads is 1.5–1.8 ms. `n·k/16384` is the 4-thread eval point at n=256 (`256²/4`). `k/12` caps 100 columns at 8 threads. 1024×100 is 6 threads. 4096×100 is 8 threads.

Wall time is one face of the cost. The algorithm (n solves for A⁻¹) is the same as sklearn. The `n×n` allocation of `I` / `W` is #142 / P2B-19.

## Addendum: n=550 with a composite kernel (2026-10-01, [#376](https://github.com/YUKIKEDA/gprx/issues/376))

The table above has no size where the factor is only part of a joint eval. Mauna Loa (matched B1-1 case, n=550, `C×RBF + C×RBF×Periodic + C×RQ + C×RBF`) was measured on a 4-logical-processor VM, so `faer_par(550)` is 4 workers. Joint MLL+grad, one eval, 3 runs each.

| State | Threads | faer as the formula says | faer `Par::Seq` (trial change) |
|---|---:|---:|---:|
| Before #373 | 1 | 34–38 ms | — |
| Before #373 | 4 | 40–49 ms | 29.5 ms |
| After #373–#375 | 1 | 17.9–18.5 ms | 17.9–18.4 ms |
| After #373–#375 | 2 | 19.1–19.5 ms | 17.5–18.5 ms |
| After #373–#375 | 3 | 18.0–18.9 ms | 17.3–17.8 ms |
| After #373–#375 | 4 | 16.4–16.7 ms | 16.3–16.5 ms |

Before #373, about 14% of the 4-thread CPU time was spindle (the pool faer's gemm runs on Rayon workers) waiting in its barrier, `_mm_pause`, and `sched_yield`. It competed with the kernel fill, which was then scalar libm and heavy. After #373–#375 spindle still spins (about 21% of the CPU time at 4 threads), but the wall time with and without faer parallelism is the same at 4 threads. It is 1–2 ms apart at 2–3 threads. Most of what remains on the main thread is gprx's own serial `n²` loops, not faer.

Decision: the formula stays. The serial loops are parallelized in [#377](https://github.com/YUKIKEDA/gprx/issues/377). A size or machine that shows faer losing wall time again reopens this record.

## Consequences

- On an unset 16-thread pool, the square kernel at n=256 uses 4 faer threads, predict 100 at n=1024 uses 6, and predict 100 at n=4096 uses 8
- Parallelism can change bitwise results. Numerical tests use a tolerance
- Pass/fail is `compare/perf`. criterion is not used. The machine log is local `.dev/bench-log.md` and is not committed
