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
- fit wall clock + joint-eval count, then predict 100 points, then peak RSS

gprx uses `Gpr::fit` with `StandardizeTarget` (same as `benches/exact.rs`). sklearn uses `normalize_y=True`. friedrich / libgp z-score `y` in the runner. friedrich has no ARD kernel: those cells are N/A. libgp Python bindings fail to build on MSVC (`M_PI`, `drand48`); that is N/A, not a loss.

Results: `compare/perf/out/results.json`. Pass / fail is recorded in `.dev/bench-log.md`. criterion is not used for these gates.
