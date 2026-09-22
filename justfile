set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

lint:
    cargo fmt --check
    cargo clippy --all-targets -- -D warnings

fmt:
    cargo fmt

test:
    cargo test

# Regenerates compare/goldens/*.json via sklearn. cargo test must not run this.
gen-goldens:
    uv run --directory compare python generate.py

# Regenerates libgp online-insert goldens (P3-6). cargo test must not run this.
gen-online-goldens:
    $env:PYTHONUTF8 = "1"; uv run --directory compare python generate_online_libgp.py

# Regenerates GPyTorch sparse / SVGP goldens (P4-11). cargo test must not run this.
gen-sparse-goldens:
    $env:PYTHONUTF8 = "1"; uv run --directory compare python generate_sparse_gpytorch.py

# Regenerates GPyTorch OnlineSgpr goldens (P4-13). cargo test must not run this.
gen-sparse-online-goldens:
    $env:PYTHONUTF8 = "1"; uv run --directory compare python generate_sparse_online_gpytorch.py

bench:
    cargo bench --bench exact

# Manual P2B-16 harness. cargo test must not run this.
perf:
    $env:PYTHONUTF8 = "1"; uv run --directory compare/perf python run.py

# Manual P3-6 insert-sequence vs libgp. cargo test must not run this.
perf-online:
    $env:PYTHONUTF8 = "1"; uv run --directory compare/perf python run_online.py

# P3-7 insert stages (gprx Forrester 256/1024). cargo test must not run this.
perf-online-stages:
    $env:PYTHONUTF8 = "1"; uv run --directory compare/perf python run_online.py --stages

# P3-7 delete n→2 (gprx only). cargo test must not run this.
perf-online-delete:
    $env:PYTHONUTF8 = "1"; uv run --directory compare/perf python run_online.py --delete

# Manual P4-12 Sparse harness. cargo test must not run this.
perf-sparse:
    $env:PYTHONUTF8 = "1"; uv run --directory compare/perf python run_sparse.py

# Manual P4-14 Sparse-online harness. cargo test must not run this.
perf-sparse-online:
    $env:PYTHONUTF8 = "1"; uv run --directory compare/perf python run_sparse_online.py
