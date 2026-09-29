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
    uv run --directory compare python -X utf8 generate.py

# Regenerates libgp online-insert goldens (P3-6). cargo test must not run this.
gen-online-goldens:
    uv run --directory compare python -X utf8 generate_online_libgp.py

# Regenerates GPyTorch sparse / SVGP goldens (P4-11). cargo test must not run this.
gen-sparse-goldens:
    uv run --directory compare python -X utf8 generate_sparse_gpytorch.py

# Regenerates GPyTorch OnlineSgpr goldens (P4-13). cargo test must not run this.
gen-sparse-online-goldens:
    uv run --directory compare python -X utf8 generate_sparse_online_gpytorch.py

bench:
    cargo bench --features bench-internals --bench exact

# Manual P2B-16 harness. cargo test must not run this.
perf:
    uv run --directory compare --group perf python -X utf8 -m perf.run

# Manual P3-6 insert-sequence vs libgp. cargo test must not run this.
perf-online:
    uv run --directory compare --group perf python -X utf8 -m perf.run_online

# P3-7 insert stages (gprx Forrester 256/1024). cargo test must not run this.
perf-online-stages:
    uv run --directory compare --group perf python -X utf8 -m perf.run_online --stages

# P3-7 delete n→2 (gprx only). cargo test must not run this.
perf-online-delete:
    uv run --directory compare --group perf python -X utf8 -m perf.run_online --delete

# Manual P4-12 Sparse harness. cargo test must not run this.
perf-sparse:
    uv run --directory compare --group perf python -X utf8 -m perf.run_sparse

# Manual P4-14 Sparse-online harness. cargo test must not run this.
perf-sparse-online:
    uv run --directory compare --group perf python -X utf8 -m perf.run_sparse_online
