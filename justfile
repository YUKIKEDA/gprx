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

bench:
    cargo bench --bench exact

# Manual P2B-16 harness. cargo test must not run this.
perf:
    $env:PYTHONUTF8 = "1"; uv run --directory compare/perf python run.py
