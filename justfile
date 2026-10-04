set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

lint:
    cargo fmt --check
    cargo clippy --all-targets -- -D warnings

fmt:
    cargo fmt

test:
    cargo test

# Line coverage of src/ from the default-feature tests. Not part of `test`.
# The floor is the integer part of the percent measured when this gate was added.
coverage:
    cargo llvm-cov --fail-under-lines 90

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

# B1-1 real-dataset comparison. Manual: needs the network for the data. cargo test must not run this.
# Example: just perf-real --datasets yacht,energy --splits 2 --protocol native
# Example: just perf-real --datasets power_plant --from-split 4
perf-real *args:
    uv run --directory compare --group perf python -X utf8 -m perf.run_real {{args}}

# B1-1 fixed-θ agreement of every library (run before trusting a comparison).
perf-real-check *args:
    uv run --directory compare --group perf python -X utf8 -m perf.real.check {{args}}

# B1-1 every path (dataset kind × model × protocol × library) on tiny problems; a path that cannot
# run must say why. Add --data to load two splits of every dataset (downloads them once).
perf-real-smoke *args:
    uv run --directory compare --group perf python -X utf8 -m perf.real.smoke {{args}}

# B1-1 download the datasets and pin their checksums (compare/perf/real/checksums.json).
perf-real-data:
    uv run --directory compare --group perf python -X utf8 -m perf.real.data --pin

# B1-1 figures, aggregated results, and the README tables from compare/perf/out/real.
perf-real-report *args:
    uv run --directory compare --group perf python -X utf8 -m perf.real.report {{args}}

# B1-1 the comparison that the README tables come from (hours).
# One optimizer (matched). Exact only where one fit stays under about 5 minutes:
# Snelson, Mauna Loa, yacht, energy. Concrete is the same size class as energy
# and is not repeated. Wine and larger sets are Sgpr. kin8nm and protein match
# kin40k; buzz matches song; those three are not repeated. HouseElectric Sgpr
# returns N/A when K(X, Z) does not fit. A timeline run writes RSS files only.
perf-real-full:
    just perf-real-data
    just perf-real-smoke --data
    just perf-real-check yacht
    just perf-real-check maunaloa
    just perf-real --datasets snelson,maunaloa --protocol matched --libs gprx,sklearn,gpytorch,gpy
    just perf-real --datasets yacht,energy --protocol matched --libs gprx,sklearn,gpytorch,gpy
    just perf-real --datasets wine_red,power_plant,naval --model sgpr --protocol matched --libs gprx,gpytorch,gpy
    just perf-real --datasets kin40k --splits 5 --model sgpr --protocol matched --libs gprx,gpytorch,gpy
    just perf-real --datasets kin40k,3droad,song,houseelectric --splits 1 --model svgp --protocol matched --libs gprx,gpytorch
    just perf-real --datasets 3droad,song,houseelectric --splits 1 --model sgpr --protocol matched --libs gprx,gpytorch,gpy
    just perf-real --datasets energy --splits 1 --protocol matched --libs gprx,sklearn,gpytorch,gpy --timeline
    just perf-real --datasets kin40k --splits 1 --model sgpr --protocol matched --libs gprx,gpytorch,gpy --timeline
    just perf-real-report
