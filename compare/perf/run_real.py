"""B1-1: fit every library on the real datasets and print the tables.

```text
python -m perf.run_real [--datasets yacht,energy] [--splits N] [--protocol native|matched] [--libs gprx,sklearn] [--timeline] [--force-exact] [--model exact|sgpr|svgp] [--m 512]
python -m perf.run_real --reprint
```

A cell is one process: one split of one dataset for one library. Results go
to ``out/real/results.json``.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import urllib.error

from common import harness
from common.harness import fmt_rss, fmt_s, na_row, print_table, read_rows, start, write_json

from .real.cases import N_INDUCING, case_n_rows, splits_of, write_case, write_curve_case
from .real.curves import CURVES
from .real.data import DATASETS, OUT
from .real.libs import FIT_FIELDS, RUNNERS, runners_for
from .real.optimizers import OPTIMIZERS, meta

RESULTS = OUT / "results.json"

BANNER = (
    "one fit per cell (untimed warm-up fit first when n <= 5000); "
    "mean ± standard error over splits; RMSE / NLPD in original y units"
)


def option(argv: list[str], flag: str, default: str) -> str:
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            return argv[i + 1]
    return default


def mean_se(values: list[float]) -> str:
    if not values:
        return "N/A"
    mean = statistics.fmean(values)
    if len(values) < 2:
        return f"{mean:.4g}"
    return f"{mean:.4g} ± {statistics.stdev(values) / math.sqrt(len(values)):.2g}"


def print_tables(rows: list[dict]) -> None:
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        dataset = row["name"].rsplit("_s", 1)[0]
        groups.setdefault((f'{row["protocol"]} / {row.get("model", "exact")}', dataset, row["lib"]), []).append(row)
    for protocol in sorted({key[0] for key in groups}):
        lines = []
        for (proto, dataset, lib), cells in groups.items():
            if proto != protocol:
                continue
            ok = [c for c in cells if c.get("status") == "ok"]

            def col(key: str) -> list[float]:
                return [c[key] for c in ok if c.get(key) is not None]

            lines.append(
                (
                    dataset,
                    lib,
                    f"{len(ok)}/{len(cells)}",
                    mean_se(col("rmse")),
                    mean_se(col("nlpd")),
                    mean_se(col("coverage95")),
                    mean_se(col("fit_s")) if ok else "N/A",
                    mean_se(col("joint_evals")),
                    mean_se(col("nlml")),
                    fmt_rss(max((c["peak_rss_bytes"] for c in ok), default=None)),
                )
            )
        print_table(
            f"protocol: {protocol}",
            ("dataset", "lib", "ok", "RMSE", "NLPD", "cov95", "fit [s]", "joint evals", "NLML", "peak RSS"),
            lines,
        )


def merge_rows(path, fresh: list[dict]) -> list[dict]:
    """Earlier results, with the cells this run measured replaced."""
    def key(row: dict) -> tuple:
        return (row.get("model", "exact"), row["protocol"], row["name"], row["lib"])

    kept = {key(row): row for row in (read_rows(path) if path.is_file() else [])}
    kept.update({key(row): row for row in fresh})
    return list(kept.values())


def available_gib() -> float | None:
    try:
        for line in open("/proc/meminfo", encoding="utf-8"):
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    return None


def exact_skip_reason(n_rows: int, force: bool) -> str | None:
    """Why an exact fit is not attempted: the kernel matrix and its factor
    (two n × n f64 matrices, a lower bound of what any library needs) do not
    fit in the available memory. ``--force-exact`` tries anyway."""
    have = available_gib()
    if force or have is None:
        return None
    need = 2 * n_rows * n_rows * 8 / 2**30
    if need > 0.8 * have:
        return f"K and its factor need at least {need:.1f} GiB; {have:.1f} GiB available"
    return None


def print_optimizers() -> None:
    print_table(
        "optimizers",
        ("lib", "native", "matched", "space", "bounds"),
        ((lib, o["native"], o["matched"], o["space"], o["bounds"]) for lib, o in OPTIMIZERS.items()),
    )


def main(argv: list[str]) -> int:
    if "--reprint" in argv:
        print_tables(read_rows(RESULTS))
        return 0
    datasets = option(argv, "--datasets", "yacht").split(",")
    protocol = option(argv, "--protocol", "native")
    model = option(argv, "--model", "exact")
    n_inducing = int(option(argv, "--m", str(N_INDUCING)))
    runners = runners_for(model)
    libs = option(argv, "--libs", ",".join(RUNNERS)).split(",")
    limit = int(option(argv, "--splits", "0"))
    unknown = [d for d in datasets if d not in DATASETS and d not in CURVES] + [l for l in libs if l not in runners]
    if unknown:
        print(f"unknown: {unknown}", file=sys.stderr)
        return 2
    start(BANNER)
    rows: list[dict] = []
    for dataset in datasets:
        splits = [0] if dataset in CURVES else list(splits_of(dataset))
        if limit:
            splits = splits[:limit]
        for split in splits:
            try:
                path = (
                    write_curve_case(dataset, protocol)
                    if dataset in CURVES
                    else write_case(dataset, split, protocol, model, n_inducing)
                )
            except urllib.error.URLError as err:
                # The source is not reachable from this machine (e.g. a proxy
                # denies the host): every library is N/A for this case, with why.
                reason = f"source not reachable: {err.reason}"
                for lib in libs:
                    row = na_row(reason, FIT_FIELDS)
                    row.update(lib=lib, name=f"{dataset}_s{split}", protocol=protocol, model=model)
                    rows.append(row)
                print(f"# {dataset}_s{split}: N/A {reason}", flush=True)
                continue
            print(f"# {path.name}", flush=True)
            for lib in libs:
                reason = (
                    exact_skip_reason(case_n_rows(path), "--force-exact" in argv)
                    if model == "exact" and protocol != "fixed" and DATASETS.get(dataset) and DATASETS[dataset].tier in ("T2", "T3")
                    else None
                )
                if reason is not None:
                    row = na_row(reason, FIT_FIELDS)
                    row.update(lib=lib, name=f"{dataset}_s{split}", protocol=protocol, model=model)
                    rows.append(row)
                    print(f"  {lib}: N/A {reason}", flush=True)
                    continue
                if "--timeline" in argv:
                    harness.TIMELINE = (OUT / "timeline", f"{dataset}_{model}_s{split}_{protocol}_{lib}")
                row = runners[lib](path)
                row["model"] = model
                row.setdefault("lib", lib)
                row.setdefault("name", f"{dataset}_s{split}")
                row.setdefault("protocol", protocol)
                rows.append(row)
                print(
                    f"  {lib}: {row.get('status')} fit={fmt_s(row.get('fit_s'))} "
                    f"evals={row.get('joint_evals')} rmse={row.get('rmse')} nlpd={row.get('nlpd')}"
                    + (f" [{row.get('note')}]" if row.get("status") != "ok" else ""),
                    flush=True,
                )
    if "--timeline" not in argv:
        # A timeline run only writes its RSS files: the sampler costs a little
        # CPU, so its timings are not results.
        write_json(RESULTS, merge_rows(RESULTS, rows))
    write_json(OUT / "meta.json", meta())
    print_tables(rows)
    print_optimizers()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
