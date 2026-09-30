"""B1-1: fit every library on the real datasets and print the tables.

```text
python -m perf.run_real [--datasets yacht,energy] [--splits N] [--protocol native|matched] [--libs gprx,sklearn]
python -m perf.run_real --reprint
```

A cell is one process: one split of one dataset for one library. Results go
to ``out/real/results.json``.
"""

from __future__ import annotations

import math
import statistics
import sys

from common.harness import fmt_rss, fmt_s, na_row, print_table, read_rows, start, write_json

from .real.cases import splits_of, write_case
from .real.data import DATASETS, OUT
from .real.libs import RUNNERS

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
        groups.setdefault((row["protocol"], dataset, row["lib"]), []).append(row)
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


def main(argv: list[str]) -> int:
    if "--reprint" in argv:
        print_tables(read_rows(RESULTS))
        return 0
    datasets = option(argv, "--datasets", "yacht").split(",")
    protocol = option(argv, "--protocol", "native")
    libs = option(argv, "--libs", ",".join(RUNNERS)).split(",")
    limit = int(option(argv, "--splits", "0"))
    unknown = [d for d in datasets if d not in DATASETS] + [l for l in libs if l not in RUNNERS]
    if unknown:
        print(f"unknown: {unknown}", file=sys.stderr)
        return 2
    start(BANNER)
    rows: list[dict] = []
    for dataset in datasets:
        splits = list(splits_of(dataset))
        if limit:
            splits = splits[:limit]
        for split in splits:
            path = write_case(dataset, split, protocol)
            print(f"# {path.name}", flush=True)
            for lib in libs:
                row = RUNNERS[lib](path)
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
    write_json(RESULTS, rows)
    print_tables(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
