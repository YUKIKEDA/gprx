"""Run every P4-14 Sparse-online cell and print the bench-log table plus gates."""

from __future__ import annotations

import sys

from common.harness import (
    FAIL,
    NA,
    OPS_FIELDS,
    Row,
    by_case,
    case_filter,
    combine_strict,
    fmt_rss,
    fmt_time,
    print_gates,
    print_table,
    ratio_verdict,
    read_rows,
    run_cells,
    select,
    split_case,
    start,
    write_json,
)

from .runners import OUT, run_gprx, run_python
from .sparse_online_cases import write_sparse_online_cases

PROBLEMS = OUT / "sparse_online_problems"
RESULTS = OUT / "sparse_online_results.json"
VERDICTS = OUT / "sparse_online_verdicts.json"
BAND = 0.05
LIBS = ("gprx-inc", "gprx-full", "gpytorch")

BANNER = (
    "P4-14 Sparse-online times: prefix untimed; 32 ops as one wall clock; "
    "discard PERF_WARMUP (default 1) then median of PERF_REPS "
    "(default 51/21/7 by n); RSS is one-process peak; CPU; raw y; "
    "no predict / NLML"
)

RUNNERS = (
    ("gprx-inc", lambda path: run_gprx("sparse-online", path, "incremental")),
    ("gprx-full", lambda path: run_gprx("sparse-online", path, "full")),
    ("gpytorch", lambda path: run_python("gpytorch_sparse_online", path, OPS_FIELDS)),
)


def judge(inc: Row, other: Row) -> str:
    """The incremental 32-op wall smaller than ``other`` (5% inconclusive)."""
    if inc.get("status") != "ok" or other.get("status") != "ok":
        return NA
    return ratio_verdict(inc.get("ops_s"), other.get("ops_s"), BAND)


def emit_tables(rows: list[Row]) -> int:
    grouped = by_case(rows)
    verdicts: list[tuple[str, str, str]] = []
    lines: list[list[str]] = []
    for name, libs in grouped.items():
        inc = libs.get("gprx-inc", {})
        _, problem, n = split_case(name)
        for lib in LIBS:
            row = libs.get(lib, {})
            if lib == "gprx-inc":
                gate = "-"
            else:
                gate = judge(inc, row)
                verdicts.append((name, lib, gate))
            lines.append(
                [problem, n, lib, fmt_time(row, "ops"), fmt_rss(row.get("peak_rss_bytes")), gate]
            )
    print_table(
        "cells (Sparse online, CPU)",
        ("問題", "n", "lib", "ops 32", "peak RSS", "vs inc"),
        lines,
    )
    cell_verdicts: list[tuple[str, str]] = []
    for name, libs in grouped.items():
        inc = libs.get("gprx-inc", {})
        cell_verdicts.append(
            (
                name,
                combine_strict(
                    judge(inc, libs.get("gprx-full", {})),
                    judge(inc, libs.get("gpytorch", {})),
                ),
            )
        )
    print_gates("gates", (f"{name}: {v}" for name, v in cell_verdicts))
    fails = [(n, lib, v) for n, lib, v in verdicts if v == FAIL]
    write_json(VERDICTS, {"verdicts": verdicts, "cells": cell_verdicts, "fails": fails})
    return 0


def main() -> int:
    start(BANNER)
    OUT.mkdir(parents=True, exist_ok=True)
    if "--reprint" in sys.argv:
        return emit_tables(read_rows(RESULTS))
    only = case_filter(sys.argv)
    cases = select(write_sparse_online_cases(PROBLEMS), only, lambda p: p.stem in only)
    if cases is None:
        return 2
    rows = run_cells(cases, RUNNERS, lambda row: f"ops={fmt_time(row, 'ops')}")
    write_json(RESULTS, rows)
    return emit_tables(rows)


if __name__ == "__main__":
    raise SystemExit(main())
