"""Run every P4-12 Sparse cell and print the bench-log table plus gates."""

from __future__ import annotations

import sys

from common.harness import (
    FAIL,
    NA,
    PASS,
    UNDECIDED,
    Row,
    by_case,
    case_filter,
    combine_times,
    fmt_evals,
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
from .sparse_cases import MODELS, write_sparse_cases

PROBLEMS = OUT / "sparse_problems"
RESULTS = OUT / "sparse_results.json"
VERDICTS = OUT / "sparse_verdicts.json"
BAND = 0.05
LIBS = ("gprx", "gpytorch", "gpy")

BANNER = (
    "P4-12 Sparse times: discard PERF_WARMUP (default 1) then median of "
    "PERF_REPS (default 51/21/7 by n); eval N is N × median of one joint "
    "call; RSS is one-process peak; CPU; y population-standardized"
)

RUNNERS = (
    ("gprx", lambda path: run_gprx("sparse", path)),
    ("gpytorch", lambda path: run_python("gpytorch_sparse", path)),
    ("gpy", lambda path: run_python("gpy_sparse", path)),
)


def judge(gprx: Row, other: Row) -> str:
    """factor, joint, predict, and RSS each smaller than the opponent (5%
    inconclusive); cells the opponent could not write are N/A."""
    if gprx.get("status") != "ok" or other.get("status") != "ok":
        return NA
    time_v = combine_times(
        *(
            ratio_verdict(gprx.get(key), other.get(key), BAND)
            for key in ("factor_s", "eval_s", "predict_s")
        )
    )
    rss_v = ratio_verdict(gprx.get("peak_rss_bytes"), other.get("peak_rss_bytes"), BAND)
    if time_v == PASS and rss_v == PASS:
        return PASS
    if FAIL in (time_v, rss_v):
        return FAIL
    if NA in (time_v, rss_v):
        return NA
    return UNDECIDED


def emit_tables(rows: list[Row]) -> int:
    verdicts: list[tuple[str, str, str]] = []
    lines: list[list[str]] = []
    for name, libs in by_case(rows).items():
        gprx = libs.get("gprx", {})
        model, problem, n = split_case(name, MODELS)
        for lib in LIBS:
            row = libs.get(lib, {})
            if lib == "gprx":
                gate = "-"
            else:
                gate = judge(gprx, row)
                verdicts.append((name, lib, gate))
            lines.append(
                [
                    model,
                    problem,
                    n,
                    lib,
                    fmt_time(row, "factor"),
                    fmt_time(row, "eval"),
                    fmt_evals(row.get("joint_evals")),
                    fmt_time(row, "predict"),
                    fmt_rss(row.get("peak_rss_bytes")),
                    gate,
                ]
            )
    print_table(
        "cells (Sparse, CPU)",
        ("面", "問題", "n", "lib", "factor", "eval N", "evals", "predict 100", "peak RSS", "vs gprx"),
        lines,
    )
    print_gates("gates", (f"{name} vs {lib}: {v}" for name, lib, v in verdicts))
    fails = [(n, lib, v) for n, lib, v in verdicts if v == FAIL]
    write_json(VERDICTS, {"verdicts": verdicts, "fails": fails})
    return 0


def main() -> int:
    start(BANNER)
    OUT.mkdir(parents=True, exist_ok=True)
    if "--reprint" in sys.argv:
        return emit_tables(read_rows(RESULTS))
    only = case_filter(sys.argv)
    cases = select(write_sparse_cases(PROBLEMS), only, lambda p: p.stem in only)
    if cases is None:
        return 2
    rows = run_cells(
        cases,
        RUNNERS,
        lambda row: f"factor={fmt_time(row, 'factor')} eval={fmt_time(row, 'eval')}",
    )
    write_json(RESULTS, rows)
    return emit_tables(rows)


if __name__ == "__main__":
    raise SystemExit(main())
