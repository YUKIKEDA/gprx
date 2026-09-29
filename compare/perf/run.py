"""Run every P2B-16 Exact cell and print the bench-log tables plus gates."""

from __future__ import annotations

import sys

from common.harness import (
    FAIL,
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
    within_band_or_better,
    write_json,
)
from common.problems import all_cases, write_cases

from .runners import OUT, run_friedrich, run_gprx, run_libgp, run_python

PROBLEMS = OUT / "problems"
RESULTS = OUT / "results.json"
VERDICTS = OUT / "verdicts.json"

SKLEARN_BAND = 0.05
LIBGP_BAND = 0.10

BANNER = (
    "times: discard PERF_WARMUP (default 1) then median of PERF_REPS "
    "(default 51/21/7 by n); eval N is N × median of one joint call; "
    "RSS is one-process peak"
)
COLUMNS = ("問題", "n", "lib", "factor", "eval N", "evals", "predict 100", "peak RSS")


RUNNERS = (
    ("gprx", lambda path: run_gprx("exact", path)),
    ("gprx-memory", lambda path: run_gprx("exact", path, "--memory")),
    ("sklearn", lambda path: run_python("sklearn_run", path)),
    ("libgp", lambda path: run_libgp("libgp-perf", path)),
    ("friedrich", run_friedrich),
)


def judge_sklearn(gprx: Row, other: Row) -> str:
    """Time (factor, joint) and RSS both smaller than sklearn (5% inconclusive)."""
    if gprx.get("status") != "ok" or other.get("status") != "ok":
        return "N/A"
    time_v = combine_times(
        ratio_verdict(gprx["factor_s"], other["factor_s"], SKLEARN_BAND),
        ratio_verdict(gprx["eval_s"], other["eval_s"], SKLEARN_BAND),
    )
    rss_v = ratio_verdict(gprx["peak_rss_bytes"], other["peak_rss_bytes"], SKLEARN_BAND)
    if time_v == PASS and rss_v == PASS:
        return PASS
    if FAIL in (time_v, rss_v):
        return FAIL
    return UNDECIDED


def judge_libgp(gprx: Row, other: Row) -> str:
    """Time (factor, joint) and RSS within 10% of libgp or better."""
    if gprx.get("status") != "ok" or other.get("status") != "ok":
        return "N/A"
    time_v = combine_times(
        within_band_or_better(gprx["factor_s"], other["factor_s"], LIBGP_BAND),
        within_band_or_better(gprx["eval_s"], other["eval_s"], LIBGP_BAND),
    )
    rss_v = within_band_or_better(gprx["peak_rss_bytes"], other["peak_rss_bytes"], LIBGP_BAND)
    judged = [v for v in (time_v, rss_v) if v != UNDECIDED]
    if any(v == FAIL for v in judged):
        return FAIL
    if judged and all(v == PASS for v in judged):
        return PASS
    return UNDECIDED


def judge_friedrich(gprx: Row, other: Row) -> str:
    """Factor time or RSS smaller than friedrich (5% inconclusive)."""
    if gprx.get("status") != "ok" or other.get("status") != "ok":
        return "N/A"
    time_v = ratio_verdict(gprx["factor_s"], other["factor_s"], SKLEARN_BAND)
    rss_v = ratio_verdict(gprx["peak_rss_bytes"], other["peak_rss_bytes"], SKLEARN_BAND)
    if PASS in (time_v, rss_v):
        return PASS
    if time_v == FAIL and rss_v == FAIL:
        return FAIL
    return UNDECIDED


JUDGES = {
    "sklearn": judge_sklearn,
    "libgp": judge_libgp,
    "friedrich": judge_friedrich,
}


def cells(row: Row) -> list[str]:
    return [
        fmt_time(row, "factor"),
        fmt_time(row, "eval"),
        fmt_evals(row.get("joint_evals")),
        fmt_time(row, "predict"),
        fmt_rss(row.get("peak_rss_bytes")),
    ]


def emit_tables(rows: list[Row]) -> int:
    grouped = by_case(rows)
    verdicts: list[tuple[str, str, str]] = []
    speed: list[list[str]] = []
    for name, libs in grouped.items():
        gprx = libs.get("gprx", {})
        _, problem, n = split_case(name)
        for lib in ("gprx", *JUDGES):
            row = libs.get(lib, {})
            if lib == "gprx":
                gate = "-"
            else:
                gate = JUDGES[lib](gprx, row)
                verdicts.append((name, lib, gate))
            speed.append([problem, n, lib, *cells(row), gate])
    print_table("cells (CachedDistances, default)", (*COLUMNS, "vs gprx"), speed)
    memory: list[list[str]] = []
    for name, libs in grouped.items():
        _, problem, n = split_case(name)
        for lib in ("gprx-memory", "libgp"):
            memory.append([problem, n, lib, *cells(libs.get(lib, {})), "record"])
    print_table(
        "cells (with_prefer_memory = UncachedDistances + ReuseCholesky)",
        (*COLUMNS, "vs libgp RSS"),
        memory,
    )
    print_gates("gates", (f"{name} vs {lib}: {v}" for name, lib, v in verdicts))
    print_gates(
        "P2B-23 memory pole (record only; no new RSS gate)",
        ["table 2 is with_prefer_memory vs libgp; P2B-21 Uncached+Retain stays in bench-log"],
    )
    fails = [(n, lib, v) for n, lib, v in verdicts if v == FAIL]
    write_json(VERDICTS, {"verdicts": verdicts, "fails": fails})
    return 0


def main() -> int:
    start(BANNER)
    OUT.mkdir(parents=True, exist_ok=True)
    if "--reprint" in sys.argv:
        return emit_tables(read_rows(RESULTS))
    only = case_filter(sys.argv)
    cases = select(write_cases(PROBLEMS, all_cases()), only, lambda p: p.stem in only)
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
