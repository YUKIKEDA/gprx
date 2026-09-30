"""Run the P3-6 online insert (or P3-7 stages / delete) cells and print the tables."""

from __future__ import annotations

import sys
from pathlib import Path

from common.harness import (
    FAIL,
    NA,
    Row,
    by_case,
    case_filter,
    fmt_rss,
    fmt_s_range,
    fmt_time,
    print_gates,
    print_table,
    read_rows,
    run_cells,
    select,
    split_case,
    start,
    within_band_or_better,
    write_json,
)
from common.problems import FORRESTER_NS, SPHERE_SIDES, make_forrester, make_sphere, write_cases

from .runners import OUT, run_gprx, run_libgp

PROBLEMS = OUT / "problems"
VERDICTS = OUT / "online_verdicts.json"
ONLINE_BAND = 0.05
GATED_NS = {"256", "1024"}
STAGES_CASES = {"forrester_n256", "forrester_n1024"}


def results_path(delete: bool) -> Path:
    return OUT / ("online_delete_results.json" if delete else "online_results.json")


def banner(stages: bool, delete: bool) -> str:
    clock = (
        "clock is delete from n to 2 (last remaining PointId; insert untimed); "
        if delete
        else "clock is add from n=2 to n (first two points untimed); "
    )
    if stages:
        scope = "stages are kernel / bordered LDLT / X·y (gprx only)"
    elif delete:
        scope = "gprx only; no libgp"
    else:
        scope = "gate is 256/1024 median <= libgp (5% inconclusive); 4096 record"
    return (
        "online "
        + ("delete" if delete else "insert")
        + ": discard PERF_WARMUP then median of PERF_REPS; "
        + clock
        + scope
    )


def online_cases() -> list[Path]:
    cases = [make_forrester(n) for n in FORRESTER_NS]
    cases.extend(make_sphere(side) for side in SPHERE_SIDES)
    return write_cases(PROBLEMS, cases, prefix="online_")


def runners(stages: bool, delete: bool) -> tuple:
    if delete:
        gprx = ("gprx", lambda path: run_gprx("online", path, "--delete"))
    elif stages:
        gprx = (
            "gprx",
            lambda path: run_gprx("online", path, "--stages", features="insert-stages"),
        )
    else:
        gprx = ("gprx", lambda path: run_gprx("online", path))
    if stages or delete:
        return (gprx,)
    return (gprx, ("libgp", lambda path: run_libgp("libgp-online", path, "time")))


def judge_insert(gprx: Row, other: Row) -> str:
    """The gprx insert sequence within 5% of libgp or faster."""
    if gprx.get("status") != "ok" or other.get("status") != "ok":
        return NA
    if gprx.get("factor_s") is None or other.get("factor_s") is None:
        return NA
    return within_band_or_better(gprx["factor_s"], other["factor_s"], ONLINE_BAND)


def progress(stages: bool, delete: bool):
    label = "delete" if delete else "insert"

    def line(row: Row) -> str:
        extra = ""
        if stages:
            extra = (
                f" kernel={fmt_s_range(row.get('kernel_s'), None, None)}"
                f" border={fmt_s_range(row.get('border_s'), None, None)}"
                f" rest={fmt_s_range(row.get('rest_s'), None, None)}"
            )
        return f"{label}={fmt_time(row, 'factor')}{extra}"

    return line


def emit_tables(rows: list[Row], stages: bool, delete: bool) -> int:
    grouped = by_case(rows)
    if delete:
        lines = []
        for name, libs in grouped.items():
            row = libs.get("gprx", {})
            _, problem, n = split_case(name.removeprefix("online_"))
            lines.append([problem, n, fmt_time(row, "factor"), fmt_rss(row.get("peak_rss_bytes"))])
        print_table("online delete n→2 (gprx)", ("問題", "n", "delete n→2", "peak RSS"), lines)
        return 0
    if stages:
        lines = []
        for name, libs in grouped.items():
            row = libs.get("gprx", {})
            _, problem, n = split_case(name.removeprefix("online_"))
            lines.append(
                [
                    problem,
                    n,
                    fmt_time(row, "factor"),
                    fmt_s_range(row.get("kernel_s"), None, None),
                    fmt_s_range(row.get("border_s"), None, None),
                    fmt_s_range(row.get("rest_s"), None, None),
                ]
            )
        print_table(
            "online insert stages (gprx)",
            ("問題", "n", "insert", "kernel", "border", "rest"),
            lines,
        )
        return 0
    verdicts: list[tuple[str, str]] = []
    lines = []
    for name, libs in grouped.items():
        gprx = libs.get("gprx", {})
        libgp = libs.get("libgp", {})
        _, problem, n = split_case(name.removeprefix("online_"))
        gated = n in GATED_NS
        gate = judge_insert(gprx, libgp) if gated else "record"
        if gated:
            verdicts.append((name, gate))
        for lib, row in (("gprx", gprx), ("libgp", libgp)):
            lines.append(
                [
                    problem,
                    n,
                    lib,
                    fmt_time(row, "factor"),
                    fmt_rss(row.get("peak_rss_bytes")),
                    gate if lib == "gprx" else "-",
                ]
            )
    print_table(
        "online insert (raw y)",
        ("問題", "n", "lib", "insert n=2→n", "peak RSS", "ゲート"),
        lines,
    )
    print_gates("gates (256 / 1024)", (f"{name}: {v}" for name, v in verdicts))
    fails = [(n, v) for n, v in verdicts if v == FAIL]
    write_json(VERDICTS, {"verdicts": verdicts, "fails": fails})
    return 0


def main() -> int:
    stages = "--stages" in sys.argv[1:]
    delete = "--delete" in sys.argv[1:]
    start(banner(stages, delete))
    if "--reprint" in sys.argv:
        return emit_tables(read_rows(results_path(delete)), stages, delete)
    only = case_filter(sys.argv)
    if stages and not only:
        only = set(STAGES_CASES)
    cases = select(
        online_cases(),
        only,
        lambda p: p.stem in only or p.stem.removeprefix("online_") in only,
    )
    if cases is None:
        return 2
    rows = run_cells(cases, runners(stages, delete), progress(stages, delete))
    write_json(results_path(delete), rows)
    return emit_tables(rows, stages, delete)


if __name__ == "__main__":
    raise SystemExit(main())
