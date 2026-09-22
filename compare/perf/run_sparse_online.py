"""Run every P4-14 Sparse-online cell and print the bench-log table plus gates."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from sparse_online_cases import write_sparse_online_cases

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
OUT = ROOT / "out"
PROBLEMS = OUT / "sparse_online_problems"
BAND = 0.05
LIBS = ("gprx-inc", "gprx-full", "gpytorch")


def run_cmd(args: list[str], cwd: Path | None = None) -> dict[str, Any]:
    proc = subprocess.run(
        args,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        return {
            "status": "na",
            "note": f"runner failed ({proc.returncode}): {err}",
            "ops_s": None,
            "peak_rss_bytes": None,
        }
    line = proc.stdout.strip().splitlines()[-1]
    return json.loads(line)


def run_gprx(case_path: Path, mode: str) -> dict[str, Any]:
    return run_cmd(
        [
            "cargo",
            "run",
            "--release",
            "--quiet",
            "--manifest-path",
            str(ROOT / "gprx" / "Cargo.toml"),
            "--bin",
            "gprx-sparse-online-perf",
            "--",
            str(case_path),
            mode,
        ],
        cwd=REPO,
    )


def run_gpytorch(case_path: Path) -> dict[str, Any]:
    return run_cmd([sys.executable, str(ROOT / "gpytorch_sparse_online.py"), str(case_path)], cwd=ROOT)


def ratio_verdict(ours: float | None, theirs: float | None, band: float) -> str:
    if ours is None or theirs is None:
        return "N/A"
    if theirs == 0.0:
        return "判定不能"
    rel = (ours - theirs) / theirs
    if abs(rel) <= band:
        return "判定不能"
    if ours < theirs:
        return "pass"
    return "fail"


def combine(*verdicts: str) -> str:
    judged = [v for v in verdicts if v not in ("判定不能", "N/A")]
    if any(v == "fail" for v in judged):
        return "fail"
    if judged and all(v == "pass" for v in judged) and "N/A" not in verdicts:
        if any(v == "判定不能" for v in verdicts):
            return "判定不能"
        return "pass"
    if any(v == "N/A" for v in verdicts):
        return "N/A"
    return "判定不能"


def judge(inc: dict[str, Any], other: dict[str, Any]) -> str:
    if inc.get("status") != "ok" or other.get("status") != "ok":
        return "N/A"
    return ratio_verdict(inc.get("ops_s"), other.get("ops_s"), BAND)


def fmt_s(value: float | None) -> str:
    if value is None:
        return "N/A"
    if value >= 1.0:
        return f"{value:.3f} s"
    return f"{value * 1000:.2f} ms"


def fmt_s_range(median: float | None, lo: float | None, hi: float | None) -> str:
    if median is None:
        return "N/A"
    if lo is None or hi is None:
        return fmt_s(median)
    if abs(hi - lo) <= max(median, 1e-12) * 1e-6:
        return fmt_s(median)
    return f"{fmt_s(median)} ({fmt_s(lo)}–{fmt_s(hi)})"


def fmt_rss(value: int | None) -> str:
    if value is None:
        return "N/A"
    return f"{value / (1024 * 1024):.1f} MiB"


def split_name(name: str) -> tuple[str, str]:
    problem = "forrester" if name.startswith("forrester") else "sphere"
    n = name.rsplit("n", 1)[-1]
    return problem, n


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(
        "P4-14 Sparse-online times: prefix untimed; 32 ops as one wall clock; "
        "discard PERF_WARMUP (default 1) then median of PERF_REPS "
        "(default 51/21/7 by n); RSS is one-process peak; CPU; raw y; "
        "no predict / NLML",
        flush=True,
    )
    only = {arg for arg in sys.argv[1:] if not arg.startswith("-")}
    OUT.mkdir(parents=True, exist_ok=True)
    if "--reprint" in sys.argv:
        path = OUT / "sparse_online_results.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        return emit_tables(rows)
    cases = write_sparse_online_cases(PROBLEMS)
    if only:
        cases = [p for p in cases if p.stem in only]
        if not cases:
            print(f"no cases match {sorted(only)}", file=sys.stderr)
            return 2
    rows: list[dict[str, Any]] = []
    runners = (
        ("gprx-inc", lambda p: run_gprx(p, "incremental")),
        ("gprx-full", lambda p: run_gprx(p, "full")),
        ("gpytorch", run_gpytorch),
    )
    for case_path in cases:
        print(f"# {case_path.name}", flush=True)
        for lib, fn in runners:
            print(f"  {lib}...", flush=True)
            row = fn(case_path)
            row["lib"] = lib
            row.setdefault("name", case_path.stem)
            rows.append(row)
            print(
                f"    {row.get('status')} ops="
                f"{fmt_s_range(row.get('ops_s'), row.get('ops_min_s'), row.get('ops_max_s'))}",
                flush=True,
            )

    (OUT / "sparse_online_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return emit_tables(rows)


def emit_tables(rows: list[dict[str, Any]]) -> int:
    by_name: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_name.setdefault(row["name"], {})[row["lib"]] = row

    print("\n## cells (Sparse online, CPU)\n")
    print("| 問題 | n | lib | ops 32 | peak RSS | vs inc |")
    print("| --- | --- | --- | --- | --- | --- |")
    verdicts: list[tuple[str, str, str]] = []
    for name, libs in by_name.items():
        inc = libs.get("gprx-inc", {})
        problem, n = split_name(name)
        for lib in LIBS:
            row = libs.get(lib, {})
            if lib == "gprx-inc":
                gate = "-"
            else:
                gate = judge(inc, row)
                verdicts.append((name, lib, gate))
            print(
                f"| {problem} | {n} | {lib} | "
                f"{fmt_s_range(row.get('ops_s'), row.get('ops_min_s'), row.get('ops_max_s'))} | "
                f"{fmt_rss(row.get('peak_rss_bytes'))} | {gate} |"
            )

    cell_verdicts: list[tuple[str, str]] = []
    for name, libs in by_name.items():
        inc = libs.get("gprx-inc", {})
        full_v = judge(inc, libs.get("gprx-full", {}))
        gpt_v = judge(inc, libs.get("gpytorch", {}))
        cell_verdicts.append((name, combine(full_v, gpt_v)))

    fails = [(n, lib, v) for n, lib, v in verdicts if v == "fail"]
    print("\n## gates\n")
    for name, v in cell_verdicts:
        print(f"- {name}: {v}")
    (OUT / "sparse_online_verdicts.json").write_text(
        json.dumps({"verdicts": verdicts, "cells": cell_verdicts, "fails": fails}, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
