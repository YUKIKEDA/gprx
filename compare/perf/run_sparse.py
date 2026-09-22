"""Run every P4-12 Sparse cell and print the bench-log table plus gate verdicts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from sparse_cases import write_sparse_cases

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
OUT = ROOT / "out"
PROBLEMS = OUT / "sparse_problems"
BAND = 0.05
LIBS = ("gprx", "gpytorch", "gpflow", "gpy")


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
            "factor_s": None,
            "eval_s": None,
            "predict_s": None,
            "joint_evals": None,
            "peak_rss_bytes": None,
        }
    line = proc.stdout.strip().splitlines()[-1]
    return json.loads(line)


def run_gprx(case_path: Path) -> dict[str, Any]:
    return run_cmd(
        [
            "cargo",
            "run",
            "--release",
            "--quiet",
            "--manifest-path",
            str(ROOT / "gprx" / "Cargo.toml"),
            "--bin",
            "gprx-sparse-perf",
            "--",
            str(case_path),
        ],
        cwd=REPO,
    )


def run_python(script: str, case_path: Path) -> dict[str, Any]:
    return run_cmd([sys.executable, str(ROOT / script), str(case_path)], cwd=ROOT)


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


def combine_times(*verdicts: str) -> str:
    judged = [v for v in verdicts if v not in ("判定不能", "N/A")]
    if any(v == "fail" for v in judged):
        return "fail"
    if judged and all(v == "pass" for v in judged):
        return "pass"
    if any(v == "N/A" for v in verdicts):
        return "N/A"
    return "判定不能"


def judge(gprx: dict[str, Any], other: dict[str, Any]) -> dict[str, str]:
    if gprx.get("status") != "ok" or other.get("status") != "ok":
        return {"time": "N/A", "rss": "N/A", "cell": "N/A"}
    factor_v = ratio_verdict(gprx.get("factor_s"), other.get("factor_s"), BAND)
    eval_v = ratio_verdict(gprx.get("eval_s"), other.get("eval_s"), BAND)
    predict_v = ratio_verdict(gprx.get("predict_s"), other.get("predict_s"), BAND)
    rss_v = ratio_verdict(gprx.get("peak_rss_bytes"), other.get("peak_rss_bytes"), BAND)
    time_note = combine_times(factor_v, eval_v, predict_v)
    if time_note == "pass" and rss_v == "pass":
        cell = "pass"
    elif "fail" in (time_note, rss_v):
        cell = "fail"
    elif "N/A" in (time_note, rss_v):
        cell = "N/A"
    else:
        cell = "判定不能"
    return {"time": time_note, "rss": rss_v, "cell": cell}


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


def fmt_evals(value: int | None) -> str:
    return "N/A" if value is None else str(value)


def split_name(name: str) -> tuple[str, str, str]:
    if name.startswith("sgpr_"):
        model = "sgpr"
        rest = name[len("sgpr_") :]
    elif name.startswith("svgp_"):
        model = "svgp"
        rest = name[len("svgp_") :]
    else:
        model = "?"
        rest = name
    problem = "forrester" if rest.startswith("forrester") else "sphere"
    n = rest.rsplit("n", 1)[-1]
    return model, problem, n


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(
        "P4-12 Sparse times: discard PERF_WARMUP (default 1) then median of "
        "PERF_REPS (default 51/21/7 by n); eval N is N × median of one joint "
        "call; RSS is one-process peak; CPU; y population-standardized",
        flush=True,
    )
    only = {arg for arg in sys.argv[1:] if not arg.startswith("-")}
    OUT.mkdir(parents=True, exist_ok=True)
    if "--reprint" in sys.argv:
        path = OUT / "sparse_results.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        return emit_tables(rows)
    cases = write_sparse_cases(PROBLEMS)
    if only:
        cases = [p for p in cases if p.stem in only]
        if not cases:
            print(f"no cases match {sorted(only)}", file=sys.stderr)
            return 2
    rows: list[dict[str, Any]] = []
    runners = (
        ("gprx", run_gprx),
        ("gpytorch", lambda p: run_python("gpytorch_sparse.py", p)),
        ("gpflow", lambda p: run_python("gpflow_sparse.py", p)),
        ("gpy", lambda p: run_python("gpy_sparse.py", p)),
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
                f"    {row.get('status')} factor="
                f"{fmt_s_range(row.get('factor_s'), row.get('factor_min_s'), row.get('factor_max_s'))} "
                f"eval="
                f"{fmt_s_range(row.get('eval_s'), row.get('eval_min_s'), row.get('eval_max_s'))}",
                flush=True,
            )

    (OUT / "sparse_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return emit_tables(rows)


def emit_tables(rows: list[dict[str, Any]]) -> int:
    by_name: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_name.setdefault(row["name"], {})[row["lib"]] = row

    print("\n## cells (Sparse, CPU)\n")
    print(
        "| 面 | 問題 | n | lib | factor | eval N | evals | predict 100 | peak RSS | vs gprx |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    verdicts: list[tuple[str, str, str]] = []
    for name, libs in by_name.items():
        gprx = libs.get("gprx", {})
        model, problem, n = split_name(name)
        for lib in LIBS:
            row = libs.get(lib, {})
            if lib == "gprx":
                gate = "-"
            else:
                judged = judge(gprx, row)
                gate = judged["cell"]
                verdicts.append((name, lib, gate))
            print(
                f"| {model} | {problem} | {n} | {lib} | "
                f"{fmt_s_range(row.get('factor_s'), row.get('factor_min_s'), row.get('factor_max_s'))} | "
                f"{fmt_s_range(row.get('eval_s'), row.get('eval_min_s'), row.get('eval_max_s'))} | "
                f"{fmt_evals(row.get('joint_evals'))} | "
                f"{fmt_s_range(row.get('predict_s'), row.get('predict_min_s'), row.get('predict_max_s'))} | "
                f"{fmt_rss(row.get('peak_rss_bytes'))} | {gate} |"
            )

    fails = [(n, lib, v) for n, lib, v in verdicts if v == "fail"]
    print("\n## gates\n")
    for name, lib, v in verdicts:
        print(f"- {name} vs {lib}: {v}")
    (OUT / "sparse_verdicts.json").write_text(
        json.dumps({"verdicts": verdicts, "fails": fails}, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
