"""Run every P2B-16 cell and print the bench-log table plus gate verdicts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from problems import write_cases

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
OUT = ROOT / "out"
PROBLEMS = OUT / "problems"

SKLEARN_BAND = 0.05
LIBGP_BAND = 0.10


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


def run_gprx(case_path: Path, memory: bool = False) -> dict[str, Any]:
    args = [
        "cargo",
        "run",
        "--release",
        "--quiet",
        "--manifest-path",
        str(ROOT / "gprx" / "Cargo.toml"),
        "--",
        str(case_path),
    ]
    if memory:
        args.append("--memory")
    return run_cmd(args, cwd=REPO)


def run_friedrich(case_path: Path) -> dict[str, Any]:
    return run_cmd(
        [
            "cargo",
            "run",
            "--release",
            "--quiet",
            "--manifest-path",
            str(ROOT / "friedrich" / "Cargo.toml"),
            "--",
            str(case_path),
        ],
        cwd=REPO,
    )


def run_python(script: str, case_path: Path) -> dict[str, Any]:
    return run_cmd([sys.executable, str(ROOT / script), str(case_path)], cwd=ROOT)


def libgp_exe_candidates(build: Path) -> list[Path]:
    return [
        build / "Release" / "libgp-perf.exe",
        build / "RelWithDebInfo" / "libgp-perf.exe",
        build / "libgp-perf.exe",
        build / "libgp-perf",
    ]


def libgp_online_candidates(build: Path) -> list[Path]:
    return [
        build / "Release" / "libgp-online.exe",
        build / "RelWithDebInfo" / "libgp-online.exe",
        build / "libgp-online.exe",
        build / "libgp-online",
    ]


def ensure_libgp() -> Path | dict[str, Any]:
    src = ROOT / "libgp"
    build = src / "build"
    for candidate in libgp_exe_candidates(build):
        if candidate.is_file():
            return candidate
    configure = subprocess.run(
        [
            "cmake",
            "-S",
            str(src),
            "-B",
            str(build),
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if configure.returncode != 0:
        err = (configure.stderr or configure.stdout or "").strip()
        return {
            "status": "na",
            "note": f"libgp cmake configure failed ({configure.returncode}): {err}",
            "factor_s": None,
            "eval_s": None,
            "predict_s": None,
            "joint_evals": None,
            "peak_rss_bytes": None,
        }
    built = subprocess.run(
        ["cmake", "--build", str(build), "--config", "Release"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if built.returncode != 0:
        err = (built.stderr or built.stdout or "").strip()
        return {
            "status": "na",
            "note": f"libgp cmake build failed ({built.returncode}): {err}",
            "factor_s": None,
            "eval_s": None,
            "predict_s": None,
            "joint_evals": None,
            "peak_rss_bytes": None,
        }
    for candidate in libgp_exe_candidates(build):
        if candidate.is_file():
            return candidate
    return {
        "status": "na",
        "note": "libgp-perf binary not found after cmake --build",
        "factor_s": None,
        "eval_s": None,
        "predict_s": None,
        "joint_evals": None,
        "peak_rss_bytes": None,
    }


def run_libgp(case_path: Path) -> dict[str, Any]:
    exe = ensure_libgp()
    if isinstance(exe, dict):
        return exe
    return run_cmd([str(exe), str(case_path)], cwd=ROOT)


def ensure_libgp_online() -> Path | dict[str, Any]:
    src = ROOT / "libgp"
    build = src / "build"
    configure = subprocess.run(
        [
            "cmake",
            "-S",
            str(src),
            "-B",
            str(build),
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if configure.returncode != 0:
        err = (configure.stderr or configure.stdout or "").strip()
        return {
            "status": "na",
            "note": f"libgp cmake configure failed ({configure.returncode}): {err}",
            "factor_s": None,
            "eval_s": None,
            "predict_s": None,
            "joint_evals": None,
            "peak_rss_bytes": None,
        }
    built = subprocess.run(
        ["cmake", "--build", str(build), "--config", "Release", "--target", "libgp-online"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if built.returncode != 0:
        err = (built.stderr or built.stdout or "").strip()
        return {
            "status": "na",
            "note": f"libgp-online cmake build failed ({built.returncode}): {err}",
            "factor_s": None,
            "eval_s": None,
            "predict_s": None,
            "joint_evals": None,
            "peak_rss_bytes": None,
        }
    for candidate in libgp_online_candidates(build):
        if candidate.is_file():
            return candidate
    return {
        "status": "na",
        "note": "libgp-online binary not found after cmake --build",
        "factor_s": None,
        "eval_s": None,
        "predict_s": None,
        "joint_evals": None,
        "peak_rss_bytes": None,
    }


def ratio_verdict(ours: float, theirs: float, band: float) -> str:
    if theirs == 0.0:
        return "判定不能"
    rel = (ours - theirs) / theirs
    if abs(rel) <= band:
        return "判定不能"
    if ours < theirs:
        return "pass"
    return "fail"


def combine_times(*verdicts: str) -> str:
    judged = [v for v in verdicts if v != "判定不能"]
    if any(v == "fail" for v in judged):
        return "fail"
    if judged and all(v == "pass" for v in judged):
        return "pass"
    return "判定不能"


def judge_sklearn(gprx: dict[str, Any], other: dict[str, Any]) -> dict[str, str]:
    if gprx.get("status") != "ok" or other.get("status") != "ok":
        return {"time": "N/A", "rss": "N/A", "cell": "N/A"}
    factor_v = ratio_verdict(gprx["factor_s"], other["factor_s"], SKLEARN_BAND)
    eval_v = ratio_verdict(gprx["eval_s"], other["eval_s"], SKLEARN_BAND)
    rss_v = ratio_verdict(gprx["peak_rss_bytes"], other["peak_rss_bytes"], SKLEARN_BAND)
    time_note = combine_times(factor_v, eval_v)
    if time_note == "pass" and rss_v == "pass":
        cell = "pass"
    elif "fail" in (time_note, rss_v):
        cell = "fail"
    else:
        cell = "判定不能"
    return {"time": time_note, "rss": rss_v, "cell": cell}


def within_band_or_better(ours: float, theirs: float, band: float) -> str:
    if theirs == 0.0:
        return "判定不能"
    rel = (ours - theirs) / theirs
    if rel > band:
        return "fail"
    return "pass"


def judge_libgp(gprx: dict[str, Any], other: dict[str, Any]) -> dict[str, str]:
    if gprx.get("status") != "ok" or other.get("status") != "ok":
        return {"time": "N/A", "rss": "N/A", "cell": "N/A"}
    factor_v = within_band_or_better(gprx["factor_s"], other["factor_s"], LIBGP_BAND)
    eval_v = within_band_or_better(gprx["eval_s"], other["eval_s"], LIBGP_BAND)
    rss_v = within_band_or_better(
        gprx["peak_rss_bytes"], other["peak_rss_bytes"], LIBGP_BAND
    )
    time_v = combine_times(factor_v, eval_v)
    judged = [v for v in (time_v, rss_v) if v != "判定不能"]
    if any(v == "fail" for v in judged):
        cell = "fail"
    elif judged and all(v == "pass" for v in judged):
        cell = "pass"
    else:
        cell = "判定不能"
    return {"time": time_v, "rss": rss_v, "cell": cell}


def judge_friedrich(gprx: dict[str, Any], other: dict[str, Any]) -> dict[str, str]:
    if gprx.get("status") != "ok" or other.get("status") != "ok":
        return {"time": "N/A", "rss": "N/A", "cell": "N/A"}
    time_v = ratio_verdict(gprx["factor_s"], other["factor_s"], SKLEARN_BAND)
    rss_v = ratio_verdict(gprx["peak_rss_bytes"], other["peak_rss_bytes"], SKLEARN_BAND)
    if time_v == "pass" or rss_v == "pass":
        cell = "pass"
    elif time_v == "fail" and rss_v == "fail":
        cell = "fail"
    else:
        cell = "判定不能"
    return {"time": time_v, "rss": rss_v, "cell": cell}


def fmt_s(value: float | None) -> str:
    if value is None:
        return "N/A"
    if value >= 1.0:
        return f"{value:.3f} s"
    return f"{value * 1000:.2f} ms"


def fmt_s_range(
    median: float | None, lo: float | None, hi: float | None
) -> str:
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


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(
        "times: discard PERF_WARMUP (default 1) then median of PERF_REPS "
        "(default 51/21/7 by n); eval N is N × median of one joint call; "
        "RSS is one-process peak",
        flush=True,
    )
    only = {arg for arg in sys.argv[1:] if not arg.startswith("-")}
    OUT.mkdir(parents=True, exist_ok=True)
    if "--reprint" in sys.argv:
        path = OUT / "results.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        return emit_tables(rows)
    cases = write_cases(PROBLEMS)
    if only:
        cases = [p for p in cases if p.stem in only]
        if not cases:
            print(f"no cases match {sorted(only)}", file=sys.stderr)
            return 2
    rows: list[dict[str, Any]] = []
    runners = (
        ("gprx", run_gprx),
        ("gprx-memory", lambda p: run_gprx(p, memory=True)),
        ("sklearn", lambda p: run_python("sklearn_run.py", p)),
        ("libgp", run_libgp),
        ("friedrich", run_friedrich),
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

    (OUT / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return emit_tables(rows)


def emit_tables(rows: list[dict[str, Any]]) -> int:
    by_name: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_name.setdefault(row["name"], {})[row["lib"]] = row

    print("\n## cells (CachedDistances, default)\n")
    print(
        "| 問題 | n | lib | factor | eval N | evals | predict 100 | peak RSS | vs gprx |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    verdicts: list[tuple[str, str, str]] = []
    for name, libs in by_name.items():
        gprx = libs.get("gprx", {})
        n = name.rsplit("n", 1)[-1]
        problem = "forrester" if name.startswith("forrester") else "sphere"
        for lib in ("gprx", "sklearn", "libgp", "friedrich"):
            row = libs.get(lib, {})
            if lib == "gprx":
                gate = "-"
            elif lib == "sklearn":
                judged = judge_sklearn(gprx, row)
                gate = judged["cell"]
                verdicts.append((name, lib, gate))
            elif lib == "libgp":
                judged = judge_libgp(gprx, row)
                gate = judged["cell"]
                verdicts.append((name, lib, gate))
            else:
                judged = judge_friedrich(gprx, row)
                gate = judged["cell"]
                verdicts.append((name, lib, gate))
            print(
                f"| {problem} | {n} | {lib} | "
                f"{fmt_s_range(row.get('factor_s'), row.get('factor_min_s'), row.get('factor_max_s'))} | "
                f"{fmt_s_range(row.get('eval_s'), row.get('eval_min_s'), row.get('eval_max_s'))} | "
                f"{fmt_evals(row.get('joint_evals'))} | "
                f"{fmt_s_range(row.get('predict_s'), row.get('predict_min_s'), row.get('predict_max_s'))} | "
                f"{fmt_rss(row.get('peak_rss_bytes'))} | {gate} |"
            )

    print("\n## cells (with_prefer_memory = UncachedDistances + ReuseCholesky)\n")
    print(
        "| 問題 | n | lib | factor | eval N | evals | predict 100 | peak RSS | vs libgp RSS |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for name, libs in by_name.items():
        memory = libs.get("gprx-memory", {})
        libgp = libs.get("libgp", {})
        n = name.rsplit("n", 1)[-1]
        problem = "forrester" if name.startswith("forrester") else "sphere"
        for lib, row in (
            ("gprx-memory", memory),
            ("libgp", libgp),
        ):
            print(
                f"| {problem} | {n} | {lib} | "
                f"{fmt_s_range(row.get('factor_s'), row.get('factor_min_s'), row.get('factor_max_s'))} | "
                f"{fmt_s_range(row.get('eval_s'), row.get('eval_min_s'), row.get('eval_max_s'))} | "
                f"{fmt_evals(row.get('joint_evals'))} | "
                f"{fmt_s_range(row.get('predict_s'), row.get('predict_min_s'), row.get('predict_max_s'))} | "
                f"{fmt_rss(row.get('peak_rss_bytes'))} | record |"
            )

    fails = [(n, lib, v) for n, lib, v in verdicts if v == "fail"]
    print("\n## gates\n")
    for name, lib, v in verdicts:
        print(f"- {name} vs {lib}: {v}")
    print("\n## P2B-23 memory pole (record only; no new RSS gate)\n")
    print("- table 2 is with_prefer_memory vs libgp; P2B-21 Uncached+Retain stays in bench-log")
    (OUT / "verdicts.json").write_text(
        json.dumps({"verdicts": verdicts, "fails": fails}, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
