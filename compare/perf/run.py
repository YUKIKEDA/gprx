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
            "fit_s": None,
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
            "--",
            str(case_path),
        ],
        cwd=REPO,
    )


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


def ratio_verdict(ours: float, theirs: float, band: float) -> str:
    if theirs == 0.0:
        return "判定不能"
    rel = (ours - theirs) / theirs
    if abs(rel) <= band:
        return "判定不能"
    if ours < theirs:
        return "pass"
    return "fail"


def judge_sklearn(gprx: dict[str, Any], other: dict[str, Any]) -> dict[str, str]:
    if gprx.get("status") != "ok" or other.get("status") != "ok":
        return {"time": "N/A", "rss": "N/A", "cell": "N/A"}
    time_v = ratio_verdict(gprx["fit_s"], other["fit_s"], SKLEARN_BAND)
    rss_v = ratio_verdict(gprx["peak_rss_bytes"], other["peak_rss_bytes"], SKLEARN_BAND)
    evals_match = gprx.get("joint_evals") == other.get("joint_evals")
    if not evals_match:
        time_note = "判定不能"
    else:
        time_note = time_v
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
    time_v = within_band_or_better(gprx["fit_s"], other["fit_s"], LIBGP_BAND)
    rss_v = within_band_or_better(
        gprx["peak_rss_bytes"], other["peak_rss_bytes"], LIBGP_BAND
    )
    evals_match = gprx.get("joint_evals") == other.get("joint_evals")
    if not evals_match:
        time_v = "判定不能"
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
    time_v = ratio_verdict(gprx["fit_s"], other["fit_s"], SKLEARN_BAND)
    rss_v = ratio_verdict(gprx["peak_rss_bytes"], other["peak_rss_bytes"], SKLEARN_BAND)
    evals_match = (
        gprx.get("joint_evals") is not None
        and other.get("joint_evals") is not None
        and gprx.get("joint_evals") == other.get("joint_evals")
    )
    if not evals_match:
        time_speed = "判定不能"
    else:
        time_speed = time_v
    if time_v == "pass" or rss_v == "pass":
        cell = "pass"
    elif time_v == "fail" and rss_v == "fail":
        cell = "fail"
    else:
        cell = "判定不能"
    return {"time": time_speed, "rss": rss_v, "cell": cell}


def fmt_s(value: float | None) -> str:
    if value is None:
        return "N/A"
    if value >= 1.0:
        return f"{value:.3f} s"
    return f"{value * 1000:.2f} ms"


def fmt_rss(value: int | None) -> str:
    if value is None:
        return "N/A"
    return f"{value / (1024 * 1024):.1f} MiB"


def fmt_evals(value: int | None) -> str:
    return "N/A" if value is None else str(value)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    only = {arg for arg in sys.argv[1:] if not arg.startswith("-")}
    OUT.mkdir(parents=True, exist_ok=True)
    cases = write_cases(PROBLEMS)
    if only:
        cases = [p for p in cases if p.stem in only]
        if not cases:
            print(f"no cases match {sorted(only)}", file=sys.stderr)
            return 2
    rows: list[dict[str, Any]] = []
    runners = (
        ("gprx", run_gprx),
        ("sklearn", lambda p: run_python("sklearn_run.py", p)),
        ("libgp", lambda p: run_python("libgp_run.py", p)),
        ("friedrich", run_friedrich),
    )
    for case_path in cases:
        print(f"# {case_path.name}", flush=True)
        for lib, fn in runners:
            print(f"  {lib}...", flush=True)
            row = fn(case_path)
            row.setdefault("lib", lib)
            row.setdefault("name", case_path.stem)
            rows.append(row)
            print(f"    {row.get('status')} fit={fmt_s(row.get('fit_s'))}", flush=True)

    (OUT / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    by_name: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_name.setdefault(row["name"], {})[row["lib"]] = row

    print("\n## cells\n")
    print(
        "| 問題 | n | lib | fit | evals | predict 100 | peak RSS | vs gprx |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
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
                f"| {problem} | {n} | {lib} | {fmt_s(row.get('fit_s'))} | "
                f"{fmt_evals(row.get('joint_evals'))} | {fmt_s(row.get('predict_s'))} | "
                f"{fmt_rss(row.get('peak_rss_bytes'))} | {gate} |"
            )

    fails = [(n, lib, v) for n, lib, v in verdicts if v == "fail"]
    print("\n## gates\n")
    for name, lib, v in verdicts:
        print(f"- {name} vs {lib}: {v}")
    (OUT / "verdicts.json").write_text(
        json.dumps({"verdicts": verdicts, "fails": fails}, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
