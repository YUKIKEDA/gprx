"""P3-6: time sequential insert vs libgp add_pattern."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from problems import FORRESTER_NS, SPHERE_SIDES, make_forrester, make_sphere
from run import (
    ROOT,
    ensure_libgp_online,
    fmt_rss,
    fmt_s_range,
    run_cmd,
    within_band_or_better,
)

OUT = ROOT / "out"
PROBLEMS = OUT / "problems"
ONLINE_BAND = 0.05


def online_cases() -> list[Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    PROBLEMS.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for n in FORRESTER_NS:
        case = make_forrester(n)
        path = PROBLEMS / f"online_{case['name']}.json"
        path.write_text(json.dumps(case), encoding="utf-8")
        paths.append(path)
    for side in SPHERE_SIDES:
        case = make_sphere(side)
        path = PROBLEMS / f"online_{case['name']}.json"
        path.write_text(json.dumps(case), encoding="utf-8")
        paths.append(path)
    return paths


def run_gprx_online(
    case_path: Path, *, stages: bool = False, delete: bool = False
) -> dict[str, Any]:
    cmd = [
        "cargo",
        "run",
        "--release",
        "--quiet",
        "--manifest-path",
        str(ROOT / "gprx" / "Cargo.toml"),
    ]
    if stages:
        cmd.extend(["--features", "insert-stages"])
    cmd.extend(["--", str(case_path)])
    if delete:
        cmd.append("--delete")
    else:
        cmd.append("--online")
        if stages:
            cmd.append("--stages")
    row = run_cmd(cmd, cwd=ROOT.parent.parent)
    row["lib"] = "gprx"
    return row


def run_libgp_online(case_path: Path) -> dict[str, Any]:
    exe = ensure_libgp_online()
    if isinstance(exe, dict):
        exe["lib"] = "libgp"
        return exe
    row = run_cmd([str(exe), "time", str(case_path)], cwd=ROOT)
    row["lib"] = "libgp"
    return row


def judge_insert(gprx: dict[str, Any], other: dict[str, Any]) -> str:
    if gprx.get("status") != "ok" or other.get("status") != "ok":
        return "N/A"
    if gprx.get("factor_s") is None or other.get("factor_s") is None:
        return "N/A"
    return within_band_or_better(gprx["factor_s"], other["factor_s"], ONLINE_BAND)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    stages = "--stages" in sys.argv[1:]
    delete = "--delete" in sys.argv[1:]
    print(
        "online "
        + ("delete" if delete else "insert")
        + ": discard PERF_WARMUP then median of PERF_REPS; "
        + (
            "clock is delete from n to 2 (last remaining PointId; insert untimed); "
            if delete
            else "clock is add from n=2 to n (first two points untimed); "
        )
        + (
            "stages are kernel / bordered LDLT / X·y (gprx only)"
            if stages
            else (
                "gprx only; no libgp"
                if delete
                else "gate is 256/1024 median <= libgp (5% inconclusive); 4096 record"
            )
        ),
        flush=True,
    )
    only = {arg for arg in sys.argv[1:] if not arg.startswith("-")}
    cases = online_cases()
    if stages and not only:
        only = {"forrester_n256", "forrester_n1024"}
    if only:
        cases = [p for p in cases if p.stem in only or p.stem.removeprefix("online_") in only]
        if not cases:
            print(f"no cases match {sorted(only)}", file=sys.stderr)
            return 2
    rows: list[dict[str, Any]] = []
    libs = (("gprx", lambda path: run_gprx_online(path, stages=stages, delete=delete)),)
    if not stages and not delete:
        libs = libs + (("libgp", run_libgp_online),)
    for case_path in cases:
        print(f"# {case_path.name}", flush=True)
        for lib, fn in libs:
            print(f"  {lib}...", flush=True)
            row = fn(case_path)
            row.setdefault("name", case_path.stem)
            rows.append(row)
            extra = ""
            if stages:
                extra = (
                    f" kernel={fmt_s_range(row.get('kernel_s'), None, None)}"
                    f" border={fmt_s_range(row.get('border_s'), None, None)}"
                    f" rest={fmt_s_range(row.get('rest_s'), None, None)}"
                )
            label = "delete" if delete else "insert"
            print(
                f"    {row.get('status')} {label}="
                f"{fmt_s_range(row.get('factor_s'), row.get('factor_min_s'), row.get('factor_max_s'))}"
                f"{extra}",
                flush=True,
            )

    out_name = "online_delete_results.json" if delete else "online_results.json"
    (OUT / out_name).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    by_name: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_name.setdefault(row["name"], {})[row["lib"]] = row

    if delete:
        print("\n## online delete n→2 (gprx)\n")
        print("| 問題 | n | delete n→2 | peak RSS |")
        print("| --- | --- | --- | --- |")
        for name, libs in by_name.items():
            row = libs.get("gprx", {})
            stem = name.removeprefix("online_")
            n = stem.rsplit("n", 1)[-1]
            problem = "forrester" if "forrester" in stem else "sphere"
            print(
                f"| {problem} | {n} | "
                f"{fmt_s_range(row.get('factor_s'), row.get('factor_min_s'), row.get('factor_max_s'))} | "
                f"{fmt_rss(row.get('peak_rss_bytes'))} |"
            )
        return 0

    if stages:
        print("\n## online insert stages (gprx)\n")
        print("| 問題 | n | insert | kernel | border | rest |")
        print("| --- | --- | --- | --- | --- | --- |")
        for name, libs in by_name.items():
            row = libs.get("gprx", {})
            stem = name.removeprefix("online_")
            n = stem.rsplit("n", 1)[-1]
            problem = "forrester" if "forrester" in stem else "sphere"
            print(
                f"| {problem} | {n} | "
                f"{fmt_s_range(row.get('factor_s'), row.get('factor_min_s'), row.get('factor_max_s'))} | "
                f"{fmt_s_range(row.get('kernel_s'), None, None)} | "
                f"{fmt_s_range(row.get('border_s'), None, None)} | "
                f"{fmt_s_range(row.get('rest_s'), None, None)} |"
            )
        return 0

    print("\n## online insert (raw y)\n")
    print("| 問題 | n | lib | insert n=2→n | peak RSS | ゲート |")
    print("| --- | --- | --- | --- | --- | --- |")
    verdicts: list[tuple[str, str]] = []
    for name, libs in by_name.items():
        gprx = libs.get("gprx", {})
        libgp = libs.get("libgp", {})
        stem = name.removeprefix("online_")
        n = stem.rsplit("n", 1)[-1]
        problem = "forrester" if "forrester" in stem else "sphere"
        gated = n in {"256", "1024"}
        gate = judge_insert(gprx, libgp) if gated else "record"
        if gated:
            verdicts.append((name, gate))
        for lib, row in (("gprx", gprx), ("libgp", libgp)):
            cell_gate = gate if lib == "gprx" else "-"
            print(
                f"| {problem} | {n} | {lib} | "
                f"{fmt_s_range(row.get('factor_s'), row.get('factor_min_s'), row.get('factor_max_s'))} | "
                f"{fmt_rss(row.get('peak_rss_bytes'))} | {cell_gate} |"
            )

    print("\n## gates (256 / 1024)\n")
    for name, v in verdicts:
        print(f"- {name}: {v}")
    fails = [(n, v) for n, v in verdicts if v == "fail"]
    (OUT / "online_verdicts.json").write_text(
        json.dumps({"verdicts": verdicts, "fails": fails}, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
