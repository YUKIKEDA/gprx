"""How the harnesses start each library's runner on one JSON case."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from common.harness import TIME_FIELDS, Row, na_row, run_cmd

PERF = Path(__file__).resolve().parent
COMPARE = PERF.parent
REPO = COMPARE.parent
OUT = PERF / "out"


def run_gprx(mode: str, case_path: Path, *flags: str, features: str | None = None) -> Row:
    """The ``gprx-perf`` runner: ``gprx-perf <mode> <case> [flags]``."""
    args = [
        "cargo",
        "run",
        "--release",
        "--quiet",
        "--manifest-path",
        str(PERF / "gprx" / "Cargo.toml"),
    ]
    if features is not None:
        args.extend(["--features", features])
    args.extend(["--", mode, str(case_path), *flags])
    fields = ("ops_s", "peak_rss_bytes") if mode == "sparse-online" else TIME_FIELDS
    return run_cmd(args, cwd=REPO, fields=fields)


def run_friedrich(case_path: Path) -> Row:
    return run_cmd(
        [
            "cargo",
            "run",
            "--release",
            "--quiet",
            "--manifest-path",
            str(PERF / "friedrich" / "Cargo.toml"),
            "--",
            str(case_path),
        ],
        cwd=REPO,
    )


def run_python(module: str, case_path: Path, fields: tuple[str, ...] = TIME_FIELDS) -> Row:
    """A Python runner, ``python -m perf.<module> <case>`` from ``compare/``."""
    return run_cmd(
        [sys.executable, "-X", "utf8", "-m", f"perf.{module}", str(case_path)],
        cwd=COMPARE,
        fields=fields,
    )


def _libgp_candidates(build: Path, name: str) -> list[Path]:
    return [
        build / "Release" / f"{name}.exe",
        build / "RelWithDebInfo" / f"{name}.exe",
        build / f"{name}.exe",
        build / name,
    ]


def _cmake(args: list[str], what: str) -> Row | None:
    proc = subprocess.run(
        ["cmake", *args], cwd=PERF, check=False, capture_output=True, text=True
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        return na_row(f"{what} failed ({proc.returncode}): {err}")
    return None


def ensure_libgp(target: str) -> Path | Row:
    """The native libgp runner ``target`` (``libgp-perf`` or
    ``libgp-online``), configured and built with CMake when missing."""
    src = PERF / "libgp"
    build = src / "build"
    if target == "libgp-perf":
        for candidate in _libgp_candidates(build, target):
            if candidate.is_file():
                return candidate
    configure = ["-S", str(src), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"]
    failed = _cmake(configure, "libgp cmake configure")
    if failed is not None:
        return failed
    build_args = ["--build", str(build), "--config", "Release"]
    if target != "libgp-perf":
        build_args.extend(["--target", target])
    failed = _cmake(build_args, f"{'libgp' if target == 'libgp-perf' else target} cmake build")
    if failed is not None:
        return failed
    for candidate in _libgp_candidates(build, target):
        if candidate.is_file():
            return candidate
    return na_row(f"{target} binary not found after cmake --build")


def run_libgp(target: str, case_path: Path, *args: str) -> Row:
    """The native libgp runner ``target`` on one case: ``<exe> [args] <case>``."""
    exe = ensure_libgp(target)
    if isinstance(exe, dict):
        return exe
    return run_cmd([str(exe), *args, str(case_path)], cwd=PERF)
