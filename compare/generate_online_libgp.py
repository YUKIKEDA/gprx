"""Write libgp online-insert goldens (P3-6). cargo test must not run this."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from common.problems import make_forrester, make_sphere
from perf.runners import PERF, ensure_libgp

ROOT = Path(__file__).resolve().parent
GOLDENS = ROOT / "goldens"


def prefix_colmajor(values: list[float], n: int, d: int, keep: int) -> list[float]:
    out: list[float] = []
    for feature in range(d):
        base = feature * n
        out.extend(values[base : base + keep])
    return out


def prefix_case(full: dict, keep: int, xs: list[float], xs_cols: int, name: str) -> dict:
    n = int(full["n_rows"])
    d = int(full["n_cols"])
    return {
        "name": name,
        "problem": full["problem"],
        "ard": full["ard"],
        "n_rows": keep,
        "n_cols": d,
        "x": prefix_colmajor(full["x"], n, d, keep),
        "y": full["y"][:keep],
        "xs_n_rows": 1,
        "xs_n_cols": xs_cols,
        "xs": xs,
        "lengthscales_init": full["lengthscales_init"],
        "noise_variance_init": full["noise_variance_init"],
        "joint_evals": 1,
        "start_n": 4,
    }


def write_golden(case: dict, dest: Path, exe: Path) -> None:
    GOLDENS.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".case.json")
    tmp.write_text(json.dumps(case), encoding="utf-8")
    proc = subprocess.run(
        [str(exe), "golden", str(tmp)],
        cwd=PERF,
        check=False,
        capture_output=True,
        text=True,
    )
    tmp.unlink(missing_ok=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"libgp-online golden failed ({proc.returncode}): "
            f"{(proc.stderr or proc.stdout or '').strip()}"
        )
    dest.write_text(
        json.dumps(json.loads(proc.stdout), indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    exe = ensure_libgp("libgp-online")
    if isinstance(exe, dict):
        raise RuntimeError(exe.get("note", "libgp-online build failed"))
    cases = [
        (
            prefix_case(make_forrester(256), 32, [0.5], 1, "online_libgp_forrester"),
            GOLDENS / "online_libgp_forrester.json",
        ),
        (
            prefix_case(
                make_sphere(16), 32, [0.25, 0.75], 2, "online_libgp_sphere"
            ),
            GOLDENS / "online_libgp_sphere.json",
        ),
    ]
    for case, dest in cases:
        write_golden(case, dest, exe)
        print(f"wrote {dest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
