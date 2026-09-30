"""Every path of the harness on tiny problems: the matrix of
(dataset kind × model × protocol × library), each cell run for real.

A path that cannot run must say why with a reason this file knows
(``EXPECTED_NA``); anything else, and any ok row with a hole in it, is a FAIL.
Run it after touching a runner, and before a long measurement:

```text
python -m perf.real.smoke [--data]
```

``--data`` also loads two splits of every dataset (downloads them once).
It never writes results and never pins a checksum.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import tempfile
import zipfile
from pathlib import Path

os.environ["PERF_SMOKE"] = "1"  # no pinning while a stand-in archive is in use
os.environ.setdefault("PERF_WARMUP", "0")

import numpy as np  # noqa: E402

from common import harness  # noqa: E402

from .cases import CASES, write_case, write_curve_case  # noqa: E402
from .data import DATASETS, load_split, n_splits  # noqa: E402
from .libs import runners_for  # noqa: E402

TINY_N, TINY_TEST, TINY_M = 60, 20, 8

#: The only reasons a cell may be N/A.
EXPECTED_NA = (
    r"friedrich has no ARD kernel",
    r"friedrich has no sparse GP",
    r"libgp has no swappable optimizer",
    r"libgp has no sparse GP",
    r"scikit-learn has no sparse GP",
    r"GPy's SVGP has no minibatch fit",
    r"the composite Mauna Loa kernel is not wired",
    r"SVGP has no native protocol",
    r"SVGP has no fixed protocol",
)
#: Fields an ok row must fill.
REQUIRED = ("fit_s", "predict_s", "peak_rss_bytes", "joint_evals")


def _shrink(path: Path) -> Path:
    """The case cut to a few training points, test points and inducing points."""
    case = json.loads(path.read_text(encoding="utf-8"))
    n, d = case["n_rows"], case["n_cols"]
    k = min(TINY_N, n)
    case["x"] = [case["x"][j * n + i] for j in range(d) for i in range(k)]
    case["y"] = case["y"][:k]
    case["n_rows"] = k
    if "n_test" not in case:  # a dataset case: keep a few test points
        m, keep = case["xs_n_rows"], min(TINY_TEST, case["xs_n_rows"])
        case["xs"] = [case["xs"][j * m + i] for j in range(d) for i in range(keep)]
        case["ys"] = case["ys"][:keep]
        case["xs_n_rows"] = keep
    if case.get("z"):
        m = case["n_inducing"]
        keep = min(TINY_M, m)
        case["z"] = [case["z"][j * m + i] for j in range(d) for i in range(keep)]
        case["n_inducing"] = keep
    tiny = path.with_name("smoke_" + path.name)
    tiny.write_text(json.dumps(case), encoding="utf-8")
    return tiny


def _stand_in_snelson() -> str:
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0, 6, 60))
    y = np.sin(x) * np.cos(2 * x) + 0.1 * rng.normal(size=60)
    path = Path(tempfile.mkdtemp()) / "SPGP_dist.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("train_inputs", "\n".join(map(str, x)))
        archive.writestr("train_outputs", "\n".join(map(str, y)))
        archive.writestr("test_inputs", "\n".join(map(str, np.linspace(-0.5, 6.5, 40))))
    return str(path)


def judge(row: dict, protocol: str) -> tuple[str, str]:
    """(``ok`` | ``na`` | ``FAIL``, why)."""
    if row.get("status") == "ok":
        holes = [f for f in REQUIRED if row.get(f) is None]
        if holes:
            return "FAIL", f"ok row without {holes}"
        for key in ("rmse", "nlpd", "nlml"):
            value = row.get(key)
            if value is not None and not math.isfinite(value):
                return "FAIL", f"{key} = {value}"
        return "ok", ""
    note = str(row.get("note", ""))
    if any(re.search(p, note) for p in EXPECTED_NA):
        return "na", note
    return "FAIL", note[:300]


def cells() -> list[tuple[str, str, str]]:
    """(dataset, model, protocol) of every path."""
    out = []
    for dataset in ("yacht", "snelson", "maunaloa"):
        for protocol in ("native", "matched", "fixed"):
            out.append((dataset, "exact", protocol))
    for model in ("sgpr", "svgp"):
        for protocol in ("native", "matched", "fixed"):
            out.append(("yacht", model, protocol))
    return out


def run_matrix(timeline: bool) -> int:
    if not os.environ.get("SNELSON_ZIP"):
        os.environ["SNELSON_ZIP"] = _stand_in_snelson()
    failures = 0
    for dataset, model, protocol in cells():
        if dataset in ("snelson", "maunaloa"):
            path = _shrink_curve(write_curve_case(dataset, protocol))
        else:
            path = _shrink(write_case(dataset, 0, protocol, model, TINY_M))
        runners = runners_for(model)
        for lib, run in runners.items():
            harness.TIMELINE = (Path(tempfile.mkdtemp()), "smoke") if timeline else None
            row = run(path)
            verdict, why = judge(row, protocol)
            failures += verdict == "FAIL"
            print(f"{verdict:4} {dataset:9} {model:6} {protocol:8} {lib:10} {why}", flush=True)
    harness.TIMELINE = None
    return failures


def _shrink_curve(path: Path) -> Path:
    return _shrink(path)


def run_data() -> int:
    """Two splits of every dataset load, with finite numbers and a train / test
    split that does not overlap in size terms."""
    failures = 0
    expected = {  # (rows, features) from the sources' own tables
        "yacht": (308, 6), "energy": (768, 8), "concrete": (1030, 8), "wine_red": (1599, 11),
        "power_plant": (9568, 4), "kin8nm": (8192, 8), "naval": (11934, 16),
        "protein": (45730, 9), "kin40k": (40000, 8), "3droad": (434874, 3),
        "song": (515345, 90), "buzz": (583250, 77), "houseelectric": (2049280, 11),
    }
    for name, dataset in DATASETS.items():
        for split in sorted({0, n_splits(dataset) - 1}):
            try:
                data = load_split(dataset, split)
                rows = data.x_train.shape[0] + data.x_test.shape[0]
                want = expected[name]
                problems = []
                if (rows, data.x_train.shape[1]) != want:
                    problems.append(f"shape {(rows, data.x_train.shape[1])} != {want}")
                for label, arr in (("x_train", data.x_train), ("y_train", data.y_train),
                                   ("x_test", data.x_test), ("y_test", data.y_test)):
                    if not np.isfinite(arr).all():
                        problems.append(f"{label} has non-finite values")
                verdict = "FAIL" if problems else "ok"
                why = "; ".join(problems)
            except Exception as err:  # noqa: BLE001 - report every dataset, not the first
                verdict, why = "FAIL", f"{type(err).__name__}: {err}"
            failures += verdict == "FAIL"
            print(f"{verdict:4} data {name:14} split {split:2} {why}", flush=True)
    return failures


def main(argv: list[str]) -> int:
    failures = run_matrix(timeline=False)
    failures += run_matrix(timeline=True) if "--no-timeline" not in argv else 0
    if "--data" in argv:
        failures += run_data()
    print(f"\n{failures} FAIL")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
