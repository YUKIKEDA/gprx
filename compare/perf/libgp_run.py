"""libgp cell. Missing install or missing ARD support becomes N/A, not a loss."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

from io_util import load_case, unpack_rows, write_result
from rss import peak_rss_bytes


def zscore(y: np.ndarray) -> np.ndarray:
    mean = float(y.mean())
    std = float(y.std())
    if std == 0.0:
        std = 1.0
    return (y - mean) / std


def run(case: dict) -> dict:
    try:
        from libgp import GaussianProcess, OptimizerRProp
    except Exception as exc:  # import / native build
        return {
            "lib": "libgp",
            "name": case["name"],
            "status": "na",
            "fit_s": None,
            "predict_s": None,
            "joint_evals": None,
            "peak_rss_bytes": None,
            "note": f"libgp unavailable: {exc}",
        }

    x = unpack_rows(case["x"], case["n_rows"], case["n_cols"])
    y = zscore(np.asarray(case["y"], dtype=np.float64))
    xs = unpack_rows(case["xs"], case["xs_n_rows"], case["xs_n_cols"])
    noise_std = math.sqrt(float(case["noise_variance_init"]))
    lengthscales = [float(v) for v in case["lengthscales_init"]]
    if case["ard"]:
        cov = "CovSum ( CovSEard, CovNoise)"
        loghyper = [math.log(ell) for ell in lengthscales] + [0.0, math.log(noise_std)]
    else:
        cov = "CovSum ( CovSEiso, CovNoise)"
        loghyper = [math.log(lengthscales[0]), 0.0, math.log(noise_std)]

    try:
        gp = GaussianProcess(int(case["n_cols"]), cov)
        gp.set_loghyper(np.asarray(loghyper, dtype=np.float64))
        gp.add_patterns(x, y)
    except Exception as exc:
        return {
            "lib": "libgp",
            "name": case["name"],
            "status": "na",
            "fit_s": None,
            "predict_s": None,
            "joint_evals": None,
            "peak_rss_bytes": None,
            "note": f"same problem cannot be written: {exc}",
        }

    evals = {"n": 0}
    orig = gp.get_log_likelihood_gradient

    def counted():
        evals["n"] += 1
        return orig()

    gp.get_log_likelihood_gradient = counted  # type: ignore[method-assign]
    opt = OptimizerRProp()
    t0 = time.perf_counter()
    opt.maximize(gp, n=100, verbose=False)
    fit_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    gp.predict_with_variance(xs)
    predict_s = time.perf_counter() - t1
    return {
        "lib": "libgp",
        "name": case["name"],
        "status": "ok",
        "fit_s": fit_s,
        "predict_s": predict_s,
        "joint_evals": evals["n"],
        "peak_rss_bytes": peak_rss_bytes(),
        "note": "Rprop gradient calls, not L-BFGS joint evals",
    }


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: libgp_run.py CASE.json", file=sys.stderr)
        return 2
    write_result(run(load_case(Path(sys.argv[1]))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
