"""sklearn GaussianProcessRegressor cell (same JSON case as the other runners)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from io_util import load_case, unpack_rows, write_result
from rss import peak_rss_bytes


def run(case: dict) -> dict:
    x = unpack_rows(case["x"], case["n_rows"], case["n_cols"])
    y = np.asarray(case["y"], dtype=np.float64)
    xs = unpack_rows(case["xs"], case["xs_n_rows"], case["xs_n_cols"])
    lengthscales = np.asarray(case["lengthscales_init"], dtype=np.float64)
    if case["ard"]:
        kernel = RBF(length_scale=lengthscales)
    else:
        kernel = RBF(length_scale=float(lengthscales[0]))
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=float(case["noise_variance_init"]),
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=0,
        normalize_y=True,
        random_state=0,
    )
    evals = {"n": 0}
    orig = GaussianProcessRegressor.log_marginal_likelihood

    def counted(self, theta=None, eval_gradient=False, clone_kernel=True):
        if eval_gradient:
            evals["n"] += 1
        return orig(self, theta, eval_gradient=eval_gradient, clone_kernel=clone_kernel)

    model.log_marginal_likelihood = counted.__get__(model, GaussianProcessRegressor)
    t0 = time.perf_counter()
    model.fit(x, y)
    fit_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    model.predict(xs, return_std=True)
    predict_s = time.perf_counter() - t1
    return {
        "lib": "sklearn",
        "name": case["name"],
        "status": "ok",
        "fit_s": fit_s,
        "predict_s": predict_s,
        "joint_evals": evals["n"],
        "peak_rss_bytes": peak_rss_bytes(),
        "note": None,
    }


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: sklearn_run.py CASE.json", file=sys.stderr)
        return 2
    write_result(run(load_case(Path(sys.argv[1]))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
