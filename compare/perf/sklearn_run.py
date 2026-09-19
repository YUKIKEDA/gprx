"""sklearn cell: factor at fixed theta, then N joint MLL+grad evals."""

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
    n_evals = int(case["joint_evals"])
    if case["ard"]:
        kernel = RBF(length_scale=lengthscales)
    else:
        kernel = RBF(length_scale=float(lengthscales[0]))
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=float(case["noise_variance_init"]),
        optimizer=None,
        n_restarts_optimizer=0,
        normalize_y=True,
        random_state=0,
    )
    t0 = time.perf_counter()
    model.fit(x, y)
    factor_s = time.perf_counter() - t0
    theta = np.asarray(model.kernel_.theta, dtype=np.float64)
    t1 = time.perf_counter()
    for _ in range(n_evals):
        model.log_marginal_likelihood(theta, eval_gradient=True, clone_kernel=True)
    eval_s = time.perf_counter() - t1
    t2 = time.perf_counter()
    model.predict(xs, return_std=True)
    predict_s = time.perf_counter() - t2
    return {
        "lib": "sklearn",
        "name": case["name"],
        "status": "ok",
        "factor_s": factor_s,
        "eval_s": eval_s,
        "predict_s": predict_s,
        "joint_evals": n_evals,
        "peak_rss_bytes": peak_rss_bytes(),
        "note": f"optimizer=None factor + {n_evals} joint MLL+grad at the same theta",
    }


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: sklearn_run.py CASE.json", file=sys.stderr)
        return 2
    write_result(run(load_case(Path(sys.argv[1]))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
