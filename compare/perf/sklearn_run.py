"""sklearn cell: factor at fixed theta, then N joint MLL+grad evals."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from common.problems import unpack_column_major as unpack_rows
from common.records import load_case, write_result
from common.rss import peak_rss_bytes
from common.timing import median, min_max, timed_reps, warmup_count


def make_model(case: dict, lengthscales: np.ndarray) -> GaussianProcessRegressor:
    if case["ard"]:
        kernel = RBF(length_scale=lengthscales)
    else:
        kernel = RBF(length_scale=float(lengthscales[0]))
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=float(case["noise_variance_init"]),
        optimizer=None,
        n_restarts_optimizer=0,
        normalize_y=True,
        random_state=0,
    )


def run(case: dict) -> dict:
    x = unpack_rows(case["x"], case["n_rows"], case["n_cols"])
    y = np.asarray(case["y"], dtype=np.float64)
    xs = unpack_rows(case["xs"], case["xs_n_rows"], case["xs_n_cols"])
    lengthscales = np.asarray(case["lengthscales_init"], dtype=np.float64)
    n_evals = int(case["joint_evals"])
    warmup = warmup_count()
    reps = timed_reps(int(case["n_rows"]))

    factor_samples: list[float] = []
    model = None
    for i in range(warmup + reps):
        model = None
        next_model = make_model(case, lengthscales)
        t0 = time.perf_counter()
        next_model.fit(x, y)
        dt = time.perf_counter() - t0
        if i >= warmup:
            factor_samples.append(dt)
        model = next_model
    assert model is not None

    theta = np.asarray(model.kernel_.theta, dtype=np.float64)
    eval_samples: list[float] = []
    for i in range(warmup + reps):
        t0 = time.perf_counter()
        model.log_marginal_likelihood(theta, eval_gradient=True, clone_kernel=True)
        dt = time.perf_counter() - t0
        if i >= warmup:
            eval_samples.append(dt)
    eval_scale = [s * n_evals for s in eval_samples]

    predict_samples: list[float] = []
    for i in range(warmup + reps):
        t0 = time.perf_counter()
        model.predict(xs, return_std=True)
        dt = time.perf_counter() - t0
        if i >= warmup:
            predict_samples.append(dt)

    factor_lo, factor_hi = min_max(factor_samples)
    eval_lo, eval_hi = min_max(eval_scale)
    predict_lo, predict_hi = min_max(predict_samples)
    return {
        "lib": "sklearn",
        "name": case["name"],
        "status": "ok",
        "factor_s": median(factor_samples),
        "factor_min_s": factor_lo,
        "factor_max_s": factor_hi,
        "eval_s": median(eval_scale),
        "eval_min_s": eval_lo,
        "eval_max_s": eval_hi,
        "predict_s": median(predict_samples),
        "predict_min_s": predict_lo,
        "predict_max_s": predict_hi,
        "joint_evals": n_evals,
        "peak_rss_bytes": peak_rss_bytes(),
        "warmup": warmup,
        "reps": reps,
        "note": (
            f"optimizer=None factor + {n_evals}× median of one joint MLL+grad; "
            f"discard {warmup} then {reps} timed"
        ),
    }


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: sklearn_run.py CASE.json", file=sys.stderr)
        return 2
    write_result(run(load_case(Path(sys.argv[1]))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
