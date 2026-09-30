"""GPy ``SparseGPRegression`` (VarDTC, the Titsias bound) with fixed ``Z``.

Optimizer as in ``gpy_fit``: ``native`` is ``model.optimize()``; ``matched``
is scipy ``L-BFGS-B`` with the case's caps. GPy's SVGP has no minibatch fit
in its API, so the ``svgp`` cell is N/A.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import GPy
import numpy as np

from common.records import load_case, write_result
from common.rss import peak_rss_bytes
from common.timeline import phase

from .gpy_fit import train
from .metrics import case_metrics
from .timing import warmup_fits


def build(case: dict, x, y, z):
    kernel = GPy.kern.RBF(
        input_dim=x.shape[1],
        variance=1.0,
        lengthscale=[case["lengthscale_init"]] * x.shape[1],
        ARD=True,
    )
    model = GPy.models.SparseGPRegression(x, y, kernel=kernel, Z=z.copy())
    model.Gaussian_noise.variance = case["noise_variance_init"]
    model.inducing_inputs.fix(warning=False)
    kernel.variance.fix(warning=False)  # gprx's sparse models have no signal variance
    return model


def run(case: dict) -> dict:
    if case["model"] != "sgpr":
        return {
            "lib": "gpy",
            "name": case["name"],
            "status": "na",
            "protocol": case["protocol"],
            "note": "GPy's SVGP has no minibatch fit in its API",
        }
    phase("load")
    n, d = int(case["n_rows"]), int(case["n_cols"])
    m = int(case["xs_n_rows"])
    x = np.asarray(case["x"]).reshape(d, n).T.copy()
    y = np.asarray(case["y"]).reshape(-1, 1)
    xs = np.asarray(case["xs"]).reshape(d, m).T.copy()
    z = np.asarray(case["z"]).reshape(d, int(case["n_inducing"])).T.copy()

    for _ in range(warmup_fits(n)):
        phase("warmup")
        train(case, build(case, x, y, z))
    phase("fit")
    model = build(case, x, y, z)
    t0 = time.perf_counter()
    info = train(case, model)
    fit_s = time.perf_counter() - t0
    phase("predict")
    t0 = time.perf_counter()
    mean, var = model.predict(xs)
    predict_s = time.perf_counter() - t0
    row = {
        "lib": "gpy",
        "name": case["name"],
        "status": "ok",
        "protocol": case["protocol"],
        "fit_s": fit_s,
        "predict_s": predict_s,
        "joint_evals": info["joint"],
        "value_evals": info["value"],
        "iterations": info["iterations"],
        "nlml": float(np.asarray(-model.log_likelihood()).reshape(-1)[0]),
        "peak_rss_bytes": peak_rss_bytes(),
        "note": info["message"],
    }
    row.update(case_metrics(case, mean, var))
    return row


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m perf.real.gpy_sparse_fit CASE.json", file=sys.stderr)
        return 2
    write_result(run(load_case(Path(sys.argv[1]))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
