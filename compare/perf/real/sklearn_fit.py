"""sklearn cell: ARD ``C * RBF + White`` fit from the shared start, then score.

``native`` leaves the optimizer at the scikit-learn default (scipy
``L-BFGS-B`` through ``fmin_l_bfgs_b`` defaults); ``matched`` passes the
case's iteration cap and gradient tolerance to the same scipy routine.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from common.records import load_case, write_result
from common.rss import peak_rss_bytes
from common.timeline import phase

from .metrics import score
from .timing import warmup_fits


class CountingGpr(GaussianProcessRegressor):
    """Counts the objective calls of the optimizer and keeps its stop reason."""

    protocol = "native"
    max_iterations = 100
    gtol = 1e-8
    calls = 0
    iterations: int | None = None
    message: str | None = None

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        def counted(theta):
            self.calls += 1
            return obj_func(theta, eval_gradient=True)

        if self.protocol == "native":
            # sklearn's own default: scipy minimize(L-BFGS-B, jac=True, bounds).
            result = minimize(counted, initial_theta, method="L-BFGS-B", jac=True, bounds=bounds)
        else:
            result = minimize(
                counted,
                initial_theta,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={
                    "maxiter": self.max_iterations,
                    "gtol": self.gtol,
                    "ftol": 0.0,
                    "maxcor": 10,
                },
            )
        self.iterations = int(result.nit)
        self.message = str(result.message)
        return result.x, result.fun


def make_model(case: dict) -> CountingGpr:
    kernel = ConstantKernel(case["signal_variance_init"]) * RBF(
        length_scale=[case["lengthscale_init"]] * case["n_cols"]
    ) + WhiteKernel(case["noise_variance_init"])
    model = CountingGpr(kernel=kernel, alpha=1e-10, n_restarts_optimizer=0, random_state=0)
    if case["protocol"] == "fixed":
        model.optimizer = None
    model.protocol = case["protocol"]
    model.max_iterations = int(case["max_iterations"])
    model.gtol = float(case["gtol"])
    return model


def run(case: dict) -> dict:
    phase("load")
    n, d = int(case["n_rows"]), int(case["n_cols"])
    x = np.asarray(case["x"], dtype=np.float64).reshape(d, n).T
    y = np.asarray(case["y"], dtype=np.float64)
    m = int(case["xs_n_rows"])
    xs = np.asarray(case["xs"], dtype=np.float64).reshape(d, m).T
    ys = np.asarray(case["ys"], dtype=np.float64)

    for _ in range(warmup_fits(n)):
        phase("warmup")
        make_model(case).fit(x, y)
    phase("fit")
    model = make_model(case)
    t0 = time.perf_counter()
    model.fit(x, y)
    fit_s = time.perf_counter() - t0
    phase("predict")
    t0 = time.perf_counter()
    mean, std = model.predict(xs, return_std=True)
    predict_s = time.perf_counter() - t0

    row = {
        "lib": "sklearn",
        "name": case["name"],
        "status": "ok",
        "protocol": case["protocol"],
        "fit_s": fit_s,
        "predict_s": predict_s,
        "joint_evals": int(model.calls),
        "value_evals": 0,
        "iterations": model.iterations,
        "nlml": float(-model.log_marginal_likelihood_value_),
        "peak_rss_bytes": peak_rss_bytes(),
        "note": model.message,
    }
    row.update(score(mean, std**2, ys, case["y_mean"], case["y_std"]))
    return row


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m perf.real.sklearn_fit CASE.json", file=sys.stderr)
        return 2
    write_result(run(load_case(Path(sys.argv[1]))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
