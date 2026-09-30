"""GPy exact cell: ARD ``RBF`` (with variance) + Gaussian noise, CPU.

``native`` is ``model.optimize()`` as GPy ships it (``lbfgsb``, i.e. scipy
``fmin_l_bfgs_b``, at most 1000 iterations). ``matched`` hands the same
objective to scipy ``L-BFGS-B`` with the case's iteration cap and gradient
tolerance. GPy optimizes softplus-transformed parameters with no bounds.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import GPy
import numpy as np
from scipy.optimize import minimize

from common.records import load_case, write_result
from common.rss import peak_rss_bytes
from common.timeline import phase

from .metrics import score
from .timing import warmup_fits


def build(case: dict, x: np.ndarray, y: np.ndarray):
    kernel = GPy.kern.RBF(
        input_dim=x.shape[1],
        variance=case["signal_variance_init"],
        lengthscale=[case["lengthscale_init"]] * x.shape[1],
        ARD=True,
    )
    model = GPy.models.GPRegression(x, y, kernel=kernel)
    model.Gaussian_noise.variance = case["noise_variance_init"]
    return model


class Counter:
    def __init__(self, model) -> None:  # noqa: ANN001
        self.joint = 0
        self.value = 0
        objective_grads, objective = model._objective_grads, model._objective

        def counted_joint(x):  # noqa: ANN001
            self.joint += 1
            return objective_grads(x)

        def counted_value(x):  # noqa: ANN001
            self.value += 1
            return objective(x)

        model._objective_grads = counted_joint
        model._objective = counted_value


def train(case: dict, model) -> dict:  # noqa: ANN001
    counter = Counter(model)
    protocol = case["protocol"]
    if protocol == "fixed":
        return {"joint": 0, "value": 0, "iterations": 0, "message": "fixed"}
    if protocol == "native":
        model.optimize(messages=False)
        status = model.optimization_runs[-1].status
        return {
            "joint": counter.joint,
            "value": counter.value,
            "iterations": None,
            "message": f"model.optimize() lbfgsb: {status}",
        }
    result = minimize(
        lambda x: counter_joint(counter, model, x),
        model.optimizer_array.copy(),
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": int(case["max_iterations"]),
            "gtol": float(case["gtol"]),
            "ftol": 0.0,
            "maxcor": 10,
        },
    )
    model.optimizer_array = result.x
    return {
        "joint": counter.joint,
        "value": counter.value,
        "iterations": int(result.nit),
        "message": str(result.message),
    }


def counter_joint(counter: Counter, model, x):  # noqa: ANN001
    value, grad = model._objective_grads(x)
    return value, grad


def run(case: dict) -> dict:
    phase("load")
    n, d = int(case["n_rows"]), int(case["n_cols"])
    m = int(case["xs_n_rows"])
    x = np.asarray(case["x"], dtype=np.float64).reshape(d, n).T.copy()
    y = np.asarray(case["y"], dtype=np.float64).reshape(-1, 1)
    xs = np.asarray(case["xs"], dtype=np.float64).reshape(d, m).T.copy()

    for _ in range(warmup_fits(n)):
        phase("warmup")
        train(case, build(case, x, y))
    phase("fit")
    model = build(case, x, y)
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
    if case.get("return_predictions"):
        row["pred_mean"] = (np.asarray(mean.ravel()).ravel() * case["y_std"] + case["y_mean"]).tolist()
        row["pred_var"] = (np.asarray(var.ravel()).ravel() * case["y_std"] ** 2).tolist()
    row.update(score(mean.ravel(), var.ravel(), np.asarray(case["ys"]), case["y_mean"], case["y_std"]))
    return row


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m perf.real.gpy_fit CASE.json", file=sys.stderr)
        return 2
    write_result(run(load_case(Path(sys.argv[1]))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
