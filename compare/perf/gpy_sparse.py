"""GPy SGPR / SVGP cell. CPU. cargo test must not run this."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from io_util import load_case, unpack_rows, write_result
from rss import peak_rss_bytes
from timing import median, min_max, timed_reps, warmup_count


def na(case: dict, note: str) -> dict:
    return {
        "lib": "gpy",
        "name": case["name"],
        "status": "na",
        "factor_s": None,
        "eval_s": None,
        "predict_s": None,
        "joint_evals": None,
        "peak_rss_bytes": None,
        "note": note,
    }


def arrays(case: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = unpack_rows(case["x"], case["n_rows"], case["n_cols"])
    y = np.asarray(case["y"], dtype=np.float64).reshape(-1, 1)
    z = unpack_rows(case["z"], case["n_inducing"], case["n_cols"])
    xs = unpack_rows(case["xs"], case["xs_n_rows"], case["xs_n_cols"])
    return x, y, z, xs


def make_kernel(GPy, case: dict):  # noqa: ANN001
    lengthscales = np.asarray(case["lengthscales_init"], dtype=np.float64)
    d = int(case["n_cols"])
    ard = bool(case["ard"])
    kernel = GPy.kern.RBF(input_dim=d, variance=1.0, lengthscale=1.0, ARD=ard)
    kernel.variance.fix(warning=False)
    if ard:
        kernel.lengthscale[:] = lengthscales
    else:
        kernel.lengthscale[:] = float(lengthscales[0])
    return kernel


def make_sgpr(GPy, case: dict, x, y, z):  # noqa: ANN001
    kernel = make_kernel(GPy, case)
    model = GPy.models.SparseGPRegression(x, y, kernel=kernel, Z=z)
    model.Gaussian_noise.variance = float(case["noise_variance_init"])
    model.inducing_inputs.fix(warning=False)
    model.parameters_changed()
    _ = model.log_likelihood()
    return model


def make_svgp(GPy, case: dict, x, y, z):  # noqa: ANN001
    from GPy.core.svgp import SVGP

    kernel = make_kernel(GPy, case)
    likelihood = GPy.likelihoods.Gaussian(variance=float(case["noise_variance_init"]))
    model = SVGP(x, y, z, kernel, likelihood)
    model.inducing_inputs.fix(warning=False)
    if hasattr(model, "q_u_mean"):
        model.q_u_mean[:] = 0.0
    model.parameters_changed()
    _ = model.log_likelihood()
    return model


def joint(model) -> None:  # noqa: ANN001
    model.parameters_changed()
    _ = model.log_likelihood()
    _ = model.objective_function_gradients()


def run(case: dict) -> dict:
    try:
        import GPy
    except Exception as exc:  # noqa: BLE001
        return na(case, f"GPy import failed: {exc}")

    x, y, z, xs = arrays(case)
    n_evals = int(case["joint_evals"])
    warmup = warmup_count()
    reps = timed_reps(int(case["n_rows"]))
    model_name = case["model"]
    factor_samples: list[float] = []
    model = None
    try:
        for i in range(warmup + reps):
            model = None
            t0 = time.perf_counter()
            if model_name == "sgpr":
                model = make_sgpr(GPy, case, x, y, z)
            elif model_name == "svgp":
                model = make_svgp(GPy, case, x, y, z)
            else:
                return na(case, f"unknown sparse model {model_name}")
            dt = time.perf_counter() - t0
            if i >= warmup:
                factor_samples.append(dt)
        assert model is not None

        eval_samples: list[float] = []
        for i in range(warmup + reps):
            t0 = time.perf_counter()
            joint(model)
            dt = time.perf_counter() - t0
            if i >= warmup:
                eval_samples.append(dt)
        eval_scale = [s * n_evals for s in eval_samples]

        predict_samples: list[float] = []
        for i in range(warmup + reps):
            t0 = time.perf_counter()
            _ = model.predict(xs)
            dt = time.perf_counter() - t0
            if i >= warmup:
                predict_samples.append(dt)
    except Exception as exc:  # noqa: BLE001
        return na(case, f"GPy {model_name} failed: {exc}")

    factor_lo, factor_hi = min_max(factor_samples)
    eval_lo, eval_hi = min_max(eval_scale)
    predict_lo, predict_hi = min_max(predict_samples)
    return {
        "lib": "gpy",
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
            f"GPy {model_name} + {n_evals}× median of one joint; "
            f"discard {warmup} then {reps} timed"
        ),
    }


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: gpy_sparse.py CASE.json", file=sys.stderr)
        return 2
    write_result(run(load_case(Path(sys.argv[1]))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
