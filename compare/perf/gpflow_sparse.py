"""GPflow SGPR / SVGP cell. CPU, float64. cargo test must not run this."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

from io_util import load_case, unpack_rows, write_result
from rss import peak_rss_bytes
from timing import median, min_max, timed_reps, warmup_count


def na(case: dict, note: str) -> dict:
    return {
        "lib": "gpflow",
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


def make_kernel(gpflow, case: dict):  # noqa: ANN001
    lengthscales = np.asarray(case["lengthscales_init"], dtype=np.float64)
    if case["ard"]:
        return gpflow.kernels.SquaredExponential(lengthscales=lengthscales, variance=1.0)
    return gpflow.kernels.SquaredExponential(lengthscales=float(lengthscales[0]), variance=1.0)


def make_sgpr(gpflow, tf, case: dict, x, y, z):  # noqa: ANN001
    kernel = make_kernel(gpflow, case)
    inducing = gpflow.inducing_variables.InducingPoints(z)
    model = gpflow.models.SGPR(
        data=(x, y),
        kernel=kernel,
        inducing_variable=inducing,
        noise_variance=float(case["noise_variance_init"]),
    )
    gpflow.set_trainable(model.inducing_variable, False)
    gpflow.set_trainable(model.kernel.variance, False)
    return model


def make_svgp(gpflow, tf, case: dict, z):  # noqa: ANN001
    kernel = make_kernel(gpflow, case)
    likelihood = gpflow.likelihoods.Gaussian(variance=float(case["noise_variance_init"]))
    inducing = gpflow.inducing_variables.InducingPoints(z)
    m = int(case["n_inducing"])
    model = gpflow.models.SVGP(
        kernel=kernel,
        likelihood=likelihood,
        inducing_variable=inducing,
        num_data=int(case["n_rows"]),
        whiten=True,
        q_diag=False,
    )
    gpflow.set_trainable(model.inducing_variable, False)
    gpflow.set_trainable(model.kernel.variance, False)
    model.q_mu.assign(tf.zeros((m, 1), dtype=tf.float64))
    eye = tf.eye(m, dtype=tf.float64)
    model.q_sqrt.assign(eye[None, ...])
    return model


def sgpr_joint(model) -> None:  # noqa: ANN001
    with __import__("tensorflow").GradientTape() as tape:
        loss = -model.elbo()
    _ = tape.gradient(loss, model.trainable_variables)


def svgp_joint(model, x, y) -> None:  # noqa: ANN001
    tf = __import__("tensorflow")
    with tf.GradientTape() as tape:
        loss = -model.elbo((x, y))
    _ = tape.gradient(loss, model.trainable_variables)


def run(case: dict) -> dict:
    try:
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")
        tf.config.experimental.enable_op_determinism()
        import gpflow

        gpflow.config.set_default_float(tf.float64)
    except Exception as exc:  # noqa: BLE001
        return na(case, f"gpflow import failed: {exc}")

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
                model = make_sgpr(gpflow, tf, case, x, y, z)
                _ = float(model.elbo().numpy())
            elif model_name == "svgp":
                model = make_svgp(gpflow, tf, case, z)
                _ = float(model.elbo((x, y)).numpy())
            else:
                return na(case, f"unknown sparse model {model_name}")
            dt = time.perf_counter() - t0
            if i >= warmup:
                factor_samples.append(dt)
        assert model is not None

        eval_samples: list[float] = []
        for i in range(warmup + reps):
            t0 = time.perf_counter()
            if model_name == "sgpr":
                sgpr_joint(model)
            else:
                svgp_joint(model, x, y)
            dt = time.perf_counter() - t0
            if i >= warmup:
                eval_samples.append(dt)
        eval_scale = [s * n_evals for s in eval_samples]

        predict_samples: list[float] = []
        for i in range(warmup + reps):
            t0 = time.perf_counter()
            if model_name == "sgpr":
                _ = model.predict_y(xs)
            else:
                _ = model.predict_y(xs)
            dt = time.perf_counter() - t0
            if i >= warmup:
                predict_samples.append(dt)
    except Exception as exc:  # noqa: BLE001
        return na(case, f"gpflow {model_name} failed: {exc}")

    factor_lo, factor_hi = min_max(factor_samples)
    eval_lo, eval_hi = min_max(eval_scale)
    predict_lo, predict_hi = min_max(predict_samples)
    return {
        "lib": "gpflow",
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
            f"GPflow {model_name} CPU f64 + {n_evals}× median of one joint; "
            f"discard {warmup} then {reps} timed"
        ),
    }


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: gpflow_sparse.py CASE.json", file=sys.stderr)
        return 2
    write_result(run(load_case(Path(sys.argv[1]))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
