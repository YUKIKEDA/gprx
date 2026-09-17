"""Write sklearn Exact GPR goldens for gprx (P1A-12, P1A-17, product grad, P1B-6).

Fixed-hyperparameter cases (P1A-12 / P1A-17): noise is sklearn ``alpha``,
matching ``GaussianLikelihood``, not ``WhiteKernel``.
``predict(..., return_std=True)`` is latent; observation variance adds ``alpha``.

Fit cases (P1B-6): sklearn optimizes ``RBF + WhiteKernel`` with ``normalize_y``
and a tiny ``alpha`` jitter. gprx matches that as RBF + ``GaussianLikelihood``
+ ``StandardizeTarget`` (no White leaf). ``WhiteKernel`` on the sklearn side
is the optimized noise, not a second nugget on gprx.

P1A-12: isotropic RBF. P1A-17: Sum/Product flatten plus extra leaves
(Matern, RQ, Periodic). Product and mixed trees include ``grad_theta``
once ``Gpr::value_and_gradient_into`` calls ``CompiledKernel::grad``.
sklearn ``RationalQuadratic.theta`` is ``[log(α), log(ℓ)]``; committed JSON
uses gprx order ``[log(ℓ), log(α)]``.

Run with ``just gen-goldens``. ``cargo test`` reads the JSON and must not call Python.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    ExpSineSquared,
    Matern,
    RationalQuadratic,
    WhiteKernel,
)

ROOT = Path(__file__).resolve().parent
GOLDENS = ROOT / "goldens"

# Same n=2 / n=3 1-d problems as tests/analytic.rs (column-major packing).
N2 = {
    "noise_variance": 0.1,
    "n_rows": 2,
    "n_cols": 1,
    "x": [0.0, 1.0],
    "y": [0.5, -0.25],
    "xs_n_rows": 3,
    "xs_n_cols": 1,
    "xs": [0.0, 0.5, 2.0],
}

N3 = {
    "noise_variance": 0.16,
    "n_rows": 3,
    "n_cols": 1,
    "x": [0.0, 0.8, 1.7],
    "y": [0.4, -0.2, 0.9],
    "xs_n_rows": 2,
    "xs_n_cols": 1,
    "xs": [0.8, 1.2],
}

RBF_CASES = [
    {**N2, "name": "rbf_n2", "lengthscale": 1.0},
    {**N3, "name": "rbf_n3", "lengthscale": 1.25},
]

# ``op`` is a leaf name, ``sum``, or ``product``. Leaves are depth-first,
# left-to-right — the same flatten order as KernelSpec.
COMPOSITE_CASES = [
    {
        **N2,
        "name": "rbf_sum_n2",
        "op": "sum",
        "leaves": [
            {"type": "rbf", "lengthscale": 1.0},
            {"type": "rbf", "lengthscale": 2.0},
        ],
        "eval_gradient": True,
    },
    {
        **N2,
        "name": "constant_rbf_product_n2",
        "op": "product",
        "leaves": [
            {"type": "constant", "constant": 1.5},
            {"type": "rbf", "lengthscale": 1.0},
        ],
        "eval_gradient": True,
    },
    {
        **N2,
        "name": "constant_rbf_plus_rbf_n2",
        "op": "sum",
        "leaves": [
            {
                "type": "product",
                "leaves": [
                    {"type": "constant", "constant": 1.5},
                    {"type": "rbf", "lengthscale": 1.0},
                ],
            },
            {"type": "rbf", "lengthscale": 2.0},
        ],
        "eval_gradient": True,
    },
    {
        **N2,
        "name": "matern32_n2",
        "op": "leaf",
        "leaves": [{"type": "matern", "lengthscale": 1.0, "nu": 1.5}],
        "eval_gradient": True,
    },
    {
        **N2,
        "name": "rq_n2",
        "op": "leaf",
        "leaves": [{"type": "rq", "lengthscale": 1.0, "alpha": 1.5}],
        "eval_gradient": True,
    },
    {
        **N2,
        "name": "periodic_n2",
        "op": "leaf",
        "leaves": [{"type": "periodic", "lengthscale": 1.0, "period": 2.0}],
        "eval_gradient": True,
    },
]


def unpack_column_major(values: list[float], n_rows: int, n_cols: int) -> np.ndarray:
    packed = np.asarray(values, dtype=np.float64)
    return packed.reshape((n_cols, n_rows), order="C").T


def as_f64_list(values: np.ndarray) -> list[float]:
    return [float(v) for v in np.asarray(values, dtype=np.float64).ravel()]


def forrester(x: np.ndarray) -> np.ndarray:
    return (6.0 * x - 2.0) ** 2 * np.sin(12.0 * x - 4.0)


# 1-d Forrester on [0, 1] plus seeded observation noise so MLE noise
# stays off the sklearn WhiteKernel lower bound. gprx uses RBF +
# GaussianLikelihood. Tiny alpha is Cholesky jitter, not model noise.
FORRESTER_N = 16
FORRESTER_XS_N = 7
_FORRESTER_X = np.linspace(0.0, 1.0, FORRESTER_N)
_FORRESTER_XS = np.linspace(0.05, 0.95, FORRESTER_XS_N)
FIT_JITTER = 1e-10
FIT_NOISE_STD = 1.0
FIT_SEED = 0
_FORRESTER_Y = forrester(_FORRESTER_X) + FIT_NOISE_STD * np.random.default_rng(
    FIT_SEED
).standard_normal(FORRESTER_N)
FIT_CASES = [
    {
        "name": "forrester_rbf",
        "lengthscale_init": 1.0,
        "noise_variance_init": 0.1,
        "noise_std_added": FIT_NOISE_STD,
        "n_rows": FORRESTER_N,
        "n_cols": 1,
        "x": as_f64_list(_FORRESTER_X),
        "y": as_f64_list(_FORRESTER_Y),
        "xs_n_rows": FORRESTER_XS_N,
        "xs_n_cols": 1,
        "xs": as_f64_list(_FORRESTER_XS),
    }
]


def sklearn_leaf(leaf: dict):
    kind = leaf["type"]
    if kind in ("sum", "product"):
        return sklearn_kernel(kind, leaf["leaves"])
    if kind == "rbf":
        return RBF(length_scale=float(leaf["lengthscale"]))
    if kind == "constant":
        return ConstantKernel(constant_value=float(leaf["constant"]))
    if kind == "matern":
        return Matern(length_scale=float(leaf["lengthscale"]), nu=float(leaf["nu"]))
    if kind == "rq":
        return RationalQuadratic(
            length_scale=float(leaf["lengthscale"]),
            alpha=float(leaf["alpha"]),
        )
    if kind == "periodic":
        return ExpSineSquared(
            length_scale=float(leaf["lengthscale"]),
            periodicity=float(leaf["period"]),
        )
    raise ValueError(f"unknown leaf type {kind}")


def sklearn_kernel(op: str, leaves: list[dict]):
    built = [sklearn_leaf(leaf) for leaf in leaves]
    if op == "leaf":
        if len(built) != 1:
            raise ValueError("leaf op expects one leaf")
        return built[0]
    if not built:
        raise ValueError(f"{op} expects at least one leaf")
    kernel = built[0]
    for term in built[1:]:
        if op == "sum":
            kernel = kernel + term
        elif op == "product":
            kernel = kernel * term
        else:
            raise ValueError(f"unknown op {op}")
    return kernel


def fit_gp(kernel, case: dict):
    noise = float(case["noise_variance"])
    x = unpack_column_major(case["x"], case["n_rows"], case["n_cols"])
    xs = unpack_column_major(case["xs"], case["xs_n_rows"], case["xs_n_cols"])
    y = np.asarray(case["y"], dtype=np.float64)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=noise,
        optimizer=None,
        normalize_y=False,
    )
    gp.fit(x, y)
    mean, latent_std = gp.predict(xs, return_std=True)
    latent_var = np.square(latent_std)
    return gp, mean, latent_var, noise


def common_fields(case: dict, mean, latent_var, noise: float) -> dict:
    return {
        "sklearn_version": sklearn.__version__,
        "noise_variance": noise,
        "n_rows": case["n_rows"],
        "n_cols": case["n_cols"],
        "x": case["x"],
        "y": case["y"],
        "xs_n_rows": case["xs_n_rows"],
        "xs_n_cols": case["xs_n_cols"],
        "xs": case["xs"],
        "mean": as_f64_list(mean),
        "latent_variance": as_f64_list(latent_var),
        "observation_variance": as_f64_list(latent_var + noise),
    }


def rbf_golden(case: dict) -> dict:
    ell = float(case["lengthscale"])
    gp, mean, latent_var, noise = fit_gp(RBF(length_scale=ell), case)
    lml, grad = gp.log_marginal_likelihood(gp.kernel_.theta, eval_gradient=True)
    if grad.size != 1:
        raise RuntimeError(f"expected one RBF theta, got {grad.size}")
    return {
        "kernel": "rbf",
        "sklearn_version": sklearn.__version__,
        "lengthscale": ell,
        "noise_variance": noise,
        "n_rows": case["n_rows"],
        "n_cols": case["n_cols"],
        "x": case["x"],
        "y": case["y"],
        "xs_n_rows": case["xs_n_rows"],
        "xs_n_cols": case["xs_n_cols"],
        "xs": case["xs"],
        "mean": as_f64_list(mean),
        "latent_variance": as_f64_list(latent_var),
        "observation_variance": as_f64_list(latent_var + noise),
        "log_marginal_likelihood": float(lml),
        "grad_log_lengthscale": float(grad[0]),
    }


def flatten_leaves(nodes: list[dict]) -> list[dict]:
    """Depth-first leaves, matching KernelSpec flatten order."""
    out: list[dict] = []
    for node in nodes:
        kind = node["type"]
        if kind in ("sum", "product"):
            out.extend(flatten_leaves(node["leaves"]))
        else:
            out.append(node)
    return out


def to_gprx_theta(leaves: list[dict], values: np.ndarray) -> list[float]:
    """Reorder sklearn ``kernel.theta`` to gprx flatten order.

    sklearn ``RationalQuadratic`` lists ``alpha`` before ``length_scale``
    (alphabetical ``hyperparameter_*``). gprx is ``[log(ℓ), log(α)]``.
    """
    out: list[float] = []
    offset = 0
    for leaf in flatten_leaves(leaves):
        kind = leaf["type"]
        if kind == "rq":
            out.append(float(values[offset + 1]))
            out.append(float(values[offset]))
            offset += 2
        elif kind == "periodic":
            out.extend(float(v) for v in values[offset : offset + 2])
            offset += 2
        elif kind in ("rbf", "constant", "matern"):
            out.append(float(values[offset]))
            offset += 1
        else:
            raise ValueError(f"unknown leaf type {kind}")
    if offset != values.size:
        raise RuntimeError(f"theta length {values.size} != consumed {offset}")
    return out


def composite_golden(case: dict) -> dict:
    kernel = sklearn_kernel(case["op"], case["leaves"])
    gp, mean, latent_var, noise = fit_gp(kernel, case)
    sklearn_theta = np.asarray(gp.kernel_.theta, dtype=np.float64)
    payload = common_fields(case, mean, latent_var, noise)
    payload.update(
        {
            "kernel": case["op"],
            "leaves": case["leaves"],
            "theta": to_gprx_theta(case["leaves"], sklearn_theta),
        }
    )
    if case["eval_gradient"]:
        lml, grad = gp.log_marginal_likelihood(sklearn_theta, eval_gradient=True)
        if grad.size != sklearn_theta.size:
            raise RuntimeError(
                f"{case['name']}: expected {sklearn_theta.size} grads, got {grad.size}"
            )
        payload["log_marginal_likelihood"] = float(lml)
        payload["grad_theta"] = to_gprx_theta(case["leaves"], grad)
    else:
        lml = gp.log_marginal_likelihood(sklearn_theta, eval_gradient=False)
        payload["log_marginal_likelihood"] = float(lml)
    return payload


def fit_gp_optimize(case: dict):
    """Fit with sklearn L-BFGS-B. WhiteKernel is the optimized noise."""
    ell0 = float(case["lengthscale_init"])
    noise0 = float(case["noise_variance_init"])
    x = unpack_column_major(case["x"], case["n_rows"], case["n_cols"])
    xs = unpack_column_major(case["xs"], case["xs_n_rows"], case["xs_n_cols"])
    y = np.asarray(case["y"], dtype=np.float64)
    kernel = RBF(length_scale=ell0) + WhiteKernel(noise_level=noise0)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=FIT_JITTER,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=0,
        normalize_y=True,
        random_state=0,
    )
    gp.fit(x, y)
    mean, predict_std = gp.predict(xs, return_std=True)
    return gp, mean, np.square(predict_std)


def fit_golden(case: dict) -> dict:
    gp, mean, predict_var = fit_gp_optimize(case)
    rbf, white = gp.kernel_.k1, gp.kernel_.k2
    ell = float(np.asarray(rbf.length_scale, dtype=np.float64).ravel()[0])
    noise = float(np.asarray(white.noise_level, dtype=np.float64).ravel()[0])
    theta = np.asarray(gp.kernel_.theta, dtype=np.float64)
    if theta.size != 2:
        raise RuntimeError(f"expected [log(ℓ), log(σn²)], got {theta.size}")
    lml = float(gp.log_marginal_likelihood(theta, eval_gradient=False))
    y_std = float(np.asarray(gp._y_train_std, dtype=np.float64).ravel()[0])
    # sklearn k** includes WhiteKernel, so return_std² is observation-like.
    # Latent strips the nugget in original scale: noise * s².
    latent_var = predict_var - noise * (y_std**2)
    return {
        "kernel": "rbf",
        "sklearn_version": sklearn.__version__,
        "function": "forrester1d",
        "normalize_y": True,
        "lengthscale_init": float(case["lengthscale_init"]),
        "noise_variance_init": float(case["noise_variance_init"]),
        "noise_std_added": float(case["noise_std_added"]),
        "lengthscale": ell,
        "noise_variance": noise,
        "theta": [float(theta[0]), float(theta[1])],
        "n_rows": case["n_rows"],
        "n_cols": case["n_cols"],
        "x": case["x"],
        "y": case["y"],
        "xs_n_rows": case["xs_n_rows"],
        "xs_n_cols": case["xs_n_cols"],
        "xs": case["xs"],
        "mean": as_f64_list(mean),
        "latent_variance": as_f64_list(latent_var),
        "observation_variance": as_f64_list(predict_var),
        "log_marginal_likelihood": lml,
    }


def write_golden(name: str, payload: dict) -> None:
    path = GOLDENS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


def main() -> None:
    GOLDENS.mkdir(parents=True, exist_ok=True)
    for case in RBF_CASES:
        write_golden(case["name"], rbf_golden(case))
    for case in COMPOSITE_CASES:
        write_golden(case["name"], composite_golden(case))
    for case in FIT_CASES:
        write_golden(case["name"], fit_golden(case))


if __name__ == "__main__":
    main()
