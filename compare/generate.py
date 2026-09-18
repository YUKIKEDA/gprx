"""Write Exact GPR goldens for gprx (P1A-12, P1A-17, product grad, P1B-6, P1B-7, P2B-2).

Fixed-hyperparameter cases (P1A-12 / P1A-17): noise is sklearn ``alpha``,
matching ``GaussianLikelihood``, not ``WhiteKernel``.
``predict(..., return_std=True)`` is latent; observation variance adds ``alpha``.

Fit cases (P1B-6): sklearn optimizes ``RBF + WhiteKernel`` (scalar or ARD
``length_scale``) with ``normalize_y`` and a tiny ``alpha`` jitter. gprx
matches that as isotropic RBF or ``RbfArdKernel`` + ``GaussianLikelihood``
+ ``StandardizeTarget`` (no White leaf). ``WhiteKernel`` on the sklearn side
is the optimized noise, not a second nugget on gprx.

P1B-7: sklearn has no leave-one-out API. After fit, goldens store GPML
LOO from sklearn's ``L_`` and ``alpha_``: ``μ_i = y_i - α_i / Q_ii``,
``σ_i² = 1 / Q_ii`` with ``Q = A⁻¹``. Latent LOO strips WhiteKernel noise
in the normalized space, then both maps undo ``normalize_y``.

P2B-2 Nelder–Mead: scipy ``minimize(method="Nelder-Mead")`` on sklearn's
log marginal likelihood. Written to a separate JSON; not mixed with L-BFGS
fit goldens.

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


def pack_column_major(coords: np.ndarray) -> list[float]:
    """Packs an ``n×d`` point matrix into gprx column-major order."""
    n_rows, n_cols = coords.shape
    packed = np.empty(n_rows * n_cols, dtype=np.float64)
    for dim in range(n_cols):
        packed[dim * n_rows : (dim + 1) * n_rows] = coords[:, dim]
    return as_f64_list(packed)


def forrester(x: np.ndarray) -> np.ndarray:
    return (6.0 * x - 2.0) ** 2 * np.sin(12.0 * x - 4.0)


def weighted_sphere(coords: np.ndarray) -> np.ndarray:
    """Anisotropic quadratic: shorter characteristic length in dim 0 than dim 1."""
    return (coords[:, 0] / 0.25) ** 2 + (coords[:, 1] / 1.0) ** 2


FIT_JITTER = 1e-10
FIT_NOISE_STD = 1.0
FIT_SEED = 0
FORRESTER_N = 16
FORRESTER_XS_N = 7
SPHERE_SIDE = 6


def make_forrester_case() -> dict:
    x = np.linspace(0.0, 1.0, FORRESTER_N).reshape(-1, 1)
    noise = FIT_NOISE_STD * np.random.default_rng(FIT_SEED).standard_normal(FORRESTER_N)
    y = forrester(x[:, 0]) + noise
    xs = np.linspace(0.05, 0.95, FORRESTER_XS_N).reshape(-1, 1)
    return {
        "name": "forrester_rbf",
        "kernel": "rbf",
        "function": "forrester1d",
        "lengthscales_init": [1.0],
        "noise_variance_init": 0.1,
        "noise_std_added": FIT_NOISE_STD,
        "n_rows": FORRESTER_N,
        "n_cols": 1,
        "x": pack_column_major(x),
        "y": as_f64_list(y),
        "xs_n_rows": FORRESTER_XS_N,
        "xs_n_cols": 1,
        "xs": pack_column_major(xs),
    }


def make_sphere_ard_case() -> dict:
    grid = np.linspace(0.0, 1.0, SPHERE_SIDE)
    xx, yy = np.meshgrid(grid, grid, indexing="xy")
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    n_rows = int(coords.shape[0])
    noise = FIT_NOISE_STD * np.random.default_rng(FIT_SEED).standard_normal(n_rows)
    y = weighted_sphere(coords) + noise
    xs = np.array(
        [
            [0.25, 0.25],
            [0.25, 0.75],
            [0.75, 0.25],
            [0.75, 0.75],
        ],
        dtype=np.float64,
    )
    return {
        "name": "sphere_rbf_ard",
        "kernel": "rbf_ard",
        "function": "weighted_sphere2d",
        "lengthscales_init": [1.0, 1.0],
        "noise_variance_init": 0.1,
        "noise_std_added": FIT_NOISE_STD,
        "n_rows": n_rows,
        "n_cols": 2,
        "x": pack_column_major(coords),
        "y": as_f64_list(y),
        "xs_n_rows": int(xs.shape[0]),
        "xs_n_cols": 2,
        "xs": pack_column_major(xs),
    }


FIT_CASES = [make_forrester_case(), make_sphere_ard_case()]


def make_nelder_mead_rbf_case() -> dict:
    n_rows = 16
    x = np.linspace(0.0, 8.0, n_rows)
    noise = 0.1 * np.random.default_rng(0).standard_normal(n_rows)
    y = np.sin(x) + noise
    return {
        "name": "nelder_mead_rbf_n16",
        "kernel": "rbf",
        "lengthscale_init": 1.0,
        "noise_variance_init": 0.1,
        "n_rows": n_rows,
        "n_cols": 1,
        "x": as_f64_list(x),
        "y": as_f64_list(y),
    }

NELDER_MEAD_CASES = [make_nelder_mead_rbf_case()]


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
    lengthscales = np.asarray(case["lengthscales_init"], dtype=np.float64)
    noise0 = float(case["noise_variance_init"])
    x = unpack_column_major(case["x"], case["n_rows"], case["n_cols"])
    xs = unpack_column_major(case["xs"], case["xs_n_rows"], case["xs_n_cols"])
    y = np.asarray(case["y"], dtype=np.float64)
    if lengthscales.size == 1:
        rbf = RBF(length_scale=float(lengthscales[0]))
    else:
        rbf = RBF(length_scale=lengthscales)
    kernel = rbf + WhiteKernel(noise_level=noise0)
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


def loo_from_fitted_gp(gp, noise_white: float) -> dict:
    """GPML leave-one-out from sklearn's Cholesky and α. No sklearn LOO API."""
    chol = np.tril(np.asarray(gp.L_, dtype=np.float64))
    n = int(chol.shape[0])
    alpha = np.asarray(gp.alpha_, dtype=np.float64).reshape(-1)
    y_trans = np.asarray(gp.y_train_, dtype=np.float64).reshape(-1)
    inv_l = np.linalg.solve(chol, np.eye(n, dtype=np.float64))
    q_diag = np.sum(np.square(inv_l), axis=0)
    loo_mean_t = y_trans - alpha / q_diag
    loo_obs_t = 1.0 / q_diag
    loo_lat_t = np.maximum(0.0, loo_obs_t - float(noise_white))
    y_mean = float(np.asarray(gp._y_train_mean, dtype=np.float64).ravel()[0])
    y_std = float(np.asarray(gp._y_train_std, dtype=np.float64).ravel()[0])
    scale_sq = y_std * y_std
    return {
        "loo_mean": as_f64_list(loo_mean_t * y_std + y_mean),
        "loo_latent_variance": as_f64_list(loo_lat_t * scale_sq),
        "loo_observation_variance": as_f64_list(loo_obs_t * scale_sq),
    }


def fit_golden(case: dict) -> dict:
    gp, mean, predict_var = fit_gp_optimize(case)
    rbf, white = gp.kernel_.k1, gp.kernel_.k2
    ells = np.atleast_1d(np.asarray(rbf.length_scale, dtype=np.float64).ravel())
    noise = float(np.asarray(white.noise_level, dtype=np.float64).ravel()[0])
    theta = np.asarray(gp.kernel_.theta, dtype=np.float64)
    n_cols = int(case["n_cols"])
    if ells.size != n_cols:
        raise RuntimeError(f"expected {n_cols} lengthscales, got {ells.size}")
    if theta.size != n_cols + 1:
        raise RuntimeError(f"expected {n_cols} log(ℓ) + log(σn²), got {theta.size}")
    lml = float(gp.log_marginal_likelihood(theta, eval_gradient=False))
    y_std = float(np.asarray(gp._y_train_std, dtype=np.float64).ravel()[0])
    # sklearn k** includes WhiteKernel, so return_std² is observation-like.
    # Latent strips the nugget in original scale: noise * s².
    latent_var = predict_var - noise * (y_std**2)
    loo = loo_from_fitted_gp(gp, noise)
    return {
        "kernel": case["kernel"],
        "sklearn_version": sklearn.__version__,
        "function": case["function"],
        "normalize_y": True,
        "lengthscales_init": [float(v) for v in case["lengthscales_init"]],
        "noise_variance_init": float(case["noise_variance_init"]),
        "noise_std_added": float(case["noise_std_added"]),
        "lengthscales": as_f64_list(ells),
        "noise_variance": noise,
        "theta": as_f64_list(theta),
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
        "loo_mean": loo["loo_mean"],
        "loo_latent_variance": loo["loo_latent_variance"],
        "loo_observation_variance": loo["loo_observation_variance"],
    }


def nelder_mead_golden(case: dict) -> dict:
    import scipy
    from scipy.optimize import Bounds, minimize

    x = unpack_column_major(case["x"], case["n_rows"], case["n_cols"])
    y = np.asarray(case["y"], dtype=np.float64)
    kernel = RBF(length_scale=float(case["lengthscale_init"])) + WhiteKernel(
        noise_level=float(case["noise_variance_init"])
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=FIT_JITTER,
        optimizer=None,
        normalize_y=False,
        random_state=0,
    )
    gp.fit(x, y)
    theta0 = np.asarray(gp.kernel.theta, dtype=np.float64)

    def nll(theta: np.ndarray) -> float:
        return -float(
            gp.log_marginal_likelihood(
                np.asarray(theta, dtype=np.float64), eval_gradient=False
            )
        )

    log_lo = float(np.log(1e-5))
    log_hi = float(np.log(1e5))
    result = minimize(
        nll,
        theta0,
        method="Nelder-Mead",
        bounds=Bounds(log_lo, log_hi),
    )
    theta = np.asarray(result.x, dtype=np.float64)
    if theta.size != 2:
        raise RuntimeError(f"expected log(ℓ) + log(σn²), got {theta.size}")
    lml = float(gp.log_marginal_likelihood(theta, eval_gradient=False))
    return {
        "kernel": case["kernel"],
        "sklearn_version": sklearn.__version__,
        "scipy_version": scipy.__version__,
        "lengthscale_init": float(case["lengthscale_init"]),
        "noise_variance_init": float(case["noise_variance_init"]),
        "lengthscale": float(np.exp(theta[0])),
        "noise_variance": float(np.exp(theta[1])),
        "theta": as_f64_list(theta),
        "n_rows": case["n_rows"],
        "n_cols": case["n_cols"],
        "x": case["x"],
        "y": case["y"],
        "log_marginal_likelihood": lml,
        "nfev": int(result.nfev),
        "success": bool(result.success),
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
    for case in NELDER_MEAD_CASES:
        write_golden(case["name"], nelder_mead_golden(case))


if __name__ == "__main__":
    main()
