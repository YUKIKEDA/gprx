"""Write sklearn Exact GPR goldens for gprx (P1A-12).

Noise is sklearn ``alpha``, matching ``GaussianLikelihood``, not ``WhiteKernel``.
``predict(..., return_std=True)`` is latent; observation variance adds ``alpha``.

Run with ``just gen-goldens``. ``cargo test`` reads the JSON and must not call Python.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

ROOT = Path(__file__).resolve().parent
GOLDENS = ROOT / "goldens"

# Same n=2 / n=3 1-d problems as tests/analytic.rs (column-major packing).
CASES = [
    {
        "name": "rbf_n2",
        "lengthscale": 1.0,
        "noise_variance": 0.1,
        "n_rows": 2,
        "n_cols": 1,
        "x": [0.0, 1.0],
        "y": [0.5, -0.25],
        "xs_n_rows": 3,
        "xs_n_cols": 1,
        "xs": [0.0, 0.5, 2.0],
    },
    {
        "name": "rbf_n3",
        "lengthscale": 1.25,
        "noise_variance": 0.16,
        "n_rows": 3,
        "n_cols": 1,
        "x": [0.0, 0.8, 1.7],
        "y": [0.4, -0.2, 0.9],
        "xs_n_rows": 2,
        "xs_n_cols": 1,
        "xs": [0.8, 1.2],
    },
]


def unpack_column_major(values: list[float], n_rows: int, n_cols: int) -> np.ndarray:
    packed = np.asarray(values, dtype=np.float64)
    return packed.reshape((n_cols, n_rows), order="C").T


def as_f64_list(values: np.ndarray) -> list[float]:
    return [float(v) for v in np.asarray(values, dtype=np.float64).ravel()]


def golden_for(case: dict) -> dict:
    ell = float(case["lengthscale"])
    noise = float(case["noise_variance"])
    x = unpack_column_major(case["x"], case["n_rows"], case["n_cols"])
    xs = unpack_column_major(case["xs"], case["xs_n_rows"], case["xs_n_cols"])
    y = np.asarray(case["y"], dtype=np.float64)
    gp = GaussianProcessRegressor(
        kernel=RBF(length_scale=ell),
        alpha=noise,
        optimizer=None,
        normalize_y=False,
    )
    gp.fit(x, y)
    mean, latent_std = gp.predict(xs, return_std=True)
    latent_var = np.square(latent_std)
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


def main() -> None:
    GOLDENS.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        payload = golden_for(case)
        path = GOLDENS / f"{case['name']}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
