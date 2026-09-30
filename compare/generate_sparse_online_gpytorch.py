"""Write GPyTorch OnlineSgpr goldens (P4-13). cargo test must not run this."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from common.ops import apply_op, generate_ops
from common.problems import (
    KMEANS_SEED,
    NOISE_VARIANCE_INIT,
    kmeans_z,
    make_forrester,
    make_sphere,
    pack_column_major,
    unpack_column_major,
)
from generate_sparse_gpytorch import FORRESTER_XS, M, SPHERE_XS, make_kernel, titsias_sgpr

ROOT = Path(__file__).resolve().parent
GOLDENS = ROOT / "goldens"

START_N = 32
START_M = 8
N_MAX = 256
M_MAX = 16
OPS = 32
OPS_SEED = 0
KERNEL_NAMES = ("rbf", "matern32", "rbf_ard", "rbf_white")


def snapshot(
    kernel,
    coords: np.ndarray,
    y: np.ndarray,
    z_coords: np.ndarray,
    x_in: list[int],
    z_in: list[int],
    xs: torch.Tensor,
) -> dict:
    train_x = torch.as_tensor(coords[np.asarray(x_in)], dtype=torch.float64)
    train_y = torch.as_tensor(y[np.asarray(x_in)], dtype=torch.float64)
    z = torch.as_tensor(z_coords[np.asarray(z_in)], dtype=torch.float64)
    nlml, mean, obs_var, latent_var = titsias_sgpr(
        kernel, train_x, train_y, z, xs, NOISE_VARIANCE_INIT
    )
    return {
        "n": len(x_in),
        "m": len(z_in),
        "mean": mean,
        "observation_variance": obs_var,
        "latent_variance": latent_var,
        "neg_log_marginal_likelihood": nlml,
    }


def kernel_steps(
    name: str,
    n_cols: int,
    coords: np.ndarray,
    y: np.ndarray,
    z_coords: np.ndarray,
    xs: torch.Tensor,
    ops: list[dict],
) -> dict:
    kernel, lengthscales, white = make_kernel(name, n_cols)
    kernel.double()
    x_in = list(range(START_N))
    z_in = list(range(START_M))
    steps = [snapshot(kernel, coords, y, z_coords, x_in, z_in, xs)]
    for op in ops:
        apply_op(op, x_in, z_in)
        steps.append(snapshot(kernel, coords, y, z_coords, x_in, z_in, xs))
    return {
        "name": name,
        "lengthscales": lengthscales,
        "white_variance": white,
        "steps": steps,
    }


def write_problem(stem: str, raw: dict, xs: np.ndarray) -> None:
    n = int(raw["n_rows"])
    d = int(raw["n_cols"])
    coords = unpack_column_major(raw["x"], n, d)
    y = np.asarray(raw["y"], dtype=np.float64)
    z_coords = kmeans_z(coords, M)
    probe, _, _ = make_kernel("rbf", d)
    probe.double()
    ops = generate_ops(
        probe,
        z_coords,
        start_n=START_N,
        n_max=N_MAX,
        start_m=START_M,
        m_max=M_MAX,
        n_ops=OPS,
        seed=OPS_SEED,
    )
    xs_t = torch.as_tensor(xs, dtype=torch.float64)
    payload = {
        "problem": raw["problem"],
        "n_rows": n,
        "n_cols": d,
        "x": raw["x"],
        "y": raw["y"],
        "z": pack_column_major(z_coords),
        "m": M,
        "kmeans_seed": KMEANS_SEED,
        "start_n": START_N,
        "start_m": START_M,
        "ops_seed": OPS_SEED,
        "n_max": N_MAX,
        "m_max": M_MAX,
        "xs_n_rows": int(xs.shape[0]),
        "xs_n_cols": int(xs.shape[1]),
        "xs": pack_column_major(xs),
        "noise_variance": NOISE_VARIANCE_INIT,
        "ops": ops,
        "kernels": [
            kernel_steps(name, d, coords, y, z_coords, xs_t, ops) for name in KERNEL_NAMES
        ],
    }
    GOLDENS.mkdir(parents=True, exist_ok=True)
    dest = GOLDENS / f"{stem}.json"
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {dest.relative_to(ROOT)}")


def main() -> int:
    torch.set_default_dtype(torch.float64)
    write_problem("online_sgpr_forrester", make_forrester(256), FORRESTER_XS)
    write_problem("online_sgpr_sphere", make_sphere(16), SPHERE_XS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
