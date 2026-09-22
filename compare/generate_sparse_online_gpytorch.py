"""Write GPyTorch OnlineSgpr goldens (P4-13). cargo test must not run this."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PERF = ROOT / "perf"
GOLDENS = ROOT / "goldens"

sys.path.insert(0, str(PERF))
from problems import (  # noqa: E402
    NOISE_VARIANCE_INIT,
    make_forrester,
    make_sphere,
    pack_column_major,
)

from generate_sparse_gpytorch import (  # noqa: E402
    FORRESTER_XS,
    KMEANS_SEED,
    M,
    SPHERE_XS,
    kmeans_z,
    make_kernel,
    titsias_sgpr,
    unpack_column_major,
)

START_N = 32
START_M = 8
N_MAX = 256
M_MAX = 16
OPS = 32
OPS_SEED = 0
KERNEL_NAMES = ("rbf", "matern32", "rbf_ard", "rbf_white")


def k_mm_positive_definite(kernel, z_coords: np.ndarray, z_in: list[int]) -> bool:
    z = torch.as_tensor(z_coords[np.asarray(z_in)], dtype=torch.float64)
    kernel.eval()
    with torch.no_grad():
        k_mm = kernel(z, z, diag=False)
        if hasattr(k_mm, "to_dense"):
            k_mm = k_mm.to_dense()
        dense = k_mm.detach().cpu().numpy()
    try:
        np.linalg.cholesky(dense)
    except np.linalg.LinAlgError:
        return False
    return True


def legal_kinds(
    x_in: list[int],
    x_pool: list[int],
    z_in: list[int],
    z_pool: list[int],
) -> list[str]:
    kinds: list[str] = []
    if len(x_in) < N_MAX and x_pool:
        kinds.append("insert")
    if len(x_in) > 1:
        kinds.append("delete")
    if len(z_in) < M_MAX and z_pool:
        kinds.append("insert_inducing")
    if len(z_in) > 1:
        kinds.append("delete_inducing")
    return kinds


def propose_op(
    kind: str,
    rng: np.random.Generator,
    x_in: list[int],
    x_pool: list[int],
    z_in: list[int],
    z_pool: list[int],
) -> dict:
    if kind == "insert":
        pick = int(rng.integers(0, len(x_pool)))
        return {"kind": kind, "pop_index": int(x_pool[pick])}
    if kind == "delete":
        slot = int(rng.integers(0, len(x_in)))
        return {"kind": kind, "slot": slot}
    if kind == "insert_inducing":
        pick = int(rng.integers(0, len(z_pool)))
        return {"kind": kind, "pop_index": int(z_pool[pick])}
    slot = int(rng.integers(0, len(z_in)))
    return {"kind": kind, "slot": slot}


def commit_op(
    op: dict,
    x_in: list[int],
    x_pool: list[int],
    z_in: list[int],
    z_pool: list[int],
) -> None:
    kind = op["kind"]
    if kind == "insert":
        pop_index = int(op["pop_index"])
        x_pool.remove(pop_index)
        x_in.append(pop_index)
    elif kind == "delete":
        pop_index = int(x_in.pop(int(op["slot"])))
        x_pool.append(pop_index)
    elif kind == "insert_inducing":
        pop_index = int(op["pop_index"])
        z_pool.remove(pop_index)
        z_in.append(pop_index)
    else:
        pop_index = int(z_in.pop(int(op["slot"])))
        z_pool.append(pop_index)


def generate_ops(kernel, z_coords: np.ndarray) -> list[dict]:
    rng = np.random.default_rng(OPS_SEED)
    x_in = list(range(START_N))
    x_pool = list(range(START_N, N_MAX))
    z_in = list(range(START_M))
    z_pool = list(range(START_M, M_MAX))
    ops: list[dict] = []
    attempts = 0
    while len(ops) < OPS:
        attempts += 1
        if attempts > 10_000:
            raise RuntimeError("could not draw 32 ops that keep K_mm PD")
        kinds = legal_kinds(x_in, x_pool, z_in, z_pool)
        if not kinds:
            raise RuntimeError("no legal online op remains")
        kind = str(rng.choice(np.asarray(kinds, dtype=object)))
        op = propose_op(kind, rng, x_in, x_pool, z_in, z_pool)
        z_trial = list(z_in)
        apply_op(op, list(x_in), z_trial)
        if kind in {"insert_inducing", "delete_inducing"} and not k_mm_positive_definite(
            kernel, z_coords, z_trial
        ):
            continue
        commit_op(op, x_in, x_pool, z_in, z_pool)
        ops.append(op)
    return ops


def apply_op(op: dict, x_in: list[int], z_in: list[int]) -> None:
    kind = op["kind"]
    if kind == "insert":
        x_in.append(int(op["pop_index"]))
    elif kind == "delete":
        del x_in[int(op["slot"])]
    elif kind == "insert_inducing":
        z_in.append(int(op["pop_index"]))
    elif kind == "delete_inducing":
        del z_in[int(op["slot"])]
    else:
        raise ValueError(f"unknown op {kind}")


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
    ops = generate_ops(probe, z_coords)
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
