"""Generate and apply P4-14 OnlineSgpr ops. Does not read P4-13 goldens."""

from __future__ import annotations

import numpy as np
import torch
from gpytorch.kernels import RBFKernel

OPS = 32
OPS_SEED = 0
START_M = 8
M_MAX = 16
START_N_BY_MAX = {256: 32, 1024: 128, 4096: 512}


def start_n_for(n_max: int) -> int:
    try:
        return START_N_BY_MAX[n_max]
    except KeyError as exc:
        raise ValueError(f"unsupported n_max {n_max}") from exc


def rbf_probe() -> RBFKernel:
    kernel = RBFKernel()
    kernel.initialize(lengthscale=torch.tensor([[1.0]], dtype=torch.float64))
    kernel.double()
    return kernel


def k_mm_positive_definite(kernel: RBFKernel, z_coords: np.ndarray, z_in: list[int]) -> bool:
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
    n_max: int,
    m_max: int,
) -> list[str]:
    kinds: list[str] = []
    if len(x_in) < n_max and x_pool:
        kinds.append("insert")
    if len(x_in) > 1:
        kinds.append("delete")
    if len(z_in) < m_max and z_pool:
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


def generate_ops(
    kernel: RBFKernel,
    z_coords: np.ndarray,
    *,
    start_n: int,
    n_max: int,
    start_m: int = START_M,
    m_max: int = M_MAX,
    n_ops: int = OPS,
    seed: int = OPS_SEED,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    x_in = list(range(start_n))
    x_pool = list(range(start_n, n_max))
    z_in = list(range(start_m))
    z_pool = list(range(start_m, m_max))
    ops: list[dict] = []
    attempts = 0
    while len(ops) < n_ops:
        attempts += 1
        if attempts > 10_000:
            raise RuntimeError("could not draw 32 ops that keep K_mm PD")
        kinds = legal_kinds(x_in, x_pool, z_in, z_pool, n_max, m_max)
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
