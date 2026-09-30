"""GPyTorch Titsias assemble for a 32-op OnlineSgpr sequence. No query."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
from gpytorch.kernels import RBFKernel

torch.set_default_dtype(torch.float64)
torch.set_num_threads(max(1, os.cpu_count() or 1))

from common.problems import unpack_column_major as unpack_rows
from common.records import load_case, write_result
from common.ops import apply_op
from common.rss import peak_rss_bytes
from common.timing import median, min_max, timed_reps, warmup_count

DEVICE = torch.device("cpu")


def _chol(mat: torch.Tensor) -> torch.Tensor:
    try:
        return torch.linalg.cholesky(mat)
    except RuntimeError:
        return torch.linalg.cholesky(mat + 1e-8 * torch.eye(mat.size(0), dtype=mat.dtype))


def make_kernel(case: dict) -> RBFKernel:
    lengthscales = case["lengthscales_init"]
    if bool(case["ard"]):
        kernel = RBFKernel(ard_num_dims=int(case["n_cols"]))
    else:
        kernel = RBFKernel()
    kernel.initialize(lengthscale=torch.tensor(lengthscales, dtype=torch.float64).view(1, -1))
    kernel.double()
    kernel.to(DEVICE)
    return kernel


def titsias_assemble(
    kernel: RBFKernel,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    z: torch.Tensor,
    noise: float,
) -> None:
    m = z.size(0)
    kernel.eval()
    with torch.no_grad():
        k_mm = kernel(z, z, diag=False)
        if hasattr(k_mm, "to_dense"):
            k_mm = k_mm.to_dense()
        k_mn = kernel(z, train_x, diag=False)
        if hasattr(k_mn, "to_dense"):
            k_mn = k_mn.to_dense()
        k_diag = kernel(train_x, train_x, diag=True)
        if hasattr(k_diag, "to_dense"):
            k_diag = k_diag.to_dense()
        l_mm = _chol(k_mm)
        a = torch.linalg.solve_triangular(l_mm, k_mn, upper=False)
        b = a @ a.T + noise * torch.eye(m, dtype=a.dtype, device=a.device)
        l_b = _chol(b)
        ay = a @ train_y.reshape(-1, 1)
        w = torch.linalg.solve_triangular(
            l_b.T,
            torch.linalg.solve_triangular(l_b, ay, upper=False),
            upper=True,
        )
        _ = float(k_diag.sum().item()), float((a * a).sum().item()), w


def run(case: dict) -> dict:
    n_max = int(case["n_rows"])
    d = int(case["n_cols"])
    start_n = int(case["start_n"])
    start_m = int(case["start_m"])
    coords = unpack_rows(case["x"], n_max, d)
    z_coords = unpack_rows(case["z"], int(case["n_inducing"]), d)
    y = np.asarray(case["y"], dtype=np.float64)
    noise = float(case["noise_variance_init"])
    kernel = make_kernel(case)
    warmup = warmup_count()
    reps = timed_reps(n_max)
    samples: list[float] = []
    for i in range(warmup + reps):
        x_in = list(range(start_n))
        z_in = list(range(start_m))
        t0 = time.perf_counter()
        for op in case["ops"]:
            apply_op(op, x_in, z_in)
            train_x = torch.as_tensor(coords[np.asarray(x_in)], dtype=torch.float64, device=DEVICE)
            train_y = torch.as_tensor(y[np.asarray(x_in)], dtype=torch.float64, device=DEVICE)
            z = torch.as_tensor(z_coords[np.asarray(z_in)], dtype=torch.float64, device=DEVICE)
            titsias_assemble(kernel, train_x, train_y, z, noise)
        dt = time.perf_counter() - t0
        if i >= warmup:
            samples.append(dt)
    lo, hi = min_max(samples)
    return {
        "lib": "gpytorch",
        "name": case["name"],
        "status": "ok",
        "ops_s": median(samples),
        "ops_min_s": lo,
        "ops_max_s": hi,
        "peak_rss_bytes": peak_rss_bytes(),
        "warmup": warmup,
        "reps": reps,
        "note": (
            f"GPyTorch Titsias assemble ×32, no query; prefix untimed; "
            f"discard {warmup} then {reps} timed"
        ),
    }


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: gpytorch_sparse_online.py CASE.json", file=sys.stderr)
        return 2
    write_result(run(load_case(Path(sys.argv[1]))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
