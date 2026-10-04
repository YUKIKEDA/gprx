"""P4-12 Sparse cases: P2B-16 problems, m=16 k-means, standardized y."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from common.problems import (
    all_cases,
    as_f64_list,
    kmeans_z,
    pack_column_major,
    unpack_column_major,
    write_cases,
)

M = 16
MODELS = ("sgpr", "svgp")


def standardize_targets(y: list[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(y, dtype=np.float64).reshape(-1)
    mean = float(values.mean())
    var = float(np.mean((values - mean) ** 2))
    std = var**0.5
    if not np.isfinite(std) or std <= 0.0:
        std = 1.0
    return ((values - mean) / std).astype(np.float64)


def all_sparse_cases() -> list[dict]:
    cases: list[dict] = []
    for base in all_cases():
        coords = unpack_column_major(base["x"], base["n_rows"], base["n_cols"])
        z = kmeans_z(coords, M)
        y = standardize_targets(base["y"])
        for model in MODELS:
            case = dict(base)
            case["name"] = f"{model}_{base['name']}"
            case["model"] = model
            case["y"] = as_f64_list(y)
            case["z"] = pack_column_major(z)
            case["n_inducing"] = M
            cases.append(case)
    return cases


def write_sparse_cases(out_dir: Path) -> list[Path]:
    return write_cases(out_dir, all_sparse_cases())
