"""P4-14 Sparse-online cases: raw y, k-means Z, harness-generated ops."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from common.ops import (
    M_MAX,
    OPS_SEED,
    START_M,
    generate_ops,
    rbf_probe,
    start_n_for,
)
from common.problems import (
    KMEANS_SEED,
    all_cases,
    as_f64_list,
    kmeans_z,
    pack_column_major,
    unpack_column_major,
    write_cases,
)


def all_sparse_online_cases() -> list[dict]:
    probe = rbf_probe()
    cases: list[dict] = []
    for base in all_cases():
        n_max = int(base["n_rows"])
        start_n = start_n_for(n_max)
        coords = unpack_column_major(base["x"], n_max, int(base["n_cols"]))
        z = kmeans_z(coords, M_MAX)
        ops = generate_ops(probe, z, start_n=start_n, n_max=n_max)
        case = dict(base)
        case["y"] = as_f64_list(np.asarray(base["y"], dtype=np.float64))
        case["z"] = pack_column_major(z)
        case["n_inducing"] = M_MAX
        case["start_n"] = start_n
        case["start_m"] = START_M
        case["n_max"] = n_max
        case["m_max"] = M_MAX
        case["kmeans_seed"] = KMEANS_SEED
        case["ops_seed"] = OPS_SEED
        case["ops"] = ops
        cases.append(case)
    return cases


def write_sparse_online_cases(out_dir: Path) -> list[Path]:
    return write_cases(out_dir, all_sparse_online_cases())
