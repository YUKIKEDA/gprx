"""P4-14 Sparse-online cases: raw y, k-means Z, harness-generated ops."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from io_util import unpack_rows
from problems import all_cases, as_f64_list, pack_column_major
from sparse_online_ops import (
    M_MAX,
    OPS_SEED,
    START_M,
    generate_ops,
    rbf_probe,
    start_n_for,
)

KMEANS_SEED = 0


def kmeans_z(coords: np.ndarray, m: int = M_MAX) -> np.ndarray:
    model = KMeans(n_clusters=m, random_state=KMEANS_SEED, n_init=10)
    model.fit(coords)
    return np.asarray(model.cluster_centers_, dtype=np.float64)


def all_sparse_online_cases() -> list[dict]:
    probe = rbf_probe()
    cases: list[dict] = []
    for base in all_cases():
        n_max = int(base["n_rows"])
        start_n = start_n_for(n_max)
        coords = unpack_rows(base["x"], n_max, int(base["n_cols"]))
        z = kmeans_z(coords)
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
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for case in all_sparse_online_cases():
        path = out_dir / f"{case['name']}.json"
        path.write_text(json.dumps(case), encoding="utf-8")
        paths.append(path)
    return paths
