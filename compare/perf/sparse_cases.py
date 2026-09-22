"""P4-12 Sparse cases: P2B-16 problems, m=16 k-means, standardized y."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from io_util import unpack_rows
from problems import all_cases, as_f64_list, pack_column_major

M = 16
KMEANS_SEED = 0
MODELS = ("sgpr", "svgp")


def standardize_targets(y: list[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(y, dtype=np.float64).reshape(-1)
    mean = float(values.mean())
    var = float(np.mean((values - mean) ** 2))
    std = var**0.5
    if not np.isfinite(std) or std <= 0.0:
        std = 1.0
    return ((values - mean) / std).astype(np.float64)


def kmeans_z(coords: np.ndarray, m: int = M) -> np.ndarray:
    model = KMeans(n_clusters=m, random_state=KMEANS_SEED, n_init=10)
    model.fit(coords)
    return np.asarray(model.cluster_centers_, dtype=np.float64)


def all_sparse_cases() -> list[dict]:
    cases: list[dict] = []
    for base in all_cases():
        coords = unpack_rows(base["x"], base["n_rows"], base["n_cols"])
        z = kmeans_z(coords)
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
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for case in all_sparse_cases():
        path = out_dir / f"{case['name']}.json"
        path.write_text(json.dumps(case), encoding="utf-8")
        paths.append(path)
    return paths
