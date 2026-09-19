"""Shared Forrester / ARD-sphere cases for every compare/perf runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FORRESTER_NS = (256, 1024, 4096)
SPHERE_SIDES = (16, 32, 64)
QUERY_M = 100
NOISE_STD = 1.0
NOISE_VARIANCE_INIT = 0.1
ELL_ISO = 1.0
ELL_ARD = 4.0
FORRESTER_SEED = 0
SPHERE_SEED = 9
JOINT_EVALS = 10


def forrester(x: np.ndarray) -> np.ndarray:
    return (6.0 * x - 2.0) ** 2 * np.sin(12.0 * x - 4.0)


def weighted_sphere(coords: np.ndarray) -> np.ndarray:
    return (coords[:, 0] / 0.25) ** 2 + (coords[:, 1] / 1.0) ** 2


def pack_column_major(coords: np.ndarray) -> list[float]:
    n_rows, n_cols = coords.shape
    packed = np.empty(n_rows * n_cols, dtype=np.float64)
    for dim in range(n_cols):
        packed[dim * n_rows : (dim + 1) * n_rows] = coords[:, dim]
    return packed.tolist()


def as_f64_list(values: np.ndarray) -> list[float]:
    return np.asarray(values, dtype=np.float64).ravel().tolist()


def make_forrester(n: int) -> dict:
    x = np.linspace(0.0, 1.0, n, dtype=np.float64).reshape(-1, 1)
    noise = NOISE_STD * np.random.default_rng(FORRESTER_SEED).standard_normal(n)
    y = forrester(x[:, 0]) + noise
    xs = np.linspace(0.05, 0.95, QUERY_M, dtype=np.float64).reshape(-1, 1)
    return {
        "name": f"forrester_n{n}",
        "problem": "forrester",
        "ard": False,
        "n_rows": n,
        "n_cols": 1,
        "x": pack_column_major(x),
        "y": as_f64_list(y),
        "xs_n_rows": QUERY_M,
        "xs_n_cols": 1,
        "xs": pack_column_major(xs),
        "lengthscales_init": [ELL_ISO],
        "noise_variance_init": NOISE_VARIANCE_INIT,
        "joint_evals": JOINT_EVALS,
    }


def make_sphere(side: int) -> dict:
    grid = np.linspace(0.0, 1.0, side, dtype=np.float64)
    xx, yy = np.meshgrid(grid, grid, indexing="xy")
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    n_rows = int(coords.shape[0])
    noise = NOISE_STD * np.random.default_rng(SPHERE_SEED).standard_normal(n_rows)
    y = weighted_sphere(coords) + noise
    q = np.linspace(0.05, 0.95, 10, dtype=np.float64)
    qx, qy = np.meshgrid(q, q, indexing="xy")
    xs = np.column_stack([qx.ravel(), qy.ravel()])
    return {
        "name": f"sphere_n{n_rows}",
        "problem": "sphere",
        "ard": True,
        "n_rows": n_rows,
        "n_cols": 2,
        "x": pack_column_major(coords),
        "y": as_f64_list(y),
        "xs_n_rows": int(xs.shape[0]),
        "xs_n_cols": 2,
        "xs": pack_column_major(xs),
        "lengthscales_init": [ELL_ARD, ELL_ARD],
        "noise_variance_init": NOISE_VARIANCE_INIT,
        "joint_evals": JOINT_EVALS,
    }


def all_cases() -> list[dict]:
    cases = [make_forrester(n) for n in FORRESTER_NS]
    cases.extend(make_sphere(side) for side in SPHERE_SIDES)
    return cases


def write_cases(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for case in all_cases():
        path = out_dir / f"{case['name']}.json"
        path.write_text(json.dumps(case), encoding="utf-8")
        paths.append(path)
    return paths
