"""One JSON case per (dataset, split): the arrays every runner reads."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from common.harness import write_json

from .data import DATASETS, OUT, load_split, n_splits

CASES = OUT / "cases"

#: Start point every library shares (X and y are standardized).
LENGTHSCALE_INIT = 1.0
SIGNAL_VARIANCE_INIT = 1.0
NOISE_VARIANCE_INIT = 0.1
#: The `matched` protocol (same iteration cap and gradient tolerance everywhere).
MATCHED_MAX_ITERATIONS = 100
MATCHED_GTOL = float(np.sqrt(np.finfo(np.float64).eps))


def case_path(dataset: str, split: int, protocol: str) -> Path:
    return CASES / f"{dataset}_s{split}_{protocol}.json"


def write_case(dataset: str, split: int, protocol: str) -> Path:
    data = load_split(DATASETS[dataset], split)
    path = case_path(dataset, split, protocol)
    write_json(
        path,
        {
            "name": f"{dataset}_s{split}",
            "dataset": dataset,
            "split": split,
            "protocol": protocol,
            "n_rows": int(data.x_train.shape[0]),
            "n_cols": int(data.x_train.shape[1]),
            "x": data.x_train.T.ravel().tolist(),
            "y": data.y_train.tolist(),
            "xs_n_rows": int(data.x_test.shape[0]),
            "xs": data.x_test.T.ravel().tolist(),
            "ys": data.y_test.tolist(),
            "y_mean": data.y_mean,
            "y_std": data.y_std,
            "lengthscale_init": LENGTHSCALE_INIT,
            "signal_variance_init": SIGNAL_VARIANCE_INIT,
            "noise_variance_init": NOISE_VARIANCE_INIT,
            "max_iterations": MATCHED_MAX_ITERATIONS,
            "gtol": MATCHED_GTOL,
        },
    )
    return path


def splits_of(dataset: str) -> range:
    return range(n_splits(DATASETS[dataset]))
