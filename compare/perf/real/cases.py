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


#: Inducing points and the SVGP Adam setting (Adam has no library default that
#: suits n in the hundreds of thousands, so SVGP runs one shared setting).
N_INDUCING = 512
ADAM_LR = 0.01
ADAM_BATCH_SIZE = 1024
ADAM_EPOCHS = 3
KMEANS_SEED = 0
KMEANS_SUBSAMPLE = 100_000


def case_path(dataset: str, split: int, protocol: str, model: str = "exact") -> Path:
    tag = "" if model == "exact" else f"_{model}"
    return CASES / f"{dataset}{tag}_s{split}_{protocol}.json"


def inducing_points(x: np.ndarray, m: int) -> np.ndarray:
    """k-means centres (seed fixed) of the standardized training inputs; a
    subsample of at most ``KMEANS_SUBSAMPLE`` rows when n is larger."""
    from sklearn.cluster import MiniBatchKMeans

    rng = np.random.default_rng(KMEANS_SEED)
    if x.shape[0] > KMEANS_SUBSAMPLE:
        x = x[rng.choice(x.shape[0], KMEANS_SUBSAMPLE, replace=False)]
    model = MiniBatchKMeans(n_clusters=m, random_state=KMEANS_SEED, n_init=1, batch_size=4096)
    return model.fit(x).cluster_centers_


def write_case(
    dataset: str, split: int, protocol: str, model: str = "exact", n_inducing: int = N_INDUCING
) -> Path:
    data = load_split(DATASETS[dataset], split)
    path = case_path(dataset, split, protocol, model)
    extra: dict = {}
    if model != "exact":
        z = inducing_points(data.x_train, n_inducing)
        extra = {
            "z": z.T.ravel().tolist(),
            "n_inducing": int(z.shape[0]),
            "adam_lr": ADAM_LR,
            "adam_batch_size": ADAM_BATCH_SIZE,
            "adam_epochs": ADAM_EPOCHS,
        }
    write_json(
        path,
        {
            **extra,
            "model": model,
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
