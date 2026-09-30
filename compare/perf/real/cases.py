"""One JSON case per (dataset, split): the arrays every runner reads."""

from __future__ import annotations

from pathlib import Path

import json

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


def case_n_rows(path: Path) -> int:
    """Training rows of a written case, from its small sidecar (the case
    itself is hundreds of MB for the large datasets)."""
    return int(json.loads(path.with_suffix(".size.json").read_text(encoding="utf-8"))["n_rows"])


def _write_size(path: Path, n_rows: int) -> None:
    path.with_suffix(".size.json").write_text(json.dumps({"n_rows": n_rows}), encoding="utf-8")


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
    _write_size(path, int(data.x_train.shape[0]))
    return path


def splits_of(dataset: str) -> range:
    return range(n_splits(DATASETS[dataset]))


def write_curve_case(name: str, protocol: str) -> Path:
    """A T0 case: the curve's training points, then (when the data has them)
    held-out points that are scored, then a dense grid; every runner returns
    its predictions on all of ``xs`` (``return_predictions``). Standardized
    with the training statistics (``x`` is only centred when its unit is
    meaningful, as years are)."""
    from .curves import CURVES, MAUNA_LOA_START

    curve = CURVES[name]()
    x_mean = curve.x_train.mean(axis=0)
    x_std = curve.x_train.std(axis=0) if curve.standardize_x else np.ones(1)
    y_mean, y_std = float(curve.y_train.mean()), float(curve.y_train.std())
    n_test = 0 if curve.x_test is None else int(curve.x_test.shape[0])
    points = curve.x_grid if curve.x_test is None else np.vstack([curve.x_test, curve.x_grid])
    ys = np.full(points.shape[0], y_mean)
    if curve.y_test is not None:
        ys[:n_test] = curve.y_test
    xs = (points - x_mean) / x_std
    extra: dict = {}
    if curve.kernel == "mauna_loa":
        st = MAUNA_LOA_START
        amp = lambda v: (v / y_std) ** 2  # noqa: E731  ppm -> variance of the standardized y
        extra = {
            "kernel": curve.kernel,
            "theta_init": [
                amp(st["s1"]), st["l1"], amp(st["s2"]), st["l2"], st["l3"], st["p"],
                amp(st["s3"]), st["l4"], st["alpha"], amp(st["s4"]), st["l5"], amp(st["noise"]),
            ],
        }
    path = CASES / f"{name}_s0_{protocol}.json"
    write_json(
        path,
        {
            **extra,
            "model": "exact",
            "name": f"{name}_s0",
            "dataset": name,
            "split": 0,
            "protocol": protocol,
            "return_predictions": True,
            "n_test": n_test,
            "n_rows": int(curve.x_train.shape[0]),
            "n_cols": 1,
            "x": ((curve.x_train - x_mean) / x_std).T.ravel().tolist(),
            "y": ((curve.y_train - y_mean) / y_std).tolist(),
            "xs_n_rows": int(xs.shape[0]),
            "xs": xs.T.ravel().tolist(),
            "ys": ys.tolist(),
            "y_mean": y_mean,
            "y_std": y_std,
            "x_mean": float(x_mean[0]),
            "x_std": float(x_std[0]),
            "lengthscale_init": LENGTHSCALE_INIT,
            "signal_variance_init": SIGNAL_VARIANCE_INIT,
            "noise_variance_init": NOISE_VARIANCE_INIT,
            "max_iterations": MATCHED_MAX_ITERATIONS,
            "gtol": MATCHED_GTOL,
        },
    )
    _write_size(path, int(curve.x_train.shape[0]))
    return path
