"""RMSE / NLPD / 95% coverage of a Gaussian predictive distribution, in the
original ``y`` units (``mean`` and ``var`` are standardized, ``var`` is the
observation variance)."""

from __future__ import annotations

import math

import numpy as np

Z95 = 1.959963984540054


def score(
    mean: np.ndarray, var: np.ndarray, y_test: np.ndarray, y_mean: float, y_std: float
) -> dict[str, float]:
    mu = mean * y_std + y_mean
    variance = var * y_std * y_std
    err = y_test - mu
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "nlpd": float(np.mean(0.5 * np.log(2.0 * math.pi * variance) + 0.5 * err**2 / variance)),
        "coverage95": float(np.mean(np.abs(err) <= Z95 * np.sqrt(variance))),
    }


def case_metrics(case: dict, mean, var) -> dict:
    """What a runner adds to its row once it has predicted ``case["xs"]``.

    The scores use the first ``n_test`` points (all of them unless the case
    says otherwise; none for a curve grid). ``return_predictions`` also returns
    the mean and variance of every point, in original units.
    """
    mean, var = np.asarray(mean).ravel(), np.asarray(var).ravel()
    out: dict = {}
    n_test = int(case.get("n_test", mean.shape[0]))
    if n_test > 0:
        out.update(
            score(mean[:n_test], var[:n_test], np.asarray(case["ys"])[:n_test], case["y_mean"], case["y_std"])
        )
    if case.get("return_predictions"):
        out["pred_mean"] = (mean * case["y_std"] + case["y_mean"]).tolist()
        out["pred_var"] = (var * case["y_std"] ** 2).tolist()
    return out
