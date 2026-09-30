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
