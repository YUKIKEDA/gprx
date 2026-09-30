"""Warmup + timed-rep helpers for the Python runners (P2B-16 clock)."""

from __future__ import annotations

import os
import statistics


def warmup_count() -> int:
    return max(0, int(os.environ.get("PERF_WARMUP", "1")))


def default_reps(n_rows: int) -> int:
    if n_rows <= 256:
        return 51
    if n_rows <= 1024:
        return 21
    return 7


def timed_reps(n_rows: int) -> int:
    raw = os.environ.get("PERF_REPS")
    if raw is not None and raw != "":
        return max(1, int(raw))
    return default_reps(n_rows)


def median(samples: list[float]) -> float:
    if not samples:
        raise ValueError("median of empty samples")
    return float(statistics.median(samples))


def min_max(samples: list[float]) -> tuple[float, float]:
    if not samples:
        raise ValueError("min_max of empty samples")
    return min(samples), max(samples)
