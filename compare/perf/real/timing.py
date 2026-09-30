"""Untimed warm-up fits before the one timed fit of a cell."""

from __future__ import annotations

import os


def warmup_fits(n_rows: int) -> int:
    """`PERF_WARMUP` when set; otherwise one warm-up for small problems (the
    first call pays thread-pool and allocator start-up) and none above
    n = 5000, where the start-up is negligible next to the fit."""
    raw = os.environ.get("PERF_WARMUP")
    if raw is not None and raw != "":
        return max(0, int(raw))
    return 1 if n_rows <= 5000 else 0
