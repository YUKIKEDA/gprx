"""JSON case / result helpers for the Python runners."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def load_case(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def unpack_rows(packed: list[float], n_rows: int, n_cols: int) -> np.ndarray:
    colmaj = np.asarray(packed, dtype=np.float64)
    return colmaj.reshape((n_cols, n_rows), order="C").T.copy()


def write_result(result: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(result, allow_nan=False))
    sys.stdout.write("\n")
