"""JSON case / result records of the Python runners."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def load_case(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_result(result: dict[str, Any]) -> None:
    """Writes one result row as the last stdout line, where the harness reads it."""
    sys.stdout.write(json.dumps(result, allow_nan=False))
    sys.stdout.write("\n")
