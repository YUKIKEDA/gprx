"""T0: one-dimensional data whose fit is checked by eye (B1-1).

Snelson (200 points, the sparse-GP sanity check): the archive is fetched from
the author's page, which not every network reaches; then ``SNELSON_ZIP`` may
point at a local copy. The fetched file is pinned in ``checksums.json``.
"""

from __future__ import annotations

import io
import json
import os
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import CHECKSUMS, DATA, _pins, _sha256

SNELSON_URL = "http://www.gatsby.ucl.ac.uk/~snelson/SPGP_dist.zip"
SNELSON_REL = "snelson/SPGP_dist.zip"


@dataclass
class Curve:
    x_train: np.ndarray  # (n, 1), raw units
    y_train: np.ndarray
    x_grid: np.ndarray  # (g, 1), raw units, where the curve is drawn


def snelson_archive() -> Path:
    override = os.environ.get("SNELSON_ZIP")
    path = Path(override) if override else DATA / SNELSON_REL
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(SNELSON_URL, timeout=120) as response:
            path.write_bytes(response.read())
    pinned = _pins().get(SNELSON_REL)
    digest = _sha256(path)
    if pinned is None:
        # First fetch on this checkout: pin what was downloaded.
        pins = _pins()
        pins[SNELSON_REL] = digest
        CHECKSUMS.write_text(json.dumps(pins, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"pinned {SNELSON_REL} ({digest[:12]}…): commit checksums.json")
    elif digest != pinned:
        raise RuntimeError(f"checksum mismatch for {SNELSON_REL}: {digest} != {pinned}")
    return path


def _column(archive: zipfile.ZipFile, stem: str) -> np.ndarray:
    name = next(n for n in archive.namelist() if Path(n).name == stem)
    return np.loadtxt(io.StringIO(archive.read(name).decode()))


def snelson() -> Curve:
    with zipfile.ZipFile(snelson_archive()) as archive:
        x = _column(archive, "train_inputs").reshape(-1, 1)
        y = _column(archive, "train_outputs")
        grid = _column(archive, "test_inputs").reshape(-1, 1)
    return Curve(x, y, grid)


CURVES = {"snelson": snelson}
