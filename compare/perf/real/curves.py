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
    x_test: np.ndarray | None = None  # held-out points, scored
    y_test: np.ndarray | None = None
    standardize_x: bool = True  # False keeps the unit (years) and only centres x
    kernel: str | None = None  # "mauna_loa": the composite kernel


def fetch_pinned(rel: str, url: str, override: Path | None = None) -> Path:
    """``url`` saved as ``out/real/data/<rel>`` (or ``override``, a local copy),
    checked against ``checksums.json``; the first fetch pins what it got."""
    path = override or DATA / rel
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as response:
            path.write_bytes(response.read())
    pinned = _pins().get(rel)
    digest = _sha256(path)
    if pinned is None:
        pins = _pins()
        pins[rel] = digest
        CHECKSUMS.write_text(json.dumps(pins, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"pinned {rel} ({digest[:12]}…): commit checksums.json")
    elif digest != pinned:
        raise RuntimeError(f"checksum mismatch for {rel}: {digest} != {pinned}")
    return path


def snelson_archive() -> Path:
    override = os.environ.get("SNELSON_ZIP")
    return fetch_pinned(SNELSON_REL, SNELSON_URL, Path(override) if override else None)


def _column(archive: zipfile.ZipFile, stem: str) -> np.ndarray:
    name = next(n for n in archive.namelist() if Path(n).name == stem)
    return np.loadtxt(io.StringIO(archive.read(name).decode()))


def snelson() -> Curve:
    with zipfile.ZipFile(snelson_archive()) as archive:
        x = _column(archive, "train_inputs").reshape(-1, 1)
        y = _column(archive, "train_outputs")
        grid = _column(archive, "test_inputs").reshape(-1, 1)
    return Curve(x, y, grid)


MAUNA_LOA_URL = "https://raw.githubusercontent.com/datasets/co2-ppm/main/data/co2-mm-mlo.csv"
MAUNA_LOA_REL = "co2-ppm/co2-mm-mlo.csv"
#: Rasmussen & Williams train on 1958-2003 and extrapolate.
MAUNA_LOA_SPLIT_YEAR = 2004.0


def maunaloa() -> Curve:
    """Monthly mean CO₂ [ppm] (NOAA, through ``datasets/co2-ppm``): train
    before 2004, score on everything after, draw the curve to the last month."""
    rows = []
    for line in fetch_pinned(MAUNA_LOA_REL, MAUNA_LOA_URL).read_text().splitlines()[1:]:
        fields = line.split(",")
        year, average = float(fields[1]), float(fields[2])
        if average > 0.0:  # -99.99 marks a missing month
            rows.append((year, average))
    data = np.asarray(rows)
    train = data[:, 0] < MAUNA_LOA_SPLIT_YEAR
    grid = np.linspace(data[0, 0], data[-1, 0] + 1.0, 2000).reshape(-1, 1)
    return Curve(
        data[train, :1], data[train, 1], grid,
        data[~train, :1], data[~train, 1],
        standardize_x=False, kernel="mauna_loa",
    )


#: R&W §5.4.3 starting point in ppm / years; `cases.py` rescales the amplitudes
#: to the standardized y.
MAUNA_LOA_START = {
    "s1": 66.0, "l1": 67.0, "s2": 2.4, "l2": 90.0, "l3": 1.3, "p": 1.0,
    "s3": 0.66, "l4": 1.2, "alpha": 0.78, "s4": 0.18, "l5": 1.6 / 12.0, "noise": 0.19,
}

CURVES = {"snelson": snelson, "maunaloa": maunaloa}
