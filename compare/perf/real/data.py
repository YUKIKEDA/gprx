"""Fetching the benchmark data and cutting the splits.

T1 / T2 (``yaringal``): the fixed Hernández-Lobato & Adams splits. Data are
downloaded once into ``out/real/data`` and checked against
``checksums.json`` (pin a new file with ``python -m perf.real.data --pin``).
Every split is standardized with the training statistics, as in the original
experiments; metrics are converted back to the original ``y`` units.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "out" / "real"
DATA = OUT / "data"
CHECKSUMS = HERE / "checksums.json"

YARIN = "https://raw.githubusercontent.com/yaringal/DropoutUncertaintyExps/master/UCI_Datasets"
TREVANS = "https://raw.githubusercontent.com/treforevans/uci_datasets/master/uci_datasets"


@dataclass(frozen=True)
class Dataset:
    name: str
    remote: str  # directory under the source
    tier: str
    source: str = "yaringal"  # or "trevans" (treforevans/uci_datasets: 10 splits of 90 / 10)
    files: tuple[str, ...] = ("data.csv.gz", "test_mask.csv.gz")


#: Boston is left out (removed from scikit-learn 1.2 for its features).
DATASETS: dict[str, Dataset] = {
    d.name: d
    for d in (
        Dataset("yacht", "yacht", "T1"),
        Dataset("energy", "energy", "T1"),
        Dataset("concrete", "concrete", "T1"),
        Dataset("wine_red", "wine-quality-red", "T1"),
        Dataset("power_plant", "power-plant", "T1"),
        Dataset("kin8nm", "kin8nm", "T1"),
        Dataset("naval", "naval-propulsion-plant", "T1"),
        Dataset("protein", "protein-tertiary-structure", "T2"),
        Dataset("kin40k", "kin40k", "T2", "trevans"),
        Dataset("3droad", "3droad", "T3", "trevans"),
        Dataset("song", "song", "T3", "trevans", ("data.csv.gz", "data1.csv.gz", "test_mask.csv.gz")),
        Dataset("buzz", "buzz", "T3", "trevans"),
        Dataset("houseelectric", "houseelectric", "T3", "trevans"),
    )
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pins() -> dict[str, str]:
    if CHECKSUMS.is_file():
        return json.loads(CHECKSUMS.read_text(encoding="utf-8"))
    return {}


def _download(url: str, dest: Path, attempts: int = 8) -> None:
    """``url`` into ``dest``, resuming with a Range request after a cut
    connection (a proxy may drop a long transfer half way); written as
    ``.part`` and renamed once the size matches ``Content-Length``."""
    import http.client
    import urllib.error

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    part.unlink(missing_ok=True)
    total: int | None = None
    for attempt in range(attempts):
        have = part.stat().st_size if part.exists() else 0
        request = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if have and response.status != 206:  # the server ignored Range: start over
                    have = 0
                    part.unlink(missing_ok=True)
                length = response.headers.get("Content-Length")
                if total is None and length is not None:
                    total = have + int(length)
                with part.open("ab" if have else "wb") as handle:
                    while chunk := response.read(1 << 20):
                        handle.write(chunk)
        except (http.client.IncompleteRead, urllib.error.URLError, ConnectionError, TimeoutError) as err:
            print(f"download of {url} cut ({type(err).__name__}); attempt {attempt + 1}/{attempts}", file=sys.stderr)
            continue
        if total is None or part.stat().st_size == total:
            part.rename(dest)
            return
    raise RuntimeError(f"could not download {url} completely")


def fetch_file(dataset: Dataset, filename: str) -> Path:
    """The local copy of one file of ``dataset`` (downloaded when missing,
    verified against the pinned checksum when one exists)."""
    rel = f"{dataset.source}/{dataset.remote}/{filename}"
    path = DATA / rel
    if not path.is_file():
        url = (
            f"{YARIN}/{dataset.remote}/data/{filename}"
            if dataset.source == "yaringal"
            else f"{TREVANS}/{dataset.remote}/{filename}"
        )
        _download(url, path)
    pinned = _pins().get(rel)
    if pinned is not None and _sha256(path) != pinned:
        raise RuntimeError(f"checksum mismatch for {rel}: delete {path} and fetch again")
    return path


def n_splits(dataset: Dataset) -> int:
    if dataset.source == "trevans":
        return 10
    return int(fetch_file(dataset, "n_splits.txt").read_text().split()[0])


def _ints(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=np.int64, ndmin=1)


@dataclass
class Split:
    x_train: np.ndarray  # standardized, (n, d)
    y_train: np.ndarray  # standardized, (n,)
    x_test: np.ndarray
    y_test: np.ndarray  # original units
    y_mean: float
    y_std: float


def _trevans_arrays(dataset: Dataset, split: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``x``, ``y`` and the train / test row indices of one split. The parsed
    table is cached as ``.npy`` (parsing a 2M-row csv is slow)."""
    cache = DATA / dataset.source / dataset.remote / "data.npy"
    if not cache.is_file():
        tables = [np.loadtxt(fetch_file(dataset, f), delimiter=",") for f in dataset.files if f.startswith("data")]
        np.save(cache, np.concatenate(tables, axis=0))
    data = np.load(cache)
    mask = np.loadtxt(fetch_file(dataset, "test_mask.csv.gz"), dtype=bool, delimiter=",")
    test = np.flatnonzero(mask[:, split])
    train = np.flatnonzero(~mask[:, split])
    return data[:, :-1], data[:, -1], train, test


def load_split(dataset: Dataset, split: int) -> Split:
    if dataset.source == "trevans":
        x, y, train, test = _trevans_arrays(dataset, split)
    else:
        data = np.loadtxt(fetch_file(dataset, "data.txt"), ndmin=2)
        features = _ints(fetch_file(dataset, "index_features.txt"))
        target = int(_ints(fetch_file(dataset, "index_target.txt"))[0])
        train = _ints(fetch_file(dataset, f"index_train_{split}.txt"))
        test = _ints(fetch_file(dataset, f"index_test_{split}.txt"))
        x, y = data[:, features], data[:, target]
    x_train, y_train, x_test, y_test = x[train], y[train], x[test], y[test]
    mean, std = x_train.mean(axis=0), x_train.std(axis=0)
    std[std == 0.0] = 1.0
    y_mean, y_std = float(y_train.mean()), float(y_train.std())
    return Split(
        x_train=(x_train - mean) / std,
        y_train=(y_train - y_mean) / y_std,
        x_test=(x_test - mean) / std,
        y_test=y_test,
        y_mean=y_mean,
        y_std=y_std,
    )


def pin_all() -> None:
    """Downloads every file of every dataset and writes ``checksums.json``."""
    pins = _pins()  # keep the pins of the curves (Mauna Loa, Snelson)
    for dataset in DATASETS.values():
        if dataset.source == "trevans":
            files = list(dataset.files)
        else:
            files = ["data.txt", "index_features.txt", "index_target.txt", "n_splits.txt"]
            for i in range(n_splits(dataset)):
                files += [f"index_train_{i}.txt", f"index_test_{i}.txt"]
        for filename in files:
            path = fetch_file(dataset, filename)
            pins[f"{dataset.source}/{dataset.remote}/{filename}"] = _sha256(path)
    CHECKSUMS.write_text(json.dumps(pins, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"pinned {len(pins)} files in {CHECKSUMS}")


if __name__ == "__main__":
    if "--pin" in sys.argv[1:]:
        pin_all()
    else:
        print("usage: python -m perf.real.data --pin", file=sys.stderr)
        raise SystemExit(2)
