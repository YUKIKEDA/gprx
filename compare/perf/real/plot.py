"""SVG figures for the README from ``out/real/results.json`` and the RSS
timelines (``--timeline``).

```text
python -m perf.real.plot [--out DIR] [--protocol native|matched] [--timeline-case PREFIX]
```

Palette: the validated categorical order (blue, orange, aqua, yellow,
magenta) on the light chart surface. Three of those fall under 3:1 contrast
there, so every mark also has its own marker shape, the legend is always
present, and the README carries the numbers as tables.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .data import OUT  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e1"
#: Fixed slot per library: the entity keeps its color in every figure.
STYLE = {
    "gprx": ("#2a78d6", "o"),
    "sklearn": ("#eb6834", "s"),
    "gpytorch": ("#1baf7a", "^"),
    "gpy": ("#eda100", "D"),
    "libgp": ("#e87ba4", "v"),
}
LABEL = {"gprx": "gprx", "sklearn": "scikit-learn", "gpytorch": "GPyTorch", "gpy": "GPy", "libgp": "libgp"}

plt.rcParams.update(
    {
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "text.color": INK,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    }
)


def _style_axes(ax) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)


def _legend(fig, libs: list[str]) -> None:
    handles = [
        plt.Line2D([], [], marker=STYLE[l][1], color=STYLE[l][0], linestyle="", markersize=6,
                   markeredgecolor=SURFACE, label=LABEL[l])
        for l in libs
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(libs), frameon=False,
               labelcolor=INK_2, bbox_to_anchor=(0.5, 1.0))


def _mean_se(values: list[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, 0.0
    return mean, statistics.stdev(values) / math.sqrt(len(values))


def _table(rows: list[dict], protocol: str, model: str = "exact") -> dict[str, dict[str, dict[str, tuple[float, float]]]]:
    """dataset -> lib -> metric -> (mean, se) over the ok splits."""
    cells: dict[str, dict[str, list[dict]]] = {}
    for row in rows:
        if row.get("status") != "ok" or row["protocol"] != protocol:
            continue
        if row.get("model", "exact") != model:
            continue
        dataset = row["name"].rsplit("_s", 1)[0]
        cells.setdefault(dataset, {}).setdefault(row["lib"], []).append(row)
    out: dict = {}
    for dataset, libs in cells.items():
        for lib, group in libs.items():
            out.setdefault(dataset, {})[lib] = {
                key: _mean_se([r[key] for r in group if r.get(key) is not None])
                for key in ("rmse", "nlpd", "fit_s", "joint_evals")
                if any(r.get(key) is not None for r in group)
            }
    return out


def _dots(axes, table, datasets, libs, metric, ylabel, log=False) -> None:
    for ax, dataset in zip(axes, datasets):
        _style_axes(ax)
        for i, lib in enumerate(libs):
            cell = table.get(dataset, {}).get(lib, {}).get(metric)
            if cell is None:
                continue
            color, marker = STYLE[lib]
            ax.errorbar(i, cell[0], yerr=cell[1], color=color, marker=marker, markersize=6,
                        markeredgecolor=SURFACE, markeredgewidth=1.2, linewidth=1.6, capsize=0)
        if log:
            ax.set_yscale("log")
        ax.set_xlim(-0.6, len(libs) - 0.4)
        ax.set_xticks([])
        ax.set_title(dataset, fontsize=9, color=INK)
    axes[0].set_ylabel(ylabel)


def _suffix(model: str) -> str:
    return "" if model == "exact" else f"_{model}"


def accuracy(rows: list[dict], protocol: str, out: Path, model: str = "exact") -> Path | None:
    table = _table(rows, protocol, model)
    # A curve without held-out points (Snelson) has nothing to score.
    datasets = sorted(d for d in table if any("rmse" in cell for cell in table[d].values()))
    if not datasets:
        return None
    libs = [l for l in STYLE if any(l in table[d] for d in datasets)]
    fig, axes = plt.subplots(2, len(datasets), figsize=(max(1.9 * len(datasets) + 0.6, 6.6), 4.6), squeeze=False)
    _dots(axes[0], table, datasets, libs, "rmse", "RMSE (lower is better)")
    _dots(axes[1], table, datasets, libs, "nlpd", "NLPD (lower is better)")
    _legend(fig, libs)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = out / f"accuracy{_suffix(model)}_{protocol}.svg"
    fig.savefig(path)
    plt.close(fig)
    return path


def fit_time(rows: list[dict], protocol: str, out: Path, model: str = "exact") -> Path | None:
    table = _table(rows, protocol, model)
    datasets = sorted(table)
    if not datasets:
        return None
    libs = [l for l in STYLE if any(l in table[d] for d in datasets)]
    fig, axes = plt.subplots(2, len(datasets), figsize=(max(1.9 * len(datasets) + 0.6, 6.6), 4.6), squeeze=False)
    _dots(axes[0], table, datasets, libs, "fit_s", "fit wall time [s], log", log=True)
    _dots(axes[1], table, datasets, libs, "joint_evals", "joint MLL+grad evaluations", log=True)
    _legend(fig, libs)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = out / f"fit_time{_suffix(model)}_{protocol}.svg"
    fig.savefig(path)
    plt.close(fig)
    return path


def timeline_groups(directory: Path) -> list[str]:
    """The prefixes of the timeline files there (``<dataset>_<model>_s<k>_<protocol>``)."""
    found = set()
    for file in directory.glob("*.json") if directory.is_dir() else []:
        for lib in STYLE:
            if file.stem.endswith(f"_{lib}"):
                found.add(file.stem[: -len(lib) - 1])
    return sorted(found)


def rss_timeline(case: str, out: Path, directory: Path) -> Path | None:
    files = {l: directory / f"{case}_{l}.json" for l in STYLE}
    files = {l: f for l, f in files.items() if f.is_file()}
    if not files:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    _style_axes(ax)
    for lib, file in files.items():
        data = json.loads(file.read_text(encoding="utf-8"))
        color, _ = STYLE[lib]
        live = [(t, r) for t, r in data["samples"] if r > 0]  # after exit the tree reads 0
        times = [t for t, _ in live]
        mib = [r / 2**20 for _, r in live]
        ax.plot(times, mib, color=color, linewidth=1.6, label=LABEL[lib])
        for t, name in data["phases"]:
            if name in ("fit", "predict"):
                ax.axvline(t, color=color, linewidth=0.8, linestyle=":", alpha=0.8)
    ax.set_xlabel("time since process start [s]  (dotted: start of fit / predict)")
    ax.set_ylabel("RSS of the process tree [MiB]")
    ax.legend(frameon=False, labelcolor=INK_2, loc="upper left")
    fig.tight_layout()
    path = out / f"rss_timeline_{case}.svg"
    fig.savefig(path)
    plt.close(fig)
    return path


def curve(name: str, protocol: str, out: Path) -> Path | None:
    """T0: every library's predictive mean and 95% band on the grid, over the
    training points."""
    import numpy as np

    from .curves import CURVES

    rows = [
        r
        for r in json.loads((OUT / "results.json").read_text(encoding="utf-8"))
        if r.get("status") == "ok" and r["name"] == f"{name}_s0" and r["protocol"] == protocol
        and r.get("pred_mean") is not None
    ]
    if not rows:
        return None
    data = CURVES[name]()
    grid = data.x_grid.ravel()
    expected = grid.shape[0] + (0 if data.x_test is None else data.x_test.shape[0])
    stale = [r["lib"] for r in rows if len(r["pred_mean"]) != expected]
    if stale:  # measured with another grid: re-run before plotting
        print(f"curve {name}: {stale} were measured on a different grid; skipped", file=sys.stderr)
    rows = [r for r in rows if len(r["pred_mean"]) == expected]
    if not rows:
        return None
    fig, axes = plt.subplots(1, len(rows), figsize=(max(2.6 * len(rows), 6.6), 3.0), squeeze=False, sharey=True)
    for ax, row in zip(axes[0], rows):
        _style_axes(ax)
        color, _ = STYLE[row["lib"]]
        mean = np.asarray(row["pred_mean"])[-grid.shape[0] :]  # after the scored points
        sd = np.sqrt(np.asarray(row["pred_var"])[-grid.shape[0] :])
        ax.fill_between(grid, mean - 1.96 * sd, mean + 1.96 * sd, color=color, alpha=0.18, linewidth=0)
        ax.plot(grid, mean, color=color, linewidth=1.6)
        ax.scatter(data.x_train.ravel(), data.y_train, s=6, color=INK_2, alpha=0.7, linewidths=0)
        if data.x_test is not None:  # held-out points, hollow
            ax.scatter(data.x_test.ravel(), data.y_test, s=8, facecolors="none", edgecolors=INK,
                       linewidths=0.6, alpha=0.8)
        ax.set_title(LABEL[row["lib"]], fontsize=9, color=INK)
    fig.tight_layout()
    path = out / f"curve_{name}_{protocol}.svg"
    fig.savefig(path)
    plt.close(fig)
    return path


def main(argv: list[str]) -> int:
    def option(flag: str, default: str) -> str:
        return argv[argv.index(flag) + 1] if flag in argv else default

    out = Path(option("--out", str(OUT / "plots")))
    out.mkdir(parents=True, exist_ok=True)
    protocol = option("--protocol", "native")
    rows = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    made = [accuracy(rows, protocol, out), fit_time(rows, protocol, out)]
    if "--curve" in argv:
        made.append(curve(option("--curve", ""), protocol, out))
    if "--timeline-case" in argv:
        made.append(rss_timeline(option("--timeline-case", ""), out, OUT / "timeline"))
    for path in made:
        print(path if path else "nothing to plot")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
