"""Snelson figures for the comparison doc: a sparse fit, and an online-insert GIF.

Not a CI gate. Does not write ``out/real/results.json``.

```text
python -m perf.real.visual
```

Sparse: ``Sgpr`` against GPyTorch and GPy, matched L-BFGS, 8 fixed inducing
points (k-means, seed 0). Online: points added from the smallest input,
hyperparameters held at the shared start. gprx inserts, libgp calls
``add_pattern``, GPyTorch calls ``get_fantasy_model``.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from ..runners import PERF, REPO, ensure_libgp
from .cases import LENGTHSCALE_INIT, NOISE_VARIANCE_INIT, SIGNAL_VARIANCE_INIT, write_curve_case
from .curves import snelson
from .libs import SPARSE_RUNNERS
from .optimizers import meta
from .plot import FIG_H, FIG_W, INK, INK_2, LABEL, STYLE, _style_axes

BENCH = REPO / "docs" / "bench"
SPARSE_M = 8
ONLINE_GRID = 160
GIF_FRAMES = 24
SPARSE_LIBS = ("gprx", "gpytorch", "gpy")
ONLINE_LIBS = ("gprx", "libgp", "gpytorch")


def _cargo_frames(case_path: Path) -> dict:
    proc = subprocess.run(
        [
            "cargo", "run", "--release", "--quiet",
            "--manifest-path", str(PERF / "gprx" / "Cargo.toml"),
            "--features", "fit-counts",  # the runner binary's fit module needs this feature to build
            "--", "online-frames", str(case_path),
        ],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "gprx online-frames failed").strip())
    return json.loads(proc.stdout)


def _libgp_frames(case_path: Path) -> dict:
    exe = ensure_libgp("libgp-online")
    if not isinstance(exe, Path):
        raise RuntimeError(str(exe.get("note", "libgp-online build failed")))
    proc = subprocess.run(
        [str(exe), "golden", str(case_path)],
        cwd=PERF,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "libgp golden failed").strip())
    return json.loads(proc.stdout)


def _gpytorch_frames(x: np.ndarray, y: np.ndarray, xs: np.ndarray) -> dict:
    """Fixed hyperparameters. Each step adds one training point."""
    import gpytorch
    import torch
    from gpytorch.settings import fast_pred_var, max_cholesky_size

    from .gpytorch_fit import build

    torch.set_default_dtype(torch.float64)
    case = {
        "noise_variance_init": NOISE_VARIANCE_INIT,
        "lengthscale_init": LENGTHSCALE_INIT,
        "signal_variance_init": SIGNAL_VARIANCE_INIT,
        "protocol": "fixed",
    }
    xt = torch.as_tensor(x)
    yt = torch.as_tensor(y)
    xst = torch.as_tensor(xs)
    model, likelihood = build(case, xt[:2], yt[:2])
    steps = []
    cap = max(int(xt.shape[0]), int(xst.shape[0]))
    with max_cholesky_size(cap), fast_pred_var(False):
        def snap(n: int, current) -> None:
            current.eval()
            likelihood.eval()
            with torch.no_grad():
                pred = likelihood(current(xst))
                steps.append({
                    "n": n,
                    "mean": pred.mean.detach().cpu().numpy().tolist(),
                    "observation_variance": pred.variance.detach().cpu().numpy().tolist(),
                })

        snap(2, model)
        for i in range(2, xt.shape[0]):
            model = model.get_fantasy_model(xt[i : i + 1], yt[i : i + 1])
            snap(i + 1, model)
    return {"source": "gpytorch", "steps": steps}


def _band(ax, grid, mean, var, color: str) -> None:
    sd = np.sqrt(np.maximum(var, 0.0))
    ax.fill_between(grid, mean - 1.96 * sd, mean + 1.96 * sd, color=color, alpha=0.18, linewidth=0)
    ax.plot(grid, mean, color=color, linewidth=1.6)


def sparse_figure(case: dict, rows: list[dict], train_x: np.ndarray, train_y: np.ndarray, grid: np.ndarray) -> Path:
    rows = [r for r in rows if r.get("status") == "ok" and r.get("pred_mean")]
    missing = [lib for lib in SPARSE_LIBS if lib not in {r["lib"] for r in rows}]
    if missing:
        raise RuntimeError(f"sparse figure missing {missing}")
    order = {lib: i for i, lib in enumerate(SPARSE_LIBS)}
    rows.sort(key=lambda r: order[r["lib"]])
    z = np.asarray(case["z"], dtype=float) * case["x_std"] + case["x_mean"]
    fig, axes = plt_axes(len(rows))
    for ax, row in zip(axes, rows):
        _style_axes(ax)
        color, _ = STYLE[row["lib"]]
        mean = np.asarray(row["pred_mean"], dtype=float)
        var = np.asarray(row["pred_var"], dtype=float)
        _band(ax, grid, mean, var, color)
        ax.scatter(train_x, train_y, s=6, color=INK_2, alpha=0.7, linewidths=0)
        for tick in z:
            ax.axvline(tick, color=INK, linewidth=0.6, alpha=0.35)
        ax.set_title(LABEL[row["lib"]], fontsize=9, color=INK)
    fig.tight_layout()
    path = BENCH / "curve_snelson_sgpr.svg"
    fig.savefig(path)
    plt_close(fig)
    return path


def plt_axes(n: int):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, n, figsize=(FIG_W, FIG_H), squeeze=False, sharey=True)
    return fig, axes[0]


def plt_close(fig) -> None:
    import matplotlib.pyplot as plt

    plt.close(fig)


def _stride(n_steps: int) -> list[int]:
    if n_steps <= GIF_FRAMES:
        return list(range(n_steps))
    picked = np.linspace(0, n_steps - 1, GIF_FRAMES).astype(int)
    return sorted(set(picked.tolist()))


def online_gif(
    dumps: dict[str, dict],
    train_x: np.ndarray,
    train_y: np.ndarray,
    grid: np.ndarray,
    y_mean: float,
    y_std: float,
) -> Path:
    from PIL import Image

    import matplotlib.pyplot as plt

    steps = {lib: {int(s["n"]): s for s in dump["steps"]} for lib, dump in dumps.items()}
    counts = sorted(set.intersection(*(set(s) for s in steps.values())))
    if not counts:
        raise RuntimeError("online libraries returned no shared n")
    shown = [counts[i] for i in _stride(len(counts))]
    frames = []
    # Limits from every shown frame, so the axis does not jump.
    lows, highs = [float(train_y.min())], [float(train_y.max())]
    for n in shown:
        for lib in ONLINE_LIBS:
            step = steps[lib][n]
            mean = np.asarray(step["mean"]) * y_std + y_mean
            sd = np.sqrt(np.maximum(np.asarray(step["observation_variance"]), 0.0)) * y_std
            lows.append(float(np.min(mean - 1.96 * sd)))
            highs.append(float(np.max(mean + 1.96 * sd)))
    pad = 0.05 * (max(highs) - min(lows))
    ymin, ymax = min(lows) - pad, max(highs) + pad
    for n in shown:
        fig, axes = plt.subplots(1, len(ONLINE_LIBS), figsize=(9.2, 3.3), squeeze=False, sharey=True)
        for ax, lib in zip(axes[0], ONLINE_LIBS):
            _style_axes(ax)
            step = steps[lib][n]
            mean = np.asarray(step["mean"]) * y_std + y_mean
            var = np.asarray(step["observation_variance"]) * y_std * y_std
            _band(ax, grid, mean, var, STYLE[lib][0])
            ax.scatter(train_x[:n], train_y[:n], s=8, color=INK_2, alpha=0.8, linewidths=0)
            ax.set_ylim(ymin, ymax)
            ax.set_title(f"{LABEL[lib]}   n = {n}", fontsize=9, color=INK)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB"))
    path = BENCH / "online_snelson.gif"
    # One palette for every frame, so the bands do not flicker.
    base = frames[0].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    rest = [frame.quantize(palette=base, dither=Image.Dither.NONE) for frame in frames[1:]]
    base.save(path, save_all=True, append_images=rest, duration=140, loop=0, optimize=True)
    return path


def _online_case(train_x: np.ndarray, train_y: np.ndarray, grid: np.ndarray) -> dict:
    x_mean, x_std = float(train_x.mean()), float(train_x.std())
    y_mean, y_std = float(train_y.mean()), float(train_y.std())
    x_std = x_std if x_std > 0.0 else 1.0
    y_std = y_std if y_std > 0.0 else 1.0
    return {
        "name": "snelson_online",
        "ard": False,
        "n_rows": int(train_x.shape[0]),
        "n_cols": 1,
        "x": ((train_x - x_mean) / x_std).tolist(),
        "y": ((train_y - y_mean) / y_std).tolist(),
        "xs_n_rows": int(grid.shape[0]),
        "xs_n_cols": 1,
        "xs": ((grid - x_mean) / x_std).tolist(),
        "lengthscales_init": [LENGTHSCALE_INIT],
        "noise_variance_init": NOISE_VARIANCE_INIT,
        "joint_evals": 1,
        "start_n": 2,
        "_y_mean": y_mean,
        "_y_std": y_std,
    }


def _max_gap(dumps: dict[str, dict]) -> float:
    """Largest absolute gap of the final predictive means, in standardized y."""
    last = {lib: dump["steps"][-1]["mean"] for lib, dump in dumps.items()}
    base = np.asarray(last["gprx"])
    return max(float(np.max(np.abs(base - np.asarray(last[lib])))) for lib in last if lib != "gprx")


def main() -> int:
    os.environ["PERF_WARMUP"] = "0"
    import matplotlib

    matplotlib.use("Agg")
    BENCH.mkdir(parents=True, exist_ok=True)
    curve = snelson()
    grid = curve.x_grid.ravel()
    case_path = write_curve_case("snelson", "matched", "sgpr", SPARSE_M)
    case = json.loads(case_path.read_text(encoding="utf-8"))
    rows = []
    for lib in SPARSE_LIBS:
        row = SPARSE_RUNNERS[lib](case_path)
        row["lib"] = lib
        rows.append(row)
        print(f"sparse {lib}: {row.get('status')} evals={row.get('joint_evals')} note={row.get('note')}", flush=True)
        if row.get("status") != "ok":
            return 1
    sparse_path = sparse_figure(case, rows, curve.x_train.ravel(), curve.y_train, grid)
    print(sparse_path, flush=True)

    order = np.argsort(curve.x_train.ravel(), kind="mergesort")
    train_x = curve.x_train.ravel()[order]
    train_y = curve.y_train[order]
    online_grid = grid
    if online_grid.shape[0] > ONLINE_GRID:
        pick = np.linspace(0, online_grid.shape[0] - 1, ONLINE_GRID).astype(int)
        online_grid = online_grid[pick]
    online = _online_case(train_x, train_y, online_grid)
    y_mean, y_std = online.pop("_y_mean"), online.pop("_y_std")
    online_path = case_path.with_name("snelson_online.json")
    online_path.write_text(json.dumps(online), encoding="utf-8")
    dumps = {
        "gprx": _cargo_frames(online_path),
        "libgp": _libgp_frames(online_path),
        "gpytorch": _gpytorch_frames(
            np.asarray(online["x"])[:, None],
            np.asarray(online["y"]),
            np.asarray(online["xs"])[:, None],
        ),
    }
    gap = _max_gap(dumps)
    print(f"online final-mean gap vs gprx (standardized y): {gap:.3e}", flush=True)
    gif_path = online_gif(dumps, train_x, train_y, online_grid, y_mean, y_std)
    print(gif_path, flush=True)
    info = meta()
    print(json.dumps(info["machine"]), flush=True)
    print(json.dumps(info["versions"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
