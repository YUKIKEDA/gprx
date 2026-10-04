"""From ``out/real/results.json`` (+ ``meta.json``) to what the repository
keeps: ``docs/bench/summary.json`` (mean and standard error per cell, the
machine and library versions, every optimizer), the PNG figures, and the
Markdown tables between ``<!-- bench:begin -->`` and ``<!-- bench:end -->`` in
both READMEs.

```text
python -m perf.real.report [--no-readme]
```
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

from . import plot
from .curves import CURVES
from .data import OUT
from .libs import RUNNERS
from .optimizers import OPTIMIZERS

REPO = Path(__file__).resolve().parents[3]
BENCH = REPO / "docs" / "bench"
BEGIN, END = "<!-- bench:begin -->", "<!-- bench:end -->"
SNIPPETS = REPO / "compare" / "perf" / "real" / "snippets"
READMES = (REPO / "README.md", REPO / "README.ja.md")
LIB_ORDER = list(RUNNERS)


def mean_se(values: list[float]) -> tuple[float, float] | None:
    if not values:
        return None
    mean = statistics.fmean(values)
    se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, se


def summarize(rows: list[dict]) -> list[dict]:
    cells: dict[tuple, list[dict]] = {}
    for row in rows:
        dataset = row["name"].rsplit("_s", 1)[0]
        key = (row.get("model", "exact"), row["protocol"], dataset, row["lib"])
        cells.setdefault(key, []).append(row)
    out = []
    for (model, protocol, dataset, lib), group in cells.items():
        ok = [r for r in group if r.get("status") == "ok"]
        cell = {
            "model": model,
            "protocol": protocol,
            "dataset": dataset,
            "lib": lib,
            "ok": len(ok),
            "total": len(group),
            "notes": sorted({r["note"] for r in group if r.get("status") != "ok" and r.get("note")}),
        }
        for key in ("rmse", "nlpd", "coverage95", "fit_s", "predict_s", "joint_evals", "iterations", "nlml"):
            stats = mean_se([r[key] for r in ok if r.get(key) is not None])
            cell[key] = None if stats is None else {"mean": stats[0], "se": stats[1]}
        per_eval = [
            r["fit_s"] / r["joint_evals"] * 1e3
            for r in ok
            if r.get("fit_s") is not None and r.get("joint_evals")
        ]
        cell["ms_per_eval"] = statistics.median(per_eval) if per_eval else None
        peaks = [r["peak_rss_bytes"] for r in ok if r.get("peak_rss_bytes")]
        cell["peak_rss_mib"] = max(peaks) / 2**20 if peaks else None
        out.append(cell)
    return out


def fmt_median(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4g}"


def fmt(stat: dict | None, digits: int = 4) -> str:
    if stat is None:
        return "N/A"
    if stat["se"] == 0.0:
        return f"{stat['mean']:.{digits}g}"
    return f"{stat['mean']:.{digits}g} ± {stat['se']:.2g}"


_TITLES = {
    "exact": ("Exact GP", "全学習点を使うモデル"),
    "sgpr": ("SGPR, 512 inducing points", "誘導点 512 個のモデル"),
    "svgp": ("SVGP", "ミニバッチのモデル"),
}


def _with_numbers(cells: list[dict]) -> list[dict]:
    """Drop a dataset whose every library failed. A row of blanks is not a result."""
    by_dataset: dict[str, list[dict]] = {}
    for cell in cells:
        by_dataset.setdefault(cell["dataset"], []).append(cell)
    kept: list[dict] = []
    for group in by_dataset.values():
        if any(cell["ok"] for cell in group):
            kept.extend(group)
    return kept


def tables(summary: list[dict], ja: bool = False) -> str:
    lines: list[str] = []
    groups = sorted({(c["model"], c["protocol"]) for c in summary})
    if ja:
        header = (
            "| データセット | ライブラリ | 測れた分割 | RMSE | NLPD | 95%区間 | 学習 [秒] | "
            "評価回数 | 反復 | 1回あたり [ms] | 負の対数周辺尤度 | メモリ [MiB] |"
        )
    else:
        header = (
            "| dataset | library | splits | RMSE | NLPD | 95% interval | fit [s] | "
            "evaluations | iterations | ms / evaluation | NLML | memory [MiB] |"
        )
    rule = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    for model, _protocol in groups:
        cells = _with_numbers([c for c in summary if c["model"] == model])
        if not cells:
            continue
        title = _TITLES.get(model, (model, model))[1 if ja else 0]
        lines += [f"#### {title}", "", header, rule]
        order = {lib: i for i, lib in enumerate(LIB_ORDER)}
        for c in sorted(cells, key=lambda c: (c["dataset"], order.get(c["lib"], 99))):
            rss = "N/A" if c["peak_rss_mib"] is None else f"{c['peak_rss_mib']:.1f}"
            lines.append(
                f"| {c['dataset']} | {c['lib']} | {c['ok']}/{c['total']} | {fmt(c['rmse'])} | "
                f"{fmt(c['nlpd'])} | {fmt(c['coverage95'], 3)} | {fmt(c['fit_s'])} | "
                f"{fmt(c['joint_evals'], 3)} | {fmt(c['iterations'], 3)} | {fmt_median(c['ms_per_eval'])} | "
                f"{fmt(c['nlml'])} | {rss} |"
            )
        lines.append("")
    return "\n".join(lines)


def optimizer_table(ja: bool = False) -> str:
    if ja:
        lines = [
            "各マスは、その版のソースから書き写した設定。",
            "",
            "| ライブラリ | 既定の最適化 | 条件を揃えた最適化 | 探索する量 | 範囲 |",
            "| --- | --- | --- | --- | --- |",
        ]
    else:
        lines = [
            "| library | default optimizer | shared optimizer | search space | bounds |",
            "| --- | --- | --- | --- | --- |",
        ]
    for lib, o in OPTIMIZERS.items():
        lines.append(f"| {lib} | {o['native']} | {o['matched']} | {o['space']} | {o['bounds']} |")
    return "\n".join(lines)


def machine_line(meta: dict, ja: bool = False) -> str:
    m, v = meta["machine"], meta["versions"]
    libs = ", ".join(f"{k} {v[k]}" for k in ("scikit-learn", "gpytorch", "GPy", "torch", "scipy", "argmin") if v.get(k))
    if ja:
        return (
            f"測定した機械は {m['cpu']}（論理 CPU {m['logical_cpus']}、メモリ {m['memory_gib']} GiB、{m['os']}）。"
            f"{libs}。libgp {v['libgp'][:8]}。"
        )
    return f"Measured on {m['cpu']} ({m['logical_cpus']} logical CPUs, {m['memory_gib']} GiB, {m['os']}). {libs}; libgp {v['libgp'][:8]}."


_CAPTIONS: dict[str, tuple[tuple[str, str], tuple[str, str]]] = {
    # name -> ((en title, en body), (ja title, ja body))
    "accuracy_matched.png": (
        (
            "Prediction error, Exact GP",
            ("Each column is a dataset. Top is RMSE, bottom is NLPD; lower is better. "
            "A marker is a library and the bar is the standard error across splits. "
            "Snelson has no test points, so that column is empty."),
        ),
        (
            "予測の誤差（全学習点）",
            ("列がデータセット。上は RMSE、下は NLPD で、どちらも小さいほど良い。"
            "点はライブラリ、縦棒は分割ごとのばらつき。"
            "Snelson にはテスト点が無いので、その列は空。"),
        ),
    ),
    "fit_time_matched.png": (
        (
            "Training time, Exact GP",
            ("Top is the seconds spent training, bottom is how many times the library "
            "evaluated the likelihood and its gradient together. Both axes are logarithmic. "
            "Compare the seconds only where the counts match."),
        ),
        (
            "学習の時間（全学習点）",
            ("上は学習にかかった秒、下は尤度と勾配を一緒に計算した回数。どちらも対数軸。"
            "秒を比べるときは、下の回数が揃っているかを見る。"),
        ),
    ),
    "accuracy_sgpr_matched.png": (
        (
            "Prediction error, SGPR",
            "Same reading as the Exact GP error figure. 512 inducing points. RMSE on top, NLPD below.",
        ),
        (
            "予測の誤差（誘導点 512 個）",
            "読み方は、全学習点の予測誤差の図と同じ。上は RMSE、下は NLPD。",
        ),
    ),
    "fit_time_sgpr_matched.png": (
        (
            "Training time, SGPR",
            "Same reading as the Exact GP time figure. Seconds on top, likelihood-and-gradient counts below.",
        ),
        (
            "学習の時間（誘導点 512 個）",
            "上は秒、下は尤度と勾配の計算回数。どちらも対数軸。",
        ),
    ),
    "accuracy_svgp_matched.png": (
        (
            "Prediction error, SVGP",
            ("Adam, learning rate 0.01, batch 1024, three passes over the data, in both libraries. "
            "GPy has no minibatch trainer, so it is absent. RMSE on top, NLPD below."),
        ),
        (
            "予測の誤差（ミニバッチ）",
            ("Adam で学習し、学習率 0.01、バッチ 1024、データ 3 周。gprx と GPyTorch で同じ設定。"
            "GPy にはこの学習が無い。上は RMSE、下は NLPD。"),
        ),
    ),
    "fit_time_svgp_matched.png": (
        (
            "Training time, SVGP",
            "Seconds on top, Adam updates below. The update count matches, so the seconds are the speed.",
        ),
        (
            "学習の時間（ミニバッチ）",
            "上は秒、下は Adam の更新回数。回数は揃っているので、秒の差が速さの差になる。",
        ),
    ),
    "rss_timeline_energy_exact_s0_matched.png": (
        (
            "Memory over time, energy",
            ("The line is the resident memory of the whole process. "
            "The horizontal axis is seconds since the process started. "
            "A dotted line, in that library's color, is when training or prediction starts. Split 0."),
        ),
        (
            "メモリの推移（energy、全学習点）",
            ("線はプロセス全体の常駐メモリ。横軸はプロセスが始まってからの秒。"
            "点線は、その色のライブラリが学習または予測を始めた時刻。分割は 0 番。"),
        ),
    ),
    "rss_timeline_kin40k_sgpr_s0_matched.png": (
        (
            "Memory over time, kin40k",
            "Same reading as the energy memory figure. SGPR with 512 inducing points, split 0.",
        ),
        (
            "メモリの推移（kin40k、誘導点 512 個）",
            "読み方は energy のメモリの図と同じ。分割は 0 番。",
        ),
    ),
    "curve_maunaloa_matched.png": (
        (
            "Mauna Loa predictions",
            ("One panel per library. The line is the predictive mean, the band is the 95% interval, "
            "filled points are training data, and hollow points are held out."),
        ),
        (
            "Mauna Loa の予測",
            ("1 枚が 1 ライブラリ。線が予測の平均、帯が 95% 区間。"
            "塗った点は学習データ、抜き点はテストデータ。"),
        ),
    ),
    "curve_snelson_matched.png": (
        (
            "Snelson predictions",
            ("One panel per library. The line is the predictive mean and the band is the 95% interval. "
            "The points are the training data. Nothing is held out."),
        ),
        (
            "Snelson の予測",
            "1 枚が 1 ライブラリ。線が予測の平均、帯が 95% 区間。点は学習データ。テスト用の点は無い。",
        ),
    ),
}


def figures_markdown(names: list[str], ja: bool = False) -> str:
    intro = (
        "点の色と形はどの図でも同じ。青丸が gprx、橙の四角が scikit-learn、緑の三角が GPyTorch、黄の菱形が GPy。"
        if ja
        else "The marker is the same library in every figure: a blue circle is gprx, an orange square is "
        "scikit-learn, a green triangle is GPyTorch, and a yellow diamond is GPy."
    )
    blocks = [intro, ""]
    for name in names:
        pair = _CAPTIONS.get(name)
        if pair is None:
            blocks += [f"![{name}](docs/bench/{name})", ""]
            continue
        title, body = pair[1 if ja else 0]
        blocks += [f"**{title}**", "", body, "", f"![{title}](docs/bench/{name})", ""]
    return "\n".join(blocks).rstrip()


def bench_body(meta: dict, summary: list[dict], figures: list[str], ja: bool = False) -> str:
    return "\n".join(
        [
            machine_line(meta, ja),
            "",
            optimizer_table(ja),
            "",
            tables(summary, ja),
            figures_markdown(figures, ja),
        ]
    )


def splice(readme: Path, body: str, begin: str = BEGIN, end: str = END) -> bool:
    text = readme.read_text(encoding="utf-8")
    if begin not in text or end not in text:
        return False
    head, rest = text.split(begin, 1)
    _, tail = rest.split(end, 1)
    readme.write_text(f"{head}{begin}\n{body}\n{end}{tail}", encoding="utf-8", newline="")
    return True


SNIPPET_FILES = (
    ("gprx", "gprx_ard.rs", "rust"),
    ("scikit-learn", "sklearn_ard.py", "python"),
    ("GPyTorch", "gpytorch_ard.py", "python"),
    ("GPy", "gpy_ard.py", "python"),
    ("libgp", "libgp_ard.cpp", "cpp"),
)


def snippets_block() -> str:
    """Each snippet file's part between its ``snippet:begin`` and ``snippet:end``."""
    parts = []
    for name, filename, lang in SNIPPET_FILES:
        text = (SNIPPETS / filename).read_text(encoding="utf-8")
        body = text.split("snippet:begin", 1)[1].split("\n", 1)[1].split("snippet:end", 1)[0]
        body = body.rsplit("\n", 1)[0]
        lines = body.splitlines()
        indent = min((len(l) - len(l.lstrip()) for l in lines if l.strip()), default=0)
        code = "\n".join(l[indent:] for l in lines)
        parts.append(f"<details><summary>{name}</summary>\n\n```{lang}\n{code}\n```\n\n</details>")
    return "\n\n".join(parts)


def main(argv: list[str]) -> int:
    rows = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    summary = summarize(rows)
    BENCH.mkdir(parents=True, exist_ok=True)
    (BENCH / "summary.json").write_text(
        json.dumps({"meta": meta, "cells": summary}, indent=2) + "\n", encoding="utf-8"
    )
    figures = []
    for model, protocol in sorted({(c["model"], c["protocol"]) for c in summary}):
        for maker in (plot.accuracy, plot.fit_time):
            path = maker(rows, protocol, BENCH, model)
            if path:
                figures.append(path.name)
    for prefix in plot.timeline_groups(OUT / "timeline"):
        path = plot.rss_timeline(prefix, BENCH, OUT / "timeline")
        if path:
            figures.append(path.name)
    for curve_name in sorted({c["dataset"] for c in summary if c["dataset"] in CURVES}):
        for protocol in sorted({c["protocol"] for c in summary if c["dataset"] == curve_name}):
            try:
                path = plot.curve(curve_name, protocol, BENCH)
            except OSError as err:  # the data cannot be fetched here (urllib errors are OSErrors)
                print(f"curve {curve_name}: no figure ({err})", file=sys.stderr)
                path = None
            if path:
                figures.append(path.name)
    if "--no-readme" not in argv:
        for readme in READMES:
            ja = readme.name == "README.ja.md"
            done = splice(readme, bench_body(meta, summary, figures, ja)) and splice(
                readme, snippets_block(), "<!-- snippets:begin -->", "<!-- snippets:end -->"
            )
            print(readme.name, "updated" if done else "no markers, left as is")
    print(f"wrote {BENCH / 'summary.json'} and {len(figures)} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
