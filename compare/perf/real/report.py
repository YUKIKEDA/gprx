"""From ``out/real/results.json`` (+ ``meta.json``) to what the repository
keeps: ``docs/bench/summary.json`` (mean and standard error per cell, the
machine and library versions, every optimizer), the SVG figures, and the
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
        for key in ("rmse", "nlpd", "coverage95", "fit_s", "predict_s", "joint_evals", "nlml"):
            stats = mean_se([r[key] for r in ok if r.get(key) is not None])
            cell[key] = None if stats is None else {"mean": stats[0], "se": stats[1]}
        peaks = [r["peak_rss_bytes"] for r in ok if r.get("peak_rss_bytes")]
        cell["peak_rss_mib"] = max(peaks) / 2**20 if peaks else None
        out.append(cell)
    return out


def fmt(stat: dict | None, digits: int = 4) -> str:
    if stat is None:
        return "N/A"
    if stat["se"] == 0.0:
        return f"{stat['mean']:.{digits}g}"
    return f"{stat['mean']:.{digits}g} ± {stat['se']:.2g}"


def tables(summary: list[dict]) -> str:
    lines: list[str] = []
    groups = sorted({(c["model"], c["protocol"]) for c in summary})
    for model, protocol in groups:
        cells = [c for c in summary if (c["model"], c["protocol"]) == (model, protocol)]
        lines += [
            f"#### {model} · {protocol}",
            "",
            "| dataset | library | ok | RMSE | NLPD | 95% cover | fit [s] | joint evals | NLML | peak RSS [MiB] |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        order = {lib: i for i, lib in enumerate(LIB_ORDER)}
        for c in sorted(cells, key=lambda c: (c["dataset"], order.get(c["lib"], 99))):
            rss = "N/A" if c["peak_rss_mib"] is None else f"{c['peak_rss_mib']:.1f}"
            lines.append(
                f"| {c['dataset']} | {c['lib']} | {c['ok']}/{c['total']} | {fmt(c['rmse'])} | "
                f"{fmt(c['nlpd'])} | {fmt(c['coverage95'], 3)} | {fmt(c['fit_s'])} | "
                f"{fmt(c['joint_evals'], 3)} | {fmt(c['nlml'])} | {rss} |"
            )
        reasons = sorted({(c["lib"], n) for c in cells for n in c["notes"]})
        if reasons:
            lines += [""] + [f"- N/A `{lib}`: {note}" for lib, note in reasons]
        lines.append("")
    return "\n".join(lines)


def optimizer_table() -> str:
    lines = [
        "| library | native optimizer | matched optimizer | search space | bounds |",
        "| --- | --- | --- | --- | --- |",
    ]
    for lib, o in OPTIMIZERS.items():
        lines.append(f"| {lib} | {o['native']} | {o['matched']} | {o['space']} | {o['bounds']} |")
    return "\n".join(lines)


def machine_line(meta: dict) -> str:
    m, v = meta["machine"], meta["versions"]
    libs = ", ".join(f"{k} {v[k]}" for k in ("scikit-learn", "gpytorch", "GPy", "torch", "scipy", "argmin") if v.get(k))
    return f"Measured on {m['cpu']} ({m['logical_cpus']} logical CPUs, {m['memory_gib']} GiB, {m['os']}). {libs}; libgp {v['libgp'][:8]}."


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
        if model == "exact":
            for maker in (plot.accuracy, plot.fit_time):
                path = maker(rows, protocol, BENCH)
                if path:
                    figures.append(path.name)
    body = "\n".join(
        [
            machine_line(meta),
            "",
            optimizer_table(),
            "",
            tables(summary),
            *[f"![{name}](docs/bench/{name})" for name in figures],
        ]
    )
    if "--no-readme" not in argv:
        for readme in READMES:
            done = splice(readme, body) and splice(
                readme, snippets_block(), "<!-- snippets:begin -->", "<!-- snippets:end -->"
            )
            print(readme.name, "updated" if done else "no markers, left as is")
    print(f"wrote {BENCH / 'summary.json'} and {len(figures)} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
