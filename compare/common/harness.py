"""Running the perf runners, judging cells, and printing the bench-log tables.

Every harness (``perf/run*.py``) declares its cases, its runners, and its
table columns, and uses the functions here for the rest: starting a runner
process, the pass / fail / 判定不能 / N/A verdicts, the time / RSS formats,
the Markdown tables and ``## gates`` list, and the result / verdict JSON.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

PASS = "pass"
FAIL = "fail"
UNDECIDED = "判定不能"
NA = "N/A"

#: Fields a factor / joint / predict runner reports (None when it fails).
TIME_FIELDS = ("factor_s", "eval_s", "predict_s", "joint_evals", "peak_rss_bytes")
#: Fields an op-sequence runner reports (None when it fails).
OPS_FIELDS = ("ops_s", "peak_rss_bytes")

Row = dict[str, Any]
Runner = Callable[[Path], Row]


def na_row(note: str, fields: Sequence[str] = TIME_FIELDS) -> Row:
    """The row of a cell that could not run."""
    row: Row = {"status": "na", "note": note}
    for field in fields:
        row[field] = None
    return row


def run_cmd(args: list[str], cwd: Path, fields: Sequence[str] = TIME_FIELDS) -> Row:
    """Runs one runner process and reads its JSON row from the last stdout line."""
    proc = subprocess.run(args, cwd=cwd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        return na_row(f"runner failed ({proc.returncode}): {err}", fields)
    line = proc.stdout.strip().splitlines()[-1]
    return json.loads(line)


def ratio_verdict(ours: float | None, theirs: float | None, band: float) -> str:
    """pass when ``ours`` is smaller than ``theirs`` by more than ``band``."""
    if ours is None or theirs is None:
        return NA
    if theirs == 0.0:
        return UNDECIDED
    rel = (ours - theirs) / theirs
    if abs(rel) <= band:
        return UNDECIDED
    if ours < theirs:
        return PASS
    return FAIL


def within_band_or_better(ours: float, theirs: float, band: float) -> str:
    """pass unless ``ours`` is larger than ``theirs`` by more than ``band``."""
    if theirs == 0.0:
        return UNDECIDED
    rel = (ours - theirs) / theirs
    if rel > band:
        return FAIL
    return PASS


def combine_times(*verdicts: str) -> str:
    """fail if any judged verdict fails; pass if every judged one passes
    (判定不能 is left out); N/A if nothing is judged and a verdict is N/A."""
    judged = [v for v in verdicts if v not in (UNDECIDED, NA)]
    if any(v == FAIL for v in judged):
        return FAIL
    if judged and all(v == PASS for v in judged):
        return PASS
    if any(v == NA for v in verdicts):
        return NA
    return UNDECIDED


def combine_strict(*verdicts: str) -> str:
    """Like :func:`combine_times`, but pass only when every verdict passes."""
    judged = [v for v in verdicts if v not in (UNDECIDED, NA)]
    if any(v == FAIL for v in judged):
        return FAIL
    if judged and all(v == PASS for v in judged) and NA not in verdicts:
        if any(v == UNDECIDED for v in verdicts):
            return UNDECIDED
        return PASS
    if any(v == NA for v in verdicts):
        return NA
    return UNDECIDED


def fmt_s(value: float | None) -> str:
    if value is None:
        return NA
    if value >= 1.0:
        return f"{value:.3f} s"
    return f"{value * 1000:.2f} ms"


def fmt_s_range(median: float | None, lo: float | None, hi: float | None) -> str:
    if median is None:
        return NA
    if lo is None or hi is None:
        return fmt_s(median)
    if abs(hi - lo) <= max(median, 1e-12) * 1e-6:
        return fmt_s(median)
    return f"{fmt_s(median)} ({fmt_s(lo)}–{fmt_s(hi)})"


def fmt_time(row: Row, key: str) -> str:
    """``{key}_s`` with its ``{key}_min_s`` – ``{key}_max_s`` range."""
    return fmt_s_range(row.get(f"{key}_s"), row.get(f"{key}_min_s"), row.get(f"{key}_max_s"))


def fmt_rss(value: int | None) -> str:
    if value is None:
        return NA
    return f"{value / (1024 * 1024):.1f} MiB"


def fmt_evals(value: int | None) -> str:
    return NA if value is None else str(value)


def split_case(name: str, prefixes: Iterable[str] = ()) -> tuple[str, str, str]:
    """``(prefix, problem, n)`` of a case name such as ``sgpr_forrester_n256``.

    ``prefix`` is the first of ``prefixes`` the name starts with (``"?"`` when
    ``prefixes`` is given and none matches, ``""`` when it is empty).
    """
    prefixes = tuple(prefixes)
    head = "" if not prefixes else "?"
    rest = name
    for prefix in prefixes:
        if name.startswith(f"{prefix}_"):
            head = prefix
            rest = name[len(prefix) + 1 :]
            break
    problem = "forrester" if "forrester" in rest else "sphere"
    n = rest.rsplit("n", 1)[-1]
    return head, problem, n


def start(banner: str) -> None:
    """UTF-8 stdout, then the clock banner."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(banner, flush=True)


def case_filter(argv: Sequence[str]) -> set[str]:
    """The case names on the command line (arguments that are not flags)."""
    return {arg for arg in argv[1:] if not arg.startswith("-")}


def select(cases: list[Path], only: set[str], match: Callable[[Path], bool]) -> list[Path] | None:
    """The cases ``match`` keeps when ``only`` is not empty; None (after a
    message) when none is left."""
    if not only:
        return cases
    kept = [path for path in cases if match(path)]
    if not kept:
        print(f"no cases match {sorted(only)}", file=sys.stderr)
        return None
    return kept


def run_cells(
    cases: list[Path],
    runners: Sequence[tuple[str, Runner]],
    progress: Callable[[Row], str],
) -> list[Row]:
    """Runs every runner on every case, printing ``progress(row)`` after each."""
    rows: list[Row] = []
    for case_path in cases:
        print(f"# {case_path.name}", flush=True)
        for lib, runner in runners:
            print(f"  {lib}...", flush=True)
            row = runner(case_path)
            row["lib"] = lib
            row.setdefault("name", case_path.stem)
            rows.append(row)
            print(f"    {row.get('status')} {progress(row)}", flush=True)
    return rows


def by_case(rows: list[Row]) -> dict[str, dict[str, Row]]:
    """Rows keyed by case name, then by lib, in first-seen order."""
    grouped: dict[str, dict[str, Row]] = {}
    for row in rows:
        grouped.setdefault(row["name"], {})[row["lib"]] = row
    return grouped


def print_table(title: str, header: Sequence[str], lines: Iterable[Sequence[str]]) -> None:
    """``## title`` then a Markdown table."""
    print(f"\n## {title}\n")
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join("---" for _ in header) + " |")
    for cells in lines:
        print("| " + " | ".join(cells) + " |")


def print_gates(title: str, lines: Iterable[str]) -> None:
    """``## title`` then one bullet per gate."""
    print(f"\n## {title}\n")
    for line in lines:
        print(f"- {line}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_rows(path: Path) -> list[Row]:
    return json.loads(path.read_text(encoding="utf-8"))
