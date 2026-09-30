"""Runs a runner process while sampling the RSS of its whole process tree.

A runner may print ``PHASE <name>`` lines on stderr (see ``phase``); the
sampler records when each arrived, so a plot can mark load / fit / predict.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

INTERVAL_S = 0.01


def phase(name: str) -> None:
    """Announces the start of a phase to the sampler (a no-op for readers)."""
    print(f"PHASE {name}", file=sys.stderr, flush=True)


def _tree_rss(root: psutil.Process) -> int:
    total = 0
    try:
        procs = [root, *root.children(recursive=True)]
    except psutil.Error:
        return 0
    for proc in procs:
        try:
            total += proc.memory_info().rss
        except psutil.Error:
            pass
    return total


def run_sampled(
    args: list[str], cwd: Path, directory: Path, label: str
) -> subprocess.CompletedProcess[str]:
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    samples: list[tuple[float, int]] = []
    phases: list[tuple[float, str]] = []
    stderr_lines: list[str] = []
    stop = threading.Event()

    def sample() -> None:
        root = psutil.Process(proc.pid)
        while not stop.is_set():
            samples.append((time.perf_counter() - t0, _tree_rss(root)))
            time.sleep(INTERVAL_S)

    def read_stderr() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            if line.startswith("PHASE "):
                phases.append((time.perf_counter() - t0, line.split(None, 1)[1].strip()))
            else:
                stderr_lines.append(line)

    sampler = threading.Thread(target=sample, daemon=True)
    reader = threading.Thread(target=read_stderr, daemon=True)
    sampler.start()
    reader.start()
    assert proc.stdout is not None
    stdout = proc.stdout.read()
    proc.wait()
    reader.join()
    stop.set()
    sampler.join()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{label}.json").write_text(
        json.dumps({"interval_s": INTERVAL_S, "samples": samples, "phases": phases}),
        encoding="utf-8",
    )
    return subprocess.CompletedProcess(args, proc.returncode, stdout, "".join(stderr_lines))
