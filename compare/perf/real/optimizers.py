"""Which optimizer every library runs, with its settings (B1-1).

Each entry was read from the library's source at the version recorded in
``meta.json`` (``run_real`` writes it next to the results). ``native`` is what
the library does when asked to fit; ``matched`` is the shared setting of
``cases.py`` for the libraries that can take it.
"""

from __future__ import annotations

import os
import platform
import re
from importlib import metadata
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

LBFGSB_MATCHED = (
    "scipy L-BFGS-B, maxiter 100, gtol sqrt(eps), ftol 0, maxcor 10; "
    "the library's own objective and gradient"
)

#: Per library: how ``native`` and ``matched`` fit. ``space`` is what the
#: optimizer sees, ``bounds`` what limits it.
OPTIMIZERS: dict[str, dict[str, str]] = {
    "gprx": {
        "native": "argmin 0.11 LBFGS + MoreThuente line search; history 10, max 100 iterations, "
        "gradient-norm tolerance sqrt(eps)",
        "matched": "same call: the gprx default already equals the shared setting",
        "space": "logit of log θ inside each interval, so the search is unconstrained",
        "bounds": "(1e-5, 1e5) on ℓ, signal variance and noise variance",
        "source": "src/optimizer/lbfgs.rs, src/optimizer/logit.rs",
    },
    "sklearn": {
        "native": "scipy minimize L-BFGS-B via optimizer='fmin_l_bfgs_b' with scipy defaults "
        "(maxiter 15000, ftol 2.2e-9, gtol 1e-5, maxcor 10, maxls 20)",
        "matched": LBFGSB_MATCHED,
        "space": "log θ",
        "bounds": "(1e-5, 1e5) on ℓ, constant value and noise level (kernel defaults)",
        "source": "sklearn/gaussian_process/_gpr.py _constrained_optimization",
    },
    "gpytorch": {
        "native": "torch.optim.Adam, lr 0.1, 50 steps (the exact-GP tutorial setting; GPyTorch has "
        "no default optimizer)",
        "matched": LBFGSB_MATCHED + "; gradient by autograd on -mll × n",
        "space": "raw parameters behind softplus",
        "bounds": "noise ≥ 1e-5 (set here; GPyTorch's default is 1e-4); ℓ and outputscale positive only",
        "source": "compare/perf/real/gpytorch_fit.py",
    },
    "gpy": {
        "native": "model.optimize(): paramz opt_lbfgsb = scipy fmin_l_bfgs_b with maxfun = maxiter = "
        "1000, factr 1e7, pgtol 1e-5",
        "matched": LBFGSB_MATCHED,
        "space": "softplus (Logexp) of each parameter",
        "bounds": "positive only, no upper bound",
        "source": "paramz/optimization/optimization.py opt_lbfgsb, paramz/model.py optimize",
    },
    "libgp": {
        "native": "RProp (resilient backpropagation), 100 iterations, eps_stop 0, Delta0 0.1, "
        "Deltamin 1e-6, Deltamax 50, eta- 0.5, eta+ 1.2; keeps the best likelihood seen",
        "matched": "N/A: libgp offers RProp and CG only, and RProp has no gradient tolerance",
        "space": "log ℓ, log sf, log sn (amplitude and std, not variances)",
        "bounds": "none",
        "source": "src/rprop.cc, include/rprop.h",
    },
    "friedrich": {
        "native": "N/A: no ARD kernel",
        "matched": "N/A: no ARD kernel",
        "space": "-",
        "bounds": "-",
        "source": "-",
    },
}

PY_PACKAGES = ("numpy", "scipy", "scikit-learn", "torch", "gpytorch", "GPy", "paramz")


def _lock_version(lock: Path, crate: str) -> str:
    text = lock.read_text(encoding="utf-8")
    match = re.search(rf'name = "{re.escape(crate)}"\nversion = "([^"]+)"', text)
    return match.group(1) if match else "unknown"


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _windows_memory() -> tuple[int, int] | None:
    """Total and available physical bytes, from ``GlobalMemoryStatusEx``."""
    import ctypes

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = (
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        )

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return int(status.ullTotalPhys), int(status.ullAvailPhys)


def _memory_gib() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                return round(int(line.split()[1]) / (1024 * 1024), 1)
    except OSError:
        pass
    windows = _windows_memory()
    if windows is not None:
        return round(windows[0] / 2**30, 1)
    return None


def total_gib() -> float | None:
    """Installed physical memory, in GiB."""
    return _memory_gib()


def available_gib() -> float | None:
    """Physical memory that is free right now, in GiB."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    windows = _windows_memory()
    if windows is not None:
        return windows[1] / 2**30
    return None


def meta() -> dict:
    versions = {}
    for package in PY_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    lock = REPO / "compare" / "perf" / "gprx" / "Cargo.lock"
    versions["argmin"] = _lock_version(lock, "argmin")
    versions["gprx"] = "path (this checkout)"
    versions["libgp"] = "f4a2fb7d4e1de7de8f1555318f14750830913663"
    return {
        "machine": {
            "cpu": _cpu_model(),
            "logical_cpus": os.cpu_count(),
            "memory_gib": _memory_gib(),
            "os": platform.platform(),
        },
        "versions": versions,
        "optimizers": OPTIMIZERS,
    }
