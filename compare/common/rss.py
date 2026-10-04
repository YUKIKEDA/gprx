"""Peak resident set for a Python runner process.

Linux reads ``VmHWM`` from ``/proc/self/status``. ``ru_maxrss`` is not used
there: ``exec`` folds the high-water mark of the pre-exec address space (the
parent's, under fork or vfork) into it, so a runner launched from a large
harness reports the harness's size. ``VmHWM`` belongs to the current address
space, which ``exec`` replaces.
"""

from __future__ import annotations

import sys

import psutil


def peak_rss_bytes() -> int:
    info = psutil.Process().memory_info()
    peak_wset = getattr(info, "peak_wset", None)
    if peak_wset:
        return int(peak_wset)
    if sys.platform.startswith("linux"):
        return _vm_hwm_bytes()
    if sys.platform != "win32":
        import resource

        # macOS reports bytes. The other BSDs report KiB.
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return int(usage)
        return int(usage) * 1024
    return int(info.rss)


def _vm_hwm_bytes() -> int:
    with open("/proc/self/status", encoding="ascii") as status:
        for line in status:
            if line.startswith("VmHWM:"):
                value, unit = line.split()[1:3]
                if unit != "kB":
                    raise RuntimeError(f"unexpected VmHWM unit: {line!r}")
                return int(value) * 1024
    raise RuntimeError("VmHWM missing from /proc/self/status")
