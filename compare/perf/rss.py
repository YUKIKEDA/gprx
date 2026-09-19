"""Peak resident set for a Python runner process."""

from __future__ import annotations

import sys

import psutil


def peak_rss_bytes() -> int:
    info = psutil.Process().memory_info()
    peak_wset = getattr(info, "peak_wset", None)
    if peak_wset:
        return int(peak_wset)
    if sys.platform != "win32":
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KiB; macOS reports bytes.
        if sys.platform.startswith("linux"):
            return int(usage) * 1024
        return int(usage)
    return int(info.rss)
