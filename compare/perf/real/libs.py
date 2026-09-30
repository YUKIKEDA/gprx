"""The runner of each library, by name."""

from __future__ import annotations

from pathlib import Path

from common.harness import Row, na_row

from ..runners import run_gprx, run_libgp, run_python

FIT_FIELDS = (
    "fit_s",
    "predict_s",
    "joint_evals",
    "value_evals",
    "iterations",
    "nlml",
    "rmse",
    "nlpd",
    "coverage95",
    "peak_rss_bytes",
)


def _python(module: str):
    return lambda path: run_python(f"real.{module}", path, FIT_FIELDS)


RUNNERS: dict[str, "callable[[Path], Row]"] = {
    "gprx": lambda path: run_gprx("fit", path, features="fit-counts"),
    "sklearn": _python("sklearn_fit"),
    "gpytorch": _python("gpytorch_fit"),
    "gpy": _python("gpy_fit"),
    "libgp": lambda path: run_libgp("libgp-fit", path),
    # friedrich has only isotropic kernels; the cells stay in the table as N/A.
    "friedrich": lambda path: na_row("friedrich has no ARD kernel", FIT_FIELDS),
}
