"""Do the libraries agree at one fixed θ? NLML, RMSE, NLPD of every runner on
a ``fixed`` case (no optimizer). Run before trusting any fit comparison.

```text
python -m perf.real.check [dataset] [split] [exact|sgpr]
```
"""

from __future__ import annotations

import json
import sys

from .cases import write_case, write_curve_case
from .curves import CURVES
from .libs import runners_for

#: A start that is not the default, so a wrong parameter order would show.
THETA = {"lengthscale_init": 0.8, "signal_variance_init": 1.7, "noise_variance_init": 0.05}
FIELDS = ("nlml", "rmse", "nlpd", "coverage95")
#: Relative difference each metric may show between libraries.
TOLERANCE = 1e-6
#: Differences that are understood and not an error, by (model, library): at
#: the same θ and Z, GPyTorch's InducingPointKernel gives the same marginal
#: likelihood as the Titsias bound but predicts with its own low-rank test
#: covariance, so its mean and variance differ by a fraction of a percent.
KNOWN_DIFFERENCES = {
    ("sgpr", "gpytorch"): (
        ("rmse", "nlpd", "coverage95"),
        "GPyTorch's InducingPointKernel predicts with its own low-rank test covariance",
    ),
}


def main(argv: list[str]) -> int:
    dataset = argv[0] if argv else "yacht"
    split = int(argv[1]) if len(argv) > 1 else 0
    model = argv[2] if len(argv) > 2 else "exact"  # or sgpr: the bound at fixed θ and Z
    runners = runners_for(model)
    if dataset in CURVES:
        path = write_curve_case(dataset, "fixed")  # keeps its own start (θ_init)
    else:
        path = write_case(dataset, split, "fixed", model, 16)
        case = json.loads(path.read_text(encoding="utf-8"))
        case.update(THETA)
        path.write_text(json.dumps(case), encoding="utf-8")
    rows = {lib: run(path) for lib, run in runners.items()}
    if rows["gprx"].get("status") != "ok":
        print(f"gprx: {rows['gprx'].get('note')}", file=sys.stderr)
        return 1
    for lib in list(rows):
        if rows[lib].get("status") != "ok":
            print(f"{lib}: N/A ({rows[lib].get('note')})")
            del rows[lib]
    for lib, row in rows.items():
        print(lib, {f: row[f] for f in FIELDS})
    reference = rows["gprx"]
    ok = True
    for lib, row in rows.items():
        for field in FIELDS:
            rel = abs(row[field] - reference[field]) / max(abs(reference[field]), 1e-12)
            if rel > TOLERANCE:
                known = KNOWN_DIFFERENCES.get((model, lib))
                if known and field in known[0]:
                    print(f"known difference {lib} {field}: {row[field]} vs gprx {reference[field]} (rel {rel:.2e}): {known[1]}")
                    continue
                ok = False
                print(f"MISMATCH {lib} {field}: {row[field]} vs gprx {reference[field]} (rel {rel:.2e})")
    print("agree" if ok else "DISAGREE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
