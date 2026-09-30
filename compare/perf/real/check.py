"""Do the libraries agree at one fixed θ? NLML, RMSE, NLPD of every runner on
a ``fixed`` case (no optimizer). Run before trusting any fit comparison.

```text
python -m perf.real.check [dataset] [split]
```
"""

from __future__ import annotations

import json
import sys

from .cases import write_case
from .libs import RUNNERS

#: A start that is not the default, so a wrong parameter order would show.
THETA = {"lengthscale_init": 0.8, "signal_variance_init": 1.7, "noise_variance_init": 0.05}
FIELDS = ("nlml", "rmse", "nlpd", "coverage95")
#: Relative difference each metric may show between libraries.
TOLERANCE = 1e-6


def main(argv: list[str]) -> int:
    dataset = argv[0] if argv else "yacht"
    split = int(argv[1]) if len(argv) > 1 else 0
    path = write_case(dataset, split, "fixed")
    case = json.loads(path.read_text(encoding="utf-8"))
    case.update(THETA)
    path.write_text(json.dumps(case), encoding="utf-8")
    rows = {lib: run(path) for lib, run in RUNNERS.items()}
    for lib, row in rows.items():
        if row.get("status") != "ok":
            print(f"{lib}: {row.get('note')}", file=sys.stderr)
            return 1
        print(lib, {f: row[f] for f in FIELDS})
    reference = rows["gprx"]
    ok = True
    for lib, row in rows.items():
        for field in FIELDS:
            rel = abs(row[field] - reference[field]) / max(abs(reference[field]), 1e-12)
            if rel > TOLERANCE:
                ok = False
                print(f"MISMATCH {lib} {field}: {row[field]} vs gprx {reference[field]} (rel {rel:.2e})")
    print("agree" if ok else "DISAGREE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
