English | [日本語](persist-format.ja.md)

# gprx persist format

What `save` writes and `load` reads. `format_version` is 1. The code is `src/persist/`; how it fits in the crate is in [architecture.md](architecture.md); the one-paragraph summary is in [design §2](design.md#2-architecture).

The examples in this file are real output of `save` (a fit of 8 points, then `save`), not hand-written.

## 1. A saved model is a directory

```text
<dir>/
  config.json          metadata: what the model is, and every non-tensor setting
  model.safetensors    the numbers: X, y, and (per model) Z, q(u), L, α
```

`save` creates `<dir>` (and parents) if it is missing and replaces the two files. Each file is written to a temporary file in `<dir>` and renamed over its name, `model.safetensors` first and `config.json` last, so a reader sees the old file or the new one, and a save that fails part way leaves the previous `config.json`. Nothing else is left behind. The solver and the Cholesky buffer policy are not stored: a loaded model is `Fixed` and uses `CholeskyBuffer::Retain`. To train it again, call `with_optimizer` and then `refit` on the typed model.

## 2. Which model is in the directory

The `model` key of `config.json` says. An Exact file has no `model` key and reads as `exact`.

| `model` | Written by | Loaded by | Extra in the config | Tensors |
| --- | --- | --- | --- | --- |
| `exact` (key absent) | `FittedGpr::save`, `FittedGpr::save_with_factor`, `OnlineGpr::save`, `OnlineGpr::save_with_factor` | `LoadedGpr::load` | `has_factor`, `factor_kind`, and the rest of §3 | `x`, `y`; with a factor also `l`, `alpha` |
| `sgpr` | `FittedSgpr::save` | `LoadedSgpr::load` | `m`; §4 | `x`, `y`, `z`, `z_train` |
| `online_sgpr` | `OnlineSgpr::save` | `LoadedSgpr::load` | `m`, `point_ids`, `next_point_id`, `inducing_ids`, `next_inducing_id` | `x`, `y`, `z`, `z_train` |
| `svgp` | `FittedSvgp::save` | `LoadedSvgp::load` | `m` | `x`, `y`, `z`, `z_train`, `q_mean`, `q_l` |

A directory read by the wrong loader is refused with `GprError::PersistFailed` (`kind: WrongModel`), and the message names the right loader (for example, "config.json holds a svgp model; load it with LoadedSvgp::load"). This is true of `LoadedGpr::load` too.

## 3. `config.json` of an Exact model

A JSON object, written pretty-printed. Unknown keys are ignored on read. "Omitted" means the writer leaves the key out when the value is the default, and the reader supplies the default.

| Key | Type | Presence | Meaning |
| --- | --- | --- | --- |
| `format_version` | integer | required | `1`. Checked first (§8) |
| `n` | integer | required | Number of training points. `0` is `EmptyInput` |
| `d` | integer | required | Number of features. `0` is `EmptyInput` |
| `has_factor` | bool | required | Whether `l` and `alpha` are in the tensor file (§7) |
| `factor_kind` | `"llt"` or `"ldlt"` | required | `llt` loads a `FittedGpr`; `ldlt` loads an `OnlineGpr` |
| `precision` | `"double"`, `"single"`, `"mixed"` | omitted when `double` | Storage and predict scalars |
| `residual` | `"promote_storage"`, `"reevaluate_kernel"` | omitted when `promote_storage` | Mixed-precision residual. Read only when `precision` is `mixed` |
| `math` | `"accurate"`, `"fast_approx"` | omitted when `accurate` | Kernel `exp` (`KernelExp`) |
| `kernel` | object | required | The kernel tree (§5.1) |
| `likelihood` | object | required | Observation noise (§5.2) |
| `jitter` | object | required | The `JitterPolicy` (§5.3) |
| `factor_jitter` | number | omitted when `0` | The diagonal jitter the stored factor was built with |
| `distance_cache` | `"always"`, `"never"` | optional | `DistanceCachePolicy` (`Cached` / `Uncached`). Absent reads as the default, `Cached` |
| `x_unfitted` | object or string | required | Input map as configured before fit (§5.4) |
| `y_unfitted` | object or string | required | Target map as configured before fit (§5.4) |
| `x_transform` | object or string | required | Input map as fitted: its learned statistics (§5.4) |
| `y_transform` | object or string | required | Target map as fitted (§5.4) |
| `point_ids` | array of integers | required for `ldlt`, else absent | The `PointId` of each row, in row order. Its length must equal `n` |
| `next_point_id` | integer | required for `ldlt`, else absent | The next id `insert` will hand out |

An example, from `FittedGpr::save` of a `Constant × RBF` model with `MinMaxInput` and `StandardizeTarget`:

```json
{
  "format_version": 1,
  "n": 8,
  "d": 1,
  "has_factor": false,
  "factor_kind": "llt",
  "kernel": {
    "product": {
      "left":  { "constant": { "constant": { "value": 1.0, "lo": 0.00001, "hi": 100000.0 } } },
      "right": { "rbf": { "lengthscale": { "value": 1.0, "lo": 0.00001, "hi": 100000.0 } } }
    }
  },
  "likelihood": { "noise_variance": { "value": 0.1, "lo": 0.00001, "hi": 100000.0 } },
  "jitter": { "fixed": { "jitter": 0.0 } },
  "distance_cache": "always",
  "x_unfitted": { "min_max": { "range_lo": 0.0, "range_hi": 1.0 } },
  "y_unfitted": "standardize",
  "x_transform": { "min_max": { "data_min": [0.0], "data_max": [3.5], "range_lo": 0.0, "range_hi": 1.0 } },
  "y_transform": { "standardize": { "mean": 0.45206223266450496, "std": 0.4508312964998861 } }
}
```

(The file itself is indented one key per line; it is folded here to fit. `OnlineGpr::save_with_factor` of the same model differs in `has_factor: true`, `factor_kind: "ldlt"`, and adds `"point_ids": [0, 1, 2, 3, 4, 5, 6, 7]` and `"next_point_id": 8`.)

Floating-point numbers are written in the shortest form that reads back to the same `f64`, and read with serde_json's `float_roundtrip`, so a saved number comes back to the bit.

## 4. `config.json` of a sparse model

`sgpr`, `online_sgpr`, and `svgp` share one layout. It is the Exact one with these differences:

| Key | Difference |
| --- | --- |
| `model` | Added, required: `"sgpr"`, `"online_sgpr"`, or `"svgp"` |
| `m` | Added, required: the number of inducing points. `0` is `EmptyInput` |
| `jitter` | The policy for factoring `K_mm` (the sparse default is adaptive: `initial` 1e-8, `multiplier` 10, `max_retries` 5, `max_jitter` 1e-3) |
| `has_factor`, `factor_kind`, `factor_jitter`, `distance_cache` | Not written. No factor is stored (§7) |
| `inducing_ids`, `next_inducing_id` | Added, required for `online_sgpr`, absent otherwise. `point_ids` and `next_point_id` are also required for `online_sgpr` |

`precision`, `residual`, `math`, `kernel`, `likelihood`, and the four transform keys have the same form as in §3. An example (`svgp`, same kernel, transforms, and 2 inducing points), with only the keys that differ from §3 shown in full:

```json
{
  "format_version": 1,
  "model": "svgp",
  "n": 8,
  "m": 2,
  "d": 1,
  "kernel": { "rbf": { "lengthscale": { "value": 1.0, "lo": 0.00001, "hi": 100000.0 } } },
  "likelihood": { "noise_variance": { "value": 0.1, "lo": 0.00001, "hi": 100000.0 } },
  "jitter": { "adaptive": { "initial": 1e-8, "multiplier": 10.0, "max_retries": 5, "max_jitter": 0.001 } },
  "x_unfitted": { "min_max": { "range_lo": 0.0, "range_hi": 1.0 } },
  "y_unfitted": "standardize",
  "x_transform": { "min_max": { "data_min": [0.0], "data_max": [3.5], "range_lo": 0.0, "range_hi": 1.0 } },
  "y_transform": { "standardize": { "mean": 0.45206223266450496, "std": 0.4508312964998861 } }
}
```

## 5. The JSON forms

Enums use serde's external tagging with `snake_case` names. A variant with no fields is a string (`"identity"`, `"always"`). A variant with fields is an object with one key, the tag (`{"rbf": {…}}`). This holds for kernels, transforms, jitter, and the model tag.

A bounded parameter is the object `{"value": v, "lo": lo, "hi": hi}`: the value and its interval. Decoding refuses a bound that is not finite, `lo >= hi`, and a `value` that is not finite or not strictly between `lo` and `hi`.

### 5.1 `kernel`

| Tag | Fields |
| --- | --- |
| `rbf` | `lengthscale` (bounded) |
| `rbf_ard` | `lengthscales` (array of bounded, one per feature) |
| `matern` | `lengthscale` (bounded), `nu`: `"half"`, `"three_halves"`, or `"five_halves"` |
| `matern_ard` | `lengthscales` (array of bounded), `nu` |
| `periodic` | `lengthscale` (bounded), `period` (bounded) |
| `rational_quadratic` | `lengthscale` (bounded), `alpha` (bounded) |
| `rational_quadratic_ard` | `lengthscales` (array of bounded), `alpha` (bounded) |
| `constant` | `constant` (bounded) |
| `linear` | `variance` (bounded) |
| `white` | `variance` (bounded) |
| `sum`, `product` | `left`, `right`: each a kernel |
| `custom` | `persist_id` (string), `state` (any JSON) |

The format sets no depth limit on `sum` and `product` (see §8). An empty `lengthscales` array is `EmptyInput`.

### 5.2 `likelihood`

`{"noise_variance": <bounded>}`. The value is `σn²`, not its logarithm.

### 5.3 `jitter`

`{"fixed": {"jitter": j}}`, or `{"adaptive": {"initial", "multiplier", "max_retries", "max_jitter"}}`. Decoding goes through `JitterPolicy::fixed` / `JitterPolicy::adaptive`, so an invalid value is refused as it would be at construction.

### 5.4 Transforms

Each transform is stored twice. `*_unfitted` is the configuration you gave before fit. `*_transform` is the fitted map, with the statistics learned from the training data. **Load uses the stored fitted map and does not fit again**: an online model keeps the map fitted on its first training data while its points change.

Input, unfitted (`x_unfitted`):

| Form | Fields |
| --- | --- |
| `"identity"`, `"standardize"` | none |
| `min_max` | `range_lo`, `range_hi` |
| `pipeline` | `steps`: array of unfitted input maps, applied in order |
| `columnwise` | `maps`: array of unfitted input maps, one per feature |
| `custom` | `persist_id`, `state` |

Input, fitted (`x_transform`):

| Form | Fields |
| --- | --- |
| `"identity"` | none |
| `standardize` | `mean`, `std`: arrays, one value per feature |
| `min_max` | `data_min`, `data_max`: arrays, one value per feature; `range_lo`, `range_hi` |
| `pipeline` | `steps`: array of fitted input maps |
| `columnwise` | `maps`: array of fitted input maps |
| `custom` | `persist_id`, `state` |

Target, unfitted (`y_unfitted`): `"identity"`, `"standardize"`, `min_max` (`range_lo`, `range_hi`), `pipeline` (`steps`), `custom`. Target, fitted (`y_transform`): `"identity"`, `standardize` (`mean`, `std`: numbers), `min_max` (`data_min`, `data_max`, `range_lo`, `range_hi`: numbers), `pipeline` (`steps`), `custom`. There is no `columnwise` target.

### 5.5 `custom`: a kernel or a transform that is not built in

A `custom` entry carries a `persist_id` and a `state` (any JSON the type chose). On load, `PersistRegistry` maps the `persist_id` back to a restore function for that kind: a kernel, an unfitted input map, a fitted input map, an unfitted target map, or a fitted target map. Register them before `load`: `register_kernel`, `register_unfitted_input`, `register_fitted_input`, `register_unfitted_target`, `register_fitted_target`.

- Built-in kernels and maps use the closed tags above and are never registered.
- `persist_id` must be non-empty and must not start with `gprx.` (`RESERVED_PREFIX`). `save` and `register_*` both refuse it with `PersistFailed` (`kind: InvalidPersistId`).
- A `persist_id` with no registered restore on load is `PersistFailed` (`kind: UnregisteredId`). A `persist_id` registered twice for the same kind is `PersistFailed` (`kind: InvalidPersistId`).
- `save` of a `custom` kernel or map that does not implement `persist_id` is `PersistFailed` (`kind: NotPersistable`).

## 6. `model.safetensors`

The [safetensors](https://github.com/huggingface/safetensors) format: an 8-byte little-endian header length, a JSON header (name → `dtype`, `shape`, `data_offsets`), then the raw bytes. gprx writes no metadata entry.

**Matrices are column-major.** The header `shape` of `x` is `[n, d]`, but the bytes hold feature 0 for all `n` points, then feature 1, and so on: element `(i, j)` is at `j * n + i`. The same holds for `z` (`[m, d]`), `l` (`[n, n]`), and `q_l` (`[m, m]`). A reader in another language must read them as Fortran order (`order="F"` in NumPy), not as the row-major order the header's shape suggests.

### 6.1 Exact

| Tensor | Shape | dtype | In file | Content |
| --- | --- | --- | --- | --- |
| `x` | `[n, d]` | `F64` | always | The training `X` as you passed it, before any input map |
| `y` | `[n]` | `F64` | always | The training `y` as you passed it, before any target map |
| `l` | `[n, n]` | storage scalar: `F64` for `double`; `F32` for `single` and `mixed` | `has_factor` only | The factor, §6.4 |
| `alpha` | `[n]` | predict scalar: `F32` for `single`; `F64` for `double` and `mixed` | `has_factor` only | The weights `α` of the factored system, in the space of the transformed `y` |

### 6.2 Sparse (`sgpr`, `online_sgpr`, `svgp`)

All tensors are `F64`, whatever the model's precision (the precision is in `config.json`).

| Tensor | Shape | In file | Content |
| --- | --- | --- | --- |
| `x` | `[n, d]` | always | The training `X` as you passed it |
| `y` | `[n]` | always | The training `y` as you passed it |
| `z` | `[m, d]` | always | The inducing points `Z` as you passed them (original scale) |
| `z_train` | `[m, d]` | always | `Z` in the transformed space the model trains in. This is what load uses. It is stored on its own, because a `FreeInducing` fit moves `Z` there and mapping back and forth through the input map would not return the same bits |
| `q_mean` | `[m]` | `svgp` only | The mean of the whitened `q(u)` |
| `q_l` | `[m, m]` | `svgp` only | The lower Cholesky factor of the whitened `q(u)` covariance. Zero above the diagonal, positive on it |

### 6.3 What is not stored

No Gram matrix, no `W`, no distance cache, no `A = L⁻¹ K_mn`, no VFE system. They are rebuilt on load. Only the Exact factor `l` and `alpha` can be stored (§7).

### 6.4 The layout of `l`

`l` is `n × n`, column-major, with everything above the diagonal set to zero.

- `factor_kind: "llt"`: the Cholesky factor `L` of `K + σn² I + factor_jitter · I`, on and below the diagonal.
- `factor_kind: "ldlt"`: below the diagonal the unit-lower `L` of an LDLT factorization (the implicit unit diagonal is not stored); on the diagonal, `D`. The rows are in the order of `point_ids`.

## 7. `save` and `save_with_factor`; what load rebuilds

| Call | `has_factor` | Tensors | On load |
| --- | --- | --- | --- |
| `save` (Exact) | `false` | `x`, `y` | Builds the Gram matrix at the stored `θ`, factors it (retrying by the stored `jitter` policy), and solves for `α`. Costs `O(n³)` |
| `save_with_factor` (Exact) | `true` | `x`, `y`, `l`, `alpha` | Skips the factorization. An `F64` factor is memory-mapped (a later gprx save into the same directory renames a new file over it and leaves the mapped file alone; another program must not rewrite the file in place while the model lives; the tensor bytes must be 8-byte aligned or load fails with `PersistFailed`). An `F32` factor is copied out |
| `save` (`sgpr`, `online_sgpr`, `svgp`) | not written | as §6.2 | Applies the stored fitted maps to `x`; factors `K_mm` at the stored `θ` and `z_train`; rebuilds the VFE system (`sgpr`), or checks `q` and rebuilds `A` and `k_diag` (`svgp`) |

For `ldlt` and `online_sgpr`, the saved ids are restored, so the next `insert` returns the id it would have returned before `save`.

On load, `q_mean` and `q_l` must be finite, `q_l` lower triangular with a positive diagonal; every tensor must have the shape and dtype the config implies. The tests compare predictions of a saved-then-loaded Sparse or SVGP model with the original bit for bit (`tests/sparse_persist.rs`).

## 8. Versions and errors

`format_version` is the integer `1`, and `FORMAT_VERSION` is that constant. A file with another value is refused with `GprError::UnsupportedPersistVersion { found, supported }`. There is no migration. A key that is omitted reads as its default (§3): `model`, `precision`, `residual`, `math`, `factor_jitter`, and `distance_cache` work this way, so a file without them still loads. Keys the reader does not know are ignored.

| Condition | Error |
| --- | --- |
| File cannot be read or written; not valid JSON; a missing tensor; a wrong shape or dtype; an unaligned tensor; wrong loader for the `model`; `point_ids` of the wrong length; an unregistered or reserved `persist_id`; `q` not valid | `GprError::PersistFailed { kind, reason }`: `Io` (read / write), `Config` (JSON, keys, `point_ids`), `Tensor` (tensors, `q`), `WrongModel`, `UnregisteredId`, `InvalidPersistId`, `NotPersistable` |
| `format_version` is not `1` | `GprError::UnsupportedPersistVersion` |
| `n`, `d`, or (sparse) `m` is `0`; empty `lengthscales` | `GprError::EmptyInput` |
| A stored value that a constructor refuses (a bound, a jitter, a kernel parameter) | The constructor's own error |

Treat a directory as trusted input. A `sum` / `product` / `pipeline` / `columnwise` tree nested deeper than the JSON parser's limit (128 nested arrays or objects) fails with `PersistFailed` before it is decoded. The reader checks shapes and dtypes, that every stored tensor is finite (a `NaN` or `±∞` fails with `PersistFailed`), and, for `q`, finiteness and triangularity.

## 9. Reading the files without gprx

A minimal reader in Python (NumPy only), to check a file or to move a model to another tool. It parses the safetensors header itself and reads matrices as column-major:

```python
import json, struct
import numpy as np

DTYPES = {"F64": "<f8", "F32": "<f4"}

def load_tensors(path):
    raw = open(path, "rb").read()
    (header_len,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8:8 + header_len])
    body = raw[8 + header_len:]
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        lo, hi = meta["data_offsets"]
        flat = np.frombuffer(body[lo:hi], dtype=DTYPES[meta["dtype"]])
        # The shape in the header is [rows, cols], but the bytes are column-major.
        out[name] = flat.reshape(meta["shape"], order="F")
    return out

config = json.load(open("model_dir/config.json"))
t = load_tensors("model_dir/model.safetensors")
x, y = t["x"], t["y"]          # (n, d) and (n,), as you passed them to fit
```

`order="F"` is what makes `x[i, j]` the value of feature `j` at point `i`. Read with the default row-major order, the same bytes give a scrambled matrix whenever `d > 1`.
