[English](persist-format.md) | 日本語

# gprx の保存フォーマット

`save` が書き、`load` が読むもの。`format_version` は 1。コードは `src/persist/`。クレートの中での位置づけは [architecture.ja.md](architecture.ja.md)。1 段落の要約は [design §2](design.ja.md#2-全体アーキテクチャ概要)。

この文書の例は、`save` の実際の出力（8 点で fit してから `save`）で、手で書いたものではない。

## 1. 保存したモデルはディレクトリ

```text
<dir>/
  config.json          メタデータ: モデルの種類と、テンソル以外のすべての設定
  model.safetensors    数値: X、y、そして（モデルによって）Z、q(u)、L、α
```

`save` は、`<dir>`（と親ディレクトリ）が無ければ作り、2 つのファイルを置き換える。各ファイルは `<dir>` の一時ファイルに書いてから名前を付け替える（rename）。`model.safetensors` が先、`config.json` が最後である。読む側には古いファイルか新しいファイルのどちらかが見え、途中で失敗した保存は前の `config.json` を残す。それ以外は残さない。ソルバーと Cholesky のバッファの方針は保存しない。読み込んだモデルは `Fixed` で、`CholeskyBuffer::Retain` になる。もう一度学習するには、型のついたモデルで `with_optimizer` を呼んでから `refit` する。

## 2. ディレクトリの中身はどのモデルか

`config.json` の `model` キーで決まる。Exact のファイルには `model` キーが無く、`exact` として読む。

| `model` | 書くもの | 読むもの | config の追加分 | テンソル |
| --- | --- | --- | --- | --- |
| `exact`（キー無し） | `FittedGpr::save`、`FittedGpr::save_with_factor`、`OnlineGpr::save`、`OnlineGpr::save_with_factor` | `LoadedGpr::load` | `has_factor`、`factor_kind`、と 3 節の残り | `x`、`y`。因子があれば `l`、`alpha` も |
| `sgpr` | `FittedSgpr::save` | `LoadedSgpr::load` | `m`。4 節 | `x`、`y`、`z`、`z_train` |
| `online_sgpr` | `OnlineSgpr::save` | `LoadedSgpr::load` | `m`、`point_ids`、`next_point_id`、`inducing_ids`、`next_inducing_id` | `x`、`y`、`z`、`z_train` |
| `svgp` | `FittedSvgp::save` | `LoadedSvgp::load` | `m` | `x`、`y`、`z`、`z_train`、`q_mean`、`q_l` |

間違ったローダーで読むと、`GprError::PersistFailed`（`kind: WrongModel`）で断られ、メッセージに正しいローダーの名前が入る（例: "config.json holds a svgp model; load it with LoadedSvgp::load"）。`LoadedGpr::load` も同じ。

## 3. Exact の `config.json`

JSON のオブジェクトで、整形して書く。未知のキーは読むときに無視する。「省略」は、値が既定のときに書き手がキーを書かず、読み手が既定を補うこと。

| キー | 型 | 有無 | 意味 |
| --- | --- | --- | --- |
| `format_version` | 整数 | 必須 | `1`。最初に検査する（8 節） |
| `n` | 整数 | 必須 | 学習点の数。`0` は `EmptyInput` |
| `d` | 整数 | 必須 | 特徴の数。`0` は `EmptyInput` |
| `has_factor` | bool | 必須 | `l` と `alpha` がテンソルのファイルにあるか（7 節） |
| `factor_kind` | `"llt"` か `"ldlt"` | 必須 | `llt` は `FittedGpr`、`ldlt` は `OnlineGpr` として読む |
| `precision` | `"double"`、`"single"`、`"mixed"` | `double` なら省略 | 格納と予測のスカラー |
| `residual` | `"promote_storage"`、`"reevaluate_kernel"` | `promote_storage` なら省略 | 混合精度の残差。`precision` が `mixed` のときだけ読む |
| `math` | `"accurate"`、`"fast_approx"` | `accurate` なら省略 | カーネルの `exp`（`KernelExp`） |
| `kernel` | オブジェクト | 必須 | カーネルの木（5.1 節） |
| `likelihood` | オブジェクト | 必須 | 観測ノイズ（5.2 節） |
| `jitter` | オブジェクト | 必須 | `JitterPolicy`（5.3 節） |
| `factor_jitter` | 数 | `0` なら省略 | 保存した因子を作ったときの対角のジッター |
| `distance_cache` | `"always"`、`"never"` | 任意 | `DistanceCachePolicy`（`Cached` / `Uncached`）。無ければ既定の `Cached` |
| `x_unfitted` | オブジェクトか文字列 | 必須 | fit の前に設定した入力の変換（5.4 節） |
| `y_unfitted` | オブジェクトか文字列 | 必須 | fit の前に設定した目的変数の変換（5.4 節） |
| `x_transform` | オブジェクトか文字列 | 必須 | 学習後の入力の変換。学習した統計を持つ（5.4 節） |
| `y_transform` | オブジェクトか文字列 | 必須 | 学習後の目的変数の変換（5.4 節） |
| `point_ids` | 整数の配列 | `ldlt` では必須、それ以外は無し | 各行の `PointId`。行の順。長さは `n` に等しい |
| `next_point_id` | 整数 | `ldlt` では必須、それ以外は無し | 次に `insert` が返す id |

`Constant × RBF`、`MinMaxInput`、`StandardizeTarget` のモデルを `FittedGpr::save` した例:

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

（実際のファイルは 1 キー 1 行でインデントされる。ここでは収めるために畳んだ。同じモデルを `OnlineGpr::save_with_factor` すると、`has_factor: true`、`factor_kind: "ldlt"` になり、`"point_ids": [0, 1, 2, 3, 4, 5, 6, 7]` と `"next_point_id": 8` が加わる。）

浮動小数点数は、同じ `f64` に読み戻せる最短の形で書き、serde_json の `float_roundtrip` で読むので、保存した数はビットまで戻る。

## 4. Sparse の `config.json`

`sgpr`、`online_sgpr`、`svgp` は 1 つの配置を共有する。Exact のものに対して、次の違いがある。

| キー | 違い |
| --- | --- |
| `model` | 追加、必須: `"sgpr"`、`"online_sgpr"`、`"svgp"` |
| `m` | 追加、必須: 誘導点の数。`0` は `EmptyInput` |
| `jitter` | `K_mm` を分解するときの方針（Sparse の既定は adaptive: `initial` 1e-8、`multiplier` 10、`max_retries` 5、`max_jitter` 1e-3） |
| `has_factor`、`factor_kind`、`factor_jitter`、`distance_cache` | 書かない。因子は保存しない（7 節） |
| `inducing_ids`、`next_inducing_id` | 追加。`online_sgpr` では必須、それ以外は無し。`online_sgpr` では `point_ids` と `next_point_id` も必須 |

`precision`、`residual`、`math`、`kernel`、`likelihood`、4 つの変換のキーは、3 節と同じ形。例（`svgp`、同じカーネルと変換、誘導点 2 つ）:

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

## 5. JSON の形

enum は serde の外部タグで、名前は `snake_case`。フィールドの無い variant は文字列（`"identity"`、`"always"`）。フィールドのある variant は、タグ 1 つをキーとするオブジェクト（`{"rbf": {…}}`）。カーネル、変換、ジッター、モデルのタグのすべてがこうなる。

区間つきのパラメータは、オブジェクト `{"value": v, "lo": lo, "hi": hi}`。値とその区間。読むときに、有限でない境界、`lo >= hi`、有限でない `value`、`lo` と `hi` の間（両端を含まない）に無い `value` を断る。

### 5.1 `kernel`

| タグ | フィールド |
| --- | --- |
| `rbf` | `lengthscale`（区間つき） |
| `rbf_ard` | `lengthscales`（区間つきの配列。特徴ごとに 1 つ） |
| `matern` | `lengthscale`（区間つき）、`nu`: `"half"`、`"three_halves"`、`"five_halves"` |
| `matern_ard` | `lengthscales`（区間つきの配列）、`nu` |
| `periodic` | `lengthscale`（区間つき）、`period`（区間つき） |
| `rational_quadratic` | `lengthscale`（区間つき）、`alpha`（区間つき） |
| `rational_quadratic_ard` | `lengthscales`（区間つきの配列）、`alpha`（区間つき） |
| `constant` | `constant`（区間つき） |
| `linear` | `variance`（区間つき） |
| `white` | `variance`（区間つき） |
| `sum`、`product` | `left`、`right`: それぞれカーネル |
| `custom` | `persist_id`（文字列）、`state`（任意の JSON） |

`sum` と `product` の深さに、フォーマットは上限を置かない（8 節）。`lengthscales` が空の配列なら `EmptyInput`。

### 5.2 `likelihood`

`{"noise_variance": <区間つき>}`。値は `σn²` で、その対数ではない。

### 5.3 `jitter`

`{"fixed": {"jitter": j}}`、または `{"adaptive": {"initial", "multiplier", "max_retries", "max_jitter"}}`。読むときは `JitterPolicy::fixed` / `JitterPolicy::adaptive` を通すので、不正な値は、作るときと同じように断られる。

### 5.4 変換

変換は 2 回保存する。`*_unfitted` は、fit の前に与えた設定。`*_transform` は、学習後の変換で、学習データから得た統計を持つ。**読み込みは、保存した学習後の変換を使い、もう一度当てはめない。** オンラインのモデルは、点が変わっても、最初の学習データで当てはめた変換を使い続けるためである。

入力、学習前（`x_unfitted`）:

| 形 | フィールド |
| --- | --- |
| `"identity"`、`"standardize"` | なし |
| `min_max` | `range_lo`、`range_hi` |
| `pipeline` | `steps`: 学習前の入力の変換の配列。順に適用する |
| `columnwise` | `maps`: 学習前の入力の変換の配列。特徴ごとに 1 つ |
| `custom` | `persist_id`、`state` |

入力、学習後（`x_transform`）:

| 形 | フィールド |
| --- | --- |
| `"identity"` | なし |
| `standardize` | `mean`、`std`: 配列。特徴ごとに 1 つ |
| `min_max` | `data_min`、`data_max`: 配列。特徴ごとに 1 つ。`range_lo`、`range_hi` |
| `pipeline` | `steps`: 学習後の入力の変換の配列 |
| `columnwise` | `maps`: 学習後の入力の変換の配列 |
| `custom` | `persist_id`、`state` |

目的変数、学習前（`y_unfitted`）: `"identity"`、`"standardize"`、`min_max`（`range_lo`、`range_hi`）、`pipeline`（`steps`）、`custom`。目的変数、学習後（`y_transform`）: `"identity"`、`standardize`（`mean`、`std`: 数）、`min_max`（`data_min`、`data_max`、`range_lo`、`range_hi`: 数）、`pipeline`（`steps`）、`custom`。目的変数に `columnwise` は無い。

### 5.5 `custom`: 組み込みでないカーネルや変換

`custom` の項目は、`persist_id` と `state`（その型が選んだ任意の JSON）を持つ。読み込みでは、`PersistRegistry` が `persist_id` を、その種類の復元関数に対応づける。対象は次の 5 種類で、`load` の前に関数で登録する:

| 復元対象 | 登録関数 |
| --- | --- |
| カーネル | `register_kernel` |
| 学習前の入力の変換 | `register_unfitted_input` |
| 学習後の入力の変換 | `register_fitted_input` |
| 学習前の目的変数の変換 | `register_unfitted_target` |
| 学習後の目的変数の変換 | `register_fitted_target` |

- 組み込みのカーネルと変換は、上の閉じたタグを使い、登録しない。
- `persist_id` は空でなく、`gprx.`（`RESERVED_PREFIX`）で始まってはならない。`save` と `register_*` のどちらも、`PersistFailed`（`kind: InvalidPersistId`）で断る。
- 読み込みで、復元が登録されていない `persist_id` は `PersistFailed`（`kind: UnregisteredId`）。同じ種類に同じ `persist_id` を 2 回登録すると `PersistFailed`（`kind: InvalidPersistId`）。
- `persist_id` を実装していない `custom` のカーネルや変換を `save` すると `PersistFailed`（`kind: NotPersistable`）。

## 6. `model.safetensors`

[safetensors](https://github.com/huggingface/safetensors) の形式: リトルエンディアンの 8 バイトのヘッダ長、JSON のヘッダ（名前 → `dtype`、`shape`、`data_offsets`）、そして生のバイト列。gprx はメタデータの項目を書かない。

**行列は列優先。** `x` のヘッダの `shape` は `[n, d]` だが、バイト列は、特徴 0 の全 `n` 点、次に特徴 1、という順に並ぶ。要素 `(i, j)` は `j * n + i` にある。`z`（`[m, d]`）、`l`（`[n, n]`）、`q_l`（`[m, m]`）も同じ。ほかの言語で読むときは、ヘッダの shape が示す行優先ではなく、Fortran の順で読む（NumPy なら `order="F"`）。

### 6.1 Exact

| テンソル | 形 | dtype | ファイルに | 中身 |
| --- | --- | --- | --- | --- |
| `x` | `[n, d]` | `F64` | 常に | 渡したままの学習 `X`。入力の変換の前 |
| `y` | `[n]` | `F64` | 常に | 渡したままの学習 `y`。目的変数の変換の前 |
| `l` | `[n, n]` | 格納のスカラー: `double` は `F64`、`single` と `mixed` は `F32` | `has_factor` のときだけ | 因子。6.4 節 |
| `alpha` | `[n]` | 予測のスカラー: `single` は `F32`、`double` と `mixed` は `F64` | `has_factor` のときだけ | 分解した系の重み `α`。変換後の `y` の空間 |

### 6.2 Sparse（`sgpr`、`online_sgpr`、`svgp`）

すべてのテンソルは、モデルの精度によらず `F64`（精度は `config.json` にある）。

| テンソル | 形 | ファイルに | 中身 |
| --- | --- | --- | --- |
| `x` | `[n, d]` | 常に | 渡したままの学習 `X` |
| `y` | `[n]` | 常に | 渡したままの学習 `y` |
| `z` | `[m, d]` | 常に | 渡したままの誘導点 `Z`（元の尺度） |
| `z_train` | `[m, d]` | 常に | モデルが学習する、変換後の空間の `Z`。読み込みが使うのはこちら。別に保存するのは、`FreeInducing` の fit が動かすのはこちらで、入力の変換を往復すると同じビットに戻らないため |
| `q_mean` | `[m]` | `svgp` のみ | whitened な `q(u)` の平均 |
| `q_l` | `[m, m]` | `svgp` のみ | whitened な `q(u)` の共分散の下三角の Cholesky 因子。対角より上は 0、対角は正 |

### 6.3 保存しないもの

Gram 行列、`W`、距離キャッシュ、`A = L⁻¹ K_mn`、VFE の系は保存しない。読み込みで組み直す。保存できるのは、Exact の因子 `l` と `alpha` だけ（7 節）。

### 6.4 `l` の並び

`l` は `n × n` で列優先。対角より上はすべて 0。

- `factor_kind: "llt"`: `K + σn² I + factor_jitter · I` の Cholesky 因子 `L`。対角とその下。
- `factor_kind: "ldlt"`: 対角の下に、LDLT 分解の単位下三角の `L`（暗黙の単位対角は保存しない）。対角に `D`。行の順は `point_ids` の順。

## 7. `save` と `save_with_factor`、読み込みが組み直すもの

| 呼び出し | `has_factor` | テンソル | 読み込み |
| --- | --- | --- | --- |
| `save`（Exact） | `false` | `x`、`y` | 保存した `θ` で Gram 行列を作り、分解し（保存した `jitter` の方針で再試行する）、`α` を解く。`O(n³)` |
| `save_with_factor`（Exact） | `true` | `x`、`y`、`l`、`alpha` | 分解を省く。`F64` の因子はメモリマップする（要件は下記）。`F32` の因子はコピーして取り出す |
| `save`（`sgpr`、`online_sgpr`、`svgp`） | 書かない | 6.2 節 | 保存した学習後の変換を `x` にかけ、保存した `θ` と `z_train` で `K_mm` を分解し、VFE の系を組み直す（`sgpr`）。または `q` を検査して `A` と `k_diag` を組み直す（`svgp`） |

`save_with_factor` で `F64` の因子をメモリマップする要件:
- 同じディレクトリへの gprx の保存は新しいファイルを rename で置くので、マップ中のファイルには触れない。
- モデルを使っているあいだ、他のプログラムがファイルをその場で書き換えないこと。
- テンソルのバイト列は 8 バイトに揃っていること。揃っていなければ `PersistFailed`。

`ldlt` と `online_sgpr` では、保存した id を復元するので、次の `insert` は、`save` の前に返したはずの id を返す。

読み込みでは、`q_mean` と `q_l` が有限で、`q_l` が対角の正な下三角でなければならず、すべてのテンソルが config から決まる形と dtype でなければならない。保存して読み込んだ Sparse / SVGP のモデルの予測が、元とビットまで一致することを、テストが確かめている（`tests/sparse_persist.rs`）。

## 8. 版とエラー

`format_version` は整数の `1` で、`FORMAT_VERSION` がその定数。ほかの値のファイルは `GprError::UnsupportedPersistVersion { found, supported }` で断る。移行は無い。省略されたキーは既定として読む（3 節）: `model`、`precision`、`residual`、`math`、`factor_jitter`、`distance_cache` がそうで、これらの無いファイルも読める。読み手が知らないキーは無視する。

| 状況 | エラー |
| --- | --- |
| ファイルを読み書きできない | `GprError::PersistFailed { kind: Io, reason }` |
| JSON として不正、キー、`point_ids` の長さが違う | `GprError::PersistFailed { kind: Config, reason }` |
| テンソルが無い、形か dtype が違う、テンソルが揃っていない、`q` が不正 | `GprError::PersistFailed { kind: Tensor, reason }` |
| `model` に対してローダーが違う | `GprError::PersistFailed { kind: WrongModel, reason }` |
| 登録されていない `persist_id` | `GprError::PersistFailed { kind: UnregisteredId, reason }` |
| 予約された `persist_id` | `GprError::PersistFailed { kind: InvalidPersistId, reason }` |
| 同じ種類に同じ `persist_id` を 2 回登録した | `GprError::PersistFailed { kind: InvalidPersistId, reason }` |
| `persist_id` を実装していない `custom` のカーネルや変換の保存 | `GprError::PersistFailed { kind: NotPersistable, reason }` |
| `format_version` が `1` でない | `GprError::UnsupportedPersistVersion` |
| `n`、`d`、（Sparse の）`m` が `0`、`lengthscales` が空 | `GprError::EmptyInput` |
| コンストラクタが断る保存値（境界、ジッター、カーネルのパラメータ） | そのコンストラクタ自身のエラー |

ディレクトリは、信頼できる入力として扱う。JSON パーサーの上限（配列とオブジェクトの入れ子 128 段）より深い `sum` / `product` / `pipeline` / `columnwise` の木は、デコードの前に `PersistFailed` になる。読み込みが検査するのは、形と dtype、保存したテンソルがすべて有限であること（`NaN` や `±∞` は `PersistFailed`）、および `q` の有限性と下三角であること。

## 9. gprx なしでファイルを読む

Python（NumPy のみ）の最小の読み手。ファイルの確認や、別のツールへモデルを移すのに使う。safetensors のヘッダを自分で解析し、行列を列優先で読む:

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
        # ヘッダの shape は [行, 列] だが、バイト列は列優先。
        out[name] = flat.reshape(meta["shape"], order="F")
    return out

config = json.load(open("model_dir/config.json"))
t = load_tensors("model_dir/model.safetensors")
x, y = t["x"], t["y"]          # (n, d) と (n,)。fit に渡したまま
```

`order="F"` が、`x[i, j]` を「点 `i` の特徴 `j` の値」にする。既定の行優先で読むと、`d > 1` のときは、同じバイト列が入れ替わった行列になる。
