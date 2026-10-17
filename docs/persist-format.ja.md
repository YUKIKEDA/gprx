[English](persist-format.md) | 日本語

# gprx の保存フォーマット

この文書は、`save` が書き出し、`load` が読み込む内容の仕様である。`format_version` は 1。実装は `src/persist/` にある。クレート全体の中での位置づけは [architecture.ja.md](architecture.ja.md) に、1 段落の要約は [design §2](design.ja.md#2-全体アーキテクチャ概要) にまとめてある。

文中の例は手書きではない。8 点で fit したモデルを実際に `save` した出力である。

## 1. 保存したモデルはディレクトリになる

```text
<dir>/
  config.json          メタデータ: モデルの種類と、テンソル以外のすべての設定
  model.safetensors    数値: X、y、そして（モデルによって）Z、q(u)、L、α
```

`save` は、`<dir>` が無ければ親ディレクトリごと作成し、中の 2 つのファイルを置き換える。どちらのファイルも、いったん `<dir>` 内の一時ファイルに書き出してから、本来の名前へ rename して上書きする。順序は `model.safetensors` が先で、`config.json` が最後である。このため、読み込む側から見えるのは、常に古いファイルか新しいファイルのどちらか一方になる。保存が途中で失敗した場合は、直前の `config.json` がそのまま残り、書きかけの一時ファイルは残らない。

ソルバーと Cholesky のバッファの方針は保存の対象外である。読み込んだモデルの最適化器は `Fixed` になり、バッファの方針は `CholeskyBuffer::Retain` になる。もう一度学習させたいときは、`FittedGpr` などの具体的な型のモデルに対して `with_optimizer` を呼び、続けて `refit` を呼ぶ。

## 2. ディレクトリの中身はどのモデルか

保存されているモデルの種類は、`config.json` の `model` キーで判別する。Exact のファイルにはこのキーが無く、読み込み時には `exact` として扱われる。

| `model` | 書き出すメソッド | 読み込むローダー | config に加わるキー | テンソル |
| --- | --- | --- | --- | --- |
| `exact`（キー無し） | `FittedGpr::save`、`FittedGpr::save_with_factor`、`OnlineGpr::save`、`OnlineGpr::save_with_factor` | `LoadedGpr::load` | `has_factor`、`factor_kind` ほか 3 節のキー | `x`、`y`。因子を保存したときは `l`、`alpha` も |
| `sgpr` | `FittedSgpr::save` | `LoadedSgpr::load` | `m`。ほかは 4 節 | `x`、`y`、`z`、`z_train` |
| `online_sgpr` | `OnlineSgpr::save` | `LoadedSgpr::load` | `m`、`point_ids`、`next_point_id`、`inducing_ids`、`next_inducing_id` | `x`、`y`、`z`、`z_train` |
| `svgp` | `FittedSvgp::save` | `LoadedSvgp::load` | `m` | `x`、`y`、`z`、`z_train`、`q_mean`、`q_l` |

`DistanceKernel`（二乗距離を呼び出し側が与えるカーネル）のモデルも `model` の値は同じで、そこに `distance` キーが加わる。このモデルは、カーネルのマーカーに対応する `LoadedDistanceGpr`、`LoadedDistanceSgpr`、`LoadedDistanceSvgp` で読み込む（10 節）。

ローダーを取り違えると、読み込みは `GprError::PersistFailed`（`kind: WrongModel`）で失敗する。エラーメッセージには正しいローダーの名前が入る（例: "config.json holds a svgp model; load it with LoadedSvgp::load"）。`LoadedGpr::load` を含むすべてのローダーがこの検査を行い、座標のファイルを距離のローダーに渡した場合や、その逆の場合にも `WrongModel` を返す。

## 3. Exact の `config.json`

`config.json` は整形（pretty-print）した JSON オブジェクトである。読み込み時、未知のキーは無視する。表の「省略」は、値が既定値のときに保存側がキーを書き出さず、読み込み側が既定値を補うことを指す。

| キー | 型 | 有無 | 意味 |
| --- | --- | --- | --- |
| `format_version` | 整数 | 必須 | `1`。読み込みで最初に検査する（8 節） |
| `n` | 整数 | 必須 | 学習点の数。`0` なら `EmptyInput` |
| `d` | 整数 | 必須 | 特徴量の数。`0` なら `EmptyInput`。ただし `distance_only` のモデルでは `d` が `0` になる（10 節） |
| `has_factor` | bool | 必須 | テンソルのファイルに `l` と `alpha` が入っているか（7 節） |
| `factor_kind` | `"llt"` か `"ldlt"` | 必須 | `llt` なら `FittedGpr`、`ldlt` なら `OnlineGpr` として読み込む |
| `precision` | `"double"`、`"single"`、`"mixed"` | `double` なら省略 | 格納と予測に使うスカラーの型 |
| `residual` | `"promote_storage"`、`"reevaluate_kernel"` | `promote_storage` なら省略 | 混合精度での残差の扱い。`precision` が `mixed` のときだけ参照する |
| `math` | `"accurate"`、`"fast_approx"` | `accurate` なら省略 | カーネルの `exp` の計算方法（`KernelExp`） |
| `kernel` | オブジェクト | 必須 | カーネルの木（5.1 節） |
| `likelihood` | オブジェクト | 必須 | 観測ノイズ（5.2 節） |
| `jitter` | オブジェクト | 必須 | `JitterPolicy`（5.3 節） |
| `factor_jitter` | 数 | `0` なら省略 | 保存した因子を作ったときに対角へ加えたジッター |
| `distance_cache` | `"always"`、`"never"` | 任意 | `DistanceCachePolicy`（`Cached` / `Uncached`）。キーが無ければ既定の `Cached` として読む |
| `x_unfitted` | オブジェクトか文字列 | 必須 | 学習前に設定した入力の変換（5.4 節） |
| `y_unfitted` | オブジェクトか文字列 | 必須 | 学習前に設定した目的変数の変換（5.4 節） |
| `x_transform` | オブジェクトか文字列 | 必須 | 学習後の入力の変換。fit で学習した統計を含む（5.4 節） |
| `y_transform` | オブジェクトか文字列 | 必須 | 学習後の目的変数の変換（5.4 節） |
| `point_ids` | 整数の配列 | `ldlt` なら必須、それ以外では書かない | 各行の `PointId` を行の順に並べたもの。長さは `n` と一致しなければならない |
| `next_point_id` | 整数 | `ldlt` なら必須、それ以外では書かない | 次の `insert` が割り当てる id |
| `distance` | オブジェクト | `DistanceKernel` のモデルのみ | マーカーと slot の表（10 節） |

次の例は、`Constant × RBF` のカーネルに `MinMaxInput` と `StandardizeTarget` を組み合わせたモデルを、`FittedGpr::save` で保存したものである。

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

（実際のファイルは 1 行に 1 キーずつインデントされている。ここでは収めるために折りたたんだ。同じモデルを `OnlineGpr::save_with_factor` で保存すると、`has_factor: true`、`factor_kind: "ldlt"` に変わり、`"point_ids": [0, 1, 2, 3, 4, 5, 6, 7]` と `"next_point_id": 8` が加わる。）

浮動小数点数は、読み戻したときに同じ `f64` になる最短の表記で書き出し、serde_json の `float_roundtrip` で読み込む。そのため、保存した値はビット単位で元どおりに復元される。

## 4. Sparse の `config.json`

`sgpr`、`online_sgpr`、`svgp` の 3 つは、同じキー構成を共有している。Exact の構成との違いは次のとおりである。

| キー | 違い |
| --- | --- |
| `model` | 追加（必須）。`"sgpr"`、`"online_sgpr"`、`"svgp"` のいずれか |
| `m` | 追加（必須）。誘導点の数。`0` なら `EmptyInput` |
| `jitter` | `K_mm` を分解するときの方針。Sparse の既定は adaptive（`initial` 1e-8、`multiplier` 10、`max_retries` 5、`max_jitter` 1e-3） |
| `has_factor`、`factor_kind`、`factor_jitter`、`distance_cache` | 書き出さない。Sparse では因子を保存しない（7 節） |
| `inducing_ids`、`next_inducing_id` | 追加。`online_sgpr` では必須で、それ以外では書かない。`online_sgpr` は `point_ids` と `next_point_id` も必須 |
| `distance` | `DistanceKernel` のモデルにだけ追加する。Sparse の場合は中に `inducing` も入る（10 節） |

`precision`、`residual`、`math`、`kernel`、`likelihood` と 4 つの変換のキーは、3 節と同じ形で書く。次の例は `svgp` のモデルで、データと変換は 3 節の例と同じ、カーネルは RBF 単体、誘導点は 2 つである。`save` が書き出すキーはすべて示してある。3 節の例と違うのは、`model`、`m`、`kernel`、`jitter` の値と、`has_factor`、`factor_kind`、`distance_cache` が無いことである。

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

enum は serde の外部タグ形式で表し、名前は `snake_case` にする。フィールドを持たない variant は文字列になる（`"identity"`、`"always"`）。フィールドを持つ variant は、タグを唯一のキーとするオブジェクトになる（`{"rbf": {…}}`）。カーネル、変換、ジッター、モデルのタグは、いずれもこの規則に従う。

区間つきのパラメータは `{"value": v, "lo": lo, "hi": hi}` というオブジェクトで、値とその区間を表す。デコードでは、境界が有限でないもの、`lo >= hi` のもの、`value` が有限でないもの、`value` が `lo` と `hi` の間（両端は含まない）に収まらないものをエラーにする。

### 5.1 `kernel`

| タグ | フィールド |
| --- | --- |
| `rbf` | `lengthscale`（区間つき） |
| `rbf_ard` | `lengthscales`（区間つきの配列。特徴量ごとに 1 つ） |
| `matern` | `lengthscale`（区間つき）、`nu`: `"half"`、`"three_halves"`、`"five_halves"` のいずれか |
| `matern_ard` | `lengthscales`（区間つきの配列）、`nu` |
| `periodic` | `lengthscale`（区間つき）、`period`（区間つき） |
| `rational_quadratic` | `lengthscale`（区間つき）、`alpha`（区間つき） |
| `rational_quadratic_ard` | `lengthscales`（区間つきの配列）、`alpha`（区間つき） |
| `constant` | `constant`（区間つき） |
| `linear` | `variance`（区間つき） |
| `white` | `variance`（区間つき） |
| `sum`、`product` | `left`、`right`: どちらもカーネル |
| `custom` | `persist_id`（文字列）、`state`（任意の JSON） |
| `distance` | `slot`（整数。slot の表での位置）、`leaf`: スカラーの slot なら `rbf`、`matern`、`periodic`、`rational_quadratic`、`custom` のいずれかのカーネル。ARD の slot なら、次元ごとに長さスケールを 1 つ持つ `rbf_ard`、`matern_ard`、`rational_quadratic_ard` のいずれかのカーネル（10 節） |

`sum` と `product` の入れ子の深さについて、フォーマット自体は上限を設けていない（8 節を参照）。`lengthscales` が空の配列なら `EmptyInput` になる。

### 5.2 `likelihood`

`{"noise_variance": <区間つき>}` の形で書く。値は `σn²` そのもので、対数ではない。

### 5.3 `jitter`

`{"fixed": {"jitter": j}}` か `{"adaptive": {"initial", "multiplier", "max_retries", "max_jitter"}}` のどちらかである。デコードは `JitterPolicy::fixed` / `JitterPolicy::adaptive` を経由するため、不正な値は、これらで直接作ろうとしたときと同じエラーになる。

### 5.4 変換

変換は 1 つにつき 2 回保存する。`*_unfitted` には学習前に与えた設定が、`*_transform` には学習データから得た統計を含む学習後の変換が入る。**読み込みでは保存済みの学習後の変換をそのまま使い、変換を学習し直すことはしない。** そのため、オンラインのモデルでは点が入れ替わっても、最初の学習データで学習した変換が使われ続ける。

入力の学習前の変換（`x_unfitted`）:

| 形 | フィールド |
| --- | --- |
| `"identity"`、`"standardize"` | なし |
| `min_max` | `range_lo`、`range_hi` |
| `pipeline` | `steps`: 学習前の入力の変換の配列。先頭から順に適用する |
| `columnwise` | `maps`: 学習前の入力の変換の配列。特徴量ごとに 1 つ |
| `custom` | `persist_id`、`state` |

入力の学習後の変換（`x_transform`）:

| 形 | フィールド |
| --- | --- |
| `"identity"` | なし |
| `standardize` | `mean`、`std`: 特徴量ごとに 1 つの値を持つ配列 |
| `min_max` | `data_min`、`data_max`: 特徴量ごとに 1 つの値を持つ配列。`range_lo`、`range_hi` |
| `pipeline` | `steps`: 学習後の入力の変換の配列 |
| `columnwise` | `maps`: 学習後の入力の変換の配列 |
| `custom` | `persist_id`、`state` |

目的変数の学習前の変換（`y_unfitted`）は、`"identity"`、`"standardize"`、`min_max`（`range_lo`、`range_hi`）、`pipeline`（`steps`）、`custom` のいずれかである。学習後の変換（`y_transform`）は、`"identity"`、`standardize`（`mean`、`std`: 数）、`min_max`（`data_min`、`data_max`、`range_lo`、`range_hi`: 数）、`pipeline`（`steps`）、`custom` のいずれかになる。目的変数には `columnwise` が無い。

### 5.5 `custom`: 組み込みでないカーネルや変換

`custom` の項目には、`persist_id` と `state`（その型の実装が決める任意の JSON）が入る。読み込み時には、`PersistRegistry` が `persist_id` を種類ごとの復元関数に対応づける。種類は次の 5 つある。`load` を呼ぶ前に、復元関数を種類に合った次の登録関数で `PersistRegistry` に登録しておく。

| 復元対象 | 登録関数 |
| --- | --- |
| カーネル | `register_kernel` |
| 学習前の入力の変換 | `register_unfitted_input` |
| 学習後の入力の変換 | `register_fitted_input` |
| 学習前の目的変数の変換 | `register_unfitted_target` |
| 学習後の目的変数の変換 | `register_fitted_target` |

- 組み込みのカーネルと変換は、上に挙げた固定のタグで書かれ、登録の対象にはならない。
- `persist_id` は空であってはならず、`gprx.`（`RESERVED_PREFIX`）で始まってもいけない。この条件に反する id は、`save` でも `register_*` でも `PersistFailed`（`kind: InvalidPersistId`）になる。
- 読み込み時に、復元関数が登録されていない `persist_id` があると、`PersistFailed`（`kind: UnregisteredId`）で失敗する。同じ種類に同じ `persist_id` を 2 回登録した場合は `PersistFailed`（`kind: InvalidPersistId`）になる。
- `persist_id` を実装していない `custom` のカーネルや変換を `save` しようとすると、`PersistFailed`（`kind: NotPersistable`）になる。

## 6. `model.safetensors`

形式は [safetensors](https://github.com/huggingface/safetensors) である。先頭にリトルエンディアンで 8 バイトのヘッダ長があり、続いて JSON のヘッダ（名前 → `dtype`、`shape`、`data_offsets`）、最後に生のバイト列が並ぶ。gprx はメタデータの項目を書き出さない。

**行列は列優先で並ぶ。** `x` のヘッダの `shape` は `[n, d]` だが、バイト列には全 `n` 点の特徴 0 が先に並び、そのあとに全 `n` 点の特徴 1 が並ぶ。以降も同じ順に続く。要素 `(i, j)` の位置は `j * n + i` である。`z`（`[m, d]`）、`l`（`[n, n]`）、`q_l`（`[m, m]`）も同じ並びになっている。ほかの言語から読む場合は、ヘッダの shape から思い浮かべがちな行優先ではなく、Fortran 順（NumPy なら `order="F"`）として解釈する必要がある。

### 6.1 Exact

| テンソル | 形 | dtype | 保存される条件 | 中身 |
| --- | --- | --- | --- | --- |
| `x` | `[n, d]` | `F64` | 常に | fit に渡したままの学習用の `X`。入力の変換をかける前の値 |
| `y` | `[n]` | `F64` | 常に | fit に渡したままの学習用の `y`。目的変数の変換をかける前の値 |
| `l` | `[n, n]` | 格納のスカラー: `double` なら `F64`、`single` と `mixed` なら `F32` | `has_factor` が `true` のときだけ | 因子（6.4 節） |
| `alpha` | `[n]` | 予測のスカラー: `single` なら `F32`、`double` と `mixed` なら `F64` | `has_factor` が `true` のときだけ | 分解した系の重み `α`。変換後の `y` の空間での値 |

### 6.2 Sparse（`sgpr`、`online_sgpr`、`svgp`）

Sparse のテンソルは、モデルの精度にかかわらずすべて `F64` で保存する（精度そのものは `config.json` に記録する）。

| テンソル | 形 | 保存される条件 | 中身 |
| --- | --- | --- | --- |
| `x` | `[n, d]` | 常に | fit に渡したままの学習用の `X` |
| `y` | `[n]` | 常に | fit に渡したままの学習用の `y` |
| `z` | `[m, d]` | 常に | 渡したままの誘導点 `Z`（元の尺度） |
| `z_train` | `[m, d]` | 常に | モデルが学習に使う、変換後の空間での `Z`。読み込みで使うのはこちらである。`FreeInducing` の fit はこの空間で `Z` を動かす。この `Z` を入力の変換で元の尺度へ戻し、もう一度変換しても、同じビットには戻らない。そのため `z` とは別に保存する |
| `q_mean` | `[m]` | `svgp` のみ | whitened 表現の `q(u)` の平均 |
| `q_l` | `[m, m]` | `svgp` のみ | whitened 表現の `q(u)` の共分散の、下三角の Cholesky 因子。対角より上は 0 で、対角成分は正 |

`DistanceKernel` のモデルでは、slot ごとにテンソル `d2.<k>` が 1 つ加わる（10 節）。

### 6.3 保存しないもの

Gram 行列、`W`、距離キャッシュ、`A = L⁻¹ K_mn`、VFE の系は、どれも保存せず、読み込み時に組み直す。保存できるのは、Exact の因子 `l` と重み `alpha` だけである（7 節）。

### 6.4 `l` の並び

`l` は `n × n` の列優先の行列で、対角より上はすべて 0 にしてある。

- `factor_kind: "llt"`: `K + σn² I + factor_jitter · I` の Cholesky 因子 `L` を、対角とその下に置く。
- `factor_kind: "ldlt"`: 対角の下には LDLT 分解の単位下三角行列 `L` を置き（暗黙の単位対角は保存しない）、対角には `D` を置く。行は `point_ids` の順に並ぶ。

## 7. `save` と `save_with_factor` の違いと、読み込み時に組み直すもの

| 呼び出し | `has_factor` | テンソル | 読み込み時の処理 |
| --- | --- | --- | --- |
| `save`（Exact） | `false` | `x`、`y` | 保存した `θ` で Gram 行列を作って分解し（失敗すれば保存した `jitter` の方針で再試行する）、`α` を求める。計算量は `O(n³)` |
| `save_with_factor`（Exact） | `true` | `x`、`y`、`l`、`alpha` | 分解を省く。`llt` のファイルにある `F64` の因子はメモリマップする（条件は表の下に示す）。`F32` の因子と、`ldlt` のファイルの因子（online モデル自身のバッファに入れる）は、ファイルからコピーして取り出す |
| `save`（`sgpr`、`online_sgpr`、`svgp`） | 書き出さない | 6.2 節のとおり | 保存した学習後の変換を `x` に適用し、保存した `θ` と `z_train` で `K_mm` を分解する。そのうえで、`sgpr` なら VFE の系を組み直し、`svgp` なら `q` を検査してから `A` と `k_diag` を組み直す |

`save_with_factor` で保存した `F64` の因子をメモリマップするときの前提:
- gprx が同じディレクトリへ保存し直しても、新しいファイルを rename で置き換えるだけなので、マップ中のファイルの中身は変わらない。
- モデルを使っているあいだ、ほかのプログラムがそのファイルをその場で書き換えてはならない。
- テンソルのバイト列は 8 バイト境界に揃っていなければならない。揃っていなければ、読み込みは `PersistFailed` で失敗する。

`ldlt` と `online_sgpr` では、保存した id も復元する。id の採番は保存の前から途切れずに続く。読み込んだあとの次の `insert` が返す id は、保存しなかったとして元のモデルで次に `insert` したときに返るはずの id と同じである。

読み込み時には、`q_mean` と `q_l` が有限であること、`q_l` が下三角行列で対角成分が正であることを確かめる。すべてのテンソルの形と dtype も、config の内容と一致していなければならない。保存してから読み込んだ Sparse / SVGP のモデルの予測が元のモデルとビット単位で一致することは、`tests/sparse_persist.rs` のテストで確かめている。

## 8. 版とエラー

`format_version` は整数の `1` で、定数 `FORMAT_VERSION` がこの値を持つ。これ以外の値を持つファイルは、`GprError::UnsupportedPersistVersion { found, supported }` で読み込みに失敗する。別の版のファイルを変換して読む仕組み（移行）は用意していない。省略されたキーは既定値として読む（3 節）。`model`、`precision`、`residual`、`math`、`factor_jitter`、`distance_cache` がこれに当たり、これらのキーが無いファイルもそのまま読み込める。読み込み側が知らないキーは無視する。

| 状況 | エラー |
| --- | --- |
| ファイルを読み書きできない | `GprError::PersistFailed { kind: Io, reason }` |
| JSON として不正、キーの不備、`point_ids` の長さの不一致 | `GprError::PersistFailed { kind: Config, reason }` |
| テンソルの欠落、形や dtype の不一致、テンソルのバイト列が 8 バイト境界に揃っていない、`q` が不正 | `GprError::PersistFailed { kind: Tensor, reason }` |
| `model` に合わないローダーで読んだ | `GprError::PersistFailed { kind: WrongModel, reason }` |
| 登録されていない `persist_id` | `GprError::PersistFailed { kind: UnregisteredId, reason }` |
| 予約された `persist_id` | `GprError::PersistFailed { kind: InvalidPersistId, reason }` |
| 同じ種類に同じ `persist_id` を 2 回登録した | `GprError::PersistFailed { kind: InvalidPersistId, reason }` |
| `persist_id` を実装していない `custom` のカーネルや変換を保存しようとした | `GprError::PersistFailed { kind: NotPersistable, reason }` |
| `format_version` が `1` でない | `GprError::UnsupportedPersistVersion` |
| `n`、`d`（`distance_only` を除く）、（Sparse の）`m` が `0`、または `lengthscales` が空 | `GprError::EmptyInput` |
| 保存された `d²` が有限でない、負である、または対角が 0 でない。Sparse のブロックで、誘導点の行が対称な正方行列になっていない | `GprError::InvalidDistance` |
| コンストラクタが受け付けない保存値（境界、ジッター、カーネルのパラメータ） | そのコンストラクタ自身のエラー |

保存ディレクトリは、信頼できる出所のものだけを読み込むこと。`sum` / `product` / `pipeline` / `columnwise` の木が JSON パーサーの上限（配列とオブジェクトの入れ子で 128 段）より深いと、デコードの前に `PersistFailed` で失敗する。読み込み時に検査するのは、テンソルの形と dtype、保存したすべてのテンソルが有限であること（`NaN` や `±∞` があれば `PersistFailed`）、そして `q` の有限性と下三角性である。

## 9. gprx なしでファイルを読む

NumPy だけを使う Python の最小限の読み込みコードを示す。ファイルの中身を確かめたいときや、モデルを別のツールへ移したいときに使える。safetensors のヘッダを自前で解析し、行列を列優先として読む。

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

`order="F"` を指定しているので、`x[i, j]` が「点 `i` の特徴 `j` の値」になる。既定の行優先のまま読むと、`d > 1` のときは同じバイト列から並びの崩れた行列ができてしまう。

## 10. 呼び出し側が二乗距離を与えるモデル

`DistanceKernel` のモデルでは、カーネルが、呼び出し側の与える二乗距離（`ScalarDistance`、`ArdDistance`）を読む。このモデルも同じ `save` / `save_with_factor` で、同じ 2 つのファイルに保存する。加わるのは、モデルが保持する学習用の `d²` と、slot の表である。

読み込んだカーネルには**新しい slot** が割り当てられ、保存前の `ScalarDistance` や `ArdDistance` は、そのどれにも対応しない。読み込んだモデルの `slots()`（保存したカーネルの `DistanceKernel::slots` の順に並ぶ）か `to_kernel()` から slot を取り出し、予測に渡す各 source をその slot に結び付ける。

| モデル | 読み込むローダー | マーカーの型 |
| --- | --- | --- |
| `DistanceKernel<C>` の `FittedGpr` / `OnlineGpr` | `LoadedDistanceGpr::<C>::load` | `C` は `DistanceOnly` か `WithPoints` |
| `DistanceKernel<C>` の `FittedSgpr` / `OnlineSgpr` | `LoadedDistanceSgpr::<C>::load` | 同上 |
| `DistanceKernel<C>` の `FittedSvgp` | `LoadedDistanceSvgp::<C>::load` | 同上 |

座標のローダー（`LoadedGpr`、`LoadedSgpr`、`LoadedSvgp`）の挙動は変わらず、距離のファイルを渡すと失敗する。距離のローダーも、座標のファイルや、マーカーの異なるファイルを渡すと失敗する。いずれの場合もエラーは `WrongModel` である。

### 10.1 `config.json` の `distance`

```json
"distance": {
  "points": "distance_only",
  "slots": [ { "kind": "scalar" }, { "kind": "ard", "dims": 2 } ]
}
```

- `points`: `"distance_only"` か `"with_points"` のどちらか。`"distance_only"` では、カーネルは与えられた距離だけを読み、`d` は `0`、`x`・`z`・`z_train` は空になる。`"with_points"` では、カーネルは `x` を読むカーネルの葉も持つ。
- `slots`: slot の表。カーネルの `DistanceKernel::slots` と同じ順（深さ優先で、最初に現れた順）に並ぶ。slot は `scalar`（点の組ごとに `d²` が 1 つ）か、`dims` を伴う `ard`（点の組ごと、次元ごとに `d²` が 1 つ）である。カーネルの木は表のすべての slot を表の順に読まなければならず、そうなっていなければ、読み込みは `PersistFailed`（`kind: Config`）で失敗する。slot を持たない距離カーネル（座標のカーネルの木を `DistanceKernel::from` で包んだもの）では表が空になり、`d2.<k>` のテンソルは 1 つも書き出さない。
- slot に載るカーネルの葉は、`{"distance": {"slot": k, "leaf": <kernel>}}` の形で書く（5.1 節）。`custom` のカーネルの葉は、5.5 節と同じく `PersistRegistry` で復元する。
- `inducing`（Sparse のみ。`distance` の中に置き、Sparse では必須）: 誘導点として使う学習点を、ブロックの列の順に並べたもの。要素は `m` 個で、どれも `n` 未満、重複は無い。読み込み時には、`n` 以上の添字があれば `IndexOutOfRange`、重複があれば `InvalidConfig`、個数が違えば `PersistFailed`（`kind: Config`）になる。`with_points` のモデルでは、`z` は `inducing` の指す `x` の行と、`z_train` は変換後の `x` の同じ行と、ビット単位で一致していなければならない。fit が作る `z` と `z_train` は、この条件を満たす。一致しなければ `PersistFailed`（`kind: Config`）になる。
- `distance_only` のカーネルは、座標を読むカーネルの葉を持てない（`constant` と `white` は持てる）。木にそうしたカーネルの葉を含むファイルは `PersistFailed`（`kind: Config`）になる。

実際の出力例を示す。3 点のデータに対して、スカラーの RBF と 2 次元の ARD RBF の和を `FittedGpr::save` で保存したものである。

```json
{
  "format_version": 1,
  "n": 3,
  "d": 0,
  "has_factor": false,
  "factor_kind": "llt",
  "kernel": {
    "sum": {
      "left":  { "distance": { "slot": 0, "leaf": { "rbf": { "lengthscale": { "value": 1.0, "lo": 0.00001, "hi": 100000.0 } } } } },
      "right": { "distance": { "slot": 1, "leaf": { "rbf_ard": { "lengthscales": [
        { "value": 1.0, "lo": 0.00001, "hi": 100000.0 },
        { "value": 2.0, "lo": 0.00001, "hi": 100000.0 }
      ] } } } }
    }
  },
  "likelihood": { "noise_variance": { "value": 0.1, "lo": 0.00001, "hi": 100000.0 } },
  "jitter": { "fixed": { "jitter": 0.0 } },
  "distance_cache": "always",
  "x_unfitted": "identity",
  "y_unfitted": "identity",
  "x_transform": "identity",
  "y_transform": "identity",
  "distance": {
    "points": "distance_only",
    "slots": [ { "kind": "scalar" }, { "kind": "ard", "dims": 2 } ]
  }
}
```

テンソルは `x`（`[3, 0]`）、`y`、`d2.0`（`[6]`）、`d2.1`（`[2, 6]`）の 4 つである。スカラーの slot だけを持つカーネルで、誘導点を `[0, 2]` として `FittedSgpr::save` すると、`"model": "sgpr"`、`"m": 2`、`"distance": {"points": "distance_only", "slots": [{"kind": "scalar"}], "inducing": [0, 2]}` が書き出され、`d2.0` の形は `[3, 2]` になる。（どちらも `save` の実際の出力で、キーは省略していない。実際のファイルは 1 行に 1 キーで、ここでは収めるために折りたたんだ。）

### 10.2 テンソル

`d2.<k>` には、slot `k` の学習用の `d²` が入る。

| モデル | 形（スカラーの slot） | 形（ARD の slot） | 並び | dtype |
| --- | --- | --- | --- | --- |
| Exact（`model` キー無し） | `[n(n+1)/2]` | `[dims, n(n+1)/2]` | 下三角を列ごとに並べる。列 `j` には行 `j..n`（対角を含む）が入り、その次に列 `j + 1` が続く。ARD の slot は次元ごとに順に並べる | `single` なら `F32`、`double` と `mixed` なら `F64` |
| Sparse（`sgpr`、`online_sgpr`、`svgp`） | `[n, m]` | `[dims, n, m]` | ブロックごとに列優先（`n` 個の学習点 × `m` 個の誘導点）。ARD の slot は次元ごとに順に並べる | `F64` |

- 書き出す値は、モデルが保持している値（`tidy` で修復した後のもの）で、モデルの点の順に並ぶ。online のモデルでは、現在保持している点を `point_ids` の順に並べる。更新のために ARD の slot を行単位の並びに組み替えた online のモデルも、書き出すときは上の正規の並びにする。
- `single` の Exact モデルは、`f32` の格納データをそのまま書き出す。`mixed` の Exact モデルは、`f32` の写しとは別に保持している呼び出し側の `f64` の値を書き出し、読み込み時にあらためて丸める。Sparse のモデルはどの精度でもブロックを `f64` で保持しており、そのまま書き出す。
- 読み込みでは、表にある各 slot `k` について `d2.<k>` だけを読む。どの slot にも対応しないテンソルは読まない。`has_factor` が `false` のときの `l` と `alpha` も読まない（座標のファイルと同じ扱い）。
- 読み込んだ値は、fit が source を検査するのと同じ基準で検査する。Exact では、有限で、非負で、対角が 0 であること。Sparse ではブロックに同じ条件を課し、さらに誘導点の行が対角 0 の対称な正方行列であることを求める。違反があれば `GprError::InvalidDistance` になる。
- Exact で `has_factor: false` のときは、保存した `d²` から、保存した `θ` で分解し直す。`has_factor: true` のときは、7 節のとおり `l` と `alpha` を読む。Sparse のモデルは、7 節のとおり、保存したブロックから分解し直す。
