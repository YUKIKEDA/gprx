[English](architecture.md) | 日本語

# gprx のアーキテクチャ

クレートの地図。どのモジュール（ドメイン）があり、それぞれが何に責任を持ち、依存がどちら向きかを示す。各部品の意味と挙動は [design.ja.md](design.ja.md)。ファイルの置き場所は [conventions.md](conventions.md)。決定の理由は [adr/](adr/)。保存フォーマットは [persist-format.ja.md](persist-format.ja.md)。

以下の import の矢印は、`src/` の `use crate::…`（`#[cfg(test)]` のコードを除く）から取り、この文書とスクリプトで突き合わせている（Issue [#310](https://github.com/YUKIKEDA/gprx/issues/310)）。モジュールの import が増減したら、この文書も一緒に直す。

## 1. 全体像

3 つのモデルが、共通の部品の集まりを使う。各モデルには、学習前（trainer）、学習後（fitted）があり、一部のモデルにはオンライン版がある。モデル同士は import しない。共有するものは、その下の層に置く。

```mermaid
flowchart TB
    api["<b>公開 API</b><br/>lib.rs の再エクスポート。pub mod は kernel, transform, persist"]
    subgraph models["モデル — モデルごとに 1 ディレクトリ"]
        direction LR
        gpr["<b>gpr</b><br/>Exact GPR"]
        sgpr["<b>sgpr</b><br/>Sparse GPR (VFE)"]
        svgp["<b>svgp</b><br/>SVGP (ミニバッチ)"]
    end
    sparse["<b>sparse</b><br/>sgpr と svgp が共有する crate 内の中核"]
    persist["<b>persist</b><br/>ディレクトリへの保存と読み込み"]
    subgraph services["モデルが組み合わせる部品"]
        direction LR
        kernel["<b>kernel</b><br/>spec, compiled, カーネルの葉"]
        likelihood["<b>likelihood</b>"]
        transform["<b>transform</b><br/>入力 / 目的変数の変換"]
        precision["<b>precision</b><br/>f32 / f64 / 混合"]
        optimizer["<b>optimizer</b><br/>+ objective の trait"]
        workspace["<b>workspace</b><br/>+ prediction"]
    end
    subgraph foundation["基盤 — スカラー、数値計算、検査"]
        direction LR
        f1["linalg · math · policy"]
        f2["param · data · error · rng · points"]
    end
    api --> models
    gpr --> services
    sgpr --> sparse --> services
    svgp --> sparse
    services --> foundation
    persist -.->|"読んで組み直す"| models
    models -.->|"save, persist_err"| persist
```

実線は、層の向きに沿った import を表す。破線は、層を両方向にまたぐ唯一の箇所で、`persist` がモデルを作り、モデルが保存のために `persist` を呼ぶ。何がまたぐかは 4 節に書く。

## 2. モジュールと責務

「公開」は `src/lib.rs` が出すもの（`pub mod` または `pub use`）を、「crate」は crate 内限定を指す。「Imports」は、テスト以外で使う、ほかのトップレベルのモジュールを示す。

### 基盤

| Module | 責務 | 公開 / 主な型 | Imports |
| --- | --- | --- | --- |
| `error` | 唯一のエラー型と、Cholesky の段の印 | 公開: `GprError`, `CholeskyStage` | `param` |
| `param` | 区間つきの正のパラメータと、平らな `θ` の書き込み補助 | 公開: `Interval`, `BoundedParam`, `IntervalError` | `data`, `error`, `kernel`, `likelihood` |
| `data` | 呼び出し側のデータの境界検査（形、有限、個数）と列優先の詰め込み | crate | `error`, `kernel` |
| `rng` | サンプリングと焼きなましのための、シードつきの小さな乱数 | crate: `SeededRng`（Xoshiro256++） | none |
| `math` | カーネルの `exp` の実装（厳密 / 高速近似）。`KernelExp` の方針で選ぶ | 公開: `Accurate`, `FastApprox`, `KernelMath` | `kernel` |
| `linalg` | Cholesky、LDLT、三角行列への代入、密行列の補助、faer のワーカー数の上限。モデルは自前で持たない | crate | `error`, `kernel` |
| `policy` | 実行時の方針: 距離キャッシュ、Cholesky のバッファ、カーネルの `exp`、ジッター | 公開: `DistanceCachePolicy`, `CholeskyBuffer`, `KernelExp`, `JitterPolicy`, `FixedJitter`, `AdaptiveJitter` | `error`, `math` |
| `points` | 追加・削除される点の安定した id | 公開: `PointId`。crate: `IdRegistry` | `error`, `persist` |

### 部品

| Module | 責務 | 公開 / 主な型 | Imports |
| --- | --- | --- | --- |
| `kernel` | カーネルの言語: 組み込みのカーネルの葉と `Custom` の木 `KernelSpec` を、静的ディスパッチの `CompiledKernel<T>` に平らにする。値、勾配、Hessian、座標微分。スカラーの trait `KernelScalar`。与えられた二乗距離を読む葉: スロット、`DistanceKernel` の木、呼び出し側の表の結び付けと検査、学習の `d²` のストア（`supply.rs`、`sources/`、`compiled/supplied.rs`） | pub mod。`KernelSpec`, `CompiledKernel`, カーネルの葉（`RbfKernel`, `MaternKernel`, `PeriodicKernel`, …）, `KernelTerm`, `CustomKernel`, `KernelScalar`。`DistanceKernel`, `DistanceOnly`, `WithPoints`, `ScalarDistance`, `ArdDistance`, `DistanceSlot`, `DistanceSource`, `DistanceFill`, `ModelKernel`, `PointKernel`, `Supply`, `NoSupply` | `data`, `error`, `linalg`, `math`, `param` |
| `likelihood` | ガウスの観測ノイズ `σn²`。ジッターとは別の、独立したパラメータ | 公開: `GaussianLikelihood` | `data`, `error`, `param` |
| `transform` | 入力の変換（identity、standardize、min-max、列ごと、pipeline）と目的変数の変換。それぞれ学習前と学習後の型を持つ。予測で平均と分散を戻す | pub mod: `Transform`, `UnfittedTransform`, `TargetTransform`, `UnfittedTarget`, `MinMaxInput`, `StandardizeTarget`, `Pipeline`, … | `data`, `error` |
| `precision` | 格納と予測のスカラーを 1 つの方針にまとめる。混合精度の反復改善 | 公開: `PrecisionPolicy`, `DoublePrecision`, `SinglePrecision`, `MixedPrecision`, `PromoteStorage`, `ReevaluateKernel` | `error`, `kernel`, `linalg`, `math`, `policy`, `transform` |
| `workspace` | 使い回すバッファ: Gram、`W`、距離キャッシュ、`exp` のバッファ、faer の scratch。クエリごとのバッファ | crate: `WorkspaceCore`, `FitBuffers`, `QueryWorkspace` | `error`, `kernel`, `linalg`, `policy`, `precision` |
| `prediction` | 予測が返すもの、共分散からの事後標本の生成、`DistanceKernel` のモデルが共有する予測メソッド | 公開: `Prediction`, `PredictiveCovariance`, `PredictOptions`, `VarianceKind`。クレート内: `DistanceQuery`, `QueryPoints`, `distance_predict!` | `error`, `kernel`, `linalg`, `policy`, `rng` |
| `objective` | モデルの学習の目的関数が実装する trait。ソルバーがモデルを知らなくて済む | 公開: `Objective`, `Differentiable`, `TwiceDifferentiable`, `IncrementalObjective` | `error`, `param` |
| `optimizer` | その trait の上のソルバー: argmin のアダプタ、自前の焼きなまし、`Fixed` の印。SVGP 用の Adam（`Optimizer` ではない） | 公開: `Optimizer`, `Lbfgs`, `NelderMead`, `TrustRegion`, `FastSimulatedAnnealing`, `Fixed`, `Adam`, `OptResult`, `BoundaryPolicy` | `error`, `objective`, `param`, `rng` |

### モデル

| Module | 責務 | 公開 / 主な型 | Imports |
| --- | --- | --- | --- |
| `gpr` | Exact GPR: `K + σn²I` の分解、fit / refit のための NLML とその微分、予測、共分散、標本、leave-one-out、LDLT の因子の上のオンライン insert / delete。与えられた二乗距離の上の同じもの（`distance.rs`） | 公開: `Gpr`, `FittedGpr`, `OnlineGpr` | `data`, `error`, `kernel`, `likelihood`, `linalg`, `math`, `objective`, `optimizer`, `param`, `persist`, `points`, `policy`, `precision`, `prediction`, `transform`, `workspace` |
| `sparse` | `sgpr` と `svgp` が共有するもの: trainer の設定、学習データ、`Z`、カーネルと尤度にまたがる `θ`、距離のモデルが持つ与えられた `n × m` のブロック | crate: `SparseSpec`, `SparseCore`, `SparseSupply` | `data`, `error`, `kernel`, `likelihood`, `linalg`, `math`, `param`, `policy`, `precision`, `prediction`, `transform` |
| `sgpr` | collapsed VFE の下限を使う Sparse GPR: 固定または自由な誘導点 `Z`、rank-1 のオンライン更新、点と誘導点の insert / delete、予測、共分散、標本、leave-one-out。与えられた二乗距離の上の同じもの（`distance.rs`、`online/distance.rs`） | 公開: `Sgpr`, `FittedSgpr`, `OnlineSgpr`, `FixedInducing`, `FreeInducing`, `InducingId` | `data`, `error`, `kernel`, `likelihood`, `linalg`, `math`, `objective`, `optimizer`, `param`, `persist`, `points`, `policy`, `precision`, `prediction`, `sparse`, `transform` |
| `svgp` | SVGP: whitened な `q(u)`、ELBO、1 ステップの計算量が `n` に依らないミニバッチ Adam、予測、共分散、標本。与えられた二乗距離の上の同じもの（`distance.rs`） | 公開: `Svgp`, `FittedSvgp` | `data`, `error`, `kernel`, `likelihood`, `linalg`, `math`, `optimizer`, `param`, `persist`, `policy`, `precision`, `prediction`, `rng`, `sparse`, `transform` |
| `persist` | モデル 1 つにつき 1 ディレクトリ: `config.json` と `model.safetensors`。`Custom` のカーネルと呼び出し側の変換の復元表 | pub mod: `LoadedGpr`, `LoadedSgpr`, `LoadedSvgp`, `LoadedDistanceGpr`, `LoadedDistanceSgpr`, `LoadedDistanceSvgp`, `PersistRegistry`, `FORMAT_VERSION` | `error`, `gpr`, `kernel`, `optimizer`, `param`, `points`, `policy`, `precision`, `sgpr`, `sparse`, `svgp`, `transform` |
| `internals` | ベンチマークと `compare/perf` のためのフック。`bench-internals` または `insert-stages` の feature のときだけ | pub mod、feature つき | `gpr`, `kernel`, `objective` |

## 3. 部品とモデルの間の依存

どのモジュールも基盤（`error`, `param`, `data`, `linalg`, `math`, `policy`, `rng`）を使い、多くは 4 つの部品 `kernel`, `likelihood`, `transform`, `precision` も使う。これらの矢印は図全体を横切るので、図からは外し、図の下の表に書いた。各モジュールのすべての import は 2 節にある。破線は、層をまたぐ呼び出しで、4 節で説明する。

```mermaid
flowchart TB
    gpr --> objective
    gpr --> prediction
    gpr --> optimizer
    gpr --> persist
    gpr --> points
    gpr --> workspace
    sgpr --> objective
    sgpr --> prediction
    sgpr --> optimizer
    sgpr --> persist
    sgpr --> points
    sgpr --> sparse
    svgp --> optimizer
    svgp --> prediction
    svgp --> persist
    svgp --> sparse
    sparse --> prediction
    persist -.-> gpr
    persist -.-> sgpr
    persist -.-> svgp
    persist -.-> sparse
    persist --> optimizer
    persist --> points
    optimizer --> objective
    points -.-> persist
```

広く使われる 4 つの部品を import するモジュール（図から外した矢印）:

| 部品 | import するモジュール |
| --- | --- |
| `kernel` | `data`, `gpr`, `internals`, `linalg`, `math`, `param`, `persist`, `precision`, `prediction`, `sgpr`, `sparse`, `svgp`, `workspace` |
| `likelihood` | `gpr`, `param`, `sgpr`, `sparse`, `svgp` |
| `transform` | `gpr`, `persist`, `precision`, `sgpr`, `sparse`, `svgp` |
| `precision` | `gpr`, `persist`, `sgpr`, `sparse`, `svgp`, `workspace` |

図から読めること:

- **`optimizer` と `objective` はモデルを知らない。** ソルバーは `Objective` / `Differentiable` / `TwiceDifferentiable` の上に書かれ、各モデルが自分のアダプタ（`GprObjective`, `SgprObjective`）でそれを実装する。利用者の `Optimizer` も同じ枠に入る。
- **`sgpr` と `svgp` は `sparse` を通じてのみ結合する。** `gpr` は `sparse` を使わない。
- **`precision` と `transform` はモデルを知らない。** モデルは、`f64` の参照値をクロージャで precision のコードに渡す。
- **`kernel` が最も幅の広い部品。** 全モデルと `workspace` が使い、`persist` がその木を符号化する。

## 4. 境界と、その例外

コードベースで維持されている境界（`#[cfg(test)]` 以外の `use crate::…`）:

1. **モデル同士は import しない。** `gpr`, `sgpr`, `svgp` の間に import は無い。共有するコードは 1 つ下の層に置く（2 つの Sparse モデルには `sparse`、3 つのモデルすべてには部品）。
2. **モデルを import するのは `persist` だけ。** `persist/mod.rs`（Exact）、`persist/sparse.rs`（Sparse、SVGP）、`persist/distance.rs`（`DistanceKernel` の読み込みの型）にある。具体的なモデルの型をすべて名指しする唯一の場所であり、そのため `LoadedGpr` / `LoadedSgpr` / `LoadedSvgp` と `LoadedDistanceGpr` / `LoadedDistanceSgpr` / `LoadedDistanceSvgp` が精度ごとに 1 つの variant を持てる。
3. **逆向きにまたぐものは少ない。** モデルは `persist::save_*` を呼び、`gpr` はさらに `PersistedModel`（読み込んだ Exact モデルを組み直す部品）と `MappedTensors`（メモリマップした `L`）を使う。`gpr`、`sgpr`、`points` は、エラーを作るのに `persist_err` を使う。`persist` のそれ以外を、モデルは使わない。
4. **`optimizer`、`objective`、`precision`、`transform` はモデルを import しない。** `optimizer` の単体テストは `Gpr` を作るが、テストのコードだけ。
5. **`Workspace`、`QueryWorkspace`、`LltStore`、`LdltStore`、faer の型は crate 内だけ**（[layout の規則](../.cursor/rules/layout.mdc)）。

きれいな層になっていない例外と、その維持理由:

- **基盤は互いを輪のように参照している。** `kernel/scalar.rs` が `f32` / `f64` のスカラー trait `KernelScalar` を定義し、`data`、`math`、`linalg` はそれについてジェネリックで、`kernel` はその 3 つを使う。`param` は `KernelSpec` と `GaussianLikelihood` の平らな `θ` を書くので両方を import し、`likelihood` は範囲のために `param` を import し返す。`error` は `param` の `IntervalError` を包む。これらは型と補助関数の参照で、実行時の呼び出しの循環ではない。
- **`persist` とモデルは互いを参照している**（上の 2 と 3）。

## 5. モデルごとの公開型

3 つのモデルは同じ typestate に従う。trainer、`fit`（または `factor`）、fitted の各状態を取り、fitted の値は最適化器の状態も `W` も持たない。学習と推論は別の型（[design §6](design.ja.md#6-gpmodel抽象化厳密疎の差し替え)）。

| モデル | Trainer | Fitted | Online | ディスクから読んだもの |
| --- | --- | --- | --- | --- |
| Exact | `Gpr<O, P, K>` | `FittedGpr<O, P, K>` | `OnlineGpr<O, P, K>`（`insert`, `delete`） | `LoadedGpr`（8 variant）。`DistanceKernel<C>` なら `LoadedDistanceGpr<C>`（8） |
| Sparse (VFE) | `Sgpr<O, I, P, K>` | `FittedSgpr<O, I, P, K>` | `OnlineSgpr<O, P, K>`（`insert`, `delete`, `insert_inducing`, `delete_inducing`） | `LoadedSgpr`（8 variant）。`LoadedDistanceSgpr<C>`（8） |
| SVGP | `Svgp<O, P, K>` | `FittedSvgp<P, K>` | なし | `LoadedSvgp`（4 variant）。`LoadedDistanceSvgp<C>`（4） |

型パラメータ:

| パラメータ | 意味 | 値 |
| --- | --- | --- |
| `O` | 最適化器の枠 | `Lbfgs`（Exact と Sparse の既定）, `NelderMead`, `TrustRegion`, `FastSimulatedAnnealing`, 利用者の `Optimizer`。`factor` だけなら `Fixed`。`Svgp::fit` は `Adam`（`Svgp` の既定は `Fixed`） |
| `P` | 精度。コンパイル時の選択 | `DoublePrecision`（既定）, `SinglePrecision`, `MixedPrecision`（残差は `PromoteStorage` か `ReevaluateKernel`） |
| `I` | 誘導点 `Z` の置き場 | `FixedInducing`（既定。`Z` はパラメータに入らない）, `FreeInducing`（`Z` を `θ` と一緒に最適化する。座標のカーネルだけ） |
| `K` | どのモデルでも、そのカーネルが読むもの（[design §5.1](design.ja.md#51-仕様と評価器精度ジェネリクス)） | `KernelSpec`（既定。座標）, `DistanceKernel<DistanceOnly>`（与えられた二乗距離だけ）, `DistanceKernel<WithPoints>`（与えられた二乗距離と座標） |

## 6. モデルの状態の移り方

```mermaid
flowchart LR
    G["Gpr&lt;O&gt;"] -- "fit / factor" --> F["FittedGpr"]
    F -- "into_online" --> O["OnlineGpr"]
    F -- "into_trainer" --> G
    O -- "insert / delete" --> O
    S["Sgpr&lt;O, I&gt;"] -- "fit / factor" --> FS["FittedSgpr"]
    FS -- "into_online" --> OS["OnlineSgpr"]
    OS -- "into_fitted" --> FS
    V["Svgp&lt;O&gt;"] -- "fit / factor" --> FV["FittedSvgp"]
    F -- "save" --> D[("ディレクトリ<br/>config.json +<br/>model.safetensors")]
    O -- "save" --> D
    FS -- "save" --> D
    OS -- "save" --> D
    FV -- "save" --> D
    D -- "LoadedGpr::load" --> LG["LoadedGpr"]
    D -- "LoadedSgpr::load" --> LS["LoadedSgpr"]
    D -- "LoadedSvgp::load" --> LV["LoadedSvgp"]
```

`DistanceKernel<C>` のモデルも同じように動き、`LoadedDistanceGpr<C>` / `LoadedDistanceSgpr<C>` / `LoadedDistanceSvgp<C>` で読む（[persist-format.ja.md 10 節](persist-format.ja.md#10-与えられた二乗距離のモデル)）。

読み込んだモデルは予測できる状態で、`Fixed` を持つので、探索は保存しない。もう一度学習するには、型のついたモデルで `with_optimizer` を呼んでから `refit` する。各 `save` が書くものは [persist-format.ja.md](persist-format.ja.md)。

1 回の呼び出しにおける処理順序は固定されている: 入力の変換 → 目的変数の変換 → `θ` でのカーネルと尤度 → 分解 → `α` → 予測。学習済みのモデルは、渡された `X` と `y`（変換前）を、学習済みの変換と一緒に持ち、クエリのたびにその変換をかけ直す（[design §2](design.ja.md#2-全体アーキテクチャ概要)、[§5.5](design.ja.md#55-前処理パイプライン)）。

## 7. 何を変えるとき、どこを見るか

| 変えたいもの | 見る場所 | 一緒に変更するもの |
| --- | --- | --- |
| カーネルの葉 | `kernel/<leaf>.rs` と `kernel/compiled/` | `kernel/spec.rs`、`persist/kernel.rs`（新しい JSON のタグ）、design §5。下の手順 |
| 全モデルの最適化器 | `optimizer/` | モデルは変更しない。新しい能力の trait が要るときだけ `objective.rs` |
| Exact だけがすること（オンライン LDLT、`Gpr` の LOO） | `gpr/` | 因子は `linalg/ldlt.rs` |
| 2 つの Sparse モデルが共通にすること | `sparse/` | `sgpr/` と `svgp/` が呼ぶ |
| 精度の規則 | `precision/` | クロージャを渡すモデルの `factor/` |
| 保存の配置 | `persist/` | [persist-format.ja.md](persist-format.ja.md)。古いファイルを読み誤りうるなら `FORMAT_VERSION` |
| 新しいモデル | `gpr/` の隣の新しいディレクトリ | `persist/` に `Loaded*` を 1 つ。ほかのモデルを import してはいけない |

### 組み込みのカーネルの葉を足す

カーネルの葉は静的にディスパッチする。`KernelSpec` と `CompiledKernel` の各操作はカーネルの葉ごとに 1 つの分岐を持つ `match` で、全部で約 40 ある。そのため呼び出しはコンパイラがインライン化・ベクトル化できる直接の呼び出しになる（design §5）。代わりに、新しいカーネルの葉を追加する際はそのすべてを変更する必要がある。答えがカーネルの葉によって変わる `match` はすべてのカーネルの葉を名指しし、ワイルドカードを持たないので、分岐が足りない場所はコンパイラが列挙する。残るワイルドカードは、どのカーネルの葉にも正しい既定（速い経路が無いときに座標から計算する）か、カーネルの葉と合成の区別だけ。カーネルの葉を足す手順:

1. `kernel/<leaf>.rs`: パラメータ（`θ` とその `Interval`）、距離または座標からの値・`∂K/∂θ`・`∂²K/∂θ∂θ`（正方と長方形）、対角。`FreeInducing` で動かすなら座標微分（`grad_wrt_coord_dim` と混合の Hessian）。動かさないなら `CoordGradientUnsupported` を返す
2. `kernel/spec.rs`: `KernelSpec` の variant、`From`、コンパイラが求める分岐
3. `kernel/compiled/`: `CompiledKernel` の variant、それに対応する `LeafRef` の variant（`term`）、コンパイラが求める分岐。すべてのカーネルの葉を名指しする `coord_mode`、`needs_ard_sq_diff`、`needs_grad_scratch` を含む。座標の経路の葉の分岐は `LeafRef` のメソッドで、座標の木と葉ごとの混合経路が共有する
4. `persist/kernel.rs`: JSON のタグ。古い版の保存ファイルも読めること（persist-format.md）
5. 与えられた二乗距離も読める葉なら:
   - `kernel/supply.rs`: `ScalarLeafSpec` か `ArdLeafSpec` の variant、`scalar_leaf!` / `ard_leaf!` の一覧、`with_distance_leaves!` の 1 組。保存と読み込みが使う `KernelSpec` の葉との対応は、その 1 組から生成される
   - `kernel/compiled/supplied.rs`: `ScalarLeaf` か `ArdLeaf` の variant、compile の分岐、`each_scalar_leaf!` か `each_ard_leaf!` の 1 行。組み込みの葉はメソッド名（`apply_math`、`grad_from_sq_diff` など）をそろえているので、1 つの本体ですべての葉に対応できる
   - `DISTANCE_LEAVES`（`kernel/compiled/leaf_table.rs`）への名前の追加
6. `kernel/compiled/leaf_table.rs`: `leaf_index` の番号（コンパイラが求める）と表の実例。表のテストが、パラメータ、座標と距離からの Gram、相互のブロック、対角、`∂K/∂θ` と `∂²K/∂θ∂θ` の中心差分、座標微分、保存と読み込みを通す。距離の葉は、与えられた二乗距離からの Gram と `∂K/∂θ` も通す
7. design §5 と、`kernel/mod.rs`・`lib.rs` の公開の再エクスポート
