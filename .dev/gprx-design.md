# gprx 設計ドキュメント

## 1. 目的・スコープ

最も柔軟かつ最も高速なGaussian Process Regressionライブラリを、Rustで構築する。「柔軟」はユーザー定義カーネル・前処理・厳密/疎推論・最適化器の差し替え可能性、および**データ点の逐次追加削除(オンライン学習)**を指し、「高速」はアロケーション最小化・SIMD/マルチスレッド活用・精度切り替えによる計算量/メモリ最適化を指す。

**改訂履歴**: ChatGPT・Geminiによる設計レビューを受け、数理的な誤り(混合精度の残差式、jitterとノイズ分散の混同)および実装上の不整合(アロケーション方針違反、精度ジェネリクスの欠落)を修正。妥当と判断した指摘はP0(致命的)→P2(改善)の優先度で全て反映し、実装順序は§13のロードマップに従う。

## 2. 全体アーキテクチャ概要

```
入力 X, y
  → Transform Pipeline (前処理: MinMax, Standardize等)
  → Likelihood (観測ノイズσn²、モデルパラメータとして独立管理)
  → CompiledKernel<T> (KernelSpecをコンパイルした実行計画 + Workspace)
  → Inference (ExactGP / SparseGP など、GPModel trait経由で差し替え)
       → Objective (尤度・勾配、Optimizerへ提供、アロケーションフリー)
       → Optimizer (L-BFGS / Nelder-Mead 等、勾配要否で分岐)
       → OnlineInference (ExactGPのみ: データ点の増分追加削除)
  → 予測 (mean, variance)
```

主要な設計原則:
- **静的ディスパッチを基本に、拡張点のみ`dyn`を許容**
- **gprx内部のホットパスでは新規アロケーションを行わない**(「fit中アロケーションゼロ」はユーザー定義カーネル実装まで強制できないため、この表現に修正)
- **精度はコンパイル時ジェネリクスで固定**
- **数値安定化(jitter)とモデルパラメータ(観測ノイズ)を明確に分離する**

## 3. 線形代数バックエンド: faer

Pure Rustで、OpenBLAS/LAPACK/Eigenと同等以上の性能を達成しており、RayonベースでOpenMP/TBB相当の並列化性能を持つ。

- `Mat<T>`は列優先(column-major)、行ストライドは常に1
- Cholesky分解は`llt::factor::cholesky_in_place(a: MatMut<T>, regularization, par, stack, params)`でin-place
- 動的正則化(jitter)は`LltRegularization`としてAPI組み込み済み。**ただしこれは純粋な数値安定化用であり、GPRの観測ノイズ(モデルパラメータ)とは別物として扱う**(§4.0)
- `Mat`は`reserve_exact`による容量ベース確保をサポート(§11のオンライン学習で活用)
- `llt::update::{insert,delete}_rows_and_cols_clobber`: データ点の増分追加削除用(§11)。**実装前に小規模行列(例: 2x2)でフルCholeskyとの一致を検証するテストを書くこと**(§12)
- `ldlt_diagonal::update::rank_r_update_clobber`: ランクr更新、低ランクΔKの場合のみ利用可(§5.4.1)

## 4. 精度ポリシーとノイズ/Jitterの分離

### 4.0 観測ノイズとJitterの分離(P0修正)

レビュー指摘により、当初の設計は「観測ノイズσn²」(GPRのモデルパラメータ、最適化対象)と「Jitter」(Choleskyを正定値に保つための数値安定化オフセット)を`LltRegularization`に混同していた。これを分離する。

```rust
/// モデルの尤度。観測ノイズはここで管理し、最適化対象として扱う
trait Likelihood<T: Scalar>: Send + Sync {
    fn add_noise_diag(&self, k_diag: &mut [T]);      // K += σn²・I (対角への加算)
    fn noise_params(&self) -> &[T];
    fn noise_grad_diag(&self, dK_diag: &mut [T], param_idx: usize); // ∂K/∂σn² = 2σn・I
}

struct GaussianLikelihood<T: Scalar> { log_noise_variance: T } // 正値制約はlogパラメータ化で担保

/// 純粋な数値安定化。モデルには影響しない
struct NumericalStability {
    jitter: f64,      // cholesky_in_place呼び出し時のLltRegularizationにのみ使う
    max_jitter: f64,  // これを超えて増やしてもCholeskyが成立しなければCholeskyFailedを返す
}
```

`A = K + Likelihood.noise_diag`が**実際に解きたい線形システムの行列**(GPRのモデル)であり、`jitter`は`cholesky_in_place`内部でのみ一時的に加わる分解用の摂動として扱う。反復改良の残差計算(§4.1)は`A`に対して行い、jitterは含めない。

### 4.1 精度ポリシー: f32/f64/混合精度

目的は「メモリ削減」と「計算速度」の両方。混合精度反復改良(mixed-precision iterative refinement)を採用するが、**適用範囲をfit時とpredict時で分ける**(P0修正、Gemini指摘)。

```rust
trait PrecisionPolicy {
    type Storage: Scalar;
    type Refine: Scalar;
}
struct MixedPrecision;  // Storage=f32, Refine=f64
struct SinglePrecision; // Storage=f32, Refine=f32
struct DoublePrecision; // Storage=f64, Refine=f64
```

**適用範囲の制限**: 周辺対数尤度(MLL)の`log|K| = 2Σlog(L_ii)`および勾配のトレース項`Tr(K⁻¹∂K/∂θ)`は、`α=K⁻¹y`の反復改良では高精度化されない(f32のLの対角値そのものに依存するため)。これらの項を含むfit時(ハイパーパラメータ最適化ループ)のデフォルトは**`DoublePrecision`**とする。`MixedPrecision`は`α`の線形ソルブのみで完結するpredict時(ハイパーパラメータ固定後の推論)を主対象とする。fit時にMixedPrecisionを使う場合は、log|K|・トレース項の精度検証を別途行うことを前提とする(§14未解決事項)。

手順(predict時、または固定カーネルでのソルブ):
1. `A = K + Likelihood.noise_diag`をf32のまま`cholesky_in_place::<f32>`で分解(内部でjitterによる正則化のみ適用)
2. f32の`L`で`alpha_0 = solve(L, y)`
3. **残差`r = y - A_f64 @ alpha_0`をf64精度でO(n²)計算**(`A`はjitterを含まない真のモデル行列。jitterは分解時の内部的な摂動に留め、反復改良の目標には含めない)
4. f32の`L`で`delta = solve(L, r)`、`alpha_1 = alpha_0 + delta`
5. 収束するまで数回繰り返す

実装優先度: `DoublePrecision`をデフォルトとし、`MixedPrecision`はオプション機能として後付け(§13 Phase 5)。

### 4.2 混合精度反復改良の収束判定パラメータ

古典的な反復改良理論(Higham)より、分解精度u_f(f32≈1.19×10⁻⁷)と改良精度u_r(f64≈2.22×10⁻¹⁶)を使う場合、収束速度はκ(A)·u_fに依存する。**ただし実際の収束判定は理論値ではなく実測残差で行う**(P0修正、ChatGPT指摘: 理論条件だけでは分解誤差・対称性・正定値性など多くの要因を捉えきれない)。

```rust
struct RefinementConfig {
    max_iterations: usize,   // デフォルト10
    relative_tolerance: f64, // デフォルト: 10.0 × n × u_r。判定は実測残差ノルムで行う
    stagnation_ratio: f64,   // デフォルト0.9
    fallback: RefinementFallback,
}

enum RefinementFallback {
    IncreaseNumericalJitter { max_retries: usize }, // §4.0のjitterのみ変更、noise_varianceは不変
    FallbackToDoublePrecision,
    ReturnError,
}
```

収束判定: `||r_k||∞ / (||A||∞ ||alpha_k||∞ + ||y||∞) < relative_tolerance`。`stagnation_ratio`超過が2回連続で発生したら`RefinementNotConverged`(§10)。

**jitterを増やす際の注意(P0修正、ChatGPT/Gemini共通指摘)**: リトライ時に増やすのは§4.0の`NumericalStability.jitter`のみであり、`Likelihood.noise_variance`(モデルパラメータ)には触れない。jitterを増やすことは「別のGPモデルを解く」ことを意味しないよう、数値安定化とモデルを厳密に分離する。

**位置づけ**: 理論的妥当性はあるが、実ワークロードでのパラメータ検証は今後の課題(§14)。

## 5. カーネル設計

### 5.1 Spec(宣言層)/ Evaluator(実行層)の分離、および精度ジェネリクス

`KernelSpec`(宣言層)は精度に依存しない型消去された表現とし、パラメータは常に`f64`で保持する(ユーザーが書く・読む値は精度非依存であるべきため)。`CompiledKernel<T>`(実行層)は`PrecisionPolicy::Storage`ごとにコンパイルされ、内部計算は`T`で行う(P1修正、ChatGPT指摘: KernelTermが`MatRef<f64>`固定でPrecisionPolicyと矛盾していた)。

```rust
enum KernelSpec {
    Leaf(Box<dyn KernelTermSpec>),      // 宣言層: paramsはf64固定
    Sum(Box<KernelSpec>, Box<KernelSpec>),
    Product(Box<KernelSpec>, Box<KernelSpec>),
}
trait KernelTermSpec: Send + Sync {
    fn params(&self) -> &[f64];
    fn compile<T: Scalar>(&self) -> Box<dyn KernelTerm<T>>; // 実行層への変換
}

trait KernelTerm<T: Scalar>: Send + Sync {
    fn distance_kind(&self) -> DistanceKind;
    fn apply(&self, dist: MatRef<T>, out: MatMut<T>);
    fn grad(&self, dist: MatRef<T>, dK: MatMut<T>, param_idx: usize);
    fn rank_structure(&self) -> KRankStructure { KRankStructure::Dense } // §5.4.1
    fn grad_wrt_coords(&self, x1: MatRef<T>, x2: MatRef<T>, dK: MatMut<T>, coord_idx: (usize, usize)) -> Result<(), GpError> {
        Err(GpError::CoordGradientUnsupported)
    }
}
```

`KernelSpec`は演算子オーバーロードでユーザーが自然に合成でき、`KernelTermSpec`はobject-safeなのでユーザー定義カーネルはこれを実装するだけで組み込める。`CompiledKernel<T>`への変換をfit開始時に一度だけ行う。

### 5.2 距離キャッシュとキャッシュポリシー

等方カーネルは生の座標差がfit中不変のため、距離テンソルは1回計算して使い回す。

```rust
enum DistanceKind { SqEuclidean, SqEuclideanARD, Periodic { period: usize } }
```

**キャッシュ方針はベンチマークベースのポリシーとして抽象化する**(P2修正、ChatGPT指摘: d/nだけの式はカーネル種別・SIMD効率・メモリ帯域などを考慮できておらず、決定基準としては不十分)。

```rust
enum DistanceCachePolicy {
    Never,
    Always,
    Auto { memory_budget_bytes: usize }, // n,d,メモリ予算から実装時にベンチマークして調整
}
```

理論的な参考値(目安であり決定基準ではない): `(n,n,d)`テンソルは`n²×d×sizeof(T)`バイト。基本のK行列自体もn²×sizeof(T)であり(例: n=5000,f64で約200MB)、ARDキャッシュはこれのd倍になる点に注意。d≪nの典型的GPRではキャッシュの投資対効果は薄いことが多い。`Auto`の具体的な閾値は実装後のベンチマークで決定する(§14)。

### 5.3 CompiledKernelのplan構築アルゴリズム

Sum/Productは結合則・交換則が効くため、flatten+fold評価で済む。

1. 距離キャッシュ重複排除: 合成木を走査し`DistanceKind`集合を構築
2. flatten: `(A+B)+C`を`Sum(vec![A,B,C])`に正規化
3. plan生成: `BufAllocator`(フリーリスト)で`alloc()`/`free()`を追跡し、**実際のplanから動的に最大同時使用数を計算**してWorkspaceの確保サイズを決める

**訂正(P0)**: 「必要バッファ数はネストの深さでしか増えず、実用上3を超えない」という主張は誤り。`(A*B)*(C*D)`のような合成では兄弟項間でバッファを使い回せず、必要数が増える。固定上限を仮定せず、`WorkspacePlan { max_buffers, max_bytes }`をplan構築時に実測することとする。

```rust
enum PlanOp {
    EvalLeafInto { term_id: usize, dist_id: usize, dst: BufId },
    AddLeafInto  { term_id: usize, dist_id: usize, dst: BufId },
    MulLeafInto  { term_id: usize, dist_id: usize, dst: BufId },
    AddBufInto   { src: BufId, dst: BufId },
    MulBufInto   { src: BufId, dst: BufId },
}
struct WorkspacePlan { max_buffers: usize, max_bytes: usize }
```

### 5.4 部分更新(コーディネート型最適化器)対応

**対応方針**: `RecomputeStrategy`として2種類。

```rust
trait RecomputeStrategy {}
struct FullRecompute;  // バッファ最小、常にフル再計算。既定
struct IncrementalRecompute {
    leaf_contrib: Vec<Buf>,
    param_to_leaf: Vec<LeafId>,
}
```

`IncrementalRecompute`は変更indexに対応するリーフ項のみ再評価し、最終結合(O(n²×リーフ項数)、Choleskyに対して無視できるコスト)だけ毎回やり直す。**Cholesky分解自体はKが変わる以上フルで行う必要があり、部分更新の恩恵はカーネル行列構築コストにのみ及ぶ**。デフォルトは`FullRecompute`、`IncrementalRecompute`はオプトイン。

#### 5.4.1 IncrementalRecomputeとfaer update APIの関係

`llt::update::{insert,delete}_rows_and_cols_clobber`はデータ点の追加削除用(§11)、ハイパラ変更には使えない。一方`rank_r_update_clobber`は、ハイパラ変更が`K`にもたらす差分`ΔK`が低ランクな場合(線形カーネル項のamplitude変更、全体スケール変更など)に限り使える。

```rust
enum KRankStructure { Scalar, LowRank(usize), Dense }
```

デフォルト`Dense`ならユーザー定義カーネルは安全側に倒れる。

### 5.5 前処理パイプライン

```rust
trait Transform {
    fn fit(&mut self, x: MatRef<f64>);
    fn apply(&self, x: MatMut<f64>);
}
struct Pipeline(Vec<Box<dyn Transform>>);
```

## 6. GPModel抽象化(厳密/疎の差し替え)

**`Inference`から`objective()`を切り離す**(P1修正、ChatGPT指摘: 推論モデル・ハイパラ最適化・Workspace・カーネル・Optimizerが強く結合しやすくなるため)。

```rust
trait Inference<T: Scalar> {
    fn fit(&mut self, x: MatRef<T>, y: &[T]) -> Result<(), GpError>;
    fn predict(&self, xs: MatRef<T>) -> Result<Prediction<T>, GpError>;
}

struct Prediction<T: Scalar> {
    mean: Vec<T>,
    variance: Vec<T>, // 初期実装は対角分散のみ。フル共分散は将来拡張(§13 Phase 4以降)
}
```

`ExactGP`(n≲1万)と`SparseGP`(FITC/VFE)がこれを実装。ハイパラ最適化は`Objective`(§9)を介して別途扱う。

### 6.1 Sparse GPの誘導点キャッシュ問題

`K(X,X)`対角は不変なので1回計算・流用。`K(X,Z)`, `K(Z,Z)`はZが動くたびに再計算が必要だが、m(誘導点数)が小さいためCholeskyのO(nm²)に対して無視できるコストであり、キャッシュ対象にせず毎回再計算する。

誘導点座標の勾配は`grad_wrt_coords`(§5.1)で扱い、未対応カーネルはpanicではなく`GpError::CoordGradientUnsupported`を返す。

**初期実装ではフル共分散を扱わず、対角予測分散のみを目標にする**(P1修正、ChatGPT指摘)。フル共分散・Diagonal/Full切り替えは将来拡張として`Prediction`構造体を拡張する形で対応する。

Sparse GPのオンライン学習は誘導点ZとデータXの非対称性のためスコープ外(§14)。

## 7. Workspaceとメモリ管理

### 7.1 個別バッファ構造(P1修正)

当初は単一`Vec<T>`をオフセットでスライスする設計だったが、**同一Vecから複数の可変参照を同時に取り出す操作は煩雑になりやすい**(`split_at_mut`で安全に実現可能だが、Plan実行順序に応じて動的に分割点が決まるため静的なチェーンでは扱いにくい)。バッファ数は少数・固定なので、個別フィールドとして持つ設計に変更する。この際、**精度ポリシーのStorage/Refineを明示的に反映する**(P1修正、Gemini/ChatGPT指摘)。

```rust
struct Workspace<P: PrecisionPolicy> {
    k_matrix: Mat<P::Storage>,
    dist_cache: Mat<P::Storage>,
    exp_buf: Mat<P::Storage>,
    refine_buf: Option<Mat<P::Refine>>, // MixedPrecision時のみ使用、DoublePrecisionではNone
    faer_scratch: MemBuffer,            // faer公式のスクラッチ機構をそのまま使う
    thread_scratch: Vec<Mat<P::Storage>>, // Rayonスレッド数ぶん事前分割
}
```

各バッファは`fit`開始時にサイズが確定するため、`reserve_exact`で一度だけ確保(または`Mat::zeros`で1回構築)し、以降のイテレーションでは同じ領域に上書きする。あわせて、faer公式の`PodStack`/`MemStack`をスクラッチ管理に採用し、自前でスクラッチ領域をアリーナに内包する設計はやめる。

Rayon並列クロージャ内での新規確保は厳禁。`thread_scratch`を事前分割し、`rayon::broadcast`かインデックスベースで割り当てる。

### 7.2 メモリレイアウト

faerの`Mat`は列優先・行ストライド1。

- 距離行列・カーネル行列の走査は列優先、対称性を利用し上三角/下三角のみ計算
- 入力`X(n×d)`は1データ点=1列=メモリ連続(`d×n`の列優先)で保持

### 7.3 イテレーション中のライフサイクル

```
fit()開始 → n,d確定 → 各Mat<T>を1回だけ確保 → 距離キャッシュ計算(1回)
  → 最適化ループ: k_matrixに上書き構築 → in-place Cholesky(同一領域再利用) → value/grad
fit()終了 → Workspaceは保持、predict/refitで再利用
```

## 8. 並列化・SIMD、数学関数バックエンド

- カーネル評価内側ループは`std::simd`かwideクレートでベクトル化
- 距離行列・カーネル行列構築はRayonでブロック並列化
- faer自身もRayon並列化されるため、外側との二重並列化に注意。単一の`rayon::ThreadPool`を共有

**MathBackendは最小限のAPIから始め、デフォルトは近似ではなく正確な実装にする**(P1修正、ChatGPT/Gemini共通指摘: カーネル行列の近似誤差は正定値性・Cholesky安定性・尤度・勾配・予測値すべてに波及するため)。

```rust
trait MathBackend<T: Scalar>: Send + Sync {
    fn exp_inplace(&self, buf: &mut [T]); // 最初はexpのみ。erfは実際に必要になったカーネル(probit尤度等)が出てから追加
}
enum MathMode { Accurate, FastApprox }
```

デフォルトは`Accurate`(`StdExp`または`SleefBackend`)。`FastApprox`(`PolyApproxExp`)は明示的なfeatureや設定でオプトインし、**fit(ハイパラ最適化)では使わず、ハイパラ固定後の推論や大量predictに限定するのが安全**という位置づけにする。

## 9. Optimizer設計

**アロケーションフリー化とResultラップ**(P0/P1修正、Gemini/ChatGPT共通指摘)。

```rust
trait Objective<T: Scalar> {
    fn num_params(&self) -> usize;
    fn value(&mut self, params: &[T]) -> Result<T, GpError>;
    /// 勾配をoutに書き込む。勾配計算非対応ならErr(GpError::UnsupportedKernelOperation)
    fn gradient_into(&mut self, params: &[T], out: &mut [T]) -> Result<(), GpError>;
    /// 実際に内部計算(Cholesky, exp_buf等)を共有する形で実装すること
    fn value_and_gradient_into(&mut self, params: &[T], out: &mut [T]) -> Result<T, GpError> {
        let v = self.value(params)?;
        self.gradient_into(params, out)?;
        Ok(v)
    }
}
trait Optimizer<T: Scalar> {
    fn minimize(&self, objective: &mut dyn Objective<T>, init: Vec<T>) -> Result<OptResult<T>, GpError>;
    fn requires_gradient(&self) -> bool;
}
```

`ExactGP`実装では`value_and_gradient_into`をオーバーライドし、Cholesky分解(`L`, `alpha`)と`exp_buf`を尤度・勾配間で実際に共有する(デフォルト実装のように`value`→`gradient_into`を別々に呼ぶだけでは共有されないため、明示的にオーバーライドが必須である点をコメントで明記する)。座標降下法的な最適化器を使う場合は§5.4の`ChangeSet`を`value_at`のようなAPIに伝播させ、`IncrementalRecompute`と接続する。

## 10. エラー型 GpError

数値計算固有の失敗理由を拡充する(P2修正、ChatGPT指摘)。

```rust
#[derive(Debug, thiserror::Error)]
pub enum GpError {
    #[error("入力次元が一致しません: X.ncols()={x_dim}, 期待値={expected_dim}")]
    DimensionMismatch { x_dim: usize, expected_dim: usize },
    #[error("データ点数が不足しています: n={n}, 最低{min}点必要です")]
    InsufficientData { n: usize, min: usize },
    #[error("入力が空です")]
    EmptyInput,
    #[error("入力に非有限値(NaN/Inf)が含まれます")]
    NonFiniteInput,
    #[error("カーネル評価結果に非有限値が含まれます")]
    NonFiniteKernelValue,
    #[error("Cholesky分解に失敗しました(段階={stage:?}, サイズ={matrix_size}, jitter={jitter}を適用済み)")]
    CholeskyFailed { jitter: f64, matrix_size: usize, stage: CholeskyStage },
    #[error("行列が半正定値ではありません")]
    NonPositiveDefiniteMatrix,
    #[error("混合精度反復改良が収束しませんでした({iterations}回反復後、残差ノルム={residual_norm})")]
    RefinementNotConverged { iterations: usize, residual_norm: f64 },
    #[error("このカーネル項はSparse GP用の座標微分(grad_wrt_coords)を実装していません")]
    CoordGradientUnsupported,
    #[error("最適化が収束しませんでした({iterations}回反復後)")]
    OptimizationNotConverged { iterations: usize },
    #[error("ハイパーパラメータが不正です: {reason}")]
    InvalidHyperparameter { reason: String },
    #[error("観測ノイズ分散が不正です: {reason}")]
    InvalidNoiseVariance { reason: String },
    #[error("未対応のカーネル操作です: {reason}")]
    UnsupportedKernelOperation { reason: String },
    #[error("Workspaceの容量が不足しています")]
    WorkspaceTooSmall,
    #[error("指定されたPointIdは存在しません")]
    InvalidPointId,
}

#[derive(Debug)]
pub enum CholeskyStage { Fit, Predict, OnlineInsert, OnlineDelete }
```

**Error/panicの線引き**: ユーザー入力起因(`DimensionMismatch`等)、モデル/データ起因(`CholeskyFailed`等)は`Result`で返し回復可能にする。`CoordGradientUnsupported`はライブラリ内部panic対象ではないため`unimplemented!()`ではなく本Errorを返す。

## 11. オンライン学習(データ点の追加削除)

GPRはn増加に伴いO(n³)でコストが増大するため、データの逐次追加削除を正式にスコープへ含める。バッチfit用のアリーナ方式とは別に、ExactGP向けに専用のWorkspace・更新経路を用意する。

### コスト比較

| 操作 | フル再fit | 増分更新 |
|---|---|---|
| 1点追加 | O(n³) | O(n²) |
| 1点削除 | O(n³) | O(n²) |

### 増分追加の数学的根拠(P0追加、ChatGPT指摘により明記)

新しい点を追加した行列は`K_new = [[K, k], [k^T, k_new]]`。既存のCholesky因子`L`に対し`L_new = [[L, 0], [v^T, d]]`とすると、`L_new L_new^T = K_new`を満たすには:

- `L v = k` (前進消去でvを求める)
- `d = √(k_new - v^T v)`

faerの`insert_rows_and_cols_clobber`がこの関係を内部で実装している前提だが、**実装時に小規模行列(例: 2×2)でフルCholeskyとの一致を検証するテストを書くこと**(§12)。API仕様(要求する行列形式、削除が任意インデックスで動作するか、更新後Lの正しさ)は使用前に必ず確認する。

### Workspaceの容量方式

```rust
struct OnlineWorkspace<T: Scalar> {
    k_matrix: Mat<T>,
    dist_cache: Mat<T>,
    l_factor: Mat<T>,
    alpha: Col<T>,
    v_buf: Col<T>,      // 予測分散計算用の前進消去スクラッチ(テスト点1点あたりO(n²)、P1追加、Gemini指摘)
    n_active: usize,
    n_capacity: usize,
    growth_factor: f64, // デフォルト1.5〜2.0、Vec同様の償却成長
}
```

**predict時の分散計算コストの見落とし修正**(P1、Gemini指摘): 予測平均はO(n)だが、予測分散`σ*² = k(x*,x*) - v^Tv (Lv=k*)`はテスト点1点あたりO(n²)の前進消去が必要。`OnlineWorkspace`に`v_buf`をあらかじめ確保しておく。

### 増分更新の手順と不変条件

**追加**: ①新規点と既存n点との距離計算(O(n)) → ②カーネル評価しK行列に新規行/列追加 → ③`insert_rows_and_cols_clobber`でL更新(O(n²)) → ④alpha再ソルブ(O(n²))

**削除**: ①`delete_rows_and_cols_clobber`でL更新(O(n²)) → ②距離キャッシュ・K・y・alphaから該当要素を除去(O(n)) → ③alpha再ソルブ(O(n²))

**不変条件(P0追加、ChatGPT指摘)**: 削除により内部インデックスがシフトする際、`K`, `L`, `y`, `alpha`, 距離キャッシュ, `PointRegistry`は**必ず同じ順序で同期**しなければならない。いずれか一つでも順序がずれると誤った解になる。この不変条件をテスト(§12)で明示的に検証する。

```rust
struct PointRegistry {
    id_to_index: HashMap<PointId, usize>,
    index_to_id: Vec<PointId>,
}
```

### API

**insert/deleteとハイパラ再最適化を分離する**(P1修正、ChatGPT指摘: 「現在のハイパラで更新するだけ」なのか「再最適化も含む」のかを明確にするため)。

```rust
trait OnlineInference<T: Scalar> {
    fn insert(&mut self, x_new: &[T], y_new: T) -> Result<PointId, GpError>;
    fn delete(&mut self, id: PointId) -> Result<(), GpError>;
    fn refit_hyperparameters(&mut self, optimizer: &mut dyn Optimizer<T>) -> Result<(), GpError>;
}
```

`insert`/`delete`は現在のカーネル・ハイパラのままL・alphaを更新するだけで、ハイパラ再最適化は`refit_hyperparameters`を明示的に呼んだ場合のみ行う。Sparse GPのオンライン学習はスコープ外(§14)。

## 12. テスト計画(P2追加、ChatGPT指摘: 数値計算の正当性保証が設計書に不足していた)

速度より前に正しさを保証するテストを実装の各フェーズに組み込む。

1. **カーネルの数学的正当性**: RBF/Matern/Periodicの既知値比較、対称性、対角値、数値微分と解析的勾配の比較
2. **Choleskyの正当性**: `K=LLᵀ`再構成誤差、jitterあり/なし、悪条件・重複データでの挙動
3. **オンライン更新**: 1点追加/削除とフル再fitの結果一致、任意インデックス削除、追加削除の繰り返し、PointIdと内部インデックスの整合性(§11の不変条件の検証)
4. **精度**: f32/f64/混合精度の比較、悪条件行列、収束しないケースでのフォールバック挙動
5. **推論結果**: 既知の小規模GPR実装との比較(mean, variance, log marginal likelihood, gradient)

## 13. 実装ロードマップ(P2追加、ChatGPT提案を採用)

混合精度・Sparse GP・オンライン学習・IncrementalRecompute・SIMDバックエンドを同時に進めると問題の切り分けが困難になるため、段階的に実装する。

- **Phase 1(正しいExact GP)**: f64のみ、RBF/Matern、faer Cholesky、MLLと勾配、予測mean/variance、基本Optimizer、§12のテスト一式
- **Phase 2(高速化)**: CompiledKernel、距離キャッシュ、Workspace再利用、Rayon、SIMD、ベンチマーク
- **Phase 3(オンライン学習)**: insert/delete、PointId、フル再fitとの一致テスト(§12-3)
- **Phase 4(Sparse GP)**: VFEまたはFITCのどちらか一つ、誘導点固定、予測、ハイパラ最適化
- **Phase 5(高度な最適化)**: 混合精度(predict中心)、IncrementalRecompute、低ランク更新、MathBackendのFastApprox、DistanceCachePolicy::Autoの閾値調整

## 14. 未解決事項

1. **Sparse GPのオンライン学習**: 誘導点ZとデータXの非対称性があり、Phase 4以降の別設計が必要
2. **混合精度反復改良のパラメータ検証**: §4.2のデフォルト値は理論根拠付きだが、実ワークロードでの検証は未実施。fit時にMixedPrecisionを使う場合のlog|K|・トレース項の精度検証も含む
3. **DistanceCachePolicy::Autoの具体的な閾値**: カーネル種別・SIMD効率・メモリ帯域を考慮した実測が必要
4. **faerのinsert/delete_rows_and_cols_clobberの実API検証**: §11の数学的根拠と実際のAPI挙動(要求する行列形式、削除の任意インデックス対応、バージョン差異)を小規模行列テストで確認する(Phase 3着手時に実施)