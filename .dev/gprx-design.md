# gprx 設計ドキュメント

## 1. 目的・スコープ

最も柔軟かつ最も高速なGaussian Process Regressionライブラリを、Rustで構築する。「柔軟」はユーザー定義カーネル・前処理・厳密/疎推論・最適化器の差し替え可能性、および**データ点の逐次追加削除(オンライン学習)**を指し、「高速」はアロケーション最小化・SIMD/マルチスレッド活用・精度切り替えによる計算量/メモリ最適化を指す。

**改訂履歴**:
- 第1回: ChatGPT・Geminiのレビューを受け、混合精度の残差式、jitterと観測ノイズの混同、アロケーション方針、精度ジェネリクスを修正。
- 第2回: 再レビューを受け、次を反映。(1) MLL勾配のトレース項と`W`バッファ、(2) faer 0.24.4のCholesky更新API実態(LLTにinsert/deleteは無い)、(3) `GaussianLikelihood`のパラメータ化と勾配式の一致、(4) カーネルパラメータのflatten、(5) `y`のTargetTransform、(6) 混合精度の残差行列とjitterフォールバック方針、(7) 組み込みカーネルの静的ディスパッチ、(8) 予測分散の意味。実装順序は§13のロードマップに従う。
- 第3回: 公開面を `Gpr`（トレーナー）と `FittedGpr`（学習済み）に分ける。sklearn JSON は数値照合のみ。実装は P2-8。

## 2. 全体アーキテクチャ概要

```
入力 X, y
  → Transform Pipeline (Xの前処理: MinMax, Standardize等)
  → TargetTransform (yの標準化等。predict時にmean/varianceを逆変換)
  → Likelihood (観測ノイズσn²、モデルパラメータとして独立管理)
  → CompiledKernel<T> (KernelSpecをコンパイルした実行計画 + Workspace)
  → Gpr (トレーナー: カーネル・尤度・変換・FitOptions)
       → Objective (尤度・勾配。fit 中だけ)
       → Optimizer (argmin L-BFGS)
       → fit(self) → FittedGpr | (Gpr, GprError)
  → FittedGpr (L, α, X。predict / predict_into / refit / loo)
       → Phase 3: OnlineInference (`FittedGpr` 上、`&mut self`)
       → Phase 4: SparseGpr は同様に学習済み型を返す
```

主要な設計原則:
- **静的ディスパッチを基本に、拡張点(ユーザー定義カーネル)のみ`dyn`を許容**
- **gprx内部のホットパスでは新規アロケーションを行わない**(「fit中アロケーションゼロ」はユーザー定義カーネル実装まで強制できないため、この表現に修正)
- **精度はコンパイル時ジェネリクスで固定**
- **数値安定化(jitter)とモデルパラメータ(観測ノイズ)を明確に分離する**

## 3. 線形代数バックエンド: faer

依存は **faer 0.24.x**(本稿執筆時点のlatestは0.24.4)を前提とする。`Mat<T>`のストライド・ビュー制約は、ピンしたバージョンのAPIに合わせる(設計書側でレイアウトを凍結しない)。

Pure Rustで、OpenBLAS/LAPACK/Eigenと同等以上の性能を達成しており、RayonベースでOpenMP/TBB相当の並列化性能を持つ。

- `Mat<T>`は列優先(column-major)。**連続ストライドを前提にしたカーネルSIMDは、実際の`MatRef`/`MatMut`のストライドを実装時に確認してから書く**
- バッチfitのCholeskyは`llt::factor::cholesky_in_place`(下三角LLT、in-place)
- 動的正則化(jitter)は`LltRegularization`としてAPI組み込み済み。**ただしこれは純粋な数値安定化用であり、GPRの観測ノイズ(モデルパラメータ)とは別物として扱う**(§4.0)
- `Mat`は容量ベースの再確保をサポート(§11のオンライン学習で活用)
- **Cholesky更新APIの実態(faer 0.24.4で確認済み)**:
  - `llt::update`にあるのは`rank_r_update_clobber`のみ。**LLTに行・列のinsert/delete高水準APIは存在しない**
  - `ldlt::update::delete_rows_and_cols_clobber(LD, indices: &mut [usize], ...)`は存在し、任意インデックスの複数行削除に対応
  - `ldlt::update::insert_rows_and_cols_clobber`は公開されていない(`insert_rows_and_cols_clobber_scratch`のみ。本体は非公開)
  - オンライン学習はこれに合わせて§11の方針で実装する(追加は自前、削除はLDLT API)
- `llt::update::rank_r_update_clobber` / `ldlt::update::rank_r_update_clobber`: ランクr更新、低ランクΔKの場合のみ利用可(§5.4.1)

## 4. 精度ポリシーとノイズ/Jitterの分離

### 4.0 観測ノイズとJitterの分離(P0修正)

「観測ノイズσn²」(GPRのモデルパラメータ、最適化対象)と「Jitter」(Choleskyを正定値に保つための数値安定化オフセット)を分離する。

```rust
/// モデルの尤度。観測ノイズはここで管理し、最適化対象として扱う。
/// パラメータは最適化器と同じフラット配列で get/set する。
trait Likelihood<T: Scalar>: Send + Sync {
    fn num_params(&self) -> usize;
    fn get_params(&self, out: &mut [T]);
    fn set_params(&mut self, params: &[T]);
    fn add_noise_diag(&self, k_diag: &mut [T]); // K += σn²・I (対角への加算)
    /// ∂K/∂θ_{param_idx} の対角を dK_diag に書く。θ は get/set_params と同じパラメータ化。
    fn noise_grad_diag(&self, dK_diag: &mut [T], param_idx: usize);
}

/// θ = log(σn²)。正値制約は log パラメータ化で担保する。
/// σn² = exp(θ) なので、∂K/∂θ = exp(θ) I = σn² I。
/// ※ ∂K/∂σn = 2σn I は、標準偏差 σn をパラメータにした場合の式であり、本実装では使わない。
struct GaussianLikelihood<T: Scalar> {
    log_noise_variance: T,
}

impl<T: Scalar> GaussianLikelihood<T> {
    fn noise_variance(&self) -> T { self.log_noise_variance.exp() }
}

/// 純粋な数値安定化。モデルパラメータ(観測ノイズ)には触れない。
struct NumericalStability {
    policy: JitterPolicy,
}

enum JitterPolicy {
    Fixed(f64),
    Adaptive {
        initial: f64,
        multiplier: f64,  // リトライごとに jitter *= multiplier
        max_retries: usize,
        max_jitter: f64,
    },
}
```

`GaussianLikelihood`の`num_params()`は1。`get_params`/`set_params`は長さ1のスライスで`log_noise_variance`を読み書きする。`noise_grad_diag`は対角を`exp(θ)`で埋める。

`A = K + Likelihood.noise_diag`が**実際に解きたい線形システムの行列**(GPRのモデル)である。

**jitterの適用範囲**:
- `JitterPolicy`は**Cholesky分解そのものが失敗したときだけ**使う。分解に成功した因子は `A + j I` の因子であり、その場合に得ている解は `(A + j I)^{-1} y` である。使用した `j` はログおよび`CholeskyFailed`/`FitResult`に残す。
- jitterを増やして得た因子で、元の `A` へ反復改良で「戻す」ことはしない。前処理行列 `LLᵀ ≈ A + jI` と目標 `A` の乖離が拡大し、縮小率 `||I - (LLᵀ)^{-1} A||` が1を超えて発散し得るため(§4.2)。

### 4.1 精度ポリシー: f32/f64/混合精度

目的は「メモリ削減」と「計算速度」の両方。混合精度反復改良(mixed-precision iterative refinement)を採用するが、**適用範囲をfit時とpredict時で分ける**。

```rust
trait PrecisionPolicy {
    type Storage: Scalar;
    type Refine: Scalar;
}
struct MixedPrecision;  // Storage=f32, Refine=f64
struct SinglePrecision; // Storage=f32, Refine=f32
struct DoublePrecision; // Storage=f64, Refine=f64
```

**適用範囲の制限**: 周辺対数尤度(MLL)の`log|K| = 2Σlog(L_ii)`および勾配のトレース項`Tr(K⁻¹∂K/∂θ)`は、`α=K⁻¹y`の反復改良では高精度化されない(f32のLの対角値そのものに依存するため)。これらの項を含むfit時(ハイパーパラメータ最適化ループ)のデフォルトは**`DoublePrecision`**とする。`MixedPrecision`は`α`の線形ソルブのみで完結するpredict時(ハイパーパラメータ固定後の推論)を主対象とする。fit時にMixedPrecisionを使う場合は、log|K|・トレース項の精度検証を別途行うことを前提とする(§14)。

手順(predict時、または固定カーネルでのソルブ):
1. `A = K + Likelihood.noise_diag`をf32のまま`cholesky_in_place::<f32>`で分解(内部でjitterによる正則化のみ適用)
2. f32の`L`で`alpha_0 = solve(L, y)`
3. 残差をf64で計算する。**残差の対象行列 `A_resid` の構築方法は次の2通り**で、メモリ削減と精度がトレードオフになる:
   - **`PromoteStorage`(既定)**: 保存済みf32の`A`をf64へ昇格して `r = y_f64 - A_f32→f64 @ alpha`。これは「f32で保持した線形系」の解を改良する。真のf64カーネル行列に対するIRではない。f64の`A`を別途保持しないため、メモリ削減目的と整合する。
   - **`ReevaluateKernel`**: 残差matvecのたびにカーネルをf64で再評価する。`A_f64`は保持しない。反復1回あたりO(n²)のカーネル評価が乗るが、真のf64系により近い。
   - f64の`A`を丸ごと保持する方式はメモリ削減と矛盾するため採用しない。
4. f32の`L`で`delta = solve(L, r)`、`alpha_1 = alpha_0 + delta`
5. 収束するまで数回繰り返す

実装優先度: `DoublePrecision`をデフォルトとし、`MixedPrecision`はオプション機能として後付け(§13 Phase 5)。`A_resid`の方式はPhase 5着手時に上記2通りを実装可能にしてベンチマークで選ぶ。

### 4.2 混合精度反復改良の収束判定パラメータ

古典的な反復改良理論(Higham)より、分解精度u_f(f32≈1.19×10⁻⁷)と改良精度u_r(f64≈2.22×10⁻¹⁶)を使う場合、収束速度はκ(A)·u_fに依存する。**ただし実際の収束判定は理論値ではなく実測残差で行う**。

```rust
struct RefinementConfig {
    max_iterations: usize,   // デフォルト10
    relative_tolerance: f64, // デフォルト: 10.0 × n × u_r。判定は実測残差ノルムで行う
    stagnation_ratio: f64,   // デフォルト0.9
    fallback: RefinementFallback,
}

enum RefinementFallback {
    FallbackToDoublePrecision, // 第一選択。IR不収束は精度の問題として扱う
    ReturnError,
}
```

収束判定: `||r_k||∞ / (||A||∞ ||alpha_k||∞ + ||y||∞) < relative_tolerance`。`stagnation_ratio`超過が2回連続で発生したら`RefinementNotConverged`(§10)。

**IR不収束時にjitterを増やさない**: 分解側のjitterだけを増やすと、前処理`LLᵀ`と目標`A`の乖離が拡大してIRが発散し得る。IR不収束の第一選択は`FallbackToDoublePrecision`。jitter適応は§4.0の通りCholesky失敗時専用とする。

**位置づけ**: 理論的妥当性はあるが、実ワークロードでのパラメータ検証は今後の課題(§14)。

## 5. カーネル設計

### 5.1 Spec(宣言層)/ Evaluator(実行層)の分離、および精度ジェネリクス

`KernelSpec`(宣言層)は精度に依存しない型消去された表現とし、パラメータは常に`f64`で保持する(ユーザーが書く・読む値は精度非依存であるべきため)。`CompiledKernel<T>`(実行層)は`PrecisionPolicy::Storage`ごとにコンパイルされ、内部計算は`T`で行う。

最適化器はフラットな`params: &[T]`だけを見る。複合カーネルではリーフへの対応表が必要。

```rust
struct ParameterId(usize);
struct LeafId(usize);

struct ParameterBinding {
    id: ParameterId,
    leaf_id: LeafId,
    local_index: usize,
}

enum KernelSpec {
    Leaf(Box<dyn KernelTermSpec>),
    Sum(Box<KernelSpec>, Box<KernelSpec>),
    Product(Box<KernelSpec>, Box<KernelSpec>),
}

trait KernelTermSpec: Send + Sync {
    fn num_params(&self) -> usize;
    fn get_params(&self, out: &mut [f64]);
    fn set_params(&mut self, params: &[f64]);
    fn compile<T: Scalar>(&self) -> CompiledTerm<T>;
}

impl KernelSpec {
    fn num_params(&self) -> usize { /* リーフを走査して合計 */ }
    fn get_params(&self, out: &mut [f64]);
    fn set_params(&mut self, params: &[f64]);
    fn parameter_bindings(&self) -> Vec<ParameterBinding>;
    fn compile<T: Scalar>(&self) -> CompiledKernel<T>;
}

/// 実行層。組み込みはenumで静的ディスパッチ、ユーザー定義のみ dyn。
enum CompiledKernel<T: Scalar> {
    Rbf(RbfKernel<T>),
    Matern(MaternKernel<T>),
    Periodic(PeriodicKernel<T>),
    Sum(Vec<CompiledKernel<T>>),
    Product(Vec<CompiledKernel<T>>),
    Custom(Box<dyn KernelTerm<T>>),
}

enum Triangle { Lower, Upper, Full }

trait KernelTerm<T: Scalar>: Send + Sync {
    fn distance_kind(&self) -> DistanceKind;
    /// `uplo`で書き込む三角を指定する。既定契約は Lower。
    /// faerの cholesky_in_place は下三角のみ参照するため、Fullで埋めるとカーネル評価が約2倍になる。
    fn apply(&self, dist: MatRef<T>, out: MatMut<T>, uplo: Triangle);
    fn grad(&self, dist: MatRef<T>, dK: MatMut<T>, param_idx: usize, uplo: Triangle);
    fn rank_structure(&self) -> KRankStructure { KRankStructure::Dense }
    /// 次元 dim について ∂K(X1, X2)/∂(X2_{*, dim}) を一括計算する。
    /// 点ごと (m×d 回) の vtable 呼び出しは SIMD を阻害するため、座標1個ではなく次元単位にする。
    fn grad_wrt_coord_dim(&self, x1: MatRef<T>, x2: MatRef<T>, dK: MatMut<T>, dim: usize) -> Result<(), GprError> {
        Err(GprError::CoordGradientUnsupported)
    }
}
```

`KernelSpec`は演算子オーバーロードでユーザーが自然に合成でき、`KernelTermSpec`はobject-safeなのでユーザー定義カーネルはこれを実装するだけで組み込める。`CompiledKernel<T>`への変換をfit開始時に一度だけ行う。

**Lengthscale**: 等方はスカラー `ℓ`（`θ=log(ℓ)`）。ARD は次元ごとの `ℓ_d`（`θ_d=log(ℓ_d)`）。対象は lengthscale を持つ定常カーネル（RBF / Matern / RQ）。P1A-20 で RBF に口を固定し、P1A-14 / P1A-16 が同じ口を使う。Periodic の lengthscale はスカラーのまま。  
ARD の二乗距離は `r² = Σ_d (x_d - x'_d)² / ℓ_d²`。全 `ℓ_d` が等しいとき等方に一致する。`∂K/∂θ_d` には次元ごとの差が必要で、等方の二乗距離行列だけでは足りない。`n×n×d` キャッシュは §5.2 / P2-7（生の `(Δx_d)²`。ℓ 込みの `r²` は置かない）。P1A-20 では毎回座標から組む。P2-2 は等方の n×n。

ユーザー定義カーネル(`Custom`)はホットパスで新規アロケーションしないことを推奨するが、強制はしない(§2)。Phase 1では`Workspace`をユーザーカーネルに渡さない。安全APIとunsafe高速APIの二系統は設けない。

ホットパス(距離・カーネル評価の二重ループ)では`CompiledKernel`を`match`で静的ディスパッチする。`Custom`だけvtable経由。これは§2の「静的ディスパッチを基本に、拡張点のみdyn」と一致させる。

最適化器のパラメータ配列は次の順で連結する:

```
[kernel_params | likelihood_params]
```

Sparse GPRの誘導点ZはPhase 4では最適化対象に入れない(§6.1)。

### 5.2 距離キャッシュとキャッシュポリシー

等方カーネルは生の座標差がfit中不変のため、距離テンソルは1回計算して使い回す。

```rust
enum DistanceKind { SqEuclidean, SqEuclideanARD, Periodic { period: usize } }

/// キャッシュする中間表現。DistanceKind よりこちらが実体。
enum DistanceCache<T: Scalar> {
    None,
    SquaredEuclidean(Mat<T>),     // n×n、等方RBF/Matern等
    SquaredEuclideanArd(Mat<T>),  // n×n×d相当。メモリは K の約 d 倍
    Periodic(Mat<T>),             // sin²(π|x-x'|/p) など周期変換済み。生の二乗距離ではない
}
```

Periodicは二乗ユークリッド距離ではない。ARDは次元ごとの差が必要。キャッシュの単位はカーネル種別ごとに上記の中間表現とする。このenumをカーネル追加のたびに膨らませないため、**具体レイアウトはPhase 2の実装時に再検討**する。

**キャッシュ方針はベンチマークベースのポリシーとして抽象化する**。

```rust
enum DistanceCachePolicy {
    Never,
    Always,
    Auto { memory_budget_bytes: usize }, // n,d,メモリ予算から実装時にベンチマークして調整
}
```

理論的な参考値(目安であり決定基準ではない): `(n,n,d)`テンソルは`n²×d×sizeof(T)`バイト。基本のK行列自体もn²×sizeof(T)であり(例: n=5000,f64で約200MB)、ARDキャッシュはこれのd倍になる点に注意。d≪nの典型的GPRではキャッシュの投資対効果は薄いことが多い。`Auto`の具体的な閾値は実装後のベンチマークで決定する(§14)。

P2-2（[#26](https://github.com/YUKIKEDA/gprx/issues/26)）: `Never` / `Always` は既存の `Workspace.dist_cache`（等方 Dist/Either の n×n）に載せた。デフォルトは `Always`。`Auto` は P5-5。

P2-7（[#88](https://github.com/YUKIKEDA/gprx/issues/88)）: 同じ `DistanceCachePolicy` を ARD 葉の生の `(Δx_d)²` に載せる。ℓ 込みの `r²` は置かない。公開 Policy は増やさない。`Workspace` は `n` と `d` を見る。Always の ARD fit で 1 回確保し、等方 / `Never` では空（`kernel_scratch` と同じ）。apply/grad はキャッシュを読む。埋めは逐次（P2-3 に依存しない）。3D レイアウトは実装時（列優先・下三角。`DistanceCache` enum を膨らませない）。必須の数値は同じ固定問題の ARD RBF（`mll_and_grad_ard` / `fit_lbfgs_ard`、Always vs Never）。口が共通なら Matern/RQ ARD も同じ PR。`Auto` は P5-5。train×test / LOO / Linear / iso+ARD 混在は対象外。

### 5.3 CompiledKernelのplan構築アルゴリズム

Sum/Productは結合則・交換則が効くため、flatten+fold評価で済む。

1. 距離キャッシュ重複排除: 合成木を走査し`DistanceKind`集合を構築
2. flatten: `(A+B)+C`を`Sum(vec![A,B,C])`に正規化
3. plan生成: `BufAllocator`(フリーリスト)で`alloc()`/`free()`を追跡し、**実際のplanから動的に最大同時使用数を計算**してWorkspaceの確保サイズを決める

**訂正**: 「必要バッファ数はネストの深さでしか増えず、実用上3を超えない」という主張は誤り。`(A*B)*(C*D)`のような合成では兄弟項間でバッファを使い回せず、必要数が増える。固定上限を仮定せず、`WorkspacePlan { max_buffers, max_bytes }`をplan構築時に実測することとする。

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

plan実行時、組み込みリーフは`CompiledKernel`のenumアームを直接呼び、`Custom`だけ`dyn KernelTerm`に委譲する。

### 5.4 部分更新(コーディネート型最適化器)対応

**対応方針**: `RecomputeStrategy`として2種類。初期実装には不要でPhase 5。Exact GPRではCholeskyがO(n³)のため、カーネル構築の部分更新より先にフルK構築→Cholesky→MLL→勾配を正しく高速化する。

```rust
trait RecomputeStrategy {}
struct FullRecompute;  // バッファ最小、常にフル再計算。既定
struct IncrementalRecompute {
    leaf_contrib: Vec<Buf>,
    param_to_leaf: Vec<LeafId>,
}

/// IncrementalRecompute と Optimizer を繋ぐ差分情報。Phase 5。
struct ChangeSet {
    changed_param_indices: Vec<usize>,
}
```

`IncrementalRecompute`は変更indexに対応するリーフ項のみ再評価し、最終結合(O(n²×リーフ項数)、Choleskyに対して無視できるコスト)だけ毎回やり直す。**Cholesky分解自体はKが変わる以上フルで行う必要があり、部分更新の恩恵はカーネル行列構築コストにのみ及ぶ**。デフォルトは`FullRecompute`、`IncrementalRecompute`はオプトイン。

#### 5.4.1 IncrementalRecomputeとfaer update APIの関係

行・列のinsert/deleteはデータ点の追加削除用(§11)であり、ハイパラ変更には使えない。`rank_r_update_clobber`は、ハイパラ変更が`K`にもたらす差分`ΔK`が低ランクな場合(線形カーネル項のamplitude変更、全体スケール変更など)に限り使える。

```rust
enum KRankStructure { Scalar, LowRank(usize), Dense }
```

デフォルト`Dense`ならユーザー定義カーネルは安全側に倒れる。

### 5.5 前処理パイプライン

Xとyを分ける。GPRでは平均関数を持たない場合、**yを平均0・分散1に標準化することが数値安定性の基本**になる。予測値は元スケールへ戻す。

```rust
trait Transform {
    fn fit(&mut self, x: MatRef<f64>);
    fn apply(&self, x: MatMut<f64>);
}
struct Pipeline(Vec<Box<dyn Transform>>);

trait TargetTransform<T: Scalar>: Send + Sync {
    fn fit(&mut self, y: &[T]);
    fn transform(&self, y: &mut [T]);
    fn inverse_transform_mean(&self, mean: &mut [T]);
    fn inverse_transform_variance(&self, var: &mut [T]);
}

struct IdentityTarget<T>(PhantomData<T>);
struct StandardizeTarget<T: Scalar> { mean: T, std: T }
```

既定は`StandardizeTarget`。`predict`は内部で潜在/観測分散を計算したあと、`inverse_transform_mean`/`inverse_transform_variance`を通してから返す。分散の逆変換はアフィン `y' = (y - μ)/s` なら `Var(y) = s² Var(y')`。

## 6. GPModel抽象化(厳密/疎の差し替え)

学習と推論は型で分ける。未学習の `predict` は公開 API に置かない。sklearn の同一オブジェクト `fit` / `predict` は数値照合の対象であり、公開面の契約ではない。`Objective` は `fit` のあいだだけ `Gpr` を借り、学習済み値とは結合しない。

```rust
/// カーネル・尤度・変換・最適化設定。未学習。
struct Gpr { /* FitOptions, DistanceCachePolicy, transforms */ }

impl Gpr {
    fn fit(self, x: &[f64], n_rows: usize, n_cols: usize, y: &[f64])
        -> Result<FittedGpr, (Self, GprError)>;
}

/// 学習済み。L, α, X, カーネル, 尤度, 変換。W と L-BFGS 状態は持たない。
struct FittedGpr { /* … */ }

impl FittedGpr {
    fn predict(&self, xs: &[f64], n_rows: usize, n_cols: usize)
        -> Result<Prediction, GprError>;
    fn predict_into(
        &mut self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        out: &mut Prediction,
    ) -> Result<(), GprError>;
    fn refit(&mut self) -> Result<(), GprError>;
    fn log_marginal_likelihood(&self) -> Result<f64, GprError>;
    fn loo_predict(&self) -> Result<Prediction, GprError>;
}
```

P2-8 までは暫定で単一の `Gpr` と `fitted: bool`。実装後は `Gpr`（Exact）と `SparseGpr`（Phase 4）がそれぞれ学習済み型を返す。ハイパラ最適化は`Objective`(§9)を介して `fit` 中だけ扱う。

```rust
enum VarianceKind {
    Latent,       // 潜在関数 f* の分散(ノイズなし)
    Observation,  // 観測 y* の分散(σn² 込み)。既定
}

struct Prediction<T: Scalar> {
    mean: Vec<T>,
    variance: Vec<T>,
    variance_kind: VarianceKind,
}

struct PredictOptions {
    variance_kind: VarianceKind, // 既定 Observation
}
```

初期実装は対角分散のみ。フル共分散は将来拡張(§13 Phase 4以降)。`predict`は`PredictOptions`で分散の意味を切り替える。未指定時は`Observation`(ユーザーが欲しいのは多くの場合ノイズ込みの予測分散)。

### 6.1 Sparse GPRの誘導点キャッシュ問題

`K(X,X)`対角は不変なので1回計算・流用。`K(X,Z)`, `K(Z,Z)`はZが動くたびに再計算が必要だが、m(誘導点数)が小さいためCholeskyのO(nm²)に対して無視できるコストであり、キャッシュ対象にせず毎回再計算する。

誘導点座標の勾配は`grad_wrt_coord_dim`(§5.1)で扱い、未対応カーネルはpanicではなく`GprError::CoordGradientUnsupported`を返す。Phase 4ではZ固定のためこのAPIは使わない。Z最適化を後で足すときも、点ごとではなく次元一括で呼ぶ。

**Phase 4の初期実装では誘導点Zをk-means等で固定し、最適化対象はカーネルハイパラとノイズのみとする**。Zをθと同時最適化するとパラメータ数が m×d 増え、L-BFGSのメモリと収束性に大きく影響する。同時最適化・交互最適化はPhase 4の後続タスク(§14)。

Sparse GPRのオンライン学習は誘導点ZとデータXの非対称性のためスコープ外(§14)。

### 6.2 `Gpr` のMLLと勾配(P0追加)

ハイパーパラメータ勾配のアルゴリズムと必要なメモリが無いと、勾配ループで一時行列を確保してアロケーション方針に違反するか、パラメータごとに線形ソルブを繰り返してO(p n³)になる。

負の周辺対数尤度(最小化対象):

```
L(θ) = ½ yᵀ K⁻¹ y + ½ log|K| + (n/2) log(2π)
∂L/∂θ_i = -½ αᵀ (∂K/∂θ_i) α + ½ Tr(K⁻¹ ∂K/∂θ_i)
        = -½ ⟨W, ∂K/∂θ_i⟩_F
ただし α = K⁻¹ y、W = ααᵀ - K⁻¹
```

`(n/2) log(2π)` は θ に依らない。P2-6 で孤立加算は約 650 ps、`mll_and_grad` のあり/なし差は基準のゆらぎ以下だった。公開の NLML と最適化の `Objective` は同じ `L(θ)` のままにする。API は分けない。

標準アルゴリズム(Rasmussen & Williams / GPy系):

1. `k_matrix`に `A = K + σn² I` を構築(下三角のみ、§5.1の`uplo=Lower`)
2. in-place Cholesky。`k_matrix`はLになる
3. `log|K| = 2 Σ log(L_ii)` をLの対角から計算
4. `L Lᵀ α = y` を前進・後退代入で解く(O(n²))
5. `L`から`K⁻¹`を`w_matrix`へ計算する(三角ソルブで `L Lᵀ X = I`、O(n³)が1回)。Lは`k_matrix`に残す
6. `w_matrix[i,j] ← α[i] α[j] - K⁻¹[i,j]`(対称なので下三角のみ)
7. 各θ_iについて `∂K/∂θ_i` を`exp_buf`へ評価し、`⟨W, ∂K/∂θ_i⟩_F` をO(n²)で積算。カーネルパラメータは`KernelTerm::grad`、ノイズは`Likelihood::noise_grad_diag`(対角のみ)

全体コストはO(n³ + p n²)。K⁻¹をパラメータごとに作り直さない。

`value_and_gradient_into`はこの手順を一度で実行し、Lとαと`exp_buf`を尤度・勾配で共有する。デフォルト実装の`value`→`gradient_into`の二段呼びでは共有されない。

メモリ節約の代替(オプトイン、後付け可): 最適化ループ中はLを`K⁻¹`/`W`で上書きし、fit終了時にCholeskyを1回やり直してpredict用のLを復元する。Phase 1は`w_matrix`を独立確保し、Lを保持する。

### 6.3 Exact GPR (`Gpr` / `FittedGpr`)

公開面はトレーナーと学習済みモデルを分ける。実装は P2-8。それまでは単一の `Gpr` と `fitted: bool` が暫定の公開面。

`Gpr` は `KernelSpec`・`GaussianLikelihood`・変換・`FitOptions`・距離キャッシュ方針だけを持つ。`fit(self, …)` が L-BFGS（または `FitOptions::fixed` の一回分解）を回し、成功時に `FittedGpr` を返す。失敗時は消費した `Gpr` をエラーと一緒に返し、呼び出し側はハイパラやデータを直して再試行できる。`fitted: bool` と公開経路の [`GprError::NotFitted`] は P2-8 で外す。

`FittedGpr` は推論に必要な `L`・`α`・訓練 `X`・カーネル・尤度・変換を持つ。勾配用の `W`・`∂K`・argmin 状態は `fit` のあいだだけ生き、学習済み値には残さない。同一プロセスで `fit` の直後に `predict` する経路は少数派とみなす。学習済みモデルを渡すのが主経路なので、推論オブジェクトは `FittedGpr` である。

既定の `fit` は argmin の L-BFGS でハイパラを動かす。`FitOptions::fixed` は勾配を取らず、与えたハイパラで一度だけ分解する。`FittedGpr::predict` は対角分散のみ。`loo_predict` は GPML 5.4.2 の `L` と `α` から訓練点ごとの LOO を返す。ハイパラを変えて同じデータで分解し直すのは `FittedGpr::refit(&mut self)`。

```rust
struct Gpr {
    kernel: KernelSpec,
    likelihood: GaussianLikelihood,
    x_transform: Box<dyn InputTransform>,
    y_transform: Box<dyn TargetTransform>,
    fit_options: FitOptions,
    distance_cache_policy: DistanceCachePolicy,
}

enum DistanceCachePolicy {
    Never,
    Always,
}

struct FittedGpr {
    kernel: KernelSpec,
    likelihood: GaussianLikelihood,
    x_transform: Box<dyn InputTransform>,
    y_transform: Box<dyn TargetTransform>,
    workspace: Workspace<DoublePrecision>, // L。W は空でよい
    query: QueryWorkspace<DoublePrecision>, // predict_into 用
    compiled: CompiledKernel,
    alpha: Vec<f64>,
    x: Mat<f64>,
    y: Vec<f64>,
    n: usize,
    d: usize,
}

struct FitOptions {
    pub max_iterations: u64,      // 既定 1000
    pub tolerance: f64,           // 既定 1e-5
    pub history_size: usize,      // 既定 10
    pub line_search: LineSearch,  // StrongWolfe { c1: 1e-4, c2: 0.9 }
}

impl FitOptions {
    fn fixed() -> Self { /* max_iterations = 0 */ }
}

/// Objective は `Gpr` を fit 中だけ &mut で借り、set_params → MLL/勾配 を中継する。
/// パラメータの正本は Gpr.kernel / Gpr.likelihood。
struct GprObjective<'a> {
    model: &'a mut Gpr,
}
```

`x` は列優先の `&[f64]` で受け、内部で `n×d` の `Mat` に詰める。

| メソッド | レシーバ | 確保 |
| -------- | -------- | ---- |
| `FittedGpr::predict` | `&self` | 出力 `Prediction` と、必要なら一時 query バッファ |
| `FittedGpr::predict_into` | `&mut self` | warmup 後は 0。`mean` / `variance` の容量を再利用 |

前提条件:
- 未学習の `predict` は型で起きない（P2-8 まで暫定 `NotFitted`）
- `FittedGpr::refit` は同じ `n`/`d` で L と `α` を置き換える
- クエリの入力次元`d`は固定。不一致は`DimensionMismatch`
- n=0は`EmptyInput`、nがカーネルの最低点数未満なら`InsufficientData`
- 入力のNaN/Infは`NonFiniteInput`
- Cholesky失敗時は `Err((gpr, err))`。中途半端な `FittedGpr` は返さない

`DistanceCachePolicy::Always`（既定）は `fit` 開始時に訓練点の二乗距離を一度埋め、以降のハイパライテレーションではカーネルだけを書き換える。`Never` は毎回埋め直す。キャッシュは等方カーネル向けの `n×n`。ARD の `n×n×d` は P2-7。

### 6.4 Leave-one-out(P1B-7)

Exact GPR の leave-one-out は、学習後の `L` と `α` から閉じた式で出る(Rasmussen & Williams, GPML §5.4.2)。`A = K + σn² I`、`Q = A⁻¹`、`α = A⁻¹ y` として

```
μ_i = y_i - α_i / Q_ii
σ_i² = 1 / Q_ii
```

これは観測の `p(y_i | X, y_{-i}, θ)`。潜在 `f_i` の LOO 分散は `max(0, 1/Q_ii - σn²)`。`Q_ii` は下三角 `L` から `L⁻¹` の列ノルムで取る(`A⁻¹ = L^{-T} L^{-1}`)。コストは Cholesky と同オーダーの O(n³)、追加メモリは `n×n` の一時行列。Phase 1b の n=16 / 36 では問題にならない。

`FittedGpr::loo_predict`（P2-8 までは `Gpr::loo_predict`）は学習点と同じ長さの `Prediction` を返す。既定は `VarianceKind::Observation`。平均・分散は `predict` と同じく `TargetTransform` で元スケールへ戻す。White 葉は使わず、ノイズは `GaussianLikelihood` のみ。

sklearn に LOO API は無い。`just gen-goldens` は fit 後の `L_` / `alpha_` に同じ GPML 式を適用して JSON に書く。Rust 側は sklearn が選んだ `θ` で `FitOptions::FIXED` して照合する(最適化器差を LOO に混ぜない)。

## 7. Workspaceとメモリ管理

### 7.1 個別バッファ構造

バッファ数は少数・固定なので、個別フィールドとして持つ。精度ポリシーのStorage/Refineを明示的に反映する。

```rust
struct Workspace<P: PrecisionPolicy> {
    k_matrix: Mat<P::Storage>,       // K → Cholesky後は L
    w_matrix: Mat<P::Storage>,       // W = ααᵀ - K⁻¹。勾配のトレース項(§6.2)
    dist_cache: Mat<P::Storage>,
    exp_buf: Mat<P::Storage>,        // カーネル評価、および ∂K/∂θ の一時領域
    kernel_scratch: Mat<P::Storage>, // product `∂K/∂θ`。等方 RBF では空
    thread_scratch: Vec<Mat<P::Storage>>, // Rayonスレッド数ぶん事前分割
    rhs: Mat<P::Storage>,            // n×1、訓練 Cholesky の右辺 y → α
    refine_buf: Option<Mat<P::Refine>>, // MixedPrecision時のみ。DoublePrecisionではNone
    faer_scratch: MemBuffer,         // faer公式のスクラッチ機構をそのまま使う
}

// FittedGpr が保持。predict_into の warmup で (n, m, d) に合わせる
struct QueryWorkspace<P: PrecisionPolicy> {
    query_xs: Vec<f64>,              // 変換後クエリ（列優先）
    query_x: Mat<P::Storage>,        // m×d
    query_k_star: Mat<P::Storage>,   // n×m
    query_scratch: Mat<P::Storage>,
    query_dist: Mat<P::Storage>,
    query_kss: Vec<f64>,
}
```

fit 用バッファは`fit`開始時にサイズが確定するため、`reserve_exact`で一度だけ確保(または`Mat::zeros`で1回構築)し、以降のイテレーションでは同じ領域に上書きする。query バッファは `FittedGpr`（P2-8 までは暫定で同じ `Workspace`）が持ち、最初の `predict_into` で `(n, m, d)` に合わせ、同じクエリ長では再利用する。`predict(&self)` は出力 `Vec` を毎回確保してよい。あわせて、faer公式の`PodStack`/`MemStack`をスクラッチ管理に採用し、自前でスクラッチ領域をアリーナに内包する設計はやめる。

Rayon並列クロージャ内での新規確保は厳禁。`thread_scratch`を事前分割し、**並列領域に入る直前に`Workspace`から切り離して**分配する。`&mut self`(Objective/`Gpr`)をRayonクロージャに渡さない。

```rust
// 並列領域に入る前:
let scratches = &mut self.workspace.thread_scratch[..];
// par_chunks_mut / zip でワーカーに分配。
// k_matrix 等も as_mut でローカルに束縛してから並列化する。
```

### 7.2 メモリレイアウト

faerの`Mat`は列優先。実装時に対象バージョンの`MatRef`/`MatMut`ストライドを確認する。

- 距離行列・カーネル行列の走査は列優先、対称性を利用し**下三角のみ計算**(`KernelTerm::apply`の`uplo=Lower`)
- 入力`X(n×d)`は1データ点=1列=メモリ連続(`d×n`の列優先)で保持

### 7.3 イテレーション中のライフサイクル

```
fit()開始 → n,d確定 → 各Mat<T>を1回だけ確保 → 距離キャッシュ計算(1回)
  → 最適化ループ:
       k_matrix に A を下三角構築
       in-place Cholesky(同一領域が L になる)
       α, log|K|
       w_matrix に K⁻¹ → W
       exp_buf に ∂K/∂θ を順に書き ⟨W, dK⟩
fit()終了 → FittedGpr が L, α, X を保持。W / ∂K / L-BFGS は捨ててよい
  → predict(&self): 出力を確保
  → predict_into(&mut self): query_* に上書き、`Prediction` の容量を再利用
```

バッチfitのWorkspaceはn固定。オンライン学習の容量成長は`OnlineWorkspace`(§11)が担当し、バッチ用Workspaceとはメモリ管理方針を分ける。

## 8. 並列化・SIMD、数学関数バックエンド

- カーネル評価内側ループは `wide::f64x4` でベクトル化する（P2-5）。対象は列優先・単位行ストライドの等方 RBF `apply` / `grad` / `apply_cross` と二乗距離の行ループ。ストライドが 1 でないビューはスカラーに落とす。`std::simd` は安定化まで使わない。Matérn / Periodic / RQ の内側は未導入。
- 距離行列・カーネル行列構築はRayonでブロック並列化
- faer自身もRayon並列化されるため、外側との二重並列化に注意。単一の`rayon::ThreadPool`を共有

**MathBackendは最小限のAPIから始め、デフォルトは近似ではなく正確な実装にする**。カーネル行列の近似誤差は正定値性・Cholesky安定性・尤度・勾配・予測値すべてに波及するため。

```rust
trait MathBackend<T: Scalar>: Send + Sync {
    fn exp_inplace(&self, buf: &mut [T]); // 最初はexpのみ。erfは実際に必要になったカーネル(probit尤度等)が出てから追加
}
enum MathMode { Accurate, FastApprox }
```

デフォルトは`Accurate`(`StdExp`または`SleefBackend`)。`FastApprox`(`PolyApproxExp`)は明示的なfeatureや設定でオプトインし、**fit(ハイパラ最適化)では使わず、ハイパラ固定後の推論や大量predictに限定するのが安全**という位置づけにする。Phase 5。

## 9. Optimizer設計

**アロケーションフリー化とResultラップ**。

```rust
trait Objective<T: Scalar> {
    fn num_params(&self) -> usize;
    fn value(&mut self, params: &[T]) -> Result<T, GprError>;
    /// 勾配をoutに書き込む。勾配計算非対応ならErr(GprError::UnsupportedKernelOperation)
    fn gradient_into(&mut self, params: &[T], out: &mut [T]) -> Result<(), GprError>;
    /// 実際に内部計算(Cholesky, W, exp_buf)を共有する形で実装すること
    fn value_and_gradient_into(&mut self, params: &[T], out: &mut [T]) -> Result<T, GprError> {
        let v = self.value(params)?;
        self.gradient_into(params, out)?;
        Ok(v)
    }
}
trait Optimizer<T: Scalar> {
    fn minimize(&self, objective: &mut dyn Objective<T>, init: &[T]) -> Result<OptResult<T>, GprError>;
    fn requires_gradient(&self) -> bool;
}
```

`init`はスライスにする(呼び出し側のVecを消費しない)。`Gpr`の`GprObjective`は`value_and_gradient_into`をオーバーライドし、§6.2の手順でL・α・W・`exp_buf`を共有する。座標降下法的な最適化器を使う場合は§5.4の`ChangeSet`を伝播させ、`IncrementalRecompute`と接続する(Phase 5)。

## 10. エラー型 GprError

数値計算固有の失敗理由を拡充する。

```rust
#[derive(Debug, thiserror::Error)]
pub enum GprError {
    #[error("入力次元が一致しません: X.ncols()={x_dim}, 期待値={expected_dim}")]
    DimensionMismatch { x_dim: usize, expected_dim: usize },
    #[error("データ点数が不足しています: n={n}, 最低{min}点必要です")]
    InsufficientData { n: usize, min: usize },
    #[error("入力が空です")]
    EmptyInput,
    #[error("モデルが未学習です。先に fit を呼んでください")]
    NotFitted, // P2-8 で公開の predict 経路から外す。型で未学習を表す
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
    #[error("このカーネル項はSparse GPR用の座標微分(grad_wrt_coord_dim)を実装していません")]
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

GPRはn増加に伴いO(n³)でコストが増大するため、データの逐次追加削除を正式にスコープへ含める。バッチfit用Workspace(n固定)とは別に、`FittedGpr`向けに専用の`OnlineWorkspace`・更新経路を用意する（`&mut self`）。

### コスト比較

| 操作    | フル再fit | 増分更新 |
| ------- | --------- | -------- |
| 1点追加 | O(n³)     | O(n²)    |
| 1点削除 | O(n³)     | O(n²)    |

### faer APIに合わせた実装方針(P0修正)

§3の通り、**LLTにinsert/delete APIは無い**。オンライン経路は次で進める。

1. **追加(末尾append)**: 自前で bordered update を実装する。O(n²)
2. **削除(任意インデックス)**: `OnlineWorkspace`は**LDLT因子**を保持し、`ldlt::update::delete_rows_and_cols_clobber`を使う。Phase 3着手時に小規模行列でフル分解との一致を検証する
3. フォールバック(APIが期待通り動かない場合):
   - 末尾削除のみ自前実装し、任意削除はフル再分解
   - またはLLTに対するGivens回転ベースのdowndateを自前実装

バッチfitはLLTのままにする。初回`insert`時にLLT→LDLTへO(n²)で変換する:

- `D[j] = L_llt[j,j]²`
- `L_ldlt[:, j] = L_llt[:, j] / L_llt[j, j]`(対角は1)

### 増分追加の数学的根拠

下三角LLTの場合、新しい点を追加した行列は`K_new = [[K, k], [kᵀ, k_new]]`。既存の因子`L`に対し`L_new = [[L, 0], [vᵀ, d]]`とすると、`L_new L_newᵀ = K_new`を満たすには:

- `L v = k`(前進代入。`v = L⁻¹ k`)
- `d = √(k_new - vᵀ v)`

オンライン経路はLDLTなので、対応する bordered update は次:

`A = L D Lᵀ`のとき `A_new = [[L, 0], [vᵀ, 1]] [[D, 0], [0, δ]] [[Lᵀ, v], [0, 1]]`

- `L D v = k`、すなわち `L w = k`のあと `v = D⁻¹ w`
- `δ = k_new - vᵀ D v`

実装時に小規模行列(例: 2×2、5×5)でフル分解との一致を検証する(§12)。

### Workspaceの容量方式

バッチfitとオンラインは性質が異なる(n固定 vs n増減)。容量拡張時は関連バッファを**全て同じ手順で**再確保・コピーする。

```rust
struct OnlineWorkspace<T: Scalar> {
    k_matrix: Mat<T>,     // 下三角の A(容量 n_capacity)
    dist_cache: Mat<T>,
    ld_factor: Mat<T>,    // LDLT因子(対角=D、厳密下三角=L)
    alpha: Col<T>,
    y: Col<T>,
    v_buf: Col<T>,        // 予測分散の前進消去スクラッチ(テスト点1点あたりO(n²))
    n_active: usize,
    n_capacity: usize,
    growth_factor: f64,   // デフォルト1.5〜2.0
}
```

**容量拡張手順** (`n_active == n_capacity`のとき、insertの前に行う):

1. `new_cap = max(n_capacity + 1, (n_capacity as f64 * growth_factor) as usize)`
2. `k_matrix`, `dist_cache`, `ld_factor`, `alpha`, `y`, `v_buf`を`new_cap`で再確保
3. 既存の`n_active × n_active`ブロックと長さ`n_active`のベクトルをコピー
4. `PointRegistry`のインデックスは`n_active`未満のままなので付け替え不要
5. 拡張後にinsertを実行する。更新アルゴリズムの最中には再確保しない

**predict時の分散計算**: 予測平均はO(n)だが、予測分散`σ*² = k(x*,x*) - vᵀ D v`(LDLT、`L v = k*`の変形)はテスト点1点あたりO(n²)。`v_buf`をあらかじめ確保しておく。

### 増分更新の手順と不変条件

**追加**: ①容量が足りなければ拡張 → ②新規点と既存n点との距離計算(O(n)) → ③カーネル評価しKに新規行/列追加 → ④bordered LDLT update(O(n²)) → ⑤alpha再ソルブ(O(n²)) → ⑥`PointRegistry`にPointIdを登録

**削除**: ①`ldlt::update::delete_rows_and_cols_clobber`でLD更新(O(n²)) → ②距離キャッシュ・K・y・alphaから該当要素を除去し、後ろの行/列を詰める(O(n)) → ③`PointRegistry`のインデックスを同じ順序でシフト → ④alpha再ソルブ(O(n²))

**不変条件**: 削除により内部インデックスがシフトする際、`K`, `LD`, `y`, `alpha`, 距離キャッシュ, `PointRegistry`は**必ず同じ順序で同期**しなければならない。いずれか一つでも順序がずれると誤った解になる。この不変条件をテスト(§12)で明示的に検証する。

```rust
struct PointRegistry {
    id_to_index: HashMap<PointId, usize>,
    index_to_id: Vec<PointId>,
}
```

### API

**insert/deleteとハイパラ再最適化を分離する**。

Phase 3 の対象は `FittedGpr`（`&mut self`）。未学習の `Gpr` には点を足さない。

```rust
trait OnlineInference<T: Scalar> {
    fn insert(&mut self, x_new: &[T], y_new: T) -> Result<PointId, GprError>;
    fn delete(&mut self, id: PointId) -> Result<(), GprError>;
    fn refit_hyperparameters(&mut self, optimizer: &mut dyn Optimizer<T>) -> Result<(), GprError>;
}
```

`insert`/`delete`は現在のカーネル・ハイパラのままLD・alphaを更新するだけで、ハイパラ再最適化は`refit_hyperparameters`を明示的に呼んだ場合のみ行う。Sparse GPRのオンライン学習はスコープ外(§14)。

## 12. テスト計画

速度より前に正しさを保証するテストを実装の各フェーズに組み込む。

1. **カーネルの数学的正当性**: RBF/Matern/Periodicの既知値比較、対称性、対角値、数値微分と解析的勾配の比較、`uplo=Lower`と`Full`の一致
2. **Choleskyの正当性**: `K=LLᵀ`再構成誤差、jitterあり/なし、悪条件・重複データでの挙動
3. **MLLと勾配**(推論テストから独立させる):
   - 既知の小規模問題でのMLL解析値比較
   - MLLの数値微分と解析的勾配の比較
   - 各カーネルパラメータの勾配比較
   - ノイズパラメータ(`log_noise_variance`)の勾配比較。`∂K/∂θ = σn² I`であること
   - 悪条件行列での勾配安定性
4. **オンライン更新**: 1点追加/削除とフル再fitの結果一致、任意インデックス削除、追加削除の繰り返し、PointIdと内部インデックスの整合性(§11の不変条件)
5. **オンラインのプロパティテスト**: ランダムなinsert/delete列の各段階で `incremental == full refit`(mean, variance, LML, alpha)。特に削除順をランダム化する
6. **精度**: f32/f64/混合精度の比較、悪条件行列、収束しないケースでのf64フォールバック
7. **推論結果**: 既知の小規模GPR実装との比較(mean、潜在分散、観測分散、log marginal likelihood, gradient)。sklearn JSON は数値の第二照合であり、公開 API の契約ではない。アルゴリズムの正本は GPML / Rasmussen
8. **前処理**: `StandardizeTarget`適用後のpredictが、未標準化モデルと元スケールで一致すること(アフィン変換の閉じた関係)
9. **最適化後の推論**(P1B-6): 1次元 Forrester と 2次元重み付き球関数（ARD）で sklearn L-BFGS と `Gpr::fit` を緩い許容で照合する。固定ハイパラ JSON（1e-8）とは分ける。`cargo test` は Python を呼ばない
10. **Leave-one-out**(P1B-7): n=2 の GPML 解析式、n=3 の実 leave-one-out `fit`+`predict`、および P1B-6 JSON の LOO 欄を sklearn の `θ` で照合する。`cargo test` は Python を呼ばない

## 13. 実装ロードマップ

混合精度・Sparse GPR・オンライン学習・IncrementalRecompute・SIMDバックエンドを同時に進めると問題の切り分けが困難になるため、段階的に実装する。

**タスク分解・完了条件・Issue 化は [.dev/roadmap.md](roadmap.md)。進め方は [AGENTS.md](../AGENTS.md) と `.cursor/rules/`。** 今の着手点は Phase 2（P2-8）。`phase-1b` のボトルネック順は [bench-log.md](bench-log.md)。Phase 1 は 1a（固定ハイパラ）→ 1b（argmin L-BFGS）で 0.1.0 相当。

- **M0(Spike)**: クレート初期化と faer 0.24 の Cholesky 往復。GPR は書かない
- **Phase 1a(固定ハイパラ Exact GPR)**: f64、RBF で経路を通したあと Constant/Linear/Matern/Periodic/RQ/White、LLT、§6.2 の MLL と勾配、`TargetTransform`、分散種別、解析解と sklearn golden JSON。**criterion と確保 ratchet も 1a で始める**（§15）
- **Phase 1b(Optimizer と 0.1 API)**: argmin の L-BFGS、README / rustdoc / 例。crates.io には出さない
- **Phase 2(高速化)**: `phase-1b` の数値を見て距離キャッシュ・Rayon。P2-5 で等方 RBF と二乗距離に `wide::f64x4` を入れた。P2-6 で NLML 定数項の差はノイズなので `L(θ)` は一本のまま。P2-8 で `Gpr` / `FittedGpr` の typestate（速度行のあと）
- **Phase 3(オンライン学習)**: `FittedGpr` 上でデータ点の追加削除。自前insert、LDLT delete、PointId、容量拡張、フル再fitとの一致およびプロパティテスト(§12-4, §12-5)
- **Phase 4(Sparse GPR)**: VFEまたはFITCのどちらか一つ、**誘導点Zは固定**、対角予測、ハイパラ最適化(Zは含めない)
- **Phase 5(高度な最適化)**: 混合精度(predict中心、`A_resid`の2方式)、IncrementalRecompute、低ランク更新、MathBackendのFastApprox、DistanceCachePolicy::Autoの閾値調整

## 14. 未解決事項

1. **Sparse GPRのオンライン学習**: 誘導点ZとデータXの非対称性があり、Phase 4以降の別設計が必要
2. **混合精度反復改良のパラメータ検証**: §4.2のデフォルト値は理論根拠付きだが、実ワークロードでの検証は未実施。`PromoteStorage`と`ReevaluateKernel`の精度差、fit時MixedPrecisionのlog|K|・トレース項も含む
3. **DistanceCachePolicy::Autoの具体的な閾値**: カーネル種別・SIMD効率・メモリ帯域を考慮した実測が必要
4. **`ldlt::update::delete_rows_and_cols_clobber`の実測**: 任意インデックス・複数行・更新後LDの正しさをPhase 3着手時に小規模行列で確認する。失敗時は§11のフォールバック(末尾削除+フル再分解、またはGivens downdate)
5. **Sparse GPRの誘導点Zの最適化**: Phase 4では固定。同時最適化か交互最適化かは後続で決める

## 15. ベンチマーク戦略

「最も高速」「アロケーション最小」は Phase 2 で突然測り始めても絵になる。**正しさの次に、同じ経路を測りながら積む。** Phase 2 は最適化のフェーズであり、計測の開始点ではない。詳細な運用は `.cursor/rules/bench.mdc`。

### 15.1 二系統

| 系統 | 道具 | いつ回す | 見るもの |
|---|---|---|---|
| 時間 | criterion、`benches/exact.rs` | `just bench`（ローカル）。既定 CI では回さない（ノイズ） | 壁時計。グループを分けて測る |
| 確保 | `tests/alloc.rs` | `just test`（必須） | Workspace 確保**後**の新規確保回数。上限は ratchet（減ることはあっても、Issue なしに増えない） |

時間と確保を一つの数字に混ぜない。L-BFGS 全体と「MLL+勾配 1回」も混ぜない。

### 15.2 固定問題（回帰の単位）

毎回同じ入力でないと、速くなったのかデータが変わったのか分からない。

- RNG seed `0`、`d = 8`、RBF + `GaussianLikelihood`、ハイパラ固定
- `n = 256` を P1A-18 から必須。`512` / `1024` は数秒で終わるようになってから足す
- グループ（存在する経路だけ。無いものはまだ書かない）:
  1. `kernel_rbf` — K の下三角構築
  2. `cholesky_alpha` — `A` の LLT と `α`
  3. `mll_and_grad` — §6.2 の 1 評価（P1A-10 から）
  4. `predict_100` — テスト点 100（P1A-8 から）
  5. `fit_lbfgs` — 最適化ループ全体（1b から。1 と混ぜない）
  6. `mll_and_grad_ard` / `fit_lbfgs_ard` — 同じ n,d,seed の ARD RBF（P2-7）。Always vs Never。等方 `phase-1b` とは比べない
  7. `online_insert` / `online_delete` — Phase 3

### 15.3 いつ何を足す

| 時点 | やること |
|---|---|
| M0 | 箱だけ。空の `benches/` は置かない |
| P1A-7 の直後（P1A-18） | criterion と `just bench`。`kernel_rbf` と `cholesky_alpha` |
| P1A-8 / P1A-10 | 同じファイルに `predict_100` / `mll_and_grad` を足す。P1A-19 で確保 ratchet |
| 1a 完了 | 名前付き baseline `phase-1a` を取り、機械名と数値を `.dev/bench-log.md` に残す |
| 1b 完了 | `fit_lbfgs` を足し、baseline `phase-1b` |
| Phase 2 | **新しいハーネスは不要。** `phase-1b` を見てボトルネック順に最適化する。P2-5: 等方 RBF と距離に SIMD。可否は `kernel_rbf` / `predict` / `FIXED` で判断し、`mll_and_grad` の勾配項だけを分母にしない。NLML 定数項は P2-6 で測り、差はノイズなので `L(θ)` は一本のまま。ARD 距離キャッシュは P2-7 で `mll_and_grad_ard` / `fit_lbfgs_ard` の Always vs Never |
| Phase 3+ | insert/delete などを同じ問題定義で足す |

ホットパス（`src/kernel/`、`workspace`、`exact`、`objective`、`online`）の PR は、Verification に前回 baseline との criterion 結果を貼る。速さと無関係ならその理由を書く。

### 15.4 指標

目標比は `phase-1b` を取ってから置く。それまでは「前より悪くない」がゲート。

| 指標 | 内容 |
|---|---|
| MLL+grad 1回 | n, カーネル別。最適化ループとは別 |
| Fit（L-BFGS） | イタレーション込み。1b から |
| Predict | テスト点数別。潜在 / 観測 |
| ピークメモリ | Workspace 込み。`w_matrix` を含む |
| Allocations | セットアップ後の回数。ratchet → 最終的にホットパス 0 |
| 並列 | スレッド数別。faer との二重並列に注意。Phase 2 |
| f32/f64 | 精度と速度。Phase 5 |
| Online insert/delete | 1点 vs フル再 fit。Phase 3 |

### 15.5 やらないこと

- 測らずに「速くなるはず」で Rayon / SIMD / 近似 exp を入れる
- CI の criterion を赤/緑のゲートにする（マシン差でフレークする）
- 確保 0 を 1a 初日のテストで要求する（まず数え、上限を段階的に下げる）

