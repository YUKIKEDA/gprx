# gprx 設計ドキュメント

## 1. 目的・スコープ

最も柔軟かつ最も高速なGaussian Process Regressionライブラリを、Rustで構築する。「柔軟」はユーザー定義カーネル・前処理・厳密/疎推論・最適化器の差し替え可能性、および**データ点の逐次追加削除(オンライン学習)**を指し、「高速」はアロケーション最小化・SIMD/マルチスレッド活用・精度切り替えによる計算量/メモリ最適化を指す。

この文書には今の設計だけを書く。判断の理由は [adr/](adr/)、並びと状態は [roadmap.md](roadmap.md) にある。

## 2. 全体アーキテクチャ概要

```
入力 X, y
  → UnfittedTransform / Transform（X の前処理: MinMax、Standardize、列ごと、パイプライン）
  → UnfittedTarget / TargetTransform（y の標準化など。predict 時に平均・分散を逆変換）
  → GaussianLikelihood（観測ノイズ σn²。モデルパラメータとして独立に持つ）
  → KernelSpec → CompiledKernel<T>（平らにした木。組み込みのカーネルの葉は静的ディスパッチ）
  → Gpr<O, P>（トレーナー: カーネル・尤度・変換・最適化器・方針）
       → GprObjective（NLML とその微分。fit / refit のあいだだけ）
       → O: Optimizer<GprObjective>（既定 `Lbfgs`。argmin のソルバも自作 `O` も同じ入口）
       → fit(self) → FittedGpr | (Gpr, GprError)。θ 固定は Gpr<Fixed>::factor
  → FittedGpr（LLT、α、X。predict / predict_into / 共分散 / sample / loo / refit / save）
       → OnlineGpr: `FittedGpr::into_online(self)` で LLT→LDLT。`insert` / `delete` はここだけ
  → Sgpr / Svgp: Sparse のトレーナー。学習済み型は FittedSgpr、OnlineSgpr、FittedSvgp（§6.1）
  → persist: モデルごとに 1 ディレクトリ（`config.json` + `model.safetensors`）（§6.3、§11）
```

persist は 1 ディレクトリに書く。`format_version` は 1。Exact のモデルは `factor_kind` が必須で（`llt` は `FittedGpr`、`ldlt` は `OnlineGpr` として読む）、保存した因子は mmap する。Sparse のモデルは `model` キー（`sgpr` / `online_sgpr` / `svgp`。Exact のファイルには無い）を足し、`LoadedSgpr` / `LoadedSvgp` で読む。テンソルは元の `X` / `y` / `Z`、変換後の `Z`、SVGP の `q(u)`。因子は組み直すので、読み込んだモデルは同じ値をビットで予測する。`config.json` の浮動小数点は正確に往復する（serde_json の `float_roundtrip`）。読み込んだモデルの再学習は `with_optimizer` → `refit`。供給された距離のモデル（5.6 節）は `distance` キーと学習の `d²` を足し、マーカーごとの `LoadedDistanceGpr` / `LoadedDistanceSgpr` / `LoadedDistanceSvgp` で読む。すべてのキーとテンソルは [persist-format.ja.md](persist-format.ja.md)。モジュールと依存の向きは [architecture.ja.md](architecture.ja.md)。

主要な設計原則:
- **識別子は gprx / GPR の概念を名付ける**（カーネル、尤度、θ、分解、正パラメータの区間、…）。他製品・テストハーネス・無関係なドメインの名前は置かない
- **静的ディスパッチを基本に、拡張点(ユーザー定義カーネル)のみ`dyn`を許容**
- **gprx内部のホットパスでは新規アロケーションを行わない**(「fit中アロケーションゼロ」はユーザー定義カーネル実装まで強制できないため、この表現にする)
- **精度はコンパイル時ジェネリクスで固定**
- **数値安定化(jitter)とモデルパラメータ(観測ノイズ)を明確に分離する**

## 3. 線形代数バックエンド: faer

依存は **faer 0.24.x** を前提とする。`Mat<T>`のストライド・ビュー制約は、ピンしたバージョンのAPIに合わせる(設計書側でレイアウトを凍結しない)。

Pure Rustで、OpenBLAS/LAPACK/Eigenと同等以上の性能を達成しており、RayonベースでOpenMP/TBB相当の並列化性能を持つ。

- `Mat<T>`は列優先(column-major)。**連続ストライドを前提にしたカーネルSIMDは、実際の`MatRef`/`MatMut`のストライドを実装時に確認してから書く**
- バッチfitのCholeskyは`llt::factor::cholesky_in_place`(下三角LLT、in-place)
- jitter はその分解のまわりの gprx の再試行ループ（`JitterPolicy`、§4.0）で、各試行が自分の `j` を faer の `LltRegularization` に渡す。**これは純粋な数値安定化用であり、GPRの観測ノイズ(モデルパラメータ)とは別物として扱う**
- `Mat`は容量ベースの再確保をサポート(§11のオンライン学習で活用)
- **faer 0.24 の Cholesky 更新 API**:
  - `llt::update`にあるのは`rank_r_update_clobber`のみ。**LLTに行・列のinsert/delete高水準APIは存在しない**
  - `ldlt::update::delete_rows_and_cols_clobber(LD, indices: &mut [usize], ...)`は存在し、任意インデックスの複数行削除に対応
  - `ldlt::update::insert_rows_and_cols_clobber`は公開されていない(`insert_rows_and_cols_clobber_scratch`のみ。本体は非公開)
  - オンライン学習はこれに合わせて§11の方針で実装する(追加は自前、削除はLDLT API)
- `llt::update::rank_r_update_clobber` / `ldlt::update::rank_r_update_clobber` は使わない。Sparse のオンライン更新は rank-1 の cholupdate を自前で持ち（ADR 0004 / 0005）、ハイパラの変更は常にフル再分解（§5.4）

## 4. 精度ポリシーとノイズ/Jitterの分離

### 4.0 観測ノイズとJitterの分離

「観測ノイズσn²」(GPRのモデルパラメータ、最適化対象)と「Jitter」(Choleskyを正定値に保つための数値安定化オフセット)を分離する。

```rust
/// Observation noise σn², stored as θ = log(σn²) on an open interval of σn².
/// σn² = exp(θ), so ∂K/∂θ = σn² I (not the 2σn I of a σn parameterization).
pub struct GaussianLikelihood {
    noise_variance: BoundedParam,
}

impl GaussianLikelihood {
    pub fn new(noise_variance: f64) -> Result<Self, GprError>;
    pub fn from_log_noise_variance(log_noise_variance: f64) -> Result<Self, GprError>;
    pub fn noise_variance(&self) -> f64;
    pub fn log_noise_variance(&self) -> f64;
    pub fn bounds(&self) -> Interval;
    pub fn with_bounds(self, interval: Interval) -> Result<Self, IntervalError>;
    pub fn num_params(&self) -> usize;                                   // always 1
    pub fn get_params(&self, out: &mut [f64]) -> Result<(), GprError>;   // θ
    pub fn set_params(&mut self, params: &[f64]) -> Result<(), GprError>;
    pub fn add_noise_diag(&self, k_diag: &mut [f64]);                    // K += σn² I
    pub fn noise_grad_diag(&self, dk_diag: &mut [f64], param_idx: usize) -> Result<(), GprError>;
}

/// Numerical Cholesky stabilizer. It never touches σn².
pub enum JitterPolicy {
    Fixed(FixedJitter),       // JitterPolicy::fixed(j): one retry with j ≥ 0
    Adaptive(AdaptiveJitter), // JitterPolicy::adaptive(initial, multiplier, max_retries, max_jitter)
}
```

尤度は trait ではなく具体型にする。尤度はガウスノイズだけだから。`get_params` / `set_params` は長さ 1 のスライスで `θ` を読み書きする。`noise_grad_diag` は対角を `σn²` で埋める。jitter の方針はモデルの `with_jitter_policy` で渡す。別の安定化オブジェクトは置かない。

`A = K + σn² I` が**実際に解きたい線形システムの行列**(GPRのモデル)である。

**jitterの適用範囲**:
- 最初の分解は常に、対角に何も足さない `A` で試す。`JitterPolicy` は**その Cholesky が失敗したときだけ**使う。`Fixed` は自分の `j` で 1 回だけ再試行する。`Adaptive` は `initial` で再試行し、以後は `multiplier` 倍にして、`max_retries` に達するか `j` が `max_jitter` を超えるところで止める。分解に成功した因子は `A + j I` の因子であり、その場合に得ている解は `(A + j I)^{-1} y` である。モデルはその `j` を持ち（crate 内の `factor_jitter`。因子と一緒に保存もする）、`CholeskyFailed::jitter` は最後に試した `j`。Exact の既定は `fixed(0.0)` で、再試行しない。
- jitterを増やして得た因子で、元の `A` へ反復改良で「戻す」ことはしない。前処理行列 `LLᵀ ≈ A + jI` と目標 `A` の乖離が拡大し、縮小率 `||I - (LLᵀ)^{-1} A||` が1を超えて発散し得るため(§4.2)。
- Sparse のモデルが分解する `K_mm = k(Z, Z)` には観測ノイズが入らない。`Sgpr` / `Svgp` は `K_mm` 用の `with_jitter_policy` を持ち、既定は Exact の既定ではなく `adaptive(1e-8, 10, 5, 1e-3)` にする。近い誘導点では、浮動小数点で `K_mm` が特異になるため。fit、factor、`set_params`、予測、オンライン更新は、すべてこの方針を使う。

**非正規化数**（#51 で決定）。非正規化数の `f64` は正の有限値であり、gprx は IEEE 754 のとおりに扱う。公開の入口で拒否もゼロへのフラッシュもしない。gprx は浮動小数点の制御レジスタ（x86 の MXCSR、AArch64 の FPCR）を読み書きしない。flush-to-zero と denormals-are-zero は呼び出し側のプロセスが決めることで、使うなら両方をまとめて設定する。片方だけのスイッチは置かない。「ノイズが小さすぎて役に立たない」はモデルの問題として別に扱い、パラメータの `Interval`（既定の `Interval::DEFAULT_POSITIVE` は `(1e-5, 1e5)` なので、非正規化数のノイズ分散には呼び出し側が範囲を広げたときしか届かない）と、分解が失敗したときの `JitterPolicy` が受け持つ。非正規化数の規則ではない。

### 4.1 精度ポリシー: f32/f64/混合精度

目的は「メモリ削減」と「計算速度」の両方。混合精度反復改良(mixed-precision iterative refinement)を採用するが、**適用範囲をfit時とpredict時で分ける**。

```rust
pub trait PrecisionPolicy {
    type Storage: KernelScalar; // factor and kernel matrices
    type Refine: KernelScalar;  // predictive α and the returned Prediction<Refine>
}
pub struct DoublePrecision;                                     // Storage=f64, Refine=f64 (default)
pub struct SinglePrecision;                                     // Storage=f32, Refine=f32
pub struct MixedPrecision<R: ResidualFormula = PromoteStorage>; // Storage=f32, Refine=f64
pub struct PromoteStorage;   // residual from the stored f32 matrix
pub struct ReevaluateKernel; // residual from the kernel re-evaluated in f64
```

モデルの精度は型パラメータ `P`（Exact は `GpScalar`、Sparse は `ModelPrecision`。どちらも上の 3 型に実装がある）で、`with_precision::<P2>()` で切り替える。`ModelPrecision` はモデル間で共有する精度ごとの振る舞いを持つ。

**適用範囲の制限**: 周辺対数尤度(MLL)の`log|K| = 2Σlog(L_ii)`および勾配のトレース項`Tr(K⁻¹∂K/∂θ)`は、`α=K⁻¹y`の反復改良では高精度化されない(f32のLの対角値そのものに依存するため)。これらの項を含むfit時(ハイパーパラメータ最適化ループ)のデフォルトは**`DoublePrecision`**とする。`MixedPrecision`は`α`の線形ソルブのみで完結するpredict時(ハイパーパラメータ固定後の推論)を主対象とする。fit時にMixedPrecisionを使う場合は、log|K|・トレース項の精度検証を別途行うことを前提とする(§14)。

手順(predict時、または固定カーネルでのソルブ):
1. `A = K + σn² I` をf32のまま`cholesky_in_place::<f32>`で分解(内部でjitterによる正則化のみ適用)
2. f32の`L`で`alpha_0 = solve(L, y)`
3. 残差をf64で計算する。**残差の対象行列 `A_resid` の構築方法は次の2通り**で、メモリ削減と精度がトレードオフになる:
   - **`PromoteStorage`(既定)**: 保存済みf32の`A`をf64へ昇格して `r = y_f64 - A_f32→f64 @ alpha`。これは「f32で保持した線形系」の解を改良する。真のf64カーネル行列に対するIRではない。f64の`A`を別途保持しないため、メモリ削減目的と整合する。
   - **`ReevaluateKernel`**: 残差matvecのたびにカーネルをf64で再評価する。`A_f64`は保持しない。反復1回あたりO(n²)のカーネル評価が乗るが、真のf64系により近い。
   - 反復改良は、fit（またはオンライン更新）が残した因子の上で行う。`α₀` はその因子で `y` を解いたもの、補正も同じ因子で解く。解く系は `A + (σn² + j) I` で、`j` は因子の再試行で足した jitter（再試行なしなら `0`）。fit が `j` を記録し、オンラインの追加も同じ `j` を足し、保存した因子にも `j` を残す。f32 の分解をやり直さない。収束した `PromoteStorage` の α は f64 の系で1回だけ確かめる（カーネルを列ブロックで評価し、`n×n` の f64 行列は持たない）。`κ(A) u_f32` が大きく満たさないときは、同じ系の f64 Cholesky の解に落とす。f64 へのやり直しはモデルの `JitterPolicy` で再試行し、失敗時は呼び出し元の段を返す。
   - f64の`A`を丸ごと保持する方式はメモリ削減と矛盾するため採用しない。
4. f32の`L`で`delta = solve(L, r)`、`alpha_1 = alpha_0 + delta`
5. 収束するまで数回繰り返す

省略時の精度は `DoublePrecision`。`SinglePrecision` は同じ手順を f32 で計算し、分解結果をそのまま使う。残差の型パラメータは持たない。`MixedPrecision<R>` は f32 で分解し、予測用の α だけ反復改良する。学習中の MLL と勾配は、その精度の因子を使い、反復改良は学習ループの中では行わない。残差の型パラメータは `MixedPrecision` にだけ付き、既定は `PromoteStorage`。Forrester `n=1024` の release 中央値は `PromoteStorage` が 22.40 ms、`ReevaluateKernel` が 48.77 ms。フラグと、コード上の別名は置かない。すべてのモデル（Exact、オンライン、`Sgpr`、`Svgp`）とすべての最適化器が 3 つの精度を受ける。

### 4.2 混合精度反復改良の収束判定パラメータ

古典的な反復改良理論(Higham)より、分解精度u_f(f32≈1.19×10⁻⁷)と改良精度u_r(f64≈2.22×10⁻¹⁶)を使う場合、収束速度はκ(A)·u_fに依存する。**ただし実際の収束判定は理論値ではなく実測残差で行う**。

パラメータは公開の設定ではなく、`src/precision/refine.rs` の crate 内定数に固定する。

| パラメータ | 値 |
| --- | --- |
| 補正の最大回数 | 10 |
| 相対許容 | `10 · dim · u_r`（`u_r = f64::EPSILON`）。判定は実測残差 |
| 停滞 | 残差ノルム比が `0.9` 超えを2回連続 |
| 不収束 | 同じ系の `f64` の解（エラーにはしない） |

収束判定: `||r_k||∞ / (||B||∞ ||w_k||∞ + ||b||∞) < 10 · dim · u_r`。1つのループ（`RefineSystem` に対する `refine`）が Exact の `α`、Sgpr の重み、Svgp の三角 solve を扱う。各系は残差、保存済み因子での solve、`f64` へのやり直しを与える。反復改良は収束しないエラーを返さない。やり直しは常に `f64` の解。

**IR不収束時にjitterを増やさない**: 分解側のjitterだけを増やすと、前処理`LLᵀ`と目標`A`の乖離が拡大してIRが発散し得る。IR不収束は `f64` の解に落とす。jitter適応は§4.0の通りCholesky失敗時専用とする。

**位置づけ**: 理論的妥当性はあるが、実ワークロードでのパラメータ検証は今後の課題(§14)。

## 5. カーネル設計

### 5.1 Spec(宣言層)/ Evaluator(実行層)の分離、および精度ジェネリクス

`KernelSpec`(宣言層)は精度に依存しない表現とし、パラメータは常に`f64`で保持する(ユーザーが書く・読む値は精度非依存であるべきため)。`CompiledKernel<T>`(実行層)は`PrecisionPolicy::Storage`ごとにコンパイルされ、内部計算は`T`で行う。

最適化器は log-`θ` のフラットな `params: &[f64]` だけを見る。複合カーネルは、深さ優先・左から右の順でカーネルの葉へ対応づける。

```rust
pub struct ParameterBinding {
    pub index: usize,       // position in the concatenated kernel θ
    pub leaf_id: usize,     // leaf in depth-first, left-to-right order
    pub local_index: usize, // parameter inside that leaf
}

/// Declaration. Built-in leaves are stored directly; `+` / `*` build Sum / Product.
pub enum KernelSpec {
    Rbf(RbfKernel),
    RbfArd(RbfArdKernel),
    Matern(MaternKernel),
    MaternArd(MaternArdKernel),
    Periodic(PeriodicKernel),
    RationalQuadratic(RationalQuadraticKernel),
    RationalQuadraticArd(RationalQuadraticArdKernel),
    Constant(ConstantKernel),
    Linear(LinearKernel),
    White(WhiteKernel),
    Custom(CustomKernel), // a user KernelTerm, boxed for f64 and f32
    Sum(Box<KernelSpec>, Box<KernelSpec>),
    Product(Box<KernelSpec>, Box<KernelSpec>),
}

impl KernelSpec {
    pub fn custom<K>(term: K) -> Self; // K: KernelTerm<f64> + KernelTerm<f32> + Clone
    pub fn num_params(&self) -> usize;
    pub fn get_params(&self, out: &mut [f64]) -> Result<(), GprError>;
    pub fn set_params(&mut self, params: &[f64]) -> Result<(), GprError>; // atomic
    pub fn parameter_bindings(&self) -> Vec<ParameterBinding>;
    pub fn compile(&self) -> CompiledKernel<f64>;
    pub fn compile_as<T: KernelScalar>(&self) -> CompiledKernel<T>;
}

/// Evaluator. The same leaves as enum arms (static dispatch); Sum / Product
/// are flattened lists. Only `Custom` goes through a vtable.
pub enum CompiledKernel<T: KernelScalar = f64> {
    Rbf(RbfKernel),
    // … one arm per built-in leaf, as in KernelSpec …
    Custom(CustomKernel<T>),
    Sum(Vec<CompiledKernel<T>>),
    Product(Vec<CompiledKernel<T>>),
}

pub enum Triangle { Lower, Upper, Full }

/// A user leaf. It reads squared Euclidean distances (or coordinates for
/// `hess_points`) and writes the triangle `uplo` asks for.
pub trait KernelTerm<T: KernelScalar = f64>: Send + Sync + Debug + 'static {
    fn num_params(&self) -> usize;
    fn get_params(&self, out: &mut [f64]) -> Result<(), GprError>;
    fn set_params(&mut self, params: &[f64]) -> Result<(), GprError>;
    fn bounds_into(&self, out: &mut [Interval]) -> Result<(), GprError>;
    fn apply(&self, dist: MatRef<'_, T>, out: MatMut<'_, T>, uplo: Triangle) -> Result<(), GprError>;
    fn apply_cross(&self, dist: MatRef<'_, T>, out: MatMut<'_, T>) -> Result<(), GprError>;
    fn fill_diag(&self, out: &mut [T]) -> Result<(), GprError>;
    fn grad(&self, dist: MatRef<'_, T>, d_k: MatMut<'_, T>, param_idx: usize, uplo: Triangle)
        -> Result<(), GprError>;
    fn hess(&self, dist: MatRef<'_, T>, d2_k: MatMut<'_, T>, i: usize, j: usize, uplo: Triangle)
        -> Result<(), GprError>;
    fn hess_points(&self, x: MatRef<'_, T>, d2_k: MatMut<'_, T>, i: usize, j: usize, uplo: Triangle)
        -> Result<(), GprError>;
    /// 長方形の ∂K/∂θ、∂²K/∂θ∂θ（二乗距離から、train × test）。既定は `CoordGradientUnsupported`。
    fn grad_cross(&self, dist: MatRef<'_, T>, d_k: MatMut<'_, T>, param_idx: usize)
        -> Result<(), GprError> { Err(GprError::CoordGradientUnsupported) }
    fn hess_cross(&self, dist: MatRef<'_, T>, d2_k: MatMut<'_, T>, i: usize, j: usize)
        -> Result<(), GprError> { Err(GprError::CoordGradientUnsupported) }
    /// ∂k/∂(d²)、∂²k/∂(d²)²、∂²k/∂θ∂(d²)。k は二乗距離の関数なので、これで
    /// `FreeInducing` の座標微分が決まる。既定は `CoordGradientUnsupported`。
    fn grad_wrt_sq_dist(&self, dist: MatRef<'_, T>, out: MatMut<'_, T>) -> Result<(), GprError>
        { Err(GprError::CoordGradientUnsupported) }
    fn hess_wrt_sq_dist(&self, dist: MatRef<'_, T>, out: MatMut<'_, T>) -> Result<(), GprError>
        { Err(GprError::CoordGradientUnsupported) }
    fn grad_wrt_sq_dist_theta(&self, dist: MatRef<'_, T>, out: MatMut<'_, T>, param_idx: usize)
        -> Result<(), GprError> { Err(GprError::CoordGradientUnsupported) }
    fn clone_box(&self) -> Box<dyn KernelTerm<T>>;
    fn persist_id(&self) -> &'static str { "" }                          // registry key for save / load
    fn persist_state(&self) -> Result<serde_json::Value, GprError>;      // default: PersistFailed
}
```

`uplo` の既定は `Lower`。faer の `cholesky_in_place` は下三角しか読まないので、`Full` で埋めるとカーネル評価が約 2 倍になる。`CompiledKernel` も同じ操作（`apply`、`apply_cross`、`apply_points`、`apply_cross_points`、`fill_diag`、`fill_diag_points`、`grad`、`grad_points`、`grad_wrt_coord_dim`、`hess_wrt_coord_dims`、`hess_wrt_coord_mixed`、`hess_theta_coord_dim`、`hess`、`hess_points`）を持ち、どれも §8 の `KernelMath` についてジェネリック。

カーネルの葉のパラメータは f64 のまま。`compile()` は f64、`compile_as::<T>()` は同じ apply・勾配・ヘッセを f32 で与える。組み込みのカーネルの葉はどれも `T: KernelScalar` についての実装 1 つで、`CompiledKernel<T>` のディスパッチも 1 つ。SIMD の経路は `src/kernel/simd/` にある（§8）。等方の RBF・Periodic・RQ のレーン（`simd/stationary.rs`。正方の Gram、RBF の `∂K/∂θ`・長方形・座標からの長方形の `∂K/∂θ`、Periodic / RQ の重み付き勾配）はどの格納でも通る。`f64` と `f32` は同じレーンを通り、`f32` は `f64` に広げて計算し、書くときに 1 回だけ丸める。式ごとに実装は 1 つになる。f64 の Accurate の等方 RBF で、列の長さが 4 の倍数でない端はスカラーの `exp` で計算する。このレーンでは `FastApprox` は、スカラーの経路が使う `f32` の多項式ではなく `f64` の多項式で計算する。ARD のレーン（`simd/rbf_ard.rs`、`simd/ard.rs`）へは、`T = f64` のときだけ `f64` のビューを返すスカラーのフックから入る。`f32` の ARD は同じ式のスカラー。ユーザー定義のカーネルの葉は `impl<T: KernelScalar> KernelTerm<T>` 1 つ。`KernelScalar` は式に要る四則と `exp` / `ln` / `sqrt` / `powf` / `sin` / `cos` を持ち、ジェネリックな組み込みのカーネルの葉をそこから呼べる。`CustomKernel::new` は `KernelTerm<f64> + KernelTerm<f32>` を要求し、ジェネリックな impl 1 つでそれを満たす。

モデルはコンパイル済みの木を `KernelSpec` の隣に持ち、ハイパラを書いたあとにコンパイルし直す。

**Lengthscale**: 等方はスカラー `ℓ`（`θ=log(ℓ)`）。ARD は次元ごとの `ℓ_d`（`θ_d=log(ℓ_d)`、`ArdLengthscales`）で、RBF・Matérn・RQ にある。Periodic の lengthscale はスカラー。

ARD の二乗距離は `r² = Σ_d (x_d - x'_d)² / ℓ_d²`。全 `ℓ_d` が等しいとき等方に一致する。`∂K/∂θ_d` には次元ごとの差が必要で、等方の二乗距離行列だけでは足りない。ARD のカーネルの葉は座標か、§5.2 の生の `(Δx_d)²` キャッシュを読む。

ユーザー定義カーネル(`Custom`)はホットパスで新規アロケーションしないことを推奨するが、強制はしない(§2)。ユーザーカーネルには Workspace を渡さない。安全APIとunsafe高速APIの二系統は設けない。

ホットパス(距離・カーネル評価の二重ループ)では`CompiledKernel`を`match`で静的ディスパッチする。`Custom`だけvtable経由。これは§2の「静的ディスパッチを基本に、拡張点のみdyn」と一致させる。

最適化器のパラメータ配列は次の順で連結する:

```
[kernel_params | likelihood_params]
```

`FreeInducing` の `Sgpr` は、その後ろに列優先の誘導点座標 `Z` を足す（§6.1）。

### 5.2 距離キャッシュとキャッシュポリシー

訓練座標は fit 中に変わらないので、対距離は 1 回計算し、各 `θ` でカーネルを組み直すあいだ使い回す。

```rust
pub enum DistanceCachePolicy {
    Cached,   // default: fill once per fit, reuse (the speed pole)
    Uncached, // recompute from X on every kernel build (the memory pole)
}

/// Crate-private. What Cached stores (§7.1).
struct DistCache<S> {
    dist: Option<Mat<S>>,        // n×n squared Euclidean, for distance-mode leaves
    ard_sq_diff: Option<ArdSqDiffBuf<S>>, // raw (Δx_d)², packed lower triangles, for ARD leaves
}
```

置く中間表現は、二乗ユークリッド距離（等方の RBF / Matérn / RQ / Periodic / ユーザー定義のカーネルの葉）と、次元ごとの生の `(Δx_d)²`（ARD のカーネルの葉）。ℓ 込みの `r²` は置かない。ARD のレイアウトは、次元ごとに下三角（対角を含む）だけを列ごとに詰めたもの。値は `d · n(n+1)/2` 個で、次元 `k` は先頭から `k · n(n+1)/2` 個の後、列 `j` は行 `j..n` を連続して持つ。他の三角形を読む側は、`(i, j)` の代わりに `(j, i)` を読む。どちらの枠も最初に使うときに、コンパイル済みカーネルがそれを読むときだけ埋める。`RBF + White` と `Constant * RBF` は `dist` を埋める。単独の Linear / Constant / White は何も埋めず、方針は保つが使わない。訓練×クエリや LOO のキャッシュは無い。

ほかの方針とのどの組み合わせも不正ではないので、方針は実行時の enum にする（§6.3）。`(n,n,d)` テンソルは `n²×d×sizeof(T)` バイト。`K` 自体が `n²×sizeof(T)`（n=5000、f64 で約 200MB）で、ARD キャッシュはその `d` 倍になる。方針は呼び出し側が `Cached` か `Uncached` を選ぶ。`n`・`d`・メモリ予算からの自動選択は意図的に対象外。

### 5.3 合成カーネルの評価

`KernelSpec::compile` は結合則の効く連鎖を平らにする。`(A+B)+C` は `Sum(vec![A, B, C])` になり、積も同様。和の中の積（とその逆）は入れ子のまま残る。

コンパイル済みの木はそれぞれ座標モード（crate 内の `CoordMode`）を持つ。`Dist`（等方のカーネルの葉が二乗距離を読む）、`Points`（ARD と Linear のカーネルの葉が座標を読む）、`Either`（Constant と White は形だけ読む）、`Mixed`（Dist のカーネルの葉と Points のカーネルの葉の Sum / Product。例: `RBF + Linear`）。混ぜることは実行時エラーにも型での禁止にもしない。Dist のカーネルの葉は距離、Points のカーネルの葉は座標のまま評価する。

評価は平らにしたリストをたどる。Sum は最初の項を `out` に書き、後の項を出力と同じ形の `scratch` 1 枚を通して足す。Product も同様に掛ける。項そのものが複数項の Sum / Product のときは、入れ子 1 段ごとに出力と同じ形のバッファがもう 1 枚要る（`CompiledKernel::nested_depth`）。fit と predict の経路はその段を Workspace から借りる（§7.1）。段数は仮定せず木から数える。`(A*B)*(C*D)` は積 1 つに平らになり、`A*(B+C*D)` は `B+C*D` と `C*D` に 1 段ずつ要る。組み込みのカーネルの葉は `match` のアームで、`dyn KernelTerm` を呼ぶのは `Custom` だけ。

### 5.4 部分更新(コーディネート型最適化器)対応

fit が変えたカーネルの葉だけを作り直すかどうかは型にしない。実行時に `O::USES_CHANGE_INDICES && cholesky_buffer == CholeskyBuffer::Retain` から決める。Exact GPR では Cholesky が O(n³) のため、部分更新の恩恵はカーネル行列構築にだけ及ぶ。

```rust
pub trait Optimizer<P: ?Sized> {
    fn minimize(&self, objective: &mut P, init: &[f64]) -> Result<OptResult, GprError>;
    const USES_CHANGE_INDICES: bool = false; // FastSimulatedAnnealing sets true
}

pub trait Objective {
    fn num_params(&self) -> usize;
    fn value(&mut self, params: &[f64]) -> Result<f64, GprError>;
    fn value_at_changes(&mut self, params: &[f64], indices: &[usize]) -> Result<f64, GprError> {
        self.value(params) // default: full rebuild
    }
}

pub trait IncrementalObjective: Objective {
    fn value_with_changes(&mut self, params: &[f64], indices: &[usize]) -> Result<f64, GprError>;
}
```

`indices` には、最適化器の受理済みの点からではなく、その目的関数で直前に評価した `params` から変わった座標をすべて並べる。棄却のあと FSA は戻した座標と新しい座標の両方を渡す。作り直すカーネルの葉は index だけで決める。`GprObjective` はあわせて直前と新しい `θ` をビットで比べ（`f64::to_bits`）、`indices` に無い座標が変わっていたら、誤った値を黙って返さず `IndexOutOfRange` にする。空・重複・`i >= n_params` の index は境界で `GprError`。`ChangeSet` 型は無い。

`GprObjective` はどの最適化器・バッファでも `IncrementalObjective` を impl する。`value` / `value_at_changes` がカーネルの葉の経路を通るのは上のフラグが立つときだけで、それ以外は一括の値と勾配を計算する。`CholeskyBuffer::Reuse` は常に全体を作り直す。`with_optimizer` / `refit` は新しい最適化器からフラグを決め直す。`Gpr<Fixed>::factor` は一発フル。`Lbfgs` / `NelderMead` / `TrustRegion` は既定の `false`。FSA の初回とリスタートは `value`、座標一歩は `value_at_changes`。

カーネルの葉の作り直しはコンパイル済みのカーネルの葉をキャッシュし、変更 index が変えるカーネルの葉だけ `apply` し直す。カーネルの葉ごとの Gram（カーネルの葉 `L` 個で `L · n²`）、dirty の印、直前の `θ` は、1 回の `fit` / `refit` のあいだ `GprObjective`（crate 内の `LeafCache`）が持ち、Workspace には置かない。木の結合と **Cholesky は毎回フル**。ハイパラの変更は `K` の低ランク更新ではなく、faer の `rank_r_update_clobber` もそのためには使わない。

### 5.5 前処理パイプライン

Xとyを分ける。GPRでは平均関数を持たない場合、**yを平均0・分散1に標準化することが数値安定性の基本**になる。予測値は元スケールへ戻す。

```rust
/// Unfitted input map. `fit` consumes it and returns the fitted map, so an
/// unfitted `apply` cannot be written. `x` is column-major `n_rows × n_cols`.
pub trait UnfittedTransform: Send + Sync {
    fn fit(self: Box<Self>, x: &[f64], n_rows: usize, n_cols: usize)
        -> Result<Box<dyn Transform>, GprError>;
    fn clone_box(&self) -> Box<dyn UnfittedTransform>;
    fn as_any(&self) -> &dyn Any;
    fn persist_id(&self) -> Option<&'static str> { None }   // Some for a user map
    fn persist_state(&self) -> Result<serde_json::Value, GprError>;
}

/// Fitted input map. Every map is invertible.
pub trait Transform: Send + Sync {
    fn apply(&self, x: &mut [f64], n_rows: usize, n_cols: usize) -> Result<(), GprError>;
    fn inverse_apply(&self, x: &mut [f64], n_rows: usize, n_cols: usize) -> Result<(), GprError>;
    fn clone_box(&self) -> Box<dyn Transform>;
    // as_any / persist_id / persist_state as above
}

/// Unfitted target map, fitted the same way.
pub trait UnfittedTarget: Send + Sync {
    fn fit(self: Box<Self>, y: &[f64]) -> Result<Box<dyn TargetTransform>, GprError>;
    // clone_box / as_any / persist_id / persist_state
}

pub trait TargetTransform: Send + Sync {
    fn transform(&self, y: &mut [f64]) -> Result<(), GprError>;
    fn inverse_transform_mean(&self, mean: &mut [f64]) -> Result<(), GprError>;
    fn inverse_transform_variance(&self, var: &mut [f64]) -> Result<(), GprError>;
    fn inverse_transform_covariance(&self, cov: &mut [f64]) -> Result<(), GprError> {
        self.inverse_transform_variance(cov) // affine maps scale every entry by s²
    }
    // clone_box / as_any / persist_id / persist_state
}
```

組み込みの写像は未学習と学習済みの対になっている。`X` には `IdentityInput`、`StandardizeInput` / `FittedStandardizeInput`、`MinMaxInput` / `FittedMinMaxInput`、`ColumnwiseInput` / `FittedColumnwiseInput`、`Pipeline` / `FittedPipeline`。`y` には `IdentityTarget`、`StandardizeTarget` / `FittedStandardizeTarget`、`MinMaxTarget` / `FittedMinMaxTarget`、`TargetPipeline` / `FittedTargetPipeline`。変換はモデルの精度によらず `f64` で計算する。

既定の `Gpr` は両側とも Identity。平均関数が零のときは `StandardizeTarget` が数値安定の基本。`MinMaxInput` / `MinMaxTarget` は区間スケール（既定 `[0, 1]`）。複数の写像の直列は `Pipeline`（`X`）か `TargetPipeline`（`y`）。`with_input_transform` / `with_target_transform` は写像を 1 つ受ける。`ColumnwiseInput` は列ごとに写像を持つ（一様な列は MinMax、正規に近い列は Standardize）。長さが `d` でないときはエラー。`predict` は変換後の空間で潜在/観測分散を計算したあと、`inverse_transform_mean` / `inverse_transform_variance` を通してから返す。分散の逆変換はアフィン `y' = (y - a)/s` なら `Var(y) = s² Var(y')`。ユーザー定義の写像は `persist_id` / `persist_state` で保存し、`PersistRegistry` で戻す。

`Sgpr` / `Svgp` も同じ変換を同じ既定（Identity）で受ける。誘導点 `Z` は `X` と同じ座標で渡し、`X` と一緒に学習時の入力の写像を通す。クエリと、`OnlineSgpr` に足す点は、学習時に当てはめた写像を通す。`FreeInducing` は写像後の座標で `Z` を探す。学習後のモデルは `Transform::inverse_apply` で `Z` を元の座標に戻して返す。

### 5.6 与えられた二乗距離: 要件（#470）

カーネルの葉は、座標の代わりに、呼び出し側が与えた二乗距離（測地距離、グラフ距離、別の場所で計算した距離）を読める。この節は、実装の前に要件を決める。最初の試み（#476〜#479、閉じた）は、動かすことを先にして後から測ったため、距離の経路が座標の経路より重くなった。同じことを繰り返さないよう、要件とその確認を先に置く。

**性能。** 比較の基準は、同じ `n`・`d`・カーネル・精度の座標モデル（`DistanceCachePolicy::Cached`）、つまり `X` から距離を計算して持つモデルである。距離を与えることは仕事を減らすことであり、増やしてはならない。どの操作でも、距離モデルの時間と確保は基準を超えてはならない。

| 操作 | 要件 |
| --- | --- |
| `fit` / `factor` | 時間と確保数が基準以下。クレートが持つメモリのピークが、基準の距離キャッシュ（スカラーのスロットは密な `n²`、ARD のスロットは詰めた `d · n(n+1)/2`、5.2 節）以下。呼び出し側が持つ表は数えない。`f64` のモデルに所有ごと渡した ARD の密な表（`from_vecs`、または呼び出し側の求めでコピーする `from_slices`、`d · n²`）も数えない。この表はその場で検査して、詰めた複製を作らずにそのまま持つので、fit は何もコピーしない。ARD の `fill` は、使い回すバッファへ列の塊ごとに書かせて下三角へ直接詰め、密な `d · n²` のバッファは作らない |
| Sgpr / Svgp の `fit` / `factor`（#470、D1-5） | 確保数が基準以下。時間は、座標のモデルにはない、供給された `d · n · m` 個の値の読み出しを除いて基準以下。読み出しとは、`n × m` のブロックの取り込み（1 回の検査と、借用したブロックの 1 回のコピー。所有ごと渡したブロックはそのまま持つ）と、カーネルが `K(Z, X)` を評価するときのブロックの読み出しである。座標のモデルは同じ組を `n · d` 個の座標から計算するので、ブロックをどう並べても読む値はこれより減らない。`d` 枚のブロックを持つ ARD のスロットは `d` 倍読む。そこで読み出しのほうに上限を置く。取り込みでは各ブロックを 1 回なめ、`K(Z, X)` の評価ごとに 1 回なめて転置を 1 回だけ行う。ブロックを読む経路は、座標の経路が同じ組を計算するより遅くない。残り（`K_mm`、解く処理、`B`）は基準以下 |
| `OnlineSgpr` の `insert` / `delete` / `insert_inducing` / `delete_inducing`（#493、D1-5a） | 確保数が基準（同じ問題の座標の `OnlineSgpr`）以下。時間は、座標のモデルにはない、供給された値の書き込みと移動を除いて基準以下。insert は `d · m` 個の値を書く（各ブロックの列ごとに 1 個。ブロックが持つ余白に書き、満杯なら行を 4 分の 1 増やす）。delete はその点より後ろの行、`d · (n − i) · m` 個の値をその場で移す。`insert_inducing` は列を 1 本書き、`delete_inducing` はその列より後ろの列を移す（ブロックが持つ余白に書き、満杯なら列を 4 分の 1 増やす）。そこでこれらに上限を置く。書く値は 1 回ずつ書き、移す値は 1 回ずつ移す。更新のためにブロックをコピーしない（失敗した更新は、取っておいた少しの値から自分の変更を戻す）。組み立て（誘導点の変更のあとの組み立て直しと、`f64` で精緻化する精度が変更のたびに作り直す `f64` の重み）は、Sgpr の factor の行と同じくブロックを読む。`n = 512`、`m = 64`、`d = 4` で測ると、スカラーのスロットは 4 つとも基準以下。ARD のスロットは、insert が約 `1.5 µs`、中央の点の delete が約 `40 µs`、基準より長い。差は上の移動による |
| ある `θ` での `mll`・勾配・Hessian（最適化の 1 ステップ） | ワークスペースを作った後は確保しない（基準と同じ）。時間は基準以下 |
| `predict_into`（Exact、Online、Sgpr、Svgp） | 同じ形でのウォームアップの後は確保しない。時間は基準以下。供給は新しい `Vec` に集めずに受ける。Sparse のモデルは、誘導点からクエリへの `m × q` のブロックを受け取り、Exact のモデルが `n × q` を読むのと同じくその場で読む |
| 容量内の `insert` | 確保と時間が基準の insert 以下。新しい点の二乗距離をその場で書く。scalar の正方行列はその列と鏡像の行（`2n` 個）、ARD のスロットは次元ごとに連続した 1 本（`d · n` 個）（§11） |
| `delete` | 確保と時間が基準の delete 以下。格納した二乗距離をその場で詰める。添字 `i` に対して、scalar のスロットは `O((n − i) · n)` 個、ARD のスロットは `O(d · (n² − i²) / 2)` 個を動かす。ワーカーを起こすのに見合う量の詰め直しは、因子の更新と並べて走らせる（§11） |
| `refit` / `set_params` | 時間が基準以下。挿入や削除で行の並び（§11）になった ARD のスロットは、その並びの順に読むので、Gram・勾配・ヘッセ行列の費用は列の並びと変わらない |
| `borrow` | `f64` のモデルは、借りた表をコピーせずにその場で読む |

**受け入れの確認（実装より先に書く）。** `benches/` は、上の各操作を、同じ問題で「座標・スカラーのスロット・ARD のスロット」に並べて測る。実装の各 PR は、変更前後の数値を貼る。`tests/alloc.rs` は、操作ごとに、距離の経路の確保数が座標の経路以下であること（相対の確認）と、測った数（絶対の上限、ラチェット）を固定する。要件を満たさない PR はマージしない。

**表の検査。** 学習の正方行列は対称で対角が 0、すべての値が有限で非負でなければならない。表を黙って直すことはしない。表だけからは、丸めと誤った表を区別できない。`‖a‖² + ‖b‖² − 2a·b` の誤差は `ε · (‖x_i‖² + ‖x_j‖²)` で、点の原点からの遠さで決まり、距離では決まらない。そのため、表から読む許容量では両者を分けられず、黙って直せば有向距離や転置した表まで受け付けてしまう。組ごとに計算した表（`(a − b)²` を両方の順で）は、完全に対称で対角も 0 なので、そのまま通る。

- 既定の検査は厳密である。違反は、検査が表を読む順で最初に失敗した組（並列の帯で検査する正方行列では、最初に失敗した帯の組）とその値を示すエラーになる。修復の許容誤差で判定するフィルは、正方行列が揃ってからしか判定できないので、最悪の組を示す。
- 丸めの出る作り方（Gram trick）で表を作る呼び出し側は、その供給に、自分で選んだ許容量で修正を指定する。許容量以内なら負の値と対角は `0.0` に、鏡像の組はその平均にする。超えれば表を断る。
- 不正な値は専用のエラーの variant（slot、ARD の次元、組、理由）にし、`ShapeMismatch` は形のためだけに残す。スロットの欠落・重複・未知も専用の variant（`DistanceSlot`、種類は `SlotErrorKind`）にする。
- 検査は方針によらず、正方行列ごとに表を 1 回読む（`O(n²)`）。ARD の訓練の正方行列は、三角へ詰めながら検査する。ARD の予測ブロック（訓練 × クエリ）は、カーネルが `r²` を足すループで読みながら検査するので、呼び出し側の値を読むのは 1 回だけになる（`f64` のモデルはカーネルが、`f32` のモデルは型変換が読む）。ブロックは検査済みかどうかを型（`Checked` / `Unchecked`）で持つ。未検査のブロックの値は検査つきの読み出しからしか取り出せないので、`Unchecked` のブロックを読む葉や SIMD の経路は、検査を飛ばせない。どちらの状態でブロックを作るかは束ねる側のコードが決める（`ArdBlocks::new` はどちらも作れる）。`Checked` のブロックを作るのは束ねるコードだけで、検査した値、型変換した値、詰めた値から作る。スカラーの予測ブロックは束ねるときに、CPU で使える最も広い SIMD（`pulp` の実行時の切り替え）で検査する。

**供給は型で表す。** 木の `Supply` の種類が、評価で読むものを決める。座標の木（`NoSupply`）は供給を持たない。ビューは `()` を持つので、経路には供給の引数も、それによる分岐もない。供給した距離の葉を持つ木は、すべてのスロットの供給を必須の引数として受け取る。供給した距離の葉は、同じ形状のスロットの中での自分のスロットの番号を持つ。番号は木を組んだときに一度だけ振る。どの供給（訓練の保存、束ねた予測、列の範囲）もスロットをその順に持つので、その木のために束ねた供給は、葉が読む番号をすべて持つ。別のカーネルのために束ねた供給を渡すと、引いた時点で `UnsupportedKernelOperation` を返し、範囲外を読むことはない。

**保存と読み込み。** 供給された距離のモデルは、持っている学習の `d²` を保存する。だから読み込んだモデルは、呼び出し側が学習の正方行列をもう一度渡さなくても予測できる。ディスク上の並びは、ストアの並びによらず正規の形にする。Exact の slot は下三角を列ごとに詰めたもの（`n(n+1)/2` 個、ARD の slot はそれを `dims` 個、次元を順に）で、online のモデルの行の並びも fit の詰めた三角と同じ順で書く。`f32` のモデルは `f32` で書く。`MixedPrecision` のモデルは呼び出し側の正確な `f64` の値を書き、読み込みでもう一度丸める。Sparse のモデルは `n × m` のブロックを `f64` で持ち、そのまま書き、誘導点の添字も書く。カーネルの JSON は slot の表（形と次元、`DistanceKernel::slots` の順）を持ち、葉は表での位置で slot を指す。読み込みは新しい slot を作るので、保存前のハンドルはそのどれも指さない。新しいものは読み込んだモデルの `slots()` / `to_kernel()` で得る。読み込みの型は座標のものと分ける（`LoadedGpr` と同じ 8 variant の `LoadedDistanceGpr<C>`、`LoadedDistanceSgpr<C>`、`LoadedDistanceSvgp<C>`）。マーカー `C` で型が決まるので、`LoadedGpr` / `LoadedSgpr` / `LoadedSvgp` は 0.1.0 のまま。種類やマーカーの違うファイルは `WrongModel`。座標のファイルには新しいキーが無いので、`FORMAT_VERSION` は 1 のまま。保存したディレクトリは入力として扱う。読み込みは、fit が検査するもの（`d²` の値、誘導点の添字が範囲内で重複しないこと）と、fit なら作り方から成り立つものを検査する。カーネルが表の slot を表の順に読むこと、`DistanceOnly` のカーネルが座標を読む葉を持たないこと、疎の `WithPoints` のモデルの `z` と `z_train` が誘導点の添字の指す行であること、保存した変換が学習データを有限に保つこと。違反はエラーで、panic にも、自分と食い違うモデルにもならない。

**最初の試みから、測ってから流用するもの。** 型の層（`DistanceKernel<C>`、`KernelSpec<S>` / `CompiledKernel<T, S>` の封印した `Supply` の種類、`ModelKernel`）と保存形式は、上のベンチで要件を満たすと分かれば流用する。

## 6. GPModel抽象化(厳密/疎の差し替え)

学習と推論は型で分ける。未学習の `predict` は公開 API に置かない。sklearn の同一オブジェクト `fit` / `predict` は数値照合の対象であり、公開面の契約ではない。fit の目的関数は `fit` / `refit` のあいだだけモデルを借り、学習済み値には残らない。

```rust
impl<O, P: GpScalar> Gpr<O, P> where O: for<'a> Optimizer<GprObjective<'a, P>> {
    pub fn fit(self, x: &[f64], n_rows: usize, n_cols: usize, y: &[f64])
        -> Result<FittedGpr<O, P>, (Self, GprError)>;
}
impl<P: GpScalar> Gpr<Fixed, P> {
    pub fn factor(self, x: &[f64], n_rows: usize, n_cols: usize, y: &[f64])
        -> Result<FittedGpr<Fixed, P>, (Self, GprError)>;
}

/// Fitted: L, α, X, kernel, likelihood, transforms. No W and no optimizer state.
impl<O, P: GpScalar> FittedGpr<O, P> {
    pub fn predict(&self, xs: &[f64], n_rows: usize, n_cols: usize)
        -> Result<Prediction<P::Refine>, GprError>;
    pub fn predict_into(&mut self, xs: &[f64], n_rows: usize, n_cols: usize,
        out: &mut Prediction<P::Refine>) -> Result<(), GprError>;
    pub fn predict_covariance(&self, xs: &[f64], n_rows: usize, n_cols: usize)
        -> Result<PredictiveCovariance<P::Refine>, GprError>;
    pub fn sample(&self, xs: &[f64], n_rows: usize, n_cols: usize, n_draws: usize, seed: u64)
        -> Result<Vec<P::Refine>, GprError>;
    pub fn loo_predict(&self) -> Result<Prediction<P::Refine>, GprError>;
    pub fn neg_log_marginal_likelihood(&self) -> Result<f64, GprError>;
    pub fn refit(&mut self) -> Result<(), GprError>; // re-optimize (O) or refactor (Fixed) on the same data
    // *_with variants take PredictOptions; get_params / set_params /
    // value_and_gradient_into / hessian_into; into_online / into_trainer / save
}
```

すべての `seed`（sample、最適化のリスタート、FSA、SVGP の Adam のシャッフル）は、gprx 自身の Xoshiro256++（状態を SplitMix64 で seed から作る、`src/rng.rs`）を始める。同じ seed と同じ gprx の版なら、どのプラットフォームでも、どの `rand` の版でも同じ乱数列になる。最初の出力はテストで固定する。乱数列を変えるのは破壊的変更。

```rust
pub enum VarianceKind {
    Latent,      // variance of the latent f* (no noise)
    Observation, // variance of the observation y* (includes σn²). Default
}

pub struct Prediction<T = f64> {
    pub mean: Vec<T>,
    pub variance: Vec<T>,
    pub variance_kind: VarianceKind,
}

pub struct PredictiveCovariance<T = f64> {
    pub mean: Vec<T>,
    pub covariance: Vec<T>, // column-major m × m; its diagonal is Prediction::variance
    pub variance_kind: VarianceKind,
}

pub struct PredictOptions {
    pub variance_kind: VarianceKind, // default Observation
}
```

`Gpr`（Exact）、`Sgpr`、`Svgp` がそれぞれ学習済み型を返す。ハイパラ最適化は目的関数の trait（§9）を介して `fit` / `refit` のあいだだけ扱う。

`predict` は対角の分散を返す。クエリ間の共分散と事後分布からの標本は別の呼び出し（`predict_covariance`、`sample`）で、`predict` のフラグにはしない。`*_with` の版は `PredictOptions` を受け、そうでない版は `Observation` を使う（ユーザーが欲しいのは多くの場合ノイズ込みの予測分散）。`sample` は長さ `m` の標本 `n_draws` 本を 1 つの `Vec`（`n_draws × m` 個）で返す。

### 6.1 Sparse GPRの誘導点キャッシュ問題

Sparse 近似は VFE。理由は [ADR 0002](adr/0002-sparse-vfe.md)。FITC は載らない。SVGP は別公開型（`Svgp` / `FittedSvgp`）。理由は [ADR 0006](adr/0006-sparse-svgp.md)。`Svgp<Fixed>::factor` が呼び出し側の `Z` で `K_mm` を LLT し、whitened の `q(u)` を prior（平均 0、`L = I`）で置く。`Svgp<Adam>::fit` が同じ prior からミニバッチ Adam でカーネル `θ`・尤度 `θ`・whitened `q` を動かす。1 ステップは、自分の点の `A_b = L⁻¹ K(Z, X_b)`・`k_diag`・`∂K(Z, X_b)/∂θ` だけを作り、`K_mm` を分解し直すので、計算量は `O(b (m² + m d) + m³)` で、`n` に依存しない。カーネルの勾配は VFE と同じく逆向きに作る。`∂K_mm` と `∂K(Z, X_b)` への重みを 1 ステップに 1 回 `O(m² b)` で作り、各カーネルパラメータは `O(m² + m b)` の縮約にする（パラメータごとの解は行わない）。全 `n` 点の `A` と `k_diag` は、最初のステップの前と最後のステップの後に 1 回ずつ作る。勾配は、格納の精度によらず `f64` で計算する。`Adam` は `Optimizer` ではない。`FittedSvgp` は対角の `predict` / `predict_with`（と `predict_into` / `predict_with_into`）、`neg_elbo`、全データ `value_and_gradient_into` を返す。最適 `q`（Titsias）では同じ `θ`・`X`・`Z` の `FittedSgpr` と一致する。公開型は `Sgpr` / `FittedSgpr`。既定は `Sgpr<Lbfgs, FixedInducing>`。`fit` がカーネルと尤度の `θ` を探し、`Sgpr<Fixed, I>::factor` が呼び出し側の誘導点 `Z` で `K_mm = k(Z, Z)` を LLT する。既定では `Z` は params に入らない。`with_inducing(FreeInducing)` の `fit` はカーネル `θ`・尤度 `θ`・列優先 `Z` を同じ `Optimizer` が同時に動かす。`FittedSgpr` は対角の `predict` / `predict_with`（と `predict_into` / `predict_with_into`）、`neg_log_marginal_likelihood`（VFE の負の ELBO）、`value_and_gradient_into`、`hessian_into`（row-major `p×p`）を返す。White のカーネルの葉を含まないカーネルでは、`Z = X` のとき Exact の `Gpr<Fixed>::factor` と一致する。`K(Z, X)` は `Z` と `X` の値によらず常に長方形の相互共分散なので、White のカーネルの葉はそこに何も足さない（`K_mm` と `diag K(X, X)` には足す）。そのため境界は `Z` について連続で、`Z` と `X` のビット一致や行の順序に依らず、値・勾配・Hessian はすべて同じ VFE の式から出る。White のカーネルの葉を含むと、`Z = X` の境界は Exact の尤度にならない（White は誘導点が説明しないノイズとして扱う）。k-means は置かない。外部照合は §12（5c、5d）、時間と RSS は §15。

`K(X,X)`対角は不変なので1回計算・流用。`K(X,Z)`, `K(Z,Z)`はZが動くたびに再計算が必要だが、m(誘導点数)が小さいためCholeskyのO(nm²)に対して無視できるコストであり、キャッシュ対象にせず毎回再計算する。joint の `K(X,X)` 勾配とヘッセは対角 `∂k(x_i, x_i)/∂θ` を `O(n)` で足す。`K(Z,Z)` と `K(Z,X)` の勾配は密行列のまま。

誘導点座標の勾配は`grad_wrt_coord_dim`、Hessian は`hess_wrt_coord_dims` / `hess_wrt_coord_mixed` / `hess_theta_coord_dim`(§5.1)で扱う。組み込みのすべてのカーネルの葉が持ち、Sum と Product の木が合成する（放射状のカーネルの葉は `k = g(q)`、`q = Σ w_d Δ_d²` の `g'(q)`、`g''(q)` から 1 つの実装で、Product は各項の値・1 階・2 階への積の規則で）。`ν = 1/2` の Matérn は panic ではなく`GprError::CoordGradientUnsupported`を返す。2 点が一致するところで座標微分が定義できず、`Z ⊂ X` の初期化がそこから始まるため。`Custom` のカーネルの葉は `grad_wrt_sq_dist`、`hess_wrt_sq_dist`、`grad_wrt_sq_dist_theta` で座標微分を、`grad_cross` / `hess_cross` で長方形の `∂K/∂θ` を与える。既定のまま残したカーネルの葉は`CoordGradientUnsupported`を返す。既定の `FixedInducing` の `fit` はこの API を使わない。使うのは長方形の `∂K(Z, X)/∂θ` と `∂²K(Z, X)/∂θ∂θ`（`grad_cross_points` / `hess_cross_points`）で、組み込みのすべてのカーネルの葉と、その Sum / Product の木が持つ。そのため `Constant × RBF` の信号分散を、Exact と同じく `Sgpr` と `Svgp` で学習できる。`Custom` のカーネルの葉は `KernelTerm::grad_cross` / `hess_cross` が要る（上記）。`FreeInducing` は同時最適化で次元一括で座標 API を呼ぶ。VFE の勾配は逆向きに作る。境界は `A = L⁻¹ K_mn` について `⟨G, dA⟩`（`G = B⁻¹A − (w rᵀ + A)/σ²`、`B = σ² I + A Aᵀ`、`w = B⁻¹ A y`、`r = y − Aᵀ w`）で動くので、重み `w_mn = L⁻ᵀ G`、`w_mm = −sym(L⁻ᵀ tril½(G Aᵀ) L⁻¹)`、`Σ diag K` に `1/(2σ²)` を `O(m² n)` で 1 回作る。各カーネルパラメータは自分の `∂K_mm`・`∂K_mn`・`∂ diag K` をそれと `O(m n)` で縮約し、誘導点の各座標 `z_p[dim]` は `p` の行と列だけを読む（次元ごとの座標微分から `O(m + n)`）。方向ごとの `O(m² n)` の解は行わない。理由は [ADR 0003](adr/0003-sparse-z-joint.md)。

**既定は呼び出し側が Z を渡し、最適化対象はカーネルハイパラとノイズのみとする。** 自由 Z は `FixedInducing` / `FreeInducing` で切り替え、カーネル `θ`・尤度 `θ`・列優先 `Z` を同じ `Optimizer` が同時に動かす。区間は学習データの範囲を少し開いて広げた生座標。L-BFGS 履歴の長さは `p = p_θ + m×d` で、増分は `history_size × m × d` 個の `f64`（`m` が小さいので VFE の `O(nm²)` に対して小さい）。交互は載らない。

供給された二乗距離（§5.6、D1-5）では、`Sgpr` と `Svgp` は `Gpr` と同じく `DistanceKernel` を受け取る。誘導点は添字で指定する学習点（`inducing`、重複なし）で、`fit` / `factor` はスロットごとに学習点からそれらへの `n × m` のブロックを受け取る。`K_mm` はそのブロックの誘導点の行を読む（学習の正方行列と同じく検査する）。モデルはブロックを渡されたまま `n × m` で持つ。`K(Z, X)`・その微分・勾配の縮約は、ブロックを `K(X, Z)` として読み、結果か重みを 1 回だけ転置する。転置には、供給の経路がほかに使わないバッファを使う。ミニバッチのステップは自分の行を集める。予測は誘導点からクエリへの `m × q` のブロックを、共分散は `q × q` の正方行列を受け取る。`FreeInducing` は座標を動かすので距離のカーネルには無い。

距離のカーネルの `FittedSgpr::into_online` は、ブロックを持ったままの `OnlineSgpr` を返す（D1-5a、#493）。`insert` はスロットごとに、誘導点から新しい点への `m × 1` の二乗距離（`WithPoints` ならその座標も）を受け取る。全体を検査し、その点の行として持つ。`insert_inducing` は学習点を `PointId` で指定し、学習点からその点への `n × 1` の二乗距離を受け取る。誘導点どうしの新しい `(m + 1)²` の正方行列は、学習の正方行列と同じく検査する（自分自身は 0、各組は持っている値と等しい。tidy のソースなら両方を平均に直す）。そのあと系を組み立て直す。誘導点は学習点のままでなければならない。誘導点である点の `delete` は、`delete_inducing` で外すまで `InvalidConfig` になる。ブロックは行と列の余白を持ち、その場で変わる。失敗した更新は自分の変更を戻すので、`f64` で精緻化する精度でも、更新を戻すためにブロックをコピーしない。`set_params`・勾配・Hessian・`refit` のためのモデル自身のコピーは、ブロックを共有する。

オンラインは X と誘導点を増減できる。`FittedSgpr::into_online` が `OnlineSgpr<O>` を返す（誘導 typestate は無い）。`insert` / `delete` は ADR 0004 の rank-1 で VFE 因子を更新する。`insert_inducing` / `delete_inducing` は [ADR 0005](adr/0005-sparse-inducing-update.md)（insert は bordered LLT、delete は trailing cholupdate）。識別子は `InducingId`。座標は呼び出し側。`Z` は params に入らない。`set_params` と `refit` はフル再 assemble。

**Exact との機能差。** Sparse のモデルは、次の機能で `Gpr` に揃える。

| 機能 | `FittedSgpr` / `OnlineSgpr` | `FittedSvgp` |
| --- | --- | --- |
| 入力 / 目的変数の変換 | あり | あり |
| `K_mm` の `JitterPolicy`（既定は `adaptive(1e-8, 10, 5, 1e-3)`。`K_mm` にはノイズが入らないため、§4.0） | あり | あり |
| warmup 後に確保しない `predict_into` | あり | あり |
| 予測共分散と posterior sample | あり | あり |
| LOO | あり | なし |
| 保存・読み込み | あり | あり |

Sparse の `predict_covariance` / `sample` は `predict` の経路を通し、そのバッファから非対角を作る。VFE は `K** − A*ᵀA* + σn² S*ᵀS*`（`A* = L_mm⁻¹ K_m*`、`S* = L_B⁻¹ A*`）、SVGP は `K** − AᵀA + UᵀU`（`U = L_qᵀ A`）。対角は `predict` の分散とビットで一致する。標本は Exact と同じ引き方（`μ + L z`、因子にはモデルの `JitterPolicy`）。

SVGP は LOO を持たない。collapsed VFE の `q(u)` は閉じた形の最適解なので、点 `i` を除くのは `A = K_mm + σ⁻² K_mn K_nm` の rank-1 の downdate になる（θ と `Z` を固定して 1 点 `O(m²)`）。SVGP の `q(u)` は、ミニバッチの Adam が全点に合わせた変分パラメータになっている。点を除くには `q(u)` を合わせ直す必要があり、閉じた形は無い。学習済みの `q(u)` をそのまま使っても LOO の予測にはならないので、その名前では出さない。

### 6.2 `Gpr` のMLLと勾配

ハイパーパラメータ勾配のアルゴリズムと必要なメモリが無いと、勾配ループで一時行列を確保してアロケーション方針に違反するか、パラメータごとに線形ソルブを繰り返してO(p n³)になる。

負の周辺対数尤度(最小化対象):

```
L(θ) = ½ yᵀ K⁻¹ y + ½ log|K| + (n/2) log(2π)
∂L/∂θ_i = -½ αᵀ (∂K/∂θ_i) α + ½ Tr(K⁻¹ ∂K/∂θ_i)
        = -½ ⟨W, ∂K/∂θ_i⟩_F
ただし α = K⁻¹ y、W = ααᵀ - K⁻¹
```

`(n/2) log(2π)` は θ に依らず、足しても測れるほどのコストにならないので、公開の NLML と最適化の目的関数は同じ `L(θ)` にする。API は分けない。

標準アルゴリズム(Rasmussen & Williams / GPy系):

1. `k_matrix`に `A = K + σn² I` を構築(下三角のみ、§5.1の`uplo=Lower`)
2. in-place Cholesky。`k_matrix`はLになる
3. `log|K| = 2 Σ log(L_ii)` をLの対角から計算
4. `L Lᵀ α = y` を前進・後退代入で解く(O(n²))
5. `L`から`K⁻¹`を計算する(三角ソルブで `L Lᵀ X = I`、O(n³)が1回)
6. `W[i,j] ← α[i] α[j] - K⁻¹[i,j]`(対称なので下三角のみ)
7. カーネルの全θ_iの `⟨W, ∂K/∂θ_i⟩_F` を、カーネルの木を 1 回たどって積算する（`CompiledKernel::weighted_grads`）。和は重みをそのまま各項へ渡す。積は因子 `c` へ重み `W ∘ ∏_{s≠c} K_s` を渡す。Constant の因子はスカラーで、重みを定数倍するだけ。その微分 `⟨W, K_積⟩` は別の因子の走査が返す値を使うので、定数以外の因子が 1 つの積（`C × RBF`）は Gram を作らない。定数以外の因子が 2 つ以上の積は、分解のときに残した因子の Gram（`CompiledKernel::eval_gram_keeping`。`weighted` の先頭のバッファ）を読む。残すと、走査ですべての因子を評価するより `n×n` のバッファが増えるときだけ、残さずに評価する（`CompiledKernel::kept_products`）。どの積が Gram を残し、それがバッファのどこにあるかは 1 か所（`CompiledKernel::keep_plan`）で決め、バッファ数の見積もり、分解の段、木の走査のすべてがそれを読む。分解の段はその `θ` で残した積の数を返し、勾配の走査はその値を受け取る。そのため、残した Gram を読むのは、それを書いた分解の直後だけになる。カーネルの葉は自分の `∂K/∂θ_i` を使い回す `n×n` の枠へ書き、受け取った重みとの Frobenius 積を取る（パラメータごとに O(n²)）。距離をキャッシュした Periodic と RQ のカーネルの葉は、組を 1 回たどって全部の積を直接足す。距離をキャッシュした `Accurate` の RBF のカーネルの葉も、自分の Gram を渡されたときか `⟨W, K⟩` を求められたときは同じようにする。自分の Gram を渡されたときは、その値から `∂K` を作り（RBF は `k s / ℓ²`、Periodic は `Accurate` で `k`・`sin`・`cos` から、RQ は `k` と `ln u` から）、`exp` / `pow` を計算し直さない。Constant のカーネルの葉は `c · Σ 重み`。積の因子をパラメータごとに評価し直すことはない。ノイズは`GaussianLikelihood::noise_grad_diag`(対角のみ)

全体コストはO(n³ + p n²)。K⁻¹をパラメータごとに作り直さない。

Sgpr と Svgp のカーネルパラメータの勾配も、この走査を `K(Z, Z)` に使う。矩形の全体 `K(Z, X)` と、対角の和 `Σ_i ∂k(x_i, x_i)/∂θ` も、同じ木の走査で足す。積の他の因子は、この 3 つの縮約のそれぞれで 1 回だけ評価する。VFE のヘッセは、因子の接線が行列そのものを要るので、各 `∂K` を作る。

解析 NLML ヘッセ:

```
H_ij = -½ ⟨W, ∂²K/∂θ_i∂θ_j⟩ - ½ Tr(K⁻¹ K_i K⁻¹ K_j) + αᵀ K_i K⁻¹ K_j α
```

`KernelTerm::hess` / `hess_points` が `(i, j)` 1 組の `∂²K` を書く。Custom・Sum/Product も解析。`FittedGpr::hessian_into` が公開の入口で、`GprObjective` は `TwiceDifferentiable` へ転送する。一次の項は、ノイズ（`A_i = σn² I`）を含む各パラメータ `i` について 1 回ずつ計算する。`A = L Lᵀ` として、`S_i = L⁻¹ A_i L⁻ᵀ`（三角解 2 回）と `v_i = L⁻¹ A_i α` から `Tr(A⁻¹ A_i A⁻¹ A_j) = ⟨S_i, S_j⟩_F`、`αᵀ A_i A⁻¹ A_j α = v_iᵀ v_j` を得るので、Hessian 全体は `O(p n³ + p² n²)`。`p` 枚の `S_i`（`p · n²`）、`v_i` を並べた `n × p`、長さ n のベクトル 1 本は `WorkspaceCore::hessian` に置く。最初の Hessian まで空で、以後は使い回すので、2 回目以降の Hessian は確保しない。`CholeskyBuffer::Reuse` は ⟨W, K_ij⟩ のあと Chol し直す（`W` が `L` を上書きしたため）。

`value_and_gradient_into`はこの手順を一度で実行し、Lとαと`exp_buf`を尤度・勾配で共有する。デフォルト実装の`value`→`gradient_into`の二段呼びでは共有されない。

既定の `CholeskyBuffer` は `Retain`。専用の `w_matrix` に `K⁻¹` → `W` を書き、`L` は `k_matrix` に残す。速さは変えない。公開のメモリ優先（`with_prefer_memory`）が `CholeskyBuffer::Reuse` を、`DistanceCachePolicy::Uncached` と一緒に選ぶ。`Reuse` は `K⁻¹` を `exp_buf` で解き、`W` を Cholesky 領域へ書く。最適化ループの途中では `L` を戻さない。`fit` の末と単独の `value_and_gradient_into` の末で Cholesky し直す。persist にこの方針は書かない。`load` は `Retain`。

### 6.3 Exact GPR (`Gpr` / `FittedGpr`)

公開面はトレーナーと学習済みモデルを分ける。

`Gpr<O = Lbfgs, P = DoublePrecision>` は `KernelSpec`・`GaussianLikelihood`・変換と、最適化器 `O`、実行時の方針 4 つを持つ：`DistanceCachePolicy { Cached, Uncached }`、`CholeskyBuffer { Retain, Reuse }`、`KernelExp { Accurate, FastApprox }`、`JitterPolicy`（§4.0）。どれも素の enum。不正な組み合わせが無いので型パラメータにしない。`Gpr::new` の既定は速さ優先（`Cached` + `Retain`）で `Accurate`。外部クレートが距離キャッシュと Cholesky バッファを置く入口は `with_prefer_memory` / `with_prefer_speed` だけである。呼ぶたびに両方を置き換える。メモリ優先は `Uncached` + `Reuse`。`with_math` / `with_jitter_policy` はその方針を設定する。対距離を読まないカーネル（単独の Linear / Constant / White）は方針によらず距離キャッシュを確保しない。`from_points` は無い。`FittedGpr` に `with_prefer_*` は無い（`into_trainer` → prefer → `refit`）。方針は getter で読める。型は crate ルートに残す。`Gpr<O: Optimizer>::fit(self, …)` が `O` でハイパラを動かし、成功時に `FittedGpr<O, P>` を返す。固定ハイパラは `Gpr<Fixed>::factor`。`optimize: bool` は置かない。失敗時は消費した `Gpr<O, P>` をエラーと一緒に返す。`fitted: bool` と `GprError::NotFitted` は置かない。未学習の `transform` / `apply` は型で起きない（`StandardizeTarget::fit(self)` が `FittedStandardizeTarget` を返す）。`FittedGpr` の `L` / `α` / `X` / コンパイル済みカーネルは `Option` にしないので、欠けた部品に出会う呼び出しは無い。

`FittedGpr` は推論に必要な `L`・`α`・訓練 `X`・カーネル・尤度・変換を持つ。勾配用の `W`・`∂K`・argmin 状態は `fit` のあいだだけ残り、学習済み値には残さない。同一プロセスで `fit` の直後に `predict` する経路は少数派とみなす。学習済みモデルを渡すのが主経路なので、推論オブジェクトは `FittedGpr` である。

`FittedGpr` と `OnlineGpr` は crate 内部の `GprCore`（kernel spec・コンパイル済みカーネル・尤度・変換・方針・訓練データ・`α`・クエリ用バッファ）を共有し、違うのは因子だけ。`FittedGpr` は LLT の置き場（crate 内の `LltStore`。`FitBuffers` か mmap の `L`）、`OnlineGpr` は LDLT の `LdltStore` と `PointId` の表を持つ。`StoredFactor { Llt, Ldlt }` が因子の見方で、`solve`、列への `L⁻¹`、ピボットごとの重み（`1` か `1/Dᵢ`）、`log|A|`、`diag(A⁻¹)` を持つ。predict・共分散・sample・LOO・NLML・予測用 `α` はこの見方について `GprCore` に1回だけ書く。ハイパラの書き込み（`set_params`・勾配・Hessian・`fit`・`refit`）はすべて借用の `ExactFit`（core + LLT の置き場）1つで動く。`OnlineGpr` は `L √D` を O(n²) で詰めた一時的な LLT の置き場を貸し、新しい因子を書き戻す。訓練データは複製せず、O(n³) は新しい `θ` が要る分解だけ。

既定の `Gpr` は `Gpr<Lbfgs>`。`with_optimizer` が `O` を argmin の別のソルバ（`NelderMead`、`TrustRegion`）、`FastSimulatedAnnealing`、または自作の最適化器に差し替える。カーネルの葉の作り直しは §5.4 に従い、`with_recompute_strategy` は無い。`Gpr<Fixed>::factor` は分解だけ。`FittedGpr::predict` は対角分散で、クエリ間共分散は `predict_covariance`（§6）。`loo_predict` は GPML 5.4.2 の `L` と `α` から訓練点ごとの LOO を返す。ハイパラを変えて同じデータで分解し直すのは `FittedGpr::refit`（学習済みが持つ `O` のまま）。`with_optimizer` / `factor` / `into_trainer` / `refit` は方針を保つ。

```rust
pub struct Gpr<O = Lbfgs, P = DoublePrecision> {
    kernel: KernelSpec,
    likelihood: GaussianLikelihood,
    x_transform: Box<dyn UnfittedTransform>,
    y_transform: Box<dyn UnfittedTarget>,
    optimizer: O,
    policies: Policies,
    _precision: PhantomData<P>,
}

/// Crate-private. Shared by the trainer and its fitted models.
struct Policies {
    distance_cache: DistanceCachePolicy, // Cached (default) | Uncached
    cholesky_buffer: CholeskyBuffer,     // Retain (default) | Reuse
    math: KernelExp,                     // Accurate (default) | FastApprox
    jitter: JitterPolicy,                // fixed(0.0) (default)
}

pub struct Fixed; // Gpr<Fixed>::factor only; not an Optimizer

// Optimizers. Fields are private; each has `Default` and `with_*` setters.
pub struct Lbfgs       { max_iterations: u64, tolerance: f64, history_size: NonZeroUsize, restarts: Option<Restarts> }
pub struct TrustRegion { max_iterations: u64, tolerance: f64, initial_radius: f64, max_radius: f64, restarts: Option<Restarts> }
pub struct NelderMead  { max_iterations: u64, tolerance: f64, restarts: Option<Restarts> }
pub struct FastSimulatedAnnealing {
    max_iterations: u64, restarts: Option<Restarts>,
    initial_temperature: f64, cooling_rate: f64, seed: u64, boundary: BoundaryPolicy,
}
// Defaults: max_iterations 100, tolerance √ε, history_size 10, gamma 1, no restarts;
// FSA initial_temperature 1, cooling_rate 3, seed 0, BoundaryPolicy::Clamp.

pub struct FittedGpr<O = Lbfgs, P: GpScalar = DoublePrecision> {
    core: GprCore<P>,  // crate-private: everything but the factor
    optimizer: O,
    store: LltStore<P>, // FitBuffers<P> (L; dist / W per policy), or a mapped L
}

/// Crate-private. Shared by FittedGpr and OnlineGpr.
struct GprCore<P: GpScalar> {
    kernel: KernelSpec,
    compiled: CompiledKernel<P::Storage>,
    likelihood: GaussianLikelihood,
    x_unfitted: Box<dyn UnfittedTransform>, // kept for into_trainer / refit
    y_unfitted: Box<dyn UnfittedTarget>,
    x_transform: Box<dyn Transform>,
    y_transform: Box<dyn TargetTransform>,
    policies: Policies,
    query: QueryWorkspace<P>,  // predict_into buffers (§7.1)
    x_obs: Vec<f64>,           // caller-scale X, column-major n × d
    y_obs: Vec<f64>,
    x: Mat<f64>,               // transformed X (rows past n are online capacity)
    y_train: Vec<f64>,         // transformed y
    factor_alpha: Vec<P::Storage>, // the factor's solve; the NLML reads it
    alpha: Vec<P::Refine>,     // predict weights (refined for MixedPrecision)
    n: usize,
    d: usize,
    // plus the f32 cast scratch of the storage scalar
}

/// Crate-private fit objective. Borrows the model only during fit / refit and
/// writes θ through to its kernel and likelihood.
struct GprObjective<'a, P: GpScalar = DoublePrecision> {
    model: ExactFit<'a, P>,       // crate-private view: &mut GprCore + &mut LltStore
    scratch: Vec<f64>,
    leaves: LeafCache<P::Storage>, // §5.4
    incremental: bool,             // §5.4
}
```

`x` は列優先の `&[f64]` で受け、内部で `n×d` の `Mat` に詰める。

| メソッド | レシーバ | 確保 |
| -------- | -------- | ---- |
| `FittedGpr::predict` | `&self` | 出力 `Prediction` と、必要なら一時 query バッファ |
| `FittedGpr::predict_into` | `&mut self` | warmup 後は 0。`mean` / `variance` の容量を再利用 |

前提条件:
- 未学習の `predict` は型で起きない。未学習の `transform` / `apply` も型で起きない
- `FittedGpr::refit` は同じ `n`/`d` で L と `α` を置き換える
- クエリの入力次元`d`は固定。不一致は`DimensionMismatch`
- n=0は`EmptyInput`、nがカーネルの最低点数未満なら`InsufficientData`
- 入力のNaN/Infは`NonFiniteInput`
- Cholesky失敗時は `Err((gpr, err))`。中途半端な `FittedGpr` は返さない

既定の距離キャッシュ方針は `Cached`。`fit` 開始時に訓練点の二乗距離を一度埋め、以降のハイパライテレーションではカーネルだけを書き換える。等方は `n×n`。ARD は生の `(Δx_d)²` を、詰めた下三角（`d · n(n+1)/2` 個）に置く。`Uncached` はそれらのテンソルを Workspace に置かず、等方も ARD も `X` から距離を計算する。公開のメモリ優先は `with_prefer_memory`（`Uncached` + `Reuse`）。速さ優先は既定のまま（`with_prefer_speed`、`Cached` + `Retain`）。呼ぶたびに両方の方針を置き換える。キャッシュを確保するのはコンパイル済みカーネルが距離を読むときだけ。`RBF + White` と `Constant * RBF` は読む。単独の Linear / Constant / White は読まず、方針は保つが使わない。persist タグは `always` / `never`（タグが無ければ `Cached` で読む）。`LoadedGpr` は精度と分解の種類ごとに 1 つの variant（8 つ）。どちらもモデルの型パラメータだから。`predict` / `predict_with`（`f64` に広げる）、`n`、`d`、`is_online` は match せずにどの variant でも使える。variant を match するのは型つきのモデルが要るとき（`predict_into`、`insert`、`refit`）だけ。`load` は `Retain`。

### 6.4 Leave-one-out

Exact GPR の leave-one-out は、学習後の `L` と `α` から閉じた式で出る(Rasmussen & Williams, GPML §5.4.2)。`A = K + σn² I`、`Q = A⁻¹`、`α = A⁻¹ y` として

```
μ_i = y_i - α_i / Q_ii
σ_i² = 1 / Q_ii
```

これは観測の `p(y_i | X, y_{-i}, θ)`。潜在 `f_i` の LOO 分散は `max(0, 1/Q_ii - σn²)`。`Q_ii` は下三角 `L` から `L⁻¹` の列ノルムで取る(`A⁻¹ = L^{-T} L^{-1}`)。コストは Cholesky と同オーダーの O(n³)、追加メモリは `n×n` の一時行列。

`FittedGpr::loo_predict` は学習点と同じ長さの `Prediction` を返す。既定は `VarianceKind::Observation`。平均・分散は `predict` と同じく `TargetTransform` で元スケールへ戻す。White カーネルの葉は使わず、ノイズは `GaussianLikelihood` のみ。

sklearn に LOO API は無い。`just gen-goldens` は fit 後の `L_` / `alpha_` に同じ GPML 式を適用して JSON に書く。Rust 側は sklearn が選んだ `θ` で `Gpr<Fixed>::factor` して照合する(最適化器差を LOO に混ぜない)。

`FittedSgpr` / `OnlineSgpr::loo_predict` は、θ と `Z` を固定した collapsed VFE の事後分布の LOO。点 `i` を除いた最適な `q(u)` で `x_i` を予測する。`A = L_mm⁻¹ K_mn`、`B = σn² I + A Aᵀ`、`w = B⁻¹ A y` とすると、`i` を除くのは `B` から `a_i a_iᵀ` を引くことになる。`h = a_iᵀ B⁻¹ a_i`、`g = a_iᵀ w` として Sherman–Morrison から

```
μ_i = (g - h y_i) / (1 - h)
潜在 σ_i² = k(x_i, x_i) - ‖a_i‖² + σn² h / (1 - h)
```

三角解 `L_B⁻¹ A` を 1 回解けば、全体で `O(n m²)`。White のカーネルの葉を含まないカーネルでは、`Z = X` で Exact の LOO になる。`f32` の格納は、予測と同じく VFE 系を `f64` で組み直す。SVGP は LOO を持たない（§6.1）。

## 7. Workspaceとメモリ管理

### 7.1 個別バッファ構造

バッファ数は少数・固定なので、個別フィールドとして持つ。精度ポリシーのStorage/Refineを明示的に反映する。

```rust
/// Crate-private (as are all types in this block).
struct WorkspaceCore<P: PrecisionPolicy> {
    k_matrix: Mat<P::Storage>,       // A = K + σn² I, then L. W during a Reuse gradient
    exp_buf: Mat<P::Storage>,        // kernel evaluation, ∂K/∂θ. Reuse n-RHS lives here
    kernel_scratch: Mat<P::Storage>, // product / custom ∂K/∂θ. Empty until a tree needs it
    thread_scratch: Vec<Mat<P::Storage>>, // one per Rayon worker, detached before a parallel fill
    rhs: Mat<P::Storage>,            // n×1, training Cholesky right-hand side y → α
    faer_scratch: MemBuffer,         // faer's own scratch, used as-is
    theta: Vec<f64>,                 // θ before the current write, restored when A does not factor
    nested: Vec<Mat<P::Storage>>,    // one n×n per nesting level of a sum / product (§5.3)
    hessian: HessianScratch<P::Storage>, // S_i (p · n²), v_i (n × p), one n-vector. Empty until the first Hessian (§6.2)
    factor_jitter: f64,              // j of the last successful factor (§4.0)
}

struct FitBuffers<P: PrecisionPolicy> {
    core: WorkspaceCore<P>,
    dist: Option<DistCache<P::Storage>>, // Some for Cached (§5.2)
    w_matrix: Option<Mat<P::Storage>>,   // Some for Retain. W = ααᵀ - K⁻¹ (§6.2)
}

/// Held by FittedGpr / OnlineGpr. The first predict_into sizes it to (n, m, d).
struct QueryWorkspace<P: PrecisionPolicy> {
    query_xs: Vec<f64>,                 // transformed query (column-major)
    query_x: Mat<P::Storage>,           // m×d
    query_k_star: Mat<P::Storage>,      // k(X, X*), then L⁻¹ k_* (n×m)
    query_scratch: Mat<P::Storage>,     // apply_cross scratch (n×m)
    query_nested: Vec<Mat<P::Storage>>, // nested sum / product levels of the n×m block
    query_dist: Mat<P::Storage>,        // train–query squared distances (n×m)
    query_kss: Vec<P::Storage>,         // k(x*_j, x*_j)
}
```

和・積の項がさらに複数項の和・積のときは、入れ子 1 段ごとに出力と同じ形のバッファがもう 1 枚要る（`CompiledKernel::nested_depth`）。crate 内の fit / predict の入口は、その段を `nested` / `query_nested` から借りる。最初の呼び出しで伸ばし、以後は使い回す。公開の `CompiledKernel::apply` / `grad` / `hess` などはシグネチャを変えず、その呼び出しのぶんだけ段を用意する。対角の畳み込み（`fill_diag`、`fill_diag_points` と、その勾配・Hessian）は固定長のスタック上の行ブロックで項を合わせ、確保しない。Sparse のモデル（`FittedSgpr`、`OnlineSgpr`、`FittedSvgp`）は、カーネルのスクラッチ（出力と同じ形のスクラッチ、入れ子の段、訓練–クエリの距離）を crate 内の `SparseScratch` に持ち、`&mut self` の呼び出し（`set_params`、勾配、Hessian、オンライン更新）のあいだ使い回す。学習の因子は新しい行列で返すので、`tests/alloc.rs` はゼロではなく測った数をラチェットにする。`predict_into` の予測のバッファもそこに持つ。写したクエリ、詰めた `Z` とクエリ、`K(Z, X*)` とその解、コンパイル済みのカーネル（カーネルが変わったときだけ作り直す）。丸める格納（`f32`。single と mixed）では、3 つの sparse モデルとも、平均・分散・共分散を `f64` で予測する。`K_mm` を `f64` で分解し直し（Sgpr と Svgp が共有する `PredictScratch::f64_system`）、`B` と VFE の重みを昇格する。SVGP の `q` はもともと `f64` である。結果はその精度の `Refine` へ 1 回だけ丸める。同じ形で warmup した後は確保しない。`predict`（`&self`）は自分のバッファで同じ経路を通るので、両方とも同じ値を返す。

fit 用バッファは`fit`開始時にサイズが確定するため、`reserve_exact`で一度だけ確保(または`Mat::zeros`で1回構築)し、以降のイテレーションでは同じ領域に上書きする。query バッファは `FittedGpr` の `QueryWorkspace` が持ち、最初の `predict_into` で `(n, m, d)` に合わせ、同じクエリ長では再利用する。`predict(&self)` は出力 `Vec` を毎回確保してよい。あわせて、faer公式の`PodStack`/`MemStack`をスクラッチ管理に採用し、自前でスクラッチ領域をアリーナに内包する設計はとらない。

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
       Retain: w_matrix に K⁻¹ → W。Reuse: exp_buf で K⁻¹、k_matrix へ W
       exp_buf に ∂K/∂θ を順に書き ⟨W, dK⟩
fit()終了 → FittedGpr が L, α, X を保持（Reuse はここで Chol し直す）。W / ∂K / L-BFGS は捨ててよい
  → predict(&self): 出力を確保
  → predict_into(&mut self): query_* に上書き、`Prediction` の容量を再利用
```

バッチfitのWorkspaceはn固定。オンライン学習の容量成長は`LdltStore`(§11)が担当し、バッチ用Workspaceとはメモリ管理方針を分ける。`FitBuffers`、`QueryWorkspace`、`LdltStore`、faer の型はクレート私有。

## 8. 並列化・SIMD、数学関数バックエンド

- カーネル評価内側ループは `wide::f64x4` でベクトル化する。場所は `src/kernel/simd/` で、レーンの補助関数は 1 組（`simd/mod.rs`。読み込み、書き込み、有限性、列のスライス、`(x − x0)²`）。`simd/stationary.rs`: 等方 RBF の `apply` / `grad` / `apply_cross` と座標からの長方形の `grad`、等方の Periodic と RQ の正方の Gram（キャッシュした二乗距離から、どの `uplo` も）と 1 回の走査の重み付き勾配（`wide` の `sin_cos`・`ln`・`exp`。RQ の `u^{−α}` は `exp(−α ln u)`）。列の最後の半端なレーンは詰め物をして同じレーンの関数で計算する。ただし f64 の Accurate の等方 RBF の端は、スカラーの `exp` で計算する。`simd/rbf_ard.rs`: ARD RBF の `apply` / `grad` / 長方形の `apply` と `grad`。長方形の `⟨W, ∂K/∂θ_d⟩` は、全長さスケールを 1 回の `exp` から作り、`Σ (W ∘ k) (Δ_d)²` を 1 回の行列積で足す。Accurate の正方は `k` を 1 回作り、下三角から同じ積を足す。`simd/ard.rs`: ARD Matérn と ARD RQ の値と `θ` 微分（座標から、`(Δx_d)²` キャッシュから、長方形。4 行ぶんの `r²` を作ってから動径の式を `f64x4` で評価する）。`simd/dist.rs`: 二乗距離と `(Δx_d)²` の行ループ。ストライドが 1 でないビューと、有限でない値はスカラーに落とす（スカラーがエラーを名指しする）。`std::simd` は安定化まで使わない。等方の Matérn、Periodic / RQ の長方形・座標の経路、動径のカーネルの葉の座標微分（`radial.rs`）はスカラーである。これらをベクトル化するのは速さの作業で、ベンチでそこがボトルネックだと示してから始める（`.cursor/rules/bench.mdc`）。
- 距離行列・カーネル行列構築はRayonでブロック並列化。下三角は、面積がほぼ等しい列ブロックに分ける（`lower_block_start`、`par_lower_blocks`）ので、最初のブロックに仕事が集まらない。joint gradient の `n²` の書き込み（積の因子へ渡す重み、残した因子の Gram の積、畳み込みの三角の和と積）も同じブロックで並列に行う。和（カーネルの葉の重み付き走査と Frobenius の和）は、先頭の列から順に 1 列ずつ足す（`par_lower_fold`）。列の一群をプールで評価してから列の順に畳み込む。これは逐次に足したときと同じ結合なので、和はスケジュールにもプールの大きさにも依らない。面積ブロックをまとめて足すと結合が変わり、最適点の近くでは勾配がほぼ打ち消しで決まるので、その丸めが L-BFGS の進み方を変える
- faer自身もRayon並列化されるため、外側との二重並列化に注意。単一の`rayon::ThreadPool`を共有。faer の本数は `min(プール, n/64, n·k/16384, k/12)`（[ADR 0001](adr/0001-faer-parallel-degree.md)）。`k` は RHS 列。カーネル埋めはプール全部

**カーネルの `exp` は最小限の API から始め、デフォルトは近似ではなく正確な実装にする**。カーネル行列の近似誤差は正定値性・Cholesky安定性・尤度・勾配・予測値すべてに波及するため。

```rust
/// Runtime choice on every model, set by with_math.
pub enum KernelExp { Accurate, FastApprox }

/// Zero-sized markers the kernel code is monomorphized over. `KernelMath` is
/// sealed through the crate-private `MathOps` (exp, its jet, and the f64x4 forms).
pub struct Accurate;   // f64::exp / f32::exp / wide::exp
pub struct FastApprox; // degree-7 polynomial exp in the storage scalar
pub trait KernelMath: MathOps {}
```

デフォルトは `Accurate`。`FastApprox` はカーネル評価の `exp` を fit でも predict でも置き換える。長さスケールへ戻す `exp(θ)` と `KernelTerm` の式は正確な `exp` のまま。すべてのモデル（`Gpr` / `FittedGpr` / `OnlineGpr`、`Sgpr` / `FittedSgpr` / `OnlineSgpr`、`Svgp` / `FittedSvgp`）で、モードは実行時の enum `KernelExp` で、`with_math(KernelExp::FastApprox)` で設定する。カーネル呼び出しごとに、そこから `KernelMath` の印へ 1 回だけ分岐する。`erf` などは、それを要るカーネルや尤度が出てから足す。`FittedGpr` と `OnlineGpr` はモードを保存し、欄が無いファイルは `Accurate` で読む。

## 9. Optimizer設計

**アロケーションフリー化とResultラップ**。目的関数の能力は層になっていて、ソルバは要るものを実行時のフラグではなく境界で示す: `Objective`（値、§5.4）⊂ `Differentiable` ⊂ `TwiceDifferentiable`。

```rust
pub trait Differentiable: Objective {
    fn gradient_into(&mut self, params: &[f64], out: &mut [f64]) -> Result<(), GprError>;
    /// Override this so one evaluation shares its inner work (Cholesky, W, exp_buf).
    fn value_and_gradient_into(&mut self, params: &[f64], out: &mut [f64]) -> Result<f64, GprError> {
        let value = self.value(params)?;
        self.gradient_into(params, out)?;
        Ok(value)
    }
}

pub trait TwiceDifferentiable: Differentiable {
    fn hessian_into(&mut self, params: &[f64], out: &mut [f64]) -> Result<(), GprError>; // row-major p×p
}

pub struct OptResult {
    pub params: Vec<f64>, // same space as init (log-θ for the GP models)
    pub value: f64,
    pub iterations: u64,
}

// Optimizer<P> is in §5.4: `P` is the objective type the algorithm can minimize.
```

`init`はスライスにする(呼び出し側のVecを消費しない)。`GprObjective`は`value_and_gradient_into`をオーバーライドし、§6.2の手順でL・α・W・`exp_buf`を共有する。`GprObjective` は `TwiceDifferentiable` を impl し、`hessian_into` は `FittedGpr` へ転送する。`SgprObjective` は `Sgpr` 用の同じ crate 内アダプタ。区間は各パラメータの `Interval` から `Objective::fill_intervals` で取る（公開。ユーザーの Optimizer も読め、同じ目的関数で組み込みの最適化器を呼べる。既定は `Interval::DEFAULT_POSITIVE`）。

トレーナーの境界は `O: for<'a> Optimizer<GprObjective<'a, P>>`。既定は `Lbfgs`。`Lbfgs` は `Differentiable`、`TrustRegion` は `TwiceDifferentiable`、`NelderMead` / `FastSimulatedAnnealing` は `Objective` だけを要る。argmin のアダプタは、ユーザー単位の区間を logit で写して argmin を制約なしのまま動かす（正の区間は対数一様、中点でヤコビアンが 1 になるよう縮尺）。`TrustRegion` は解析ヘッセも写す。`TrustRegion` は argmin の信頼領域法（部分問題は Steihaug）で、Hessian を使うソルバ。Hessian が非正定・特異でも、ステップが区間の外へ出ても、領域が縮むことで扱う。argmin の 3 つの最適化器は評価のキャッシュを 1 つ共有する（`src/optimizer/adapter.rs`）。評価できない点、または値・勾配・Hessian が有限でない点は、バリア値の費用と 0 の勾配・Hessian にする。線探索はそこから戻り、信頼領域は縮む。目的関数自身のエラーは型を保つ。それ以外のソルバの失敗は `OptimizationNotConverged`、ソルバを組むときに argmin が退ける設定は `InvalidConfig`。`FastSimulatedAnnealing` は gprx 自前の値だけのソルバ（Cauchy / Metropolis）で、logit は使わず受け取った log-`θ` の上を進む。自作最適化器の例でもある。目的関数の型は crate 内なので、自作の最適化器は要る能力について `Optimizer<P>` をジェネリックに impl し（`impl<P: Objective> Optimizer<P> for Mine`）、`with_optimizer` で同じ型パラメータを差し替える。その隣に無視される別のソルバ設定は置かない（`.cursor/rules/types.mdc`）。準ニュートンを gprx が自前実装しない。`Adam` は `Svgp` のミニバッチのループで、`Optimizer` ではない。実行時の NotImplemented は置かない。

## 10. エラー型 GprError

数値計算固有の失敗理由を拡充する。

集合が増えうる公開 enum は `#[non_exhaustive]` にする: `GprError`、`CholeskyStage`、`IntervalError`、`LoadedGpr` / `LoadedSgpr` / `LoadedSvgp`、`LoadedDistanceGpr` / `LoadedDistanceSgpr` / `LoadedDistanceSvgp`、`PersistKind`、`KernelSpec`、`CompiledKernel`、`DistanceSlot`、`DistanceCachePolicy`、`JitterPolicy`、`KernelExp`、`BoundaryPolicy`。これらへの variant の追加は破壊的変更にならない。クレート外の `match` には `_` が要る。閉じた集合は網羅的な `match` を書けるよう付けない: `Triangle`、`MaternNu`、`VarianceKind`、`CholeskyBuffer`。

```rust
#[derive(Clone, Debug, thiserror::Error, PartialEq)]
pub enum GprError {
    DimensionMismatch { x_dim: usize, expected_dim: usize },
    InsufficientData { n: usize, min: usize },
    EmptyInput,
    NonFiniteInput,
    NonFiniteKernelValue,
    CholeskyFailed { jitter: f64, matrix_size: usize, stage: CholeskyStage },
    NonPositiveDefiniteMatrix,
    CoordGradientUnsupported,
    OptimizationNotConverged { iterations: usize },
    InvalidHyperparameter { reason: String },
    ShapeMismatch { reason: String },
    InvalidDistance { slot: Option<usize>, dim: Option<usize>, pair: Option<(usize, usize)>, reason: String },
    DistanceSlot { kind: SlotErrorKind, slot: Option<usize> },
    LengthMismatch { reason: String },
    IndexOutOfRange { reason: String },
    InvalidConfig { reason: String },
    SizeOverflow,
    InvalidInterval(IntervalError), // #[from]
    InvalidNoiseVariance { reason: String },
    UnsupportedKernelOperation { reason: String },
    WorkspaceTooSmall,
    InvalidPointId,
    InvalidInducingId,
    PersistFailed { kind: PersistErrorKind, reason: String }, // kind: Io / Config / Tensor / InvalidPersistId / NotPersistable / UnregisteredId / WrongModel
    UnsupportedPersistVersion { found: u32, supported: u32 },
}

pub enum CholeskyStage { Fit, Predict, OnlineInsert, OnlineDelete }
```

表示文は英語（`src/error.rs`）。

`InvalidHyperparameter` はハイパーパラメータの値が定義域の外にあるときだけに使う。行列の形状・スライス長・添字の誤りは `ShapeMismatch`・`LengthMismatch`・`IndexOutOfRange`。最適化器・jitter ポリシー・変換の設定値は `InvalidConfig`。サイズの積の `usize` オーバーフローは `EmptyInput` ではなく `SizeOverflow`。値を含まない区間は `InvalidInterval`。供給した二乗距離が有限でない、負、対角が 0 でない、鏡像の要素と食い違う（ソースの修復が許す範囲を超えて）ときは `InvalidDistance`。表の slot（カーネルの `slots()` での位置）、ARD の次元、呼び出し側が渡したブロックの中の組 `(row, col)` を持ち、検査で分からなかったものは `None` になる。カーネルが読まない slot の供給（保存前の slot など）、1 つの slot への 2 つの供給、供給の無い slot は `DistanceSlot` で、`kind`（`SlotErrorKind`、non_exhaustive）でどれかを示し、位置があれば slot の位置を持つ。保存・読み込みの失敗は `PersistFailed`（どこで失敗したかを `kind`（`PersistErrorKind`、non_exhaustive）で示し、呼び出し側は `reason` を読まずに分岐できる）、別の形式バージョンのファイルは `UnsupportedPersistVersion`。

**Error/panicの線引き**: ユーザー入力起因(`DimensionMismatch`等)、モデル/データ起因(`CholeskyFailed`等)は`Result`で返し回復可能にする。`CoordGradientUnsupported`はライブラリ内部panic対象ではないため`unimplemented!()`ではなく本Errorを返す。`NotFitted` の variant は無い。未学習の呼び出しは書けない。

## 11. オンライン学習(データ点の追加削除)

GPRはn増加に伴いO(n³)でコストが増大するため、データの逐次追加削除を正式にスコープへ含める。バッチfit用Workspace(n固定)とは別に、crate-private の `LdltStore` と公開の `OnlineGpr` を置く。`FittedGpr::into_online(self)` が変換する。`insert` は `OnlineGpr` だけにある。

### コスト比較

| 操作    | フル再fit | 増分更新 |
| ------- | --------- | -------- |
| 1点追加 | O(n³)     | O(n²)    |
| 1点削除 | O(n³)     | O(n²)    |

### faer APIに合わせた実装方針

§3の通り、**LLTにinsert/delete APIは無い**。オンライン経路は次で進める。

1. **追加(末尾append)**: 自前で bordered update を実装する。O(n²)
2. **削除(任意インデックス)**: `LdltStore`は**LDLT因子**を保持し、`ldlt::update::delete_rows_and_cols_clobber`を使う。`2×2` / `5×5` の手書き SPD で、削除後の再構成 `A = L D Lᵀ` がフル LDLT と一致する。Givens downdate は置かない

バッチfitはLLTのままにする。`FittedGpr::into_online` で LLT→LDLT へ O(n²) 変換する:

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

バッチfitとオンラインは性質が異なる(n固定 vs n増減)。容量拡張時は **LD・`y`・`α`・`v_buf` を同じ手順で**再確保・コピーする。delete の faer スクラッチも同じ容量に伸ばす。予測・NLML・insert は Gram `K` と距離キャッシュを読まないので、`LdltStore` には置かない。

因子は転置して、列優先の `n_capacity × n_capacity` 行列に `Lᵀ` として持つ。`L` の各行が連続するので、追加は連続した 1 列を書き、その前進代入 `L w = k` は各行を連続に読む（8 行ずつのパネルで、パネルの行と解き終えた `w` の頭の内積を 1 回の走査で取り、8×8 の三角を解く。faer の再帰の solve と違い、平らなループ 1 本）。solve と persist は下三角のビュー `ld()`（`Lᵀ` の行優先ビュー）を読み、保存する詰めた LDLT は変わらない。`f32` の格納では LDLT の solve を `f64` で積算して最後に 1 回だけ丸める。そのビューに対する `f32` の solve は行の順に丸まり、`k*ᵀ α` で打ち消しが悪くなるため。

crate-private。`from_active(n)` で `n_active = n_capacity = n`。訓練 `X` は `OnlineGpr` が持ち、この struct には置かない。末尾 insert の前に `OnlineGpr` が `ensure_capacity` する。倍率フィールドは置かない。

```rust
struct LdltStore<T: KernelScalar = f64> { // T is the precision's Storage
    lt: Mat<T>,                // Lᵀ with D on the diagonal: row i of L is the head of column i
    y: Col<T>,
    alpha: Col<T>,
    v_buf: Vec<T>,             // k of an appended point, then its forward solve
    delete_scratch: MemBuffer, // faer delete_rows_and_cols. Grown with capacity
    n_active: usize,
    n_capacity: usize,
    factor_jitter: f64,        // j of the batch factor; every row factors A + (σn² + j) I
}
```

**容量拡張** (`ensure_capacity(needed)`。`n_capacity < needed` のとき):

1. `new_cap = max(needed, max(n_capacity, 1) * 2)`
2. `lt`, `alpha`, `y`, `v_buf`を`new_cap`で再確保。delete スクラッチも `new_cap` 用に伸ばす
3. 既存の`n_active × n_active`下三角と長さ`n_active`のベクトルをコピー
4. `PointRegistry`のインデックスは`n_active`未満のままなので付け替え不要
5. 拡張後にinsertを実行する。更新アルゴリズムの最中には再確保しない

**predict時の分散計算**: 予測平均はO(n)だが、予測分散`σ*² = k(x*,x*) - vᵀ D v`(LDLT、`L v = k*`の変形)はテスト点1点あたりO(n²)。`v_buf`をあらかじめ確保しておく。

### 増分更新の手順と不変条件

**追加（末尾）**: ①容量が足りなければ `LdltStore::ensure_capacity`（倍率 2）。`OnlineGpr` の訓練 `X` / `y` も同じ倍率で伸ばす。クエリバッファは `ensure_at_least` → ②新規点と既存n点との距離計算(O(n)。1 列は逐次、`v_buf` に `k` を直接書く) → ③カーネル対角 `k_new` だけ足す（insert は `K` の新行/列を書かない。予測・NLML は LD だけ読む） → ④bordered LDLT update(O(n²)。三角ソルブは `v_buf` を再利用) → ⑤`α` は insert では解かない（libgp `alpha_needs_update`）。O(1) で古い印を付けるだけ。最初に読む操作が LDLT で解き直す。`&mut self` の読み（`predict_into`・ハイパラの書き込み）はモデルに `α` を置き、`&self` の読み（`predict`・共分散・sample・LOO・NLML・`alpha()`・`save_with_factor`）は次の insert / delete が空にする `OnceLock` のキャッシュを埋める。解くのに失敗したら（`MixedPrecision` の f64 へのやり直しが分解できないなど）その読みの `Err` になるので、`OnlineGpr::alpha()` は `Result` を返す → ⑥`PointRegistry` に新しい `PointId` を発行。

**削除**: ①`ldlt::update::delete_rows_and_cols_clobber`でLD更新(O(n²)。スクラッチは `LdltStore` に置き再利用) → ②`OnlineGpr` の y・`X` から該当要素を除去し、後ろの行/列を詰める(O(n)) → ③`PointRegistry`のインデックスを同じ順序でシフト → ④`α` は delete でも解かない。古い印と最初の読みでの解き直しは追加の⑤と同じ。`n_capacity` は据え置く。最後の 1 点は消さない（`InsufficientData`、`min = 2`）。未知・削除済みの `PointId` は `InvalidPointId`。

**不変条件**: 削除により内部インデックスがシフトする際、workspace の `LD` / `y` / `alpha` と `OnlineGpr` の `X` と `PointRegistry`は**必ず同じ順序で同期**しなければならない。いずれか一つでも順序がずれると誤った解になる。この不変条件をテスト(§12)で明示的に検証する。

```rust
/// Crate-private. One registry type for data points and inducing points.
struct IdRegistry<I: RegistryId> {
    id_to_index: HashMap<I, usize>,
    index_to_id: Vec<I>,
    next_id: u64,
}
type PointRegistry = IdRegistry<PointId>; // OnlineGpr, OnlineSgpr
type InducingRegistry = IdRegistry<InducingId>; // OnlineSgpr's inducing points
```

### 与えられた二乗距離

与えた二乗距離のモデル（§5.6）も変換でき、学習の二乗距離（`TrainSources`）を因子と並べて持つ。`insert` はスロットごとに、生きている点から新しい点への二乗距離の `n × 1` の列（`point_ids` の順）を受け取る。ARD のスロットはこの列を `d` 本受け取る。ソースは予測が受け取るものなら何でもよい。列は学習の正方行列と同じく全体を検査し（`tidy` のソースは許容誤差の内で直す）、何かを変える前に確かめてから保持する。予測のブロックと違い、呼び出しの後も残るためである。どの並びも、点を足すのと消すのが安くなるように伸びる。

- **scalar のスロット**は密な対称の正方行列のままにする。scalar の葉（と `KernelTerm`）が `d²` を密な `MatRef` として読むためである。先頭次元 `cap ≥ n` は、いっぱいになると 4 分の 1 だけ伸ばす（`cap = n + n/4`）。並べ直しで `n²` 個をコピーするのは `n/4` 回の挿入に 1 回で、正方行列は `n²` に近いままである。挿入は新しい列（連続）と鏡像の行を書く。削除は、`i` より前の列は `i` より下の行を 1 つ上へ、`i` より後の列は列ごと 1 つ左へ動かす。2 つの部分は同じ列を持たない。
- **ARD のスロット**は、fit では下三角の列の並び（列 `j` が行 `j..n` を持つ）で読む。座標のキャッシュと同じ並びである。この並びで新しい点を足すと、すべての列に値が 1 つずつ増え、`d · n` 回の飛び飛びの書き込みになる。そこで最初の挿入か削除のときに一度だけ、スロットを**行の並び**にする。次元ごとに 1 本のバッファを持ち、行 `i` が列 `0..=i` を `i(i+1)/2` から持つ。挿入は、`Vec` が確保した余地に次元ごとに連続した 1 本を足すだけになる（倍々に伸び、クローンもその余地を保つ）。削除は `i` より後の行を前から 1 回なめ、それぞれの行から列 `i` を除く。読み出しは行の並びをその順に読む。`f64x4` の Gram と勾配のループは下三角を 4 行ずつ埋める（各行の並びの連続した 4 列を一度に読み、葉の値をレーンごとに取り、4 × 4 の転置で出力の各列に 4 行ずつ書く）。勾配の縮約は `weight ∘ K` をキャッシュと同じ並びに置き、次元ごとに連続した内積を 1 回取る。スカラーのループ（`f32`、ヘッセ行列）も同じパネルで下三角を回る。fit と座標のキャッシュは列の並びのままなので、その順序、つまり丸めは変わらない。

削除は、座標のモデルが `O(n · d)` 個を動かすところで、`O(n²)` 個を動かす。動かす量が大きいとき（`2^16` 個以上）は、このスレッドが因子を更新する間に Rayon のプールのワーカーで動かし（`rayon::in_place_scope`）、ARD のスロットの次元どうしも並べて動かす。量が小さいときは因子の更新の後に動かす。眠っているワーカーを起こす費用が同じくらいかかるためである。ワーカーに積むジョブは確保を伴う。ワーカーが 1 つのときはすべてこのスレッドで走り、何も確保しない。失敗しうる段は、どちらかが何かを変える前にすべて済ませる。ストアは自分の並びを整えて添字を検査し、因子の更新が失敗するのは、すでに拒んだ添字か点数のときだけである。そのあとはどちらも失敗しないので、因子とストアが食い違うことはない。挿入も同じく、点、目的値、新しい `PointId` の余地、すべての列を、ストアが余地を作る前に検査する。2 つの写しを持つストア（混合精度）は、どちらかを書く前に両方を検査する。

### API

**insert/deleteとハイパラ再最適化を分離する**。

未学習の `Gpr` には点を足さない。バッチの `FittedGpr` に `insert` は無い。

```rust
impl<O, P: GpScalar, K: ModelKernel> FittedGpr<O, P, K> {
    pub fn into_online(self) -> Result<OnlineGpr<O, P, K>, GprError>;
}

impl<O, P: GpScalar> OnlineGpr<O, P, DistanceKernel<DistanceOnly>> {
    pub fn insert<'s>(&mut self, sources: impl IntoIterator<Item = DistanceSource<'s>>, y_new: f64)
        -> Result<PointId, GprError>;
}
impl<O, P: GpScalar> OnlineGpr<O, P, DistanceKernel<WithPoints>> {
    pub fn insert<'s>(&mut self, sources: impl IntoIterator<Item = DistanceSource<'s>>,
        x_new: &[f64], y_new: f64) -> Result<PointId, GprError>;
}

impl<O, P: GpScalar> OnlineGpr<O, P> {
    pub fn insert(&mut self, x_new: &[f64], y_new: f64) -> Result<PointId, GprError>;
    pub fn delete(&mut self, id: PointId) -> Result<(), GprError>;
    pub fn point_ids(&self) -> &[PointId];
    pub fn alpha(&self) -> Result<&[P::Refine], GprError>; // solves a stale α (above)
    // predict / covariance / sample / loo / NLML / set_params / refit / save as on FittedGpr
}

pub struct OnlineGpr<O = Lbfgs, P: GpScalar = DoublePrecision> {
    core: GprCore<P>,             // §6.3
    optimizer: O,
    workspace: LdltStore<P::Storage>,
    registry: PointRegistry,
    alpha: AlphaState<P>,         // stale flag, stored α, and the OnceLock cache
}
```

`insert` / `delete` は現在のカーネル・ハイパラのまま LD を更新し、`α` に古い印を付ける。ハイパラ再最適化は `OnlineGpr::refit` / `set_params` を明示したときだけ。これらは一時的な LLT の置き場でバッチの fit を動かし（§6.3）、`PointId` とワークスペースの容量を保つ。`into_online` は既存 `n` 点に `0 .. n-1` を付け、以降の `insert` は単調増加で再利用しない。`PointId` に公開コンストラクタは無い。`PointRegistry` は crate-private でオンラインのモデルが持つ。persist は `FORMAT_VERSION` 1 のまま `factor_kind`（`llt` / `ldlt`）を必須にする。`llt` の load は `FittedGpr`。`ldlt` は `OnlineGpr` で、`point_ids` と `next_point_id` も必須。Sparse のオンラインは `OnlineSgpr`（§6）。

## 12. テスト計画

速度より前に正しさを保証する。どの経路も、最適化する前に正しさのテストを持つ。

1. **カーネルの数学的正当性**: RBF/Matern/Periodicの既知値比較、対称性、対角値、数値微分と解析的勾配の比較、`uplo=Lower`と`Full`の一致
2. **Choleskyの正当性**: `K=LLᵀ`再構成誤差、jitterあり/なし、悪条件・重複データでの挙動
3. **MLLと勾配**(推論テストから独立させる):
   - 既知の小規模問題でのMLL解析値比較
   - MLLの数値微分と解析的勾配の比較
   - 各カーネルパラメータの勾配比較
   - ノイズパラメータ(`log_noise_variance`)の勾配比較。`∂K/∂θ = σn² I`であること
   - 悪条件行列での勾配安定性
4. **オンライン更新**: 1点追加/削除とフル再fitの結果一致、任意インデックス削除、追加削除の繰り返し、PointIdと内部インデックスの整合性(§11の不変条件)
5. **オンラインのプロパティテスト**: ランダムな insert/delete 列の各段階で incremental == `Gpr<Fixed>::factor`（mean, variance, LML, alpha）。削除順はテスト内のシードつき `rand` でランダム化する
5b. **オンライン insert の外部照合**: 同じ θ の libgp `add_pattern` と predict（平均・観測分散）および NLML を相対 `1e-8`。delete の外部 API は無い。`cargo test` はコミット済み JSON を読む（C++ を呼ばない）
5c. **Sparse の外部照合**: 同じ初期 θ の `Sgpr<Fixed>::factor` と GPyTorch の周辺化 SGPR、`Svgp<Fixed>::factor`（prior `q`）と whitened SVGP を相対 `1e-8`（平均・Observation・Latent・NLML / ELBO）。`cargo test` はコミット済み JSON を読む（Python を呼ばない）
5d. **Sparse オンラインの外部照合**: 同じ初期 θ の `OnlineSgpr` の `insert` / `delete` / `insert_inducing` / `delete_inducing` を、各段階の GPyTorch の周辺化 SGPR（フル再組み立て）と相対 `1e-8`（平均・Observation・Latent・NLML）。`cargo test` はコミット済み JSON を読む（Python を呼ばない）
6. **精度**: f32/f64/混合精度の比較、悪条件行列、収束しないケースでのf64フォールバック
7. **推論結果**: 既知の小規模GPR実装との比較(mean、潜在分散、観測分散、log marginal likelihood, gradient)。sklearn JSON は数値の第二照合であり、公開 API の契約ではない。アルゴリズムの正本は GPML / Rasmussen
8. **前処理**: `StandardizeTarget`適用後のpredictが、未標準化モデルと元スケールで一致すること(アフィン変換の閉じた関係)
9. **最適化後の推論**: 1次元 Forrester と 2次元重み付き球関数（ARD）で sklearn L-BFGS と `Gpr::fit` を緩い許容で照合する。固定ハイパラ JSON（1e-8）とは分ける。`cargo test` は Python を呼ばない
10. **Leave-one-out**: n=2 の GPML 解析式、n=3 の実 leave-one-out `fit`+`predict`、および最適化 JSON（9 項）の LOO 欄を sklearn の `θ` で照合する。`cargo test` は Python を呼ばない
11. **persist**: どのモデル型でも save → load の往復で同じ値を予測する
12. **確保**: `tests/alloc.rs`（§15.1）

golden は `compare/goldens/` にあり、`just gen-goldens`、`gen-online-goldens`、`gen-sparse-goldens`、`gen-sparse-online-goldens` で作り直す。`cargo test` は読むだけ。

## 13. 実装ロードマップ

並びと状態は [roadmap.md](roadmap.md)。完了条件は各 Issue に残す。

## 14. 未解決事項

1. **混合精度反復改良のパラメータ検証**: §4.2のデフォルト値は理論根拠付きだが、実ワークロードでの検証は未実施。`PromoteStorage`と`ReevaluateKernel`の精度差、fit時MixedPrecisionのlog|K|・トレース項も含む

## 15. ベンチマーク戦略

**正しさの次に、同じ経路を測りながら積む。** 最適化は測った基準から始め、推測からは始めない。詳細な運用は `.cursor/rules/bench.mdc`。

### 15.1 系統

| 系統 | 道具 | いつ回す | 見るもの |
|---|---|---|---|
| 時間 | criterion、`benches/exact.rs`、`benches/distance.rs` | `just bench`（ローカル）。既定 CI では回さない（ノイズ） | 壁時計。グループを分けて測る |
| 確保 | `tests/alloc.rs` | `just test`（必須） | Workspace 確保**後**の新規確保回数。上限は ratchet（減ることはあっても、Issue なしに増えない）。計測はプロセス全体の確保を数えるので、このバイナリは libtest のハーネスを持たず（`harness = false`）、1 本のスレッドで検査を順に走らせる（#494） |
| ライブラリ横断の時間と RSS | `compare/perf/` | `just perf`、`perf-online`、`perf-online-stages`、`perf-online-delete`、`perf-sparse`、`perf-sparse-online`（手動、CI なし） | gprx と sklearn / libgp / friedrich（Exact）、libgp（オンライン insert）、GPyTorch / GPy（Sparse）、GPyTorch（Sparse オンライン）。正しさのゲートは置かない |

時間と確保を一つの数字に混ぜない。L-BFGS 全体と「MLL+勾配 1回」も混ぜない。Sparse とオンラインのグループは criterion に足さず、`compare/perf/` で測る。

### 15.2 固定問題（回帰の単位）

毎回同じ入力でないと、速くなったのかデータが変わったのか分からない。

- criterion は `n = 256`。ライブラリ横断のハーネスは `n = 256 / 1024 / 4096`（Forrester）と `16×16 / 32×32 / 64×64`（球）
- 等方: 1 次元 Forrester `f(x)=(6x-2)² sin(12x-4)`、`x ∈ [0, 1]`、RBF + `GaussianLikelihood` + `StandardizeTarget`。初期ハイパラ `ℓ = 1`、`σn² = 0.1`
- ARD: 2 次元重み付き球 `f=(x/0.25)²+(y/1)²`、`[0, 1]²` の 16×16 格子。初期 `ℓ_d = 4`（`ℓ_d = 1` では線探索が初手で止まる）
- `y` は上記の関数 + `N(0, 1)`（gprx のシードつき乱数。Forrester は seed `0`、ARD 球は seed `9`。seed `0` は `Uncached` で尾根に沿って進む）。独立な乱数系列にはしない（L-BFGS の評価回数が目的関数の形でぶれる）
- `benches/exact.rs` の criterion グループ（存在する経路だけ）:
  1. `kernel_rbf` — K の下三角構築
  2. `cholesky_alpha` — `A` の LLT と `α`
  3. `mll_and_grad` — §6.2 の 1 評価
  4. `predict_100` / `predict_100_mixed` — テスト点 100。`DoublePrecision` と `MixedPrecision`
  5. `fit_lbfgs` — 最適化ループ全体。壁時計と一緒に L-BFGS の評価回数を残す。回数が違うときの差は速度差と読まない
  6. `fit_fsa` — カーネルの葉 2 つの和での `FastSimulatedAnnealing`（§5.4 のカーネルの葉の作り直しの経路）
  7. `mll_and_grad_ard` / `fit_lbfgs_ard` — 重み付き球の ARD RBF。`Cached` と `Uncached`（ベンチ ID は `always` / `never`）。等方とは比べない。`fit_lbfgs_ard` も評価回数を残す
  8. `kernel_exp` / `kernel_exp_ard` — 距離を一度埋めたあとの `apply` と θ の `grad`。`FastApprox` と `Accurate`。`mll_and_grad` とは混ぜない

### 15.3 基準

名前付きの criterion baseline と、それを取った機械は `.dev/bench-log.md`（ローカル。コミットしない）に残す。今の比較の基準は `phase-2`。ホットパス（`src/kernel/`、`workspace`、`gpr`、`objective`、`sgpr`、`svgp`、`precision`）を変える PR は、Verification にその基準との criterion 結果を貼る。速さと無関係ならその理由を書く。

与えられた距離（5.6 節）には専用の基準 `d1-coords` がある。行が距離の場合を足す前に、座標の経路で `cargo bench --bench distance -- --save-baseline d1-coords` を回して取る。D1 の各行は、距離の場合を `benches/distance.rs` に足し、同じ機械で測った `d1-coords` と並べて貼る（`--baseline d1-coords`）。`tests/alloc.rs` は、同じ問題での座標の経路の確保数を持つ（`DISTANCE_BASELINE_ALLOCS`）。距離の場合は、同じ操作のその数を超えてはならない。

### 15.4 指標

ゲートは「基準より悪くない」。

| 指標 | 内容 |
|---|---|
| MLL+grad 1回 | n, カーネル別。最適化ループとは別 |
| Fit（L-BFGS） | イテレーション込み |
| Predict | テスト点数別。潜在 / 観測 |
| ピークメモリ | Workspace 込み。`w_matrix` を含む |
| Allocations | セットアップ後の回数。ratchet。Exact の predict / fit のホットパスは 0 |
| 並列 | スレッド数別。faer との二重並列に注意 |
| f32/f64 | 精度と速度 |
| Online insert/delete | 1点 vs フル再 fit |

### 15.5 やらないこと

- 測らずに「速くなるはず」で Rayon / SIMD / 近似 exp を入れる
- CI の criterion を赤/緑のゲートにする（マシン差でフレークする）
- 数える前に確保 0 を要求する（まず数え、上限を段階的に下げる）
