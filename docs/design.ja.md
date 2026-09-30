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
       → Optimizer (型パラメータ。既定 `Lbfgs`。差し込み口は P2B-1。argmin ソルバは P2B-2。自作 `O` は同じ口で `minimize` される。使用例は P2B-15)
       → fit(self) → FittedGpr | (Gpr, GprError)
  → FittedGpr (L, α, X。predict / predict_into / refit / loo / save)
       → persist: 1 ディレクトリ（`config.json` + `model.safetensors`）。`format_version` 1。`factor_kind` は必須（`llt` / `ldlt`）。`llt` の `load` は `FittedGpr<Fixed>`。`ldlt` は `OnlineGpr<Fixed>`。因子があるとき mmap。再学習は `with_optimizer` → `refit`
       → OnlineGpr: `FittedGpr::into_online(self)` で LLT→LDLT。末尾 `insert` は `OnlineGpr` だけ
       → Phase 4: Sgpr は同様に学習済み型を返す
```

主要な設計原則:
- **識別子は gprx / GPR の概念を名付ける**（カーネル、尤度、θ、分解、正パラメータの区間、…）。他製品・テストハーネス・無関係なドメインの名前は置かない
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
  - オンライン学習はこれに合わせて§11の方針で実装する(追加は自前、削除はLDLT API)。`delete_rows_and_cols_clobber` は P3-1 でフル LDLT 再構成と一致する
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
- Sparse のモデルが分解する `K_mm = k(Z, Z)` には観測ノイズが入らない。`Sgpr` / `Svgp` は `K_mm` 用の `with_jitter_policy` を持ち、既定は Exact の既定（再試行なし）ではなく `adaptive(1e-8, 10, 5, 1e-3)` にする。近い誘導点では、浮動小数点で `K_mm` が特異になるため（R5-4、[#282](https://github.com/YUKIKEDA/gprx/issues/282)）。fit、factor、`set_params`、予測、オンライン更新は、すべてこの方針を使う。

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
   - 反復改良は、fit（またはオンライン更新）が残した因子の上で行う。`α₀` はその因子で `y` を解いたもの、補正も同じ因子で解く。解く系は `A + (σn² + j) I` で、`j` は因子の再試行で足した jitter（再試行なしなら `0`）。fit が `j` を記録し、オンラインの追加も同じ `j` を足し、保存した因子にも `j` を残す。f32 の分解をやり直さない。収束した `PromoteStorage` の α は f64 の系で1回だけ確かめる（カーネルを列ブロックで評価し、`n×n` の f64 行列は持たない）。`κ(A) u_f32` が大きく満たさないときは、同じ系の f64 Cholesky の解に落とす。f64 へのやり直しはモデルの `JitterPolicy` で再試行し、失敗時は呼び出し元の段を返す（R3-2、[#237](https://github.com/YUKIKEDA/gprx/issues/237)）。
   - f64の`A`を丸ごと保持する方式はメモリ削減と矛盾するため採用しない。
4. f32の`L`で`delta = solve(L, r)`、`alpha_1 = alpha_0 + delta`
5. 収束するまで数回繰り返す

実装優先度: 省略時は `DoublePrecision`（Storage = f64、Refine = f64、今の f64 経路）。`SinglePrecision` は Storage = f32、Refine = f32 で同じ手順を f32 で計算し、分解結果をそのまま使う。残差の型パラメータは持たない。`MixedPrecision<R = PromoteStorage>` は Storage = f32、Refine = f64。f32 で分解し、予測用の α だけ反復改良する。学習中の MLL と勾配は、その精度の因子を使い、反復改良は学習ループの中では行わない。残差の型は `MixedPrecision` にだけ付く。`PromoteStorage` は保存した f32 行列で引き、`ReevaluateKernel` はカーネルを f64 で計算し直す。両方を残す。省略は `PromoteStorage`。フラグと、コード上の別名は置かない。Forrester `n=1024` の release 中央値は、保存した f32 行列で引く型が 22.40 ms、カーネルを f64 で計算し直す型が 48.77 ms で、5% の外である。対象は Exact、`Sgpr`、`Svgp` と、f64 が既に持つオンライン。最適化は f64 にあるものすべて。P5-2（[#40](https://github.com/YUKIKEDA/gprx/issues/40)）。

### 4.2 混合精度反復改良の収束判定パラメータ

古典的な反復改良理論(Higham)より、分解精度u_f(f32≈1.19×10⁻⁷)と改良精度u_r(f64≈2.22×10⁻¹⁶)を使う場合、収束速度はκ(A)·u_fに依存する。**ただし実際の収束判定は理論値ではなく実測残差で行う**。

パラメータは公開の設定ではなく、`src/precision/refine.rs` の crate 内定数に固定する。

| パラメータ | 値 |
| --- | --- |
| 補正の最大回数 | 10 |
| 相対許容 | `10 · dim · u_r`（`u_r = f64::EPSILON`）。判定は実測残差 |
| 停滞 | 残差ノルム比が `0.9` 超えを2回連続 |
| 不収束 | 同じ系の `f64` の解（エラーにはしない） |

収束判定: `||r_k||∞ / (||B||∞ ||w_k||∞ + ||b||∞) < 10 · dim · u_r`。1つのループ（`RefineSystem` に対する `refine`）が Exact の `α`、Sgpr の重み、Svgp の三角 solve を扱う。各系は残差、保存済み因子での solve、`f64` へのやり直しを与える（R3-3、[#238](https://github.com/YUKIKEDA/gprx/issues/238)）。反復改良は収束しないエラーを返さない。やり直しは常に `f64` の解。

**IR不収束時にjitterを増やさない**: 分解側のjitterだけを増やすと、前処理`LLᵀ`と目標`A`の乖離が拡大してIRが発散し得る。IR不収束は `f64` の解に落とす。jitter適応は§4.0の通りCholesky失敗時専用とする。

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
```

P5-1 の実体: 葉のパラメータは f64 のまま。公開の計算スカラーは `CompiledKernel<T = f64>`。型を省略した `compile()` は f64。`compile_as::<T>()` は apply・勾配・ヘッセ・組み込みの葉・和・積・ユーザー定義の葉を f32 と f64 で同じ操作にする。f64 の距離キャッシュと SIMD は f64 側。f32 は同じ式のスカラー。f32 と f64 の入れ替え変換は置かない。

```rust
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
    Cached,   // 既定。fit で 1 回埋めて使い回す
    Uncached, // カーネルを組むたびに X から計算し直す
}
```

理論的な参考値(目安であり決定基準ではない): `(n,n,d)`テンソルは`n²×d×sizeof(T)`バイト。基本のK行列自体もn²×sizeof(T)であり(例: n=5000,f64で約200MB)、ARDキャッシュはこれのd倍になる点に注意。d≪nの典型的GPRではキャッシュの投資対効果は薄いことが多い。`Auto`の具体的な閾値は実装後のベンチマークで決定する(§14)。`Auto` はまだ variant ではない。

P2-2（[#26](https://github.com/YUKIKEDA/gprx/issues/26)）: `Cached` / `Uncached` は既存の `Workspace.dist_cache`（等方 Dist/Either の n×n）に載せた。デフォルトは `Cached`。P5-5 で `Auto` を足す。

P2-7（[#88](https://github.com/YUKIKEDA/gprx/issues/88)）: 同じ `DistanceCachePolicy` を ARD 葉の生の `(Δx_d)²` に載せる。ℓ 込みの `r²` は置かない。公開 Policy は増やさない。`Workspace` は `n` と `d` を見る。Always（`Cached`）の ARD fit で 1 回確保し、等方 / Never（`Uncached`）では空（`kernel_scratch` と同じ）。レイアウトは列優先 `n × (n·d)`、次元 `k` は列 `[k n, (k+1) n)`、各ブロックは下三角。埋めと RBF ARD `apply`/`grad` は Rayon + `wide::f64x4`（単位行ストライド）。Matérn / RQ ARD は同じキャッシュをスカラーで読む。必須の数値は同じ固定問題の ARD RBF（`mll_and_grad_ard` / `fit_lbfgs_ard`、Always vs Never）。`Auto` は P5-5。train×test / LOO のキャッシュは P2-7 の対象外。Dist 葉と Points 葉の合成の評価は P2B-13（P2-7 ではキャッシュ経路を混ぜない）。

### 5.3 CompiledKernelのplan構築アルゴリズム

Sum/Productは結合則・交換則が効くため、flatten+fold評価で済む。公開の `KernelSpec *` は Dist 葉でも Points 葉でも `grad` まで通る。Dist 葉と Points 葉の Sum/Product（例: `RBF + Linear`）は混ぜて評価する。`coord_mode` は `Mixed` を返し、実行時エラーや型で混ぜを禁止しない。葉は Dist が距離、Points が座標のまま。

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

**対応方針**: fit が触られた葉だけを作り直すかどうかは型にしない。実行時に `O::USES_CHANGE_INDICES && buffer == CholeskyBuffer::Retain` から決める（R4-1 / [#239](https://github.com/YUKIKEDA/gprx/issues/239)）。葉の作り直しの本体は P2B-18（[#110](https://github.com/YUKIKEDA/gprx/issues/110)）。Exact GPR では Cholesky が O(n³) のため、部分更新の恩恵はカーネル行列構築にだけ及ぶ。

```rust
trait Optimizer<P> {
    const USES_CHANGE_INDICES: bool = false; // FSA は true
    // ...
}

trait IncrementalObjective: Objective {
    fn value_with_changes(&mut self, params: &[T], indices: &[usize]) -> Result<T, GprError>;
}
```

変更 index は `IncrementalObjective::value_with_changes` の `&[usize]`。最適化器の受理済みの点からではなく、その目的関数で直前に評価した点から変わった座標をすべて並べる。棄却のあと FSA は戻した座標と新しい座標の両方を渡す（R4-5 / [#243](https://github.com/YUKIKEDA/gprx/issues/243)）。作り直す葉は index だけで決める。設計旧稿の `ChangeSet { Vec<usize> }` と、θ の数値差分による推測は置かない。列に無い変更は誤った値を黙って返さず `IndexOutOfRange` にする。空・重複・`i >= n_params` は境界で `GprError`。フル再計算は `Objective::value`。`Objective::value_at_changes` の既定は `value`。`GprObjective` はどの最適化器・バッファでも `IncrementalObjective` を impl する。`value` / `value_at_changes` が葉の経路を通るのは上のフラグが立つときだけで、それ以外は一括の値と勾配を計算する。

`with_recompute_strategy` は無い。`CholeskyBuffer::Reuse` は常に全体を作り直す。`CholeskyBuffer::Retain` は最適化器が `USES_CHANGE_INDICES` を立てるとき葉を作り直す。`with_optimizer` / `refit` は新しい最適化器からフラグを決め直す。`Gpr<Fixed>::factor` は一発フル。L-BFGS / NCG / Nelder–Mead / Newton は既定の `false`。FSA は `true`。初回とリスタートは `value`、座標一歩は `value_at_changes`。

葉の作り直しはコンパイル済み葉だけをキャッシュし、変更 index が触る葉だけ `apply` し直す。葉ごとの Gram（葉 `L` 個で `L · n²`）と、再利用する dirty の印・直前の `θ` は、1回の `fit` / `refit` のあいだ `GprObjective` が持ち、Workspace には置かない。木の結合と **Cholesky は毎回フル**。低ランク更新はしない。Workspace に新しい `n×n` は足さない。実行時の NotImplemented は置かない。

#### 5.4.1 葉の作り直しとfaer update APIの関係

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
    fn inverse_apply(&self, x: MatMut<f64>);
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
struct MinMaxInput { /* per-column min/max, default range [0, 1] */ }
struct MinMaxTarget { /* y min/max, default range [0, 1] */ }
/// P2B-8: 長さ d。列ごとに Identity / Standardize / MinMax / 自前
struct ColumnwiseInput { maps: Vec<Box<dyn Transform>> }
```

既定の `Gpr` は Identity。平均関数が零のときは `StandardizeTarget` が数値安定の基本。`MinMaxInput` / `MinMaxTarget` は区間スケール（既定 `[0, 1]`）。未学習の `transform` / `apply` は型で起きない。複数マップの直列は `Pipeline`（`X`）と `TargetPipeline`（`y`）。1 段だけの `with_*` はそのまま残る。入力は列ごとに `ColumnwiseInput`（一様な列は MinMax、正規に近い列は Standardize。長さが `d` でないときはエラー）。`src/transform/` は `input.rs` / `target.rs` / `pipeline.rs` / `columnwise.rs`。葉ファイルに分けるかは P2B-20（[#116](https://github.com/YUKIKEDA/gprx/issues/116)）。行数ではなく、独立したアダプタかどうかで判断する。`predict`は内部で潜在/観測分散を計算したあと、`inverse_transform_mean`/`inverse_transform_variance`を通してから返す。分散の逆変換はアフィン `y' = (y - a)/s` なら `Var(y) = s² Var(y')`。

`Sgpr` / `Svgp` も同じ変換を同じ既定（Identity）で受ける（R5-3、[#281](https://github.com/YUKIKEDA/gprx/issues/281)）。誘導点 `Z` は `X` と同じ座標で渡し、`X` と一緒に学習時の入力の写像を通す。クエリと、`OnlineSgpr` に足す点は、学習時に当てはめた写像を通す。`FreeInducing` は写像後の座標で `Z` を探す。学習後のモデルは、必須メソッドの `Transform::inverse_apply` で `Z` を元の座標に戻して返す。入力の写像はすべて逆を持つ。

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

`Gpr`（Exact）と `Sgpr`（P4-2）がそれぞれ学習済み型を返す。ハイパラ最適化は`Objective`(§9)を介して `fit` 中だけ扱う。

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

対角分散は Phase 1 から既定。クエリ間の共分散と posterior sample は P2B-6 のオプトイン（既定では計算しない）。`predict`は`PredictOptions`で分散の意味を切り替える。未指定時は`Observation`(ユーザーが欲しいのは多くの場合ノイズ込みの予測分散)。

### 6.1 Sparse GPRの誘導点キャッシュ問題

Sparse 近似は VFE。理由は [ADR 0002](adr/0002-sparse-vfe.md)。FITC は載らない。SVGP は別公開型（`Svgp` / `FittedSvgp`）。理由は [ADR 0006](adr/0006-sparse-svgp.md)。`Svgp<Fixed>::factor` が呼び出し側の `Z` で `K_mm` を LLT し、whitened の `q(u)` を prior（平均 0、`L = I`）で置く。`Svgp<Adam>::fit` が同じ prior からミニバッチ Adam でカーネル `θ`・尤度 `θ`・whitened `q` を動かす。`Adam` は `Optimizer` ではない。`FittedSvgp` は対角の `predict` / `predict_with`（と `predict_into` / `predict_with_into`）、`neg_elbo`、全データ `value_and_gradient_into` を返す。最適 `q`（Titsias）では同じ `θ`・`X`・`Z` の `FittedSgpr` と一致する。公開型は `Sgpr` / `FittedSgpr`。既定は `Sgpr<Lbfgs, FixedInducing>`。`fit` がカーネルと尤度の `θ` を探し、`Sgpr<Fixed, I>::factor` が呼び出し側の誘導点 `Z` で `K_mm = k(Z, Z)` を LLT する。既定では `Z` は params に入らない。`with_inducing(FreeInducing)` の `fit` はカーネル `θ`・尤度 `θ`・列優先 `Z` を同じ `Optimizer` が同時に動かす。`FittedSgpr` は対角の `predict` / `predict_with`（と `predict_into` / `predict_with_into`）、`neg_log_marginal_likelihood`（VFE の負の ELBO）、`value_and_gradient_into`、`hessian_into`（row-major `p×p`）を返す。`Z = X` のとき Exact の `Gpr<Fixed>::factor` と一致する。k-means は置かない。バッチの外部照合は P4-11（同じ初期 θ の GPyTorch 潰し SGPR / whitened prior SVGP、相対 `1e-8`）。オンラインの外部照合は P4-13（同じ初期 θ の GPyTorch 潰し SGPR、相対 `1e-8`。`OnlineSgpr` の insert / delete / insert_inducing / delete_inducing。各段階はフル再組み立て）。バッチの時間・RSS は P4-12（`just perf-sparse`。GPyTorch / GPy。CPU。正しさゲートは置かない）。オンライン時間は P4-14（`just perf-sparse-online`。自前 `Sgpr<Fixed>::factor` と GPyTorch Titsias 組み立て。CPU。正しさゲートは置かない）。

`K(X,X)`対角は不変なので1回計算・流用。`K(X,Z)`, `K(Z,Z)`はZが動くたびに再計算が必要だが、m(誘導点数)が小さいためCholeskyのO(nm²)に対して無視できるコストであり、キャッシュ対象にせず毎回再計算する。joint の `K(X,X)` 勾配とヘッセは対角 `∂k(x_i, x_i)/∂θ` を `O(n)` で足す。`K(Z,Z)` と `K(Z,X)` の勾配は密行列のまま。

誘導点座標の勾配は`grad_wrt_coord_dim`(§5.1)で扱い、未対応カーネルはpanicではなく`GprError::CoordGradientUnsupported`を返す。既定の `FixedInducing` の `fit` はこの API を使わない。`FreeInducing` は同時最適化で次元一括で呼ぶ。理由は [ADR 0003](adr/0003-sparse-z-joint.md)。

**既定は呼び出し側が Z を渡し、最適化対象はカーネルハイパラとノイズのみとする。** 自由 Z は `FixedInducing` / `FreeInducing` で切り替え、カーネル `θ`・尤度 `θ`・列優先 `Z` を同じ `Optimizer` が同時に動かす。区間は訓練 `X` の箱を少し開いて広げた生座標。L-BFGS 履歴の長さは `p = p_θ + m×d` で、増分は `history_size × m × d` 個の `f64`（`m` が小さいので VFE の `O(nm²)` に対して小さい）。交互は載らない。

オンラインは X と誘導点を増減できる。`FittedSgpr::into_online` が `OnlineSgpr<O>` を返す（誘導 typestate は無い）。`insert` / `delete` は ADR 0004 の rank-1 で VFE 因子を更新する。`insert_inducing` / `delete_inducing` は [ADR 0005](adr/0005-sparse-inducing-update.md)（insert は bordered LLT、delete は trailing cholupdate）。識別子は `InducingId`。座標は呼び出し側。`Z` は params に入らない。`set_params` と `refit` はフル再 assemble。

**Exact との機能差**（R5-2 / [#247](https://github.com/YUKIKEDA/gprx/issues/247)）。Sparse のモデルは、次の機能で `Gpr` に揃える。まだ入っていない行は、それぞれ行と Issue を持つ。

| 機能 | `FittedSgpr` / `OnlineSgpr` | `FittedSvgp` | 行 |
| --- | --- | --- | --- |
| 入力 / 目的変数の変換 | 揃える | 揃える | R5-3（[#281](https://github.com/YUKIKEDA/gprx/issues/281)） |
| `K_mm` の `JitterPolicy`（既定は `adaptive(1e-8, 10, 5, 1e-3)` のまま。`K_mm` にはノイズが入らないため、§4.0） | 揃える | 揃える | R5-4（[#282](https://github.com/YUKIKEDA/gprx/issues/282)） |
| warmup 後に確保しない `predict_into` | 揃える | 揃える | R5-5（[#283](https://github.com/YUKIKEDA/gprx/issues/283)） |
| 予測共分散と posterior sample | 揃える | 揃える | R5-6（[#284](https://github.com/YUKIKEDA/gprx/issues/284)） |
| LOO | 揃える | 載せない | R5-7（[#285](https://github.com/YUKIKEDA/gprx/issues/285)） |
| 保存・読み込み | 揃える | 揃える | R5-8（[#286](https://github.com/YUKIKEDA/gprx/issues/286)） |

Sparse の `predict_covariance` / `sample`（R5-6、[#284](https://github.com/YUKIKEDA/gprx/issues/284)）は `predict` の経路を通し、そのバッファから非対角を作る。VFE は `K** − A*ᵀA* + σn² S*ᵀS*`（`A* = L_mm⁻¹ K_m*`、`S* = L_B⁻¹ A*`）、SVGP は `K** − AᵀA + UᵀU`（`U = L_qᵀ A`）。対角は `predict` の分散とビットで一致する。標本は Exact と同じ引き方（`μ + L z`、因子にはモデルの `JitterPolicy`）。

SVGP は LOO を持たない。collapsed VFE の `q(u)` は閉じた形の最適解なので、点 `i` を除くのは `A = K_mm + σ⁻² K_mn K_nm` の rank-1 の downdate になる（θ と `Z` を固定して 1 点 `O(m²)`）。SVGP の `q(u)` は、ミニバッチの Adam が全点に合わせた変分パラメータになっている。点を除くには `q(u)` を合わせ直す必要があり、閉じた形は無い。学習済みの `q(u)` をそのまま使っても LOO の予測にはならないので、その名前では出さない。

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
5. `L`から`K⁻¹`を計算する(三角ソルブで `L Lᵀ X = I`、O(n³)が1回)
6. `W[i,j] ← α[i] α[j] - K⁻¹[i,j]`(対称なので下三角のみ)
7. 各θ_iについて `∂K/∂θ_i` を`exp_buf`へ評価し、`⟨W, ∂K/∂θ_i⟩_F` をO(n²)で積算。カーネルパラメータは`KernelTerm::grad`、ノイズは`Likelihood::noise_grad_diag`(対角のみ)

全体コストはO(n³ + p n²)。K⁻¹をパラメータごとに作り直さない。

解析 NLML ヘッセ（P2B-17 / [#109](https://github.com/YUKIKEDA/gprx/issues/109)）:

```
H_ij = -½ ⟨W, ∂²K/∂θ_i∂θ_j⟩ - ½ Tr(K⁻¹ K_i K⁻¹ K_j) + αᵀ K_i K⁻¹ K_j α
```

`KernelTerm::hess` / `hess_points` が `(i, j)` 1 組の `∂²K` を書く。Custom・Sum/Product も解析。`FittedGpr::hessian_into` が公開口で、`GprObjective` は `TwiceDifferentiable` へ転送する。`Q_j`（`n×n` 1 枚）と長さ n のベクトル 4 本は `WorkspaceCore::hessian` に置く。最初の Hessian まで空で、以後は使い回すので、2 回目以降の Hessian は確保しない（R4-5c / [#272](https://github.com/YUKIKEDA/gprx/issues/272)）。`CholeskyBuffer::Reuse` は ⟨W, K_ij⟩ のあと Chol し直して一次項の `Q_i = K⁻¹ K_i` を解く。

`value_and_gradient_into`はこの手順を一度で実行し、Lとαと`exp_buf`を尤度・勾配で共有する。デフォルト実装の`value`→`gradient_into`の二段呼びでは共有されない。

既定の `CholeskyBuffer` は `Retain`。専用の `w_matrix` に `K⁻¹` → `W` を書き、`L` は `k_matrix` に残す。速さは変えない。公開のメモリ極（`with_prefer_memory`）が `CholeskyBuffer::Reuse` を選ぶ。`with_cholesky_buffer` はこれだけを設定する。`Reuse` は `K⁻¹` を `exp_buf` で解き、`W` を Cholesky 領域へ書く。最適化ループの途中では `L` を戻さない。`fit` の末と単独の `value_and_gradient_into` の末で Cholesky し直す。persist にこの方針は書かない。`load` は `Retain`。

### 6.3 Exact GPR (`Gpr` / `FittedGpr`)

公開面はトレーナーと学習済みモデルを分ける（P2-8）。

`Gpr<O = Lbfgs, P = DoublePrecision>` は `KernelSpec`・`GaussianLikelihood`・変換と、最適化器 `O`、実行時の方針 3 つ（R4-1 / [#239](https://github.com/YUKIKEDA/gprx/issues/239)）を持つ：`DistanceCachePolicy { Cached, Uncached }`、`CholeskyBuffer { Retain, Reuse }`、`KernelExp { Accurate, FastApprox }`。どれも素の enum。不正な組み合わせが無いので型パラメータにしない。`Gpr::new` の既定は速さ極（`Cached` + `Retain`）で `Accurate`。公開の切り替えは `with_prefer_memory` / `with_prefer_speed`。メモリ極は `Uncached` + `Reuse`。`with_distance_cache_policy` / `with_cholesky_buffer` / `with_math` はそれぞれ 1 つを設定する。対距離を読まないカーネル（単独の Linear / Constant / White）は方針によらず距離キャッシュを確保しない。`from_points` は無い。`FittedGpr` に `with_prefer_*` は無い（`into_trainer` → prefer → `refit`）。3 つの方針は getter で読める。型は crate ルートに残す。`Gpr<O: Optimizer>::fit(self, …)` が `O` でハイパラを動かし、成功時に `FittedGpr<O, P>` を返す。固定ハイパラは `Gpr<Fixed>::factor`（旧 `FitOptions::FIXED`）。`optimize: bool` は置かない。失敗時は消費した `Gpr<O, P>` をエラーと一緒に返す。`fitted: bool` と [`GprError::NotFitted`] は置かない。未学習の `transform` / `apply` は型で起きない（`StandardizeTarget::fit(self)` が `FittedStandardizeTarget` を返す）。公開 `FittedGpr` の `L` / `α` / `X` / compiled は `Option` にしない（P2B-5）。欠けるときに `EmptyInput` を返さない。

`FittedGpr` は推論に必要な `L`・`α`・訓練 `X`・カーネル・尤度・変換を持つ。勾配用の `W`・`∂K`・argmin 状態は `fit` のあいだだけ生き、学習済み値には残さない。同一プロセスで `fit` の直後に `predict` する経路は少数派とみなす。学習済みモデルを渡すのが主経路なので、推論オブジェクトは `FittedGpr` である。

`FittedGpr` と `OnlineGpr` は crate 内部の `GprCore`（kernel spec・コンパイル済みカーネル・尤度・変換・方針・訓練データ・`α`・クエリ用バッファ）を共有し、違うのは因子だけ（R4-2 / [#240](https://github.com/YUKIKEDA/gprx/issues/240)）。`FittedGpr` は LLT の置き場（`FitBuffers` か mmap の `L`）、`OnlineGpr` は LDLT の `LdltStore` と `PointId` の表を持つ。`StoredFactor { Llt, Ldlt }` が因子の見方で、`solve`、列への `L⁻¹`、ピボットごとの重み（`1` か `1/Dᵢ`）、`log|A|`、`diag(A⁻¹)` を持つ。predict・共分散・sample・LOO・NLML・予測用 `α` はこの見方について `GprCore` に1回だけ書く。ハイパラの書き込み（`set_params`・勾配・Hessian・`fit`・`refit`）はすべて借用の `ExactFit`（core + LLT の置き場）1つで動く。`OnlineGpr` は `L √D` を O(n²) で詰めた一時的な LLT の置き場を貸し、新しい因子を書き戻す。訓練データは複製せず、O(n³) は新しい `θ` が要る分解だけ。

既定の `Gpr` は `Gpr<Lbfgs>`。`with_optimizer` が `O` を差し替える（P2B-1）。argmin の `NonlinearCg` / `NelderMead` は P2B-2。argmin の `Newton` は P2B-17。葉の作り直しは §5.4 に従う（P2B-18）。`with_recompute_strategy` は無い。`Gpr<Fixed>::factor` は分解だけ。`FittedGpr::predict` の既定は対角分散。クエリ間共分散は P2B-6 の別経路（対角 `predict` のフラグでは切り替えない）。`loo_predict` は GPML 5.4.2 の `L` と `α` から訓練点ごとの LOO を返す。ハイパラを変えて同じデータで分解し直すのは `FittedGpr::refit`（学習済みが持つ `O` のまま）。`with_optimizer` / `factor` / `into_trainer` / `refit` は方針を保つ。

```rust
struct Gpr<O = Lbfgs, P = DoublePrecision> {
    kernel: KernelSpec,
    likelihood: GaussianLikelihood,
    x_transform: Box<dyn InputTransform>,
    y_transform: Box<dyn TargetTransform>,
    optimizer: O,
    policies: Policies,
    _precision: PhantomData<P>,
}

struct Policies {
    distance_cache: DistanceCachePolicy, // Cached（既定）| Uncached
    cholesky_buffer: CholeskyBuffer,     // Retain（既定）| Reuse
    math: KernelExp,                     // Accurate（既定）| FastApprox
    jitter: JitterPolicy,
}

struct Fixed;

struct Lbfgs {
    max_iterations: u64,   // 既定 100
    tolerance: f64,
    history_size: usize,   // 既定 10。L-BFGS だけ
    n_restarts: u32,       // 既定 0
}

struct Newton {
    max_iterations: u64,
    tolerance: f64,
    gamma: f64,            // 既定 1。Newton だけ
    n_restarts: u32,
}

struct NonlinearCg {
    max_iterations: u64,
    tolerance: f64,
    n_restarts: u32,
}

struct NelderMead {
    max_iterations: u64,
    tolerance: f64,
    n_restarts: u32,
}

enum DistanceCachePolicy { Cached, Uncached } // 距離モードの経路だけ（P2B-11）
enum CholeskyBuffer { Retain, Reuse }
enum KernelExp { Accurate, FastApprox }

struct FittedGpr<O = Lbfgs, P = DoublePrecision> {
    kernel: KernelSpec,
    likelihood: GaussianLikelihood,
    x_transform: Box<dyn InputTransform>,
    y_transform: Box<dyn TargetTransform>,
    optimizer: O,
    policies: Policies,
    workspace: FitBuffers<P>, // L。dist / W は方針しだい
    query: QueryWorkspace<DoublePrecision>, // predict_into 用
    compiled: CompiledKernel,
    alpha: Vec<f64>,
    x: Mat<f64>,
    y: Vec<f64>,
    n: usize,
    d: usize,
}

/// Objective は `Gpr` を fit 中だけ &mut で借り、set_params → MLL/勾配 を中継する。
/// パラメータの正本は Gpr.kernel / Gpr.likelihood。
struct GprObjective<'a, O, P = DoublePrecision> {
    model: &'a mut FittedGpr<O, P>,
    incremental: bool, // §5.4
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

既定の距離キャッシュ方針は `Cached`。`fit` 開始時に訓練点の二乗距離を一度埋め、以降のハイパライテレーションではカーネルだけを書き換える。等方は `n×n`。ARD は生の `(Δx_d)²` を `n × (n·d)` に置く（P2-7）。`Uncached` はそれらのテンソルを Workspace に置かず、等方も ARD も `X` から距離を計算する。公開のメモリ極は `with_prefer_memory`（`Uncached` + `Reuse`）。速さ極は既定のまま（`with_prefer_speed`）。P2B-21 の libgp 比 RSS 合否は `Uncached` + `Retain`（`with_distance_cache_policy` だけで作る）。キャッシュを確保するのはコンパイル済みカーネルが距離を読むときだけ。`RBF + White` と `Constant * RBF` は読む。単独の Linear / Constant / White は読まず、方針は保つが使わない。persist タグは `always` / `never`（タグが無ければ `Cached` で読む）。`LoadedGpr` は精度と分解の種類ごとに 1 つの variant（8 つ）。どちらもモデルの型パラメータだから。`predict` / `predict_with`（`f64` に広げる）、`n`、`d`、`is_online` は match せずにどの variant でも使える。variant を match するのは型つきのモデルが要るとき（`predict_into`、`insert`、`refit`）だけ（R4-6 / [#244](https://github.com/YUKIKEDA/gprx/issues/244)）。`load` は `Retain`。

### 6.4 Leave-one-out(P1B-7)

Exact GPR の leave-one-out は、学習後の `L` と `α` から閉じた式で出る(Rasmussen & Williams, GPML §5.4.2)。`A = K + σn² I`、`Q = A⁻¹`、`α = A⁻¹ y` として

```
μ_i = y_i - α_i / Q_ii
σ_i² = 1 / Q_ii
```

これは観測の `p(y_i | X, y_{-i}, θ)`。潜在 `f_i` の LOO 分散は `max(0, 1/Q_ii - σn²)`。`Q_ii` は下三角 `L` から `L⁻¹` の列ノルムで取る(`A⁻¹ = L^{-T} L^{-1}`)。コストは Cholesky と同オーダーの O(n³)、追加メモリは `n×n` の一時行列。Phase 1b の n=16 / 36 では問題にならない。

`FittedGpr::loo_predict` は学習点と同じ長さの `Prediction` を返す。既定は `VarianceKind::Observation`。平均・分散は `predict` と同じく `TargetTransform` で元スケールへ戻す。White 葉は使わず、ノイズは `GaussianLikelihood` のみ。

sklearn に LOO API は無い。`just gen-goldens` は fit 後の `L_` / `alpha_` に同じ GPML 式を適用して JSON に書く。Rust 側は sklearn が選んだ `θ` で `FitOptions::FIXED` して照合する(最適化器差を LOO に混ぜない)。

## 7. Workspaceとメモリ管理

### 7.1 個別バッファ構造

バッファ数は少数・固定なので、個別フィールドとして持つ。精度ポリシーのStorage/Refineを明示的に反映する。

```rust
struct WorkspaceCore<P: PrecisionPolicy> {
    k_matrix: Mat<P::Storage>,       // K → Cholesky後は L。Reuse の勾配中は W
    exp_buf: Mat<P::Storage>,        // カーネル評価、∂K/∂θ。Reuse の n-RHS はここ
    kernel_scratch: Mat<P::Storage>, // product / custom の `∂K/∂θ`。等方 RBF では空
    thread_scratch: Vec<Mat<P::Storage>>, // Rayonスレッド数ぶん事前分割
    rhs: Mat<P::Storage>,            // n×1、訓練 Cholesky の右辺 y → α
    faer_scratch: MemBuffer,         // faer公式のスクラッチ機構をそのまま使う
    nested: Vec<Mat<P::Storage>>,    // 和・積の中の和・積の入れ子 1 段に n×n 1 枚。無ければ空
    hessian: HessianScratch<P::Storage>, // Q_j と長さ n のベクトル 4 本。最初の Hessian まで空(§6.2)
}

struct FitBuffers<P: PrecisionPolicy> {
    core: WorkspaceCore<P>,
    dist: Option<DistCache<P::Storage>>, // 距離カーネルで Cached のとき Some
    w_matrix: Option<Mat<P::Storage>>,   // Retain のとき Some。W = ααᵀ - K⁻¹(§6.2)
}

struct DistCache<S> {
    dist_cache: Mat<S>,
    ard_sq_diff: Mat<S>,             // 等方では 0×0
}

// FittedGpr が保持。predict_into の warmup で (n, m, d) に合わせる
struct QueryWorkspace<P: PrecisionPolicy> {
    query_xs: Vec<f64>,              // 変換後クエリ（列優先）
    query_x: Mat<P::Storage>,        // m×d
    query_k_star: Mat<P::Storage>,   // n×m
    query_scratch: Mat<P::Storage>,
    query_nested: Vec<Mat<P::Storage>>, // n×m ブロックの入れ子の和・積の段
    query_dist: Mat<P::Storage>,
    query_kss: Vec<f64>,
}
```

和・積の項がさらに複数項の和・積のときは、入れ子 1 段ごとに出力と同じ形のバッファがもう 1 枚要る（`CompiledKernel::nested_depth`）。crate 内の fit / predict の入口は、その段を `nested` / `query_nested` から借りる。最初の呼び出しで伸ばし、以後は使い回す（R4-5c / [#272](https://github.com/YUKIKEDA/gprx/issues/272)）。公開の `CompiledKernel::apply` / `grad` / `hess` などはシグネチャを変えず、その呼び出しのぶんだけ段を用意する。対角の畳み込み（`fill_diag`、`fill_diag_points` と、その勾配・Hessian）は固定長のスタック上の行ブロックで項を合わせ、確保しない。Sparse のモデル（`FittedSgpr`、`OnlineSgpr`、`FittedSvgp`）は、カーネルのスクラッチ（出力と同じ形のスクラッチ、入れ子の段、訓練–クエリの距離）を crate 内の `SparseScratch` に持ち、`&mut self` の呼び出し（`set_params`、勾配、Hessian、オンライン更新）のあいだ使い回す。学習の因子は今も新しい行列で返すので、`tests/alloc.rs` はゼロではなく測った数をラチェットにする（R5-1d / [#246](https://github.com/YUKIKEDA/gprx/issues/246)）。`predict_into` の予測のバッファもそこに持つ。写したクエリ、詰めた `Z` とクエリ、`K(Z, X*)` とその解、コンパイル済みのカーネル（カーネルが変わったときだけ作り直す）。丸める精度（`f32`）は `f64` のバッファで予測する（`K_mm` を `f64` で分解し、`B` を昇格する）。同じ形で warmup した後は確保しない。例外は混合精度の SVGP の平均で、クエリごとの refine が長さ `m` の `f64` のベクトルを 3 本持つ。その `f64` の参照は呼び出しごとに 1 回作る（R5-5 / [#283](https://github.com/YUKIKEDA/gprx/issues/283)）。`predict`（`&self`）は自分のバッファで同じ経路を通るので、両方とも同じ値を返す。

fit 用バッファは`fit`開始時にサイズが確定するため、`reserve_exact`で一度だけ確保(または`Mat::zeros`で1回構築)し、以降のイテレーションでは同じ領域に上書きする。query バッファは `FittedGpr` の `QueryWorkspace` が持ち、最初の `predict_into` で `(n, m, d)` に合わせ、同じクエリ長では再利用する。`predict(&self)` は出力 `Vec` を毎回確保してよい。あわせて、faer公式の`PodStack`/`MemStack`をスクラッチ管理に採用し、自前でスクラッチ領域をアリーナに内包する設計はやめる。

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

バッチfitのWorkspaceはn固定。オンライン学習の容量成長は`LdltStore`(§11)が担当し、バッチ用Workspaceとはメモリ管理方針を分ける。`Workspace`、`QueryWorkspace`、`LdltStore`、faer の型はクレート私有。

## 8. 並列化・SIMD、数学関数バックエンド

- カーネル評価内側ループは `wide::f64x4` でベクトル化する（P2-5 / P2-7）。対象は列優先・単位行ストライドの等方 RBF `apply` / `grad` / `apply_cross`、ARD RBF `apply` / `grad`、二乗距離と `(Δx_d)²` の行ループ。ストライドが 1 でないビューはスカラーに落とす。`std::simd` は安定化まで使わない。Matérn / Periodic / RQ の内側は未導入。
- 距離行列・カーネル行列構築はRayonでブロック並列化
- faer自身もRayon並列化されるため、外側との二重並列化に注意。単一の`rayon::ThreadPool`を共有。faer の本数は `min(プール, n/64, n·k/16384, k/12)`（[ADR 0001](adr/0001-faer-parallel-degree.md)）。`k` は RHS 列。カーネル埋めはプール全部

**MathBackendは最小限のAPIから始め、デフォルトは近似ではなく正確な実装にする**。カーネル行列の近似誤差は正定値性・Cholesky安定性・尤度・勾配・予測値すべてに波及するため。

```rust
trait MathBackend<T: Scalar>: Send + Sync {
    fn exp_inplace(&self, buf: &mut [T]); // 最初はexpのみ。erfは実際に必要になったカーネル(probit尤度等)が出てから追加
}
enum MathMode { Accurate, FastApprox }
```

デフォルトは `Accurate`（`f64::exp` / `f32::exp` / `wide::exp`）。`FastApprox` はカーネル評価の `exp` を `fit` も含めて置き換える（P5-4 / [#42](https://github.com/YUKIKEDA/gprx/issues/42)）。長さスケールへ戻す `exp(θ)` と `KernelTerm` の式は正確な `exp` のまま。すべてのモデル（`Gpr` / `FittedGpr` / `OnlineGpr`、`Sgpr` / `FittedSgpr` / `OnlineSgpr`、`Svgp` / `FittedSvgp`）で、モードは実行時の enum `KernelExp` で、`with_math(KernelExp::FastApprox)` で設定する。カーネル呼び出しごとに sealed な `KernelMath` の印へ 1 回だけ分岐する（R4-1 / [#239](https://github.com/YUKIKEDA/gprx/issues/239)、R5-1a / [#246](https://github.com/YUKIKEDA/gprx/issues/246)）。上の `MathMode` 列挙は置かない。`FittedGpr` と `OnlineGpr` の save はモードを記録し、欄が無いファイルは `Accurate`。

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

`init`はスライスにする(呼び出し側のVecを消費しない)。`Gpr`の`GprObjective`は`value_and_gradient_into`をオーバーライドし、§6.2の手順でL・α・W・`exp_buf`を共有する。`GprObjective` は `TwiceDifferentiable` を impl し、`hessian_into` は `FittedGpr` へ転送する。公開面は `Gpr<O: Optimizer>`。既定は `Lbfgs`。argmin の他ソルバもユーザー実装も `with_optimizer` で同じ型パラメータを差し替える。公開 `Newton` は argmin の `Newton`（`H⁻¹` は faer の私有型。logit は L-BFGS と同じで `H_z` は解析連鎖。ノブは共有 3 つ + `with_gamma`）。自作例は `FastSimulatedAnnealing`（Cauchy / Metropolis。P2B-15 / [#106](https://github.com/YUKIKEDA/gprx/issues/106)）。logit は使わず、`minimize` が受け取る log-`θ` を歩く。`FitOptions::solver` と custom を並べて片方を無視する設計はしない（`.cursor/rules/types.mdc`）。準ニュートンを gprx が自前実装しない。目的関数の能力は `Objective`（value）⊂ `Differentiable` ⊂ `TwiceDifferentiable`。実行時の NotImplemented は置かない。部分更新は `IncrementalObjective::value_with_changes(params, indices: &[usize])`。`GprObjective` が impl する（P2B-18 / [#110](https://github.com/YUKIKEDA/gprx/issues/110)）。最適化器は `Optimizer::USES_CHANGE_INDICES` で葉の作り直しを選ぶ（§5.4）。`Objective::value_at_changes` の既定は `value`。FSA の座標一歩がそれを呼ぶ。`ChangeSet` 構造体は置かない。

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
    #[error("入力に非有限値(NaN/Inf)が含まれます")]
    NonFiniteInput,
    #[error("カーネル評価結果に非有限値が含まれます")]
    NonFiniteKernelValue,
    #[error("Cholesky分解に失敗しました(段階={stage:?}, サイズ={matrix_size}, jitter={jitter}を適用済み)")]
    CholeskyFailed { jitter: f64, matrix_size: usize, stage: CholeskyStage },
    #[error("行列が半正定値ではありません")]
    NonPositiveDefiniteMatrix,
    #[error("このカーネル項はSparse GPR用の座標微分(grad_wrt_coord_dim)を実装していません")]
    CoordGradientUnsupported,
    #[error("最適化が収束しませんでした({iterations}回反復後)")]
    OptimizationNotConverged { iterations: usize },
    #[error("ハイパーパラメータが不正です: {reason}")]
    InvalidHyperparameter { reason: String },
    #[error("shape mismatch: {reason}")]
    ShapeMismatch { reason: String },
    #[error("length mismatch: {reason}")]
    LengthMismatch { reason: String },
    #[error("index out of range: {reason}")]
    IndexOutOfRange { reason: String },
    #[error("invalid configuration: {reason}")]
    InvalidConfig { reason: String },
    #[error("size overflows usize")]
    SizeOverflow,
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

`InvalidHyperparameter` はハイパーパラメータの値が定義域の外にあるときだけに使う。行列の形状・スライス長・添字の誤りは `ShapeMismatch`・`LengthMismatch`・`IndexOutOfRange`。最適化器・jitter ポリシー・変換の設定値は `InvalidConfig`。サイズの積の `usize` オーバーフローは `EmptyInput` ではなく `SizeOverflow`。

**Error/panicの線引き**: ユーザー入力起因(`DimensionMismatch`等)、モデル/データ起因(`CholeskyFailed`等)は`Result`で返し回復可能にする。`CoordGradientUnsupported`はライブラリ内部panic対象ではないため`unimplemented!()`ではなく本Errorを返す。

## 11. オンライン学習(データ点の追加削除)

GPRはn増加に伴いO(n³)でコストが増大するため、データの逐次追加削除を正式にスコープへ含める。バッチfit用Workspace(n固定)とは別に、crate-private の `LdltStore` と公開の `OnlineGpr` を置く。`FittedGpr::into_online(self)` が変換する。`insert` は `OnlineGpr` だけにある。

### コスト比較

| 操作    | フル再fit | 増分更新 |
| ------- | --------- | -------- |
| 1点追加 | O(n³)     | O(n²)    |
| 1点削除 | O(n³)     | O(n²)    |

### faer APIに合わせた実装方針(P0修正)

§3の通り、**LLTにinsert/delete APIは無い**。オンライン経路は次で進める。

1. **追加(末尾append)**: 自前で bordered update を実装する。O(n²)
2. **削除(任意インデックス)**: `LdltStore`は**LDLT因子**を保持し、`ldlt::update::delete_rows_and_cols_clobber`を使う。`2×2` / `5×5` の手書き SPD で、削除後の再構成 `A = L D Lᵀ` がフル LDLT と一致する（P3-1 / [#30](https://github.com/YUKIKEDA/gprx/issues/30)）。Givens downdate は置かない

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

crate-private。`from_active(n)` で `n_active = n_capacity = n`。訓練 `X` は `OnlineGpr` が持ち、この struct には置かない。末尾 insert の前に `OnlineGpr` が `ensure_capacity` する。倍率フィールドは置かない。

```rust
struct LdltStore {
    ld_factor: Mat<f64>,    // LDLT因子(対角=D、厳密下三角=L)
    alpha: Col<f64>,
    y: Col<f64>,
    v_buf: Col<f64>,        // 予測分散の前進消去スクラッチ(テスト点1点あたりO(n²))
    delete_scratch: MemBuffer, // faer delete_rows_and_cols。容量に合わせて伸ばす
    n_active: usize,
    n_capacity: usize,
}
```

**容量拡張** (`ensure_capacity(needed)`。`n_capacity < needed` のとき):

1. `new_cap = max(needed, max(n_capacity, 1) * 2)`
2. `ld_factor`, `alpha`, `y`, `v_buf`を`new_cap`で再確保。delete スクラッチも `new_cap` 用に伸ばす
3. 既存の`n_active × n_active`下三角と長さ`n_active`のベクトルをコピー
4. `PointRegistry`のインデックスは`n_active`未満のままなので付け替え不要
5. 拡張後にinsertを実行する。更新アルゴリズムの最中には再確保しない

**predict時の分散計算**: 予測平均はO(n)だが、予測分散`σ*² = k(x*,x*) - vᵀ D v`(LDLT、`L v = k*`の変形)はテスト点1点あたりO(n²)。`v_buf`をあらかじめ確保しておく。

### 増分更新の手順と不変条件

**追加（末尾）**: ①容量が足りなければ `LdltStore::ensure_capacity`（倍率 2）。`OnlineGpr` の訓練 `X` / `y` も同じ倍率で伸ばす。クエリバッファは `ensure_at_least` → ②新規点と既存n点との距離計算(O(n)。1 列は逐次、`v_buf` に `k` を直接書く) → ③カーネル対角 `k_new` だけ足す（insert は `K` の新行/列を書かない。予測・NLML は LD だけ読む） → ④bordered LDLT update(O(n²)。三角ソルブは `v_buf` を再利用) → ⑤`α` は insert では解かない（libgp `alpha_needs_update`）。O(1) で古い印を付けるだけ。最初に読む操作が LDLT で解き直す。`&mut self` の読み（`predict_into`・ハイパラの書き込み）はモデルに `α` を置き、`&self` の読み（`predict`・共分散・sample・LOO・NLML・`alpha()`・`save_with_factor`）は次の insert / delete が空にする `OnceLock` のキャッシュを埋める。解くのに失敗したら（`MixedPrecision` の f64 へのやり直しが分解できないなど）その読みの `Err` になるので、`OnlineGpr::alpha()` は `Result` を返す（R4-2b / [#265](https://github.com/YUKIKEDA/gprx/issues/265)） → ⑥`PointRegistry` に新しい `PointId` を発行。

**削除**: ①`ldlt::update::delete_rows_and_cols_clobber`でLD更新(O(n²)。スクラッチは `LdltStore` に置き再利用) → ②`OnlineGpr` の y・`X` から該当要素を除去し、後ろの行/列を詰める(O(n)) → ③`PointRegistry`のインデックスを同じ順序でシフト → ④`α` は delete でも解かない。古い印と最初の読みでの解き直しは追加の⑤と同じ。`n_capacity` は据え置く。最後の 1 点は消さない（`InsufficientData`、`min = 2`）。未知・削除済みの `PointId` は `InvalidPointId`。

**不変条件**: 削除により内部インデックスがシフトする際、workspace の `LD` / `y` / `alpha` と `OnlineGpr` の `X` と `PointRegistry`は**必ず同じ順序で同期**しなければならない。いずれか一つでも順序がずれると誤った解になる。この不変条件をテスト(§12)で明示的に検証する。

```rust
struct PointRegistry {
    id_to_index: HashMap<PointId, usize>,
    index_to_id: Vec<PointId>,
    next_id: u64,
}
```

### API

**insert/deleteとハイパラ再最適化を分離する**。

未学習の `Gpr` には点を足さない。バッチの `FittedGpr` に `insert` は無い。

```rust
impl FittedGpr<O, P> {
    fn into_online(self) -> Result<OnlineGpr<O, P>, GprError>;
}

impl OnlineGpr<O, P> {
    fn insert(&mut self, x_new: &[f64], y_new: f64) -> Result<PointId, GprError>;
    fn delete(&mut self, id: PointId) -> Result<(), GprError>;
    fn point_ids(&self) -> &[PointId];
}
```

`insert` / `delete` は現在のカーネル・ハイパラのまま LD・alpha を更新する。ハイパラ再最適化は `OnlineGpr::refit` / `set_params` を明示したときだけ。これらは一時的な LLT の置き場でバッチの fit を動かし（§6.3）、`PointId` とワークスペースの容量を保つ。`into_online` は既存 `n` 点に `0 .. n-1` を付け、以降の `insert` は単調増加で再利用しない。`PointId` に公開コンストラクタは無い。`PointRegistry` は crate-private で `OnlineGpr` が持つ。persist は `FORMAT_VERSION` 1 のまま `factor_kind`（`llt` / `ldlt`）を必須にする。`llt` の load は `FittedGpr`。`ldlt` は `OnlineGpr` で、`point_ids` と `next_point_id` も必須。Sparse のオンラインは `OnlineSgpr`（§6）。

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
5. **オンラインのプロパティテスト**: ランダムな insert/delete 列の各段階で incremental == `Gpr<Fixed>::factor`（mean, variance, LML, alpha）。削除順は `SmallRng` でランダム化する
5b. **オンライン insert の外部照合**(P3-6): 同じ θ の libgp `add_pattern` と predict（平均・観測分散）および NLML を相対 `1e-8`。delete の外部 API は無い。`cargo test` はコミット済み JSON を読む（C++ を呼ばない）
5c. **Sparse の外部照合**(P4-11): 同じ初期 θ の `Sgpr<Fixed>::factor` と GPyTorch 潰し SGPR、`Svgp<Fixed>::factor`（prior `q`）と whitened SVGP を相対 `1e-8`（平均・Observation・Latent・NLML / ELBO）。`cargo test` はコミット済み JSON を読む（Python を呼ばない）。バッチの時間・RSS は P4-12（`just perf-sparse`。GPyTorch / GPy。手動、CI なし）
5d. **Sparse オンラインの外部照合**(P4-13): 同じ初期 θ の `OnlineSgpr` の `insert` / `delete` / `insert_inducing` / `delete_inducing` を、各段階の GPyTorch 潰し SGPR（フル再組み立て）と相対 `1e-8`（平均・Observation・Latent・NLML）。`cargo test` はコミット済み JSON を読む（Python を呼ばない）
5e. **Sparse オンラインの時間比較**(P4-14): 同じ初期 θ の `OnlineSgpr` の `insert` / `delete` / `insert_inducing` / `delete_inducing` を、自前の `Sgpr<Fixed>::factor` フル再組み立ておよび GPyTorch の Titsias 組み立て（クエリなし）と時間比較する。正しさの相手は P4-13。プレフィックスは計時外。32 手を 1 本の壁時計（捨て 1 + 中央値。回数は P2B-16 の段ルール）。`cargo test` は走らせない。`just perf-sparse-online`（手動、CI なし）
6. **精度**: f32/f64/混合精度の比較、悪条件行列、収束しないケースでのf64フォールバック
7. **推論結果**: 既知の小規模GPR実装との比較(mean、潜在分散、観測分散、log marginal likelihood, gradient)。sklearn JSON は数値の第二照合であり、公開 API の契約ではない。アルゴリズムの正本は GPML / Rasmussen
8. **前処理**: `StandardizeTarget`適用後のpredictが、未標準化モデルと元スケールで一致すること(アフィン変換の閉じた関係)
9. **最適化後の推論**(P1B-6): 1次元 Forrester と 2次元重み付き球関数（ARD）で sklearn L-BFGS と `Gpr::fit` を緩い許容で照合する。固定ハイパラ JSON（1e-8）とは分ける。`cargo test` は Python を呼ばない
10. **Leave-one-out**(P1B-7): n=2 の GPML 解析式、n=3 の実 leave-one-out `fit`+`predict`、および P1B-6 JSON の LOO 欄を sklearn の `θ` で照合する。`cargo test` は Python を呼ばない

## 13. 実装ロードマップ

並びと状態は [roadmap.md](roadmap.md)。完了条件は各 Issue に残す。

## 14. 未解決事項

1. **混合精度反復改良のパラメータ検証**: §4.2のデフォルト値は理論根拠付きだが、実ワークロードでの検証は未実施。`PromoteStorage`と`ReevaluateKernel`の精度差、fit時MixedPrecisionのlog|K|・トレース項も含む
2. **`DistanceCachePolicy::Auto` の閾値**: カーネル種別・SIMD効率・メモリ帯域を考慮した実測が必要（P5-5。完了条件は Grill 後）

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

- `n = 256` を P1A-18 から必須。`512` / `1024` は数秒で終わるようになってから足す
- 等方: 1 次元 Forrester `f(x)=(6x-2)² sin(12x-4)`、`x ∈ [0, 1]`、RBF + `GaussianLikelihood` + `StandardizeTarget`。初期ハイパラ `ℓ = 1`、`σn² = 0.1`
- ARD: 2 次元重み付き球 `f=(x/0.25)²+(y/1)²`、`[0, 1]²` の 16×16 格子。初期 `ℓ_d = 4`（`ℓ_d = 1` では線探索が初手で止まる）
- `y` は上記の関数 + `N(0, 1)`（`SmallRng`。Forrester は seed `0`、ARD 球は seed `9`。seed `0` は Never で尾根を歩く）。独立な乱数系列にはしない（L-BFGS の評価回数が景観でぶれる）
- 歴史的な `phase-1a` / `phase-1b` ログの一部は `d = 8` と独立乱数 `y`。Forrester 上の `phase-1b` 再測は P2-9（`.dev/bench-log.md`（ローカル。コミットしない））。d = 8 の時間とは混ぜない
- グループ（存在する経路だけ。無いものはまだ書かない）:
  1. `kernel_rbf` — K の下三角構築
  2. `cholesky_alpha` — `A` の LLT と `α`
  3. `mll_and_grad` — §6.2 の 1 評価（P1A-10 から）
  4. `predict_100` — テスト点 100（P1A-8 から）
  5. `fit_lbfgs` — 最適化ループ全体（1b から。1 と混ぜない）。壁時計と一緒に L-BFGS の評価回数を残す。回数が違うときの差は速度差と読まない
  6. `mll_and_grad_ard` / `fit_lbfgs_ard` — 重み付き球の ARD RBF（P2-7）。Always vs Never。等方とは比べない。`fit_lbfgs_ard` も評価回数を残す
  7. `kernel_exp` / `kernel_exp_ard` — 距離を一度埋めたあとの `apply` と θ の `grad`（P5-4）。`FastApprox` と `Accurate`。`mll_and_grad` とは混ぜない
  8. `online_insert` / `online_delete` — Phase 3

### 15.3 いつ何を足す

| 時点 | やること |
|---|---|
| M0 | 箱だけ。空の `benches/` は置かない |
| P1A-7 の直後（P1A-18） | criterion と `just bench`。`kernel_rbf` と `cholesky_alpha` |
| P1A-8 / P1A-10 | 同じファイルに `predict_100` / `mll_and_grad` を足す。P1A-19 で確保 ratchet |
| 1a 完了 | 名前付き baseline `phase-1a` を取り、機械名と数値を `.dev/bench-log.md`（ローカル。コミットしない） に残す |
| 1b 完了 | `fit_lbfgs` を足し、baseline `phase-1b` |
| Phase 2 | **新しいハーネスは不要。** `phase-1b` を見てボトルネック順に最適化する。P2-5: 等方 RBF と距離に SIMD。可否は `kernel_rbf` / `predict` / `FIXED` で判断し、`mll_and_grad` の勾配項だけを分母にしない。NLML 定数項は P2-6 で測り、差はノイズなので `L(θ)` は一本のまま。ARD 距離キャッシュは P2-7 で `mll_and_grad_ard` / `fit_lbfgs_ard` の Always vs Never。埋めと RBF ARD は Rayon + SIMD |
| 2 完了（P2-9） | 名前付き baseline `phase-2` を取り、機械名と数値を `.dev/bench-log.md`（ローカル。コミットしない） に残す。等方は `phase-1b` と比較。ARD は Always vs Never。`FittedGpr` 経路で `just test` と alloc 0 |
| Phase 3+ | insert/delete などを同じ問題定義で足す。比較の基準は `phase-2`。Sparse の壁時計・RSS は `just perf-sparse`（P4-12）。オンライン時間は `just perf-sparse-online`（P4-14）。criterion に Sparse グループは足さない |

ホットパス（`src/kernel/`、`workspace`、`exact`、`objective`、`online`）の PR は、Verification に前回 baseline との criterion 結果を貼る。速さと無関係ならその理由を書く。

### 15.4 指標

目標比は `phase-1b` を取ってから置く。Phase 2 完了後の基準は `phase-2`。それまでは「前より悪くない」がゲート。

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

