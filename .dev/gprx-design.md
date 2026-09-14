# Rust製GPRライブラリ (gprx) 設計ドキュメント

## 1. 目的・スコープ

最も柔軟かつ最も高速なGaussian Process Regressionライブラリを、Rustで構築する。「柔軟」はユーザー定義カーネル・前処理・厳密/疎推論・最適化器の差し替え可能性、および**データ点の逐次追加削除(オンライン学習)**を指し、「高速」はアロケーション最小化・SIMD/マルチスレッド活用・精度切り替えによる計算量/メモリ最適化を指す。

## 2. 全体アーキテクチャ概要

```
入力 X, y
  → Transform Pipeline (前処理: MinMax, Standardize等)
  → CompiledKernel (KernelSpecをコンパイルした実行計画 + Workspace)
  → Inference (ExactGP / SparseGP など、GPModel trait経由で差し替え)
       → Objective (尤度・勾配、Optimizerへ提供)
       → Optimizer (L-BFGS / Nelder-Mead 等、勾配要否で分岐)
       → OnlineInference (ExactGPのみ: データ点の増分追加削除)
  → 予測 (mean, variance)
```

主要な設計原則:
- **静的ディスパッチを基本に、拡張点のみ`dyn`を許容**(カーネルのリーフ項、前処理、数学バックエンドなど頻度の低い呼び出しは`dyn`可)
- **バッチfitのアロケーションはfit開始時の1回のみ**。イテレーション内で新規確保しない(オンライン学習は§11で別方式)
- **精度はコンパイル時ジェネリクスで固定**(実行時分岐は挟まない)

## 3. 線形代数バックエンド: faer

Pure Rustで、OpenBLAS/LAPACK/Eigenと同等以上の性能を達成しており、RayonベースでOpenMP/TBB相当の並列化性能を持つ。FFI依存がなくビルドが単純な点もメリット。

- `Mat<T>`は**列優先(column-major)**、行ストライドは常に1、列末尾にアライメント用パディングが入りうる。自前ループ・入力データXの持ち方もこれに揃える(§7.2)
- Cholesky分解は`llt::factor::cholesky_in_place(a: MatMut<T>, regularization, par, stack, params)`で**in-place**。独立した`chol_factor`バッファは不要、`k_matrix`をそのまま上書きする
- 動的正則化(jitter)は`LltRegularization { dynamic_regularization_delta, dynamic_regularization_epsilon }`としてAPI組み込み済み。自前実装は不要
- `cholesky_in_place`はfaer側のスクラッチ領域(`MemStack`)を要求する。`cholesky_in_place_scratch::<T>(dim, par, params)`でサイズ照会し、これも自前アリーナの一部として確保する(§7.1)
- `Mat`は`reserve_exact(row_capacity, col_capacity)`による容量ベース確保をサポート。オンライン学習(§11)で活用する
- `llt::update::{insert,delete}_rows_and_cols_clobber`: 分解済みLに対し指定インデックスの行/列を追加/削除するAPI。データ点の増分追加削除(§11)に使う
- `ldlt_diagonal::update::rank_r_update_clobber`: ランクr更新(`A' = A + αww^T`)。ハイパーパラメータ変更が低ランクな`ΔK`をもたらす特殊ケースでのみ利用可(§5.4-1)

## 4. 精度ポリシー: f32/f64/混合精度

目的は「メモリ削減」と「計算速度」の両方。単純な二層分離(構築はf32、Cholesky直前でf64にpromote)だと、コスト支配的なO(n³)のCholesky部分がf64のままになり速度メリットを取り逃す。そのため**混合精度反復改良(mixed-precision iterative refinement)**を採用する。

```rust
trait PrecisionPolicy {
    type Storage: Scalar;  // カーネル行列・距離キャッシュ・Cholesky分解の精度
    type Refine: Scalar;   // 残差計算の精度
}

struct MixedPrecision;  // Storage=f32, Refine=f64 (推奨デフォルト)
struct SinglePrecision; // Storage=f32, Refine=f32 (改良なし)
struct DoublePrecision; // Storage=f64, Refine=f64
```

手順:
1. `K`をf32のまま`cholesky_in_place::<f32>`で分解(SIMDレーン2倍、最も重いO(n³)部分がf32速度)
2. f32の`L`で`alpha_0 = solve(L, y)`(近似解)
3. 残差`r = y - K_f64 @ alpha_0`をf64精度でO(n²)計算(`K`はf32保持のまま都度f64キャストしてGEMV、恒常的なf64バッファは不要)
4. f32の`L`で`delta = solve(L, r)`、`alpha_1 = alpha_0 + delta`
5. 収束するまで3〜5を数回(通常2〜3回)繰り返す

追加コストはO(n²)の残差計算を数回だけなので、O(n³)全体に対しては無視できる。**安全弁**として、混合精度モードでは`LltRegularization`のjitterをやや大きめにデフォルト設定し、規定回数内に残差ノルムが縮小しなければ`GpError`を返す。

実装優先度: まず`DoublePrecision`をデフォルト実装として提供し、`MixedPrecision`はオプション機能として後付けする。trait設計は両方を同じ枠組みに収めているため手戻りは小さい。

### 4.1 混合精度反復改良の収束判定パラメータ

古典的な反復改良理論(Higham)より、分解精度u_f(f32、単位丸め誤差≈1.19×10⁻⁷)と改良精度u_r(f64、≈2.22×10⁻¹⁶)を使う場合、収束可否と速度は`κ(K)·u_f`で決まる(収束条件`κ(K)·u_f<1`、収束すれば1反復あたりの誤差縮小率はおおよそ`κ(K)·u_f`)。`LltRegularization`のjitter σは`κ(K+σI) ≤ λ_max/σ+1`という上限を与えるため、**jitterの値が反復改良の収束可否を事実上決める**(前述の安全弁と直結)。

```rust
struct RefinementConfig {
    max_iterations: usize,   // デフォルト10。収束する場合は通常3反復以内で機械精度近くに達する
    tolerance_factor: f64,   // デフォルト10.0。判定式: τ = tolerance_factor × n × u_r
    stagnation_ratio: f64,   // デフォルト0.9。前回残差との比がこれを超えたら停滞とみなし早期中断
}
```

収束判定: `||r_k||∞ / (||K||∞ ||alpha_k||∞ + ||y||∞) < tolerance_factor × n × u_r`。`stagnation_ratio`超過が2回連続で発生したら`RefinementNotConverged`(§10)を返す。

自動リトライ: `RefinementNotConverged`時はjitterを10倍にしてf32分解からやり直す(最大3回)。f64への全面フォールバックよりコストが低く、多くのケースをカバーできる想定。

```rust
enum RefinementFallback {
    IncreaseJitterAndRetry { max_retries: usize }, // 推奨デフォルト: 3
    FallbackToDoublePrecision,
    ReturnError,
}
```

**位置づけ**: 理論的妥当性はあるが、実際のGPRワークロード(典型的なκ(K)の分布、n・dの規模)での最適値検証は今後の課題として残る(§12参照)。

## 5. カーネル設計

### 5.1 Spec(宣言層)/ Evaluator(実行層)の分離

合成のたびのヒープアロケーションを避けるため三層構造にする。

```rust
enum KernelSpec {
    Leaf(Box<dyn KernelTerm>),
    Sum(Box<KernelSpec>, Box<KernelSpec>),
    Product(Box<KernelSpec>, Box<KernelSpec>),
}

trait KernelTerm {
    fn distance_kind(&self) -> DistanceKind;
    fn apply(&self, dist: MatRef<f64>, out: MatMut<f64>);
    fn params(&self) -> &[f64];
    fn grad(&self, dist: MatRef<f64>, dK: MatMut<f64>, param_idx: usize);
}
```

`KernelSpec`は演算子オーバーロード(`Add`/`Mul`)でユーザーが自然に合成できる。`KernelTerm`はobject-safeなので**ユーザー定義カーネルはこれを実装するだけで組み込める**(必須要件)。`KernelSpec → CompiledKernel`への変換をfit開始時に一度だけ行い、以降は`Workspace`内の既存バッファへの書き込みのみにする。

### 5.2 距離キャッシュ

等方カーネル(RBF, Matern等)は生の座標差`(x_i-x_j)²`がfit中不変で、lengthscaleは後段のスケーリングに過ぎない。**生の距離テンソルは1回計算してfit中使い回す**。

```rust
enum DistanceKind { SqEuclidean, SqEuclideanARD, Periodic { period: usize } }
```

`CompiledKernel`構築時に合成木を走査し、必要な`DistanceKind`集合を重複排除して計算する。同じ`DistanceKind`を要求する複数の項があれば共有する。

#### 5.2.1 ARDキャッシュ閾値

判断基準は「メモリコスト」と「Choleskyコストに対する再計算コストの相対的な重さ」の比較。

- メモリコスト: `(n,n,d)`テンソルは`n² × d × sizeof(T)`バイト。n=5000, d=100, f64なら20GBに達するなど、n・dの積で急増する
- 再計算コスト: キャッシュなしの場合、毎イテレーションの再計算はO(n²d)。CholeskyはO(n³)で支配的。**両者の比はd/n**
  - d≪n(典型的GPR: d=10〜20、n=数千〜万)ではCholeskyに対し再計算コストが無視できるほど小さく、キャッシュの投資対効果が薄い
  - dがnに対し相対的に大きい高次元ARD(特徴選択目的でd=100〜1000クラス)では再計算コストが無視できなくなり、キャッシュの効果が出る

```rust
fn should_cache_ard(n: usize, d: usize, elem_size: usize, mem_budget_bytes: usize) -> bool {
    let cache_bytes = n * n * d * elem_size;
    let recompute_relative_cost = d as f64 / n as f64;
    cache_bytes <= mem_budget_bytes && recompute_relative_cost > RECOMPUTE_THRESHOLD
}
```

- `mem_budget_bytes`: ユーザー設定可能(デフォルト目安256MB程度、既存の`k_matrix`/`exp_buf`と同オーダーに収める)
- `RECOMPUTE_THRESHOLD`: 経験的にはd/n ≳ 0.05〜0.1あたりが投資対効果の分岐点という目安。理論だけでは確定できないため、実装後のベンチマーク(exp呼び出しコスト・メモリ帯域の実測)で調整する前提とし、デフォルト値として残す(§12)

### 5.3 CompiledKernelのplan構築アルゴリズム

Sum/Productは結合則・交換則が効くため、汎用レジスタ割当を持ち出さずシンプルなflatten+fold評価で済む。

1. **距離キャッシュ重複排除**: 合成木を走査し`DistanceKind`の`IndexSet`を構築
2. **flatten**: `(A+B)+C`と`A+(B+C)`を`Sum(vec![A,B,C])`に正規化(Productも同様)。ネストの形に依存しないplanにするため
3. **plan生成**: Sum/Productはそれぞれ単体では1バッファに畳み込める(上書き→以降は加算/乗算)。**必要バッファ数はネストの深さでしか増えない**(実用的な合成ではまず3を超えない)。`BufAllocator`(フリーリスト)で`alloc()`/`free()`を追跡し、最大同時使用数を計測してWorkspaceの確保サイズを決める

```rust
enum PlanOp {
    EvalLeafInto { term_id: usize, dist_id: usize, dst: BufId },
    AddLeafInto  { term_id: usize, dist_id: usize, dst: BufId },
    MulLeafInto  { term_id: usize, dist_id: usize, dst: BufId },
    AddBufInto   { src: BufId, dst: BufId },
    MulBufInto   { src: BufId, dst: BufId },
}
```

### 5.4 部分更新(コーディネート型最適化器)対応

**課題**: 5.3の畳み込みplanは上書き型のため、一部パラメータのみ変更された場合でも全項を再計算しないと結果を再現できない。座標降下法や一部パラメータのみ更新する最適化器を使う場合、変更されていない項の再計算(特に`exp`呼び出し)を省略したい。

**対応方針**: `RecomputeStrategy`として2種類を選択可能にする。

```rust
trait RecomputeStrategy {}

/// 5.3のplan。バッファ最小、常にフル再計算。nが大きい/メモリ重視の場合の既定。
struct FullRecompute;

/// リーフ項ごとに寄与行列(n×n)を独立バッファとして保持。
/// 変更された項のみ再計算し、最終結合(Sum/Productの合算)だけ毎回やり直す。
/// リーフ項数ぶんメモリを消費するため、項数が少ない/nが小さい場合向け。
struct IncrementalRecompute {
    leaf_contrib: Vec<Buf>,      // リーフ項ごとの寄与行列
    param_to_leaf: Vec<LeafId>,  // パラメータindex→リーフ項の逆引き
}
```

`IncrementalRecompute`の動作:
1. `CompiledKernel`構築時、パラメータ全体のインデックス範囲をリーフ項ごとに区切り、`param_to_leaf`を確定
2. 最適化器が`update_params(&[T], changed: ChangeSet)`を呼ぶ際、`ChangeSet::Indices(&[usize])`で変更indexを明示する(座標降下法はどのパラメータを更新したか自明なので検出コストは不要)
3. 変更indexに対応するリーフ項のみ`apply`/`grad`を再実行し、`leaf_contrib`を更新
4. 最終結合(Sum全項の和、Product全項の積)はO(n²×リーフ項数)で毎回やり直す。これはO(n³)のCholeskyに対して無視できるコストなので、フル再結合で問題ない

**重要な制約**: **Cholesky分解自体は`K`全体が変わる以上、部分更新の恩恵を受けられずフルで行う必要がある**。したがって部分更新が効くのは「カーネル行列構築コスト(O(n²)、特に`exp`呼び出しの回数)」のみで、O(n³)のCholeskyコストには効かない。項数が多い/評価が重いカーネル(周期カーネルの三角関数など)を多用する場合に構築コストの比重が相対的に上がるため、そうしたケースで`IncrementalRecompute`の恩恵が大きい。

**選択指針**: nが大きくメモリを切り詰めたい場合や単純なカーネル(項数少)は`FullRecompute`、項数が多く座標降下法的最適化を使う場合は`IncrementalRecompute`を推奨する。デフォルトは`FullRecompute`とし、`IncrementalRecompute`はオプトイン。

#### 5.4.1 IncrementalRecomputeとfaer update APIの関係

faerの`llt::update::{insert,delete}_rows_and_cols_clobber`は、分解済み`L`に対し指定インデックスの行/列を追加/削除するAPIで、**データ点の追加削除(次元nの変更)用**。ハイパラ変更には使えない(§11でオンライン学習として活用)。

一方`ldlt_diagonal::update::rank_r_update_clobber`(`A' = A + αww^T`型のランクr更新)は、**ハイパラ変更が`K`にもたらす差分`ΔK`が低ランクな場合に限り使える**:

- 線形カーネル項のamplitude変更: `ΔK`のランクはd(入力次元)相当 → `d≪n`ならrank-r updateでO(n²d)、フルCholesky(O(n³))を回避可能
- 全体スケール(outputscale)のみの変更: `K_new = c·K_old`なら`L_new = √c·L_old`で自明に更新、rank-r update自体も不要
- RBF/Maternのlengthscale変更など一般ケース: `ΔK`はランクnに近い密行列 → 高速パスの恩恵なし、フルCholesky必須

```rust
enum KRankStructure { Scalar, LowRank(usize), Dense }
trait KernelTerm {
    fn rank_structure(&self) -> KRankStructure { KRankStructure::Dense } // 安全側デフォルト
}
```

`IncrementalRecompute`は変更リーフ項の`rank_structure()`を見て高速パスを選択可能にする。デフォルト`Dense`ならユーザー定義カーネルは何もせず安全側に倒れるため、オプトインの上乗せとして安全に追加できる。

### 5.5 前処理パイプライン

```rust
trait Transform {
    fn fit(&mut self, x: MatRef<f64>);
    fn apply(&self, x: MatMut<f64>);
}
struct Pipeline(Vec<Box<dyn Transform>>);
```

`GPModel`のbuilderに`.with_transform(MinMaxScaler::new())`のように積める。fit時に統計量推定・保存、predict時に自動適用。ホットパスではないため`dyn`で問題ない。

## 6. GPModel抽象化(厳密/疎の差し替え)

```rust
trait Inference<T: Scalar> {
    fn fit(&mut self, x: MatRef<T>, y: &[T], kernel: &mut CompiledKernel<T>, mean: &dyn MeanFn<T>) -> Result<(), GpError>;
    fn predict(&self, xs: MatRef<T>) -> (Vec<T>, Vec<T>);
    fn objective(&mut self) -> &mut dyn Objective<T>;
}
```

`ExactGP`(n≲1万)と`SparseGP`(FITC/VFE、誘導点法)がこれを実装。

### 6.1 Sparse GPの誘導点キャッシュ問題

Sparse GP(FITC/VFE)は`K(X,Z)`(n×m)、`K(Z,Z)`(m×m)、`K(X,X)`対角、の3種類を使う。誘導点`Z`が最適化対象になると`K(X,Z)`/`K(Z,Z)`はイテレーションごとに変わり、§5.2の「両側不変」前提の距離キャッシュが崩れる。

**片側不変性の活用**:
- `K(X,X)`対角: Xのみに依存し完全に不変。fit開始時に1回計算、既存の距離キャッシュ機構をそのまま流用
- `K(X,Z)`, `K(Z,Z)`: Zが動くたびに再計算が必要。ただしm(誘導点数、通常n≫m)が小さいため、この再計算コストはCholeskyのO(nm²)(Sparse GPの主コスト)に対して十分小さく、**キャッシュ対象にせず毎回再計算する**のが既定方針でよい

```rust
struct SparseWorkspace<T: Scalar> {
    kxx_diag: Buf<T>,   // 不変、fit開始時に1回計算
    kxz_buf: Buf<T>,    // n×m、可変、毎イテレーション上書き(キャッシュしない)
    kzz_buf: Buf<T>,    // m×m、可変、Cholesky対象
    faer_scratch: Buf<T>,
}
```

§7.1の「アリーナ+オフセットビュー」方式は共通で流用可能。この「不変/可変の分離」は§5.4の`IncrementalRecompute`(同一座標でハイパラのみ変わる場合の部分更新)とは別軸の問題であり、混同しないよう明確に区別する。

**誘導点座標の勾配**: 誘導点Z自体が最適化パラメータ(座標値)である点はハイパラと扱いが異なり、`∂K(X,Z)/∂Z`という座標微分が必要。`KernelTerm`にデフォルト実装付きで座標微分メソッドを追加し、Sparse GPを使わないユーザーの実装負担を増やさない。デフォルト実装は`unimplemented!()`ではなく`GpError::CoordGradientUnsupported`(§10)を返す(ユーザーの構成ミスであってライブラリ内部のpanic対象ではないため)。

```rust
trait KernelTerm {
    fn grad(&self, dist: MatRef<f64>, dK: MatMut<f64>, param_idx: usize);
    fn grad_wrt_coords(&self, x1: MatRef<f64>, x2: MatRef<f64>, dK: MatMut<f64>, coord_idx: (usize, usize)) -> Result<(), GpError> {
        Err(GpError::CoordGradientUnsupported)
    }
}
```

**Sparse GPのオンライン学習は誘導点ZとデータXの非対称性のため今回のスコープ外**(§12参照)。

## 7. Workspaceとメモリ管理(バッチfit)

### 7.1 単一アリーナ + オフセットビュー

fit開始時にn,dが既知なので必要サイズを事前計算し、単一の連続領域を1回だけ確保する。イテレーション中は新規確保しない。

```rust
struct Workspace<T: Scalar> {
    arena: Vec<T>,
    dist_cache_offset: usize,
    k_matrix_offset: usize,   // Cholesky後はin-place上書きでLになる、独立バッファ不要
    exp_buf_offset: usize,    // value計算時に書き込み、gradient計算で再利用(exp再評価を回避)
    faer_scratch_offset: usize, // faerのcholesky_in_place_scratchが要求する領域
    thread_scratch: Vec<(usize, usize)>, // Rayonスレッド数ぶん事前分割(offset, len)
}
```

`'static`借用は自己参照になり扱いにくいため、実装は生ポインタではなく**オフセット(usize)方式**を推奨(借用チェッカーと相性が良い)。faer自身のスクラッチ要求(`cholesky_in_place_scratch`)も見落としやすいので、アリーナサイズ計算に必ず含める。

Rayon並列クロージャ内での`Vec::new()`は呼び出し回数ぶんアロケーションが走るため厳禁。スレッド数ぶん`thread_scratch`を事前分割し、`rayon::broadcast`かインデックスベースで割り当てる。

### 7.2 メモリレイアウト

faerの`Mat`は列優先・行ストライド1。自前ループもこれに揃える:

- 距離行列・カーネル行列の走査は**列優先**(`for j in 0..n { for i in 0..=j { ... } }`)
- **対称性を利用し上三角/下三角のみ計算**(faerのLLTも下三角を扱うため整合)。SIMD化はブロック単位(例8x8タイル)で、対角ブロックのみ三角処理、それ以外は矩形として処理する
- 入力`X(n×d)`は**1データ点=1列=メモリ連続**(`d×n`の列優先)で保持。ARDの次元ごと差分をSIMDレーンに載せる際、1点=連続dスカラーという前提と一致させる

### 7.3 イテレーション中のライフサイクル

```
fit()開始 → n,d確定 → arena確保(1回) → 距離キャッシュ計算(1回)
  → 最適化ループ:
      各iter: k_matrix領域に上書き構築 → in-place Cholesky(同一領域再利用)
             → faer scratchも同一バッファ使い回し → value/grad
fit()終了 → Workspaceは保持、predict/refitで再利用
```

## 8. 並列化・SIMD

- カーネル評価内側ループ(特に`exp`)は`std::simd`かwideクレートでベクトル化
- 距離行列・カーネル行列構築はRayonでブロック並列化
- **faer自身もRayon並列化されるため、外側Rayonとの二重並列化でスレッド過剰生成に注意**。単一の`rayon::ThreadPool`をアプリ全体で共有する
- `exp`は`MathBackend` traitで差し替え可能にする:

```rust
trait MathBackend<T: Scalar>: Send + Sync {
    fn exp_inplace(&self, buf: &mut [T]);
    fn erf_inplace(&self, buf: &mut [T]);
}
struct StdExp;              // libm、正確だが遅い、依存なし
struct SleefBackend;        // SIMDベクトル化、高精度、Cライブラリ依存
struct PolyApproxExp { degree: u8 } // pure Rust多項式近似、依存なし
```

デフォルトは`PolyApproxExp`(依存ゼロ維持)。`SleefBackend`はfeatureフラグでオプトイン。GPRの精度要求はCholeskyの数値誤差に埋もれることが多く、多項式近似で実用上十分なケースが多い。

## 9. Optimizer設計

`value`と`gradient`を分離し、勾配不要な最適化器(Nelder-Mead等)がgradientパスに一切触れないようにする。

```rust
trait Objective<T: Scalar> {
    fn value(&mut self, params: &[T]) -> T;
    fn gradient(&mut self, params: &[T]) -> Option<Vec<T>>;
    fn value_and_gradient(&mut self, params: &[T]) -> (T, Option<Vec<T>>) {
        (self.value(params), self.gradient(params))
    }
}
trait Optimizer<T: Scalar> {
    fn minimize(&self, objective: &mut dyn Objective<T>, init: Vec<T>) -> OptResult<T>;
    fn requires_gradient(&self) -> bool;
}
```

`ExactGP`の`value_and_gradient`実装はCholesky分解(`L`, `alpha`)を尤度と勾配の両方で共有し、`exp_buf`も同様に再利用する。座標降下法的な最適化器を使う場合は§5.4の`ChangeSet`を`Objective`側のAPIにも伝播させ(`value_at(params, changed)`のような形)、`IncrementalRecompute`と接続する。

## 10. エラー型 GpError

```rust
#[derive(Debug, thiserror::Error)]
pub enum GpError {
    #[error("入力次元が一致しません: X.ncols()={x_dim}, 期待値={expected_dim}")]
    DimensionMismatch { x_dim: usize, expected_dim: usize },

    #[error("データ点数が不足しています: n={n}, 最低{min}点必要です")]
    InsufficientData { n: usize, min: usize },

    #[error("Cholesky分解に失敗しました(行列が半正定値ではありません, jitter={jitter}を適用済み)")]
    CholeskyFailed { jitter: f64 },

    #[error("混合精度反復改良が収束しませんでした({iterations}回反復後、残差ノルム={residual_norm})")]
    RefinementNotConverged { iterations: usize, residual_norm: f64 },

    #[error("このカーネル項はSparse GP用の座標微分(grad_wrt_coords)を実装していません")]
    CoordGradientUnsupported,

    #[error("最適化が収束しませんでした({iterations}回反復後)")]
    OptimizationNotConverged { iterations: usize },

    #[error("ハイパーパラメータが不正です: {reason}")]
    InvalidHyperparameter { reason: String },
}
```

**Error/panicの線引き**:
- `DimensionMismatch`, `InsufficientData`: ユーザー入力起因、`Result`で返し回復可能にする
- `CholeskyFailed`: jitter適用済みでも半正定値にならない場合。データ/カーネル設計の問題の可能性が高く`Result`で返す
- `RefinementNotConverged`: §4の安全弁に対応。`Result`で返し、呼び出し側が`DoublePrecision`へのフォールバックを選べるようにする
- `CoordGradientUnsupported`: §6.1の`grad_wrt_coords`デフォルト実装はpanicではなく本Errorを返す

## 11. オンライン学習(データ点の追加削除)

GPRはn増加に伴いO(n³)でコストが増大するため、データの逐次追加削除は実用上避けて通れないユースケースとして正式にスコープへ含める。§7.1の「fit開始時に1回だけアリーナ確保、以降不変」という前提とは相容れないため、**ExactGP向けに専用のWorkspace・更新経路を用意する**(バッチfit用のアリーナ方式とは別実装)。

### コスト比較

| 操作 | フル再fit | 増分更新 |
|---|---|---|
| 1点追加 | O(n³) | O(n²)(距離O(n)+カーネル評価O(n)+faer insert O(n²)+alpha再ソルブO(n²)) |
| 1点削除 | O(n³) | O(n²)(faer delete O(n²)+alpha再ソルブO(n²)) |

### Workspaceの容量方式

faerの`Mat`自体が`reserve_exact(row_capacity, col_capacity)`をサポートするため、これを直接活用する。Vec同様の償却成長戦略(growth_factor 1.5〜2.0)で容量超過時のみ再確保。

```rust
struct OnlineWorkspace<T: Scalar> {
    k_matrix: Mat<T>,
    dist_cache: Mat<T>,
    l_factor: Mat<T>,      // insert/delete_rows_and_cols_clobberで直接更新
    alpha: Col<T>,
    n_active: usize,
    n_capacity: usize,
    growth_factor: f64,
}
```

### 増分更新の手順

**追加**: ①新規点と既存n点との距離計算(O(n)、ARDはO(nd))、距離キャッシュに新規行/列追加 → ②カーネル評価しK行列に新規行/列追加(`IncrementalRecompute`使用時は各リーフ項バッファも同様に追加) → ③`insert_rows_and_cols_clobber`でL更新(O(n²)) → ④alpha再ソルブ(O(n²))

**削除**: ①`delete_rows_and_cols_clobber`でL更新(O(n²)) → ②距離キャッシュ・K・y・alphaから該当要素を除去(O(n)) → ③alpha再ソルブ(O(n²))

### インデックス管理

faerのAPIは内部行列インデックス(0..n_active)を直接操作し、削除のたびに後続点のインデックスがシフトする。安定した点ID⇔内部インデックスのマッピング層が必要。

```rust
struct PointRegistry {
    id_to_index: HashMap<PointId, usize>,
    index_to_id: Vec<PointId>, // 削除のたびにシフト、O(n)
}
```

### API

```rust
trait OnlineInference<T: Scalar> {
    fn insert(&mut self, x_new: &[T], y_new: T, kernel: &mut CompiledKernel<T>) -> Result<PointId, GpError>;
    fn delete(&mut self, id: PointId, kernel: &mut CompiledKernel<T>) -> Result<(), GpError>;
}
```

`ExactGP`がこれを追加実装する。**Sparse GPのオンライン学習は誘導点ZとデータXの非対称性のためスコープ外**(§12未解決事項)。

### 運用上の注意

insert/deleteは現在のハイパーパラメータのままL・alphaを更新するだけで、ハイパーパラメータ自体は再最適化されない。データ分布の変化が大きい場合は、増分更新を続けつつ定期的に通常の`fit`(§9)を挟んでハイパーパラメータを追従させる運用が必要。

## 12. 未解決事項

1. **Sparse GPのオンライン学習**: §11でExactGPの増分更新は解決したが、誘導点ZとデータXの非対称性がありSparse GPへの適用は別設計が必要
2. **混合精度反復改良のパラメータ検証**: §4.1のデフォルト値は理論根拠付きだが、実ワークロードでのベンチマーク検証は未実施
3. **ARDキャッシュのRECOMPUTE_THRESHOLD**: §5.2.1の目安値(d/n ≳ 0.05〜0.1)もベンチマークでの調整が必要
