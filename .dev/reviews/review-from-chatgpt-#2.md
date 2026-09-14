# gprx 設計ドキュメント レビュー

## 総合評価

# B+ 相当

設計の方向性は良好

結論：Phase 1のExact GPR実装に進める段階。ただし、オンライン更新・精度ポリシー・カーネルAPIには実装前に詰めるべき重要な課題が残っています。

今回の設計書は、以前のレビューで指摘された問題をかなり丁寧に反映しています。特に、ノイズとjitterの分離、f32/f64の扱い、Workspace、Objective、オンライン更新の不変条件などは改善されています。

レビュー対象：`gprx 設計ドキュメント` v1相当、全463行。

## 1. 良い点

### 1.1 過去のレビュー指摘を反映できている

設計書の改訂履歴と各節を見る限り、以前問題になっていた部分はかなり整理されています。

|
項目

|

評価

|
| --- | --- |
|

観測ノイズとjitterの分離

|

良い

|
|

f32/f64の型設計

|

改善済み

|
|

混合精度の残差計算

|

方向性は正しい

|
|

カーネル宣言と実行の分離

|

良い

|
|

Workspace再利用

|

良い

|
|

ObjectiveとInferenceの分離

|

良い

|
|

オンライン更新の不変条件

|

重要な指摘を反映

|
|

テスト計画

|

以前より大幅に改善

|
|

実装ロードマップ

|

現実的になった

|

特に、単に「前の指摘を修正した」だけでなく、修正に伴う設計変更を周辺にも反映しようとしている点は評価できます。

### 1.2 Phase分割が適切

```
Phase 1: 正しいExact GPR
    ↓
Phase 2: 高速化
    ↓
Phase 3: オンライン学習
    ↓
Phase 4: Sparse GPR
    ↓
Phase 5: 高度な最適化
```

これはかなり良い順序です。

GPRライブラリでは、以下を同時に実装すると問題の切り分けが難しくなります。

* 数値計算の正しさ

* カーネルの正定値性

* Cholesky分解

* 勾配

* 最適化

* 並列化

* オンライン更新

まずf64のExact GPRを完成させ、フル再計算と一致することを確認してから高速化する方針は、そのまま維持すべきです。

### 1.3 Exact GPRとSparse GPRを同じ抽象に押し込んでいない

`Inference`を分離し、`Gpr`と`SparseGpr`を差し替え可能にする方針は良いです。

Rust

```
trait Inference<T: Scalar> {
    fn fit(&mut self, x: MatRef<T>, y: &[T]) -> Result<(), GprError>;
    fn predict(&self, xs: MatRef<T>) -> Result<Prediction<T>, GprError>;
}
```

Exact GPRとSparse GPRは、計算量だけでなく、尤度・パラメータ・誘導点・予測分散の扱いが異なります。

したがって、共通化するのは「GPとしての利用インターフェース」に留め、内部アルゴリズムは分離するのが妥当です。

# 2. 重要な指摘（P0〜P1）

以下は、実装開始前に修正・明確化したい項目です。

## P0-1. オンライン追加時のCholesky更新式とfaer APIの整合性

該当：§11、385〜390行

設計書では、新しい点を追加する行列を次のようにしています。

Knew=\(KkkTknew\)K_{\mathrm{new}}= \begin{bmatrix} K & k\\ k^T & k_{\mathrm{new}} \end{bmatrix}Knew=\(KkT​kknew​​\)

そして、

Lnew=\(L0vTd\)L_{\mathrm{new}}= \begin{bmatrix} L & 0\\ v^T & d \end{bmatrix}Lnew=\(LvT​0d​\)

に対して、

Lv=k,d=knew−vTvLv=k,\qquad d=\sqrt{k_{\mathrm{new}}-v^Tv}Lv=k,d=knew−vTv

としています。

ここは数式の向きが不整合です。

標準的な下三角Choleskyでは、

Lnew=\(L0vTd\)L_{\mathrm{new}}= \begin{bmatrix} L & 0\\ v^T & d \end{bmatrix}Lnew=\(LvT​0d​\)

なので、

Lv=kL v = kLv=k

ではなく、

Lv=kL v = kLv=k

という形のベクトル vvv を求め、下段に vTv^TvT を置くのが正しいです。

つまり、数式自体は概ね正しいですが、faerのAPIが行列のどの領域に何を格納するかは別問題です。

### 問題点

`insert_rows_and_cols_clobber`が次のどれを想定しているか、設計書ではまだ確定していません。

* 元の行列がフル行列か

* Cholesky因子が下三角行列か

* 追加行・列の挿入位置

* 追加する要素が元のKかLか

* 追加後に対角成分をどう扱うか

* 任意インデックス削除が可能か

### 修正提案

Phase 3に入る前に、API検証用の独立テストを作るべきです。

Rust

```
#[test]
fn cholesky_insert_matches_full_factorization() {
    // 1. 小規模な正定値行列を生成
    // 2. フルCholesky分解
    // 3. 1点追加
    // 4. faer update APIで更新
    // 5. フルCholeskyとの一致を検証
}
```

優先度：P0

## P0-2. `Likelihood`のノイズパラメータAPIに矛盾がある

該当：§4.0、46〜54行

Rust

```
trait Likelihood<T: Scalar>: Send + Sync {
    fn add_noise_diag(&self, k_diag: &mut [T]);
    fn noise_params(&self) -> &[T];
    fn noise_grad_diag(
        &self,
        dK_diag: &mut [T],
        param_idx: usize
    );
}
```

Rust

```
struct GaussianLikelihood<T: Scalar> {
    log_noise_variance: T
}
```

ここで、`GaussianLikelihood`はパラメータを1個持っていますが、`noise_params()`はスライスを返す設計です。

さらに、コメントには、

Rust

```
∂K/∂σn² = 2σn・I
```

とありますが、パラメータが `log_noise_variance` なら勾配は異なります。

### 何が問題か

ノイズ分散を

σn2=exp⁡(θ)\sigma_n^2=\exp(\theta)σn2=exp(θ)

とパラメータ化する場合、

∂K∂θ=σn2I\frac{\partial K}{\partial \theta} = \sigma_n^2 I∂θ∂K=σn2I

です。

一方、標準偏差 σn\sigma_nσnをパラメータ化するなら、

∂K∂σn=2σnI\frac{\partial K}{\partial \sigma_n} = 2\sigma_n I∂σn∂K=2σnI

になります。

現在の設計書は、パラメータの意味と勾配の式が一致していません。

### 修正提案

パラメータを明示的に定義してください。

Rust

```
struct GaussianLikelihood<T: Scalar> {
    log_noise_variance: T,
}
```

なら、

Rust

```
fn noise_grad_diag(
    &self,
    dK_diag: &mut [T],
) {
    let noise_variance = self.log_noise_variance.exp();

    // ∂K / ∂log_noise_variance = noise_variance * I
}
```

のようにするべきです。

また、`noise_params()`をスライスで返すなら内部にパラメータ配列を持つ設計にするか、単一パラメータなら別のAPIにする方が自然です。

優先度：P0

## P1-1. `KernelSpec`のパラメータ管理が未定義

該当：§5.1

Rust

```
enum KernelSpec {
    Leaf(Box<dyn KernelTermSpec>),
    Sum(Box<KernelSpec>, Box<KernelSpec>),
    Product(Box<KernelSpec>, Box<KernelSpec>),
}
```

Rust

```
trait KernelTermSpec: Send + Sync {
    fn params(&self) -> &[f64];
    fn compile<T: Scalar>(&self) -> Box<dyn KernelTerm<T>>;
}
```

ここで、次の問題があります。

### 問題点

`params()`が読み取り専用ですが、ハイパーパラメータ最適化で更新する仕組みがありません。

また、複合カーネルで、

```
RBF(length_scale=1.0)
+
Matern(length_scale=2.0)
*
Periodic(period=3.0)
```

のような構造を作った場合、最適化器のパラメータ配列と、各リーフのパラメータとの対応をどこで管理するのかが未定義です。

### 修正提案

パラメータをフラット化するための明示的な仕組みが必要です。

Rust

```
struct ParameterId(usize);

struct ParameterBinding {
    id: ParameterId,
    leaf_id: LeafId,
    local_index: usize,
}
```

そして、

Rust

```
trait KernelSpec {
    fn num_params(&self) -> usize;
    fn get_params(&self, out: &mut [f64]);
    fn set_params(&mut self, params: &[f64]);
}
```

のように、最適化器との接続を明示した方が良いでしょう。

優先度：P1

## P1-2. `CompiledKernel<T>`の`dyn KernelTerm<T>`がホットパスに残る

該当：§2、§5.1

設計原則は、

> 静的ディスパッチを基本に、拡張点のみ`dyn`を許容

ですが、現在の設計では、

Rust

```
fn compile<T: Scalar>(&self) -> Box<dyn KernelTerm<T>>;
```

となっています。

これは、カーネル評価のたびにtrait object経由の呼び出しが発生する可能性があります。

### 問題点

高速化を目指すライブラリでは、次のような処理がホットパスです。

```
for i in 0..n {
    for j in 0..n {
        kernel.eval(...)
    }
}
```

この部分に動的ディスパッチが残ると、カーネル合成・SIMD・インライン化の恩恵を受けにくくなります。

### 修正提案

少なくとも以下の2層に分けるのが良いです。

Rust

```
// 宣言層
trait KernelTermSpec { ... }

// 実行層
trait KernelEvaluator<T> {
    fn eval_block(&self, ...);
}
```

そして、内部のホットパスは、

Rust

```
enum CompiledKernel<T> {
    Rbf(RbfKernel<T>),
    Matern(MaternKernel<T>),
    Periodic(PeriodicKernel<T>),
    Sum(Vec<CompiledKernel<T>>),
    Product(Vec<CompiledKernel<T>>),
}
```

のようなenumベースの静的ディスパッチも検討してください。

ユーザー定義カーネルだけは `dyn` を許容する、という設計にすれば原則と整合します。

優先度：P1

## P1-3. `Workspace`の設計が「固定容量」と「オンライン成長」で分かれていない

該当：§7、§11

バッチfitでは、

Rust

```
fit()開始 → n,d確定 → 各Mat<T>を1回だけ確保
```

としています。

一方、オンライン更新では、

Rust

```
growth_factor: f64
```

を持っています。

この2つは性質が異なります。

### バッチfit

* nが固定

* メモリ容量を最適化できる

* 反復中のアロケーションを抑えやすい

### オンライン更新

* nが増減する

* 容量拡張が必要

* `Mat`の再確保やデータ移動が起こり得る

### 問題点

オンライン更新で、容量拡張時に以下をどう扱うかが未定義です。

* `K`

* `L`

* `alpha`

* 距離キャッシュ

* Choleskyのスクラッチ

* PointIdと内部インデックス

### 修正提案

オンライン更新は、バッチfitとは別のメモリ管理方針を明示してください。

Rust

```
struct OnlineWorkspace<T: Scalar> {
    capacity: usize,
    n_active: usize,

    k_matrix: Mat<T>,
    l_factor: Mat<T>,
    alpha: Col<T>,

    // 拡張用の再確保・移動処理
}
```

また、`capacity`拡張時は、単に`Mat`を拡張するだけでなく、既存の内部インデックスと全ての関連バッファの整合性を保つ必要があります。

優先度：P1

# 3. 数値計算に関する重要な指摘

## 3.1 混合精度反復改良は、現在の説明だけでは安全性を保証できない

該当：§4.1〜4.2

設計書の方針は、

> f32でCholesky分解し、f64で残差を計算して反復改良する

というものです。

これは一般的な混合精度線形ソルブの考え方として妥当です。

ただし、以下の点を追加で明確化した方が良いです。

### 問題点

`A`はf64で再構成する必要があります。

```
A = K + noise_diag
```

の行列をf32のまま保持し、f64に変換して残差を計算すると、元のf64精度の行列を使ったことにはなりません。

つまり、

```
f32 K → f64へ変換 → 残差
```

では、f64精度の入力行列に対する反復改良とは異なります。

### 修正提案

混合精度の場合、以下の2つを明確に分けるべきです。

1. f32で保存する計算用行列

2. f64で保持する残差計算用の行列、または高精度で再評価する仕組み

ただし、f64のK行列を保持するとメモリ削減効果が小さくなるため、混合精度の目的とのトレードオフになります。

結論：混合精度はPhase 5で良いが、`A_f64`をどう構築するかは実装前に仕様化すべきです。

## 3.2 jitterの扱いは、もう一段厳密にする必要がある

該当：§4.0

設計書では、

> jitterは分解時の内部的な摂動に留め、反復改良の目標には含めない

としています。

これは「モデルのノイズと数値安定化を分離する」という目的には合っています。

ただし、実際にCholesky分解で使った行列が、

Ajitter=A+jIA_{\mathrm{jitter}}=A+jIAjitter=A+jI

である場合、その因子を使って得た解は厳密には、

Ajitter−1yA_{\mathrm{jitter}}^{-1}yAjitter−1y

です。

一方、反復改良の目標は、

A−1yA^{-1}yA−1y

です。

### 問題点

jitterが大きい場合、f32のCholesky因子を使った反復改良では、目的の線形システムとの差が大きくなります。

したがって、jitterを増やすほど反復改良で元の行列の解に戻せるとは限りません。

### 修正提案

`NumericalStability`に、次の方針を明記してください。

Rust

```
enum JitterPolicy {
    Fixed(f64),
    Adaptive {
        initial: f64,
        multiplier: f64,
        max_retries: usize,
        max_jitter: f64,
    },
}
```

また、jitterが大きくなった場合は、

* 元のAに対する残差を使って反復改良する

* 収束しなければf64のフル分解にフォールバックする

* それでも失敗すればエラー

という動作を明確にする必要があります。

# 4. カーネル設計

## 4.1 距離キャッシュは、カーネル種別ごとに考えるべき

該当：§5.2

設計書では、距離キャッシュを抽象化して、

Rust

```
enum DistanceCachePolicy {
    Never,
    Always,
    Auto { memory_budget_bytes: usize },
}
```

としています。

これは良いです。

ただし、現在の設計では、`DistanceKind`と距離キャッシュの関係がまだ曖昧です。

Rust

```
enum DistanceKind {
    SqEuclidean,
    SqEuclideanARD,
    Periodic { period: usize },
}
```

### 問題点

Periodicカーネルは、単純な二乗距離だけではありません。

例えば、

k(x,x′)=σ2exp⁡(−2sin⁡2(π∣x−x′∣/p)ℓ2)k(x,x') = \sigma^2 \exp\left( -\frac{2\sin^2(\pi|x-x'|/p)}{\ell^2} \right)k(x,x′)=σ2exp(−ℓ22sin2(π∣x−x′∣/p))

のようなカーネルでは、距離差の扱いが異なります。

また、ARDカーネルでは、各次元の距離が必要です。

したがって、キャッシュの単位は、

* 生の座標差

* 二乗距離

* 周期変換済みの距離

* カーネル固有の中間値

のどれなのかを決める必要があります。

### 修正提案

`DistanceKind`よりも、キャッシュする中間表現を明示する方が良いです。

Rust

```
enum DistanceCache {
    None,
    SquaredEuclidean(Mat<T>),
    SquaredEuclideanArd(Vec<Mat<T>>),
    Periodic(Vec<Mat<T>>),
}
```

ただし、カーネルが増えるとこのenumは肥大化するため、Phase 2の実装時に設計を再検討してください。

## 4.2 `IncrementalRecompute`は、現時点では優先度が低い

該当：§5.4

設計書では、ハイパーパラメータの変更時に一部のカーネル項だけを再計算する設計です。

これは合理的ですが、次の理由からPhase 5で十分です。

* Choleskyは基本的にフル再分解

* 最終結合はO(n²×leaf_count)

* カーネル行列構築が支配的かどうかはデータサイズ次第

* カーネル項数が少ないと効果が薄い可能性

特に、Exact GPRではCholeskyがO(n³)です。

そのため、まずは、

```
フルK構築
→ Cholesky
→ MLL
→ 勾配
```

を正しく高速化することが先です。

評価：設計としては妥当。ただし初期実装には不要。

# 5. API設計

## 5.1 `Inference`の`fit`がモデルの状態をどう持つか不明

Rust

```
trait Inference<T: Scalar> {
    fn fit(&mut self, x: MatRef<T>, y: &[T]) -> Result<(), GprError>;
    fn predict(&self, xs: MatRef<T>) -> Result<Prediction<T>, GprError>;
}
```

ここで、`fit()`が何を更新するのかを明確にすると良いです。

* カーネルパラメータ

* Likelihood

* 学習データ

* Cholesky因子

* alpha

* Workspace

* Optimizerの状態

特に、`Inference`と`Objective`を分離したことで、どこがモデルパラメータの所有者なのかが重要になります。

### 提案

モデルと学習器の責務を分ける形を検討してください。

Rust

```
struct Gpr<T: Scalar, K> {
    kernel: K,
    likelihood: GaussianLikelihood<T>,
    workspace: Workspace<T>,
    state: GprState<T>,
}
```

Rust

```
struct GprState<T: Scalar> {
    alpha: Col<T>,
    l_factor: Mat<T>,
    n: usize,
}
```

また、`fit()`と`predict()`の前提条件も明示してください。

* fit前のpredictはエラー

* fit後の入力次元は固定

* データ点数0の場合

* NaN/Infを含む場合

* Cholesky失敗時の状態

## 5.2 `Prediction`に不確実性の意味を明示すべき

Rust

```
struct Prediction<T: Scalar> {
    mean: Vec<T>,
    variance: Vec<T>,
}
```

`variance`が何を表すかを決める必要があります。

GPの予測分散には、一般に、

* 潜在関数 f∗f_*f∗ の分散

* 観測値 y∗y_*y∗ の分散（ノイズ込み）

の2種類があります。

### 提案

Rust

```
struct Prediction<T: Scalar> {
    mean: Vec<T>,
    variance: Vec<T>,
    variance_kind: VarianceKind,
}

enum VarianceKind {
    Latent,
    Observation,
}
```

または、APIの引数で切り替える設計です。

これはSparse GPRにも関係するため、Phase 1の段階で定義した方が良いでしょう。

# 6. テスト計画の評価

該当：§12

テスト計画は良い方向に改善されています。

特に以下は重要です。

```
1. カーネルの数学的正当性
2. Choleskyの正当性
3. オンライン更新
4. 精度
5. 推論結果
```

ただし、もう少し追加したいです。

## 6.1 MLLと勾配のテストを独立させる

現在は推論結果のテストに含まれていますが、MLLと勾配は非常に重要です。

### 追加推奨

```
- MLLの解析値との比較
- MLLの数値微分との比較
- 各カーネルパラメータの勾配比較
- ノイズパラメータの勾配比較
- 悪条件行列での勾配安定性
```

## 6.2 プロパティベーステスト

オンライン更新を含むなら、固定されたテストケースだけでなく、ランダムな操作列を試したいです。

```
insert
insert
delete
insert
delete
...
```

各段階で、

```
incremental result == full refit result
```

を確認します。

特に、PointIdと内部インデックスの整合性は、ランダムな削除順序で壊れやすい部分です。

## 6.3 ベンチマークの基準を先に決める

「最も高速なGaussian Process Regressionライブラリ」を目指すなら、ベンチマークの定義が必要です。

例：

|
指標

|

内容

|
| --- | --- |
|

Fit時間

|

n, d, カーネル別

|
|

Predict時間

|

テスト点数別

|
|

メモリ使用量

|

Workspace込み

|
|

Allocations

|

fit/predict中

|
|

並列スケーリング

|

スレッド数別

|
|

f32/f64

|

精度と速度

|
|

Online insert

|

1点追加

|
|

Online delete

|

任意点削除

|

高速化の目標値がないと、Phase 2以降の設計判断が難しくなります。

# 7. 追加で気になる点

## 7.1 `Mat<T>`のメモリレイアウトは実際のfaer APIで確認する

設計書では、

> `Mat<T>`は列優先、行ストライドは常に1

としています。

ただし、`Mat<T>`と`MatRef<T>`、`MatMut<T>`のストライドやビューの扱いは、使用するfaerのバージョンに依存します。

特に、

Rust

```
MatRef<T>
MatMut<T>
```

をカーネルAPIに公開する場合、以下を確認してください。

* 行列の形状

* 列優先・行優先

* ストライド

* 対称行列の扱い

* `MatMut`への書き込みの制約

設計書で固定するのではなく、使用するfaerバージョンのAPIに合わせるのが安全です。

## 7.2 `std::simd`は採用時期に注意

`std::simd`を使う方針は理解できますが、Rustのバージョン・feature状況に依存します。

また、SIMDを自前で実装するより、まずはfaerとRayonの性能を測定した方が良いです。

カーネル評価のSIMD化が本当にボトルネックなのか、距離行列構築やCholeskyが支配的なのかは、ベンチマークで判断すべきです。

## 7.3 `MathBackend`の抽象化は慎重に

Rust

```
trait MathBackend<T: Scalar> {
    fn exp_inplace(&self, buf: &mut [T]);
}
```

この抽象化は悪くありません。

ただし、`exp`を差し替えると、MLL・勾配・予測結果の再現性や数値安定性に影響します。

そのため、`FastApprox`はPhase 5で十分です。

# 8. 私ならこう修正する

## 実装開始前に確定させる項目

### P0：必須

GaussianLikelihoodのパラメータ化と勾配式を統一する

faer Cholesky insert/delete APIを小規模行列で検証する

混合精度の残差行列Aの構築方法を確定する

jitter適用時の解と反復改良の仕様を明確化する

### P1：Phase 1〜2で整理

カーネルパラメータのflatten / binding設計

CompiledKernelの静的ディスパッチ方針

バッチWorkspaceとオンラインWorkspaceの責務分離

Predictionの潜在分散・観測分散の意味を確定する

MLL・勾配・ランダム操作列のテストを追加する

# 9. 最終結論

この設計書は、以前のレビューで問題だった点をかなり改善した、実装に進める設計になっています。

特に良いのは、

* Exact GPRを最初に完成させる

* 高速化を後回しにする

* Sparse GPRを別モデルとして扱う

* オンライン更新を独立経路にする

* 数値安定化とモデルパラメータを分離する

* テスト計画を明示する

という構成です。

一方、最優先で修正したいのは次の4点です。

1. GaussianLikelihoodのパラメータ化と勾配式の矛盾

2. faerのCholesky更新APIと数学式の実装整合性

3. 混合精度反復改良におけるAの高精度残差計算

4. カーネルパラメータの最適化器への接続

これらを確定すれば、Phase 1の実装に進んで問題ないと思います。

設計全体としては、現時点では「汎用性・高速性を追求するGPRライブラリ」の方向性は良いです。ただし、最も柔軟かつ最も高速という目標を達成するには、まずf64 Exact GPRの正確な基準実装を完成させ、そこからベンチマークに基づいて最適化を進めるのが重要です。
