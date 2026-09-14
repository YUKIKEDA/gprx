# gprx 設計ドキュメント レビュー

Rust製Gaussian Process Regression（GPR）ライブラリ `gprx` の設計書をレビューしました。

総評：方向性は良いですが、現状は「実装可能な設計仕様」よりも「高性能GPRの技術構想」に近いです。

特に、`faer`を中心とした線形代数、Exact GP / Sparse GPの抽象化、カーネルのコンパイル、オンライン更新、混合精度まで考慮している点は評価できます。一方で、GPRの数値計算として重要な前提の誤りや、実装時に破綻する可能性のあるAPI・メモリ設計が残っています。

以下、設計書の内容を基に、優先度の高い順に指摘します。

要修正

## レビュー判定

最優先で見直すべき領域

P0

混合精度反復改良の理論・実装

収束条件、残差計算、jitterの扱いに重大な問題があります。

P0

オンライン追加・削除の正当性

Cholesky更新APIの前提、削除後の順序、alpha更新を検証する必要があります。

P1

Exact GP / Sparse GPの抽象化

`Inference` traitでは、実際の尤度・勾配・予測分散の差異を十分表現できていません。

P1

カーネル・精度・メモリの設計整合性

API上の型やバッファの扱いに不整合があります。

## 1. 最重要：混合精度反復改良に重大な問題

対象：§4、§4.1（40〜87行）

設計書では、f32でCholesky分解し、f64で残差を計算することで高速化するとしています。

```
K_f32 → Cholesky(f32)
           ↓
       alpha_0
           ↓
r = y - K_f64 @ alpha_0
           ↓
       delta
           ↓
alpha_1 = alpha_0 + delta
```

この方向性自体は妥当ですが、現状の説明には理論的な誤りと、実装上の重要な抜けがあります。

### 1.1 残差の計算式が間違っている

設計書の55行目：

```
r = y - K_f64 @ alpha_0
```

しかし、f32 Choleskyの近似解 `alpha_0` が何を解いているかで、残差の計算は変わります。

通常のExact GPでは、

(K+σI)α=y(K+\sigma I)\alpha=y(K+σI)α=y

を解きます。

jitterを含む実際のシステム行列を

A=K+σIA=K+\sigma IA=K+σI

とした場合、残差は

r=y−Aα0r=y-A\alpha_0r=y−Aα0

です。

`K`だけで残差を計算してしまうと、jitterを含む実際の線形方程式に対する残差になりません。

これは、反復改良の収束判定にも影響します。

### 修正案

```
A = K + jitter * I

alpha_0 = solve_f32(L, y)

r = y - A_f64 @ alpha_0

delta = solve_f32(L, r)

alpha_1 = alpha_0 + delta
```

ただし、`A_f64`を常時保持する必要はありません。残差計算時に、対角要素へjitterを加えて計算する方法もあります。

### 1.2 「jitterを10倍にすれば収束する」は保証されない

77行目では、

> RefinementNotConverged時はjitterを10倍にしてf32分解からやり直す

としています。

これは安全な一般解ではありません。

jitterを増やすと、実際に解いている行列が変わります。

(K+σI)α=y(K+\sigma I)\alpha=y(K+σI)α=y

で、σを増やせば別のGPモデルになります。

特に、GPRではjitterは単なる数値計算の補助ではなく、観測ノイズや正則化に関わるパラメータです。

数値安定化のためにjitterを増やすことと、モデルのノイズを変更することは分離すべきです。

修正案：

Rust

```
enum RefinementFallback {
    IncreaseNumericalJitter {
        max_retries: usize,
    },
    FallbackToDoublePrecision,
    ReturnError,
}
```

さらに、jitterを増やす場合は、モデルのnoise varianceと数値安定化用jitterを分けて管理してください。

Rust

```
struct NumericalStability {
    noise_variance: f64,
    jitter: f64,
    max_jitter: f64,
}
```

### 1.3 反復改良の収束条件が単純化されすぎている

65行目：

> 収束条件 κ(K)·u_f < 1

これは、反復改良の説明として不十分です。

実際の収束は、以下に依存します。

* 実際の分解誤差

* 行列の条件数

* f32の丸め誤差

* 残差計算の精度

* 前進・後退代入の誤差

* 行列の対称性・正定値性

* f32のCholesky分解が成立しているか

特に、GPRのカーネル行列はlengthscaleやデータの重複によって非常に悪条件になることがあります。

修正案としては、条件数の理論値を直接使うより、

Rust

```
struct RefinementConfig {
    max_iterations: usize,
    relative_tolerance: f64,
    stagnation_ratio: f64,
    fallback: RefinementFallback,
}
```

とし、実際の残差で判定するのがよいです。

また、f32のCholeskyが成立したからといって、f64相当の精度で解けるとは限りません。

混合精度は後付けではなく、数値安定性の独立した設計項目として扱うべきです。

## 2. オンライン学習の設計は、まだ成立性の検証が必要

対象：§3、§11（33〜35、378〜435行）

オンライン学習で、

> 1点追加：O(n²)

を目指しているのは良いです。

ただし、ここは設計書の中でもかなり危険な部分です。

### 2.1 faerのCholesky更新APIの利用前提を検証すべき

設計書では、

```
llt::update::{insert,delete}_rows_and_cols_clobber
```

を利用して、分解済みLに対して行・列を追加削除するとしています。

しかし、実装上は次を確認する必要があります。

1. APIが要求する行列形式は何か。

2. 追加する行列要素はどの形式で渡すのか。

3. 追加後の行列が正定値であることをどう保証するか。

4. 削除対象が任意のインデックスでも動作するか。

5. 更新後のLが、元の行列の正しいCholesky因子になるか。

6. 更新APIが変更するバッファの範囲。

7. faerのバージョンによるAPI差異。

ここは、実際に小さな行列でテストするべきです。

例えば、

K=\(2112\)K= \begin{bmatrix} 2 & 1\\ 1 & 2 \end{bmatrix}K=\(21​12​\)

に1点追加し、更新後のLとフルCholeskyの結果を比較するテストを作ってください。

### 2.2 新しい点を追加する場合の行列更新が不完全

設計書407行目：

```
①距離計算
②カーネル評価しK行列に新規行/列追加
③insert_rows_and_cols_clobberでL更新
④alpha再ソルブ
```

この順番だけでは、実際の更新に必要な情報が不足しています。

新しい点を追加した行列は、

Knew=\(KkkTknew\)K_{new}= \begin{bmatrix} K & k\\ k^T & k_{new} \end{bmatrix}Knew=\(KkT​kknew​​\)

です。

既存のCholesky因子をLとして、

Lnew=\(L0vTd\)L_{new}= \begin{bmatrix} L & 0\\ v^T & d \end{bmatrix}Lnew=\(LvT​0d​\)

とする場合、

Lv=kLv=kLv=k

d=knew−vTvd=\sqrt{k_{new}-v^Tv}d=knew−vTv

という関係を満たす必要があります。

特に重要なのは、新しい点とのカーネル値を計算しただけではなく、更新後のLが正しい因子になっているかです。

APIを利用する場合でも、上記の数学的関係を検証するテストが必要です。

### 2.3 削除時の順序に問題がある

設計書409行目：

```
①delete_rows_and_cols_clobberでL更新
②距離キャッシュ・K・y・alphaから該当要素を除去
③alpha再ソルブ
```

これは、内部行列のインデックス管理と整合している必要があります。

例えば、3点から2点目を削除すると、

```
Before:
index 0 → Point A
index 1 → Point B
index 2 → Point C

After:
index 0 → Point A
index 1 → Point C
```

となります。

`L`だけでなく、以下のすべてを同じ順序で更新しなければなりません。

* `K`

* `L`

* `y`

* `alpha`

* 距離キャッシュ

* PointRegistry

特に、`alpha`を単純に該当要素削除するだけでは、正しい新しい解にはなりません。

設計書ではalpha再ソルブを行っているので方向性は良いですが、削除後の行列・yの順序とPointRegistryの対応を不変条件として明示してください。

### 2.4 オンライン更新後のハイパーパラメータ最適化

435行目では、

> 定期的に通常のfitを挟んでハイパーパラメータを追従させる

としています。

これは妥当です。

ただし、APIとしては以下を分けた方がよいです。

Rust

```
trait OnlineInference<T> {
    fn insert(&mut self, x: &[T], y: T) -> Result<PointId, GpError>;

    fn delete(&mut self, id: PointId) -> Result<(), GpError>;

    fn refit_hyperparameters(
        &mut self,
        optimizer: &mut dyn Optimizer<T>,
    ) -> Result<(), GpError>;
}
```

`insert/delete`が、現在のカーネルとハイパーパラメータを使って更新するだけなのか、再最適化も含むのかを明確に分けると使いやすいです。

## 3. カーネル設計：柔軟性と高速性の両立が難しい

対象：§5（89〜208行）

カーネルの宣言層と実行層を分ける設計は良いです。

Rust

```
KernelSpec
    ↓
CompiledKernel
    ↓
Workspace
```

この方向は、Rustで高速な数値計算ライブラリを作るうえで合理的です。

ただし、いくつか修正すべき点があります。

### 3.1 `KernelTerm`の`dist: MatRef<f64>`が精度設計と矛盾する

設計書102〜105行目：

Rust

```
trait KernelTerm {
    fn distance_kind(&self) -> DistanceKind;
    fn apply(&self, dist: MatRef<f64>, out: MatMut<f64>);
    fn params(&self) -> &[f64];
    fn grad(&self, dist: MatRef<f64>, dK: MatMut<f64>, param_idx: usize);
}
```

一方、§4ではf32/f64の精度切り替えを設計しています。

しかし、カーネルの入力が常に`f64`だと、

* f32で距離キャッシュを保持したい

* f32でカーネル行列を構築したい

* f32のSIMDを最大限活用したい

という設計が難しくなります。

修正案：

Rust

```
trait KernelTerm<T: Scalar> {
    fn apply(
        &self,
        dist: MatRef<T>,
        out: MatMut<T>,
    );

    fn grad(
        &self,
        dist: MatRef<T>,
        dK: MatMut<T>,
        param_idx: usize,
    );
}
```

ただし、ユーザー定義カーネルを`dyn KernelTerm<T>`として利用する場合は、trait objectとジェネリクスの関係を整理する必要があります。

### 3.2 `KernelSpec`と`CompiledKernel<T>`の関係が不明瞭

設計書では、

Rust

```
enum KernelSpec {
    Leaf(Box<dyn KernelTerm>),
    Sum(Box<KernelSpec>, Box<KernelSpec>),
    Product(Box<KernelSpec>, Box<KernelSpec>),
}
```

となっています。

一方で、`CompiledKernel<T>`は型パラメータを持つ設計です。

ここで、以下を明確にする必要があります。

* `KernelSpec`はf32/f64に依存しない宣言層なのか

* `CompiledKernel<T>`は精度ごとに生成されるのか

* ユーザー定義カーネルの`T`はどこで固定されるのか

* パラメータの型はf32なのかf64なのか

* `params()`の返却型をどうするのか

個人的には、宣言層は型を固定しない方が扱いやすいです。

Rust

```
trait KernelTerm: Send + Sync {
    fn params(&self) -> &[f64];
}
```

実行層では、

Rust

```
trait KernelEvaluator<T: Scalar> {
    fn eval(
        &self,
        dist: MatRef<T>,
        out: MatMut<T>,
    );
}
```

のように分ける方が、精度設計とユーザー定義カーネルの責務が整理しやすいと思います。

### 3.3 Productのバッファ数の説明が危険

148行目：

> 必要バッファ数はネストの深さでしか増えない(実用的な合成ではまず3を超えない)

これは一般論としては成立しません。

例えば、

```
(A * B) * (C * D)
```

や、

```
(A + B) * (C + D) * (E + F)
```

のように、複雑な合成では一時バッファが増えます。

また、最適化器がパラメータ変更を行う場合、再計算戦略によって必要なバッファ数が変わります。

ここは、

Rust

```
struct WorkspacePlan {
    max_buffers: usize,
    max_bytes: usize,
}
```

のように、plan構築時に実際の最大同時使用数を計算する設計でよいです。

「深さ3以内」などの経験則を、設計上の保証のように書かない方がよいです。

## 4. Exact GP / Sparse GPの抽象化は、もう一段整理が必要

対象：§6、§6.1（222〜264行）

Rust

```
trait Inference<T: Scalar> {
    fn fit(...);
    fn predict(...);
    fn objective(...);
}
```

という形は理解しやすいですが、実際のGPRではExact GPとSparse GPの内部構造がかなり異なります。

### 4.1 `objective()`をtraitに含めるのは結合が強い

Exact GPとSparse GPでは、尤度の計算方法や勾配の構造が異なります。

`Inference`が直接、

Rust

```
fn objective(&mut self) -> &mut dyn Objective<T>;
```

を持つと、

* 推論モデル

* ハイパーパラメータ最適化

* Workspace

* カーネル

* Optimizer

が強く結合しやすくなります。

修正案：

Rust

```
trait Inference<T: Scalar> {
    fn fit(
        &mut self,
        x: MatRef<T>,
        y: &[T],
    ) -> Result<(), GpError>;

    fn predict(
        &self,
        xs: MatRef<T>,
    ) -> Result<Prediction<T>, GpError>;
}
```

そして、

Rust

```
trait Objective<T: Scalar> {
    fn value(&mut self, params: &[T]) -> Result<T, GpError>;

    fn gradient(
        &mut self,
        params: &[T],
    ) -> Result<Option<Vec<T>>, GpError>;
}
```

と分離する方がよいです。

### 4.2 Sparse GPの予測分散が未定義

Exact GP / Sparse GPのpredictは、単に

Rust

```
(mean, variance)
```

だけでは不十分な可能性があります。

Sparse GPでは、

* 近似手法

* 予測共分散

* 対角分散

* フル共分散

* 観測ノイズ込みの分散

* latent functionの分散

をどう扱うかが重要です。

APIとしては、

Rust

```
struct Prediction<T> {
    mean: Vec<T>,
    variance: Vec<T>,
}
```

に加えて、将来的には

Rust

```
enum Covariance {
    Diagonal,
    Full,
}
```

のような拡張を検討してもよいと思います。

ただし、最初からフル共分散を実装する必要はありません。Exact GPの対角予測分散を最初の完成目標にするのが現実的です。

## 5. メモリ設計：単一アリーナの思想は良いが、実装上の不整合がある

対象：§7（266〜302行）

単一アリーナを使い、

> fit開始時の1回だけ確保

とする思想は、性能最優先のライブラリとして良いです。

ただし、設計書のWorkspace定義には問題があります。

### 5.1 `Workspace<T>`に距離キャッシュの型がない

Rust

```
struct Workspace<T: Scalar> {
    arena: Vec<T>,
    dist_cache_offset: usize,
    k_matrix_offset: usize,
    exp_buf_offset: usize,
    faer_scratch_offset: usize,
    thread_scratch: Vec<(usize, usize)>,
}
```

これでは、距離キャッシュがf32なのかf64なのかが、`T`に依存します。

混合精度で、

```
Storage = f32
Refine = f64
```

とするなら、

* 距離キャッシュ：f32

* K：f32

* L：f32

* 残差計算：f64

* y：f64?

* alpha：f32 or f64?

のような設計が必要です。

したがって、`PrecisionPolicy`をWorkspaceにも明示的に反映させるべきです。

Rust

```
struct Workspace<P: PrecisionPolicy> {
    storage_arena: Vec<P::Storage>,
    refine_arena: Vec<P::Refine>,
}
```

あるいは、最初はDoublePrecisionだけ実装して、混合精度の導入時にWorkspaceを拡張する方が安全です。

### 5.2 `Mat<T>`をarenaの中に直接置く設計が不明瞭

faerの`Mat<T>`を使う場合、

Rust

```
k_matrix_offset: usize
```

からビューを生成する方式と、

Rust

```
k_matrix: Mat<T>
```

を保持する方式を混在させない方がよいです。

特にfaerの`Mat`は、単なる`Vec<T>`ではなく、内部のstrideやcapacity管理を持つため、単純にアリーナの一部として扱う場合は慎重な検証が必要です。

修正案として、初期実装では、

Rust

```
struct Workspace<T: Scalar> {
    k_matrix: Mat<T>,
    dist_cache: Mat<T>,
    exp_buf: Vec<T>,
    faer_scratch: Vec<T>,
}
```

のようにして、各バッファの所有権を明確にする方が安全です。

その後、プロファイリングでアリーナ化の効果が確認できた箇所だけ変更するのがよいと思います。

### 5.3 「fit中のアロケーションゼロ」は過剰な制約になり得る

22行目では、

> バッチfitのアロケーションはfit開始時の1回のみ

としています。

これは理想としては理解できますが、最適化器やユーザー定義カーネルまで含めて保証するのは難しいです。

例えば、ユーザーが作成した`KernelTerm::apply`内でアロケーションする可能性があります。

そのため、仕様としては、

> gprx内部のホットパスでは新規アロケーションを行わない

とする方が現実的です。

ユーザー定義カーネルについては、

* アロケーション禁止を要求するのか

* `Workspace`を渡すのか

* 安全なAPIとunsafeな高速APIを分けるのか

を決める必要があります。

## 6. `MathBackend`の設計は、精度と正確性をもっと厳密にすべき

対象：§8（305〜322行）

Rust

```
trait MathBackend<T: Scalar>: Send + Sync {
    fn exp_inplace(&self, buf: &mut [T]);
    fn erf_inplace(&self, buf: &mut [T]);
}
```

この設計は、SIMD・数学関数の差し替えとして面白いです。

ただし、以下の点が問題になります。

### 6.1 `PolyApproxExp`をデフォルトにするのは危険

322行目：

> デフォルトはPolyApproxExp(依存ゼロ維持)

GPRでは、カーネル行列の各要素を計算するために`exp`を使います。

RBFカーネルの場合、

k(x,x′)=σ2exp⁡(−∥x−x′∥22ℓ2)k(x,x')=\sigma^2\exp\left(-\frac{\|x-x'\|^2}{2\ell^2}\right)k(x,x′)=σ2exp(−2ℓ2∥x−x′∥2)

です。

`exp`の近似誤差は、カーネル行列全体の誤差につながり、最終的に、

* Choleskyの安定性

* 尤度

* 勾配

* 予測値

* ハイパーパラメータ最適化

に影響します。

特に、カーネル行列の正定値性を壊す可能性があるため、近似精度だけでなく、行列の性質を壊さないことが重要です。

修正案：

Rust

```
enum MathMode {
    Accurate,
    FastApprox,
}
```

デフォルトは正確な数学関数を使用し、近似は明示的なfeatureや設定で有効化する方がよいです。

### 6.2 `erf`が必要なカーネルを明確にする

一般的なRBF / Matern / Periodicカーネルでは、通常`erf`は必須ではありません。

`erf`をどのカーネルで使用するのか、設計書に具体例があるとよいです。

不要なAPIを先に増やすより、最初は、

Rust

```
trait MathBackend<T> {
    fn exp(x: T) -> T;
}
```

だけで始めてもよいと思います。

## 7. ARDキャッシュの判断基準は再検討が必要

対象：§5.2.1（122〜140行）

Rust

```
recompute_relative_cost = d as f64 / n as f64;
```

として、d/nを基準にキャッシュするか判断しています。

これは概算としては理解できますが、設計上の決定基準としては不十分です。

実際のコストは、

* n

* d

* カーネルの種類

* SIMDの効率

* メモリ帯域

* CPUキャッシュ

* スレッド数

* キャッシュのデータ型

* カーネルの再評価回数

に依存します。

例えば、RBFとPeriodicでは、同じd/nでも`exp`と三角関数のコストが大きく異なります。

また、`n=5000,d=100,f64`で20GBになるという例は正しいですが、通常のExact GPのK行列自体が約200MBです。

したがって、距離テンソルをキャッシュするより、Kの構築中に距離を計算する方が良い場合も多いです。

ここは理論式で閾値を決めるより、ベンチマークで決めるべきです。

修正案：

Rust

```
enum DistanceCachePolicy {
    Never,
    Always,
    Auto {
        memory_budget_bytes: usize,
    },
}
```

`Auto`は、

* メモリ上限

* カーネル種別

* n,d

* 予測される再計算回数

を考慮する方式にするとよいと思います。

## 8. Optimizer APIの設計を改善したい

対象：§9（324〜342行）

Rust

```
trait Objective<T: Scalar> {
    fn value(&mut self, params: &[T]) -> T;
    fn gradient(&mut self, params: &[T]) -> Option<Vec<T>>;
}
```

このAPIは、最初の実装としてはシンプルです。

ただし、次の問題があります。

### 8.1 `value()`と`gradient()`が別々に呼ばれる可能性

`value_and_gradient()`で共有するとしていますが、

Rust

```
fn value_and_gradient(&mut self, params: &[T]) -> (T, Option<Vec<T>>) {
    (self.value(params), self.gradient(params))
}
```

では、内部的には別々の処理です。

コメントで、

> Cholesky分解(L, alpha)を尤度と勾配の両方で共有

としていますが、この実装では保証されません。

修正案：

Rust

```
trait Objective<T: Scalar> {
    fn value(
        &mut self,
        params: &[T],
    ) -> Result<T, GpError>;

    fn value_and_gradient(
        &mut self,
        params: &[T],
    ) -> Result<(T, Option<Vec<T>>), GpError>;
}
```

`value_and_gradient`を実際に共有計算するAPIとして実装し、必要に応じて`gradient()`はデフォルト実装にするのがよいと思います。

### 8.2 `Vec<T>`の返却はアロケーション方針と矛盾する

Rust

```
fn gradient(&mut self, params: &[T]) -> Option<Vec<T>>;
```

最適化器の反復ごとに`Vec<T>`を返すと、アロケーションが発生します。

アロケーションを抑えるなら、

Rust

```
fn gradient(
    &mut self,
    params: &[T],
    out: &mut [T],
) -> Result<(), GpError>;
```

とする方がよいです。

ただし、最初から全APIでアロケーションゼロを目指す必要はありません。

高性能な内部APIと、使いやすい公開APIを分けるのが現実的です。

## 9. エラー設計は良いが、数値計算の失敗理由を増やしたい

対象：§10（344〜376行）

`GpError`を設計している点は良いです。

特に、

* DimensionMismatch

* InsufficientData

* CholeskyFailed

* RefinementNotConverged

* CoordGradientUnsupported

* OptimizationNotConverged

* InvalidHyperparameter

は、ライブラリとして必要なエラーです。

ただし、以下も追加した方がよいです。

Rust

```
enum GpError {
    EmptyInput,
    NonFiniteInput,
    NonFiniteKernelValue,
    NonPositiveDefiniteMatrix,
    InvalidNoiseVariance,
    InvalidLengthscale,
    InvalidOutputscale,
    NumericalOverflow,
    NumericalUnderflow,
    UnsupportedKernelOperation,
    WorkspaceTooSmall,
    InvalidPointId,
}
```

特に、GPRではNaN / Infの入力や、カーネル評価結果の非有限値が問題になることがあります。

また、`CholeskyFailed`は、

Rust

```
CholeskyFailed { jitter: f64 }
```

だけでなく、どの段階で失敗したかが分かるとデバッグしやすいです。

Rust

```
CholeskyFailed {
    jitter: f64,
    matrix_size: usize,
    stage: CholeskyStage,
}
```

## 10. 実装前に追加すべきテスト

この設計書で最も不足しているのは、数値計算の正当性を保証するテスト設計です。

GPRライブラリでは、速度よりもまず「正しい結果が出ること」が重要です。

### 必須テスト

### テスト計画

1. カーネルの数学的正当性

* RBF / Matern / Periodic

* 対称性

* 対角値

* 既知の値との比較

* 数値微分と勾配の比較

2. Choleskyの正当性

* `K = LLᵀ`の再構成誤差

* jitterあり・なし

* 近似的に特異な行列

* 重複データ

3. オンライン更新

* 1点追加とフル再fitの一致

* 1点削除とフル再fitの一致

* 任意インデックス削除

* 追加・削除の繰り返し

* PointIdと内部インデックスの整合性

4. 精度

* f32 / f64の比較

* 混合精度とf64の比較

* 悪条件行列

* 収束しないケース

* jitter増加時の挙動

5. 推論結果

* 既知の小規模GPRと比較

* mean

* variance

* log marginal likelihood

* gradient

# 優先度付き修正リスト

|
優先度

|

修正項目

|
| --- | --- |
|

P0

|

混合精度反復改良の残差式・収束条件を修正

|
|

P0

|

jitterとnoise varianceを分離

|
|

P0

|

faer Cholesky更新APIの実動作を検証

|
|

P0

|

オンライン追加削除とフル再fitの一致テスト

|
|

P1

|

Exact / Sparse GPのAPIを整理

|
|

P1

|

KernelTermの精度型とCompiledKernelの関係を整理

|
|

P1

|

Workspaceの所有権・アリーナ設計を明確化

|
|

P1

|

MathBackendのデフォルトを正確な実装にする

|
|

P2

|

ARDキャッシュ閾値をベンチマークベースに変更

|
|

P2

|

Optimizerのvalue/gradient共有APIを改善

|
|

P2

|

エラー型の拡充

|
|

P2

|

ベンチマーク・数値正当性テストを設計書に追加

|

## 私なら、実装順序をこう変更します

現状の設計は、最初から多くの高度な機能を同時に実現しようとしています。

特に、

* 混合精度

* Sparse GP

* オンライン学習

* カーネルのIncrementalRecompute

* SIMD数学バックエンド

* アリーナアロケータ

を一度に進めると、どこに問題があるのか分からなくなりやすいです。

### Phase 1：正しいExact GP

* f64のみ

* RBF / Matern

* `faer` Cholesky

* log marginal likelihood

* 勾配

* 予測mean / variance

* 基本Optimizer

* 数値テスト

### Phase 2：高速化

* カーネルのCompiledKernel

* 距離キャッシュ

* Workspace再利用

* Rayon

* SIMD

* ベンチマーク

### Phase 3：オンライン学習

* 追加

* 削除

* PointId

* フル再fitとの比較テスト

### Phase 4：Sparse GP

* VFEまたはFITCのどちらか一つ

* 誘導点固定

* 予測

* ハイパーパラメータ最適化

### Phase 5：高度な最適化

* 混合精度

* IncrementalRecompute

* 低ランク更新

* 数学関数バックエンド

* 自動キャッシュポリシー

## 最終評価

設計の発想はかなり良いです。

特に、単なるGPRラッパーではなく、

* 高速なカーネル実行計画

* 精度の切り替え

* オンライン更新

* Exact / Sparse GPの差し替え

* Rustらしい所有権・静的ディスパッチ

まで考えているのは、実用的な数値計算ライブラリを作ろうとしている設計として評価できます。

ただし、現状では以下の3点を最優先で修正したいです。

1. 混合精度反復改良の数学的設計を修正する。

2. オンラインCholesky更新を実装前に小規模行列で検証する。

3. f64 Exact GPを正しいリファレンス実装として先に完成させる。

この3つができれば、その後の高速化・Sparse GP・オンライン学習を安心して積み上げられる設計になります。

レビュー対象：gprx-design.md（全442行）
