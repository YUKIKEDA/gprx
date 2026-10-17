[English](README.md) | 日本語

# gprx

gprx は、Rust でガウス過程回帰を行うクレートである。全学習点を使う厳密な GPR、スパース GPR（VFE）、SVGP の 3 つを備えている。`Gpr` は学習前のトレーナーである。`Gpr::fit` はこのトレーナーを消費し、負の対数周辺尤度を既定の `Lbfgs` で最小化したうえで `FittedGpr` を返す。`Sgpr` と `Svgp` も、同じ部品で組み立てられている。厳密な GPR とスパース GPR のモデルは学習後もオンラインで更新でき、どのモデルもディレクトリへ保存して読み込める。カーネルに渡す入力は座標でもよいし、呼び出し側が求めた二乗距離でもよい。

入力 `X` は列優先で並べる。`n` 個の点と `d` 個の特徴があるなら、全行の特徴 0 を先に置き、そのあとに特徴 1 を続ける。`fit` はトレーナーを値で受け取って消費する。観測ノイズの大きさは `GaussianLikelihood` が持つ。公開 API として約束しているのは、**0.2.0**（タグ `v0.2.0`）のうち既定のフィーチャで使える部分である。MSRV は 1.88 とする。0.x のあいだは、マイナー版の更新でもこの API に互換性のない変更が入ることがある。0.1.0 から 0.2.0 で何が変わり、どう移行すればよいかは[変更履歴](https://github.com/YUKIKEDA/gprx/blob/main/CHANGELOG.md)（英語）にまとめてある。`internals`（`bench-internals` と `insert-stages`）は、この約束に含まれない。

```toml
[dependencies]
gprx = "0.2"
```

## 例

```rust
use gprx::kernel::{KernelSpec, RbfKernel};
use gprx::{GaussianLikelihood, Gpr};

fn main() -> Result<(), gprx::GprError> {
    let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    let likelihood = GaussianLikelihood::new(0.1)?;
    let gpr = Gpr::new(kernel, likelihood);
    let fitted = gpr.fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0])?;
    let pred = fitted.predict(&[0.5], 1, 1)?;
    println!("mean = {}, variance = {}", pred.mean[0], pred.variance[0]);
    Ok(())
}
```

同じプログラムは `cargo run --example fit_predict` で実行できる。

## 使用方法

`fit` と `factor` は、失敗したときにトレーナーをエラーと一緒に返す。戻り値の型は `Result<Fitted, (Trainer, GprError)>` である。`Gpr` には `From<(Gpr<O, P, K>, GprError)> for GprError` があるので、`?` で受ければトレーナーは捨てられ、`GprError` だけが呼び出し元へ伝わる。`Sgpr` と `Svgp` にはこの変換がないため、以下の例と同じく `.map_err(|(_, e)| e)?` と書く。同じトレーナーで再試行したいなら、`Err((trainer, err))` のパターンで受けてトレーナーを取り戻せばよい。

モデルのメソッドは、`fitted.predict(..)` のように値に対して呼ぶのが基本である。型のパスから呼ぶときは注意が要る。座標のモデルと距離のモデルには、同じ名前のメソッドがいくつもある。`fit`、`factor`、`predict` / 共分散 / `sample` の系統、`insert`、`insert_inducing`、疎なモデルの `into_online` と `save` である。これらを `FittedGpr::predict(&fitted, ..)` のように型パラメータを省いて呼ぶと、`E0034` でコンパイルに失敗する。パスだけでは、同じ名前のどのメソッドを指すのかが決まらないからである。この場合は値が持つ型パラメータまで書き、既定の型パラメータの `FittedGpr` なら `<FittedGpr>::predict(&fitted, ..)`、`factor` で作ったモデルなら `<FittedGpr<Fixed>>::predict(&frozen, ..)` とする。

入力 `X` と誘導点 `Z` は、列優先に並べた `f64` の配列で渡す。全行の特徴 0 が先で、特徴 1 がその次に来る。点の数を `n_rows`、特徴の数を `n_cols` で指定し、`y` の長さは `n_rows` にそろえる。入力が空のとき、長さが `n_rows * n_cols` と一致しないとき、`NaN` や `Inf` を含むときは、いずれも `GprError` になる。

scikit-learn の `n_jobs` に当たるスレッド数の設定は、このクレートにはない。距離の計算はプロセス全体で共有する Rayon のプールで行うので、プールの大きさがそのままスレッド数になる。大きさを変えるには、プロセスを起動する前に `RAYON_NUM_THREADS` を設定するか、最初の fit や predict より前に `rayon::ThreadPoolBuilder::new().num_threads(n).build_global()` を呼ぶ。プールを初期化できるのは一度だけで、ワーカーを 1 つにすれば計算は逐次になる。

線形代数のバックエンドは、プール全体ではなく、問題の大きさに見合った数のワーカーで動く。`n` 行 `k` 列の分解や求解に使うワーカーは `min(プール, n / 64, n · k / 16384, k / 12)` 個で、少なくとも 1 個は使う。学習時のコレスキー分解と、勾配の計算に必要な `n` 個の右辺の求解では `k = n` になる。予測や共分散の求解では、クエリ点 1 つにつき 1 列である。カーネルの値を埋める処理は、変わらずプール全体を使う。

`gprx::internals`（`bench-internals`、`insert-stages`）はセマンティックバージョニングの対象外なので、このモジュールに依存するコードは書かないこと。

### 全学習点: `Gpr`、`FittedGpr`、`OnlineGpr`

`KernelSpec` を渡して `Gpr::new(kernel, likelihood)` を作ると、型は `Gpr<Lbfgs, DoublePrecision, KernelSpec>` になる。`fit(x, n_rows, n_cols, y)` はこのトレーナーを消費して負の対数周辺尤度を最適化し、型パラメータが同じ `FittedGpr` を返す。

```rust
use gprx::kernel::{KernelSpec, RbfKernel};
use gprx::{
    DoublePrecision, Fixed, FittedGpr, GaussianLikelihood, Gpr, Lbfgs, PredictOptions, Prediction,
    VarianceKind,
};

fn main() -> Result<(), gprx::GprError> {
    let trainer: Gpr<Lbfgs, DoublePrecision, KernelSpec> = Gpr::new(
        KernelSpec::from(RbfKernel::new(1.0)?),
        GaussianLikelihood::new(0.1)?,
    );
    let fitted: FittedGpr = trainer.fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0])?;
    // The same call by path, with the type's defaults.
    let _ = <FittedGpr>::predict(&fitted, &[0.5], 1, 1)?;

    let pred = fitted.predict(&[0.5], 1, 1)?;
    let latent = fitted.predict_with(
        &[0.5],
        1,
        1,
        PredictOptions {
            variance_kind: VarianceKind::Latent,
        },
    )?;
    let _ = (pred.mean[0], pred.variance[0], latent.variance_kind);

    let mut fitted = fitted;
    let mut reused = Prediction::default();
    fitted.predict_into(&[0.5], 1, 1, &mut reused)?;
    fitted.predict_with_into(&[0.5], 1, 1, PredictOptions::default(), &mut reused)?;

    let frozen = fitted
        .into_trainer()
        .with_optimizer(Fixed)
        .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0])?;
    let _ = frozen.n();
    Ok(())
}
```

`predict` が返す分散は `VarianceKind::Observation`（`latent + σn²`）で、観測値の分散を元の `y` の尺度で表す。`VarianceKind::Latent` を選ぶと `σn²` が除かれる。この選択は、`predict_with` に渡す `PredictOptions` のフィールド `variance_kind` で行う。`predict` は呼ぶたびに結果のバッファを確保する。確保を避けるには `predict_into` か `predict_with_into` を使う。この 2 つは渡した `Prediction`（フィールドは `mean`、`variance`、`variance_kind`）へ結果を書き込み、クエリの長さが前回と同じなら `mean` と `variance` の領域を使い回す。

クエリ点どうしの共分散まで要るなら `predict_covariance` を使う。返ってくる `PredictiveCovariance` のフィールドは `mean`、`covariance`、`variance_kind` で、`covariance` は列優先の `m × m` 行列（添字は `col * m + row`）になっている。その対角は、同じクエリとオプションで呼んだ `predict` の分散と一致する。オプションを指定したいときは `predict_covariance_with` に `PredictOptions` を渡す。

`sample(xs, n_rows, n_cols, n_draws, seed)` は、この共分散から `μ + Lz` の形でサンプルを取り出し、列優先の `m × n_draws` で返す。`seed` は gprx 内蔵の Xoshiro256++ の開始状態なので、どのプラットフォームでも同じ乱数列が得られる。`PredictOptions` を指定したいときは `sample_with` を使う。

`loo_predict` が返すのは新しい `x` での予測ではなく、全学習点での GPML 流の leave-one-out の平均と分散である。オプション付きで呼ぶなら `loo_predict_with` を使う。

```rust
use gprx::kernel::{KernelSpec, RbfKernel};
use gprx::{GaussianLikelihood, Gpr, PredictOptions, PredictiveCovariance, VarianceKind};

fn main() -> Result<(), gprx::GprError> {
    let mut fitted = Gpr::new(
        KernelSpec::from(RbfKernel::new(1.0)?),
        GaussianLikelihood::new(0.1)?,
    )
    .fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0])?;
    let cov: PredictiveCovariance = fitted.predict_covariance(&[0.25, 0.75], 2, 1)?;
    let cov_latent = fitted.predict_covariance_with(
        &[0.25, 0.75],
        2,
        1,
        PredictOptions {
            variance_kind: VarianceKind::Latent,
        },
    )?;
    let draws = fitted.sample(&[0.25, 0.75], 2, 1, 2, 1)?;
    let _draws_with = fitted.sample_with(&[0.25], 1, 1, PredictOptions::default(), 1, 2)?;
    let loo = fitted.loo_predict()?;
    let _loo_with = fitted.loo_predict_with(PredictOptions::default())?;
    let nll = fitted.neg_log_marginal_likelihood()?;
    let mut theta = vec![0.0; fitted.num_params()];
    fitted.get_params(&mut theta)?;
    let value = fitted.value_and_gradient_into(&theta, &mut theta.clone())?;
    let mut hess = vec![0.0; fitted.num_params() * fitted.num_params()];
    fitted.hessian_into(&theta, &mut hess)?;
    fitted.set_params(&theta)?;
    let _ = (
        cov.covariance[0],
        cov_latent.mean[0],
        draws[0],
        loo.mean[0],
        nll,
        value,
    );
    Ok(())
}
```

`neg_log_marginal_likelihood` は、今の `θ` での目的関数の値を返す。`num_params`、`get_params`、`set_params` が扱うのは一列に並べた log-`θ` のベクトルで、カーネルのパラメータのあとに尤度のパラメータが続く。この目的関数を評価するのが `value_and_gradient_into` と `hessian_into` である。`set_params` は `θ` を書き換え、因子分解もあわせて更新する。

学習済みモデルの中身は `n`、`d`、`kernel`、`likelihood`、`x`、`y`、`alpha` で読める。設定した方針を読むのは `distance_cache_policy`、`cholesky_buffer`、`math`、`jitter_policy` の 4 つである。`fit` の前の `Gpr` でも、`kernel`、`likelihood`、`num_params`、`get_params` と、方針を読むこの 4 つは使える。`set_params` が使えるようになるのは `fit` のあとである。`alpha` の返し方はモデルによって違う。`FittedGpr::alpha` は、直前の因子分解で得たスライスを返す。`OnlineGpr::alpha` が `Result` を返すのは、`insert` や `delete` のあと `α` が古いまま残り、その次に呼んだときに解き直すからである。

`into_trainer` は、今の `θ` を引き継いだ未学習の `Gpr` を返す。学習済みモデルの `with_optimizer` を呼んでも、変わるのはその後の `refit` の動きだけで、`OnlineGpr::with_optimizer` も同様である。`refit` の中身は最適化器の型で分かれる。`FittedGpr<O: Optimizer>` では、保存してある学習データの上で、今の `θ` から探索をやり直す。`FittedGpr<Fixed>` では `L` と `α` を組み直すだけで、探索はしない。どちらの場合も、変換は学習し直さない。

点を 1 つずつ足したり消したりしたいときは、`into_online` で `OnlineGpr` に変換する。`insert(x_new, y_new)` は 1 点を追加して `PointId` を返し、`delete(id)` はその点を取り除く。最後に残った点は削除できず、`GprError::InsufficientData`（`min` は 2）になる。`PointId` には公開コンストラクタがない。`into_online` が既存の点に `0 .. n-1` を割り当て、以降の挿入には増えていく番号が振られる。一度使った番号は再利用しない。いまモデルにある id の一覧は `point_ids` で得られ、モデルにない id を渡すと `InvalidPointId` になる。オンラインモデルの `into_trainer` は `Gpr` を返す。`OnlineGpr` の `refit` も学習済みモデルと同じ分かれ方で、`Optimizer` なら探索し、`Fixed` なら因子を組み直す。

```rust
use gprx::kernel::{KernelSpec, RbfKernel};
use gprx::{GaussianLikelihood, Gpr, OnlineGpr, PointId};

fn main() -> Result<(), gprx::GprError> {
    let fitted = Gpr::new(
        KernelSpec::from(RbfKernel::new(1.0)?),
        GaussianLikelihood::new(0.1)?,
    )
    .fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0])?;
    let mut online: OnlineGpr = fitted.into_online()?;
    let id: PointId = online.insert(&[0.5], 0.25)?;
    assert!(online.point_ids().contains(&id));
    let _ = online.alpha()?;
    online.delete(id)?;
    let trainer = online.into_trainer();
    let _ = trainer.kernel();
    Ok(())
}
```

`save(dir)` は、因子を含めずに `config.json` と `model.safetensors` を書き出す。`save_with_factor` を使うと因子と `α` も保存され、その因子は `FittedGpr` なら列優先の `L`、`OnlineGpr` なら詰めた形式の LDLT の因子になる。

`fit` の前の `Gpr` では、次のメソッドで設定を変えられる。

| メソッド | 効果 |
| --- | --- |
| `with_optimizer(solver)` | `Lbfgs` を置き換える。`Fixed` にすると `fit` がなくなり、代わりに `factor` が使える。 |
| `with_precision::<P>()` | `DoublePrecision`（既定）、`SinglePrecision`、`MixedPrecision` から選ぶ。 |
| `with_math(KernelExp::FastApprox)` | カーネルの `exp` を多項式で近似する。既定は `KernelExp::Accurate`。 |
| `with_input_transform(map)` | 既定は恒等写像。 |
| `with_target_transform(map)` | 既定は恒等写像。事前分布の平均は 0 なので、`y` の中心が 0 から離れているときは `StandardizeTarget::new()` を使う。 |
| `with_jitter_policy(policy)` | 既定は `JitterPolicy::fixed(0.0)`。 |
| `with_prefer_speed` | `DistanceCachePolicy::Cached` と `CholeskyBuffer::Retain`（どちらも既定）。両方をまとめて置き換える。 |
| `with_prefer_memory` | `DistanceCachePolicy::Uncached` と `CholeskyBuffer::Reuse`。両方をまとめて置き換える。 |

`with_prefer_speed` と `with_prefer_memory` は `Gpr` のメソッドで、距離のキャッシュとコレスキー分解のバッファをまとめて設定する。`DistanceCachePolicy` の既定は `Cached` で、`Uncached` にすると距離を毎回計算し直す。`DistanceCachePolicy` と `KernelExp` は non-exhaustive である。`CholeskyBuffer` がとる値は `Retain` と `Reuse` の 2 つである。

`Sgpr` と `Svgp` で使える設定は `with_optimizer`、`with_precision`、`with_math`、`with_jitter_policy`、`with_input_transform`、`with_target_transform` で、`Sgpr` にはさらに `with_inducing` がある。この 2 つのモデルでは、`K_mm` のジッタが `JitterPolicy::adaptive(1e-8, 10.0, 5, 1e-3)` から始まる。`fit` の前でも、`kernel`、`likelihood`、`math`、`jitter_policy`、`num_params`、`get_params` で値を読める。`set_params` で `θ` を更新することもできる。

```rust
use gprx::kernel::{KernelSpec, RbfKernel};
use gprx::transform::StandardizeTarget;
use gprx::{
    CholeskyBuffer, DistanceCachePolicy, GaussianLikelihood, Gpr, JitterPolicy, KernelExp,
    Prediction, SinglePrecision,
};

fn main() -> Result<(), gprx::GprError> {
    let gpr = Gpr::new(
        KernelSpec::from(RbfKernel::new(1.0)?),
        GaussianLikelihood::new(0.1)?,
    )
    .with_precision::<SinglePrecision>()
    .with_math(KernelExp::FastApprox)
    .with_input_transform(gprx::transform::IdentityInput)
    .with_target_transform(StandardizeTarget::new())
    .with_jitter_policy(JitterPolicy::fixed(0.0)?)
    .with_prefer_memory()
    .with_prefer_speed();
    assert_eq!(gpr.distance_cache_policy(), DistanceCachePolicy::Cached);
    assert_eq!(gpr.cholesky_buffer(), CholeskyBuffer::Retain);
    let fitted = gpr.fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0]).map_err(|(_, e)| e)?;
    // A single-precision model predicts in `f32`.
    let pred: Prediction<f32> = fitted.predict(&[0.5], 1, 1)?;
    let _ = pred.mean[0];
    Ok(())
}
```

### 誘導点: `Sgpr`、`FittedSgpr`、`OnlineSgpr`

`KernelSpec` を渡した `Sgpr::new` の型は `Sgpr<Lbfgs, FixedInducing, DoublePrecision, KernelSpec>` である。既定では誘導点を渡した位置から動かさない。`with_inducing(FreeInducing)` を指定すると、`Z` も探索の対象になる。

```rust
use gprx::kernel::{KernelSpec, RbfKernel};
use gprx::{
    Fixed, FittedSgpr, FixedInducing, FreeInducing, GaussianLikelihood, InducingId, OnlineSgpr,
    Sgpr,
};

fn main() -> Result<(), gprx::GprError> {
    let x = &[0.0, 1.0, 2.0, 3.0];
    let y = &[0.0, 1.0, 0.5, 0.25];
    let z = &[0.5, 2.5];
    let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    let likelihood = GaussianLikelihood::new(0.1)?;

    let fitted: FittedSgpr = Sgpr::new(kernel.clone(), likelihood.clone())
        .with_inducing(FixedInducing)
        .fit(x, 4, 1, y, z, 2)
        .map_err(|(_, e)| e)?;
    let _moved = Sgpr::new(kernel.clone(), likelihood.clone())
        .with_inducing(FreeInducing)
        .fit(x, 4, 1, y, z, 2)
        .map_err(|(_, e)| e)?;
    let frozen = Sgpr::new(kernel, likelihood)
        .with_optimizer(Fixed)
        .factor(x, 4, 1, y, z, 2)
        .map_err(|(_, e)| e)?;
    assert_eq!(fitted.neg_log_marginal_likelihood()?.is_finite(), true);
    let _ = frozen.m();

    // Online: add a point and an inducing point, then return to a fitted model.
    let mut online: OnlineSgpr = fitted.into_online();
    let point = online.insert(&[1.5], 0.75)?;
    let inducing: InducingId = online.insert_inducing(&[1.5])?;
    online.delete_inducing(inducing)?;
    online.delete(point)?;
    let back = online.into_fitted();
    assert_eq!(back.m(), 2);
    Ok(())
}
```

`fit` と `factor` の引数は `(x, n_rows, n_cols, y, z, n_inducing)` で、`z` には `n_inducing` 個の点を `x` と同じ `n_cols` 個の特徴で列優先に並べて渡す。`factor` は `θ` も `Z` も動かさない。誘導点を自由にすると、パラメータのベクトルはカーネルの `θ`、尤度の `θ` の後ろに列優先の `Z` が続く形になる。`Z` の各座標が動ける区間は、入力の変換をかけたあとの学習データから決まる。データの範囲を両側にその幅の 10%（ただし `0.1` 以上）広げ、下端が `0` より大きければ `0` まで下げたものが、その区間である。Matérn の `ν = 1/2` は座標についての微分を持たないので、`Z` を自由にすると `GprError::CoordGradientUnsupported` になる。

`FittedSgpr` には、`FittedGpr` と同じ予測、共分散、サンプル、leave-one-out のメソッドがそろっている。目的関数のメソッド（`neg_log_marginal_likelihood`、`num_params`、`get_params`、`set_params`、`value_and_gradient_into`、`hessian_into`）も共通である。モデルの中身は `n`、`m`、`d`、`kernel`、`likelihood`、`math`、`jitter_policy`、`x`、`y`、`z` で読める。`z` が返すのは元の座標での誘導点である。

`into_online` が返す `OnlineSgpr` でも、これらの読み取りと目的関数のメソッドはそのまま使え、さらに `point_ids` と `inducing_ids` が加わる。学習点を `insert` / `delete` するときは `PointId` で、誘導点を `insert_inducing` / `delete_inducing` するときは `InducingId` で指定する。`InducingId` にも公開コンストラクタはなく、番号は再利用されない。モデルにない id を渡すと `InvalidInducingId` になる。`OnlineSgpr<O: Optimizer>` の `refit` は探索をやり直すが、`Z` は固定したままである。`OnlineSgpr<Fixed>` には `refit` がない。学習済みのモデルに戻すには `into_fitted` を呼び、`FittedSgpr<_, FixedInducing, _>` を受け取る。`save` でディレクトリへ保存できるが、疎なモデルには `save_with_factor` がない。

### ミニバッチ: `Svgp`、`FittedSvgp`

`Svgp::new` の型は `Svgp<Fixed>` である。`factor` は、今の `θ` で白色化した変分事後分布 `q` を作る。探索はしない。学習させるには `with_optimizer(Adam::new()).fit(...)` とし、ミニバッチの Adam で `θ` と `q` を動かす。`Adam` は `Optimizer` を実装していないので、`Gpr` と `Sgpr` には渡せない。

```rust
use gprx::kernel::{KernelSpec, RbfKernel};
use gprx::{Adam, FittedSvgp, GaussianLikelihood, Svgp};

fn main() -> Result<(), gprx::GprError> {
    let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    let likelihood = GaussianLikelihood::new(0.1)?;
    let x = &[0.0, 1.0, 2.0, 3.0];
    let y = &[0.0, 1.0, 0.5, 0.25];
    let z = &[0.5, 2.5];
    let factored: FittedSvgp = Svgp::new(kernel.clone(), likelihood.clone())
        .factor(x, 4, 1, y, z, 2)
        .map_err(|(_, e)| e)?;
    let trained = Svgp::new(kernel, likelihood)
        .with_optimizer(Adam::new())
        .fit(x, 4, 1, y, z, 2)
        .map_err(|(_, e)| e)?;
    let _ = (factored.neg_elbo()?, trained.n());
    Ok(())
}
```

`FittedSvgp::neg_elbo` は、証拠下限（ELBO）の符号を反転した値を返す。`num_params`、`get_params`、`set_params`、`value_and_gradient_into` は、全データでの目的関数のパラメータと値を扱う。`hessian_into` はない。Adam のループでは、データ項を `n / batch` 倍し、KL 項には倍率をかけない。

予測、共分散、サンプルのメソッドはほかのモデルと同じである。予測には `predict`、`predict_with`、`predict_into`、`predict_with_into` を使う。共分散とサンプルには `predict_covariance`、`predict_covariance_with`、`sample`、`sample_with` を使う。モデルの中身は `n`、`m`、`d`、`kernel`、`likelihood`、`math`、`jitter_policy`、`x`、`y`、`z` で読める。leave-one-out とオンライン型、`refit` は用意していない。`save` でディレクトリへ保存できる。

`Adam::new` の既定値は、学習率 `1e-3`、`β1 = 0.9`、`β2 = 0.999`、`ε = 1e-8`、バッチの大きさ 32、エポック数 100、シード 0 である。変えるには `with_learning_rate`、`with_beta1`、`with_beta2`、`with_epsilon`、`with_batch_size`（`NonZeroUsize`）、`with_epochs`（`NonZeroU64`）、`with_seed` を使う。このうち学習率、2 つのベータ、イプシロンのセッターは `Result` を返す。

```rust
use std::num::{NonZeroU64, NonZeroUsize};

use gprx::kernel::{KernelSpec, RbfKernel};
use gprx::{Adam, GaussianLikelihood, Svgp};

fn main() -> Result<(), gprx::GprError> {
    let adam = Adam::new()
        .with_learning_rate(1e-3)?
        .with_beta1(0.9)?
        .with_beta2(0.999)?
        .with_epsilon(1e-8)?
        .with_batch_size(NonZeroUsize::MIN)
        .with_epochs(NonZeroU64::MIN)
        .with_seed(0);
    let _ = Svgp::new(
        KernelSpec::from(RbfKernel::new(1.0)?),
        GaussianLikelihood::new(0.1)?,
    )
    .with_optimizer(adam);
    Ok(())
}
```

### カーネル

カーネルは `KernelSpec` の木として組み立てる。カーネルの葉から作るなら `KernelSpec::from(leaf)`、自作の項から作るなら `KernelSpec::custom(term)` を使う。和は `+`、積は `*` で書き、`*` のほうが `+` より先に結合するので、`c * k + k2` は定数倍したカーネルと別のカーネルとの和になる。`num_params`、`get_params`、`set_params` が扱う log-`θ` は、カーネルの葉を深さ優先でたどった順に一列に並ぶ。各パラメータがどのカーネルの葉に属するかは、`parameter_bindings` が `Vec<ParameterBinding>` で返す。`ParameterBinding` のフィールドは `index`、`leaf_id`、`local_index` である。

評価用の形にするには `compile` を呼び、`CompiledKernel<f64>` を得る。保存するスカラーの型を選びたいなら `compile_as::<T>()` を使う。`T` の境界は `KernelScalar` で、実装しているのは `f32` と `f64` だけである。このトレイトは数値の演算をメソッドとして持つ。`f64` との変換は `from_f64` と `to_f64` である。関数は `exp`、`ln`、`sqrt`、`abs`、`powf`、`sin`、`cos` で、判定と比較は `is_finite`、`max`、`min` である。`CompiledKernel` でも `num_params`、`get_params`、`set_params` が使える。

`KernelSpec` と `CompiledKernel` は non-exhaustive な列挙で、`match` の分岐に使うバリアントは `Rbf`、`RbfArd`、`Matern`、`MaternArd`、`Periodic`、`RationalQuadratic`、`RationalQuadraticArd`、`Constant`、`Linear`、`White`、`Custom`、`Sum`、`Product` である。`Sum` と `Product` の中身は、`KernelSpec` では `Box` の組、`CompiledKernel` では平坦なベクタになっている。

```rust
use gprx::kernel::{ConstantKernel, KernelSpec, RbfKernel};

fn main() -> Result<(), gprx::GprError> {
    let scaled = KernelSpec::from(ConstantKernel::new(1.5)?)
        * KernelSpec::from(RbfKernel::new(1.0)?);
    let kernel = scaled + KernelSpec::from(RbfKernel::new(2.0)?);
    assert_eq!(kernel.num_params(), 3);
    Ok(())
}
```

組み込みのカーネルの葉は、正のパラメータを `log` の値で持つ。`new` には利用者の単位（`ℓ`、分散、周期、`α`）で値を渡し、`from_log_*` には最適化の座標で値を渡す。区間は `bounds` で読み、`with_bounds` で設定する。区間の型は開区間の `Interval` で、既定は `(1e-5, 1e5)`（`Interval::DEFAULT_POSITIVE`）である。今の値が新しい区間から外れると、`with_bounds` は `IntervalError` を返す。

| カーネルの葉 | 構築 | パラメータ |
| --- | --- | --- |
| `RbfKernel` | `new(ℓ)` | 等方の長さスケール |
| `RbfArdKernel` | `new(&[ℓ_d])` | 特徴ごとの長さスケール（`ArdLengthscales`） |
| `MaternKernel` | `new(ℓ, MaternNu)` | `MaternNu::Half`、`ThreeHalves`、`FiveHalves`（`value` は 0.5、1.5、2.5）。`ν` は最適化しない |
| `MaternArdKernel` | `new(&[ℓ_d], nu)` | ARD の長さスケールと、固定した `ν` |
| `PeriodicKernel` | `new(ℓ, period)` | 長さスケールと周期 |
| `RationalQuadraticKernel` | `new(ℓ, alpha)` | 長さスケールと `α` |
| `RationalQuadraticArdKernel` | `new(&[ℓ_d], alpha)` | ARD の長さスケール、その後ろに `α` |
| `ConstantKernel` | `new(c)` | 信号分散。RBF と掛けると `c * k` になる |
| `LinearKernel` | `new(variance)` | `σ² xᵀ x'` |
| `WhiteKernel` | `new(variance)` | 対角ノイズ |

観測ノイズは `GaussianLikelihood` が受け持ち、`WhiteKernel` はカーネルに追加する項である。尤度のノイズとホワイト項の両方を大きくすると、同じノイズを二重に数えることになる。

等方のカーネルの葉は、長さスケールを `lengthscale` と `log_lengthscale` で読める。Matérn ではさらに `nu` も読める。ほかのカーネルの葉で取得できる値は、周期カーネルが `period` と `log_period`、有理二次が `alpha` と `log_alpha`、線形とホワイトが `variance` と `log_variance`、定数が `constant` と `log_constant` である。

最適化の座標から作るコンストラクタには `from_log_lengthscale`、`from_log_lengthscales`、`from_log`、`from_log_variance`、`from_log_constant` がある。このうち `from_log` は 2 つの値を取り、周期カーネルには log 長さスケールと log 周期を、有理二次には log 長さスケールと log `α` を渡す。

区間の持ち方もカーネルの葉ごとに違う。等方の RBF と Matérn、定数、線形、ホワイトのカーネルの葉は、`bounds` を 1 つだけ持つ。ARD のカーネルの葉では、長さスケールの区間を `ArdLengthscales` が管理する。周期カーネルの区間は `lengthscale_bounds` と `period_bounds` に分かれ、`with_bounds` には両方の区間を渡す。有理二次も同様に `lengthscale_bounds` と `alpha_bounds` に分かれ、`with_bounds` で両方を設定する。

ARD のカーネルの葉では、`lengthscale(dim)` が 1 つの `ℓ_d` を返し、`log_lengthscales` が保存されたベクトルを返す。`lengthscales()` は `ArdLengthscales` を返す。`ArdLengthscales` は `new` か `from_log_lengthscales` で作り、区間は `with_bounds` で設定する。値は `lengthscale(dim)` と `log_lengthscales` で読める。`num_params`、`get_params`、`set_params` も使える。ARD の `with_bounds` が受け取る区間は 1 つで、すべての `ℓ_d` に同じ区間を使う。有理二次 ARD の `with_bounds` だけは、長さスケールの区間と `α` の区間の 2 つを取る。

カーネルの葉と `CompiledKernel` は、`faer` の行列（`MatRef` / `MatMut`）の上で `apply`、`apply_cross`、`grad`、`hess` を使って評価する。これらを呼ぶには、依存に `faer = "0.24"` を加える。`LinearKernel` 以外のカーネルの葉は、これらのメソッドで二乗距離を読み、対角を `fill_diag` で書き込む。`LinearKernel` だけは、この 4 つのメソッドで座標を読み、対角を `fill_diag_points` で書き込む。また、`ConstantKernel` と `WhiteKernel` は `apply_points`、`apply_cross_points`、`grad_points`、`hess_points` も備え、`RbfKernel`、`RbfArdKernel`、等方の `MaternKernel`、`WhiteKernel` は `grad_wrt_coord_dim` も備える。

行列のどの要素を書き込むかは `Triangle` で選ぶ。コレスキー分解に使う `Triangle::Lower` のほかに、`Upper` と `Full` がある。カーネルの葉を直接評価すると、`exp` は `Accurate` で計算される。`CompiledKernel::apply::<M>` と、そのほかの評価メソッド（`fill_diag` と `fill_diag_points` は除く）では、`KernelMath` の型 `M` で `exp` の計算方法を選び、出力と同じ形の作業用行列も渡す。`Accurate` は libm / SIMD の `exp` を使い、`FastApprox` は 7 次の多項式で近似する。`f64` での `FastApprox` と `f64::exp` の相対差は、`2^{-23}` 以内に収まる。ハイパーパラメータの `exp(θ)` の計算には、この選択は使われない。

二乗距離を読む自作のカーネルの葉は、トレイト `KernelTerm<T>` を実装して作る。実装するメソッドは `num_params`、`get_params`、`set_params`、`bounds_into`、`apply`、`apply_cross`、`fill_diag`、`grad`、`hess`、`hess_points` である。疎なモデルで使うなら、そのモデルが必要とする微分のメソッドも要る。`grad_cross` と `hess_cross` を実装し、`FreeInducing` で使うにはさらに `grad_wrt_sq_dist`、`hess_wrt_sq_dist`、`grad_wrt_sq_dist_theta` を実装する。保存したい場合は `persist_id` と `persist_state` も実装する。この 2 つは `serde_json::Value` を返すので、依存に `serde_json = "1"` を加える。

実装は、すべての `T: KernelScalar` に対して一度書けばよい。モデルで使うには、そのカーネルの葉が `f64` と `f32` の両方で実装され、`Clone`、`Debug`、`Send`、`Sync`、`'static` を満たしている必要がある。`CustomKernel::new(term)` は、その実装を `Box` に包む。カーネルの木に入れるには、`KernelSpec::from(custom)` か `KernelSpec::custom(term)` を使う。疎なモデルが必要とする微分を省いた自作のカーネルの葉は、`CoordGradientUnsupported` を返す。実装の例は、後の保存と読み込みの例にある。

```rust
use gprx::kernel::{
    ArdLengthscales, CompiledKernel, ConstantKernel, KernelSpec, LinearKernel, MaternArdKernel,
    MaternKernel, MaternNu, PeriodicKernel, RationalQuadraticArdKernel, RationalQuadraticKernel,
    RbfArdKernel, RbfKernel, Triangle, WhiteKernel,
};
use gprx::Interval;

fn main() -> Result<(), gprx::GprError> {
    let rbf = RbfKernel::from_log_lengthscale(0.0)?;
    let _ = (rbf.lengthscale(), rbf.log_lengthscale(), rbf.bounds());
    let rbf = rbf.with_bounds(Interval::new(1e-3, 1e2)?)?;
    let ard = RbfArdKernel::new(&[0.5, 2.0])?;
    let _ = ard.lengthscale(0)?;
    let scales: &ArdLengthscales = ard.lengthscales();
    let _ = scales.log_lengthscales();
    let matern = MaternKernel::new(1.0, MaternNu::FiveHalves)?;
    let _ = matern.nu().value();
    let _ = MaternArdKernel::new(&[1.0, 1.0], MaternNu::Half)?;
    let periodic = PeriodicKernel::new(1.0, 0.5)?;
    let _ = (periodic.period(), periodic.log_period());
    let rq = RationalQuadraticKernel::new(1.0, 1.5)?;
    let _ = (rq.alpha(), rq.log_alpha());
    let _ = RationalQuadraticArdKernel::new(&[1.0], 1.5)?;
    let constant = ConstantKernel::new(1.5)?;
    let _ = (constant.constant(), constant.log_constant());
    let linear = LinearKernel::new(0.2)?;
    let _ = (linear.variance(), linear.log_variance());
    let white = WhiteKernel::new(0.01)?;
    let _ = white.log_variance();
    let spec = KernelSpec::from(rbf) + KernelSpec::from(white);
    let bindings = spec.parameter_bindings();
    let _ = (bindings[0].index, bindings[0].leaf_id, bindings[0].local_index);
    let compiled: CompiledKernel = spec.compile();
    let compiled_f32 = spec.compile_as::<f32>();
    let mut theta = vec![0.0; compiled.num_params()];
    compiled.get_params(&mut theta)?;
    let _ = (compiled_f32.num_params(), Triangle::Lower);
    Ok(())
}
```

次の例では、カーネルの葉とコンパイルしたカーネルの木を二乗距離の上で評価する。パラメータから作るコンストラクタと、スカラーの型について総称な関数も含めている。

```rust
use faer::Mat;
use gprx::kernel::{
    ConstantKernel, KernelScalar, KernelSpec, LinearKernel, ParameterBinding, PeriodicKernel,
    RationalQuadraticKernel, RbfKernel, Triangle, WhiteKernel,
};
use gprx::{Accurate, FastApprox, KernelMath};

/// `exp(−d² / 2)` at `f32` or `f64`.
fn unit_rbf<T: KernelScalar>(d2: f64) -> f64 {
    (T::from_f64(-0.5) * T::from_f64(d2)).exp().to_f64()
}

/// The `(0, 1)` entry of `k` for the `KernelMath` mode `M`.
fn off_diagonal<M: KernelMath>(spec: &KernelSpec, d2: &Mat<f64>) -> Result<f64, gprx::GprError> {
    let mut k = Mat::<f64>::zeros(2, 2);
    let mut scratch = Mat::<f64>::zeros(2, 2);
    spec.compile()
        .apply::<M>(d2.as_ref(), k.as_mut(), Triangle::Full, scratch.as_mut())?;
    Ok(k[(0, 1)])
}

fn main() -> Result<(), gprx::GprError> {
    // Two points at 0 and 1: d² is [[0, 1], [1, 0]].
    let d2 = Mat::from_fn(2, 2, |i, j| (i as f64 - j as f64).powi(2));
    let rbf = RbfKernel::new(1.0)?;
    let mut k = Mat::<f64>::zeros(2, 2);
    rbf.apply(d2.as_ref(), k.as_mut(), Triangle::Lower)?;
    assert!((k[(1, 0)] - unit_rbf::<f64>(1.0)).abs() < 1e-12);
    assert!((unit_rbf::<f32>(1.0) - unit_rbf::<f64>(1.0)).abs() < 1e-6);
    rbf.apply_cross(d2.as_ref(), k.as_mut())?;
    let mut diag = [0.0; 2];
    rbf.fill_diag(&mut diag);
    rbf.grad(d2.as_ref(), k.as_mut(), 0, Triangle::Full)?;
    rbf.hess(d2.as_ref(), k.as_mut(), 0, 0, Triangle::Full)?;
    let x = Mat::from_fn(2, 1, |i, _| i as f64);
    LinearKernel::from_log_variance(0.0)?.fill_diag_points(x.as_ref(), &mut diag)?;

    let _ = (WhiteKernel::from_log_variance(-2.0)?, ConstantKernel::from_log_constant(0.0)?);
    let periodic = PeriodicKernel::from_log(0.0, 0.5_f64.ln())?;
    let _ = (periodic.lengthscale_bounds(), periodic.period_bounds());
    let rq = RationalQuadraticKernel::from_log(0.0, 0.0)?;
    let _ = (rq.lengthscale_bounds(), rq.alpha_bounds());

    let spec = KernelSpec::from(rbf);
    let bindings: Vec<ParameterBinding> = spec.parameter_bindings();
    assert_eq!(bindings.len(), 1);
    let accurate = off_diagonal::<Accurate>(&spec, &d2)?;
    let fast = off_diagonal::<FastApprox>(&spec, &d2)?;
    assert!((accurate - fast).abs() < 1e-6);
    Ok(())
}
```

### 与えられた二乗距離（`ScalarDistance`、`ArdDistance`）

カーネルの葉には、座標の代わりに、呼び出し側が与えた二乗距離を読ませることもできる。測地距離やグラフ上の距離、別のプログラムで求めた距離を、そのままカーネルに使える。

距離の受け取り先を、ここではスロットと呼ぶ。`ScalarDistance::new()` は、点の対ごとに `d²` を 1 つ受け取るスロットを作る。そのスロットの `kernel(leaf)` に渡せるのは、`RbfKernel`、`MaternKernel`、`PeriodicKernel`、`RationalQuadraticKernel` と、自作のカーネルの葉である。自作のカーネルの葉は `KernelTerm` を `f64` と `f32` で実装し、`Clone + Debug + Send + Sync + 'static` を満たしている必要がある。この条件を表す境界が `ScalarDistanceLeaf` である。

ARD のカーネルの葉には `ArdDistance::from_leaf(leaf)` を使う。受け付けるのは `RbfArdKernel`、`MaternArdKernel`、`RationalQuadraticArdKernel` で（境界は `ArdDistanceLeaf`）、戻り値はスロットと、そのスロットに載せたカーネルの葉である。このスロットは次元ごとの `(Δ_k)²` を 1 つずつ、計 `d` 個のブロックとして受け取る。ここでの `d` は、そのカーネルの葉が持つ長さスケールの数である。同じスロットに 2 つ目以降のカーネルの葉を載せるには `kernel(leaf)` を使い、長さスケールの数が `d` と違えば `DimensionMismatch` になる。1 つのスロットに載ったカーネルの葉は、どれも同じ供給を読む。`ConstantKernel`、`WhiteKernel`、`LinearKernel` はスロットに載せず、これまでどおり `KernelSpec` のカーネルの葉として使う。

こうして組んだカーネルは、`KernelSpec` とは別の型 `DistanceKernel<C>` になる。型パラメータの `C` はマーカー型で、座標を読まない `DistanceOnly` か、座標を読むカーネルの葉も含む `WithPoints` のどちらかである。`DistanceKernel` どうしを `+` や `*` で組み合わせると、`JoinPoints` が両者のマーカー型から結果のマーカー型を決める。`ConstantKernel` か `WhiteKernel` と組み合わせた場合は、どちらの順に書いても `C` は変わらない。右側に `KernelSpec` を置いた `DistanceKernel` は `WithPoints` になる。座標の項を左側に置きたいときは、`DistanceKernel::from(spec)`（スロットを持たない `DistanceKernel<WithPoints>`）と書く。`KernelSpec + DistanceKernel` は実装していないため、`KernelSpec + leaf.into()` と書くと、座標どうしの和として型が推論されてしまう。

`num_params`、`get_params`、`set_params`、`parameter_bindings` の振る舞いは `KernelSpec` と同じである。`slots()` は、深さ優先でたどって最初に現れた順に `DistanceSlot` を返す。型の側では、`KernelSpec<S>` と `CompiledKernel<T, S>` の最後の型パラメータ `S` が、封印されたトレイト `Supply` によって供給の種類を表す。`S` の既定は `NoSupply` で、座標の木を表す。`NoSupply` は値を持たない。型パラメータを省いた `KernelSpec` と `CompiledKernel<T>` では `S` が `NoSupply` になるので、この 2 つは座標の木の型を指す。もう一方の種類の供給を持つのは `DistanceKernel` だけである。

`Gpr::new` は任意の `ModelKernel`、つまり `KernelSpec` と `DistanceKernel<C>` のどちらでも受け付ける。`with_input_transform` を呼べるのは、`PointKernel` を満たすモデル（`KernelSpec` と `DistanceKernel<WithPoints>`）だけである。各スロットには、呼び出しのたびに `DistanceSource` を 1 つ渡す。どの方法で供給を作るかによって、コピーが起きるかどうかが次のように変わる。

| 供給 | コピー |
| --- | --- |
| `from_vec(d2)` / `from_vecs(blocks)` | 呼び出しへムーブする。`f64` のモデルは受け取った表をそのまま保存領域に使うので、fit では何もコピーしない。そのため ARD のスロットは、詰めた `d · n(n+1)/2` 個ではなく `d · n²` 個の値を持つ（`tidy` で補正した ARD の表は詰めて持つ）。`f32` のモデルは自分のバッファへ型変換して持ち、ARD なら詰める。`MixedPrecision` のモデルは `f64` の表を持ったうえで、その型変換も持つ |
| `from_slice(d2)` / `from_slices(blocks)` | 呼び出しの時点でコピーし、以降は `from_vec` / `from_vecs` と同じに扱う。`f64` のモデルはそのコピーを持つ（ARD のスロットでは `d · n²` 個）。このコピーなしで fit したいなら `borrow` を使う |
| `borrow(d2)` / `borrow(blocks)` | `f64` のモデルの `predict` はその場で読み、`f32` のモデルは 1 回だけ型変換する。`fit` ではコピーする（ARD の正方行列は、詰めた三角形へ直接コピーする） |
| `fill(&filler)` | `DistanceFill::fill_column(col, rows, out)` が、`rows` の各行 `i` について `d²(i, col)` を書き込む。ARD の fill は `d` 本の並びを順に書き込む。正方行列では、各列の `col..n` 行だけを求める |

表は列優先で、要素 `(i, j)` を `dist[i + j * n_rows]` に置く。値はすべて有限かつ非負でなければならない。さらに学習の正方行列（とクエリの正方行列）は、対角がちょうど 0 で、厳密に対称である必要がある。この条件を破る値があると、最初に見つかった位置で `InvalidDistance { slot, dim, pair, reason }` を返す。`slot` は `slots()` の中での位置、`dim` は ARD の次元、`pair` は渡したブロックの中の `(row, col)` である。

`‖a‖² + ‖b‖² − 2a·b` で計算した表のように、丸め誤差でわずかに条件を外れる表もある。こうした表は、供給の側で補正を指定したときに限って受け付ける。その指定が `source.tidy(rel_tol)` である。表の最大値の `rel_tol` 倍以内に収まる負の値と対角の値は `0.0` にし、同じ範囲で食い違う鏡像の組は平均にそろえる。この範囲を超えるずれは拒否する。借用した表は、補正で値が変わる場合でも書き換えず、コピーを作ってから補正する。

供給とスロットが対応しないときは `DistanceSlot { kind, slot }` になる。供給のないスロットは `SlotErrorKind::Missing`、供給が 2 つあるスロットは `Duplicate`、カーネルにないスロットへの供給は `NotRead` である。読み込んだモデルは新しいスロットを読むので、保存前のスロットへの供給は `NotRead` になる。

学習の fill は、`DistanceCachePolicy` の設定にかかわらず、1 回の学習で 1 回しか呼ばれない。fill が書いた二乗距離は、モデルが保持しているからである。`MixedPrecision` のモデルは学習の `d²` を `f32` の写しと並べて `f64` でも持つので、`f64` での精緻化は呼び出し側が渡した値そのものを読む。`predict_into` は、モデルが保持するバッファに供給を結び付けて計算する。そのため、同じ形の呼び出しを一度実行したあとは、メモリの確保が起きない。

| モデル | `fit` / `factor` | `predict` 系 | 共分散と `sample` |
| --- | --- | --- | --- |
| `Gpr`、`DistanceOnly` | `(sources, n, y)` | `(sources, q)` | `(cross, square, q)` |
| `Gpr`、`WithPoints` | `(sources, n, x, n_cols, y)` | `(sources, xs, q, n_cols)` | `(cross, square, xs, q, n_cols)` |
| `Sgpr` / `Svgp`、`DistanceOnly` | `(sources, n, y, inducing)` | `(sources, q)` | `(cross, square, q)` |
| `Sgpr` / `Svgp`、`WithPoints` | `(sources, n, x, n_cols, y, inducing)` | `(sources, xs, q, n_cols)` | `(cross, square, xs, q, n_cols)` |

`cross` は学習点からクエリへの `n × q` のブロック、`square` はクエリどうしの `q × q` のブロックである。距離のモデルでは、`to_kernel()` で `DistanceKernel` のコピーを、`slots()` でそのスロットを取り出せる。`d` と `x` を持つのは `WithPoints` のモデルだけである。

`into_online` を使えば、距離のモデルもオンラインのモデルに変換できる。その `insert` には、スロットごとに、今ある点から新しい点への二乗距離を `n × 1` の列（`point_ids` の順）で渡す。ARD のスロットではこの列を `d` 本渡す。どの列も、上の表のどの供給で作ってもよい。呼び方は、`DistanceOnly` なら `insert(sources, y_new)`、`WithPoints` なら `insert(sources, x_new, y_new)` である。渡した列は学習の正方行列と同じように検査され（`tidy` を指定していれば補正され）、モデルに保持される。`delete(id)` は、保持している二乗距離からその点をその場で取り除く。オンラインのモデルは、学習済みのモデルと同じ引数で予測する。どちらのモデルも、保持する領域を一度広げたあとはメモリを確保しない（領域を広げるとき、スカラーのスロットの正方行列は 4 分の 1 ずつ、ARD のスロットは倍々に大きくなる）。例外は、ワーカーが 2 つ以上ある Rayon のプールで大きな削除をする場合である。このときは削除の処理が、因子の更新と並んで動くジョブを 1 つプールに積む。

距離を受け取る疎なモデルでは、誘導点を学習点の添字で指定する。誘導点にする学習点の添字を `inducing` に重複なく並べ、`fit` / `factor` の各供給には、学習点からそれらの誘導点への `n × m` のブロックを渡す（列 `a` が学習点 `inducing[a]` に対応する）。`K_mm` は、このブロックのうち誘導点に当たる行を読む。モデルはブロックを渡された形のまま保持し、ムーブで渡した表はコピーせず、借用した表は 1 回だけコピーする。

予測で渡す `cross` は誘導点からクエリへの `m × q` のブロックで、`square` は先ほどと同じである。誘導点の並びは `inducing()` の順になる。`OnlineSgpr` では `inducing_points()` が返す `PointId` の順で、削除があってもこの順はずれない。`FreeInducing` が誘導点を動かすには座標が要るため、距離のモデルの誘導点は動かない。

`into_online` は、ブロックを保持したままオンラインのモデルに変換する。`insert` には、スロットごとに誘導点から新しい点への `m × 1` の二乗距離を渡す。`insert_inducing` では誘導点にする学習点を `PointId` で指定し、学習点からその点への `n × 1` の二乗距離を渡す。渡した値は、保持している組と照合して検査する。誘導点になっている点は、`delete_inducing` で外すまで `delete` できない。

計算量は座標のモデルと少し異なる。座標のモデルは距離の組を `n · d` 個の座標から計算するが、距離のモデルの因子分解は与えられた `d · n · m` 個の値を読む。そのため ARD のスロットでは、因子分解の計算量が座標のモデルより大きくなる。一方、勾配と予測の計算量は座標のモデルを上回らない（[設計 §5.6](https://github.com/YUKIKEDA/gprx/blob/main/docs/design.ja.md#56-与えられた二乗距離-要件470)）。

```rust
use gprx::kernel::{
    ArdDistance, ArdDistanceLeaf, ConstantKernel, DistanceFill, DistanceKernel, DistanceOnly,
    DistanceSlot, DistanceSource, JoinPoints, KernelSpec, ModelKernel, NoSupply, PointKernel,
    PointUse, RbfArdKernel, RbfKernel, ScalarDistance, ScalarDistanceLeaf, Supply, WithPoints,
};
use gprx::{Fixed, GaussianLikelihood, Gpr, Sgpr, Svgp};

/// Squared distances of points 0, 1, 2, … on a line.
struct Line;

impl DistanceFill for Line {
    fn fill_column(&self, col: usize, rows: std::ops::Range<usize>, out: &mut [f64]) {
        for (slot, i) in out.iter_mut().zip(rows) {
            *slot = (i as f64 - col as f64).powi(2);
        }
    }
}

fn scalar_leaf(leaf: impl ScalarDistanceLeaf, slot: ScalarDistance) -> DistanceKernel {
    slot.kernel(leaf)
}

fn takes<K: ModelKernel>(_: &K) {}
fn supply_of<S: Supply>(_: &KernelSpec<S>) {}
fn takes_points<K: PointKernel>(_: &K) {}
fn joined<A: JoinPoints<B>, B: PointUse>(_: DistanceKernel<A>, _: DistanceKernel<B>) {}

fn main() -> Result<(), gprx::GprError> {
    // A coordinate tree is `KernelSpec<NoSupply>`, the default.
    let coords: KernelSpec<NoSupply> = KernelSpec::from(RbfKernel::new(1.0)?);
    supply_of(&coords);
    let image = ScalarDistance::new();
    let other = ScalarDistance::new();
    let kernel = scalar_leaf(RbfKernel::new(1.0)?, image) * ConstantKernel::new(0.8)?;
    assert_eq!(kernel.slots(), vec![DistanceSlot::Scalar(image)]);
    let mut theta = vec![0.0; kernel.num_params()];
    kernel.get_params(&mut theta)?;
    let _ = kernel.parameter_bindings();
    takes(&kernel);
    joined(image.kernel(RbfKernel::new(1.0)?), other.kernel(RbfKernel::new(2.0)?));

    // Four points 0, 1, 2, 3: d²[i + j·4] = (i − j)².
    let d2 = vec![0.0, 1.0, 4.0, 9.0, 1.0, 0.0, 1.0, 4.0, 4.0, 1.0, 0.0, 1.0, 9.0, 4.0, 1.0, 0.0];
    let y = [0.0, 1.0, 0.5, 0.25];
    let fitted = Gpr::new(kernel.clone(), GaussianLikelihood::new(0.1)?)
        .with_optimizer(Fixed)
        .factor([image.from_vec(d2.clone())], 4, &y)?;
    // Two queries at 0.5 and 1.5.
    let cross = [0.25, 0.25, 2.25, 6.25, 2.25, 0.25, 0.25, 2.25];
    let square = [0.0, 1.0, 1.0, 0.0];
    let pred = fitted.predict([image.borrow(&cross)], 2)?;
    let cov = fitted.predict_covariance([image.borrow(&cross)], [image.borrow(&square)], 2)?;
    assert_eq!((pred.mean.len(), cov.covariance.len()), (2, 4));
    let by_fill = Gpr::new(kernel.clone(), GaussianLikelihood::new(0.1)?)
        .with_optimizer(Fixed)
        .factor([image.fill(&Line)], 4, &y)?;
    let source: DistanceSource<'_> = image.from_slice(&cross);
    assert_eq!(by_fill.predict([source], 2)?, pred);
    // A table the Gram trick rounded: repaired on request, refused without.
    let rounded = [0.25, 0.25, 2.25, 6.25, 2.25, 0.25, 0.25, -1e-15];
    assert!(fitted.predict([image.borrow(&rounded)], 2).is_err());
    let _ = fitted.predict([image.borrow(&rounded).tidy(1e-12)?], 2)?;

    // ARD: one block per dimension, here two copies of the line. The slot
    // takes its dimensions from its first leaf.
    let (bands, ard) = ArdDistance::from_leaf(RbfArdKernel::new(&[1.0, 2.0])?);
    assert_eq!(bands.dims(), 2);
    let ard: DistanceKernel<DistanceOnly> = ard;
    let blocks: [&[f64]; 2] = [&d2, &d2];
    let fitted = Gpr::new(ard, GaussianLikelihood::new(0.1)?)
        .with_optimizer(Fixed)
        .factor([bands.from_slices(&blocks)], 4, &y)?;
    let cross_blocks: [&[f64]; 2] = [&cross, &cross];
    let _ = fitted.predict([bands.borrow(&cross_blocks)], 2)?;
    let _ = (bands.from_vecs(vec![d2.clone(), d2.clone()]), bands.fill(&Line));

    // A distance leaf times a coordinate leaf.
    let mixed: DistanceKernel<WithPoints> =
        image.kernel(RbfKernel::new(1.0)?) * KernelSpec::from(RbfKernel::new(2.0)?);
    takes_points(&mixed);
    let x = [0.0, 1.0, 2.0, 3.0];
    let fitted = Gpr::new(mixed, GaussianLikelihood::new(0.1)?)
        .with_optimizer(Fixed)
        .factor([image.from_vec(d2)], 4, &x, 1, &y)?;
    assert_eq!((fitted.d(), fitted.x()), (1, &x[..]));
    let _ = fitted.predict([image.borrow(&cross)], &[0.5, 1.5], 2, 1)?;
    let _ = (fitted.to_kernel(), fitted.slots());

    // Sgpr and Svgp: the inducing points are training points 0 and 2, and a
    // source is the 4 × 2 block from the training points to them.
    let train_z = [0.0, 1.0, 4.0, 9.0, 4.0, 1.0, 0.0, 1.0];
    let sgpr = Sgpr::new(kernel.clone(), GaussianLikelihood::new(0.1)?)
        .with_optimizer(Fixed)
        .factor([image.borrow(&train_z)], 4, &y, &[0, 2])
        .map_err(|(_, e)| e)?;
    assert_eq!(sgpr.inducing(), &[0, 2]);
    // The 2 × 2 block from the inducing points to the queries at 0.5 and 1.5.
    let cross_z = [0.25, 2.25, 2.25, 0.25];
    let _ = sgpr.predict([image.borrow(&cross_z)], 2)?;
    // OnlineSgpr: a point at 4 (its squared distances to the inducing
    // points), then training point 1 as a third inducing point (its squared
    // distances to the five training points).
    let mut online = sgpr.clone().into_online();
    online.insert([image.from_vec(vec![16.0, 4.0])], 0.1)?;
    let point = online.point_ids()[1];
    online.insert_inducing(point, [image.from_vec(vec![1.0, 0.0, 1.0, 4.0, 9.0])])?;
    let ids = online.point_ids();
    assert_eq!(online.inducing_points().collect::<Vec<_>>(), [ids[0], ids[2], point]);
    let _ = online.predict([image.from_vec(vec![0.25, 2.25, 0.25])], 1)?;
    let _ = (online.to_kernel(), online.slots());
    online.delete_inducing(online.inducing_ids()[2])?;
    let svgp = Svgp::new(kernel, GaussianLikelihood::new(0.1)?)
        .factor([image.from_vec(train_z.to_vec())], 4, &y, &[0, 2])
        .map_err(|(_, e)| e)?;
    let cov = svgp.predict_covariance([image.borrow(&cross_z)], [image.borrow(&square)], 2)?;
    assert_eq!(cov.covariance.len(), 4);
    Ok(())
}

fn _ard_bound(leaf: impl ArdDistanceLeaf, slot: ArdDistance) -> Result<DistanceKernel, gprx::GprError> {
    slot.kernel(leaf)
}
```

距離のモデルのほかの使い方も、同じ 4 つの学習点で示す。`fit`、`predict` の系統、`sample`、オンラインの `insert` / `delete`、`InvalidDistance` を `match` で受ける処理、`slots()` に現れる ARD のスロット、座標を持つモデルでの入力変換、疎なモデルの保存と読み込みを扱う。

```rust
use gprx::kernel::{
    ArdDistance, DistanceOnly, DistanceSlot, KernelSpec, RbfArdKernel, RbfKernel, ScalarDistance,
};
use gprx::persist::{LoadedDistanceSgpr, LoadedDistanceSvgp, PersistRegistry};
use gprx::transform::MinMaxInput;
use gprx::{
    Fixed, GaussianLikelihood, Gpr, GprError, PredictOptions, Prediction, Sgpr, Svgp,
    VarianceKind,
};

fn main() -> Result<(), GprError> {
    let image = ScalarDistance::new();
    let kernel = image.kernel(RbfKernel::new(1.0)?);
    // Four points 0, 1, 2, 3 and two queries at 0.5 and 1.5, as above.
    let d2 = vec![0.0, 1.0, 4.0, 9.0, 1.0, 0.0, 1.0, 4.0, 4.0, 1.0, 0.0, 1.0, 9.0, 4.0, 1.0, 0.0];
    let y = [0.0, 1.0, 0.5, 0.25];
    let cross = [0.25, 0.25, 2.25, 6.25, 2.25, 0.25, 0.25, 2.25];
    let square = [0.0, 1.0, 1.0, 0.0];

    // `fit` searches θ (L-BFGS by default) on the supplied training square.
    let trainer = Gpr::new(kernel.clone(), GaussianLikelihood::new(0.1)?);
    assert_eq!(trainer.to_kernel().slots(), vec![DistanceSlot::Scalar(image)]);
    let mut fitted = trainer
        .fit([image.from_vec(d2.clone())], 4, &y)
        .map_err(|(_, e)| e)?;
    let latent = PredictOptions {
        variance_kind: VarianceKind::Latent,
    };
    let _ = fitted.predict_with([image.borrow(&cross)], 2, latent)?;
    let mut out = Prediction::default();
    fitted.predict_into([image.borrow(&cross)], 2, &mut out)?;
    let _ = fitted.predict_covariance_with([image.borrow(&cross)], [image.borrow(&square)], 2, latent)?;
    // Three draws, column-major 2 × 3, from seed 7.
    let draws = fitted.sample([image.borrow(&cross)], [image.borrow(&square)], 2, 3, 7)?;
    assert_eq!(draws.len(), 6);

    // A value that is not a squared distance names its slot and pair.
    let negative = [0.25, 0.25, 2.25, 6.25, 2.25, 0.25, 0.25, -2.25];
    match fitted.predict([image.borrow(&negative)], 2) {
        Err(GprError::InvalidDistance { slot, pair, .. }) => {
            assert_eq!((slot, pair), (Some(0), Some((3, 1))));
        }
        other => panic!("{other:?}"),
    }

    // OnlineGpr: a point at 4 (its squared distances to the live points, in
    // `point_ids` order), then its delete.
    let mut online = fitted.into_online()?;
    let id = online.insert([image.from_vec(vec![16.0, 9.0, 4.0, 1.0])], 0.3)?;
    online.delete(id)?;
    let _ = online.predict([image.borrow(&cross)], 2)?;

    // An ARD kernel lists its slot as `DistanceSlot::Ard`.
    let (bands, ard) = ArdDistance::from_leaf(RbfArdKernel::new(&[1.0, 2.0])?);
    assert_eq!(ard.slots(), vec![DistanceSlot::Ard(bands)]);

    // A kernel with points maps its coordinates with an input transform.
    let mixed = image.kernel(RbfKernel::new(1.0)?) * KernelSpec::from(RbfKernel::new(2.0)?);
    let x = [0.0, 10.0, 20.0, 30.0];
    let _ = Gpr::new(mixed, GaussianLikelihood::new(0.1)?)
        .with_optimizer(Fixed)
        .with_input_transform(MinMaxInput::new())
        .factor([image.from_vec(d2)], 4, &x, 1, &y)?;

    // Svgp and OnlineSgpr save; a loaded model binds its new slots.
    let train_z = vec![0.0, 1.0, 4.0, 9.0, 4.0, 1.0, 0.0, 1.0];
    let cross_z = [0.25, 2.25, 2.25, 0.25];
    let svgp = Svgp::new(kernel.clone(), GaussianLikelihood::new(0.1)?)
        .factor([image.from_vec(train_z.clone())], 4, &y, &[0, 2])
        .map_err(|(_, e)| e)?;
    assert_eq!((svgp.inducing(), svgp.slots()), (&[0, 2][..], vec![DistanceSlot::Scalar(image)]));
    let _ = svgp.to_kernel();
    let dir = std::env::temp_dir().join(format!("gprx-readme-distance-svgp-ja-{}", std::process::id()));
    svgp.save(&dir)?;
    let loaded = LoadedDistanceSvgp::<DistanceOnly>::load(&dir, &PersistRegistry::new())?;
    assert_eq!((loaded.n(), loaded.inducing()), (4, &[0, 2][..]));
    let _ = loaded.to_kernel();
    let [DistanceSlot::Scalar(slot)] = loaded.slots()[..] else {
        return Err(GprError::EmptyInput);
    };
    let _ = loaded.predict([slot.borrow(&cross_z)], 2)?;
    let _ = std::fs::remove_dir_all(&dir);
    let online = Sgpr::new(kernel, GaussianLikelihood::new(0.1)?)
        .with_optimizer(Fixed)
        .factor([image.from_vec(train_z)], 4, &y, &[0, 2])
        .map_err(|(_, e)| e)?
        .into_online();
    let dir = std::env::temp_dir().join(format!("gprx-readme-distance-online-sgpr-ja-{}", std::process::id()));
    online.save(&dir)?;
    let _ = LoadedDistanceSgpr::<DistanceOnly>::load(&dir, &PersistRegistry::new())?;
    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}
```

### 尤度

`GaussianLikelihood::new(noise_variance)` は、観測ノイズの分散 `σn²` を log のパラメータとして持つ。log の値から作るには `from_log_noise_variance` を使う。値は `noise_variance` と `log_noise_variance` で読み、区間は `bounds` で読んで `with_bounds` で設定する。`num_params`、`get_params`、`set_params` も使える。`add_noise_diag` はカーネル行列の対角に `σn²` を足し、`noise_grad_diag` はその対角を 1 つのパラメータで微分した値を返す。ノイズの分散が有限でないか正でなければ `InvalidNoiseVariance` になる。正の値でも、ノイズの区間（既定は `(1e-5, 1e5)`）から外れていれば `InvalidInterval` になる。

```rust
use gprx::{GaussianLikelihood, Interval};

fn main() -> Result<(), gprx::GprError> {
    let like = GaussianLikelihood::from_log_noise_variance(0.1_f64.ln())?;
    let _ = (like.noise_variance(), like.log_noise_variance(), like.bounds());
    let like = like.with_bounds(Interval::new(1e-4, 10.0)?)?;
    let mut diag = [1.0, 1.0];
    like.add_noise_diag(&mut diag);
    let mut grad = [0.0, 0.0];
    like.noise_grad_diag(&mut grad, 0)?;
    let mut theta = [0.0; 1];
    like.get_params(&mut theta)?;
    let _ = like.num_params();
    Ok(())
}
```

### 変換（`gprx::transform`）

変換を使うには、学習前の写像を `fit` の前に `with_input_transform` か `with_target_transform` へ渡す。写像を学習データに合わせるのはモデルの側である。何も渡さなければ `IdentityInput` と `IdentityTarget` が使われる。

| 型 | 役割 |
| --- | --- |
| `IdentityInput`、`IdentityTarget` | 何も変えない。`fit` は同じ写像を返す |
| `StandardizeInput` | 特徴ごとの平均と標準偏差を使う。学習後は `FittedStandardizeInput::mean` と `std` で読める |
| `StandardizeTarget` | `y` 全体で 1 組の平均と標準偏差を使う。学習後は `FittedStandardizeTarget::mean` と `std` で読める |
| `MinMaxInput` | 特徴ごとに値を区間へ写す。`new` の区間は `[0, 1]` で、`with_feature_range(lo, hi)` で変えられ、`feature_range` で読める。学習すると `FittedMinMaxInput`（`min`、`max`、`feature_range`）になる |
| `MinMaxTarget` | 同じ変換を `y` にかける。学習すると `FittedMinMaxTarget` になる |
| `Pipeline` | `Pipeline::new().then(step)` で段をつなぐ。`len` と `is_empty` がある。学習すると `FittedPipeline` になる |
| `TargetPipeline` | `y` に対する同じ仕組みで、学習すると `FittedTargetPipeline` になる |
| `ColumnwiseInput` | `new().then(map)` を呼ぶたびに、次の特徴へ写像を割り当てる。学習すると `FittedColumnwiseInput` になる |

具象の写像は、型自身のメソッドとしても `fit` を持つ。入力の写像には `(x, n_rows, n_cols)` を、目的変数の写像には `y` を渡す。トレイトのメソッド `UnfittedTransform::fit` と `UnfittedTarget::fit` は `Box<Self>` を取る。自作の写像では `clone_box` と `as_any` を実装し、保存もしたいなら `persist_id` と `persist_state` も実装する。後の保存と読み込みの例で、入力の写像と目的変数の写像を 1 つずつ定義している。

学習後の写像を表すトレイトは、入力が `Transform`、目的変数が `TargetTransform` である。`Transform` の `apply` と `inverse_apply` は、列優先の配列をその場で書き換える。`TargetTransform` のメソッドは `transform`、`inverse_transform_mean`、`inverse_transform_variance`、`inverse_transform_covariance` で、`inverse_transform_covariance` の既定の実装は、分散と同じ倍率を全要素に掛ける。

```rust
use gprx::transform::{
    ColumnwiseInput, FittedColumnwiseInput, FittedMinMaxInput, FittedMinMaxTarget, FittedPipeline,
    FittedStandardizeInput, FittedStandardizeTarget, FittedTargetPipeline, IdentityInput,
    IdentityTarget, MinMaxInput, MinMaxTarget, Pipeline, StandardizeInput, StandardizeTarget,
    TargetPipeline, TargetTransform, Transform,
};

fn main() -> Result<(), gprx::GprError> {
    let x = [0.0, 2.0, 10.0, 30.0];
    let input: FittedStandardizeInput = StandardizeInput::new().fit(&x, 2, 2)?;
    let _ = (input.mean(), input.std());
    let mut applied = x;
    input.apply(&mut applied, 2, 2)?;
    input.inverse_apply(&mut applied, 2, 2)?;
    let y = [0.0, 1.0, 3.0];
    let target: FittedStandardizeTarget = StandardizeTarget::new().fit(&y)?;
    let _ = (target.mean(), target.std());
    let mut mean = [0.0];
    target.inverse_transform_mean(&mut mean)?;
    let mut variance = [1.0];
    target.inverse_transform_variance(&mut variance)?;
    let mut cov = [1.0];
    target.inverse_transform_covariance(&mut cov)?;
    let unfitted = MinMaxInput::with_feature_range(0.0, 1.0)?;
    let _ = unfitted.feature_range();
    let minmax: FittedMinMaxInput = unfitted.fit(&x, 2, 2)?;
    let _ = (minmax.feature_range(), minmax.min(), minmax.max());
    let target_range: FittedMinMaxTarget = MinMaxTarget::new().fit(&y)?;
    let _ = (target_range.min(), target_range.max(), target_range.feature_range());
    let pipeline = Pipeline::new()
        .then(IdentityInput)
        .then(StandardizeInput::new());
    assert!(!pipeline.is_empty());
    let _ = pipeline.len();
    let fitted_pipeline: FittedPipeline = pipeline.fit(&x, 2, 2)?;
    assert_eq!(fitted_pipeline.len(), 2);
    let targets = TargetPipeline::new()
        .then(IdentityTarget)
        .then(StandardizeTarget::new());
    let fitted_targets: FittedTargetPipeline = targets.fit(&y)?;
    assert!(!fitted_targets.is_empty());
    let columns = ColumnwiseInput::new()
        .then(StandardizeInput::new())
        .then(IdentityInput);
    let fitted_columns: FittedColumnwiseInput = columns.fit(&x, 2, 2)?;
    assert_eq!(fitted_columns.len(), 2);
    Ok(())
}
```

### 最適化

最適化器はモデルの型パラメータの 1 つで、`Gpr::new` では `Lbfgs` が入り、`with_optimizer` で置き換えられる。

| 型 | 探索 | セッター |
| --- | --- | --- |
| `Lbfgs` | L-BFGS と More–Thuente の直線探索。`Differentiable` が必要 | `new` の既定は 100 反復、許容誤差 `sqrt(ε)`、履歴 10。`with_max_iterations`、`with_tolerance`、`with_history_size`（`NonZeroUsize`）、`with_restarts(n, seed)` |
| `NelderMead` | 微分を使わない。`Objective` が必要 | `with_max_iterations`、`with_tolerance`、`with_restarts` |
| `TrustRegion` | ヘッセ行列を使う。`TwiceDifferentiable` が必要 | `with_max_iterations`、`with_tolerance`、`with_restarts`、`with_radii(initial, max)` |
| `FastSimulatedAnnealing` | Cauchy 分布による提案、Metropolis 判定、Ingber の冷却スケジュール。`Objective` が必要 | `with_max_iterations`、`with_restarts`、`with_initial_temperature`、`with_cooling_rate`、`with_seed`、`with_boundary` |
| `Fixed` | 探索しない。使えるのは `factor` だけで、`Optimizer` は実装しない | ユニット構造体 |
| `Adam` | ミニバッチ。`Svgp` 専用 | ミニバッチの節を参照 |

`with_restarts(n, seed)` は、log 空間で一様に選んだ開始点を `n` 個（`NonZeroU32`）追加する。各開始点から探索した結果のうち、目的関数の値が最も小さいものを残す。最初の開始点はモデルの `θ` である。提案が区間の外に出たときの扱いは `BoundaryPolicy` で決まる。既定の `BoundaryPolicy::Clamp` は開区間のすぐ内側へ寄せ、`BoundaryPolicy::Periodic` は反対側へ折り返す。

`Optimizer::minimize` は `OptResult { params, value, iterations }` を返す。定数 `USES_CHANGE_INDICES` の既定は `false` で、`FastSimulatedAnnealing` は `true` にしている。どの座標が変わったかを報告する自作のソルバも `true` にする。そうすると、`CholeskyBuffer::Retain` のときの学習では、変わったカーネルの葉だけを組み直す。自作のソルバは `Optimizer<P>` を実装して作り、`P` には `Objective`、`Differentiable`、`TwiceDifferentiable` のいずれかを指定する。`IncrementalObjective::value_with_changes` は、渡した添字が変えるカーネルの葉だけを組み直す。添字が空のとき、重複しているとき、範囲外のときは `GprError` になる。

目的関数のトレイトは 3 段に分かれている。`Objective` では、`num_params` がパラメータの長さを、`value` がスカラーの値を返す。`value_at_changes` は、その目的関数の直前の評価から変わった座標をすべて列挙する。`fill_intervals` は、各パラメータの開区間を利用者の単位で書き込み、組み込みのソルバはこの区間の内側だけを探す。`Differentiable` は `gradient_into` と `value_and_gradient_into` を加え、`TwiceDifferentiable` はさらに `hessian_into` と `value_gradient_hessian_into` を加える。

```rust
use std::num::{NonZeroU32, NonZeroUsize};

use gprx::kernel::{KernelSpec, RbfKernel};
use gprx::{
    BoundaryPolicy, FastSimulatedAnnealing, Fixed, GaussianLikelihood, Gpr, Lbfgs, NelderMead,
    TrustRegion,
};

fn main() -> Result<(), gprx::GprError> {
    let lbfgs = Lbfgs::new()
        .with_max_iterations(40)
        .with_tolerance(1e-6)?
        .with_history_size(NonZeroUsize::MIN)
        .with_restarts(NonZeroU32::MIN, 1);
    let nm = NelderMead::new()
        .with_max_iterations(20)
        .with_tolerance(1e-6)?;
    let tr = TrustRegion::new()
        .with_max_iterations(20)
        .with_tolerance(1e-6)?
        .with_radii(1.0, 10.0)?;
    let fsa = FastSimulatedAnnealing::new()
        .with_max_iterations(20)
        .with_initial_temperature(1.0)?
        .with_cooling_rate(0.95)?
        .with_seed(1)
        .with_boundary(BoundaryPolicy::Periodic);
    let _clamp = BoundaryPolicy::Clamp;
    let _ = Gpr::new(
        KernelSpec::from(RbfKernel::new(1.0)?),
        GaussianLikelihood::new(0.1)?,
    )
    .with_optimizer(lbfgs)
    .with_optimizer(nm)
    .with_optimizer(tr)
    .with_optimizer(fsa)
    .with_optimizer(Fixed);
    Ok(())
}
```

自作の目的関数とソルバの例を次に示す。

```rust
use gprx::{
    Differentiable, GprError, IncrementalObjective, Interval, Lbfgs, Objective, OptResult,
    Optimizer, TrustRegion, TwiceDifferentiable,
};

/// `f(θ) = (θ − ln 2)²` on one log-parameter: its minimum is at `ℓ = 2`.
struct Bowl;

impl Objective for Bowl {
    fn num_params(&self) -> usize {
        1
    }

    fn value(&mut self, params: &[f64]) -> Result<f64, GprError> {
        Ok((params[0] - 2.0_f64.ln()).powi(2))
    }
}

impl Differentiable for Bowl {
    fn gradient_into(&mut self, params: &[f64], out: &mut [f64]) -> Result<(), GprError> {
        out[0] = 2.0 * (params[0] - 2.0_f64.ln());
        Ok(())
    }
}

impl TwiceDifferentiable for Bowl {
    fn hessian_into(&mut self, _params: &[f64], out: &mut [f64]) -> Result<(), GprError> {
        out[0] = 2.0;
        Ok(())
    }
}

impl IncrementalObjective for Bowl {
    fn value_with_changes(&mut self, params: &[f64], indices: &[usize]) -> Result<f64, GprError> {
        // One parameter: every change is a full evaluation.
        let _ = indices;
        self.value(params)
    }
}

/// Starts at the middle of each interval, then runs L-BFGS.
struct FromMiddle;

impl<P: Differentiable> Optimizer<P> for FromMiddle {
    const USES_CHANGE_INDICES: bool = false;

    fn minimize(&self, objective: &mut P, _init: &[f64]) -> Result<OptResult, GprError> {
        let mut intervals = vec![Interval::DEFAULT_POSITIVE; objective.num_params()];
        objective.fill_intervals(&mut intervals)?;
        let start: Vec<f64> = intervals.iter().map(|i| (i.lo() * i.hi()).sqrt().ln()).collect();
        Lbfgs::new().minimize(objective, &start)
    }
}

fn main() -> Result<(), GprError> {
    let mut bowl = Bowl;
    let _ = (bowl.value_at_changes(&[0.0], &[0])?, bowl.value_with_changes(&[0.0], &[0])?);
    let mut grad = [0.0];
    let value = bowl.value_and_gradient_into(&[0.0], &mut grad)?;
    let mut hess = [0.0];
    let _ = bowl.value_gradient_hessian_into(&[0.0], &mut grad, &mut hess)?;
    let OptResult { params, value: best, iterations } = FromMiddle.minimize(&mut bowl, &[0.0])?;
    assert!((params[0] - 2.0_f64.ln()).abs() < 1e-4 && best <= value);
    let _ = iterations;
    let _ = TrustRegion::new().minimize(&mut bowl, &[0.0])?;
    Ok(())
}
```

### 精度、数学、ジッタ

`DoublePrecision` は、保存も求解も `f64` で行う（`PrecisionPolicy::Storage` と `Refine`）。`SinglePrecision` は `f32` で因子分解してその結果を保持し、予測も `Prediction<f32>` で返す。`MixedPrecision` の既定は `MixedPrecision<PromoteStorage>` で、`f32` で因子分解したあと、予測の重みを `f64` で精緻化し、`f64` で予測する。`ResidualFormula` のもう一つの選択肢が `MixedPrecision<ReevaluateKernel>` である。学習済みモデルの精度の型パラメータ `P` には `GpScalar` という境界があり、3 つの精度ポリシーはどれもこれを実装している。精度は `with_precision::<SinglePrecision>()` のように指定する。

`KernelExp::Accurate` と `FastApprox` は実行時に切り替える値で、`with_math` に渡す。クレート直下にある `Accurate` と `FastApprox` は、それぞれに対応する `KernelMath` の型で、`CompiledKernel::apply::<M>` で使う（カーネルの節を参照）。

`JitterPolicy::fixed(j)` は、コレスキー分解が失敗したとき、対角に `j ≥ 0` を足して一度だけやり直す（`JitterPolicy::Fixed`、`FixedJitter`）。`adaptive(initial, multiplier, max_retries, max_jitter)` は、正則化なしの分解が失敗したあと、足すオフセットを増やしながらやり直す（`JitterPolicy::Adaptive`、`AdaptiveJitter`）。引数は `initial > 0`、`multiplier > 1`、`max_retries ≥ 1`、`max_jitter ≥ initial` を満たす必要がある。既定は、全学習点のモデルが `fixed(0.0)`、`Sgpr` と `Svgp` の `K_mm` が `adaptive(1e-8, 10.0, 5, 1e-3)` である。保存された数値は、`jitter`、または `initial`、`multiplier`、`max_retries`、`max_jitter` で読める。この列挙は non-exhaustive である。

```rust
use gprx::kernel::{KernelSpec, RbfKernel};
use gprx::{
    AdaptiveJitter, DoublePrecision, FittedGpr, FixedJitter, GaussianLikelihood, GpScalar, Gpr,
    JitterPolicy, KernelExp, MixedPrecision, PrecisionPolicy, PromoteStorage, ReevaluateKernel,
    ResidualFormula,
};

/// The storage scalar of a precision policy, by name.
fn storage<P: PrecisionPolicy>() -> &'static str {
    std::any::type_name::<P::Storage>()
}

/// Reads the observation count of any fitted model on a `KernelSpec`.
fn points<P: GpScalar>(model: &FittedGpr<gprx::Lbfgs, P>) -> usize {
    model.n()
}

fn formula<R: ResidualFormula>(_: MixedPrecision<R>) {}

fn main() -> Result<(), gprx::GprError> {
    assert_eq!(storage::<DoublePrecision>(), "f64");
    assert_eq!(storage::<MixedPrecision<PromoteStorage>>(), "f32");
    let _ = formula::<ReevaluateKernel>;
    let mixed = Gpr::new(
        KernelSpec::from(RbfKernel::new(1.0)?),
        GaussianLikelihood::new(0.1)?,
    )
    .with_precision::<MixedPrecision<ReevaluateKernel>>()
    .with_math(KernelExp::Accurate)
    .with_jitter_policy(JitterPolicy::adaptive(1e-8, 10.0, 5, 1e-3)?)
    .fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
    .map_err(|(_, e)| e)?;
    assert_eq!(points(&mixed), 2);

    match JitterPolicy::adaptive(1e-8, 10.0, 5, 1e-3)? {
        JitterPolicy::Adaptive(adaptive) => {
            let adaptive: AdaptiveJitter = adaptive;
            let _ = (adaptive.initial(), adaptive.multiplier());
            let _ = (adaptive.max_retries(), adaptive.max_jitter());
        }
        JitterPolicy::Fixed(fixed) => {
            let _ = fixed.jitter();
        }
        _ => {}
    }
    if let JitterPolicy::Fixed(fixed) = JitterPolicy::fixed(1e-6)? {
        let fixed: FixedJitter = fixed;
        assert_eq!(fixed.jitter(), 1e-6);
    }
    Ok(())
}
```

### パラメータ

`Interval::new(lo, hi)` は、`lo < hi` を満たす有限の開区間を作る。区間の中身は `lo`、`hi`、`contains` で調べられる。エラーの種類は `IntervalError::InvalidBounds` と `OutOfRange` で、`GprError::InvalidInterval` がこのエラーを包む。`BoundedParam::new(value, interval)` は、区間の厳密な内側にある利用者単位の値を持つ。`default_positive(value)` を使うと、区間は `Interval::DEFAULT_POSITIVE` になる。`BoundedParam` の値は `value` で、区間は `interval` で取得できる。`ln` は `log(value)` を返す。`with_value` は区間を保ったまま値を、`with_interval` は値を保ったまま区間を差し替える。カーネルの葉と `GaussianLikelihood` は内部に `BoundedParam` を持つが、呼び出し側はふつう `with_bounds` を通して扱う。

```rust
use gprx::{BoundedParam, Interval, IntervalError};

fn main() -> Result<(), gprx::GprError> {
    assert!(matches!(
        Interval::new(2.0, 1.0),
        Err(IntervalError::InvalidBounds { .. })
    ));
    let interval = Interval::new(1e-3, 1e2)?;
    assert!(interval.contains(1.0));
    let _ = (interval.lo(), interval.hi(), Interval::DEFAULT_POSITIVE);
    let param = BoundedParam::new(1.0, interval)?;
    let _ = (param.value(), param.interval(), param.ln());
    let param = param.with_value(2.0)?.with_interval(interval)?;
    let _ = BoundedParam::default_positive(0.1)?;
    assert!(matches!(
        param.with_value(1e3),
        Err(IntervalError::OutOfRange { .. })
    ));
    Ok(())
}
```

### 保存と読み込み（`gprx::persist`）

保存形式の版を表す `FORMAT_VERSION` は `1`、`RESERVED_PREFIX` は `"gprx."` である。呼び出し側が決める `persist_id` には、この接頭辞を使ってはならない。

座標のモデルのディレクトリは、`LoadedGpr::load(dir, registry)`、`LoadedSgpr::load`、`LoadedSvgp::load` で読み込む。与えられた二乗距離のモデルのディレクトリを読むのは `LoadedDistanceGpr::<C>::load`、`LoadedDistanceSgpr::<C>::load`、`LoadedDistanceSvgp::<C>::load` で、詳しくは後で述べる。`PersistRegistry::new` が作るレジストリは空だが、組み込みのカーネルや変換は登録しなくてよい。自作のカーネルや変換を使っているときは、読み込む前に次のメソッドで登録する。

- `register_kernel`
- `register_unfitted_input`、`register_fitted_input`
- `register_unfitted_target`、`register_fitted_target`

登録する復元関数の型は、`KernelRestore`、`UnfittedInputRestore`、`FittedInputRestore`、`UnfittedTargetRestore`、`FittedTargetRestore` である。

```rust
use gprx::kernel::{KernelSpec, RbfKernel};
use gprx::persist::{
    LoadedGpr, LoadedSgpr, LoadedSvgp, PersistRegistry, FORMAT_VERSION, RESERVED_PREFIX,
};
use gprx::{GaussianLikelihood, Gpr, Lbfgs, PersistErrorKind};

fn main() -> Result<(), gprx::GprError> {
    let fitted = Gpr::new(
        KernelSpec::from(RbfKernel::new(1.0)?),
        GaussianLikelihood::new(0.1)?,
    )
    .fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0])?;
    let dir = std::env::temp_dir().join(format!("gprx-readme-save-ja-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    fitted.save(&dir)?;
    assert_eq!(FORMAT_VERSION, 1);
    assert!(!"mine.kernel".starts_with(RESERVED_PREFIX));

    let loaded = LoadedGpr::load(&dir, &PersistRegistry::new())?;
    let pred = loaded.predict(&[0.5], 1, 1)?;
    assert_eq!(loaded.n(), 2);
    assert_eq!(loaded.d(), 1);
    assert!(!loaded.is_online());
    let LoadedGpr::Double(model) = loaded else {
        return Err(gprx::GprError::PersistFailed {
            kind: PersistErrorKind::WrongModel,
            reason: "expected Double".into(),
        });
    };
    let mut model = model.with_optimizer(Lbfgs::new());
    model.refit()?;
    let factor_dir = std::env::temp_dir().join(format!("gprx-readme-factor-ja-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&factor_dir);
    model.save_with_factor(&factor_dir)?;
    let _ = pred.mean[0];
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&factor_dir);

    let x = &[0.0, 1.0, 2.0, 3.0];
    let y = &[0.0, 1.0, 0.5, 0.25];
    let z = &[0.5, 2.5];
    let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    let likelihood = GaussianLikelihood::new(0.1)?;
    let sparse = gprx::Sgpr::new(kernel.clone(), likelihood.clone())
        .fit(x, 4, 1, y, z, 2)
        .map_err(|(_, e)| e)?;
    let sparse_dir = std::env::temp_dir().join(format!("gprx-readme-sgpr-ja-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&sparse_dir);
    sparse.save(&sparse_dir)?;
    let loaded = LoadedSgpr::load(&sparse_dir, &PersistRegistry::new())?;
    let _ = loaded.m();
    let LoadedSgpr::Double(model) = loaded else {
        return Err(gprx::GprError::PersistFailed {
            kind: PersistErrorKind::WrongModel,
            reason: "expected Double".into(),
        });
    };
    let mut online = model.into_online();
    let id = online.insert_inducing(&[1.5])?;
    assert!(online.inducing_ids().contains(&id));
    online.delete_inducing(id)?;
    let _ = std::fs::remove_dir_all(&sparse_dir);

    let svgp = gprx::Svgp::new(kernel, likelihood)
        .factor(x, 4, 1, y, z, 2)
        .map_err(|(_, e)| e)?;
    let svgp_dir = std::env::temp_dir().join(format!("gprx-readme-svgp-ja-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&svgp_dir);
    svgp.save(&svgp_dir)?;
    let loaded = LoadedSvgp::load(&svgp_dir, &PersistRegistry::new())?;
    let LoadedSvgp::Double(model) = loaded else {
        return Err(gprx::GprError::PersistFailed {
            kind: PersistErrorKind::WrongModel,
            reason: "expected Double".into(),
        });
    };
    let _ = model.neg_elbo()?;
    let _ = std::fs::remove_dir_all(&svgp_dir);
    Ok(())
}
```

次の例では、自作のカーネルの葉、入力の写像、目的変数の写像をレジストリ経由で保存し、読み込む。

```rust
use std::any::Any;
use std::sync::Arc;

use faer::{MatMut, MatRef};
use gprx::kernel::{CustomKernel, KernelScalar, KernelSpec, KernelTerm, RbfKernel, Triangle};
use gprx::persist::{
    FittedInputRestore, FittedTargetRestore, KernelRestore, LoadedGpr, PersistRegistry,
    UnfittedInputRestore, UnfittedTargetRestore,
};
use gprx::transform::{TargetTransform, Transform, UnfittedTarget, UnfittedTransform};
use gprx::{Fixed, GaussianLikelihood, Gpr, GprError, Interval};

/// `k(d²) = exp(−d² / 2)`: an RBF with `ℓ = 1` and no parameter.
#[derive(Clone, Debug)]
struct UnitRbf;

fn no_params(len: usize) -> Result<(), GprError> {
    if len == 0 {
        Ok(())
    } else {
        Err(GprError::LengthMismatch {
            reason: format!("UnitRbf has no parameters, got {len}"),
        })
    }
}

fn no_param(i: usize) -> GprError {
    GprError::IndexOutOfRange {
        reason: format!("UnitRbf has no parameter {i}"),
    }
}

impl<T: KernelScalar> KernelTerm<T> for UnitRbf {
    fn num_params(&self) -> usize {
        0
    }

    fn get_params(&self, out: &mut [f64]) -> Result<(), GprError> {
        no_params(out.len())
    }

    fn set_params(&mut self, params: &[f64]) -> Result<(), GprError> {
        no_params(params.len())
    }

    fn bounds_into(&self, out: &mut [Interval]) -> Result<(), GprError> {
        no_params(out.len())
    }

    fn apply(&self, dist: MatRef<'_, T>, mut out: MatMut<'_, T>, uplo: Triangle) -> Result<(), GprError> {
        for col in 0..dist.ncols() {
            let rows = match uplo {
                Triangle::Lower => col..dist.nrows(),
                Triangle::Upper => 0..col + 1,
                Triangle::Full => 0..dist.nrows(),
            };
            for row in rows {
                out[(row, col)] = (T::from_f64(-0.5) * dist[(row, col)]).exp();
            }
        }
        Ok(())
    }

    fn apply_cross(&self, dist: MatRef<'_, T>, out: MatMut<'_, T>) -> Result<(), GprError> {
        self.apply(dist, out, Triangle::Full)
    }

    fn fill_diag(&self, out: &mut [T]) -> Result<(), GprError> {
        out.fill(T::from_f64(1.0));
        Ok(())
    }

    fn grad(&self, _: MatRef<'_, T>, _: MatMut<'_, T>, i: usize, _: Triangle) -> Result<(), GprError> {
        Err(no_param(i))
    }

    fn hess(&self, _: MatRef<'_, T>, _: MatMut<'_, T>, i: usize, _: usize, _: Triangle) -> Result<(), GprError> {
        Err(no_param(i))
    }

    fn hess_points(&self, _: MatRef<'_, T>, _: MatMut<'_, T>, i: usize, _: usize, _: Triangle) -> Result<(), GprError> {
        Err(no_param(i))
    }

    fn persist_id(&self) -> &'static str {
        "mine.unit_rbf"
    }

    fn persist_state(&self) -> Result<serde_json::Value, GprError> {
        Ok(serde_json::json!({}))
    }
}

/// Moves every input by `−1`. Fitting learns nothing, so one type is both forms.
#[derive(Clone, Debug)]
struct Shift;

impl UnfittedTransform for Shift {
    fn fit(self: Box<Self>, _x: &[f64], _n_rows: usize, _n_cols: usize) -> Result<Box<dyn Transform>, GprError> {
        Ok(self)
    }

    fn clone_box(&self) -> Box<dyn UnfittedTransform> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn persist_id(&self) -> Option<&'static str> {
        Some("mine.shift")
    }

    fn persist_state(&self) -> Result<serde_json::Value, GprError> {
        Ok(serde_json::json!({}))
    }
}

impl Transform for Shift {
    fn apply(&self, x: &mut [f64], _n_rows: usize, _n_cols: usize) -> Result<(), GprError> {
        x.iter_mut().for_each(|v| *v -= 1.0);
        Ok(())
    }

    fn inverse_apply(&self, x: &mut [f64], _n_rows: usize, _n_cols: usize) -> Result<(), GprError> {
        x.iter_mut().for_each(|v| *v += 1.0);
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn Transform> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn persist_id(&self) -> Option<&'static str> {
        Some("mine.shift")
    }

    fn persist_state(&self) -> Result<serde_json::Value, GprError> {
        Ok(serde_json::json!({}))
    }
}

/// Halves `y` for the model and doubles the predictions back.
#[derive(Clone, Debug)]
struct Half;

impl UnfittedTarget for Half {
    fn fit(self: Box<Self>, _y: &[f64]) -> Result<Box<dyn TargetTransform>, GprError> {
        Ok(self)
    }

    fn clone_box(&self) -> Box<dyn UnfittedTarget> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn persist_id(&self) -> Option<&'static str> {
        Some("mine.half")
    }

    fn persist_state(&self) -> Result<serde_json::Value, GprError> {
        Ok(serde_json::json!({}))
    }
}

impl TargetTransform for Half {
    fn transform(&self, y: &mut [f64]) -> Result<(), GprError> {
        y.iter_mut().for_each(|v| *v *= 0.5);
        Ok(())
    }

    fn inverse_transform_mean(&self, mean: &mut [f64]) -> Result<(), GprError> {
        mean.iter_mut().for_each(|v| *v *= 2.0);
        Ok(())
    }

    fn inverse_transform_variance(&self, var: &mut [f64]) -> Result<(), GprError> {
        var.iter_mut().for_each(|v| *v *= 4.0);
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn TargetTransform> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn persist_id(&self) -> Option<&'static str> {
        Some("mine.half")
    }

    fn persist_state(&self) -> Result<serde_json::Value, GprError> {
        Ok(serde_json::json!({}))
    }
}

fn main() -> Result<(), GprError> {
    // Two copies of the custom leaf, summed, times a built-in leaf.
    let kernel = KernelSpec::from(CustomKernel::new(UnitRbf)) + KernelSpec::custom(UnitRbf);
    let kernel = kernel * KernelSpec::from(RbfKernel::new(2.0)?);
    let fitted = Gpr::new(kernel, GaussianLikelihood::new(0.1)?)
        .with_optimizer(Fixed)
        .with_input_transform(Shift)
        .with_target_transform(Half)
        .factor(&[0.0, 1.0, 2.0], 3, 1, &[0.0, 1.0, 0.5])?;
    let dir = std::env::temp_dir().join(format!("gprx-readme-custom-ja-{}", std::process::id()));
    fitted.save(&dir)?;

    // One restore function per persist id.
    let kernel: KernelRestore = Arc::new(|_state| Ok(CustomKernel::new(UnitRbf)));
    let unfitted_input: UnfittedInputRestore = Arc::new(|_state| Ok(Box::new(Shift)));
    let fitted_input: FittedInputRestore = Arc::new(|_state| Ok(Box::new(Shift)));
    let unfitted_target: UnfittedTargetRestore = Arc::new(|_state| Ok(Box::new(Half)));
    let fitted_target: FittedTargetRestore = Arc::new(|_state| Ok(Box::new(Half)));
    let mut registry = PersistRegistry::new();
    registry.register_kernel("mine.unit_rbf", move |state| kernel(state))?;
    registry.register_unfitted_input("mine.shift", move |state| unfitted_input(state))?;
    registry.register_fitted_input("mine.shift", move |state| fitted_input(state))?;
    registry.register_unfitted_target("mine.half", move |state| unfitted_target(state))?;
    registry.register_fitted_target("mine.half", move |state| fitted_target(state))?;

    let loaded = LoadedGpr::load(&dir, &registry)?;
    let query = [0.5];
    assert_eq!(loaded.predict(&query, 1, 1)?, fitted.predict(&query, 1, 1)?);
    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}
```

与えられた二乗距離のモデルも、同じ `save` / `save_with_factor` で保存する。保存したファイルには、モデルが保持する学習の `d²`（全学習点のモデルなら下三角、疎なモデルなら `n × m` のブロックと誘導点の添字）と、カーネルのスロットの表も入る。ローダーの型はカーネルの印（`DistanceOnly` か `WithPoints`）で決まり、座標のモデルのファイルや、もう一方の印のファイルを読もうとすると `WrongModel` になる。読み込んだカーネルが持つのは新しいスロットで、保存前の `ScalarDistance` や `ArdDistance` はそのどれにも当たらない。予測では、`slots()`（保存したカーネルでの順）からスロットを取り出し、クエリの供給をそれに結び付ける。形式の詳細は [persist-format.ja.md 10 節](https://github.com/YUKIKEDA/gprx/blob/main/docs/persist-format.ja.md#10-与えられた二乗距離のモデル)にある。

```rust
use gprx::kernel::{
    DistanceOnly, DistanceSlot, KernelSpec, RbfKernel, ScalarDistance, WithPoints,
};
use gprx::persist::{LoadedDistanceGpr, LoadedDistanceSgpr, LoadedDistanceSvgp, PersistRegistry};
use gprx::{Fixed, GaussianLikelihood, GprError, Gpr, Sgpr, Svgp};

/// The one scalar slot of a loaded kernel.
fn scalar(slots: &[DistanceSlot]) -> Result<ScalarDistance, GprError> {
    match slots {
        [DistanceSlot::Scalar(slot)] => Ok(*slot),
        _ => Err(GprError::InvalidConfig {
            reason: "expected one scalar slot".into(),
        }),
    }
}

fn main() -> Result<(), GprError> {
    let registry = PersistRegistry::new();
    let dir = std::env::temp_dir().join(format!("gprx-readme-distance-ja-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let image = ScalarDistance::new();
    // Three samples at 0, 1, 2: their 3 × 3 squared distances.
    let train = [0.0, 1.0, 4.0, 1.0, 0.0, 1.0, 4.0, 1.0, 0.0];
    let y = [0.0, 1.0, 0.5];
    // The three samples × one query at 0.5.
    let cross = [0.25, 0.25, 2.25];

    let fitted = Gpr::new(image.kernel(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?)
        .with_optimizer(Fixed)
        .factor([image.borrow(&train)], 3, &y)
        .map_err(|(_, e)| e)?;
    fitted.save(&dir)?;
    // The marker is part of the loaded type: name it.
    let loaded = LoadedDistanceGpr::<DistanceOnly>::load(&dir, &registry)?;
    assert_eq!((loaded.n(), loaded.is_online()), (3, false));
    // The loaded kernel has new slots: bind the query to them.
    let slot = scalar(&loaded.slots())?;
    assert_eq!(loaded.to_kernel().slots(), loaded.slots());
    let pred = loaded.predict([slot.borrow(&cross)], 1)?;
    assert_eq!(pred.mean, fitted.predict([image.borrow(&cross)], 1)?.mean);

    // A kernel with coordinate leaves is `WithPoints`: the query adds its coordinates.
    let kernel = image.kernel(RbfKernel::new(1.0)?) * KernelSpec::from(RbfKernel::new(0.5)?);
    Gpr::new(kernel, GaussianLikelihood::new(0.1)?)
        .with_optimizer(Fixed)
        .factor([image.borrow(&train)], 3, &[0.0, 1.0, 2.0], 1, &y)
        .map_err(|(_, e)| e)?
        .save(&dir)?;
    let loaded = LoadedDistanceGpr::<WithPoints>::load(&dir, &registry)?;
    assert_eq!(loaded.d(), 1);
    let slot = scalar(&loaded.slots())?;
    let _ = loaded.predict_with([slot.borrow(&cross)], &[0.5], 1, 1, gprx::PredictOptions::default())?;

    // Sparse: the blocks from the three samples to the inducing samples 0 and 2.
    let blocks = [0.0, 1.0, 4.0, 4.0, 1.0, 0.0];
    Sgpr::new(image.kernel(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?)
        .with_optimizer(Fixed)
        .factor([image.borrow(&blocks)], 3, &y, &[0, 2])
        .map_err(|(_, e)| e)?
        .save(&dir)?;
    let loaded = LoadedDistanceSgpr::<DistanceOnly>::load(&dir, &registry)?;
    assert_eq!((loaded.m(), loaded.inducing()), (2, Some(&[0, 2][..])));
    // A loaded online model names its inducing points by `PointId` instead.
    assert!(loaded.inducing_points().is_none());
    let slot = scalar(&loaded.slots())?;
    // The inducing samples × the query at 0.5.
    let _ = loaded.predict([slot.borrow(&[2.25, 0.25])], 1)?;
    let LoadedDistanceSgpr::Double(model) = loaded else {
        return Err(GprError::InvalidConfig {
            reason: "expected Double".into(),
        });
    };
    let _online = model.into_online();

    Svgp::new(image.kernel(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?)
        .factor([image.borrow(&blocks)], 3, &y, &[0, 2])
        .map_err(|(_, e)| e)?
        .save(&dir)?;
    let loaded = LoadedDistanceSvgp::<DistanceOnly>::load(&dir, &registry)?;
    let slot = scalar(&loaded.slots())?;
    let _ = loaded.predict([slot.borrow(&[2.25, 0.25])], 1)?;
    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}
```

読み込んだモデルの `predict` と `predict_with` は、ファイルが `f32` で書かれていても `f64` で結果を返す。`n` と `d`、疎なモデルでは `m` も読める。`is_online` は、`OnlineGpr` か `OnlineSgpr` が書いたファイルなら真を返す。型の決まったモデルとして扱うには、次のバリアントで `match` する。

| 列挙 | バリアント |
| --- | --- |
| `LoadedGpr` | `Double`、`Single`、`Mixed`、`Reevaluate`、`OnlineDouble`、`OnlineSingle`、`OnlineMixed`、`OnlineReevaluate` |
| `LoadedSgpr` | 同じ名前の 8 つ。`Double` の中身は `FittedSgpr<Fixed>`、オンラインのバリアントは `OnlineSgpr<Fixed, _>` |
| `LoadedSvgp` | `Double`、`Single`、`Mixed`、`Reevaluate`。オンラインのバリアントはない |
| `LoadedDistanceGpr<C>` | `LoadedGpr` と同じ名前の 8 つで、中身は `FittedGpr<Fixed, _, DistanceKernel<C>>` / `OnlineGpr<Fixed, _, DistanceKernel<C>>`。`C` は `DistanceOnly` か `WithPoints` で、`load` を呼ぶときに指定する |
| `LoadedDistanceSgpr<C>` | `LoadedSgpr` と同じ名前の 8 つで、中身は `DistanceKernel<C>` を使うモデル |
| `LoadedDistanceSvgp<C>` | `LoadedSvgp` と同じ名前の 4 つで、中身は `FittedSvgp<_, DistanceKernel<C>>` |

読み込んだ全学習点のモデルは、`Fixed` と `CholeskyBuffer::Retain` の組み合わせになる。ファイルにはソルバが保存されないので、もう一度探索したいときは、`match` で取り出したモデルの `with_optimizer` を呼んでから `refit` する。`save` で因子なしに保存したファイルは、読み込むときに因子分解をやり直す。`save_with_factor` で保存したファイルなら分解し直さず、`f64` のモデルは `L` をメモリマップしたまま使い、`f32` のモデルはコピーして使う。

### エラー

`GprError` は non-exhaustive で、表示される文字列は英語である。

| バリアント | 起きる場面 |
| --- | --- |
| `DimensionMismatch { x_dim, expected_dim }` | クエリの特徴数が学習時と違う |
| `InsufficientData { n, min }` | 点の数が足りない |
| `EmptyInput` | 次元のどれかが 0 |
| `NonFiniteInput` | 呼び出し側のデータに `NaN` か `Inf` がある |
| `NonFiniteKernelValue` | カーネルの評価値が有限でない |
| `CholeskyFailed { jitter, matrix_size, stage }` | ジッタを足しても因子分解に失敗した |
| `NonPositiveDefiniteMatrix` | 共分散行列であるべき行列がそうなっていない |
| `CoordGradientUnsupported` | 疎なモデルに必要な微分をカーネルが持たない |
| `OptimizationNotConverged { iterations }` | ソルバが収束判定を満たす前に止まった |
| `InvalidHyperparameter { reason }` | カーネルのパラメータが定義域の外にある |
| `ShapeMismatch { reason }` | 行列の形が合わない |
| `InvalidDistance { slot, dim, pair, reason }` | 与えられた二乗距離が有限でない、負である、または正方行列の対角が 0 でないか対称でない。`slot` はその表のカーネルの `slots()` での位置、`dim` は ARD の次元、`pair` は渡したブロックの中の `(row, col)` で、分からないものは `None` になる |
| `DistanceSlot { kind, slot }` | 供給がカーネルのスロットと合わない。種類は `SlotErrorKind::NotRead`（カーネルが読まないスロット。保存前のスロットなど）、`Duplicate`、`Missing` のいずれか。`slot` は、供給が 2 つあるか 1 つもないスロットの `slots()` での位置で、`NotRead` では `None` になる |
| `LengthMismatch { reason }` | スライスの長さが合わない |
| `IndexOutOfRange { reason }` | パラメータ、カーネルの葉、次元の添字が範囲外 |
| `InvalidConfig { reason }` | 最適化器、ジッタ、変換の設定が正しくない |
| `SizeOverflow` | `n_rows * n_cols` が `usize` に収まらない |
| `InvalidInterval` | `Interval` を作れないか、値が区間の外にある（既定では、ノイズの分散が `(1e-5, 1e5)` の外） |
| `InvalidNoiseVariance { reason }` | 観測ノイズが定義域の外にある |
| `UnsupportedKernelOperation { reason }` | そのカーネルの葉がその操作を実装していない |
| `WorkspaceTooSmall` | バッファが問題の大きさより短い |
| `InvalidPointId` | `PointId` がモデルにない |
| `InvalidInducingId` | `InducingId` がモデルにない |
| `PersistFailed { kind, reason }` | 保存か読み込みに失敗した |
| `UnsupportedPersistVersion { found, supported }` | `format_version` が `FORMAT_VERSION` と違う |

`CholeskyStage` は `Fit`、`Predict`、`OnlineInsert`、`OnlineDelete` のいずれかで、`PersistErrorKind` は `Io`、`Config`、`Tensor`、`InvalidPersistId`、`NotPersistable`、`UnregisteredId`、`WrongModel` のいずれかである。プログラムで分岐するときは `kind` を使う。`reason` は人が読むための説明文である。

```rust
use gprx::{CholeskyStage, GprError, PersistErrorKind, SlotErrorKind};

fn main() -> Result<(), GprError> {
    let missing = GprError::DistanceSlot {
        kind: SlotErrorKind::Missing,
        slot: Some(0),
    };
    assert!(matches!(missing, GprError::DistanceSlot { kind: SlotErrorKind::Missing, .. }));
    let err = GprError::DimensionMismatch {
        x_dim: 2,
        expected_dim: 1,
    };
    match err {
        GprError::DimensionMismatch { x_dim, expected_dim } => {
            let _ = (x_dim, expected_dim);
        }
        GprError::CholeskyFailed {
            jitter,
            matrix_size,
            stage,
        } => {
            let _ = (jitter, matrix_size, stage);
        }
        GprError::PersistFailed { kind, reason } => {
            let _ = (kind, reason);
        }
        GprError::UnsupportedPersistVersion { found, supported } => {
            let _ = (found, supported);
        }
        other => {
            let _ = other.to_string();
        }
    }
    let _ = (CholeskyStage::Fit, PersistErrorKind::Io);
    Ok(())
}
```

## アーキテクチャと保存フォーマット

3 つのモデル（`Gpr`、`Sgpr`、`Svgp`）は共通の部品から組み立てられており、互いを import しない。クレート全体の地図は[アーキテクチャの全体像](https://github.com/YUKIKEDA/gprx/blob/main/docs/architecture.ja.md#1-全体像)にある。

- [`docs/architecture.ja.md`](https://github.com/YUKIKEDA/gprx/blob/main/docs/architecture.ja.md): すべてのモジュールの責務と import の向き、モデルごとの公開型、変えたい内容ごとに見るべき場所をまとめている。
- [`docs/persist-format.ja.md`](https://github.com/YUKIKEDA/gprx/blob/main/docs/persist-format.ja.md): `save` が書き出す内容を説明している。`config.json` のキー、`model.safetensors` のテンソル（名前、形、dtype、列優先の並び）、カーネルと変換の JSON の形、`Custom` の復元、版、エラーを扱う。

## 比較

精度、学習時間、メモリ使用量を他のライブラリと比べた結果は、[ライブラリの比較](https://github.com/YUKIKEDA/gprx/blob/main/docs/comparison.ja.md)にまとめてある。英語版は [comparison.md](https://github.com/YUKIKEDA/gprx/blob/main/docs/comparison.md) である。

## ライセンス

次のいずれかのライセンスを選んで利用できる。

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))
