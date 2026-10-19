[English](README.md) | 日本語

# gprx

gprx は、Rust でガウス過程回帰を行うクレートである。全学習点を使う厳密な GPR、スパース GPR（VFE）、SVGP の 3 つを備えている。`Gpr` は学習前のトレーナーである。`Gpr::fit` はこのトレーナーを値で受け取って消費し、負の対数周辺尤度を既定の `Lbfgs` で最小化したうえで `FittedGpr` を返す。`Sgpr` と `Svgp` も、同じカーネルと尤度を受け取って組み立てる。厳密な GPR とスパース GPR のモデルは学習後もオンラインで更新でき、どのモデルもディレクトリへ保存して読み込める。カーネルに渡す入力は座標でもよいし、呼び出し側が求めた二乗距離でもよい。

入力 `x` は列優先で並べる。`n` 個の点と `d` 個の特徴があるなら、全行の特徴 0 を先に置き、そのあとに特徴 1 を続ける。観測ノイズの大きさは `GaussianLikelihood` が持つ。

公開 API として約束しているのは、**0.2.0**（タグ `v0.2.0`）のうち既定のフィーチャで使える部分である。MSRV は 1.88 とする。0.x のあいだは、マイナー版の更新でもこの API に互換性のない変更が入ることがある。0.1.0 から 0.2.0 で何が変わり、どう移行すればよいかは[変更履歴](https://github.com/YUKIKEDA/gprx/blob/main/CHANGELOG.md)（英語）にまとめてある。`gprx::internals` モジュールは、フィーチャ `bench-internals` か `insert-stages` を有効にしたときだけ現れ、この約束には含まれない。

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

## 最初に知っておくこと

`fit` と `factor` が失敗すると、エラーと一緒にトレーナーが戻ってくる。戻り値の型は `Result<Fitted, (Trainer, GprError)>` である。`Gpr` には `(Gpr, GprError)` から `GprError` への変換があるので、`?` でそのまま受けられる。このときトレーナーは捨てられ、`GprError` だけが呼び出し元へ伝わる。`Sgpr` と `Svgp` にはこの変換がないので、以下の例のように `.map_err(|(_, e)| e)?` と書く。

`x` には、`n_rows` 個の点と `n_cols` 個の特徴を列優先で並べた `f64` を渡す。`y` の長さは `n_rows` にそろえる。入力が空のとき、長さが `n_rows * n_cols` と合わないとき、`NaN` や `Inf` が混じっているときは、パニックせずに `GprError` を返す。

メソッドは `fitted.predict(..)` のように値に対して呼ぶ。`FittedGpr::predict(&fitted, ..)` のように型パラメータを省いて型のパスから呼ぶと、`E0034` のエラーになり、コンパイルできない。二乗距離を受け取るモデルにも同じ名前のメソッドがあって、どれを指すのかが決まらないからだ。パスで呼びたいなら、値が持つ型パラメータを書く。型パラメータが既定のモデルなら `<FittedGpr>::predict(&fitted, ..)`、`factor` で作ったモデルなら `<FittedGpr<Fixed>>::predict(&frozen, ..)` と書く。

## モデルの選び方

| モデル | 向いている場面 | 学習の計算量 |
| --- | --- | --- |
| `Gpr` | `n × n` のカーネル行列がメモリに載る。予測に近似が入らない | 時間 `O(n³)`、メモリ `O(n²)` |
| `Sgpr` | `n` が大きい。`m` 個の誘導点でデータを要約する（VFE） | 目的関数の評価 1 回につき `O(n m²)` |
| `Svgp` | `n` が大きく、`Adam` でミニバッチ学習をしたい | バッチ `b` の 1 ステップにつき `O(b (m² + m d) + m³)` |

学習のあとで点を足したり除いたりするなら、`FittedGpr::into_online` と `FittedSgpr::into_online` でそれぞれ `OnlineGpr` と `OnlineSgpr` に変える。どちらも fit をやり直さずに点を追加・削除できる。`Svgp` にはオンライン版がない。

## 使い方

以下のコードはどれもそのまま動くプログラムである。特に断りがなければ、依存は `gprx = "0.2"` だけでよい。

### カーネルを選ぶ・組み合わせる

カーネルは `+` で足し、`*` で掛ける。`*` は `+` より先に結合する。特徴ごとに別々の長さスケールを持たせたいときは、`RbfArdKernel` のような ARD（automatic relevance determination、関連度の自動決定）のカーネルの葉を使う。用意されているカーネルの葉と、それぞれのパラメータや範囲は、[`gprx::kernel`](https://docs.rs/gprx/latest/gprx/kernel/) の rustdoc に一覧がある。

```rust
use gprx::kernel::{ConstantKernel, KernelSpec, PeriodicKernel, RbfArdKernel, RbfKernel};
use gprx::{GaussianLikelihood, Gpr};

fn main() -> Result<(), gprx::GprError> {
    // A signal variance times a smooth trend, plus a periodic term.
    let trend = KernelSpec::from(ConstantKernel::new(2.0)?) * KernelSpec::from(RbfKernel::new(5.0)?);
    let seasonal = trend + KernelSpec::from(PeriodicKernel::new(1.0, 0.5)?);
    assert_eq!(seasonal.num_params(), 4);

    // ARD: three points with two features, column-major.
    let ard = KernelSpec::from(RbfArdKernel::new(&[1.0, 10.0])?);
    let x = [0.0, 1.0, 2.0, 5.0, 3.0, 8.0];
    let fitted = Gpr::new(ard, GaussianLikelihood::new(0.1)?).fit(&x, 3, 2, &[0.0, 1.0, 0.5])?;
    let pred = fitted.predict(&[1.5, 4.0], 1, 2)?;
    println!("mean = {}", pred.mean[0]);
    Ok(())
}
```

### 自分の距離行列を使う

測地線距離やグラフ上の距離のように、座標からは求まらない距離でモデルを作りたいことがある。`ScalarDistance` のスロットを作り、その `kernel` にカーネルの葉を渡すと、返ってきたカーネルは座標の代わりに、呼び出し側で計算した二乗距離を読む。二乗距離の表は列優先で並べる。値はどれも有限で 0 以上でなければならず、学習用の表はさらに対称で、対角が 0 でなければならない。条件から外れると `GprError::InvalidDistance` になる。丸め誤差でわずかにずれただけの表なら、`from_vec` や `borrow` が返す `DistanceSource` に `tidy(rel_tol)` をかけて直せる。

特徴ごとに表を分ける `ArdDistance`、座標を読むカーネルの葉との組み合わせ（`WithPoints`）、疎なモデルに渡す引数は、[`gprx::kernel`](https://docs.rs/gprx/latest/gprx/kernel/) の rustdoc で説明している。列を必要になったときに埋める `fill` の例は [`ArdDistance`](https://docs.rs/gprx/latest/gprx/kernel/struct.ArdDistance.html) に、カーネルの組み合わせ方の例は [`DistanceKernel`](https://docs.rs/gprx/latest/gprx/kernel/struct.DistanceKernel.html) の rustdoc にある。

`Sgpr` と `Svgp` の因子分解は、座標のモデルなら `n · d` 個の座標から計算する点の組ごとの二乗距離を、与えられた `d · n · m` 個の値として読む。そのため ARD のスロットでは、因子分解の計算量が座標のモデルの因子分解の計算量より大きくなる。一方、勾配と予測の計算量は、座標のモデルの計算量を上回らない（[設計 §5.6](https://github.com/YUKIKEDA/gprx/blob/main/docs/design.ja.md#56-与えられた二乗距離-要件470)）。

```rust
use gprx::kernel::{RbfKernel, ScalarDistance};
use gprx::{GaussianLikelihood, Gpr};

fn main() -> Result<(), gprx::GprError> {
    let slot = ScalarDistance::new();
    let kernel = slot.kernel(RbfKernel::new(1.0)?);
    // Three training points: their 3 × 3 squared distances.
    let d2 = vec![0.0, 1.0, 4.0, 1.0, 0.0, 1.0, 4.0, 1.0, 0.0];
    let y = [0.0, 1.0, 0.5];
    let fitted = Gpr::new(kernel, GaussianLikelihood::new(0.1)?).fit([slot.from_vec(d2)], 3, &y)?;
    // One query: its squared distances to the three training points (3 × 1).
    let cross = [0.25, 0.25, 2.25];
    let pred = fitted.predict([slot.borrow(&cross)], 1)?;
    println!("mean = {}", pred.mean[0]);
    Ok(())
}
```

### 誘導点で大きなデータを扱う

`Sgpr` には、`y` のあとに誘導点の入力 `z` を渡す。並べ方は `x` と同じ列優先である。誘導点は渡した位置から動かない。`with_inducing(FreeInducing)` を付ければ、ハイパーパラメータと一緒に誘導点の位置も最適化する。そのほかの設定は [`Sgpr`](https://docs.rs/gprx/latest/gprx/struct.Sgpr.html) の rustdoc で説明している。

```rust
use gprx::kernel::{KernelSpec, RbfKernel};
use gprx::{GaussianLikelihood, Sgpr};

fn main() -> Result<(), gprx::GprError> {
    // 500 points on [0, 25), one feature.
    let x: Vec<f64> = (0..500).map(|i| f64::from(i) * 0.05).collect();
    let y: Vec<f64> = x.iter().map(|v| v.sin()).collect();
    // 20 inducing points.
    let z: Vec<f64> = (0..20).map(|i| f64::from(i) * 1.25).collect();
    let fitted = Sgpr::new(KernelSpec::from(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?)
        .fit(&x, 500, 1, &y, &z, 20)
        .map_err(|(_, e)| e)?;
    let pred = fitted.predict(&[3.3], 1, 1)?;
    println!("mean = {}, variance = {}", pred.mean[0], pred.variance[0]);
    Ok(())
}
```

### ミニバッチで学習する

`Svgp::new` で作ったモデルは、そのままでは学習しない。`factor` は変分事後分布を事前分布と同じ値で初期化するだけだ。`with_optimizer(Adam::new())` を付けると、`fit` がミニバッチの Adam でハイパーパラメータと変分事後分布を最適化する。`Adam` の既定値は、学習率 `1e-3`、バッチの大きさ 32、エポック数 100、シード 0 である。`Adam` を使えるのは `Svgp` だけだ。そのほかの設定は [`Svgp`](https://docs.rs/gprx/latest/gprx/struct.Svgp.html) と [`Adam`](https://docs.rs/gprx/latest/gprx/struct.Adam.html) の rustdoc にある。

```rust
use gprx::kernel::{KernelSpec, RbfKernel};
use gprx::{Adam, GaussianLikelihood, Svgp};

fn main() -> Result<(), gprx::GprError> {
    let x: Vec<f64> = (0..200).map(|i| f64::from(i) * 0.05).collect();
    let y: Vec<f64> = x.iter().map(|v| v.sin()).collect();
    let z: Vec<f64> = (0..10).map(f64::from).collect();
    let adam = Adam::new().with_learning_rate(1e-2)?.with_seed(7);
    let fitted = Svgp::new(KernelSpec::from(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?)
        .with_optimizer(adam)
        .fit(&x, 200, 1, &y, &z, 10)
        .map_err(|(_, e)| e)?;
    println!("negative ELBO = {}", fitted.neg_elbo()?);
    let pred = fitted.predict(&[2.5], 1, 1)?;
    println!("mean = {}", pred.mean[0]);
    Ok(())
}
```

### 点をあとから追加・削除する

`into_online` で変えたモデルは、ハイパーパラメータをそのままにして、1 点ずつ分解を更新する。`OnlineGpr` なら 1 回の更新は `O(n²)` で済む。`insert` が返す `PointId` を `delete` に渡すと、その点を取り除ける。最後に残った 1 点は削除できない。今ある点でハイパーパラメータを探し直したくなったら `refit` を呼ぶ。`OnlineSgpr` では、誘導点も `insert_inducing` と `delete_inducing` で増やしたり減らしたりできる。`point_ids` など、ほかのメソッドは [`OnlineGpr`](https://docs.rs/gprx/latest/gprx/struct.OnlineGpr.html) と [`OnlineSgpr`](https://docs.rs/gprx/latest/gprx/struct.OnlineSgpr.html) の rustdoc で説明している。

```rust
use gprx::kernel::{KernelSpec, RbfKernel};
use gprx::{GaussianLikelihood, Gpr, OnlineGpr, PointId};

fn main() -> Result<(), gprx::GprError> {
    let fitted = Gpr::new(KernelSpec::from(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?)
        .fit(&[0.0, 1.0, 2.0], 3, 1, &[0.0, 1.0, 0.5])?;
    let mut online: OnlineGpr = fitted.into_online()?;
    // A new point at x = 3 with y = 0.25.
    let id: PointId = online.insert(&[3.0], 0.25)?;
    let pred = online.predict(&[2.5], 1, 1)?;
    online.delete(id)?;
    online.refit()?;
    println!("mean = {}, points = {}", pred.mean[0], online.n());
    Ok(())
}
```

### ハイパーパラメータを固定する・最適化器を選ぶ

`Gpr::new` と `Sgpr::new` は `Lbfgs` でハイパーパラメータを探す。`Lbfgs` は、`with_max_iterations` で変えない限り 100 回の反復で止まる。`with_optimizer` で `NelderMead`、`TrustRegion`、`FastSimulatedAnnealing` に替えることもでき、どれにも専用の設定メソッドがある。与えた値のまま使いたいなら `with_optimizer(Fixed)` とする。このモデルには `fit` がなく、代わりに `factor` を呼ぶ。探索してよい範囲は、カーネルの葉や `GaussianLikelihood` の `with_bounds` で狭められる。最適化器どうしの違いと、自分で最適化器を書く方法は、[`Optimizer`](https://docs.rs/gprx/latest/gprx/trait.Optimizer.html) の rustdoc にまとめてある。

```rust
use std::num::NonZeroU32;

use gprx::kernel::{KernelSpec, RbfKernel};
use gprx::{Fixed, GaussianLikelihood, Gpr, Interval, Lbfgs, NelderMead};

fn main() -> Result<(), gprx::GprError> {
    let (x, y) = ([0.0, 1.0, 2.0], [0.0, 1.0, 0.5]);
    // Keep the lengthscale inside (0.1, 10).
    let rbf = RbfKernel::new(1.0)?.with_bounds(Interval::new(0.1, 10.0)?)?;
    let kernel = KernelSpec::from(rbf);
    let likelihood = GaussianLikelihood::new(0.1)?;

    let fixed = Gpr::new(kernel.clone(), likelihood.clone()).with_optimizer(Fixed).factor(&x, 3, 1, &y)?;
    // 200 iterations and one extra start drawn with seed 42.
    let lbfgs = Lbfgs::new().with_max_iterations(200).with_restarts(NonZeroU32::MIN, 42);
    let searched = Gpr::new(kernel.clone(), likelihood.clone()).with_optimizer(lbfgs).fit(&x, 3, 1, &y)?;
    let simplex = Gpr::new(kernel, likelihood).with_optimizer(NelderMead::new()).fit(&x, 3, 1, &y)?;
    for nlml in [fixed.neg_log_marginal_likelihood()?, searched.neg_log_marginal_likelihood()?, simplex.neg_log_marginal_likelihood()?] {
        println!("negative log marginal likelihood = {nlml}");
    }
    Ok(())
}
```

### 入力と目的変数を変換する

変換は、学習前のトレーナーに `with_input_transform` と `with_target_transform` で渡す。渡さなければ、`x` と `y` はそのままモデルに入る。モデルは変換のパラメータを学習データから求め、予測のたびにクエリにも同じ変換をかけ、予測値を `y` の元の単位に戻して返す。事前平均は 0 なので、`y` が 0 の近くに集まっていないなら `StandardizeTarget` を使う。値を決まった範囲に収める min-max 変換、変換の連結、列ごとに違う変換、自作の変換に必要なトレイトは [`gprx::transform`](https://docs.rs/gprx/latest/gprx/transform/) にある。

```rust
use gprx::kernel::{KernelSpec, RbfKernel};
use gprx::transform::{StandardizeInput, StandardizeTarget};
use gprx::{GaussianLikelihood, Gpr};

fn main() -> Result<(), gprx::GprError> {
    // Inputs in the thousands, targets far from zero.
    let x = [1000.0, 2000.0, 3000.0, 4000.0];
    let y = [250.0, 260.0, 255.0, 270.0];
    let fitted = Gpr::new(KernelSpec::from(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?)
        .with_input_transform(StandardizeInput::new())
        .with_target_transform(StandardizeTarget::new())
        .fit(&x, 4, 1, &y)?;
    // The query and the prediction are in the original units.
    let pred = fitted.predict(&[2500.0], 1, 1)?;
    println!("mean = {}", pred.mean[0]);
    Ok(())
}
```

### 保存して読み込む

`save(dir)` は、ディレクトリに `config.json` と `model.safetensors` を書き出す。`LoadedGpr::load` が返す enum は、精度とオンラインかどうかの組み合わせごとに variant が分かれている。`predict` はどの variant でも呼べて、結果は `f64` で返る。型のついたモデルが要るときは、variant で分岐して取り出す。`FittedGpr::save_with_factor` はコレスキー因子も書き出すので、読み込むときに分解し直さずに済む。読み込んだ `FittedGpr` と `OnlineGpr` は、`with_optimizer` のあとに `refit` を呼べばハイパーパラメータを探し直せる。読み込んだ SGPR と SVGP のモデルにはその手段がないので、探し直すときはそのカーネルと尤度を新しい `Sgpr` か `Svgp` に渡す。`LoadedSgpr`、`LoadedSvgp`、二乗距離を受け取るモデルのローダー、自作のカーネルや変換を登録する `PersistRegistry` は [`gprx::persist`](https://docs.rs/gprx/latest/gprx/persist/) にある。ファイルの形式は [persist-format.ja.md](https://github.com/YUKIKEDA/gprx/blob/main/docs/persist-format.ja.md) で説明している。

```rust
use gprx::kernel::{KernelSpec, RbfKernel};
use gprx::persist::{LoadedGpr, PersistRegistry};
use gprx::{GaussianLikelihood, Gpr};

fn main() -> Result<(), gprx::GprError> {
    let fitted = Gpr::new(KernelSpec::from(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?)
        .fit(&[0.0, 1.0, 2.0], 3, 1, &[0.0, 1.0, 0.5])?;
    let dir = std::env::temp_dir().join(format!("gprx-readme-save-ja-{}", std::process::id()));
    fitted.save(&dir)?;
    // Built-in kernels and transforms need no registration.
    let loaded = LoadedGpr::load(&dir, &PersistRegistry::new())?;
    assert_eq!(loaded.predict(&[0.5], 1, 1)?, fitted.predict(&[0.5], 1, 1)?);
    if let LoadedGpr::Double(model) = loaded {
        println!("loaded {} points", model.n());
    }
    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}
```

### 速さとメモリ、f32 と混合精度

既定の `with_prefer_speed` では、学習点どうしの距離を一度だけ計算し、カーネルを組み立て直すたびに使い回す。勾配の計算に使う `n × n` 行列も、コレスキー因子とは別に持つ。`with_prefer_memory` にすると、距離を毎回計算し直し、勾配用の行列をコレスキー因子の上に書いて、あとで分解し直す。そのぶん `n × n` 行列 1 つ分のメモリが減る。

既定の `KernelExp::Accurate` は、カーネルの `exp` を標準の `exp` で計算する。`with_math(KernelExp::FastApprox)` にすると多項式で近似し、`f64` での相対誤差は `f64::exp` に対して `2^{-23}` 以内に収まる。精度の既定は `DoublePrecision` で、値の保持も分解も予測も `f64` で行う。`SinglePrecision` は `f32` で値を持って分解し、予測も `f32` で返す。`MixedPrecision` は `f32` で分解したあと `f64` で補正し、予測は `f64` で返す。`Sgpr` と `Svgp` にも `with_precision` と `with_math` がある。精度ごとの違いは [`GpScalar`](https://docs.rs/gprx/latest/gprx/trait.GpScalar.html) の rustdoc で比べている。

```rust
use gprx::kernel::{KernelSpec, RbfKernel};
use gprx::{GaussianLikelihood, Gpr, KernelExp, MixedPrecision, Prediction, SinglePrecision};

fn main() -> Result<(), gprx::GprError> {
    let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    let likelihood = GaussianLikelihood::new(0.1)?;
    let (x, y) = ([0.0, 1.0, 2.0], [0.0, 1.0, 0.5]);

    let small = Gpr::new(kernel.clone(), likelihood.clone())
        .with_prefer_memory()
        .with_math(KernelExp::FastApprox)
        .with_precision::<SinglePrecision>()
        .fit(&x, 3, 1, &y)?;
    let single: Prediction<f32> = small.predict(&[0.5], 1, 1)?;

    let mixed = Gpr::new(kernel, likelihood).with_precision::<MixedPrecision>().fit(&x, 3, 1, &y)?;
    let refined: Prediction<f64> = mixed.predict(&[0.5], 1, 1)?;
    println!("f32 mean = {}, mixed mean = {}", single.mean[0], refined.mean[0]);
    Ok(())
}
```

### 自分のカーネルを書く

自作のカーネルの葉は、`Clone + Debug + Send + Sync + 'static` を満たす型に、すべての `T: KernelScalar` について `KernelTerm<T>` を実装して作る。その型の値を `KernelSpec::custom` に渡せば、カーネルに組み込める。カーネルの葉は二乗距離を `faer` の行列で受け取るので、依存に `faer = "0.24"` を加える。疎なモデルは、下の例で実装しているメソッドのほかに、別の微分のメソッド（`grad_cross` と `hess_cross`、`FreeInducing` ではさらに `*_wrt_sq_dist` の系統）も呼ぶ。これらを実装しないままだと、既定の実装が `GprError::CoordGradientUnsupported` を返す。保存できるようにするには `persist_id` と `persist_state` も実装する（依存に `serde_json = "1"` が要る）。パラメータを持つカーネルの葉に、すべての微分と保存・読み込みまで実装した例は、[`KernelTerm`](https://docs.rs/gprx/latest/gprx/kernel/trait.KernelTerm.html) の rustdoc にある。

```rust
use faer::{MatMut, MatRef};
use gprx::kernel::{ConstantKernel, KernelScalar, KernelSpec, KernelTerm, Triangle};
use gprx::{GaussianLikelihood, Gpr, GprError, Interval};

/// `k(d²) = exp(−d² / 2)`, with no parameter.
#[derive(Clone, Debug)]
struct UnitRbf;

fn none(len: usize) -> Result<(), GprError> {
    match len {
        0 => Ok(()),
        _ => Err(GprError::LengthMismatch { reason: format!("UnitRbf has no parameters, got {len}") }),
    }
}

fn no_param(i: usize) -> Result<(), GprError> {
    Err(GprError::IndexOutOfRange { reason: format!("UnitRbf has no parameter {i}") })
}

impl<T: KernelScalar> KernelTerm<T> for UnitRbf {
    fn num_params(&self) -> usize { 0 }
    fn get_params(&self, out: &mut [f64]) -> Result<(), GprError> { none(out.len()) }
    fn set_params(&mut self, params: &[f64]) -> Result<(), GprError> { none(params.len()) }
    fn bounds_into(&self, out: &mut [Interval]) -> Result<(), GprError> { none(out.len()) }
    fn apply(&self, d2: MatRef<'_, T>, mut out: MatMut<'_, T>, uplo: Triangle) -> Result<(), GprError> {
        for col in 0..d2.ncols() {
            let rows = match uplo {
                Triangle::Lower => col..d2.nrows(),
                Triangle::Upper => 0..col + 1,
                Triangle::Full => 0..d2.nrows(),
            };
            for row in rows {
                out[(row, col)] = (T::from_f64(-0.5) * d2[(row, col)]).exp();
            }
        }
        Ok(())
    }
    fn apply_cross(&self, d2: MatRef<'_, T>, out: MatMut<'_, T>) -> Result<(), GprError> { self.apply(d2, out, Triangle::Full) }
    fn fill_diag(&self, out: &mut [T]) -> Result<(), GprError> { out.fill(T::from_f64(1.0)); Ok(()) }
    fn grad(&self, _: MatRef<'_, T>, _: MatMut<'_, T>, i: usize, _: Triangle) -> Result<(), GprError> { no_param(i) }
    fn hess(&self, _: MatRef<'_, T>, _: MatMut<'_, T>, i: usize, _: usize, _: Triangle) -> Result<(), GprError> { no_param(i) }
    fn hess_points(&self, _: MatRef<'_, T>, _: MatMut<'_, T>, i: usize, _: usize, _: Triangle) -> Result<(), GprError> { no_param(i) }
}

fn main() -> Result<(), GprError> {
    // A signal variance times the custom leaf.
    let kernel = KernelSpec::from(ConstantKernel::new(1.0)?) * KernelSpec::custom(UnitRbf);
    let fitted = Gpr::new(kernel, GaussianLikelihood::new(0.1)?).fit(&[0.0, 1.0, 2.0], 3, 1, &[0.0, 1.0, 0.5])?;
    println!("mean = {}", fitted.predict(&[0.5], 1, 1)?.mean[0]);
    Ok(())
}
```

### エラーを扱う

`GprError` には今後 variant が増えうるので（non-exhaustive）、`match` の最後には残りをまとめて受ける分岐を置く。`fit` が失敗したときはトレーナーがエラーと一緒に戻ってくるので、設定を変えてもう一度試せる。既定の jitter は `Gpr` では 0（`JitterPolicy::fixed(0.0)`）だが、`Sgpr` と `Svgp` は `K_mm` の分解に失敗すると、`1e-8` から大きくしていく jitter を対角に足してやり直す。どのエラーかを判断するときは、表示用の文字列ではなく、variant とそのフィールドで `match` する。`kind` フィールドを持つのは `PersistFailed` と `DistanceSlot` だけだ。variant の一覧は [`GprError`](https://docs.rs/gprx/latest/gprx/enum.GprError.html) の rustdoc にある。

```rust
use gprx::kernel::{KernelSpec, RbfKernel};
use gprx::{GaussianLikelihood, Gpr, GprError, JitterPolicy};

fn main() -> Result<(), GprError> {
    let (x, y) = ([0.0, 1.0, 2.0], [0.0, 1.0, 0.5]);
    let trainer = Gpr::new(KernelSpec::from(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?);
    let fitted = match trainer.fit(&x, 3, 1, &y) {
        Ok(fitted) => fitted,
        // Retry with jitter on the diagonal.
        Err((trainer, GprError::CholeskyFailed { .. })) => {
            trainer.with_jitter_policy(JitterPolicy::fixed(1e-6)?).fit(&x, 3, 1, &y)?
        }
        Err((_, other)) => return Err(other),
    };
    // Two features in the query, one in training.
    match fitted.predict(&[0.5, 1.0], 1, 2) {
        Err(GprError::DimensionMismatch { x_dim, expected_dim }) => {
            println!("query has {x_dim} features, the model {expected_dim}");
        }
        other => println!("{other:?}"),
    }
    Ok(())
}
```

### スレッド数を決める

gprx には `n_jobs` のような設定がなく、プロセス全体で共有する Rayon のプールで動く。スレッド数を変えるには、プロセスを起動する前に `RAYON_NUM_THREADS` を設定するか、最初の fit や predict より前にグローバルプールを作る。後者には依存に `rayon = "1"` が要る。グローバルプールを作れるのは、1 つのプロセスで一度だけである。スレッドを 1 つにすれば、計算はすべて逐次になる。線形代数のバックエンドは、プールのスレッドのうち、問題の大きさに見合った数だけを使う。その数の決め方は[クレートの rustdoc](https://docs.rs/gprx/latest/gprx/) に書いてある。

```rust
use gprx::kernel::{KernelSpec, RbfKernel};
use gprx::{GaussianLikelihood, Gpr};

fn main() -> Result<(), gprx::GprError> {
    // Fails if the global pool was already built in this process.
    if let Err(err) = rayon::ThreadPoolBuilder::new().num_threads(4).build_global() {
        eprintln!("thread pool not changed: {err}");
    }
    let fitted = Gpr::new(KernelSpec::from(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?)
        .fit(&[0.0, 1.0, 2.0], 3, 1, &[0.0, 1.0, 0.5])?;
    println!("threads = {}, mean = {}", rayon::current_num_threads(), fitted.predict(&[0.5], 1, 1)?.mean[0]);
    Ok(())
}
```

## API の詳細

公開しているすべての項目について、使用例つきの API リファレンスが [docs.rs](https://docs.rs/gprx) にある（英語）。入口は次の 4 つである。

- [`gprx`](https://docs.rs/gprx/latest/gprx/): モデル（`Gpr`、`Sgpr`、`Svgp` とそれぞれの学習済み版、オンライン版の `OnlineGpr` と `OnlineSgpr`）、`GaussianLikelihood`、最適化器、精度、`GprError`
- [`gprx::kernel`](https://docs.rs/gprx/latest/gprx/kernel/): カーネルの葉、`KernelSpec`、二乗距離の受け渡し、`KernelTerm`
- [`gprx::transform`](https://docs.rs/gprx/latest/gprx/transform/): 入力と目的変数の変換
- [`gprx::persist`](https://docs.rs/gprx/latest/gprx/persist/): ローダーと `PersistRegistry`

クレートの中身を知りたいときは、次の文書が役に立つ。[`docs/architecture.ja.md`](https://github.com/YUKIKEDA/gprx/blob/main/docs/architecture.ja.md) にはモジュールごとの責務と import の向き、[`docs/persist-format.ja.md`](https://github.com/YUKIKEDA/gprx/blob/main/docs/persist-format.ja.md) には `save` が書き出す内容、[`docs/design.ja.md`](https://github.com/YUKIKEDA/gprx/blob/main/docs/design.ja.md) にはアルゴリズムと設計の判断をまとめてある。

## 比較

精度、学習時間、メモリ使用量を他のライブラリと比べた結果は、[ライブラリの比較](https://github.com/YUKIKEDA/gprx/blob/main/docs/comparison.ja.md)にまとめてある。英語版は [comparison.md](https://github.com/YUKIKEDA/gprx/blob/main/docs/comparison.md) である。

## ライセンス

次のいずれかのライセンスを選んで利用できる。

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))
