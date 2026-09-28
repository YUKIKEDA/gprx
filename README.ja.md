[English](README.md) | 日本語

# gprx

Rust の Exact ガウス過程回帰。`Gpr` は未学習のトレーナー。`Gpr::fit` はそれを消費し、負の対数周辺尤度を argmin の L-BFGS で最小化して `FittedGpr` を返す。このクレートは crates.io に**公開しない**（`Cargo.toml` の `publish = false`）。

## 状態

手元の **0.1.0** 品質: `Gpr` / `FittedGpr`、カーネル、`fit` / `predict` / `predict_into` / leave-one-out、英語の rustdoc、`examples/`。依存は git か path。crates.io ではない。

設計: [`docs/design.ja.md`](docs/design.ja.md)。タスク: [`docs/roadmap.md`](docs/roadmap.md)。エージェント向け: [`AGENTS.md`](AGENTS.md)。他ライブラリとの壁時計とピーク RSS: [`compare/perf/`](compare/perf/)（P2B-16 Exact は `just perf`。P4-12 Sparse は `just perf-sparse`。P4-14 Sparse オンラインは `just perf-sparse-online`。criterion ではない）。

## 例

`X` は列優先（`n` 点 × `d` 特徴: 特徴 0 の全行、次に特徴 1、…）。

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

同じプログラム: `cargo run --example fit_predict`。`fit` の `?` はトレーナーを落とす（`From<(Gpr, GprError)> for GprError`）。エラーだけ欲しいときは `.map_err(|(_, e)| e)`。同じトレーナーでやり直すときは `Err((gpr, err))` を照合する。

既定の `predict` の分散は観測（`latent + σn²`）。潜在関数は `predict_with` と `VarianceKind::Latent`。学習後の `loo_predict` は、全学習点での GPML leave-one-out。`predict` はクエリバッファを確保する。`predict_into` はウォームアップのあとそれを再利用する。`FittedGpr::refit` は保存した学習データで因子を作り直すか、最適化し直す。

変換の既定は恒等。平均関数が零のときは `fit` の前に `with_target_transform(StandardizeTarget::new())`。特徴は `MinMaxInput`（既定 `[0, 1]`）。観測ノイズは `GaussianLikelihood`。`WhiteKernel` はオプトインの合成。両方を大きい値で足すとノイズを二重に数える。

`Gpr<Fixed>::factor`（`with_optimizer(Fixed)` のあと）は、トレーナーに既にあるカーネルと尤度の `θ` で因子を作る。L-BFGS のノブは `Lbfgs`（`with_max_iterations`、`with_tolerance`、`with_history_size`、`with_restarts`）。Nonlinear CG と Nelder–Mead は `history_size` 以外の最初の 3 つを共有する（`NonlinearCg`、`NelderMead`）。Newton は `Newton`（`with_max_iterations`、`with_tolerance`、`with_restarts`、`with_gamma`）。自作の Fast Simulated Annealing は `FastSimulatedAnnealing`（`with_max_iterations`、`with_restarts`、`with_initial_temperature`、`with_cooling_rate`、`with_seed`、`with_boundary`）。

## ライセンス

次のいずれか。

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))
