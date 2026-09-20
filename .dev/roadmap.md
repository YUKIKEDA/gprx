# gprx 実装ロードマップ

進め方は [AGENTS.md](../AGENTS.md) と `.cursor/rules/`。設計の詳細は `.dev/gprx-design.md`。

**今やること: P2B-16（他ライブラリとの時間・RSS 比較）。** Phase 2 は P2-9 で閉じた。Phase 3 の前に公開骨格（2b）を載せる。比較の基準は [`.dev/bench-log.md`](bench-log.md) の `phase-2`。

進め方の正本は `.cursor/rules/workflow.mdc`: Grill（必要なとき）→ Issue 作成 → Grill で DoD を確定して Issue を更新 → 作業 → PR → 人間レビュー → マージ。DoD をエージェントが先に書かない。1 Issue = 1 PR。ブランチは `type/{issue}-{slug}`（例: `chore/1-crate-bootstrap`）。

## GitHub Issue 対応

| Roadmap | Issue                                             | Roadmap | Issue                                             | Roadmap | Issue                                             |
| ------- | ------------------------------------------------- | ------- | ------------------------------------------------- | ------- | ------------------------------------------------- |
| M0-1    | [#1](https://github.com/YUKIKEDA/gprx/issues/1)   | P1A-8   | [#10](https://github.com/YUKIKEDA/gprx/issues/10) | P2-5    | [#29](https://github.com/YUKIKEDA/gprx/issues/29) |
| M0-2    | [#2](https://github.com/YUKIKEDA/gprx/issues/2)   | P1A-9   | [#11](https://github.com/YUKIKEDA/gprx/issues/11) | P3-1    | [#30](https://github.com/YUKIKEDA/gprx/issues/30) |
| P1A-1   | [#3](https://github.com/YUKIKEDA/gprx/issues/3)   | P1A-10  | [#12](https://github.com/YUKIKEDA/gprx/issues/12) | P3-2    | [#31](https://github.com/YUKIKEDA/gprx/issues/31) |
| P1A-2   | [#4](https://github.com/YUKIKEDA/gprx/issues/4)   | P1A-11  | [#13](https://github.com/YUKIKEDA/gprx/issues/13) | P3-3    | [#32](https://github.com/YUKIKEDA/gprx/issues/32) |
| P1A-3   | [#5](https://github.com/YUKIKEDA/gprx/issues/5)   | P1A-12  | [#14](https://github.com/YUKIKEDA/gprx/issues/14) | P3-4    | [#33](https://github.com/YUKIKEDA/gprx/issues/33) |
| P1A-4   | [#6](https://github.com/YUKIKEDA/gprx/issues/6)   | P1A-13  | [#15](https://github.com/YUKIKEDA/gprx/issues/15) | P3-5    | [#34](https://github.com/YUKIKEDA/gprx/issues/34) |
| P1A-5   | [#7](https://github.com/YUKIKEDA/gprx/issues/7)   | P1A-14  | [#16](https://github.com/YUKIKEDA/gprx/issues/16) | P4-1    | [#35](https://github.com/YUKIKEDA/gprx/issues/35) |
| P1A-6   | [#8](https://github.com/YUKIKEDA/gprx/issues/8)   | P1A-15  | [#17](https://github.com/YUKIKEDA/gprx/issues/17) | P4-2    | [#36](https://github.com/YUKIKEDA/gprx/issues/36) |
| P1A-7   | [#9](https://github.com/YUKIKEDA/gprx/issues/9)   | P1A-16  | [#18](https://github.com/YUKIKEDA/gprx/issues/18) | P4-3    | [#37](https://github.com/YUKIKEDA/gprx/issues/37) |
| P1A-18  | [#44](https://github.com/YUKIKEDA/gprx/issues/44) | P1A-17  | [#19](https://github.com/YUKIKEDA/gprx/issues/19) | P4-4    | [#38](https://github.com/YUKIKEDA/gprx/issues/38) |
| P1A-19  | [#45](https://github.com/YUKIKEDA/gprx/issues/45) | P1B-1   | [#20](https://github.com/YUKIKEDA/gprx/issues/20) | P5-1    | [#39](https://github.com/YUKIKEDA/gprx/issues/39) |
| P1A-20  | [#54](https://github.com/YUKIKEDA/gprx/issues/54) | P1B-2   | [#21](https://github.com/YUKIKEDA/gprx/issues/21) | P5-2    | [#40](https://github.com/YUKIKEDA/gprx/issues/40) |
| P2-6    | [#60](https://github.com/YUKIKEDA/gprx/issues/60) | P1B-3   | [#22](https://github.com/YUKIKEDA/gprx/issues/22) | P2B-18  | [#110](https://github.com/YUKIKEDA/gprx/issues/110) |
| P1B-6   | [#80](https://github.com/YUKIKEDA/gprx/issues/80) | P1B-4   | [#23](https://github.com/YUKIKEDA/gprx/issues/23) | P5-4    | [#42](https://github.com/YUKIKEDA/gprx/issues/42) |
| P1B-7   | [#83](https://github.com/YUKIKEDA/gprx/issues/83) | P1B-5   | [#24](https://github.com/YUKIKEDA/gprx/issues/24) | P5-5    | [#43](https://github.com/YUKIKEDA/gprx/issues/43) |
| P2-7    | [#88](https://github.com/YUKIKEDA/gprx/issues/88) | P2-1    | [#25](https://github.com/YUKIKEDA/gprx/issues/25) | P2B-19  | [#111](https://github.com/YUKIKEDA/gprx/issues/111) |
| P2-8    | [#98](https://github.com/YUKIKEDA/gprx/issues/98) | P2-2    | [#26](https://github.com/YUKIKEDA/gprx/issues/26) | P2B-14  | [#63](https://github.com/YUKIKEDA/gprx/issues/63) |
| P2-9    | [#97](https://github.com/YUKIKEDA/gprx/issues/97) | P2-3    | [#27](https://github.com/YUKIKEDA/gprx/issues/27) | P2B-15  | [#106](https://github.com/YUKIKEDA/gprx/issues/106) |
| P2B-17  | [#109](https://github.com/YUKIKEDA/gprx/issues/109) | P2-4    | [#28](https://github.com/YUKIKEDA/gprx/issues/28) | P2B-16  | [#103](https://github.com/YUKIKEDA/gprx/issues/103) |
| P2B-1   | [#108](https://github.com/YUKIKEDA/gprx/issues/108) | P2B-2   | [#114](https://github.com/YUKIKEDA/gprx/issues/114) | P2B-20  | [#116](https://github.com/YUKIKEDA/gprx/issues/116) |
| P2B-3   | [#117](https://github.com/YUKIKEDA/gprx/issues/117) | P2B-4   | [#119](https://github.com/YUKIKEDA/gprx/issues/119) | P2B-5   | [#121](https://github.com/YUKIKEDA/gprx/issues/121) |
| P2B-6   | [#123](https://github.com/YUKIKEDA/gprx/issues/123) | P2B-10  | [#125](https://github.com/YUKIKEDA/gprx/issues/125) | P2B-7   | [#127](https://github.com/YUKIKEDA/gprx/issues/127) |
| P2B-8   | [#129](https://github.com/YUKIKEDA/gprx/issues/129) | P2B-9   | [#131](https://github.com/YUKIKEDA/gprx/issues/131) | P2B-11  | [#133](https://github.com/YUKIKEDA/gprx/issues/133) |
| P2B-12  | [#135](https://github.com/YUKIKEDA/gprx/issues/135) | P2B-13  | [#137](https://github.com/YUKIKEDA/gprx/issues/137) | P2B-21  | [#142](https://github.com/YUKIKEDA/gprx/issues/142) |
| P2B-22  | [#143](https://github.com/YUKIKEDA/gprx/issues/143) | P2B-23  | [#148](https://github.com/YUKIKEDA/gprx/issues/148) |         |                                                   |

## マイルストーン

| ID  | 名前                  | 目的                           | 完了条件                                                                                |
| --- | --------------------- | ------------------------------ | --------------------------------------------------------------------------------------- |
| M0  | Spike                 | 箱と faer 0.24 を確認する      | `just lint` / `just test` が通る。2×2 と 5×5 で Cholesky 往復が一致する。GPR はまだ無い |
| 1a  | 固定ハイパラ Exact GPR | 正しい推論と勾配               | 解析解、sklearn JSON、criterion `phase-1a`、確保 ratchet、Phase 1 カーネル              |
| 1b  | Optimizer と 0.1 API  | ハイパラ最適化と使えるクレート | L-BFGS で lengthscale / ノイズ回収。README / rustdoc / 例。baseline `phase-1b`          |
| 2   | 高速化                | Phase 1 を壊さず速くする       | ボトルネック順に最適化。キャッシュ・Rayon・SIMD。P2-8 typestate。P2-9 で `phase-2`、alloc 0、README / rustdoc / 例 |
| 2b  | Exact GPR 公開骨格    | §1 の拡張点を公開面に載せる    | `Gpr<O>` / `Gpr<Fixed>`。argmin と自作 Optimizer は同じ型スロット。変換の fitted 型、距離キャッシュは距離経路だけ。カスタムカーネル、jitter、学習済みの読み書き、予測共分散は別経路、Pipeline と列ごと前処理。Product の points 勾配と Dist+Points 合成。ファイル persist、カスタム Optimizer 例、他ライブラリ比較（P2B-14…16。DoD は Grill 後）。NLML ヘッセ impl（P2B-17。DoD は Grill 後）。`IncrementalRecompute`（P2B-18）と fit 中の `L`/`W` 共用（P2B-19。DoD は Grill 後）。モジュール分割（P2B-20）。P3-1 より前 |
| 3   | オンライン学習        | 点の追加削除                   | 任意 delete を含む incremental == full refit。プロパティテスト                          |
| 4   | Sparse GPR             | 大きい n                       | VFE または FITC の一方。初期は Z 固定。あとから Z 最適化と Sparse オンライン |
| 5   | 高度な最適化          | 混合精度など                   | predict 中心の MixedPrecision。失敗時は f64 フォールバック |

## 依存

```
M0 → 1a → 1b → 2 → 2b → 3
                     → 4 → P4-5（Z 最適化）→ P4-7（Sparse オンライン、3 のあと）
                2 の計測のあと → 5
```

3 は 2b のあと。4 の初期（P4-1…4）は 1b のあと並行してよい（自作 Optimizer を使うなら P2B-9 のあと）。Z 最適化は P4-5。Sparse オンラインは P4-4 と Phase 3。5 は `phase-2` が無いと「速くなった」と言えない。

計測は 1a から始める（P1A-18 / P1A-19）。Phase 2 でハーネスを新しく作らない。

---

## M0 — Spike

設計 §3。GPR は書かない。

| ID   | 種別  | タイトル                                                                                   | 依存 | DoD                                                                                                                                  |
| ---- | ----- | ------------------------------------------------------------------------------------------ | ---- | ------------------------------------------------------------------------------------------------------------------------------------ |
| M0-1 | Task  | クレート初期化（Cargo.toml, MIT OR Apache-2.0, justfile, CI yaml, gitignore, README stub） | —    | 規約のディレクトリ構成に従う。ルートの箱と `src/lib.rs`。`clippy.toml` と `[lints.clippy]` は `rust.mdc` どおり。`just lint` が通る。CI yaml はあるが制限中は実行されなくてよい |
| M0-2 | Spike | faer 0.24 `cholesky_in_place` 往復                                                         | M0-1 | SPD の 2×2 / 5×5 で `K ≈ LLᵀ`、`solve` が解析解と一致。`MemStack` と `LltRegularization` の呼び方がテストに残る                      |

---

## Phase 1a — 固定ハイパラ

設計 §4.0, §5, §6.2, §6.3, §7, §10, §12。ハイパラは与える。最適化しない。

経路を通す順（これより前に他カーネルを増やさない）:

| ID     | 種別 | タイトル                                                                              | 依存                | DoD                                                                                                      |
| ------ | ---- | ------------------------------------------------------------------------------------- | ------------------- | -------------------------------------------------------------------------------------------------------- |
| P1A-1  | Task | `GprError` と `DoublePrecision` のみの型                                               | M0-2                | 設計 §10 の主要バリアント。ライブラリ経路に `unwrap` なし                                                |
| P1A-2  | Task | `Workspace`（f64, `k_matrix` / `w_matrix` / `dist_cache` / `exp_buf` / faer scratch） | P1A-1               | fit 開始時に1回確保。テストでサイズが分かる                                                              |
| P1A-3  | Feat | `GaussianLikelihood`（`θ=log(σn²)`, `∂K/∂θ=σn² I`）                                   | P1A-1               | `add_noise_diag` / `noise_grad_diag` の単体テスト                                                        |
| P1A-4  | Feat | `TargetTransform` と X の `Transform`（Identity / Standardize）                       | P1A-1               | 平均・分散の逆変換。§12-8                                                                                |
| P1A-5  | Feat | RBF カーネル（`apply`/`grad`, `uplo=Lower`）                                          | P1A-1               | 対称性、対角、既知値、数値微分。Lower と Full の一致                                                     |
| P1A-6  | Feat | `KernelSpec` / `CompiledKernel`（RBF + Sum/Product + param flatten）                  | P1A-5               | `get/set_params` がリーフに届く。組み込みは enum                                                         |
| P1A-7  | Feat | `Gpr`: `A=K+σn²I`、LLT、`α`                                                             | P1A-2, P1A-3, P1A-6 | 小規模で `A α = y`。失敗時 `fitted=false`。P1A-18 から crate 内で同じ経路をベンチできる                  |
| P1A-8  | Feat | `predict`（mean, `VarianceKind::{Latent,Observation}`）                               | P1A-7, P1A-4        | 観測分散 = 潜在 + σn²（逆変換後）。`benches/exact.rs` に `predict_100` を足す                            |
| P1A-9  | Feat | 負の MLL                                                                              | P1A-7               | `log det K = 2Σ log L_ii`。既知の小問題と一致                                                            |
| P1A-10 | Feat | `W=ααᵀ-K⁻¹` による勾配                                                                | P1A-9               | `value_and_gradient_into` が L/α/W を共有。数値微分一致。bench に `mll_and_grad` を足す                  |
| P1A-11 | Task | 解析解の統合テスト（n=2/3 RBF）                                                       | P1A-8, P1A-10       | mean / variance / LML / 勾配が一つのテスト列で固定                                                       |
| P1A-12 | Task | sklearn golden（`compare/` + `just gen-goldens` + JSON）                              | P1A-11              | RBF 固定ハイパラの JSON をコミット。Rust がそれを読む。`cargo test` は Python 不要                       |
| P1A-18 | Task | criterion ハーネス（`benches/exact.rs`, `just bench`）                                | P1A-7               | 固定問題 n=256,d=8,seed=0。`kernel_rbf` と `cholesky_alpha`。空の benches を M0 では置かない             |
| P1A-19 | Task | 確保 ratchet（`tests/alloc.rs`）                                                      | P1A-7, P1A-2        | Workspace 確保後の回数を数え、上限定数をテストに置く。1a では 0 を要求しない。Issue なしに上限を上げない |

1a の残りカーネル（RBF 経路のあと）:

| ID     | 種別 | タイトル                  | 依存       | DoD                                                                     |
| ------ | ---- | ------------------------- | ---------- | ----------------------------------------------------------------------- |
| P1A-20 | Feat | ARD lengthscale（まず RBF） | P1A-5      | `θ_d=log(ℓ_d)` を葉の共通の口にする。RBF で等方と一致。数値微分。Lower と Full。次元ごとの差が要る |
| P1A-13 | Feat | Constant, Linear, White   | P1A-6      | 各リーフの対称性・勾配。White と Likelihood の二重計上を rustdoc で禁止 |
| P1A-14 | Feat | Matern ν=1/2, 3/2, 5/2    | P1A-6, P1A-20 | 等方と ARD。ν ごとの既知値または数値微分                                      |
| P1A-15 | Feat | Periodic                  | P1A-6      | 周期距離は二乗ユークリッドではない（§5.2）。lengthscale はスカラーのまま     |
| P1A-16 | Feat | Rational Quadratic        | P1A-6, P1A-20 | 等方と ARD。数値微分一致                                                    |
| P1A-17 | Task | 合成カーネルと追加 golden | P1A-12〜16 | Sum/Product の flatten と sklearn または解析の照合                      |

**1a 完了:** P1A-1…20 がマージ済み。`just test` が Python なしで緑。`.dev/bench-log.md` に `phase-1a` がある。

---

## Phase 1b — Optimizer と 0.1

設計 §9。

| ID    | 種別 | タイトル                          | 依存          | DoD                                                                                  |
| ----- | ---- | --------------------------------- | ------------- | ------------------------------------------------------------------------------------ |
| P1B-1 | Feat | `Objective` と `GprObjective`      | P1A-10        | `set_params` が kernel+likelihood の連結配列                                         |
| P1B-2 | Feat | argmin L-BFGS アダプタ            | P1B-1         | `value_and_gradient_into` を1回の評価で使う                                          |
| P1B-3 | Feat | `Gpr::fit` が最適化する            | P1B-2, P1A-8  | 未学習 `predict` は `NotFitted`。成功後は L と α を保持。bench に `fit_lbfgs` を足す |
| P1B-4 | Task | パラメータ回収テスト              | P1B-3         | 合成データで lengthscale とノイズが真値の近くに戻る。LML が初期より下がる            |
| P1B-6 | Task | fit 込み sklearn golden（Forrester / ARD） | P1B-4, P1A-12, P1A-20 | 1 次元 Forrester と 2 次元重み付き球関数（ARD）を sklearn が L-BFGS した JSON をコミット。`Gpr::fit` が NLML・予測で緩い許容。`θ` は相対。1a の 1e-8 とは分ける。`cargo test` は Python 不要 |
| P1B-7 | Feat | leave-one-out 予測                | P1B-6, P1A-8  | GPML の `μ_i = y_i - α_i / Q_ii`。観測/潜在。n=2 解析と n=3 実 LOO。fit golden に LOO を足し sklearn の `θ` で照合。`cargo test` は Python 不要 |
| P1B-5 | Docs | README, rustdoc, `examples/`      | P1B-7, P1A-17 | 英語 rustdoc。最短例で fit→predict                                                   |

**1b 完了 = Phase 1 完了 = 0.1.0 相当。** crates.io には出さない。`.dev/bench-log.md` に `phase-1b` がある。

---

## Phase 2 — 高速化

設計 §5.2, §7, §8, §15。1a からある criterion / alloc を使う。新しいハーネスは作らない。1b の数値テストを回帰として残す。Phase 2 の出口は P2-9。

| ID   | 種別  | タイトル                                                  | 依存         | DoD                                                                          |
| ---- | ----- | --------------------------------------------------------- | ------------ | ---------------------------------------------------------------------------- |
| P2-1 | Task  | `phase-1b` を読み、ボトルネック順を決める                 | 1b           | `.dev/` に短い順序（kernel vs Cholesky vs その他）。目標比はここで置いてよい |
| P2-2 | Feat  | 距離キャッシュ                                            | P2-1         | `Never` / `Always`。等方 RBF の数値が 1b と一致。bench が改善または同等      |
| P2-7 | Feat  | ARD 距離キャッシュ                                        | P2-2         | `Never` / `Always` が `(Δx_d)²` に効く。`mll_and_grad_ard` / `fit_lbfgs_ard` で Always vs Never（改善または同等、bench-log、メモリ）。Workspace は n と d。fit 開始時に確保、等方/Never は空。必須数値は RBF ARD。埋めと RBF ARD apply/grad は Rayon + `wide::f64x4`。`Auto` は P5-5 |
| P2-3 | Feat  | Rayon でカーネル構築。`thread_scratch` を並列前に切り離す | P2-1         | 1b と数値一致。`kernel_rbf` が速くなることを bench で示す                    |
| P2-4 | Task  | 確保 ratchet をホットパス 0 まで下げる                    | P2-3, P1A-19 | `tests/alloc.rs` の上限が 0。ユーザーカーネル除く                            |
| P2-5 | Feat  | 等方 RBF と距離に SIMD                                    | P2-1, P2-3   | `wide::f64x4`。`kernel_rbf` / `predict_100` が Rayon のみより速い。数値は 1b と一致。可否はカーネル経路で判断し、`mll_and_grad` の勾配項だけを分母にしない |
| P2-6 | Spike | NLML 定数項 `(n/2) log(2π)` の速度寄与                    | P2-1, P1A-10 | `mll_and_grad`（あれば `fit_lbfgs`）を定数あり/なしで測る。差がノイズなら一本のまま。結果を `.dev/bench-log.md` に残す。この行では API を分けない |
| P2-8 | Feat  | `Gpr` / `FittedGpr` の typestate                          | P2-6         | `fit(self) → FittedGpr`。失敗は `(Gpr, GprError)`。`predict(&self)` と `predict_into(&mut self)`。`refit` は学習済み型。`Workspace` は fit、`QueryWorkspace` は `FittedGpr`（同じ struct に詰め込まない）。sklearn JSON は数値照合のみ。文書・baseline・alloc の締めは P2-9。Issue は実装時 |
| P2-9 | Task  | Phase 2 締め                                              | P2-8         | 名前付き `phase-2` を取り、機械名と数値を `.dev/bench-log.md` に残す。等方は `phase-1b` と比較。ARD は Always vs Never（等方とは比べない）。`just test`（解析解・sklearn JSON・L-BFGS 回収）が `FittedGpr` 経路で通る。`tests/alloc.rs` 上限 0 を `FittedGpr::predict_into` で再確認（ユーザーカーネル除く）。README / rustdoc / `examples/` を `Gpr` + `FittedGpr`。Issue は実装時 |

---

## Phase 2b — Exact GPR 公開骨格

設計 §1, §4.0, §5.1, §5.4, §5.5, §6, §9。P2-9 のあと、P3-1 の前。組み込みの fit→predict は 1b / 2 で通っている。欠けているのは設計が公開すると書いた拡張点（`Gpr<O>` のソルバ差し替え、カスタムカーネル、jitter、学習済みの読み書き）、予測共分散の別経路、Pipeline と列ごと前処理、公開の `*` と Dist+Points 合成が実行時エラーで落ちる穴、プロセスをまたぐ persist、自作 Optimizer の例、他ライブラリとの時間・RSS、逐次更新の本体（P2B-18）と fit 中の `L`/`W` 共用（P2B-19）。各行は Grill のあと Issue で DoD を確定してから作業する（`.cursor/rules/workflow.mdc`）。設定の排他は型（`.cursor/rules/types.mdc`）。

前処理のユーザー実装（`Transform` / `TargetTransform` + `with_*`）は P1A-4 で載済み。学習前後は型で分ける（P2B-10）。複数マップの直列は `Pipeline` / `TargetPipeline`。列ごとの指定は `ColumnwiseInput`。`input.rs` / `target.rs` は葉に分けない。`compiled` と `FittedGpr` の分割は P2B-20（[#116](https://github.com/YUKIKEDA/gprx/issues/116)）。

| ID    | 種別 | タイトル                                      | 依存   | DoD                                                                                                                                 |
| ----- | ---- | --------------------------------------------- | ------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| P2B-1 | Feat | 最適化ノブと `Gpr<Fixed>`                     | P2-9   | 現行の `FitOptions { optimize: bool }` / `FIXED` / `if options.optimize` / rustdoc「When false…」/ README の skip 文を型に置き換える。`Gpr<Lbfgs, FullRecompute>::fit` と `Gpr<Fixed>::factor`。`Gpr<Fixed>` に `S` は無い。`optimize: bool` は置かない。`with_optimizer` が `O` を差し替える。`minimize` は generic（hot path に `dyn Objective` を置かない）。公開トレイト: `Objective` ⊂ `Differentiable` ⊂ `TwiceDifferentiable`。`IncrementalObjective`（`value_with_changes(params, indices: &[usize])`）。`RecomputeStrategy` マーカーと `FullRecompute`。`UsesChangeIndices`。`GprObjective` の impl は value+grad まで。ヘッセ impl は P2B-17。Incremental impl は P2B-18。実行時の NotImplemented は置かない。`with_recompute_strategy` は `Gpr<O, S>` にだけある。`IncrementalRecompute` への差し替えは `O: UsesChangeIndices` のときだけ。L-BFGS / NCG / Nelder–Mead は impl しない。`FittedGpr<O, S>` は `PhantomData<S>`。`refit` は同じ `O` と `S`。predict は `S` を読まない。`ChangeSet` 構造体は置かない。θ の数値差分で index を推測しない。共有ノブは最適化側だけ: `max_iterations`（既定 100）、`tolerance`、`n_restarts`（`with_restarts(n, seed)`、`n ≥ 1`）。`history_size`（既定 10）は `Lbfgs` 上。`factor` に最適化ノブは無い。境界は葉と尤度の `Interval` + `BoundedParam`（開区間）。ユーザー単位。sklearn 相当の有限既定。argmin L-BFGS は unconstrained + logit。範囲外は型で表し、`InvalidHyperparameter` で設定排他しない。ノブが `fit` / `refit` に効くテスト。`FitOptions::solver` は置かない。gprx は準ニュートンを自前実装しない |
| P2B-2 | Feat | argmin ソルバの選択                           | P2B-1  | `Gpr<Lbfgs>`（既定）/ `Gpr<NonlinearCg>` / `Gpr<NelderMead>`。`with_optimizer` は P2B-1 の口。gprx は準ニュートンを自前実装しない。NCG: 勾配経路 + 回収ゴールデン。Nelder–Mead: `value` のみ + NLML 低下 + sklearn / scipy `Nelder-Mead` JSON（`compare/goldens/`。L-BFGS と混ぜない） |
| P2B-3 | Feat | `KernelTerm` と `KernelSpec::Custom`          | P2-9   | 設計 §5.1。ユーザー葉が Sum/Product に載る。`apply` / `grad` の数値微分または既知値。`tests/alloc.rs` はユーザーカーネル除く（既存） |
| P2B-4 | Feat | `JitterPolicy`                                | P2-9   | 設計 §4.0。`Fixed` / `Adaptive`。分解失敗時だけ jitter。いまの `jitter = 0` 固定をやめる。`CholeskyFailed.jitter` に使用値。観測ノイズとは混ぜない |
| P2B-5 | Feat | 学習済みの読み書きと Clone                    | P2-8   | `FittedGpr::set_params` のあと `Gpr<Fixed>` / `factor` で再分解。訓練 `X` / `y` の参照。`Gpr` / `FittedGpr` が Clone（`Transform` / `TargetTransform` に clone）。`kernel()` は `&` のまま、書き換えは `set_params`。公開 `FittedGpr` の `compiled` / `workspace` / `X` / `y` / `α` を `Option` にしない。欠けるときに `EmptyInput` を返さない |
| P2B-6 | Feat | 任意の予測共分散と posterior sample           | P2-9   | 既定 `predict` は対角のまま（共分散フィールドを持たない）。クエリ間共分散は別メソッド。対角は既存 `predict` と一致。共分散経路から `sample`（seed 付き）。フラグで「計算しない」を表さない |
| P2B-7 | Feat | 変換 Pipeline                                 | P2B-10 | 複数マップの直列（例: MinMax のあと Standardize）。`X` と `y` それぞれ。1 段だけのいまの `with_*` は残す |
| P2B-8 | Feat | 入力変換を列ごとに指定                        | P2B-7  | 列 `d` ごとに Identity / Standardize / MinMax / 自前 `Transform`。一様な列は MinMax、正規に近い列は Standardize、という使い分け。長さが `d` でないときはエラー |
| P2B-9 | Feat | 自作 `Optimizer` の差し替え                   | P2B-1  | トレイトと差し込み口は P2B-1。この行は自作 `O` のダミーが `minimize` されるテスト。`Gpr<O>::with_optimizer` は P2B-1 の口。Phase 3 の `refit` は学習済み型の `O` と同じ口。`solver` と custom を並べない |
| P2B-10 | Feat | 変換の fitted を型にする                      | P1A-4  | `StandardizeTarget` / `MinMax*` の `fitted: bool` と `GprError::NotFitted` をやめる。`fit(self)` が学習済み型を返す。`transform` / `apply` は学習済みにだけある。`error.rs` の `fitted: bool` 例を消す |
| P2B-11 | Feat | 距離キャッシュを距離経路専用にする            | P2-9   | rustdoc「Linear ignores this setting」をやめる。`Linear` / `Constant` / `White`（距離を使わない spec）の trainer に `DistanceCachePolicy` を持たせない。`with_distance_cache_policy` は距離モードの経路にだけ存在する |
| P2B-12 | Feat | Product の points 勾配                        | P2-9   | 公開の `KernelSpec *` が points 葉（Linear / ARD）でも `grad` と fit の MLL+grad まで通る。現行 `grad_points` の `UnsupportedKernelOperation`（dedicated scratch）を消す。数値微分または既知値。Dist Product の既存 `grad` は壊さない |
| P2B-13 | Feat | Dist と Points の Sum/Product                 | P2B-12 | `RBF + Linear` など Dist 葉と Points 葉の合成を評価する。`coord_mode` で混ぜを `UnsupportedKernelOperation` しない。型で混ぜ不可にもしない。葉は従来どおり Dist は距離、Points は座標。解析または sklearn golden（L-BFGS と混ぜない） |
| P2B-14 | Feat | 学習済みモデルの保存・読み込み                | P2-8   | 学習済みだけを保存する（未学習の `Gpr` は置かない）。ディレクトリ一つ（`config.json` + `model.safetensors`）。`save` と `save_with_factor`（bool ではない）。`load` は予測用で中身は `FittedGpr<Fixed>`、距離 / Points は公開 enum。再学習は `with_optimizer` → `refit`（ファイルにソルバは書かない）。テンソルは safetensors。`L` があるとき mmap を保持。元の `X` / `y` を書き、変換は load で `apply`。`L` は正方 `n×n` 列優先、下側が正本。config は `format_version`（この行は `1`、未知は拒否）、jitter、距離経路だけキャッシュ方針。組み込みは閉じたタグ。Custom / 自前変換は `persist_id` + JSON。`gprx.` は予約。レジストリは明示登録 |
| P2B-15 | Feat | カスタム Optimizer の使用例                   | P2B-9  | 公開 `FastSimulatedAnnealing` と `BoundaryPolicy` を `src/optimizer/fsa.rs` に置き、`Lbfgs` と同じ段で再エクスポートする。`examples/` 専用にはしない。Cauchy / Metropolis、component-wise、Ingber 冷却。ノブは型の上（共有 `with_max_iterations` / `with_restarts`、FSA 専用 `with_initial_temperature` / `with_cooling_rate` / `with_seed` / `with_boundary`）。境界は `HasBounds`。既定反復 100。未使用の `tolerance` / `FsaConfig` / 独自 `Bound` は置かない。探索は `minimize` の log-θ。開区間 `(ln lo, ln hi)`。`BoundaryPolicy::{Clamp, Periodic}`、既定 Clamp。政策ごとの追加ノブなし。乱数は `rand` の `SmallRng`（公開は `u64` seed）。FSA・`sample`・リスタートと bench／テストの `y` を同じ生成器に寄せる。`sample` の rustdoc から SplitMix64 を外す。rustdoc が正本（Szu & Hartley 1987 / Ingber 1989 と Example）。README のソルバ段落に1行。新しい example バイナリは置かない。試験: 固定 seed の 1 次元受理／Clamp／Periodic。`Gpr<FastSimulatedAnnealing>` の `fit` が `FittedGpr` を返し同じ seed で NLML が初期より下がる。Rosenbrock 2D は座標が両軸 `(-0.5, 2.5)` に入る（既知最小一致は求めない） |
| P2B-16 | Spike | 他ライブラリとの時間・RSS 比較               | P2-9   | sklearn / libgp / friedrich だけ。等方 Forrester と ARD 球、`n = 256 / 1024 / 4096`（ARD は 16×16 / 32×32 / 64×64）。各セルで同じ初期 θ の factor（最適化なし）・その θ で joint MLL+grad 10 回・predict 100・ピーク RSS。ソルバは回さない。回数差がある時間は速度差と書かない。走った全セルにゲート。sklearn は factor・eval・RSS がどれも小さい（5% 以内は判定不能）。libgp は ±10%。friedrich は factor 時間か RSS の一方。同じ問題を書けないセルは N/A。libgp は C++ 本体。スクリプトは `compare/perf/`（gprx も同じ手順）。記録と合否は `.dev/bench-log.md`（機械名）。手動、CI なし。criterion は合否に使わない。負けたセルは同じ変更で改善行を足してこの Spike を閉じる（P2B-21 / P2B-22）。README から bench-log へ 1 行 |
| P2B-17 | Feat | `GprObjective` の NLML ヘッセ                 | P2B-1  | `KernelTerm` に `hess` / `hess_points`（`(i, j)` 1 組。Custom・Sum/Product 必須。数値微分フォールバックなし）。解析 NLML ヘッセ。`FittedGpr::hessian_into` 公開、`GprObjective: TwiceDifferentiable` は転送。公開 `Newton` は argmin `Newton`。`H⁻¹` は faer の私有型。logit は L-BFGS と同じ、`H_z` は解析連鎖。ノブは共有 3 つ + `with_gamma`（既定 1）。`history_size` なし。特異は `OptimizationNotConverged`。Workspace に新しい `n×n` は足さない。葉 `hess` と NLML `hessian_into` は勾配の数値微分と一致。`Gpr<Newton>::fit` が `FittedGpr` を返し NLML が初期より下がる。Forrester は既存 JSON で回収（`θ` 相対、NLML / 予測は P1B-6 と同じ緩い許容）。新しい golden なし。既定の `tests/alloc.rs` 上限は上げない。`just lint` / `just test`。rustdoc。§6.2 / §9 と `layout.mdc` を現在形。README のソルバ段落に 1 行。criterion / `just perf` は合否にしない |
| P2B-18 | Feat | `IncrementalRecompute` の本体                 | P2B-1  | `GprObjective<IncrementalRecompute>` が `IncrementalObjective` を impl。葉 Gram は Objective にだけ置き、コンパイル済み葉だけキャッシュして dirty 葉を `apply` し直し、木の結合と Cholesky は毎回。低ランク更新なし。Workspace に新しい `n×n` は足さない。`FullRecompute` は `IncrementalObjective` を impl しない。`Objective` に既定 `value_at_changes`（既定は `value`）。FSA は `UsesChangeIndices`。初回とリスタートは `value`、座標一歩は `value_at_changes`。公開切替は `with_prefer_memory` / `with_prefer_speed`。`with_recompute_strategy` は外す。極は `B`（`RetainCholesky` なら UsesChangeIndices のとき Incremental、`ReuseCholesky` なら Full）。`with_optimizer` も同じ規則。L-BFGS に Incremental は無い。同じ θ で Incremental の値が Full の `value` と一致（単葉 RBF、Sum、Product、ノイズだけ）。空・重複・範囲外は `GprError`。`Gpr<FSA>` の速さ極・メモリ極のどちらも `fit` でき NLML が初期より下がる。同じ seed なら両極の最終 θ / NLML が一致。既定の `tests/alloc.rs` 上限は上げない。新しい golden なし。`just lint` / `just test`。rustdoc。§5.4 / §9 と `layout.mdc` を現在形。criterion / `just perf` は合否にしない |
| P2B-19 | Feat | fit 中の `L`/`W` バッファ共用                 | P2B-1  | `Gpr` / `FittedGpr` の第4型（既定 `RetainCholesky`）。`with_cholesky_buffer(ReuseCholesky)` で差し替え。既定は W を別確保（速さは変えない）。`ReuseCholesky` は勾配中に Cholesky 領域へ W を書き、`fit` 末と単独の `value_and_gradient_into` 末で Chol し直す（最適化ループ途中は戻さない）。persist にスロットは書かない。`load` は `RetainCholesky`。`Workspace` は core + 2 struct。`ReuseCholesky` に `w_matrix` は置かない。同じ θ で NLML・勾配・fit 後 predict が `RetainCholesky` と一致。既定の `tests/alloc.rs` 上限は上げない。`ReuseCholesky` に第2の `n×n`（W）が無いテスト。`just lint` / `just test`。rustdoc。§6.2 を現在形。criterion / `just perf` は合否にしない（既定 RSS は [#142](https://github.com/YUKIKEDA/gprx/issues/142)） |
| P2B-20 | Task | モジュール分割（transform は維持、compiled / FittedGpr を分ける） | P2B-10 | `input.rs` / `target.rs` は領域ファイルのまま。`compiled.rs` を `src/kernel/compiled/` に分け、`mod.rs`（enum・葉ナビ）+ `apply.rs` + `grad.rs` + `hess.rs` + `tests.rs`。`model.rs` に両 struct と `Gpr`、`FittedGpr` の impl は `fitted.rs`。`gpr/tests.rs` は分けない。公開パスは `CompiledKernel` / `Gpr` / `FittedGpr` のまま。配置と可視性以外は変えない。新しい試験・golden なし。既定の `tests/alloc.rs` 上限は上げない。`just lint` / `just test`。`layout.mdc` を現在形。criterion / `just perf` は合否にしない |
| P2B-21 | Feat | libgp 比のピーク RSS                          | P2B-16 | `DistanceCachePolicy` をトレイト。既定 `C` は `CachedDistances`。`with_distance_cache_policy(UncachedDistances)` で差し替え（速さは既定のまま）。`UncachedDistances` の Workspace に `dist_cache` / `ard_sq_diff` は置かない。等方は `X` から距離を計算。Workspace は `WithDist` / `WithW`。persist タグは `always` / `never`。`LoadedGpr::Distance` は inner enum。`load` は `RetainCholesky`。同じ θ で NLML・勾配・fit 後 predict が `CachedDistances` と一致。`UncachedDistances` に距離キャッシュが無いテスト。既定の `tests/alloc.rs` 上限は上げない。`just lint` / `just test`。rustdoc。§6.3 を現在形。`layout.mdc`。`just perf` は表2つ（既定と Uncached）。合否は Uncached + `RetainCholesky` の n=1024 / 4096（Forrester / 球）が libgp RSS ±10%。n=256 は記録。記録は `.dev/bench-log.md`。手動、CI なし。criterion / Uncached の時間は合否にしない |
| P2B-22 | Feat | n=4096 の MLL+grad（sklearn 比）              | P2B-16 | factor の Cholesky、α/W の `solve_in_place`、predict / LOO / `predict_covariance` の三角ソルブが `faer_par(n)`（`min(プール, max(1, n/64))`）。カーネルはプール全部。scratch は同じ `Par`。公開 `n_jobs` なし。rustdoc に式。ADR `.dev/adr/0001-faer-parallel-degree.md`、§8 から 1 行。`just lint` / `just test`。n=4096 Forrester / sphere で sklearn の factor・eval が P2B-16 ゲートを通る。n=256 の factor が sklearn より短い。n=256 / 1024 の eval が同じセッションの HEAD 逐次 5 回平均 +10% を超えない（PR 冒頭 HEAD、実装後に式入り、各 5 回）。記録は `.dev/bench-log.md`。手動、CI なし。criterion / RSS は合否にしない（RSS は [#142](https://github.com/YUKIKEDA/gprx/issues/142)） |
| P2B-23 | Feat | 速さ / メモリのプリセット                     | P2B-19 / P2B-21 | `Gpr` に `with_prefer_memory` / `with_prefer_speed`。メモリ極は `UncachedDistances` + `ReuseCholesky`、速さ極は既定の `CachedDistances` + `RetainCholesky`。`with_distance_cache_policy` / `with_cholesky_buffer` は `pub(crate)`。`from_points` でも同じメソッド（`C` は `NoDistanceCache`、`B` だけ）。`FittedGpr` に `with_prefer_*` は置かない。persist は `always` / `never`、`load` は `RetainCholesky`。型は crate ルートに残す。戻り型テスト、メモリ極 Workspace に dist / 専用 W が無いテスト、同じ θ の NLML・勾配・predict が既定と一致する 1 本。既定の `tests/alloc.rs` 上限は上げない。`just lint` / `just test`。rustdoc。§6.3 と `layout.mdc` を現在形。`just perf` は表2つ（既定とメモリ極）。回して `.dev/bench-log.md` に記録。新しい RSS ゲートは置かない。P2B-21 の Uncached+Retain 表は残す。criterion は合否にしない |

**2b 完了:** P2B-1…23 がマージ済み。`just test` が緑。P3-1 に進む。P2B-14…23 の作業は各 Issue の Grill と DoD 確定のあと。

---

## Phase 3 — オンライン学習

設計 §11。2b のあと。着手時に LDLT delete を先に Spike する。

| ID   | 種別  | タイトル                                    | 依存       | DoD                                                                   |
| ---- | ----- | ------------------------------------------- | ---------- | --------------------------------------------------------------------- |
| P3-1 | Spike | `ldlt::delete_rows_and_cols_clobber` の実測 | 2b         | 任意インデックス削除がフル分解と一致。ダメなら末尾削除+再分解に落とす |
| P3-2 | Feat  | `OnlineWorkspace` と容量拡張                | P3-1       | 拡張時に K/LD/y/α/cache が同期する                                    |
| P3-3 | Feat  | 末尾 insert（自前 bordered LDLT）           | P3-2       | 1点追加 == フル再 fit                                                 |
| P3-4 | Feat  | delete + `PointId` / `PointRegistry`        | P3-2       | 不変条件: 全バッファが同じ順序                                        |
| P3-5 | Task  | ランダム insert/delete のプロパティテスト   | P3-3, P3-4 | 各段階で incremental == full refit                                    |

---

## Phase 4 — Sparse GPR

設計 §6.1。初期は Z を k-means 等で固定。Z の最適化と Sparse オンラインはあとの行。

| ID   | 種別  | タイトル                                 | 依存 | DoD                                           |
| ---- | ----- | ---------------------------------------- | ---- | --------------------------------------------- |
| P4-1 | Spike | VFE か FITC か一つ選ぶ                   | 1b   | 選択理由を `.dev/` に1ページ                  |
| P4-2 | Feat  | `SparseGpr`、誘導点固定                   | P4-1 | m≪n で fit が終わる                           |
| P4-3 | Feat  | 対角予測と MLL                           | P4-2 | 小問題で Exact に近い（完全一致は要求しない） |
| P4-4 | Feat  | ハイパラ最適化（Z は params に入れない） | P4-3, P2B-9 | 2b と同じ公開 Optimizer 経路。Z は params に入れない |
| P4-5 | Spike | 誘導点 Z の最適化方式                     | P4-4 | 同時最適化か交互最適化かを `.dev/` に1ページ。理由とメモリ（`m×d`） |
| P4-6 | Feat  | Z を最適化対象にする                      | P4-5 | P4-5 の方式。カーネル・ノイズに加え Z が動く。小問題で固定 Z より NLML が下がるか記録 |
| P4-7 | Feat  | Sparse のオンライン学習                   | P4-4, P3-5 | 点の追加削除。Exact Phase 3 と同じ不変条件は要求しない。設計 §14 の非対称を `.dev/` に残し、incremental == その Sparse のフル再 fit |

---

## Phase 5 — 高度な最適化

設計 §4.1, §4.2。2 の計測のあと。`IncrementalRecompute` は P2B-18。fit 中の `L`/`W` 共用は P2B-19。

| ID   | 種別  | タイトル                                                 | 依存 | DoD                                        |
| ---- | ----- | -------------------------------------------------------- | ---- | ------------------------------------------ |
| P5-1 | Spike | 混合精度の残差（`PromoteStorage` vs `ReevaluateKernel`） | P2-1 | 方式を選ぶ。IR 不収束は f64 フォールバック |
| P5-2 | Feat  | predict 経路の MixedPrecision                            | P5-1 | 既定 fit は f64 のまま                     |
| P5-4 | Feat  | `MathMode::FastApprox` オプトイン                        | 1b   | 既定 Accurate。fit では使わない            |
| P5-5 | Task  | `DistanceCachePolicy::Auto` の閾値                       | P2-2, P2-7 | ベンチで決める。式だけで決めない           |

---

## 意図的に今やらない

この節の追加・削除は Grill → Issue（`.cursor/rules/workflow.mdc`）。エージェントは合意なしに行を足さない。

- 自前の L-BFGS / 準ニュートン実装（argmin のソルバを選んで呼ぶ）
- クラウド CI を制限中の完了条件にすること
- crates.io 公開、MSRV 約束、カバレッジ必須
