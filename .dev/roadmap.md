# gprx 実装ロードマップ

進め方は [AGENTS.md](../AGENTS.md) と `.cursor/rules/`。設計の詳細は `.dev/gprx-design.md`。

**今やること: [M0-1](https://github.com/YUKIKEDA/gprx/issues/1)。** 1a は M0 の DoD が埋まってから。後の Phase は 1b 完了まで着手しない。

Issue は 1 タスクにつき 1 本。ブランチは `type/{issue}-{slug}`（例: `chore/1-crate-bootstrap`）。

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
| P2-6    | [#60](https://github.com/YUKIKEDA/gprx/issues/60) | P1B-3   | [#22](https://github.com/YUKIKEDA/gprx/issues/22) | P5-3    | [#41](https://github.com/YUKIKEDA/gprx/issues/41) |
| P1B-6   | [#80](https://github.com/YUKIKEDA/gprx/issues/80) | P1B-4   | [#23](https://github.com/YUKIKEDA/gprx/issues/23) | P5-4    | [#42](https://github.com/YUKIKEDA/gprx/issues/42) |
| P1B-7   | [#83](https://github.com/YUKIKEDA/gprx/issues/83) | P1B-5   | [#24](https://github.com/YUKIKEDA/gprx/issues/24) | P5-5    | [#43](https://github.com/YUKIKEDA/gprx/issues/43) |
|         |                                                   | P2-1    | [#25](https://github.com/YUKIKEDA/gprx/issues/25) |         |                                                   |
|         |                                                   | P2-2    | [#26](https://github.com/YUKIKEDA/gprx/issues/26) |         |                                                   |
|         |                                                   | P2-3    | [#27](https://github.com/YUKIKEDA/gprx/issues/27) |         |                                                   |
|         |                                                   | P2-4    | [#28](https://github.com/YUKIKEDA/gprx/issues/28) |         |                                                   |

## マイルストーン

| ID  | 名前                  | 目的                           | 完了条件                                                                                |
| --- | --------------------- | ------------------------------ | --------------------------------------------------------------------------------------- |
| M0  | Spike                 | 箱と faer 0.24 を確認する      | `just lint` / `just test` が通る。2×2 と 5×5 で Cholesky 往復が一致する。GPR はまだ無い |
| 1a  | 固定ハイパラ Exact GPR | 正しい推論と勾配               | 解析解、sklearn JSON、criterion `phase-1a`、確保 ratchet、Phase 1 カーネル              |
| 1b  | Optimizer と 0.1 API  | ハイパラ最適化と使えるクレート | L-BFGS で lengthscale / ノイズ回収。README / rustdoc / 例。baseline `phase-1b`          |
| 2   | 高速化                | Phase 1 を壊さず速くする       | `phase-1b` を見てボトルネック順に最適化。キャッシュと Rayon。SIMD は測定後だけ          |
| 3   | オンライン学習        | 点の追加削除                   | 任意 delete を含む incremental == full refit。プロパティテスト                          |
| 4   | Sparse GPR             | 大きい n                       | VFE または FITC の一方。Z 固定。対角予測                                                |
| 5   | 高度な最適化          | 混合精度など                   | predict 中心の MixedPrecision。失敗時は f64 フォールバック                              |

## 依存

```
M0 → 1a → 1b → 2
                → 3
                → 4 → （Z 最適化は未決、§14）
                2 の計測のあと → 5
```

3 と 4 は 1b のあと並行してよい。5 は `phase-1b` の数値と Phase 2 の最適化対象が無いと「速くなった」と言えない。

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

設計 §5.2, §7, §8, §15。1a からある criterion / alloc を使う。新しいハーネスは作らない。1b の数値テストを回帰として残す。

| ID   | 種別  | タイトル                                                  | 依存         | DoD                                                                          |
| ---- | ----- | --------------------------------------------------------- | ------------ | ---------------------------------------------------------------------------- |
| P2-1 | Task  | `phase-1b` を読み、ボトルネック順を決める                 | 1b           | `.dev/` に短い順序（kernel vs Cholesky vs その他）。目標比はここで置いてよい |
| P2-2 | Feat  | 距離キャッシュ                                            | P2-1         | `Never` / `Always`。RBF の数値が 1b と一致。bench が改善または同等           |
| P2-3 | Feat  | Rayon でカーネル構築。`thread_scratch` を並列前に切り離す | P2-1         | 1b と数値一致。`kernel_rbf` が速くなることを bench で示す                    |
| P2-4 | Task  | 確保 ratchet をホットパス 0 まで下げる                    | P2-3, P1A-19 | `tests/alloc.rs` の上限が 0。ユーザーカーネル除く                            |
| P2-5 | Spike | カーネル SIMD が必要か                                    | P2-1, P2-3   | `kernel_rbf` が支配的なら検討。そうでなければやらない                        |
| P2-6 | Spike | NLML 定数項 `(n/2) log(2π)` の速度寄与                    | P2-1, P1A-10 | `mll_and_grad`（あれば `fit_lbfgs`）を定数あり/なしで測る。差がノイズなら一本のまま。結果を `.dev/bench-log.md` に残す。この行では API を分けない |

---

## Phase 3 — オンライン学習

設計 §11。着手時に LDLT delete を先に Spike する。

| ID   | 種別  | タイトル                                    | 依存       | DoD                                                                   |
| ---- | ----- | ------------------------------------------- | ---------- | --------------------------------------------------------------------- |
| P3-1 | Spike | `ldlt::delete_rows_and_cols_clobber` の実測 | 1b         | 任意インデックス削除がフル分解と一致。ダメなら末尾削除+再分解に落とす |
| P3-2 | Feat  | `OnlineWorkspace` と容量拡張                | P3-1       | 拡張時に K/LD/y/α/cache が同期する                                    |
| P3-3 | Feat  | 末尾 insert（自前 bordered LDLT）           | P3-2       | 1点追加 == フル再 fit                                                 |
| P3-4 | Feat  | delete + `PointId` / `PointRegistry`        | P3-2       | 不変条件: 全バッファが同じ順序                                        |
| P3-5 | Task  | ランダム insert/delete のプロパティテスト   | P3-3, P3-4 | 各段階で incremental == full refit                                    |

---

## Phase 4 — Sparse GPR

設計 §6.1。Z は k-means 等で固定。

| ID   | 種別  | タイトル                                 | 依存 | DoD                                           |
| ---- | ----- | ---------------------------------------- | ---- | --------------------------------------------- |
| P4-1 | Spike | VFE か FITC か一つ選ぶ                   | 1b   | 選択理由を `.dev/` に1ページ                  |
| P4-2 | Feat  | `SparseGpr`、誘導点固定                   | P4-1 | m≪n で fit が終わる                           |
| P4-3 | Feat  | 対角予測と MLL                           | P4-2 | 小問題で Exact に近い（完全一致は要求しない） |
| P4-4 | Feat  | ハイパラ最適化（Z は params に入れない） | P4-3 | 1b と同じ Optimizer 経路                      |

---

## Phase 5 — 高度な最適化

設計 §4.1, §4.2, §5.4。2 の計測のあと。

| ID   | 種別  | タイトル                                                 | 依存 | DoD                                        |
| ---- | ----- | -------------------------------------------------------- | ---- | ------------------------------------------ |
| P5-1 | Spike | 混合精度の残差（`PromoteStorage` vs `ReevaluateKernel`） | P2-1 | 方式を選ぶ。IR 不収束は f64 フォールバック |
| P5-2 | Feat  | predict 経路の MixedPrecision                            | P5-1 | 既定 fit は f64 のまま                     |
| P5-3 | Feat  | `IncrementalRecompute`（オプトイン）                     | 1b   | FullRecompute と数値が一致                 |
| P5-4 | Feat  | `MathMode::FastApprox` オプトイン                        | 1b   | 既定 Accurate。fit では使わない            |
| P5-5 | Task  | `DistanceCachePolicy::Auto` の閾値                       | P2-2 | ベンチで決める。式だけで決めない           |

---

## 意図的に今やらない

- crates.io 公開、MSRV 約束、カバレッジ必須
- 自前 L-BFGS
- 誘導点 Z の最適化、Sparse のオンライン学習
- Likelihood と White を両方既定で足すこと
- クラウド CI を制限中の完了条件にすること
