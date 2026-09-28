# gprx 実装ロードマップ

進め方は [CONTRIBUTING.md](../CONTRIBUTING.md)。設計は [design.md](design.md)。計測は [.dev/bench-log.md](../.dev/bench-log.md)。

完了条件は各 Issue に残す。このファイルは ID、タイトル、Issue、状態だけを持つ。

## 今やること

R1-1（[#227](https://github.com/YUKIKEDA/gprx/issues/227)、f32 / f64 共通の数値スカラー trait）。完了条件は Grill 後に #227 で確定。R 段（コードベース整理、親 [#226](https://github.com/YUKIKEDA/gprx/issues/226)）を P5-5 より先に進める。P5-5 は R4-1 のあと。

## 依存

```
M0 → 1a → 1b → 2 → 2b → 3
                     → 4
                2 の計測のあと → 5
R1 → R2 → R3 → R4 → R5 → R6
R4-1 → P5-5
```

3 は 2b のあと。4 の初期は 1b のあとでも並ぶ。5 は `phase-2` が無いと速くなったと言えない。R の各行の依存は各 Issue に書く。

## M0

| ID | 種別 | タイトル | Issue | 状態 |
| --- | --- | --- | --- | --- |
| M0-1 | Task | クレート初期化（Cargo.toml, MIT OR Apache-2.0, justfile, CI yaml, gitignore, README stub） | [#1](https://github.com/YUKIKEDA/gprx/issues/1) | 済 |
| M0-2 | Spike | faer 0.24 `cholesky_in_place` 往復 | [#2](https://github.com/YUKIKEDA/gprx/issues/2) | 済 |

## 1a

| ID | 種別 | タイトル | Issue | 状態 |
| --- | --- | --- | --- | --- |
| P1A-1 | Task | `GprError` と `DoublePrecision` のみの型 | [#3](https://github.com/YUKIKEDA/gprx/issues/3) | 済 |
| P1A-2 | Task | `Workspace`（f64, `k_matrix` / `w_matrix` / `dist_cache` / `exp_buf` / faer scratch） | [#4](https://github.com/YUKIKEDA/gprx/issues/4) | 済 |
| P1A-3 | Feat | `GaussianLikelihood`（`θ=log(σn²)`, `∂K/∂θ=σn² I`） | [#5](https://github.com/YUKIKEDA/gprx/issues/5) | 済 |
| P1A-4 | Feat | `TargetTransform` と X の `Transform`（Identity / Standardize） | [#6](https://github.com/YUKIKEDA/gprx/issues/6) | 済 |
| P1A-5 | Feat | RBF カーネル（`apply`/`grad`, `uplo=Lower`） | [#7](https://github.com/YUKIKEDA/gprx/issues/7) | 済 |
| P1A-6 | Feat | `KernelSpec` / `CompiledKernel`（RBF + Sum/Product + param flatten） | [#8](https://github.com/YUKIKEDA/gprx/issues/8) | 済 |
| P1A-7 | Feat | `Gpr`: `A=K+σn²I`、LLT、`α` | [#9](https://github.com/YUKIKEDA/gprx/issues/9) | 済 |
| P1A-8 | Feat | `predict`（mean, `VarianceKind::{Latent,Observation}`） | [#10](https://github.com/YUKIKEDA/gprx/issues/10) | 済 |
| P1A-9 | Feat | 負の MLL | [#11](https://github.com/YUKIKEDA/gprx/issues/11) | 済 |
| P1A-10 | Feat | `W=ααᵀ-K⁻¹` による勾配 | [#12](https://github.com/YUKIKEDA/gprx/issues/12) | 済 |
| P1A-11 | Task | 解析解の統合テスト（n=2/3 RBF） | [#13](https://github.com/YUKIKEDA/gprx/issues/13) | 済 |
| P1A-12 | Task | sklearn golden（`compare/` + `just gen-goldens` + JSON） | [#14](https://github.com/YUKIKEDA/gprx/issues/14) | 済 |
| P1A-13 | Feat | Constant, Linear, White | [#15](https://github.com/YUKIKEDA/gprx/issues/15) | 済 |
| P1A-14 | Feat | Matern ν=1/2, 3/2, 5/2 | [#16](https://github.com/YUKIKEDA/gprx/issues/16) | 済 |
| P1A-15 | Feat | Periodic | [#17](https://github.com/YUKIKEDA/gprx/issues/17) | 済 |
| P1A-16 | Feat | Rational Quadratic | [#18](https://github.com/YUKIKEDA/gprx/issues/18) | 済 |
| P1A-17 | Task | 合成カーネルと追加 golden | [#19](https://github.com/YUKIKEDA/gprx/issues/19) | 済 |
| P1A-18 | Task | criterion ハーネス（`benches/exact.rs`, `just bench`） | [#44](https://github.com/YUKIKEDA/gprx/issues/44) | 済 |
| P1A-19 | Task | 確保 ratchet（`tests/alloc.rs`） | [#45](https://github.com/YUKIKEDA/gprx/issues/45) | 済 |
| P1A-20 | Feat | ARD lengthscale（まず RBF） | [#54](https://github.com/YUKIKEDA/gprx/issues/54) | 済 |

## 1b

| ID | 種別 | タイトル | Issue | 状態 |
| --- | --- | --- | --- | --- |
| P1B-1 | Feat | `Objective` と `GprObjective` | [#20](https://github.com/YUKIKEDA/gprx/issues/20) | 済 |
| P1B-2 | Feat | argmin L-BFGS アダプタ | [#21](https://github.com/YUKIKEDA/gprx/issues/21) | 済 |
| P1B-3 | Feat | `Gpr::fit` が最適化する | [#22](https://github.com/YUKIKEDA/gprx/issues/22) | 済 |
| P1B-4 | Task | パラメータ回収テスト | [#23](https://github.com/YUKIKEDA/gprx/issues/23) | 済 |
| P1B-5 | Docs | README, rustdoc, `examples/` | [#24](https://github.com/YUKIKEDA/gprx/issues/24) | 済 |
| P1B-6 | Task | fit 込み sklearn golden（Forrester / ARD） | [#80](https://github.com/YUKIKEDA/gprx/issues/80) | 済 |
| P1B-7 | Feat | leave-one-out 予測 | [#83](https://github.com/YUKIKEDA/gprx/issues/83) | 済 |

## 2

| ID | 種別 | タイトル | Issue | 状態 |
| --- | --- | --- | --- | --- |
| P2-1 | Task | `phase-1b` を読み、ボトルネック順を決める | [#25](https://github.com/YUKIKEDA/gprx/issues/25) | 済 |
| P2-2 | Feat | 距離キャッシュ | [#26](https://github.com/YUKIKEDA/gprx/issues/26) | 済 |
| P2-3 | Feat | Rayon でカーネル構築。`thread_scratch` を並列前に切り離す | [#27](https://github.com/YUKIKEDA/gprx/issues/27) | 済 |
| P2-4 | Task | 確保 ratchet をホットパス 0 まで下げる | [#28](https://github.com/YUKIKEDA/gprx/issues/28) | 済 |
| P2-5 | Feat | 等方 RBF と距離に SIMD | [#29](https://github.com/YUKIKEDA/gprx/issues/29) | 済 |
| P2-6 | Spike | NLML 定数項 `(n/2) log(2π)` の速度寄与 | [#60](https://github.com/YUKIKEDA/gprx/issues/60) | 済 |
| P2-7 | Feat | ARD 距離キャッシュ | [#88](https://github.com/YUKIKEDA/gprx/issues/88) | 済 |
| P2-8 | Feat | `Gpr` / `FittedGpr` の typestate | [#98](https://github.com/YUKIKEDA/gprx/issues/98) | 済 |
| P2-9 | Task | Phase 2 締め | [#97](https://github.com/YUKIKEDA/gprx/issues/97) | 済 |

## 2b

| ID | 種別 | タイトル | Issue | 状態 |
| --- | --- | --- | --- | --- |
| P2B-1 | Feat | 最適化ノブと `Gpr<Fixed>` | [#108](https://github.com/YUKIKEDA/gprx/issues/108) | 済 |
| P2B-2 | Feat | argmin ソルバの選択 | [#114](https://github.com/YUKIKEDA/gprx/issues/114) | 済 |
| P2B-3 | Feat | `KernelTerm` と `KernelSpec::Custom` | [#117](https://github.com/YUKIKEDA/gprx/issues/117) | 済 |
| P2B-4 | Feat | `JitterPolicy` | [#119](https://github.com/YUKIKEDA/gprx/issues/119) | 済 |
| P2B-5 | Feat | 学習済みの読み書きと Clone | [#121](https://github.com/YUKIKEDA/gprx/issues/121) | 済 |
| P2B-6 | Feat | 任意の予測共分散と posterior sample | [#123](https://github.com/YUKIKEDA/gprx/issues/123) | 済 |
| P2B-7 | Feat | 変換 Pipeline | [#127](https://github.com/YUKIKEDA/gprx/issues/127) | 済 |
| P2B-8 | Feat | 入力変換を列ごとに指定 | [#129](https://github.com/YUKIKEDA/gprx/issues/129) | 済 |
| P2B-9 | Feat | 自作 `Optimizer` の差し替え | [#131](https://github.com/YUKIKEDA/gprx/issues/131) | 済 |
| P2B-10 | Feat | 変換の fitted を型にする | [#125](https://github.com/YUKIKEDA/gprx/issues/125) | 済 |
| P2B-11 | Feat | 距離キャッシュを距離経路専用にする | [#133](https://github.com/YUKIKEDA/gprx/issues/133) | 済 |
| P2B-12 | Feat | Product の points 勾配 | [#135](https://github.com/YUKIKEDA/gprx/issues/135) | 済 |
| P2B-13 | Feat | Dist と Points の Sum/Product | [#137](https://github.com/YUKIKEDA/gprx/issues/137) | 済 |
| P2B-14 | Feat | 学習済みモデルの保存・読み込み | [#63](https://github.com/YUKIKEDA/gprx/issues/63) | 済 |
| P2B-15 | Feat | カスタム Optimizer の使用例 | [#106](https://github.com/YUKIKEDA/gprx/issues/106) | 済 |
| P2B-16 | Spike | 他ライブラリとの時間・RSS 比較 | [#103](https://github.com/YUKIKEDA/gprx/issues/103) | 済 |
| P2B-17 | Feat | `GprObjective` の NLML ヘッセ | [#109](https://github.com/YUKIKEDA/gprx/issues/109) | 済 |
| P2B-18 | Feat | `IncrementalRecompute` の本体 | [#110](https://github.com/YUKIKEDA/gprx/issues/110) | 済 |
| P2B-19 | Feat | fit 中の `L`/`W` バッファ共用 | [#111](https://github.com/YUKIKEDA/gprx/issues/111) | 済 |
| P2B-20 | Task | モジュール分割（transform は維持、compiled / FittedGpr を分ける） | [#116](https://github.com/YUKIKEDA/gprx/issues/116) | 済 |
| P2B-21 | Feat | libgp 比のピーク RSS | [#142](https://github.com/YUKIKEDA/gprx/issues/142) | 済 |
| P2B-22 | Feat | n=4096 の MLL+grad（sklearn 比） | [#143](https://github.com/YUKIKEDA/gprx/issues/143) | 済 |
| P2B-23 | Feat | 速さ / メモリのプリセット | [#148](https://github.com/YUKIKEDA/gprx/issues/148) | 済 |

## 3

| ID | 種別 | タイトル | Issue | 状態 |
| --- | --- | --- | --- | --- |
| P3-1 | Spike | `ldlt::delete_rows_and_cols_clobber` の実測 | [#30](https://github.com/YUKIKEDA/gprx/issues/30) | 済 |
| P3-2 | Feat | `OnlineWorkspace` と容量拡張 | [#31](https://github.com/YUKIKEDA/gprx/issues/31) | 済 |
| P3-3 | Feat | 末尾 insert（自前 bordered LDLT） | [#32](https://github.com/YUKIKEDA/gprx/issues/32) | 済 |
| P3-4 | Feat | delete + `PointId` / `PointRegistry` | [#33](https://github.com/YUKIKEDA/gprx/issues/33) | 済 |
| P3-5 | Task | ランダム insert/delete のプロパティテスト | [#34](https://github.com/YUKIKEDA/gprx/issues/34) | 済 |
| P3-6 | Spike | libgp との online insert 照合と時間比較 | [#176](https://github.com/YUKIKEDA/gprx/issues/176) | 済 |
| P3-7 | Feat | online insert を libgp 比で速くする | [#177](https://github.com/YUKIKEDA/gprx/issues/177) | 済 |

## 4

| ID | 種別 | タイトル | Issue | 状態 |
| --- | --- | --- | --- | --- |
| P4-1 | Spike | VFE か FITC か一つ選ぶ | [#35](https://github.com/YUKIKEDA/gprx/issues/35) | 済 |
| P4-2 | Feat | `SparseGpr`、誘導点固定 | [#36](https://github.com/YUKIKEDA/gprx/issues/36) | 済 |
| P4-3 | Feat | 対角予測と MLL | [#37](https://github.com/YUKIKEDA/gprx/issues/37) | 済 |
| P4-4 | Feat | ハイパラ最適化（Z は params に入れない） | [#38](https://github.com/YUKIKEDA/gprx/issues/38) | 済 |
| P4-5 | Spike | 誘導点 Z の最適化方式 | [#184](https://github.com/YUKIKEDA/gprx/issues/184) | 済 |
| P4-6 | Feat | Z を最適化対象にする | [#186](https://github.com/YUKIKEDA/gprx/issues/186) | 済 |
| P4-7 | Spike | VFE 因子の rank-1 更新 | [#188](https://github.com/YUKIKEDA/gprx/issues/188) | 済 |
| P4-8 | Feat | Sparse のオンライン学習 | [#190](https://github.com/YUKIKEDA/gprx/issues/190) | 済 |
| P4-9 | Spike | 誘導点の増分 insert / delete | [#192](https://github.com/YUKIKEDA/gprx/issues/192) | 済 |
| P4-10 | Feat | 誘導点の増減 | [#193](https://github.com/YUKIKEDA/gprx/issues/193) | 済 |
| P4-11 | Spike | Sparse の GPyTorch 照合 | [#196](https://github.com/YUKIKEDA/gprx/issues/196) | 済 |
| P4-12 | Spike | Sparse の時間・RSS 比較 | [#197](https://github.com/YUKIKEDA/gprx/issues/197) | 済 |
| P4-13 | Spike | Sparse オンラインの GPyTorch 照合 | [#198](https://github.com/YUKIKEDA/gprx/issues/198) | 済 |
| P4-14 | Spike | Sparse オンラインの時間比較 | [#199](https://github.com/YUKIKEDA/gprx/issues/199) | 済 |
| P4-15 | Feat | SVGP の factor / ELBO / 対角予測 | [#201](https://github.com/YUKIKEDA/gprx/issues/201) | 済 |
| P4-16 | Feat | SVGP の Adam / ミニバッチ fit | [#202](https://github.com/YUKIKEDA/gprx/issues/202) | 済 |
| P4-17 | Feat | `SparseGpr` を `Sgpr` に改名 | [#205](https://github.com/YUKIKEDA/gprx/issues/205) | 済 |
| P4-18 | Spike | Sparse の時間（GPy / GPyTorch 比） | [#209](https://github.com/YUKIKEDA/gprx/issues/209) | 済 |
| P4-19 | Spike | Sparse のピーク RSS（n=4096） | [#210](https://github.com/YUKIKEDA/gprx/issues/210) | 済 |
| P4-20 | Spike | Sparse joint の残り（GPy 比） | [#214](https://github.com/YUKIKEDA/gprx/issues/214) | 済 |
| P4-21 | Spike | SVGP joint の残り（GPy 比） | [#217](https://github.com/YUKIKEDA/gprx/issues/217) | 済 |

## 5

| ID | 種別 | タイトル | Issue | 状態 |
| --- | --- | --- | --- | --- |
| P5-1 | Spike | 混合精度の残差（`PromoteStorage` vs `ReevaluateKernel`） | [#39](https://github.com/YUKIKEDA/gprx/issues/39) | 済 |
| P5-2 | Feat | f32・f64・混合精度 | [#40](https://github.com/YUKIKEDA/gprx/issues/40) | 済 |
| P5-4 | Feat | `MathMode::FastApprox` オプトイン | [#42](https://github.com/YUKIKEDA/gprx/issues/42) | 済 |
| P5-5 | Task | `DistanceCachePolicy::Auto` の閾値 | [#43](https://github.com/YUKIKEDA/gprx/issues/43) | Grill 後に #43 で確定 |

## R（コードベース整理、親 [#226](https://github.com/YUKIKEDA/gprx/issues/226)）

| ID | 種別 | タイトル | Issue | 状態 |
| --- | --- | --- | --- | --- |
| R1-1 | Task | f32 / f64 共通の数値スカラー trait を導入し、size_of による型判定を無くす | [#227](https://github.com/YUKIKEDA/gprx/issues/227) | Grill 後に #227 で確定 |
| R1-2 | Task | 線形代数の処理を linalg モジュールに集約する | [#228](https://github.com/YUKIKEDA/gprx/issues/228) | Grill 後に #228 で確定 |
| R1-3 | Task | 入力検証と列優先パッキングを data モジュールに集約する | [#229](https://github.com/YUKIKEDA/gprx/issues/229) | Grill 後に #229 で確定 |
| R1-4 | Task | テストの共通ヘルパーと問題データ生成を1か所にする | [#230](https://github.com/YUKIKEDA/gprx/issues/230) | Grill 後に #230 で確定 |
| R1-5 | Task | `GprError` の分類を整理する | [#231](https://github.com/YUKIKEDA/gprx/issues/231) | Grill 後に #231 で確定 |
| R2-1 | Task | カーネル評価の入口を1つの入力構造体に統一する | [#232](https://github.com/YUKIKEDA/gprx/issues/232) | Grill 後に #232 で確定 |
| R2-2 | Task | 定常カーネルを式（profile）と走査（driver）に分け、f32_eval.rs を無くす | [#233](https://github.com/YUKIKEDA/gprx/issues/233) | Grill 後に #233 で確定 |
| R2-3 | Feat | ユーザー定義カーネルを f32 / f64 共通の1実装で書けるようにする | [#234](https://github.com/YUKIKEDA/gprx/issues/234) | Grill 後に #234 で確定 |
| R2-4 | Task | カーネルと crate の公開面の不揃いと漏れを直す | [#235](https://github.com/YUKIKEDA/gprx/issues/235) | Grill 後に #235 で確定 |
| R3-1 | Bug | `MixedPrecision` の predict が呼ぶたびに反復改良とカーネル再コンパイルを行う | [#236](https://github.com/YUKIKEDA/gprx/issues/236) | Grill 後に #236 で確定 |
| R3-2 | Bug | 反復改良が保存済みの因子と `JitterPolicy` を使わない | [#237](https://github.com/YUKIKEDA/gprx/issues/237) | Grill 後に #237 で確定 |
| R3-3 | Task | 精度ポリシーの trait 群を1つにまとめ、反復改良を Refiner にする | [#238](https://github.com/YUKIKEDA/gprx/issues/238) | Grill 後に #238 で確定 |
| R4-1 | Task | 距離キャッシュ・Cholesky バッファ・exp モード・再計算戦略を実行時の値にする | [#239](https://github.com/YUKIKEDA/gprx/issues/239) | Grill 後に #239 で確定 |
| R4-2 | Task | 因子の抽象（LLT / LDLT）とモデル共通部で `FittedGpr` と `OnlineGpr` を統合する | [#240](https://github.com/YUKIKEDA/gprx/issues/240) | Grill 後に #240 で確定 |
| R4-3 | Task | predict の経路の重複を1本にする | [#241](https://github.com/YUKIKEDA/gprx/issues/241) | Grill 後に #241 で確定 |
| R4-4 | Task | 学習済みモデルの状態・ロールバック・任意バッファの表し方を統一する | [#242](https://github.com/YUKIKEDA/gprx/issues/242) | Grill 後に #242 で確定 |
| R4-5 | Bug | `IncrementalRecompute` がビット差分で変更した葉を推定し、呼ぶたびに確保する | [#243](https://github.com/YUKIKEDA/gprx/issues/243) | Grill 後に #243 で確定 |
| R4-6 | Task | persist の読み込み結果（Loaded* の 48 型）を縮約する | [#244](https://github.com/YUKIKEDA/gprx/issues/244) | Grill 後に #244 で確定 |
| R4-7 | Task | src/gpr のファイル構成を整理する | [#245](https://github.com/YUKIKEDA/gprx/issues/245) | Grill 後に #245 で確定 |
| R5-1 | Task | `Sgpr` / `Svgp` を共通の core・linalg・精度に載せ、依存方向を直す | [#246](https://github.com/YUKIKEDA/gprx/issues/246) | Grill 後に #246 で確定 |
| R5-2 | Spike | `Sgpr` / `Svgp` と Exact の機能差を揃えるかを決める | [#247](https://github.com/YUKIKEDA/gprx/issues/247) | Grill 後に #247 で確定 |
| R6-1 | Task | compare/ の Python ハーネスと問題定義の重複をまとめる | [#248](https://github.com/YUKIKEDA/gprx/issues/248) | Grill 後に #248 で確定 |
| R6-2 | Task | design.md の擬似コードと記述を実装に合わせる | [#249](https://github.com/YUKIKEDA/gprx/issues/249) | Grill 後に #249 で確定 |

## 意図的に今やらない

この節の追加・削除は Grill → Issue（`.cursor/rules/workflow.mdc`）。エージェントは合意なしに行を足さない。

- 自前の L-BFGS / 準ニュートン実装（argmin のソルバを選んで呼ぶ）
- クラウド CI を制限中の完了条件にすること
- crates.io 公開、MSRV 約束、カバレッジ必須
