# Bench log

時間は [criterion](https://docs.rs/criterion)、`benches/exact.rs`、`just bench`。確保は `tests/alloc.rs`。他ライブラリとの壁時計・ピーク RSS は `compare/perf/`（`just perf`。手動、CI なし。criterion は合否に使わない）。固定問題（P2-9 再測以降）: `n = 256`。等方は 1 次元 Forrester + `StandardizeTarget`、ARD は 2 次元重み付き球（16×16）。`y` は関数 + `N(0, 1)`。P2B-15 以降は `SmallRng`（Forrester seed `0`、ARD seed `9`）。それ以前の `phase-2` ARD 行は SplitMix64 seed `0` なので、新しい ARD 時間と混ぜない。`fit_lbfgs` だけ最適化ループ。P2-9 より前の行は `d = 8`・独立乱数 `y`。Forrester 上の `phase-1b` 再測は P2-9 節。

目標比はまだ置かない（P2-1 / [#25](https://github.com/YUKIKEDA/gprx/issues/25)）。名前付き `phase-2` は P2-9 で取った（等方は `phase-1b` と比較。ARD は Always vs Never）。

## 機械

| 項目  | 値                                |
| ----- | --------------------------------- |
| OS    | Windows 11 Home 10.0.26200        |
| CPU   | 13th Gen Intel Core i5-13400F     |
| RAM   | 48 GB                             |
| rustc | 1.97.1 (`x86_64-pc-windows-msvc`) |
| 日付  | 2026-09-18                        |

criterion は gnuplot なし、plotters。baseline 名 `phase-1b` / `phase-2` はこの機械の `target/criterion` に保存した（git には入れない）。比較: `cargo bench --bench exact -- --baseline phase-2`。

## `phase-1b`（n = 256、d = 8・独立乱数 `y`）

criterion 中央値（括弧は 95% 区間の両端）。

| グループ         | 時間                | 確保（Workspace 後） |
| ---------------- | ------------------- | -------------------- |
| `kernel_rbf`     | 266 µs (263–269)    | —                    |
| `cholesky_alpha` | 159 µs (157–162)    | —                    |
| `predict_100`    | 379 µs (376–382)    | 9（上限 9）          |
| `mll_and_grad`   | 1.42 ms (1.39–1.45) | 1（上限 16）         |
| `fit_lbfgs`      | 257 ms (256–258)    | —                    |

メモリ（解析、ピーク RSS は測っていない）: Workspace の密 `n×n` f64 が 4 枚（`k_matrix` / `w_matrix` / `dist_cache` / `exp_buf`）。等方 RBF では `kernel_scratch` は空。`4 × 256² × 8 B = 2.00 MiB`。これに `X`（16 KiB）、`y` / `α`、faer scratch が乗る。

## ボトルネック順（P2-1）

`mll_and_grad` 1 回（1.42 ms）を単位にする。孤立ベンチの和は完全には足し合わないが、大小は十分に分かれる。

1. **その他** — `W = ααᵀ - A⁻¹` と `∂K/∂θ`。1 評価のおよそ 7 割（1.42 ms − kernel − Cholesky）。
2. **kernel** — `kernel_rbf`（距離 + RBF 下三角）266 µs。Cholesky より重い。
3. **Cholesky** — `cholesky_alpha`（LLT + `α`）159 µs。n = 256 では部品として最速。

`predict_100` は別経路（379 µs）。`fit_lbfgs` は上の 1 評価を繰り返す（257 ms）。

Phase 2 の順: 距離キャッシュと Rayon は **kernel**（2）向け。Cholesky を先に触らない。`mll_and_grad` の支配項は勾配用の `W`/`∂K` だが、SIMD の可否はカーネル経路（`kernel_rbf` / `predict` / `FIXED`）で決める。P2-5 で RBF と距離に SIMD を入れた。定数項 `(n/2) log(2π)` は P2-6 でノイズと分かり、`L(θ)` は一本のまま。

## P2-2（距離キャッシュ、[#26](https://github.com/YUKIKEDA/gprx/issues/26)）

同一機械。比較: `cargo bench --bench exact -- --baseline phase-1b`。デフォルトは `DistanceCachePolicy::Always`（fit 中に距離を 1 回だけ埋める）。孤立 `kernel_rbf` は Gpr を通らず毎回距離を埋めるので、キャッシュの効果は `mll_and_grad` / `fit_lbfgs` に出る。

| グループ         | phase-1b | この PR             | 変化（中央値）                            |
| ---------------- | -------- | ------------------- | ----------------------------------------- |
| `kernel_rbf`     | 266 µs   | 249 µs (246–254)    | −4.9%（孤立経路。Gpr キャッシュ外）       |
| `cholesky_alpha` | 159 µs   | 151 µs (150–151)    | −7.8%（この行は触っていない。機械ゆらぎ） |
| `predict_100`    | 379 µs   | 362 µs (358–368)    | −5.0%                                     |
| `mll_and_grad`   | 1.42 ms  | 1.23 ms (1.21–1.26) | **−13.3%**                                |
| `fit_lbfgs`      | 257 ms   | 224 ms (222–226)    | **−13.0%**                                |

DoD（改善または同等）は満たす。確保上限は変えていない。

## P2-3（Rayon カーネル構築、[#27](https://github.com/YUKIKEDA/gprx/issues/27)）

同一機械。比較: `cargo bench --bench exact -- --baseline phase-1b`。等方距離は下三角を列パーティションで並列し、上三角は逐次コピー。`write_triangle(Lower)` も同じ分割。Cholesky は `Par::Seq`。`thread_scratch` はワーカー数ぶんの空 `0×0` で、並列前に `mem::take`。孤立 `kernel_rbf` は毎回距離埋め + RBF 下三角なので、この行の効果はそこに出る。

| グループ         | phase-1b | この PR               | 変化（中央値）                            |
| ---------------- | -------- | --------------------- | ----------------------------------------- |
| `kernel_rbf`     | 266 µs   | 149 µs (148–149)      | **−43.8%**                                |
| `cholesky_alpha` | 159 µs   | 153 µs (152–155)      | −7.0%（この行は触っていない。機械ゆらぎ） |
| `predict_100`    | 379 µs   | 374 µs (370–383)      | −1.8%（ノイズ域）                         |
| `mll_and_grad`   | 1.42 ms  | 1.18 ms (1.176–1.184) | **−15.7%**                                |
| `fit_lbfgs`      | 257 ms   | 231 ms (230–233)      | **−10.1%**                                |

DoD（`kernel_rbf` が速くなること、1b と数値一致）は満たす。確保: `mll_and_grad` は 1（上限 16）。`predict_100` は warmup 後 8 に下がったので上限を 9→8。P2-2 Always（mll 1.23 ms / fit 224 ms）と比べると mll はもう一段速い。`fit_lbfgs` は 231 ms で P2-2 よりわずかに遅い（キャッシュ済み apply の並列オーバーヘッド + ゆらぎ）。

## P2-4（ホットパス確保 0、[#28](https://github.com/YUKIKEDA/gprx/issues/28)）

同一機械。比較: `cargo bench --bench exact -- --baseline phase-1b`。Workspace に `rhs`（n×1）と query バッファを置き、`value_and_gradient_into` と `predict_into` は warmup 後に新規確保しない。`predict_100` ベンチは `predict_into`。Cholesky は `Par::Seq`。カーネル並列は触っていない。

| グループ         | phase-1b | この PR               | 変化（中央値）                                |
| ---------------- | -------- | --------------------- | --------------------------------------------- |
| `kernel_rbf`     | 266 µs   | 152 µs (152–153)      | **−42.5%**（P2-3 相当。この行は触っていない） |
| `cholesky_alpha` | 159 µs   | 156 µs (155–158)      | −2.8%（ノイズ域）                             |
| `predict_100`    | 379 µs   | 368 µs (364–371)      | **−3.8%**（query バッファ再利用）             |
| `mll_and_grad`   | 1.42 ms  | 1.21 ms (1.205–1.215) | **−14.1%**                                    |
| `fit_lbfgs`      | 257 ms   | 238 ms (237–240)      | **−7.4%**                                     |

DoD（`tests/alloc.rs` 上限 0、ユーザーカーネル除く）は満たす。確保: `mll_and_grad` 0、`predict_100`（`predict_into`）0。P2-3（mll 1.18 ms / fit 231 ms / predict 374 µs）と比べると時間は同等域。`predict` の便利 API は出力 `Vec` を毎回確保する。

## P2-5（カーネル SIMD、[#29](https://github.com/YUKIKEDA/gprx/issues/29)）

同一機械。`wide::f64x4`。列優先・単位行ストライドの等方 RBF `apply` / `grad` / `apply_cross` と二乗距離の行ループ。既定 rustc（SSE2、`target-cpu=native` なし。phase-1b と同じフラグ）。

最初の下書きは `mll_and_grad` を分母にして「13% だから入れない」とした。勾配込み 1 評価ではカーネルは最大部品ではないが、`FIXED` と `predict` は勾配を払わない。その結論は取り消した。

比較: `cargo bench --bench exact -- --baseline phase-1b "kernel_rbf|cholesky_alpha|predict_100|mll_and_grad"`。

| グループ         | phase-1b | Rayon のみ（P2-5 初稿） | この PR（Rayon + SIMD） | vs 1b                   | vs Rayon のみ |
| ---------------- | -------- | ----------------------- | ----------------------- | ----------------------- | ------------- |
| `kernel_rbf`     | 266 µs   | 165 µs (162–167)        | 125 µs (125.2–125.7)    | **−52.4%**              | **−24%**      |
| `cholesky_alpha` | 159 µs   | 165 µs (163–167)        | 155 µs (153–158)        | −5.7%（この行は未変更） | ノイズ域      |
| `predict_100`    | 379 µs   | 368 µs（P2-4）          | 284 µs (280–288)        | **−26.0%**              | **−23%**      |
| `mll_and_grad`   | 1.42 ms  | 1.25 ms (1.243–1.260)   | 1.16 ms (1.159–1.163)   | **−17.4%**              | **−7%**       |

`kernel_rbf` は距離埋め + RBF 下三角。SIMD 後は Cholesky（155 µs）より短い。Matérn / Periodic / RQ の内側はまだスカラー。

## P2-6（NLML 定数項は一本のまま、[#60](https://github.com/YUKIKEDA/gprx/issues/60)）

同一機械。固定問題で `(n/2) log(2π)` のあり/なしを測った。公開 API は分けていない（計測時だけ内部フラグを切替）。数値は P2-5 初稿（Rayon のみ）上。

| 経路                                     | 時間（中央値）         |
| ---------------------------------------- | ---------------------- |
| `nlml_constant`（孤立の `0.5 n ln(2π)`） | 648 ps (642–655)       |
| `mll_and_grad` 定数なし                  | 1.188 ms (1.187–1.190) |
| `mll_and_grad` 定数あり（直後の再測）    | 1.200 ms (1.195–1.205) |

あり/なしの差は 12 µs（約 1%）。孤立加算 650 ps の 1.8 万倍で、criterion の連測ゆらぎに埋まる。`fit_lbfgs` は同じ定数を評価ごとに 1 回足すだけなので測っていない（200 回でも 0.13 µs 対 238 ms）。

**判断: 差はノイズ。公開 NLML と `Objective` は同じ `L(θ)` のまま。API は分けない。**

## P2-7（ARD 距離キャッシュ、[#88](https://github.com/YUKIKEDA/gprx/issues/88)）

同一機械。固定問題の ARD RBF（`ℓ_d = 1`、d = 8）。公開 Policy は増やしていない。Always は `n × (n·d)` の生 `(Δx_d)²` を fit 開始時に 1 回埋め、RBF ARD の apply/grad は Rayon + `wide::f64x4`。Never は座標から毎回（同じく SIMD）。Matérn / RQ ARD は同じキャッシュをスカラーで読む。等方 `phase-1b` とは比べない。

比較: `cargo bench --bench exact -- "mll_and_grad_ard|fit_lbfgs_ard"`。

| グループ           | Always                 | Never                  | Always vs Never             |
| ------------------ | ---------------------- | ---------------------- | --------------------------- |
| `mll_and_grad_ard` | 2.247 ms (2.228–2.269) | 2.113 ms (2.101–2.124) | **+6.3%**（Always が遅い）  |
| `fit_lbfgs_ard`    | 5.647 ms (5.609–5.729) | 4.526 ms (4.509–4.544) | **+24.8%**（Always が遅い） |

メモリ（解析）: Always の ARD テンソルは `256 × (256·8) × 8 B = 4.00 MiB`。Never / 等方は 0。X 自体は 16 KiB。n = 256, d = 8 ではキャッシュ読み（n²d）より座標 SIMD（nd）の方が帯域が小さい。設計 §5.2 の「d≪n では投資対効果が薄い」と一致。DoD の Always vs Never は「改善または同等」だが、この問題では Never が速い。Policy と数値一致（Never ≡ Always）はテストで満たす。`Auto` は P5-5。

## P2-9（`phase-2`、[#97](https://github.com/YUKIKEDA/gprx/issues/97)）

同一機械。`cargo bench --bench exact -- --save-baseline phase-2`。経路は `FittedGpr`（P2-8 typestate）。確保は `tests/alloc.rs` の `FittedGpr::predict_into` / `value_and_gradient_into` で上限 0。

### 等方 vs `phase-1b`

| グループ         | phase-1b | phase-2               | 変化（中央値）                                       |
| ---------------- | -------- | --------------------- | ---------------------------------------------------- |
| `kernel_rbf`     | 266 µs   | 123 µs (123.2–123.7)  | **−53.6%**                                           |
| `cholesky_alpha` | 159 µs   | 156 µs (155.6–157.3)  | −1.7%（ノイズ域。この経路は Phase 2 で触っていない） |
| `predict_100`    | 379 µs   | 265 µs (264.3–266.7)  | **−30.0%**                                           |
| `mll_and_grad`   | 1.42 ms  | 1.15 ms (1.144–1.148) | **−19.3%**                                           |
| `fit_lbfgs`      | 257 ms   | 337 ms (336.5–338.2)  | **+31.3%**                                           |

`kernel_rbf` / `predict_100` / `mll_and_grad` は Phase 2 のキャッシュ・Rayon・SIMD のまま 1b より短い。`fit_lbfgs` の +31.3% は 1 評価が遅くなったのではなく、独立乱数 `y` の平坦な尾根で L-BFGS の評価回数が増えた（当時 ~291、1b 当時はそれより少ない）。この表は `d = 8`・独立乱数 `y` の歴史記録。再測は次節。

### ARD Always vs Never（初回、d = 8・独立乱数 `y`）

等方とは比べない。P2-7 と同じ問題（ARD RBF、`ℓ_d = 1`、d = 8）。

| グループ           | Always                 | Never                  | Always vs Never             |
| ------------------ | ---------------------- | ---------------------- | --------------------------- |
| `mll_and_grad_ard` | 2.244 ms (2.229–2.259) | 2.103 ms (2.092–2.116) | **+6.7%**（Always が遅い）  |
| `fit_lbfgs_ard`    | 5.679 ms (5.660–5.720) | 4.751 ms (4.741–4.760) | **+19.5%**（Always が遅い） |

n = 256, d = 8 では Never が速い（P2-7 と同じ向き）。`Auto` は P5-5。

### 再測（Forrester / 球、この問題が以降の `phase-2`）

同一機械。`y` を独立乱数から外し、P1B-6 と同じ関数を n = 256 に伸ばした。等方は 1 次元 Forrester + `StandardizeTarget`（初期 `ℓ = 1`）。ARD は 2 次元重み付き球・16×16（初期 `ℓ_d = 4`。`ℓ_d = 1` では線探索が初手で止まる）。

`phase-2`: 現行コードで `cargo bench --bench exact -- --save-baseline phase-2`。`phase-1b` 再測: コミット `766b37c`（`task/25-phase-1b-baseline`）に同じ Forrester 入力を載せ、等方グループだけ測った。上の d = 8 表とは混ぜない。ARD は 1b にグループが無い。

L-BFGS の joint eval（Forrester）: 1b は 69 回（12 iter）、phase-2 は 26 回（11 iter）。終値 NLML は一致（−1.758）。勾配の ulp 差で線探索の歩数が変わる。壁時計の fit 比は速度差と読まない。

| グループ         | phase-1b（Forrester）         | phase-2                         | 変化（中央値）                                       |
| ---------------- | ----------------------------- | ------------------------------- | ---------------------------------------------------- |
| `kernel_rbf`     | 290 µs (289–292)              | 102 µs (100.9–102.5)            | **−65.0%**                                           |
| `cholesky_alpha` | 156 µs (155–157)              | 152 µs (150.9–152.7)            | −2.6%（ノイズ域。この経路は Phase 2 で触っていない） |
| `predict_100`    | 300 µs (298–301)              | 250 µs (248.1–251.6)            | **−16.7%**                                           |
| `mll_and_grad`   | 1.40 ms (1.394–1.407)         | 1.17 ms (1.153–1.199)           | **−16.1%**                                           |
| `fit_lbfgs`      | 98.5 ms (98.0–99.4) / 69 eval | 30.3 ms (30.10–30.50) / 26 eval | 壁時計 −69%。1 評価は 1.43 ms → **1.17 ms（−18%）**  |

| グループ           | Always                          | Never                           | 注                                              |
| ------------------ | ------------------------------- | ------------------------------- | ----------------------------------------------- |
| `mll_and_grad_ard` | 1.282 ms (1.280–1.285)          | 1.287 ms (1.277–1.303)          | 同等（ノイズ域）                                |
| `fit_lbfgs_ard`    | 86.2 ms (86.19–86.31) / 67 eval | 59.0 ms (58.86–59.03) / 46 eval | 壁時計は評価回数。1 評価はどちらも **~1.28 ms** |

1 評価の改善はフィットに乗る（回数を揃えたとき）。ARD の壁時計差（Always が長い）を速度差と読まない。1 評価は Always ≈ Never（d = 2）。`Auto` は P5-5。以降の比較基準はこの節の `phase-2`。

## P2B-16（他ライブラリ、`compare/perf/`）

同一機械。日付 2026-09-19。`just perf`（`uv run --directory compare/perf python run.py`）。criterion ではない。

問題: 等方 Forrester（`ℓ = 1`）と ARD 球（`ℓ_d = 4`）、`n = 256 / 1024 / 4096`（ARD は 16×16 / 32×32 / 64×64）。`y = f(x) + N(0, 1)`（NumPy Generator、Forrester seed `0`、球 seed `9`）。各セルは **同じ初期 θ で factor（最適化なし）** → **MLL+grad を 10 回**（同じ θ）→ predict 100 → ピーク RSS（Windows `PeakWorkingSet`）。ソルバは回さない。gprx は `Gpr<Fixed>::factor` + `value_and_gradient_into` + `StandardizeTarget`。sklearn は `optimizer=None` + `normalize_y=True` + `log_marginal_likelihood(..., eval_gradient=True)`。libgp は C++ 本体（`compare/perf/libgp/`）の `add_patterns` + `log_likelihood_gradient`。friedrich / libgp は runner 側で `y` を z-score。friedrich に ARD と公開 MLL+grad はない。

ゲート: sklearn は factor 時間・eval 時間・RSS がすべて小さい（5% 以内は判定不能）。libgp は ±10%。friedrich は factor 時間か RSS の一方。

負け（この節の計測時）: sklearn は n=4096 の 2 セル（eval が約 2 倍遅い。RSS は gprx が小さい）。libgp は走った 6 セルすべて（RSS が +10% 超。eval 時間は gprx の方が短い）。friedrich の等方 3 セルは pass。n=4096 eval は P2B-22 / [#143](https://github.com/YUKIKEDA/gprx/issues/143) で faer 並列のあと再判定。

| 問題      | n    | lib       | factor    | eval 10   | predict 100 | peak RSS   | ゲート                         |
| --------- | ---- | --------- | --------- | --------- | ----------- | ---------- | ------------------------------ |
| Forrester | 256  | gprx      | 1.34 ms   | 11.36 ms  | 0.46 ms     | 7.9 MiB    | —                              |
| Forrester | 256  | sklearn   | 3.61 ms   | 25.70 ms  | 0.72 ms     | 110.2 MiB  | pass                           |
| Forrester | 256  | libgp     | 1.30 ms   | 60.17 ms  | 0.93 ms     | 7.1 MiB    | fail（RSS +11%）               |
| Forrester | 256  | friedrich | 1.90 ms   | N/A       | 2.49 ms     | 4.9 MiB    | pass（時間）                   |
| Forrester | 1024 | gprx      | 16.35 ms  | 534.89 ms | 3.72 ms     | 41.7 MiB   | —                              |
| Forrester | 1024 | sklearn   | 146.54 ms | 1.315 s   | 3.34 ms     | 178.7 MiB  | pass                           |
| Forrester | 1024 | libgp     | 35.81 ms  | 2.075 s   | 9.59 ms     | 33.2 MiB   | fail（RSS +26%）               |
| Forrester | 1024 | friedrich | 58.55 ms  | N/A       | 26.97 ms    | 13.7 MiB   | pass（時間）                   |
| Forrester | 4096 | gprx      | 1.349 s   | 41.994 s  | 44.46 ms    | 535.1 MiB  | —                              |
| Forrester | 4096 | sklearn   | 1.206 s   | 21.214 s  | 34.92 ms    | 1150.5 MiB | fail（eval 約 2 倍）           |
| Forrester | 4096 | libgp     | 1.579 s   | 108.680 s | 374.82 ms   | 405.2 MiB  | fail（RSS +32%）               |
| Forrester | 4096 | friedrich | 5.079 s   | N/A       | 724.00 ms   | 138.6 MiB  | pass（時間）                   |
| 球 ARD    | 256  | gprx      | 1.58 ms   | 13.29 ms  | 0.81 ms     | 9.0 MiB    | —                              |
| 球 ARD    | 256  | sklearn   | 3.44 ms   | 52.51 ms  | 0.77 ms     | 110.5 MiB  | pass                           |
| 球 ARD    | 256  | libgp     | 1.45 ms   | 76.03 ms  | 1.19 ms     | 7.3 MiB    | fail（RSS +23%）               |
| 球 ARD    | 256  | friedrich | N/A       | N/A       | N/A         | N/A        | N/A                            |
| 球 ARD    | 1024 | gprx      | 18.44 ms  | 571.76 ms | 4.17 ms     | 57.8 MiB   | —                              |
| 球 ARD    | 1024 | sklearn   | 132.31 ms | 1.561 s   | 3.38 ms     | 186.8 MiB  | pass                           |
| 球 ARD    | 1024 | libgp     | 38.20 ms  | 2.388 s   | 11.78 ms    | 33.3 MiB   | fail（RSS +74%）               |
| 球 ARD    | 1024 | friedrich | N/A       | N/A       | N/A         | N/A        | N/A                            |
| 球 ARD    | 4096 | gprx      | 1.576 s   | 41.729 s  | 47.44 ms    | 791.4 MiB  | —                              |
| 球 ARD    | 4096 | sklearn   | 1.296 s   | 24.552 s  | 42.55 ms    | 1278.5 MiB | fail（eval 約 1.7 倍）         |
| 球 ARD    | 4096 | libgp     | 1.591 s   | 110.568 s | 385.57 ms   | 405.2 MiB  | fail（RSS +95%）               |
| 球 ARD    | 4096 | friedrich | N/A       | N/A       | N/A         | N/A        | N/A                            |

時間は同じ仕事（factor 1 回 + MLL+grad 10 回）で比較できる。libgp の eval は gprx より遅い。n=4096 の sklearn eval は（この節では）gprx より短い。RSS は sklearn 全セルで gprx が小さく、libgp / friedrich より大きい。改善行: RSS vs libgp は P2B-21 / [#142](https://github.com/YUKIKEDA/gprx/issues/142)。n=4096 eval vs sklearn は次節 P2B-22。

## P2B-22（faer 並列度、[#143](https://github.com/YUKIKEDA/gprx/issues/143)）

同一機械。日付 2026-09-19。ADR [`.dev/adr/0001-faer-parallel-degree.md`](adr/0001-faer-parallel-degree.md)。`faer_par(n) = min(プール, max(1, n/64))`。カーネルはプール全部。criterion は合否に使っていない。

対照: PR 冒頭の HEAD（faer `Par::Seq`）5 回平均と、実装後 5 回。`RAYON_NUM_THREADS` 未設定（16 論理）。

**n=256 / 1024 eval 10（逐次比、+10% まで）**

| 問題 | n | HEAD 逐次 | 実装後 | 比 | 判定 |
|---|---:|---:|---:|---:|---|
| Forrester | 256 | 12.45 ms | 12.22 ms | 0.98 | pass |
| sphere | 256 | 12.70 ms | 13.42 ms | 1.06 | pass |
| Forrester | 1024 | 530 ms | 208 ms | 0.39 | pass |
| sphere | 1024 | 547 ms | 288 ms | 0.53 | pass |

**n=256 factor vs 同セッション sklearn**

| 問題 | gprx | sklearn | 判定 |
|---|---:|---:|---|
| Forrester | 1.98 ms | 3.77 ms | pass |
| sphere | 2.22 ms | 8.83 ms | pass |

**n=4096 vs 同セッション sklearn（P2B-16 ゲート: factor・eval が小さい）**

| 問題 | 項目 | gprx（5 回平均） | sklearn | 判定 |
|---|---|---:|---:|---|
| Forrester | factor | 0.465 s | 1.221 s | pass |
| Forrester | eval 10 | 7.90 s | 20.45 s | pass |
| sphere | factor | 0.432 s | 1.320 s | pass |
| sphere | eval 10 | 8.73 s | 23.30 s | pass |

predict 100 は合否に使っていない（Forrester 4096: 55 ms、sklearn 42 ms）。

