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

同一機械。初回 2026-09-19（faer `Par::Seq`）。表は **2026-09-20、libgp の `set_loghyper` を eval の時計の外に出したあと `just perf` を 1 回通した値だけ**。セルの差し替えはない。`uv run --directory compare/perf python run.py`。criterion ではない。

時間は **捨て 1 回 + 中央値（括弧は min–max）**。回数の既定は `n ≤ 256` で 51、`n ≤ 1024` で 21、それ以外 7（`PERF_REPS` で上書き）。`eval 10` は 1 回の MLL+grad の中央 × 10。ピーク RSS は同じプロセスの `PeakWorkingSet`（前の factor は次を組む前に捨てる。8 個重ねると n=4096 で約 2 倍になる）。

問題: 等方 Forrester（`ℓ = 1`）と ARD 球（`ℓ_d = 4`）、`n = 256 / 1024 / 4096`（ARD は 16×16 / 32×32 / 64×64）。`y = f(x) + N(0, 1)`（NumPy Generator、Forrester seed `0`、球 seed `9`）。各セルは **同じ初期 θ で factor（最適化なし）** → joint MLL+grad → predict 100。ソルバは回さない。gprx は `Gpr<Fixed>::factor` + `value_and_gradient_into` + `StandardizeTarget`。sklearn は `optimizer=None` + `normalize_y=True` + `log_marginal_likelihood(..., eval_gradient=True)`。libgp は C++ 本体（`compare/perf/libgp/`）の `add_patterns` + `log_likelihood_gradient`。friedrich / libgp は runner 側で `y` を z-score。friedrich に ARD と公開 MLL+grad はない。

ゲートは中央値。sklearn は factor・eval・RSS がすべて小さい（5% 以内は判定不能）。libgp は +10% 超で fail。friedrich は factor 時間か RSS の一方。

負け: sklearn は 6 セルすべて pass。以前の一発表で球 n=256 factor が 4.39 vs 3.30 だったのは、捨て回なしの初回 Cholesky。今回の中央は 0.92 ms（0.84–1.38）対 2.16 ms（1.88–4.45）。libgp は走った 6 セルすべて fail（RSS が +10% 超。eval 時間は gprx の方が短い）。friedrich の等方 3 セルは pass。

| 問題      | n    | lib       | factor                   | eval 10                      | predict 100             | peak RSS   | ゲート           |
| --------- | ---- | --------- | ------------------------ | ---------------------------- | ----------------------- | ---------- | ---------------- |
| Forrester | 256  | gprx      | 0.86 ms（0.71–1.38）     | 11.40 ms（10.18–14.36）      | 0.41 ms（0.40–0.61）    | 9.6 MiB    | —                |
| Forrester | 256  | sklearn   | 2.28 ms（1.88–8.67）     | 26.59 ms（24.59–34.48）      | 0.40 ms（0.38–0.69）    | 110.9 MiB  | pass             |
| Forrester | 256  | libgp     | 1.28 ms（1.10–1.71）     | 63.95 ms（58.30–81.16）      | 1.02 ms（0.92–1.32）    | 7.2 MiB    | fail（RSS +33%） |
| Forrester | 256  | friedrich | 1.83 ms（1.71–2.35）     | N/A                          | 2.43 ms（2.27–3.78）    | 4.9 MiB    | pass（時間）     |
| Forrester | 1024 | gprx      | 17.10 ms（12.51–58.57）  | 232 ms（171–975）            | 2.53 ms（2.33–3.21）    | 43.6 MiB   | —                |
| Forrester | 1024 | sklearn   | 86.92 ms（59.35–152.24） | 1.226 s（704 ms–1.690 s）    | 2.32 ms（2.21–3.13）    | 179.8 MiB  | pass             |
| Forrester | 1024 | libgp     | 36.03 ms（33.08–39.57）  | 2.079 s（2.005–2.152）       | 10.05 ms（9.71–11.12）  | 33.2 MiB   | fail（RSS +31%） |
| Forrester | 1024 | friedrich | 57.92 ms（55.06–69.37）  | N/A                          | 25.90 ms（24.80–28.79） | 13.8 MiB   | pass（時間）     |
| Forrester | 4096 | gprx      | 463 ms（440–623）        | 7.267 s（7.097–8.408）       | 16.27 ms（15.42–17.14） | 537.2 MiB  | —                |
| Forrester | 4096 | sklearn   | 1.240 s（1.187–1.416）   | 20.574 s（20.189–20.957）    | 31.19 ms（26.29–33.29） | 1151.3 MiB | pass             |
| Forrester | 4096 | libgp     | 1.532 s（1.482–1.576）   | 105.594 s（105.055–105.857） | 344 ms（339–370）       | 405.2 MiB  | fail（RSS +33%） |
| Forrester | 4096 | friedrich | 4.965 s（4.927–5.132）   | N/A                          | 721 ms（716–733）       | 138.8 MiB  | pass（時間）     |
| 球 ARD    | 256  | gprx      | 0.92 ms（0.84–1.38）     | 12.52 ms（11.10–14.43）      | 0.41 ms（0.39–0.53）    | 10.7 MiB   | —                |
| 球 ARD    | 256  | sklearn   | 2.16 ms（1.88–4.45）     | 41.00 ms（38.38–58.47）      | 0.42 ms（0.37–0.60）    | 111.0 MiB  | pass             |
| 球 ARD    | 256  | libgp     | 1.27 ms（1.21–1.54）     | 74.46 ms（68.05–224.21）     | 0.97 ms（0.91–1.19）    | 7.2 MiB    | fail（RSS +49%） |
| 球 ARD    | 256  | friedrich | N/A                      | N/A                          | N/A                     | N/A        | N/A              |
| 球 ARD    | 1024 | gprx      | 14.91 ms（13.20–36.68）  | 189 ms（175–293）            | 2.12 ms（1.79–2.64）    | 59.7 MiB   | —                |
| 球 ARD    | 1024 | sklearn   | 99.96 ms（56.32–139.87） | 1.648 s（1.488–1.999）       | 2.36 ms（2.23–3.52）    | 187.8 MiB  | pass             |
| 球 ARD    | 1024 | libgp     | 37.40 ms（34.31–41.39）  | 2.293 s（2.215–2.753）       | 10.78 ms（9.81–13.41）  | 33.3 MiB   | fail（RSS +79%） |
| 球 ARD    | 1024 | friedrich | N/A                      | N/A                          | N/A                     | N/A        | N/A              |
| 球 ARD    | 4096 | gprx      | 407 ms（404–427）        | 7.982 s（7.271–9.534）       | 16.64 ms（16.16–19.13） | 793.4 MiB  | —                |
| 球 ARD    | 4096 | sklearn   | 1.170 s（1.155–1.399）   | 23.447 s（22.682–24.032）    | 31.14 ms（24.96–36.96） | 1279.5 MiB | pass             |
| 球 ARD    | 4096 | libgp     | 1.553 s（1.541–1.584）   | 105.275 s（105.082–106.291） | 341 ms（329–353）       | 405.2 MiB  | fail（RSS +96%） |
| 球 ARD    | 4096 | friedrich | N/A                      | N/A                          | N/A                     | N/A        | N/A              |

ゲートは中央値。gprx の n=4096 eval は sklearn より短い。libgp の eval は gprx より遅い。RSS は sklearn 全セルで gprx が小さく、libgp / friedrich より大きい。RSS vs libgp は P2B-21 / [#142](https://github.com/YUKIKEDA/gprx/issues/142)。HEAD 逐次との 5 回比は次節 P2B-22。

## P2B-22（faer 並列度、[#143](https://github.com/YUKIKEDA/gprx/issues/143)）

同一機械。日付 2026-09-19。ADR [`.dev/adr/0001-faer-parallel-degree.md`](adr/0001-faer-parallel-degree.md)。正方核は `faer_par(n) = min(プール, n/64)`。`n×k` ソルブはさらに `n·k/16384` と `k/12`。カーネルはプール全部。criterion は合否に使っていない。

対照: PR 冒頭の HEAD（faer `Par::Seq`）5 回平均と、実装後 5 回。`RAYON_NUM_THREADS` 未設定（16 論理）。

**n=256 / 1024 eval 10（逐次比、+10% まで）**

| 問題      |    n | HEAD 逐次 |   実装後 |   比 | 判定 |
| --------- | ---: | --------: | -------: | ---: | ---- |
| Forrester |  256 |  12.45 ms | 12.22 ms | 0.98 | pass |
| sphere    |  256 |  12.70 ms | 13.42 ms | 1.06 | pass |
| Forrester | 1024 |    530 ms |   208 ms | 0.39 | pass |
| sphere    | 1024 |    547 ms |   288 ms | 0.53 | pass |

**n=256 factor vs 同セッション sklearn**

| 問題      |    gprx | sklearn | 判定 |
| --------- | ------: | ------: | ---- |
| Forrester | 1.98 ms | 3.77 ms | pass |
| sphere    | 2.22 ms | 8.83 ms | pass |

**n=4096 vs 同セッション sklearn（P2B-16 ゲート: factor・eval が小さい）**

| 問題      | 項目    | gprx（5 回平均） | sklearn | 判定 |
| --------- | ------- | ---------------: | ------: | ---- |
| Forrester | factor  |          0.465 s | 1.221 s | pass |
| Forrester | eval 10 |           7.90 s | 20.45 s | pass |
| sphere    | factor  |          0.432 s | 1.320 s | pass |
| sphere    | eval 10 |           8.73 s | 23.30 s | pass |

**Forrester n=1024 predict 100（段階計時）**

`just perf` の一発は gprx 3.84 ms / sklearn 2.76 ms だった。プロセス 20 回だと gprx は 2.45–19.25 ms（stdev 5.1）。内訳は確保 ~0.44 ms・カーネル ~0.17 ms・平均/分散 ~0.05 ms で安定し、跳ぶのは `L⁻¹ k_*` だけ（1.5–22 ms）。原因は `faer_par(1024)` が 16 本で 1024×100 を割ること。n=4096 でも 16 本のまま 5 回中 1 回が 103 ms。Seq / 1 本は 2.4–3.0 ms、4 本は 1.6–2.1 ms、8 本は 1.5–1.8 ms。sklearn の `solve_triangular` は 1.6–2.4 ms、API 全体は 12 回で 2.69–3.86 ms（中央 2.88）。`faer_par_dims`（1024×100 は 6 本）後の gprx 20 回は 2.44–3.59 ms（中央 2.68、stdev 0.26）。n=4096 はさらに `k/12` で 8 本。

**球 ARD predict 100（`faer_par_dims` 後、同機）**

| n    |  回 |     gprx 中央（範囲） |  sklearn 中央（範囲） |
| ---- | --: | --------------------: | --------------------: |
| 256  |   5 |  0.63 ms（0.58–0.79） |  0.84 ms（0.77–1.06） |
| 1024 |   8 |  3.47 ms（3.28–3.94） |  2.94 ms（2.76–6.57） |
| 4096 |   6 | 21.04 ms（18.4–25.3） | 36.59 ms（33.8–42.2） |

n=1024 の中央は当初 sklearn が短い（3.47 vs 2.94）。段階計時ではソルブは gprx の方が短く（1.81 vs 2.18 ms）、差は `RbfArdKernel::apply_cross` の直列二重ループ（kernel 0.88 ms、等方 RBF は 0.21 ms、sklearn ARD は 0.86 ms）。矩形 ARD に SIMD+Rayon を足したあと kernel は 0.12 ms、プロセス 8 回は gprx 中央 2.76 ms / sklearn 2.82 ms。n=4096 は 16 本時の 41.45 ms から 21 ms。

## P2B-21（libgp 比ピーク RSS、[#142](https://github.com/YUKIKEDA/gprx/issues/142)）

同一機械。日付 2026-09-20。`just perf`（gprx 既定 `CachedDistances` と `--uncached` の `UncachedDistances` + `RetainCholesky`）。criterion ではない。CI なし。問題・回数・時計の取り方は P2B-16 と同じ。

合否は表2 の n=1024 / 4096（Forrester / 球）が libgp RSS ±10%。n=256 は記録。Uncached の時間は合否にしない。表1 の libgp 列は P2B-16 どおり RSS で fail（既定は距離キャッシュを持つ）。

初回の表印刷は runner が両方 `lib=gprx` と書いたため Uncached が既定を上書きした。下の表は同じ `results.json` をキー修正した値。以降は `run.py` が runner 名で `lib` を上書きする。

**表1 CachedDistances（既定）**

| 問題      | n    | lib       | factor                    | eval 10                      | predict 100             | peak RSS   | ゲート       |
| --------- | ---- | --------- | ------------------------- | ---------------------------- | ----------------------- | ---------- | ------------ |
| Forrester | 256  | gprx      | 0.87 ms（0.71–1.64）      | 11.50 ms（10.38–15.12）      | 0.45 ms（0.40–0.69）    | 10.1 MiB   | —            |
| Forrester | 256  | sklearn   | 2.23 ms（2.07–3.81）      | 25.63 ms（21.36–38.13）      | 0.43 ms（0.35–0.77）    | 110.8 MiB  | pass         |
| Forrester | 256  | libgp     | 1.25 ms（1.18–2.60）      | 61.15 ms（58.18–84.63）      | 0.99 ms（0.92–1.33）    | 7.2 MiB    | fail（RSS）  |
| Forrester | 256  | friedrich | 1.74 ms（1.65–2.44）      | N/A                          | 2.46 ms（2.09–2.77）    | 5.4 MiB    | pass（時間） |
| Forrester | 1024 | gprx      | 15.99 ms（11.04–60.76）   | 211 ms（178–328）            | 2.29 ms（1.99–2.94）    | 46.4 MiB   | —            |
| Forrester | 1024 | sklearn   | 93.33 ms（46.47–139.42）  | 1.221 s（749 ms–1.442 s）    | 2.76 ms（2.28–3.85）    | 179.4 MiB  | pass         |
| Forrester | 1024 | libgp     | 37.69 ms（35.46–60.89）   | 2.253 s（2.143–2.477）       | 10.36 ms（9.45–12.00）  | 33.2 MiB   | fail（RSS）  |
| Forrester | 1024 | friedrich | 61.93 ms（56.45–64.36）   | N/A                          | 28.60 ms（25.05–30.33） | 14.0 MiB   | pass（時間） |
| Forrester | 4096 | gprx      | 517 ms（464–633）         | 7.203 s（7.011–7.554）       | 16.87 ms（16.29–18.14） | 537.2 MiB  | —            |
| Forrester | 4096 | sklearn   | 1.197 s（1.143–1.247）    | 20.901 s（20.427–21.330）    | 31.43 ms（27.36–35.60） | 1151.0 MiB | pass         |
| Forrester | 4096 | libgp     | 1.525 s（1.512–1.584）    | 103.240 s（102.139–104.814） | 339 ms（334–357）       | 405.2 MiB  | fail（RSS）  |
| Forrester | 4096 | friedrich | 4.944 s（4.886–5.035）    | N/A                          | 705 ms（700–712）       | 138.8 MiB  | pass（時間） |
| 球 ARD    | 256  | gprx      | 0.89 ms（0.60–1.53）      | 12.22 ms（10.82–17.47）      | 0.41 ms（0.39–0.56）    | 10.6 MiB   | —            |
| 球 ARD    | 256  | sklearn   | 2.16 ms（1.87–4.87）      | 45.42 ms（42.02–59.04）      | 0.42 ms（0.37–0.73）    | 111.3 MiB  | pass         |
| 球 ARD    | 256  | libgp     | 1.29 ms（1.20–1.67）      | 71.64 ms（67.11–79.33）      | 0.97 ms（0.92–1.25）    | 7.2 MiB    | fail（RSS）  |
| 球 ARD    | 256  | friedrich | N/A                       | N/A                          | N/A                     | N/A        | N/A          |
| 球 ARD    | 1024 | gprx      | 14.56 ms（12.73–29.80）   | 188 ms（171–297）            | 2.17 ms（2.01–2.69）    | 59.6 MiB   | —            |
| 球 ARD    | 1024 | sklearn   | 103.70 ms（47.73–156.82） | 1.633 s（1.425–1.895）       | 2.40 ms（2.33–3.03）    | 187.8 MiB  | pass         |
| 球 ARD    | 1024 | libgp     | 37.50 ms（34.48–58.78）   | 2.272 s（2.223–2.348）       | 10.12 ms（9.65–10.91）  | 33.3 MiB   | fail（RSS）  |
| 球 ARD    | 1024 | friedrich | N/A                       | N/A                          | N/A                     | N/A        | N/A          |
| 球 ARD    | 4096 | gprx      | 431 ms（413–470）         | 7.476 s（7.194–9.224）       | 15.95 ms（15.56–17.04） | 793.5 MiB  | —            |
| 球 ARD    | 4096 | sklearn   | 1.193 s（1.157–1.362）    | 23.799 s（23.600–26.137）    | 29.25 ms（27.12–32.30） | 1279.5 MiB | pass         |
| 球 ARD    | 4096 | libgp     | 1.549 s（1.517–1.645）    | 109.033 s（106.614–112.393） | 359 ms（349–455）       | 405.2 MiB  | fail（RSS）  |
| 球 ARD    | 4096 | friedrich | N/A                       | N/A                          | N/A                     | N/A        | N/A          |

**表2 UncachedDistances + RetainCholesky（P2B-21 合否）**

| 問題      | n    | lib           | factor                  | eval 10                      | predict 100             | peak RSS  | vs libgp RSS   |
| --------- | ---- | ------------- | ----------------------- | ---------------------------- | ----------------------- | --------- | -------------- |
| Forrester | 256  | gprx-uncached | 0.87 ms（0.73–1.35）    | 20.47 ms（17.90–25.45）      | 0.49 ms（0.40–0.76）    | 9.1 MiB   | —              |
| Forrester | 256  | libgp         | 1.25 ms（1.18–2.60）    | 61.15 ms（58.18–84.63）      | 0.99 ms（0.92–1.33）    | 7.2 MiB   | record（+27%） |
| Forrester | 1024 | gprx-uncached | 21.89 ms（13.96–58.65） | 317 ms（286–667）            | 2.20 ms（2.03–2.82）    | 35.6 MiB  | —              |
| Forrester | 1024 | libgp         | 37.69 ms（35.46–60.89） | 2.253 s（2.143–2.477）       | 10.36 ms（9.45–12.00）  | 33.2 MiB  | pass（+7%）    |
| Forrester | 4096 | gprx-uncached | 429 ms（403–511）       | 9.386 s（9.131–10.082）      | 18.59 ms（16.75–22.69） | 409.3 MiB | —              |
| Forrester | 4096 | libgp         | 1.525 s（1.512–1.584）  | 103.240 s（102.139–104.814） | 339 ms（334–357）       | 405.2 MiB | pass（+1%）    |
| 球 ARD    | 256  | gprx-uncached | 0.60 ms（0.51–1.15）    | 12.03 ms（10.05–16.29）      | 0.41 ms（0.39–0.61）    | 9.0 MiB   | —              |
| 球 ARD    | 256  | libgp         | 1.29 ms（1.20–1.67）    | 71.64 ms（67.11–79.33）      | 0.97 ms（0.92–1.25）    | 7.2 MiB   | record（+26%） |
| 球 ARD    | 1024 | gprx-uncached | 9.24 ms（7.94–23.96）   | 174 ms（156–309）            | 2.16 ms（1.98–2.98）    | 35.6 MiB  | —              |
| 球 ARD    | 1024 | libgp         | 37.50 ms（34.48–58.78） | 2.272 s（2.223–2.348）       | 10.12 ms（9.65–10.91）  | 33.3 MiB  | pass（+7%）    |
| 球 ARD    | 4096 | gprx-uncached | 345 ms（323–433）       | 7.371 s（7.247–8.134）       | 16.10 ms（15.37–18.57） | 409.4 MiB | —              |
| 球 ARD    | 4096 | libgp         | 1.549 s（1.517–1.645）  | 109.033 s（106.614–112.393） | 359 ms（349–455）       | 405.2 MiB | pass（+1%）    |

ゲート 4 セルはすべて pass。n=4096 の削減は距離テンソルと一致する（等方は `n×n` の 128 MiB、537.2 → 409.3。ARD はさらに `n×(n·2)` の 256 MiB、793.5 → 409.4）。Uncached の eval は等方で既定より長い（n=4096 で 7.20 s → 9.39 s）。ARD は同程度。

## P2B-23（速さ / メモリのプリセット、[#148](https://github.com/YUKIKEDA/gprx/issues/148)）

同一機械。日付 2026-09-20。下の表は同日の 2 回目（初回は Forrester n=1024 / 4096 の eval が P2B-21 より長く、負荷の振れと見て取り直し）。`just perf`（gprx 既定 = 速さ極 `CachedDistances` + `RetainCholesky`、表2は `with_prefer_memory` = `UncachedDistances` + `ReuseCholesky`）。criterion ではない。CI なし。問題・回数・時計の取り方は P2B-16 と同じ。新しい RSS ゲートは置かない。上の P2B-21 Uncached+Retain 表はそのまま。

表1 の libgp 列は P2B-16 どおり RSS で fail（既定は距離キャッシュと専用 `W` を持つ）。sklearn / friedrich の時間ゲートは pass。

**表1 CachedDistances + RetainCholesky（速さ極 / 既定）**

| 問題      | n    | lib       | factor                   | eval 10                      | predict 100             | peak RSS   | ゲート       |
| --------- | ---- | --------- | ------------------------ | ---------------------------- | ----------------------- | ---------- | ------------ |
| Forrester | 256  | gprx      | 0.53 ms（0.43–1.07）     | 10.28 ms（8.78–13.78）       | 0.42 ms（0.39–0.64）    | 9.6 MiB    | —            |
| Forrester | 256  | sklearn   | 2.16 ms（1.86–4.33）     | 26.01 ms（24.37–34.33）      | 0.40 ms（0.36–0.86）    | 110.9 MiB  | pass         |
| Forrester | 256  | libgp     | 1.28 ms（1.13–1.80）     | 59.64 ms（56.64–70.69）      | 0.87 ms（0.84–1.33）    | 7.2 MiB    | fail（RSS）  |
| Forrester | 256  | friedrich | 1.65 ms（1.54–2.41）     | N/A                          | 2.18 ms（2.07–2.77）    | 4.9 MiB    | pass（時間） |
| Forrester | 1024 | gprx      | 14.47 ms（10.89–51.91）  | 180 ms（164–262）            | 2.20 ms（1.99–2.66）    | 43.5 MiB   | —            |
| Forrester | 1024 | sklearn   | 117.0 ms（53.23–140.04） | 987 ms（739–1.644 s）        | 2.55 ms（2.35–3.38）    | 179.4 MiB  | pass         |
| Forrester | 1024 | libgp     | 35.89 ms（33.58–37.69）  | 2.104 s（2.014–2.221）       | 9.69 ms（9.09–12.41）   | 33.2 MiB   | fail（RSS）  |
| Forrester | 1024 | friedrich | 57.62 ms（55.53–61.72）  | N/A                          | 25.41 ms（24.60–29.12） | 13.8 MiB   | pass（時間） |
| Forrester | 4096 | gprx      | 455 ms（446–464）        | 7.252 s（7.168–9.159）       | 16.46 ms（15.49–16.81） | 537.3 MiB  | —            |
| Forrester | 4096 | sklearn   | 1.235 s（1.211–1.390）   | 20.641 s（20.429–21.101）    | 31.00 ms（27.26–32.51） | 1151.7 MiB | pass         |
| Forrester | 4096 | libgp     | 1.530 s（1.507–1.540）   | 103.360 s（102.534–105.022） | 342 ms（334–348）       | 405.2 MiB  | fail（RSS）  |
| Forrester | 4096 | friedrich | 4.939 s（4.891–4.991）   | N/A                          | 709 ms（702–728）       | 138.8 MiB  | pass（時間） |
| 球 ARD    | 256  | gprx      | 1.03 ms（0.85–1.41）     | 12.60 ms（11.71–15.26）      | 0.41 ms（0.38–0.56）    | 10.6 MiB   | —            |
| 球 ARD    | 256  | sklearn   | 2.19 ms（2.05–2.87）     | 46.20 ms（42.58–59.72）      | 0.42 ms（0.37–0.58）    | 111.2 MiB  | pass         |
| 球 ARD    | 256  | libgp     | 1.28 ms（1.19–1.72）     | 71.63 ms（67.99–78.65）      | 0.94 ms（0.90–1.25）    | 7.2 MiB    | fail（RSS）  |
| 球 ARD    | 256  | friedrich | N/A                      | N/A                          | N/A                     | N/A        | N/A          |
| 球 ARD    | 1024 | gprx      | 15.04 ms（13.90–22.84）  | 185 ms（170–289）            | 2.19 ms（1.99–2.74）    | 59.6 MiB   | —            |
| 球 ARD    | 1024 | sklearn   | 112.4 ms（49.55–150.41） | 1.659 s（1.275–2.054）       | 2.50 ms（2.25–3.12）    | 187.7 MiB  | pass         |
| 球 ARD    | 1024 | libgp     | 36.59 ms（34.12–64.10）  | 2.285 s（2.185–2.370）       | 10.31 ms（9.69–10.84）  | 33.3 MiB   | fail（RSS）  |
| 球 ARD    | 1024 | friedrich | N/A                      | N/A                          | N/A                     | N/A        | N/A          |
| 球 ARD    | 4096 | gprx      | 413 ms（397–484）        | 7.876 s（7.329–9.352）       | 15.73 ms（14.97–16.35） | 793.5 MiB  | —            |
| 球 ARD    | 4096 | sklearn   | 1.189 s（1.154–1.267）   | 23.332 s（23.174–23.969）    | 33.92 ms（28.11–37.61） | 1279.4 MiB | pass         |
| 球 ARD    | 4096 | libgp     | 1.590 s（1.565–1.605）   | 105.144 s（105.014–108.291） | 338 ms（331–345）       | 405.2 MiB  | fail（RSS）  |
| 球 ARD    | 4096 | friedrich | N/A                      | N/A                          | N/A                     | N/A        | N/A          |

**表2 `with_prefer_memory`（UncachedDistances + ReuseCholesky、記録のみ）**

| 問題      | n    | lib         | factor                  | eval 10                      | predict 100             | peak RSS  | vs libgp RSS |
| --------- | ---- | ----------- | ----------------------- | ---------------------------- | ----------------------- | --------- | ------------ |
| Forrester | 256  | gprx-memory | 0.81 ms（0.56–1.56）    | 28.41 ms（23.96–33.96）      | 0.43 ms（0.40–0.70）    | 8.7 MiB   | record       |
| Forrester | 256  | libgp       | 1.28 ms（1.13–1.80）    | 59.64 ms（56.64–70.69）      | 0.87 ms（0.84–1.33）    | 7.2 MiB   | record       |
| Forrester | 1024 | gprx-memory | 15.85 ms（13.65–21.41） | 448 ms（366–1.131 s）        | 2.24 ms（2.10–3.16）    | 27.7 MiB  | record       |
| Forrester | 1024 | libgp       | 35.89 ms（33.58–37.69） | 2.104 s（2.014–2.221）       | 9.69 ms（9.09–12.41）   | 33.2 MiB  | record       |
| Forrester | 4096 | gprx-memory | 410 ms（391–417）       | 13.050 s（12.551–14.607）    | 16.18 ms（14.77–17.28） | 281.3 MiB | record       |
| Forrester | 4096 | libgp       | 1.530 s（1.507–1.540）  | 103.360 s（102.534–105.022） | 342 ms（334–348）       | 405.2 MiB | record       |
| 球 ARD    | 256  | gprx-memory | 0.51 ms（0.44–3.25）    | 16.73 ms（13.98–28.90）      | 0.41 ms（0.39–0.60）    | 8.6 MiB   | record       |
| 球 ARD    | 256  | libgp       | 1.28 ms（1.19–1.72）    | 71.63 ms（67.99–78.65）      | 0.94 ms（0.90–1.25）    | 7.2 MiB   | record       |
| 球 ARD    | 1024 | gprx-memory | 8.04 ms（6.62–13.16）   | 259 ms（199–561）            | 2.18 ms（2.01–2.63）    | 27.7 MiB  | record       |
| 球 ARD    | 1024 | libgp       | 36.59 ms（34.12–64.10） | 2.285 s（2.185–2.370）       | 10.31 ms（9.69–10.84）  | 33.3 MiB  | record       |
| 球 ARD    | 4096 | gprx-memory | 311 ms（305–332）       | 9.903 s（9.827–11.242）      | 16.09 ms（15.48–16.56） | 281.5 MiB | record       |
| 球 ARD    | 4096 | libgp       | 1.590 s（1.565–1.605）  | 105.144 s（105.014–108.291） | 338 ms（331–345）       | 405.2 MiB | record       |

速さ極の Forrester n=1024 / 4096 は P2B-21（factor 16.0 ms / eval 211 ms / 7.20 s）と同程度（14.5 ms / 180 ms / 7.25 s）。初回の 41.6 ms / 571 ms / 8.47 s は残していない。球 n=4096 eval は 7.48 s → 7.88 s（範囲 7.33–9.35）。n=4096 のメモリ極 RSS は P2B-21 Uncached+Retain（409.3 / 409.4 MiB）からさらに専用 `W`（`n×n` の 128 MiB）を外し、281.3 / 281.5 MiB。libgp 405.2 より小さい。eval は等方で既定より長い（n=4096 で 7.25 s → 13.05 s）。ARD も長い（7.88 s → 9.90 s）。criterion は合否にしない。

## P3-6（online insert vs libgp、[#176](https://github.com/YUKIKEDA/gprx/issues/176)）

同一機械。日付 2026-09-20。`just perf-online`。生の `y`（`StandardizeTarget` なし、libgp も z-score なし）。時計は `n = 2` まで組んだあと、3 点目から `n` までの `insert` / `add_pattern`。捨て 1 回 + 中央値（括弧は min–max）。回数は P2B-16 と同じ（`n ≤ 256` で 51、`n ≤ 1024` で 21、それ以外 7）。

ゲートは `n = 256 / 1024` の中央値が libgp 以下（5% 以内は判定不能）。`4096` と RSS は記録。4 ゲートセルすべて fail。改善は P3-7 / [#177](https://github.com/YUKIKEDA/gprx/issues/177)。

| 問題      | n    | lib   | insert n=2→n               | peak RSS  | ゲート |
| --------- | ---- | ----- | -------------------------- | --------- | ------ |
| Forrester | 256  | gprx  | 16.10 ms（14.65–17.84）    | 7.7 MiB   | fail   |
| Forrester | 256  | libgp | 1.39 ms（1.20–1.90）       | 5.8 MiB   | —      |
| Forrester | 1024 | gprx  | 202.60 ms（200.57–206.03） | 36.8 MiB  | fail   |
| Forrester | 1024 | libgp | 47.01 ms（42.38–52.85）    | 24.5 MiB  | —      |
| Forrester | 4096 | gprx  | 12.844 s（12.756–12.935）  | 487.4 MiB | record |
| Forrester | 4096 | libgp | 3.846 s（3.801–3.918）     | 265.4 MiB | —      |
| 球 ARD    | 256  | gprx  | 13.77 ms（8.67–14.56）     | 8.1 MiB   | fail   |
| 球 ARD    | 256  | libgp | 1.87 ms（1.73–2.26）       | 5.9 MiB   | —      |
| 球 ARD    | 1024 | gprx  | 190.85 ms（188.24–196.64） | 37.5 MiB  | fail   |
| 球 ARD    | 1024 | libgp | 49.69 ms（46.43–53.22）    | 24.6 MiB  | —      |
| 球 ARD    | 4096 | gprx  | 12.828 s（12.758–12.931）  | 488.5 MiB | record |
| 球 ARD    | 4096 | libgp | 3.928 s（3.822–4.000）     | 265.5 MiB | —      |

数値照合（Forrester / 球の先頭 32 点、各段階の平均・観測分散・NLML）は相対 `1e-8` で pass。`just test` はコミット済み JSON を読む。criterion は合否にしない。

## P3-7（online insert 高速化、[#177](https://github.com/YUKIKEDA/gprx/issues/177)）

同一機械。日付 2026-09-20。`just perf-online`。P3-6 と同じ時計。insert は bordered LDLT と `v_buf` 再利用、1 列の距離 / RBF は逐次、`α` は insert / delete とも読み出しまで遅延。delete の faer スクラッチは `OnlineWorkspace` に置いて再利用する。

ゲートは `n = 256 / 1024` の中央値が libgp 以下（5% 以内は判定不能）。`OnlineWorkspace` は LD / `y` / `α` / `v_buf` だけ伸ばす（予測が読まない `K` と距離キャッシュは置かない）。4 ゲートセルは pass。RSS は 1024 / 4096 で libgp より小さい。`4096` の時間は記録。criterion は合否にしない。

自前 `f64x4` 単位下三角は Forrester 1024 で faer より遅い（61 ms 対 45 ms）ので入れない。1 列の Rayon と LLT 全面乗り換えは測ったうえで不採用（delete / persist が LDLT）。

| 問題      | n    | lib   | insert n=2→n            | peak RSS  | ゲート |
| --------- | ---- | ----- | ----------------------- | --------- | ------ |
| Forrester | 256  | gprx  | 1.03 ms（0.99–1.67）    | 5.8 MiB   | pass   |
| Forrester | 256  | libgp | 1.43 ms（1.24–1.88）    | 5.9 MiB   | —      |
| Forrester | 1024 | gprx  | 44.83 ms（40.64–57.52） | 15.4 MiB  | pass   |
| Forrester | 1024 | libgp | 47.93 ms（44.94–49.79） | 24.7 MiB  | —      |
| Forrester | 4096 | gprx  | 4.493 s（4.107–7.086）  | 166.0 MiB | record |
| Forrester | 4096 | libgp | 3.845 s（3.748–3.887）  | 265.3 MiB | —      |
| 球 ARD    | 256  | gprx  | 1.17 ms（1.12–1.41）    | 5.9 MiB   | pass   |
| 球 ARD    | 256  | libgp | 1.51 ms（1.34–1.70）    | 5.9 MiB   | —      |
| 球 ARD    | 1024 | gprx  | 44.29 ms（40.33–50.59） | 15.5 MiB  | pass   |
| 球 ARD    | 1024 | libgp | 48.30 ms（46.68–71.28） | 24.5 MiB  | —      |
| 球 ARD    | 4096 | gprx  | 4.121 s（4.105–4.182）  | 166.2 MiB | record |
| 球 ARD    | 4096 | libgp | 3.972 s（3.895–4.047）  | 265.7 MiB | —      |

同一機械。日付 2026-09-20。`just perf-online-delete`。時計は n まで insert（計時外）のあと末尾 `PointId` を 1 点ずつ消して n→2。比較相手なし。前は毎回 α 再ソルブと都度 `MemBuffer`。後は遅延 α とスクラッチ再利用。後の方が 256 で約 5 倍、1024 で約 11 倍、4096 で約 40 倍短いので採用する。

| 問題      | n    | 版  | delete n→2                 | peak RSS  |
| --------- | ---- | --- | -------------------------- | --------- |
| Forrester | 256  | 前  | 1.75 ms（1.66–2.11）       | 6.4 MiB   |
| Forrester | 256  | 後  | 0.32 ms（0.30–0.51）       | 6.2 MiB   |
| Forrester | 1024 | 前  | 78.35 ms（74.75–116.22）   | 16.7 MiB  |
| Forrester | 1024 | 後  | 6.68 ms（5.73–9.63）       | 16.6 MiB  |
| Forrester | 4096 | 前  | 8.216 s（8.162–8.367）     | 168.0 MiB |
| Forrester | 4096 | 後  | 181.08 ms（169.70–184.11） | 166.9 MiB |
| 球 ARD    | 256  | 前  | 1.86 ms（1.68–2.14）       | 6.2 MiB   |
| 球 ARD    | 256  | 後  | 0.37 ms（0.36–0.53）       | 6.0 MiB   |
| 球 ARD    | 1024 | 前  | 79.10 ms（73.02–90.44）    | 16.5 MiB  |
| 球 ARD    | 1024 | 後  | 7.71 ms（6.04–9.33）       | 15.8 MiB  |
| 球 ARD    | 4096 | 前  | 8.167 s（8.103–8.259）     | 166.6 MiB |
| 球 ARD    | 4096 | 後  | 192.50 ms（178.34–205.79） | 166.3 MiB |

## P4-12（Sparse 時間・RSS、[#197](https://github.com/YUKIKEDA/gprx/issues/197)）

同一機械。日付 2026-09-22。`just perf-sparse`。`y` は母集団 σ で標準化。`m = 16`（生成時 k-means、seed `0`）。全部 CPU。時計は P2B-16 と同じ（捨て 1 回 + 中央値。`n ≤ 256` で 51、`n ≤ 1024` で 21、それ以外 7。eval は 10 × 1 回の中央値）。Exact の `Gpr` 列は無い。GPflow は置かない。

ゲートは書けた相手すべてで factor / joint / predict とピーク RSS が小さい（5% 以内は判定不能）。時間負けは P4-18 / [#209](https://github.com/YUKIKEDA/gprx/issues/209)。RSS 負けは P4-19 / [#210](https://github.com/YUKIKEDA/gprx/issues/210)。criterion は合否にしない。

| 面   | 問題      | n    | lib      | factor    | eval N    | evals | predict 100 | peak RSS  | vs gprx  |
| ---- | --------- | ---- | -------- | --------- | --------- | ----- | ----------- | --------- | -------- |
| sgpr | forrester | 256  | gprx     | 0.68 ms   | 10.51 ms  | 10    | 0.10 ms     | 6.2 MiB   | -        |
| sgpr | forrester | 256  | gpytorch | 554.94 ms | 2.766 s   | 10    | 170.11 ms   | 246.9 MiB | pass     |
| sgpr | forrester | 256  | gpy      | 12.62 ms  | 18.05 ms  | 10    | 0.19 ms     | 115.6 MiB | pass     |
| svgp | forrester | 256  | gprx     | 0.65 ms   | 9.36 ms   | 10    | 0.11 ms     | 6.1 MiB   | -        |
| svgp | forrester | 256  | gpytorch | 527.18 ms | 1.841 s   | 10    | 83.09 ms    | 246.3 MiB | pass     |
| svgp | forrester | 256  | gpy      | 10.90 ms  | 10.84 ms  | 10    | 0.29 ms     | 116.0 MiB | pass     |
| sgpr | forrester | 1024 | gprx     | 1.36 ms   | 81.23 ms  | 10    | 0.12 ms     | 21.6 MiB  | -        |
| sgpr | forrester | 1024 | gpytorch | 556.19 ms | 3.978 s   | 10    | 300.36 ms   | 247.8 MiB | pass     |
| sgpr | forrester | 1024 | gpy      | 16.29 ms  | 20.26 ms  | 10    | 0.20 ms     | 117.0 MiB | fail     |
| svgp | forrester | 1024 | gprx     | 0.76 ms   | 74.72 ms  | 10    | 0.13 ms     | 21.5 MiB  | -        |
| svgp | forrester | 1024 | gpytorch | 663.92 ms | 2.370 s   | 10    | 34.29 ms    | 246.6 MiB | pass     |
| svgp | forrester | 1024 | gpy      | 15.63 ms  | 21.24 ms  | 10    | 0.66 ms     | 117.1 MiB | fail     |
| sgpr | forrester | 4096 | gprx     | 1.83 ms   | 1.052 s   | 10    | 0.26 ms     | 263.4 MiB | -        |
| sgpr | forrester | 4096 | gpytorch | 741.54 ms | 5.925 s   | 10    | 763.64 ms   | 251.1 MiB | 判定不能 |
| sgpr | forrester | 4096 | gpy      | 50.18 ms  | 45.32 ms  | 10    | 0.24 ms     | 123.0 MiB | fail     |
| svgp | forrester | 4096 | gprx     | 1.60 ms   | 952.15 ms | 10    | 0.43 ms     | 263.0 MiB | -        |
| svgp | forrester | 4096 | gpytorch | 974.55 ms | 7.112 s   | 10    | 108.23 ms   | 249.4 MiB | fail     |
| svgp | forrester | 4096 | gpy      | 24.82 ms  | 33.47 ms  | 10    | 0.44 ms     | 123.0 MiB | fail     |
| sgpr | sphere    | 256  | gprx     | 0.84 ms   | 16.51 ms  | 10    | 0.08 ms     | 6.2 MiB   | -        |
| sgpr | sphere    | 256  | gpytorch | 459.22 ms | 76.98 ms  | 10    | 423.97 ms   | 246.1 MiB | pass     |
| sgpr | sphere    | 256  | gpy      | 13.86 ms  | 19.41 ms  | 10    | 0.22 ms     | 115.9 MiB | pass     |
| svgp | sphere    | 256  | gprx     | 0.60 ms   | 12.95 ms  | 10    | 0.23 ms     | 6.1 MiB   | -        |
| svgp | sphere    | 256  | gpytorch | 573.35 ms | 2.343 s   | 10    | 108.32 ms   | 246.4 MiB | pass     |
| svgp | sphere    | 256  | gpy      | 11.11 ms  | 11.28 ms  | 10    | 0.24 ms     | 115.8 MiB | fail     |
| sgpr | sphere    | 1024 | gprx     | 0.78 ms   | 135.01 ms | 10    | 0.09 ms     | 21.8 MiB  | -        |
| sgpr | sphere    | 1024 | gpytorch | 629.68 ms | 74.32 ms  | 10    | 112.38 ms   | 248.2 MiB | fail     |
| sgpr | sphere    | 1024 | gpy      | 15.76 ms  | 22.37 ms  | 10    | 0.21 ms     | 117.2 MiB | fail     |
| svgp | sphere    | 1024 | gprx     | 0.81 ms   | 136.05 ms | 10    | 0.10 ms     | 21.5 MiB  | -        |
| svgp | sphere    | 1024 | gpytorch | 477.15 ms | 2.998 s   | 10    | 114.84 ms   | 246.7 MiB | pass     |
| svgp | sphere    | 1024 | gpy      | 12.79 ms  | 14.83 ms  | 10    | 0.24 ms     | 117.4 MiB | fail     |
| sgpr | sphere    | 4096 | gprx     | 1.56 ms   | 1.807 s   | 10    | 0.14 ms     | 264.0 MiB | -        |
| sgpr | sphere    | 4096 | gpytorch | 590.70 ms | 9.462 s   | 10    | 947.17 ms   | 251.2 MiB | fail     |
| sgpr | sphere    | 4096 | gpy      | 29.74 ms  | 40.20 ms  | 10    | 0.21 ms     | 123.2 MiB | fail     |
| svgp | sphere    | 4096 | gprx     | 1.21 ms   | 1.955 s   | 10    | 0.20 ms     | 263.0 MiB | -        |
| svgp | sphere    | 4096 | gpytorch | 617.85 ms | 5.931 s   | 10    | 97.22 ms    | 251.2 MiB | 判定不能 |
| svgp | sphere    | 4096 | gpy      | 29.29 ms  | 33.18 ms  | 10    | 0.28 ms     | 122.9 MiB | fail     |

時間 fail の主因は joint（GPy が n≥1024 で短い。GPyTorch は球 SGPR n=1024 の eval が短い）。n=4096 の RSS は gprx 約 263 MiB、GPy 約 123 MiB、GPyTorch 約 250 MiB。factor は全セルで gprx が短い。

## P4-14（Sparse オンライン時間、[#199](https://github.com/YUKIKEDA/gprx/issues/199)）

同一機械。日付 2026-09-23。`just perf-sparse-online`。生の `y`。`m_max = 16`（生成時 k-means、seed `0`）。初期はプレフィックス `start_n = 32/128/512`・`start_m = 8`。32 操作はハーネス生成（`ops_seed = 0`、RBF probe の PD フィルタ。P4-13 の JSON は読まない）。全部 CPU。プレフィックスは計時外。時計は 32 手を 1 本の壁時計（捨て 1 回 + 中央値。`n ≤ 256` で 51、`n ≤ 1024` で 21、それ以外 7）。predict / NLML は時計に入れない。GPy は置かない。

ゲートは 3 段×2 問題すべてで増分の中央値が自前 `Sgpr<Fixed>::factor` フルより小さく、かつ GPyTorch Titsias 組み立て（クエリなし）より小さい（5% 以内は判定不能）。RSS は記録。6 セルは pass。改善行は足さない。criterion は合否にしない。

| 問題      | n    | lib       | ops 32  | peak RSS  | vs inc |
| --------- | ---- | --------- | ------- | --------- | ------ |
| Forrester | 256  | gprx-inc  | 1.49 ms | 5.3 MiB   | -      |
| Forrester | 256  | gprx-full | 2.56 ms | 5.1 MiB   | pass   |
| Forrester | 256  | gpytorch  | 1.900 s | 208.7 MiB | pass   |
| Forrester | 1024 | gprx-inc  | 1.65 ms | 5.3 MiB   | -      |
| Forrester | 1024 | gprx-full | 2.89 ms | 5.2 MiB   | pass   |
| Forrester | 1024 | gpytorch  | 2.704 s | 207.4 MiB | pass   |
| Forrester | 4096 | gprx-inc  | 5.71 ms | 5.7 MiB   | -      |
| Forrester | 4096 | gprx-full | 6.53 ms | 5.5 MiB   | pass   |
| Forrester | 4096 | gpytorch  | 2.896 s | 208.9 MiB | pass   |
| 球 ARD    | 256  | gprx-inc  | 1.47 ms | 5.3 MiB   | -      |
| 球 ARD    | 256  | gprx-full | 2.01 ms | 5.2 MiB   | pass   |
| 球 ARD    | 256  | gpytorch  | 1.834 s | 207.4 MiB | pass   |
| 球 ARD    | 1024 | gprx-inc  | 1.87 ms | 5.4 MiB   | -      |
| 球 ARD    | 1024 | gprx-full | 2.51 ms | 5.3 MiB   | pass   |
| 球 ARD    | 1024 | gpytorch  | 2.922 s | 207.5 MiB | pass   |
| 球 ARD    | 4096 | gprx-inc  | 5.62 ms | 5.9 MiB   | -      |
| 球 ARD    | 4096 | gprx-full | 6.09 ms | 5.8 MiB   | pass   |
| 球 ARD    | 4096 | gpytorch  | 2.619 s | 209.4 MiB | pass   |

増分は全セルで自前フルより短い。GPyTorch は秒オーダー（Python + 毎回の kernel 密行列）。RSS は記録のみ（gprx 約 5–6 MiB、GPyTorch 約 208 MiB）。

## P4-18（Sparse joint 時間、[#209](https://github.com/YUKIKEDA/gprx/issues/209)）

同一機械。日付 2026-09-27。`just perf-sparse`。`K(X,X)` の勾配とヘッセは対角を `O(n)` で足す。`K(Z,Z)` と `K(Z,X)` の密勾配はそのまま。eval は 10 回。factor / predict / RSS は記録。criterion は合否にしない。

ゲートは P4-12 で joint が GPy より長かったセルと、球 SGPR `n=1024` の GPyTorch eval。5% 以内は判定不能。すでに joint で勝っていたセルは、2026-09-22 の自前中央値より 5% を超えて遅くしない。残った joint 負けは P4-20 / [#214](https://github.com/YUKIKEDA/gprx/issues/214)。

| 面   | 問題      | n    | lib      | factor   | eval N    | evals | predict 100 | peak RSS  | joint |
| ---- | --------- | ---- | -------- | -------- | --------- | ----- | ----------- | --------- | ----- |
| sgpr | forrester | 256  | gprx     | 0.12 ms  | 4.74 ms   | 10    | 0.05 ms     | 6.4 MiB   | -     |
| sgpr | forrester | 256  | gpytorch | 4.50 ms  | 77.01 ms  | 10    | 4.31 ms     | 247.5 MiB | 記録  |
| sgpr | forrester | 256  | gpy      | 9.36 ms  | 13.34 ms  | 10    | 0.10 ms     | 135.2 MiB | 記録  |
| svgp | forrester | 256  | gprx     | 0.11 ms  | 3.91 ms   | 10    | 0.06 ms     | 5.7 MiB   | -     |
| svgp | forrester | 256  | gpytorch | 3.21 ms  | 44.35 ms  | 10    | 1.77 ms     | 244.8 MiB | 記録  |
| svgp | forrester | 256  | gpy      | 8.77 ms  | 9.42 ms   | 10    | 0.12 ms     | 133.2 MiB | 記録  |
| sgpr | forrester | 1024 | gprx     | 0.35 ms  | 14.36 ms  | 10    | 0.04 ms     | 6.4 MiB   | -     |
| sgpr | forrester | 1024 | gpytorch | 4.32 ms  | 78.17 ms  | 10    | 2.41 ms     | 249.1 MiB | 記録  |
| sgpr | forrester | 1024 | gpy      | 11.25 ms | 15.74 ms  | 10    | 0.12 ms     | 136.6 MiB | pass  |
| svgp | forrester | 1024 | gprx     | 0.23 ms  | 10.55 ms  | 10    | 0.06 ms     | 6.2 MiB   | -     |
| svgp | forrester | 1024 | gpytorch | 5.43 ms  | 32.36 ms  | 10    | 1.62 ms     | 246.3 MiB | 記録  |
| svgp | forrester | 1024 | gpy      | 8.46 ms  | 9.82 ms   | 10    | 0.11 ms     | 135.3 MiB | fail  |
| sgpr | forrester | 4096 | gprx     | 0.97 ms  | 50.85 ms  | 10    | 0.05 ms     | 8.5 MiB   | -     |
| sgpr | forrester | 4096 | gpytorch | 6.20 ms  | 66.39 ms  | 10    | 2.24 ms     | 260.6 MiB | 記録  |
| sgpr | forrester | 4096 | gpy      | 17.64 ms | 22.66 ms  | 10    | 0.11 ms     | 152.9 MiB | fail  |
| svgp | forrester | 4096 | gprx     | 0.68 ms  | 35.96 ms  | 10    | 0.06 ms     | 8.0 MiB   | -     |
| svgp | forrester | 4096 | gpytorch | 6.07 ms  | 67.58 ms  | 10    | 1.85 ms     | 250.8 MiB | 記録  |
| svgp | forrester | 4096 | gpy      | 13.98 ms | 16.91 ms  | 10    | 0.12 ms     | 151.7 MiB | fail  |
| sgpr | sphere    | 256  | gprx     | 0.11 ms  | 6.62 ms   | 10    | 0.03 ms     | 5.9 MiB   | -     |
| sgpr | sphere    | 256  | gpytorch | 4.29 ms  | 40.73 ms  | 10    | 2.28 ms     | 247.3 MiB | 記録  |
| sgpr | sphere    | 256  | gpy      | 9.43 ms  | 14.75 ms  | 10    | 0.11 ms     | 132.8 MiB | 記録  |
| svgp | sphere    | 256  | gprx     | 0.08 ms  | 4.09 ms   | 10    | 0.03 ms     | 5.8 MiB   | -     |
| svgp | sphere    | 256  | gpytorch | 3.40 ms  | 34.74 ms  | 10    | 1.63 ms     | 246.3 MiB | 記録  |
| svgp | sphere    | 256  | gpy      | 6.79 ms  | 7.65 ms   | 10    | 0.10 ms     | 133.1 MiB | pass  |
| sgpr | sphere    | 1024 | gprx     | 0.27 ms  | 20.18 ms  | 10    | 0.03 ms     | 6.5 MiB   | -     |
| sgpr | sphere    | 1024 | gpytorch | 4.18 ms  | 60.57 ms  | 10    | 2.35 ms     | 249.2 MiB | pass  |
| sgpr | sphere    | 1024 | gpy      | 11.48 ms | 16.98 ms  | 10    | 0.13 ms     | 135.3 MiB | fail  |
| svgp | sphere    | 1024 | gprx     | 0.20 ms  | 15.37 ms  | 10    | 0.03 ms     | 6.0 MiB   | -     |
| svgp | sphere    | 1024 | gpytorch | 3.49 ms  | 39.33 ms  | 10    | 1.61 ms     | 247.5 MiB | 記録  |
| svgp | sphere    | 1024 | gpy      | 8.40 ms  | 10.42 ms  | 10    | 0.11 ms     | 135.6 MiB | fail  |
| sgpr | sphere    | 4096 | gprx     | 0.79 ms  | 70.75 ms  | 10    | 0.03 ms     | 9.2 MiB   | -     |
| sgpr | sphere    | 4096 | gpytorch | 3.93 ms  | 50.90 ms  | 10    | 3.56 ms     | 259.6 MiB | 記録  |
| sgpr | sphere    | 4096 | gpy      | 19.71 ms | 24.93 ms  | 10    | 0.11 ms     | 153.5 MiB | fail  |
| svgp | sphere    | 4096 | gprx     | 0.52 ms  | 50.43 ms  | 10    | 0.03 ms     | 8.0 MiB   | -     |
| svgp | sphere    | 4096 | gpytorch | 4.53 ms  | 46.71 ms  | 10    | 1.59 ms     | 252.4 MiB | 記録  |
| svgp | sphere    | 4096 | gpy      | 19.95 ms | 20.42 ms  | 10    | 0.12 ms     | 151.6 MiB | fail  |

P4-12 の GPy 負けのうち pass は sgpr Forrester n=1024（14.36 ms 対 15.74 ms）と svgp 球 n=256（4.09 ms 対 7.65 ms）。球 SGPR n=1024 の GPyTorch eval は 20.18 ms 対 60.57 ms で pass。残る GPy 負けは 7 セル（svgp Forrester n=1024、Forrester n=4096 の sgpr / svgp、球 n=1024 の sgpr / svgp、球 n=4096 の sgpr / svgp）。すでに勝っていた joint はすべて 2026-09-22 の自前中央値より短い。n=4096 の RSS は gprx 約 8–9 MiB。合否は下の P4-19 節。

## P4-19（Sparse のピーク RSS、[#210](https://github.com/YUKIKEDA/gprx/issues/210)）

再計測はしていない。数値は上の P4-18 節（2026-09-27、`just perf-sparse`）。ゲートは `n=4096` の 4 セル。ピーク RSS が GPy と GPyTorch の両方より小さい。5% 以内は判定不能。`n≤1024` は記録。criterion は合否にしない。

| 面   | 問題      | gprx    | GPy       | GPyTorch  | rss  |
| ---- | --------- | ------- | --------- | --------- | ---- |
| sgpr | forrester | 8.5 MiB | 152.9 MiB | 260.6 MiB | pass |
| svgp | forrester | 8.0 MiB | 151.7 MiB | 250.8 MiB | pass |
| sgpr | sphere    | 9.2 MiB | 153.5 MiB | 259.6 MiB | pass |
| svgp | sphere    | 8.0 MiB | 151.6 MiB | 252.4 MiB | pass |

4 セルとも pass。改善行は足さない。

## P4-20（Sparse joint の残り、[#214](https://github.com/YUKIKEDA/gprx/issues/214)）

同一機械。日付 2026-09-27。`just perf-sparse`。式は解析のまま。`∂K(Z,X)/∂θ` は密のまま。同じ θ の再分解を省き、VFE の O(m² n) 縮約は faer の行列積にした。SVGP は点ごとの統計を 1 回にまとめた。段階計時では対のループより、その縮約と再分解が支配的だった。SIMD は足していない。eval は 10 回。factor / predict / RSS は記録。criterion は合否にしない。

ゲートは、P4-18 で GPy より長かった 7 比較の eval 中央値が、この実行の GPy より小さいこと。球 `n=4096` の sgpr と svgp は、この実行の GPyTorch より小さいこと。5% 以内は判定不能。2026-09-27 に joint で相手より短かったセルは、いずれもその自前中央値より短い。

| 面   | 問題      | n    | lib      | factor   | eval N   | evals | predict 100 | peak RSS  | joint |
| ---- | --------- | ---- | -------- | -------- | -------- | ----- | ----------- | --------- | ----- |
| sgpr | forrester | 256  | gprx     | 0.17 ms  | 1.23 ms  | 10    | 0.05 ms     | 5.8 MiB   | -     |
| sgpr | forrester | 256  | gpytorch | 5.78 ms  | 43.16 ms | 10    | 3.23 ms     | 245.9 MiB | 記録  |
| sgpr | forrester | 256  | gpy      | 9.58 ms  | 15.36 ms | 10    | 0.11 ms     | 129.3 MiB | 記録  |
| svgp | forrester | 256  | gprx     | 0.16 ms  | 2.62 ms  | 10    | 0.09 ms     | 5.8 MiB   | -     |
| svgp | forrester | 256  | gpytorch | 6.05 ms  | 49.32 ms | 10    | 2.76 ms     | 244.7 MiB | 記録  |
| svgp | forrester | 256  | gpy      | 7.49 ms  | 8.24 ms  | 10    | 0.12 ms     | 129.3 MiB | 記録  |
| sgpr | forrester | 1024 | gprx     | 0.28 ms  | 4.24 ms  | 10    | 0.07 ms     | 6.2 MiB   | -     |
| sgpr | forrester | 1024 | gpytorch | 3.91 ms  | 44.08 ms | 10    | 2.14 ms     | 249.5 MiB | 記録  |
| sgpr | forrester | 1024 | gpy      | 11.03 ms | 18.39 ms | 10    | 0.11 ms     | 131.9 MiB | 記録  |
| svgp | forrester | 1024 | gprx     | 0.63 ms  | 6.39 ms  | 10    | 0.05 ms     | 6.2 MiB   | -     |
| svgp | forrester | 1024 | gpytorch | 3.74 ms  | 52.09 ms | 10    | 2.83 ms     | 246.8 MiB | 記録  |
| svgp | forrester | 1024 | gpy      | 8.31 ms  | 10.78 ms | 10    | 0.12 ms     | 132.2 MiB | pass  |
| sgpr | forrester | 4096 | gprx     | 0.90 ms  | 14.37 ms | 10    | 0.05 ms     | 7.6 MiB   | -     |
| sgpr | forrester | 4096 | gpytorch | 5.19 ms  | 72.03 ms | 10    | 2.20 ms     | 258.7 MiB | 記録  |
| sgpr | forrester | 4096 | gpy      | 17.72 ms | 23.60 ms | 10    | 0.12 ms     | 145.6 MiB | pass  |
| svgp | forrester | 4096 | gprx     | 1.08 ms  | 23.22 ms | 10    | 0.05 ms     | 8.1 MiB   | -     |
| svgp | forrester | 4096 | gpytorch | 6.10 ms  | 39.05 ms | 10    | 2.77 ms     | 249.2 MiB | 記録  |
| svgp | forrester | 4096 | gpy      | 14.02 ms | 18.38 ms | 10    | 0.12 ms     | 144.7 MiB | fail  |
| sgpr | sphere    | 256  | gprx     | 0.11 ms  | 2.36 ms  | 10    | 0.03 ms     | 5.8 MiB   | -     |
| sgpr | sphere    | 256  | gpytorch | 5.56 ms  | 58.67 ms | 10    | 3.18 ms     | 245.1 MiB | 記録  |
| sgpr | sphere    | 256  | gpy      | 10.07 ms | 15.47 ms | 10    | 0.11 ms     | 129.7 MiB | 記録  |
| svgp | sphere    | 256  | gprx     | 0.14 ms  | 2.72 ms  | 10    | 0.03 ms     | 5.8 MiB   | -     |
| svgp | sphere    | 256  | gpytorch | 6.38 ms  | 40.25 ms | 10    | 1.75 ms     | 245.5 MiB | 記録  |
| svgp | sphere    | 256  | gpy      | 7.13 ms  | 8.82 ms  | 10    | 0.12 ms     | 129.5 MiB | 記録  |
| sgpr | sphere    | 1024 | gprx     | 0.21 ms  | 7.59 ms  | 10    | 0.05 ms     | 6.2 MiB   | -     |
| sgpr | sphere    | 1024 | gpytorch | 5.61 ms  | 66.97 ms | 10    | 2.11 ms     | 248.3 MiB | 記録  |
| sgpr | sphere    | 1024 | gpy      | 10.28 ms | 15.10 ms | 10    | 0.10 ms     | 131.8 MiB | pass  |
| svgp | sphere    | 1024 | gprx     | 0.33 ms  | 13.32 ms | 10    | 0.06 ms     | 6.2 MiB   | -     |
| svgp | sphere    | 1024 | gpytorch | 6.16 ms  | 63.13 ms | 10    | 2.74 ms     | 246.9 MiB | 記録  |
| svgp | sphere    | 1024 | gpy      | 7.80 ms  | 9.15 ms  | 10    | 0.10 ms     | 132.3 MiB | fail  |
| sgpr | sphere    | 4096 | gprx     | 0.69 ms  | 24.00 ms | 10    | 0.03 ms     | 7.7 MiB   | -     |
| sgpr | sphere    | 4096 | gpytorch | 6.35 ms  | 58.47 ms | 10    | 2.83 ms     | 257.1 MiB | pass  |
| sgpr | sphere    | 4096 | gpy      | 19.28 ms | 26.35 ms | 10    | 0.11 ms     | 145.0 MiB | pass  |
| svgp | sphere    | 4096 | gprx     | 0.48 ms  | 34.57 ms | 10    | 0.03 ms     | 8.2 MiB   | -     |
| svgp | sphere    | 4096 | gpytorch | 6.90 ms  | 54.60 ms | 10    | 2.70 ms     | 251.6 MiB | pass  |
| svgp | sphere    | 4096 | gpy      | 16.09 ms | 22.93 ms | 10    | 0.13 ms     | 144.8 MiB | fail  |

7 比較のうち pass は svgp Forrester `n=1024`（6.39 ms 対 10.78 ms）、sgpr Forrester `n=4096`（14.37 ms 対 23.60 ms）、sgpr 球 `n=1024`（7.59 ms 対 15.10 ms）、sgpr 球 `n=4096`（24.00 ms 対 26.35 ms）。球 `n=4096` の GPyTorch は sgpr 24.00 ms 対 58.47 ms、svgp 34.57 ms 対 54.60 ms で pass。残る GPy 負けは 3 セル（svgp Forrester `n=4096`、svgp 球 `n=1024`、svgp 球 `n=4096`）。改善行は P4-21 / [#217](https://github.com/YUKIKEDA/gprx/issues/217)。

## P4-21（SVGP joint の残り、[#217](https://github.com/YUKIKEDA/gprx/issues/217)）

同一機械。日付 2026-09-27。`just perf-sparse`。式は解析のまま。`∂K(Z,X)/∂θ` の密行列は残した。点ごとの ELBO 勾配は列方向の内積にし、RBF と RBF ARD の対の勾配は `f64x4` にした。ARD は長さスケールを 1 回の exp にまとめ、`n>1024` は出力の列へ直接書いた。Matern は式を増やさないのでスカラーのまま。段階計時では対の勾配が残りの joint を支配していた。Exact の密 Gram は触っていない。eval は 10 回。factor / predict / RSS は記録。criterion は合否にしない。

ゲートは、P4-20 で GPy より長かった 3 比較（svgp Forrester `n=4096`、svgp 球 `n=1024`、svgp 球 `n=4096`）の eval 中央値が、この実行の GPy より小さいこと。5% 以内は判定不能。P4-20 で joint が相手より短かったセルは、その自前中央値から 5% を超えて遅くしない。

| 面   | 問題      | n    | lib      | factor   | eval N   | evals | predict 100 | peak RSS  | joint |
| ---- | --------- | ---- | -------- | -------- | -------- | ----- | ----------- | --------- | ----- |
| sgpr | forrester | 256  | gprx     | 0.12 ms  | 1.08 ms  | 10    | 0.04 ms     | 5.8 MiB   | -     |
| sgpr | forrester | 256  | gpytorch | 4.94 ms  | 58.91 ms | 10    | 3.15 ms     | 247.8 MiB | 記録  |
| sgpr | forrester | 256  | gpy      | 9.82 ms  | 15.81 ms | 10    | 0.11 ms     | 132.8 MiB | 記録  |
| svgp | forrester | 256  | gprx     | 0.11 ms  | 1.52 ms  | 10    | 0.05 ms     | 5.8 MiB   | -     |
| svgp | forrester | 256  | gpytorch | 5.26 ms  | 47.32 ms | 10    | 2.58 ms     | 246.0 MiB | 記録  |
| svgp | forrester | 256  | gpy      | 7.16 ms  | 8.62 ms  | 10    | 0.11 ms     | 133.1 MiB | 記録  |
| sgpr | forrester | 1024 | gprx     | 0.28 ms  | 3.41 ms  | 10    | 0.05 ms     | 6.2 MiB   | -     |
| sgpr | forrester | 1024 | gpytorch | 5.47 ms  | 59.73 ms | 10    | 2.08 ms     | 249.9 MiB | 記録  |
| sgpr | forrester | 1024 | gpy      | 11.19 ms | 16.84 ms | 10    | 0.11 ms     | 136.3 MiB | 記録  |
| svgp | forrester | 1024 | gprx     | 0.28 ms  | 4.42 ms  | 10    | 0.05 ms     | 6.3 MiB   | -     |
| svgp | forrester | 1024 | gpytorch | 5.68 ms  | 46.48 ms | 10    | 2.72 ms     | 246.7 MiB | 記録  |
| svgp | forrester | 1024 | gpy      | 8.27 ms  | 10.72 ms | 10    | 0.11 ms     | 136.2 MiB | 記録  |
| sgpr | forrester | 4096 | gprx     | 0.73 ms  | 11.05 ms | 10    | 0.04 ms     | 7.7 MiB   | -     |
| sgpr | forrester | 4096 | gpytorch | 6.25 ms  | 78.05 ms | 10    | 3.49 ms     | 260.0 MiB | 記録  |
| sgpr | forrester | 4096 | gpy      | 18.06 ms | 24.10 ms | 10    | 0.11 ms     | 153.0 MiB | 記録  |
| svgp | forrester | 4096 | gprx     | 0.72 ms  | 15.04 ms | 10    | 0.05 ms     | 8.2 MiB   | -     |
| svgp | forrester | 4096 | gpytorch | 4.02 ms  | 58.22 ms | 10    | 2.69 ms     | 250.1 MiB | 記録  |
| svgp | forrester | 4096 | gpy      | 21.27 ms | 18.89 ms | 10    | 0.12 ms     | 151.3 MiB | pass  |
| sgpr | sphere    | 256  | gprx     | 0.09 ms  | 2.34 ms  | 10    | 0.03 ms     | 5.9 MiB   | -     |
| sgpr | sphere    | 256  | gpytorch | 3.33 ms  | 56.23 ms | 10    | 3.18 ms     | 247.2 MiB | 記録  |
| sgpr | sphere    | 256  | gpy      | 9.89 ms  | 15.46 ms | 10    | 0.11 ms     | 132.8 MiB | 記録  |
| svgp | sphere    | 256  | gprx     | 0.07 ms  | 2.31 ms  | 10    | 0.03 ms     | 5.9 MiB   | -     |
| svgp | sphere    | 256  | gpytorch | 3.57 ms  | 56.21 ms | 10    | 2.73 ms     | 247.5 MiB | 記録  |
| svgp | sphere    | 256  | gpy      | 7.20 ms  | 8.80 ms  | 10    | 0.11 ms     | 132.9 MiB | 記録  |
| sgpr | sphere    | 1024 | gprx     | 0.22 ms  | 7.18 ms  | 10    | 0.03 ms     | 6.2 MiB   | -     |
| sgpr | sphere    | 1024 | gpytorch | 6.22 ms  | 66.13 ms | 10    | 3.45 ms     | 249.4 MiB | 記録  |
| sgpr | sphere    | 1024 | gpy      | 11.63 ms | 18.45 ms | 10    | 0.11 ms     | 135.3 MiB | 記録  |
| svgp | sphere    | 1024 | gprx     | 0.20 ms  | 6.27 ms  | 10    | 0.04 ms     | 6.5 MiB   | -     |
| svgp | sphere    | 1024 | gpytorch | 5.48 ms  | 62.71 ms | 10    | 2.67 ms     | 247.1 MiB | 記録  |
| svgp | sphere    | 1024 | gpy      | 8.59 ms  | 10.82 ms | 10    | 0.11 ms     | 135.2 MiB | pass  |
| sgpr | sphere    | 4096 | gprx     | 0.84 ms  | 24.02 ms | 10    | 0.03 ms     | 7.7 MiB   | -     |
| sgpr | sphere    | 4096 | gpytorch | 6.25 ms  | 80.52 ms | 10    | 2.71 ms     | 260.0 MiB | 記録  |
| sgpr | sphere    | 4096 | gpy      | 26.86 ms | 29.73 ms | 10    | 0.12 ms     | 153.4 MiB | 記録  |
| svgp | sphere    | 4096 | gprx     | 0.63 ms  | 17.12 ms | 10    | 0.05 ms     | 9.1 MiB   | -     |
| svgp | sphere    | 4096 | gpytorch | 6.88 ms  | 77.79 ms | 10    | 2.71 ms     | 252.4 MiB | 記録  |
| svgp | sphere    | 4096 | gpy      | 15.28 ms | 21.12 ms | 10    | 0.11 ms     | 152.8 MiB | pass  |

3 比較とも pass。svgp Forrester `n=4096` は 15.04 ms 対 GPy 18.89 ms、svgp 球 `n=1024` は 6.27 ms 対 10.82 ms、svgp 球 `n=4096` は 17.12 ms 対 21.12 ms。いずれも 5% 帯の外。P4-20 で相手より短かったセルは、その自前中央値から 5% を超えて遅くなっていない（最も近いのは sgpr 球 `n=4096` の 24.02 ms 対 24.00 ms）。残った負けはない。改善行は足さない。

## P5-1（混合精度の残差、[#39](https://github.com/YUKIKEDA/gprx/issues/39)）

同一機械。日付 2026-09-27。`cargo test --release --lib time_forrester_1024 -- --ignored`。ウォームアップ 1 回のあと 11 回の中央値。含めるのは f32 の `K`、f32 の Cholesky、反復、不収束時の f64 の解き直し。データの生成は含めない。対象は Exact の密な `Aα = y`（RBF、Forrester、`n=1024`、`ℓ = 1`、`σn² = 0.1`、seed `0`）。

| 残差 | 中央値 |
| ---- | ------ |
| 保存した f32 行列で引く | 22.40 ms |
| カーネルを f64 で計算し直す | 48.77 ms |

差は 5% を超える。省略時の既定は、保存した f32 行列で引く型。コード上の別名は置かない。`Gpr` の `fit` と `predict` は f64 のまま。

桁は単体テスト。Forrester `n=256` と `n=1024`（`ℓ = 1`、`σn² = 0.1`）と、同じ `n=256` の `x,y` で `ℓ = 1e4`、`σn² = 1e-5`（f32 Cholesky は成功し、`κ(A)·u_f32 > 1`）の三本が、両方の残差で `‖α − α_f64‖∞ / ‖α_f64‖∞ < 10 · n · u_f32` に届いた。不収束は f64 の α。jitter は増やしていない。

