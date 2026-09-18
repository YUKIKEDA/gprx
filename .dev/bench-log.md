# Bench log

時間は [criterion](https://docs.rs/criterion)、`benches/exact.rs`、`just bench`。確保は `tests/alloc.rs`。固定問題: RNG seed `0`、`n = 256`、`d = 8`、RBF + `GaussianLikelihood`、ハイパラ固定。`fit_lbfgs` だけ最適化ループ。新しいハーネスは作っていない。

目標比はまだ置かない（P2-1 / [#25](https://github.com/YUKIKEDA/gprx/issues/25)）。名前付き `phase-2` は P2-9 で取った（等方は `phase-1b` と比較。ARD は Always vs Never）。

## 機械

| 項目 | 値 |
| ---- | -- |
| OS | Windows 11 Home 10.0.26200 |
| CPU | 13th Gen Intel Core i5-13400F |
| RAM | 48 GB |
| rustc | 1.97.1 (`x86_64-pc-windows-msvc`) |
| 日付 | 2026-09-18 |

criterion は gnuplot なし、plotters。baseline 名 `phase-1b` / `phase-2` はこの機械の `target/criterion` に保存した（git には入れない）。比較: `cargo bench --bench exact -- --baseline phase-2`。

## `phase-1b`（n = 256）

criterion 中央値（括弧は 95% 区間の両端）。

| グループ | 時間 | 確保（Workspace 後） |
| -------- | ---- | -------------------- |
| `kernel_rbf` | 266 µs (263–269) | — |
| `cholesky_alpha` | 159 µs (157–162) | — |
| `predict_100` | 379 µs (376–382) | 9（上限 9） |
| `mll_and_grad` | 1.42 ms (1.39–1.45) | 1（上限 16） |
| `fit_lbfgs` | 257 ms (256–258) | — |

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

| グループ | phase-1b | この PR | 変化（中央値） |
| -------- | -------- | ------- | ------------- |
| `kernel_rbf` | 266 µs | 249 µs (246–254) | −4.9%（孤立経路。Gpr キャッシュ外） |
| `cholesky_alpha` | 159 µs | 151 µs (150–151) | −7.8%（この行は触っていない。機械ゆらぎ） |
| `predict_100` | 379 µs | 362 µs (358–368) | −5.0% |
| `mll_and_grad` | 1.42 ms | 1.23 ms (1.21–1.26) | **−13.3%** |
| `fit_lbfgs` | 257 ms | 224 ms (222–226) | **−13.0%** |

DoD（改善または同等）は満たす。確保上限は変えていない。

## P2-3（Rayon カーネル構築、[#27](https://github.com/YUKIKEDA/gprx/issues/27)）

同一機械。比較: `cargo bench --bench exact -- --baseline phase-1b`。等方距離は下三角を列パーティションで並列し、上三角は逐次コピー。`write_triangle(Lower)` も同じ分割。Cholesky は `Par::Seq`。`thread_scratch` はワーカー数ぶんの空 `0×0` で、並列前に `mem::take`。孤立 `kernel_rbf` は毎回距離埋め + RBF 下三角なので、この行の効果はそこに出る。

| グループ | phase-1b | この PR | 変化（中央値） |
| -------- | -------- | ------- | ------------- |
| `kernel_rbf` | 266 µs | 149 µs (148–149) | **−43.8%** |
| `cholesky_alpha` | 159 µs | 153 µs (152–155) | −7.0%（この行は触っていない。機械ゆらぎ） |
| `predict_100` | 379 µs | 374 µs (370–383) | −1.8%（ノイズ域） |
| `mll_and_grad` | 1.42 ms | 1.18 ms (1.176–1.184) | **−15.7%** |
| `fit_lbfgs` | 257 ms | 231 ms (230–233) | **−10.1%** |

DoD（`kernel_rbf` が速くなること、1b と数値一致）は満たす。確保: `mll_and_grad` は 1（上限 16）。`predict_100` は warmup 後 8 に下がったので上限を 9→8。P2-2 Always（mll 1.23 ms / fit 224 ms）と比べると mll はもう一段速い。`fit_lbfgs` は 231 ms で P2-2 よりわずかに遅い（キャッシュ済み apply の並列オーバーヘッド + ゆらぎ）。

## P2-4（ホットパス確保 0、[#28](https://github.com/YUKIKEDA/gprx/issues/28)）

同一機械。比較: `cargo bench --bench exact -- --baseline phase-1b`。Workspace に `rhs`（n×1）と query バッファを置き、`value_and_gradient_into` と `predict_into` は warmup 後に新規確保しない。`predict_100` ベンチは `predict_into`。Cholesky は `Par::Seq`。カーネル並列は触っていない。

| グループ | phase-1b | この PR | 変化（中央値） |
| -------- | -------- | ------- | ------------- |
| `kernel_rbf` | 266 µs | 152 µs (152–153) | **−42.5%**（P2-3 相当。この行は触っていない） |
| `cholesky_alpha` | 159 µs | 156 µs (155–158) | −2.8%（ノイズ域） |
| `predict_100` | 379 µs | 368 µs (364–371) | **−3.8%**（query バッファ再利用） |
| `mll_and_grad` | 1.42 ms | 1.21 ms (1.205–1.215) | **−14.1%** |
| `fit_lbfgs` | 257 ms | 238 ms (237–240) | **−7.4%** |

DoD（`tests/alloc.rs` 上限 0、ユーザーカーネル除く）は満たす。確保: `mll_and_grad` 0、`predict_100`（`predict_into`）0。P2-3（mll 1.18 ms / fit 231 ms / predict 374 µs）と比べると時間は同等域。`predict` の便利 API は出力 `Vec` を毎回確保する。

## P2-5（カーネル SIMD、[#29](https://github.com/YUKIKEDA/gprx/issues/29)）

同一機械。`wide::f64x4`。列優先・単位行ストライドの等方 RBF `apply` / `grad` / `apply_cross` と二乗距離の行ループ。既定 rustc（SSE2、`target-cpu=native` なし。phase-1b と同じフラグ）。

最初の下書きは `mll_and_grad` を分母にして「13% だから入れない」とした。勾配込み 1 評価ではカーネルは最大部品ではないが、`FIXED` と `predict` は勾配を払わない。その結論は取り消した。

比較: `cargo bench --bench exact -- --baseline phase-1b "kernel_rbf|cholesky_alpha|predict_100|mll_and_grad"`。

| グループ | phase-1b | Rayon のみ（P2-5 初稿） | この PR（Rayon + SIMD） | vs 1b | vs Rayon のみ |
| -------- | -------- | ----------------------- | ----------------------- | ----- | ------------- |
| `kernel_rbf` | 266 µs | 165 µs (162–167) | 125 µs (125.2–125.7) | **−52.4%** | **−24%** |
| `cholesky_alpha` | 159 µs | 165 µs (163–167) | 155 µs (153–158) | −5.7%（この行は未変更） | ノイズ域 |
| `predict_100` | 379 µs | 368 µs（P2-4） | 284 µs (280–288) | **−26.0%** | **−23%** |
| `mll_and_grad` | 1.42 ms | 1.25 ms (1.243–1.260) | 1.16 ms (1.159–1.163) | **−17.4%** | **−7%** |

`kernel_rbf` は距離埋め + RBF 下三角。SIMD 後は Cholesky（155 µs）より短い。Matérn / Periodic / RQ の内側はまだスカラー。

## P2-6（NLML 定数項は一本のまま、[#60](https://github.com/YUKIKEDA/gprx/issues/60)）

同一機械。固定問題で `(n/2) log(2π)` のあり/なしを測った。公開 API は分けていない（計測時だけ内部フラグを切替）。数値は P2-5 初稿（Rayon のみ）上。

| 経路 | 時間（中央値） |
| ---- | -------------- |
| `nlml_constant`（孤立の `0.5 n ln(2π)`） | 648 ps (642–655) |
| `mll_and_grad` 定数なし | 1.188 ms (1.187–1.190) |
| `mll_and_grad` 定数あり（直後の再測） | 1.200 ms (1.195–1.205) |

あり/なしの差は 12 µs（約 1%）。孤立加算 650 ps の 1.8 万倍で、criterion の連測ゆらぎに埋まる。`fit_lbfgs` は同じ定数を評価ごとに 1 回足すだけなので測っていない（200 回でも 0.13 µs 対 238 ms）。

**判断: 差はノイズ。公開 NLML と `Objective` は同じ `L(θ)` のまま。API は分けない。**

## P2-7（ARD 距離キャッシュ、[#88](https://github.com/YUKIKEDA/gprx/issues/88)）

同一機械。固定問題の ARD RBF（`ℓ_d = 1`、d = 8）。公開 Policy は増やしていない。Always は `n × (n·d)` の生 `(Δx_d)²` を fit 開始時に 1 回埋め、RBF ARD の apply/grad は Rayon + `wide::f64x4`。Never は座標から毎回（同じく SIMD）。Matérn / RQ ARD は同じキャッシュをスカラーで読む。等方 `phase-1b` とは比べない。

比較: `cargo bench --bench exact -- "mll_and_grad_ard|fit_lbfgs_ard"`。

| グループ | Always | Never | Always vs Never |
| -------- | ------ | ----- | --------------- |
| `mll_and_grad_ard` | 2.247 ms (2.228–2.269) | 2.113 ms (2.101–2.124) | **+6.3%**（Always が遅い） |
| `fit_lbfgs_ard` | 5.647 ms (5.609–5.729) | 4.526 ms (4.509–4.544) | **+24.8%**（Always が遅い） |

メモリ（解析）: Always の ARD テンソルは `256 × (256·8) × 8 B = 4.00 MiB`。Never / 等方は 0。X 自体は 16 KiB。n = 256, d = 8 ではキャッシュ読み（n²d）より座標 SIMD（nd）の方が帯域が小さい。設計 §5.2 の「d≪n では投資対効果が薄い」と一致。DoD の Always vs Never は「改善または同等」だが、この問題では Never が速い。Policy と数値一致（Never ≡ Always）はテストで満たす。`Auto` は P5-5。

## P2-9（`phase-2`、[#97](https://github.com/YUKIKEDA/gprx/issues/97)）

同一機械。`cargo bench --bench exact -- --save-baseline phase-2`。経路は `FittedGpr`（P2-8 typestate）。確保は `tests/alloc.rs` の `FittedGpr::predict_into` / `value_and_gradient_into` で上限 0。

### 等方 vs `phase-1b`

| グループ | phase-1b | phase-2 | 変化（中央値） |
| -------- | -------- | ------- | ------------- |
| `kernel_rbf` | 266 µs | 123 µs (123.2–123.7) | **−53.6%** |
| `cholesky_alpha` | 159 µs | 156 µs (155.6–157.3) | −1.7%（ノイズ域。この経路は Phase 2 で触っていない） |
| `predict_100` | 379 µs | 265 µs (264.3–266.7) | **−30.0%** |
| `mll_and_grad` | 1.42 ms | 1.15 ms (1.144–1.148) | **−19.3%** |
| `fit_lbfgs` | 257 ms | 337 ms (336.5–338.2) | **+31.3%** |

`kernel_rbf` / `predict_100` / `mll_and_grad` は Phase 2 のキャッシュ・Rayon・SIMD のまま 1b より短い。`fit_lbfgs` は 1 評価より遅い（L-BFGS ループ + `fit(self) → FittedGpr` の構築）。1 評価の改善はフィット全体には乗っていない。以降の比較基準は `phase-2`。

### ARD Always vs Never

等方とは比べない。P2-7 と同じ問題（ARD RBF、`ℓ_d = 1`、d = 8）。

| グループ | Always | Never | Always vs Never |
| -------- | ------ | ----- | --------------- |
| `mll_and_grad_ard` | 2.244 ms (2.229–2.259) | 2.103 ms (2.092–2.116) | **+6.7%**（Always が遅い） |
| `fit_lbfgs_ard` | 5.679 ms (5.660–5.720) | 4.751 ms (4.741–4.760) | **+19.5%**（Always が遅い） |

n = 256, d = 8 では Never が速い（P2-7 と同じ向き）。`Auto` は P5-5。

