# Bench log

時間は [criterion](https://docs.rs/criterion)、`benches/exact.rs`、`just bench`。確保は `tests/alloc.rs`。固定問題: RNG seed `0`、`n = 256`、`d = 8`、RBF + `GaussianLikelihood`、ハイパラ固定。`fit_lbfgs` だけ最適化ループ。新しいハーネスは作っていない。

目標比はまだ置かない（P2-1 / [#25](https://github.com/YUKIKEDA/gprx/issues/25)）。

## 機械

| 項目 | 値 |
| ---- | -- |
| OS | Windows 11 Home 10.0.26200 |
| CPU | 13th Gen Intel Core i5-13400F |
| RAM | 48 GB |
| rustc | 1.97.1 (`x86_64-pc-windows-msvc`) |
| 日付 | 2026-09-18 |

criterion は gnuplot なし、plotters。baseline 名 `phase-1b` はこの機械の `target/criterion` に保存した（git には入れない）。比較: `cargo bench --bench exact -- --baseline phase-1b`。

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

Phase 2 の順: 距離キャッシュと Rayon は **kernel**（2）向け。Cholesky を先に触らない。kernel は MLL 1 回の支配項ではないので、SIMD は P2-5 でこの表を見てから。定数項 `(n/2) log(2π)` は P2-6。

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
