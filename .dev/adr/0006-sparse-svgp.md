# ADR 0006: SVGP は VFE と別型

- 状態: 採用
- 日付: 2026-09-22
- Issue: [#201](https://github.com/YUKIKEDA/gprx/issues/201)（P4-15）

## 文脈

Phase 4 は誘導点 Sparse を [ADR 0002](0002-sparse-vfe.md) で VFE（Titsias / SGPR）に固定し、`Sgpr` に載せた。FITC は載らない。同じ誘導点近似でも、変分事後 `q(u)` を陽に持つ SVGP（Hensman et al.）はミニバッチ ELBO ができる。P4-11 の外部照合の前に、全データ ELBO と対角予測が要る。VFE を置き換えるか、同じ型にフラグを足すかは、取れない状態を型で表す規則と衝突する。

## 決定

- VFE の `Sgpr` / `FittedSgpr` は残す
- SVGP は別公開型 `Svgp` / `FittedSvgp`
- FITC は載らない（ADR 0002 のまま）
- 最初の `q(u)` は whitened の full-rank Cholesky。`factor` は呼び出し側 `Z` の `m` で prior（平均 0、`L = I`）を置く
- `Z` は呼び出し側。params に入れない。k-means は置かない
- `Svgp<Fixed>::factor` と全データ ELBO・対角予測は P4-15。`Adam` / ミニバッチ `fit` は P4-16

## 根拠

VFE は `q(u)` を閉じた形で消す。既に `Z = X` で Exact と一致し、オンラインの rank-1 もその因子に載っている。SVGP は同じ ELBO の未崩壊形で、最適 `q` では VFE に戻る。置き換えると P4-2…10 の経路を捨てる。同じ struct に `q` の有無をフラグで足すと、無視されるフィールドか実行時エラーになる。別型なら VFE は今のまま、SVGP は `q` を常に持つ。

FITC を足す理由は ADR 0002 から増えていない。尤度の過大評価と、参照実装の既定から外れる点が同じである。

## 棄却した案

- **VFE を SVGP に置き換える**: 崩壊形の因子とオンライン更新を作り直す。最適 `q` 以外では Exact 一致も消える
- **`Sgpr` に `q` フラグを足す**: 未使用フィールドか実行時の設定エラーになる。型で分けられない
- **FITC を載せる**: ADR 0002 を覆す。この行の対象ではない

## 帰結

- P4-15 は `Svgp<Fixed>::factor`、`neg_elbo`、対角 `predict` から始める
- 最適 whitened `q`（Titsias）では同じ `θ`・`X`・`Z` の `FittedSgpr` と一致する
- `Adam` とミニバッチは P4-16。L-BFGS にノイズ付き勾配は渡さない
- FITC を後から足す行は切らない。戻すなら新しい Grill → Issue
