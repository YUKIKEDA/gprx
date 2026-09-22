# ADR 0004: Sparse オンラインの VFE 因子は rank-1

- 状態: 採用
- 日付: 2026-09-21
- Issue: [#188](https://github.com/YUKIKEDA/gprx/issues/188)（P4-7）

## 文脈

VFE のコストは `O(nm²)` である。`Z` と `m` を固定して訓練点 `X` だけを増減するとき、毎回 `assemble_vfe` すると `K_mm` の LLT と `A` の全列をやり直す。`B = σn²I + AAᵀ` は `m×m` なので、列 1 本の追加・削除は rank-1 の cholupdate / choldown で足りるはずである。公開のオンライン型はまだ無い。このメモは因子の更新だけを固定する。

## 決定

- 対象は `X` の 1 点 insert / delete だけ。`Z` と `m` は動かさない
- `K_mm` と `L_mm` は据え置く
- insert は `k(Z, x_new)` を `L_mm` で解いて `A` の末尾列にし、`B` を cholupdate する
- delete は該当列を抜き、`B` を choldown する
- `w = B⁻¹ Ay` は `B` のあと LLT で解き直す。Sherman–Morrison は使わない
- `k_diag_sum` と `‖A‖_F²` は対角と列ノルムで増減する
- 照合は再構成した `B = LLᵀ` であり、`L` の符号は問わない
- 公開のオンライン型と insert API は置かない（P4-8）
- `m` の増減はこの行に入れない（P4-9）

RBF / Matern ν=3/2 / RBF ARD（2-D）/ RBF+White の各 `n = 4`・`m = 2` で、1 点 insert と真ん中 1 点 delete のあと、`A` / 再構成 `B` / `w` / `k_diag_sum` / `‖A‖_F²` が同じ `θ`・`Z` の `Sgpr<Fixed>::factor` と相対 `1e-12` で一致した。insert と delete はどちらも rank-1 で通った。

## 根拠

`Z` が止まっていれば `K_mm` は不変である。新しい列 `a` に対して `B ← B + aaᵀ`、削除では `B ← B − aaᵀ` である。`m` は Sparse では小さく、cholupdate は `O(m²)`、列のカーネルは `O(md)` である。フル再 factor の `O(nm²)` より安い。

`w` を増分で直すと `y` の詰めと符号を別に持つ。既存の `B` LLT で `Ay` を解くと同じ作業領域で足りる。

downdate は `r² ≤ 0` で落ちることがある。この小問題では落ちなかった。落ちた経路は delete を再 factor に戻す（案 C）。今のテストはその分岐を要求しない。

## 棄却した案

- **毎回 `assemble_vfe`**: 正しい。`n` が増えるたびに `O(nm²)` を払う
- **`w` の Sherman–Morrison**: `B` の更新と二重に数える。実装と照合が増える
- **この行で公開 `insert`**: 因子の一致と公開面が同じ PR になる。公開は P4-8
- **この行で `m` を増やす**: `K_mm` が変わる。X だけの更新ではない

## 帰結

- P4-8 の公開オンラインは、この rank-1 を `X` の増減に使う
- delete の downdate が大きな問題で落ちたら、同じ ADR のまま delete だけ再 factor に落とす。新しい Grill は要らない
- `Z` や `m` を動かす更新はこの ADR の外である
