# ADR 0005: Sparse 誘導点の増減は bordered insert と trailing delete

- 状態: 採用
- 日付: 2026-09-21
- Issue: [#192](https://github.com/YUKIKEDA/gprx/issues/192)（P4-9）

## 文脈

`OnlineSgpr` は `X` だけを rank-1 で増減する。理由は [ADR 0004](0004-sparse-online-rank1.md)。`n` が増えると、近似品質のために `m` も増やしたくなる。毎回 `assemble_vfe` すると `K_mm` の LLT と `A` の全列を `O(nm²)` でやり直す。公開の `InducingId` / `insert_inducing` はまだ無い。このメモは因子の更新だけを固定する。

## 決定

- 対象は誘導点 1 点の末尾 insert と任意位置 delete。`X` と `θ` は動かさない
- insert は `k(z_new, Z)` を `L_mm` で解き、`K_mm` を bordered LLT する。新しい `A` の行は `(k(z_new, X) − lᵀ A) / ℓ`。`B` も bordered LLT する
- delete は `L_mm` の該当行を除き、trailing 三角を cholupdate する。`K(Z, X) = L A` を再利用して行を抜き、新しい `L_mm` で `A` を解く。`B` は新しい `A` から作り直す
- `w = B⁻¹ Ay` は `B` のあと LLT で解き直す
- `k_diag_sum` は `X` の対角なので据え置く。`‖A‖_F²` は insert で行ノルムを足し、delete では作り直す
- 照合は再構成した `K_mm = LLᵀ` と `B = LLᵀ` であり、`L` の符号は問わない
- 公開 `InducingId` / `insert_inducing` / `delete_inducing` は置かない（P4-10）
- 座標は呼び出し側。k-means はこの行に入れない

RBF / Matern ν=3/2 / RBF ARD（2-D）/ RBF+White の各 `n = 4`・`m = 2` で、末尾 1 点 insert と先頭 1 点 delete（`m = 2` では末尾以外）のあと、再構成 `K_mm` / `A` / 再構成 `B` / `w` / `k_diag_sum` / `‖A‖_F²` が同じ `θ`・`X`・`Z` の `Sgpr<Fixed>::factor` と相対 `1e-12` で一致した。insert と delete はどちらも増分で通った。

## 根拠

末尾 insert では新しい行と列が `K_mm` の端に付く。既存の `L_mm` と `A` はそのまま使え、新しい行は `O(nm)`、bordered `B` は `O(m²)` である。フル再 assemble の `O(nm²)` より安い。

真ん中の delete は `K_mm` の行と列を抜く。`L` の先頭ブロックは据え置き、trailing を rank-1 update すれば `O(m²)` で足りる。残った `A` の行は `L` が変わるのでそのまま使えない。`K(Z, X) = L A` を再利用すればカーネルの再評価は不要で、三角ソルブは `O(nm²)` のままである。`B` は `A` の全行が動くので作り直す。

`w` を増分で直すと `y` の詰めと符号を別に持つ。既存の `B` LLT で `Ay` を解くと同じ作業領域で足りる。

## 棄却した案

- **毎回 `assemble_vfe`**: 正しい。`m` を増やすたびに `O(nm²)` と `k(Z, X)` の再評価を払う
- **delete も bordered の逆だけ**: 末尾以外では先に置換が要る。trailing cholupdate の方が置換を避ける
- **この行で公開 `insert_inducing`**: 因子の一致と公開面が同じ PR になる。公開は P4-10
- **この行で k-means**: 座標の選び方は因子の更新とは別である

## 帰結

- P4-10 の公開誘導点 API は、この増分を `m` の増減に使う
- delete の trailing update や `B` の再 factor が大きな問題で落ちたら、同じ ADR のまま delete だけ再 assemble に落とす。新しい Grill は要らない
- `X` の増減はこの ADR の外である（[ADR 0004](0004-sparse-online-rank1.md)）
