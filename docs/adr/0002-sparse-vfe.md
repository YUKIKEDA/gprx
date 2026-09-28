# ADR 0002: Sparse GPR は VFE

- 状態: 採用
- 日付: 2026-09-21
- Issue: [#35](https://github.com/YUKIKEDA/gprx/issues/35)（P4-1）

## 文脈

Phase 4 は大きい n 向けに誘導点 Sparse GPR を一つ載せる。設計は当初「VFE または FITC の一方」と書いてあり、両方は置かない。正しさを先に置くクレートなので、周辺尤度の扱いと参照実装の既定が選択を決める。ELBO の展開と `SparseGpr` の API は P4-2。このメモは近似の種類だけを固定する。

## 決定

- Sparse 近似は VFE（Titsias 2009 / SGPR）
- FITC は載らない。両方の実装も置かない
- 初期（P4-1…4）は誘導点 Z 固定のまま。Z 最適化は P4-5 / P4-6

## 根拠

VFE は Exact の周辺尤度の下界になる。FITC は訓練条件付きを完全独立と置く近似で、尤度を過大評価して観測ノイズを過学習しやすい（Bauer, van der Wilk, Rasmussen 2016）。gprx は解析解と外部照合をゲートにしているので、過信しやすい FITC より下界の VFE の方が合う。

GPyTorch の SGPR と GPflow の SGPR は VFE 系が既定である。P4-2 以降の数値照合はそちらに寄せる。sklearn の公開 GPR に同じ Sparse 経路は無い。

## 棄却した案

- **FITC**: 実装がやや単純で、一部の問題では予測平均が良いことがある。尤度の過大評価と、現代の参照実装の既定から外れる点がこのクレートと合わない
- **両方載せる**: マイルストーンが一方と書いてある。比較用の第二実装は今の行に無い

## 帰結

- P4-2 は VFE の `SparseGpr`（Z 固定）から始める
- FITC を後から足す行は切らない。戻すなら新しい Grill → Issue
- 式・型・golden はこの ADR に書かない
