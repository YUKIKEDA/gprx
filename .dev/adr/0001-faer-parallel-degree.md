# ADR 0001: faer の並列度

- 状態: 採用
- 日付: 2026-09-19
- Issue: [#143](https://github.com/YUKIKEDA/gprx/issues/143)（P2B-22）

## 文脈

n=4096 の joint MLL+grad は Cholesky と `W` 用の n 本三角ソルブが壁時計の 97% 以上を占める。sklearn も `cho_solve(L, I)` で A⁻¹ を作る。gprx は `Par::Seq` だった。未設定の Rayon プール（この機では 16 本）で常時 `Par::rayon(0)` にすると、n=256 の eval が数倍遅くなる。

## 決定

- factor の Cholesky、α / W の `solve_in_place`、predict / LOO / `predict_covariance` の三角ソルブは `faer_par(n) = Par::rayon(min(pool, max(1, n/64)))`
- カーネル埋めはプロセス広域プールのまま
- 公開の `n_jobs` / 並列 on-off は置かない。`RAYON_NUM_THREADS=1` は 1 本
- `Workspace` の faer scratch は同じ `Par` で取る
- `benches/exact.rs` の Cholesky は Seq のまま

## 根拠（この機、16 論理、各 5 回）

eval 10 の平均。1 本に対する比。

| n | 問題 | 1 本 | 4 本 | 8 本 | 16 本 |
|---:|---|---:|---:|---:|---:|
| 256 | Forrester | 12.3 ms | **10.5 ms** | 18.5 ms | 105 ms |
| 256 | sphere | 14.5 ms | **10.2 ms** | 31.2 ms | 32.8 ms |
| 1024 | Forrester | 546 ms | 209 ms | **168 ms** | 193 ms |
| 1024 | sphere | 585 ms | 233 ms | **189 ms** | 250 ms |
| 2048 | Forrester | 4.54 s | 1.54 s | **1.06 s** | 1.07 s |
| 2025 | sphere | 3.90 s | 1.41 s | **0.98 s** | 1.01 s |
| 4096 | Forrester | 38.8 s | 11.7 s | 8.68 s | **8.34 s** |
| 4096 | sphere | 37.8 s | 12.1 s | 9.09 s | **8.68 s** |

`n/64` は 256→4、1024→16、4096→16。faer だけ 4 本・カーネル 16 本の混在は、256 で 16 本事故を避け、両方 4 本より数 ms 遅い。n≥1024 では式がプール全本数になり、カーネルを絞っても差は出ない。

壁時計は一面である。アルゴリズム（A⁻¹ の n 本ソルブ）は sklearn と同じ。アロケーション（`I` / `W` の n×n）は #142 / P2B-19。

## 帰結

- 未設定 16 本でも n=256 は faer 4 本に落ちる
- 並列は bitwise を変えうる。数値テストは公差
- 合否は `compare/perf` と `.dev/bench-log.md`。criterion は使わない
