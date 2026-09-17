# AGENTS

エージェント向けの正本はここ（入口）と `.cursor/rules/`。設計の正本は `.dev/gprx-design.md`、タスクは `.dev/roadmap.md`。

## 必読ルール

| ファイル                                 | 内容                                          |
| ---------------------------------------- | --------------------------------------------- |
| `.cursor/rules/conventional-commits.mdc` | コミットメッセージ                            |
| `.cursor/rules/git.mdc`                  | 破壊的 git。質問は許可ではない                |
| `.cursor/rules/consent.mdc`              | 質問は決定ではない。選択肢には推奨とメリデメ。Issue / ロードマップを独断で閉じない |
| `.cursor/rules/pull-requests.mdc`        | PR タイトルと本文                             |
| `.cursor/rules/workflow.mdc`             | Issue → ブランチ → PR。品質ゲート。フェーズ順 |
| `.cursor/rules/layout.mdc`               | ディレクトリと公開 API                        |
| `.cursor/rules/rust.mdc`                 | 安全性、Clippy、浮動小数の比較                |
| `.cursor/rules/rust-api.mdc`             | 命名、所有権、公開面（API Guidelines）        |
| `.cursor/rules/rust-docs.mdc`            | rustdoc（`///`、Examples / Errors / Panics）  |
| `.cursor/rules/rust-hpc.mdc`             | 確保・レイアウト・Rayon（数値計算）           |
| `.cursor/rules/bench.mdc`                | ベンチと確保カウント。高速化は測ってから      |

正本の外部ドキュメント: [API Guidelines](https://rust-lang.github.io/api-guidelines/), [rustdoc book](https://doc.rust-lang.org/stable/rustdoc/how-to-write-documentation.html)。

`.dev/conventions.md` は要約ではない。中身は rules に移した。

## 今の着手点

Phase 2 の P2-8（`Gpr` / `FittedGpr`）。Issue は実装時。`phase-1b` の順は [`.dev/bench-log.md`](.dev/bench-log.md)。
