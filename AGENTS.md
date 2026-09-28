# AGENTS

エージェント向けの入口はここ。手順は [CONTRIBUTING.md](CONTRIBUTING.md)。強制は `.cursor/rules/`。

| 正本 | パス |
| --- | --- |
| 設計 | [docs/design.md](docs/design.md) |
| 並びと状態 | [docs/roadmap.md](docs/roadmap.md) |
| ディレクトリ | [docs/conventions.md](docs/conventions.md) |
| 決定の理由 | [docs/adr/](docs/adr/) |
| 計測 | [.dev/bench-log.md](.dev/bench-log.md) |

`.dev/` は計測ログ、レビュー、下書きだけ。決定は残さない。

## 必読ルール

| ファイル | 内容 |
| --- | --- |
| `.cursor/rules/conventional-commits.mdc` | コミットメッセージ |
| `.cursor/rules/git.mdc` | 破壊的 git。質問は許可ではない |
| `.cursor/rules/consent.mdc` | 質問は決定ではない。選択肢には推奨とメリデメ。Issue / ロードマップを独断で閉じない |
| `.cursor/rules/pull-requests.mdc` | PR タイトルと本文 |
| `.cursor/rules/workflow.mdc` | 禁止事項と品質ゲート。手順の全文は CONTRIBUTING |
| `.cursor/rules/defer.mdc` | 計画のない「暫定」「今はやらない」は禁止 |
| `.cursor/rules/scope.mdc` | 部分実装を実装量で推奨しない。分けるなら同時に計画する |
| `.cursor/rules/layout.mdc` | 単一クレート。公開面の写しをルールに書かない |
| `.cursor/rules/rust.mdc` | 安全性、Clippy、浮動小数の比較 |
| `.cursor/rules/rust-api.mdc` | 命名、所有権、公開面（API Guidelines） |
| `.cursor/rules/types.mdc` | 取れない状態は型で表す。無視・実行時の設定エラーは禁止 |
| `.cursor/rules/rust-docs.mdc` | rustdoc（`///`、Examples / Errors / Panics） |
| `.cursor/rules/rust-hpc.mdc` | 確保・レイアウト・Rayon（数値計算） |
| `.cursor/rules/bench.mdc` | ベンチと確保カウント。高速化は測ってから |

正本の外部ドキュメント: [API Guidelines](https://rust-lang.github.io/api-guidelines/), [rustdoc book](https://doc.rust-lang.org/stable/rustdoc/how-to-write-documentation.html)。

現在地の ID は [docs/roadmap.md](docs/roadmap.md) だけが持つ。
