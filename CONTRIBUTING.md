# gprx の開発手順

人間向けの手順。エージェントの入口は [AGENTS.md](AGENTS.md)。強制は `.cursor/rules/`。設計は [docs/design.md](docs/design.md)。並びと状態は [docs/roadmap.md](docs/roadmap.md)。ディレクトリは [docs/conventions.md](docs/conventions.md)。

`.dev/` は計測ログ、レビュー、下書きだけ。決定は残さない。

## 順

コードと規約に触れる作業は、この順だけ。

```text
Grill（設計が分岐するとき）→ Issue → 完了条件は Issue に書く → ブランチ → 作業 → PR → 人間レビュー → マージ
```

1. **Grill** — API、意味、フェーズの境界が分岐するとき。スキルは [`.cursor/skills/grill-me/SKILL.md`](.cursor/skills/grill-me/SKILL.md)。Issue に完了条件があり、[docs/design.md](docs/design.md) と矛盾しないときは省いてよい。質問のターンで成果物を閉じない。
2. **Issue** — テンプレートは Bug / Feat / Task / Spike。1 Issue = 1 PR。
3. **完了条件** — Grill で決めた文を、その Issue に書く。ロードマップには写さない。ロードマップは ID、タイトル、Issue、状態だけ。
4. **ブランチ** — `{type}/{issue番号}-{slug}`（例: `docs/223-rehome-docs`）。
5. **作業** — その Issue だけ。状態が `Grill 後に #n で確定` の行は作業しない。現在地は [docs/roadmap.md](docs/roadmap.md)。
6. **PR** — [`.github/pull_request_template.md`](.github/pull_request_template.md)。`## Related` に `Closes #N`。ゲートは `just lint` と `just test`。
7. **人間がレビューしてマージする。** エージェントはマージしない。

「進めなさい」は、Issue があるときの作業である。Issue の作成や範囲の決定の代用ではない。

## ゲート

- `just lint` は `cargo fmt --check` と `cargo clippy --all-targets -- -D warnings`
- `just test` は `cargo test`。Python を呼ばない
- `just bench` は criterion。速度の作業は数がないと範囲に入らない。詳細は `.cursor/rules/bench.mdc`
