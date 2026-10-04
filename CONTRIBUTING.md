# Developing gprx

Procedure for humans. Agent entry: [AGENTS.md](AGENTS.md). Enforcement: `.cursor/rules/`. Design: [docs/design.md](docs/design.md). Order and status: [docs/roadmap.md](docs/roadmap.md). Directories: [docs/conventions.md](docs/conventions.md). `.dev/` is local scratch and is not committed.

## Order

Work that touches code or conventions follows this order only.

```text
Grill (when the design branches) → Issue → acceptance text stays on the Issue → branch → work → PR → human review → merge
```

1. **Grill** — when the API, the meaning, or a phase boundary branches. Skill: [`.cursor/skills/grill-me/SKILL.md`](.cursor/skills/grill-me/SKILL.md). Skip it when the Issue already has acceptance text and that text does not contradict [docs/design.md](docs/design.md). Do not close an artifact in the same turn as a question.
2. **Issue** — template is Bug / Feat / Task / Spike. One Issue, one PR.
3. **Acceptance** — write the text decided in Grill on that Issue. Do not copy it onto the roadmap. The roadmap keeps ID, title, Issue, and status.
4. **Branch** — `{type}/{issue}-{slug}` (example: `docs/223-rehome-docs`).
5. **Work** — that Issue only. Do not start a row whose status is `Set after Grill on #n`. The current row is [docs/roadmap.md](docs/roadmap.md).
6. **PR** — [`.github/pull_request_template.md`](.github/pull_request_template.md). `Closes #N` under `## Related`. Gates: `just lint` and `just test`. Open the PR before that work stops.
7. **A human reviews and merges.** Agents do not merge.

A branch with no Issue and no pull request is deleted. Do not leave one, locally or on the remote. The branch name contains the Issue number, the branch is pushed, and the PR is open. A human deletes the branch. Agents do not run `git branch -D` because of this rule.

「進めなさい」 is work on an Issue that already exists. It is not a substitute for creating an Issue or deciding scope.

## Gates

- `just lint` is `cargo fmt --check` and `cargo clippy --all-targets -- -D warnings`
- `just test` is `cargo test`. It does not invoke Python
- `just bench` is criterion. Speed work without a number is out of scope. Detail: `.cursor/rules/bench.mdc`
