# AGENTS

Entry point for agents. Procedure: [CONTRIBUTING.md](CONTRIBUTING.md). Enforcement: `.cursor/rules/`.

| Source | Path |
| --- | --- |
| Design | [docs/design.md](docs/design.md) |
| Modules, responsibilities, dependency direction | [docs/architecture.md](docs/architecture.md) |
| Saved directory (`config.json` and `model.safetensors`) | [docs/persist-format.md](docs/persist-format.md) |
| Order and status | [docs/roadmap.md](docs/roadmap.md) |
| Directories | [docs/conventions.md](docs/conventions.md) |
| Why a decision was made | [docs/adr/](docs/adr/) |

## Required rules

| File | Contents |
| --- | --- |
| `.cursor/rules/conventional-commits.mdc` | Commit messages |
| `.cursor/rules/git.mdc` | Destructive git. A question is not permission |
| `.cursor/rules/consent.mdc` | A question is not a decision. Options need a recommendation and a pro / con. Do not close an Issue or the roadmap on your own |
| `.cursor/rules/pull-requests.mdc` | PR title and body |
| `.cursor/rules/workflow.mdc` | Prohibitions and quality gates. The full procedure is CONTRIBUTING |
| `.cursor/rules/defer.mdc` | Do not ship an unplanned "temporary" or "not now" |
| `.cursor/rules/scope.mdc` | Do not recommend a partial slice because it is smaller. If you split, plan the rest in the same decision |
| `.cursor/rules/layout.mdc` | Single crate. Do not copy the public surface into a rule |
| `.cursor/rules/scratch.mdc` | Delete a file you created once that step no longer needs it. Gitignore is not storage |
| `.cursor/rules/rust.mdc` | Safety, Clippy, floating-point comparison |
| `.cursor/rules/rust-api.mdc` | Naming, ownership, public API (API Guidelines) |
| `.cursor/rules/types.mdc` | Illegal states are types. Do not ignore a field or reject a config at runtime |
| `.cursor/rules/rust-docs.mdc` | rustdoc (`///`, Examples / Errors / Panics) |
| `.cursor/rules/rust-hpc.mdc` | Allocation, layout, Rayon (numerical code) |
| `.cursor/rules/bench.mdc` | Benchmarks and allocation counts. Measure before speeding up |
| `.cursor/rules/publish.mdc` | crates.io gate: MSRV, coverage floor, package files, 0.x contract |

External sources: [API Guidelines](https://rust-lang.github.io/api-guidelines/), [rustdoc book](https://doc.rust-lang.org/stable/rustdoc/how-to-write-documentation.html).

The current row's ID lives only in [docs/roadmap.md](docs/roadmap.md).
