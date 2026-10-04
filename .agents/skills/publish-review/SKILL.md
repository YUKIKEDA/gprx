---
name: publish-review
description: >-
  Read gprx's README, comparison docs, and public rustdoc before crates.io
  publish. Use when preparing a release, running cargo package, running
  cargo publish --dry-run, or checking that a stranger can understand the
  crate from the docs alone.
---

# Publish review

Run this on the commit that will be published, before `cargo package` and before `cargo publish --dry-run`. A single failure stops those commands. Do not run `cargo publish` without `--dry-run`.

## Read first

- `.cursor/rules/reader-docs.mdc`
- `.cursor/rules/rust-docs.mdc`
- `.agents/skills/natural-japanese/SKILL.md` for `README.ja.md` and `docs/comparison.ja.md`

Do not restate those files here. Judge the pages against them.

## Pages

- `README.md` and `README.ja.md`
- `docs/comparison.md` and `docs/comparison.ja.md`
- Crate-level `//!` and every `///` on an item an external crate can name with default features
- `Cargo.toml` `description`

`gprx::internals` is outside the contract. `docs/design.md`, `docs/roadmap.md`, and `docs/adr/` are the decision record. Do not fail them for keeping issue and phase IDs.

## How to judge

Read as someone who has the crate page and has not opened this repository's issue tracker.

Fail a page when any of these is true:

- A name, default, layout, error, version, MSRV, or command does not match the code
- A comparison number has no committed source and the sentence does not say where it came from
- The README is not in the order reader-docs requires, or the English and Japanese sections diverge
- The README still contains the comparison tables or figures
- A reader must know an issue number, a pull request number, a phase ID, a roadmap ID, `roadmap.md`, `AGENTS.md`, `.cursor/rules`, Grill, or a developer `just` recipe. The comparison doc may keep the `just` commands that regenerate its figures, and no other `just` recipe
- Public rustdoc fails `.cursor/rules/rust-docs.mdc`, including Examples and Errors. `missing_docs` passing does not clear this
- The first example is not pasteable after the documented dependency, or it uses `unwrap`
- Column-major layout, `fit` consuming the trainer, `GaussianLikelihood`, the 0.x break policy, or `internals` being outside the contract is missing before the first example
- The README, the Japanese README, the crate-level rustdoc, and `description` contradict each other
- The README links relatively at a file that is not in the package, or embeds an image
- The Japanese is a gloss of the English under the natural-japanese skill

## Report

For each failure, name the file, the text, the rule it breaks, and why a new reader stops. Fix the failures in the publish issue. When none remain, the package build and the dry-run are allowed.
