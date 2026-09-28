# 開発規約

人間向けの置き場所。手順は [CONTRIBUTING.md](../CONTRIBUTING.md)。公開面と意味は [design.md](design.md)。強制は `.cursor/rules/`。

## リポジトリ

単一クレート `gprx`。Cargo workspace にはしない。Issue が必要とするまで、空ファイルを先に置かない。

公開面の写しはここにも `.cursor/rules/` にも書かない。今の仕様は [design.md](design.md)。

```text
src/lib.rs                 公開の再エクスポート
src/error.rs
src/likelihood.rs
src/math.rs                Accurate / FastApprox
src/objective.rs
src/online.rs              crate-private の OnlineWorkspace
src/param.rs
src/precision.rs
src/rng.rs                 crate-private の SmallRng
src/workspace.rs           Workspace と QueryWorkspace。同じ struct に詰め込まない
src/gpr/                   model.rs に Gpr と FittedGpr。fitted.rs、factor.rs、online.rs、types.rs、tests.rs
src/kernel/                葉はここ。src/*.rs に置かない
  compiled/                mod.rs、apply.rs、grad.rs、hess.rs、gram.rs、f32_eval.rs、tests.rs
  constant.rs linear.rs white.rs
  rbf.rs rbf_ard.rs matern.rs matern_ard.rs periodic.rs rq.rs rq_ard.rs
  spec.rs term.rs dist.rs simd.rs lengthscale.rs scalar.rs
src/optimizer/             mod.rs、logit.rs。lbfgs.rs ncg.rs neldermead.rs fsa.rs newton.rs。adam.rs は Optimizer を実装しない
src/persist/               config.rs kernel.rs registry.rs tensors.rs transform.rs
src/sgpr/                  model.rs、fitted.rs、factor.rs、online.rs、tests.rs
src/svgp/                  model.rs、fitted.rs、factor.rs、tests.rs
src/transform/             target.rs input.rs pipeline.rs columnwise.rs。葉に分けない
tests/                     統合テスト。golden は compare/goldens/ だけ
benches/exact.rs           criterion
compare/                   sklearn と GPyTorch の生成。perf/ は手動の時間・RSS
examples/fit_predict.rs
docs/                      設計、ロードマップ、この規約、adr/
.dev/                      bench-log.md、reviews/、issue-seed.json。決定は残さない
```

`Workspace`、`QueryWorkspace`、`OnlineWorkspace`、faer の型はクレート私有。

単体テストは各モジュールの `#[cfg(test)]`。統合テストは `tests/`。
