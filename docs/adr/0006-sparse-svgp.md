# ADR 0006: SVGP is a type separate from VFE

- Status: accepted
- Date: 2026-09-22
- Issue: [#201](https://github.com/YUKIKEDA/gprx/issues/201) (P4-15)

## Context

Phase 4 fixed inducing-point Sparse as VFE (Titsias / SGPR) in [ADR 0002](0002-sparse-vfe.md) and shipped it as `Sgpr`. FITC is not shipped. The same inducing-point approximation, with an explicit variational posterior `q(u)`, is SVGP (Hensman et al.) and can take a minibatch ELBO. Before the external check in P4-11, the full-data ELBO and diagonal prediction are required. Replacing VFE, or adding a flag on the same type, collides with the rule that impossible states are types.

## Decision

- VFE `Sgpr` / `FittedSgpr` stay
- SVGP is a separate public type, `Svgp` / `FittedSvgp`
- FITC is not shipped (ADR 0002 stands)
- The first `q(u)` is a whitened full-rank Cholesky. `factor` places the prior at the caller's `Z` of length `m` (mean 0, `L = I`)
- `Z` comes from the caller. It is not in params. There is no k-means
- `Svgp<Fixed>::factor`, the full-data ELBO, and diagonal prediction are P4-15. `Adam` / minibatch `fit` and the full-data `value_and_gradient_into` are P4-16

## Rationale

VFE eliminates `q(u)` in closed form. It already matches Exact at `Z = X`, and the online rank-1 update sits on that factor. SVGP is the uncollapsed form of the same ELBO, and the optimal `q` returns to VFE. Replacing VFE would throw away the P4-2…10 path. A flag for whether `q` is present adds an ignored field or a runtime error. A separate type leaves VFE as it is, and SVGP always has `q`.

Nothing new has been added to the reason for shipping FITC since ADR 0002. Overestimating the likelihood, and leaving the default of the reference implementations, is the same mismatch.

## Rejected

- **Replace VFE with SVGP**: rebuild the collapsed factor and the online update. The Exact match also disappears away from the optimal `q`
- **Add a `q` flag on `Sgpr`**: an unused field or a runtime configuration error. The types would not be separate
- **Ship FITC**: overturns ADR 0002. It is not the subject of this row

## Consequences

- P4-15 starts from `Svgp<Fixed>::factor`, `neg_elbo`, and diagonal `predict`
- At the optimal whitened `q` (Titsias), it matches `FittedSgpr` at the same `θ`, `X`, and `Z`
- `Adam` and minibatches are `Svgp<Adam>::fit`. `Adam` does not implement `Optimizer`. A noisy gradient is not passed to L-BFGS
- No later row adds FITC. Reversing this is a new Grill → Issue
