# ADR 0002: Sparse GPR is VFE

- Status: accepted
- Date: 2026-09-21
- Issue: [#35](https://github.com/YUKIKEDA/gprx/issues/35) (P4-1)

## Context

Phase 4 ships one inducing-point Sparse GPR for large n. The design originally said "one of VFE or FITC", and both are not shipped. This crate puts correctness first, so the treatment of the marginal likelihood and the default of the reference implementations decide the choice. The ELBO expansion and the `SparseGpr` API are P4-2. This note fixes only the kind of approximation.

## Decision

- The Sparse approximation is VFE (Titsias 2009 / SGPR)
- FITC is not shipped. An implementation of both is not shipped either
- The early rows (P4-1…4) keep inducing locations Z fixed. Optimizing Z is P4-5 / P4-6

## Rationale

VFE is a lower bound on the Exact marginal likelihood. FITC treats the training conditional as fully independent. It can overestimate the likelihood and overfit observation noise (Bauer, van der Wilk, Rasmussen 2016). gprx gates on analytic solutions and external checks, so the lower bound of VFE fits better than FITC, which is easy to over-trust.

GPyTorch SGPR and GPflow SGPR default to the VFE family. Numerical checks from P4-2 onward follow those. sklearn's public GPR has no matching Sparse path.

## Rejected

- **FITC**: the implementation is somewhat simpler, and the predictive mean can be good on some problems. Overestimating the likelihood, and leaving the default of current reference implementations, does not fit this crate
- **Ship both**: the milestone says one. A second implementation for comparison is not a current row

## Consequences

- P4-2 starts from a VFE `SparseGpr` (Z fixed)
- No later row adds FITC. Reversing this is a new Grill → Issue
- Formulas, types, and goldens are not written in this ADR
