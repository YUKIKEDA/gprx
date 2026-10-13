# Changelog

All notable changes to gprx are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). A 0.x minor
version may break the public API (the default-feature items an external
crate can name); `gprx::internals` is outside that contract.

## [0.2.0]

### Breaking

- **MSRV 1.88** (was 1.85). The crate uses let chains (stable in 1.88).
  0.1.0 declared `rust-version = "1.85"`, but its dependency argmin 0.11
  uses `is_multiple_of` (stable in 1.87), so 0.1.0 did not build before
  1.87 either; 0.2.0 declares what it needs.
- **`Gpr::new`, `Sgpr::new`, and `Svgp::new` are generic over the
  kernel** (`K: ModelKernel`: a `KernelSpec` or a `DistanceKernel`). A
  call that converts the kernel with `.into()` no longer infers its type
  and fails with `E0283` (type annotations needed):

  ```rust,ignore
  // 0.1.0
  let gpr = Gpr::new(RbfKernel::new(1.0)?.into(), GaussianLikelihood::new(0.1)?);
  ```

  Name the type instead, with `KernelSpec::from` or an annotated binding:

  ```rust
  # use gprx::kernel::{KernelSpec, RbfKernel};
  # use gprx::{GaussianLikelihood, Gpr};
  # fn main() -> Result<(), gprx::GprError> {
  let gpr = Gpr::new(KernelSpec::from(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?);
  let kernel: KernelSpec = RbfKernel::new(1.0)?.into();
  let gpr = Gpr::new(kernel, GaussianLikelihood::new(0.1)?);
  # let _ = gpr;
  # Ok(())
  # }
  ```

  A `KernelSpec` passed as a value, as in the 0.1.0 README and examples,
  builds unchanged.
- **A method called by its type's path without the type's parameters**
  (`FittedGpr::predict(&model, ..)`) fails with `E0034` (multiple
  applicable items) when the distance models have a method of the same
  name: the bare path does not say which. That is `fit` and `factor` on
  `Gpr`, `Sgpr`, and `Svgp`; the `predict` / `predict_covariance` /
  `sample` family on every fitted and online model; `insert` on
  `OnlineGpr` and `OnlineSgpr`; `insert_inducing`; `into_online` on
  `FittedSgpr`; and `save` on `FittedSgpr`, `OnlineSgpr`, and
  `FittedSvgp`. Methods that only the coordinate models have, such as
  `FittedGpr::save` or `loo_predict`, build as before. Call the method on
  the value, or name the type with the parameters the value has
  (`<FittedGpr>::predict` for the defaults, `<FittedGpr<Fixed>>::predict`
  for a model from `factor`, `<Svgp<Adam>>::fit`):

  ```rust
  # use gprx::kernel::{KernelSpec, RbfKernel};
  # use gprx::{FittedGpr, GaussianLikelihood, Gpr};
  # fn main() -> Result<(), gprx::GprError> {
  # let model = Gpr::new(KernelSpec::from(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?)
  #     .fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0])?;
  let a = model.predict(&[0.5], 1, 1)?;
  let b = <FittedGpr>::predict(&model, &[0.5], 1, 1)?;
  # assert_eq!(a, b);
  # Ok(())
  # }
  ```

- **`KernelTerm::clone_box` is gone.** gprx clones a custom leaf through
  its `Clone` (which `CustomKernel::new` and `KernelSpec::custom` already
  require), so the method was never called. Delete `clone_box` from your
  `impl KernelTerm<T>`; nothing replaces it. The `clone_box` of the
  transform traits stays.

### Added

- **Supplied squared distances.** A model can take the training and query
  squared distances from the caller instead of coordinates:
  `kernel::ScalarDistance` (one `n × n` table) and `kernel::ArdDistance`
  (one table per dimension) name a slot, a leaf on a slot builds a
  `kernel::DistanceKernel`, and each call passes one `kernel::DistanceSource`
  per slot (owned, borrowed, or filled). Exact, `Sgpr`, `Svgp`, and the
  online `OnlineGpr` / `OnlineSgpr` all accept it, and a distance kernel can
  be combined with coordinate leaves (`WithPoints`).
- Tables are checked strictly (finite, non-negative, zero diagonal,
  symmetric); `DistanceSource::tidy` repairs rounding within a tolerance.
  New errors: `GprError::InvalidDistance` (slot, dimension, and pair of the
  first bad value) and `GprError::DistanceSlot` with `SlotErrorKind`.
- Save and load of distance models (`persist::LoadedDistanceGpr`,
  `LoadedDistanceSgpr`, `LoadedDistanceSvgp`): the file keeps the training
  `d²` the model owns and the kernel's slot table.
- `kernel::ModelKernel`, `PointKernel`, `Supply`, and `NoSupply` name the
  two kinds of kernel; `KernelSpec` and `CompiledKernel` take the kind as a
  last type parameter whose default is the 0.1.0 coordinate kernel.

### Changed

- A save streams each tensor into the file instead of building the whole
  file in memory first: its allocations no longer grow with `n`.

### Unchanged

- The save format: `FORMAT_VERSION` is still 1. A directory 0.1.0 saved
  loads in 0.2.0 and predicts the same values (an online exact model to
  within a relative 1e-14), and a coordinate model 0.2.0 saves loads in
  0.1.0.
- The 0.1.0 README and examples build and run against 0.2.0 without
  changes. So do its tests, except `tests/custom_kernel.rs`, whose
  `KernelTerm` impls define `clone_box` (see above).

## [0.1.0] - 2026-10-06

First release: exact GPR, sparse GPR (VFE), and SVGP, with online updates
and directory save/load.

[0.2.0]: https://github.com/YUKIKEDA/gprx/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/YUKIKEDA/gprx/releases/tag/v0.1.0
