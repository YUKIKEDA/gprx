//! The supplied squared distances a model reads: binding and checking the
//! caller's sources, the training `d²` a model owns, and the per-call blocks
//! of a prediction.
//!
//! A source arrives as `f64`. The training store keeps it in the model's
//! storage scalar: a scalar slot as a dense square, an ARD slot as the
//! packed lower triangles of [`ArdSqDiffBuf`]. A prediction block of an
//! `f64` model is read in place.

mod blocks;
mod check;
mod query;
mod refined;
mod slot;
mod train;

use std::any::Any;
use std::borrow::Cow;
use std::fmt;

use faer::MatRef;
use rayon::prelude::*;

use super::compiled::supplied::{ArdRect, ArdSquare, RectSlots, SquareSlots, unbound};
use super::dist::{ArdBlocks, ArdSqDiff, ArdSqDiffBuf, BlockList, Checked, packed_len, packed_run};
use super::simd::SquareOut;
use super::{ArdData, DistanceFill, ScalarData, ScalarOps, SourceData, Tidy};
use super::{DistanceSlot, DistanceSource, KernelScalar, SlotId, SlotShape};
use crate::error::{GprError, SlotErrorKind};
#[allow(
    unused_imports,
    reason = "the parent re-exports the whole split file; each user reads some"
)]
pub(crate) use blocks::*;
#[allow(
    unused_imports,
    reason = "the parent re-exports the whole split file; each user reads some"
)]
pub(crate) use check::*;
#[allow(
    unused_imports,
    reason = "the parent re-exports the whole split file; each user reads some"
)]
pub(crate) use query::*;
#[allow(
    unused_imports,
    reason = "the parent re-exports the whole split file; each user reads some"
)]
pub(crate) use refined::*;
#[allow(
    unused_imports,
    reason = "the parent re-exports the whole split file; each user reads some"
)]
pub(crate) use slot::*;
#[allow(
    unused_imports,
    reason = "the parent re-exports the whole split file; each user reads some"
)]
pub(crate) use train::*;

/// Whether `T` reads `f64` tables in place.
fn reads_in_place<T: ScalarOps>() -> bool {
    T::from_f64_slice(&[]).is_some()
}

#[cfg(test)]
mod tests;
