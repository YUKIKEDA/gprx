//! Crate-private pieces shared by the sparse models ([`crate::Sgpr`] and
//! [`crate::Svgp`]): the settings every trainer holds, the training data
//! every fitted model holds, and the kernel + likelihood `θ` over both.

mod core;
mod scratch;
mod supply;

use std::fmt;

use dyn_stack::MemBuffer;
use faer::{Mat, MatMut, MatRef};

#[allow(
    unused_imports,
    reason = "the parent re-exports the whole split file; each user reads some"
)]
pub(crate) use self::core::*;
use crate::data::{validate_inducing, validate_query, validate_training};
use crate::error::GprError;
use crate::kernel::{BlockStore, KernelSpec, Supply, SupplyViews};
use crate::kernel::{
    CompiledKernel, CrossViews, DiagAccum, GramInputs, KernelScalar, NoSupply, Triangle,
    WeightedWalk,
};
use crate::likelihood::GaussianLikelihood;
use crate::param::{Interval, write_params};
use crate::policy::KernelExp;
use crate::policy::{AdaptiveJitter, JitterPolicy};
use crate::precision::{InverseBuffers, ModelPrecision};
use crate::prediction::{Prediction, PredictiveCovariance};
use crate::transform::{
    IdentityInput, IdentityTarget, TargetTransform, Transform, UnfittedTarget, UnfittedTransform,
};
#[allow(
    unused_imports,
    reason = "the parent re-exports the whole split file; each user reads some"
)]
pub(crate) use scratch::*;
#[allow(
    unused_imports,
    reason = "the parent re-exports the whole split file; each user reads some"
)]
pub(crate) use supply::*;

/// Public read accessors of a fitted sparse model, from its `core:
/// SparseCore` field (or the field path given, such as `state.core`). One
/// set of docs for [`crate::FittedSgpr`], [`crate::OnlineSgpr`], and
/// [`crate::FittedSvgp`], whatever their kernel.
macro_rules! sparse_core_accessors {
    () => {
        $crate::sparse::sparse_core_accessors!(core);
    };
    ($($core:ident).+) => {
        /// Returns the number of training points.
        pub fn n(&self) -> usize {
            self.$($core).+.n
        }

        /// Returns the number of inducing points.
        pub fn m(&self) -> usize {
            self.$($core).+.m
        }

        /// Returns the observation-noise model.
        pub fn likelihood(&self) -> &$crate::GaussianLikelihood {
            &self.$($core).+.likelihood
        }

        /// Returns the jitter retries used when `K_mm` fails to factor.
        pub fn jitter_policy(&self) -> $crate::JitterPolicy {
            self.$($core).+.jitter
        }

        /// Returns the kernel `exp` mode the trainer set with `with_math`.
        pub fn math(&self) -> $crate::KernelExp {
            self.$($core).+.math
        }

        /// Returns the original training targets.
        pub fn y(&self) -> &[f64] {
            &self.$($core).+.y_obs
        }
    };
}

/// The coordinate accessors of a fitted sparse model whose kernel reads
/// coordinates ([`crate::kernel::PointKernel`]).
macro_rules! sparse_point_accessors {
    () => {
        $crate::sparse::sparse_point_accessors!(core);
    };
    ($($core:ident).+) => {
        /// Returns the feature dimension.
        pub fn d(&self) -> usize {
            self.$($core).+.d
        }

        /// Returns the original training features in column-major order.
        pub fn x(&self) -> &[f64] {
            &self.$($core).+.x_obs
        }

        /// Returns the inducing features in column-major order, in the original coordinates of `X`.
        pub fn z(&self) -> &[f64] {
            &self.$($core).+.z_obs
        }
    };
}

/// The kernel accessor of a fitted sparse model on coordinates.
macro_rules! sparse_kernel_accessor {
    () => {
        $crate::sparse::sparse_kernel_accessor!(core);
    };
    ($($core:ident).+) => {
        /// Returns the kernel whose hyperparameters this model owns.
        pub fn kernel(&self) -> &$crate::kernel::KernelSpec {
            &self.$($core).+.kernel
        }
    };
}

pub(crate) use sparse_core_accessors;
pub(crate) use sparse_kernel_accessor;
pub(crate) use sparse_point_accessors;

#[cfg(test)]
mod tests;
