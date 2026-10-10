//! Collapsed variational SGPR (Titsias / VFE) with caller-supplied inducing points.

mod distance;
pub(crate) mod factor;
mod fitted;
mod model;
mod objective;
mod online;

#[cfg(test)]
mod tests;

use crate::kernel::{CompiledKernel, KernelScalar, NoSupply, Supply};

pub use fitted::FittedSgpr;
pub use model::Sgpr;
pub(crate) use objective::SgprObjective;
pub(crate) use online::InducingRegistry;
pub use online::{InducingId, OnlineSgpr};

/// Keeps inducing coordinates fixed during [`Sgpr::fit`].
///
/// `Z` is an argument of `fit` / `factor` and is not a parameter.
///
/// See the example on [`Sgpr`].
#[derive(Clone, Copy, Debug, Default)]
pub struct FixedInducing;

/// Optimizes inducing coordinates jointly with kernel and likelihood `θ`.
///
/// Switch with [`Sgpr::with_inducing`]. Params append column-major `Z`
/// after kernel and likelihood `θ`.
///
/// See the example on [`Sgpr`].
#[derive(Clone, Copy, Debug, Default)]
pub struct FreeInducing;

/// How the inducing points of a model whose kernel tree is of kind `U` enter
/// its parameters. [`FreeInducing`] moves coordinates, so it is a layout of
/// coordinate trees ([`NoSupply`]) only: a model of supplied distances has
/// no `Z` to move.
pub trait InducingLayout<U: Supply>: Clone {
    fn z_params(m: usize, d: usize) -> usize;

    /// The coordinate tree whose derivatives the free `Z` reads, or `None`
    /// when `Z` is not a parameter.
    fn free_z<T: KernelScalar>(
        compiled: &CompiledKernel<T, U>,
    ) -> Option<&CompiledKernel<T, NoSupply>>;
}

impl<U: Supply> InducingLayout<U> for FixedInducing {
    fn z_params(_m: usize, _d: usize) -> usize {
        0
    }

    fn free_z<T: KernelScalar>(
        _compiled: &CompiledKernel<T, U>,
    ) -> Option<&CompiledKernel<T, NoSupply>> {
        None
    }
}

impl InducingLayout<NoSupply> for FreeInducing {
    fn z_params(m: usize, d: usize) -> usize {
        m * d
    }

    fn free_z<T: KernelScalar>(
        compiled: &CompiledKernel<T, NoSupply>,
    ) -> Option<&CompiledKernel<T, NoSupply>> {
        Some(compiled)
    }
}
