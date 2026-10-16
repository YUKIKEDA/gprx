//! VFE assembly, derivatives, rank-1 `X` updates, and inducing `m` updates.

use crate::kernel::KernelScalar;

mod derivatives;
mod loo;
mod predict;
mod updates;
mod vfe;

pub(crate) use derivatives::{analytic_gradient, analytic_hessian};
pub(crate) use loo::vfe_loo;
pub(crate) use predict::{VfeSystem, predict_vfe_covariance, predict_vfe_into};
pub(crate) use updates::{
    InducingBorder, a_times_y, append_point, coordinate_border, factor_pivots_hold,
    inducing_append, inducing_delete, kernel_column, kernel_diag_at, point_at, push_column,
    remove_column_in_place, remove_point,
};
pub(crate) use vfe::{
    VfeExact, VfeState, assemble_fitted, assemble_vfe, f64_weights, fill_z_intervals,
    publish_sgpr_weights, vfe_neg_log_marginal_likelihood,
};

pub(super) fn lit<T: KernelScalar>(value: f64) -> T {
    T::from_f64(value)
}
