//! VFE assembly, derivatives, rank-1 `X` updates, and inducing `m` updates.

use crate::kernel::KernelScalar;

mod derivatives;
mod predict;
mod updates;
mod vfe;

pub(crate) use derivatives::{analytic_gradient, analytic_hessian};
pub(crate) use predict::vfe_predict;
pub(crate) use updates::{
    append_column, append_point, inducing_delete, inducing_insert, kernel_column, kernel_diag_at,
    point_at, remove_column, remove_point, solve_lmm,
};
pub(crate) use vfe::{
    VfeState, assemble_fitted, assemble_vfe, fill_z_intervals, publish_sgpr_weights, refresh_w,
    vfe_neg_log_marginal_likelihood,
};

pub(super) fn lit<T: KernelScalar>(value: f64) -> T {
    T::from_f64(value)
}
