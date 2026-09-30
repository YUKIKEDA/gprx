//! Whitened SVGP assembly, ELBO, and diagonal prediction.

mod adam;
mod assemble;
mod gradient;
mod predict;

pub(crate) use adam::run_adam_fit;
pub(crate) use assemble::{
    assemble_fitted, assemble_svgp, pack_q, q_param_len, svgp_neg_elbo, unpack_q,
};
pub(crate) use gradient::svgp_value_and_gradient;
pub(crate) use predict::svgp_predict;
