//! Whitened SVGP assembly, ELBO, and diagonal prediction.

mod adam;
mod assemble;
mod gradient;
mod predict;
mod step;

pub(crate) use adam::run_adam_fit;
pub(crate) use assemble::{
    assemble_data_terms, assemble_fitted, assemble_kmm, assemble_svgp, pack_q, q_param_len,
    svgp_neg_elbo, unpack_q,
};
pub(crate) use gradient::svgp_value_and_gradient;
pub(crate) use predict::{SvgpSystem, predict_svgp_covariance, predict_svgp_into};
#[cfg(test)]
pub(crate) use step::AdamStep;
