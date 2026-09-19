//! Shared JSON case / result records for the Rust runners.

use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct Case {
    pub name: String,
    pub ard: bool,
    pub n_rows: usize,
    pub n_cols: usize,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub xs_n_rows: usize,
    pub xs_n_cols: usize,
    pub xs: Vec<f64>,
    pub lengthscales_init: Vec<f64>,
    pub noise_variance_init: f64,
    pub joint_evals: u64,
}

#[derive(Debug, Serialize)]
pub struct ResultRow {
    pub lib: String,
    pub name: String,
    pub status: String,
    pub factor_s: Option<f64>,
    pub eval_s: Option<f64>,
    pub predict_s: Option<f64>,
    pub joint_evals: Option<u64>,
    pub peak_rss_bytes: Option<u64>,
    pub note: Option<String>,
}
