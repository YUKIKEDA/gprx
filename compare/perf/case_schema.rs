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

/// P4-12 Sparse cell. `y` is already population-standardized.
#[derive(Debug, Deserialize)]
pub struct SparseCase {
    pub name: String,
    pub model: String,
    pub ard: bool,
    pub n_rows: usize,
    pub n_cols: usize,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub z: Vec<f64>,
    pub n_inducing: usize,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub factor_min_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub factor_max_s: Option<f64>,
    pub eval_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_min_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_max_s: Option<f64>,
    pub predict_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub predict_min_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub predict_max_s: Option<f64>,
    pub joint_evals: Option<u64>,
    pub peak_rss_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warmup: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reps: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kernel_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub border_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rest_s: Option<f64>,
    pub note: Option<String>,
}

/// P4-14 Sparse-online cell. `y` is raw. Clock is one 32-op wall.
#[derive(Debug, Deserialize)]
pub struct SparseOnlineCase {
    pub name: String,
    pub ard: bool,
    pub n_rows: usize,
    pub n_cols: usize,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub z: Vec<f64>,
    pub n_inducing: usize,
    pub start_n: usize,
    pub start_m: usize,
    pub n_max: usize,
    pub m_max: usize,
    pub lengthscales_init: Vec<f64>,
    pub noise_variance_init: f64,
    pub ops: Vec<SparseOnlineOp>,
}

#[derive(Debug, Deserialize)]
pub struct SparseOnlineOp {
    pub kind: String,
    #[serde(default)]
    pub pop_index: Option<usize>,
    #[serde(default)]
    pub slot: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct SparseOnlineResult {
    pub lib: String,
    pub name: String,
    pub status: String,
    pub ops_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ops_min_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ops_max_s: Option<f64>,
    pub peak_rss_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warmup: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reps: Option<u64>,
    pub note: Option<String>,
}
