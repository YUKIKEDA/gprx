//! JSON case / result records of every runner mode. `friedrich-perf`
//! includes this file by path, so it does not use gprx.

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

/// One factor / joint / predict (or online insert / delete) cell.
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

impl ResultRow {
    /// The row of a cell that could not run.
    pub fn na(lib: &str, name: &str, note: String) -> Self {
        Self {
            lib: lib.to_string(),
            name: name.to_string(),
            status: "na".to_string(),
            factor_s: None,
            factor_min_s: None,
            factor_max_s: None,
            eval_s: None,
            eval_min_s: None,
            eval_max_s: None,
            predict_s: None,
            predict_min_s: None,
            predict_max_s: None,
            joint_evals: None,
            peak_rss_bytes: None,
            warmup: None,
            reps: None,
            kernel_s: None,
            border_s: None,
            rest_s: None,
            note: Some(note),
        }
    }
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

impl SparseOnlineResult {
    /// The row of a cell that could not run.
    pub fn na(lib: &str, name: &str, note: String) -> Self {
        Self {
            lib: lib.to_string(),
            name: name.to_string(),
            status: "na".to_string(),
            ops_s: None,
            ops_min_s: None,
            ops_max_s: None,
            peak_rss_bytes: None,
            warmup: None,
            reps: None,
            note: Some(note),
        }
    }
}

/// B1-1 real-dataset cell. `x` / `xs` are column-major and standardized, `y`
/// is standardized with the training statistics, `ys` is in original units.
#[derive(Debug, Deserialize)]
pub struct RealCase {
    pub name: String,
    pub n_rows: usize,
    pub n_cols: usize,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub xs_n_rows: usize,
    pub xs: Vec<f64>,
    pub ys: Vec<f64>,
    pub y_mean: f64,
    pub y_std: f64,
    pub lengthscale_init: f64,
    pub signal_variance_init: f64,
    pub noise_variance_init: f64,
    /// `native` (the library default) or `matched` (`max_iterations` / `gtol`).
    pub protocol: String,
    pub max_iterations: u64,
    pub gtol: f64,
}

/// One real-dataset fit row (also written by the Python runners).
#[derive(Debug, Serialize)]
pub struct FitRow {
    pub lib: String,
    pub name: String,
    pub status: String,
    pub protocol: String,
    pub fit_s: Option<f64>,
    pub predict_s: Option<f64>,
    pub joint_evals: Option<u64>,
    pub value_evals: Option<u64>,
    pub iterations: Option<u64>,
    pub nlml: Option<f64>,
    pub rmse: Option<f64>,
    pub nlpd: Option<f64>,
    pub coverage95: Option<f64>,
    pub peak_rss_bytes: Option<u64>,
    pub note: Option<String>,
}

impl FitRow {
    /// The row of a cell that could not run.
    pub fn na(lib: &str, name: &str, protocol: &str, note: String) -> Self {
        Self {
            lib: lib.to_string(),
            name: name.to_string(),
            status: "na".to_string(),
            protocol: protocol.to_string(),
            fit_s: None,
            predict_s: None,
            joint_evals: None,
            value_evals: None,
            iterations: None,
            nlml: None,
            rmse: None,
            nlpd: None,
            coverage95: None,
            peak_rss_bytes: None,
            note: Some(note),
        }
    }
}
