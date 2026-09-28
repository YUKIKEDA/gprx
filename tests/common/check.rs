//! Tolerance checks shared by unit tests, integration tests, and benches.
//!
//! Every comparison is relative to `max(|expected|, 1)`, so values near zero
//! use an absolute tolerance. Unit tests include this file with `#[path]`.

use faer::{Mat, MatRef};

/// `|actual - expected| / max(|expected|, 1)`.
pub fn rel_err(actual: f64, expected: f64) -> f64 {
    (actual - expected).abs() / expected.abs().max(1.0)
}

/// Asserts [`rel_err`] `<= tol`.
#[track_caller]
pub fn assert_close(actual: f64, expected: f64, tol: f64) {
    assert!(
        rel_err(actual, expected) <= tol,
        "actual={actual}, expected={expected}, tol={tol}"
    );
}

/// [`assert_close`] with a label in the failure message.
#[track_caller]
pub fn assert_close_named(label: &str, actual: f64, expected: f64, tol: f64) {
    let err = rel_err(actual, expected);
    assert!(
        err <= tol,
        "{label}: actual={actual}, expected={expected}, rel_err={err}, tol={tol}"
    );
}

/// Same length, then [`assert_close`] element by element.
#[track_caller]
pub fn assert_slice_close(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected.iter()) {
        assert_close(*a, *e, tol);
    }
}

/// [`assert_slice_close`] with `label[i]` in the failure message.
#[track_caller]
pub fn assert_slice_close_named(label: &str, actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_close_named(&format!("{label}[{i}]"), *a, *e, tol);
    }
}

/// Every entry of two same-shape matrices within `tol`.
#[track_caller]
pub fn assert_mat_close(actual: MatRef<'_, f64>, expected: MatRef<'_, f64>, tol: f64) {
    assert_eq!(actual.nrows(), expected.nrows());
    assert_eq!(actual.ncols(), expected.ncols());
    for col in 0..actual.ncols() {
        for row in 0..actual.nrows() {
            assert_close(actual[(row, col)], expected[(row, col)], tol);
        }
    }
}

/// The lower triangle (`row >= col`) of two square matrices within `tol`.
#[track_caller]
pub fn assert_lower_close(actual: MatRef<'_, f64>, expected: MatRef<'_, f64>, tol: f64) {
    assert_eq!(actual.nrows(), expected.nrows());
    assert_eq!(actual.ncols(), expected.ncols());
    let n = actual.nrows();
    for col in 0..n {
        for row in col..n {
            assert_close(actual[(row, col)], expected[(row, col)], tol);
        }
    }
}

/// Compile-time check that `T` is `Send + Sync`.
pub fn assert_send_sync<T: Send + Sync>() {}

/// `n×n` matrix with every entry `value`.
pub fn fill(n: usize, value: f64) -> Mat<f64> {
    Mat::from_fn(n, n, |_, _| value)
}

/// Pairwise squared distances of 1-D points.
pub fn sq_dist_1d(x: &[f64]) -> Mat<f64> {
    let n = x.len();
    Mat::from_fn(n, n, |i, j| {
        let d = x[i] - x[j];
        d * d
    })
}

/// `n×2` points from row pairs.
pub fn points_2d(rows: &[[f64; 2]]) -> Mat<f64> {
    Mat::from_fn(rows.len(), 2, |i, j| rows[i][j])
}

/// Predictive mean and variance against expected slices.
#[track_caller]
pub fn assert_mean_var_close(
    mean: &[f64],
    variance: &[f64],
    want_mean: &[f64],
    want_variance: &[f64],
    tol: f64,
) {
    assert_slice_close(mean, want_mean, tol);
    assert_slice_close(variance, want_variance, tol);
}

/// [`assert_mean_var_close`] with `label mean[i]` / `label var[i]` messages.
#[track_caller]
pub fn assert_mean_var_close_named(
    label: &str,
    mean: &[f64],
    variance: &[f64],
    want_mean: &[f64],
    want_variance: &[f64],
    tol: f64,
) {
    assert_slice_close_named(&format!("{label} mean"), mean, want_mean, tol);
    assert_slice_close_named(&format!("{label} var"), variance, want_variance, tol);
}
