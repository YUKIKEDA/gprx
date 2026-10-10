//! Coordinates, their squared differences and prediction checks shared by
//! the supplied-distance tests.

use super::check::assert_slice_close;
use gprx::kernel::KernelScalar;
use gprx::{GaussianLikelihood, Prediction};

/// Coordinate `k` of `rows` samples, shifted by `offset` (queries use 0.5).
pub fn coord(k: usize, rows: usize, offset: f64) -> Vec<f64> {
    (0..rows)
        .map(|i| ((i as f64 + offset) * (0.41 + 0.17 * k as f64)).sin() * (1.0 + 0.5 * k as f64))
        .collect()
}

/// Column-major `a.len() × b.len()` squared differences.
pub fn sq(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(a.len() * b.len());
    for bj in b {
        for ai in a {
            out.push((ai - bj) * (ai - bj));
        }
    }
    out
}

/// `Σ_k` of the blocks of [`sq`].
pub fn sum(blocks: &[Vec<f64>]) -> Vec<f64> {
    (0..blocks.first().map_or(0, Vec::len))
        .map(|i| blocks.iter().map(|b| b[i]).sum())
        .collect()
}

/// `values` widened to `f64`.
pub fn to64<T: KernelScalar>(values: &[T]) -> Vec<f64> {
    values.iter().map(|v| v.to_f64()).collect()
}

/// Gaussian likelihood with noise `0.05`.
#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
pub fn lik() -> GaussianLikelihood {
    GaussianLikelihood::new(0.05).expect("noise")
}

/// The means and variances of `got` and `want` agree within `tol`.
#[track_caller]
pub fn assert_pred<T: KernelScalar>(got: &Prediction<T>, want: &Prediction<T>, tol: f64) {
    assert_slice_close(&to64(&got.mean), &to64(&want.mean), tol);
    assert_slice_close(&to64(&got.variance), &to64(&want.variance), tol);
}
