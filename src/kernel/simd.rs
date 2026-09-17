//! Column-major SIMD helpers for squared distance and isotropic RBF.
//!
//! Uses [`wide::f64x4`]. When a view is not unit row-stride, callers keep the
//! scalar path. `wide::exp` may differ from scalar `f64::exp` by a few ULP.

use super::dist::{col_chunk, worker_count};
use super::{Triangle, finite_dist, require_same_shape, require_square_pair};
use crate::error::GprError;
use faer::reborrow::ReborrowMut;
use faer::{MatMut, MatRef};
use rayon::prelude::*;
use wide::f64x4;

const LANES: usize = 4;

fn load4(src: &[f64], i: usize) -> f64x4 {
    f64x4::new([src[i], src[i + 1], src[i + 2], src[i + 3]])
}

fn store4(dest: &mut [f64], i: usize, v: f64x4) {
    let a = v.to_array();
    dest[i] = a[0];
    dest[i + 1] = a[1];
    dest[i + 2] = a[2];
    dest[i + 3] = a[3];
}

fn all_finite4(v: f64x4) -> bool {
    let a = v.to_array();
    a[0].is_finite() && a[1].is_finite() && a[2].is_finite() && a[3].is_finite()
}

fn col_slice(mat: MatRef<'_, f64>, col: usize) -> Option<&[f64]> {
    mat.col(col).try_as_col_major().map(|c| c.as_slice())
}

fn col_slice_mut(mat: MatMut<'_, f64>, col: usize) -> Option<&mut [f64]> {
    mat.col_mut(col)
        .try_as_col_major_mut()
        .map(|c| c.as_slice_mut())
}

fn unit_row_stride(mat: MatRef<'_, f64>) -> bool {
    mat.ncols() == 0 || col_slice(mat, 0).is_some()
}

/// Adds `(x[i] - x0)²` into `acc[i]` with `f64x4` lanes.
pub(crate) fn add_squared_diff(x: &[f64], x0: f64, acc: &mut [f64]) {
    debug_assert_eq!(x.len(), acc.len());
    let x0v = f64x4::new([x0; LANES]);
    let mut i = 0;
    while i + LANES <= x.len() {
        let xv = load4(x, i);
        let av = load4(acc, i);
        let d = xv - x0v;
        store4(acc, i, av + d * d);
        i += LANES;
    }
    while i < x.len() {
        let d = x[i] - x0;
        acc[i] += d * d;
        i += 1;
    }
}

fn rbf_exp_slice(dist: &[f64], out: &mut [f64], inv_two_ell_sq: f64) -> Result<(), GprError> {
    debug_assert_eq!(dist.len(), out.len());
    let scale = f64x4::new([-inv_two_ell_sq; LANES]);
    let mut i = 0;
    while i + LANES <= dist.len() {
        let d = load4(dist, i);
        if !all_finite4(d) {
            return Err(GprError::NonFiniteInput);
        }
        store4(out, i, (d * scale).exp());
        i += LANES;
    }
    while i < dist.len() {
        let d = finite_dist(dist[i])?;
        out[i] = (-d * inv_two_ell_sq).exp();
        i += 1;
    }
    Ok(())
}

fn rbf_grad_slice(
    dist: &[f64],
    out: &mut [f64],
    inv_two_ell_sq: f64,
    inv_ell_sq: f64,
) -> Result<(), GprError> {
    debug_assert_eq!(dist.len(), out.len());
    let scale = f64x4::new([-inv_two_ell_sq; LANES]);
    let inv = f64x4::new([inv_ell_sq; LANES]);
    let mut i = 0;
    while i + LANES <= dist.len() {
        let d = load4(dist, i);
        if !all_finite4(d) {
            return Err(GprError::NonFiniteInput);
        }
        let k = (d * scale).exp();
        store4(out, i, k * d * inv);
        i += LANES;
    }
    while i < dist.len() {
        let d = finite_dist(dist[i])?;
        let k = (-d * inv_two_ell_sq).exp();
        out[i] = k * d * inv_ell_sq;
        i += 1;
    }
    Ok(())
}

/// Fills a lower-triangle distance chunk when `x` and `dist` are column-major.
///
/// Returns `false` if a view is not unit row-stride so the caller can use
/// the scalar loop.
pub(crate) fn try_fill_lower_chunk(
    x: MatRef<'_, f64>,
    mut dist_chunk: MatMut<'_, f64>,
    chunk_idx: usize,
    n_chunks: usize,
) -> bool {
    if !unit_row_stride(x) {
        return false;
    }
    let n = x.nrows();
    let d = x.ncols();
    let (start, len) = col_chunk(n, chunk_idx, n_chunks);
    if len > 0 && col_slice_mut(dist_chunk.rb_mut(), 0).is_none() {
        return false;
    }
    for local in 0..len {
        let col = start + local;
        let Some(dest) = col_slice_mut(dist_chunk.rb_mut(), local) else {
            return false;
        };
        let dest = &mut dest[col..];
        dest.fill(0.0);
        for dim in 0..d {
            let Some(xdim) = col_slice(x, dim) else {
                return false;
            };
            add_squared_diff(&xdim[col..], xdim[col], dest);
        }
    }
    true
}

/// Fills a rectangular train–test distance chunk when views are column-major.
pub(crate) fn try_fill_cross_chunk(
    x_train: MatRef<'_, f64>,
    x_test: MatRef<'_, f64>,
    mut dist_chunk: MatMut<'_, f64>,
    chunk_idx: usize,
    n_chunks: usize,
) -> bool {
    if !unit_row_stride(x_train) || !unit_row_stride(x_test) {
        return false;
    }
    let n = x_train.nrows();
    let d = x_train.ncols();
    let m = x_test.nrows();
    let (start, len) = col_chunk(m, chunk_idx, n_chunks);
    if len > 0 && col_slice_mut(dist_chunk.rb_mut(), 0).is_none() {
        return false;
    }
    for local in 0..len {
        let col = start + local;
        let Some(dest) = col_slice_mut(dist_chunk.rb_mut(), local) else {
            return false;
        };
        dest.fill(0.0);
        for dim in 0..d {
            let Some(x_tr) = col_slice(x_train, dim) else {
                return false;
            };
            let Some(x_te) = col_slice(x_test, dim) else {
                return false;
            };
            add_squared_diff(x_tr, x_te[col], dest);
        }
        debug_assert_eq!(dest.len(), n);
    }
    true
}

struct ColWindow {
    dist_col: usize,
    out_col: usize,
    row_start: usize,
    row_end: usize,
}

fn map_column_range(
    dist: MatRef<'_, f64>,
    mut out: MatMut<'_, f64>,
    window: ColWindow,
    f: impl Fn(&[f64], &mut [f64]) -> Result<(), GprError>,
) -> Result<(), GprError> {
    let Some(src) = col_slice(dist, window.dist_col) else {
        return Err(GprError::UnsupportedKernelOperation {
            reason: "expected unit row-stride for SIMD RBF".to_owned(),
        });
    };
    let Some(dest) = col_slice_mut(out.rb_mut(), window.out_col) else {
        return Err(GprError::UnsupportedKernelOperation {
            reason: "expected unit row-stride for SIMD RBF".to_owned(),
        });
    };
    f(
        &src[window.row_start..window.row_end],
        &mut dest[window.row_start..window.row_end],
    )
}

fn apply_rbf_range(
    dist: MatRef<'_, f64>,
    out: MatMut<'_, f64>,
    window: ColWindow,
    inv_two_ell_sq: f64,
) -> Result<(), GprError> {
    map_column_range(dist, out, window, |src, dest| {
        rbf_exp_slice(src, dest, inv_two_ell_sq)
    })
}

fn grad_rbf_range(
    dist: MatRef<'_, f64>,
    out: MatMut<'_, f64>,
    window: ColWindow,
    inv_two_ell_sq: f64,
    inv_ell_sq: f64,
) -> Result<(), GprError> {
    map_column_range(dist, out, window, |src, dest| {
        rbf_grad_slice(src, dest, inv_two_ell_sq, inv_ell_sq)
    })
}

fn rbf_lower_parallel(
    dist: MatRef<'_, f64>,
    out: MatMut<'_, f64>,
    inv_two_ell_sq: f64,
    inv_ell_sq: Option<f64>,
) -> Result<(), GprError> {
    let n = dist.nrows();
    let n_parts = worker_count();
    out.par_col_partition_mut(n_parts)
        .enumerate()
        .try_for_each(|(chunk_idx, mut part)| {
            let (start, len) = col_chunk(n, chunk_idx, n_parts);
            for local in 0..len {
                let col = start + local;
                match inv_ell_sq {
                    None => apply_rbf_range(
                        dist,
                        part.rb_mut(),
                        ColWindow {
                            dist_col: col,
                            out_col: local,
                            row_start: col,
                            row_end: n,
                        },
                        inv_two_ell_sq,
                    )?,
                    Some(inv_ell) => grad_rbf_range(
                        dist,
                        part.rb_mut(),
                        ColWindow {
                            dist_col: col,
                            out_col: local,
                            row_start: col,
                            row_end: n,
                        },
                        inv_two_ell_sq,
                        inv_ell,
                    )?,
                }
            }
            Ok(())
        })
}

/// Writes isotropic RBF `exp(-d / (2ℓ²))` when `dist`/`out` are column-major.
///
/// Returns `Ok(false)` if SIMD cannot run so the caller uses the scalar
/// [`super::write_triangle`] path.
pub(crate) fn try_apply_rbf(
    dist: MatRef<'_, f64>,
    mut out: MatMut<'_, f64>,
    uplo: Triangle,
    inv_two_ell_sq: f64,
) -> Result<bool, GprError> {
    let n = require_square_pair(dist, out.as_ref())?;
    if !unit_row_stride(dist) || !unit_row_stride(out.as_ref()) {
        return Ok(false);
    }
    match uplo {
        Triangle::Lower => rbf_lower_parallel(dist, out, inv_two_ell_sq, None)?,
        Triangle::Full => {
            for col in 0..n {
                apply_rbf_range(
                    dist,
                    out.rb_mut(),
                    ColWindow {
                        dist_col: col,
                        out_col: col,
                        row_start: 0,
                        row_end: n,
                    },
                    inv_two_ell_sq,
                )?;
            }
        }
        Triangle::Upper => {
            for col in 0..n {
                apply_rbf_range(
                    dist,
                    out.rb_mut(),
                    ColWindow {
                        dist_col: col,
                        out_col: col,
                        row_start: 0,
                        row_end: col + 1,
                    },
                    inv_two_ell_sq,
                )?;
            }
        }
    }
    Ok(true)
}

/// Writes rectangular RBF `k(X, X*)` when views are column-major.
pub(crate) fn try_apply_rbf_cross(
    dist: MatRef<'_, f64>,
    out: MatMut<'_, f64>,
    inv_two_ell_sq: f64,
) -> Result<bool, GprError> {
    require_same_shape(dist, out.as_ref())?;
    if !unit_row_stride(dist) || !unit_row_stride(out.as_ref()) {
        return Ok(false);
    }
    let n = dist.nrows();
    let m = dist.ncols();
    let n_parts = worker_count();
    out.par_col_partition_mut(n_parts)
        .enumerate()
        .try_for_each(|(chunk_idx, mut part)| {
            let (start, len) = col_chunk(m, chunk_idx, n_parts);
            for local in 0..len {
                let col = start + local;
                apply_rbf_range(
                    dist,
                    part.rb_mut(),
                    ColWindow {
                        dist_col: col,
                        out_col: local,
                        row_start: 0,
                        row_end: n,
                    },
                    inv_two_ell_sq,
                )?;
            }
            Ok(())
        })?;
    Ok(true)
}

/// Writes `∂k/∂θ = k · d / ℓ²` when views are column-major.
pub(crate) fn try_grad_rbf(
    dist: MatRef<'_, f64>,
    mut d_k: MatMut<'_, f64>,
    uplo: Triangle,
    inv_two_ell_sq: f64,
    inv_ell_sq: f64,
) -> Result<bool, GprError> {
    let n = require_square_pair(dist, d_k.as_ref())?;
    if !unit_row_stride(dist) || !unit_row_stride(d_k.as_ref()) {
        return Ok(false);
    }
    match uplo {
        Triangle::Lower => {
            rbf_lower_parallel(dist, d_k, inv_two_ell_sq, Some(inv_ell_sq))?;
        }
        Triangle::Full => {
            for col in 0..n {
                grad_rbf_range(
                    dist,
                    d_k.rb_mut(),
                    ColWindow {
                        dist_col: col,
                        out_col: col,
                        row_start: 0,
                        row_end: n,
                    },
                    inv_two_ell_sq,
                    inv_ell_sq,
                )?;
            }
        }
        Triangle::Upper => {
            for col in 0..n {
                grad_rbf_range(
                    dist,
                    d_k.rb_mut(),
                    ColWindow {
                        dist_col: col,
                        out_col: col,
                        row_start: 0,
                        row_end: col + 1,
                    },
                    inv_two_ell_sq,
                    inv_ell_sq,
                )?;
            }
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::{add_squared_diff, rbf_exp_slice, rbf_grad_slice};
    use crate::error::GprError;

    const TOL: f64 = 1e-12;

    fn assert_close(actual: f64, expected: f64) {
        let scale = expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= TOL * scale,
            "actual={actual}, expected={expected}"
        );
    }

    #[test]
    fn add_squared_diff_matches_scalar() {
        let x: Vec<f64> = (0..11).map(|i| i as f64 * 0.3).collect();
        let mut acc = vec![1.0; 11];
        let mut expected = acc.clone();
        let x0 = 1.25;
        add_squared_diff(&x, x0, &mut acc);
        for i in 0..11 {
            let d = x[i] - x0;
            expected[i] += d * d;
            assert_close(acc[i], expected[i]);
        }
    }

    #[test]
    fn rbf_exp_slice_matches_scalar_within_tol() {
        let dist: Vec<f64> = (0..10).map(|i| (i as f64) * 0.4).collect();
        let inv = 0.5;
        let mut out = vec![0.0; 10];
        rbf_exp_slice(&dist, &mut out, inv).expect("finite");
        for i in 0..10 {
            let expected = (-dist[i] * inv).exp();
            assert_close(out[i], expected);
        }
    }

    #[test]
    fn rbf_grad_slice_matches_scalar_within_tol() {
        let dist: Vec<f64> = (0..9).map(|i| 0.2 + i as f64).collect();
        let inv_two = 0.25;
        let inv_ell = 0.5;
        let mut out = vec![0.0; 9];
        rbf_grad_slice(&dist, &mut out, inv_two, inv_ell).expect("finite");
        for i in 0..9 {
            let k = (-dist[i] * inv_two).exp();
            assert_close(out[i], k * dist[i] * inv_ell);
        }
    }

    #[test]
    fn rbf_exp_slice_rejects_nan() {
        let dist = [0.0, 1.0, f64::NAN, 3.0, 4.0];
        let mut out = [0.0; 5];
        assert!(matches!(
            rbf_exp_slice(&dist, &mut out, 0.5),
            Err(GprError::NonFiniteInput)
        ));
    }
}
