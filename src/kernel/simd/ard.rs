//! Vectorized pair loops for the Matérn-ARD and RQ-ARD leaves.
//!
//! A column of the output is filled four rows at a time: `r² = Σ_d w_d Δ_d²`
//! is accumulated over contiguous rows (from the coordinates or the
//! `(Δx_d)²` cache) into stack blocks, then a [`Profile`] turns `r²` and the
//! picked dimension's term `w_d Δ_d²` into four values with `f64x4`
//! arithmetic. Nothing is allocated. A layout these loops cannot read
//! (non-unit row stride), or a value that is not finite, makes them report
//! `false`, and the caller runs its scalar loop, which also names the error.

use super::col_slice;
use crate::error::GprError;
use crate::kernel::Triangle;
use crate::kernel::dist::ArdSqDiff;
use faer::{MatMut, MatRef};
use wide::f64x4;

/// Rows per stack block.
const BLOCK: usize = 256;

/// Where `Δ_d²` of a pair `(row, col)` comes from.
#[derive(Clone, Copy)]
pub(crate) enum Source<'a> {
    /// `x[row, d] − y[col, d]`.
    Points {
        x: MatRef<'a, f64>,
        y: MatRef<'a, f64>,
    },
    /// The `(Δx_d)²` cache, as column runs or row runs. Only
    /// [`Rows::Square`] [`Triangle::Lower`] reads it; other triangles fall
    /// back to the scalar loop.
    Cache { cache: ArdSqDiff<'a, f64> },
}

/// Which rows of each column a loop writes.
#[derive(Clone, Copy)]
pub(crate) enum Rows {
    /// A square output, by triangle.
    Square(Triangle),
    /// Every row of a rectangular output.
    All,
}

/// The value of four pairs from `r²` and the picked term `t = w_d Δ_d²`
/// (zero when no dimension is picked).
pub(crate) trait Profile: Sync {
    fn eval(&self, r2: f64x4, t: f64x4) -> f64x4;
}

/// Fills `out` (`rows` of each column) with `profile` of every pair.
///
/// Returns `Ok(false)` without a usable result when a view does not have
/// unit row stride or a value is not finite; the caller then runs its
/// scalar loop over the same output.
pub(crate) fn try_fill<P: Profile>(
    source: Source<'_>,
    mut out: MatMut<'_, f64>,
    inv_ell_sq: &[f64],
    pick: Option<usize>,
    rows: Rows,
    profile: &P,
) -> Result<bool, GprError> {
    if out.nrows() > 0 && out.row_stride() != 1 {
        return Ok(false);
    }
    match source {
        Source::Points { x, y } => {
            if (x.nrows() > 0 && x.row_stride() != 1) || (y.nrows() > 0 && y.row_stride() != 1) {
                return Ok(false);
            }
        }
        Source::Cache { cache } => {
            if !matches!(rows, Rows::Square(Triangle::Lower)) {
                return Ok(false);
            }
            if let Some(runs) = cache.rows() {
                return super::rows::try_fill_lower(runs, out, inv_ell_sq, pick, &|r2, t| {
                    profile.eval(r2, t)
                });
            }
        }
    }
    let n_rows = out.nrows();
    let mut r2 = [0.0f64; BLOCK];
    let mut t = [0.0f64; BLOCK];
    for col in 0..out.ncols() {
        let (begin, end) = match rows {
            Rows::All | Rows::Square(Triangle::Full) => (0, n_rows),
            Rows::Square(Triangle::Lower) => (col, n_rows),
            Rows::Square(Triangle::Upper) => (0, (col + 1).min(n_rows)),
        };
        let mut start = begin;
        while start < end {
            let len = BLOCK.min(end - start);
            let (r2, t) = (&mut r2[..len], &mut t[..len]);
            r2.fill(0.0);
            t.fill(0.0);
            for (dim, &w) in inv_ell_sq.iter().enumerate() {
                let picked = pick == Some(dim);
                match source {
                    Source::Points { x, y } => {
                        let z = y[(col, dim)];
                        let Some(xs) = column(x, dim, start, len) else {
                            return Ok(false);
                        };
                        add_weighted(xs, |v| (v - z) * (v - z), w, r2, picked.then_some(&mut *t));
                    }
                    Source::Cache { cache } => {
                        let Some(lower) = cache.lower() else {
                            return Ok(false);
                        };
                        let stored = lower.column(dim, col);
                        let offset = start - col;
                        let sq = &stored[offset..offset + len];
                        add_weighted(sq, |v| v, w, r2, picked.then_some(&mut *t));
                    }
                }
            }
            let mut i = 0;
            while i < len {
                let lanes = (len - i).min(4);
                let mut rr = [0.0; 4];
                let mut tt = [0.0; 4];
                rr[..lanes].copy_from_slice(&r2[i..i + lanes]);
                tt[..lanes].copy_from_slice(&t[i..i + lanes]);
                let v = profile.eval(f64x4::new(rr), f64x4::new(tt)).to_array();
                for (lane, &value) in v.iter().take(lanes).enumerate() {
                    if !value.is_finite() {
                        return Ok(false);
                    }
                    out[(start + i + lane, col)] = value;
                }
                i += lanes;
            }
            start += len;
        }
    }
    Ok(true)
}

/// Rows `start..start + len` of column `col` of `m`, when `m` is
/// column-major.
fn column(m: MatRef<'_, f64>, col: usize, start: usize, len: usize) -> Option<&[f64]> {
    col_slice(m, col).map(|values| &values[start..start + len])
}

/// `r2 += w · sq(v)` over `src`, and `t = w · sq(v)` when `t` is given.
#[inline(always)]
fn add_weighted(
    src: &[f64],
    sq: impl Fn(f64) -> f64,
    w: f64,
    r2: &mut [f64],
    t: Option<&mut [f64]>,
) {
    match t {
        Some(t) => {
            for ((acc, slot), &v) in r2.iter_mut().zip(t.iter_mut()).zip(src) {
                let term = w * sq(v);
                *acc += term;
                *slot = term;
            }
        }
        None => {
            for (acc, &v) in r2.iter_mut().zip(src) {
                *acc += w * sq(v);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::kernel::ard::{Pick, r2_from_cache, r2_from_coords};
    use crate::kernel::{
        ArdSqDiffBuf, MaternArdKernel, MaternNu, RationalQuadraticArdKernel, Triangle,
    };
    use crate::math::{Accurate, FastApprox, KernelMath};
    use faer::Mat;

    /// Points with several rows (more than one stack block of four and a
    /// tail), and a pair that coincides (`r = 0`).
    fn points(n: usize, d: usize, shift: f64) -> Mat<f64> {
        Mat::from_fn(n, d, |i, j| {
            if i == 1 {
                0.3 * j as f64
            } else {
                ((i * 37 + j * 11) % 23) as f64 / 7.0 - 1.4 + shift
            }
        })
    }

    fn assert_close(got: f64, want: f64, what: &str) {
        let scale = want.abs().max(1.0);
        assert!(
            (got - want).abs() <= 1e-13 * scale,
            "{what}: {got} vs {want}"
        );
    }

    fn check_matern<M: KernelMath>(nu: MaternNu) {
        let (n, d) = (11, 3);
        let x = points(n, d, 0.0);
        let mut x0 = x.clone();
        for j in 0..d {
            x0[(0, j)] = 0.3 * j as f64; // equals row 1
        }
        let xs = points(7, d, 0.25);
        let kernel = MaternArdKernel::new(&[0.8, 1.7, 1.1], nu).expect("ell");
        let w = kernel.lengthscales().inv_ell_sq().to_vec();
        let cache = ArdSqDiffBuf::new(x0.as_ref()).expect("cache");
        for p in [None, Some(0), Some(2)] {
            let mut out = Mat::zeros(n, n);
            let mut from_cache = Mat::zeros(n, n);
            match p {
                None => {
                    kernel
                        .apply_math::<M, f64>(x0.as_ref(), out.as_mut(), Triangle::Full)
                        .expect("apply");
                    kernel
                        .apply_from_sq_diff::<M, f64>(
                            cache.view(),
                            from_cache.as_mut(),
                            Triangle::Full,
                        )
                        .expect("cache");
                }
                Some(idx) => {
                    kernel
                        .grad_math::<M, f64>(x0.as_ref(), out.as_mut(), idx, Triangle::Full)
                        .expect("grad");
                    kernel
                        .grad_from_sq_diff::<M, f64>(
                            cache.view(),
                            from_cache.as_mut(),
                            idx,
                            Triangle::Full,
                        )
                        .expect("cache");
                }
            }
            for col in 0..n {
                for row in 0..n {
                    let pick = p.map_or(Pick::NONE, Pick::one);
                    let t =
                        r2_from_coords(x0.as_ref(), row, x0.as_ref(), col, &w, pick).expect("r2");
                    let want = match p {
                        None => {
                            crate::kernel::matern::matern_from_r::<M, f64>(nu, t.r2.max(0.0).sqrt())
                        }
                        Some(_) => crate::kernel::matern::matern_dk_dtheta_ard::<M, f64>(
                            nu,
                            t.r2.max(0.0).sqrt(),
                            t.dim_i,
                        ),
                    };
                    assert_close(
                        out[(row, col)],
                        want,
                        &format!("points {p:?} ({row}, {col})"),
                    );
                    // The cache holds the lower triangle, which `Lower` reads.
                    if row >= col {
                        let tc = r2_from_cache(cache.view(), row, col, &w, pick).expect("r2");
                        assert_close(tc.r2, t.r2, "cache r2");
                        assert_close(
                            from_cache[(row, col)],
                            want,
                            &format!("cache {p:?} ({row}, {col})"),
                        );
                    }
                }
            }
            let mut cross = Mat::zeros(n, 7);
            match p {
                None => kernel
                    .apply_cross_math::<M, f64>(x0.as_ref(), xs.as_ref(), cross.as_mut())
                    .expect("cross"),
                Some(idx) => kernel
                    .grad_cross_from_coords::<M, f64>(x0.as_ref(), xs.as_ref(), cross.as_mut(), idx)
                    .expect("cross grad"),
            }
            for col in 0..7 {
                for row in 0..n {
                    let pick = p.map_or(Pick::NONE, Pick::one);
                    let t =
                        r2_from_coords(x0.as_ref(), row, xs.as_ref(), col, &w, pick).expect("r2");
                    let r = t.r2.max(0.0).sqrt();
                    let want = match p {
                        None => crate::kernel::matern::matern_from_r::<M, f64>(nu, r),
                        Some(_) => {
                            crate::kernel::matern::matern_dk_dtheta_ard::<M, f64>(nu, r, t.dim_i)
                        }
                    };
                    assert_close(
                        cross[(row, col)],
                        want,
                        &format!("cross {p:?} ({row}, {col})"),
                    );
                }
            }
        }
    }

    #[test]
    fn matern_simd_matches_scalar_formulas() {
        for nu in [MaternNu::Half, MaternNu::ThreeHalves, MaternNu::FiveHalves] {
            check_matern::<Accurate>(nu);
            check_matern::<FastApprox>(nu);
        }
    }

    #[test]
    fn rq_simd_matches_scalar_formulas() {
        use crate::kernel::rq::{rq_dk_dtheta_alpha, rq_dk_dtheta_ard_dim, rq_from_r2};
        let (n, d) = (9, 2);
        let x = points(n, d, 0.0);
        let xs = points(6, d, -0.3);
        let alpha = 1.3;
        let kernel = RationalQuadraticArdKernel::new(&[0.9, 1.6], alpha).expect("rq");
        let w = kernel.lengthscales().inv_ell_sq().to_vec();
        let cache = ArdSqDiffBuf::new(x.as_ref()).expect("cache");
        for p in [None, Some(0), Some(1), Some(2)] {
            let mut out = Mat::zeros(n, n);
            let mut from_cache = Mat::zeros(n, n);
            let mut cross = Mat::zeros(n, 6);
            match p {
                None => {
                    kernel
                        .apply(x.as_ref(), out.as_mut(), Triangle::Lower)
                        .expect("apply");
                    kernel
                        .apply_from_sq_diff::<crate::math::Accurate, _>(
                            cache.view(),
                            from_cache.as_mut(),
                            Triangle::Lower,
                        )
                        .expect("cache");
                    kernel
                        .apply_cross(x.as_ref(), xs.as_ref(), cross.as_mut())
                        .expect("cross");
                }
                Some(idx) => {
                    kernel
                        .grad(x.as_ref(), out.as_mut(), idx, Triangle::Lower)
                        .expect("grad");
                    kernel
                        .grad_from_sq_diff::<crate::math::Accurate, _>(
                            cache.view(),
                            from_cache.as_mut(),
                            idx,
                            Triangle::Lower,
                        )
                        .expect("cache");
                    kernel
                        .grad_cross_from_coords(x.as_ref(), xs.as_ref(), cross.as_mut(), idx)
                        .expect("cross");
                }
            }
            let entry = |t: crate::kernel::ard::ArdR2<f64>| {
                let r2 = t.r2.max(0.0);
                match p {
                    None => rq_from_r2(r2, alpha),
                    Some(2) => rq_dk_dtheta_alpha(r2, alpha),
                    Some(_) => rq_dk_dtheta_ard_dim(r2, alpha, t.dim_i),
                }
            };
            let pick = match p {
                Some(idx) if idx < d => Pick::one(idx),
                _ => Pick::NONE,
            };
            for col in 0..n {
                for row in col..n {
                    let want = entry(
                        r2_from_coords(x.as_ref(), row, x.as_ref(), col, &w, pick).expect("r2"),
                    );
                    assert_close(out[(row, col)], want, &format!("points {p:?}"));
                    assert_close(from_cache[(row, col)], want, &format!("cache {p:?}"));
                }
                // `Lower` leaves the upper triangle untouched.
                for row in 0..col {
                    assert_eq!(out[(row, col)].to_bits(), 0.0f64.to_bits());
                }
            }
            for col in 0..6 {
                for row in 0..n {
                    let want = entry(
                        r2_from_coords(x.as_ref(), row, xs.as_ref(), col, &w, pick).expect("r2"),
                    );
                    assert_close(cross[(row, col)], want, &format!("cross {p:?}"));
                }
            }
        }
    }
}
