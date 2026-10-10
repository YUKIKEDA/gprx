//! `f64x4` loops over an ARD cache laid out as row runs
//! ([`crate::kernel::dist::RowRuns`]): row `i` holds `(Δ_d)²` of the pairs
//! `(i, 0..=i)`, contiguous.
//!
//! The lower triangle of a column-major square is filled four rows at a
//! time: for a panel of rows `r0..r0 + 4` and four columns `j..j + 4`,
//! `r² = Σ_d w_d (Δ_d)²` is summed along each row's run (four contiguous
//! columns per load), the leaf's value is taken lane-wise, and a 4 × 4
//! transpose turns the four row vectors into four column vectors, each
//! stored as four contiguous rows of its output column. The pairs near the
//! diagonal take the same value one at a time. Panels run on the Rayon
//! pool in blocks of about equal area; nothing is allocated.

use super::{LANES, col_slice_mut, load4};
use crate::error::GprError;
use crate::kernel::dist::RowRuns;
use crate::math::f64x4_all_finite as all_finite;
use faer::MatMut;
use faer::reborrow::ReborrowMut;
use wide::f64x4;

/// Rows per panel.
const PANEL: usize = LANES;

/// Fills the lower triangle of the square `out` with `eval(r², t)` of every
/// pair `(i, j)`, `i ≥ j`: `r² = Σ_d w_d (Δ_d)²` and `t = w_p (Δ_p)²` of the
/// picked dimension `p` (zero when none is picked).
///
/// Returns `Ok(false)` without a usable result when `out` does not have
/// unit row stride or a value is not finite; the caller then runs its
/// scalar loop over the same output, which names the error.
pub(crate) fn try_fill_lower<F>(
    rows: RowRuns<'_, f64>,
    mut out: MatMut<'_, f64>,
    inv_ell_sq: &[f64],
    pick: Option<usize>,
    eval: &F,
) -> Result<bool, GprError>
where
    F: Fn(f64x4, f64x4) -> f64x4 + Sync,
{
    let n = rows.n();
    if out.nrows() != n || out.ncols() != n {
        return Ok(false);
    }
    if n > 1 && out.row_stride() != 1 {
        return Ok(false);
    }
    let panels = n.div_ceil(PANEL);
    let ok = split(out.rb_mut(), (0, panels), &|first, block| {
        fill_block(rows, block, first * PANEL, inv_ell_sq, pick, eval)
    });
    Ok(ok)
}

/// Runs `f(first_panel, rows)` over panels `lo..hi`, split by halving into
/// blocks of about equal lower-triangle area, joined on the Rayon pool when
/// it has more than one worker. `out` holds every row of the panels.
fn split<F>(out: MatMut<'_, f64>, (lo, hi): (usize, usize), f: &F) -> bool
where
    F: Fn(usize, MatMut<'_, f64>) -> bool + Sync,
{
    if hi - lo <= 1 || rayon::current_num_threads() <= 1 {
        return f(lo, out);
    }
    // Rows `0..r` hold `r(r + 1)/2` pairs; cut at the panel nearest half
    // the pairs of `lo..hi`.
    let area = |p: usize| {
        let r = p * PANEL;
        r * (r + 1) / 2
    };
    let half = (area(lo) + area(hi)) / 2;
    let mut mid = lo + 1;
    while mid + 1 < hi && area(mid) < half {
        mid += 1;
    }
    let cut = ((mid - lo) * PANEL).min(out.nrows());
    let (top, bottom) = out.split_at_row_mut(cut);
    let (a, b) = rayon::join(|| split(top, (lo, mid), f), || split(bottom, (mid, hi), f));
    a && b
}

/// Fills the lower-triangle pairs of rows `row0..row0 + block.nrows()`;
/// `block` holds those rows of every column.
fn fill_block<F>(
    rows: RowRuns<'_, f64>,
    mut block: MatMut<'_, f64>,
    row0: usize,
    inv_ell_sq: &[f64],
    pick: Option<usize>,
    eval: &F,
) -> bool
where
    F: Fn(f64x4, f64x4) -> f64x4,
{
    let end = row0 + block.nrows();
    let mut r0 = row0;
    while r0 < end {
        let len = PANEL.min(end - r0);
        let local = r0 - row0;
        // Columns `0..full` hold four rows of the panel below the diagonal.
        let full = if len == PANEL { r0 / LANES * LANES } else { 0 };
        let mut j = 0;
        while j < full {
            let mut v = [f64x4::ZERO; PANEL];
            for (lane, value) in v.iter_mut().enumerate() {
                let i = r0 + lane;
                let mut r2 = f64x4::ZERO;
                let mut t = f64x4::ZERO;
                for (dim, &w) in inv_ell_sq.iter().enumerate() {
                    let term = f64x4::splat(w) * load4(rows.row(dim, i), j);
                    r2 += term;
                    if pick == Some(dim) {
                        t = term;
                    }
                }
                *value = eval(r2, t);
                if !all_finite(*value) {
                    return false;
                }
            }
            let [a0, a1, a2, a3] = v.map(|x| x.to_array());
            for (c, column) in (j..j + LANES).enumerate() {
                let Some(dest) = col_slice_mut(block.rb_mut(), column) else {
                    return false;
                };
                dest[local..local + PANEL].copy_from_slice(&[a0[c], a1[c], a2[c], a3[c]]);
            }
            j += LANES;
        }
        // The pairs of columns `full..=i`, one at a time.
        for lane in 0..len {
            let i = r0 + lane;
            for col in full..=i {
                let mut r2 = 0.0;
                let mut t = 0.0;
                for (dim, &w) in inv_ell_sq.iter().enumerate() {
                    let term = w * rows.row(dim, i)[col];
                    r2 += term;
                    if pick == Some(dim) {
                        t = term;
                    }
                }
                let value = eval(f64x4::splat(r2), f64x4::splat(t)).to_array()[0];
                if !value.is_finite() {
                    return false;
                }
                block[(local + lane, col)] = value;
            }
        }
        r0 += len;
    }
    true
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::try_fill_lower;
    use crate::kernel::dist::ArdSqDiffBuf;
    use crate::kernel::{
        MaternArdKernel, MaternNu, RationalQuadraticArdKernel, RbfArdKernel, Triangle,
    };
    use crate::math::Accurate;
    use faer::Mat;
    use wide::f64x4;

    const D: usize = 3;

    /// Coordinate `k` of point `p`.
    fn coord(k: usize, p: usize) -> f64 {
        (p as f64 * (0.37 + 0.11 * k as f64)).sin() * (1.0 + k as f64)
    }

    fn pair(k: usize, i: usize, j: usize) -> f64 {
        let diff = coord(k, i) - coord(k, j);
        diff * diff
    }

    /// The same pairs as column runs and as row runs.
    fn caches(n: usize) -> (ArdSqDiffBuf<f64>, ArdSqDiffBuf<f64>) {
        let lower = ArdSqDiffBuf::<f64>::from_pairs(n, D, pair).expect("cache");
        let mut rows = lower.clone();
        rows.ready_to_change().expect("rows");
        (lower, rows)
    }

    /// Every pair of the lower triangle, and nothing above it, for each
    /// panel shape: no full panel, one, several, and a split on the pool.
    #[test]
    fn rows_fill_every_lower_pair_once() {
        let w = [0.7, 1.3, 2.1];
        let eval = |r2: f64x4, t: f64x4| r2 * f64x4::splat(2.0) + t;
        for n in [1, 3, 4, 5, 8, 13, 40] {
            let (_, rows) = caches(n);
            for pick in [None, Some(1)] {
                let mut out = Mat::<f64>::from_fn(n, n, |_, _| f64::NAN);
                let runs = rows.view().rows().expect("rows");
                assert!(try_fill_lower(runs, out.as_mut(), &w, pick, &eval).expect("fill"));
                for j in 0..n {
                    for i in 0..n {
                        if i < j {
                            assert!(out[(i, j)].is_nan(), "({i}, {j}) above the diagonal");
                            continue;
                        }
                        let mut r2 = 0.0;
                        let mut t = 0.0;
                        for (k, &wk) in w.iter().enumerate() {
                            let term = wk * pair(k, i, j);
                            r2 += term;
                            if pick == Some(k) {
                                t = term;
                            }
                        }
                        assert_eq!(
                            out[(i, j)].to_bits(),
                            (r2 * 2.0 + t).to_bits(),
                            "({i}, {j})"
                        );
                    }
                }
            }
        }
    }

    /// A value that is not finite hands the output back to the scalar loop.
    #[test]
    fn rows_fill_reports_a_value_that_is_not_finite() {
        let (_, rows) = caches(9);
        let mut out = Mat::<f64>::zeros(9, 9);
        let runs = rows.view().rows().expect("rows");
        let eval = |r2: f64x4, _| r2 / f64x4::ZERO;
        assert!(!try_fill_lower(runs, out.as_mut(), &[1.0; D], None, &eval).expect("fill"));
    }

    /// Each ARD leaf reads row runs as it reads column runs: values,
    /// gradients, and Hessians agree on the lower triangle.
    #[test]
    fn every_ard_leaf_reads_row_runs_as_column_runs() {
        let n = 23;
        let (lower, rows) = caches(n);
        let (lower, rows) = (lower.view(), rows.view());
        let rbf = RbfArdKernel::new(&[0.8, 1.4, 2.0]).expect("ell");
        let matern = MaternArdKernel::new(&[0.8, 1.4, 2.0], MaternNu::FiveHalves).expect("ell");
        let rq = RationalQuadraticArdKernel::new(&[0.8, 1.4, 2.0], 1.5).expect("ell");
        type Fill<'a> =
            Box<dyn Fn(crate::kernel::dist::ArdSqDiff<'_, f64>, faer::MatMut<'_, f64>) + 'a>;
        let fills: Vec<Fill<'_>> = vec![
            Box::new(|c, o| {
                rbf.apply_from_sq_diff::<Accurate, _>(c, o, Triangle::Lower)
                    .expect("rbf")
            }),
            Box::new(|c, o| {
                rbf.grad_from_sq_diff::<Accurate, _>(c, o, 1, Triangle::Lower)
                    .expect("rbf")
            }),
            Box::new(|c, o| {
                rbf.hess_from_sq_diff::<Accurate, _>(c, o, 0, 2, Triangle::Lower)
                    .expect("rbf")
            }),
            Box::new(|c, o| {
                matern
                    .apply_from_sq_diff::<Accurate, _>(c, o, Triangle::Lower)
                    .expect("matern")
            }),
            Box::new(|c, o| {
                matern
                    .grad_from_sq_diff::<Accurate, _>(c, o, 2, Triangle::Lower)
                    .expect("matern")
            }),
            Box::new(|c, o| {
                matern
                    .hess_from_sq_diff::<Accurate, _>(c, o, 1, 1, Triangle::Lower)
                    .expect("matern")
            }),
            Box::new(|c, o| {
                rq.apply_from_sq_diff::<Accurate, _>(c, o, Triangle::Lower)
                    .expect("rq")
            }),
            Box::new(|c, o| {
                rq.grad_from_sq_diff::<Accurate, _>(c, o, 0, Triangle::Lower)
                    .expect("rq")
            }),
            Box::new(|c, o| {
                rq.hess_from_sq_diff::<Accurate, _>(c, o, 0, 3, Triangle::Lower)
                    .expect("rq")
            }),
        ];
        for (at, fill) in fills.iter().enumerate() {
            let (mut a, mut b) = (Mat::<f64>::zeros(n, n), Mat::<f64>::zeros(n, n));
            fill(lower, a.as_mut());
            fill(rows, b.as_mut());
            for j in 0..n {
                for i in j..n {
                    let (x, y) = (a[(i, j)], b[(i, j)]);
                    assert!(
                        (x - y).abs() <= 1e-14 * x.abs().max(1.0),
                        "fill {at} ({i}, {j}): {x} vs {y}"
                    );
                }
            }
        }
        // The `f32` loop reads them alike too.
        let lower32 =
            ArdSqDiffBuf::<f32>::from_pairs(n, D, |k, i, j| pair(k, i, j) as f32).expect("cache");
        let mut rows32 = lower32.clone();
        rows32.ready_to_change().expect("rows");
        let (mut a, mut b) = (Mat::<f32>::zeros(n, n), Mat::<f32>::zeros(n, n));
        rbf.apply_from_sq_diff::<Accurate, _>(lower32.view(), a.as_mut(), Triangle::Lower)
            .expect("rbf");
        rbf.apply_from_sq_diff::<Accurate, _>(rows32.view(), b.as_mut(), Triangle::Lower)
            .expect("rbf");
        for j in 0..n {
            for i in j..n {
                assert_eq!(a[(i, j)].to_bits(), b[(i, j)].to_bits(), "({i}, {j})");
            }
        }
    }

    /// The gradient contraction reads row runs as it reads column runs.
    #[test]
    fn the_contraction_reads_row_runs_as_column_runs() {
        let n = 29;
        let (lower, rows) = caches(n);
        let rbf = RbfArdKernel::new(&[0.8, 1.4, 2.0]).expect("ell");
        let weight = Mat::<f64>::from_fn(n, n, |i, j| ((i * 7 + j * 3) % 11) as f64 - 5.0);
        let k = Mat::<f64>::from_fn(n, n, |i, j| 1.0 / (1.0 + (i + j) as f64));
        let (mut a, mut b) = ([0.0; D], [0.0; D]);
        let mut fold = Vec::new();
        rbf.contract_square_from_sq_diff(
            weight.as_ref(),
            k.as_ref(),
            lower.view(),
            &mut a,
            &mut fold,
        )
        .expect("lower");
        rbf.contract_square_from_sq_diff(
            weight.as_ref(),
            k.as_ref(),
            rows.view(),
            &mut b,
            &mut fold,
        )
        .expect("rows");
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() <= 1e-12 * x.abs().max(1.0), "{x} vs {y}");
        }
    }
}
