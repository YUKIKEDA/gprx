//! Spike for faer 0.24 `ldlt::update::delete_rows_and_cols_clobber`.
//!
//! After delete, faer packs the remaining LDLT into the leading
//! `(n − r)×(n − r)` of the same matrix (`delete_rows_and_cols_triangular`,
//! then a rank update from the first removed index). This file reconstructs
//! `A = L D Lᵀ` from that block and compares it to a full LDLT of the
//! reduced matrix. This is not a Gaussian process test.

use dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::ldlt;
use faer::{Mat, MatMut, MatRef, Par, mat};

const TOL: f64 = 1e-12;

fn assert_close(actual: f64, expected: f64) {
    let scale = expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= TOL * scale,
        "actual={actual}, expected={expected}"
    );
}

fn assert_lower_close(actual: MatRef<'_, f64>, expected: MatRef<'_, f64>) {
    assert_eq!(actual.nrows(), expected.nrows());
    assert_eq!(actual.ncols(), expected.ncols());
    let n = actual.nrows();
    for j in 0..n {
        for i in j..n {
            assert_close(actual[(i, j)], expected[(i, j)]);
        }
    }
}

fn reconstruct_a(ld: MatRef<'_, f64>) -> Mat<f64> {
    let n = ld.nrows();
    assert_eq!(n, ld.ncols());
    let mut l = Mat::zeros(n, n);
    for j in 0..n {
        l[(j, j)] = 1.0;
        for i in (j + 1)..n {
            l[(i, j)] = ld[(i, j)];
        }
    }
    let mut ld_times = l.clone();
    for j in 0..n {
        let d = ld[(j, j)];
        for i in j..n {
            ld_times[(i, j)] *= d;
        }
    }
    &ld_times * l.transpose()
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn factor_ldlt(a: MatMut<'_, f64>, stack: &mut MemStack) {
    ldlt::factor::cholesky_in_place(a, Default::default(), Par::Seq, stack, Default::default())
        .expect("SPD test matrix must factor");
}

fn stack_for(n: usize, n_removed: usize) -> MemBuffer {
    let factor = ldlt::factor::cholesky_in_place_scratch::<f64>(n, Par::Seq, Default::default());
    let delete = ldlt::update::delete_rows_and_cols_clobber_scratch::<f64>(n, n_removed);
    MemBuffer::new(factor.and(delete))
}

fn drop_rows_cols(a: MatRef<'_, f64>, indices: &[usize]) -> Mat<f64> {
    let n = a.nrows();
    let keep: Vec<usize> = (0..n).filter(|i| !indices.contains(i)).collect();
    let m = keep.len();
    Mat::from_fn(m, m, |i, j| a[(keep[i], keep[j])])
}

fn delete_matches_full_factor(a: MatRef<'_, f64>, indices: &mut [usize]) {
    let n = a.nrows();
    let r = indices.len();
    let m = n - r;
    let expected_a = drop_rows_cols(a, indices);

    let mut memory = stack_for(n, r);
    let stack = MemStack::new(&mut memory);

    let mut ld = a.cloned();
    factor_ldlt(ld.as_mut(), stack);
    ldlt::update::delete_rows_and_cols_clobber(ld.as_mut(), indices, Par::Seq, stack);
    let deleted = reconstruct_a(ld.as_ref().submatrix(0, 0, m, m));

    let mut full = expected_a.clone();
    factor_ldlt(full.as_mut(), stack);
    let from_full = reconstruct_a(full.as_ref());

    assert_lower_close(deleted.as_ref(), from_full.as_ref());
    assert_lower_close(deleted.as_ref(), expected_a.as_ref());
}

fn two_by_two() -> Mat<f64> {
    mat![[4.0, 1.0], [1.0, 3.0]]
}

fn five_by_five() -> Mat<f64> {
    mat![
        [6.0, 1.0, 0.5, 0.0, 0.0],
        [1.0, 5.0, 1.0, 0.4, 0.0],
        [0.5, 1.0, 6.0, 1.0, 0.3],
        [0.0, 0.4, 1.0, 5.0, 1.0],
        [0.0, 0.0, 0.3, 1.0, 6.0],
    ]
}

#[test]
fn two_by_two_delete_first_matches_full_ldlt() {
    delete_matches_full_factor(two_by_two().as_ref(), &mut [0]);
}

#[test]
fn two_by_two_delete_last_matches_full_ldlt() {
    delete_matches_full_factor(two_by_two().as_ref(), &mut [1]);
}

#[test]
fn five_by_five_delete_first_matches_full_ldlt() {
    delete_matches_full_factor(five_by_five().as_ref(), &mut [0]);
}

#[test]
fn five_by_five_delete_middle_matches_full_ldlt() {
    delete_matches_full_factor(five_by_five().as_ref(), &mut [2]);
}

#[test]
fn five_by_five_delete_last_matches_full_ldlt() {
    delete_matches_full_factor(five_by_five().as_ref(), &mut [4]);
}

#[test]
fn five_by_five_delete_nonadjacent_pair_matches_full_ldlt() {
    delete_matches_full_factor(five_by_five().as_ref(), &mut [1, 3]);
}
