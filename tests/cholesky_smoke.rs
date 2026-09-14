//! Smoke test for faer 0.24 in-place LLT (`cholesky_in_place`).
//!
//! Pins the `MemStack` + `LltRegularization` calling convention used later by
//! Exact GP. This is not a Gaussian process test.

use dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt;
use faer::linalg::cholesky::llt::factor::{LltError, LltInfo, LltRegularization};
use faer::{Mat, MatMut, MatRef, Par, mat};

const TOL: f64 = 1e-12;

fn assert_close(actual: f64, expected: f64) {
    let scale = expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= TOL * scale,
        "actual={actual}, expected={expected}"
    );
}

fn assert_mat_close(actual: MatRef<'_, f64>, expected: MatRef<'_, f64>) {
    assert_eq!(actual.nrows(), expected.nrows());
    assert_eq!(actual.ncols(), expected.ncols());
    for j in 0..actual.ncols() {
        for i in 0..actual.nrows() {
            assert_close(actual[(i, j)], expected[(i, j)]);
        }
    }
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

fn copy_lower(src: MatRef<'_, f64>) -> Mat<f64> {
    let n = src.nrows();
    Mat::from_fn(n, n, |i, j| if i >= j { src[(i, j)] } else { 0.0 })
}

fn factor_in_place(a: MatMut<'_, f64>, stack: &mut MemStack) -> Result<LltInfo, LltError> {
    let regularization = LltRegularization {
        dynamic_regularization_delta: 0.0,
        dynamic_regularization_epsilon: 0.0,
    };
    llt::factor::cholesky_in_place(a, regularization, Par::Seq, stack, Default::default())
}

fn stack_for(n: usize, rhs_ncols: usize) -> MemBuffer {
    let chol = llt::factor::cholesky_in_place_scratch::<f64>(n, Par::Seq, Default::default());
    let solve = llt::solve::solve_in_place_scratch::<f64>(n, rhs_ncols, Par::Seq);
    MemBuffer::new(chol.or(solve))
}

fn reconstruct_k(l_factor: MatRef<'_, f64>) -> Mat<f64> {
    let l = copy_lower(l_factor);
    &l * l.transpose()
}

#[test]
fn two_by_two_matches_analytic_llt_and_solve() {
    // K = [[4, 2], [2, 3]], L = [[2, 0], [1, √2]], Kx = [2, 1] ⇒ x = [1/2, 0].
    let k = mat![[4.0, 2.0], [2.0, 3.0]];
    let l_true = mat![[2.0, 0.0], [1.0, 2.0_f64.sqrt()]];
    let x_true = mat![[0.5], [0.0]];
    let b = mat![[2.0], [1.0]];

    let n = 2;
    let mut memory = stack_for(n, 1);
    let stack = MemStack::new(&mut memory);

    let mut l = k.clone();
    let info = factor_in_place(l.as_mut(), stack).expect("SPD test matrix must factor");
    assert_eq!(info.dynamic_regularization_count, 0);
    assert_lower_close(l.as_ref(), l_true.as_ref());
    assert_mat_close(reconstruct_k(l.as_ref()).as_ref(), k.as_ref());

    let mut x = b.clone();
    llt::solve::solve_in_place(l.as_ref(), x.as_mut(), Par::Seq, stack);
    assert_mat_close(x.as_ref(), x_true.as_ref());
}

#[test]
fn five_by_five_matches_analytic_llt_and_solve() {
    let l_true = mat![
        [2.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 3.0, 0.0, 0.0, 0.0],
        [0.5, 1.0, 2.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 2.0, 0.0],
        [0.0, 0.5, 1.0, 1.0, 1.0],
    ];
    let k = &l_true * l_true.transpose();
    let x_true = mat![[1.0], [-1.0], [2.0], [0.5], [-2.0]];
    let b = &k * &x_true;

    let n = 5;
    let mut memory = stack_for(n, 1);
    let stack = MemStack::new(&mut memory);

    let mut l = k.clone();
    let info = factor_in_place(l.as_mut(), stack).expect("SPD test matrix must factor");
    assert_eq!(info.dynamic_regularization_count, 0);
    assert_lower_close(l.as_ref(), l_true.as_ref());
    assert_mat_close(reconstruct_k(l.as_ref()).as_ref(), k.as_ref());

    let mut x = b.clone();
    llt::solve::solve_in_place(l.as_ref(), x.as_mut(), Par::Seq, stack);
    assert_mat_close(x.as_ref(), x_true.as_ref());
}
