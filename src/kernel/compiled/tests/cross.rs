//! `∂K(Z, X)/∂θ` and `∂²K(Z, X)/∂θ∂θ` of a rectangular block: each entry is
//! the same entry of the square derivative over the stacked points `[Z; X]`.

use super::*;
use crate::math::Accurate;

const M: usize = 3;
const N: usize = 4;

/// `M` points, then `N` points, in `d` dimensions; deterministic, no two equal.
fn stacked(d: usize) -> Mat<f64> {
    Mat::from_fn(M + N, d, |row, col| {
        0.3 + 0.41 * row as f64 - 0.17 * ((row * (col + 2)) % 5) as f64 + 0.08 * col as f64
    })
}

fn rows(x: MatRef<'_, f64>, start: usize, count: usize) -> Mat<f64> {
    Mat::from_fn(count, x.ncols(), |r, c| x[(start + r, c)])
}

fn constant(v: f64) -> KernelSpec {
    KernelSpec::from(ConstantKernel::new(v).expect("valid"))
}

fn ard(d: usize) -> KernelSpec {
    let ells: Vec<f64> = (0..d).map(|k| 0.9 + 0.4 * k as f64).collect();
    KernelSpec::from(RbfArdKernel::new(&ells).expect("valid"))
}

fn matern_ard(d: usize) -> KernelSpec {
    let ells: Vec<f64> = (0..d).map(|k| 0.8 + 0.3 * k as f64).collect();
    KernelSpec::from(MaternArdKernel::new(&ells, MaternNu::FiveHalves).expect("valid"))
}

fn rq_ard(d: usize) -> KernelSpec {
    let ells: Vec<f64> = (0..d).map(|k| 1.1 + 0.2 * k as f64).collect();
    KernelSpec::from(RationalQuadraticArdKernel::new(&ells, 0.7).expect("valid"))
}

fn matern(nu: MaternNu) -> KernelSpec {
    KernelSpec::from(MaternKernel::new(1.2, nu).expect("valid"))
}

fn rq() -> KernelSpec {
    KernelSpec::from(RationalQuadraticKernel::new(1.1, 0.8).expect("valid"))
}

fn periodic() -> KernelSpec {
    KernelSpec::from(PeriodicKernel::new(0.9, 1.7).expect("valid"))
}

fn linear() -> KernelSpec {
    KernelSpec::from(LinearKernel::new(0.6).expect("valid"))
}

fn white() -> KernelSpec {
    KernelSpec::from(WhiteKernel::new(0.05).expect("valid"))
}

/// Every built-in leaf, and the trees that need a product: with one factor per
/// leaf kind, with nested sums, and with three factors.
fn kernels(d: usize) -> Vec<(&'static str, KernelSpec)> {
    vec![
        ("rbf", rbf(1.3)),
        ("matern_1_2", matern(MaternNu::Half)),
        ("matern_3_2", matern(MaternNu::ThreeHalves)),
        ("matern_5_2", matern(MaternNu::FiveHalves)),
        ("rq", rq()),
        ("periodic", periodic()),
        ("constant", constant(1.7)),
        ("linear", linear()),
        ("rbf_ard", ard(d)),
        ("matern_ard", matern_ard(d)),
        ("rq_ard", rq_ard(d)),
        ("constant * rbf", constant(1.7) * rbf(1.3)),
        ("constant * ard", constant(1.7) * ard(d)),
        ("rbf * periodic", rbf(2.0) * periodic()),
        ("constant * rq", constant(0.6) * rq()),
        ("linear * rbf", linear() * rbf(1.4)),
        ("constant * matern_ard", constant(0.8) * matern_ard(d)),
        ("constant * rq_ard", constant(2.1) * rq_ard(d)),
        (
            "constant * rbf * periodic",
            constant(1.2) * rbf(1.9) * periodic(),
        ),
        (
            "constant * (rbf + matern)",
            constant(1.4) * (rbf(1.3) + matern(MaternNu::ThreeHalves)),
        ),
        (
            "rbf + constant * periodic",
            rbf(1.3) + constant(0.9) * periodic(),
        ),
        (
            "(rbf + rq) * (periodic + constant)",
            (rbf(1.3) + rq()) * (periodic() + constant(0.5)),
        ),
        ("constant * rbf + white", constant(1.5) * rbf(1.2) + white()),
    ]
}

fn check(actual: f64, expected: f64, what: &str, at: (usize, usize)) {
    assert!(
        (actual - expected).abs() <= 1e-12 * expected.abs().max(1.0),
        "{what} at {at:?}: cross {actual} vs square {expected}"
    );
}

fn scratch_for(rows: usize, cols: usize) -> Mat<f64> {
    Mat::zeros(rows, cols)
}

#[test]
fn rectangular_gradient_matches_the_block_of_the_square_gradient() {
    for d in [1, 3] {
        let all = stacked(d);
        let z = rows(all.as_ref(), 0, M);
        let x = rows(all.as_ref(), M, N);
        for (name, spec) in kernels(d) {
            let compiled = spec.compile();
            for p in 0..compiled.num_params() {
                let mut square = Mat::zeros(M + N, M + N);
                compiled
                    .grad_points::<Accurate>(
                        all.as_ref(),
                        square.as_mut(),
                        p,
                        Triangle::Full,
                        scratch_for(M + N, M + N).as_mut(),
                    )
                    .unwrap_or_else(|e| panic!("{name} d={d} p={p}: square: {e}"));
                let mut cross = Mat::zeros(M, N);
                compiled
                    .grad_cross_points::<Accurate>(
                        z.as_ref(),
                        x.as_ref(),
                        cross.as_mut(),
                        p,
                        scratch_for(M, N).as_mut(),
                    )
                    .unwrap_or_else(|e| panic!("{name} d={d} p={p}: cross: {e}"));
                for r in 0..M {
                    for c in 0..N {
                        check(
                            cross[(r, c)],
                            square[(r, M + c)],
                            &format!("grad {name} d={d} p={p}"),
                            (r, c),
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn rectangular_hessian_matches_the_block_of_the_square_hessian() {
    for d in [1, 3] {
        let all = stacked(d);
        let z = rows(all.as_ref(), 0, M);
        let x = rows(all.as_ref(), M, N);
        for (name, spec) in kernels(d) {
            let compiled = spec.compile();
            let p = compiled.num_params();
            for i in 0..p {
                for j in 0..p {
                    let mut square = Mat::zeros(M + N, M + N);
                    compiled
                        .hess_points::<Accurate>(
                            all.as_ref(),
                            square.as_mut(),
                            i,
                            j,
                            Triangle::Full,
                            scratch_for(M + N, M + N).as_mut(),
                        )
                        .unwrap_or_else(|e| panic!("{name} d={d} ({i},{j}): square: {e}"));
                    let mut cross = Mat::zeros(M, N);
                    compiled
                        .hess_cross_points::<Accurate>(
                            z.as_ref(),
                            x.as_ref(),
                            cross.as_mut(),
                            i,
                            j,
                            scratch_for(M, N).as_mut(),
                        )
                        .unwrap_or_else(|e| panic!("{name} d={d} ({i},{j}): cross: {e}"));
                    for r in 0..M {
                        for c in 0..N {
                            check(
                                cross[(r, c)],
                                square[(r, M + c)],
                                &format!("hess {name} d={d} ({i},{j})"),
                                (r, c),
                            );
                        }
                    }
                }
            }
        }
    }
}

/// The rectangular values from coordinates match the square ones, for the
/// isotropic leaves that had no such path.
#[test]
fn rectangular_values_from_coordinates_match_the_block_of_the_square_values() {
    for d in [1, 3] {
        let all = stacked(d);
        let z = rows(all.as_ref(), 0, M);
        let x = rows(all.as_ref(), M, N);
        for (name, spec) in kernels(d) {
            let compiled = spec.compile();
            let mut square = Mat::zeros(M + N, M + N);
            compiled
                .apply_points::<Accurate>(
                    all.as_ref(),
                    square.as_mut(),
                    Triangle::Full,
                    scratch_for(M + N, M + N).as_mut(),
                )
                .unwrap_or_else(|e| panic!("{name} d={d}: square: {e}"));
            let mut cross = Mat::zeros(M, N);
            compiled
                .apply_cross_points::<Accurate>(
                    z.as_ref(),
                    x.as_ref(),
                    cross.as_mut(),
                    scratch_for(M, N).as_mut(),
                )
                .unwrap_or_else(|e| panic!("{name} d={d}: cross: {e}"));
            for r in 0..M {
                for c in 0..N {
                    check(
                        cross[(r, c)],
                        square[(r, M + c)],
                        &format!("value {name} d={d}"),
                        (r, c),
                    );
                }
            }
        }
    }
}

/// A custom leaf has a rectangular value but no rectangular derivative.
#[test]
fn custom_leaf_has_no_rectangular_derivative() {
    let compiled = (constant(1.0) * custom_rbf(1.0)).compile();
    let all = stacked(1);
    let z = rows(all.as_ref(), 0, M);
    let x = rows(all.as_ref(), M, N);
    let mut out = Mat::zeros(M, N);
    let result = compiled.grad_cross_points::<Accurate>(
        z.as_ref(),
        x.as_ref(),
        out.as_mut(),
        1,
        scratch_for(M, N).as_mut(),
    );
    assert!(matches!(
        result,
        Err(crate::GprError::CoordGradientUnsupported)
    ));
}

/// The same block identity in `f32` (the sparse models' storage scalar).
#[test]
fn rectangular_gradient_in_f32_matches_the_square_block() {
    let d = 3;
    let all64 = stacked(d);
    let all = Mat::<f32>::from_fn(M + N, d, |r, c| all64[(r, c)] as f32);
    let z = Mat::<f32>::from_fn(M, d, |r, c| all[(r, c)]);
    let x = Mat::<f32>::from_fn(N, d, |r, c| all[(M + r, c)]);
    for (name, spec) in kernels(d) {
        let compiled = spec.compile_as::<f32>();
        let p = compiled.num_params();
        for i in 0..p {
            let mut square = Mat::<f32>::zeros(M + N, M + N);
            compiled
                .grad_points::<Accurate>(
                    all.as_ref(),
                    square.as_mut(),
                    i,
                    Triangle::Full,
                    Mat::<f32>::zeros(M + N, M + N).as_mut(),
                )
                .unwrap_or_else(|e| panic!("{name} p={i}: square: {e}"));
            let mut cross = Mat::<f32>::zeros(M, N);
            compiled
                .grad_cross_points::<Accurate>(
                    z.as_ref(),
                    x.as_ref(),
                    cross.as_mut(),
                    i,
                    Mat::<f32>::zeros(M, N).as_mut(),
                )
                .unwrap_or_else(|e| panic!("{name} p={i}: cross: {e}"));
            for r in 0..M {
                for c in 0..N {
                    let (a, b) = (cross[(r, c)] as f64, square[(r, M + c)] as f64);
                    assert!(
                        (a - b).abs() <= 1e-5 * b.abs().max(1.0),
                        "f32 grad {name} p={i} at ({r}, {c}): {a} vs {b}"
                    );
                }
            }
        }
    }
}
