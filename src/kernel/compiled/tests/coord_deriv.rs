//! Coordinate derivatives of a rectangular block, for `Sgpr<FreeInducing>`:
//! each is checked against a central difference of the value (or of the
//! derivative one order below), for every leaf and the trees that need the
//! product rule.

use super::cross::{M, N, kernels, rows, stacked};
use super::*;
use crate::GprError;
use crate::math::Accurate;

const H: f64 = 1e-5;

fn value(k: &CompiledKernel, x1: MatRef<'_, f64>, x2: MatRef<'_, f64>) -> Mat<f64> {
    let mut out = Mat::zeros(x1.nrows(), x2.nrows());
    let mut scratch = Mat::zeros(x1.nrows(), x2.nrows());
    k.apply_cross_points::<Accurate>(x1, x2, out.as_mut(), scratch.as_mut())
        .expect("value");
    out
}

fn d_x2(k: &CompiledKernel, x1: MatRef<'_, f64>, x2: MatRef<'_, f64>, dim: usize) -> Mat<f64> {
    let mut out = Mat::zeros(x1.nrows(), x2.nrows());
    k.grad_wrt_coord_dim::<Accurate>(x1, x2, out.as_mut(), dim)
        .expect("first derivative");
    out
}

fn shifted(x: MatRef<'_, f64>, dim: usize, by: f64) -> Mat<f64> {
    Mat::from_fn(x.nrows(), x.ncols(), |r, c| {
        x[(r, c)] + if c == dim { by } else { 0.0 }
    })
}

fn close(actual: &Mat<f64>, expected: &Mat<f64>, what: &str) {
    for r in 0..actual.nrows() {
        for c in 0..actual.ncols() {
            let (a, e) = (actual[(r, c)], expected[(r, c)]);
            assert!(
                (a - e).abs() <= 1e-4 * e.abs().max(1.0),
                "{what} at ({r},{c}): {a} vs finite difference {e}"
            );
        }
    }
}

fn central(plus: Mat<f64>, minus: Mat<f64>) -> Mat<f64> {
    Mat::from_fn(plus.nrows(), plus.ncols(), |r, c| {
        (plus[(r, c)] - minus[(r, c)]) / (2.0 * H)
    })
}

/// The point sets: distinct blocks `Z, X`, and `Z, Z` (coincident points).
fn pairs(d: usize) -> Vec<(Mat<f64>, Mat<f64>)> {
    let all = stacked(d);
    let z = rows(all.as_ref(), 0, M);
    let x = rows(all.as_ref(), M, N);
    vec![(z.clone(), x), (z.clone(), z)]
}

/// Matérn `ν = 1/2` has no coordinate derivative at coincident points.
fn skipped(name: &str) -> bool {
    name == "matern_1_2"
}

#[test]
fn first_derivative_matches_a_difference_of_the_value() {
    for d in [1, 3] {
        for (x1, x2) in pairs(d) {
            for (name, spec) in kernels(d) {
                if skipped(name) {
                    continue;
                }
                let k = spec.compile();
                for dim in 0..d {
                    let expected = central(
                        value(&k, x1.as_ref(), shifted(x2.as_ref(), dim, H).as_ref()),
                        value(&k, x1.as_ref(), shifted(x2.as_ref(), dim, -H).as_ref()),
                    );
                    close(
                        &d_x2(&k, x1.as_ref(), x2.as_ref(), dim),
                        &expected,
                        &format!("{name} d={d} dim={dim} ∂/∂x2"),
                    );
                }
            }
        }
    }
}

#[test]
fn second_derivatives_match_a_difference_of_the_first() {
    for d in [1, 3] {
        for (x1, x2) in pairs(d) {
            for (name, spec) in kernels(d) {
                if skipped(name) {
                    continue;
                }
                let k = spec.compile();
                for a in 0..d {
                    for b in 0..d {
                        let mut xx = Mat::zeros(x1.nrows(), x2.nrows());
                        let mut scratch = Mat::zeros(x1.nrows(), x2.nrows());
                        k.hess_wrt_coord_dims::<Accurate>(
                            x1.as_ref(),
                            x2.as_ref(),
                            xx.as_mut(),
                            a,
                            b,
                            scratch.as_mut(),
                        )
                        .unwrap_or_else(|e| panic!("{name} d={d} ({a},{b}): {e}"));
                        let expected = central(
                            d_x2(&k, x1.as_ref(), shifted(x2.as_ref(), b, H).as_ref(), a),
                            d_x2(&k, x1.as_ref(), shifted(x2.as_ref(), b, -H).as_ref(), a),
                        );
                        close(&xx, &expected, &format!("{name} d={d} ∂x2[{a}]∂x2[{b}]"));

                        let mut mixed = Mat::zeros(x1.nrows(), x2.nrows());
                        k.hess_wrt_coord_mixed::<Accurate>(
                            x1.as_ref(),
                            x2.as_ref(),
                            mixed.as_mut(),
                            a,
                            b,
                            scratch.as_mut(),
                        )
                        .unwrap_or_else(|e| panic!("{name} d={d} ({a},{b}): {e}"));
                        let expected = central(
                            d_x2(&k, shifted(x1.as_ref(), a, H).as_ref(), x2.as_ref(), b),
                            d_x2(&k, shifted(x1.as_ref(), a, -H).as_ref(), x2.as_ref(), b),
                        );
                        close(&mixed, &expected, &format!("{name} d={d} ∂x1[{a}]∂x2[{b}]"));
                    }
                }
            }
        }
    }
}

#[test]
fn parameter_and_coordinate_derivative_matches_a_difference_over_the_parameter() {
    for d in [1, 3] {
        for (x1, x2) in pairs(d) {
            for (name, spec) in kernels(d) {
                if skipped(name) {
                    continue;
                }
                let k = spec.compile();
                let n_params = k.num_params();
                let mut theta = vec![0.0; n_params];
                k.get_params(&mut theta).expect("params");
                for p in 0..n_params {
                    let at = |by: f64| {
                        let mut t = theta.clone();
                        t[p] += by;
                        let mut k = spec.compile();
                        k.set_params(&t).expect("set");
                        k
                    };
                    for dim in 0..d {
                        let mut got = Mat::zeros(x1.nrows(), x2.nrows());
                        k.hess_theta_coord_dim::<Accurate>(
                            x1.as_ref(),
                            x2.as_ref(),
                            got.as_mut(),
                            p,
                            dim,
                        )
                        .unwrap_or_else(|e| panic!("{name} d={d} p={p} dim={dim}: {e}"));
                        let expected = central(
                            d_x2(&at(H), x1.as_ref(), x2.as_ref(), dim),
                            d_x2(&at(-H), x1.as_ref(), x2.as_ref(), dim),
                        );
                        close(&got, &expected, &format!("{name} d={d} ∂θ{p}∂x2[{dim}]"));
                    }
                }
            }
        }
    }
}

#[test]
fn matern_half_is_the_one_unsupported_kernel() {
    let k = kernels(2)
        .into_iter()
        .find(|(name, _)| *name == "matern_1_2")
        .expect("listed")
        .1
        .compile();
    let x = stacked(2);
    let mut out = Mat::zeros(x.nrows(), x.nrows());
    assert!(matches!(
        k.grad_wrt_coord_dim::<Accurate>(x.as_ref(), x.as_ref(), out.as_mut(), 0),
        Err(GprError::CoordGradientUnsupported)
    ));
}

#[test]
fn custom_leaf_matches_the_built_in_rbf() {
    for d in [1, 3] {
        for (x1, x2) in pairs(d) {
            for tree in [
                (custom_rbf(1.3), rbf(1.3)),
                (custom_rbf(1.3) * rbf(0.9), rbf(1.3) * rbf(0.9)),
                (custom_rbf(1.3) + rbf(0.9), rbf(1.3) + rbf(0.9)),
            ] {
                let (custom, built_in) = (tree.0.compile(), tree.1.compile());
                for dim in 0..d {
                    close(
                        &d_x2(&custom, x1.as_ref(), x2.as_ref(), dim),
                        &d_x2(&built_in, x1.as_ref(), x2.as_ref(), dim),
                        "custom ∂/∂x2",
                    );
                    let mut a = Mat::zeros(x1.nrows(), x2.nrows());
                    let mut b = Mat::zeros(x1.nrows(), x2.nrows());
                    for p in 0..custom.num_params() {
                        custom
                            .hess_theta_coord_dim::<Accurate>(
                                x1.as_ref(),
                                x2.as_ref(),
                                a.as_mut(),
                                p,
                                dim,
                            )
                            .expect("custom");
                        built_in
                            .hess_theta_coord_dim::<Accurate>(
                                x1.as_ref(),
                                x2.as_ref(),
                                b.as_mut(),
                                p,
                                dim,
                            )
                            .expect("built-in");
                        close(&a, &b, "custom ∂θ∂x2");
                    }
                }
                for p in 0..custom.num_params() {
                    let mut a = Mat::zeros(x1.nrows(), x2.nrows());
                    let mut b = Mat::zeros(x1.nrows(), x2.nrows());
                    let mut sa = Mat::zeros(x1.nrows(), x2.nrows());
                    let mut sb = Mat::zeros(x1.nrows(), x2.nrows());
                    custom
                        .grad_cross_points::<Accurate>(
                            x1.as_ref(),
                            x2.as_ref(),
                            a.as_mut(),
                            p,
                            sa.as_mut(),
                        )
                        .expect("custom cross");
                    built_in
                        .grad_cross_points::<Accurate>(
                            x1.as_ref(),
                            x2.as_ref(),
                            b.as_mut(),
                            p,
                            sb.as_mut(),
                        )
                        .expect("built-in cross");
                    close(&a, &b, "custom ∂K/∂θ (rectangular)");
                    for q in 0..custom.num_params() {
                        custom
                            .hess_cross_points::<Accurate>(
                                x1.as_ref(),
                                x2.as_ref(),
                                a.as_mut(),
                                p,
                                q,
                                sa.as_mut(),
                            )
                            .expect("custom cross hess");
                        built_in
                            .hess_cross_points::<Accurate>(
                                x1.as_ref(),
                                x2.as_ref(),
                                b.as_mut(),
                                p,
                                q,
                                sb.as_mut(),
                            )
                            .expect("built-in cross hess");
                        close(&a, &b, "custom ∂²K/∂θ∂θ (rectangular)");
                    }
                }
            }
        }
    }
}
