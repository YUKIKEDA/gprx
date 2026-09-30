//! User-defined leaves inside compiled trees.

use super::*;

#[test]
fn custom_plus_rbf_apply_matches_two_rbf() {
    let compiled = (custom_rbf(1.0) + rbf(2.0)).compile();
    let builtin = (rbf(1.0) + rbf(2.0)).compile();
    let dist = sq_dist_1d(&[0.0, 1.0, 2.0]);
    let out = apply_compiled(&compiled, dist.as_ref());
    let expected = apply_compiled(&builtin, dist.as_ref());
    for col in 0..3 {
        for row in 0..3 {
            assert_close(out[(row, col)], expected[(row, col)], TOL);
        }
    }
}

#[test]
fn custom_times_rbf_apply_matches_two_rbf() {
    let compiled = (custom_rbf(1.0) * rbf(0.5)).compile();
    let builtin = (rbf(1.0) * rbf(0.5)).compile();
    let dist = sq_dist_1d(&[0.0, 1.2]);
    let out = apply_compiled(&compiled, dist.as_ref());
    let expected = apply_compiled(&builtin, dist.as_ref());
    assert_close(out[(0, 1)], expected[(0, 1)], TOL);
    assert_close(out[(0, 0)], expected[(0, 0)], TOL);
}

#[test]
fn custom_sum_grad_matches_finite_difference() {
    let spec = custom_rbf(1.0) + rbf(2.0);
    let compiled = spec.compile();
    let dist = sq_dist_1d(&[0.0, 0.8, 1.5]);
    let mut params = [0.0; 2];
    spec.get_params(&mut params).expect("len 2");
    let h = 1e-6;
    let mut spec_plus = spec.clone();
    let mut spec_minus = spec.clone();
    params[1] += h;
    spec_plus.set_params(&params).expect("valid");
    params[1] -= 2.0 * h;
    spec_minus.set_params(&params).expect("valid");
    let kp = apply_compiled(&spec_plus.compile(), dist.as_ref());
    let km = apply_compiled(&spec_minus.compile(), dist.as_ref());
    let mut dk = fill(3, 0.0);
    let mut scratch = fill(3, 0.0);
    compiled
        .grad::<crate::math::Accurate>(
            dist.as_ref(),
            dk.as_mut(),
            1,
            Triangle::Full,
            scratch.as_mut(),
        )
        .expect("idx 1");
    for col in 0..3 {
        for row in 0..3 {
            let fd = (kp[(row, col)] - km[(row, col)]) / (2.0 * h);
            assert!(
                (dk[(row, col)] - fd).abs() <= 1e-4 * fd.abs().max(1.0),
                "row={row} col={col} analytic={} fd={fd}",
                dk[(row, col)]
            );
        }
    }
}

#[test]
fn custom_plus_linear_is_mixed() {
    let spec = custom_rbf(1.0) + KernelSpec::from(LinearKernel::new(1.0).expect("valid"));
    let compiled = spec.compile();
    assert_eq!(
        compiled.coord_mode().expect("mix"),
        crate::kernel::compiled::CoordMode::Mixed
    );
}
