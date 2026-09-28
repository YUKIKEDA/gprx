//! Sum and product trees: values, gradients, and Hessians.

use super::*;

#[test]
fn combine_sum_from_leaf_grams_overwrites_dirty_dest() {
    let compiled = (rbf(1.0) + rbf(2.0)).compile();
    let dist = sq_dist_1d(&[0.0, 1.0, 2.0]);
    let expected = apply_compiled(&compiled, dist.as_ref());
    let mut grams = [fill(3, 0.0), fill(3, 0.0)];
    compiled
        .leaf_at(0)
        .expect("leaf 0")
        .apply::<crate::math::Accurate>(
            dist.as_ref(),
            grams[0].as_mut(),
            Triangle::Lower,
            fill(3, 0.0).as_mut(),
        )
        .expect("leaf 0");
    compiled
        .leaf_at(1)
        .expect("leaf 1")
        .apply::<crate::math::Accurate>(
            dist.as_ref(),
            grams[1].as_mut(),
            Triangle::Lower,
            fill(3, 0.0).as_mut(),
        )
        .expect("leaf 1");
    let mut dirty = fill(3, 999.0);
    let mut scratch = fill(3, 0.0);
    compiled
        .combine_from_leaf_grams(&grams, dirty.as_mut(), scratch.as_mut(), Triangle::Lower)
        .expect("combine");
    assert_lower_close(dirty.as_ref(), expected.as_ref(), TOL);
}

#[test]
fn sum_apply_adds_leaves() {
    let compiled = (rbf(1.0) + rbf(2.0)).compile();
    let dist = sq_dist_1d(&[0.0, 1.0, 2.0]);
    let out = apply_compiled(&compiled, dist.as_ref());
    let k1 = apply_rbf(1.0, dist.as_ref());
    let k2 = apply_rbf(2.0, dist.as_ref());
    for col in 0..3 {
        for row in 0..3 {
            assert_close(out[(row, col)], k1[(row, col)] + k2[(row, col)], TOL);
        }
    }
}

#[test]
fn product_apply_multiplies_leaves() {
    let compiled = (rbf(1.0) * rbf(0.5)).compile();
    let dist = sq_dist_1d(&[0.0, 1.2]);
    let out = apply_compiled(&compiled, dist.as_ref());
    let k1 = apply_rbf(1.0, dist.as_ref());
    let k2 = apply_rbf(0.5, dist.as_ref());
    assert_close(out[(0, 1)], k1[(0, 1)] * k2[(0, 1)], TOL);
    assert_close(out[(0, 0)], 1.0, TOL);
}

#[test]
fn mixed_product_of_sum_matches_leaves() {
    let spec = rbf(1.0) * (rbf(2.0) + rbf(3.0));
    let compiled = spec.compile();
    let dist = sq_dist_1d(&[0.0, 0.7, 1.4]);
    let out = apply_compiled(&compiled, dist.as_ref());
    let k1 = apply_rbf(1.0, dist.as_ref());
    let k2 = apply_rbf(2.0, dist.as_ref());
    let k3 = apply_rbf(3.0, dist.as_ref());
    for col in 0..3 {
        for row in 0..3 {
            assert_close(
                out[(row, col)],
                k1[(row, col)] * (k2[(row, col)] + k3[(row, col)]),
                TOL,
            );
        }
    }
}

#[test]
fn product_of_two_sums_matches_leaves() {
    let spec = (rbf(1.0) + rbf(2.0)) * (rbf(0.5) + rbf(1.5));
    let compiled = spec.compile();
    let dist = sq_dist_1d(&[0.0, 1.0]);
    let out = apply_compiled(&compiled, dist.as_ref());
    let a = apply_rbf(1.0, dist.as_ref());
    let b = apply_rbf(2.0, dist.as_ref());
    let c = apply_rbf(0.5, dist.as_ref());
    let d = apply_rbf(1.5, dist.as_ref());
    for col in 0..2 {
        for row in 0..2 {
            assert_close(
                out[(row, col)],
                (a[(row, col)] + b[(row, col)]) * (c[(row, col)] + d[(row, col)]),
                TOL,
            );
        }
    }
}

#[test]
fn sum_lower_matches_full_and_leaves_upper() {
    let compiled = (rbf(1.0) + rbf(2.0)).compile();
    let dist = sq_dist_1d(&[0.0, 1.0, 2.0]);
    let full = apply_compiled(&compiled, dist.as_ref());
    let sentinel = 42.0;
    let mut lower = fill(3, sentinel);
    let mut scratch = fill(3, 0.0);
    compiled
        .apply::<crate::math::Accurate>(
            dist.as_ref(),
            lower.as_mut(),
            Triangle::Lower,
            scratch.as_mut(),
        )
        .expect("shape");
    assert_lower_close(lower.as_ref(), full.as_ref(), TOL);
    assert_close(lower[(0, 1)], sentinel, TOL);
    assert_close(lower[(0, 2)], sentinel, TOL);
    assert_close(lower[(1, 2)], sentinel, TOL);
}

#[test]
fn sum_grad_matches_finite_difference() {
    let spec = rbf(1.0) + rbf(2.0);
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
            assert_close(dk[(row, col)], fd, TOL);
        }
    }
}

#[test]
fn product_hess_matches_finite_difference_of_grad() {
    let spec = rbf(1.25) * KernelSpec::from(ConstantKernel::new(1.4).expect("valid"));
    let compiled = spec.compile();
    let dist = mat![[0.0, 0.64, 2.89], [0.64, 0.0, 0.81], [2.89, 0.81, 0.0]];
    let mut params = [0.0; 2];
    spec.get_params(&mut params).expect("len 2");
    let h = 1e-6;
    for j in 0..2 {
        let mut plus = params;
        let mut minus = params;
        plus[j] += h;
        minus[j] -= h;
        let mut spec_plus = spec.clone();
        let mut spec_minus = spec.clone();
        spec_plus.set_params(&plus).expect("valid");
        spec_minus.set_params(&minus).expect("valid");
        let mut gp = fill(3, 0.0);
        let mut gm = fill(3, 0.0);
        let mut scratch = fill(3, 0.0);
        spec_plus
            .compile()
            .grad::<crate::math::Accurate>(
                dist.as_ref(),
                gp.as_mut(),
                0,
                Triangle::Full,
                scratch.as_mut(),
            )
            .expect("plus");
        spec_minus
            .compile()
            .grad::<crate::math::Accurate>(
                dist.as_ref(),
                gm.as_mut(),
                0,
                Triangle::Full,
                scratch.as_mut(),
            )
            .expect("minus");
        let mut d2 = fill(3, 0.0);
        compiled
            .hess::<crate::math::Accurate>(
                dist.as_ref(),
                d2.as_mut(),
                0,
                j,
                Triangle::Full,
                scratch.as_mut(),
            )
            .expect("hess");
        let fd = (gp[(0, 1)] - gm[(0, 1)]) / (2.0 * h);
        assert_close(d2[(0, 1)], fd, TOL);
    }
}

#[test]
fn product_grad_matches_finite_difference() {
    let spec = rbf(1.0) * rbf(2.0);
    let compiled = spec.compile();
    let dist = mat![[0.0, 1.0], [1.0, 0.0]];
    let mut params = [0.0; 2];
    spec.get_params(&mut params).expect("len 2");
    let h = 1e-6;
    let mut spec_plus = spec.clone();
    let mut spec_minus = spec.clone();
    params[0] += h;
    spec_plus.set_params(&params).expect("valid");
    params[0] -= 2.0 * h;
    spec_minus.set_params(&params).expect("valid");
    let kp = apply_compiled(&spec_plus.compile(), dist.as_ref());
    let km = apply_compiled(&spec_minus.compile(), dist.as_ref());
    let mut dk = fill(2, 0.0);
    let mut scratch = fill(2, 0.0);
    compiled
        .grad::<crate::math::Accurate>(
            dist.as_ref(),
            dk.as_mut(),
            0,
            Triangle::Full,
            scratch.as_mut(),
        )
        .expect("idx 0");
    let fd = (kp[(0, 1)] - km[(0, 1)]) / (2.0 * h);
    assert_close(dk[(0, 1)], fd, TOL);
}

#[test]
fn nested_product_grad_matches_finite_difference() {
    let spec = rbf(1.0) * (rbf(2.0) + rbf(3.0));
    let compiled = spec.compile();
    let dist = sq_dist_1d(&[0.0, 0.9]);
    let mut params = [0.0; 3];
    spec.get_params(&mut params).expect("len 3");
    let h = 1e-6;
    let mut spec_plus = spec.clone();
    let mut spec_minus = spec.clone();
    params[2] += h;
    spec_plus.set_params(&params).expect("valid");
    params[2] -= 2.0 * h;
    spec_minus.set_params(&params).expect("valid");
    let kp = apply_compiled(&spec_plus.compile(), dist.as_ref());
    let km = apply_compiled(&spec_minus.compile(), dist.as_ref());
    let mut dk = fill(2, 0.0);
    let mut scratch = fill(2, 0.0);
    compiled
        .grad::<crate::math::Accurate>(
            dist.as_ref(),
            dk.as_mut(),
            2,
            Triangle::Full,
            scratch.as_mut(),
        )
        .expect("idx 2");
    let fd = (kp[(0, 1)] - km[(0, 1)]) / (2.0 * h);
    assert_close(dk[(0, 1)], fd, TOL);
}

fn assert_points_product_grad_fd(spec: KernelSpec, x: Mat<f64>, param_idx: usize) {
    let compiled = spec.compile();
    let mut params = vec![0.0; spec.num_params()];
    spec.get_params(&mut params).expect("len");
    let h = 1e-6;
    let mut spec_plus = spec.clone();
    let mut spec_minus = spec.clone();
    params[param_idx] += h;
    spec_plus.set_params(&params).expect("valid");
    params[param_idx] -= 2.0 * h;
    spec_minus.set_params(&params).expect("valid");
    let kp = apply_compiled_points(&spec_plus.compile(), x.as_ref());
    let km = apply_compiled_points(&spec_minus.compile(), x.as_ref());
    let mut dk = fill(x.nrows(), 0.0);
    let mut scratch = fill(x.nrows(), 0.0);
    compiled
        .grad_points::<crate::math::Accurate>(
            x.as_ref(),
            dk.as_mut(),
            param_idx,
            Triangle::Full,
            scratch.as_mut(),
        )
        .expect("grad");
    let fd = (kp[(0, 1)] - km[(0, 1)]) / (2.0 * h);
    assert_close(dk[(0, 1)], fd, TOL);
}

#[test]
fn linear_times_linear_grad_matches_finite_difference() {
    let spec = KernelSpec::from(LinearKernel::new(1.0).expect("valid"))
        * KernelSpec::from(LinearKernel::new(0.5).expect("valid"));
    let x = Mat::from_fn(2, 1, |i, _| 0.5 + i as f64);
    assert_points_product_grad_fd(spec, x, 0);
}

#[test]
fn linear_times_constant_grad_matches_finite_difference() {
    let spec = KernelSpec::from(LinearKernel::new(1.0).expect("valid"))
        * KernelSpec::from(ConstantKernel::new(1.5).expect("valid"));
    let x = Mat::from_fn(2, 1, |i, _| 0.5 + i as f64);
    assert_points_product_grad_fd(spec, x, 1);
}

#[test]
fn linear_times_ard_grad_matches_finite_difference() {
    let spec = KernelSpec::from(LinearKernel::new(1.0).expect("valid"))
        * KernelSpec::from(RbfArdKernel::new(&[1.2, 0.8]).expect("valid"));
    let x = points_2d(&[[0.0, 0.0], [1.0, 0.4], [0.2, 1.1]]);
    assert_points_product_grad_fd(spec, x, 2);
}

#[test]
fn nested_points_product_grad_matches_finite_difference() {
    let spec = KernelSpec::from(LinearKernel::new(1.0).expect("valid"))
        * (KernelSpec::from(LinearKernel::new(0.8).expect("valid"))
            + KernelSpec::from(ConstantKernel::new(0.5).expect("valid")));
    let x = Mat::from_fn(2, 1, |i, _| 0.5 + i as f64);
    assert_points_product_grad_fd(spec, x, 2);
}

#[test]
fn empty_sum_is_unsupported() {
    let compiled = CompiledKernel::<f64>::Sum(Vec::new());
    let dist = sq_dist_1d(&[0.0, 1.0]);
    let mut out = fill(2, 0.0);
    let mut scratch = fill(2, 0.0);
    assert!(matches!(
        compiled.apply::<crate::math::Accurate>(
            dist.as_ref(),
            out.as_mut(),
            Triangle::Full,
            scratch.as_mut()
        ),
        Err(crate::error::GprError::UnsupportedKernelOperation { .. })
    ));
}

#[test]
fn fill_diag_adds_rbf_leaves() {
    let compiled = (rbf(1.0) + rbf(2.0)).compile();
    let mut diag = [0.0, 0.0];
    compiled.fill_diag(&mut diag).expect("two terms");
    assert_close(diag[0], 2.0, TOL);
    assert_close(diag[1], 2.0, TOL);
}

#[test]
fn apply_cross_matches_full_block() {
    let compiled = rbf(1.0).compile();
    let train = sq_dist_1d(&[0.0, 1.0]);
    let mut k_nn = fill(2, 0.0);
    let mut scratch = fill(2, 0.0);
    compiled
        .apply::<crate::math::Accurate>(
            train.as_ref(),
            k_nn.as_mut(),
            Triangle::Full,
            scratch.as_mut(),
        )
        .expect("square");
    let dist_cross = faer::mat![[0.0, 1.0], [1.0, 0.0]];
    let mut k_cross = fill(2, 0.0);
    let mut scratch_cross = fill(2, 0.0);
    compiled
        .apply_cross::<crate::math::Accurate>(
            dist_cross.as_ref(),
            k_cross.as_mut(),
            scratch_cross.as_mut(),
        )
        .expect("rect");
    assert_close(k_cross[(0, 0)], k_nn[(0, 0)], TOL);
    assert_close(k_cross[(0, 1)], k_nn[(0, 1)], TOL);
}
