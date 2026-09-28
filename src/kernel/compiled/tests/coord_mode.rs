//! Distance, coordinate, and mixed evaluation paths.

use super::*;

#[test]
fn ard_apply_dist_is_unsupported_points_match_isotropic() {
    let ell = 1.3;
    let compiled = KernelSpec::from(RbfArdKernel::new(&[ell, ell]).expect("valid")).compile();
    let x = points_2d(&[[0.0, 0.0], [1.0, 0.4], [0.2, 1.1]]);
    let dist = {
        let n = x.nrows();
        Mat::from_fn(n, n, |row, col| {
            let mut sum = 0.0;
            for dim in 0..2 {
                let diff = x[(row, dim)] - x[(col, dim)];
                sum += diff * diff;
            }
            sum
        })
    };
    let mut out = fill(3, 0.0);
    let mut scratch = fill(3, 0.0);
    assert!(matches!(
        compiled.apply::<crate::math::Accurate>(
            dist.as_ref(),
            out.as_mut(),
            Triangle::Full,
            scratch.as_mut()
        ),
        Err(crate::error::GprError::UnsupportedKernelOperation { .. })
    ));
    compiled
        .apply_points::<crate::math::Accurate>(
            x.as_ref(),
            out.as_mut(),
            Triangle::Full,
            scratch.as_mut(),
        )
        .expect("points");
    let iso = apply_rbf(ell, dist.as_ref());
    for col in 0..3 {
        for row in 0..3 {
            assert_close(out[(row, col)], iso[(row, col)], TOL);
        }
    }
}

#[test]
fn mixed_isotropic_and_ard_is_mixed() {
    let spec = rbf(1.0) + KernelSpec::from(RbfArdKernel::new(&[1.0, 2.0]).expect("valid"));
    let compiled = spec.compile();
    assert_eq!(
        compiled.coord_mode().expect("mix"),
        crate::kernel::compiled::CoordMode::Mixed
    );
}

#[test]
fn rbf_plus_linear_apply_adds_leaves() {
    let spec = rbf(1.0) + KernelSpec::from(LinearKernel::new(1.0).expect("valid"));
    let compiled = spec.compile();
    assert_eq!(
        compiled.coord_mode().expect("mix"),
        crate::kernel::compiled::CoordMode::Mixed
    );
    let xs = [0.5, 1.5];
    let x = Mat::from_fn(2, 1, |i, _| xs[i]);
    let dist = sq_dist_1d(&xs);
    let mut out = fill(2, 0.0);
    let mut scratch = fill(2, 0.0);
    compiled
        .apply_mixed::<crate::math::Accurate>(
            MixedKernelViews::new(dist.as_ref(), x.as_ref()),
            out.as_mut(),
            Triangle::Full,
            scratch.as_mut(),
        )
        .expect("mix");
    let kr = apply_compiled(&rbf(1.0).compile(), dist.as_ref());
    let mut kl = fill(2, 0.0);
    LinearKernel::new(1.0)
        .expect("valid")
        .apply(x.as_ref(), kl.as_mut(), Triangle::Full)
        .expect("linear");
    for col in 0..2 {
        for row in 0..2 {
            assert_close(out[(row, col)], kr[(row, col)] + kl[(row, col)], TOL);
        }
    }
}

#[test]
fn rbf_times_linear_apply_multiplies_leaves() {
    let spec = rbf(1.0) * KernelSpec::from(LinearKernel::new(1.0).expect("valid"));
    let compiled = spec.compile();
    assert_eq!(
        compiled.coord_mode().expect("mix"),
        crate::kernel::compiled::CoordMode::Mixed
    );
    let xs = [0.5, 1.5];
    let x = Mat::from_fn(2, 1, |i, _| xs[i]);
    let dist = sq_dist_1d(&xs);
    let mut out = fill(2, 0.0);
    let mut scratch = fill(2, 0.0);
    compiled
        .apply_mixed::<crate::math::Accurate>(
            MixedKernelViews::new(dist.as_ref(), x.as_ref()),
            out.as_mut(),
            Triangle::Full,
            scratch.as_mut(),
        )
        .expect("mix");
    let kr = apply_compiled(&rbf(1.0).compile(), dist.as_ref());
    let mut kl = fill(2, 0.0);
    LinearKernel::new(1.0)
        .expect("valid")
        .apply(x.as_ref(), kl.as_mut(), Triangle::Full)
        .expect("linear");
    for col in 0..2 {
        for row in 0..2 {
            assert_close(out[(row, col)], kr[(row, col)] * kl[(row, col)], TOL);
        }
    }
}

#[test]
fn rbf_plus_linear_grad_matches_finite_difference() {
    let spec = rbf(1.0) + KernelSpec::from(LinearKernel::new(0.8).expect("valid"));
    let compiled = spec.compile();
    let xs = [0.5, 1.5];
    let x = Mat::from_fn(2, 1, |i, _| xs[i]);
    let dist = sq_dist_1d(&xs);
    let mut params = [0.0; 2];
    spec.get_params(&mut params).expect("len 2");
    let h = 1e-6;
    let mut spec_plus = spec.clone();
    let mut spec_minus = spec.clone();
    params[1] += h;
    spec_plus.set_params(&params).expect("valid");
    params[1] -= 2.0 * h;
    spec_minus.set_params(&params).expect("valid");
    let mut kp = fill(2, 0.0);
    let mut km = fill(2, 0.0);
    let mut scratch = fill(2, 0.0);
    spec_plus
        .compile()
        .apply_mixed::<crate::math::Accurate>(
            MixedKernelViews::new(dist.as_ref(), x.as_ref()),
            kp.as_mut(),
            Triangle::Full,
            scratch.as_mut(),
        )
        .expect("plus");
    spec_minus
        .compile()
        .apply_mixed::<crate::math::Accurate>(
            MixedKernelViews::new(dist.as_ref(), x.as_ref()),
            km.as_mut(),
            Triangle::Full,
            scratch.as_mut(),
        )
        .expect("minus");
    let mut dk = fill(2, 0.0);
    compiled
        .grad_mixed::<crate::math::Accurate>(
            MixedKernelViews::new(dist.as_ref(), x.as_ref()),
            dk.as_mut(),
            1,
            Triangle::Full,
            scratch.as_mut(),
        )
        .expect("grad");
    let fd = (kp[(0, 1)] - km[(0, 1)]) / (2.0 * h);
    assert_close(dk[(0, 1)], fd, TOL);
}

#[test]
fn rbf_plus_white_is_distance_mode() {
    let spec = rbf(1.0) + KernelSpec::from(WhiteKernel::new(0.1).expect("valid"));
    let compiled = spec.compile();
    assert_eq!(
        compiled.coord_mode().expect("compat"),
        crate::kernel::compiled::CoordMode::Dist
    );
    let dist = sq_dist_1d(&[0.0, 1.0]);
    let mut out = fill(2, 0.0);
    let mut scratch = fill(2, 0.0);
    compiled
        .apply::<crate::math::Accurate>(
            dist.as_ref(),
            out.as_mut(),
            Triangle::Full,
            scratch.as_mut(),
        )
        .expect("shape");
    assert_close(out[(0, 0)], 1.0 + 0.1, TOL);
    assert_close(out[(0, 1)], apply_rbf(1.0, dist.as_ref())[(0, 1)], TOL);
}

#[test]
fn linear_plus_constant_is_points_mode() {
    let spec = KernelSpec::from(LinearKernel::new(1.0).expect("valid"))
        + KernelSpec::from(ConstantKernel::new(0.5).expect("valid"));
    let compiled = spec.compile();
    assert_eq!(
        compiled.coord_mode().expect("compat"),
        crate::kernel::compiled::CoordMode::Points
    );
    let x = Mat::from_fn(2, 1, |i, _| i as f64);
    let mut out = fill(2, 0.0);
    let mut scratch = fill(2, 0.0);
    compiled
        .apply_points::<crate::math::Accurate>(
            x.as_ref(),
            out.as_mut(),
            Triangle::Full,
            scratch.as_mut(),
        )
        .expect("points");
    assert_close(out[(0, 0)], 0.5, TOL);
    assert_close(out[(1, 1)], 1.0 + 0.5, TOL);
    assert_close(out[(1, 0)], 0.5, TOL);
}

#[test]
fn matern_plus_white_is_distance_mode() {
    let spec = KernelSpec::from(MaternKernel::new(1.0, MaternNu::ThreeHalves).expect("valid"))
        + KernelSpec::from(WhiteKernel::new(0.1).expect("valid"));
    let compiled = spec.compile();
    assert_eq!(
        compiled.coord_mode().expect("compat"),
        crate::kernel::compiled::CoordMode::Dist
    );
    let dist = sq_dist_1d(&[0.0, 1.0]);
    let mut out = fill(2, 0.0);
    let mut scratch = fill(2, 0.0);
    compiled
        .apply::<crate::math::Accurate>(
            dist.as_ref(),
            out.as_mut(),
            Triangle::Full,
            scratch.as_mut(),
        )
        .expect("shape");
    let rho = 3.0_f64.sqrt();
    let k01 = (1.0 + rho) * (-rho).exp();
    assert_close(out[(0, 0)], 1.0 + 0.1, TOL);
    assert_close(out[(0, 1)], k01, TOL);
}

#[test]
fn matern_ard_plus_constant_is_points_mode() {
    let spec = KernelSpec::from(MaternArdKernel::new(&[1.0], MaternNu::Half).expect("valid"))
        + KernelSpec::from(ConstantKernel::new(0.5).expect("valid"));
    let compiled = spec.compile();
    assert_eq!(
        compiled.coord_mode().expect("compat"),
        crate::kernel::compiled::CoordMode::Points
    );
    let x = Mat::from_fn(2, 1, |i, _| i as f64);
    let mut out = fill(2, 0.0);
    let mut scratch = fill(2, 0.0);
    compiled
        .apply_points::<crate::math::Accurate>(
            x.as_ref(),
            out.as_mut(),
            Triangle::Full,
            scratch.as_mut(),
        )
        .expect("points");
    assert_close(out[(0, 0)], 1.5, TOL);
    assert_close(out[(1, 1)], 1.5, TOL);
    assert_close(out[(1, 0)], (-1.0_f64).exp() + 0.5, TOL);
}

#[test]
fn periodic_plus_white_is_distance_mode() {
    let spec = KernelSpec::from(PeriodicKernel::new(1.0, 2.0).expect("valid"))
        + KernelSpec::from(WhiteKernel::new(0.1).expect("valid"));
    let compiled = spec.compile();
    assert_eq!(
        compiled.coord_mode().expect("compat"),
        crate::kernel::compiled::CoordMode::Dist
    );
    let dist = sq_dist_1d(&[0.0, 1.0]);
    let mut out = fill(2, 0.0);
    let mut scratch = fill(2, 0.0);
    compiled
        .apply::<crate::math::Accurate>(
            dist.as_ref(),
            out.as_mut(),
            Triangle::Full,
            scratch.as_mut(),
        )
        .expect("shape");
    let s = (std::f64::consts::PI * 0.5).sin();
    let k01 = (-2.0 * s * s).exp();
    assert_close(out[(0, 0)], 1.0 + 0.1, TOL);
    assert_close(out[(0, 1)], k01, TOL);
}

#[test]
fn rational_quadratic_plus_white_is_distance_mode() {
    let spec = KernelSpec::from(RationalQuadraticKernel::new(1.0, 1.0).expect("valid"))
        + KernelSpec::from(WhiteKernel::new(0.1).expect("valid"));
    let compiled = spec.compile();
    assert_eq!(
        compiled.coord_mode().expect("compat"),
        crate::kernel::compiled::CoordMode::Dist
    );
    let dist = sq_dist_1d(&[0.0, 1.0]);
    let mut out = fill(2, 0.0);
    let mut scratch = fill(2, 0.0);
    compiled
        .apply::<crate::math::Accurate>(
            dist.as_ref(),
            out.as_mut(),
            Triangle::Full,
            scratch.as_mut(),
        )
        .expect("shape");
    assert_close(out[(0, 0)], 1.1, TOL);
    assert_close(out[(0, 1)], 2.0 / 3.0, TOL);
}

#[test]
fn rational_quadratic_ard_plus_constant_is_points_mode() {
    let spec = KernelSpec::from(RationalQuadraticArdKernel::new(&[1.0], 1.0).expect("valid"))
        + KernelSpec::from(ConstantKernel::new(0.5).expect("valid"));
    let compiled = spec.compile();
    assert_eq!(
        compiled.coord_mode().expect("compat"),
        crate::kernel::compiled::CoordMode::Points
    );
    let x = Mat::from_fn(2, 1, |i, _| i as f64);
    let mut out = fill(2, 0.0);
    let mut scratch = fill(2, 0.0);
    compiled
        .apply_points::<crate::math::Accurate>(
            x.as_ref(),
            out.as_mut(),
            Triangle::Full,
            scratch.as_mut(),
        )
        .expect("points");
    assert_close(out[(0, 0)], 1.5, TOL);
    assert_close(out[(1, 1)], 1.5, TOL);
    assert_close(out[(1, 0)], 2.0 / 3.0 + 0.5, TOL);
}
