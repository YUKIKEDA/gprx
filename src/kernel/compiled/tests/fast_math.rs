//! `FastApprox` leaves against the polynomial `exp` and finite differences.

use super::*;

fn fast_exp(x: f64) -> f64 {
    <crate::math::FastApprox as crate::math::KernelMath>::exp(x)
}

fn apply_fast(spec: &KernelSpec, x: MatRef<'_, f64>, out: &mut Mat<f64>) {
    let compiled = spec.compile();
    let mut scratch = Mat::zeros(out.nrows(), out.ncols());
    compiled
        .apply_points::<crate::math::FastApprox>(x, out.as_mut(), Triangle::Full, scratch.as_mut())
        .expect("fast apply");
}

#[test]
fn fast_rbf_matches_polynomial_and_grad_fd() {
    let ell = 1.3;
    let spec = rbf(ell);
    let x = Mat::from_fn(3, 1, |i, _| [0.0, 0.7, 1.6][i]);
    let mut k = fill(3, 0.0);
    apply_fast(&spec, x.as_ref(), &mut k);
    let inv = 1.0 / (ell * ell);
    for col in 0..3 {
        for row in 0..3 {
            let d = x[(row, 0)] - x[(col, 0)];
            assert_close(k[(row, col)], fast_exp(-0.5 * d * d * inv), TOL);
        }
    }
    let h = 1e-6;
    let mut plus = spec.clone();
    let mut minus = spec.clone();
    let mut theta = vec![0.0; spec.num_params()];
    spec.get_params(&mut theta).expect("theta");
    plus.set_params(&[theta[0] + h]).expect("plus");
    minus.set_params(&[theta[0] - h]).expect("minus");
    let mut k_plus = fill(3, 0.0);
    let mut k_minus = fill(3, 0.0);
    apply_fast(&plus, x.as_ref(), &mut k_plus);
    apply_fast(&minus, x.as_ref(), &mut k_minus);
    let compiled = spec.compile();
    let mut dk = fill(3, 0.0);
    let mut scratch = fill(3, 0.0);
    compiled
        .grad_points::<crate::math::FastApprox>(
            x.as_ref(),
            dk.as_mut(),
            0,
            Triangle::Full,
            scratch.as_mut(),
        )
        .expect("grad");
    for col in 0..3 {
        for row in 0..3 {
            let fd = (k_plus[(row, col)] - k_minus[(row, col)]) / (2.0 * h);
            assert_close(dk[(row, col)], fd, TOL);
        }
    }
}

fn fd_fast_grad_and_hess(label: &str, spec: &KernelSpec, x: MatRef<'_, f64>) {
    let n = x.nrows();
    let p = spec.num_params();
    let mut theta = vec![0.0; p];
    spec.get_params(&mut theta).expect("theta");
    let compiled = spec.compile();
    let h = 1e-5;
    let tol = 1e-4;
    for i in 0..p {
        let mut plus = spec.clone();
        let mut minus = spec.clone();
        let mut tp = theta.clone();
        let mut tm = theta.clone();
        tp[i] += h;
        tm[i] -= h;
        plus.set_params(&tp).expect("plus");
        minus.set_params(&tm).expect("minus");
        let mut k_plus = fill(n, 0.0);
        let mut k_minus = fill(n, 0.0);
        apply_fast(&plus, x, &mut k_plus);
        apply_fast(&minus, x, &mut k_minus);
        let mut dk = fill(n, 0.0);
        let mut scratch = fill(n, 0.0);
        compiled
            .grad_points::<crate::math::FastApprox>(
                x,
                dk.as_mut(),
                i,
                Triangle::Full,
                scratch.as_mut(),
            )
            .expect("grad");
        for col in 0..n {
            for row in 0..n {
                let fd = (k_plus[(row, col)] - k_minus[(row, col)]) / (2.0 * h);
                let scale = fd.abs().max(1.0);
                assert!(
                    (dk[(row, col)] - fd).abs() <= tol * scale,
                    "{label} grad i={i} ({row},{col}) analytic={} fd={fd}",
                    dk[(row, col)]
                );
            }
        }
        for j in 0..=i {
            let bump = |di: f64, dj: f64| {
                let mut shifted = spec.clone();
                let mut t = theta.clone();
                t[i] += di;
                t[j] += dj;
                shifted.set_params(&t).expect("shift");
                let mut k = fill(n, 0.0);
                apply_fast(&shifted, x, &mut k);
                k
            };
            let k_pp = bump(h, h);
            let k_pm = bump(h, -h);
            let k_mp = bump(-h, h);
            let k_mm = bump(-h, -h);
            let mut d2 = fill(n, 0.0);
            let mut scratch = fill(n, 0.0);
            compiled
                .hess_points::<crate::math::FastApprox>(
                    x,
                    d2.as_mut(),
                    i,
                    j,
                    Triangle::Full,
                    scratch.as_mut(),
                )
                .expect("hess");
            for col in 0..n {
                for row in 0..n {
                    let fd = (k_pp[(row, col)] - k_pm[(row, col)] - k_mp[(row, col)]
                        + k_mm[(row, col)])
                        / (4.0 * h * h);
                    let scale = fd.abs().max(1.0);
                    assert!(
                        (d2[(row, col)] - fd).abs() <= tol * scale,
                        "{label} hess i={i} j={j} ({row},{col}) analytic={} fd={fd}",
                        d2[(row, col)]
                    );
                }
            }
        }
    }
}

#[test]
fn fast_exp_leaves_match_polynomial_derivatives() {
    let iso = Mat::from_fn(3, 1, |i, _| [0.0, 0.4, 1.1][i]);
    let ard = Mat::from_fn(3, 2, |row, col| {
        [[0.0, 0.2], [0.5, -0.3], [1.1, 0.7]][row][col]
    });
    for nu in [MaternNu::Half, MaternNu::ThreeHalves, MaternNu::FiveHalves] {
        fd_fast_grad_and_hess(
            &format!("matern {nu:?}"),
            &KernelSpec::from(MaternKernel::new(1.1, nu).expect("matern")),
            iso.as_ref(),
        );
        fd_fast_grad_and_hess(
            &format!("matern-ard {nu:?}"),
            &KernelSpec::from(MaternArdKernel::new(&[0.9, 1.4], nu).expect("matern ard")),
            ard.as_ref(),
        );
    }
    fd_fast_grad_and_hess(
        "periodic",
        &KernelSpec::from(PeriodicKernel::new(1.2, 0.7).expect("periodic")),
        iso.as_ref(),
    );
    fd_fast_grad_and_hess(
        "rbf-ard",
        &KernelSpec::from(RbfArdKernel::new(&[1.2, 0.8]).expect("rbf ard")),
        ard.as_ref(),
    );
    fd_fast_grad_and_hess("rbf", &rbf(1.3), iso.as_ref());
}

#[test]
fn fast_approx_leaves_non_exp_kernels_unchanged() {
    let x = Mat::from_fn(2, 1, |i, _| i as f64);
    for spec in [
        KernelSpec::from(RationalQuadraticKernel::new(1.0, 1.0).expect("rq")),
        KernelSpec::from(LinearKernel::new(1.1).expect("linear")),
        KernelSpec::from(ConstantKernel::new(0.4).expect("constant")),
        KernelSpec::from(WhiteKernel::new(0.2).expect("white")),
    ] {
        let mut accurate = fill(2, 0.0);
        let mut fast = fill(2, 0.0);
        let compiled = spec.compile();
        let mut scratch = fill(2, 0.0);
        compiled
            .apply_points::<crate::math::Accurate>(
                x.as_ref(),
                accurate.as_mut(),
                Triangle::Full,
                scratch.as_mut(),
            )
            .expect("accurate");
        apply_fast(&spec, x.as_ref(), &mut fast);
        for col in 0..2 {
            for row in 0..2 {
                assert_eq!(accurate[(row, col)].to_bits(), fast[(row, col)].to_bits());
            }
        }
    }
}
