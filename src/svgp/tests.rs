use super::*;
use crate::error::GprError;
use crate::kernel::{KernelSpec, MaternKernel, MaternNu, RbfArdKernel, RbfKernel, WhiteKernel};
use crate::likelihood::GaussianLikelihood;
use crate::linalg::{cholesky_lower_with_retries, faer_par, faer_par_dims};
use crate::policy::JitterPolicy;
use crate::{Adam, Fixed, PredictOptions, Sgpr, VarianceKind};
use dyn_stack::MemBuffer;
use faer::linalg::cholesky::llt;
use faer::{Mat, MatRef};

const TOL: f64 = 1e-12;

use crate::test_check::assert_close;

const X_1D: [f64; 4] = [0.0, 1.0, 2.0, 3.0];
const Y: [f64; 4] = [0.0, 1.0, 0.5, 0.25];
const Z_1D: [f64; 2] = [0.5, 2.5];
const X_ARD: [f64; 8] = [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0];
const Z_ARD: [f64; 4] = [0.25, 0.75, 0.25, 0.75];

fn kernel_rbf() -> KernelSpec {
    KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"))
}

fn kernel_matern() -> KernelSpec {
    KernelSpec::from(MaternKernel::new(1.0, MaternNu::ThreeHalves).expect("ℓ"))
}

fn kernel_ard() -> KernelSpec {
    KernelSpec::from(RbfArdKernel::new(&[1.0, 1.5]).expect("ℓ"))
}

fn kernel_rbf_white() -> KernelSpec {
    KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"))
        + KernelSpec::from(WhiteKernel::new(0.05).expect("white"))
}

struct Case {
    kernel: KernelSpec,
    x: &'static [f64],
    n: usize,
    d: usize,
    z: &'static [f64],
}

fn cases() -> [Case; 4] {
    [
        Case {
            kernel: kernel_rbf(),
            x: &X_1D,
            n: 4,
            d: 1,
            z: &Z_1D,
        },
        Case {
            kernel: kernel_matern(),
            x: &X_1D,
            n: 4,
            d: 1,
            z: &Z_1D,
        },
        Case {
            kernel: kernel_ard(),
            x: &X_ARD,
            n: 4,
            d: 2,
            z: &Z_ARD,
        },
        Case {
            kernel: kernel_rbf_white(),
            x: &X_1D,
            n: 4,
            d: 1,
            z: &Z_1D,
        },
    ]
}

fn factor_svgp(kernel: KernelSpec, x: &[f64], n: usize, d: usize, z: &[f64]) -> FittedSvgp {
    Svgp::new(kernel, GaussianLikelihood::new(0.1).expect("noise"))
        .factor(x, n, d, &Y, z, 2)
        .map_err(|(_, e)| e)
        .expect("svgp factor")
}

fn factor_vfe(
    kernel: KernelSpec,
    x: &[f64],
    n: usize,
    d: usize,
    z: &[f64],
) -> crate::FittedSgpr<Fixed> {
    Sgpr::new(kernel, GaussianLikelihood::new(0.1).expect("noise"))
        .with_optimizer(Fixed)
        .factor(x, n, d, &Y, z, 2)
        .map_err(|(_, e)| e)
        .expect("vfe factor")
}

fn cholesky_lower(mat: &mut Mat<f64>) {
    let n = mat.nrows();
    let req = llt::factor::cholesky_in_place_scratch::<f64>(n, faer_par(n), Default::default());
    let mut scratch = MemBuffer::new(req);
    cholesky_lower_with_retries(
        mat,
        &mut scratch,
        JitterPolicy::default().retry_jitters(),
        crate::error::CholeskyStage::Fit,
    )
    .expect("chol");
}

fn titsias_whitened_q(vfe: &crate::FittedSgpr<Fixed>) -> (Vec<f64>, Mat<f64>) {
    let m = vfe.m();
    let noise = vfe.likelihood().noise_variance();
    let (w, b_l) = vfe.vfe_w_and_b_l();
    let mean = w.to_vec();
    let mut inv = Mat::zeros(m, m);
    for i in 0..m {
        inv[(i, i)] = 1.0;
    }
    faer::linalg::triangular_solve::solve_lower_triangular_in_place(
        b_l,
        inv.as_mut(),
        faer_par_dims(m, m),
    );
    let mut s = Mat::zeros(m, m);
    for j in 0..m {
        for i in j..m {
            let mut sum = 0.0;
            for k in 0..m {
                sum += inv[(k, i)] * inv[(k, j)];
            }
            s[(i, j)] = noise * sum;
        }
    }
    cholesky_lower(&mut s);
    (mean, s)
}

fn write_q_params(fitted: &FittedSvgp, mean: &[f64], l: MatRef<'_, f64>) -> Vec<f64> {
    let mut params = vec![0.0; fitted.num_params()];
    fitted.get_params(&mut params).expect("get");
    let n_theta = fitted.kernel().num_params() + fitted.likelihood().num_params();
    super::factor::pack_q(mean, l, &mut params[n_theta..]);
    params
}

fn independent_neg_elbo(
    a: MatRef<'_, f64>,
    q_mean: &[f64],
    q_l: MatRef<'_, f64>,
    y: &[f64],
    k_diag: &[f64],
    noise: f64,
) -> f64 {
    let n = y.len();
    let m = q_mean.len();
    let mut tr_s = 0.0;
    let mut log_det_s = 0.0;
    for i in 0..m {
        for j in 0..m {
            tr_s += q_l[(i, j)] * q_l[(i, j)];
        }
        log_det_s += q_l[(i, i)].ln();
    }
    log_det_s *= 2.0;
    let mean_norm2: f64 = q_mean.iter().map(|v| v * v).sum();
    let kl = 0.5 * (tr_s + mean_norm2 - m as f64 - log_det_s);
    let c = 0.5 * (2.0 * std::f64::consts::PI * noise).ln();
    let inv_noise = 1.0 / noise;
    let mut ell = 0.0;
    for i in 0..n {
        let mut mu = 0.0;
        let mut a_norm = 0.0;
        for r in 0..m {
            mu += a[(r, i)] * q_mean[r];
            a_norm += a[(r, i)] * a[(r, i)];
        }
        // ‖Lᵀ a‖² via the product (Lᵀ a)_j = Σ_i L_{ij} a_i
        let mut lt_norm = 0.0;
        for j in 0..m {
            let mut acc = 0.0;
            for i_row in 0..m {
                acc += q_l[(i_row, j)] * a[(i_row, i)];
            }
            lt_norm += acc * acc;
        }
        let var = k_diag[i] - a_norm + lt_norm;
        let resid = y[i] - mu;
        ell += -c - 0.5 * inv_noise * (resid * resid + var);
    }
    -(ell - kl)
}

#[test]
fn factor_installs_whitened_prior() {
    let fitted = factor_svgp(kernel_rbf(), &X_1D, 4, 1, &Z_1D);
    let mut params = vec![0.0; fitted.num_params()];
    fitted.get_params(&mut params).expect("get");
    let n_theta = fitted.kernel().num_params() + fitted.likelihood().num_params();
    assert_eq!(n_theta, 2);
    assert_close(params[n_theta], 0.0, TOL);
    assert_close(params[n_theta + 1], 0.0, TOL);
    // packed L = I: L00, L10, L11
    assert_close(params[n_theta + 2], 1.0, TOL);
    assert_close(params[n_theta + 3], 0.0, TOL);
    assert_close(params[n_theta + 4], 1.0, TOL);
}

#[test]
fn set_params_rejects_non_positive_l_diag() {
    let mut fitted = factor_svgp(kernel_rbf(), &X_1D, 4, 1, &Z_1D);
    let mut params = vec![0.0; fitted.num_params()];
    fitted.get_params(&mut params).expect("get");
    let n_theta = fitted.kernel().num_params() + fitted.likelihood().num_params();
    params[n_theta + 2] = 0.0;
    let err = fitted.set_params(&params).expect_err("diag");
    assert!(matches!(err, GprError::InvalidHyperparameter { .. }));
}

#[test]
fn titsias_q_matches_vfe_nlml_and_predict() {
    for case in cases() {
        let vfe = factor_vfe(case.kernel.clone(), case.x, case.n, case.d, case.z);
        let mut svgp = factor_svgp(case.kernel.clone(), case.x, case.n, case.d, case.z);
        let (mean, l) = titsias_whitened_q(&vfe);
        let params = write_q_params(&svgp, &mean, l.as_ref());
        svgp.set_params(&params).expect("set q");
        assert_close(
            svgp.neg_elbo().expect("elbo"),
            vfe.neg_log_marginal_likelihood().expect("nlml"),
            TOL,
        );
        let pred_s = svgp.predict(case.x, case.n, case.d).expect("svgp pred");
        let pred_v = vfe.predict(case.x, case.n, case.d).expect("vfe pred");
        for i in 0..case.n {
            assert_close(pred_s.mean[i], pred_v.mean[i], TOL);
            assert_close(pred_s.variance[i], pred_v.variance[i], TOL);
        }
        let latent_s = svgp
            .predict_with(
                case.x,
                case.n,
                case.d,
                PredictOptions {
                    variance_kind: VarianceKind::Latent,
                },
            )
            .expect("svgp latent");
        let latent_v = vfe
            .predict_with(
                case.x,
                case.n,
                case.d,
                PredictOptions {
                    variance_kind: VarianceKind::Latent,
                },
            )
            .expect("vfe latent");
        for i in 0..case.n {
            assert_close(latent_s.variance[i], latent_v.variance[i], TOL);
        }
    }
}

#[test]
fn shifted_q_matches_independent_elbo() {
    for case in cases() {
        let vfe = factor_vfe(case.kernel.clone(), case.x, case.n, case.d, case.z);
        let mut svgp = factor_svgp(case.kernel.clone(), case.x, case.n, case.d, case.z);
        let (mut mean, mut l) = titsias_whitened_q(&vfe);
        mean[0] += 0.15;
        l[(0, 0)] *= 1.1;
        l[(1, 0)] += 0.05;
        let params = write_q_params(&svgp, &mean, l.as_ref());
        svgp.set_params(&params).expect("set q");
        let independent = independent_neg_elbo(
            svgp.a.as_ref(),
            &svgp.q_mean,
            svgp.q_l.as_ref(),
            svgp.y(),
            &svgp.k_diag,
            svgp.likelihood().noise_variance(),
        );
        assert_close(svgp.neg_elbo().expect("elbo"), independent, TOL);
    }
}

const GRAD_FD: f64 = 1e-5;
const GRAD_TOL: f64 = 1e-8;

fn fd_grad(model: &mut FittedSvgp, params: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; params.len()];
    let mut plus = params.to_vec();
    let mut minus = params.to_vec();
    for i in 0..params.len() {
        plus.copy_from_slice(params);
        minus.copy_from_slice(params);
        plus[i] += GRAD_FD;
        minus[i] -= GRAD_FD;
        model.set_params(&plus).expect("plus");
        let fp = model.neg_elbo().expect("fp");
        model.set_params(&minus).expect("minus");
        let fm = model.neg_elbo().expect("fm");
        out[i] = (fp - fm) / (2.0 * GRAD_FD);
    }
    model.set_params(params).expect("restore");
    out
}

fn assert_grad_close(analytic: &[f64], fd: &[f64]) {
    for (i, (a, e)) in analytic.iter().zip(fd).enumerate() {
        let scale = e.abs().max(1.0);
        assert!(
            (a - e).abs() <= GRAD_TOL * scale,
            "i={i} analytic={a} fd={e}"
        );
    }
}

fn check_full_grad(kernel: KernelSpec, x: &[f64], n: usize, d: usize, z: &[f64]) {
    let mut fitted = factor_svgp(kernel, x, n, d, z);
    let mut params = vec![0.0; fitted.num_params()];
    fitted.get_params(&mut params).expect("get");
    let mut analytic = vec![0.0; params.len()];
    let value = fitted
        .value_and_gradient_into(&params, &mut analytic)
        .expect("grad");
    assert_close(value, fitted.neg_elbo().expect("elbo"), TOL);
    let fd = fd_grad(&mut fitted, &params);
    assert_grad_close(&analytic, &fd);
}

#[test]
fn rbf_prior_grad_matches_fd() {
    check_full_grad(kernel_rbf(), &X_1D, 4, 1, &Z_1D);
}

#[test]
fn matern_prior_grad_matches_fd() {
    check_full_grad(kernel_matern(), &X_1D, 4, 1, &Z_1D);
}

#[test]
fn rbf_ard_prior_grad_matches_fd() {
    check_full_grad(kernel_ard(), &X_ARD, 4, 2, &Z_ARD);
}

#[test]
fn rbf_white_prior_grad_matches_fd() {
    check_full_grad(kernel_rbf_white(), &X_1D, 4, 1, &Z_1D);
}

#[test]
fn shifted_q_grad_matches_fd() {
    for case in cases() {
        let vfe = factor_vfe(case.kernel.clone(), case.x, case.n, case.d, case.z);
        let mut svgp = factor_svgp(case.kernel.clone(), case.x, case.n, case.d, case.z);
        let (mut mean, mut l) = titsias_whitened_q(&vfe);
        mean[0] += 0.15;
        l[(0, 0)] *= 1.1;
        l[(1, 0)] += 0.05;
        let params = write_q_params(&svgp, &mean, l.as_ref());
        svgp.set_params(&params).expect("set q");
        let mut analytic = vec![0.0; params.len()];
        svgp.value_and_gradient_into(&params, &mut analytic)
            .expect("grad");
        let fd = fd_grad(&mut svgp, &params);
        assert_grad_close(&analytic, &fd);
    }
}

#[test]
fn fast_approx_adam_elbo_does_not_rise() {
    let kernel = KernelSpec::from(RbfKernel::new(1.0).expect("ell"));
    let start = Svgp::new(kernel.clone(), GaussianLikelihood::new(0.1).expect("noise"))
        .with_math(crate::KernelExp::FastApprox)
        .factor(&X_1D, 4, 1, &Y, &Z_1D, 2)
        .map_err(|(_, err)| err)
        .expect("factor")
        .neg_elbo()
        .expect("start");
    let end = Svgp::new(kernel, GaussianLikelihood::new(0.1).expect("noise"))
        .with_math(crate::KernelExp::FastApprox)
        .with_optimizer(Adam::new())
        .fit(&X_1D, 4, 1, &Y, &Z_1D, 2)
        .map_err(|(_, err)| err)
        .expect("adam")
        .neg_elbo()
        .expect("end");
    assert!(end.is_finite(), "elbo={end}");
    assert!(end <= start, "end={end} start={start}");
}

fn fit_svgp(
    kernel: KernelSpec,
    x: &[f64],
    n: usize,
    d: usize,
    z: &[f64],
    adam: Adam,
) -> FittedSvgp {
    Svgp::new(kernel, GaussianLikelihood::new(0.1).expect("noise"))
        .with_optimizer(adam)
        .fit(x, n, d, &Y, z, 2)
        .map_err(|(_, e)| e)
        .expect("svgp fit")
}

fn check_fit_seed_and_elbo(kernel: KernelSpec, x: &[f64], n: usize, d: usize, z: &[f64]) {
    let start = factor_svgp(kernel.clone(), x, n, d, z)
        .neg_elbo()
        .expect("start");
    let a = fit_svgp(kernel.clone(), x, n, d, z, Adam::new());
    let b = fit_svgp(kernel, x, n, d, z, Adam::new());
    let mut pa = vec![0.0; a.num_params()];
    let mut pb = vec![0.0; b.num_params()];
    a.get_params(&mut pa).expect("a");
    b.get_params(&mut pb).expect("b");
    for (x, y) in pa.iter().zip(&pb) {
        assert_close(*x, *y, TOL);
    }
    let end = a.neg_elbo().expect("end");
    assert!(
        end <= start + GRAD_TOL * start.abs().max(1.0),
        "end={end} start={start}"
    );
}

#[test]
fn rbf_fit_reproduces_and_does_not_worsen() {
    check_fit_seed_and_elbo(kernel_rbf(), &X_1D, 4, 1, &Z_1D);
}

#[test]
fn matern_fit_reproduces_and_does_not_worsen() {
    check_fit_seed_and_elbo(kernel_matern(), &X_1D, 4, 1, &Z_1D);
}

#[test]
fn rbf_ard_fit_reproduces_and_does_not_worsen() {
    check_fit_seed_and_elbo(kernel_ard(), &X_ARD, 4, 2, &Z_ARD);
}

#[test]
fn rbf_white_fit_reproduces_and_does_not_worsen() {
    check_fit_seed_and_elbo(kernel_rbf_white(), &X_1D, 4, 1, &Z_1D);
}

#[test]
fn mini_batch_fit_reproduces() {
    use std::num::NonZeroUsize;
    let adam = Adam::new().with_batch_size(NonZeroUsize::MIN);
    let a = fit_svgp(kernel_rbf(), &X_1D, 4, 1, &Z_1D, adam.clone());
    let b = fit_svgp(kernel_rbf(), &X_1D, 4, 1, &Z_1D, adam);
    let mut pa = vec![0.0; a.num_params()];
    let mut pb = vec![0.0; b.num_params()];
    a.get_params(&mut pa).expect("a");
    b.get_params(&mut pb).expect("b");
    for (x, y) in pa.iter().zip(&pb) {
        assert_close(*x, *y, TOL);
    }
}

#[test]
fn kernel_exp_is_a_runtime_value() {
    let kernel = KernelSpec::from(RbfKernel::new(1.0).expect("ell"));
    let trainer = Svgp::new(kernel, GaussianLikelihood::new(0.1).expect("noise"))
        .with_math(crate::KernelExp::FastApprox);
    assert_eq!(trainer.math(), crate::KernelExp::FastApprox);
    let fitted = trainer
        .factor(&X_1D, 4, 1, &Y, &Z_1D, 2)
        .map_err(|(_, err)| err)
        .expect("factor");
    assert_eq!(fitted.math(), crate::KernelExp::FastApprox);
}

#[test]
fn k_mm_jitter_policy_applies_to_factor() {
    let x = [0.0, 1.0, 2.0, 3.0];
    let y = [0.0, 1.0, 0.5, 0.25];
    let z = [0.5, 0.5, 2.0];
    let trainer = Svgp::new(kernel_rbf(), GaussianLikelihood::new(0.1).expect("noise"));
    assert_eq!(
        trainer.jitter_policy(),
        JitterPolicy::adaptive(1e-8, 10.0, 5, 1e-3).expect("valid")
    );
    let result = trainer
        .clone()
        .with_jitter_policy(JitterPolicy::default())
        .factor(&x, 4, 1, &y, &z, 3)
        .map_err(|(_, e)| e);
    assert!(matches!(result, Err(GprError::CholeskyFailed { .. })));
    let policy = JitterPolicy::fixed(1e-4).expect("valid");
    let fitted = trainer
        .with_jitter_policy(policy)
        .factor(&x, 4, 1, &y, &z, 3)
        .map_err(|(_, e)| e)
        .expect("factor");
    assert_eq!(fitted.jitter_policy(), policy);
    let l = fitted.k_mm_l.as_ref();
    for i in 0..3 {
        for j in 0..=i {
            let mut llt = 0.0;
            for k in 0..=j {
                llt += l[(i, k)] * l[(j, k)];
            }
            let diff: f64 = z[i] - z[j];
            let mut expected = (-diff * diff / 2.0).exp();
            if i == j {
                expected += 1e-4;
            }
            assert_close(llt, expected, 1e-12);
        }
    }
}

#[test]
fn titsias_q_covariance_matches_vfe() {
    for case in cases() {
        let vfe = factor_vfe(case.kernel.clone(), case.x, case.n, case.d, case.z);
        let mut svgp = factor_svgp(case.kernel.clone(), case.x, case.n, case.d, case.z);
        let (mean, l) = titsias_whitened_q(&vfe);
        let params = write_q_params(&svgp, &mean, l.as_ref());
        svgp.set_params(&params).expect("set q");
        for kind in [VarianceKind::Latent, VarianceKind::Observation] {
            let options = PredictOptions {
                variance_kind: kind,
            };
            let cov_s = svgp
                .predict_covariance_with(case.x, case.n, case.d, options)
                .expect("svgp covariance");
            let cov_v = vfe
                .predict_covariance_with(case.x, case.n, case.d, options)
                .expect("vfe covariance");
            for (s, v) in cov_s.covariance.iter().zip(cov_v.covariance.iter()) {
                assert_close(*s, *v, TOL);
            }
            for (s, v) in cov_s.mean.iter().zip(cov_v.mean.iter()) {
                assert_close(*s, *v, TOL);
            }
        }
    }
}
