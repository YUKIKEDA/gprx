//! Analytic Exact GPR checks for n = 2 and n = 3 isotropic RBF.
//!
//! P1A-11 pins mean, both variance kinds, negative LML, and ∂L/∂θ in one
//! sequence against a closed-form 2×2 / 3×3 inverse. This file does not
//! call faer or sklearn; P1A-12 adds committed JSON as a second check.

use gprx::kernel::{KernelSpec, RbfKernel};
use gprx::{FitOptions, GaussianLikelihood, Gpr, GprError, PredictOptions, VarianceKind};

const TOL: f64 = 1e-9;

fn assert_close(actual: f64, expected: f64) {
    let scale = expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= TOL * scale,
        "actual={actual}, expected={expected}"
    );
}

fn rbf(dist2: f64, ell: f64) -> f64 {
    (-0.5 * dist2 / (ell * ell)).exp()
}

fn dist2(x: &[f64], i: usize, j: usize) -> f64 {
    let d = x[i] - x[j];
    d * d
}

fn kernel_matrix(x: &[f64], ell: f64, noise: f64) -> Vec<f64> {
    let n = x.len();
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = rbf(dist2(x, i, j), ell);
        }
        a[i * n + i] += noise;
    }
    a
}

fn d_k_d_log_ell(x: &[f64], ell: f64) -> Vec<f64> {
    let n = x.len();
    let inv_ell_sq = 1.0 / (ell * ell);
    let mut d_k = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let d2 = dist2(x, i, j);
            d_k[i * n + j] = rbf(d2, ell) * d2 * inv_ell_sq;
        }
    }
    d_k
}

fn invert2(a: &[f64]) -> (Vec<f64>, f64) {
    let det = a[0] * a[3] - a[1] * a[2];
    let inv = vec![a[3] / det, -a[1] / det, -a[2] / det, a[0] / det];
    (inv, det)
}

fn invert3(a: &[f64]) -> (Vec<f64>, f64) {
    let c00 = a[4] * a[8] - a[5] * a[7];
    let c01 = a[5] * a[6] - a[3] * a[8];
    let c02 = a[3] * a[7] - a[4] * a[6];
    let c10 = a[2] * a[7] - a[1] * a[8];
    let c11 = a[0] * a[8] - a[2] * a[6];
    let c12 = a[1] * a[6] - a[0] * a[7];
    let c20 = a[1] * a[5] - a[2] * a[4];
    let c21 = a[2] * a[3] - a[0] * a[5];
    let c22 = a[0] * a[4] - a[1] * a[3];
    let det = a[0] * c00 + a[1] * c01 + a[2] * c02;
    let inv = vec![
        c00 / det,
        c10 / det,
        c20 / det,
        c01 / det,
        c11 / det,
        c21 / det,
        c02 / det,
        c12 / det,
        c22 / det,
    ];
    (inv, det)
}

fn matvec(a: &[f64], v: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            out[i] += a[i * n + j] * v[j];
        }
    }
    out
}

fn dot(u: &[f64], v: &[f64]) -> f64 {
    u.iter().zip(v).map(|(a, b)| a * b).sum()
}

fn frobenius(w: &[f64], d_a: &[f64]) -> f64 {
    w.iter().zip(d_a).map(|(a, b)| a * b).sum()
}

fn w_matrix(alpha: &[f64], a_inv: &[f64], n: usize) -> Vec<f64> {
    let mut w = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            w[i * n + j] = alpha[i] * alpha[j] - a_inv[i * n + j];
        }
    }
    w
}

fn nlml(y: &[f64], alpha: &[f64], det: f64) -> f64 {
    let n = y.len() as f64;
    0.5 * (dot(y, alpha) + det.ln() + n * (2.0 * std::f64::consts::PI).ln())
}

fn k_star(x: &[f64], xs: &[f64], ell: f64) -> Vec<f64> {
    let n = x.len();
    let m = xs.len();
    let mut k = vec![0.0; n * m];
    for col in 0..m {
        for row in 0..n {
            let d = x[row] - xs[col];
            k[row * m + col] = rbf(d * d, ell);
        }
    }
    k
}

fn predict_oracle(
    x: &[f64],
    xs: &[f64],
    ell: f64,
    noise: f64,
    alpha: &[f64],
    a_inv: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = x.len();
    let m = xs.len();
    let ks = k_star(x, xs, ell);
    let mut mean = vec![0.0; m];
    let mut latent = vec![0.0; m];
    for col in 0..m {
        let mut kcol = vec![0.0; n];
        for row in 0..n {
            kcol[row] = ks[row * m + col];
        }
        mean[col] = dot(&kcol, alpha);
        let ainv_k = matvec(a_inv, &kcol, n);
        let mut var = 1.0 - dot(&kcol, &ainv_k);
        if var < 0.0 {
            var = 0.0;
        }
        latent[col] = var;
    }
    let obs: Vec<f64> = latent.iter().map(|v| v + noise).collect();
    (mean, latent, obs)
}

fn check_rbf_case(ell: f64, noise: f64, x: &[f64], y: &[f64], xs: &[f64]) -> Result<(), GprError> {
    let n = x.len();
    assert_eq!(y.len(), n);
    assert!(n == 2 || n == 3);
    let a = kernel_matrix(x, ell, noise);
    let (a_inv, det) = if n == 2 { invert2(&a) } else { invert3(&a) };
    let restored = matvec(&a, &matvec(&a_inv, y, n), n);
    for i in 0..n {
        assert_close(restored[i], y[i]);
    }
    let alpha = matvec(&a_inv, y, n);
    let expected_nlml = nlml(y, &alpha, det);
    let w = w_matrix(&alpha, &a_inv, n);
    let d_ell = d_k_d_log_ell(x, ell);
    let mut d_noise = vec![0.0; n * n];
    for i in 0..n {
        d_noise[i * n + i] = noise;
    }
    let expected_grad = [-0.5 * frobenius(&w, &d_ell), -0.5 * frobenius(&w, &d_noise)];
    let (mean, latent, obs) = predict_oracle(x, xs, ell, noise, &alpha, &a_inv);

    let mut gpr = Gpr::new(
        KernelSpec::from(RbfKernel::new(ell)?),
        GaussianLikelihood::new(noise)?,
    )
    .fit_with(x, n, 1, y, FitOptions::FIXED)?;
    assert_eq!(gpr.n(), n);
    assert_eq!(gpr.d(), 1);

    let pred_lat = gpr.predict_with(
        xs,
        xs.len(),
        1,
        PredictOptions {
            variance_kind: VarianceKind::Latent,
        },
    )?;
    let pred_obs = gpr.predict(xs, xs.len(), 1)?;
    assert_eq!(pred_obs.variance_kind, VarianceKind::Observation);
    for i in 0..xs.len() {
        assert_close(pred_lat.mean[i], mean[i]);
        assert_close(pred_obs.mean[i], mean[i]);
        assert_close(pred_lat.variance[i], latent[i]);
        assert_close(pred_obs.variance[i], obs[i]);
        assert_close(pred_obs.variance[i], pred_lat.variance[i] + noise);
    }

    assert_close(gpr.neg_log_marginal_likelihood()?, expected_nlml);

    let mut params = [0.0; 2];
    gpr.get_params(&mut params)?;
    assert_close(params[0], ell.ln());
    assert_close(params[1], noise.ln());
    let mut grad = [0.0; 2];
    let value = gpr.value_and_gradient_into(&params, &mut grad)?;
    assert_close(value, expected_nlml);
    assert_close(grad[0], expected_grad[0]);
    assert_close(grad[1], expected_grad[1]);
    Ok(())
}

#[test]
fn rbf_n_two_and_three_pin_mean_variance_lml_and_grad() {
    check_rbf_case(1.0, 0.1, &[0.0, 1.0], &[0.5, -0.25], &[0.0, 0.5, 2.0]).expect("n = 2 RBF");
    check_rbf_case(1.25, 0.16, &[0.0, 0.8, 1.7], &[0.4, -0.2, 0.9], &[0.8, 1.2])
        .expect("n = 3 RBF");
}
