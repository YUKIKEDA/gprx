//! Sparse predictive covariance and posterior samples (R5-6 / #284).
//!
//! The covariance is checked against the dense formulas written out here,
//! against Exact at `Z = X`, against `predict` on the diagonal, and the
//! samples against the covariance. This file uses only the public API.

use gprx::kernel::{KernelSpec, RbfKernel};
use gprx::transform::{StandardizeInput, StandardizeTarget};
use gprx::{
    Fixed, GaussianLikelihood, Gpr, PredictOptions, PredictiveCovariance, Sgpr, Svgp, VarianceKind,
};

mod common;
use common::{assert_close, assert_close_named};

const N: usize = 10;
const M: usize = 4;
const Q: usize = 5;
const D: usize = 2;
const ELL: f64 = 1.2;
const NOISE: f64 = 0.05;
const TOL: f64 = 1e-10;
/// VFE at `Z = X` against Exact, and a rank-1 update against a refactor.
const REFACTOR_TOL: f64 = 1e-9;

type Dense = Vec<Vec<f64>>;

fn points(rows: usize, shift: f64) -> Vec<f64> {
    let mut x = vec![0.0; rows * D];
    for i in 0..rows {
        let t = i as f64 + shift;
        x[i] = 0.5 * t;
        x[rows + i] = (0.9 * t).cos();
    }
    x
}

fn targets(x: &[f64], rows: usize) -> Vec<f64> {
    (0..rows)
        .map(|i| (1.3 * x[i]).sin() + 0.4 * x[rows + i])
        .collect()
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn kernel() -> KernelSpec {
    KernelSpec::from(RbfKernel::new(ELL).expect("valid"))
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn likelihood() -> GaussianLikelihood {
    GaussianLikelihood::new(NOISE).expect("valid")
}

fn options(kind: VarianceKind) -> PredictOptions {
    PredictOptions {
        variance_kind: kind,
    }
}

/// `k(a_i, b_j)` of the RBF kernel for column-major point sets.
fn cross(a: &[f64], na: usize, b: &[f64], nb: usize) -> Dense {
    (0..na)
        .map(|i| {
            (0..nb)
                .map(|j| {
                    let mut sq = 0.0;
                    for dim in 0..D {
                        let diff = a[dim * na + i] - b[dim * nb + j];
                        sq += diff * diff;
                    }
                    (-sq / (2.0 * ELL * ELL)).exp()
                })
                .collect()
        })
        .collect()
}

fn transpose(a: &Dense) -> Dense {
    (0..a[0].len())
        .map(|j| a.iter().map(|row| row[j]).collect())
        .collect()
}

fn matmul(a: &Dense, b: &Dense) -> Dense {
    a.iter()
        .map(|row| {
            (0..b[0].len())
                .map(|j| row.iter().zip(b).map(|(x, brow)| x * brow[j]).sum())
                .collect()
        })
        .collect()
}

#[allow(clippy::needless_range_loop)] // index form mirrors the matrix formula
fn cholesky(a: &Dense) -> Dense {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];
    for j in 0..n {
        let mut diag = a[j][j];
        for k in 0..j {
            diag -= l[j][k] * l[j][k];
        }
        l[j][j] = diag.sqrt();
        for i in j + 1..n {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            l[i][j] = sum / l[j][j];
        }
    }
    l
}

/// `L⁻¹ B` for lower `L`.
#[allow(clippy::needless_range_loop)] // index form mirrors the matrix formula
fn solve_lower(l: &Dense, b: &Dense) -> Dense {
    let n = l.len();
    let mut x = b.clone();
    for col in 0..b[0].len() {
        for i in 0..n {
            let mut sum = x[i][col];
            for k in 0..i {
                sum -= l[i][k] * x[k][col];
            }
            x[i][col] = sum / l[i][i];
        }
    }
    x
}

/// `(L⁻¹ K_m*)ᵀ (L⁻¹ K_m*) = K*m A⁻¹ Km*` for `A = L Lᵀ`.
fn quad(a: &Dense, k_ms: &Dense) -> Dense {
    let v = solve_lower(&cholesky(a), k_ms);
    matmul(&transpose(&v), &v)
}

#[allow(clippy::needless_range_loop)] // index form mirrors the matrix formula
fn assert_covariance(label: &str, actual: &PredictiveCovariance, expected: &Dense, tol: f64) {
    for col in 0..Q {
        for row in 0..Q {
            assert_close_named(
                &format!("{label}[{row}, {col}]"),
                actual.covariance[col * Q + row],
                expected[row][col],
                tol,
            );
        }
    }
}

fn with_noise(mut latent: Dense, kind: VarianceKind) -> Dense {
    if kind == VarianceKind::Observation {
        for (i, row) in latent.iter_mut().enumerate() {
            row[i] += NOISE;
        }
    }
    latent
}

#[test]
fn sgpr_covariance_matches_dense_vfe_formula() {
    let x = points(N, 0.0);
    let y = targets(&x, N);
    let z = points(M, 0.35);
    let xs = points(Q, 0.6);
    let fitted = Sgpr::new(kernel(), likelihood())
        .with_optimizer(Fixed)
        .factor(&x, N, D, &y, &z, M)
        .map_err(|(_, e)| e)
        .expect("factor");
    let k_mm = cross(&z, M, &z, M);
    let k_mn = cross(&z, M, &x, N);
    let k_ms = cross(&z, M, &xs, Q);
    let k_ss = cross(&xs, Q, &xs, Q);
    let mut a = matmul(&k_mn, &transpose(&k_mn));
    for (i, row) in a.iter_mut().enumerate() {
        for (j, value) in row.iter_mut().enumerate() {
            *value = k_mm[i][j] + *value / NOISE;
        }
    }
    let q_ss = quad(&k_mm, &k_ms);
    let sigma = quad(&a, &k_ms);
    let latent: Dense = (0..Q)
        .map(|i| {
            (0..Q)
                .map(|j| k_ss[i][j] - q_ss[i][j] + sigma[i][j])
                .collect()
        })
        .collect();
    for kind in [VarianceKind::Latent, VarianceKind::Observation] {
        let cov = fitted
            .predict_covariance_with(&xs, Q, D, options(kind))
            .expect("covariance");
        assert_covariance("vfe", &cov, &with_noise(latent.clone(), kind), TOL);
    }
}

#[test]
fn svgp_covariance_matches_dense_whitened_formula() {
    let x = points(N, 0.0);
    let y = targets(&x, N);
    let z = points(M, 0.35);
    let xs = points(Q, 0.6);
    let mut fitted = Svgp::new(kernel(), likelihood())
        .factor(&x, N, D, &y, &z, M)
        .map_err(|(_, e)| e)
        .expect("factor");
    let mut params = vec![0.0; fitted.num_params()];
    fitted.get_params(&mut params).expect("params");
    let n_theta = fitted.num_params() - M - M * (M + 1) / 2;
    let q_mean: Vec<f64> = (0..M).map(|i| 0.3 - 0.2 * i as f64).collect();
    let mut q_l = vec![vec![0.0; M]; M];
    let mut packed = n_theta + M;
    params[n_theta..n_theta + M].copy_from_slice(&q_mean);
    for j in 0..M {
        for (i, row) in q_l.iter_mut().enumerate().skip(j) {
            let value = if i == j {
                0.6 + 0.1 * j as f64
            } else {
                0.05 * (i + j) as f64
            };
            row[j] = value;
            params[packed] = value;
            packed += 1;
        }
    }
    fitted.set_params(&params).expect("set q");
    let k_mm = cross(&z, M, &z, M);
    let k_ms = cross(&z, M, &xs, Q);
    let k_ss = cross(&xs, Q, &xs, Q);
    let a = solve_lower(&cholesky(&k_mm), &k_ms);
    let u = matmul(&transpose(&q_l), &a);
    let a_a = matmul(&transpose(&a), &a);
    let u_u = matmul(&transpose(&u), &u);
    let latent: Dense = (0..Q)
        .map(|i| (0..Q).map(|j| k_ss[i][j] - a_a[i][j] + u_u[i][j]).collect())
        .collect();
    let mean: Vec<f64> = (0..Q)
        .map(|j| (0..M).map(|i| a[i][j] * q_mean[i]).sum())
        .collect();
    for kind in [VarianceKind::Latent, VarianceKind::Observation] {
        let cov = fitted
            .predict_covariance_with(&xs, Q, D, options(kind))
            .expect("covariance");
        assert_covariance("svgp", &cov, &with_noise(latent.clone(), kind), TOL);
        for (actual, expected) in cov.mean.iter().zip(&mean) {
            assert_close(*actual, *expected, TOL);
        }
    }
}

#[test]
fn sgpr_at_training_inducing_matches_exact_covariance() {
    let x = points(N, 0.0);
    let y = targets(&x, N);
    let xs = points(Q, 0.6);
    let sparse = Sgpr::new(kernel(), likelihood())
        .with_optimizer(Fixed)
        .factor(&x, N, D, &y, &x, N)
        .map_err(|(_, e)| e)
        .expect("factor");
    let exact = Gpr::new(kernel(), likelihood())
        .with_optimizer(Fixed)
        .factor(&x, N, D, &y)
        .map_err(|(_, e)| e)
        .expect("factor");
    for kind in [VarianceKind::Latent, VarianceKind::Observation] {
        let cov_s = sparse
            .predict_covariance_with(&xs, Q, D, options(kind))
            .expect("sparse");
        let cov_e = exact
            .predict_covariance_with(&xs, Q, D, options(kind))
            .expect("exact");
        for (s, e) in cov_s.covariance.iter().zip(&cov_e.covariance) {
            assert_close(*s, *e, REFACTOR_TOL);
        }
        for (s, e) in cov_s.mean.iter().zip(&cov_e.mean) {
            assert_close(*s, *e, REFACTOR_TOL);
        }
    }
}

/// The diagonal and the mean are exactly `predict`'s, and the matrix is
/// symmetric, with transforms, for every model.
#[test]
fn covariance_diagonal_is_predict_variance() {
    let x = points(N, 0.0);
    let y: Vec<f64> = targets(&x, N).iter().map(|v| 40.0 + 3.0 * v).collect();
    let z = points(M, 0.35);
    let xs = points(Q, 0.6);
    let fitted = Sgpr::new(kernel(), likelihood())
        .with_optimizer(Fixed)
        .with_input_transform(StandardizeInput::new())
        .with_target_transform(StandardizeTarget::new())
        .factor(&x, N, D, &y, &z, M)
        .map_err(|(_, e)| e)
        .expect("factor");
    let mut online = fitted.clone().into_online();
    online.insert(&[1.7, 0.2], 41.0).expect("insert");
    let svgp = Svgp::new(kernel(), likelihood())
        .with_input_transform(StandardizeInput::new())
        .with_target_transform(StandardizeTarget::new())
        .factor(&x, N, D, &y, &z, M)
        .map_err(|(_, e)| e)
        .expect("factor");
    for kind in [VarianceKind::Latent, VarianceKind::Observation] {
        let o = options(kind);
        let cases = [
            (
                "sgpr",
                fitted.predict_covariance_with(&xs, Q, D, o).expect("cov"),
                fitted.predict_with(&xs, Q, D, o).expect("pred"),
            ),
            (
                "online",
                online.predict_covariance_with(&xs, Q, D, o).expect("cov"),
                online.predict_with(&xs, Q, D, o).expect("pred"),
            ),
            (
                "svgp",
                svgp.predict_covariance_with(&xs, Q, D, o).expect("cov"),
                svgp.predict_with(&xs, Q, D, o).expect("pred"),
            ),
        ];
        for (label, cov, pred) in cases {
            assert_eq!(cov.mean, pred.mean, "{label} mean");
            assert_eq!(cov.variance_kind, kind);
            for i in 0..Q {
                assert_eq!(
                    cov.covariance[i * Q + i].to_bits(),
                    pred.variance[i].to_bits(),
                    "{label} diag {i}"
                );
                for j in 0..Q {
                    assert_eq!(
                        cov.covariance[i * Q + j].to_bits(),
                        cov.covariance[j * Q + i].to_bits(),
                        "{label} symmetry ({i}, {j})"
                    );
                }
            }
        }
    }
}

#[test]
fn online_covariance_matches_refactor() {
    let x = points(N, 0.0);
    let y = targets(&x, N);
    let z = points(M, 0.35);
    let xs = points(Q, 0.6);
    let mut online = Sgpr::new(kernel(), likelihood())
        .with_optimizer(Fixed)
        .factor(&x, N, D, &y, &z, M)
        .map_err(|(_, e)| e)
        .expect("factor")
        .into_online();
    let x_new = [2.3, -0.4];
    online.insert(&x_new, 0.6).expect("insert");
    let mut x_all = Vec::new();
    for dim in 0..D {
        x_all.extend_from_slice(&x[dim * N..(dim + 1) * N]);
        x_all.push(x_new[dim]);
    }
    let mut y_all = y.clone();
    y_all.push(0.6);
    let refactor = Sgpr::new(kernel(), likelihood())
        .with_optimizer(Fixed)
        .factor(&x_all, N + 1, D, &y_all, &z, M)
        .map_err(|(_, e)| e)
        .expect("factor");
    let cov_o = online.predict_covariance(&xs, Q, D).expect("online");
    let cov_r = refactor.predict_covariance(&xs, Q, D).expect("refactor");
    for (o, r) in cov_o.covariance.iter().zip(&cov_r.covariance) {
        assert_close(*o, *r, REFACTOR_TOL);
    }
}

#[test]
fn samples_follow_the_covariance() {
    let x = points(N, 0.0);
    let y = targets(&x, N);
    let z = points(M, 0.35);
    let xs = points(3, 0.6);
    let fitted = Sgpr::new(kernel(), likelihood())
        .with_optimizer(Fixed)
        .factor(&x, N, D, &y, &z, M)
        .map_err(|(_, e)| e)
        .expect("factor");
    let svgp = Svgp::new(kernel(), likelihood())
        .factor(&x, N, D, &y, &z, M)
        .map_err(|(_, e)| e)
        .expect("factor");
    let draws = 40_000;
    for (label, cov, samples, again, empty) in [
        (
            "sgpr",
            fitted.predict_covariance(&xs, 3, D).expect("cov"),
            fitted.sample(&xs, 3, D, draws, 7).expect("sample"),
            fitted.sample(&xs, 3, D, 2, 7).expect("sample"),
            fitted.sample(&xs, 3, D, 0, 7).expect("sample"),
        ),
        (
            "svgp",
            svgp.predict_covariance(&xs, 3, D).expect("cov"),
            svgp.sample(&xs, 3, D, draws, 7).expect("sample"),
            svgp.sample(&xs, 3, D, 2, 7).expect("sample"),
            svgp.sample(&xs, 3, D, 0, 7).expect("sample"),
        ),
    ] {
        assert_eq!(samples.len(), 3 * draws, "{label}");
        assert_eq!(
            &samples[..6],
            again.as_slice(),
            "{label}: same seed, same draws"
        );
        assert!(empty.is_empty(), "{label}");
        for i in 0..3 {
            let mean = (0..draws).map(|k| samples[k * 3 + i]).sum::<f64>() / draws as f64;
            assert!(
                (mean - cov.mean[i]).abs() < 0.03,
                "{label} mean {i}: {mean} vs {}",
                cov.mean[i]
            );
            for j in 0..3 {
                let mean_j = (0..draws).map(|k| samples[k * 3 + j]).sum::<f64>() / draws as f64;
                let c = (0..draws)
                    .map(|k| (samples[k * 3 + i] - mean) * (samples[k * 3 + j] - mean_j))
                    .sum::<f64>()
                    / (draws - 1) as f64;
                assert!(
                    (c - cov.covariance[j * 3 + i]).abs() < 0.03,
                    "{label} cov ({i}, {j}): {c} vs {}",
                    cov.covariance[j * 3 + i]
                );
            }
        }
    }
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn check_diagonal<P: gprx::GpScalar>(label: &str)
where
    P::Refine: PartialEq + std::fmt::Debug,
{
    let x = points(N, 0.0);
    let y = targets(&x, N);
    let z = points(M, 0.35);
    let xs = points(Q, 0.6);
    let fitted = Sgpr::new(kernel(), likelihood())
        .with_optimizer(Fixed)
        .with_precision::<P>()
        .factor(&x, N, D, &y, &z, M)
        .map_err(|(_, e)| e)
        .expect("factor");
    let svgp = Svgp::new(kernel(), likelihood())
        .with_precision::<P>()
        .factor(&x, N, D, &y, &z, M)
        .map_err(|(_, e)| e)
        .expect("factor");
    let cov = fitted.predict_covariance(&xs, Q, D).expect("cov");
    let pred = fitted.predict(&xs, Q, D).expect("pred");
    let cov_s = svgp.predict_covariance(&xs, Q, D).expect("cov");
    let pred_s = svgp.predict(&xs, Q, D).expect("pred");
    for i in 0..Q {
        assert_eq!(
            cov.covariance[i * Q + i],
            pred.variance[i],
            "{label} sgpr {i}"
        );
        assert_eq!(
            cov_s.covariance[i * Q + i],
            pred_s.variance[i],
            "{label} svgp {i}"
        );
    }
    assert_eq!(cov.mean, pred.mean, "{label} sgpr mean");
    assert_eq!(cov_s.mean, pred_s.mean, "{label} svgp mean");
}

#[test]
fn covariance_diagonal_is_predict_variance_other_precisions() {
    check_diagonal::<gprx::SinglePrecision>("f32");
    check_diagonal::<gprx::MixedPrecision<gprx::PromoteStorage>>("mixed promote");
    check_diagonal::<gprx::MixedPrecision<gprx::ReevaluateKernel>>("mixed reevaluate");
}
