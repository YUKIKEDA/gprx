//! Hyperparameter recovery for Exact GPR L-BFGS (P1B-4).
//!
//! Draws `y ~ N(0, K(ℓ*) + σn²* I)` on a 1-d grid, then [`Gpr::fit`]
//! starts from a far-away `θ`. Recovered lengthscale and noise sit near
//! the generating values. Negative log marginal likelihood is lower than
//! at the initial `θ` (log marginal likelihood improved).

use gprx::kernel::{KernelSpec, RbfKernel};
use gprx::{FitOptions, GaussianLikelihood, Gpr, GprError};

const N: usize = 40;
const D: usize = 1;
const ELL_TRUE: f64 = 1.0;
const NOISE_TRUE: f64 = 0.1;
const ELL_INIT: f64 = 4.0;
const NOISE_INIT: f64 = 1.0;
const X_MAX: f64 = 8.0;
const SEED: u64 = 0;
/// Recovered `ℓ` and `σn²` must lie within this relative band of the truth.
const REL_TOL: f64 = 0.5;

fn splitmix64(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^= z >> 31;
    (z >> 11) as f64 / ((1u64 << 53) as f64)
}

fn standard_normal(state: &mut u64) -> f64 {
    loop {
        let u = 2.0 * splitmix64(state) - 1.0;
        let v = 2.0 * splitmix64(state) - 1.0;
        let s = u * u + v * v;
        if s > 0.0 && s < 1.0 {
            return u * (-2.0 * s.ln() / s).sqrt();
        }
    }
}

fn grid_x(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| X_MAX * (i as f64) / ((n - 1) as f64))
        .collect()
}

fn rbf_cov(x: &[f64], ell: f64, noise: f64) -> Vec<f64> {
    let n = x.len();
    let inv_ell_sq = 1.0 / (ell * ell);
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let d = x[i] - x[j];
            let k = (-0.5 * d * d * inv_ell_sq).exp();
            a[i * n + j] = k;
            a[j * n + i] = k;
        }
        a[i * n + i] += noise;
    }
    a
}

fn cholesky_lower(a: &mut [f64], n: usize) {
    for j in 0..n {
        let mut diag = a[j * n + j];
        for k in 0..j {
            diag -= a[j * n + k] * a[j * n + k];
        }
        assert!(diag > 0.0, "covariance is not SPD at column {j}");
        let ljj = diag.sqrt();
        a[j * n + j] = ljj;
        for i in (j + 1)..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = s / ljj;
        }
    }
}

fn sample_gp(x: &[f64], ell: f64, noise: f64, seed: u64) -> Vec<f64> {
    let n = x.len();
    let mut a = rbf_cov(x, ell, noise);
    cholesky_lower(&mut a, n);
    let mut state = seed;
    let z: Vec<f64> = (0..n).map(|_| standard_normal(&mut state)).collect();
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..=i {
            s += a[i * n + j] * z[j];
        }
        y[i] = s;
    }
    y
}

fn rbf_gpr(ell: f64, noise: f64) -> Result<Gpr, GprError> {
    Ok(Gpr::new(
        KernelSpec::from(RbfKernel::new(ell)?),
        GaussianLikelihood::new(noise)?,
    ))
}

fn rel_err(actual: f64, expected: f64) -> f64 {
    (actual - expected).abs() / expected.abs()
}

fn assert_near(label: &str, actual: f64, expected: f64) {
    let err = rel_err(actual, expected);
    assert!(
        err <= REL_TOL,
        "{label}: actual={actual}, expected={expected}, rel_err={err}, tol={REL_TOL}"
    );
}

#[test]
fn fit_recovers_rbf_lengthscale_and_noise() {
    let x = grid_x(N);
    let y = sample_gp(&x, ELL_TRUE, NOISE_TRUE, SEED);

    let mut at_init = rbf_gpr(ELL_INIT, NOISE_INIT).expect("valid init");
    at_init
        .fit_with(&x, N, D, &y, FitOptions::FIXED)
        .expect("spd at init");
    let nlml_init = at_init.neg_log_marginal_likelihood().expect("fitted init");

    let mut gpr = rbf_gpr(ELL_INIT, NOISE_INIT).expect("valid init");
    gpr.fit(&x, N, D, &y).expect("lbfgs");
    let nlml_fit = gpr.neg_log_marginal_likelihood().expect("fitted");

    assert!(
        nlml_fit < nlml_init,
        "NLML should fall (LML should rise): init={nlml_init}, fit={nlml_fit}"
    );

    let mut params = [0.0; 2];
    gpr.get_params(&mut params).expect("len 2");
    let ell = params[0].exp();
    let noise = params[1].exp();
    assert_near("lengthscale", ell, ELL_TRUE);
    assert_near("noise", noise, NOISE_TRUE);
    assert!(
        rel_err(ell, ELL_TRUE) < rel_err(ELL_INIT, ELL_TRUE),
        "lengthscale should move toward the truth: fitted={ell}, init={ELL_INIT}, true={ELL_TRUE}"
    );
    assert!(
        rel_err(noise, NOISE_TRUE) < rel_err(NOISE_INIT, NOISE_TRUE),
        "noise should move toward the truth: fitted={noise}, init={NOISE_INIT}, true={NOISE_TRUE}"
    );
}
