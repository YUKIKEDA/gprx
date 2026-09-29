use super::{DoublePrecision, PrecisionPolicy};

use crate::test_check::assert_send_sync;

#[test]
fn double_precision_is_f64() {
    let storage: <DoublePrecision as PrecisionPolicy>::Storage = 0.0;
    let refine: <DoublePrecision as PrecisionPolicy>::Refine = 0.0;
    let _ = (storage, refine);
    assert_send_sync::<DoublePrecision>();
}

#[test]
fn single_precision_is_f32() {
    let storage: <super::SinglePrecision as PrecisionPolicy>::Storage = 0.0;
    let refine: <super::SinglePrecision as PrecisionPolicy>::Refine = 0.0;
    let _ = (storage, refine);
    assert_send_sync::<super::SinglePrecision>();
}

#[test]
fn mixed_precision_stores_f32_and_refines_f64() {
    let storage: <super::MixedPrecision as PrecisionPolicy>::Storage = 0.0;
    let refine: <super::MixedPrecision as PrecisionPolicy>::Refine = 0.0;
    let fresh: <super::MixedPrecision<super::ReevaluateKernel> as PrecisionPolicy>::Refine = 0.0;
    let _ = (storage, refine, fresh);
    assert_send_sync::<super::MixedPrecision>();
    assert_send_sync::<super::MixedPrecision<super::ReevaluateKernel>>();
}

use super::refine::{f64_alpha, refine_alpha};
use super::{PromoteStorage, ReevaluateKernel, ResidualFormula, StoredFactor, TrainSystem};
use crate::error::CholeskyStage;
use crate::kernel::ScalarOps;
use crate::kernel::{KernelSpec, RbfKernel, Triangle};
use faer::{Mat, MatRef};
use std::time::Instant;

/// `ℓ` and `σn²` for the ill-conditioned Forrester probe (`n = 256`).
const ILL_LENGTHSCALE: f64 = 1.0e4;
const ILL_NOISE: f64 = 1.0e-5;

fn forrester(n: usize) -> (Mat<f64>, Vec<f64>) {
    let (x, y) = crate::test_problems::forrester_xy(n, 0, 1.0);
    (Mat::from_fn(n, 1, |i, _| x[i]), y)
}

fn rbf(
    ell: f64,
) -> (
    crate::kernel::CompiledKernel<f32>,
    crate::kernel::CompiledKernel<f64>,
) {
    let spec = KernelSpec::from(RbfKernel::new(ell).expect("lengthscale"));
    (spec.compile_as::<f32>(), spec.compile())
}

fn rel_inf(got: &[f64], expect: &[f64]) -> f64 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (g, e) in got.iter().zip(expect) {
        num = num.max((g - e).abs());
        den = den.max(e.abs());
    }
    num / den
}

/// `A + σn² I` factored in `f32` the way fit does, with its `α₀`.
struct Fresh {
    spec: KernelSpec,
    k32: crate::kernel::CompiledKernel<f32>,
    k64: crate::kernel::CompiledKernel<f64>,
    l: Mat<f32>,
    factor_alpha: Vec<f32>,
}

fn fresh_factor(ell: f64, noise: f64, x: MatRef<'_, f64>, y: &[f64]) -> Fresh {
    let spec = KernelSpec::from(RbfKernel::new(ell).expect("lengthscale"));
    let (k32, k64) = (spec.compile_as::<f32>(), spec.compile());
    let n = x.nrows();
    let x32 = Mat::<f32>::from_fn(n, x.ncols(), |i, j| x[(i, j)] as f32);
    let mut l = Mat::<f32>::zeros(n, n);
    let mut scratch = Mat::<f32>::zeros(n, n);
    k32.apply_points::<crate::math::Accurate>(
        x32.as_ref(),
        l.as_mut(),
        Triangle::Lower,
        scratch.as_mut(),
    )
    .expect("gram");
    for i in 0..n {
        l[(i, i)] += noise as f32;
    }
    let req = faer::linalg::cholesky::llt::factor::cholesky_in_place_scratch::<f32>(
        n,
        faer::Par::Seq,
        Default::default(),
    );
    let mut buf = dyn_stack::MemBuffer::new(req);
    <f32 as ScalarOps>::cholesky_lower(&mut l, &mut buf, 0.0, CholeskyStage::Fit)
        .expect("f32 factor");
    let mut rhs = Mat::<f32>::from_fn(n, 1, |i, _| y[i] as f32);
    f32::solve_llt_owned_scratch(l.as_ref(), rhs.as_mut());
    let factor_alpha: Vec<f32> = (0..n).map(|i| rhs[(i, 0)]).collect();
    Fresh {
        spec,
        k32,
        k64,
        l,
        factor_alpha,
    }
}

impl Fresh {
    fn system<'a>(&'a self, x: MatRef<'a, f64>, y: &'a [f64], noise: f64) -> TrainSystem<'a, f32> {
        TrainSystem {
            kernel: &self.spec,
            compiled: &self.k32,
            x,
            y,
            noise,
            jitter: 0.0,
            factor: StoredFactor::Llt(self.l.as_ref()),
            factor_alpha: &self.factor_alpha,
            policy: crate::policy::JitterPolicy::default(),
            stage: CholeskyStage::Fit,
        }
    }
}

/// Factors `A + σn² I` in `f32` the way fit does, then refines on that
/// factor. Returns the refined `α` and the `f64` Cholesky `α`.
fn refine_fresh<R: ResidualFormula>(
    ell: f64,
    noise: f64,
    x: MatRef<'_, f64>,
    y: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let fresh = fresh_factor(ell, noise, x, y);
    let sys = fresh.system(x, y, noise);
    let alpha = refine_alpha::<crate::math::Accurate, R>(&sys).expect("refine");
    let truth = f64_alpha::<crate::math::Accurate>(&fresh.k64, &sys, noise).expect("f64");
    (alpha, truth)
}

fn digits<R: ResidualFormula>(ell: f64, noise: f64, x: MatRef<'_, f64>, y: &[f64]) {
    let (alpha, truth) = refine_fresh::<R>(ell, noise, x, y);
    let rel = rel_inf(&alpha, &truth);
    let bar = 10.0 * y.len() as f64 * f64::from(f32::EPSILON);
    assert!(rel < bar, "relative {rel} bar {bar}");
}

fn kappa(kernel: &crate::kernel::CompiledKernel<f64>, x: MatRef<'_, f64>, noise: f64) -> f64 {
    let n = x.nrows();
    let mut a = Mat::<f64>::zeros(n, n);
    let mut scratch = Mat::<f64>::zeros(n, n);
    kernel
        .apply_points::<crate::math::Accurate>(x, a.as_mut(), Triangle::Lower, scratch.as_mut())
        .expect("gram");
    for i in 0..n {
        a[(i, i)] += noise;
    }
    for col in 0..n {
        for row in (col + 1)..n {
            a[(col, row)] = a[(row, col)];
        }
    }
    let mut v = vec![1.0 / (n as f64).sqrt(); n];
    for _ in 0..40 {
        let mut w = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                w[i] += a[(i, j)] * v[j];
            }
        }
        let norm = w.iter().map(|t| t * t).sum::<f64>().sqrt();
        for (slot, value) in v.iter_mut().zip(&w) {
            *slot = value / norm;
        }
    }
    let mut av = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            av[i] += a[(i, j)] * v[j];
        }
    }
    let lam_max = v.iter().zip(&av).map(|(vi, avi)| vi * avi).sum::<f64>();
    let mut factor = a.clone();
    crate::linalg::cholesky_lower_owned(&mut factor, [], crate::error::CholeskyStage::Predict)
        .expect("f64 factor");
    let mut z = v.clone();
    for _ in 0..40 {
        let mut rhs = Mat::<f64>::from_fn(n, 1, |i, _| z[i]);
        crate::linalg::solve_llt_faer_owned(factor.as_ref(), rhs.as_mut());
        let mut norm = 0.0;
        for i in 0..n {
            z[i] = rhs[(i, 0)];
            norm += z[i] * z[i];
        }
        norm = norm.sqrt();
        for slot in &mut z {
            *slot /= norm;
        }
    }
    for i in 0..n {
        av[i] = 0.0;
        for j in 0..n {
            av[i] += a[(i, j)] * z[j];
        }
    }
    let lam_min = z.iter().zip(&av).map(|(zi, avi)| zi * avi).sum::<f64>();
    lam_max / lam_min
}

#[test]
fn both_residuals_hit_the_digit_bar() {
    let (x256, y256) = forrester(256);
    let (x1024, y1024) = forrester(1024);
    let (_, k64) = rbf(ILL_LENGTHSCALE);
    let cond = kappa(&k64, x256.as_ref(), ILL_NOISE);
    assert!(
        cond * f64::from(f32::EPSILON) > 1.0,
        "κ {cond} with ℓ={ILL_LENGTHSCALE} σn²={ILL_NOISE}"
    );
    for (x, y, ell, noise) in [
        (x256.as_ref(), y256.as_slice(), 1.0, 0.1),
        (x1024.as_ref(), y1024.as_slice(), 1.0, 0.1),
        (x256.as_ref(), y256.as_slice(), ILL_LENGTHSCALE, ILL_NOISE),
    ] {
        digits::<PromoteStorage>(ell, noise, x, y);
        digits::<ReevaluateKernel>(ell, noise, x, y);
    }
}

#[test]
#[ignore]
fn time_forrester_1024() {
    let (x, y) = forrester(1024);
    let median = |tag: &str, run: &dyn Fn()| {
        run();
        let mut samples = [0.0; 11];
        for sample in &mut samples {
            let start = Instant::now();
            run();
            *sample = start.elapsed().as_secs_f64() * 1e3;
        }
        samples.sort_by(|a, b| a.total_cmp(b));
        println!("{tag} {:.4} ms", samples[5]);
    };
    // Refinement only: the f32 factor is fit's, built once outside the timing.
    let fresh = fresh_factor(1.0, 0.1, x.as_ref(), &y);
    let sys = fresh.system(x.as_ref(), &y, 0.1);
    median("promote", &|| {
        refine_alpha::<crate::math::Accurate, PromoteStorage>(&sys).expect("promote");
    });
    median("reevaluate", &|| {
        refine_alpha::<crate::math::Accurate, ReevaluateKernel>(&sys).expect("reevaluate");
    });
}

use crate::kernel::KernelScalar;
use crate::{Fixed, GaussianLikelihood, Gpr, MixedPrecision, Sgpr, SinglePrecision, Svgp};

fn must<T>(result: Result<T, crate::GprError>) -> T {
    match result {
        Ok(value) => value,
        Err(err) => panic!("gpr call failed: {err}"),
    }
}

fn likelihood_at(noise: f64) -> GaussianLikelihood {
    if crate::param::Interval::DEFAULT_POSITIVE.contains(noise) {
        return must(GaussianLikelihood::new(noise));
    }
    let interval = match crate::Interval::new(1.0e-12, 1.0e5) {
        Ok(interval) => interval,
        Err(err) => panic!("noise interval: {err}"),
    };
    let mut likelihood = match GaussianLikelihood::new(0.1) {
        Ok(likelihood) => match likelihood.with_bounds(interval) {
            Ok(likelihood) => likelihood,
            Err(err) => panic!("noise bounds: {err}"),
        },
        Err(err) => panic!("gpr call failed: {err}"),
    };
    must(likelihood.set_params(&[noise.ln()]));
    likelihood
}

fn digit_tol(n: usize) -> f64 {
    10.0 * n as f64 * f64::from(f32::EPSILON)
}

fn near(got: f64, expect: f64, n: usize) -> bool {
    let tol = digit_tol(n);
    let err = (got - expect).abs();
    if expect.abs() < tol {
        err < tol
    } else {
        err / expect.abs() < tol
    }
}

fn assert_near(got: f64, expect: f64, n: usize) {
    assert!(
        near(got, expect, n),
        "got {got} expect {expect} tol {}",
        digit_tol(n)
    );
}

fn pack_col(x: &Mat<f64>) -> Vec<f64> {
    (0..x.nrows()).map(|i| x[(i, 0)]).collect()
}

fn inducing8() -> Vec<f64> {
    (0..8).map(|i| i as f64 / 7.0).collect()
}

fn queries() -> [f64; 2] {
    [0.25, 0.75]
}

/// 32 near-duplicate pairs: the `f32` factor of `K + σn² I` needs a jitter retry.
fn jitter_problem() -> (Vec<f64>, Vec<f64>) {
    let mut x = Vec::new();
    for i in 0..32 {
        let t = f64::from(i) * 0.25;
        x.push(t);
        x.push(t + 1e-4);
    }
    let y = x.iter().map(|t| (1.7 * t).sin()).collect();
    (x, y)
}

fn jitter_fit<R>(x: &[f64], y: &[f64], noise: f64) -> crate::FittedGpr<Fixed, MixedPrecision<R>>
where
    MixedPrecision<R>: crate::precision::GpScalar,
    R: ResidualFormula,
{
    let kernel = KernelSpec::from(must(RbfKernel::new(1.0)));
    let policy = must(crate::JitterPolicy::adaptive(1e-6, 10.0, 8, 1e-1));
    must(
        Gpr::new(kernel, likelihood_at(noise))
            .with_optimizer(Fixed)
            .with_jitter_policy(policy)
            .with_precision::<MixedPrecision<R>>()
            .factor(x, y.len(), 1, y)
            .map_err(|(_, err)| err),
    )
}

/// R3-2 (#237): refinement solves the jittered system fit factored, with
/// that factor, so fit and predict succeed and match `f64` on `A + (σn² + j) I`.
#[test]
fn mixed_refines_on_the_fit_factor_and_jitter() {
    let (x, y) = jitter_problem();
    let n = y.len();
    let noise = 1e-8;
    let q = [0.3, 2.0, 5.1];
    let check = |mean: &[f64], jitter: f64| {
        assert!(jitter > 0.0, "the f32 factor should need a jitter retry");
        let truth = factor_exact_noise(&x, &y, noise + jitter);
        let expect = must(truth.predict(&q, q.len(), 1));
        for (got, want) in mean.iter().zip(&expect.mean) {
            assert_near(*got, *want, n);
        }
    };
    let promote = jitter_fit::<PromoteStorage>(&x, &y, noise);
    let p = must(promote.predict(&q, q.len(), 1));
    check(&p.mean, promote.factor_jitter());
    let fresh = jitter_fit::<ReevaluateKernel>(&x, &y, noise);
    let r = must(fresh.predict(&q, q.len(), 1));
    check(&r.mean, fresh.factor_jitter());
}

fn factor_exact_noise(x: &[f64], y: &[f64], noise: f64) -> crate::FittedGpr<Fixed> {
    let kernel = KernelSpec::from(must(RbfKernel::new(1.0)));
    must(
        Gpr::new(kernel, likelihood_at(noise))
            .with_optimizer(Fixed)
            .factor(x, y.len(), 1, y)
            .map_err(|(_, err)| err),
    )
}

fn factor_exact<P>(x: &[f64], y: &[f64], ell: f64, noise: f64) -> crate::FittedGpr<Fixed, P>
where
    P: crate::precision::GpScalar,
{
    let kernel = KernelSpec::from(must(RbfKernel::new(ell)));
    let likelihood = likelihood_at(noise);
    let n = y.len();
    must(
        Gpr::new(kernel, likelihood)
            .with_optimizer(Fixed)
            .with_precision::<P>()
            .factor(x, n, 1, y)
            .map_err(|(_, err)| err),
    )
}

fn assert_predict_pair<T: KernelScalar, U: KernelScalar>(
    got_mean: &[T],
    got_var: &[T],
    exp_mean: &[U],
    exp_var: &[U],
    n: usize,
    digits: bool,
) {
    for (got, expect) in got_mean.iter().zip(exp_mean) {
        let g = got.to_f64();
        let e = expect.to_f64();
        if digits {
            assert!(
                near(g, e, n),
                "mean got {g} expect {e} tol {}",
                digit_tol(n)
            );
        } else {
            assert!(g.is_finite(), "mean {g}");
        }
    }
    for (got, expect) in got_var.iter().zip(exp_var) {
        let g = got.to_f64();
        let e = expect.to_f64();
        if digits {
            assert!(near(g, e, n), "var got {g} expect {e} tol {}", digit_tol(n));
        } else {
            assert!(g.is_finite(), "variance {g}");
        }
    }
}

fn exact_digits(n: usize) {
    let (x, y) = forrester(n);
    let packed = pack_col(&x);
    let f64_model = factor_exact::<DoublePrecision>(&packed, &y, 1.0, 0.1);
    let single = factor_exact::<SinglePrecision>(&packed, &y, 1.0, 0.1);
    let promote = factor_exact::<MixedPrecision<PromoteStorage>>(&packed, &y, 1.0, 0.1);
    let fresh = factor_exact::<MixedPrecision<ReevaluateKernel>>(&packed, &y, 1.0, 0.1);
    let q = queries();
    let truth = must(f64_model.predict(&q, 2, 1));
    let s = must(single.predict(&q, 2, 1));
    let p = must(promote.predict(&q, 2, 1));
    let r = must(fresh.predict(&q, 2, 1));
    assert_predict_pair(&s.mean, &s.variance, &truth.mean, &truth.variance, n, true);
    assert_predict_pair(&p.mean, &p.variance, &truth.mean, &truth.variance, n, true);
    assert_predict_pair(&r.mean, &r.variance, &truth.mean, &truth.variance, n, true);
    let cov_t = must(f64_model.predict_covariance(&q, 2, 1));
    let cov_s = must(single.predict_covariance(&q, 2, 1));
    let cov_p = must(promote.predict_covariance(&q, 2, 1));
    let cov_r = must(fresh.predict_covariance(&q, 2, 1));
    for (got, expect) in cov_s.covariance.iter().zip(&cov_t.covariance) {
        assert_near(got.to_f64(), expect.to_f64(), n);
    }
    for (got, expect) in cov_p.covariance.iter().zip(&cov_t.covariance) {
        assert_near(got.to_f64(), expect.to_f64(), n);
    }
    for (got, expect) in cov_r.covariance.iter().zip(&cov_t.covariance) {
        assert_near(got.to_f64(), expect.to_f64(), n);
    }
    let loo_t = must(f64_model.loo_predict());
    let loo_s = must(single.loo_predict());
    let loo_p = must(promote.loo_predict());
    let loo_r = must(fresh.loo_predict());
    assert_predict_pair(
        &loo_s.mean,
        &loo_s.variance,
        &loo_t.mean,
        &loo_t.variance,
        n,
        true,
    );
    assert_predict_pair(
        &loo_p.mean,
        &loo_p.variance,
        &loo_t.mean,
        &loo_t.variance,
        n,
        true,
    );
    assert_predict_pair(
        &loo_r.mean,
        &loo_r.variance,
        &loo_t.mean,
        &loo_t.variance,
        n,
        true,
    );
    let draws = must(single.sample(&q, 2, 1, 3, 7));
    assert_eq!(draws.len(), 6);
    assert!(draws.iter().all(|v| v.is_finite()));
    let draws_m = must(promote.sample(&q, 2, 1, 3, 7));
    assert_eq!(draws_m.len(), 6);
    assert!(draws_m.iter().all(|v| v.is_finite()));
    let draws_r = must(fresh.sample(&q, 2, 1, 3, 7));
    assert_eq!(draws_r.len(), 6);
    assert!(draws_r.iter().all(|v| v.is_finite()));
}

#[test]
fn single_and_mixed_predict_match_f64_digits() {
    exact_digits(256);
}

#[test]
fn single_and_mixed_predict_match_f64_digits_n1024() {
    exact_digits(1024);
}

#[test]
fn ill_conditioned_single_is_finite_and_mixed_mean_hits_digits() {
    let (x, y) = forrester(256);
    let packed = pack_col(&x);
    let n = y.len();
    let noise = ILL_NOISE;
    let f64_model = factor_exact::<DoublePrecision>(&packed, &y, ILL_LENGTHSCALE, noise);
    let single = factor_exact::<SinglePrecision>(&packed, &y, ILL_LENGTHSCALE, noise);
    let promote =
        factor_exact::<MixedPrecision<PromoteStorage>>(&packed, &y, ILL_LENGTHSCALE, noise);
    let fresh =
        factor_exact::<MixedPrecision<ReevaluateKernel>>(&packed, &y, ILL_LENGTHSCALE, noise);
    let q = queries();
    let truth = must(f64_model.predict(&q, 2, 1));
    let s = must(single.predict(&q, 2, 1));
    let p = must(promote.predict(&q, 2, 1));
    let r = must(fresh.predict(&q, 2, 1));
    assert_predict_pair(&s.mean, &s.variance, &truth.mean, &truth.variance, n, false);
    assert_predict_pair(&p.mean, &p.variance, &truth.mean, &truth.variance, n, false);
    assert_predict_pair(&r.mean, &r.variance, &truth.mean, &truth.variance, n, false);
    for (got, expect) in p.mean.iter().zip(&truth.mean) {
        assert_near(got.to_f64(), expect.to_f64(), n);
    }
    for (got, expect) in r.mean.iter().zip(&truth.mean) {
        assert_near(got.to_f64(), expect.to_f64(), n);
    }
    let cov_s = must(single.predict_covariance(&q, 2, 1));
    assert!(cov_s.covariance.iter().all(|v| v.is_finite()));
    let cov_p = must(promote.predict_covariance(&q, 2, 1));
    assert!(cov_p.covariance.iter().all(|v| v.is_finite()));
}

fn factor_sgpr<P>(
    x: &[f64],
    y: &[f64],
    ell: f64,
    noise: f64,
) -> crate::FittedSgpr<Fixed, crate::FixedInducing, P>
where
    P: crate::precision::GpScalar,
{
    let kernel = KernelSpec::from(must(RbfKernel::new(ell)));
    let likelihood = likelihood_at(noise);
    let z = inducing8();
    must(
        Sgpr::new(kernel, likelihood)
            .with_optimizer(Fixed)
            .with_precision::<P>()
            .factor(x, y.len(), 1, y, &z, 8)
            .map_err(|(_, err)| err),
    )
}

fn sgpr_case(n: usize, ell: f64, noise: f64, mean_digits: bool, var_digits: bool) {
    let (x, y) = forrester(n);
    let packed = pack_col(&x);
    let truth = factor_sgpr::<DoublePrecision>(&packed, &y, ell, noise);
    let single = factor_sgpr::<SinglePrecision>(&packed, &y, ell, noise);
    let promote = factor_sgpr::<MixedPrecision<PromoteStorage>>(&packed, &y, ell, noise);
    let fresh = factor_sgpr::<MixedPrecision<ReevaluateKernel>>(&packed, &y, ell, noise);
    let q = queries();
    let t = must(truth.predict(&q, 2, 1));
    let s = must(single.predict(&q, 2, 1));
    let p = must(promote.predict(&q, 2, 1));
    let r = must(fresh.predict(&q, 2, 1));
    let single_mean_digits = mean_digits && ell < ILL_LENGTHSCALE;
    assert_predict_pair(
        &s.mean,
        &s.variance,
        &t.mean,
        &t.variance,
        n,
        single_mean_digits,
    );
    assert_predict_pair(&p.mean, &p.variance, &t.mean, &t.variance, n, var_digits);
    assert_predict_pair(&r.mean, &r.variance, &t.mean, &t.variance, n, var_digits);
    if !var_digits {
        for (got, expect) in p.mean.iter().zip(&t.mean) {
            assert_near(got.to_f64(), expect.to_f64(), n);
        }
        for (got, expect) in r.mean.iter().zip(&t.mean) {
            assert_near(got.to_f64(), expect.to_f64(), n);
        }
    }
}

#[test]
fn sgpr_precisions_match_f64_predict() {
    sgpr_case(256, 1.0, 0.1, true, true);
    sgpr_case(256, ILL_LENGTHSCALE, ILL_NOISE, false, false);
}

#[test]
fn sgpr_precisions_match_f64_predict_n1024() {
    sgpr_case(1024, 1.0, 0.1, true, true);
}

fn factor_svgp<P>(x: &[f64], y: &[f64], ell: f64, noise: f64) -> crate::FittedSvgp<P>
where
    P: crate::precision::GpScalar,
{
    let kernel = KernelSpec::from(must(RbfKernel::new(ell)));
    let likelihood = likelihood_at(noise);
    let z = inducing8();
    must(
        Svgp::new(kernel, likelihood)
            .with_precision::<P>()
            .factor(x, y.len(), 1, y, &z, 8)
            .map_err(|(_, err)| err),
    )
}

#[test]
fn svgp_precisions_match_f64_predict() {
    for (n, ell, noise, var_digits) in [
        (256usize, 1.0, 0.1, true),
        (1024usize, 1.0, 0.1, true),
        (256usize, ILL_LENGTHSCALE, ILL_NOISE, false),
    ] {
        let (x, y) = forrester(n);
        let packed = pack_col(&x);
        let truth = factor_svgp::<DoublePrecision>(&packed, &y, ell, noise);
        let single = factor_svgp::<SinglePrecision>(&packed, &y, ell, noise);
        let promote = factor_svgp::<MixedPrecision<PromoteStorage>>(&packed, &y, ell, noise);
        let fresh = factor_svgp::<MixedPrecision<ReevaluateKernel>>(&packed, &y, ell, noise);
        let q = queries();
        let t = must(truth.predict(&q, 2, 1));
        let s = must(single.predict(&q, 2, 1));
        let p = must(promote.predict(&q, 2, 1));
        let r = must(fresh.predict(&q, 2, 1));
        assert_predict_pair(&s.mean, &s.variance, &t.mean, &t.variance, n, var_digits);
        assert_predict_pair(&p.mean, &p.variance, &t.mean, &t.variance, n, var_digits);
        assert_predict_pair(&r.mean, &r.variance, &t.mean, &t.variance, n, var_digits);
        if !var_digits {
            for (got, expect) in p.mean.iter().zip(&t.mean) {
                assert_near(got.to_f64(), expect.to_f64(), n);
            }
            for (got, expect) in r.mean.iter().zip(&t.mean) {
                assert_near(got.to_f64(), expect.to_f64(), n);
            }
        }
    }
}

fn online_exact_round<P>(
    packed: &[f64],
    y: &[f64],
    ell: f64,
    noise: f64,
) -> crate::OnlineGpr<Fixed, P>
where
    P: crate::precision::GpScalar,
{
    let mut online = must(factor_exact::<P>(packed, y, ell, noise).into_online());
    must(online.insert(&[0.33], 0.2));
    must(online.delete(online.point_ids()[0]));
    online
}

#[test]
fn online_exact_insert_delete_matches_f64() {
    for n in [256usize, 1024] {
        let (x, y) = forrester(n);
        let packed = pack_col(&x);
        let base = online_exact_round::<DoublePrecision>(&packed, &y, 1.0, 0.1);
        let single = online_exact_round::<SinglePrecision>(&packed, &y, 1.0, 0.1);
        let promote = online_exact_round::<MixedPrecision<PromoteStorage>>(&packed, &y, 1.0, 0.1);
        let fresh = online_exact_round::<MixedPrecision<ReevaluateKernel>>(&packed, &y, 1.0, 0.1);
        let q = queries();
        let t = must(base.predict(&q, 2, 1));
        let s = must(single.predict(&q, 2, 1));
        let p = must(promote.predict(&q, 2, 1));
        let r = must(fresh.predict(&q, 2, 1));
        assert_predict_pair(&s.mean, &s.variance, &t.mean, &t.variance, n, true);
        assert_predict_pair(&p.mean, &p.variance, &t.mean, &t.variance, n, true);
        assert_predict_pair(&r.mean, &r.variance, &t.mean, &t.variance, n, true);
    }
    let (x, y) = forrester(256);
    let packed = pack_col(&x);
    let n = y.len();
    let noise = ILL_NOISE;
    let base = online_exact_round::<DoublePrecision>(&packed, &y, ILL_LENGTHSCALE, noise);
    let single = online_exact_round::<SinglePrecision>(&packed, &y, ILL_LENGTHSCALE, noise);
    let promote =
        online_exact_round::<MixedPrecision<PromoteStorage>>(&packed, &y, ILL_LENGTHSCALE, noise);
    let fresh =
        online_exact_round::<MixedPrecision<ReevaluateKernel>>(&packed, &y, ILL_LENGTHSCALE, noise);
    let q = queries();
    let t = must(base.predict(&q, 2, 1));
    let s = must(single.predict(&q, 2, 1));
    let p = must(promote.predict(&q, 2, 1));
    let r = must(fresh.predict(&q, 2, 1));
    assert_predict_pair(&s.mean, &s.variance, &t.mean, &t.variance, n, false);
    assert_predict_pair(&p.mean, &p.variance, &t.mean, &t.variance, n, false);
    assert_predict_pair(&r.mean, &r.variance, &t.mean, &t.variance, n, false);
    for (got, expect) in p.mean.iter().zip(&t.mean) {
        assert_near(got.to_f64(), expect.to_f64(), n);
    }
    for (got, expect) in r.mean.iter().zip(&t.mean) {
        assert_near(got.to_f64(), expect.to_f64(), n);
    }
}

fn online_sgpr_round<P>(
    packed: &[f64],
    y: &[f64],
    ell: f64,
    noise: f64,
) -> crate::OnlineSgpr<Fixed, P>
where
    P: crate::precision::GpScalar,
{
    let mut online = factor_sgpr::<P>(packed, y, ell, noise).into_online();
    must(online.insert(&[0.33], 0.2));
    must(online.delete(online.point_ids()[0]));
    online
}

#[test]
fn online_sgpr_insert_delete_matches_f64() {
    let (x, y) = forrester(256);
    let packed = pack_col(&x);
    let n = y.len();
    let base = online_sgpr_round::<DoublePrecision>(&packed, &y, 1.0, 0.1);
    let single = online_sgpr_round::<SinglePrecision>(&packed, &y, 1.0, 0.1);
    let promote = online_sgpr_round::<MixedPrecision<PromoteStorage>>(&packed, &y, 1.0, 0.1);
    let fresh = online_sgpr_round::<MixedPrecision<ReevaluateKernel>>(&packed, &y, 1.0, 0.1);
    let q = queries();
    let t = must(base.predict(&q, 2, 1));
    let s = must(single.predict(&q, 2, 1));
    let p = must(promote.predict(&q, 2, 1));
    let r = must(fresh.predict(&q, 2, 1));
    assert_predict_pair(&s.mean, &s.variance, &t.mean, &t.variance, n, true);
    assert_predict_pair(&p.mean, &p.variance, &t.mean, &t.variance, n, true);
    assert_predict_pair(&r.mean, &r.variance, &t.mean, &t.variance, n, true);

    let noise = ILL_NOISE;
    let base = online_sgpr_round::<DoublePrecision>(&packed, &y, ILL_LENGTHSCALE, noise);
    let single = online_sgpr_round::<SinglePrecision>(&packed, &y, ILL_LENGTHSCALE, noise);
    let promote =
        online_sgpr_round::<MixedPrecision<PromoteStorage>>(&packed, &y, ILL_LENGTHSCALE, noise);
    let fresh =
        online_sgpr_round::<MixedPrecision<ReevaluateKernel>>(&packed, &y, ILL_LENGTHSCALE, noise);
    let t = must(base.predict(&q, 2, 1));
    let s = must(single.predict(&q, 2, 1));
    let p = must(promote.predict(&q, 2, 1));
    let r = must(fresh.predict(&q, 2, 1));
    assert_predict_pair(&s.mean, &s.variance, &t.mean, &t.variance, n, false);
    assert_predict_pair(&p.mean, &p.variance, &t.mean, &t.variance, n, false);
    assert_predict_pair(&r.mean, &r.variance, &t.mean, &t.variance, n, false);
    for (got, expect) in p.mean.iter().zip(&t.mean) {
        assert_near(got.to_f64(), expect.to_f64(), n);
    }
    for (got, expect) in r.mean.iter().zip(&t.mean) {
        assert_near(got.to_f64(), expect.to_f64(), n);
    }
}

#[test]
fn single_and_mixed_gradient_matches_own_nlml() {
    let n = 8usize;
    let (x, y) = forrester(n);
    let packed = pack_col(&x);
    check_grad::<SinglePrecision>(&packed, &y);
    check_grad::<MixedPrecision<PromoteStorage>>(&packed, &y);
    check_grad::<MixedPrecision<ReevaluateKernel>>(&packed, &y);
}

fn check_grad<P>(x: &[f64], y: &[f64])
where
    P: crate::precision::GpScalar,
{
    let mut fitted = factor_exact::<P>(x, y, 1.0, 0.1);
    let mut params = [0.0; 2];
    must(fitted.get_params(&mut params));
    let mut grad = [0.0; 2];
    let _ = must(fitted.value_and_gradient_into(&params, &mut grad));
    let step = 1.0e-3;
    for i in 0..2 {
        let mut up = params;
        let mut down = params;
        up[i] += step;
        down[i] -= step;
        must(fitted.set_params(&up));
        let plus = must(fitted.neg_log_marginal_likelihood());
        must(fitted.set_params(&down));
        let minus = must(fitted.neg_log_marginal_likelihood());
        let fd = (plus - minus) / (2.0 * step);
        let scale = fd.abs().max(1.0);
        let err = (grad[i] - fd).abs() / scale;
        assert!(err < 2.0e-3, "param {i} grad {} fd {fd} rel {err}", grad[i]);
    }
}

#[test]
fn single_and_mixed_hessian_matches_own_gradient() {
    let n = 8usize;
    let (x, y) = forrester(n);
    let packed = pack_col(&x);
    check_hess_exact::<SinglePrecision>(&packed, &y);
    check_hess_exact::<MixedPrecision<PromoteStorage>>(&packed, &y);
    check_hess_exact::<MixedPrecision<ReevaluateKernel>>(&packed, &y);
    check_hess_sgpr::<SinglePrecision>(&packed, &y);
    check_hess_sgpr::<MixedPrecision<PromoteStorage>>(&packed, &y);
    check_hess_sgpr::<MixedPrecision<ReevaluateKernel>>(&packed, &y);
}

fn check_hess_exact<P>(x: &[f64], y: &[f64])
where
    P: crate::precision::GpScalar,
{
    let mut fitted = factor_exact::<P>(x, y, 1.0, 0.1);
    let mut params = [0.0; 2];
    must(fitted.get_params(&mut params));
    let mut hess = [0.0; 4];
    must(fitted.hessian_into(&params, &mut hess));
    let step = 1.0e-3;
    for j in 0..2 {
        let mut up = params;
        let mut down = params;
        up[j] += step;
        down[j] -= step;
        let mut grad_up = [0.0; 2];
        let mut grad_down = [0.0; 2];
        must(fitted.set_params(&up));
        let _ = must(fitted.value_and_gradient_into(&up, &mut grad_up));
        must(fitted.set_params(&down));
        let _ = must(fitted.value_and_gradient_into(&down, &mut grad_down));
        for i in 0..2 {
            let fd = (grad_up[i] - grad_down[i]) / (2.0 * step);
            let analytic = hess[i * 2 + j];
            let scale = fd.abs().max(1.0);
            let err = (analytic - fd).abs() / scale;
            assert!(
                err < 5.0e-2,
                "exact hess[{i},{j}] {analytic} fd {fd} rel {err}"
            );
        }
    }
}

fn check_hess_sgpr<P>(x: &[f64], y: &[f64])
where
    P: crate::precision::GpScalar,
{
    let mut fitted = factor_sgpr::<P>(x, y, 1.0, 0.1);
    let mut params = [0.0; 2];
    must(fitted.get_params(&mut params));
    let mut hess = [0.0; 4];
    must(fitted.hessian_into(&params, &mut hess));
    let step = 1.0e-3;
    for j in 0..2 {
        let mut up = params;
        let mut down = params;
        up[j] += step;
        down[j] -= step;
        let mut grad_up = [0.0; 2];
        let mut grad_down = [0.0; 2];
        must(fitted.set_params(&up));
        let _ = must(fitted.value_and_gradient_into(&up, &mut grad_up));
        must(fitted.set_params(&down));
        let _ = must(fitted.value_and_gradient_into(&down, &mut grad_down));
        for i in 0..2 {
            let fd = (grad_up[i] - grad_down[i]) / (2.0 * step);
            let analytic = hess[i * 2 + j];
            let scale = fd.abs().max(1.0);
            let err = (analytic - fd).abs() / scale;
            assert!(
                err < 5.0e-2,
                "sgpr hess[{i},{j}] {analytic} fd {fd} rel {err}"
            );
        }
    }
}
