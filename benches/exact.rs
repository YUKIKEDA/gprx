//! Criterion benches for the Exact GPR path.
//!
//! Fixed problems match the P1B-6 goldens, scaled to n = 256:
//! isotropic groups use 1-D Forrester with `StandardizeTarget`; ARD groups
//! use the 2-D weighted sphere on a 16×16 grid. `y` is the named function
//! plus N(0, 1) noise (seed 0), not an independent random series. Do not
//! mix one MLL+grad with a full L-BFGS fit.

#![allow(missing_docs, clippy::expect_used, clippy::unwrap_used)]

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt;
use faer::linalg::cholesky::llt::factor::LltRegularization;
use faer::{Mat, MatMut, Par};
use gprx::internals::{
    apply_from_ard_cache, fill_ard_squared_diff, fill_pairwise_sq_euclidean, grad_from_ard_cache,
};
use gprx::kernel::{KernelSpec, RbfArdKernel, RbfKernel, Triangle};
use gprx::transform::StandardizeTarget;
use gprx::{
    FastSimulatedAnnealing, FittedGpr, Fixed, GaussianLikelihood, Gpr, KernelExp, MixedPrecision,
    Prediction,
};

#[path = "../tests/common/problems.rs"]
mod problems;
#[path = "../src/rng.rs"]
#[allow(dead_code)]
mod rng;

/// Forrester `n = N`, seed 0.
fn forrester_xy() -> (Vec<f64>, Vec<f64>) {
    problems::forrester_xy(N, SEED, NOISE_STD)
}

fn forrester_query() -> Vec<f64> {
    problems::linspace(0.05, 0.95, M)
}

/// Weighted sphere on a `SPHERE_SIDE²` grid. Same seed as `sphere_bench_xy`
/// in `src/optimizer/lbfgs.rs`. Seed 0 walks a ridge on the memory pole
/// (`DistanceCachePolicy::Uncached`).
fn sphere_xy() -> (Vec<f64>, Vec<f64>) {
    problems::sphere_xy(SPHERE_SIDE, 9, NOISE_STD)
}

const N: usize = 256;
const D_ISO: usize = 1;
const D_ARD: usize = 2;
const SPHERE_SIDE: usize = 16;
const M: usize = 100;
const SEED: u64 = 0;
const ELL: f64 = 1.0;
const ELL_ARD: f64 = 4.0;
const NOISE: f64 = 0.1;
const NOISE_STD: f64 = 1.0;

fn pack_points(x: &[f64], n_rows: usize, n_cols: usize) -> Mat<f64> {
    Mat::from_fn(n_rows, n_cols, |row, col| x[col * n_rows + row])
}

fn add_noise_to_diag(mut k: MatMut<'_, f64>, noise: f64) {
    let n = k.nrows();
    for i in 0..n {
        k[(i, i)] += noise;
    }
}

fn chol_scratch(n: usize, rhs_ncols: usize) -> MemBuffer {
    let chol = llt::factor::cholesky_in_place_scratch::<f64>(n, Par::Seq, Default::default());
    let solve = llt::solve::solve_in_place_scratch::<f64>(n, rhs_ncols, Par::Seq);
    MemBuffer::new(chol.or(solve))
}

fn kernel_and_a() -> (Mat<f64>, Mat<f64>) {
    let compiled = KernelSpec::from(RbfKernel::new(ELL).expect("valid lengthscale")).compile();
    let (x, y) = forrester_xy();
    let x_mat = pack_points(&x, N, D_ISO);
    let mut dist = Mat::zeros(N, N);
    let mut a = Mat::zeros(N, N);
    let mut scratch = Mat::zeros(N, N);
    fill_pairwise_sq_euclidean(x_mat.as_ref(), dist.as_mut());
    compiled
        .apply::<gprx::Accurate>(dist.as_ref(), a.as_mut(), Triangle::Lower, scratch.as_mut())
        .expect("shape");
    add_noise_to_diag(a.as_mut(), NOISE);
    let rhs = Mat::from_fn(N, 1, |i, _| y[i]);
    (a, rhs)
}

fn fitted_model() -> (FittedGpr<Fixed>, Vec<f64>) {
    let kernel = KernelSpec::from(RbfKernel::new(ELL).expect("valid lengthscale"));
    let likelihood = GaussianLikelihood::new(NOISE).expect("valid noise");
    let (x, y) = forrester_xy();
    let gpr = Gpr::new(kernel, likelihood)
        .with_target_transform(StandardizeTarget::new())
        .with_optimizer(Fixed)
        .factor(&x, N, D_ISO, &y)
        .expect("training Cholesky");
    (gpr, forrester_query())
}

fn kernel_rbf(c: &mut Criterion) {
    let compiled = KernelSpec::from(RbfKernel::new(ELL).expect("valid lengthscale")).compile();
    let (x, _) = forrester_xy();
    let x_mat = pack_points(&x, N, D_ISO);
    let mut dist = Mat::zeros(N, N);
    let mut k = Mat::zeros(N, N);
    let mut scratch = Mat::zeros(N, N);
    c.bench_function("kernel_rbf", |b| {
        b.iter(|| {
            fill_pairwise_sq_euclidean(x_mat.as_ref(), dist.as_mut());
            compiled
                .apply::<gprx::Accurate>(
                    std::hint::black_box(dist.as_ref()),
                    k.as_mut(),
                    Triangle::Lower,
                    scratch.as_mut(),
                )
                .expect("shape");
            std::hint::black_box(&k);
        });
    });
}

fn kernel_exp(c: &mut Criterion) {
    let compiled = KernelSpec::from(RbfKernel::new(ELL).expect("valid lengthscale")).compile();
    let (x, _) = forrester_xy();
    let x_mat = pack_points(&x, N, D_ISO);
    let mut dist = Mat::zeros(N, N);
    fill_pairwise_sq_euclidean(x_mat.as_ref(), dist.as_mut());
    let mut k = Mat::zeros(N, N);
    let mut dk = Mat::zeros(N, N);
    let mut scratch = Mat::zeros(N, N);
    let mut group = c.benchmark_group("kernel_exp");
    group.bench_function("accurate", |b| {
        b.iter(|| {
            compiled
                .apply::<gprx::Accurate>(
                    std::hint::black_box(dist.as_ref()),
                    k.as_mut(),
                    Triangle::Lower,
                    scratch.as_mut(),
                )
                .expect("apply");
            compiled
                .grad::<gprx::Accurate>(
                    dist.as_ref(),
                    dk.as_mut(),
                    0,
                    Triangle::Lower,
                    scratch.as_mut(),
                )
                .expect("grad");
            std::hint::black_box(k[(0, 0)] + dk[(1, 0)])
        });
    });
    group.bench_function("fast_approx", |b| {
        b.iter(|| {
            compiled
                .apply::<gprx::FastApprox>(
                    std::hint::black_box(dist.as_ref()),
                    k.as_mut(),
                    Triangle::Lower,
                    scratch.as_mut(),
                )
                .expect("apply");
            compiled
                .grad::<gprx::FastApprox>(
                    dist.as_ref(),
                    dk.as_mut(),
                    0,
                    Triangle::Lower,
                    scratch.as_mut(),
                )
                .expect("grad");
            std::hint::black_box(k[(0, 0)] + dk[(1, 0)])
        });
    });
    group.finish();
}

fn kernel_exp_ard(c: &mut Criterion) {
    let ells = [ELL_ARD; D_ARD];
    let compiled = KernelSpec::from(RbfArdKernel::new(&ells).expect("valid lengthscale")).compile();
    let (x, _) = sphere_xy();
    let x_mat = pack_points(&x, N, D_ARD);
    let mut cache = Mat::zeros(N, N * D_ARD);
    fill_ard_squared_diff(x_mat.as_ref(), cache.as_mut(), &mut []);
    let mut k = Mat::zeros(N, N);
    let mut dk = Mat::zeros(N, N);
    let mut scratch = Mat::zeros(N, N);
    let n_theta = compiled.num_params();
    let mut group = c.benchmark_group("kernel_exp_ard");
    group.bench_function("accurate", |b| {
        b.iter(|| {
            apply_from_ard_cache::<gprx::Accurate>(
                &compiled,
                std::hint::black_box(cache.as_ref()),
                x_mat.as_ref(),
                k.as_mut(),
                Triangle::Lower,
                scratch.as_mut(),
            )
            .expect("apply");
            for idx in 0..n_theta {
                grad_from_ard_cache::<gprx::Accurate>(
                    &compiled,
                    cache.as_ref(),
                    x_mat.as_ref(),
                    dk.as_mut(),
                    idx,
                    Triangle::Lower,
                    scratch.as_mut(),
                )
                .expect("grad");
            }
            std::hint::black_box(k[(0, 0)] + dk[(1, 0)])
        });
    });
    group.bench_function("fast_approx", |b| {
        b.iter(|| {
            apply_from_ard_cache::<gprx::FastApprox>(
                &compiled,
                std::hint::black_box(cache.as_ref()),
                x_mat.as_ref(),
                k.as_mut(),
                Triangle::Lower,
                scratch.as_mut(),
            )
            .expect("apply");
            for idx in 0..n_theta {
                grad_from_ard_cache::<gprx::FastApprox>(
                    &compiled,
                    cache.as_ref(),
                    x_mat.as_ref(),
                    dk.as_mut(),
                    idx,
                    Triangle::Lower,
                    scratch.as_mut(),
                )
                .expect("grad");
            }
            std::hint::black_box(k[(0, 0)] + dk[(1, 0)])
        });
    });
    group.finish();
}

fn cholesky_alpha(c: &mut Criterion) {
    let (a_template, rhs_template) = kernel_and_a();
    let mut faer_scratch = chol_scratch(N, 1);
    c.bench_function("cholesky_alpha", |b| {
        b.iter_batched(
            || (a_template.clone(), rhs_template.clone()),
            |(mut a, mut rhs)| {
                let regularization = LltRegularization {
                    dynamic_regularization_delta: 0.0,
                    dynamic_regularization_epsilon: 0.0,
                };
                {
                    let stack = MemStack::new(&mut faer_scratch);
                    llt::factor::cholesky_in_place(
                        a.as_mut(),
                        regularization,
                        Par::Seq,
                        stack,
                        Default::default(),
                    )
                    .expect("spd");
                }
                let stack = MemStack::new(&mut faer_scratch);
                llt::solve::solve_in_place(a.as_ref(), rhs.as_mut(), Par::Seq, stack);
                std::hint::black_box(rhs)
            },
            BatchSize::LargeInput,
        );
    });
}

fn predict_100(c: &mut Criterion) {
    let (mut gpr, xs) = fitted_model();
    let mut pred = Prediction::default();
    gpr.predict_into(&xs, M, D_ISO, &mut pred).expect("warmup");
    c.bench_function("predict_100", |b| {
        b.iter(|| {
            gpr.predict_into(
                std::hint::black_box(&xs),
                M,
                D_ISO,
                std::hint::black_box(&mut pred),
            )
            .expect("predict");
            std::hint::black_box(pred.mean[0] + pred.variance[0])
        });
    });
}

type MixedFitted = FittedGpr<Fixed, MixedPrecision>;

/// Same problem as [`fitted_model`] at [`MixedPrecision`] (f32 factor, f64 `α`).
fn fitted_mixed() -> (MixedFitted, Vec<f64>) {
    let kernel = KernelSpec::from(RbfKernel::new(ELL).expect("valid lengthscale"));
    let likelihood = GaussianLikelihood::new(NOISE).expect("valid noise");
    let (x, y) = forrester_xy();
    let gpr = Gpr::new(kernel, likelihood)
        .with_target_transform(StandardizeTarget::new())
        .with_optimizer(Fixed)
        .with_precision::<MixedPrecision>()
        .factor(&x, N, D_ISO, &y)
        .map_err(|(_, e)| e)
        .expect("training Cholesky");
    (gpr, forrester_query())
}

/// `predict_100` at [`MixedPrecision`]. Predict reads the `α` published at
/// factor time; it does not refine again (R3-1).
fn predict_100_mixed(c: &mut Criterion) {
    let (mut gpr, xs) = fitted_mixed();
    let mut pred = Prediction::default();
    gpr.predict_into(&xs, M, D_ISO, &mut pred).expect("warmup");
    c.bench_function("predict_100_mixed", |b| {
        b.iter(|| {
            gpr.predict_into(
                std::hint::black_box(&xs),
                M,
                D_ISO,
                std::hint::black_box(&mut pred),
            )
            .expect("predict");
            std::hint::black_box(pred.mean[0] + pred.variance[0])
        });
    });
}

fn fitted_fast() -> FittedGpr<Fixed> {
    let kernel = KernelSpec::from(RbfKernel::new(ELL).expect("valid lengthscale"));
    let likelihood = GaussianLikelihood::new(NOISE).expect("valid noise");
    let (x, y) = forrester_xy();
    Gpr::new(kernel, likelihood)
        .with_target_transform(StandardizeTarget::new())
        .with_math(KernelExp::FastApprox)
        .with_optimizer(Fixed)
        .factor(&x, N, D_ISO, &y)
        .expect("training Cholesky")
}

macro_rules! bench_mll {
    ($group:expr, $name:expr, $gpr:expr) => {{
        let mut gpr = $gpr;
        let mut params = vec![0.0; gpr.num_params()];
        gpr.get_params(&mut params).expect("param length");
        let mut grad = vec![0.0; params.len()];
        $group.bench_function($name, |b| {
            b.iter(|| {
                let value = gpr.value_and_gradient_into(
                    std::hint::black_box(&params),
                    std::hint::black_box(&mut grad),
                );
                std::hint::black_box(value)
            });
        });
    }};
}

fn mll_and_grad(c: &mut Criterion) {
    let mut group = c.benchmark_group("mll_and_grad");
    let (accurate, _) = fitted_model();
    bench_mll!(group, "accurate", accurate);
    bench_mll!(group, "fast_approx", fitted_fast());
    group.finish();
}

fn fit_lbfgs(c: &mut Criterion) {
    let (x, y) = forrester_xy();
    let mut group = c.benchmark_group("fit_lbfgs");
    group.sample_size(10);
    group.bench_function("fit_lbfgs", |b| {
        b.iter_batched(
            || {
                let kernel = KernelSpec::from(RbfKernel::new(ELL).expect("valid lengthscale"));
                let likelihood = GaussianLikelihood::new(NOISE).expect("valid noise");
                (
                    Gpr::new(kernel, likelihood).with_target_transform(StandardizeTarget::new()),
                    x.clone(),
                    y.clone(),
                )
            },
            |(gpr, x, y)| {
                let fitted = gpr
                    .fit(std::hint::black_box(&x), N, D_ISO, std::hint::black_box(&y))
                    .expect("lbfgs");
                std::hint::black_box(fitted)
            },
            BatchSize::LargeInput,
        );
    });
    group.finish();
}

/// FSA on a sum of two RBF leaves: every coordinate step rebuilds only the
/// leaf it touches (`Optimizer::USES_CHANGE_INDICES`, R4-5 / #243).
fn fit_fsa(c: &mut Criterion) {
    let (x, y) = forrester_xy();
    let mut group = c.benchmark_group("fit_fsa");
    group.sample_size(10);
    group.bench_function("sum_of_two", |b| {
        b.iter_batched(
            || {
                let kernel = KernelSpec::from(RbfKernel::new(ELL).expect("valid lengthscale"))
                    + KernelSpec::from(RbfKernel::new(2.0 * ELL).expect("valid lengthscale"));
                let likelihood = GaussianLikelihood::new(NOISE).expect("valid noise");
                (
                    Gpr::new(kernel, likelihood)
                        .with_target_transform(StandardizeTarget::new())
                        .with_optimizer(
                            FastSimulatedAnnealing::new()
                                .with_max_iterations(10)
                                .with_seed(7),
                        ),
                    x.clone(),
                    y.clone(),
                )
            },
            |(gpr, x, y)| {
                let fitted = gpr
                    .fit(std::hint::black_box(&x), N, D_ISO, std::hint::black_box(&y))
                    .expect("fsa");
                std::hint::black_box(fitted)
            },
            BatchSize::LargeInput,
        );
    });
    group.finish();
}

fn fitted_ard_fast() -> FittedGpr<Fixed> {
    let ells = [ELL_ARD; D_ARD];
    let kernel = KernelSpec::from(RbfArdKernel::new(&ells).expect("valid lengthscale"));
    let likelihood = GaussianLikelihood::new(NOISE).expect("valid noise");
    let (x, y) = sphere_xy();
    Gpr::new(kernel, likelihood)
        .with_prefer_speed()
        .with_target_transform(StandardizeTarget::new())
        .with_math(KernelExp::FastApprox)
        .with_optimizer(Fixed)
        .factor(&x, N, D_ARD, &y)
        .expect("training Cholesky")
}

fn fitted_ard_speed() -> FittedGpr<Fixed> {
    let ells = [ELL_ARD; D_ARD];
    let kernel = KernelSpec::from(RbfArdKernel::new(&ells).expect("valid lengthscale"));
    let likelihood = GaussianLikelihood::new(NOISE).expect("valid noise");
    let (x, y) = sphere_xy();
    Gpr::new(kernel, likelihood)
        .with_prefer_speed()
        .with_target_transform(StandardizeTarget::new())
        .with_optimizer(Fixed)
        .factor(&x, N, D_ARD, &y)
        .expect("training Cholesky")
}

fn fitted_ard_memory() -> FittedGpr<Fixed> {
    let ells = [ELL_ARD; D_ARD];
    let kernel = KernelSpec::from(RbfArdKernel::new(&ells).expect("valid lengthscale"));
    let likelihood = GaussianLikelihood::new(NOISE).expect("valid noise");
    let (x, y) = sphere_xy();
    Gpr::new(kernel, likelihood)
        .with_prefer_memory()
        .with_target_transform(StandardizeTarget::new())
        .with_optimizer(Fixed)
        .factor(&x, N, D_ARD, &y)
        .expect("training Cholesky")
}

fn bench_mll_ard<F>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    name: &str,
    fitted: F,
) where
    F: FnOnce() -> FittedGpr<Fixed>,
{
    let mut gpr = fitted();
    let mut params = vec![0.0; gpr.num_params()];
    gpr.get_params(&mut params).expect("param length");
    let mut grad = vec![0.0; params.len()];
    group.bench_function(name, move |b| {
        b.iter(|| {
            let value = gpr.value_and_gradient_into(
                std::hint::black_box(&params),
                std::hint::black_box(&mut grad),
            );
            std::hint::black_box(value)
        });
    });
}

fn mll_and_grad_ard(c: &mut Criterion) {
    let mut group = c.benchmark_group("mll_and_grad_ard");
    bench_mll_ard(&mut group, "always", fitted_ard_speed);
    bench_mll!(group, "fast_approx", fitted_ard_fast());
    let mut gpr = fitted_ard_memory();
    let mut params = vec![0.0; gpr.num_params()];
    gpr.get_params(&mut params).expect("param length");
    let mut grad = vec![0.0; params.len()];
    group.bench_function("never", move |b| {
        b.iter(|| {
            let value = gpr.value_and_gradient_into(
                std::hint::black_box(&params),
                std::hint::black_box(&mut grad),
            );
            std::hint::black_box(value)
        });
    });
    group.finish();
}

fn fit_lbfgs_ard(c: &mut Criterion) {
    let (x, y) = sphere_xy();
    let ells = [ELL_ARD; D_ARD];
    let mut group = c.benchmark_group("fit_lbfgs_ard");
    group.sample_size(10);
    group.bench_function("always", |b| {
        b.iter_batched(
            || {
                let kernel = KernelSpec::from(RbfArdKernel::new(&ells).expect("valid lengthscale"));
                let likelihood = GaussianLikelihood::new(NOISE).expect("valid noise");
                (
                    Gpr::new(kernel, likelihood)
                        .with_prefer_speed()
                        .with_target_transform(StandardizeTarget::new()),
                    x.clone(),
                    y.clone(),
                )
            },
            |(gpr, x, y)| {
                let fitted = gpr
                    .fit(std::hint::black_box(&x), N, D_ARD, std::hint::black_box(&y))
                    .expect("lbfgs");
                std::hint::black_box(fitted)
            },
            BatchSize::LargeInput,
        );
    });
    group.bench_function("never", |b| {
        b.iter_batched(
            || {
                let kernel = KernelSpec::from(RbfArdKernel::new(&ells).expect("valid lengthscale"));
                let likelihood = GaussianLikelihood::new(NOISE).expect("valid noise");
                (
                    Gpr::new(kernel, likelihood)
                        .with_prefer_memory()
                        .with_target_transform(StandardizeTarget::new()),
                    x.clone(),
                    y.clone(),
                )
            },
            |(gpr, x, y)| {
                let fitted = gpr
                    .fit(std::hint::black_box(&x), N, D_ARD, std::hint::black_box(&y))
                    .expect("lbfgs");
                std::hint::black_box(fitted)
            },
            BatchSize::LargeInput,
        );
    });
    group.finish();
}

criterion_group!(
    exact,
    kernel_rbf,
    kernel_exp,
    kernel_exp_ard,
    cholesky_alpha,
    predict_100,
    predict_100_mixed,
    mll_and_grad,
    fit_lbfgs,
    fit_fsa,
    mll_and_grad_ard,
    fit_lbfgs_ard
);
criterion_main!(exact);
