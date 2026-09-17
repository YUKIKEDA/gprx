//! Criterion benches for the Exact GPR path.
//!
//! P1A-18 adds `kernel_rbf` and `cholesky_alpha` on the fixed problem
//! (n = 256, d = 8, seed = 0). P1A-8 adds `predict_100`. P1A-10 adds
//! `mll_and_grad`. Do not mix those with a full L-BFGS fit.

#![allow(missing_docs, clippy::expect_used, clippy::unwrap_used)]

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt;
use faer::linalg::cholesky::llt::factor::LltRegularization;
use faer::{Mat, MatMut, MatRef, Par};
use gprx::kernel::{KernelSpec, RbfKernel, Triangle};
use gprx::{GaussianLikelihood, Gpr};

const N: usize = 256;
const D: usize = 8;
const M: usize = 100;
const SEED: u64 = 0;
const ELL: f64 = 1.0;
const NOISE: f64 = 0.1;

fn splitmix64(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^= z >> 31;
    (z >> 11) as f64 / ((1u64 << 53) as f64)
}

fn fill_column_major(n: usize, d: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    let mut x = vec![0.0; n * d];
    for col in 0..d {
        for row in 0..n {
            x[col * n + row] = splitmix64(&mut state);
        }
    }
    x
}

fn training_xy() -> (Vec<f64>, Vec<f64>) {
    let x = fill_column_major(N, D, SEED);
    let mut state = SEED ^ 0xA5A5_A5A5_A5A5_A5A5;
    let y: Vec<f64> = (0..N).map(|_| splitmix64(&mut state)).collect();
    (x, y)
}

fn pack_points(x: &[f64], n_rows: usize, n_cols: usize) -> Mat<f64> {
    Mat::from_fn(n_rows, n_cols, |row, col| x[col * n_rows + row])
}

fn fill_squared_euclidean(x: MatRef<'_, f64>, mut dist: MatMut<'_, f64>) {
    let n = x.nrows();
    let d = x.ncols();
    for col in 0..n {
        for row in col..n {
            let mut sum = 0.0;
            for dim in 0..d {
                let diff = x[(row, dim)] - x[(col, dim)];
                sum += diff * diff;
            }
            dist[(row, col)] = sum;
            dist[(col, row)] = sum;
        }
    }
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
    let (x, y) = training_xy();
    let x_mat = pack_points(&x, N, D);
    let mut dist = Mat::zeros(N, N);
    let mut a = Mat::zeros(N, N);
    let mut scratch = Mat::zeros(N, N);
    fill_squared_euclidean(x_mat.as_ref(), dist.as_mut());
    compiled
        .apply(dist.as_ref(), a.as_mut(), Triangle::Lower, scratch.as_mut())
        .expect("shape");
    add_noise_to_diag(a.as_mut(), NOISE);
    let rhs = Mat::from_fn(N, 1, |i, _| y[i]);
    (a, rhs)
}

fn fitted_model() -> (Gpr, Vec<f64>) {
    let kernel = KernelSpec::from(RbfKernel::new(ELL).expect("valid lengthscale"));
    let likelihood = GaussianLikelihood::new(NOISE).expect("valid noise");
    let mut gpr = Gpr::new(kernel, likelihood);
    let (x, y) = training_xy();
    gpr.fit(&x, N, D, &y).expect("training Cholesky");
    let xs = fill_column_major(M, D, SEED.wrapping_add(1));
    (gpr, xs)
}

fn kernel_rbf(c: &mut Criterion) {
    let compiled = KernelSpec::from(RbfKernel::new(ELL).expect("valid lengthscale")).compile();
    let (x, _) = training_xy();
    let x_mat = pack_points(&x, N, D);
    let mut dist = Mat::zeros(N, N);
    let mut k = Mat::zeros(N, N);
    let mut scratch = Mat::zeros(N, N);
    c.bench_function("kernel_rbf", |b| {
        b.iter(|| {
            fill_squared_euclidean(x_mat.as_ref(), dist.as_mut());
            compiled
                .apply(
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
    let (gpr, xs) = fitted_model();
    c.bench_function("predict_100", |b| {
        b.iter(|| {
            let pred = gpr.predict(std::hint::black_box(&xs), M, D);
            std::hint::black_box(pred)
        });
    });
}

fn mll_and_grad(c: &mut Criterion) {
    let (mut gpr, _) = fitted_model();
    let mut params = vec![0.0; gpr.num_params()];
    gpr.get_params(&mut params).expect("param length");
    let mut grad = vec![0.0; params.len()];
    c.bench_function("mll_and_grad", |b| {
        b.iter(|| {
            let value = gpr.value_and_gradient_into(
                std::hint::black_box(&params),
                std::hint::black_box(&mut grad),
            );
            std::hint::black_box(value)
        });
    });
}

criterion_group!(exact, kernel_rbf, cholesky_alpha, predict_100, mll_and_grad);
criterion_main!(exact);
