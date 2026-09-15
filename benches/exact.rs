//! Criterion benches for the Exact GPR path.
//!
//! P1A-8 adds `predict_100`. P1A-10 adds `mll_and_grad`. P1A-18 adds
//! `kernel_rbf` and `cholesky_alpha` on the same fixed problem.

#![allow(missing_docs, clippy::expect_used, clippy::unwrap_used)]

use criterion::{Criterion, criterion_group, criterion_main};
use gprx::kernel::{KernelSpec, RbfKernel};
use gprx::{GaussianLikelihood, Gpr};

const N: usize = 256;
const D: usize = 8;
const M: usize = 100;
const SEED: u64 = 0;

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

fn fitted_model() -> (Gpr, Vec<f64>) {
    let kernel = KernelSpec::from(RbfKernel::new(1.0).expect("valid lengthscale"));
    let likelihood = GaussianLikelihood::new(0.1).expect("valid noise");
    let mut gpr = Gpr::new(kernel, likelihood);
    let x = fill_column_major(N, D, SEED);
    let mut state = SEED ^ 0xA5A5_A5A5_A5A5_A5A5;
    let y: Vec<f64> = (0..N).map(|_| splitmix64(&mut state)).collect();
    gpr.fit(&x, N, D, &y).expect("training Cholesky");
    let xs = fill_column_major(M, D, SEED.wrapping_add(1));
    (gpr, xs)
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

criterion_group!(exact, predict_100, mll_and_grad);
criterion_main!(exact);
