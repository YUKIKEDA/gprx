//! Allocation ratchet after Workspace setup (P1A-19 / P2-4).
//!
//! Counts heap allocations on the Exact GPR hot path once `fit` has already
//! sized the workspace. Caps may fall, never rise without an Issue. P2-4
//! requires zero on isotropic RBF (`value_and_gradient_into` and
//! `predict_into` after warmup). User kernels are excluded.

use gprx::kernel::{KernelSpec, RbfKernel};
use gprx::{FittedGpr, Fixed, GaussianLikelihood, Gpr, GprError, Prediction};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};
use std::alloc::System;

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

const N: usize = 256;
const D: usize = 8;
const M: usize = 100;
const SEED: u64 = 0;
const ELL: f64 = 1.0;
const NOISE: f64 = 0.1;

/// One `value_and_gradient_into` after a warmup call. Do not raise without an Issue.
const MAX_MLL_AND_GRAD_ALLOCS: usize = 0;

/// One `predict_into` of 100 points after a warmup call. Do not raise without an Issue.
const MAX_PREDICT_100_ALLOCS: usize = 0;

fn small_rng(seed: u64) -> SmallRng {
    SmallRng::seed_from_u64(seed)
}

fn open_unit(rng: &mut SmallRng) -> f64 {
    let u: f64 = rng.random();
    let eps = 1.0 / ((1u64 << 53) as f64);
    if u <= eps {
        eps
    } else if u >= 1.0 - eps {
        1.0 - eps
    } else {
        u
    }
}

fn fill_column_major(n: usize, d: usize, seed: u64) -> Vec<f64> {
    let mut rng = small_rng(seed);
    let mut x = vec![0.0; n * d];
    for col in 0..d {
        for row in 0..n {
            x[col * n + row] = open_unit(&mut rng);
        }
    }
    x
}

fn fitted_model() -> Result<(FittedGpr<Fixed>, Vec<f64>), GprError> {
    let kernel = KernelSpec::from(RbfKernel::new(ELL)?);
    let likelihood = GaussianLikelihood::new(NOISE)?;
    let x = fill_column_major(N, D, SEED);
    let mut rng = small_rng(SEED ^ 0xA5A5_A5A5_A5A5_A5A5);
    let y: Vec<f64> = (0..N).map(|_| open_unit(&mut rng)).collect();
    let gpr = Gpr::new(kernel, likelihood)
        .with_optimizer(Fixed)
        .factor(&x, N, D, &y)?;
    let xs = fill_column_major(M, D, SEED.wrapping_add(1));
    Ok((gpr, xs))
}

fn allocs_in(f: impl FnOnce()) -> usize {
    let region = Region::new(GLOBAL);
    f();
    let stats = region.change();
    stats.allocations + stats.reallocations
}

fn assert_alloc_cap(label: &str, count: usize, cap: usize) {
    eprintln!("{label}: allocations={count} cap={cap}");
    assert!(
        count <= cap,
        "{label}: allocations={count}, cap={cap}; lower the cap when allocs drop, do not raise it without an Issue"
    );
}

#[test]
fn mll_and_grad_allocs_after_workspace() {
    let (mut gpr, _) = fitted_model().expect("spd");
    let mut params = vec![0.0; gpr.num_params()];
    gpr.get_params(&mut params).expect("len");
    let mut grad = vec![0.0; params.len()];
    gpr.value_and_gradient_into(&params, &mut grad)
        .expect("warmup");
    let count = allocs_in(|| {
        gpr.value_and_gradient_into(&params, &mut grad)
            .expect("counted");
    });
    assert_alloc_cap("mll_and_grad", count, MAX_MLL_AND_GRAD_ALLOCS);
}

#[test]
fn predict_100_allocs_after_workspace() {
    let (mut gpr, xs) = fitted_model().expect("spd");
    let mut pred = Prediction::default();
    gpr.predict_into(&xs, M, D, &mut pred).expect("warmup");
    let count = allocs_in(|| {
        gpr.predict_into(&xs, M, D, &mut pred).expect("counted");
    });
    assert_alloc_cap("predict_100", count, MAX_PREDICT_100_ALLOCS);
}
