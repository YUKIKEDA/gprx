//! Allocation ratchet after Workspace setup (P1A-19).
//!
//! Counts heap allocations on the Exact GPR hot path once `fit` has already
//! sized the workspace. Caps may fall, never rise without an Issue. Phase 1a
//! does not require zero.

use gprx::kernel::{KernelSpec, RbfKernel};
use gprx::{FitOptions, GaussianLikelihood, Gpr, GprError};
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
const MAX_MLL_AND_GRAD_ALLOCS: usize = 16;

/// One `predict` of 100 points after a warmup call. Do not raise without an Issue.
const MAX_PREDICT_100_ALLOCS: usize = 9;

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

fn fitted_model() -> Result<(Gpr, Vec<f64>), GprError> {
    let kernel = KernelSpec::from(RbfKernel::new(ELL)?);
    let likelihood = GaussianLikelihood::new(NOISE)?;
    let mut gpr = Gpr::new(kernel, likelihood);
    let x = fill_column_major(N, D, SEED);
    let mut state = SEED ^ 0xA5A5_A5A5_A5A5_A5A5;
    let y: Vec<f64> = (0..N).map(|_| splitmix64(&mut state)).collect();
    gpr.fit_with(&x, N, D, &y, FitOptions::FIXED)?;
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
    let (gpr, xs) = fitted_model().expect("spd");
    gpr.predict(&xs, M, D).expect("warmup");
    let count = allocs_in(|| {
        gpr.predict(&xs, M, D).expect("counted");
    });
    assert_alloc_cap("predict_100", count, MAX_PREDICT_100_ALLOCS);
}
