//! Allocation ratchet after Workspace setup (P1A-19 / P2-4).
//!
//! Counts heap allocations on the Exact GPR hot path once `fit` has already
//! sized the workspace. Caps may fall, never rise without an Issue. P2-4
//! requires zero on isotropic RBF (`value_and_gradient_into` and
//! `predict_into` after warmup). User kernels are excluded. P2B-22 (#143)
//! pins `RAYON_NUM_THREADS=1` in this binary so faer `Par::rayon(1)` does
//! not allocate worker scratch that a multi-thread pool would.

mod common;
use common::rng::{open_unit, small_rng};
use gprx::Adam;
use gprx::kernel::{ConstantKernel, KernelSpec, RbfArdKernel, RbfKernel};
use gprx::{
    DoublePrecision, FittedGpr, Fixed, GaussianLikelihood, GpScalar, Gpr, GprError, KernelExp,
    MixedPrecision, Prediction, ReevaluateKernel, Sgpr, SinglePrecision, Svgp,
};
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};
use std::alloc::System;
use std::num::{NonZeroU64, NonZeroUsize};
use std::sync::{Mutex, OnceLock};

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

/// One coordinate step (`value_at_changes`) of an incremental fit on a sum
/// of two leaves, after a warmup step (R4-5 / #243, #270, #272). Do not
/// raise without an Issue.
const MAX_LEAF_STEP_ALLOCS: usize = 0;

/// One `value_and_gradient_into` on composite kernels after a warmup call,
/// by kernel (#270, #272). Do not raise without an Issue.
const MAX_COMPOSITE_MLL_AND_GRAD_ALLOCS: [(&str, usize); 6] = [
    ("sum_with_ard", 0),
    ("product", 0),
    ("ard", 0),
    ("sum", 0),
    ("nested", 0),
    ("nested_mixed", 0),
];

/// One `hessian_into` after a warmup call, by kernel: the isotropic RBF leaf,
/// then the composite kernels (#272). Do not raise without an Issue.
const MAX_HESSIAN_ALLOCS: [(&str, usize); 7] = [
    ("rbf", 0),
    ("sum_with_ard", 0),
    ("product", 0),
    ("ard", 0),
    ("sum", 0),
    ("nested", 0),
    ("nested_mixed", 0),
];

/// Training points of the Hessian ratchet: it solves `n×n` systems per
/// parameter pair, and its allocations do not depend on `n`.
const N_HESSIAN: usize = 32;

/// One `predict_into` of 100 points on composite kernels after a warmup call,
/// by kernel (#272). Do not raise without an Issue.
const MAX_COMPOSITE_PREDICT_100_ALLOCS: [(&str, usize); 6] = [
    ("sum_with_ard", 0),
    ("product", 0),
    ("ard", 0),
    ("sum", 0),
    ("nested", 0),
    ("nested_mixed", 0),
];

/// One `predict_into` of 100 points after a warmup call. Do not raise without an Issue.
const MAX_PREDICT_100_ALLOCS: usize = 0;

/// Bytes one [`MixedPrecision`] `predict_into` of 100 points may allocate after
/// a warmup call: below one `n×n` `f32` matrix, so predict never rebuilds or
/// refines the training system (R3-1 / #236). Do not raise without an Issue.
const MAX_MIXED_PREDICT_100_BYTES: usize = N * N * std::mem::size_of::<f32>();

/// Inducing points of the sparse ratchets.
const M_SPARSE: usize = 32;

/// Allocations of one call on the sparse models after a warmup call, by
/// path (R5-1d / #246). The sparse fits return new matrices for their
/// factors, so these are not zero; the kernel scratch of the `&mut self`
/// paths is kept on the model between calls. Do not raise without an Issue.
const MAX_SPARSE_ALLOCS: [(&str, usize); 5] = [
    ("sgpr_mll_and_grad", 19),
    ("sgpr_hessian", 111),
    ("online_sgpr_insert", 8),
    ("online_sgpr_insert_nested", 16),
    ("svgp_mll_and_grad", 20),
];

/// One sparse `predict_into` of 100 points after a warmup call, by model,
/// kernel, and precision (R5-5 / #283). Do not raise without an Issue.
const MAX_SPARSE_PREDICT_100_ALLOCS: usize = 0;

/// Bytes one [`MixedPrecision`] SVGP `predict_into` of 100 points may
/// allocate after a warmup call: the mixed mean refines `L_mm⁻¹ k_*` in
/// `f64` for each query (#39), and the refinement loop keeps three `f64`
/// vectors of length `m` per query. The `f64` reference (`L_mm`, `K(Z, X*)`)
/// is built once per call into buffers the model keeps. The other sparse
/// paths allocate nothing. Do not raise without an Issue.
const MAX_SVGP_MIXED_PREDICT_100_BYTES: usize = M * 3 * M_SPARSE * std::mem::size_of::<f64>();

fn ensure_one_rayon_worker() {
    static INIT: OnceLock<()> = OnceLock::new();
    INIT.get_or_init(|| {
        // SAFETY: this integration test has not started the Rayon pool yet.
        // One worker keeps faer at `Par::rayon(1)` so the 0-alloc ratchet
        // still applies after P2B-22 (#143).
        unsafe {
            std::env::set_var("RAYON_NUM_THREADS", "1");
        }
    });
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
    ensure_one_rayon_worker();
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

fn alloc_lock() -> std::sync::MutexGuard<'static, ()> {
    // `stats_alloc` counts the process, not the calling thread.
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
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
    let _guard = alloc_lock();
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
fn fast_approx_mll_and_grad_allocs_after_workspace() {
    let _guard = alloc_lock();
    ensure_one_rayon_worker();
    let kernel = KernelSpec::from(RbfKernel::new(ELL).expect("ell"));
    let likelihood = GaussianLikelihood::new(NOISE).expect("noise");
    let x = fill_column_major(N, D, SEED);
    let mut rng = small_rng(SEED ^ 0xA5A5_A5A5_A5A5_A5A5);
    let y: Vec<f64> = (0..N).map(|_| open_unit(&mut rng)).collect();
    let mut gpr = Gpr::new(kernel, likelihood)
        .with_math(KernelExp::FastApprox)
        .with_optimizer(Fixed)
        .factor(&x, N, D, &y)
        .expect("spd");
    let mut params = vec![0.0; gpr.num_params()];
    gpr.get_params(&mut params).expect("len");
    let mut grad = vec![0.0; params.len()];
    gpr.value_and_gradient_into(&params, &mut grad)
        .expect("warmup");
    let count = allocs_in(|| {
        gpr.value_and_gradient_into(&params, &mut grad)
            .expect("counted");
    });
    assert_alloc_cap("fast_mll_and_grad", count, MAX_MLL_AND_GRAD_ALLOCS);
    let _typed: FittedGpr<Fixed> = gpr;
}

#[test]
fn predict_100_allocs_after_workspace() {
    let _guard = alloc_lock();
    let (mut gpr, xs) = fitted_model().expect("spd");
    let mut pred = Prediction::default();
    gpr.predict_into(&xs, M, D, &mut pred).expect("warmup");
    let count = allocs_in(|| {
        gpr.predict_into(&xs, M, D, &mut pred).expect("counted");
    });
    assert_alloc_cap("predict_100", count, MAX_PREDICT_100_ALLOCS);
}

type MixedFitted<R> = FittedGpr<Fixed, MixedPrecision<R>>;

fn fitted_mixed<R>() -> Result<(MixedFitted<R>, Vec<f64>), GprError>
where
    MixedPrecision<R>: gprx::GpScalar,
    R: gprx::ResidualFormula,
{
    ensure_one_rayon_worker();
    let kernel = KernelSpec::from(RbfKernel::new(ELL)?);
    let likelihood = GaussianLikelihood::new(NOISE)?;
    let x = fill_column_major(N, D, SEED);
    let mut rng = small_rng(SEED ^ 0xA5A5_A5A5_A5A5_A5A5);
    let y: Vec<f64> = (0..N).map(|_| open_unit(&mut rng)).collect();
    let gpr = Gpr::new(kernel, likelihood)
        .with_optimizer(Fixed)
        .with_precision::<MixedPrecision<R>>()
        .factor(&x, N, D, &y)
        .map_err(|(_, e)| e)?;
    let xs = fill_column_major(M, D, SEED.wrapping_add(1));
    Ok((gpr, xs))
}

fn bytes_in(f: impl FnOnce()) -> usize {
    let region = Region::new(GLOBAL);
    f();
    let stats = region.change();
    stats.bytes_allocated + stats.bytes_reallocated.max(0) as usize
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn assert_mixed_predict_bytes<R>(label: &str)
where
    MixedPrecision<R>: gprx::GpScalar,
    R: gprx::ResidualFormula,
{
    let _guard = alloc_lock();
    let (mut gpr, xs) = fitted_mixed::<R>().expect("spd");
    let mut pred = Prediction::default();
    gpr.predict_into(&xs, M, D, &mut pred).expect("warmup");
    let bytes = bytes_in(|| {
        gpr.predict_into(&xs, M, D, &mut pred).expect("counted");
    });
    eprintln!("{label}: bytes={bytes} cap={MAX_MIXED_PREDICT_100_BYTES}");
    assert!(
        bytes < MAX_MIXED_PREDICT_100_BYTES,
        "{label}: bytes={bytes}, cap={MAX_MIXED_PREDICT_100_BYTES}; predict must not rebuild the n×n system"
    );
}

#[test]
fn mixed_promote_predict_100_bytes_after_workspace() {
    assert_mixed_predict_bytes::<gprx::PromoteStorage>("mixed_promote_predict_100");
}

#[test]
fn mixed_reevaluate_predict_100_bytes_after_workspace() {
    assert_mixed_predict_bytes::<ReevaluateKernel>("mixed_reevaluate_predict_100");
}

/// Optimizer that measures one incremental coordinate step inside `minimize`.
#[derive(Clone, Copy, Debug)]
struct LeafStepProbe;

static LEAF_STEP_ALLOCS: Mutex<Option<usize>> = Mutex::new(None);

impl<P: gprx::Objective> gprx::Optimizer<P> for LeafStepProbe {
    const USES_CHANGE_INDICES: bool = true;

    fn minimize(&self, objective: &mut P, init: &[f64]) -> Result<gprx::OptResult, GprError> {
        let mut step = init.to_vec();
        let value = objective.value(&step)?;
        step[0] += 0.01;
        objective.value_at_changes(&step, &[0])?;
        step[0] += 0.01;
        let mut result = Ok(0.0);
        let count = allocs_in(|| {
            result = objective.value_at_changes(&step, &[0]);
        });
        result?;
        if let Ok(mut slot) = LEAF_STEP_ALLOCS.lock() {
            *slot = Some(count);
        }
        Ok(gprx::OptResult {
            params: init.to_vec(),
            value,
            iterations: 0,
        })
    }
}

#[test]
fn incremental_leaf_step_allocs_after_warmup() {
    let _guard = alloc_lock();
    ensure_one_rayon_worker();
    let kernel = KernelSpec::from(RbfKernel::new(ELL).expect("ell"))
        + KernelSpec::from(RbfKernel::new(2.0 * ELL).expect("ell"));
    let likelihood = GaussianLikelihood::new(NOISE).expect("noise");
    let x = fill_column_major(N, D, SEED);
    let mut rng = small_rng(SEED ^ 0xA5A5_A5A5_A5A5_A5A5);
    let y: Vec<f64> = (0..N).map(|_| open_unit(&mut rng)).collect();
    Gpr::new(kernel, likelihood)
        .with_optimizer(LeafStepProbe)
        .fit(&x, N, D, &y)
        .map_err(|(_, e)| e)
        .expect("fit");
    let count = LEAF_STEP_ALLOCS.lock().expect("lock").expect("probe ran");
    assert_alloc_cap("leaf_step", count, MAX_LEAF_STEP_ALLOCS);
}

/// Flat sums and products, one ARD leaf, and a sum / product nested in
/// another in distance mode and in mixed coordinate mode.
#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn composite_kernels() -> [(&'static str, KernelSpec); 6] {
    let rbf = |ell: f64| KernelSpec::from(RbfKernel::new(ell).expect("ell"));
    let ard = || KernelSpec::from(RbfArdKernel::new(&[ELL; D]).expect("ell"));
    let constant = |value: f64| KernelSpec::from(ConstantKernel::new(value).expect("constant"));
    [
        ("sum_with_ard", rbf(ELL) + ard()),
        ("product", constant(1.5) * rbf(ELL)),
        ("ard", ard()),
        ("sum", rbf(ELL) + rbf(2.0 * ELL)),
        (
            "nested",
            (rbf(ELL) + rbf(2.0 * ELL)) * (constant(1.5) * rbf(3.0 * ELL) + constant(0.5)),
        ),
        (
            "nested_mixed",
            (rbf(ELL) + ard()) * (constant(1.5) * rbf(3.0 * ELL) + constant(0.5)),
        ),
    ]
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn fitted_with(kernel: KernelSpec, n: usize) -> FittedGpr<Fixed> {
    ensure_one_rayon_worker();
    let likelihood = GaussianLikelihood::new(NOISE).expect("noise");
    let x = fill_column_major(n, D, SEED);
    let mut rng = small_rng(SEED ^ 0xA5A5_A5A5_A5A5_A5A5);
    let y: Vec<f64> = (0..n).map(|_| open_unit(&mut rng)).collect();
    Gpr::new(kernel, likelihood)
        .with_optimizer(Fixed)
        .factor(&x, n, D, &y)
        .map_err(|(_, e)| e)
        .expect("spd")
}

/// `value_and_gradient_into` on composite kernels after a warmup call.
#[test]
fn composite_mll_and_grad_allocs_after_workspace() {
    let _guard = alloc_lock();
    for ((label, cap), (name, kernel)) in MAX_COMPOSITE_MLL_AND_GRAD_ALLOCS
        .into_iter()
        .zip(composite_kernels())
    {
        assert_eq!(label, name);
        let mut gpr = fitted_with(kernel, N);
        let mut params = vec![0.0; gpr.num_params()];
        gpr.get_params(&mut params).expect("len");
        let mut grad = vec![0.0; params.len()];
        gpr.value_and_gradient_into(&params, &mut grad)
            .expect("warmup");
        let count = allocs_in(|| {
            gpr.value_and_gradient_into(&params, &mut grad)
                .expect("counted");
        });
        assert_alloc_cap(label, count, cap);
    }
}

/// `hessian_into` on the RBF leaf and the composite kernels after a warmup call.
#[test]
fn hessian_allocs_after_warmup() {
    let _guard = alloc_lock();
    let kernels = std::iter::once(("rbf", KernelSpec::from(RbfKernel::new(ELL).expect("ell"))))
        .chain(composite_kernels());
    for ((label, cap), (name, kernel)) in MAX_HESSIAN_ALLOCS.into_iter().zip(kernels) {
        assert_eq!(label, name);
        let mut gpr = fitted_with(kernel, N_HESSIAN);
        let mut params = vec![0.0; gpr.num_params()];
        gpr.get_params(&mut params).expect("len");
        let mut hess = vec![0.0; params.len() * params.len()];
        gpr.hessian_into(&params, &mut hess).expect("warmup");
        let count = allocs_in(|| {
            gpr.hessian_into(&params, &mut hess).expect("counted");
        });
        assert_alloc_cap(&format!("hessian_{label}"), count, cap);
    }
}

/// `predict_into` of 100 points on the composite kernels after a warmup call.
#[test]
fn composite_predict_100_allocs_after_workspace() {
    let _guard = alloc_lock();
    let xs = fill_column_major(M, D, SEED.wrapping_add(1));
    for ((label, cap), (name, kernel)) in MAX_COMPOSITE_PREDICT_100_ALLOCS
        .into_iter()
        .zip(composite_kernels())
    {
        assert_eq!(label, name);
        let mut gpr = fitted_with(kernel, N);
        let mut pred = Prediction::default();
        gpr.predict_into(&xs, M, D, &mut pred).expect("warmup");
        let count = allocs_in(|| {
            gpr.predict_into(&xs, M, D, &mut pred).expect("counted");
        });
        assert_alloc_cap(&format!("predict_100_{label}"), count, cap);
    }
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn sparse_allocs(label: &str, kernel: KernelSpec) -> Vec<(String, usize)> {
    ensure_one_rayon_worker();
    let x = fill_column_major(N, D, SEED);
    let z = fill_column_major(M_SPARSE, D, SEED.wrapping_add(2));
    let mut rng = small_rng(SEED ^ 0xA5A5_A5A5_A5A5_A5A5);
    let y: Vec<f64> = (0..N).map(|_| open_unit(&mut rng)).collect();
    let x_new: Vec<f64> = (0..D).map(|dim| 0.1 * dim as f64).collect();
    let likelihood = GaussianLikelihood::new(NOISE).expect("noise");
    let mut fitted = Sgpr::new(kernel.clone(), likelihood)
        .with_optimizer(Fixed)
        .factor(&x, N, D, &y, &z, M_SPARSE)
        .map_err(|(_, e)| e)
        .expect("spd");
    let mut out = Vec::new();
    if label == "rbf" {
        let mut params = vec![0.0; fitted.num_params()];
        fitted.get_params(&mut params).expect("len");
        let mut grad = vec![0.0; params.len()];
        let mut hess = vec![0.0; params.len() * params.len()];
        fitted
            .value_and_gradient_into(&params, &mut grad)
            .expect("warmup");
        let count = allocs_in(|| {
            fitted
                .value_and_gradient_into(&params, &mut grad)
                .expect("counted");
        });
        out.push(("sgpr_mll_and_grad".to_owned(), count));
        fitted.hessian_into(&params, &mut hess).expect("warmup");
        let count = allocs_in(|| {
            fitted.hessian_into(&params, &mut hess).expect("counted");
        });
        out.push(("sgpr_hessian".to_owned(), count));
    }
    let mut online = fitted.into_online();
    online.insert(&x_new, 0.5).expect("warmup");
    let count = allocs_in(|| {
        online.insert(&x_new, 0.25).expect("counted");
    });
    let insert_label = if label == "rbf" {
        "online_sgpr_insert"
    } else {
        "online_sgpr_insert_nested"
    };
    out.push((insert_label.to_owned(), count));
    if label == "rbf" {
        let mut svgp = Svgp::new(kernel, likelihood)
            .factor(&x, N, D, &y, &z, M_SPARSE)
            .map_err(|(_, e)| e)
            .expect("spd");
        let mut params = vec![0.0; svgp.num_params()];
        svgp.get_params(&mut params).expect("len");
        let mut grad = vec![0.0; params.len()];
        svgp.value_and_gradient_into(&params, &mut grad)
            .expect("warmup");
        let count = allocs_in(|| {
            svgp.value_and_gradient_into(&params, &mut grad)
                .expect("counted");
        });
        out.push(("svgp_mll_and_grad".to_owned(), count));
    }
    out
}

/// One call on each sparse path after a warmup call. The nested kernel (a
/// product of sums) covers the nested scratch levels on the insert path.
#[test]
fn sparse_allocs_after_warmup() {
    let _guard = alloc_lock();
    let rbf = || KernelSpec::from(RbfKernel::new(ELL).expect("ell"));
    let constant = |v: f64| KernelSpec::from(ConstantKernel::new(v).expect("constant"));
    let nested = (rbf() + KernelSpec::from(RbfKernel::new(2.0 * ELL).expect("ell")))
        * (constant(1.5) * rbf() + constant(0.5));
    let mut counts = sparse_allocs("rbf", rbf());
    counts.extend(sparse_allocs("nested", nested));
    for (label, cap) in MAX_SPARSE_ALLOCS {
        let count = counts
            .iter()
            .find(|(name, _)| name == label)
            .map(|(_, count)| *count)
            .expect("measured");
        assert_alloc_cap(label, count, cap);
    }
}

/// Allocations (or, with `count = bytes_in`, bytes) of one `predict_into` of
/// `M` points after a warmup call on a [`gprx::FittedSgpr`], the
/// [`gprx::OnlineSgpr`] after one insert, and a [`gprx::FittedSvgp`], at
/// precision `P`.
#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn sparse_predict_counts<P: GpScalar>(
    kernel: KernelSpec,
    count: fn(&mut dyn FnMut()) -> usize,
) -> [(&'static str, usize); 3] {
    ensure_one_rayon_worker();
    let x = fill_column_major(N, D, SEED);
    let z = fill_column_major(M_SPARSE, D, SEED.wrapping_add(2));
    let xs = fill_column_major(M, D, SEED.wrapping_add(1));
    let mut rng = small_rng(SEED ^ 0xA5A5_A5A5_A5A5_A5A5);
    let y: Vec<f64> = (0..N).map(|_| open_unit(&mut rng)).collect();
    let likelihood = GaussianLikelihood::new(NOISE).expect("noise");
    let mut pred = Prediction::default();
    let mut fitted = Sgpr::new(kernel.clone(), likelihood)
        .with_optimizer(Fixed)
        .with_precision::<P>()
        .factor(&x, N, D, &y, &z, M_SPARSE)
        .map_err(|(_, e)| e)
        .expect("spd");
    fitted.predict_into(&xs, M, D, &mut pred).expect("warmup");
    let sgpr = count(&mut || fitted.predict_into(&xs, M, D, &mut pred).expect("counted"));
    let mut online = fitted.into_online();
    let x_new: Vec<f64> = (0..D).map(|dim| 0.1 * dim as f64).collect();
    online.insert(&x_new, 0.5).expect("insert");
    online.predict_into(&xs, M, D, &mut pred).expect("warmup");
    let online = count(&mut || online.predict_into(&xs, M, D, &mut pred).expect("counted"));
    let mut svgp = Svgp::new(kernel, likelihood)
        .with_precision::<P>()
        .factor(&x, N, D, &y, &z, M_SPARSE)
        .map_err(|(_, e)| e)
        .expect("spd");
    svgp.predict_into(&xs, M, D, &mut pred).expect("warmup");
    let svgp = count(&mut || svgp.predict_into(&xs, M, D, &mut pred).expect("counted"));
    [("sgpr", sgpr), ("online_sgpr", online), ("svgp", svgp)]
}

fn allocs_in_dyn(f: &mut dyn FnMut()) -> usize {
    allocs_in(f)
}

fn bytes_in_dyn(f: &mut dyn FnMut()) -> usize {
    bytes_in(f)
}

/// Sparse `predict_into` after a warmup call: every model, a leaf and a
/// nested kernel, `f64` and `f32` storage, and mixed precision.
#[test]
fn sparse_predict_into_allocs_after_warmup() {
    let _guard = alloc_lock();
    let rbf = || KernelSpec::from(RbfKernel::new(ELL).expect("ell"));
    let constant = |v: f64| KernelSpec::from(ConstantKernel::new(v).expect("constant"));
    let nested = || {
        (rbf() + KernelSpec::from(RbfKernel::new(2.0 * ELL).expect("ell")))
            * (constant(1.5) * rbf() + constant(0.5))
    };
    let cases = [
        (
            "f64",
            sparse_predict_counts::<DoublePrecision>(rbf(), allocs_in_dyn),
        ),
        (
            "f64_nested",
            sparse_predict_counts::<DoublePrecision>(nested(), allocs_in_dyn),
        ),
        (
            "f32",
            sparse_predict_counts::<SinglePrecision>(rbf(), allocs_in_dyn),
        ),
        (
            "f32_nested",
            sparse_predict_counts::<SinglePrecision>(nested(), allocs_in_dyn),
        ),
    ];
    for (precision, counts) in cases {
        for (model, count) in counts {
            assert_alloc_cap(
                &format!("{model}_predict_100_{precision}"),
                count,
                MAX_SPARSE_PREDICT_100_ALLOCS,
            );
        }
    }
    for (label, counts) in [
        (
            "promote",
            sparse_predict_counts::<MixedPrecision<gprx::PromoteStorage>>(rbf(), allocs_in_dyn),
        ),
        (
            "reevaluate",
            sparse_predict_counts::<MixedPrecision<ReevaluateKernel>>(rbf(), allocs_in_dyn),
        ),
    ] {
        for (model, count) in counts {
            if model != "svgp" {
                assert_alloc_cap(
                    &format!("{model}_predict_100_mixed_{label}"),
                    count,
                    MAX_SPARSE_PREDICT_100_ALLOCS,
                );
            }
        }
    }
    for (label, counts) in [
        (
            "promote",
            sparse_predict_counts::<MixedPrecision<gprx::PromoteStorage>>(rbf(), bytes_in_dyn),
        ),
        (
            "reevaluate",
            sparse_predict_counts::<MixedPrecision<ReevaluateKernel>>(rbf(), bytes_in_dyn),
        ),
    ] {
        let bytes = counts[2].1;
        eprintln!(
            "svgp_predict_100_mixed_{label}: bytes={bytes} cap={MAX_SVGP_MIXED_PREDICT_100_BYTES}"
        );
        assert!(
            bytes <= MAX_SVGP_MIXED_PREDICT_100_BYTES,
            "svgp_predict_100_mixed_{label}: bytes={bytes}, cap={MAX_SVGP_MIXED_PREDICT_100_BYTES}"
        );
    }
}

/// Bytes one Adam step of `Svgp::fit` may allocate at `n = 4096`, relative to
/// `n = 512` at the same batch size: a step costs `O(batch · m²)`, so its
/// allocations do not grow with `n` (#300). It grew with `n` (8×) while a step
/// rebuilt `A` and the tangents of every point.
const MAX_SVGP_STEP_BYTES_GROWTH: f64 = 1.25;

/// Bytes of one `Svgp::fit` with `epochs` epochs of mini-batches of 32.
#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn svgp_fit_bytes(n: usize, epochs: u64) -> usize {
    ensure_one_rayon_worker();
    let x = fill_column_major(n, D, SEED);
    let z = fill_column_major(M_SPARSE, D, SEED.wrapping_add(2));
    let mut rng = small_rng(SEED ^ 0xA5A5_A5A5_A5A5_A5A5);
    let y: Vec<f64> = (0..n).map(|_| open_unit(&mut rng)).collect();
    let kernel = KernelSpec::from(RbfKernel::new(ELL).expect("ell"));
    let adam = Adam::new()
        .with_batch_size(NonZeroUsize::new(32).expect("batch"))
        .with_epochs(NonZeroU64::new(epochs).expect("epochs"));
    let trainer =
        Svgp::new(kernel, GaussianLikelihood::new(NOISE).expect("noise")).with_optimizer(adam);
    let mut fitted = None;
    let bytes = bytes_in(|| {
        fitted = Some(
            trainer
                .fit(&x, n, D, &y, &z, M_SPARSE)
                .map_err(|(_, e)| e)
                .expect("fit"),
        );
    });
    drop(fitted);
    bytes
}

/// The bytes of one step: one more epoch is `n / 32` more steps, and the
/// setup and the final rebuild of `A` cancel in the difference.
fn svgp_step_bytes(n: usize) -> f64 {
    let extra = svgp_fit_bytes(n, 2) as f64 - svgp_fit_bytes(n, 1) as f64;
    extra / (n / 32) as f64
}

#[test]
fn svgp_adam_step_bytes_do_not_grow_with_n() {
    let _guard = alloc_lock();
    let _ = svgp_fit_bytes(64, 1); // one-time allocations (thread pool, statics)
    let small = svgp_step_bytes(512);
    let large = svgp_step_bytes(4096);
    eprintln!("svgp step bytes: n=512 {small:.0}, n=4096 {large:.0}");
    assert!(
        large <= MAX_SVGP_STEP_BYTES_GROWTH * small,
        "an Adam step allocates {large:.0} bytes at n=4096 and {small:.0} at n=512: it scales with n"
    );
}
