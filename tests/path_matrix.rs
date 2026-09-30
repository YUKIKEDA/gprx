//! Every option combination of the sparse fit either succeeds or is one of the
//! listed, documented exceptions. A combination that fails for another reason
//! fails this test.
//!
//! The default test runs an all-pairs cover of the options (every pair of
//! values of any two options appears in some combination); the exhaustive
//! matrix is `#[ignore]`d.
//!
//! Exceptions:
//! - Matérn `ν = 1/2` with `FreeInducing`: `CoordGradientUnsupported` (its
//!   coordinate derivative is undefined where two points coincide).

#![allow(clippy::unwrap_used)] // fixtures outside the `#[test]` body

use gprx::kernel::{
    ConstantKernel, KernelSpec, LinearKernel, MaternArdKernel, MaternKernel, MaternNu,
    PeriodicKernel, RationalQuadraticArdKernel, RationalQuadraticKernel, RbfArdKernel, RbfKernel,
    WhiteKernel,
};
use gprx::{
    DoublePrecision, FastSimulatedAnnealing, FreeInducing, GaussianLikelihood, GprError, KernelExp,
    Lbfgs, NelderMead, Sgpr, SinglePrecision, TrustRegion,
};

const N: usize = 12;
const D: usize = 2;
const M: usize = 3;

fn data() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..N * D)
        .map(|i| 0.05 + 0.9 * (((i * 7 + 3) % 23) as f64) / 23.0)
        .collect();
    let y: Vec<f64> = (0..N)
        .map(|i| (x[i] * 3.0).sin() + 0.3 * x[N + i] - 0.4)
        .collect();
    let z: Vec<f64> = (0..M * D)
        .map(|i| 0.1 + 0.8 * (i as f64) / (M * D) as f64)
        .collect();
    (x, y, z)
}

fn k(spec: impl Into<KernelSpec>) -> KernelSpec {
    spec.into()
}

fn kernels() -> Vec<(&'static str, KernelSpec, bool)> {
    let rbf = || k(RbfKernel::new(0.9).unwrap());
    let matern = |nu| k(MaternKernel::new(1.0, nu).unwrap());
    let rq = || k(RationalQuadraticKernel::new(1.1, 0.8).unwrap());
    let periodic = || k(PeriodicKernel::new(0.9, 1.7).unwrap());
    let linear = || k(LinearKernel::new(0.6).unwrap());
    let constant = |v| k(ConstantKernel::new(v).unwrap());
    let white = || k(WhiteKernel::new(0.05).unwrap());
    let rbf_ard = || k(RbfArdKernel::new(&[0.9, 1.3]).unwrap());
    let matern_ard = |nu| k(MaternArdKernel::new(&[0.9, 1.3], nu).unwrap());
    let rq_ard = || k(RationalQuadraticArdKernel::new(&[0.9, 1.3], 0.7).unwrap());
    // (name, kernel, free inducing unsupported)
    vec![
        ("rbf", rbf(), false),
        ("matern 1/2", matern(MaternNu::Half), true),
        ("matern 3/2", matern(MaternNu::ThreeHalves), false),
        ("matern 5/2", matern(MaternNu::FiveHalves), false),
        ("rq", rq(), false),
        ("periodic", periodic(), false),
        ("linear", linear(), false),
        ("rbf ard", rbf_ard(), false),
        ("matern ard 3/2", matern_ard(MaternNu::ThreeHalves), false),
        ("matern ard 5/2", matern_ard(MaternNu::FiveHalves), false),
        ("rq ard", rq_ard(), false),
        ("constant * rbf", constant(1.4) * rbf(), false),
        ("constant * rbf ard", constant(1.4) * rbf_ard(), false),
        ("rbf + white", rbf() + white(), false),
        ("rbf + linear", rbf() + linear(), false),
        ("rbf * periodic", rbf() * periodic(), false),
        ("linear * rbf", linear() * rbf(), false),
        (
            "constant * matern 1/2",
            constant(1.4) * matern(MaternNu::Half),
            true,
        ),
        (
            "(rbf + rq) * (periodic + constant)",
            (rbf() + rq()) * (periodic() + constant(0.5)),
            false,
        ),
        (
            "constant * matern ard 5/2",
            constant(0.8) * matern_ard(MaternNu::FiveHalves),
            false,
        ),
    ]
}

fn allowed(err: &GprError, free: bool, unsupported: bool) -> bool {
    matches!(err, GprError::CoordGradientUnsupported) && free && unsupported
}

const OPTIMIZERS: [&str; 4] = ["lbfgs", "nelder-mead", "trust-region", "fsa"];
const PRECISIONS: [&str; 2] = ["f64", "f32"];
const MATHS: [KernelExp; 2] = [KernelExp::Accurate, KernelExp::FastApprox];

/// One combination: the failure message, or `None` when it fits (or is a
/// listed exception).
macro_rules! fit_case {
    ($name:expr, $kernel:expr, $unsupported:expr, $opt:expr, $opt_name:expr,
     $precision:ty, $precision_name:expr, $math:expr, $free:expr) => {{
        let (x, y, z) = data();
        let likelihood = GaussianLikelihood::new(0.1).unwrap();
        let base = Sgpr::new($kernel.clone(), likelihood)
            .with_optimizer($opt)
            .with_precision::<$precision>()
            .with_math($math);
        let result = if $free {
            base.with_inducing(FreeInducing)
                .fit(&x, N, D, &y, &z, M)
                .map(|f| f.neg_log_marginal_likelihood().unwrap_or(f64::NAN))
                .map_err(|(_, e)| e)
        } else {
            base.fit(&x, N, D, &y, &z, M)
                .map(|f| f.neg_log_marginal_likelihood().unwrap_or(f64::NAN))
                .map_err(|(_, e)| e)
        };
        let label = format!(
            "{} · {} · {} · {:?} · {}",
            $name,
            $opt_name,
            $precision_name,
            $math,
            if $free { "free" } else { "fixed" }
        );
        match result {
            Ok(value) if value.is_finite() => None,
            Ok(value) => Some(format!("{label}: non-finite NLML {value}")),
            Err(err) if allowed(&err, $free, $unsupported) => None,
            Err(err) => Some(format!("{label}: {err}")),
        }
    }};
}

macro_rules! by_precision {
    ($name:expr, $kernel:expr, $unsupported:expr, $opt:expr, $opt_name:expr,
     $precision:expr, $math:expr, $free:expr) => {
        match $precision {
            0 => fit_case!(
                $name,
                $kernel,
                $unsupported,
                $opt,
                $opt_name,
                DoublePrecision,
                "f64",
                $math,
                $free
            ),
            _ => fit_case!(
                $name,
                $kernel,
                $unsupported,
                $opt,
                $opt_name,
                SinglePrecision,
                "f32",
                $math,
                $free
            ),
        }
    };
}

/// Indices: kernel, optimizer, precision, exp math, free inducing points.
type Case = [usize; 5];

fn run_case([k_i, opt, precision, math, free]: Case) -> Option<String> {
    let (name, kernel, unsupported) = kernels().swap_remove(k_i);
    let math = MATHS[math];
    let free = free == 1;
    match opt {
        0 => by_precision!(
            name,
            kernel,
            unsupported,
            Lbfgs::new(),
            OPTIMIZERS[0],
            precision,
            math,
            free
        ),
        1 => by_precision!(
            name,
            kernel,
            unsupported,
            NelderMead::new(),
            OPTIMIZERS[1],
            precision,
            math,
            free
        ),
        2 => by_precision!(
            name,
            kernel,
            unsupported,
            TrustRegion::new(),
            OPTIMIZERS[2],
            precision,
            math,
            free
        ),
        _ => by_precision!(
            name,
            kernel,
            unsupported,
            FastSimulatedAnnealing::new().with_seed(7),
            OPTIMIZERS[3],
            precision,
            math,
            free
        ),
    }
}

/// A deterministic all-pairs cover: every pair of values of any two options
/// (kernel, optimizer, precision, exp math, inducing points) is in at least one
/// chosen combination. About a hundred combinations instead of all 640.
fn pairwise_cover(sizes: [usize; 5]) -> Vec<Case> {
    let mut all: Vec<Case> = vec![[0; 5]];
    for (dim, &size) in sizes.iter().enumerate() {
        all = all
            .into_iter()
            .flat_map(|case| {
                (0..size).map(move |v| {
                    let mut next = case;
                    next[dim] = v;
                    next
                })
            })
            .collect();
    }
    let pairs = |case: &Case| -> Vec<(usize, usize, usize, usize)> {
        let mut out = Vec::new();
        for i in 0..5 {
            for j in i + 1..5 {
                out.push((i, case[i], j, case[j]));
            }
        }
        out
    };
    let mut uncovered: std::collections::HashSet<_> = all.iter().flat_map(pairs).collect();
    let mut chosen = Vec::new();
    while !uncovered.is_empty() {
        let best = all
            .iter()
            .max_by_key(|case| {
                (
                    pairs(case)
                        .iter()
                        .filter(|p| uncovered.contains(*p))
                        .count(),
                    std::cmp::Reverse(case.to_vec()),
                )
            })
            .copied()
            .unwrap();
        for pair in pairs(&best) {
            uncovered.remove(&pair);
        }
        chosen.push(best);
    }
    chosen
}

fn assert_no_failures(cases: &[Case]) {
    let failures: Vec<String> = cases.iter().filter_map(|c| run_case(*c)).collect();
    assert!(
        failures.is_empty(),
        "{} of {} combinations failed:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
}

/// The default run: an all-pairs cover of the options.
#[test]
fn sparse_option_pairs_fit_or_are_a_listed_exception() {
    let sizes = [
        kernels().len(),
        OPTIMIZERS.len(),
        PRECISIONS.len(),
        MATHS.len(),
        2,
    ];
    let cases = pairwise_cover(sizes);
    assert!(cases.len() < 200, "cover has {} cases", cases.len());
    assert_no_failures(&cases);
}

/// Every combination (about 640 fits): `cargo test --test path_matrix -- --ignored`.
#[test]
#[ignore = "exhaustive; minutes in a debug build"]
fn every_sparse_option_combination_fits_or_is_a_listed_exception() {
    let sizes = [
        kernels().len(),
        OPTIMIZERS.len(),
        PRECISIONS.len(),
        MATHS.len(),
        2,
    ];
    let mut cases: Vec<Case> = vec![[0; 5]];
    for (dim, &size) in sizes.iter().enumerate() {
        cases = cases
            .into_iter()
            .flat_map(|case| {
                (0..size).map(move |v| {
                    let mut next = case;
                    next[dim] = v;
                    next
                })
            })
            .collect();
    }
    assert_no_failures(&cases);
}
