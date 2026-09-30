//! Every option combination of the sparse fit either succeeds or is one of the
//! listed, documented exceptions. A combination that fails for another reason
//! fails this test.
//!
//! Known failures are listed in `KNOWN_FAILURES` and belong to
//! [#306](https://github.com/YUKIKEDA/gprx/issues/306) (Newton and
//! NonlinearCg robustness). The test fails when one of them starts to pass, so
//! the list shrinks to empty with that Issue.
//!
//! Exceptions:
//! - Matérn `ν = 1/2` with `FreeInducing`: `CoordGradientUnsupported` (its
//!   coordinate derivative is undefined where two points coincide).
//! - `Newton` with an unlucky start: `OptimizationNotConverged`. Newton is the
//!   plain method (no line search), documented on [`gprx::Newton`]; it is
//!   allowed to fail on every model, and must not fail with anything else.

#![allow(clippy::unwrap_used)] // fixtures outside the `#[test]` body

use gprx::kernel::{
    ConstantKernel, KernelSpec, LinearKernel, MaternArdKernel, MaternKernel, MaternNu,
    PeriodicKernel, RationalQuadraticArdKernel, RationalQuadraticKernel, RbfArdKernel, RbfKernel,
    WhiteKernel,
};
use gprx::{
    DoublePrecision, FastSimulatedAnnealing, FreeInducing, GaussianLikelihood, GprError, KernelExp,
    Lbfgs, NelderMead, Newton, NonlinearCg, Sgpr, SinglePrecision,
};

/// Labels of the combinations that fail today (#306).
const KNOWN_FAILURES: [&str; 15] = [
    "constant * rbf ard · ncg · SinglePrecision · Accurate · free",
    "linear * rbf · newton · SinglePrecision · FastApprox · free",
    "(rbf + rq) * (periodic + constant) · newton · DoublePrecision · Accurate · free",
    "(rbf + rq) * (periodic + constant) · newton · SinglePrecision · Accurate · free",
    "(rbf + rq) * (periodic + constant) · newton · DoublePrecision · FastApprox · free",
    "(rbf + rq) * (periodic + constant) · newton · SinglePrecision · FastApprox · free",
    "constant * matern ard 5/2 · newton · SinglePrecision · FastApprox · fixed",
    "linear · ncg · DoublePrecision · Accurate · fixed",
    "linear · ncg · DoublePrecision · Accurate · free",
    "linear · ncg · SinglePrecision · Accurate · free",
    "linear · ncg · DoublePrecision · FastApprox · fixed",
    "linear · ncg · DoublePrecision · FastApprox · free",
    "linear · ncg · SinglePrecision · FastApprox · free",
    "matern ard 5/2 · newton · DoublePrecision · FastApprox · free",
    "linear * rbf · ncg · DoublePrecision · FastApprox · free",
];

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

fn allowed(err: &GprError, free: bool, unsupported: bool, newton: bool) -> bool {
    match err {
        GprError::CoordGradientUnsupported => free && unsupported,
        GprError::OptimizationNotConverged { .. } => newton,
        _ => false,
    }
}

macro_rules! run {
    ($failures:ident, $name:expr, $kernel:expr, $unsupported:expr, $opt_name:expr, $newton:expr,
     $opt:expr, $precision:ty, $math:expr, $free:expr) => {{
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
            stringify!($precision),
            $math,
            if $free { "free" } else { "fixed" }
        );
        match result {
            Ok(value) if value.is_finite() => {}
            Ok(value) => $failures.push(format!("{label}: non-finite NLML {value}")),
            Err(err) if allowed(&err, $free, $unsupported, $newton) => {}
            Err(err) => $failures.push(format!("{label}: {err}")),
        }
    }};
}

macro_rules! all_optimizers {
    ($failures:ident, $name:expr, $kernel:expr, $unsupported:expr, $precision:ty, $math:expr, $free:expr) => {
        run!(
            $failures,
            $name,
            $kernel,
            $unsupported,
            "lbfgs",
            false,
            Lbfgs::new(),
            $precision,
            $math,
            $free
        );
        run!(
            $failures,
            $name,
            $kernel,
            $unsupported,
            "ncg",
            false,
            NonlinearCg::new(),
            $precision,
            $math,
            $free
        );
        run!(
            $failures,
            $name,
            $kernel,
            $unsupported,
            "nelder-mead",
            false,
            NelderMead::new(),
            $precision,
            $math,
            $free
        );
        run!(
            $failures,
            $name,
            $kernel,
            $unsupported,
            "newton",
            true,
            Newton::new(),
            $precision,
            $math,
            $free
        );
        run!(
            $failures,
            $name,
            $kernel,
            $unsupported,
            "fsa",
            false,
            FastSimulatedAnnealing::new().with_seed(7),
            $precision,
            $math,
            $free
        );
    };
}

#[test]
fn every_sparse_option_combination_fits_or_is_a_listed_exception() {
    let mut failures = Vec::new();
    for (name, kernel, unsupported) in kernels() {
        for math in [KernelExp::Accurate, KernelExp::FastApprox] {
            for free in [false, true] {
                all_optimizers!(
                    failures,
                    name,
                    kernel,
                    unsupported,
                    DoublePrecision,
                    math,
                    free
                );
                all_optimizers!(
                    failures,
                    name,
                    kernel,
                    unsupported,
                    SinglePrecision,
                    math,
                    free
                );
            }
        }
    }
    let is_known = |failure: &String| {
        KNOWN_FAILURES
            .iter()
            .any(|known| failure.starts_with(&format!("{known}:")))
    };
    let unexpected: Vec<&String> = failures.iter().filter(|f| !is_known(f)).collect();
    let fixed: Vec<&&str> = KNOWN_FAILURES
        .iter()
        .filter(|known| !failures.iter().any(|f| f.starts_with(&format!("{known}:"))))
        .collect();
    assert!(
        unexpected.is_empty(),
        "{} unexpected failures:\n{}",
        unexpected.len(),
        unexpected
            .iter()
            .map(|f| f.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    );
    assert!(
        fixed.is_empty(),
        "now passing, remove from KNOWN_FAILURES: {fixed:?}"
    );
}
