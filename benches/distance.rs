//! Criterion baseline for supplied squared distances (design §5.6, §15.3).
//!
//! Every operation a distance model must not make slower is measured here on
//! the coordinate path (`DistanceCachePolicy::Cached`, the default), on one
//! fixed problem: `n = 512`, `d = 4`, `q = 100`, `m = 64`
//! (`tests/common/problems.rs`). Save it as the named baseline `d1-coords`.
//! The rows that add supplied distances add their cases to this file and
//! must not exceed it.

#![allow(missing_docs, clippy::expect_used, clippy::unwrap_used)]

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use gprx::kernel::{KernelSpec, RbfArdKernel, RbfKernel};
use gprx::{FittedGpr, Fixed, GaussianLikelihood, Gpr, OnlineGpr, Prediction, Sgpr, Svgp};

#[path = "../tests/common/problems.rs"]
#[allow(dead_code)]
mod problems;
#[path = "../src/rng.rs"]
#[allow(dead_code)]
mod rng;

use problems::{DistanceBaseline, distance_baseline};

fn lik() -> GaussianLikelihood {
    GaussianLikelihood::new(0.1).expect("noise")
}

/// The two kernels of the baseline: isotropic RBF and ARD RBF.
fn kernels() -> [(&'static str, KernelSpec); 2] {
    [
        ("rbf", KernelSpec::from(RbfKernel::new(0.5).expect("ell"))),
        (
            "rbf_ard",
            KernelSpec::from(RbfArdKernel::new(&[0.5, 0.6, 0.7, 0.8]).expect("ell")),
        ),
    ]
}

fn fitted(p: &DistanceBaseline, kernel: &KernelSpec) -> FittedGpr<Fixed> {
    Gpr::new(kernel.clone(), lik())
        .with_optimizer(Fixed)
        .factor(&p.x, p.n, p.d, &p.y)
        .expect("factor")
}

/// An online model with room for one more point: an insert and a delete
/// of it have grown the factor once.
fn online(p: &DistanceBaseline, kernel: &KernelSpec) -> OnlineGpr<Fixed> {
    let mut online = fitted(p, kernel).into_online().expect("online");
    let id = online.insert(&p.x_new, p.y_new).expect("grow");
    online.delete(id).expect("shrink");
    online
}

fn exact(c: &mut Criterion) {
    let p = distance_baseline();
    let mut group = c.benchmark_group("distance_baseline");
    group.sample_size(20);
    for (name, kernel) in kernels() {
        group.bench_function(format!("{name}/factor"), |b| {
            b.iter(|| fitted(&p, &kernel));
        });
        let mut model = fitted(&p, &kernel);
        let mut theta = vec![0.0; model.num_params()];
        model.get_params(&mut theta).expect("theta");
        let mut grad = vec![0.0; theta.len()];
        group.bench_function(format!("{name}/mll_and_grad"), |b| {
            b.iter(|| {
                model
                    .value_and_gradient_into(&theta, &mut grad)
                    .expect("mll")
            });
        });
        let mut pred = Prediction::default();
        group.bench_function(format!("{name}/predict_into"), |b| {
            b.iter(|| {
                model
                    .predict_into(&p.xq, p.q, p.d, &mut pred)
                    .expect("predict")
            });
        });
        let base = online(&p, &kernel);
        group.bench_function(format!("{name}/online_insert"), |b| {
            b.iter_batched(
                || base.clone(),
                |mut online| online.insert(&p.x_new, p.y_new).expect("insert"),
                BatchSize::LargeInput,
            );
        });
        for (label, index) in [("first", 0), ("middle", p.n / 2), ("last", p.n - 1)] {
            let id = base.point_ids()[index];
            group.bench_function(format!("{name}/online_delete_{label}"), |b| {
                b.iter_batched(
                    || base.clone(),
                    |mut online| online.delete(id).expect("delete"),
                    BatchSize::LargeInput,
                );
            });
        }
        group.bench_function(format!("{name}/online_refit"), |b| {
            b.iter_batched(
                || base.clone(),
                |mut online| online.refit().expect("refit"),
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

fn sparse(c: &mut Criterion) {
    let p = distance_baseline();
    let mut group = c.benchmark_group("distance_baseline");
    group.sample_size(20);
    for (name, kernel) in kernels() {
        let sgpr = || {
            Sgpr::new(kernel.clone(), lik())
                .with_optimizer(Fixed)
                .factor(&p.x, p.n, p.d, &p.y, &p.z, p.m)
                .map_err(|(_, e)| e)
                .expect("sgpr")
        };
        group.bench_function(format!("{name}/sgpr_factor"), |b| {
            b.iter(sgpr);
        });
        let mut model = sgpr();
        let mut pred = Prediction::default();
        group.bench_function(format!("{name}/sgpr_predict_into"), |b| {
            b.iter(|| {
                model
                    .predict_into(&p.xq, p.q, p.d, &mut pred)
                    .expect("predict")
            });
        });
        let mut model = Svgp::new(kernel.clone(), lik())
            .factor(&p.x, p.n, p.d, &p.y, &p.z, p.m)
            .map_err(|(_, e)| e)
            .expect("svgp");
        group.bench_function(format!("{name}/svgp_predict_into"), |b| {
            b.iter(|| {
                model
                    .predict_into(&p.xq, p.q, p.d, &mut pred)
                    .expect("predict")
            });
        });
    }
    group.finish();
}

criterion_group!(distance, exact, sparse);
criterion_main!(distance);
