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
use gprx::kernel::{
    ArdDistance, DistanceKernel, DistanceSource, KernelSpec, RbfArdKernel, RbfKernel,
    ScalarDistance,
};
use gprx::{
    FittedGpr, FittedSgpr, Fixed, GaussianLikelihood, Gpr, OnlineGpr, Prediction, Sgpr, Svgp,
};

#[path = "../tests/common/problems.rs"]
#[allow(dead_code)]
mod problems;
#[path = "../src/rng.rs"]
#[allow(dead_code)]
mod rng;

use problems::{DistanceBaseline, Supplied, distance_baseline};

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
            b.iter_batched_ref(
                || base.clone(),
                |online| online.insert(&p.x_new, p.y_new).expect("insert"),
                BatchSize::LargeInput,
            );
        });
        for (label, index) in [("first", 0), ("middle", p.n / 2), ("last", p.n - 1)] {
            let id = base.point_ids()[index];
            group.bench_function(format!("{name}/online_delete_{label}"), |b| {
                b.iter_batched_ref(
                    || base.clone(),
                    |online| online.delete(id).expect("delete"),
                    BatchSize::LargeInput,
                );
            });
        }
        group.bench_function(format!("{name}/online_refit"), |b| {
            b.iter_batched_ref(
                || base.clone(),
                |online| online.refit().expect("refit"),
                BatchSize::LargeInput,
            );
        });
        online_mll_and_grad(&mut group, &format!("{name}/online_mll_and_grad"), base);
    }
    group.finish();
}

/// The NLML and its gradient of an online model that has grown and shrunk
/// once (an ARD slot of a distance model then reads its row runs).
fn online_mll_and_grad<K: gprx::kernel::ModelKernel>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    id: &str,
    mut online: OnlineGpr<Fixed, gprx::DoublePrecision, K>,
) {
    let mut theta = vec![0.0; online.num_params()];
    online.get_params(&mut theta).expect("theta");
    let mut grad = vec![0.0; theta.len()];
    group.bench_function(id, |b| {
        b.iter(|| {
            online
                .value_and_gradient_into(&theta, &mut grad)
                .expect("mll")
        });
    });
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
        sgpr_mll_and_grad(&mut group, &format!("{name}/sgpr_mll_and_grad"), &mut model);
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
        // Training point `m` becomes inducing; room for a point and an
        // inducing point, as after the first changes.
        let z_new: Vec<f64> = (0..p.d).map(|k| p.x[p.m + k * p.n]).collect();
        let mut base = sgpr().into_online();
        let id = base.insert(&p.x_new, p.y_new).expect("grow");
        base.delete(id).expect("shrink");
        let id = base.insert_inducing(&z_new).expect("grow");
        base.delete_inducing(id).expect("shrink");
        online_sgpr_changes(
            &mut group,
            &format!("{name}/online_sgpr"),
            &base,
            &|model| {
                let id = model.insert(&p.x_new, p.y_new).expect("settle");
                model.delete(id).expect("settle");
            },
            [
                &|model| {
                    model.insert(&p.x_new, p.y_new).expect("insert");
                },
                &|model| model.delete(model.point_ids()[p.n / 2]).expect("delete"),
                &|model| {
                    model.insert_inducing(&z_new).expect("insert_inducing");
                },
                &|model| {
                    let id = model.inducing_ids()[p.m / 2];
                    model.delete_inducing(id).expect("delete_inducing");
                },
            ],
        );
    }
    group.finish();
}

/// The four changes of an online SGPR, each on a copy of `base` that a
/// first insert and delete (`settle`) made its own (as a model is after its
/// first changes; the copy is not timed): `[insert, delete of the middle
/// point, insert_inducing, delete_inducing of the middle one]`.
fn online_sgpr_changes<M: Clone>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    prefix: &str,
    base: &M,
    settle: &dyn Fn(&mut M),
    changes: [&dyn Fn(&mut M); 4],
) {
    for (label, change) in [
        "insert",
        "delete_middle",
        "insert_inducing",
        "delete_inducing",
    ]
    .into_iter()
    .zip(changes)
    {
        group.bench_function(format!("{prefix}_{label}"), |b| {
            b.iter_batched_ref(
                || {
                    let mut model = base.clone();
                    settle(&mut model);
                    model
                },
                |model| change(model),
                BatchSize::LargeInput,
            );
        });
    }
}

/// The bound and its gradient of a fitted SGPR at its own `θ`.
fn sgpr_mll_and_grad<K: gprx::kernel::ModelKernel>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    id: &str,
    model: &mut FittedSgpr<Fixed, gprx::FixedInducing, gprx::DoublePrecision, K>,
) {
    let mut theta = vec![0.0; model.num_params()];
    model.get_params(&mut theta).expect("theta");
    let mut grad = vec![0.0; theta.len()];
    group.bench_function(id, |b| {
        b.iter(|| {
            model
                .value_and_gradient_into(&theta, &mut grad)
                .expect("mll")
        });
    });
}

/// The sparse operations of `sparse` on supplied distances, as `dist_*`:
/// the inducing points are the training points `0..m`, the points `z`
/// of the coordinate rows.
fn sparse_supplied(c: &mut Criterion) {
    let p = distance_baseline();
    let s = p.supplied();
    let nm = p.n * p.m;
    let train_refs: Vec<&[f64]> = s.train.iter().map(|b| &b[..nm]).collect();
    let cross_refs: Vec<&[f64]> = s.sparse_cross.iter().map(Vec::as_slice).collect();
    let inducing: Vec<usize> = (0..p.m).collect();
    let row_refs: Vec<&[f64]> = s.new.iter().map(|b| &b[..p.m]).collect();
    let row_sum = &s.new_sum;
    let column_refs: Vec<&[f64]> = s
        .train
        .iter()
        .map(|b| &b[p.m * p.n..(p.m + 1) * p.n])
        .collect();
    let column_sum = &s.train_sum[p.m * p.n..(p.m + 1) * p.n];
    let mut group = c.benchmark_group("distance_baseline");
    group.sample_size(20);
    for (name, slot, kernel) in distance_kernels(p.d) {
        let train = || match &slot {
            Slot::Scalar(image) => image.borrow(&s.train_sum[..nm]),
            Slot::Ard(bands) => bands.borrow(&train_refs),
        };
        let cross = || match &slot {
            Slot::Scalar(image) => image.borrow(&s.sparse_cross_sum),
            Slot::Ard(bands) => bands.borrow(&cross_refs),
        };
        let sgpr = || {
            Sgpr::new(kernel.clone(), lik())
                .with_optimizer(Fixed)
                .factor([train()], p.n, &p.y, &inducing)
                .map_err(|(_, e)| e)
                .expect("sgpr")
        };
        group.bench_function(format!("{name}/dist_sgpr_factor"), |b| {
            b.iter(sgpr);
        });
        let mut model = sgpr();
        sgpr_mll_and_grad(
            &mut group,
            &format!("{name}/dist_sgpr_mll_and_grad"),
            &mut model,
        );
        let mut pred = Prediction::default();
        group.bench_function(format!("{name}/dist_sgpr_predict_into"), |b| {
            b.iter(|| {
                model
                    .predict_into([cross()], p.q, &mut pred)
                    .expect("predict")
            });
        });
        let mut model = Svgp::new(kernel.clone(), lik())
            .factor([train()], p.n, &p.y, &inducing)
            .map_err(|(_, e)| e)
            .expect("svgp");
        group.bench_function(format!("{name}/dist_svgp_predict_into"), |b| {
            b.iter(|| {
                model
                    .predict_into([cross()], p.q, &mut pred)
                    .expect("predict")
            });
        });
        // As `sparse`: the new point's squared distances to the inducing
        // points, and training point `m`'s to every training point.
        let row = || match &slot {
            Slot::Scalar(image) => image.borrow(&row_sum[..p.m]),
            Slot::Ard(bands) => bands.borrow(&row_refs),
        };
        let column = || match &slot {
            Slot::Scalar(image) => image.borrow(&column_sum),
            Slot::Ard(bands) => bands.borrow(&column_refs),
        };
        let mut base = sgpr().into_online();
        let id = base.insert([row()], p.y_new).expect("grow");
        base.delete(id).expect("shrink");
        let point = base.point_ids()[p.m];
        let id = base.insert_inducing(point, [column()]).expect("grow");
        base.delete_inducing(id).expect("shrink");
        online_sgpr_changes(
            &mut group,
            &format!("{name}/dist_online_sgpr"),
            &base,
            &|model| {
                let id = model.insert([row()], p.y_new).expect("settle");
                model.delete(id).expect("settle");
            },
            [
                &|model| {
                    model.insert([row()], p.y_new).expect("insert");
                },
                &|model| model.delete(model.point_ids()[p.n / 2]).expect("delete"),
                &|model| {
                    let point = model.point_ids()[p.m];
                    model
                        .insert_inducing(point, [column()])
                        .expect("insert_inducing");
                },
                &|model| {
                    let id = model.inducing_ids()[p.m / 2];
                    model.delete_inducing(id).expect("delete_inducing");
                },
            ],
        );
    }
    group.finish();
}

/// The same two kernels as [`kernels`] on one supplied-distance slot.
enum Slot {
    Scalar(ScalarDistance),
    Ard(ArdDistance),
}

fn distance_kernels(d: usize) -> [(&'static str, Slot, DistanceKernel); 2] {
    let image = ScalarDistance::new();
    let (bands, ard) =
        ArdDistance::from_leaf(RbfArdKernel::new(&[0.5, 0.6, 0.7, 0.8][..d]).expect("ell"));
    [
        (
            "rbf",
            Slot::Scalar(image),
            image.kernel(RbfKernel::new(0.5).expect("ell")),
        ),
        ("rbf_ard", Slot::Ard(bands), ard),
    ]
}

impl Slot {
    /// The training square, borrowed (a fit copies it into the model).
    fn train<'a>(&self, s: &'a Supplied, refs: &'a [&'a [f64]]) -> DistanceSource<'a> {
        match self {
            Self::Scalar(slot) => slot.borrow(&s.train_sum),
            Self::Ard(slot) => slot.borrow(refs),
        }
    }

    /// The training points' column to `x_new`, borrowed.
    fn new_point<'a>(&self, s: &'a Supplied, refs: &'a [&'a [f64]]) -> DistanceSource<'a> {
        match self {
            Self::Scalar(slot) => slot.borrow(&s.new_sum),
            Self::Ard(slot) => slot.borrow(refs),
        }
    }

    /// The train × query block, borrowed.
    fn cross<'a>(&self, s: &'a Supplied, refs: &'a [&'a [f64]]) -> DistanceSource<'a> {
        match self {
            Self::Scalar(slot) => slot.borrow(&s.cross_sum),
            Self::Ard(slot) => slot.borrow(refs),
        }
    }
}

/// The exact operations of `exact` on supplied distances, as `dist_*`.
fn exact_supplied(c: &mut Criterion) {
    let p = distance_baseline();
    let s = p.supplied();
    let train_refs: Vec<&[f64]> = s.train.iter().map(Vec::as_slice).collect();
    let cross_refs: Vec<&[f64]> = s.cross.iter().map(Vec::as_slice).collect();
    let new_refs: Vec<&[f64]> = s.new.iter().map(Vec::as_slice).collect();
    let mut group = c.benchmark_group("distance_baseline");
    group.sample_size(20);
    for (name, slot, kernel) in distance_kernels(p.d) {
        let fit = || {
            Gpr::new(kernel.clone(), lik())
                .with_optimizer(Fixed)
                .factor([slot.train(&s, &train_refs)], p.n, &p.y)
                .expect("factor")
        };
        group.bench_function(format!("{name}/dist_factor"), |b| {
            b.iter(fit);
        });
        let mut model = fit();
        let mut theta = vec![0.0; model.num_params()];
        model.get_params(&mut theta).expect("theta");
        let mut grad = vec![0.0; theta.len()];
        group.bench_function(format!("{name}/dist_mll_and_grad"), |b| {
            b.iter(|| {
                model
                    .value_and_gradient_into(&theta, &mut grad)
                    .expect("mll")
            });
        });
        let mut pred = Prediction::default();
        group.bench_function(format!("{name}/dist_predict_into"), |b| {
            b.iter(|| {
                model
                    .predict_into([slot.cross(&s, &cross_refs)], p.q, &mut pred)
                    .expect("predict")
            });
        });
        group.bench_function(format!("{name}/dist_refit"), |b| {
            b.iter(|| model.refit().expect("refit"));
        });
        // As `online`: room for one more point.
        let new = || slot.new_point(&s, &new_refs);
        let mut base = model.into_online().expect("online");
        let id = base.insert([new()], p.y_new).expect("grow");
        base.delete(id).expect("shrink");
        group.bench_function(format!("{name}/dist_online_insert"), |b| {
            b.iter_batched_ref(
                || base.clone(),
                |online| online.insert([new()], p.y_new).expect("insert"),
                BatchSize::LargeInput,
            );
        });
        for (label, index) in [("first", 0), ("middle", p.n / 2), ("last", p.n - 1)] {
            let id = base.point_ids()[index];
            group.bench_function(format!("{name}/dist_online_delete_{label}"), |b| {
                b.iter_batched_ref(
                    || base.clone(),
                    |online| online.delete(id).expect("delete"),
                    BatchSize::LargeInput,
                );
            });
        }
        group.bench_function(format!("{name}/dist_online_refit"), |b| {
            b.iter_batched_ref(
                || base.clone(),
                |online| online.refit().expect("refit"),
                BatchSize::LargeInput,
            );
        });
        online_mll_and_grad(
            &mut group,
            &format!("{name}/dist_online_mll_and_grad"),
            base,
        );
    }
    // The coordinate refit of a `FittedGpr`, beside `dist_refit`.
    for (name, kernel) in kernels() {
        let mut model = fitted(&p, &kernel);
        group.bench_function(format!("{name}/refit"), |b| {
            b.iter(|| model.refit().expect("refit"));
        });
    }
    group.finish();
}

criterion_group!(distance, exact, sparse, exact_supplied, sparse_supplied);
criterion_main!(distance);
