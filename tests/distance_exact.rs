//! Exact GPR on supplied squared distances against the same model on
//! coordinates. Public API only.

mod common;

use common::{assert_close, assert_slice_close};
use gprx::kernel::{
    ArdDistance, ConstantKernel, DistanceFill, KernelSpec, MaternKernel, MaternNu, RbfArdKernel,
    RbfKernel, ScalarDistance,
};
use gprx::{
    DistanceCachePolicy, Fixed, GaussianLikelihood, Gpr, GprError, PredictOptions, Prediction,
    SinglePrecision, VarianceKind,
};

const N: usize = 6;
const M: usize = 3;
const TOL: f64 = 1e-10;

/// Coordinate `k` of the training (`N`) or query (`M`) samples.
fn coord(k: usize, rows: usize, offset: f64) -> Vec<f64> {
    (0..rows)
        .map(|i| ((i as f64 + offset) * (0.41 + 0.17 * k as f64)).sin() * (1.0 + 0.5 * k as f64))
        .collect()
}

/// Column-major `a.len() × b.len()` squared differences.
fn sq(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(a.len() * b.len());
    for bj in b {
        for ai in a {
            out.push((ai - bj) * (ai - bj));
        }
    }
    out
}

/// `Σ_k` of the blocks of `sq`.
fn sum(blocks: &[Vec<f64>]) -> Vec<f64> {
    (0..blocks[0].len())
        .map(|i| blocks.iter().map(|b| b[i]).sum())
        .collect()
}

fn targets() -> Vec<f64> {
    (0..N).map(|i| (i as f64 * 0.7).cos()).collect()
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn lik() -> GaussianLikelihood {
    GaussianLikelihood::new(0.05).expect("noise")
}

fn assert_pred(a: &Prediction, b: &Prediction, tol: f64) {
    assert_slice_close(&a.mean, &b.mean, tol);
    assert_slice_close(&a.variance, &b.variance, tol);
}

#[test]
fn scalar_rbf_on_supplied_euclidean_matches_coordinates() {
    let (c0, c1) = (coord(0, N, 0.0), coord(1, N, 0.0));
    let (q0, q1) = (coord(0, M, 0.5), coord(1, M, 0.5));
    let x: Vec<f64> = c0.iter().chain(&c1).copied().collect();
    let xs: Vec<f64> = q0.iter().chain(&q1).copied().collect();
    let y = targets();
    let train = sum(&[sq(&c0, &c0), sq(&c1, &c1)]);
    let cross = sum(&[sq(&c0, &q0), sq(&c1, &q1)]);
    let query = sum(&[sq(&q0, &q0), sq(&q1, &q1)]);
    let rbf = RbfKernel::new(0.9).expect("ell");
    let coords = Gpr::new(KernelSpec::from(rbf), lik())
        .with_optimizer(Fixed)
        .factor(&x, N, 2, &y)
        .expect("coords");
    let image = ScalarDistance::new();
    let dist = Gpr::new(image.kernel(rbf), lik())
        .with_optimizer(Fixed)
        .factor([image.from_vec(train)], N, &y)
        .expect("distances");
    assert_close(
        dist.neg_log_marginal_likelihood().expect("nlml"),
        coords.neg_log_marginal_likelihood().expect("nlml"),
        TOL,
    );
    let expect = coords.predict(&xs, M, 2).expect("predict");
    let got = dist.predict([image.borrow(&cross)], M).expect("predict");
    assert_pred(&got, &expect, TOL);
    let cov_c = coords.predict_covariance(&xs, M, 2).expect("cov");
    let cov_d = dist
        .predict_covariance([image.borrow(&cross)], [image.borrow(&query)], M)
        .expect("cov");
    assert_slice_close(&cov_d.covariance, &cov_c.covariance, TOL);
    let loo_c = coords.loo_predict().expect("loo");
    let loo_d = dist.loo_predict().expect("loo");
    assert_pred(&loo_d, &loo_c, TOL);
    let draws_c = coords.sample(&xs, M, 2, 3, 11).expect("sample");
    let draws_d = dist
        .sample([image.borrow(&cross)], [image.borrow(&query)], M, 3, 11)
        .expect("sample");
    assert_slice_close(&draws_d, &draws_c, 1e-8);
    // The gradient and Hessian of the NLML agree too.
    let mut theta = vec![0.0; 2];
    coords.get_params(&mut theta).expect("theta");
    let (mut gc, mut gd) = (vec![0.0; 2], vec![0.0; 2]);
    let mut coords = coords;
    let mut dist = dist;
    coords
        .value_and_gradient_into(&theta, &mut gc)
        .expect("grad");
    dist.value_and_gradient_into(&theta, &mut gd).expect("grad");
    assert_slice_close(&gd, &gc, 1e-9);
    let (mut hc, mut hd) = (vec![0.0; 4], vec![0.0; 4]);
    coords.hessian_into(&theta, &mut hc).expect("hess");
    dist.hessian_into(&theta, &mut hd).expect("hess");
    assert_slice_close(&hd, &hc, 1e-8);
}

#[test]
fn ard_on_supplied_squared_differences_matches_coordinates() {
    let cols: Vec<Vec<f64>> = (0..3).map(|k| coord(k, N, 0.0)).collect();
    let qcols: Vec<Vec<f64>> = (0..3).map(|k| coord(k, M, 0.5)).collect();
    let x: Vec<f64> = cols.concat();
    let xs: Vec<f64> = qcols.concat();
    let y = targets();
    let ard = RbfArdKernel::new(&[0.8, 1.4, 2.0]).expect("ell");
    let coords = Gpr::new(KernelSpec::from(ard.clone()), lik())
        .with_optimizer(Fixed)
        .factor(&x, N, 3, &y)
        .expect("coords");
    let bands = ArdDistance::new(3).expect("dims");
    let train: Vec<Vec<f64>> = cols.iter().map(|c| sq(c, c)).collect();
    let cross: Vec<Vec<f64>> = cols.iter().zip(&qcols).map(|(c, q)| sq(c, q)).collect();
    let dist = Gpr::new(bands.kernel(ard).expect("dims"), lik())
        .with_optimizer(Fixed)
        .factor([bands.from_vecs(train)], N, &y)
        .expect("distances");
    let refs: Vec<&[f64]> = cross.iter().map(Vec::as_slice).collect();
    let got = dist.predict([bands.borrow(&refs)], M).expect("predict");
    let expect = coords.predict(&xs, M, 3).expect("predict");
    assert_pred(&got, &expect, TOL);
    assert_close(
        dist.neg_log_marginal_likelihood().expect("nlml"),
        coords.neg_log_marginal_likelihood().expect("nlml"),
        TOL,
    );
}

#[test]
fn a_distance_rbf_times_a_coordinate_rbf_is_one_ard_rbf() {
    let (c0, c1) = (coord(0, N, 0.0), coord(1, N, 0.0));
    let (q0, q1) = (coord(0, M, 0.5), coord(1, M, 0.5));
    let y = targets();
    let (ell0, ell1) = (0.7, 1.3);
    // exp(-Δ0²/2ℓ0²) · exp(-Δ1²/2ℓ1²) is the ARD RBF of (ℓ0, ℓ1).
    let reference = Gpr::new(
        KernelSpec::from(RbfArdKernel::new(&[ell0, ell1]).expect("ell")),
        lik(),
    )
    .with_optimizer(Fixed)
    .factor(&[c0.clone(), c1.clone()].concat(), N, 2, &y)
    .expect("reference");
    let image = ScalarDistance::new();
    let kernel = image.kernel(RbfKernel::new(ell0).expect("ell"))
        * KernelSpec::from(RbfKernel::new(ell1).expect("ell"));
    let model = Gpr::new(kernel, lik())
        .with_optimizer(Fixed)
        .factor([image.from_vec(sq(&c0, &c0))], N, &c1, 1, &y)
        .expect("model");
    let cross = sq(&c0, &q0);
    let got = model
        .predict([image.borrow(&cross)], &q1, M, 1)
        .expect("predict");
    let expect = reference
        .predict(&[q0.clone(), q1.clone()].concat(), M, 2)
        .expect("predict");
    assert_pred(&got, &expect, TOL);
    let cov = model
        .predict_covariance(
            [image.borrow(&cross)],
            [image.borrow(&sq(&q0, &q0))],
            &q1,
            M,
            1,
        )
        .expect("cov");
    let cov_ref = reference
        .predict_covariance(&[q0, q1].concat(), M, 2)
        .expect("cov");
    assert_slice_close(&cov.covariance, &cov_ref.covariance, TOL);
}

#[test]
fn two_slots_read_their_own_supplies_and_one_slot_twice_reads_one() {
    let (c0, c1) = (coord(0, N, 0.0), coord(1, N, 0.0));
    let (q0, q1) = (coord(0, M, 0.5), coord(1, M, 0.5));
    let y = targets();
    let (a, b) = (ScalarDistance::new(), ScalarDistance::new());
    let product =
        a.kernel(RbfKernel::new(0.7).expect("ell")) * b.kernel(RbfKernel::new(1.3).expect("ell"));
    let model = Gpr::new(product, lik())
        .with_optimizer(Fixed)
        .factor([b.from_vec(sq(&c1, &c1)), a.from_vec(sq(&c0, &c0))], N, &y)
        .expect("product");
    let reference = Gpr::new(
        KernelSpec::from(RbfArdKernel::new(&[0.7, 1.3]).expect("ell")),
        lik(),
    )
    .with_optimizer(Fixed)
    .factor(&[c0.clone(), c1.clone()].concat(), N, 2, &y)
    .expect("reference");
    let got = model
        .predict([a.borrow(&sq(&c0, &q0)), b.borrow(&sq(&c1, &q1))], M)
        .expect("predict");
    let expect = reference
        .predict(&[q0.clone(), q1].concat(), M, 2)
        .expect("predict");
    assert_pred(&got, &expect, TOL);
    // One slot used twice is one supply.
    let image = ScalarDistance::new();
    let sum_kernel = image.kernel(RbfKernel::new(1.0).expect("ell"))
        + image.kernel(MaternKernel::new(1.0, MaternNu::FiveHalves).expect("ell"));
    let reference = Gpr::new(
        KernelSpec::from(RbfKernel::new(1.0).expect("ell"))
            + KernelSpec::from(MaternKernel::new(1.0, MaternNu::FiveHalves).expect("ell")),
        lik(),
    )
    .with_optimizer(Fixed)
    .factor(&c0, N, 1, &y)
    .expect("reference");
    let fitted = Gpr::new(sum_kernel, lik())
        .with_optimizer(Fixed)
        .factor([image.from_slice(&sq(&c0, &c0))], N, &y)
        .expect("sum");
    let got = fitted
        .predict([image.borrow(&sq(&c0, &q0))], M)
        .expect("predict");
    let expect = reference.predict(&q0, M, 1).expect("predict");
    assert_pred(&got, &expect, TOL);
    // Two sources for one slot, a missing slot, or a foreign slot.
    let twice = Gpr::new(image.kernel(RbfKernel::new(1.0).expect("ell")), lik())
        .with_optimizer(Fixed)
        .factor(
            [
                image.from_slice(&sq(&c0, &c0)),
                image.from_slice(&sq(&c0, &c0)),
            ],
            N,
            &y,
        );
    assert!(matches!(twice, Err((_, GprError::LengthMismatch { .. }))));
    let foreign = ScalarDistance::new();
    let wrong = fitted.predict([foreign.borrow(&sq(&c0, &q0))], M);
    assert!(matches!(wrong, Err(GprError::LengthMismatch { .. })));
    let none = fitted.predict(Vec::new(), M);
    assert!(matches!(none, Err(GprError::LengthMismatch { .. })));
}

/// Writes `d²` between two one-dimensional sample sets.
struct Pairs<'a> {
    rows: &'a [f64],
    cols: &'a [f64],
}

impl DistanceFill for Pairs<'_> {
    fn fill(&self, n_rows: usize, n_cols: usize, out: &mut [f64]) {
        for j in 0..n_cols {
            for i in 0..n_rows {
                let diff = self.rows[i] - self.cols[j];
                out[i + j * n_rows] = diff * diff;
            }
        }
    }
}

#[test]
fn a_fill_matches_the_same_table_and_squares_are_checked() {
    let c0 = coord(0, N, 0.0);
    let q0 = coord(0, M, 0.5);
    let y = targets();
    let image = ScalarDistance::new();
    let kernel = image.kernel(RbfKernel::new(1.0).expect("ell"));
    let table = Gpr::new(kernel.clone(), lik())
        .with_optimizer(Fixed)
        .factor([image.from_slice(&sq(&c0, &c0))], N, &y)
        .expect("table");
    for policy in [false, true] {
        let mut trainer = Gpr::new(kernel.clone(), lik());
        if policy {
            trainer = trainer.with_prefer_memory();
        }
        let fill = trainer
            .fit(
                [image.fill(&Pairs {
                    rows: &c0,
                    cols: &c0,
                })],
                N,
                &y,
            )
            .expect("fill");
        if policy {
            assert_eq!(fill.distance_cache_policy(), DistanceCachePolicy::Uncached);
        }
        let by_fill = fill
            .predict(
                [image.fill(&Pairs {
                    rows: &c0,
                    cols: &q0,
                })],
                M,
            )
            .expect("predict");
        let by_table = fill
            .predict([image.borrow(&sq(&c0, &q0))], M)
            .expect("predict");
        assert_pred(&by_fill, &by_table, 0.0);
    }
    let a = table
        .predict(
            [image.fill(&Pairs {
                rows: &c0,
                cols: &q0,
            })],
            M,
        )
        .expect("fill");
    let b = table
        .predict([image.borrow(&sq(&c0, &q0))], M)
        .expect("table");
    assert_pred(&a, &b, 0.0);
    let mut bad = sq(&c0, &c0);
    bad[0] = 0.5;
    let diag = Gpr::new(kernel.clone(), lik())
        .with_optimizer(Fixed)
        .factor([image.from_vec(bad)], N, &y);
    assert!(matches!(diag, Err((_, GprError::ShapeMismatch { .. }))));
    let mut bad = sq(&c0, &c0);
    bad[1] += 0.25;
    let asym = Gpr::new(kernel.clone(), lik())
        .with_optimizer(Fixed)
        .factor([image.from_vec(bad)], N, &y);
    assert!(matches!(asym, Err((_, GprError::ShapeMismatch { .. }))));
    let mut nan = sq(&c0, &q0);
    nan[2] = f64::NAN;
    assert_eq!(
        table.predict([image.borrow(&nan)], M).err(),
        Some(GprError::NonFiniteInput)
    );
    assert!(matches!(
        table.predict([image.borrow(&nan[..4])], M),
        Err(GprError::LengthMismatch { .. })
    ));
    let mut query = sq(&q0, &q0);
    query[1] += 1.0;
    assert!(matches!(
        table.predict_covariance([image.borrow(&sq(&c0, &q0))], [image.borrow(&query)], M),
        Err(GprError::ShapeMismatch { .. })
    ));
}

#[test]
fn online_inserts_match_a_fit_on_the_whole_matrix() {
    let all = coord(0, N + 2, 0.0);
    let q0 = coord(0, M, 0.5);
    let y_all: Vec<f64> = (0..N + 2).map(|i| (i as f64 * 0.7).cos()).collect();
    let image = ScalarDistance::new();
    let kernel =
        ConstantKernel::new(1.5).expect("c") * image.kernel(RbfKernel::new(0.8).expect("ell"));
    let first = &all[..N];
    let fitted = Gpr::new(kernel.clone(), lik())
        .with_optimizer(Fixed)
        .factor([image.from_vec(sq(first, first))], N, &y_all[..N])
        .expect("fit");
    let mut online = fitted.into_online().expect("online");
    for k in N..N + 2 {
        let col = sq(&all[..k], &all[k..=k]);
        online
            .insert([image.from_slice(&col)], y_all[k])
            .expect("insert");
    }
    let whole = Gpr::new(kernel, lik())
        .with_optimizer(Fixed)
        .factor([image.from_vec(sq(&all, &all))], N + 2, &y_all)
        .expect("whole");
    let cross = sq(&all, &q0);
    let got = online.predict([image.borrow(&cross)], M).expect("online");
    let expect = whole.predict([image.borrow(&cross)], M).expect("whole");
    assert_pred(&got, &expect, 1e-10);
    // Refactoring the online model reads the distances it kept.
    let mut theta = vec![0.0; 3];
    online.get_params(&mut theta).expect("theta");
    online.set_params(&theta).expect("refactor");
    let got = online.predict([image.borrow(&cross)], M).expect("online");
    assert_pred(&got, &expect, 1e-10);
    // Deleting the last point is the fit on the first N + 1.
    let id = online.point_ids()[N + 1];
    online.delete(id).expect("delete");
    let part = &all[..N + 1];
    let partial = Gpr::new(
        ConstantKernel::new(1.5).expect("c") * image.kernel(RbfKernel::new(0.8).expect("ell")),
        lik(),
    )
    .with_optimizer(Fixed)
    .factor([image.from_vec(sq(part, part))], N + 1, &y_all[..N + 1])
    .expect("partial");
    let cross = sq(part, &q0);
    online.refit().expect("refit");
    let got = online.predict([image.borrow(&cross)], M).expect("online");
    let expect = partial.predict([image.borrow(&cross)], M).expect("partial");
    assert_pred(&got, &expect, 1e-10);
}

#[test]
fn hyperparameter_search_matches_the_coordinate_search() {
    let c0 = coord(0, N, 0.0);
    let q0 = coord(0, M, 0.5);
    let y = targets();
    let image = ScalarDistance::new();
    let coords = Gpr::new(KernelSpec::from(RbfKernel::new(1.0).expect("ell")), lik())
        .fit(&c0, N, 1, &y)
        .expect("coords");
    let dist = Gpr::new(image.kernel(RbfKernel::new(1.0).expect("ell")), lik())
        .fit([image.from_slice(&sq(&c0, &c0))], N, &y)
        .expect("dist");
    let (mut a, mut b) = (vec![0.0; 2], vec![0.0; 2]);
    coords.get_params(&mut a).expect("theta");
    dist.get_params(&mut b).expect("theta");
    assert_slice_close(&b, &a, 1e-6);
    let got = dist
        .predict_with(
            [image.borrow(&sq(&c0, &q0))],
            M,
            PredictOptions {
                variance_kind: VarianceKind::Latent,
            },
        )
        .expect("predict");
    let expect = coords
        .predict_with(
            &q0,
            M,
            1,
            PredictOptions {
                variance_kind: VarianceKind::Latent,
            },
        )
        .expect("predict");
    assert_pred(&got, &expect, 1e-5);
    assert_eq!(dist.slots().len(), 1);
    assert_eq!(dist.to_kernel().num_params(), 1);
}

#[test]
fn single_precision_converts_the_supplied_distances() {
    let c0 = coord(0, N, 0.0);
    let q0 = coord(0, M, 0.5);
    let y = targets();
    let image = ScalarDistance::new();
    let bands = ArdDistance::new(1).expect("dims");
    let kernel = image.kernel(RbfKernel::new(1.0).expect("ell"))
        * bands
            .kernel(RbfArdKernel::new(&[2.0]).expect("ell"))
            .expect("dims");
    let double = Gpr::new(kernel.clone(), lik())
        .with_optimizer(Fixed)
        .factor(
            [
                image.from_slice(&sq(&c0, &c0)),
                bands.from_vecs(vec![sq(&c0, &c0)]),
            ],
            N,
            &y,
        )
        .expect("f64");
    let single = Gpr::new(kernel, lik())
        .with_precision::<SinglePrecision>()
        .with_optimizer(Fixed)
        .factor(
            [
                image.from_slice(&sq(&c0, &c0)),
                bands.from_vecs(vec![sq(&c0, &c0)]),
            ],
            N,
            &y,
        )
        .expect("f32");
    let cross = sq(&c0, &q0);
    let a = double
        .predict([image.borrow(&cross), bands.borrow(&[&cross])], M)
        .expect("f64");
    let b = single
        .predict([image.borrow(&cross), bands.borrow(&[&cross])], M)
        .expect("f32");
    for (x, y) in a.mean.iter().zip(&b.mean) {
        assert!((x - f64::from(*y)).abs() < 1e-4);
    }
}

#[test]
fn mixed_precision_refines_on_the_supplied_distances() {
    use gprx::{MixedPrecision, PromoteStorage, ReevaluateKernel};
    let c0 = coord(0, N, 0.0);
    let q0 = coord(0, M, 0.5);
    let y = targets();
    let image = ScalarDistance::new();
    let kernel = image.kernel(RbfKernel::new(1.0).expect("ell"));
    let train = sq(&c0, &c0);
    let cross = sq(&c0, &q0);
    let double = Gpr::new(kernel.clone(), lik())
        .with_optimizer(Fixed)
        .factor([image.from_slice(&train)], N, &y)
        .expect("f64");
    let expect = double.predict([image.borrow(&cross)], M).expect("f64");
    let promote = Gpr::new(kernel.clone(), lik())
        .with_precision::<MixedPrecision<PromoteStorage>>()
        .with_optimizer(Fixed)
        .factor([image.from_slice(&train)], N, &y)
        .expect("promote");
    let reevaluate = Gpr::new(kernel, lik())
        .with_precision::<MixedPrecision<ReevaluateKernel>>()
        .with_optimizer(Fixed)
        .factor([image.from_slice(&train)], N, &y)
        .expect("reevaluate");
    for model_pred in [
        promote.predict([image.borrow(&cross)], M).expect("promote"),
        reevaluate
            .predict([image.borrow(&cross)], M)
            .expect("reevaluate"),
    ] {
        assert_slice_close(&model_pred.mean, &expect.mean, 1e-6);
    }
    let loo = promote.loo_predict().expect("loo");
    assert_slice_close(&loo.mean, &double.loo_predict().expect("loo").mean, 1e-4);
}

/// [`Pairs`] that counts its calls.
struct Counted<'a> {
    pairs: Pairs<'a>,
    calls: std::sync::atomic::AtomicUsize,
}

impl DistanceFill for Counted<'_> {
    fn fill(&self, n_rows: usize, n_cols: usize, out: &mut [f64]) {
        self.calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.pairs.fill(n_rows, n_cols, out);
    }
}

#[test]
fn a_fit_calls_the_fill_once_whatever_the_cache_policy() {
    let c0 = coord(0, N, 0.0);
    let image = ScalarDistance::new();
    let fill = Counted {
        pairs: Pairs {
            rows: &c0,
            cols: &c0,
        },
        calls: std::sync::atomic::AtomicUsize::new(0),
    };
    let fitted = Gpr::new(image.kernel(RbfKernel::new(1.0).expect("ell")), lik())
        .with_prefer_memory()
        .with_optimizer(Fixed)
        .factor([image.fill(&fill)], N, &targets())
        .expect("factor");
    assert_eq!(
        fitted.distance_cache_policy(),
        DistanceCachePolicy::Uncached
    );
    // The model keeps what the bind wrote, whatever the cache policy: the
    // second factor (it restores `L` over the reused `W` buffer) reads it.
    assert_eq!(fill.calls.load(std::sync::atomic::Ordering::Relaxed), 1);
}

#[test]
fn a_kernel_with_coordinate_leaves_needs_a_feature_column() {
    let c0 = coord(0, N, 0.0);
    let y = targets();
    let image = ScalarDistance::new();
    let kernel = image.kernel(RbfKernel::new(1.0).expect("ell"))
        * KernelSpec::from(RbfKernel::new(0.5).expect("ell"));
    let fit = Gpr::new(kernel.clone(), lik())
        .with_optimizer(Fixed)
        .factor([image.from_slice(&sq(&c0, &c0))], N, &[], 0, &y);
    assert!(matches!(fit, Err((_, GprError::EmptyInput))));
    let fit = Gpr::new(kernel, lik()).fit([image.from_slice(&sq(&c0, &c0))], N, &[], 0, &y);
    assert!(matches!(fit, Err((_, GprError::EmptyInput))));
}

#[test]
fn mixed_precision_refines_on_the_f64_distances_it_was_given() {
    use gprx::{MixedPrecision, PromoteStorage, ReevaluateKernel};
    // `d²` near 1e4 with fine digits: rounding it to `f32` moves `K`.
    let c0: Vec<f64> = coord(0, N, 0.0).iter().map(|v| 100.0 * v).collect();
    let q0: Vec<f64> = coord(0, M, 0.5).iter().map(|v| 100.0 * v).collect();
    let y = targets();
    let image = ScalarDistance::new();
    let kernel = image.kernel(RbfKernel::new(60.0).expect("ell"));
    let train = sq(&c0, &c0);
    let cross = sq(&c0, &q0);
    let double = Gpr::new(kernel.clone(), lik())
        .with_optimizer(Fixed)
        .factor([image.from_slice(&train)], N, &y)
        .expect("f64");
    let expect = double.predict([image.borrow(&cross)], M).expect("f64");
    let promote = Gpr::new(kernel.clone(), lik())
        .with_precision::<MixedPrecision<PromoteStorage>>()
        .with_optimizer(Fixed)
        .factor([image.from_slice(&train)], N, &y)
        .expect("promote")
        .predict([image.borrow(&cross)], M)
        .expect("promote");
    let reevaluate = Gpr::new(kernel, lik())
        .with_precision::<MixedPrecision<ReevaluateKernel>>()
        .with_optimizer(Fixed)
        .factor([image.from_slice(&train)], N, &y)
        .expect("reevaluate")
        .predict([image.borrow(&cross)], M)
        .expect("reevaluate");
    assert_slice_close(&promote.mean, &expect.mean, 1e-10);
    assert_slice_close(&reevaluate.mean, &expect.mean, 1e-10);
}

#[test]
fn a_white_term_adds_its_diagonal_to_the_query_covariance() {
    use gprx::kernel::WhiteKernel;
    let cols: Vec<Vec<f64>> = (0..2).map(|k| coord(k, N, 0.0)).collect();
    let qcols: Vec<Vec<f64>> = (0..2).map(|k| coord(k, M, 0.5)).collect();
    let x: Vec<f64> = cols.concat();
    let xs: Vec<f64> = qcols.concat();
    let y = targets();
    let white = WhiteKernel::new(0.3).expect("white");
    // A scalar slot.
    let rbf = RbfKernel::new(0.9).expect("ell");
    let coords = Gpr::new(KernelSpec::from(rbf) + KernelSpec::from(white), lik())
        .with_optimizer(Fixed)
        .factor(&x, N, 2, &y)
        .expect("coords");
    let expect = coords.predict_covariance(&xs, M, 2).expect("cov");
    let image = ScalarDistance::new();
    let dist = Gpr::new(image.kernel(rbf) + white, lik())
        .with_optimizer(Fixed)
        .factor(
            [image.from_vec(sum(&[sq(&cols[0], &cols[0]), sq(&cols[1], &cols[1])]))],
            N,
            &y,
        )
        .expect("distances");
    let cross = sum(&[sq(&cols[0], &qcols[0]), sq(&cols[1], &qcols[1])]);
    let query = sum(&[sq(&qcols[0], &qcols[0]), sq(&qcols[1], &qcols[1])]);
    let got = dist
        .predict_covariance([image.borrow(&cross)], [image.borrow(&query)], M)
        .expect("cov");
    assert_slice_close(&got.covariance, &expect.covariance, TOL);
    // An ARD slot.
    let ard = RbfArdKernel::new(&[0.8, 1.4]).expect("ell");
    let coords = Gpr::new(
        KernelSpec::from(ard.clone()) + KernelSpec::from(white),
        lik(),
    )
    .with_optimizer(Fixed)
    .factor(&x, N, 2, &y)
    .expect("coords");
    let expect = coords.predict_covariance(&xs, M, 2).expect("cov");
    let bands = ArdDistance::new(2).expect("dims");
    let dist = Gpr::new(bands.kernel(ard).expect("dims") + white, lik())
        .with_optimizer(Fixed)
        .factor(
            [bands.from_vecs(cols.iter().map(|c| sq(c, c)).collect())],
            N,
            &y,
        )
        .expect("distances");
    let cross: Vec<Vec<f64>> = cols.iter().zip(&qcols).map(|(c, q)| sq(c, q)).collect();
    let query: Vec<Vec<f64>> = qcols.iter().map(|q| sq(q, q)).collect();
    let cross: Vec<&[f64]> = cross.iter().map(Vec::as_slice).collect();
    let query: Vec<&[f64]> = query.iter().map(Vec::as_slice).collect();
    let got = dist
        .predict_covariance([bands.borrow(&cross)], [bands.borrow(&query)], M)
        .expect("cov");
    assert_slice_close(&got.covariance, &expect.covariance, TOL);
}

#[test]
fn a_negative_squared_distance_is_rejected() {
    let c0 = coord(0, N, 0.0);
    let q0 = coord(0, M, 0.5);
    let y = targets();
    let image = ScalarDistance::new();
    let kernel = image.kernel(RbfKernel::new(1.0).expect("ell"));
    let mut train = sq(&c0, &c0);
    train[1] = -1.0;
    train[N] = -1.0;
    let fit = Gpr::new(kernel.clone(), lik())
        .with_optimizer(Fixed)
        .factor([image.from_vec(train)], N, &y);
    assert!(matches!(fit, Err((_, GprError::ShapeMismatch { .. }))));
    let fitted = Gpr::new(kernel, lik())
        .with_optimizer(Fixed)
        .factor([image.from_vec(sq(&c0, &c0))], N, &y)
        .expect("fit");
    let mut cross = sq(&c0, &q0);
    cross[0] = -0.5;
    assert!(matches!(
        fitted.predict([image.borrow(&cross)], M),
        Err(GprError::ShapeMismatch { .. })
    ));
}

/// Per-dimension squared differences of two one-dimensional sample sets,
/// every dimension the same.
struct Bands<'a> {
    rows: &'a [f64],
    cols: &'a [f64],
    dims: usize,
}

impl DistanceFill for Bands<'_> {
    fn fill(&self, n_rows: usize, n_cols: usize, out: &mut [f64]) {
        let len = n_rows * n_cols;
        for k in 0..self.dims {
            Pairs {
                rows: self.rows,
                cols: self.cols,
            }
            .fill(n_rows, n_cols, &mut out[k * len..(k + 1) * len]);
        }
    }
}

#[test]
fn mixed_precision_refines_an_ard_slot_and_an_uncached_fill_fits_the_same() {
    use gprx::{MixedPrecision, ReevaluateKernel};
    let c0 = coord(0, N, 0.0);
    let q0 = coord(0, M, 0.5);
    let y = targets();
    let bands = ArdDistance::new(2).expect("dims");
    let kernel = bands
        .kernel(RbfArdKernel::new(&[0.9, 1.6]).expect("ell"))
        .expect("dims");
    let train = Bands {
        rows: &c0,
        cols: &c0,
        dims: 2,
    };
    let cross = Bands {
        rows: &c0,
        cols: &q0,
        dims: 2,
    };
    let double = Gpr::new(kernel.clone(), lik())
        .with_optimizer(Fixed)
        .factor([bands.fill(&train)], N, &y)
        .expect("f64");
    let expect = double.predict([bands.fill(&cross)], M).expect("f64");
    let mixed = Gpr::new(kernel.clone(), lik())
        .with_precision::<MixedPrecision<ReevaluateKernel>>()
        .with_optimizer(Fixed)
        .factor([bands.fill(&train)], N, &y)
        .expect("mixed");
    let got = mixed.predict([bands.fill(&cross)], M).expect("mixed");
    assert_slice_close(&got.mean, &expect.mean, 1e-10);
    // A search under the memory pole lands where the default search lands.
    let cached = Gpr::new(kernel.clone(), lik())
        .fit([bands.fill(&train)], N, &y)
        .expect("cached");
    let uncached = Gpr::new(kernel, lik())
        .with_precision::<MixedPrecision<ReevaluateKernel>>()
        .with_prefer_memory()
        .fit([bands.fill(&train)], N, &y)
        .expect("uncached");
    assert_close(
        uncached.neg_log_marginal_likelihood().expect("nlml"),
        cached.neg_log_marginal_likelihood().expect("nlml"),
        1e-4,
    );
}

#[test]
fn rounding_in_a_table_of_squared_distances_is_tidied() {
    // `‖a‖² + ‖b‖² − 2ab` of points far from the origin: the diagonal is
    // not exactly zero and some values come out slightly negative.
    let shift = 1.0e3;
    let c0: Vec<f64> = coord(0, N, 0.0).iter().map(|v| v + shift).collect();
    let q0: Vec<f64> = coord(0, M, 0.5).iter().map(|v| v + shift).collect();
    let expanded = |a: &[f64], b: &[f64], flip: bool| -> Vec<f64> {
        let mut out = Vec::with_capacity(a.len() * b.len());
        for bj in b {
            for ai in a {
                let (x, z) = if flip { (bj, ai) } else { (ai, bj) };
                out.push(x * x + z * z - 2.0 * x * z);
            }
        }
        out
    };
    let mut train = expanded(&c0, &c0, false);
    // One mirror entry from the other direction, and one rounded below zero.
    train[1] = expanded(&c0, &c0, true)[1];
    train[N + 1] = -1.0e-12;
    let cross: Vec<f64> = expanded(&c0, &q0, false)
        .iter()
        .map(|v| if v.abs() < 1.0e-9 { -1.0e-13 } else { *v })
        .collect();
    let y = targets();
    let image = ScalarDistance::new();
    let kernel = image.kernel(MaternKernel::new(1.1, MaternNu::ThreeHalves).expect("ell"));
    let tidy = Gpr::new(kernel.clone(), lik())
        .with_optimizer(Fixed)
        .factor([image.from_vec(train)], N, &y)
        .expect("rounded table");
    let exact = Gpr::new(kernel, lik())
        .with_optimizer(Fixed)
        .factor([image.from_vec(sq(&c0, &c0))], N, &y)
        .expect("exact table");
    let got = tidy.predict([image.borrow(&cross)], M).expect("predict");
    let expect = exact
        .predict([image.borrow(&sq(&c0, &q0))], M)
        .expect("predict");
    assert!(got.mean.iter().all(|v| v.is_finite()));
    assert_slice_close(&got.mean, &expect.mean, 1e-6);
    // Past the tolerance it is still an error.
    let mut wrong = sq(&c0, &c0);
    wrong[1] += 1.0e-3 * wrong.iter().fold(0.0f64, |a, v| a.max(*v));
    assert!(matches!(
        Gpr::new(image.kernel(RbfKernel::new(1.0).expect("ell")), lik())
            .with_optimizer(Fixed)
            .factor([image.from_vec(wrong)], N, &y),
        Err((_, GprError::ShapeMismatch { .. }))
    ));
}
