//! Exact GPR on supplied squared distances against the same model on
//! coordinates. Public API only.

mod common;

use common::{assert_close, assert_slice_close};
use gprx::kernel::{
    ArdDistance, DistanceFill, KernelSpec, MaternKernel, MaternNu, RbfArdKernel, RbfKernel,
    ScalarDistance,
};
use gprx::{
    DistanceCachePolicy, Fixed, GaussianLikelihood, Gpr, GprError, PredictOptions, Prediction,
    SinglePrecision, SlotErrorKind, VarianceKind,
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
    let (bands, ard) = ArdDistance::from_leaf(ard);
    let train: Vec<Vec<f64>> = cols.iter().map(|c| sq(c, c)).collect();
    let cross: Vec<Vec<f64>> = cols.iter().zip(&qcols).map(|(c, q)| sq(c, q)).collect();
    let dist = Gpr::new(ard, lik())
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
    // The gradient (one contraction over the packed `(Δ_d)²`) and the
    // Hessian of the NLML agree, at `θ` and away from it.
    let (mut coords, mut dist) = (coords, dist);
    let p = coords.num_params();
    let mut theta = vec![0.0; p];
    coords.get_params(&mut theta).expect("theta");
    for shift in [0.0, 0.3] {
        let at: Vec<f64> = theta.iter().map(|t| t + shift).collect();
        let (mut gc, mut gd) = (vec![0.0; p], vec![0.0; p]);
        let vc = coords.value_and_gradient_into(&at, &mut gc).expect("grad");
        let vd = dist.value_and_gradient_into(&at, &mut gd).expect("grad");
        assert_close(vd, vc, 1e-10);
        assert_slice_close(&gd, &gc, 1e-9);
        let (mut hc, mut hd) = (vec![0.0; p * p], vec![0.0; p * p]);
        coords.hessian_into(&at, &mut hc).expect("hess");
        dist.hessian_into(&at, &mut hd).expect("hess");
        assert_slice_close(&hd, &hc, 1e-8);
    }
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
        .predict(&[q0.clone(), q1.clone()].concat(), M, 2)
        .expect("predict");
    assert_pred(&got, &expect, TOL);
    // The second slot (`b`, place 1) with no source, then with two.
    let only_a = model.predict([a.borrow(&sq(&c0, &q0))], M);
    assert_eq!(
        only_a.map(drop),
        Err(GprError::DistanceSlot {
            kind: SlotErrorKind::Missing,
            slot: Some(1)
        })
    );
    let b_twice = model.predict(
        [
            a.borrow(&sq(&c0, &q0)),
            b.borrow(&sq(&c1, &q1)),
            b.borrow(&sq(&c1, &q1)),
        ],
        M,
    );
    assert_eq!(
        b_twice.map(drop),
        Err(GprError::DistanceSlot {
            kind: SlotErrorKind::Duplicate,
            slot: Some(1)
        })
    );
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
    let slot = |kind, slot| GprError::DistanceSlot { kind, slot };
    assert_eq!(
        twice.map(drop).map_err(|(_, e)| e),
        Err(slot(SlotErrorKind::Duplicate, Some(0)))
    );
    let foreign = ScalarDistance::new();
    let wrong = fitted.predict([foreign.borrow(&sq(&c0, &q0))], M);
    assert_eq!(wrong.map(drop), Err(slot(SlotErrorKind::NotRead, None)));
    let none = fitted.predict(Vec::new(), M);
    let missing = slot(SlotErrorKind::Missing, Some(0));
    assert_eq!(none.map(drop), Err(missing.clone()));
    assert_eq!(
        missing.to_string(),
        "distance slot mismatch in slot 0: a distance slot of the kernel has no source"
    );
}

/// Writes `d²` between two one-dimensional sample sets.
struct Pairs<'a> {
    rows: &'a [f64],
    cols: &'a [f64],
}

impl DistanceFill for Pairs<'_> {
    fn fill_column(&self, col: usize, rows: std::ops::Range<usize>, out: &mut [f64]) {
        for (slot, i) in out.iter_mut().zip(rows) {
            let diff = self.rows[i] - self.cols[col];
            *slot = diff * diff;
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
    assert!(matches!(
        diag,
        Err((
            _,
            GprError::InvalidDistance {
                pair: Some((0, 0)),
                ..
            }
        ))
    ));
    let mut bad = sq(&c0, &c0);
    bad[1] += 0.25;
    let asym = Gpr::new(kernel.clone(), lik())
        .with_optimizer(Fixed)
        .factor([image.from_vec(bad)], N, &y);
    assert!(matches!(
        asym,
        Err((
            _,
            GprError::InvalidDistance {
                pair: Some((1, 0)),
                ..
            }
        ))
    ));
    let mut nan = sq(&c0, &q0);
    nan[2] = f64::NAN;
    assert!(matches!(
        table.predict([image.borrow(&nan)], M),
        Err(GprError::InvalidDistance {
            pair: Some((2, 0)),
            ..
        })
    ));
    assert!(matches!(
        table.predict([image.borrow(&nan[..4])], M),
        Err(GprError::LengthMismatch { .. })
    ));
    let mut query = sq(&q0, &q0);
    query[1] += 1.0;
    assert!(matches!(
        table.predict_covariance([image.borrow(&sq(&c0, &q0))], [image.borrow(&query)], M),
        Err(GprError::InvalidDistance {
            pair: Some((1, 0)),
            ..
        })
    ));
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
    let (bands, ard) = ArdDistance::from_leaf(RbfArdKernel::new(&[2.0]).expect("ell"));
    let kernel = image.kernel(RbfKernel::new(1.0).expect("ell")) * ard;
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
    fn fill_column(&self, col: usize, rows: std::ops::Range<usize>, out: &mut [f64]) {
        self.calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.pairs.fill_column(col, rows, out);
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
    // Each column once: the model keeps what the fill wrote, whatever the
    // cache policy, and the second factor (it restores `L` over the reused
    // `W` buffer) reads it.
    assert_eq!(fill.calls.load(std::sync::atomic::Ordering::Relaxed), N);
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
    let (bands, ard) = ArdDistance::from_leaf(ard);
    let dist = Gpr::new(ard + white, lik())
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

/// A borrowed ARD query square that `.tidy` repairs is copied first: the
/// caller's tables stay as they were, and the repaired copy predicts as the
/// symmetric square does.
#[test]
fn a_tidied_borrowed_ard_query_square_is_repaired_on_a_copy() {
    let cols = [coord(0, N, 0.0), coord(1, N, 0.3)];
    let qcols = [coord(0, M, 0.5), coord(1, M, 0.2)];
    let y = targets();
    let (bands, ard) = ArdDistance::from_leaf(RbfArdKernel::new(&[0.8, 1.4]).expect("ell"));
    let dist = Gpr::new(ard, lik())
        .with_optimizer(Fixed)
        .factor(
            [bands.from_vecs(cols.iter().map(|c| sq(c, c)).collect())],
            N,
            &y,
        )
        .expect("distances");
    let cross: Vec<Vec<f64>> = cols.iter().zip(&qcols).map(|(c, q)| sq(c, q)).collect();
    let cross: Vec<&[f64]> = cross.iter().map(Vec::as_slice).collect();
    let exact: Vec<Vec<f64>> = qcols.iter().map(|q| sq(q, q)).collect();
    let mut skewed = exact.clone();
    // One pair of the second dimension off by a relative 1e-9.
    skewed[1][1] *= 1.0 + 1e-9;
    let before = skewed.clone();
    let exact: Vec<&[f64]> = exact.iter().map(Vec::as_slice).collect();
    let skewed_refs: Vec<&[f64]> = skewed.iter().map(Vec::as_slice).collect();
    assert!(
        dist.predict_covariance([bands.borrow(&cross)], [bands.borrow(&skewed_refs)], M)
            .is_err()
    );
    let want = dist
        .predict_covariance([bands.borrow(&cross)], [bands.borrow(&exact)], M)
        .expect("cov");
    let got = dist
        .predict_covariance(
            [bands.borrow(&cross)],
            [bands.borrow(&skewed_refs).tidy(1e-6).expect("tol")],
            M,
        )
        .expect("cov");
    assert_slice_close(&got.covariance, &want.covariance, 1e-8);
    assert_eq!(skewed, before);
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
    assert!(matches!(
        fit,
        Err((
            _,
            GprError::InvalidDistance {
                pair: Some((1, 0)),
                ..
            }
        ))
    ));
    let fitted = Gpr::new(kernel, lik())
        .with_optimizer(Fixed)
        .factor([image.from_vec(sq(&c0, &c0))], N, &y)
        .expect("fit");
    let mut cross = sq(&c0, &q0);
    cross[0] = -0.5;
    assert!(matches!(
        fitted.predict([image.borrow(&cross)], M),
        Err(GprError::InvalidDistance {
            pair: Some((0, 0)),
            ..
        })
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
    fn fill_column(&self, col: usize, rows: std::ops::Range<usize>, out: &mut [f64]) {
        let len = rows.len();
        for k in 0..self.dims {
            Pairs {
                rows: self.rows,
                cols: self.cols,
            }
            .fill_column(col, rows.clone(), &mut out[k * len..(k + 1) * len]);
        }
    }
}

#[test]
fn mixed_precision_refines_an_ard_slot_and_an_uncached_fill_fits_the_same() {
    use gprx::{MixedPrecision, ReevaluateKernel};
    let c0 = coord(0, N, 0.0);
    let q0 = coord(0, M, 0.5);
    let y = targets();
    let (bands, kernel) = ArdDistance::from_leaf(RbfArdKernel::new(&[0.9, 1.6]).expect("ell"));

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
fn rounding_is_refused_exactly_and_repaired_on_request() {
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
    // Without a repair, the rounded table is refused.
    assert!(matches!(
        Gpr::new(kernel.clone(), lik())
            .with_optimizer(Fixed)
            .factor([image.borrow(&train)], N, &y),
        Err((_, GprError::InvalidDistance { .. }))
    ));
    let tidy = Gpr::new(kernel.clone(), lik())
        .with_optimizer(Fixed)
        .factor([image.from_vec(train).tidy(1e-6).expect("tol")], N, &y)
        .expect("rounded table");
    let exact = Gpr::new(kernel, lik())
        .with_optimizer(Fixed)
        .factor([image.from_vec(sq(&c0, &c0))], N, &y)
        .expect("exact table");
    let got = tidy
        .predict([image.borrow(&cross).tidy(1e-6).expect("tol")], M)
        .expect("predict");
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
            .factor([image.from_vec(wrong).tidy(1e-6).expect("tol")], N, &y),
        Err((
            _,
            GprError::InvalidDistance {
                pair: Some((1, 0)),
                ..
            }
        ))
    ));
    // A tolerance is finite and non-negative.
    assert!(matches!(
        image.borrow(&cross).tidy(-1.0),
        Err(GprError::InvalidConfig { .. })
    ));
    assert!(image.borrow(&cross).tidy(f64::NAN).is_err());
    // At `1` and past it any table would pass as rounding.
    for rel_tol in [1.0, 2.0, f64::INFINITY] {
        assert!(matches!(
            image.borrow(&cross).tidy(rel_tol),
            Err(GprError::InvalidConfig { .. })
        ));
    }
}

/// Fits an ARD distance model of the slot `bands` at `P` on two dimensions and predicts on
/// cross blocks that hold `bad` at `(row, col)` of dimension `dim`: every
/// predict entry point reports it there.
#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn ard_query_value_is_reported<P: gprx::GpScalar>(
    (bands, kernel): (ArdDistance, gprx::kernel::DistanceKernel),
    bad: f64,
    (dim, row, col): (usize, usize, usize),
) {
    let cols: Vec<Vec<f64>> = (0..2).map(|k| coord(k, N, 0.0)).collect();
    let qcols: Vec<Vec<f64>> = (0..2).map(|k| coord(k, M, 0.5)).collect();
    let train: Vec<Vec<f64>> = cols.iter().map(|c| sq(c, c)).collect();
    let mut cross: Vec<Vec<f64>> = cols.iter().zip(&qcols).map(|(c, q)| sq(c, q)).collect();
    let mut fitted = Gpr::new(kernel, lik())
        .with_precision::<P>()
        .with_optimizer(Fixed)
        .factor([bands.from_vecs(train)], N, &targets())
        .map_err(|(_, e)| e)
        .expect("fit");
    let square: Vec<Vec<f64>> = qcols.iter().map(|q| sq(q, q)).collect();
    let square_refs: Vec<&[f64]> = square.iter().map(Vec::as_slice).collect();
    // A valid table predicts, then the same model refuses the bad one.
    let refs: Vec<&[f64]> = cross.iter().map(Vec::as_slice).collect();
    fitted.predict([bands.borrow(&refs)], M).expect("valid");
    cross[dim][row + col * N] = bad;
    let refs: Vec<&[f64]> = cross.iter().map(Vec::as_slice).collect();
    let at = |result: Result<(), GprError>, call: &str| {
        assert!(
            matches!(
                result,
                Err(GprError::InvalidDistance { pair: Some((r, c)), .. }) if (r, c) == (row, col)
            ),
            "{call} of {bad} at {:?}: {result:?}",
            (row, col)
        );
    };
    at(
        fitted.predict([bands.borrow(&refs)], M).map(drop),
        "predict",
    );
    let mut out = Prediction::default();
    at(
        fitted.predict_into([bands.borrow(&refs)], M, &mut out),
        "predict_into",
    );
    at(
        fitted
            .predict_covariance([bands.borrow(&refs)], [bands.borrow(&square_refs)], M)
            .map(drop),
        "predict_covariance",
    );
    // The model is unchanged: the valid table predicts again.
    cross[dim][row + col * N] = 0.25;
    let refs: Vec<&[f64]> = cross.iter().map(Vec::as_slice).collect();
    fitted
        .predict_into([bands.borrow(&refs)], M, &mut out)
        .expect("valid again");
}

/// A prediction block of an ARD slot is checked as the kernel reads it
/// (one pass over the caller's values), in place for `f64` and as it is
/// cast otherwise: a negative, `NaN`, or infinite value is reported at its
/// place in the caller's table, by every ARD leaf and precision.
#[test]
fn an_invalid_ard_query_value_is_reported_where_it_is() {
    use gprx::kernel::{MaternArdKernel, RationalQuadraticArdKernel};
    use gprx::{DoublePrecision, MixedPrecision, ReevaluateKernel};
    let (bands, _) = ArdDistance::from_leaf(RbfArdKernel::new(&[0.9, 1.6]).expect("ell"));
    let kernels = || {
        [
            bands
                .kernel(RbfArdKernel::new(&[0.8, 1.4]).expect("ell"))
                .expect("dims"),
            bands
                .kernel(MaternArdKernel::new(&[0.8, 1.4], MaternNu::FiveHalves).expect("ell"))
                .expect("dims"),
            bands
                .kernel(RationalQuadraticArdKernel::new(&[0.8, 1.4], 1.5).expect("ell"))
                .expect("dims"),
        ]
    };
    for bad in [-0.5, f64::NAN, f64::INFINITY] {
        for place in [(0, 0, 0), (1, 4, 2), (0, N - 1, M - 1)] {
            for kernel in kernels() {
                ard_query_value_is_reported::<DoublePrecision>((bands, kernel), bad, place);
            }
            for kernel in kernels() {
                ard_query_value_is_reported::<SinglePrecision>((bands, kernel), bad, place);
            }
            for kernel in kernels() {
                ard_query_value_is_reported::<MixedPrecision<ReevaluateKernel>>(
                    (bands, kernel),
                    bad,
                    place,
                );
            }
        }
    }
}

/// Predicts `sources` into `out` on `model` and checks every value against
/// `expect` (the first `m` queries of it).
#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn predict_into_matches<P: gprx::GpScalar>(
    model: &mut gprx::FittedGpr<Fixed, P, gprx::kernel::DistanceKernel>,
    source: gprx::kernel::DistanceSource<'_>,
    m: usize,
    out: &mut Prediction<P::Refine>,
    expect: &Prediction<P::Refine>,
    tol: f64,
) where
    P::Refine: gprx::kernel::KernelScalar,
{
    use gprx::kernel::KernelScalar;
    model.predict_into([source], m, out).expect("predict");
    assert_eq!(out.mean.len(), m);
    for i in 0..m {
        assert_close(out.mean[i].to_f64(), expect.mean[i].to_f64(), tol);
        assert_close(out.variance[i].to_f64(), expect.variance[i].to_f64(), tol);
    }
}

/// One model predicts into one `out` from every kind of source, at two
/// query counts, in any order: the buffers it keeps between calls hold no
/// state. A borrowed table that a repair changes is copied, not written.
#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn every_source_kind_predicts_alike<P: gprx::GpScalar>(tol: f64)
where
    P::Refine: gprx::kernel::KernelScalar,
{
    let c0 = coord(0, N, 0.0);
    let q0 = coord(0, M, 0.5);
    let y = targets();
    let cross = sq(&c0, &q0);
    // Rounding a repair takes back: one value slightly negative.
    let mut rounded = cross.clone();
    let at = 1 + N;
    rounded[at] = -1e-14;
    let mut repaired = cross.clone();
    repaired[at] = 0.0;
    let pairs = Pairs {
        rows: &c0,
        cols: &q0,
    };
    // Scalar slot.
    let image = ScalarDistance::new();
    let mut scalar = Gpr::new(image.kernel(RbfKernel::new(1.1).expect("ell")), lik())
        .with_precision::<P>()
        .with_optimizer(Fixed)
        .factor([image.from_vec(sq(&c0, &c0))], N, &y)
        .map_err(|(_, e)| e)
        .expect("scalar");
    let expect = scalar
        .predict([image.from_slice(&cross)], M)
        .expect("expect");
    let fixed = scalar
        .predict([image.from_slice(&repaired)], M)
        .expect("repaired");
    let mut out = Prediction::default();
    for _ in 0..2 {
        predict_into_matches(&mut scalar, image.borrow(&cross), M, &mut out, &expect, tol);
        predict_into_matches(&mut scalar, image.fill(&pairs), M, &mut out, &expect, tol);
        predict_into_matches(
            &mut scalar,
            image.borrow(&cross[..N]),
            1,
            &mut out,
            &expect,
            tol,
        );
        predict_into_matches(
            &mut scalar,
            image.from_vec(cross.clone()),
            M,
            &mut out,
            &expect,
            tol,
        );
        let tidy = image.borrow(&rounded).tidy(1e-6).expect("tol");
        predict_into_matches(&mut scalar, tidy, M, &mut out, &fixed, tol);
    }
    assert!(rounded[at] < 0.0, "a borrowed table is not written");
    // ARD slot: two dimensions with the same differences.
    let (bands, ard) = ArdDistance::from_leaf(RbfArdKernel::new(&[0.9, 1.6]).expect("ell"));
    let mut ard = Gpr::new(ard, lik())
        .with_precision::<P>()
        .with_optimizer(Fixed)
        .factor([bands.from_vecs(vec![sq(&c0, &c0); 2])], N, &y)
        .map_err(|(_, e)| e)
        .expect("ard");
    let expect = ard
        .predict([bands.from_vecs(vec![cross.clone(); 2])], M)
        .expect("expect");
    let fixed = ard
        .predict([bands.from_vecs(vec![repaired.clone(), cross.clone()])], M)
        .expect("repaired");
    let fill = Bands {
        rows: &c0,
        cols: &q0,
        dims: 2,
    };
    let both: [&[f64]; 2] = [&cross, &cross];
    let first: [&[f64]; 2] = [&cross[..N], &cross[..N]];
    let tidied: [&[f64]; 2] = [&rounded, &cross];
    for _ in 0..2 {
        predict_into_matches(&mut ard, bands.borrow(&both), M, &mut out, &expect, tol);
        predict_into_matches(&mut ard, bands.fill(&fill), M, &mut out, &expect, tol);
        predict_into_matches(&mut ard, bands.borrow(&first), 1, &mut out, &expect, tol);
        predict_into_matches(
            &mut ard,
            bands.from_slices(&both),
            M,
            &mut out,
            &expect,
            tol,
        );
        let tidy = bands.borrow(&tidied).tidy(1e-6).expect("tol");
        predict_into_matches(&mut ard, tidy, M, &mut out, &fixed, tol);
    }
    assert!(rounded[at] < 0.0, "a borrowed table is not written");
}

#[test]
fn every_source_kind_predicts_alike_at_each_precision() {
    use gprx::{DoublePrecision, MixedPrecision, ReevaluateKernel};
    every_source_kind_predicts_alike::<DoublePrecision>(1e-12);
    every_source_kind_predicts_alike::<SinglePrecision>(1e-5);
    every_source_kind_predicts_alike::<MixedPrecision<ReevaluateKernel>>(1e-12);
}

/// A value finite in `f64` but past the range of `f32` is reported where it
/// is, for the training square and a prediction block, by an `f32` and a
/// mixed model, on a scalar and an ARD slot; an `f64` model accepts the table.
#[test]
fn a_value_past_the_storage_range_is_reported_where_it_is() {
    use gprx::{DoublePrecision, MixedPrecision, ReevaluateKernel};
    fn check<P: gprx::GpScalar>(narrow: bool) {
        let c0 = coord(0, N, 0.0);
        let q0 = coord(0, M, 0.5);
        let y = targets();
        let huge = 1e300;
        let mut train = sq(&c0, &c0);
        train[2 + 4 * N] = huge;
        train[4 + 2 * N] = huge;
        let mut cross = sq(&c0, &q0);
        cross[3 + N] = huge;
        let image = ScalarDistance::new();
        let (bands, ard) = ArdDistance::from_leaf(RbfArdKernel::new(&[0.9, 1.6]).expect("ell"));
        let scalar = || image.kernel(RbfKernel::new(1.0).expect("ell"));
        let ard = || ard.clone();
        let at = |r: Result<(), GprError>, row: usize, col: usize, what: &str| {
            if narrow {
                assert!(
                    matches!(r, Err(GprError::InvalidDistance { pair: Some((a, b)), .. }) if (a, b) == (row, col)),
                    "{what}: {r:?}"
                );
            } else {
                // In range for `f64`: no table error (a pair at `d² = 1e300`
                // may still leave the Gram short of definite).
                assert!(
                    !matches!(r, Err(GprError::InvalidDistance { .. })),
                    "{what}: {r:?}"
                );
            }
        };
        let fit = |k: gprx::kernel::DistanceKernel, s: gprx::kernel::DistanceSource<'_>| {
            Gpr::new(k, lik())
                .with_precision::<P>()
                .with_optimizer(Fixed)
                .factor([s], N, &y)
                .map(drop)
                .map_err(|(_, e)| e)
        };
        at(fit(scalar(), image.from_slice(&train)), 4, 2, "scalar fit");
        at(
            fit(ard(), bands.from_vecs(vec![sq(&c0, &c0), train.clone()])),
            4,
            2,
            "ard fit",
        );
        let model = Gpr::new(scalar(), lik())
            .with_precision::<P>()
            .with_optimizer(Fixed)
            .factor([image.from_vec(sq(&c0, &c0))], N, &y)
            .map_err(|(_, e)| e)
            .expect("fit");
        at(
            model.predict([image.borrow(&cross)], M).map(drop),
            3,
            1,
            "scalar predict",
        );
        let model = Gpr::new(ard(), lik())
            .with_precision::<P>()
            .with_optimizer(Fixed)
            .factor([bands.from_vecs(vec![sq(&c0, &c0); 2])], N, &y)
            .map_err(|(_, e)| e)
            .expect("fit");
        let ok = sq(&c0, &q0);
        let blocks: [&[f64]; 2] = [&ok, &cross];
        at(
            model.predict([bands.borrow(&blocks)], M).map(drop),
            3,
            1,
            "ard predict",
        );
    }
    check::<SinglePrecision>(true);
    check::<MixedPrecision<ReevaluateKernel>>(true);
    check::<DoublePrecision>(false);
}

/// Reads `d²` squares through a fill, column runs from the diagonal down:
/// `values` holds one `rows × rows` square per dimension, or one square
/// that every dimension reads.
struct Table<'a> {
    values: &'a [f64],
    rows: usize,
    dims: usize,
}

impl DistanceFill for Table<'_> {
    fn fill_column(&self, col: usize, rows: std::ops::Range<usize>, out: &mut [f64]) {
        let len = rows.len();
        for k in 0..self.dims {
            for (slot, i) in out[k * len..(k + 1) * len].iter_mut().zip(rows.clone()) {
                let square = (k * self.rows * self.rows) % self.values.len();
                *slot = self.values[square + i + col * self.rows];
            }
        }
    }
}

/// A rounded training square is refused exactly and repaired on request
/// whichever way it arrives (a scalar or an ARD table, a borrowed one, or a
/// fill), and past the tolerance it is refused there too.
#[test]
fn a_rounded_square_is_repaired_alike_from_every_source() {
    let c0 = coord(0, N, 0.0);
    let q0 = coord(0, M, 0.5);
    let y = targets();
    let exact = sq(&c0, &c0);
    let mut rounded = exact.clone();
    // A mirror pair a little apart, whose mean is the exact value, and a
    // diagonal a little off zero.
    rounded[2 + 3 * N] += 1e-13;
    rounded[3 + 2 * N] -= 1e-13;
    rounded[4 + 4 * N] = 1e-13;
    let mut far = exact.clone();
    far[1] = -0.5;
    far[N] = -0.5;
    let mut far_diag = exact.clone();
    far_diag[2 + 2 * N] = 0.5;
    let cross = sq(&c0, &q0);
    let image = ScalarDistance::new();
    let (bands, ard) = ArdDistance::from_leaf(RbfArdKernel::new(&[0.9, 1.6]).expect("ell"));
    let scalar = image.kernel(RbfKernel::new(1.0).expect("ell"));

    fn fit(
        kernel: &gprx::kernel::DistanceKernel,
        source: gprx::kernel::DistanceSource<'_>,
        y: &[f64],
    ) -> Result<gprx::FittedGpr<Fixed, gprx::DoublePrecision, gprx::kernel::DistanceKernel>, GprError>
    {
        Gpr::new(kernel.clone(), lik())
            .with_optimizer(Fixed)
            .factor([source], N, y)
            .map_err(|(_, e)| e)
    }
    let fill = |values: &[f64], dims| Table {
        values: values.to_vec().leak(),
        rows: N,
        dims,
    };
    let (rounded, far, far_diag) = (&rounded[..], &far[..], &far_diag[..]);
    // Scalar: the fill path.
    let want = fit(&scalar, image.from_vec(exact.clone()), &y)
        .expect("exact")
        .predict([image.borrow(&cross)], M)
        .expect("predict");
    let t = fill(rounded, 1);
    assert!(matches!(
        fit(&scalar, image.fill(&t), &y),
        Err(GprError::InvalidDistance {
            pair: Some((4, 4)),
            ..
        })
    ));
    let got = fit(&scalar, image.fill(&t).tidy(1e-6).expect("tol"), &y)
        .expect("repaired fill")
        .predict([image.borrow(&cross)], M)
        .expect("predict");
    assert_pred(&got, &want, 1e-9);
    for (bad, row, col) in [(far, 1, 0), (far_diag, 2, 2)] {
        let t = fill(bad, 1);
        assert!(matches!(
            fit(&scalar, image.fill(&t).tidy(1e-6).expect("tol"), &y),
            Err(GprError::InvalidDistance { pair: Some((r, c)), .. }) if (r, c) == (row, col)
        ));
        assert!(matches!(
            fit(&scalar, image.fill(&t), &y),
            Err(GprError::InvalidDistance { .. })
        ));
    }
    // ARD: owned, borrowed, and filled tables.
    let both: [&[f64]; 2] = [&exact, &exact];
    let want = fit(&ard, bands.borrow(&both), &y)
        .expect("exact")
        .predict([bands.from_vecs(vec![cross.clone(); 2])], M)
        .expect("predict");
    let mixed: [&[f64]; 2] = [&exact, rounded];
    let t = fill(rounded, 2);
    for source in [
        bands.from_vecs(vec![exact.clone(), rounded.to_vec()]),
        bands.borrow(&mixed),
        bands.fill(&t),
    ] {
        let got = fit(&ard, source.tidy(1e-6).expect("tol"), &y)
            .expect("repaired")
            .predict([bands.from_vecs(vec![cross.clone(); 2])], M)
            .expect("predict");
        assert_pred(&got, &want, 1e-9);
    }
    let t = fill(far, 2);
    let far_pair: [&[f64]; 2] = [&exact, far];
    for source in [
        bands.from_vecs(vec![exact.clone(), far.to_vec()]),
        bands.borrow(&far_pair),
        bands.fill(&t),
    ] {
        assert!(matches!(
            fit(&ard, source.tidy(1e-6).expect("tol"), &y),
            Err(GprError::InvalidDistance {
                pair: Some((1, 0)),
                ..
            })
        ));
    }
    assert!(rounded[4 + 4 * N] > 0.0, "a borrowed table is not written");
}

/// A slot gets as many tables as it has blocks: one for a scalar slot, `d`
/// for an ARD slot, at fit and at predict.
#[test]
fn a_slot_takes_as_many_tables_as_it_has_blocks() {
    let c0 = coord(0, N, 0.0);
    let q0 = coord(0, M, 0.5);
    let y = targets();
    let image = ScalarDistance::new();
    let (bands, ard) = ArdDistance::from_leaf(RbfArdKernel::new(&[0.9, 1.6]).expect("ell"));
    let train = sq(&c0, &c0);
    let cross = sq(&c0, &q0);
    let ard = Gpr::new(ard, lik()).with_optimizer(Fixed);
    for source in [
        bands.from_vecs(vec![train.clone()]),
        bands.from_vecs(vec![train.clone(); 3]),
        bands.from_slices(&[&train]),
    ] {
        assert!(matches!(
            ard.clone().factor([source], N, &y),
            Err((_, GprError::LengthMismatch { .. }))
        ));
    }
    let model = ard
        .factor([bands.from_vecs(vec![train.clone(); 2])], N, &y)
        .expect("fit");
    let one: [&[f64]; 1] = [&cross];
    assert!(matches!(
        model.predict([bands.borrow(&one)], M),
        Err(GprError::LengthMismatch { .. })
    ));
    let _ = image;
}

/// ARD predictions on supplied blocks match the coordinate model whatever
/// the number of dimensions (past the ones the kernel keeps at hand) and
/// queries (many per worker, so the columns are pipelined), including a
/// value checked in a later column and a covariance whose query square is
/// a fill.
#[test]
fn ard_predictions_match_coordinates_at_many_dimensions_and_queries() {
    let m = 23;
    for dims in [3, 17] {
        let cols: Vec<Vec<f64>> = (0..dims).map(|k| coord(k, N, 0.0)).collect();
        let qcols: Vec<Vec<f64>> = (0..dims).map(|k| coord(k, m, 0.5)).collect();
        let x: Vec<f64> = cols.concat();
        let xs: Vec<f64> = qcols.concat();
        let y = targets();
        let ell: Vec<f64> = (0..dims).map(|k| 1.5 + 0.3 * k as f64).collect();
        let ard = RbfArdKernel::new(&ell).expect("ell");
        let coords = Gpr::new(KernelSpec::from(ard.clone()), lik())
            .with_optimizer(Fixed)
            .factor(&x, N, dims, &y)
            .expect("coords");
        let (bands, ard) = ArdDistance::from_leaf(ard);
        let train: Vec<Vec<f64>> = cols.iter().map(|c| sq(c, c)).collect();
        let mut cross: Vec<Vec<f64>> = cols.iter().zip(&qcols).map(|(c, q)| sq(c, q)).collect();
        let mut dist = Gpr::new(ard, lik())
            .with_optimizer(Fixed)
            .factor([bands.from_vecs(train)], N, &y)
            .expect("distances");
        let refs: Vec<&[f64]> = cross.iter().map(Vec::as_slice).collect();
        let mut got = Prediction::default();
        dist.predict_into([bands.borrow(&refs)], m, &mut got)
            .expect("predict");
        let expect = coords.predict(&xs, m, dims).expect("predict");
        assert_pred(&got, &expect, TOL);
        // The query square through a fill (every dimension of query 0 → its own).
        let squares: Vec<f64> = qcols.iter().flat_map(|q| sq(q, q)).collect();
        let fill = Table {
            values: squares.leak(),
            rows: m,
            dims,
        };
        let cov = dist
            .predict_covariance([bands.borrow(&refs)], [bands.fill(&fill)], m)
            .expect("covariance");
        let want = coords.predict_covariance(&xs, m, dims).expect("covariance");
        assert_slice_close(&cov.covariance, &want.covariance, 1e-9);
        // A bad value in a later column is found where it is.
        cross[dims - 1][4 + 17 * N] = f64::NAN;
        let refs: Vec<&[f64]> = cross.iter().map(Vec::as_slice).collect();
        assert!(matches!(
            dist.predict_into([bands.borrow(&refs)], m, &mut got),
            Err(GprError::InvalidDistance {
                pair: Some((4, 17)),
                ..
            })
        ));
    }
}

/// The query square of a covariance is read where it was bound: a borrowed,
/// moved, filled, or repaired square gives the coordinate model's
/// covariance, on a scalar and an ARD slot, at every precision.
#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn covariance_from_every_square<P: gprx::GpScalar>(tol: f64)
where
    P::Refine: gprx::kernel::KernelScalar,
{
    use gprx::kernel::KernelScalar;
    let m = 5;
    let dims = 2;
    let cols: Vec<Vec<f64>> = (0..dims).map(|k| coord(k, N, 0.0)).collect();
    let qcols: Vec<Vec<f64>> = (0..dims).map(|k| coord(k, m, 0.5)).collect();
    let y = targets();
    let close = |a: &[P::Refine], b: &[P::Refine]| {
        for (x, z) in a.iter().zip(b) {
            assert_close(x.to_f64(), z.to_f64(), tol);
        }
    };
    // Scalar slot on the first coordinate.
    let image = ScalarDistance::new();
    let coords = Gpr::new(KernelSpec::from(RbfKernel::new(1.2).expect("ell")), lik())
        .with_precision::<P>()
        .with_optimizer(Fixed)
        .factor(&cols[0], N, 1, &y)
        .map_err(|(_, e)| e)
        .expect("coords");
    let want = coords
        .predict_covariance(&qcols[0], m, 1)
        .expect("covariance")
        .covariance;
    let dist = Gpr::new(image.kernel(RbfKernel::new(1.2).expect("ell")), lik())
        .with_precision::<P>()
        .with_optimizer(Fixed)
        .factor([image.from_vec(sq(&cols[0], &cols[0]))], N, &y)
        .map_err(|(_, e)| e)
        .expect("distances");
    let cross = sq(&cols[0], &qcols[0]);
    let square = sq(&qcols[0], &qcols[0]);
    let mut rounded = square.clone();
    rounded[1 + 2 * m] += 1e-14;
    rounded[2 + m] -= 1e-14;
    let pairs = Pairs {
        rows: &qcols[0],
        cols: &qcols[0],
    };
    for sq_source in [
        image.borrow(&square),
        image.from_vec(square.clone()),
        image.fill(&pairs),
        image.borrow(&rounded).tidy(1e-9).expect("tol"),
    ] {
        let got = dist
            .predict_covariance([image.borrow(&cross)], [sq_source], m)
            .expect("covariance");
        close(&got.covariance, &want);
    }
    assert!(
        rounded[1 + 2 * m] > square[1 + 2 * m],
        "a borrowed square is not written"
    );
    // ARD slot on both coordinates.
    let x: Vec<f64> = cols.concat();
    let xs: Vec<f64> = qcols.concat();
    let ard = RbfArdKernel::new(&[0.9, 1.6]).expect("ell");
    let coords = Gpr::new(KernelSpec::from(ard.clone()), lik())
        .with_precision::<P>()
        .with_optimizer(Fixed)
        .factor(&x, N, dims, &y)
        .map_err(|(_, e)| e)
        .expect("coords");
    let want = coords
        .predict_covariance(&xs, m, dims)
        .expect("covariance")
        .covariance;
    let (bands, ard) = ArdDistance::from_leaf(ard);
    let dist = Gpr::new(ard, lik())
        .with_precision::<P>()
        .with_optimizer(Fixed)
        .factor(
            [bands.from_vecs(cols.iter().map(|c| sq(c, c)).collect())],
            N,
            &y,
        )
        .map_err(|(_, e)| e)
        .expect("distances");
    let cross: Vec<Vec<f64>> = cols.iter().zip(&qcols).map(|(c, q)| sq(c, q)).collect();
    let squares: Vec<Vec<f64>> = qcols.iter().map(|q| sq(q, q)).collect();
    let cross_refs: Vec<&[f64]> = cross.iter().map(Vec::as_slice).collect();
    let square_refs: Vec<&[f64]> = squares.iter().map(Vec::as_slice).collect();
    let fill = Table {
        values: squares.concat().leak(),
        rows: m,
        dims,
    };
    for sq_source in [
        bands.borrow(&square_refs),
        bands.from_vecs(squares.clone()),
        bands.fill(&fill),
    ] {
        let got = dist
            .predict_covariance([bands.borrow(&cross_refs)], [sq_source], m)
            .expect("covariance");
        close(&got.covariance, &want);
    }
}

#[test]
fn a_covariance_reads_its_query_square_where_it_was_bound() {
    use gprx::{DoublePrecision, MixedPrecision, ReevaluateKernel};
    covariance_from_every_square::<DoublePrecision>(1e-9);
    covariance_from_every_square::<SinglePrecision>(1e-4);
    covariance_from_every_square::<MixedPrecision<ReevaluateKernel>>(1e-4);
}

/// An invalid value names its slot (its place in `slots()`), its ARD
/// dimension, and its pair: in a training square, in a prediction's block
/// read unchecked as the kernel reads it, and in the message.
#[test]
fn an_invalid_distance_names_its_slot_dimension_and_pair() {
    let image = ScalarDistance::new();
    let (bands, ard) = ArdDistance::from_leaf(RbfArdKernel::new(&[0.9, 1.6]).expect("ell"));
    let kernel = image.kernel(RbfKernel::new(1.0).expect("ell")) * ard;
    assert_eq!(kernel.slots().len(), 2);
    let (a, b) = ([0.0, 1.0, 2.5], [0.5, -1.0, 2.0]);
    let y = [0.1, 0.4, -0.2];
    let trainer = Gpr::new(kernel, lik()).with_optimizer(Fixed);
    let train = |second: Vec<f64>| {
        [
            image.from_vec(sq(&a, &a)),
            bands.from_vecs(vec![sq(&b, &b), second]),
        ]
    };
    let mut bad = sq(&a, &a);
    bad[2 + 3] = -1.0;
    let refused = trainer.clone().factor(train(bad), 3, &y);
    let Err((_, err)) = refused else {
        panic!("a negative distance was accepted");
    };
    assert!(
        matches!(
            &err,
            GprError::InvalidDistance {
                slot: Some(1),
                dim: Some(1),
                pair: Some((2, 1)),
                ..
            }
        ),
        "{err:?}"
    );
    assert!(
        err.to_string()
            .starts_with("invalid squared distance in slot 1, dimension 1 at (2, 1): "),
        "{err}"
    );
    let fitted = trainer
        .factor(train(sq(&a, &a)), 3, &y)
        .map_err(|(_, e)| e)
        .expect("factor");
    // A prediction's ARD block is borrowed and checked as it is read.
    let q = [0.25, 1.5];
    let mut cross = sq(&a, &q);
    cross[4] = f64::NAN;
    let (first, second) = (sq(&b, &q), cross);
    let tables: [&[f64]; 2] = [&first, &second];
    let got = fitted.predict([image.from_vec(sq(&a, &q)), bands.from_slices(&tables)], 2);
    assert!(
        matches!(
            got,
            Err(GprError::InvalidDistance {
                slot: Some(1),
                dim: Some(1),
                pair: Some((1, 1)),
                ..
            })
        ),
        "{got:?}"
    );
}
