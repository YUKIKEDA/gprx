//! SGPR and SVGP on supplied squared distances against the same models on
//! coordinates, with the inducing points at the coordinates of the
//! training samples they name. Public API only.

mod common;

use common::{assert_close, assert_slice_close};
use gprx::kernel::{
    ArdDistance, KernelScalar, KernelSpec, RbfArdKernel, RbfKernel, ScalarDistance,
};
use gprx::{
    Adam, DoublePrecision, Fixed, GaussianLikelihood, GpScalar, GprError, MixedPrecision,
    PredictOptions, Prediction, PromoteStorage, Sgpr, SinglePrecision, Svgp, VarianceKind,
};
use std::num::{NonZeroU64, NonZeroUsize};

const N: usize = 10;
const Q: usize = 4;
const INDUCING: [usize; 4] = [7, 0, 4, 9];

/// Coordinate `k` of the training (`N`) or query (`Q`) samples.
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

/// The rows `INDUCING` of a column.
fn at_inducing(col: &[f64]) -> Vec<f64> {
    INDUCING.iter().map(|&i| col[i]).collect()
}

fn targets() -> Vec<f64> {
    (0..N).map(|i| (i as f64 * 0.7).cos()).collect()
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn lik() -> GaussianLikelihood {
    GaussianLikelihood::new(0.05).expect("noise")
}

fn to64<T: KernelScalar>(values: &[T]) -> Vec<f64> {
    values.iter().map(|v| v.to_f64()).collect()
}

fn assert_pred<T: KernelScalar>(got: &Prediction<T>, expect: &Prediction<T>, tol: f64) {
    assert_slice_close(&to64(&got.mean), &to64(&expect.mean), tol);
    assert_slice_close(&to64(&got.variance), &to64(&expect.variance), tol);
}

/// The coordinates and supplied blocks of one kernel: training columns
/// `cols` (the coordinate model reads all of them), query columns `qcols`,
/// and how the distance model reads them.
struct Case {
    cols: Vec<Vec<f64>>,
    qcols: Vec<Vec<f64>>,
}

impl Case {
    fn new(d: usize) -> Self {
        Self {
            cols: (0..d).map(|k| coord(k, N, 0.0)).collect(),
            qcols: (0..d).map(|k| coord(k, Q, 0.5)).collect(),
        }
    }

    fn x(&self) -> Vec<f64> {
        self.cols.concat()
    }

    fn z(&self) -> Vec<f64> {
        self.cols
            .iter()
            .map(|c| at_inducing(c))
            .collect::<Vec<_>>()
            .concat()
    }

    fn xs(&self) -> Vec<f64> {
        self.qcols.concat()
    }

    /// Per dimension `k`: the `N × m` training block, the `m × Q` cross
    /// block, and the `Q × Q` query square.
    fn blocks(&self, k: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let (c, q) = (&self.cols[k], &self.qcols[k]);
        let z = at_inducing(c);
        (sq(c, &z), sq(&z, q), sq(q, q))
    }
}

/// The supplied blocks of a scalar slot over every dimension of `case`.
fn summed(case: &Case) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let parts: Vec<_> = (0..case.cols.len()).map(|k| case.blocks(k)).collect();
    (
        sum(&parts.iter().map(|p| p.0.clone()).collect::<Vec<_>>()),
        sum(&parts.iter().map(|p| p.1.clone()).collect::<Vec<_>>()),
        sum(&parts.iter().map(|p| p.2.clone()).collect::<Vec<_>>()),
    )
}

/// Compares an SGPR on supplied distances with the coordinate SGPR: the
/// bound, its gradient and Hessian, LOO, and every prediction.
macro_rules! sgpr_matches {
    ($dist:expr, $coords:expr, $cross:expr, $square:expr, ($($pts:expr),*), [$($tail:expr),*], $tol:expr) => {{
        let (mut dist, mut coords) = ($dist, $coords);
        let tol: f64 = $tol;
        assert_close(
            dist.neg_log_marginal_likelihood().expect("nlml"),
            coords.neg_log_marginal_likelihood().expect("nlml"),
            tol,
        );
        let p = coords.num_params();
        assert_eq!(dist.num_params(), p);
        let mut theta = vec![0.0; p];
        coords.get_params(&mut theta).expect("theta");
        for shift in [0.0, 0.2] {
            let at: Vec<f64> = theta.iter().map(|t| t + shift).collect();
            let (mut gc, mut gd) = (vec![0.0; p], vec![0.0; p]);
            let vc = coords.value_and_gradient_into(&at, &mut gc).expect("grad");
            let vd = dist.value_and_gradient_into(&at, &mut gd).expect("grad");
            assert_close(vd, vc, tol);
            assert_slice_close(&gd, &gc, 10.0 * tol);
            let (mut hc, mut hd) = (vec![0.0; p * p], vec![0.0; p * p]);
            coords.hessian_into(&at, &mut hc).expect("hess");
            dist.hessian_into(&at, &mut hd).expect("hess");
            assert_slice_close(&hd, &hc, 100.0 * tol);
        }
        coords.set_params(&theta).expect("theta");
        dist.set_params(&theta).expect("theta");
        assert_pred(
            &dist.loo_predict().expect("loo"),
            &coords.loo_predict().expect("loo"),
            tol,
        );
        sparse_predictions_match!(dist, coords, $cross, $square, ($($pts),*), [$($tail),*], tol);
    }};
}

/// Compares every prediction of a sparse model on supplied distances
/// with the coordinate model's.
macro_rules! sparse_predictions_match {
    ($dist:ident, $coords:ident, $cross:expr, $square:expr, ($($pts:expr),*), [$($tail:expr),*], $tol:expr) => {{
        let d = $coords.d();
        let qx = queries(d);
        let latent = PredictOptions {
            variance_kind: VarianceKind::Latent,
        };
        let expect = $coords.predict(&qx, Q, d).expect("predict");
        let got = $dist.predict($cross(), $($pts,)* Q $(, $tail)*).expect("predict");
        assert_pred(&got, &expect, $tol);
        let expect = $coords.predict_with(&qx, Q, d, latent).expect("predict");
        let got = $dist.predict_with($cross(), $($pts,)* Q, $($tail,)* latent).expect("predict");
        assert_pred(&got, &expect, $tol);
        let mut out = Prediction::default();
        $dist.predict_into($cross(), $($pts,)* Q, $($tail,)* &mut out).expect("into");
        let mut expect_into = Prediction::default();
        $coords.predict_into(&qx, Q, d, &mut expect_into).expect("into");
        assert_pred(&out, &expect_into, $tol);
        let cov_c = $coords.predict_covariance(&qx, Q, d).expect("cov");
        let cov_d = $dist
            .predict_covariance($cross(), $square(), $($pts,)* Q $(, $tail)*)
            .expect("cov");
        assert_slice_close(&to64(&cov_d.covariance), &to64(&cov_c.covariance), $tol);
        assert_slice_close(&to64(&cov_d.mean), &to64(&cov_c.mean), $tol);
        let draws_c = $coords.sample(&qx, Q, d, 3, 11).expect("sample");
        let draws_d = $dist
            .sample($cross(), $square(), $($pts,)* Q, $($tail,)* 3, 11)
            .expect("sample");
        assert_slice_close(&to64(&draws_d), &to64(&draws_c), 1e3 * $tol);
    }};
}

/// The query coordinates of the coordinate model: the columns of
/// [`Case::new`]`(d)`.
fn queries(d: usize) -> Vec<f64> {
    Case::new(d).xs()
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn sgpr_scalar<P: GpScalar + std::fmt::Debug>(tol: f64) {
    let case = Case::new(2);
    let (train, cross, square) = summed(&case);
    let y = targets();
    let rbf = RbfKernel::new(0.9).expect("ell");
    let coords = Sgpr::new(KernelSpec::from(rbf), lik())
        .with_precision::<P>()
        .with_optimizer(Fixed)
        .factor(&case.x(), N, 2, &y, &case.z(), INDUCING.len())
        .expect("coords");
    let image = ScalarDistance::new();
    let dist = Sgpr::new(image.kernel(rbf), lik())
        .with_precision::<P>()
        .with_optimizer(Fixed)
        .factor([image.from_vec(train)], N, &y, &INDUCING)
        .expect("distances");
    assert_eq!(dist.inducing(), &INDUCING);
    assert_eq!(dist.slots().len(), 1);
    sgpr_matches!(
        dist,
        coords,
        || [image.borrow(&cross)],
        || [image.borrow(&square)],
        (),
        [],
        tol
    );
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn sgpr_ard<P: GpScalar + std::fmt::Debug>(tol: f64) {
    let case = Case::new(3);
    let y = targets();
    let ard = RbfArdKernel::new(&[0.8, 1.4, 2.0]).expect("ell");
    let coords = Sgpr::new(KernelSpec::from(ard.clone()), lik())
        .with_precision::<P>()
        .with_optimizer(Fixed)
        .factor(&case.x(), N, 3, &y, &case.z(), INDUCING.len())
        .expect("coords");
    let (bands, ard) = ArdDistance::from_leaf(ard);
    let parts: Vec<_> = (0..3).map(|k| case.blocks(k)).collect();
    let train: Vec<Vec<f64>> = parts.iter().map(|p| p.0.clone()).collect();
    let cross: Vec<&[f64]> = parts.iter().map(|p| p.1.as_slice()).collect();
    let square: Vec<&[f64]> = parts.iter().map(|p| p.2.as_slice()).collect();
    let dist = Sgpr::new(ard, lik())
        .with_precision::<P>()
        .with_optimizer(Fixed)
        .factor([bands.from_vecs(train)], N, &y, &INDUCING)
        .expect("distances");
    sgpr_matches!(
        dist,
        coords,
        || [bands.borrow(&cross)],
        || [bands.borrow(&square)],
        (),
        [],
        tol
    );
}

/// `exp(-Δ0²/2ℓ0²) · exp(-Δ1²/2ℓ1²)`, the first factor on supplied
/// distances and the second on coordinates, is the ARD RBF of `(ℓ0, ℓ1)`.
#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn sgpr_with_points<P: GpScalar + std::fmt::Debug>(tol: f64) {
    let case = Case::new(2);
    let y = targets();
    let (ell0, ell1) = (0.7, 1.3);
    let coords = Sgpr::new(
        KernelSpec::from(RbfArdKernel::new(&[ell0, ell1]).expect("ell")),
        lik(),
    )
    .with_precision::<P>()
    .with_optimizer(Fixed)
    .factor(&case.x(), N, 2, &y, &case.z(), INDUCING.len())
    .expect("coords");
    let image = ScalarDistance::new();
    let kernel = image.kernel(RbfKernel::new(ell0).expect("ell"))
        * KernelSpec::from(RbfKernel::new(ell1).expect("ell"));
    let (train, cross, square) = case.blocks(0);
    let dist = Sgpr::new(kernel, lik())
        .with_precision::<P>()
        .with_optimizer(Fixed)
        .factor([image.from_vec(train)], N, &case.cols[1], 1, &y, &INDUCING)
        .expect("distances");
    assert_eq!(dist.z(), at_inducing(&case.cols[1]).as_slice());
    let q1 = case.qcols[1].clone();
    sgpr_matches!(
        dist,
        coords,
        || [image.borrow(&cross)],
        || [image.borrow(&square)],
        (&q1),
        [1],
        tol
    );
}

#[test]
fn sgpr_on_supplied_distances_matches_coordinates() {
    sgpr_scalar::<DoublePrecision>(1e-9);
    sgpr_ard::<DoublePrecision>(1e-9);
    sgpr_with_points::<DoublePrecision>(1e-9);
}

#[test]
fn sgpr_on_supplied_distances_matches_coordinates_in_single_and_mixed() {
    sgpr_scalar::<SinglePrecision>(2e-3);
    sgpr_ard::<SinglePrecision>(2e-3);
    sgpr_with_points::<SinglePrecision>(2e-3);
    sgpr_scalar::<MixedPrecision<PromoteStorage>>(2e-3);
    sgpr_ard::<MixedPrecision<PromoteStorage>>(2e-3);
    sgpr_with_points::<MixedPrecision<PromoteStorage>>(2e-3);
}

/// Compares an SVGP on supplied distances with the coordinate SVGP: the
/// ELBO, its gradient, and every prediction.
macro_rules! svgp_matches {
    ($dist:expr, $coords:expr, $cross:expr, $square:expr, ($($pts:expr),*), [$($tail:expr),*], $tol:expr) => {{
        let (mut dist, mut coords) = ($dist, $coords);
        let tol: f64 = $tol;
        assert_close(
            dist.neg_elbo().expect("elbo"),
            coords.neg_elbo().expect("elbo"),
            tol,
        );
        let p = coords.num_params();
        assert_eq!(dist.num_params(), p);
        let mut theta = vec![0.0; p];
        coords.get_params(&mut theta).expect("theta");
        let (mut gc, mut gd) = (vec![0.0; p], vec![0.0; p]);
        let vc = coords.value_and_gradient_into(&theta, &mut gc).expect("grad");
        let vd = dist.value_and_gradient_into(&theta, &mut gd).expect("grad");
        assert_close(vd, vc, tol);
        assert_slice_close(&gd, &gc, 10.0 * tol);
        sparse_predictions_match!(dist, coords, $cross, $square, ($($pts),*), [$($tail),*], tol);
    }};
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn adam() -> Adam {
    Adam::new()
        .with_batch_size(NonZeroUsize::new(3).expect("batch"))
        .with_epochs(NonZeroU64::new(4).expect("epochs"))
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn svgp_all<P: GpScalar + std::fmt::Debug>(tol: f64) {
    let y = targets();
    // Scalar slot.
    let case = Case::new(2);
    let (train, cross, square) = summed(&case);
    let rbf = RbfKernel::new(0.9).expect("ell");
    let image = ScalarDistance::new();
    for fit in [false, true] {
        let (dist, coords) = if fit {
            (
                Svgp::new(image.kernel(rbf), lik())
                    .with_precision::<P>()
                    .with_optimizer(adam())
                    .fit([image.from_slice(&train)], N, &y, &INDUCING)
                    .expect("distances"),
                Svgp::new(KernelSpec::from(rbf), lik())
                    .with_precision::<P>()
                    .with_optimizer(adam())
                    .fit(&case.x(), N, 2, &y, &case.z(), INDUCING.len())
                    .expect("coords"),
            )
        } else {
            (
                Svgp::new(image.kernel(rbf), lik())
                    .with_precision::<P>()
                    .factor([image.from_slice(&train)], N, &y, &INDUCING)
                    .expect("distances"),
                Svgp::new(KernelSpec::from(rbf), lik())
                    .with_precision::<P>()
                    .factor(&case.x(), N, 2, &y, &case.z(), INDUCING.len())
                    .expect("coords"),
            )
        };
        assert_eq!(dist.inducing(), &INDUCING);
        let (mut a, mut b) = (vec![0.0; dist.num_params()], vec![0.0; coords.num_params()]);
        dist.get_params(&mut a).expect("theta");
        coords.get_params(&mut b).expect("theta");
        assert_slice_close(&a, &b, 100.0 * tol);
        svgp_matches!(
            dist,
            coords,
            || [image.borrow(&cross)],
            || [image.borrow(&square)],
            (),
            [],
            100.0 * tol
        );
    }
    // ARD slot.
    let case = Case::new(3);
    let ard = RbfArdKernel::new(&[0.8, 1.4, 2.0]).expect("ell");
    let coords = Svgp::new(KernelSpec::from(ard.clone()), lik())
        .with_precision::<P>()
        .with_optimizer(adam())
        .fit(&case.x(), N, 3, &y, &case.z(), INDUCING.len())
        .expect("coords");
    let (bands, ard) = ArdDistance::from_leaf(ard);
    let parts: Vec<_> = (0..3).map(|k| case.blocks(k)).collect();
    let train: Vec<Vec<f64>> = parts.iter().map(|p| p.0.clone()).collect();
    let cross: Vec<&[f64]> = parts.iter().map(|p| p.1.as_slice()).collect();
    let square: Vec<&[f64]> = parts.iter().map(|p| p.2.as_slice()).collect();
    let dist = Svgp::new(ard, lik())
        .with_precision::<P>()
        .with_optimizer(adam())
        .fit([bands.from_vecs(train)], N, &y, &INDUCING)
        .expect("distances");
    svgp_matches!(
        dist,
        coords,
        || [bands.borrow(&cross)],
        || [bands.borrow(&square)],
        (),
        [],
        100.0 * tol
    );
    // A slot next to a coordinate leaf.
    let case = Case::new(2);
    let (ell0, ell1) = (0.7, 1.3);
    let coords = Svgp::new(
        KernelSpec::from(RbfArdKernel::new(&[ell0, ell1]).expect("ell")),
        lik(),
    )
    .with_precision::<P>()
    .with_optimizer(adam())
    .fit(&case.x(), N, 2, &y, &case.z(), INDUCING.len())
    .expect("coords");
    let kernel = image.kernel(RbfKernel::new(ell0).expect("ell"))
        * KernelSpec::from(RbfKernel::new(ell1).expect("ell"));
    let (train, cross, square) = case.blocks(0);
    let dist = Svgp::new(kernel, lik())
        .with_precision::<P>()
        .with_optimizer(adam())
        .fit([image.from_vec(train)], N, &case.cols[1], 1, &y, &INDUCING)
        .expect("distances");
    let q1 = case.qcols[1].clone();
    svgp_matches!(
        dist,
        coords,
        || [image.borrow(&cross)],
        || [image.borrow(&square)],
        (&q1),
        [1],
        100.0 * tol
    );
}

#[test]
fn svgp_on_supplied_distances_matches_coordinates() {
    svgp_all::<DoublePrecision>(1e-10);
}

#[test]
fn svgp_on_supplied_distances_matches_coordinates_in_single_and_mixed() {
    svgp_all::<SinglePrecision>(2e-5);
    svgp_all::<MixedPrecision<PromoteStorage>>(2e-5);
}

#[test]
fn a_hyperparameter_search_matches_the_coordinate_search() {
    let case = Case::new(1);
    let y = targets();
    let image = ScalarDistance::new();
    let (train, cross, _) = case.blocks(0);
    let coords = Sgpr::new(KernelSpec::from(RbfKernel::new(1.0).expect("ell")), lik())
        .fit(&case.x(), N, 1, &y, &case.z(), INDUCING.len())
        .expect("coords");
    let dist = Sgpr::new(image.kernel(RbfKernel::new(1.0).expect("ell")), lik())
        .fit([image.from_slice(&train)], N, &y, &INDUCING)
        .expect("dist");
    let (mut a, mut b) = (vec![0.0; 2], vec![0.0; 2]);
    coords.get_params(&mut a).expect("theta");
    dist.get_params(&mut b).expect("theta");
    assert_slice_close(&b, &a, 1e-6);
    let got = dist.predict([image.borrow(&cross)], Q).expect("predict");
    let expect = coords.predict(&case.xs(), Q, 1).expect("predict");
    assert_pred(&got, &expect, 1e-5);
    assert_eq!(dist.to_kernel().num_params(), 1);
}

#[test]
fn inducing_indices_are_checked() {
    let case = Case::new(1);
    let y = targets();
    let image = ScalarDistance::new();
    let trainer =
        || Sgpr::new(image.kernel(RbfKernel::new(1.0).expect("ell")), lik()).with_optimizer(Fixed);
    let block = |inducing: &[usize]| {
        let z: Vec<f64> = inducing.iter().map(|&i| case.cols[0][i]).collect();
        sq(&case.cols[0], &z)
    };
    let err = |inducing: &[usize], table: Vec<f64>| -> GprError {
        trainer()
            .factor([image.from_vec(table)], N, &y, inducing)
            .map(|_| ())
            .map_err(|(_, e)| e)
            .expect_err("rejected")
    };
    assert!(matches!(err(&[], Vec::new()), GprError::EmptyInput));
    assert!(matches!(
        err(&[0, N], block(&[0, 1])),
        GprError::IndexOutOfRange { .. }
    ));
    assert!(matches!(
        err(&[3, 1, 3], block(&[3, 1, 2])),
        GprError::InvalidConfig { .. }
    ));
    // A block whose inducing rows are not a square: not the distances of
    // the samples it names.
    assert!(matches!(
        err(&[0, 1], block(&[0, 2])),
        GprError::InvalidDistance { .. }
    ));
    assert!(matches!(
        err(&[0, 1], vec![0.0; N]),
        GprError::LengthMismatch { .. }
    ));
    let svgp = Svgp::new(image.kernel(RbfKernel::new(1.0).expect("ell")), lik())
        .factor([image.from_vec(block(&[0, 1]))], N, &y, &[1, 0])
        .map(|_| ())
        .map_err(|(_, e)| e);
    assert!(matches!(svgp, Err(GprError::InvalidDistance { .. })));
    // A query block is `m × q`, not `n × q`.
    let model = trainer()
        .factor([image.from_vec(block(&[0, 5]))], N, &y, &[0, 5])
        .expect("model");
    let wrong: Vec<f64> = vec![1.0; N * Q];
    assert!(matches!(
        model.predict([image.borrow(&wrong)], Q),
        Err(GprError::LengthMismatch { .. })
    ));
}

/// Fills the `n × m` blocks from the training points to the inducing ones
/// (column `a` is `INDUCING[a]`): one per coordinate, or their sum.
struct Block<'a> {
    cols: &'a [Vec<f64>],
    summed: bool,
}

impl gprx::kernel::DistanceFill for Block<'_> {
    fn fill_column(&self, col: usize, rows: std::ops::Range<usize>, out: &mut [f64]) {
        // An ARD fill writes its runs one after another, dimension by dimension.
        let len = rows.len();
        out.fill(0.0);
        for (k, c) in self.cols.iter().enumerate() {
            let z = c[INDUCING[col]];
            let at = if self.summed { 0 } else { k * len };
            for (slot, i) in out[at..at + len].iter_mut().zip(rows.clone()) {
                *slot += (c[i] - z) * (c[i] - z);
            }
        }
    }
}

/// Every way to hand over the training blocks (borrowed, copied, moved,
/// filled; repaired by `tidy` when rounded) gives the same model.
#[test]
fn every_source_kind_fits_the_same_sparse_model() {
    let case = Case::new(2);
    let y = targets();
    let (train, cross, _) = summed(&case);
    let image = ScalarDistance::new();
    let rbf = RbfKernel::new(0.9).expect("ell");
    let fit = |source: gprx::kernel::DistanceSource<'_>| {
        Sgpr::new(image.kernel(rbf), lik())
            .with_optimizer(Fixed)
            .factor([source], N, &y, &INDUCING)
            .map_err(|(_, e)| e)
            .expect("fit")
    };
    let expect = fit(image.borrow(&train))
        .predict([image.borrow(&cross)], Q)
        .expect("predict");
    // A rounded table: one value a hair below zero, repaired by `tidy`.
    let mut rounded = train.clone();
    rounded[INDUCING[0]] = -1e-15;
    for source in [
        image.from_slice(&train),
        image.from_vec(train.clone()),
        image.fill(&Block {
            cols: &case.cols,
            summed: true,
        }),
        image.from_vec(rounded).tidy(1e-9).expect("tidy"),
    ] {
        let got = fit(source)
            .predict([image.borrow(&cross)], Q)
            .expect("predict");
        assert_pred(&got, &expect, 1e-12);
    }
    // The same for an ARD slot.
    let ard = RbfArdKernel::new(&[0.8, 1.4]).expect("ell");
    let (bands, ard) = ArdDistance::from_leaf(ard);
    let parts: Vec<_> = (0..2).map(|k| case.blocks(k)).collect();
    let blocks: Vec<Vec<f64>> = parts.iter().map(|p| p.0.clone()).collect();
    let refs: Vec<&[f64]> = blocks.iter().map(Vec::as_slice).collect();
    let cross: Vec<&[f64]> = parts.iter().map(|p| p.1.as_slice()).collect();
    let fit = |source: gprx::kernel::DistanceSource<'_>| {
        Svgp::new(ard.clone(), lik())
            .factor([source], N, &y, &INDUCING)
            .map_err(|(_, e)| e)
            .expect("fit")
    };
    let expect = fit(bands.borrow(&refs))
        .predict([bands.borrow(&cross)], Q)
        .expect("predict");
    for source in [
        bands.from_slices(&refs),
        bands.from_vecs(blocks.clone()),
        bands.fill(&Block {
            cols: &case.cols,
            summed: false,
        }),
        bands.from_vecs(blocks.clone()).tidy(1e-9).expect("tidy"),
    ] {
        let got = fit(source)
            .predict([bands.borrow(&cross)], Q)
            .expect("predict");
        assert_pred(&got, &expect, 1e-12);
    }
}
