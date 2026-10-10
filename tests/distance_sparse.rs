//! SGPR and SVGP on supplied squared distances against the same models on
//! coordinates, with the inducing points at the coordinates of the
//! training samples they name. Public API only.

mod common;

use common::distance::{assert_pred, coord, lik, sq, sum, to64};

use common::{assert_close, assert_slice_close};
use gprx::kernel::{
    ArdDistance, DistanceFill, DistanceSource, KernelSpec, RbfArdKernel, RbfKernel, ScalarDistance,
};
use gprx::transform::{MinMaxInput, Transform};
use gprx::{
    Adam, DoublePrecision, Fixed, GpScalar, GprError, MixedPrecision, PredictOptions, Prediction,
    PromoteStorage, Sgpr, SinglePrecision, SlotErrorKind, Svgp, VarianceKind,
};
use std::num::{NonZeroU64, NonZeroUsize};

/// One problem: `n` training samples of `d` coordinates, the inducing
/// points (training indices), and `q` queries.
struct Case {
    n: usize,
    q: usize,
    inducing: Vec<usize>,
    cols: Vec<Vec<f64>>,
    qcols: Vec<Vec<f64>>,
}

impl Case {
    fn new(d: usize, n: usize, inducing: &[usize], q: usize) -> Self {
        Self {
            n,
            q,
            inducing: inducing.to_vec(),
            cols: (0..d).map(|k| coord(k, n, 0.0)).collect(),
            qcols: (0..d).map(|k| coord(k, q, 0.5)).collect(),
        }
    }

    /// Ten samples, four inducing points in no order, four queries.
    fn standard(d: usize) -> Self {
        Self::new(d, 10, &[7, 0, 4, 9], 4)
    }

    fn m(&self) -> usize {
        self.inducing.len()
    }

    fn at_inducing(&self, col: &[f64]) -> Vec<f64> {
        self.inducing.iter().map(|&i| col[i]).collect()
    }

    fn y(&self) -> Vec<f64> {
        (0..self.n).map(|i| (i as f64 * 0.7).cos()).collect()
    }

    fn x(&self) -> Vec<f64> {
        self.cols.concat()
    }

    fn z(&self) -> Vec<f64> {
        self.cols
            .iter()
            .map(|c| self.at_inducing(c))
            .collect::<Vec<_>>()
            .concat()
    }

    fn xs(&self) -> Vec<f64> {
        self.qcols.concat()
    }

    /// Per dimension `k`: the `n × m` training block, the `m × q` cross
    /// block, and the `q × q` query square.
    fn blocks(&self, k: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let (c, q) = (&self.cols[k], &self.qcols[k]);
        let z = self.at_inducing(c);
        (sq(c, &z), sq(&z, q), sq(q, q))
    }

    /// The blocks of a scalar slot over every dimension.
    fn summed(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let parts: Vec<_> = (0..self.cols.len()).map(|k| self.blocks(k)).collect();
        (
            sum(&parts.iter().map(|p| p.0.clone()).collect::<Vec<_>>()),
            sum(&parts.iter().map(|p| p.1.clone()).collect::<Vec<_>>()),
            sum(&parts.iter().map(|p| p.2.clone()).collect::<Vec<_>>()),
        )
    }
}

/// Compares an SGPR on supplied distances with the coordinate SGPR: the
/// bound, its gradient and Hessian at two `θ`, LOO, and every prediction.
macro_rules! sgpr_matches {
    ($case:expr, $dist:expr, $coords:expr, $cross:expr, $square:expr, ($($pts:expr),*), [$($tail:expr),*], $tol:expr) => {{
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
            assert_slice_close(&gd, &gc, tol);
            let (mut hc, mut hd) = (vec![0.0; p * p], vec![0.0; p * p]);
            coords.hessian_into(&at, &mut hc).expect("hess");
            dist.hessian_into(&at, &mut hd).expect("hess");
            assert_slice_close(&hd, &hc, tol);
        }
        coords.set_params(&theta).expect("theta");
        dist.set_params(&theta).expect("theta");
        assert_pred(
            &dist.loo_predict().expect("loo"),
            &coords.loo_predict().expect("loo"),
            tol,
        );
        sparse_predictions_match!($case, dist, coords, $cross, $square, ($($pts),*), [$($tail),*], tol);
    }};
}

/// Compares every prediction of a sparse model on supplied distances
/// with the coordinate model's.
macro_rules! sparse_predictions_match {
    ($case:expr, $dist:ident, $coords:ident, $cross:expr, $square:expr, ($($pts:expr),*), [$($tail:expr),*], $tol:expr) => {{
        let (case, tol): (&Case, f64) = (&$case, $tol);
        let (d, q) = ($coords.d(), case.q);
        let qx = case.xs();
        let latent = PredictOptions {
            variance_kind: VarianceKind::Latent,
        };
        let expect = $coords.predict(&qx, q, d).expect("predict");
        let got = $dist.predict($cross(), $($pts,)* q $(, $tail)*).expect("predict");
        assert_pred(&got, &expect, tol);
        let expect = $coords.predict_with(&qx, q, d, latent).expect("predict");
        let got = $dist.predict_with($cross(), $($pts,)* q, $($tail,)* latent).expect("predict");
        assert_pred(&got, &expect, tol);
        let mut out = Prediction::default();
        let mut expect_into = Prediction::default();
        $dist.predict_into($cross(), $($pts,)* q, $($tail,)* &mut out).expect("into");
        $coords.predict_into(&qx, q, d, &mut expect_into).expect("into");
        assert_pred(&out, &expect_into, tol);
        $dist.predict_with_into($cross(), $($pts,)* q, $($tail,)* latent, &mut out).expect("into");
        $coords.predict_with_into(&qx, q, d, latent, &mut expect_into).expect("into");
        assert_pred(&out, &expect_into, tol);
        let cov_c = $coords.predict_covariance(&qx, q, d).expect("cov");
        let cov_d = $dist
            .predict_covariance($cross(), $square(), $($pts,)* q $(, $tail)*)
            .expect("cov");
        assert_slice_close(&to64(&cov_d.covariance), &to64(&cov_c.covariance), tol);
        assert_slice_close(&to64(&cov_d.mean), &to64(&cov_c.mean), tol);
        let cov_c = $coords.predict_covariance_with(&qx, q, d, latent).expect("cov");
        let cov_d = $dist
            .predict_covariance_with($cross(), $square(), $($pts,)* q, $($tail,)* latent)
            .expect("cov");
        assert_slice_close(&to64(&cov_d.covariance), &to64(&cov_c.covariance), tol);
        let draws_c = $coords.sample(&qx, q, d, 3, 11).expect("sample");
        let draws_d = $dist
            .sample($cross(), $square(), $($pts,)* q, $($tail,)* 3, 11)
            .expect("sample");
        assert_slice_close(&to64(&draws_d), &to64(&draws_c), 10.0 * tol);
        let draws_c = $coords.sample_with(&qx, q, d, latent, 2, 5).expect("sample");
        let draws_d = $dist
            .sample_with($cross(), $square(), $($pts,)* q, $($tail,)* latent, 2, 5)
            .expect("sample");
        assert_slice_close(&to64(&draws_d), &to64(&draws_c), 10.0 * tol);
    }};
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn sgpr_scalar<P: GpScalar + std::fmt::Debug>(case: &Case, tol: f64) {
    let (train, cross, square) = case.summed();
    let (n, d, y) = (case.n, case.cols.len(), case.y());
    let rbf = RbfKernel::new(0.9).expect("ell");
    let coords = Sgpr::new(KernelSpec::from(rbf), lik())
        .with_precision::<P>()
        .with_optimizer(Fixed)
        .factor(&case.x(), n, d, &y, &case.z(), case.m())
        .expect("coords");
    let image = ScalarDistance::new();
    let dist = Sgpr::new(image.kernel(rbf), lik())
        .with_precision::<P>()
        .with_optimizer(Fixed)
        .factor([image.from_vec(train)], n, &y, &case.inducing)
        .expect("distances");
    assert_eq!(dist.inducing(), case.inducing.as_slice());
    assert_eq!(dist.slots().len(), 1);
    sgpr_matches!(
        case,
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
    let case = Case::standard(3);
    let (n, y) = (case.n, case.y());
    let ard = RbfArdKernel::new(&[0.8, 1.4, 2.0]).expect("ell");
    let coords = Sgpr::new(KernelSpec::from(ard.clone()), lik())
        .with_precision::<P>()
        .with_optimizer(Fixed)
        .factor(&case.x(), n, 3, &y, &case.z(), case.m())
        .expect("coords");
    let (bands, ard) = ArdDistance::from_leaf(ard);
    let parts: Vec<_> = (0..3).map(|k| case.blocks(k)).collect();
    let train: Vec<Vec<f64>> = parts.iter().map(|p| p.0.clone()).collect();
    let cross: Vec<&[f64]> = parts.iter().map(|p| p.1.as_slice()).collect();
    let square: Vec<&[f64]> = parts.iter().map(|p| p.2.as_slice()).collect();
    let dist = Sgpr::new(ard, lik())
        .with_precision::<P>()
        .with_optimizer(Fixed)
        .factor([bands.from_vecs(train)], n, &y, &case.inducing)
        .expect("distances");
    sgpr_matches!(
        case,
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
    let case = Case::standard(2);
    let (n, y) = (case.n, case.y());
    let (ell0, ell1) = (0.7, 1.3);
    let coords = Sgpr::new(
        KernelSpec::from(RbfArdKernel::new(&[ell0, ell1]).expect("ell")),
        lik(),
    )
    .with_precision::<P>()
    .with_optimizer(Fixed)
    .factor(&case.x(), n, 2, &y, &case.z(), case.m())
    .expect("coords");
    let image = ScalarDistance::new();
    let kernel = image.kernel(RbfKernel::new(ell0).expect("ell"))
        * KernelSpec::from(RbfKernel::new(ell1).expect("ell"));
    let (train, cross, square) = case.blocks(0);
    let dist = Sgpr::new(kernel, lik())
        .with_precision::<P>()
        .with_optimizer(Fixed)
        .factor(
            [image.from_vec(train)],
            n,
            &case.cols[1],
            1,
            &y,
            &case.inducing,
        )
        .expect("distances");
    assert_eq!(dist.z(), case.at_inducing(&case.cols[1]).as_slice());
    let q1 = case.qcols[1].clone();
    sgpr_matches!(
        case,
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
    sgpr_scalar::<DoublePrecision>(&Case::standard(2), 1e-9);
    sgpr_ard::<DoublePrecision>(1e-9);
    sgpr_with_points::<DoublePrecision>(1e-9);
}

/// Every training point inducing (`m = n`, in another order), one
/// inducing point, one query, and more queries than inducing points.
#[test]
fn sgpr_on_supplied_distances_matches_coordinates_at_every_shape() {
    sgpr_scalar::<DoublePrecision>(&Case::new(2, 6, &[5, 3, 1, 0, 2, 4], 3), 1e-8);
    sgpr_scalar::<DoublePrecision>(&Case::new(2, 8, &[3], 1), 1e-9);
    sgpr_scalar::<DoublePrecision>(&Case::new(2, 9, &[8, 2, 5], 5), 1e-9);
}

#[test]
fn sgpr_on_supplied_distances_matches_coordinates_in_single_and_mixed() {
    sgpr_scalar::<SinglePrecision>(&Case::standard(2), 2e-3);
    sgpr_ard::<SinglePrecision>(2e-3);
    sgpr_with_points::<SinglePrecision>(2e-3);
    sgpr_scalar::<MixedPrecision<PromoteStorage>>(&Case::standard(2), 2e-3);
    sgpr_ard::<MixedPrecision<PromoteStorage>>(2e-3);
    sgpr_with_points::<MixedPrecision<PromoteStorage>>(2e-3);
}

/// Compares an SVGP on supplied distances with the coordinate SVGP: the
/// ELBO and its gradient at two `θ`, and every prediction.
macro_rules! svgp_matches {
    ($case:expr, $dist:expr, $coords:expr, $cross:expr, $square:expr, ($($pts:expr),*), [$($tail:expr),*], $tol:expr) => {{
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
        for shift in [0.0, 0.2] {
            // Every parameter moves up; the diagonal of `L` stays positive.
            let at: Vec<f64> = theta.iter().map(|t| t + shift).collect();
            let (mut gc, mut gd) = (vec![0.0; p], vec![0.0; p]);
            let vc = coords.value_and_gradient_into(&at, &mut gc).expect("grad");
            let vd = dist.value_and_gradient_into(&at, &mut gd).expect("grad");
            assert_close(vd, vc, tol);
            assert_slice_close(&gd, &gc, tol);
        }
        coords.set_params(&theta).expect("theta");
        dist.set_params(&theta).expect("theta");
        sparse_predictions_match!($case, dist, coords, $cross, $square, ($($pts),*), [$($tail),*], tol);
    }};
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn adam() -> Adam {
    Adam::new()
        .with_batch_size(NonZeroUsize::new(3).expect("batch"))
        .with_epochs(NonZeroU64::new(4).expect("epochs"))
}

/// The parameters of a fitted SVGP.
#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn svgp_params<P: GpScalar, K: gprx::kernel::ModelKernel>(
    model: &gprx::FittedSvgp<P, K>,
) -> Vec<f64> {
    let mut out = vec![0.0; model.num_params()];
    model.get_params(&mut out).expect("theta");
    out
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn svgp_all<P: GpScalar + std::fmt::Debug>(tol: f64) {
    // Scalar slot: factored, then fitted by mini-batch Adam (batches of 3
    // of 10, so every step gathers rows of the stored blocks).
    let case = Case::standard(2);
    let (n, y) = (case.n, case.y());
    let (train, cross, square) = case.summed();
    let rbf = RbfKernel::new(0.9).expect("ell");
    let image = ScalarDistance::new();
    let start = svgp_params(
        &Svgp::new(KernelSpec::from(rbf), lik())
            .with_precision::<P>()
            .factor(&case.x(), n, 2, &y, &case.z(), case.m())
            .expect("prior"),
    );
    for fit in [false, true] {
        let (dist, coords) = if fit {
            (
                Svgp::new(image.kernel(rbf), lik())
                    .with_precision::<P>()
                    .with_optimizer(adam())
                    .fit([image.from_slice(&train)], n, &y, &case.inducing)
                    .expect("distances"),
                Svgp::new(KernelSpec::from(rbf), lik())
                    .with_precision::<P>()
                    .with_optimizer(adam())
                    .fit(&case.x(), n, 2, &y, &case.z(), case.m())
                    .expect("coords"),
            )
        } else {
            (
                Svgp::new(image.kernel(rbf), lik())
                    .with_precision::<P>()
                    .factor([image.from_slice(&train)], n, &y, &case.inducing)
                    .expect("distances"),
                Svgp::new(KernelSpec::from(rbf), lik())
                    .with_precision::<P>()
                    .factor(&case.x(), n, 2, &y, &case.z(), case.m())
                    .expect("coords"),
            )
        };
        assert_eq!(dist.inducing(), case.inducing.as_slice());
        let (a, b) = (svgp_params(&dist), svgp_params(&coords));
        assert_slice_close(&a, &b, tol);
        if fit {
            // The steps moved the parameters well past the tolerance.
            let moved = a
                .iter()
                .zip(&start)
                .map(|(x, s)| (x - s).abs())
                .fold(0.0, f64::max);
            assert!(moved > 100.0 * tol, "Adam moved the parameters by {moved}");
        }
        svgp_matches!(
            case,
            dist,
            coords,
            || [image.borrow(&cross)],
            || [image.borrow(&square)],
            (),
            [],
            tol
        );
    }
    // ARD slot.
    let case = Case::standard(3);
    let ard = RbfArdKernel::new(&[0.8, 1.4, 2.0]).expect("ell");
    let coords = Svgp::new(KernelSpec::from(ard.clone()), lik())
        .with_precision::<P>()
        .with_optimizer(adam())
        .fit(&case.x(), n, 3, &y, &case.z(), case.m())
        .expect("coords");
    let (bands, ard) = ArdDistance::from_leaf(ard);
    let parts: Vec<_> = (0..3).map(|k| case.blocks(k)).collect();
    let train: Vec<Vec<f64>> = parts.iter().map(|p| p.0.clone()).collect();
    let cross: Vec<&[f64]> = parts.iter().map(|p| p.1.as_slice()).collect();
    let square: Vec<&[f64]> = parts.iter().map(|p| p.2.as_slice()).collect();
    let dist = Svgp::new(ard, lik())
        .with_precision::<P>()
        .with_optimizer(adam())
        .fit([bands.from_vecs(train)], n, &y, &case.inducing)
        .expect("distances");
    svgp_matches!(
        case,
        dist,
        coords,
        || [bands.borrow(&cross)],
        || [bands.borrow(&square)],
        (),
        [],
        tol
    );
    // A slot next to a coordinate leaf.
    let case = Case::standard(2);
    let (ell0, ell1) = (0.7, 1.3);
    let coords = Svgp::new(
        KernelSpec::from(RbfArdKernel::new(&[ell0, ell1]).expect("ell")),
        lik(),
    )
    .with_precision::<P>()
    .with_optimizer(adam())
    .fit(&case.x(), n, 2, &y, &case.z(), case.m())
    .expect("coords");
    let kernel = image.kernel(RbfKernel::new(ell0).expect("ell"))
        * KernelSpec::from(RbfKernel::new(ell1).expect("ell"));
    let (train, cross, square) = case.blocks(0);
    let dist = Svgp::new(kernel, lik())
        .with_precision::<P>()
        .with_optimizer(adam())
        .fit(
            [image.from_vec(train)],
            n,
            &case.cols[1],
            1,
            &y,
            &case.inducing,
        )
        .expect("distances");
    let q1 = case.qcols[1].clone();
    svgp_matches!(
        case,
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
fn svgp_on_supplied_distances_matches_coordinates() {
    svgp_all::<DoublePrecision>(1e-9);
}

#[test]
fn svgp_on_supplied_distances_matches_coordinates_in_single_and_mixed() {
    svgp_all::<SinglePrecision>(1e-4);
    svgp_all::<MixedPrecision<PromoteStorage>>(1e-4);
}

/// A search moves `θ` from where it started, and lands where the
/// coordinate search lands, on a scalar slot and on an ARD slot.
#[test]
fn a_hyperparameter_search_matches_the_coordinate_search() {
    let case = Case::standard(2);
    let (n, y) = (case.n, case.y());
    let image = ScalarDistance::new();
    let (train, cross, _) = case.blocks(0);
    let start = [0.0, 0.05f64.ln()];
    let coords = Sgpr::new(KernelSpec::from(RbfKernel::new(1.0).expect("ell")), lik())
        .fit(
            &case.cols[0],
            n,
            1,
            &y,
            &case.at_inducing(&case.cols[0]),
            case.m(),
        )
        .expect("coords");
    let dist = Sgpr::new(image.kernel(RbfKernel::new(1.0).expect("ell")), lik())
        .fit([image.from_slice(&train)], n, &y, &case.inducing)
        .expect("dist");
    let (mut a, mut b) = (vec![0.0; 2], vec![0.0; 2]);
    coords.get_params(&mut a).expect("theta");
    dist.get_params(&mut b).expect("theta");
    assert!((b[0] - start[0]).abs() > 1e-3 || (b[1] - start[1]).abs() > 1e-3);
    assert_slice_close(&b, &a, 1e-6);
    let got = dist
        .predict([image.borrow(&cross)], case.q)
        .expect("predict");
    let expect = coords.predict(&case.qcols[0], case.q, 1).expect("predict");
    assert_pred(&got, &expect, 1e-5);
    assert_eq!(dist.to_kernel().num_params(), 1);
    // An ARD slot.
    let ard = RbfArdKernel::new(&[1.0, 1.0]).expect("ell");
    let coords = Sgpr::new(KernelSpec::from(ard.clone()), lik())
        .fit(&case.x(), n, 2, &y, &case.z(), case.m())
        .expect("coords");
    let (bands, ard) = ArdDistance::from_leaf(ard);
    let blocks: Vec<Vec<f64>> = (0..2).map(|k| case.blocks(k).0).collect();
    let dist = Sgpr::new(ard, lik())
        .fit([bands.from_vecs(blocks)], n, &y, &case.inducing)
        .expect("dist");
    let (mut a, mut b) = (vec![0.0; 3], vec![0.0; 3]);
    coords.get_params(&mut a).expect("theta");
    dist.get_params(&mut b).expect("theta");
    assert!(b.iter().take(2).any(|t| t.abs() > 1e-3));
    assert_slice_close(&b, &a, 1e-6);
}

/// Two scalar slots, handed over in either order, are the ARD RBF of
/// their two lengthscales; a sum of a slot and a coordinate leaf is the
/// sum of two slots on the same distances.
#[test]
fn two_slots_and_a_sum_with_points_match() {
    let case = Case::standard(2);
    let (n, y, q) = (case.n, case.y(), case.q);
    let (ell0, ell1) = (0.7, 1.3);
    let (a, b) = (ScalarDistance::new(), ScalarDistance::new());
    let (train0, cross0, _) = case.blocks(0);
    let (train1, cross1, _) = case.blocks(1);
    let product =
        a.kernel(RbfKernel::new(ell0).expect("ell")) * b.kernel(RbfKernel::new(ell1).expect("ell"));
    let reference = Sgpr::new(
        KernelSpec::from(RbfArdKernel::new(&[ell0, ell1]).expect("ell")),
        lik(),
    )
    .with_optimizer(Fixed)
    .factor(&case.x(), n, 2, &y, &case.z(), case.m())
    .expect("coords")
    .predict(&case.xs(), q, 2)
    .expect("predict");
    for order in [false, true] {
        let sources = if order {
            [b.borrow(&train1), a.borrow(&train0)]
        } else {
            [a.borrow(&train0), b.borrow(&train1)]
        };
        let model = Sgpr::new(product.clone(), lik())
            .with_optimizer(Fixed)
            .factor(sources, n, &y, &case.inducing)
            .expect("two slots");
        let got = model
            .predict([b.borrow(&cross1), a.borrow(&cross0)], q)
            .expect("predict");
        assert_pred(&got, &reference, 1e-9);
    }
    // `k_a(Δ0) + k(x1)` with `x1` as coordinates, or as a second slot.
    let with_points = a.kernel(RbfKernel::new(ell0).expect("ell"))
        + KernelSpec::from(RbfKernel::new(ell1).expect("ell"));
    let two_slots =
        a.kernel(RbfKernel::new(ell0).expect("ell")) + b.kernel(RbfKernel::new(ell1).expect("ell"));
    let got = Svgp::new(with_points, lik())
        .factor([a.borrow(&train0)], n, &case.cols[1], 1, &y, &case.inducing)
        .expect("with points")
        .predict([a.borrow(&cross0)], &case.qcols[1], q, 1)
        .expect("predict");
    let expect = Svgp::new(two_slots, lik())
        .factor(
            [a.borrow(&train0), b.borrow(&train1)],
            n,
            &y,
            &case.inducing,
        )
        .expect("two slots")
        .predict([a.borrow(&cross0), b.borrow(&cross1)], q)
        .expect("predict");
    assert_pred(&got, &expect, 1e-10);
}

#[test]
fn bad_inputs_are_refused_and_the_trainer_comes_back_unchanged() {
    let case = Case::standard(1);
    let (n, y, q) = (case.n, case.y(), case.q);
    let image = ScalarDistance::new();
    let rbf = RbfKernel::new(1.3).expect("ell");
    let block = |inducing: &[usize]| {
        let z: Vec<f64> = inducing.iter().map(|&i| case.cols[0][i]).collect();
        sq(&case.cols[0], &z)
    };
    let good = block(&case.inducing);
    let err = |sources: Vec<DistanceSource<'_>>, y: &[f64], inducing: &[usize]| -> GprError {
        let trainer = Sgpr::new(image.kernel(rbf), lik()).with_optimizer(Fixed);
        let mut before = vec![0.0; trainer.num_params()];
        trainer.get_params(&mut before).expect("theta");
        let (trainer, err) = trainer
            .factor(sources, n, y, inducing)
            .map(|_| ())
            .expect_err("rejected");
        // The same trainer comes back, and it still factors good input.
        let mut after = vec![0.0; trainer.num_params()];
        trainer.get_params(&mut after).expect("theta");
        assert_eq!(
            after.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            before.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        trainer
            .factor([image.borrow(&good)], n, &y_of(n), &[7, 0, 4, 9])
            .expect("the trainer still fits");
        err
    };
    fn y_of(n: usize) -> Vec<f64> {
        (0..n).map(|i| (i as f64 * 0.7).cos()).collect()
    }
    let ind = case.inducing.clone();
    assert!(matches!(
        err(Vec::from([image.from_vec(Vec::new())]), &y, &[]),
        GprError::EmptyInput
    ));
    assert!(matches!(
        err(Vec::from([image.from_vec(block(&[0, 1]))]), &y, &[0, n]),
        GprError::IndexOutOfRange { .. }
    ));
    assert!(matches!(
        err(
            Vec::from([image.from_vec(block(&[3, 1, 2]))]),
            &y,
            &[3, 1, 3]
        ),
        GprError::InvalidConfig { .. }
    ));
    // Inducing rows that are not a square: not the samples they name.
    assert!(matches!(
        err(Vec::from([image.from_vec(block(&[0, 2]))]), &y, &[0, 1]),
        GprError::InvalidDistance { .. }
    ));
    // The `m × n` transpose of a block with `m < n` is the wrong length.
    let transposed = {
        let z = case.at_inducing(&case.cols[0]);
        sq(&z, &case.cols[0])
    };
    assert_eq!(transposed.len(), good.len());
    assert!(matches!(
        err(Vec::from([image.from_vec(transposed)]), &y, &ind),
        GprError::InvalidDistance { .. }
    ));
    assert!(matches!(
        err(Vec::from([image.from_vec(vec![0.0; n])]), &y, &ind),
        GprError::LengthMismatch { .. }
    ));
    // A value that is not finite, a negative one, and either in a row
    // that is not an inducing point.
    for bad in [f64::NAN, f64::INFINITY, -0.5] {
        let mut table = good.clone();
        table[1] = bad;
        assert!(matches!(
            err(Vec::from([image.from_vec(table)]), &y, &ind),
            GprError::InvalidDistance { .. }
        ));
    }
    // A mirror pair that differs, past no tolerance.
    let mut table = good.clone();
    table[ind[1] + n] += 1e-9;
    assert!(matches!(
        err(Vec::from([image.from_vec(table)]), &y, &ind),
        GprError::InvalidDistance { .. }
    ));
    // No source, two sources of one slot, a source of another slot.
    let other = ScalarDistance::new();
    let slot = |kind, slot| GprError::DistanceSlot { kind, slot };
    assert_eq!(
        err(Vec::new(), &y, &ind),
        slot(SlotErrorKind::Missing, Some(0))
    );
    assert_eq!(
        err(
            Vec::from([image.borrow(&good), image.borrow(&good)]),
            &y,
            &ind
        ),
        slot(SlotErrorKind::Duplicate, Some(0))
    );
    assert_eq!(
        err(
            Vec::from([image.borrow(&good), other.borrow(&good)]),
            &y,
            &ind
        ),
        slot(SlotErrorKind::NotRead, None)
    );
    // Targets of the wrong length or not finite.
    assert!(matches!(
        err(Vec::from([image.borrow(&good)]), &y[1..], &ind),
        GprError::LengthMismatch { .. }
    ));
    let mut nan_y = y.clone();
    nan_y[2] = f64::NAN;
    assert!(matches!(
        err(Vec::from([image.borrow(&good)]), &nan_y, &ind),
        GprError::NonFiniteInput
    ));
    // Coordinates of the wrong shape on a slot next to a coordinate leaf.
    let mixed = image.kernel(rbf) * KernelSpec::from(RbfKernel::new(2.0).expect("ell"));
    let x = case.cols[0].clone();
    for (x, n_cols) in [(&x[1..], 1), (&x[..], 2), (&x[..], 0), (&x[..0], 0)] {
        let refused = Svgp::new(mixed.clone(), lik())
            .factor([image.borrow(&good)], n, x, n_cols, &y, &ind)
            .map(|_| ())
            .map_err(|(_, e)| e);
        if x.is_empty() {
            assert!(matches!(refused, Err(GprError::EmptyInput)));
        } else {
            assert!(
                matches!(refused, Err(GprError::LengthMismatch { .. })),
                "x of {} values, n_cols {n_cols}",
                x.len()
            );
        }
    }
    // Queries: none, a block of `n` rows, a query square that is not one.
    let model = Sgpr::new(image.kernel(rbf), lik())
        .with_optimizer(Fixed)
        .factor([image.borrow(&good)], n, &y, &ind)
        .expect("model");
    let (_, cross, square) = case.blocks(0);
    assert!(matches!(
        model.predict([image.borrow(&[])], 0),
        Err(GprError::EmptyInput)
    ));
    assert!(matches!(
        model.predict([image.borrow(&vec![1.0; n * q])], q),
        Err(GprError::LengthMismatch { .. })
    ));
    let mut skew = square.clone();
    skew[1] += 1e-9;
    assert!(matches!(
        model.predict_covariance([image.borrow(&cross)], [image.borrow(&skew)], q),
        Err(GprError::InvalidDistance { .. })
    ));
    let mut diagonal = square;
    diagonal[0] = 1e-9;
    assert!(matches!(
        model.sample([image.borrow(&cross)], [image.borrow(&diagonal)], q, 1, 0),
        Err(GprError::InvalidDistance { .. })
    ));
}

/// Fills the `n × m` training blocks (column `a` is `inducing[a]`), or the
/// `m × q` blocks from the inducing points to the queries: one per
/// coordinate, or their sum.
struct Block<'a> {
    rows: &'a [Vec<f64>],
    cols: Vec<Vec<f64>>,
    summed: bool,
}

impl<'a> Block<'a> {
    fn train(case: &'a Case, summed: bool) -> Self {
        Self {
            rows: &case.cols,
            cols: case.cols.iter().map(|c| case.at_inducing(c)).collect(),
            summed,
        }
    }

    fn cross(case: &Case, rows: &'a [Vec<f64>], summed: bool) -> Self {
        let _ = case;
        Self {
            rows,
            cols: case.qcols.clone(),
            summed,
        }
    }
}

impl DistanceFill for Block<'_> {
    fn fill_column(&self, col: usize, rows: std::ops::Range<usize>, out: &mut [f64]) {
        // An ARD fill writes its runs one after another, dimension by dimension.
        let len = rows.len();
        out.fill(0.0);
        for (k, (r, c)) in self.rows.iter().zip(&self.cols).enumerate() {
            let z = c[col];
            let at = if self.summed { 0 } else { k * len };
            for (slot, i) in out[at..at + len].iter_mut().zip(rows.clone()) {
                *slot += (r[i] - z) * (r[i] - z);
            }
        }
    }
}

/// Every way to hand over the training blocks and the query blocks
/// (borrowed, copied, moved, filled) gives the same model and the same
/// predictions, on an SGPR factor and an SVGP fit.
#[test]
fn every_source_kind_fits_and_predicts_alike() {
    let case = Case::standard(2);
    let (n, y, q) = (case.n, case.y(), case.q);
    let (train, cross, square) = case.summed();
    let image = ScalarDistance::new();
    let rbf = RbfKernel::new(0.9).expect("ell");
    let z_rows: Vec<Vec<f64>> = case.cols.iter().map(|c| case.at_inducing(c)).collect();
    let fit = |source: DistanceSource<'_>| {
        Sgpr::new(image.kernel(rbf), lik())
            .with_optimizer(Fixed)
            .factor([source], n, &y, &case.inducing)
            .map_err(|(_, e)| e)
            .expect("fit")
    };
    let model = fit(image.borrow(&train));
    let expect = model.predict([image.borrow(&cross)], q).expect("predict");
    let filled = Block::train(&case, true);
    for source in [
        image.from_slice(&train),
        image.from_vec(train.clone()),
        image.fill(&filled),
    ] {
        let got = fit(source)
            .predict([image.borrow(&cross)], q)
            .expect("predict");
        assert_pred(&got, &expect, 1e-12);
    }
    let query_fill = Block::cross(&case, &z_rows, true);
    for source in [
        image.from_slice(&cross),
        image.from_vec(cross.clone()),
        image.fill(&query_fill),
    ] {
        let got = model.predict([source], q).expect("predict");
        assert_pred(&got, &expect, 1e-12);
    }
    let cov = model
        .predict_covariance([image.borrow(&cross)], [image.borrow(&square)], q)
        .expect("cov");
    let q_rows = case.qcols.clone();
    let square_fill = Block {
        rows: &q_rows,
        cols: case.qcols.clone(),
        summed: true,
    };
    for square in [image.from_vec(square.clone()), image.fill(&square_fill)] {
        let got = model
            .predict_covariance([image.from_slice(&cross)], [square], q)
            .expect("cov");
        assert_slice_close(&got.covariance, &cov.covariance, 1e-12);
    }
    // An SVGP fit by mini-batch Adam, and an ARD slot.
    let ard = RbfArdKernel::new(&[0.8, 1.4]).expect("ell");
    let (bands, ard) = ArdDistance::from_leaf(ard);
    let parts: Vec<_> = (0..2).map(|k| case.blocks(k)).collect();
    let blocks: Vec<Vec<f64>> = parts.iter().map(|p| p.0.clone()).collect();
    let refs: Vec<&[f64]> = blocks.iter().map(Vec::as_slice).collect();
    let cross: Vec<&[f64]> = parts.iter().map(|p| p.1.as_slice()).collect();
    let fit = |source: DistanceSource<'_>| {
        Svgp::new(ard.clone(), lik())
            .with_optimizer(adam())
            .fit([source], n, &y, &case.inducing)
            .map_err(|(_, e)| e)
            .expect("fit")
    };
    let model = fit(bands.borrow(&refs));
    let expect = model.predict([bands.borrow(&cross)], q).expect("predict");
    let filled = Block::train(&case, false);
    for source in [
        bands.from_slices(&refs),
        bands.from_vecs(blocks.clone()),
        bands.fill(&filled),
    ] {
        let got = fit(source)
            .predict([bands.borrow(&cross)], q)
            .expect("predict");
        assert_pred(&got, &expect, 1e-12);
    }
    let query_fill = Block::cross(&case, &z_rows, false);
    let owned: Vec<Vec<f64>> = cross.iter().map(|c| c.to_vec()).collect();
    for source in [
        bands.from_slices(&cross),
        bands.from_vecs(owned),
        bands.fill(&query_fill),
    ] {
        let got = model.predict([source], q).expect("predict");
        assert_pred(&got, &expect, 1e-12);
    }
}

/// A rounded table repaired by `tidy` (a value a hair below zero in a row
/// that is not an inducing point, a mirror pair of the inducing rows a
/// hair apart, a diagonal a hair above zero) fits the model of the table
/// repaired by hand: `K_mm` and `K(Z, X)` read the same repaired values.
#[test]
fn a_tidied_table_is_the_table_repaired_by_hand() {
    let case = Case::standard(2);
    let (n, y, q) = (case.n, case.y(), case.q);
    let ind = case.inducing.clone();
    let m = ind.len();
    let (train, cross, _) = case.summed();
    let mut rounded = train.clone();
    let outside = (0..n)
        .find(|i| !ind.contains(i))
        .expect("a row that is not inducing");
    rounded[outside + n] = -1e-13;
    rounded[ind[0] + n] += 3e-13; // pair (inducing 0, inducing 1)
    rounded[ind[2] + 2 * n] = 2e-13; // the diagonal of inducing 2
    // By hand: negatives to zero, the mirror pair to its mean, the diagonal
    // to zero.
    let mut fixed = rounded.clone();
    for v in &mut fixed {
        *v = v.max(0.0);
    }
    for a in 0..m {
        for b in 0..a {
            let (ab, ba) = (ind[a] + b * n, ind[b] + a * n);
            let mean = 0.5 * (fixed[ab] + fixed[ba]);
            fixed[ab] = mean;
            fixed[ba] = mean;
        }
        fixed[ind[a] + a * n] = 0.0;
    }
    let image = ScalarDistance::new();
    let fit = |source: DistanceSource<'_>| {
        Sgpr::new(image.kernel(RbfKernel::new(0.9).expect("ell")), lik())
            .with_optimizer(Fixed)
            .factor([source], n, &y, &ind)
            .map_err(|(_, e)| e)
    };
    assert!(matches!(
        fit(image.borrow(&rounded)),
        Err(GprError::InvalidDistance { .. })
    ));
    let tidied = fit(image.borrow(&rounded).tidy(1e-9).expect("tidy")).expect("tidied");
    let by_hand = fit(image.borrow(&fixed)).expect("by hand");
    assert_close(
        tidied.neg_log_marginal_likelihood().expect("nlml"),
        by_hand.neg_log_marginal_likelihood().expect("nlml"),
        1e-13,
    );
    let got = tidied.predict([image.borrow(&cross)], q).expect("predict");
    let expect = by_hand.predict([image.borrow(&cross)], q).expect("predict");
    assert_pred(&got, &expect, 1e-13);
    // The same on an ARD slot, each block rounded as above.
    let (bands, ard) = ArdDistance::from_leaf(RbfArdKernel::new(&[0.8, 1.4]).expect("ell"));
    let fit = |source: DistanceSource<'_>| {
        Svgp::new(ard.clone(), lik())
            .factor([source], n, &y, &ind)
            .map_err(|(_, e)| e)
            .expect("fit")
    };
    let blocks: Vec<Vec<f64>> = (0..2).map(|k| case.blocks(k).0).collect();
    let round = |b: &[f64]| {
        let mut b = b.to_vec();
        b[outside + n] = -1e-13;
        b[ind[0] + n] += 3e-13;
        b
    };
    let repair = |b: &[f64]| {
        let mut b: Vec<f64> = round(b).iter().map(|v| v.max(0.0)).collect();
        let (ab, ba) = (ind[1], ind[0] + n);
        let mean = 0.5 * (b[ab] + b[ba]);
        b[ab] = mean;
        b[ba] = mean;
        b
    };
    let rounded: Vec<Vec<f64>> = blocks.iter().map(|b| round(b)).collect();
    let fixed: Vec<Vec<f64>> = blocks.iter().map(|b| repair(b)).collect();
    let ard_cross: Vec<Vec<f64>> = (0..2).map(|k| case.blocks(k).1).collect();
    let refs: Vec<&[f64]> = ard_cross.iter().map(Vec::as_slice).collect();
    let got = fit(bands.from_vecs(rounded).tidy(1e-9).expect("tidy"))
        .predict([bands.borrow(&refs)], q)
        .expect("predict");
    let expect = fit(bands.from_vecs(fixed))
        .predict([bands.borrow(&refs)], q)
        .expect("predict");
    assert_pred(&got, &expect, 1e-13);
}

/// An input transform maps the coordinate part of a model with points:
/// fitted on raw coordinates with `MinMaxInput`, an SGPR and an SVGP
/// predict as the same models fitted on the mapped coordinates, and report
/// the inducing points in raw coordinates.
#[test]
fn an_input_transform_maps_the_coordinates_of_a_model_with_points() {
    let case = Case::standard(2);
    let (n, q, y) = (case.n, case.q, case.y());
    let image = ScalarDistance::new();
    let kernel = || {
        image.kernel(RbfKernel::new(0.7).expect("ell"))
            * KernelSpec::from(RbfKernel::new(1.3).expect("ell"))
    };
    let (train, cross, _) = case.blocks(0);
    let (x, xq) = (&case.cols[1], &case.qcols[1]);
    let map = MinMaxInput::new().fit(x, n, 1).expect("map");
    let (mut mx, mut mq) = (x.clone(), xq.clone());
    map.apply(&mut mx, n, 1).expect("apply");
    map.apply(&mut mq, q, 1).expect("apply");
    let raw_z = case.at_inducing(x);

    let fitted = Sgpr::new(kernel(), lik())
        .with_optimizer(Fixed)
        .with_input_transform(MinMaxInput::new())
        .factor([image.from_vec(train.clone())], n, x, 1, &y, &case.inducing)
        .expect("transformed");
    let reference = Sgpr::new(kernel(), lik())
        .with_optimizer(Fixed)
        .factor(
            [image.from_vec(train.clone())],
            n,
            &mx,
            1,
            &y,
            &case.inducing,
        )
        .expect("mapped");
    assert_pred(
        &fitted
            .predict([image.borrow(&cross)], xq, q, 1)
            .expect("predict"),
        &reference
            .predict([image.borrow(&cross)], &mq, q, 1)
            .expect("predict"),
        1e-12,
    );
    assert_slice_close(fitted.z(), &raw_z, 1e-12);

    let fitted = Svgp::new(kernel(), lik())
        .with_input_transform(MinMaxInput::new())
        .factor([image.from_vec(train.clone())], n, x, 1, &y, &case.inducing)
        .expect("transformed");
    let reference = Svgp::new(kernel(), lik())
        .factor([image.from_vec(train)], n, &mx, 1, &y, &case.inducing)
        .expect("mapped");
    assert_pred(
        &fitted
            .predict([image.borrow(&cross)], xq, q, 1)
            .expect("predict"),
        &reference
            .predict([image.borrow(&cross)], &mq, q, 1)
            .expect("predict"),
        1e-12,
    );
    assert_slice_close(fitted.z(), &raw_z, 1e-12);
}
