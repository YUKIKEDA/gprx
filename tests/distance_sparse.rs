//! SGPR and SVGP on supplied squared distances against the same models on
//! coordinates, with the inducing points at the chosen training rows.
//! Public API only.

mod common;

use std::num::{NonZeroU64, NonZeroUsize};

use common::{assert_close, assert_slice_close};
use gprx::kernel::{
    ArdDistance, ConstantKernel, KernelSpec, RbfArdKernel, RbfKernel, ScalarDistance,
};
use gprx::{Adam, Fixed, GaussianLikelihood, GprError, Prediction, Sgpr, SinglePrecision, Svgp};

const N: usize = 8;
const Q: usize = 3;
const TOL: f64 = 1e-9;
const INDUCING: [usize; 3] = [0, 3, 5];

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

fn sum(blocks: &[Vec<f64>]) -> Vec<f64> {
    (0..blocks[0].len())
        .map(|i| blocks.iter().map(|b| b[i]).sum())
        .collect()
}

/// The rows `INDUCING` of the column-major `N × d` `x`.
fn rows(x: &[f64], d: usize) -> Vec<f64> {
    (0..d)
        .flat_map(|k| INDUCING.iter().map(move |&i| x[i + k * N]))
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

/// Two-dimensional training and query samples and their squared distances.
struct Data {
    x: Vec<f64>,
    xs: Vec<f64>,
    train: Vec<f64>,
    cross: Vec<f64>,
    query: Vec<f64>,
}

fn data() -> Data {
    let (c0, c1) = (coord(0, N, 0.0), coord(1, N, 0.0));
    let (q0, q1) = (coord(0, Q, 0.5), coord(1, Q, 0.5));
    Data {
        x: [c0.clone(), c1.clone()].concat(),
        xs: [q0.clone(), q1.clone()].concat(),
        train: sum(&[sq(&c0, &c0), sq(&c1, &c1)]),
        cross: sum(&[sq(&c0, &q0), sq(&c1, &q1)]),
        query: sum(&[sq(&q0, &q0), sq(&q1, &q1)]),
    }
}

#[test]
fn sgpr_on_supplied_distances_matches_coordinates_at_the_chosen_rows() {
    let data = data();
    let y = targets();
    let rbf = RbfKernel::new(0.9).expect("ell");
    let scale = ConstantKernel::new(0.8).expect("constant");
    // A product: the derivatives go through the nested product buffers.
    let mut coords = Sgpr::new(KernelSpec::from(scale) * KernelSpec::from(rbf), lik())
        .with_optimizer(Fixed)
        .factor(&data.x, N, 2, &y, &rows(&data.x, 2), INDUCING.len())
        .expect("coords");
    let image = ScalarDistance::new();
    let mut dist = Sgpr::new(scale * image.kernel(rbf), lik())
        .with_optimizer(Fixed)
        .factor([image.from_vec(data.train.clone())], N, &y, &INDUCING)
        .expect("distances");
    assert_eq!(dist.inducing(), &INDUCING);
    assert_eq!(dist.m(), INDUCING.len());
    assert_close(
        dist.neg_log_marginal_likelihood().expect("nlml"),
        coords.neg_log_marginal_likelihood().expect("nlml"),
        TOL,
    );
    let n_params = coords.num_params();
    assert_eq!(dist.num_params(), n_params);
    let mut params = vec![0.0; n_params];
    coords.get_params(&mut params).expect("params");
    let (mut g_d, mut g_c) = (vec![0.0; n_params], vec![0.0; n_params]);
    dist.value_and_gradient_into(&params, &mut g_d)
        .expect("grad");
    coords
        .value_and_gradient_into(&params, &mut g_c)
        .expect("grad");
    assert_slice_close(&g_d, &g_c, TOL);
    let (mut h_d, mut h_c) = (
        vec![0.0; n_params * n_params],
        vec![0.0; n_params * n_params],
    );
    dist.hessian_into(&params, &mut h_d).expect("hess");
    coords.hessian_into(&params, &mut h_c).expect("hess");
    assert_slice_close(&h_d, &h_c, TOL);

    let got = dist
        .predict([image.borrow(&data.cross)], Q)
        .expect("predict");
    let expect = coords.predict(&data.xs, Q, 2).expect("predict");
    assert_pred(&got, &expect, TOL);
    let mut into = Prediction::default();
    dist.predict_into([image.borrow(&data.cross)], Q, &mut into)
        .expect("predict");
    assert_pred(&into, &expect, TOL);
    let cov = dist
        .predict_covariance([image.borrow(&data.cross)], [image.borrow(&data.query)], Q)
        .expect("cov");
    let cov_ref = coords.predict_covariance(&data.xs, Q, 2).expect("cov");
    assert_slice_close(&cov.covariance, &cov_ref.covariance, TOL);
    let draws = dist
        .sample(
            [image.borrow(&data.cross)],
            [image.borrow(&data.query)],
            Q,
            4,
            11,
        )
        .expect("sample");
    let draws_ref = coords.sample(&data.xs, Q, 2, 4, 11).expect("sample");
    assert_slice_close(&draws, &draws_ref, 1e-7);
}

#[test]
fn sgpr_search_on_supplied_distances_matches_the_coordinate_search() {
    let data = data();
    let y = targets();
    let rbf = RbfKernel::new(0.9).expect("ell");
    let coords = Sgpr::new(KernelSpec::from(rbf), lik())
        .fit(&data.x, N, 2, &y, &rows(&data.x, 2), INDUCING.len())
        .expect("coords");
    let image = ScalarDistance::new();
    let dist = Sgpr::new(image.kernel(rbf), lik())
        .fit([image.from_vec(data.train)], N, &y, &INDUCING)
        .expect("distances");
    let mut a = vec![0.0; coords.num_params()];
    let mut b = vec![0.0; coords.num_params()];
    coords.get_params(&mut a).expect("params");
    dist.get_params(&mut b).expect("params");
    assert_slice_close(&b, &a, 1e-6);
}

#[test]
fn sgpr_ard_on_supplied_squared_differences_matches_coordinates() {
    let cols: Vec<Vec<f64>> = (0..3).map(|k| coord(k, N, 0.0)).collect();
    let qcols: Vec<Vec<f64>> = (0..3).map(|k| coord(k, Q, 0.5)).collect();
    let x = cols.concat();
    let y = targets();
    let ard = RbfArdKernel::new(&[0.8, 1.4, 2.0]).expect("ell");
    let coords = Sgpr::new(KernelSpec::from(ard.clone()), lik())
        .with_optimizer(Fixed)
        .factor(&x, N, 3, &y, &rows(&x, 3), INDUCING.len())
        .expect("coords");
    let bands = ArdDistance::new(3).expect("dims");
    let train: Vec<Vec<f64>> = cols.iter().map(|c| sq(c, c)).collect();
    let cross: Vec<Vec<f64>> = cols.iter().zip(&qcols).map(|(c, q)| sq(c, q)).collect();
    let dist = Sgpr::new(bands.kernel(ard).expect("dims"), lik())
        .with_optimizer(Fixed)
        .factor([bands.from_vecs(train)], N, &y, &INDUCING)
        .expect("distances");
    let refs: Vec<&[f64]> = cross.iter().map(Vec::as_slice).collect();
    let got = dist.predict([bands.borrow(&refs)], Q).expect("predict");
    let expect = coords.predict(&qcols.concat(), Q, 3).expect("predict");
    assert_pred(&got, &expect, TOL);
    assert_close(
        dist.neg_log_marginal_likelihood().expect("nlml"),
        coords.neg_log_marginal_likelihood().expect("nlml"),
        TOL,
    );
}

#[test]
fn sgpr_distance_rbf_times_coordinate_rbf_is_one_ard_rbf() {
    let (c0, c1) = (coord(0, N, 0.0), coord(1, N, 0.0));
    let (q0, q1) = (coord(0, Q, 0.5), coord(1, Q, 0.5));
    let y = targets();
    let x = [c0.clone(), c1.clone()].concat();
    let reference = Sgpr::new(
        KernelSpec::from(RbfArdKernel::new(&[0.7, 1.3]).expect("ell")),
        lik(),
    )
    .with_optimizer(Fixed)
    .factor(&x, N, 2, &y, &rows(&x, 2), INDUCING.len())
    .expect("reference");
    let image = ScalarDistance::new();
    let kernel = image.kernel(RbfKernel::new(0.7).expect("ell"))
        * KernelSpec::from(RbfKernel::new(1.3).expect("ell"));
    let model = Sgpr::new(kernel, lik())
        .with_optimizer(Fixed)
        .factor([image.from_vec(sq(&c0, &c0))], N, &c1, 1, &y, &INDUCING)
        .expect("model");
    assert_eq!(model.z(), &rows(&c1, 1)[..]);
    let cross = sq(&c0, &q0);
    let got = model
        .predict([image.borrow(&cross)], &q1, Q, 1)
        .expect("predict");
    let xs = [q0.clone(), q1.clone()].concat();
    assert_pred(&got, &reference.predict(&xs, Q, 2).expect("predict"), TOL);
    let cov = model
        .predict_covariance(
            [image.borrow(&cross)],
            [image.borrow(&sq(&q0, &q0))],
            &q1,
            Q,
            1,
        )
        .expect("cov");
    let cov_ref = reference.predict_covariance(&xs, Q, 2).expect("cov");
    assert_slice_close(&cov.covariance, &cov_ref.covariance, TOL);
}

#[test]
fn svgp_on_supplied_distances_matches_coordinates_at_the_chosen_rows() {
    let data = data();
    let y = targets();
    let rbf = RbfKernel::new(0.9).expect("ell");
    let mut coords = Svgp::new(KernelSpec::from(rbf), lik())
        .factor(&data.x, N, 2, &y, &rows(&data.x, 2), INDUCING.len())
        .expect("coords");
    let image = ScalarDistance::new();
    let mut dist = Svgp::new(image.kernel(rbf), lik())
        .factor([image.from_vec(data.train.clone())], N, &y, &INDUCING)
        .expect("distances");
    assert_close(
        dist.neg_elbo().expect("elbo"),
        coords.neg_elbo().expect("elbo"),
        TOL,
    );
    let n_params = coords.num_params();
    let mut params = vec![0.0; n_params];
    coords.get_params(&mut params).expect("params");
    let (mut g_d, mut g_c) = (vec![0.0; n_params], vec![0.0; n_params]);
    dist.value_and_gradient_into(&params, &mut g_d)
        .expect("grad");
    coords
        .value_and_gradient_into(&params, &mut g_c)
        .expect("grad");
    assert_slice_close(&g_d, &g_c, TOL);
    let got = dist
        .predict([image.borrow(&data.cross)], Q)
        .expect("predict");
    assert_pred(&got, &coords.predict(&data.xs, Q, 2).expect("predict"), TOL);
    let cov = dist
        .predict_covariance([image.borrow(&data.cross)], [image.borrow(&data.query)], Q)
        .expect("cov");
    let cov_ref = coords.predict_covariance(&data.xs, Q, 2).expect("cov");
    assert_slice_close(&cov.covariance, &cov_ref.covariance, TOL);

    let adam = Adam::new()
        .with_batch_size(NonZeroUsize::new(3).expect("batch"))
        .with_epochs(NonZeroU64::new(5).expect("epochs"))
        .with_seed(4);
    let coords = Svgp::new(KernelSpec::from(rbf), lik())
        .with_optimizer(adam.clone())
        .fit(&data.x, N, 2, &y, &rows(&data.x, 2), INDUCING.len())
        .expect("coords");
    let dist = Svgp::new(image.kernel(rbf), lik())
        .with_optimizer(adam)
        .fit([image.from_vec(data.train)], N, &y, &INDUCING)
        .expect("distances");
    let (mut a, mut b) = (vec![0.0; n_params], vec![0.0; n_params]);
    coords.get_params(&mut a).expect("params");
    dist.get_params(&mut b).expect("params");
    assert_slice_close(&b, &a, 1e-8);
}

#[test]
fn svgp_distance_rbf_times_coordinate_rbf_is_one_ard_rbf() {
    let (c0, c1) = (coord(0, N, 0.0), coord(1, N, 0.0));
    let (q0, q1) = (coord(0, Q, 0.5), coord(1, Q, 0.5));
    let y = targets();
    let x = [c0.clone(), c1.clone()].concat();
    let reference = Svgp::new(
        KernelSpec::from(RbfArdKernel::new(&[0.7, 1.3]).expect("ell")),
        lik(),
    )
    .factor(&x, N, 2, &y, &rows(&x, 2), INDUCING.len())
    .expect("reference");
    let image = ScalarDistance::new();
    let kernel = image.kernel(RbfKernel::new(0.7).expect("ell"))
        * KernelSpec::from(RbfKernel::new(1.3).expect("ell"));
    let model = Svgp::new(kernel, lik())
        .factor([image.from_vec(sq(&c0, &c0))], N, &c1, 1, &y, &INDUCING)
        .expect("model");
    let got = model
        .predict([image.borrow(&sq(&c0, &q0))], &q1, Q, 1)
        .expect("predict");
    let xs = [q0, q1].concat();
    assert_pred(&got, &reference.predict(&xs, Q, 2).expect("predict"), TOL);
}

#[test]
fn single_precision_sgpr_reads_the_supplied_distances() {
    let data = data();
    let y = targets();
    let rbf = RbfKernel::new(0.9).expect("ell");
    let coords = Sgpr::new(KernelSpec::from(rbf), lik())
        .with_optimizer(Fixed)
        .with_precision::<SinglePrecision>()
        .factor(&data.x, N, 2, &y, &rows(&data.x, 2), INDUCING.len())
        .expect("coords");
    let image = ScalarDistance::new();
    let dist = Sgpr::new(image.kernel(rbf), lik())
        .with_optimizer(Fixed)
        .with_precision::<SinglePrecision>()
        .factor([image.from_vec(data.train)], N, &y, &INDUCING)
        .expect("distances");
    let got = dist
        .predict([image.borrow(&data.cross)], Q)
        .expect("predict");
    let expect = coords.predict(&data.xs, Q, 2).expect("predict");
    for (a, b) in got.mean.iter().zip(&expect.mean) {
        assert!((a - b).abs() < 1e-4, "{a} vs {b}");
    }
}

#[test]
fn sparse_inputs_are_checked() {
    let data = data();
    let y = targets();
    let image = ScalarDistance::new();
    let kernel = image.kernel(RbfKernel::new(0.9).expect("ell"));
    let err = Sgpr::new(kernel.clone(), lik())
        .with_optimizer(Fixed)
        .factor([image.borrow(&data.train)], N, &y, &[0, N])
        .map(|_| ())
        .map_err(|(_, e)| e);
    assert!(
        matches!(err, Err(GprError::IndexOutOfRange { .. })),
        "{err:?}"
    );
    let err = Svgp::new(kernel.clone(), lik())
        .factor([image.borrow(&data.train)], N, &y, &[])
        .map(|_| ())
        .map_err(|(_, e)| e);
    assert!(matches!(err, Err(GprError::EmptyInput)), "{err:?}");
    let mut skew = data.train.clone();
    skew[1] += 0.5;
    let err = Sgpr::new(kernel.clone(), lik())
        .with_optimizer(Fixed)
        .factor([image.borrow(&skew)], N, &y, &INDUCING)
        .map(|_| ())
        .map_err(|(_, e)| e);
    assert!(
        matches!(err, Err(GprError::ShapeMismatch { .. })),
        "{err:?}"
    );
    let model = Sgpr::new(kernel, lik())
        .with_optimizer(Fixed)
        .factor([image.borrow(&data.train)], N, &y, &INDUCING)
        .expect("model");
    let mut query = data.query.clone();
    query[0] = 0.1;
    let err = model.predict_covariance([image.borrow(&data.cross)], [image.borrow(&query)], Q);
    assert!(
        matches!(err, Err(GprError::ShapeMismatch { .. })),
        "{err:?}"
    );
    let err = model.predict([image.borrow(&data.cross[1..])], Q);
    assert!(
        matches!(err, Err(GprError::LengthMismatch { .. })),
        "{err:?}"
    );
}

#[test]
fn a_sparse_kernel_with_coordinate_leaves_needs_a_feature_column() {
    let data = data();
    let y = targets();
    let image = ScalarDistance::new();
    let kernel = image.kernel(RbfKernel::new(0.9).expect("ell"))
        * KernelSpec::from(RbfKernel::new(1.1).expect("ell"));
    let err = Sgpr::new(kernel.clone(), lik())
        .with_optimizer(Fixed)
        .factor([image.borrow(&data.train)], N, &[], 0, &y, &INDUCING)
        .map(|_| ())
        .map_err(|(_, e)| e);
    assert!(matches!(err, Err(GprError::EmptyInput)), "{err:?}");
    let err = Svgp::new(kernel, lik())
        .factor([image.borrow(&data.train)], N, &[], 0, &y, &INDUCING)
        .map(|_| ())
        .map_err(|(_, e)| e);
    assert!(matches!(err, Err(GprError::EmptyInput)), "{err:?}");
}

#[test]
fn a_sparse_white_term_adds_its_diagonal_to_the_query_covariance() {
    use gprx::kernel::WhiteKernel;
    let data = data();
    let y = targets();
    let rbf = RbfKernel::new(0.9).expect("ell");
    let white = WhiteKernel::new(0.3).expect("white");
    let image = ScalarDistance::new();
    let coords = Sgpr::new(KernelSpec::from(rbf) + KernelSpec::from(white), lik())
        .with_optimizer(Fixed)
        .factor(&data.x, N, 2, &y, &rows(&data.x, 2), INDUCING.len())
        .expect("coords");
    let dist = Sgpr::new(image.kernel(rbf) + white, lik())
        .with_optimizer(Fixed)
        .factor([image.borrow(&data.train)], N, &y, &INDUCING)
        .expect("distances");
    let got = dist
        .predict_covariance([image.borrow(&data.cross)], [image.borrow(&data.query)], Q)
        .expect("cov");
    let expect = coords.predict_covariance(&data.xs, Q, 2).expect("cov");
    assert_slice_close(&got.covariance, &expect.covariance, TOL);
    let coords = Svgp::new(KernelSpec::from(rbf) + KernelSpec::from(white), lik())
        .factor(&data.x, N, 2, &y, &rows(&data.x, 2), INDUCING.len())
        .expect("coords");
    let dist = Svgp::new(image.kernel(rbf) + white, lik())
        .factor([image.borrow(&data.train)], N, &y, &INDUCING)
        .expect("distances");
    let got = dist
        .predict_covariance([image.borrow(&data.cross)], [image.borrow(&data.query)], Q)
        .expect("cov");
    let expect = coords.predict_covariance(&data.xs, Q, 2).expect("cov");
    assert_slice_close(&got.covariance, &expect.covariance, TOL);
}

/// One source per slot of `slots`: a scalar slot reads `scalar`, an ARD slot
/// reads `blocks` (the same two dimensions for every ARD slot).
fn sources_for<'a>(
    slots: &[gprx::kernel::DistanceSlot],
    scalar: &'a [f64],
    blocks: &'a [&'a [f64]],
) -> Vec<gprx::kernel::DistanceSource<'a>> {
    slots
        .iter()
        .map(|slot| match slot {
            gprx::kernel::DistanceSlot::Scalar(s) => s.borrow(scalar),
            gprx::kernel::DistanceSlot::Ard(a) => a.borrow(blocks),
            _ => unreachable!("two slot kinds"),
        })
        .collect()
}

#[test]
fn every_distance_leaf_matches_coordinates_in_a_sparse_model_and_round_trips() {
    use gprx::kernel::{
        DistanceKernel, DistanceOnly, MaternArdKernel, MaternNu, PeriodicKernel,
        RationalQuadraticArdKernel, RationalQuadraticKernel,
    };
    use gprx::{DoublePrecision, FittedSgpr, FixedInducing, PersistRegistry};
    let data = data();
    let y = targets();
    let (c0, c1) = (coord(0, N, 0.0), coord(1, N, 0.0));
    let (q0, q1) = (coord(0, Q, 0.5), coord(1, Q, 0.5));
    let periodic = PeriodicKernel::new(1.1, 2.5).expect("periodic");
    let rq = RationalQuadraticKernel::new(0.9, 1.7).expect("rq");
    let matern = MaternArdKernel::new(&[0.8, 1.3], MaternNu::ThreeHalves).expect("matern");
    let rq_ard = RationalQuadraticArdKernel::new(&[1.2, 0.7], 0.9).expect("rq ard");
    let coords_kernel = KernelSpec::from(periodic)
        + KernelSpec::from(rq)
        + KernelSpec::from(matern.clone()) * KernelSpec::from(rq_ard.clone());
    let image = ScalarDistance::new();
    let bands = ArdDistance::new(2).expect("dims");
    let other = ArdDistance::new(2).expect("dims");
    let dist_kernel = image.kernel(periodic)
        + image.kernel(rq)
        + bands.kernel(matern).expect("dims") * other.kernel(rq_ard).expect("dims");
    let slots = dist_kernel.slots();
    let train_blocks = [sq(&c0, &c0), sq(&c1, &c1)];
    let train_refs: Vec<&[f64]> = train_blocks.iter().map(Vec::as_slice).collect();
    let cross_blocks = [sq(&c0, &q0), sq(&c1, &q1)];
    let cross_refs: Vec<&[f64]> = cross_blocks.iter().map(Vec::as_slice).collect();

    let mut coords = Sgpr::new(coords_kernel.clone(), lik())
        .with_optimizer(Fixed)
        .factor(&data.x, N, 2, &y, &rows(&data.x, 2), INDUCING.len())
        .expect("coords");
    let mut dist = Sgpr::new(dist_kernel.clone(), lik())
        .with_optimizer(Fixed)
        .factor(
            sources_for(&slots, &data.train, &train_refs),
            N,
            &y,
            &INDUCING,
        )
        .expect("distances");
    let n_params = coords.num_params();
    let mut params = vec![0.0; n_params];
    coords.get_params(&mut params).expect("params");
    let (mut g_d, mut g_c) = (vec![0.0; n_params], vec![0.0; n_params]);
    let v_d = dist
        .value_and_gradient_into(&params, &mut g_d)
        .expect("grad");
    let v_c = coords
        .value_and_gradient_into(&params, &mut g_c)
        .expect("grad");
    assert_close(v_d, v_c, TOL);
    assert_slice_close(&g_d, &g_c, 1e-8);
    let (mut h_d, mut h_c) = (
        vec![0.0; n_params * n_params],
        vec![0.0; n_params * n_params],
    );
    dist.hessian_into(&params, &mut h_d).expect("hess");
    coords.hessian_into(&params, &mut h_c).expect("hess");
    assert_slice_close(&h_d, &h_c, 1e-7);
    let expect = coords.predict(&data.xs, Q, 2).expect("predict");
    let got = dist
        .predict(sources_for(&slots, &data.cross, &cross_refs), Q)
        .expect("predict");
    assert_pred(&got, &expect, 1e-8);

    // The same leaves through save and load.
    let dir = std::env::temp_dir().join(format!(
        "gprx-distance-sparse-{}-leaves",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    dist.save(&dir).expect("save");
    type Model = FittedSgpr<Fixed, FixedInducing, DoublePrecision, DistanceKernel<DistanceOnly>>;
    let loaded = Model::load(&dir, &PersistRegistry::new()).expect("load");
    let loaded_slots = loaded.slots();
    let again = loaded
        .predict(sources_for(&loaded_slots, &data.cross, &cross_refs), Q)
        .expect("predict");
    assert_pred(&again, &got, 0.0);
    let _ = std::fs::remove_dir_all(&dir);

    // A search over the same leaves, and SVGP's Adam steps through them.
    let fitted = Sgpr::new(dist_kernel.clone(), lik())
        .fit(
            sources_for(&slots, &data.train, &train_refs),
            N,
            &y,
            &INDUCING,
        )
        .expect("fit");
    assert!(
        fitted
            .neg_log_marginal_likelihood()
            .expect("nlml")
            .is_finite()
    );
    let adam = Adam::new()
        .with_batch_size(NonZeroUsize::new(4).expect("batch"))
        .with_epochs(NonZeroU64::new(2).expect("epochs"))
        .with_seed(3);
    let svgp = Svgp::new(dist_kernel, lik())
        .with_optimizer(adam)
        .fit(
            sources_for(&slots, &data.train, &train_refs),
            N,
            &y,
            &INDUCING,
        )
        .expect("svgp");
    let pred = svgp
        .predict(sources_for(&slots, &data.cross, &cross_refs), Q)
        .expect("predict");
    assert!(pred.mean.iter().all(|v| v.is_finite()));
}

#[test]
fn sparse_fits_with_points_read_the_coordinates() {
    let data = data();
    let y = targets();
    let image = ScalarDistance::new();
    let kernel = image.kernel(RbfKernel::new(0.9).expect("ell"))
        * KernelSpec::from(RbfKernel::new(1.1).expect("ell"));
    let c2 = coord(2, N, 0.0);
    let q2 = coord(2, Q, 0.5);
    let sgpr = Sgpr::new(kernel.clone(), lik())
        .fit([image.borrow(&data.train)], N, &c2, 1, &y, &INDUCING)
        .expect("sgpr");
    assert_eq!(sgpr.inducing(), &INDUCING);
    let pred = sgpr
        .predict([image.borrow(&data.cross)], &q2, Q, 1)
        .expect("predict");
    assert!(pred.mean.iter().all(|v| v.is_finite()));
    let adam = Adam::new()
        .with_batch_size(NonZeroUsize::new(4).expect("batch"))
        .with_epochs(NonZeroU64::new(2).expect("epochs"))
        .with_seed(3);
    let svgp = Svgp::new(kernel, lik())
        .with_optimizer(adam)
        .fit([image.borrow(&data.train)], N, &c2, 1, &y, &INDUCING)
        .expect("svgp");
    let pred = svgp
        .predict([image.borrow(&data.cross)], &q2, Q, 1)
        .expect("predict");
    assert!(pred.mean.iter().all(|v| v.is_finite()));
}

/// Repeated inducing indices are allowed, as repeated rows of `Z` are: the
/// `K_mm` jitter retries make the singular `K_mm` factor.
#[test]
fn repeated_inducing_indices_are_allowed() {
    let data = data();
    let y = targets();
    let image = ScalarDistance::new();
    let kernel = image.kernel(RbfKernel::new(0.9).expect("ell"));
    let repeated = [0, 0, 3];
    let sgpr = Sgpr::new(kernel.clone(), lik())
        .with_optimizer(Fixed)
        .factor([image.borrow(&data.train)], N, &y, &repeated)
        .map_err(|(_, e)| e)
        .expect("sgpr");
    assert_eq!(sgpr.inducing(), &repeated);
    let svgp = Svgp::new(kernel, lik())
        .factor([image.borrow(&data.train)], N, &y, &repeated)
        .map_err(|(_, e)| e)
        .expect("svgp");
    for pred in [
        sgpr.predict([image.borrow(&data.cross)], Q)
            .expect("predict"),
        svgp.predict([image.borrow(&data.cross)], Q)
            .expect("predict"),
    ] {
        assert!(
            pred.mean
                .iter()
                .chain(&pred.variance)
                .all(|v| v.is_finite())
        );
    }
}
