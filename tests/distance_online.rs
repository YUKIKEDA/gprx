//! Online insert and delete on supplied squared distances against a model
//! factored afresh on the live points. Public API only.

mod common;

use common::assert_close;
use gprx::kernel::{
    ArdDistance, DistanceFill, DistanceKernel, DistanceSource, KernelScalar, KernelSpec,
    RbfArdKernel, RbfKernel, ScalarDistance, WithPoints,
};
use gprx::{
    DoublePrecision, FittedGpr, Fixed, GaussianLikelihood, GpScalar, Gpr, GprError, MixedPrecision,
    OnlineGpr, Prediction, PromoteStorage, ReevaluateKernel, SinglePrecision,
};

/// Points in the pool; the model starts on the first `START`.
const POOL: usize = 12;
const START: usize = 3;
const M: usize = 3;
const DIMS: usize = 3;

/// Coordinate `k` of pool point `i`, and of query `q` (offset by a half).
fn at(k: usize, i: f64) -> f64 {
    (i * (0.41 + 0.17 * k as f64)).sin() * (1.0 + 0.5 * k as f64)
}

fn point(k: usize, i: usize) -> f64 {
    at(k, i as f64)
}

fn query(k: usize, q: usize) -> f64 {
    at(k, q as f64 + 0.5)
}

fn target(i: usize) -> f64 {
    (i as f64 * 0.7).cos()
}

/// Column-major `rows.len() × cols.len()` squared differences of coordinate `k`.
fn sq(k: usize, rows: &[f64], cols: &[f64]) -> Vec<f64> {
    let _ = k;
    let mut out = Vec::with_capacity(rows.len() * cols.len());
    for c in cols {
        for r in rows {
            out.push((r - c) * (r - c));
        }
    }
    out
}

fn coords(k: usize, live: &[usize]) -> Vec<f64> {
    live.iter().map(|&i| point(k, i)).collect()
}

fn queries(k: usize) -> Vec<f64> {
    (0..M).map(|q| query(k, q)).collect()
}

/// Per-dimension blocks between `rows` and `cols` (sets of pool indices,
/// or the queries when `None`).
fn blocks(rows: &[usize], cols: Option<&[usize]>) -> Vec<Vec<f64>> {
    (0..DIMS)
        .map(|k| {
            let r = coords(k, rows);
            let c = cols.map_or_else(|| queries(k), |c| coords(k, c));
            sq(k, &r, &c)
        })
        .collect()
}

fn summed(blocks: &[Vec<f64>]) -> Vec<f64> {
    (0..blocks[0].len())
        .map(|i| blocks.iter().map(|b| b[i]).sum())
        .collect()
}

/// The fill of the summed squares between pool `rows` and pool `cols`.
struct Pool<'a> {
    rows: &'a [usize],
    cols: &'a [usize],
    dims: usize,
    per_dim: bool,
}

impl DistanceFill for Pool<'_> {
    fn fill_column(&self, col: usize, rows: std::ops::Range<usize>, out: &mut [f64]) {
        let len = rows.len();
        for (r, i) in rows.enumerate() {
            let (a, b) = (self.rows[i], self.cols[col]);
            if self.per_dim {
                for k in 0..self.dims {
                    let d = point(k, a) - point(k, b);
                    out[k * len + r] = d * d;
                }
            } else {
                out[r] = (0..self.dims)
                    .map(|k| {
                        let d = point(k, a) - point(k, b);
                        d * d
                    })
                    .sum();
            }
        }
    }
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn lik() -> GaussianLikelihood {
    GaussianLikelihood::new(0.05).expect("noise")
}

fn assert_pred<T: KernelScalar>(got: &Prediction<T>, expect: &Prediction<T>, tol: f64) {
    assert_eq!(got.mean.len(), expect.mean.len());
    for (g, e) in got.mean.iter().zip(&expect.mean) {
        assert_close(g.to_f64(), e.to_f64(), tol);
    }
    for (g, e) in got.variance.iter().zip(&expect.variance) {
        assert_close(g.to_f64(), e.to_f64(), tol);
    }
}

/// Which source an insert binds its column through.
#[derive(Clone, Copy, Debug)]
enum Kind {
    Borrow,
    Slice,
    Vec,
    Fill,
}

const KINDS: [Kind; 4] = [Kind::Borrow, Kind::Slice, Kind::Vec, Kind::Fill];

/// The steps every model takes: inserts that grow the store past two
/// capacities, deletes at the head, the middle and the tail, and inserts
/// into the compacted store.
#[derive(Clone, Copy, Debug)]
enum Step {
    Insert(usize),
    Delete(usize),
}

fn steps() -> Vec<Step> {
    let mut steps: Vec<Step> = (START..9).map(Step::Insert).collect();
    steps.extend([Step::Delete(0), Step::Delete(3), Step::Delete(usize::MAX)]);
    steps.extend((9..POOL).map(Step::Insert));
    steps.push(Step::Delete(1));
    steps
}

/// A kernel on one slot of summed squares, or on one ARD slot of `DIMS`
/// blocks, and the slot its sources bind to.
struct Case {
    ard: bool,
    kernel: DistanceKernel,
    source: Source,
}

impl Case {
    #[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
    fn new(ard: bool) -> Self {
        if ard {
            let (bands, kernel) =
                ArdDistance::from_leaf(RbfArdKernel::new(&[0.8, 1.4, 2.0]).expect("ell"));
            Self {
                ard,
                kernel,
                source: Source::Ard(bands),
            }
        } else {
            let image = ScalarDistance::new();
            Self {
                ard,
                kernel: image.kernel(RbfKernel::new(1.3).expect("ell")),
                source: Source::Scalar(image),
            }
        }
    }
}

#[derive(Clone)]
enum Source {
    Scalar(ScalarDistance),
    Ard(ArdDistance),
}

impl Source {
    /// Binds the blocks `b` (per dimension) through `kind`; `fill` is the
    /// fill of the same blocks.
    fn bind<'a>(
        &self,
        kind: Kind,
        b: &'a [Vec<f64>],
        sum: &'a [f64],
        refs: &'a [&'a [f64]],
        fill: &'a Pool<'a>,
    ) -> DistanceSource<'a> {
        match self {
            Source::Scalar(image) => match kind {
                Kind::Borrow => image.borrow(sum),
                Kind::Slice => image.from_slice(sum),
                Kind::Vec => image.from_vec(sum.to_vec()),
                Kind::Fill => image.fill(fill),
            },
            Source::Ard(bands) => match kind {
                Kind::Borrow => bands.borrow(refs),
                Kind::Slice => bands.from_slices(refs),
                Kind::Vec => bands.from_vecs(b.to_vec()),
                Kind::Fill => bands.fill(fill),
            },
        }
    }
}

/// Factors the model afresh on `live`.
#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn fresh<P: GpScalar>(case: &Case, live: &[usize]) -> FittedGpr<Fixed, P, DistanceKernel> {
    let (kernel, source) = (case.kernel.clone(), &case.source);
    let b = blocks(live, Some(live));
    let sum = summed(&b);
    let refs: Vec<&[f64]> = b.iter().map(Vec::as_slice).collect();
    let fill = Pool {
        rows: live,
        cols: live,
        dims: DIMS,
        per_dim: case.ard,
    };
    let y: Vec<f64> = live.iter().map(|&i| target(i)).collect();
    Gpr::new(kernel, lik())
        .with_precision::<P>()
        .with_optimizer(Fixed)
        .factor(
            [source.bind(Kind::Vec, &b, &sum, &refs, &fill)],
            live.len(),
            &y,
        )
        .map_err(|(_, e)| e)
        .expect("fresh")
}

/// Checks `online` on `live` against a fresh factor: predictions, the
/// NLML, its gradient (read from the kept training squares), and a refit
/// from the kept squares.
#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn check<P: GpScalar>(
    case: &Case,
    online: &mut OnlineGpr<Fixed, P, DistanceKernel>,
    live: &[usize],
    tol: f64,
) where
    P::Refine: KernelScalar,
{
    let source = &case.source;
    let mut expect = fresh::<P>(case, live);
    assert_eq!(online.n(), live.len());
    let b = blocks(live, None);
    let sum = summed(&b);
    let refs: Vec<&[f64]> = b.iter().map(Vec::as_slice).collect();
    let fill = Pool {
        rows: live,
        cols: &[],
        dims: DIMS,
        per_dim: case.ard,
    };
    let cross = || source.bind(Kind::Borrow, &b, &sum, &refs, &fill);
    let got = online.predict([cross()], M).expect("online");
    let want = expect.predict([cross()], M).expect("fresh");
    assert_pred(&got, &want, tol);
    let mut out = Prediction::default();
    online.predict_into([cross()], M, &mut out).expect("into");
    assert_pred(&out, &want, tol);
    assert_close(
        online.neg_log_marginal_likelihood().expect("nlml"),
        expect.neg_log_marginal_likelihood().expect("nlml"),
        tol,
    );
    let p = online.num_params();
    let mut theta = vec![0.0; p];
    online.get_params(&mut theta).expect("theta");
    let (mut go, mut ge) = (vec![0.0; p], vec![0.0; p]);
    let vo = online
        .value_and_gradient_into(&theta, &mut go)
        .expect("grad");
    let ve = expect
        .value_and_gradient_into(&theta, &mut ge)
        .expect("grad");
    assert_close(vo, ve, tol);
    for (g, e) in go.iter().zip(&ge) {
        assert_close(*g, *e, tol * 10.0);
    }
}

/// Runs [`steps`] on `case`, binding each insert through the next kind,
/// and checks every step against a fresh factor.
#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn run<P: GpScalar>(case: &Case, tol: f64)
where
    P::Refine: KernelScalar,
{
    let mut live: Vec<usize> = (0..START).collect();
    let source = &case.source;
    let mut online = fresh::<P>(case, &live).into_online().expect("online");
    check(case, &mut online, &live, tol);
    for (s, step) in steps().into_iter().enumerate() {
        match step {
            Step::Insert(p) => {
                let new = [p];
                let b = blocks(&live, Some(&new));
                let sum = summed(&b);
                let refs: Vec<&[f64]> = b.iter().map(Vec::as_slice).collect();
                let fill = Pool {
                    rows: &live,
                    cols: &new,
                    dims: DIMS,
                    per_dim: case.ard,
                };
                let kind = KINDS[s % KINDS.len()];
                let id = online
                    .insert([source.bind(kind, &b, &sum, &refs, &fill)], target(p))
                    .expect("insert");
                assert_eq!(online.point_ids().last(), Some(&id));
                live.push(p);
            }
            Step::Delete(index) => {
                let index = index.min(live.len() - 1);
                let id = online.point_ids()[index];
                online.delete(id).expect("delete");
                live.remove(index);
            }
        }
        check(case, &mut online, &live, tol);
    }
    // A refit reads the kept training squares, compacted and grown.
    online.refit().expect("refit");
    check(case, &mut online, &live, tol);
}

#[test]
fn scalar_insert_and_delete_match_a_fresh_factor() {
    run::<DoublePrecision>(&Case::new(false), 1e-10);
}

#[test]
fn ard_insert_and_delete_match_a_fresh_factor() {
    run::<DoublePrecision>(&Case::new(true), 1e-10);
}

#[test]
fn single_and_mixed_precision_insert_and_delete_match_a_fresh_factor() {
    run::<SinglePrecision>(&Case::new(false), 2e-3);
    run::<SinglePrecision>(&Case::new(true), 2e-3);
    run::<MixedPrecision<PromoteStorage>>(&Case::new(false), 1e-4);
    run::<MixedPrecision<ReevaluateKernel>>(&Case::new(true), 1e-4);
}

/// A distance slot times a coordinate leaf: the insert takes the new
/// point's coordinates next to its column.
#[test]
fn with_points_insert_and_delete_match_a_fresh_factor() {
    let image = ScalarDistance::new();
    let kernel = image.kernel(RbfKernel::new(1.3).expect("ell"))
        * KernelSpec::from(RbfKernel::new(0.7).expect("ell"));
    // The coordinate leaf reads coordinate 1; the slot the sum of the rest.
    let x = |live: &[usize]| coords(1, live);
    let dist = |rows: &[usize], cols: &[f64], ck: &dyn Fn(usize) -> Vec<f64>| {
        let r0 = coords(0, rows);
        let r2 = coords(2, rows);
        let c0 = ck(0);
        let c2 = ck(2);
        summed(&[sq(0, &r0, &c0), sq(2, &r2, &c2)])
            .into_iter()
            .take(rows.len() * cols.len())
            .collect::<Vec<f64>>()
    };
    let fit = |live: &[usize]| -> FittedGpr<Fixed, DoublePrecision, DistanceKernel<WithPoints>> {
        let train = dist(live, &x(live), &|k| coords(k, live));
        let y: Vec<f64> = live.iter().map(|&i| target(i)).collect();
        Gpr::new(kernel.clone(), lik())
            .with_optimizer(Fixed)
            .factor([image.from_vec(train)], live.len(), &x(live), 1, &y)
            .map_err(|(_, e)| e)
            .expect("fit")
    };
    let mut live: Vec<usize> = (0..START).collect();
    let mut online = fit(&live).into_online().expect("online");
    let xs = queries(1);
    for step in steps() {
        match step {
            Step::Insert(p) => {
                let column = dist(&live, &[point(1, p)], &|k| vec![point(k, p)]);
                online
                    .insert([image.borrow(&column)], &[point(1, p)], target(p))
                    .expect("insert");
                live.push(p);
            }
            Step::Delete(index) => {
                let index = index.min(live.len() - 1);
                online.delete(online.point_ids()[index]).expect("delete");
                live.remove(index);
            }
        }
        assert_eq!(online.x(), x(&live).as_slice());
        let cross = dist(&live, &xs, &queries);
        let got = online
            .predict([image.borrow(&cross)], &xs, M, 1)
            .expect("online");
        let want = fit(&live)
            .predict([image.borrow(&cross)], &xs, M, 1)
            .expect("fresh");
        assert_pred(&got, &want, 1e-10);
    }
    online.refit().expect("refit");
    let cross = dist(&live, &xs, &queries);
    let got = online
        .predict([image.borrow(&cross)], &xs, M, 1)
        .expect("online");
    let want = fit(&live)
        .predict([image.borrow(&cross)], &xs, M, 1)
        .expect("fresh");
    assert_pred(&got, &want, 1e-10);
}

/// A column that is the wrong length, negative, or not finite is refused
/// before any change; the model inserts the right column after.
#[test]
fn an_invalid_column_is_refused_and_leaves_the_model_as_it_was() {
    let live: Vec<usize> = (0..START).collect();
    for ard in [false, true] {
        let case = Case::new(ard);
        let source = &case.source;
        let mut online = fresh::<DoublePrecision>(&case, &live)
            .into_online()
            .expect("online");
        let before = online.point_ids().to_vec();
        let new = [START];
        let b = blocks(&live, Some(&new));
        let fill = Pool {
            rows: &live,
            cols: &new,
            dims: DIMS,
            per_dim: ard,
        };
        let short: Vec<Vec<f64>> = b.iter().map(|c| c[..START - 1].to_vec()).collect();
        let mut negative = b.clone();
        for block in &mut negative {
            block[1] = -0.5;
        }
        let mut nan = b.clone();
        nan[0][2] = f64::NAN;
        for (bad, is_shape) in [(short, true), (negative, false), (nan, false)] {
            let sum = summed(&bad);
            let refs: Vec<&[f64]> = bad.iter().map(Vec::as_slice).collect();
            for kind in [Kind::Borrow, Kind::Vec] {
                let err = online
                    .insert([source.bind(kind, &bad, &sum, &refs, &fill)], 0.5)
                    .expect_err("invalid column");
                if is_shape {
                    assert!(matches!(err, GprError::LengthMismatch { .. }), "{err:?}");
                } else {
                    assert!(matches!(err, GprError::InvalidDistance { .. }), "{err:?}");
                }
                assert_eq!(online.point_ids(), before.as_slice());
            }
        }
        let err = online
            .insert([source.bind(Kind::Fill, &b, &[], &[], &fill)], f64::NAN)
            .expect_err("nan target");
        assert!(matches!(err, GprError::NonFiniteInput), "{err:?}");
        let err = online.insert([], 0.5).expect_err("no slot");
        assert!(matches!(err, GprError::LengthMismatch { .. }), "{err:?}");
        assert_eq!(online.n(), START);
        online
            .insert(
                [source.bind(Kind::Fill, &b, &[], &[], &fill)],
                target(START),
            )
            .expect("insert");
        check(&case, &mut online, &[0, 1, 2, START], 1e-10);
    }
}
