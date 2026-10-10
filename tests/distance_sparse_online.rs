//! `OnlineSgpr` on supplied squared distances against the coordinate
//! `Sgpr` factored from scratch on the same points and inducing points,
//! after every insert and delete. Public API only.

mod common;

use common::distance::{coord, lik, to64};

use common::{assert_close, assert_slice_close};
use gprx::kernel::{
    ArdDistance, DistanceFill, DistanceKernel, DistanceOnly, DistanceSource, KernelScalar,
    KernelSpec, RbfArdKernel, RbfKernel, ScalarDistance,
};
use gprx::{
    DoublePrecision, Fixed, GpScalar, GprError, MixedPrecision, OnlineSgpr, PromoteStorage, Sgpr,
    SinglePrecision, SlotErrorKind,
};

/// Points the test can insert, and queries.
const TOTAL: usize = 16;
const Q: usize = 5;
const D: usize = 2;
const ELL: [f64; 2] = [0.8, 1.4];

/// Every sample's coordinates and target, and the queries.
struct World {
    cols: Vec<Vec<f64>>,
    y: Vec<f64>,
    queries: Vec<Vec<f64>>,
}

impl World {
    fn new() -> Self {
        Self {
            cols: (0..D).map(|k| coord(k, TOTAL, 0.0)).collect(),
            y: (0..TOTAL).map(|i| (i as f64 * 0.7).cos()).collect(),
            queries: (0..D).map(|k| coord(k, Q, 0.5)).collect(),
        }
    }

    /// Column-major `rows × cols` squared differences along dimension `k`
    /// between samples (or queries, `None`) by index.
    fn sq(&self, k: usize, rows: &[usize], cols: Option<&[usize]>) -> Vec<f64> {
        let a = &self.cols[k];
        let mut out = Vec::new();
        match cols {
            Some(cols) => {
                for &j in cols {
                    out.extend(rows.iter().map(|&i| (a[i] - a[j]).powi(2)));
                }
            }
            None => {
                for &qj in &self.queries[k] {
                    out.extend(rows.iter().map(|&i| (a[i] - qj).powi(2)));
                }
            }
        }
        out
    }

    /// The sum over `dims` of [`Self::sq`].
    fn summed(&self, dims: &[usize], rows: &[usize], cols: Option<&[usize]>) -> Vec<f64> {
        let blocks: Vec<Vec<f64>> = dims.iter().map(|&k| self.sq(k, rows, cols)).collect();
        (0..blocks[0].len())
            .map(|i| blocks.iter().map(|b| b[i]).sum())
            .collect()
    }

    /// Column-major coordinates of `rows` along `dims`.
    fn x(&self, dims: &[usize], rows: &[usize]) -> Vec<f64> {
        dims.iter()
            .flat_map(|&k| rows.iter().map(move |&i| self.cols[k][i]))
            .collect()
    }

    fn xq(&self, dims: &[usize]) -> Vec<f64> {
        dims.iter().flat_map(|&k| self.queries[k].clone()).collect()
    }

    fn y(&self, rows: &[usize]) -> Vec<f64> {
        rows.iter().map(|&i| self.y[i]).collect()
    }
}

/// The points and inducing points (sample indices) the online model holds,
/// in its buffer order.
struct Live {
    points: Vec<usize>,
    inducing: Vec<usize>,
}

impl Live {
    fn start() -> Self {
        Self {
            points: (0..10).collect(),
            inducing: vec![7, 0, 4, 9],
        }
    }

    /// Where sample `i` is in the buffer.
    #[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
    fn at(&self, i: usize) -> usize {
        self.points.iter().position(|&p| p == i).expect("live")
    }
}

/// One kernel of the comparison. Each is the ARD RBF of `ELL` over the
/// two coordinates, written as distance slots (and a coordinate leaf).
#[derive(Clone, Copy, Debug)]
enum Kind {
    /// One scalar slot of the summed squares (the isotropic RBF of
    /// `ELL[0]`, compared with the coordinate RBF of `ELL[0]`).
    Scalar,
    /// One ARD slot of both dimensions, from moved-in tables.
    Ard,
    /// The same ARD slot from borrowed tables (packed when bound).
    ArdBorrowed,
    /// A scalar slot of dimension 0 times a coordinate leaf of dimension 1.
    WithPoints,
    /// A scalar slot of dimension 0 times a one-dimensional ARD slot of
    /// dimension 1, their sources handed over ARD first.
    SlotPair,
    /// Two scalar slots, one per dimension.
    TwoScalars,
}

const KINDS: [Kind; 6] = [
    Kind::Scalar,
    Kind::Ard,
    Kind::ArdBorrowed,
    Kind::WithPoints,
    Kind::SlotPair,
    Kind::TwoScalars,
];

/// The squares a kind's slots read, one block per slot dimension, in the
/// order `wrap` takes them.
fn blocks_of(kind: Kind, world: &World, rows: &[usize], cols: Option<&[usize]>) -> Vec<Vec<f64>> {
    match kind {
        Kind::Scalar => vec![world.summed(&[0, 1], rows, cols)],
        Kind::Ard | Kind::ArdBorrowed | Kind::SlotPair | Kind::TwoScalars => {
            vec![world.sq(0, rows, cols), world.sq(1, rows, cols)]
        }
        Kind::WithPoints => vec![world.sq(0, rows, cols)],
    }
}

/// Runs `$body` with `$online` the online model of `$kind` on `$live`'s
/// points, `$blocks(rows, cols)` the raw squares between samples (`cols:
/// None` for the queries), `$wrap(blocks)` their sources, `$src` the two
/// together, `$coords` the coordinate model's kernel and dimensions, and
/// `$extra(rows)` the coordinates a `WithPoints` model takes (empty
/// otherwise).
macro_rules! with_online {
    ($p:ty, $kind:expr, $world:expr, $live:expr, |$online:ident, $src:ident, $blocks:ident, $wrap:ident, $extra:ident, $coords:ident| $body:block) => {{
        let world: &World = $world;
        let live: &Live = &$live;
        let kind: Kind = $kind;
        let y = world.y(&live.points);
        let n = live.points.len();
        let $blocks = |rows: &[usize], cols: Option<&[usize]>| blocks_of(kind, world, rows, cols);
        let $extra = |rows: &[usize]| -> Vec<f64> {
            match kind {
                Kind::WithPoints => world.x(&[1], rows),
                _ => Vec::new(),
            }
        };
        // The coordinate model of a summed slot is the isotropic RBF.
        let $coords = (
            match kind {
                Kind::Scalar => KernelSpec::from(RbfKernel::new(ELL[0]).expect("ell")),
                _ => KernelSpec::from(RbfArdKernel::new(&ELL).expect("ell")),
            },
            vec![0usize, 1],
        );
        let train = $blocks(&live.points, Some(&live.inducing));
        let rbf = |ell: f64| RbfKernel::new(ell).expect("ell");
        macro_rules! run {
            ($kernel:expr, $first:expr, $wrapped:expr) => {{
                let $wrap = $wrapped;
                let $src = |rows: &[usize], cols: Option<&[usize]>| $wrap($blocks(rows, cols));
                #[allow(unused_mut)]
                let mut $online = Sgpr::new($kernel, lik())
                    .with_precision::<$p>()
                    .with_optimizer(Fixed)
                    .factor($first, n, &y, &live.inducing)
                    .map_err(|(_, e)| e)
                    .expect("factor")
                    .into_online();
                #[allow(unused_macros)]
                macro_rules! insert {
                    ($s:expr, $x:expr, $y:expr) => {{
                        let _: Vec<f64> = $x;
                        $online.insert($s, $y)
                    }};
                }
                macro_rules! predict {
                    ($s:expr, $xq:expr) => {
                        $online.predict($s, Q)
                    };
                }
                $body
            }};
        }
        match kind {
            Kind::Scalar => {
                let image = ScalarDistance::new();
                run!(
                    image.kernel(rbf(ELL[0])),
                    [image.from_vec(train[0].clone())],
                    |b: Vec<Vec<f64>>| -> Vec<DistanceSource<'static>> {
                        b.into_iter().map(|v| image.from_vec(v)).collect()
                    }
                )
            }
            Kind::Ard => {
                let (bands, kernel) = ArdDistance::from_leaf(RbfArdKernel::new(&ELL).expect("ell"));
                run!(kernel, [bands.from_vecs(train.clone())], |b: Vec<
                    Vec<f64>,
                >|
                 -> Vec<
                    DistanceSource<'static>,
                > {
                    vec![bands.from_vecs(b)]
                })
            }
            Kind::ArdBorrowed => {
                let (bands, kernel) = ArdDistance::from_leaf(RbfArdKernel::new(&ELL).expect("ell"));
                let refs: Vec<&[f64]> = train.iter().map(Vec::as_slice).collect();
                run!(kernel, [bands.borrow(&refs)], |b: Vec<Vec<f64>>| -> Vec<
                    DistanceSource<'static>,
                > {
                    vec![bands.from_vecs(b)]
                })
            }
            Kind::SlotPair => {
                let image = ScalarDistance::new();
                let (bands, ard) =
                    ArdDistance::from_leaf(RbfArdKernel::new(&ELL[1..]).expect("ell"));
                run!(
                    image.kernel(rbf(ELL[0])) * ard,
                    [
                        bands.from_vecs(vec![train[1].clone()]),
                        image.from_vec(train[0].clone()),
                    ],
                    |b: Vec<Vec<f64>>| -> Vec<DistanceSource<'static>> {
                        let mut b = b.into_iter();
                        let (first, second) =
                            (b.next().unwrap_or_default(), b.next().unwrap_or_default());
                        vec![bands.from_vecs(vec![second]), image.from_vec(first)]
                    }
                )
            }
            Kind::TwoScalars => {
                let (a, b) = (ScalarDistance::new(), ScalarDistance::new());
                run!(
                    a.kernel(rbf(ELL[0])) * b.kernel(rbf(ELL[1])),
                    [b.from_vec(train[1].clone()), a.from_vec(train[0].clone())],
                    |v: Vec<Vec<f64>>| -> Vec<DistanceSource<'static>> {
                        let mut v = v.into_iter();
                        let (first, second) =
                            (v.next().unwrap_or_default(), v.next().unwrap_or_default());
                        vec![a.from_vec(first), b.from_vec(second)]
                    }
                )
            }
            Kind::WithPoints => {
                let image = ScalarDistance::new();
                let $wrap = |b: Vec<Vec<f64>>| -> Vec<DistanceSource<'static>> {
                    b.into_iter().map(|v| image.from_vec(v)).collect()
                };
                let $src = |rows: &[usize], cols: Option<&[usize]>| $wrap($blocks(rows, cols));
                let kernel = image.kernel(rbf(ELL[0])) * KernelSpec::from(rbf(ELL[1]));
                #[allow(unused_mut)]
                let mut $online = Sgpr::new(kernel, lik())
                    .with_precision::<$p>()
                    .with_optimizer(Fixed)
                    .factor(
                        [image.from_vec(train[0].clone())],
                        n,
                        &world.x(&[1], &live.points),
                        1,
                        &y,
                        &live.inducing,
                    )
                    .map_err(|(_, e)| e)
                    .expect("factor")
                    .into_online();
                #[allow(unused_macros)]
                macro_rules! insert {
                    ($s:expr, $x:expr, $y:expr) => {
                        $online.insert($s, &$x, $y)
                    };
                }
                macro_rules! predict {
                    ($s:expr, $xq:expr) => {
                        $online.predict($s, &$xq, Q, 1)
                    };
                }
                $body
            }
        }
    }};
}

/// The online model holds what `live` says: the inducing points at their
/// buffer places, and the same predictions, bound, gradient, and Hessian
/// as the coordinate model factored from scratch. The gradient and the
/// Hessian are taken at a shifted `θ` on a copy, so they assemble the
/// system again from the stored blocks; the copy then predicts at that
/// `θ` from the blocks it lent and got back.
macro_rules! check {
    ($p:ty, $online:ident, $live:expr, $world:expr, $src:ident, $coords:ident, $tol:expr) => {{
        let (live, world): (&Live, &World) = (&$live, $world);
        let tol: f64 = $tol;
        let ids = $online.point_ids();
        let want: Vec<_> = live.inducing.iter().map(|&i| ids[live.at(i)]).collect();
        assert_eq!($online.inducing_points().collect::<Vec<_>>(), want);
        assert_eq!($online.n(), live.points.len());
        assert_eq!($online.m(), live.inducing.len());
        let (kernel, dims) = &$coords;
        let mut reference = Sgpr::new(kernel.clone(), lik())
            .with_precision::<$p>()
            .with_optimizer(Fixed)
            .factor(
                &world.x(dims, &live.points),
                live.points.len(),
                dims.len(),
                &world.y(&live.points),
                &world.x(dims, &live.inducing),
                live.inducing.len(),
            )
            .map_err(|(_, e)| e)
            .expect("reference");
        let expect = reference
            .predict(&world.xq(dims), Q, dims.len())
            .expect("predict");
        let got = predict!($src(&live.inducing, None), world.xq(&[1])).expect("predict");
        assert_slice_close(&to64(&got.mean), &to64(&expect.mean), tol);
        assert_slice_close(&to64(&got.variance), &to64(&expect.variance), tol);
        assert_close(
            $online.neg_log_marginal_likelihood().expect("nlml"),
            reference.neg_log_marginal_likelihood().expect("nlml"),
            tol * 10.0,
        );
        let p = reference.num_params();
        assert_eq!($online.num_params(), p);
        let mut theta = vec![0.0; p];
        reference.get_params(&mut theta).expect("theta");
        let at: Vec<f64> = theta.iter().map(|t| t + 0.1).collect();
        let saved = $online.clone();
        let (mut gr, mut go) = (vec![0.0; p], vec![0.0; p]);
        let vr = reference
            .value_and_gradient_into(&at, &mut gr)
            .expect("grad");
        let vo = $online.value_and_gradient_into(&at, &mut go).expect("grad");
        assert_close(vo, vr, tol * 10.0);
        assert_slice_close(&go, &gr, tol * 10.0);
        let (mut hr, mut ho) = (vec![0.0; p * p], vec![0.0; p * p]);
        reference.hessian_into(&at, &mut hr).expect("hess");
        $online.hessian_into(&at, &mut ho).expect("hess");
        assert_slice_close(&ho, &hr, tol * 10.0);
        let expect = reference
            .predict(&world.xq(dims), Q, dims.len())
            .expect("predict");
        let got = predict!($src(&live.inducing, None), world.xq(&[1])).expect("predict");
        assert_slice_close(&to64(&got.mean), &to64(&expect.mean), tol);
        $online = saved;
    }};
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn online_matches<P: GpScalar + std::fmt::Debug>(kind: Kind, tol: f64) {
    let world = World::new();
    with_online!(
        P,
        kind,
        &world,
        Live::start(),
        |online, src, blocks, wrap, extra, coords| {
            let _ = (&blocks, &wrap);
            let mut live = Live::start();
            check!(P, online, live, &world, src, coords, tol);
            // Two inserts fill the room the first one makes (a quarter more).
            for new in [10, 11] {
                insert!(
                    src(&live.inducing, Some(&[new])),
                    extra(&[new]),
                    world.y[new]
                )
                .expect("insert");
                live.points.push(new);
                check!(P, online, live, &world, src, coords, tol);
            }
            // Sample 3 becomes an inducing point.
            let id = online.point_ids()[live.at(3)];
            online
                .insert_inducing(id, src(&live.points, Some(&[3])))
                .expect("insert_inducing");
            live.inducing.push(3);
            check!(P, online, live, &world, src, coords, tol);
            // A point in the middle that is not inducing.
            let id = online.point_ids()[live.at(5)];
            online.delete(id).expect("delete");
            live.points.retain(|&p| p != 5);
            check!(P, online, live, &world, src, coords, tol);
            // An inducing point's sample cannot go; once it is not inducing, it can.
            let id = online.point_ids()[live.at(0)];
            let before = predict!(src(&live.inducing, None), world.xq(&[1])).expect("predict");
            assert!(matches!(
                online.delete(id),
                Err(GprError::InvalidConfig { .. })
            ));
            let after = predict!(src(&live.inducing, None), world.xq(&[1])).expect("predict");
            assert_eq!(bits(&after.mean), bits(&before.mean));
            let at = live
                .inducing
                .iter()
                .position(|&i| i == 0)
                .expect("inducing");
            online
                .delete_inducing(online.inducing_ids()[at])
                .expect("delete_inducing");
            live.inducing.remove(at);
            check!(P, online, live, &world, src, coords, tol);
            online.delete(id).expect("delete");
            live.points.retain(|&p| p != 0);
            check!(P, online, live, &world, src, coords, tol);
            // A new point, then the same point as an inducing point; then more
            // points than the room holds, so the blocks are laid out again.
            insert!(src(&live.inducing, Some(&[12])), extra(&[12]), world.y[12]).expect("insert");
            live.points.push(12);
            let id = online.point_ids()[live.at(12)];
            online
                .insert_inducing(id, src(&live.points, Some(&[12])))
                .expect("insert_inducing");
            live.inducing.push(12);
            check!(P, online, live, &world, src, coords, tol);
            for new in 13..TOTAL {
                insert!(
                    src(&live.inducing, Some(&[new])),
                    extra(&[new]),
                    world.y[new]
                )
                .expect("insert");
                live.points.push(new);
                check!(P, online, live, &world, src, coords, tol);
            }
            // The first point, and the last.
            for gone in [live.points[0], *live.points.last().expect("points")] {
                let id = online.point_ids()[live.at(gone)];
                online.delete(id).expect("delete");
                live.points.retain(|&p| p != gone);
                check!(P, online, live, &world, src, coords, tol);
            }
        }
    );
}

fn bits<T: KernelScalar>(values: &[T]) -> Vec<u64> {
    values.iter().map(|v| v.to_f64().to_bits()).collect()
}

#[test]
fn online_sgpr_on_supplied_distances_matches_a_fresh_factor() {
    for kind in KINDS {
        online_matches::<DoublePrecision>(kind, 1e-9);
    }
}

#[test]
fn online_sgpr_on_supplied_distances_matches_in_single_and_mixed() {
    for kind in KINDS {
        online_matches::<SinglePrecision>(kind, 5e-4);
        online_matches::<MixedPrecision<PromoteStorage>>(kind, 5e-4);
    }
}

/// Columns that are not finite, negative, or of the wrong length are
/// refused, for every kernel and precision, and leave the model's
/// predictions, points, and inducing points as they were.
#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn refused<P: GpScalar + std::fmt::Debug>(kind: Kind) {
    let world = World::new();
    with_online!(
        P,
        kind,
        &world,
        Live::start(),
        |online, src, blocks, wrap, extra, coords| {
            let _ = &coords;
            let live = Live::start();
            let before = predict!(src(&live.inducing, None), world.xq(&[1])).expect("predict");
            let unchanged = |n: usize, m: usize, got: &gprx::Prediction<P::Refine>| {
                assert_eq!(n, live.points.len());
                assert_eq!(m, live.inducing.len());
                assert_eq!(bits(&got.mean), bits(&before.mean));
                assert_eq!(bits(&got.variance), bits(&before.variance));
            };
            let row = blocks(&live.inducing, Some(&[10]));
            let column = blocks(&live.points, Some(&[3]));
            for bad in [f64::NAN, f64::INFINITY, -1.0] {
                for b in 0..row.len() {
                    let mut rows = row.clone();
                    rows[b][1] = bad;
                    assert!(
                        matches!(
                            insert!(wrap(rows), extra(&[10]), world.y[10]),
                            Err(GprError::InvalidDistance { .. })
                        ),
                        "{kind:?} insert of {bad}"
                    );
                    let got = predict!(src(&live.inducing, None), world.xq(&[1])).expect("predict");
                    unchanged(online.n(), online.m(), &got);
                    let mut cols = column.clone();
                    cols[b][live.at(9)] = bad;
                    let id = online.point_ids()[live.at(3)];
                    assert!(
                        matches!(
                            online.insert_inducing(id, wrap(cols)),
                            Err(GprError::InvalidDistance { .. })
                        ),
                        "{kind:?} insert_inducing of {bad}"
                    );
                    let got = predict!(src(&live.inducing, None), world.xq(&[1])).expect("predict");
                    unchanged(online.n(), online.m(), &got);
                }
            }
            let mut short = row.clone();
            short[0].pop();
            assert!(matches!(
                insert!(wrap(short), extra(&[10]), world.y[10]),
                Err(GprError::LengthMismatch { .. })
            ));
            let mut short = column.clone();
            short[0].pop();
            let id = online.point_ids()[live.at(3)];
            assert!(matches!(
                online.insert_inducing(id, wrap(short)),
                Err(GprError::LengthMismatch { .. })
            ));
            let got = predict!(src(&live.inducing, None), world.xq(&[1])).expect("predict");
            unchanged(online.n(), online.m(), &got);
            // The model still takes good columns.
            insert!(wrap(row), extra(&[10]), world.y[10]).expect("insert");
        }
    );
}

#[test]
fn bad_columns_are_refused_for_every_kernel_and_precision() {
    for kind in KINDS {
        refused::<DoublePrecision>(kind);
        refused::<SinglePrecision>(kind);
        refused::<MixedPrecision<PromoteStorage>>(kind);
    }
}

/// One inducing point, which cannot be deleted; then every training point
/// inducing (`m = n`), each step matching the coordinate model.
#[test]
fn one_inducing_point_and_every_point_inducing() {
    let world = World::new();
    let start = Live {
        points: (0..6).collect(),
        inducing: vec![2],
    };
    with_online!(
        DoublePrecision,
        Kind::Ard,
        &world,
        start,
        |online, src, blocks, wrap, extra, coords| {
            let _ = (&blocks, &wrap, &extra);
            let mut live = Live {
                points: (0..6).collect(),
                inducing: vec![2],
            };
            check!(DoublePrecision, online, live, &world, src, coords, 1e-9);
            assert!(matches!(
                online.delete_inducing(online.inducing_ids()[0]),
                Err(GprError::InsufficientData { .. })
            ));
            for next in [5, 0, 3, 1, 4] {
                let id = online.point_ids()[live.at(next)];
                online
                    .insert_inducing(id, src(&live.points, Some(&[next])))
                    .expect("insert_inducing");
                live.inducing.push(next);
                check!(DoublePrecision, online, live, &world, src, coords, 1e-8);
            }
            assert_eq!(online.m(), online.n());
            // Every point is inducing: none can be deleted until it is not.
            let id = online.point_ids()[0];
            assert!(matches!(
                online.delete(id),
                Err(GprError::InvalidConfig { .. })
            ));
        }
    );
}

/// Fills one `m × 1` column (the inducing samples to sample `point`).
struct Column<'a> {
    world: &'a World,
    rows: Vec<usize>,
    point: usize,
}

impl DistanceFill for Column<'_> {
    fn fill_column(&self, col: usize, rows: std::ops::Range<usize>, out: &mut [f64]) {
        assert_eq!(col, 0);
        let block = self.world.summed(&[0, 1], &self.rows, Some(&[self.point]));
        out.copy_from_slice(&block[rows]);
    }
}

/// Every source kind inserts the same point; refused columns leave the
/// model as it was, and a tidy source repairs a rounded inducing column.
#[test]
fn inserts_check_their_columns_and_change_nothing_on_an_error() {
    let world = World::new();
    let image = ScalarDistance::new();
    let rbf = RbfKernel::new(ELL[0]).expect("ell");
    let live = Live::start();
    let n = live.points.len();
    let block = |rows: &[usize], cols: &[usize]| world.summed(&[0, 1], rows, Some(cols));
    let fresh = || {
        Sgpr::new(image.kernel(rbf), lik())
            .with_optimizer(Fixed)
            .factor(
                [image.from_vec(block(&live.points, &live.inducing))],
                n,
                &world.y(&live.points),
                &live.inducing,
            )
            .map_err(|(_, e)| e)
            .expect("factor")
            .into_online()
    };
    let query = world.summed(&[0, 1], &live.inducing, None);
    let row = block(&live.inducing, &[10]);
    let mut reference = fresh();
    reference
        .insert([image.borrow(&row)], world.y[10])
        .expect("insert");
    let expect = reference
        .predict([image.borrow(&query)], Q)
        .expect("predict");
    let fill = Column {
        world: &world,
        rows: live.inducing.clone(),
        point: 10,
    };
    for source in [
        image.from_slice(&row),
        image.from_vec(row.clone()),
        image.fill(&fill),
    ] {
        let mut online = fresh();
        online.insert([source], world.y[10]).expect("insert");
        let got = online.predict([image.borrow(&query)], Q).expect("predict");
        assert_eq!(bits(&got.mean), bits(&expect.mean));
    }
    // Refused inserts.
    let mut online = fresh();
    let before = online.predict([image.borrow(&query)], Q).expect("predict");
    let unchanged = |online: &OnlineSgpr<Fixed, DoublePrecision, DistanceKernel<DistanceOnly>>| {
        let got = online.predict([image.borrow(&query)], Q).expect("predict");
        assert_eq!(bits(&got.mean), bits(&before.mean));
        assert_eq!(online.n(), n);
        assert_eq!(online.m(), live.inducing.len());
    };
    let mut bad = row.clone();
    bad[1] = -1.0;
    assert!(matches!(
        online.insert([image.from_vec(bad)], 0.1),
        Err(GprError::InvalidDistance { .. })
    ));
    unchanged(&online);
    assert!(matches!(
        online.insert([image.from_vec(row[1..].to_vec())], 0.1),
        Err(GprError::LengthMismatch { .. })
    ));
    unchanged(&online);
    assert!(matches!(
        online.insert([image.borrow(&row)], f64::NAN),
        Err(GprError::NonFiniteInput)
    ));
    unchanged(&online);
    assert!(matches!(
        online.insert(Vec::<DistanceSource<'_>>::new(), 0.1),
        Err(GprError::DistanceSlot {
            kind: SlotErrorKind::Missing,
            ..
        })
    ));
    unchanged(&online);
    // Refused inducing points: one already inducing, a column whose own
    // value is not zero, one that disagrees with a stored pair, one of
    // the wrong length, and an unknown point.
    let id_of = |online: &OnlineSgpr<Fixed, DoublePrecision, DistanceKernel<DistanceOnly>>,
                 i: usize| { online.point_ids()[live.at(i)] };
    let column = block(&live.points, &[3]);
    let id = id_of(&online, 7);
    assert!(matches!(
        online.insert_inducing(id, [image.from_vec(block(&live.points, &[7]))]),
        Err(GprError::InvalidConfig { .. })
    ));
    unchanged(&online);
    let id = id_of(&online, 3);
    let mut diagonal = column.clone();
    diagonal[live.at(3)] = 1e-9;
    assert!(matches!(
        online.insert_inducing(id, [image.from_vec(diagonal)]),
        Err(GprError::InvalidDistance { .. })
    ));
    unchanged(&online);
    let mut pair = column.clone();
    pair[live.at(7)] += 1e-9;
    match online.insert_inducing(id, [image.from_vec(pair.clone())]) {
        Err(GprError::InvalidDistance {
            pair: Some((row, _)),
            ..
        }) => assert_eq!(row, live.at(7)),
        other => panic!("expected InvalidDistance, got {other:?}"),
    }
    unchanged(&online);
    assert!(matches!(
        online.insert_inducing(id, [image.from_vec(column[1..].to_vec())]),
        Err(GprError::LengthMismatch { .. })
    ));
    unchanged(&online);
    let gone = online.insert([image.borrow(&row)], 0.1).expect("insert");
    online.delete(gone).expect("delete");
    assert!(matches!(
        online.insert_inducing(gone, [image.borrow(&column)]),
        Err(GprError::InvalidPointId)
    ));
    // A tidy source repairs the pair to its mean, which the stored block
    // holds after; the repaired model is the model of the repaired table.
    let mut repaired = fresh();
    repaired
        .insert_inducing(id, [image.from_vec(pair).tidy(1e-6).expect("tidy")])
        .expect("tidy");
    let mut exact = fresh();
    exact
        .insert_inducing(id, [image.borrow(&column)])
        .expect("exact");
    let query = world.summed(&[0, 1], &[7, 0, 4, 9, 3], None);
    let got = repaired
        .predict([image.borrow(&query)], Q)
        .expect("predict");
    let want = exact.predict([image.borrow(&query)], Q).expect("predict");
    assert_slice_close(&got.mean, &want.mean, 1e-8);
    assert_slice_close(&got.variance, &want.variance, 1e-8);
}

/// An inducing point that leaves `K_mm` singular (a copy of one already
/// inducing) does not factor without jitter: the model keeps its inducing
/// points, its blocks, and its predictions, at every precision.
#[test]
fn an_inducing_point_that_does_not_factor_changes_nothing() {
    fn run<P: GpScalar>() {
        let world = World::new();
        let image = ScalarDistance::new();
        let live = Live::start();
        let n = live.points.len();
        let block = |rows: &[usize], cols: &[usize]| world.summed(&[0, 1], rows, Some(cols));
        let mut online = Sgpr::new(image.kernel(RbfKernel::new(ELL[0]).expect("ell")), lik())
            .with_precision::<P>()
            .with_optimizer(Fixed)
            .with_jitter_policy(gprx::JitterPolicy::fixed(0.0).expect("jitter"))
            .factor(
                [image.from_vec(block(&live.points, &live.inducing))],
                n,
                &world.y(&live.points),
                &live.inducing,
            )
            .map_err(|(_, e)| e)
            .expect("factor")
            .into_online();
        // A copy of sample 7 (inducing): its squared distances to the
        // inducing points are those of sample 7.
        online
            .insert([image.from_vec(block(&live.inducing, &[7]))], world.y[7])
            .expect("insert");
        let mut points = live.points.clone();
        points.push(7);
        let query = world.summed(&[0, 1], &live.inducing, None);
        let before = online.predict([image.borrow(&query)], Q).expect("predict");
        let copy = online.point_ids()[n];
        assert!(matches!(
            online.insert_inducing(copy, [image.from_vec(block(&points, &[7]))]),
            Err(GprError::CholeskyFailed { .. })
        ));
        assert_eq!(online.m(), live.inducing.len());
        assert_eq!(common::inducing_places(&online), [7, 0, 4, 9]);
        let after = online.predict([image.borrow(&query)], Q).expect("predict");
        assert_eq!(bits(&after.mean), bits(&before.mean));
        assert_eq!(bits(&after.variance), bits(&before.variance));
        // The blocks are as they were: a point that factors goes in.
        let id = online.point_ids()[live.at(3)];
        online
            .insert_inducing(id, [image.from_vec(block(&points, &[3]))])
            .expect("insert_inducing");
        assert_eq!(common::inducing_places(&online), [7, 0, 4, 9, 3]);
    }
    run::<DoublePrecision>();
    run::<SinglePrecision>();
    run::<MixedPrecision<PromoteStorage>>();
}
