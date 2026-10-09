//! `OnlineSgpr` on supplied squared distances against the coordinate
//! `Sgpr` factored from scratch on the same points and inducing points,
//! after every insert and delete. Public API only.

mod common;

use common::{assert_close, assert_slice_close};
use gprx::kernel::{
    ArdDistance, DistanceFill, DistanceKernel, DistanceOnly, DistanceSource, KernelScalar,
    KernelSpec, RbfArdKernel, RbfKernel, ScalarDistance,
};
use gprx::{
    DoublePrecision, Fixed, GaussianLikelihood, GpScalar, GprError, MixedPrecision, OnlineSgpr,
    PromoteStorage, Sgpr, SinglePrecision,
};

/// Points the test can insert, and queries.
const TOTAL: usize = 16;
const Q: usize = 5;
const D: usize = 2;
const ELL: [f64; 2] = [0.8, 1.4];

/// Coordinate `k` of `rows` samples.
fn coord(k: usize, rows: usize, offset: f64) -> Vec<f64> {
    (0..rows)
        .map(|i| ((i as f64 + offset) * (0.41 + 0.17 * k as f64)).sin() * (1.0 + 0.5 * k as f64))
        .collect()
}

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

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn lik() -> GaussianLikelihood {
    GaussianLikelihood::new(0.05).expect("noise")
}

fn to64<T: KernelScalar>(values: &[T]) -> Vec<f64> {
    values.iter().map(|v| v.to_f64()).collect()
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

/// One kernel of the comparison: the distance kernel's slot reads `dist`
/// (summed for a scalar slot), and coordinates of `coords`.
#[derive(Clone, Copy)]
enum Kind {
    Scalar,
    Ard,
    WithPoints,
}

/// Runs `$body` with `$online` the online model of `$kind` on the starting
/// points, `$src(rows, cols)` a source of the blocks between samples
/// (`cols: None` for the queries), and `$coords` the coordinate model's
/// kernel and dimensions; `$extra(rows)` is the coordinates a `WithPoints`
/// model takes (empty otherwise).
macro_rules! with_online {
    ($p:ty, $kind:expr, $world:expr, |$online:ident, $src:ident, $extra:ident, $coords:ident| $body:block) => {{
        let world: &World = $world;
        let live = Live::start();
        let y = world.y(&live.points);
        let n = live.points.len();
        match $kind {
            Kind::Scalar => {
                let image = ScalarDistance::new();
                let rbf = RbfKernel::new(ELL[0]).expect("ell");
                let $src = |rows: &[usize], cols: Option<&[usize]>| -> DistanceSource<'static> {
                    image.from_vec(world.summed(&[0, 1], rows, cols))
                };
                let $extra = |_rows: &[usize]| -> Vec<f64> { Vec::new() };
                let $coords = (KernelSpec::from(rbf), vec![0usize, 1]);
                #[allow(unused_mut)]
                let mut $online = Sgpr::new(image.kernel(rbf), lik())
                    .with_precision::<$p>()
                    .with_optimizer(Fixed)
                    .factor(
                        [$src(&live.points, Some(&live.inducing))],
                        n,
                        &y,
                        &live.inducing,
                    )
                    .map_err(|(_, e)| e)
                    .expect("factor")
                    .into_online();
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
            }
            Kind::Ard => {
                let ard = RbfArdKernel::new(&ELL).expect("ell");
                let (bands, kernel) = ArdDistance::from_leaf(ard.clone());
                let $src = |rows: &[usize], cols: Option<&[usize]>| -> DistanceSource<'static> {
                    bands.from_vecs((0..D).map(|k| world.sq(k, rows, cols)).collect())
                };
                let $extra = |_rows: &[usize]| -> Vec<f64> { Vec::new() };
                let $coords = (KernelSpec::from(ard), vec![0usize, 1]);
                #[allow(unused_mut)]
                let mut $online = Sgpr::new(kernel, lik())
                    .with_precision::<$p>()
                    .with_optimizer(Fixed)
                    .factor(
                        [$src(&live.points, Some(&live.inducing))],
                        n,
                        &y,
                        &live.inducing,
                    )
                    .map_err(|(_, e)| e)
                    .expect("factor")
                    .into_online();
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
            }
            Kind::WithPoints => {
                let image = ScalarDistance::new();
                let $src = |rows: &[usize], cols: Option<&[usize]>| -> DistanceSource<'static> {
                    image.from_vec(world.sq(0, rows, cols))
                };
                let $extra = |rows: &[usize]| -> Vec<f64> { world.x(&[1], rows) };
                let $coords = (
                    KernelSpec::from(RbfArdKernel::new(&ELL).expect("ell")),
                    vec![0usize, 1],
                );
                let kernel = image.kernel(RbfKernel::new(ELL[0]).expect("ell"))
                    * KernelSpec::from(RbfKernel::new(ELL[1]).expect("ell"));
                #[allow(unused_mut)]
                let mut $online = Sgpr::new(kernel, lik())
                    .with_precision::<$p>()
                    .with_optimizer(Fixed)
                    .factor(
                        [$src(&live.points, Some(&live.inducing))],
                        n,
                        &$extra(&live.points),
                        1,
                        &y,
                        &live.inducing,
                    )
                    .map_err(|(_, e)| e)
                    .expect("factor")
                    .into_online();
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
/// buffer places, the same predictions and bound as the coordinate model
/// factored from scratch.
macro_rules! check {
    ($p:ty, $online:ident, $live:expr, $world:expr, $src:ident, $coords:ident, $tol:expr) => {{
        let (live, world): (&Live, &World) = (&$live, $world);
        let places: Vec<usize> = live.inducing.iter().map(|&i| live.at(i)).collect();
        assert_eq!($online.inducing(), places.as_slice());
        assert_eq!($online.n(), live.points.len());
        assert_eq!($online.m(), live.inducing.len());
        let (kernel, dims) = &$coords;
        let reference = Sgpr::new(kernel.clone(), lik())
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
        let got = predict!([$src(&live.inducing, None)], world.xq(&[1])).expect("predict");
        assert_slice_close(&to64(&got.mean), &to64(&expect.mean), $tol);
        assert_slice_close(&to64(&got.variance), &to64(&expect.variance), $tol);
        assert_close(
            $online.neg_log_marginal_likelihood().expect("nlml"),
            reference.neg_log_marginal_likelihood().expect("nlml"),
            $tol * 10.0,
        );
    }};
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn online_matches<P: GpScalar + std::fmt::Debug>(kind: Kind, tol: f64) {
    let world = World::new();
    with_online!(P, kind, &world, |online, src, extra, coords| {
        let mut live = Live::start();
        check!(P, online, live, &world, src, coords, tol);
        // Two inserts fill the room the first one makes (a quarter more).
        for new in [10, 11] {
            insert!(
                [src(&live.inducing, Some(&[new]))],
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
            .insert_inducing(id, [src(&live.points, Some(&[3]))])
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
        let before = predict!([src(&live.inducing, None)], world.xq(&[1])).expect("predict");
        assert!(matches!(
            online.delete(id),
            Err(GprError::InvalidConfig { .. })
        ));
        let after = predict!([src(&live.inducing, None)], world.xq(&[1])).expect("predict");
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
        insert!(
            [src(&live.inducing, Some(&[12]))],
            extra(&[12]),
            world.y[12]
        )
        .expect("insert");
        live.points.push(12);
        let id = online.point_ids()[live.at(12)];
        online
            .insert_inducing(id, [src(&live.points, Some(&[12]))])
            .expect("insert_inducing");
        live.inducing.push(12);
        check!(P, online, live, &world, src, coords, tol);
        for new in 13..TOTAL {
            insert!(
                [src(&live.inducing, Some(&[new]))],
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
    });
}

fn bits<T: KernelScalar>(values: &[T]) -> Vec<u64> {
    values.iter().map(|v| v.to_f64().to_bits()).collect()
}

#[test]
fn online_sgpr_on_supplied_distances_matches_a_fresh_factor() {
    for kind in [Kind::Scalar, Kind::Ard, Kind::WithPoints] {
        online_matches::<DoublePrecision>(kind, 1e-9);
    }
}

#[test]
fn online_sgpr_on_supplied_distances_matches_in_single_and_mixed() {
    for kind in [Kind::Scalar, Kind::Ard, Kind::WithPoints] {
        online_matches::<SinglePrecision>(kind, 2e-3);
        online_matches::<MixedPrecision<PromoteStorage>>(kind, 2e-3);
    }
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
        Err(GprError::LengthMismatch { .. })
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
        Err(GprError::InvalidDistance { row, .. }) => assert_eq!(row, live.at(7)),
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
