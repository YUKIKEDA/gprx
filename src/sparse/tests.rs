use super::*;
use crate::kernel::{
    ArdDistance, BlockAt, ModelKernelParts, RbfArdKernel, RbfKernel, ScalarDistance,
};

/// Everything a change and its undo must give back: the inducing
/// points, and both copies of the blocks and squares.
type Values = (usize, usize, Vec<f64>);

fn state(supply: &SparseSupply) -> (Vec<usize>, Values, Values, Values, Values) {
    let f32 = supply.at::<f32>().expect("f32");
    (
        supply.inducing.clone(),
        supply.exact().xz.values(),
        supply.exact().zz.values(),
        f32.xz.values(),
        f32.zz.values(),
    )
}

/// A supply of a scalar slot and a two-dimensional ARD slot over `n`
/// points, `inducing` among them, with its `f32` copy made, and the
/// points' coordinates (the scalar slot's, then the ARD slot's two).
fn supply(n: usize, inducing: &[usize]) -> (SparseSupply, [Vec<f64>; 3]) {
    let image = ScalarDistance::new();
    let (bands, ard) = ArdDistance::from_leaf(RbfArdKernel::new(&[1.0, 2.0]).expect("ell"));
    let kernel = image.kernel(RbfKernel::new(1.0).expect("ell")) * ard;
    let slots = crate::kernel::spec_slots(&kernel.into_spec());
    let coords = [0.3, 0.7, 1.1].map(|step| {
        (0..n)
            .map(|i| (i as f64 * step).sin() * 2.0)
            .collect::<Vec<f64>>()
    });
    let block = |x: &[f64]| -> Vec<f64> {
        inducing
            .iter()
            .flat_map(|&j| x.iter().map(move |xi| (xi - x[j]).powi(2)))
            .collect()
    };
    let (zz, xz) = crate::kernel::bind_inducing(
        &slots,
        [
            bands.from_vecs(vec![block(&coords[1]), block(&coords[2])]),
            image.from_vec(block(&coords[0])),
        ],
        n,
        inducing,
    )
    .expect("bind");
    let supply = SparseSupply::new::<f32>(&[], inducing.to_vec(), zz, xz).expect("supply");
    supply.at::<f32>().expect("f32");
    (supply, coords)
}

/// Each change of the training blocks and its undo give back the same
/// inducing points, blocks, squares, and `f32` copy.
#[test]
fn supply_changes_undo_to_the_same_state() {
    let (n, inducing) = (7, [5, 1, 3]);
    let (mut s, coords) = supply(n, &inducing);
    let original = state(&s);
    // A point, then its undo.
    for _ in 0..2 {
        s.reserve_point().expect("room");
        s.push_point(|b, col| 10.0 + (b * 3 + col) as f64);
        assert_eq!(s.exact().xz.values().0, n + 1);
        s.pop_point();
        assert_eq!(state(&s), original);
    }
    // A point that is not inducing, removed and put back: the inducing
    // indices past it move down and back up.
    for index in [0, 2, 6] {
        let mut saved = Vec::new();
        s.remove_point(index, Some(&mut saved));
        let shifted: Vec<usize> = inducing
            .iter()
            .map(|&i| if i > index { i - 1 } else { i })
            .collect();
        assert_eq!(s.inducing, shifted);
        s.restore_point(index, &saved);
        assert_eq!(state(&s), original);
    }
    // Point 4 as one more inducing point, with mirror values that differ
    // from the stored ones (a tidy repair), then its undo.
    let column = |b: usize, row: usize| -> f64 {
        // Blocks in slot order: the scalar slot's, then the ARD slot's two.
        (coords[b][row] - coords[b][4]).powi(2)
    };
    for _ in 0..2 {
        s.reserve_inducing().expect("room");
        let mut saved = Vec::new();
        s.add_inducing(
            4,
            column,
            |b, col| column(b, inducing[col]) + 0.5,
            &mut saved,
        );
        assert_eq!(s.inducing, [5, 1, 3, 4]);
        assert_eq!(s.exact().zz.values().0, 4);
        assert_eq!(
            s.exact().xz.get(BlockAt::Scalar(0), 4, 0).to_bits(),
            (column(0, 5) + 0.5).to_bits()
        );
        let f32 = s.at::<f32>().expect("f32");
        assert_eq!(
            f32.xz.get(BlockAt::Ard(0, 1), 3, 3).to_bits(),
            (column(2, 3) as f32).to_bits()
        );
        s.undo_add_inducing(4, &saved);
        assert_eq!(state(&s), original);
    }
    // Each inducing point removed and put back.
    for (at, &point) in inducing.iter().enumerate() {
        let mut saved = Vec::new();
        s.remove_inducing(at, &mut saved);
        assert_eq!(s.exact().zz.values().0, 2);
        s.undo_remove_inducing(at, point, &saved);
        assert_eq!(state(&s), original);
    }
}
