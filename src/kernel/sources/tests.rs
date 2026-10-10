use super::*;
use crate::kernel::{ArdDistance, ScalarDistance};

/// A store of one scalar slot and two ARD slots (packed, and moved-in
/// tables), `rows × cols`, every value distinct.
fn store(rows: usize, cols: usize) -> BlockStore<f64> {
    let block =
        |seed: f64| -> Vec<f64> { (0..rows * cols).map(|i| seed + i as f64 * 0.25).collect() };
    let packed: Vec<f64> = [block(100.0), block(200.0)].concat();
    BlockStore::packed(
        rows,
        cols,
        vec![block(0.0)],
        vec![
            SlotBlocks::Flat(packed),
            SlotBlocks::Tables(vec![block(300.0), block(400.0), block(500.0)]),
        ],
    )
}

/// What a kernel reads from `store`: every value through
/// [`RectSlots`], in [`BlockStore::values`] order.
fn read(store: &BlockStore<f64>) -> Vec<f64> {
    let (rows, cols, _) = store.values();
    let mut out = Vec::new();
    let scalar = RectSlots::scalar(store, 0).expect("scalar");
    for c in 0..cols {
        out.extend((0..rows).map(|r| scalar[(r, c)]));
    }
    for at in 0..2 {
        let ArdRect::Checked(blocks) = RectSlots::ard(store, at).expect("ard") else {
            panic!("a stored block is checked");
        };
        for k in 0..blocks.d() {
            for c in 0..cols {
                for r in 0..rows {
                    out.push(blocks.read(k, r, c).expect("read"));
                }
            }
        }
    }
    out
}

/// Each change of a store and its undo give back every value, and a
/// kernel reads what the store holds, laid out again or not.
#[test]
fn block_store_changes_undo_to_the_same_values() {
    let mut s = store(5, 3);
    let original = s.values();
    assert_eq!(read(&s), original.2);
    // A row: room (a new layout), then in place.
    for _ in 0..2 {
        s.reserve_row().expect("room");
        s.push_row(|b, c| 1000.0 + (b * 10 + c) as f64);
        assert_eq!(s.values().0, 6);
        assert_eq!(read(&s), s.values().2);
        assert_eq!(
            s.get(BlockAt::Ard(1, 2), 5, 1).to_bits(),
            1051.0f64.to_bits()
        );
        s.pop_row();
        assert_eq!(s.values(), original);
    }
    for index in [0, 2, 4] {
        let mut saved = Vec::new();
        s.row_into(index, &mut saved);
        s.remove_row(index);
        assert_eq!(read(&s), s.values().2);
        s.insert_row(index, &saved);
        assert_eq!(s.values(), original);
    }
    // A column: room, then in place.
    for _ in 0..2 {
        s.reserve_col().expect("room");
        s.push_col(|b, r| 2000.0 + (b * 10 + r) as f64);
        assert_eq!(read(&s), s.values().2);
        assert_eq!(
            s.get(BlockAt::Scalar(0), 4, 3).to_bits(),
            2004.0f64.to_bits()
        );
        s.pop_col();
        assert_eq!(s.values(), original);
    }
    for index in [0, 1, 2] {
        let mut saved = Vec::new();
        s.remove_col(index, Some(&mut saved));
        assert_eq!(read(&s), s.values().2);
        s.insert_col(index, &saved);
        assert_eq!(s.values(), original);
        assert_eq!(read(&s), original.2);
    }
    // The same changes on the store as bound (moved-in tables, no room).
    let mut fresh = store(5, 3);
    let mut saved = Vec::new();
    fresh.remove_col(1, Some(&mut saved));
    fresh.insert_col(1, &saved);
    assert_eq!(fresh.values(), original);
    let mut saved = Vec::new();
    fresh.row_into(3, &mut saved);
    fresh.remove_row(3);
    fresh.insert_row(3, &saved);
    assert_eq!(fresh.values(), original);
    // A cast keeps the layout; the inducing rows are read back to back.
    let cast = s.cast::<f32>().expect("cast");
    assert_eq!(cast.values().2, original.2);
    let mut square = BlockStore::default();
    s.rows_into(&[4, 0], &mut square);
    assert_eq!(
        square.values().2[..2],
        [
            s.get(BlockAt::Scalar(0), 4, 0),
            s.get(BlockAt::Scalar(0), 0, 0)
        ]
    );
}

fn line(scale: f64, rows: std::ops::Range<usize>, cols: std::ops::Range<usize>) -> Vec<f64> {
    cols.flat_map(|j| {
        rows.clone()
            .map(move |i| scale * (i as f64 - j as f64).powi(2))
    })
    .collect()
}

/// A repair keeps huge equal pairs as they are and averages a huge
/// unequal pair without overflowing to infinity.
#[test]
fn a_repair_of_huge_values_stays_finite() {
    let big = 1.0e308;
    let mut block = vec![0.0, big, -1.0e-300, big, 0.0, big, 0.0, 0.9 * big, 0.0];
    repair_block(&mut block, 3, 3, BlockKind::Square);
    assert!(block.iter().all(|v| v.is_finite()));
    assert_eq!(block[1].to_bits(), big.to_bits());
    assert_eq!(block[2].to_bits(), 0.0_f64.to_bits());
    assert_eq!(block[5].to_bits(), block[7].to_bits());
    assert!(block[5] > 0.9 * big && block[5] < big);
}

#[test]
fn the_tiled_walk_visits_every_pair_below_the_diagonal_once() {
    for n in [0, 1, 63, 64, 65, 150] {
        let mut seen = vec![0_u8; n * n];
        let walked = for_each_lower_pair::<()>(n, |i, j| {
            seen[i + j * n] += 1;
            Ok(())
        });
        assert!(walked.is_ok());
        for j in 0..n {
            for i in 0..n {
                assert_eq!(seen[i + j * n], u8::from(i > j), "n={n} ({i}, {j})");
            }
        }
    }
}

#[test]
fn a_mirror_pair_far_from_the_first_tile_is_checked_and_repaired() {
    let n = 150;
    let mut block = line(1.0, 0..n, 0..n);
    assert_eq!(
        check_block(&block, n, n, BlockKind::Square, Tidy::Exact),
        Ok(false)
    );
    // Any gap is refused by the exact check, at the pair in a tile off
    // the diagonal.
    block[140 + 3 * n] += 1e-9;
    let refused = check_block(&block, n, n, BlockKind::Square, Tidy::Exact);
    assert!(
        matches!(
            &refused,
            Err(GprError::InvalidDistance {
                pair: Some((140, 3)),
                ..
            })
        ),
        "{refused:?}"
    );
    // Within a repair's tolerance it is set to the pair's mean.
    let within = Tidy::Within(1e-6);
    assert_eq!(
        check_block(&block, n, n, BlockKind::Square, within),
        Ok(true)
    );
    repair_block(&mut block, n, n, BlockKind::Square);
    assert_eq!(block[140 + 3 * n].to_bits(), block[3 + 140 * n].to_bits());
    // Past it, refused.
    block[140 + 3 * n] += 1.0;
    assert!(matches!(
        check_block(&block, n, n, BlockKind::Square, within),
        Err(GprError::InvalidDistance {
            pair: Some((140, 3)),
            ..
        })
    ));
}

/// The buffers show their capacity, and a clone starts without them.
#[test]
fn query_scratch_holds_no_state_to_clone() {
    let image = ScalarDistance::new();
    let slots = [DistanceSlot::Scalar(image)];
    let mut scratch = QueryScratch::<f32>::new();
    let cross = [0.5, 1.0, 1.5, 2.0];
    let bound = QuerySources::bind_rect(&slots, [image.borrow(&cross)], (2, 2), &mut scratch)
        .expect("bind");
    drop(bound);
    assert!(scratch.cast.capacity() >= 4);
    assert!(format!("{scratch:?}").starts_with("QueryScratch"));
    assert_eq!(scratch.clone().cast.capacity(), 0);
}

/// An `f32` model checks an ARD block as it casts it, so the `f64`
/// view of the caller's block (the refinement's) reads it as checked,
/// not again.
#[test]
fn a_cast_ard_block_is_read_as_checked_in_f64() {
    let bands = ArdDistance::of_dims(2);
    let slots = [DistanceSlot::Ard(bands)];
    let (b0, b1) = ([0.5, 1.0, 1.5, 2.0], [0.25, 0.5, 0.75, 1.0]);
    let tables: [&[f64]; 2] = [&b0, &b1];
    let mut scratch = QueryScratch::<f32>::new();
    let bound = QuerySources::bind(
        &slots,
        [bands.borrow(&tables)],
        2,
        2,
        BlockKind::Rect,
        true,
        &mut scratch,
    )
    .expect("bind");
    let view = bound.f64_view();
    assert!(matches!(view.ard(0), Ok(ArdRect::Checked(_))));
}

#[test]
fn two_scalar_slots_and_an_ard_slot_bind_into_one_store() {
    let slots = vec![
        DistanceSlot::Scalar(ScalarDistance::new()),
        DistanceSlot::Scalar(ScalarDistance::new()),
        DistanceSlot::Ard(ArdDistance::of_dims(2)),
    ];
    let sources = slots.iter().enumerate().map(|(k, slot)| match *slot {
        DistanceSlot::Scalar(s) => s.from_vec(line(k as f64 + 1.0, 0..3, 0..3)),
        DistanceSlot::Ard(a) => a.from_vecs(vec![line(4.0, 0..3, 0..3), line(5.0, 0..3, 0..3)]),
    });
    let store = TrainSources::<f64>::bind(&slots, sources, 3).expect("store");
    assert_eq!(
        store.dense_f64(),
        vec![
            (SlotShape::Scalar, line(1.0, 0..3, 0..3)),
            (SlotShape::Scalar, line(2.0, 0..3, 0..3)),
            (
                SlotShape::Ard(2),
                [line(4.0, 0..3, 0..3), line(5.0, 0..3, 0..3)].concat(),
            ),
        ]
    );
}

/// Sources in any order land at the number a compiled leaf reads: by
/// shape, in the kernel's slot order.
#[test]
fn sources_bind_by_shape_in_the_kernels_slot_order() {
    let (s1, s2) = (ScalarDistance::new(), ScalarDistance::new());
    let a = ArdDistance::of_dims(2);
    let slots = [
        DistanceSlot::Scalar(s1),
        DistanceSlot::Ard(a),
        DistanceSlot::Scalar(s2),
    ];
    let first = |sources: &dyn SquareSlots<f64>| sources.scalar(0).expect("slot")[(1, 0)];
    use crate::test_check::assert_close;
    let store = TrainSources::<f64>::bind(
        &slots,
        [
            s2.from_vec(line(2.0, 0..3, 0..3)),
            a.from_vecs(vec![line(4.0, 0..3, 0..3), line(5.0, 0..3, 0..3)]),
            s1.from_vec(line(1.0, 0..3, 0..3)),
        ],
        3,
    )
    .expect("store");
    assert_close(first(&store), 1.0, 0.0);
    assert_close(store.scalar(1).expect("slot")[(1, 0)], 2.0, 0.0);
    let Ok(ArdSquare::Packed(triangles)) = store.ard(0) else {
        panic!("packed");
    };
    assert_close(triangles.get(1, 1, 0), 5.0, 0.0);
    let cross = [line(10.0, 0..3, 0..2), line(20.0, 0..3, 0..2)];
    let bands = [line(40.0, 0..3, 0..2), line(50.0, 0..3, 0..2)];
    let band_refs: Vec<&[f64]> = bands.iter().map(Vec::as_slice).collect();
    let mut scratch = QueryScratch::new();
    let bound = QuerySources::<f64>::bind_rect(
        &slots,
        [
            a.borrow(&band_refs),
            s2.borrow(&cross[1]),
            s1.borrow(&cross[0]),
        ],
        (3, 2),
        &mut scratch,
    )
    .expect("bind");
    assert_close(bound.scalar(0).expect("slot")[(1, 0)], 10.0, 0.0);
    assert_close(bound.scalar(1).expect("slot")[(1, 0)], 20.0, 0.0);
    let Ok(ArdRect::Unchecked(blocks)) = bound.ard(0) else {
        panic!("read in place");
    };
    let column = blocks.column(1, 0).expect("checked").expect("dense");
    assert_close(column[1], 50.0, 0.0);
    // Blocks bound to be checked as they are read are not a square.
    let squares = QuerySquares(bound);
    assert!(matches!(
        squares.ard(0),
        Err(GprError::UnsupportedKernelOperation { .. })
    ));
    assert!(matches!(
        squares.ard(1),
        Err(GprError::UnsupportedKernelOperation { .. })
    ));
}

#[test]
fn a_borrowed_table_is_read_in_place_by_an_f64_model() {
    let image = ScalarDistance::new();
    let cross = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
    let slots = [DistanceSlot::Scalar(image)];
    let mut scratch = QueryScratch::new();
    let bound =
        QuerySources::<f64>::bind_rect(&slots, [image.borrow(&cross)], (3, 2), &mut scratch)
            .expect("bind");
    let view = bound.scalar(0).expect("slot");
    assert_eq!(view.as_ptr(), cross.as_ptr());
}

#[test]
fn an_owned_table_moves_into_an_f64_store() {
    let image = ScalarDistance::new();
    let train = vec![0.0, 1.0, 1.0, 0.0];
    let ptr = train.as_ptr();
    let slots = [DistanceSlot::Scalar(image)];
    let store = TrainSources::<f64>::bind(&slots, [image.from_vec(train)], 2).expect("store");
    let view = store.scalar(0).expect("slot");
    assert_eq!(view.as_ptr(), ptr);
}

/// An `f64` store keeps owned ARD tables as they are, after the exact
/// check; an `f32` store packs them. Both read the same pairs.
#[test]
fn owned_ard_tables_move_into_an_f64_store() {
    use crate::test_check::assert_close;
    let bands = ArdDistance::of_dims(2);
    let slots = [DistanceSlot::Ard(bands)];
    let tables = vec![line(1.0, 0..3, 0..3), line(2.0, 0..3, 0..3)];
    let kept = tables.clone();
    let ptr = kept[1].as_ptr();
    let store = TrainSources::<f64>::bind(&slots, [bands.from_vecs(kept)], 3).expect("f64");
    let Ok(ArdSquare::Packed(view)) = store.ard(0) else {
        panic!("ard slot");
    };
    assert_eq!(view.lower().expect("lower").column(1, 0).as_ptr(), ptr);
    assert!(view.lower().expect("lower").packed_block(0).is_none());
    let narrow =
        TrainSources::<f32>::bind(&slots, [bands.from_vecs(tables.clone())], 3).expect("f32");
    let Ok(ArdSquare::Packed(packed)) = narrow.ard(0) else {
        panic!("ard slot");
    };
    assert!(packed.lower().expect("lower").packed_block(0).is_some());
    for (k, table) in tables.iter().enumerate() {
        for j in 0..3 {
            for i in 0..3 {
                assert_close(view.get(k, i, j), table[i + j * 3], 0.0);
                assert_close(f64::from(packed.get(k, i, j)), table[i + j * 3], 0.0);
            }
        }
    }
    // An asymmetric table is refused before it is kept.
    let mut skewed = tables;
    skewed[0][1] += 1.0;
    assert!(matches!(
        TrainSources::<f64>::bind(&slots, [bands.from_vecs(skewed)], 3),
        Err(GprError::InvalidDistance { .. })
    ));
}
