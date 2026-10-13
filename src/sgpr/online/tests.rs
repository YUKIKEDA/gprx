//! Which path an inducing change of [`OnlineSgpr`] takes: the incremental
//! update of ADR 0005, or an assembly of the changed inducing set.

use super::*;
use crate::kernel::{MaternKernel, MaternNu, RbfArdKernel, RbfKernel, WhiteKernel};
use crate::{Fixed, GaussianLikelihood, Sgpr};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

/// One inducing change.
enum Change {
    /// An insert at these coordinates (as the caller gives them).
    Insert(Vec<f64>),
    /// A delete at this index.
    Delete(usize),
}

/// Whether `change` updates the factors of `model` in place: the gate of
/// [`OnlineSgpr::updates_inducing_in_place`] holds and the update succeeds
/// on a copy of the factors, so [`OnlineSgpr::update_inducing`] does not
/// fall back to an assembly either.
fn updates_in_place(model: &mut OnlineSgpr<Fixed>, change: &Change) -> bool {
    if !model.updates_inducing_in_place() {
        return false;
    }
    let mut state = model.vfe_state();
    let noise = model.state.core.likelihood.noise_variance();
    let result = match change {
        Change::Insert(z_obs) => {
            let mut z_new = Vec::new();
            model
                .state
                .core
                .map_point(z_obs, &mut z_new)
                .expect("map point");
            let core = &model.state.core;
            let border = with_kernel_exp!(core.math, M => coordinate_border::<M, _>(
                &core.kernel,
                &core.x_train,
                core.n,
                core.d,
                &core.z_train,
                core.m,
                &z_new,
                &mut model.scratch.storage,
            ))
            .expect("border");
            inducing_append(&mut state, border, None, noise, &model.state.core.y_train)
        }
        Change::Delete(idx) => inducing_delete(&mut state, noise, &model.state.core.y_train, *idx),
    };
    result.is_ok()
}

/// Applies `change` through the public methods.
fn apply(model: &mut OnlineSgpr<Fixed>, change: &Change) {
    match change {
        Change::Insert(z) => {
            model.insert_inducing(z).expect("insert inducing");
        }
        Change::Delete(idx) => {
            let id = model.inducing_ids()[*idx];
            model.delete_inducing(id).expect("delete inducing");
        }
    }
}

fn online(kernel: KernelSpec, x: &[f64], d: usize, y: &[f64], z: &[f64]) -> OnlineSgpr<Fixed> {
    Sgpr::new(kernel, GaussianLikelihood::new(0.1).expect("noise"))
        .with_optimizer(Fixed)
        .factor(x, y.len(), d, y, z, z.len() / d)
        .map_err(|(_, e)| e)
        .expect("factor")
        .into_online()
}

/// The inducing sequences of `tests/sgpr_ops.rs` (`run_inducing_sequence`:
/// the same data, kernels, seeds, gap, and draws of the random number
/// generator) take the incremental update at every step: no factor they
/// reach has a pivot at or below the threshold, and no update falls back
/// to an assembly.
#[test]
fn gap_inducing_sequences_update_in_place() {
    const OPS: usize = 32;
    const M_MAX: usize = 16;
    const MIN_GAP: f64 = 0.25;
    const DRAWS: usize = 64;
    let y0 = [0.0, 1.0, 0.5, 0.25];
    let x1 = [0.0, 1.0, 2.0, 3.0];
    let z1 = [0.5, 2.5];
    let x2 = [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0];
    let z2 = [0.25, 0.75, 0.25, 0.75];
    let rbf = || KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"));
    let cases: [(KernelSpec, &[f64], &[f64], usize); 4] = [
        (rbf(), &x1, &z1, 1),
        (
            KernelSpec::from(MaternKernel::new(1.0, MaternNu::ThreeHalves).expect("ℓ")),
            &x1,
            &z1,
            1,
        ),
        (
            KernelSpec::from(RbfArdKernel::new(&[1.0, 1.5]).expect("ℓ")),
            &x2,
            &z2,
            2,
        ),
        (
            rbf() + KernelSpec::from(WhiteKernel::new(0.05).expect("white")),
            &x1,
            &z1,
            1,
        ),
    ];
    for (case, (kernel, x, z0, d)) in cases.into_iter().enumerate() {
        for seed in [0, 1, 2] {
            let mut model = online(kernel.clone(), x, d, &y0, z0);
            let mut rng = SmallRng::seed_from_u64(seed);
            for step in 0..OPS {
                let m = model.m();
                let z = model.z().to_vec();
                let wants_insert = m == 1 || (m < M_MAX && rng.random::<bool>());
                let drawn = wants_insert
                    .then(|| {
                        (0..DRAWS).find_map(|_| {
                            let candidate: Vec<f64> =
                                (0..d).map(|_| 4.0 * rng.random::<f64>()).collect();
                            let clear = (0..m).all(|i| {
                                let dist2: f64 =
                                    (0..d).map(|f| (z[f * m + i] - candidate[f]).powi(2)).sum();
                                dist2 >= MIN_GAP * MIN_GAP
                            });
                            clear.then_some(candidate)
                        })
                    })
                    .flatten();
                let change = match drawn {
                    Some(z_new) => Change::Insert(z_new),
                    None => Change::Delete(rng.random_range(0..m)),
                };
                assert!(
                    updates_in_place(&mut model, &change),
                    "case={case} seed={seed} step={step}: the change assembles again"
                );
                apply(&mut model, &change);
            }
        }
    }
}

/// An exact duplicate the factor takes without jitter leaves a pivot of
/// `K_mm` at the rounding residue: every inducing change after it assembles
/// the system again until a delete of the duplicate gives a factor whose
/// pivots hold, after which the changes update in place again.
#[test]
fn a_factor_with_a_rounding_pivot_assembles_inducing_changes() {
    let x: Vec<f64> = (0..12).map(|i| 0.3 * f64::from(i)).collect();
    let y: Vec<f64> = x.iter().map(|&v| (1.3 * v).sin()).collect();
    let rbf = KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"));
    // Factored with the duplicate, as an insert of it assembles; at these
    // points the residue of its pivot is positive, so the factor takes no
    // jitter (at `[0.5, 2.5, 0.5]` it is not, and the factor takes one).
    let mut model = online(rbf, &x, 1, &y, &[0.5, 1.7, 1.7]);
    assert!(model.state.k_mm_jitter <= 0.0, "the factor took no jitter");
    assert!(!factor_pivots_hold(model.state.k_mm_l.as_ref()));
    let insert = Change::Insert(vec![3.0]);
    assert!(!updates_in_place(&mut model, &insert));
    apply(&mut model, &insert);
    let delete_other = Change::Delete(0);
    assert!(!updates_in_place(&mut model, &delete_other));
    let delete_duplicate = Change::Delete(2);
    assert!(!updates_in_place(&mut model, &delete_duplicate));
    apply(&mut model, &delete_duplicate);
    assert!(factor_pivots_hold(model.state.k_mm_l.as_ref()));
    assert!(updates_in_place(&mut model, &Change::Delete(2)));
}
