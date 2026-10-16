//! Which path an inducing change or an `X` step of [`OnlineSgpr`] takes: the
//! incremental update of ADR 0005 or 0004, or an assembly of the changed system.

use super::*;
use crate::kernel::{MaternKernel, MaternNu, RbfArdKernel, RbfKernel, WhiteKernel};
use crate::precision::GpScalar;
use crate::{Fixed, GaussianLikelihood, MixedPrecision, Sgpr, SinglePrecision};
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
    let z2 = [
        0.45, 0.65, 0.104, 0.9, 0.357, 0.467, 0.933, 0.3, 0.696, 0.233, 0.442, 0.9, 0.914, 0.058,
        0.633, 0.242,
    ];
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
    assert!(!inducing_pivots_hold(&model));
    let insert = Change::Insert(vec![3.0]);
    assert!(!updates_in_place(&mut model, &insert));
    apply(&mut model, &insert);
    let delete_other = Change::Delete(0);
    assert!(!updates_in_place(&mut model, &delete_other));
    let delete_duplicate = Change::Delete(2);
    assert!(!updates_in_place(&mut model, &delete_duplicate));
    apply(&mut model, &delete_duplicate);
    assert!(inducing_pivots_hold(&model));
    assert!(updates_in_place(&mut model, &Change::Delete(2)));
}

/// Whether the factor of `K_mm` of `model` holds the threshold of an
/// inducing change, read from the factor itself.
fn inducing_pivots_hold(model: &OnlineSgpr<Fixed>) -> bool {
    PivotGate::InducingStep.holds(min_pivot_ratio(model.state.k_mm_l.as_ref()))
}

/// The pivot check an `X` step reads ([`OnlineSgpr::updates_x_in_place`])
/// is kept with the factor of `K_mm`: every change that replaces the
/// factor (an inducing change, new `θ`) reads it again, and an `X` step,
/// which leaves the factor as it is, keeps it. It is not the threshold of
/// an inducing change: a pivot far above the rounding residue (a near
/// duplicate `1e-3` away, `r ≈ 5e-7`) is below the threshold of an
/// inducing change and above that of an `X` step.
#[test]
fn x_steps_read_the_pivot_check_of_the_current_factor() {
    let x: Vec<f64> = (0..12).map(|i| 0.3 * f64::from(i)).collect();
    let y: Vec<f64> = x.iter().map(|&v| (1.3 * v).sin()).collect();
    let rbf = KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"));
    let mut model = online(rbf, &x, 1, &y, &[0.5, 1.7, 1.7]);
    let kept_matches = |model: &OnlineSgpr<Fixed>| {
        model.updates_x_in_place()
            == PivotGate::XStep.holds(min_pivot_ratio(model.state.k_mm_l.as_ref()))
    };
    assert!(!model.updates_x_in_place());
    model.insert(&[0.9], 0.4).expect("insert");
    assert!(!model.updates_x_in_place());
    model.delete(model.point_ids()[0]).expect("delete");
    assert!(!model.updates_x_in_place());
    apply(&mut model, &Change::Delete(2));
    assert!(model.updates_x_in_place() && kept_matches(&model));
    model.insert(&[2.9], -0.2).expect("insert");
    assert!(model.updates_x_in_place());
    apply(&mut model, &Change::Insert(vec![1.7 + 1e-3]));
    let ratio = min_pivot_ratio(model.state.k_mm_l.as_ref());
    assert!((1e-8..1e-5).contains(&ratio), "pivot ratio {ratio:e}");
    assert!(model.updates_x_in_place() && kept_matches(&model));
    assert!(!model.updates_inducing_in_place());
    model.insert(&[1.1], 0.3).expect("insert");
    assert!(model.updates_x_in_place());
    let mut theta = vec![0.0; model.num_params()];
    model.get_params(&mut theta).expect("θ");
    // A lengthscale of `1e-2` takes the near duplicate apart.
    theta[0] = 1e-2_f64.ln();
    model.set_params(&theta).expect("set θ");
    assert!(model.updates_x_in_place() && kept_matches(&model));
    assert!(model.updates_inducing_in_place());
}

/// The Forrester function of the `compare/perf` benchmark at `n` points of
/// `[0, 1]`, with the noise a fixed sequence of the crate's generator
/// gives.
fn forrester(n: usize) -> (Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
    let y = x
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            (6.0 * v - 2.0).powi(2) * (12.0 * v - 4.0).sin() + ((i * 7919) % 13) as f64 / 6.0 - 1.0
        })
        .collect();
    (x, y)
}

/// A smooth kernel on a few inducing points has a `K_mm` far more
/// ill-conditioned than a duplicate-free set needs it to be, with no
/// duplicate in it: the eight k-means centres of the Forrester problem of
/// `compare/perf` (`[0, 1]`, 1-D RBF at `ℓ = 1`) have a smallest pivot
/// ratio of about `1e-13`, and those of the weighted sphere (2-D ARD at
/// `ℓ = 4`) of about `1e-7`. An `X` step takes the rank-1 path there, for
/// every storage scalar. It assembled the system again at every step when
/// the threshold of an `X` step was that of an inducing change (`1e-5`,
/// `1e-3` for `f32`); an inducing change still assembles. The accuracy of
/// the steps against the exact system is checked in `tests/sgpr_ops.rs`.
#[test]
fn x_steps_stay_rank_1_for_a_smooth_ill_conditioned_k_mm() {
    fn run<P: GpScalar>(
        kernel: KernelSpec,
        x: &[f64],
        d: usize,
        y: &[f64],
        z: &[f64],
        ratio: std::ops::Range<f64>,
    ) {
        let n = y.len();
        let mut model = Sgpr::new(kernel, GaussianLikelihood::new(0.1).expect("noise"))
            .with_optimizer(Fixed)
            .with_precision::<P>()
            .factor(x, n, d, y, z, z.len() / d)
            .map_err(|(_, e)| e)
            .expect("factor")
            .into_online();
        let measured = min_pivot_ratio(model.state.k_mm_l.as_ref());
        assert!(ratio.contains(&measured), "pivot ratio {measured:e}");
        assert!(model.state.k_mm_jitter <= 0.0, "the factor took no jitter");
        assert!(!model.updates_inducing_in_place());
        for step in 0..16 {
            assert!(model.updates_x_in_place(), "insert {step}");
            let at = f64::from(step) / 16.0 + 0.02;
            model
                .insert(&vec![at; d], 0.1 * f64::from(step))
                .expect("insert");
        }
        for step in 0..16 {
            assert!(model.updates_x_in_place(), "delete {step}");
            let id = model.point_ids()[3 * step];
            model.delete(id).expect("delete");
        }
    }
    let (x, y) = forrester(64);
    // The first eight k-means centres of the benchmark (rounded to three digits).
    let z1 = [0.225, 0.606, 0.975, 0.09, 0.42, 0.737, 0.802, 0.543];
    let rbf = || KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"));
    run::<DoublePrecision>(rbf(), &x, 1, &y, &z1, 1e-14..1e-11);
    run::<SinglePrecision>(rbf(), &x, 1, &y, &z1, 1e-14..1e-11);
    run::<MixedPrecision>(rbf(), &x, 1, &y, &z1, 1e-14..1e-11);
    // The sphere problem: a grid of `[0, 1]²`, `ℓ_d = 4`, its first eight k-means centres.
    let grid: Vec<f64> = (0..8).map(|i| f64::from(i) / 7.0).collect();
    let n = 64;
    let x2: Vec<f64> = (0..2 * n)
        .map(|i| {
            if i < n {
                grid[i % 8]
            } else {
                grid[(i - n) / 8]
            }
        })
        .collect();
    let y2: Vec<f64> = (0..n)
        .map(|i| (x2[i] / 0.25).powi(2) + x2[n + i].powi(2))
        .collect();
    let z2 = [
        0.1, 0.45, 0.8, 0.3, 0.6, 0.95, 0.2, 0.7, 0.4, 0.85, 0.15, 0.55, 0.9, 0.05, 0.65, 0.35,
    ];
    let ard = || KernelSpec::from(RbfArdKernel::new(&[4.0, 4.0]).expect("ℓ"));
    run::<DoublePrecision>(ard(), &x2, 2, &y2, &z2, 1e-9..1e-5);
    run::<SinglePrecision>(ard(), &x2, 2, &y2, &z2, 1e-9..1e-5);
}
