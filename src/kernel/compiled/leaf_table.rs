//! Every built-in leaf through every core operation.
//!
//! [`leaf_index`] matches every [`KernelSpec`] variant without a wildcard,
//! so a new built-in leaf is a compile error here until it has an index,
//! and [`every_builtin_leaf_is_in_the_table`] fails until the table holds
//! an instance of it. The table then runs it through the operations a model
//! calls: parameters, the Gram from coordinates and from distances, the
//! rectangular cross, the diagonal, `∂K/∂θ` and `∂²K/∂θ∂θ` against central
//! differences, the coordinate derivative where the leaf has one, and a save
//! and load. See docs/architecture.md §7 (adding a leaf).

use crate::kernel::GramInputs;
use crate::kernel::{
    ConstantKernel, KernelSpec, LinearKernel, MaternArdKernel, MaternKernel, MaternNu,
    PeriodicKernel, RationalQuadraticArdKernel, RationalQuadraticKernel, RbfArdKernel, RbfKernel,
    Triangle, WhiteKernel,
};
use crate::math::Accurate;
use faer::Mat;

/// Names of the built-in leaves, by [`leaf_index`].
const LEAVES: [&str; 10] = [
    "rbf",
    "rbf_ard",
    "matern",
    "matern_ard",
    "periodic",
    "rational_quadratic",
    "rational_quadratic_ard",
    "constant",
    "linear",
    "white",
];

/// The index of a built-in leaf in [`LEAVES`]; `None` for a custom leaf or a
/// composite. No wildcard: a new variant does not compile until it is here.
fn leaf_index(spec: &KernelSpec) -> Option<usize> {
    match spec {
        KernelSpec::Rbf(_) => Some(0),
        KernelSpec::RbfArd(_) => Some(1),
        KernelSpec::Matern(_) => Some(2),
        KernelSpec::MaternArd(_) => Some(3),
        KernelSpec::Periodic(_) => Some(4),
        KernelSpec::RationalQuadratic(_) => Some(5),
        KernelSpec::RationalQuadraticArd(_) => Some(6),
        KernelSpec::Constant(_) => Some(7),
        KernelSpec::Linear(_) => Some(8),
        KernelSpec::White(_) => Some(9),
        KernelSpec::Custom(_)
        | KernelSpec::Supplied(_)
        | KernelSpec::Sum(..)
        | KernelSpec::Product(..) => None,
    }
}

/// One instance of every built-in leaf, on two input dimensions.
fn table() -> Vec<KernelSpec> {
    vec![
        KernelSpec::from(RbfKernel::new(0.9).expect("rbf")),
        KernelSpec::from(RbfArdKernel::new(&[0.8, 1.4]).expect("rbf ard")),
        KernelSpec::from(MaternKernel::new(1.1, MaternNu::FiveHalves).expect("matern")),
        KernelSpec::from(
            MaternArdKernel::new(&[0.7, 1.3], MaternNu::ThreeHalves).expect("matern ard"),
        ),
        KernelSpec::from(PeriodicKernel::new(0.9, 1.7).expect("periodic")),
        KernelSpec::from(RationalQuadraticKernel::new(1.2, 0.8).expect("rq")),
        KernelSpec::from(RationalQuadraticArdKernel::new(&[0.9, 1.5], 1.3).expect("rq ard")),
        KernelSpec::from(ConstantKernel::new(1.6).expect("constant")),
        KernelSpec::from(LinearKernel::new(0.7).expect("linear")),
        KernelSpec::from(WhiteKernel::new(0.3).expect("white")),
    ]
}

/// Five points in two dimensions, no two equal (a coordinate derivative is
/// undefined where points coincide for some leaves).
fn points() -> Mat<f64> {
    Mat::from_fn(5, 2, |i, j| {
        0.37 * i as f64 - 0.21 * j as f64 + 0.05 * (i * j) as f64
    })
}

fn queries() -> Mat<f64> {
    Mat::from_fn(3, 2, |i, j| 0.5 - 0.29 * i as f64 + 0.17 * j as f64)
}

fn close(got: f64, want: f64, tol: f64) -> bool {
    (got - want).abs() <= tol * want.abs().max(1.0)
}

fn gram(spec: &KernelSpec, x: &Mat<f64>) -> Mat<f64> {
    let compiled = spec.compile();
    let n = x.nrows();
    let mut out = Mat::zeros(n, n);
    let mut scratch = Mat::zeros(n, n);
    compiled
        .apply_points::<Accurate>(x.as_ref(), out.as_mut(), Triangle::Full, scratch.as_mut())
        .expect("gram");
    out
}

fn with_params(spec: &KernelSpec, params: &[f64]) -> KernelSpec {
    let mut moved = spec.clone();
    moved.set_params(params).expect("params");
    moved
}

#[test]
fn every_builtin_leaf_is_in_the_table() {
    let mut seen = [false; LEAVES.len()];
    for spec in table() {
        let index = leaf_index(&spec).expect("a built-in leaf");
        seen[index] = true;
    }
    for (name, hit) in LEAVES.iter().zip(seen) {
        assert!(hit, "the leaf table has no {name}");
    }
}

#[test]
fn every_builtin_leaf_runs_every_core_operation() {
    let x = points();
    let xs = queries();
    let n = x.nrows();
    let h = 1e-6;
    for spec in table() {
        let name = LEAVES[leaf_index(&spec).expect("built-in")];
        let compiled = spec.compile();
        let p = spec.num_params();
        assert_eq!(compiled.num_params(), p, "{name}");
        let mut params = vec![0.0; p];
        spec.get_params(&mut params).expect("get");
        assert_eq!(
            with_params(&spec, &params),
            spec,
            "{name}: set(get) is the identity"
        );

        // The Gram from coordinates: Lower agrees with Full, and the
        // distance path (when the leaf reads distances) agrees with it.
        let full = gram(&spec, &x);
        let mut lower = Mat::zeros(n, n);
        let mut scratch = Mat::zeros(n, n);
        compiled
            .apply_points::<Accurate>(
                x.as_ref(),
                lower.as_mut(),
                Triangle::Lower,
                scratch.as_mut(),
            )
            .expect("lower");
        let mut dist = Mat::zeros(n, n);
        crate::kernel::fill_squared_euclidean(x.as_ref(), dist.as_mut(), &mut []);
        let mut from_dist = Mat::zeros(n, n);
        let mut nested = Vec::new();
        compiled
            .eval_gram::<Accurate>(
                GramInputs {
                    x: x.as_ref(),
                    dist: Some(dist.as_ref()),
                    ard: None,
                    slots: None,
                },
                from_dist.as_mut(),
                Triangle::Lower,
                scratch.as_mut(),
                &mut nested,
            )
            .expect("from distances");
        for j in 0..n {
            for i in j..n {
                assert!(close(lower[(i, j)], full[(i, j)], 1e-14), "{name} lower");
                assert!(
                    close(from_dist[(i, j)], full[(i, j)], 1e-12),
                    "{name} distances"
                );
            }
        }

        // The diagonal and the cross block of the same points.
        let mut diag = vec![0.0; n];
        compiled
            .fill_diag_points(x.as_ref(), &mut diag)
            .expect("diag");
        let mut cross = Mat::zeros(n, xs.nrows());
        let mut cross_scratch = Mat::zeros(n, xs.nrows());
        compiled
            .apply_cross_points::<Accurate>(
                x.as_ref(),
                xs.as_ref(),
                cross.as_mut(),
                cross_scratch.as_mut(),
            )
            .expect("cross");
        let mut joint = Mat::zeros(n + xs.nrows(), 2);
        for j in 0..2 {
            for i in 0..n {
                joint[(i, j)] = x[(i, j)];
            }
            for i in 0..xs.nrows() {
                joint[(n + i, j)] = xs[(i, j)];
            }
        }
        let joint_gram = gram(&spec, &joint);
        for i in 0..n {
            assert!(close(diag[i], full[(i, i)], 1e-12), "{name} diagonal");
            for c in 0..xs.nrows() {
                // White has no cross covariance; every other leaf's cross
                // block is the off-diagonal block of the joint Gram.
                let want = if name == "white" {
                    0.0
                } else {
                    joint_gram[(i, n + c)]
                };
                assert!(close(cross[(i, c)], want, 1e-12), "{name} cross");
            }
        }

        // `∂K/∂θ` against central differences of the Gram, and `∂²K/∂θ∂θ`
        // against central differences of `∂K/∂θ`.
        for a in 0..p {
            let mut d_k = Mat::zeros(n, n);
            compiled
                .grad_points::<Accurate>(
                    x.as_ref(),
                    d_k.as_mut(),
                    a,
                    Triangle::Full,
                    scratch.as_mut(),
                )
                .expect("grad");
            let mut up = params.clone();
            let mut down = params.clone();
            up[a] += h;
            down[a] -= h;
            let (g_up, g_down) = (
                gram(&with_params(&spec, &up), &x),
                gram(&with_params(&spec, &down), &x),
            );
            for j in 0..n {
                for i in 0..n {
                    let fd = (g_up[(i, j)] - g_down[(i, j)]) / (2.0 * h);
                    assert!(
                        close(d_k[(i, j)], fd, 1e-6),
                        "{name} ∂K/∂θ{a}: {} vs {fd}",
                        d_k[(i, j)]
                    );
                }
            }
            for b in 0..p {
                let mut d2_k = Mat::zeros(n, n);
                compiled
                    .hess_points::<Accurate>(
                        x.as_ref(),
                        d2_k.as_mut(),
                        a,
                        b,
                        Triangle::Full,
                        scratch.as_mut(),
                    )
                    .expect("hess");
                let grad_at = |theta: &[f64]| {
                    let moved = with_params(&spec, theta).compile();
                    let mut out = Mat::zeros(n, n);
                    let mut s = Mat::zeros(n, n);
                    moved
                        .grad_points::<Accurate>(
                            x.as_ref(),
                            out.as_mut(),
                            a,
                            Triangle::Full,
                            s.as_mut(),
                        )
                        .expect("grad");
                    out
                };
                let mut up = params.clone();
                let mut down = params.clone();
                up[b] += h;
                down[b] -= h;
                let (d_up, d_down) = (grad_at(&up), grad_at(&down));
                for j in 0..n {
                    for i in 0..n {
                        let fd = (d_up[(i, j)] - d_down[(i, j)]) / (2.0 * h);
                        assert!(close(d2_k[(i, j)], fd, 1e-5), "{name} ∂²K/∂θ{a}∂θ{b}");
                    }
                }
            }
        }

        // The coordinate derivative `∂k(x_i, xs_c)/∂xs_c[dim]`, where the leaf
        // has one.
        for dim in 0..2 {
            let mut d_k = Mat::zeros(n, xs.nrows());
            match compiled.grad_wrt_coord_dim::<Accurate>(
                x.as_ref(),
                xs.as_ref(),
                d_k.as_mut(),
                dim,
            ) {
                Ok(()) => {
                    for c in 0..xs.nrows() {
                        let mut up = xs.clone();
                        let mut down = xs.clone();
                        up[(c, dim)] += h;
                        down[(c, dim)] -= h;
                        let mut k_up = Mat::zeros(n, xs.nrows());
                        let mut k_down = Mat::zeros(n, xs.nrows());
                        compiled
                            .apply_cross_points::<Accurate>(
                                x.as_ref(),
                                up.as_ref(),
                                k_up.as_mut(),
                                cross_scratch.as_mut(),
                            )
                            .expect("cross");
                        compiled
                            .apply_cross_points::<Accurate>(
                                x.as_ref(),
                                down.as_ref(),
                                k_down.as_mut(),
                                cross_scratch.as_mut(),
                            )
                            .expect("cross");
                        for i in 0..n {
                            let fd = (k_up[(i, c)] - k_down[(i, c)]) / (2.0 * h);
                            assert!(close(d_k[(i, c)], fd, 1e-6), "{name} ∂k/∂z[{dim}]");
                        }
                    }
                }
                Err(crate::error::GprError::CoordGradientUnsupported) => {}
                Err(err) => panic!("{name} coordinate derivative: {err}"),
            }
        }

        // A save and load keeps the leaf.
        let fitted = crate::Gpr::new(
            spec.clone(),
            crate::GaussianLikelihood::new(0.2).expect("noise"),
        )
        .with_optimizer(crate::Fixed)
        .factor(&[0.0, 0.4, 0.9, 0.2, 0.7, 1.3], 3, 2, &[0.1, -0.2, 0.4])
        .expect("factor");
        let dir =
            std::env::temp_dir().join(format!("gprx-leaf-table-{name}-{}", std::process::id()));
        fitted.save(&dir).expect("save");
        let loaded = crate::LoadedGpr::load(&dir, &crate::PersistRegistry::new()).expect("load");
        let _ = std::fs::remove_dir_all(&dir);
        match loaded {
            crate::LoadedGpr::Double(model) => assert_eq!(model.kernel(), &spec, "{name} save"),
            other => panic!("{name}: loaded {other:?}"),
        }
    }
}
