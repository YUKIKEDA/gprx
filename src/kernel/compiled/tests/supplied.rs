//! Leaves on supplied squared distances against the same leaves on
//! coordinates.

use crate::GprError;
use crate::kernel::compiled::CrossViews;
use crate::kernel::compiled::gram::GramInputs;
use crate::kernel::compiled::supplied::{ArdRect, ArdSquare, RectSlots, SquareTable, unbound};
use crate::kernel::compiled::weighted::{DiagAccum, WeightedWalk};
use crate::kernel::dist::{ArdBlocks, ArdSqDiffBuf, BlockList};
use crate::kernel::{
    ArdDistance, CompiledKernel, ConstantKernel, DistanceKernel, KernelSpec, MaternArdKernel,
    MaternKernel, MaternNu, PeriodicKernel, RationalQuadraticArdKernel, RationalQuadraticKernel,
    RbfArdKernel, RbfKernel, ScalarDistance, SuppliedSpec, Triangle, WhiteKernel, WithPoints,
};
use crate::math::Accurate;
use crate::test_check::assert_close;
use faer::{Mat, MatRef};

/// Rectangular supplies by shape in slot order.
struct RectTable<'a> {
    scalar: Vec<MatRef<'a, f64>>,
    ard: Vec<ArdRect<'a, f64>>,
}

impl RectSlots<f64> for RectTable<'_> {
    fn scalar(&self, at: usize) -> Result<MatRef<'_, f64>, GprError> {
        self.scalar.get(at).copied().ok_or_else(unbound)
    }

    fn ard(&self, at: usize) -> Result<ArdRect<'_, f64>, GprError> {
        self.ard.get(at).copied().ok_or_else(unbound)
    }
}

const N: usize = 5;
const M: usize = 3;

/// Column `k` of the training (`N`) or query (`M`) samples.
fn coords(k: usize, rows: usize, offset: f64) -> Vec<f64> {
    (0..rows)
        .map(|i| ((i as f64 + offset) * (0.37 + 0.11 * k as f64)).sin() * (1.0 + k as f64))
        .collect()
}

fn mat_from_cols(cols: &[Vec<f64>]) -> Mat<f64> {
    let rows = cols[0].len();
    Mat::from_fn(rows, cols.len(), |i, j| cols[j][i])
}

fn sq(a: &[f64], b: &[f64]) -> Mat<f64> {
    Mat::from_fn(a.len(), b.len(), |i, j| (a[i] - b[j]) * (a[i] - b[j]))
}

struct Problem {
    image: ScalarDistance,
    bands: ArdDistance,
    spec: DistanceKernel<WithPoints>,
    /// The coordinate leaf's column, train and query.
    x: Mat<f64>,
    xs: Mat<f64>,
    /// Scalar slot squares.
    train_sq: Mat<f64>,
    cross_sq: Mat<f64>,
    /// ARD slot.
    train_ard: ArdSqDiffBuf<f64>,
    cross_blocks: Vec<Vec<f64>>,
    /// Reference kernels on coordinates, one per slot plus the coordinate leaf.
    ref_image: CompiledKernel<f64>,
    ref_bands: CompiledKernel<f64>,
    ref_points: CompiledKernel<f64>,
    image_x: (Mat<f64>, Mat<f64>),
    bands_x: (Mat<f64>, Mat<f64>),
}

fn problem() -> Problem {
    let image = ScalarDistance::new();
    let bands = ArdDistance::new(2).expect("dims");
    let rbf = RbfKernel::new(0.8).expect("ell");
    let ard = RbfArdKernel::new(&[0.7, 1.3]).expect("ell");
    let point = RbfKernel::new(0.9).expect("ell");
    let spec = image.kernel(rbf) * bands.kernel(ard.clone()).expect("dims")
        + ConstantKernel::new(0.3).expect("c")
        + image.kernel(MaternKernel::new(1.1, MaternNu::FiveHalves).expect("ell"))
            * KernelSpec::from(point);
    let c0 = (coords(0, N, 0.0), coords(0, M, 0.5));
    let c1 = (coords(1, N, 0.0), coords(1, M, 0.5));
    let c2 = (coords(2, N, 0.0), coords(2, M, 0.5));
    let c3 = (coords(3, N, 0.0), coords(3, M, 0.5));
    let bands_train = mat_from_cols(&[c1.0.clone(), c2.0.clone()]);
    let cross_blocks = vec![
        sq(&c1.0, &c1.1)
            .col_iter()
            .flat_map(|c| c.iter().copied().collect::<Vec<_>>())
            .collect(),
        sq(&c2.0, &c2.1)
            .col_iter()
            .flat_map(|c| c.iter().copied().collect::<Vec<_>>())
            .collect(),
    ];
    Problem {
        image,
        bands,
        spec,
        x: mat_from_cols(std::slice::from_ref(&c3.0)),
        xs: mat_from_cols(std::slice::from_ref(&c3.1)),
        train_sq: sq(&c0.0, &c0.0),
        cross_sq: sq(&c0.0, &c0.1),
        train_ard: ArdSqDiffBuf::new(bands_train.as_ref()).expect("cache"),
        cross_blocks,
        ref_image: (KernelSpec::from(rbf)).compile(),
        ref_bands: KernelSpec::from(ard).compile(),
        ref_points: KernelSpec::from(point).compile(),
        image_x: (mat_from_cols(&[c0.0]), mat_from_cols(&[c0.1])),
        bands_x: (bands_train, mat_from_cols(&[c1.1, c2.1])),
    }
}

impl Problem {
    fn square_table(&self) -> SquareTable<'_, f64> {
        SquareTable {
            scalar: vec![self.train_sq.as_ref()],
            ard: vec![ArdSquare::Packed(self.train_ard.view())],
        }
    }

    fn rect_table(&self) -> RectTable<'_> {
        RectTable {
            scalar: vec![self.cross_sq.as_ref()],
            ard: vec![ArdRect::Checked(ArdBlocks::new(
                BlockList::Vecs(&self.cross_blocks),
                N,
                M,
                0,
            ))],
        }
    }

    fn gram(&self, compiled: &CompiledKernel<f64, SuppliedSpec>) -> Mat<f64> {
        let table = self.square_table();
        let inputs = GramInputs::<_, SuppliedSpec>::supplied(self.x.as_ref(), &table);
        let mut out = Mat::zeros(N, N);
        let mut scratch = Mat::zeros(N, N);
        compiled
            .eval_gram::<Accurate>(
                inputs,
                out.as_mut(),
                Triangle::Full,
                scratch.as_mut(),
                &mut Vec::new(),
            )
            .expect("gram");
        out
    }

    fn cross(&self, compiled: &CompiledKernel<f64, SuppliedSpec>) -> Mat<f64> {
        let table = self.rect_table();
        let mut out = Mat::zeros(N, M);
        let mut scratch = Mat::zeros(N, M);
        compiled
            .eval_cross_slots::<Accurate>(
                self.x.as_ref(),
                self.xs.as_ref(),
                &table,
                None,
                out.as_mut(),
                scratch.as_mut(),
                &mut Vec::new(),
                &mut [],
            )
            .expect("cross");
        out
    }
}

/// `K` of one coordinate kernel on `(a, b)`.
fn coord_block(k: &CompiledKernel<f64>, a: MatRef<'_, f64>, b: MatRef<'_, f64>) -> Mat<f64> {
    let mut out = Mat::zeros(a.nrows(), b.nrows());
    let mut scratch = Mat::zeros(a.nrows(), b.nrows());
    k.apply_cross_points::<Accurate>(a, b, out.as_mut(), scratch.as_mut())
        .expect("coords");
    out
}

#[test]
fn gram_and_cross_match_the_coordinate_leaves() {
    let p = problem();
    let compiled = p.spec.spec().compile();
    let matern =
        KernelSpec::from(MaternKernel::new(1.1, MaternNu::FiveHalves).expect("ell")).compile();
    let check = |got: &Mat<f64>,
                 a: (MatRef<'_, f64>, MatRef<'_, f64>, MatRef<'_, f64>),
                 b: (MatRef<'_, f64>, MatRef<'_, f64>, MatRef<'_, f64>)| {
        let k_img = coord_block(&p.ref_image, a.0, b.0);
        let k_band = coord_block(&p.ref_bands, a.1, b.1);
        let k_pt = coord_block(&p.ref_points, a.2, b.2);
        let k_mat = coord_block(&matern, a.0, b.0);
        for j in 0..got.ncols() {
            for i in 0..got.nrows() {
                let expect = k_img[(i, j)] * k_band[(i, j)] + 0.3 + k_mat[(i, j)] * k_pt[(i, j)];
                assert_close(got[(i, j)], expect, 1e-12);
            }
        }
    };
    let gram = p.gram(&compiled);
    check(
        &gram,
        (p.image_x.0.as_ref(), p.bands_x.0.as_ref(), p.x.as_ref()),
        (p.image_x.0.as_ref(), p.bands_x.0.as_ref(), p.x.as_ref()),
    );
    let cross = p.cross(&compiled);
    check(
        &cross,
        (p.image_x.0.as_ref(), p.bands_x.0.as_ref(), p.x.as_ref()),
        (p.image_x.1.as_ref(), p.bands_x.1.as_ref(), p.xs.as_ref()),
    );
}

fn with_theta(
    spec: &DistanceKernel<WithPoints>,
    theta: &[f64],
) -> CompiledKernel<f64, SuppliedSpec> {
    let mut spec = spec.clone();
    spec.set_params(theta).expect("theta");
    spec.spec().compile()
}

#[test]
fn gradients_and_hessians_match_finite_differences() {
    let p = problem();
    check_derivatives(&p, &p.spec);
}

#[test]
fn the_other_leaves_match_finite_differences() {
    let p = problem();
    let spec = p
        .image
        .kernel(RationalQuadraticKernel::new(0.9, 1.4).expect("rq"))
        * p.bands
            .kernel(MaternArdKernel::new(&[0.8, 1.2], MaternNu::ThreeHalves).expect("matern"))
            .expect("dims")
        + p.image
            .kernel(PeriodicKernel::new(1.1, 2.3).expect("periodic"))
            * p.bands
                .kernel(RationalQuadraticArdKernel::new(&[0.6, 1.1], 0.9).expect("rq ard"))
                .expect("dims")
        + WhiteKernel::new(0.2).expect("white")
        + KernelSpec::from(RbfKernel::new(0.9).expect("ell"));
    check_derivatives(&p, &spec);
}

/// `∂K/∂θ` and `∂²K/∂θ∂θ` of `spec`, square and rectangular, against central
/// differences of its Gram and cross block.
fn check_derivatives(p: &Problem, spec: &DistanceKernel<WithPoints>) {
    let compiled = spec.spec().compile();
    let n_params = compiled.num_params();
    let mut theta = vec![0.0; n_params];
    spec.get_params(&mut theta).expect("theta");
    let h = 1e-6;
    let table = p.square_table();
    let rect = p.rect_table();
    let inputs = GramInputs::<_, SuppliedSpec>::supplied(p.x.as_ref(), &table);
    let views = CrossViews {
        x1: p.x.as_ref(),
        x2: p.xs.as_ref(),
        dist: None,
        slots: &rect as &dyn RectSlots<f64>,
    };
    let grad_at = |c: &CompiledKernel<f64, SuppliedSpec>, k: usize| {
        let mut d_k = Mat::zeros(N, N);
        let mut scratch = Mat::zeros(N, N);
        c.grad_gram::<Accurate>(
            inputs,
            d_k.as_mut(),
            k,
            Triangle::Full,
            scratch.as_mut(),
            &mut Vec::new(),
        )
        .expect("grad");
        d_k
    };
    let grad_cross_at = |c: &CompiledKernel<f64, SuppliedSpec>, k: usize| {
        let mut d_k = Mat::zeros(N, M);
        let mut scratch = Mat::zeros(N, M);
        c.grad_cross_views::<Accurate>(views, d_k.as_mut(), k, scratch.as_mut(), &mut [])
            .expect("grad cross");
        d_k
    };
    for k in 0..n_params {
        let mut up = theta.clone();
        up[k] += h;
        let mut down = theta.clone();
        down[k] -= h;
        let (c_up, c_down) = (with_theta(spec, &up), with_theta(spec, &down));
        let (g_up, g_down) = (p.gram(&c_up), p.gram(&c_down));
        let analytic = grad_at(&compiled, k);
        let (x_up, x_down) = (p.cross(&c_up), p.cross(&c_down));
        let cross = grad_cross_at(&compiled, k);
        for j in 0..N {
            for i in 0..N {
                let fd = (g_up[(i, j)] - g_down[(i, j)]) / (2.0 * h);
                assert_close(analytic[(i, j)], fd, 1e-6);
            }
            if j < M {
                for i in 0..N {
                    let fd = (x_up[(i, j)] - x_down[(i, j)]) / (2.0 * h);
                    assert_close(cross[(i, j)], fd, 1e-6);
                }
            }
        }
        for l in 0..n_params {
            let mut d2 = Mat::zeros(N, N);
            let mut scratch = Mat::zeros(N, N);
            compiled
                .hess_gram::<Accurate>(
                    inputs,
                    d2.as_mut(),
                    (k, l),
                    Triangle::Full,
                    scratch.as_mut(),
                    &mut Vec::new(),
                )
                .expect("hess");
            let (gl_up, gl_down) = (grad_at(&c_up, l), grad_at(&c_down, l));
            let mut d2x = Mat::zeros(N, M);
            let mut scratch = Mat::zeros(N, M);
            compiled
                .hess_cross_views::<Accurate>(
                    views,
                    d2x.as_mut(),
                    (k, l),
                    scratch.as_mut(),
                    &mut [],
                )
                .expect("hess cross");
            let (xl_up, xl_down) = (grad_cross_at(&c_up, l), grad_cross_at(&c_down, l));
            for j in 0..N {
                for i in 0..N {
                    let fd = (gl_up[(i, j)] - gl_down[(i, j)]) / (2.0 * h);
                    assert_close(d2[(i, j)], fd, 1e-5);
                }
                if j < M {
                    for i in 0..N {
                        let fd = (xl_up[(i, j)] - xl_down[(i, j)]) / (2.0 * h);
                        assert_close(d2x[(i, j)], fd, 1e-5);
                    }
                }
            }
        }
        // The diagonal: `∂k(x, x)/∂θ` from the zero distance.
        let mut diag = vec![0.0; N];
        compiled
            .grad_diag_points::<Accurate>(p.x.as_ref(), &mut diag, k)
            .expect("diag");
        for (i, value) in diag.iter().enumerate() {
            assert_close(*value, analytic[(i, i)], 1e-12);
        }
    }
}

#[test]
fn weighted_walks_match_one_parameter_at_a_time() {
    let p = problem();
    let compiled = p.spec.spec().compile();
    let n_params = compiled.num_params();
    let table = p.square_table();
    let rect = p.rect_table();
    let inputs = GramInputs::<_, SuppliedSpec>::supplied(p.x.as_ref(), &table);
    let weight = Mat::from_fn(N, N, |i, j| ((i + j) as f64 * 0.3).cos());
    let mut scratch = Mat::zeros(N, N);
    let mut nested = Vec::new();
    let mut fold = Vec::new();
    let mut walk = WeightedWalk {
        inputs,
        scratch: scratch.as_mut(),
        nested: &mut nested,
        kept: &[],
        kept_products: 0,
        fold: &mut fold,
    };
    let mut bufs: Vec<Mat<f64>> = (0..compiled.contraction_buffers())
        .map(|_| Mat::zeros(N, N))
        .collect();
    let mut out = vec![0.0; n_params];
    compiled
        .weighted_grads::<Accurate>(&mut walk, weight.as_ref(), &mut out, &mut bufs)
        .expect("walk");
    let rect_weight = Mat::from_fn(N, M, |i, j| ((i * 2 + j) as f64 * 0.7).sin());
    let views = CrossViews {
        x1: p.x.as_ref(),
        x2: p.xs.as_ref(),
        dist: None,
        slots: &rect as &dyn RectSlots<f64>,
    };
    let mut rect_bufs: Vec<Mat<f64>> = (0..compiled.contraction_buffers())
        .map(|_| Mat::zeros(N, M))
        .collect();
    let mut rect_out = vec![0.0; n_params];
    let mut rect_scratch = Mat::zeros(N, M);
    compiled
        .weighted_cross_grads_views::<Accurate>(
            views,
            rect_weight.as_ref(),
            &mut rect_out,
            &mut rect_bufs,
            rect_scratch.as_mut(),
            &mut [],
            &mut Vec::new(),
        )
        .expect("rect walk");
    let mut diag_out = vec![0.0; n_params];
    compiled
        .weighted_diag_sums::<Accurate>(p.x.as_ref(), &mut diag_out, &mut DiagAccum::new())
        .expect("diag walk");
    for k in 0..n_params {
        let mut d_k = Mat::zeros(N, N);
        let mut s = Mat::zeros(N, N);
        compiled
            .grad_gram::<Accurate>(
                inputs,
                d_k.as_mut(),
                k,
                Triangle::Full,
                s.as_mut(),
                &mut Vec::new(),
            )
            .expect("grad");
        let mut expect = 0.0;
        let mut diag = 0.0;
        for j in 0..N {
            diag += d_k[(j, j)];
            for i in 0..N {
                // The walk reads the weight's lower triangle as symmetric.
                let (r, c) = if i >= j { (i, j) } else { (j, i) };
                expect += weight[(r, c)] * d_k[(i, j)];
            }
        }
        assert_close(out[k], expect, 1e-10);
        assert_close(diag_out[k], diag, 1e-10);
        let mut d_x = Mat::zeros(N, M);
        let mut s = Mat::zeros(N, M);
        compiled
            .grad_cross_views::<Accurate>(views, d_x.as_mut(), k, s.as_mut(), &mut [])
            .expect("grad cross");
        let mut expect = 0.0;
        for j in 0..M {
            for i in 0..N {
                expect += rect_weight[(i, j)] * d_x[(i, j)];
            }
        }
        assert_close(rect_out[k], expect, 1e-10);
    }
}

#[test]
fn f32_reads_the_supplied_ard_tables() {
    let p = problem();
    let spec = p
        .bands
        .kernel(RationalQuadraticArdKernel::new(&[0.7, 1.3], 1.5).expect("leaf"))
        .expect("dims");
    let c64 = spec.spec().compile();
    let c32 = spec.spec().compile_as::<f32>();
    let cache32 =
        ArdSqDiffBuf::new(Mat::<f32>::from_fn(N, 2, |i, j| p.bands_x.0[(i, j)] as f32).as_ref())
            .expect("cache");
    let table32 = SquareTable {
        scalar: Vec::new(),
        ard: vec![ArdSquare::Packed(cache32.view())],
    };
    let x32 = Mat::<f32>::zeros(N, 0);
    let inputs32 = GramInputs::<_, SuppliedSpec>::supplied(x32.as_ref(), &table32);
    let mut out32 = Mat::<f32>::zeros(N, N);
    let mut s32 = Mat::<f32>::zeros(N, N);
    c32.eval_gram::<Accurate>(
        inputs32,
        out32.as_mut(),
        Triangle::Full,
        s32.as_mut(),
        &mut Vec::new(),
    )
    .expect("f32");
    let table = p.square_table();
    let x = Mat::<f64>::zeros(N, 0);
    let inputs = GramInputs::<_, SuppliedSpec>::supplied(x.as_ref(), &table);
    let mut out = Mat::zeros(N, N);
    let mut s = Mat::zeros(N, N);
    c64.eval_gram::<Accurate>(
        inputs,
        out.as_mut(),
        Triangle::Full,
        s.as_mut(),
        &mut Vec::new(),
    )
    .expect("f64");
    for j in 0..N {
        for i in 0..N {
            assert_close(f64::from(out32[(i, j)]), out[(i, j)], 1e-5);
        }
    }
}

#[test]
fn ard_leaves_on_the_diagonal_match_the_gram_diagonal() {
    let bands = ArdDistance::new(2).expect("dims");
    let rq = RationalQuadraticArdKernel::new(&[0.7, 1.3], 0.8).expect("rq");
    let spec = bands.kernel(rq).expect("dims") * ConstantKernel::new(1.5).expect("c")
        + bands
            .kernel(RbfArdKernel::new(&[0.9, 0.4]).expect("ell"))
            .expect("dims");
    let compiled = spec.spec().compile();
    let n_params = compiled.num_params();
    let train = mat_from_cols(&[coords(1, N, 0.0), coords(2, N, 0.0)]);
    let cache = ArdSqDiffBuf::new(train.as_ref()).expect("cache");
    let table = SquareTable {
        scalar: Vec::new(),
        ard: vec![ArdSquare::Packed(cache.view())],
    };
    let x = Mat::<f64>::zeros(N, 0);
    let inputs = GramInputs::<_, SuppliedSpec>::supplied(x.as_ref(), &table);
    let mut full = Mat::zeros(N, N);
    let mut scratch = Mat::zeros(N, N);
    let mut diag = vec![0.0; N];
    for k in 0..n_params {
        compiled
            .grad_gram::<Accurate>(
                inputs,
                full.as_mut(),
                k,
                Triangle::Full,
                scratch.as_mut(),
                &mut Vec::new(),
            )
            .expect("grad");
        compiled
            .grad_diag_points::<Accurate>(x.as_ref(), &mut diag, k)
            .expect("grad diag");
        for (i, value) in diag.iter().enumerate() {
            assert_close(*value, full[(i, i)], 1e-12);
        }
        for l in 0..n_params {
            compiled
                .hess_gram::<Accurate>(
                    inputs,
                    full.as_mut(),
                    (k, l),
                    Triangle::Full,
                    scratch.as_mut(),
                    &mut Vec::new(),
                )
                .expect("hess");
            compiled
                .hess_diag_points::<Accurate>(x.as_ref(), &mut diag, k, l)
                .expect("hess diag");
            for (i, value) in diag.iter().enumerate() {
                assert_close(*value, full[(i, i)], 1e-12);
            }
        }
    }
    let past = n_params;
    assert!(
        compiled
            .grad_diag_points::<Accurate>(x.as_ref(), &mut diag, past)
            .is_err()
    );
}

#[test]
fn a_coordinate_tree_needs_columns_and_a_distance_tree_does_not() {
    use crate::error::GprError;
    use crate::kernel::WhiteKernel;
    let constant = ConstantKernel::new(1.5).expect("c");
    let white = WhiteKernel::new(0.2).expect("white");
    let x = Mat::<f64>::zeros(N, 0);
    let mut out = Mat::<f64>::zeros(N, N);
    let mut scratch = Mat::<f64>::zeros(N, N);
    let mut diag = vec![0.0; N];
    // The leaves and a coordinate tree keep their column check.
    assert_eq!(
        constant.apply_points(x.as_ref(), out.as_mut(), Triangle::Lower),
        Err(GprError::EmptyInput)
    );
    assert_eq!(
        white.apply_points(x.as_ref(), out.as_mut(), Triangle::Lower),
        Err(GprError::EmptyInput)
    );
    let coords = (KernelSpec::from(constant) + KernelSpec::from(white)).compile();
    assert_eq!(
        coords.apply_points::<Accurate>(
            x.as_ref(),
            out.as_mut(),
            Triangle::Lower,
            scratch.as_mut()
        ),
        Err(GprError::EmptyInput)
    );
    assert_eq!(
        coords.fill_diag_points(x.as_ref(), &mut diag),
        Err(GprError::EmptyInput)
    );
    // A tree that reads supplied distances takes none.
    let image = ScalarDistance::new();
    let dist = (image.kernel(RbfKernel::new(1.0).expect("ell")) * constant + white)
        .spec()
        .compile();
    dist.fill_diag_rows(x.as_ref(), &mut diag).expect("diag");
    assert_close(diag[0], 1.5 + 0.2, 1e-12);
}

/// The slot numbers of the supplied leaves of `k`, depth first.
fn leaf_numbers(k: &CompiledKernel<f64, SuppliedSpec>, out: &mut Vec<usize>) {
    match k {
        CompiledKernel::Supplied(leaf) => out.push(leaf.at),
        CompiledKernel::Sum(terms) | CompiledKernel::Product(terms) => {
            for t in terms {
                leaf_numbers(t, out);
            }
        }
        _ => {}
    }
}

/// A compiled leaf numbers its slot among the slots of its shape in the
/// order the tree lists them, the order every supply keeps.
#[test]
fn compiled_leaves_number_their_slots_per_shape_in_slot_order() {
    let (s1, s2) = (ScalarDistance::new(), ScalarDistance::new());
    let a = ArdDistance::new(2).expect("dims");
    let rbf = RbfKernel::new(0.8).expect("ell");
    let ard = a
        .kernel(RbfArdKernel::new(&[0.7, 1.3]).expect("ell"))
        .expect("dims");
    let spec = s2.kernel(rbf) + ard * s1.kernel(rbf) + s2.kernel(rbf);
    let listed: Vec<_> = spec.slots().iter().map(|slot| slot.id()).collect();
    let expected = [
        crate::kernel::DistanceSlot::Scalar(s2).id(),
        crate::kernel::DistanceSlot::Ard(a).id(),
        crate::kernel::DistanceSlot::Scalar(s1).id(),
    ];
    assert_eq!(listed, expected);
    let mut numbers = Vec::new();
    leaf_numbers(&spec.spec().compile(), &mut numbers);
    assert_eq!(numbers, [0, 0, 1, 0]);
}

/// A supply bound for another kernel, which holds fewer slots than the
/// tree numbers, is reported, not read out of range.
#[test]
fn a_supply_without_the_trees_slot_is_an_error() {
    let p = problem();
    let (s1, s2) = (ScalarDistance::new(), ScalarDistance::new());
    let rbf = RbfKernel::new(0.8).expect("ell");
    let spec = s1.kernel(rbf) + s2.kernel(rbf);
    let compiled = spec.spec().compile();
    let table = p.square_table();
    let x = Mat::<f64>::zeros(N, 0);
    let mut out = Mat::<f64>::zeros(N, N);
    let mut scratch = Mat::<f64>::zeros(N, N);
    let got = compiled.eval_gram::<Accurate>(
        GramInputs::<_, SuppliedSpec>::supplied(x.as_ref(), &table),
        out.as_mut(),
        Triangle::Lower,
        scratch.as_mut(),
        &mut Vec::new(),
    );
    assert!(matches!(
        got,
        Err(GprError::UnsupportedKernelOperation { .. })
    ));
}
