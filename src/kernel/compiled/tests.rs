use super::{CompiledKernel, MixedKernelViews};
use crate::kernel::{
    ConstantKernel, KernelSpec, KernelTerm, LinearKernel, MaternArdKernel, MaternKernel, MaternNu,
    PeriodicKernel, RationalQuadraticArdKernel, RationalQuadraticKernel, RbfArdKernel, RbfKernel,
    Triangle, WhiteKernel,
};
use crate::param::Interval;
use faer::{Mat, MatRef, mat};

const TOL: f64 = 1e-9;

fn assert_close(actual: f64, expected: f64) {
    let scale = expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= TOL * scale,
        "actual={actual}, expected={expected}"
    );
}

fn assert_send_sync<T: Send + Sync>() {}

fn rbf(ell: f64) -> KernelSpec {
    KernelSpec::from(RbfKernel::new(ell).expect("valid"))
}

#[derive(Clone, Debug)]
struct RbfAsTerm(RbfKernel);

impl KernelTerm for RbfAsTerm {
    fn num_params(&self) -> usize {
        self.0.num_params()
    }

    fn get_params(&self, out: &mut [f64]) -> Result<(), crate::GprError> {
        self.0.get_params(out)
    }

    fn set_params(&mut self, params: &[f64]) -> Result<(), crate::GprError> {
        self.0.set_params(params)
    }

    fn bounds_into(&self, out: &mut [Interval]) -> Result<(), crate::GprError> {
        if out.len() != 1 {
            return Err(crate::GprError::InvalidHyperparameter {
                reason: format!("expected 1 bound, got {}", out.len()),
            });
        }
        out[0] = self.0.bounds();
        Ok(())
    }

    fn apply(
        &self,
        dist: MatRef<'_, f64>,
        out: faer::MatMut<'_, f64>,
        uplo: Triangle,
    ) -> Result<(), crate::GprError> {
        self.0.apply(dist, out, uplo)
    }

    fn apply_cross(
        &self,
        dist: MatRef<'_, f64>,
        out: faer::MatMut<'_, f64>,
    ) -> Result<(), crate::GprError> {
        self.0.apply_cross(dist, out)
    }

    fn fill_diag(&self, out: &mut [f64]) -> Result<(), crate::GprError> {
        self.0.fill_diag(out);
        Ok(())
    }

    fn grad(
        &self,
        dist: MatRef<'_, f64>,
        d_k: faer::MatMut<'_, f64>,
        param_idx: usize,
        uplo: Triangle,
    ) -> Result<(), crate::GprError> {
        self.0.grad(dist, d_k, param_idx, uplo)
    }

    fn hess(
        &self,
        dist: MatRef<'_, f64>,
        d2_k: faer::MatMut<'_, f64>,
        i: usize,
        j: usize,
        uplo: Triangle,
    ) -> Result<(), crate::GprError> {
        self.0.hess(dist, d2_k, i, j, uplo)
    }

    fn hess_points(
        &self,
        x: MatRef<'_, f64>,
        d2_k: faer::MatMut<'_, f64>,
        i: usize,
        j: usize,
        uplo: Triangle,
    ) -> Result<(), crate::GprError> {
        self.0.hess_from_coords(x, d2_k, i, j, uplo)
    }

    fn clone_box(&self) -> Box<dyn KernelTerm> {
        Box::new(self.clone())
    }
}

fn custom_rbf(ell: f64) -> KernelSpec {
    KernelSpec::custom(RbfAsTerm(RbfKernel::new(ell).expect("valid")))
}

fn sq_dist_1d(x: &[f64]) -> Mat<f64> {
    let n = x.len();
    Mat::from_fn(n, n, |i, j| {
        let d = x[i] - x[j];
        d * d
    })
}

fn fill(n: usize, value: f64) -> Mat<f64> {
    Mat::from_fn(n, n, |_, _| value)
}

fn apply_compiled(compiled: &CompiledKernel, dist: MatRef<'_, f64>) -> Mat<f64> {
    let n = dist.nrows();
    let mut out = fill(n, 0.0);
    let mut scratch = fill(n, 0.0);
    compiled
        .apply(dist, out.as_mut(), Triangle::Full, scratch.as_mut())
        .expect("shape");
    out
}

fn apply_compiled_points(compiled: &CompiledKernel, x: MatRef<'_, f64>) -> Mat<f64> {
    let n = x.nrows();
    let mut out = fill(n, 0.0);
    let mut scratch = fill(n, 0.0);
    compiled
        .apply_points(x, out.as_mut(), Triangle::Full, scratch.as_mut())
        .expect("shape");
    out
}

fn apply_rbf(ell: f64, dist: MatRef<'_, f64>) -> Mat<f64> {
    let n = dist.nrows();
    let mut out = fill(n, 0.0);
    RbfKernel::new(ell)
        .expect("valid")
        .apply(dist, out.as_mut(), Triangle::Full)
        .expect("shape");
    out
}

fn lower_matches(actual: MatRef<'_, f64>, expected: MatRef<'_, f64>) {
    let n = actual.nrows();
    for col in 0..n {
        for row in col..n {
            assert_close(actual[(row, col)], expected[(row, col)]);
        }
    }
}

#[test]
fn is_send_sync() {
    assert_send_sync::<CompiledKernel>();
}

#[test]
fn custom_plus_rbf_apply_matches_two_rbf() {
    let compiled = (custom_rbf(1.0) + rbf(2.0)).compile();
    let builtin = (rbf(1.0) + rbf(2.0)).compile();
    let dist = sq_dist_1d(&[0.0, 1.0, 2.0]);
    let out = apply_compiled(&compiled, dist.as_ref());
    let expected = apply_compiled(&builtin, dist.as_ref());
    for col in 0..3 {
        for row in 0..3 {
            assert_close(out[(row, col)], expected[(row, col)]);
        }
    }
}

#[test]
fn custom_times_rbf_apply_matches_two_rbf() {
    let compiled = (custom_rbf(1.0) * rbf(0.5)).compile();
    let builtin = (rbf(1.0) * rbf(0.5)).compile();
    let dist = sq_dist_1d(&[0.0, 1.2]);
    let out = apply_compiled(&compiled, dist.as_ref());
    let expected = apply_compiled(&builtin, dist.as_ref());
    assert_close(out[(0, 1)], expected[(0, 1)]);
    assert_close(out[(0, 0)], expected[(0, 0)]);
}

#[test]
fn custom_sum_grad_matches_finite_difference() {
    let spec = custom_rbf(1.0) + rbf(2.0);
    let compiled = spec.compile();
    let dist = sq_dist_1d(&[0.0, 0.8, 1.5]);
    let mut params = [0.0; 2];
    spec.get_params(&mut params).expect("len 2");
    let h = 1e-6;
    let mut spec_plus = spec.clone();
    let mut spec_minus = spec.clone();
    params[1] += h;
    spec_plus.set_params(&params).expect("valid");
    params[1] -= 2.0 * h;
    spec_minus.set_params(&params).expect("valid");
    let kp = apply_compiled(&spec_plus.compile(), dist.as_ref());
    let km = apply_compiled(&spec_minus.compile(), dist.as_ref());
    let mut dk = fill(3, 0.0);
    let mut scratch = fill(3, 0.0);
    compiled
        .grad(
            dist.as_ref(),
            dk.as_mut(),
            1,
            Triangle::Full,
            scratch.as_mut(),
        )
        .expect("idx 1");
    for col in 0..3 {
        for row in 0..3 {
            let fd = (kp[(row, col)] - km[(row, col)]) / (2.0 * h);
            assert!(
                (dk[(row, col)] - fd).abs() <= 1e-4 * fd.abs().max(1.0),
                "row={row} col={col} analytic={} fd={fd}",
                dk[(row, col)]
            );
        }
    }
}

#[test]
fn combine_sum_from_leaf_grams_overwrites_dirty_dest() {
    let compiled = (rbf(1.0) + rbf(2.0)).compile();
    let dist = sq_dist_1d(&[0.0, 1.0, 2.0]);
    let expected = apply_compiled(&compiled, dist.as_ref());
    let mut grams = [fill(3, 0.0), fill(3, 0.0)];
    compiled
        .leaf_at(0)
        .expect("leaf 0")
        .apply(
            dist.as_ref(),
            grams[0].as_mut(),
            Triangle::Lower,
            fill(3, 0.0).as_mut(),
        )
        .expect("leaf 0");
    compiled
        .leaf_at(1)
        .expect("leaf 1")
        .apply(
            dist.as_ref(),
            grams[1].as_mut(),
            Triangle::Lower,
            fill(3, 0.0).as_mut(),
        )
        .expect("leaf 1");
    let mut dirty = fill(3, 999.0);
    let mut scratch = fill(3, 0.0);
    compiled
        .combine_from_leaf_grams(&grams, dirty.as_mut(), scratch.as_mut(), Triangle::Lower)
        .expect("combine");
    lower_matches(dirty.as_ref(), expected.as_ref());
}

#[test]
fn sum_apply_adds_leaves() {
    let compiled = (rbf(1.0) + rbf(2.0)).compile();
    let dist = sq_dist_1d(&[0.0, 1.0, 2.0]);
    let out = apply_compiled(&compiled, dist.as_ref());
    let k1 = apply_rbf(1.0, dist.as_ref());
    let k2 = apply_rbf(2.0, dist.as_ref());
    for col in 0..3 {
        for row in 0..3 {
            assert_close(out[(row, col)], k1[(row, col)] + k2[(row, col)]);
        }
    }
}

#[test]
fn product_apply_multiplies_leaves() {
    let compiled = (rbf(1.0) * rbf(0.5)).compile();
    let dist = sq_dist_1d(&[0.0, 1.2]);
    let out = apply_compiled(&compiled, dist.as_ref());
    let k1 = apply_rbf(1.0, dist.as_ref());
    let k2 = apply_rbf(0.5, dist.as_ref());
    assert_close(out[(0, 1)], k1[(0, 1)] * k2[(0, 1)]);
    assert_close(out[(0, 0)], 1.0);
}

#[test]
fn mixed_product_of_sum_matches_leaves() {
    let spec = rbf(1.0) * (rbf(2.0) + rbf(3.0));
    let compiled = spec.compile();
    let dist = sq_dist_1d(&[0.0, 0.7, 1.4]);
    let out = apply_compiled(&compiled, dist.as_ref());
    let k1 = apply_rbf(1.0, dist.as_ref());
    let k2 = apply_rbf(2.0, dist.as_ref());
    let k3 = apply_rbf(3.0, dist.as_ref());
    for col in 0..3 {
        for row in 0..3 {
            assert_close(
                out[(row, col)],
                k1[(row, col)] * (k2[(row, col)] + k3[(row, col)]),
            );
        }
    }
}

#[test]
fn product_of_two_sums_matches_leaves() {
    let spec = (rbf(1.0) + rbf(2.0)) * (rbf(0.5) + rbf(1.5));
    let compiled = spec.compile();
    let dist = sq_dist_1d(&[0.0, 1.0]);
    let out = apply_compiled(&compiled, dist.as_ref());
    let a = apply_rbf(1.0, dist.as_ref());
    let b = apply_rbf(2.0, dist.as_ref());
    let c = apply_rbf(0.5, dist.as_ref());
    let d = apply_rbf(1.5, dist.as_ref());
    for col in 0..2 {
        for row in 0..2 {
            assert_close(
                out[(row, col)],
                (a[(row, col)] + b[(row, col)]) * (c[(row, col)] + d[(row, col)]),
            );
        }
    }
}

#[test]
fn sum_lower_matches_full_and_leaves_upper() {
    let compiled = (rbf(1.0) + rbf(2.0)).compile();
    let dist = sq_dist_1d(&[0.0, 1.0, 2.0]);
    let full = apply_compiled(&compiled, dist.as_ref());
    let sentinel = 42.0;
    let mut lower = fill(3, sentinel);
    let mut scratch = fill(3, 0.0);
    compiled
        .apply(
            dist.as_ref(),
            lower.as_mut(),
            Triangle::Lower,
            scratch.as_mut(),
        )
        .expect("shape");
    lower_matches(lower.as_ref(), full.as_ref());
    assert_close(lower[(0, 1)], sentinel);
    assert_close(lower[(0, 2)], sentinel);
    assert_close(lower[(1, 2)], sentinel);
}

#[test]
fn sum_grad_matches_finite_difference() {
    let spec = rbf(1.0) + rbf(2.0);
    let compiled = spec.compile();
    let dist = sq_dist_1d(&[0.0, 0.8, 1.5]);
    let mut params = [0.0; 2];
    spec.get_params(&mut params).expect("len 2");
    let h = 1e-6;
    let mut spec_plus = spec.clone();
    let mut spec_minus = spec.clone();
    params[1] += h;
    spec_plus.set_params(&params).expect("valid");
    params[1] -= 2.0 * h;
    spec_minus.set_params(&params).expect("valid");
    let kp = apply_compiled(&spec_plus.compile(), dist.as_ref());
    let km = apply_compiled(&spec_minus.compile(), dist.as_ref());
    let mut dk = fill(3, 0.0);
    let mut scratch = fill(3, 0.0);
    compiled
        .grad(
            dist.as_ref(),
            dk.as_mut(),
            1,
            Triangle::Full,
            scratch.as_mut(),
        )
        .expect("idx 1");
    for col in 0..3 {
        for row in 0..3 {
            let fd = (kp[(row, col)] - km[(row, col)]) / (2.0 * h);
            assert_close(dk[(row, col)], fd);
        }
    }
}

#[test]
fn product_hess_matches_finite_difference_of_grad() {
    let spec = rbf(1.25) * KernelSpec::from(ConstantKernel::new(1.4).expect("valid"));
    let compiled = spec.compile();
    let dist = mat![[0.0, 0.64, 2.89], [0.64, 0.0, 0.81], [2.89, 0.81, 0.0]];
    let mut params = [0.0; 2];
    spec.get_params(&mut params).expect("len 2");
    let h = 1e-6;
    for j in 0..2 {
        let mut plus = params;
        let mut minus = params;
        plus[j] += h;
        minus[j] -= h;
        let mut spec_plus = spec.clone();
        let mut spec_minus = spec.clone();
        spec_plus.set_params(&plus).expect("valid");
        spec_minus.set_params(&minus).expect("valid");
        let mut gp = fill(3, 0.0);
        let mut gm = fill(3, 0.0);
        let mut scratch = fill(3, 0.0);
        spec_plus
            .compile()
            .grad(
                dist.as_ref(),
                gp.as_mut(),
                0,
                Triangle::Full,
                scratch.as_mut(),
            )
            .expect("plus");
        spec_minus
            .compile()
            .grad(
                dist.as_ref(),
                gm.as_mut(),
                0,
                Triangle::Full,
                scratch.as_mut(),
            )
            .expect("minus");
        let mut d2 = fill(3, 0.0);
        compiled
            .hess(
                dist.as_ref(),
                d2.as_mut(),
                0,
                j,
                Triangle::Full,
                scratch.as_mut(),
            )
            .expect("hess");
        let fd = (gp[(0, 1)] - gm[(0, 1)]) / (2.0 * h);
        assert_close(d2[(0, 1)], fd);
    }
}

#[test]
fn product_grad_matches_finite_difference() {
    let spec = rbf(1.0) * rbf(2.0);
    let compiled = spec.compile();
    let dist = mat![[0.0, 1.0], [1.0, 0.0]];
    let mut params = [0.0; 2];
    spec.get_params(&mut params).expect("len 2");
    let h = 1e-6;
    let mut spec_plus = spec.clone();
    let mut spec_minus = spec.clone();
    params[0] += h;
    spec_plus.set_params(&params).expect("valid");
    params[0] -= 2.0 * h;
    spec_minus.set_params(&params).expect("valid");
    let kp = apply_compiled(&spec_plus.compile(), dist.as_ref());
    let km = apply_compiled(&spec_minus.compile(), dist.as_ref());
    let mut dk = fill(2, 0.0);
    let mut scratch = fill(2, 0.0);
    compiled
        .grad(
            dist.as_ref(),
            dk.as_mut(),
            0,
            Triangle::Full,
            scratch.as_mut(),
        )
        .expect("idx 0");
    let fd = (kp[(0, 1)] - km[(0, 1)]) / (2.0 * h);
    assert_close(dk[(0, 1)], fd);
}

#[test]
fn nested_product_grad_matches_finite_difference() {
    let spec = rbf(1.0) * (rbf(2.0) + rbf(3.0));
    let compiled = spec.compile();
    let dist = sq_dist_1d(&[0.0, 0.9]);
    let mut params = [0.0; 3];
    spec.get_params(&mut params).expect("len 3");
    let h = 1e-6;
    let mut spec_plus = spec.clone();
    let mut spec_minus = spec.clone();
    params[2] += h;
    spec_plus.set_params(&params).expect("valid");
    params[2] -= 2.0 * h;
    spec_minus.set_params(&params).expect("valid");
    let kp = apply_compiled(&spec_plus.compile(), dist.as_ref());
    let km = apply_compiled(&spec_minus.compile(), dist.as_ref());
    let mut dk = fill(2, 0.0);
    let mut scratch = fill(2, 0.0);
    compiled
        .grad(
            dist.as_ref(),
            dk.as_mut(),
            2,
            Triangle::Full,
            scratch.as_mut(),
        )
        .expect("idx 2");
    let fd = (kp[(0, 1)] - km[(0, 1)]) / (2.0 * h);
    assert_close(dk[(0, 1)], fd);
}

fn assert_points_product_grad_fd(spec: KernelSpec, x: Mat<f64>, param_idx: usize) {
    let compiled = spec.compile();
    let mut params = vec![0.0; spec.num_params()];
    spec.get_params(&mut params).expect("len");
    let h = 1e-6;
    let mut spec_plus = spec.clone();
    let mut spec_minus = spec.clone();
    params[param_idx] += h;
    spec_plus.set_params(&params).expect("valid");
    params[param_idx] -= 2.0 * h;
    spec_minus.set_params(&params).expect("valid");
    let kp = apply_compiled_points(&spec_plus.compile(), x.as_ref());
    let km = apply_compiled_points(&spec_minus.compile(), x.as_ref());
    let mut dk = fill(x.nrows(), 0.0);
    let mut scratch = fill(x.nrows(), 0.0);
    compiled
        .grad_points(
            x.as_ref(),
            dk.as_mut(),
            param_idx,
            Triangle::Full,
            scratch.as_mut(),
        )
        .expect("grad");
    let fd = (kp[(0, 1)] - km[(0, 1)]) / (2.0 * h);
    assert_close(dk[(0, 1)], fd);
}

#[test]
fn linear_times_linear_grad_matches_finite_difference() {
    let spec = KernelSpec::from(LinearKernel::new(1.0).expect("valid"))
        * KernelSpec::from(LinearKernel::new(0.5).expect("valid"));
    let x = Mat::from_fn(2, 1, |i, _| 0.5 + i as f64);
    assert_points_product_grad_fd(spec, x, 0);
}

#[test]
fn linear_times_constant_grad_matches_finite_difference() {
    let spec = KernelSpec::from(LinearKernel::new(1.0).expect("valid"))
        * KernelSpec::from(ConstantKernel::new(1.5).expect("valid"));
    let x = Mat::from_fn(2, 1, |i, _| 0.5 + i as f64);
    assert_points_product_grad_fd(spec, x, 1);
}

#[test]
fn linear_times_ard_grad_matches_finite_difference() {
    let spec = KernelSpec::from(LinearKernel::new(1.0).expect("valid"))
        * KernelSpec::from(RbfArdKernel::new(&[1.2, 0.8]).expect("valid"));
    let x = points_2d(&[[0.0, 0.0], [1.0, 0.4], [0.2, 1.1]]);
    assert_points_product_grad_fd(spec, x, 2);
}

#[test]
fn nested_points_product_grad_matches_finite_difference() {
    let spec = KernelSpec::from(LinearKernel::new(1.0).expect("valid"))
        * (KernelSpec::from(LinearKernel::new(0.8).expect("valid"))
            + KernelSpec::from(ConstantKernel::new(0.5).expect("valid")));
    let x = Mat::from_fn(2, 1, |i, _| 0.5 + i as f64);
    assert_points_product_grad_fd(spec, x, 2);
}

#[test]
fn get_set_params_roundtrip_and_atomic() {
    let mut compiled = (rbf(1.0) + rbf(2.0)).compile();
    let mut params = [0.0; 2];
    compiled.get_params(&mut params).expect("len 2");
    assert_close(params[0], 1.0_f64.ln());
    assert_close(params[1], 2.0_f64.ln());
    params[0] = 0.5_f64.ln();
    compiled.set_params(&params).expect("valid");
    compiled.get_params(&mut params).expect("len 2");
    assert_close(params[0], 0.5_f64.ln());
    let before = compiled.clone();
    assert!(compiled.set_params(&[0.0, f64::INFINITY]).is_err());
    assert_eq!(compiled, before);
}

#[test]
fn rejects_bad_index_and_scratch_shape() {
    let compiled = (rbf(1.0) + rbf(2.0)).compile();
    let dist = sq_dist_1d(&[0.0, 1.0]);
    let mut dk = fill(2, 0.0);
    let mut scratch = fill(2, 0.0);
    assert!(matches!(
        compiled.grad(
            dist.as_ref(),
            dk.as_mut(),
            2,
            Triangle::Lower,
            scratch.as_mut()
        ),
        Err(crate::error::GprError::InvalidHyperparameter { .. })
    ));
    let mut small = fill(1, 0.0);
    assert!(matches!(
        compiled.apply(dist.as_ref(), dk.as_mut(), Triangle::Full, small.as_mut()),
        Err(crate::error::GprError::WorkspaceTooSmall)
    ));
}

#[test]
fn empty_sum_is_unsupported() {
    let compiled = CompiledKernel::Sum(Vec::new());
    let dist = sq_dist_1d(&[0.0, 1.0]);
    let mut out = fill(2, 0.0);
    let mut scratch = fill(2, 0.0);
    assert!(matches!(
        compiled.apply(
            dist.as_ref(),
            out.as_mut(),
            Triangle::Full,
            scratch.as_mut()
        ),
        Err(crate::error::GprError::UnsupportedKernelOperation { .. })
    ));
}

#[test]
fn fill_diag_adds_rbf_leaves() {
    let compiled = (rbf(1.0) + rbf(2.0)).compile();
    let mut diag = [0.0, 0.0];
    compiled.fill_diag(&mut diag).expect("two terms");
    assert_close(diag[0], 2.0);
    assert_close(diag[1], 2.0);
}

#[test]
fn apply_cross_matches_full_block() {
    let compiled = rbf(1.0).compile();
    let train = sq_dist_1d(&[0.0, 1.0]);
    let mut k_nn = fill(2, 0.0);
    let mut scratch = fill(2, 0.0);
    compiled
        .apply(
            train.as_ref(),
            k_nn.as_mut(),
            Triangle::Full,
            scratch.as_mut(),
        )
        .expect("square");
    let dist_cross = faer::mat![[0.0, 1.0], [1.0, 0.0]];
    let mut k_cross = fill(2, 0.0);
    let mut scratch_cross = fill(2, 0.0);
    compiled
        .apply_cross(
            dist_cross.as_ref(),
            k_cross.as_mut(),
            scratch_cross.as_mut(),
        )
        .expect("rect");
    assert_close(k_cross[(0, 0)], k_nn[(0, 0)]);
    assert_close(k_cross[(0, 1)], k_nn[(0, 1)]);
}

fn points_2d(rows: &[[f64; 2]]) -> Mat<f64> {
    Mat::from_fn(rows.len(), 2, |i, j| rows[i][j])
}

#[test]
fn ard_apply_dist_is_unsupported_points_match_isotropic() {
    let ell = 1.3;
    let compiled = KernelSpec::from(RbfArdKernel::new(&[ell, ell]).expect("valid")).compile();
    let x = points_2d(&[[0.0, 0.0], [1.0, 0.4], [0.2, 1.1]]);
    let dist = {
        let n = x.nrows();
        Mat::from_fn(n, n, |row, col| {
            let mut sum = 0.0;
            for dim in 0..2 {
                let diff = x[(row, dim)] - x[(col, dim)];
                sum += diff * diff;
            }
            sum
        })
    };
    let mut out = fill(3, 0.0);
    let mut scratch = fill(3, 0.0);
    assert!(matches!(
        compiled.apply(
            dist.as_ref(),
            out.as_mut(),
            Triangle::Full,
            scratch.as_mut()
        ),
        Err(crate::error::GprError::UnsupportedKernelOperation { .. })
    ));
    compiled
        .apply_points(x.as_ref(), out.as_mut(), Triangle::Full, scratch.as_mut())
        .expect("points");
    let iso = apply_rbf(ell, dist.as_ref());
    for col in 0..3 {
        for row in 0..3 {
            assert_close(out[(row, col)], iso[(row, col)]);
        }
    }
}

#[test]
fn custom_plus_linear_is_mixed() {
    let spec = custom_rbf(1.0) + KernelSpec::from(LinearKernel::new(1.0).expect("valid"));
    let compiled = spec.compile();
    assert_eq!(compiled.coord_mode().expect("mix"), super::CoordMode::Mixed);
}

#[test]
fn mixed_isotropic_and_ard_is_mixed() {
    let spec = rbf(1.0) + KernelSpec::from(RbfArdKernel::new(&[1.0, 2.0]).expect("valid"));
    let compiled = spec.compile();
    assert_eq!(compiled.coord_mode().expect("mix"), super::CoordMode::Mixed);
}

#[test]
fn rbf_plus_linear_apply_adds_leaves() {
    let spec = rbf(1.0) + KernelSpec::from(LinearKernel::new(1.0).expect("valid"));
    let compiled = spec.compile();
    assert_eq!(compiled.coord_mode().expect("mix"), super::CoordMode::Mixed);
    let xs = [0.5, 1.5];
    let x = Mat::from_fn(2, 1, |i, _| xs[i]);
    let dist = sq_dist_1d(&xs);
    let mut out = fill(2, 0.0);
    let mut scratch = fill(2, 0.0);
    compiled
        .apply_mixed(
            MixedKernelViews::new(dist.as_ref(), x.as_ref()),
            out.as_mut(),
            Triangle::Full,
            scratch.as_mut(),
        )
        .expect("mix");
    let kr = apply_compiled(&rbf(1.0).compile(), dist.as_ref());
    let mut kl = fill(2, 0.0);
    LinearKernel::new(1.0)
        .expect("valid")
        .apply(x.as_ref(), kl.as_mut(), Triangle::Full)
        .expect("linear");
    for col in 0..2 {
        for row in 0..2 {
            assert_close(out[(row, col)], kr[(row, col)] + kl[(row, col)]);
        }
    }
}

#[test]
fn rbf_times_linear_apply_multiplies_leaves() {
    let spec = rbf(1.0) * KernelSpec::from(LinearKernel::new(1.0).expect("valid"));
    let compiled = spec.compile();
    assert_eq!(compiled.coord_mode().expect("mix"), super::CoordMode::Mixed);
    let xs = [0.5, 1.5];
    let x = Mat::from_fn(2, 1, |i, _| xs[i]);
    let dist = sq_dist_1d(&xs);
    let mut out = fill(2, 0.0);
    let mut scratch = fill(2, 0.0);
    compiled
        .apply_mixed(
            MixedKernelViews::new(dist.as_ref(), x.as_ref()),
            out.as_mut(),
            Triangle::Full,
            scratch.as_mut(),
        )
        .expect("mix");
    let kr = apply_compiled(&rbf(1.0).compile(), dist.as_ref());
    let mut kl = fill(2, 0.0);
    LinearKernel::new(1.0)
        .expect("valid")
        .apply(x.as_ref(), kl.as_mut(), Triangle::Full)
        .expect("linear");
    for col in 0..2 {
        for row in 0..2 {
            assert_close(out[(row, col)], kr[(row, col)] * kl[(row, col)]);
        }
    }
}

#[test]
fn rbf_plus_linear_grad_matches_finite_difference() {
    let spec = rbf(1.0) + KernelSpec::from(LinearKernel::new(0.8).expect("valid"));
    let compiled = spec.compile();
    let xs = [0.5, 1.5];
    let x = Mat::from_fn(2, 1, |i, _| xs[i]);
    let dist = sq_dist_1d(&xs);
    let mut params = [0.0; 2];
    spec.get_params(&mut params).expect("len 2");
    let h = 1e-6;
    let mut spec_plus = spec.clone();
    let mut spec_minus = spec.clone();
    params[1] += h;
    spec_plus.set_params(&params).expect("valid");
    params[1] -= 2.0 * h;
    spec_minus.set_params(&params).expect("valid");
    let mut kp = fill(2, 0.0);
    let mut km = fill(2, 0.0);
    let mut scratch = fill(2, 0.0);
    spec_plus
        .compile()
        .apply_mixed(
            MixedKernelViews::new(dist.as_ref(), x.as_ref()),
            kp.as_mut(),
            Triangle::Full,
            scratch.as_mut(),
        )
        .expect("plus");
    spec_minus
        .compile()
        .apply_mixed(
            MixedKernelViews::new(dist.as_ref(), x.as_ref()),
            km.as_mut(),
            Triangle::Full,
            scratch.as_mut(),
        )
        .expect("minus");
    let mut dk = fill(2, 0.0);
    compiled
        .grad_mixed(
            MixedKernelViews::new(dist.as_ref(), x.as_ref()),
            dk.as_mut(),
            1,
            Triangle::Full,
            scratch.as_mut(),
        )
        .expect("grad");
    let fd = (kp[(0, 1)] - km[(0, 1)]) / (2.0 * h);
    assert_close(dk[(0, 1)], fd);
}

#[test]
fn rbf_plus_white_is_distance_mode() {
    let spec = rbf(1.0) + KernelSpec::from(WhiteKernel::new(0.1).expect("valid"));
    let compiled = spec.compile();
    assert_eq!(
        compiled.coord_mode().expect("compat"),
        super::CoordMode::Dist
    );
    let dist = sq_dist_1d(&[0.0, 1.0]);
    let mut out = fill(2, 0.0);
    let mut scratch = fill(2, 0.0);
    compiled
        .apply(
            dist.as_ref(),
            out.as_mut(),
            Triangle::Full,
            scratch.as_mut(),
        )
        .expect("shape");
    assert_close(out[(0, 0)], 1.0 + 0.1);
    assert_close(out[(0, 1)], apply_rbf(1.0, dist.as_ref())[(0, 1)]);
}

#[test]
fn linear_plus_constant_is_points_mode() {
    let spec = KernelSpec::from(LinearKernel::new(1.0).expect("valid"))
        + KernelSpec::from(ConstantKernel::new(0.5).expect("valid"));
    let compiled = spec.compile();
    assert_eq!(
        compiled.coord_mode().expect("compat"),
        super::CoordMode::Points
    );
    let x = Mat::from_fn(2, 1, |i, _| i as f64);
    let mut out = fill(2, 0.0);
    let mut scratch = fill(2, 0.0);
    compiled
        .apply_points(x.as_ref(), out.as_mut(), Triangle::Full, scratch.as_mut())
        .expect("points");
    assert_close(out[(0, 0)], 0.5);
    assert_close(out[(1, 1)], 1.0 + 0.5);
    assert_close(out[(1, 0)], 0.5);
}

#[test]
fn matern_plus_white_is_distance_mode() {
    let spec = KernelSpec::from(MaternKernel::new(1.0, MaternNu::ThreeHalves).expect("valid"))
        + KernelSpec::from(WhiteKernel::new(0.1).expect("valid"));
    let compiled = spec.compile();
    assert_eq!(
        compiled.coord_mode().expect("compat"),
        super::CoordMode::Dist
    );
    let dist = sq_dist_1d(&[0.0, 1.0]);
    let mut out = fill(2, 0.0);
    let mut scratch = fill(2, 0.0);
    compiled
        .apply(
            dist.as_ref(),
            out.as_mut(),
            Triangle::Full,
            scratch.as_mut(),
        )
        .expect("shape");
    let rho = 3.0_f64.sqrt();
    let k01 = (1.0 + rho) * (-rho).exp();
    assert_close(out[(0, 0)], 1.0 + 0.1);
    assert_close(out[(0, 1)], k01);
}

#[test]
fn matern_ard_plus_constant_is_points_mode() {
    let spec = KernelSpec::from(MaternArdKernel::new(&[1.0], MaternNu::Half).expect("valid"))
        + KernelSpec::from(ConstantKernel::new(0.5).expect("valid"));
    let compiled = spec.compile();
    assert_eq!(
        compiled.coord_mode().expect("compat"),
        super::CoordMode::Points
    );
    let x = Mat::from_fn(2, 1, |i, _| i as f64);
    let mut out = fill(2, 0.0);
    let mut scratch = fill(2, 0.0);
    compiled
        .apply_points(x.as_ref(), out.as_mut(), Triangle::Full, scratch.as_mut())
        .expect("points");
    assert_close(out[(0, 0)], 1.5);
    assert_close(out[(1, 1)], 1.5);
    assert_close(out[(1, 0)], (-1.0_f64).exp() + 0.5);
}

#[test]
fn periodic_plus_white_is_distance_mode() {
    let spec = KernelSpec::from(PeriodicKernel::new(1.0, 2.0).expect("valid"))
        + KernelSpec::from(WhiteKernel::new(0.1).expect("valid"));
    let compiled = spec.compile();
    assert_eq!(
        compiled.coord_mode().expect("compat"),
        super::CoordMode::Dist
    );
    let dist = sq_dist_1d(&[0.0, 1.0]);
    let mut out = fill(2, 0.0);
    let mut scratch = fill(2, 0.0);
    compiled
        .apply(
            dist.as_ref(),
            out.as_mut(),
            Triangle::Full,
            scratch.as_mut(),
        )
        .expect("shape");
    let s = (std::f64::consts::PI * 0.5).sin();
    let k01 = (-2.0 * s * s).exp();
    assert_close(out[(0, 0)], 1.0 + 0.1);
    assert_close(out[(0, 1)], k01);
}

#[test]
fn rational_quadratic_plus_white_is_distance_mode() {
    let spec = KernelSpec::from(RationalQuadraticKernel::new(1.0, 1.0).expect("valid"))
        + KernelSpec::from(WhiteKernel::new(0.1).expect("valid"));
    let compiled = spec.compile();
    assert_eq!(
        compiled.coord_mode().expect("compat"),
        super::CoordMode::Dist
    );
    let dist = sq_dist_1d(&[0.0, 1.0]);
    let mut out = fill(2, 0.0);
    let mut scratch = fill(2, 0.0);
    compiled
        .apply(
            dist.as_ref(),
            out.as_mut(),
            Triangle::Full,
            scratch.as_mut(),
        )
        .expect("shape");
    assert_close(out[(0, 0)], 1.1);
    assert_close(out[(0, 1)], 2.0 / 3.0);
}

#[test]
fn rational_quadratic_ard_plus_constant_is_points_mode() {
    let spec = KernelSpec::from(RationalQuadraticArdKernel::new(&[1.0], 1.0).expect("valid"))
        + KernelSpec::from(ConstantKernel::new(0.5).expect("valid"));
    let compiled = spec.compile();
    assert_eq!(
        compiled.coord_mode().expect("compat"),
        super::CoordMode::Points
    );
    let x = Mat::from_fn(2, 1, |i, _| i as f64);
    let mut out = fill(2, 0.0);
    let mut scratch = fill(2, 0.0);
    compiled
        .apply_points(x.as_ref(), out.as_mut(), Triangle::Full, scratch.as_mut())
        .expect("points");
    assert_close(out[(0, 0)], 1.5);
    assert_close(out[(1, 1)], 1.5);
    assert_close(out[(1, 0)], 2.0 / 3.0 + 0.5);
}
