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
        self.0
            .hess_from_coords::<crate::math::Accurate>(x, d2_k, i, j, uplo)
    }

    fn clone_box(&self) -> Box<dyn KernelTerm> {
        Box::new(self.clone())
    }
}

impl crate::kernel::KernelTerm<f32> for RbfAsTerm {
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
        dist: MatRef<'_, f32>,
        mut out: faer::MatMut<'_, f32>,
        uplo: Triangle,
    ) -> Result<(), crate::GprError> {
        let ell = self.0.lengthscale() as f32;
        let inv_two = 0.5 / (ell * ell);
        write_f32_triangle(dist, out.as_mut(), uplo, |d| Ok((-d * inv_two).exp()))
    }

    fn apply_cross(
        &self,
        dist: MatRef<'_, f32>,
        mut out: faer::MatMut<'_, f32>,
    ) -> Result<(), crate::GprError> {
        let ell = self.0.lengthscale() as f32;
        let inv_two = 0.5 / (ell * ell);
        for col in 0..dist.ncols() {
            for row in 0..dist.nrows() {
                out[(row, col)] = (-dist[(row, col)] * inv_two).exp();
            }
        }
        Ok(())
    }

    fn fill_diag(&self, out: &mut [f32]) -> Result<(), crate::GprError> {
        out.fill(1.0);
        Ok(())
    }

    fn grad(
        &self,
        dist: MatRef<'_, f32>,
        mut d_k: faer::MatMut<'_, f32>,
        param_idx: usize,
        uplo: Triangle,
    ) -> Result<(), crate::GprError> {
        if param_idx != 0 {
            return Err(crate::GprError::InvalidHyperparameter {
                reason: "RBF term has a single parameter at index 0".to_owned(),
            });
        }
        let ell = self.0.lengthscale() as f32;
        let inv_two = 0.5 / (ell * ell);
        let inv_ell_sq = 1.0 / (ell * ell);
        write_f32_triangle(dist, d_k.as_mut(), uplo, |d| {
            let k = (-d * inv_two).exp();
            Ok(k * d * inv_ell_sq)
        })
    }

    fn hess(
        &self,
        dist: MatRef<'_, f32>,
        mut d2_k: faer::MatMut<'_, f32>,
        i: usize,
        j: usize,
        uplo: Triangle,
    ) -> Result<(), crate::GprError> {
        if i != 0 || j != 0 {
            return Err(crate::GprError::InvalidHyperparameter {
                reason: format!("RBF term has a single parameter; got pair ({i}, {j})"),
            });
        }
        let ell = self.0.lengthscale() as f32;
        let inv_two = 0.5 / (ell * ell);
        let inv_ell_sq = 1.0 / (ell * ell);
        write_f32_triangle(dist, d2_k.as_mut(), uplo, |d| {
            let k = (-d * inv_two).exp();
            let u = d * inv_ell_sq;
            Ok(k * u * (u - 2.0))
        })
    }

    fn hess_points(
        &self,
        x: MatRef<'_, f32>,
        d2_k: faer::MatMut<'_, f32>,
        i: usize,
        j: usize,
        uplo: Triangle,
    ) -> Result<(), crate::GprError> {
        let n = x.nrows();
        let mut dist = Mat::zeros(n, n);
        for col in 0..n {
            for row in 0..n {
                let mut s = 0.0_f32;
                for dim in 0..x.ncols() {
                    let delta = x[(row, dim)] - x[(col, dim)];
                    s += delta * delta;
                }
                dist[(row, col)] = s;
            }
        }
        self.hess(dist.as_ref(), d2_k, i, j, uplo)
    }

    fn clone_box(&self) -> Box<dyn crate::kernel::KernelTerm<f32>> {
        Box::new(self.clone())
    }
}

fn write_f32_triangle(
    dist: MatRef<'_, f32>,
    mut out: faer::MatMut<'_, f32>,
    uplo: Triangle,
    mut f: impl FnMut(f32) -> Result<f32, crate::GprError>,
) -> Result<(), crate::GprError> {
    let n = dist.nrows();
    for col in 0..n {
        let start = match uplo {
            Triangle::Lower => col,
            Triangle::Upper | Triangle::Full => 0,
        };
        let end = match uplo {
            Triangle::Upper => col + 1,
            Triangle::Lower | Triangle::Full => n,
        };
        for row in start..end {
            out[(row, col)] = f(dist[(row, col)])?;
        }
    }
    Ok(())
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
        .apply::<crate::math::Accurate>(dist, out.as_mut(), Triangle::Full, scratch.as_mut())
        .expect("shape");
    out
}

fn apply_compiled_points(compiled: &CompiledKernel, x: MatRef<'_, f64>) -> Mat<f64> {
    let n = x.nrows();
    let mut out = fill(n, 0.0);
    let mut scratch = fill(n, 0.0);
    compiled
        .apply_points::<crate::math::Accurate>(x, out.as_mut(), Triangle::Full, scratch.as_mut())
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
        .grad::<crate::math::Accurate>(
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
        .apply::<crate::math::Accurate>(
            dist.as_ref(),
            grams[0].as_mut(),
            Triangle::Lower,
            fill(3, 0.0).as_mut(),
        )
        .expect("leaf 0");
    compiled
        .leaf_at(1)
        .expect("leaf 1")
        .apply::<crate::math::Accurate>(
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
        .apply::<crate::math::Accurate>(
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
        .grad::<crate::math::Accurate>(
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
            .grad::<crate::math::Accurate>(
                dist.as_ref(),
                gp.as_mut(),
                0,
                Triangle::Full,
                scratch.as_mut(),
            )
            .expect("plus");
        spec_minus
            .compile()
            .grad::<crate::math::Accurate>(
                dist.as_ref(),
                gm.as_mut(),
                0,
                Triangle::Full,
                scratch.as_mut(),
            )
            .expect("minus");
        let mut d2 = fill(3, 0.0);
        compiled
            .hess::<crate::math::Accurate>(
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
        .grad::<crate::math::Accurate>(
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
        .grad::<crate::math::Accurate>(
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
        .grad_points::<crate::math::Accurate>(
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
        compiled.grad::<crate::math::Accurate>(
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
        compiled.apply::<crate::math::Accurate>(
            dist.as_ref(),
            dk.as_mut(),
            Triangle::Full,
            small.as_mut()
        ),
        Err(crate::error::GprError::WorkspaceTooSmall)
    ));
}

#[test]
fn empty_sum_is_unsupported() {
    let compiled = CompiledKernel::<f64>::Sum(Vec::new());
    let dist = sq_dist_1d(&[0.0, 1.0]);
    let mut out = fill(2, 0.0);
    let mut scratch = fill(2, 0.0);
    assert!(matches!(
        compiled.apply::<crate::math::Accurate>(
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
        .apply::<crate::math::Accurate>(
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
        .apply_cross::<crate::math::Accurate>(
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
        compiled.apply::<crate::math::Accurate>(
            dist.as_ref(),
            out.as_mut(),
            Triangle::Full,
            scratch.as_mut()
        ),
        Err(crate::error::GprError::UnsupportedKernelOperation { .. })
    ));
    compiled
        .apply_points::<crate::math::Accurate>(
            x.as_ref(),
            out.as_mut(),
            Triangle::Full,
            scratch.as_mut(),
        )
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
        .apply_mixed::<crate::math::Accurate>(
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
        .apply_mixed::<crate::math::Accurate>(
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
        .apply_mixed::<crate::math::Accurate>(
            MixedKernelViews::new(dist.as_ref(), x.as_ref()),
            kp.as_mut(),
            Triangle::Full,
            scratch.as_mut(),
        )
        .expect("plus");
    spec_minus
        .compile()
        .apply_mixed::<crate::math::Accurate>(
            MixedKernelViews::new(dist.as_ref(), x.as_ref()),
            km.as_mut(),
            Triangle::Full,
            scratch.as_mut(),
        )
        .expect("minus");
    let mut dk = fill(2, 0.0);
    compiled
        .grad_mixed::<crate::math::Accurate>(
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
        .apply::<crate::math::Accurate>(
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
        .apply_points::<crate::math::Accurate>(
            x.as_ref(),
            out.as_mut(),
            Triangle::Full,
            scratch.as_mut(),
        )
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
        .apply::<crate::math::Accurate>(
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
        .apply_points::<crate::math::Accurate>(
            x.as_ref(),
            out.as_mut(),
            Triangle::Full,
            scratch.as_mut(),
        )
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
        .apply::<crate::math::Accurate>(
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
        .apply::<crate::math::Accurate>(
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
        .apply_points::<crate::math::Accurate>(
            x.as_ref(),
            out.as_mut(),
            Triangle::Full,
            scratch.as_mut(),
        )
        .expect("points");
    assert_close(out[(0, 0)], 1.5);
    assert_close(out[(1, 1)], 1.5);
    assert_close(out[(1, 0)], 2.0 / 3.0 + 0.5);
}

fn fast_exp(x: f64) -> f64 {
    <crate::math::FastApprox as crate::math::KernelMath>::exp_f64(x)
}

fn apply_fast(spec: &KernelSpec, x: MatRef<'_, f64>, out: &mut Mat<f64>) {
    let compiled = spec.compile();
    let mut scratch = Mat::zeros(out.nrows(), out.ncols());
    compiled
        .apply_points::<crate::math::FastApprox>(x, out.as_mut(), Triangle::Full, scratch.as_mut())
        .expect("fast apply");
}

#[test]
fn fast_rbf_matches_polynomial_and_grad_fd() {
    let ell = 1.3;
    let spec = rbf(ell);
    let x = Mat::from_fn(3, 1, |i, _| [0.0, 0.7, 1.6][i]);
    let mut k = fill(3, 0.0);
    apply_fast(&spec, x.as_ref(), &mut k);
    let inv = 1.0 / (ell * ell);
    for col in 0..3 {
        for row in 0..3 {
            let d = x[(row, 0)] - x[(col, 0)];
            assert_close(k[(row, col)], fast_exp(-0.5 * d * d * inv));
        }
    }
    let h = 1e-6;
    let mut plus = spec.clone();
    let mut minus = spec.clone();
    let mut theta = vec![0.0; spec.num_params()];
    spec.get_params(&mut theta).expect("theta");
    plus.set_params(&[theta[0] + h]).expect("plus");
    minus.set_params(&[theta[0] - h]).expect("minus");
    let mut k_plus = fill(3, 0.0);
    let mut k_minus = fill(3, 0.0);
    apply_fast(&plus, x.as_ref(), &mut k_plus);
    apply_fast(&minus, x.as_ref(), &mut k_minus);
    let compiled = spec.compile();
    let mut dk = fill(3, 0.0);
    let mut scratch = fill(3, 0.0);
    compiled
        .grad_points::<crate::math::FastApprox>(
            x.as_ref(),
            dk.as_mut(),
            0,
            Triangle::Full,
            scratch.as_mut(),
        )
        .expect("grad");
    for col in 0..3 {
        for row in 0..3 {
            let fd = (k_plus[(row, col)] - k_minus[(row, col)]) / (2.0 * h);
            assert_close(dk[(row, col)], fd);
        }
    }
}

fn fd_fast_grad_and_hess(label: &str, spec: &KernelSpec, x: MatRef<'_, f64>) {
    let n = x.nrows();
    let p = spec.num_params();
    let mut theta = vec![0.0; p];
    spec.get_params(&mut theta).expect("theta");
    let compiled = spec.compile();
    let h = 1e-5;
    let tol = 1e-4;
    for i in 0..p {
        let mut plus = spec.clone();
        let mut minus = spec.clone();
        let mut tp = theta.clone();
        let mut tm = theta.clone();
        tp[i] += h;
        tm[i] -= h;
        plus.set_params(&tp).expect("plus");
        minus.set_params(&tm).expect("minus");
        let mut k_plus = fill(n, 0.0);
        let mut k_minus = fill(n, 0.0);
        apply_fast(&plus, x, &mut k_plus);
        apply_fast(&minus, x, &mut k_minus);
        let mut dk = fill(n, 0.0);
        let mut scratch = fill(n, 0.0);
        compiled
            .grad_points::<crate::math::FastApprox>(
                x,
                dk.as_mut(),
                i,
                Triangle::Full,
                scratch.as_mut(),
            )
            .expect("grad");
        for col in 0..n {
            for row in 0..n {
                let fd = (k_plus[(row, col)] - k_minus[(row, col)]) / (2.0 * h);
                let scale = fd.abs().max(1.0);
                assert!(
                    (dk[(row, col)] - fd).abs() <= tol * scale,
                    "{label} grad i={i} ({row},{col}) analytic={} fd={fd}",
                    dk[(row, col)]
                );
            }
        }
        for j in 0..=i {
            let bump = |di: f64, dj: f64| {
                let mut shifted = spec.clone();
                let mut t = theta.clone();
                t[i] += di;
                t[j] += dj;
                shifted.set_params(&t).expect("shift");
                let mut k = fill(n, 0.0);
                apply_fast(&shifted, x, &mut k);
                k
            };
            let k_pp = bump(h, h);
            let k_pm = bump(h, -h);
            let k_mp = bump(-h, h);
            let k_mm = bump(-h, -h);
            let mut d2 = fill(n, 0.0);
            let mut scratch = fill(n, 0.0);
            compiled
                .hess_points::<crate::math::FastApprox>(
                    x,
                    d2.as_mut(),
                    i,
                    j,
                    Triangle::Full,
                    scratch.as_mut(),
                )
                .expect("hess");
            for col in 0..n {
                for row in 0..n {
                    let fd = (k_pp[(row, col)] - k_pm[(row, col)] - k_mp[(row, col)]
                        + k_mm[(row, col)])
                        / (4.0 * h * h);
                    let scale = fd.abs().max(1.0);
                    assert!(
                        (d2[(row, col)] - fd).abs() <= tol * scale,
                        "{label} hess i={i} j={j} ({row},{col}) analytic={} fd={fd}",
                        d2[(row, col)]
                    );
                }
            }
        }
    }
}

#[test]
fn fast_exp_leaves_match_polynomial_derivatives() {
    let iso = Mat::from_fn(3, 1, |i, _| [0.0, 0.4, 1.1][i]);
    let ard = Mat::from_fn(3, 2, |row, col| {
        [[0.0, 0.2], [0.5, -0.3], [1.1, 0.7]][row][col]
    });
    for nu in [MaternNu::Half, MaternNu::ThreeHalves, MaternNu::FiveHalves] {
        fd_fast_grad_and_hess(
            &format!("matern {nu:?}"),
            &KernelSpec::from(MaternKernel::new(1.1, nu).expect("matern")),
            iso.as_ref(),
        );
        fd_fast_grad_and_hess(
            &format!("matern-ard {nu:?}"),
            &KernelSpec::from(MaternArdKernel::new(&[0.9, 1.4], nu).expect("matern ard")),
            ard.as_ref(),
        );
    }
    fd_fast_grad_and_hess(
        "periodic",
        &KernelSpec::from(PeriodicKernel::new(1.2, 0.7).expect("periodic")),
        iso.as_ref(),
    );
    fd_fast_grad_and_hess(
        "rbf-ard",
        &KernelSpec::from(RbfArdKernel::new(&[1.2, 0.8]).expect("rbf ard")),
        ard.as_ref(),
    );
    fd_fast_grad_and_hess("rbf", &rbf(1.3), iso.as_ref());
}

#[test]
fn fast_approx_leaves_non_exp_kernels_unchanged() {
    let x = Mat::from_fn(2, 1, |i, _| i as f64);
    for spec in [
        KernelSpec::from(RationalQuadraticKernel::new(1.0, 1.0).expect("rq")),
        KernelSpec::from(LinearKernel::new(1.1).expect("linear")),
        KernelSpec::from(ConstantKernel::new(0.4).expect("constant")),
        KernelSpec::from(WhiteKernel::new(0.2).expect("white")),
    ] {
        let mut accurate = fill(2, 0.0);
        let mut fast = fill(2, 0.0);
        let compiled = spec.compile();
        let mut scratch = fill(2, 0.0);
        compiled
            .apply_points::<crate::math::Accurate>(
                x.as_ref(),
                accurate.as_mut(),
                Triangle::Full,
                scratch.as_mut(),
            )
            .expect("accurate");
        apply_fast(&spec, x.as_ref(), &mut fast);
        for col in 0..2 {
            for row in 0..2 {
                assert_eq!(accurate[(row, col)].to_bits(), fast[(row, col)].to_bits());
            }
        }
    }
}
