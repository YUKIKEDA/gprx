mod compose;
mod coord_mode;
mod custom;
mod fast_math;
mod params;
mod scalar;

use super::{CompiledKernel, MixedKernelViews};
use crate::kernel::{
    ConstantKernel, KernelSpec, KernelTerm, LinearKernel, MaternArdKernel, MaternKernel, MaternNu,
    PeriodicKernel, RationalQuadraticArdKernel, RationalQuadraticKernel, RbfArdKernel, RbfKernel,
    Triangle, WhiteKernel,
};
use crate::param::Interval;
use faer::{Mat, MatRef, mat};

const TOL: f64 = 1e-9;

use crate::test_check::{
    assert_close, assert_lower_close, assert_send_sync, fill, points_2d, sq_dist_1d,
};

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
            return Err(crate::GprError::LengthMismatch {
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
            .hess_from_coords::<crate::math::Accurate, _>(x, d2_k, i, j, uplo)
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
            return Err(crate::GprError::LengthMismatch {
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
            return Err(crate::GprError::IndexOutOfRange {
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
            return Err(crate::GprError::IndexOutOfRange {
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

#[test]
fn is_send_sync() {
    assert_send_sync::<CompiledKernel>();
}
