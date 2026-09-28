mod compose;
mod coord_mode;
mod custom;
mod fast_math;
mod params;
mod scalar;

use super::{CompiledKernel, MixedKernelViews};
use crate::kernel::{
    ConstantKernel, KernelScalar, KernelSpec, KernelTerm, LinearKernel, MaternArdKernel,
    MaternKernel, MaternNu, PeriodicKernel, RationalQuadraticArdKernel, RationalQuadraticKernel,
    RbfArdKernel, RbfKernel, Triangle, WhiteKernel,
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

/// One generic implementation serves `f32` and `f64` by delegating to the built-in leaf.
impl<T: KernelScalar> KernelTerm<T> for RbfAsTerm {
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
        dist: MatRef<'_, T>,
        out: faer::MatMut<'_, T>,
        uplo: Triangle,
    ) -> Result<(), crate::GprError> {
        self.0.apply(dist, out, uplo)
    }

    fn apply_cross(
        &self,
        dist: MatRef<'_, T>,
        out: faer::MatMut<'_, T>,
    ) -> Result<(), crate::GprError> {
        self.0.apply_cross(dist, out)
    }

    fn fill_diag(&self, out: &mut [T]) -> Result<(), crate::GprError> {
        self.0.fill_diag(out);
        Ok(())
    }

    fn grad(
        &self,
        dist: MatRef<'_, T>,
        d_k: faer::MatMut<'_, T>,
        param_idx: usize,
        uplo: Triangle,
    ) -> Result<(), crate::GprError> {
        self.0.grad(dist, d_k, param_idx, uplo)
    }

    fn hess(
        &self,
        dist: MatRef<'_, T>,
        d2_k: faer::MatMut<'_, T>,
        i: usize,
        j: usize,
        uplo: Triangle,
    ) -> Result<(), crate::GprError> {
        self.0.hess(dist, d2_k, i, j, uplo)
    }

    fn hess_points(
        &self,
        x: MatRef<'_, T>,
        d2_k: faer::MatMut<'_, T>,
        i: usize,
        j: usize,
        uplo: Triangle,
    ) -> Result<(), crate::GprError> {
        self.0
            .hess_from_coords::<crate::math::Accurate, _>(x, d2_k, i, j, uplo)
    }

    fn clone_box(&self) -> Box<dyn KernelTerm<T>> {
        Box::new(self.clone())
    }
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
