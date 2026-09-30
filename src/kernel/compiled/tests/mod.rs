mod compose;
mod coord_deriv;
mod coord_mode;
mod cross;
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

    fn grad_cross(
        &self,
        dist: MatRef<'_, T>,
        d_k: faer::MatMut<'_, T>,
        param_idx: usize,
    ) -> Result<(), crate::GprError> {
        if param_idx != 0 {
            return Err(crate::GprError::IndexOutOfRange {
                reason: "one parameter".to_owned(),
            });
        }
        let ell = T::from_f64(self.0.lengthscale());
        crate::kernel::write_rect(d_k, |row, col| {
            let d2 = dist[(row, col)];
            Ok((T::from_f64(-0.5) * d2 / (ell * ell)).exp() * d2 / (ell * ell))
        })
    }

    fn hess_cross(
        &self,
        dist: MatRef<'_, T>,
        d2_k: faer::MatMut<'_, T>,
        i: usize,
        j: usize,
    ) -> Result<(), crate::GprError> {
        if i != 0 || j != 0 {
            return Err(crate::GprError::IndexOutOfRange {
                reason: "one parameter".to_owned(),
            });
        }
        let ell = T::from_f64(self.0.lengthscale());
        crate::kernel::write_rect(d2_k, |row, col| {
            let s = dist[(row, col)] / (ell * ell);
            Ok((T::from_f64(-0.5) * s).exp() * (s * s - T::from_f64(2.0) * s))
        })
    }

    fn grad_wrt_sq_dist(
        &self,
        dist: MatRef<'_, T>,
        out: faer::MatMut<'_, T>,
    ) -> Result<(), crate::GprError> {
        let ell = T::from_f64(self.0.lengthscale());
        crate::kernel::write_rect(out, |row, col| {
            let s = dist[(row, col)] / (ell * ell);
            Ok(T::from_f64(-0.5) * (T::from_f64(-0.5) * s).exp() / (ell * ell))
        })
    }

    fn hess_wrt_sq_dist(
        &self,
        dist: MatRef<'_, T>,
        out: faer::MatMut<'_, T>,
    ) -> Result<(), crate::GprError> {
        let ell = T::from_f64(self.0.lengthscale());
        crate::kernel::write_rect(out, |row, col| {
            let s = dist[(row, col)] / (ell * ell);
            Ok(T::from_f64(0.25) * (T::from_f64(-0.5) * s).exp() / (ell * ell * ell * ell))
        })
    }

    fn grad_wrt_sq_dist_theta(
        &self,
        dist: MatRef<'_, T>,
        out: faer::MatMut<'_, T>,
        param_idx: usize,
    ) -> Result<(), crate::GprError> {
        if param_idx != 0 {
            return Err(crate::GprError::IndexOutOfRange {
                reason: "one parameter".to_owned(),
            });
        }
        let ell = T::from_f64(self.0.lengthscale());
        crate::kernel::write_rect(out, |row, col| {
            let s = dist[(row, col)] / (ell * ell);
            Ok((T::from_f64(-0.5) * s).exp() / (ell * ell)
                * (T::from_f64(1.0) - T::from_f64(0.5) * s))
        })
    }

    fn clone_box(&self) -> Box<dyn KernelTerm<T>> {
        Box::new(self.clone())
    }
}

/// [`RbfAsTerm`] without the rectangular and coordinate derivatives (the trait
/// defaults).
#[derive(Clone, Debug)]
struct NoCrossDerivatives(RbfAsTerm);

impl<T: KernelScalar> KernelTerm<T> for NoCrossDerivatives {
    fn num_params(&self) -> usize {
        KernelTerm::<T>::num_params(&self.0)
    }

    fn get_params(&self, out: &mut [f64]) -> Result<(), crate::GprError> {
        KernelTerm::<T>::get_params(&self.0, out)
    }

    fn set_params(&mut self, params: &[f64]) -> Result<(), crate::GprError> {
        KernelTerm::<T>::set_params(&mut self.0, params)
    }

    fn bounds_into(&self, out: &mut [Interval]) -> Result<(), crate::GprError> {
        KernelTerm::<T>::bounds_into(&self.0, out)
    }

    fn apply(
        &self,
        dist: MatRef<'_, T>,
        out: faer::MatMut<'_, T>,
        uplo: Triangle,
    ) -> Result<(), crate::GprError> {
        KernelTerm::<T>::apply(&self.0, dist, out, uplo)
    }

    fn apply_cross(
        &self,
        dist: MatRef<'_, T>,
        out: faer::MatMut<'_, T>,
    ) -> Result<(), crate::GprError> {
        KernelTerm::<T>::apply_cross(&self.0, dist, out)
    }

    fn fill_diag(&self, out: &mut [T]) -> Result<(), crate::GprError> {
        KernelTerm::<T>::fill_diag(&self.0, out)
    }

    fn grad(
        &self,
        dist: MatRef<'_, T>,
        d_k: faer::MatMut<'_, T>,
        param_idx: usize,
        uplo: Triangle,
    ) -> Result<(), crate::GprError> {
        KernelTerm::<T>::grad(&self.0, dist, d_k, param_idx, uplo)
    }

    fn hess(
        &self,
        dist: MatRef<'_, T>,
        d2_k: faer::MatMut<'_, T>,
        i: usize,
        j: usize,
        uplo: Triangle,
    ) -> Result<(), crate::GprError> {
        KernelTerm::<T>::hess(&self.0, dist, d2_k, i, j, uplo)
    }

    fn hess_points(
        &self,
        x: MatRef<'_, T>,
        d2_k: faer::MatMut<'_, T>,
        i: usize,
        j: usize,
        uplo: Triangle,
    ) -> Result<(), crate::GprError> {
        KernelTerm::<T>::hess_points(&self.0, x, d2_k, i, j, uplo)
    }

    fn clone_box(&self) -> Box<dyn KernelTerm<T>> {
        Box::new(self.clone())
    }
}

fn custom_rbf_without_derivatives(ell: f64) -> KernelSpec {
    KernelSpec::custom(NoCrossDerivatives(RbfAsTerm(
        RbfKernel::new(ell).expect("valid"),
    )))
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
