//! `f32` and `f64` evaluate the same formulas.

use crate::error::GprError;
use crate::kernel::visit_triangle;
use crate::kernel::{
    ConstantKernel, KernelScalar, KernelSpec, KernelTerm, LinearKernel, MaternArdKernel,
    MaternKernel, MaternNu, PeriodicKernel, RationalQuadraticArdKernel, RationalQuadraticKernel,
    RbfArdKernel, RbfKernel, Triangle, WhiteKernel,
};
use crate::param::Interval;
use faer::{Mat, MatMut, MatRef};

fn close(what: &str, got: f32, expect: f64) {
    let tol = 10.0 * f64::from(f32::EPSILON);
    let got = f64::from(got);
    let diff = (got - expect).abs();
    if expect.abs() < tol {
        assert!(diff < tol, "{what} absolute {diff} tol {tol}");
    } else {
        let rel = diff / expect.abs();
        assert!(rel < tol, "{what} relative {rel} got {got} expect {expect}");
    }
}

fn cmp_lower(what: &str, got: MatRef<'_, f32>, expect: MatRef<'_, f64>) {
    let n = expect.nrows();
    visit_triangle(n, Triangle::Lower, |row, col| {
        close(
            &format!("{what} ({row},{col})"),
            got[(row, col)],
            expect[(row, col)],
        );
    });
}

fn points() -> Mat<f64> {
    Mat::from_fn(2, 2, |row, col| [[0.0, 0.0], [1.0, 0.0]][row][col])
}

fn sq_pair(x: MatRef<'_, f64>, i: usize, j: usize) -> (f64, f32) {
    let mut s64 = 0.0;
    let mut s32 = 0.0f32;
    for col in 0..x.ncols() {
        let a = x[(i, col)];
        let b = x[(j, col)];
        s64 += (a - b) * (a - b);
        let a32 = a as f32;
        let b32 = b as f32;
        s32 += (a32 - b32) * (a32 - b32);
    }
    (s64, s32)
}

fn agree(name: &str, spec: &KernelSpec, x: &Mat<f64>, on_dist: bool) {
    let n = x.nrows();
    let k64 = spec.compile();
    let k32 = spec.compile_as::<f32>();
    let mut x32 = Mat::<f32>::zeros(n, x.ncols());
    for col in 0..x.ncols() {
        for row in 0..n {
            x32[(row, col)] = x[(row, col)] as f32;
        }
    }
    let mut o64 = Mat::<f64>::zeros(n, n);
    let mut o32 = Mat::<f32>::zeros(n, n);
    let mut s64 = Mat::<f64>::zeros(n, n);
    let mut s32 = Mat::<f32>::zeros(n, n);
    k64.apply_points::<crate::math::Accurate>(
        x.as_ref(),
        o64.as_mut(),
        Triangle::Lower,
        s64.as_mut(),
    )
    .expect("f64 points");
    k32.apply_points::<crate::math::Accurate>(
        x32.as_ref(),
        o32.as_mut(),
        Triangle::Lower,
        s32.as_mut(),
    )
    .expect("f32 points");
    cmp_lower(&format!("{name} points"), o32.as_ref(), o64.as_ref());
    let p = k64.num_params();
    for idx in 0..p {
        k64.grad_points::<crate::math::Accurate>(
            x.as_ref(),
            o64.as_mut(),
            idx,
            Triangle::Lower,
            s64.as_mut(),
        )
        .expect("f64 grad");
        k32.grad_points::<crate::math::Accurate>(
            x32.as_ref(),
            o32.as_mut(),
            idx,
            Triangle::Lower,
            s32.as_mut(),
        )
        .expect("f32 grad");
        cmp_lower(&format!("{name} grad {idx}"), o32.as_ref(), o64.as_ref());
        for j in 0..=idx {
            k64.hess_points::<crate::math::Accurate>(
                x.as_ref(),
                o64.as_mut(),
                idx,
                j,
                Triangle::Lower,
                s64.as_mut(),
            )
            .expect("f64 hess");
            k32.hess_points::<crate::math::Accurate>(
                x32.as_ref(),
                o32.as_mut(),
                idx,
                j,
                Triangle::Lower,
                s32.as_mut(),
            )
            .expect("f32 hess");
            cmp_lower(
                &format!("{name} hess {idx},{j}"),
                o32.as_ref(),
                o64.as_ref(),
            );
        }
    }
    if !on_dist {
        return;
    }
    let mut d64 = Mat::<f64>::zeros(n, n);
    let mut d32 = Mat::<f32>::zeros(n, n);
    for col in 0..n {
        for row in col..n {
            let (s64v, s32v) = sq_pair(x.as_ref(), row, col);
            d64[(row, col)] = s64v;
            d32[(row, col)] = s32v;
        }
    }
    k64.apply::<crate::math::Accurate>(d64.as_ref(), o64.as_mut(), Triangle::Lower, s64.as_mut())
        .expect("f64 dist");
    k32.apply::<crate::math::Accurate>(d32.as_ref(), o32.as_mut(), Triangle::Lower, s32.as_mut())
        .expect("f32 dist");
    cmp_lower(&format!("{name} dist"), o32.as_ref(), o64.as_ref());
}

#[derive(Clone, Debug)]
struct ScaleLeaf(f64);

impl<T: KernelScalar> KernelTerm<T> for ScaleLeaf {
    fn clone_box(&self) -> Box<dyn KernelTerm<T>> {
        Box::new(self.clone())
    }
    fn num_params(&self) -> usize {
        1
    }
    fn get_params(&self, out: &mut [f64]) -> Result<(), GprError> {
        out[0] = self.0.ln();
        Ok(())
    }
    fn set_params(&mut self, params: &[f64]) -> Result<(), GprError> {
        self.0 = params[0].exp();
        Ok(())
    }
    fn bounds_into(&self, out: &mut [Interval]) -> Result<(), GprError> {
        out[0] = Interval::new(1e-6, 1e6).expect("interval");
        Ok(())
    }
    fn apply(
        &self,
        dist: MatRef<'_, T>,
        out: MatMut<'_, T>,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        let _ = dist;
        paint(out, uplo, T::from_f64(self.0));
        Ok(())
    }
    fn apply_cross(&self, dist: MatRef<'_, T>, mut out: MatMut<'_, T>) -> Result<(), GprError> {
        let _ = dist;
        let value = T::from_f64(self.0);
        for col in 0..out.ncols() {
            for row in 0..out.nrows() {
                out[(row, col)] = value;
            }
        }
        Ok(())
    }
    fn fill_diag(&self, out: &mut [T]) -> Result<(), GprError> {
        out.fill(T::from_f64(self.0));
        Ok(())
    }
    fn grad(
        &self,
        dist: MatRef<'_, T>,
        out: MatMut<'_, T>,
        param_idx: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        if param_idx != 0 {
            return Err(GprError::IndexOutOfRange {
                reason: "scale leaf has one parameter".to_owned(),
            });
        }
        self.apply(dist, out, uplo)
    }
    fn hess(
        &self,
        dist: MatRef<'_, T>,
        out: MatMut<'_, T>,
        i: usize,
        j: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        if i != 0 || j != 0 {
            return Err(GprError::IndexOutOfRange {
                reason: "scale leaf has one parameter".to_owned(),
            });
        }
        self.apply(dist, out, uplo)
    }
    fn hess_points(
        &self,
        x: MatRef<'_, T>,
        out: MatMut<'_, T>,
        i: usize,
        j: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        let _ = x;
        self.hess(x, out, i, j, uplo)
    }
}

fn paint<T: KernelScalar>(mut out: MatMut<'_, T>, uplo: Triangle, value: T) {
    visit_triangle(out.nrows(), uplo, |row, col| out[(row, col)] = value);
}

#[test]
fn f32_leaves_match_f64() {
    let x = points();
    let rbf = KernelSpec::from(RbfKernel::new(1.0).expect("rbf"));
    agree("rbf", &rbf, &x, true);
    agree(
        "rbf ard",
        &KernelSpec::from(RbfArdKernel::new(&[1.0, 2.0]).expect("ard")),
        &x,
        false,
    );
    for nu in [MaternNu::Half, MaternNu::ThreeHalves, MaternNu::FiveHalves] {
        agree(
            "matern",
            &KernelSpec::from(MaternKernel::new(1.0, nu).expect("matern")),
            &x,
            true,
        );
    }
    agree(
        "matern ard",
        &KernelSpec::from(
            MaternArdKernel::new(&[1.0, 2.0], MaternNu::ThreeHalves).expect("matern ard"),
        ),
        &x,
        false,
    );
    agree(
        "periodic",
        &KernelSpec::from(PeriodicKernel::new(1.0, 4.0).expect("periodic")),
        &x,
        true,
    );
    agree(
        "rq",
        &KernelSpec::from(RationalQuadraticKernel::new(1.0, 1.0).expect("rq")),
        &x,
        true,
    );
    agree(
        "rq ard",
        &KernelSpec::from(RationalQuadraticArdKernel::new(&[1.0, 1.0], 1.0).expect("rq ard")),
        &x,
        false,
    );
    agree(
        "constant",
        &KernelSpec::from(ConstantKernel::new(1.5).expect("constant")),
        &x,
        true,
    );
    agree(
        "linear",
        &KernelSpec::from(LinearKernel::new(1.0).expect("linear")),
        &x,
        false,
    );
    agree(
        "white",
        &KernelSpec::from(WhiteKernel::new(0.0625).expect("white")),
        &x,
        true,
    );
    let other = KernelSpec::from(RbfKernel::new(2.0).expect("rbf"));
    agree("sum", &(rbf.clone() + other.clone()), &x, true);
    agree("product", &(rbf * other), &x, true);
    agree("custom", &KernelSpec::custom(ScaleLeaf(1.25)), &x, true);
}
