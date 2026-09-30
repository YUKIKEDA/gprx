//! A user leaf written once over `T: KernelScalar` runs in `f32` and `f64`.
//!
//! `ExpQuad` spells out the RBF formula with [`KernelScalar`] arithmetic.
//! `Delegating` forwards to the generic built-in [`RbfKernel`]. Both must
//! match the built-in leaf through the public fit / gradient / Hessian /
//! predict API at [`DoublePrecision`] and [`SinglePrecision`].

use faer::{Mat, MatMut, MatRef};
use gprx::kernel::{KernelScalar, KernelSpec, KernelTerm, RbfKernel, Triangle};
use gprx::{
    DoublePrecision, Fixed, GaussianLikelihood, GpScalar, Gpr, GprError, Interval, SinglePrecision,
};

mod common;
use common::{assert_close_named, assert_slice_close_named};

const ELL: f64 = 0.7;
const NOISE: f64 = 0.05;

/// `k = exp(-s / (2ℓ²))` with the optimizer parameter `θ = log(ℓ)`.
#[derive(Clone, Debug)]
struct ExpQuad {
    log_ell: f64,
}

impl ExpQuad {
    fn scales<T: KernelScalar>(&self) -> (T, T) {
        let ell = self.log_ell.exp();
        let inv_ell_sq = 1.0 / (ell * ell);
        (T::from_f64(0.5 * inv_ell_sq), T::from_f64(inv_ell_sq))
    }
}

fn require_one(idx: usize) -> Result<(), GprError> {
    if idx == 0 {
        Ok(())
    } else {
        Err(GprError::IndexOutOfRange {
            reason: format!("ExpQuad has one parameter, got index {idx}"),
        })
    }
}

/// Writes `f(dist[row, col])` over `uplo` of a square `out`.
fn write_uplo<T: KernelScalar>(
    dist: MatRef<'_, T>,
    mut out: MatMut<'_, T>,
    uplo: Triangle,
    f: impl Fn(T) -> T,
) -> Result<(), GprError> {
    let n = dist.nrows();
    if n == 0 {
        return Err(GprError::EmptyInput);
    }
    if dist.ncols() != n || out.nrows() != n || out.ncols() != n {
        return Err(GprError::ShapeMismatch {
            reason: "ExpQuad needs matching square matrices".to_owned(),
        });
    }
    for col in 0..n {
        let rows = match uplo {
            Triangle::Lower => col..n,
            Triangle::Upper => 0..col + 1,
            Triangle::Full => 0..n,
        };
        for row in rows {
            out[(row, col)] = f(dist[(row, col)]);
        }
    }
    Ok(())
}

fn sq_dist<T: KernelScalar>(x: MatRef<'_, T>) -> Mat<T> {
    Mat::from_fn(x.nrows(), x.nrows(), |row, col| {
        let mut s = T::from_f64(0.0);
        for dim in 0..x.ncols() {
            let diff = x[(row, dim)] - x[(col, dim)];
            s += diff * diff;
        }
        s
    })
}

impl<T: KernelScalar> KernelTerm<T> for ExpQuad {
    fn num_params(&self) -> usize {
        1
    }

    fn get_params(&self, out: &mut [f64]) -> Result<(), GprError> {
        match out {
            [theta] => {
                *theta = self.log_ell;
                Ok(())
            }
            _ => Err(GprError::LengthMismatch {
                reason: format!("expected 1 parameter, got {}", out.len()),
            }),
        }
    }

    fn set_params(&mut self, params: &[f64]) -> Result<(), GprError> {
        match params {
            [theta] => {
                self.log_ell = *theta;
                Ok(())
            }
            _ => Err(GprError::LengthMismatch {
                reason: format!("expected 1 parameter, got {}", params.len()),
            }),
        }
    }

    fn bounds_into(&self, out: &mut [Interval]) -> Result<(), GprError> {
        match out {
            [bound] => {
                *bound = Interval::new(1e-5, 1e5).map_err(|e| GprError::InvalidHyperparameter {
                    reason: e.to_string(),
                })?;
                Ok(())
            }
            _ => Err(GprError::LengthMismatch {
                reason: format!("expected 1 bound, got {}", out.len()),
            }),
        }
    }

    fn apply(
        &self,
        dist: MatRef<'_, T>,
        out: MatMut<'_, T>,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        let (half, _) = self.scales::<T>();
        write_uplo(dist, out, uplo, |s| (-s * half).exp())
    }

    fn apply_cross(&self, dist: MatRef<'_, T>, mut out: MatMut<'_, T>) -> Result<(), GprError> {
        let (half, _) = self.scales::<T>();
        for col in 0..dist.ncols() {
            for row in 0..dist.nrows() {
                out[(row, col)] = (-dist[(row, col)] * half).exp();
            }
        }
        Ok(())
    }

    fn fill_diag(&self, out: &mut [T]) -> Result<(), GprError> {
        out.fill(T::from_f64(1.0));
        Ok(())
    }

    fn grad(
        &self,
        dist: MatRef<'_, T>,
        d_k: MatMut<'_, T>,
        param_idx: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        require_one(param_idx)?;
        let (half, full) = self.scales::<T>();
        write_uplo(dist, d_k, uplo, |s| (-s * half).exp() * s * full)
    }

    fn hess(
        &self,
        dist: MatRef<'_, T>,
        d2_k: MatMut<'_, T>,
        i: usize,
        j: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        require_one(i)?;
        require_one(j)?;
        let (half, full) = self.scales::<T>();
        let two = T::from_f64(2.0);
        write_uplo(dist, d2_k, uplo, |s| {
            let u = s * full;
            (-s * half).exp() * u * (u - two)
        })
    }

    fn hess_points(
        &self,
        x: MatRef<'_, T>,
        d2_k: MatMut<'_, T>,
        i: usize,
        j: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        KernelTerm::<T>::hess(self, sq_dist(x).as_ref(), d2_k, i, j, uplo)
    }

    fn clone_box(&self) -> Box<dyn KernelTerm<T>> {
        Box::new(self.clone())
    }
}

/// Forwards every operation to the generic built-in leaf.
#[derive(Clone, Debug)]
struct Delegating(RbfKernel);

impl<T: KernelScalar> KernelTerm<T> for Delegating {
    fn num_params(&self) -> usize {
        self.0.num_params()
    }

    fn get_params(&self, out: &mut [f64]) -> Result<(), GprError> {
        self.0.get_params(out)
    }

    fn set_params(&mut self, params: &[f64]) -> Result<(), GprError> {
        self.0.set_params(params)
    }

    fn bounds_into(&self, out: &mut [Interval]) -> Result<(), GprError> {
        match out {
            [bound] => {
                *bound = self.0.bounds();
                Ok(())
            }
            _ => Err(GprError::LengthMismatch {
                reason: format!("expected 1 bound, got {}", out.len()),
            }),
        }
    }

    fn apply(
        &self,
        dist: MatRef<'_, T>,
        out: MatMut<'_, T>,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        self.0.apply(dist, out, uplo)
    }

    fn apply_cross(&self, dist: MatRef<'_, T>, out: MatMut<'_, T>) -> Result<(), GprError> {
        self.0.apply_cross(dist, out)
    }

    fn fill_diag(&self, out: &mut [T]) -> Result<(), GprError> {
        self.0.fill_diag(out);
        Ok(())
    }

    fn grad(
        &self,
        dist: MatRef<'_, T>,
        d_k: MatMut<'_, T>,
        param_idx: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        self.0.grad(dist, d_k, param_idx, uplo)
    }

    fn hess(
        &self,
        dist: MatRef<'_, T>,
        d2_k: MatMut<'_, T>,
        i: usize,
        j: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        self.0.hess(dist, d2_k, i, j, uplo)
    }

    fn hess_points(
        &self,
        x: MatRef<'_, T>,
        d2_k: MatMut<'_, T>,
        i: usize,
        j: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        self.0.hess(sq_dist(x).as_ref(), d2_k, i, j, uplo)
    }

    fn clone_box(&self) -> Box<dyn KernelTerm<T>> {
        Box::new(self.clone())
    }
}

fn problem() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..12).map(|i| f64::from(i) * 0.37).collect();
    let y: Vec<f64> = x.iter().map(|v| (1.3 * v).sin() + 0.1 * v).collect();
    let q = vec![0.15, 1.1, 2.9, 4.2];
    (x, y, q)
}

/// NLML, its gradient, the Hessian, and the predictive mean / variance at `θ`.
struct Probe {
    nlml: f64,
    grad: Vec<f64>,
    hess: Vec<f64>,
    mean: Vec<f64>,
    var: Vec<f64>,
}

/// Fits at `θ` with precision `P` and reads [`Probe`].
#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn probe<P: GpScalar>(kernel: KernelSpec) -> Probe {
    let (x, y, q) = problem();
    let n = y.len();
    let mut fitted = Gpr::new(kernel, GaussianLikelihood::new(NOISE).expect("noise"))
        .with_optimizer(Fixed)
        .with_precision::<P>()
        .factor(&x, n, 1, &y)
        .map_err(|(_, e)| e)
        .expect("fit");
    let mut params = vec![0.0; fitted.num_params()];
    fitted.get_params(&mut params).expect("params");
    let mut grad = vec![0.0; params.len()];
    let nlml = fitted
        .value_and_gradient_into(&params, &mut grad)
        .expect("grad");
    let mut hess = vec![0.0; params.len() * params.len()];
    fitted.hessian_into(&params, &mut hess).expect("hess");
    let pred = fitted.predict(&q, q.len(), 1).expect("predict");
    Probe {
        nlml,
        grad,
        hess,
        mean: pred.mean.iter().map(|v| v.to_f64()).collect(),
        var: pred.variance.iter().map(|v| v.to_f64()).collect(),
    }
}

fn assert_probe_close(what: &str, got: &Probe, expect: &Probe, tol: f64) {
    assert_close_named(&format!("{what} nlml"), got.nlml, expect.nlml, tol);
    for (name, actual, expected) in [
        ("grad", &got.grad, &expect.grad),
        ("hess", &got.hess, &expect.hess),
        ("mean", &got.mean, &expect.mean),
        ("variance", &got.var, &expect.var),
    ] {
        assert_slice_close_named(&format!("{what} {name}"), actual, expected, tol);
    }
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn builtin() -> KernelSpec {
    KernelSpec::from(RbfKernel::new(ELL).expect("rbf"))
}

fn exp_quad() -> KernelSpec {
    KernelSpec::custom(ExpQuad { log_ell: ELL.ln() })
}

#[allow(clippy::expect_used)] // helper is outside `#[test]`; clippy.toml allows only the test body
fn delegating() -> KernelSpec {
    KernelSpec::custom(Delegating(RbfKernel::new(ELL).expect("rbf")))
}

#[test]
fn generic_leaf_matches_builtin_in_f64() {
    let expect = probe::<DoublePrecision>(builtin());
    assert_probe_close(
        "ExpQuad",
        &probe::<DoublePrecision>(exp_quad()),
        &expect,
        1e-12,
    );
    assert_probe_close(
        "Delegating",
        &probe::<DoublePrecision>(delegating()),
        &expect,
        0.0,
    );
}

#[test]
fn generic_leaf_matches_builtin_in_f32() {
    let expect32 = probe::<SinglePrecision>(builtin());
    assert_probe_close(
        "ExpQuad f32",
        &probe::<SinglePrecision>(exp_quad()),
        &expect32,
        1e-5,
    );
    assert_probe_close(
        "Delegating f32",
        &probe::<SinglePrecision>(delegating()),
        &expect32,
        0.0,
    );
    // The f32 run is the same model as the f64 run, to f32 digits.
    let expect64 = probe::<DoublePrecision>(builtin());
    assert_probe_close(
        "ExpQuad f32 vs f64",
        &probe::<SinglePrecision>(exp_quad()),
        &expect64,
        1e-3,
    );
}
