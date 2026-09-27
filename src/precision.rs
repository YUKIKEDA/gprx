//! Storage scalar and the crate-private mixed-precision residual solver.
//!
//! [`DoublePrecision`] is the public policy. `Gpr` fit and predict stay `f64`.
//! [`PromoteStorage`] and [`ReevaluateKernel`] are type parameters of
//! [`refine`]: one subtracts a saved `f32` matrix, the other recomputes the
//! kernel in `f64`. There is no flag and no alias that picks one of them.

use dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt;
use faer::linalg::cholesky::llt::factor::{LltError, LltRegularization};
use faer::{Mat, MatRef};

use crate::error::{CholeskyStage, GprError};
use crate::kernel::{CompiledKernel, Triangle};
use crate::workspace::{faer_par, faer_par_dims};

/// Selects storage and residual-refinement scalar types for GP computations.
pub trait PrecisionPolicy {
    /// Scalar used for `K`, `L`, and other stored buffers.
    type Storage;
    /// Scalar used when refining a solve against a higher-precision residual.
    type Refine;
}

/// Uses `f64` for both stored buffers and residual refinement.
///
/// Fit and predict on [`crate::Gpr`] stay in this precision. The crate-private
/// solver in this module refines an `f32` factor with an `f64` residual.
///
/// # Examples
///
/// ```rust
/// use gprx::{DoublePrecision, PrecisionPolicy};
///
/// type Storage = <DoublePrecision as PrecisionPolicy>::Storage;
/// let _: Storage = 0.0_f64;
/// ```
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DoublePrecision;

impl PrecisionPolicy for DoublePrecision {
    type Storage = f64;
    type Refine = f64;
}

/// Residual `r = y − Aα` from the `f32` matrix promoted to `f64`.
///
/// `Gpr` fit and predict stay `f64`, so only the unit tests call this solver.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct PromoteStorage;

/// Residual `r = y − Aα` from a fresh `f64` kernel evaluation.
///
/// `Gpr` fit and predict stay `f64`, so only the unit tests call this solver.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct ReevaluateKernel;

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) trait ResidualFormula {
    fn residual(
        saved: MatRef<'_, f32>,
        kernel: &CompiledKernel<f64>,
        x: MatRef<'_, f64>,
        noise: f64,
        alpha: &[f64],
        y: &[f64],
        r: &mut [f64],
    ) -> Result<f64, GprError>;
}

impl ResidualFormula for PromoteStorage {
    fn residual(
        saved: MatRef<'_, f32>,
        kernel: &CompiledKernel<f64>,
        x: MatRef<'_, f64>,
        noise: f64,
        alpha: &[f64],
        y: &[f64],
        r: &mut [f64],
    ) -> Result<f64, GprError> {
        let _ = (kernel, x, noise);
        Ok(row_sum_matvec(saved, alpha, y, r))
    }
}

impl ResidualFormula for ReevaluateKernel {
    fn residual(
        saved: MatRef<'_, f32>,
        kernel: &CompiledKernel<f64>,
        x: MatRef<'_, f64>,
        noise: f64,
        alpha: &[f64],
        y: &[f64],
        r: &mut [f64],
    ) -> Result<f64, GprError> {
        let _ = saved;
        fresh_residual(kernel, x, noise, alpha, y, r)
    }
}

/// Solves `Aα = y` with an `f32` Cholesky factor and an `f64` residual.
///
/// `A = K + σn² I`. The residual formula is `R`. At most 10 corrections are
/// applied. The stop test is `‖r‖∞ / (‖A‖∞ ‖α‖∞ + ‖y‖∞) < 10 n u_r` with
/// `u_r = f64::EPSILON`. Two consecutive residual-norm ratios above `0.9`,
/// or exhausting the 10 corrections, replaces `α` with the `f64` Cholesky
/// solution. A failed `f32` factorization returns [`GprError::CholeskyFailed`]
/// and does not add jitter.
///
/// # Errors
///
/// Returns the kernel's shape errors, or [`GprError::CholeskyFailed`] when the
/// `f32` or fallback `f64` factor is not positive definite.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn refine<R: ResidualFormula>(
    kernel_f32: &CompiledKernel<f32>,
    kernel_f64: &CompiledKernel<f64>,
    x: MatRef<'_, f64>,
    y: &[f64],
    noise: f64,
) -> Result<Vec<f64>, GprError> {
    let n = x.nrows();
    if n == 0 || y.len() != n {
        return Err(GprError::EmptyInput);
    }
    let mut x32 = Mat::<f32>::zeros(n, x.ncols());
    for col in 0..x.ncols() {
        for row in 0..n {
            x32[(row, col)] = x[(row, col)] as f32;
        }
    }
    let mut a = Mat::<f32>::zeros(n, n);
    let mut scratch = Mat::<f32>::zeros(n, n);
    kernel_f32.apply_points(x32.as_ref(), a.as_mut(), Triangle::Lower, scratch.as_mut())?;
    let noise32 = noise as f32;
    for i in 0..n {
        a[(i, i)] += noise32;
    }
    mirror_lower(&mut a);
    let saved = a.clone();
    factor_f32(&mut a)?;
    let mut rhs = Mat::<f32>::from_fn(n, 1, |i, _| y[i] as f32);
    solve_f32(a.as_ref(), &mut rhs);
    let mut alpha = vec![0.0; n];
    for i in 0..n {
        alpha[i] = f64::from(rhs[(i, 0)]);
    }

    let tol = 10.0 * n as f64 * f64::EPSILON;
    let mut resid = vec![0.0; n];
    let mut prev: Option<f64> = None;
    let mut streak = 0usize;
    for _ in 0..10 {
        let a_inf = R::residual(saved.as_ref(), kernel_f64, x, noise, &alpha, y, &mut resid)?;
        let r_inf = inf_norm(&resid);
        let denom = a_inf * inf_norm(&alpha) + inf_norm(y);
        if denom > 0.0 && r_inf / denom < tol {
            return Ok(alpha);
        }
        if let Some(prev_r) = prev {
            let ratio = if prev_r == 0.0 { 0.0 } else { r_inf / prev_r };
            if ratio > 0.9 {
                streak += 1;
                if streak >= 2 {
                    return f64_alpha(kernel_f64, x, y, noise);
                }
            } else {
                streak = 0;
            }
        }
        prev = Some(r_inf);
        for i in 0..n {
            rhs[(i, 0)] = resid[i] as f32;
        }
        solve_f32(a.as_ref(), &mut rhs);
        for i in 0..n {
            alpha[i] += f64::from(rhs[(i, 0)]);
        }
    }
    f64_alpha(kernel_f64, x, y, noise)
}

#[cfg_attr(not(test), allow(dead_code))]
fn row_sum_matvec(a: MatRef<'_, f32>, alpha: &[f64], y: &[f64], r: &mut [f64]) -> f64 {
    let n = y.len();
    let mut a_inf = 0.0f64;
    for i in 0..n {
        let mut row = 0.0;
        let mut sum = 0.0;
        for j in 0..n {
            let aij = f64::from(a[(i, j)]);
            row += aij.abs();
            sum += aij * alpha[j];
        }
        a_inf = a_inf.max(row);
        r[i] = y[i] - sum;
    }
    a_inf
}

#[cfg_attr(not(test), allow(dead_code))]
fn fresh_residual(
    kernel: &CompiledKernel<f64>,
    x: MatRef<'_, f64>,
    noise: f64,
    alpha: &[f64],
    y: &[f64],
    r: &mut [f64],
) -> Result<f64, GprError> {
    let n = y.len();
    let mut k = Mat::<f64>::zeros(n, n);
    let mut scratch = Mat::<f64>::zeros(n, n);
    kernel.apply_points(x, k.as_mut(), Triangle::Lower, scratch.as_mut())?;
    for i in 0..n {
        k[(i, i)] += noise;
    }
    let mut a_inf = 0.0f64;
    for i in 0..n {
        let mut row = 0.0;
        let mut sum = 0.0;
        for j in 0..n {
            let kij = if i >= j { k[(i, j)] } else { k[(j, i)] };
            row += kij.abs();
            sum += kij * alpha[j];
        }
        a_inf = a_inf.max(row);
        r[i] = y[i] - sum;
    }
    Ok(a_inf)
}

#[cfg_attr(not(test), allow(dead_code))]
fn f64_alpha(
    kernel: &CompiledKernel<f64>,
    x: MatRef<'_, f64>,
    y: &[f64],
    noise: f64,
) -> Result<Vec<f64>, GprError> {
    let n = y.len();
    let mut a = Mat::<f64>::zeros(n, n);
    let mut scratch = Mat::<f64>::zeros(n, n);
    kernel.apply_points(x, a.as_mut(), Triangle::Lower, scratch.as_mut())?;
    for i in 0..n {
        a[(i, i)] += noise;
    }
    factor_f64(&mut a)?;
    let mut rhs = Mat::<f64>::from_fn(n, 1, |i, _| y[i]);
    solve_f64(a.as_ref(), &mut rhs);
    Ok((0..n).map(|i| rhs[(i, 0)]).collect())
}

#[cfg_attr(not(test), allow(dead_code))]
fn mirror_lower(a: &mut Mat<f32>) {
    let n = a.nrows();
    for col in 0..n {
        for row in (col + 1)..n {
            a[(col, row)] = a[(row, col)];
        }
    }
}

#[cfg_attr(not(test), allow(dead_code))]
fn inf_norm(values: &[f64]) -> f64 {
    values.iter().fold(0.0, |acc, v| acc.max(v.abs()))
}

#[cfg_attr(not(test), allow(dead_code))]
fn factor_f32(a: &mut Mat<f32>) -> Result<(), GprError> {
    let n = a.nrows();
    let par = faer_par(n);
    let req = llt::factor::cholesky_in_place_scratch::<f32>(n, par, Default::default());
    let mut scratch = MemBuffer::new(req);
    let regularization = LltRegularization::<f32> {
        dynamic_regularization_delta: 0.0,
        dynamic_regularization_epsilon: 0.0,
    };
    match llt::factor::cholesky_in_place(
        a.as_mut(),
        regularization,
        par,
        MemStack::new(&mut scratch),
        Default::default(),
    ) {
        Ok(_) => Ok(()),
        Err(LltError::NonPositivePivot { .. }) => Err(GprError::CholeskyFailed {
            jitter: 0.0,
            matrix_size: n,
            stage: CholeskyStage::Predict,
        }),
    }
}

#[cfg_attr(not(test), allow(dead_code))]
fn factor_f64(a: &mut Mat<f64>) -> Result<(), GprError> {
    let n = a.nrows();
    let par = faer_par(n);
    let req = llt::factor::cholesky_in_place_scratch::<f64>(n, par, Default::default());
    let mut scratch = MemBuffer::new(req);
    let regularization = LltRegularization::<f64> {
        dynamic_regularization_delta: 0.0,
        dynamic_regularization_epsilon: 0.0,
    };
    match llt::factor::cholesky_in_place(
        a.as_mut(),
        regularization,
        par,
        MemStack::new(&mut scratch),
        Default::default(),
    ) {
        Ok(_) => Ok(()),
        Err(LltError::NonPositivePivot { .. }) => Err(GprError::CholeskyFailed {
            jitter: 0.0,
            matrix_size: n,
            stage: CholeskyStage::Predict,
        }),
    }
}

#[cfg_attr(not(test), allow(dead_code))]
fn solve_f32(l: MatRef<'_, f32>, rhs: &mut Mat<f32>) {
    let n = l.nrows();
    let par = faer_par_dims(n, 1);
    let req = llt::solve::solve_in_place_scratch::<f32>(n, 1, par);
    let mut scratch = MemBuffer::new(req);
    llt::solve::solve_in_place(l, rhs.as_mut(), par, MemStack::new(&mut scratch));
}

#[cfg_attr(not(test), allow(dead_code))]
fn solve_f64(l: MatRef<'_, f64>, rhs: &mut Mat<f64>) {
    let n = l.nrows();
    let par = faer_par_dims(n, 1);
    let req = llt::solve::solve_in_place_scratch::<f64>(n, 1, par);
    let mut scratch = MemBuffer::new(req);
    llt::solve::solve_in_place(l, rhs.as_mut(), par, MemStack::new(&mut scratch));
}

#[cfg(test)]
mod tests {
    use super::{DoublePrecision, PrecisionPolicy};

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn double_precision_is_f64() {
        let storage: <DoublePrecision as PrecisionPolicy>::Storage = 0.0;
        let refine: <DoublePrecision as PrecisionPolicy>::Refine = 0.0;
        let _ = (storage, refine);
        assert_send_sync::<DoublePrecision>();
    }

    use super::{PromoteStorage, ReevaluateKernel, ResidualFormula, f64_alpha, refine};
    use crate::kernel::{KernelSpec, RbfKernel, Triangle};
    use faer::{Mat, MatRef};
    use std::time::Instant;

    /// `ℓ` and `σn²` for the ill-conditioned Forrester probe (`n = 256`).
    const ILL_LENGTHSCALE: f64 = 1.0e4;
    const ILL_NOISE: f64 = 1.0e-5;

    fn forrester(n: usize) -> (Mat<f64>, Vec<f64>) {
        let x: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let mut rng = crate::rng::small_rng(0);
        let y: Vec<f64> = x
            .iter()
            .map(|&xi| {
                let t = 6.0 * xi - 2.0;
                t * t * (12.0 * xi - 4.0).sin() + crate::rng::unit_normal(&mut rng)
            })
            .collect();
        (Mat::from_fn(n, 1, |i, _| x[i]), y)
    }

    fn rbf(
        ell: f64,
    ) -> (
        crate::kernel::CompiledKernel<f32>,
        crate::kernel::CompiledKernel<f64>,
    ) {
        let spec = KernelSpec::from(RbfKernel::new(ell).expect("lengthscale"));
        (spec.compile_as::<f32>(), spec.compile())
    }

    fn rel_inf(got: &[f64], expect: &[f64]) -> f64 {
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for (g, e) in got.iter().zip(expect) {
            num = num.max((g - e).abs());
            den = den.max(e.abs());
        }
        num / den
    }

    fn digits<R: ResidualFormula>(ell: f64, noise: f64, x: MatRef<'_, f64>, y: &[f64]) {
        let (k32, k64) = rbf(ell);
        let alpha = refine::<R>(&k32, &k64, x, y, noise).expect("refine");
        let truth = f64_alpha(&k64, x, y, noise).expect("f64");
        let rel = rel_inf(&alpha, &truth);
        let bar = 10.0 * y.len() as f64 * f64::from(f32::EPSILON);
        assert!(rel < bar, "relative {rel} bar {bar}");
    }

    fn kappa(kernel: &crate::kernel::CompiledKernel<f64>, x: MatRef<'_, f64>, noise: f64) -> f64 {
        let n = x.nrows();
        let mut a = Mat::<f64>::zeros(n, n);
        let mut scratch = Mat::<f64>::zeros(n, n);
        kernel
            .apply_points(x, a.as_mut(), Triangle::Lower, scratch.as_mut())
            .expect("gram");
        for i in 0..n {
            a[(i, i)] += noise;
        }
        for col in 0..n {
            for row in (col + 1)..n {
                a[(col, row)] = a[(row, col)];
            }
        }
        let mut v = vec![1.0 / (n as f64).sqrt(); n];
        for _ in 0..40 {
            let mut w = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    w[i] += a[(i, j)] * v[j];
                }
            }
            let norm = w.iter().map(|t| t * t).sum::<f64>().sqrt();
            for (slot, value) in v.iter_mut().zip(&w) {
                *slot = value / norm;
            }
        }
        let mut av = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                av[i] += a[(i, j)] * v[j];
            }
        }
        let lam_max = v.iter().zip(&av).map(|(vi, avi)| vi * avi).sum::<f64>();
        let mut factor = a.clone();
        super::factor_f64(&mut factor).expect("f64 factor");
        let mut z = v.clone();
        for _ in 0..40 {
            let mut rhs = Mat::<f64>::from_fn(n, 1, |i, _| z[i]);
            super::solve_f64(factor.as_ref(), &mut rhs);
            let mut norm = 0.0;
            for i in 0..n {
                z[i] = rhs[(i, 0)];
                norm += z[i] * z[i];
            }
            norm = norm.sqrt();
            for slot in &mut z {
                *slot /= norm;
            }
        }
        for i in 0..n {
            av[i] = 0.0;
            for j in 0..n {
                av[i] += a[(i, j)] * z[j];
            }
        }
        let lam_min = z.iter().zip(&av).map(|(zi, avi)| zi * avi).sum::<f64>();
        lam_max / lam_min
    }

    #[test]
    fn both_residuals_hit_the_digit_bar() {
        let (x256, y256) = forrester(256);
        let (x1024, y1024) = forrester(1024);
        let (_, k64) = rbf(ILL_LENGTHSCALE);
        let cond = kappa(&k64, x256.as_ref(), ILL_NOISE);
        assert!(
            cond * f64::from(f32::EPSILON) > 1.0,
            "κ {cond} with ℓ={ILL_LENGTHSCALE} σn²={ILL_NOISE}"
        );
        for (x, y, ell, noise) in [
            (x256.as_ref(), y256.as_slice(), 1.0, 0.1),
            (x1024.as_ref(), y1024.as_slice(), 1.0, 0.1),
            (x256.as_ref(), y256.as_slice(), ILL_LENGTHSCALE, ILL_NOISE),
        ] {
            digits::<PromoteStorage>(ell, noise, x, y);
            digits::<ReevaluateKernel>(ell, noise, x, y);
        }
    }

    #[test]
    #[ignore]
    fn time_forrester_1024() {
        let (x, y) = forrester(1024);
        let (k32, k64) = rbf(1.0);
        let median = |tag: &str, run: &dyn Fn()| {
            run();
            let mut samples = [0.0; 11];
            for sample in &mut samples {
                let start = Instant::now();
                run();
                *sample = start.elapsed().as_secs_f64() * 1e3;
            }
            samples.sort_by(|a, b| a.total_cmp(b));
            println!("{tag} {:.4} ms", samples[5]);
        };
        median("promote", &|| {
            refine::<PromoteStorage>(&k32, &k64, x.as_ref(), &y, 0.1).expect("promote");
        });
        median("reevaluate", &|| {
            refine::<ReevaluateKernel>(&k32, &k64, x.as_ref(), &y, 0.1).expect("reevaluate");
        });
    }
}
