//! Mixed-precision iterative refinement (§4.1, §4.2): one loop, three systems.
//!
//! [`refine`] runs the loop on any [`RefineSystem`]: Exact `α` ([`ExactSystem`]),
//! Sgpr weights, and the Svgp triangular solve. The stopping rules live here only.

use std::borrow::Cow;

use faer::{Mat, MatMut, MatRef};

use super::ResidualFormula;
use crate::error::{CholeskyStage, GprError};
use crate::kernel::{
    CompiledKernel, GramInputs, KernelScalar, KernelSpec, NoSupply, ScalarOps, Supply,
    TrainSources, Triangle,
};
use crate::linalg::{cholesky_lower_owned, inf_norm, symmetrize_lower};

/// Most corrections before falling back to the `f64` solve (§4.2).
const MAX_CORRECTIONS: usize = 10;

/// A residual-norm ratio above this counts toward stagnation (§4.2).
const STAGNATION_RATIO: f64 = 0.9;

/// Consecutive stagnating steps that end refinement with the `f64` solve.
const STAGNATION_STEPS: usize = 2;

/// Columns per `f64` kernel block in a residual. Bounds the scratch at
/// `3 · n · 32` `f64` (plus `n · 32` per dimension of an ARD distance slot),
/// so no `n×n` `f64` matrix is held.
pub(crate) const COLUMN_BLOCK: usize = 32;

/// A linear system `B w = b` with a low-precision factor of `B`.
pub(crate) trait RefineSystem {
    /// The right-hand side `b`.
    fn rhs(&self) -> &[f64];

    /// Writes `r = b − B w` and returns `‖B‖∞`.
    fn residual(&self, w: &[f64], r: &mut [f64]) -> Result<f64, GprError>;

    /// Adds the stored factor's solve of `r` to `w`.
    fn correct(&self, r: &[f64], w: &mut [f64]);

    /// Whether a `w` that met the stop test is kept. `false` takes
    /// [`Self::fallback`].
    fn accept(&self, _w: &[f64], _tol: f64) -> Result<bool, GprError> {
        Ok(true)
    }

    /// The `f64` solution used when refinement does not converge.
    fn fallback(&self) -> Result<Vec<f64>, GprError>;
}

/// Refines `w` from the factor's solution on `system`.
///
/// Stops when `‖r‖∞ / (‖B‖∞ ‖w‖∞ + ‖b‖∞) < 10 · dim · u_r` (`u_r =
/// f64::EPSILON`). Two consecutive residual-norm ratios above 0.9, or
/// [`MAX_CORRECTIONS`] corrections without meeting the test, return
/// [`RefineSystem::fallback`]. Jitter is never raised here.
///
/// # Errors
///
/// Propagates the system's residual and fallback errors.
pub(crate) fn refine<S: RefineSystem>(system: &S, mut w: Vec<f64>) -> Result<Vec<f64>, GprError> {
    let dim = w.len();
    let tol = 10.0 * dim as f64 * f64::EPSILON;
    let mut resid = vec![0.0; dim];
    let mut prev: Option<f64> = None;
    let mut streak = 0usize;
    for _ in 0..MAX_CORRECTIONS {
        let b_inf = system.residual(&w, &mut resid)?;
        let r_inf = inf_norm(&resid);
        let denom = b_inf * inf_norm(&w) + inf_norm(system.rhs());
        if denom > 0.0 && r_inf / denom < tol {
            if !system.accept(&w, tol)? {
                return system.fallback();
            }
            return Ok(w);
        }
        if let Some(prev_r) = prev {
            let ratio = if prev_r > 0.0 { r_inf / prev_r } else { 0.0 };
            if ratio > STAGNATION_RATIO {
                streak += 1;
                if streak >= STAGNATION_STEPS {
                    return system.fallback();
                }
            } else {
                streak = 0;
            }
        }
        prev = Some(r_inf);
        system.correct(&resid, &mut w);
    }
    system.fallback()
}

/// The training factor that fit or an online update left behind.
#[derive(Clone, Copy)]
pub enum StoredFactor<'a, T> {
    /// Lower `L` of `L Lᵀ` ([`crate::FittedGpr`]).
    Llt(MatRef<'a, T>),
    /// Unit-lower `L` with `D` on the diagonal ([`crate::OnlineGpr`]).
    Ldlt(MatRef<'a, T>),
}

impl<T: KernelScalar> StoredFactor<'_, T> {
    /// Overwrites `rhs` with `(A + (σn² + j) I)⁻¹ rhs` through the stored factor.
    pub(crate) fn solve_in_place(&self, rhs: MatMut<'_, T>) {
        match *self {
            Self::Llt(l) => T::solve_llt_owned_scratch(l, rhs),
            Self::Ldlt(ld) => {
                let n = rhs.nrows();
                crate::linalg::solve_ldlt_in_place(ld, rhs, n);
            }
        }
    }
}

/// The training system `A + (σn² + j) I` that the stored factor solves.
///
/// `j` is the jitter the last factorization added (`0` without a retry).
pub struct TrainSystem<'a, T: KernelScalar, S: Supply = NoSupply> {
    pub kernel: &'a KernelSpec<S>,
    pub compiled: &'a CompiledKernel<T, S>,
    /// Transformed training inputs (`n × d`).
    pub x: MatRef<'a, f64>,
    /// Training squared distances of a distance model (empty otherwise).
    pub sources: &'a TrainSources<T>,
    /// The same squares at `f64` without rounding, when the model keeps
    /// them ([`MixedPrecision`](super::MixedPrecision)).
    pub exact: Option<&'a TrainSources<f64>>,
    /// Transformed training targets.
    pub y: &'a [f64],
    pub noise: f64,
    pub jitter: f64,
    pub factor: StoredFactor<'a, T>,
    /// `factor⁻¹ y` in the storage scalar.
    pub factor_alpha: &'a [T],
    /// Retries for the `f64` fallback factor.
    pub policy: crate::policy::JitterPolicy,
    /// Reported on a failed fallback factor.
    pub stage: CholeskyStage,
}

/// Exact `α`: `(A + (σn² + j) I) α = y` through the stored `f32` factor.
///
/// [`PromoteStorage`](super::PromoteStorage) forms the residual with the
/// stored `f32` matrix and checks a converged `α` once against the `f64`
/// system; [`ReevaluateKernel`](super::ReevaluateKernel) forms it from a fresh
/// `f64` kernel. The fallback is the `f64` Cholesky solution of the same system.
pub(crate) struct ExactSystem<'s, 'a, M, R, S: Supply> {
    sys: &'s TrainSystem<'a, f32, S>,
    kernel_f64: CompiledKernel<f64, S>,
    /// The training squares at `f64`, widened once here when the model
    /// does not keep them, so no residual copies them again.
    exact: Cow<'a, TrainSources<f64>>,
    saved: Mat<f32>,
    diag: f64,
    _marker: core::marker::PhantomData<(M, R)>,
}

impl<'s, 'a, M: crate::math::KernelMath, R: ResidualFormula, S: Supply>
    ExactSystem<'s, 'a, M, R, S>
{
    /// Builds the system. [`PromoteStorage`](super::PromoteStorage) rebuilds
    /// the `f32` matrix once here.
    pub(crate) fn new(sys: &'s TrainSystem<'a, f32, S>) -> Result<Self, GprError> {
        let n = sys.x.nrows();
        if n == 0 || sys.y.len() != n || sys.factor_alpha.len() != n {
            return Err(GprError::EmptyInput);
        }
        let saved = if R::READS_STORAGE {
            storage_system::<M, _>(sys)?
        } else {
            Mat::<f32>::zeros(0, 0)
        };
        Ok(Self {
            sys,
            kernel_f64: sys.kernel.compile(),
            exact: exact_sources(sys)?,
            saved,
            diag: sys.noise + sys.jitter,
            _marker: core::marker::PhantomData,
        })
    }

    /// The factor's `α₀`, promoted to `f64`.
    pub(crate) fn start(&self) -> Vec<f64> {
        self.sys
            .factor_alpha
            .iter()
            .map(|&v| f64::from(v))
            .collect()
    }
}

impl<M: crate::math::KernelMath, R: ResidualFormula, S: Supply> RefineSystem
    for ExactSystem<'_, '_, M, R, S>
{
    fn rhs(&self) -> &[f64] {
        self.sys.y
    }

    fn residual(&self, w: &[f64], r: &mut [f64]) -> Result<f64, GprError> {
        if R::READS_STORAGE {
            Ok(row_sum_matvec(self.saved.as_ref(), w, self.sys.y, r))
        } else {
            fresh_residual::<M, _>(&self.kernel_f64, self.sys, &self.exact, self.diag, w, r)
        }
    }

    fn correct(&self, r: &[f64], w: &mut [f64]) {
        let mut rhs = Mat::<f32>::from_fn(r.len(), 1, |i, _| r[i] as f32);
        self.sys.factor.solve_in_place(rhs.as_mut());
        for (slot, i) in w.iter_mut().zip(0..) {
            *slot += f64::from(rhs[(i, 0)]);
        }
    }

    fn accept(&self, w: &[f64], tol: f64) -> Result<bool, GprError> {
        if R::READS_STORAGE {
            meets_f64_system::<M, _>(&self.kernel_f64, self.sys, &self.exact, self.diag, w, tol)
        } else {
            Ok(true)
        }
    }

    fn fallback(&self) -> Result<Vec<f64>, GprError> {
        f64_alpha::<M, _>(&self.kernel_f64, self.sys, &self.exact, self.diag)
    }
}

/// The training squares of `sys` at `f64`: the exact values when the model
/// keeps them, else the storage values widened.
pub(super) fn exact_sources<'a, S: Supply>(
    sys: &TrainSystem<'a, f32, S>,
) -> Result<Cow<'a, TrainSources<f64>>, GprError> {
    crate::kernel::widened(sys.exact, sys.sources)
}

/// Refined predict `α` for [`MixedPrecision`](super::MixedPrecision).
///
/// # Errors
///
/// Returns the kernel's shape errors, or [`GprError::CholeskyFailed`] at
/// `sys.stage` when the fallback `f64` factor is not positive definite.
pub(crate) fn refine_alpha<M: crate::math::KernelMath, R: ResidualFormula, S: Supply>(
    sys: &TrainSystem<'_, f32, S>,
) -> Result<Vec<f64>, GprError> {
    let system = ExactSystem::<M, R, S>::new(sys)?;
    let start = system.start();
    refine(&system, start)
}

/// Whether `α` also meets the stop test on the `f64` system.
///
/// [`PromoteStorage`] converges to the solution of the rounded `f32` system.
/// When `κ(A) u_f32` is large that solution is far from the `f64` one, so the
/// converged `α` is checked once against `K_f64 + diag · I` before it is kept.
fn meets_f64_system<M: crate::math::KernelMath, S: Supply>(
    kernel_f64: &CompiledKernel<f64, S>,
    sys: &TrainSystem<'_, f32, S>,
    exact: &TrainSources<f64>,
    diag: f64,
    alpha: &[f64],
    tol: f64,
) -> Result<bool, GprError> {
    let mut resid = vec![0.0; alpha.len()];
    let a_inf = fresh_residual::<M, _>(kernel_f64, sys, exact, diag, alpha, &mut resid)?;
    let denom = a_inf * inf_norm(alpha) + inf_norm(sys.y);
    Ok(denom > 0.0 && inf_norm(&resid) / denom < tol)
}

/// The `f32` training matrix `A + σn² I + j I`, full, as fit assembled it.
fn storage_system<M: crate::math::KernelMath, S: Supply>(
    sys: &TrainSystem<'_, f32, S>,
) -> Result<Mat<f32>, GprError> {
    let x = sys.x;
    let n = x.nrows();
    let mut x32 = Mat::<f32>::zeros(n, x.ncols());
    for col in 0..x.ncols() {
        for row in 0..n {
            x32[(row, col)] = x[(row, col)] as f32;
        }
    }
    let mut a = Mat::<f32>::zeros(n, n);
    let mut scratch = Mat::<f32>::zeros(n, n);
    sys.compiled.eval_gram::<M>(
        GramInputs::supplied(x32.as_ref(), S::squares(sys.sources)),
        a.as_mut(),
        Triangle::Lower,
        scratch.as_mut(),
        &mut Vec::new(),
    )?;
    let noise32 = sys.noise as f32;
    let jitter32 = sys.jitter as f32;
    for i in 0..n {
        a[(i, i)] += noise32;
        if sys.jitter != 0.0 {
            a[(i, i)] += jitter32;
        }
    }
    symmetrize_lower(a.as_mut(), n);
    Ok(a)
}

fn row_sum_matvec(a: MatRef<'_, f32>, alpha: &[f64], y: &[f64], r: &mut [f64]) -> f64 {
    // Column-major walk; each row still sums its terms in `j` order.
    let n = y.len();
    let mut row_abs = vec![0.0f64; n];
    let mut sum = vec![0.0f64; n];
    for (j, &aj) in alpha.iter().enumerate().take(n) {
        let col = a.col(j);
        for i in 0..n {
            let aij = f64::from(col[i]);
            row_abs[i] += aij.abs();
            sum[i] += aij * aj;
        }
    }
    for i in 0..n {
        r[i] = y[i] - sum[i];
    }
    row_abs.iter().fold(0.0f64, |acc, &v| acc.max(v))
}

/// `r = y − (K + diag · I) α` with `K` evaluated in `f64`, [`COLUMN_BLOCK`]
/// training columns at a time, so no `n×n` `f64` matrix is held. The
/// training squares of a distance kernel are read in place in the same
/// column ranges ([`crate::kernel::TrainSources::columns`]). Returns
/// `‖K + diag · I‖∞`.
fn fresh_residual<M: crate::math::KernelMath, S: Supply>(
    kernel: &CompiledKernel<f64, S>,
    sys: &TrainSystem<'_, f32, S>,
    exact: &TrainSources<f64>,
    diag: f64,
    alpha: &[f64],
    r: &mut [f64],
) -> Result<f64, GprError> {
    let (x, y) = (sys.x, sys.y);
    let n = y.len();
    let d = x.ncols();
    let block = COLUMN_BLOCK.min(n.max(1));
    let mut rows = Mat::<f64>::zeros(block, d);
    let mut k_block = Mat::<f64>::zeros(n, block);
    let mut scratch = Mat::<f64>::zeros(n, block);
    let mut dist = Mat::<f64>::zeros(n, block);
    let mut nested = Vec::new();
    let mut row_abs = vec![0.0f64; n];
    let mut sum = vec![0.0f64; n];
    let mut start = 0;
    while start < n {
        let len = block.min(n - start);
        for dim in 0..d {
            for jj in 0..len {
                rows[(jj, dim)] = x[(start + jj, dim)];
            }
        }
        let cols = exact.columns(start..start + len);
        kernel.eval_cross_slots::<M>(
            x,
            rows.as_ref().submatrix(0, 0, len, d),
            S::rects(&cols),
            Some(dist.as_mut().submatrix_mut(0, 0, n, len)),
            k_block.as_mut().submatrix_mut(0, 0, n, len),
            scratch.as_mut().submatrix_mut(0, 0, n, len),
            &mut nested,
            &mut [],
        )?;
        for jj in 0..len {
            let j = start + jj;
            let aj = alpha[j];
            for i in 0..n {
                let mut kij = k_block[(i, jj)];
                if i == j {
                    kij += diag;
                }
                row_abs[i] += kij.abs();
                sum[i] += kij * aj;
            }
        }
        start += len;
    }
    for i in 0..n {
        r[i] = y[i] - sum[i];
    }
    Ok(row_abs.iter().fold(0.0f64, |acc, &v| acc.max(v)))
}

/// `f64` Cholesky solution of `(A + diag · I) α = y`, retrying with `sys.policy`.
pub(crate) fn f64_alpha<M: crate::math::KernelMath, S: Supply>(
    kernel: &CompiledKernel<f64, S>,
    sys: &TrainSystem<'_, f32, S>,
    sources: &TrainSources<f64>,
    diag: f64,
) -> Result<Vec<f64>, GprError> {
    let x = sys.x;
    let y = sys.y;
    let n = y.len();
    let mut a = Mat::<f64>::zeros(n, n);
    let mut scratch = Mat::<f64>::zeros(n, n);
    kernel.eval_gram::<M>(
        GramInputs::supplied(x, S::squares(sources)),
        a.as_mut(),
        Triangle::Lower,
        scratch.as_mut(),
        &mut Vec::new(),
    )?;
    for i in 0..n {
        a[(i, i)] += diag;
    }
    cholesky_lower_owned(&mut a, sys.policy.retry_jitters(), sys.stage)?;
    let mut rhs = Mat::<f64>::from_fn(n, 1, |i, _| y[i]);
    f64::solve_llt_owned_scratch(a.as_ref(), rhs.as_mut());
    Ok((0..n).map(|i| rhs[(i, 0)]).collect())
}
