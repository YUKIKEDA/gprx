//! Storage scalar and the mixed-precision residual solver.
//!
//! Omitted model precision is [`DoublePrecision`] (`f64`). [`SinglePrecision`]
//! runs the same procedures in `f32` and keeps the factorization as-is.
//! [`MixedPrecision`] factors in `f32` and refines the predict `α` in `f64`.
//! The residual type parameter exists only on [`MixedPrecision`]. There is no
//! flag and no alias that picks a residual formula.

use dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt;
use faer::linalg::cholesky::llt::factor::{LltError, LltRegularization};
use faer::{Mat, MatMut, MatRef};

use crate::error::{CholeskyStage, GprError};
use crate::kernel::{CompiledKernel, FillDistances, KernelScalar, KernelSpec, Triangle};
use crate::transform::TargetTransform;
use crate::workspace::{faer_par, faer_par_dims};

/// Selects storage and residual-refinement scalar types for GP computations.
pub trait PrecisionPolicy {
    /// Scalar used for `K`, `L`, and other stored buffers.
    type Storage: KernelScalar;
    /// Scalar used when refining a solve against a higher-precision residual.
    type Refine: KernelScalar;
}

/// Uses `f64` for stored factors and for refinement.
///
/// This is the precision when a model omits the parameter. Fit and predict
/// stay on the `f64` factorization.
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

/// Uses `f32` for stored factors and keeps that factorization as-is.
///
/// The same Gram, Cholesky, and predict steps as [`DoublePrecision`] run in
/// `f32`. There is no residual type parameter.
///
/// # Examples
///
/// ```rust
/// use gprx::{PrecisionPolicy, SinglePrecision};
///
/// type Storage = <SinglePrecision as PrecisionPolicy>::Storage;
/// let _: Storage = 0.0_f32;
/// ```
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SinglePrecision;

impl PrecisionPolicy for SinglePrecision {
    type Storage = f32;
    type Refine = f32;
}

/// Factors in `f32` and refines the predict `α` in `f64`.
///
/// `R` is the residual formula. Omitting it selects [`PromoteStorage`].
/// [`ReevaluateKernel`] recomputes the kernel in `f64` for each residual.
/// Training marginal likelihood uses the `f32` factor and does not refine `α`.
///
/// # Examples
///
/// ```rust
/// use gprx::{MixedPrecision, PrecisionPolicy, PromoteStorage, ReevaluateKernel};
///
/// type DefaultResidual = MixedPrecision;
/// type FreshKernel = MixedPrecision<ReevaluateKernel>;
/// let _: <DefaultResidual as PrecisionPolicy>::Storage = 0.0_f32;
/// let _: <FreshKernel as PrecisionPolicy>::Refine = 0.0_f64;
/// let _ = core::marker::PhantomData::<PromoteStorage>;
/// ```
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MixedPrecision<R: ResidualFormula = PromoteStorage>(core::marker::PhantomData<R>);

impl<R: ResidualFormula> PrecisionPolicy for MixedPrecision<R> {
    type Storage = f32;
    type Refine = f64;
}

mod residual_seal {
    pub trait Sealed {}
}

/// How [`MixedPrecision`] builds `r = y − Aα`.
///
/// The only implementations are [`PromoteStorage`] and [`ReevaluateKernel`].
#[allow(private_bounds)]
pub trait ResidualFormula: residual_seal::Sealed {
    /// Writes the residual and returns `‖A‖∞` for the stopping test.
    ///
    /// # Errors
    ///
    /// Returns [`GprError`] when the kernel evaluation fails or a value is
    /// non-finite.
    fn residual<M: crate::math::KernelMath>(
        saved: MatRef<'_, f32>,
        kernel: &CompiledKernel<f64>,
        x: MatRef<'_, f64>,
        noise: f64,
        alpha: &[f64],
        y: &[f64],
        r: &mut [f64],
    ) -> Result<f64, GprError>;
}

/// Residual `r = y − Aα` from the saved `f32` matrix promoted to `f64`.
///
/// This is the residual when [`MixedPrecision`] omits its type parameter.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PromoteStorage;

/// Residual `r = y − Aα` from a fresh `f64` kernel evaluation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ReevaluateKernel;

impl residual_seal::Sealed for PromoteStorage {}
impl residual_seal::Sealed for ReevaluateKernel {}

#[allow(private_bounds)]
impl ResidualFormula for PromoteStorage {
    fn residual<M: crate::math::KernelMath>(
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

#[allow(private_bounds)]
impl ResidualFormula for ReevaluateKernel {
    fn residual<M: crate::math::KernelMath>(
        saved: MatRef<'_, f32>,
        kernel: &CompiledKernel<f64>,
        x: MatRef<'_, f64>,
        noise: f64,
        alpha: &[f64],
        y: &[f64],
        r: &mut [f64],
    ) -> Result<f64, GprError> {
        let _ = saved;
        fresh_residual::<M>(kernel, x, noise, alpha, y, r)
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
pub(crate) fn refine<M: crate::math::KernelMath, R: ResidualFormula>(
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
    kernel_f32.apply_points::<M>(x32.as_ref(), a.as_mut(), Triangle::Lower, scratch.as_mut())?;
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
        let a_inf = R::residual::<M>(saved.as_ref(), kernel_f64, x, noise, &alpha, y, &mut resid)?;
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
                    return f64_alpha::<M>(kernel_f64, x, y, noise);
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
    f64_alpha::<M>(kernel_f64, x, y, noise)
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
fn fresh_residual<M: crate::math::KernelMath>(
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
    kernel.apply_points::<M>(x, k.as_mut(), Triangle::Lower, scratch.as_mut())?;
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
fn f64_alpha<M: crate::math::KernelMath>(
    kernel: &CompiledKernel<f64>,
    x: MatRef<'_, f64>,
    y: &[f64],
    noise: f64,
) -> Result<Vec<f64>, GprError> {
    let n = y.len();
    let mut a = Mat::<f64>::zeros(n, n);
    let mut scratch = Mat::<f64>::zeros(n, n);
    kernel.apply_points::<M>(x, a.as_mut(), Triangle::Lower, scratch.as_mut())?;
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

/// Which precision a persist directory records. Absent on disk means double.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PersistKind {
    Double,
    Single,
    MixedPromote,
    MixedReevaluate,
}

pub(crate) trait ResidualTag: ResidualFormula {
    const REEVALUATES: bool;
}

impl ResidualTag for PromoteStorage {
    const REEVALUATES: bool = false;
}

impl ResidualTag for ReevaluateKernel {
    const REEVALUATES: bool = true;
}

/// Bounds a model precision so distance fills and both scalars are known.
pub(crate) trait ModelPrecision:
    PrecisionPolicy<Storage: FillDistances> + Copy + Send + Sync + 'static
{
    /// `true` when predict weights are refined in `f64` from an `f32` factor.
    const REFINES_IN_F64: bool;

    fn persist_kind() -> PersistKind;
}

impl ModelPrecision for DoublePrecision {
    const REFINES_IN_F64: bool = false;

    fn persist_kind() -> PersistKind {
        PersistKind::Double
    }
}
impl ModelPrecision for SinglePrecision {
    const REFINES_IN_F64: bool = false;

    fn persist_kind() -> PersistKind {
        PersistKind::Single
    }
}
impl<R> ModelPrecision for MixedPrecision<R>
where
    R: ResidualTag + Copy + Send + Sync + 'static,
{
    const REFINES_IN_F64: bool = true;

    fn persist_kind() -> PersistKind {
        if R::REEVALUATES {
            PersistKind::MixedReevaluate
        } else {
            PersistKind::MixedPromote
        }
    }
}

/// Writes predict `α` from the storage factor, or refines it.
pub(crate) trait PublishPredictAlpha: ModelPrecision {
    /// Stores predict weights in `alpha`.
    ///
    /// [`DoublePrecision`] and [`SinglePrecision`] copy `factor_alpha`.
    /// [`MixedPrecision`] calls [`refine`].
    fn publish_predict_alpha<M: crate::math::KernelMath>(
        kernel: &KernelSpec,
        compiled: &CompiledKernel<Self::Storage>,
        x: MatRef<'_, f64>,
        y: &[f64],
        noise: f64,
        factor_alpha: &[Self::Storage],
        alpha: &mut Vec<Self::Refine>,
    ) -> Result<(), GprError>;
}

fn copy_factor_to_refine<P: ModelPrecision>(
    factor_alpha: &[P::Storage],
    alpha: &mut Vec<P::Refine>,
) {
    if alpha.len() != factor_alpha.len() {
        alpha.resize(factor_alpha.len(), P::Refine::from_f64(0.0));
    }
    for (slot, &value) in alpha.iter_mut().zip(factor_alpha.iter()) {
        *slot = P::Refine::from_f64(value.to_f64());
    }
}

impl PublishPredictAlpha for DoublePrecision {
    fn publish_predict_alpha<M: crate::math::KernelMath>(
        kernel: &KernelSpec,
        compiled: &CompiledKernel<Self::Storage>,
        x: MatRef<'_, f64>,
        y: &[f64],
        noise: f64,
        factor_alpha: &[Self::Storage],
        alpha: &mut Vec<Self::Refine>,
    ) -> Result<(), GprError> {
        let _ = (kernel, compiled, x, y, noise);
        copy_factor_to_refine::<Self>(factor_alpha, alpha);
        Ok(())
    }
}

impl PublishPredictAlpha for SinglePrecision {
    fn publish_predict_alpha<M: crate::math::KernelMath>(
        kernel: &KernelSpec,
        compiled: &CompiledKernel<Self::Storage>,
        x: MatRef<'_, f64>,
        y: &[f64],
        noise: f64,
        factor_alpha: &[Self::Storage],
        alpha: &mut Vec<Self::Refine>,
    ) -> Result<(), GprError> {
        let _ = (kernel, compiled, x, y, noise);
        copy_factor_to_refine::<Self>(factor_alpha, alpha);
        Ok(())
    }
}

impl<R> PublishPredictAlpha for MixedPrecision<R>
where
    R: ResidualTag + Copy + Send + Sync + 'static,
{
    fn publish_predict_alpha<M: crate::math::KernelMath>(
        kernel: &KernelSpec,
        compiled: &CompiledKernel<Self::Storage>,
        x: MatRef<'_, f64>,
        y: &[f64],
        noise: f64,
        factor_alpha: &[Self::Storage],
        alpha: &mut Vec<Self::Refine>,
    ) -> Result<(), GprError> {
        let _ = factor_alpha;
        let kernel_f64 = kernel.compile();
        *alpha = refine::<M, R>(compiled, &kernel_f64, x, y, noise)?;
        Ok(())
    }
}

/// Predictive mean from storage `k_*`, or a fresh `f64` column for [`ReevaluateKernel`].
pub(crate) trait PredictMean: ModelPrecision {
    /// Dot of query column `col` with predict `α`, as [`PrecisionPolicy::Refine`].
    fn column_mean<M: crate::math::KernelMath>(
        kernel: &KernelSpec,
        k_storage: MatRef<'_, Self::Storage>,
        x_train: MatRef<'_, f64>,
        x_query: &[f64],
        n_cols: usize,
        alpha: &[Self::Refine],
        col: usize,
    ) -> Result<Self::Refine, GprError>;
}

fn storage_column_mean<P: ModelPrecision>(
    k_storage: MatRef<'_, P::Storage>,
    alpha: &[P::Refine],
    col: usize,
) -> P::Refine {
    let mut sum = 0.0f64;
    for (row, &weight) in alpha.iter().enumerate() {
        sum += k_storage[(row, col)].to_f64() * weight.to_f64();
    }
    P::Refine::from_f64(sum)
}

impl PredictMean for DoublePrecision {
    fn column_mean<M: crate::math::KernelMath>(
        kernel: &KernelSpec,
        k_storage: MatRef<'_, Self::Storage>,
        x_train: MatRef<'_, f64>,
        x_query: &[f64],
        n_cols: usize,
        alpha: &[Self::Refine],
        col: usize,
    ) -> Result<Self::Refine, GprError> {
        let _ = (kernel, x_train, x_query, n_cols);
        Ok(storage_column_mean::<Self>(k_storage, alpha, col))
    }
}

impl PredictMean for SinglePrecision {
    fn column_mean<M: crate::math::KernelMath>(
        kernel: &KernelSpec,
        k_storage: MatRef<'_, Self::Storage>,
        x_train: MatRef<'_, f64>,
        x_query: &[f64],
        n_cols: usize,
        alpha: &[Self::Refine],
        col: usize,
    ) -> Result<Self::Refine, GprError> {
        let _ = (kernel, x_train, x_query, n_cols);
        Ok(storage_column_mean::<Self>(k_storage, alpha, col))
    }
}

impl PredictMean for MixedPrecision<PromoteStorage> {
    fn column_mean<M: crate::math::KernelMath>(
        kernel: &KernelSpec,
        k_storage: MatRef<'_, Self::Storage>,
        x_train: MatRef<'_, f64>,
        x_query: &[f64],
        n_cols: usize,
        alpha: &[Self::Refine],
        col: usize,
    ) -> Result<Self::Refine, GprError> {
        let _ = k_storage;
        f64_cross_dot::<M>(kernel, x_train, x_query, n_cols, alpha, col)
    }
}

impl PredictMean for MixedPrecision<ReevaluateKernel> {
    fn column_mean<M: crate::math::KernelMath>(
        kernel: &KernelSpec,
        k_storage: MatRef<'_, Self::Storage>,
        x_train: MatRef<'_, f64>,
        x_query: &[f64],
        n_cols: usize,
        alpha: &[Self::Refine],
        col: usize,
    ) -> Result<Self::Refine, GprError> {
        let _ = k_storage;
        f64_cross_dot::<M>(kernel, x_train, x_query, n_cols, alpha, col)
    }
}

fn f64_cross_dot<M: crate::math::KernelMath>(
    kernel: &KernelSpec,
    x_train: MatRef<'_, f64>,
    x_query: &[f64],
    n_cols: usize,
    alpha: &[f64],
    col: usize,
) -> Result<f64, GprError> {
    let kernel_f64 = kernel.compile();
    let n = x_train.nrows();
    let n_rows = x_query.len() / n_cols;
    let mut row = Mat::<f64>::zeros(1, n_cols);
    for dim in 0..n_cols {
        row[(0, dim)] = x_query[dim * n_rows + col];
    }
    let mut k_col = Mat::<f64>::zeros(n, 1);
    let mut scratch = Mat::<f64>::zeros(n, 1);
    match kernel_f64.coord_mode()? {
        crate::kernel::CoordMode::Points => {
            kernel_f64.apply_cross_points::<M>(
                x_train,
                row.as_ref(),
                k_col.as_mut(),
                scratch.as_mut(),
            )?;
        }
        crate::kernel::CoordMode::Dist | crate::kernel::CoordMode::Either => {
            let mut dist = Mat::<f64>::zeros(n, 1);
            f64::write_cross(x_train, row.as_ref(), dist.as_mut(), &mut []);
            kernel_f64.apply_cross::<M>(dist.as_ref(), k_col.as_mut(), scratch.as_mut())?;
        }
        crate::kernel::CoordMode::Mixed => {
            let mut dist = Mat::<f64>::zeros(n, 1);
            f64::write_cross(x_train, row.as_ref(), dist.as_mut(), &mut []);
            kernel_f64.apply_cross_mixed::<M>(
                dist.as_ref(),
                x_train,
                row.as_ref(),
                k_col.as_mut(),
                scratch.as_mut(),
            )?;
        }
    }
    let mut sum = 0.0;
    for i in 0..n {
        sum += k_col[(i, 0)] * alpha[i];
    }
    Ok(sum)
}

/// Inverse target map. `f32` predictions pass through an `f64` buffer.
pub(crate) trait ScalePrediction: ModelPrecision {
    fn inverse_mean_variance(
        transform: &dyn TargetTransform,
        mean: &mut [Self::Refine],
        variance: &mut [Self::Refine],
    ) -> Result<(), GprError>;

    fn inverse_covariance(
        transform: &dyn TargetTransform,
        covariance: &mut [Self::Refine],
    ) -> Result<(), GprError>;
}

fn inverse_f64_mean_variance(
    transform: &dyn TargetTransform,
    mean: &mut [f64],
    variance: &mut [f64],
) -> Result<(), GprError> {
    transform.inverse_transform_mean(mean)?;
    transform.inverse_transform_variance(variance)
}

impl ScalePrediction for DoublePrecision {
    fn inverse_mean_variance(
        transform: &dyn TargetTransform,
        mean: &mut [Self::Refine],
        variance: &mut [Self::Refine],
    ) -> Result<(), GprError> {
        inverse_f64_mean_variance(transform, mean, variance)
    }

    fn inverse_covariance(
        transform: &dyn TargetTransform,
        covariance: &mut [Self::Refine],
    ) -> Result<(), GprError> {
        transform.inverse_transform_covariance(covariance)
    }
}

impl ScalePrediction for MixedPrecision<PromoteStorage> {
    fn inverse_mean_variance(
        transform: &dyn TargetTransform,
        mean: &mut [Self::Refine],
        variance: &mut [Self::Refine],
    ) -> Result<(), GprError> {
        inverse_f64_mean_variance(transform, mean, variance)
    }

    fn inverse_covariance(
        transform: &dyn TargetTransform,
        covariance: &mut [Self::Refine],
    ) -> Result<(), GprError> {
        transform.inverse_transform_covariance(covariance)
    }
}

impl ScalePrediction for MixedPrecision<ReevaluateKernel> {
    fn inverse_mean_variance(
        transform: &dyn TargetTransform,
        mean: &mut [Self::Refine],
        variance: &mut [Self::Refine],
    ) -> Result<(), GprError> {
        inverse_f64_mean_variance(transform, mean, variance)
    }

    fn inverse_covariance(
        transform: &dyn TargetTransform,
        covariance: &mut [Self::Refine],
    ) -> Result<(), GprError> {
        transform.inverse_transform_covariance(covariance)
    }
}

impl ScalePrediction for SinglePrecision {
    fn inverse_mean_variance(
        transform: &dyn TargetTransform,
        mean: &mut [Self::Refine],
        variance: &mut [Self::Refine],
    ) -> Result<(), GprError> {
        let mut mean64: Vec<f64> = mean.iter().copied().map(f32::to_f64).collect();
        let mut var64: Vec<f64> = variance.iter().copied().map(f32::to_f64).collect();
        inverse_f64_mean_variance(transform, &mut mean64, &mut var64)?;
        for (slot, value) in mean.iter_mut().zip(mean64) {
            *slot = f32::from_f64(value);
        }
        for (slot, value) in variance.iter_mut().zip(var64) {
            *slot = f32::from_f64(value);
        }
        Ok(())
    }

    fn inverse_covariance(
        transform: &dyn TargetTransform,
        covariance: &mut [Self::Refine],
    ) -> Result<(), GprError> {
        let mut buf: Vec<f64> = covariance.iter().copied().map(f32::to_f64).collect();
        transform.inverse_transform_covariance(&mut buf)?;
        for (slot, value) in covariance.iter_mut().zip(buf) {
            *slot = f32::from_f64(value);
        }
        Ok(())
    }
}

/// Training factor view. Only [`DoublePrecision`] reads a mapped `f64` `L`.
pub(crate) trait ViewFactor: ModelPrecision {
    fn view_factor<'a>(
        mapped_l: Option<MatRef<'a, f64>>,
        workspace_l: MatRef<'a, Self::Storage>,
    ) -> MatRef<'a, Self::Storage>;

    fn copy_mapped_l(src: MatRef<'_, f64>, dest: MatMut<'_, Self::Storage>);
}

impl ViewFactor for DoublePrecision {
    fn view_factor<'a>(
        mapped_l: Option<MatRef<'a, f64>>,
        workspace_l: MatRef<'a, Self::Storage>,
    ) -> MatRef<'a, Self::Storage> {
        match mapped_l {
            Some(mapped) => mapped,
            None => workspace_l,
        }
    }

    fn copy_mapped_l(src: MatRef<'_, f64>, mut dest: MatMut<'_, Self::Storage>) {
        let n = src.nrows().min(dest.nrows());
        let cols = src.ncols().min(dest.ncols());
        for col in 0..cols {
            for row in 0..n {
                dest[(row, col)] = src[(row, col)];
            }
        }
    }
}

impl ViewFactor for SinglePrecision {
    fn view_factor<'a>(
        _mapped_l: Option<MatRef<'a, f64>>,
        workspace_l: MatRef<'a, Self::Storage>,
    ) -> MatRef<'a, Self::Storage> {
        workspace_l
    }

    fn copy_mapped_l(_src: MatRef<'_, f64>, _dest: MatMut<'_, Self::Storage>) {}
}

impl ViewFactor for MixedPrecision<PromoteStorage> {
    fn view_factor<'a>(
        _mapped_l: Option<MatRef<'a, f64>>,
        workspace_l: MatRef<'a, Self::Storage>,
    ) -> MatRef<'a, Self::Storage> {
        workspace_l
    }

    fn copy_mapped_l(_src: MatRef<'_, f64>, _dest: MatMut<'_, Self::Storage>) {}
}

impl ViewFactor for MixedPrecision<ReevaluateKernel> {
    fn view_factor<'a>(
        _mapped_l: Option<MatRef<'a, f64>>,
        workspace_l: MatRef<'a, Self::Storage>,
    ) -> MatRef<'a, Self::Storage> {
        workspace_l
    }

    fn copy_mapped_l(_src: MatRef<'_, f64>, _dest: MatMut<'_, Self::Storage>) {}
}

/// Storage, predict-`α`, and target scaling for one precision policy.
pub(crate) trait GpScalar:
    ModelPrecision + PublishPredictAlpha + PredictMean + ScalePrediction + ViewFactor
{
}

impl GpScalar for DoublePrecision {}
impl GpScalar for SinglePrecision {}
impl GpScalar for MixedPrecision<PromoteStorage> {}
impl GpScalar for MixedPrecision<ReevaluateKernel> {}

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

    #[test]
    fn single_precision_is_f32() {
        let storage: <super::SinglePrecision as PrecisionPolicy>::Storage = 0.0;
        let refine: <super::SinglePrecision as PrecisionPolicy>::Refine = 0.0;
        let _ = (storage, refine);
        assert_send_sync::<super::SinglePrecision>();
    }

    #[test]
    fn mixed_precision_stores_f32_and_refines_f64() {
        let storage: <super::MixedPrecision as PrecisionPolicy>::Storage = 0.0;
        let refine: <super::MixedPrecision as PrecisionPolicy>::Refine = 0.0;
        let fresh: <super::MixedPrecision<super::ReevaluateKernel> as PrecisionPolicy>::Refine =
            0.0;
        let _ = (storage, refine, fresh);
        assert_send_sync::<super::MixedPrecision>();
        assert_send_sync::<super::MixedPrecision<super::ReevaluateKernel>>();
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
        let alpha = refine::<crate::math::Accurate, R>(&k32, &k64, x, y, noise).expect("refine");
        let truth = f64_alpha::<crate::math::Accurate>(&k64, x, y, noise).expect("f64");
        let rel = rel_inf(&alpha, &truth);
        let bar = 10.0 * y.len() as f64 * f64::from(f32::EPSILON);
        assert!(rel < bar, "relative {rel} bar {bar}");
    }

    fn kappa(kernel: &crate::kernel::CompiledKernel<f64>, x: MatRef<'_, f64>, noise: f64) -> f64 {
        let n = x.nrows();
        let mut a = Mat::<f64>::zeros(n, n);
        let mut scratch = Mat::<f64>::zeros(n, n);
        kernel
            .apply_points::<crate::math::Accurate>(x, a.as_mut(), Triangle::Lower, scratch.as_mut())
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
            refine::<crate::math::Accurate, PromoteStorage>(&k32, &k64, x.as_ref(), &y, 0.1)
                .expect("promote");
        });
        median("reevaluate", &|| {
            refine::<crate::math::Accurate, ReevaluateKernel>(&k32, &k64, x.as_ref(), &y, 0.1)
                .expect("reevaluate");
        });
    }

    use crate::kernel::KernelScalar;
    use crate::{Fixed, GaussianLikelihood, Gpr, MixedPrecision, Sgpr, SinglePrecision, Svgp};

    fn must<T>(result: Result<T, crate::GprError>) -> T {
        match result {
            Ok(value) => value,
            Err(err) => panic!("gpr call failed: {err}"),
        }
    }

    fn likelihood_at(noise: f64) -> GaussianLikelihood {
        if crate::param::Interval::DEFAULT_POSITIVE.contains(noise) {
            return must(GaussianLikelihood::new(noise));
        }
        let interval = match crate::Interval::new(1.0e-12, 1.0e5) {
            Ok(interval) => interval,
            Err(err) => panic!("noise interval: {err}"),
        };
        let mut likelihood = match GaussianLikelihood::new(0.1) {
            Ok(likelihood) => match likelihood.with_bounds(interval) {
                Ok(likelihood) => likelihood,
                Err(err) => panic!("noise bounds: {err}"),
            },
            Err(err) => panic!("gpr call failed: {err}"),
        };
        must(likelihood.set_params(&[noise.ln()]));
        likelihood
    }

    fn digit_tol(n: usize) -> f64 {
        10.0 * n as f64 * f64::from(f32::EPSILON)
    }

    fn near(got: f64, expect: f64, n: usize) -> bool {
        let tol = digit_tol(n);
        let err = (got - expect).abs();
        if expect.abs() < tol {
            err < tol
        } else {
            err / expect.abs() < tol
        }
    }

    fn assert_near(got: f64, expect: f64, n: usize) {
        assert!(
            near(got, expect, n),
            "got {got} expect {expect} tol {}",
            digit_tol(n)
        );
    }

    fn pack_col(x: &Mat<f64>) -> Vec<f64> {
        (0..x.nrows()).map(|i| x[(i, 0)]).collect()
    }

    fn inducing8() -> Vec<f64> {
        (0..8).map(|i| i as f64 / 7.0).collect()
    }

    fn queries() -> [f64; 2] {
        [0.25, 0.75]
    }

    fn factor_exact<P>(
        x: &[f64],
        y: &[f64],
        ell: f64,
        noise: f64,
    ) -> crate::FittedGpr<
        Fixed,
        crate::FullRecompute,
        crate::CachedDistances,
        crate::RetainCholesky,
        crate::Accurate,
        P,
    >
    where
        P: crate::precision::GpScalar,
        crate::kernel::CompiledKernel<P::Storage>: crate::kernel::GramKernel<T = P::Storage>,
    {
        let kernel = KernelSpec::from(must(RbfKernel::new(ell)));
        let likelihood = likelihood_at(noise);
        let n = y.len();
        must(
            Gpr::new(kernel, likelihood)
                .with_optimizer(Fixed)
                .with_precision::<P>()
                .factor(x, n, 1, y)
                .map_err(|(_, err)| err),
        )
    }

    fn assert_predict_pair<T: KernelScalar, U: KernelScalar>(
        got_mean: &[T],
        got_var: &[T],
        exp_mean: &[U],
        exp_var: &[U],
        n: usize,
        digits: bool,
    ) {
        for (got, expect) in got_mean.iter().zip(exp_mean) {
            let g = got.to_f64();
            let e = expect.to_f64();
            if digits {
                assert!(
                    near(g, e, n),
                    "mean got {g} expect {e} tol {}",
                    digit_tol(n)
                );
            } else {
                assert!(g.is_finite(), "mean {g}");
            }
        }
        for (got, expect) in got_var.iter().zip(exp_var) {
            let g = got.to_f64();
            let e = expect.to_f64();
            if digits {
                assert!(near(g, e, n), "var got {g} expect {e} tol {}", digit_tol(n));
            } else {
                assert!(g.is_finite(), "variance {g}");
            }
        }
    }

    fn exact_digits(n: usize) {
        let (x, y) = forrester(n);
        let packed = pack_col(&x);
        let f64_model = factor_exact::<DoublePrecision>(&packed, &y, 1.0, 0.1);
        let single = factor_exact::<SinglePrecision>(&packed, &y, 1.0, 0.1);
        let promote = factor_exact::<MixedPrecision<PromoteStorage>>(&packed, &y, 1.0, 0.1);
        let fresh = factor_exact::<MixedPrecision<ReevaluateKernel>>(&packed, &y, 1.0, 0.1);
        let q = queries();
        let truth = must(f64_model.predict(&q, 2, 1));
        let s = must(single.predict(&q, 2, 1));
        let p = must(promote.predict(&q, 2, 1));
        let r = must(fresh.predict(&q, 2, 1));
        assert_predict_pair(&s.mean, &s.variance, &truth.mean, &truth.variance, n, true);
        assert_predict_pair(&p.mean, &p.variance, &truth.mean, &truth.variance, n, true);
        assert_predict_pair(&r.mean, &r.variance, &truth.mean, &truth.variance, n, true);
        let cov_t = must(f64_model.predict_covariance(&q, 2, 1));
        let cov_s = must(single.predict_covariance(&q, 2, 1));
        let cov_p = must(promote.predict_covariance(&q, 2, 1));
        let cov_r = must(fresh.predict_covariance(&q, 2, 1));
        for (got, expect) in cov_s.covariance.iter().zip(&cov_t.covariance) {
            assert_near(got.to_f64(), expect.to_f64(), n);
        }
        for (got, expect) in cov_p.covariance.iter().zip(&cov_t.covariance) {
            assert_near(got.to_f64(), expect.to_f64(), n);
        }
        for (got, expect) in cov_r.covariance.iter().zip(&cov_t.covariance) {
            assert_near(got.to_f64(), expect.to_f64(), n);
        }
        let loo_t = must(f64_model.loo_predict());
        let loo_s = must(single.loo_predict());
        let loo_p = must(promote.loo_predict());
        let loo_r = must(fresh.loo_predict());
        assert_predict_pair(
            &loo_s.mean,
            &loo_s.variance,
            &loo_t.mean,
            &loo_t.variance,
            n,
            true,
        );
        assert_predict_pair(
            &loo_p.mean,
            &loo_p.variance,
            &loo_t.mean,
            &loo_t.variance,
            n,
            true,
        );
        assert_predict_pair(
            &loo_r.mean,
            &loo_r.variance,
            &loo_t.mean,
            &loo_t.variance,
            n,
            true,
        );
        let draws = must(single.sample(&q, 2, 1, 3, 7));
        assert_eq!(draws.len(), 6);
        assert!(draws.iter().all(|v| v.is_finite()));
        let draws_m = must(promote.sample(&q, 2, 1, 3, 7));
        assert_eq!(draws_m.len(), 6);
        assert!(draws_m.iter().all(|v| v.is_finite()));
        let draws_r = must(fresh.sample(&q, 2, 1, 3, 7));
        assert_eq!(draws_r.len(), 6);
        assert!(draws_r.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn single_and_mixed_predict_match_f64_digits() {
        exact_digits(256);
    }

    #[test]
    fn single_and_mixed_predict_match_f64_digits_n1024() {
        exact_digits(1024);
    }

    #[test]
    fn ill_conditioned_single_is_finite_and_mixed_mean_hits_digits() {
        let (x, y) = forrester(256);
        let packed = pack_col(&x);
        let n = y.len();
        let noise = ILL_NOISE;
        let f64_model = factor_exact::<DoublePrecision>(&packed, &y, ILL_LENGTHSCALE, noise);
        let single = factor_exact::<SinglePrecision>(&packed, &y, ILL_LENGTHSCALE, noise);
        let promote =
            factor_exact::<MixedPrecision<PromoteStorage>>(&packed, &y, ILL_LENGTHSCALE, noise);
        let fresh =
            factor_exact::<MixedPrecision<ReevaluateKernel>>(&packed, &y, ILL_LENGTHSCALE, noise);
        let q = queries();
        let truth = must(f64_model.predict(&q, 2, 1));
        let s = must(single.predict(&q, 2, 1));
        let p = must(promote.predict(&q, 2, 1));
        let r = must(fresh.predict(&q, 2, 1));
        assert_predict_pair(&s.mean, &s.variance, &truth.mean, &truth.variance, n, false);
        assert_predict_pair(&p.mean, &p.variance, &truth.mean, &truth.variance, n, false);
        assert_predict_pair(&r.mean, &r.variance, &truth.mean, &truth.variance, n, false);
        for (got, expect) in p.mean.iter().zip(&truth.mean) {
            assert_near(got.to_f64(), expect.to_f64(), n);
        }
        for (got, expect) in r.mean.iter().zip(&truth.mean) {
            assert_near(got.to_f64(), expect.to_f64(), n);
        }
        let cov_s = must(single.predict_covariance(&q, 2, 1));
        assert!(cov_s.covariance.iter().all(|v| v.is_finite()));
        let cov_p = must(promote.predict_covariance(&q, 2, 1));
        assert!(cov_p.covariance.iter().all(|v| v.is_finite()));
    }

    fn factor_sgpr<P>(
        x: &[f64],
        y: &[f64],
        ell: f64,
        noise: f64,
    ) -> crate::FittedSgpr<Fixed, crate::FixedInducing, crate::Accurate, P>
    where
        P: crate::precision::GpScalar
            + crate::sgpr::factor::MeanDot
            + crate::sgpr::factor::PublishSgprWeights,
        crate::kernel::CompiledKernel<P::Storage>: crate::kernel::GramKernel<T = P::Storage>,
    {
        let kernel = KernelSpec::from(must(RbfKernel::new(ell)));
        let likelihood = likelihood_at(noise);
        let z = inducing8();
        must(
            Sgpr::new(kernel, likelihood)
                .with_optimizer(Fixed)
                .with_precision::<P>()
                .factor(x, y.len(), 1, y, &z, 8)
                .map_err(|(_, err)| err),
        )
    }

    fn sgpr_case(n: usize, ell: f64, noise: f64, mean_digits: bool, var_digits: bool) {
        let (x, y) = forrester(n);
        let packed = pack_col(&x);
        let truth = factor_sgpr::<DoublePrecision>(&packed, &y, ell, noise);
        let single = factor_sgpr::<SinglePrecision>(&packed, &y, ell, noise);
        let promote = factor_sgpr::<MixedPrecision<PromoteStorage>>(&packed, &y, ell, noise);
        let fresh = factor_sgpr::<MixedPrecision<ReevaluateKernel>>(&packed, &y, ell, noise);
        let q = queries();
        let t = must(truth.predict(&q, 2, 1));
        let s = must(single.predict(&q, 2, 1));
        let p = must(promote.predict(&q, 2, 1));
        let r = must(fresh.predict(&q, 2, 1));
        let single_mean_digits = mean_digits && ell < ILL_LENGTHSCALE;
        assert_predict_pair(
            &s.mean,
            &s.variance,
            &t.mean,
            &t.variance,
            n,
            single_mean_digits,
        );
        assert_predict_pair(&p.mean, &p.variance, &t.mean, &t.variance, n, var_digits);
        assert_predict_pair(&r.mean, &r.variance, &t.mean, &t.variance, n, var_digits);
        if !var_digits {
            for (got, expect) in p.mean.iter().zip(&t.mean) {
                assert_near(got.to_f64(), expect.to_f64(), n);
            }
            for (got, expect) in r.mean.iter().zip(&t.mean) {
                assert_near(got.to_f64(), expect.to_f64(), n);
            }
        }
    }

    #[test]
    fn sgpr_precisions_match_f64_predict() {
        sgpr_case(256, 1.0, 0.1, true, true);
        sgpr_case(256, ILL_LENGTHSCALE, ILL_NOISE, false, false);
    }

    #[test]
    fn sgpr_precisions_match_f64_predict_n1024() {
        sgpr_case(1024, 1.0, 0.1, true, true);
    }

    fn factor_svgp<P>(
        x: &[f64],
        y: &[f64],
        ell: f64,
        noise: f64,
    ) -> crate::FittedSvgp<crate::Accurate, P>
    where
        P: crate::precision::GpScalar + crate::svgp::factor::SvgpMean,
        crate::kernel::CompiledKernel<P::Storage>: crate::kernel::GramKernel<T = P::Storage>,
    {
        let kernel = KernelSpec::from(must(RbfKernel::new(ell)));
        let likelihood = likelihood_at(noise);
        let z = inducing8();
        must(
            Svgp::new(kernel, likelihood)
                .with_precision::<P>()
                .factor(x, y.len(), 1, y, &z, 8)
                .map_err(|(_, err)| err),
        )
    }

    #[test]
    fn svgp_precisions_match_f64_predict() {
        for (n, ell, noise, var_digits) in [
            (256usize, 1.0, 0.1, true),
            (1024usize, 1.0, 0.1, true),
            (256usize, ILL_LENGTHSCALE, ILL_NOISE, false),
        ] {
            let (x, y) = forrester(n);
            let packed = pack_col(&x);
            let truth = factor_svgp::<DoublePrecision>(&packed, &y, ell, noise);
            let single = factor_svgp::<SinglePrecision>(&packed, &y, ell, noise);
            let promote = factor_svgp::<MixedPrecision<PromoteStorage>>(&packed, &y, ell, noise);
            let fresh = factor_svgp::<MixedPrecision<ReevaluateKernel>>(&packed, &y, ell, noise);
            let q = queries();
            let t = must(truth.predict(&q, 2, 1));
            let s = must(single.predict(&q, 2, 1));
            let p = must(promote.predict(&q, 2, 1));
            let r = must(fresh.predict(&q, 2, 1));
            assert_predict_pair(&s.mean, &s.variance, &t.mean, &t.variance, n, var_digits);
            assert_predict_pair(&p.mean, &p.variance, &t.mean, &t.variance, n, var_digits);
            assert_predict_pair(&r.mean, &r.variance, &t.mean, &t.variance, n, var_digits);
            if !var_digits {
                for (got, expect) in p.mean.iter().zip(&t.mean) {
                    assert_near(got.to_f64(), expect.to_f64(), n);
                }
                for (got, expect) in r.mean.iter().zip(&t.mean) {
                    assert_near(got.to_f64(), expect.to_f64(), n);
                }
            }
        }
    }

    fn online_exact_round<P>(
        packed: &[f64],
        y: &[f64],
        ell: f64,
        noise: f64,
    ) -> crate::OnlineGpr<
        Fixed,
        crate::FullRecompute,
        crate::CachedDistances,
        crate::RetainCholesky,
        crate::Accurate,
        P,
    >
    where
        P: crate::precision::GpScalar,
        crate::kernel::CompiledKernel<P::Storage>: crate::kernel::GramKernel<T = P::Storage>,
    {
        let mut online = must(factor_exact::<P>(packed, y, ell, noise).into_online());
        must(online.insert(&[0.33], 0.2));
        must(online.delete(online.point_ids()[0]));
        online
    }

    #[test]
    fn online_exact_insert_delete_matches_f64() {
        for n in [256usize, 1024] {
            let (x, y) = forrester(n);
            let packed = pack_col(&x);
            let base = online_exact_round::<DoublePrecision>(&packed, &y, 1.0, 0.1);
            let single = online_exact_round::<SinglePrecision>(&packed, &y, 1.0, 0.1);
            let promote =
                online_exact_round::<MixedPrecision<PromoteStorage>>(&packed, &y, 1.0, 0.1);
            let fresh =
                online_exact_round::<MixedPrecision<ReevaluateKernel>>(&packed, &y, 1.0, 0.1);
            let q = queries();
            let t = must(base.predict(&q, 2, 1));
            let s = must(single.predict(&q, 2, 1));
            let p = must(promote.predict(&q, 2, 1));
            let r = must(fresh.predict(&q, 2, 1));
            assert_predict_pair(&s.mean, &s.variance, &t.mean, &t.variance, n, true);
            assert_predict_pair(&p.mean, &p.variance, &t.mean, &t.variance, n, true);
            assert_predict_pair(&r.mean, &r.variance, &t.mean, &t.variance, n, true);
        }
        let (x, y) = forrester(256);
        let packed = pack_col(&x);
        let n = y.len();
        let noise = ILL_NOISE;
        let base = online_exact_round::<DoublePrecision>(&packed, &y, ILL_LENGTHSCALE, noise);
        let single = online_exact_round::<SinglePrecision>(&packed, &y, ILL_LENGTHSCALE, noise);
        let promote = online_exact_round::<MixedPrecision<PromoteStorage>>(
            &packed,
            &y,
            ILL_LENGTHSCALE,
            noise,
        );
        let fresh = online_exact_round::<MixedPrecision<ReevaluateKernel>>(
            &packed,
            &y,
            ILL_LENGTHSCALE,
            noise,
        );
        let q = queries();
        let t = must(base.predict(&q, 2, 1));
        let s = must(single.predict(&q, 2, 1));
        let p = must(promote.predict(&q, 2, 1));
        let r = must(fresh.predict(&q, 2, 1));
        assert_predict_pair(&s.mean, &s.variance, &t.mean, &t.variance, n, false);
        assert_predict_pair(&p.mean, &p.variance, &t.mean, &t.variance, n, false);
        assert_predict_pair(&r.mean, &r.variance, &t.mean, &t.variance, n, false);
        for (got, expect) in p.mean.iter().zip(&t.mean) {
            assert_near(got.to_f64(), expect.to_f64(), n);
        }
        for (got, expect) in r.mean.iter().zip(&t.mean) {
            assert_near(got.to_f64(), expect.to_f64(), n);
        }
    }

    fn online_sgpr_round<P>(
        packed: &[f64],
        y: &[f64],
        ell: f64,
        noise: f64,
    ) -> crate::OnlineSgpr<Fixed, crate::Accurate, P>
    where
        P: crate::precision::GpScalar
            + crate::sgpr::factor::MeanDot
            + crate::sgpr::factor::PublishSgprWeights,
        crate::kernel::CompiledKernel<P::Storage>: crate::kernel::GramKernel<T = P::Storage>,
    {
        let mut online = factor_sgpr::<P>(packed, y, ell, noise).into_online();
        must(online.insert(&[0.33], 0.2));
        must(online.delete(online.point_ids()[0]));
        online
    }

    #[test]
    fn online_sgpr_insert_delete_matches_f64() {
        let (x, y) = forrester(256);
        let packed = pack_col(&x);
        let n = y.len();
        let base = online_sgpr_round::<DoublePrecision>(&packed, &y, 1.0, 0.1);
        let single = online_sgpr_round::<SinglePrecision>(&packed, &y, 1.0, 0.1);
        let promote = online_sgpr_round::<MixedPrecision<PromoteStorage>>(&packed, &y, 1.0, 0.1);
        let fresh = online_sgpr_round::<MixedPrecision<ReevaluateKernel>>(&packed, &y, 1.0, 0.1);
        let q = queries();
        let t = must(base.predict(&q, 2, 1));
        let s = must(single.predict(&q, 2, 1));
        let p = must(promote.predict(&q, 2, 1));
        let r = must(fresh.predict(&q, 2, 1));
        assert_predict_pair(&s.mean, &s.variance, &t.mean, &t.variance, n, true);
        assert_predict_pair(&p.mean, &p.variance, &t.mean, &t.variance, n, true);
        assert_predict_pair(&r.mean, &r.variance, &t.mean, &t.variance, n, true);

        let noise = ILL_NOISE;
        let base = online_sgpr_round::<DoublePrecision>(&packed, &y, ILL_LENGTHSCALE, noise);
        let single = online_sgpr_round::<SinglePrecision>(&packed, &y, ILL_LENGTHSCALE, noise);
        let promote = online_sgpr_round::<MixedPrecision<PromoteStorage>>(
            &packed,
            &y,
            ILL_LENGTHSCALE,
            noise,
        );
        let fresh = online_sgpr_round::<MixedPrecision<ReevaluateKernel>>(
            &packed,
            &y,
            ILL_LENGTHSCALE,
            noise,
        );
        let t = must(base.predict(&q, 2, 1));
        let s = must(single.predict(&q, 2, 1));
        let p = must(promote.predict(&q, 2, 1));
        let r = must(fresh.predict(&q, 2, 1));
        assert_predict_pair(&s.mean, &s.variance, &t.mean, &t.variance, n, false);
        assert_predict_pair(&p.mean, &p.variance, &t.mean, &t.variance, n, false);
        assert_predict_pair(&r.mean, &r.variance, &t.mean, &t.variance, n, false);
        for (got, expect) in p.mean.iter().zip(&t.mean) {
            assert_near(got.to_f64(), expect.to_f64(), n);
        }
        for (got, expect) in r.mean.iter().zip(&t.mean) {
            assert_near(got.to_f64(), expect.to_f64(), n);
        }
    }

    #[test]
    fn single_and_mixed_gradient_matches_own_nlml() {
        let n = 8usize;
        let (x, y) = forrester(n);
        let packed = pack_col(&x);
        check_grad::<SinglePrecision>(&packed, &y);
        check_grad::<MixedPrecision<PromoteStorage>>(&packed, &y);
        check_grad::<MixedPrecision<ReevaluateKernel>>(&packed, &y);
    }

    fn check_grad<P>(x: &[f64], y: &[f64])
    where
        P: crate::precision::GpScalar,
        crate::kernel::CompiledKernel<P::Storage>: crate::kernel::GramKernel<T = P::Storage>,
    {
        let mut fitted = factor_exact::<P>(x, y, 1.0, 0.1);
        let mut params = [0.0; 2];
        must(fitted.get_params(&mut params));
        let mut grad = [0.0; 2];
        let _ = must(fitted.value_and_gradient_into(&params, &mut grad));
        let step = 1.0e-3;
        for i in 0..2 {
            let mut up = params;
            let mut down = params;
            up[i] += step;
            down[i] -= step;
            must(fitted.set_params(&up));
            let plus = must(fitted.neg_log_marginal_likelihood());
            must(fitted.set_params(&down));
            let minus = must(fitted.neg_log_marginal_likelihood());
            let fd = (plus - minus) / (2.0 * step);
            let scale = fd.abs().max(1.0);
            let err = (grad[i] - fd).abs() / scale;
            assert!(err < 2.0e-3, "param {i} grad {} fd {fd} rel {err}", grad[i]);
        }
    }

    #[test]
    fn single_and_mixed_hessian_matches_own_gradient() {
        let n = 8usize;
        let (x, y) = forrester(n);
        let packed = pack_col(&x);
        check_hess_exact::<SinglePrecision>(&packed, &y);
        check_hess_exact::<MixedPrecision<PromoteStorage>>(&packed, &y);
        check_hess_exact::<MixedPrecision<ReevaluateKernel>>(&packed, &y);
        check_hess_sgpr::<SinglePrecision>(&packed, &y);
        check_hess_sgpr::<MixedPrecision<PromoteStorage>>(&packed, &y);
        check_hess_sgpr::<MixedPrecision<ReevaluateKernel>>(&packed, &y);
    }

    fn check_hess_exact<P>(x: &[f64], y: &[f64])
    where
        P: crate::precision::GpScalar,
        crate::kernel::CompiledKernel<P::Storage>: crate::kernel::GramKernel<T = P::Storage>,
    {
        let mut fitted = factor_exact::<P>(x, y, 1.0, 0.1);
        let mut params = [0.0; 2];
        must(fitted.get_params(&mut params));
        let mut hess = [0.0; 4];
        must(fitted.hessian_into(&params, &mut hess));
        let step = 1.0e-3;
        for j in 0..2 {
            let mut up = params;
            let mut down = params;
            up[j] += step;
            down[j] -= step;
            let mut grad_up = [0.0; 2];
            let mut grad_down = [0.0; 2];
            must(fitted.set_params(&up));
            let _ = must(fitted.value_and_gradient_into(&up, &mut grad_up));
            must(fitted.set_params(&down));
            let _ = must(fitted.value_and_gradient_into(&down, &mut grad_down));
            for i in 0..2 {
                let fd = (grad_up[i] - grad_down[i]) / (2.0 * step);
                let analytic = hess[i * 2 + j];
                let scale = fd.abs().max(1.0);
                let err = (analytic - fd).abs() / scale;
                assert!(
                    err < 5.0e-2,
                    "exact hess[{i},{j}] {analytic} fd {fd} rel {err}"
                );
            }
        }
    }

    fn check_hess_sgpr<P>(x: &[f64], y: &[f64])
    where
        P: crate::precision::GpScalar
            + crate::sgpr::factor::MeanDot
            + crate::sgpr::factor::PublishSgprWeights,
        crate::kernel::CompiledKernel<P::Storage>: crate::kernel::GramKernel<T = P::Storage>,
    {
        let mut fitted = factor_sgpr::<P>(x, y, 1.0, 0.1);
        let mut params = [0.0; 2];
        must(fitted.get_params(&mut params));
        let mut hess = [0.0; 4];
        must(fitted.hessian_into(&params, &mut hess));
        let step = 1.0e-3;
        for j in 0..2 {
            let mut up = params;
            let mut down = params;
            up[j] += step;
            down[j] -= step;
            let mut grad_up = [0.0; 2];
            let mut grad_down = [0.0; 2];
            must(fitted.set_params(&up));
            let _ = must(fitted.value_and_gradient_into(&up, &mut grad_up));
            must(fitted.set_params(&down));
            let _ = must(fitted.value_and_gradient_into(&down, &mut grad_down));
            for i in 0..2 {
                let fd = (grad_up[i] - grad_down[i]) / (2.0 * step);
                let analytic = hess[i * 2 + j];
                let scale = fd.abs().max(1.0);
                let err = (analytic - fd).abs() / scale;
                assert!(
                    err < 5.0e-2,
                    "sgpr hess[{i},{j}] {analytic} fd {fd} rel {err}"
                );
            }
        }
    }
}
