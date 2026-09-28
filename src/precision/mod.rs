//! Storage scalar and the mixed-precision residual solver.
//!
//! Omitted model precision is [`DoublePrecision`] (`f64`). [`SinglePrecision`]
//! runs the same procedures in `f32` and keeps the factorization as-is.
//! [`MixedPrecision`] factors in `f32` and refines the predict `α` in `f64`.
//! The residual type parameter exists only on [`MixedPrecision`]. There is no
//! flag and no alias that picks a residual formula.

use faer::{Mat, MatMut, MatRef};

use crate::error::GprError;
use crate::kernel::{KernelScalar, KernelSpec};
use crate::transform::TargetTransform;

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

mod refine;

pub(crate) use refine::{COLUMN_BLOCK, RefineSystem, refine};
pub use refine::{StoredFactor, TrainSystem};

mod residual_seal {
    pub trait Sealed {
        /// `true` for [`super::PromoteStorage`] (residual from the stored `f32`
        /// matrix), `false` for [`super::ReevaluateKernel`] (fresh `f64` kernel).
        const READS_STORAGE: bool;
    }
}

/// How [`MixedPrecision`] builds the refinement residual `r = y − Aα`.
///
/// The only implementations are [`PromoteStorage`] and [`ReevaluateKernel`].
pub trait ResidualFormula: residual_seal::Sealed + Copy + Send + Sync + 'static {}

/// Residual `r = y − Aα` from the saved `f32` matrix promoted to `f64`.
///
/// This is the residual when [`MixedPrecision`] omits its type parameter.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PromoteStorage;

/// Residual `r = y − Aα` from a fresh `f64` kernel evaluation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ReevaluateKernel;

impl residual_seal::Sealed for PromoteStorage {
    const READS_STORAGE: bool = true;
}
impl residual_seal::Sealed for ReevaluateKernel {
    const READS_STORAGE: bool = false;
}

impl ResidualFormula for PromoteStorage {}
impl ResidualFormula for ReevaluateKernel {}

/// Which precision a persist directory records. Absent on disk means double.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PersistKind {
    Double,
    Single,
    MixedPromote,
    MixedReevaluate,
}

/// Every operation that differs by model precision, in one sealed trait.
///
/// Three impls: [`DoublePrecision`] and [`SinglePrecision`] read the storage
/// factor as-is; [`MixedPrecision`] refines in `f64` with the shared
/// [`refine`] loop, whatever its [`ResidualFormula`]. `pub` in a private
/// module, so it is nameable only inside the crate.
pub trait ModelPrecision: PrecisionPolicy + Copy + Send + Sync + 'static {
    /// `true` when predict weights are refined in `f64` from an `f32` factor.
    const REFINES_IN_F64: bool;

    fn persist_kind() -> PersistKind;

    /// Exact: stores predict `α` from the stored factor (copy, or refine).
    fn publish_predict_alpha<M: crate::math::KernelMath>(
        sys: &TrainSystem<'_, Self::Storage>,
        alpha: &mut Vec<Self::Refine>,
    ) -> Result<(), GprError>;

    /// Exact: writes `k_*ᵀ α` for every query column into `out`.
    ///
    /// `x_query` is the transformed query, column-major `out.len() × n_cols`.
    fn predict_means<M: crate::math::KernelMath>(
        kernel: &KernelSpec,
        k_storage: MatRef<'_, Self::Storage>,
        x_train: MatRef<'_, f64>,
        x_query: &[f64],
        n_cols: usize,
        alpha: &[Self::Refine],
        out: &mut [Self::Refine],
    ) -> Result<(), GprError>;

    /// Maps predicted mean and variance back through the target transform.
    fn inverse_mean_variance(
        transform: &dyn TargetTransform,
        mean: &mut [Self::Refine],
        variance: &mut [Self::Refine],
    ) -> Result<(), GprError>;

    /// Maps a predicted covariance back through the target transform.
    fn inverse_covariance(
        transform: &dyn TargetTransform,
        covariance: &mut [Self::Refine],
    ) -> Result<(), GprError>;

    /// The training factor to read. Only [`DoublePrecision`] reads a mapped `L`.
    fn view_factor<'a>(
        mapped_l: Option<MatRef<'a, f64>>,
        workspace_l: MatRef<'a, Self::Storage>,
    ) -> MatRef<'a, Self::Storage>;

    /// Copies a mapped `f64` `L` into the workspace. A no-op off [`DoublePrecision`].
    fn copy_mapped_l(src: MatRef<'_, f64>, dest: MatMut<'_, Self::Storage>);

    /// Sgpr: predictive mean from one storage kernel column and the weights.
    fn mean_dot(column: &[Self::Storage], weights: &[Self::Refine]) -> Self::Refine;

    /// Sgpr: predict weights after a factor or an online update.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::CholeskyFailed`] when the `f64` fallback factor
    /// is not positive definite. Refinement does not add jitter.
    #[allow(clippy::too_many_arguments)]
    fn publish_weights<M: crate::math::KernelMath>(
        kernel: &KernelSpec,
        a: MatRef<'_, Self::Storage>,
        b_l: MatRef<'_, Self::Storage>,
        w: &[Self::Storage],
        x: &[f64],
        y: &[f64],
        z: &[f64],
        noise: f64,
        n: usize,
        m: usize,
        d: usize,
    ) -> Result<Vec<Self::Refine>, GprError>;

    /// Svgp: mean from the storage triangular solve (refined for mixed).
    fn mean_from_factor<M: crate::math::KernelMath>(
        kernel: &KernelSpec,
        z_obs: &[f64],
        query: &[f64],
        k_mm_l: MatRef<'_, Self::Storage>,
        solved: &[Self::Storage],
        rhs: &[Self::Storage],
        q_mean: &[f64],
    ) -> Result<Self::Refine, GprError>;
}

fn copy_to_refine<P: ModelPrecision>(values: &[P::Storage], out: &mut Vec<P::Refine>) {
    if out.len() != values.len() {
        out.resize(values.len(), P::Refine::from_f64(0.0));
    }
    for (slot, &value) in out.iter_mut().zip(values.iter()) {
        *slot = P::Refine::from_f64(value.to_f64());
    }
}

fn storage_means<P: ModelPrecision>(
    k_storage: MatRef<'_, P::Storage>,
    alpha: &[P::Refine],
    out: &mut [P::Refine],
) {
    for (col, slot) in out.iter_mut().enumerate() {
        let mut sum = 0.0f64;
        for (row, &weight) in alpha.iter().enumerate() {
            sum += k_storage[(row, col)].to_f64() * weight.to_f64();
        }
        *slot = P::Refine::from_f64(sum);
    }
}

fn inverse_f64_mean_variance(
    transform: &dyn TargetTransform,
    mean: &mut [f64],
    variance: &mut [f64],
) -> Result<(), GprError> {
    transform.inverse_transform_mean(mean)?;
    transform.inverse_transform_variance(variance)
}

impl ModelPrecision for DoublePrecision {
    const REFINES_IN_F64: bool = false;

    fn persist_kind() -> PersistKind {
        PersistKind::Double
    }

    fn publish_predict_alpha<M: crate::math::KernelMath>(
        sys: &TrainSystem<'_, Self::Storage>,
        alpha: &mut Vec<Self::Refine>,
    ) -> Result<(), GprError> {
        copy_to_refine::<Self>(sys.factor_alpha, alpha);
        Ok(())
    }

    fn predict_means<M: crate::math::KernelMath>(
        _kernel: &KernelSpec,
        k_storage: MatRef<'_, Self::Storage>,
        _x_train: MatRef<'_, f64>,
        _x_query: &[f64],
        _n_cols: usize,
        alpha: &[Self::Refine],
        out: &mut [Self::Refine],
    ) -> Result<(), GprError> {
        storage_means::<Self>(k_storage, alpha, out);
        Ok(())
    }

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

    fn mean_dot(column: &[Self::Storage], weights: &[Self::Refine]) -> Self::Refine {
        crate::sgpr::factor::storage_dot(column, weights)
    }

    fn publish_weights<M: crate::math::KernelMath>(
        _kernel: &KernelSpec,
        _a: MatRef<'_, Self::Storage>,
        _b_l: MatRef<'_, Self::Storage>,
        w: &[Self::Storage],
        _x: &[f64],
        _y: &[f64],
        _z: &[f64],
        _noise: f64,
        _n: usize,
        _m: usize,
        _d: usize,
    ) -> Result<Vec<Self::Refine>, GprError> {
        let mut out = Vec::new();
        copy_to_refine::<Self>(w, &mut out);
        Ok(out)
    }

    fn mean_from_factor<M: crate::math::KernelMath>(
        _kernel: &KernelSpec,
        _z_obs: &[f64],
        _query: &[f64],
        _k_mm_l: MatRef<'_, Self::Storage>,
        solved: &[Self::Storage],
        _rhs: &[Self::Storage],
        q_mean: &[f64],
    ) -> Result<Self::Refine, GprError> {
        Ok(crate::svgp::factor::storage_q_dot(solved, q_mean))
    }
}

impl ModelPrecision for SinglePrecision {
    const REFINES_IN_F64: bool = false;

    fn persist_kind() -> PersistKind {
        PersistKind::Single
    }

    fn publish_predict_alpha<M: crate::math::KernelMath>(
        sys: &TrainSystem<'_, Self::Storage>,
        alpha: &mut Vec<Self::Refine>,
    ) -> Result<(), GprError> {
        copy_to_refine::<Self>(sys.factor_alpha, alpha);
        Ok(())
    }

    fn predict_means<M: crate::math::KernelMath>(
        _kernel: &KernelSpec,
        k_storage: MatRef<'_, Self::Storage>,
        _x_train: MatRef<'_, f64>,
        _x_query: &[f64],
        _n_cols: usize,
        alpha: &[Self::Refine],
        out: &mut [Self::Refine],
    ) -> Result<(), GprError> {
        storage_means::<Self>(k_storage, alpha, out);
        Ok(())
    }

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

    fn view_factor<'a>(
        _mapped_l: Option<MatRef<'a, f64>>,
        workspace_l: MatRef<'a, Self::Storage>,
    ) -> MatRef<'a, Self::Storage> {
        workspace_l
    }

    fn copy_mapped_l(_src: MatRef<'_, f64>, _dest: MatMut<'_, Self::Storage>) {}

    fn mean_dot(column: &[Self::Storage], weights: &[Self::Refine]) -> Self::Refine {
        let mut sum = 0.0f64;
        for (kernel, weight) in column.iter().zip(weights.iter()) {
            sum += kernel.to_f64() * weight.to_f64();
        }
        Self::Refine::from_f64(sum)
    }

    fn publish_weights<M: crate::math::KernelMath>(
        _kernel: &KernelSpec,
        _a: MatRef<'_, Self::Storage>,
        _b_l: MatRef<'_, Self::Storage>,
        w: &[Self::Storage],
        _x: &[f64],
        _y: &[f64],
        _z: &[f64],
        _noise: f64,
        _n: usize,
        _m: usize,
        _d: usize,
    ) -> Result<Vec<Self::Refine>, GprError> {
        let mut out = Vec::new();
        copy_to_refine::<Self>(w, &mut out);
        Ok(out)
    }

    fn mean_from_factor<M: crate::math::KernelMath>(
        _kernel: &KernelSpec,
        _z_obs: &[f64],
        _query: &[f64],
        _k_mm_l: MatRef<'_, Self::Storage>,
        solved: &[Self::Storage],
        _rhs: &[Self::Storage],
        q_mean: &[f64],
    ) -> Result<Self::Refine, GprError> {
        Ok(crate::svgp::factor::storage_q_dot(solved, q_mean))
    }
}

impl<R: ResidualFormula> ModelPrecision for MixedPrecision<R> {
    const REFINES_IN_F64: bool = true;

    fn persist_kind() -> PersistKind {
        if R::READS_STORAGE {
            PersistKind::MixedPromote
        } else {
            PersistKind::MixedReevaluate
        }
    }

    fn publish_predict_alpha<M: crate::math::KernelMath>(
        sys: &TrainSystem<'_, Self::Storage>,
        alpha: &mut Vec<Self::Refine>,
    ) -> Result<(), GprError> {
        *alpha = refine::refine_alpha::<M, R>(sys)?;
        Ok(())
    }

    fn predict_means<M: crate::math::KernelMath>(
        kernel: &KernelSpec,
        _k_storage: MatRef<'_, Self::Storage>,
        x_train: MatRef<'_, f64>,
        x_query: &[f64],
        n_cols: usize,
        alpha: &[Self::Refine],
        out: &mut [Self::Refine],
    ) -> Result<(), GprError> {
        f64_cross_means::<M>(kernel, x_train, x_query, n_cols, alpha, out)
    }

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

    fn view_factor<'a>(
        _mapped_l: Option<MatRef<'a, f64>>,
        workspace_l: MatRef<'a, Self::Storage>,
    ) -> MatRef<'a, Self::Storage> {
        workspace_l
    }

    fn copy_mapped_l(_src: MatRef<'_, f64>, _dest: MatMut<'_, Self::Storage>) {}

    fn mean_dot(column: &[Self::Storage], weights: &[Self::Refine]) -> Self::Refine {
        crate::sgpr::factor::promoted_dot(column, weights)
    }

    fn publish_weights<M: crate::math::KernelMath>(
        kernel: &KernelSpec,
        a: MatRef<'_, Self::Storage>,
        b_l: MatRef<'_, Self::Storage>,
        w: &[Self::Storage],
        x: &[f64],
        y: &[f64],
        z: &[f64],
        noise: f64,
        n: usize,
        m: usize,
        d: usize,
    ) -> Result<Vec<Self::Refine>, GprError> {
        crate::sgpr::factor::refine_mixed_weights::<M, R>(
            kernel, a, b_l, w, x, y, z, noise, n, m, d,
        )
    }

    fn mean_from_factor<M: crate::math::KernelMath>(
        kernel: &KernelSpec,
        z_obs: &[f64],
        query: &[f64],
        k_mm_l: MatRef<'_, Self::Storage>,
        solved: &[Self::Storage],
        rhs: &[Self::Storage],
        q_mean: &[f64],
    ) -> Result<Self::Refine, GprError> {
        crate::svgp::factor::refined_mean::<M, R>(kernel, z_obs, query, k_mm_l, solved, rhs, q_mean)
    }
}

/// `k_*ᵀ α` with `k_*` evaluated in `f64`: one compile, then blocks of
/// [`COLUMN_BLOCK`] query columns.
fn f64_cross_means<M: crate::math::KernelMath>(
    kernel: &KernelSpec,
    x_train: MatRef<'_, f64>,
    x_query: &[f64],
    n_cols: usize,
    alpha: &[f64],
    out: &mut [f64],
) -> Result<(), GprError> {
    let kernel_f64 = kernel.compile();
    let n = x_train.nrows();
    let m = out.len();
    let block = COLUMN_BLOCK.min(m.max(1));
    let mut rows = Mat::<f64>::zeros(block, n_cols);
    let mut k_block = Mat::<f64>::zeros(n, block);
    let mut scratch = Mat::<f64>::zeros(n, block);
    let mut dist = Mat::<f64>::zeros(n, block);
    let mut start = 0;
    while start < m {
        let len = block.min(m - start);
        for dim in 0..n_cols {
            for j in 0..len {
                rows[(j, dim)] = x_query[dim * m + start + j];
            }
        }
        kernel_f64.eval_cross::<M>(
            x_train,
            rows.as_ref().submatrix(0, 0, len, n_cols),
            Some(dist.as_mut().submatrix_mut(0, 0, n, len)),
            k_block.as_mut().submatrix_mut(0, 0, n, len),
            scratch.as_mut().submatrix_mut(0, 0, n, len),
            &mut [],
        )?;
        for j in 0..len {
            let mut sum = 0.0;
            for i in 0..n {
                sum += k_block[(i, j)] * alpha[i];
            }
            out[start + j] = sum;
        }
        start += len;
    }
    Ok(())
}

/// A model precision: [`DoublePrecision`], [`SinglePrecision`], or a
/// [`MixedPrecision`].
///
/// This is the `P` parameter of `with_precision` on [`crate::Gpr`],
/// [`crate::Sgpr`], and [`crate::Svgp`]. It is sealed: only those four
/// precisions implement it, and its operations are crate-private.
///
/// # Examples
///
/// ```rust
/// use gprx::kernel::{KernelSpec, RbfKernel};
/// use gprx::{DoublePrecision, GaussianLikelihood, GpScalar, Gpr, SinglePrecision};
///
/// fn mean_at_zero<P: GpScalar>() -> Result<f64, gprx::GprError> {
///     let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
///     let fitted = Gpr::new(kernel, GaussianLikelihood::new(0.1)?)
///         .with_precision::<P>()
///         .fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
///         .map_err(|(_, e)| e)?;
///     let pred = fitted.predict(&[0.0], 1, 1)?;
///     Ok(gprx::kernel::KernelScalar::to_f64(pred.mean[0]))
/// }
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let a = mean_at_zero::<DoublePrecision>()?;
/// let b = mean_at_zero::<SinglePrecision>()?;
/// assert!((a - b).abs() < 1e-4);
/// # Ok(())
/// # }
/// ```
pub trait GpScalar: ModelPrecision {}

impl GpScalar for DoublePrecision {}
impl GpScalar for SinglePrecision {}
impl<R: ResidualFormula> GpScalar for MixedPrecision<R> {}

#[cfg(test)]
mod tests;
