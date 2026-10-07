//! Storage scalar and the mixed-precision residual solver.
//!
//! Omitted model precision is [`DoublePrecision`] (`f64`). [`SinglePrecision`]
//! runs the same procedures in `f32` and keeps the factorization as-is.
//! [`MixedPrecision`] factors in `f32` and refines the predict `α` in `f64`.
//! The residual type parameter exists only on [`MixedPrecision`]. There is no
//! flag and no alias that picks a residual formula.

use faer::{Mat, MatMut, MatRef};

use crate::error::GprError;
use crate::kernel::{
    ColRange, KernelScalar, KernelSpec, RectSlots, RefinedSources, SourceStore, Supply,
    TrainSources,
};
use crate::transform::TargetTransform;

/// Selects storage and residual-refinement scalar types for GP computations.
///
/// See the example on [`DoublePrecision`].
pub trait PrecisionPolicy {
    /// Represents the scalar used for `K`, `L`, and other stored buffers.
    ///
    /// See the example on [`crate::Gpr`].
    type Storage: KernelScalar;
    /// Represents the scalar used when refining a solve against a higher-precision residual.
    ///
    /// See the example on [`crate::Gpr`].
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
mod sparse;

pub(crate) use refine::{COLUMN_BLOCK, RefineSystem, refine};
pub use refine::{StoredFactor, TrainSystem};
pub use sparse::F64Vfe;

mod residual_seal {
    pub trait Sealed {
        /// `true` for [`super::PromoteStorage`] (residual from the stored `f32`
        /// matrix), `false` for [`super::ReevaluateKernel`] (fresh `f64` kernel).
        const READS_STORAGE: bool;
    }
}

/// Describes how [`MixedPrecision`] builds the refinement residual `r = y − Aα`.
///
/// The only implementations are [`PromoteStorage`] and [`ReevaluateKernel`].
///
/// See the example on [`MixedPrecision`].
pub trait ResidualFormula: residual_seal::Sealed + Copy + Send + Sync + 'static {}

/// Represents the residual `r = y − Aα` from the saved `f32` matrix promoted to `f64`.
///
/// This is the residual when [`MixedPrecision`] omits its type parameter.
///
/// See the example on [`crate::Gpr`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PromoteStorage;

/// Represents the residual `r = y − Aα` from a fresh `f64` kernel evaluation.
///
/// See the example on [`crate::Gpr`].
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
#[non_exhaustive]
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

    /// The training `d²` a distance model of this precision keeps.
    type Sources: SourceStore<Self::Storage>;

    fn persist_kind() -> PersistKind;

    /// Exact: stores predict `α` from the stored factor (copy, or refine).
    fn publish_predict_alpha<M: crate::math::KernelMath, S: Supply>(
        sys: &TrainSystem<'_, Self::Storage, S>,
        alpha: &mut Vec<Self::Refine>,
    ) -> Result<(), GprError>;

    /// Exact: writes `k_*ᵀ α` for every query column into `out`.
    ///
    /// `x_query` is the transformed query, column-major `out.len() × n_cols`.
    // The system, the query and its two supply views, the scratch, and `out`.
    #[allow(clippy::too_many_arguments)]
    fn predict_means<M: crate::math::KernelMath, S: Supply>(
        kernel: &KernelSpec<S>,
        k_storage: MatRef<'_, Self::Storage>,
        x_train: MatRef<'_, f64>,
        x_query: &[f64],
        n_cols: usize,
        cross64: Option<&dyn RectSlots<f64>>,
        alpha: &[Self::Refine],
        out: &mut [Self::Refine],
    ) -> Result<(), GprError>;

    /// Maps predicted mean and variance back through the target transform.
    ///
    /// A storage-scalar `Refine` (`f32`) maps through `f64` copies kept in
    /// `buffers`, so a caller that keeps them allocates nothing.
    fn inverse_mean_variance(
        transform: &dyn TargetTransform,
        mean: &mut [Self::Refine],
        variance: &mut [Self::Refine],
        buffers: &mut InverseBuffers,
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
    /// `reference` assembles the VFE system in `f64`; only a refining
    /// precision calls it (for its residual or its fallback).
    ///
    /// # Errors
    ///
    /// Returns [`GprError::CholeskyFailed`] when the `f64` fallback factor
    /// is not positive definite. Refinement does not add jitter.
    fn publish_weights(
        a: MatRef<'_, Self::Storage>,
        b_l: MatRef<'_, Self::Storage>,
        w: &[Self::Storage],
        y: &[f64],
        noise: f64,
        reference: &dyn Fn() -> Result<F64Vfe, GprError>,
    ) -> Result<Vec<Self::Refine>, GprError>;
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

/// `f64` copies an `f32` prediction maps through the target transform.
#[derive(Debug, Default)]
pub struct InverseBuffers {
    mean64: Vec<f64>,
    variance64: Vec<f64>,
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
    type Sources = TrainSources<f64>;

    fn persist_kind() -> PersistKind {
        PersistKind::Double
    }

    fn publish_predict_alpha<M: crate::math::KernelMath, S: Supply>(
        sys: &TrainSystem<'_, Self::Storage, S>,
        alpha: &mut Vec<Self::Refine>,
    ) -> Result<(), GprError> {
        copy_to_refine::<Self>(sys.factor_alpha, alpha);
        Ok(())
    }

    fn predict_means<M: crate::math::KernelMath, S: Supply>(
        _kernel: &KernelSpec<S>,
        k_storage: MatRef<'_, Self::Storage>,
        _x_train: MatRef<'_, f64>,
        _x_query: &[f64],
        _n_cols: usize,
        _cross64: Option<&dyn RectSlots<f64>>,
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
        _buffers: &mut InverseBuffers,
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
        sparse::storage_dot(column, weights)
    }

    fn publish_weights(
        _a: MatRef<'_, Self::Storage>,
        _b_l: MatRef<'_, Self::Storage>,
        w: &[Self::Storage],
        _y: &[f64],
        _noise: f64,
        _reference: &dyn Fn() -> Result<F64Vfe, GprError>,
    ) -> Result<Vec<Self::Refine>, GprError> {
        let mut out = Vec::new();
        copy_to_refine::<Self>(w, &mut out);
        Ok(out)
    }
}

impl ModelPrecision for SinglePrecision {
    const REFINES_IN_F64: bool = false;
    type Sources = TrainSources<f32>;

    fn persist_kind() -> PersistKind {
        PersistKind::Single
    }

    fn publish_predict_alpha<M: crate::math::KernelMath, S: Supply>(
        sys: &TrainSystem<'_, Self::Storage, S>,
        alpha: &mut Vec<Self::Refine>,
    ) -> Result<(), GprError> {
        copy_to_refine::<Self>(sys.factor_alpha, alpha);
        Ok(())
    }

    fn predict_means<M: crate::math::KernelMath, S: Supply>(
        _kernel: &KernelSpec<S>,
        k_storage: MatRef<'_, Self::Storage>,
        _x_train: MatRef<'_, f64>,
        _x_query: &[f64],
        _n_cols: usize,
        _cross64: Option<&dyn RectSlots<f64>>,
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
        buffers: &mut InverseBuffers,
    ) -> Result<(), GprError> {
        let InverseBuffers { mean64, variance64 } = buffers;
        mean64.clear();
        mean64.extend(mean.iter().copied().map(f32::to_f64));
        variance64.clear();
        variance64.extend(variance.iter().copied().map(f32::to_f64));
        inverse_f64_mean_variance(transform, mean64, variance64)?;
        for (slot, value) in mean.iter_mut().zip(mean64.iter()) {
            *slot = f32::from_f64(*value);
        }
        for (slot, value) in variance.iter_mut().zip(variance64.iter()) {
            *slot = f32::from_f64(*value);
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

    fn publish_weights(
        _a: MatRef<'_, Self::Storage>,
        _b_l: MatRef<'_, Self::Storage>,
        w: &[Self::Storage],
        _y: &[f64],
        _noise: f64,
        _reference: &dyn Fn() -> Result<F64Vfe, GprError>,
    ) -> Result<Vec<Self::Refine>, GprError> {
        let mut out = Vec::new();
        copy_to_refine::<Self>(w, &mut out);
        Ok(out)
    }
}

impl<R: ResidualFormula> ModelPrecision for MixedPrecision<R> {
    const REFINES_IN_F64: bool = true;
    type Sources = RefinedSources;

    fn persist_kind() -> PersistKind {
        if R::READS_STORAGE {
            PersistKind::MixedPromote
        } else {
            PersistKind::MixedReevaluate
        }
    }

    fn publish_predict_alpha<M: crate::math::KernelMath, S: Supply>(
        sys: &TrainSystem<'_, Self::Storage, S>,
        alpha: &mut Vec<Self::Refine>,
    ) -> Result<(), GprError> {
        *alpha = refine::refine_alpha::<M, R, _>(sys)?;
        Ok(())
    }

    fn predict_means<M: crate::math::KernelMath, S: Supply>(
        kernel: &KernelSpec<S>,
        _k_storage: MatRef<'_, Self::Storage>,
        x_train: MatRef<'_, f64>,
        x_query: &[f64],
        n_cols: usize,
        cross64: Option<&dyn RectSlots<f64>>,
        alpha: &[Self::Refine],
        out: &mut [Self::Refine],
    ) -> Result<(), GprError> {
        f64_cross_means::<M, _>(kernel, x_train, x_query, n_cols, cross64, alpha, out)
    }

    fn inverse_mean_variance(
        transform: &dyn TargetTransform,
        mean: &mut [Self::Refine],
        variance: &mut [Self::Refine],
        _buffers: &mut InverseBuffers,
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
        sparse::promoted_dot(column, weights)
    }

    fn publish_weights(
        a: MatRef<'_, Self::Storage>,
        b_l: MatRef<'_, Self::Storage>,
        w: &[Self::Storage],
        y: &[f64],
        noise: f64,
        reference: &dyn Fn() -> Result<F64Vfe, GprError>,
    ) -> Result<Vec<Self::Refine>, GprError> {
        sparse::refine_vfe_weights::<R>(a, b_l, w, y, noise, reference)
    }
}

/// `k_*ᵀ α` with `k_*` evaluated in `f64`: one compile, then blocks of
/// [`COLUMN_BLOCK`] query columns.
fn f64_cross_means<M: crate::math::KernelMath, S: Supply>(
    kernel: &KernelSpec<S>,
    x_train: MatRef<'_, f64>,
    x_query: &[f64],
    n_cols: usize,
    cross64: Option<&dyn RectSlots<f64>>,
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
    let mut nested = Vec::new();
    let mut start = 0;
    while start < m {
        let len = block.min(m - start);
        for dim in 0..n_cols {
            for j in 0..len {
                rows[(j, dim)] = x_query[dim * m + start + j];
            }
        }
        let cols = cross64.map(|inner| ColRange { inner, start, len });
        kernel_f64.eval_cross_slots::<M>(
            x_train,
            rows.as_ref().submatrix(0, 0, len, n_cols),
            cols.as_ref().map(|c| c as &dyn RectSlots<f64>),
            Some(dist.as_mut().submatrix_mut(0, 0, n, len)),
            k_block.as_mut().submatrix_mut(0, 0, n, len),
            scratch.as_mut().submatrix_mut(0, 0, n, len),
            &mut nested,
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

/// Represents the A model precision: [`DoublePrecision`], [`SinglePrecision`], or a [`MixedPrecision`].
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
