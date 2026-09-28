//! VFE assembly, derivatives, rank-1 `X` updates, and inducing `m` updates.

use std::marker::PhantomData;

use dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt;
use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, MatMut, MatRef};

use crate::error::{CholeskyStage, GprError};
use crate::gpr::JitterPolicy;
use crate::gpr::factor::{
    cholesky_lower_with_policy, pack_points, symmetrize_lower, validate_query, validate_training,
};
use crate::kernel::{
    CompiledKernel, CoordMode, FillDistances, GramKernel, KernelScalar, KernelSpec, Triangle,
};
use crate::likelihood::GaussianLikelihood;
use crate::param::Interval;
use crate::precision::{ModelPrecision, StorageScalar};
use crate::workspace::{faer_par, faer_par_dims};
use crate::{PredictOptions, Prediction, VarianceKind};

fn lit<T: StorageScalar>(value: f64) -> T {
    T::from_f64(value)
}

fn storage_hypot<T: StorageScalar>(a: T, b: T) -> T {
    let sq = a * a + b * b;
    faer_traits::math_utils::sqrt(&sq)
}

fn storage_sqrt<T: StorageScalar>(value: T) -> T {
    faer_traits::math_utils::sqrt(&value)
}

/// Predictive mean from one storage kernel column and the predict weights.
pub(crate) trait MeanDot: ModelPrecision {
    fn mean_dot(column: &[Self::Storage], weights: &[Self::Refine]) -> Self::Refine;
}

fn storage_dot<T: StorageScalar>(column: &[T], weights: &[T]) -> T {
    let mut sum = lit::<T>(0.0);
    for (kernel, weight) in column.iter().zip(weights.iter()) {
        sum += *kernel * *weight;
    }
    sum
}

fn promoted_dot(column: &[f32], weights: &[f64]) -> f64 {
    let mut sum = 0.0;
    for (kernel, weight) in column.iter().zip(weights.iter()) {
        sum += kernel.to_f64() * *weight;
    }
    sum
}

impl MeanDot for crate::precision::DoublePrecision {
    fn mean_dot(column: &[Self::Storage], weights: &[Self::Refine]) -> Self::Refine {
        storage_dot(column, weights)
    }
}

impl MeanDot for crate::precision::SinglePrecision {
    fn mean_dot(column: &[Self::Storage], weights: &[Self::Refine]) -> Self::Refine {
        let mut sum = 0.0f64;
        for (kernel, weight) in column.iter().zip(weights.iter()) {
            sum += kernel.to_f64() * weight.to_f64();
        }
        Self::Refine::from_f64(sum)
    }
}

impl MeanDot for crate::precision::MixedPrecision<crate::precision::PromoteStorage> {
    fn mean_dot(column: &[Self::Storage], weights: &[Self::Refine]) -> Self::Refine {
        promoted_dot(column, weights)
    }
}

impl MeanDot for crate::precision::MixedPrecision<crate::precision::ReevaluateKernel> {
    fn mean_dot(column: &[Self::Storage], weights: &[Self::Refine]) -> Self::Refine {
        promoted_dot(column, weights)
    }
}

use super::InducingLayout;
use super::fitted::FittedSgpr;

// `K_mm` only. Public default stays Fixed(0). Forrester m=16 / ℓ=1 is not PD in f64.
/// Predict weights after a factor or an online update.
pub(crate) trait PublishSgprWeights: ModelPrecision {
    /// Copies storage `w`, or refines the mixed-precision solve `B w = A y`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::CholeskyFailed`] when the `f64` fallback factor
    /// is not positive definite. Iterative refinement does not add jitter.
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
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn publish_sgpr_weights<M: crate::math::KernelMath, P: PublishSgprWeights>(
    kernel: &KernelSpec,
    a: MatRef<'_, P::Storage>,
    b_l: MatRef<'_, P::Storage>,
    w: &[P::Storage],
    x: &[f64],
    y: &[f64],
    z: &[f64],
    noise: f64,
    n: usize,
    m: usize,
    d: usize,
) -> Result<Vec<P::Refine>, GprError> {
    P::publish_weights::<M>(kernel, a, b_l, w, x, y, z, noise, n, m, d)
}

fn copy_storage_weights<P: ModelPrecision>(w: &[P::Storage]) -> Vec<P::Refine> {
    w.iter()
        .map(|value| P::Refine::from_f64(value.to_f64()))
        .collect()
}

impl PublishSgprWeights for crate::precision::DoublePrecision {
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
        let _ = (kernel, a, b_l, x, y, z, noise, n, m, d);
        Ok(copy_storage_weights::<Self>(w))
    }
}

impl PublishSgprWeights for crate::precision::SinglePrecision {
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
        let _ = (kernel, a, b_l, x, y, z, noise, n, m, d);
        Ok(copy_storage_weights::<Self>(w))
    }
}

impl PublishSgprWeights for crate::precision::MixedPrecision<crate::precision::PromoteStorage> {
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
        refine_mixed_weights::<M, crate::precision::PromoteStorage>(
            kernel, a, b_l, w, x, y, z, noise, n, m, d,
        )
    }
}

impl PublishSgprWeights for crate::precision::MixedPrecision<crate::precision::ReevaluateKernel> {
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
        refine_mixed_weights::<M, crate::precision::ReevaluateKernel>(
            kernel, a, b_l, w, x, y, z, noise, n, m, d,
        )
    }
}

trait MixedWeightSystem: crate::precision::ResidualFormula {
    #[allow(clippy::too_many_arguments)]
    fn weight_system<M: crate::math::KernelMath>(
        kernel: &KernelSpec,
        a: MatRef<'_, f32>,
        x: &[f64],
        y: &[f64],
        z: &[f64],
        noise: f64,
        n: usize,
        m: usize,
        d: usize,
    ) -> Result<WeightSystem, GprError>;
}

struct WeightSystem {
    b32: Option<Mat<f32>>,
    b64: Option<Mat<f64>>,
    rhs: Vec<f64>,
}

impl MixedWeightSystem for crate::precision::PromoteStorage {
    fn weight_system<M: crate::math::KernelMath>(
        kernel: &KernelSpec,
        a: MatRef<'_, f32>,
        x: &[f64],
        y: &[f64],
        z: &[f64],
        noise: f64,
        n: usize,
        m: usize,
        d: usize,
    ) -> Result<WeightSystem, GprError> {
        let _ = (kernel, x, z, n, m, d);
        Ok(WeightSystem {
            b32: Some(gram_b32(a, noise)),
            b64: None,
            rhs: matvec_f32(a, y),
        })
    }
}

impl MixedWeightSystem for crate::precision::ReevaluateKernel {
    fn weight_system<M: crate::math::KernelMath>(
        kernel: &KernelSpec,
        a: MatRef<'_, f32>,
        x: &[f64],
        y: &[f64],
        z: &[f64],
        noise: f64,
        n: usize,
        m: usize,
        d: usize,
    ) -> Result<WeightSystem, GprError> {
        let _ = a;
        let state = assemble_vfe::<M, f64>(kernel, noise_likelihood(noise)?, x, n, d, y, z, m)?;
        Ok(WeightSystem {
            b32: None,
            b64: Some(gram_b64(state.a.as_ref(), noise)),
            rhs: matvec_f64(state.a.as_ref(), y),
        })
    }
}

fn noise_likelihood(noise: f64) -> Result<GaussianLikelihood, GprError> {
    if crate::param::Interval::DEFAULT_POSITIVE.contains(noise) {
        return GaussianLikelihood::new(noise);
    }
    let interval = crate::Interval::new(1.0e-12, 1.0e5)?;
    let mut likelihood = GaussianLikelihood::new(1.0)?.with_bounds(interval)?;
    likelihood.set_params(&[noise.ln()])?;
    Ok(likelihood)
}

fn gram_b32(a: MatRef<'_, f32>, noise: f64) -> Mat<f32> {
    let m = a.nrows();
    let n = a.ncols();
    let mut b = Mat::<f32>::zeros(m, m);
    let noise32 = noise as f32;
    for i in 0..m {
        for j in 0..m {
            let mut sum = 0.0f32;
            for k in 0..n {
                sum += a[(i, k)] * a[(j, k)];
            }
            if i == j {
                sum += noise32;
            }
            b[(i, j)] = sum;
        }
    }
    b
}

fn gram_b64(a: MatRef<'_, f64>, noise: f64) -> Mat<f64> {
    let m = a.nrows();
    let n = a.ncols();
    let mut b = Mat::<f64>::zeros(m, m);
    for i in 0..m {
        for j in 0..m {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a[(i, k)] * a[(j, k)];
            }
            if i == j {
                sum += noise;
            }
            b[(i, j)] = sum;
        }
    }
    b
}

fn matvec_f32(a: MatRef<'_, f32>, y: &[f64]) -> Vec<f64> {
    let m = a.nrows();
    let mut rhs = vec![0.0; m];
    for i in 0..m {
        let mut sum = 0.0;
        for j in 0..a.ncols() {
            sum += f64::from(a[(i, j)]) * y[j];
        }
        rhs[i] = sum;
    }
    rhs
}

fn matvec_f64(a: MatRef<'_, f64>, y: &[f64]) -> Vec<f64> {
    let m = a.nrows();
    let mut rhs = vec![0.0; m];
    for i in 0..m {
        let mut sum = 0.0;
        for j in 0..a.ncols() {
            sum += a[(i, j)] * y[j];
        }
        rhs[i] = sum;
    }
    rhs
}

fn residual_inf(
    b32: Option<MatRef<'_, f32>>,
    b64: Option<MatRef<'_, f64>>,
    w: &[f64],
    rhs: &[f64],
    r: &mut [f64],
) -> f64 {
    let m = rhs.len();
    let mut b_inf = 0.0f64;
    for i in 0..m {
        let mut row = 0.0;
        let mut sum = 0.0;
        for j in 0..m {
            let bij = if let Some(b) = b32 {
                f64::from(b[(i, j)])
            } else if let Some(b) = b64 {
                b[(i, j)]
            } else {
                0.0
            };
            row += bij.abs();
            sum += bij * w[j];
        }
        b_inf = b_inf.max(row);
        r[i] = rhs[i] - sum;
    }
    b_inf
}

fn inf_norm_f64(values: &[f64]) -> f64 {
    values.iter().fold(0.0, |acc, value| acc.max(value.abs()))
}

#[allow(clippy::too_many_arguments)]
fn refine_mixed_weights<M: crate::math::KernelMath, R: MixedWeightSystem>(
    kernel: &KernelSpec,
    a: MatRef<'_, f32>,
    b_l: MatRef<'_, f32>,
    w_storage: &[f32],
    x: &[f64],
    y: &[f64],
    z: &[f64],
    noise: f64,
    n: usize,
    m: usize,
    d: usize,
) -> Result<Vec<f64>, GprError> {
    if m == 0 {
        return Ok(Vec::new());
    }
    let system = R::weight_system::<M>(kernel, a, x, y, z, noise, n, m, d)?;
    let mut w: Vec<f64> = w_storage.iter().map(|value| f64::from(*value)).collect();
    let tol = 10.0 * m as f64 * f64::EPSILON;
    let mut resid = vec![0.0; m];
    let mut prev: Option<f64> = None;
    let mut streak = 0usize;
    let b32 = system.b32.as_ref().map(Mat::as_ref);
    let b64 = system.b64.as_ref().map(Mat::as_ref);
    for _ in 0..10 {
        let b_inf = residual_inf(b32, b64, &w, &system.rhs, &mut resid);
        let r_inf = inf_norm_f64(&resid);
        let denom = b_inf * inf_norm_f64(&w) + inf_norm_f64(&system.rhs);
        if denom > 0.0 && r_inf / denom < tol {
            return Ok(w);
        }
        if let Some(prev_r) = prev {
            let ratio = if prev_r > 0.0 { r_inf / prev_r } else { 0.0 };
            if ratio > 0.9 {
                streak += 1;
                if streak >= 2 {
                    return f64_assembly_w::<M>(kernel, x, y, z, noise, n, m, d);
                }
            } else {
                streak = 0;
            }
        }
        prev = Some(r_inf);
        let mut delta = Mat::<f32>::from_fn(m, 1, |i, _| resid[i] as f32);
        solve_llt_in_place(b_l, delta.as_mut());
        for i in 0..m {
            w[i] += f64::from(delta[(i, 0)]);
        }
    }
    f64_assembly_w::<M>(kernel, x, y, z, noise, n, m, d)
}

#[allow(clippy::too_many_arguments)]
fn f64_assembly_w<M: crate::math::KernelMath>(
    kernel: &KernelSpec,
    x: &[f64],
    y: &[f64],
    z: &[f64],
    noise: f64,
    n: usize,
    m: usize,
    d: usize,
) -> Result<Vec<f64>, GprError> {
    let state = assemble_vfe::<M, f64>(kernel, noise_likelihood(noise)?, x, n, d, y, z, m)?;
    Ok(state.w)
}

fn k_mm_jitter_policy() -> JitterPolicy {
    JitterPolicy::adaptive(1e-8, 10.0, 5, 1e-3).unwrap_or_default()
}

pub(crate) struct VfeState<T: StorageScalar> {
    pub(crate) k_mm_l: Mat<T>,
    pub(crate) a: Mat<T>,
    pub(crate) b_l: Mat<T>,
    pub(crate) w: Vec<T>,
    pub(crate) k_diag_sum: T,
    pub(crate) a_frobenius2: T,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn assemble_fitted<O, I: InducingLayout, M: crate::math::KernelMath, P>(
    kernel: KernelSpec,
    likelihood: GaussianLikelihood,
    optimizer: O,
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    y: &[f64],
    z: &[f64],
    n_inducing: usize,
) -> Result<FittedSgpr<O, I, M, P>, GprError>
where
    P: ModelPrecision + PublishSgprWeights,
    P::Storage: FillDistances,
    CompiledKernel<P::Storage>: GramKernel<T = P::Storage>,
{
    let state =
        assemble_vfe::<M, P::Storage>(&kernel, likelihood, x, n_rows, n_cols, y, z, n_inducing)?;
    let predict_w = if std::mem::size_of::<P::Storage>() == std::mem::size_of::<f32>()
        && std::mem::size_of::<P::Refine>() == std::mem::size_of::<f64>()
    {
        assemble_vfe::<M, f64>(&kernel, likelihood, x, n_rows, n_cols, y, z, n_inducing)?
            .w
            .into_iter()
            .map(P::Refine::from_f64)
            .collect()
    } else {
        publish_sgpr_weights::<M, P>(
            &kernel,
            state.a.as_ref(),
            state.b_l.as_ref(),
            &state.w,
            x,
            y,
            z,
            likelihood.noise_variance(),
            n_rows,
            n_inducing,
            n_cols,
        )?
    };
    Ok(FittedSgpr {
        kernel,
        likelihood,
        optimizer,
        inducing: PhantomData,
        _math: PhantomData,
        x_obs: x.to_vec(),
        z_obs: z.to_vec(),
        y: y.to_vec(),
        k_mm_l: state.k_mm_l,
        a: state.a,
        b_l: state.b_l,
        w: state.w,
        predict_w,
        k_diag_sum: state.k_diag_sum,
        a_frobenius2: state.a_frobenius2,
        n: n_rows,
        m: n_inducing,
        d: n_cols,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn assemble_vfe<M: crate::math::KernelMath, T>(
    kernel: &KernelSpec,
    likelihood: GaussianLikelihood,
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    y: &[f64],
    z: &[f64],
    n_inducing: usize,
) -> Result<VfeState<T>, GprError>
where
    T: StorageScalar + FillDistances,
    CompiledKernel<T>: GramKernel<T = T>,
{
    validate_training(x, n_rows, n_cols, y)?;
    validate_inducing(z, n_inducing, n_cols)?;
    if std::mem::size_of::<T>() == std::mem::size_of::<f32>() {
        let state =
            assemble_vfe::<M, f64>(kernel, likelihood, x, n_rows, n_cols, y, z, n_inducing)?;
        return Ok(VfeState {
            k_mm_l: round_mat::<T>(state.k_mm_l.as_ref()),
            a: round_mat::<T>(state.a.as_ref()),
            b_l: round_mat::<T>(state.b_l.as_ref()),
            w: state.w.iter().map(|value| T::from_f64(*value)).collect(),
            k_diag_sum: T::from_f64(state.k_diag_sum),
            a_frobenius2: T::from_f64(state.a_frobenius2),
        });
    }
    let compiled = kernel.compile_as::<T>();
    let x64 = pack_points(x, n_rows, n_cols);
    let z64 = pack_points(z, n_inducing, n_cols);
    let mut x_cast = T::empty_cols();
    let mut z_cast = T::empty_cols();
    let mut y_cast = T::empty_rows();
    let x_mat = T::storage_cols(x64.as_ref(), &mut x_cast);
    let z_mat = T::storage_cols(z64.as_ref(), &mut z_cast);
    let y_s = T::storage_rows(y, &mut y_cast);
    let round_kernel = std::mem::size_of::<T>() == std::mem::size_of::<f32>();
    let mut k_mm = Mat::zeros(n_inducing, n_inducing);
    if round_kernel {
        let compiled64 = kernel.compile();
        let mut k64 = Mat::<f64>::zeros(n_inducing, n_inducing);
        let mut scratch64 = Mat::<f64>::zeros(n_inducing, n_inducing);
        compiled64.apply_points::<M>(
            z64.as_ref(),
            k64.as_mut(),
            Triangle::Lower,
            scratch64.as_mut(),
        )?;
        for col in 0..n_inducing {
            for row in col..n_inducing {
                k_mm[(row, col)] = T::from_f64(k64[(row, col)]);
            }
        }
    } else {
        let mut scratch = Mat::zeros(n_inducing, n_inducing);
        compiled.apply_points::<M>(
            z_mat.as_ref(),
            k_mm.as_mut(),
            Triangle::Lower,
            scratch.as_mut(),
        )?;
    }
    let req = llt::factor::cholesky_in_place_scratch::<T>(
        n_inducing,
        faer_par(n_inducing),
        Default::default(),
    );
    let mut chol_scratch = MemBuffer::new(req);
    cholesky_lower_with_policy(
        &mut k_mm,
        &mut chol_scratch,
        k_mm_jitter_policy(),
        CholeskyStage::Fit,
    )?;
    // Same packed `X` and `Z` share a training White diagonal. Rectangular
    // `apply_cross` leaves White at zero.
    let mut a = if round_kernel {
        let compiled64 = kernel.compile();
        if x == z {
            let mut gram64 = Mat::<f64>::zeros(n_rows, n_rows);
            let mut gram_scratch = Mat::<f64>::zeros(n_rows, n_rows);
            compiled64.apply_points::<M>(
                x64.as_ref(),
                gram64.as_mut(),
                Triangle::Lower,
                gram_scratch.as_mut(),
            )?;
            let mut gram = Mat::<T>::zeros(n_rows, n_rows);
            for col in 0..n_rows {
                for row in col..n_rows {
                    gram[(row, col)] = T::from_f64(gram64[(row, col)]);
                }
            }
            symmetrize_lower(gram.as_mut(), n_rows);
            gram
        } else {
            let cross = kernel_cross::<M, f64>(&compiled64, z64.as_ref(), x64.as_ref())?;
            let mut stored = Mat::<T>::zeros(n_inducing, n_rows);
            for col in 0..n_rows {
                for row in 0..n_inducing {
                    stored[(row, col)] = T::from_f64(cross[(row, col)]);
                }
            }
            stored
        }
    } else if x == z {
        let mut gram = Mat::zeros(n_rows, n_rows);
        let mut gram_scratch = Mat::zeros(n_rows, n_rows);
        compiled.apply_points::<M>(
            x_mat.as_ref(),
            gram.as_mut(),
            Triangle::Lower,
            gram_scratch.as_mut(),
        )?;
        symmetrize_lower(gram.as_mut(), n_rows);
        gram
    } else {
        kernel_cross::<M, _>(&compiled, z_mat.as_ref(), x_mat.as_ref())?
    };
    solve_lower(k_mm.as_ref(), a.as_mut());
    let noise = likelihood.noise_variance();
    let mut b = gram_aat_plus_noise(a.as_ref(), noise);
    let b_req = llt::factor::cholesky_in_place_scratch::<T>(
        n_inducing,
        faer_par(n_inducing),
        Default::default(),
    );
    let mut b_scratch = MemBuffer::new(b_req);
    cholesky_lower_with_policy(
        &mut b,
        &mut b_scratch,
        JitterPolicy::default(),
        CholeskyStage::Fit,
    )?;
    let mut k_diag = vec![lit::<T>(0.0); n_rows];
    if round_kernel {
        let compiled64 = kernel.compile();
        let mut diag = vec![0.0f64; n_rows];
        compiled64.fill_diag_points(x64.as_ref(), &mut diag)?;
        for (slot, value) in k_diag.iter_mut().zip(diag) {
            *slot = T::from_f64(value);
        }
    } else {
        compiled.fill_diag_points(x_mat.as_ref(), &mut k_diag)?;
    }
    let k_diag_sum = k_diag.iter().fold(lit::<T>(0.0), |acc, v| acc + *v);
    let a_frobenius2 = frobenius2(a.as_ref());
    let mut ay = Mat::zeros(n_inducing, 1);
    if round_kernel {
        for i in 0..n_inducing {
            let mut sum = 0.0f64;
            for j in 0..n_rows {
                sum += a[(i, j)].to_f64() * y[j];
            }
            ay[(i, 0)] = T::from_f64(sum);
        }
    } else {
        matvec_columns(a.as_ref(), y_s, ay.as_mut());
    }
    solve_llt_in_place(b.as_ref(), ay.as_mut());
    let mut w = vec![lit::<T>(0.0); n_inducing];
    for i in 0..n_inducing {
        w[i] = ay[(i, 0)];
    }
    Ok(VfeState::<T> {
        k_mm_l: k_mm,
        a,
        b_l: b,
        w,
        k_diag_sum,
        a_frobenius2,
    })
}

pub(crate) fn fill_z_intervals(
    x: &[f64],
    n: usize,
    d: usize,
    out: &mut [Interval],
) -> Result<(), GprError> {
    if d == 0 || out.len() % d != 0 {
        return Err(GprError::InvalidHyperparameter {
            reason: "inducing interval length is not a multiple of d".to_owned(),
        });
    }
    let m = out.len() / d;
    for dim in 0..d {
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        for i in 0..n {
            let v = x[i + dim * n];
            min_x = min_x.min(v);
            max_x = max_x.max(v);
        }
        let range = (max_x - min_x).max(0.0);
        let slack = (0.1 * range).max(0.1);
        let mut lo = min_x - slack;
        let hi = max_x + slack;
        if lo > 0.0 {
            lo = 0.0;
        }
        let interval = Interval::new(lo, hi)?;
        for p in 0..m {
            out[p + dim * m] = interval;
        }
    }
    Ok(())
}

pub(crate) struct KernelVar<T: StorageScalar> {
    pub(crate) d_kmm: Mat<T>,
    pub(crate) d_kmn: Mat<T>,
    pub(crate) d_kdiag: T,
    pub(crate) d_noise: T,
}

pub(crate) struct VfeEngine<'a, T: StorageScalar> {
    pub(crate) l: MatRef<'a, T>,
    pub(crate) a: MatRef<'a, T>,
    pub(crate) b_l: MatRef<'a, T>,
    pub(crate) w: &'a [T],
    pub(crate) noise: T,
    pub(crate) y: &'a [T],
    pub(crate) n: usize,
    pub(crate) m: usize,
    quad: T,
    trace: T,
}

impl<'a, T: StorageScalar> VfeEngine<'a, T> {
    fn from_model<O, I, M: crate::math::KernelMath, P>(
        model: &'a FittedSgpr<O, I, M, P>,
        y: &'a [T],
    ) -> Self
    where
        P: ModelPrecision<Storage = T>,
    {
        let m = model.m;
        let n = model.n;
        let noise = lit::<T>(model.likelihood.noise_variance());
        let a = model.a.as_ref();
        let w = model.w.as_slice();
        let mut y_norm2 = lit::<T>(0.0);
        for v in y {
            y_norm2 += v * v;
        }
        let mut ay_dot_w = lit::<T>(0.0);
        for i in 0..m {
            let mut ay_i = lit::<T>(0.0);
            for j in 0..n {
                ay_i += a[(i, j)] * y[j];
            }
            ay_dot_w += ay_i * w[i];
        }
        Self {
            l: model.k_mm_l.as_ref(),
            a,
            b_l: model.b_l.as_ref(),
            w,
            noise,
            y,
            n,
            m,
            quad: (y_norm2 - ay_dot_w) / noise,
            trace: (model.k_diag_sum - model.a_frobenius2) / (lit::<T>(2.0) * noise),
        }
    }

    fn tangent_from(
        &self,
        d_kmm: MatRef<'_, T>,
        mut da: Mat<T>,
        d_kdiag: T,
        d_noise: T,
    ) -> VfeTangent<T> {
        let phi = chol_phi(self.l, d_kmm);
        let phi_l = tril_half(phi.as_ref());
        solve_lower(self.l, da.as_mut());
        mat_sub_mul(&mut da, phi_l.as_ref(), self.a);
        let db = noise_plus_sym_prod(da.as_ref(), self.a, d_noise);
        let mut u = vec![lit::<T>(0.0); self.m];
        for i in 0..self.m {
            let mut sum = lit::<T>(0.0);
            for j in 0..self.n {
                sum += da[(i, j)] * self.y[j];
            }
            u[i] = sum;
        }
        VfeTangent::<T> {
            phi,
            phi_l,
            da,
            db,
            u,
            d_kdiag,
            d_noise,
        }
    }

    fn first_tangent(&self, var: &KernelVar<T>) -> VfeTangent<T> {
        self.tangent_from(
            var.d_kmm.as_ref(),
            var.d_kmn.clone(),
            var.d_kdiag,
            var.d_noise,
        )
    }

    fn directional_owned(&self, var: KernelVar<T>) -> T {
        let t = self.tangent_from(var.d_kmm.as_ref(), var.d_kmn, var.d_kdiag, var.d_noise);
        self.directional_from_tangent(&t)
    }

    /// Likelihood `θ` has `∂K = 0` and `∂σn² = σn²`, so the `m×n` products are zero.
    fn directional_noise(&self, d_noise: f64) -> T {
        let d_noise = lit::<T>(d_noise);
        let m = self.m;
        let mut db = Mat::zeros(m, m);
        for i in 0..m {
            db[(i, i)] = d_noise;
        }
        let quad = self.quad;
        let trace = self.trace;
        let d_logdet_b = trace_solve(self.b_l, db.as_ref());
        let d_q = -quad_form(self.w, db.as_ref());
        let d_quad = -d_q / self.noise - quad * d_noise / self.noise;
        let d_trace = -trace * d_noise / self.noise;
        let n_minus_m = lit::<T>(self.n as f64) - lit::<T>(self.m as f64);
        lit::<T>(0.5) * (n_minus_m * d_noise / self.noise + d_logdet_b + d_quad) + d_trace
    }

    fn directional_from_tangent(&self, t: &VfeTangent<T>) -> T {
        let quad = self.quad;
        let trace = self.trace;
        let d_logdet_b = trace_solve(self.b_l, t.db.as_ref());
        let d_q = lit::<T>(2.0) * dot(self.w, &t.u) - quad_form(self.w, t.db.as_ref());
        let d_af = lit::<T>(2.0) * frobenius_dot(self.a, t.da.as_ref());
        let d_quad = -d_q / self.noise - quad * t.d_noise / self.noise;
        let d_trace =
            (t.d_kdiag - d_af) / (lit::<T>(2.0) * self.noise) - trace * t.d_noise / self.noise;
        let n_minus_m = lit::<T>(self.n as f64) - lit::<T>(self.m as f64);
        lit::<T>(0.5) * (n_minus_m * t.d_noise / self.noise + d_logdet_b + d_quad) + d_trace
    }

    fn second_directional(&self, ti: &VfeTangent<T>, tj: &VfeTangent<T>, dd: &KernelVar<T>) -> T {
        let quad = self.quad;
        let trace = self.trace;
        let phi_dd = chol_phi(self.l, dd.d_kmm.as_ref());
        let dphi_j_on_i = dphi_from(ti.phi.as_ref(), tj.phi_l.as_ref(), phi_dd.as_ref());
        let dphi_l = tril_half(dphi_j_on_i.as_ref());
        let mut linv_di_kmn = ti.da.clone();
        mat_add_mul(&mut linv_di_kmn, ti.phi_l.as_ref(), self.a);
        let mut dda = dd.d_kmn.clone();
        solve_lower(self.l, dda.as_mut());
        mat_sub_mul(&mut dda, tj.phi_l.as_ref(), linv_di_kmn.as_ref());
        mat_sub_mul(&mut dda, dphi_l.as_ref(), self.a);
        mat_sub_mul(&mut dda, ti.phi_l.as_ref(), tj.da.as_ref());
        let ddb = second_db(
            ti.da.as_ref(),
            tj.da.as_ref(),
            dda.as_ref(),
            self.a,
            dd.d_noise,
        );
        let mut ddu = vec![lit::<T>(0.0); self.m];
        for i in 0..self.m {
            let mut sum = lit::<T>(0.0);
            for j in 0..self.n {
                sum += dda[(i, j)] * self.y[j];
            }
            ddu[i] = sum;
        }
        let d_logdet_b_i = trace_solve(self.b_l, ti.db.as_ref());
        let _ = d_logdet_b_i;
        let d2_logdet_b = second_logdet_b(self.b_l, ti.db.as_ref(), tj.db.as_ref(), ddb.as_ref());
        let dwi = dw_from(self.b_l, self.w, ti.db.as_ref(), &ti.u);
        let dwj = dw_from(self.b_l, self.w, tj.db.as_ref(), &tj.u);
        let d_q_i = lit::<T>(2.0) * dot(self.w, &ti.u) - quad_form(self.w, ti.db.as_ref());
        let d_q_j = lit::<T>(2.0) * dot(self.w, &tj.u) - quad_form(self.w, tj.db.as_ref());
        let d2_q = lit::<T>(2.0) * dot(&dwj, &ti.u) + lit::<T>(2.0) * dot(self.w, &ddu)
            - dot(&dwj, &mat_vec(ti.db.as_ref(), self.w))
            - quad_form(self.w, ddb.as_ref())
            - dot(self.w, &mat_vec(ti.db.as_ref(), &dwj));
        let d_af_i = lit::<T>(2.0) * frobenius_dot(self.a, ti.da.as_ref());
        let d_af_j = lit::<T>(2.0) * frobenius_dot(self.a, tj.da.as_ref());
        let d2_af = lit::<T>(2.0) * frobenius_dot(tj.da.as_ref(), ti.da.as_ref())
            + lit::<T>(2.0) * frobenius_dot(self.a, dda.as_ref());
        let d_quad_j = -d_q_j / self.noise - quad * tj.d_noise / self.noise;
        let d2_quad = -d2_q / self.noise + d_q_i * tj.d_noise / (self.noise * self.noise)
            - d_quad_j * ti.d_noise / self.noise
            - quad * dd.d_noise / self.noise
            + quad * ti.d_noise * tj.d_noise / (self.noise * self.noise);
        let d_trace_j =
            (tj.d_kdiag - d_af_j) / (lit::<T>(2.0) * self.noise) - trace * tj.d_noise / self.noise;
        let d2_trace = (dd.d_kdiag - d2_af) / (lit::<T>(2.0) * self.noise)
            - (ti.d_kdiag - d_af_i) * tj.d_noise / (lit::<T>(2.0) * self.noise * self.noise)
            - d_trace_j * ti.d_noise / self.noise
            - trace * dd.d_noise / self.noise
            + trace * ti.d_noise * tj.d_noise / (self.noise * self.noise);
        let n_minus_m = lit::<T>(self.n as f64) - lit::<T>(self.m as f64);
        let d2_log_noise =
            dd.d_noise / self.noise - ti.d_noise * tj.d_noise / (self.noise * self.noise);
        let _ = dwi;
        lit::<T>(0.5) * (n_minus_m * d2_log_noise + d2_logdet_b + d2_quad) + d2_trace
    }
}

pub(crate) struct VfeTangent<T: StorageScalar> {
    pub(crate) phi: Mat<T>,
    pub(crate) phi_l: Mat<T>,
    pub(crate) da: Mat<T>,
    pub(crate) db: Mat<T>,
    pub(crate) u: Vec<T>,
    pub(crate) d_kdiag: T,
    pub(crate) d_noise: T,
}

pub(crate) fn analytic_gradient<M: crate::math::KernelMath, O, I, P>(
    model: &FittedSgpr<O, I, M, P>,
    out: &mut [f64],
    include_z: bool,
) -> Result<(), GprError>
where
    P: ModelPrecision,
    P::Storage: FillDistances,
    CompiledKernel<P::Storage>: GramKernel<T = P::Storage>,
{
    let mut y_cast = P::Storage::empty_rows();
    let y_s = P::Storage::storage_rows(&model.y, &mut y_cast);
    let engine = VfeEngine::<P::Storage>::from_model(model, y_s);
    let compiled = model.kernel.compile_as::<P::Storage>();
    let x64 = pack_points(&model.x_obs, model.n, model.d);
    let z64 = pack_points(&model.z_obs, model.m, model.d);
    let mut x_cast = P::Storage::empty_cols();
    let mut z_cast = P::Storage::empty_cols();
    let x = P::Storage::storage_cols(x64.as_ref(), &mut x_cast);
    let z = P::Storage::storage_cols(z64.as_ref(), &mut z_cast);
    let n_kernel = model.kernel.num_params();
    for (i, slot) in out.iter_mut().enumerate().take(n_kernel) {
        let var = kernel_theta_var::<M, _>(&compiled, x, z, model.n, i)?;
        *slot = engine.directional_owned(var).to_f64();
    }
    out[n_kernel] = engine
        .directional_noise(model.likelihood.noise_variance())
        .to_f64();
    if include_z {
        let mut idx = n_kernel + 1;
        for dim in 0..model.d {
            for p in 0..model.m {
                let var = z_coord_var::<M, _>(&compiled, x, z, p, dim)?;
                out[idx] = engine.directional_owned(var).to_f64();
                idx += 1;
            }
        }
    }
    Ok(())
}

pub(crate) fn analytic_hessian<M: crate::math::KernelMath, O, I, P>(
    model: &FittedSgpr<O, I, M, P>,
    out: &mut [f64],
    include_z: bool,
) -> Result<(), GprError>
where
    P: ModelPrecision,
    P::Storage: FillDistances,
    CompiledKernel<P::Storage>: GramKernel<T = P::Storage>,
{
    let mut y_cast = P::Storage::empty_rows();
    let y_s = P::Storage::storage_rows(&model.y, &mut y_cast);
    let engine = VfeEngine::<P::Storage>::from_model(model, y_s);
    let vars = collect_first_vars(model, include_z)?;
    let tangents: Vec<VfeTangent<P::Storage>> =
        vars.iter().map(|v| engine.first_tangent(v)).collect();
    let p = vars.len();
    for j in 0..p {
        for i in j..p {
            let dd = second_var(model, i, j, include_z)?;
            let hij = engine
                .second_directional(&tangents[i], &tangents[j], &dd)
                .to_f64();
            out[i * p + j] = hij;
            out[j * p + i] = hij;
        }
    }
    Ok(())
}

pub(crate) fn collect_first_vars<M: crate::math::KernelMath, O, I, P>(
    model: &FittedSgpr<O, I, M, P>,
    include_z: bool,
) -> Result<Vec<KernelVar<P::Storage>>, GprError>
where
    P: ModelPrecision,
    P::Storage: FillDistances,
    CompiledKernel<P::Storage>: GramKernel<T = P::Storage>,
{
    let compiled = model.kernel.compile_as::<P::Storage>();
    let x64 = pack_points(&model.x_obs, model.n, model.d);
    let z64 = pack_points(&model.z_obs, model.m, model.d);
    let mut x_cast = P::Storage::empty_cols();
    let mut z_cast = P::Storage::empty_cols();
    let x = P::Storage::storage_cols(x64.as_ref(), &mut x_cast);
    let z = P::Storage::storage_cols(z64.as_ref(), &mut z_cast);
    let n_kernel = model.kernel.num_params();
    let n_theta = n_kernel + model.likelihood.num_params();
    let mut vars = Vec::with_capacity(n_theta + if include_z { model.m * model.d } else { 0 });
    for i in 0..n_kernel {
        vars.push(kernel_theta_var::<M, _>(&compiled, x, z, model.n, i)?);
    }
    vars.push(likelihood_var(
        model.m,
        model.n,
        model.likelihood.noise_variance(),
    ));
    if include_z {
        for dim in 0..model.d {
            for p in 0..model.m {
                vars.push(z_coord_var::<M, _>(&compiled, x, z, p, dim)?);
            }
        }
    }
    Ok(vars)
}

pub(crate) fn kernel_theta_var<M: crate::math::KernelMath, T>(
    compiled: &CompiledKernel<T>,
    x: MatRef<'_, T>,
    z: MatRef<'_, T>,
    n: usize,
    param_idx: usize,
) -> Result<KernelVar<T>, GprError>
where
    T: StorageScalar + FillDistances,
    CompiledKernel<T>: GramKernel<T = T>,
{
    let m = z.nrows();
    let mut d_kmm = Mat::zeros(m, m);
    let mut scratch_mm = Mat::zeros(m, m);
    compiled.grad_points::<M>(
        z,
        d_kmm.as_mut(),
        param_idx,
        Triangle::Full,
        scratch_mm.as_mut(),
    )?;
    let mut d_kmn = Mat::zeros(m, n);
    let mut scratch_mn = Mat::zeros(m, n);
    compiled.grad_cross_points::<M>(z, x, d_kmn.as_mut(), param_idx, scratch_mn.as_mut())?;
    let mut diag = vec![lit::<T>(0.0); n];
    compiled.grad_diag_points::<M>(x, &mut diag, param_idx)?;
    let d_kdiag = diag.iter().fold(lit::<T>(0.0), |acc, v| acc + *v);
    Ok(KernelVar::<T> {
        d_kmm,
        d_kmn,
        d_kdiag,
        d_noise: lit::<T>(0.0),
    })
}

pub(crate) fn likelihood_var<T: StorageScalar>(m: usize, n: usize, noise: f64) -> KernelVar<T> {
    let _ = n;
    KernelVar::<T> {
        d_kmm: Mat::zeros(m, m),
        d_kmn: Mat::zeros(m, n),
        d_kdiag: lit::<T>(0.0),
        d_noise: lit::<T>(noise),
    }
}

pub(crate) fn z_coord_var<M: crate::math::KernelMath, T>(
    compiled: &CompiledKernel<T>,
    x: MatRef<'_, T>,
    z: MatRef<'_, T>,
    point: usize,
    dim: usize,
) -> Result<KernelVar<T>, GprError>
where
    T: StorageScalar + FillDistances,
    CompiledKernel<T>: GramKernel<T = T>,
{
    let m = z.nrows();
    let n = x.nrows();
    let mut g2 = Mat::zeros(m, m);
    compiled.grad_wrt_coord_dim::<M>(z, z, g2.as_mut(), dim)?;
    let mut g_xz = Mat::zeros(n, m);
    compiled.grad_wrt_coord_dim::<M>(x, z, g_xz.as_mut(), dim)?;
    let mut d_kmm = Mat::zeros(m, m);
    for i in 0..m {
        d_kmm[(i, point)] += g2[(i, point)];
        d_kmm[(point, i)] += g2[(i, point)];
    }
    let mut d_kmn = Mat::zeros(m, n);
    for col in 0..n {
        d_kmn[(point, col)] = g_xz[(col, point)];
    }
    Ok(KernelVar::<T> {
        d_kmm,
        d_kmn,
        d_kdiag: lit::<T>(0.0),
        d_noise: lit::<T>(0.0),
    })
}

pub(crate) fn second_var<M: crate::math::KernelMath, O, I, P>(
    model: &FittedSgpr<O, I, M, P>,
    i: usize,
    j: usize,
    include_z: bool,
) -> Result<KernelVar<P::Storage>, GprError>
where
    P: ModelPrecision,
    P::Storage: FillDistances,
    CompiledKernel<P::Storage>: GramKernel<T = P::Storage>,
{
    let n_kernel = model.kernel.num_params();
    let n_theta = n_kernel + model.likelihood.num_params();
    let compiled = model.kernel.compile_as::<P::Storage>();
    let x64 = pack_points(&model.x_obs, model.n, model.d);
    let z64 = pack_points(&model.z_obs, model.m, model.d);
    let mut x_cast = P::Storage::empty_cols();
    let mut z_cast = P::Storage::empty_cols();
    let x = P::Storage::storage_cols(x64.as_ref(), &mut x_cast);
    let z = P::Storage::storage_cols(z64.as_ref(), &mut z_cast);
    let m = model.m;
    let n = model.n;
    let z_index = |idx: usize| -> Option<(usize, usize)> {
        if !include_z || idx < n_theta {
            None
        } else {
            let local = idx - n_theta;
            Some((local % m, local / m))
        }
    };
    if i < n_kernel && j < n_kernel {
        return kernel_theta_second::<M, _>(&compiled, x, z, n, i, j);
    }
    if i == n_kernel && j == n_kernel {
        return Ok(likelihood_var(m, n, model.likelihood.noise_variance()));
    }
    if i < n_theta && j < n_theta {
        return Ok(KernelVar::<P::Storage> {
            d_kmm: Mat::zeros(m, m),
            d_kmn: Mat::zeros(m, n),
            d_kdiag: lit::<P::Storage>(0.0),
            d_noise: lit::<P::Storage>(0.0),
        });
    }
    if let (Some((pi, ei)), Some((pj, ej))) = (z_index(i), z_index(j)) {
        return z_z_second::<M, _>(&compiled, x, z, pi, ei, pj, ej);
    }
    let (theta, (p, e)) = if i < n_theta {
        (
            i,
            z_index(j).ok_or_else(|| GprError::InvalidHyperparameter {
                reason: "expected a free inducing coordinate".to_owned(),
            })?,
        )
    } else {
        (
            j,
            z_index(i).ok_or_else(|| GprError::InvalidHyperparameter {
                reason: "expected a free inducing coordinate".to_owned(),
            })?,
        )
    };
    if theta == n_kernel {
        return Ok(KernelVar::<P::Storage> {
            d_kmm: Mat::zeros(m, m),
            d_kmn: Mat::zeros(m, n),
            d_kdiag: lit::<P::Storage>(0.0),
            d_noise: lit::<P::Storage>(0.0),
        });
    }
    theta_z_second::<M, _>(&compiled, x, z, theta, p, e)
}

pub(crate) fn kernel_theta_second<M: crate::math::KernelMath, T>(
    compiled: &CompiledKernel<T>,
    x: MatRef<'_, T>,
    z: MatRef<'_, T>,
    n: usize,
    i: usize,
    j: usize,
) -> Result<KernelVar<T>, GprError>
where
    T: StorageScalar + FillDistances,
    CompiledKernel<T>: GramKernel<T = T>,
{
    let m = z.nrows();
    let mut d_kmm = Mat::zeros(m, m);
    let mut scratch_mm = Mat::zeros(m, m);
    compiled.hess_points::<M>(z, d_kmm.as_mut(), i, j, Triangle::Full, scratch_mm.as_mut())?;
    let mut d_kmn = Mat::zeros(m, n);
    let mut scratch_mn = Mat::zeros(m, n);
    compiled.hess_cross_points::<M>(z, x, d_kmn.as_mut(), i, j, scratch_mn.as_mut())?;
    let mut diag = vec![lit::<T>(0.0); n];
    compiled.hess_diag_points::<M>(x, &mut diag, i, j)?;
    let d_kdiag = diag.iter().fold(lit::<T>(0.0), |acc, v| acc + *v);
    Ok(KernelVar::<T> {
        d_kmm,
        d_kmn,
        d_kdiag,
        d_noise: lit::<T>(0.0),
    })
}

pub(crate) fn z_z_second<M: crate::math::KernelMath, T>(
    compiled: &CompiledKernel<T>,
    x: MatRef<'_, T>,
    z: MatRef<'_, T>,
    p: usize,
    e: usize,
    q: usize,
    f: usize,
) -> Result<KernelVar<T>, GprError>
where
    T: StorageScalar + FillDistances,
    CompiledKernel<T>: GramKernel<T = T>,
{
    let m = z.nrows();
    let n = x.nrows();
    let mut h22 = Mat::zeros(m, m);
    compiled.hess_wrt_coord_dims::<M>(z, z, h22.as_mut(), e, f)?;
    let mut d_kmm = Mat::zeros(m, m);
    if p == q {
        for i in 0..m {
            if i != p {
                d_kmm[(i, p)] += h22[(i, p)];
                d_kmm[(p, i)] += h22[(i, p)];
            }
        }
    } else {
        let mut h12 = Mat::zeros(m, m);
        compiled.hess_wrt_coord_mixed::<M>(z, z, h12.as_mut(), e, f)?;
        let mut h21 = Mat::zeros(m, m);
        compiled.hess_wrt_coord_mixed::<M>(z, z, h21.as_mut(), f, e)?;
        d_kmm[(p, q)] = h12[(p, q)];
        d_kmm[(q, p)] = h21[(q, p)];
    }
    let mut d_kmn = Mat::zeros(m, n);
    if p == q {
        let mut h_xz = Mat::zeros(n, m);
        compiled.hess_wrt_coord_dims::<M>(x, z, h_xz.as_mut(), e, f)?;
        for col in 0..n {
            d_kmn[(p, col)] = h_xz[(col, p)];
        }
    }
    Ok(KernelVar::<T> {
        d_kmm,
        d_kmn,
        d_kdiag: lit::<T>(0.0),
        d_noise: lit::<T>(0.0),
    })
}

pub(crate) fn theta_z_second<M: crate::math::KernelMath, T>(
    compiled: &CompiledKernel<T>,
    x: MatRef<'_, T>,
    z: MatRef<'_, T>,
    theta: usize,
    p: usize,
    e: usize,
) -> Result<KernelVar<T>, GprError>
where
    T: StorageScalar + FillDistances,
    CompiledKernel<T>: GramKernel<T = T>,
{
    let m = z.nrows();
    let n = x.nrows();
    let mut g2 = Mat::zeros(m, m);
    compiled.hess_theta_coord_dim::<M>(z, z, g2.as_mut(), theta, e)?;
    let mut d_kmm = Mat::zeros(m, m);
    for i in 0..m {
        d_kmm[(i, p)] += g2[(i, p)];
        d_kmm[(p, i)] += g2[(i, p)];
    }
    let mut g_xz = Mat::zeros(n, m);
    compiled.hess_theta_coord_dim::<M>(x, z, g_xz.as_mut(), theta, e)?;
    let mut d_kmn = Mat::zeros(m, n);
    for col in 0..n {
        d_kmn[(p, col)] = g_xz[(col, p)];
    }
    Ok(KernelVar::<T> {
        d_kmm,
        d_kmn,
        d_kdiag: lit::<T>(0.0),
        d_noise: lit::<T>(0.0),
    })
}

pub(crate) fn chol_phi<T: StorageScalar>(l: MatRef<'_, T>, dk: MatRef<'_, T>) -> Mat<T> {
    let m = l.nrows();
    let mut tmp = Mat::zeros(m, m);
    copy_mat(dk, tmp.as_mut());
    solve_lower(l, tmp.as_mut());
    let mut u = Mat::zeros(m, m);
    for i in 0..m {
        for j in 0..m {
            u[(i, j)] = tmp[(j, i)];
        }
    }
    solve_lower(l, u.as_mut());
    let mut phi = Mat::zeros(m, m);
    for i in 0..m {
        for j in 0..m {
            phi[(i, j)] = u[(j, i)];
        }
    }
    phi
}

pub(crate) fn tril_half<T: StorageScalar>(phi: MatRef<'_, T>) -> Mat<T> {
    let m = phi.nrows();
    let mut out = Mat::zeros(m, m);
    for j in 0..m {
        for i in j..m {
            out[(i, j)] = if i == j {
                lit::<T>(0.5) * phi[(i, j)]
            } else {
                phi[(i, j)]
            };
        }
    }
    out
}

pub(crate) fn dphi_from<T: StorageScalar>(
    phi_i: MatRef<'_, T>,
    phi_l_j: MatRef<'_, T>,
    phi_dd: MatRef<'_, T>,
) -> Mat<T> {
    let m = phi_i.nrows();
    let mut out = Mat::zeros(m, m);
    for i in 0..m {
        for j in 0..m {
            let mut sum = phi_dd[(i, j)];
            for k in 0..m {
                sum -= phi_l_j[(i, k)] * phi_i[(k, j)];
                sum -= phi_i[(i, k)] * phi_l_j[(j, k)];
            }
            out[(i, j)] = sum;
        }
    }
    out
}

pub(crate) fn noise_plus_sym_prod<T: StorageScalar>(
    da: MatRef<'_, T>,
    a: MatRef<'_, T>,
    d_noise: T,
) -> Mat<T> {
    let m = da.nrows();
    let mut db = Mat::zeros(m, m);
    // `da Aᵀ + A daᵀ`. Diagonal terms are `2 Σ_k da_ik a_ik`, matching the scalar sum.
    gemm(
        db.as_mut(),
        Accum::Replace,
        da,
        a.transpose(),
        lit::<T>(1.0),
    );
    gemm(db.as_mut(), Accum::Add, a, da.transpose(), lit::<T>(1.0));
    for j in 0..m {
        db[(j, j)] += d_noise;
    }
    db
}

pub(crate) fn second_db<T: StorageScalar>(
    dai: MatRef<'_, T>,
    daj: MatRef<'_, T>,
    dda: MatRef<'_, T>,
    a: MatRef<'_, T>,
    dd_noise: T,
) -> Mat<T> {
    let m = a.nrows();
    let mut db = Mat::zeros(m, m);
    gemm(
        db.as_mut(),
        Accum::Replace,
        dda,
        a.transpose(),
        lit::<T>(1.0),
    );
    gemm(db.as_mut(), Accum::Add, a, dda.transpose(), lit::<T>(1.0));
    gemm(db.as_mut(), Accum::Add, dai, daj.transpose(), lit::<T>(1.0));
    gemm(db.as_mut(), Accum::Add, daj, dai.transpose(), lit::<T>(1.0));
    for col in 0..m {
        db[(col, col)] += dd_noise;
    }
    db
}

pub(crate) fn trace_solve<T: StorageScalar>(b_l: MatRef<'_, T>, db: MatRef<'_, T>) -> T {
    let mut solved = Mat::zeros(db.nrows(), db.ncols());
    copy_mat(db, solved.as_mut());
    solve_llt_in_place(b_l, solved.as_mut());
    let mut tr = lit::<T>(0.0);
    for i in 0..solved.nrows() {
        tr += solved[(i, i)];
    }
    tr
}

pub(crate) fn second_logdet_b<T: StorageScalar>(
    b_l: MatRef<'_, T>,
    dbi: MatRef<'_, T>,
    dbj: MatRef<'_, T>,
    ddb: MatRef<'_, T>,
) -> T {
    let mut si = Mat::zeros(dbi.nrows(), dbi.ncols());
    copy_mat(dbi, si.as_mut());
    solve_llt_in_place(b_l, si.as_mut());
    let mut sj = Mat::zeros(dbj.nrows(), dbj.ncols());
    copy_mat(dbj, sj.as_mut());
    solve_llt_in_place(b_l, sj.as_mut());
    let mut tr = lit::<T>(0.0);
    for i in 0..si.nrows() {
        for k in 0..si.ncols() {
            tr -= si[(k, i)] * sj[(i, k)];
        }
    }
    tr + trace_solve(b_l, ddb)
}

pub(crate) fn dw_from<T: StorageScalar>(
    b_l: MatRef<'_, T>,
    w: &[T],
    db: MatRef<'_, T>,
    u: &[T],
) -> Vec<T> {
    let m = w.len();
    let mut rhs = Mat::zeros(m, 1);
    let dbw = mat_vec(db, w);
    for i in 0..m {
        rhs[(i, 0)] = u[i] - dbw[i];
    }
    solve_llt_in_place(b_l, rhs.as_mut());
    let mut out = vec![lit::<T>(0.0); m];
    for i in 0..m {
        out[i] = rhs[(i, 0)];
    }
    out
}

pub(crate) fn solve_lower<T: StorageScalar>(l: MatRef<'_, T>, rhs: MatMut<'_, T>) {
    let n = l.nrows();
    let n_rhs = rhs.ncols();
    faer::linalg::triangular_solve::solve_lower_triangular_in_place(
        l,
        rhs,
        faer_par_dims(n, n_rhs),
    );
}

fn matvec_columns<T: StorageScalar>(a: MatRef<'_, T>, y: &[T], mut ay: MatMut<'_, T>) {
    let m = a.nrows();
    let n = a.ncols();
    if std::mem::size_of::<T>() == std::mem::size_of::<f32>() {
        for i in 0..m {
            let mut sum = 0.0f64;
            for j in 0..n {
                sum += a[(i, j)].to_f64() * y[j].to_f64();
            }
            ay[(i, 0)] = T::from_f64(sum);
        }
        return;
    }
    for i in 0..m {
        let mut sum = lit::<T>(0.0);
        for j in 0..n {
            sum += a[(i, j)] * y[j];
        }
        ay[(i, 0)] = sum;
    }
}

pub(crate) fn copy_mat<T: StorageScalar>(src: MatRef<'_, T>, mut dest: MatMut<'_, T>) {
    for j in 0..src.ncols() {
        for i in 0..src.nrows() {
            dest[(i, j)] = src[(i, j)];
        }
    }
}

pub(crate) fn mat_sub_mul<T: StorageScalar>(
    dest: &mut Mat<T>,
    left: MatRef<'_, T>,
    right: MatRef<'_, T>,
) {
    gemm(dest.as_mut(), Accum::Add, left, right, lit::<T>(-1.0));
}

pub(crate) fn mat_add_mul<T: StorageScalar>(
    dest: &mut Mat<T>,
    left: MatRef<'_, T>,
    right: MatRef<'_, T>,
) {
    gemm(dest.as_mut(), Accum::Add, left, right, lit::<T>(1.0));
}

pub(crate) fn mat_mul_into<T: StorageScalar>(
    dest: &mut Mat<T>,
    left: MatRef<'_, T>,
    right: MatRef<'_, T>,
) {
    gemm(dest.as_mut(), Accum::Replace, left, right, lit::<T>(1.0));
}

fn gemm<T: StorageScalar>(
    dest: MatMut<'_, T>,
    accum: Accum,
    lhs: MatRef<'_, T>,
    rhs: MatRef<'_, T>,
    alpha: T,
) {
    let par = faer_par_dims(dest.nrows(), dest.ncols());
    matmul(dest, accum, lhs, rhs, alpha, par);
}

pub(crate) fn frobenius_dot<T: StorageScalar>(a: MatRef<'_, T>, b: MatRef<'_, T>) -> T {
    let mut sum = lit::<T>(0.0);
    for col in 0..a.ncols() {
        for row in 0..a.nrows() {
            sum += a[(row, col)] * b[(row, col)];
        }
    }
    sum
}

pub(crate) fn dot<T: StorageScalar>(a: &[T], b: &[T]) -> T {
    let mut sum = lit::<T>(0.0);
    for (x, y) in a.iter().zip(b.iter()) {
        sum += *x * *y;
    }
    sum
}

pub(crate) fn quad_form<T: StorageScalar>(w: &[T], m: MatRef<'_, T>) -> T {
    let mw = mat_vec(m, w);
    dot(w, &mw)
}

pub(crate) fn mat_vec<T: StorageScalar>(m: MatRef<'_, T>, v: &[T]) -> Vec<T> {
    let mut out = vec![lit::<T>(0.0); m.nrows()];
    for i in 0..m.nrows() {
        let mut sum = lit::<T>(0.0);
        for j in 0..m.ncols() {
            sum += m[(i, j)] * v[j];
        }
        out[i] = sum;
    }
    out
}

fn round_mat<T: StorageScalar>(src: MatRef<'_, f64>) -> Mat<T> {
    let mut out = Mat::<T>::zeros(src.nrows(), src.ncols());
    for col in 0..src.ncols() {
        for row in 0..src.nrows() {
            out[(row, col)] = T::from_f64(src[(row, col)]);
        }
    }
    out
}

fn promote_mat<T: StorageScalar>(src: MatRef<'_, T>) -> Mat<f64> {
    let mut out = Mat::<f64>::zeros(src.nrows(), src.ncols());
    for col in 0..src.ncols() {
        for row in 0..src.nrows() {
            out[(row, col)] = src[(row, col)].to_f64();
        }
    }
    out
}

pub(crate) fn kernel_cross<M: crate::math::KernelMath, T>(
    compiled: &CompiledKernel<T>,
    x: MatRef<'_, T>,
    xs: MatRef<'_, T>,
) -> Result<Mat<T>, GprError>
where
    T: StorageScalar + FillDistances,
    CompiledKernel<T>: GramKernel<T = T>,
{
    let n = x.nrows();
    let q = xs.nrows();
    let mut out = Mat::zeros(n, q);
    let mut scratch = Mat::zeros(n, q);
    match compiled.coord_mode()? {
        CoordMode::Dist | CoordMode::Either => {
            let mut dist = Mat::zeros(n, q);
            let mut thread_scratch = Vec::new();
            T::write_cross(x, xs, dist.as_mut(), &mut thread_scratch);
            compiled.apply_cross::<M>(dist.as_ref(), out.as_mut(), scratch.as_mut())?;
        }
        CoordMode::Points => {
            compiled.apply_cross_points::<M>(x, xs, out.as_mut(), scratch.as_mut())?;
        }
        CoordMode::Mixed => {
            let mut dist = Mat::zeros(n, q);
            let mut thread_scratch = Vec::new();
            T::write_cross(x, xs, dist.as_mut(), &mut thread_scratch);
            compiled.apply_cross_mixed::<M>(
                dist.as_ref(),
                x,
                xs,
                out.as_mut(),
                scratch.as_mut(),
            )?;
        }
    }
    Ok(out)
}

pub(crate) fn gram_aat_plus_noise<T: StorageScalar>(a: MatRef<'_, T>, noise: f64) -> Mat<T> {
    let m = a.nrows();
    let n = a.ncols();
    let mut b = Mat::zeros(m, m);
    if std::mem::size_of::<T>() == std::mem::size_of::<f32>() {
        for i in 0..m {
            for j in 0..m {
                let mut sum = 0.0f64;
                for t in 0..n {
                    sum += a[(i, t)].to_f64() * a[(j, t)].to_f64();
                }
                b[(i, j)] = T::from_f64(sum);
            }
        }
    } else {
        gemm(b.as_mut(), Accum::Replace, a, a.transpose(), lit::<T>(1.0));
    }
    for j in 0..m {
        b[(j, j)] += lit::<T>(noise);
    }
    b
}

pub(crate) fn frobenius2<T: StorageScalar>(a: MatRef<'_, T>) -> T {
    let mut sum = lit::<T>(0.0);
    for col in 0..a.ncols() {
        for row in 0..a.nrows() {
            let v = a[(row, col)];
            sum += v * v;
        }
    }
    sum
}

pub(crate) fn solve_llt_in_place<T: StorageScalar>(l: MatRef<'_, T>, mut rhs: MatMut<'_, T>) {
    let n = l.nrows();
    let n_rhs = rhs.ncols();
    if std::mem::size_of::<T>() == std::mem::size_of::<f32>() {
        for col in 0..n_rhs {
            let mut y = vec![0.0f64; n];
            let mut x = vec![0.0f64; n];
            for i in 0..n {
                let mut sum = rhs[(i, col)].to_f64();
                for j in 0..i {
                    sum -= l[(i, j)].to_f64() * y[j];
                }
                y[i] = sum / l[(i, i)].to_f64();
            }
            for i in (0..n).rev() {
                let mut sum = y[i];
                for j in (i + 1)..n {
                    sum -= l[(j, i)].to_f64() * x[j];
                }
                x[i] = sum / l[(i, i)].to_f64();
            }
            for i in 0..n {
                rhs[(i, col)] = T::from_f64(x[i]);
            }
        }
        return;
    }
    let par = faer_par_dims(n, n_rhs);
    let req = llt::solve::solve_in_place_scratch::<T>(n, n_rhs, par);
    let mut buf = MemBuffer::new(req);
    let stack = MemStack::new(&mut buf);
    llt::solve::solve_in_place(l, rhs.as_mut(), par, stack);
}

pub(crate) fn validate_inducing(z: &[f64], m: usize, d: usize) -> Result<(), GprError> {
    if m == 0 || d == 0 {
        return Err(GprError::EmptyInput);
    }
    if z.len() % m == 0 {
        let z_dim = z.len() / m;
        if z_dim != d {
            return Err(GprError::DimensionMismatch {
                x_dim: z_dim,
                expected_dim: d,
            });
        }
    }
    let expected = m.checked_mul(d).ok_or(GprError::EmptyInput)?;
    if z.len() != expected {
        return Err(GprError::InvalidHyperparameter {
            reason: format!(
                "expected {expected} inducing feature values, got {}",
                z.len()
            ),
        });
    }
    if z.iter().any(|v| !v.is_finite()) {
        return Err(GprError::NonFiniteInput);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn vfe_neg_log_marginal_likelihood<T: StorageScalar>(
    a: MatRef<'_, T>,
    b_l: MatRef<'_, T>,
    w: &[T],
    y: &[f64],
    k_diag_sum: T,
    a_frobenius2: T,
    noise: f64,
    n: usize,
    m: usize,
) -> Result<f64, GprError> {
    let mut y_cast = T::empty_rows();
    let y_s = T::storage_rows(y, &mut y_cast);
    let noise_s = lit::<T>(noise);
    let mut log_det_b = lit::<T>(0.0);
    for i in 0..m {
        log_det_b += StorageScalar::ln(b_l[(i, i)]);
    }
    log_det_b *= lit::<T>(2.0);
    let n_minus_m = lit::<T>(n as f64) - lit::<T>(m as f64);
    let log_det = n_minus_m * StorageScalar::ln(noise_s) + log_det_b;
    let mut y_norm2 = lit::<T>(0.0);
    for value in y_s {
        y_norm2 += *value * *value;
    }
    let mut ay_dot_w = lit::<T>(0.0);
    for i in 0..m {
        let mut ay_i = lit::<T>(0.0);
        for j in 0..n {
            ay_i += a[(i, j)] * y_s[j];
        }
        ay_dot_w += ay_i * w[i];
    }
    let quad = (y_norm2 - ay_dot_w) / noise_s;
    let trace = (k_diag_sum - a_frobenius2) / (lit::<T>(2.0) * noise_s);
    let log_two_pi = lit::<T>((2.0 * std::f64::consts::PI).ln());
    let value = lit::<T>(0.5) * (lit::<T>(n as f64) * log_two_pi + log_det + quad) + trace;
    Ok(value.to_f64())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn vfe_predict<M: crate::math::KernelMath, P>(
    kernel: &KernelSpec,
    z_obs: &[f64],
    k_mm_l: MatRef<'_, P::Storage>,
    b_l: MatRef<'_, P::Storage>,
    predict_w: &[P::Refine],
    noise: f64,
    m: usize,
    d: usize,
    xs: &[f64],
    n_rows: usize,
    n_cols: usize,
    options: PredictOptions,
) -> Result<Prediction<P::Refine>, GprError>
where
    P: ModelPrecision + MeanDot,
    P::Storage: FillDistances,
    CompiledKernel<P::Storage>: GramKernel<T = P::Storage>,
{
    if n_cols != d {
        return Err(GprError::DimensionMismatch {
            x_dim: n_cols,
            expected_dim: d,
        });
    }
    validate_query(xs, n_rows, n_cols)?;
    if std::mem::size_of::<P::Storage>() == std::mem::size_of::<f32>() {
        let compiled64 = kernel.compile();
        let z64 = pack_points(z_obs, m, d);
        let mut k64 = Mat::<f64>::zeros(m, m);
        let mut scratch_k = Mat::<f64>::zeros(m, m);
        compiled64.apply_points::<M>(
            z64.as_ref(),
            k64.as_mut(),
            Triangle::Lower,
            scratch_k.as_mut(),
        )?;
        let req = llt::factor::cholesky_in_place_scratch::<f64>(m, faer_par(m), Default::default());
        let mut chol_scratch = MemBuffer::new(req);
        cholesky_lower_with_policy(
            &mut k64,
            &mut chol_scratch,
            k_mm_jitter_policy(),
            CholeskyStage::Predict,
        )?;
        let b64 = promote_mat(b_l);
        let w64: Vec<f64> = predict_w.iter().map(|value| value.to_f64()).collect();
        let pred = vfe_predict::<M, crate::precision::DoublePrecision>(
            kernel,
            z_obs,
            k64.as_ref(),
            b64.as_ref(),
            &w64,
            noise,
            m,
            d,
            xs,
            n_rows,
            n_cols,
            options,
        )?;
        return Ok(Prediction {
            mean: pred
                .mean
                .iter()
                .map(|value| P::Refine::from_f64(*value))
                .collect(),
            variance: pred
                .variance
                .iter()
                .map(|value| P::Refine::from_f64(*value))
                .collect(),
            variance_kind: pred.variance_kind,
        });
    }
    let compiled = kernel.compile_as::<P::Storage>();
    let z64 = pack_points(z_obs, m, d);
    let query64 = pack_points(xs, n_rows, n_cols);
    let mut k_sz = if std::mem::size_of::<P::Storage>() == std::mem::size_of::<f32>() {
        let compiled64 = kernel.compile();
        let cross = kernel_cross::<M, f64>(&compiled64, z64.as_ref(), query64.as_ref())?;
        let mut stored = Mat::<P::Storage>::zeros(m, n_rows);
        for col in 0..cross.ncols() {
            for row in 0..cross.nrows() {
                stored[(row, col)] = P::Storage::from_f64(cross[(row, col)]);
            }
        }
        stored
    } else {
        let mut z_cast = P::Storage::empty_cols();
        let z_mat = P::Storage::storage_cols(z64.as_ref(), &mut z_cast);
        let mut q_cast = P::Storage::empty_cols();
        let query_x = P::Storage::storage_cols(query64.as_ref(), &mut q_cast);
        kernel_cross::<M, _>(&compiled, z_mat, query_x)?
    };
    solve_lower(k_mm_l, k_sz.as_mut());
    let mut kss = vec![lit::<P::Storage>(0.0); n_rows];
    if std::mem::size_of::<P::Storage>() == std::mem::size_of::<f32>() {
        let compiled64 = kernel.compile();
        let mut diag = vec![0.0f64; n_rows];
        compiled64.fill_diag_points(query64.as_ref(), &mut diag)?;
        for (slot, value) in kss.iter_mut().zip(diag) {
            *slot = P::Storage::from_f64(value);
        }
    } else {
        let mut q_cast = P::Storage::empty_cols();
        let query_x = P::Storage::storage_cols(query64.as_ref(), &mut q_cast);
        compiled.fill_diag_points(query_x, &mut kss)?;
    }
    let mut binv_astar = k_sz.clone();
    solve_llt_in_place(b_l, binv_astar.as_mut());
    let noise_s = lit::<P::Storage>(noise);
    let zero = P::Refine::from_f64(0.0);
    let mut out = Prediction {
        mean: vec![zero; n_rows],
        variance: vec![zero; n_rows],
        variance_kind: options.variance_kind,
    };
    for col in 0..n_rows {
        let mut column = vec![lit::<P::Storage>(0.0); m];
        let mut a_norm = lit::<P::Storage>(0.0);
        let mut binv_norm = lit::<P::Storage>(0.0);
        for row in 0..m {
            let a_star = k_sz[(row, col)];
            column[row] = a_star;
            a_norm += a_star * a_star;
            let solved = binv_astar[(row, col)];
            binv_norm += a_star * solved;
        }
        let mut latent = kss[col] - a_norm + noise_s * binv_norm;
        if latent.to_f64() < 0.0 {
            latent = lit::<P::Storage>(0.0);
        }
        let latent_r = P::Refine::from_f64(latent.to_f64());
        out.mean[col] = P::mean_dot(&column, predict_w);
        out.variance[col] = match options.variance_kind {
            VarianceKind::Latent => latent_r,
            VarianceKind::Observation => P::Refine::from_f64(latent.to_f64() + noise),
        };
    }
    Ok(out)
}

pub(crate) fn chol_rank1_update<T: StorageScalar>(l: &mut Mat<T>, v: &mut [T]) {
    let n = l.nrows();
    for k in 0..n {
        let lkk = l[(k, k)];
        let vk = v[k];
        let r = storage_hypot(lkk, vk);
        let c = r / lkk;
        let s = vk / lkk;
        l[(k, k)] = r;
        for i in (k + 1)..n {
            let li = l[(i, k)];
            let vi = v[i];
            l[(i, k)] = (li + s * vi) / c;
            v[i] = c * vi - s * l[(i, k)];
        }
    }
}

pub(crate) fn chol_rank1_downdate<T: StorageScalar>(l: &mut Mat<T>, v: &mut [T]) -> bool {
    let n = l.nrows();
    for k in 0..n {
        let lkk = l[(k, k)];
        let vk = v[k];
        let r2 = lkk * lkk - vk * vk;
        let res = {
            let r2_f = r2.to_f64();
            r2_f <= 0.0 || !r2_f.is_finite()
        };
        if res {
            return false;
        }
        let r = storage_sqrt(r2);
        let c = r / lkk;
        let s = vk / lkk;
        l[(k, k)] = r;
        for i in (k + 1)..n {
            let li = l[(i, k)];
            let vi = v[i];
            l[(i, k)] = (li - s * vi) / c;
            v[i] = c * vi - s * l[(i, k)];
        }
    }
    true
}

pub(crate) fn refresh_w<T: StorageScalar>(a: MatRef<'_, T>, b_l: MatRef<'_, T>, y: &[T]) -> Vec<T> {
    let m = a.nrows();
    let mut ay = Mat::zeros(m, 1);
    matvec_columns(a, y, ay.as_mut());
    solve_llt_in_place(b_l, ay.as_mut());
    let mut w = vec![lit::<T>(0.0); m];
    for i in 0..m {
        w[i] = ay[(i, 0)];
    }
    w
}

pub(crate) fn append_column<T: StorageScalar>(a: &Mat<T>, col: MatRef<'_, T>) -> Mat<T> {
    let m = a.nrows();
    let n = a.ncols();
    let mut out = Mat::zeros(m, n + 1);
    for j in 0..n {
        for i in 0..m {
            out[(i, j)] = a[(i, j)];
        }
    }
    for i in 0..m {
        out[(i, n)] = col[(i, 0)];
    }
    out
}

pub(crate) fn remove_column<T: StorageScalar>(a: &Mat<T>, idx: usize) -> Mat<T> {
    let m = a.nrows();
    let n = a.ncols();
    let mut out = Mat::zeros(m, n - 1);
    let mut dest = 0;
    for j in 0..n {
        if j == idx {
            continue;
        }
        for i in 0..m {
            out[(i, dest)] = a[(i, j)];
        }
        dest += 1;
    }
    out
}

pub(crate) fn append_point(x: &[f64], n: usize, d: usize, x_new: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; (n + 1) * d];
    for dim in 0..d {
        for i in 0..n {
            out[i + (n + 1) * dim] = x[i + n * dim];
        }
        out[n + (n + 1) * dim] = x_new[dim];
    }
    out
}

pub(crate) fn remove_point(x: &[f64], n: usize, d: usize, idx: usize) -> Vec<f64> {
    let mut out = vec![0.0; (n - 1) * d];
    for dim in 0..d {
        let mut dest = 0;
        for i in 0..n {
            if i == idx {
                continue;
            }
            out[dest + (n - 1) * dim] = x[i + n * dim];
            dest += 1;
        }
    }
    out
}

pub(crate) fn point_at(x: &[f64], n: usize, d: usize, idx: usize) -> Vec<f64> {
    let mut out = vec![0.0; d];
    for dim in 0..d {
        out[dim] = x[idx + n * dim];
    }
    out
}

pub(crate) fn kernel_column<M: crate::math::KernelMath, T>(
    kernel: &KernelSpec,
    z: &[f64],
    m: usize,
    x_pt: &[f64],
    d: usize,
) -> Result<Mat<T>, GprError>
where
    T: StorageScalar + FillDistances,
    CompiledKernel<T>: GramKernel<T = T>,
{
    let compiled = kernel.compile_as::<T>();
    let z64 = pack_points(z, m, d);
    let x64 = pack_points(x_pt, 1, d);
    let mut z_cast = T::empty_cols();
    let mut x_cast = T::empty_cols();
    let z_mat = T::storage_cols(z64.as_ref(), &mut z_cast);
    let x_mat = T::storage_cols(x64.as_ref(), &mut x_cast);
    kernel_cross::<M, _>(&compiled, z_mat, x_mat)
}

pub(crate) fn kernel_diag_at<T>(kernel: &KernelSpec, x_pt: &[f64], d: usize) -> Result<T, GprError>
where
    T: StorageScalar + FillDistances,
    CompiledKernel<T>: GramKernel<T = T>,
{
    let compiled = kernel.compile_as::<T>();
    let x64 = pack_points(x_pt, 1, d);
    let mut x_cast = T::empty_cols();
    let x_mat = T::storage_cols(x64.as_ref(), &mut x_cast);
    let mut diag = vec![lit::<T>(0.0); 1];
    compiled.fill_diag_points(x_mat, &mut diag)?;
    Ok(diag[0])
}

pub(crate) fn solve_lmm<T: StorageScalar>(k_mm_l: MatRef<'_, T>, col: MatMut<'_, T>) {
    solve_lower(k_mm_l, col);
}

/// Appends one inducing point at the end by a bordered LLT of `K_mm` and `B`.
///
/// `A` gains a row. `k_diag_sum` is unchanged. `w` is solved from the new `B`.
#[allow(clippy::too_many_arguments)] // kernel, data, and new `Z` stay explicit
pub(crate) fn inducing_insert<M: crate::math::KernelMath, T>(
    state: &mut VfeState<T>,
    kernel: &KernelSpec,
    noise: f64,
    x: &[f64],
    n: usize,
    d: usize,
    y: &[f64],
    z: &[f64],
    m: usize,
    z_new: &[f64],
) -> Result<(), GprError>
where
    T: StorageScalar + FillDistances,
    CompiledKernel<T>: GramKernel<T = T>,
{
    validate_inducing(z, m, d)?;
    validate_inducing(z_new, 1, d)?;
    if n == 0 {
        return Err(GprError::EmptyInput);
    }
    let compiled = kernel.compile_as::<T>();
    let z64 = pack_points(z, m, d);
    let z_new64 = pack_points(z_new, 1, d);
    let x64 = pack_points(x, n, d);
    let mut z_cast = T::empty_cols();
    let mut zn_cast = T::empty_cols();
    let mut x_cast = T::empty_cols();
    let mut y_cast = T::empty_rows();
    let z_mat = T::storage_cols(z64.as_ref(), &mut z_cast);
    let z_new_mat = T::storage_cols(z_new64.as_ref(), &mut zn_cast);
    let x_mat = T::storage_cols(x64.as_ref(), &mut x_cast);
    let y_s = T::storage_rows(y, &mut y_cast);
    let mut k_zz = kernel_cross::<M, _>(&compiled, z_mat, z_new_mat)?;
    let k_nn = kernel_diag_at(kernel, z_new, d)?;
    let k_zx = kernel_cross::<M, _>(&compiled, z_new_mat, x_mat)?;
    solve_lmm(state.k_mm_l.as_ref(), k_zz.as_mut());
    let mut ell2 = k_nn;
    for i in 0..m {
        let li = k_zz[(i, 0)];
        ell2 -= li * li;
    }
    if ell2.to_f64() <= 0.0 || !ell2.to_f64().is_finite() {
        return Err(GprError::CholeskyFailed {
            jitter: 0.0,
            matrix_size: m + 1,
            stage: CholeskyStage::OnlineInsert,
        });
    }
    let ell = storage_sqrt(ell2);
    let mut a_new = vec![lit::<T>(0.0); n];
    let mut a_new_norm2 = lit::<T>(0.0);
    for j in 0..n {
        let mut dot = lit::<T>(0.0);
        for i in 0..m {
            dot += k_zz[(i, 0)] * state.a[(i, j)];
        }
        let value = (k_zx[(0, j)] - dot) / ell;
        a_new[j] = value;
        a_new_norm2 += value * value;
    }
    let mut v = vec![lit::<T>(0.0); m];
    for (i, slot) in v.iter_mut().enumerate() {
        let mut sum = lit::<T>(0.0);
        for (j, a_val) in a_new.iter().enumerate() {
            sum += state.a[(i, j)] * a_val;
        }
        *slot = sum;
    }
    let mut b_border = Mat::zeros(m, 1);
    for i in 0..m {
        b_border[(i, 0)] = v[i];
    }
    solve_lmm(state.b_l.as_ref(), b_border.as_mut());
    let mut beta2 = lit::<T>(noise) + a_new_norm2;
    for i in 0..m {
        let bi = b_border[(i, 0)];
        beta2 -= bi * bi;
    }
    if beta2.to_f64() <= 0.0 || !beta2.to_f64().is_finite() {
        return Err(GprError::CholeskyFailed {
            jitter: 0.0,
            matrix_size: m + 1,
            stage: CholeskyStage::OnlineInsert,
        });
    }
    let mut l_col = vec![lit::<T>(0.0); m];
    for i in 0..m {
        l_col[i] = k_zz[(i, 0)];
    }
    let mut b_col = vec![lit::<T>(0.0); m];
    for i in 0..m {
        b_col[i] = b_border[(i, 0)];
    }
    state.k_mm_l = append_chol_border(&state.k_mm_l, &l_col, ell);
    state.b_l = append_chol_border(&state.b_l, &b_col, storage_sqrt(beta2));
    state.a = append_row(&state.a, &a_new);
    state.a_frobenius2 += a_new_norm2;
    state.w = refresh_w(state.a.as_ref(), state.b_l.as_ref(), y_s);
    Ok(())
}

/// Drops inducing row `idx` by a trailing cholupdate of `L_mm`.
///
/// Reuses `K(Z, X) = L A`, drops that row, and solves the reduced `A`.
/// `B` is formed again from the new `A`. `k_diag_sum` is unchanged.
pub(crate) fn inducing_delete<T: StorageScalar>(
    state: &mut VfeState<T>,
    noise: f64,
    y: &[f64],
    idx: usize,
) -> Result<(), GprError> {
    let m = state.a.nrows();
    if m <= 1 {
        return Err(GprError::EmptyInput);
    }
    if idx >= m {
        return Err(GprError::InvalidHyperparameter {
            reason: "inducing index is out of range".to_owned(),
        });
    }
    let k_zx = mul_lower_left(state.k_mm_l.as_ref(), state.a.as_ref());
    let k_zx = remove_row(&k_zx, idx);
    state.k_mm_l = delete_chol_row(&state.k_mm_l, idx);
    let mut a = k_zx;
    solve_lower(state.k_mm_l.as_ref(), a.as_mut());
    let mut b = gram_aat_plus_noise(a.as_ref(), noise);
    factor_lower_in_place(&mut b, CholeskyStage::OnlineDelete)?;
    state.a = a;
    state.b_l = b;
    state.a_frobenius2 = frobenius2(state.a.as_ref());
    let mut y_cast = T::empty_rows();
    let y_s = T::storage_rows(y, &mut y_cast);
    state.w = refresh_w(state.a.as_ref(), state.b_l.as_ref(), y_s);
    Ok(())
}

fn append_chol_border<T: StorageScalar>(l: &Mat<T>, row: &[T], ell: T) -> Mat<T> {
    let m = l.nrows();
    let mut out = Mat::zeros(m + 1, m + 1);
    for j in 0..m {
        for i in j..m {
            out[(i, j)] = l[(i, j)];
        }
        out[(m, j)] = row[j];
    }
    out[(m, m)] = ell;
    out
}

fn delete_chol_row<T: StorageScalar>(l: &Mat<T>, idx: usize) -> Mat<T> {
    let m = l.nrows();
    let trail = m - idx - 1;
    let mut work = l.clone();
    if trail > 0 {
        let mut l22 = Mat::zeros(trail, trail);
        let mut v = vec![lit::<T>(0.0); trail];
        for j in 0..trail {
            for i in j..trail {
                l22[(i, j)] = work[(idx + 1 + i, idx + 1 + j)];
            }
            v[j] = work[(idx + 1 + j, idx)];
        }
        chol_rank1_update(&mut l22, &mut v);
        for j in 0..trail {
            for i in j..trail {
                work[(idx + 1 + i, idx + 1 + j)] = l22[(i, j)];
            }
        }
    }
    let mut out = Mat::zeros(m - 1, m - 1);
    let mut jo = 0;
    for j in 0..m {
        if j == idx {
            continue;
        }
        let mut io = 0;
        for i in 0..m {
            if i == idx {
                continue;
            }
            if io >= jo {
                out[(io, jo)] = work[(i, j)];
            }
            io += 1;
        }
        jo += 1;
    }
    out
}

fn append_row<T: StorageScalar>(a: &Mat<T>, row: &[T]) -> Mat<T> {
    let m = a.nrows();
    let n = a.ncols();
    let mut out = Mat::zeros(m + 1, n);
    for j in 0..n {
        for i in 0..m {
            out[(i, j)] = a[(i, j)];
        }
        out[(m, j)] = row[j];
    }
    out
}

fn remove_row<T: StorageScalar>(a: &Mat<T>, idx: usize) -> Mat<T> {
    let m = a.nrows();
    let n = a.ncols();
    let mut out = Mat::zeros(m - 1, n);
    let mut dest = 0;
    for i in 0..m {
        if i == idx {
            continue;
        }
        for j in 0..n {
            out[(dest, j)] = a[(i, j)];
        }
        dest += 1;
    }
    out
}

fn mul_lower_left<T: StorageScalar>(l: MatRef<'_, T>, a: MatRef<'_, T>) -> Mat<T> {
    let m = l.nrows();
    let n = a.ncols();
    let mut out = Mat::zeros(m, n);
    for j in 0..n {
        for i in 0..m {
            let mut sum = lit::<T>(0.0);
            for t in 0..=i {
                sum += l[(i, t)] * a[(t, j)];
            }
            out[(i, j)] = sum;
        }
    }
    out
}

fn factor_lower_in_place<T: StorageScalar>(
    mat: &mut Mat<T>,
    stage: CholeskyStage,
) -> Result<(), GprError> {
    let n = mat.nrows();
    let req = llt::factor::cholesky_in_place_scratch::<T>(n, faer_par(n), Default::default());
    let mut scratch = MemBuffer::new(req);
    cholesky_lower_with_policy(mat, &mut scratch, JitterPolicy::default(), stage)
}
