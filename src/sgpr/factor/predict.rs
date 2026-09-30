//! VFE predictive mean and variance.

use super::lit;
use crate::error::{CholeskyStage, GprError};
use crate::kernel::GramInputs;
use crate::kernel::ScalarOps;
use crate::kernel::{CompiledKernel, KernelScalar, KernelSpec, Triangle};
use crate::linalg::{cholesky_lower_with_backup, solve_lower};
use crate::policy::{JitterPolicy, with_kernel_exp};
use crate::precision::{DoublePrecision, ModelPrecision};
use crate::sparse::{PredictBuffers, PredictScratch, SparseCore, pack_into, view};
use crate::{PredictOptions, Prediction, PredictiveCovariance, VarianceKind};
use faer::{Mat, MatRef};

/// The fitted VFE system a prediction reads, in transformed units.
pub(crate) struct VfeSystem<'a, P: ModelPrecision> {
    pub(crate) kernel: &'a KernelSpec,
    pub(crate) k_mm_jitter: JitterPolicy,
    pub(crate) z: &'a [f64],
    pub(crate) k_mm_l: MatRef<'a, P::Storage>,
    pub(crate) b_l: MatRef<'a, P::Storage>,
    pub(crate) predict_w: &'a [P::Refine],
    pub(crate) noise: f64,
    pub(crate) m: usize,
    pub(crate) d: usize,
}

impl<'a, P: ModelPrecision> VfeSystem<'a, P> {
    pub(crate) fn new(
        core: &'a SparseCore,
        k_mm_l: MatRef<'a, P::Storage>,
        b_l: MatRef<'a, P::Storage>,
        predict_w: &'a [P::Refine],
    ) -> Self {
        Self {
            kernel: &core.kernel,
            k_mm_jitter: core.jitter,
            z: &core.z_train,
            k_mm_l,
            b_l,
            predict_w,
            noise: core.likelihood.noise_variance(),
            m: core.m,
            d: core.d,
        }
    }
}

/// Maps the queries `xs` (original coordinates) through the input
/// transform, writes the diagonal VFE prediction into `out`, and maps it
/// back through the target transform. After a warmup call with the same
/// shapes, this allocates nothing.
///
/// # Errors
///
/// Returns the query errors of [`SparseCore::map_query_into`], or
/// [`GprError::CholeskyFailed`] when a rounding storage cannot factor
/// `K_mm` in `f64`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn predict_vfe_into<P: ModelPrecision>(
    core: &SparseCore,
    sys: &VfeSystem<'_, P>,
    xs: &[f64],
    n_rows: usize,
    n_cols: usize,
    options: PredictOptions,
    scratch: &mut PredictScratch<P::Storage>,
    out: &mut Prediction<P::Refine>,
) -> Result<(), GprError> {
    let mut mapped = std::mem::take(&mut scratch.xs);
    let result = core
        .map_query_into(xs, n_rows, n_cols, &mut mapped)
        .and_then(|()| {
            with_kernel_exp!(core.math, M => vfe_predict_into::<M, P>(
                sys, &mapped, n_rows, options, scratch, out
            ))
        });
    scratch.xs = mapped;
    result?;
    core.inverse_prediction_in_place::<P>(out, &mut scratch.inverse)
}

/// Writes the diagonal VFE prediction at the transformed queries `xs`
/// (`n_rows × d`, column-major) into `out`, in transformed units.
///
/// A rounding storage (`f32`) predicts in `f64`: `K_mm` factored again in
/// `f64`, `B` and the weights promoted.
pub(crate) fn vfe_predict_into<M: crate::math::KernelMath, P: ModelPrecision>(
    sys: &VfeSystem<'_, P>,
    xs: &[f64],
    n_rows: usize,
    options: PredictOptions,
    scratch: &mut PredictScratch<P::Storage>,
    out: &mut Prediction<P::Refine>,
) -> Result<(), GprError> {
    let zero = P::Refine::from_f64(0.0);
    out.mean.clear();
    out.mean.resize(n_rows, zero);
    out.variance.clear();
    out.variance.resize(n_rows, zero);
    out.variance_kind = options.variance_kind;
    let (m, d) = (sys.m, sys.d);
    let kind = options.variance_kind;
    if <P::Storage as ScalarOps>::ROUNDS_FROM_F64 {
        let PredictScratch {
            plan64,
            f64: bufs,
            k_mm64,
            k_mm64_backup,
            llt64,
            b_l64,
            w64,
            ..
        } = scratch;
        let compiled = plan64.get(sys.kernel);
        if k_mm64.nrows() != m || k_mm64.ncols() != m {
            *k_mm64 = Mat::zeros(m, m);
        }
        {
            let z64 = pack_into(&mut bufs.z, sys.z, m, d);
            bufs.kernel.gram::<M>(
                compiled,
                GramInputs::points(z64.as_ref()),
                k_mm64.as_mut(),
                Triangle::Lower,
            )?;
        }
        cholesky_lower_with_backup(
            k_mm64,
            k_mm64_backup,
            PredictScratch::<P::Storage>::llt64(llt64, m),
            sys.k_mm_jitter.retry_jitters(),
            CholeskyStage::Predict,
        )?;
        let mut b64 = view(b_l64, m, m);
        for col in 0..m {
            for row in 0..m {
                b64[(row, col)] = sys.b_l[(row, col)].to_f64();
            }
        }
        w64.clear();
        w64.extend(sys.predict_w.iter().map(|value| value.to_f64()));
        let w64 = w64.as_slice();
        return vfe_latent::<M, f64>(
            compiled,
            bufs,
            sys.z,
            m,
            d,
            xs,
            n_rows,
            k_mm64.as_ref(),
            b_l64.as_ref().submatrix(0, 0, m, m),
            sys.noise,
            |col, column, latent| {
                let mean = <DoublePrecision as ModelPrecision>::mean_dot(column, w64);
                out.mean[col] = P::Refine::from_f64(mean);
                out.variance[col] = P::Refine::from_f64(variance(latent, sys.noise, kind));
            },
        );
    }
    let PredictScratch {
        plan,
        storage: bufs,
        ..
    } = scratch;
    let compiled = plan.get(sys.kernel);
    vfe_latent::<M, P::Storage>(
        compiled,
        bufs,
        sys.z,
        m,
        d,
        xs,
        n_rows,
        sys.k_mm_l,
        sys.b_l,
        sys.noise,
        |col, column, latent| {
            out.mean[col] = P::mean_dot(column, sys.predict_w);
            out.variance[col] = P::Refine::from_f64(variance(latent.to_f64(), sys.noise, kind));
        },
    )
}

/// Latent or observation variance from the clamped latent variance.
fn variance(latent: f64, noise: f64, kind: VarianceKind) -> f64 {
    match kind {
        VarianceKind::Latent => latent,
        VarianceKind::Observation => latent + noise,
    }
}

/// For each query column: `a* = L_mm⁻¹ k(Z, x*)` and the latent variance
/// `k(x*, x*) − ‖a*‖² + σn² ‖L_B⁻¹ a*‖²` clamped at zero, passed to
/// `write(col, a*, latent)`.
#[allow(clippy::too_many_arguments)]
fn vfe_latent<M: crate::math::KernelMath, S: KernelScalar>(
    compiled: &CompiledKernel<S>,
    bufs: &mut PredictBuffers<S>,
    z: &[f64],
    m: usize,
    d: usize,
    xs: &[f64],
    n_rows: usize,
    k_mm_l: MatRef<'_, S>,
    b_l: MatRef<'_, S>,
    noise: f64,
    mut write: impl FnMut(usize, &[S], S),
) -> Result<(), GprError> {
    let PredictBuffers {
        kernel,
        z: z_buf,
        query,
        k_sz,
        solved,
        kss,
        column,
        ..
    } = bufs;
    let z_mat = pack_into(z_buf, z, m, d);
    let q_mat = pack_into(query, xs, n_rows, d);
    let mut a_star = view(k_sz, m, n_rows);
    kernel.cross_into::<M>(compiled, z_mat.as_ref(), q_mat.as_ref(), a_star.as_mut())?;
    solve_lower(k_mm_l, a_star.as_mut());
    kss.clear();
    kss.resize(n_rows, lit::<S>(0.0));
    compiled.fill_diag_points(q_mat.as_ref(), kss)?;
    let mut b_solved = view(solved, m, n_rows);
    b_solved.copy_from(a_star.as_ref());
    solve_lower(b_l, b_solved.as_mut());
    let noise_s = lit::<S>(noise);
    column.clear();
    column.resize(m, lit::<S>(0.0));
    for col in 0..n_rows {
        let mut a_norm = lit::<S>(0.0);
        let mut b_norm = lit::<S>(0.0);
        for row in 0..m {
            let a = a_star[(row, col)];
            column[row] = a;
            a_norm += a * a;
            let b = b_solved[(row, col)];
            b_norm += b * b;
        }
        let mut latent = kss[col] - a_norm + noise_s * b_norm;
        if latent.to_f64() < 0.0 {
            latent = lit::<S>(0.0);
        }
        write(col, column, latent);
    }
    Ok(())
}

/// The VFE predictive mean and query–query covariance at `xs` (original
/// coordinates): the diagonal is [`predict_vfe_into`]'s variance, the
/// off-diagonal the latent `K** − A*ᵀ A* + σn² S*ᵀ S*` with
/// `A* = L_mm⁻¹ K_m*` and `S* = L_B⁻¹ A*`.
///
/// # Errors
///
/// Same as [`predict_vfe_into`].
pub(crate) fn predict_vfe_covariance<P: ModelPrecision>(
    core: &SparseCore,
    sys: &VfeSystem<'_, P>,
    xs: &[f64],
    n_rows: usize,
    n_cols: usize,
    options: PredictOptions,
) -> Result<PredictiveCovariance<P::Refine>, GprError> {
    let mut scratch = PredictScratch::<P::Storage>::default();
    let mut mapped = Vec::new();
    core.map_query_into(xs, n_rows, n_cols, &mut mapped)?;
    let mut pred = Prediction::default();
    with_kernel_exp!(core.math, M => {
        vfe_predict_into::<M, P>(sys, &mapped, n_rows, options, &mut scratch, &mut pred)?;
        if <P::Storage as ScalarOps>::ROUNDS_FROM_F64 {
            let compiled = scratch.plan64.get(sys.kernel);
            let latent = vfe_latent_covariance::<M, f64>(
                compiled, &mut scratch.f64, sys.m, sys.d, n_rows, sys.noise,
            )?;
            core.finish_covariance::<P, f64>(latent.as_ref(), pred)
        } else {
            let compiled = scratch.plan.get(sys.kernel);
            let latent = vfe_latent_covariance::<M, P::Storage>(
                compiled, &mut scratch.storage, sys.m, sys.d, n_rows, sys.noise,
            )?;
            core.finish_covariance::<P, P::Storage>(latent.as_ref(), pred)
        }
    })
}

/// `K** − A*ᵀ A* + σn² S*ᵀ S*` (`q × q`) from the buffers [`vfe_latent`]
/// left: the packed queries, `A*`, and `S*`.
fn vfe_latent_covariance<M: crate::math::KernelMath, S: KernelScalar>(
    compiled: &CompiledKernel<S>,
    bufs: &mut PredictBuffers<S>,
    m: usize,
    d: usize,
    q: usize,
    noise: f64,
) -> Result<Mat<S>, GprError> {
    let PredictBuffers {
        kernel,
        query,
        k_sz,
        solved,
        ..
    } = bufs;
    let queries = view(query, q, d);
    let mut cov = Mat::<S>::zeros(q, q);
    kernel.gram::<M>(
        compiled,
        GramInputs::points(queries.as_ref()),
        cov.as_mut(),
        Triangle::Full,
    )?;
    let a_star = k_sz.as_ref().submatrix(0, 0, m, q);
    let s_star = solved.as_ref().submatrix(0, 0, m, q);
    let noise_s = lit::<S>(noise);
    for col in 0..q {
        for row in 0..q {
            let mut a_dot = lit::<S>(0.0);
            let mut s_dot = lit::<S>(0.0);
            for k in 0..m {
                a_dot += a_star[(k, row)] * a_star[(k, col)];
                s_dot += s_star[(k, row)] * s_star[(k, col)];
            }
            cov[(row, col)] = cov[(row, col)] - a_dot + noise_s * s_dot;
        }
    }
    Ok(cov)
}
