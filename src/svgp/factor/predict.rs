//! SVGP predictive mean and variance.

use crate::error::GprError;
use crate::kernel::{CompiledKernel, GramInputs, KernelScalar, KernelSpec, ScalarOps, Triangle};
use crate::kernel::{GatheredRect, TrainSources};
use crate::linalg::solve_lower;
use crate::policy::{JitterPolicy, with_kernel_exp};
use crate::precision::ModelPrecision;
use crate::sparse::{
    F64System, PredictBuffers, PredictScratch, QueryDist, SparseCore, cast_blocks, pack_into,
    predictive_variance, rect_slots, reset_prediction, view,
};
use crate::{PredictOptions, Prediction, PredictiveCovariance};
use faer::{Mat, MatRef};

/// The fitted SVGP a prediction reads, in transformed units.
pub(crate) struct SvgpSystem<'a, P: ModelPrecision> {
    pub(crate) kernel: &'a KernelSpec,
    pub(crate) k_mm_jitter: JitterPolicy,
    pub(crate) z: &'a [f64],
    /// The `Z × Z` squares of a distance kernel.
    pub(crate) zz: Option<&'a TrainSources<f64>>,
    pub(crate) k_mm_l: MatRef<'a, P::Storage>,
    pub(crate) q_mean: &'a [f64],
    pub(crate) q_l: MatRef<'a, f64>,
    pub(crate) noise: f64,
    pub(crate) m: usize,
    pub(crate) d: usize,
}

impl<'a, P: ModelPrecision> SvgpSystem<'a, P> {
    pub(crate) fn new(
        core: &'a SparseCore,
        k_mm_l: MatRef<'a, P::Storage>,
        q_mean: &'a [f64],
        q_l: MatRef<'a, f64>,
    ) -> Self {
        Self {
            kernel: &core.kernel,
            k_mm_jitter: core.jitter,
            z: &core.z_train,
            zz: core.dist.as_ref().map(|dist| &dist.zz),
            k_mm_l,
            q_mean,
            q_l,
            noise: core.likelihood.noise_variance(),
            m: core.m,
            d: core.d,
        }
    }
}

/// Maps the queries `xs` (original coordinates) through the input
/// transform, writes the diagonal SVGP prediction into `out`, and maps it
/// back through the target transform. After a warmup call with the same
/// shapes, this allocates nothing.
///
/// # Errors
///
/// Returns the query errors of [`SparseCore::map_query_into`], or
/// [`GprError::CholeskyFailed`] when a rounding storage cannot factor
/// `K_mm` in `f64`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn predict_svgp_into<P: ModelPrecision>(
    core: &SparseCore,
    sys: &SvgpSystem<'_, P>,
    xs: &[f64],
    n_rows: usize,
    n_cols: usize,
    qd: &QueryDist,
    options: PredictOptions,
    scratch: &mut PredictScratch<P::Storage>,
    out: &mut Prediction<P::Refine>,
) -> Result<(), GprError> {
    let mut mapped = std::mem::take(&mut scratch.xs);
    let result = core
        .map_query_into(xs, n_rows, n_cols, &mut mapped)
        .and_then(|()| {
            with_kernel_exp!(core.math, M => svgp_predict_into::<M, P>(
                sys, &mapped, n_rows, qd, options, scratch, out
            ))
        });
    scratch.xs = mapped;
    result?;
    core.inverse_prediction_in_place::<P>(out, &mut scratch.inverse)
}

/// Writes the diagonal SVGP prediction at the transformed queries `xs`
/// (`n_rows × d`, column-major) into `out`, in transformed units.
///
/// A rounding storage (`f32`) predicts in `f64`, as the other sparse models
/// do: `K_mm` factored again in `f64`; `q` is `f64` already.
pub(crate) fn svgp_predict_into<M: crate::math::KernelMath, P: ModelPrecision>(
    sys: &SvgpSystem<'_, P>,
    xs: &[f64],
    n_rows: usize,
    qd: &QueryDist,
    options: PredictOptions,
    scratch: &mut PredictScratch<P::Storage>,
    out: &mut Prediction<P::Refine>,
) -> Result<(), GprError> {
    reset_prediction(out, n_rows, options.variance_kind);
    let (m, d) = (sys.m, sys.d);
    let kind = options.variance_kind;
    let mut write = |col: usize, mean: f64, latent: f64| {
        out.mean[col] = P::Refine::from_f64(mean);
        out.variance[col] = P::Refine::from_f64(predictive_variance(latent, sys.noise, kind));
    };
    if <P::Storage as ScalarOps>::ROUNDS_FROM_F64 {
        let F64System {
            compiled,
            bufs,
            k_mm_l,
            ..
        } = scratch.f64_system::<M>(sys.kernel, sys.z, m, d, sys.k_mm_jitter, sys.zz)?;
        return svgp_latent::<M, f64>(
            compiled,
            bufs,
            sys,
            xs,
            n_rows,
            qd.zq.as_ref(),
            k_mm_l,
            &mut write,
        );
    }
    let PredictScratch {
        plan,
        storage: bufs,
        ..
    } = scratch;
    let compiled = plan.get(sys.kernel);
    svgp_latent::<M, P::Storage>(
        compiled,
        bufs,
        sys,
        xs,
        n_rows,
        qd.zq.as_ref(),
        sys.k_mm_l,
        &mut write,
    )
}

/// For each query column: `a* = L_mm⁻¹ k(Z, x*)`, the mean `a*ᵀ q_mean`,
/// and the latent variance `k(x*, x*) − ‖a*‖² + ‖L_qᵀ a*‖²` clamped at zero,
/// passed to `write(col, mean, latent)`, all in `S`.
#[allow(clippy::too_many_arguments)]
fn svgp_latent<M: crate::math::KernelMath, S: KernelScalar>(
    compiled: &CompiledKernel<S>,
    bufs: &mut PredictBuffers<S>,
    sys: &SvgpSystem<'_, impl ModelPrecision>,
    xs: &[f64],
    n_rows: usize,
    zq: Option<&GatheredRect<f64>>,
    k_mm_l: MatRef<'_, S>,
    write: &mut impl FnMut(usize, f64, f64),
) -> Result<(), GprError> {
    let (m, d) = (sys.m, sys.d);
    let zero = S::from_f64(0.0);
    let PredictBuffers {
        kernel,
        z: z_buf,
        query,
        k_sz,
        kss,
        ..
    } = bufs;
    let z_mat = pack_into(z_buf, sys.z, m, d);
    let q_mat = pack_into(query, xs, n_rows, d);
    let mut a_star = view(k_sz, m, n_rows);
    let zq = cast_blocks::<S>(zq);
    let zq = zq.as_ref().map(GatheredRect::table);
    kernel.cross_into::<M>(
        compiled,
        z_mat.as_ref(),
        q_mat.as_ref(),
        rect_slots(&zq),
        a_star.as_mut(),
    )?;
    solve_lower(k_mm_l, a_star.as_mut());
    kss.clear();
    kss.resize(n_rows, zero);
    compiled.fill_diag_points(q_mat.as_ref(), kss)?;
    for col in 0..n_rows {
        let (mut mean, mut a_norm, mut lt_norm) = (zero, zero, zero);
        for j in 0..m {
            let a = a_star[(j, col)];
            mean += a * S::from_f64(sys.q_mean[j]);
            a_norm += a * a;
            let mut lt_j = zero;
            for i in j..m {
                lt_j += S::from_f64(sys.q_l[(i, j)]) * a_star[(i, col)];
            }
            lt_norm += lt_j * lt_j;
        }
        let mut latent = kss[col] - a_norm + lt_norm;
        if latent.to_f64() < 0.0 {
            latent = zero;
        }
        write(col, mean.to_f64(), latent.to_f64());
    }
    Ok(())
}

/// The SVGP predictive mean and query–query covariance at `xs` (original
/// coordinates): the diagonal is [`predict_svgp_into`]'s variance, the
/// off-diagonal the latent `K** − AᵀA + UᵀU` with `A = L_mm⁻¹ K_m*` and
/// `U = L_qᵀ A`.
///
/// # Errors
///
/// Same as [`predict_svgp_into`].
pub(crate) fn predict_svgp_covariance<P: ModelPrecision>(
    core: &SparseCore,
    sys: &SvgpSystem<'_, P>,
    xs: &[f64],
    n_rows: usize,
    n_cols: usize,
    qd: &QueryDist,
    options: PredictOptions,
) -> Result<PredictiveCovariance<P::Refine>, GprError> {
    let mut scratch = PredictScratch::<P::Storage>::default();
    let mut mapped = Vec::new();
    core.map_query_into(xs, n_rows, n_cols, &mut mapped)?;
    let mut pred = Prediction::default();
    with_kernel_exp!(core.math, M => {
        svgp_predict_into::<M, P>(sys, &mapped, n_rows, qd, options, &mut scratch, &mut pred)?;
        if <P::Storage as ScalarOps>::ROUNDS_FROM_F64 {
            let compiled = scratch.plan64.get(sys.kernel);
            let latent = svgp_latent_covariance::<M, f64>(compiled, &mut scratch.f64, sys, n_rows, qd.qq.as_ref())?;
            core.finish_covariance::<P, f64>(latent.as_ref(), pred)
        } else {
            let compiled = scratch.plan.get(sys.kernel);
            let latent =
                svgp_latent_covariance::<M, P::Storage>(compiled, &mut scratch.storage, sys, n_rows, qd.qq.as_ref())?;
            core.finish_covariance::<P, P::Storage>(latent.as_ref(), pred)
        }
    })
}

/// `K** − AᵀA + UᵀU` (`q × q`) from the buffers [`svgp_predict_into`] left:
/// the packed queries and `A`.
fn svgp_latent_covariance<M: crate::math::KernelMath, S: KernelScalar>(
    compiled: &CompiledKernel<S>,
    bufs: &mut PredictBuffers<S>,
    sys: &SvgpSystem<'_, impl ModelPrecision>,
    q: usize,
    qq: Option<&GatheredRect<f64>>,
) -> Result<Mat<S>, GprError> {
    let (m, d) = (sys.m, sys.d);
    let zero = S::from_f64(0.0);
    let PredictBuffers {
        kernel,
        query,
        k_sz,
        ..
    } = bufs;
    let queries = view(query, q, d);
    let mut cov = Mat::<S>::zeros(q, q);
    match cast_blocks::<S>(qq) {
        Some(qq) => {
            let table = Some(qq.table());
            kernel.cross_into::<M>(
                compiled,
                queries.as_ref(),
                queries.as_ref(),
                rect_slots(&table),
                cov.as_mut(),
            )?;
        }
        None => kernel.gram::<M>(
            compiled,
            GramInputs::points(queries.as_ref()),
            cov.as_mut(),
            Triangle::Full,
        )?,
    }
    let a = k_sz.as_ref().submatrix(0, 0, m, q);
    let mut u = Mat::<S>::zeros(m, q);
    for col in 0..q {
        for j in 0..m {
            let mut lt_j = zero;
            for i in j..m {
                lt_j += S::from_f64(sys.q_l[(i, j)]) * a[(i, col)];
            }
            u[(j, col)] = lt_j;
        }
    }
    for col in 0..q {
        for row in 0..q {
            let mut a_dot = zero;
            let mut u_dot = zero;
            for k in 0..m {
                a_dot += a[(k, row)] * a[(k, col)];
                u_dot += u[(k, row)] * u[(k, col)];
            }
            cov[(row, col)] = cov[(row, col)] - a_dot + u_dot;
        }
    }
    Ok(cov)
}
