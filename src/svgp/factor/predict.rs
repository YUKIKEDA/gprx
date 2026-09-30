//! SVGP predictive mean and variance.

use crate::error::{CholeskyStage, GprError};
use crate::kernel::GramInputs;
use crate::kernel::{KernelScalar, KernelSpec, Triangle};
use crate::linalg::{cholesky_lower_with_backup, solve_lower};
use crate::policy::JitterPolicy;
use crate::policy::with_kernel_exp;
use crate::precision::ModelPrecision;
use crate::sparse::{PredictBuffers, PredictScratch, SparseCore, pack_into, view};
use crate::{PredictOptions, Prediction, VarianceKind};
use faer::{Mat, MatRef};

/// The fitted SVGP a prediction reads, in transformed units.
pub(crate) struct SvgpSystem<'a, P: ModelPrecision> {
    pub(crate) kernel: &'a KernelSpec,
    pub(crate) k_mm_jitter: JitterPolicy,
    pub(crate) z: &'a [f64],
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
/// shapes, only a mixed precision that refines the mean in `f64`
/// allocates.
///
/// # Errors
///
/// Returns the query errors of [`SparseCore::map_query_into`], or the
/// error of a mixed-precision mean refinement.
#[allow(clippy::too_many_arguments)]
pub(crate) fn predict_svgp_into<P: ModelPrecision>(
    core: &SparseCore,
    sys: &SvgpSystem<'_, P>,
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
            with_kernel_exp!(core.math, M => svgp_predict_into::<M, P>(
                sys, &mapped, n_rows, options, scratch, out
            ))
        });
    scratch.xs = mapped;
    result?;
    core.inverse_prediction_in_place::<P>(out, &mut scratch.inverse)
}

/// Writes the diagonal SVGP prediction at the transformed queries `xs`
/// (`n_rows × d`, column-major) into `out`, in transformed units.
pub(crate) fn svgp_predict_into<M: crate::math::KernelMath, P: ModelPrecision>(
    sys: &SvgpSystem<'_, P>,
    xs: &[f64],
    n_rows: usize,
    options: PredictOptions,
    scratch: &mut PredictScratch<P::Storage>,
    out: &mut Prediction<P::Refine>,
) -> Result<(), GprError> {
    let zero_r = P::Refine::from_f64(0.0);
    out.mean.clear();
    out.mean.resize(n_rows, zero_r);
    out.variance.clear();
    out.variance.resize(n_rows, zero_r);
    out.variance_kind = options.variance_kind;
    let (m, d) = (sys.m, sys.d);
    let zero = P::Storage::from_f64(0.0);
    let PredictScratch {
        plan,
        storage: bufs,
        plan64,
        f64: bufs64,
        k_mm64,
        k_mm64_backup,
        llt64,
        k_zs64,
        ..
    } = scratch;
    let compiled = plan.get(sys.kernel);
    let PredictBuffers {
        kernel,
        z: z_buf,
        query,
        k_sz,
        solved,
        kss,
        column,
    } = bufs;
    let z_mat = pack_into(z_buf, sys.z, m, d);
    let q_mat = pack_into(query, xs, n_rows, d);
    let mut a_star = view(k_sz, m, n_rows);
    kernel.cross_into::<M>(compiled, z_mat.as_ref(), q_mat.as_ref(), a_star.as_mut())?;
    let mut k_zs = view(solved, m, n_rows);
    k_zs.copy_from(a_star.as_ref());
    P::svgp_mean_reference(
        sys.k_mm_l,
        k_zs.as_ref(),
        &mut |l64, k64| {
            let compiled64 = plan64.get(sys.kernel);
            let PredictBuffers {
                kernel: ks64,
                z: z64_buf,
                query: q64_buf,
                ..
            } = &mut *bufs64;
            let z64 = pack_into(z64_buf, sys.z, m, d);
            let q64 = pack_into(q64_buf, xs, n_rows, d);
            if l64.nrows() != m || l64.ncols() != m {
                *l64 = Mat::zeros(m, m);
            }
            ks64.gram::<M>(
                compiled64,
                GramInputs::points(z64.as_ref()),
                l64.as_mut(),
                Triangle::Lower,
            )?;
            cholesky_lower_with_backup(
                l64,
                k_mm64_backup,
                PredictScratch::<P::Storage>::llt64(llt64, m),
                sys.k_mm_jitter.retry_jitters(),
                CholeskyStage::Predict,
            )?;
            if k64.nrows() != m || k64.ncols() != n_rows {
                *k64 = Mat::zeros(m, n_rows);
            }
            ks64.cross_into::<M>(compiled64, z64.as_ref(), q64.as_ref(), k64.as_mut())
        },
        k_mm64,
        k_zs64,
    )?;
    solve_lower(sys.k_mm_l, a_star.as_mut());
    kss.clear();
    kss.resize(n_rows, zero);
    compiled.fill_diag_points(q_mat.as_ref(), kss)?;
    column.clear();
    column.resize(m, zero);
    for col in 0..n_rows {
        let mut a_norm = zero;
        let mut lt_norm = zero;
        for j in 0..m {
            let a = a_star[(j, col)];
            column[j] = a;
            a_norm += a * a;
            let mut lt_j = zero;
            for i in j..m {
                lt_j += P::Storage::from_f64(sys.q_l[(i, j)]) * a_star[(i, col)];
            }
            lt_norm += lt_j * lt_j;
        }
        let mut latent = kss[col] - a_norm + lt_norm;
        if latent.to_f64() < 0.0 {
            latent = zero;
        }
        let k64_col: &[f64] = if P::REFINES_IN_F64 {
            k_zs64.col_as_slice(col)
        } else {
            &[]
        };
        out.mean[col] =
            P::mean_from_factor(sys.k_mm_l, column, sys.q_mean, k_mm64.as_ref(), k64_col)?;
        out.variance[col] = P::Refine::from_f64(match options.variance_kind {
            VarianceKind::Latent => latent.to_f64(),
            VarianceKind::Observation => latent.to_f64() + sys.noise,
        });
    }
    Ok(())
}
