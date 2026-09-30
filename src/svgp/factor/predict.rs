//! SVGP predictive mean and variance.

use crate::data::{pack_points, validate_query};
use crate::error::{CholeskyStage, GprError};
use crate::kernel::GramInputs;
use crate::kernel::ScalarOps;
use crate::kernel::{KernelScalar, KernelSpec, Triangle};
use crate::linalg::{cholesky_lower_with_retries, llt_scratch, solve_lower};
use crate::precision::ModelPrecision;
use crate::sparse::{k_mm_jitter_policy, kernel_cross};
use crate::{PredictOptions, Prediction, VarianceKind};
use faer::{Mat, MatRef};

#[allow(clippy::too_many_arguments)]
pub(crate) fn svgp_predict<M: crate::math::KernelMath, P: ModelPrecision>(
    kernel: &KernelSpec,
    z_obs: &[f64],
    k_mm_l: MatRef<'_, P::Storage>,
    q_mean: &[f64],
    q_l: MatRef<'_, f64>,
    noise: f64,
    m: usize,
    d: usize,
    xs: &[f64],
    n_rows: usize,
    n_cols: usize,
    options: PredictOptions,
) -> Result<Prediction<P::Refine>, GprError>
where
{
    if n_cols != d {
        return Err(GprError::DimensionMismatch {
            x_dim: n_cols,
            expected_dim: d,
        });
    }
    validate_query(xs, n_rows, n_cols)?;
    let compiled = kernel.compile_as::<P::Storage>();
    let z64 = pack_points(z_obs, m, d);
    let query64 = pack_points(xs, n_rows, n_cols);
    let mut z_cast = P::Storage::empty_cols();
    let mut q_cast = P::Storage::empty_cols();
    let z_mat = P::Storage::storage_cols(z64.as_ref(), &mut z_cast);
    let query_x = P::Storage::storage_cols(query64.as_ref(), &mut q_cast);
    let mut k_sz = kernel_cross::<M, _>(&compiled, z_mat, query_x)?;
    let rhs = k_sz.clone();
    solve_lower(k_mm_l, k_sz.as_mut());
    let mut kss = vec![P::Storage::from_f64(0.0); n_rows];
    compiled.fill_diag_points(query_x, &mut kss)?;
    let zero = P::Refine::from_f64(0.0);
    let mut out = Prediction {
        mean: vec![zero; n_rows],
        variance: vec![zero; n_rows],
        variance_kind: options.variance_kind,
    };
    for col in 0..n_rows {
        let mut solved = vec![P::Storage::from_f64(0.0); m];
        let mut rhs_col = vec![P::Storage::from_f64(0.0); m];
        let mut a_norm = P::Storage::from_f64(0.0);
        let mut lt_norm = P::Storage::from_f64(0.0);
        for j in 0..m {
            let a_star = k_sz[(j, col)];
            solved[j] = a_star;
            rhs_col[j] = rhs[(j, col)];
            a_norm += a_star * a_star;
            let mut lt_j = P::Storage::from_f64(0.0);
            for i in j..m {
                lt_j += P::Storage::from_f64(q_l[(i, j)]) * k_sz[(i, col)];
            }
            lt_norm += lt_j * lt_j;
        }
        let mut latent = kss[col] - a_norm + lt_norm;
        if latent.to_f64() < 0.0 {
            latent = P::Storage::from_f64(0.0);
        }
        let latent_r = P::Refine::from_f64(latent.to_f64());
        let mut query_row = vec![0.0; d];
        for dim in 0..d {
            query_row[dim] = xs[col + n_rows * dim];
        }
        out.mean[col] = P::mean_from_factor(k_mm_l, &solved, &rhs_col, q_mean, &|| {
            f64_mean_reference::<M>(kernel, z_obs, &query_row, m)
        })?;
        out.variance[col] = match options.variance_kind {
            VarianceKind::Latent => latent_r,
            VarianceKind::Observation => P::Refine::from_f64(latent.to_f64() + noise),
        };
    }
    let _ = kernel;
    Ok(out)
}

/// `K_mm`'s `f64` factor and `k(Z, x_*)` at one query point, the
/// [`crate::ReevaluateKernel`] reference of the mixed-precision mean.
pub(super) fn f64_mean_reference<M: crate::math::KernelMath>(
    kernel: &KernelSpec,
    z_obs: &[f64],
    query: &[f64],
    m: usize,
) -> Result<(Mat<f64>, Vec<f64>), GprError> {
    let d = query.len();
    let compiled = kernel.compile();
    let z64 = pack_points(z_obs, m, d);
    let q64 = pack_points(query, 1, d);
    let mut k_mm = Mat::<f64>::zeros(m, m);
    let mut scratch = Mat::<f64>::zeros(m, m);
    compiled.eval_gram::<M>(
        GramInputs::points(z64.as_ref()),
        k_mm.as_mut(),
        Triangle::Lower,
        scratch.as_mut(),
        &mut Vec::new(),
    )?;
    let mut chol_scratch = llt_scratch::<f64>(m);
    cholesky_lower_with_retries(
        &mut k_mm,
        &mut chol_scratch,
        k_mm_jitter_policy().retry_jitters(),
        CholeskyStage::Predict,
    )?;
    let k_star = kernel_cross::<M, _>(&compiled, z64.as_ref(), q64.as_ref())?;
    Ok((k_mm, (0..m).map(|i| k_star[(i, 0)]).collect()))
}
