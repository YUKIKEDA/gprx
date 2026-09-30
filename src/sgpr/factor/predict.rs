//! VFE predictive mean and variance.

use super::lit;
use crate::data::{pack_points, validate_query};
use crate::error::{CholeskyStage, GprError};
use crate::kernel::GramInputs;
use crate::kernel::ScalarOps;
use crate::kernel::{KernelScalar, KernelSpec, Triangle};
use crate::linalg::{
    cholesky_lower_with_retries, llt_scratch, promote_mat, solve_llt, solve_lower,
};
use crate::precision::ModelPrecision;
use crate::sparse::{k_mm_jitter_policy, kernel_cross};
use crate::{PredictOptions, Prediction, VarianceKind};
use faer::{Mat, MatRef};

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
    P: ModelPrecision,
{
    if n_cols != d {
        return Err(GprError::DimensionMismatch {
            x_dim: n_cols,
            expected_dim: d,
        });
    }
    validate_query(xs, n_rows, n_cols)?;
    if <P::Storage as ScalarOps>::ROUNDS_FROM_F64 {
        let compiled64 = kernel.compile();
        let z64 = pack_points(z_obs, m, d);
        let mut k64 = Mat::<f64>::zeros(m, m);
        let mut scratch_k = Mat::<f64>::zeros(m, m);
        compiled64.eval_gram::<M>(
            GramInputs::points(z64.as_ref()),
            k64.as_mut(),
            Triangle::Lower,
            scratch_k.as_mut(),
            &mut Vec::new(),
        )?;
        let mut chol_scratch = llt_scratch::<f64>(m);
        cholesky_lower_with_retries(
            &mut k64,
            &mut chol_scratch,
            k_mm_jitter_policy().retry_jitters(),
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
    let mut k_sz = if <P::Storage as ScalarOps>::ROUNDS_FROM_F64 {
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
    if <P::Storage as ScalarOps>::ROUNDS_FROM_F64 {
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
    solve_llt(b_l, binv_astar.as_mut());
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
