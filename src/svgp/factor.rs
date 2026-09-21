//! Whitened SVGP assembly, ELBO, and diagonal prediction.

use dyn_stack::MemBuffer;
use faer::linalg::cholesky::llt;
use faer::{Mat, MatRef};

use crate::error::{CholeskyStage, GprError};
use crate::gpr::JitterPolicy;
use crate::gpr::factor::{
    cholesky_lower_with_policy, pack_points, pack_points_into, require_param_len, symmetrize_lower,
    validate_query, validate_training,
};
use crate::kernel::{KernelSpec, Triangle};
use crate::likelihood::GaussianLikelihood;
use crate::sparse::{kernel_cross, validate_inducing};
use crate::workspace::{faer_par, faer_par_dims};
use crate::{PredictOptions, Prediction, VarianceKind};

use super::fitted::FittedSvgp;

pub(crate) struct SvgpState {
    pub(crate) k_mm_l: Mat<f64>,
    pub(crate) a: Mat<f64>,
    pub(crate) q_mean: Vec<f64>,
    pub(crate) q_l: Mat<f64>,
    pub(crate) k_diag: Vec<f64>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn assemble_fitted(
    kernel: KernelSpec,
    likelihood: GaussianLikelihood,
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    y: &[f64],
    z: &[f64],
    n_inducing: usize,
    q: Option<(Vec<f64>, Mat<f64>)>,
) -> Result<FittedSvgp, GprError> {
    let state = assemble_svgp(&kernel, x, n_rows, n_cols, y, z, n_inducing, q)?;
    Ok(FittedSvgp {
        kernel,
        likelihood,
        x_obs: x.to_vec(),
        z_obs: z.to_vec(),
        y: y.to_vec(),
        k_mm_l: state.k_mm_l,
        a: state.a,
        q_mean: state.q_mean,
        q_l: state.q_l,
        k_diag: state.k_diag,
        n: n_rows,
        m: n_inducing,
        d: n_cols,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn assemble_svgp(
    kernel: &KernelSpec,
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    y: &[f64],
    z: &[f64],
    n_inducing: usize,
    q: Option<(Vec<f64>, Mat<f64>)>,
) -> Result<SvgpState, GprError> {
    validate_training(x, n_rows, n_cols, y)?;
    validate_inducing(z, n_inducing, n_cols)?;
    let compiled = kernel.compile();
    let x_mat = pack_points(x, n_rows, n_cols);
    let z_mat = pack_points(z, n_inducing, n_cols);
    let mut k_mm = Mat::zeros(n_inducing, n_inducing);
    let mut scratch = Mat::zeros(n_inducing, n_inducing);
    compiled.apply_points(
        z_mat.as_ref(),
        k_mm.as_mut(),
        Triangle::Lower,
        scratch.as_mut(),
    )?;
    let req = llt::factor::cholesky_in_place_scratch::<f64>(
        n_inducing,
        faer_par(n_inducing),
        Default::default(),
    );
    let mut chol_scratch = MemBuffer::new(req);
    cholesky_lower_with_policy(
        &mut k_mm,
        &mut chol_scratch,
        JitterPolicy::default(),
        CholeskyStage::Fit,
    )?;
    // Same packed `X` and `Z` share a training White diagonal. Rectangular
    // `apply_cross` leaves White at zero.
    let mut a = if x == z {
        let mut gram = Mat::zeros(n_rows, n_rows);
        let mut gram_scratch = Mat::zeros(n_rows, n_rows);
        compiled.apply_points(
            x_mat.as_ref(),
            gram.as_mut(),
            Triangle::Lower,
            gram_scratch.as_mut(),
        )?;
        symmetrize_lower(gram.as_mut(), n_rows);
        gram
    } else {
        kernel_cross(&compiled, z_mat.as_ref(), x_mat.as_ref())?
    };
    faer::linalg::triangular_solve::solve_lower_triangular_in_place(
        k_mm.as_ref(),
        a.as_mut(),
        faer_par_dims(n_inducing, n_rows),
    );
    let mut k_diag = vec![0.0; n_rows];
    compiled.fill_diag_points(x_mat.as_ref(), &mut k_diag)?;
    let (q_mean, q_l) = match q {
        Some((mean, l)) => (mean, l),
        None => prior_q(n_inducing),
    };
    Ok(SvgpState {
        k_mm_l: k_mm,
        a,
        q_mean,
        q_l,
        k_diag,
    })
}

pub(crate) fn prior_q(m: usize) -> (Vec<f64>, Mat<f64>) {
    let mean = vec![0.0; m];
    let mut l = Mat::zeros(m, m);
    for i in 0..m {
        l[(i, i)] = 1.0;
    }
    (mean, l)
}

pub(crate) fn q_chol_len(m: usize) -> usize {
    m.saturating_mul(m.saturating_add(1)) / 2
}

pub(crate) fn q_param_len(m: usize) -> usize {
    m + q_chol_len(m)
}

pub(crate) fn pack_q(mean: &[f64], l: MatRef<'_, f64>, out: &mut [f64]) {
    let m = mean.len();
    debug_assert_eq!(out.len(), q_param_len(m));
    out[..m].copy_from_slice(mean);
    let mut k = 0;
    for j in 0..m {
        for i in j..m {
            out[m + k] = l[(i, j)];
            k += 1;
        }
    }
}

pub(crate) fn unpack_q(params: &[f64], m: usize) -> Result<(Vec<f64>, Mat<f64>), GprError> {
    require_param_len(params.len(), q_param_len(m))?;
    let mut mean = vec![0.0; m];
    for (i, slot) in mean.iter_mut().enumerate() {
        let v = params[i];
        if !v.is_finite() {
            return Err(GprError::NonFiniteInput);
        }
        *slot = v;
    }
    let mut l = Mat::zeros(m, m);
    let mut k = 0;
    for j in 0..m {
        for i in j..m {
            let v = params[m + k];
            if !v.is_finite() {
                return Err(GprError::NonFiniteInput);
            }
            if i == j && v <= 0.0 {
                return Err(GprError::InvalidHyperparameter {
                    reason: "variational Cholesky diagonal must be positive".to_owned(),
                });
            }
            l[(i, j)] = v;
            k += 1;
        }
    }
    Ok((mean, l))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn svgp_neg_elbo(
    a: MatRef<'_, f64>,
    q_mean: &[f64],
    q_l: MatRef<'_, f64>,
    y: &[f64],
    k_diag: &[f64],
    noise: f64,
    n: usize,
    m: usize,
) -> f64 {
    let mut tr_s = 0.0;
    let mut log_det_s = 0.0;
    for j in 0..m {
        log_det_s += q_l[(j, j)].ln();
        for i in j..m {
            let v = q_l[(i, j)];
            tr_s += v * v;
        }
    }
    log_det_s *= 2.0;
    let mut mean_norm2 = 0.0;
    for &v in q_mean {
        mean_norm2 += v * v;
    }
    let kl = 0.5 * (tr_s + mean_norm2 - m as f64 - log_det_s);
    let log_2pi_noise = (2.0 * std::f64::consts::PI * noise).ln();
    let inv_noise = 1.0 / noise;
    let mut expected_ll = 0.0;
    for col in 0..n {
        let mut mu = 0.0;
        let mut a_norm = 0.0;
        let mut lt_norm = 0.0;
        for j in 0..m {
            let a_j = a[(j, col)];
            mu += a_j * q_mean[j];
            a_norm += a_j * a_j;
            let mut lt_j = 0.0;
            for i in j..m {
                lt_j += q_l[(i, j)] * a[(i, col)];
            }
            lt_norm += lt_j * lt_j;
        }
        let var = k_diag[col] - a_norm + lt_norm;
        let resid = y[col] - mu;
        expected_ll += -0.5 * log_2pi_noise - 0.5 * inv_noise * (resid * resid + var);
    }
    -(expected_ll - kl)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn svgp_predict(
    kernel: &KernelSpec,
    z_obs: &[f64],
    k_mm_l: MatRef<'_, f64>,
    q_mean: &[f64],
    q_l: MatRef<'_, f64>,
    noise: f64,
    m: usize,
    d: usize,
    xs: &[f64],
    n_rows: usize,
    n_cols: usize,
    options: PredictOptions,
) -> Result<Prediction, GprError> {
    if n_cols != d {
        return Err(GprError::DimensionMismatch {
            x_dim: n_cols,
            expected_dim: d,
        });
    }
    validate_query(xs, n_rows, n_cols)?;
    let compiled = kernel.compile();
    let z_mat = pack_points(z_obs, m, d);
    let mut query_x = Mat::zeros(n_rows, n_cols);
    pack_points_into(xs, n_rows, n_cols, query_x.as_mut());
    let mut k_sz = kernel_cross(&compiled, z_mat.as_ref(), query_x.as_ref())?;
    faer::linalg::triangular_solve::solve_lower_triangular_in_place(
        k_mm_l,
        k_sz.as_mut(),
        faer_par_dims(m, n_rows),
    );
    let mut kss = vec![0.0; n_rows];
    compiled.fill_diag_points(query_x.as_ref(), &mut kss)?;
    let mut out = Prediction {
        mean: vec![0.0; n_rows],
        variance: vec![0.0; n_rows],
        variance_kind: options.variance_kind,
    };
    for col in 0..n_rows {
        let mut mean = 0.0;
        let mut a_norm = 0.0;
        let mut lt_norm = 0.0;
        for j in 0..m {
            let a_star = k_sz[(j, col)];
            mean += a_star * q_mean[j];
            a_norm += a_star * a_star;
            let mut lt_j = 0.0;
            for i in j..m {
                lt_j += q_l[(i, j)] * k_sz[(i, col)];
            }
            lt_norm += lt_j * lt_j;
        }
        let mut latent = kss[col] - a_norm + lt_norm;
        if latent < 0.0 {
            latent = 0.0;
        }
        out.mean[col] = mean;
        out.variance[col] = match options.variance_kind {
            VarianceKind::Latent => latent,
            VarianceKind::Observation => latent + noise,
        };
    }
    Ok(out)
}
