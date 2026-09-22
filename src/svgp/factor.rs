//! Whitened SVGP assembly, ELBO, and diagonal prediction.

use dyn_stack::MemBuffer;
use faer::linalg::cholesky::llt;
use faer::{Mat, MatRef};

use rand::RngExt;
use rand::rngs::SmallRng;

use crate::error::{CholeskyStage, GprError};
use crate::gpr::JitterPolicy;
use crate::gpr::factor::{
    cholesky_lower_with_policy, pack_points, pack_points_into, require_param_len, symmetrize_lower,
    validate_query, validate_training,
};
use crate::kernel::{KernelSpec, Triangle};
use crate::likelihood::GaussianLikelihood;
use crate::optimizer::{Adam, chain_logit_grad, log_theta_to_z, z_to_log_theta};
use crate::param::Interval;
use crate::rng::small_rng;
use crate::sgpr::{kernel_cross, validate_inducing};
use crate::workspace::{faer_par, faer_par_dims};
use crate::{PredictOptions, Prediction, VarianceKind};

use super::fitted::FittedSvgp;

// `K_mm` only. Public default stays Fixed(0). Forrester m=16 / ℓ=1 is not PD in f64.
fn k_mm_jitter_policy() -> JitterPolicy {
    JitterPolicy::adaptive(1e-8, 10.0, 5, 1e-3).unwrap_or_default()
}

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
        k_mm_jitter_policy(),
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

pub(crate) fn svgp_value_and_gradient(
    model: &FittedSvgp,
    out: &mut [f64],
    batch: &[usize],
) -> Result<f64, GprError> {
    let n = model.n;
    let m = model.m;
    let n_kernel = model.kernel.num_params();
    let n_theta = n_kernel + model.likelihood.num_params();
    require_param_len(out.len(), n_theta + q_param_len(m))?;
    if batch.is_empty() {
        return Err(GprError::EmptyInput);
    }
    out.fill(0.0);
    let noise = model.likelihood.noise_variance();
    let inv_noise = 1.0 / noise;
    let scale = n as f64 / batch.len() as f64;
    let kl = accumulate_kl_grad(model, out, n_theta, m);
    let (ell, resid2_var) = accumulate_data_q_grad(model, out, batch, n_theta, m, inv_noise, scale);
    out[n_kernel] = -scale * (-0.5 * batch.len() as f64 + 0.5 * inv_noise * resid2_var);
    accumulate_kernel_grad(model, out, batch, n_kernel, m, inv_noise, scale)?;
    Ok(kl - scale * ell)
}

fn accumulate_kl_grad(model: &FittedSvgp, out: &mut [f64], n_theta: usize, m: usize) -> f64 {
    let mut tr_s = 0.0;
    let mut log_det_s = 0.0;
    let mut packed = 0;
    for j in 0..m {
        log_det_s += model.q_l[(j, j)].ln();
        for i in j..m {
            let v = model.q_l[(i, j)];
            tr_s += v * v;
            out[n_theta + m + packed] = if i == j { v - 1.0 / v } else { v };
            packed += 1;
        }
    }
    log_det_s *= 2.0;
    let mut mean_norm2 = 0.0;
    for k in 0..m {
        let v = model.q_mean[k];
        mean_norm2 += v * v;
        out[n_theta + k] = v;
    }
    0.5 * (tr_s + mean_norm2 - m as f64 - log_det_s)
}

fn accumulate_data_q_grad(
    model: &FittedSvgp,
    out: &mut [f64],
    batch: &[usize],
    n_theta: usize,
    m: usize,
    inv_noise: f64,
    scale: f64,
) -> (f64, f64) {
    let noise = model.likelihood.noise_variance();
    let log_2pi_noise = (2.0 * std::f64::consts::PI * noise).ln();
    let mut ell = 0.0;
    let mut resid2_var = 0.0;
    let mut u = vec![0.0; m];
    for &col in batch {
        let (var, resid) = point_stats(model, col, m, &mut u);
        ell += -0.5 * log_2pi_noise - 0.5 * inv_noise * (resid * resid + var);
        resid2_var += resid * resid + var;
        for k in 0..m {
            out[n_theta + k] -= scale * inv_noise * resid * model.a[(k, col)];
        }
        let mut packed = 0;
        for (j, u_j) in u.iter().enumerate() {
            for i in j..m {
                out[n_theta + m + packed] += scale * inv_noise * u_j * model.a[(i, col)];
                packed += 1;
            }
        }
    }
    (ell, resid2_var)
}

fn point_stats(model: &FittedSvgp, col: usize, m: usize, u: &mut [f64]) -> (f64, f64) {
    let mut mu = 0.0;
    let mut a_norm = 0.0;
    for (j, u_j) in u.iter_mut().enumerate() {
        let a_j = model.a[(j, col)];
        mu += a_j * model.q_mean[j];
        a_norm += a_j * a_j;
        let mut lt_j = 0.0;
        for i in j..m {
            lt_j += model.q_l[(i, j)] * model.a[(i, col)];
        }
        *u_j = lt_j;
    }
    let lt_norm: f64 = u.iter().map(|v| v * v).sum();
    let var = model.k_diag[col] - a_norm + lt_norm;
    let resid = model.y[col] - mu;
    (var, resid)
}

fn accumulate_kernel_grad(
    model: &FittedSvgp,
    out: &mut [f64],
    batch: &[usize],
    n_kernel: usize,
    m: usize,
    inv_noise: f64,
    scale: f64,
) -> Result<(), GprError> {
    let compiled = model.kernel.compile();
    let x_mat = pack_points(&model.x_obs, model.n, model.d);
    let z_mat = pack_points(&model.z_obs, model.m, model.d);
    let same_xz = model.x_obs == model.z_obs;
    let mut u = vec![0.0; m];
    let mut da_col = vec![0.0; m];
    let mut l_t_da = vec![0.0; m];
    for (param_idx, slot) in out.iter_mut().take(n_kernel).enumerate() {
        let (d_a, d_kdiag) = kernel_theta_tangents(
            &compiled,
            x_mat.as_ref(),
            z_mat.as_ref(),
            model,
            same_xz,
            param_idx,
        )?;
        let mut g = 0.0;
        for &col in batch {
            let (_var, resid) = point_stats(model, col, m, &mut u);
            for (r, dest) in da_col.iter_mut().enumerate() {
                *dest = d_a[(r, col)];
            }
            let mut dmu = 0.0;
            let mut d_anorm = 0.0;
            for (r, da_r) in da_col.iter().enumerate() {
                dmu += da_r * model.q_mean[r];
                d_anorm += 2.0 * model.a[(r, col)] * da_r;
            }
            for (j, dest) in l_t_da.iter_mut().enumerate() {
                let mut acc = 0.0;
                for (i, da_i) in da_col.iter().enumerate().skip(j) {
                    acc += model.q_l[(i, j)] * da_i;
                }
                *dest = acc;
            }
            let d_lt: f64 = u
                .iter()
                .zip(l_t_da.iter())
                .map(|(uj, lj)| 2.0 * uj * lj)
                .sum();
            let dvar = d_kdiag[col] - d_anorm + d_lt;
            g += inv_noise * resid * dmu - 0.5 * inv_noise * dvar;
        }
        *slot = -scale * g;
    }
    Ok(())
}

fn kernel_theta_tangents(
    compiled: &crate::kernel::CompiledKernel,
    x: faer::MatRef<'_, f64>,
    z: faer::MatRef<'_, f64>,
    model: &FittedSvgp,
    same_xz: bool,
    param_idx: usize,
) -> Result<(Mat<f64>, Vec<f64>), GprError> {
    let m = model.m;
    let n = model.n;
    let mut d_kmm = Mat::zeros(m, m);
    let mut scratch_mm = Mat::zeros(m, m);
    compiled.grad_points(
        z,
        d_kmm.as_mut(),
        param_idx,
        Triangle::Full,
        scratch_mm.as_mut(),
    )?;
    let mut d_kmn = if same_xz {
        let mut gram = Mat::zeros(n, n);
        let mut scratch = Mat::zeros(n, n);
        compiled.grad_points(
            x,
            gram.as_mut(),
            param_idx,
            Triangle::Full,
            scratch.as_mut(),
        )?;
        gram
    } else {
        let mut cross = Mat::zeros(m, n);
        let mut scratch = Mat::zeros(m, n);
        compiled.grad_cross_points(z, x, cross.as_mut(), param_idx, scratch.as_mut())?;
        cross
    };
    let mut d_xx = Mat::zeros(n, n);
    let mut scratch_xx = Mat::zeros(n, n);
    compiled.grad_points(
        x,
        d_xx.as_mut(),
        param_idx,
        Triangle::Lower,
        scratch_xx.as_mut(),
    )?;
    let mut d_kdiag = vec![0.0; n];
    for i in 0..n {
        d_kdiag[i] = d_xx[(i, i)];
    }
    let mut d_l = Mat::zeros(m, m);
    cholesky_sensitivity(model.k_mm_l.as_ref(), d_kmm.as_ref(), d_l.as_mut(), m);
    for col in 0..n {
        for i in 0..m {
            let mut acc = 0.0;
            for k in 0..=i {
                acc += d_l[(i, k)] * model.a[(k, col)];
            }
            d_kmn[(i, col)] -= acc;
        }
    }
    faer::linalg::triangular_solve::solve_lower_triangular_in_place(
        model.k_mm_l.as_ref(),
        d_kmn.as_mut(),
        faer_par_dims(m, n),
    );
    Ok((d_kmn, d_kdiag))
}

fn cholesky_sensitivity(
    l: faer::MatRef<'_, f64>,
    d_k: faer::MatRef<'_, f64>,
    mut d_l: faer::MatMut<'_, f64>,
    m: usize,
) {
    for j in 0..m {
        let mut acc = d_k[(j, j)];
        for k in 0..j {
            acc -= 2.0 * l[(j, k)] * d_l[(j, k)];
        }
        d_l[(j, j)] = acc / (2.0 * l[(j, j)]);
        for i in j + 1..m {
            let mut acc = d_k[(i, j)];
            for k in 0..j {
                acc -= d_l[(i, k)] * l[(j, k)] + l[(i, k)] * d_l[(j, k)];
            }
            acc -= l[(i, j)] * d_l[(j, j)];
            d_l[(i, j)] = acc / l[(j, j)];
        }
    }
}

fn theta_intervals(
    kernel: &KernelSpec,
    likelihood: &GaussianLikelihood,
) -> Result<Vec<Interval>, GprError> {
    let n_kernel = kernel.num_params();
    let n_theta = n_kernel + likelihood.num_params();
    let mut out = vec![Interval::DEFAULT_POSITIVE; n_theta];
    let mut offset = 0;
    kernel.write_intervals(&mut out[..n_kernel], &mut offset)?;
    out[n_kernel] = likelihood.bounds();
    Ok(out)
}

fn user_to_unconstrained(
    user: &[f64],
    n_theta: usize,
    m: usize,
    intervals: &[Interval],
) -> Result<Vec<f64>, GprError> {
    let mut z = vec![0.0; user.len()];
    let mapped = log_theta_to_z(&user[..n_theta], intervals)?;
    z[..n_theta].copy_from_slice(&mapped);
    z[n_theta..n_theta + m].copy_from_slice(&user[n_theta..n_theta + m]);
    let mut packed = 0;
    for j in 0..m {
        for i in j..m {
            let idx = n_theta + m + packed;
            z[idx] = if i == j { user[idx].ln() } else { user[idx] };
            packed += 1;
        }
    }
    Ok(z)
}

fn unconstrained_to_user(
    z: &[f64],
    n_theta: usize,
    m: usize,
    intervals: &[Interval],
) -> Result<Vec<f64>, GprError> {
    let mut user = vec![0.0; z.len()];
    let mapped = z_to_log_theta(&z[..n_theta], intervals)?;
    user[..n_theta].copy_from_slice(&mapped);
    user[n_theta..n_theta + m].copy_from_slice(&z[n_theta..n_theta + m]);
    let mut packed = 0;
    for j in 0..m {
        for i in j..m {
            let idx = n_theta + m + packed;
            user[idx] = if i == j { z[idx].exp() } else { z[idx] };
            packed += 1;
        }
    }
    Ok(user)
}

fn user_grad_to_unconstrained(
    user: &[f64],
    z: &[f64],
    intervals: &[Interval],
    g_user: &[f64],
    g_z: &mut [f64],
    n_theta: usize,
    m: usize,
) {
    g_z.copy_from_slice(g_user);
    chain_logit_grad(
        &z[..n_theta],
        intervals,
        &user[..n_theta],
        &mut g_z[..n_theta],
    );
    let mut packed = 0;
    for j in 0..m {
        for i in j..m {
            let idx = n_theta + m + packed;
            if i == j {
                g_z[idx] = g_user[idx] * user[idx];
            }
            packed += 1;
        }
    }
}

fn shuffle_indices(idx: &mut [usize], rng: &mut SmallRng) {
    for i in (1..idx.len()).rev() {
        let j = rng.random_range(0..=i);
        idx.swap(i, j);
    }
}

pub(crate) fn run_adam_fit(model: &mut FittedSvgp, adam: &Adam) -> Result<(), GprError> {
    let n = model.n;
    let m = model.m;
    let n_theta = model.kernel.num_params() + model.likelihood.num_params();
    let p = model.num_params();
    let intervals = theta_intervals(&model.kernel, &model.likelihood)?;
    let mut user = vec![0.0; p];
    model.get_params(&mut user)?;
    let mut z = user_to_unconstrained(&user, n_theta, m, &intervals)?;
    let mut moment1 = vec![0.0; p];
    let mut moment2 = vec![0.0; p];
    let mut g_user = vec![0.0; p];
    let mut g_z = vec![0.0; p];
    let mut order: Vec<usize> = (0..n).collect();
    let mut rng = small_rng(adam.seed());
    let mut timestep = 0_u64;
    let batch_size = adam.batch_size();
    for _ in 0..adam.epochs() {
        shuffle_indices(&mut order, &mut rng);
        let mut start = 0;
        while start < n {
            let end = start.saturating_add(batch_size).min(n);
            let batch = &order[start..end];
            user = unconstrained_to_user(&z, n_theta, m, &intervals)?;
            model.set_params(&user)?;
            svgp_value_and_gradient(model, &mut g_user, batch)?;
            user_grad_to_unconstrained(&user, &z, &intervals, &g_user, &mut g_z, n_theta, m);
            adam.step(&mut z, &g_z, &mut moment1, &mut moment2, &mut timestep);
            start = end;
        }
    }
    user = unconstrained_to_user(&z, n_theta, m, &intervals)?;
    model.set_params(&user)
}
