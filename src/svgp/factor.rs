//! Whitened SVGP assembly, ELBO, and diagonal prediction.

use dyn_stack::MemBuffer;
use faer::linalg::cholesky::llt;
use faer::{Mat, MatRef};
use wide::f64x4;

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
use crate::sgpr::{kernel_cross, mat_mul_into, validate_inducing};
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
    let cache = point_cache(model, batch, m);
    let (ell, resid2_var) =
        accumulate_data_q_grad(model, out, batch, &cache, n_theta, inv_noise, scale);
    out[n_kernel] = -scale * (-0.5 * batch.len() as f64 + 0.5 * inv_noise * resid2_var);
    accumulate_kernel_grad(model, out, batch, &cache, n_kernel, inv_noise, scale)?;
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

struct ColMajor<'a> {
    data: &'a [f64],
    m: usize,
}

impl<'a> ColMajor<'a> {
    fn new(mat: MatRef<'a, f64>) -> Option<Self> {
        let m = mat.nrows();
        let n = mat.ncols();
        if m == 0 || n == 0 {
            return Some(Self { data: &[], m });
        }
        if mat.row_stride() != 1 || mat.col_stride() != m as isize {
            return None;
        }
        let len = m * n;
        // SAFETY: row stride is +1 and column stride equals `m`, so the
        // values are one contiguous column-major buffer.
        let data = unsafe { std::slice::from_raw_parts(mat.as_ptr(), len) };
        Some(Self { data, m })
    }

    fn col(&self, j: usize) -> &'a [f64] {
        let start = j * self.m;
        &self.data[start..start + self.m]
    }
}

fn load4(src: &[f64], i: usize) -> f64x4 {
    f64x4::new([src[i], src[i + 1], src[i + 2], src[i + 3]])
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    let n = left.len().min(right.len());
    let mut acc = f64x4::new([0.0; 4]);
    let mut i = 0;
    while i + 4 <= n {
        acc += load4(left, i) * load4(right, i);
        i += 4;
    }
    let parts = acc.to_array();
    let mut sum = parts[0] + parts[1] + parts[2] + parts[3];
    while i < n {
        sum += left[i] * right[i];
        i += 1;
    }
    sum
}

fn norm2(values: &[f64]) -> f64 {
    dot(values, values)
}

struct PointCache {
    resid: Vec<f64>,
    var: Vec<f64>,
    /// Columns are batch rows. `u[(j, b)] = (Lᵀ a)[j]` for that row.
    u: Mat<f64>,
}

fn point_cache(model: &FittedSvgp, batch: &[usize], m: usize) -> PointCache {
    let b = batch.len();
    let a = batch_columns(model.a.as_ref(), batch);
    let mut u = Mat::zeros(m, b);
    mat_mul_into(&mut u, model.q_l.transpose(), a.as_ref());
    let mut resid = vec![0.0; b];
    let mut var = vec![0.0; b];
    let a_ref = a.as_ref();
    if let (Some(a_cm), Some(u_cm)) = (ColMajor::new(a_ref), ColMajor::new(u.as_ref())) {
        let mean = model.q_mean.as_slice();
        for (b_idx, &col) in batch.iter().enumerate() {
            let a_col = a_cm.col(b_idx);
            let u_col = u_cm.col(b_idx);
            let mu = dot(a_col, mean);
            var[b_idx] = model.k_diag[col] - norm2(a_col) + norm2(u_col);
            resid[b_idx] = model.y[col] - mu;
        }
    } else {
        for (b_idx, &col) in batch.iter().enumerate() {
            let mut mu = 0.0;
            let mut a_norm = 0.0;
            let mut lt_norm = 0.0;
            for j in 0..m {
                let a_j = a_ref[(j, b_idx)];
                let u_j = u[(j, b_idx)];
                mu += a_j * model.q_mean[j];
                a_norm += a_j * a_j;
                lt_norm += u_j * u_j;
            }
            var[b_idx] = model.k_diag[col] - a_norm + lt_norm;
            resid[b_idx] = model.y[col] - mu;
        }
    }
    PointCache { resid, var, u }
}

enum BatchCols<'a> {
    Full(MatRef<'a, f64>),
    Owned(Mat<f64>),
}

impl<'a> BatchCols<'a> {
    fn as_ref(&self) -> MatRef<'_, f64> {
        match self {
            Self::Full(mat) => *mat,
            Self::Owned(mat) => mat.as_ref(),
        }
    }
}

fn batch_columns<'a>(full: MatRef<'a, f64>, batch: &[usize]) -> BatchCols<'a> {
    let n = full.ncols();
    if batch.len() == n && batch.iter().enumerate().all(|(i, col)| *col == i) {
        return BatchCols::Full(full);
    }
    let m = full.nrows();
    let mut owned = Mat::zeros(m, batch.len());
    for (b_idx, &col) in batch.iter().enumerate() {
        for row in 0..m {
            owned[(row, b_idx)] = full[(row, col)];
        }
    }
    BatchCols::Owned(owned)
}

fn accumulate_data_q_grad(
    model: &FittedSvgp,
    out: &mut [f64],
    batch: &[usize],
    cache: &PointCache,
    n_theta: usize,
    inv_noise: f64,
    scale: f64,
) -> (f64, f64) {
    let m = model.m;
    let noise = model.likelihood.noise_variance();
    let log_2pi_noise = (2.0 * std::f64::consts::PI * noise).ln();
    let mut ell = 0.0;
    let mut resid2_var = 0.0;
    let b = batch.len();
    let mut resid_col = Mat::zeros(b, 1);
    for (b_idx, &resid) in cache.resid.iter().enumerate() {
        let var = cache.var[b_idx];
        ell += -0.5 * log_2pi_noise - 0.5 * inv_noise * (resid * resid + var);
        resid2_var += resid * resid + var;
        resid_col[(b_idx, 0)] = resid;
    }
    let a = batch_columns(model.a.as_ref(), batch);
    let mut mean = Mat::zeros(m, 1);
    mat_mul_into(&mut mean, a.as_ref(), resid_col.as_ref());
    let c = scale * inv_noise;
    for k in 0..m {
        out[n_theta + k] -= c * mean[(k, 0)];
    }
    // `M_ij = Σ_b a_i u_j` for `i ≥ j`, the packed `L` derivative.
    let mut gram = Mat::zeros(m, m);
    mat_mul_into(&mut gram, a.as_ref(), cache.u.transpose());
    let mut packed = 0;
    for j in 0..m {
        for i in j..m {
            out[n_theta + m + packed] += c * gram[(i, j)];
            packed += 1;
        }
    }
    (ell, resid2_var)
}

fn accumulate_kernel_grad(
    model: &FittedSvgp,
    out: &mut [f64],
    batch: &[usize],
    cache: &PointCache,
    n_kernel: usize,
    inv_noise: f64,
    scale: f64,
) -> Result<(), GprError> {
    let m = model.m;
    let compiled = model.kernel.compile();
    let x_mat = pack_points(&model.x_obs, model.n, model.d);
    let z_mat = pack_points(&model.z_obs, model.m, model.d);
    let same_xz = model.x_obs == model.z_obs;
    let mut ard_cross = match &compiled {
        crate::kernel::CompiledKernel::RbfArd(leaf) if !same_xz => {
            Some(leaf.grad_cross_all_from_coords(z_mat.as_ref(), x_mat.as_ref())?)
        }
        _ => None,
    };
    for (param_idx, slot) in out.iter_mut().take(n_kernel).enumerate() {
        let pre = ard_cross
            .as_mut()
            .map(|mats| std::mem::replace(&mut mats[param_idx], Mat::zeros(0, 0)));
        let (d_a, d_kdiag) = kernel_theta_tangents(
            &compiled,
            x_mat.as_ref(),
            z_mat.as_ref(),
            model,
            same_xz,
            param_idx,
            pre,
        )?;
        let da_b = batch_columns(d_a.as_ref(), batch);
        let mut lt = Mat::zeros(m, batch.len());
        mat_mul_into(&mut lt, model.q_l.transpose(), da_b.as_ref());
        let da_ref = da_b.as_ref();
        let mut g = 0.0;
        if let (Some(da_cm), Some(a_cm), Some(u_cm), Some(lt_cm)) = (
            ColMajor::new(da_ref),
            ColMajor::new(model.a.as_ref()),
            ColMajor::new(cache.u.as_ref()),
            ColMajor::new(lt.as_ref()),
        ) {
            let mean = model.q_mean.as_slice();
            for (b_idx, &col) in batch.iter().enumerate() {
                let da_col = da_cm.col(b_idx);
                let dmu = dot(da_col, mean);
                let d_anorm = 2.0 * dot(a_cm.col(col), da_col);
                let d_lt = 2.0 * dot(u_cm.col(b_idx), lt_cm.col(b_idx));
                let dvar = d_kdiag[col] - d_anorm + d_lt;
                let resid = cache.resid[b_idx];
                g += inv_noise * resid * dmu - 0.5 * inv_noise * dvar;
            }
        } else {
            for (b_idx, &col) in batch.iter().enumerate() {
                let resid = cache.resid[b_idx];
                let mut dmu = 0.0;
                let mut d_anorm = 0.0;
                let mut d_lt = 0.0;
                for r in 0..m {
                    let da_r = da_ref[(r, b_idx)];
                    dmu += da_r * model.q_mean[r];
                    d_anorm += 2.0 * model.a[(r, col)] * da_r;
                    d_lt += 2.0 * cache.u[(r, b_idx)] * lt[(r, b_idx)];
                }
                let dvar = d_kdiag[col] - d_anorm + d_lt;
                g += inv_noise * resid * dmu - 0.5 * inv_noise * dvar;
            }
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
    pre_cross: Option<Mat<f64>>,
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
    let mut d_kmn = if let Some(pre) = pre_cross {
        pre
    } else if same_xz {
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
    let mut d_kdiag = vec![0.0; n];
    if same_xz {
        for i in 0..n {
            d_kdiag[i] = d_kmn[(i, i)];
        }
    } else {
        compiled.grad_diag_points(x, &mut d_kdiag, param_idx)?;
    }
    let mut d_l = Mat::zeros(m, m);
    cholesky_sensitivity(model.k_mm_l.as_ref(), d_kmm.as_ref(), d_l.as_mut(), m);
    // Upper of `d_l` stays zero, so this is the lower-triangular product.
    crate::sgpr::mat_sub_mul(&mut d_kmn, d_l.as_ref(), model.a.as_ref());
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
