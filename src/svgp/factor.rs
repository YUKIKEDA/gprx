//! Whitened SVGP assembly, ELBO, and diagonal prediction.

use faer::{Mat, MatMut, MatRef};

use rand::RngExt;
use rand::rngs::SmallRng;

use crate::data::{pack_points, validate_inducing, validate_query, validate_training};
use crate::error::{CholeskyStage, GprError};
use crate::policy::KernelExp;
use crate::sparse::SparseCore;

use crate::kernel::GramInputs;
use crate::kernel::ScalarOps;
use crate::kernel::{CompiledKernel, KernelScalar, KernelSpec, Triangle};
use crate::likelihood::GaussianLikelihood;
use crate::linalg::{
    cholesky_lower_with_retries, dot_f64x4, llt_scratch, mat_mul_into, norm2_f64x4, solve_lower,
    symmetrize_lower,
};
use crate::optimizer::{Adam, chain_logit_grad, log_theta_to_z, z_to_log_theta};
use crate::param::Interval;
use crate::precision::ModelPrecision;
use crate::rng::small_rng;
use crate::sparse::{k_mm_jitter_policy, kernel_cross};
use crate::{PredictOptions, Prediction, VarianceKind};

use super::fitted::FittedSvgp;

// `K_mm` only. Public default stays Fixed(0). Forrester m=16 / ℓ=1 is not PD in f64.

pub(crate) struct SvgpState<T: KernelScalar> {
    pub(crate) k_mm_l: Mat<T>,
    pub(crate) a: Mat<T>,
    pub(crate) q_mean: Vec<f64>,
    pub(crate) q_l: Mat<f64>,
    pub(crate) k_diag: Vec<T>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn assemble_fitted<M: crate::math::KernelMath, P>(
    kernel: KernelSpec,
    likelihood: GaussianLikelihood,
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    y: &[f64],
    z: &[f64],
    n_inducing: usize,
    q: Option<(Vec<f64>, Mat<f64>)>,
) -> Result<FittedSvgp<P>, GprError>
where
    P: ModelPrecision,
{
    let state = assemble_svgp::<M, P::Storage>(&kernel, x, n_rows, n_cols, y, z, n_inducing, q)?;
    Ok(FittedSvgp {
        core: SparseCore {
            kernel,
            likelihood,
            x_obs: x.to_vec(),
            z_obs: z.to_vec(),
            y: y.to_vec(),
            n: n_rows,
            m: n_inducing,
            d: n_cols,
            math: KernelExp::of::<M>(),
        },
        k_mm_l: state.k_mm_l,
        a: state.a,
        q_mean: state.q_mean,
        q_l: state.q_l,
        k_diag: state.k_diag,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn assemble_svgp<M: crate::math::KernelMath, T>(
    kernel: &KernelSpec,
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    y: &[f64],
    z: &[f64],
    n_inducing: usize,
    q: Option<(Vec<f64>, Mat<f64>)>,
) -> Result<SvgpState<T>, GprError>
where
    T: KernelScalar,
{
    validate_training(x, n_rows, n_cols, y)?;
    validate_inducing(z, n_inducing, n_cols)?;
    let compiled = kernel.compile_as::<T>();
    let x64 = pack_points(x, n_rows, n_cols);
    let z64 = pack_points(z, n_inducing, n_cols);
    let mut x_cast = T::empty_cols();
    let mut z_cast = T::empty_cols();
    let x_mat = T::storage_cols(x64.as_ref(), &mut x_cast);
    let z_mat = T::storage_cols(z64.as_ref(), &mut z_cast);
    let mut k_mm = Mat::zeros(n_inducing, n_inducing);
    let mut scratch = Mat::zeros(n_inducing, n_inducing);
    compiled.eval_gram::<M>(
        GramInputs::points(z_mat),
        k_mm.as_mut(),
        Triangle::Lower,
        scratch.as_mut(),
        &mut Vec::new(),
    )?;
    let mut chol_scratch = llt_scratch::<T>(n_inducing);
    cholesky_lower_with_retries(
        &mut k_mm,
        &mut chol_scratch,
        k_mm_jitter_policy().retry_jitters(),
        CholeskyStage::Fit,
    )?;
    // Same packed `X` and `Z` share a training White diagonal. Rectangular
    // `apply_cross` leaves White at zero.
    let mut a = if x == z {
        let mut gram = Mat::zeros(n_rows, n_rows);
        let mut gram_scratch = Mat::zeros(n_rows, n_rows);
        compiled.eval_gram::<M>(
            GramInputs::points(x_mat),
            gram.as_mut(),
            Triangle::Lower,
            gram_scratch.as_mut(),
            &mut Vec::new(),
        )?;
        symmetrize_lower(gram.as_mut(), n_rows);
        gram
    } else {
        kernel_cross::<M, _>(&compiled, z_mat, x_mat)?
    };
    solve_lower(k_mm.as_ref(), a.as_mut());
    let mut k_diag = vec![T::from_f64(0.0); n_rows];
    compiled.fill_diag_points(x_mat, &mut k_diag)?;
    let (q_mean, q_l) = match q {
        Some((mean, l)) => (mean, l),
        None => prior_q(n_inducing),
    };
    Ok(SvgpState::<T> {
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
    crate::data::require_count(params.len(), q_param_len(m), "parameters")?;
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
pub(crate) fn svgp_neg_elbo<T: KernelScalar>(
    a: MatRef<'_, T>,
    q_mean: &[f64],
    q_l: MatRef<'_, f64>,
    y: &[f64],
    k_diag: &[T],
    noise: f64,
    n: usize,
    m: usize,
) -> f64 {
    let noise_s = T::from_f64(noise);
    let mut tr_s = T::from_f64(0.0);
    let mut log_det_s = T::from_f64(0.0);
    for j in 0..m {
        log_det_s += KernelScalar::ln(T::from_f64(q_l[(j, j)]));
        for i in j..m {
            let v = T::from_f64(q_l[(i, j)]);
            tr_s += v * v;
        }
    }
    log_det_s *= T::from_f64(2.0);
    let mut mean_norm2 = T::from_f64(0.0);
    for &v in q_mean {
        let v_s = T::from_f64(v);
        mean_norm2 += v_s * v_s;
    }
    let kl = T::from_f64(0.5) * (tr_s + mean_norm2 - T::from_f64(m as f64) - log_det_s);
    let log_2pi_noise = T::from_f64((2.0 * std::f64::consts::PI * noise).ln());
    let inv_noise = T::from_f64(1.0) / noise_s;
    let mut expected_ll = T::from_f64(0.0);
    let half = T::from_f64(0.5);
    for col in 0..n {
        let mut mu = T::from_f64(0.0);
        let mut a_norm = T::from_f64(0.0);
        let mut lt_norm = T::from_f64(0.0);
        for j in 0..m {
            let a_j = a[(j, col)];
            mu += a_j * T::from_f64(q_mean[j]);
            a_norm += a_j * a_j;
            let mut lt_j = T::from_f64(0.0);
            for i in j..m {
                lt_j += T::from_f64(q_l[(i, j)]) * a[(i, col)];
            }
            lt_norm += lt_j * lt_j;
        }
        let var = k_diag[col] - a_norm + lt_norm;
        let resid = T::from_f64(y[col]) - mu;
        expected_ll =
            expected_ll + -half * log_2pi_noise + -half * inv_noise * (resid * resid + var);
    }
    (-(expected_ll - kl)).to_f64()
}

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

pub(crate) fn svgp_value_and_gradient<M: crate::math::KernelMath, P>(
    model: &FittedSvgp<P>,
    out: &mut [f64],
    batch: &[usize],
) -> Result<f64, GprError>
where
    P: ModelPrecision,
{
    if !<P::Storage as ScalarOps>::ROUNDS_FROM_F64 {
        let shadow = promote_svgp_f64::<_>(model)?;
        return svgp_value_and_gradient_f64::<M>(&shadow, out, batch);
    }
    svgp_value_and_gradient_storage::<M, _>(model, out, batch)
}

fn svgp_value_and_gradient_storage<M: crate::math::KernelMath, P>(
    model: &FittedSvgp<P>,
    out: &mut [f64],
    batch: &[usize],
) -> Result<f64, GprError>
where
    P: ModelPrecision,
{
    let n = model.core.n;
    let m = model.core.m;
    let n_kernel = model.core.kernel.num_params();
    let n_theta = n_kernel + model.core.likelihood.num_params();
    crate::data::require_count(out.len(), n_theta + q_param_len(m), "parameters")?;
    if batch.is_empty() {
        return Err(GprError::EmptyInput);
    }
    out.fill(0.0);
    let noise = model.core.likelihood.noise_variance();
    let inv_noise = 1.0 / noise;
    let scale = n as f64 / batch.len() as f64;
    let kl = storage_kl_grad::<P>(model, out, n_theta, m);
    let cache = storage_point_cache::<P>(model, batch, m);
    let (ell, resid2_var) =
        storage_data_q_grad::<P>(model, out, batch, &cache, n_theta, inv_noise, scale);
    out[n_kernel] = -scale * (-0.5 * batch.len() as f64 + 0.5 * inv_noise * resid2_var);
    storage_kernel_grad::<M, P>(model, out, batch, &cache, n_kernel, inv_noise, scale)?;
    Ok(kl - scale * ell)
}

fn storage_kl_grad<P: ModelPrecision>(
    model: &FittedSvgp<P>,
    out: &mut [f64],
    n_theta: usize,
    m: usize,
) -> f64 {
    let mut tr_s = P::Storage::from_f64(0.0);
    let mut log_det_s = P::Storage::from_f64(0.0);
    let mut packed = 0;
    for j in 0..m {
        let diag = P::Storage::from_f64(model.q_l[(j, j)]);
        log_det_s += KernelScalar::ln(diag);
        for i in j..m {
            let v = P::Storage::from_f64(model.q_l[(i, j)]);
            tr_s += v * v;
            let grad = if i == j {
                v - P::Storage::from_f64(1.0) / v
            } else {
                v
            };
            out[n_theta + m + packed] = grad.to_f64();
            packed += 1;
        }
    }
    log_det_s *= P::Storage::from_f64(2.0);
    let mut mean_norm2 = P::Storage::from_f64(0.0);
    for k in 0..m {
        let v = P::Storage::from_f64(model.q_mean[k]);
        mean_norm2 += v * v;
        out[n_theta + k] = v.to_f64();
    }
    (P::Storage::from_f64(0.5) * (tr_s + mean_norm2 - P::Storage::from_f64(m as f64) - log_det_s))
        .to_f64()
}

struct StoragePointCache<T: KernelScalar> {
    resid: Vec<T>,
    var: Vec<T>,
    u: Mat<T>,
}

fn storage_point_cache<P: ModelPrecision>(
    model: &FittedSvgp<P>,
    batch: &[usize],
    m: usize,
) -> StoragePointCache<P::Storage> {
    let b = batch.len();
    let a = storage_batch_columns(model.a.as_ref(), batch);
    let mut q_l = Mat::<P::Storage>::zeros(m, m);
    for j in 0..m {
        for i in 0..m {
            q_l[(i, j)] = P::Storage::from_f64(model.q_l[(i, j)]);
        }
    }
    let mut u = Mat::<P::Storage>::zeros(m, b);
    crate::linalg::mat_mul_into(&mut u, q_l.transpose(), a.as_ref());
    let mut resid = vec![P::Storage::from_f64(0.0); b];
    let mut var = vec![P::Storage::from_f64(0.0); b];
    let a_ref = a.as_ref();
    for (b_idx, &col) in batch.iter().enumerate() {
        let mut mu = P::Storage::from_f64(0.0);
        let mut a_norm = P::Storage::from_f64(0.0);
        let mut lt_norm = P::Storage::from_f64(0.0);
        for j in 0..m {
            let a_j = a_ref[(j, b_idx)];
            let u_j = u[(j, b_idx)];
            mu += a_j * P::Storage::from_f64(model.q_mean[j]);
            a_norm += a_j * a_j;
            lt_norm += u_j * u_j;
        }
        var[b_idx] = model.k_diag[col] - a_norm + lt_norm;
        resid[b_idx] = P::Storage::from_f64(model.core.y[col]) - mu;
    }
    StoragePointCache { resid, var, u }
}

fn storage_batch_columns<T: KernelScalar>(full: MatRef<'_, T>, batch: &[usize]) -> Mat<T> {
    let m = full.nrows();
    let mut owned = Mat::zeros(m, batch.len());
    for (b_idx, &col) in batch.iter().enumerate() {
        for row in 0..m {
            owned[(row, b_idx)] = full[(row, col)];
        }
    }
    owned
}

fn storage_data_q_grad<P: ModelPrecision>(
    model: &FittedSvgp<P>,
    out: &mut [f64],
    batch: &[usize],
    cache: &StoragePointCache<P::Storage>,
    n_theta: usize,
    inv_noise: f64,
    scale: f64,
) -> (f64, f64) {
    let m = model.core.m;
    let noise = model.core.likelihood.noise_variance();
    let log_2pi_noise = P::Storage::from_f64((2.0 * std::f64::consts::PI * noise).ln());
    let inv = P::Storage::from_f64(inv_noise);
    let half = P::Storage::from_f64(0.5);
    let mut ell = P::Storage::from_f64(0.0);
    let mut resid2_var = P::Storage::from_f64(0.0);
    let b = batch.len();
    let mut resid_col = Mat::<P::Storage>::zeros(b, 1);
    for (b_idx, resid) in cache.resid.iter().enumerate() {
        let var = cache.var[b_idx];
        ell = ell + -half * log_2pi_noise + -half * inv * (*resid * *resid + var);
        resid2_var = resid2_var + *resid * *resid + var;
        resid_col[(b_idx, 0)] = *resid;
    }
    let a = storage_batch_columns(model.a.as_ref(), batch);
    let mut mean = Mat::<P::Storage>::zeros(m, 1);
    crate::linalg::mat_mul_into(&mut mean, a.as_ref(), resid_col.as_ref());
    let c = scale * inv_noise;
    for k in 0..m {
        out[n_theta + k] -= c * mean[(k, 0)].to_f64();
    }
    let mut gram = Mat::<P::Storage>::zeros(m, m);
    crate::linalg::mat_mul_into(&mut gram, a.as_ref(), cache.u.transpose());
    let mut packed = 0;
    for j in 0..m {
        for i in j..m {
            out[n_theta + m + packed] += c * gram[(i, j)].to_f64();
            packed += 1;
        }
    }
    (ell.to_f64(), resid2_var.to_f64())
}

fn storage_kernel_grad<M: crate::math::KernelMath, P>(
    model: &FittedSvgp<P>,
    out: &mut [f64],
    batch: &[usize],
    cache: &StoragePointCache<P::Storage>,
    n_kernel: usize,
    inv_noise: f64,
    scale: f64,
) -> Result<(), GprError>
where
    P: ModelPrecision,
{
    let m = model.core.m;
    let compiled = model.core.kernel.compile_as::<P::Storage>();
    let x64 = pack_points(&model.core.x_obs, model.core.n, model.core.d);
    let z64 = pack_points(&model.core.z_obs, model.core.m, model.core.d);
    let mut x_cast = P::Storage::empty_cols();
    let mut z_cast = P::Storage::empty_cols();
    let x_mat = P::Storage::storage_cols(x64.as_ref(), &mut x_cast);
    let z_mat = P::Storage::storage_cols(z64.as_ref(), &mut z_cast);
    let same_xz = model.core.x_obs == model.core.z_obs;
    let mut q_l = Mat::<P::Storage>::zeros(m, m);
    for j in 0..m {
        for i in 0..m {
            q_l[(i, j)] = P::Storage::from_f64(model.q_l[(i, j)]);
        }
    }
    for (param_idx, slot) in out.iter_mut().take(n_kernel).enumerate() {
        let (d_a, d_kdiag) =
            storage_kernel_tangents::<M, P>(&compiled, x_mat, z_mat, model, same_xz, param_idx)?;
        let da_b = storage_batch_columns(d_a.as_ref(), batch);
        let mut lt = Mat::<P::Storage>::zeros(m, batch.len());
        crate::linalg::mat_mul_into(&mut lt, q_l.transpose(), da_b.as_ref());
        let mut g = P::Storage::from_f64(0.0);
        let inv = P::Storage::from_f64(inv_noise);
        let two = P::Storage::from_f64(2.0);
        let half = P::Storage::from_f64(0.5);
        for (b_idx, &col) in batch.iter().enumerate() {
            let mut dmu = P::Storage::from_f64(0.0);
            let mut d_anorm = P::Storage::from_f64(0.0);
            let mut d_lt = P::Storage::from_f64(0.0);
            for r in 0..m {
                let da_r = da_b[(r, b_idx)];
                dmu += da_r * P::Storage::from_f64(model.q_mean[r]);
                d_anorm += two * model.a[(r, col)] * da_r;
                d_lt += two * cache.u[(r, b_idx)] * lt[(r, b_idx)];
            }
            let dvar = d_kdiag[col] - d_anorm + d_lt;
            let resid = cache.resid[b_idx];
            g = g + inv * resid * dmu - half * inv * dvar;
        }
        *slot = -scale * g.to_f64();
    }
    Ok(())
}

#[allow(clippy::type_complexity)]
fn storage_kernel_tangents<M: crate::math::KernelMath, P>(
    compiled: &CompiledKernel<P::Storage>,
    x: MatRef<'_, P::Storage>,
    z: MatRef<'_, P::Storage>,
    model: &FittedSvgp<P>,
    same_xz: bool,
    param_idx: usize,
) -> Result<(Mat<P::Storage>, Vec<P::Storage>), GprError>
where
    P: ModelPrecision,
{
    let m = model.core.m;
    let n = model.core.n;
    let mut d_kmm = Mat::<P::Storage>::zeros(m, m);
    let mut scratch_mm = Mat::<P::Storage>::zeros(m, m);
    compiled.grad_gram::<M>(
        GramInputs::points(z),
        d_kmm.as_mut(),
        param_idx,
        Triangle::Full,
        scratch_mm.as_mut(),
        &mut Vec::new(),
    )?;
    let mut d_kmn = if same_xz {
        let mut gram = Mat::<P::Storage>::zeros(n, n);
        let mut scratch = Mat::<P::Storage>::zeros(n, n);
        compiled.grad_gram::<M>(
            GramInputs::points(x),
            gram.as_mut(),
            param_idx,
            Triangle::Full,
            scratch.as_mut(),
            &mut Vec::new(),
        )?;
        gram
    } else {
        let mut cross = Mat::<P::Storage>::zeros(m, n);
        let mut scratch = Mat::<P::Storage>::zeros(m, n);
        compiled.grad_cross_points::<M>(z, x, cross.as_mut(), param_idx, scratch.as_mut())?;
        cross
    };
    let mut d_kdiag = vec![P::Storage::from_f64(0.0); n];
    if same_xz {
        for i in 0..n {
            d_kdiag[i] = d_kmn[(i, i)];
        }
    } else {
        compiled.grad_diag_points::<M>(x, &mut d_kdiag, param_idx)?;
    }
    let mut d_l = Mat::<P::Storage>::zeros(m, m);
    storage_cholesky_sensitivity(model.k_mm_l.as_ref(), d_kmm.as_ref(), d_l.as_mut(), m);
    crate::linalg::mat_sub_mul(&mut d_kmn, d_l.as_ref(), model.a.as_ref());
    solve_lower(model.k_mm_l.as_ref(), d_kmn.as_mut());
    Ok((d_kmn, d_kdiag))
}

fn storage_cholesky_sensitivity<T: KernelScalar>(
    l: MatRef<'_, T>,
    d_k: MatRef<'_, T>,
    mut d_l: MatMut<'_, T>,
    m: usize,
) {
    let two = T::from_f64(2.0);
    for j in 0..m {
        let mut acc = d_k[(j, j)];
        for k in 0..j {
            acc -= two * l[(j, k)] * d_l[(j, k)];
        }
        d_l[(j, j)] = acc / (two * l[(j, j)]);
        for i in j + 1..m {
            let mut acc = d_k[(i, j)];
            for k in 0..j {
                acc = acc - d_l[(i, k)] * l[(j, k)] - l[(i, k)] * d_l[(j, k)];
            }
            acc -= l[(i, j)] * d_l[(j, j)];
            d_l[(i, j)] = acc / l[(j, j)];
        }
    }
}

fn promote_svgp_f64<P: ModelPrecision>(
    model: &FittedSvgp<P>,
) -> Result<FittedSvgp<crate::precision::DoublePrecision>, GprError>
where
{
    let mut k_mm_l = Mat::<f64>::zeros(model.core.m, model.core.m);
    let mut a = Mat::<f64>::zeros(model.core.m, model.core.n);
    for j in 0..model.core.m {
        for i in j..model.core.m {
            k_mm_l[(i, j)] = model.k_mm_l[(i, j)].to_f64();
        }
    }
    for col in 0..model.core.n {
        for row in 0..model.core.m {
            a[(row, col)] = model.a[(row, col)].to_f64();
        }
    }
    let k_diag: Vec<f64> = model.k_diag.iter().map(|value| value.to_f64()).collect();
    Ok(FittedSvgp {
        core: SparseCore {
            kernel: model.core.kernel.clone(),
            likelihood: model.core.likelihood,
            x_obs: model.core.x_obs.clone(),
            z_obs: model.core.z_obs.clone(),
            y: model.core.y.clone(),
            n: model.core.n,
            m: model.core.m,
            d: model.core.d,
            math: model.core.math,
        },
        k_mm_l,
        a,
        q_mean: model.q_mean.clone(),
        q_l: model.q_l.clone(),
        k_diag,
    })
}

fn svgp_value_and_gradient_f64<M: crate::math::KernelMath>(
    model: &FittedSvgp<crate::precision::DoublePrecision>,
    out: &mut [f64],
    batch: &[usize],
) -> Result<f64, GprError> {
    let n = model.core.n;
    let m = model.core.m;
    let n_kernel = model.core.kernel.num_params();
    let n_theta = n_kernel + model.core.likelihood.num_params();
    crate::data::require_count(out.len(), n_theta + q_param_len(m), "parameters")?;
    if batch.is_empty() {
        return Err(GprError::EmptyInput);
    }
    out.fill(0.0);
    let noise = model.core.likelihood.noise_variance();
    let inv_noise = 1.0 / noise;
    let scale = n as f64 / batch.len() as f64;
    let kl = accumulate_kl_grad(model, out, n_theta, m);
    let cache = point_cache(model, batch, m);
    let (ell, resid2_var) =
        accumulate_data_q_grad(model, out, batch, &cache, n_theta, inv_noise, scale);
    out[n_kernel] = -scale * (-0.5 * batch.len() as f64 + 0.5 * inv_noise * resid2_var);
    accumulate_kernel_grad::<M>(model, out, batch, &cache, n_kernel, inv_noise, scale)?;
    Ok(kl - scale * ell)
}

fn accumulate_kl_grad(
    model: &FittedSvgp<crate::precision::DoublePrecision>,
    out: &mut [f64],
    n_theta: usize,
    m: usize,
) -> f64 {
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

struct PointCache {
    resid: Vec<f64>,
    var: Vec<f64>,
    /// Columns are batch rows. `u[(j, b)] = (Lᵀ a)[j]` for that row.
    u: Mat<f64>,
}

fn point_cache(
    model: &FittedSvgp<crate::precision::DoublePrecision>,
    batch: &[usize],
    m: usize,
) -> PointCache {
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
            let mu = dot_f64x4(a_col, mean);
            var[b_idx] = model.k_diag[col] - norm2_f64x4(a_col) + norm2_f64x4(u_col);
            resid[b_idx] = model.core.y[col] - mu;
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
            resid[b_idx] = model.core.y[col] - mu;
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
    model: &FittedSvgp<crate::precision::DoublePrecision>,
    out: &mut [f64],
    batch: &[usize],
    cache: &PointCache,
    n_theta: usize,
    inv_noise: f64,
    scale: f64,
) -> (f64, f64) {
    let m = model.core.m;
    let noise = model.core.likelihood.noise_variance();
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

fn accumulate_kernel_grad<M: crate::math::KernelMath>(
    model: &FittedSvgp<crate::precision::DoublePrecision>,
    out: &mut [f64],
    batch: &[usize],
    cache: &PointCache,
    n_kernel: usize,
    inv_noise: f64,
    scale: f64,
) -> Result<(), GprError> {
    let m = model.core.m;
    let compiled = model.core.kernel.compile();
    let x_mat = pack_points(&model.core.x_obs, model.core.n, model.core.d);
    let z_mat = pack_points(&model.core.z_obs, model.core.m, model.core.d);
    let same_xz = model.core.x_obs == model.core.z_obs;
    let mut ard_cross = match &compiled {
        crate::kernel::CompiledKernel::RbfArd(leaf) if !same_xz => {
            Some(leaf.grad_cross_all_from_coords::<M, _>(z_mat.as_ref(), x_mat.as_ref())?)
        }
        _ => None,
    };
    for (param_idx, slot) in out.iter_mut().take(n_kernel).enumerate() {
        let pre = ard_cross
            .as_mut()
            .map(|mats| std::mem::replace(&mut mats[param_idx], Mat::zeros(0, 0)));
        let (d_a, d_kdiag) = kernel_theta_tangents::<M>(
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
                let dmu = dot_f64x4(da_col, mean);
                let d_anorm = 2.0 * dot_f64x4(a_cm.col(col), da_col);
                let d_lt = 2.0 * dot_f64x4(u_cm.col(b_idx), lt_cm.col(b_idx));
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

fn kernel_theta_tangents<M: crate::math::KernelMath>(
    compiled: &crate::kernel::CompiledKernel,
    x: MatRef<'_, f64>,
    z: MatRef<'_, f64>,
    model: &FittedSvgp<crate::precision::DoublePrecision>,
    same_xz: bool,
    param_idx: usize,
    pre_cross: Option<Mat<f64>>,
) -> Result<(Mat<f64>, Vec<f64>), GprError> {
    let m = model.core.m;
    let n = model.core.n;
    let mut d_kmm = Mat::zeros(m, m);
    let mut scratch_mm = Mat::zeros(m, m);
    compiled.grad_gram::<M>(
        GramInputs::points(z),
        d_kmm.as_mut(),
        param_idx,
        Triangle::Full,
        scratch_mm.as_mut(),
        &mut Vec::new(),
    )?;
    let mut d_kmn = if let Some(pre) = pre_cross {
        pre
    } else if same_xz {
        let mut gram = Mat::zeros(n, n);
        let mut scratch = Mat::zeros(n, n);
        compiled.grad_gram::<M>(
            GramInputs::points(x),
            gram.as_mut(),
            param_idx,
            Triangle::Full,
            scratch.as_mut(),
            &mut Vec::new(),
        )?;
        gram
    } else {
        let mut cross = Mat::zeros(m, n);
        let mut scratch = Mat::zeros(m, n);
        compiled.grad_cross_points::<M>(z, x, cross.as_mut(), param_idx, scratch.as_mut())?;
        cross
    };
    let mut d_kdiag = vec![0.0; n];
    if same_xz {
        for i in 0..n {
            d_kdiag[i] = d_kmn[(i, i)];
        }
    } else {
        compiled.grad_diag_points::<M>(x, &mut d_kdiag, param_idx)?;
    }
    let mut d_l = Mat::zeros(m, m);
    cholesky_sensitivity(model.k_mm_l.as_ref(), d_kmm.as_ref(), d_l.as_mut(), m);
    // Upper of `d_l` stays zero, so this is the lower-triangular product.
    crate::linalg::mat_sub_mul(&mut d_kmn, d_l.as_ref(), model.a.as_ref());
    solve_lower(model.k_mm_l.as_ref(), d_kmn.as_mut());
    Ok((d_kmn, d_kdiag))
}

fn cholesky_sensitivity(
    l: MatRef<'_, f64>,
    d_k: MatRef<'_, f64>,
    mut d_l: MatMut<'_, f64>,
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

pub(crate) fn run_adam_fit<M: crate::math::KernelMath, P>(
    model: &mut FittedSvgp<P>,
    adam: &Adam,
) -> Result<(), GprError>
where
    P: crate::precision::GpScalar,
{
    let n = model.core.n;
    let m = model.core.m;
    let n_theta = model.core.kernel.num_params() + model.core.likelihood.num_params();
    let p = model.num_params();
    let mut intervals = vec![Interval::DEFAULT_POSITIVE; model.core.theta_len()];
    model.core.theta_intervals(&mut intervals)?;
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
            svgp_value_and_gradient::<M, _>(model, &mut g_user, batch)?;
            user_grad_to_unconstrained(&user, &z, &intervals, &g_user, &mut g_z, n_theta, m);
            adam.step(&mut z, &g_z, &mut moment1, &mut moment2, &mut timestep);
            start = end;
        }
    }
    user = unconstrained_to_user(&z, n_theta, m, &intervals)?;
    model.set_params(&user)
}

/// `K_mm`'s `f64` factor and `k(Z, x_*)` at one query point, the
/// [`crate::ReevaluateKernel`] reference of the mixed-precision mean.
fn f64_mean_reference<M: crate::math::KernelMath>(
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
