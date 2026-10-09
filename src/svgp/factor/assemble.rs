//! SVGP assembly (`K_mm`, `A`, the whitened `q`) and the negative ELBO.

use std::marker::PhantomData;

use crate::data::pack_points;
use crate::error::{CholeskyStage, GprError};
use crate::kernel::GramInputs;
use crate::kernel::{KernelScalar, KernelSpec, ModelKernel, Supply, Triangle};
use crate::linalg::{cholesky_lower_with_retries, llt_scratch, solve_lower};
use crate::policy::JitterPolicy;
use crate::precision::ModelPrecision;
use crate::sparse::{KernelScratch, SparseScratch};
use crate::sparse::{SparseCore, SparseData, SparseSets};
use crate::svgp::FittedSvgp;
use faer::{Mat, MatRef};

// `K_mm` only. Public default stays Fixed(0). Forrester m=16 / ℓ=1 is not PD in f64.
pub(crate) struct SvgpState<T: KernelScalar> {
    pub(crate) k_mm_l: Mat<T>,
    pub(crate) a: Mat<T>,
    pub(crate) q_mean: Vec<f64>,
    pub(crate) q_l: Mat<f64>,
    pub(crate) k_diag: Vec<T>,
}

pub(crate) fn assemble_fitted<M: crate::math::KernelMath, P, K: ModelKernel>(
    core: SparseCore<K::Supply>,
    q: Option<(Vec<f64>, Mat<f64>)>,
) -> Result<FittedSvgp<P, K>, GprError>
where
    P: ModelPrecision,
{
    let mut scratch = SparseScratch::<P::Storage, K::Supply>::default();
    let state = assemble_svgp::<M, P::Storage, K::Supply>(
        &core.kernel,
        core.jitter,
        core.data(),
        q,
        &mut scratch.storage,
    )?;
    Ok(FittedSvgp {
        core,
        scratch,
        k_mm_l: state.k_mm_l,
        a: state.a,
        q_mean: state.q_mean,
        q_l: state.q_l,
        k_diag: state.k_diag,
        _kernel: PhantomData,
    })
}

pub(crate) fn assemble_svgp<M: crate::math::KernelMath, T, U: Supply>(
    kernel: &KernelSpec<U>,
    k_mm_jitter: JitterPolicy,
    data: SparseData<'_>,
    q: Option<(Vec<f64>, Mat<f64>)>,
    ks: &mut KernelScratch<T>,
) -> Result<SvgpState<T>, GprError>
where
    T: KernelScalar,
{
    data.validate()?;
    let k_mm = assemble_kmm::<M, T, U>(kernel, k_mm_jitter, data, ks)?;
    let (a, k_diag) = assemble_data_terms::<M, T, U>(kernel, data, k_mm.as_ref(), ks)?;
    let (q_mean, q_l) = match q {
        Some((mean, l)) => (mean, l),
        None => prior_q(data.m),
    };
    Ok(SvgpState::<T> {
        k_mm_l: k_mm,
        a,
        q_mean,
        q_l,
        k_diag,
    })
}

/// The lower factor `L_mm` of `K_mm = k(Z, Z)`, retried by `k_mm_jitter`. It
/// reads the inducing points of `data` alone.
pub(crate) fn assemble_kmm<M: crate::math::KernelMath, T, U: Supply>(
    kernel: &KernelSpec<U>,
    k_mm_jitter: JitterPolicy,
    data: SparseData<'_>,
    ks: &mut KernelScratch<T>,
) -> Result<Mat<T>, GprError>
where
    T: KernelScalar,
{
    data.validate_inducing()?;
    let n_inducing = data.m;
    let compiled = kernel.compile_as::<T>();
    let z64 = pack_points(data.z, n_inducing, data.d);
    let mut z_cast = T::empty_cols();
    let z_mat = T::storage_cols(z64.as_ref(), &mut z_cast);
    let at = data.supply.at::<T>()?;
    let mut k_mm = Mat::zeros(n_inducing, n_inducing);
    ks.gram::<M, U>(
        &compiled,
        GramInputs::supplied(z_mat, U::squares(&at.zz)),
        k_mm.as_mut(),
        Triangle::Lower,
    )?;
    let mut chol_scratch = llt_scratch::<T>(n_inducing);
    cholesky_lower_with_retries(
        &mut k_mm,
        &mut chol_scratch,
        k_mm_jitter.retry_jitters(),
        CholeskyStage::Fit,
    )?;
    Ok(k_mm)
}

/// `A = L_mm⁻¹ K(Z, X)` (`m × n`) and `k(x_i, x_i)` for every training
/// point: the part of the assembly that costs `O(n)`.
pub(crate) fn assemble_data_terms<M: crate::math::KernelMath, T, U: Supply>(
    kernel: &KernelSpec<U>,
    data: SparseData<'_>,
    k_mm_l: MatRef<'_, T>,
    ks: &mut KernelScratch<T>,
) -> Result<(Mat<T>, Vec<T>), GprError>
where
    T: KernelScalar,
{
    let compiled = kernel.compile_as::<T>();
    let x64 = pack_points(data.x, data.n, data.d);
    let z64 = pack_points(data.z, data.m, data.d);
    let mut x_cast = T::empty_cols();
    let mut z_cast = T::empty_cols();
    let x_mat = T::storage_cols(x64.as_ref(), &mut x_cast);
    let z_mat = T::storage_cols(z64.as_ref(), &mut z_cast);
    let sets = SparseSets::<T, U>::new(x_mat, z_mat, data.supply.at::<T>()?);
    // Rectangular whatever the values of `Z` and `X` (see the Sgpr VFE).
    let mut a = ks.cross::<M, U>(&compiled, sets.k_mn())?;
    solve_lower(k_mm_l, a.as_mut());
    let mut k_diag = vec![T::from_f64(0.0); data.n];
    compiled.fill_diag_rows(x_mat, &mut k_diag)?;
    Ok((a, k_diag))
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
    let mut mean = vec![0.0; m];
    let mut l = Mat::zeros(m, m);
    unpack_q_into(params, m, &mut mean, &mut l)?;
    Ok((mean, l))
}

/// [`unpack_q`] into `mean` (length `m`) and the lower triangle of `l`
/// (`m × m`, its upper triangle zeroed). On error their contents are
/// partly written and mean nothing.
pub(crate) fn unpack_q_into(
    params: &[f64],
    m: usize,
    mean: &mut Vec<f64>,
    l: &mut Mat<f64>,
) -> Result<(), GprError> {
    crate::data::require_count(params.len(), q_param_len(m), "parameters")?;
    mean.resize(m, 0.0);
    for (slot, &v) in mean.iter_mut().zip(params) {
        if !v.is_finite() {
            return Err(GprError::NonFiniteInput);
        }
        *slot = v;
    }
    if l.nrows() != m || l.ncols() != m {
        *l = Mat::zeros(m, m);
    }
    let mut k = 0;
    for j in 0..m {
        for i in 0..j {
            l[(i, j)] = 0.0;
        }
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
    Ok(())
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
