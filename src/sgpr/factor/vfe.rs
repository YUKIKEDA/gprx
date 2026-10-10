//! VFE assembly: `K_mm`, `A`, `B`, the weights, and the bound.

use super::lit;
use crate::data::pack_points;
use crate::error::{CholeskyStage, GprError};
use crate::kernel::{KernelScalar, KernelSpec, Supply, Triangle};
use crate::likelihood::GaussianLikelihood;
use crate::linalg::{
    cholesky_lower_with_retries, dot_ay, frobenius2, gram_aat_plus_noise, llt_scratch,
    matvec_columns, round_mat, solve_llt, solve_lower,
};
use crate::param::Interval;
use crate::policy::JitterPolicy;
use crate::precision::{F64Vfe, ModelPrecision};
use crate::sgpr::FittedSgpr;
use crate::sgpr::InducingLayout;
use crate::sparse::{KernelScratch, SparseCore, SparseData, SparseScratch, SparseSets};
use faer::{Mat, MatRef};
use std::marker::PhantomData;

/// Predict weights after a factor or an online update. See
/// [`ModelPrecision::publish_weights`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn publish_sgpr_weights<M: crate::math::KernelMath, P: ModelPrecision, U: Supply>(
    kernel: &KernelSpec<U>,
    k_mm_jitter: JitterPolicy,
    a: MatRef<'_, P::Storage>,
    b_l: MatRef<'_, P::Storage>,
    w: &[P::Storage],
    data: SparseData<'_, U>,
    noise: f64,
) -> Result<Vec<P::Refine>, GprError> {
    let reference = || {
        let state = assemble_vfe::<M, f64, U>(
            kernel,
            k_mm_jitter,
            noise_likelihood(noise)?,
            data.clone(),
            &mut KernelScratch::new(),
            &mut KernelScratch::new(),
        )?;
        Ok(F64Vfe {
            a: state.a,
            w: state.w,
        })
    };
    P::publish_weights(a, b_l, w, data.y, noise, &reference)
}

pub(super) fn noise_likelihood(noise: f64) -> Result<GaussianLikelihood, GprError> {
    if crate::param::Interval::DEFAULT_POSITIVE.contains(noise) {
        return GaussianLikelihood::new(noise);
    }
    let interval = crate::Interval::new(1.0e-12, 1.0e5)?;
    let mut likelihood = GaussianLikelihood::new(1.0)?.with_bounds(interval)?;
    likelihood.set_params(&[noise.ln()])?;
    Ok(likelihood)
}

pub(crate) struct VfeState<T: KernelScalar> {
    pub(crate) k_mm_l: Mat<T>,
    pub(crate) a: Mat<T>,
    pub(crate) b_l: Mat<T>,
    pub(crate) w: Vec<T>,
    pub(crate) k_diag_sum: T,
    pub(crate) a_frobenius2: T,
}

pub(crate) fn assemble_fitted<O, I: InducingLayout<K::Supply>, M: crate::math::KernelMath, P, K>(
    core: SparseCore<K::Supply>,
    optimizer: O,
) -> Result<FittedSgpr<O, I, P, K>, GprError>
where
    P: ModelPrecision,
    K: crate::kernel::ModelKernel,
{
    let mut scratch = SparseScratch::<P::Storage, K::Supply>::default();
    let data = core.data();
    let (state, w64) = assemble_vfe_with_f64_w::<M, P::Storage, K::Supply>(
        &core.kernel,
        core.jitter,
        core.likelihood,
        data.clone(),
        &mut scratch.storage,
        &mut scratch.f64,
    )?;
    // `data` borrows `core` until here, whichever branch reads it.
    let predict_w = if let (true, Some(w64)) = (P::REFINES_IN_F64, w64) {
        drop(data);
        w64.into_iter().map(P::Refine::from_f64).collect()
    } else if P::REFINES_IN_F64 {
        assemble_vfe::<M, f64, K::Supply>(
            &core.kernel,
            core.jitter,
            core.likelihood,
            data,
            &mut scratch.f64,
            &mut KernelScratch::new(),
        )?
        .w
        .into_iter()
        .map(P::Refine::from_f64)
        .collect()
    } else {
        publish_sgpr_weights::<M, P, K::Supply>(
            &core.kernel,
            core.jitter,
            state.a.as_ref(),
            state.b_l.as_ref(),
            &state.w,
            data,
            core.likelihood.noise_variance(),
        )?
    };
    Ok(FittedSgpr {
        core,
        scratch,
        optimizer,
        inducing: PhantomData,
        _kernel: PhantomData,
        k_mm_l: state.k_mm_l,
        a: state.a,
        b_l: state.b_l,
        w: state.w,
        predict_w,
        k_diag_sum: state.k_diag_sum,
        a_frobenius2: state.a_frobenius2,
    })
}

pub(crate) fn assemble_vfe<M: crate::math::KernelMath, T, U: Supply>(
    kernel: &KernelSpec<U>,
    k_mm_jitter: JitterPolicy,
    likelihood: GaussianLikelihood,
    data: SparseData<'_, U>,
    ks: &mut KernelScratch<T>,
    ks64: &mut KernelScratch<f64>,
) -> Result<VfeState<T>, GprError>
where
    T: KernelScalar,
{
    data.validate()?;
    if T::ROUNDS_FROM_F64 {
        let state = assemble_vfe::<M, f64, U>(
            kernel,
            k_mm_jitter,
            likelihood,
            data,
            ks64,
            &mut KernelScratch::new(),
        )?;
        return Ok(round_vfe(&state));
    }
    let (n_rows, n_inducing) = (data.n, data.m);
    let compiled = kernel.compile_as::<T>();
    let x64 = pack_points(data.x, n_rows, data.d);
    let z64 = pack_points(data.z, n_inducing, data.d);
    let mut x_cast = T::empty_cols();
    let mut z_cast = T::empty_cols();
    let mut y_cast = T::empty_rows();
    let x_mat = T::storage_cols(x64.as_ref(), &mut x_cast);
    let z_mat = T::storage_cols(z64.as_ref(), &mut z_cast);
    let y_s = T::storage_rows(data.y, &mut y_cast);
    let sets = SparseSets::<T, U>::new(
        x_mat,
        z_mat,
        U::try_map_held(data.supply.clone(), crate::sparse::SparseSupply::at::<T>)?,
    );
    // A rounding scalar returned above, so `T` is evaluated as stored below.
    let mut k_mm = Mat::zeros(n_inducing, n_inducing);
    ks.gram::<M, U>(&compiled, sets.k_mm(), k_mm.as_mut(), Triangle::Lower)?;
    let mut chol_scratch = llt_scratch::<T>(n_inducing);
    cholesky_lower_with_retries(
        &mut k_mm,
        &mut chol_scratch,
        k_mm_jitter.retry_jitters(),
        CholeskyStage::Fit,
    )?;
    // `K(Z, X)` is the rectangular cross covariance whatever the values of
    // `Z` and `X`: a White leaf adds nothing to it, so the objective does not
    // jump when a free `Z` leaves `X` (docs/design.md §5).
    let mut a = ks.cross_mn::<M, U>(&compiled, sets)?;
    solve_lower(k_mm.as_ref(), a.as_mut());
    let noise = likelihood.noise_variance();
    let mut b = gram_aat_plus_noise(a.as_ref(), noise);
    let mut b_scratch = llt_scratch::<T>(n_inducing);
    cholesky_lower_with_retries(
        &mut b,
        &mut b_scratch,
        JitterPolicy::default().retry_jitters(),
        CholeskyStage::Fit,
    )?;
    let mut k_diag = vec![lit::<T>(0.0); n_rows];
    compiled.fill_diag_rows(x_mat.as_ref(), &mut k_diag)?;
    let k_diag_sum = k_diag.iter().fold(lit::<T>(0.0), |acc, v| acc + *v);
    let a_frobenius2 = frobenius2(a.as_ref());
    let mut ay = Mat::zeros(n_inducing, 1);
    matvec_columns(a.as_ref(), y_s, ay.as_mut());
    solve_llt(b.as_ref(), ay.as_mut());
    let mut w = vec![lit::<T>(0.0); n_inducing];
    for i in 0..n_inducing {
        w[i] = ay[(i, 0)];
    }
    Ok(VfeState::<T> {
        k_mm_l: k_mm,
        a,
        b_l: b,
        w,
        k_diag_sum,
        a_frobenius2,
    })
}

/// `state` rounded to a storage scalar that is evaluated in `f64`.
fn round_vfe<T: KernelScalar>(state: &VfeState<f64>) -> VfeState<T> {
    VfeState {
        k_mm_l: round_mat::<T>(state.k_mm_l.as_ref()),
        a: round_mat::<T>(state.a.as_ref()),
        b_l: round_mat::<T>(state.b_l.as_ref()),
        w: state.w.iter().map(|value| T::from_f64(*value)).collect(),
        k_diag_sum: T::from_f64(state.k_diag_sum),
        a_frobenius2: T::from_f64(state.a_frobenius2),
    }
}

/// [`assemble_vfe`] in the storage scalar `T`, plus the `f64` weights `w`
/// when `T` is evaluated in `f64` and rounded (`ROUNDS_FROM_F64`).
///
/// A refining precision publishes those `f64` weights as its predict
/// weights. Taking them from the same `f64` assembly saves assembling the
/// whole system a second time. A scalar that is not rounded returns `None`.
pub(crate) fn assemble_vfe_with_f64_w<M: crate::math::KernelMath, T: KernelScalar, U: Supply>(
    kernel: &KernelSpec<U>,
    k_mm_jitter: JitterPolicy,
    likelihood: GaussianLikelihood,
    data: SparseData<'_, U>,
    ks: &mut KernelScratch<T>,
    ks64: &mut KernelScratch<f64>,
) -> Result<(VfeState<T>, Option<Vec<f64>>), GprError> {
    if T::ROUNDS_FROM_F64 {
        data.validate()?;
        let state = assemble_vfe::<M, f64, U>(
            kernel,
            k_mm_jitter,
            likelihood,
            data,
            ks64,
            &mut KernelScratch::new(),
        )?;
        let rounded = round_vfe(&state);
        return Ok((rounded, Some(state.w)));
    }
    let state = assemble_vfe::<M, T, U>(kernel, k_mm_jitter, likelihood, data, ks, ks64)?;
    Ok((state, None))
}

pub(crate) fn fill_z_intervals(
    x: &[f64],
    n: usize,
    d: usize,
    out: &mut [Interval],
) -> Result<(), GprError> {
    if d == 0 || !out.len().is_multiple_of(d) {
        return Err(GprError::LengthMismatch {
            reason: "inducing interval length is not a multiple of d".to_owned(),
        });
    }
    let m = out.len() / d;
    for dim in 0..d {
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        for i in 0..n {
            let v = x[i + dim * n];
            min_x = min_x.min(v);
            max_x = max_x.max(v);
        }
        let range = (max_x - min_x).max(0.0);
        let slack = (0.1 * range).max(0.1);
        let mut lo = min_x - slack;
        let hi = max_x + slack;
        if lo > 0.0 {
            lo = 0.0;
        }
        let interval = Interval::new(lo, hi)?;
        for p in 0..m {
            out[p + dim * m] = interval;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn vfe_neg_log_marginal_likelihood<T: KernelScalar>(
    a: MatRef<'_, T>,
    b_l: MatRef<'_, T>,
    w: &[T],
    y: &[f64],
    k_diag_sum: T,
    a_frobenius2: T,
    noise: f64,
    n: usize,
    m: usize,
) -> Result<f64, GprError> {
    let mut y_cast = T::empty_rows();
    let y_s = T::storage_rows(y, &mut y_cast);
    let noise_s = lit::<T>(noise);
    let mut log_det_b = lit::<T>(0.0);
    for i in 0..m {
        log_det_b += KernelScalar::ln(b_l[(i, i)]);
    }
    log_det_b *= lit::<T>(2.0);
    let n_minus_m = lit::<T>(n as f64) - lit::<T>(m as f64);
    let log_det = n_minus_m * KernelScalar::ln(noise_s) + log_det_b;
    let mut y_norm2 = lit::<T>(0.0);
    for value in y_s {
        y_norm2 += *value * *value;
    }
    let ay_dot_w = dot_ay(a, y_s, w);
    let quad = (y_norm2 - ay_dot_w) / noise_s;
    let trace = (k_diag_sum - a_frobenius2) / (lit::<T>(2.0) * noise_s);
    let log_two_pi = lit::<T>((2.0 * std::f64::consts::PI).ln());
    let value = lit::<T>(0.5) * (lit::<T>(n as f64) * log_two_pi + log_det + quad) + trace;
    Ok(value.to_f64())
}

pub(crate) fn refresh_w<T: KernelScalar>(a: MatRef<'_, T>, b_l: MatRef<'_, T>, y: &[T]) -> Vec<T> {
    let m = a.nrows();
    let mut ay = Mat::zeros(m, 1);
    matvec_columns(a, y, ay.as_mut());
    solve_llt(b_l, ay.as_mut());
    let mut w = vec![lit::<T>(0.0); m];
    for i in 0..m {
        w[i] = ay[(i, 0)];
    }
    w
}
