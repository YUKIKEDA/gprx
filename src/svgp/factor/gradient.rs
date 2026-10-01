//! Full-data and mini-batch gradients of the negative ELBO.
//!
//! One implementation, in `f64` whatever the storage scalar is (an `f32`
//! model only widens the `m × m` factor of `K_mm`). The terms of the batch
//! points are formed here: `A_b = L⁻¹ K(Z, X_b)` (`m × b`), `k_diag`, and
//! `∂K(Z, X_b)/∂θ`. A step costs `O(b·(m² + m·d) + m³)`; nothing in it scales
//! with `n`. A full-data gradient is the batch of every point.

use super::assemble::q_param_len;
use crate::data::pack_points;
use crate::error::GprError;
use crate::kernel::{GramInputs, KernelScalar, Triangle};
use crate::linalg::{dot_f64x4, mat_mul_into, mat_sub_mul, norm2_f64x4, solve_lower};
use crate::precision::ModelPrecision;
use crate::sparse::{KernelScratch, SparseCore, SparseScratch};
use crate::svgp::FittedSvgp;
use faer::{Mat, MatMut, MatRef};

/// The mini-batch negative ELBO `KL − (n / b) Σ_{i ∈ batch} ell_i` and its
/// gradient in `out` (kernel `θ`, likelihood `θ`, the whitened mean, then the
/// packed `L`). The batch is every point for the full-data value.
///
/// Reads `θ`, `K_mm`'s factor, and `q` of `model`; never its cached `A` or
/// `k_diag`, which a mini-batch fit leaves stale until it ends.
pub(crate) fn svgp_value_and_gradient<M: crate::math::KernelMath, P>(
    model: &FittedSvgp<P>,
    out: &mut [f64],
    batch: &[usize],
    scratch: &mut SparseScratch<P::Storage>,
) -> Result<f64, GprError>
where
    P: ModelPrecision,
{
    let core = &model.core;
    let (n, m) = (core.n, core.m);
    let n_kernel = core.kernel.num_params();
    let n_theta = n_kernel + core.likelihood.num_params();
    crate::data::require_count(out.len(), n_theta + q_param_len(m), "parameters")?;
    if batch.is_empty() {
        return Err(GprError::EmptyInput);
    }
    let mut k_mm_l = Mat::<f64>::zeros(m, m);
    for j in 0..m {
        for i in j..m {
            k_mm_l[(i, j)] = model.k_mm_l[(i, j)].to_f64();
        }
    }
    let ks = &mut scratch.f64;
    let terms = batch_terms::<M>(core, k_mm_l.as_ref(), batch, ks)?;
    out.fill(0.0);
    let noise = core.likelihood.noise_variance();
    let inv_noise = 1.0 / noise;
    let scale = n as f64 / batch.len() as f64;
    let kl = accumulate_kl_grad(&model.q_mean, model.q_l.as_ref(), out, n_theta, m);
    let cache = point_cache(
        &terms,
        &core.y_train,
        batch,
        &model.q_mean,
        model.q_l.as_ref(),
        m,
    );
    let (ell, resid2_var) =
        accumulate_data_q_grad(&terms, &cache, out, n_theta, noise, inv_noise, scale);
    out[n_kernel] = -scale * (-0.5 * batch.len() as f64 + 0.5 * inv_noise * resid2_var);
    let inputs = KernelGradInputs {
        core,
        terms: &terms,
        k_mm_l: k_mm_l.as_ref(),
        q_mean: &model.q_mean,
        q_l: model.q_l.as_ref(),
        cache: &cache,
        batch,
    };
    accumulate_kernel_grad::<M>(&inputs, out, n_kernel, inv_noise, scale, ks)?;
    Ok(kl - scale * ell)
}

/// What the batch points contribute, in `f64`.
struct BatchTerms {
    /// `X_b`, `b × d`.
    x: Mat<f64>,
    /// `Z`, `m × d`.
    z: Mat<f64>,
    /// `A_b = L⁻¹ K(Z, X_b)`, `m × b`.
    a: Mat<f64>,
    /// `k(x, x)` of each batch point.
    k_diag: Vec<f64>,
}

fn batch_terms<M: crate::math::KernelMath>(
    core: &SparseCore,
    k_mm_l: MatRef<'_, f64>,
    batch: &[usize],
    ks: &mut KernelScratch<f64>,
) -> Result<BatchTerms, GprError> {
    let (n, m, d) = (core.n, core.m, core.d);
    let b = batch.len();
    let compiled = core.kernel.compile();
    let mut x = Mat::<f64>::zeros(b, d);
    for (b_idx, &row) in batch.iter().enumerate() {
        for j in 0..d {
            x[(b_idx, j)] = core.x_train[j * n + row];
        }
    }
    let z = pack_points(&core.z_train, m, d);
    let mut a = ks.cross::<M>(&compiled, z.as_ref(), x.as_ref())?;
    solve_lower(k_mm_l, a.as_mut());
    let mut k_diag = vec![0.0; b];
    compiled.fill_diag_points(x.as_ref(), &mut k_diag)?;
    Ok(BatchTerms { x, z, a, k_diag })
}

fn accumulate_kl_grad(
    q_mean: &[f64],
    q_l: MatRef<'_, f64>,
    out: &mut [f64],
    n_theta: usize,
    m: usize,
) -> f64 {
    let mut tr_s = 0.0;
    let mut log_det_s = 0.0;
    let mut packed = 0;
    for j in 0..m {
        log_det_s += q_l[(j, j)].ln();
        for i in j..m {
            let v = q_l[(i, j)];
            tr_s += v * v;
            out[n_theta + m + packed] = if i == j { v - 1.0 / v } else { v };
            packed += 1;
        }
    }
    log_det_s *= 2.0;
    let mut mean_norm2 = 0.0;
    for k in 0..m {
        let v = q_mean[k];
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
    /// Columns are batch points. `u[(j, b)] = (Lᵀ a)[j]` for that point.
    u: Mat<f64>,
}

fn point_cache(
    terms: &BatchTerms,
    y_train: &[f64],
    batch: &[usize],
    q_mean: &[f64],
    q_l: MatRef<'_, f64>,
    m: usize,
) -> PointCache {
    let b = batch.len();
    let mut u = Mat::zeros(m, b);
    mat_mul_into(&mut u, q_l.transpose(), terms.a.as_ref());
    let mut resid = vec![0.0; b];
    let mut var = vec![0.0; b];
    if let (Some(a_cm), Some(u_cm)) = (ColMajor::new(terms.a.as_ref()), ColMajor::new(u.as_ref())) {
        for b_idx in 0..b {
            let a_col = a_cm.col(b_idx);
            let u_col = u_cm.col(b_idx);
            let mu = dot_f64x4(a_col, q_mean);
            var[b_idx] = terms.k_diag[b_idx] - norm2_f64x4(a_col) + norm2_f64x4(u_col);
            resid[b_idx] = y_train[batch[b_idx]] - mu;
        }
    } else {
        for b_idx in 0..b {
            let mut mu = 0.0;
            let mut a_norm = 0.0;
            let mut lt_norm = 0.0;
            for j in 0..m {
                let a_j = terms.a[(j, b_idx)];
                let u_j = u[(j, b_idx)];
                mu += a_j * q_mean[j];
                a_norm += a_j * a_j;
                lt_norm += u_j * u_j;
            }
            var[b_idx] = terms.k_diag[b_idx] - a_norm + lt_norm;
            resid[b_idx] = y_train[batch[b_idx]] - mu;
        }
    }
    PointCache { resid, var, u }
}

fn accumulate_data_q_grad(
    terms: &BatchTerms,
    cache: &PointCache,
    out: &mut [f64],
    n_theta: usize,
    noise: f64,
    inv_noise: f64,
    scale: f64,
) -> (f64, f64) {
    let m = terms.a.nrows();
    let log_2pi_noise = (2.0 * std::f64::consts::PI * noise).ln();
    let mut ell = 0.0;
    let mut resid2_var = 0.0;
    let b = terms.a.ncols();
    let mut resid_col = Mat::zeros(b, 1);
    for (b_idx, &resid) in cache.resid.iter().enumerate() {
        let var = cache.var[b_idx];
        ell += -0.5 * log_2pi_noise - 0.5 * inv_noise * (resid * resid + var);
        resid2_var += resid * resid + var;
        resid_col[(b_idx, 0)] = resid;
    }
    let mut mean = Mat::zeros(m, 1);
    mat_mul_into(&mut mean, terms.a.as_ref(), resid_col.as_ref());
    let c = scale * inv_noise;
    for k in 0..m {
        out[n_theta + k] -= c * mean[(k, 0)];
    }
    // `M_ij = Σ_b a_i u_j` for `i ≥ j`, the packed `L` derivative.
    let mut gram = Mat::zeros(m, m);
    mat_mul_into(&mut gram, terms.a.as_ref(), cache.u.transpose());
    let mut packed = 0;
    for j in 0..m {
        for i in j..m {
            out[n_theta + m + packed] += c * gram[(i, j)];
            packed += 1;
        }
    }
    (ell, resid2_var)
}

/// What the kernel-parameter gradient reads.
struct KernelGradInputs<'a> {
    core: &'a SparseCore,
    terms: &'a BatchTerms,
    k_mm_l: MatRef<'a, f64>,
    q_mean: &'a [f64],
    q_l: MatRef<'a, f64>,
    cache: &'a PointCache,
    batch: &'a [usize],
}

fn accumulate_kernel_grad<M: crate::math::KernelMath>(
    inputs: &KernelGradInputs<'_>,
    out: &mut [f64],
    n_kernel: usize,
    inv_noise: f64,
    scale: f64,
    ks: &mut KernelScratch<f64>,
) -> Result<(), GprError> {
    let KernelGradInputs {
        core,
        terms,
        q_mean,
        q_l,
        cache,
        batch,
        ..
    } = *inputs;
    let m = core.m;
    let b = batch.len();
    let compiled = core.kernel.compile();
    let mut ard_cross = match &compiled {
        crate::kernel::CompiledKernel::RbfArd(leaf) => {
            Some(leaf.grad_cross_all_from_coords::<M, _>(terms.z.as_ref(), terms.x.as_ref())?)
        }
        _ => None,
    };
    for (param_idx, slot) in out.iter_mut().take(n_kernel).enumerate() {
        let pre = ard_cross
            .as_mut()
            .map(|mats| std::mem::replace(&mut mats[param_idx], Mat::zeros(0, 0)));
        let (d_a, d_kdiag) = kernel_theta_tangents::<M>(&compiled, inputs, ks, param_idx, pre)?;
        let mut lt = Mat::zeros(m, b);
        mat_mul_into(&mut lt, q_l.transpose(), d_a.as_ref());
        let mut g = 0.0;
        if let (Some(da_cm), Some(a_cm), Some(u_cm), Some(lt_cm)) = (
            ColMajor::new(d_a.as_ref()),
            ColMajor::new(terms.a.as_ref()),
            ColMajor::new(cache.u.as_ref()),
            ColMajor::new(lt.as_ref()),
        ) {
            for (b_idx, &d_kdiag_b) in d_kdiag.iter().enumerate() {
                let da_col = da_cm.col(b_idx);
                let dmu = dot_f64x4(da_col, q_mean);
                let d_anorm = 2.0 * dot_f64x4(a_cm.col(b_idx), da_col);
                let d_lt = 2.0 * dot_f64x4(u_cm.col(b_idx), lt_cm.col(b_idx));
                let dvar = d_kdiag_b - d_anorm + d_lt;
                let resid = cache.resid[b_idx];
                g += inv_noise * resid * dmu - 0.5 * inv_noise * dvar;
            }
        } else {
            for b_idx in 0..b {
                let resid = cache.resid[b_idx];
                let mut dmu = 0.0;
                let mut d_anorm = 0.0;
                let mut d_lt = 0.0;
                for r in 0..m {
                    let da_r = d_a[(r, b_idx)];
                    dmu += da_r * q_mean[r];
                    d_anorm += 2.0 * terms.a[(r, b_idx)] * da_r;
                    d_lt += 2.0 * cache.u[(r, b_idx)] * lt[(r, b_idx)];
                }
                let dvar = d_kdiag[b_idx] - d_anorm + d_lt;
                g += inv_noise * resid * dmu - 0.5 * inv_noise * dvar;
            }
        }
        *slot = -scale * g;
    }
    Ok(())
}

/// `∂A_b/∂θ_{param_idx}` (`m × b`) and `∂k_diag/∂θ` of the batch points.
fn kernel_theta_tangents<M: crate::math::KernelMath>(
    compiled: &crate::kernel::CompiledKernel,
    inputs: &KernelGradInputs<'_>,
    ks: &mut KernelScratch<f64>,
    param_idx: usize,
    pre_cross: Option<Mat<f64>>,
) -> Result<(Mat<f64>, Vec<f64>), GprError> {
    let (core, terms, batch) = (inputs.core, inputs.terms, inputs.batch);
    let (m, b) = (core.m, batch.len());
    let mut d_kmm = Mat::zeros(m, m);
    ks.grad::<M>(
        compiled,
        GramInputs::points(terms.z.as_ref()),
        d_kmm.as_mut(),
        param_idx,
        Triangle::Full,
    )?;
    let mut d_kdiag = vec![0.0; b];
    let mut d_kzx = if let Some(pre) = pre_cross {
        compiled.grad_diag_points::<M>(terms.x.as_ref(), &mut d_kdiag, param_idx)?;
        pre
    } else {
        let mut cross = Mat::zeros(m, b);
        compiled.grad_cross_points::<M>(
            terms.z.as_ref(),
            terms.x.as_ref(),
            cross.as_mut(),
            param_idx,
            ks.scratch(m, b),
        )?;
        compiled.grad_diag_points::<M>(terms.x.as_ref(), &mut d_kdiag, param_idx)?;
        cross
    };
    let mut d_l = Mat::zeros(m, m);
    cholesky_sensitivity(inputs.k_mm_l, d_kmm.as_ref(), d_l.as_mut(), m);
    // Upper of `d_l` stays zero, so this is the lower-triangular product.
    mat_sub_mul(&mut d_kzx, d_l.as_ref(), terms.a.as_ref());
    solve_lower(inputs.k_mm_l, d_kzx.as_mut());
    Ok((d_kzx, d_kdiag))
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
