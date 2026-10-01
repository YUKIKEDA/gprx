//! Full-data and mini-batch gradients of the negative ELBO.
//!
//! One implementation, in `f64` whatever the storage scalar is (an `f32`
//! model only widens the `m × m` factor of `K_mm`). The terms of the batch
//! points are formed here: `A_b = L⁻¹ K(Z, X_b)` (`m × b`), `k_diag`, and
//! `∂K(Z, X_b)/∂θ`. A step costs `O(b·(m² + m·d) + m³)`; nothing in it scales
//! with `n`. A full-data gradient is the batch of every point.
//!
//! Every intermediate lives in [`GradBuffers`], which the Adam loop keeps
//! between steps: once they have grown to the largest batch, a step
//! allocates nothing here. The kernel routines themselves allocate nothing
//! for the isotropic leaves and their sums and products; the RBF-ARD
//! routines still build small per-call working vectors (`tests/alloc.rs`,
//! `svgp_adam_epoch_rbf_ard`), which #167 removes with the ARD SIMD paths.

use super::assemble::q_param_len;
use crate::error::GprError;
use crate::kernel::{CompiledKernel, GramInputs, KernelScalar, Triangle};
use crate::linalg::{dot_f64x4, gemm, norm2_f64x4, solve_lower};
use crate::precision::ModelPrecision;
use crate::sparse::{KernelScratch, SparseScratch, view};
use crate::svgp::FittedSvgp;
use faer::reborrow::IntoConst;
use faer::{Accum, Mat, MatMut, MatRef};

/// The intermediates of one gradient, grown to the largest batch and viewed
/// at each call's shape. Scratch: contents mean nothing between calls.
pub(crate) struct GradBuffers {
    k_mm_l: Mat<f64>,
    x: Mat<f64>,
    z: Mat<f64>,
    a: Mat<f64>,
    k_diag: Vec<f64>,
    u: Mat<f64>,
    resid: Vec<f64>,
    var: Vec<f64>,
    resid_col: Mat<f64>,
    mean: Mat<f64>,
    gram: Mat<f64>,
    d_kmm: Mat<f64>,
    d_kdiag: Vec<f64>,
    d_kzx: Mat<f64>,
    d_l: Mat<f64>,
    lt: Mat<f64>,
    ard: Vec<Mat<f64>>,
    /// `K(X, X)` or its derivative when `X` equals `Z` (`n = m`).
    full: Mat<f64>,
    ks: KernelScratch<f64>,
}

impl Default for GradBuffers {
    fn default() -> Self {
        Self {
            k_mm_l: Mat::new(),
            x: Mat::new(),
            z: Mat::new(),
            a: Mat::new(),
            k_diag: Vec::new(),
            u: Mat::new(),
            resid: Vec::new(),
            var: Vec::new(),
            resid_col: Mat::new(),
            mean: Mat::new(),
            gram: Mat::new(),
            d_kmm: Mat::new(),
            d_kdiag: Vec::new(),
            d_kzx: Mat::new(),
            d_l: Mat::new(),
            lt: Mat::new(),
            ard: Vec::new(),
            full: Mat::new(),
            ks: KernelScratch::new(),
        }
    }
}

/// The mini-batch negative ELBO `KL − (n / b) Σ_{i ∈ batch} ell_i` and its
/// gradient in `out` (kernel `θ`, likelihood `θ`, the whitened mean, then the
/// packed `L`). The batch is every point for the full-data value.
///
/// Reads `θ`, `K_mm`'s factor, and `q` of `model`; never its cached `A` or
/// `k_diag`, which a mini-batch fit leaves stale until it ends. Compiles the
/// kernel and sizes new buffers for this call; the Adam loop keeps both with
/// [`svgp_value_and_gradient_with`].
pub(crate) fn svgp_value_and_gradient<M: crate::math::KernelMath, P>(
    model: &FittedSvgp<P>,
    out: &mut [f64],
    batch: &[usize],
    scratch: &mut SparseScratch<P::Storage>,
) -> Result<f64, GprError>
where
    P: ModelPrecision,
{
    let compiled = model.core.kernel.compile();
    let mut bufs = GradBuffers {
        ks: std::mem::take(&mut scratch.f64),
        ..GradBuffers::default()
    };
    let result = svgp_value_and_gradient_with::<M, P>(model, out, batch, &compiled, &mut bufs);
    scratch.f64 = bufs.ks;
    result
}

/// [`svgp_value_and_gradient`] with the kernel compiled at the model's `θ`
/// (`compiled`) and the buffers of earlier calls.
pub(crate) fn svgp_value_and_gradient_with<M: crate::math::KernelMath, P>(
    model: &FittedSvgp<P>,
    out: &mut [f64],
    batch: &[usize],
    compiled: &CompiledKernel<f64>,
    bufs: &mut GradBuffers,
) -> Result<f64, GprError>
where
    P: ModelPrecision,
{
    let core = &model.core;
    let (n, m, d) = (core.n, core.m, core.d);
    let b = batch.len();
    let n_kernel = core.kernel.num_params();
    let n_theta = n_kernel + core.likelihood.num_params();
    crate::data::require_count(out.len(), n_theta + q_param_len(m), "parameters")?;
    if b == 0 {
        return Err(GprError::EmptyInput);
    }
    let GradBuffers {
        k_mm_l,
        x,
        z,
        a,
        k_diag,
        u,
        resid,
        var,
        resid_col,
        mean,
        gram,
        d_kmm,
        d_kdiag,
        d_kzx,
        d_l,
        lt,
        ard,
        full,
        ks,
    } = bufs;
    // The whole `X` when it equals `Z` (`n = m`): its Gram carries the White
    // diagonal that a rectangular `K(Z, X_b)` leaves at zero.
    let same_points = core.x_train == core.z_train;
    let mut k_mm_l = view(k_mm_l, m, m);
    for j in 0..m {
        for i in j..m {
            k_mm_l[(i, j)] = model.k_mm_l[(i, j)].to_f64();
        }
    }
    let k_mm_l = k_mm_l.into_const();
    let mut x = view(x, b, d);
    for (b_idx, &row) in batch.iter().enumerate() {
        for j in 0..d {
            x[(b_idx, j)] = core.x_train[j * n + row];
        }
    }
    let x = x.into_const();
    let mut z = view(z, m, d);
    for j in 0..d {
        for i in 0..m {
            z[(i, j)] = core.z_train[j * m + i];
        }
    }
    let z = z.into_const();
    let mut a = view(a, m, b);
    if same_points {
        let mut gram = view(full, n, n);
        ks.gram::<M>(
            compiled,
            GramInputs::points(z),
            gram.as_mut(),
            Triangle::Full,
        )?;
        copy_columns(gram.as_ref(), batch, a.as_mut());
    } else {
        ks.cross_into::<M>(compiled, z, x, a.as_mut())?;
    }
    solve_lower(k_mm_l, a.as_mut());
    let a = a.into_const();
    k_diag.resize(b, 0.0);
    compiled.fill_diag_points(x, k_diag)?;
    out.fill(0.0);
    let noise = core.likelihood.noise_variance();
    let inv_noise = 1.0 / noise;
    let scale = n as f64 / b as f64;
    let kl = accumulate_kl_grad(&model.q_mean, model.q_l.as_ref(), out, n_theta, m);
    let mut u = view(u, m, b);
    gemm(u.as_mut(), Accum::Replace, model.q_l.transpose(), a, 1.0);
    let u = u.into_const();
    resid.resize(b, 0.0);
    var.resize(b, 0.0);
    point_terms(
        a,
        u,
        k_diag,
        &core.y_train,
        batch,
        &model.q_mean,
        resid,
        var,
    );
    let (ell, resid2_var) = accumulate_data_q_grad(
        a,
        u,
        resid,
        var,
        DataBuffers {
            resid_col: view(resid_col, b, 1),
            mean: view(mean, m, 1),
            gram: view(gram, m, m),
        },
        out,
        n_theta,
        noise,
        inv_noise,
        scale,
    );
    out[n_kernel] = -scale * (-0.5 * b as f64 + 0.5 * inv_noise * resid2_var);
    let ard_ready = match compiled {
        CompiledKernel::RbfArd(leaf) if !same_points => {
            leaf.grad_cross_all_from_coords_into::<M>(z, x, ard)?;
            true
        }
        _ => false,
    };
    d_kdiag.resize(b, 0.0);
    for param_idx in 0..n_kernel {
        let mut d_kmm = view(d_kmm, m, m);
        ks.grad::<M>(
            compiled,
            GramInputs::points(z),
            d_kmm.as_mut(),
            param_idx,
            Triangle::Full,
        )?;
        let mut d_a = if same_points {
            let mut grad_full = view(full, n, n);
            ks.grad::<M>(
                compiled,
                GramInputs::points(z),
                grad_full.as_mut(),
                param_idx,
                Triangle::Full,
            )?;
            for (b_idx, &row) in batch.iter().enumerate() {
                d_kdiag[b_idx] = grad_full[(row, row)];
            }
            let mut cross = view(d_kzx, m, b);
            copy_columns(grad_full.as_ref(), batch, cross.as_mut());
            cross
        } else if ard_ready {
            compiled.grad_diag_points::<M>(x, d_kdiag, param_idx)?;
            ard[param_idx].as_mut().submatrix_mut(0, 0, m, b)
        } else {
            compiled.grad_diag_points::<M>(x, d_kdiag, param_idx)?;
            let mut cross = view(d_kzx, m, b);
            ks.grad_cross::<M>(compiled, z, x, cross.as_mut(), param_idx)?;
            cross
        };
        let mut d_l = view(d_l, m, m);
        d_l.fill(0.0);
        cholesky_sensitivity(k_mm_l, d_kmm.as_ref(), d_l.as_mut(), m);
        // Upper of `d_l` stays zero, so this is the lower-triangular product.
        gemm(d_a.as_mut(), Accum::Add, d_l.as_ref(), a, -1.0);
        solve_lower(k_mm_l, d_a.as_mut());
        let mut lt = view(lt, m, b);
        gemm(
            lt.as_mut(),
            Accum::Replace,
            model.q_l.transpose(),
            d_a.as_ref(),
            1.0,
        );
        out[param_idx] = -scale
            * kernel_param_term(
                d_a.as_ref(),
                a,
                u,
                lt.as_ref(),
                d_kdiag,
                resid,
                &model.q_mean,
                inv_noise,
            );
    }
    Ok(kl - scale * ell)
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

/// The columns `batch` of `full`, in that order, into `out`.
fn copy_columns(full: MatRef<'_, f64>, batch: &[usize], mut out: MatMut<'_, f64>) {
    for (b_idx, &col) in batch.iter().enumerate() {
        for row in 0..full.nrows() {
            out[(row, b_idx)] = full[(row, col)];
        }
    }
}

/// `resid = y − A_bᵀ m` and `var = k_diag − ‖a‖² + ‖Lᵀ a‖²` of each batch
/// point (columns of `a` and `u = Lᵀ A_b`).
#[allow(clippy::too_many_arguments)]
fn point_terms(
    a: MatRef<'_, f64>,
    u: MatRef<'_, f64>,
    k_diag: &[f64],
    y_train: &[f64],
    batch: &[usize],
    q_mean: &[f64],
    resid: &mut [f64],
    var: &mut [f64],
) {
    let m = a.nrows();
    if let (Some(a_cm), Some(u_cm)) = (ColMajor::new(a), ColMajor::new(u)) {
        for (b_idx, &row) in batch.iter().enumerate() {
            let a_col = a_cm.col(b_idx);
            let u_col = u_cm.col(b_idx);
            let mu = dot_f64x4(a_col, q_mean);
            var[b_idx] = k_diag[b_idx] - norm2_f64x4(a_col) + norm2_f64x4(u_col);
            resid[b_idx] = y_train[row] - mu;
        }
    } else {
        for (b_idx, &row) in batch.iter().enumerate() {
            let mut mu = 0.0;
            let mut a_norm = 0.0;
            let mut lt_norm = 0.0;
            for j in 0..m {
                let a_j = a[(j, b_idx)];
                let u_j = u[(j, b_idx)];
                mu += a_j * q_mean[j];
                a_norm += a_j * a_j;
                lt_norm += u_j * u_j;
            }
            var[b_idx] = k_diag[b_idx] - a_norm + lt_norm;
            resid[b_idx] = y_train[row] - mu;
        }
    }
}

/// Views [`accumulate_data_q_grad`] writes.
struct DataBuffers<'a> {
    resid_col: MatMut<'a, f64>,
    mean: MatMut<'a, f64>,
    gram: MatMut<'a, f64>,
}

#[allow(clippy::too_many_arguments)]
fn accumulate_data_q_grad(
    a: MatRef<'_, f64>,
    u: MatRef<'_, f64>,
    resid: &[f64],
    var: &[f64],
    bufs: DataBuffers<'_>,
    out: &mut [f64],
    n_theta: usize,
    noise: f64,
    inv_noise: f64,
    scale: f64,
) -> (f64, f64) {
    let DataBuffers {
        mut resid_col,
        mut mean,
        mut gram,
    } = bufs;
    let m = a.nrows();
    let log_2pi_noise = (2.0 * std::f64::consts::PI * noise).ln();
    let mut ell = 0.0;
    let mut resid2_var = 0.0;
    for (b_idx, (&r, &v)) in resid.iter().zip(var).enumerate() {
        ell += -0.5 * log_2pi_noise - 0.5 * inv_noise * (r * r + v);
        resid2_var += r * r + v;
        resid_col[(b_idx, 0)] = r;
    }
    gemm(mean.as_mut(), Accum::Replace, a, resid_col.as_ref(), 1.0);
    let c = scale * inv_noise;
    for k in 0..m {
        out[n_theta + k] -= c * mean[(k, 0)];
    }
    // `M_ij = Σ_b a_i u_j` for `i ≥ j`, the packed `L` derivative.
    gemm(gram.as_mut(), Accum::Replace, a, u.transpose(), 1.0);
    let mut packed = 0;
    for j in 0..m {
        for i in j..m {
            out[n_theta + m + packed] += c * gram[(i, j)];
            packed += 1;
        }
    }
    (ell, resid2_var)
}

/// `Σ_b inv_noise · resid · ∂μ − ½ inv_noise · ∂var` for one kernel `θ`.
#[allow(clippy::too_many_arguments)]
fn kernel_param_term(
    d_a: MatRef<'_, f64>,
    a: MatRef<'_, f64>,
    u: MatRef<'_, f64>,
    lt: MatRef<'_, f64>,
    d_kdiag: &[f64],
    resid: &[f64],
    q_mean: &[f64],
    inv_noise: f64,
) -> f64 {
    let m = a.nrows();
    let mut g = 0.0;
    if let (Some(da_cm), Some(a_cm), Some(u_cm), Some(lt_cm)) = (
        ColMajor::new(d_a),
        ColMajor::new(a),
        ColMajor::new(u),
        ColMajor::new(lt),
    ) {
        for (b_idx, &d_kdiag_b) in d_kdiag.iter().enumerate() {
            let da_col = da_cm.col(b_idx);
            let dmu = dot_f64x4(da_col, q_mean);
            let d_anorm = 2.0 * dot_f64x4(a_cm.col(b_idx), da_col);
            let d_lt = 2.0 * dot_f64x4(u_cm.col(b_idx), lt_cm.col(b_idx));
            let dvar = d_kdiag_b - d_anorm + d_lt;
            g += inv_noise * resid[b_idx] * dmu - 0.5 * inv_noise * dvar;
        }
    } else {
        for (b_idx, &d_kdiag_b) in d_kdiag.iter().enumerate() {
            let mut dmu = 0.0;
            let mut d_anorm = 0.0;
            let mut d_lt = 0.0;
            for r in 0..m {
                let da_r = d_a[(r, b_idx)];
                dmu += da_r * q_mean[r];
                d_anorm += 2.0 * a[(r, b_idx)] * da_r;
                d_lt += 2.0 * u[(r, b_idx)] * lt[(r, b_idx)];
            }
            let dvar = d_kdiag_b - d_anorm + d_lt;
            g += inv_noise * resid[b_idx] * dmu - 0.5 * inv_noise * dvar;
        }
    }
    g
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
