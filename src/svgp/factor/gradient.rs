//! Full-data and mini-batch gradients of the negative ELBO.
//!
//! One implementation, in `f64` whatever the storage scalar is (an `f32`
//! model only widens the `m × m` factor of `K_mm`). The terms of the batch
//! points are formed here: `A_b = L⁻¹ K(Z, X_b)` (`m × b`) and `k_diag`.
//! The kernel-parameter term is three contractions, each one walk of the
//! kernel tree. A step costs `O(b·(m² + m·d) + m³)`; nothing in it scales
//! with `n`. The `m·d` term is one matrix product per ARD contraction.
//! A full-data gradient is the batch of every point.
//!
//! Every intermediate lives in [`GradBuffers`], which the Adam loop keeps
//! between steps: once they have grown to the largest batch, a step
//! allocates nothing, the kernel routines included (`tests/alloc.rs`,
//! `svgp_adam_epoch_*`).

use super::assemble::q_param_len;
use crate::error::GprError;
use crate::kernel::{BlockStore, CompiledKernel, KernelScalar, ModelKernel, SupplyViews};
use crate::linalg::{dot_f64x4, gemm, norm2_f64x4, solve_lower, solve_lower_transpose};
use crate::precision::ModelPrecision;
use crate::sparse::{KernelScratch, SparseScratch, SparseSets, view};
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
    /// `G`, then the weight `L⁻ᵀ G` on `∂K(Z, X_b)` (`m × b`).
    w_mn: Mat<f64>,
    /// `G Aᵀ`, then the weight on `∂K_mm` (`m × m`).
    w_mm: Mat<f64>,
    /// `tril½(G Aᵀ)` through `L⁻ᵀ` (`m × m`).
    half: Mat<f64>,
    /// The supplied `d²` from the batch points to `Z` (`b × m`), gathered
    /// from the stored `n × m` blocks.
    xz: BlockStore<f64>,
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
            w_mn: Mat::new(),
            w_mm: Mat::new(),
            half: Mat::new(),
            xz: BlockStore::default(),
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
pub(crate) fn svgp_value_and_gradient<M: crate::math::KernelMath, P, K: ModelKernel>(
    model: &FittedSvgp<P, K>,
    out: &mut [f64],
    batch: &[usize],
    scratch: &mut SparseScratch<P::Storage, K::Supply>,
) -> Result<f64, GprError>
where
    P: ModelPrecision,
{
    let compiled = model.core.kernel.compile();
    let mut bufs = GradBuffers {
        ks: std::mem::take(&mut scratch.f64),
        ..GradBuffers::default()
    };
    let result = svgp_value_and_gradient_with::<M, P, K>(model, out, batch, &compiled, &mut bufs);
    scratch.f64 = bufs.ks;
    result
}

/// [`svgp_value_and_gradient`] with the kernel compiled at the model's `θ`
/// (`compiled`) and the buffers of earlier calls.
pub(crate) fn svgp_value_and_gradient_with<M: crate::math::KernelMath, P, K: ModelKernel>(
    model: &FittedSvgp<P, K>,
    out: &mut [f64],
    batch: &[usize],
    compiled: &CompiledKernel<f64, K::Supply>,
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
        w_mn,
        w_mm,
        half,
        xz,
        ks,
    } = bufs;
    let supply = core.supply.at::<f64>()?;
    // Every point in order reads the stored blocks; a batch reads its rows.
    let xz: &BlockStore<f64> = if b == n && batch.iter().enumerate().all(|(i, &row)| i == row) {
        &supply.xz
    } else {
        supply.xz.rows_into(batch, xz);
        xz
    };
    let xz = <K::Supply as SupplyViews>::rects(xz);
    let zz = <K::Supply as SupplyViews>::squares(&supply.zz);
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
    // `K(Z, X_b)` is the rectangular cross covariance even when `Z` equals
    // `X`: a White leaf adds nothing to it.
    let sets = SparseSets::<f64, K::Supply> { x, z, zz, xz };
    ks.cross_mn_into::<M, K::Supply>(compiled, sets, a.as_mut())?;
    solve_lower(k_mm_l, a.as_mut());
    let a = a.into_const();
    k_diag.resize(b, 0.0);
    compiled.fill_diag_rows(x, k_diag)?;
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
    let (w_mm, w_mn) = adjoint_weights(
        AdjointInputs {
            k_mm_l,
            a,
            u,
            q_l: model.q_l.as_ref(),
            q_mean: &model.q_mean,
            resid,
            inv_noise,
        },
        view(w_mm, m, m),
        view(w_mn, m, b),
        view(half, m, m),
    );
    let (w_mm, w_mn) = (w_mm.into_const(), w_mn.into_const());
    // `g = ⟨w_mm, ∂K_mm⟩ + ⟨w_mn, ∂K(Z, X_b)⟩ − Σ ∂k_ii / (2σ²)`, then `−scale · g`.
    ks.write_square_contraction::<M, K::Supply>(compiled, sets.k_mm(), w_mm, &mut out[..n_kernel])?;
    ks.add_cross_contraction_mn::<M, K::Supply>(compiled, sets, w_mn, 1.0, &mut out[..n_kernel])?;
    ks.add_diag_contraction::<M, K::Supply>(compiled, x, -0.5 * inv_noise, &mut out[..n_kernel])?;
    for slot in &mut out[..n_kernel] {
        *slot *= -scale;
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

/// What [`adjoint_weights`] reads.
struct AdjointInputs<'a> {
    k_mm_l: MatRef<'a, f64>,
    a: MatRef<'a, f64>,
    u: MatRef<'a, f64>,
    q_l: MatRef<'a, f64>,
    q_mean: &'a [f64],
    resid: &'a [f64],
    inv_noise: f64,
}

/// The batch term's derivative as weights on the kernel matrices: for any
/// kernel direction, `g = ⟨w_mm, ∂K_mm⟩ + ⟨w_mn, ∂K(Z, X_b)⟩ − Σ ∂k_diag / (2σ²)`.
///
/// `g` moves with `A_b` as `⟨G, dA⟩` for `G = (m residᵀ + A_b − L_q U) / σ²`
/// (`U = L_qᵀ A_b`), and `dA = L⁻¹ ∂K(Z, X_b) − Φ_L A_b` with
/// `Φ = L⁻¹ ∂K_mm L⁻ᵀ` and `Φ_L` its lower triangle with the diagonal
/// halved. So `w_mn = L⁻ᵀ G` and `w_mm = −sym(L⁻ᵀ tril½(G A_bᵀ) L⁻¹)`,
/// formed once per step in `O(m² b)`; each parameter is then `O(m² + m b)`.
fn adjoint_weights<'a>(
    inputs: AdjointInputs<'_>,
    mut w_mm: MatMut<'a, f64>,
    mut w_mn: MatMut<'a, f64>,
    mut half: MatMut<'_, f64>,
) -> (MatMut<'a, f64>, MatMut<'a, f64>) {
    let AdjointInputs {
        k_mm_l,
        a,
        u,
        q_l,
        q_mean,
        resid,
        inv_noise,
    } = inputs;
    let m = a.nrows();
    gemm(w_mn.as_mut(), Accum::Replace, q_l, u, -inv_noise);
    for (j, &r) in resid.iter().enumerate() {
        for i in 0..m {
            w_mn[(i, j)] += inv_noise * (q_mean[i] * r + a[(i, j)]);
        }
    }
    gemm(
        w_mm.as_mut(),
        Accum::Replace,
        w_mn.as_ref(),
        a.transpose(),
        1.0,
    );
    solve_lower_transpose(k_mm_l, w_mn.as_mut());
    for j in 0..m {
        for i in 0..m {
            half[(i, j)] = match i.cmp(&j) {
                std::cmp::Ordering::Greater => w_mm[(i, j)],
                std::cmp::Ordering::Equal => 0.5 * w_mm[(i, j)],
                std::cmp::Ordering::Less => 0.0,
            };
        }
    }
    // `L⁻ᵀ T`, transposed into `w_mm`, then `L⁻ᵀ (L⁻ᵀ T)ᵀ = (L⁻ᵀ T L⁻¹)ᵀ`.
    solve_lower_transpose(k_mm_l, half.as_mut());
    for j in 0..m {
        for i in 0..m {
            w_mm[(i, j)] = half[(j, i)];
        }
    }
    solve_lower_transpose(k_mm_l, w_mm.as_mut());
    for j in 0..m {
        for i in j..m {
            let sym = -0.5 * (w_mm[(i, j)] + w_mm[(j, i)]);
            w_mm[(i, j)] = sym;
            w_mm[(j, i)] = sym;
        }
    }
    (w_mm, w_mn)
}
