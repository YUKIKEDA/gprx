//! VFE assembly, derivatives, rank-1 `X` updates, and inducing `m` updates.

use std::marker::PhantomData;

use dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt;
use faer::{Mat, MatMut, MatRef};

use crate::error::{CholeskyStage, GprError};
use crate::gpr::JitterPolicy;
use crate::gpr::factor::{
    cholesky_lower_with_policy, pack_points, pack_points_into, symmetrize_lower, validate_query,
    validate_training,
};
use crate::kernel::{
    CompiledKernel, CoordMode, KernelSpec, Triangle, fill_squared_euclidean_cross,
};
use crate::likelihood::GaussianLikelihood;
use crate::param::Interval;
use crate::workspace::{faer_par, faer_par_dims};
use crate::{PredictOptions, Prediction, VarianceKind};

use super::InducingLayout;
use super::fitted::FittedSgpr;

// `K_mm` only. Public default stays Fixed(0). Forrester m=16 / ℓ=1 is not PD in f64.
fn k_mm_jitter_policy() -> JitterPolicy {
    JitterPolicy::adaptive(1e-8, 10.0, 5, 1e-3).unwrap_or_default()
}

pub(crate) struct VfeState {
    pub(crate) k_mm_l: Mat<f64>,
    pub(crate) a: Mat<f64>,
    pub(crate) b_l: Mat<f64>,
    pub(crate) w: Vec<f64>,
    pub(crate) k_diag_sum: f64,
    pub(crate) a_frobenius2: f64,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn assemble_fitted<O, I: InducingLayout>(
    kernel: KernelSpec,
    likelihood: GaussianLikelihood,
    optimizer: O,
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    y: &[f64],
    z: &[f64],
    n_inducing: usize,
) -> Result<FittedSgpr<O, I>, GprError> {
    let state = assemble_vfe(&kernel, likelihood, x, n_rows, n_cols, y, z, n_inducing)?;
    Ok(FittedSgpr {
        kernel,
        likelihood,
        optimizer,
        inducing: PhantomData,
        x_obs: x.to_vec(),
        z_obs: z.to_vec(),
        y: y.to_vec(),
        k_mm_l: state.k_mm_l,
        a: state.a,
        b_l: state.b_l,
        w: state.w,
        k_diag_sum: state.k_diag_sum,
        a_frobenius2: state.a_frobenius2,
        n: n_rows,
        m: n_inducing,
        d: n_cols,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn assemble_vfe(
    kernel: &KernelSpec,
    likelihood: GaussianLikelihood,
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    y: &[f64],
    z: &[f64],
    n_inducing: usize,
) -> Result<VfeState, GprError> {
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
    let noise = likelihood.noise_variance();
    let mut b = gram_aat_plus_noise(a.as_ref(), noise);
    let b_req = llt::factor::cholesky_in_place_scratch::<f64>(
        n_inducing,
        faer_par(n_inducing),
        Default::default(),
    );
    let mut b_scratch = MemBuffer::new(b_req);
    cholesky_lower_with_policy(
        &mut b,
        &mut b_scratch,
        JitterPolicy::default(),
        CholeskyStage::Fit,
    )?;
    let mut k_diag = vec![0.0; n_rows];
    compiled.fill_diag_points(x_mat.as_ref(), &mut k_diag)?;
    let k_diag_sum = k_diag.iter().sum();
    let a_frobenius2 = frobenius2(a.as_ref());
    let mut ay = Mat::zeros(n_inducing, 1);
    for i in 0..n_inducing {
        let mut sum = 0.0;
        for j in 0..n_rows {
            sum += a[(i, j)] * y[j];
        }
        ay[(i, 0)] = sum;
    }
    solve_llt_in_place(b.as_ref(), ay.as_mut());
    let mut w = vec![0.0; n_inducing];
    for i in 0..n_inducing {
        w[i] = ay[(i, 0)];
    }
    Ok(VfeState {
        k_mm_l: k_mm,
        a,
        b_l: b,
        w,
        k_diag_sum,
        a_frobenius2,
    })
}

pub(crate) fn fill_z_intervals(
    x: &[f64],
    n: usize,
    d: usize,
    out: &mut [Interval],
) -> Result<(), GprError> {
    if d == 0 || out.len() % d != 0 {
        return Err(GprError::InvalidHyperparameter {
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

pub(crate) struct KernelVar {
    pub(crate) d_kmm: Mat<f64>,
    pub(crate) d_kmn: Mat<f64>,
    pub(crate) d_kdiag: f64,
    pub(crate) d_noise: f64,
}

pub(crate) struct VfeEngine {
    pub(crate) l: Mat<f64>,
    pub(crate) a: Mat<f64>,
    pub(crate) b_l: Mat<f64>,
    pub(crate) w: Vec<f64>,
    pub(crate) noise: f64,
    pub(crate) k_diag_sum: f64,
    pub(crate) a_fro: f64,
    pub(crate) y: Vec<f64>,
    pub(crate) n: usize,
    pub(crate) m: usize,
}

impl VfeEngine {
    fn from_model<O, I>(model: &FittedSgpr<O, I>) -> Self {
        Self {
            l: model.k_mm_l.clone(),
            a: model.a.clone(),
            b_l: model.b_l.clone(),
            w: model.w.clone(),
            noise: model.likelihood.noise_variance(),
            k_diag_sum: model.k_diag_sum,
            a_fro: model.a_frobenius2,
            y: model.y.clone(),
            n: model.n,
            m: model.m,
        }
    }

    fn quad_and_trace(&self) -> (f64, f64) {
        let y_norm2: f64 = self.y.iter().map(|v| v * v).sum();
        let mut ay_dot_w = 0.0;
        for i in 0..self.m {
            let mut ay_i = 0.0;
            for j in 0..self.n {
                ay_i += self.a[(i, j)] * self.y[j];
            }
            ay_dot_w += ay_i * self.w[i];
        }
        let quad = (y_norm2 - ay_dot_w) / self.noise;
        let trace = (self.k_diag_sum - self.a_fro) / (2.0 * self.noise);
        (quad, trace)
    }

    fn first_tangent(&self, var: &KernelVar) -> VfeTangent {
        let phi = chol_phi(self.l.as_ref(), var.d_kmm.as_ref());
        let phi_l = tril_half(phi.as_ref());
        let mut da = var.d_kmn.clone();
        solve_lower(self.l.as_ref(), da.as_mut());
        mat_sub_mul(&mut da, phi_l.as_ref(), self.a.as_ref());
        let db = noise_plus_sym_prod(da.as_ref(), self.a.as_ref(), var.d_noise);
        let mut u = vec![0.0; self.m];
        for i in 0..self.m {
            let mut sum = 0.0;
            for j in 0..self.n {
                sum += da[(i, j)] * self.y[j];
            }
            u[i] = sum;
        }
        VfeTangent {
            phi,
            phi_l,
            da,
            db,
            u,
            d_kdiag: var.d_kdiag,
            d_noise: var.d_noise,
        }
    }

    fn directional(&self, var: &KernelVar) -> f64 {
        let t = self.first_tangent(var);
        self.directional_from_tangent(&t)
    }

    fn directional_from_tangent(&self, t: &VfeTangent) -> f64 {
        let (quad, trace) = self.quad_and_trace();
        let d_logdet_b = trace_solve(self.b_l.as_ref(), t.db.as_ref());
        let d_q = 2.0 * dot(&self.w, &t.u) - quad_form(&self.w, t.db.as_ref());
        let d_af = 2.0 * frobenius_dot(self.a.as_ref(), t.da.as_ref());
        let d_quad = -d_q / self.noise - quad * t.d_noise / self.noise;
        let d_trace = (t.d_kdiag - d_af) / (2.0 * self.noise) - trace * t.d_noise / self.noise;
        let n_minus_m = self.n as f64 - self.m as f64;
        0.5 * (n_minus_m * t.d_noise / self.noise + d_logdet_b + d_quad) + d_trace
    }

    fn second_directional(&self, ti: &VfeTangent, tj: &VfeTangent, dd: &KernelVar) -> f64 {
        let (quad, trace) = self.quad_and_trace();
        let phi_dd = chol_phi(self.l.as_ref(), dd.d_kmm.as_ref());
        let dphi_j_on_i = dphi_from(ti.phi.as_ref(), tj.phi_l.as_ref(), phi_dd.as_ref());
        let dphi_l = tril_half(dphi_j_on_i.as_ref());
        let mut linv_di_kmn = ti.da.clone();
        mat_add_mul(&mut linv_di_kmn, ti.phi_l.as_ref(), self.a.as_ref());
        let mut dda = dd.d_kmn.clone();
        solve_lower(self.l.as_ref(), dda.as_mut());
        mat_sub_mul(&mut dda, tj.phi_l.as_ref(), linv_di_kmn.as_ref());
        mat_sub_mul(&mut dda, dphi_l.as_ref(), self.a.as_ref());
        mat_sub_mul(&mut dda, ti.phi_l.as_ref(), tj.da.as_ref());
        let ddb = second_db(
            ti.da.as_ref(),
            tj.da.as_ref(),
            dda.as_ref(),
            self.a.as_ref(),
            dd.d_noise,
        );
        let mut ddu = vec![0.0; self.m];
        for i in 0..self.m {
            let mut sum = 0.0;
            for j in 0..self.n {
                sum += dda[(i, j)] * self.y[j];
            }
            ddu[i] = sum;
        }
        let d_logdet_b_i = trace_solve(self.b_l.as_ref(), ti.db.as_ref());
        let _ = d_logdet_b_i;
        let d2_logdet_b = second_logdet_b(
            self.b_l.as_ref(),
            ti.db.as_ref(),
            tj.db.as_ref(),
            ddb.as_ref(),
        );
        let dwi = dw_from(self.b_l.as_ref(), &self.w, ti.db.as_ref(), &ti.u);
        let dwj = dw_from(self.b_l.as_ref(), &self.w, tj.db.as_ref(), &tj.u);
        let d_q_i = 2.0 * dot(&self.w, &ti.u) - quad_form(&self.w, ti.db.as_ref());
        let d_q_j = 2.0 * dot(&self.w, &tj.u) - quad_form(&self.w, tj.db.as_ref());
        let d2_q = 2.0 * dot(&dwj, &ti.u) + 2.0 * dot(&self.w, &ddu)
            - dot(&dwj, &mat_vec(ti.db.as_ref(), &self.w))
            - quad_form(&self.w, ddb.as_ref())
            - dot(&self.w, &mat_vec(ti.db.as_ref(), &dwj));
        let d_af_i = 2.0 * frobenius_dot(self.a.as_ref(), ti.da.as_ref());
        let d_af_j = 2.0 * frobenius_dot(self.a.as_ref(), tj.da.as_ref());
        let d2_af = 2.0 * frobenius_dot(tj.da.as_ref(), ti.da.as_ref())
            + 2.0 * frobenius_dot(self.a.as_ref(), dda.as_ref());
        let d_quad_j = -d_q_j / self.noise - quad * tj.d_noise / self.noise;
        let d2_quad = -d2_q / self.noise + d_q_i * tj.d_noise / (self.noise * self.noise)
            - d_quad_j * ti.d_noise / self.noise
            - quad * dd.d_noise / self.noise
            + quad * ti.d_noise * tj.d_noise / (self.noise * self.noise);
        let d_trace_j =
            (tj.d_kdiag - d_af_j) / (2.0 * self.noise) - trace * tj.d_noise / self.noise;
        let d2_trace = (dd.d_kdiag - d2_af) / (2.0 * self.noise)
            - (ti.d_kdiag - d_af_i) * tj.d_noise / (2.0 * self.noise * self.noise)
            - d_trace_j * ti.d_noise / self.noise
            - trace * dd.d_noise / self.noise
            + trace * ti.d_noise * tj.d_noise / (self.noise * self.noise);
        let n_minus_m = self.n as f64 - self.m as f64;
        let d2_log_noise =
            dd.d_noise / self.noise - ti.d_noise * tj.d_noise / (self.noise * self.noise);
        let _ = dwi;
        0.5 * (n_minus_m * d2_log_noise + d2_logdet_b + d2_quad) + d2_trace
    }
}

pub(crate) struct VfeTangent {
    pub(crate) phi: Mat<f64>,
    pub(crate) phi_l: Mat<f64>,
    pub(crate) da: Mat<f64>,
    pub(crate) db: Mat<f64>,
    pub(crate) u: Vec<f64>,
    pub(crate) d_kdiag: f64,
    pub(crate) d_noise: f64,
}

pub(crate) fn analytic_gradient<O, I>(
    model: &FittedSgpr<O, I>,
    out: &mut [f64],
    include_z: bool,
) -> Result<(), GprError> {
    let engine = VfeEngine::from_model(model);
    let vars = collect_first_vars(model, include_z)?;
    for (i, var) in vars.iter().enumerate() {
        out[i] = engine.directional(var);
    }
    Ok(())
}

pub(crate) fn analytic_hessian<O, I>(
    model: &FittedSgpr<O, I>,
    out: &mut [f64],
    include_z: bool,
) -> Result<(), GprError> {
    let engine = VfeEngine::from_model(model);
    let vars = collect_first_vars(model, include_z)?;
    let tangents: Vec<VfeTangent> = vars.iter().map(|v| engine.first_tangent(v)).collect();
    let p = vars.len();
    for j in 0..p {
        for i in j..p {
            let dd = second_var(model, i, j, include_z)?;
            let hij = engine.second_directional(&tangents[i], &tangents[j], &dd);
            out[i * p + j] = hij;
            out[j * p + i] = hij;
        }
    }
    Ok(())
}

pub(crate) fn collect_first_vars<O, I>(
    model: &FittedSgpr<O, I>,
    include_z: bool,
) -> Result<Vec<KernelVar>, GprError> {
    let compiled = model.kernel.compile();
    let x = pack_points(&model.x_obs, model.n, model.d);
    let z = pack_points(&model.z_obs, model.m, model.d);
    let n_kernel = model.kernel.num_params();
    let n_theta = n_kernel + model.likelihood.num_params();
    let mut vars = Vec::with_capacity(n_theta + if include_z { model.m * model.d } else { 0 });
    for i in 0..n_kernel {
        vars.push(kernel_theta_var(
            &compiled,
            x.as_ref(),
            z.as_ref(),
            model.n,
            i,
        )?);
    }
    vars.push(likelihood_var(
        model.m,
        model.n,
        model.likelihood.noise_variance(),
    ));
    if include_z {
        for dim in 0..model.d {
            for p in 0..model.m {
                vars.push(z_coord_var(&compiled, x.as_ref(), z.as_ref(), p, dim)?);
            }
        }
    }
    Ok(vars)
}

pub(crate) fn kernel_theta_var(
    compiled: &CompiledKernel,
    x: MatRef<'_, f64>,
    z: MatRef<'_, f64>,
    n: usize,
    param_idx: usize,
) -> Result<KernelVar, GprError> {
    let m = z.nrows();
    let mut d_kmm = Mat::zeros(m, m);
    let mut scratch_mm = Mat::zeros(m, m);
    compiled.grad_points(
        z,
        d_kmm.as_mut(),
        param_idx,
        Triangle::Full,
        scratch_mm.as_mut(),
    )?;
    let mut d_kmn = Mat::zeros(m, n);
    let mut scratch_mn = Mat::zeros(m, n);
    compiled.grad_cross_points(z, x, d_kmn.as_mut(), param_idx, scratch_mn.as_mut())?;
    let mut d_xx = Mat::zeros(n, n);
    let mut scratch_xx = Mat::zeros(n, n);
    compiled.grad_points(
        x,
        d_xx.as_mut(),
        param_idx,
        Triangle::Lower,
        scratch_xx.as_mut(),
    )?;
    let mut d_kdiag = 0.0;
    for i in 0..n {
        d_kdiag += d_xx[(i, i)];
    }
    Ok(KernelVar {
        d_kmm,
        d_kmn,
        d_kdiag,
        d_noise: 0.0,
    })
}

pub(crate) fn likelihood_var(m: usize, n: usize, noise: f64) -> KernelVar {
    let _ = n;
    KernelVar {
        d_kmm: Mat::zeros(m, m),
        d_kmn: Mat::zeros(m, n),
        d_kdiag: 0.0,
        d_noise: noise,
    }
}

pub(crate) fn z_coord_var(
    compiled: &CompiledKernel,
    x: MatRef<'_, f64>,
    z: MatRef<'_, f64>,
    point: usize,
    dim: usize,
) -> Result<KernelVar, GprError> {
    let m = z.nrows();
    let n = x.nrows();
    let mut g2 = Mat::zeros(m, m);
    compiled.grad_wrt_coord_dim(z, z, g2.as_mut(), dim)?;
    let mut g_xz = Mat::zeros(n, m);
    compiled.grad_wrt_coord_dim(x, z, g_xz.as_mut(), dim)?;
    let mut d_kmm = Mat::zeros(m, m);
    for i in 0..m {
        d_kmm[(i, point)] += g2[(i, point)];
        d_kmm[(point, i)] += g2[(i, point)];
    }
    let mut d_kmn = Mat::zeros(m, n);
    for col in 0..n {
        d_kmn[(point, col)] = g_xz[(col, point)];
    }
    Ok(KernelVar {
        d_kmm,
        d_kmn,
        d_kdiag: 0.0,
        d_noise: 0.0,
    })
}

pub(crate) fn second_var<O, I>(
    model: &FittedSgpr<O, I>,
    i: usize,
    j: usize,
    include_z: bool,
) -> Result<KernelVar, GprError> {
    let n_kernel = model.kernel.num_params();
    let n_theta = n_kernel + model.likelihood.num_params();
    let compiled = model.kernel.compile();
    let x = pack_points(&model.x_obs, model.n, model.d);
    let z = pack_points(&model.z_obs, model.m, model.d);
    let m = model.m;
    let n = model.n;
    let z_index = |idx: usize| -> Option<(usize, usize)> {
        if !include_z || idx < n_theta {
            None
        } else {
            let local = idx - n_theta;
            Some((local % m, local / m))
        }
    };
    if i < n_kernel && j < n_kernel {
        return kernel_theta_second(&compiled, x.as_ref(), z.as_ref(), n, i, j);
    }
    if i == n_kernel && j == n_kernel {
        return Ok(likelihood_var(m, n, model.likelihood.noise_variance()));
    }
    if i < n_theta && j < n_theta {
        return Ok(KernelVar {
            d_kmm: Mat::zeros(m, m),
            d_kmn: Mat::zeros(m, n),
            d_kdiag: 0.0,
            d_noise: 0.0,
        });
    }
    if let (Some((pi, ei)), Some((pj, ej))) = (z_index(i), z_index(j)) {
        return z_z_second(&compiled, x.as_ref(), z.as_ref(), pi, ei, pj, ej);
    }
    let (theta, (p, e)) = if i < n_theta {
        (
            i,
            z_index(j).ok_or_else(|| GprError::InvalidHyperparameter {
                reason: "expected a free inducing coordinate".to_owned(),
            })?,
        )
    } else {
        (
            j,
            z_index(i).ok_or_else(|| GprError::InvalidHyperparameter {
                reason: "expected a free inducing coordinate".to_owned(),
            })?,
        )
    };
    if theta == n_kernel {
        return Ok(KernelVar {
            d_kmm: Mat::zeros(m, m),
            d_kmn: Mat::zeros(m, n),
            d_kdiag: 0.0,
            d_noise: 0.0,
        });
    }
    theta_z_second(&compiled, x.as_ref(), z.as_ref(), theta, p, e)
}

pub(crate) fn kernel_theta_second(
    compiled: &CompiledKernel,
    x: MatRef<'_, f64>,
    z: MatRef<'_, f64>,
    n: usize,
    i: usize,
    j: usize,
) -> Result<KernelVar, GprError> {
    let m = z.nrows();
    let mut d_kmm = Mat::zeros(m, m);
    let mut scratch_mm = Mat::zeros(m, m);
    compiled.hess_points(z, d_kmm.as_mut(), i, j, Triangle::Full, scratch_mm.as_mut())?;
    let mut d_kmn = Mat::zeros(m, n);
    let mut scratch_mn = Mat::zeros(m, n);
    compiled.hess_cross_points(z, x, d_kmn.as_mut(), i, j, scratch_mn.as_mut())?;
    let mut d_xx = Mat::zeros(n, n);
    let mut scratch_xx = Mat::zeros(n, n);
    compiled.hess_points(x, d_xx.as_mut(), i, j, Triangle::Lower, scratch_xx.as_mut())?;
    let mut d_kdiag = 0.0;
    for r in 0..n {
        d_kdiag += d_xx[(r, r)];
    }
    Ok(KernelVar {
        d_kmm,
        d_kmn,
        d_kdiag,
        d_noise: 0.0,
    })
}

pub(crate) fn z_z_second(
    compiled: &CompiledKernel,
    x: MatRef<'_, f64>,
    z: MatRef<'_, f64>,
    p: usize,
    e: usize,
    q: usize,
    f: usize,
) -> Result<KernelVar, GprError> {
    let m = z.nrows();
    let n = x.nrows();
    let mut h22 = Mat::zeros(m, m);
    compiled.hess_wrt_coord_dims(z, z, h22.as_mut(), e, f)?;
    let mut d_kmm = Mat::zeros(m, m);
    if p == q {
        for i in 0..m {
            if i != p {
                d_kmm[(i, p)] += h22[(i, p)];
                d_kmm[(p, i)] += h22[(i, p)];
            }
        }
    } else {
        let mut h12 = Mat::zeros(m, m);
        compiled.hess_wrt_coord_mixed(z, z, h12.as_mut(), e, f)?;
        let mut h21 = Mat::zeros(m, m);
        compiled.hess_wrt_coord_mixed(z, z, h21.as_mut(), f, e)?;
        d_kmm[(p, q)] = h12[(p, q)];
        d_kmm[(q, p)] = h21[(q, p)];
    }
    let mut d_kmn = Mat::zeros(m, n);
    if p == q {
        let mut h_xz = Mat::zeros(n, m);
        compiled.hess_wrt_coord_dims(x, z, h_xz.as_mut(), e, f)?;
        for col in 0..n {
            d_kmn[(p, col)] = h_xz[(col, p)];
        }
    }
    Ok(KernelVar {
        d_kmm,
        d_kmn,
        d_kdiag: 0.0,
        d_noise: 0.0,
    })
}

pub(crate) fn theta_z_second(
    compiled: &CompiledKernel,
    x: MatRef<'_, f64>,
    z: MatRef<'_, f64>,
    theta: usize,
    p: usize,
    e: usize,
) -> Result<KernelVar, GprError> {
    let m = z.nrows();
    let n = x.nrows();
    let mut g2 = Mat::zeros(m, m);
    compiled.hess_theta_coord_dim(z, z, g2.as_mut(), theta, e)?;
    let mut d_kmm = Mat::zeros(m, m);
    for i in 0..m {
        d_kmm[(i, p)] += g2[(i, p)];
        d_kmm[(p, i)] += g2[(i, p)];
    }
    let mut g_xz = Mat::zeros(n, m);
    compiled.hess_theta_coord_dim(x, z, g_xz.as_mut(), theta, e)?;
    let mut d_kmn = Mat::zeros(m, n);
    for col in 0..n {
        d_kmn[(p, col)] = g_xz[(col, p)];
    }
    Ok(KernelVar {
        d_kmm,
        d_kmn,
        d_kdiag: 0.0,
        d_noise: 0.0,
    })
}

pub(crate) fn chol_phi(l: MatRef<'_, f64>, dk: MatRef<'_, f64>) -> Mat<f64> {
    let m = l.nrows();
    let mut tmp = Mat::zeros(m, m);
    copy_mat(dk, tmp.as_mut());
    solve_lower(l, tmp.as_mut());
    let mut u = Mat::zeros(m, m);
    for i in 0..m {
        for j in 0..m {
            u[(i, j)] = tmp[(j, i)];
        }
    }
    solve_lower(l, u.as_mut());
    let mut phi = Mat::zeros(m, m);
    for i in 0..m {
        for j in 0..m {
            phi[(i, j)] = u[(j, i)];
        }
    }
    phi
}

pub(crate) fn tril_half(phi: MatRef<'_, f64>) -> Mat<f64> {
    let m = phi.nrows();
    let mut out = Mat::zeros(m, m);
    for j in 0..m {
        for i in j..m {
            out[(i, j)] = if i == j {
                0.5 * phi[(i, j)]
            } else {
                phi[(i, j)]
            };
        }
    }
    out
}

pub(crate) fn dphi_from(
    phi_i: MatRef<'_, f64>,
    phi_l_j: MatRef<'_, f64>,
    phi_dd: MatRef<'_, f64>,
) -> Mat<f64> {
    let m = phi_i.nrows();
    let mut out = Mat::zeros(m, m);
    for i in 0..m {
        for j in 0..m {
            let mut sum = phi_dd[(i, j)];
            for k in 0..m {
                sum -= phi_l_j[(i, k)] * phi_i[(k, j)];
                sum -= phi_i[(i, k)] * phi_l_j[(j, k)];
            }
            out[(i, j)] = sum;
        }
    }
    out
}

pub(crate) fn noise_plus_sym_prod(
    da: MatRef<'_, f64>,
    a: MatRef<'_, f64>,
    d_noise: f64,
) -> Mat<f64> {
    let m = da.nrows();
    let n = da.ncols();
    let mut db = Mat::zeros(m, m);
    for j in 0..m {
        for i in j..m {
            let mut sum = 0.0;
            for k in 0..n {
                sum += da[(i, k)] * a[(j, k)] + a[(i, k)] * da[(j, k)];
            }
            db[(i, j)] = sum;
            db[(j, i)] = sum;
        }
        db[(j, j)] += d_noise;
    }
    db
}

pub(crate) fn second_db(
    dai: MatRef<'_, f64>,
    daj: MatRef<'_, f64>,
    dda: MatRef<'_, f64>,
    a: MatRef<'_, f64>,
    dd_noise: f64,
) -> Mat<f64> {
    let m = a.nrows();
    let n = a.ncols();
    let mut db = Mat::zeros(m, m);
    for col in 0..m {
        for row in col..m {
            let mut sum = 0.0;
            for k in 0..n {
                sum += dda[(row, k)] * a[(col, k)]
                    + a[(row, k)] * dda[(col, k)]
                    + dai[(row, k)] * daj[(col, k)]
                    + daj[(row, k)] * dai[(col, k)];
            }
            db[(row, col)] = sum;
            db[(col, row)] = sum;
        }
        db[(col, col)] += dd_noise;
    }
    db
}

pub(crate) fn trace_solve(b_l: MatRef<'_, f64>, db: MatRef<'_, f64>) -> f64 {
    let mut solved = Mat::zeros(db.nrows(), db.ncols());
    copy_mat(db, solved.as_mut());
    solve_llt_in_place(b_l, solved.as_mut());
    let mut tr = 0.0;
    for i in 0..solved.nrows() {
        tr += solved[(i, i)];
    }
    tr
}

pub(crate) fn second_logdet_b(
    b_l: MatRef<'_, f64>,
    dbi: MatRef<'_, f64>,
    dbj: MatRef<'_, f64>,
    ddb: MatRef<'_, f64>,
) -> f64 {
    let mut si = Mat::zeros(dbi.nrows(), dbi.ncols());
    copy_mat(dbi, si.as_mut());
    solve_llt_in_place(b_l, si.as_mut());
    let mut sj = Mat::zeros(dbj.nrows(), dbj.ncols());
    copy_mat(dbj, sj.as_mut());
    solve_llt_in_place(b_l, sj.as_mut());
    let mut tr = 0.0;
    for i in 0..si.nrows() {
        for k in 0..si.ncols() {
            tr -= si[(k, i)] * sj[(i, k)];
        }
    }
    tr + trace_solve(b_l, ddb)
}

pub(crate) fn dw_from(b_l: MatRef<'_, f64>, w: &[f64], db: MatRef<'_, f64>, u: &[f64]) -> Vec<f64> {
    let m = w.len();
    let mut rhs = Mat::zeros(m, 1);
    let dbw = mat_vec(db, w);
    for i in 0..m {
        rhs[(i, 0)] = u[i] - dbw[i];
    }
    solve_llt_in_place(b_l, rhs.as_mut());
    let mut out = vec![0.0; m];
    for i in 0..m {
        out[i] = rhs[(i, 0)];
    }
    out
}

pub(crate) fn solve_lower(l: MatRef<'_, f64>, rhs: MatMut<'_, f64>) {
    let n = l.nrows();
    let n_rhs = rhs.ncols();
    faer::linalg::triangular_solve::solve_lower_triangular_in_place(
        l,
        rhs,
        faer_par_dims(n, n_rhs),
    );
}

pub(crate) fn copy_mat(src: MatRef<'_, f64>, mut dest: MatMut<'_, f64>) {
    for j in 0..src.ncols() {
        for i in 0..src.nrows() {
            dest[(i, j)] = src[(i, j)];
        }
    }
}

pub(crate) fn mat_sub_mul(dest: &mut Mat<f64>, left: MatRef<'_, f64>, right: MatRef<'_, f64>) {
    let m = dest.nrows();
    let n = dest.ncols();
    let k = left.ncols();
    for col in 0..n {
        for row in 0..m {
            let mut sum = 0.0;
            for t in 0..k {
                sum += left[(row, t)] * right[(t, col)];
            }
            dest[(row, col)] -= sum;
        }
    }
}

pub(crate) fn mat_add_mul(dest: &mut Mat<f64>, left: MatRef<'_, f64>, right: MatRef<'_, f64>) {
    let m = dest.nrows();
    let n = dest.ncols();
    let k = left.ncols();
    for col in 0..n {
        for row in 0..m {
            let mut sum = 0.0;
            for t in 0..k {
                sum += left[(row, t)] * right[(t, col)];
            }
            dest[(row, col)] += sum;
        }
    }
}

pub(crate) fn frobenius_dot(a: MatRef<'_, f64>, b: MatRef<'_, f64>) -> f64 {
    let mut sum = 0.0;
    for col in 0..a.ncols() {
        for row in 0..a.nrows() {
            sum += a[(row, col)] * b[(row, col)];
        }
    }
    sum
}

pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub(crate) fn quad_form(w: &[f64], m: MatRef<'_, f64>) -> f64 {
    let mw = mat_vec(m, w);
    dot(w, &mw)
}

pub(crate) fn mat_vec(m: MatRef<'_, f64>, v: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; m.nrows()];
    for i in 0..m.nrows() {
        let mut sum = 0.0;
        for j in 0..m.ncols() {
            sum += m[(i, j)] * v[j];
        }
        out[i] = sum;
    }
    out
}

pub(crate) fn kernel_cross(
    compiled: &CompiledKernel,
    x: MatRef<'_, f64>,
    xs: MatRef<'_, f64>,
) -> Result<Mat<f64>, GprError> {
    let n = x.nrows();
    let q = xs.nrows();
    let mut out = Mat::zeros(n, q);
    let mut scratch = Mat::zeros(n, q);
    match compiled.coord_mode()? {
        CoordMode::Dist | CoordMode::Either => {
            let mut dist = Mat::zeros(n, q);
            let mut thread_scratch = Vec::new();
            fill_squared_euclidean_cross(x, xs, dist.as_mut(), &mut thread_scratch);
            compiled.apply_cross(dist.as_ref(), out.as_mut(), scratch.as_mut())?;
        }
        CoordMode::Points => {
            compiled.apply_cross_points(x, xs, out.as_mut(), scratch.as_mut())?;
        }
        CoordMode::Mixed => {
            let mut dist = Mat::zeros(n, q);
            let mut thread_scratch = Vec::new();
            fill_squared_euclidean_cross(x, xs, dist.as_mut(), &mut thread_scratch);
            compiled.apply_cross_mixed(dist.as_ref(), x, xs, out.as_mut(), scratch.as_mut())?;
        }
    }
    Ok(out)
}

pub(crate) fn gram_aat_plus_noise(a: MatRef<'_, f64>, noise: f64) -> Mat<f64> {
    let m = a.nrows();
    let n = a.ncols();
    let mut b = Mat::zeros(m, m);
    for j in 0..m {
        for i in j..m {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a[(i, k)] * a[(j, k)];
            }
            b[(i, j)] = sum;
        }
        b[(j, j)] += noise;
    }
    b
}

pub(crate) fn frobenius2(a: MatRef<'_, f64>) -> f64 {
    let mut sum = 0.0;
    for col in 0..a.ncols() {
        for row in 0..a.nrows() {
            let v = a[(row, col)];
            sum += v * v;
        }
    }
    sum
}

pub(crate) fn solve_llt_in_place(l: MatRef<'_, f64>, mut rhs: MatMut<'_, f64>) {
    let n = l.nrows();
    let n_rhs = rhs.ncols();
    let par = faer_par_dims(n, n_rhs);
    let req = llt::solve::solve_in_place_scratch::<f64>(n, n_rhs, par);
    let mut buf = MemBuffer::new(req);
    let stack = MemStack::new(&mut buf);
    llt::solve::solve_in_place(l, rhs.as_mut(), par, stack);
}

pub(crate) fn validate_inducing(z: &[f64], m: usize, d: usize) -> Result<(), GprError> {
    if m == 0 || d == 0 {
        return Err(GprError::EmptyInput);
    }
    if z.len() % m == 0 {
        let z_dim = z.len() / m;
        if z_dim != d {
            return Err(GprError::DimensionMismatch {
                x_dim: z_dim,
                expected_dim: d,
            });
        }
    }
    let expected = m.checked_mul(d).ok_or(GprError::EmptyInput)?;
    if z.len() != expected {
        return Err(GprError::InvalidHyperparameter {
            reason: format!(
                "expected {expected} inducing feature values, got {}",
                z.len()
            ),
        });
    }
    if z.iter().any(|v| !v.is_finite()) {
        return Err(GprError::NonFiniteInput);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn vfe_neg_log_marginal_likelihood(
    a: MatRef<'_, f64>,
    b_l: MatRef<'_, f64>,
    w: &[f64],
    y: &[f64],
    k_diag_sum: f64,
    a_frobenius2: f64,
    noise: f64,
    n: usize,
    m: usize,
) -> Result<f64, GprError> {
    let mut log_det_b = 0.0;
    for i in 0..m {
        log_det_b += b_l[(i, i)].ln();
    }
    log_det_b *= 2.0;
    let n_minus_m = n as f64 - m as f64;
    let log_det = n_minus_m * noise.ln() + log_det_b;
    let y_norm2: f64 = y.iter().map(|v| v * v).sum();
    let mut ay_dot_w = 0.0;
    for i in 0..m {
        let mut ay_i = 0.0;
        for j in 0..n {
            ay_i += a[(i, j)] * y[j];
        }
        ay_dot_w += ay_i * w[i];
    }
    let quad = (y_norm2 - ay_dot_w) / noise;
    let trace = (k_diag_sum - a_frobenius2) / (2.0 * noise);
    let log_two_pi = (2.0 * std::f64::consts::PI).ln();
    Ok(0.5 * ((n as f64) * log_two_pi + log_det + quad) + trace)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn vfe_predict(
    kernel: &KernelSpec,
    z_obs: &[f64],
    k_mm_l: MatRef<'_, f64>,
    b_l: MatRef<'_, f64>,
    w: &[f64],
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
    let mut binv_astar = k_sz.clone();
    solve_llt_in_place(b_l, binv_astar.as_mut());
    let mut out = Prediction {
        mean: vec![0.0; n_rows],
        variance: vec![0.0; n_rows],
        variance_kind: options.variance_kind,
    };
    for col in 0..n_rows {
        let mut mean = 0.0;
        let mut a_norm = 0.0;
        let mut binv_norm = 0.0;
        for row in 0..m {
            let a_star = k_sz[(row, col)];
            mean += a_star * w[row];
            a_norm += a_star * a_star;
            let solved = binv_astar[(row, col)];
            binv_norm += a_star * solved;
        }
        let mut latent = kss[col] - a_norm + noise * binv_norm;
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

pub(crate) fn chol_rank1_update(l: &mut Mat<f64>, v: &mut [f64]) {
    let n = l.nrows();
    for k in 0..n {
        let lkk = l[(k, k)];
        let vk = v[k];
        let r = lkk.hypot(vk);
        let c = r / lkk;
        let s = vk / lkk;
        l[(k, k)] = r;
        for i in (k + 1)..n {
            let li = l[(i, k)];
            let vi = v[i];
            l[(i, k)] = (li + s * vi) / c;
            v[i] = c * vi - s * l[(i, k)];
        }
    }
}

pub(crate) fn chol_rank1_downdate(l: &mut Mat<f64>, v: &mut [f64]) -> bool {
    let n = l.nrows();
    for k in 0..n {
        let lkk = l[(k, k)];
        let vk = v[k];
        let r2 = lkk * lkk - vk * vk;
        if r2 <= 0.0 || !r2.is_finite() {
            return false;
        }
        let r = r2.sqrt();
        let c = r / lkk;
        let s = vk / lkk;
        l[(k, k)] = r;
        for i in (k + 1)..n {
            let li = l[(i, k)];
            let vi = v[i];
            l[(i, k)] = (li - s * vi) / c;
            v[i] = c * vi - s * l[(i, k)];
        }
    }
    true
}

pub(crate) fn refresh_w(a: MatRef<'_, f64>, b_l: MatRef<'_, f64>, y: &[f64]) -> Vec<f64> {
    let m = a.nrows();
    let n = a.ncols();
    let mut ay = Mat::zeros(m, 1);
    for i in 0..m {
        let mut sum = 0.0;
        for j in 0..n {
            sum += a[(i, j)] * y[j];
        }
        ay[(i, 0)] = sum;
    }
    solve_llt_in_place(b_l, ay.as_mut());
    let mut w = vec![0.0; m];
    for i in 0..m {
        w[i] = ay[(i, 0)];
    }
    w
}

pub(crate) fn append_column(a: &Mat<f64>, col: MatRef<'_, f64>) -> Mat<f64> {
    let m = a.nrows();
    let n = a.ncols();
    let mut out = Mat::zeros(m, n + 1);
    for j in 0..n {
        for i in 0..m {
            out[(i, j)] = a[(i, j)];
        }
    }
    for i in 0..m {
        out[(i, n)] = col[(i, 0)];
    }
    out
}

pub(crate) fn remove_column(a: &Mat<f64>, idx: usize) -> Mat<f64> {
    let m = a.nrows();
    let n = a.ncols();
    let mut out = Mat::zeros(m, n - 1);
    let mut dest = 0;
    for j in 0..n {
        if j == idx {
            continue;
        }
        for i in 0..m {
            out[(i, dest)] = a[(i, j)];
        }
        dest += 1;
    }
    out
}

pub(crate) fn append_point(x: &[f64], n: usize, d: usize, x_new: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; (n + 1) * d];
    for dim in 0..d {
        for i in 0..n {
            out[i + (n + 1) * dim] = x[i + n * dim];
        }
        out[n + (n + 1) * dim] = x_new[dim];
    }
    out
}

pub(crate) fn remove_point(x: &[f64], n: usize, d: usize, idx: usize) -> Vec<f64> {
    let mut out = vec![0.0; (n - 1) * d];
    for dim in 0..d {
        let mut dest = 0;
        for i in 0..n {
            if i == idx {
                continue;
            }
            out[dest + (n - 1) * dim] = x[i + n * dim];
            dest += 1;
        }
    }
    out
}

pub(crate) fn point_at(x: &[f64], n: usize, d: usize, idx: usize) -> Vec<f64> {
    let mut out = vec![0.0; d];
    for dim in 0..d {
        out[dim] = x[idx + n * dim];
    }
    out
}

pub(crate) fn kernel_column(
    kernel: &KernelSpec,
    z: &[f64],
    m: usize,
    x_pt: &[f64],
    d: usize,
) -> Result<Mat<f64>, GprError> {
    let compiled = kernel.compile();
    let z_mat = pack_points(z, m, d);
    let x_mat = pack_points(x_pt, 1, d);
    kernel_cross(&compiled, z_mat.as_ref(), x_mat.as_ref())
}

pub(crate) fn kernel_diag_at(kernel: &KernelSpec, x_pt: &[f64], d: usize) -> Result<f64, GprError> {
    let compiled = kernel.compile();
    let x_mat = pack_points(x_pt, 1, d);
    let mut diag = vec![0.0; 1];
    compiled.fill_diag_points(x_mat.as_ref(), &mut diag)?;
    Ok(diag[0])
}

pub(crate) fn solve_lmm(k_mm_l: MatRef<'_, f64>, mut col: MatMut<'_, f64>) {
    let m = k_mm_l.nrows();
    faer::linalg::triangular_solve::solve_lower_triangular_in_place(
        k_mm_l,
        col.as_mut(),
        faer_par_dims(m, 1),
    );
}

/// Appends one inducing point at the end by a bordered LLT of `K_mm` and `B`.
///
/// `A` gains a row. `k_diag_sum` is unchanged. `w` is solved from the new `B`.
#[allow(clippy::too_many_arguments)] // kernel, data, and new `Z` stay explicit
pub(crate) fn inducing_insert(
    state: &mut VfeState,
    kernel: &KernelSpec,
    noise: f64,
    x: &[f64],
    n: usize,
    d: usize,
    y: &[f64],
    z: &[f64],
    m: usize,
    z_new: &[f64],
) -> Result<(), GprError> {
    validate_inducing(z, m, d)?;
    validate_inducing(z_new, 1, d)?;
    if n == 0 {
        return Err(GprError::EmptyInput);
    }
    let compiled = kernel.compile();
    let z_mat = pack_points(z, m, d);
    let z_new_mat = pack_points(z_new, 1, d);
    let x_mat = pack_points(x, n, d);
    let mut k_zz = kernel_cross(&compiled, z_mat.as_ref(), z_new_mat.as_ref())?;
    let k_nn = kernel_diag_at(kernel, z_new, d)?;
    let k_zx = kernel_cross(&compiled, z_new_mat.as_ref(), x_mat.as_ref())?;
    solve_lmm(state.k_mm_l.as_ref(), k_zz.as_mut());
    let mut ell2 = k_nn;
    for i in 0..m {
        let li = k_zz[(i, 0)];
        ell2 -= li * li;
    }
    if ell2 <= 0.0 || !ell2.is_finite() {
        return Err(GprError::CholeskyFailed {
            jitter: 0.0,
            matrix_size: m + 1,
            stage: CholeskyStage::OnlineInsert,
        });
    }
    let ell = ell2.sqrt();
    let mut a_new = vec![0.0; n];
    let mut a_new_norm2 = 0.0;
    for j in 0..n {
        let mut dot = 0.0;
        for i in 0..m {
            dot += k_zz[(i, 0)] * state.a[(i, j)];
        }
        let value = (k_zx[(0, j)] - dot) / ell;
        a_new[j] = value;
        a_new_norm2 += value * value;
    }
    let mut v = vec![0.0; m];
    for (i, slot) in v.iter_mut().enumerate() {
        let mut sum = 0.0;
        for (j, a_val) in a_new.iter().enumerate() {
            sum += state.a[(i, j)] * a_val;
        }
        *slot = sum;
    }
    let mut b_border = Mat::zeros(m, 1);
    for i in 0..m {
        b_border[(i, 0)] = v[i];
    }
    solve_lmm(state.b_l.as_ref(), b_border.as_mut());
    let mut beta2 = noise + a_new_norm2;
    for i in 0..m {
        let bi = b_border[(i, 0)];
        beta2 -= bi * bi;
    }
    if beta2 <= 0.0 || !beta2.is_finite() {
        return Err(GprError::CholeskyFailed {
            jitter: 0.0,
            matrix_size: m + 1,
            stage: CholeskyStage::OnlineInsert,
        });
    }
    let mut l_col = vec![0.0; m];
    for i in 0..m {
        l_col[i] = k_zz[(i, 0)];
    }
    let mut b_col = vec![0.0; m];
    for i in 0..m {
        b_col[i] = b_border[(i, 0)];
    }
    state.k_mm_l = append_chol_border(&state.k_mm_l, &l_col, ell);
    state.b_l = append_chol_border(&state.b_l, &b_col, beta2.sqrt());
    state.a = append_row(&state.a, &a_new);
    state.a_frobenius2 += a_new_norm2;
    state.w = refresh_w(state.a.as_ref(), state.b_l.as_ref(), y);
    Ok(())
}

/// Drops inducing row `idx` by a trailing cholupdate of `L_mm`.
///
/// Reuses `K(Z, X) = L A`, drops that row, and solves the reduced `A`.
/// `B` is formed again from the new `A`. `k_diag_sum` is unchanged.
pub(crate) fn inducing_delete(
    state: &mut VfeState,
    noise: f64,
    y: &[f64],
    idx: usize,
) -> Result<(), GprError> {
    let m = state.a.nrows();
    if m <= 1 {
        return Err(GprError::EmptyInput);
    }
    if idx >= m {
        return Err(GprError::InvalidHyperparameter {
            reason: "inducing index is out of range".to_owned(),
        });
    }
    let k_zx = mul_lower_left(state.k_mm_l.as_ref(), state.a.as_ref());
    let k_zx = remove_row(&k_zx, idx);
    state.k_mm_l = delete_chol_row(&state.k_mm_l, idx);
    let mut a = k_zx;
    solve_lower(state.k_mm_l.as_ref(), a.as_mut());
    let mut b = gram_aat_plus_noise(a.as_ref(), noise);
    factor_lower_in_place(&mut b, CholeskyStage::OnlineDelete)?;
    state.a = a;
    state.b_l = b;
    state.a_frobenius2 = frobenius2(state.a.as_ref());
    state.w = refresh_w(state.a.as_ref(), state.b_l.as_ref(), y);
    Ok(())
}

fn append_chol_border(l: &Mat<f64>, row: &[f64], ell: f64) -> Mat<f64> {
    let m = l.nrows();
    let mut out = Mat::zeros(m + 1, m + 1);
    for j in 0..m {
        for i in j..m {
            out[(i, j)] = l[(i, j)];
        }
        out[(m, j)] = row[j];
    }
    out[(m, m)] = ell;
    out
}

fn delete_chol_row(l: &Mat<f64>, idx: usize) -> Mat<f64> {
    let m = l.nrows();
    let trail = m - idx - 1;
    let mut work = l.clone();
    if trail > 0 {
        let mut l22 = Mat::zeros(trail, trail);
        let mut v = vec![0.0; trail];
        for j in 0..trail {
            for i in j..trail {
                l22[(i, j)] = work[(idx + 1 + i, idx + 1 + j)];
            }
            v[j] = work[(idx + 1 + j, idx)];
        }
        chol_rank1_update(&mut l22, &mut v);
        for j in 0..trail {
            for i in j..trail {
                work[(idx + 1 + i, idx + 1 + j)] = l22[(i, j)];
            }
        }
    }
    let mut out = Mat::zeros(m - 1, m - 1);
    let mut jo = 0;
    for j in 0..m {
        if j == idx {
            continue;
        }
        let mut io = 0;
        for i in 0..m {
            if i == idx {
                continue;
            }
            if io >= jo {
                out[(io, jo)] = work[(i, j)];
            }
            io += 1;
        }
        jo += 1;
    }
    out
}

fn append_row(a: &Mat<f64>, row: &[f64]) -> Mat<f64> {
    let m = a.nrows();
    let n = a.ncols();
    let mut out = Mat::zeros(m + 1, n);
    for j in 0..n {
        for i in 0..m {
            out[(i, j)] = a[(i, j)];
        }
        out[(m, j)] = row[j];
    }
    out
}

fn remove_row(a: &Mat<f64>, idx: usize) -> Mat<f64> {
    let m = a.nrows();
    let n = a.ncols();
    let mut out = Mat::zeros(m - 1, n);
    let mut dest = 0;
    for i in 0..m {
        if i == idx {
            continue;
        }
        for j in 0..n {
            out[(dest, j)] = a[(i, j)];
        }
        dest += 1;
    }
    out
}

fn mul_lower_left(l: MatRef<'_, f64>, a: MatRef<'_, f64>) -> Mat<f64> {
    let m = l.nrows();
    let n = a.ncols();
    let mut out = Mat::zeros(m, n);
    for j in 0..n {
        for i in 0..m {
            let mut sum = 0.0;
            for t in 0..=i {
                sum += l[(i, t)] * a[(t, j)];
            }
            out[(i, j)] = sum;
        }
    }
    out
}

fn factor_lower_in_place(mat: &mut Mat<f64>, stage: CholeskyStage) -> Result<(), GprError> {
    let n = mat.nrows();
    let req = llt::factor::cholesky_in_place_scratch::<f64>(n, faer_par(n), Default::default());
    let mut scratch = MemBuffer::new(req);
    cholesky_lower_with_policy(mat, &mut scratch, JitterPolicy::default(), stage)
}
