//! Analytic gradient and Hessian of the negative VFE bound.

use super::lit;
use crate::data::pack_points;
use crate::error::GprError;
use crate::kernel::GramInputs;
use crate::kernel::ScalarOps;
use crate::kernel::{CompiledKernel, KernelScalar, Triangle};
use crate::linalg::{
    copy_mat, dot, frobenius_dot, gemm, mat_add_mul, mat_sub_mul, mat_vec, quad_form, solve_llt,
    solve_lower,
};
use crate::precision::ModelPrecision;
use crate::sgpr::FittedSgpr;
use crate::sparse::KernelScratch;
use faer::{Accum, Mat, MatRef};

pub(crate) struct KernelVar<T: KernelScalar> {
    pub(crate) d_kmm: Mat<T>,
    pub(crate) d_kmn: Mat<T>,
    pub(crate) d_kdiag: T,
    pub(crate) d_noise: T,
}

pub(crate) struct VfeEngine<'a, T: KernelScalar> {
    pub(crate) l: MatRef<'a, T>,
    pub(crate) a: MatRef<'a, T>,
    pub(crate) b_l: MatRef<'a, T>,
    pub(crate) w: &'a [T],
    pub(crate) noise: T,
    pub(crate) y: &'a [T],
    pub(crate) n: usize,
    pub(crate) m: usize,
    quad: T,
    trace: T,
}

impl<'a, T: KernelScalar> VfeEngine<'a, T> {
    fn from_model<O, I, P>(model: &'a FittedSgpr<O, I, P>, y: &'a [T]) -> Self
    where
        P: ModelPrecision<Storage = T>,
    {
        let m = model.core.m;
        let n = model.core.n;
        let noise = lit::<T>(model.core.likelihood.noise_variance());
        let a = model.a.as_ref();
        let w = model.w.as_slice();
        let mut y_norm2 = lit::<T>(0.0);
        for v in y {
            y_norm2 += v * v;
        }
        let mut ay_dot_w = lit::<T>(0.0);
        for i in 0..m {
            let mut ay_i = lit::<T>(0.0);
            for j in 0..n {
                ay_i += a[(i, j)] * y[j];
            }
            ay_dot_w += ay_i * w[i];
        }
        Self {
            l: model.k_mm_l.as_ref(),
            a,
            b_l: model.b_l.as_ref(),
            w,
            noise,
            y,
            n,
            m,
            quad: (y_norm2 - ay_dot_w) / noise,
            trace: (model.k_diag_sum - model.a_frobenius2) / (lit::<T>(2.0) * noise),
        }
    }

    fn tangent_from(
        &self,
        d_kmm: MatRef<'_, T>,
        mut da: Mat<T>,
        d_kdiag: T,
        d_noise: T,
    ) -> VfeTangent<T> {
        let phi = chol_phi(self.l, d_kmm);
        let phi_l = tril_half(phi.as_ref());
        solve_lower(self.l, da.as_mut());
        mat_sub_mul(&mut da, phi_l.as_ref(), self.a);
        let db = noise_plus_sym_prod(da.as_ref(), self.a, d_noise);
        let mut u = vec![lit::<T>(0.0); self.m];
        for i in 0..self.m {
            let mut sum = lit::<T>(0.0);
            for j in 0..self.n {
                sum += da[(i, j)] * self.y[j];
            }
            u[i] = sum;
        }
        VfeTangent::<T> {
            phi,
            phi_l,
            da,
            db,
            u,
            d_kdiag,
            d_noise,
        }
    }

    fn first_tangent(&self, var: &KernelVar<T>) -> VfeTangent<T> {
        self.tangent_from(
            var.d_kmm.as_ref(),
            var.d_kmn.clone(),
            var.d_kdiag,
            var.d_noise,
        )
    }

    fn directional_owned(&self, var: KernelVar<T>) -> T {
        let t = self.tangent_from(var.d_kmm.as_ref(), var.d_kmn, var.d_kdiag, var.d_noise);
        self.directional_from_tangent(&t)
    }

    /// Likelihood `θ` has `∂K = 0` and `∂σn² = σn²`, so the `m×n` products are zero.
    fn directional_noise(&self, d_noise: f64) -> T {
        let d_noise = lit::<T>(d_noise);
        let m = self.m;
        let mut db = Mat::zeros(m, m);
        for i in 0..m {
            db[(i, i)] = d_noise;
        }
        let quad = self.quad;
        let trace = self.trace;
        let d_logdet_b = trace_solve(self.b_l, db.as_ref());
        let d_q = -quad_form(self.w, db.as_ref());
        let d_quad = -d_q / self.noise - quad * d_noise / self.noise;
        let d_trace = -trace * d_noise / self.noise;
        let n_minus_m = lit::<T>(self.n as f64) - lit::<T>(self.m as f64);
        lit::<T>(0.5) * (n_minus_m * d_noise / self.noise + d_logdet_b + d_quad) + d_trace
    }

    fn directional_from_tangent(&self, t: &VfeTangent<T>) -> T {
        let quad = self.quad;
        let trace = self.trace;
        let d_logdet_b = trace_solve(self.b_l, t.db.as_ref());
        let d_q = lit::<T>(2.0) * dot(self.w, &t.u) - quad_form(self.w, t.db.as_ref());
        let d_af = lit::<T>(2.0) * frobenius_dot(self.a, t.da.as_ref());
        let d_quad = -d_q / self.noise - quad * t.d_noise / self.noise;
        let d_trace =
            (t.d_kdiag - d_af) / (lit::<T>(2.0) * self.noise) - trace * t.d_noise / self.noise;
        let n_minus_m = lit::<T>(self.n as f64) - lit::<T>(self.m as f64);
        lit::<T>(0.5) * (n_minus_m * t.d_noise / self.noise + d_logdet_b + d_quad) + d_trace
    }

    fn second_directional(&self, ti: &VfeTangent<T>, tj: &VfeTangent<T>, dd: &KernelVar<T>) -> T {
        let quad = self.quad;
        let trace = self.trace;
        let phi_dd = chol_phi(self.l, dd.d_kmm.as_ref());
        let dphi_j_on_i = dphi_from(ti.phi.as_ref(), tj.phi_l.as_ref(), phi_dd.as_ref());
        let dphi_l = tril_half(dphi_j_on_i.as_ref());
        let mut linv_di_kmn = ti.da.clone();
        mat_add_mul(&mut linv_di_kmn, ti.phi_l.as_ref(), self.a);
        let mut dda = dd.d_kmn.clone();
        solve_lower(self.l, dda.as_mut());
        mat_sub_mul(&mut dda, tj.phi_l.as_ref(), linv_di_kmn.as_ref());
        mat_sub_mul(&mut dda, dphi_l.as_ref(), self.a);
        mat_sub_mul(&mut dda, ti.phi_l.as_ref(), tj.da.as_ref());
        let ddb = second_db(
            ti.da.as_ref(),
            tj.da.as_ref(),
            dda.as_ref(),
            self.a,
            dd.d_noise,
        );
        let mut ddu = vec![lit::<T>(0.0); self.m];
        for i in 0..self.m {
            let mut sum = lit::<T>(0.0);
            for j in 0..self.n {
                sum += dda[(i, j)] * self.y[j];
            }
            ddu[i] = sum;
        }
        let d_logdet_b_i = trace_solve(self.b_l, ti.db.as_ref());
        let _ = d_logdet_b_i;
        let d2_logdet_b = second_logdet_b(self.b_l, ti.db.as_ref(), tj.db.as_ref(), ddb.as_ref());
        let dwi = dw_from(self.b_l, self.w, ti.db.as_ref(), &ti.u);
        let dwj = dw_from(self.b_l, self.w, tj.db.as_ref(), &tj.u);
        let d_q_i = lit::<T>(2.0) * dot(self.w, &ti.u) - quad_form(self.w, ti.db.as_ref());
        let d_q_j = lit::<T>(2.0) * dot(self.w, &tj.u) - quad_form(self.w, tj.db.as_ref());
        let d2_q = lit::<T>(2.0) * dot(&dwj, &ti.u) + lit::<T>(2.0) * dot(self.w, &ddu)
            - dot(&dwj, &mat_vec(ti.db.as_ref(), self.w))
            - quad_form(self.w, ddb.as_ref())
            - dot(self.w, &mat_vec(ti.db.as_ref(), &dwj));
        let d_af_i = lit::<T>(2.0) * frobenius_dot(self.a, ti.da.as_ref());
        let d_af_j = lit::<T>(2.0) * frobenius_dot(self.a, tj.da.as_ref());
        let d2_af = lit::<T>(2.0) * frobenius_dot(tj.da.as_ref(), ti.da.as_ref())
            + lit::<T>(2.0) * frobenius_dot(self.a, dda.as_ref());
        let d_quad_j = -d_q_j / self.noise - quad * tj.d_noise / self.noise;
        let d2_quad = -d2_q / self.noise + d_q_i * tj.d_noise / (self.noise * self.noise)
            - d_quad_j * ti.d_noise / self.noise
            - quad * dd.d_noise / self.noise
            + quad * ti.d_noise * tj.d_noise / (self.noise * self.noise);
        let d_trace_j =
            (tj.d_kdiag - d_af_j) / (lit::<T>(2.0) * self.noise) - trace * tj.d_noise / self.noise;
        let d2_trace = (dd.d_kdiag - d2_af) / (lit::<T>(2.0) * self.noise)
            - (ti.d_kdiag - d_af_i) * tj.d_noise / (lit::<T>(2.0) * self.noise * self.noise)
            - d_trace_j * ti.d_noise / self.noise
            - trace * dd.d_noise / self.noise
            + trace * ti.d_noise * tj.d_noise / (self.noise * self.noise);
        let n_minus_m = lit::<T>(self.n as f64) - lit::<T>(self.m as f64);
        let d2_log_noise =
            dd.d_noise / self.noise - ti.d_noise * tj.d_noise / (self.noise * self.noise);
        let _ = dwi;
        lit::<T>(0.5) * (n_minus_m * d2_log_noise + d2_logdet_b + d2_quad) + d2_trace
    }
}

pub(crate) struct VfeTangent<T: KernelScalar> {
    pub(crate) phi: Mat<T>,
    pub(crate) phi_l: Mat<T>,
    pub(crate) da: Mat<T>,
    pub(crate) db: Mat<T>,
    pub(crate) u: Vec<T>,
    pub(crate) d_kdiag: T,
    pub(crate) d_noise: T,
}

pub(crate) fn analytic_gradient<M: crate::math::KernelMath, O, I, P>(
    model: &FittedSgpr<O, I, P>,
    out: &mut [f64],
    include_z: bool,
    ks: &mut KernelScratch<P::Storage>,
) -> Result<(), GprError>
where
    P: ModelPrecision,
{
    let mut y_cast = P::Storage::empty_rows();
    let y_s = P::Storage::storage_rows(&model.core.y, &mut y_cast);
    let engine = VfeEngine::<P::Storage>::from_model::<_, _, _>(model, y_s);
    let compiled = model.core.kernel.compile_as::<P::Storage>();
    let x64 = pack_points(&model.core.x_obs, model.core.n, model.core.d);
    let z64 = pack_points(&model.core.z_obs, model.core.m, model.core.d);
    let mut x_cast = P::Storage::empty_cols();
    let mut z_cast = P::Storage::empty_cols();
    let x = P::Storage::storage_cols(x64.as_ref(), &mut x_cast);
    let z = P::Storage::storage_cols(z64.as_ref(), &mut z_cast);
    let n_kernel = model.core.kernel.num_params();
    for (i, slot) in out.iter_mut().enumerate().take(n_kernel) {
        let var = kernel_theta_var::<M, _>(&compiled, ks, x, z, model.core.n, i)?;
        *slot = engine.directional_owned(var).to_f64();
    }
    out[n_kernel] = engine
        .directional_noise(model.core.likelihood.noise_variance())
        .to_f64();
    if include_z {
        let mut idx = n_kernel + 1;
        for dim in 0..model.core.d {
            for p in 0..model.core.m {
                let var = z_coord_var::<M, _>(&compiled, ks, x, z, p, dim)?;
                out[idx] = engine.directional_owned(var).to_f64();
                idx += 1;
            }
        }
    }
    Ok(())
}

pub(crate) fn analytic_hessian<M: crate::math::KernelMath, O, I, P>(
    model: &FittedSgpr<O, I, P>,
    out: &mut [f64],
    include_z: bool,
    ks: &mut KernelScratch<P::Storage>,
) -> Result<(), GprError>
where
    P: ModelPrecision,
{
    let mut y_cast = P::Storage::empty_rows();
    let y_s = P::Storage::storage_rows(&model.core.y, &mut y_cast);
    let engine = VfeEngine::<P::Storage>::from_model::<_, _, _>(model, y_s);
    let vars = collect_first_vars::<M, _, _, _>(model, include_z, ks)?;
    let tangents: Vec<VfeTangent<P::Storage>> =
        vars.iter().map(|v| engine.first_tangent(v)).collect();
    let p = vars.len();
    for j in 0..p {
        for i in j..p {
            let dd = second_var::<M, _, _, _>(model, i, j, include_z, ks)?;
            let hij = engine
                .second_directional(&tangents[i], &tangents[j], &dd)
                .to_f64();
            out[i * p + j] = hij;
            out[j * p + i] = hij;
        }
    }
    Ok(())
}

pub(crate) fn collect_first_vars<M: crate::math::KernelMath, O, I, P>(
    model: &FittedSgpr<O, I, P>,
    include_z: bool,
    ks: &mut KernelScratch<P::Storage>,
) -> Result<Vec<KernelVar<P::Storage>>, GprError>
where
    P: ModelPrecision,
{
    let compiled = model.core.kernel.compile_as::<P::Storage>();
    let x64 = pack_points(&model.core.x_obs, model.core.n, model.core.d);
    let z64 = pack_points(&model.core.z_obs, model.core.m, model.core.d);
    let mut x_cast = P::Storage::empty_cols();
    let mut z_cast = P::Storage::empty_cols();
    let x = P::Storage::storage_cols(x64.as_ref(), &mut x_cast);
    let z = P::Storage::storage_cols(z64.as_ref(), &mut z_cast);
    let n_kernel = model.core.kernel.num_params();
    let n_theta = n_kernel + model.core.likelihood.num_params();
    let mut vars = Vec::with_capacity(
        n_theta
            + if include_z {
                model.core.m * model.core.d
            } else {
                0
            },
    );
    for i in 0..n_kernel {
        vars.push(kernel_theta_var::<M, _>(
            &compiled,
            ks,
            x,
            z,
            model.core.n,
            i,
        )?);
    }
    vars.push(likelihood_var(
        model.core.m,
        model.core.n,
        model.core.likelihood.noise_variance(),
    ));
    if include_z {
        for dim in 0..model.core.d {
            for p in 0..model.core.m {
                vars.push(z_coord_var::<M, _>(&compiled, ks, x, z, p, dim)?);
            }
        }
    }
    Ok(vars)
}

pub(crate) fn kernel_theta_var<M: crate::math::KernelMath, T>(
    compiled: &CompiledKernel<T>,
    ks: &mut KernelScratch<T>,
    x: MatRef<'_, T>,
    z: MatRef<'_, T>,
    n: usize,
    param_idx: usize,
) -> Result<KernelVar<T>, GprError>
where
    T: KernelScalar,
{
    let m = z.nrows();
    let mut d_kmm = Mat::zeros(m, m);
    ks.grad::<M>(
        compiled,
        GramInputs::points(z),
        d_kmm.as_mut(),
        param_idx,
        Triangle::Full,
    )?;
    let mut d_kmn = Mat::zeros(m, n);
    compiled.grad_cross_points::<M>(z, x, d_kmn.as_mut(), param_idx, ks.scratch(m, n))?;
    let mut diag = vec![lit::<T>(0.0); n];
    compiled.grad_diag_points::<M>(x, &mut diag, param_idx)?;
    let d_kdiag = diag.iter().fold(lit::<T>(0.0), |acc, v| acc + *v);
    Ok(KernelVar::<T> {
        d_kmm,
        d_kmn,
        d_kdiag,
        d_noise: lit::<T>(0.0),
    })
}

pub(crate) fn likelihood_var<T: KernelScalar>(m: usize, n: usize, noise: f64) -> KernelVar<T> {
    let _ = n;
    KernelVar::<T> {
        d_kmm: Mat::zeros(m, m),
        d_kmn: Mat::zeros(m, n),
        d_kdiag: lit::<T>(0.0),
        d_noise: lit::<T>(noise),
    }
}

pub(crate) fn z_coord_var<M: crate::math::KernelMath, T>(
    compiled: &CompiledKernel<T>,
    ks: &mut KernelScratch<T>,
    x: MatRef<'_, T>,
    z: MatRef<'_, T>,
    point: usize,
    dim: usize,
) -> Result<KernelVar<T>, GprError>
where
    T: KernelScalar,
{
    let m = z.nrows();
    let n = x.nrows();
    let mut g2 = Mat::zeros(m, m);
    compiled.grad_wrt_coord_dim_with::<M>(z, z, g2.as_mut(), dim, ks.scratch(m, m))?;
    let mut g_xz = Mat::zeros(n, m);
    compiled.grad_wrt_coord_dim_with::<M>(x, z, g_xz.as_mut(), dim, ks.scratch(n, m))?;
    let mut d_kmm = Mat::zeros(m, m);
    for i in 0..m {
        d_kmm[(i, point)] += g2[(i, point)];
        d_kmm[(point, i)] += g2[(i, point)];
    }
    let mut d_kmn = Mat::zeros(m, n);
    for col in 0..n {
        d_kmn[(point, col)] = g_xz[(col, point)];
    }
    Ok(KernelVar::<T> {
        d_kmm,
        d_kmn,
        d_kdiag: lit::<T>(0.0),
        d_noise: lit::<T>(0.0),
    })
}

pub(crate) fn second_var<M: crate::math::KernelMath, O, I, P>(
    model: &FittedSgpr<O, I, P>,
    i: usize,
    j: usize,
    include_z: bool,
    ks: &mut KernelScratch<P::Storage>,
) -> Result<KernelVar<P::Storage>, GprError>
where
    P: ModelPrecision,
{
    let n_kernel = model.core.kernel.num_params();
    let n_theta = n_kernel + model.core.likelihood.num_params();
    let compiled = model.core.kernel.compile_as::<P::Storage>();
    let x64 = pack_points(&model.core.x_obs, model.core.n, model.core.d);
    let z64 = pack_points(&model.core.z_obs, model.core.m, model.core.d);
    let mut x_cast = P::Storage::empty_cols();
    let mut z_cast = P::Storage::empty_cols();
    let x = P::Storage::storage_cols(x64.as_ref(), &mut x_cast);
    let z = P::Storage::storage_cols(z64.as_ref(), &mut z_cast);
    let m = model.core.m;
    let n = model.core.n;
    let z_index = |idx: usize| -> Option<(usize, usize)> {
        if !include_z || idx < n_theta {
            None
        } else {
            let local = idx - n_theta;
            Some((local % m, local / m))
        }
    };
    if i < n_kernel && j < n_kernel {
        return kernel_theta_second::<M, _>(&compiled, ks, x, z, n, i, j);
    }
    if i == n_kernel && j == n_kernel {
        return Ok(likelihood_var(m, n, model.core.likelihood.noise_variance()));
    }
    if i < n_theta && j < n_theta {
        return Ok(KernelVar::<P::Storage> {
            d_kmm: Mat::zeros(m, m),
            d_kmn: Mat::zeros(m, n),
            d_kdiag: lit::<P::Storage>(0.0),
            d_noise: lit::<P::Storage>(0.0),
        });
    }
    if let (Some((pi, ei)), Some((pj, ej))) = (z_index(i), z_index(j)) {
        return z_z_second::<M, _>(&compiled, ks, x, z, pi, ei, pj, ej);
    }
    let (theta, (p, e)) = if i < n_theta {
        (
            i,
            z_index(j).ok_or_else(|| GprError::IndexOutOfRange {
                reason: "expected a free inducing coordinate".to_owned(),
            })?,
        )
    } else {
        (
            j,
            z_index(i).ok_or_else(|| GprError::IndexOutOfRange {
                reason: "expected a free inducing coordinate".to_owned(),
            })?,
        )
    };
    if theta == n_kernel {
        return Ok(KernelVar::<P::Storage> {
            d_kmm: Mat::zeros(m, m),
            d_kmn: Mat::zeros(m, n),
            d_kdiag: lit::<P::Storage>(0.0),
            d_noise: lit::<P::Storage>(0.0),
        });
    }
    theta_z_second::<M, _>(&compiled, x, z, theta, p, e)
}

pub(crate) fn kernel_theta_second<M: crate::math::KernelMath, T>(
    compiled: &CompiledKernel<T>,
    ks: &mut KernelScratch<T>,
    x: MatRef<'_, T>,
    z: MatRef<'_, T>,
    n: usize,
    i: usize,
    j: usize,
) -> Result<KernelVar<T>, GprError>
where
    T: KernelScalar,
{
    let m = z.nrows();
    let mut d_kmm = Mat::zeros(m, m);
    ks.hess::<M>(
        compiled,
        GramInputs::points(z),
        d_kmm.as_mut(),
        (i, j),
        Triangle::Full,
    )?;
    let mut d_kmn = Mat::zeros(m, n);
    compiled.hess_cross_points::<M>(z, x, d_kmn.as_mut(), i, j, ks.scratch(m, n))?;
    let mut diag = vec![lit::<T>(0.0); n];
    compiled.hess_diag_points::<M>(x, &mut diag, i, j)?;
    let d_kdiag = diag.iter().fold(lit::<T>(0.0), |acc, v| acc + *v);
    Ok(KernelVar::<T> {
        d_kmm,
        d_kmn,
        d_kdiag,
        d_noise: lit::<T>(0.0),
    })
}

// The kernel, both views, its scratch, and the two (point, dimension) pairs.
#[allow(clippy::too_many_arguments)]
pub(crate) fn z_z_second<M: crate::math::KernelMath, T>(
    compiled: &CompiledKernel<T>,
    ks: &mut KernelScratch<T>,
    x: MatRef<'_, T>,
    z: MatRef<'_, T>,
    p: usize,
    e: usize,
    q: usize,
    f: usize,
) -> Result<KernelVar<T>, GprError>
where
    T: KernelScalar,
{
    let m = z.nrows();
    let n = x.nrows();
    let mut h22 = Mat::zeros(m, m);
    compiled.hess_wrt_coord_dims::<M>(z, z, h22.as_mut(), e, f, ks.scratch(m, m))?;
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
        compiled.hess_wrt_coord_mixed::<M>(z, z, h12.as_mut(), e, f, ks.scratch(m, m))?;
        let mut h21 = Mat::zeros(m, m);
        compiled.hess_wrt_coord_mixed::<M>(z, z, h21.as_mut(), f, e, ks.scratch(m, m))?;
        d_kmm[(p, q)] = h12[(p, q)];
        d_kmm[(q, p)] = h21[(q, p)];
    }
    let mut d_kmn = Mat::zeros(m, n);
    if p == q {
        let mut h_xz = Mat::zeros(n, m);
        compiled.hess_wrt_coord_dims::<M>(x, z, h_xz.as_mut(), e, f, ks.scratch(n, m))?;
        for col in 0..n {
            d_kmn[(p, col)] = h_xz[(col, p)];
        }
    }
    Ok(KernelVar::<T> {
        d_kmm,
        d_kmn,
        d_kdiag: lit::<T>(0.0),
        d_noise: lit::<T>(0.0),
    })
}

pub(crate) fn theta_z_second<M: crate::math::KernelMath, T>(
    compiled: &CompiledKernel<T>,
    x: MatRef<'_, T>,
    z: MatRef<'_, T>,
    theta: usize,
    p: usize,
    e: usize,
) -> Result<KernelVar<T>, GprError>
where
    T: KernelScalar,
{
    let m = z.nrows();
    let n = x.nrows();
    let mut g2 = Mat::zeros(m, m);
    compiled.hess_theta_coord_dim::<M>(z, z, g2.as_mut(), theta, e)?;
    let mut d_kmm = Mat::zeros(m, m);
    for i in 0..m {
        d_kmm[(i, p)] += g2[(i, p)];
        d_kmm[(p, i)] += g2[(i, p)];
    }
    let mut g_xz = Mat::zeros(n, m);
    compiled.hess_theta_coord_dim::<M>(x, z, g_xz.as_mut(), theta, e)?;
    let mut d_kmn = Mat::zeros(m, n);
    for col in 0..n {
        d_kmn[(p, col)] = g_xz[(col, p)];
    }
    Ok(KernelVar::<T> {
        d_kmm,
        d_kmn,
        d_kdiag: lit::<T>(0.0),
        d_noise: lit::<T>(0.0),
    })
}

pub(crate) fn chol_phi<T: KernelScalar>(l: MatRef<'_, T>, dk: MatRef<'_, T>) -> Mat<T> {
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

pub(crate) fn tril_half<T: KernelScalar>(phi: MatRef<'_, T>) -> Mat<T> {
    let m = phi.nrows();
    let mut out = Mat::zeros(m, m);
    for j in 0..m {
        for i in j..m {
            out[(i, j)] = if i == j {
                lit::<T>(0.5) * phi[(i, j)]
            } else {
                phi[(i, j)]
            };
        }
    }
    out
}

pub(crate) fn dphi_from<T: KernelScalar>(
    phi_i: MatRef<'_, T>,
    phi_l_j: MatRef<'_, T>,
    phi_dd: MatRef<'_, T>,
) -> Mat<T> {
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

pub(crate) fn noise_plus_sym_prod<T: KernelScalar>(
    da: MatRef<'_, T>,
    a: MatRef<'_, T>,
    d_noise: T,
) -> Mat<T> {
    let m = da.nrows();
    let mut db = Mat::zeros(m, m);
    // `da Aᵀ + A daᵀ`. Diagonal terms are `2 Σ_k da_ik a_ik`, matching the scalar sum.
    gemm(
        db.as_mut(),
        Accum::Replace,
        da,
        a.transpose(),
        lit::<T>(1.0),
    );
    gemm(db.as_mut(), Accum::Add, a, da.transpose(), lit::<T>(1.0));
    for j in 0..m {
        db[(j, j)] += d_noise;
    }
    db
}

pub(crate) fn second_db<T: KernelScalar>(
    dai: MatRef<'_, T>,
    daj: MatRef<'_, T>,
    dda: MatRef<'_, T>,
    a: MatRef<'_, T>,
    dd_noise: T,
) -> Mat<T> {
    let m = a.nrows();
    let mut db = Mat::zeros(m, m);
    gemm(
        db.as_mut(),
        Accum::Replace,
        dda,
        a.transpose(),
        lit::<T>(1.0),
    );
    gemm(db.as_mut(), Accum::Add, a, dda.transpose(), lit::<T>(1.0));
    gemm(db.as_mut(), Accum::Add, dai, daj.transpose(), lit::<T>(1.0));
    gemm(db.as_mut(), Accum::Add, daj, dai.transpose(), lit::<T>(1.0));
    for col in 0..m {
        db[(col, col)] += dd_noise;
    }
    db
}

pub(crate) fn trace_solve<T: KernelScalar>(b_l: MatRef<'_, T>, db: MatRef<'_, T>) -> T {
    let mut solved = Mat::zeros(db.nrows(), db.ncols());
    copy_mat(db, solved.as_mut());
    solve_llt(b_l, solved.as_mut());
    let mut tr = lit::<T>(0.0);
    for i in 0..solved.nrows() {
        tr += solved[(i, i)];
    }
    tr
}

pub(crate) fn second_logdet_b<T: KernelScalar>(
    b_l: MatRef<'_, T>,
    dbi: MatRef<'_, T>,
    dbj: MatRef<'_, T>,
    ddb: MatRef<'_, T>,
) -> T {
    let mut si = Mat::zeros(dbi.nrows(), dbi.ncols());
    copy_mat(dbi, si.as_mut());
    solve_llt(b_l, si.as_mut());
    let mut sj = Mat::zeros(dbj.nrows(), dbj.ncols());
    copy_mat(dbj, sj.as_mut());
    solve_llt(b_l, sj.as_mut());
    let mut tr = lit::<T>(0.0);
    for i in 0..si.nrows() {
        for k in 0..si.ncols() {
            tr -= si[(k, i)] * sj[(i, k)];
        }
    }
    tr + trace_solve(b_l, ddb)
}

pub(crate) fn dw_from<T: KernelScalar>(
    b_l: MatRef<'_, T>,
    w: &[T],
    db: MatRef<'_, T>,
    u: &[T],
) -> Vec<T> {
    let m = w.len();
    let mut rhs = Mat::zeros(m, 1);
    let dbw = mat_vec(db, w);
    for i in 0..m {
        rhs[(i, 0)] = u[i] - dbw[i];
    }
    solve_llt(b_l, rhs.as_mut());
    let mut out = vec![lit::<T>(0.0); m];
    for i in 0..m {
        out[i] = rhs[(i, 0)];
    }
    out
}
