use super::factor::*;
use super::*;
use crate::data::pack_points;
use crate::error::GprError;
use crate::kernel::{
    ConstantKernel, KernelSpec, LinearKernel, MaternArdKernel, MaternKernel, MaternNu,
    PeriodicKernel, RationalQuadraticArdKernel, RationalQuadraticKernel, RbfArdKernel, RbfKernel,
    WhiteKernel,
};
use crate::likelihood::GaussianLikelihood;
use crate::linalg::{faer_par_dims, frobenius2, solve_llt};
use crate::sgpr::SgprObjective;
use crate::{
    FastSimulatedAnnealing, Fixed, FreeInducing, Gpr, Lbfgs, NelderMead, Optimizer, PredictOptions,
    TrustRegion, VarianceKind,
};
use faer::{Mat, MatMut, MatRef};

const TOL: f64 = 1e-12;
const GRAD_FD: f64 = 1e-5;
const HESS_FD: f64 = 1e-4;

use crate::test_check::{assert_close, assert_send_sync, assert_slice_close};

struct Rank1Vfe {
    a: Mat<f64>,
    b_l: Mat<f64>,
    w: Vec<f64>,
    k_diag_sum: f64,
    a_frobenius2: f64,
    k_mm_l: Mat<f64>,
}

impl Rank1Vfe {
    fn from_fitted(fitted: &FittedSgpr<Fixed>) -> Self {
        Self {
            a: fitted.a.clone(),
            b_l: fitted.b_l.clone(),
            w: fitted.w.clone(),
            k_diag_sum: fitted.k_diag_sum,
            a_frobenius2: fitted.a_frobenius2,
            k_mm_l: fitted.k_mm_l.clone(),
        }
    }
}

fn reconstruct_llt(l: &Mat<f64>, i: usize, j: usize) -> f64 {
    let k_max = i.min(j);
    let mut sum = 0.0;
    for k in 0..=k_max {
        sum += l[(i, k)] * l[(j, k)];
    }
    sum
}

fn chol_rank1_update(l: &mut Mat<f64>, v: &mut [f64]) {
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

fn chol_rank1_downdate(l: &mut Mat<f64>, v: &mut [f64]) -> Result<(), &'static str> {
    let n = l.nrows();
    for k in 0..n {
        let lkk = l[(k, k)];
        let vk = v[k];
        let r2 = lkk * lkk - vk * vk;
        if r2 <= 0.0 || !r2.is_finite() {
            return Err("chol downdate lost positive definiteness");
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
    Ok(())
}

fn refresh_w(a: MatRef<'_, f64>, b_l: MatRef<'_, f64>, y: &[f64]) -> Vec<f64> {
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
    solve_llt(b_l, ay.as_mut());
    let mut w = vec![0.0; m];
    for i in 0..m {
        w[i] = ay[(i, 0)];
    }
    w
}

fn append_column(a: &Mat<f64>, col: MatRef<'_, f64>) -> Mat<f64> {
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

fn remove_column(a: &Mat<f64>, idx: usize) -> Mat<f64> {
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

fn append_point(x: &[f64], n: usize, d: usize, x_new: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; (n + 1) * d];
    for dim in 0..d {
        for i in 0..n {
            out[i + (n + 1) * dim] = x[i + n * dim];
        }
        out[n + (n + 1) * dim] = x_new[dim];
    }
    out
}

fn remove_point(x: &[f64], n: usize, d: usize, idx: usize) -> Vec<f64> {
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

fn point_at(x: &[f64], n: usize, d: usize, idx: usize) -> Vec<f64> {
    let mut out = vec![0.0; d];
    for dim in 0..d {
        out[dim] = x[idx + n * dim];
    }
    out
}

fn kernel_column(
    kernel: &KernelSpec,
    z: &[f64],
    m: usize,
    x_pt: &[f64],
    d: usize,
) -> Result<Mat<f64>, GprError> {
    let compiled = kernel.compile();
    let z_mat = pack_points(z, m, d);
    let x_mat = pack_points(x_pt, 1, d);
    crate::sparse::KernelScratch::new().cross::<crate::math::Accurate>(
        &compiled,
        z_mat.as_ref(),
        x_mat.as_ref(),
    )
}

fn kernel_diag_at(kernel: &KernelSpec, x_pt: &[f64], d: usize) -> Result<f64, GprError> {
    let compiled = kernel.compile();
    let x_mat = pack_points(x_pt, 1, d);
    let mut diag = vec![0.0; 1];
    compiled.fill_diag_points(x_mat.as_ref(), &mut diag)?;
    Ok(diag[0])
}

fn solve_lmm(k_mm_l: MatRef<'_, f64>, mut col: MatMut<'_, f64>) {
    let m = k_mm_l.nrows();
    faer::linalg::triangular_solve::solve_lower_triangular_in_place(
        k_mm_l,
        col.as_mut(),
        faer_par_dims(m, 1),
    );
}

fn rank1_insert(
    state: &mut Rank1Vfe,
    kernel: &KernelSpec,
    z: &[f64],
    d: usize,
    y: &mut Vec<f64>,
    x_new: &[f64],
    y_new: f64,
) -> Result<(), GprError> {
    let m = state.a.nrows();
    let mut a_col = kernel_column(kernel, z, m, x_new, d)?;
    solve_lmm(state.k_mm_l.as_ref(), a_col.as_mut());
    let mut v = vec![0.0; m];
    for (i, slot) in v.iter_mut().enumerate() {
        *slot = a_col[(i, 0)];
    }
    state.a_frobenius2 += frobenius2(a_col.as_ref());
    state.k_diag_sum += kernel_diag_at(kernel, x_new, d)?;
    state.a = append_column(&state.a, a_col.as_ref());
    chol_rank1_update(&mut state.b_l, &mut v);
    y.push(y_new);
    state.w = refresh_w(state.a.as_ref(), state.b_l.as_ref(), y);
    Ok(())
}

fn rank1_delete(
    state: &mut Rank1Vfe,
    kernel: &KernelSpec,
    x: &[f64],
    n: usize,
    d: usize,
    y: &mut Vec<f64>,
    idx: usize,
) -> Result<(), String> {
    let m = state.a.nrows();
    let mut v = vec![0.0; m];
    for (i, slot) in v.iter_mut().enumerate() {
        *slot = state.a[(i, idx)];
    }
    let x_pt = point_at(x, n, d, idx);
    let diag = kernel_diag_at(kernel, &x_pt, d).map_err(|e| e.to_string())?;
    let mut col_norm = 0.0;
    for value in &v {
        col_norm += *value * *value;
    }
    state.k_diag_sum -= diag;
    state.a_frobenius2 -= col_norm;
    state.a = remove_column(&state.a, idx);
    chol_rank1_downdate(&mut state.b_l, &mut v)?;
    y.remove(idx);
    state.w = refresh_w(state.a.as_ref(), state.b_l.as_ref(), y);
    Ok(())
}

fn factor_sparse(
    kernel: KernelSpec,
    x: &[f64],
    n: usize,
    d: usize,
    y: &[f64],
    z: &[f64],
    m: usize,
) -> FittedSgpr<Fixed> {
    let likelihood = GaussianLikelihood::new(0.1).expect("noise");
    Sgpr::new(kernel, likelihood)
        .with_optimizer(Fixed)
        .factor(x, n, d, y, z, m)
        .map_err(|(_, e)| e)
        .expect("factor")
}

fn factor_ok(kernel: KernelSpec, x: &[f64], n: usize, d: usize, y: &[f64], z: &[f64], m: usize) {
    let fitted = factor_sparse(kernel, x, n, d, y, z, m);
    assert_eq!(fitted.n(), n);
    assert_eq!(fitted.m(), m);
    assert_eq!(fitted.d(), d);
    assert_eq!(fitted.x(), x);
    assert_eq!(fitted.y(), y);
    assert_eq!(fitted.z(), z);
}

fn assert_matches_exact(
    kernel: KernelSpec,
    x: &[f64],
    n: usize,
    d: usize,
    y: &[f64],
    xs_extra: &[f64],
    n_extra: usize,
) {
    let likelihood = GaussianLikelihood::new(0.1).expect("noise");
    let sparse = Sgpr::new(kernel.clone(), likelihood)
        .with_optimizer(Fixed)
        .factor(x, n, d, y, x, n)
        .map_err(|(_, e)| e)
        .expect("sparse factor");
    let exact = Gpr::new(kernel, likelihood)
        .with_optimizer(Fixed)
        .factor(x, n, d, y)
        .map_err(|(_, e)| e)
        .expect("exact factor");
    assert_close(
        sparse.neg_log_marginal_likelihood().expect("sparse nlml"),
        exact.neg_log_marginal_likelihood().expect("exact nlml"),
        TOL,
    );
    assert_pred_close(&sparse, &exact, x, n, d);
    assert_pred_close(&sparse, &exact, xs_extra, n_extra, d);
}

fn assert_pred_close(
    sparse: &FittedSgpr<Fixed>,
    exact: &crate::FittedGpr<Fixed>,
    xs: &[f64],
    n_rows: usize,
    d: usize,
) {
    let kinds = [VarianceKind::Observation, VarianceKind::Latent];
    for kind in kinds {
        let options = PredictOptions {
            variance_kind: kind,
        };
        let got = sparse
            .predict_with(xs, n_rows, d, options)
            .expect("sparse predict");
        let want = exact
            .predict_with(xs, n_rows, d, options)
            .expect("exact predict");
        assert_eq!(got.variance_kind, kind);
        assert_eq!(got.mean.len(), n_rows);
        for i in 0..n_rows {
            assert_close(got.mean[i], want.mean[i], TOL);
            assert_close(got.variance[i], want.variance[i], TOL);
        }
    }
}

#[test]
fn factor_rbf_n4_m2() {
    factor_ok(
        KernelSpec::from(RbfKernel::new(1.0).expect("ℓ")),
        &[0.0, 1.0, 2.0, 3.0],
        4,
        1,
        &[0.0, 1.0, 0.5, 0.25],
        &[0.5, 2.5],
        2,
    );
}

#[test]
fn factor_matern_three_halves_n4_m2() {
    factor_ok(
        KernelSpec::from(MaternKernel::new(1.0, MaternNu::ThreeHalves).expect("ℓ")),
        &[0.0, 1.0, 2.0, 3.0],
        4,
        1,
        &[0.0, 1.0, 0.5, 0.25],
        &[0.5, 2.5],
        2,
    );
}

#[test]
fn factor_rbf_ard_2d_n4_m2() {
    factor_ok(
        KernelSpec::from(RbfArdKernel::new(&[1.0, 1.5]).expect("ℓ")),
        &[0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        4,
        2,
        &[0.0, 1.0, 0.5, 0.25],
        &[0.25, 0.75, 0.25, 0.75],
        2,
    );
}

#[test]
fn factor_rbf_plus_white_n4_m2() {
    factor_ok(
        KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"))
            + KernelSpec::from(WhiteKernel::new(0.05).expect("white")),
        &[0.0, 1.0, 2.0, 3.0],
        4,
        1,
        &[0.0, 1.0, 0.5, 0.25],
        &[0.5, 2.5],
        2,
    );
}

#[test]
fn rbf_n2_z_eq_x_matches_analytic_gram() {
    let x = [0.0, 1.0];
    let y = [0.0, 1.0];
    let likelihood = GaussianLikelihood::new(0.1).expect("noise");
    let fitted = Sgpr::new(
        KernelSpec::from(RbfKernel::new(1.0).expect("ℓ")),
        likelihood,
    )
    .with_optimizer(Fixed)
    .factor(&x, 2, 1, &y, &x, 2)
    .map_err(|(_, e)| e)
    .expect("factor");
    let l = fitted.k_mm_l();
    let reconstructed = |i: usize, j: usize| {
        let k_max = i.min(j);
        let mut sum = 0.0;
        for k in 0..=k_max {
            sum += l[(i, k)] * l[(j, k)];
        }
        sum
    };
    let off = (-0.5_f64).exp();
    assert_close(reconstructed(0, 0), 1.0, TOL);
    assert_close(reconstructed(1, 1), 1.0, TOL);
    assert_close(reconstructed(1, 0), off, TOL);
}

#[test]
fn rbf_n4_z_eq_x_matches_exact() {
    assert_matches_exact(
        KernelSpec::from(RbfKernel::new(1.0).expect("ℓ")),
        &[0.0, 1.0, 2.0, 3.0],
        4,
        1,
        &[0.0, 1.0, 0.5, 0.25],
        &[0.25, 3.5],
        2,
    );
}

#[test]
fn matern_n4_z_eq_x_matches_exact() {
    assert_matches_exact(
        KernelSpec::from(MaternKernel::new(1.0, MaternNu::ThreeHalves).expect("ℓ")),
        &[0.0, 1.0, 2.0, 3.0],
        4,
        1,
        &[0.0, 1.0, 0.5, 0.25],
        &[0.25, 3.5],
        2,
    );
}

#[test]
fn rbf_ard_n4_z_eq_x_matches_exact() {
    assert_matches_exact(
        KernelSpec::from(RbfArdKernel::new(&[1.0, 1.5]).expect("ℓ")),
        &[0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        4,
        2,
        &[0.0, 1.0, 0.5, 0.25],
        &[0.25, 0.75, 0.25, 0.75],
        2,
    );
}

#[test]
fn rbf_plus_white_n4_z_eq_x_matches_exact() {
    assert_matches_exact(
        KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"))
            + KernelSpec::from(WhiteKernel::new(0.05).expect("white")),
        &[0.0, 1.0, 2.0, 3.0],
        4,
        1,
        &[0.0, 1.0, 0.5, 0.25],
        &[0.25, 3.5],
        2,
    );
}

#[test]
fn rbf_n4_m2_predict_and_nlml_succeed() {
    let likelihood = GaussianLikelihood::new(0.1).expect("noise");
    let fitted = Sgpr::new(
        KernelSpec::from(RbfKernel::new(1.0).expect("ℓ")),
        likelihood,
    )
    .with_optimizer(Fixed)
    .factor(
        &[0.0, 1.0, 2.0, 3.0],
        4,
        1,
        &[0.0, 1.0, 0.5, 0.25],
        &[0.5, 2.5],
        2,
    )
    .map_err(|(_, e)| e)
    .expect("factor");
    let pred = fitted.predict(&[0.25, 3.5], 2, 1).expect("predict");
    assert_eq!(pred.mean.len(), 2);
    assert!(pred.mean.iter().all(|v| v.is_finite()));
    assert!(pred.variance.iter().all(|v| v.is_finite()));
    let nlml = fitted.neg_log_marginal_likelihood().expect("nlml");
    assert!(nlml.is_finite());
}

#[test]
fn empty_training_is_empty_input() {
    let kernel = KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"));
    let likelihood = GaussianLikelihood::new(0.1).expect("noise");
    let err = Sgpr::new(kernel, likelihood)
        .with_optimizer(Fixed)
        .factor(&[], 0, 1, &[], &[0.0], 1)
        .map_err(|(_, e)| e)
        .expect_err("empty n");
    assert_eq!(err, GprError::EmptyInput);
}

#[test]
fn zero_inducing_is_empty_input() {
    let kernel = KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"));
    let likelihood = GaussianLikelihood::new(0.1).expect("noise");
    let err = Sgpr::new(kernel, likelihood)
        .with_optimizer(Fixed)
        .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0], &[], 0)
        .map_err(|(_, e)| e)
        .expect_err("m = 0");
    assert_eq!(err, GprError::EmptyInput);
}

#[test]
fn inducing_feature_mismatch_is_dimension_mismatch() {
    let kernel = KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"));
    let likelihood = GaussianLikelihood::new(0.1).expect("noise");
    let err = Sgpr::new(kernel, likelihood)
        .with_optimizer(Fixed)
        .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0], &[0.0, 1.0, 0.0, 1.0], 2)
        .map_err(|(_, e)| e)
        .expect_err("z d = 2");
    assert_eq!(
        err,
        GprError::DimensionMismatch {
            x_dim: 2,
            expected_dim: 1,
        }
    );
}

#[test]
fn inducing_length_mismatch_is_length_mismatch() {
    let kernel = KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"));
    let likelihood = GaussianLikelihood::new(0.1).expect("noise");
    let err = Sgpr::new(kernel, likelihood)
        .with_optimizer(Fixed)
        .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0], &[0.0], 2)
        .map_err(|(_, e)| e)
        .expect_err("short z");
    assert!(matches!(err, GprError::LengthMismatch { .. }));
}

#[test]
fn empty_query_is_empty_input() {
    let kernel = KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"));
    let likelihood = GaussianLikelihood::new(0.1).expect("noise");
    let fitted = Sgpr::new(kernel, likelihood)
        .with_optimizer(Fixed)
        .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0], &[0.0, 1.0], 2)
        .map_err(|(_, e)| e)
        .expect("factor");
    let err = fitted.predict(&[], 0, 1).expect_err("empty query");
    assert_eq!(err, GprError::EmptyInput);
}

#[test]
fn query_feature_mismatch_is_dimension_mismatch() {
    let kernel = KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"));
    let likelihood = GaussianLikelihood::new(0.1).expect("noise");
    let fitted = Sgpr::new(kernel, likelihood)
        .with_optimizer(Fixed)
        .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0], &[0.0, 1.0], 2)
        .map_err(|(_, e)| e)
        .expect("factor");
    let err = fitted.predict(&[0.0, 0.0], 1, 2).expect_err("query d = 2");
    assert_eq!(
        err,
        GprError::DimensionMismatch {
            x_dim: 2,
            expected_dim: 1,
        }
    );
}

fn kernel_ard() -> KernelSpec {
    KernelSpec::from(RbfArdKernel::new(&[1.0, 1.5]).expect("ℓ"))
}

fn fd_grad_from_value<I: InducingLayout>(
    model: &mut FittedSgpr<Fixed, I>,
    params: &[f64],
) -> Vec<f64> {
    let mut out = vec![0.0; params.len()];
    let mut plus = params.to_vec();
    let mut minus = params.to_vec();
    for i in 0..params.len() {
        plus.copy_from_slice(params);
        minus.copy_from_slice(params);
        plus[i] += GRAD_FD;
        minus[i] -= GRAD_FD;
        model.set_params(&plus).expect("plus");
        let fp = model.neg_log_marginal_likelihood().expect("fp");
        model.set_params(&minus).expect("minus");
        let fm = model.neg_log_marginal_likelihood().expect("fm");
        out[i] = (fp - fm) / (2.0 * GRAD_FD);
    }
    model.set_params(params).expect("restore");
    out
}

fn fd_hess_from_grad<I: InducingLayout>(
    model: &mut FittedSgpr<Fixed, I>,
    params: &[f64],
) -> Vec<f64> {
    let p = params.len();
    let mut out = vec![0.0; p * p];
    let mut plus = params.to_vec();
    let mut minus = params.to_vec();
    let mut gp = vec![0.0; p];
    let mut gm = vec![0.0; p];
    for j in 0..p {
        plus.copy_from_slice(params);
        minus.copy_from_slice(params);
        plus[j] += HESS_FD;
        minus[j] -= HESS_FD;
        model
            .value_and_gradient_into(&plus, &mut gp)
            .expect("grad+");
        model
            .value_and_gradient_into(&minus, &mut gm)
            .expect("grad-");
        for i in 0..p {
            out[i * p + j] = (gp[i] - gm[i]) / (2.0 * HESS_FD);
        }
    }
    model.set_params(params).expect("restore");
    out
}

fn assert_z_eq_x_matches_exact_derivs(
    kernel: KernelSpec,
    x: &[f64],
    n: usize,
    d: usize,
    y: &[f64],
) {
    let likelihood = GaussianLikelihood::new(0.1).expect("noise");
    let mut sparse = Sgpr::new(kernel.clone(), likelihood)
        .with_optimizer(Fixed)
        .factor(x, n, d, y, x, n)
        .map_err(|(_, e)| e)
        .expect("sparse");
    let mut exact = Gpr::new(kernel, likelihood)
        .with_optimizer(Fixed)
        .factor(x, n, d, y)
        .map_err(|(_, e)| e)
        .expect("exact");
    let p = sparse.num_params();
    let mut params = vec![0.0; p];
    sparse.get_params(&mut params).expect("params");
    let mut g_s = vec![0.0; p];
    let mut g_e = vec![0.0; p];
    let vs = sparse
        .value_and_gradient_into(&params, &mut g_s)
        .expect("sparse vg");
    let ve = exact
        .value_and_gradient_into(&params, &mut g_e)
        .expect("exact vg");
    assert_close(vs, ve, TOL);
    assert_slice_close(&g_s, &g_e, TOL);
    let mut h_s = vec![0.0; p * p];
    let mut h_e = vec![0.0; p * p];
    sparse.hessian_into(&params, &mut h_s).expect("sparse hess");
    exact.hessian_into(&params, &mut h_e).expect("exact hess");
    assert_slice_close(&h_s, &h_e, TOL);
    let fd_g = fd_grad_from_value(&mut sparse, &params);
    assert_slice_close(&g_s, &fd_g, 1e-5);
}

#[test]
fn rbf_n4_z_eq_x_matches_exact_value_grad_hess() {
    assert_z_eq_x_matches_exact_derivs(
        KernelSpec::from(RbfKernel::new(1.0).expect("ℓ")),
        &[0.0, 1.0, 2.0, 3.0],
        4,
        1,
        &[0.0, 1.0, 0.5, 0.25],
    );
}

#[test]
fn matern_n4_z_eq_x_matches_exact_value_grad_hess() {
    assert_z_eq_x_matches_exact_derivs(
        KernelSpec::from(MaternKernel::new(1.0, MaternNu::ThreeHalves).expect("ℓ")),
        &[0.0, 1.0, 2.0, 3.0],
        4,
        1,
        &[0.0, 1.0, 0.5, 0.25],
    );
}

#[test]
fn rbf_ard_n4_z_eq_x_matches_exact_value_grad_hess() {
    assert_z_eq_x_matches_exact_derivs(
        kernel_ard(),
        &[0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        4,
        2,
        &[0.0, 1.0, 0.5, 0.25],
    );
}

#[test]
fn rbf_plus_white_n4_z_eq_x_matches_exact_value_grad_hess() {
    assert_z_eq_x_matches_exact_derivs(
        KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"))
            + KernelSpec::from(WhiteKernel::new(0.05).expect("white")),
        &[0.0, 1.0, 2.0, 3.0],
        4,
        1,
        &[0.0, 1.0, 0.5, 0.25],
    );
}

#[test]
fn rbf_n4_z_eq_x_hessian_matches_grad_fd() {
    let kernel = KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"));
    let mut sparse = factor_sparse(
        kernel,
        &[0.0, 1.0, 2.0, 3.0],
        4,
        1,
        &[0.0, 1.0, 0.5, 0.25],
        &[0.0, 1.0, 2.0, 3.0],
        4,
    );
    let p = sparse.num_params();
    let mut params = vec![0.0; p];
    sparse.get_params(&mut params).expect("params");
    let mut hess = vec![0.0; p * p];
    sparse.hessian_into(&params, &mut hess).expect("hess");
    let fd = fd_hess_from_grad(&mut sparse, &params);
    assert_slice_close(&hess, &fd, 2e-4);
}

/// A signal variance (`Constant × …`) and the leaves that had no rectangular
/// θ-derivative: at `Z = X` value, gradient, and Hessian equal Exact's (#301).
fn constant(v: f64) -> KernelSpec {
    KernelSpec::from(ConstantKernel::new(v).expect("constant"))
}

const X_2D: [f64; 8] = [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0];
const Y_4: [f64; 4] = [0.0, 1.0, 0.5, 0.25];
const X_1D: [f64; 4] = [0.0, 1.0, 2.0, 3.0];

#[test]
fn constant_times_rbf_z_eq_x_matches_exact_value_grad_hess() {
    assert_z_eq_x_matches_exact_derivs(
        constant(1.7) * KernelSpec::from(RbfKernel::new(1.3).expect("ℓ")),
        &X_1D,
        4,
        1,
        &Y_4,
    );
}

#[test]
fn constant_times_ard_z_eq_x_matches_exact_value_grad_hess() {
    assert_z_eq_x_matches_exact_derivs(constant(1.7) * kernel_ard(), &X_2D, 4, 2, &Y_4);
}

#[test]
fn constant_times_matern_ard_z_eq_x_matches_exact_value_grad_hess() {
    let matern =
        KernelSpec::from(MaternArdKernel::new(&[1.0, 1.4], MaternNu::FiveHalves).expect("ℓ"));
    assert_z_eq_x_matches_exact_derivs(constant(0.8) * matern, &X_2D, 4, 2, &Y_4);
}

#[test]
fn constant_times_rq_ard_z_eq_x_matches_exact_value_grad_hess() {
    let rq = KernelSpec::from(RationalQuadraticArdKernel::new(&[1.1, 0.9], 0.7).expect("ℓ"));
    assert_z_eq_x_matches_exact_derivs(constant(2.1) * rq, &X_2D, 4, 2, &Y_4);
}

#[test]
fn rbf_times_periodic_z_eq_x_matches_exact_value_grad_hess() {
    assert_z_eq_x_matches_exact_derivs(
        KernelSpec::from(RbfKernel::new(2.0).expect("ℓ"))
            * KernelSpec::from(PeriodicKernel::new(0.9, 1.7).expect("periodic")),
        &X_1D,
        4,
        1,
        &Y_4,
    );
}

#[test]
fn constant_times_rq_plus_linear_z_eq_x_matches_exact_value_grad_hess() {
    let rq = KernelSpec::from(RationalQuadraticKernel::new(1.1, 0.8).expect("rq"));
    let linear = KernelSpec::from(LinearKernel::new(0.6).expect("linear"));
    assert_z_eq_x_matches_exact_derivs(constant(0.6) * rq + linear, &X_1D, 4, 1, &Y_4);
}

/// A `fit` on `Constant × RBF` learns the signal variance and does not
/// raise the negative bound, with the gradient (L-BFGS) and with the Hessian (trust region).
fn assert_constant_times_rbf_fit_learns_the_variance<O>(optimizer: O)
where
    O: Clone + for<'a> Optimizer<SgprObjective<'a, O, FixedInducing>>,
{
    let x = [0.0, 0.7, 1.1, 2.0, 2.6, 3.0];
    let y = [1.1, 1.9, 1.6, 1.2, 0.7, 0.5];
    let z = [0.4, 1.5, 2.8];
    let kernel = constant(0.6) * KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"));
    let likelihood = GaussianLikelihood::new(0.1).expect("noise");
    let start = Sgpr::new(kernel.clone(), likelihood)
        .with_optimizer(Fixed)
        .factor(&x, 6, 1, &y, &z, 3)
        .map_err(|(_, e)| e)
        .expect("start");
    let mut before = vec![0.0; start.num_params()];
    start.get_params(&mut before).expect("params");
    let fitted = Sgpr::new(kernel, likelihood)
        .with_optimizer(optimizer)
        .fit(&x, 6, 1, &y, &z, 3)
        .map_err(|(_, e)| e)
        .expect("fit");
    let mut after = vec![0.0; fitted.num_params()];
    fitted.get_params(&mut after).expect("params");
    assert!(
        (after[0] - before[0]).abs() > 1e-3,
        "the signal variance did not move: {} -> {}",
        before[0],
        after[0]
    );
    assert!(
        fitted.neg_log_marginal_likelihood().expect("end")
            <= start.neg_log_marginal_likelihood().expect("start"),
    );
}

#[test]
fn constant_times_rbf_lbfgs_fit_learns_the_variance() {
    assert_constant_times_rbf_fit_learns_the_variance(Lbfgs::new());
}

#[test]
fn constant_times_rbf_newton_fit_learns_the_variance() {
    assert_constant_times_rbf_fit_learns_the_variance(TrustRegion::new());
}

/// `Z ≠ X`: the gradient of the collapsed bound is the finite difference of
/// its value, and the Hessian that of the gradient.
#[test]
fn constant_times_rbf_z_ne_x_gradient_and_hessian_match_fd() {
    let kernel = constant(1.7) * KernelSpec::from(RbfKernel::new(1.3).expect("ℓ"));
    let x = [0.0, 0.7, 1.1, 2.0, 2.6, 3.0];
    let y = [0.1, 0.9, 0.6, 0.2, -0.3, -0.5];
    let mut sparse = factor_sparse(kernel, &x, 6, 1, &y, &[0.4, 1.5, 2.8], 3);
    let p = sparse.num_params();
    let mut params = vec![0.0; p];
    sparse.get_params(&mut params).expect("params");
    let mut grad = vec![0.0; p];
    sparse
        .value_and_gradient_into(&params, &mut grad)
        .expect("grad");
    assert_slice_close(&grad, &fd_grad_from_value(&mut sparse, &params), 1e-5);
    let mut hess = vec![0.0; p * p];
    sparse.hessian_into(&params, &mut hess).expect("hess");
    assert_slice_close(&hess, &fd_hess_from_grad(&mut sparse, &params), 2e-4);
}

fn assert_fit_finishes_and_nlml_drops<O>(optimizer: O)
where
    O: Clone + for<'a> Optimizer<SgprObjective<'a, O, FixedInducing>>,
{
    let x = [0.0, 1.0, 2.0, 3.0];
    let y = [0.0, 1.0, 0.5, 0.25];
    let z = [0.5, 2.5];
    let kernel = KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"));
    let likelihood = GaussianLikelihood::new(0.1).expect("noise");
    let start = Sgpr::new(kernel.clone(), likelihood)
        .with_optimizer(Fixed)
        .factor(&x, 4, 1, &y, &z, 2)
        .map_err(|(_, e)| e)
        .expect("start")
        .neg_log_marginal_likelihood()
        .expect("start nlml");
    let fitted = Sgpr::new(kernel, likelihood)
        .with_optimizer(optimizer)
        .fit(&x, 4, 1, &y, &z, 2)
        .map_err(|(_, e)| e)
        .expect("fit");
    let end = fitted.neg_log_marginal_likelihood().expect("end nlml");
    assert!(end.is_finite(), "nlml={end}");
    assert!(end <= start, "end={end} start={start}");
}

#[test]
fn rbf_n4_m2_fit_lbfgs_drops_nlml() {
    assert_fit_finishes_and_nlml_drops(Lbfgs::new());
}

#[test]
fn fast_approx_rbf_n4_m2_fit_nlml_does_not_rise() {
    let x = [0.0, 1.0, 2.0, 3.0];
    let y = [0.0, 1.0, 0.5, 0.25];
    let z = [0.5, 2.5];
    let kernel = || KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"));
    let likelihood = || GaussianLikelihood::new(0.1).expect("noise");
    let start = Sgpr::new(kernel(), likelihood())
        .with_math(crate::KernelExp::FastApprox)
        .with_optimizer(Fixed)
        .factor(&x, 4, 1, &y, &z, 2)
        .map_err(|(_, err)| err)
        .expect("start")
        .neg_log_marginal_likelihood()
        .expect("start nlml");
    let check = |end: f64| {
        assert!(end.is_finite(), "nlml={end}");
        assert!(end <= start, "end={end} start={start}");
    };
    check(
        Sgpr::new(kernel(), likelihood())
            .with_math(crate::KernelExp::FastApprox)
            .fit(&x, 4, 1, &y, &z, 2)
            .map_err(|(_, err)| err)
            .expect("lbfgs")
            .neg_log_marginal_likelihood()
            .expect("end"),
    );
    check(
        Sgpr::new(kernel(), likelihood())
            .with_math(crate::KernelExp::FastApprox)
            .with_optimizer(NelderMead::new())
            .fit(&x, 4, 1, &y, &z, 2)
            .map_err(|(_, err)| err)
            .expect("nelder")
            .neg_log_marginal_likelihood()
            .expect("end"),
    );
    check(
        Sgpr::new(kernel(), likelihood())
            .with_math(crate::KernelExp::FastApprox)
            .with_optimizer(TrustRegion::new())
            .fit(&x, 4, 1, &y, &z, 2)
            .map_err(|(_, err)| err)
            .expect("newton")
            .neg_log_marginal_likelihood()
            .expect("end"),
    );
    check(
        Sgpr::new(kernel(), likelihood())
            .with_math(crate::KernelExp::FastApprox)
            .with_optimizer(FastSimulatedAnnealing::new())
            .fit(&x, 4, 1, &y, &z, 2)
            .map_err(|(_, err)| err)
            .expect("fsa")
            .neg_log_marginal_likelihood()
            .expect("end"),
    );
}

#[test]
fn rbf_n4_m2_fit_nelder_mead_drops_nlml() {
    assert_fit_finishes_and_nlml_drops(NelderMead::new());
}

#[test]
fn rbf_n4_m2_fit_newton_drops_nlml() {
    assert_fit_finishes_and_nlml_drops(TrustRegion::new());
}

#[test]
fn rbf_n4_m2_fit_fsa_drops_nlml() {
    assert_fit_finishes_and_nlml_drops(FastSimulatedAnnealing::new());
}

#[test]
fn is_send_sync() {
    assert_send_sync::<Sgpr>();
    assert_send_sync::<FittedSgpr>();
    assert_send_sync::<Sgpr<Lbfgs, FreeInducing>>();
    assert_send_sync::<FittedSgpr<Lbfgs, FreeInducing>>();
}

fn free_rbf_n8() -> (KernelSpec, [f64; 8], [f64; 8], [f64; 2]) {
    let x = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0];
    let y = [0.0, 0.35, 0.1, -0.4, 0.2, 0.55, -0.15, 0.3];
    (
        KernelSpec::from(RbfKernel::new(1.0).expect("ℓ")),
        x,
        y,
        [0.05, 0.12],
    )
}

fn free_fit_ok(kernel: KernelSpec, x: &[f64], n: usize, d: usize, y: &[f64], z: &[f64], m: usize) {
    let likelihood = GaussianLikelihood::new(0.1).expect("noise");
    let fitted = Sgpr::new(kernel, likelihood)
        .with_inducing(FreeInducing)
        .fit(x, n, d, y, z, m)
        .map_err(|(_, e)| e)
        .expect("free fit");
    assert!(
        fitted
            .neg_log_marginal_likelihood()
            .expect("nlml")
            .is_finite()
    );
}

#[test]
fn rbf_n8_m2_free_lbfgs_beats_fixed_factor() {
    let (kernel, x, y, z) = free_rbf_n8();
    let likelihood = GaussianLikelihood::new(0.1).expect("noise");
    let fixed = Sgpr::new(kernel.clone(), likelihood)
        .with_optimizer(Fixed)
        .factor(&x, 8, 1, &y, &z, 2)
        .map_err(|(_, e)| e)
        .expect("factor")
        .neg_log_marginal_likelihood()
        .expect("fixed nlml");
    let free = Sgpr::new(kernel, likelihood)
        .with_inducing(FreeInducing)
        .fit(&x, 8, 1, &y, &z, 2)
        .map_err(|(_, e)| e)
        .expect("free fit")
        .neg_log_marginal_likelihood()
        .expect("free nlml");
    assert!(free.is_finite(), "free={free}");
    assert!(free < fixed, "free={free} fixed={fixed}");
}

#[test]
fn matern_n8_m2_free_fit_succeeds() {
    let (_, x, y, z) = free_rbf_n8();
    free_fit_ok(
        KernelSpec::from(MaternKernel::new(1.0, MaternNu::ThreeHalves).expect("ℓ")),
        &x,
        8,
        1,
        &y,
        &z,
        2,
    );
}

#[test]
fn rbf_ard_n8_m2_free_fit_succeeds() {
    let x = [
        0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0, 0.0, 0.2, 0.1, 0.8, 0.4, 0.6, 0.3, 0.9,
    ];
    let y = [0.0, 0.35, 0.1, -0.4, 0.2, 0.55, -0.15, 0.3];
    let z = [0.05, 0.12, 0.08, 0.18];
    free_fit_ok(kernel_ard(), &x, 8, 2, &y, &z, 2);
}

#[test]
fn rbf_plus_white_n8_m2_free_fit_succeeds() {
    let (_, x, y, z) = free_rbf_n8();
    free_fit_ok(
        KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"))
            + KernelSpec::from(WhiteKernel::new(0.05).expect("white")),
        &x,
        8,
        1,
        &y,
        &z,
        2,
    );
}

fn assert_free_solver_nlml_drops<O>(optimizer: O)
where
    O: Clone + for<'a> Optimizer<SgprObjective<'a, O, FreeInducing>>,
{
    let (kernel, x, y, z) = free_rbf_n8();
    let likelihood = GaussianLikelihood::new(0.1).expect("noise");
    let start = Sgpr::new(kernel.clone(), likelihood)
        .with_optimizer(Fixed)
        .with_inducing(FreeInducing)
        .factor(&x, 8, 1, &y, &z, 2)
        .map_err(|(_, e)| e)
        .expect("start")
        .neg_log_marginal_likelihood()
        .expect("start nlml");
    let fitted = Sgpr::new(kernel, likelihood)
        .with_inducing(FreeInducing)
        .with_optimizer(optimizer)
        .fit(&x, 8, 1, &y, &z, 2)
        .map_err(|(_, e)| e)
        .expect("fit");
    let end = fitted.neg_log_marginal_likelihood().expect("end nlml");
    assert!(end.is_finite(), "nlml={end}");
    assert!(end <= start, "end={end} start={start}");
}

#[test]
fn rbf_n8_m2_free_lbfgs_drops_nlml() {
    assert_free_solver_nlml_drops(Lbfgs::new());
}

#[test]
fn rbf_n8_m2_free_nelder_mead_drops_nlml() {
    assert_free_solver_nlml_drops(NelderMead::new());
}

#[test]
fn rbf_n8_m2_free_newton_drops_nlml() {
    assert_free_solver_nlml_drops(TrustRegion::new());
}

#[test]
fn rbf_n8_m2_free_fsa_drops_nlml() {
    assert_free_solver_nlml_drops(FastSimulatedAnnealing::new());
}

fn assert_free_derivs(
    kernel: KernelSpec,
    x: &[f64],
    n: usize,
    d: usize,
    y: &[f64],
    z: &[f64],
    m: usize,
) {
    assert_free_derivs_tol(kernel, (x, n, d), y, (z, m), 2e-4);
}

/// `hess_tol` is the absolute tolerance of the Hessian against a difference of
/// gradients, whose own error grows with the curvature.
fn assert_free_derivs_tol(
    kernel: KernelSpec,
    (x, n, d): (&[f64], usize, usize),
    y: &[f64],
    (z, m): (&[f64], usize),
    hess_tol: f64,
) {
    let likelihood = GaussianLikelihood::new(0.1).expect("noise");
    let mut fitted = Sgpr::new(kernel, likelihood)
        .with_optimizer(Fixed)
        .with_inducing(FreeInducing)
        .factor(x, n, d, y, z, m)
        .map_err(|(_, e)| e)
        .expect("factor");
    let p = fitted.num_params();
    let mut params = vec![0.0; p];
    fitted.get_params(&mut params).expect("params");
    let mut grad = vec![0.0; p];
    fitted
        .value_and_gradient_into(&params, &mut grad)
        .expect("grad");
    let fd_g = fd_grad_from_value(&mut fitted, &params);
    assert_slice_close(&grad, &fd_g, 1e-5);
    let mut hess = vec![0.0; p * p];
    fitted.hessian_into(&params, &mut hess).expect("hess");
    let fd_h = fd_hess_from_grad(&mut fitted, &params);
    assert_slice_close(&hess, &fd_h, hess_tol);
}

#[test]
fn rbf_n8_m2_free_grad_hess_match_fd() {
    let (kernel, x, y, z) = free_rbf_n8();
    assert_free_derivs(kernel, &x, 8, 1, &y, &z, 2);
}

#[test]
fn matern_n8_m2_free_grad_hess_match_fd() {
    let (_, x, y, z) = free_rbf_n8();
    assert_free_derivs(
        KernelSpec::from(MaternKernel::new(1.0, MaternNu::ThreeHalves).expect("ℓ")),
        &x,
        8,
        1,
        &y,
        &z,
        2,
    );
}

#[test]
fn rbf_ard_n8_m2_free_grad_hess_match_fd() {
    let x = [
        0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0, 0.0, 0.2, 0.1, 0.8, 0.4, 0.6, 0.3, 0.9,
    ];
    let y = [0.0, 0.35, 0.1, -0.4, 0.2, 0.55, -0.15, 0.3];
    let z = [0.05, 0.12, 0.08, 0.18];
    assert_free_derivs(kernel_ard(), &x, 8, 2, &y, &z, 2);
}

#[test]
fn rbf_plus_white_n8_m2_free_grad_hess_match_fd() {
    let (_, x, y, z) = free_rbf_n8();
    assert_free_derivs(
        KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"))
            + KernelSpec::from(WhiteKernel::new(0.05).expect("white")),
        &x,
        8,
        1,
        &y,
        &z,
        2,
    );
}

const FREE_X_2D: [f64; 16] = [
    0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0, 0.0, 0.2, 0.1, 0.8, 0.4, 0.6, 0.3, 0.9,
];
const FREE_Y_8: [f64; 8] = [0.0, 0.35, 0.1, -0.4, 0.2, 0.55, -0.15, 0.3];

/// Kernels whose coordinate derivatives came with #302: every leaf that had
/// none, Matérn 5/2, and products.
fn free_kernels_1d() -> Vec<(&'static str, KernelSpec)> {
    let rbf = || KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"));
    let matern = |nu| KernelSpec::from(MaternKernel::new(1.0, nu).expect("ℓ"));
    let rq = || KernelSpec::from(RationalQuadraticKernel::new(1.1, 0.8).expect("rq"));
    let periodic = || KernelSpec::from(PeriodicKernel::new(0.9, 1.7).expect("periodic"));
    let linear = || KernelSpec::from(LinearKernel::new(0.6).expect("linear"));
    vec![
        ("matern 5/2", matern(MaternNu::FiveHalves)),
        ("rq", rq()),
        ("periodic", periodic()),
        ("linear + rbf", linear() + rbf()),
        ("constant * rbf", constant(1.7) * rbf()),
        (
            "constant * matern 5/2",
            constant(0.8) * matern(MaternNu::FiveHalves),
        ),
        ("constant * rq", constant(0.6) * rq()),
        ("rbf * periodic", rbf() * periodic()),
        ("linear * rbf", linear() * rbf()),
        (
            "constant * (rbf + matern 3/2)",
            constant(1.4) * (rbf() + matern(MaternNu::ThreeHalves)),
        ),
        (
            "(rbf + rq) * (periodic + constant)",
            (rbf() + rq()) * (periodic() + constant(0.5)),
        ),
    ]
}

fn free_kernels_2d() -> Vec<(&'static str, KernelSpec)> {
    let matern = |nu| KernelSpec::from(MaternArdKernel::new(&[1.0, 1.4], nu).expect("ℓ"));
    let rq = || KernelSpec::from(RationalQuadraticArdKernel::new(&[1.1, 0.9], 0.7).expect("ℓ"));
    vec![
        ("matern ard 3/2", matern(MaternNu::ThreeHalves)),
        ("matern ard 5/2", matern(MaternNu::FiveHalves)),
        ("rq ard", rq()),
        ("constant * rbf ard", constant(1.7) * kernel_ard()),
        (
            "constant * matern ard",
            constant(0.8) * matern(MaternNu::FiveHalves),
        ),
        ("constant * rq ard", constant(2.1) * rq()),
    ]
}

#[test]
fn free_inducing_grad_hess_match_fd_for_every_kernel() {
    let (_, x, y, z) = free_rbf_n8();
    for (name, kernel) in free_kernels_1d() {
        eprintln!("{name}");
        assert_free_derivs(kernel, &x, 8, 1, &y, &z, 2);
    }
    let z2 = [0.05, 0.12, 0.08, 0.18];
    for (name, kernel) in free_kernels_2d() {
        eprintln!("{name}");
        assert_free_derivs_tol(kernel, (&FREE_X_2D, 8, 2), &FREE_Y_8, (&z2, 2), 2e-3);
    }
}

#[test]
fn free_inducing_fits_with_every_kernel() {
    let (_, x, y, z) = free_rbf_n8();
    for (name, kernel) in free_kernels_1d() {
        eprintln!("{name}");
        free_fit_ok(kernel, &x, 8, 1, &y, &z, 2);
    }
    let z2 = [0.05, 0.12, 0.08, 0.18];
    for (name, kernel) in free_kernels_2d() {
        eprintln!("{name}");
        free_fit_ok(kernel, &FREE_X_2D, 8, 2, &FREE_Y_8, &z2, 2);
    }
}

#[test]
fn free_inducing_lbfgs_learns_a_product_kernel() {
    let (_, x, y, z) = free_rbf_n8();
    let kernel = constant(0.6) * KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"));
    let likelihood = GaussianLikelihood::new(0.1).expect("noise");
    let fixed = Sgpr::new(kernel.clone(), likelihood)
        .with_optimizer(Fixed)
        .factor(&x, 8, 1, &y, &z, 2)
        .map_err(|(_, e)| e)
        .expect("factor")
        .neg_log_marginal_likelihood()
        .expect("fixed nlml");
    let free = Sgpr::new(kernel, likelihood)
        .with_inducing(FreeInducing)
        .fit(&x, 8, 1, &y, &z, 2)
        .map_err(|(_, e)| e)
        .expect("free fit")
        .neg_log_marginal_likelihood()
        .expect("free nlml");
    assert!(free < fixed, "free={free} fixed={fixed}");
}

#[test]
fn matern_half_free_inducing_is_unsupported() {
    let (_, x, y, z) = free_rbf_n8();
    let kernel = KernelSpec::from(MaternKernel::new(1.0, MaternNu::Half).expect("ℓ"));
    let err = Sgpr::new(kernel, GaussianLikelihood::new(0.1).expect("noise"))
        .with_inducing(FreeInducing)
        .fit(&x, 8, 1, &y, &z, 2)
        .map(|_| ())
        .map_err(|(_, e)| e);
    assert_eq!(err, Err(GprError::CoordGradientUnsupported));
}

fn assert_rank1_matches_factor(got: &Rank1Vfe, want: &FittedSgpr<Fixed>) {
    assert_eq!(got.a.nrows(), want.a.nrows());
    assert_eq!(got.a.ncols(), want.a.ncols());
    for j in 0..got.a.ncols() {
        for i in 0..got.a.nrows() {
            assert_close(got.a[(i, j)], want.a[(i, j)], TOL);
        }
    }
    let m = got.b_l.nrows();
    assert_eq!(want.b_l.nrows(), m);
    for j in 0..m {
        for i in j..m {
            assert_close(
                reconstruct_llt(&got.b_l, i, j),
                reconstruct_llt(&want.b_l, i, j),
                TOL,
            );
        }
    }
    assert_eq!(got.w.len(), want.w.len());
    for i in 0..got.w.len() {
        assert_close(got.w[i], want.w[i], TOL);
    }
    assert_close(got.k_diag_sum, want.k_diag_sum, TOL);
    assert_close(got.a_frobenius2, want.a_frobenius2, TOL);
}

struct Rank1Case<'a> {
    kernel: KernelSpec,
    x: &'a [f64],
    n: usize,
    d: usize,
    y: &'a [f64],
    z: &'a [f64],
    m: usize,
    x_new: &'a [f64],
    y_new: f64,
    delete_idx: usize,
}

fn assert_rank1_insert_delete(case: Rank1Case<'_>) {
    let Rank1Case {
        kernel,
        x,
        n,
        d,
        y,
        z,
        m,
        x_new,
        y_new,
        delete_idx,
    } = case;
    let fitted = factor_sparse(kernel.clone(), x, n, d, y, z, m);
    let mut inserted = Rank1Vfe::from_fitted(&fitted);
    let mut y_ins = y.to_vec();
    rank1_insert(&mut inserted, &kernel, z, d, &mut y_ins, x_new, y_new).expect("insert");
    let x_ins = append_point(x, n, d, x_new);
    let oracle_ins = factor_sparse(kernel.clone(), &x_ins, n + 1, d, &y_ins, z, m);
    assert_rank1_matches_factor(&inserted, &oracle_ins);

    let mut deleted = Rank1Vfe::from_fitted(&fitted);
    let mut y_del = y.to_vec();
    rank1_delete(&mut deleted, &kernel, x, n, d, &mut y_del, delete_idx).expect("delete");
    let x_del = remove_point(x, n, d, delete_idx);
    let oracle_del = factor_sparse(kernel, &x_del, n - 1, d, &y_del, z, m);
    assert_rank1_matches_factor(&deleted, &oracle_del);
}

#[test]
fn chol_rank1_update_and_downdate_2x2() {
    let mut l = Mat::zeros(2, 2);
    l[(0, 0)] = 2.0;
    l[(1, 0)] = 1.0;
    l[(1, 1)] = 1.0;
    let mut v = [1.0, 0.0];
    chol_rank1_update(&mut l, &mut v);
    assert_close(reconstruct_llt(&l, 0, 0), 5.0, TOL);
    assert_close(reconstruct_llt(&l, 1, 0), 2.0, TOL);
    assert_close(reconstruct_llt(&l, 1, 1), 2.0, TOL);
    let mut back = [1.0, 0.0];
    chol_rank1_downdate(&mut l, &mut back).expect("downdate");
    assert_close(reconstruct_llt(&l, 0, 0), 4.0, TOL);
    assert_close(reconstruct_llt(&l, 1, 0), 2.0, TOL);
    assert_close(reconstruct_llt(&l, 1, 1), 2.0, TOL);
}

#[test]
fn rank1_rbf_n4_m2_matches_factor() {
    assert_rank1_insert_delete(Rank1Case {
        kernel: KernelSpec::from(RbfKernel::new(1.0).expect("ℓ")),
        x: &[0.0, 1.0, 2.0, 3.0],
        n: 4,
        d: 1,
        y: &[0.0, 1.0, 0.5, 0.25],
        z: &[0.5, 2.5],
        m: 2,
        x_new: &[4.0],
        y_new: 0.1,
        delete_idx: 1,
    });
}

#[test]
fn rank1_matern_n4_m2_matches_factor() {
    assert_rank1_insert_delete(Rank1Case {
        kernel: KernelSpec::from(MaternKernel::new(1.0, MaternNu::ThreeHalves).expect("ℓ")),
        x: &[0.0, 1.0, 2.0, 3.0],
        n: 4,
        d: 1,
        y: &[0.0, 1.0, 0.5, 0.25],
        z: &[0.5, 2.5],
        m: 2,
        x_new: &[4.0],
        y_new: 0.1,
        delete_idx: 1,
    });
}

#[test]
fn rank1_rbf_ard_n4_m2_matches_factor() {
    assert_rank1_insert_delete(Rank1Case {
        kernel: KernelSpec::from(RbfArdKernel::new(&[1.0, 1.5]).expect("ℓ")),
        x: &[0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        n: 4,
        d: 2,
        y: &[0.0, 1.0, 0.5, 0.25],
        z: &[0.25, 0.75, 0.25, 0.75],
        m: 2,
        x_new: &[0.5, 0.5],
        y_new: 0.1,
        delete_idx: 1,
    });
}

#[test]
fn rank1_rbf_plus_white_n4_m2_matches_factor() {
    assert_rank1_insert_delete(Rank1Case {
        kernel: KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"))
            + KernelSpec::from(WhiteKernel::new(0.05).expect("white")),
        x: &[0.0, 1.0, 2.0, 3.0],
        n: 4,
        d: 1,
        y: &[0.0, 1.0, 0.5, 0.25],
        z: &[0.5, 2.5],
        m: 2,
        x_new: &[4.0],
        y_new: 0.1,
        delete_idx: 1,
    });
}

fn vfe_from_fitted(fitted: &FittedSgpr<Fixed>) -> VfeState<f64> {
    VfeState::<f64> {
        k_mm_l: fitted.k_mm_l.clone(),
        a: fitted.a.clone(),
        b_l: fitted.b_l.clone(),
        w: fitted.w.clone(),
        k_diag_sum: fitted.k_diag_sum,
        a_frobenius2: fitted.a_frobenius2,
    }
}

fn assert_inducing_matches_factor(got: &VfeState<f64>, want: &FittedSgpr<Fixed>) {
    assert_eq!(got.a.nrows(), want.a.nrows());
    assert_eq!(got.a.ncols(), want.a.ncols());
    for j in 0..got.a.ncols() {
        for i in 0..got.a.nrows() {
            assert_close(got.a[(i, j)], want.a[(i, j)], TOL);
        }
    }
    let m = got.k_mm_l.nrows();
    assert_eq!(want.k_mm_l.nrows(), m);
    assert_eq!(got.b_l.nrows(), m);
    assert_eq!(want.b_l.nrows(), m);
    for j in 0..m {
        for i in j..m {
            assert_close(
                reconstruct_llt(&got.k_mm_l, i, j),
                reconstruct_llt(&want.k_mm_l, i, j),
                TOL,
            );
            assert_close(
                reconstruct_llt(&got.b_l, i, j),
                reconstruct_llt(&want.b_l, i, j),
                TOL,
            );
        }
    }
    assert_eq!(got.w.len(), want.w.len());
    for i in 0..got.w.len() {
        assert_close(got.w[i], want.w[i], TOL);
    }
    assert_close(got.k_diag_sum, want.k_diag_sum, TOL);
    assert_close(got.a_frobenius2, want.a_frobenius2, TOL);
}

struct InducingCase<'a> {
    kernel: KernelSpec,
    x: &'a [f64],
    n: usize,
    d: usize,
    y: &'a [f64],
    z: &'a [f64],
    m: usize,
    z_new: &'a [f64],
    delete_idx: usize,
}

fn assert_inducing_insert_delete(case: InducingCase<'_>) {
    let InducingCase {
        kernel,
        x,
        n,
        d,
        y,
        z,
        m,
        z_new,
        delete_idx,
    } = case;
    let fitted = factor_sparse(kernel.clone(), x, n, d, y, z, m);
    let noise = fitted.likelihood().noise_variance();
    let mut inserted = vfe_from_fitted(&fitted);
    inducing_insert::<crate::math::Accurate, _>(
        &mut inserted,
        &kernel,
        noise,
        x,
        n,
        d,
        y,
        z,
        m,
        z_new,
        &mut crate::sparse::KernelScratch::new(),
    )
    .expect("insert inducing");
    let z_ins = append_point(z, m, d, z_new);
    let oracle_ins = factor_sparse(kernel.clone(), x, n, d, y, &z_ins, m + 1);
    assert_inducing_matches_factor(&inserted, &oracle_ins);

    let mut deleted = vfe_from_fitted(&fitted);
    inducing_delete(&mut deleted, noise, y, delete_idx).expect("delete inducing");
    let z_del = remove_point(z, m, d, delete_idx);
    let oracle_del = factor_sparse(kernel, x, n, d, y, &z_del, m - 1);
    assert_inducing_matches_factor(&deleted, &oracle_del);
}

#[test]
fn inducing_rbf_n4_m2_matches_factor() {
    assert_inducing_insert_delete(InducingCase {
        kernel: KernelSpec::from(RbfKernel::new(1.0).expect("ℓ")),
        x: &[0.0, 1.0, 2.0, 3.0],
        n: 4,
        d: 1,
        y: &[0.0, 1.0, 0.5, 0.25],
        z: &[0.5, 2.5],
        m: 2,
        z_new: &[1.5],
        delete_idx: 0,
    });
}

#[test]
fn inducing_matern_n4_m2_matches_factor() {
    assert_inducing_insert_delete(InducingCase {
        kernel: KernelSpec::from(MaternKernel::new(1.0, MaternNu::ThreeHalves).expect("ℓ")),
        x: &[0.0, 1.0, 2.0, 3.0],
        n: 4,
        d: 1,
        y: &[0.0, 1.0, 0.5, 0.25],
        z: &[0.5, 2.5],
        m: 2,
        z_new: &[1.5],
        delete_idx: 0,
    });
}

#[test]
fn inducing_rbf_ard_n4_m2_matches_factor() {
    assert_inducing_insert_delete(InducingCase {
        kernel: KernelSpec::from(RbfArdKernel::new(&[1.0, 1.5]).expect("ℓ")),
        x: &[0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        n: 4,
        d: 2,
        y: &[0.0, 1.0, 0.5, 0.25],
        z: &[0.25, 0.75, 0.25, 0.75],
        m: 2,
        z_new: &[0.5, 0.5],
        delete_idx: 0,
    });
}

#[test]
fn inducing_rbf_plus_white_n4_m2_matches_factor() {
    assert_inducing_insert_delete(InducingCase {
        kernel: KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"))
            + KernelSpec::from(WhiteKernel::new(0.05).expect("white")),
        x: &[0.0, 1.0, 2.0, 3.0],
        n: 4,
        d: 1,
        y: &[0.0, 1.0, 0.5, 0.25],
        z: &[0.5, 2.5],
        m: 2,
        z_new: &[1.5],
        delete_idx: 0,
    });
}

/// The runtime `exp` mode reaches the kernel and survives `fit`,
/// `into_online`, and `into_fitted`.
#[test]
fn kernel_exp_is_a_runtime_value() {
    let x = [0.0, 1.0, 2.0, 3.0];
    let y = [0.0, 1.0, 0.5, 0.25];
    let z = [0.5, 2.5];
    let fitted = |math| {
        Sgpr::new(
            KernelSpec::from(RbfKernel::new(0.7).expect("ell")),
            GaussianLikelihood::new(0.1).expect("noise"),
        )
        .with_math(math)
        .with_optimizer(Fixed)
        .factor(&x, 4, 1, &y, &z, 2)
        .map_err(|(_, err)| err)
        .expect("factor")
    };
    let accurate = fitted(crate::KernelExp::Accurate);
    let fast = fitted(crate::KernelExp::FastApprox);
    assert_eq!(accurate.math(), crate::KernelExp::Accurate);
    assert_eq!(fast.math(), crate::KernelExp::FastApprox);
    let a = accurate.neg_log_marginal_likelihood().expect("nlml");
    let f = fast.neg_log_marginal_likelihood().expect("nlml");
    assert!((a - f).abs() < 1e-4, "accurate={a} fast={f}");
    assert!(
        a.to_bits() != f.to_bits(),
        "FastApprox must change the kernel"
    );
    let online = fast.into_online();
    assert_eq!(online.math(), crate::KernelExp::FastApprox);
    assert_eq!(online.into_fitted().math(), crate::KernelExp::FastApprox);
}

/// `Z` with a repeated point: `K_mm` is singular without jitter.
const DUP_Z: [f64; 3] = [0.5, 0.5, 2.0];
const JITTER_X: [f64; 4] = [0.0, 1.0, 2.0, 3.0];
const JITTER_Y: [f64; 4] = [0.0, 1.0, 0.5, 0.25];

fn jitter_trainer() -> Sgpr<Fixed> {
    Sgpr::new(
        KernelSpec::from(RbfKernel::new(1.0).expect("valid")),
        GaussianLikelihood::new(0.1).expect("valid"),
    )
    .with_optimizer(Fixed)
}

/// Checks `L Lᵀ = k(Z, Z) + j I` for the 1-D RBF kernel of length scale `ell`.
fn assert_k_mm_with_jitter(l: MatRef<'_, f64>, z: &[f64], ell: f64, jitter: f64) {
    let l = l.to_owned();
    for i in 0..z.len() {
        for j in 0..=i {
            let diff = z[i] - z[j];
            let mut expected = (-diff * diff / (2.0 * ell * ell)).exp();
            if i == j {
                expected += jitter;
            }
            assert_close(reconstruct_llt(&l, i, j), expected, TOL);
        }
    }
}

#[test]
fn k_mm_jitter_default_is_adaptive() {
    let expected = crate::JitterPolicy::adaptive(1e-8, 10.0, 5, 1e-3).expect("valid");
    let trainer = jitter_trainer();
    assert_eq!(trainer.jitter_policy(), expected);
    let fitted = trainer
        .factor(&JITTER_X, 4, 1, &JITTER_Y, &DUP_Z, 3)
        .map_err(|(_, e)| e)
        .expect("the default retries a singular K_mm");
    assert_eq!(fitted.jitter_policy(), expected);
    assert_eq!(fitted.clone().into_online().jitter_policy(), expected);
}

#[test]
fn k_mm_without_retry_rejects_singular_inducing_set() {
    let result = jitter_trainer()
        .with_jitter_policy(crate::JitterPolicy::default())
        .factor(&JITTER_X, 4, 1, &JITTER_Y, &DUP_Z, 3)
        .map_err(|(_, e)| e);
    assert!(matches!(result, Err(GprError::CholeskyFailed { .. })));
}

#[test]
fn k_mm_uses_the_policy_jitter_in_factor_and_set_params() {
    let cases = [
        (crate::JitterPolicy::fixed(1e-4).expect("valid"), 1e-4),
        (
            crate::JitterPolicy::adaptive(1e-6, 10.0, 5, 1e-3).expect("valid"),
            1e-6,
        ),
    ];
    for (policy, jitter) in cases {
        let mut fitted = jitter_trainer()
            .with_jitter_policy(policy)
            .factor(&JITTER_X, 4, 1, &JITTER_Y, &DUP_Z, 3)
            .map_err(|(_, e)| e)
            .expect("factor");
        assert_eq!(fitted.jitter_policy(), policy);
        assert_k_mm_with_jitter(fitted.k_mm_l(), &DUP_Z, 1.0, jitter);
        let mut params = [0.0; 2];
        fitted.get_params(&mut params).expect("params");
        params[0] = 2.0_f64.ln();
        fitted.set_params(&params).expect("set params");
        assert_k_mm_with_jitter(fitted.k_mm_l(), &DUP_Z, 2.0, jitter);
    }
}

#[test]
fn online_inducing_insert_failing_k_mm_leaves_model_unchanged() {
    let z = [0.5, 2.0];
    let xs = [0.25, 1.75];
    let mut online = jitter_trainer()
        .with_jitter_policy(crate::JitterPolicy::default())
        .factor(&JITTER_X, 4, 1, &JITTER_Y, &z, 2)
        .map_err(|(_, e)| e)
        .expect("factor")
        .into_online();
    let before = online.predict(&xs, 2, 1).expect("predict");
    let ids = online.inducing_ids().to_vec();
    assert!(matches!(
        online.insert_inducing(&[0.5]),
        Err(GprError::CholeskyFailed { .. })
    ));
    assert_eq!(online.m(), 2);
    assert_eq!(online.z(), &z);
    assert_eq!(online.inducing_ids(), ids.as_slice());
    let after = online.predict(&xs, 2, 1).expect("predict");
    assert_eq!(after.mean, before.mean);
    assert_eq!(after.variance, before.variance);

    let mut retrying = jitter_trainer()
        .factor(&JITTER_X, 4, 1, &JITTER_Y, &z, 2)
        .map_err(|(_, e)| e)
        .expect("factor")
        .into_online();
    retrying
        .insert_inducing(&[0.5])
        .expect("the default retries");
    assert_eq!(retrying.m(), 3);
}

#[test]
fn delete_by_reassembly_publishes_weights_of_the_remaining_points() {
    let x = [0.0, 0.7, 1.3, 2.0, 2.6, 3.1];
    let y = [0.2, 0.9, 0.4, -0.3, 0.1, 0.6];
    let z = [0.5, 1.5, 2.8];
    let sgpr = || {
        Sgpr::new(
            KernelSpec::from(RbfKernel::new(0.8).expect("valid")),
            GaussianLikelihood::new(0.05).expect("valid"),
        )
        .with_optimizer(Fixed)
    };
    let mut online = sgpr()
        .factor(&x, 6, 1, &y, &z, 3)
        .expect("factor")
        .into_online();
    let removed = 2;
    let x_next = remove_point(online.core.x_train.as_slice(), 6, 1, removed);
    let y_next: Vec<f64> = y
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != removed)
        .map(|(_, v)| *v)
        .collect();
    online
        .delete_by_reassembly(
            x_next.clone(),
            y_next.clone(),
            x_next.clone(),
            y_next.clone(),
        )
        .expect("reassemble");
    let rebuilt = sgpr()
        .factor(&x_next, 5, 1, &y_next, &z, 3)
        .expect("factor");
    let xs = [0.4, 1.3, 2.9];
    let got = online.predict(&xs, 3, 1).expect("predict");
    let want = rebuilt.predict(&xs, 3, 1).expect("predict");
    assert_slice_close(&got.mean, &want.mean, 1e-10);
    assert_slice_close(&got.variance, &want.variance, 1e-10);
}

/// Everything an online update could leave behind, compared bit for bit.
#[derive(Debug, PartialEq)]
struct OnlineFingerprint {
    n: usize,
    m: usize,
    points: Vec<crate::PointId>,
    inducing: Vec<InducingId>,
    mean: Vec<f64>,
    variance: Vec<f64>,
    nlml: f64,
}

fn online_fingerprint(online: &OnlineSgpr<Fixed, crate::MixedPrecision>) -> OnlineFingerprint {
    let pred = online.predict(&[0.3, 1.7, 3.2], 3, 1).expect("predict");
    OnlineFingerprint {
        n: online.n(),
        m: online.m(),
        points: online.point_ids().to_vec(),
        inducing: online.inducing_ids().to_vec(),
        mean: pred.mean,
        variance: pred.variance,
        nlml: online.neg_log_marginal_likelihood().expect("nlml"),
    }
}

#[test]
fn failed_online_updates_leave_the_model_unchanged() {
    // A refining precision is the one whose updates can fail after a write.
    let mut online = Sgpr::new(
        KernelSpec::from(RbfKernel::new(1.0).expect("valid")),
        GaussianLikelihood::new(0.1).expect("valid"),
    )
    .with_optimizer(Fixed)
    .with_precision::<crate::MixedPrecision>()
    .factor(
        &[0.0, 1.0, 2.0, 3.0],
        4,
        1,
        &[0.0, 1.0, 0.5, 0.25],
        &[0.5, 2.5],
        2,
    )
    .expect("factor")
    .into_online();
    let before = online_fingerprint(&online);
    let first_point = online.point_ids()[0];
    let first_inducing = online.inducing_ids()[0];
    // Every update succeeds, then a later step of the same call fails.
    let err = online
        .atomically(|model| {
            model.insert(&[4.0], 0.1)?;
            model.delete(first_point)?;
            model.insert_inducing(&[1.5])?;
            model.delete_inducing(first_inducing)?;
            Err::<(), _>(GprError::NonFiniteInput)
        })
        .expect_err("injected failure");
    assert!(matches!(err, GprError::NonFiniteInput));
    assert_eq!(online_fingerprint(&online), before);
    // The model is still usable and its next update takes the next id.
    let id = online.insert(&[4.0], 0.1).expect("insert");
    assert_eq!(online.n(), 5);
    assert_eq!(online.point_ids().last().copied(), Some(id));
}
