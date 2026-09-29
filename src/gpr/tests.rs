use super::{FittedGpr, Gpr};
use crate::data::pack_points;
use crate::error::{CholeskyStage, GprError};
use crate::gpr::factor::{FactorPolicy, factor_written_k_with_policy};
use crate::gpr::{
    AdaptiveJitter, CholeskyBuffer, DistanceCachePolicy, FitBuffers, FixedJitter, JitterPolicy,
    KernelExp, PredictOptions, Prediction, PredictiveCovariance, VarianceKind,
};
use crate::kernel::{
    ConstantKernel, KernelSpec, KernelTerm, LinearKernel, MaternArdKernel, MaternKernel, MaternNu,
    PeriodicKernel, RationalQuadraticArdKernel, RationalQuadraticKernel, RbfArdKernel, RbfKernel,
    Triangle, WhiteKernel,
};
use crate::likelihood::GaussianLikelihood;
use crate::linalg::{add_to_diag, cholesky_and_solve, log_det_from_l};
use crate::objective::{IncrementalObjective, Objective};
use crate::optimizer::{
    FastSimulatedAnnealing, Fixed, Lbfgs, NelderMead, Newton, NonlinearCg, OptResult, Optimizer,
};
use crate::param::Interval;
use crate::precision::DoublePrecision;
use crate::transform::{
    ColumnwiseInput, MinMaxInput, MinMaxTarget, Pipeline, StandardizeInput, StandardizeTarget,
    TargetPipeline, TargetTransform,
};
use crate::workspace::FitWorkspace;
use faer::{Mat, MatMut, MatRef};

const TOL: f64 = 1e-9;

use crate::test_check::{assert_close, assert_send_sync};

fn rbf_gpr(ell: f64, noise: f64) -> Gpr {
    Gpr::new(
        KernelSpec::from(RbfKernel::new(ell).expect("valid")),
        GaussianLikelihood::new(noise).expect("valid"),
    )
}

fn rbf_ard_gpr(ells: &[f64], noise: f64) -> Gpr {
    Gpr::new(
        KernelSpec::from(RbfArdKernel::new(ells).expect("valid")),
        GaussianLikelihood::new(noise).expect("valid"),
    )
}

fn dense_a(kernel: &KernelSpec, noise: f64, x: &[f64], n: usize, d: usize) -> Mat<f64> {
    let compiled = kernel.compile();
    let x_mat = pack_points(x, n, d);
    let mut dist = Mat::zeros(n, n);
    crate::kernel::fill_squared_euclidean(x_mat.as_ref(), dist.as_mut(), &mut []);
    let mut k = Mat::zeros(n, n);
    let mut scratch = Mat::zeros(n, n);
    compiled
        .apply::<crate::math::Accurate>(dist.as_ref(), k.as_mut(), Triangle::Full, scratch.as_mut())
        .expect("shape");
    add_to_diag(k.as_mut(), noise);
    k
}

fn copy_lower(src: faer::MatRef<'_, f64>) -> Mat<f64> {
    let n = src.nrows();
    Mat::from_fn(n, n, |i, j| if i >= j { src[(i, j)] } else { 0.0 })
}

fn matvec_sym(a: &Mat<f64>, x: &[f64]) -> Vec<f64> {
    let n = a.nrows();
    let mut out = vec![0.0; n];
    for col in 0..n {
        for row in 0..n {
            out[row] += a[(row, col)] * x[col];
        }
    }
    out
}

#[test]
fn is_send_sync() {
    assert_send_sync::<Gpr>();
    assert_send_sync::<Gpr<NonlinearCg>>();
    assert_send_sync::<Gpr<NelderMead>>();
    assert_send_sync::<FittedGpr>();
    assert_send_sync::<FittedGpr<NonlinearCg>>();
    assert_send_sync::<FittedGpr<NelderMead>>();
    assert_send_sync::<Prediction>();
    assert_send_sync::<PredictiveCovariance>();
    assert_send_sync::<VarianceKind>();
    assert_send_sync::<PredictOptions>();
    assert_send_sync::<JitterPolicy>();
    assert_send_sync::<FixedJitter>();
    assert_send_sync::<AdaptiveJitter>();
    assert_send_sync::<DistanceCachePolicy>();
    assert_send_sync::<CholeskyBuffer>();
    assert_send_sync::<KernelExp>();
    fn assert_clone<T: Clone>() {}
    assert_clone::<Gpr>();
    assert_clone::<Gpr<Fixed>>();
    assert_clone::<FittedGpr>();
    assert_clone::<FittedGpr<Fixed>>();
}

#[test]
fn fit_restores_thread_scratch() {
    let gpr = rbf_gpr(1.0, 0.1)
        .fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
        .expect("spd");
    let ws = &gpr.store.buffers.core;
    assert_eq!(ws.thread_scratch.len(), rayon::current_num_threads().max(1));
    assert!(
        ws.thread_scratch
            .iter()
            .all(|m| m.nrows() == 0 && m.ncols() == 0)
    );
}

#[test]
fn predict_restores_thread_scratch_when_apply_cross_fails() {
    let mut gpr = rbf_gpr(1.0, 0.1)
        .fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
        .expect("spd");
    gpr.core.query.ensure(2, 1, 1).expect("query");
    gpr.core.query.query_scratch = Mat::<f64>::zeros(1, 1);
    assert!(matches!(
        gpr.predict_into(&[0.5], 1, 1, &mut Prediction::default()),
        Err(GprError::WorkspaceTooSmall)
    ));
    let ws = &gpr.store.buffers.core;
    assert_eq!(ws.thread_scratch.len(), rayon::current_num_threads().max(1));
    assert!(
        ws.thread_scratch
            .iter()
            .all(|m| m.nrows() == 0 && m.ncols() == 0)
    );
}

#[test]
fn predict_into_matches_predict() {
    let mut gpr = rbf_gpr(1.0, 0.1)
        .fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
        .expect("spd");
    let owned = gpr.predict(&[0.5], 1, 1).expect("fitted");
    let mut into = Prediction::default();
    gpr.predict_into(&[0.5], 1, 1, &mut into).expect("fitted");
    assert_eq!(into.mean, owned.mean);
    assert_eq!(into.variance, owned.variance);
    assert_eq!(into.variance_kind, owned.variance_kind);
    let ws = &gpr.store.buffers.core;
    assert_eq!(ws.thread_scratch.len(), rayon::current_num_threads().max(1));
    assert!(
        ws.thread_scratch
            .iter()
            .all(|m| m.nrows() == 0 && m.ncols() == 0)
    );
}

fn cov_diag(cov: &PredictiveCovariance) -> Vec<f64> {
    let m = cov.mean.len();
    (0..m).map(|i| cov.covariance[i * m + i]).collect()
}

#[test]
fn predict_covariance_diagonal_matches_predict() {
    let gpr = rbf_gpr(1.0, 0.1)
        .with_optimizer(Fixed)
        .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
        .expect("spd");
    let xs = [0.25, 0.75];
    let pred = gpr.predict(&xs, 2, 1).expect("fitted");
    let cov = gpr.predict_covariance(&xs, 2, 1).expect("fitted");
    assert_eq!(cov.mean, pred.mean);
    assert_eq!(cov.variance_kind, pred.variance_kind);
    let diag = cov_diag(&cov);
    for (d, v) in diag.iter().zip(pred.variance.iter()) {
        assert_close(*d, *v, TOL);
    }
    assert_close(cov.covariance[1], cov.covariance[2], TOL);
    let lat = gpr
        .predict_covariance_with(
            &xs,
            2,
            1,
            PredictOptions {
                variance_kind: VarianceKind::Latent,
            },
        )
        .expect("fitted");
    let lat_pred = gpr
        .predict_with(
            &xs,
            2,
            1,
            PredictOptions {
                variance_kind: VarianceKind::Latent,
            },
        )
        .expect("fitted");
    for (d, v) in cov_diag(&lat).iter().zip(lat_pred.variance.iter()) {
        assert_close(*d, *v, TOL);
    }
    assert_close(cov.covariance[0], lat.covariance[0] + 0.1, TOL);
    assert_close(cov.covariance[3], lat.covariance[3] + 0.1, TOL);
    assert_close(cov.covariance[1], lat.covariance[1], TOL);
}

#[test]
fn predict_covariance_n1_matches_predict() {
    let gpr = rbf_gpr(1.0, 0.1)
        .with_optimizer(Fixed)
        .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
        .expect("spd");
    let pred = gpr.predict(&[0.5], 1, 1).expect("fitted");
    let cov = gpr.predict_covariance(&[0.5], 1, 1).expect("fitted");
    assert_eq!(cov.mean, pred.mean);
    assert_eq!(cov.covariance.len(), 1);
    assert_close(cov.covariance[0], pred.variance[0], TOL);
}

#[test]
fn predict_covariance_standardize_diagonal_matches_predict() {
    let y = [0.0, 4.0];
    let gpr = rbf_gpr(1.0, 0.16)
        .with_target_transform(StandardizeTarget::new())
        .with_optimizer(Fixed)
        .factor(&[0.0, 1.0], 2, 1, &y)
        .expect("spd");
    let xs = [0.25, 0.75];
    let pred = gpr.predict(&xs, 2, 1).expect("fitted");
    let cov = gpr.predict_covariance(&xs, 2, 1).expect("fitted");
    assert_eq!(cov.mean, pred.mean);
    for (d, v) in cov_diag(&cov).iter().zip(pred.variance.iter()) {
        assert_close(*d, *v, TOL);
    }
}

#[test]
fn predict_covariance_ard_diagonal_matches_predict() {
    let x = [0.0, 0.5, 1.5, 0.0, 1.0, 0.5];
    let y = [0.2, -1.0, 0.7];
    let xs = [0.25, 1.0, 0.75, 0.5];
    let gpr = rbf_ard_gpr(&[1.25, 0.8], 0.1)
        .with_optimizer(Fixed)
        .factor(&x, 3, 2, &y)
        .expect("spd");
    let pred = gpr.predict(&xs, 2, 2).expect("fitted");
    let cov = gpr.predict_covariance(&xs, 2, 2).expect("fitted");
    assert_eq!(cov.mean, pred.mean);
    for (d, v) in cov_diag(&cov).iter().zip(pred.variance.iter()) {
        assert_close(*d, *v, TOL);
    }
}

#[test]
fn sample_is_deterministic_for_seed() {
    let gpr = rbf_gpr(1.0, 0.1)
        .with_optimizer(Fixed)
        .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
        .expect("spd");
    let xs = [0.25, 0.75];
    let a = gpr.sample(&xs, 2, 1, 3, 7).expect("fitted");
    let b = gpr.sample(&xs, 2, 1, 3, 7).expect("fitted");
    assert_eq!(a, b);
    assert_eq!(a.len(), 6);
    let c = gpr.sample(&xs, 2, 1, 3, 8).expect("fitted");
    assert_ne!(a, c);
    let empty = gpr.sample(&xs, 2, 1, 0, 7).expect("fitted");
    assert!(empty.is_empty());
}

#[test]
fn fit_solves_a_alpha_equals_y() {
    let x = [0.0, 0.5, 1.5, 0.0, 1.0, 0.5];
    let y = [0.2, -1.0, 0.7];
    let gpr = rbf_gpr(1.25, 0.1)
        .with_optimizer(Fixed)
        .factor(&x, 3, 2, &y)
        .expect("spd");
    assert_eq!(gpr.n(), 3);
    assert_eq!(gpr.d(), 2);
    let a = dense_a(gpr.kernel(), gpr.likelihood().noise_variance(), &x, 3, 2);
    let alpha = gpr.alpha();
    let restored = matvec_sym(&a, alpha);
    for i in 0..3 {
        assert_close(restored[i], y[i], TOL);
    }
    let ws = &gpr.store.buffers.core;
    let l = copy_lower(ws.k_matrix.as_ref());
    let a_from_l = &l * l.transpose();
    for col in 0..3 {
        for row in col..3 {
            assert_close(a_from_l[(row, col)], a[(row, col)], TOL);
        }
    }
}

#[test]
fn refit_replaces_size_and_still_solves() {
    let gpr = rbf_gpr(1.25, 0.1)
        .with_optimizer(Fixed)
        .factor(&[0.0, 0.5, 1.5, 0.0, 1.0, 0.5], 3, 2, &[0.2, -1.0, 0.7])
        .expect("spd");
    let gpr = gpr
        .into_trainer()
        .with_optimizer(Fixed)
        .factor(&[0.0, 1.0], 2, 1, &[0.5, -0.25])
        .expect("refit");
    assert_eq!(gpr.n(), 2);
    assert_eq!(gpr.d(), 1);
    let a = dense_a(
        gpr.kernel(),
        gpr.likelihood().noise_variance(),
        &[0.0, 1.0],
        2,
        1,
    );
    let alpha = gpr.alpha();
    let restored = matvec_sym(&a, alpha);
    assert_close(restored[0], 0.5, TOL);
    assert_close(restored[1], -0.25, TOL);
}

#[test]
fn fit_n_one_matches_scalar_solve() {
    let noise = 0.25;
    let gpr = rbf_gpr(1.0, noise)
        .with_optimizer(Fixed)
        .factor(&[0.0], 1, 1, &[2.0])
        .expect("spd");
    let a = 1.0 + noise;
    assert_close(gpr.alpha()[0], 2.0 / a, TOL);
}

#[test]
fn validation_error_does_not_yield_fitted_model() {
    let gpr = rbf_gpr(1.0, 0.1)
        .with_optimizer(Fixed)
        .factor(&[0.0, 1.0], 2, 1, &[1.0, 2.0])
        .expect("spd");
    let alpha = gpr.alpha().to_vec();
    assert!(matches!(
        gpr.into_trainer().factor(&[0.0], 0, 1, &[]),
        Err((_, GprError::EmptyInput))
    ));
    let gpr = rbf_gpr(1.0, 0.1)
        .with_optimizer(Fixed)
        .factor(&[0.0, 1.0], 2, 1, &[1.0, 2.0])
        .expect("spd");
    assert_close(gpr.alpha()[0], alpha[0], TOL);
    assert_close(gpr.alpha()[1], alpha[1], TOL);
}

#[test]
fn indefinite_matrix_returns_cholesky_failed() {
    let mut a = faer::mat![[1.0, 2.0], [2.0, 1.0]];
    let mut rhs = faer::mat![[1.0], [0.0]];
    let mut ws =
        FitBuffers::<DoublePrecision>::new(2, DistanceCachePolicy::Cached, CholeskyBuffer::Retain)
            .expect("n > 0");
    let err = cholesky_and_solve(
        &mut a,
        &mut rhs,
        &mut ws.core.faer_scratch,
        0.0,
        CholeskyStage::Fit,
    )
    .expect_err("indefinite");
    assert!(matches!(
        err,
        GprError::CholeskyFailed {
            stage: CholeskyStage::Fit,
            matrix_size: 2,
            jitter: 0.0,
        }
    ));
}

#[derive(Clone, Debug)]
struct IndefiniteLeaf;

impl<T: crate::kernel::KernelScalar> KernelTerm<T> for IndefiniteLeaf {
    fn num_params(&self) -> usize {
        0
    }

    fn get_params(&self, out: &mut [f64]) -> Result<(), GprError> {
        if out.is_empty() {
            Ok(())
        } else {
            Err(GprError::IndexOutOfRange {
                reason: "indefinite leaf has no parameters".to_owned(),
            })
        }
    }

    fn set_params(&mut self, params: &[f64]) -> Result<(), GprError> {
        KernelTerm::<T>::get_params(self, &mut params.to_vec())
    }

    fn bounds_into(&self, out: &mut [Interval]) -> Result<(), GprError> {
        if out.is_empty() {
            Ok(())
        } else {
            Err(GprError::IndexOutOfRange {
                reason: "indefinite leaf has no parameters".to_owned(),
            })
        }
    }

    fn apply(
        &self,
        dist: MatRef<'_, T>,
        mut out: MatMut<'_, T>,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        let n = dist.nrows();
        if n == 0 || dist.ncols() != n || out.nrows() != n || out.ncols() != n {
            return Err(GprError::ShapeMismatch {
                reason: "indefinite leaf needs matching square matrices".to_owned(),
            });
        }
        for col in 0..n {
            let start = match uplo {
                Triangle::Lower => col,
                Triangle::Upper | Triangle::Full => 0,
            };
            let end = match uplo {
                Triangle::Upper => col + 1,
                Triangle::Lower | Triangle::Full => n,
            };
            for row in start..end {
                out[(row, col)] = if row == col {
                    T::from_f64(1.0)
                } else {
                    T::from_f64(2.0)
                };
            }
        }
        Ok(())
    }

    fn apply_cross(&self, dist: MatRef<'_, T>, mut out: MatMut<'_, T>) -> Result<(), GprError> {
        for col in 0..out.ncols() {
            for row in 0..out.nrows() {
                out[(row, col)] = if row == col {
                    T::from_f64(1.0)
                } else {
                    T::from_f64(2.0)
                };
            }
        }
        let _ = dist;
        Ok(())
    }

    fn fill_diag(&self, out: &mut [T]) -> Result<(), GprError> {
        out.fill(T::from_f64(1.0));
        Ok(())
    }

    fn grad(
        &self,
        _dist: MatRef<'_, T>,
        _d_k: MatMut<'_, T>,
        param_idx: usize,
        _uplo: Triangle,
    ) -> Result<(), GprError> {
        Err(GprError::IndexOutOfRange {
            reason: format!("indefinite leaf has no parameter {param_idx}"),
        })
    }

    fn hess(
        &self,
        _dist: MatRef<'_, T>,
        _d2_k: MatMut<'_, T>,
        i: usize,
        j: usize,
        _uplo: Triangle,
    ) -> Result<(), GprError> {
        Err(GprError::IndexOutOfRange {
            reason: format!("indefinite leaf has no parameter pair ({i}, {j})"),
        })
    }

    fn hess_points(
        &self,
        _x: MatRef<'_, T>,
        _d2_k: MatMut<'_, T>,
        i: usize,
        j: usize,
        _uplo: Triangle,
    ) -> Result<(), GprError> {
        Err(GprError::IndexOutOfRange {
            reason: format!("indefinite leaf has no parameter pair ({i}, {j})"),
        })
    }

    fn clone_box(&self) -> Box<dyn KernelTerm<T>> {
        Box::new(self.clone())
    }
}

fn indefinite_gpr(noise: f64) -> Gpr {
    Gpr::new(
        KernelSpec::custom(IndefiniteLeaf),
        GaussianLikelihood::new(noise).expect("valid"),
    )
}

#[test]
fn jitter_constructors_reject_invalid_values() {
    assert!(JitterPolicy::fixed(-1e-8).is_err());
    assert!(JitterPolicy::fixed(f64::NAN).is_err());
    assert!(JitterPolicy::adaptive(0.0, 10.0, 3, 1.0).is_err());
    assert!(JitterPolicy::adaptive(1e-8, 1.0, 3, 1.0).is_err());
    assert!(JitterPolicy::adaptive(1e-8, 10.0, 0, 1.0).is_err());
    assert!(JitterPolicy::adaptive(1.0, 10.0, 3, 0.5).is_err());
}

#[test]
fn default_jitter_policy_reports_zero_on_failure() {
    let err = indefinite_gpr(0.1)
        .with_optimizer(Fixed)
        .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
        .expect_err("indefinite")
        .1;
    assert!(matches!(
        err,
        GprError::CholeskyFailed {
            jitter: 0.0,
            matrix_size: 2,
            stage: CholeskyStage::Fit,
        }
    ));
}

#[test]
fn fixed_jitter_recovers_without_changing_noise() {
    let noise = 0.1;
    let fitted = indefinite_gpr(noise)
        .with_jitter_policy(JitterPolicy::fixed(1.0).expect("valid"))
        .with_optimizer(Fixed)
        .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
        .expect("A + j I is spd");
    assert_close(fitted.likelihood().noise_variance(), noise, TOL);
    assert_eq!(fitted.alpha().len(), 2);
    assert!(fitted.alpha().iter().all(|a| a.is_finite()));
}

#[test]
fn adaptive_jitter_recovers_after_growth() {
    let fitted = indefinite_gpr(0.1)
        .with_jitter_policy(JitterPolicy::adaptive(0.1, 10.0, 3, 10.0).expect("valid"))
        .with_optimizer(Fixed)
        .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
        .expect("j grows past the negative eigenvalue");
    assert_close(fitted.likelihood().noise_variance(), 0.1, TOL);
}

#[test]
fn factor_retry_does_not_accumulate_into_failed_cholesky() {
    let n = 2;
    let mut ws =
        FitBuffers::<DoublePrecision>::new(n, DistanceCachePolicy::Cached, CholeskyBuffer::Retain)
            .expect("n > 0");
    let y = [1.0, 0.0];
    let noise = 0.1;
    let jitter = 1.0;
    factor_written_k_with_policy(
        &mut ws,
        &y,
        noise,
        FactorPolicy {
            jitter: JitterPolicy::fixed(jitter).expect("valid"),
            stage: CholeskyStage::Fit,
        },
        |ws| {
            let mut k = ws.core_mut().k_matrix.as_mut();
            k[(0, 0)] += 1.0;
            k[(1, 0)] += 2.0;
            k[(1, 1)] += 1.0;
            Ok(())
        },
    )
    .expect("jitter recovers an indefinite Gram");
    let l = ws.core().k_matrix.as_ref();
    let diag = noise + jitter;
    let expected = [[1.0 + diag, 2.0], [2.0, 1.0 + diag]];
    for col in 0..n {
        for row in col..n {
            let mut a = 0.0;
            for k in 0..=row.min(col) {
                a += l[(row, k)] * l[(col, k)];
            }
            assert_close(a, expected[row][col], TOL);
        }
    }
}

#[test]
fn adaptive_jitter_reports_last_attempt_when_capped() {
    let err = indefinite_gpr(0.1)
        .with_jitter_policy(JitterPolicy::adaptive(0.1, 10.0, 5, 0.5).expect("valid"))
        .with_optimizer(Fixed)
        .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
        .expect_err("max_jitter too small")
        .1;
    assert!(matches!(
        err,
        GprError::CholeskyFailed {
            jitter: j,
            matrix_size: 2,
            stage: CholeskyStage::Fit,
        } if (j - 0.1).abs() <= 1e-18
    ));
}

#[test]
fn set_params_refactors_and_training_xy_roundtrip_through_factor() {
    let x = [0.0, 1.0];
    let y = [0.25, -0.5];
    let mut fitted = rbf_gpr(1.0, 0.1)
        .with_optimizer(Fixed)
        .factor(&x, 2, 1, &y)
        .expect("spd");
    assert_eq!(fitted.x(), x.as_slice());
    assert_eq!(fitted.y(), y.as_slice());
    let mut params = [0.0; 2];
    fitted.get_params(&mut params).expect("len 2");
    params[0] = 2.0_f64.ln();
    fitted.set_params(&params).expect("spd at new theta");
    let mut got = [0.0; 2];
    fitted.get_params(&mut got).expect("len 2");
    assert_close(got[0], params[0], TOL);
    assert_close(got[1], params[1], TOL);
    let pred = fitted.predict(&[0.5], 1, 1).expect("fitted");
    assert!(pred.mean[0].is_finite());

    let n = fitted.n();
    let d = fitted.d();
    let x_obs = fitted.x().to_vec();
    let y_obs = fitted.y().to_vec();
    let alpha = fitted.alpha().to_vec();
    let rebuilt = fitted
        .into_trainer()
        .with_optimizer(Fixed)
        .factor(&x_obs, n, d, &y_obs)
        .expect("same observations");
    assert_eq!(rebuilt.alpha().len(), alpha.len());
    for (a, b) in rebuilt.alpha().iter().zip(alpha.iter()) {
        assert_close(*a, *b, TOL);
    }
}

#[test]
fn set_params_rejects_wrong_length_without_changing_theta() {
    let mut fitted = rbf_gpr(1.0, 0.1)
        .with_optimizer(Fixed)
        .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
        .expect("spd");
    let mut before = [0.0; 2];
    fitted.get_params(&mut before).expect("len 2");
    assert!(fitted.set_params(&[0.0]).is_err());
    let mut after = [0.0; 2];
    fitted.get_params(&mut after).expect("len 2");
    assert_close(before[0], after[0], TOL);
    assert_close(before[1], after[1], TOL);
}

#[test]
fn set_params_cholesky_failure_leaves_theta_and_factorization() {
    let mut fitted = Gpr::new(
        KernelSpec::from(RbfKernel::new(1.0).expect("valid")),
        GaussianLikelihood::new(0.1)
            .expect("valid")
            .with_bounds(Interval::new(1e-30, 1e5).expect("open"))
            .expect("inside"),
    )
    .with_optimizer(Fixed)
    .factor(&[0.0, 0.0], 2, 1, &[0.5, -0.25])
    .expect("spd");
    let snapshot = fitted.clone();
    let mut bad = [0.0; 2];
    fitted.get_params(&mut bad).expect("len 2");
    bad[0] = 0.5;
    bad[1] = (1e-20_f64).ln();
    assert!(matches!(
        fitted.set_params(&bad),
        Err(GprError::CholeskyFailed { .. })
    ));
    let mut after = [0.0; 2];
    fitted.get_params(&mut after).expect("len 2");
    let mut before = [0.0; 2];
    snapshot.get_params(&mut before).expect("len 2");
    assert_close(after[0], before[0], TOL);
    assert_close(after[1], before[1], TOL);
    let pred = fitted
        .predict(&[0.0], 1, 1)
        .expect("usable after failed set_params");
    let pred0 = snapshot.predict(&[0.0], 1, 1).expect("snapshot predict");
    assert_close(pred.mean[0], pred0.mean[0], TOL);
    assert_close(pred.variance[0], pred0.variance[0], TOL);
    assert_close(
        fitted.neg_log_marginal_likelihood().expect("nlml"),
        snapshot
            .neg_log_marginal_likelihood()
            .expect("snapshot nlml"),
        TOL,
    );
    assert_eq!(fitted.alpha().len(), snapshot.alpha().len());
    for (a, b) in fitted.alpha().iter().zip(snapshot.alpha()) {
        assert_close(*a, *b, TOL);
    }
}

#[test]
fn clone_preserves_trainer_and_fitted_predict() {
    let trainer = rbf_gpr(1.25, 0.1);
    let trainer_clone = trainer.clone();
    let fitted = trainer
        .with_optimizer(Fixed)
        .factor(&[0.0, 1.0], 2, 1, &[0.5, -0.25])
        .expect("spd");
    let fitted_clone = fitted.clone();
    let p1 = fitted.predict(&[0.25], 1, 1).expect("fitted");
    let p2 = fitted_clone.predict(&[0.25], 1, 1).expect("clone");
    assert_close(p1.mean[0], p2.mean[0], TOL);
    assert_close(p1.variance[0], p2.variance[0], TOL);
    let other = trainer_clone
        .with_optimizer(Fixed)
        .factor(&[0.0, 1.0], 2, 1, &[0.5, -0.25])
        .expect("spd");
    let p3 = other.predict(&[0.25], 1, 1).expect("cloned trainer");
    assert_close(p1.mean[0], p3.mean[0], TOL);
}

#[test]
fn training_y_is_original_scale_with_standardize_target() {
    let y = [1.0, 3.0];
    let fitted = rbf_gpr(1.0, 0.1)
        .with_target_transform(StandardizeTarget::new())
        .with_optimizer(Fixed)
        .factor(&[0.0, 1.0], 2, 1, &y)
        .expect("spd");
    assert_eq!(fitted.y(), y.as_slice());
    let x_obs = fitted.x().to_vec();
    let y_obs = fitted.y().to_vec();
    let n = fitted.n();
    let d = fitted.d();
    let pred = fitted.predict(&[0.5], 1, 1).expect("fitted");
    let rebuilt = fitted
        .into_trainer()
        .with_optimizer(Fixed)
        .factor(&x_obs, n, d, &y_obs)
        .expect("roundtrip");
    let pred2 = rebuilt.predict(&[0.5], 1, 1).expect("rebuilt");
    assert_close(pred.mean[0], pred2.mean[0], TOL);
    assert_close(pred.variance[0], pred2.variance[0], TOL);
}

#[test]
fn minmax_input_fit_predicts() {
    let gpr = rbf_gpr(1.0, 0.1)
        .with_input_transform(MinMaxInput::new())
        .with_optimizer(Fixed)
        .factor(&[0.0, 10.0], 2, 1, &[0.0, 1.0])
        .expect("spd");
    let pred = gpr.predict(&[5.0], 1, 1).expect("fitted");
    assert!(pred.mean[0].is_finite());
    assert!(pred.variance[0].is_finite());
    assert!(pred.variance[0] >= 0.0);
}

#[test]
fn one_step_target_pipeline_matches_direct_map() {
    let y = [1.0, 3.0];
    let direct = rbf_gpr(1.0, 0.1)
        .with_target_transform(StandardizeTarget::new())
        .with_optimizer(Fixed)
        .factor(&[0.0, 1.0], 2, 1, &y)
        .expect("spd");
    let via = rbf_gpr(1.0, 0.1)
        .with_target_transform(TargetPipeline::new().then(StandardizeTarget::new()))
        .with_optimizer(Fixed)
        .factor(&[0.0, 1.0], 2, 1, &y)
        .expect("spd");
    let p1 = direct.predict(&[0.5], 1, 1).expect("fitted");
    let p2 = via.predict(&[0.5], 1, 1).expect("fitted");
    assert_close(p1.mean[0], p2.mean[0], TOL);
    assert_close(p1.variance[0], p2.variance[0], TOL);
}

#[test]
fn stacked_transforms_fit_predict_and_roundtrip() {
    let x = [0.0, 10.0];
    let y = [1.0, 5.0];
    let fitted = rbf_gpr(1.0, 0.1)
        .with_input_transform(
            Pipeline::new()
                .then(MinMaxInput::new())
                .then(StandardizeInput::new()),
        )
        .with_target_transform(
            TargetPipeline::new()
                .then(MinMaxTarget::new())
                .then(StandardizeTarget::new()),
        )
        .with_optimizer(Fixed)
        .factor(&x, 2, 1, &y)
        .expect("spd");
    let pred = fitted.predict(&[5.0], 1, 1).expect("fitted");
    assert!(pred.mean[0].is_finite());
    assert!(pred.variance[0].is_finite());
    assert!(pred.variance[0] >= 0.0);
    let x_obs = fitted.x().to_vec();
    let y_obs = fitted.y().to_vec();
    let n = fitted.n();
    let d = fitted.d();
    let rebuilt = fitted
        .into_trainer()
        .with_optimizer(Fixed)
        .factor(&x_obs, n, d, &y_obs)
        .expect("roundtrip");
    let pred2 = rebuilt.predict(&[5.0], 1, 1).expect("rebuilt");
    assert_close(pred.mean[0], pred2.mean[0], TOL);
    assert_close(pred.variance[0], pred2.variance[0], TOL);
}

#[test]
fn columnwise_input_minmax_and_standardize() {
    let x = [0.0, 1.0, 2.0, 3.0, 1.0, 2.0, 1.5, 2.5];
    let y = [0.0, 1.0, 0.5, 1.5];
    let fitted = rbf_gpr(1.0, 0.1)
        .with_input_transform(
            ColumnwiseInput::new()
                .then(MinMaxInput::new())
                .then(StandardizeInput::new()),
        )
        .with_optimizer(Fixed)
        .factor(&x, 4, 2, &y)
        .expect("spd");
    let pred = fitted.predict(&[1.5, 1.75], 1, 2).expect("fitted");
    assert!(pred.mean[0].is_finite());
    assert!(pred.variance[0].is_finite());
    assert!(pred.variance[0] >= 0.0);
    let x_obs = fitted.x().to_vec();
    let y_obs = fitted.y().to_vec();
    let n = fitted.n();
    let d = fitted.d();
    let rebuilt = fitted
        .into_trainer()
        .with_optimizer(Fixed)
        .factor(&x_obs, n, d, &y_obs)
        .expect("roundtrip");
    let pred2 = rebuilt.predict(&[1.5, 1.75], 1, 2).expect("rebuilt");
    assert_close(pred.mean[0], pred2.mean[0], TOL);
    assert_close(pred.variance[0], pred2.variance[0], TOL);
}

#[test]
fn columnwise_input_rejects_length_mismatch() {
    assert!(matches!(
        rbf_gpr(1.0, 0.1)
            .with_input_transform(ColumnwiseInput::new().then(MinMaxInput::new()))
            .with_optimizer(Fixed)
            .factor(&[0.0, 1.0, 2.0, 3.0], 2, 2, &[0.0, 1.0]),
        Err((
            _,
            GprError::DimensionMismatch {
                x_dim: 2,
                expected_dim: 1
            }
        ))
    ));
}

#[test]
fn fit_rejects_bad_shapes_and_non_finite() {
    assert!(matches!(
        rbf_gpr(1.0, 0.1).fit(&[0.0], 2, 1, &[0.0, 1.0]),
        Err((_, GprError::LengthMismatch { .. }))
    ));
    assert!(matches!(
        rbf_gpr(1.0, 0.1).fit(&[0.0, 1.0], 2, 1, &[0.0]),
        Err((_, GprError::LengthMismatch { .. }))
    ));
    assert!(matches!(
        rbf_gpr(1.0, 0.1).fit(&[0.0, f64::NAN], 2, 1, &[0.0, 1.0]),
        Err((_, GprError::NonFiniteInput))
    ));
}

#[test]
fn neg_mll_n_one_matches_closed_form() {
    let noise = 0.25;
    let y = 2.0;
    let gpr = rbf_gpr(1.0, noise)
        .with_optimizer(Fixed)
        .factor(&[0.0], 1, 1, &[y])
        .expect("spd");
    let a = 1.0 + noise;
    let log_det = a.ln();
    let ws = &gpr.store.buffers.core;
    assert_close(log_det_from_l(ws.k_matrix.as_ref(), 1), log_det, TOL);
    let quad = y * y / a;
    let expected = 0.5 * (quad + log_det + (2.0 * std::f64::consts::PI).ln());
    assert_close(
        gpr.neg_log_marginal_likelihood().expect("fitted"),
        expected,
        TOL,
    );
}

#[test]
fn neg_mll_n_two_matches_analytic_det_and_quad() {
    let ell = 1.0;
    let noise = 0.1;
    let x = [0.0, 1.0];
    let y = [0.5, -0.25];
    let gpr = rbf_gpr(ell, noise)
        .with_optimizer(Fixed)
        .factor(&x, 2, 1, &y)
        .expect("spd");
    let k01 = (-0.5 * (1.0 / ell) * (1.0 / ell)).exp();
    let diag = 1.0 + noise;
    let det = diag * diag - k01 * k01;
    let log_det = det.ln();
    let ws = &gpr.store.buffers.core;
    assert_close(log_det_from_l(ws.k_matrix.as_ref(), 2), log_det, TOL);
    let inv_scale = 1.0 / det;
    let quad = inv_scale * (y[0] * (diag * y[0] - k01 * y[1]) + y[1] * (-k01 * y[0] + diag * y[1]));
    let expected = 0.5 * (quad + log_det + 2.0 * (2.0 * std::f64::consts::PI).ln());
    assert_close(
        gpr.neg_log_marginal_likelihood().expect("fitted"),
        expected,
        TOL,
    );
}

#[test]
fn neg_mll_uses_transformed_targets() {
    let noise = 0.16;
    let y = [0.0, 4.0];
    let gpr = rbf_gpr(1.0, noise)
        .with_target_transform(StandardizeTarget::new())
        .with_optimizer(Fixed)
        .factor(&[0.0, 1.0], 2, 1, &y)
        .expect("spd");
    let t = StandardizeTarget::new().fit(&y).expect("finite");
    let mut y_t = y;
    t.transform(&mut y_t).expect("fitted");
    let k01 = (-0.5_f64).exp();
    let diag = 1.0 + noise;
    let det = diag * diag - k01 * k01;
    let log_det = det.ln();
    let inv_scale = 1.0 / det;
    let quad = inv_scale
        * (y_t[0] * (diag * y_t[0] - k01 * y_t[1]) + y_t[1] * (-k01 * y_t[0] + diag * y_t[1]));
    let expected = 0.5 * (quad + log_det + 2.0 * (2.0 * std::f64::consts::PI).ln());
    assert_close(
        gpr.neg_log_marginal_likelihood().expect("fitted"),
        expected,
        TOL,
    );
    let raw = 0.5
        * (inv_scale * (y[0] * (diag * y[0] - k01 * y[1]) + y[1] * (-k01 * y[0] + diag * y[1]))
            + log_det
            + 2.0 * (2.0 * std::f64::consts::PI).ln());
    assert!((gpr.neg_log_marginal_likelihood().expect("fitted") - raw).abs() > TOL);
}

#[test]
fn value_and_gradient_rejects_bad_len() {
    let mut gpr = rbf_gpr(1.0, 0.1)
        .with_optimizer(Fixed)
        .factor(&[0.0, 1.0], 2, 1, &[0.5, -0.25])
        .expect("spd");
    let mut grad = [0.0, 0.0];
    assert!(matches!(
        gpr.value_and_gradient_into(&[0.0], &mut grad),
        Err(GprError::LengthMismatch { .. })
    ));
    assert!(matches!(
        gpr.get_params(&mut [0.0]),
        Err(GprError::LengthMismatch { .. })
    ));
}

#[test]
fn value_and_gradient_set_params_is_atomic() {
    let mut gpr = rbf_gpr(1.0, 0.1)
        .with_optimizer(Fixed)
        .factor(&[0.0, 1.0], 2, 1, &[0.5, -0.25])
        .expect("spd");
    let mut before = [0.0; 2];
    gpr.get_params(&mut before).expect("len 2");
    let mut bad = before;
    bad[0] = 0.5;
    bad[1] = f64::INFINITY;
    let mut grad = [0.0; 2];
    assert!(matches!(
        gpr.value_and_gradient_into(&bad, &mut grad),
        Err(GprError::InvalidNoiseVariance { .. })
    ));
    let mut after = [0.0; 2];
    gpr.get_params(&mut after).expect("len 2");
    assert_close(after[0], before[0], TOL);
    assert_close(after[1], before[1], TOL);
}

#[test]
fn value_and_gradient_cholesky_failure_keeps_params() {
    let mut gpr = Gpr::new(
        KernelSpec::from(RbfKernel::new(1.0).expect("valid")),
        GaussianLikelihood::new(0.1)
            .expect("valid")
            .with_bounds(Interval::new(1e-30, 1e5).expect("open"))
            .expect("inside"),
    )
    .with_optimizer(Fixed)
    .factor(&[0.0, 0.0], 2, 1, &[0.5, -0.25])
    .expect("spd");
    let snapshot = gpr.clone();
    let mut before = [0.0; 2];
    gpr.get_params(&mut before).expect("len 2");
    let mut bad = before;
    bad[0] = 0.5;
    bad[1] = (1e-20_f64).ln();
    let mut grad = [0.0; 2];
    assert!(matches!(
        gpr.value_and_gradient_into(&bad, &mut grad),
        Err(GprError::CholeskyFailed { .. })
    ));
    let mut after = [0.0; 2];
    gpr.get_params(&mut after).expect("len 2");
    assert_close(after[0], before[0], TOL);
    assert_close(after[1], before[1], TOL);
    let pred = gpr.predict(&[0.0], 1, 1).expect("usable after failed grad");
    let pred0 = snapshot.predict(&[0.0], 1, 1).expect("snapshot predict");
    assert_close(pred.mean[0], pred0.mean[0], TOL);
    assert_close(pred.variance[0], pred0.variance[0], TOL);
    assert_close(
        gpr.neg_log_marginal_likelihood().expect("nlml"),
        snapshot
            .neg_log_marginal_likelihood()
            .expect("snapshot nlml"),
        TOL,
    );
    assert_eq!(gpr.alpha().len(), snapshot.alpha().len());
    for (a, b) in gpr.alpha().iter().zip(snapshot.alpha()) {
        assert_close(*a, *b, TOL);
    }
    gpr.value_and_gradient_into(&before, &mut grad)
        .expect("restore");
}

#[test]
fn uncached_workspace_has_no_distance_cache() {
    let iso = rbf_gpr(1.25, 0.16)
        .with_distance_cache_policy(DistanceCachePolicy::Uncached)
        .with_optimizer(Fixed)
        .factor(&[0.0, 0.8, 1.7], 3, 1, &[0.4, -0.2, 0.9])
        .expect("spd");
    assert!(!iso.store.buffers.has_distance_cache());
    let ard = rbf_ard_gpr(&[1.25, 0.8], 0.16)
        .with_distance_cache_policy(DistanceCachePolicy::Uncached)
        .with_optimizer(Fixed)
        .factor(&[0.0, 0.8, 1.7, 0.2, -0.4, 0.9], 3, 2, &[0.4, -0.2, 0.9])
        .expect("spd");
    assert!(!ard.store.buffers.has_distance_cache());
}

#[test]
fn always_reuses_poisoned_dist_cache() {
    let x = [0.0, 0.8, 1.7];
    let y = [0.4, -0.2, 0.9];
    let mut gpr = rbf_gpr(1.25, 0.16)
        .with_distance_cache_policy(DistanceCachePolicy::Cached)
        .with_optimizer(Fixed)
        .factor(&x, 3, 1, &y)
        .expect("spd");
    let mut params = [0.0; 2];
    gpr.get_params(&mut params).expect("len 2");
    let mut grad = [0.0; 2];
    let good = gpr
        .value_and_gradient_into(&params, &mut grad)
        .expect("spd");
    {
        let ws = gpr.store.buffers.dist.as_mut().expect("distance cache");
        let n = ws.dist.as_ref().expect("filled at fit").nrows();
        ws.dist = Some(Mat::from_fn(n, n, |_, _| 999.0));
    }
    let poisoned = gpr
        .value_and_gradient_into(&params, &mut grad)
        .expect("spd");
    assert!(
        (poisoned - good).abs() > 1e-3,
        "Always should keep the poisoned distances: good={good}, poisoned={poisoned}"
    );
}

#[test]
fn never_and_always_match_rbf_nlml_grad_and_predict() {
    let x = [0.0, 0.8, 1.7];
    let y = [0.4, -0.2, 0.9];
    let mut never = rbf_gpr(1.25, 0.16)
        .with_distance_cache_policy(DistanceCachePolicy::Uncached)
        .with_optimizer(Fixed)
        .factor(&x, 3, 1, &y)
        .expect("spd");
    let mut always = rbf_gpr(1.25, 0.16)
        .with_distance_cache_policy(DistanceCachePolicy::Cached)
        .with_optimizer(Fixed)
        .factor(&x, 3, 1, &y)
        .expect("spd");
    let mut params = [0.0; 2];
    never.get_params(&mut params).expect("len 2");
    let mut grad_n = [0.0; 2];
    let mut grad_a = [0.0; 2];
    let vn = never
        .value_and_gradient_into(&params, &mut grad_n)
        .expect("spd");
    let va = always
        .value_and_gradient_into(&params, &mut grad_a)
        .expect("spd");
    assert_close(vn, va, TOL);
    assert_close(grad_n[0], grad_a[0], TOL);
    assert_close(grad_n[1], grad_a[1], TOL);
    let pn = never.predict(&[0.5], 1, 1).expect("fitted");
    let pa = always.predict(&[0.5], 1, 1).expect("fitted");
    assert_close(pn.mean[0], pa.mean[0], TOL);
    assert_close(pn.variance[0], pa.variance[0], TOL);
}

#[test]
fn never_and_always_match_rbf_ard_nlml_grad_and_predict() {
    let x = [0.0, 0.8, 1.7, 0.2, -0.4, 0.9];
    let y = [0.4, -0.2, 0.9];
    let xs = [0.5, 0.1];
    let mut never = rbf_ard_gpr(&[1.25, 0.8], 0.16)
        .with_distance_cache_policy(DistanceCachePolicy::Uncached)
        .with_optimizer(Fixed)
        .factor(&x, 3, 2, &y)
        .expect("spd");
    let mut always = rbf_ard_gpr(&[1.25, 0.8], 0.16)
        .with_distance_cache_policy(DistanceCachePolicy::Cached)
        .with_optimizer(Fixed)
        .factor(&x, 3, 2, &y)
        .expect("spd");
    assert!(!never.store.buffers.has_distance_cache());
    assert!(always.store.buffers.has_distance_cache());
    assert_eq!(
        always
            .store
            .buffers
            .dist
            .as_ref()
            .expect("distance cache")
            .ard_sq_diff
            .as_ref()
            .expect("filled at fit")
            .ncols(),
        6
    );
    let mut params = [0.0; 3];
    never.get_params(&mut params).expect("len 3");
    let mut grad_n = [0.0; 3];
    let mut grad_a = [0.0; 3];
    let vn = never
        .value_and_gradient_into(&params, &mut grad_n)
        .expect("spd");
    let va = always
        .value_and_gradient_into(&params, &mut grad_a)
        .expect("spd");
    assert_close(vn, va, TOL);
    assert_close(grad_n[0], grad_a[0], TOL);
    assert_close(grad_n[1], grad_a[1], TOL);
    assert_close(grad_n[2], grad_a[2], TOL);
    let pn = never.predict(&xs, 1, 2).expect("fitted");
    let pa = always.predict(&xs, 1, 2).expect("fitted");
    assert_close(pn.mean[0], pa.mean[0], TOL);
    assert_close(pn.variance[0], pa.variance[0], TOL);
}

#[test]
fn rbf_ard_fit_optimizes_with_always_cache() {
    let x = [0.0, 0.8, 1.7, 0.2, -0.4, 0.9];
    let y = [0.4, -0.2, 0.9];
    let gpr =
        rbf_ard_gpr(&[1.25, 0.8], 0.16).with_distance_cache_policy(DistanceCachePolicy::Cached);
    let mut before = [0.0; 3];
    gpr.get_params(&mut before).expect("len 3");
    let gpr = gpr.fit(&x, 3, 2, &y).expect("optimize");
    let mut after = [0.0; 3];
    gpr.get_params(&mut after).expect("len 3");
    assert!(
        before.iter().zip(&after).any(|(a, b)| (a - b).abs() > 1e-9),
        "L-BFGS should move ARD θ: before={before:?}, after={after:?}"
    );
    let dist = gpr.store.buffers.dist.as_ref().expect("distance cache");
    let ard = dist.ard_sq_diff.as_ref().expect("filled at fit");
    assert_eq!(ard.ncols(), 6);
}

#[test]
fn retain_and_reuse_match_nlml_grad_and_predict() {
    let x = [0.0, 0.4, 1.0];
    let y = [0.2, -0.1, 0.8];
    let xs = [0.5];
    let mut retain = rbf_gpr(1.0, 0.1)
        .with_optimizer(Fixed)
        .factor(&x, 3, 1, &y)
        .expect("spd");
    let mut reuse = rbf_gpr(1.0, 0.1)
        .with_cholesky_buffer(CholeskyBuffer::Reuse)
        .with_optimizer(Fixed)
        .factor(&x, 3, 1, &y)
        .expect("spd");
    let mut params = [0.0; 2];
    retain.get_params(&mut params).expect("len 2");
    let mut grad_r = [0.0; 2];
    let mut grad_u = [0.0; 2];
    let vr = retain
        .value_and_gradient_into(&params, &mut grad_r)
        .expect("spd");
    let vu = reuse
        .value_and_gradient_into(&params, &mut grad_u)
        .expect("spd");
    assert_close(vr, vu, TOL);
    assert_close(grad_r[0], grad_u[0], TOL);
    assert_close(grad_r[1], grad_u[1], TOL);
    let pr = retain.predict(&xs, 1, 1).expect("fitted");
    let pu = reuse.predict(&xs, 1, 1).expect("fitted");
    assert_close(pr.mean[0], pu.mean[0], TOL);
    assert_close(pr.variance[0], pu.variance[0], TOL);

    let fitted_r = rbf_gpr(1.0, 0.1).fit(&x, 3, 1, &y).expect("optimize");
    let fitted_u = rbf_gpr(1.0, 0.1)
        .with_cholesky_buffer(CholeskyBuffer::Reuse)
        .fit(&x, 3, 1, &y)
        .expect("optimize");
    let fr = fitted_r.predict(&xs, 1, 1).expect("fitted");
    let fu = fitted_u.predict(&xs, 1, 1).expect("fitted");
    assert_close(fr.mean[0], fu.mean[0], TOL);
    assert_close(fr.variance[0], fu.variance[0], TOL);
}

#[test]
fn always_reuses_poisoned_ard_cache() {
    let x = [0.0, 0.8, 1.7, 0.2, -0.4, 0.9];
    let y = [0.4, -0.2, 0.9];
    let mut gpr = rbf_ard_gpr(&[1.25, 0.8], 0.16)
        .with_distance_cache_policy(DistanceCachePolicy::Cached)
        .with_optimizer(Fixed)
        .factor(&x, 3, 2, &y)
        .expect("spd");
    let mut params = [0.0; 3];
    gpr.get_params(&mut params).expect("len 3");
    let mut grad = [0.0; 3];
    let good = gpr
        .value_and_gradient_into(&params, &mut grad)
        .expect("spd");
    {
        let ws = gpr.store.buffers.dist.as_mut().expect("distance cache");
        let ard = ws.ard_sq_diff.as_mut().expect("filled at fit");
        let n = 3;
        for dim in 0..2 {
            for col in 0..n {
                for row in col..n {
                    ard[(row, dim * n + col)] = 999.0;
                }
            }
        }
    }
    let poisoned = gpr
        .value_and_gradient_into(&params, &mut grad)
        .expect("spd");
    assert!(
        (poisoned - good).abs() > 1e-3,
        "Always should keep the poisoned ARD cache: good={good}, poisoned={poisoned}"
    );
}

#[test]
fn always_ard_cache_retiling_follows_n() {
    let gpr = rbf_ard_gpr(&[1.0, 1.5], 0.16)
        .with_distance_cache_policy(DistanceCachePolicy::Cached)
        .with_optimizer(Fixed)
        .factor(&[0.0, 0.8, 1.7, 0.2, -0.4, 0.9], 3, 2, &[0.4, -0.2, 0.9])
        .expect("spd n=3");
    {
        let ws = gpr.store.buffers.dist.as_ref().expect("distance cache");
        let ard = ws.ard_sq_diff.as_ref().expect("filled at fit");
        assert_eq!(ard.nrows(), 3);
        assert_eq!(ard.ncols(), 6);
    }
    let gpr = gpr
        .into_trainer()
        .with_optimizer(Fixed)
        .factor(
            &[0.0, 0.8, 1.7, 2.1, 0.2, -0.4, 0.9, 0.3],
            4,
            2,
            &[0.4, -0.2, 0.9, 0.1],
        )
        .expect("spd n=4");
    let ws = gpr.store.buffers.dist.as_ref().expect("distance cache");
    let ard = ws.ard_sq_diff.as_ref().expect("filled at fit");
    assert_eq!(ard.nrows(), 4);
    assert_eq!(ard.ncols(), 8);
}

#[test]
fn isotropic_always_leaves_ard_cache_empty() {
    let gpr = rbf_gpr(1.25, 0.16)
        .with_distance_cache_policy(DistanceCachePolicy::Cached)
        .with_optimizer(Fixed)
        .factor(&[0.0, 0.8, 1.7], 3, 1, &[0.4, -0.2, 0.9])
        .expect("spd");
    let ws = gpr.store.buffers.dist.as_ref().expect("distance cache");
    assert!(ws.ard_sq_diff.is_none());
    assert!(ws.dist.is_some());
}

#[test]
fn never_and_always_match_matern_ard_nlml() {
    let x = [0.0, 0.8, 1.7, 0.2, -0.4, 0.9];
    let y = [0.4, -0.2, 0.9];
    let nu = MaternNu::ThreeHalves;
    let mut never = Gpr::new(
        KernelSpec::from(MaternArdKernel::new(&[1.25, 0.8], nu).expect("valid")),
        GaussianLikelihood::new(0.16).expect("valid"),
    )
    .with_distance_cache_policy(DistanceCachePolicy::Uncached)
    .with_optimizer(Fixed)
    .factor(&x, 3, 2, &y)
    .expect("spd");
    let mut always = Gpr::new(
        KernelSpec::from(MaternArdKernel::new(&[1.25, 0.8], nu).expect("valid")),
        GaussianLikelihood::new(0.16).expect("valid"),
    )
    .with_distance_cache_policy(DistanceCachePolicy::Cached)
    .with_optimizer(Fixed)
    .factor(&x, 3, 2, &y)
    .expect("spd");
    let mut params = [0.0; 3];
    never.get_params(&mut params).expect("len 3");
    let mut grad_n = [0.0; 3];
    let mut grad_a = [0.0; 3];
    let vn = never
        .value_and_gradient_into(&params, &mut grad_n)
        .expect("spd");
    let va = always
        .value_and_gradient_into(&params, &mut grad_a)
        .expect("spd");
    assert_close(vn, va, TOL);
    assert_close(grad_n[0], grad_a[0], TOL);
    assert_close(grad_n[1], grad_a[1], TOL);
    assert_close(grad_n[2], grad_a[2], TOL);
}

#[test]
fn value_and_gradient_matches_nlml_and_finite_difference() {
    let mut gpr = rbf_gpr(1.25, 0.16)
        .with_optimizer(Fixed)
        .factor(&[0.0, 0.8, 1.7], 3, 1, &[0.4, -0.2, 0.9])
        .expect("spd");
    let mut params = [0.0; 2];
    gpr.get_params(&mut params).expect("len 2");
    let mut grad = [0.0; 2];
    let value = gpr
        .value_and_gradient_into(&params, &mut grad)
        .expect("spd");
    assert_close(
        value,
        gpr.neg_log_marginal_likelihood().expect("fitted"),
        TOL,
    );
    let h = 1e-5;
    let mut dummy = [0.0; 2];
    for i in 0..2 {
        let mut plus = params;
        let mut minus = params;
        plus[i] += h;
        minus[i] -= h;
        let v_plus = gpr
            .value_and_gradient_into(&plus, &mut dummy)
            .expect("plus");
        let v_minus = gpr
            .value_and_gradient_into(&minus, &mut dummy)
            .expect("minus");
        let fd = (v_plus - v_minus) / (2.0 * h);
        let scale = fd.abs().max(1.0);
        assert!(
            (grad[i] - fd).abs() <= 1e-5 * scale,
            "param {i}: analytic={}, fd={}",
            grad[i],
            fd
        );
    }
    gpr.value_and_gradient_into(&params, &mut dummy)
        .expect("restore");
}

#[test]
fn hessian_matches_finite_difference_of_gradient() {
    let mut gpr = rbf_gpr(1.25, 0.16)
        .with_optimizer(Fixed)
        .factor(&[0.0, 0.8, 1.7], 3, 1, &[0.4, -0.2, 0.9])
        .expect("spd");
    let mut params = [0.0; 2];
    gpr.get_params(&mut params).expect("len 2");
    let mut hess = [0.0; 4];
    gpr.hessian_into(&params, &mut hess).expect("spd");
    let h = 1e-5;
    let mut g_plus = [0.0; 2];
    let mut g_minus = [0.0; 2];
    for j in 0..2 {
        let mut plus = params;
        let mut minus = params;
        plus[j] += h;
        minus[j] -= h;
        gpr.value_and_gradient_into(&plus, &mut g_plus)
            .expect("plus");
        gpr.value_and_gradient_into(&minus, &mut g_minus)
            .expect("minus");
        for i in 0..2 {
            let fd = (g_plus[i] - g_minus[i]) / (2.0 * h);
            let scale = fd.abs().max(1.0);
            assert!(
                (hess[i * 2 + j] - fd).abs() <= 2e-4 * scale,
                "H[{i},{j}]: analytic={}, fd={}",
                hess[i * 2 + j],
                fd
            );
        }
    }
}

#[test]
fn hessian_product_matches_finite_difference_of_gradient() {
    let kernel = KernelSpec::from(RbfKernel::new(1.25).expect("valid"))
        * KernelSpec::from(crate::kernel::ConstantKernel::new(1.4).expect("valid"));
    let mut gpr = Gpr::new(kernel, GaussianLikelihood::new(0.16).expect("valid"))
        .with_optimizer(Fixed)
        .factor(&[0.0, 0.8, 1.7], 3, 1, &[0.4, -0.2, 0.9])
        .expect("spd");
    let n = gpr.num_params();
    let mut params = vec![0.0; n];
    gpr.get_params(&mut params).expect("len");
    let mut hess = vec![0.0; n * n];
    gpr.hessian_into(&params, &mut hess).expect("spd");
    let h = 1e-5;
    let mut g_plus = vec![0.0; n];
    let mut g_minus = vec![0.0; n];
    for j in 0..n {
        let mut plus = params.clone();
        let mut minus = params.clone();
        plus[j] += h;
        minus[j] -= h;
        gpr.value_and_gradient_into(&plus, &mut g_plus)
            .expect("plus");
        gpr.value_and_gradient_into(&minus, &mut g_minus)
            .expect("minus");
        for i in 0..n {
            let fd = (g_plus[i] - g_minus[i]) / (2.0 * h);
            let scale = fd.abs().max(1.0);
            assert!(
                (hess[i * n + j] - fd).abs() <= 2e-4 * scale,
                "H[{i},{j}]: analytic={}, fd={}",
                hess[i * n + j],
                fd
            );
        }
    }
}

#[test]
fn value_and_gradient_sum_rbf_matches_finite_difference() {
    let kernel = KernelSpec::from(RbfKernel::new(1.25).expect("valid"))
        + KernelSpec::from(RbfKernel::new(0.7).expect("valid"));
    let mut gpr = Gpr::new(kernel, GaussianLikelihood::new(0.16).expect("valid"))
        .with_optimizer(Fixed)
        .factor(&[0.0, 0.8, 1.7], 3, 1, &[0.4, -0.2, 0.9])
        .expect("spd");
    let n_params = gpr.num_params();
    let mut params = vec![0.0; n_params];
    gpr.get_params(&mut params).expect("len");
    let mut grad = vec![0.0; n_params];
    gpr.value_and_gradient_into(&params, &mut grad)
        .expect("spd");
    let h = 1e-5;
    let mut dummy = vec![0.0; n_params];
    for i in 0..n_params {
        let mut plus = params.clone();
        let mut minus = params.clone();
        plus[i] += h;
        minus[i] -= h;
        let v_plus = gpr
            .value_and_gradient_into(&plus, &mut dummy)
            .expect("plus");
        let v_minus = gpr
            .value_and_gradient_into(&minus, &mut dummy)
            .expect("minus");
        let fd = (v_plus - v_minus) / (2.0 * h);
        let scale = fd.abs().max(1.0);
        assert!(
            (grad[i] - fd).abs() <= 1e-5 * scale,
            "param {i}: analytic={}, fd={}",
            grad[i],
            fd
        );
    }
}

#[test]
fn value_and_gradient_product_matches_finite_difference() {
    let kernel = KernelSpec::from(RbfKernel::new(1.0).expect("valid"))
        * KernelSpec::from(RbfKernel::new(2.0).expect("valid"));
    let mut gpr = Gpr::new(kernel, GaussianLikelihood::new(0.1).expect("valid"))
        .with_optimizer(Fixed)
        .factor(&[0.0, 1.0], 2, 1, &[0.5, -0.25])
        .expect("spd");
    let n_params = gpr.num_params();
    let mut params = vec![0.0; n_params];
    gpr.get_params(&mut params).expect("len");
    let mut grad = vec![0.0; n_params];
    gpr.value_and_gradient_into(&params, &mut grad)
        .expect("spd");
    let h = 1e-5;
    let mut dummy = vec![0.0; n_params];
    for i in 0..n_params {
        let mut plus = params.clone();
        let mut minus = params.clone();
        plus[i] += h;
        minus[i] -= h;
        let v_plus = gpr
            .value_and_gradient_into(&plus, &mut dummy)
            .expect("plus");
        let v_minus = gpr
            .value_and_gradient_into(&minus, &mut dummy)
            .expect("minus");
        let fd = (v_plus - v_minus) / (2.0 * h);
        let scale = fd.abs().max(1.0);
        assert!(
            (grad[i] - fd).abs() <= 1e-5 * scale,
            "param {i}: analytic={}, fd={}",
            grad[i],
            fd
        );
    }
}

fn assert_mll_grad_matches_finite_difference(gpr: &mut FittedGpr<Fixed>) {
    let n_params = gpr.num_params();
    let mut params = vec![0.0; n_params];
    gpr.get_params(&mut params).expect("len");
    let mut grad = vec![0.0; n_params];
    gpr.value_and_gradient_into(&params, &mut grad)
        .expect("spd");
    let h = 1e-5;
    let mut dummy = vec![0.0; n_params];
    for i in 0..n_params {
        let mut plus = params.clone();
        let mut minus = params.clone();
        plus[i] += h;
        minus[i] -= h;
        let v_plus = gpr
            .value_and_gradient_into(&plus, &mut dummy)
            .expect("plus");
        let v_minus = gpr
            .value_and_gradient_into(&minus, &mut dummy)
            .expect("minus");
        let fd = (v_plus - v_minus) / (2.0 * h);
        let scale = fd.abs().max(1.0);
        assert!(
            (grad[i] - fd).abs() <= 1e-5 * scale,
            "param {i}: analytic={}, fd={}",
            grad[i],
            fd
        );
    }
}

#[test]
fn value_and_gradient_linear_times_constant_matches_finite_difference() {
    let kernel = KernelSpec::from(LinearKernel::new(1.0).expect("valid"))
        * KernelSpec::from(ConstantKernel::new(1.5).expect("valid"));
    let mut gpr = Gpr::new(kernel, GaussianLikelihood::new(0.1).expect("valid"))
        .with_optimizer(Fixed)
        .factor(&[0.5, 1.5], 2, 1, &[0.5, -0.25])
        .expect("spd");
    assert_mll_grad_matches_finite_difference(&mut gpr);
}

#[test]
fn value_and_gradient_linear_times_ard_matches_finite_difference() {
    let kernel = KernelSpec::from(LinearKernel::new(1.0).expect("valid"))
        * KernelSpec::from(RbfArdKernel::new(&[1.2, 0.8]).expect("valid"));
    let x = [0.0, 1.0, 0.2, 0.0, 0.4, 1.1];
    let mut gpr = Gpr::new(kernel, GaussianLikelihood::new(0.1).expect("valid"))
        .with_optimizer(Fixed)
        .factor(&x, 3, 2, &[0.2, -1.0, 0.7])
        .expect("spd");
    assert_mll_grad_matches_finite_difference(&mut gpr);
}

#[test]
fn rbf_plus_linear_fits_predicts_and_matches_finite_difference() {
    let kernel = KernelSpec::from(RbfKernel::new(1.0).expect("valid"))
        + KernelSpec::from(LinearKernel::new(1.0).expect("valid"));
    let mut gpr = Gpr::new(kernel, GaussianLikelihood::new(0.1).expect("valid"))
        .with_optimizer(Fixed)
        .factor(&[0.5, 1.5], 2, 1, &[0.5, -0.25])
        .expect("spd");
    let pred = gpr.predict(&[1.0], 1, 1).expect("fitted");
    assert!(pred.mean[0].is_finite());
    assert!(pred.variance[0] > 0.0);
    assert_mll_grad_matches_finite_difference(&mut gpr);
}

#[test]
fn rbf_times_linear_fits_and_matches_finite_difference() {
    let kernel = KernelSpec::from(RbfKernel::new(1.0).expect("valid"))
        * KernelSpec::from(LinearKernel::new(1.0).expect("valid"));
    let mut gpr = Gpr::new(kernel, GaussianLikelihood::new(0.1).expect("valid"))
        .with_optimizer(Fixed)
        .factor(&[0.5, 1.5], 2, 1, &[0.5, -0.25])
        .expect("spd");
    let pred = gpr.predict(&[1.0], 1, 1).expect("fitted");
    assert!(pred.mean[0].is_finite());
    assert_mll_grad_matches_finite_difference(&mut gpr);
}

#[test]
fn value_and_gradient_sum_of_product_matches_finite_difference() {
    let kernel = KernelSpec::from(ConstantKernel::new(1.5).expect("valid"))
        * KernelSpec::from(RbfKernel::new(1.0).expect("valid"))
        + KernelSpec::from(RbfKernel::new(2.0).expect("valid"));
    let mut gpr = Gpr::new(kernel, GaussianLikelihood::new(0.1).expect("valid"))
        .with_optimizer(Fixed)
        .factor(&[0.0, 1.0], 2, 1, &[0.5, -0.25])
        .expect("spd");
    let n_params = gpr.num_params();
    let mut params = vec![0.0; n_params];
    gpr.get_params(&mut params).expect("len");
    let mut grad = vec![0.0; n_params];
    gpr.value_and_gradient_into(&params, &mut grad)
        .expect("spd");
    let h = 1e-5;
    let mut dummy = vec![0.0; n_params];
    for i in 0..n_params {
        let mut plus = params.clone();
        let mut minus = params.clone();
        plus[i] += h;
        minus[i] -= h;
        let v_plus = gpr
            .value_and_gradient_into(&plus, &mut dummy)
            .expect("plus");
        let v_minus = gpr
            .value_and_gradient_into(&minus, &mut dummy)
            .expect("minus");
        let fd = (v_plus - v_minus) / (2.0 * h);
        let scale = fd.abs().max(1.0);
        assert!(
            (grad[i] - fd).abs() <= 1e-5 * scale,
            "param {i}: analytic={}, fd={}",
            grad[i],
            fd
        );
    }
}

#[test]
fn value_and_gradient_n_one_noise_matches_closed_form() {
    let noise = 0.25;
    let y = 2.0;
    let mut gpr = rbf_gpr(1.0, noise)
        .with_optimizer(Fixed)
        .factor(&[0.0], 1, 1, &[y])
        .expect("spd");
    let mut params = [0.0; 2];
    gpr.get_params(&mut params).expect("len 2");
    let mut grad = [0.0; 2];
    gpr.value_and_gradient_into(&params, &mut grad)
        .expect("spd");
    let a = 1.0 + noise;
    let w = (y / a) * (y / a) - 1.0 / a;
    assert_close(grad[0], 0.0, TOL);
    assert_close(grad[1], -0.5 * w * noise, TOL);
}

#[test]
fn predict_rejects_wrong_dim() {
    let gpr = rbf_gpr(1.0, 0.1)
        .with_optimizer(Fixed)
        .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
        .expect("spd");
    assert!(matches!(
        gpr.predict(&[0.0, 1.0], 1, 2),
        Err(GprError::DimensionMismatch {
            x_dim: 2,
            expected_dim: 1
        })
    ));
}

#[test]
fn loo_n_one_is_prior() {
    let noise = 0.25;
    let gpr = rbf_gpr(1.0, noise)
        .with_optimizer(Fixed)
        .factor(&[0.0], 1, 1, &[2.0])
        .expect("spd");
    let loo = gpr.loo_predict().expect("fitted");
    assert_eq!(loo.variance_kind, VarianceKind::Observation);
    assert_close(loo.mean[0], 0.0, TOL);
    assert_close(loo.variance[0], 1.0 + noise, TOL);
    let lat = gpr
        .loo_predict_with(PredictOptions {
            variance_kind: VarianceKind::Latent,
        })
        .expect("fitted");
    assert_close(lat.mean[0], 0.0, TOL);
    assert_close(lat.variance[0], 1.0, TOL);
}

#[test]
fn loo_n_two_matches_closed_form() {
    let ell = 1.0;
    let noise = 0.25;
    let x = [0.0, 1.0];
    let y = [0.5, 1.5];
    let gpr = rbf_gpr(ell, noise)
        .with_optimizer(Fixed)
        .factor(&x, 2, 1, &y)
        .expect("spd");
    let k01 = (-0.5 / (ell * ell)).exp();
    let a = 1.0 + noise;
    let det = a * a - k01 * k01;
    let qii = a / det;
    let inv01 = -k01 / det;
    let alpha0 = qii * y[0] + inv01 * y[1];
    let alpha1 = inv01 * y[0] + qii * y[1];
    let loo = gpr.loo_predict().expect("fitted");
    assert_eq!(loo.mean.len(), 2);
    assert_eq!(loo.variance_kind, VarianceKind::Observation);
    assert_close(loo.mean[0], y[0] - alpha0 / qii, TOL);
    assert_close(loo.mean[1], y[1] - alpha1 / qii, TOL);
    assert_close(loo.variance[0], 1.0 / qii, TOL);
    assert_close(loo.variance[1], 1.0 / qii, TOL);
    let lat = gpr
        .loo_predict_with(PredictOptions {
            variance_kind: VarianceKind::Latent,
        })
        .expect("fitted");
    assert_close(lat.variance[0], (1.0 / qii - noise).max(0.0), TOL);
    assert_close(lat.variance[1], (1.0 / qii - noise).max(0.0), TOL);
}

fn omit_training_row(
    x: &[f64],
    y: &[f64],
    n: usize,
    d: usize,
    skip: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n_out = n - 1;
    let mut xo = vec![0.0; n_out * d];
    let mut yo = Vec::with_capacity(n_out);
    let mut xs = vec![0.0; d];
    let mut o = 0;
    for i in 0..n {
        if i == skip {
            for dim in 0..d {
                xs[dim] = x[dim * n + i];
            }
            continue;
        }
        for dim in 0..d {
            xo[dim * n_out + o] = x[dim * n + i];
        }
        yo.push(y[i]);
        o += 1;
    }
    (xo, yo, xs)
}

#[test]
fn loo_n_three_matches_refit_predict() {
    let ell = 1.25;
    let noise = 0.16;
    let n = 3;
    let d = 1;
    let x = [0.0, 0.5, 1.5];
    let y = [0.2, -1.0, 0.7];
    let gpr = rbf_gpr(ell, noise)
        .with_optimizer(Fixed)
        .factor(&x, n, d, &y)
        .expect("spd");
    let loo_obs = gpr.loo_predict().expect("fitted");
    let loo_lat = gpr
        .loo_predict_with(PredictOptions {
            variance_kind: VarianceKind::Latent,
        })
        .expect("fitted");
    for skip in 0..n {
        let (xo, yo, xs) = omit_training_row(&x, &y, n, d, skip);
        let held = rbf_gpr(ell, noise)
            .with_optimizer(Fixed)
            .factor(&xo, n - 1, d, &yo)
            .expect("spd");
        let pred_obs = held.predict(&xs, 1, d).expect("fitted");
        let pred_lat = held
            .predict_with(
                &xs,
                1,
                d,
                PredictOptions {
                    variance_kind: VarianceKind::Latent,
                },
            )
            .expect("fitted");
        assert_close(loo_obs.mean[skip], pred_obs.mean[0], TOL);
        assert_close(loo_obs.variance[skip], pred_obs.variance[0], TOL);
        assert_close(loo_lat.mean[skip], pred_lat.mean[0], TOL);
        assert_close(loo_lat.variance[skip], pred_lat.variance[0], TOL);
    }
}

#[test]
fn loo_observation_is_latent_plus_noise_after_inverse() {
    let noise = 0.16;
    let y = [0.0, 4.0];
    let gpr = rbf_gpr(1.0, noise)
        .with_target_transform(StandardizeTarget::new())
        .with_optimizer(Fixed)
        .factor(&[0.0, 1.0], 2, 1, &y)
        .expect("spd");
    let t = StandardizeTarget::new().fit(&y).expect("finite");
    let scale = t.std();
    let scale_sq = scale * scale;
    let lat = gpr
        .loo_predict_with(PredictOptions {
            variance_kind: VarianceKind::Latent,
        })
        .expect("fitted");
    let obs = gpr.loo_predict().expect("fitted");
    assert_close(obs.variance[0], lat.variance[0] + scale_sq * noise, TOL);
    assert_close(obs.variance[1], lat.variance[1] + scale_sq * noise, TOL);
}

#[test]
fn predict_n_one_matches_closed_form() {
    let noise = 0.25;
    let gpr = rbf_gpr(1.0, noise)
        .with_optimizer(Fixed)
        .factor(&[0.0], 1, 1, &[2.0])
        .expect("spd");
    let pred = gpr
        .predict_with(
            &[0.0],
            1,
            1,
            PredictOptions {
                variance_kind: VarianceKind::Latent,
            },
        )
        .expect("fitted");
    let a = 1.0 + noise;
    assert_close(pred.mean[0], 2.0 / a, TOL);
    assert_close(pred.variance[0], 1.0 - 1.0 / a, TOL);
    let obs = gpr.predict(&[0.0], 1, 1).expect("fitted");
    assert_eq!(obs.variance_kind, VarianceKind::Observation);
    assert_close(obs.variance[0], pred.variance[0] + noise, TOL);
}

#[test]
fn observation_variance_is_latent_plus_noise_after_inverse() {
    let noise = 0.16;
    let y = [0.0, 4.0];
    let gpr = rbf_gpr(1.0, noise)
        .with_target_transform(StandardizeTarget::new())
        .with_optimizer(Fixed)
        .factor(&[0.0, 1.0], 2, 1, &y)
        .expect("spd");
    let t = StandardizeTarget::new().fit(&y).expect("finite");
    let scale = t.std();
    let scale_sq = scale * scale;
    let lat = gpr
        .predict_with(
            &[0.5],
            1,
            1,
            PredictOptions {
                variance_kind: VarianceKind::Latent,
            },
        )
        .expect("fitted");
    let obs = gpr.predict(&[0.5], 1, 1).expect("fitted");
    assert_close(obs.variance[0], lat.variance[0] + scale_sq * noise, TOL);
    let mut recovered = y;
    t.transform(&mut recovered).expect("fitted");
    t.inverse_transform_mean(&mut recovered).expect("fitted");
    assert_close(recovered[0], y[0], TOL);
    assert_close(recovered[1], y[1], TOL);
}

#[test]
fn ard_equal_lengthscales_match_isotropic_predict() {
    let ell = 1.25;
    let noise = 0.1;
    let x = [0.0, 0.5, 1.5, 0.0, 1.0, 0.5];
    let y = [0.2, -1.0, 0.7];
    let xs = [0.25, 1.0];
    let iso = rbf_gpr(ell, noise)
        .with_optimizer(Fixed)
        .factor(&x, 3, 2, &y)
        .expect("spd");
    let mut ard = Gpr::new(
        KernelSpec::from(RbfArdKernel::new(&[ell, ell]).expect("valid")),
        GaussianLikelihood::new(noise).expect("valid"),
    )
    .with_optimizer(Fixed)
    .factor(&x, 3, 2, &y)
    .expect("spd");
    let p_iso = iso.predict(&xs, 1, 2).expect("fitted");
    let p_ard = ard.predict(&xs, 1, 2).expect("fitted");
    assert_close(p_ard.mean[0], p_iso.mean[0], TOL);
    assert_close(p_ard.variance[0], p_iso.variance[0], TOL);
    let mut params = vec![0.0; ard.num_params()];
    ard.get_params(&mut params).expect("len");
    let mut grad = vec![0.0; params.len()];
    let nlml = ard
        .value_and_gradient_into(&params, &mut grad)
        .expect("spd");
    assert!(nlml.is_finite());
    assert_eq!(params.len(), 3);
    assert!(grad.iter().all(|g| g.is_finite()));
}

#[test]
fn rbf_plus_white_fits() {
    let gpr = Gpr::new(
        KernelSpec::from(RbfKernel::new(1.0).expect("valid"))
            + KernelSpec::from(WhiteKernel::new(0.05).expect("valid")),
        GaussianLikelihood::new(0.1).expect("valid"),
    )
    .with_optimizer(Fixed)
    .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
    .expect("spd");
    let pred = gpr.predict(&[0.5], 1, 1).expect("fitted");
    assert!(pred.mean[0].is_finite());
    assert!(pred.variance[0] > 0.0);
}

#[test]
fn linear_kernel_fits_and_predicts() {
    let x = [0.0, 1.0, 2.0];
    let y = [0.0, 1.0, 2.0];
    let mut gpr = Gpr::new(
        KernelSpec::from(LinearKernel::new(1.0).expect("valid")),
        GaussianLikelihood::new(0.1).expect("valid"),
    )
    .with_optimizer(Fixed)
    .factor(&x, 3, 1, &y)
    .expect("spd");
    let pred = gpr.predict(&[1.5], 1, 1).expect("fitted");
    assert!(pred.mean[0].is_finite());
    let mut params = vec![0.0; gpr.num_params()];
    gpr.get_params(&mut params).expect("len");
    let mut grad = vec![0.0; params.len()];
    let nlml = gpr
        .value_and_gradient_into(&params, &mut grad)
        .expect("spd");
    assert!(nlml.is_finite());
    assert!(grad.iter().all(|g| g.is_finite()));
}

#[test]
fn points_kernel_allocates_no_distance_cache() {
    let fitted = Gpr::new(
        KernelSpec::from(LinearKernel::new(1.0).expect("valid")),
        GaussianLikelihood::new(0.1).expect("valid"),
    )
    .with_optimizer(Fixed)
    .factor(&[0.0, 1.0], 2, 1, &[0.5, -0.25])
    .expect("spd");
    assert_eq!(fitted.distance_cache_policy(), DistanceCachePolicy::Cached);
    assert!(!fitted.store.buffers.has_distance_cache());
    assert!(fitted.store.buffers.has_dedicated_w());
}

#[test]
fn new_trainer_defaults_to_speed_pole_and_accurate_exp() {
    let gpr = rbf_gpr(1.0, 0.1);
    assert_eq!(gpr.distance_cache_policy(), DistanceCachePolicy::Cached);
    assert_eq!(gpr.cholesky_buffer(), CholeskyBuffer::Retain);
    assert_eq!(gpr.math(), KernelExp::Accurate);
}

#[test]
fn prefer_memory_sets_uncached_and_reuse() {
    let gpr = rbf_gpr(1.0, 0.1).with_prefer_memory();
    assert_eq!(gpr.distance_cache_policy(), DistanceCachePolicy::Uncached);
    assert_eq!(gpr.cholesky_buffer(), CholeskyBuffer::Reuse);
}

#[test]
fn prefer_speed_sets_cached_and_retain() {
    let gpr = rbf_gpr(1.0, 0.1).with_prefer_memory().with_prefer_speed();
    assert_eq!(gpr.distance_cache_policy(), DistanceCachePolicy::Cached);
    assert_eq!(gpr.cholesky_buffer(), CholeskyBuffer::Retain);
}

#[test]
fn policies_survive_fit_and_into_trainer() {
    let fitted = rbf_gpr(1.0, 0.1)
        .with_prefer_memory()
        .with_math(KernelExp::FastApprox)
        .with_optimizer(Fixed)
        .factor(&[0.0, 1.0], 2, 1, &[0.5, -0.25])
        .expect("spd");
    assert_eq!(
        fitted.distance_cache_policy(),
        DistanceCachePolicy::Uncached
    );
    assert_eq!(fitted.cholesky_buffer(), CholeskyBuffer::Reuse);
    assert_eq!(fitted.math(), KernelExp::FastApprox);
    let trainer = fitted.into_trainer();
    assert_eq!(
        trainer.distance_cache_policy(),
        DistanceCachePolicy::Uncached
    );
    assert_eq!(trainer.cholesky_buffer(), CholeskyBuffer::Reuse);
    assert_eq!(trainer.math(), KernelExp::FastApprox);
}

#[test]
fn prefer_memory_workspace_has_no_dist_or_dedicated_w() {
    let mem = rbf_gpr(1.25, 0.16)
        .with_prefer_memory()
        .with_optimizer(Fixed)
        .factor(&[0.0, 0.8, 1.7], 3, 1, &[0.4, -0.2, 0.9])
        .expect("spd");
    assert!(!mem.store.buffers.has_distance_cache());
    assert!(!mem.store.buffers.has_dedicated_w());
    let speed = rbf_gpr(1.25, 0.16)
        .with_prefer_speed()
        .with_optimizer(Fixed)
        .factor(&[0.0, 0.8, 1.7], 3, 1, &[0.4, -0.2, 0.9])
        .expect("spd");
    assert!(speed.store.buffers.has_distance_cache());
    assert!(speed.store.buffers.has_dedicated_w());
}

#[test]
fn prefer_memory_matches_default_nlml_grad_and_predict() {
    let x = [0.0, 0.8, 1.7];
    let y = [0.4, -0.2, 0.9];
    let mut memory = rbf_gpr(1.25, 0.16)
        .with_prefer_memory()
        .with_optimizer(Fixed)
        .factor(&x, 3, 1, &y)
        .expect("spd");
    let mut speed = rbf_gpr(1.25, 0.16)
        .with_prefer_speed()
        .with_optimizer(Fixed)
        .factor(&x, 3, 1, &y)
        .expect("spd");
    let mut params = [0.0; 2];
    memory.get_params(&mut params).expect("len 2");
    let mut grad_m = [0.0; 2];
    let mut grad_s = [0.0; 2];
    let vm = memory
        .value_and_gradient_into(&params, &mut grad_m)
        .expect("spd");
    let vs = speed
        .value_and_gradient_into(&params, &mut grad_s)
        .expect("spd");
    assert_close(vm, vs, TOL);
    assert_close(grad_m[0], grad_s[0], TOL);
    assert_close(grad_m[1], grad_s[1], TOL);
    let pm = memory.predict(&[0.5], 1, 1).expect("fitted");
    let ps = speed.predict(&[0.5], 1, 1).expect("fitted");
    assert_close(pm.mean[0], ps.mean[0], TOL);
    assert_close(pm.variance[0], ps.variance[0], TOL);
}

#[test]
fn constant_kernel_fits_and_predicts() {
    let mut gpr = Gpr::new(
        KernelSpec::from(ConstantKernel::new(1.5).expect("valid")),
        GaussianLikelihood::new(0.1).expect("valid"),
    )
    .with_optimizer(Fixed)
    .factor(&[0.0, 1.0], 2, 1, &[0.5, -0.25])
    .expect("spd");
    let pred = gpr.predict(&[0.5], 1, 1).expect("fitted");
    assert!(pred.mean[0].is_finite());
    let mut params = vec![0.0; gpr.num_params()];
    gpr.get_params(&mut params).expect("len");
    let mut grad = vec![0.0; params.len()];
    let nlml = gpr
        .value_and_gradient_into(&params, &mut grad)
        .expect("spd");
    assert!(nlml.is_finite());
    assert!(grad.iter().all(|g| g.is_finite()));
}

#[test]
fn white_kernel_fits_and_predicts() {
    let mut gpr = Gpr::new(
        KernelSpec::from(WhiteKernel::new(0.2).expect("valid")),
        GaussianLikelihood::new(0.1).expect("valid"),
    )
    .with_optimizer(Fixed)
    .factor(&[0.0, 1.0], 2, 1, &[0.5, -0.25])
    .expect("spd");
    let pred = gpr.predict(&[0.5], 1, 1).expect("fitted");
    assert!(pred.mean[0].is_finite());
    let mut params = vec![0.0; gpr.num_params()];
    gpr.get_params(&mut params).expect("len");
    let mut grad = vec![0.0; params.len()];
    let nlml = gpr
        .value_and_gradient_into(&params, &mut grad)
        .expect("spd");
    assert!(nlml.is_finite());
    assert!(grad.iter().all(|g| g.is_finite()));
}

#[test]
fn matern_fits_and_predicts() {
    let mut gpr = Gpr::new(
        KernelSpec::from(MaternKernel::new(1.0, MaternNu::FiveHalves).expect("valid")),
        GaussianLikelihood::new(0.1).expect("valid"),
    )
    .with_optimizer(Fixed)
    .factor(&[0.0, 1.0, 2.0], 3, 1, &[0.0, 0.5, 1.0])
    .expect("spd");
    let pred = gpr.predict(&[0.5], 1, 1).expect("fitted");
    assert!(pred.mean[0].is_finite());
    assert!(pred.variance[0] > 0.0);
    let mut params = vec![0.0; gpr.num_params()];
    gpr.get_params(&mut params).expect("len");
    let mut grad = vec![0.0; params.len()];
    let nlml = gpr
        .value_and_gradient_into(&params, &mut grad)
        .expect("spd");
    assert!(nlml.is_finite());
    assert!(grad.iter().all(|g| g.is_finite()));
}

#[test]
fn matern_ard_equal_lengthscales_match_isotropic() {
    let ell = 1.25;
    let noise = 0.1;
    let nu = MaternNu::ThreeHalves;
    let x = [0.0, 0.5, 1.5, 0.0, 1.0, 0.5];
    let y = [0.2, -1.0, 0.7];
    let xs = [0.25, 1.0];
    let iso = Gpr::new(
        KernelSpec::from(MaternKernel::new(ell, nu).expect("valid")),
        GaussianLikelihood::new(noise).expect("valid"),
    )
    .with_optimizer(Fixed)
    .factor(&x, 3, 2, &y)
    .expect("spd");
    let mut ard = Gpr::new(
        KernelSpec::from(MaternArdKernel::new(&[ell, ell], nu).expect("valid")),
        GaussianLikelihood::new(noise).expect("valid"),
    )
    .with_optimizer(Fixed)
    .factor(&x, 3, 2, &y)
    .expect("spd");
    let p_iso = iso.predict(&xs, 1, 2).expect("fitted");
    let p_ard = ard.predict(&xs, 1, 2).expect("fitted");
    assert_close(p_ard.mean[0], p_iso.mean[0], TOL);
    assert_close(p_ard.variance[0], p_iso.variance[0], TOL);
    let mut params = vec![0.0; ard.num_params()];
    ard.get_params(&mut params).expect("len");
    let mut grad = vec![0.0; params.len()];
    let nlml = ard
        .value_and_gradient_into(&params, &mut grad)
        .expect("spd");
    assert!(nlml.is_finite());
    assert_eq!(params.len(), 3);
    assert!(grad.iter().all(|g| g.is_finite()));
}

#[test]
fn periodic_fits_and_predicts() {
    let mut gpr = Gpr::new(
        KernelSpec::from(PeriodicKernel::new(1.0, 2.0).expect("valid")),
        GaussianLikelihood::new(0.1).expect("valid"),
    )
    .with_optimizer(Fixed)
    .factor(&[0.0, 0.5, 1.0], 3, 1, &[0.0, 0.4, 0.1])
    .expect("spd");
    let pred = gpr.predict(&[2.0], 1, 1).expect("fitted");
    assert!(pred.mean[0].is_finite());
    assert!(pred.variance[0] > 0.0);
    let mut params = vec![0.0; gpr.num_params()];
    gpr.get_params(&mut params).expect("len");
    assert_eq!(params.len(), 3);
    let mut grad = vec![0.0; params.len()];
    let nlml = gpr
        .value_and_gradient_into(&params, &mut grad)
        .expect("spd");
    assert!(nlml.is_finite());
    assert!(grad.iter().all(|g| g.is_finite()));
}

#[test]
fn rational_quadratic_fits_and_predicts() {
    let mut gpr = Gpr::new(
        KernelSpec::from(RationalQuadraticKernel::new(1.0, 1.5).expect("valid")),
        GaussianLikelihood::new(0.1).expect("valid"),
    )
    .with_optimizer(Fixed)
    .factor(&[0.0, 0.5, 1.0], 3, 1, &[0.0, 0.4, 0.1])
    .expect("spd");
    let pred = gpr.predict(&[0.25], 1, 1).expect("fitted");
    assert!(pred.mean[0].is_finite());
    assert!(pred.variance[0] > 0.0);
    let mut params = vec![0.0; gpr.num_params()];
    gpr.get_params(&mut params).expect("len");
    assert_eq!(params.len(), 3);
    let mut grad = vec![0.0; params.len()];
    let nlml = gpr
        .value_and_gradient_into(&params, &mut grad)
        .expect("spd");
    assert!(nlml.is_finite());
    assert!(grad.iter().all(|g| g.is_finite()));
}

#[test]
fn rational_quadratic_ard_equal_lengthscales_match_isotropic() {
    let ell = 1.25;
    let alpha = 0.8;
    let noise = 0.1;
    let x = [0.0, 0.5, 1.5, 0.0, 1.0, 0.5];
    let y = [0.2, -1.0, 0.7];
    let xs = [0.25, 1.0];
    let iso = Gpr::new(
        KernelSpec::from(RationalQuadraticKernel::new(ell, alpha).expect("valid")),
        GaussianLikelihood::new(noise).expect("valid"),
    )
    .with_optimizer(Fixed)
    .factor(&x, 3, 2, &y)
    .expect("spd");
    let mut ard = Gpr::new(
        KernelSpec::from(RationalQuadraticArdKernel::new(&[ell, ell], alpha).expect("valid")),
        GaussianLikelihood::new(noise).expect("valid"),
    )
    .with_optimizer(Fixed)
    .factor(&x, 3, 2, &y)
    .expect("spd");
    let p_iso = iso.predict(&xs, 1, 2).expect("fitted");
    let p_ard = ard.predict(&xs, 1, 2).expect("fitted");
    assert_close(p_ard.mean[0], p_iso.mean[0], TOL);
    assert_close(p_ard.variance[0], p_iso.variance[0], TOL);
    let mut params = vec![0.0; ard.num_params()];
    ard.get_params(&mut params).expect("len");
    let mut grad = vec![0.0; params.len()];
    let nlml = ard
        .value_and_gradient_into(&params, &mut grad)
        .expect("spd");
    assert!(nlml.is_finite());
    assert_eq!(params.len(), 4);
    assert!(grad.iter().all(|g| g.is_finite()));
}

#[test]
fn fit_optimizes_and_keeps_l_and_alpha() {
    let gpr = rbf_gpr(2.0, 0.1)
        .fit(&[0.0, 1.0], 2, 1, &[0.5, -0.25])
        .expect("lbfgs");
    let alpha = gpr.alpha().to_vec();
    assert_eq!(alpha.len(), 2);
    assert!(alpha.iter().all(|a| a.is_finite()));
    {
        let ws = &gpr.store.buffers.core;
        assert_eq!(ws.k_matrix.nrows(), 2);
    }
    let pred = gpr.predict(&[0.5], 1, 1).expect("fitted");
    assert_eq!(pred.mean.len(), 1);
    assert!(pred.mean[0].is_finite());
    assert!(
        gpr.neg_log_marginal_likelihood()
            .expect("fitted")
            .is_finite()
    );
    let a = dense_a(
        gpr.kernel(),
        gpr.likelihood().noise_variance(),
        &[0.0, 1.0],
        2,
        1,
    );
    let restored = matvec_sym(&a, &alpha);
    assert_close(restored[0], 0.5, TOL);
    assert_close(restored[1], -0.25, TOL);
}

#[test]
fn fit_fixed_keeps_construction_params() {
    let gpr = rbf_gpr(1.25, 0.16)
        .with_optimizer(Fixed)
        .factor(&[0.0, 1.0], 2, 1, &[0.5, -0.25])
        .expect("spd");
    let mut params = [0.0; 2];
    gpr.get_params(&mut params).expect("len 2");
    assert_close(params[0], 1.25_f64.ln(), TOL);
    assert_close(params[1], 0.16_f64.ln(), TOL);
    let pred = gpr.predict(&[0.5], 1, 1).expect("fitted");
    assert_eq!(pred.mean.len(), 1);
}

#[test]
fn non_finite_optimize_result_restores_theta() {
    let mut gpr = rbf_gpr(1.25, 0.16)
        .with_optimizer(Fixed)
        .factor(&[0.0, 1.0], 2, 1, &[0.5, -0.25])
        .expect("spd");
    let kernel_before = gpr.kernel().clone();
    let likelihood_before = *gpr.likelihood();
    let mut before = [0.0; 2];
    gpr.get_params(&mut before).expect("len 2");
    let moved = [2.0_f64.ln(), 0.5_f64.ln()];
    let mut grad = [0.0; 2];
    gpr.value_and_gradient_into(&moved, &mut grad)
        .expect("moved");
    let mut mid = [0.0; 2];
    gpr.get_params(&mut mid).expect("len 2");
    assert!((mid[0] - before[0]).abs() > TOL);
    let err = gpr
        .fit_view()
        .commit_or_revert_optimize(
            kernel_before,
            likelihood_before,
            Ok(OptResult {
                params: moved.to_vec(),
                value: f64::NAN,
                iterations: 4,
            }),
        )
        .expect_err("nan nlml");
    assert!(matches!(
        err,
        GprError::OptimizationNotConverged { iterations: 4 }
    ));
    let mut after = [0.0; 2];
    gpr.get_params(&mut after).expect("len 2");
    assert_close(after[0], before[0], TOL);
    assert_close(after[1], before[1], TOL);
    assert_eq!(gpr.alpha().len(), 2);
    assert!(gpr.alpha().iter().all(|a| a.is_finite()));
}

#[test]
fn failed_optimize_err_restores_theta() {
    let mut gpr = rbf_gpr(1.25, 0.16)
        .with_optimizer(Fixed)
        .factor(&[0.0, 1.0], 2, 1, &[0.5, -0.25])
        .expect("spd");
    let kernel_before = gpr.kernel().clone();
    let likelihood_before = *gpr.likelihood();
    let mut before = [0.0; 2];
    gpr.get_params(&mut before).expect("len 2");
    let moved = [2.0_f64.ln(), 0.5_f64.ln()];
    let mut grad = [0.0; 2];
    gpr.value_and_gradient_into(&moved, &mut grad)
        .expect("moved");
    gpr.fit_view()
        .commit_or_revert_optimize(
            kernel_before,
            likelihood_before,
            Err(GprError::CholeskyFailed {
                jitter: 0.0,
                matrix_size: 2,
                stage: CholeskyStage::Fit,
            }),
        )
        .expect_err("chol");
    let mut after = [0.0; 2];
    gpr.get_params(&mut after).expect("len 2");
    assert_close(after[0], before[0], TOL);
    assert_close(after[1], before[1], TOL);
}

#[test]
fn factor_refit_keeps_construction_theta() {
    let mut gpr = rbf_gpr(1.25, 0.16)
        .with_optimizer(Fixed)
        .factor(&[0.0, 1.0], 2, 1, &[0.5, -0.25])
        .expect("spd");
    let mut before = [0.0; 2];
    gpr.get_params(&mut before).expect("len 2");
    gpr.refit().expect("refit");
    let mut after = [0.0; 2];
    gpr.get_params(&mut after).expect("len 2");
    assert_close(after[0], before[0], TOL);
    assert_close(after[1], before[1], TOL);
}

#[test]
fn lbfgs_knobs_affect_fit_and_refit() {
    let x = [0.0, 0.25, 0.6, 1.0];
    let y = [0.1, -0.4, 0.2, 0.8];
    let frozen = rbf_gpr(2.0, 0.2)
        .with_optimizer(Lbfgs::new().with_max_iterations(0))
        .fit(&x, 4, 1, &y)
        .expect("zero iters");
    let mut frozen_params = [0.0; 2];
    frozen.get_params(&mut frozen_params).expect("len 2");
    assert_close(frozen_params[0], 2.0_f64.ln(), TOL);
    assert_close(frozen_params[1], 0.2_f64.ln(), TOL);

    let searched = rbf_gpr(2.0, 0.2)
        .with_optimizer(
            Lbfgs::new()
                .with_max_iterations(80)
                .with_history_size(std::num::NonZeroUsize::MIN)
                .with_tolerance(1e-8)
                .expect("tol"),
        )
        .fit(&x, 4, 1, &y)
        .expect("search");
    let mut searched_params = [0.0; 2];
    searched.get_params(&mut searched_params).expect("len 2");
    assert!(
        frozen_params
            .iter()
            .zip(&searched_params)
            .any(|(a, b)| (a - b).abs() > 1e-9),
        "a real search should move θ: frozen={frozen_params:?} searched={searched_params:?}"
    );

    let mut restarted = rbf_gpr(2.0, 0.2)
        .with_optimizer(Lbfgs::new().with_restarts(std::num::NonZeroU32::MIN, 11))
        .fit(&x, 4, 1, &y)
        .expect("restarts");
    let nlml_fit = restarted.neg_log_marginal_likelihood().expect("nlml");
    restarted.refit().expect("refit");
    let nlml_refit = restarted.neg_log_marginal_likelihood().expect("nlml");
    assert!(
        nlml_refit <= nlml_fit + 1e-9,
        "refit should not raise NLML: fit={nlml_fit}, refit={nlml_refit}"
    );
}

#[test]
fn ncg_knobs_affect_fit_and_refit() {
    let x = [0.0, 0.25, 0.6, 1.0];
    let y = [0.1, -0.4, 0.2, 0.8];
    let frozen = rbf_gpr(2.0, 0.2)
        .with_optimizer(NonlinearCg::new().with_max_iterations(0))
        .fit(&x, 4, 1, &y)
        .expect("zero iters");
    let mut frozen_params = [0.0; 2];
    frozen.get_params(&mut frozen_params).expect("len 2");
    assert_close(frozen_params[0], 2.0_f64.ln(), TOL);
    assert_close(frozen_params[1], 0.2_f64.ln(), TOL);

    let searched = rbf_gpr(2.0, 0.2)
        .with_optimizer(
            NonlinearCg::new()
                .with_max_iterations(80)
                .with_tolerance(1e-8)
                .expect("tol"),
        )
        .fit(&x, 4, 1, &y)
        .expect("search");
    let mut searched_params = [0.0; 2];
    searched.get_params(&mut searched_params).expect("len 2");
    assert!(
        frozen_params
            .iter()
            .zip(&searched_params)
            .any(|(a, b)| (a - b).abs() > 1e-9),
        "a real search should move θ: frozen={frozen_params:?} searched={searched_params:?}"
    );

    let mut restarted = rbf_gpr(2.0, 0.2)
        .with_optimizer(NonlinearCg::new().with_restarts(std::num::NonZeroU32::MIN, 11))
        .fit(&x, 4, 1, &y)
        .expect("restarts");
    let nlml_fit = restarted.neg_log_marginal_likelihood().expect("nlml");
    restarted.refit().expect("refit");
    let nlml_refit = restarted.neg_log_marginal_likelihood().expect("nlml");
    assert!(
        nlml_refit <= nlml_fit + 1e-9,
        "refit should not raise NLML: fit={nlml_fit}, refit={nlml_refit}"
    );
}

#[test]
fn neldermead_fit_lowers_nlml() {
    let x = [0.0, 0.25, 0.6, 1.0];
    let y = [0.1, -0.4, 0.2, 0.8];
    let at_init = rbf_gpr(2.0, 0.2)
        .with_optimizer(Fixed)
        .factor(&x, 4, 1, &y)
        .expect("spd");
    let nlml_init = at_init.neg_log_marginal_likelihood().expect("init");
    let mut fitted = rbf_gpr(2.0, 0.2)
        .with_optimizer(NelderMead::new().with_max_iterations(80))
        .fit(&x, 4, 1, &y)
        .expect("nm");
    let nlml_fit = fitted.neg_log_marginal_likelihood().expect("nlml");
    assert!(
        nlml_fit < nlml_init,
        "NLML should fall: init={nlml_init}, fit={nlml_fit}"
    );
    fitted.refit().expect("refit");
    let nlml_refit = fitted.neg_log_marginal_likelihood().expect("nlml");
    assert!(
        nlml_refit <= nlml_fit + 1e-9,
        "refit should not raise NLML: fit={nlml_fit}, refit={nlml_refit}"
    );
}

#[derive(Clone, Debug)]
struct DummyOpt {
    calls: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl DummyOpt {
    fn new() -> Self {
        Self {
            calls: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    fn calls(&self) -> u64 {
        self.calls.load(std::sync::atomic::Ordering::SeqCst)
    }
}

impl<P: Objective> Optimizer<P> for DummyOpt {
    fn minimize(&self, objective: &mut P, init: &[f64]) -> Result<OptResult, GprError> {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let value = objective.value(init)?;
        Ok(OptResult {
            params: init.to_vec(),
            value,
            iterations: 0,
        })
    }
}

#[test]
fn custom_optimizer_minimize_is_called_on_fit_and_refit() {
    let dummy = DummyOpt::new();
    let mut fitted = rbf_gpr(1.0, 0.1)
        .with_optimizer(dummy.clone())
        .fit(&[0.0, 1.0], 2, 1, &[0.5, -0.25])
        .expect("fit");
    assert_eq!(dummy.calls(), 1);
    fitted.refit().expect("refit");
    assert_eq!(dummy.calls(), 2);
}

#[derive(Clone, Copy, Debug)]
struct IndexUsingOpt;

impl<P: Objective> Optimizer<P> for IndexUsingOpt {
    const USES_CHANGE_INDICES: bool = true;

    fn minimize(&self, objective: &mut P, init: &[f64]) -> Result<OptResult, GprError> {
        let value = objective.value(init)?;
        Ok(OptResult {
            params: init.to_vec(),
            value,
            iterations: 0,
        })
    }
}

#[test]
fn prefer_speed_is_incremental_only_with_uses_change_indices() {
    let is_incremental = |gpr: Gpr<Fixed>, uses_change_indices: bool| {
        let mut fitted = gpr.factor(&[0.0, 1.0], 2, 1, &[0.5, -0.25]).expect("spd");
        fitted
            .objective()
            .with_change_indices(uses_change_indices)
            .is_incremental()
    };
    let speed = rbf_gpr(1.0, 0.1).with_optimizer(Fixed);
    assert!(is_incremental(speed.clone(), true));
    assert!(!is_incremental(speed, false));
    let memory = rbf_gpr(1.0, 0.1).with_optimizer(Fixed).with_prefer_memory();
    assert!(!is_incremental(memory, true));
    let fitted = rbf_gpr(1.0, 0.1)
        .with_optimizer(IndexUsingOpt)
        .fit(&[0.0, 1.0], 2, 1, &[0.5, -0.25])
        .expect("fit");
    assert_eq!(fitted.n(), 2);
}

fn incremental_at_init(kernel: KernelSpec, noise: f64) -> FittedGpr<FastSimulatedAnnealing> {
    Gpr::new(kernel, GaussianLikelihood::new(noise).expect("valid"))
        .with_optimizer(FastSimulatedAnnealing::new().with_max_iterations(0))
        .fit(&[0.0, 0.8, 1.7], 3, 1, &[0.4, -0.2, 0.9])
        .expect("spd")
}

fn full_at_init(kernel: KernelSpec, noise: f64) -> FittedGpr<Fixed> {
    Gpr::new(kernel, GaussianLikelihood::new(noise).expect("valid"))
        .with_optimizer(Fixed)
        .factor(&[0.0, 0.8, 1.7], 3, 1, &[0.4, -0.2, 0.9])
        .expect("spd")
}

fn assert_incremental_matches_full(kernel: KernelSpec, noise: f64, change: usize) {
    let mut full = full_at_init(kernel.clone(), noise);
    let mut incr = incremental_at_init(kernel, noise);
    let n = full.num_params();
    let mut start = vec![0.0; n];
    full.get_params(&mut start).expect("len");
    let mut params = start.clone();
    params[change] += 0.15;
    let v_full = {
        let mut obj = full.objective();
        obj.value(&params).expect("full")
    };
    let v_incr = {
        let mut obj = incr.objective().with_change_indices(true);
        let primed = obj.value(&start).expect("prime");
        assert!(primed.is_finite());
        IncrementalObjective::value_with_changes(&mut obj, &params, &[change]).expect("incr")
    };
    assert_close(v_incr, v_full, TOL);
}

#[test]
fn incremental_rbf_matches_full_value() {
    assert_incremental_matches_full(
        KernelSpec::from(RbfKernel::new(1.25).expect("valid")),
        0.16,
        0,
    );
}

#[test]
fn incremental_sum_matches_full_value() {
    let kernel = KernelSpec::from(RbfKernel::new(1.25).expect("valid"))
        + KernelSpec::from(RbfKernel::new(0.8).expect("valid"));
    assert_incremental_matches_full(kernel, 0.16, 1);
}

/// Cached non-dirty leaf Grams must still match a full rebuild after a later
/// coordinate step, and after an eval that is not a neighbor of the last eval
/// (FSA rejection).
#[test]
fn incremental_cached_leaves_match_full_after_later_steps() {
    let kernel = KernelSpec::from(RbfKernel::new(1.25).expect("valid"))
        + KernelSpec::from(RbfKernel::new(0.8).expect("valid"));
    let mut full = full_at_init(kernel.clone(), 0.16);
    let mut incr = incremental_at_init(kernel, 0.16);
    let n = full.num_params();
    let mut start = vec![0.0; n];
    full.get_params(&mut start).expect("len");
    let mut after_first = start.clone();
    after_first[0] += 0.15;
    let mut after_second = after_first.clone();
    after_second[1] += 0.2;
    let mut rejected_then_other = start.clone();
    rejected_then_other[1] += 0.2;
    {
        let mut obj = incr.objective().with_change_indices(true);
        obj.value(&start).expect("prime");
        IncrementalObjective::value_with_changes(&mut obj, &after_first, &[0]).expect("leaf 0");
        let sequential = IncrementalObjective::value_with_changes(&mut obj, &after_second, &[1])
            .expect("leaf 1 after accept");
        let v_full = full.objective().value(&after_second).expect("full seq");
        assert_close(sequential, v_full, TOL);
    }
    let mut incr = incremental_at_init(
        KernelSpec::from(RbfKernel::new(1.25).expect("valid"))
            + KernelSpec::from(RbfKernel::new(0.8).expect("valid")),
        0.16,
    );
    let v_incr = {
        let mut obj = incr.objective().with_change_indices(true);
        obj.value(&start).expect("prime");
        IncrementalObjective::value_with_changes(&mut obj, &after_first, &[0]).expect("rejected");
        IncrementalObjective::value_with_changes(&mut obj, &rejected_then_other, &[0, 1])
            .expect("other coord from start")
    };
    let v_full = full
        .objective()
        .value(&rejected_then_other)
        .expect("full reject");
    assert_close(v_incr, v_full, TOL);
}

#[test]
fn incremental_rejects_an_unlisted_change() {
    let kernel = KernelSpec::from(RbfKernel::new(1.25).expect("valid"))
        + KernelSpec::from(RbfKernel::new(0.8).expect("valid"));
    let mut incr = incremental_at_init(kernel, 0.16);
    let n = incr.num_params();
    let mut start = vec![0.0; n];
    incr.get_params(&mut start).expect("len");
    let mut first = start.clone();
    first[0] += 0.15;
    let mut both_moved = start.clone();
    both_moved[1] += 0.2;
    let mut obj = incr.objective().with_change_indices(true);
    obj.value(&start).expect("prime");
    IncrementalObjective::value_with_changes(&mut obj, &first, &[0]).expect("leaf 0");
    assert!(matches!(
        IncrementalObjective::value_with_changes(&mut obj, &both_moved, &[1]),
        Err(GprError::IndexOutOfRange { .. })
    ));
}

#[test]
fn incremental_sum_jitter_retry_matches_full() {
    let kernel = KernelSpec::custom(IndefiniteLeaf)
        + KernelSpec::from(ConstantKernel::new(0.5).expect("valid"));
    let jitter = JitterPolicy::fixed(1.0).expect("valid");
    let likelihood = GaussianLikelihood::new(0.1).expect("valid");
    let x = [0.0, 1.0];
    let y = [0.0, 1.0];
    let mut incr = Gpr::new(kernel.clone(), likelihood)
        .with_optimizer(FastSimulatedAnnealing::new().with_max_iterations(0))
        .with_jitter_policy(jitter)
        .fit(&x, 2, 1, &y)
        .expect("incr");
    let mut full = Gpr::new(kernel, likelihood)
        .with_optimizer(Fixed)
        .with_jitter_policy(jitter)
        .factor(&x, 2, 1, &y)
        .expect("full");
    let mut params = vec![0.0; full.num_params()];
    full.get_params(&mut params).expect("len");
    let v_full = full.objective().value(&params).expect("full");
    let v_incr = incr
        .objective()
        .with_change_indices(true)
        .value(&params)
        .expect("incr");
    assert_close(v_incr, v_full, TOL);
}

#[test]
fn incremental_product_matches_full_value() {
    let kernel = KernelSpec::from(RbfKernel::new(1.25).expect("valid"))
        * KernelSpec::from(ConstantKernel::new(1.4).expect("valid"));
    assert_incremental_matches_full(kernel, 0.16, 1);
}

#[test]
fn incremental_noise_only_matches_full_value() {
    let kernel = KernelSpec::from(RbfKernel::new(1.25).expect("valid"));
    let n = kernel.num_params();
    assert_incremental_matches_full(kernel, 0.16, n);
}

#[test]
fn incremental_rejects_empty_duplicate_and_oob_indices() {
    let mut incr =
        incremental_at_init(KernelSpec::from(RbfKernel::new(1.25).expect("valid")), 0.16);
    let n = incr.num_params();
    let mut params = vec![0.0; n];
    incr.get_params(&mut params).expect("len");
    let mut obj = incr.objective().with_change_indices(true);
    obj.value(&params).expect("prime");
    assert!(matches!(
        obj.value_at_changes(&params, &[]),
        Err(GprError::IndexOutOfRange { .. })
    ));
    assert!(matches!(
        obj.value_at_changes(&params, &[0, 0]),
        Err(GprError::IndexOutOfRange { .. })
    ));
    assert!(matches!(
        obj.value_at_changes(&params, &[n]),
        Err(GprError::IndexOutOfRange { .. })
    ));
}

#[test]
fn fsa_speed_and_memory_poles_fit_and_match() {
    let kernel = KernelSpec::from(RbfKernel::new(4.0).expect("valid"));
    let likelihood = GaussianLikelihood::new(1.0).expect("valid");
    let x = [0.0, 1.0];
    let y = [0.5, -0.25];
    let start = Gpr::new(kernel.clone(), likelihood)
        .with_optimizer(Fixed)
        .factor(&x, 2, 1, &y)
        .expect("spd")
        .neg_log_marginal_likelihood()
        .expect("start");
    let speed = Gpr::new(kernel.clone(), likelihood)
        .with_optimizer(FastSimulatedAnnealing::new().with_seed(7))
        .with_prefer_speed();
    assert_eq!(speed.cholesky_buffer(), CholeskyBuffer::Retain);
    let memory = Gpr::new(kernel, likelihood)
        .with_optimizer(FastSimulatedAnnealing::new().with_seed(7))
        .with_prefer_memory();
    assert_eq!(memory.cholesky_buffer(), CholeskyBuffer::Reuse);
    let speed = speed.fit(&x, 2, 1, &y).expect("speed");
    let memory = memory.fit(&x, 2, 1, &y).expect("memory");
    let speed_nlml = speed.neg_log_marginal_likelihood().expect("speed nlml");
    let memory_nlml = memory.neg_log_marginal_likelihood().expect("memory nlml");
    assert!(speed_nlml < start, "speed={speed_nlml}, start={start}");
    assert!(memory_nlml < start, "memory={memory_nlml}, start={start}");
    assert_close(speed_nlml, memory_nlml, TOL);
    let mut speed_theta = vec![0.0; speed.num_params()];
    let mut memory_theta = vec![0.0; memory.num_params()];
    speed.get_params(&mut speed_theta).expect("speed θ");
    memory.get_params(&mut memory_theta).expect("memory θ");
    for (a, b) in speed_theta.iter().zip(memory_theta.iter()) {
        assert_close(*a, *b, TOL);
    }
}

#[test]
fn fast_approx_fit_nlml_does_not_rise() {
    let x = [0.0, 0.8, 1.7];
    let y = [0.4, -0.2, 0.9];
    let kernel = || KernelSpec::from(RbfKernel::new(1.25).expect("ell"));
    let noise = || GaussianLikelihood::new(0.16).expect("noise");
    let mut factored = Gpr::new(kernel(), noise())
        .with_math(KernelExp::FastApprox)
        .with_optimizer(Fixed)
        .factor(&x, 3, 1, &y)
        .map_err(|(_, err)| err)
        .expect("factor");
    let mut params = [0.0; 2];
    factored.get_params(&mut params).expect("len");
    let mut grad = [0.0; 2];
    let value = factored
        .value_and_gradient_into(&params, &mut grad)
        .expect("grad");
    let h = 1e-5;
    let mut dummy = [0.0; 2];
    for i in 0..2 {
        let mut plus = params;
        let mut minus = params;
        plus[i] += h;
        minus[i] -= h;
        let v_plus = factored
            .value_and_gradient_into(&plus, &mut dummy)
            .expect("plus");
        let v_minus = factored
            .value_and_gradient_into(&minus, &mut dummy)
            .expect("minus");
        let fd = (v_plus - v_minus) / (2.0 * h);
        let scale = fd.abs().max(1.0);
        assert!(
            (grad[i] - fd).abs() <= 1e-4 * scale,
            "param {i}: analytic={}, fd={}",
            grad[i],
            fd
        );
    }
    let _ = value;
    let start = Gpr::new(kernel(), noise())
        .with_math(KernelExp::FastApprox)
        .with_optimizer(Fixed)
        .factor(&x, 3, 1, &y)
        .map_err(|(_, err)| err)
        .expect("factor")
        .neg_log_marginal_likelihood()
        .expect("start");
    let check = |end: f64| {
        assert!(end.is_finite(), "nlml={end}");
        assert!(end <= start, "end={end} start={start}");
    };
    check(
        Gpr::new(kernel(), noise())
            .with_math(KernelExp::FastApprox)
            .fit(&x, 3, 1, &y)
            .map_err(|(_, err)| err)
            .expect("lbfgs")
            .neg_log_marginal_likelihood()
            .expect("end"),
    );
    check(
        Gpr::new(kernel(), noise())
            .with_math(KernelExp::FastApprox)
            .with_optimizer(NonlinearCg::new())
            .fit(&x, 3, 1, &y)
            .map_err(|(_, err)| err)
            .expect("ncg")
            .neg_log_marginal_likelihood()
            .expect("end"),
    );
    check(
        Gpr::new(kernel(), noise())
            .with_math(KernelExp::FastApprox)
            .with_optimizer(NelderMead::new())
            .fit(&x, 3, 1, &y)
            .map_err(|(_, err)| err)
            .expect("nelder")
            .neg_log_marginal_likelihood()
            .expect("end"),
    );
    check(
        Gpr::new(kernel(), noise())
            .with_math(KernelExp::FastApprox)
            .with_optimizer(Newton::new())
            .fit(&x, 3, 1, &y)
            .map_err(|(_, err)| err)
            .expect("newton")
            .neg_log_marginal_likelihood()
            .expect("end"),
    );
    check(
        Gpr::new(kernel(), noise())
            .with_math(KernelExp::FastApprox)
            .with_optimizer(FastSimulatedAnnealing::new())
            .fit(&x, 3, 1, &y)
            .map_err(|(_, err)| err)
            .expect("fsa")
            .neg_log_marginal_likelihood()
            .expect("end"),
    );
}
