//! Batch exact Gaussian process: `A = K + σn² I`, LLT, and `α`.

use std::fmt;

use dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt;
use faer::linalg::cholesky::llt::factor::{LltError, LltRegularization};
use faer::{Mat, MatMut, MatRef, Par};

use crate::error::{CholeskyStage, GpError};
use crate::kernel::{CompiledKernel, KernelSpec, Triangle};
use crate::likelihood::GaussianLikelihood;
use crate::precision::DoublePrecision;
use crate::transform::{IdentityInput, IdentityTarget, TargetTransform, Transform};
use crate::workspace::Workspace;

/// Which predictive variance [`Prediction`] reports.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VarianceKind {
    /// Variance of the latent function `f*`, without observation noise.
    Latent,
    /// Variance of a new observation `y*`, including `σn²`. This is the default.
    Observation,
}

/// Options for [`ExactGP::predict`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PredictOptions {
    /// Which variance to return. Defaults to [`VarianceKind::Observation`].
    pub variance_kind: VarianceKind,
}

impl Default for PredictOptions {
    fn default() -> Self {
        Self {
            variance_kind: VarianceKind::Observation,
        }
    }
}

/// Predictive mean and (diagonal) variance at the query points.
#[derive(Clone, Debug, PartialEq)]
pub struct Prediction {
    /// Predictive mean on the original target scale.
    pub mean: Vec<f64>,
    /// Predictive variance on the original target scale.
    pub variance: Vec<f64>,
    /// Whether [`Self::variance`] is latent or observation variance.
    pub variance_kind: VarianceKind,
}

/// Exact GP with fixed hyperparameters.
///
/// [`Self::fit`] builds the lower triangle of `A = K + σn² I`, factors it
/// in place as `L Lᵀ`, and solves `A α = y`. `L` lives in the workspace;
/// `α` is kept on the model. [`Self::predict`] returns the mean and a
/// diagonal variance. Input and target transforms default to identity.
///
/// # Examples
///
/// ```rust
/// use gprx::kernel::{KernelSpec, RbfKernel};
/// use gprx::{ExactGP, GaussianLikelihood};
///
/// # fn main() -> Result<(), gprx::GpError> {
/// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
/// let likelihood = GaussianLikelihood::new(0.1)?;
/// let mut gp = ExactGP::new(kernel, likelihood);
/// // Column-major `X` with n = 2 points and d = 1 feature.
/// gp.fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0])?;
/// let pred = gp.predict(&[0.5], 1, 1)?;
/// assert_eq!(pred.mean.len(), 1);
/// # Ok(())
/// # }
/// ```
pub struct ExactGP {
    kernel: KernelSpec,
    compiled: Option<CompiledKernel>,
    likelihood: GaussianLikelihood,
    x_transform: Box<dyn Transform>,
    y_transform: Box<dyn TargetTransform>,
    workspace: Option<Workspace<DoublePrecision>>,
    x: Option<Mat<f64>>,
    #[allow(dead_code)] // MLL (P1A-9)
    y: Option<Vec<f64>>,
    alpha: Option<Vec<f64>>,
    fitted: bool,
    n: usize,
    d: usize,
}

impl fmt::Debug for ExactGP {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExactGP")
            .field("fitted", &self.fitted)
            .field("n", &self.n)
            .field("d", &self.d)
            .field("kernel", &self.kernel)
            .field("likelihood", &self.likelihood)
            .finish_non_exhaustive()
    }
}

impl ExactGP {
    /// Builds an unfitted model that owns the kernel and observation noise.
    ///
    /// Input and target maps default to identity. Call
    /// [`Self::with_input_transform`] / [`Self::with_target_transform`] before
    /// [`Self::fit`] to standardize.
    pub fn new(kernel: KernelSpec, likelihood: GaussianLikelihood) -> Self {
        Self {
            kernel,
            compiled: None,
            likelihood,
            x_transform: Box::new(IdentityInput),
            y_transform: Box::new(IdentityTarget),
            workspace: None,
            x: None,
            y: None,
            alpha: None,
            fitted: false,
            n: 0,
            d: 0,
        }
    }

    /// Replaces the input (`X`) transform. Intended to be called before fit.
    pub fn with_input_transform(mut self, transform: impl Transform + 'static) -> Self {
        self.x_transform = Box::new(transform);
        self
    }

    /// Replaces the target (`y`) transform. Intended to be called before fit.
    pub fn with_target_transform(mut self, transform: impl TargetTransform + 'static) -> Self {
        self.y_transform = Box::new(transform);
        self
    }

    /// Returns whether the last [`Self::fit`] produced `L` and `α`.
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }

    /// Returns the number of training points from the last successful fit.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Returns the feature dimension from the last successful fit.
    pub fn d(&self) -> usize {
        self.d
    }

    /// Returns the kernel whose hyperparameters this model owns.
    pub fn kernel(&self) -> &KernelSpec {
        &self.kernel
    }

    /// Returns the observation-noise model.
    pub fn likelihood(&self) -> &GaussianLikelihood {
        &self.likelihood
    }

    /// Returns `α = A⁻¹ y` from the last successful fit.
    ///
    /// # Errors
    ///
    /// Returns [`GpError::NotFitted`] if [`Self::fit`] has not succeeded.
    pub fn alpha(&self) -> Result<&[f64], GpError> {
        self.alpha.as_deref().ok_or(GpError::NotFitted)
    }

    /// Factors `A = K + σn² I` and solves `A α = y`.
    ///
    /// `x` is column-major with `n_rows` points and `n_cols` features. A
    /// previous successful fit is kept when this call fails validation.
    /// Cholesky failure clears the fitted flag and does not leave a usable
    /// `α`.
    ///
    /// # Errors
    ///
    /// Returns [`GpError::EmptyInput`] if `n_rows` or `n_cols` is zero,
    /// [`GpError::InvalidHyperparameter`] if `x` or `y` has the wrong length,
    /// [`GpError::NonFiniteInput`] if a value is `NaN` or `Inf`, or
    /// [`GpError::CholeskyFailed`] if `A` cannot be factored.
    pub fn fit(
        &mut self,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
        y: &[f64],
    ) -> Result<(), GpError> {
        validate_training(x, n_rows, n_cols, y)?;
        self.clear_solution();
        self.prepare_workspace(n_rows)?;
        let mut x_buf = x.to_vec();
        self.x_transform.fit(&x_buf, n_rows, n_cols)?;
        self.x_transform.apply(&mut x_buf, n_rows, n_cols)?;
        let mut y_buf = y.to_vec();
        self.y_transform.fit(&y_buf)?;
        self.y_transform.transform(&mut y_buf)?;
        let x_mat = pack_points(&x_buf, n_rows, n_cols);
        let compiled = self.kernel.compile();
        {
            let ws = workspace_mut(&mut self.workspace)?;
            fill_squared_euclidean(x_mat.as_ref(), ws.dist_cache.as_mut());
            compiled.apply(
                ws.dist_cache.as_ref(),
                ws.k_matrix.as_mut(),
                Triangle::Lower,
                ws.exp_buf.as_mut(),
            )?;
            add_noise_to_diag(ws.k_matrix.as_mut(), self.likelihood.noise_variance());
        }
        let mut rhs = Mat::from_fn(n_rows, 1, |i, _| y_buf[i]);
        {
            let ws = workspace_mut(&mut self.workspace)?;
            cholesky_and_solve(
                &mut ws.k_matrix,
                &mut rhs,
                &mut ws.faer_scratch,
                0.0,
                CholeskyStage::Fit,
            )?;
        }
        self.compiled = Some(compiled);
        self.x = Some(x_mat);
        self.y = Some(y_buf);
        self.alpha = Some((0..n_rows).map(|i| rhs[(i, 0)]).collect());
        self.n = n_rows;
        self.d = n_cols;
        self.fitted = true;
        Ok(())
    }

    /// Predicts at `xs` with [`PredictOptions::default`] (observation variance).
    ///
    /// `xs` is column-major with `n_rows` query points and `n_cols` features.
    ///
    /// # Errors
    ///
    /// Returns [`GpError::NotFitted`] if [`Self::fit`] has not succeeded,
    /// [`GpError::DimensionMismatch`] if `n_cols` differs from the training
    /// features, [`GpError::EmptyInput`] if a dimension is zero, or
    /// [`GpError::InvalidHyperparameter`] / [`GpError::NonFiniteInput`] for a
    /// badly packed or non-finite `xs`.
    pub fn predict(&self, xs: &[f64], n_rows: usize, n_cols: usize) -> Result<Prediction, GpError> {
        self.predict_with(xs, n_rows, n_cols, PredictOptions::default())
    }

    /// Predicts at `xs` with an explicit variance kind.
    ///
    /// Latent variance is `k(x*, x*) - ‖L⁻¹ k_*‖²`. Observation variance adds
    /// `σn²` in the transformed space, then both mean and variance are mapped
    /// back by the target transform.
    ///
    /// # Errors
    ///
    /// Same as [`Self::predict`].
    pub fn predict_with(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        options: PredictOptions,
    ) -> Result<Prediction, GpError> {
        if !self.fitted {
            return Err(GpError::NotFitted);
        }
        if n_cols != self.d {
            return Err(GpError::DimensionMismatch {
                x_dim: n_cols,
                expected_dim: self.d,
            });
        }
        validate_query(xs, n_rows, n_cols)?;
        let compiled = self.compiled.as_ref().ok_or(GpError::NotFitted)?;
        let x_train = self.x.as_ref().ok_or(GpError::NotFitted)?;
        let alpha = self.alpha.as_deref().ok_or(GpError::NotFitted)?;
        let ws = self.workspace.as_ref().ok_or(GpError::NotFitted)?;
        let mut xs_buf = xs.to_vec();
        self.x_transform.apply(&mut xs_buf, n_rows, n_cols)?;
        let x_test = pack_points(&xs_buf, n_rows, n_cols);
        let n = self.n;
        let m = n_rows;
        let mut dist = Mat::zeros(n, m);
        let mut k_star = Mat::zeros(n, m);
        let mut scratch = Mat::zeros(n, m);
        fill_squared_euclidean_cross(x_train.as_ref(), x_test.as_ref(), dist.as_mut());
        compiled.apply_cross(dist.as_ref(), k_star.as_mut(), scratch.as_mut())?;
        let mut mean = vec![0.0; m];
        for col in 0..m {
            let mut sum = 0.0;
            for row in 0..n {
                sum += k_star[(row, col)] * alpha[row];
            }
            mean[col] = sum;
        }
        faer::linalg::triangular_solve::solve_lower_triangular_in_place(
            ws.k_matrix.as_ref(),
            k_star.as_mut(),
            Par::Seq,
        );
        let mut kss = vec![0.0; m];
        compiled.fill_diag(&mut kss)?;
        let noise = self.likelihood.noise_variance();
        let mut variance = vec![0.0; m];
        for col in 0..m {
            let mut vnorm = 0.0;
            for row in 0..n {
                let v = k_star[(row, col)];
                vnorm += v * v;
            }
            let mut latent = kss[col] - vnorm;
            if latent < 0.0 {
                latent = 0.0;
            }
            variance[col] = match options.variance_kind {
                VarianceKind::Latent => latent,
                VarianceKind::Observation => latent + noise,
            };
        }
        self.y_transform.inverse_transform_mean(&mut mean)?;
        self.y_transform.inverse_transform_variance(&mut variance)?;
        Ok(Prediction {
            mean,
            variance,
            variance_kind: options.variance_kind,
        })
    }

    fn clear_solution(&mut self) {
        self.fitted = false;
        self.compiled = None;
        self.x = None;
        self.y = None;
        self.alpha = None;
        self.n = 0;
        self.d = 0;
    }

    fn prepare_workspace(&mut self, n: usize) -> Result<(), GpError> {
        match &mut self.workspace {
            Some(ws) => ws.ensure(n),
            None => {
                self.workspace = Some(Workspace::new(n)?);
                Ok(())
            }
        }
    }
}

fn workspace_mut(
    workspace: &mut Option<Workspace<DoublePrecision>>,
) -> Result<&mut Workspace<DoublePrecision>, GpError> {
    workspace.as_mut().ok_or(GpError::EmptyInput)
}

fn validate_training(x: &[f64], n_rows: usize, n_cols: usize, y: &[f64]) -> Result<(), GpError> {
    if n_rows == 0 || n_cols == 0 {
        return Err(GpError::EmptyInput);
    }
    let expected_x = n_rows.checked_mul(n_cols).ok_or(GpError::EmptyInput)?;
    if x.len() != expected_x {
        return Err(GpError::InvalidHyperparameter {
            reason: format!("expected {expected_x} feature values, got {}", x.len()),
        });
    }
    if y.len() != n_rows {
        return Err(GpError::InvalidHyperparameter {
            reason: format!("expected {n_rows} targets, got {}", y.len()),
        });
    }
    if x.iter().any(|v| !v.is_finite()) || y.iter().any(|v| !v.is_finite()) {
        return Err(GpError::NonFiniteInput);
    }
    Ok(())
}

fn validate_query(xs: &[f64], n_rows: usize, n_cols: usize) -> Result<(), GpError> {
    if n_rows == 0 || n_cols == 0 {
        return Err(GpError::EmptyInput);
    }
    let expected = n_rows.checked_mul(n_cols).ok_or(GpError::EmptyInput)?;
    if xs.len() != expected {
        return Err(GpError::InvalidHyperparameter {
            reason: format!("expected {expected} feature values, got {}", xs.len()),
        });
    }
    if xs.iter().any(|v| !v.is_finite()) {
        return Err(GpError::NonFiniteInput);
    }
    Ok(())
}

fn pack_points(x: &[f64], n_rows: usize, n_cols: usize) -> Mat<f64> {
    Mat::from_fn(n_rows, n_cols, |row, col| x[col * n_rows + row])
}

fn fill_squared_euclidean(x: MatRef<'_, f64>, mut dist: MatMut<'_, f64>) {
    let n = x.nrows();
    let d = x.ncols();
    for col in 0..n {
        for row in col..n {
            let mut sum = 0.0;
            for dim in 0..d {
                let diff = x[(row, dim)] - x[(col, dim)];
                sum += diff * diff;
            }
            dist[(row, col)] = sum;
            dist[(col, row)] = sum;
        }
    }
}

fn fill_squared_euclidean_cross(
    x_train: MatRef<'_, f64>,
    x_test: MatRef<'_, f64>,
    mut dist: MatMut<'_, f64>,
) {
    let n = x_train.nrows();
    let m = x_test.nrows();
    let d = x_train.ncols();
    for col in 0..m {
        for row in 0..n {
            let mut sum = 0.0;
            for dim in 0..d {
                let diff = x_train[(row, dim)] - x_test[(col, dim)];
                sum += diff * diff;
            }
            dist[(row, col)] = sum;
        }
    }
}

fn add_noise_to_diag(mut k: MatMut<'_, f64>, noise: f64) {
    let n = k.nrows();
    for i in 0..n {
        k[(i, i)] += noise;
    }
}

/// Factors `A` in place as `L Lᵀ` and overwrites `rhs` with `A⁻¹ rhs`.
///
/// P1A-18 can call this on the same `Workspace` buffers as [`ExactGP::fit`].
pub(crate) fn cholesky_and_solve(
    a: &mut Mat<f64>,
    rhs: &mut Mat<f64>,
    scratch: &mut MemBuffer,
    jitter: f64,
    stage: CholeskyStage,
) -> Result<(), GpError> {
    let n = a.nrows();
    let regularization = LltRegularization {
        dynamic_regularization_delta: jitter,
        dynamic_regularization_epsilon: 0.0,
    };
    {
        let stack = MemStack::new(scratch);
        match llt::factor::cholesky_in_place(
            a.as_mut(),
            regularization,
            Par::Seq,
            stack,
            Default::default(),
        ) {
            Ok(_) => {}
            Err(LltError::NonPositivePivot { .. }) => {
                return Err(GpError::CholeskyFailed {
                    jitter,
                    matrix_size: n,
                    stage,
                });
            }
        }
    }
    let stack = MemStack::new(scratch);
    llt::solve::solve_in_place(a.as_ref(), rhs.as_mut(), Par::Seq, stack);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{ExactGP, cholesky_and_solve, pack_points};
    use crate::error::{CholeskyStage, GpError};
    use crate::kernel::{KernelSpec, RbfKernel, Triangle};
    use crate::likelihood::GaussianLikelihood;
    use crate::precision::DoublePrecision;
    use crate::transform::{StandardizeTarget, TargetTransform};
    use crate::workspace::Workspace;
    use faer::Mat;

    const TOL: f64 = 1e-9;

    fn assert_close(actual: f64, expected: f64) {
        let scale = expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= TOL * scale,
            "actual={actual}, expected={expected}"
        );
    }

    fn assert_send_sync<T: Send + Sync>() {}

    fn rbf_gp(ell: f64, noise: f64) -> ExactGP {
        ExactGP::new(
            KernelSpec::from(RbfKernel::new(ell).expect("valid")),
            GaussianLikelihood::new(noise).expect("valid"),
        )
    }

    fn dense_a(kernel: &KernelSpec, noise: f64, x: &[f64], n: usize, d: usize) -> Mat<f64> {
        let compiled = kernel.compile();
        let x_mat = pack_points(x, n, d);
        let mut dist = Mat::zeros(n, n);
        super::fill_squared_euclidean(x_mat.as_ref(), dist.as_mut());
        let mut k = Mat::zeros(n, n);
        let mut scratch = Mat::zeros(n, n);
        compiled
            .apply(dist.as_ref(), k.as_mut(), Triangle::Full, scratch.as_mut())
            .expect("shape");
        super::add_noise_to_diag(k.as_mut(), noise);
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
        assert_send_sync::<ExactGP>();
        assert_send_sync::<super::Prediction>();
        assert_send_sync::<super::VarianceKind>();
        assert_send_sync::<super::PredictOptions>();
    }

    #[test]
    fn fit_solves_a_alpha_equals_y() {
        let mut gp = rbf_gp(1.25, 0.1);
        let x = [0.0, 0.5, 1.5, 0.0, 1.0, 0.5];
        let y = [0.2, -1.0, 0.7];
        gp.fit(&x, 3, 2, &y).expect("spd");
        assert!(gp.is_fitted());
        assert_eq!(gp.n(), 3);
        assert_eq!(gp.d(), 2);
        let a = dense_a(gp.kernel(), gp.likelihood().noise_variance(), &x, 3, 2);
        let alpha = gp.alpha().expect("fitted");
        let restored = matvec_sym(&a, alpha);
        for i in 0..3 {
            assert_close(restored[i], y[i]);
        }
        let ws = gp.workspace.as_ref().expect("workspace");
        let l = copy_lower(ws.k_matrix.as_ref());
        let a_from_l = &l * l.transpose();
        for col in 0..3 {
            for row in col..3 {
                assert_close(a_from_l[(row, col)], a[(row, col)]);
            }
        }
    }

    #[test]
    fn refit_replaces_size_and_still_solves() {
        let mut gp = rbf_gp(1.25, 0.1);
        gp.fit(&[0.0, 0.5, 1.5, 0.0, 1.0, 0.5], 3, 2, &[0.2, -1.0, 0.7])
            .expect("spd");
        gp.fit(&[0.0, 1.0], 2, 1, &[0.5, -0.25]).expect("refit");
        assert_eq!(gp.n(), 2);
        assert_eq!(gp.d(), 1);
        let a = dense_a(
            gp.kernel(),
            gp.likelihood().noise_variance(),
            &[0.0, 1.0],
            2,
            1,
        );
        let alpha = gp.alpha().expect("fitted");
        let restored = matvec_sym(&a, alpha);
        assert_close(restored[0], 0.5);
        assert_close(restored[1], -0.25);
    }

    #[test]
    fn fit_n_one_matches_scalar_solve() {
        let noise = 0.25;
        let mut gp = rbf_gp(1.0, noise);
        gp.fit(&[0.0], 1, 1, &[2.0]).expect("spd");
        let a = 1.0 + noise;
        assert_close(gp.alpha().expect("fitted")[0], 2.0 / a);
    }

    #[test]
    fn validation_error_keeps_previous_fit() {
        let mut gp = rbf_gp(1.0, 0.1);
        gp.fit(&[0.0, 1.0], 2, 1, &[1.0, 2.0]).expect("spd");
        let alpha = gp.alpha().expect("fitted").to_vec();
        assert!(matches!(
            gp.fit(&[0.0], 0, 1, &[]),
            Err(GpError::EmptyInput)
        ));
        assert!(gp.is_fitted());
        assert_close(gp.alpha().expect("fitted")[0], alpha[0]);
        assert_close(gp.alpha().expect("fitted")[1], alpha[1]);
    }

    #[test]
    fn indefinite_matrix_returns_cholesky_failed() {
        let mut a = faer::mat![[1.0, 2.0], [2.0, 1.0]];
        let mut rhs = faer::mat![[1.0], [0.0]];
        let mut ws = Workspace::<DoublePrecision>::new(2).expect("n > 0");
        let err = cholesky_and_solve(
            &mut a,
            &mut rhs,
            &mut ws.faer_scratch,
            0.0,
            CholeskyStage::Fit,
        )
        .expect_err("indefinite");
        assert!(matches!(
            err,
            GpError::CholeskyFailed {
                stage: CholeskyStage::Fit,
                matrix_size: 2,
                jitter: _,
            }
        ));
    }

    #[test]
    fn fit_rejects_bad_shapes_and_non_finite() {
        let mut gp = rbf_gp(1.0, 0.1);
        assert!(matches!(
            gp.fit(&[0.0], 2, 1, &[0.0, 1.0]),
            Err(GpError::InvalidHyperparameter { .. })
        ));
        assert!(matches!(
            gp.fit(&[0.0, 1.0], 2, 1, &[0.0]),
            Err(GpError::InvalidHyperparameter { .. })
        ));
        assert!(matches!(
            gp.fit(&[0.0, f64::NAN], 2, 1, &[0.0, 1.0]),
            Err(GpError::NonFiniteInput)
        ));
        assert!(!gp.is_fitted());
    }

    #[test]
    fn unfitted_alpha_is_not_fitted() {
        let gp = rbf_gp(1.0, 0.1);
        assert!(matches!(gp.alpha(), Err(GpError::NotFitted)));
        assert!(!gp.is_fitted());
    }

    #[test]
    fn predict_rejects_unfitted_and_wrong_dim() {
        let mut gp = rbf_gp(1.0, 0.1);
        assert!(matches!(gp.predict(&[0.0], 1, 1), Err(GpError::NotFitted)));
        gp.fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0]).expect("spd");
        assert!(matches!(
            gp.predict(&[0.0, 1.0], 1, 2),
            Err(GpError::DimensionMismatch {
                x_dim: 2,
                expected_dim: 1
            })
        ));
    }

    #[test]
    fn predict_n_one_matches_closed_form() {
        let noise = 0.25;
        let mut gp = rbf_gp(1.0, noise);
        gp.fit(&[0.0], 1, 1, &[2.0]).expect("spd");
        let pred = gp
            .predict_with(
                &[0.0],
                1,
                1,
                super::PredictOptions {
                    variance_kind: super::VarianceKind::Latent,
                },
            )
            .expect("fitted");
        let a = 1.0 + noise;
        assert_close(pred.mean[0], 2.0 / a);
        assert_close(pred.variance[0], 1.0 - 1.0 / a);
        let obs = gp.predict(&[0.0], 1, 1).expect("fitted");
        assert_eq!(obs.variance_kind, super::VarianceKind::Observation);
        assert_close(obs.variance[0], pred.variance[0] + noise);
    }

    #[test]
    fn observation_variance_is_latent_plus_noise_after_inverse() {
        let noise = 0.16;
        let mut gp = rbf_gp(1.0, noise).with_target_transform(StandardizeTarget::new());
        let y = [0.0, 4.0];
        gp.fit(&[0.0, 1.0], 2, 1, &y).expect("spd");
        let mut t = StandardizeTarget::new();
        t.fit(&y).expect("finite");
        let scale = t.std().expect("fitted");
        let scale_sq = scale * scale;
        let lat = gp
            .predict_with(
                &[0.5],
                1,
                1,
                super::PredictOptions {
                    variance_kind: super::VarianceKind::Latent,
                },
            )
            .expect("fitted");
        let obs = gp.predict(&[0.5], 1, 1).expect("fitted");
        assert_close(obs.variance[0], lat.variance[0] + scale_sq * noise);
        let mut recovered = y;
        t.transform(&mut recovered).expect("fitted");
        t.inverse_transform_mean(&mut recovered).expect("fitted");
        assert_close(recovered[0], y[0]);
        assert_close(recovered[1], y[1]);
    }
}
