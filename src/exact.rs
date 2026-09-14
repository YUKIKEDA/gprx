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
use crate::workspace::Workspace;

/// Exact GP with fixed hyperparameters.
///
/// [`Self::fit`] builds the lower triangle of `A = K + σn² I`, factors it
/// in place as `L Lᵀ`, and solves `A α = y`. `L` lives in the workspace;
/// `α` is kept on the model. Prediction and the marginal likelihood wait for
/// later issues. Transforms are not applied here.
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
/// assert!(gp.is_fitted());
/// # Ok(())
/// # }
/// ```
pub struct ExactGP {
    kernel: KernelSpec,
    #[allow(dead_code)] // predict (P1A-8)
    compiled: Option<CompiledKernel>,
    likelihood: GaussianLikelihood,
    workspace: Option<Workspace<DoublePrecision>>,
    #[allow(dead_code)] // predict (P1A-8)
    x: Option<Mat<f64>>,
    #[allow(dead_code)] // predict (P1A-8)
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
    pub fn new(kernel: KernelSpec, likelihood: GaussianLikelihood) -> Self {
        Self {
            kernel,
            compiled: None,
            likelihood,
            workspace: None,
            x: None,
            y: None,
            alpha: None,
            fitted: false,
            n: 0,
            d: 0,
        }
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
        let x_mat = pack_points(x, n_rows, n_cols);
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
        let mut rhs = Mat::from_fn(n_rows, 1, |i, _| y[i]);
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
        self.y = Some(y.to_vec());
        self.alpha = Some((0..n_rows).map(|i| rhs[(i, 0)]).collect());
        self.n = n_rows;
        self.d = n_cols;
        self.fitted = true;
        Ok(())
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
}
