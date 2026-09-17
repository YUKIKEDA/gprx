//! Batch Gaussian process regression: `A = K + σn² I`, LLT, and `α`.

use std::fmt;

use dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt;
use faer::linalg::cholesky::llt::factor::{LltError, LltRegularization};
use faer::{Mat, MatMut, MatRef, Par};

use crate::error::{CholeskyStage, GprError};
use crate::kernel::{CompiledKernel, CoordMode, KernelSpec, Triangle};
use crate::likelihood::GaussianLikelihood;
use crate::objective::GprObjective;
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

/// Options for [`Gpr::predict`].
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

/// Gaussian process regression with fixed hyperparameters.
///
/// [`Self::fit`] builds the lower triangle of `A = K + σn² I`, factors it
/// in place as `L Lᵀ`, and solves `A α = y`. `L` lives in the workspace;
/// `α` is kept on the model. [`Self::neg_log_marginal_likelihood`] is
/// `½ yᵀ α + ½ log|A| + (n/2) log(2π)` with `log|A| = 2 Σ log(L_ii)`.
/// [`Self::value_and_gradient_into`] rebuilds `L`, `α`, and `W` once and
/// writes `∂L/∂θ = -½ ⟨W, ∂A/∂θ⟩`. [`Self::predict`] returns the mean and
/// a diagonal variance. Input and target transforms default to identity.
///
/// # Examples
///
/// ```rust
/// use gprx::kernel::{KernelSpec, RbfKernel};
/// use gprx::{Gpr, GaussianLikelihood};
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
/// let likelihood = GaussianLikelihood::new(0.1)?;
/// let mut gpr = Gpr::new(kernel, likelihood);
/// // Column-major `X` with n = 2 points and d = 1 feature.
/// gpr.fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0])?;
/// let pred = gpr.predict(&[0.5], 1, 1)?;
/// assert_eq!(pred.mean.len(), 1);
/// let _nlml = gpr.neg_log_marginal_likelihood()?;
/// # Ok(())
/// # }
/// ```
pub struct Gpr {
    kernel: KernelSpec,
    compiled: Option<CompiledKernel>,
    likelihood: GaussianLikelihood,
    x_transform: Box<dyn Transform>,
    y_transform: Box<dyn TargetTransform>,
    workspace: Option<Workspace<DoublePrecision>>,
    x: Option<Mat<f64>>,
    y: Option<Vec<f64>>,
    alpha: Option<Vec<f64>>,
    fitted: bool,
    n: usize,
    d: usize,
}

impl fmt::Debug for Gpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Gpr")
            .field("fitted", &self.fitted)
            .field("n", &self.n)
            .field("d", &self.d)
            .field("kernel", &self.kernel)
            .field("likelihood", &self.likelihood)
            .finish_non_exhaustive()
    }
}

impl Gpr {
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
    /// Returns [`GprError::NotFitted`] if [`Self::fit`] has not succeeded.
    pub fn alpha(&self) -> Result<&[f64], GprError> {
        self.alpha.as_deref().ok_or(GprError::NotFitted)
    }

    /// Returns the negative log marginal likelihood of the last successful fit.
    ///
    /// Evaluates `½ yᵀ A⁻¹ y + ½ log|A| + (n/2) log(2π)` from the stored
    /// `α` and the Cholesky factor `L` in the workspace, using
    /// `log|A| = 2 Σ log(L_ii)`. `y` is the target after the target
    /// transform.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::NotFitted`] if [`Self::fit`] has not succeeded.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Gpr, GaussianLikelihood};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    /// let likelihood = GaussianLikelihood::new(0.1)?;
    /// let mut gpr = Gpr::new(kernel, likelihood);
    /// gpr.fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0])?;
    /// let nlml = gpr.neg_log_marginal_likelihood()?;
    /// assert!(nlml.is_finite());
    /// # Ok(())
    /// # }
    /// ```
    pub fn neg_log_marginal_likelihood(&self) -> Result<f64, GprError> {
        if !self.fitted {
            return Err(GprError::NotFitted);
        }
        let y = self.y.as_deref().ok_or(GprError::NotFitted)?;
        let alpha = self.alpha.as_deref().ok_or(GprError::NotFitted)?;
        let ws = self.workspace.as_ref().ok_or(GprError::NotFitted)?;
        Ok(neg_mll_from_factor(ws.k_matrix.as_ref(), y, alpha, self.n))
    }

    /// Returns the concatenated kernel and likelihood parameter count.
    pub fn num_params(&self) -> usize {
        self.kernel.num_params() + self.likelihood.num_params()
    }

    /// Writes kernel `θ` then likelihood `θ` into `out`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if `out` is the wrong length.
    pub fn get_params(&self, out: &mut [f64]) -> Result<(), GprError> {
        let n_kernel = self.kernel.num_params();
        require_param_len(out.len(), self.num_params())?;
        self.kernel.get_params(&mut out[..n_kernel])?;
        self.likelihood.get_params(&mut out[n_kernel..])
    }

    #[allow(dead_code)] // P1B-3 Gpr::fit
    pub(crate) fn objective(&mut self) -> GprObjective<'_> {
        GprObjective::new(self)
    }

    /// Sets kernel and likelihood `θ`, rebuilds `L` / `α` / `W`, and writes `∂L/∂θ`.
    ///
    /// `params` and `out` are kernel parameters followed by the likelihood
    /// parameter. One Cholesky produces `L` and `α`; `W = ααᵀ - A⁻¹` is
    /// formed in the workspace without overwriting `L`. Kernel `∂A/∂θ` goes
    /// through `exp_buf`. Product trees also use `kernel_scratch`. The
    /// returned value is the same as
    /// [`Self::neg_log_marginal_likelihood`] after a successful call.
    ///
    /// Training `X` / `y` must already come from [`Self::fit`]. Transforms
    /// are not re-fit.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::NotFitted`] if [`Self::fit`] has not stored data,
    /// [`GprError::InvalidHyperparameter`] if a slice length is wrong,
    /// [`GprError::InvalidNoiseVariance`] if the likelihood `θ` is invalid,
    /// [`GprError::CholeskyFailed`] if `A` cannot be factored, or
    /// [`GprError::UnsupportedKernelOperation`] if a points-mode product tree
    /// needs a gradient. Distance-mode product trees are supported. Kernel
    /// and likelihood `θ` are committed together only after `A` factors. A
    /// rejected slice or a Cholesky failure leaves stored `θ` unchanged.
    /// Cholesky failure still sets `fitted = false` because `L` is
    /// overwritten, but it keeps the training data.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Gpr, GaussianLikelihood};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    /// let likelihood = GaussianLikelihood::new(0.1)?;
    /// let mut gpr = Gpr::new(kernel, likelihood);
    /// gpr.fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0])?;
    /// let mut params = [0.0; 2];
    /// gpr.get_params(&mut params)?;
    /// let mut grad = [0.0; 2];
    /// let nlml = gpr.value_and_gradient_into(&params, &mut grad)?;
    /// assert!(nlml.is_finite());
    /// # Ok(())
    /// # }
    /// ```
    pub fn value_and_gradient_into(
        &mut self,
        params: &[f64],
        out: &mut [f64],
    ) -> Result<f64, GprError> {
        if self.x.is_none() || self.y.is_none() || self.workspace.is_none() {
            return Err(GprError::NotFitted);
        }
        let n_kernel = self.kernel.num_params();
        let n_params = self.num_params();
        require_param_len(params.len(), n_params)?;
        require_param_len(out.len(), n_params)?;
        let (kernel, compiled, likelihood) = self.prepared_params(params, n_kernel)?;
        let n = self.n;
        let y = self.y.as_deref().ok_or(GprError::NotFitted)?;
        {
            let x = self.x.as_ref().ok_or(GprError::NotFitted)?;
            let ws = workspace_mut(&mut self.workspace)?;
            apply_train_kernel(&compiled, x.as_ref(), ws)?;
            add_noise_to_diag(ws.k_matrix.as_mut(), likelihood.noise_variance());
        }
        let mut rhs = Mat::from_fn(n, 1, |i, _| y[i]);
        {
            let ws = workspace_mut(&mut self.workspace)?;
            if let Err(err) = cholesky_and_solve(
                &mut ws.k_matrix,
                &mut rhs,
                &mut ws.faer_scratch,
                0.0,
                CholeskyStage::Fit,
            ) {
                self.fitted = false;
                self.alpha = None;
                return Err(err);
            }
        }
        let alpha = self.alpha.get_or_insert_with(|| vec![0.0; n]);
        if alpha.len() != n {
            alpha.resize(n, 0.0);
        }
        for i in 0..n {
            alpha[i] = rhs[(i, 0)];
        }
        self.kernel = kernel;
        self.likelihood = likelihood;
        self.compiled = Some(compiled);
        self.fitted = true;
        let nlml = {
            let ws = self.workspace.as_ref().ok_or(GprError::NotFitted)?;
            let alpha = self.alpha.as_deref().ok_or(GprError::NotFitted)?;
            let y = self.y.as_deref().ok_or(GprError::NotFitted)?;
            neg_mll_from_factor(ws.k_matrix.as_ref(), y, alpha, n)
        };
        {
            let compiled = self.compiled.as_ref().ok_or(GprError::NotFitted)?;
            let alpha = self.alpha.as_deref().ok_or(GprError::NotFitted)?;
            let ws = workspace_mut(&mut self.workspace)?;
            fill_identity(ws.w_matrix.as_mut());
            {
                let stack = MemStack::new(&mut ws.faer_scratch);
                llt::solve::solve_in_place(
                    ws.k_matrix.as_ref(),
                    ws.w_matrix.as_mut(),
                    Par::Seq,
                    stack,
                );
            }
            form_w_lower(ws.w_matrix.as_mut(), alpha, n);
            let x = self.x.as_ref().ok_or(GprError::NotFitted)?;
            if compiled.needs_product_grad_scratch() {
                ws.ensure_kernel_scratch(n)?;
            }
            for (i, slot) in out.iter_mut().enumerate().take(n_kernel) {
                write_kernel_grad(
                    compiled,
                    ws.dist_cache.as_ref(),
                    x.as_ref(),
                    ws.exp_buf.as_mut(),
                    ws.kernel_scratch.as_mut(),
                    i,
                )?;
                let inner = frobenius_lower(ws.w_matrix.as_ref(), ws.exp_buf.as_ref(), n);
                *slot = -0.5 * inner;
            }
            let mut noise_inner = 0.0;
            let d_noise = self.likelihood.noise_variance();
            for i in 0..n {
                noise_inner += ws.w_matrix[(i, i)] * d_noise;
            }
            out[n_kernel] = -0.5 * noise_inner;
        }
        Ok(nlml)
    }

    /// Builds kernel, compiled kernel, and likelihood `θ` without storing them.
    ///
    /// Each `set_params` is atomic on its own type. The caller commits the
    /// triple only after `A` factors, so a later Cholesky failure cannot
    /// leave stored kernel and likelihood `θ` mixed or half-applied.
    fn prepared_params(
        &self,
        params: &[f64],
        n_kernel: usize,
    ) -> Result<(KernelSpec, CompiledKernel, GaussianLikelihood), GprError> {
        let mut likelihood = self.likelihood;
        likelihood.set_params(&params[n_kernel..])?;
        let mut kernel = self.kernel.clone();
        kernel.set_params(&params[..n_kernel])?;
        let mut compiled = match self.compiled.as_ref() {
            Some(compiled) => compiled.clone(),
            None => kernel.compile(),
        };
        compiled.set_params(&params[..n_kernel])?;
        Ok((kernel, compiled, likelihood))
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
    /// Returns [`GprError::EmptyInput`] if `n_rows` or `n_cols` is zero,
    /// [`GprError::InvalidHyperparameter`] if `x` or `y` has the wrong length,
    /// [`GprError::NonFiniteInput`] if a value is `NaN` or `Inf`, or
    /// [`GprError::CholeskyFailed`] if `A` cannot be factored.
    pub fn fit(
        &mut self,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
        y: &[f64],
    ) -> Result<(), GprError> {
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
            apply_train_kernel(&compiled, x_mat.as_ref(), ws)?;
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
    /// Returns [`GprError::NotFitted`] if [`Self::fit`] has not succeeded,
    /// [`GprError::DimensionMismatch`] if `n_cols` differs from the training
    /// features, [`GprError::EmptyInput`] if a dimension is zero, or
    /// [`GprError::InvalidHyperparameter`] / [`GprError::NonFiniteInput`] for a
    /// badly packed or non-finite `xs`.
    pub fn predict(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
    ) -> Result<Prediction, GprError> {
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
    ) -> Result<Prediction, GprError> {
        if !self.fitted {
            return Err(GprError::NotFitted);
        }
        if n_cols != self.d {
            return Err(GprError::DimensionMismatch {
                x_dim: n_cols,
                expected_dim: self.d,
            });
        }
        validate_query(xs, n_rows, n_cols)?;
        let compiled = self.compiled.as_ref().ok_or(GprError::NotFitted)?;
        let x_train = self.x.as_ref().ok_or(GprError::NotFitted)?;
        let alpha = self.alpha.as_deref().ok_or(GprError::NotFitted)?;
        let ws = self.workspace.as_ref().ok_or(GprError::NotFitted)?;
        let mut xs_buf = xs.to_vec();
        self.x_transform.apply(&mut xs_buf, n_rows, n_cols)?;
        let x_test = pack_points(&xs_buf, n_rows, n_cols);
        let n = self.n;
        let m = n_rows;
        let mut k_star = Mat::zeros(n, m);
        let mut scratch = Mat::zeros(n, m);
        match compiled.coord_mode()? {
            CoordMode::Dist | CoordMode::Either => {
                let mut dist = Mat::zeros(n, m);
                fill_squared_euclidean_cross(x_train.as_ref(), x_test.as_ref(), dist.as_mut());
                compiled.apply_cross(dist.as_ref(), k_star.as_mut(), scratch.as_mut())?;
            }
            CoordMode::Points => {
                compiled.apply_cross_points(
                    x_train.as_ref(),
                    x_test.as_ref(),
                    k_star.as_mut(),
                    scratch.as_mut(),
                )?;
            }
        }
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
        match compiled.coord_mode()? {
            CoordMode::Dist | CoordMode::Either => compiled.fill_diag(&mut kss)?,
            CoordMode::Points => compiled.fill_diag_points(x_test.as_ref(), &mut kss)?,
        }
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

    fn prepare_workspace(&mut self, n: usize) -> Result<(), GprError> {
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
) -> Result<&mut Workspace<DoublePrecision>, GprError> {
    workspace.as_mut().ok_or(GprError::EmptyInput)
}

/// Writes the training Gram matrix. Distance-mode leaves get squared Euclidean
/// distances from `x` first so MLL/grad does not reuse a stale `dist_cache`.
fn apply_train_kernel(
    compiled: &CompiledKernel,
    x: MatRef<'_, f64>,
    ws: &mut Workspace<DoublePrecision>,
) -> Result<(), GprError> {
    match compiled.coord_mode()? {
        CoordMode::Dist | CoordMode::Either => {
            fill_squared_euclidean(x, ws.dist_cache.as_mut());
            compiled.apply(
                ws.dist_cache.as_ref(),
                ws.k_matrix.as_mut(),
                Triangle::Lower,
                ws.exp_buf.as_mut(),
            )
        }
        CoordMode::Points => compiled.apply_points(
            x,
            ws.k_matrix.as_mut(),
            Triangle::Lower,
            ws.exp_buf.as_mut(),
        ),
    }
}

fn validate_training(x: &[f64], n_rows: usize, n_cols: usize, y: &[f64]) -> Result<(), GprError> {
    if n_rows == 0 || n_cols == 0 {
        return Err(GprError::EmptyInput);
    }
    let expected_x = n_rows.checked_mul(n_cols).ok_or(GprError::EmptyInput)?;
    if x.len() != expected_x {
        return Err(GprError::InvalidHyperparameter {
            reason: format!("expected {expected_x} feature values, got {}", x.len()),
        });
    }
    if y.len() != n_rows {
        return Err(GprError::InvalidHyperparameter {
            reason: format!("expected {n_rows} targets, got {}", y.len()),
        });
    }
    if x.iter().any(|v| !v.is_finite()) || y.iter().any(|v| !v.is_finite()) {
        return Err(GprError::NonFiniteInput);
    }
    Ok(())
}

fn validate_query(xs: &[f64], n_rows: usize, n_cols: usize) -> Result<(), GprError> {
    if n_rows == 0 || n_cols == 0 {
        return Err(GprError::EmptyInput);
    }
    let expected = n_rows.checked_mul(n_cols).ok_or(GprError::EmptyInput)?;
    if xs.len() != expected {
        return Err(GprError::InvalidHyperparameter {
            reason: format!("expected {expected} feature values, got {}", xs.len()),
        });
    }
    if xs.iter().any(|v| !v.is_finite()) {
        return Err(GprError::NonFiniteInput);
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

fn log_det_from_l(l: MatRef<'_, f64>, n: usize) -> f64 {
    let mut log_diag = 0.0;
    for i in 0..n {
        log_diag += l[(i, i)].ln();
    }
    2.0 * log_diag
}

fn neg_mll_from_factor(l: MatRef<'_, f64>, y: &[f64], alpha: &[f64], n: usize) -> f64 {
    let mut quad = 0.0;
    for i in 0..n {
        quad += y[i] * alpha[i];
    }
    let log_det = log_det_from_l(l, n);
    let log_two_pi = (2.0 * std::f64::consts::PI).ln();
    0.5 * (quad + log_det + n as f64 * log_two_pi)
}

fn require_param_len(actual: usize, expected: usize) -> Result<(), GprError> {
    if actual == expected {
        Ok(())
    } else {
        Err(GprError::InvalidHyperparameter {
            reason: format!("expected {expected} parameters, got {actual}"),
        })
    }
}

fn fill_identity(mut a: MatMut<'_, f64>) {
    let n = a.nrows();
    for col in 0..n {
        for row in 0..n {
            a[(row, col)] = if row == col { 1.0 } else { 0.0 };
        }
    }
}

fn form_w_lower(mut w: MatMut<'_, f64>, alpha: &[f64], n: usize) {
    for col in 0..n {
        for row in col..n {
            w[(row, col)] = alpha[row] * alpha[col] - w[(row, col)];
        }
    }
}

fn frobenius_lower(w: MatRef<'_, f64>, d_k: MatRef<'_, f64>, n: usize) -> f64 {
    let mut inner = 0.0;
    for col in 0..n {
        inner += w[(col, col)] * d_k[(col, col)];
        for row in col + 1..n {
            inner += 2.0 * w[(row, col)] * d_k[(row, col)];
        }
    }
    inner
}

fn write_kernel_grad(
    compiled: &CompiledKernel,
    dist: MatRef<'_, f64>,
    x: MatRef<'_, f64>,
    d_k: MatMut<'_, f64>,
    scratch: MatMut<'_, f64>,
    param_idx: usize,
) -> Result<(), GprError> {
    match compiled.coord_mode()? {
        CoordMode::Dist | CoordMode::Either => {
            compiled.grad(dist, d_k, param_idx, Triangle::Lower, scratch)
        }
        CoordMode::Points => compiled.grad_points(x, d_k, param_idx, Triangle::Lower, scratch),
    }
}

/// Factors `A` in place as `L Lᵀ` and overwrites `rhs` with `A⁻¹ rhs`.
///
/// P1A-18 can call this on the same `Workspace` buffers as [`Gpr::fit`].
pub(crate) fn cholesky_and_solve(
    a: &mut Mat<f64>,
    rhs: &mut Mat<f64>,
    scratch: &mut MemBuffer,
    jitter: f64,
    stage: CholeskyStage,
) -> Result<(), GprError> {
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
                return Err(GprError::CholeskyFailed {
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
    use super::{Gpr, cholesky_and_solve, pack_points};
    use crate::error::{CholeskyStage, GprError};
    use crate::kernel::{
        ConstantKernel, KernelSpec, LinearKernel, MaternArdKernel, MaternKernel, MaternNu,
        PeriodicKernel, RationalQuadraticArdKernel, RationalQuadraticKernel, RbfArdKernel,
        RbfKernel, Triangle, WhiteKernel,
    };
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

    fn rbf_gpr(ell: f64, noise: f64) -> Gpr {
        Gpr::new(
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
        assert_send_sync::<Gpr>();
        assert_send_sync::<super::Prediction>();
        assert_send_sync::<super::VarianceKind>();
        assert_send_sync::<super::PredictOptions>();
    }

    #[test]
    fn fit_solves_a_alpha_equals_y() {
        let mut gpr = rbf_gpr(1.25, 0.1);
        let x = [0.0, 0.5, 1.5, 0.0, 1.0, 0.5];
        let y = [0.2, -1.0, 0.7];
        gpr.fit(&x, 3, 2, &y).expect("spd");
        assert!(gpr.is_fitted());
        assert_eq!(gpr.n(), 3);
        assert_eq!(gpr.d(), 2);
        let a = dense_a(gpr.kernel(), gpr.likelihood().noise_variance(), &x, 3, 2);
        let alpha = gpr.alpha().expect("fitted");
        let restored = matvec_sym(&a, alpha);
        for i in 0..3 {
            assert_close(restored[i], y[i]);
        }
        let ws = gpr.workspace.as_ref().expect("workspace");
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
        let mut gpr = rbf_gpr(1.25, 0.1);
        gpr.fit(&[0.0, 0.5, 1.5, 0.0, 1.0, 0.5], 3, 2, &[0.2, -1.0, 0.7])
            .expect("spd");
        gpr.fit(&[0.0, 1.0], 2, 1, &[0.5, -0.25]).expect("refit");
        assert_eq!(gpr.n(), 2);
        assert_eq!(gpr.d(), 1);
        let a = dense_a(
            gpr.kernel(),
            gpr.likelihood().noise_variance(),
            &[0.0, 1.0],
            2,
            1,
        );
        let alpha = gpr.alpha().expect("fitted");
        let restored = matvec_sym(&a, alpha);
        assert_close(restored[0], 0.5);
        assert_close(restored[1], -0.25);
    }

    #[test]
    fn fit_n_one_matches_scalar_solve() {
        let noise = 0.25;
        let mut gpr = rbf_gpr(1.0, noise);
        gpr.fit(&[0.0], 1, 1, &[2.0]).expect("spd");
        let a = 1.0 + noise;
        assert_close(gpr.alpha().expect("fitted")[0], 2.0 / a);
    }

    #[test]
    fn validation_error_keeps_previous_fit() {
        let mut gpr = rbf_gpr(1.0, 0.1);
        gpr.fit(&[0.0, 1.0], 2, 1, &[1.0, 2.0]).expect("spd");
        let alpha = gpr.alpha().expect("fitted").to_vec();
        assert!(matches!(
            gpr.fit(&[0.0], 0, 1, &[]),
            Err(GprError::EmptyInput)
        ));
        assert!(gpr.is_fitted());
        assert_close(gpr.alpha().expect("fitted")[0], alpha[0]);
        assert_close(gpr.alpha().expect("fitted")[1], alpha[1]);
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
            GprError::CholeskyFailed {
                stage: CholeskyStage::Fit,
                matrix_size: 2,
                jitter: _,
            }
        ));
    }

    #[test]
    fn fit_rejects_bad_shapes_and_non_finite() {
        let mut gpr = rbf_gpr(1.0, 0.1);
        assert!(matches!(
            gpr.fit(&[0.0], 2, 1, &[0.0, 1.0]),
            Err(GprError::InvalidHyperparameter { .. })
        ));
        assert!(matches!(
            gpr.fit(&[0.0, 1.0], 2, 1, &[0.0]),
            Err(GprError::InvalidHyperparameter { .. })
        ));
        assert!(matches!(
            gpr.fit(&[0.0, f64::NAN], 2, 1, &[0.0, 1.0]),
            Err(GprError::NonFiniteInput)
        ));
        assert!(!gpr.is_fitted());
    }

    #[test]
    fn unfitted_alpha_is_not_fitted() {
        let gpr = rbf_gpr(1.0, 0.1);
        assert!(matches!(gpr.alpha(), Err(GprError::NotFitted)));
        assert!(matches!(
            gpr.neg_log_marginal_likelihood(),
            Err(GprError::NotFitted)
        ));
        assert!(!gpr.is_fitted());
    }

    #[test]
    fn neg_mll_n_one_matches_closed_form() {
        let noise = 0.25;
        let y = 2.0;
        let mut gpr = rbf_gpr(1.0, noise);
        gpr.fit(&[0.0], 1, 1, &[y]).expect("spd");
        let a = 1.0 + noise;
        let log_det = a.ln();
        let ws = gpr.workspace.as_ref().expect("workspace");
        assert_close(super::log_det_from_l(ws.k_matrix.as_ref(), 1), log_det);
        let quad = y * y / a;
        let expected = 0.5 * (quad + log_det + (2.0 * std::f64::consts::PI).ln());
        assert_close(gpr.neg_log_marginal_likelihood().expect("fitted"), expected);
    }

    #[test]
    fn neg_mll_n_two_matches_analytic_det_and_quad() {
        let ell = 1.0;
        let noise = 0.1;
        let x = [0.0, 1.0];
        let y = [0.5, -0.25];
        let mut gpr = rbf_gpr(ell, noise);
        gpr.fit(&x, 2, 1, &y).expect("spd");
        let k01 = (-0.5 * (1.0 / ell) * (1.0 / ell)).exp();
        let diag = 1.0 + noise;
        let det = diag * diag - k01 * k01;
        let log_det = det.ln();
        let ws = gpr.workspace.as_ref().expect("workspace");
        assert_close(super::log_det_from_l(ws.k_matrix.as_ref(), 2), log_det);
        let inv_scale = 1.0 / det;
        let quad =
            inv_scale * (y[0] * (diag * y[0] - k01 * y[1]) + y[1] * (-k01 * y[0] + diag * y[1]));
        let expected = 0.5 * (quad + log_det + 2.0 * (2.0 * std::f64::consts::PI).ln());
        assert_close(gpr.neg_log_marginal_likelihood().expect("fitted"), expected);
    }

    #[test]
    fn neg_mll_uses_transformed_targets() {
        let noise = 0.16;
        let y = [0.0, 4.0];
        let mut gpr = rbf_gpr(1.0, noise).with_target_transform(StandardizeTarget::new());
        gpr.fit(&[0.0, 1.0], 2, 1, &y).expect("spd");
        let mut t = StandardizeTarget::new();
        t.fit(&y).expect("finite");
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
        assert_close(gpr.neg_log_marginal_likelihood().expect("fitted"), expected);
        let raw = 0.5
            * (inv_scale
                * (y[0] * (diag * y[0] - k01 * y[1]) + y[1] * (-k01 * y[0] + diag * y[1]))
                + log_det
                + 2.0 * (2.0 * std::f64::consts::PI).ln());
        assert!((gpr.neg_log_marginal_likelihood().expect("fitted") - raw).abs() > TOL);
    }

    #[test]
    fn value_and_gradient_rejects_unfitted_and_bad_len() {
        let mut gpr = rbf_gpr(1.0, 0.1);
        let params = [0.0, 0.0];
        let mut grad = [0.0, 0.0];
        assert!(matches!(
            gpr.value_and_gradient_into(&params, &mut grad),
            Err(GprError::NotFitted)
        ));
        gpr.fit(&[0.0, 1.0], 2, 1, &[0.5, -0.25]).expect("spd");
        assert!(matches!(
            gpr.value_and_gradient_into(&[0.0], &mut grad),
            Err(GprError::InvalidHyperparameter { .. })
        ));
        assert!(matches!(
            gpr.get_params(&mut [0.0]),
            Err(GprError::InvalidHyperparameter { .. })
        ));
    }

    #[test]
    fn value_and_gradient_set_params_is_atomic() {
        let mut gpr = rbf_gpr(1.0, 0.1);
        gpr.fit(&[0.0, 1.0], 2, 1, &[0.5, -0.25]).expect("spd");
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
        assert_close(after[0], before[0]);
        assert_close(after[1], before[1]);
    }

    #[test]
    fn value_and_gradient_cholesky_failure_keeps_params() {
        let mut gpr = rbf_gpr(1.0, 0.1);
        gpr.fit(&[0.0, 0.0], 2, 1, &[0.5, -0.25]).expect("spd");
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
        assert!(!gpr.is_fitted());
        let mut after = [0.0; 2];
        gpr.get_params(&mut after).expect("len 2");
        assert_close(after[0], before[0]);
        assert_close(after[1], before[1]);
        gpr.value_and_gradient_into(&before, &mut grad)
            .expect("restore");
        assert!(gpr.is_fitted());
    }

    #[test]
    fn value_and_gradient_refills_stale_dist_cache() {
        let mut gpr = rbf_gpr(1.25, 0.16);
        gpr.fit(&[0.0, 0.8, 1.7], 3, 1, &[0.4, -0.2, 0.9])
            .expect("spd");
        if let Some(ws) = gpr.workspace.as_mut() {
            let n = ws.dist_cache.nrows();
            ws.dist_cache = Mat::from_fn(n, n, |_, _| 999.0);
        }
        let mut params = [0.0; 2];
        gpr.get_params(&mut params).expect("len 2");
        let mut grad = [0.0; 2];
        let value = gpr
            .value_and_gradient_into(&params, &mut grad)
            .expect("spd");
        assert_close(value, gpr.neg_log_marginal_likelihood().expect("fitted"));
        assert!(grad.iter().all(|g| g.is_finite()));
    }

    #[test]
    fn value_and_gradient_matches_nlml_and_finite_difference() {
        let mut gpr = rbf_gpr(1.25, 0.16);
        gpr.fit(&[0.0, 0.8, 1.7], 3, 1, &[0.4, -0.2, 0.9])
            .expect("spd");
        let mut params = [0.0; 2];
        gpr.get_params(&mut params).expect("len 2");
        let mut grad = [0.0; 2];
        let value = gpr
            .value_and_gradient_into(&params, &mut grad)
            .expect("spd");
        assert_close(value, gpr.neg_log_marginal_likelihood().expect("fitted"));
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
    fn value_and_gradient_sum_rbf_matches_finite_difference() {
        let kernel = KernelSpec::from(RbfKernel::new(1.25).expect("valid"))
            + KernelSpec::from(RbfKernel::new(0.7).expect("valid"));
        let mut gpr = Gpr::new(kernel, GaussianLikelihood::new(0.16).expect("valid"));
        gpr.fit(&[0.0, 0.8, 1.7], 3, 1, &[0.4, -0.2, 0.9])
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
        let mut gpr = Gpr::new(kernel, GaussianLikelihood::new(0.1).expect("valid"));
        gpr.fit(&[0.0, 1.0], 2, 1, &[0.5, -0.25]).expect("spd");
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
    fn value_and_gradient_sum_of_product_matches_finite_difference() {
        let kernel = KernelSpec::from(ConstantKernel::new(1.5).expect("valid"))
            * KernelSpec::from(RbfKernel::new(1.0).expect("valid"))
            + KernelSpec::from(RbfKernel::new(2.0).expect("valid"));
        let mut gpr = Gpr::new(kernel, GaussianLikelihood::new(0.1).expect("valid"));
        gpr.fit(&[0.0, 1.0], 2, 1, &[0.5, -0.25]).expect("spd");
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
        let mut gpr = rbf_gpr(1.0, noise);
        gpr.fit(&[0.0], 1, 1, &[y]).expect("spd");
        let mut params = [0.0; 2];
        gpr.get_params(&mut params).expect("len 2");
        let mut grad = [0.0; 2];
        gpr.value_and_gradient_into(&params, &mut grad)
            .expect("spd");
        let a = 1.0 + noise;
        let w = (y / a) * (y / a) - 1.0 / a;
        assert_close(grad[0], 0.0);
        assert_close(grad[1], -0.5 * w * noise);
    }

    #[test]
    fn predict_rejects_unfitted_and_wrong_dim() {
        let mut gpr = rbf_gpr(1.0, 0.1);
        assert!(matches!(
            gpr.predict(&[0.0], 1, 1),
            Err(GprError::NotFitted)
        ));
        gpr.fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0]).expect("spd");
        assert!(matches!(
            gpr.predict(&[0.0, 1.0], 1, 2),
            Err(GprError::DimensionMismatch {
                x_dim: 2,
                expected_dim: 1
            })
        ));
    }

    #[test]
    fn predict_n_one_matches_closed_form() {
        let noise = 0.25;
        let mut gpr = rbf_gpr(1.0, noise);
        gpr.fit(&[0.0], 1, 1, &[2.0]).expect("spd");
        let pred = gpr
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
        let obs = gpr.predict(&[0.0], 1, 1).expect("fitted");
        assert_eq!(obs.variance_kind, super::VarianceKind::Observation);
        assert_close(obs.variance[0], pred.variance[0] + noise);
    }

    #[test]
    fn observation_variance_is_latent_plus_noise_after_inverse() {
        let noise = 0.16;
        let mut gpr = rbf_gpr(1.0, noise).with_target_transform(StandardizeTarget::new());
        let y = [0.0, 4.0];
        gpr.fit(&[0.0, 1.0], 2, 1, &y).expect("spd");
        let mut t = StandardizeTarget::new();
        t.fit(&y).expect("finite");
        let scale = t.std().expect("fitted");
        let scale_sq = scale * scale;
        let lat = gpr
            .predict_with(
                &[0.5],
                1,
                1,
                super::PredictOptions {
                    variance_kind: super::VarianceKind::Latent,
                },
            )
            .expect("fitted");
        let obs = gpr.predict(&[0.5], 1, 1).expect("fitted");
        assert_close(obs.variance[0], lat.variance[0] + scale_sq * noise);
        let mut recovered = y;
        t.transform(&mut recovered).expect("fitted");
        t.inverse_transform_mean(&mut recovered).expect("fitted");
        assert_close(recovered[0], y[0]);
        assert_close(recovered[1], y[1]);
    }

    #[test]
    fn ard_equal_lengthscales_match_isotropic_predict() {
        let ell = 1.25;
        let noise = 0.1;
        let x = [0.0, 0.5, 1.5, 0.0, 1.0, 0.5];
        let y = [0.2, -1.0, 0.7];
        let xs = [0.25, 1.0];
        let mut iso = rbf_gpr(ell, noise);
        iso.fit(&x, 3, 2, &y).expect("spd");
        let mut ard = Gpr::new(
            KernelSpec::from(RbfArdKernel::new(&[ell, ell]).expect("valid")),
            GaussianLikelihood::new(noise).expect("valid"),
        );
        ard.fit(&x, 3, 2, &y).expect("spd");
        let p_iso = iso.predict(&xs, 1, 2).expect("fitted");
        let p_ard = ard.predict(&xs, 1, 2).expect("fitted");
        assert_close(p_ard.mean[0], p_iso.mean[0]);
        assert_close(p_ard.variance[0], p_iso.variance[0]);
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
        let mut gpr = Gpr::new(
            KernelSpec::from(RbfKernel::new(1.0).expect("valid"))
                + KernelSpec::from(WhiteKernel::new(0.05).expect("valid")),
            GaussianLikelihood::new(0.1).expect("valid"),
        );
        gpr.fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0]).expect("spd");
        let pred = gpr.predict(&[0.5], 1, 1).expect("fitted");
        assert!(pred.mean[0].is_finite());
        assert!(pred.variance[0] > 0.0);
    }

    #[test]
    fn linear_kernel_fits_and_predicts() {
        let mut gpr = Gpr::new(
            KernelSpec::from(LinearKernel::new(1.0).expect("valid")),
            GaussianLikelihood::new(0.1).expect("valid"),
        );
        let x = [0.0, 1.0, 2.0];
        let y = [0.0, 1.0, 2.0];
        gpr.fit(&x, 3, 1, &y).expect("spd");
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
    fn matern_fits_and_predicts() {
        let mut gpr = Gpr::new(
            KernelSpec::from(MaternKernel::new(1.0, MaternNu::FiveHalves).expect("valid")),
            GaussianLikelihood::new(0.1).expect("valid"),
        );
        gpr.fit(&[0.0, 1.0, 2.0], 3, 1, &[0.0, 0.5, 1.0])
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
        let mut iso = Gpr::new(
            KernelSpec::from(MaternKernel::new(ell, nu).expect("valid")),
            GaussianLikelihood::new(noise).expect("valid"),
        );
        iso.fit(&x, 3, 2, &y).expect("spd");
        let mut ard = Gpr::new(
            KernelSpec::from(MaternArdKernel::new(&[ell, ell], nu).expect("valid")),
            GaussianLikelihood::new(noise).expect("valid"),
        );
        ard.fit(&x, 3, 2, &y).expect("spd");
        let p_iso = iso.predict(&xs, 1, 2).expect("fitted");
        let p_ard = ard.predict(&xs, 1, 2).expect("fitted");
        assert_close(p_ard.mean[0], p_iso.mean[0]);
        assert_close(p_ard.variance[0], p_iso.variance[0]);
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
        );
        gpr.fit(&[0.0, 0.5, 1.0], 3, 1, &[0.0, 0.4, 0.1])
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
        );
        gpr.fit(&[0.0, 0.5, 1.0], 3, 1, &[0.0, 0.4, 0.1])
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
        let mut iso = Gpr::new(
            KernelSpec::from(RationalQuadraticKernel::new(ell, alpha).expect("valid")),
            GaussianLikelihood::new(noise).expect("valid"),
        );
        iso.fit(&x, 3, 2, &y).expect("spd");
        let mut ard = Gpr::new(
            KernelSpec::from(RationalQuadraticArdKernel::new(&[ell, ell], alpha).expect("valid")),
            GaussianLikelihood::new(noise).expect("valid"),
        );
        ard.fit(&x, 3, 2, &y).expect("spd");
        let p_iso = iso.predict(&xs, 1, 2).expect("fitted");
        let p_ard = ard.predict(&xs, 1, 2).expect("fitted");
        assert_close(p_ard.mean[0], p_iso.mean[0]);
        assert_close(p_ard.variance[0], p_iso.variance[0]);
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
}
