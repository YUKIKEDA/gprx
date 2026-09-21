//! Variational sparse GPR with caller-supplied inducing points.

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
use crate::optimizer::Fixed;
use crate::workspace::{faer_par, faer_par_dims};
use crate::{PredictOptions, Prediction, VarianceKind};

/// Trainer for variational sparse GPR at a fixed inducing set `Z`.
///
/// [`SparseGpr<Fixed>::factor`] prepares the VFE system. Hyperparameter
/// search is P4-4.
///
/// # Examples
///
/// ```rust
/// use gprx::kernel::{KernelSpec, RbfKernel};
/// use gprx::{GaussianLikelihood, SparseGpr};
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
/// let likelihood = GaussianLikelihood::new(0.1)?;
/// let fitted = SparseGpr::new(kernel, likelihood)
///     .factor(&[0.0, 1.0, 2.0, 3.0], 4, 1, &[0.0, 1.0, 0.5, 0.25], &[0.5, 2.5], 2)
///     .map_err(|(_, e)| e)?;
/// assert_eq!(fitted.n(), 4);
/// assert_eq!(fitted.m(), 2);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct SparseGpr<O = Fixed> {
    kernel: KernelSpec,
    likelihood: GaussianLikelihood,
    _optimizer: PhantomData<O>,
}

/// Factored variational sparse GPR at the `θ` used by [`SparseGpr<Fixed>::factor`].
///
/// Stores the LLT of `K_mm = k(Z, Z)` and the VFE factors used by
/// [`Self::predict`] and [`Self::neg_log_marginal_likelihood`]. Observation
/// noise is not added to `K_mm`.
#[derive(Clone, Debug)]
pub struct FittedSparseGpr<O = Fixed> {
    kernel: KernelSpec,
    likelihood: GaussianLikelihood,
    x_obs: Vec<f64>,
    z_obs: Vec<f64>,
    y: Vec<f64>,
    /// Lower `L` from `K_mm = L Lᵀ`.
    k_mm_l: Mat<f64>,
    /// `A = L_mm⁻¹ K(Z, X)` (`m × n`).
    a: Mat<f64>,
    /// Lower `L_B` from `B = σn² I + A Aᵀ`.
    b_l: Mat<f64>,
    /// `B⁻¹ A y`.
    w: Vec<f64>,
    k_diag_sum: f64,
    a_frobenius2: f64,
    n: usize,
    m: usize,
    d: usize,
    _optimizer: PhantomData<O>,
}

impl SparseGpr<Fixed> {
    /// Builds a trainer with identity transforms and the current kernel `θ`.
    ///
    /// Inducing coordinates are an argument of [`Self::factor`], not of this
    /// constructor.
    pub fn new(kernel: KernelSpec, likelihood: GaussianLikelihood) -> Self {
        Self {
            kernel,
            likelihood,
            _optimizer: PhantomData,
        }
    }

    /// Factors `K_mm = k(Z, Z)` at the current `θ` without a search.
    ///
    /// `x` and `z` are column-major (`n` / `m` points by `d` features). `Z`
    /// is supplied by the caller and is not moved. Likelihood noise is not
    /// added to `K_mm`. Also forms the VFE factors used by
    /// [`FittedSparseGpr::predict`] and
    /// [`FittedSparseGpr::neg_log_marginal_likelihood`].
    ///
    /// # Errors
    ///
    /// Returns [`GprError::EmptyInput`] when `n`, `d`, or `m` is zero.
    /// Returns [`GprError::DimensionMismatch`] when `z` is packed with a
    /// different feature count than `x` (the same `n_cols` is required).
    /// Length and finiteness errors match [`crate::Gpr<Fixed>::factor`].
    /// [`GprError::CholeskyFailed`] when `K_mm` or the VFE `B` matrix cannot
    /// be factored.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{GaussianLikelihood, SparseGpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    /// let likelihood = GaussianLikelihood::new(0.1)?;
    /// let fitted = SparseGpr::new(kernel, likelihood)
    ///     .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0], &[0.0, 1.0], 2)
    ///     .map_err(|(_, e)| e)?;
    /// assert_eq!(fitted.m(), fitted.n());
    /// # Ok(())
    /// # }
    /// ```
    #[allow(clippy::result_large_err)] // failure returns the trainer so the caller can retry
    pub fn factor(
        self,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
        y: &[f64],
        z: &[f64],
        n_inducing: usize,
    ) -> Result<FittedSparseGpr<Fixed>, (Self, GprError)> {
        match self.factor_inner(x, n_rows, n_cols, y, z, n_inducing) {
            Ok(fitted) => Ok(fitted),
            Err(err) => Err((self, err)),
        }
    }

    fn factor_inner(
        &self,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
        y: &[f64],
        z: &[f64],
        n_inducing: usize,
    ) -> Result<FittedSparseGpr<Fixed>, GprError> {
        validate_training(x, n_rows, n_cols, y)?;
        validate_inducing(z, n_inducing, n_cols)?;
        let compiled = self.kernel.compile();
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
            JitterPolicy::default(),
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
        let noise = self.likelihood.noise_variance();
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
        Ok(FittedSparseGpr {
            kernel: self.kernel.clone(),
            likelihood: self.likelihood,
            x_obs: x.to_vec(),
            z_obs: z.to_vec(),
            y: y.to_vec(),
            k_mm_l: k_mm,
            a,
            b_l: b,
            w,
            k_diag_sum,
            a_frobenius2,
            n: n_rows,
            m: n_inducing,
            d: n_cols,
            _optimizer: PhantomData,
        })
    }
}

impl<O> FittedSparseGpr<O> {
    /// Returns the number of training points.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Returns the number of inducing points.
    pub fn m(&self) -> usize {
        self.m
    }

    /// Returns the feature dimension.
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

    /// Returns the original training features in column-major order.
    pub fn x(&self) -> &[f64] {
        &self.x_obs
    }

    /// Returns the inducing features in column-major order.
    pub fn z(&self) -> &[f64] {
        &self.z_obs
    }

    /// Returns the original training targets.
    pub fn y(&self) -> &[f64] {
        &self.y
    }

    /// Returns the negative VFE evidence lower bound (the sparse NLML).
    ///
    /// When `Z = X` this matches [`crate::FittedGpr::neg_log_marginal_likelihood`]
    /// of [`crate::Gpr<Fixed>::factor`] on the same data.
    ///
    /// # Errors
    ///
    /// The stored factors are already valid, so this returns `Ok`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{GaussianLikelihood, SparseGpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    /// let likelihood = GaussianLikelihood::new(0.1)?;
    /// let fitted = SparseGpr::new(kernel, likelihood)
    ///     .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0], &[0.0, 1.0], 2)
    ///     .map_err(|(_, e)| e)?;
    /// let nlml = fitted.neg_log_marginal_likelihood()?;
    /// assert!(nlml.is_finite());
    /// # Ok(())
    /// # }
    /// ```
    pub fn neg_log_marginal_likelihood(&self) -> Result<f64, GprError> {
        let noise = self.likelihood.noise_variance();
        let mut log_det_b = 0.0;
        for i in 0..self.m {
            log_det_b += self.b_l[(i, i)].ln();
        }
        log_det_b *= 2.0;
        let n_minus_m = (self.n - self.m) as f64;
        let log_det = n_minus_m * noise.ln() + log_det_b;
        let y_norm2: f64 = self.y.iter().map(|v| v * v).sum();
        let mut ay_dot_w = 0.0;
        for i in 0..self.m {
            let mut ay_i = 0.0;
            for j in 0..self.n {
                ay_i += self.a[(i, j)] * self.y[j];
            }
            ay_dot_w += ay_i * self.w[i];
        }
        let quad = (y_norm2 - ay_dot_w) / noise;
        let trace = (self.k_diag_sum - self.a_frobenius2) / (2.0 * noise);
        let log_two_pi = (2.0 * std::f64::consts::PI).ln();
        Ok(0.5 * ((self.n as f64) * log_two_pi + log_det + quad) + trace)
    }

    /// Predicts at `xs` with [`PredictOptions::default`] (observation variance).
    ///
    /// `xs` is column-major with `n_rows` query points and `n_cols` features.
    /// Returns the diagonal VFE predictive mean and variance.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::DimensionMismatch`] if `n_cols` differs from the
    /// training features, [`GprError::EmptyInput`] if a dimension is zero, or
    /// [`GprError::InvalidHyperparameter`] / [`GprError::NonFiniteInput`] for a
    /// badly packed or non-finite `xs`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{GaussianLikelihood, SparseGpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    /// let likelihood = GaussianLikelihood::new(0.1)?;
    /// let fitted = SparseGpr::new(kernel, likelihood)
    ///     .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0], &[0.0, 1.0], 2)
    ///     .map_err(|(_, e)| e)?;
    /// let pred = fitted.predict(&[0.5], 1, 1)?;
    /// assert_eq!(pred.mean.len(), 1);
    /// # Ok(())
    /// # }
    /// ```
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
    /// Latent variance is the VFE predictive variance of `f*`. Observation
    /// variance adds `σn²`.
    ///
    /// # Errors
    ///
    /// Same as [`Self::predict`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{GaussianLikelihood, PredictOptions, SparseGpr, VarianceKind};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
    /// let likelihood = GaussianLikelihood::new(0.1)?;
    /// let fitted = SparseGpr::new(kernel, likelihood)
    ///     .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0], &[0.0, 1.0], 2)
    ///     .map_err(|(_, e)| e)?;
    /// let pred = fitted.predict_with(
    ///     &[0.5],
    ///     1,
    ///     1,
    ///     PredictOptions {
    ///         variance_kind: VarianceKind::Latent,
    ///     },
    /// )?;
    /// assert_eq!(pred.variance_kind, VarianceKind::Latent);
    /// # Ok(())
    /// # }
    /// ```
    pub fn predict_with(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        options: PredictOptions,
    ) -> Result<Prediction, GprError> {
        if n_cols != self.d {
            return Err(GprError::DimensionMismatch {
                x_dim: n_cols,
                expected_dim: self.d,
            });
        }
        validate_query(xs, n_rows, n_cols)?;
        let compiled = self.kernel.compile();
        let z_mat = pack_points(&self.z_obs, self.m, self.d);
        let mut query_x = Mat::zeros(n_rows, n_cols);
        pack_points_into(xs, n_rows, n_cols, query_x.as_mut());
        let mut k_sz = kernel_cross(&compiled, z_mat.as_ref(), query_x.as_ref())?;
        faer::linalg::triangular_solve::solve_lower_triangular_in_place(
            self.k_mm_l.as_ref(),
            k_sz.as_mut(),
            faer_par_dims(self.m, n_rows),
        );
        let mut kss = vec![0.0; n_rows];
        compiled.fill_diag_points(query_x.as_ref(), &mut kss)?;
        let mut binv_astar = k_sz.clone();
        solve_llt_in_place(self.b_l.as_ref(), binv_astar.as_mut());
        let noise = self.likelihood.noise_variance();
        let mut out = Prediction {
            mean: vec![0.0; n_rows],
            variance: vec![0.0; n_rows],
            variance_kind: options.variance_kind,
        };
        for col in 0..n_rows {
            let mut mean = 0.0;
            let mut a_norm = 0.0;
            let mut binv_norm = 0.0;
            for row in 0..self.m {
                let a_star = k_sz[(row, col)];
                mean += a_star * self.w[row];
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

    #[cfg(test)]
    fn k_mm_l(&self) -> MatRef<'_, f64> {
        self.k_mm_l.as_ref()
    }
}

fn kernel_cross(
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

fn gram_aat_plus_noise(a: MatRef<'_, f64>, noise: f64) -> Mat<f64> {
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

fn frobenius2(a: MatRef<'_, f64>) -> f64 {
    let mut sum = 0.0;
    for col in 0..a.ncols() {
        for row in 0..a.nrows() {
            let v = a[(row, col)];
            sum += v * v;
        }
    }
    sum
}

fn solve_llt_in_place(l: MatRef<'_, f64>, mut rhs: MatMut<'_, f64>) {
    let n = l.nrows();
    let n_rhs = rhs.ncols();
    let par = faer_par_dims(n, n_rhs);
    let req = llt::solve::solve_in_place_scratch::<f64>(n, n_rhs, par);
    let mut buf = MemBuffer::new(req);
    let stack = MemStack::new(&mut buf);
    llt::solve::solve_in_place(l, rhs.as_mut(), par, stack);
}

fn validate_inducing(z: &[f64], m: usize, d: usize) -> Result<(), GprError> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{MaternKernel, MaternNu, RbfArdKernel, RbfKernel, WhiteKernel};
    use crate::{Fixed, Gpr};

    const TOL: f64 = 1e-12;

    fn assert_close(actual: f64, expected: f64) {
        let scale = expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= TOL * scale,
            "actual={actual}, expected={expected}"
        );
    }

    fn factor_ok(
        kernel: KernelSpec,
        x: &[f64],
        n: usize,
        d: usize,
        y: &[f64],
        z: &[f64],
        m: usize,
    ) {
        let likelihood = GaussianLikelihood::new(0.1).expect("noise");
        let fitted = SparseGpr::new(kernel, likelihood)
            .factor(x, n, d, y, z, m)
            .map_err(|(_, e)| e)
            .expect("factor");
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
        let sparse = SparseGpr::new(kernel.clone(), likelihood)
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
        );
        assert_pred_close(&sparse, &exact, x, n, d);
        assert_pred_close(&sparse, &exact, xs_extra, n_extra, d);
    }

    fn assert_pred_close(
        sparse: &FittedSparseGpr,
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
                assert_close(got.mean[i], want.mean[i]);
                assert_close(got.variance[i], want.variance[i]);
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
        let fitted = SparseGpr::new(
            KernelSpec::from(RbfKernel::new(1.0).expect("ℓ")),
            likelihood,
        )
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
        assert_close(reconstructed(0, 0), 1.0);
        assert_close(reconstructed(1, 1), 1.0);
        assert_close(reconstructed(1, 0), off);
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
        let fitted = SparseGpr::new(
            KernelSpec::from(RbfKernel::new(1.0).expect("ℓ")),
            likelihood,
        )
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
        let err = SparseGpr::new(kernel, likelihood)
            .factor(&[], 0, 1, &[], &[0.0], 1)
            .map_err(|(_, e)| e)
            .expect_err("empty n");
        assert_eq!(err, GprError::EmptyInput);
    }

    #[test]
    fn zero_inducing_is_empty_input() {
        let kernel = KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"));
        let likelihood = GaussianLikelihood::new(0.1).expect("noise");
        let err = SparseGpr::new(kernel, likelihood)
            .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0], &[], 0)
            .map_err(|(_, e)| e)
            .expect_err("m = 0");
        assert_eq!(err, GprError::EmptyInput);
    }

    #[test]
    fn inducing_feature_mismatch_is_dimension_mismatch() {
        let kernel = KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"));
        let likelihood = GaussianLikelihood::new(0.1).expect("noise");
        let err = SparseGpr::new(kernel, likelihood)
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
    fn inducing_length_mismatch_is_invalid_hyperparameter() {
        let kernel = KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"));
        let likelihood = GaussianLikelihood::new(0.1).expect("noise");
        let err = SparseGpr::new(kernel, likelihood)
            .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0], &[0.0], 2)
            .map_err(|(_, e)| e)
            .expect_err("short z");
        assert!(matches!(err, GprError::InvalidHyperparameter { .. }));
    }

    #[test]
    fn empty_query_is_empty_input() {
        let kernel = KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"));
        let likelihood = GaussianLikelihood::new(0.1).expect("noise");
        let fitted = SparseGpr::new(kernel, likelihood)
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
        let fitted = SparseGpr::new(kernel, likelihood)
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

    #[test]
    fn is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SparseGpr>();
        assert_send_sync::<FittedSparseGpr>();
    }
}
