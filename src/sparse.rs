//! Variational sparse GPR with caller-supplied inducing points.

use std::marker::PhantomData;

use dyn_stack::MemBuffer;
use faer::Mat;
#[cfg(test)]
use faer::MatRef;
use faer::linalg::cholesky::llt;

use crate::error::{CholeskyStage, GprError};
use crate::gpr::JitterPolicy;
use crate::gpr::factor::{cholesky_lower_with_policy, pack_points, validate_training};
use crate::kernel::{KernelSpec, Triangle};
use crate::likelihood::GaussianLikelihood;
use crate::optimizer::Fixed;
use crate::workspace::faer_par;

/// Trainer for variational sparse GPR at a fixed inducing set `Z`.
///
/// This row only implements [`SparseGpr<Fixed>::factor`]. Hyperparameter
/// search is P4-4. Predictive mean / variance and the ELBO are P4-3.
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
/// Stores the LLT of `K_mm = k(Z, Z)`. Observation noise is not added to
/// `K_mm`. Predict and the ELBO are P4-3.
#[derive(Clone, Debug)]
pub struct FittedSparseGpr<O = Fixed> {
    kernel: KernelSpec,
    likelihood: GaussianLikelihood,
    x_obs: Vec<f64>,
    z_obs: Vec<f64>,
    y: Vec<f64>,
    /// Lower `L` from `K_mm = L Lᵀ`. Tests reconstruct the Gram; P4-3 reads it.
    #[allow(dead_code)]
    k_mm_l: Mat<f64>,
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
    /// added to `K_mm`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::EmptyInput`] when `n`, `d`, or `m` is zero.
    /// Returns [`GprError::DimensionMismatch`] when `z` is packed with a
    /// different feature count than `x` (the same `n_cols` is required).
    /// Length and finiteness errors match [`crate::Gpr<Fixed>::factor`].
    /// [`GprError::CholeskyFailed`] when `K_mm` cannot be factored.
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
        Ok(FittedSparseGpr {
            kernel: self.kernel.clone(),
            likelihood: self.likelihood,
            x_obs: x.to_vec(),
            z_obs: z.to_vec(),
            y: y.to_vec(),
            k_mm_l: k_mm,
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

    #[cfg(test)]
    fn k_mm_l(&self) -> MatRef<'_, f64> {
        self.k_mm_l.as_ref()
    }
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
    fn is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SparseGpr>();
        assert_send_sync::<FittedSparseGpr>();
    }
}
