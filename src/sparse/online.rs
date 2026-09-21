//! Online variational sparse GPR.

use std::marker::PhantomData;

use faer::Mat;

use crate::error::GprError;
use crate::gpr::PointId;
use crate::gpr::factor::write_params;
use crate::gpr::online::PointRegistry;
use crate::kernel::KernelSpec;
use crate::likelihood::GaussianLikelihood;
use crate::objective::SparseGprObjective;
use crate::optimizer::{Lbfgs, Optimizer};
use crate::{PredictOptions, Prediction};

use super::FixedInducing;
use super::factor::{
    append_column, append_point, assemble_vfe, chol_rank1_downdate, chol_rank1_update, frobenius2,
    kernel_column, kernel_diag_at, point_at, refresh_w, remove_column, remove_point, solve_lmm,
    vfe_neg_log_marginal_likelihood, vfe_predict,
};
use super::fitted::FittedSparseGpr;

/// Online variational sparse GPR after [`FittedSparseGpr::into_online`].
///
/// [`Self::insert`] appends one training point and returns a [`PointId`].
/// [`Self::delete`] removes one point by that identifier. Inducing
/// coordinates stay fixed. VFE factors update with a rank-1 cholupdate of
/// `B` while `K_mm` stays put.
/// Parameters are kernel `θ` then likelihood `θ`. [`Self::set_params`] and
/// [`Self::refit`] rebuild the VFE system.
///
/// # Examples
///
/// ```rust
/// use gprx::kernel::{KernelSpec, RbfKernel};
/// use gprx::{Fixed, GaussianLikelihood, SparseGpr};
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let fitted = SparseGpr::new(
///     KernelSpec::from(RbfKernel::new(1.0)?),
///     GaussianLikelihood::new(0.1)?,
/// )
/// .with_optimizer(Fixed)
/// .factor(
///     &[0.0, 1.0, 2.0, 3.0],
///     4,
///     1,
///     &[0.0, 1.0, 0.5, 0.25],
///     &[0.5, 2.5],
///     2,
/// )
/// .map_err(|(_, e)| e)?;
/// let mut online = fitted.into_online();
/// online.insert(&[4.0], 0.1)?;
/// let pred = online.predict(&[0.5], 1, 1)?;
/// assert_eq!(pred.mean.len(), 1);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct OnlineSparseGpr<O = Lbfgs> {
    kernel: KernelSpec,
    likelihood: GaussianLikelihood,
    optimizer: O,
    x_obs: Vec<f64>,
    z_obs: Vec<f64>,
    y: Vec<f64>,
    k_mm_l: Mat<f64>,
    a: Mat<f64>,
    b_l: Mat<f64>,
    w: Vec<f64>,
    k_diag_sum: f64,
    a_frobenius2: f64,
    n: usize,
    m: usize,
    d: usize,
    registry: PointRegistry,
}

impl<O> OnlineSparseGpr<O> {
    pub(crate) fn from_fitted<I>(fitted: FittedSparseGpr<O, I>) -> Self {
        let registry = PointRegistry::from_count(fitted.n);
        Self {
            kernel: fitted.kernel,
            likelihood: fitted.likelihood,
            optimizer: fitted.optimizer,
            x_obs: fitted.x_obs,
            z_obs: fitted.z_obs,
            y: fitted.y,
            k_mm_l: fitted.k_mm_l,
            a: fitted.a,
            b_l: fitted.b_l,
            w: fitted.w,
            k_diag_sum: fitted.k_diag_sum,
            a_frobenius2: fitted.a_frobenius2,
            n: fitted.n,
            m: fitted.m,
            d: fitted.d,
            registry,
        }
    }

    fn snapshot_fitted(&self) -> FittedSparseGpr<O, FixedInducing>
    where
        O: Clone,
    {
        FittedSparseGpr {
            kernel: self.kernel.clone(),
            likelihood: self.likelihood,
            optimizer: self.optimizer.clone(),
            inducing: PhantomData,
            x_obs: self.x_obs.clone(),
            z_obs: self.z_obs.clone(),
            y: self.y.clone(),
            k_mm_l: self.k_mm_l.clone(),
            a: self.a.clone(),
            b_l: self.b_l.clone(),
            w: self.w.clone(),
            k_diag_sum: self.k_diag_sum,
            a_frobenius2: self.a_frobenius2,
            n: self.n,
            m: self.m,
            d: self.d,
        }
    }

    fn adopt_fitted(&mut self, fitted: FittedSparseGpr<O, FixedInducing>) {
        self.kernel = fitted.kernel;
        self.likelihood = fitted.likelihood;
        self.optimizer = fitted.optimizer;
        self.x_obs = fitted.x_obs;
        self.z_obs = fitted.z_obs;
        self.y = fitted.y;
        self.k_mm_l = fitted.k_mm_l;
        self.a = fitted.a;
        self.b_l = fitted.b_l;
        self.w = fitted.w;
        self.k_diag_sum = fitted.k_diag_sum;
        self.a_frobenius2 = fitted.a_frobenius2;
        self.n = fitted.n;
        self.m = fitted.m;
        self.d = fitted.d;
    }

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

    /// Returns training-point identifiers in buffer order.
    pub fn point_ids(&self) -> &[PointId] {
        self.registry.ids()
    }

    /// Returns the concatenated kernel and likelihood parameter count.
    ///
    /// Inducing coordinates are not counted.
    pub fn num_params(&self) -> usize {
        self.kernel.num_params() + self.likelihood.num_params()
    }

    /// Writes kernel `θ` then likelihood `θ` into `out`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if `out` is the wrong length
    /// or a custom leaf rejects the write.
    pub fn get_params(&self, out: &mut [f64]) -> Result<(), GprError> {
        write_params(&self.kernel, &self.likelihood, out)
    }

    /// Sets kernel then likelihood `θ` and rebuilds the VFE factors.
    ///
    /// `params` matches [`Self::get_params`]. Training `X` / `y` and
    /// inducing `Z` are not changed. Values are committed together only
    /// after the VFE system factors.
    ///
    /// # Errors
    ///
    /// Same as [`FittedSparseGpr::set_params`].
    pub fn set_params(&mut self, params: &[f64]) -> Result<(), GprError>
    where
        O: Clone,
    {
        let mut fitted = self.snapshot_fitted();
        fitted.set_params(params)?;
        self.adopt_fitted(fitted);
        Ok(())
    }

    /// Sets parameters, rebuilds the VFE system, and writes `∂L/∂θ` of the
    /// negative ELBO.
    ///
    /// `params` and `out` match [`Self::get_params`].
    ///
    /// # Errors
    ///
    /// Same as [`FittedSparseGpr::value_and_gradient_into`].
    pub fn value_and_gradient_into(
        &mut self,
        params: &[f64],
        out: &mut [f64],
    ) -> Result<f64, GprError>
    where
        O: Clone,
    {
        let mut fitted = self.snapshot_fitted();
        let value = fitted.value_and_gradient_into(params, out)?;
        self.adopt_fitted(fitted);
        Ok(value)
    }

    /// Writes the Hessian of the negative ELBO (row-major `p×p`) into `out`.
    ///
    /// # Errors
    ///
    /// Same as [`FittedSparseGpr::hessian_into`].
    pub fn hessian_into(&mut self, params: &[f64], out: &mut [f64]) -> Result<(), GprError>
    where
        O: Clone,
    {
        let mut fitted = self.snapshot_fitted();
        fitted.hessian_into(params, out)?;
        self.adopt_fitted(fitted);
        Ok(())
    }

    /// Returns the negative VFE evidence lower bound (the sparse NLML).
    ///
    /// # Errors
    ///
    /// The stored factors are already valid, so this returns `Ok`.
    pub fn neg_log_marginal_likelihood(&self) -> Result<f64, GprError> {
        vfe_neg_log_marginal_likelihood(
            self.a.as_ref(),
            self.b_l.as_ref(),
            &self.w,
            &self.y,
            self.k_diag_sum,
            self.a_frobenius2,
            self.likelihood.noise_variance(),
            self.n,
            self.m,
        )
    }

    /// Predicts at `xs` with [`PredictOptions::default`] (observation variance).
    ///
    /// # Errors
    ///
    /// Same as [`FittedSparseGpr::predict`].
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
    /// # Errors
    ///
    /// Same as [`FittedSparseGpr::predict_with`].
    pub fn predict_with(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        options: PredictOptions,
    ) -> Result<Prediction, GprError> {
        vfe_predict(
            &self.kernel,
            &self.z_obs,
            self.k_mm_l.as_ref(),
            self.b_l.as_ref(),
            &self.w,
            self.likelihood.noise_variance(),
            self.m,
            self.d,
            xs,
            n_rows,
            n_cols,
            options,
        )
    }

    /// Appends one training point at the current `θ` with a rank-1 VFE update.
    ///
    /// `x_new` has length [`Self::d`]. Inducing coordinates are not moved.
    /// The returned [`PointId`] is new and is never reused after a later
    /// [`Self::delete`].
    ///
    /// # Errors
    ///
    /// Returns [`GprError::DimensionMismatch`] if `x_new` is the wrong length,
    /// [`GprError::NonFiniteInput`] if a value is `NaN` or `Inf`, or
    /// [`GprError::EmptyInput`] if `d` is zero.
    pub fn insert(&mut self, x_new: &[f64], y_new: f64) -> Result<PointId, GprError> {
        if x_new.len() != self.d {
            return Err(GprError::DimensionMismatch {
                x_dim: x_new.len(),
                expected_dim: self.d,
            });
        }
        if self.d == 0 {
            return Err(GprError::EmptyInput);
        }
        if x_new.iter().any(|v| !v.is_finite()) || !y_new.is_finite() {
            return Err(GprError::NonFiniteInput);
        }
        let mut a_col = kernel_column(&self.kernel, &self.z_obs, self.m, x_new, self.d)?;
        solve_lmm(self.k_mm_l.as_ref(), a_col.as_mut());
        let mut v = vec![0.0; self.m];
        for (i, slot) in v.iter_mut().enumerate() {
            *slot = a_col[(i, 0)];
        }
        self.a_frobenius2 += frobenius2(a_col.as_ref());
        self.k_diag_sum += kernel_diag_at(&self.kernel, x_new, self.d)?;
        self.a = append_column(&self.a, a_col.as_ref());
        chol_rank1_update(&mut self.b_l, &mut v);
        self.x_obs = append_point(&self.x_obs, self.n, self.d, x_new);
        self.y.push(y_new);
        self.n += 1;
        self.w = refresh_w(self.a.as_ref(), self.b_l.as_ref(), &self.y);
        Ok(self.registry.insert())
    }

    /// Removes the training point identified by `id` and packs every buffer.
    ///
    /// Updates `B` with a rank-1 downdate. If that loses positive
    /// definiteness, the remaining points are factored again. The last
    /// remaining point cannot be deleted.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::EmptyInput`] when `n == 1`, or
    /// [`GprError::InvalidPointId`] when `id` is unknown or already deleted.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Fixed, GaussianLikelihood, SparseGpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let fitted = SparseGpr::new(
    ///     KernelSpec::from(RbfKernel::new(1.0)?),
    ///     GaussianLikelihood::new(0.1)?,
    /// )
    /// .with_optimizer(Fixed)
    /// .factor(
    ///     &[0.0, 1.0, 2.0, 3.0],
    ///     4,
    ///     1,
    ///     &[0.0, 1.0, 0.5, 0.25],
    ///     &[0.5, 2.5],
    ///     2,
    /// )
    /// .map_err(|(_, e)| e)?;
    /// let mut online = fitted.into_online();
    /// let id = online.point_ids()[1];
    /// online.delete(id)?;
    /// assert_eq!(online.n(), 3);
    /// # Ok(())
    /// # }
    /// ```
    pub fn delete(&mut self, id: PointId) -> Result<(), GprError> {
        if self.n <= 1 {
            return Err(GprError::EmptyInput);
        }
        let idx = self.registry.index_of(id)?;
        let mut v = vec![0.0; self.m];
        for (i, slot) in v.iter_mut().enumerate() {
            *slot = self.a[(i, idx)];
        }
        let x_pt = point_at(&self.x_obs, self.n, self.d, idx);
        let diag = kernel_diag_at(&self.kernel, &x_pt, self.d)?;
        let col_norm: f64 = v.iter().map(|value| *value * *value).sum();
        let x_next = remove_point(&self.x_obs, self.n, self.d, idx);
        let mut y_next = self.y.clone();
        y_next.remove(idx);
        let mut b_trial = self.b_l.clone();
        let mut v_trial = v;
        if chol_rank1_downdate(&mut b_trial, &mut v_trial) {
            self.k_diag_sum -= diag;
            self.a_frobenius2 -= col_norm;
            self.a = remove_column(&self.a, idx);
            self.b_l = b_trial;
            self.x_obs = x_next;
            self.y = y_next;
            self.n -= 1;
            self.w = refresh_w(self.a.as_ref(), self.b_l.as_ref(), &self.y);
        } else {
            let state = assemble_vfe(
                &self.kernel,
                self.likelihood,
                &x_next,
                self.n - 1,
                self.d,
                &y_next,
                &self.z_obs,
                self.m,
            )?;
            self.x_obs = x_next;
            self.y = y_next;
            self.n -= 1;
            self.k_mm_l = state.k_mm_l;
            self.a = state.a;
            self.b_l = state.b_l;
            self.w = state.w;
            self.k_diag_sum = state.k_diag_sum;
            self.a_frobenius2 = state.a_frobenius2;
        }
        self.registry.remove_at(idx);
        Ok(())
    }

    /// Converts this model back to a batch sparse GPR with fixed inducing
    /// points.
    ///
    /// Point identifiers are discarded. Parameters stay kernel then
    /// likelihood `θ`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Fixed, GaussianLikelihood, SparseGpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let fitted = SparseGpr::new(
    ///     KernelSpec::from(RbfKernel::new(1.0)?),
    ///     GaussianLikelihood::new(0.1)?,
    /// )
    /// .with_optimizer(Fixed)
    /// .factor(
    ///     &[0.0, 1.0, 2.0, 3.0],
    ///     4,
    ///     1,
    ///     &[0.0, 1.0, 0.5, 0.25],
    ///     &[0.5, 2.5],
    ///     2,
    /// )
    /// .map_err(|(_, e)| e)?;
    /// let online = fitted.into_online();
    /// let fitted = online.into_fitted();
    /// assert_eq!(fitted.n(), 4);
    /// # Ok(())
    /// # }
    /// ```
    pub fn into_fitted(self) -> FittedSparseGpr<O, FixedInducing> {
        FittedSparseGpr {
            kernel: self.kernel,
            likelihood: self.likelihood,
            optimizer: self.optimizer,
            inducing: PhantomData,
            x_obs: self.x_obs,
            z_obs: self.z_obs,
            y: self.y,
            k_mm_l: self.k_mm_l,
            a: self.a,
            b_l: self.b_l,
            w: self.w,
            k_diag_sum: self.k_diag_sum,
            a_frobenius2: self.a_frobenius2,
            n: self.n,
            m: self.m,
            d: self.d,
        }
    }
}

#[allow(private_bounds)]
impl<O> OnlineSparseGpr<O>
where
    O: Clone + for<'a> Optimizer<SparseGprObjective<'a, O, FixedInducing>>,
{
    /// Re-runs the stored optimizer on the stored training data.
    ///
    /// Inducing coordinates stay fixed. The VFE system is rebuilt after the
    /// search.
    ///
    /// # Errors
    ///
    /// Same as [`SparseGpr<O, FixedInducing>::fit`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{GaussianLikelihood, SparseGpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let fitted = SparseGpr::new(
    ///     KernelSpec::from(RbfKernel::new(1.0)?),
    ///     GaussianLikelihood::new(0.1)?,
    /// )
    /// .fit(
    ///     &[0.0, 1.0, 2.0, 3.0],
    ///     4,
    ///     1,
    ///     &[0.0, 1.0, 0.5, 0.25],
    ///     &[0.5, 2.5],
    ///     2,
    /// )
    /// .map_err(|(_, e)| e)?;
    /// let mut online = fitted.into_online();
    /// online.refit()?;
    /// assert!(online.neg_log_marginal_likelihood()?.is_finite());
    /// # Ok(())
    /// # }
    /// ```
    pub fn refit(&mut self) -> Result<(), GprError> {
        let mut fitted = self.snapshot_fitted();
        fitted.optimize_hyperparameters()?;
        self.adopt_fitted(fitted);
        Ok(())
    }
}
