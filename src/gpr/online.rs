//! Incremental tail insert and delete on a converted [`crate::FittedGpr`].

use std::fmt;
use std::marker::PhantomData;
use std::sync::OnceLock;
#[cfg(feature = "insert-stages")]
use std::time::Instant;

use faer::{Mat, MatMut, MatRef};

use crate::data::pack_storage;
use crate::error::PersistErrorKind;
use crate::error::{CholeskyStage, GprError};
use crate::gpr::GprObjective;
use crate::kernel::{
    BlockKind, DistanceKernel, DistanceSlot, DistanceSource, KernelScalar, KernelSpec, ModelKernel,
    ModelKernelParts, PointKernel, PointUse, QuerySources, RectSlots, spec_slots,
};
use crate::kernel::{ScalarOps, SourceStore};
use crate::likelihood::GaussianLikelihood;
use crate::optimizer::Lbfgs;
use crate::optimizer::{Fixed, Optimizer};
use crate::persist::{self, PersistedModel, persist_err};
use crate::precision::{DoublePrecision, GpScalar, StoredFactor};
use crate::transform::{TargetTransform, Transform, UnfittedTarget, UnfittedTransform};
use crate::workspace::{FitWorkspace, QueryWorkspace};
use crate::{PredictOptions, Prediction, PredictiveCovariance};

use super::shared::Query;
use super::{ExactFit, FittedGpr, Gpr, GprCore, LdltStore, LltStore, Policies, fit_buffers};
use crate::points::{PointId, PointRegistry};
use crate::policy::with_kernel_exp;

#[cfg(feature = "insert-stages")]
mod insert_stages {
    use std::cell::Cell;

    thread_local! {
        static KERNEL_S: Cell<f64> = const { Cell::new(0.0) };
        static BORDER_S: Cell<f64> = const { Cell::new(0.0) };
        static REST_S: Cell<f64> = const { Cell::new(0.0) };
    }

    pub(super) fn add_kernel(dt: f64) {
        KERNEL_S.with(|slot| slot.set(slot.get() + dt));
    }

    pub(super) fn add_border(dt: f64) {
        BORDER_S.with(|slot| slot.set(slot.get() + dt));
    }

    pub(super) fn add_rest(dt: f64) {
        REST_S.with(|slot| slot.set(slot.get() + dt));
    }

    pub(super) fn take() -> (f64, f64, f64) {
        (
            KERNEL_S.with(|slot| slot.replace(0.0)),
            BORDER_S.with(|slot| slot.replace(0.0)),
            REST_S.with(|slot| slot.replace(0.0)),
        )
    }
}

/// Returns accumulated `insert` stage seconds `(kernel, border, rest)` and resets them.
///
/// Enabled only with the `insert-stages` crate feature used by `compare/perf`.
#[cfg(feature = "insert-stages")]
pub fn take_insert_stages() -> (f64, f64, f64) {
    insert_stages::take()
}

/// Stores an online Exact GPR after [`FittedGpr::into_online`]: LDLT factor, tail insert, and delete.
///
/// [`Self::insert`] appends one training point with a bordered LDLT update
/// and returns a [`PointId`]. [`Self::delete`] removes one point by that
/// identifier. Predictive mean and variance use the stored LDLT
/// (`L w = k_*`, then `k(x*,x*) − Σ wᵢ² / Dᵢ`). Covariance, sampling, and
/// leave-one-out read the same LDLT. Hyperparameter writes
/// ([`Self::set_params`], [`Self::refit`], gradient, Hessian) run the batch
/// fit code on temporary LLT buffers filled from the LDLT, then write the
/// new factor back. The training data is not copied and [`PointId`] values
/// are kept.
///
/// # Examples
///
/// ```rust
/// use gprx::kernel::{KernelSpec, RbfKernel};
/// use gprx::{Fixed, GaussianLikelihood, Gpr};
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let fitted = Gpr::new(
///     KernelSpec::from(RbfKernel::new(1.0)?),
///     GaussianLikelihood::new(0.1)?,
/// )
/// .with_optimizer(Fixed)
/// .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
/// .map_err(|(_, e)| e)?;
/// let mut online = fitted.into_online()?;
/// online.insert(&[1.5], 0.5)?;
/// let pred = online.predict(&[0.5], 1, 1)?;
/// assert_eq!(pred.mean.len(), 1);
/// # Ok(())
/// # }
/// ```
pub struct OnlineGpr<O = Lbfgs, P: GpScalar = DoublePrecision, K: ModelKernel = KernelSpec> {
    pub(crate) core: GprCore<P, K>,
    pub(crate) optimizer: O,
    pub(crate) workspace: LdltStore<P::Storage>,
    pub(crate) registry: PointRegistry,
    pub(crate) alpha: AlphaState<P>,
    pub(crate) _kernel: PhantomData<K>,
}

/// `α` for an online model. Insert and delete only mark it stale (libgp's
/// `alpha_needs_update`); the first read solves it.
pub(crate) struct AlphaState<P: GpScalar> {
    /// `core.factor_alpha` / `core.alpha` match the current factor.
    fresh: bool,
    /// Stage a failed stale solve reports.
    stage: CholeskyStage,
    /// Solved on the first `&self` read while stale; cleared by every update.
    cache: OnceLock<Result<SolvedAlpha<P>, GprError>>,
}

/// Factor `α` (storage scalar) and predict `α`.
type AlphaRefs<'a, P> = (
    &'a [<P as crate::precision::PrecisionPolicy>::Storage],
    &'a [<P as crate::precision::PrecisionPolicy>::Refine],
);

pub(crate) struct SolvedAlpha<P: GpScalar> {
    factor: Vec<P::Storage>,
    predict: Vec<P::Refine>,
}

impl<P: GpScalar> Clone for SolvedAlpha<P> {
    fn clone(&self) -> Self {
        Self {
            factor: self.factor.clone(),
            predict: self.predict.clone(),
        }
    }
}

impl<P: GpScalar> Clone for AlphaState<P> {
    fn clone(&self) -> Self {
        Self {
            fresh: self.fresh,
            stage: self.stage,
            cache: self.cache.clone(),
        }
    }
}

impl<P: GpScalar> AlphaState<P> {
    fn fresh() -> Self {
        Self {
            fresh: true,
            stage: CholeskyStage::OnlineInsert,
            cache: OnceLock::new(),
        }
    }

    fn mark_stale(&mut self, stage: CholeskyStage) {
        self.fresh = false;
        self.stage = stage;
        self.cache = OnceLock::new();
    }

    fn mark_fresh(&mut self) {
        self.fresh = true;
        self.cache = OnceLock::new();
    }
}

impl<O, P, K> Clone for OnlineGpr<O, P, K>
where
    O: Clone,
    P: GpScalar,
    K: ModelKernel,
{
    fn clone(&self) -> Self {
        Self {
            core: self.core.clone(),
            optimizer: self.optimizer.clone(),
            workspace: self.workspace.clone(),
            registry: self.registry.clone(),
            alpha: self.alpha.clone(),
            _kernel: PhantomData,
        }
    }
}

impl<O, P, K> fmt::Debug for OnlineGpr<O, P, K>
where
    O: fmt::Debug,
    P: GpScalar,
    K: ModelKernel,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OnlineGpr")
            .field("n", &self.core.n)
            .field("d", &self.core.d)
            .field("kernel", &self.core.kernel)
            .field("likelihood", &self.core.likelihood)
            .field("optimizer", &self.optimizer)
            .field("distance_cache", &self.core.policies.distance_cache)
            .field("cholesky_buffer", &self.core.policies.cholesky_buffer)
            .field("math", &self.core.policies.math)
            .field("jitter_policy", &self.core.policies.jitter)
            .finish_non_exhaustive()
    }
}

impl<O, P, K> OnlineGpr<O, P, K>
where
    P: GpScalar,
    K: ModelKernel,
{
    pub(crate) fn from_core(
        core: GprCore<P, K>,
        optimizer: O,
        workspace: LdltStore<P::Storage>,
    ) -> Self {
        let registry = PointRegistry::from_count(core.n);
        Self {
            core,
            optimizer,
            workspace,
            registry,
            alpha: AlphaState::fresh(),
            _kernel: PhantomData,
        }
    }

    /// Drops the LDLT factor and returns a trainer with the current kernel, likelihood, transforms, optimizer, and policies.
    ///
    /// See the example on [`OnlineGpr`].
    pub fn into_trainer(self) -> Gpr<O, P, K> {
        self.core.into_trainer(self.optimizer)
    }

    /// Replaces the optimizer used by a later [`Self::refit`].
    ///
    /// Same incremental-rebuild rule as [`FittedGpr::with_optimizer`].
    ///
    /// See the example on [`OnlineGpr`].
    pub fn with_optimizer<O2>(self, optimizer: O2) -> OnlineGpr<O2, P, K> {
        OnlineGpr {
            core: self.core,
            optimizer,
            workspace: self.workspace,
            registry: self.registry,
            alpha: self.alpha,
            _kernel: PhantomData,
        }
    }

    /// Returns the number of training points.
    ///
    /// See the example on [`OnlineGpr`].
    pub fn n(&self) -> usize {
        self.core.n
    }

    /// Returns the observation-noise model.
    ///
    /// See the example on [`OnlineGpr`].
    pub fn likelihood(&self) -> &GaussianLikelihood {
        &self.core.likelihood
    }

    /// Returns predict `α` for the current training points.
    ///
    /// Insert and delete leave `α` stale; the first call after them solves it.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::CholeskyFailed`] if a [`crate::MixedPrecision`]
    /// model cannot build the `f64` fallback factor.
    ///
    /// See the example on [`OnlineGpr`].
    pub fn alpha(&self) -> Result<&[P::Refine], GprError> {
        Ok(self.alphas()?.1)
    }

    /// Returns the original training targets.
    ///
    /// See the example on [`OnlineGpr`].
    pub fn y(&self) -> &[f64] {
        &self.core.y_obs
    }

    pub(crate) fn factor(&self) -> StoredFactor<'_, P::Storage> {
        StoredFactor::Ldlt(self.ld_factor())
    }

    /// Solves the factor and predict `α` from the stored LDLT.
    fn solve_alpha(&self) -> Result<SolvedAlpha<P>, GprError> {
        let factor = self.core.solve_factor_alpha(self.factor());
        let mut predict = Vec::with_capacity(factor.len());
        self.core.write_predict_alpha(
            self.factor(),
            &factor,
            self.workspace.factor_jitter,
            self.alpha.stage,
            &mut predict,
        )?;
        Ok(SolvedAlpha { factor, predict })
    }

    /// Factor `α` and predict `α`, solving them once while stale.
    fn alphas(&self) -> Result<AlphaRefs<'_, P>, GprError> {
        if self.alpha.fresh {
            return Ok((&self.core.factor_alpha, &self.core.alpha));
        }
        match self.alpha.cache.get_or_init(|| self.solve_alpha()) {
            Ok(solved) => Ok((&solved.factor, &solved.predict)),
            Err(err) => Err(err.clone()),
        }
    }

    /// Makes `core.factor_alpha` / `core.alpha` current before a `&mut self` read.
    fn refresh_alpha(&mut self) -> Result<(), GprError> {
        if self.alpha.fresh {
            return Ok(());
        }
        let solved = match self.alpha.cache.take() {
            Some(result) => result?,
            None => self.solve_alpha()?,
        };
        self.core.factor_alpha = solved.factor;
        self.core.alpha = solved.predict;
        self.alpha.mark_fresh();
        Ok(())
    }

    /// Records the outcome of a hyperparameter write on the LDLT.
    fn after_write<R>(&mut self, result: Result<R, GprError>) -> Result<R, GprError> {
        if result.is_ok() {
            self.alpha.mark_fresh();
        }
        result
    }

    /// Returns training-point identifiers in workspace buffer order.
    ///
    /// See the example on [`OnlineGpr`].
    pub fn point_ids(&self) -> &[PointId] {
        self.registry.ids()
    }

    pub(crate) fn persist_point_ids(&self) -> Vec<u64> {
        self.registry.raw_ids()
    }

    pub(crate) fn persist_next_point_id(&self) -> u64 {
        self.registry.next_id()
    }

    pub(crate) fn apply_persisted_ids(
        &mut self,
        ids: &[u64],
        next_id: u64,
    ) -> Result<(), GprError> {
        let registry = PointRegistry::from_persisted(ids, next_id)?;
        if registry.len() != self.core.n {
            return Err(persist_err(
                PersistErrorKind::Config,
                format!(
                    "point_ids has {} values, expected n = {}",
                    registry.len(),
                    self.core.n
                ),
            ));
        }
        self.registry = registry;
        Ok(())
    }

    /// Diagonal jitter every row of the current factor carries.
    pub(crate) fn factor_jitter(&self) -> f64 {
        self.workspace.factor_jitter
    }

    pub(crate) fn policies(&self) -> Policies {
        self.core.policies
    }

    /// Returns the distance-cache policy carried from the trainer.
    ///
    /// See the example on [`OnlineGpr`].
    pub fn distance_cache_policy(&self) -> crate::DistanceCachePolicy {
        self.core.policies.distance_cache
    }

    /// Returns the Cholesky buffer policy carried from the trainer.
    ///
    /// See the example on [`OnlineGpr`].
    pub fn cholesky_buffer(&self) -> crate::CholeskyBuffer {
        self.core.policies.cholesky_buffer
    }

    /// Returns the kernel `exp` used by fit and predict.
    ///
    /// See the example on [`OnlineGpr`].
    pub fn math(&self) -> crate::KernelExp {
        self.core.policies.math
    }

    /// Returns the jitter retries used when `K + σn² I` fails to factor.
    ///
    /// See the example on [`OnlineGpr`].
    pub fn jitter_policy(&self) -> crate::JitterPolicy {
        self.core.policies.jitter
    }

    pub(crate) fn x_unfitted(&self) -> &dyn UnfittedTransform {
        self.core.x_unfitted.as_ref()
    }

    pub(crate) fn y_unfitted(&self) -> &dyn UnfittedTarget {
        self.core.y_unfitted.as_ref()
    }

    pub(crate) fn x_transform(&self) -> &dyn Transform {
        self.core.x_transform.as_ref()
    }

    pub(crate) fn y_transform(&self) -> &dyn TargetTransform {
        self.core.y_transform.as_ref()
    }

    pub(crate) fn ld_factor(&self) -> MatRef<'_, P::Storage> {
        self.workspace.ld()
    }

    /// Appends one point: coordinates `x_new`, the `n × 1` columns of
    /// supplied distances to the live points, and the target.
    pub(crate) fn insert_point(
        &mut self,
        x_new: &[f64],
        sources: Vec<DistanceSource<'_>>,
        y_new: f64,
    ) -> Result<PointId, GprError> {
        if x_new.len() != self.core.d {
            return Err(GprError::DimensionMismatch {
                x_dim: x_new.len(),
                expected_dim: self.core.d,
            });
        }
        if x_new.iter().any(|v| !v.is_finite()) || !y_new.is_finite() {
            return Err(GprError::NonFiniteInput);
        }
        self.registry.require_room()?;
        let mut columns = QuerySources::<P::Storage>::bind(
            &self.core.slots,
            sources,
            self.core.n,
            1,
            BlockKind::Rect,
        )?;
        #[cfg(feature = "insert-stages")]
        let kernel_start = Instant::now();
        let n = self.core.n;
        let d = self.core.d;
        self.core.query.ensure_at_least(n, 1, d)?;
        let xs_len = d;
        if self.core.query.query_xs.len() < xs_len {
            self.core.query.query_xs.resize(xs_len, 0.0);
        }
        self.core.query.query_xs[..xs_len].copy_from_slice(x_new);
        if d > 0 {
            self.core
                .x_transform
                .apply(&mut self.core.query.query_xs[..xs_len], 1, d)?;
        }
        let mut y_trans = [y_new];
        self.core.y_transform.transform(&mut y_trans)?;
        {
            let x_train = P::Storage::storage_cols(
                self.core.x.as_ref().submatrix(0, 0, n, d),
                &mut self.core.x_cast,
            );
            let dest = MatMut::from_column_major_slice_mut(&mut self.workspace.v_buf[..n], n, 1);
            let QueryWorkspace {
                query_xs,
                query_x,
                query_dist,
                query_scratch,
                query_nested,
                ..
            } = &mut self.core.query;
            pack_storage(
                &query_xs[..xs_len],
                1,
                d,
                query_x.as_mut().submatrix_mut(0, 0, 1, d),
            );
            let table = columns.table();
            let cross =
                (!self.core.slots.is_empty()).then_some(&table as &dyn RectSlots<P::Storage>);
            with_kernel_exp!(self.core.policies.math, M => self.core.compiled.eval_cross_slots::<M>(
                x_train,
                query_x.as_ref().submatrix(0, 0, 1, d),
                cross,
                Some(query_dist.as_mut().submatrix_mut(0, 0, n, 1)),
                dest,
                query_scratch.as_mut().submatrix_mut(0, 0, n, 1),
                query_nested,
                &mut [],
            ))?;
        }
        let mut kss = [P::Storage::from_f64(0.0)];
        self.core.compiled.eval_diag(
            self.core.query.query_x.as_ref().submatrix(0, 0, 1, d),
            &mut kss,
        )?;
        let k_new = kss[0]
            + P::Storage::from_f64(
                self.core.likelihood.noise_variance() + self.workspace.factor_jitter,
            );
        #[cfg(feature = "insert-stages")]
        insert_stages::add_kernel(kernel_start.elapsed().as_secs_f64());
        #[cfg(feature = "insert-stages")]
        let border_start = Instant::now();
        // Stage the squares first: the border is the last step that can fail.
        let staged = self.core.sources.stage_append(columns.raw())?;
        self.workspace.append_border(k_new)?;
        self.core.sources.commit(staged);
        #[cfg(feature = "insert-stages")]
        insert_stages::add_border(border_start.elapsed().as_secs_f64());
        #[cfg(feature = "insert-stages")]
        let rest_start = Instant::now();
        append_colmajor(&mut self.core.x_obs, n, d, x_new);
        self.core.y_obs.push(y_new);
        append_point_mat_inplace(&mut self.core.x, n, &self.core.query.query_xs[..xs_len]);
        self.core.y_train.push(y_trans[0]);
        self.core.n += 1;
        LdltStore::set_f64_prefix(&mut self.workspace.y, &self.core.y_train);
        self.alpha.mark_stale(CholeskyStage::OnlineInsert);
        let id = self.registry.insert();
        #[cfg(feature = "insert-stages")]
        insert_stages::add_rest(rest_start.elapsed().as_secs_f64());
        Ok(id)
    }

    /// Predicts `q` into `out` through the model's query buffers.
    pub(crate) fn predict_query_into(
        &mut self,
        q: Query<'_, P::Storage>,
        options: PredictOptions,
        out: &mut Prediction<P::Refine>,
    ) -> Result<(), GprError> {
        self.refresh_alpha()?;
        let ld = self.workspace.ld();
        self.core
            .predict_with_into(StoredFactor::Ldlt(ld), &mut [], q, options, out)
    }

    /// The predict `α` of the current points (solved once while stale).
    pub(crate) fn predict_alpha(&self) -> Result<&[P::Refine], GprError> {
        Ok(self.alphas()?.1)
    }

    /// Removes the training point identified by `id` and packs every buffer.
    ///
    /// Updates the stored LDLT with
    /// `ldlt::update::delete_rows_and_cols_clobber`. Workspace capacity is
    /// unchanged. `α` is not solved here. The first later read solves it.
    /// The last remaining point cannot be deleted.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InsufficientData`] when `n == 1`, or
    /// [`GprError::InvalidPointId`] when `id` is unknown or already deleted.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Fixed, GaussianLikelihood, Gpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let fitted = Gpr::new(
    ///     KernelSpec::from(RbfKernel::new(1.0)?),
    ///     GaussianLikelihood::new(0.1)?,
    /// )
    /// .with_optimizer(Fixed)
    /// .factor(&[0.0, 1.0, 2.0], 3, 1, &[0.0, 1.0, 0.5])
    /// .map_err(|(_, e)| e)?;
    /// let mut online = fitted.into_online()?;
    /// let id = online.point_ids()[1];
    /// online.delete(id)?;
    /// assert_eq!(online.n(), 2);
    /// # Ok(())
    /// # }
    /// ```
    pub fn delete(&mut self, id: PointId) -> Result<(), GprError> {
        if self.core.n <= 1 {
            return Err(GprError::InsufficientData {
                n: self.core.n,
                min: 2,
            });
        }
        let index = self.registry.index_of(id)?;
        let staged = self.core.sources.stage_delete(index)?;
        self.workspace.delete_index(index)?;
        self.core.sources.commit(staged);
        remove_colmajor(&mut self.core.x_obs, self.core.n, self.core.d, index);
        self.core.y_obs.remove(index);
        remove_point_mat_inplace(&mut self.core.x, self.core.n, index);
        self.core.y_train.remove(index);
        self.registry.remove_at(index);
        self.core.n -= 1;
        LdltStore::set_f64_prefix(&mut self.workspace.y, &self.core.y_train);
        self.alpha.mark_stale(CholeskyStage::OnlineDelete);
        Ok(())
    }

    /// Returns the negative log marginal likelihood from the stored LDLT factor.
    ///
    /// Uses `log|A| = Σ log(Dᵢ)` and the stored `α`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::CholeskyFailed`] if a stored `Dᵢ` is not positive.
    ///
    /// See the example on [`OnlineGpr`].
    pub fn neg_log_marginal_likelihood(&self) -> Result<f64, GprError> {
        let (factor_alpha, _) = self.alphas()?;
        Ok(self
            .core
            .neg_log_marginal_likelihood(self.factor(), factor_alpha))
    }

    /// Returns the concatenated kernel and likelihood parameter count.
    ///
    /// See the example on [`OnlineGpr`].
    pub fn num_params(&self) -> usize {
        self.core.num_params()
    }

    /// Writes kernel `θ` then likelihood `θ` into `out`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::LengthMismatch`] if `out` is the wrong length
    /// or a custom leaf rejects the write.
    ///
    /// See the example on [`OnlineGpr`].
    pub fn get_params(&self, out: &mut [f64]) -> Result<(), GprError> {
        self.core.get_params(out)
    }

    /// Sets kernel then likelihood `θ` and rebuilds the LDLT factor.
    ///
    /// Runs [`FittedGpr::set_params`] on LLT buffers filled from the stored
    /// LDLT and writes the new factor back. Transforms are not re-fit. A
    /// failure leaves this model unchanged.
    ///
    /// # Errors
    ///
    /// Same as [`FittedGpr::set_params`].
    ///
    /// See the example on [`OnlineGpr`].
    pub fn set_params(&mut self, params: &[f64]) -> Result<(), GprError> {
        let result = with_llt_view(&mut self.core, &mut self.workspace, |view| {
            view.set_params(params)
        });
        self.after_write(result)
    }

    /// Writes the joint NLML and gradient at `params`.
    ///
    /// # Errors
    ///
    /// Same as [`FittedGpr::value_and_gradient_into`].
    ///
    /// See the example on [`OnlineGpr`].
    pub fn value_and_gradient_into(
        &mut self,
        params: &[f64],
        out: &mut [f64],
    ) -> Result<f64, GprError> {
        let result = with_llt_view(&mut self.core, &mut self.workspace, |view| {
            view.value_and_gradient_into(params, out)
        });
        self.after_write(result)
    }

    /// Writes the analytic NLML Hessian (row-major `p×p`) at `params`.
    ///
    /// # Errors
    ///
    /// Same as [`FittedGpr::hessian_into`].
    ///
    /// See the example on [`OnlineGpr`].
    pub fn hessian_into(&mut self, params: &[f64], out: &mut [f64]) -> Result<(), GprError> {
        let result = with_llt_view(&mut self.core, &mut self.workspace, |view| {
            view.hessian_into(params, out)
        });
        self.after_write(result)
    }

    /// Returns the leave-one-out predictive mean and variance on the training set.
    ///
    /// # Errors
    ///
    /// Same as [`FittedGpr::loo_predict`].
    ///
    /// See the example on [`OnlineGpr`].
    pub fn loo_predict(&self) -> Result<Prediction<P::Refine>, GprError> {
        self.loo_predict_with(PredictOptions::default())
    }

    /// Returns the leave-one-out prediction with an explicit variance kind.
    ///
    /// # Errors
    ///
    /// Same as [`Self::loo_predict`].
    ///
    /// See the example on [`OnlineGpr`].
    pub fn loo_predict_with(
        &self,
        options: PredictOptions,
    ) -> Result<Prediction<P::Refine>, GprError> {
        let (_, alpha) = self.alphas()?;
        self.core.loo_predict_with(self.factor(), alpha, options)
    }
}

impl<O, P: GpScalar> OnlineGpr<O, P> {
    /// Writes this model to `dir/config.json` and `dir/model.safetensors`.
    ///
    /// Omits the LDLT factor and `α`. [`crate::persist::LoadedGpr::load`]
    /// rebuilds an [`OnlineGpr`] (`factor_kind` is `ldlt`) and restores
    /// [`Self::point_ids`].
    ///
    /// # Errors
    ///
    /// Returns [`GprError::PersistFailed`] when the directory cannot be
    /// created or a Custom leaf / caller transform has no persist form.
    ///
    /// See the example on [`OnlineGpr`].
    pub fn save(&self, dir: impl AsRef<std::path::Path>) -> Result<(), GprError> {
        persist::save_online(self, dir.as_ref(), false)
    }

    /// Writes this model including the packed LDLT factor and `α`.
    ///
    /// # Errors
    ///
    /// Same as [`Self::save`].
    ///
    /// See the example on [`OnlineGpr`].
    pub fn save_with_factor(&self, dir: impl AsRef<std::path::Path>) -> Result<(), GprError> {
        persist::save_online(self, dir.as_ref(), true)
    }
}

impl<O, P: GpScalar> OnlineGpr<O, P> {
    /// Appends one training point at the current `θ` with a bordered LDLT update.
    ///
    /// `x_new` has length [`Self::d`]. Transforms already stored on this model
    /// are applied; they are not re-fit. Grows the online workspace when the
    /// next row does not fit. `α` is not solved here. The first later read
    /// solves it, and [`Self::alpha`] returns [`Result`]. The returned [`PointId`] is new and is never
    /// reused after a later [`Self::delete`].
    ///
    /// # Errors
    ///
    /// Returns [`GprError::DimensionMismatch`] if `x_new` is the wrong length,
    /// [`GprError::NonFiniteInput`] if a value is `NaN` or `Inf`,
    /// [`GprError::EmptyInput`] if the model has no feature (`d = 0`) or the
    /// workspace cannot accept a row,
    /// [`GprError::IndexOutOfRange`] if no new [`PointId`] is left (only a
    /// loaded `next_point_id` near `u64::MAX` reaches this), or
    /// [`GprError::CholeskyFailed`] if the new pivot `δ` is not positive.
    ///
    /// See the example on [`OnlineGpr`].
    pub fn insert(&mut self, x_new: &[f64], y_new: f64) -> Result<PointId, GprError> {
        // A coordinate model reads at least one feature.
        if self.core.d == 0 {
            return Err(GprError::EmptyInput);
        }
        self.insert_point(x_new, Vec::new(), y_new)
    }

    /// Returns the kernel whose hyperparameters this model owns.
    ///
    /// See the example on [`OnlineGpr`].
    pub fn kernel(&self) -> &KernelSpec {
        &self.core.kernel
    }

    /// Predicts at `xs` with [`PredictOptions::default`] (observation variance).
    ///
    /// # Errors
    ///
    /// Same as [`FittedGpr::predict`].
    ///
    /// See the example on [`OnlineGpr`].
    pub fn predict(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
    ) -> Result<Prediction<P::Refine>, GprError> {
        self.predict_with(xs, n_rows, n_cols, PredictOptions::default())
    }

    /// Writes [`Self::predict`] into `out`.
    ///
    /// # Errors
    ///
    /// Same as [`Self::predict`].
    ///
    /// See the example on [`OnlineGpr`].
    pub fn predict_into(
        &mut self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        out: &mut Prediction<P::Refine>,
    ) -> Result<(), GprError> {
        self.predict_with_into(xs, n_rows, n_cols, PredictOptions::default(), out)
    }

    /// Predicts at `xs` with an explicit variance kind.
    ///
    /// Latent variance is `k(x*, x*) − Σ (L⁻¹ k_*)ᵢ² / Dᵢ`. Observation
    /// variance adds `σn²` in the transformed space.
    ///
    /// # Errors
    ///
    /// Same as [`Self::predict`].
    ///
    /// See the example on [`OnlineGpr`].
    pub fn predict_with(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        options: PredictOptions,
    ) -> Result<Prediction<P::Refine>, GprError> {
        let mut out = Prediction::default();
        let (_, alpha) = self.alphas()?;
        self.core.write_prediction(
            self.factor(),
            alpha,
            Query::points(xs, n_rows, n_cols),
            options,
            &mut out,
        )?;
        Ok(out)
    }

    /// Writes [`Self::predict_with`] into `out`.
    ///
    /// # Errors
    ///
    /// Same as [`Self::predict`].
    ///
    /// See the example on [`OnlineGpr`].
    pub fn predict_with_into(
        &mut self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        options: PredictOptions,
        out: &mut Prediction<P::Refine>,
    ) -> Result<(), GprError> {
        self.predict_query_into(Query::points(xs, n_rows, n_cols), options, out)
    }

    /// Returns the predictive mean and query–query covariance at `xs`.
    ///
    /// # Errors
    ///
    /// Same as [`FittedGpr::predict_covariance`].
    ///
    /// See the example on [`OnlineGpr`].
    pub fn predict_covariance(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
    ) -> Result<PredictiveCovariance<P::Refine>, GprError> {
        self.predict_covariance_with(xs, n_rows, n_cols, PredictOptions::default())
    }

    /// Returns query–query covariance with an explicit variance kind.
    ///
    /// # Errors
    ///
    /// Same as [`Self::predict_covariance`].
    ///
    /// See the example on [`OnlineGpr`].
    pub fn predict_covariance_with(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        options: PredictOptions,
    ) -> Result<PredictiveCovariance<P::Refine>, GprError> {
        let (_, alpha) = self.alphas()?;
        self.core.write_covariance(
            self.factor(),
            alpha,
            Query::points(xs, n_rows, n_cols),
            options,
        )
    }

    /// Draws posterior samples at `xs`.
    ///
    /// # Errors
    ///
    /// Same as [`FittedGpr::sample`].
    ///
    /// See the example on [`OnlineGpr`].
    pub fn sample(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        n_draws: usize,
        seed: u64,
    ) -> Result<Vec<P::Refine>, GprError> {
        self.sample_with(xs, n_rows, n_cols, PredictOptions::default(), n_draws, seed)
    }

    /// Draws posterior samples with an explicit variance kind.
    ///
    /// # Errors
    ///
    /// Same as [`Self::sample`].
    ///
    /// See the example on [`OnlineGpr`].
    pub fn sample_with(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        options: PredictOptions,
        n_draws: usize,
        seed: u64,
    ) -> Result<Vec<P::Refine>, GprError> {
        let (_, alpha) = self.alphas()?;
        self.core.sample_with(
            self.factor(),
            alpha,
            Query::points(xs, n_rows, n_cols),
            options,
            n_draws,
            seed,
        )
    }
}

impl<O, P: GpScalar, K: PointKernel> OnlineGpr<O, P, K> {
    /// Returns the feature dimension.
    ///
    /// See the example on [`OnlineGpr`].
    pub fn d(&self) -> usize {
        self.core.d
    }

    /// Returns the original training features in column-major order.
    ///
    /// See the example on [`OnlineGpr`].
    pub fn x(&self) -> &[f64] {
        &self.core.x_obs
    }
}

impl<O, P: GpScalar, C: PointUse> OnlineGpr<O, P, DistanceKernel<C>> {
    /// Returns a copy of the kernel whose hyperparameters this model owns.
    ///
    /// See the example on [`DistanceKernel`].
    pub fn to_kernel(&self) -> DistanceKernel<C> {
        <DistanceKernel<C> as ModelKernelParts>::from_spec(self.core.kernel.clone())
    }

    /// Returns the slots of the kernel, in the order of
    /// [`DistanceKernel::slots`].
    ///
    /// See the example on [`DistanceKernel`].
    pub fn slots(&self) -> Vec<DistanceSlot> {
        spec_slots(&self.core.kernel)
    }
}

impl<O, P, K> OnlineGpr<O, P, K>
where
    P: GpScalar,
    K: ModelKernel,
    O: for<'a> Optimizer<GprObjective<'a, P, K>>,
{
    /// Re-runs the stored optimizer on the stored training data.
    ///
    /// # Errors
    ///
    /// Same as [`FittedGpr::refit`].
    ///
    /// See the example on [`OnlineGpr`].
    pub fn refit(&mut self) -> Result<(), GprError> {
        let optimizer = &self.optimizer;
        let result = with_llt_view(&mut self.core, &mut self.workspace, |view| {
            view.optimize(optimizer)
        });
        self.after_write(result)
    }
}

impl<P: GpScalar, K: ModelKernel> OnlineGpr<Fixed, P, K> {
    /// Rebuilds the LDLT factor at the current `θ` without a search.
    ///
    /// # Errors
    ///
    /// Same as [`Gpr<Fixed>::factor`].
    ///
    /// See the example on [`OnlineGpr`].
    pub fn refit(&mut self) -> Result<(), GprError> {
        let result = with_llt_view(&mut self.core, &mut self.workspace, |view| view.refactor());
        self.after_write(result)
    }
}

impl<P: GpScalar, K: ModelKernel> OnlineGpr<Fixed, P, K> {
    pub(crate) fn from_persisted(parts: PersistedModel<P, K>) -> Result<Self, GprError> {
        FittedGpr::from_persisted(parts)?.into_online_preserving_factor()
    }
}

/// Runs `f` on LLT fit buffers filled from the online LDLT, then writes the
/// resulting factor back.
///
/// `L_llt = L √D`. On failure the LDLT, `α`, and `θ` stay as they were: the
/// fit code restores `θ`, and the `α` it may have rebuilt is put back here.
fn with_llt_view<P, K, R>(
    core: &mut GprCore<P, K>,
    online: &mut LdltStore<P::Storage>,
    f: impl FnOnce(&mut ExactFit<'_, P, K>) -> Result<R, GprError>,
) -> Result<R, GprError>
where
    P: GpScalar,
    K: ModelKernel,
{
    let n = core.n;
    let mut store = LltStore::new(fit_buffers::<P, _>(n, core.policies, &core.compiled)?);
    online.fill_llt_into(store.buffers.core_mut().k_matrix.as_mut(), n);
    store.buffers.core_mut().factor_jitter = online.factor_jitter;
    let factor_alpha = core.factor_alpha.clone();
    let alpha = core.alpha.clone();
    let result = f(&mut ExactFit {
        core: &mut *core,
        store: &mut store,
    });
    match result {
        Ok(value) => {
            online.fill_ld_from_llt(store.l(), n)?;
            online.factor_jitter = store.buffers.core().factor_jitter;
            LdltStore::set_f64_prefix(&mut online.y, &core.y_train);
            LdltStore::set_vector_prefix(&mut online.alpha, &core.factor_alpha);
            Ok(value)
        }
        Err(err) => {
            core.factor_alpha = factor_alpha;
            core.alpha = alpha;
            Err(err)
        }
    }
}

fn append_colmajor(x: &mut Vec<f64>, n: usize, d: usize, x_new: &[f64]) {
    let next_len = (n + 1) * d;
    if x.capacity() < next_len {
        let grow_to = next_len.max(x.capacity().max(1).saturating_mul(2));
        x.reserve(grow_to.saturating_sub(x.len()));
    }
    x.resize(next_len, 0.0);
    for feature in (0..d).rev() {
        let src = feature * n;
        let dest = feature * (n + 1);
        for i in (0..n).rev() {
            x[dest + i] = x[src + i];
        }
        x[dest + n] = x_new[feature];
    }
}

fn append_point_mat_inplace(x: &mut Mat<f64>, n: usize, x_new: &[f64]) {
    let d = x.ncols();
    if x.nrows() <= n {
        let new_rows = (n + 1).max(x.nrows().max(1).saturating_mul(2));
        let mut next = Mat::zeros(new_rows, d);
        for feature in 0..d {
            for i in 0..n {
                next[(i, feature)] = x[(i, feature)];
            }
        }
        *x = next;
    }
    for feature in 0..d {
        x[(n, feature)] = x_new[feature];
    }
}

fn remove_colmajor(x: &mut Vec<f64>, n: usize, d: usize, index: usize) {
    let mut next = Vec::with_capacity((n - 1) * d);
    for feature in 0..d {
        for i in 0..n {
            if i != index {
                next.push(x[feature * n + i]);
            }
        }
    }
    *x = next;
}

fn remove_point_mat_inplace(x: &mut Mat<f64>, n: usize, index: usize) {
    let d = x.ncols();
    for feature in 0..d {
        for i in index..(n - 1) {
            x[(i, feature)] = x[(i + 1, feature)];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{KernelSpec, MaternKernel, MaternNu, RbfArdKernel, RbfKernel, WhiteKernel};
    use crate::persist::{LoadedGpr, PersistRegistry};
    use crate::points::RegistryId;
    use crate::{Fixed, GaussianLikelihood, GprError, PointId};

    const TOL: f64 = 1e-12;

    use crate::test_check::assert_mean_var_close;

    #[allow(clippy::too_many_arguments)]
    fn insert_matches_factor(
        kernel: KernelSpec,
        x2: &[f64],
        y2: &[f64],
        n: usize,
        d: usize,
        x_new: &[f64],
        y_new: f64,
        xs: &[f64],
        n_query: usize,
    ) {
        let likelihood = GaussianLikelihood::new(0.1).expect("noise");
        let fitted = Gpr::new(kernel.clone(), likelihood)
            .with_optimizer(Fixed)
            .factor(x2, n, d, y2)
            .map_err(|(_, e)| e)
            .expect("factor n");
        let mut online = fitted.into_online().expect("into_online");
        online.insert(x_new, y_new).expect("insert");
        let got = online.predict(xs, n_query, d).expect("online predict");

        let mut x3 = x2.to_vec();
        append_colmajor(&mut x3, n, d, x_new);
        let mut y3 = y2.to_vec();
        y3.push(y_new);
        let full = Gpr::new(kernel, likelihood)
            .with_optimizer(Fixed)
            .factor(&x3, n + 1, d, &y3)
            .map_err(|(_, e)| e)
            .expect("factor n+1");
        let want = full.predict(xs, n_query, d).expect("full predict");
        assert_mean_var_close(&got.mean, &got.variance, &want.mean, &want.variance, TOL);
    }

    #[test]
    fn insert_rbf_matches_factor() {
        insert_matches_factor(
            KernelSpec::from(RbfKernel::new(1.0).expect("ℓ")),
            &[0.0, 1.0],
            &[0.0, 1.0],
            2,
            1,
            &[1.5],
            0.5,
            &[0.5],
            1,
        );
    }

    #[test]
    fn insert_matern_three_halves_matches_factor() {
        insert_matches_factor(
            KernelSpec::from(MaternKernel::new(1.0, MaternNu::ThreeHalves).expect("ℓ")),
            &[0.0, 1.0],
            &[0.0, 1.0],
            2,
            1,
            &[1.5],
            0.5,
            &[0.5],
            1,
        );
    }

    #[test]
    fn insert_rbf_ard_2d_matches_factor() {
        insert_matches_factor(
            KernelSpec::from(RbfArdKernel::new(&[1.0, 1.5]).expect("ℓ")),
            &[0.0, 1.0, 0.0, 1.0],
            &[0.0, 1.0],
            2,
            2,
            &[0.5, 0.25],
            0.4,
            &[0.25, 0.75],
            1,
        );
    }

    #[test]
    fn insert_rbf_plus_white_matches_factor() {
        insert_matches_factor(
            KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"))
                + KernelSpec::from(WhiteKernel::new(0.05).expect("white")),
            &[0.0, 1.0],
            &[0.0, 1.0],
            2,
            1,
            &[1.5],
            0.5,
            &[0.5],
            1,
        );
    }

    #[test]
    fn online_save_load_roundtrip() {
        let fitted = Gpr::new(
            KernelSpec::from(RbfKernel::new(1.0).expect("ℓ")),
            GaussianLikelihood::new(0.1).expect("noise"),
        )
        .with_optimizer(Fixed)
        .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
        .map_err(|(_, e)| e)
        .expect("factor");
        let mut online = fitted.into_online().expect("into_online");
        online.insert(&[1.5], 0.5).expect("insert");
        let want = online.predict(&[0.5], 1, 1).expect("predict");
        let dir = std::env::temp_dir().join(format!(
            "gprx-online-save-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        online.save_with_factor(&dir).expect("save");
        let loaded = LoadedGpr::load(&dir, &PersistRegistry::new()).expect("load");
        let LoadedGpr::OnlineDouble(model) = loaded else {
            panic!("online RBF should load as OnlineDouble");
        };
        let got = model.predict(&[0.5], 1, 1).expect("loaded predict");
        assert_mean_var_close(&got.mean, &got.variance, &want.mean, &want.variance, TOL);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[allow(clippy::too_many_arguments)]
    fn delete_matches_factor(
        kernel: KernelSpec,
        x3: &[f64],
        y3: &[f64],
        n: usize,
        d: usize,
        index: usize,
        xs: &[f64],
        n_query: usize,
    ) {
        let likelihood = GaussianLikelihood::new(0.1).expect("noise");
        let fitted = Gpr::new(kernel.clone(), likelihood)
            .with_optimizer(Fixed)
            .factor(x3, n, d, y3)
            .map_err(|(_, e)| e)
            .expect("factor n");
        let mut online = fitted.into_online().expect("into_online");
        let cap = online.workspace.n_capacity;
        let id = online.point_ids()[index];
        online.delete(id).expect("delete");
        assert_eq!(online.n(), n - 1);
        assert_eq!(online.workspace.n_capacity, cap);
        let got = online.predict(xs, n_query, d).expect("online predict");

        let mut x_rest = x3.to_vec();
        remove_colmajor(&mut x_rest, n, d, index);
        let mut y_rest = y3.to_vec();
        y_rest.remove(index);
        let full = Gpr::new(kernel, likelihood)
            .with_optimizer(Fixed)
            .factor(&x_rest, n - 1, d, &y_rest)
            .map_err(|(_, e)| e)
            .expect("factor remaining");
        let want = full.predict(xs, n_query, d).expect("full predict");
        assert_mean_var_close(&got.mean, &got.variance, &want.mean, &want.variance, TOL);
    }

    fn delete_each_index(
        kernel: KernelSpec,
        x3: &[f64],
        y3: &[f64],
        n: usize,
        d: usize,
        xs: &[f64],
        n_query: usize,
    ) {
        for index in 0..n {
            delete_matches_factor(kernel.clone(), x3, y3, n, d, index, xs, n_query);
        }
    }

    #[test]
    fn delete_rbf_matches_factor() {
        delete_each_index(
            KernelSpec::from(RbfKernel::new(1.0).expect("ℓ")),
            &[0.0, 1.0, 2.0],
            &[0.0, 1.0, 0.5],
            3,
            1,
            &[0.5],
            1,
        );
    }

    #[test]
    fn delete_matern_three_halves_matches_factor() {
        delete_each_index(
            KernelSpec::from(MaternKernel::new(1.0, MaternNu::ThreeHalves).expect("ℓ")),
            &[0.0, 1.0, 2.0],
            &[0.0, 1.0, 0.5],
            3,
            1,
            &[0.5],
            1,
        );
    }

    #[test]
    fn delete_rbf_ard_2d_matches_factor() {
        delete_each_index(
            KernelSpec::from(RbfArdKernel::new(&[1.0, 1.5]).expect("ℓ")),
            &[0.0, 1.0, 0.5, 0.0, 1.0, 0.25],
            &[0.0, 1.0, 0.4],
            3,
            2,
            &[0.25, 0.75],
            1,
        );
    }

    #[test]
    fn delete_rbf_plus_white_matches_factor() {
        delete_each_index(
            KernelSpec::from(RbfKernel::new(1.0).expect("ℓ"))
                + KernelSpec::from(WhiteKernel::new(0.05).expect("white")),
            &[0.0, 1.0, 2.0],
            &[0.0, 1.0, 0.5],
            3,
            1,
            &[0.5],
            1,
        );
    }

    #[test]
    fn delete_unknown_id_is_invalid_point_id() {
        let fitted = Gpr::new(
            KernelSpec::from(RbfKernel::new(1.0).expect("ℓ")),
            GaussianLikelihood::new(0.1).expect("noise"),
        )
        .with_optimizer(Fixed)
        .factor(&[0.0, 1.0, 2.0], 3, 1, &[0.0, 1.0, 0.5])
        .map_err(|(_, e)| e)
        .expect("factor");
        let mut online = fitted.into_online().expect("into_online");
        let gone = online.point_ids()[1];
        online.delete(gone).expect("first delete");
        assert_eq!(online.delete(gone), Err(GprError::InvalidPointId));
        assert_eq!(
            online.delete(PointId::from_raw(99)),
            Err(GprError::InvalidPointId)
        );
    }

    #[test]
    fn delete_last_point_is_insufficient_data() {
        let fitted = Gpr::new(
            KernelSpec::from(RbfKernel::new(1.0).expect("ℓ")),
            GaussianLikelihood::new(0.1).expect("noise"),
        )
        .with_optimizer(Fixed)
        .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
        .map_err(|(_, e)| e)
        .expect("factor");
        let mut online = fitted.into_online().expect("into_online");
        let ids = online.point_ids().to_vec();
        online.delete(ids[0]).expect("first delete");
        assert_eq!(
            online.delete(ids[1]),
            Err(GprError::InsufficientData { n: 1, min: 2 })
        );
        assert_eq!(online.n(), 1);
    }

    #[test]
    fn delete_save_load_keeps_point_ids() {
        let fitted = Gpr::new(
            KernelSpec::from(RbfKernel::new(1.0).expect("ℓ")),
            GaussianLikelihood::new(0.1).expect("noise"),
        )
        .with_optimizer(Fixed)
        .factor(&[0.0, 1.0, 2.0], 3, 1, &[0.0, 1.0, 0.5])
        .map_err(|(_, e)| e)
        .expect("factor");
        let mut online = fitted.into_online().expect("into_online");
        let middle = online.point_ids()[1];
        online.delete(middle).expect("delete");
        let want_ids = online.point_ids().to_vec();
        let want = online.predict(&[0.5], 1, 1).expect("predict");
        let dir = std::env::temp_dir().join(format!(
            "gprx-online-delete-save-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        online.save_with_factor(&dir).expect("save");
        let loaded = LoadedGpr::load(&dir, &PersistRegistry::new()).expect("load");
        let LoadedGpr::OnlineDouble(model) = loaded else {
            panic!("online RBF should load as OnlineDouble");
        };
        assert_eq!(model.point_ids(), want_ids.as_slice());
        let got = model.predict(&[0.5], 1, 1).expect("loaded predict");
        assert_mean_var_close(&got.mean, &got.variance, &want.mean, &want.variance, TOL);
        let _ = std::fs::remove_dir_all(&dir);
    }

    const MATCH_TOL: f64 = 1e-9;

    /// Online model grown by one insert, and the batch fit on the same four points.
    fn online_and_batch() -> (OnlineGpr<Fixed>, FittedGpr<Fixed>) {
        let kernel = KernelSpec::from(RbfKernel::new(0.8).expect("ell"));
        let likelihood = GaussianLikelihood::new(0.1).expect("noise");
        let x = [0.0, 0.7, 1.9, 2.4];
        let y = [0.3, -0.2, 0.8, 0.1];
        let mut online = Gpr::new(kernel.clone(), likelihood)
            .with_optimizer(Fixed)
            .factor(&x[..3], 3, 1, &y[..3])
            .map_err(|(_, e)| e)
            .expect("factor 3")
            .into_online()
            .expect("into_online");
        online.insert(&x[3..], y[3]).expect("insert");
        let batch = Gpr::new(kernel, likelihood)
            .with_optimizer(Fixed)
            .factor(&x, 4, 1, &y)
            .map_err(|(_, e)| e)
            .expect("factor 4");
        (online, batch)
    }

    fn assert_all_close(got: &[f64], want: &[f64]) {
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want) {
            assert!(
                (g - w).abs() <= MATCH_TOL * w.abs().max(1.0),
                "got={g} want={w}"
            );
        }
    }

    #[test]
    fn online_reads_match_batch() {
        let (online, batch) = online_and_batch();
        let xs = [0.35, 1.2, 3.0];
        let got = online.predict_covariance(&xs, 3, 1).expect("online cov");
        let want = batch.predict_covariance(&xs, 3, 1).expect("batch cov");
        assert_all_close(&got.mean, &want.mean);
        assert_all_close(&got.covariance, &want.covariance);
        let got = online.sample(&xs, 3, 1, 4, 11).expect("online sample");
        let want = batch.sample(&xs, 3, 1, 4, 11).expect("batch sample");
        assert_all_close(&got, &want);
        let got = online.loo_predict().expect("online loo");
        let want = batch.loo_predict().expect("batch loo");
        assert_all_close(&got.mean, &want.mean);
        assert_all_close(&got.variance, &want.variance);
        assert_all_close(
            &[online.neg_log_marginal_likelihood().expect("online nlml")],
            &[batch.neg_log_marginal_likelihood().expect("batch nlml")],
        );
    }

    #[test]
    fn online_hyperparameter_writes_match_batch() {
        let (mut online, mut batch) = online_and_batch();
        let ids = online.point_ids().to_vec();
        let params = [1.3_f64.ln(), 0.05_f64.ln()];
        let mut got_grad = [0.0; 2];
        let mut want_grad = [0.0; 2];
        let got = online
            .value_and_gradient_into(&params, &mut got_grad)
            .expect("online grad");
        let want = batch
            .value_and_gradient_into(&params, &mut want_grad)
            .expect("batch grad");
        assert_all_close(&[got], &[want]);
        assert_all_close(&got_grad, &want_grad);
        let mut got_hess = [0.0; 4];
        let mut want_hess = [0.0; 4];
        online
            .hessian_into(&params, &mut got_hess)
            .expect("online hess");
        batch
            .hessian_into(&params, &mut want_hess)
            .expect("batch hess");
        assert_all_close(&got_hess, &want_hess);
        let moved = [0.6_f64.ln(), 0.2_f64.ln()];
        online.set_params(&moved).expect("online set");
        batch.set_params(&moved).expect("batch set");
        let got = online.predict(&[0.5, 2.0], 2, 1).expect("online predict");
        let want = batch.predict(&[0.5, 2.0], 2, 1).expect("batch predict");
        assert_all_close(&got.mean, &want.mean);
        assert_all_close(&got.variance, &want.variance);
        assert_all_close(online.alpha().expect("alpha"), batch.alpha());
        assert_eq!(online.point_ids(), ids.as_slice());
        online.insert(&[3.1], 0.4).expect("insert after set_params");
        assert_eq!(online.n(), 5);
    }

    #[test]
    fn failed_set_params_leaves_online_unchanged() {
        let (mut online, _) = online_and_batch();
        let mut before = [0.0; 2];
        online.get_params(&mut before).expect("len");
        let alpha = online.alpha().expect("alpha").to_vec();
        let nlml = online.neg_log_marginal_likelihood().expect("nlml");
        let bad = [before[0], f64::INFINITY];
        assert!(matches!(
            online.set_params(&bad),
            Err(GprError::InvalidNoiseVariance { .. })
        ));
        let mut after = [0.0; 2];
        online.get_params(&mut after).expect("len");
        assert_eq!(after.map(f64::to_bits), before.map(f64::to_bits));
        assert_eq!(online.alpha().expect("alpha"), alpha.as_slice());
        assert_eq!(
            online
                .neg_log_marginal_likelihood()
                .expect("nlml")
                .to_bits(),
            nlml.to_bits()
        );
    }

    #[test]
    fn stale_alpha_is_solved_on_first_read() {
        let (mut online, batch) = online_and_batch();
        assert!(!online.alpha.fresh);
        let cached = online.alpha().expect("stale alpha").to_vec();
        assert!(!online.alpha.fresh);
        assert_all_close(&cached, batch.alpha());
        let mut into = Prediction::default();
        online
            .predict_into(&[0.5, 2.0], 2, 1, &mut into)
            .expect("predict_into");
        assert!(online.alpha.fresh);
        let want = batch.predict(&[0.5, 2.0], 2, 1).expect("batch predict");
        assert_all_close(&into.mean, &want.mean);
        assert_all_close(&into.variance, &want.variance);
        let id = online.point_ids()[1];
        online.delete(id).expect("delete");
        assert!(!online.alpha.fresh);
        let rebuilt = Gpr::new(
            KernelSpec::from(RbfKernel::new(0.8).expect("ell")),
            GaussianLikelihood::new(0.1).expect("noise"),
        )
        .with_optimizer(Fixed)
        .factor(&[0.0, 1.9, 2.4], 3, 1, &[0.3, 0.8, 0.1])
        .map_err(|(_, e)| e)
        .expect("factor 3");
        assert_all_close(
            &[online.neg_log_marginal_likelihood().expect("online nlml")],
            &[rebuilt.neg_log_marginal_likelihood().expect("batch nlml")],
        );
        assert_all_close(online.alpha().expect("alpha"), rebuilt.alpha());
    }
}
