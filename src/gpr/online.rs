//! Incremental tail insert and delete on a converted [`crate::FittedGpr`].

use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;
#[cfg(feature = "insert-stages")]
use std::time::Instant;

use faer::{Mat, MatMut, MatRef};

use crate::data::{pack_storage, validate_query};
use crate::error::GprError;
use crate::kernel::ScalarOps;
use crate::kernel::{CompiledKernel, KernelScalar, KernelSpec};
use crate::likelihood::GaussianLikelihood;
use crate::objective::GprObjective;
use crate::online::OnlineWorkspace;
use crate::optimizer::Lbfgs;
use crate::optimizer::{Fixed, FullRecompute, Optimizer, PoleRecompute};
use crate::persist::{self, PersistedModel, persist_err};
use crate::precision::{DoublePrecision, GpScalar};
use crate::transform::{TargetTransform, Transform, UnfittedTarget, UnfittedTransform};
use crate::workspace::QueryWorkspace;
use crate::{PredictOptions, Prediction, PredictiveCovariance};

use super::{
    AllocWorkspace, DistanceCacheSlot, FittedGpr, Gpr, JitterPolicy, PointId, RetainCholesky,
};

#[derive(Clone, Debug)]
pub(crate) struct PointRegistry {
    id_to_index: HashMap<PointId, usize>,
    index_to_id: Vec<PointId>,
    next_id: u64,
}

impl PointRegistry {
    pub(crate) fn from_count(n: usize) -> Self {
        let index_to_id: Vec<PointId> = (0..n as u64).map(PointId::from_raw).collect();
        let id_to_index = index_to_id
            .iter()
            .copied()
            .enumerate()
            .map(|(index, id)| (id, index))
            .collect();
        Self {
            id_to_index,
            index_to_id,
            next_id: n as u64,
        }
    }

    fn from_persisted(ids: &[u64], next_id: u64) -> Result<Self, GprError> {
        let mut id_to_index = HashMap::with_capacity(ids.len());
        let mut index_to_id = Vec::with_capacity(ids.len());
        let mut max_id = None;
        for (index, &raw) in ids.iter().enumerate() {
            let id = PointId::from_raw(raw);
            if id_to_index.insert(id, index).is_some() {
                return Err(persist_err("ldlt config has duplicate point_ids"));
            }
            index_to_id.push(id);
            max_id = Some(max_id.map_or(raw, |seen: u64| seen.max(raw)));
        }
        if let Some(max_id) = max_id {
            if next_id <= max_id {
                return Err(persist_err(
                    "ldlt config next_point_id must exceed every stored PointId",
                ));
            }
        }
        Ok(Self {
            id_to_index,
            index_to_id,
            next_id,
        })
    }

    pub(crate) fn ids(&self) -> &[PointId] {
        &self.index_to_id
    }

    fn next_id(&self) -> u64 {
        self.next_id
    }

    fn len(&self) -> usize {
        self.index_to_id.len()
    }

    pub(crate) fn index_of(&self, id: PointId) -> Result<usize, GprError> {
        self.id_to_index
            .get(&id)
            .copied()
            .ok_or(GprError::InvalidPointId)
    }

    pub(crate) fn insert(&mut self) -> PointId {
        let id = PointId::from_raw(self.next_id);
        let index = self.index_to_id.len();
        self.next_id = self.next_id.saturating_add(1);
        self.index_to_id.push(id);
        self.id_to_index.insert(id, index);
        id
    }
}

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
#[doc(hidden)]
pub fn take_insert_stages() -> (f64, f64, f64) {
    insert_stages::take()
}

impl PointRegistry {
    pub(crate) fn remove_at(&mut self, index: usize) {
        let id = self.index_to_id.remove(index);
        self.id_to_index.remove(&id);
        for (shifted, remaining) in self.index_to_id.iter().enumerate().skip(index) {
            self.id_to_index.insert(*remaining, shifted);
        }
    }
}

/// Online Exact GPR after [`FittedGpr::into_online`]: LDLT factor, tail insert, and delete.
///
/// [`Self::insert`] appends one training point with a bordered LDLT update
/// and returns a [`PointId`]. [`Self::delete`] removes one point by that
/// identifier. Predictive mean and variance use the stored LDLT
/// (`L w = k_*`, then `k(x*,x*) − Σ wᵢ² / Dᵢ`). Hyperparameter writes
/// ([`Self::set_params`], [`Self::refit`], gradient, Hessian) snapshot
/// through a batch [`FittedGpr`] at the current `θ` and convert back,
/// keeping the same [`PointId`] values.
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
#[allow(private_bounds)] // `DistanceCacheSlot` / `AllocWorkspace` are crate-private.
pub struct OnlineGpr<
    O = Lbfgs,
    S = FullRecompute,
    C: DistanceCacheSlot = crate::CachedDistances,
    B: AllocWorkspace = RetainCholesky,
    M = crate::math::Accurate,
    P: GpScalar = DoublePrecision,
> {
    pub(crate) kernel: KernelSpec,
    pub(crate) compiled: CompiledKernel<P::Storage>,
    pub(crate) likelihood: GaussianLikelihood,
    pub(crate) x_unfitted: Box<dyn UnfittedTransform>,
    pub(crate) y_unfitted: Box<dyn UnfittedTarget>,
    pub(crate) x_transform: Box<dyn Transform>,
    pub(crate) y_transform: Box<dyn TargetTransform>,
    pub(crate) optimizer: O,
    pub(crate) distance_cache: C,
    pub(crate) jitter_policy: JitterPolicy,
    pub(crate) workspace: OnlineWorkspace<P::Storage>,
    pub(crate) query: QueryWorkspace<P>,
    pub(crate) x_obs: Vec<f64>,
    pub(crate) y_obs: Vec<f64>,
    pub(crate) x: Mat<f64>,
    pub(crate) y_train: Vec<f64>,
    pub(crate) factor_alpha: Vec<P::Storage>,
    pub(crate) alpha: Vec<P::Refine>,
    pub(crate) x_cast: <P::Storage as crate::kernel::ScalarOps>::ColCast,
    pub(crate) y_cast: <P::Storage as crate::kernel::ScalarOps>::RowCast,
    pub(crate) n: usize,
    pub(crate) d: usize,
    pub(crate) registry: PointRegistry,
    pub(crate) _recompute: PhantomData<S>,
    pub(crate) _math: PhantomData<M>,
    pub(crate) _cholesky: PhantomData<B>,
}

impl<O, S, C, B, M, P> Clone for OnlineGpr<O, S, C, B, M, P>
where
    O: Clone,
    C: Copy + DistanceCacheSlot,
    B: AllocWorkspace,
    P: GpScalar,
{
    fn clone(&self) -> Self {
        Self {
            kernel: self.kernel.clone(),
            compiled: self.compiled.clone(),
            likelihood: self.likelihood,
            x_unfitted: self.x_unfitted.clone_box(),
            y_unfitted: self.y_unfitted.clone_box(),
            x_transform: self.x_transform.clone_box(),
            y_transform: self.y_transform.clone_box(),
            optimizer: self.optimizer.clone(),
            distance_cache: self.distance_cache,
            jitter_policy: self.jitter_policy,
            workspace: self.workspace.clone(),
            query: self.query.clone(),
            x_obs: self.x_obs.clone(),
            y_obs: self.y_obs.clone(),
            x: self.x.clone(),
            y_train: self.y_train.clone(),
            factor_alpha: self.factor_alpha.clone(),
            alpha: self.alpha.clone(),
            x_cast: self.x_cast.clone(),
            y_cast: self.y_cast.clone(),
            n: self.n,
            d: self.d,
            registry: self.registry.clone(),
            _recompute: PhantomData,
            _math: PhantomData,
            _cholesky: PhantomData,
        }
    }
}

impl<O, S, C, B, M, P> fmt::Debug for OnlineGpr<O, S, C, B, M, P>
where
    O: fmt::Debug,
    C: fmt::Debug + DistanceCacheSlot,
    B: AllocWorkspace,
    P: GpScalar,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OnlineGpr")
            .field("n", &self.n)
            .field("d", &self.d)
            .field("kernel", &self.kernel)
            .field("likelihood", &self.likelihood)
            .field("distance_cache", &self.distance_cache)
            .field("jitter_policy", &self.jitter_policy)
            .finish_non_exhaustive()
    }
}

#[allow(private_bounds)] // `DistanceCacheSlot` is crate-private; insert and predict read it.
impl<O, S, C, B, M, P> OnlineGpr<O, S, C, B, M, P>
where
    C: DistanceCacheSlot,
    B: AllocWorkspace,
    P: GpScalar,
    M: crate::math::KernelMath,
{
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_parts(
        kernel: KernelSpec,
        compiled: CompiledKernel<P::Storage>,
        likelihood: GaussianLikelihood,
        x_unfitted: Box<dyn UnfittedTransform>,
        y_unfitted: Box<dyn UnfittedTarget>,
        x_transform: Box<dyn Transform>,
        y_transform: Box<dyn TargetTransform>,
        optimizer: O,
        distance_cache: C,
        jitter_policy: JitterPolicy,
        workspace: OnlineWorkspace<P::Storage>,
        query: QueryWorkspace<P>,
        x_obs: Vec<f64>,
        y_obs: Vec<f64>,
        x: Mat<f64>,
        y_train: Vec<f64>,
        factor_alpha: Vec<P::Storage>,
        alpha: Vec<P::Refine>,
        n: usize,
        d: usize,
    ) -> Self {
        Self {
            kernel,
            compiled,
            likelihood,
            x_unfitted,
            y_unfitted,
            x_transform,
            y_transform,
            optimizer,
            distance_cache,
            jitter_policy,
            workspace,
            query,
            x_obs,
            y_obs,
            x,
            y_train,
            factor_alpha,
            alpha,
            x_cast: P::Storage::empty_cols(),
            y_cast: P::Storage::empty_rows(),
            n,
            d,
            registry: PointRegistry::from_count(n),
            _recompute: PhantomData,
            _math: PhantomData,
            _cholesky: PhantomData,
        }
    }

    /// Drops the LDLT factor and returns a trainer with the current kernel,
    /// likelihood, transforms, optimizer, and policies.
    pub fn into_trainer(self) -> Gpr<O, S, C, B, M, P> {
        Gpr::from_owned(
            self.kernel,
            self.likelihood,
            self.x_unfitted,
            self.y_unfitted,
            self.optimizer,
            self.distance_cache,
            self.jitter_policy,
        )
    }

    /// Replaces the optimizer used by a later [`Self::refit`].
    ///
    /// Same Cholesky-pole rule as [`FittedGpr::with_optimizer`].
    pub fn with_optimizer<O2: PoleRecompute<B>>(
        self,
        optimizer: O2,
    ) -> OnlineGpr<O2, O2::Strategy, C, B, M, P> {
        OnlineGpr {
            kernel: self.kernel,
            compiled: self.compiled,
            likelihood: self.likelihood,
            x_unfitted: self.x_unfitted,
            y_unfitted: self.y_unfitted,
            x_transform: self.x_transform,
            y_transform: self.y_transform,
            optimizer,
            distance_cache: self.distance_cache,
            jitter_policy: self.jitter_policy,
            workspace: self.workspace,
            query: self.query,
            x_obs: self.x_obs,
            y_obs: self.y_obs,
            x: self.x,
            y_train: self.y_train,
            factor_alpha: self.factor_alpha,
            alpha: self.alpha,
            x_cast: self.x_cast,
            y_cast: self.y_cast,
            n: self.n,
            d: self.d,
            registry: self.registry,
            _recompute: PhantomData,
            _math: PhantomData,
            _cholesky: PhantomData,
        }
    }

    /// Returns the number of training points.
    pub fn n(&self) -> usize {
        self.n
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

    /// Returns predict `α` after the last insert, delete, or hyperparameter write.
    pub fn alpha(&self) -> &[P::Refine] {
        &self.alpha
    }

    fn refresh_factor_alpha(&mut self) {
        let n = self.n;
        let mut rhs = Mat::<P::Storage>::zeros(n, 1);
        for i in 0..n {
            rhs[(i, 0)] = P::Storage::from_f64(self.y_train[i]);
        }
        OnlineWorkspace::solve_ldlt_in_place(self.workspace.ld_factor.as_ref(), rhs.as_mut(), n);
        if self.factor_alpha.len() != n {
            self.factor_alpha.resize(n, P::Storage::from_f64(0.0));
        }
        for i in 0..n {
            let value = rhs[(i, 0)];
            self.factor_alpha[i] = value;
            self.workspace.alpha[i] = value;
        }
    }

    fn publish_predict_alpha(&mut self) -> Result<(), GprError> {
        let x = self.x.as_ref().submatrix(0, 0, self.n, self.d);
        P::publish_predict_alpha::<M>(
            &self.kernel,
            &self.compiled,
            x,
            &self.y_train,
            self.likelihood.noise_variance(),
            &self.factor_alpha,
            &mut self.alpha,
        )
    }

    /// Returns the original training features in column-major order.
    pub fn x(&self) -> &[f64] {
        &self.x_obs
    }

    /// Returns the original training targets.
    pub fn y(&self) -> &[f64] {
        &self.y_obs
    }

    /// Returns training-point identifiers in workspace buffer order.
    pub fn point_ids(&self) -> &[PointId] {
        self.registry.ids()
    }

    pub(crate) fn persist_point_ids(&self) -> Vec<u64> {
        self.registry.ids().iter().map(|id| id.raw()).collect()
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
        if registry.len() != self.n {
            return Err(persist_err(format!(
                "point_ids has {} values, expected n = {}",
                registry.len(),
                self.n
            )));
        }
        self.registry = registry;
        Ok(())
    }

    fn adopt_fitted(&mut self, fitted: FittedGpr<O, S, C, B, M, P>) -> Result<(), GprError> {
        let registry = self.registry.clone();
        *self = fitted.into_online()?;
        debug_assert_eq!(self.n, registry.len());
        self.registry = registry;
        Ok(())
    }

    pub(crate) fn jitter_policy(&self) -> JitterPolicy {
        self.jitter_policy
    }

    pub(crate) fn distance_cache_slot(&self) -> C {
        self.distance_cache
    }

    pub(crate) fn x_unfitted(&self) -> &dyn UnfittedTransform {
        self.x_unfitted.as_ref()
    }

    pub(crate) fn y_unfitted(&self) -> &dyn UnfittedTarget {
        self.y_unfitted.as_ref()
    }

    pub(crate) fn x_transform(&self) -> &dyn Transform {
        self.x_transform.as_ref()
    }

    pub(crate) fn y_transform(&self) -> &dyn TargetTransform {
        self.y_transform.as_ref()
    }

    pub(crate) fn ld_factor(&self) -> MatRef<'_, P::Storage> {
        self.workspace
            .ld_factor
            .as_ref()
            .submatrix(0, 0, self.n, self.n)
    }

    fn x_active(&self) -> MatRef<'_, f64> {
        self.x.as_ref().submatrix(0, 0, self.n, self.d)
    }

    /// Appends one training point at the current `θ` with a bordered LDLT update.
    ///
    /// `x_new` has length [`Self::d`]. Transforms already stored on this model
    /// are applied; they are not re-fit. Grows the online workspace when the
    /// next row does not fit. `α` is not solved here; the next
    /// [`Self::predict`], [`Self::neg_log_marginal_likelihood`], or
    /// [`Self::alpha`] fills it. The returned [`PointId`] is new and is never
    /// reused after a later [`Self::delete`].
    ///
    /// # Errors
    ///
    /// Returns [`GprError::DimensionMismatch`] if `x_new` is the wrong length,
    /// [`GprError::NonFiniteInput`] if a value is `NaN` or `Inf`,
    /// [`GprError::EmptyInput`] if the workspace cannot accept a row, or
    /// [`GprError::CholeskyFailed`] if the new pivot `δ` is not positive.
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
        #[cfg(feature = "insert-stages")]
        let kernel_start = Instant::now();
        let n = self.n;
        let d = self.d;
        self.query.ensure_at_least(n, 1, d)?;
        let xs_len = d;
        if self.query.query_xs.len() < xs_len {
            self.query.query_xs.resize(xs_len, 0.0);
        }
        self.query.query_xs[..xs_len].copy_from_slice(x_new);
        self.x_transform
            .apply(&mut self.query.query_xs[..xs_len], 1, d)?;
        let mut y_trans = [y_new];
        self.y_transform.transform(&mut y_trans)?;
        {
            let x_train =
                P::Storage::storage_cols(self.x.as_ref().submatrix(0, 0, n, d), &mut self.x_cast);
            let dest = self.workspace.v_buf.as_mat_mut().submatrix_mut(0, 0, n, 1);
            let QueryWorkspace {
                query_xs,
                query_x,
                query_dist,
                query_scratch,
                ..
            } = &mut self.query;
            pack_storage(
                &query_xs[..xs_len],
                1,
                d,
                query_x.as_mut().submatrix_mut(0, 0, 1, d),
            );
            self.compiled.eval_cross::<M>(
                x_train,
                query_x.as_ref().submatrix(0, 0, 1, d),
                Some(query_dist.as_mut().submatrix_mut(0, 0, n, 1)),
                dest,
                query_scratch.as_mut().submatrix_mut(0, 0, n, 1),
                &mut [],
            )?;
        }
        let mut kss = [P::Storage::from_f64(0.0)];
        self.compiled
            .eval_diag(self.query.query_x.as_ref().submatrix(0, 0, 1, d), &mut kss)?;
        let k_new = kss[0] + P::Storage::from_f64(self.likelihood.noise_variance());
        #[cfg(feature = "insert-stages")]
        insert_stages::add_kernel(kernel_start.elapsed().as_secs_f64());
        #[cfg(feature = "insert-stages")]
        let border_start = Instant::now();
        self.workspace.append_border(k_new)?;
        #[cfg(feature = "insert-stages")]
        insert_stages::add_border(border_start.elapsed().as_secs_f64());
        #[cfg(feature = "insert-stages")]
        let rest_start = Instant::now();
        append_colmajor(&mut self.x_obs, n, d, x_new);
        self.y_obs.push(y_new);
        append_point_mat_inplace(&mut self.x, n, &self.query.query_xs[..xs_len]);
        self.y_train.push(y_trans[0]);
        self.n += 1;
        OnlineWorkspace::set_f64_prefix(&mut self.workspace.y, &self.y_train);
        self.refresh_factor_alpha();
        self.publish_predict_alpha()?;
        let id = self.registry.insert();
        #[cfg(feature = "insert-stages")]
        insert_stages::add_rest(rest_start.elapsed().as_secs_f64());
        Ok(id)
    }

    /// Removes the training point identified by `id` and packs every buffer.
    ///
    /// Updates the stored LDLT with
    /// `ldlt::update::delete_rows_and_cols_clobber`. Workspace capacity is
    /// unchanged. `α` is not solved here; the next
    /// [`Self::predict`], [`Self::neg_log_marginal_likelihood`], or
    /// [`Self::alpha`] fills it. The last remaining point cannot be deleted.
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
        if self.n <= 1 {
            return Err(GprError::InsufficientData { n: self.n, min: 2 });
        }
        let index = self.registry.index_of(id)?;
        self.workspace.delete_index(index)?;
        remove_colmajor(&mut self.x_obs, self.n, self.d, index);
        self.y_obs.remove(index);
        remove_point_mat_inplace(&mut self.x, self.n, index);
        self.y_train.remove(index);
        self.registry.remove_at(index);
        self.n -= 1;
        OnlineWorkspace::set_f64_prefix(&mut self.workspace.y, &self.y_train);
        self.refresh_factor_alpha();
        self.publish_predict_alpha()?;
        Ok(())
    }

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
    pub fn save(&self, dir: impl AsRef<std::path::Path>) -> Result<(), GprError> {
        persist::save_online(self, dir.as_ref(), false)
    }

    /// Writes this model including the packed LDLT factor and `α`.
    ///
    /// # Errors
    ///
    /// Same as [`Self::save`].
    pub fn save_with_factor(&self, dir: impl AsRef<std::path::Path>) -> Result<(), GprError> {
        persist::save_online(self, dir.as_ref(), true)
    }

    /// Returns the negative log marginal likelihood from the stored LDLT factor.
    ///
    /// Uses `log|A| = Σ log(Dᵢ)` and the stored `α`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::CholeskyFailed`] if a stored `Dᵢ` is not positive.
    pub fn neg_log_marginal_likelihood(&self) -> Result<f64, GprError> {
        let mut y_cast = P::Storage::empty_rows();
        let y = P::Storage::storage_rows(&self.y_train, &mut y_cast);
        Ok(neg_mll_from_ldlt(
            self.workspace.ld_factor.as_ref(),
            y,
            &self.factor_alpha,
            self.n,
        ))
    }

    /// Returns the concatenated kernel and likelihood parameter count.
    pub fn num_params(&self) -> usize {
        self.kernel.num_params() + self.likelihood.num_params()
    }

    /// Writes kernel `θ` then likelihood `θ` into `out`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::LengthMismatch`] if `out` is the wrong length
    /// or a custom leaf rejects the write.
    pub fn get_params(&self, out: &mut [f64]) -> Result<(), GprError> {
        let n_kernel = self.kernel.num_params();
        crate::data::require_count(out.len(), self.num_params(), "parameters")?;
        self.kernel.get_params(&mut out[..n_kernel])?;
        self.likelihood.get_params(&mut out[n_kernel..])
    }

    /// Sets kernel then likelihood `θ` and rebuilds the LDLT factor.
    ///
    /// Snapshots through [`FittedGpr::set_params`] at the current observations
    /// and converts back. Transforms are not re-fit.
    ///
    /// # Errors
    ///
    /// Same as [`FittedGpr::set_params`], plus conversion failures from
    /// [`FittedGpr::into_online`].
    pub fn set_params(&mut self, params: &[f64]) -> Result<(), GprError>
    where
        O: Clone,
        C: Copy,
    {
        let mut fitted = FittedGpr::from_online_snapshot(self)?;
        fitted.set_params(params)?;
        self.adopt_fitted(fitted)
    }

    /// Writes the joint NLML and gradient at `params`.
    ///
    /// # Errors
    ///
    /// Same as [`FittedGpr::value_and_gradient_into`].
    pub fn value_and_gradient_into(
        &mut self,
        params: &[f64],
        out: &mut [f64],
    ) -> Result<f64, GprError>
    where
        O: Clone,
        C: Copy,
    {
        let mut fitted = FittedGpr::from_online_snapshot(self)?;
        let nlml = fitted.value_and_gradient_into(params, out)?;
        self.adopt_fitted(fitted)?;
        Ok(nlml)
    }

    /// Writes the analytic NLML Hessian (row-major `p×p`) at `params`.
    ///
    /// # Errors
    ///
    /// Same as [`FittedGpr::hessian_into`].
    pub fn hessian_into(&mut self, params: &[f64], out: &mut [f64]) -> Result<(), GprError>
    where
        O: Clone,
        C: Copy,
    {
        let mut fitted = FittedGpr::from_online_snapshot(self)?;
        fitted.hessian_into(params, out)?;
        self.adopt_fitted(fitted)
    }

    /// Predicts at `xs` with [`PredictOptions::default`] (observation variance).
    ///
    /// # Errors
    ///
    /// Same as [`FittedGpr::predict`].
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
    pub fn predict_with(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        options: PredictOptions,
    ) -> Result<Prediction<P::Refine>, GprError> {
        let mut out = Prediction::default();
        self.write_prediction(xs, n_rows, n_cols, options, &mut out)?;
        Ok(out)
    }

    /// Writes [`Self::predict_with`] into `out`.
    ///
    /// # Errors
    ///
    /// Same as [`Self::predict`].
    pub fn predict_with_into(
        &mut self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        options: PredictOptions,
        out: &mut Prediction<P::Refine>,
    ) -> Result<(), GprError> {
        if n_cols != self.d {
            return Err(GprError::DimensionMismatch {
                x_dim: n_cols,
                expected_dim: self.d,
            });
        }
        validate_query(xs, n_rows, n_cols)?;
        self.publish_predict_alpha()?;
        let n = self.n;
        let m = n_rows;
        self.query.ensure(n, m, n_cols)?;
        self.query.query_xs.copy_from_slice(xs);
        self.x_transform
            .apply(&mut self.query.query_xs, n_rows, n_cols)?;
        {
            let x_train = P::Storage::storage_cols(
                self.x.as_ref().submatrix(0, 0, n, n_cols),
                &mut self.x_cast,
            );
            let QueryWorkspace {
                query_xs,
                query_x,
                query_dist,
                query_k_star,
                query_scratch,
                ..
            } = &mut self.query;
            pack_storage(query_xs, n_rows, n_cols, query_x.as_mut());
            self.compiled.eval_cross::<M>(
                x_train,
                query_x.as_ref(),
                Some(query_dist.as_mut()),
                query_k_star.as_mut(),
                query_scratch.as_mut(),
                &mut [],
            )?;
        }
        let ld = self
            .workspace
            .ld_factor
            .as_ref()
            .submatrix(0, 0, self.n, self.n);
        write_ldlt_prediction::<M, P>(
            &self.kernel,
            ld,
            &self.alpha,
            &self.compiled,
            self.x.as_ref().submatrix(0, 0, self.n, self.d),
            &self.query.query_xs,
            self.query.query_x.as_ref(),
            self.query.query_k_star.as_mut(),
            &mut self.query.query_kss,
            n,
            m,
            n_cols,
            self.likelihood.noise_variance(),
            options,
            self.y_transform.as_ref(),
            out,
        )
    }

    fn write_prediction(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        options: PredictOptions,
        out: &mut Prediction<P::Refine>,
    ) -> Result<(), GprError> {
        if n_cols != self.d {
            return Err(GprError::DimensionMismatch {
                x_dim: n_cols,
                expected_dim: self.d,
            });
        }
        validate_query(xs, n_rows, n_cols)?;
        let n = self.n;
        let m = n_rows;
        let mut alpha = Vec::new();
        P::publish_predict_alpha::<M>(
            &self.kernel,
            &self.compiled,
            self.x_active(),
            &self.y_train,
            self.likelihood.noise_variance(),
            &self.factor_alpha,
            &mut alpha,
        )?;
        let mut query_xs = xs.to_vec();
        self.x_transform.apply(&mut query_xs, n_rows, n_cols)?;
        let mut x_cast = P::Storage::empty_cols();
        let x_train = P::Storage::storage_cols(self.x_active(), &mut x_cast);
        let mut query_x = Mat::<P::Storage>::zeros(m, n_cols);
        pack_storage(&query_xs, n_rows, n_cols, query_x.as_mut());
        let mut query_dist = Mat::<P::Storage>::zeros(n, m);
        let mut query_k_star = Mat::<P::Storage>::zeros(n, m);
        let mut query_scratch = Mat::<P::Storage>::zeros(n, m);
        let mut query_kss = vec![P::Storage::from_f64(0.0); m];
        self.compiled.eval_cross::<M>(
            x_train,
            query_x.as_ref(),
            Some(query_dist.as_mut()),
            query_k_star.as_mut(),
            query_scratch.as_mut(),
            &mut [],
        )?;
        write_ldlt_prediction::<M, P>(
            &self.kernel,
            self.ld_factor(),
            &alpha,
            &self.compiled,
            self.x_active(),
            &query_xs,
            query_x.as_ref(),
            query_k_star.as_mut(),
            &mut query_kss,
            n,
            m,
            n_cols,
            self.likelihood.noise_variance(),
            options,
            self.y_transform.as_ref(),
            out,
        )
    }

    /// Returns the predictive mean and query–query covariance at `xs`.
    ///
    /// # Errors
    ///
    /// Same as [`FittedGpr::predict_covariance`].
    pub fn predict_covariance(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
    ) -> Result<PredictiveCovariance<P::Refine>, GprError>
    where
        O: Clone,
        C: Copy,
    {
        self.to_fitted()?.predict_covariance(xs, n_rows, n_cols)
    }

    /// Returns query–query covariance with an explicit variance kind.
    ///
    /// # Errors
    ///
    /// Same as [`Self::predict_covariance`].
    pub fn predict_covariance_with(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        options: PredictOptions,
    ) -> Result<PredictiveCovariance<P::Refine>, GprError>
    where
        O: Clone,
        C: Copy,
    {
        self.to_fitted()?
            .predict_covariance_with(xs, n_rows, n_cols, options)
    }

    /// Draws posterior samples at `xs`.
    ///
    /// # Errors
    ///
    /// Same as [`FittedGpr::sample`].
    pub fn sample(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        n_draws: usize,
        seed: u64,
    ) -> Result<Vec<P::Refine>, GprError>
    where
        O: Clone,
        C: Copy,
    {
        self.to_fitted()?.sample(xs, n_rows, n_cols, n_draws, seed)
    }

    /// Draws posterior samples with an explicit variance kind.
    ///
    /// # Errors
    ///
    /// Same as [`Self::sample`].
    pub fn sample_with(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        options: PredictOptions,
        n_draws: usize,
        seed: u64,
    ) -> Result<Vec<P::Refine>, GprError>
    where
        O: Clone,
        C: Copy,
    {
        self.to_fitted()?
            .sample_with(xs, n_rows, n_cols, options, n_draws, seed)
    }

    /// Leave-one-out predictive mean and variance on the training set.
    ///
    /// # Errors
    ///
    /// Same as [`FittedGpr::loo_predict`].
    pub fn loo_predict(&self) -> Result<Prediction<P::Refine>, GprError>
    where
        O: Clone,
        C: Copy,
    {
        self.to_fitted()?.loo_predict()
    }

    /// Leave-one-out prediction with an explicit variance kind.
    ///
    /// # Errors
    ///
    /// Same as [`Self::loo_predict`].
    pub fn loo_predict_with(
        &self,
        options: PredictOptions,
    ) -> Result<Prediction<P::Refine>, GprError>
    where
        O: Clone,
        C: Copy,
    {
        self.to_fitted()?.loo_predict_with(options)
    }

    fn to_fitted(&self) -> Result<FittedGpr<O, S, C, B, M, P>, GprError>
    where
        O: Clone,
        C: Copy,
    {
        FittedGpr::from_online_snapshot(self)
    }
}

#[allow(private_bounds)]
impl<O, S, C, B, M, P> OnlineGpr<O, S, C, B, M, P>
where
    C: DistanceCacheSlot,
    B: AllocWorkspace,
    P: GpScalar,
    M: crate::math::KernelMath,
    O: Clone + for<'a> Optimizer<GprObjective<'a, O, S, C, B, M, P>>,
{
    /// Re-runs the stored optimizer on the stored training data.
    ///
    /// # Errors
    ///
    /// Same as [`FittedGpr::refit`].
    pub fn refit(&mut self) -> Result<(), GprError>
    where
        C: Copy,
    {
        let mut fitted = FittedGpr::from_online_snapshot(self)?;
        fitted.refit()?;
        self.adopt_fitted(fitted)
    }
}

#[allow(private_bounds)]
impl<C, M, P> OnlineGpr<Fixed, FullRecompute, C, RetainCholesky, M, P>
where
    C: DistanceCacheSlot,
    P: GpScalar,
    M: crate::math::KernelMath,
{
    /// Rebuilds the LDLT factor at the current `θ` without a search.
    ///
    /// # Errors
    ///
    /// Same as [`Gpr<Fixed>::factor`].
    pub fn refit(&mut self) -> Result<(), GprError> {
        let mut fitted = FittedGpr::from_online_snapshot(self)?;
        fitted.refit()?;
        self.adopt_fitted(fitted)
    }
}

#[allow(private_bounds)]
impl<C, M, P> OnlineGpr<Fixed, FullRecompute, C, RetainCholesky, M, P>
where
    C: DistanceCacheSlot,
    P: crate::precision::GpScalar,
    M: crate::math::KernelMath,
{
    pub(crate) fn from_persisted(parts: PersistedModel<C, P>) -> Result<Self, GprError> {
        FittedGpr::from_persisted(parts)?.into_online_preserving_factor()
    }
}

#[allow(clippy::too_many_arguments)]
fn write_ldlt_prediction<M: crate::math::KernelMath, P>(
    kernel: &KernelSpec,
    ld: MatRef<'_, P::Storage>,
    alpha: &[P::Refine],
    compiled: &CompiledKernel<P::Storage>,
    x_train: MatRef<'_, f64>,
    x_query: &[f64],
    query_x: MatRef<'_, P::Storage>,
    mut query_k_star: MatMut<'_, P::Storage>,
    query_kss: &mut [P::Storage],
    n: usize,
    m: usize,
    n_cols: usize,
    noise: f64,
    options: PredictOptions,
    y_transform: &dyn TargetTransform,
    out: &mut Prediction<P::Refine>,
) -> Result<(), GprError>
where
    P: GpScalar,
{
    let zero = P::Refine::from_f64(0.0);
    if out.mean.len() != m {
        out.mean.resize(m, zero);
    }
    if out.variance.len() != m {
        out.variance.resize(m, zero);
    }
    for (col, mean) in out.mean.iter_mut().enumerate() {
        *mean = P::column_mean::<M>(
            kernel,
            query_k_star.as_ref(),
            x_train,
            x_query,
            n_cols,
            alpha,
            col,
        )?;
    }
    OnlineWorkspace::apply_inv_l(ld, query_k_star.as_mut(), n);
    compiled.eval_diag(query_x, query_kss)?;
    let noise_s = P::Storage::from_f64(noise);
    let zero_s = P::Storage::from_f64(0.0);
    for col in 0..m {
        let mut quad = 0.0f64;
        for row in 0..n {
            let w = query_k_star[(row, col)].to_f64();
            quad += w * w / ld[(row, row)].to_f64();
        }
        let mut latent = query_kss[col] - P::Storage::from_f64(quad);
        if latent.to_f64() < 0.0 {
            latent = zero_s;
        }
        let stored = match options.variance_kind {
            crate::VarianceKind::Latent => latent,
            crate::VarianceKind::Observation => latent + noise_s,
        };
        out.variance[col] = P::Refine::from_f64(stored.to_f64());
    }
    P::inverse_mean_variance(y_transform, &mut out.mean, &mut out.variance)?;
    out.variance_kind = options.variance_kind;
    Ok(())
}

fn neg_mll_from_ldlt<T: KernelScalar>(ld: MatRef<'_, T>, y: &[T], alpha: &[T], n: usize) -> f64 {
    let mut quad = T::from_f64(0.0);
    let mut log_det = T::from_f64(0.0);
    for i in 0..n {
        quad += y[i] * alpha[i];
        log_det += ld[(i, i)].ln();
    }
    let log_two_pi = (2.0 * std::f64::consts::PI).ln();
    0.5 * (quad.to_f64() + log_det.to_f64() + n as f64 * log_two_pi)
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
        let LoadedGpr::OnlineDistance(crate::persist::LoadedOnlineDistance::Cached(model)) = loaded
        else {
            panic!("online RBF should load as OnlineDistance::Cached");
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
        let LoadedGpr::OnlineDistance(crate::persist::LoadedOnlineDistance::Cached(model)) = loaded
        else {
            panic!("online RBF should load as OnlineDistance::Cached");
        };
        assert_eq!(model.point_ids(), want_ids.as_slice());
        let got = model.predict(&[0.5], 1, 1).expect("loaded predict");
        assert_mean_var_close(&got.mean, &got.variance, &want.mean, &want.variance, TOL);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
