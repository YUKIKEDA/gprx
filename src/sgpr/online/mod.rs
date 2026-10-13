//! Online collapsed variational SGPR.

use std::marker::PhantomData;

use faer::Mat;

use crate::data::pack_points;
use crate::error::GprError;
use crate::error::PersistErrorKind;
use crate::kernel::ScalarOps;
use crate::kernel::{
    CrossViews, DistanceKernel, DistanceOnly, DistanceSlot, DistanceSource, KernelScalar,
    KernelSpec, ModelKernel, NoSupply, PointUse, QueryScratch, QuerySources, SuppliedSpec, Supply,
    SupplyViews, WithPoints, column_into, new_inducing_column,
};
use crate::linalg::{chol_rank1_downdate, chol_rank1_update, frobenius2, solve_llt};
use crate::optimizer::{Lbfgs, Optimizer};
use crate::points::PointId;
use crate::points::{IdRegistry, PointRegistry, RegistryId};
use crate::policy::JitterPolicy;
use crate::policy::with_kernel_exp;
use crate::precision::{DoublePrecision, ModelPrecision};
use crate::prediction::{DistanceQuery, QueryPoints, distance_predict};
use crate::sgpr::SgprObjective;
use crate::sparse::{
    KernelScratch, PredictScratch, SparseCore, SparseScratch, SparseSupply, sparse_core_accessors,
    sparse_kernel_accessor, sparse_point_accessors,
};
use crate::{PredictOptions, Prediction, PredictiveCovariance};

use super::FixedInducing;
use super::factor::{
    InducingBorder, VfeState, VfeSystem, a_times_y, append_point, assemble_vfe,
    assemble_vfe_with_f64_w, coordinate_border, inducing_append, inducing_delete, kernel_column,
    kernel_diag_at, point_at, predict_vfe_covariance, predict_vfe_into, publish_sgpr_weights,
    push_column, remove_column_in_place, remove_point, solve_lmm, vfe_loo,
    vfe_neg_log_marginal_likelihood,
};
use super::fitted::FittedSgpr;

mod distance;

/// Represents the stable identity of one inducing point on [`OnlineSgpr`].
///
/// [`crate::FittedSgpr::into_online`] assigns identifiers `0.. m-1` in buffer order. Later
/// `insert_inducing` calls on [`OnlineSgpr`] return values that increase monotonically and are never reused after
/// [`OnlineSgpr::delete_inducing`]. There is no public constructor.
///
/// # Examples
///
/// ```rust
/// use gprx::kernel::{KernelSpec, RbfKernel};
/// use gprx::{Fixed, GaussianLikelihood, Sgpr};
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let fitted = Sgpr::new(
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
/// let id = online.insert_inducing(&[1.5])?;
/// assert_eq!(online.inducing_ids().last().copied(), Some(id));
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct InducingId(u64);

impl RegistryId for InducingId {
    const PERSIST_KEY: &'static str = "inducing_ids";

    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    fn raw(self) -> u64 {
        self.0
    }

    fn unknown() -> GprError {
        GprError::InvalidInducingId
    }
}

pub(crate) type InducingRegistry = IdRegistry<InducingId>;

/// One change of the inducing set of an [`OnlineSgpr`], applied by the
/// incremental update of ADR 0005.
pub(super) enum InducingChange<T> {
    /// Appends the inducing point of these kernel values; `replace` is
    /// that of [`inducing_append`].
    Insert {
        border: InducingBorder<T>,
        replace: Option<usize>,
    },
    /// Drops the inducing point at this index.
    Delete(usize),
}

/// Represents the online collapsed variational SGPR after [`FittedSgpr::into_online`].
///
/// The type parameters are the optimizer `O` that [`Self::refit`] runs, the
/// precision `P`, and the kernel `K`. A [`KernelSpec`] model inserts points
/// and inducing points from coordinates; a
/// [`crate::kernel::DistanceKernel`] model inserts them from supplied
/// squared distances, and its inducing points are training points, named
/// by their [`PointId`]s.
///
/// `insert` appends one training point and returns a [`PointId`].
/// [`Self::delete`] removes one point by that identifier. VFE factors for
/// `X` update with a rank-1 cholupdate of `B` while `K_mm` stays put.
/// `insert_inducing` appends one inducing point and returns an
/// [`InducingId`]. [`Self::delete_inducing`] removes one inducing point
/// by that identifier. Inducing points change the factors in place (ADR
/// 0005): an insert borders the factors of `K_mm` and `B` and adds a row
/// to `A` (`O(n·m)` plus the `O(m²)` border), and a delete drops a row of
/// the factor of `K_mm` by a trailing rank-1 update and solves `A` and `B`
/// again from the kept `K(Z, X) = L A` (`O(n·m²)`, no kernel evaluation).
/// Only when that does not factor (a bordered pivot that is not positive,
/// such as a new inducing point on top of one the model holds, or a `B`
/// that does not factor) is the VFE system assembled again on the changed
/// inducing set. A precision that refines in `f64` assembles its `f64`
/// predict weights again after every change, as for `X`.
/// Neither moves `X` or `θ`.
/// Parameters are kernel `θ` then likelihood `θ`. [`Self::set_params`] and
/// [`Self::refit`] rebuild the VFE system.
///
/// # Examples
///
/// ```rust
/// use gprx::kernel::{KernelSpec, RbfKernel};
/// use gprx::{
///     Fixed, GaussianLikelihood, JitterPolicy, KernelExp, PredictOptions, Prediction, Sgpr,
///     VarianceKind,
/// };
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let fitted = Sgpr::new(
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
/// let id = online.insert(&[4.0], 0.1)?;
/// let pred = online.predict(&[0.5], 1, 1)?;
/// assert_eq!(pred.mean.len(), 1);
///
/// // The training data, the identifiers, and the settings.
/// assert_eq!((online.n(), online.m(), online.d()), (5, 2, 1));
/// assert_eq!(online.x(), &[0.0, 1.0, 2.0, 3.0, 4.0]);
/// assert_eq!(online.y(), &[0.0, 1.0, 0.5, 0.25, 0.1]);
/// assert_eq!(online.z(), &[0.5, 2.5]);
/// assert_eq!(online.point_ids().last().copied(), Some(id));
/// assert_eq!(online.inducing_ids().len(), 2);
/// assert_eq!(online.math(), KernelExp::Accurate);
/// assert!(matches!(online.jitter_policy(), JitterPolicy::Adaptive(_)));
/// assert!(online.neg_log_marginal_likelihood()?.is_finite());
///
/// // θ: read, write, and differentiate at the current points.
/// let mut theta = vec![0.0; online.num_params()];
/// online.get_params(&mut theta)?;
/// assert_eq!(theta.len(), online.kernel().num_params() + 1);
/// assert!((theta[1] - online.likelihood().log_noise_variance()).abs() < 1e-12);
/// theta[0] = 0.8_f64.ln();
/// online.set_params(&theta)?;
/// let mut grad = vec![0.0; theta.len()];
/// assert!(online.value_and_gradient_into(&theta, &mut grad)?.is_finite());
/// let mut hess = vec![0.0; theta.len() * theta.len()];
/// online.hessian_into(&theta, &mut hess)?;
///
/// // Every query form, with the default or latent variance.
/// let latent = PredictOptions { variance_kind: VarianceKind::Latent };
/// let xs = [0.25, 0.75];
/// let mut out = Prediction::default();
/// online.predict_into(&xs, 2, 1, &mut out)?;
/// assert_eq!(out, online.predict(&xs, 2, 1)?);
/// online.predict_with_into(&xs, 2, 1, latent, &mut out)?;
/// assert_eq!(out, online.predict_with(&xs, 2, 1, latent)?);
/// assert_eq!(online.predict_covariance_with(&xs, 2, 1, latent)?.covariance.len(), 4);
/// assert_eq!(online.sample(&xs, 2, 1, 3, 7)?.len(), 6);
/// assert_eq!(online.sample_with(&xs, 2, 1, latent, 3, 7)?.len(), 6);
/// assert_eq!(online.loo_predict_with(latent)?.mean.len(), 5);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct OnlineSgpr<O = Lbfgs, P: ModelPrecision = DoublePrecision, K: ModelKernel = KernelSpec> {
    /// Everything an online update writes; [`OnlineSgpr::atomically`]
    /// copies and restores it as one value.
    pub(super) state: OnlineState<P, K::Supply>,
    /// Kernel scratch kept between `&mut self` calls.
    pub(super) scratch: SparseScratch<P::Storage, K::Supply>,
    optimizer: O,
    _kernel: PhantomData<K>,
}

/// The fields of an [`OnlineSgpr`] an online update writes. A field added
/// here is undone by [`OnlineSgpr::atomically`] with the rest.
#[derive(Clone, Debug)]
pub(super) struct OnlineState<P: ModelPrecision, U: Supply = NoSupply> {
    pub(super) core: SparseCore<U>,
    /// Lower `L` from `K_mm + j I = L Lᵀ`.
    k_mm_l: Mat<P::Storage>,
    /// The offset `j` of `k_mm_l`: the one the last assembly of `K_mm`
    /// took under the jitter policy, which an inducing insert borders with.
    k_mm_jitter: f64,
    a: Mat<P::Storage>,
    b_l: Mat<P::Storage>,
    w: Vec<P::Storage>,
    predict_w: Vec<P::Refine>,
    k_diag_sum: P::Storage,
    a_frobenius2: P::Storage,
    /// `A y` in `f64`, kept through rank-1 updates so `w = B⁻¹ A y` costs
    /// one `O(m²)` solve instead of a pass over all `n` columns.
    ay: Vec<f64>,
    registry: PointRegistry,
    inducing: InducingRegistry,
    /// The [`PointId`] of each inducing point of a model on supplied
    /// distances (in inducing order): ids do not move when a delete shifts
    /// the rows of the training blocks. A coordinate model has none.
    pub(super) inducing_points: <U as SupplyViews>::Held<Vec<PointId>>,
}

/// The [`PointId`]s of the inducing points of `core` (a model on supplied
/// distances) among the training points `registry` names.
///
/// # Errors
///
/// Returns [`GprError::PersistFailed`] if an inducing index is not a row
/// of `registry`.
fn inducing_points_of<S: Supply>(
    core: &SparseCore<S>,
    registry: &PointRegistry,
) -> Result<<S as SupplyViews>::Held<Vec<PointId>>, GprError> {
    S::try_map_held(core.supply_held(), |supply| {
        supply
            .inducing
            .iter()
            .map(|&row| {
                registry.ids().get(row).copied().ok_or_else(|| {
                    crate::persist::persist_err(
                        PersistErrorKind::Config,
                        format!(
                            "inducing index {row} is not one of the {} points",
                            registry.len()
                        ),
                    )
                })
            })
            .collect()
    })
}

impl<O, P, K: ModelKernel> OnlineSgpr<O, P, K>
where
    P: crate::precision::GpScalar,
{
    pub(crate) fn core(&self) -> &SparseCore<K::Supply> {
        &self.state.core
    }

    pub(crate) fn point_registry(&self) -> &PointRegistry {
        &self.state.registry
    }

    pub(crate) fn inducing_registry(&self) -> &InducingRegistry {
        &self.state.inducing
    }

    /// The online model of a persist directory: `fitted` with the saved
    /// point and inducing identifiers.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::PersistFailed`] when an identifier list is
    /// invalid or its length is not `n` / `m`.
    pub(crate) fn from_persisted<I>(
        fitted: FittedSgpr<O, I, P, K>,
        points: PointRegistry,
        inducing: InducingRegistry,
    ) -> Result<Self, GprError> {
        let mut online = Self::from_fitted(fitted);
        if points.len() != online.state.core.n || inducing.len() != online.state.core.m {
            return Err(crate::persist::persist_err(
                PersistErrorKind::Config,
                format!(
                    "config has {} point ids and {} inducing ids, expected n = {} and m = {}",
                    points.len(),
                    inducing.len(),
                    online.state.core.n,
                    online.state.core.m
                ),
            ));
        }
        online.state.inducing_points =
            inducing_points_of::<K::Supply>(&online.state.core, &points)?;
        online.state.registry = points;
        online.state.inducing = inducing;
        Ok(online)
    }

    pub(crate) fn from_fitted<I>(fitted: FittedSgpr<O, I, P, K>) -> Self {
        let registry = PointRegistry::from_count(fitted.core.n);
        let inducing = InducingRegistry::from_count(fitted.core.m);
        let ay = a_times_y(fitted.a.as_ref(), &fitted.core.y_train);
        // A fresh registry names row `i` as id `i`, so each inducing row's
        // id follows from the row itself: one id per row, none dropped.
        let inducing_points =
            <K::Supply as SupplyViews>::map_held(fitted.core.supply_held(), |supply| {
                supply
                    .inducing
                    .iter()
                    .map(|&row| <PointId as RegistryId>::from_raw(row as u64))
                    .collect()
            });
        Self {
            state: OnlineState {
                ay,
                core: fitted.core,
                k_mm_l: fitted.k_mm_l,
                k_mm_jitter: fitted.k_mm_jitter,
                a: fitted.a,
                b_l: fitted.b_l,
                w: fitted.w,
                predict_w: fitted.predict_w,
                k_diag_sum: fitted.k_diag_sum,
                a_frobenius2: fitted.a_frobenius2,
                registry,
                inducing,
                inducing_points,
            },
            scratch: fitted.scratch,
            optimizer: fitted.optimizer,
            _kernel: PhantomData,
        }
    }

    fn snapshot_fitted(&mut self) -> FittedSgpr<O, FixedInducing, P, K>
    where
        O: Clone,
    {
        // The training blocks are lent, not copied; `with_snapshot` takes
        // them back whatever the call returns.
        let supplied = <K::Supply>::take_held(&mut self.state.core.supplied);
        let mut core = self.state.core.clone();
        core.supplied = supplied;
        FittedSgpr {
            core,
            // Lent for the call; `adopt_fitted` takes it back.
            scratch: std::mem::take(&mut self.scratch),
            optimizer: self.optimizer.clone(),
            inducing: PhantomData,
            _kernel: PhantomData,
            k_mm_l: self.state.k_mm_l.clone(),
            k_mm_jitter: self.state.k_mm_jitter,
            a: self.state.a.clone(),
            b_l: self.state.b_l.clone(),
            w: self.state.w.clone(),
            predict_w: self.state.predict_w.clone(),
            k_diag_sum: self.state.k_diag_sum,
            a_frobenius2: self.state.a_frobenius2,
        }
    }

    /// Runs `call` on a batch model of this one: on success this model
    /// takes its state, on an error only the lent training blocks and
    /// scratch come back.
    fn with_snapshot<R>(
        &mut self,
        call: impl FnOnce(&mut FittedSgpr<O, FixedInducing, P, K>) -> Result<R, GprError>,
    ) -> Result<R, GprError>
    where
        O: Clone,
    {
        let mut fitted = self.snapshot_fitted();
        match call(&mut fitted) {
            Ok(value) => {
                self.adopt_fitted(fitted);
                Ok(value)
            }
            Err(err) => {
                self.state.core.supplied = fitted.core.supplied;
                self.scratch = fitted.scratch;
                Err(err)
            }
        }
    }

    fn adopt_fitted(&mut self, fitted: FittedSgpr<O, FixedInducing, P, K>) {
        self.state.core = fitted.core;
        self.scratch = fitted.scratch;
        self.optimizer = fitted.optimizer;
        self.state.k_mm_l = fitted.k_mm_l;
        self.state.k_mm_jitter = fitted.k_mm_jitter;
        self.state.a = fitted.a;
        self.state.b_l = fitted.b_l;
        self.state.w = fitted.w;
        self.state.predict_w = fitted.predict_w;
        self.state.k_diag_sum = fitted.k_diag_sum;
        self.state.a_frobenius2 = fitted.a_frobenius2;
        self.state.ay = a_times_y(self.state.a.as_ref(), &self.state.core.y_train);
    }

    /// Runs one online update so that it either completes or changes nothing.
    ///
    /// Every update runs its failing steps (kernel evaluation, transforms,
    /// re-assembly, Cholesky) before its first write, and the writes after
    /// that cannot fail, except one: a precision that refines in `f64`
    /// ([`ModelPrecision::REFINES_IN_F64`]) assembles its predict weights
    /// again from the updated data, and that can fail after the buffers
    /// changed. For such a precision this copies [`OnlineState`], every
    /// field an update writes, and puts it back on failure. Other precisions
    /// skip the copy, so their updates allocate nothing for it; they rely on
    /// every failing step coming before the first write.
    pub(super) fn atomically<R>(
        &mut self,
        update: impl FnOnce(&mut Self) -> Result<R, GprError>,
    ) -> Result<R, GprError> {
        if !P::REFINES_IN_F64 {
            return update(self);
        }
        // The training blocks are not copied: an update that changes them
        // undoes its own change when it fails.
        let supplied = <K::Supply>::take_held(&mut self.state.core.supplied);
        let undo = self.state.clone();
        self.state.core.supplied = supplied;
        let result = update(self);
        if result.is_err() {
            let supplied = <K::Supply>::take_held(&mut self.state.core.supplied);
            self.state = undo;
            self.state.core.supplied = supplied;
        }
        result
    }

    fn vfe_state(&self) -> VfeState<P::Storage> {
        VfeState::<P::Storage> {
            k_mm_l: self.state.k_mm_l.clone(),
            k_mm_jitter: self.state.k_mm_jitter,
            a: self.state.a.clone(),
            b_l: self.state.b_l.clone(),
            w: self.state.w.clone(),
            k_diag_sum: self.state.k_diag_sum,
            a_frobenius2: self.state.a_frobenius2,
        }
    }

    fn apply_vfe(&mut self, state: VfeState<P::Storage>) -> Result<(), GprError> {
        self.set_vfe(state);
        self.refresh_predict_w()
    }

    /// Takes `state` as the VFE system. The predict weights are the caller's.
    fn set_vfe(&mut self, state: VfeState<P::Storage>) {
        self.state.k_mm_l = state.k_mm_l;
        self.state.k_mm_jitter = state.k_mm_jitter;
        self.state.a = state.a;
        self.state.b_l = state.b_l;
        self.state.w = state.w;
        self.state.k_diag_sum = state.k_diag_sum;
        self.state.a_frobenius2 = state.a_frobenius2;
        self.state.ay = a_times_y(self.state.a.as_ref(), &self.state.core.y_train);
    }

    fn refresh_predict_w(&mut self) -> Result<(), GprError> {
        if P::REFINES_IN_F64 {
            self.state.predict_w =
                with_kernel_exp!(self.state.core.math, M => assemble_vfe::<M, f64, K::Supply>(
                    &self.state.core.kernel,
                    self.state.core.jitter,
                    self.state.core.likelihood,
                    self.state.core.data(),
                    &mut self.scratch.f64,
                    &mut KernelScratch::new(),
                ))?
                .w
                .into_iter()
                .map(P::Refine::from_f64)
                .collect();
            return Ok(());
        }
        self.state.predict_w = with_kernel_exp!(self.state.core.math, M => publish_sgpr_weights::<M, P, K::Supply>(
            &self.state.core.kernel,
            self.state.core.jitter,
            self.state.a.as_ref(),
            self.state.b_l.as_ref(),
            &self.state.w,
            self.state.core.data(),
            self.state.core.likelihood.noise_variance(),
        ))?;
        Ok(())
    }

    /// `w = B⁻¹ A y` from the kept `A y`: one `O(m²)` solve.
    fn recompute_w(&mut self) -> Result<(), GprError> {
        let m = self.state.core.m;
        let mut rhs = Mat::<P::Storage>::zeros(m, 1);
        for (row, value) in self.state.ay.iter().enumerate() {
            rhs[(row, 0)] = P::Storage::from_f64(*value);
        }
        solve_llt(self.state.b_l.as_ref(), rhs.as_mut());
        self.state.w.clear();
        self.state.w.extend((0..m).map(|row| rhs[(row, 0)]));
        self.refresh_predict_w()
    }

    sparse_core_accessors!(state.core);

    /// Returns training-point identifiers in buffer order.
    ///
    /// See the example on [`OnlineSgpr`].
    pub fn point_ids(&self) -> &[PointId] {
        self.state.registry.ids()
    }

    /// Returns inducing-point identifiers in buffer order.
    ///
    /// See the example on [`OnlineSgpr`].
    pub fn inducing_ids(&self) -> &[InducingId] {
        self.state.inducing.ids()
    }

    /// Returns the concatenated kernel and likelihood parameter count.
    ///
    /// Inducing coordinates are not counted.
    ///
    /// See the example on [`OnlineSgpr`].
    pub fn num_params(&self) -> usize {
        self.state.core.theta_len()
    }

    /// Writes kernel `θ` then likelihood `θ` into `out`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::LengthMismatch`] if `out` is the wrong length
    /// or a custom leaf rejects the write.
    ///
    /// See the example on [`OnlineSgpr`].
    pub fn get_params(&self, out: &mut [f64]) -> Result<(), GprError> {
        self.state.core.read_theta(out)
    }

    /// Sets kernel then likelihood `θ` and rebuilds the VFE factors.
    ///
    /// `params` matches [`Self::get_params`]. Training `X` / `y` and
    /// inducing `Z` are not changed. Values are committed together only
    /// after the VFE system factors.
    ///
    /// # Errors
    ///
    /// Same as [`FittedSgpr::set_params`].
    ///
    /// See the example on [`OnlineSgpr`].
    pub fn set_params(&mut self, params: &[f64]) -> Result<(), GprError>
    where
        O: Clone,
    {
        self.with_snapshot(|fitted| fitted.set_params(params))
    }

    /// Sets parameters, rebuilds the VFE system, and writes `∂L/∂θ` of the negative ELBO.
    ///
    /// `params` and `out` match [`Self::get_params`].
    ///
    /// # Errors
    ///
    /// Same as [`FittedSgpr::value_and_gradient_into`].
    ///
    /// See the example on [`OnlineSgpr`].
    pub fn value_and_gradient_into(
        &mut self,
        params: &[f64],
        out: &mut [f64],
    ) -> Result<f64, GprError>
    where
        O: Clone,
    {
        self.with_snapshot(|fitted| fitted.value_and_gradient_into(params, out))
    }

    /// Writes the Hessian of the negative ELBO (row-major `p×p`) into `out`.
    ///
    /// # Errors
    ///
    /// Same as [`FittedSgpr::hessian_into`].
    ///
    /// See the example on [`OnlineSgpr`].
    pub fn hessian_into(&mut self, params: &[f64], out: &mut [f64]) -> Result<(), GprError>
    where
        O: Clone,
    {
        self.with_snapshot(|fitted| fitted.hessian_into(params, out))
    }

    /// Returns the negative VFE evidence lower bound (the sparse NLML).
    ///
    /// # Errors
    ///
    /// The stored factors are already valid, so this returns `Ok`.
    ///
    /// See the example on [`OnlineSgpr`].
    pub fn neg_log_marginal_likelihood(&self) -> Result<f64, GprError> {
        vfe_neg_log_marginal_likelihood(
            self.state.a.as_ref(),
            self.state.b_l.as_ref(),
            &self.state.w,
            &self.state.core.y_train,
            self.state.k_diag_sum,
            self.state.a_frobenius2,
            self.state.core.likelihood.noise_variance(),
            self.state.core.n,
            self.state.core.m,
        )
    }

    /// Returns the leave-one-out mean and observation variance at every training point.
    ///
    /// `p(y_i | X, y_{-i}, θ, Z)` of the collapsed VFE posterior: the
    /// optimal `q(u)` without point `i` at fixed `θ` and `Z`, predicted at
    /// `x_i`. It is a rank-1 downdate of `B = σn² I + A Aᵀ` per point
    /// (Sherman–Morrison), `O(n m²)` in all. At `Z = X` it matches
    /// [`crate::FittedGpr::loo_predict`]. Mean and variance are
    /// inverse-transformed like those of `predict`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::NonPositiveDefiniteMatrix`] if a downdated `B` is
    /// not positive definite, or [`GprError::CholeskyFailed`] if an `f32` storage cannot factor `K_mm` again in `f64`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Fixed, GaussianLikelihood, Sgpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let mut fitted = Sgpr::new(
    ///     KernelSpec::from(RbfKernel::new(1.0)?),
    ///     GaussianLikelihood::new(0.1)?,
    /// )
    /// .with_optimizer(Fixed)
    /// .factor(&[0.0, 1.0, 2.0], 3, 1, &[0.0, 1.0, 0.5], &[0.5, 1.5], 2)
    /// .map_err(|(_, e)| e)?
    /// .into_online();
    /// fitted.insert(&[3.0], 0.2)?;
    /// let loo = fitted.loo_predict()?;
    /// assert_eq!(loo.mean.len(), fitted.n());
    /// # Ok(())
    /// # }
    /// ```
    pub fn loo_predict(&self) -> Result<Prediction<P::Refine>, GprError> {
        self.loo_predict_with(PredictOptions::default())
    }

    /// Returns leave-one-out mean and variance with an explicit variance kind.
    ///
    /// Latent variance is the VFE variance of `f(x_i)` without point `i`;
    /// observation variance adds `σn²`.
    ///
    /// # Errors
    ///
    /// Same as [`Self::loo_predict`].
    ///
    /// See the example on [`OnlineSgpr`].
    pub fn loo_predict_with(
        &self,
        options: PredictOptions,
    ) -> Result<Prediction<P::Refine>, GprError> {
        vfe_loo::<P, K::Supply>(
            &self.state.core,
            self.state.a.as_ref(),
            self.state.b_l.as_ref(),
            &self.state.w,
            options,
        )
    }

    /// Removes the training point identified by `id` and packs every buffer.
    ///
    /// Updates `B` with a rank-1 downdate. If that loses positive
    /// definiteness, the remaining points are factored again. The last
    /// remaining point cannot be deleted.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InsufficientData`] when `n == 1`,
    /// [`GprError::InvalidPointId`] when `id` is unknown or already deleted,
    /// [`GprError::InvalidConfig`] when the point is an inducing point of a
    /// model on supplied distances (remove it with [`Self::delete_inducing`]
    /// first), or [`GprError::CholeskyFailed`] when the system of the
    /// remaining points does not factor (a downdate that failed falls back
    /// to factoring it again, and a precision that refines in `f64` factors
    /// its predict weights again). On an error the model holds the same
    /// points.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Fixed, GaussianLikelihood, Sgpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let fitted = Sgpr::new(
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
        if self.state.core.n <= 1 {
            return Err(GprError::InsufficientData {
                n: self.state.core.n,
                min: 2,
            });
        }
        let idx = self.state.registry.index_of(id)?;
        if let Some(at) = self
            .state
            .core
            .supply()
            .and_then(|supply| supply.inducing_at(idx))
        {
            return Err(GprError::InvalidConfig {
                reason: format!(
                    "the point is inducing point {at}: delete it with delete_inducing first"
                ),
            });
        }
        self.atomically(|model| model.delete_at(idx))
    }

    /// [`Self::delete`] of the point at buffer index `idx`.
    fn delete_at(&mut self, idx: usize) -> Result<(), GprError> {
        let mut v = vec![P::Storage::from_f64(0.0); self.state.core.m];
        for (i, slot) in v.iter_mut().enumerate() {
            *slot = self.state.a[(i, idx)];
        }
        let x_pt = point_at(
            &self.state.core.x_train,
            self.state.core.n,
            self.state.core.d,
            idx,
        );
        let diag = kernel_diag_at::<P::Storage, K::Supply>(
            &self.state.core.kernel,
            &x_pt,
            self.state.core.d,
        )?;
        let mut col_norm = P::Storage::from_f64(0.0);
        for value in &v {
            col_norm += *value * *value;
        }
        let x_next = remove_point(
            &self.state.core.x_train,
            self.state.core.n,
            self.state.core.d,
            idx,
        );
        let mut y_next = self.state.core.y_train.clone();
        y_next.remove(idx);
        let x_obs_next = remove_point(
            &self.state.core.x_obs,
            self.state.core.n,
            self.state.core.d,
            idx,
        );
        let mut y_obs_next = self.state.core.y_obs.clone();
        y_obs_next.remove(idx);
        let mut b_trial = self.state.b_l.clone();
        let mut v_trial = v;
        if chol_rank1_downdate(&mut b_trial, &mut v_trial) {
            self.state.k_diag_sum -= diag;
            self.state.a_frobenius2 -= col_norm;
            let y_idx = self.state.core.y_train[idx];
            for (row, slot) in self.state.ay.iter_mut().enumerate() {
                *slot -= self.state.a[(row, idx)].to_f64() * y_idx;
            }
            remove_column_in_place(&mut self.state.a, idx);
            self.state.b_l = b_trial;
            // Only a precision whose weights can still fail keeps the row.
            let mut saved = Vec::new();
            if let Some(supply) = self.state.core.supply_mut() {
                supply.remove_point(idx, P::REFINES_IN_F64.then_some(&mut saved));
            }
            self.state.core.x_train = x_next;
            self.state.core.y_train = y_next;
            self.state.core.x_obs = x_obs_next;
            self.state.core.y_obs = y_obs_next;
            self.state.core.n -= 1;
            if let Err(err) = self.recompute_w() {
                if let Some(supply) = self.state.core.supply_mut() {
                    supply.restore_point(idx, &saved);
                }
                return Err(err);
            }
        } else {
            self.delete_by_reassembly(x_next, y_next, x_obs_next, y_obs_next, idx)?;
        }
        self.state.registry.remove_at(idx);
        Ok(())
    }

    /// The [`Self::delete`] path when the downdate of `B` fails: assembles
    /// the VFE system again from the remaining points (one fewer than now)
    /// and publishes its predict weights with it.
    pub(super) fn delete_by_reassembly(
        &mut self,
        x_next: Vec<f64>,
        y_next: Vec<f64>,
        x_obs_next: Vec<f64>,
        y_obs_next: Vec<f64>,
        idx: usize,
    ) -> Result<(), GprError> {
        let mut saved = Vec::new();
        if let Some(supply) = self.state.core.supply_mut() {
            supply.remove_point(idx, Some(&mut saved));
        }
        let result = self.reassemble_without(x_next, y_next, x_obs_next, y_obs_next);
        if result.is_err()
            && let Some(supply) = self.state.core.supply_mut()
        {
            supply.restore_point(idx, &saved);
        }
        result
    }

    /// [`Self::delete_by_reassembly`] once the training blocks dropped the
    /// point.
    fn reassemble_without(
        &mut self,
        x_next: Vec<f64>,
        y_next: Vec<f64>,
        x_obs_next: Vec<f64>,
        y_obs_next: Vec<f64>,
    ) -> Result<(), GprError> {
        let state = with_kernel_exp!(self.state.core.math, M => assemble_vfe::<M, P::Storage, K::Supply>(
            &self.state.core.kernel,
            self.state.core.jitter,
            self.state.core.likelihood,
            crate::sparse::SparseData {
                x: &x_next,
                n: self.state.core.n - 1,
                y: &y_next,
                ..self.state.core.data()
            },
            &mut self.scratch.storage,
            &mut self.scratch.f64,
        ))?;
        self.state.core.x_train = x_next;
        self.state.core.y_train = y_next;
        self.state.core.x_obs = x_obs_next;
        self.state.core.y_obs = y_obs_next;
        self.state.core.n -= 1;
        self.apply_vfe(state)
    }

    /// Commits one change of the inducing set: `change` gives it from the
    /// model once the training blocks of a model on supplied distances hold
    /// the new inducing set, and `z_train` / `z_obs` (`m × d`) are the
    /// inducing points after it. On an error `undo` puts those blocks back,
    /// so the model is unchanged.
    fn commit_inducing(
        &mut self,
        change: impl FnOnce(&mut Self) -> Result<InducingChange<P::Storage>, GprError>,
        z_train: Vec<f64>,
        z_obs: Vec<f64>,
        m: usize,
        undo: impl FnOnce(&mut SparseSupply),
    ) -> Result<(), GprError> {
        let result =
            change(self).and_then(|change| self.update_inducing(change, z_train, z_obs, m));
        if result.is_err()
            && let Some(supply) = self.state.core.supply_mut()
        {
            undo(supply);
        }
        result
    }

    /// Applies `change` by the incremental update of ADR 0005 on a copy of
    /// the factors, then takes that copy and publishes its predict weights
    /// ([`Self::apply_vfe`], as an `X` insert does). When the update fails
    /// to factor (a bordered pivot of an insert that is not positive, or a
    /// `B` of a delete that does not factor), the VFE system of the new
    /// inducing set is assembled again instead ([`Self::factor_inducing`]).
    fn update_inducing(
        &mut self,
        change: InducingChange<P::Storage>,
        z_train: Vec<f64>,
        z_obs: Vec<f64>,
        m: usize,
    ) -> Result<(), GprError> {
        let mut state = self.vfe_state();
        let noise = self.state.core.likelihood.noise_variance();
        let y = &self.state.core.y_train;
        let updated = match change {
            InducingChange::Insert { border, replace } => {
                inducing_append(&mut state, border, replace, noise, y)
            }
            InducingChange::Delete(idx) => inducing_delete(&mut state, noise, y, idx),
        };
        match updated {
            Ok(()) => {
                self.state.core.z_train = z_train;
                self.state.core.z_obs = z_obs;
                self.state.core.m = m;
                self.apply_vfe(state)
            }
            Err(GprError::CholeskyFailed { .. }) => self.factor_inducing(z_train, z_obs, m),
            Err(err) => Err(err),
        }
    }

    /// Assembles the VFE system at the inducing set `z_train` (`m × d`)
    /// again (`O(n·m²)`), with the offset of `K_mm + j I` the jitter policy
    /// gives it, and commits it with `z_obs`: the path of
    /// [`Self::update_inducing`] when the incremental update fails.
    fn factor_inducing(
        &mut self,
        z_train: Vec<f64>,
        z_obs: Vec<f64>,
        m: usize,
    ) -> Result<(), GprError> {
        let (state, w64) = with_kernel_exp!(self.state.core.math, M => assemble_vfe_with_f64_w::<M, P::Storage, K::Supply>(
            &self.state.core.kernel,
            self.state.core.jitter,
            self.state.core.likelihood,
            crate::sparse::SparseData {
                z: &z_train,
                m,
                ..self.state.core.data()
            },
            &mut self.scratch.storage,
            &mut self.scratch.f64,
        ))?;
        self.state.core.z_train = z_train;
        self.state.core.z_obs = z_obs;
        self.state.core.m = m;
        match w64 {
            // The `f64` weights of this assembly are the refined predict weights.
            Some(w64) if P::REFINES_IN_F64 => {
                self.state.predict_w = w64.into_iter().map(P::Refine::from_f64).collect();
                self.set_vfe(state);
                Ok(())
            }
            _ => self.apply_vfe(state),
        }
    }

    /// Removes the inducing point identified by `id` and packs every buffer.
    ///
    /// Drops that row of the factor of `K_mm + j I` by a trailing rank-1
    /// update, removes the row of the kept `K(Z, X) = L A`, and solves `A`
    /// against the new factor; `B` is formed and factored again from that
    /// `A` (`O(n·m²)`, no kernel evaluation). If that `B` does not factor,
    /// the VFE system of the reduced inducing set is assembled again
    /// instead. The last remaining inducing point cannot be deleted.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InsufficientData`] when `m == 1`,
    /// [`GprError::InvalidInducingId`] when `id` is unknown or already
    /// deleted, or [`GprError::CholeskyFailed`] when the reassembled
    /// smaller inducing set does not factor under [`Self::jitter_policy`]
    /// (or, for a precision that refines in `f64`, its `f64` weights do
    /// not). On an error the model holds the same inducing points.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Fixed, GaussianLikelihood, Sgpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let fitted = Sgpr::new(
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
    /// let id = online.inducing_ids()[0];
    /// online.delete_inducing(id)?;
    /// assert_eq!(online.m(), 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn delete_inducing(&mut self, id: InducingId) -> Result<(), GprError> {
        if self.state.core.m <= 1 {
            return Err(GprError::InsufficientData {
                n: self.state.core.m,
                min: 2,
            });
        }
        let idx = self.state.inducing.index_of(id)?;
        // One id per inducing point: checked before the model changes.
        if <K::Supply as SupplyViews>::held(&self.state.inducing_points)
            .is_some_and(|points| points.len() != self.state.core.m)
        {
            return Err(GprError::InvalidConfig {
                reason:
                    "the model's inducing point ids do not match its inducing points (an internal \
                         error)"
                        .to_owned(),
            });
        }
        self.atomically(|model| model.delete_inducing_at(idx))
    }

    /// [`Self::delete_inducing`] of the inducing point at index `idx`.
    fn delete_inducing_at(&mut self, idx: usize) -> Result<(), GprError> {
        let (m, d) = (self.state.core.m, self.state.core.d);
        let z_train = remove_point(&self.state.core.z_train, m, d, idx);
        let z_obs = remove_point(&self.state.core.z_obs, m, d, idx);
        // The training point of a model on supplied distances, kept to put
        // its column back if the smaller system does not factor.
        let mut saved = Vec::new();
        let point = self.state.core.supply_mut().and_then(|supply| {
            let point = supply.inducing.get(idx).copied();
            if point.is_some() {
                supply.remove_inducing(idx, &mut saved);
            }
            point
        });
        self.commit_inducing(
            |_| Ok(InducingChange::Delete(idx)),
            z_train,
            z_obs,
            m - 1,
            |supply| {
                if let Some(point) = point {
                    supply.undo_remove_inducing(idx, point, &saved);
                }
            },
        )?;
        self.state.inducing.remove_at(idx);
        if let Some(points) = <K::Supply as SupplyViews>::held_mut(&mut self.state.inducing_points)
            && idx < points.len()
        {
            points.remove(idx);
        }
        Ok(())
    }

    /// Converts this model back to a batch sparse GPR with fixed inducing points.
    ///
    /// Point and inducing identifiers are discarded. Parameters stay
    /// kernel then likelihood `θ`. The snapshot `Z` is the current
    /// inducing coordinates.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Fixed, GaussianLikelihood, Sgpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let fitted = Sgpr::new(
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
    pub fn into_fitted(self) -> FittedSgpr<O, FixedInducing, P, K> {
        FittedSgpr {
            core: self.state.core,
            scratch: self.scratch,
            optimizer: self.optimizer,
            inducing: PhantomData,
            _kernel: PhantomData,
            k_mm_l: self.state.k_mm_l,
            k_mm_jitter: self.state.k_mm_jitter,
            a: self.state.a,
            b_l: self.state.b_l,
            w: self.state.w,
            predict_w: self.state.predict_w,
            k_diag_sum: self.state.k_diag_sum,
            a_frobenius2: self.state.a_frobenius2,
        }
    }
}

impl<O, P, K> OnlineSgpr<O, P, K>
where
    P: crate::precision::GpScalar,
    K: crate::kernel::PointKernel,
{
    sparse_point_accessors!(state.core);
}

impl<O, P> OnlineSgpr<O, P>
where
    P: crate::precision::GpScalar,
{
    sparse_kernel_accessor!(state.core);

    /// Predicts at `xs` with [`PredictOptions::default`] (observation variance).
    ///
    /// # Errors
    ///
    /// Same as [`FittedSgpr::predict`].
    ///
    /// See the example on [`OnlineSgpr`].
    pub fn predict(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
    ) -> Result<Prediction<P::Refine>, GprError> {
        self.predict_with(xs, n_rows, n_cols, PredictOptions::default())
    }

    /// Predicts at `xs` with an explicit variance kind.
    ///
    /// # Errors
    ///
    /// Same as [`FittedSgpr::predict_with`].
    ///
    /// See the example on [`OnlineSgpr`].
    pub fn predict_with(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        options: PredictOptions,
    ) -> Result<Prediction<P::Refine>, GprError> {
        let mut out = Prediction::default();
        predict_vfe_into::<P, NoSupply>(
            &self.state.core,
            &VfeSystem::new(
                &self.state.core,
                self.state.k_mm_l.as_ref(),
                self.state.b_l.as_ref(),
                &self.state.predict_w,
            ),
            xs,
            n_rows,
            n_cols,
            std::iter::empty(),
            options,
            &mut PredictScratch::default(),
            &mut out,
        )?;
        Ok(out)
    }

    /// Returns the predictive mean and query–query covariance at `xs`.
    ///
    /// Default [`PredictOptions`] uses [`crate::VarianceKind::Observation`]:
    /// `σn²` is added on the diagonal in the transformed space. The diagonal
    /// is exactly the variance of `predict` for the same query. This
    /// path allocates an `m × m` matrix; `predict` stays diagonal.
    ///
    /// # Errors
    ///
    /// Same as [`FittedSgpr::predict`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Fixed, GaussianLikelihood, Sgpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let mut fitted = Sgpr::new(
    ///     KernelSpec::from(RbfKernel::new(1.0)?),
    ///     GaussianLikelihood::new(0.1)?,
    /// )
    /// .with_optimizer(Fixed)
    /// .factor(&[0.0, 1.0, 2.0], 3, 1, &[0.0, 1.0, 0.5], &[0.5, 1.5], 2)
    /// .map_err(|(_, e)| e)?
    /// .into_online();
    /// fitted.insert(&[3.0], 0.2)?;
    /// let cov = fitted.predict_covariance(&[0.25, 0.75], 2, 1)?;
    /// assert_eq!(cov.covariance.len(), 4);
    /// assert_eq!(cov.covariance[0], fitted.predict(&[0.25, 0.75], 2, 1)?.variance[0]);
    /// # Ok(())
    /// # }
    /// ```
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
    /// The VFE posterior covariance is `K** − Q** + K*m Σ Km*` with
    /// `Q** = K*m K_mm⁻¹ Km*` and `Σ = (K_mm + σn⁻² K_mn K_nm)⁻¹`.
    /// Latent diagonals are clipped at 0. Observation adds `σn²` on the
    /// diagonal in the transformed space, then the target transform scales
    /// the whole matrix.
    ///
    /// # Errors
    ///
    /// Same as [`FittedSgpr::predict`].
    ///
    /// See the example on [`OnlineSgpr`].
    pub fn predict_covariance_with(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        options: PredictOptions,
    ) -> Result<PredictiveCovariance<P::Refine>, GprError> {
        predict_vfe_covariance::<P, NoSupply>(
            &self.state.core,
            &VfeSystem::new(
                &self.state.core,
                self.state.k_mm_l.as_ref(),
                self.state.b_l.as_ref(),
                &self.state.predict_w,
            ),
            xs,
            n_rows,
            n_cols,
            std::iter::empty(),
            std::iter::empty(),
            options,
        )
    }

    /// Draws posterior samples at `xs` from `predict_covariance`.
    ///
    /// Each column of the returned column-major `m × n_draws` matrix is
    /// `μ + L z` with `z ∼ N(0, I)` and `L` the Cholesky factor of the
    /// posterior covariance, the same draw as [`crate::FittedGpr::sample`].
    /// `seed` is the start state of gprx's seeded generator (Xoshiro256++; the same seed gives the same draws on every platform). Zero draws
    /// returns an empty vector after the covariance is formed.
    ///
    /// # Errors
    ///
    /// Same as [`FittedSgpr::predict`], plus [`GprError::CholeskyFailed`] with
    /// [`CholeskyStage::Predict`](crate::CholeskyStage::Predict) if the
    /// posterior covariance cannot be factored after the retries of
    /// [`Self::jitter_policy`].
    ///
    /// See the example on [`OnlineSgpr`].
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
    /// Same as `sample` at `xs`.
    ///
    /// See the example on [`OnlineSgpr`].
    pub fn sample_with(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        options: PredictOptions,
        n_draws: usize,
        seed: u64,
    ) -> Result<Vec<P::Refine>, GprError> {
        self.predict_covariance_with(xs, n_rows, n_cols, options)?
            .draw(n_draws, seed, self.state.core.jitter)
    }

    /// Predicts at `xs` with [`PredictOptions::default`] into `out`.
    ///
    /// Same values as `predict` at `xs`. The buffers are kept on the model
    /// and `out` keeps its capacity, so a call after one with the same
    /// shapes allocates nothing.
    ///
    /// # Errors
    ///
    /// Same as [`FittedSgpr::predict`].
    ///
    /// See the example on [`OnlineSgpr`].
    pub fn predict_into(
        &mut self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        out: &mut Prediction<P::Refine>,
    ) -> Result<(), GprError> {
        self.predict_with_into(xs, n_rows, n_cols, PredictOptions::default(), out)
    }

    /// Predicts at `xs` with an explicit variance kind into `out`.
    ///
    /// Same values as `predict_with` at `xs`, with the buffers of
    /// `predict_into`.
    ///
    /// # Errors
    ///
    /// Same as [`FittedSgpr::predict`].
    ///
    /// See the example on [`OnlineSgpr`].
    pub fn predict_with_into(
        &mut self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        options: PredictOptions,
        out: &mut Prediction<P::Refine>,
    ) -> Result<(), GprError> {
        predict_vfe_into::<P, NoSupply>(
            &self.state.core,
            &VfeSystem::new(
                &self.state.core,
                self.state.k_mm_l.as_ref(),
                self.state.b_l.as_ref(),
                &self.state.predict_w,
            ),
            xs,
            n_rows,
            n_cols,
            std::iter::empty(),
            options,
            &mut self.scratch.predict,
            out,
        )
    }

    /// Appends one training point at the current `θ` with a rank-1 VFE update.
    ///
    /// Costs `O(m² + n·d)`, amortized: `A` grows in place, `A y` is updated
    /// with the new column, and `w = B⁻¹ A y` is one solve with the updated
    /// factor of `B`. The `O(n·d)` part moves the column-major training `X`.
    /// A precision that refines in `f64` ([`crate::MixedPrecision`]) also
    /// assembles its `f64` predict weights again from all `n` points,
    /// `O(n·m²)`: its weights are the exact `f64` solution, which no rank-1
    /// update of the stored `f32` factor reproduces.
    ///
    /// `x_new` has length [`Self::d`]. `x_new` and `y_new` are in the
    /// original units and go through the transforms fitted at training.
    /// Inducing coordinates are not moved.
    /// The returned [`PointId`] is new and is never reused after a later
    /// [`Self::delete`].
    ///
    /// # Errors
    ///
    /// Returns [`GprError::DimensionMismatch`] if `x_new` is the wrong length,
    /// [`GprError::NonFiniteInput`] if a value is `NaN` or `Inf`,
    /// [`GprError::IndexOutOfRange`] if no new [`PointId`] is left (only a
    /// loaded `next_point_id` near `u64::MAX` reaches this), or
    /// [`GprError::EmptyInput`] if `d` is zero.
    ///
    /// See the example on [`OnlineSgpr`].
    pub fn insert(&mut self, x_new: &[f64], y_new: f64) -> Result<PointId, GprError> {
        if x_new.len() != self.state.core.d {
            return Err(GprError::DimensionMismatch {
                x_dim: x_new.len(),
                expected_dim: self.state.core.d,
            });
        }
        if self.state.core.d == 0 {
            return Err(GprError::EmptyInput);
        }
        if x_new.iter().any(|v| !v.is_finite()) || !y_new.is_finite() {
            return Err(GprError::NonFiniteInput);
        }
        self.state.registry.require_room()?;
        let mut mapped = std::mem::take(&mut self.scratch.point);
        let result = self.atomically(|model| model.insert_mapped(x_new, y_new, &mut mapped));
        self.scratch.point = mapped;
        result
    }

    /// [`Self::insert`] after the checks. `mapped` receives `x_obs` through
    /// the input transform.
    fn insert_mapped(
        &mut self,
        x_obs: &[f64],
        y_obs: f64,
        mapped: &mut Vec<f64>,
    ) -> Result<PointId, GprError> {
        self.state.core.map_point(x_obs, mapped)?;
        let x_new = mapped.as_slice();
        let y_new = self.state.core.map_target(y_obs)?;
        let a_col = with_kernel_exp!(self.state.core.math, M => kernel_column::<M, P::Storage>(
            &self.state.core.kernel,
            &self.state.core.z_train,
            self.state.core.m,
            x_new,
            self.state.core.d,
            &mut self.scratch.storage,
        ))?;
        let k_diag = kernel_diag_at::<P::Storage, NoSupply>(
            &self.state.core.kernel,
            x_new,
            self.state.core.d,
        )?;
        let point = NewPoint {
            x: x_new,
            x_obs,
            y: y_new,
            y_obs,
        };
        self.append_point(a_col, k_diag, point, |_| {})
    }
}

impl<O, P, K: ModelKernel> OnlineSgpr<O, P, K>
where
    P: crate::precision::GpScalar,
{
    /// Appends a training point at the current `θ` from `K(Z, x)`
    /// (`a_col`, `m × 1`) and `k(x, x)`, and `point` its coordinates and
    /// target. `supply`
    /// writes its training blocks once every step that can fail has run.
    pub(super) fn append_point(
        &mut self,
        mut a_col: Mat<P::Storage>,
        k_diag: P::Storage,
        point: NewPoint<'_>,
        supply: impl FnOnce(&mut SparseSupply),
    ) -> Result<PointId, GprError> {
        let NewPoint {
            x: x_new,
            x_obs,
            y: y_new,
            y_obs,
        } = point;
        solve_lmm(self.state.k_mm_l.as_ref(), a_col.as_mut());
        let mut v = vec![P::Storage::from_f64(0.0); self.state.core.m];
        for (i, slot) in v.iter_mut().enumerate() {
            *slot = a_col[(i, 0)];
        }
        // No step below fails until the predict weights (see `atomically`).
        self.state.a_frobenius2 += frobenius2(a_col.as_ref());
        self.state.k_diag_sum += k_diag;
        for (row, slot) in self.state.ay.iter_mut().enumerate() {
            *slot += a_col[(row, 0)].to_f64() * y_new;
        }
        push_column(&mut self.state.a, a_col.as_ref());
        chol_rank1_update(&mut self.state.b_l, &mut v);
        append_point(
            &mut self.state.core.x_train,
            self.state.core.n,
            self.state.core.d,
            x_new,
        );
        append_point(
            &mut self.state.core.x_obs,
            self.state.core.n,
            self.state.core.d,
            x_obs,
        );
        self.state.core.y_train.push(y_new);
        self.state.core.y_obs.push(y_obs);
        if let Some(store) = self.state.core.supply_mut() {
            supply(store);
        }
        self.state.core.n += 1;
        if let Err(err) = self.recompute_w() {
            if let Some(store) = self.state.core.supply_mut() {
                store.pop_point();
            }
            return Err(err);
        }
        Ok(self.state.registry.insert())
    }
}

impl<O, P> OnlineSgpr<O, P>
where
    P: crate::precision::GpScalar,
{
    /// Appends one inducing point at the current `θ` by a bordered update of the VFE factors.
    ///
    /// `z_new` has length [`Self::d`], in the original coordinates of `X`;
    /// it goes through the input transform fitted at training. Training
    /// `X` / `y` are not moved.
    /// The returned [`InducingId`] is new and is never reused after a later
    /// [`Self::delete_inducing`].
    ///
    /// With `l = L⁻¹ k(Z, z_new)`, the factor of `K_mm + j I` gains the row
    /// `[lᵀ, ℓ]` with `ℓ² = k(z_new, z_new) + j − ‖l‖²`, where `j` is the
    /// offset the factor of `K_mm` took under [`Self::jitter_policy`].
    /// `A` gains the row `(k(z_new, X) − lᵀ A) / ℓ` and the factor of `B`
    /// its border; `w` is solved again (`O(n·m)` plus `O(m²)`). When `ℓ²`
    /// or the pivot of `B` is not positive (`z_new` on top of an inducing
    /// point, say), the VFE system of the enlarged inducing set is
    /// assembled again instead (`O(n·m²)`).
    ///
    /// # Errors
    ///
    /// Returns [`GprError::DimensionMismatch`] if `z_new` is the wrong length,
    /// [`GprError::NonFiniteInput`] if a value is `NaN` or `Inf`,
    /// [`GprError::EmptyInput`] if `d` is zero,
    /// [`GprError::IndexOutOfRange`] if no new [`InducingId`] is left (only a
    /// loaded `next_inducing_id` near `u64::MAX` reaches this), or
    /// [`GprError::CholeskyFailed`] if the reassembled enlarged inducing
    /// set does not factor under [`Self::jitter_policy`] (or, for a
    /// precision that refines in `f64`, its `f64` weights do not). On an
    /// error the model holds the same inducing points.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Fixed, GaussianLikelihood, Sgpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let fitted = Sgpr::new(
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
    /// online.insert_inducing(&[1.5])?;
    /// assert_eq!(online.m(), 3);
    /// # Ok(())
    /// # }
    /// ```
    pub fn insert_inducing(&mut self, z_new: &[f64]) -> Result<InducingId, GprError> {
        if z_new.len() != self.state.core.d {
            return Err(GprError::DimensionMismatch {
                x_dim: z_new.len(),
                expected_dim: self.state.core.d,
            });
        }
        if self.state.core.d == 0 {
            return Err(GprError::EmptyInput);
        }
        if z_new.iter().any(|v| !v.is_finite()) {
            return Err(GprError::NonFiniteInput);
        }
        self.state.inducing.require_room()?;
        self.atomically(|model| model.insert_inducing_checked(z_new))
    }

    /// [`Self::insert_inducing`] after the checks.
    fn insert_inducing_checked(&mut self, z_new: &[f64]) -> Result<InducingId, GprError> {
        let z_obs = z_new;
        let mut z_new = Vec::with_capacity(z_obs.len());
        self.state.core.map_point(z_obs, &mut z_new)?;
        let z_new = z_new.as_slice();
        let (n, m, d) = (self.state.core.n, self.state.core.m, self.state.core.d);
        let border = with_kernel_exp!(self.state.core.math, M => coordinate_border::<M, _>(
            &self.state.core.kernel,
            &self.state.core.x_train,
            n,
            d,
            &self.state.core.z_train,
            m,
            z_new,
            &mut self.scratch.storage,
        ))?;
        let mut z_train = self.state.core.z_train.clone();
        append_point(&mut z_train, m, d, z_new);
        let mut z_obs_next = self.state.core.z_obs.clone();
        append_point(&mut z_obs_next, m, d, z_obs);
        self.commit_inducing(
            |_| {
                Ok(InducingChange::Insert {
                    border,
                    replace: None,
                })
            },
            z_train,
            z_obs_next,
            m + 1,
            |_| {},
        )?;
        Ok(self.state.inducing.insert())
    }

    /// Writes this model to `dir` as `config.json` and `model.safetensors`.
    ///
    /// Stores the kernel, likelihood, kernel `exp`, `K_mm` jitter policy,
    /// precision, transforms (unfitted and fitted), the original `X`, `y`,
    /// and `Z`, and `Z` in transformed coordinates, and the point and inducing identifiers. The factors are
    /// not stored; [`crate::LoadedSgpr::load`] factors the system again at the saved `θ`
    /// and `Z`. Caller-defined kernels and transforms need their
    /// `persist_id` / `persist_state` and a [`crate::PersistRegistry`] entry.
    /// The optimizer and the inducing-point search are not stored.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::PersistFailed`] when the directory cannot be
    /// written or a kernel or transform has no persist form.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{Fixed, GaussianLikelihood, Sgpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let mut model = Sgpr::new(KernelSpec::from(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?)
    ///     .with_optimizer(Fixed)
    ///     .factor(&[0.0, 1.0, 2.0], 3, 1, &[0.0, 1.0, 0.5], &[0.5, 1.5], 2)
    ///     .map_err(|(_, e)| e)?
    ///     .into_online();
    /// model.insert(&[3.0], 0.2)?;
    /// let dir = std::env::temp_dir().join(format!("gprx-doctest-save-online-{}", std::process::id()));
    /// let _ = std::fs::remove_dir_all(&dir);
    /// model.save(&dir)?;
    /// let loaded = gprx::LoadedSgpr::load(&dir, &gprx::PersistRegistry::new())?;
    /// assert_eq!(loaded.n(), model.n());
    /// let _ = std::fs::remove_dir_all(&dir);
    /// # Ok(())
    /// # }
    /// ```
    pub fn save(&self, dir: impl AsRef<std::path::Path>) -> Result<(), GprError> {
        crate::persist::save_online_sgpr(self, dir.as_ref())
    }
}

impl<O, P, K: ModelKernel> OnlineSgpr<O, P, K>
where
    P: crate::precision::GpScalar,
    O: Clone + for<'a> Optimizer<SgprObjective<'a, O, FixedInducing, P, K>>,
{
    /// Re-runs the stored optimizer on the stored training data.
    ///
    /// Inducing coordinates stay fixed. The VFE system is rebuilt after the
    /// search.
    ///
    /// # Errors
    ///
    /// Same as `fit` of the [`Sgpr`](crate::Sgpr) of this kernel.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel};
    /// use gprx::{GaussianLikelihood, Sgpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let fitted = Sgpr::new(
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
        self.with_snapshot(FittedSgpr::optimize_hyperparameters)
    }
}

/// A point an insert appends: its coordinates and target through the
/// fitted transforms (`x`, `y`) and as the caller gave them.
pub(super) struct NewPoint<'a> {
    x: &'a [f64],
    x_obs: &'a [f64],
    y: f64,
    y_obs: f64,
}
