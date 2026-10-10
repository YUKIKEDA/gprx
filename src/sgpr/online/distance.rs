//! [`OnlineSgpr`] on supplied squared distances: save, the query of the
//! predict family, and the insert and delete of points and of inducing
//! points, each with the caller's `d²` per slot.

#[allow(unused_imports)]
use super::*;

impl<O, P: crate::precision::GpScalar, C: PointUse> OnlineSgpr<O, P, DistanceKernel<C>> {
    /// Writes this model to `dir` as `config.json` and `model.safetensors`.
    ///
    /// Stores what [`crate::FittedSgpr::save`] of a [`DistanceKernel`]
    /// stores, with the point and inducing identifiers.
    /// [`crate::persist::LoadedDistanceSgpr::load`] loads it as an online
    /// model.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::PersistFailed`] when the directory cannot be
    /// written or a kernel or transform has no persist form.
    ///
    /// See the example on [`crate::persist::LoadedDistanceSgpr`].
    pub fn save(&self, dir: impl AsRef<std::path::Path>) -> Result<(), GprError> {
        crate::persist::save_online_sgpr(self, dir.as_ref())
    }

    /// The factors a prediction reads.
    fn vfe_system(&self) -> VfeSystem<'_, P, SuppliedSpec> {
        VfeSystem::new(
            &self.state.core,
            self.state.k_mm_l.as_ref(),
            self.state.b_l.as_ref(),
            &self.state.predict_w,
        )
    }

    /// Appends one point from its supplied columns (each slot's `m × 1`
    /// squared distances from the inducing points to the new point) and,
    /// for a kernel that also reads coordinates, `x_obs`.
    fn insert_sources<'s>(
        &mut self,
        sources: impl IntoIterator<Item = DistanceSource<'s>>,
        x_obs: &[f64],
        y_obs: f64,
    ) -> Result<PointId, GprError> {
        let d = self.state.core.d;
        if x_obs.len() != d {
            return Err(GprError::DimensionMismatch {
                x_dim: x_obs.len(),
                expected_dim: d,
            });
        }
        if x_obs.iter().any(|v| !v.is_finite()) || !y_obs.is_finite() {
            return Err(GprError::NonFiniteInput);
        }
        self.state.registry.require_room()?;
        // The model's buffers, taken for the call as a predict takes them.
        let mut scratch = std::mem::take(&mut self.scratch.predict.query_storage);
        let result = self.insert_bound(sources, x_obs, y_obs, &mut scratch);
        self.scratch.predict.query_storage = scratch;
        result
    }

    fn insert_bound<'s>(
        &mut self,
        sources: impl IntoIterator<Item = DistanceSource<'s>>,
        x_obs: &[f64],
        y_obs: f64,
        scratch: &mut QueryScratch<P::Storage>,
    ) -> Result<PointId, GprError> {
        let (m, d) = (self.state.core.m, self.state.core.d);
        let cols = QuerySources::bind_column(&self.state.core.supplied.slots, sources, m, scratch)?;
        let mut mapped = std::mem::take(&mut self.scratch.point);
        let result = (|| {
            self.state.core.map_point(x_obs, &mut mapped)?;
            let y_new = self.state.core.map_target(y_obs)?;
            let compiled = self.state.core.kernel.compile_as::<P::Storage>();
            let z64 = pack_points(&self.state.core.z_train, m, d);
            let x64 = pack_points(&mapped, 1, d);
            let (mut z_cast, mut x_cast) = (P::Storage::empty_cols(), P::Storage::empty_cols());
            let z = P::Storage::storage_cols(z64.as_ref(), &mut z_cast);
            let x = P::Storage::storage_cols(x64.as_ref(), &mut x_cast);
            let views = CrossViews {
                x1: z,
                x2: x,
                dist: None,
                slots: <SuppliedSpec as SupplyViews>::rects(&cols),
            };
            let a_col = with_kernel_exp!(self.state.core.math, M => self
                .scratch
                .storage
                .cross::<M, SuppliedSpec>(&compiled, views))?;
            let k_diag =
                kernel_diag_at::<P::Storage, SuppliedSpec>(&self.state.core.kernel, &mapped, d)?;
            // The new row of every training block, read before anything
            // changes: the columns are checked, so nothing below fails.
            let exact = cols.f64_view();
            let blocks = self.state.core.supplied.supply.exact().xz.block_ids();
            let mut row = Vec::with_capacity(blocks.len() * m);
            for &at in &blocks {
                column_into(&exact, at, m, &mut row)?;
            }
            self.state.core.supplied.supply.reserve_point()?;
            self.atomically(|model| {
                let point = NewPoint {
                    x: &mapped,
                    x_obs,
                    y: y_new,
                    y_obs,
                };
                model.append_point(a_col, k_diag, point, |supply| {
                    supply.push_point(|b, col| row[b * m + col]);
                })
            })
        })();
        self.scratch.point = mapped;
        result
    }

    /// Makes training point `point` one more inducing point, from its
    /// supplied columns (each slot's `n × 1` squared distances from the
    /// training points to it).
    fn insert_inducing_sources<'s>(
        &mut self,
        point: PointId,
        sources: impl IntoIterator<Item = DistanceSource<'s>>,
    ) -> Result<InducingId, GprError> {
        let row = self.state.registry.index_of(point)?;
        if let Some(at) = self.state.core.supplied.supply.inducing_at(row) {
            return Err(GprError::InvalidConfig {
                reason: format!("the point is inducing point {at} already"),
            });
        }
        self.state.inducing.require_room()?;
        let mut scratch = std::mem::take(&mut self.scratch.predict.query_storage);
        let result = self.insert_inducing_bound(row, sources, &mut scratch);
        self.scratch.predict.query_storage = scratch;
        result
    }

    fn insert_inducing_bound<'s>(
        &mut self,
        row: usize,
        sources: impl IntoIterator<Item = DistanceSource<'s>>,
        scratch: &mut QueryScratch<P::Storage>,
    ) -> Result<InducingId, GprError> {
        let (n, m, d) = (self.state.core.n, self.state.core.m, self.state.core.d);
        let cols = QuerySources::bind_column(&self.state.core.supplied.slots, sources, n, scratch)?;
        let supply = &self.state.core.supplied.supply;
        let exact = cols.f64_view();
        let (column, mirror) = new_inducing_column(
            &supply.exact().xz,
            &supply.exact().zz,
            &supply.inducing,
            row,
            &exact,
            |at| cols.tidy(at),
            |at, err| cols.locate(at, err),
        )?;
        // A kernel that also reads coordinates takes the point's as `z`.
        let mut z_train = self.state.core.z_train.clone();
        append_point(
            &mut z_train,
            m,
            d,
            &point_at(&self.state.core.x_train, n, d, row),
        );
        let mut z_obs = self.state.core.z_obs.clone();
        append_point(
            &mut z_obs,
            m,
            d,
            &point_at(&self.state.core.x_obs, n, d, row),
        );
        self.state.core.supplied.supply.reserve_inducing()?;
        self.atomically(|model| {
            let mut saved = Vec::new();
            model.state.core.supplied.supply.add_inducing(
                row,
                |b, i| column[b * n + i],
                |b, col| mirror[b * m + col],
                &mut saved,
            );
            model.commit_inducing(z_train, z_obs, m + 1, |supply| {
                supply.undo_add_inducing(row, &saved);
            })?;
            Ok(model.state.inducing.insert())
        })
    }

    /// Returns a copy of the kernel whose hyperparameters this model owns.
    ///
    /// See the example on `insert` of a [`DistanceKernel<DistanceOnly>`]
    /// model.
    pub fn to_kernel(&self) -> DistanceKernel<C> {
        <DistanceKernel<C> as crate::kernel::ModelKernelParts>::from_spec(
            self.state.core.kernel.clone(),
        )
    }

    /// Returns the slots of the kernel, in the order of
    /// [`DistanceKernel::slots`]; bind supplies to these.
    ///
    /// See the example on `insert` of a [`DistanceKernel<DistanceOnly>`]
    /// model.
    pub fn slots(&self) -> Vec<DistanceSlot> {
        crate::kernel::spec_slots(&self.state.core.kernel)
    }

    /// Returns the training points that are the inducing points, in the
    /// order of the columns of the training blocks and the rows of a
    /// prediction's blocks. An id stays the same across inserts and
    /// deletes; [`Self::insert_inducing`] appends one.
    ///
    /// See the example on [`Self::insert_inducing`].
    pub fn inducing_points(&self) -> impl Iterator<Item = PointId> + '_ {
        let ids = self.point_ids();
        self.inducing()
            .iter()
            .filter_map(move |&place| ids.get(place).copied())
    }

    /// The places of the inducing points in [`Self::point_ids`]: the rows
    /// of the training blocks they are. A delete shifts them.
    pub(crate) fn inducing(&self) -> &[usize] {
        &self.state.core.supplied.supply.inducing
    }
}

impl<O, P: crate::precision::GpScalar, C: PointUse> DistanceQuery
    for OnlineSgpr<O, P, DistanceKernel<C>>
{
    type Refine = P::Refine;

    fn query_distances<'s>(
        &self,
        cross: impl IntoIterator<Item = DistanceSource<'s>>,
        points: QueryPoints<'_>,
        q: usize,
        options: PredictOptions,
    ) -> Result<Prediction<P::Refine>, GprError> {
        let mut out = Prediction::default();
        predict_vfe_into::<P, SuppliedSpec>(
            &self.state.core,
            &self.vfe_system(),
            points.xs,
            q,
            points.n_cols,
            cross,
            options,
            &mut PredictScratch::default(),
            &mut out,
        )?;
        Ok(out)
    }

    fn query_distances_into<'s>(
        &mut self,
        cross: impl IntoIterator<Item = DistanceSource<'s>>,
        points: QueryPoints<'_>,
        q: usize,
        options: PredictOptions,
        out: &mut Prediction<P::Refine>,
    ) -> Result<(), GprError> {
        let sys = VfeSystem::new(
            &self.state.core,
            self.state.k_mm_l.as_ref(),
            self.state.b_l.as_ref(),
            &self.state.predict_w,
        );
        predict_vfe_into::<P, SuppliedSpec>(
            &self.state.core,
            &sys,
            points.xs,
            q,
            points.n_cols,
            cross,
            options,
            &mut self.scratch.predict,
            out,
        )
    }

    fn query_distance_covariance<'s>(
        &self,
        cross: impl IntoIterator<Item = DistanceSource<'s>>,
        square: impl IntoIterator<Item = DistanceSource<'s>>,
        points: QueryPoints<'_>,
        q: usize,
        options: PredictOptions,
    ) -> Result<PredictiveCovariance<P::Refine>, GprError> {
        predict_vfe_covariance::<P, SuppliedSpec>(
            &self.state.core,
            &self.vfe_system(),
            points.xs,
            q,
            points.n_cols,
            cross,
            square,
            options,
        )
    }

    fn draw_jitter(&self) -> JitterPolicy {
        self.state.core.jitter
    }
}

impl<O, P: crate::precision::GpScalar> OnlineSgpr<O, P, DistanceKernel<DistanceOnly>> {
    /// Appends one training point at the current `θ` with the rank-1 VFE
    /// update of [`OnlineSgpr::insert`], from its squared distances to the
    /// inducing points.
    ///
    /// `sources` binds, per slot, the `m × 1` column from the `m` inducing
    /// points (in [`Self::inducing_points`] order) to the new point; an ARD slot
    /// binds one such column per dimension. A table may be borrowed, owned,
    /// or filled. Every value is checked (finite, `≥ 0`; see
    /// [`DistanceSource::tidy`]) and the column is kept as the point's row
    /// of the training blocks: the blocks keep room for more rows (a
    /// quarter more each time they fill), so most inserts write only the
    /// row. The returned [`PointId`] is never reused after a later
    /// [`Self::delete`].
    ///
    /// # Errors
    ///
    /// Returns [`GprError::NonFiniteInput`] if `y_new` is `NaN` or `Inf`,
    /// [`GprError::InvalidDistance`] for a negative or non-finite distance,
    /// [`GprError::LengthMismatch`] for a column whose length is not `m`,
    /// [`GprError::DistanceSlot`] if a source names a slot the kernel does
    /// not read, two name one slot, or a slot has none,
    /// [`GprError::IndexOutOfRange`] if no new
    /// [`PointId`] is left, [`GprError::SizeOverflow`] if the training
    /// blocks cannot grow, or [`GprError::CholeskyFailed`] if a precision
    /// that refines in `f64` cannot factor its predict weights again. On an
    /// error the model holds the same points.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{RbfKernel, ScalarDistance};
    /// use gprx::{Fixed, GaussianLikelihood, Sgpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let image = ScalarDistance::new();
    /// // Training points at x = 0, 1, 2, 3; inducing points: samples 0 and 2.
    /// let train = [0.0, 1.0, 4.0, 9.0, 4.0, 1.0, 0.0, 1.0];
    /// let fitted = Sgpr::new(image.kernel(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?)
    ///     .with_optimizer(Fixed)
    ///     .factor([image.borrow(&train)], 4, &[0.0, 1.0, 0.5, 0.25], &[0, 2])
    ///     .map_err(|(_, e)| e)?;
    /// let mut online = fitted.into_online();
    /// // A point at x = 4: its squared distances to the inducing points.
    /// let id = online.insert([image.from_vec(vec![16.0, 4.0])], 0.1)?;
    /// assert_eq!(online.n(), 5);
    /// assert_eq!(online.slots().len(), 1);
    /// let _kernel = online.to_kernel();
    /// // A query at x = 0.5: inducing points × query.
    /// let pred = online.predict([image.from_vec(vec![0.25, 2.25])], 1)?;
    /// assert_eq!(pred.mean.len(), 1);
    /// online.delete(id)?;
    /// assert_eq!(online.n(), 4);
    /// # Ok(())
    /// # }
    /// ```
    pub fn insert<'s>(
        &mut self,
        sources: impl IntoIterator<Item = DistanceSource<'s>>,
        y_new: f64,
    ) -> Result<PointId, GprError> {
        self.insert_sources(sources, &[], y_new)
    }

    /// Makes training point `point` one more inducing point at the current
    /// `θ`, from its squared distances to the training points.
    ///
    /// `sources` binds, per slot, the `n × 1` column from the `n` training
    /// points (in [`Self::point_ids`] order) to `point`; an ARD slot binds
    /// one such column per dimension. The column is checked as a training
    /// block is, and its square among the inducing points as a training
    /// square is: the value at `point` itself is zero, and the value at
    /// each inducing point equals the one the training blocks hold for that
    /// pair (a [`DistanceSource::tidy`] source repairs both to their mean).
    /// The column is kept as a new column of the training blocks, and the
    /// VFE system is assembled again, as [`OnlineSgpr::insert_inducing`]
    /// does when its bordered update is not taken. The returned
    /// [`InducingId`] is never reused after a later
    /// [`Self::delete_inducing`].
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidPointId`] if `point` is unknown or
    /// deleted, [`GprError::InvalidConfig`] if it is an inducing point
    /// already, [`GprError::InvalidDistance`] for a value the check refuses
    /// (located in the caller's column), [`GprError::LengthMismatch`] for a
    /// column whose length is not `n`, [`GprError::DistanceSlot`] if a
    /// source names a slot the kernel does not read, two name one slot, or a
    /// slot has none, [`GprError::IndexOutOfRange`] if no new [`InducingId`] is left,
    /// [`GprError::SizeOverflow`] if the training blocks cannot grow, or
    /// [`GprError::CholeskyFailed`] if the enlarged system does not factor.
    /// On an error the model holds the same inducing points.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{RbfKernel, ScalarDistance};
    /// use gprx::{Fixed, GaussianLikelihood, Sgpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let image = ScalarDistance::new();
    /// // Training points at x = 0, 1, 2, 3; inducing points: samples 0 and 2.
    /// let train = [0.0, 1.0, 4.0, 9.0, 4.0, 1.0, 0.0, 1.0];
    /// let mut online = Sgpr::new(image.kernel(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?)
    ///     .with_optimizer(Fixed)
    ///     .factor([image.borrow(&train)], 4, &[0.0, 1.0, 0.5, 0.25], &[0, 2])
    ///     .map_err(|(_, e)| e)?
    ///     .into_online();
    /// // Sample 1 (x = 1) becomes an inducing point: its squared distances
    /// // to the four training points.
    /// let point = online.point_ids()[1];
    /// let id = online.insert_inducing(point, [image.from_vec(vec![1.0, 0.0, 1.0, 4.0])])?;
    /// let ids = online.point_ids();
    /// let inducing: Vec<_> = online.inducing_points().collect();
    /// assert_eq!(inducing, [ids[0], ids[2], point]);
    /// // A query at x = 0.5: the three inducing points × query.
    /// let pred = online.predict([image.from_vec(vec![0.25, 2.25, 0.25])], 1)?;
    /// assert_eq!(pred.mean.len(), 1);
    /// online.delete_inducing(id)?;
    /// assert_eq!(online.m(), 2);
    /// # Ok(())
    /// # }
    /// ```
    pub fn insert_inducing<'s>(
        &mut self,
        point: PointId,
        sources: impl IntoIterator<Item = DistanceSource<'s>>,
    ) -> Result<InducingId, GprError> {
        self.insert_inducing_sources(point, sources)
    }
}

impl<O, P: crate::precision::GpScalar> OnlineSgpr<O, P, DistanceKernel<WithPoints>> {
    /// Appends one training point at the current `θ`, from its squared
    /// distances to the inducing points and its coordinates `x_new`
    /// (length [`Self::d`]).
    ///
    /// The columns are as in the insert of a [`DistanceOnly`] model;
    /// `x_new` goes through the stored input transform, which is not
    /// re-fit.
    ///
    /// # Errors
    ///
    /// Those of the insert of a [`DistanceOnly`] model, and
    /// [`GprError::DimensionMismatch`] if `x_new` is the wrong length or
    /// [`GprError::NonFiniteInput`] if one of its values is `NaN` or `Inf`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{KernelSpec, RbfKernel, ScalarDistance};
    /// use gprx::{Fixed, GaussianLikelihood, Sgpr};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let image = ScalarDistance::new();
    /// let kernel = image.kernel(RbfKernel::new(1.0)?) * KernelSpec::from(RbfKernel::new(2.0)?);
    /// let train = [0.0, 1.0, 4.0, 9.0, 4.0, 1.0, 0.0, 1.0];
    /// let x = [0.0, 0.5, 1.0, 1.5];
    /// let mut online = Sgpr::new(kernel, GaussianLikelihood::new(0.1)?)
    ///     .with_optimizer(Fixed)
    ///     .factor([image.borrow(&train)], 4, &x, 1, &[0.0, 1.0, 0.5, 0.25], &[0, 2])
    ///     .map_err(|(_, e)| e)?
    ///     .into_online();
    /// online.insert([image.from_vec(vec![16.0, 4.0])], &[2.0], 0.1)?;
    /// // Sample 1 becomes an inducing point; its coordinates come along.
    /// let point = online.point_ids()[1];
    /// online.insert_inducing(point, [image.from_vec(vec![1.0, 0.0, 1.0, 4.0, 9.0])])?;
    /// assert_eq!(online.z(), &[0.0, 1.0, 0.5]);
    /// let pred = online.predict([image.from_vec(vec![0.25, 2.25, 0.25])], &[0.25], 1, 1)?;
    /// assert_eq!(pred.mean.len(), 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn insert<'s>(
        &mut self,
        sources: impl IntoIterator<Item = DistanceSource<'s>>,
        x_new: &[f64],
        y_new: f64,
    ) -> Result<PointId, GprError> {
        self.insert_sources(sources, x_new, y_new)
    }

    /// Makes training point `point` one more inducing point, as the
    /// [`DistanceOnly`] model does; its coordinates are those of `point`.
    ///
    /// # Errors
    ///
    /// Those of the [`DistanceOnly`] model's `insert_inducing`.
    ///
    /// See the example on [`Self::insert`].
    pub fn insert_inducing<'s>(
        &mut self,
        point: PointId,
        sources: impl IntoIterator<Item = DistanceSource<'s>>,
    ) -> Result<InducingId, GprError> {
        self.insert_inducing_sources(point, sources)
    }
}

distance_predict!(
    impl [O, P: crate::precision::GpScalar] OnlineSgpr<O, P, DistanceKernel<DistanceOnly>>,
    refine = P::Refine,
    args = (),
    tail = (),
    points = QueryPoints::NONE,
    count = q,
    cross = {
        /// the `m × q` squared distances from the `m` inducing points (in
        /// [`Self::inducing_points`] order) to the `q` queries.
    },
    reads = {
        /// A table is read in place for this call (an `f32` model reads it
        /// in `f64`, as it predicts); a fill writes scratch once.
    },
    predict_doc = {
        /// See the example on [`Self::insert`].
    },
    covariance_doc = {
        /// See [`crate::FittedSgpr::predict_covariance`] of a
        /// [`DistanceKernel<DistanceOnly>`]: the same arguments.
    },
);

distance_predict!(
    impl [O, P: crate::precision::GpScalar] OnlineSgpr<O, P, DistanceKernel<WithPoints>>,
    refine = P::Refine,
    args = (xs: &[f64]),
    tail = (n_cols: usize),
    points = QueryPoints { xs, n_cols },
    count = q,
    cross = {
        /// the `m × q` squared distances from the `m` inducing points (in
        /// [`Self::inducing_points`] order) to the `q` queries.
    },
    reads = {
        /// A table is read in place for this call (an `f32` model reads it
        /// in `f64`, as it predicts); a fill writes scratch once.
    },
    predict_doc = {
        /// See the example on [`Self::insert`].
    },
    covariance_doc = {
        /// See [`crate::FittedSgpr::predict_covariance`] of a
        /// [`DistanceKernel<WithPoints>`]: the same arguments.
    },
);
