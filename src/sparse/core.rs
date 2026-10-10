//! The settings every sparse trainer holds ([`SparseSpec`]), the training
//! data and `θ` every fitted sparse model holds ([`SparseCore`]), and the
//! kernel + likelihood `θ` over both.

#[allow(
    unused_imports,
    reason = "a split file takes its parent's imports whole; each uses some"
)]
use super::*;

/// Kernel, likelihood, kernel `exp`, and the unfitted input / target
/// transforms of an untrained sparse model.
pub(crate) struct SparseSpec<U: Supply = NoSupply> {
    pub(crate) kernel: KernelSpec<U>,
    pub(crate) likelihood: GaussianLikelihood,
    pub(crate) math: KernelExp,
    /// Retries for factoring `K_mm`.
    pub(crate) jitter: JitterPolicy,
    pub(crate) x_transform: Box<dyn UnfittedTransform>,
    pub(crate) y_transform: Box<dyn UnfittedTarget>,
}

impl<U: Supply> Clone for SparseSpec<U> {
    fn clone(&self) -> Self {
        Self {
            kernel: self.kernel.clone(),
            likelihood: self.likelihood,
            math: self.math,
            jitter: self.jitter,
            x_transform: self.x_transform.clone_box(),
            y_transform: self.y_transform.clone_box(),
        }
    }
}

impl<U: Supply> fmt::Debug for SparseSpec<U> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SparseSpec")
            .field("kernel", &self.kernel)
            .field("likelihood", &self.likelihood)
            .field("math", &self.math)
            .field("jitter", &self.jitter)
            .finish_non_exhaustive()
    }
}

impl<U: Supply> SparseSpec<U> {
    pub(crate) fn new(kernel: KernelSpec<U>, likelihood: GaussianLikelihood) -> Self {
        Self {
            kernel,
            likelihood,
            math: KernelExp::default(),
            jitter: default_k_mm_jitter(),
            x_transform: Box::new(IdentityInput),
            y_transform: Box::new(IdentityTarget),
        }
    }

    /// Kernel `θ` then likelihood `θ`.
    pub(crate) fn theta_len(&self) -> usize {
        theta_len(&self.kernel, &self.likelihood)
    }

    pub(crate) fn read_theta(&self, out: &mut [f64]) -> Result<(), GprError> {
        write_params(&self.kernel, &self.likelihood, out)
    }

    /// Writes kernel then likelihood `θ`, both or neither.
    pub(crate) fn write_theta(&mut self, params: &[f64]) -> Result<(), GprError> {
        crate::data::require_count(params.len(), self.theta_len(), "parameters")?;
        let (kernel, likelihood) = stage_theta(&self.kernel, &self.likelihood, params)?;
        self.kernel = kernel;
        self.likelihood = likelihood;
        Ok(())
    }
}

/// The saved parts of a fitted sparse model ([`SparseCore::from_persisted`]).
pub(crate) struct PersistedSparse<U: Supply = NoSupply> {
    pub(crate) spec: SparseSpec<U>,
    pub(crate) x_transform: Box<dyn Transform>,
    pub(crate) y_transform: Box<dyn TargetTransform>,
    pub(crate) x_obs: Vec<f64>,
    pub(crate) y_obs: Vec<f64>,
    pub(crate) z_obs: Vec<f64>,
    pub(crate) z_train: Vec<f64>,
    pub(crate) n: usize,
    pub(crate) m: usize,
    pub(crate) d: usize,
    /// The supplied `d²`, the inducing indices, and the slots of a kernel
    /// on supplied distances; nothing for a coordinate kernel.
    pub(crate) supplied: <U as SupplyViews>::Held<SparseSupplied>,
}

/// The squares `store` holds as kind `U` reads them: none for a
/// coordinate kernel, which has no store.
pub(crate) fn squares_of<T: KernelScalar, U: Supply>(
    store: Option<&BlockStore<T>>,
) -> U::Squares<'_, T> {
    U::squares(
        store.map_or(&crate::kernel::NO_SLOTS as &dyn SquareSlots<T>, |store| {
            store
        }),
    )
}

/// The blocks `store` holds as kind `U` reads them: none for a coordinate
/// kernel, which has no store.
pub(crate) fn rects_of<T: KernelScalar, U: Supply>(
    store: Option<&BlockStore<T>>,
) -> U::Rects<'_, T> {
    U::rects(store.map_or(&crate::kernel::NO_SLOTS as &dyn RectSlots<T>, |store| store))
}

/// What a sparse model of a kernel on supplied distances holds beside its
/// kernel ([`SparseCore::supplied`]).
#[derive(Clone, Debug, Default)]
pub(crate) struct SparseSupplied {
    /// The supplied `d²` and the inducing indices.
    pub(crate) supply: SparseSupply,
    /// The kernel's slots, in the order of
    /// [`crate::kernel::DistanceKernel::slots`].
    pub(crate) slots: Vec<crate::kernel::DistanceSlot>,
}

/// The transformed training data a sparse model's kernel and objective
/// read: `x` (`n × d`, column-major; `d` is `0` for a kernel on supplied
/// distances alone), `y`, the inducing points `z` (`m × d`), and the
/// supplied `d²` (none for a coordinate kernel).
#[derive(Clone, Copy)]
pub(crate) struct SparseData<'a> {
    pub(crate) x: &'a [f64],
    pub(crate) n: usize,
    pub(crate) d: usize,
    pub(crate) y: &'a [f64],
    pub(crate) z: &'a [f64],
    pub(crate) m: usize,
    /// The supply of a kernel on supplied distances; `None` for a
    /// coordinate kernel.
    pub(crate) supply: Option<&'a SparseSupply>,
}

impl SparseData<'_> {
    /// Checks the shapes and values.
    ///
    /// # Errors
    ///
    /// The errors of [`validate_training`] and [`validate_inducing`]; with
    /// no feature (`d = 0`, a kernel on supplied distances alone), only
    /// `n`, `m`, and `y` are checked.
    pub(crate) fn validate(&self) -> Result<(), GprError> {
        if self.d == 0 && self.supply.is_some() {
            crate::data::require_nonempty(self.n)?;
            crate::data::require_nonempty(self.m)?;
            crate::data::require_count(self.x.len(), 0, "feature values")?;
            crate::data::require_count(self.z.len(), 0, "inducing feature values")?;
            crate::data::require_count(self.y.len(), self.n, "targets")?;
            return crate::data::require_finite(self.y);
        }
        validate_training(self.x, self.n, self.d, self.y)?;
        validate_inducing(self.z, self.m, self.d)
    }

    /// Checks the inducing points alone, as [`Self::validate`] does.
    ///
    /// # Errors
    ///
    /// The errors of [`validate_inducing`]; with no feature, only `m`.
    pub(crate) fn validate_inducing(&self) -> Result<(), GprError> {
        if self.d == 0 && self.supply.is_some() {
            crate::data::require_nonempty(self.m)?;
            return crate::data::require_count(self.z.len(), 0, "inducing feature values");
        }
        validate_inducing(self.z, self.m, self.d)
    }
}

/// The two point sets a sparse kernel reads at `T`: the training points
/// `x` (`n × d`) and the inducing points `z` (`m × d`), with the supplied
/// `d²` among the inducing points (`zz`) and from the training points to
/// them (`xz`, `n × m`, as the caller laid them out).
pub(crate) struct SparseSets<'a, T: KernelScalar, U: Supply> {
    pub(crate) x: MatRef<'a, T>,
    pub(crate) z: MatRef<'a, T>,
    pub(crate) zz: U::Squares<'a, T>,
    pub(crate) xz: U::Rects<'a, T>,
}

impl<T: KernelScalar, U: Supply> Clone for SparseSets<'_, T, U> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: KernelScalar, U: Supply> Copy for SparseSets<'_, T, U> {}

impl<'a, T: KernelScalar, U: Supply> SparseSets<'a, T, U> {
    /// The sets `x` and `z` with the supply `at`; a coordinate kernel has
    /// none.
    pub(crate) fn new(x: MatRef<'a, T>, z: MatRef<'a, T>, at: Option<&'a SupplyAt<T>>) -> Self {
        Self {
            x,
            z,
            zz: squares_of::<T, U>(at.map(|at| &at.zz)),
            xz: rects_of::<T, U>(at.map(|at| &at.xz)),
        }
    }

    /// The inputs of `K_mm = k(Z, Z)`.
    pub(crate) fn k_mm(&self) -> GramInputs<'a, T, U> {
        GramInputs::supplied(self.z, self.zz)
    }

    /// The views of `K(X, Z)` (`n × m`): the supplied blocks are read in
    /// their own layout. A coordinate kernel forms `K(Z, X)` directly
    /// ([`KernelScratch::cross_mn_into`]).
    pub(crate) fn k_nm(&self) -> CrossViews<'a, T, U> {
        CrossViews {
            x1: self.x,
            x2: self.z,
            dist: None,
            slots: self.xz,
        }
    }
}

/// Training data and settings of a fitted sparse model.
///
/// `x_obs` / `z_obs` / `y_obs` are what the caller passed (column-major
/// `n × d` and `m × d`). `x_train` / `z_train` / `y_train` are the same data through the
/// fitted transforms; every factor, gradient, and prediction reads those.
pub(crate) struct SparseCore<U: Supply = NoSupply> {
    pub(crate) kernel: KernelSpec<U>,
    pub(crate) likelihood: GaussianLikelihood,
    pub(crate) math: KernelExp,
    /// Retries for factoring `K_mm`.
    pub(crate) jitter: JitterPolicy,
    pub(crate) x_unfitted: Box<dyn UnfittedTransform>,
    pub(crate) y_unfitted: Box<dyn UnfittedTarget>,
    pub(crate) x_transform: Box<dyn Transform>,
    pub(crate) y_transform: Box<dyn TargetTransform>,
    pub(crate) x_obs: Vec<f64>,
    pub(crate) z_obs: Vec<f64>,
    pub(crate) y_obs: Vec<f64>,
    pub(crate) x_train: Vec<f64>,
    pub(crate) z_train: Vec<f64>,
    pub(crate) y_train: Vec<f64>,
    pub(crate) n: usize,
    pub(crate) m: usize,
    pub(crate) d: usize,
    /// The supplied `d²` the kernel reads and its slots; nothing for a
    /// coordinate kernel.
    pub(crate) supplied: <U as SupplyViews>::Held<SparseSupplied>,
}

impl<U: Supply> Clone for SparseCore<U> {
    fn clone(&self) -> Self {
        Self {
            kernel: self.kernel.clone(),
            likelihood: self.likelihood,
            math: self.math,
            jitter: self.jitter,
            x_unfitted: self.x_unfitted.clone_box(),
            y_unfitted: self.y_unfitted.clone_box(),
            x_transform: self.x_transform.clone_box(),
            y_transform: self.y_transform.clone_box(),
            x_obs: self.x_obs.clone(),
            z_obs: self.z_obs.clone(),
            y_obs: self.y_obs.clone(),
            x_train: self.x_train.clone(),
            z_train: self.z_train.clone(),
            y_train: self.y_train.clone(),
            n: self.n,
            m: self.m,
            d: self.d,
            supplied: self.supplied.clone(),
        }
    }
}

impl<U: Supply> fmt::Debug for SparseCore<U> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SparseCore")
            .field("kernel", &self.kernel)
            .field("likelihood", &self.likelihood)
            .field("math", &self.math)
            .field("jitter", &self.jitter)
            .field("n", &self.n)
            .field("m", &self.m)
            .field("d", &self.d)
            .finish_non_exhaustive()
    }
}

impl SparseCore {
    /// Checks the training data and the inducing points, fits the input
    /// transform on `X` and the target transform on `y`, and maps `X`, `Z`,
    /// and `y` through them.
    ///
    /// # Errors
    ///
    /// Returns the input errors of [`crate::data::validate_training`] and
    /// [`crate::data::validate_inducing`], or the error of a transform fit
    /// or map.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare(
        spec: &SparseSpec,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
        y: &[f64],
        z: &[f64],
        n_inducing: usize,
    ) -> Result<Self, GprError> {
        validate_training(x, n_rows, n_cols, y)?;
        validate_inducing(z, n_inducing, n_cols)?;
        let x_transform = spec.x_transform.clone_box().fit(x, n_rows, n_cols)?;
        let y_transform = spec.y_transform.clone_box().fit(y)?;
        let mut x_train = x.to_vec();
        x_transform.apply(&mut x_train, n_rows, n_cols)?;
        let mut z_train = z.to_vec();
        x_transform.apply(&mut z_train, n_inducing, n_cols)?;
        let mut y_train = y.to_vec();
        y_transform.transform(&mut y_train)?;
        Ok(Self {
            kernel: spec.kernel.clone(),
            likelihood: spec.likelihood,
            math: spec.math,
            jitter: spec.jitter,
            x_unfitted: spec.x_transform.clone_box(),
            y_unfitted: spec.y_transform.clone_box(),
            x_transform,
            y_transform,
            x_obs: x.to_vec(),
            z_obs: z.to_vec(),
            y_obs: y.to_vec(),
            x_train,
            z_train,
            y_train,
            n: n_rows,
            m: n_inducing,
            d: n_cols,
            supplied: (),
        })
    }
}

impl<U: Supply> SparseCore<U> {
    /// A fitted core read back from a persist directory. The fitted
    /// transforms are the saved ones, not fitted again: an online model's
    /// were fitted on its first training set. `X` and `y` go through them;
    /// `z_train` is the saved transformed `Z`, so a moved `Z` is not mapped
    /// back and forth. A kernel on supplied distances alone has `d = 0`
    /// and no coordinates.
    ///
    /// # Errors
    ///
    /// Returns the input errors of [`crate::data::validate_training`] and
    /// [`crate::data::validate_inducing`], or the error of a transform map.
    pub(crate) fn from_persisted(parts: PersistedSparse<U>) -> Result<Self, GprError> {
        let PersistedSparse {
            spec,
            x_transform,
            y_transform,
            x_obs,
            y_obs,
            z_obs,
            z_train,
            n,
            m,
            d,
            supplied,
        } = parts;
        if d == 0 {
            crate::data::require_nonempty(n)?;
            crate::data::require_nonempty(m)?;
            crate::data::require_count(x_obs.len(), 0, "feature values")?;
            crate::data::require_count(z_obs.len(), 0, "inducing values")?;
            crate::data::require_count(z_train.len(), 0, "inducing values")?;
            crate::data::require_count(y_obs.len(), n, "targets")?;
            crate::data::require_finite(&y_obs)?;
        } else {
            validate_training(&x_obs, n, d, &y_obs)?;
            validate_inducing(&z_obs, m, d)?;
            validate_inducing(&z_train, m, d)?;
        }
        let mut x_train = x_obs.clone();
        if d > 0 {
            x_transform.apply(&mut x_train, n, d)?;
        }
        let mut y_train = y_obs.clone();
        y_transform.transform(&mut y_train)?;
        // The saved maps are read back, not fitted: a map that sends the
        // data past `f64` is refused here, not later in a factor.
        crate::data::require_finite(&x_train)?;
        crate::data::require_finite(&y_train)?;
        crate::data::require_finite(&z_train)?;
        Ok(Self {
            kernel: spec.kernel,
            likelihood: spec.likelihood,
            math: spec.math,
            jitter: spec.jitter,
            x_unfitted: spec.x_transform,
            y_unfitted: spec.y_transform,
            x_transform,
            y_transform,
            x_obs,
            z_obs,
            y_obs,
            x_train,
            z_train,
            y_train,
            n,
            m,
            d,
            supplied,
        })
    }

    /// The core of a model on supplied distances: `sources` bind each
    /// slot's `n × m` block from the `n` training points to the inducing
    /// points, which are the training points `inducing` (in that order).
    /// `x` (`n × n_cols`, column-major) holds the coordinates of the
    /// coordinate leaves, empty with `n_cols = 0` for a kernel on supplied
    /// distances alone; the inducing points' coordinates are their rows.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::EmptyInput`] if `n` or `inducing` is empty, the
    /// input errors of [`Self::prepare`], and the errors of
    /// [`SparseSupply::bind`].
    pub(crate) fn prepare_supplied<'s, S: KernelScalar>(
        spec: &SparseSpec<U>,
        sources: impl IntoIterator<Item = crate::kernel::DistanceSource<'s>>,
        n: usize,
        (x, n_cols): (&[f64], usize),
        y: &[f64],
        inducing: &[usize],
    ) -> Result<Self, GprError> {
        crate::data::require_nonempty(n)?;
        crate::data::require_nonempty(inducing.len())?;
        let m = inducing.len();
        if n_cols == 0 {
            crate::data::require_count(x.len(), 0, "feature values")?;
            crate::data::require_count(y.len(), n, "targets")?;
            crate::data::require_finite(y)?;
        } else {
            validate_training(x, n, n_cols, y)?;
        }
        let supplied = U::try_hold(|| {
            let slots = crate::kernel::spec_slots(&spec.kernel);
            let supply = SparseSupply::bind::<S>(&slots, sources, n, inducing)?;
            Ok::<_, GprError>(SparseSupplied { supply, slots })
        })?;
        let mut z_obs = Vec::with_capacity(m * n_cols);
        for j in 0..n_cols {
            z_obs.extend(inducing.iter().map(|&i| x[j * n + i]));
        }
        let x_transform: Box<dyn Transform> = if n_cols == 0 {
            Box::new(IdentityInput)
        } else {
            spec.x_transform.clone_box().fit(x, n, n_cols)?
        };
        let y_transform = spec.y_transform.clone_box().fit(y)?;
        let mut x_train = x.to_vec();
        let mut z_train = z_obs.clone();
        if n_cols > 0 {
            x_transform.apply(&mut x_train, n, n_cols)?;
            x_transform.apply(&mut z_train, m, n_cols)?;
        }
        let mut y_train = y.to_vec();
        y_transform.transform(&mut y_train)?;
        Ok(Self {
            kernel: spec.kernel.clone(),
            likelihood: spec.likelihood,
            math: spec.math,
            jitter: spec.jitter,
            x_unfitted: spec.x_transform.clone_box(),
            y_unfitted: spec.y_transform.clone_box(),
            x_transform,
            y_transform,
            x_obs: x.to_vec(),
            z_obs,
            y_obs: y.to_vec(),
            x_train,
            z_train,
            y_train,
            n,
            m,
            d: n_cols,
            supplied,
        })
    }

    /// The transformed training data and the supply.
    pub(crate) fn data(&self) -> SparseData<'_> {
        SparseData {
            x: &self.x_train,
            n: self.n,
            d: self.d,
            y: &self.y_train,
            z: &self.z_train,
            m: self.m,
            supply: self.supply(),
        }
    }

    /// The supplied `d²` of a kernel on supplied distances; `None` for a
    /// coordinate kernel.
    pub(crate) fn supply(&self) -> Option<&SparseSupply> {
        U::held(&self.supplied).map(|held| &held.supply)
    }

    /// [`Self::supply`], to change.
    pub(crate) fn supply_mut(&mut self) -> Option<&mut SparseSupply> {
        U::held_mut(&mut self.supplied).map(|held| &mut held.supply)
    }

    /// The kernel's slots, in its order; none for a coordinate kernel.
    pub(crate) fn slots(&self) -> &[crate::kernel::DistanceSlot] {
        U::held(&self.supplied).map_or(&[], |held| &held.slots)
    }

    /// Kernel `θ` then likelihood `θ`.
    pub(crate) fn theta_len(&self) -> usize {
        theta_len(&self.kernel, &self.likelihood)
    }

    pub(crate) fn read_theta(&self, out: &mut [f64]) -> Result<(), GprError> {
        write_params(&self.kernel, &self.likelihood, out)
    }

    /// Kernel and likelihood with `params` (kernel `θ` then likelihood `θ`)
    /// written, leaving `self` unchanged.
    pub(crate) fn stage_theta(
        &self,
        params: &[f64],
    ) -> Result<(KernelSpec<U>, GaussianLikelihood), GprError> {
        stage_theta(&self.kernel, &self.likelihood, params)
    }

    /// Kernel intervals then the likelihood interval.
    pub(crate) fn theta_intervals(&self, out: &mut [Interval]) -> Result<(), GprError> {
        let n_kernel = self.kernel.num_params();
        crate::data::require_count(out.len(), self.theta_len(), "intervals")?;
        let mut offset = 0;
        self.kernel
            .write_intervals(&mut out[..n_kernel], &mut offset)?;
        out[n_kernel] = self.likelihood.bounds();
        Ok(())
    }

    /// The original coordinates of transformed inducing points `z`
    /// (`rows × d`).
    pub(crate) fn inducing_obs(&self, z: &[f64], rows: usize) -> Result<Vec<f64>, GprError> {
        let mut z_obs = z.to_vec();
        self.x_transform.inverse_apply(&mut z_obs, rows, self.d)?;
        Ok(z_obs)
    }

    /// One point (`d` features) through the fitted input transform, into
    /// `out`.
    pub(crate) fn map_point(&self, point: &[f64], out: &mut Vec<f64>) -> Result<(), GprError> {
        out.clear();
        out.extend_from_slice(point);
        // A model of supplied distances alone has no coordinates to map.
        if self.d == 0 {
            return Ok(());
        }
        self.x_transform.apply(out, 1, self.d)
    }

    /// One target through the fitted target transform.
    pub(crate) fn map_target(&self, target: f64) -> Result<f64, GprError> {
        let mut mapped = [target];
        self.y_transform.transform(&mut mapped)?;
        Ok(mapped[0])
    }

    /// Query points (`n_rows × d`, column-major) through the fitted input
    /// transform, into `out`. Checks the feature count and the packing first.
    pub(crate) fn map_query_into(
        &self,
        xs: &[f64],
        n_rows: usize,
        n_cols: usize,
        out: &mut Vec<f64>,
    ) -> Result<(), GprError> {
        if n_cols != self.d {
            return Err(GprError::DimensionMismatch {
                x_dim: n_cols,
                expected_dim: self.d,
            });
        }
        out.clear();
        if n_cols == 0 {
            // A kernel on supplied distances alone reads no coordinates.
            crate::data::require_nonempty(n_rows)?;
            return crate::data::require_count(xs.len(), 0, "query feature values");
        }
        validate_query(xs, n_rows, n_cols)?;
        out.extend_from_slice(xs);
        self.x_transform.apply(out, n_rows, n_cols)
    }

    /// Maps a prediction in transformed units back through the target
    /// transform, in place.
    pub(crate) fn inverse_prediction_in_place<P: ModelPrecision>(
        &self,
        prediction: &mut Prediction<P::Refine>,
        buffers: &mut InverseBuffers,
    ) -> Result<(), GprError> {
        P::inverse_mean_variance(
            self.y_transform.as_ref(),
            &mut prediction.mean,
            &mut prediction.variance,
            buffers,
        )
    }

    /// The public covariance from the latent query covariance `latent`
    /// (`q × q`, transformed units) and the diagonal prediction `pred` of
    /// the same queries (transformed units): the off-diagonal from `latent`,
    /// the diagonal from `pred`, both mapped back through the target
    /// transform. The diagonal is exactly `pred`'s variance after that map,
    /// as `predict` returns it.
    pub(crate) fn finish_covariance<P: ModelPrecision, S: KernelScalar>(
        &self,
        latent: MatRef<'_, S>,
        mut pred: Prediction<P::Refine>,
    ) -> Result<PredictiveCovariance<P::Refine>, GprError> {
        let q = pred.mean.len();
        let mut covariance = vec![P::Refine::from_f64(0.0); q * q];
        for col in 0..q {
            for row in 0..q {
                covariance[col * q + row] = if row == col {
                    pred.variance[row]
                } else {
                    P::Refine::from_f64(latent[(row, col)].to_f64())
                };
            }
        }
        P::inverse_covariance(self.y_transform.as_ref(), &mut covariance)?;
        self.inverse_prediction_in_place::<P>(&mut pred, &mut InverseBuffers::default())?;
        for (i, variance) in pred.variance.iter().enumerate() {
            covariance[i * q + i] = *variance;
        }
        Ok(PredictiveCovariance {
            mean: pred.mean,
            covariance,
            variance_kind: pred.variance_kind,
        })
    }

    /// The trainer settings this model was fitted with.
    pub(crate) fn spec(&self) -> SparseSpec<U> {
        SparseSpec {
            kernel: self.kernel.clone(),
            likelihood: self.likelihood,
            math: self.math,
            jitter: self.jitter,
            x_transform: self.x_unfitted.clone_box(),
            y_transform: self.y_unfitted.clone_box(),
        }
    }
}

pub(super) fn theta_len<U: Supply>(
    kernel: &KernelSpec<U>,
    likelihood: &GaussianLikelihood,
) -> usize {
    kernel.num_params() + likelihood.num_params()
}

pub(super) fn stage_theta<U: Supply>(
    kernel: &KernelSpec<U>,
    likelihood: &GaussianLikelihood,
    params: &[f64],
) -> Result<(KernelSpec<U>, GaussianLikelihood), GprError> {
    let n_kernel = kernel.num_params();
    let n_theta = theta_len(kernel, likelihood);
    crate::data::require_count(params.len(), n_theta, "parameters")?;
    let mut kernel = kernel.clone();
    kernel.set_params(&params[..n_kernel])?;
    let mut likelihood = *likelihood;
    likelihood.set_params(&params[n_kernel..])?;
    Ok((kernel, likelihood))
}

/// Default retries for factoring `K_mm = k(Z, Z)`:
/// `adaptive(1e-8, 10, 5, 1e-3)`. Observation noise is not on `K_mm`
/// (design §4.0), so close inducing points need a small diagonal offset;
/// the Exact default (no retry) would fail there.
pub(crate) fn default_k_mm_jitter() -> JitterPolicy {
    JitterPolicy::Adaptive(AdaptiveJitter::K_MM_DEFAULT)
}
