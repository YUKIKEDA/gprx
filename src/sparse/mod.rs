//! Crate-private pieces shared by the sparse models ([`crate::Sgpr`] and
//! [`crate::Svgp`]): the settings every trainer holds, the training data
//! every fitted model holds, and the kernel + likelihood `θ` over both.

use std::fmt;

use dyn_stack::MemBuffer;
use faer::{Mat, MatMut, MatRef};

use crate::data::{validate_inducing, validate_query, validate_training};
use crate::error::GprError;
use crate::kernel::{
    CompiledKernel, CrossViews, DiagAccum, GramInputs, KernelScalar, NoSupply, Triangle,
    WeightedWalk,
};
use crate::kernel::{KernelSpec, RectStore, Supply, TrainSources};
use crate::likelihood::GaussianLikelihood;
use crate::param::{Interval, write_params};
use crate::policy::KernelExp;
use crate::policy::{AdaptiveJitter, JitterPolicy};
use crate::precision::{InverseBuffers, ModelPrecision};
use crate::prediction::{Prediction, PredictiveCovariance};
use crate::transform::{
    IdentityInput, IdentityTarget, TargetTransform, Transform, UnfittedTarget, UnfittedTransform,
};

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
pub(crate) struct PersistedSparse {
    pub(crate) spec: SparseSpec,
    pub(crate) x_transform: Box<dyn Transform>,
    pub(crate) y_transform: Box<dyn TargetTransform>,
    pub(crate) x_obs: Vec<f64>,
    pub(crate) y_obs: Vec<f64>,
    pub(crate) z_obs: Vec<f64>,
    pub(crate) z_train: Vec<f64>,
    pub(crate) n: usize,
    pub(crate) m: usize,
    pub(crate) d: usize,
}

/// The transformed training data a sparse model's kernel and objective
/// read: `x` (`n × d`, column-major; `d` is `0` for a kernel on supplied
/// distances alone), `y`, the inducing points `z` (`m × d`), and the
/// supplied `d²` (empty for a coordinate kernel).
#[derive(Clone, Copy)]
pub(crate) struct SparseData<'a> {
    pub(crate) x: &'a [f64],
    pub(crate) n: usize,
    pub(crate) d: usize,
    pub(crate) y: &'a [f64],
    pub(crate) z: &'a [f64],
    pub(crate) m: usize,
    pub(crate) supply: &'a SparseSupply,
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
        if self.d == 0 && !self.supply.inducing.is_empty() {
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
        if self.d == 0 && !self.supply.inducing.is_empty() {
            crate::data::require_nonempty(self.m)?;
            return crate::data::require_count(self.z.len(), 0, "inducing feature values");
        }
        validate_inducing(self.z, self.m, self.d)
    }
}

/// The two point sets a sparse kernel reads at `T`: the training points
/// `x` (`n × d`) and the inducing points `z` (`m × d`), with the supplied
/// `d²` among the inducing points (`zz`) and from them to the training
/// points (`zx`, `m × n`).
pub(crate) struct SparseSets<'a, T: KernelScalar, U: Supply> {
    pub(crate) x: MatRef<'a, T>,
    pub(crate) z: MatRef<'a, T>,
    pub(crate) zz: U::Squares<'a, T>,
    pub(crate) zx: U::Rects<'a, T>,
}

impl<T: KernelScalar, U: Supply> Clone for SparseSets<'_, T, U> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: KernelScalar, U: Supply> Copy for SparseSets<'_, T, U> {}

impl<'a, T: KernelScalar, U: Supply> SparseSets<'a, T, U> {
    /// The sets `x` and `z` with the supply `at`.
    pub(crate) fn new(x: MatRef<'a, T>, z: MatRef<'a, T>, at: &'a SupplyAt<T>) -> Self {
        Self {
            x,
            z,
            zz: U::squares(&at.zz),
            zx: U::rects(&at.zx),
        }
    }

    /// The inputs of `K_mm = k(Z, Z)`.
    pub(crate) fn k_mm(&self) -> GramInputs<'a, T, U> {
        GramInputs::supplied(self.z, self.zz)
    }

    /// The views of `K(Z, X)` (`m × n`).
    pub(crate) fn k_mn(&self) -> CrossViews<'a, T, U> {
        CrossViews {
            x1: self.z,
            x2: self.x,
            dist: None,
            slots: self.zx,
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
    /// The supplied `d²` the kernel reads (empty for a coordinate kernel).
    pub(crate) supply: SparseSupply,
    /// The kernel's slots, in the order of
    /// [`crate::kernel::DistanceKernel::slots`] (none for a coordinate kernel).
    pub(crate) slots: Vec<crate::kernel::DistanceSlot>,
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
            supply: self.supply.clone(),
            slots: self.slots.clone(),
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
            supply: SparseSupply::default(),
            slots: Vec::new(),
        })
    }

    /// A fitted core read back from a persist directory. The fitted
    /// transforms are the saved ones, not fitted again: an online model's
    /// were fitted on its first training set. `X` and `y` go through them;
    /// `z_train` is the saved transformed `Z`, so a moved `Z` is not mapped
    /// back and forth.
    ///
    /// # Errors
    ///
    /// Returns the input errors of [`crate::data::validate_training`] and
    /// [`crate::data::validate_inducing`], or the error of a transform map.
    pub(crate) fn from_persisted(parts: PersistedSparse) -> Result<Self, GprError> {
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
        } = parts;
        validate_training(&x_obs, n, d, &y_obs)?;
        validate_inducing(&z_obs, m, d)?;
        validate_inducing(&z_train, m, d)?;
        let mut x_train = x_obs.clone();
        x_transform.apply(&mut x_train, n, d)?;
        let mut y_train = y_obs.clone();
        y_transform.transform(&mut y_train)?;
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
            supply: SparseSupply::default(),
            slots: Vec::new(),
        })
    }
}

impl<U: Supply> SparseCore<U> {
    /// The core of a model on supplied distances: `sources` bind each
    /// slot's `n × m` block from the `n` training points to the inducing
    /// points, which are the training points `inducing` (in that order).
    /// `x` (`n × n_cols`, column-major) holds the coordinates of the
    /// coordinate leaves, empty with `n_cols = 0` for a kernel on supplied
    /// distances alone; the inducing points' coordinates are their rows.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::EmptyInput`] if `n` or `inducing` is empty,
    /// [`GprError::IndexOutOfRange`] for an index past `n`,
    /// [`GprError::InvalidConfig`] for an index listed twice, the errors of
    /// [`crate::kernel::bind_inducing`], and the input errors of
    /// [`Self::prepare`].
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
        check_inducing(inducing, n)?;
        let m = inducing.len();
        if n_cols == 0 {
            crate::data::require_count(x.len(), 0, "feature values")?;
            crate::data::require_count(y.len(), n, "targets")?;
            crate::data::require_finite(y)?;
        } else {
            validate_training(x, n, n_cols, y)?;
        }
        let slots = crate::kernel::spec_slots(&spec.kernel);
        let (zz, zx) = crate::kernel::bind_inducing(&slots, sources, n, inducing)?;
        let supply = SparseSupply::new::<S>(inducing.to_vec(), zz, zx)?;
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
            supply,
            slots,
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
            supply: &self.supply,
        }
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

fn theta_len<U: Supply>(kernel: &KernelSpec<U>, likelihood: &GaussianLikelihood) -> usize {
    kernel.num_params() + likelihood.num_params()
}

fn stage_theta<U: Supply>(
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

/// Public read accessors of a fitted sparse model, from its `core:
/// SparseCore` field (or the field path given, such as `state.core`). One
/// set of docs for [`crate::FittedSgpr`], [`crate::OnlineSgpr`], and
/// [`crate::FittedSvgp`], whatever their kernel.
macro_rules! sparse_core_accessors {
    () => {
        $crate::sparse::sparse_core_accessors!(core);
    };
    ($($core:ident).+) => {
        /// Returns the number of training points.
        pub fn n(&self) -> usize {
            self.$($core).+.n
        }

        /// Returns the number of inducing points.
        pub fn m(&self) -> usize {
            self.$($core).+.m
        }

        /// Returns the observation-noise model.
        pub fn likelihood(&self) -> &$crate::GaussianLikelihood {
            &self.$($core).+.likelihood
        }

        /// Returns the jitter retries used when `K_mm` fails to factor.
        pub fn jitter_policy(&self) -> $crate::JitterPolicy {
            self.$($core).+.jitter
        }

        /// Returns the kernel `exp` mode the trainer set with `with_math`.
        pub fn math(&self) -> $crate::KernelExp {
            self.$($core).+.math
        }

        /// Returns the original training targets.
        pub fn y(&self) -> &[f64] {
            &self.$($core).+.y_obs
        }
    };
}

/// The coordinate accessors of a fitted sparse model whose kernel reads
/// coordinates ([`crate::kernel::PointKernel`]).
macro_rules! sparse_point_accessors {
    () => {
        $crate::sparse::sparse_point_accessors!(core);
    };
    ($($core:ident).+) => {
        /// Returns the feature dimension.
        pub fn d(&self) -> usize {
            self.$($core).+.d
        }

        /// Returns the original training features in column-major order.
        pub fn x(&self) -> &[f64] {
            &self.$($core).+.x_obs
        }

        /// Returns the inducing features in column-major order, in the original coordinates of `X`.
        pub fn z(&self) -> &[f64] {
            &self.$($core).+.z_obs
        }
    };
}

/// The kernel accessor of a fitted sparse model on coordinates.
macro_rules! sparse_kernel_accessor {
    () => {
        $crate::sparse::sparse_kernel_accessor!(core);
    };
    ($($core:ident).+) => {
        /// Returns the kernel whose hyperparameters this model owns.
        pub fn kernel(&self) -> &$crate::kernel::KernelSpec {
            &self.$($core).+.kernel
        }
    };
}

pub(crate) use sparse_core_accessors;
pub(crate) use sparse_kernel_accessor;
pub(crate) use sparse_point_accessors;

/// Default retries for factoring `K_mm = k(Z, Z)`:
/// `adaptive(1e-8, 10, 5, 1e-3)`. Observation noise is not on `K_mm`
/// (design §4.0), so close inducing points need a small diagonal offset;
/// the Exact default (no retry) would fail there.
pub(crate) fn default_k_mm_jitter() -> JitterPolicy {
    JitterPolicy::Adaptive(AdaptiveJitter::K_MM_DEFAULT)
}

/// Kernel-evaluation buffers of one sparse operation: the output-shaped
/// scratch, the nested sum / product levels, and the train–query distance
/// block. Every buffer grows to the largest shape asked for and is viewed at
/// the shape of each call, so one operation's kernel calls share them.
/// Scratch: contents mean nothing between calls.
pub(crate) struct KernelScratch<T> {
    scratch: Mat<T>,
    nested: Vec<Mat<T>>,
    dist: Mat<T>,
    /// Exact `m × m` buffers of one square contraction.
    square: Vec<Mat<T>>,
    /// Exact `m × n` buffers of one rectangular contraction.
    cross: Vec<Mat<T>>,
    /// One parameter block, reused by the rectangular and diagonal adds.
    partial: Vec<f64>,
    /// Scratch for the ARD lengthscale matrix products. Grown on the first
    /// call and kept, so a later step does not allocate.
    jobs: Vec<f64>,
    diag: DiagAccum<T>,
}

impl<T> Clone for KernelScratch<T> {
    /// Scratch: a clone starts empty.
    fn clone(&self) -> Self {
        Self {
            scratch: Mat::new(),
            nested: Vec::new(),
            dist: Mat::new(),
            square: Vec::new(),
            cross: Vec::new(),
            partial: Vec::new(),
            jobs: Vec::new(),
            diag: DiagAccum::new(),
        }
    }
}

impl<T> std::fmt::Debug for KernelScratch<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KernelScratch").finish_non_exhaustive()
    }
}

impl<T: KernelScalar> Default for KernelScratch<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: KernelScalar> KernelScratch<T> {
    pub(crate) fn new() -> Self {
        Self {
            scratch: Mat::new(),
            nested: Vec::new(),
            dist: Mat::new(),
            square: Vec::new(),
            cross: Vec::new(),
            partial: Vec::new(),
            jobs: Vec::new(),
            diag: DiagAccum::new(),
        }
    }

    /// `K` for `uplo` into `out`.
    pub(crate) fn gram<M: crate::math::KernelMath, U: Supply>(
        &mut self,
        compiled: &CompiledKernel<T, U>,
        inputs: GramInputs<'_, T, U>,
        out: MatMut<'_, T>,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        let scratch = view(&mut self.scratch, out.nrows(), out.ncols());
        compiled.eval_gram::<M>(inputs, out, uplo, scratch, &mut self.nested)
    }

    /// `∂K/∂θ_{param_idx}` for `uplo` into `d_k`.
    pub(crate) fn grad<M: crate::math::KernelMath, U: Supply>(
        &mut self,
        compiled: &CompiledKernel<T, U>,
        inputs: GramInputs<'_, T, U>,
        d_k: MatMut<'_, T>,
        param_idx: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        let scratch = view(&mut self.scratch, d_k.nrows(), d_k.ncols());
        compiled.grad_gram::<M>(inputs, d_k, param_idx, uplo, scratch, &mut self.nested)
    }

    /// `∂²K/∂θ_i ∂θ_j` for `uplo` into `d2_k`.
    pub(crate) fn hess<M: crate::math::KernelMath, U: Supply>(
        &mut self,
        compiled: &CompiledKernel<T, U>,
        inputs: GramInputs<'_, T, U>,
        d2_k: MatMut<'_, T>,
        pair: (usize, usize),
        uplo: Triangle,
    ) -> Result<(), GprError> {
        let scratch = view(&mut self.scratch, d2_k.nrows(), d2_k.ncols());
        compiled.hess_gram::<M>(inputs, d2_k, pair, uplo, scratch, &mut self.nested)
    }

    /// `K(x, xs)` (`n × q`) into `out`; `cols` holds the supplied `d²`
    /// between the two sets.
    pub(crate) fn cross_into<M: crate::math::KernelMath, U: Supply>(
        &mut self,
        compiled: &CompiledKernel<T, U>,
        views: CrossViews<'_, T, U>,
        out: MatMut<'_, T>,
    ) -> Result<(), GprError> {
        let (rows, cols) = (out.nrows(), out.ncols());
        let dist = view(&mut self.dist, rows, cols);
        let scratch = view(&mut self.scratch, rows, cols);
        compiled.eval_cross_slots::<M>(
            views.x1,
            views.x2,
            views.slots,
            Some(dist),
            out,
            scratch,
            &mut self.nested,
            &mut [],
        )
    }

    /// The rectangular `∂K(x, xs)/∂θ_p` (`n × q`) into `out`.
    pub(crate) fn grad_cross_into<M: crate::math::KernelMath, U: Supply>(
        &mut self,
        compiled: &CompiledKernel<T, U>,
        views: CrossViews<'_, T, U>,
        out: MatMut<'_, T>,
        param_idx: usize,
    ) -> Result<(), GprError> {
        let (rows, cols) = (out.nrows(), out.ncols());
        let KernelScratch {
            scratch, nested, ..
        } = self;
        crate::kernel::ensure_nested_levels(nested, compiled, rows, cols);
        compiled.grad_cross_views::<M>(views, out, param_idx, view(scratch, rows, cols), nested)
    }

    /// The rectangular `∂²K(x, xs)/∂θ_i ∂θ_j` (`n × q`) into `out`, on the
    /// kept scratch and nested levels.
    pub(crate) fn hess_cross_into<M: crate::math::KernelMath, U: Supply>(
        &mut self,
        compiled: &CompiledKernel<T, U>,
        views: CrossViews<'_, T, U>,
        out: MatMut<'_, T>,
        pair: (usize, usize),
    ) -> Result<(), GprError> {
        let (rows, cols) = (out.nrows(), out.ncols());
        let KernelScratch {
            scratch, nested, ..
        } = self;
        crate::kernel::ensure_nested_levels(nested, compiled, rows, cols);
        compiled.hess_cross_views::<M>(views, out, pair, view(scratch, rows, cols), nested)
    }

    /// `K(x, xs)` (`n × q`) in a new matrix.
    pub(crate) fn cross<M: crate::math::KernelMath, U: Supply>(
        &mut self,
        compiled: &CompiledKernel<T, U>,
        views: CrossViews<'_, T, U>,
    ) -> Result<Mat<T>, GprError> {
        let mut out = Mat::zeros(views.x1.nrows(), views.x2.nrows());
        self.cross_into::<M, U>(compiled, views, out.as_mut())?;
        Ok(out)
    }

    /// Writes `⟨weight, ∂K(x, x)/∂θ⟩_F` for every kernel parameter into `out`.
    ///
    /// One walk, keeping no Grams. `out` is replaced. Squared distances of
    /// the coordinates are filled when the tree reads them; ARD leaves fall
    /// back to `x`; supplied leaves read `slots`.
    pub(crate) fn write_square_contraction<M: crate::math::KernelMath, U: Supply>(
        &mut self,
        compiled: &CompiledKernel<T, U>,
        inputs: GramInputs<'_, T, U>,
        weight: MatRef<'_, T>,
        out: &mut [f64],
    ) -> Result<(), GprError> {
        let x = inputs.x;
        let m = x.nrows();
        let reads = compiled.reads_distances()?;
        let KernelScratch {
            scratch,
            nested,
            dist,
            square,
            jobs,
            ..
        } = self;
        if reads {
            let mut filled = view(dist, m, m);
            let mut none: [Mat<T>; 0] = [];
            T::write_squared(x, filled.as_mut(), &mut none);
        }
        let dist_view = reads.then(|| dist.as_ref().submatrix(0, 0, m, m));
        let nbuf = compiled.contraction_buffers();
        fit_exact(square, nbuf, m, m);
        crate::kernel::ensure_nested_levels(nested, compiled, m, m);
        let mut walk = WeightedWalk {
            inputs: GramInputs {
                x,
                dist: dist_view,
                ard: None,
                slots: U::shorter_squares(inputs.slots),
            },
            scratch: view(scratch, m, m),
            nested,
            kept: &[],
            kept_products: 0,
            fold: jobs,
        };
        compiled.weighted_grads::<M>(&mut walk, weight, out, &mut square[..nbuf])
    }

    /// Adds `coeff · ⟨weight, ∂K(x1, x2)/∂θ⟩_F` for every kernel parameter.
    ///
    /// One walk of the full rectangle. A product evaluates each non-constant
    /// factor once.
    pub(crate) fn add_cross_contraction<M: crate::math::KernelMath, U: Supply>(
        &mut self,
        compiled: &CompiledKernel<T, U>,
        views: CrossViews<'_, T, U>,
        weight: MatRef<'_, T>,
        coeff: f64,
        out: &mut [f64],
    ) -> Result<(), GprError> {
        let n_params = compiled.num_params();
        let (rows, cols) = (weight.nrows(), weight.ncols());
        let nbuf = compiled.contraction_buffers();
        let KernelScratch {
            scratch,
            nested,
            cross,
            partial,
            jobs,
            ..
        } = self;
        if partial.len() < n_params {
            partial.resize(n_params, 0.0);
        }
        fit_exact(cross, nbuf, rows, cols);
        crate::kernel::ensure_nested_levels(nested, compiled, rows, cols);
        compiled.weighted_cross_grads_views::<M>(
            views,
            weight,
            &mut partial[..n_params],
            &mut cross[..nbuf],
            view(scratch, rows, cols),
            nested,
            jobs,
        )?;
        for (slot, part) in out.iter_mut().zip(partial.iter()) {
            *slot += coeff * part;
        }
        Ok(())
    }

    /// Adds `coeff · Σ_i ∂k(x_i, x_i)/∂θ` for every kernel parameter.
    ///
    /// One walk. A product reads each non-constant factor's diagonal once.
    pub(crate) fn add_diag_contraction<M: crate::math::KernelMath, U: Supply>(
        &mut self,
        compiled: &CompiledKernel<T, U>,
        x: MatRef<'_, T>,
        coeff: f64,
        out: &mut [f64],
    ) -> Result<(), GprError> {
        let n_params = compiled.num_params();
        let KernelScratch { partial, diag, .. } = self;
        if partial.len() < n_params {
            partial.resize(n_params, 0.0);
        }
        compiled.weighted_diag_sums::<M>(x, &mut partial[..n_params], diag)?;
        for (slot, part) in out.iter_mut().zip(partial.iter()) {
            *slot += coeff * part;
        }
        Ok(())
    }

    /// An output-shaped scratch for a kernel call that takes one directly.
    pub(crate) fn scratch(&mut self, rows: usize, cols: usize) -> MatMut<'_, T> {
        view(&mut self.scratch, rows, cols)
    }
}

/// `pool` grown to `count` matrices of exactly `rows × cols`.
///
/// A later call of the same shape allocates nothing. A different shape
/// replaces the matrix: a walk reads `Mat::nrows`, so a larger buffer viewed
/// as a corner would contract the wrong pairs.
fn fit_exact<T: KernelScalar>(pool: &mut Vec<Mat<T>>, count: usize, rows: usize, cols: usize) {
    if pool.len() < count {
        pool.resize_with(count, Mat::new);
    }
    for mat in &mut pool[..count] {
        if mat.nrows() != rows || mat.ncols() != cols {
            *mat = Mat::zeros(rows, cols);
        }
    }
}

/// `buf` grown to at least `rows × cols`, viewed at that shape.
pub(crate) fn view<T: KernelScalar>(buf: &mut Mat<T>, rows: usize, cols: usize) -> MatMut<'_, T> {
    if buf.nrows() < rows || buf.ncols() < cols {
        *buf = Mat::zeros(rows.max(buf.nrows()), cols.max(buf.ncols()));
    }
    buf.as_mut().submatrix_mut(0, 0, rows, cols)
}

/// A compiled kernel at `S`, rebuilt only when the kernel changes.
pub(crate) struct KernelPlan<S: KernelScalar, U: Supply = NoSupply>(
    Option<(KernelSpec<U>, CompiledKernel<S, U>)>,
);

impl<S: KernelScalar, U: Supply> KernelPlan<S, U> {
    /// The plan of `kernel`, compiled unless the kept one is already for it.
    pub(crate) fn get(&mut self, kernel: &KernelSpec<U>) -> &CompiledKernel<S, U> {
        if !matches!(&self.0, Some((spec, _)) if spec == kernel) {
            self.0 = None;
        }
        &self
            .0
            .get_or_insert_with(|| (kernel.clone(), kernel.compile_as::<S>()))
            .1
    }
}

/// Buffers of one sparse prediction at the storage scalar `S`: the kernel
/// scratch, the packed `Z` and queries, `K(Z, X*)` and its solves, and
/// per-query columns.
pub(crate) struct PredictBuffers<S: KernelScalar> {
    pub(crate) kernel: KernelScratch<S>,
    pub(crate) z: Mat<S>,
    pub(crate) query: Mat<S>,
    pub(crate) k_sz: Mat<S>,
    pub(crate) solved: Mat<S>,
    pub(crate) kss: Vec<S>,
    pub(crate) column: Vec<S>,
}

impl<S: KernelScalar> Default for PredictBuffers<S> {
    fn default() -> Self {
        Self {
            kernel: KernelScratch::new(),
            z: Mat::new(),
            query: Mat::new(),
            k_sz: Mat::new(),
            solved: Mat::new(),
            kss: Vec::new(),
            column: Vec::new(),
        }
    }
}

/// Packs `x` (`rows × d`, column-major) into `buf` as `S` and views it.
pub(crate) fn pack_into<'a, S: KernelScalar>(
    buf: &'a mut Mat<S>,
    x: &[f64],
    rows: usize,
    d: usize,
) -> MatMut<'a, S> {
    let mut out = view(buf, rows, d);
    crate::data::pack_storage(x, rows, d, out.as_mut());
    out
}

/// Buffers of a sparse `predict_into`, kept on the fitted model: the query
/// through the input transform, the storage-scalar buffers, and the `f64`
/// ones a rounding storage predicts through (`K_mm` factored in `f64`, `B`
/// and the weights promoted).
pub(crate) struct PredictScratch<S: KernelScalar, U: Supply = NoSupply> {
    pub(crate) xs: Vec<f64>,
    pub(crate) inverse: InverseBuffers,
    pub(crate) plan: KernelPlan<S, U>,
    pub(crate) storage: PredictBuffers<S>,
    pub(crate) plan64: KernelPlan<f64, U>,
    pub(crate) f64: PredictBuffers<f64>,
    pub(crate) k_mm64: Mat<f64>,
    pub(crate) k_mm64_backup: Mat<f64>,
    pub(crate) llt64: Option<(usize, MemBuffer)>,
    pub(crate) b_l64: Mat<f64>,
    pub(crate) w64: Vec<f64>,
    /// The bound blocks of a prediction on supplied distances, at `f64`
    /// and at `S`.
    pub(crate) query64: crate::kernel::QueryScratch<f64>,
    pub(crate) query_storage: crate::kernel::QueryScratch<S>,
}

impl<S: KernelScalar, U: Supply> Default for PredictScratch<S, U> {
    fn default() -> Self {
        Self {
            xs: Vec::new(),
            inverse: InverseBuffers::default(),
            plan: KernelPlan(None),
            storage: PredictBuffers::default(),
            plan64: KernelPlan(None),
            f64: PredictBuffers::default(),
            k_mm64: Mat::new(),
            k_mm64_backup: Mat::new(),
            llt64: None,
            b_l64: Mat::new(),
            w64: Vec::new(),
            query64: crate::kernel::QueryScratch::new(),
            query_storage: crate::kernel::QueryScratch::new(),
        }
    }
}

/// A clone starts with empty buffers.
impl<S: KernelScalar, U: Supply> Clone for PredictScratch<S, U> {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl<S: KernelScalar, U: Supply> fmt::Debug for PredictScratch<S, U> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PredictScratch").finish_non_exhaustive()
    }
}

impl<S: KernelScalar, U: Supply> PredictScratch<S, U> {
    /// faer scratch for an `m × m` `f64` LLT, reused while `m` is unchanged.
    pub(crate) fn llt64(llt: &mut Option<(usize, MemBuffer)>, m: usize) -> &mut MemBuffer {
        if matches!(llt, Some((size, _)) if *size != m) {
            *llt = None;
        }
        &mut llt
            .get_or_insert_with(|| (m, crate::linalg::llt_scratch::<f64>(m)))
            .1
    }

    /// The `f64` system a rounding storage predicts through, shared by every
    /// sparse model: `kernel` compiled in `f64`, and `K_mm = k(Z, Z)`
    /// evaluated (from `zz`, the inducing points' supplied squares) and
    /// factored in `f64` with the `K_mm` retries of `jitter`.
    ///
    /// # Errors
    ///
    /// Returns the kernel's evaluation errors, or
    /// [`GprError::CholeskyFailed`] when `K_mm` does not factor in `f64`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn f64_system<'a, M: crate::math::KernelMath>(
        &'a mut self,
        kernel: &KernelSpec<U>,
        z: &[f64],
        m: usize,
        d: usize,
        zz: &'a TrainSources<f64>,
        jitter: JitterPolicy,
    ) -> Result<F64System<'a, U>, GprError> {
        let Self {
            plan64,
            f64: bufs,
            k_mm64,
            k_mm64_backup,
            llt64,
            b_l64,
            w64,
            ..
        } = self;
        let compiled = plan64.get(kernel);
        if k_mm64.nrows() != m || k_mm64.ncols() != m {
            *k_mm64 = Mat::zeros(m, m);
        }
        let z64 = pack_into(&mut bufs.z, z, m, d);
        bufs.kernel.gram::<M, U>(
            compiled,
            GramInputs::supplied(z64.as_ref(), U::squares(zz)),
            k_mm64.as_mut(),
            Triangle::Lower,
        )?;
        crate::linalg::cholesky_lower_with_backup(
            k_mm64,
            k_mm64_backup,
            Self::llt64(llt64, m),
            jitter.retry_jitters(),
            crate::error::CholeskyStage::Predict,
        )?;
        Ok(F64System {
            compiled,
            bufs,
            k_mm_l: k_mm64.as_ref(),
            b_l64,
            w64,
        })
    }
}

/// The `f64` system of [`PredictScratch::f64_system`], with the buffers a
/// model promotes its own factors into (`B`'s factor and the weights of a VFE
/// prediction).
pub(crate) struct F64System<'a, U: Supply = NoSupply> {
    pub(crate) compiled: &'a CompiledKernel<f64, U>,
    pub(crate) bufs: &'a mut PredictBuffers<f64>,
    pub(crate) k_mm_l: MatRef<'a, f64>,
    pub(crate) b_l64: &'a mut Mat<f64>,
    pub(crate) w64: &'a mut Vec<f64>,
}

/// Clears `out` to `n_rows` zero means and variances of `kind`.
pub(crate) fn reset_prediction<R: KernelScalar>(
    out: &mut Prediction<R>,
    n_rows: usize,
    kind: crate::VarianceKind,
) {
    let zero = R::from_f64(0.0);
    out.mean.clear();
    out.mean.resize(n_rows, zero);
    out.variance.clear();
    out.variance.resize(n_rows, zero);
    out.variance_kind = kind;
}

/// Latent or observation variance from the clamped latent variance.
pub(crate) fn predictive_variance(latent: f64, noise: f64, kind: crate::VarianceKind) -> f64 {
    match kind {
        crate::VarianceKind::Latent => latent,
        crate::VarianceKind::Observation => latent + noise,
    }
}

/// Kernel scratch a fitted sparse model keeps between its `&mut self` calls
/// (`set_params`, gradient, Hessian, online updates): one for the storage
/// scalar `S`, one for the `f64` assembly a rounding precision starts from.
#[derive(Clone, Debug)]
pub(crate) struct SparseScratch<S: KernelScalar, U: Supply = NoSupply> {
    pub(crate) storage: KernelScratch<S>,
    pub(crate) f64: KernelScratch<f64>,
    /// One point through the input transform (online inserts).
    pub(crate) point: Vec<f64>,
    /// Buffers of `predict_into`.
    pub(crate) predict: PredictScratch<S, U>,
}

impl<S: KernelScalar, U: Supply> Default for SparseScratch<S, U> {
    fn default() -> Self {
        Self {
            storage: KernelScratch::new(),
            f64: KernelScratch::new(),
            point: Vec::new(),
            predict: PredictScratch::default(),
        }
    }
}

/// The training `d²` of a sparse model on supplied distances, at `f64` and,
/// for an `f32` storage, cast once: the `m × m` squares among the inducing
/// points and the `m × n` blocks from them to the training points. Empty
/// for a coordinate kernel.
#[derive(Clone, Debug, Default)]
pub(crate) struct SparseSupply {
    /// The training points that are the inducing points, in order.
    pub(crate) inducing: Vec<usize>,
    f64: SupplyAt<f64>,
    f32: SupplyAt<f32>,
}

/// Checks the inducing indices of a model on supplied distances: each
/// below `n`, none twice.
fn check_inducing(inducing: &[usize], n: usize) -> Result<(), GprError> {
    let mut seen = vec![false; n];
    for &i in inducing {
        let slot = seen.get_mut(i).ok_or_else(|| GprError::IndexOutOfRange {
            reason: format!("inducing index {i} is not below the {n} training points"),
        })?;
        if *slot {
            return Err(GprError::InvalidConfig {
                reason: format!("inducing index {i} is listed twice"),
            });
        }
        *slot = true;
    }
    Ok(())
}

/// The supplied `d²` of [`SparseSupply`] at one scalar.
#[derive(Clone, Debug, Default)]
pub(crate) struct SupplyAt<T: KernelScalar> {
    /// `m × m` among the inducing points.
    pub(crate) zz: TrainSources<T>,
    /// `m × n` from the inducing points to the training points.
    pub(crate) zx: RectStore<T>,
}

impl SparseSupply {
    /// The supply of `inducing` from its `f64` squares and blocks, cast
    /// once for an `f32` storage `S`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidDistance`] for a value past the range of
    /// `S`.
    pub(crate) fn new<S: KernelScalar>(
        inducing: Vec<usize>,
        zz: TrainSources<f64>,
        zx: RectStore<f64>,
    ) -> Result<Self, GprError> {
        let f32 = if std::any::TypeId::of::<S>() == std::any::TypeId::of::<f32>() {
            SupplyAt {
                zz: zz.cast()?,
                zx: zx.cast()?,
            }
        } else {
            SupplyAt::default()
        };
        Ok(Self {
            inducing,
            f64: SupplyAt { zz, zx },
            f32,
        })
    }

    /// The supply at `T` (`f64`, or the `f32` cast).
    ///
    /// # Errors
    ///
    /// The supply holds `f64` and `f32` only; any other `T` is reported as
    /// unbound.
    pub(crate) fn at<T: KernelScalar>(&self) -> Result<&SupplyAt<T>, GprError> {
        let f64: &dyn std::any::Any = &self.f64;
        let f32: &dyn std::any::Any = &self.f32;
        f64.downcast_ref()
            .or_else(|| f32.downcast_ref())
            .ok_or_else(crate::kernel::unbound)
    }
}
