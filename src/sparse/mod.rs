//! Crate-private pieces shared by the sparse models ([`crate::Sgpr`] and
//! [`crate::Svgp`]): the settings every trainer holds, the training data
//! every fitted model holds, and the kernel + likelihood `θ` over both.

use std::fmt;

use dyn_stack::MemBuffer;
use faer::{Mat, MatMut, MatRef};

use crate::data::{validate_inducing, validate_query, validate_training};
use crate::error::GprError;
use crate::kernel::KernelSpec;
use crate::kernel::{CompiledKernel, GramInputs, KernelScalar, Triangle};
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
pub(crate) struct SparseSpec {
    pub(crate) kernel: KernelSpec,
    pub(crate) likelihood: GaussianLikelihood,
    pub(crate) math: KernelExp,
    /// Retries for factoring `K_mm`.
    pub(crate) jitter: JitterPolicy,
    pub(crate) x_transform: Box<dyn UnfittedTransform>,
    pub(crate) y_transform: Box<dyn UnfittedTarget>,
}

impl Clone for SparseSpec {
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

impl fmt::Debug for SparseSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SparseSpec")
            .field("kernel", &self.kernel)
            .field("likelihood", &self.likelihood)
            .field("math", &self.math)
            .field("jitter", &self.jitter)
            .finish_non_exhaustive()
    }
}

impl SparseSpec {
    pub(crate) fn new(kernel: KernelSpec, likelihood: GaussianLikelihood) -> Self {
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

/// Training data and settings of a fitted sparse model.
///
/// `x_obs` / `z_obs` / `y_obs` are what the caller passed (column-major
/// `n × d` and `m × d`). `x_train` / `z_train` / `y_train` are the same data through the
/// fitted transforms; every factor, gradient, and prediction reads those.
pub(crate) struct SparseCore {
    pub(crate) kernel: KernelSpec,
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
}

impl Clone for SparseCore {
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
        }
    }
}

impl fmt::Debug for SparseCore {
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
        })
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
    ) -> Result<(KernelSpec, GaussianLikelihood), GprError> {
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
        validate_query(xs, n_rows, n_cols)?;
        out.clear();
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
    pub(crate) fn spec(&self) -> SparseSpec {
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

fn theta_len(kernel: &KernelSpec, likelihood: &GaussianLikelihood) -> usize {
    kernel.num_params() + likelihood.num_params()
}

fn stage_theta(
    kernel: &KernelSpec,
    likelihood: &GaussianLikelihood,
    params: &[f64],
) -> Result<(KernelSpec, GaussianLikelihood), GprError> {
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
/// SparseCore` field (or the field path given, such as `state.core`). One set of docs for [`crate::FittedSgpr`],
/// [`crate::OnlineSgpr`], and [`crate::FittedSvgp`].
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

        /// Returns the feature dimension.
        pub fn d(&self) -> usize {
            self.$($core).+.d
        }

        /// Returns the kernel whose hyperparameters this model owns.
        pub fn kernel(&self) -> &$crate::kernel::KernelSpec {
            &self.$($core).+.kernel
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

        /// Returns the original training features in column-major order.
        pub fn x(&self) -> &[f64] {
            &self.$($core).+.x_obs
        }

        /// Returns the inducing features in column-major order, in the
        /// original coordinates of `X`.
        pub fn z(&self) -> &[f64] {
            &self.$($core).+.z_obs
        }

        /// Returns the original training targets.
        pub fn y(&self) -> &[f64] {
            &self.$($core).+.y_obs
        }
    };
}

pub(crate) use sparse_core_accessors;

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
}

impl<T> Clone for KernelScratch<T> {
    /// Scratch: a clone starts empty.
    fn clone(&self) -> Self {
        Self {
            scratch: Mat::new(),
            nested: Vec::new(),
            dist: Mat::new(),
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
        }
    }

    /// `K` for `uplo` into `out`.
    pub(crate) fn gram<M: crate::math::KernelMath>(
        &mut self,
        compiled: &CompiledKernel<T>,
        inputs: GramInputs<'_, T>,
        out: MatMut<'_, T>,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        let scratch = view(&mut self.scratch, out.nrows(), out.ncols());
        compiled.eval_gram::<M>(inputs, out, uplo, scratch, &mut self.nested)
    }

    /// `∂K/∂θ_{param_idx}` for `uplo` into `d_k`.
    pub(crate) fn grad<M: crate::math::KernelMath>(
        &mut self,
        compiled: &CompiledKernel<T>,
        inputs: GramInputs<'_, T>,
        d_k: MatMut<'_, T>,
        param_idx: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        let scratch = view(&mut self.scratch, d_k.nrows(), d_k.ncols());
        compiled.grad_gram::<M>(inputs, d_k, param_idx, uplo, scratch, &mut self.nested)
    }

    /// `∂²K/∂θ_i ∂θ_j` for `uplo` into `d2_k`.
    pub(crate) fn hess<M: crate::math::KernelMath>(
        &mut self,
        compiled: &CompiledKernel<T>,
        inputs: GramInputs<'_, T>,
        d2_k: MatMut<'_, T>,
        pair: (usize, usize),
        uplo: Triangle,
    ) -> Result<(), GprError> {
        let scratch = view(&mut self.scratch, d2_k.nrows(), d2_k.ncols());
        compiled.hess_gram::<M>(inputs, d2_k, pair, uplo, scratch, &mut self.nested)
    }

    /// `K(x, xs)` (`n × q`) into `out`.
    pub(crate) fn cross_into<M: crate::math::KernelMath>(
        &mut self,
        compiled: &CompiledKernel<T>,
        x: MatRef<'_, T>,
        xs: MatRef<'_, T>,
        out: MatMut<'_, T>,
    ) -> Result<(), GprError> {
        let (rows, cols) = (out.nrows(), out.ncols());
        let dist = view(&mut self.dist, rows, cols);
        let scratch = view(&mut self.scratch, rows, cols);
        compiled.eval_cross::<M>(x, xs, Some(dist), out, scratch, &mut self.nested, &mut [])
    }

    /// `K(x, xs)` (`n × q`) in a new matrix.
    pub(crate) fn cross<M: crate::math::KernelMath>(
        &mut self,
        compiled: &CompiledKernel<T>,
        x: MatRef<'_, T>,
        xs: MatRef<'_, T>,
    ) -> Result<Mat<T>, GprError> {
        let mut out = Mat::zeros(x.nrows(), xs.nrows());
        self.cross_into::<M>(compiled, x, xs, out.as_mut())?;
        Ok(out)
    }

    /// `∂K(x1, x2)/∂θ_{param_idx}` (`x1.nrows() × x2.nrows()`) into `d_k`,
    /// with this scratch's output-shaped buffer and nesting levels.
    pub(crate) fn grad_cross<M: crate::math::KernelMath>(
        &mut self,
        compiled: &CompiledKernel<T>,
        x1: MatRef<'_, T>,
        x2: MatRef<'_, T>,
        d_k: MatMut<'_, T>,
        param_idx: usize,
    ) -> Result<(), GprError> {
        let (rows, cols) = (d_k.nrows(), d_k.ncols());
        crate::kernel::ensure_nested_levels(&mut self.nested, compiled, rows, cols);
        let scratch = view(&mut self.scratch, rows, cols);
        compiled.grad_cross_points_with::<M>(x1, x2, d_k, param_idx, scratch, &mut self.nested)
    }

    /// An output-shaped scratch for a kernel call that takes one directly.
    pub(crate) fn scratch(&mut self, rows: usize, cols: usize) -> MatMut<'_, T> {
        view(&mut self.scratch, rows, cols)
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
pub(crate) struct KernelPlan<S: KernelScalar>(Option<(KernelSpec, CompiledKernel<S>)>);

impl<S: KernelScalar> KernelPlan<S> {
    /// The plan of `kernel`, compiled unless the kept one is already for it.
    pub(crate) fn get(&mut self, kernel: &KernelSpec) -> &CompiledKernel<S> {
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
pub(crate) struct PredictScratch<S: KernelScalar> {
    pub(crate) xs: Vec<f64>,
    pub(crate) inverse: InverseBuffers,
    pub(crate) plan: KernelPlan<S>,
    pub(crate) storage: PredictBuffers<S>,
    pub(crate) plan64: KernelPlan<f64>,
    pub(crate) f64: PredictBuffers<f64>,
    pub(crate) k_mm64: Mat<f64>,
    pub(crate) k_mm64_backup: Mat<f64>,
    pub(crate) llt64: Option<(usize, MemBuffer)>,
    pub(crate) b_l64: Mat<f64>,
    pub(crate) w64: Vec<f64>,
    /// `K(Z, X*)` in `f64`, the reference of a mixed SVGP mean.
    pub(crate) k_zs64: Mat<f64>,
}

impl<S: KernelScalar> Default for PredictScratch<S> {
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
            k_zs64: Mat::new(),
        }
    }
}

/// A clone starts with empty buffers.
impl<S: KernelScalar> Clone for PredictScratch<S> {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl<S: KernelScalar> fmt::Debug for PredictScratch<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PredictScratch").finish_non_exhaustive()
    }
}

impl<S: KernelScalar> PredictScratch<S> {
    /// faer scratch for an `m × m` `f64` LLT, reused while `m` is unchanged.
    pub(crate) fn llt64(llt: &mut Option<(usize, MemBuffer)>, m: usize) -> &mut MemBuffer {
        if matches!(llt, Some((size, _)) if *size != m) {
            *llt = None;
        }
        &mut llt
            .get_or_insert_with(|| (m, crate::linalg::llt_scratch::<f64>(m)))
            .1
    }
}

/// Kernel scratch a fitted sparse model keeps between its `&mut self` calls
/// (`set_params`, gradient, Hessian, online updates): one for the storage
/// scalar `S`, one for the `f64` assembly a rounding precision starts from.
#[derive(Clone, Debug, Default)]
pub(crate) struct SparseScratch<S: KernelScalar> {
    pub(crate) storage: KernelScratch<S>,
    pub(crate) f64: KernelScratch<f64>,
    /// One point through the input transform (online inserts).
    pub(crate) point: Vec<f64>,
    /// Buffers of `predict_into`.
    pub(crate) predict: PredictScratch<S>,
}
