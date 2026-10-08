//! State and read paths shared by [`crate::FittedGpr`] and [`crate::OnlineGpr`].
//!
//! [`GprCore`] holds everything except the training factor. Prediction,
//! covariance, sampling, leave-one-out, the marginal likelihood, and the
//! predict `α` are written once here against a [`StoredFactor`] view, which
//! is the LLT of a batch fit or the LDLT of an online model.

use dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt;
use faer::{Mat, MatMut, MatRef};

use super::factor::TrainPoints;
use crate::data::{pack_storage, validate_query};
use crate::error::{CholeskyStage, GprError};
use crate::kernel::{
    CompiledKernel, CompiledOf, DistanceSlot, DistanceSource, GramInputs, KernelScalar, KernelSpec,
    ModelKernel, NoSupply, QueryScratch, ScalarOps, SourceStore, SpecOf, Supply, SupplyViews,
    Triangle,
};
use crate::likelihood::GaussianLikelihood;
use crate::linalg::{
    cholesky_lower, faer_par, faer_par_dims, inv_diag_from_chol_l, log_det_from_l,
};
use crate::param::write_params;
use crate::precision::{GpScalar, InverseBuffers, StoredFactor, TrainSystem};
use crate::transform::{TargetTransform, Transform, UnfittedTarget, UnfittedTransform};
use crate::workspace::{QueryWorkspace, empty_thread_scratch};
use crate::{PredictOptions, Prediction, PredictiveCovariance, VarianceKind};

use super::Gpr;
use crate::policy::{
    CholeskyBuffer, DistanceCachePolicy, JitterPolicy, KernelExp, with_kernel_exp,
};

/// The runtime policies a trainer and its fitted model share.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Policies {
    pub(crate) distance_cache: DistanceCachePolicy,
    pub(crate) cholesky_buffer: CholeskyBuffer,
    pub(crate) math: KernelExp,
    pub(crate) jitter: JitterPolicy,
}

/// Everything a fitted Exact GPR holds except its training factor.
pub(crate) struct GprCore<P: GpScalar, K: ModelKernel> {
    pub(crate) kernel: SpecOf<K>,
    pub(crate) compiled: CompiledOf<P::Storage, K>,
    pub(crate) likelihood: GaussianLikelihood,
    pub(crate) x_unfitted: Box<dyn UnfittedTransform>,
    pub(crate) y_unfitted: Box<dyn UnfittedTarget>,
    pub(crate) x_transform: Box<dyn Transform>,
    pub(crate) y_transform: Box<dyn TargetTransform>,
    pub(crate) policies: Policies,
    pub(crate) query: QueryWorkspace<P>,
    /// Training features on the caller's scale, column-major `n × d`.
    pub(crate) x_obs: Vec<f64>,
    /// Training targets on the caller's scale.
    pub(crate) y_obs: Vec<f64>,
    /// Transformed training features. Rows past `n` are spare online capacity.
    pub(crate) x: Mat<f64>,
    /// Transformed training targets.
    pub(crate) y_train: Vec<f64>,
    /// Factor solve `α` in the storage scalar. The marginal likelihood uses this.
    pub(crate) factor_alpha: Vec<P::Storage>,
    /// Predict weights. [`crate::DoublePrecision`] and [`crate::SinglePrecision`]
    /// copy [`Self::factor_alpha`]. [`crate::MixedPrecision`] stores the refined
    /// `f64` `α`.
    pub(crate) alpha: Vec<P::Refine>,
    pub(crate) x_cast: <P::Storage as ScalarOps>::ColCast,
    pub(crate) y_cast: <P::Storage as ScalarOps>::RowCast,
    /// Training squared distances of a distance model; empty otherwise.
    pub(crate) sources: P::Sources,
    /// The distance slots of `kernel`, in order (empty for a coordinate
    /// kernel). Fixed with the kernel's tree.
    pub(crate) slots: Vec<DistanceSlot>,
    /// Buffers a prediction on supplied distances binds its blocks on.
    pub(crate) query_sources: QueryScratch<P::Storage>,
    pub(crate) n: usize,
    pub(crate) d: usize,
}

impl<P: GpScalar, K: ModelKernel> Clone for GprCore<P, K> {
    fn clone(&self) -> Self {
        Self {
            kernel: self.kernel.clone(),
            compiled: self.compiled.clone(),
            likelihood: self.likelihood,
            x_unfitted: self.x_unfitted.clone_box(),
            y_unfitted: self.y_unfitted.clone_box(),
            x_transform: self.x_transform.clone_box(),
            y_transform: self.y_transform.clone_box(),
            policies: self.policies,
            query: self.query.clone(),
            x_obs: self.x_obs.clone(),
            y_obs: self.y_obs.clone(),
            x: self.x.clone(),
            y_train: self.y_train.clone(),
            factor_alpha: self.factor_alpha.clone(),
            alpha: self.alpha.clone(),
            x_cast: self.x_cast.clone(),
            y_cast: self.y_cast.clone(),
            sources: self.sources.clone(),
            slots: self.slots.clone(),
            query_sources: self.query_sources.clone(),
            n: self.n,
            d: self.d,
        }
    }
}

/// The training points of a core for a Gram evaluation: `x` (live `n × d`)
/// in the storage scalar, and the training squares of a distance model.
pub(crate) fn train_points<'a, P: GpScalar, S: Supply>(
    x: &'a Mat<f64>,
    (n, d): (usize, usize),
    x_cast: &'a mut <P::Storage as ScalarOps>::ColCast,
    sources: &'a P::Sources,
) -> TrainPoints<'a, P::Storage, S> {
    TrainPoints {
        x: P::Storage::storage_cols(x.as_ref().submatrix(0, 0, n, d), x_cast),
        slots: S::squares(sources.storage()),
    }
}

/// Binds the training squares of `slots` (the kernel's), `n × n` each.
pub(crate) fn bind_training<T: KernelScalar, Store: SourceStore<T>>(
    slots: &[DistanceSlot],
    sources: Vec<DistanceSource<'_>>,
    n: usize,
) -> Result<Store, GprError> {
    if slots.is_empty() && sources.is_empty() {
        return Ok(Store::empty());
    }
    Store::bind(slots, sources, n)
}

impl<P: GpScalar, K: ModelKernel> GprCore<P, K> {
    /// Transformed training features of the live points (`n × d`).
    pub(crate) fn x_active(&self) -> MatRef<'_, f64> {
        self.x.as_ref().submatrix(0, 0, self.n, self.d)
    }

    /// Drops the training data and returns a trainer with `optimizer`.
    pub(crate) fn into_trainer<O>(self, optimizer: O) -> Gpr<O, P, K> {
        Gpr::from_owned(
            self.kernel,
            self.likelihood,
            self.x_unfitted,
            self.y_unfitted,
            optimizer,
            self.policies,
        )
    }

    pub(crate) fn num_params(&self) -> usize {
        self.kernel.num_params() + self.likelihood.num_params()
    }

    pub(crate) fn get_params(&self, out: &mut [f64]) -> Result<(), GprError> {
        write_params(&self.kernel, &self.likelihood, out)
    }

    /// Returns `factor⁻¹ y` in the storage scalar.
    pub(crate) fn solve_factor_alpha(
        &self,
        factor: StoredFactor<'_, P::Storage>,
    ) -> Vec<P::Storage> {
        let n = self.n;
        let mut rhs =
            Mat::<P::Storage>::from_fn(n, 1, |i, _| P::Storage::from_f64(self.y_train[i]));
        factor.solve_in_place(rhs.as_mut());
        (0..n).map(|i| rhs[(i, 0)]).collect()
    }

    /// Writes the predict `α` from `factor` and [`Self::factor_alpha`].
    pub(crate) fn publish_predict_alpha(
        &mut self,
        factor: StoredFactor<'_, P::Storage>,
        jitter: f64,
        stage: CholeskyStage,
    ) -> Result<(), GprError> {
        let mut alpha = std::mem::take(&mut self.alpha);
        let written =
            self.write_predict_alpha(factor, &self.factor_alpha, jitter, stage, &mut alpha);
        self.alpha = alpha;
        written
    }

    /// Writes the predict `α` for `factor_alpha = factor⁻¹ y` into `out`.
    pub(crate) fn write_predict_alpha(
        &self,
        factor: StoredFactor<'_, P::Storage>,
        factor_alpha: &[P::Storage],
        jitter: f64,
        stage: CholeskyStage,
        out: &mut Vec<P::Refine>,
    ) -> Result<(), GprError> {
        let sys = TrainSystem {
            kernel: &self.kernel,
            compiled: &self.compiled,
            x: self.x_active(),
            sources: self.sources.storage(),
            exact: self.sources.exact(),
            y: &self.y_train,
            noise: self.likelihood.noise_variance(),
            jitter,
            factor,
            factor_alpha,
            policy: self.policies.jitter,
            stage,
        };
        with_kernel_exp!(self.policies.math, M => P::publish_predict_alpha::<M, _>(&sys, out))
    }

    /// `½ yᵀ α + ½ log|A| + (n/2) log(2π)` from `factor` and `factor_alpha`.
    pub(crate) fn neg_log_marginal_likelihood(
        &self,
        factor: StoredFactor<'_, P::Storage>,
        factor_alpha: &[P::Storage],
    ) -> f64 {
        let mut rows = P::Storage::empty_rows();
        let y = P::Storage::storage_rows(&self.y_train, &mut rows);
        let mut quad = P::Storage::from_f64(0.0);
        for (yi, ai) in y.iter().zip(factor_alpha).take(self.n) {
            quad += *yi * *ai;
        }
        let log_two_pi = P::Storage::from_f64((2.0 * std::f64::consts::PI).ln());
        (P::Storage::from_f64(0.5)
            * (quad + factor.log_det() + P::Storage::from_f64(self.n as f64) * log_two_pi))
            .to_f64()
    }

    /// Checks the query coordinates: `m × d` against the training `d`. A
    /// model on supplied distances alone has `d = 0` and an empty `xs`.
    fn check_query(&self, q: Query<'_, P::Storage, K::Supply>) -> Result<(), GprError> {
        if q.n_cols != self.d {
            return Err(GprError::DimensionMismatch {
                x_dim: q.n_cols,
                expected_dim: self.d,
            });
        }
        if self.d == 0 {
            crate::data::require_nonempty(q.m)?;
            return crate::data::require_count(q.xs.len(), 0, "feature values");
        }
        validate_query(q.xs, q.m, q.n_cols)
    }

    /// Predicts into `out` through [`Self::query`], reusing its buffers.
    pub(crate) fn predict_with_into(
        &mut self,
        factor: StoredFactor<'_, P::Storage>,
        thread_scratch: &mut [Mat<P::Storage>],
        q: Query<'_, P::Storage, K::Supply>,
        options: PredictOptions,
        out: &mut Prediction<P::Refine>,
    ) -> Result<(), GprError> {
        let mut query = std::mem::replace(&mut self.query, QueryWorkspace::new());
        let mut x_cast = std::mem::replace(&mut self.x_cast, P::Storage::empty_cols());
        let predicted = self.predict_query(
            QueryBuffers {
                query: &mut query,
                x_cast: &mut x_cast,
                thread_scratch,
            },
            factor,
            &self.alpha,
            q,
            options,
            out,
        );
        self.query = query;
        self.x_cast = x_cast;
        predicted
    }

    /// Predicts into `out` with query buffers allocated for this call.
    pub(crate) fn write_prediction(
        &self,
        factor: StoredFactor<'_, P::Storage>,
        alpha: &[P::Refine],
        q: Query<'_, P::Storage, K::Supply>,
        options: PredictOptions,
        out: &mut Prediction<P::Refine>,
    ) -> Result<(), GprError> {
        let mut query = QueryWorkspace::new();
        let mut x_cast = P::Storage::empty_cols();
        let mut thread_scratch = empty_thread_scratch::<P::Storage>();
        self.predict_query(
            QueryBuffers {
                query: &mut query,
                x_cast: &mut x_cast,
                thread_scratch: &mut thread_scratch,
            },
            factor,
            alpha,
            q,
            options,
            out,
        )
    }

    /// The one predict body: [`Self::fill_query`], then the moments.
    fn predict_query(
        &self,
        buffers: QueryBuffers<'_, P>,
        factor: StoredFactor<'_, P::Storage>,
        alpha: &[P::Refine],
        q: Query<'_, P::Storage, K::Supply>,
        options: PredictOptions,
        out: &mut Prediction<P::Refine>,
    ) -> Result<(), GprError> {
        let query = self.fill_query(buffers, q)?;
        let QueryWorkspace {
            query_xs,
            query_x,
            query_k_star,
            query_kss,
            ..
        } = query;
        let m = q.m;
        write_moments::<P, _>(
            MomentInputs {
                core: self.refs(alpha),
                factor,
                query_xs: &query_xs[..m * q.n_cols],
                query_x: query_x.as_ref().submatrix(0, 0, m, q.n_cols),
                k_star: query_k_star.as_mut().submatrix_mut(0, 0, self.n, m),
                kss: &mut query_kss[..m],
                n_cols: q.n_cols,
                cross64: <K::Supply>::shorter_rects(q.cross64),
                options,
            },
            out,
        )
    }

    fn refs<'a>(&'a self, alpha: &'a [P::Refine]) -> CoreRefs<'a, P, K::Supply> {
        CoreRefs {
            kernel: &self.kernel,
            compiled: &self.compiled,
            alpha,
            x_train: self.x_active(),
            noise: self.likelihood.noise_variance(),
            y_transform: self.y_transform.as_ref(),
            math: self.policies.math,
        }
    }

    /// Checks and transforms the query coordinates, packs them into
    /// `query`, and writes `K(X, xs)` (`n × m`) into `query.query_k_star`.
    fn fill_query<'q>(
        &self,
        buffers: QueryBuffers<'q, P>,
        q: Query<'_, P::Storage, K::Supply>,
    ) -> Result<&'q mut QueryWorkspace<P>, GprError> {
        let QueryBuffers {
            query,
            x_cast,
            thread_scratch,
        } = buffers;
        self.check_query(q)?;
        let (m, d) = (q.m, q.n_cols);
        query.ensure(self.n, m, d)?;
        query.query_xs.copy_from_slice(q.xs);
        if d > 0 {
            self.x_transform.apply(&mut query.query_xs, m, d)?;
        }
        let x_train = P::Storage::storage_cols(self.x_active(), x_cast);
        let QueryWorkspace {
            query_xs,
            query_x,
            query_dist,
            query_k_star,
            query_scratch,
            query_nested,
            ..
        } = &mut *query;
        pack_storage(query_xs, m, d, query_x.as_mut());
        with_kernel_exp!(self.policies.math, M => self.compiled.eval_cross_slots::<M>(
            x_train,
            query_x.as_ref(),
            q.cross,
            Some(query_dist.as_mut()),
            query_k_star.as_mut(),
            query_scratch.as_mut(),
            query_nested,
            thread_scratch,
        ))?;
        Ok(query)
    }

    /// Predictive mean and query–query covariance at the query.
    /// `square` holds the query × query squares of every slot, read as a
    /// Gram of one set (nothing for a coordinate kernel).
    pub(crate) fn write_covariance(
        &self,
        factor: StoredFactor<'_, P::Storage>,
        alpha: &[P::Refine],
        q: Query<'_, P::Storage, K::Supply>,
        square: <K::Supply as SupplyViews>::Squares<'_, P::Storage>,
        options: PredictOptions,
    ) -> Result<PredictiveCovariance<P::Refine>, GprError> {
        let n = self.n;
        let m = q.m;
        let mut query = QueryWorkspace::new();
        let mut x_cast = P::Storage::empty_cols();
        let mut thread_scratch = empty_thread_scratch::<P::Storage>();
        self.fill_query(
            QueryBuffers {
                query: &mut query,
                x_cast: &mut x_cast,
                thread_scratch: &mut thread_scratch,
            },
            q,
        )?;
        let QueryWorkspace {
            query_xs,
            query_x,
            query_k_star: mut k_star,
            ..
        } = query;
        let mut mean = vec![P::Refine::from_f64(0.0); m];
        with_kernel_exp!(self.policies.math, M => P::predict_means::<M, _>(
            &self.kernel,
            k_star.as_ref(),
            self.x_active(),
            &query_xs,
            q.n_cols,
            q.cross64,
            alpha,
            &mut mean,
        ))?;
        factor.inv_l_in_place(k_star.as_mut());
        let mut kss = Mat::<P::Storage>::zeros(m, m);
        let mut kss_scratch = Mat::<P::Storage>::zeros(m, m);
        // One set: a `WhiteKernel` term adds its diagonal, as for points.
        with_kernel_exp!(self.policies.math, M => self.compiled.eval_gram_from_points::<M>(
            query_x.as_ref(),
            square,
            kss.as_mut(),
            Triangle::Full,
            kss_scratch.as_mut(),
            &mut Vec::new(),
            &mut thread_scratch,
        ))?;
        let zero_s = P::Storage::from_f64(0.0);
        for col in 0..m {
            for row in 0..m {
                let mut dot = 0.0f64;
                for k in 0..n {
                    dot += factor.scaled_product(
                        k,
                        k_star[(k, row)].to_f64(),
                        k_star[(k, col)].to_f64(),
                    );
                }
                kss[(row, col)] -= P::Storage::from_f64(dot);
            }
        }
        let noise_s = P::Storage::from_f64(self.likelihood.noise_variance());
        for i in 0..m {
            let mut latent = kss[(i, i)];
            if latent.to_f64() < 0.0 {
                latent = zero_s;
            }
            kss[(i, i)] = match options.variance_kind {
                VarianceKind::Latent => latent,
                VarianceKind::Observation => latent + noise_s,
            };
        }
        P::inverse_mean_variance(
            self.y_transform.as_ref(),
            &mut mean,
            &mut [],
            &mut InverseBuffers::default(),
        )?;
        let mut covariance = vec![P::Refine::from_f64(0.0); m * m];
        for col in 0..m {
            for row in 0..m {
                covariance[col * m + row] = P::Refine::from_f64(kss[(row, col)].to_f64());
            }
        }
        P::inverse_covariance(self.y_transform.as_ref(), &mut covariance)?;
        Ok(PredictiveCovariance {
            mean,
            covariance,
            variance_kind: options.variance_kind,
        })
    }

    /// Leave-one-out mean and variance at every training point (GPML §5.4.2).
    pub(crate) fn loo_predict_with(
        &self,
        factor: StoredFactor<'_, P::Storage>,
        alpha: &[P::Refine],
        options: PredictOptions,
    ) -> Result<Prediction<P::Refine>, GprError> {
        if <P::Storage as ScalarOps>::ROUNDS_FROM_F64 {
            return self.loo_from_rounded_kernel(options);
        }
        let mut rows = P::Storage::empty_rows();
        let y = P::Storage::storage_rows(&self.y_train, &mut rows);
        let n = self.n;
        let mut q_diag = vec![P::Storage::from_f64(0.0); n];
        factor.inv_diag(&mut q_diag);
        let noise = self.likelihood.noise_variance();
        let mut mean = vec![P::Refine::from_f64(0.0); n];
        let mut variance = vec![P::Refine::from_f64(0.0); n];
        for i in 0..n {
            let qii = q_diag[i].to_f64();
            if !qii.is_finite() || qii <= 0.0 {
                return Err(GprError::NonPositiveDefiniteMatrix);
            }
            mean[i] = P::Refine::from_f64(y[i].to_f64() - alpha[i].to_f64() / qii);
            let obs = 1.0 / qii;
            variance[i] = P::Refine::from_f64(match options.variance_kind {
                VarianceKind::Observation => obs,
                VarianceKind::Latent => (obs - noise).max(0.0),
            });
        }
        P::inverse_mean_variance(
            self.y_transform.as_ref(),
            &mut mean,
            &mut variance,
            &mut InverseBuffers::default(),
        )?;
        Ok(Prediction {
            mean,
            variance,
            variance_kind: options.variance_kind,
        })
    }

    /// Leave-one-out from an `f64` factor of the kernel rounded to `f32`.
    ///
    /// The stored `f32` factor is the predict factor. A cancelled
    /// `y_i - α_i / Q_ii` needs the inverse diagonal of that rounded matrix
    /// solved in `f64`, which is the same LOO formula with a tighter residual.
    fn loo_from_rounded_kernel(
        &self,
        options: PredictOptions,
    ) -> Result<Prediction<P::Refine>, GprError> {
        let n = self.n;
        let kernel = self.kernel.compile();
        let mut a = Mat::<f64>::zeros(n, n);
        let mut scratch_k = Mat::<f64>::zeros(n, n);
        let sources = self.sources.to_f64()?;
        let sources = sources.as_ref();
        with_kernel_exp!(self.policies.math, M => kernel.eval_gram::<M>(
            GramInputs::supplied(self.x_active(), <K::Supply>::squares(sources)),
            a.as_mut(),
            Triangle::Lower,
            scratch_k.as_mut(),
            &mut Vec::new(),
        ))?;
        let noise = self.likelihood.noise_variance();
        for i in 0..n {
            a[(i, i)] += noise;
        }
        for col in 0..n {
            for row in (col + 1)..n {
                a[(col, row)] = a[(row, col)];
            }
        }
        for col in 0..n {
            for row in 0..n {
                a[(row, col)] = f64::from(a[(row, col)] as f32);
            }
        }
        let par = faer_par(n);
        let factor_req = llt::factor::cholesky_in_place_scratch::<f64>(n, par, Default::default());
        let mut factor_scratch = MemBuffer::new(factor_req);
        cholesky_lower(&mut a, &mut factor_scratch, 0.0, CholeskyStage::Predict)?;
        let solve_par = faer_par_dims(n, 1);
        let solve_req = llt::solve::solve_in_place_scratch::<f64>(n, 1, solve_par);
        let mut solve_scratch = MemBuffer::new(solve_req);
        let mut rhs = Mat::<f64>::from_fn(n, 1, |i, _| self.y_train[i]);
        llt::solve::solve_in_place(
            a.as_ref(),
            rhs.as_mut(),
            solve_par,
            MemStack::new(&mut solve_scratch),
        );
        let alpha: Vec<f64> = (0..n).map(|i| rhs[(i, 0)]).collect();
        let mut mean = vec![P::Refine::from_f64(0.0); n];
        let mut variance = vec![P::Refine::from_f64(0.0); n];
        for i in 0..n {
            for row in 0..n {
                rhs[(row, 0)] = if row == i { 1.0 } else { 0.0 };
            }
            llt::solve::solve_in_place(
                a.as_ref(),
                rhs.as_mut(),
                solve_par,
                MemStack::new(&mut solve_scratch),
            );
            let qii = rhs[(i, 0)];
            if !qii.is_finite() || qii <= 0.0 {
                return Err(GprError::NonPositiveDefiniteMatrix);
            }
            mean[i] = P::Refine::from_f64(self.y_train[i] - alpha[i] / qii);
            let obs = 1.0 / qii;
            variance[i] = P::Refine::from_f64(match options.variance_kind {
                VarianceKind::Observation => obs,
                VarianceKind::Latent => (obs - noise).max(0.0),
            });
        }
        P::inverse_mean_variance(
            self.y_transform.as_ref(),
            &mut mean,
            &mut variance,
            &mut InverseBuffers::default(),
        )?;
        Ok(Prediction {
            mean,
            variance,
            variance_kind: options.variance_kind,
        })
    }
}

impl<P: GpScalar, K: ModelKernel<Supply = NoSupply>> GprCore<P, K> {
    /// Posterior draws at the query from [`Self::write_covariance`].
    pub(crate) fn sample_with(
        &self,
        factor: StoredFactor<'_, P::Storage>,
        alpha: &[P::Refine],
        q: Query<'_, P::Storage>,
        options: PredictOptions,
        n_draws: usize,
        seed: u64,
    ) -> Result<Vec<P::Refine>, GprError> {
        self.write_covariance(factor, alpha, q, (), options)?.draw(
            n_draws,
            seed,
            self.policies.jitter,
        )
    }
}

/// One predict's query: the coordinates (`m × n_cols`, column-major; no
/// columns on a model of supplied distances alone) and, for a distance
/// model, the train × query blocks (`cross`, and in `f64` for a refining
/// precision). A covariance takes the query × query blocks next to it.
pub(crate) struct Query<'q, T: KernelScalar, S: Supply = NoSupply> {
    pub(crate) xs: &'q [f64],
    pub(crate) m: usize,
    pub(crate) n_cols: usize,
    /// The train × query blocks of every slot (nothing for a coordinate
    /// kernel).
    pub(crate) cross: S::Rects<'q, T>,
    /// The same blocks read at `f64`, for a model that refines in `f64`.
    pub(crate) cross64: S::Rects<'q, f64>,
}

impl<T: KernelScalar, S: Supply> Clone for Query<'_, T, S> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: KernelScalar, S: Supply> Copy for Query<'_, T, S> {}

impl<'q, T: KernelScalar> Query<'q, T> {
    /// Coordinates only.
    pub(crate) fn points(xs: &'q [f64], m: usize, n_cols: usize) -> Self {
        Self {
            xs,
            m,
            n_cols,
            cross: (),
            cross64: (),
        }
    }
}

/// Query buffers a predict fills: the workspace, the cast cache for the
/// training inputs, and per-thread kernel scratch.
struct QueryBuffers<'a, P: GpScalar> {
    query: &'a mut QueryWorkspace<P>,
    x_cast: &'a mut <P::Storage as ScalarOps>::ColCast,
    thread_scratch: &'a mut [Mat<P::Storage>],
}

/// Borrowed model pieces the predictive moments read.
struct CoreRefs<'a, P: GpScalar, S: Supply> {
    kernel: &'a KernelSpec<S>,
    compiled: &'a CompiledKernel<P::Storage, S>,
    alpha: &'a [P::Refine],
    x_train: MatRef<'a, f64>,
    noise: f64,
    y_transform: &'a dyn TargetTransform,
    math: crate::policy::KernelExp,
}

struct MomentInputs<'a, P: GpScalar, S: Supply> {
    core: CoreRefs<'a, P, S>,
    factor: StoredFactor<'a, P::Storage>,
    /// Transformed query, column-major `m × d`.
    query_xs: &'a [f64],
    query_x: MatRef<'a, P::Storage>,
    /// `K(X, xs)` on entry; `L⁻¹ K(X, xs)` on return.
    k_star: MatMut<'a, P::Storage>,
    kss: &'a mut [P::Storage],
    n_cols: usize,
    cross64: S::Rects<'a, f64>,
    options: PredictOptions,
}

/// Writes the predictive mean and diagonal variance from `K(X, xs)`.
///
/// Latent variance is `k(x*, x*) − k_*ᵀ A⁻¹ k_*`, clipped at 0. Observation
/// variance adds `σn²` in the transformed space. Both are mapped back by the
/// target transform.
fn write_moments<P: GpScalar, S: Supply>(
    inputs: MomentInputs<'_, P, S>,
    out: &mut Prediction<P::Refine>,
) -> Result<(), GprError> {
    let MomentInputs {
        core,
        factor,
        query_xs,
        query_x,
        mut k_star,
        kss,
        n_cols,
        cross64,
        options,
    } = inputs;
    let n = k_star.nrows();
    let m = k_star.ncols();
    let zero = P::Refine::from_f64(0.0);
    if out.mean.len() != m {
        out.mean.resize(m, zero);
    }
    if out.variance.len() != m {
        out.variance.resize(m, zero);
    }
    with_kernel_exp!(core.math, M => P::predict_means::<M, _>(
        core.kernel,
        k_star.as_ref(),
        core.x_train,
        query_xs,
        n_cols,
        cross64,
        core.alpha,
        &mut out.mean,
    ))?;
    factor.inv_l_in_place(k_star.as_mut());
    core.compiled.eval_diag(query_x, kss)?;
    let noise_s = P::Storage::from_f64(core.noise);
    let zero_s = P::Storage::from_f64(0.0);
    for col in 0..m {
        let mut quad = 0.0f64;
        for row in 0..n {
            let v = k_star[(row, col)].to_f64();
            quad += factor.scaled_product(row, v, v);
        }
        let mut latent = kss[col] - P::Storage::from_f64(quad);
        if latent.to_f64() < 0.0 {
            latent = zero_s;
        }
        let var_s = match options.variance_kind {
            VarianceKind::Latent => latent,
            VarianceKind::Observation => latent + noise_s,
        };
        out.variance[col] = P::Refine::from_f64(var_s.to_f64());
    }
    P::inverse_mean_variance(
        core.y_transform,
        &mut out.mean,
        &mut out.variance,
        &mut InverseBuffers::default(),
    )?;
    out.variance_kind = options.variance_kind;
    Ok(())
}

/// Factor operations the shared read paths need.
///
/// For LLT `A = L Lᵀ`; for LDLT `A = L D Lᵀ` with unit-lower `L`. Either way
/// `kᵀ A⁻¹ c = Σᵢ (L⁻¹k)ᵢ (L⁻¹c)ᵢ / wᵢ` with `wᵢ = 1` (LLT) or `Dᵢ` (LDLT).
impl<T: KernelScalar> StoredFactor<'_, T> {
    fn order(&self) -> usize {
        match *self {
            Self::Llt(l) => l.nrows(),
            Self::Ldlt(ld) => ld.nrows(),
        }
    }

    /// Overwrites each column of `rhs` (`n × m`) with `L⁻¹` of that column.
    pub(crate) fn inv_l_in_place(&self, rhs: MatMut<'_, T>) {
        let n = self.order();
        let m = rhs.ncols();
        match *self {
            Self::Llt(l) => faer::linalg::triangular_solve::solve_lower_triangular_in_place(
                l,
                rhs,
                faer_par_dims(n, m),
            ),
            Self::Ldlt(ld) => crate::linalg::apply_ldlt_inv_l(ld, rhs, n),
        }
    }

    /// `a · b / wᵢ` for row `row` of two [`Self::inv_l_in_place`] columns.
    pub(crate) fn scaled_product(&self, row: usize, a: f64, b: f64) -> f64 {
        match *self {
            Self::Llt(_) => a * b,
            Self::Ldlt(ld) => a * b / ld[(row, row)].to_f64(),
        }
    }

    /// `log |A|`.
    pub(crate) fn log_det(&self) -> T {
        match *self {
            Self::Llt(l) => log_det_from_l(l, l.nrows()),
            Self::Ldlt(ld) => {
                let mut log_det = T::from_f64(0.0);
                for i in 0..ld.nrows() {
                    log_det += ld[(i, i)].ln();
                }
                log_det
            }
        }
    }

    /// Writes `diag(A⁻¹)` into `out`.
    pub(crate) fn inv_diag(&self, out: &mut [T]) {
        match *self {
            Self::Llt(l) => inv_diag_from_chol_l(l, out),
            Self::Ldlt(ld) => {
                let n = ld.nrows();
                let mut inv_l = Mat::<T>::from_fn(n, n, |row, col| {
                    T::from_f64(if row == col { 1.0 } else { 0.0 })
                });
                crate::linalg::apply_ldlt_inv_l(ld, inv_l.as_mut(), n);
                for (i, slot) in out.iter_mut().enumerate() {
                    let mut q = 0.0f64;
                    for k in i..n {
                        let v = inv_l[(k, i)].to_f64();
                        q += v * v / ld[(k, k)].to_f64();
                    }
                    *slot = T::from_f64(q);
                }
            }
        }
    }
}
