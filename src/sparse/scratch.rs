//! The buffers of one sparse operation: kernel evaluation
//! ([`KernelScratch`]), prediction ([`PredictScratch`]), and both together
//! ([`SparseScratch`]).

#[allow(
    unused_imports,
    reason = "a split file takes its parent's imports whole; each uses some"
)]
use super::*;

/// Kernel-evaluation buffers of one sparse operation: the output-shaped
/// scratch, the nested sum / product levels, and the train–query distance
/// block. Every buffer grows to the largest shape asked for and is viewed at
/// the shape of each call, so one operation's kernel calls share them.
/// Scratch: contents mean nothing between calls.
pub(crate) struct KernelScratch<T> {
    scratch: Mat<T>,
    nested: Vec<Mat<T>>,
    dist: Mat<T>,
    /// `K(X, Z)` (`n × m`) or a transposed weight on supplied blocks, before
    /// its one transpose ([`Self::cross_mn_into`]).
    transposed: Mat<T>,
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
            transposed: Mat::new(),
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
            transposed: Mat::new(),
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

    /// `K(Z, X)` (`m × n`) of `sets` into `out`. A coordinate kernel forms
    /// it directly; a kernel on supplied blocks (`n × m`) forms `K(X, Z)` in
    /// a kept buffer and writes its transpose.
    pub(crate) fn cross_mn_into<M: crate::math::KernelMath, U: Supply>(
        &mut self,
        compiled: &CompiledKernel<T, U>,
        sets: SparseSets<'_, T, U>,
        mut out: MatMut<'_, T>,
    ) -> Result<(), GprError> {
        if let Some(coords) = U::coordinates(compiled) {
            return self.cross_into::<M, NoSupply>(coords, CrossViews::points(sets.z, sets.x), out);
        }
        let (n, m) = (sets.x.nrows(), sets.z.nrows());
        let reads = compiled.reads_distances()?;
        let KernelScratch {
            dist,
            transposed,
            scratch,
            nested,
            ..
        } = self;
        let mut k = view(transposed, n, m);
        let views = sets.k_nm();
        // The coordinate leaves of a mixed tree read their `d²` here, kept
        // from call to call as the coordinate path keeps them.
        compiled.eval_cross_slots::<M>(
            views.x1,
            views.x2,
            views.slots,
            reads.then(|| view(dist, n, m)),
            k.as_mut(),
            view(scratch, n, m),
            nested,
            &mut [],
        )?;
        out.copy_from(k.transpose());
        Ok(())
    }

    /// [`Self::cross_mn_into`] into a new matrix.
    pub(crate) fn cross_mn<M: crate::math::KernelMath, U: Supply>(
        &mut self,
        compiled: &CompiledKernel<T, U>,
        sets: SparseSets<'_, T, U>,
    ) -> Result<Mat<T>, GprError> {
        let mut out = Mat::zeros(sets.z.nrows(), sets.x.nrows());
        self.cross_mn_into::<M, U>(compiled, sets, out.as_mut())?;
        Ok(out)
    }

    /// `∂K(Z, X)/∂θ_p` (`m × n`) into `out`, as [`Self::cross_mn_into`].
    pub(crate) fn grad_cross_mn_into<M: crate::math::KernelMath, U: Supply>(
        &mut self,
        compiled: &CompiledKernel<T, U>,
        sets: SparseSets<'_, T, U>,
        mut out: MatMut<'_, T>,
        param_idx: usize,
    ) -> Result<(), GprError> {
        if let Some(coords) = U::coordinates(compiled) {
            let views = CrossViews::points(sets.z, sets.x);
            return self.grad_cross_into::<M, NoSupply>(coords, views, out, param_idx);
        }
        let (n, m) = (sets.x.nrows(), sets.z.nrows());
        let mut k = std::mem::replace(&mut self.transposed, Mat::new());
        let result =
            self.grad_cross_into::<M, U>(compiled, sets.k_nm(), view(&mut k, n, m), param_idx);
        if result.is_ok() {
            out.copy_from(k.as_ref().submatrix(0, 0, n, m).transpose());
        }
        self.transposed = k;
        result
    }

    /// `∂²K(Z, X)/∂θ_i ∂θ_j` (`m × n`) into `out`, as
    /// [`Self::cross_mn_into`].
    pub(crate) fn hess_cross_mn_into<M: crate::math::KernelMath, U: Supply>(
        &mut self,
        compiled: &CompiledKernel<T, U>,
        sets: SparseSets<'_, T, U>,
        mut out: MatMut<'_, T>,
        pair: (usize, usize),
    ) -> Result<(), GprError> {
        if let Some(coords) = U::coordinates(compiled) {
            let views = CrossViews::points(sets.z, sets.x);
            return self.hess_cross_into::<M, NoSupply>(coords, views, out, pair);
        }
        let (n, m) = (sets.x.nrows(), sets.z.nrows());
        let mut k = std::mem::replace(&mut self.transposed, Mat::new());
        let result = self.hess_cross_into::<M, U>(compiled, sets.k_nm(), view(&mut k, n, m), pair);
        if result.is_ok() {
            out.copy_from(k.as_ref().submatrix(0, 0, n, m).transpose());
        }
        self.transposed = k;
        result
    }

    /// Adds `coeff · ⟨weight, ∂K(Z, X)/∂θ⟩_F` (`weight` is `m × n`) for
    /// every kernel parameter. On supplied blocks the weight is transposed
    /// once into a kept buffer and contracted with `K(X, Z)`.
    pub(crate) fn add_cross_contraction_mn<M: crate::math::KernelMath, U: Supply>(
        &mut self,
        compiled: &CompiledKernel<T, U>,
        sets: SparseSets<'_, T, U>,
        weight: MatRef<'_, T>,
        coeff: f64,
        out: &mut [f64],
    ) -> Result<(), GprError> {
        if let Some(coords) = U::coordinates(compiled) {
            let views = CrossViews::points(sets.z, sets.x);
            return self.add_cross_contraction::<M, NoSupply>(coords, views, weight, coeff, out);
        }
        let (n, m) = (sets.x.nrows(), sets.z.nrows());
        let mut w = std::mem::replace(&mut self.transposed, Mat::new());
        let mut wt = view(&mut w, n, m);
        wt.copy_from(weight.transpose());
        let result = self.add_cross_contraction::<M, U>(
            compiled,
            sets.k_nm(),
            w.as_ref().submatrix(0, 0, n, m),
            coeff,
            out,
        );
        self.transposed = w;
        result
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
pub(super) fn fit_exact<T: KernelScalar>(
    pool: &mut Vec<Mat<T>>,
    count: usize,
    rows: usize,
    cols: usize,
) {
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
        zz: &'a BlockStore<f64>,
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
