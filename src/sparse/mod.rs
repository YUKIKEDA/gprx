//! Crate-private pieces shared by the sparse models ([`crate::Sgpr`] and
//! [`crate::Svgp`]): the settings every trainer holds, the training data
//! every fitted model holds, and the kernel + likelihood `θ` over both.

use faer::{Mat, MatMut, MatRef};

use crate::error::GprError;
use crate::kernel::KernelSpec;
use crate::kernel::{CompiledKernel, GramInputs, KernelScalar, Triangle};
use crate::likelihood::GaussianLikelihood;
use crate::param::{Interval, write_params};
use crate::policy::JitterPolicy;
use crate::policy::KernelExp;

/// Kernel, likelihood, and kernel `exp` of an untrained sparse model.
#[derive(Clone, Debug)]
pub(crate) struct SparseSpec {
    pub(crate) kernel: KernelSpec,
    pub(crate) likelihood: GaussianLikelihood,
    pub(crate) math: KernelExp,
}

impl SparseSpec {
    pub(crate) fn new(kernel: KernelSpec, likelihood: GaussianLikelihood) -> Self {
        Self {
            kernel,
            likelihood,
            math: KernelExp::default(),
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

/// Training data and settings of a fitted sparse model: kernel,
/// likelihood, kernel `exp`, column-major `X` (`n × d`) and `Z` (`m × d`),
/// and `y`.
#[derive(Clone, Debug)]
pub(crate) struct SparseCore {
    pub(crate) kernel: KernelSpec,
    pub(crate) likelihood: GaussianLikelihood,
    pub(crate) math: KernelExp,
    pub(crate) x_obs: Vec<f64>,
    pub(crate) z_obs: Vec<f64>,
    pub(crate) y: Vec<f64>,
    pub(crate) n: usize,
    pub(crate) m: usize,
    pub(crate) d: usize,
}

impl SparseCore {
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

    /// The trainer settings this model was fitted with.
    pub(crate) fn spec(&self) -> SparseSpec {
        SparseSpec {
            kernel: self.kernel.clone(),
            likelihood: self.likelihood,
            math: self.math,
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
/// SparseCore` field. One set of docs for [`crate::FittedSgpr`],
/// [`crate::OnlineSgpr`], and [`crate::FittedSvgp`].
macro_rules! sparse_core_accessors {
    () => {
        /// Returns the number of training points.
        pub fn n(&self) -> usize {
            self.core.n
        }

        /// Returns the number of inducing points.
        pub fn m(&self) -> usize {
            self.core.m
        }

        /// Returns the feature dimension.
        pub fn d(&self) -> usize {
            self.core.d
        }

        /// Returns the kernel whose hyperparameters this model owns.
        pub fn kernel(&self) -> &$crate::kernel::KernelSpec {
            &self.core.kernel
        }

        /// Returns the observation-noise model.
        pub fn likelihood(&self) -> &$crate::GaussianLikelihood {
            &self.core.likelihood
        }

        /// Returns the kernel `exp` mode the trainer set with `with_math`.
        pub fn math(&self) -> $crate::KernelExp {
            self.core.math
        }

        /// Returns the original training features in column-major order.
        pub fn x(&self) -> &[f64] {
            &self.core.x_obs
        }

        /// Returns the inducing features in column-major order.
        pub fn z(&self) -> &[f64] {
            &self.core.z_obs
        }

        /// Returns the original training targets.
        pub fn y(&self) -> &[f64] {
            &self.core.y
        }
    };
}

pub(crate) use sparse_core_accessors;

/// Jitter retries for factoring `K_mm = k(Z, Z)`. Observation noise is not
/// on `K_mm`, so close inducing points need a small diagonal offset.
pub(crate) fn k_mm_jitter_policy() -> JitterPolicy {
    JitterPolicy::adaptive(1e-8, 10.0, 5, 1e-3).unwrap_or_default()
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

    /// An output-shaped scratch for a kernel call that takes one directly.
    pub(crate) fn scratch(&mut self, rows: usize, cols: usize) -> MatMut<'_, T> {
        view(&mut self.scratch, rows, cols)
    }
}

/// `buf` grown to at least `rows × cols`, viewed at that shape.
fn view<T: KernelScalar>(buf: &mut Mat<T>, rows: usize, cols: usize) -> MatMut<'_, T> {
    if buf.nrows() < rows || buf.ncols() < cols {
        *buf = Mat::zeros(rows.max(buf.nrows()), cols.max(buf.ncols()));
    }
    buf.as_mut().submatrix_mut(0, 0, rows, cols)
}

/// Kernel scratch a fitted sparse model keeps between its `&mut self` calls
/// (`set_params`, gradient, Hessian, online updates): one for the storage
/// scalar `S`, one for the `f64` assembly a rounding precision starts from.
#[derive(Clone, Debug, Default)]
pub(crate) struct SparseScratch<S: KernelScalar> {
    pub(crate) storage: KernelScratch<S>,
    pub(crate) f64: KernelScratch<f64>,
}
