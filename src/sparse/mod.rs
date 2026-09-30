//! Crate-private pieces shared by the sparse models ([`crate::Sgpr`] and
//! [`crate::Svgp`]): the settings every trainer holds, the training data
//! every fitted model holds, and the kernel + likelihood `θ` over both.

use crate::error::GprError;
use crate::gpr::KernelExp;
use crate::kernel::KernelSpec;
use crate::likelihood::GaussianLikelihood;
use crate::param::{Interval, write_params};

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
