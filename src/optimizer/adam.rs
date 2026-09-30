//! Mini-batch Adam for [`crate::Svgp`]. Not an [`super::Optimizer`].

use std::num::{NonZeroU64, NonZeroUsize};

use crate::error::GprError;

/// Mini-batch Adam used by [`crate::Svgp<Adam>::fit`].
///
/// Kingma defaults: step size `1e-3`, `β1 = 0.9`, `β2 = 0.999`, `ε = 1e-8`,
/// batch size 32, 100 epochs, seed 0. Bias correction is always on. This type
/// does not implement [`super::Optimizer`]; [`crate::Gpr`] cannot take it.
///
/// # Examples
///
/// ```rust
/// use std::num::{NonZeroU64, NonZeroUsize};
/// use gprx::Adam;
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let _adam = Adam::new()
///     .with_learning_rate(1e-3)?
///     .with_beta1(0.9)?
///     .with_beta2(0.999)?
///     .with_epsilon(1e-8)?
///     .with_batch_size(NonZeroUsize::MIN)
///     .with_epochs(NonZeroU64::MIN)
///     .with_seed(0);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Adam {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    batch_size: NonZeroUsize,
    epochs: NonZeroU64,
    seed: u64,
}

impl Default for Adam {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            batch_size: match NonZeroUsize::new(32) {
                Some(n) => n,
                None => NonZeroUsize::MIN,
            },
            epochs: match NonZeroU64::new(100) {
                Some(n) => n,
                None => NonZeroU64::MIN,
            },
            seed: 0,
        }
    }
}

impl Adam {
    /// Builds Adam with Kingma defaults and seed 0.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the step size (default `1e-3`).
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidConfig`] if `learning_rate` is not
    /// finite or is not strictly positive.
    pub fn with_learning_rate(mut self, learning_rate: f64) -> Result<Self, GprError> {
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err(GprError::InvalidConfig {
                reason: "Adam learning rate must be finite and > 0".to_owned(),
            });
        }
        self.learning_rate = learning_rate;
        Ok(self)
    }

    /// Sets `β1` (default `0.9`).
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidConfig`] if `beta1` is not in `[0, 1)`.
    pub fn with_beta1(mut self, beta1: f64) -> Result<Self, GprError> {
        if !beta1.is_finite() || !(0.0..1.0).contains(&beta1) {
            return Err(GprError::InvalidConfig {
                reason: "Adam β1 must be finite and in [0, 1)".to_owned(),
            });
        }
        self.beta1 = beta1;
        Ok(self)
    }

    /// Sets `β2` (default `0.999`).
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidConfig`] if `beta2` is not in `[0, 1)`.
    pub fn with_beta2(mut self, beta2: f64) -> Result<Self, GprError> {
        if !beta2.is_finite() || !(0.0..1.0).contains(&beta2) {
            return Err(GprError::InvalidConfig {
                reason: "Adam β2 must be finite and in [0, 1)".to_owned(),
            });
        }
        self.beta2 = beta2;
        Ok(self)
    }

    /// Sets the denominator offset `ε` (default `1e-8`).
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidConfig`] if `epsilon` is not finite
    /// or is not strictly positive.
    pub fn with_epsilon(mut self, epsilon: f64) -> Result<Self, GprError> {
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(GprError::InvalidConfig {
                reason: "Adam ε must be finite and > 0".to_owned(),
            });
        }
        self.epsilon = epsilon;
        Ok(self)
    }

    /// Sets the mini-batch length (default 32). A value `≥ n` is one full batch.
    pub fn with_batch_size(mut self, batch_size: NonZeroUsize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Sets the number of passes over the data (default 100).
    pub fn with_epochs(mut self, epochs: NonZeroU64) -> Self {
        self.epochs = epochs;
        self
    }

    /// Sets the shuffle seed used at the start of each epoch (default 0).
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub(crate) fn batch_size(&self) -> usize {
        self.batch_size.get()
    }

    pub(crate) fn epochs(&self) -> u64 {
        self.epochs.get()
    }

    pub(crate) fn seed(&self) -> u64 {
        self.seed
    }

    pub(crate) fn step(
        &self,
        z: &mut [f64],
        grad: &[f64],
        moment1: &mut [f64],
        moment2: &mut [f64],
        timestep: &mut u64,
    ) {
        *timestep += 1;
        let t = *timestep as f64;
        let corr1 = 1.0 - self.beta1.powf(t);
        let corr2 = 1.0 - self.beta2.powf(t);
        for i in 0..z.len() {
            moment1[i] = self.beta1 * moment1[i] + (1.0 - self.beta1) * grad[i];
            moment2[i] = self.beta2 * moment2[i] + (1.0 - self.beta2) * grad[i] * grad[i];
            let mhat = moment1[i] / corr1;
            let vhat = moment2[i] / corr2;
            z[i] -= self.learning_rate * mhat / (vhat.sqrt() + self.epsilon);
        }
    }
}
