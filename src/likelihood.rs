//! Gaussian observation noise, stored as `θ = log(σn²)`.

use crate::error::GpError;

/// Gaussian likelihood with observation noise variance `σn²`.
///
/// The optimizer parameter is `θ = log(σn²)`, so positivity is unconstrained.
/// Differentiating the kernel matrix with respect to that parameter gives
/// `∂K/∂θ = σn² I`, not `2 σn I`.
///
/// Observation noise lives here. Do not also add a large white kernel term;
/// that double-counts noise. Cholesky jitter (`LltRegularization`) is a
/// separate numerical stabilizer and is not this parameter.
///
/// # Examples
///
/// ```rust
/// use gprx::GaussianLikelihood;
///
/// # fn main() -> Result<(), gprx::GpError> {
/// let lik = GaussianLikelihood::new(0.25)?;
/// let mut diag = [1.0, 1.0];
/// lik.add_noise_diag(&mut diag);
/// # let _ = diag;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaussianLikelihood {
    log_noise_variance: f64,
}

impl GaussianLikelihood {
    /// Builds a likelihood from a positive finite noise variance `σn²`.
    ///
    /// # Errors
    ///
    /// Returns [`GpError::InvalidNoiseVariance`] if `noise_variance` is not
    /// finite or not strictly positive.
    pub fn new(noise_variance: f64) -> Result<Self, GpError> {
        if !noise_variance.is_finite() {
            return Err(invalid_noise("noise variance must be finite"));
        }
        if noise_variance <= 0.0 {
            return Err(invalid_noise("noise variance must be positive"));
        }
        Ok(Self {
            log_noise_variance: noise_variance.ln(),
        })
    }

    /// Builds a likelihood from `θ = log(σn²)`.
    ///
    /// # Errors
    ///
    /// Returns [`GpError::InvalidNoiseVariance`] if `θ` is not finite, if
    /// `exp(θ)` overflows to a non-finite variance, or if `exp(θ)` underflows
    /// to zero.
    pub fn from_log_noise_variance(log_noise_variance: f64) -> Result<Self, GpError> {
        Ok(Self {
            log_noise_variance: validate_log_noise_variance(log_noise_variance)?,
        })
    }

    /// Returns `σn² = exp(θ)`.
    pub fn noise_variance(&self) -> f64 {
        self.log_noise_variance.exp()
    }

    /// Returns `θ = log(σn²)`.
    pub fn log_noise_variance(&self) -> f64 {
        self.log_noise_variance
    }

    /// Returns the number of optimizer parameters (always 1).
    pub fn num_params(&self) -> usize {
        1
    }

    /// Writes `θ` into a length-1 slice.
    ///
    /// # Errors
    ///
    /// Returns [`GpError::InvalidHyperparameter`] if `out` is not length 1.
    pub fn get_params(&self, out: &mut [f64]) -> Result<(), GpError> {
        expect_one_param(out.len())?;
        out[0] = self.log_noise_variance;
        Ok(())
    }

    /// Replaces `θ` from a length-1 slice.
    ///
    /// # Errors
    ///
    /// Returns [`GpError::InvalidHyperparameter`] if `params` is not length 1,
    /// or [`GpError::InvalidNoiseVariance`] if the new `θ` is invalid.
    pub fn set_params(&mut self, params: &[f64]) -> Result<(), GpError> {
        expect_one_param(params.len())?;
        self.log_noise_variance = validate_log_noise_variance(params[0])?;
        Ok(())
    }

    /// Adds `σn²` to each entry of a kernel diagonal (implements `K += σn² I`).
    pub fn add_noise_diag(&self, k_diag: &mut [f64]) {
        let noise = self.noise_variance();
        for entry in k_diag {
            *entry += noise;
        }
    }

    /// Writes the diagonal of `∂K/∂θ_i`. For `i = 0` this is `σn²` in every entry.
    ///
    /// # Errors
    ///
    /// Returns [`GpError::InvalidHyperparameter`] if `param_idx` is not 0.
    pub fn noise_grad_diag(&self, dk_diag: &mut [f64], param_idx: usize) -> Result<(), GpError> {
        if param_idx != 0 {
            return Err(GpError::InvalidHyperparameter {
                reason: "likelihood has a single parameter at index 0".to_owned(),
            });
        }
        let dkd_theta = self.noise_variance();
        for entry in dk_diag {
            *entry = dkd_theta;
        }
        Ok(())
    }
}

fn invalid_noise(reason: &'static str) -> GpError {
    GpError::InvalidNoiseVariance {
        reason: reason.to_owned(),
    }
}

fn validate_log_noise_variance(theta: f64) -> Result<f64, GpError> {
    if !theta.is_finite() {
        return Err(invalid_noise("log noise variance must be finite"));
    }
    let variance = theta.exp();
    if !variance.is_finite() {
        return Err(invalid_noise(
            "noise variance overflowed to a non-finite value",
        ));
    }
    if variance <= 0.0 {
        return Err(invalid_noise("noise variance underflowed to zero"));
    }
    Ok(theta)
}

fn expect_one_param(len: usize) -> Result<(), GpError> {
    if len == 1 {
        Ok(())
    } else {
        Err(GpError::InvalidHyperparameter {
            reason: format!("expected 1 likelihood parameter, got {len}"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::GaussianLikelihood;
    use crate::error::GpError;

    const TOL: f64 = 1e-10;

    fn assert_close(actual: f64, expected: f64) {
        let scale = expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= TOL * scale,
            "actual={actual}, expected={expected}"
        );
    }

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn is_send_sync() {
        assert_send_sync::<GaussianLikelihood>();
    }

    #[test]
    fn add_noise_diag_adds_variance() {
        let lik = GaussianLikelihood::new(0.25).expect("valid");
        let mut diag = [1.0, 2.0, 3.0];
        lik.add_noise_diag(&mut diag);
        assert_close(diag[0], 1.25);
        assert_close(diag[1], 2.25);
        assert_close(diag[2], 3.25);
    }

    #[test]
    fn noise_grad_diag_is_variance_not_twice_sigma() {
        let variance = 0.25;
        let lik = GaussianLikelihood::new(variance).expect("valid");
        let mut diag = [0.0, 0.0];
        lik.noise_grad_diag(&mut diag, 0).expect("index 0");
        assert_close(diag[0], variance);
        assert_close(diag[1], variance);
        let twice_sigma = 2.0 * variance.sqrt();
        assert!(
            (diag[0] - twice_sigma).abs() > 0.1,
            "must not use ∂K/∂σn = 2σn I"
        );
    }

    #[test]
    fn noise_grad_matches_finite_difference_of_variance() {
        let lik = GaussianLikelihood::from_log_noise_variance(-1.5).expect("valid");
        let theta = lik.log_noise_variance();
        let h = 1e-6;
        let plus = GaussianLikelihood::from_log_noise_variance(theta + h).expect("valid");
        let minus = GaussianLikelihood::from_log_noise_variance(theta - h).expect("valid");
        let fd = (plus.noise_variance() - minus.noise_variance()) / (2.0 * h);
        let mut diag = [0.0];
        lik.noise_grad_diag(&mut diag, 0).expect("index 0");
        assert_close(diag[0], fd);
        assert_close(diag[0], lik.noise_variance());
    }

    #[test]
    fn get_set_params_roundtrip() {
        let mut lik = GaussianLikelihood::new(0.5).expect("valid");
        let mut params = [0.0];
        lik.get_params(&mut params).expect("len 1");
        assert_close(params[0], 0.5_f64.ln());
        params[0] = 0.25_f64.ln();
        lik.set_params(&params).expect("len 1");
        assert_close(lik.noise_variance(), 0.25);
    }

    #[test]
    fn rejects_non_positive_and_non_finite_variance() {
        assert!(matches!(
            GaussianLikelihood::new(0.0),
            Err(GpError::InvalidNoiseVariance { .. })
        ));
        assert!(matches!(
            GaussianLikelihood::new(-1.0),
            Err(GpError::InvalidNoiseVariance { .. })
        ));
        assert!(matches!(
            GaussianLikelihood::new(f64::NAN),
            Err(GpError::InvalidNoiseVariance { .. })
        ));
        assert!(matches!(
            GaussianLikelihood::from_log_noise_variance(f64::INFINITY),
            Err(GpError::InvalidNoiseVariance { .. })
        ));
        assert!(matches!(
            GaussianLikelihood::from_log_noise_variance(-800.0),
            Err(GpError::InvalidNoiseVariance { .. })
        ));
        let mut lik = GaussianLikelihood::new(1.0).expect("valid");
        assert!(matches!(
            lik.set_params(&[-800.0]),
            Err(GpError::InvalidNoiseVariance { .. })
        ));
    }

    #[test]
    fn noise_grad_rejects_other_param_index() {
        let lik = GaussianLikelihood::new(1.0).expect("valid");
        let mut diag = [0.0];
        assert!(matches!(
            lik.noise_grad_diag(&mut diag, 1),
            Err(GpError::InvalidHyperparameter { .. })
        ));
    }
}
