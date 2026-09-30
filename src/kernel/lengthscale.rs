//! Per-dimension lengthscales with optimizer parameters `θ_d = log(ℓ_d)`.

use crate::error::GprError;
use crate::param::{BoundedParam, Interval};

/// ARD lengthscales for stationary leaves that scale each input dimension.
///
/// `num_params` is the feature dimension `d`. [`Self::get_params`] /
/// [`Self::set_params`] read and write `θ_d = log(ℓ_d)`. Isotropic leaves keep
/// a single `θ = log(ℓ)` and do not use this type. [`super::MaternArdKernel`]
/// and [`super::RationalQuadraticArdKernel`] use this mouth.
///
/// Cloning copies the `d`-vectors.
///
/// # Examples
///
/// ```rust
/// use gprx::kernel::ArdLengthscales;
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let scales = ArdLengthscales::new(&[1.0, 2.0])?;
/// assert_eq!(scales.num_params(), 2);
/// assert!((scales.lengthscale(1)? - 2.0).abs() < 1e-12);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct ArdLengthscales {
    params: Vec<BoundedParam>,
    log_lengthscales: Vec<f64>,
    inv_ell_sq: Vec<f64>,
}

impl ArdLengthscales {
    /// Builds ARD lengthscales from positive finite `ℓ_d`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if the slice is empty or a
    /// value is not finite and strictly positive.
    pub fn new(lengthscales: &[f64]) -> Result<Self, GprError> {
        require_non_empty(lengthscales.len())?;
        let mut params = Vec::with_capacity(lengthscales.len());
        for &ell in lengthscales {
            validate_lengthscale(ell)?;
            params.push(BoundedParam::default_positive(ell)?);
        }
        Self::from_params(params)
    }

    /// Builds ARD lengthscales from `θ_d = log(ℓ_d)`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if the slice is empty, a
    /// `θ_d` is not finite, or `exp(θ_d)` overflows or underflows to zero.
    pub fn from_log_lengthscales(log_lengthscales: &[f64]) -> Result<Self, GprError> {
        require_non_empty(log_lengthscales.len())?;
        let mut params = Vec::with_capacity(log_lengthscales.len());
        for &theta in log_lengthscales {
            let log = validate_log_lengthscale(theta)?;
            params.push(BoundedParam::default_positive(log.exp())?);
        }
        Self::from_params(params)
    }

    pub(crate) fn from_bounded(params: Vec<BoundedParam>) -> Result<Self, GprError> {
        Self::from_params(params)
    }

    pub(crate) fn bounded_params(&self) -> &[BoundedParam] {
        &self.params
    }

    fn from_params(params: Vec<BoundedParam>) -> Result<Self, GprError> {
        let log_lengthscales: Vec<f64> = params.iter().map(|p| p.ln()).collect();
        let inv_ell_sq = inv_ell_sq_from_log(&log_lengthscales)?;
        Ok(Self {
            params,
            log_lengthscales,
            inv_ell_sq,
        })
    }

    /// Rebuilds every `ℓ_d` with the same open interval.
    ///
    /// # Errors
    ///
    /// Returns [`crate::IntervalError`] if any current `ℓ_d` is not strictly
    /// inside `interval`.
    pub fn with_bounds(self, interval: Interval) -> Result<Self, crate::IntervalError> {
        let mut params = Vec::with_capacity(self.params.len());
        for param in self.params {
            params.push(param.with_interval(interval)?);
        }
        let log_lengthscales: Vec<f64> = params.iter().map(|p| p.ln()).collect();
        let inv_ell_sq = inv_ell_sq_from_log(&log_lengthscales).map_err(|_| {
            crate::IntervalError::OutOfRange {
                value: params[0].value(),
                lo: interval.lo(),
                hi: interval.hi(),
            }
        })?;
        Ok(Self {
            params,
            log_lengthscales,
            inv_ell_sq,
        })
    }

    pub(crate) fn write_intervals(&self, out: &mut [Interval], offset: &mut usize) {
        for param in &self.params {
            out[*offset] = param.interval();
            *offset += 1;
        }
    }

    /// Returns the number of optimizer parameters, equal to the feature dimension.
    pub fn num_params(&self) -> usize {
        self.log_lengthscales.len()
    }

    /// Returns `θ_d = log(ℓ_d)`.
    pub fn log_lengthscales(&self) -> &[f64] {
        &self.log_lengthscales
    }

    /// Returns `1 / ℓ_d²` for the inner ARD distance loop.
    pub(crate) fn inv_ell_sq(&self) -> &[f64] {
        &self.inv_ell_sq
    }

    /// Returns `ℓ_d = exp(θ_d)` for feature `dim`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if `dim` is out of range.
    pub fn lengthscale(&self, dim: usize) -> Result<f64, GprError> {
        self.params
            .get(dim)
            .map(|param| param.value())
            .ok_or_else(|| GprError::InvalidHyperparameter {
                reason: format!(
                    "lengthscale dimension {dim} is out of range (d={})",
                    self.num_params()
                ),
            })
    }

    /// Writes `θ_d` into `out`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if `out` is the wrong length.
    pub fn get_params(&self, out: &mut [f64]) -> Result<(), GprError> {
        crate::data::require_count(out.len(), self.num_params(), "ARD lengthscale parameters")?;
        out.copy_from_slice(&self.log_lengthscales);
        Ok(())
    }

    /// Replaces `θ_d` from `params`. The previous values are kept on error.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if `params` is the wrong
    /// length or a `θ_d` is invalid.
    pub fn set_params(&mut self, params: &[f64]) -> Result<(), GprError> {
        crate::data::require_count(
            params.len(),
            self.num_params(),
            "ARD lengthscale parameters",
        )?;
        let mut next = Vec::with_capacity(self.params.len());
        for (i, param) in self.params.iter().enumerate() {
            let log = validate_log_lengthscale(params[i])?;
            next.push(BoundedParam::new(log.exp(), param.interval())?);
        }
        *self = Self::from_params(next)?;
        Ok(())
    }
}

pub(crate) fn validate_lengthscale(lengthscale: f64) -> Result<(), GprError> {
    if !lengthscale.is_finite() {
        return Err(invalid_length("lengthscale must be finite"));
    }
    if lengthscale <= 0.0 {
        return Err(invalid_length("lengthscale must be positive"));
    }
    Ok(())
}

pub(crate) fn validate_log_lengthscale(theta: f64) -> Result<f64, GprError> {
    if !theta.is_finite() {
        return Err(invalid_length("log lengthscale must be finite"));
    }
    let lengthscale = theta.exp();
    if !lengthscale.is_finite() {
        return Err(invalid_length(
            "lengthscale overflowed to a non-finite value",
        ));
    }
    if lengthscale <= 0.0 {
        return Err(invalid_length("lengthscale underflowed to zero"));
    }
    Ok(theta)
}

fn inv_ell_sq_from_log(log_lengthscales: &[f64]) -> Result<Vec<f64>, GprError> {
    let mut inv_ell_sq = Vec::with_capacity(log_lengthscales.len());
    for &theta in log_lengthscales {
        let ell = theta.exp();
        let ell_sq = ell * ell;
        if !ell_sq.is_finite() || ell_sq <= 0.0 {
            return Err(invalid_length(
                "lengthscale overflowed to a non-finite value",
            ));
        }
        inv_ell_sq.push(1.0 / ell_sq);
    }
    Ok(inv_ell_sq)
}

fn invalid_length(reason: &'static str) -> GprError {
    GprError::InvalidHyperparameter {
        reason: reason.to_owned(),
    }
}

fn require_non_empty(len: usize) -> Result<(), GprError> {
    if len == 0 {
        Err(invalid_length("ARD lengthscales must be non-empty"))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::ArdLengthscales;
    use crate::error::GprError;

    const TOL: f64 = 1e-12;

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
        assert_send_sync::<ArdLengthscales>();
    }

    #[test]
    fn new_and_log_roundtrip() {
        let scales = ArdLengthscales::new(&[1.5, 0.25]).expect("valid");
        assert_eq!(scales.num_params(), 2);
        assert_close(scales.lengthscale(0).expect("dim 0"), 1.5);
        assert_close(scales.lengthscale(1).expect("dim 1"), 0.25);
        let from_log =
            ArdLengthscales::from_log_lengthscales(&[1.5_f64.ln(), 0.25_f64.ln()]).expect("valid");
        assert_eq!(scales, from_log);
    }

    #[test]
    fn get_set_params_is_atomic() {
        let mut scales = ArdLengthscales::new(&[1.0, 2.0]).expect("valid");
        let mut params = [0.0; 2];
        scales.get_params(&mut params).expect("len 2");
        assert_close(params[0], 1.0_f64.ln());
        assert_close(params[1], 2.0_f64.ln());
        params[1] = 4.0_f64.ln();
        scales.set_params(&params).expect("valid");
        assert_close(scales.lengthscale(1).expect("dim 1"), 4.0);
        let before = scales.clone();
        assert!(scales.set_params(&[0.0, f64::INFINITY]).is_err());
        assert_eq!(scales, before);
    }

    #[test]
    fn rejects_empty_and_non_positive() {
        assert!(matches!(
            ArdLengthscales::new(&[]),
            Err(GprError::InvalidHyperparameter { .. })
        ));
        assert!(matches!(
            ArdLengthscales::new(&[1.0, 0.0]),
            Err(GprError::InvalidHyperparameter { .. })
        ));
        assert!(matches!(
            ArdLengthscales::from_log_lengthscales(&[f64::INFINITY]),
            Err(GprError::InvalidHyperparameter { .. })
        ));
        let scales = ArdLengthscales::new(&[1.0]).expect("valid");
        assert!(matches!(
            scales.lengthscale(1),
            Err(GprError::InvalidHyperparameter { .. })
        ));
    }
}
