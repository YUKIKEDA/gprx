//! Recoverable failures from gprx operations.

use thiserror::Error;

/// Stage at which a Cholesky factorization failed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CholeskyStage {
    /// Factorization during batch fit.
    Fit,
    /// Factorization while forming a prediction.
    Predict,
    /// Incremental insert on an online model.
    OnlineInsert,
    /// Incremental delete on an online model.
    OnlineDelete,
}

/// Error type returned by gprx operations.
///
/// Recoverable failures from user input (for example
/// [`GpError::DimensionMismatch`]) and from the model or data (for example
/// [`GpError::CholeskyFailed`]) both use this type. Library paths return
/// [`Result`] instead of panicking.
///
/// Display text is English, matching crate identifiers and rustdoc.
///
/// # Examples
///
/// ```rust
/// use gprx::GpError;
///
/// fn require_fitted(fitted: bool) -> Result<(), GpError> {
///     if !fitted {
///         return Err(GpError::NotFitted);
///     }
///     Ok(())
/// }
///
/// # fn main() -> Result<(), GpError> {
/// require_fitted(true)?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, Error, PartialEq)]
pub enum GpError {
    /// Input feature dimension does not match the fitted model.
    #[error("input dimension mismatch: X.ncols()={x_dim}, expected {expected_dim}")]
    DimensionMismatch {
        /// Number of columns in the provided `X`.
        x_dim: usize,
        /// Feature dimension expected by the model.
        expected_dim: usize,
    },
    /// Too few observations for the requested operation.
    #[error("insufficient data: n={n}, at least {min} points required")]
    InsufficientData {
        /// Number of points provided.
        n: usize,
        /// Minimum number of points required.
        min: usize,
    },
    /// An input array or matrix was empty.
    #[error("input is empty")]
    EmptyInput,
    /// `predict` or another post-fit operation ran before a successful `fit`.
    #[error("model is not fitted; call fit first")]
    NotFitted,
    /// User-provided values contained `NaN` or `Inf`.
    #[error("input contains a non-finite value (NaN/Inf)")]
    NonFiniteInput,
    /// A kernel evaluation produced `NaN` or `Inf`.
    #[error("kernel evaluation produced a non-finite value")]
    NonFiniteKernelValue,
    /// Cholesky factorization of `A = K + noise` failed after applying jitter.
    #[error(
        "Cholesky factorization failed (stage={stage:?}, size={matrix_size}, jitter={jitter} already applied)"
    )]
    CholeskyFailed {
        /// Jitter already added to the diagonal when factorization failed.
        jitter: f64,
        /// Order of the matrix that failed to factor.
        matrix_size: usize,
        /// Call site of the failed factorization.
        stage: CholeskyStage,
    },
    /// A matrix that must be positive definite (or semidefinite) is not.
    #[error("matrix is not positive semidefinite")]
    NonPositiveDefiniteMatrix,
    /// Mixed-precision iterative refinement did not meet the residual tolerance.
    #[error(
        "mixed-precision iterative refinement did not converge after {iterations} iterations (residual norm={residual_norm})"
    )]
    RefinementNotConverged {
        /// Number of refinement iterations performed.
        iterations: usize,
        /// Residual norm at termination.
        residual_norm: f64,
    },
    /// The kernel does not implement `grad_wrt_coord_dim`.
    #[error(
        "this kernel term does not implement Sparse GP coordinate derivatives (grad_wrt_coord_dim)"
    )]
    CoordGradientUnsupported,
    /// The hyperparameter optimizer stopped without meeting its convergence test.
    #[error("optimizer did not converge after {iterations} iterations")]
    OptimizationNotConverged {
        /// Number of optimizer iterations performed.
        iterations: usize,
    },
    /// A kernel hyperparameter is outside its valid domain.
    #[error("invalid hyperparameter: {reason}")]
    InvalidHyperparameter {
        /// Why the value is invalid.
        reason: String,
    },
    /// Observation-noise variance is outside its valid domain.
    #[error("invalid observation noise variance: {reason}")]
    InvalidNoiseVariance {
        /// Why the value is invalid.
        reason: String,
    },
    /// The requested kernel operation is not implemented for this term.
    #[error("unsupported kernel operation: {reason}")]
    UnsupportedKernelOperation {
        /// Why the operation is unsupported.
        reason: String,
    },
    /// A workspace buffer is smaller than the current problem size.
    #[error("workspace capacity is insufficient")]
    WorkspaceTooSmall,
    /// An online-learning point identifier is not in the current model.
    #[error("the given PointId does not exist")]
    InvalidPointId,
}

#[cfg(test)]
mod tests {
    use super::{CholeskyStage, GpError};

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn error_is_send_sync() {
        assert_send_sync::<GpError>();
        assert_send_sync::<CholeskyStage>();
    }

    #[test]
    fn display_matches_english_messages() {
        assert_eq!(
            GpError::DimensionMismatch {
                x_dim: 3,
                expected_dim: 2
            }
            .to_string(),
            "input dimension mismatch: X.ncols()=3, expected 2"
        );
        assert_eq!(
            GpError::NotFitted.to_string(),
            "model is not fitted; call fit first"
        );
        let chol = GpError::CholeskyFailed {
            jitter: 1e-6,
            matrix_size: 4,
            stage: CholeskyStage::Fit,
        };
        assert_eq!(
            chol.to_string(),
            format!(
                "Cholesky factorization failed (stage=Fit, size=4, jitter={jitter} already applied)",
                jitter = 1e-6
            )
        );
    }

    #[test]
    fn error_trait_is_implemented() {
        let err = GpError::EmptyInput;
        let _: &dyn std::error::Error = &err;
    }
}
