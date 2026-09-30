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
/// [`GprError::DimensionMismatch`]) and from the model or data (for example
/// [`GprError::CholeskyFailed`]) both use this type. Library paths return
/// [`Result`] instead of panicking.
///
/// Display text is English, matching crate identifiers and rustdoc.
///
/// # Examples
///
/// ```rust
/// use gprx::GprError;
///
/// fn reject_empty(n: usize) -> Result<(), GprError> {
///     if n == 0 {
///         return Err(GprError::EmptyInput);
///     }
///     Ok(())
/// }
///
/// # fn main() -> Result<(), GprError> {
/// reject_empty(1)?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, Error, PartialEq)]
pub enum GprError {
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
    /// The kernel does not implement `grad_wrt_coord_dim`.
    #[error(
        "this kernel term does not implement Sparse GPR coordinate derivatives (grad_wrt_coord_dim)"
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
    /// A matrix argument has the wrong number of rows or columns.
    #[error("shape mismatch: {reason}")]
    ShapeMismatch {
        /// Which matrix and which shape was expected.
        reason: String,
    },
    /// A slice argument has the wrong length.
    #[error("length mismatch: {reason}")]
    LengthMismatch {
        /// Which slice and which length was expected.
        reason: String,
    },
    /// A parameter, dimension, leaf, or change index is out of range.
    #[error("index out of range: {reason}")]
    IndexOutOfRange {
        /// Which index and which range.
        reason: String,
    },
    /// An optimizer, jitter policy, or transform setting is outside its domain.
    #[error("invalid configuration: {reason}")]
    InvalidConfig {
        /// Which setting and why it is invalid.
        reason: String,
    },
    /// `n_rows × n_cols` (or another buffer size) overflows `usize`.
    #[error("size overflows usize")]
    SizeOverflow,
    /// An [`crate::Interval`] or [`crate::BoundedParam`] could not be built.
    #[error(transparent)]
    InvalidInterval(#[from] crate::param::IntervalError),
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
    /// An online inducing-point identifier is not in the current model.
    #[error("the given InducingId does not exist")]
    InvalidInducingId,
    /// Saving or loading a fitted model failed.
    #[error("persist failed: {reason}")]
    PersistFailed {
        /// Why the save or load could not finish.
        reason: String,
    },
    /// `config.json` `format_version` is not supported by this crate.
    #[error("unsupported persist format version {found}; this crate reads version {supported}")]
    UnsupportedPersistVersion {
        /// Version written in the file.
        found: u32,
        /// Version this crate reads.
        supported: u32,
    },
}

#[cfg(test)]
mod tests {
    use super::{CholeskyStage, GprError};

    use crate::test_check::assert_send_sync;

    #[test]
    fn error_is_send_sync() {
        assert_send_sync::<GprError>();
        assert_send_sync::<CholeskyStage>();
    }

    #[test]
    fn display_matches_english_messages() {
        assert_eq!(
            GprError::DimensionMismatch {
                x_dim: 3,
                expected_dim: 2
            }
            .to_string(),
            "input dimension mismatch: X.ncols()=3, expected 2"
        );
        assert_eq!(GprError::EmptyInput.to_string(), "input is empty");
        assert_eq!(
            GprError::InvalidInducingId.to_string(),
            "the given InducingId does not exist"
        );
        assert_eq!(
            GprError::PersistFailed {
                reason: "missing l".to_owned()
            }
            .to_string(),
            "persist failed: missing l"
        );
        assert_eq!(
            GprError::UnsupportedPersistVersion {
                found: 2,
                supported: 1
            }
            .to_string(),
            "unsupported persist format version 2; this crate reads version 1"
        );
        let chol = GprError::CholeskyFailed {
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
    fn classified_input_errors_display_their_kind() {
        let reason = || "expected 2 values, got 1".to_owned();
        assert_eq!(
            GprError::LengthMismatch { reason: reason() }.to_string(),
            "length mismatch: expected 2 values, got 1"
        );
        assert_eq!(
            GprError::ShapeMismatch { reason: reason() }.to_string(),
            "shape mismatch: expected 2 values, got 1"
        );
        assert_eq!(
            GprError::IndexOutOfRange { reason: reason() }.to_string(),
            "index out of range: expected 2 values, got 1"
        );
        assert_eq!(
            GprError::InvalidConfig { reason: reason() }.to_string(),
            "invalid configuration: expected 2 values, got 1"
        );
        assert_eq!(GprError::SizeOverflow.to_string(), "size overflows usize");
    }

    #[test]
    fn error_trait_is_implemented() {
        let err = GprError::EmptyInput;
        let _: &dyn std::error::Error = &err;
    }
}
