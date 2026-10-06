//! Recoverable failures from gprx operations.

use thiserror::Error;

/// Stage at which a Cholesky factorization failed.
///
/// # Examples
///
/// ```rust
/// use gprx::CholeskyStage;
///
/// let stage = CholeskyStage::Predict;
/// assert_eq!(format!("{stage:?}"), "Predict");
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum CholeskyStage {
    /// Marks a factorization during batch fit.
    Fit,
    /// Marks a factorization while forming a prediction.
    Predict,
    /// Marks an incremental insert on an online model.
    OnlineInsert,
    /// Marks an incremental delete on an online model.
    OnlineDelete,
}
/// The part of a save or load that failed ([`GprError::PersistFailed`]).
///
/// # Examples
///
/// ```rust
/// use gprx::PersistErrorKind;
///
/// assert_eq!(PersistErrorKind::Io.to_string(), "io");
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum PersistErrorKind {
    /// Marks the file system: creating the directory, or reading, writing, renaming, opening, or mapping a file.
    Io,
    /// `config.json` is not valid JSON, misses a key, or holds values that disagree with each other (for example point ids and `n`).
    Config,
    /// `model.safetensors` is damaged: a missing tensor, a wrong dtype or shape, a misaligned or out-of-range buffer, or values the model cannot hold.
    Tensor,
    /// A `persist_id` is empty, uses the reserved prefix, or is registered twice.
    InvalidPersistId,
    /// A custom kernel or transform has no persist form: it does not implement `persist_id` or `persist_state`.
    NotPersistable,
    /// A saved custom kernel or transform names a `persist_id` the [`crate::PersistRegistry`] passed to the load does not hold.
    UnregisteredId,
    /// The directory holds another kind of model than the one being loaded.
    WrongModel,
}

impl std::fmt::Display for PersistErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Io => "io",
            Self::Config => "config",
            Self::Tensor => "tensor",
            Self::InvalidPersistId => "invalid persist_id",
            Self::NotPersistable => "not persistable",
            Self::UnregisteredId => "unregistered persist_id",
            Self::WrongModel => "wrong model",
        })
    }
}

/// Reports a recoverable failure from a gprx operation.
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
#[non_exhaustive]
pub enum GprError {
    /// Input feature dimension does not match the fitted model.
    #[error("input dimension mismatch: X.ncols()={x_dim}, expected {expected_dim}")]
    DimensionMismatch {
        /// Holds the number of columns in the provided `X`.
        x_dim: usize,
        /// Holds the feature dimension expected by the model.
        expected_dim: usize,
    },
    /// Reports too few observations for the requested operation.
    #[error("insufficient data: n={n}, at least {min} points required")]
    InsufficientData {
        /// Holds the number of points provided.
        n: usize,
        /// Holds the minimum number of points required.
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
    /// The kernel has no derivative a sparse model needs.
    ///
    /// Every built-in kernel and every Sum / Product tree of them has all of them, except
    /// Matérn with `ν = 1/2`, whose coordinate derivative (`FreeInducing`) is undefined where
    /// two points coincide, as `Z ⊂ X` starts. A `Custom` leaf has them only when it implements
    /// the corresponding `KernelTerm` methods (`grad_cross` / `hess_cross` for `Sgpr` and
    /// `Svgp`, `grad_wrt_sq_dist*` and `hess_wrt_sq_dist` for `FreeInducing`).
    #[error(
        "this kernel does not implement the derivative a sparse model needs \
         (Matern nu = 1/2 has no coordinate derivative for FreeInducing points; \
         a Custom leaf must implement the KernelTerm cross / squared-distance derivatives)"
    )]
    CoordGradientUnsupported,
    /// Marks the hyperparameter optimizer stopped without meeting its convergence test.
    #[error("optimizer did not converge after {iterations} iterations")]
    OptimizationNotConverged {
        /// Holds the number of optimizer iterations performed.
        iterations: usize,
    },
    /// A kernel hyperparameter is outside its valid domain.
    #[error("invalid hyperparameter: {reason}")]
    InvalidHyperparameter {
        /// Reports why the value is invalid.
        reason: String,
    },
    /// A matrix argument has the wrong number of rows or columns.
    #[error("shape mismatch: {reason}")]
    ShapeMismatch {
        /// Records which matrix and which shape was expected.
        reason: String,
    },
    /// A slice argument has the wrong length.
    #[error("length mismatch: {reason}")]
    LengthMismatch {
        /// Records which slice and which length was expected.
        reason: String,
    },
    /// A parameter, dimension, leaf, or change index is out of range.
    #[error("index out of range: {reason}")]
    IndexOutOfRange {
        /// Records which index and which range.
        reason: String,
    },
    /// An optimizer, jitter policy, or transform setting is outside its domain.
    #[error("invalid configuration: {reason}")]
    InvalidConfig {
        /// Records which setting and why it is invalid.
        reason: String,
    },
    /// `n_rows × n_cols` (or another buffer size) overflows `usize`.
    #[error("size overflows usize")]
    SizeOverflow,
    /// Marks an [`crate::Interval`] or [`crate::BoundedParam`] could not be built.
    #[error(transparent)]
    InvalidInterval(#[from] crate::param::IntervalError),
    /// Observation-noise variance is outside its valid domain.
    #[error("invalid observation noise variance: {reason}")]
    InvalidNoiseVariance {
        /// Reports why the value is invalid.
        reason: String,
    },
    /// The requested kernel operation is not implemented for this term.
    #[error("unsupported kernel operation: {reason}")]
    UnsupportedKernelOperation {
        /// Reports why the operation is unsupported.
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
    #[error("persist failed ({kind}): {reason}")]
    PersistFailed {
        /// Records which part of the save or load failed, for a caller to branch on.
        kind: PersistErrorKind,
        /// Reports why the save or load could not finish, for a person to read.
        reason: String,
    },
    /// `config.json` `format_version` is not supported by this crate.
    #[error(
        "unsupported persist format version {found}; this crate reads versions 1 to {supported}"
    )]
    UnsupportedPersistVersion {
        /// Holds the version written in the file.
        found: u32,
        /// Holds the newest version this crate reads ([`crate::persist::FORMAT_VERSION`]);
        /// it reads every version from 1 up to it.
        supported: u32,
    },
}

#[cfg(test)]
mod tests {
    use super::PersistErrorKind;
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
                kind: PersistErrorKind::Tensor,
                reason: "missing l".to_owned()
            }
            .to_string(),
            "persist failed (tensor): missing l"
        );
        assert_eq!(
            GprError::UnsupportedPersistVersion {
                found: 2,
                supported: 1
            }
            .to_string(),
            "unsupported persist format version 2; this crate reads versions 1 to 1"
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
