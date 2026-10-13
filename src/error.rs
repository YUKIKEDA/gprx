//! Recoverable failures from gprx operations.

use thiserror::Error;

/// Names the stage at which a Cholesky factorization failed
/// ([`GprError::CholeskyFailed`]).
///
/// # Examples
///
/// ```rust
/// use gprx::CholeskyStage;
///
/// let stage = CholeskyStage::Predict;
/// assert_eq!(format!("{stage:?}"), "Predict");
///
/// // A caller that reports where a factorization failed matches every stage.
/// fn during(stage: CholeskyStage) -> &'static str {
///     match stage {
///         CholeskyStage::Fit => "a batch fit",
///         CholeskyStage::Predict => "a prediction",
///         CholeskyStage::OnlineInsert => "an online insert",
///         CholeskyStage::OnlineDelete => "an online delete",
///         _ => "another stage",
///     }
/// }
/// assert_eq!(during(stage), "a prediction");
/// assert_eq!(during(CholeskyStage::Fit), "a batch fit");
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
/// Names the part of a save or load that failed ([`GprError::PersistFailed`]).
///
/// # Examples
///
/// ```rust
/// use gprx::kernel::{KernelSpec, RbfKernel};
/// use gprx::persist::LoadedSgpr;
/// use gprx::{GaussianLikelihood, Gpr, GprError, PersistErrorKind, PersistRegistry};
///
/// // A caller branches on the kind; `reason` is for a person to read.
/// fn advice(kind: PersistErrorKind) -> &'static str {
///     match kind {
///         PersistErrorKind::Io => "check the path and its permissions",
///         PersistErrorKind::Config | PersistErrorKind::Tensor => "the files are damaged",
///         PersistErrorKind::InvalidPersistId
///         | PersistErrorKind::NotPersistable
///         | PersistErrorKind::UnregisteredId => "check the custom kernels and transforms",
///         PersistErrorKind::WrongModel => "load it with the loader of its model",
///         _ => "see the reason",
///     }
/// }
///
/// # fn main() -> Result<(), GprError> {
/// assert_eq!(PersistErrorKind::Io.to_string(), "io");
///
/// let fitted = Gpr::new(
///     KernelSpec::from(RbfKernel::new(1.0)?),
///     GaussianLikelihood::new(0.1)?,
/// )
/// .fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
/// .map_err(|(_, e)| e)?;
/// let dir = std::env::temp_dir().join(format!("gprx-doc-persist-kind-{}", std::process::id()));
/// let _ = std::fs::remove_dir_all(&dir);
/// fitted.save(&dir)?;
/// // The directory holds an exact GPR, not a sparse one.
/// let refused = LoadedSgpr::load(&dir, &PersistRegistry::new());
/// let _ = std::fs::remove_dir_all(&dir);
/// let kind = match refused {
///     Err(GprError::PersistFailed { kind, .. }) => Some(kind),
///     _ => None,
/// };
/// assert_eq!(kind, Some(PersistErrorKind::WrongModel));
/// assert_eq!(kind.map(advice), Some("load it with the loader of its model"));
/// # Ok(())
/// # }
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

/// Names what is wrong with how a call's [`crate::kernel::DistanceSource`]s
/// match the slots of the kernel ([`GprError::DistanceSlot`]).
///
/// # Examples
///
/// ```rust
/// use gprx::kernel::{RbfKernel, ScalarDistance};
/// use gprx::{Fixed, GaussianLikelihood, Gpr, GprError, SlotErrorKind};
///
/// # fn main() -> Result<(), GprError> {
/// let image = ScalarDistance::new();
/// let other = ScalarDistance::new();
/// let train = [0.0, 1.0, 1.0, 0.0];
/// let fitted = Gpr::new(image.kernel(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?)
///     .with_optimizer(Fixed)
///     .factor([image.borrow(&train)], 2, &[0.0, 1.0])
///     .map_err(|(_, e)| e)?;
/// // `other` is not a slot of the kernel.
/// let refused = fitted.predict([other.from_vec(vec![0.25, 0.25])], 1);
/// assert!(matches!(
///     refused,
///     Err(GprError::DistanceSlot { kind: SlotErrorKind::NotRead, slot: None })
/// ));
/// // Two sources for slot 0.
/// let twice = fitted.predict(
///     [image.from_vec(vec![0.25, 0.25]), image.from_vec(vec![0.25, 0.25])],
///     1,
/// );
/// assert!(matches!(
///     twice,
///     Err(GprError::DistanceSlot { kind: SlotErrorKind::Duplicate, slot: Some(0) })
/// ));
/// // No source for slot 0.
/// let none = fitted.predict(std::iter::empty(), 1);
/// assert!(matches!(
///     none,
///     Err(GprError::DistanceSlot { kind: SlotErrorKind::Missing, slot: Some(0) })
/// ));
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum SlotErrorKind {
    /// A source names a slot the kernel does not read. A loaded model reads
    /// new slots ([`crate::persist::LoadedDistanceGpr::slots`] and its
    /// twins), so a slot of the model before it was saved is one.
    NotRead,
    /// Two sources name the same slot.
    Duplicate,
    /// A slot of the kernel has no source.
    Missing,
}

impl std::fmt::Display for SlotErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::NotRead => "squared distances were supplied for a slot the kernel does not read",
            Self::Duplicate => "two sources were supplied for one distance slot",
            Self::Missing => "a distance slot of the kernel has no source",
        })
    }
}

/// Where an [`GprError::InvalidDistance`] is, as its message shows it: the
/// parts that are known, in the order slot, dimension, pair.
struct DistancePlace<'a> {
    slot: &'a Option<usize>,
    dim: &'a Option<usize>,
    pair: &'a Option<(usize, usize)>,
}

impl std::fmt::Display for DistancePlace<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(slot) = self.slot {
            write!(f, " in slot {slot}")?;
        }
        if let Some(dim) = self.dim {
            let lead = if self.slot.is_some() { "," } else { " in" };
            write!(f, "{lead} dimension {dim}")?;
        }
        if let Some((row, col)) = self.pair {
            write!(f, " at ({row}, {col})")?;
        }
        Ok(())
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
/// The type is `#[non_exhaustive]`: a `match` needs a wildcard arm, and a
/// later 0.x release may add variants.
///
/// # Examples
///
/// ```rust
/// use gprx::kernel::{KernelSpec, RbfKernel};
/// use gprx::{GaussianLikelihood, Gpr, GprError, JitterPolicy};
///
/// // A caller's own check can return the crate's error.
/// fn reject_empty(n: usize) -> Result<(), GprError> {
///     if n == 0 {
///         return Err(GprError::EmptyInput);
///     }
///     Ok(())
/// }
///
/// // Groups every variant by what the caller fixes.
/// fn advice(err: &GprError) -> String {
///     match err {
///         // The data or arguments of the call.
///         GprError::DimensionMismatch { x_dim, expected_dim } => {
///             format!("pass {expected_dim} feature columns, not {x_dim}")
///         }
///         GprError::InsufficientData { n, min } => format!("keep at least {min} points, not {n}"),
///         GprError::EmptyInput | GprError::NonFiniteInput | GprError::SizeOverflow => {
///             format!("check the input values and sizes: {err}")
///         }
///         GprError::ShapeMismatch { reason }
///         | GprError::LengthMismatch { reason }
///         | GprError::IndexOutOfRange { reason } => format!("fix the call: {reason}"),
///         GprError::InvalidDistance { slot, dim, pair, reason } => {
///             format!("repair the squared distances (slot {slot:?}, dim {dim:?}, pair {pair:?}): {reason}")
///         }
///         GprError::DistanceSlot { kind, slot } => format!("match the sources to the slots ({slot:?}): {kind}"),
///         GprError::InvalidPointId | GprError::InvalidInducingId => "use an id the model holds".to_owned(),
///         // The settings of the model.
///         GprError::InvalidHyperparameter { reason }
///         | GprError::InvalidNoiseVariance { reason }
///         | GprError::InvalidConfig { reason }
///         | GprError::UnsupportedKernelOperation { reason } => format!("change the setting: {reason}"),
///         GprError::InvalidInterval(interval) => format!("change the bounds or the value: {interval}"),
///         GprError::CoordGradientUnsupported => "pick a kernel with the derivative".to_owned(),
///         // The numerics of the model at this data.
///         GprError::CholeskyFailed { jitter, matrix_size, stage } => {
///             format!("raise the noise or the jitter ({stage:?}, n = {matrix_size}, jitter {jitter})")
///         }
///         GprError::NonFiniteKernelValue
///         | GprError::NonPositiveDefiniteMatrix
///         | GprError::WorkspaceTooSmall => format!("the model cannot evaluate here: {err}"),
///         GprError::OptimizationNotConverged { iterations } => {
///             format!("try another start or optimizer (after {iterations} iterations)")
///         }
///         // Save and load.
///         GprError::PersistFailed { kind, reason } => format!("save or load ({kind}): {reason}"),
///         GprError::UnsupportedPersistVersion { found, supported } => {
///             format!("the files are version {found}; this crate reads {supported}")
///         }
///         _ => err.to_string(),
///     }
/// }
///
/// # fn main() -> Result<(), GprError> {
/// reject_empty(1)?;
/// assert_eq!(reject_empty(0), Err(GprError::EmptyInput));
///
/// let negative = GaussianLikelihood::new(-1.0);
/// assert!(matches!(negative, Err(GprError::InvalidNoiseVariance { .. })));
/// // Positive, but outside the default interval (1e-5, 1e5).
/// let tiny = GaussianLikelihood::new(1e-6);
/// assert!(matches!(tiny, Err(GprError::InvalidInterval(_))));
/// let jitter = JitterPolicy::fixed(-1.0);
/// assert!(matches!(jitter, Err(GprError::InvalidConfig { .. })));
///
/// let fitted = Gpr::new(
///     KernelSpec::from(RbfKernel::new(1.0)?),
///     GaussianLikelihood::new(0.1)?,
/// )
/// .fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
/// .map_err(|(_, e)| e)?;
/// // The model was fit on one feature; the query has two.
/// let wrong = fitted.predict(&[0.5, 0.5], 1, 2);
/// assert!(matches!(wrong, Err(GprError::DimensionMismatch { x_dim: 2, expected_dim: 1 })));
/// if let Err(err) = wrong {
///     assert_eq!(advice(&err), "pass 1 feature columns, not 2");
/// }
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
    /// Reports that a Cholesky factorization failed after the retries the
    /// [`crate::JitterPolicy`] allows.
    ///
    /// The matrix is usually `A = K + σn² I`; a posterior draw factors the
    /// predictive covariance instead.
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
    /// Reports that the hyperparameter search found no usable point.
    ///
    /// The solver of the first run stopped with an error of its own (not a
    /// [`GprError`] of the objective), the best point of a run is one where
    /// the value, gradient, or Hessian was not finite, no run reached a
    /// finite objective value,
    /// or an optimizer returned a result of the wrong length or with a
    /// non-finite value. A built-in optimizer that stops at its iteration cap
    /// returns its best point instead of this error.
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
    /// A supplied squared distance is not finite, is negative, is a
    /// non-zero diagonal, or differs from its mirror entry, past what the
    /// source's repair (if any) allows.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{RbfKernel, ScalarDistance};
    /// use gprx::{Fixed, GaussianLikelihood, Gpr, GprError};
    ///
    /// # fn main() -> Result<(), GprError> {
    /// let image = ScalarDistance::new();
    /// // The pair (1, 0) is negative.
    /// let train = [0.0, -1.0, 1.0, 0.0];
    /// let refused = Gpr::new(image.kernel(RbfKernel::new(1.0)?), GaussianLikelihood::new(0.1)?)
    ///     .with_optimizer(Fixed)
    ///     .factor([image.borrow(&train)], 2, &[0.0, 1.0]);
    /// assert!(matches!(
    ///     refused,
    ///     Err((_, GprError::InvalidDistance { slot: Some(0), dim: None, pair: Some((1, 0)), .. }))
    /// ));
    /// # Ok(())
    /// # }
    /// ```
    #[error("invalid squared distance{}: {reason}", DistancePlace { slot, dim, pair })]
    InvalidDistance {
        /// Holds the slot of the table: its place in the kernel's `slots()`.
        /// `None` when the check that refused the value could not name it.
        slot: Option<usize>,
        /// Holds the dimension of the table in an ARD slot; `None` for a
        /// scalar slot.
        dim: Option<usize>,
        /// Holds the pair `(row, col)` in its table: the row and column of
        /// the block the caller passed. `None` when a check refused the
        /// table without locating a value.
        pair: Option<(usize, usize)>,
        /// Holds what is wrong with it.
        reason: String,
    },
    /// The sources of a call do not match the slots of the kernel: one names
    /// a slot the kernel does not read, two name one slot, or a slot has
    /// none. See [`SlotErrorKind`] for an example.
    #[error(
        "distance slot mismatch{}: {kind}",
        DistancePlace { slot, dim: &None, pair: &None }
    )]
    DistanceSlot {
        /// Records what is wrong.
        kind: SlotErrorKind,
        /// Holds the slot with two sources or none: its place in the
        /// kernel's `slots()`. `None` for [`SlotErrorKind::NotRead`].
        slot: Option<usize>,
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
    /// Reports that an [`crate::Interval`] or [`crate::BoundedParam`] could not be built.
    ///
    /// A hyperparameter or noise variance that leaves its interval (for
    /// example [`crate::GaussianLikelihood::new`] below `1e-5`) also returns
    /// this.
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
    #[error("unsupported persist format version {found}; this crate reads version {supported}")]
    UnsupportedPersistVersion {
        /// Holds the version written in the file.
        found: u32,
        /// Holds the version this crate reads.
        supported: u32,
    },
}

impl GprError {
    /// Names the slot (its place in the kernel's slots) of an
    /// [`Self::InvalidDistance`] that does not name one yet; any other
    /// error as it is.
    pub(crate) fn in_slot(mut self, place: usize) -> Self {
        if let Self::InvalidDistance { slot, .. } = &mut self {
            slot.get_or_insert(place);
        }
        self
    }

    /// Names the ARD dimension of an [`Self::InvalidDistance`] that does
    /// not name one yet; any other error as it is.
    pub(crate) fn in_dim(mut self, k: usize) -> Self {
        if let Self::InvalidDistance { dim, .. } = &mut self {
            dim.get_or_insert(k);
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::{CholeskyStage, GprError};
    use super::{PersistErrorKind, SlotErrorKind};

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
    fn a_distance_error_shows_the_parts_of_its_place_it_knows() {
        let at = |slot, dim, pair| {
            GprError::InvalidDistance {
                slot,
                dim,
                pair,
                reason: "-1 is negative".to_owned(),
            }
            .to_string()
        };
        assert_eq!(
            at(Some(1), Some(2), Some((3, 0))),
            "invalid squared distance in slot 1, dimension 2 at (3, 0): -1 is negative"
        );
        assert_eq!(
            at(None, Some(2), None),
            "invalid squared distance in dimension 2: -1 is negative"
        );
        assert_eq!(
            at(None, None, None),
            "invalid squared distance: -1 is negative"
        );
        let err = GprError::InvalidDistance {
            slot: None,
            dim: None,
            pair: Some((3, 0)),
            reason: String::new(),
        };
        assert_eq!(err.in_dim(2).in_slot(1).in_slot(4), {
            GprError::InvalidDistance {
                slot: Some(1),
                dim: Some(2),
                pair: Some((3, 0)),
                reason: String::new(),
            }
        });
        assert_eq!(GprError::EmptyInput.in_slot(0), GprError::EmptyInput);
        assert_eq!(
            GprError::DistanceSlot {
                kind: SlotErrorKind::Duplicate,
                slot: Some(0)
            }
            .to_string(),
            "distance slot mismatch in slot 0: two sources were supplied for one distance slot"
        );
        assert_eq!(
            GprError::DistanceSlot {
                kind: SlotErrorKind::NotRead,
                slot: None
            }
            .to_string(),
            format!("distance slot mismatch: {}", SlotErrorKind::NotRead)
        );
    }

    #[test]
    fn error_trait_is_implemented() {
        let err = GprError::EmptyInput;
        let _: &dyn std::error::Error = &err;
    }
}
