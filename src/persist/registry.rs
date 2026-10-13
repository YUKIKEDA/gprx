//! Explicit restore factories for Custom kernels and caller-defined transforms.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use crate::error::GprError;
use crate::error::PersistErrorKind;
use crate::kernel::CustomKernel;
use crate::transform::{TargetTransform, Transform, UnfittedTarget, UnfittedTransform};

use super::RESERVED_PREFIX;
use super::persist_err;

/// Names the shared form of a restore function for a [`CustomKernel`]: it rebuilds the leaf from the JSON its [`KernelTerm::persist_state`] wrote.
///
/// The restored leaf implements the same operations at `f32` and `f64`.
///
/// [`KernelTerm::persist_state`]: crate::kernel::KernelTerm::persist_state
///
/// See the example on [`PersistRegistry`].
pub type KernelRestore =
    Arc<dyn Fn(&serde_json::Value) -> Result<CustomKernel, GprError> + Send + Sync>;

/// Names the shared form of a restore function for an unfitted input map: it rebuilds the map from the JSON its [`UnfittedTransform::persist_state`] wrote.
///
/// See the example on [`UnfittedTransform`].
pub type UnfittedInputRestore =
    Arc<dyn Fn(&serde_json::Value) -> Result<Box<dyn UnfittedTransform>, GprError> + Send + Sync>;

/// Names the shared form of a restore function for a fitted input map: it rebuilds the map from the JSON its [`Transform::persist_state`] wrote.
///
/// See the example on [`UnfittedTransform`].
pub type FittedInputRestore =
    Arc<dyn Fn(&serde_json::Value) -> Result<Box<dyn Transform>, GprError> + Send + Sync>;

/// Names the shared form of a restore function for an unfitted target map: it rebuilds the map from the JSON its [`UnfittedTarget::persist_state`] wrote.
///
/// See the example on [`UnfittedTarget`].
pub type UnfittedTargetRestore =
    Arc<dyn Fn(&serde_json::Value) -> Result<Box<dyn UnfittedTarget>, GprError> + Send + Sync>;

/// Names the shared form of a restore function for a fitted target map: it rebuilds the map from the JSON its [`TargetTransform::persist_state`] wrote.
///
/// See the example on [`UnfittedTarget`].
pub type FittedTargetRestore =
    Arc<dyn Fn(&serde_json::Value) -> Result<Box<dyn TargetTransform>, GprError> + Send + Sync>;

/// Represents the explicit restore table for Custom kernels and caller-defined transforms.
///
/// Built-in leaves and maps use closed tags in `config.json` and are not
/// registered here. A custom kernel leaf or a caller-defined transform is
/// saved as its `persist_id` and `persist_state`; a load looks the
/// `persist_id` up here and calls the restore function on the state. Every
/// `register_*` rejects an empty id, an id that starts with
/// [`super::RESERVED_PREFIX`], and an id already registered for the same
/// kind of item.
///
/// The example below saves and loads a model with a custom kernel leaf.
/// The examples on [`UnfittedTransform`] and [`UnfittedTarget`] register
/// caller-defined input and target maps.
///
/// # Examples
///
/// ```rust
/// use std::sync::Arc;
///
/// use faer::{MatMut, MatRef};
/// use gprx::kernel::{CustomKernel, KernelScalar, KernelSpec, KernelTerm, RbfKernel, Triangle};
/// use gprx::persist::{KernelRestore, LoadedGpr, PersistRegistry};
/// use gprx::{GaussianLikelihood, Gpr, GprError, Interval, PersistErrorKind};
///
/// /// A constant kernel `k = c`, with one parameter `log c`.
/// #[derive(Clone, Debug)]
/// struct Bias {
///     log_c: f64,
/// }
///
/// /// Writes `value` into the `uplo` triangle of `out`.
/// fn fill<T: KernelScalar>(mut out: MatMut<'_, T>, uplo: Triangle, value: f64) {
///     for col in 0..out.ncols() {
///         for row in 0..out.nrows() {
///             let keep = match uplo {
///                 Triangle::Lower => row >= col,
///                 Triangle::Upper => row <= col,
///                 Triangle::Full => true,
///             };
///             if keep {
///                 out[(row, col)] = T::from_f64(value);
///             }
///         }
///     }
/// }
///
/// /// Checks that a slice holds the one parameter.
/// fn one(len: usize) -> Result<(), GprError> {
///     if len == 1 {
///         Ok(())
///     } else {
///         Err(GprError::LengthMismatch { reason: format!("bias kernel has 1 parameter, got {len}") })
///     }
/// }
///
/// /// Checks a parameter index pair (`(0, 0)` is the only one).
/// fn index(i: usize, j: usize) -> Result<(), GprError> {
///     if (i, j) == (0, 0) {
///         Ok(())
///     } else {
///         Err(GprError::IndexOutOfRange { reason: format!("bias kernel has no index ({i}, {j})") })
///     }
/// }
///
/// impl<T: KernelScalar> KernelTerm<T> for Bias {
///     fn num_params(&self) -> usize {
///         1
///     }
///     fn get_params(&self, out: &mut [f64]) -> Result<(), GprError> {
///         one(out.len())?;
///         out[0] = self.log_c;
///         Ok(())
///     }
///     fn set_params(&mut self, params: &[f64]) -> Result<(), GprError> {
///         one(params.len())?;
///         self.log_c = params[0];
///         Ok(())
///     }
///     fn bounds_into(&self, out: &mut [Interval]) -> Result<(), GprError> {
///         one(out.len())?;
///         out[0] = Interval::DEFAULT_POSITIVE;
///         Ok(())
///     }
///     fn apply(&self, _: MatRef<'_, T>, out: MatMut<'_, T>, uplo: Triangle) -> Result<(), GprError> {
///         fill(out, uplo, self.log_c.exp());
///         Ok(())
///     }
///     fn apply_cross(&self, _: MatRef<'_, T>, out: MatMut<'_, T>) -> Result<(), GprError> {
///         fill(out, Triangle::Full, self.log_c.exp());
///         Ok(())
///     }
///     fn fill_diag(&self, out: &mut [T]) -> Result<(), GprError> {
///         out.fill(T::from_f64(self.log_c.exp()));
///         Ok(())
///     }
///     // `∂k/∂log c` and `∂²k/∂(log c)²` are both `c`.
///     fn grad(&self, _: MatRef<'_, T>, d_k: MatMut<'_, T>, p: usize, uplo: Triangle) -> Result<(), GprError> {
///         index(p, 0)?;
///         fill(d_k, uplo, self.log_c.exp());
///         Ok(())
///     }
///     fn hess(&self, _: MatRef<'_, T>, d2_k: MatMut<'_, T>, i: usize, j: usize, uplo: Triangle) -> Result<(), GprError> {
///         index(i, j)?;
///         fill(d2_k, uplo, self.log_c.exp());
///         Ok(())
///     }
///     fn hess_points(&self, _: MatRef<'_, T>, d2_k: MatMut<'_, T>, i: usize, j: usize, uplo: Triangle) -> Result<(), GprError> {
///         index(i, j)?;
///         fill(d2_k, uplo, self.log_c.exp());
///         Ok(())
///     }
///     // What a save writes for this leaf, and what the restore reads back.
///     fn persist_id(&self) -> &'static str {
///         "example.bias"
///     }
///     fn persist_state(&self) -> Result<serde_json::Value, GprError> {
///         Ok(serde_json::json!({ "log_c": self.log_c }))
///     }
/// }
///
/// # fn main() -> Result<(), GprError> {
/// let kernel = KernelSpec::custom(Bias { log_c: 0.0 }) + KernelSpec::from(RbfKernel::new(1.0)?);
/// let fitted = Gpr::new(kernel, GaussianLikelihood::new(0.1)?)
///     .fit(&[0.0, 1.0, 2.0], 3, 1, &[1.0, 2.0, 1.5])
///     .map_err(|(_, e)| e)?;
/// let dir = std::env::temp_dir().join(format!("gprx-doctest-registry-{}", std::process::id()));
/// let _ = std::fs::remove_dir_all(&dir);
/// fitted.save(&dir)?;
///
/// // Loading without a restore for "example.bias" fails.
/// let missing = LoadedGpr::load(&dir, &PersistRegistry::new());
/// assert!(matches!(
///     missing,
///     Err(GprError::PersistFailed { kind: PersistErrorKind::UnregisteredId, .. })
/// ));
///
/// // One restore function, which an `Arc` lets several registries share.
/// let restore: KernelRestore = Arc::new(|state| {
///     let log_c = state["log_c"].as_f64().ok_or_else(|| GprError::PersistFailed {
///         kind: PersistErrorKind::Config,
///         reason: "bias state needs a number log_c".into(),
///     })?;
///     Ok(CustomKernel::new(Bias { log_c }))
/// });
/// let mut registry = PersistRegistry::new();
/// let shared = Arc::clone(&restore);
/// registry.register_kernel("example.bias", move |state| shared(state))?;
/// // An id registers once per kind.
/// let twice = registry.register_kernel("example.bias", move |state| restore(state));
/// assert!(matches!(
///     twice,
///     Err(GprError::PersistFailed { kind: PersistErrorKind::InvalidPersistId, .. })
/// ));
///
/// let loaded = LoadedGpr::load(&dir, &registry)?;
/// assert_eq!(loaded.predict(&[0.5], 1, 1)?, fitted.predict(&[0.5], 1, 1)?);
/// let _ = std::fs::remove_dir_all(&dir);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Default)]
pub struct PersistRegistry {
    kernels: HashMap<String, KernelRestore>,
    unfitted_inputs: HashMap<String, UnfittedInputRestore>,
    fitted_inputs: HashMap<String, FittedInputRestore>,
    unfitted_targets: HashMap<String, UnfittedTargetRestore>,
    fitted_targets: HashMap<String, FittedTargetRestore>,
}

impl PersistRegistry {
    /// Returns an empty registry.
    ///
    /// Built-ins do not need registration.
    ///
    /// See the example on [`PersistRegistry`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a Custom kernel restore for `persist_id`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::PersistFailed`] with
    /// [`PersistErrorKind::InvalidPersistId`](crate::PersistErrorKind::InvalidPersistId)
    /// if `persist_id` is empty, starts with [`super::RESERVED_PREFIX`], or
    /// is already registered as a kernel.
    ///
    /// See the example on [`PersistRegistry`].
    pub fn register_kernel(
        &mut self,
        persist_id: impl Into<String>,
        restore: impl Fn(&serde_json::Value) -> Result<CustomKernel, GprError> + Send + Sync + 'static,
    ) -> Result<(), GprError> {
        insert_unique(
            &mut self.kernels,
            persist_id.into(),
            Arc::new(restore),
            "kernel",
        )
    }

    /// Registers an unfitted input-transform restore for `persist_id`.
    ///
    /// # Errors
    ///
    /// Same as [`Self::register_kernel`], for an id already registered as
    /// an unfitted input transform.
    ///
    /// See the example on [`UnfittedTransform`].
    pub fn register_unfitted_input(
        &mut self,
        persist_id: impl Into<String>,
        restore: impl Fn(&serde_json::Value) -> Result<Box<dyn UnfittedTransform>, GprError>
        + Send
        + Sync
        + 'static,
    ) -> Result<(), GprError> {
        insert_unique(
            &mut self.unfitted_inputs,
            persist_id.into(),
            Arc::new(restore),
            "unfitted input transform",
        )
    }

    /// Registers a fitted input-transform restore for `persist_id`.
    ///
    /// # Errors
    ///
    /// Same as [`Self::register_kernel`], for an id already registered as
    /// a fitted input transform.
    ///
    /// See the example on [`UnfittedTransform`].
    pub fn register_fitted_input(
        &mut self,
        persist_id: impl Into<String>,
        restore: impl Fn(&serde_json::Value) -> Result<Box<dyn Transform>, GprError>
        + Send
        + Sync
        + 'static,
    ) -> Result<(), GprError> {
        insert_unique(
            &mut self.fitted_inputs,
            persist_id.into(),
            Arc::new(restore),
            "fitted input transform",
        )
    }

    /// Registers an unfitted target-transform restore for `persist_id`.
    ///
    /// # Errors
    ///
    /// Same as [`Self::register_kernel`], for an id already registered as
    /// an unfitted target transform.
    ///
    /// See the example on [`UnfittedTarget`].
    pub fn register_unfitted_target(
        &mut self,
        persist_id: impl Into<String>,
        restore: impl Fn(&serde_json::Value) -> Result<Box<dyn UnfittedTarget>, GprError>
        + Send
        + Sync
        + 'static,
    ) -> Result<(), GprError> {
        insert_unique(
            &mut self.unfitted_targets,
            persist_id.into(),
            Arc::new(restore),
            "unfitted target transform",
        )
    }

    /// Registers a fitted target-transform restore for `persist_id`.
    ///
    /// # Errors
    ///
    /// Same as [`Self::register_kernel`], for an id already registered as
    /// a fitted target transform.
    ///
    /// See the example on [`UnfittedTarget`].
    pub fn register_fitted_target(
        &mut self,
        persist_id: impl Into<String>,
        restore: impl Fn(&serde_json::Value) -> Result<Box<dyn TargetTransform>, GprError>
        + Send
        + Sync
        + 'static,
    ) -> Result<(), GprError> {
        insert_unique(
            &mut self.fitted_targets,
            persist_id.into(),
            Arc::new(restore),
            "fitted target transform",
        )
    }

    pub(super) fn restore_kernel(
        &self,
        persist_id: &str,
        state: &serde_json::Value,
    ) -> Result<CustomKernel, GprError> {
        lookup(&self.kernels, persist_id, "kernel")?(state)
    }

    pub(super) fn restore_unfitted_input(
        &self,
        persist_id: &str,
        state: &serde_json::Value,
    ) -> Result<Box<dyn UnfittedTransform>, GprError> {
        lookup(
            &self.unfitted_inputs,
            persist_id,
            "unfitted input transform",
        )?(state)
    }

    pub(super) fn restore_fitted_input(
        &self,
        persist_id: &str,
        state: &serde_json::Value,
    ) -> Result<Box<dyn Transform>, GprError> {
        lookup(&self.fitted_inputs, persist_id, "fitted input transform")?(state)
    }

    pub(super) fn restore_unfitted_target(
        &self,
        persist_id: &str,
        state: &serde_json::Value,
    ) -> Result<Box<dyn UnfittedTarget>, GprError> {
        lookup(
            &self.unfitted_targets,
            persist_id,
            "unfitted target transform",
        )?(state)
    }

    pub(super) fn restore_fitted_target(
        &self,
        persist_id: &str,
        state: &serde_json::Value,
    ) -> Result<Box<dyn TargetTransform>, GprError> {
        lookup(&self.fitted_targets, persist_id, "fitted target transform")?(state)
    }
}

impl fmt::Debug for PersistRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PersistRegistry")
            .field("kernels", &self.kernels.len())
            .field("unfitted_inputs", &self.unfitted_inputs.len())
            .field("fitted_inputs", &self.fitted_inputs.len())
            .field("unfitted_targets", &self.unfitted_targets.len())
            .field("fitted_targets", &self.fitted_targets.len())
            .finish()
    }
}

fn insert_unique<T>(
    map: &mut HashMap<String, T>,
    persist_id: String,
    value: T,
    kind: &str,
) -> Result<(), GprError> {
    validate_persist_id(&persist_id)?;
    if map.contains_key(&persist_id) {
        return Err(persist_err(
            PersistErrorKind::InvalidPersistId,
            format!("{kind} persist_id {persist_id:?} is already registered"),
        ));
    }
    map.insert(persist_id, value);
    Ok(())
}

fn lookup<'a, T>(
    map: &'a HashMap<String, T>,
    persist_id: &str,
    kind: &str,
) -> Result<&'a T, GprError> {
    map.get(persist_id).ok_or_else(|| {
        persist_err(
            PersistErrorKind::UnregisteredId,
            format!("{kind} persist_id {persist_id:?} is not registered"),
        )
    })
}

pub(super) fn validate_persist_id(persist_id: &str) -> Result<(), GprError> {
    if persist_id.is_empty() {
        return Err(persist_err(
            PersistErrorKind::InvalidPersistId,
            "persist_id must not be empty",
        ));
    }
    if persist_id.starts_with(RESERVED_PREFIX) {
        return Err(persist_err(
            PersistErrorKind::InvalidPersistId,
            format!("persist_id {persist_id:?} uses the reserved {RESERVED_PREFIX} prefix"),
        ));
    }
    Ok(())
}
