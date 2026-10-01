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

/// Rebuilds a [`CustomKernel`] from its persist JSON.
///
/// The restored leaf implements the same operations at `f32` and `f64`.
pub type KernelRestore =
    Arc<dyn Fn(&serde_json::Value) -> Result<CustomKernel, GprError> + Send + Sync>;

/// Rebuilds an unfitted input map from its persist JSON.
pub type UnfittedInputRestore =
    Arc<dyn Fn(&serde_json::Value) -> Result<Box<dyn UnfittedTransform>, GprError> + Send + Sync>;

/// Rebuilds a fitted input map from its persist JSON.
pub type FittedInputRestore =
    Arc<dyn Fn(&serde_json::Value) -> Result<Box<dyn Transform>, GprError> + Send + Sync>;

/// Rebuilds an unfitted target map from its persist JSON.
pub type UnfittedTargetRestore =
    Arc<dyn Fn(&serde_json::Value) -> Result<Box<dyn UnfittedTarget>, GprError> + Send + Sync>;

/// Rebuilds a fitted target map from its persist JSON.
pub type FittedTargetRestore =
    Arc<dyn Fn(&serde_json::Value) -> Result<Box<dyn TargetTransform>, GprError> + Send + Sync>;

/// Explicit restore table for Custom kernels and caller-defined transforms.
///
/// Built-in leaves and maps use closed tags in `config.json` and are not
/// registered here. [`Self::register_kernel`] rejects ids that start with
/// [`super::RESERVED_PREFIX`].
///
/// # Examples
///
/// ```rust
/// use gprx::persist::PersistRegistry;
///
/// let registry = PersistRegistry::new();
/// let _ = registry;
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
    /// Returns an empty registry. Built-ins do not need registration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a Custom kernel restore for `persist_id`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::PersistFailed`] if `persist_id` is empty, starts
    /// with `gprx.`, or is already registered.
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
    /// Same as [`Self::register_kernel`].
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
    /// Same as [`Self::register_kernel`].
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
    /// Same as [`Self::register_kernel`].
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
    /// Same as [`Self::register_kernel`].
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
