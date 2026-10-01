//! Stable training-point identities of [`crate::OnlineGpr`] and
//! [`crate::OnlineSgpr`], and the registry that keeps any such identifier.

use std::collections::HashMap;

use crate::error::GprError;
use crate::error::PersistErrorKind;
use crate::persist::persist_err;

/// Stable identity of one training point on [`crate::OnlineGpr`].
///
/// [`crate::FittedGpr::into_online`] assigns identifiers `0 .. n-1` in buffer
/// order. Later [`crate::OnlineGpr::insert`] values increase monotonically and
/// are never reused after [`crate::OnlineGpr::delete`]. There is no public
/// constructor.
///
/// # Examples
///
/// ```rust
/// use gprx::kernel::{KernelSpec, RbfKernel};
/// use gprx::{Fixed, GaussianLikelihood, Gpr};
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let fitted = Gpr::new(
///     KernelSpec::from(RbfKernel::new(1.0)?),
///     GaussianLikelihood::new(0.1)?,
/// )
/// .with_optimizer(Fixed)
/// .factor(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
/// .map_err(|(_, e)| e)?;
/// let mut online = fitted.into_online()?;
/// let id = online.insert(&[1.5], 0.5)?;
/// assert_eq!(online.point_ids().last().copied(), Some(id));
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PointId(u64);

/// A stable identifier kept by an [`IdRegistry`].
pub(crate) trait RegistryId: Copy + Eq + std::hash::Hash {
    /// The config key the identifiers are persisted under.
    const PERSIST_KEY: &'static str;

    fn from_raw(raw: u64) -> Self;

    fn raw(self) -> u64;

    /// The error for an identifier that is unknown or already removed.
    fn unknown() -> GprError;
}

impl RegistryId for PointId {
    const PERSIST_KEY: &'static str = "point_ids";

    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    fn raw(self) -> u64 {
        self.0
    }

    fn unknown() -> GprError {
        GprError::InvalidPointId
    }
}

/// Identifiers in buffer order and the next one to hand out. Handed-out
/// identifiers increase monotonically and are never reused.
#[derive(Clone, Debug)]
pub(crate) struct IdRegistry<I: RegistryId> {
    id_to_index: HashMap<I, usize>,
    index_to_id: Vec<I>,
    next_id: u64,
}

/// Training-point identities of the online models.
pub(crate) type PointRegistry = IdRegistry<PointId>;

impl<I: RegistryId> IdRegistry<I> {
    /// Identifiers `0 .. n-1` in buffer order.
    pub(crate) fn from_count(n: usize) -> Self {
        let index_to_id: Vec<I> = (0..n as u64).map(I::from_raw).collect();
        let id_to_index = index_to_id
            .iter()
            .copied()
            .enumerate()
            .map(|(index, id)| (id, index))
            .collect();
        Self {
            id_to_index,
            index_to_id,
            next_id: n as u64,
        }
    }

    /// Identifiers read back from a persist directory.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::PersistFailed`] when an identifier repeats or
    /// `next_id` does not exceed every stored one.
    pub(crate) fn from_persisted(ids: &[u64], next_id: u64) -> Result<Self, GprError> {
        let key = I::PERSIST_KEY;
        let mut id_to_index = HashMap::with_capacity(ids.len());
        let mut index_to_id = Vec::with_capacity(ids.len());
        let mut max_id = None;
        for (index, &raw) in ids.iter().enumerate() {
            let id = I::from_raw(raw);
            if id_to_index.insert(id, index).is_some() {
                return Err(persist_err(
                    PersistErrorKind::Config,
                    format!("config has duplicate {key}"),
                ));
            }
            index_to_id.push(id);
            max_id = Some(max_id.map_or(raw, |seen: u64| seen.max(raw)));
        }
        if let Some(max_id) = max_id {
            if next_id <= max_id {
                return Err(persist_err(
                    PersistErrorKind::Config,
                    format!("config next id must exceed every stored value of {key}"),
                ));
            }
        }
        Ok(Self {
            id_to_index,
            index_to_id,
            next_id,
        })
    }

    pub(crate) fn ids(&self) -> &[I] {
        &self.index_to_id
    }

    /// The identifiers as raw values, for persist.
    pub(crate) fn raw_ids(&self) -> Vec<u64> {
        self.index_to_id.iter().map(|id| id.raw()).collect()
    }

    pub(crate) fn next_id(&self) -> u64 {
        self.next_id
    }

    pub(crate) fn len(&self) -> usize {
        self.index_to_id.len()
    }

    pub(crate) fn index_of(&self, id: I) -> Result<usize, GprError> {
        self.id_to_index.get(&id).copied().ok_or_else(I::unknown)
    }

    pub(crate) fn insert(&mut self) -> I {
        let id = I::from_raw(self.next_id);
        let index = self.index_to_id.len();
        self.next_id = self.next_id.saturating_add(1);
        self.index_to_id.push(id);
        self.id_to_index.insert(id, index);
        id
    }

    pub(crate) fn remove_at(&mut self, index: usize) {
        let id = self.index_to_id.remove(index);
        self.id_to_index.remove(&id);
        for (shifted, remaining) in self.index_to_id.iter().enumerate().skip(index) {
            self.id_to_index.insert(*remaining, shifted);
        }
    }
}
