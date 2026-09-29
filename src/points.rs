//! Stable training-point identities of [`crate::OnlineGpr`].

use std::collections::HashMap;

use crate::error::GprError;
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

impl PointId {
    pub(crate) fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    pub(crate) fn raw(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PointRegistry {
    id_to_index: HashMap<PointId, usize>,
    index_to_id: Vec<PointId>,
    next_id: u64,
}

impl PointRegistry {
    pub(crate) fn from_count(n: usize) -> Self {
        let index_to_id: Vec<PointId> = (0..n as u64).map(PointId::from_raw).collect();
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

    pub(crate) fn from_persisted(ids: &[u64], next_id: u64) -> Result<Self, GprError> {
        let mut id_to_index = HashMap::with_capacity(ids.len());
        let mut index_to_id = Vec::with_capacity(ids.len());
        let mut max_id = None;
        for (index, &raw) in ids.iter().enumerate() {
            let id = PointId::from_raw(raw);
            if id_to_index.insert(id, index).is_some() {
                return Err(persist_err("ldlt config has duplicate point_ids"));
            }
            index_to_id.push(id);
            max_id = Some(max_id.map_or(raw, |seen: u64| seen.max(raw)));
        }
        if let Some(max_id) = max_id {
            if next_id <= max_id {
                return Err(persist_err(
                    "ldlt config next_point_id must exceed every stored PointId",
                ));
            }
        }
        Ok(Self {
            id_to_index,
            index_to_id,
            next_id,
        })
    }

    pub(crate) fn ids(&self) -> &[PointId] {
        &self.index_to_id
    }

    pub(crate) fn next_id(&self) -> u64 {
        self.next_id
    }

    pub(crate) fn len(&self) -> usize {
        self.index_to_id.len()
    }

    pub(crate) fn index_of(&self, id: PointId) -> Result<usize, GprError> {
        self.id_to_index
            .get(&id)
            .copied()
            .ok_or(GprError::InvalidPointId)
    }

    pub(crate) fn insert(&mut self) -> PointId {
        let id = PointId::from_raw(self.next_id);
        let index = self.index_to_id.len();
        self.next_id = self.next_id.saturating_add(1);
        self.index_to_id.push(id);
        self.id_to_index.insert(id, index);
        id
    }
}

impl PointRegistry {
    pub(crate) fn remove_at(&mut self, index: usize) {
        let id = self.index_to_id.remove(index);
        self.id_to_index.remove(&id);
        for (shifted, remaining) in self.index_to_id.iter().enumerate().skip(index) {
            self.id_to_index.insert(*remaining, shifted);
        }
    }
}
