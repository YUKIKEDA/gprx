//! The supplied `n × m` training blocks and `m × m` squares of a sparse
//! model on distances ([`SparseSupply`]), in `f64` and, for an `f32` model,
//! cast once.

#[allow(unused_imports)]
use super::*;

/// The training `d²` of a sparse model on supplied distances, at `f64` and,
/// for an `f32` storage, cast once: the `m × m` squares among the inducing
/// points and the `n × m` blocks from the training points to them. Empty
/// for a coordinate kernel.
#[derive(Clone, Debug, Default)]
pub(crate) struct SparseSupply {
    /// The training points that are the inducing points, in order.
    pub(crate) inducing: Vec<usize>,
    f64: SupplyAt<f64>,
    /// The `f32` copy, cast when an `f32` kernel first reads it: a factor
    /// that runs in `f64` (an `f32` SGPR's) never makes it. Checked to fit
    /// when the supply is made.
    f32: std::sync::OnceLock<Result<SupplyAt<f32>, GprError>>,
}

/// Checks the inducing indices of a model on supplied distances: each
/// below `n`, none twice.
pub(crate) fn check_inducing(inducing: &[usize], n: usize) -> Result<(), GprError> {
    // `O(m²)` comparisons, below the `O(m³)` factor of `K_mm`; no buffer.
    for (at, &i) in inducing.iter().enumerate() {
        if i >= n {
            return Err(GprError::IndexOutOfRange {
                reason: format!("inducing index {i} is not below the {n} training points"),
            });
        }
        if inducing[..at].contains(&i) {
            return Err(GprError::InvalidConfig {
                reason: format!("inducing index {i} is listed twice"),
            });
        }
    }
    Ok(())
}

/// The supplied `d²` of [`SparseSupply`] at one scalar.
#[derive(Clone, Debug, Default)]
pub(crate) struct SupplyAt<T: KernelScalar> {
    /// `m × m` among the inducing points.
    pub(crate) zz: BlockStore<T>,
    /// `n × m` from the training points to the inducing points, as the
    /// caller laid them out.
    pub(crate) xz: BlockStore<T>,
}

impl SparseSupply {
    /// The supply of the inducing points `inducing` (training indices) of
    /// `n` training points from `sources`, one per slot of `slots` (the
    /// kernel's): what a fit binds and a load binds again.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::IndexOutOfRange`] for an index not below `n`,
    /// [`GprError::InvalidConfig`] for an index listed twice, and the
    /// errors of [`crate::kernel::bind_inducing`] and [`Self::new`].
    pub(crate) fn bind<'s, S: KernelScalar>(
        slots: &[crate::kernel::DistanceSlot],
        sources: impl IntoIterator<Item = crate::kernel::DistanceSource<'s>>,
        n: usize,
        inducing: &[usize],
    ) -> Result<Self, GprError> {
        // The binding reads the blocks at these rows.
        check_inducing(inducing, n)?;
        let (zz, xz) = crate::kernel::bind_inducing(slots, sources, n, inducing)?;
        Self::new::<S>(slots, inducing.to_vec(), zz, xz)
    }

    /// The supply of `inducing` from its `f64` squares and blocks, cast
    /// once for an `f32` storage `S`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidDistance`] for a value past the range of
    /// `S`.
    pub(super) fn new<S: KernelScalar>(
        slots: &[crate::kernel::DistanceSlot],
        inducing: Vec<usize>,
        zz: BlockStore<f64>,
        xz: BlockStore<f64>,
    ) -> Result<Self, GprError> {
        // The squares are the blocks' inducing rows (a repaired pair is the
        // mean of two values in range), so the blocks are the values to check.
        xz.require_in_range::<S>(slots)?;
        Ok(Self {
            inducing,
            f64: SupplyAt { zz, xz },
            f32: std::sync::OnceLock::new(),
        })
    }

    /// The `f64` supply, always held.
    pub(crate) fn exact(&self) -> &SupplyAt<f64> {
        &self.f64
    }

    /// Whether training blocks are held (a model on supplied distances).
    pub(crate) fn is_supplied(&self) -> bool {
        !self.f64.xz.block_ids().is_empty()
    }

    /// Where training point `index` is an inducing point, if it is one.
    pub(crate) fn inducing_at(&self, index: usize) -> Option<usize> {
        self.inducing.iter().position(|&i| i == index)
    }

    /// Both copies of the blocks, writable: the `f64` one always, the
    /// `f32` one when it was made. A cast that failed is dropped, so the
    /// next `f32` read casts the changed blocks again rather than keep an
    /// error about the old ones.
    fn copies_mut(&mut self) -> (&mut SupplyAt<f64>, Option<&mut SupplyAt<f32>>) {
        if matches!(self.f32.get(), Some(Err(_))) {
            self.f32 = std::sync::OnceLock::new();
        }
        let f32 = self.f32.get_mut().and_then(|made| made.as_mut().ok());
        (&mut self.f64, f32)
    }

    /// Forms the squares among the inducing points again from the blocks'
    /// inducing rows, in place.
    fn rebuild_squares(&mut self) {
        let inducing = std::mem::take(&mut self.inducing);
        let (f64, f32) = self.copies_mut();
        let SupplyAt { zz, xz } = f64;
        xz.rows_into(&inducing, zz);
        if let Some(SupplyAt { zz, xz }) = f32 {
            xz.rows_into(&inducing, zz);
        }
        self.inducing = inducing;
    }

    /// Makes room in the training blocks for one more point, so
    /// [`Self::push_point`] cannot fail.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::SizeOverflow`] if the blocks cannot grow.
    pub(crate) fn reserve_point(&mut self) -> Result<(), GprError> {
        let (f64, f32) = self.copies_mut();
        f64.xz.reserve_row()?;
        if let Some(f32) = f32 {
            f32.xz.reserve_row()?;
        }
        Ok(())
    }

    /// Appends a training point whose squared distance to inducing point
    /// `col` in block `b` ([`BlockStore::block_ids`] order) is
    /// `value(b, col)`, once [`Self::reserve_point`] made room: nothing in
    /// it fails.
    pub(crate) fn push_point(&mut self, value: impl Fn(usize, usize) -> f64) {
        let (f64, f32) = self.copies_mut();
        f64.xz.push_row(&value);
        if let Some(f32) = f32 {
            f32.xz.push_row(|b, col| f32::from_f64(value(b, col)));
        }
    }

    /// Removes the last training point (the undo of [`Self::push_point`]).
    pub(crate) fn pop_point(&mut self) {
        if self.is_supplied() {
            let (f64, f32) = self.copies_mut();
            f64.xz.pop_row();
            if let Some(f32) = f32 {
                f32.xz.pop_row();
            }
        }
    }

    /// Removes training point `index`, which is not an inducing point: its
    /// row of every block, and one off the inducing indices past it. With
    /// `saved`, its row is appended there for [`Self::restore_point`].
    pub(crate) fn remove_point(&mut self, index: usize, saved: Option<&mut Vec<f64>>) {
        debug_assert!(self.inducing_at(index).is_none(), "an inducing point");
        if self.is_supplied() {
            if let Some(saved) = saved {
                self.f64.xz.row_into(index, saved);
            }
            let (f64, f32) = self.copies_mut();
            f64.xz.remove_row(index);
            if let Some(f32) = f32 {
                f32.xz.remove_row(index);
            }
        }
        for i in &mut self.inducing {
            if *i > index {
                *i -= 1;
            }
        }
    }

    /// Puts back training point `index` from `saved` (the undo of
    /// [`Self::remove_point`]).
    pub(crate) fn restore_point(&mut self, index: usize, saved: &[f64]) {
        for i in &mut self.inducing {
            if *i >= index {
                *i += 1;
            }
        }
        if self.is_supplied() {
            let (f64, f32) = self.copies_mut();
            f64.xz.insert_row(index, saved);
            if let Some(f32) = f32 {
                f32.xz.insert_row(index, saved);
            }
        }
    }

    /// Makes room for one more inducing point, so [`Self::add_inducing`]
    /// cannot fail.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::SizeOverflow`] if the blocks cannot grow.
    pub(crate) fn reserve_inducing(&mut self) -> Result<(), GprError> {
        let (f64, f32) = self.copies_mut();
        f64.xz.reserve_col()?;
        if let Some(f32) = f32 {
            f32.xz.reserve_col()?;
        }
        Ok(())
    }

    /// Makes training point `point` (not an inducing point) one more
    /// inducing point, once [`Self::reserve_inducing`] ran: `value(b, row)`
    /// is its squared distance to training point `row` in block `b`
    /// ([`BlockStore::block_ids`] order), and `mirror(b, col)` the value
    /// its row of block `b` holds for inducing point `col` from now on (the
    /// stored one, or a pair a tidy source repaired); the values it held
    /// are appended to `saved` for [`Self::undo_add_inducing`].
    pub(crate) fn add_inducing(
        &mut self,
        point: usize,
        value: impl Fn(usize, usize) -> f64,
        mirror: impl Fn(usize, usize) -> f64,
        saved: &mut Vec<f64>,
    ) {
        let m = self.inducing.len();
        self.f64.xz.row_into(point, saved);
        let (f64, f32) = self.copies_mut();
        for (b, at) in f64.xz.block_ids().into_iter().enumerate() {
            for col in 0..m {
                f64.xz.set(at, point, col, mirror(b, col));
            }
        }
        f64.xz.push_col(&value);
        if let Some(f32) = f32 {
            for (b, at) in f32.xz.block_ids().into_iter().enumerate() {
                for col in 0..m {
                    f32.xz.set(at, point, col, f32::from_f64(mirror(b, col)));
                }
            }
            f32.xz.push_col(|b, row| f32::from_f64(value(b, row)));
        }
        self.inducing.push(point);
        self.rebuild_squares();
    }

    /// The undo of [`Self::add_inducing`] of training point `point`, with
    /// the values it saved.
    pub(crate) fn undo_add_inducing(&mut self, point: usize, saved: &[f64]) {
        debug_assert_eq!(self.inducing.last(), Some(&point), "not the last added");
        self.inducing.retain(|&i| i != point);
        let m = self.inducing.len();
        let (f64, f32) = self.copies_mut();
        f64.xz.pop_col();
        for (b, at) in f64.xz.block_ids().into_iter().enumerate() {
            for col in 0..m {
                f64.xz.set(at, point, col, saved[b * m + col]);
            }
        }
        if let Some(f32) = f32 {
            f32.xz.pop_col();
            for (b, at) in f32.xz.block_ids().into_iter().enumerate() {
                for col in 0..m {
                    f32.xz
                        .set(at, point, col, f32::from_f64(saved[b * m + col]));
                }
            }
        }
        self.rebuild_squares();
    }

    /// Removes inducing point `at` (its column of the blocks) in place; the
    /// column is appended to `saved` for [`Self::undo_remove_inducing`].
    pub(crate) fn remove_inducing(&mut self, at: usize, saved: &mut Vec<f64>) {
        let (f64, f32) = self.copies_mut();
        f64.xz.remove_col(at, Some(saved));
        if let Some(f32) = f32 {
            f32.xz.remove_col(at, None);
        }
        self.inducing.remove(at);
        self.rebuild_squares();
    }

    /// The undo of [`Self::remove_inducing`]: training point `point` is
    /// inducing point `at` again, with the column it saved.
    pub(crate) fn undo_remove_inducing(&mut self, at: usize, point: usize, saved: &[f64]) {
        let (f64, f32) = self.copies_mut();
        f64.xz.insert_col(at, saved);
        if let Some(f32) = f32 {
            f32.xz.insert_col(at, saved);
        }
        self.inducing.insert(at, point);
        self.rebuild_squares();
    }

    /// The supply at `T` (`f64`, or the `f32` cast, made on the first read).
    ///
    /// # Errors
    ///
    /// The supply holds `f64` and `f32` only; any other `T` is reported as
    /// unbound. A value past the range of `f32` is reported as
    /// [`GprError::InvalidDistance`] (an `f32` model's supply was checked
    /// when it was made).
    pub(crate) fn at<T: KernelScalar>(&self) -> Result<&SupplyAt<T>, GprError> {
        let f64: &dyn std::any::Any = &self.f64;
        if let Some(at) = f64.downcast_ref() {
            return Ok(at);
        }
        let f32 = self
            .f32
            .get_or_init(|| {
                Ok(SupplyAt {
                    zz: self.f64.zz.cast()?,
                    xz: self.f64.xz.cast()?,
                })
            })
            .as_ref()
            .map_err(Clone::clone)?;
        let f32: &dyn std::any::Any = f32;
        f32.downcast_ref().ok_or_else(crate::kernel::unbound)
    }
}
