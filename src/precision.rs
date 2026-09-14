//! Storage and refinement scalar policy.
//!
//! Phase 1 implements [`DoublePrecision`] only. Mixed and single precision wait
//! until Phase 5.

/// Selects storage and residual-refinement scalar types for GP computations.
pub trait PrecisionPolicy {
    /// Scalar used for `K`, `L`, and other stored buffers.
    type Storage;
    /// Scalar used when refining a solve against a higher-precision residual.
    type Refine;
}

/// Uses `f64` for both stored buffers and residual refinement.
///
/// This is the Phase 1 default. Fit (MLL and gradients) stays in double
/// precision; mixed-precision iterative refinement is a later option.
///
/// # Examples
///
/// ```rust
/// use gprx::{DoublePrecision, PrecisionPolicy};
///
/// type Storage = <DoublePrecision as PrecisionPolicy>::Storage;
/// let _: Storage = 0.0_f64;
/// ```
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DoublePrecision;

impl PrecisionPolicy for DoublePrecision {
    type Storage = f64;
    type Refine = f64;
}

#[cfg(test)]
mod tests {
    use super::{DoublePrecision, PrecisionPolicy};

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn double_precision_is_f64() {
        let storage: <DoublePrecision as PrecisionPolicy>::Storage = 0.0;
        let refine: <DoublePrecision as PrecisionPolicy>::Refine = 0.0;
        let _ = (storage, refine);
        assert_send_sync::<DoublePrecision>();
    }
}
