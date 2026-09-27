//! Compute scalar for [`super::CompiledKernel`].

use std::fmt;

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Scalar used when a [`super::CompiledKernel`] evaluates a kernel.
///
/// Parameters stay `f64`. `f64` keeps the distance cache and SIMD paths.
/// `f32` evaluates the same formulas in scalar arithmetic.
pub trait KernelScalar:
    sealed::Sealed + Copy + Clone + fmt::Debug + PartialEq + Send + Sync + 'static
{
    /// Casts a stored `f64` parameter into this compute scalar.
    fn from_f64(value: f64) -> Self;

    /// Promotes a computed value back to `f64` for comparisons.
    fn to_f64(self) -> f64;
}

impl KernelScalar for f64 {
    fn from_f64(value: f64) -> Self {
        value
    }

    fn to_f64(self) -> f64 {
        self
    }
}

impl KernelScalar for f32 {
    fn from_f64(value: f64) -> Self {
        value as f32
    }

    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}
