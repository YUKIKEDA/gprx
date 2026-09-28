//! Crate-private linear algebra shared by every model.
//!
//! `f64` goes through faer's blocked, parallel routines. `f32` storage
//! accumulates in `f64` where the model stores an `f32` factor; the choice is
//! made per scalar by [`crate::kernel::KernelScalar`].

mod chol;
mod dense;
mod ldlt;
mod par;

pub(crate) use chol::*;
pub(crate) use dense::*;
pub(crate) use ldlt::*;
pub(crate) use par::*;
