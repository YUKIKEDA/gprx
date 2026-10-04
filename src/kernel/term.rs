//! User-defined distance kernel leaf ([`KernelTerm`] / [`CustomKernel`]).

use faer::{MatMut, MatRef};
use std::fmt::{self, Debug};
use std::marker::PhantomData;

use crate::error::GprError;
use crate::kernel::KernelScalar;
use crate::param::Interval;

use super::Triangle;

/// Represents the user-defined kernel leaf evaluated from a squared-distance matrix.
///
/// Built-in leaves stay as [`super::KernelSpec`] enum arms. Implement this
/// trait and wrap with [`super::KernelSpec::custom`] to sit on Sum/Product.
/// Hot-path dispatch is static for built-ins; only this leaf uses a vtable.
/// Coordinate (points) kernels are not this trait. Sum/Product with
/// points-mode leaves evaluates each leaf in its own mode.
///
/// Write one generic `impl<T: KernelScalar> KernelTerm<T>`. [`KernelScalar`]
/// carries the arithmetic and `exp` / `ln` / `sqrt` / … a formula needs, and
/// built-in leaves such as [`super::RbfKernel`] are generic too, so the same
/// code serves `f64` and `f32` ([`crate::SinglePrecision`]). Parameters stay
/// `f64` at both scalars.
///
/// `get_params` / `set_params` / [`Self::bounds_into`] use the same log-`θ`
/// convention as built-in leaves. [`Self::apply`] writes `uplo`; entries
/// outside that triangle stay untouched.
///
/// # Errors
///
/// Trait methods return [`GprError`] for shape mismatches, non-finite
/// distances, and out-of-range parameter indices — the same data errors as
/// built-in leaves.
///
/// # Examples
///
/// ```rust
/// use faer::{MatMut, MatRef};
/// use gprx::kernel::{KernelScalar, KernelSpec, KernelTerm, RbfKernel, Triangle};
/// use gprx::{GaussianLikelihood, Gpr, Interval};
///
/// #[derive(Clone, Debug)]
/// struct UnitKernel;
///
/// impl<T: KernelScalar> KernelTerm<T> for UnitKernel {
///     fn num_params(&self) -> usize {
///         0
///     }
///
///     fn get_params(&self, out: &mut [f64]) -> Result<(), gprx::GprError> {
///         if out.is_empty() {
///             Ok(())
///         } else {
///             Err(gprx::GprError::IndexOutOfRange {
///                 reason: "unit kernel has no parameters".to_owned(),
///             })
///         }
///     }
///
///     fn set_params(&mut self, params: &[f64]) -> Result<(), gprx::GprError> {
///         KernelTerm::<T>::get_params(self, &mut params.to_vec())
///     }
///
///     fn bounds_into(&self, out: &mut [Interval]) -> Result<(), gprx::GprError> {
///         if out.is_empty() {
///             Ok(())
///         } else {
///             Err(gprx::GprError::IndexOutOfRange {
///                 reason: "unit kernel has no parameters".to_owned(),
///             })
///         }
///     }
///
///     fn apply(
///         &self,
///         dist: MatRef<'_, T>,
///         mut out: MatMut<'_, T>,
///         uplo: Triangle,
///     ) -> Result<(), gprx::GprError> {
///         if dist.nrows() != dist.ncols()
///             || out.nrows() != dist.nrows()
///             || out.ncols() != dist.ncols()
///         {
///             return Err(gprx::GprError::ShapeMismatch {
///                 reason: "unit kernel needs matching square matrices".to_owned(),
///             });
///         }
///         let n = dist.nrows();
///         if n == 0 {
///             return Err(gprx::GprError::EmptyInput);
///         }
///         for col in 0..n {
///             let start = match uplo {
///                 Triangle::Lower => col,
///                 Triangle::Upper | Triangle::Full => 0,
///             };
///             let end = match uplo {
///                 Triangle::Upper => col + 1,
///                 Triangle::Lower | Triangle::Full => n,
///             };
///             for row in start..end {
///                 out[(row, col)] = T::from_f64(1.0);
///             }
///         }
///         Ok(())
///     }
///
///     fn apply_cross(
///         &self,
///         dist: MatRef<'_, T>,
///         mut out: MatMut<'_, T>,
///     ) -> Result<(), gprx::GprError> {
///         if out.nrows() != dist.nrows() || out.ncols() != dist.ncols() {
///             return Err(gprx::GprError::ShapeMismatch {
///                 reason: "unit kernel cross shape mismatch".to_owned(),
///             });
///         }
///         if dist.nrows() == 0 || dist.ncols() == 0 {
///             return Err(gprx::GprError::EmptyInput);
///         }
///         for col in 0..out.ncols() {
///             for row in 0..out.nrows() {
///                 out[(row, col)] = T::from_f64(1.0);
///             }
///         }
///         Ok(())
///     }
///
///     fn fill_diag(&self, out: &mut [T]) -> Result<(), gprx::GprError> {
///         out.fill(T::from_f64(1.0));
///         Ok(())
///     }
///
///     fn grad(
///         &self,
///         _dist: MatRef<'_, T>,
///         _d_k: MatMut<'_, T>,
///         param_idx: usize,
///         _uplo: Triangle,
///     ) -> Result<(), gprx::GprError> {
///         Err(gprx::GprError::IndexOutOfRange {
///             reason: format!("unit kernel has no parameter {param_idx}"),
///         })
///     }
///
///     fn hess(
///         &self,
///         _dist: MatRef<'_, T>,
///         _d2_k: MatMut<'_, T>,
///         i: usize,
///         j: usize,
///         _uplo: Triangle,
///     ) -> Result<(), gprx::GprError> {
///         Err(gprx::GprError::IndexOutOfRange {
///             reason: format!("unit kernel has no parameter pair ({i}, {j})"),
///         })
///     }
///
///     fn hess_points(
///         &self,
///         _x: MatRef<'_, T>,
///         _d2_k: MatMut<'_, T>,
///         i: usize,
///         j: usize,
///         _uplo: Triangle,
///     ) -> Result<(), gprx::GprError> {
///         Err(gprx::GprError::IndexOutOfRange {
///             reason: format!("unit kernel has no parameter pair ({i}, {j})"),
///         })
///     }
///
///     fn clone_box(&self) -> Box<dyn KernelTerm<T>> {
///         Box::new(self.clone())
///     }
/// }
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let spec = KernelSpec::custom(UnitKernel) + KernelSpec::from(RbfKernel::new(1.0)?);
/// let _fitted = Gpr::new(spec, GaussianLikelihood::new(0.1)?)
///     .fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
///     .map_err(|(_, e)| e)?;
/// # Ok(())
/// # }
/// ```
pub trait KernelTerm<T: KernelScalar = f64>: Send + Sync + Debug + 'static {
    /// Returns the number of flattened log-`θ` parameters.
    ///
    /// See the example on [`KernelTerm`].
    fn num_params(&self) -> usize;

    /// Writes log-`θ` into `out`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::LengthMismatch`] if `out` is the wrong length.
    ///
    /// See the example on [`KernelTerm`].
    fn get_params(&self, out: &mut [f64]) -> Result<(), GprError>;

    /// Replaces log-`θ` from `params`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::LengthMismatch`] if `params` is the wrong
    /// length, or [`GprError::InvalidHyperparameter`] if a value is rejected.
    ///
    /// See the example on [`KernelTerm`].
    fn set_params(&mut self, params: &[f64]) -> Result<(), GprError>;

    /// Writes the open interval on each user-unit parameter (`ℓ`, `c`, …).
    ///
    /// # Errors
    ///
    /// Returns [`GprError::LengthMismatch`] if `out` is the wrong length.
    ///
    /// See the example on [`KernelTerm`].
    fn bounds_into(&self, out: &mut [Interval]) -> Result<(), GprError>;

    /// Writes `k` from squared distances into `out` for `uplo`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError`] if shapes mismatch, a matrix is empty, or a
    /// distance is non-finite.
    ///
    /// See the example on [`KernelTerm`].
    fn apply(
        &self,
        dist: MatRef<'_, T>,
        out: MatMut<'_, T>,
        uplo: Triangle,
    ) -> Result<(), GprError>;

    /// Writes rectangular `k(dist)` (train × test) into `out`.
    ///
    /// # Errors
    ///
    /// Same shape / non-finite errors as [`Self::apply`].
    ///
    /// See the example on [`KernelTerm`].
    fn apply_cross(&self, dist: MatRef<'_, T>, out: MatMut<'_, T>) -> Result<(), GprError>;

    /// Writes the stationary diagonal `k(x, x)` into `out`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError`] if the leaf cannot fill `out`.
    ///
    /// See the example on [`KernelTerm`].
    fn fill_diag(&self, out: &mut [T]) -> Result<(), GprError>;

    /// Writes `∂K/∂θ_{param_idx}` from squared distances into `d_k`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::IndexOutOfRange`] if `param_idx` is out of
    /// range, or the same shape errors as [`Self::apply`].
    ///
    /// See the example on [`KernelTerm`].
    fn grad(
        &self,
        dist: MatRef<'_, T>,
        d_k: MatMut<'_, T>,
        param_idx: usize,
        uplo: Triangle,
    ) -> Result<(), GprError>;

    /// Writes `∂²K/∂θ_i ∂θ_j` from squared distances into `d2_k`.
    ///
    /// One index pair per call. There is no numeric-difference fallback.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::IndexOutOfRange`] if `i` or `j` is out of
    /// range, or the same shape errors as [`Self::apply`].
    ///
    /// See the example on [`KernelTerm`].
    fn hess(
        &self,
        dist: MatRef<'_, T>,
        d2_k: MatMut<'_, T>,
        i: usize,
        j: usize,
        uplo: Triangle,
    ) -> Result<(), GprError>;

    /// Writes `∂²K/∂θ_i ∂θ_j` from point coordinates into `d2_k`.
    ///
    /// Distance leaves typically fill squared Euclidean distances from `x`
    /// and then call [`Self::hess`].
    ///
    /// # Errors
    ///
    /// Same index and shape errors as [`Self::hess`].
    ///
    /// See the example on [`KernelTerm`].
    fn hess_points(
        &self,
        x: MatRef<'_, T>,
        d2_k: MatMut<'_, T>,
        i: usize,
        j: usize,
        uplo: Triangle,
    ) -> Result<(), GprError>;

    /// Writes `∂K/∂θ_{param_idx}` of a rectangular block from squared distances (train × test) into `d_k`.
    ///
    /// `Sgpr` and `Svgp` read `∂K(Z, X)/∂θ`. The default is
    /// [`GprError::CoordGradientUnsupported`].
    ///
    /// # Errors
    ///
    /// Returns [`GprError::CoordGradientUnsupported`] when this leaf has no
    /// rectangular derivative, [`GprError::IndexOutOfRange`] if `param_idx` is
    /// out of range, or the same shape errors as [`Self::apply_cross`].
    ///
    /// See the example on [`KernelTerm`].
    fn grad_cross(
        &self,
        _dist: MatRef<'_, T>,
        _d_k: MatMut<'_, T>,
        _param_idx: usize,
    ) -> Result<(), GprError> {
        Err(GprError::CoordGradientUnsupported)
    }

    /// Writes `∂²K/∂θ_i ∂θ_j` of a rectangular block from squared distances into `d2_k`.
    ///
    /// The default is [`GprError::CoordGradientUnsupported`].
    ///
    /// # Errors
    ///
    /// Same as [`Self::grad_cross`], with the pair `(i, j)`.
    ///
    /// See the example on [`KernelTerm`].
    fn hess_cross(
        &self,
        _dist: MatRef<'_, T>,
        _d2_k: MatMut<'_, T>,
        _i: usize,
        _j: usize,
    ) -> Result<(), GprError> {
        Err(GprError::CoordGradientUnsupported)
    }

    /// Writes `∂k/∂(d²)`, the derivative of the kernel with respect to the squared distance, into `out` (same shape as `dist`).
    ///
    /// With this and [`Self::grad_wrt_sq_dist_theta`] and
    /// [`Self::hess_wrt_sq_dist`], `Sgpr<FreeInducing>` differentiates the
    /// kernel by the coordinates of the inducing points. The default is
    /// [`GprError::CoordGradientUnsupported`].
    ///
    /// # Errors
    ///
    /// Returns [`GprError::CoordGradientUnsupported`] when this leaf has no
    /// such derivative, or the same shape errors as [`Self::apply_cross`].
    ///
    /// See the example on [`KernelTerm`].
    fn grad_wrt_sq_dist(&self, _dist: MatRef<'_, T>, _out: MatMut<'_, T>) -> Result<(), GprError> {
        Err(GprError::CoordGradientUnsupported)
    }

    /// Writes `∂²k/∂(d²)²` into `out`.
    ///
    /// The default is [`GprError::CoordGradientUnsupported`].
    ///
    /// # Errors
    ///
    /// Same as [`Self::grad_wrt_sq_dist`].
    ///
    /// See the example on [`KernelTerm`].
    fn hess_wrt_sq_dist(&self, _dist: MatRef<'_, T>, _out: MatMut<'_, T>) -> Result<(), GprError> {
        Err(GprError::CoordGradientUnsupported)
    }

    /// Writes `∂²k/∂θ_{param_idx} ∂(d²)` into `out`.
    ///
    /// The default is [`GprError::CoordGradientUnsupported`].
    ///
    /// # Errors
    ///
    /// Same as [`Self::grad_wrt_sq_dist`], plus [`GprError::IndexOutOfRange`]
    /// if `param_idx` is out of range.
    ///
    /// See the example on [`KernelTerm`].
    fn grad_wrt_sq_dist_theta(
        &self,
        _dist: MatRef<'_, T>,
        _out: MatMut<'_, T>,
        _param_idx: usize,
    ) -> Result<(), GprError> {
        Err(GprError::CoordGradientUnsupported)
    }

    /// Clones this leaf into a new box.
    ///
    /// Used by [`super::KernelSpec::clone`].
    ///
    /// See the example on [`KernelTerm`].
    fn clone_box(&self) -> Box<dyn KernelTerm<T>>;

    /// Returns the stable registry key for persist.
    ///
    /// Must not start with `gprx.`.
    ///
    /// Built-in leaves do not use this. The default empty string is rejected
    /// when saving a [`super::KernelSpec::Custom`] leaf.
    ///
    /// See the example on [`KernelTerm`].
    fn persist_id(&self) -> &'static str {
        ""
    }

    /// Returns the JSON state paired with [`Self::persist_id`].
    ///
    /// # Errors
    ///
    /// Returns [`GprError::PersistFailed`] when this leaf has no persist form.
    ///
    /// See the example on [`KernelTerm`].
    fn persist_state(&self) -> Result<serde_json::Value, GprError> {
        Err(GprError::PersistFailed {
            kind: crate::error::PersistErrorKind::NotPersistable,
            reason: "this custom kernel does not implement persist_state".to_owned(),
        })
    }
}

/// A user leaf with its `f64` and `f32` implementations behind one box.
trait DualLeaf: Send + Sync + Debug {
    fn f64_term(&self) -> &dyn KernelTerm<f64>;
    fn f64_term_mut(&mut self) -> &mut dyn KernelTerm<f64>;
    fn f32_term(&self) -> &dyn KernelTerm<f32>;
    fn clone_dual(&self) -> Box<dyn DualLeaf>;
    fn type_label(&self) -> &'static str;
}

#[derive(Debug)]
struct Erased<K>(K);

impl<K> DualLeaf for Erased<K>
where
    K: KernelTerm<f64> + KernelTerm<f32> + Clone + Debug + Send + Sync + 'static,
{
    fn f64_term(&self) -> &dyn KernelTerm<f64> {
        &self.0
    }
    fn f64_term_mut(&mut self) -> &mut dyn KernelTerm<f64> {
        &mut self.0
    }
    fn f32_term(&self) -> &dyn KernelTerm<f32> {
        &self.0
    }
    fn clone_dual(&self) -> Box<dyn DualLeaf> {
        Box::new(Erased(self.0.clone()))
    }
    fn type_label(&self) -> &'static str {
        std::any::type_name::<K>()
    }
}

/// Represents the wrapper stored as [`super::KernelSpec::Custom`] / [`super::CompiledKernel::Custom`].
///
/// The leaf implements the same operations at `f32` and `f64`. Cloning copies
/// that leaf. [`PartialEq`] compares the leaf type and log-`θ` bits.
///
/// See the example on [`KernelTerm`].
pub struct CustomKernel<T: KernelScalar = f64> {
    inner: Box<dyn DualLeaf>,
    _scalar: PhantomData<fn() -> T>,
}

impl CustomKernel<f64> {
    /// Boxes a user leaf that implements the same operations at `f32` and `f64`.
    ///
    /// One generic `impl<T: KernelScalar> KernelTerm<T>` meets this bound.
    ///
    /// See the example on [`CustomKernel`].
    pub fn new<K>(term: K) -> Self
    where
        K: KernelTerm<f64> + KernelTerm<f32> + Clone + Debug + Send + Sync + 'static,
    {
        Self {
            inner: Box::new(Erased(term)),
            _scalar: PhantomData,
        }
    }

    pub(crate) fn with_scalar<T: KernelScalar>(&self) -> CustomKernel<T> {
        CustomKernel {
            inner: self.inner.clone_dual(),
            _scalar: PhantomData,
        }
    }
}

impl<T: KernelScalar> CustomKernel<T> {
    /// The leaf's implementation at `T`.
    fn term(&self) -> &dyn KernelTerm<T> {
        T::pick_term(self.inner.f64_term(), self.inner.f32_term())
    }

    /// Parameters are `f64` and shared by both scalars; the `f64` side owns them.
    fn params(&self) -> &dyn KernelTerm<f64> {
        self.inner.f64_term()
    }

    pub(super) fn num_params(&self) -> usize {
        self.params().num_params()
    }

    pub(crate) fn persist_id(&self) -> &'static str {
        self.params().persist_id()
    }

    pub(crate) fn persist_state(&self) -> Result<serde_json::Value, GprError> {
        self.params().persist_state()
    }

    pub(super) fn write_params(&self, out: &mut [f64], offset: &mut usize) -> Result<(), GprError> {
        let n = self.num_params();
        self.params().get_params(&mut out[*offset..*offset + n])?;
        *offset += n;
        Ok(())
    }

    pub(super) fn write_intervals(
        &self,
        out: &mut [Interval],
        offset: &mut usize,
    ) -> Result<(), GprError> {
        let n = self.num_params();
        self.params().bounds_into(&mut out[*offset..*offset + n])?;
        *offset += n;
        Ok(())
    }

    pub(super) fn apply_params(
        &mut self,
        params: &[f64],
        offset: &mut usize,
    ) -> Result<(), GprError> {
        let n = self.num_params();
        self.inner
            .f64_term_mut()
            .set_params(&params[*offset..*offset + n])?;
        *offset += n;
        Ok(())
    }

    pub(super) fn apply(
        &self,
        dist: MatRef<'_, T>,
        out: MatMut<'_, T>,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        self.term().apply(dist, out, uplo)
    }

    pub(super) fn apply_cross(
        &self,
        dist: MatRef<'_, T>,
        out: MatMut<'_, T>,
    ) -> Result<(), GprError> {
        self.term().apply_cross(dist, out)
    }

    pub(super) fn fill_diag(&self, out: &mut [T]) -> Result<(), GprError> {
        self.term().fill_diag(out)
    }

    pub(super) fn grad(
        &self,
        dist: MatRef<'_, T>,
        d_k: MatMut<'_, T>,
        param_idx: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        self.term().grad(dist, d_k, param_idx, uplo)
    }

    pub(super) fn hess(
        &self,
        dist: MatRef<'_, T>,
        d2_k: MatMut<'_, T>,
        i: usize,
        j: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        self.term().hess(dist, d2_k, i, j, uplo)
    }

    pub(super) fn hess_points(
        &self,
        x: MatRef<'_, T>,
        d2_k: MatMut<'_, T>,
        i: usize,
        j: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        self.term().hess_points(x, d2_k, i, j, uplo)
    }

    pub(super) fn grad_cross(
        &self,
        dist: MatRef<'_, T>,
        d_k: MatMut<'_, T>,
        param_idx: usize,
    ) -> Result<(), GprError> {
        self.term().grad_cross(dist, d_k, param_idx)
    }

    pub(super) fn hess_cross(
        &self,
        dist: MatRef<'_, T>,
        d2_k: MatMut<'_, T>,
        i: usize,
        j: usize,
    ) -> Result<(), GprError> {
        self.term().hess_cross(dist, d2_k, i, j)
    }

    pub(super) fn grad_wrt_sq_dist(
        &self,
        dist: MatRef<'_, T>,
        out: MatMut<'_, T>,
    ) -> Result<(), GprError> {
        self.term().grad_wrt_sq_dist(dist, out)
    }

    pub(super) fn hess_wrt_sq_dist(
        &self,
        dist: MatRef<'_, T>,
        out: MatMut<'_, T>,
    ) -> Result<(), GprError> {
        self.term().hess_wrt_sq_dist(dist, out)
    }

    pub(super) fn grad_wrt_sq_dist_theta(
        &self,
        dist: MatRef<'_, T>,
        out: MatMut<'_, T>,
        param_idx: usize,
    ) -> Result<(), GprError> {
        self.term().grad_wrt_sq_dist_theta(dist, out, param_idx)
    }
}

impl<T: KernelScalar> Clone for CustomKernel<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone_dual(),
            _scalar: PhantomData,
        }
    }
}

impl<T: KernelScalar> Debug for CustomKernel<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("CustomKernel").field(&self.inner).finish()
    }
}

impl<T: KernelScalar> PartialEq for CustomKernel<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.inner.type_label() != other.inner.type_label() {
            return false;
        }
        let n = self.num_params();
        if n != other.num_params() {
            return false;
        }
        let mut a = vec![0.0; n];
        let mut b = vec![0.0; n];
        match (
            self.params().get_params(&mut a),
            other.params().get_params(&mut b),
        ) {
            (Ok(()), Ok(())) => a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{CustomKernel, KernelTerm};
    use crate::kernel::{KernelSpec, RbfKernel, Triangle};
    use crate::param::Interval;
    use crate::{GaussianLikelihood, Gpr};
    use faer::{MatMut, MatRef};

    use crate::test_check::assert_send_sync;

    #[derive(Clone, Debug)]
    struct UnitKernel;

    impl<T: crate::kernel::KernelScalar> KernelTerm<T> for UnitKernel {
        fn num_params(&self) -> usize {
            0
        }

        fn get_params(&self, out: &mut [f64]) -> Result<(), crate::GprError> {
            if out.is_empty() {
                Ok(())
            } else {
                Err(crate::GprError::IndexOutOfRange {
                    reason: "unit kernel has no parameters".to_owned(),
                })
            }
        }

        fn set_params(&mut self, params: &[f64]) -> Result<(), crate::GprError> {
            KernelTerm::<T>::get_params(self, &mut params.to_vec())
        }

        fn bounds_into(&self, out: &mut [Interval]) -> Result<(), crate::GprError> {
            if out.is_empty() {
                Ok(())
            } else {
                Err(crate::GprError::IndexOutOfRange {
                    reason: "unit kernel has no parameters".to_owned(),
                })
            }
        }

        fn apply(
            &self,
            dist: MatRef<'_, T>,
            mut out: MatMut<'_, T>,
            uplo: Triangle,
        ) -> Result<(), crate::GprError> {
            if dist.nrows() != dist.ncols()
                || out.nrows() != dist.nrows()
                || out.ncols() != dist.ncols()
            {
                return Err(crate::GprError::ShapeMismatch {
                    reason: "unit kernel needs matching square matrices".to_owned(),
                });
            }
            let n = dist.nrows();
            if n == 0 {
                return Err(crate::GprError::EmptyInput);
            }
            for col in 0..n {
                let start = match uplo {
                    Triangle::Lower => col,
                    Triangle::Upper | Triangle::Full => 0,
                };
                let end = match uplo {
                    Triangle::Upper => col + 1,
                    Triangle::Lower | Triangle::Full => n,
                };
                for row in start..end {
                    out[(row, col)] = T::from_f64(1.0);
                }
            }
            Ok(())
        }

        fn apply_cross(
            &self,
            dist: MatRef<'_, T>,
            mut out: MatMut<'_, T>,
        ) -> Result<(), crate::GprError> {
            if out.nrows() != dist.nrows() || out.ncols() != dist.ncols() {
                return Err(crate::GprError::ShapeMismatch {
                    reason: "unit kernel cross shape mismatch".to_owned(),
                });
            }
            if dist.nrows() == 0 || dist.ncols() == 0 {
                return Err(crate::GprError::EmptyInput);
            }
            for col in 0..out.ncols() {
                for row in 0..out.nrows() {
                    out[(row, col)] = T::from_f64(1.0);
                }
            }
            Ok(())
        }

        fn fill_diag(&self, out: &mut [T]) -> Result<(), crate::GprError> {
            out.fill(T::from_f64(1.0));
            Ok(())
        }

        fn grad(
            &self,
            _dist: MatRef<'_, T>,
            _d_k: MatMut<'_, T>,
            param_idx: usize,
            _uplo: Triangle,
        ) -> Result<(), crate::GprError> {
            Err(crate::GprError::IndexOutOfRange {
                reason: format!("unit kernel has no parameter {param_idx}"),
            })
        }

        fn hess(
            &self,
            _dist: MatRef<'_, T>,
            _d2_k: MatMut<'_, T>,
            i: usize,
            j: usize,
            _uplo: Triangle,
        ) -> Result<(), crate::GprError> {
            Err(crate::GprError::IndexOutOfRange {
                reason: format!("unit kernel has no parameter pair ({i}, {j})"),
            })
        }

        fn hess_points(
            &self,
            _x: MatRef<'_, T>,
            _d2_k: MatMut<'_, T>,
            i: usize,
            j: usize,
            _uplo: Triangle,
        ) -> Result<(), crate::GprError> {
            Err(crate::GprError::IndexOutOfRange {
                reason: format!("unit kernel has no parameter pair ({i}, {j})"),
            })
        }

        fn clone_box(&self) -> Box<dyn KernelTerm<T>> {
            Box::new(self.clone())
        }
    }

    #[test]
    fn is_send_sync() {
        assert_send_sync::<CustomKernel>();
        assert_send_sync::<KernelSpec>();
    }

    #[test]
    fn clone_preserves_params() {
        let a = CustomKernel::new(UnitKernel);
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn custom_plus_rbf_fits() {
        let spec = KernelSpec::custom(UnitKernel) + KernelSpec::from(RbfKernel::new(1.0).unwrap());
        let fitted = Gpr::new(spec, GaussianLikelihood::new(0.1).unwrap())
            .fit(&[0.0, 1.0], 2, 1, &[0.0, 1.0])
            .expect("spd");
        assert!(fitted.neg_log_marginal_likelihood().unwrap().is_finite());
    }

    #[derive(Clone, Debug)]
    struct FailingRead;

    impl<T: crate::kernel::KernelScalar> KernelTerm<T> for FailingRead {
        fn num_params(&self) -> usize {
            1
        }

        fn get_params(&self, out: &mut [f64]) -> Result<(), crate::GprError> {
            if out.len() != 1 {
                return Err(crate::GprError::LengthMismatch {
                    reason: format!("expected 1 parameter, got {}", out.len()),
                });
            }
            Err(crate::GprError::InvalidHyperparameter {
                reason: "cannot read parameters".to_owned(),
            })
        }

        fn set_params(&mut self, params: &[f64]) -> Result<(), crate::GprError> {
            if params.len() == 1 {
                Ok(())
            } else {
                Err(crate::GprError::LengthMismatch {
                    reason: format!("expected 1 parameter, got {}", params.len()),
                })
            }
        }

        fn bounds_into(&self, out: &mut [Interval]) -> Result<(), crate::GprError> {
            if out.len() != 1 {
                return Err(crate::GprError::LengthMismatch {
                    reason: format!("expected 1 bound, got {}", out.len()),
                });
            }
            Err(crate::GprError::InvalidHyperparameter {
                reason: "cannot write bounds".to_owned(),
            })
        }

        fn apply(
            &self,
            _dist: MatRef<'_, T>,
            _out: MatMut<'_, T>,
            _uplo: Triangle,
        ) -> Result<(), crate::GprError> {
            Ok(())
        }

        fn apply_cross(
            &self,
            _dist: MatRef<'_, T>,
            _out: MatMut<'_, T>,
        ) -> Result<(), crate::GprError> {
            Ok(())
        }

        fn fill_diag(&self, _out: &mut [T]) -> Result<(), crate::GprError> {
            Ok(())
        }

        fn grad(
            &self,
            _dist: MatRef<'_, T>,
            _d_k: MatMut<'_, T>,
            _param_idx: usize,
            _uplo: Triangle,
        ) -> Result<(), crate::GprError> {
            Ok(())
        }

        fn hess(
            &self,
            _dist: MatRef<'_, T>,
            _d2_k: MatMut<'_, T>,
            _i: usize,
            _j: usize,
            _uplo: Triangle,
        ) -> Result<(), crate::GprError> {
            Ok(())
        }

        fn hess_points(
            &self,
            _x: MatRef<'_, T>,
            _d2_k: MatMut<'_, T>,
            _i: usize,
            _j: usize,
            _uplo: Triangle,
        ) -> Result<(), crate::GprError> {
            Ok(())
        }

        fn clone_box(&self) -> Box<dyn KernelTerm<T>> {
            Box::new(self.clone())
        }
    }

    #[test]
    fn get_params_propagates_custom_error() {
        let spec = KernelSpec::custom(FailingRead);
        let mut out = [0.0];
        assert!(matches!(
            spec.get_params(&mut out),
            Err(crate::GprError::InvalidHyperparameter { .. })
        ));
        assert!(matches!(
            spec.compile().get_params(&mut out),
            Err(crate::GprError::InvalidHyperparameter { .. })
        ));
    }

    #[test]
    fn write_intervals_propagates_custom_error() {
        let spec = KernelSpec::custom(FailingRead);
        let mut out = [Interval::DEFAULT_POSITIVE];
        let mut offset = 0;
        assert!(matches!(
            spec.write_intervals(&mut out, &mut offset),
            Err(crate::GprError::InvalidHyperparameter { .. })
        ));
    }
}
