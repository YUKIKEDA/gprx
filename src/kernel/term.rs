//! User-defined distance kernel leaf ([`KernelTerm`] / [`CustomKernel`]).

use std::fmt::{self, Debug};

use faer::{MatMut, MatRef};

use crate::error::GprError;
use crate::param::Interval;

use super::Triangle;

/// User-defined kernel leaf evaluated from a squared-distance matrix.
///
/// Built-in leaves stay as [`super::KernelSpec`] enum arms. Implement this
/// trait and wrap with [`super::KernelSpec::custom`] to sit on Sum/Product.
/// Hot-path dispatch is static for built-ins; only this leaf uses a vtable.
/// Coordinate (points) kernels are not this trait; Dist+Points mixing is
/// P2B-13.
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
/// use gprx::kernel::{KernelSpec, KernelTerm, RbfKernel, Triangle};
/// use gprx::{GaussianLikelihood, Gpr, Interval};
///
/// #[derive(Clone, Debug)]
/// struct UnitKernel;
///
/// impl KernelTerm for UnitKernel {
///     fn num_params(&self) -> usize {
///         0
///     }
///
///     fn get_params(&self, out: &mut [f64]) -> Result<(), gprx::GprError> {
///         if out.is_empty() {
///             Ok(())
///         } else {
///             Err(gprx::GprError::InvalidHyperparameter {
///                 reason: "unit kernel has no parameters".to_owned(),
///             })
///         }
///     }
///
///     fn set_params(&mut self, params: &[f64]) -> Result<(), gprx::GprError> {
///         self.get_params(&mut params.to_vec())
///     }
///
///     fn bounds_into(&self, out: &mut [Interval]) -> Result<(), gprx::GprError> {
///         if out.is_empty() {
///             Ok(())
///         } else {
///             Err(gprx::GprError::InvalidHyperparameter {
///                 reason: "unit kernel has no parameters".to_owned(),
///             })
///         }
///     }
///
///     fn apply(
///         &self,
///         dist: MatRef<'_, f64>,
///         mut out: MatMut<'_, f64>,
///         uplo: Triangle,
///     ) -> Result<(), gprx::GprError> {
///         if dist.nrows() != dist.ncols()
///             || out.nrows() != dist.nrows()
///             || out.ncols() != dist.ncols()
///         {
///             return Err(gprx::GprError::InvalidHyperparameter {
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
///                 out[(row, col)] = 1.0;
///             }
///         }
///         Ok(())
///     }
///
///     fn apply_cross(
///         &self,
///         dist: MatRef<'_, f64>,
///         mut out: MatMut<'_, f64>,
///     ) -> Result<(), gprx::GprError> {
///         if out.nrows() != dist.nrows() || out.ncols() != dist.ncols() {
///             return Err(gprx::GprError::InvalidHyperparameter {
///                 reason: "unit kernel cross shape mismatch".to_owned(),
///             });
///         }
///         if dist.nrows() == 0 || dist.ncols() == 0 {
///             return Err(gprx::GprError::EmptyInput);
///         }
///         for col in 0..out.ncols() {
///             for row in 0..out.nrows() {
///                 out[(row, col)] = 1.0;
///             }
///         }
///         Ok(())
///     }
///
///     fn fill_diag(&self, out: &mut [f64]) -> Result<(), gprx::GprError> {
///         out.fill(1.0);
///         Ok(())
///     }
///
///     fn grad(
///         &self,
///         _dist: MatRef<'_, f64>,
///         _d_k: MatMut<'_, f64>,
///         param_idx: usize,
///         _uplo: Triangle,
///     ) -> Result<(), gprx::GprError> {
///         Err(gprx::GprError::InvalidHyperparameter {
///             reason: format!("unit kernel has no parameter {param_idx}"),
///         })
///     }
///
///     fn clone_box(&self) -> Box<dyn KernelTerm> {
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
pub trait KernelTerm: Send + Sync + Debug + 'static {
    /// Returns the number of flattened log-`θ` parameters.
    fn num_params(&self) -> usize;

    /// Writes log-`θ` into `out`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if `out` is the wrong length.
    fn get_params(&self, out: &mut [f64]) -> Result<(), GprError>;

    /// Replaces log-`θ` from `params`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if `params` is the wrong
    /// length or a value is rejected.
    fn set_params(&mut self, params: &[f64]) -> Result<(), GprError>;

    /// Writes the open interval on each user-unit parameter (`ℓ`, `c`, …).
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if `out` is the wrong length.
    fn bounds_into(&self, out: &mut [Interval]) -> Result<(), GprError>;

    /// Writes `k` from squared distances into `out` for `uplo`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError`] if shapes mismatch, a matrix is empty, or a
    /// distance is non-finite.
    fn apply(
        &self,
        dist: MatRef<'_, f64>,
        out: MatMut<'_, f64>,
        uplo: Triangle,
    ) -> Result<(), GprError>;

    /// Writes rectangular `k(dist)` (train × test) into `out`.
    ///
    /// # Errors
    ///
    /// Same shape / non-finite errors as [`Self::apply`].
    fn apply_cross(&self, dist: MatRef<'_, f64>, out: MatMut<'_, f64>) -> Result<(), GprError>;

    /// Writes the stationary diagonal `k(x, x)` into `out`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError`] if the leaf cannot fill `out`.
    fn fill_diag(&self, out: &mut [f64]) -> Result<(), GprError>;

    /// Writes `∂K/∂θ_{param_idx}` from squared distances into `d_k`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if `param_idx` is out of
    /// range, or the same shape errors as [`Self::apply`].
    fn grad(
        &self,
        dist: MatRef<'_, f64>,
        d_k: MatMut<'_, f64>,
        param_idx: usize,
        uplo: Triangle,
    ) -> Result<(), GprError>;

    /// Clones this leaf into a new box. Used by [`super::KernelSpec::clone`].
    fn clone_box(&self) -> Box<dyn KernelTerm>;
}

/// Wrapper stored as [`super::KernelSpec::Custom`] / [`super::CompiledKernel::Custom`].
///
/// Cloning copies the boxed leaf via [`KernelTerm::clone_box`].
/// [`PartialEq`] compares [`std::any::type_name_of_val`] and log-`θ` bits.
///
/// See [`KernelTerm`] for construction.
pub struct CustomKernel {
    inner: Box<dyn KernelTerm>,
}

impl CustomKernel {
    /// Boxes a user leaf. See [`KernelTerm`].
    pub fn new(term: impl KernelTerm) -> Self {
        Self {
            inner: Box::new(term),
        }
    }

    pub(super) fn num_params(&self) -> usize {
        self.inner.num_params()
    }

    pub(super) fn write_params(&self, out: &mut [f64], offset: &mut usize) -> Result<(), GprError> {
        let n = self.inner.num_params();
        self.inner.get_params(&mut out[*offset..*offset + n])?;
        *offset += n;
        Ok(())
    }

    pub(super) fn write_intervals(
        &self,
        out: &mut [Interval],
        offset: &mut usize,
    ) -> Result<(), GprError> {
        let n = self.inner.num_params();
        self.inner.bounds_into(&mut out[*offset..*offset + n])?;
        *offset += n;
        Ok(())
    }

    pub(super) fn apply_params(
        &mut self,
        params: &[f64],
        offset: &mut usize,
    ) -> Result<(), GprError> {
        let n = self.inner.num_params();
        self.inner.set_params(&params[*offset..*offset + n])?;
        *offset += n;
        Ok(())
    }

    pub(super) fn apply(
        &self,
        dist: MatRef<'_, f64>,
        out: MatMut<'_, f64>,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        self.inner.apply(dist, out, uplo)
    }

    pub(super) fn apply_cross(
        &self,
        dist: MatRef<'_, f64>,
        out: MatMut<'_, f64>,
    ) -> Result<(), GprError> {
        self.inner.apply_cross(dist, out)
    }

    pub(super) fn fill_diag(&self, out: &mut [f64]) -> Result<(), GprError> {
        self.inner.fill_diag(out)
    }

    pub(super) fn grad(
        &self,
        dist: MatRef<'_, f64>,
        d_k: MatMut<'_, f64>,
        param_idx: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        self.inner.grad(dist, d_k, param_idx, uplo)
    }
}

impl Clone for CustomKernel {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone_box(),
        }
    }
}

impl Debug for CustomKernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("CustomKernel").field(&self.inner).finish()
    }
}

impl PartialEq for CustomKernel {
    fn eq(&self, other: &Self) -> bool {
        if std::any::type_name_of_val(&*self.inner) != std::any::type_name_of_val(&*other.inner) {
            return false;
        }
        let n = self.inner.num_params();
        if n != other.inner.num_params() {
            return false;
        }
        let mut a = vec![0.0; n];
        let mut b = vec![0.0; n];
        match (
            self.inner.get_params(&mut a),
            other.inner.get_params(&mut b),
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

    fn assert_send_sync<T: Send + Sync>() {}

    #[derive(Clone, Debug)]
    struct UnitKernel;

    impl KernelTerm for UnitKernel {
        fn num_params(&self) -> usize {
            0
        }

        fn get_params(&self, out: &mut [f64]) -> Result<(), crate::GprError> {
            if out.is_empty() {
                Ok(())
            } else {
                Err(crate::GprError::InvalidHyperparameter {
                    reason: "unit kernel has no parameters".to_owned(),
                })
            }
        }

        fn set_params(&mut self, params: &[f64]) -> Result<(), crate::GprError> {
            self.get_params(&mut params.to_vec())
        }

        fn bounds_into(&self, out: &mut [Interval]) -> Result<(), crate::GprError> {
            if out.is_empty() {
                Ok(())
            } else {
                Err(crate::GprError::InvalidHyperparameter {
                    reason: "unit kernel has no parameters".to_owned(),
                })
            }
        }

        fn apply(
            &self,
            dist: MatRef<'_, f64>,
            mut out: MatMut<'_, f64>,
            uplo: Triangle,
        ) -> Result<(), crate::GprError> {
            if dist.nrows() != dist.ncols()
                || out.nrows() != dist.nrows()
                || out.ncols() != dist.ncols()
            {
                return Err(crate::GprError::InvalidHyperparameter {
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
                    out[(row, col)] = 1.0;
                }
            }
            Ok(())
        }

        fn apply_cross(
            &self,
            dist: MatRef<'_, f64>,
            mut out: MatMut<'_, f64>,
        ) -> Result<(), crate::GprError> {
            if out.nrows() != dist.nrows() || out.ncols() != dist.ncols() {
                return Err(crate::GprError::InvalidHyperparameter {
                    reason: "unit kernel cross shape mismatch".to_owned(),
                });
            }
            if dist.nrows() == 0 || dist.ncols() == 0 {
                return Err(crate::GprError::EmptyInput);
            }
            for col in 0..out.ncols() {
                for row in 0..out.nrows() {
                    out[(row, col)] = 1.0;
                }
            }
            Ok(())
        }

        fn fill_diag(&self, out: &mut [f64]) -> Result<(), crate::GprError> {
            out.fill(1.0);
            Ok(())
        }

        fn grad(
            &self,
            _dist: MatRef<'_, f64>,
            _d_k: MatMut<'_, f64>,
            param_idx: usize,
            _uplo: Triangle,
        ) -> Result<(), crate::GprError> {
            Err(crate::GprError::InvalidHyperparameter {
                reason: format!("unit kernel has no parameter {param_idx}"),
            })
        }

        fn clone_box(&self) -> Box<dyn KernelTerm> {
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

    impl KernelTerm for FailingRead {
        fn num_params(&self) -> usize {
            1
        }

        fn get_params(&self, out: &mut [f64]) -> Result<(), crate::GprError> {
            if out.len() != 1 {
                return Err(crate::GprError::InvalidHyperparameter {
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
                Err(crate::GprError::InvalidHyperparameter {
                    reason: format!("expected 1 parameter, got {}", params.len()),
                })
            }
        }

        fn bounds_into(&self, out: &mut [Interval]) -> Result<(), crate::GprError> {
            if out.len() != 1 {
                return Err(crate::GprError::InvalidHyperparameter {
                    reason: format!("expected 1 bound, got {}", out.len()),
                });
            }
            Err(crate::GprError::InvalidHyperparameter {
                reason: "cannot write bounds".to_owned(),
            })
        }

        fn apply(
            &self,
            _dist: MatRef<'_, f64>,
            _out: MatMut<'_, f64>,
            _uplo: Triangle,
        ) -> Result<(), crate::GprError> {
            Ok(())
        }

        fn apply_cross(
            &self,
            _dist: MatRef<'_, f64>,
            _out: MatMut<'_, f64>,
        ) -> Result<(), crate::GprError> {
            Ok(())
        }

        fn fill_diag(&self, _out: &mut [f64]) -> Result<(), crate::GprError> {
            Ok(())
        }

        fn grad(
            &self,
            _dist: MatRef<'_, f64>,
            _d_k: MatMut<'_, f64>,
            _param_idx: usize,
            _uplo: Triangle,
        ) -> Result<(), crate::GprError> {
            Ok(())
        }

        fn clone_box(&self) -> Box<dyn KernelTerm> {
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
