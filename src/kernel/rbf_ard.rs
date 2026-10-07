//! ARD squared-exponential (RBF) kernel.

use super::ard::{self, ArdR2, Pick};
use super::dist::{ArdBlocks, ArdSqDiff, require_ard_sq_diff_shape};
use super::scalar::f64_pair;
use super::simd::rbf_ard::{self as lanes, Which};
use super::{ArdLengthscales, KernelScalar, Triangle, finite_kernel, write_square};
use crate::error::GprError;
use crate::math::KernelMath;
use faer::reborrow::ReborrowMut;
use faer::{MatMut, MatRef};

/// Evaluates the ARD RBF `k = exp( -½ Σ_d (x_d - x'_d)² / ℓ_d² )`.
///
/// Optimizer parameters are `θ_d = log(ℓ_d)` via [`ArdLengthscales`]. When every
/// `ℓ_d` equals a scalar `ℓ`, values match isotropic [`super::RbfKernel`].
/// `apply` / `grad` take the `n×d` coordinate matrix; a scalar squared-distance
/// matrix is not enough for `∂K/∂θ_d`. Amplitude is not stored here.
///
/// Cloning copies the lengthscale vectors. When [`crate::DistanceCachePolicy::Cached`] is
/// set, [`crate::Gpr`] caches raw `(Δx_d)²` as `n × (n·d)` and evaluates from that tensor.
/// Column-major views with unit row stride use four-wide `f64` SIMD for [`Self::apply`]
/// and [`Self::grad`].
///
/// # Examples
///
/// ```rust
/// use gprx::kernel::RbfArdKernel;
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let rbf = RbfArdKernel::new(&[1.0, 2.5])?;
/// assert_eq!(rbf.num_params(), 2);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct RbfArdKernel {
    lengthscales: ArdLengthscales,
}

impl RbfArdKernel {
    /// Builds an ARD RBF kernel from positive finite `ℓ_d`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if the slice is empty or a
    /// lengthscale is invalid.
    ///
    /// See the example on [`RbfArdKernel`].
    pub fn new(lengthscales: &[f64]) -> Result<Self, GprError> {
        Ok(Self {
            lengthscales: ArdLengthscales::new(lengthscales)?,
        })
    }

    /// Builds an ARD RBF kernel from `θ_d = log(ℓ_d)`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if the slice is empty or a
    /// `θ_d` is invalid.
    ///
    /// See the example on [`RbfArdKernel`].
    pub fn from_log_lengthscales(log_lengthscales: &[f64]) -> Result<Self, GprError> {
        Ok(Self {
            lengthscales: ArdLengthscales::from_log_lengthscales(log_lengthscales)?,
        })
    }

    /// Returns the shared ARD lengthscale mouth.
    ///
    /// See the example on [`RbfArdKernel`].
    pub fn lengthscales(&self) -> &ArdLengthscales {
        &self.lengthscales
    }

    pub(crate) fn from_ard(lengthscales: ArdLengthscales) -> Self {
        Self { lengthscales }
    }

    /// Returns `ℓ_d` for feature `dim`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::IndexOutOfRange`] if `dim` is out of range.
    ///
    /// See the example on [`RbfArdKernel`].
    pub fn lengthscale(&self, dim: usize) -> Result<f64, GprError> {
        self.lengthscales.lengthscale(dim)
    }

    /// Returns `θ_d = log(ℓ_d)`.
    ///
    /// See the example on [`RbfArdKernel`].
    pub fn log_lengthscales(&self) -> &[f64] {
        self.lengthscales.log_lengthscales()
    }

    /// Rebuilds every `ℓ_d` with the same open interval.
    ///
    /// # Errors
    ///
    /// Returns [`crate::IntervalError`] if any current `ℓ_d` is not strictly
    /// inside `interval`.
    ///
    /// See the example on [`RbfArdKernel`].
    pub fn with_bounds(
        self,
        interval: crate::param::Interval,
    ) -> Result<Self, crate::IntervalError> {
        Ok(Self {
            lengthscales: self.lengthscales.with_bounds(interval)?,
        })
    }

    /// Returns the number of optimizer parameters (`d`).
    ///
    /// See the example on [`RbfArdKernel`].
    pub fn num_params(&self) -> usize {
        self.lengthscales.num_params()
    }

    /// Writes `θ_d` into `out`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::LengthMismatch`] if `out` is the wrong length.
    ///
    /// See the example on [`RbfArdKernel`].
    pub fn get_params(&self, out: &mut [f64]) -> Result<(), GprError> {
        self.lengthscales.get_params(out)
    }

    /// Replaces `θ_d` from `params`.
    ///
    /// The previous values are kept on error.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::LengthMismatch`] if `params` is the wrong
    /// length, or [`GprError::InvalidHyperparameter`] if a `θ_d` is invalid.
    ///
    /// See the example on [`RbfArdKernel`].
    pub fn set_params(&mut self, params: &[f64]) -> Result<(), GprError> {
        self.lengthscales.set_params(params)
    }

    /// Writes `k(x, x')` into `out` for the requested triangle.
    ///
    /// Writes `k(x, x')` into `out` for the requested triangle.
    ///
    /// `x` is `n×d` (rows are points). The default contract is
    /// [`Triangle::Lower`]. Entries outside that triangle are left unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`GprError`] if `x` is empty, `d` does not match the
    /// lengthscales, `out` is not `n×n`, or a coordinate is non-finite.
    ///
    /// See the example on [`RbfArdKernel`].
    pub fn apply<T: KernelScalar>(
        &self,
        x: MatRef<'_, T>,
        out: MatMut<'_, T>,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        self.apply_math::<crate::math::Accurate, T>(x, out, uplo)
    }

    pub(crate) fn apply_math<M: KernelMath, T: KernelScalar>(
        &self,
        x: MatRef<'_, T>,
        mut out: MatMut<'_, T>,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        ard::require_square_points(x, out.as_ref(), self.num_params())?;
        let w = self.lengthscales.inv_ell_sq();
        if let Some((xf, of)) = f64_pair(x, out.rb_mut())
            && lanes::try_apply_points::<M>(xf, of, uplo, w)?
        {
            return Ok(());
        }
        write_square(out, uplo, |row, col| {
            rbf_value::<M, T>(ard::r2_from_coords(x, row, x, col, w, Pick::NONE)?)
        })
    }

    /// Writes rectangular `k(x, xs)` (train × test) into `out`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError`] if a matrix is empty, feature dimensions differ,
    /// `out` is the wrong shape, or a coordinate is non-finite.
    ///
    /// See the example on [`RbfArdKernel`].
    pub fn apply_cross<T: KernelScalar>(
        &self,
        x: MatRef<'_, T>,
        xs: MatRef<'_, T>,
        out: MatMut<'_, T>,
    ) -> Result<(), GprError> {
        self.apply_cross_math::<crate::math::Accurate, T>(x, xs, out)
    }

    pub(crate) fn apply_cross_math<M: KernelMath, T: KernelScalar>(
        &self,
        x: MatRef<'_, T>,
        xs: MatRef<'_, T>,
        mut out: MatMut<'_, T>,
    ) -> Result<(), GprError> {
        ard::require_cross(x, xs, out.as_ref(), self.num_params())?;
        let w = self.lengthscales.inv_ell_sq();
        if let (Some(xf), Some((xsf, of))) = (T::as_f64_ref(x), f64_pair(xs, out.rb_mut()))
            && lanes::try_apply_cross::<M>(xf, xsf, of, w)?
        {
            return Ok(());
        }
        super::write_rect(out, |row, col| {
            rbf_value::<M, T>(ard::r2_from_coords(x, row, xs, col, w, Pick::NONE)?)
        })
    }

    /// Writes the stationary diagonal `k(x, x) = 1` into `out`.
    ///
    /// See the example on [`RbfArdKernel`].
    pub fn fill_diag<T: KernelScalar>(&self, out: &mut [T]) {
        out.fill(T::from_f64(1.0));
    }

    /// Writes `∂K/∂θ_d` for `θ_d = log(ℓ_d)` into `d_k`.
    ///
    /// `∂k/∂θ_d = k · (x_d - x'_d)² / ℓ_d²`. This is not `∂k/∂ℓ_d`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::IndexOutOfRange`] if `param_idx` is out of
    /// range, or the same shape / non-finite errors as [`Self::apply`].
    ///
    /// See the example on [`RbfArdKernel`].
    pub fn grad<T: KernelScalar>(
        &self,
        x: MatRef<'_, T>,
        d_k: MatMut<'_, T>,
        param_idx: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        self.grad_math::<crate::math::Accurate, T>(x, d_k, param_idx, uplo)
    }

    pub(crate) fn grad_math<M: KernelMath, T: KernelScalar>(
        &self,
        x: MatRef<'_, T>,
        mut d_k: MatMut<'_, T>,
        param_idx: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        ard::require_param(NAME, param_idx, self.num_params())?;
        ard::require_square_points(x, d_k.as_ref(), self.num_params())?;
        let w = self.lengthscales.inv_ell_sq();
        if let Some((xf, of)) = f64_pair(x, d_k.rb_mut())
            && lanes::try_grad_points::<M>(xf, of, uplo, w, param_idx)?
        {
            return Ok(());
        }
        write_square(d_k, uplo, |row, col| {
            rbf_grad::<M, T>(ard::r2_from_coords(
                x,
                row,
                x,
                col,
                w,
                Pick::one(param_idx),
            )?)
        })
    }

    /// Folds `⟨W, ∂K/∂θ_d⟩` for every lengthscale from one lower-triangle Gram.
    ///
    /// Accurate math has `∂k/∂θ_d = k · (Δ_d)² / ℓ_d²`. The diagonal is zero
    /// and each strict lower pair is counted twice, which is the symmetric
    /// product `2 (Σ_i x_i² row_sum_i − xᵀ S x)` with `S_ij = S_ji = W_ij k_ij`
    /// off the diagonal. `fold` keeps that workspace between calls.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::LengthMismatch`] when `out` is not one entry per
    /// lengthscale, [`GprError::ShapeMismatch`] or [`GprError::NonFiniteInput`]
    /// from the point and Gram checks, [`GprError::SizeOverflow`] when the
    /// workspace does not fit in `usize`, and [`GprError::NonFiniteKernelValue`]
    /// when a folded total is not finite.
    pub(crate) fn contract_square<T: KernelScalar>(
        &self,
        x: MatRef<'_, T>,
        weight: MatRef<'_, T>,
        k: MatRef<'_, T>,
        diffs: Option<ArdSqDiff<'_, T>>,
        out: &mut [f64],
        fold: &mut Vec<f64>,
    ) -> Result<(), GprError> {
        let w = self.lengthscales.inv_ell_sq();
        let d = w.len();
        if out.len() != d {
            return Err(GprError::LengthMismatch {
                reason: format!("gradient has {} entries, expected {d}", out.len()),
            });
        }
        let n = ard::require_square_points(x, k, d)?;
        if weight.nrows() != n || weight.ncols() != n {
            return Err(GprError::ShapeMismatch {
                reason: format!(
                    "weight is {}x{}, expected {n}x{n}",
                    weight.nrows(),
                    weight.ncols()
                ),
            });
        }
        if let Some(cache) = diffs {
            require_ard_sq_diff_shape(cache, n, d)?;
        }
        let x_len = n.checked_mul(d).ok_or(GprError::SizeOverflow)?;
        let s_len = n.checked_mul(n).ok_or(GprError::SizeOverflow)?;
        let need = x_len
            .checked_add(s_len)
            .and_then(|sum| sum.checked_add(n))
            .and_then(|sum| sum.checked_add(x_len))
            .ok_or(GprError::SizeOverflow)?;
        if fold.len() < need {
            fold.resize(need, 0.0);
        }
        let (x64, rest) = fold.split_at_mut(x_len);
        let (s, rest) = rest.split_at_mut(s_len);
        let (row_sum, prod) = rest.split_at_mut(n);
        let prod = &mut prod[..x_len];
        for dim in 0..d {
            for row in 0..n {
                x64[dim * n + row] = x[(row, dim)].to_f64();
            }
        }
        for col in 0..n {
            s[col * n + col] = 0.0;
            for row in (col + 1)..n {
                let value = weight[(row, col)].to_f64() * k[(row, col)].to_f64();
                s[col * n + row] = value;
                s[row * n + col] = value;
            }
        }
        lanes::fold_square_lengthscales(x64, s, w, row_sum, prod, out)
    }

    /// Writes `⟨weight, ∂K/∂θ_d⟩_F` for every lengthscale into `out`, from
    /// the Gram `k` and the packed `(Δ_d)²` of a supplied ARD slot, as
    /// [`Self::contract_square`] does from coordinates. With
    /// `∂k/∂θ_d = k · w_d (Δ_d)²`, `S = weight ∘ k` is packed once into `fold`
    /// in the cache's layout (the lower triangle, column by column), so each
    /// lengthscale is one contiguous dot product: `2 w_d ⟨S, (Δ_d)²⟩` over
    /// the strict lower triangle (the diagonal `(Δ_d)²` is zero).
    ///
    /// # Errors
    ///
    /// Returns [`GprError::LengthMismatch`] when `out` is not one entry per
    /// lengthscale, and [`GprError::ShapeMismatch`] when `weight`, `k`, or
    /// the cache is not of the same `n` points.
    pub(crate) fn contract_square_from_sq_diff<T: KernelScalar>(
        &self,
        weight: MatRef<'_, T>,
        k: MatRef<'_, T>,
        cache: ArdSqDiff<'_, T>,
        out: &mut [f64],
        fold: &mut Vec<f64>,
    ) -> Result<(), GprError> {
        let w = self.lengthscales.inv_ell_sq();
        let d = w.len();
        if out.len() != d {
            return Err(GprError::LengthMismatch {
                reason: format!("gradient has {} entries, expected {d}", out.len()),
            });
        }
        let n = cache.n();
        require_ard_sq_diff_shape(cache, n, d)?;
        for (name, m) in [("weight", weight), ("k", k)] {
            if m.nrows() != n || m.ncols() != n {
                return Err(GprError::ShapeMismatch {
                    reason: format!("{name} is {}x{}, expected {n}x{n}", m.nrows(), m.ncols()),
                });
            }
        }
        let len = n
            .checked_add(1)
            .and_then(|n1| n.checked_mul(n1))
            .map(|cells| cells / 2)
            .ok_or(GprError::SizeOverflow)?;
        if fold.len() < len {
            fold.resize(len, 0.0);
        }
        let s = &mut fold[..len];
        let mut at = 0;
        for col in 0..n {
            for row in col..n {
                s[at] = weight[(row, col)].to_f64() * k[(row, col)].to_f64();
                at += 1;
            }
        }
        for (dim, slot) in out.iter_mut().enumerate() {
            let block = cache.block(dim);
            // Four running sums, folded in a fixed order.
            let mut acc = [0.0f64; 4];
            let chunks = s.as_chunks::<4>().0.iter().zip(block.as_chunks::<4>().0);
            for (a, b) in chunks {
                for lane in 0..4 {
                    acc[lane] += a[lane] * b[lane].to_f64();
                }
            }
            let tail = len - len % 4;
            let rest: f64 = s[tail..]
                .iter()
                .zip(&block[tail..])
                .map(|(a, b)| a * b.to_f64())
                .sum();
            *slot = 2.0 * w[dim] * (((acc[0] + acc[1]) + (acc[2] + acc[3])) + rest);
        }
        Ok(())
    }

    pub(crate) fn apply_from_sq_diff<M: KernelMath, T: KernelScalar>(
        &self,
        cache: ArdSqDiff<'_, T>,
        mut out: MatMut<'_, T>,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        let n = ard::require_square_out(out.as_ref())?;
        require_ard_sq_diff_shape(cache, n, self.num_params())?;
        let w = self.lengthscales.inv_ell_sq();
        if let (Some(cf), Some(of)) = (cache.as_f64(), T::as_f64_mut(out.rb_mut()))
            && lanes::try_apply_cache::<M>(cf, of, uplo, w)?
        {
            return Ok(());
        }
        write_square(out, uplo, |row, col| {
            rbf_value::<M, T>(ard::r2_from_cache(cache, row, col, w, Pick::NONE)?)
        })
    }

    /// Rectangular `K` from `(Δ_d)²` blocks.
    pub(crate) fn apply_cross_from_blocks<M: KernelMath, T: KernelScalar>(
        &self,
        blocks: ArdBlocks<'_, T>,
        mut out: MatMut<'_, T>,
    ) -> Result<(), GprError> {
        let w = self.lengthscales.inv_ell_sq();
        ard::require_blocks(blocks, out.as_ref(), self.num_params())?;
        if let Some(of) = T::as_f64_mut(out.rb_mut())
            && lanes::try_apply_cross_from_blocks::<M>(
                |dim| T::as_f64_slice(blocks.block(dim)),
                blocks.col0,
                of,
                w,
            )?
        {
            return Ok(());
        }
        ard::write_from_blocks(blocks, out, self.num_params(), |row, col| {
            rbf_value::<M, T>(ard::r2_from_blocks(blocks, row, col, w, Pick::NONE)?)
        })
    }

    /// Rectangular `∂K/∂θ` from `(Δ_d)²` blocks.
    pub(crate) fn grad_cross_from_blocks<M: KernelMath, T: KernelScalar>(
        &self,
        blocks: ArdBlocks<'_, T>,
        d_k: MatMut<'_, T>,
        param_idx: usize,
    ) -> Result<(), GprError> {
        ard::require_param(NAME, param_idx, self.num_params())?;
        let w = self.lengthscales.inv_ell_sq();
        ard::write_from_blocks(blocks, d_k, self.num_params(), |row, col| {
            rbf_grad::<M, T>(ard::r2_from_blocks(
                blocks,
                row,
                col,
                w,
                Pick::one(param_idx),
            )?)
        })
    }

    /// Rectangular `∂²K/∂θ_i ∂θ_j` from `(Δ_d)²` blocks.
    pub(crate) fn hess_cross_from_blocks<M: KernelMath, T: KernelScalar>(
        &self,
        blocks: ArdBlocks<'_, T>,
        d2_k: MatMut<'_, T>,
        i: usize,
        j: usize,
    ) -> Result<(), GprError> {
        ard::require_param_pair(NAME, i, j, self.num_params())?;
        let w = self.lengthscales.inv_ell_sq();
        ard::write_from_blocks(blocks, d2_k, self.num_params(), |row, col| {
            rbf_hess::<M, T>(
                ard::r2_from_blocks(blocks, row, col, w, Pick::pair(i, j))?,
                i == j,
            )
        })
    }

    pub(crate) fn grad_from_sq_diff<M: KernelMath, T: KernelScalar>(
        &self,
        cache: ArdSqDiff<'_, T>,
        mut d_k: MatMut<'_, T>,
        param_idx: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        ard::require_param(NAME, param_idx, self.num_params())?;
        let n = ard::require_square_out(d_k.as_ref())?;
        require_ard_sq_diff_shape(cache, n, self.num_params())?;
        let w = self.lengthscales.inv_ell_sq();
        if let (Some(cf), Some(of)) = (cache.as_f64(), T::as_f64_mut(d_k.rb_mut()))
            && lanes::try_grad_cache::<M>(cf, of, uplo, w, param_idx)?
        {
            return Ok(());
        }
        write_square(d_k, uplo, |row, col| {
            rbf_grad::<M, T>(ard::r2_from_cache(
                cache,
                row,
                col,
                w,
                Pick::one(param_idx),
            )?)
        })
    }

    /// Writes `∂²K/∂θ_i ∂θ_j` for `θ_d = log(ℓ_d)` into `d2_k`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::IndexOutOfRange`] if `i` or `j` is out of
    /// range, or the same shape / non-finite errors as [`Self::apply`].
    ///
    /// See the example on [`RbfArdKernel`].
    pub fn hess<T: KernelScalar>(
        &self,
        x: MatRef<'_, T>,
        d2_k: MatMut<'_, T>,
        i: usize,
        j: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        self.hess_math::<crate::math::Accurate, T>(x, d2_k, i, j, uplo)
    }

    pub(crate) fn hess_math<M: KernelMath, T: KernelScalar>(
        &self,
        x: MatRef<'_, T>,
        d2_k: MatMut<'_, T>,
        i: usize,
        j: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        ard::require_param_pair(NAME, i, j, self.num_params())?;
        let w = self.lengthscales.inv_ell_sq();
        ard::write_from_points(x, d2_k, self.num_params(), uplo, |row, col| {
            rbf_hess::<M, T>(
                ard::r2_from_coords(x, row, x, col, w, Pick::pair(i, j))?,
                i == j,
            )
        })
    }

    pub(crate) fn hess_from_sq_diff<M: KernelMath, T: KernelScalar>(
        &self,
        cache: ArdSqDiff<'_, T>,
        d2_k: MatMut<'_, T>,
        i: usize,
        j: usize,
        uplo: Triangle,
    ) -> Result<(), GprError> {
        ard::require_param_pair(NAME, i, j, self.num_params())?;
        let w = self.lengthscales.inv_ell_sq();
        ard::write_from_cache(cache, d2_k, self.num_params(), uplo, |row, col| {
            rbf_hess::<M, T>(
                ard::r2_from_cache(cache, row, col, w, Pick::pair(i, j))?,
                i == j,
            )
        })
    }

    /// Writes `∂K(X1, X2)/∂X2[*, dim]` into `d_k`.
    ///
    /// `∂k/∂x2_e = k (x1_e - x2_e) / ℓ_e²`.
    ///
    /// # Errors
    ///
    /// Same shape / non-finite errors as [`RbfKernel::grad_wrt_coord_dim`](super::RbfKernel::grad_wrt_coord_dim), or
    /// [`GprError::IndexOutOfRange`] when `dim` does not match `ℓ_d`.
    ///
    /// See the example on [`RbfArdKernel`].
    pub fn grad_wrt_coord_dim<T: KernelScalar>(
        &self,
        x1: MatRef<'_, T>,
        x2: MatRef<'_, T>,
        d_k: MatMut<'_, T>,
        dim: usize,
    ) -> Result<(), GprError> {
        super::radial::grad_wrt_coord_dim::<crate::math::Accurate, T>(self, x1, x2, d_k, dim)
    }

    pub(crate) fn grad_cross_from_coords<M: KernelMath, T: KernelScalar>(
        &self,
        x1: MatRef<'_, T>,
        x2: MatRef<'_, T>,
        mut d_k: MatMut<'_, T>,
        param_idx: usize,
    ) -> Result<(), GprError> {
        ard::require_param(NAME, param_idx, self.num_params())?;
        super::require_coord_grad(x1, x2, d_k.as_ref(), 0)?;
        let w = self.lengthscales.inv_ell_sq();
        if let (Some(af), Some((bf, of))) = (T::as_f64_ref(x1), f64_pair(x2, d_k.rb_mut()))
            && lanes::try_grad_cross::<M>(af, bf, &mut [of], w, Which::One(param_idx))?
        {
            return Ok(());
        }
        super::write_rect(d_k, |row, col| {
            rbf_grad::<M, T>(ard::r2_from_coords(
                x1,
                row,
                x2,
                col,
                w,
                Pick::one(param_idx),
            )?)
        })
    }

    /// `⟨weight, ∂K(x1, x2)/∂θ_d⟩` for every lengthscale into `out`, and
    /// `⟨weight, K⟩` when `want_value` is set.
    ///
    /// One `exp` per pair. A column-major `f64` view uses the lane path.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::LengthMismatch`] when `out` is not one slot per
    /// lengthscale, [`GprError::EmptyInput`] or [`GprError::DimensionMismatch`]
    /// when the point matrices do not match this leaf, [`GprError::ShapeMismatch`]
    /// when `weight` is not `x1.nrows() × x2.nrows()`, or a non-finite error
    /// from the kernel value.
    pub(crate) fn contract_cross<M: KernelMath, T: KernelScalar>(
        &self,
        x1: MatRef<'_, T>,
        x2: MatRef<'_, T>,
        weight: MatRef<'_, T>,
        out: &mut [f64],
        jobs: &mut Vec<f64>,
        want_value: bool,
    ) -> Result<f64, GprError> {
        let d = self.num_params();
        crate::data::require_count(out.len(), d, "kernel parameters")?;
        if x1.nrows() == 0 || x2.nrows() == 0 || x1.ncols() == 0 {
            return Err(GprError::EmptyInput);
        }
        if x1.ncols() != d || x2.ncols() != d {
            return Err(GprError::DimensionMismatch {
                x_dim: x2.ncols(),
                expected_dim: d,
            });
        }
        if weight.nrows() != x1.nrows() || weight.ncols() != x2.nrows() {
            return Err(GprError::ShapeMismatch {
                reason: format!(
                    "weight is {}x{}, expected {}x{}",
                    weight.nrows(),
                    weight.ncols(),
                    x1.nrows(),
                    x2.nrows()
                ),
            });
        }
        let w = self.lengthscales.inv_ell_sq();
        if let (Some(a), Some(b), Some(wt)) =
            (T::as_f64_ref(x1), T::as_f64_ref(x2), T::as_f64_ref(weight))
            && let Some(value) = lanes::try_contract_cross::<M>(a, b, wt, w, out, jobs, want_value)?
        {
            return Ok(value);
        }
        out.fill(0.0);
        let mut value = 0.0;
        for col in 0..x2.nrows() {
            for row in 0..x1.nrows() {
                let mut r2 = T::from_f64(0.0);
                for dim in 0..d {
                    let diff = x1[(row, dim)] - x2[(col, dim)];
                    if !diff.is_finite() {
                        return Err(GprError::NonFiniteInput);
                    }
                    r2 += diff * diff * T::from_f64(w[dim]);
                }
                let k = finite_kernel(ard_d1::<M, T>(r2))?;
                let wk = weight[(row, col)] * k;
                if want_value {
                    value += wk.to_f64();
                }
                for dim in 0..d {
                    let diff = x1[(row, dim)] - x2[(col, dim)];
                    let term = diff * diff * T::from_f64(w[dim]);
                    out[dim] += (wk * term).to_f64();
                }
            }
        }
        if !value.is_finite() || out.iter().any(|g| !g.is_finite()) {
            return Err(GprError::NonFiniteKernelValue);
        }
        Ok(value)
    }

    pub(crate) fn hess_cross_from_coords<M: KernelMath, T: KernelScalar>(
        &self,
        x1: MatRef<'_, T>,
        x2: MatRef<'_, T>,
        d2_k: MatMut<'_, T>,
        i: usize,
        j: usize,
    ) -> Result<(), GprError> {
        ard::require_param_pair(NAME, i, j, self.num_params())?;
        super::require_coord_grad(x1, x2, d2_k.as_ref(), 0)?;
        let w = self.lengthscales.inv_ell_sq();
        super::write_rect(d2_k, |row, col| {
            rbf_hess::<M, T>(
                ard::r2_from_coords(x1, row, x2, col, w, Pick::pair(i, j))?,
                i == j,
            )
        })
    }
}

const NAME: &str = "RBF";

fn ard_value<M: KernelMath, T: KernelScalar>(r2: T) -> T {
    M::exp(T::from_f64(-0.5) * r2)
}

fn ard_d1<M: KernelMath, T: KernelScalar>(r2: T) -> T {
    if M::ACCURATE {
        (T::from_f64(-0.5) * r2).exp()
    } else {
        M::jet(T::from_f64(-0.5) * r2).d1
    }
}

fn ard_hess_terms<M: KernelMath, T: KernelScalar>(r2: T, dim_i: T, dim_j: T, same: bool) -> T {
    let two = T::from_f64(2.0);
    if M::ACCURATE {
        let k = ard_value::<M, T>(r2);
        if same {
            k * dim_i * (dim_i - two)
        } else {
            k * dim_i * dim_j
        }
    } else {
        let jet = M::jet(T::from_f64(-0.5) * r2);
        if same {
            dim_i * (jet.d2 * dim_i - two * jet.d1)
        } else {
            jet.d2 * dim_i * dim_j
        }
    }
}

fn rbf_value<M: KernelMath, T: KernelScalar>(t: ArdR2<T>) -> Result<T, GprError> {
    finite_kernel(ard_value::<M, T>(t.r2))
}

fn rbf_grad<M: KernelMath, T: KernelScalar>(t: ArdR2<T>) -> Result<T, GprError> {
    finite_kernel(ard_d1::<M, T>(t.r2) * t.dim_i)
}

fn rbf_hess<M: KernelMath, T: KernelScalar>(t: ArdR2<T>, same: bool) -> Result<T, GprError> {
    finite_kernel(ard_hess_terms::<M, T>(t.r2, t.dim_i, t.dim_j, same))
}

#[cfg(test)]
mod tests {
    use super::RbfArdKernel;
    use crate::error::GprError;
    use crate::kernel::{RbfKernel, Triangle};
    use faer::{Mat, MatRef};

    const TOL: f64 = 1e-10;

    use crate::test_check::{assert_close, assert_lower_close, assert_send_sync, fill, points_2d};

    fn sq_dist(x: MatRef<'_, f64>) -> Mat<f64> {
        let n = x.nrows();
        let d = x.ncols();
        Mat::from_fn(n, n, |row, col| {
            let mut sum = 0.0;
            for dim in 0..d {
                let diff = x[(row, dim)] - x[(col, dim)];
                sum += diff * diff;
            }
            sum
        })
    }

    #[test]
    fn is_send_sync() {
        assert_send_sync::<RbfArdKernel>();
    }

    #[test]
    fn diagonal_is_one() {
        let rbf = RbfArdKernel::new(&[1.0, 2.0]).expect("valid");
        let x = points_2d(&[[0.0, 0.0], [1.0, 0.5], [0.2, 1.3]]);
        let mut k = fill(3, f64::NAN);
        rbf.apply(x.as_ref(), k.as_mut(), Triangle::Full)
            .expect("shape");
        assert_close(k[(0, 0)], 1.0, TOL);
        assert_close(k[(1, 1)], 1.0, TOL);
        assert_close(k[(2, 2)], 1.0, TOL);
    }

    #[test]
    fn known_values_use_per_dimension_lengthscales() {
        let rbf = RbfArdKernel::new(&[1.0, 2.0]).expect("valid");
        // (0,1) differs only in dim 0 by 1 ⇒ k = exp(-1/(2ℓ₀²)) = exp(-1/2)
        // (0,2) differs only in dim 1 by 2 ⇒ k = exp(-4/(2ℓ₁²)) = exp(-1/2)
        let x = points_2d(&[[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]]);
        let mut k = fill(3, 0.0);
        rbf.apply(x.as_ref(), k.as_mut(), Triangle::Full)
            .expect("shape");
        assert_close(k[(1, 0)], (-0.5_f64).exp(), TOL);
        assert_close(k[(2, 0)], (-0.5_f64).exp(), TOL);
        assert_close(k[(2, 1)], (-0.5_f64 * (1.0 + 1.0)).exp(), TOL);
    }

    #[test]
    fn equal_lengthscales_match_isotropic_rbf() {
        let ell = 1.4;
        let x = points_2d(&[[0.0, 0.0], [1.0, 0.5], [0.2, 1.3], [-0.4, 0.8]]);
        let dist = sq_dist(x.as_ref());
        let iso = RbfKernel::new(ell).expect("valid");
        let ard = RbfArdKernel::new(&[ell, ell]).expect("valid");
        let mut k_iso = fill(4, 0.0);
        let mut k_ard = fill(4, 0.0);
        iso.apply(dist.as_ref(), k_iso.as_mut(), Triangle::Full)
            .expect("shape");
        ard.apply(x.as_ref(), k_ard.as_mut(), Triangle::Full)
            .expect("shape");
        for col in 0..4 {
            for row in 0..4 {
                assert_close(k_ard[(row, col)], k_iso[(row, col)], TOL);
            }
        }
    }

    /// Every triangle of `got` that `uplo` writes matches `want`.
    fn assert_uplo_close(got: MatRef<'_, f64>, want: MatRef<'_, f64>, uplo: Triangle) {
        let n = got.nrows();
        for col in 0..n {
            for row in 0..n {
                let written = match uplo {
                    Triangle::Lower => row >= col,
                    Triangle::Upper => row <= col,
                    Triangle::Full => true,
                };
                if written {
                    assert_close(got[(row, col)], want[(row, col)], TOL);
                }
            }
        }
    }

    #[test]
    fn apply_from_sq_diff_matches_apply_for_every_triangle() {
        let rbf = RbfArdKernel::new(&[1.25, 0.8]).expect("valid");
        let x = points_2d(&[[0.0, 0.0], [1.0, 0.5], [0.2, 1.3], [-0.4, 0.8]]);
        let n = 4;
        let cache = crate::kernel::ArdSqDiffBuf::new(x.as_ref()).expect("size");
        for uplo in [Triangle::Lower, Triangle::Upper, Triangle::Full] {
            let mut from_points = fill(n, 0.0);
            let mut from_cache = fill(n, f64::NAN);
            rbf.apply(x.as_ref(), from_points.as_mut(), uplo)
                .expect("points");
            rbf.apply_from_sq_diff::<crate::math::Accurate, _>(
                cache.view(),
                from_cache.as_mut(),
                uplo,
            )
            .expect("cache");
            assert_uplo_close(from_cache.as_ref(), from_points.as_ref(), uplo);
        }
    }

    #[test]
    fn grad_from_sq_diff_matches_grad_for_every_triangle() {
        let rbf = RbfArdKernel::new(&[1.25, 0.8]).expect("valid");
        let x = points_2d(&[[0.0, 0.0], [1.0, 0.5], [0.2, 1.3], [-0.4, 0.8], [0.7, -1.1]]);
        let n = 5;
        let d = 2;
        let cache = crate::kernel::ArdSqDiffBuf::new(x.as_ref()).expect("size");
        for uplo in [Triangle::Lower, Triangle::Upper, Triangle::Full] {
            for param_idx in 0..d {
                let mut from_points = fill(n, 0.0);
                let mut from_cache = fill(n, f64::NAN);
                rbf.grad(x.as_ref(), from_points.as_mut(), param_idx, uplo)
                    .expect("points");
                rbf.grad_from_sq_diff::<crate::math::Accurate, _>(
                    cache.view(),
                    from_cache.as_mut(),
                    param_idx,
                    uplo,
                )
                .expect("cache");
                assert_uplo_close(from_cache.as_ref(), from_points.as_ref(), uplo);
            }
        }
    }

    #[test]
    fn full_is_symmetric() {
        let rbf = RbfArdKernel::new(&[0.8, 1.7]).expect("valid");
        let x = points_2d(&[[0.0, 0.0], [0.5, 1.0], [2.0, -0.3], [2.5, 0.4]]);
        let mut k = fill(4, 0.0);
        rbf.apply(x.as_ref(), k.as_mut(), Triangle::Full)
            .expect("shape");
        for col in 0..4 {
            for row in 0..4 {
                assert_close(k[(row, col)], k[(col, row)], TOL);
            }
        }
    }

    #[test]
    fn lower_matches_full_and_leaves_upper() {
        let rbf = RbfArdKernel::new(&[0.75, 1.25]).expect("valid");
        let x = points_2d(&[[0.0, 0.0], [1.0, 0.2], [2.0, -0.5]]);
        let mut full = fill(3, 0.0);
        rbf.apply(x.as_ref(), full.as_mut(), Triangle::Full)
            .expect("shape");
        let sentinel = 42.0;
        let mut lower = fill(3, sentinel);
        rbf.apply(x.as_ref(), lower.as_mut(), Triangle::Lower)
            .expect("shape");
        assert_lower_close(lower.as_ref(), full.as_ref(), TOL);
        assert_close(lower[(0, 1)], sentinel, TOL);
        assert_close(lower[(0, 2)], sentinel, TOL);
        assert_close(lower[(1, 2)], sentinel, TOL);
    }

    #[test]
    fn upper_matches_full() {
        let rbf = RbfArdKernel::new(&[1.0, 1.5]).expect("valid");
        let x = points_2d(&[[0.0, 0.0], [1.0, 1.0], [2.0, 0.5]]);
        let mut full = fill(3, 0.0);
        let mut upper = fill(3, -1.0);
        rbf.apply(x.as_ref(), full.as_mut(), Triangle::Full)
            .expect("shape");
        rbf.apply(x.as_ref(), upper.as_mut(), Triangle::Upper)
            .expect("shape");
        for col in 0..3 {
            for row in 0..=col {
                assert_close(upper[(row, col)], full[(row, col)], TOL);
            }
        }
        assert_close(upper[(1, 0)], -1.0, TOL);
        assert_close(upper[(2, 0)], -1.0, TOL);
        assert_close(upper[(2, 1)], -1.0, TOL);
    }

    #[test]
    fn hess_matches_finite_difference_of_grad() {
        let rbf = RbfArdKernel::from_log_lengthscales(&[-0.3, 0.4]).expect("valid");
        let x = points_2d(&[[0.0, 0.0], [1.2, -0.4], [0.3, 2.4]]);
        let h = 1e-6;
        let mut theta = [0.0; 2];
        rbf.get_params(&mut theta).expect("len 2");
        for i in 0..2 {
            for j in 0..2 {
                let mut plus_th = theta;
                let mut minus_th = theta;
                plus_th[j] += h;
                minus_th[j] -= h;
                let plus = RbfArdKernel::from_log_lengthscales(&plus_th).expect("valid");
                let minus = RbfArdKernel::from_log_lengthscales(&minus_th).expect("valid");
                let mut g_plus = fill(3, 0.0);
                let mut g_minus = fill(3, 0.0);
                let mut d2 = fill(3, 0.0);
                plus.grad(x.as_ref(), g_plus.as_mut(), i, Triangle::Full)
                    .expect("plus");
                minus
                    .grad(x.as_ref(), g_minus.as_mut(), i, Triangle::Full)
                    .expect("minus");
                rbf.hess(x.as_ref(), d2.as_mut(), i, j, Triangle::Full)
                    .expect("pair");
                for col in 0..3 {
                    for row in 0..3 {
                        let fd = (g_plus[(row, col)] - g_minus[(row, col)]) / (2.0 * h);
                        assert_close(d2[(row, col)], fd, TOL);
                    }
                }
            }
        }
    }

    #[test]
    fn grad_matches_finite_difference_per_dimension() {
        let rbf = RbfArdKernel::from_log_lengthscales(&[-0.3, 0.4]).expect("valid");
        let x = points_2d(&[[0.0, 0.0], [1.2, -0.4], [0.3, 2.4]]);
        let h = 1e-6;
        for dim in 0..2 {
            let mut params = [0.0; 2];
            rbf.get_params(&mut params).expect("len 2");
            params[dim] += h;
            let plus = RbfArdKernel::from_log_lengthscales(&params).expect("valid");
            params[dim] -= 2.0 * h;
            let minus = RbfArdKernel::from_log_lengthscales(&params).expect("valid");
            let mut k_plus = fill(3, 0.0);
            let mut k_minus = fill(3, 0.0);
            let mut dk = fill(3, 0.0);
            plus.apply(x.as_ref(), k_plus.as_mut(), Triangle::Full)
                .expect("shape");
            minus
                .apply(x.as_ref(), k_minus.as_mut(), Triangle::Full)
                .expect("shape");
            rbf.grad(x.as_ref(), dk.as_mut(), dim, Triangle::Full)
                .expect("index");
            for col in 0..3 {
                for row in 0..3 {
                    let fd = (k_plus[(row, col)] - k_minus[(row, col)]) / (2.0 * h);
                    assert_close(dk[(row, col)], fd, TOL);
                }
            }
        }
    }

    #[test]
    fn unused_dimension_has_zero_grad() {
        let rbf = RbfArdKernel::new(&[1.0, 2.0]).expect("valid");
        let x = points_2d(&[[0.0, 0.0], [1.0, 0.0]]);
        let mut dk0 = fill(2, 0.0);
        let mut dk1 = fill(2, 0.0);
        rbf.grad(x.as_ref(), dk0.as_mut(), 0, Triangle::Full)
            .expect("dim 0");
        rbf.grad(x.as_ref(), dk1.as_mut(), 1, Triangle::Full)
            .expect("dim 1");
        assert_close(dk1[(1, 0)], 0.0, TOL);
        assert!(dk0[(1, 0)].abs() > 1e-8);
    }

    #[test]
    fn grad_lower_matches_full() {
        let rbf = RbfArdKernel::new(&[1.0, 0.5]).expect("valid");
        let x = points_2d(&[[0.0, 0.0], [0.8, 0.3], [1.6, -0.2]]);
        let mut full = fill(3, 0.0);
        let mut lower = fill(3, 99.0);
        rbf.grad(x.as_ref(), full.as_mut(), 1, Triangle::Full)
            .expect("index 1");
        rbf.grad(x.as_ref(), lower.as_mut(), 1, Triangle::Lower)
            .expect("index 1");
        assert_lower_close(lower.as_ref(), full.as_ref(), TOL);
        assert_close(lower[(0, 1)], 99.0, TOL);
    }

    #[test]
    fn get_set_params_roundtrip() {
        let mut rbf = RbfArdKernel::new(&[2.0, 0.5]).expect("valid");
        let mut params = [0.0; 2];
        rbf.get_params(&mut params).expect("len 2");
        assert_close(params[0], 2.0_f64.ln(), TOL);
        assert_close(params[1], 0.5_f64.ln(), TOL);
        params[0] = 0.5_f64.ln();
        rbf.set_params(&params).expect("len 2");
        assert_close(rbf.lengthscale(0).expect("dim 0"), 0.5, TOL);
    }

    #[test]
    fn apply_cross_matches_square_block() {
        let rbf = RbfArdKernel::new(&[1.0, 2.0]).expect("valid");
        let train = points_2d(&[[0.0, 0.0], [1.0, 0.5]]);
        let test = points_2d(&[[0.2, -0.1], [1.0, 0.5]]);
        let mut square = fill(2, 0.0);
        rbf.apply(train.as_ref(), square.as_mut(), Triangle::Full)
            .expect("square");
        let mut cross = fill(2, 0.0);
        rbf.apply_cross(train.as_ref(), test.as_ref(), cross.as_mut())
            .expect("rect");
        // test[:, 1] == train[:, 1]
        assert_close(cross[(0, 1)], square[(0, 1)], TOL);
        assert_close(cross[(1, 1)], square[(1, 1)], TOL);
    }

    #[test]
    fn rejects_bad_index_dim_and_non_finite() {
        assert!(matches!(
            RbfArdKernel::new(&[1.0, 0.0]),
            Err(GprError::InvalidHyperparameter { .. })
        ));
        let rbf = RbfArdKernel::new(&[1.0, 2.0]).expect("valid");
        let x = points_2d(&[[0.0, 0.0], [1.0, 1.0]]);
        let mut dk = fill(2, 0.0);
        assert!(matches!(
            rbf.grad(x.as_ref(), dk.as_mut(), 2, Triangle::Lower),
            Err(GprError::IndexOutOfRange { .. })
        ));
        let bad_d = Mat::from_fn(2, 3, |_, _| 0.0);
        let mut k = fill(2, 0.0);
        assert!(matches!(
            rbf.apply(bad_d.as_ref(), k.as_mut(), Triangle::Full),
            Err(GprError::DimensionMismatch { .. })
        ));
        let nan = points_2d(&[[0.0, 0.0], [f64::NAN, 1.0]]);
        assert!(matches!(
            rbf.apply(nan.as_ref(), k.as_mut(), Triangle::Full),
            Err(GprError::NonFiniteInput)
        ));
    }

    fn f32_contract_matches<M: crate::math::KernelMath>() {
        let rbf = RbfArdKernel::new(&[0.8, 1.6]).expect("ard");
        let x1 = Mat::from_fn(4, 2, |i, j| 0.3 * i as f32 - 0.2 * j as f32);
        let x2 = Mat::from_fn(5, 2, |i, j| 0.2 * j as f32 - 0.1 * i as f32);
        let weight = Mat::from_fn(4, 5, |i, j| 0.1 * (i + 1) as f32 - 0.05 * j as f32);
        let mut out = [0.0; 2];
        let value = rbf
            .contract_cross::<M, f32>(
                x1.as_ref(),
                x2.as_ref(),
                weight.as_ref(),
                &mut out,
                &mut Vec::new(),
                true,
            )
            .expect("contract");
        let mut k = Mat::zeros(4, 5);
        rbf.apply_cross_math::<M, f32>(x1.as_ref(), x2.as_ref(), k.as_mut())
            .expect("value");
        let mut expect_v = 0.0;
        for col in 0..5 {
            for row in 0..4 {
                expect_v += f64::from(weight[(row, col)]) * f64::from(k[(row, col)]);
            }
        }
        assert_close(value, expect_v, 1e-5);
        let mut dk = Mat::zeros(4, 5);
        for (dim, &got) in out.iter().enumerate() {
            rbf.grad_cross_from_coords::<M, f32>(x1.as_ref(), x2.as_ref(), dk.as_mut(), dim)
                .expect("grad");
            let mut dot = 0.0;
            for col in 0..5 {
                for row in 0..4 {
                    dot += f64::from(weight[(row, col)]) * f64::from(dk[(row, col)]);
                }
            }
            assert_close(got, dot, 1e-5);
        }
    }

    #[test]
    fn f32_cross_contraction_matches_per_parameter() {
        f32_contract_matches::<crate::math::Accurate>();
        f32_contract_matches::<crate::math::FastApprox>();
    }

    #[test]
    fn f32_square_contraction_matches_per_parameter() {
        let rbf = RbfArdKernel::new(&[0.8, 1.6, 0.4]).expect("ard");
        let n = 7;
        let d = 3;
        let x = Mat::from_fn(n, d, |i, j| 0.2 * i as f32 - 0.15 * j as f32);
        let mut k = Mat::zeros(n, n);
        rbf.apply(x.as_ref(), k.as_mut(), Triangle::Lower)
            .expect("value");
        let weight = Mat::from_fn(n, n, |i, j| 0.1 * (i + 1) as f32 - 0.05 * j as f32);
        let mut out = [0.0; 3];
        let mut fold = Vec::new();
        rbf.contract_square(
            x.as_ref(),
            weight.as_ref(),
            k.as_ref(),
            None,
            &mut out,
            &mut fold,
        )
        .expect("contract");
        let mut dk = Mat::zeros(n, n);
        for (dim, &got) in out.iter().enumerate() {
            rbf.grad(x.as_ref(), dk.as_mut(), dim, Triangle::Lower)
                .expect("grad");
            let mut dot = 0.0;
            for col in 0..n {
                dot += f64::from(weight[(col, col)]) * f64::from(dk[(col, col)]);
                for row in (col + 1)..n {
                    dot += 2.0 * f64::from(weight[(row, col)]) * f64::from(dk[(row, col)]);
                }
            }
            assert_close(got, dot, 1e-5);
        }
    }
}
