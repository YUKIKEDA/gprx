//! Compute scalar for [`super::CompiledKernel`] and the stored Gram factors.

use std::fmt;

use faer::{Mat, MatRef};

use crate::math::{self, ExpJet};

pub(crate) mod sealed {
    use dyn_stack::MemBuffer;
    use faer::{Mat, MatMut, MatRef};

    use crate::error::{CholeskyStage, GprError};
    use crate::math::ExpJet;

    /// Crate-private operations whose algorithm differs between `f32` and `f64`.
    ///
    /// `f64` calls faer's blocked, parallel routines. `f32` accumulates in
    /// `f64` so a stored `f32` factor keeps the digits of an `f64` sum.
    pub trait ScalarOps: Sized {
        /// Scratch that holds `y` cast to this scalar. `f64` uses no buffer.
        type RowCast: Clone + Send + Sync + 'static;
        /// Scratch that holds `X` cast to this scalar. `f64` uses no buffer.
        type ColCast: Clone + Send + Sync + 'static;

        /// Sparse assembly and predict, the SVGP gradient, and Exact
        /// leave-one-out compute in `f64` and round into this storage.
        /// `true` only for `f32`.
        const ROUNDS_FROM_F64: bool;

        /// safetensors dtype of this scalar.
        const DTYPE: safetensors::Dtype;

        fn empty_rows() -> Self::RowCast;

        fn empty_cols() -> Self::ColCast;

        /// Views `y` as this scalar. `f64` returns `y`. `f32` fills `cast`.
        fn storage_rows<'a>(y: &'a [f64], cast: &'a mut Self::RowCast) -> &'a [Self];

        /// Views `x` as this scalar. `f64` returns `x`. `f32` fills `cast`.
        fn storage_cols<'a>(x: MatRef<'a, f64>, cast: &'a mut Self::ColCast) -> MatRef<'a, Self>;

        /// Degree-7 polynomial `exp` ([`crate::FastApprox`]).
        fn fast_exp(self) -> Self;

        /// Value and first two derivatives of [`Self::fast_exp`].
        fn fast_jet(self) -> ExpJet<Self>;

        /// Factors `A` in place as `L Lᵀ`. The strictly upper triangle is unspecified.
        fn cholesky_lower(
            a: &mut Mat<Self>,
            scratch: &mut MemBuffer,
            jitter: f64,
            stage: CholeskyStage,
        ) -> Result<(), GprError>;

        /// Overwrites `rhs` with `(L Lᵀ)⁻¹ rhs`. `scratch` sizes faer's solve.
        fn solve_llt_in_place(l: MatRef<'_, Self>, rhs: MatMut<'_, Self>, scratch: &mut MemBuffer);

        /// Overwrites `rhs` with `(L Lᵀ)⁻¹ rhs` with no caller scratch.
        fn solve_llt_owned_scratch(l: MatRef<'_, Self>, rhs: MatMut<'_, Self>);

        /// Writes `diag(A⁻¹)` from the lower Cholesky factor `L` of `A`.
        fn inv_diag_from_chol_l(l: MatRef<'_, Self>, q_diag: &mut [Self]);

        /// Writes `a · y` into column 0 of `ay`.
        fn matvec_columns(a: MatRef<'_, Self>, y: &[Self], ay: MatMut<'_, Self>);

        /// Writes `a aᵀ` into `b` (`b` is `m×m`, zero on entry).
        fn gram_aat(a: MatRef<'_, Self>, b: MatMut<'_, Self>);

        /// Solves `L w = v` in place for unit-lower `L` (`ld` is `n×n`).
        fn solve_unit_lower_in_place(ld: MatRef<'_, Self>, v: MatMut<'_, Self>);

        /// Deletes row and column `index` from the LDLT factor held in the
        /// leading `n×n` of `ld`.
        fn ldlt_delete_row_col(
            ld: MatMut<'_, Self>,
            index: usize,
            n: usize,
            scratch: &mut MemBuffer,
        );
    }
}

/// Scalar used when a [`super::CompiledKernel`] evaluates a kernel.
///
/// Only `f32` and `f64` implement it. Parameters stay `f64`. The trait
/// carries faer's field arithmetic plus the transcendental functions that
/// kernel formulas need, so a leaf can be written once over `T`.
///
/// # Examples
///
/// ```rust
/// use gprx::kernel::KernelScalar;
///
/// fn half_exp<T: KernelScalar>(x: T) -> T {
///     (x * T::from_f64(0.5)).exp()
/// }
///
/// assert!((half_exp(2.0_f64) - 1.0_f64.exp()).abs() < 1e-15);
/// assert!((half_exp(2.0_f32) - 1.0_f32.exp()).abs() < 1e-6);
/// ```
pub trait KernelScalar:
    sealed::ScalarOps + faer_traits::RealField + Copy + fmt::Debug + Default + Send + Sync + 'static
{
    /// Casts a stored `f64` parameter into this compute scalar.
    fn from_f64(value: f64) -> Self;

    /// Promotes a computed value back to `f64` for comparisons.
    fn to_f64(self) -> f64;

    /// `eˣ`.
    fn exp(self) -> Self;

    /// Natural logarithm.
    fn ln(self) -> Self;

    /// Square root.
    fn sqrt(self) -> Self;

    /// Absolute value.
    fn abs(self) -> Self;

    /// Whether the value is neither infinite nor `NaN`.
    fn is_finite(self) -> bool;
}

macro_rules! impl_kernel_scalar {
    ($t:ty) => {
        impl KernelScalar for $t {
            #[inline(always)]
            fn from_f64(value: f64) -> Self {
                value as $t
            }

            #[inline(always)]
            fn to_f64(self) -> f64 {
                f64::from(self)
            }

            #[inline(always)]
            fn exp(self) -> Self {
                <$t>::exp(self)
            }

            #[inline(always)]
            fn ln(self) -> Self {
                <$t>::ln(self)
            }

            #[inline(always)]
            fn sqrt(self) -> Self {
                <$t>::sqrt(self)
            }

            #[inline(always)]
            fn abs(self) -> Self {
                <$t>::abs(self)
            }

            #[inline(always)]
            fn is_finite(self) -> bool {
                <$t>::is_finite(self)
            }
        }
    };
}

impl_kernel_scalar!(f32);
impl_kernel_scalar!(f64);

impl sealed::ScalarOps for f64 {
    type RowCast = ();
    type ColCast = ();

    const ROUNDS_FROM_F64: bool = false;
    const DTYPE: safetensors::Dtype = safetensors::Dtype::F64;

    fn empty_rows() -> Self::RowCast {}

    fn empty_cols() -> Self::ColCast {}

    fn storage_rows<'a>(y: &'a [f64], _cast: &'a mut Self::RowCast) -> &'a [Self] {
        y
    }

    fn storage_cols<'a>(x: MatRef<'a, f64>, _cast: &'a mut Self::ColCast) -> MatRef<'a, Self> {
        x
    }

    #[inline(always)]
    fn fast_exp(self) -> Self {
        math::fast_exp_f64(self)
    }

    #[inline(always)]
    fn fast_jet(self) -> ExpJet<Self> {
        math::fast_jet_f64(self)
    }

    fn cholesky_lower(
        a: &mut Mat<Self>,
        scratch: &mut dyn_stack::MemBuffer,
        jitter: f64,
        stage: crate::error::CholeskyStage,
    ) -> Result<(), crate::error::GprError> {
        crate::linalg::cholesky_lower_faer(a, scratch, jitter, stage)
    }

    fn solve_llt_in_place(
        l: MatRef<'_, Self>,
        rhs: faer::MatMut<'_, Self>,
        scratch: &mut dyn_stack::MemBuffer,
    ) {
        crate::linalg::solve_llt_faer(l, rhs, scratch);
    }

    fn solve_llt_owned_scratch(l: MatRef<'_, Self>, rhs: faer::MatMut<'_, Self>) {
        crate::linalg::solve_llt_faer_owned(l, rhs);
    }

    fn inv_diag_from_chol_l(l: MatRef<'_, Self>, q_diag: &mut [Self]) {
        crate::linalg::inv_diag_from_chol_l_faer(l, q_diag);
    }

    fn matvec_columns(a: MatRef<'_, Self>, y: &[Self], ay: faer::MatMut<'_, Self>) {
        crate::linalg::matvec_columns_native(a, y, ay);
    }

    fn gram_aat(a: MatRef<'_, Self>, b: faer::MatMut<'_, Self>) {
        crate::linalg::gram_aat_faer(a, b);
    }

    fn solve_unit_lower_in_place(ld: MatRef<'_, Self>, v: faer::MatMut<'_, Self>) {
        crate::linalg::solve_unit_lower_faer(ld, v);
    }

    fn ldlt_delete_row_col(
        ld: faer::MatMut<'_, Self>,
        index: usize,
        n: usize,
        scratch: &mut dyn_stack::MemBuffer,
    ) {
        crate::linalg::ldlt_delete_faer(ld, index, n, scratch);
    }
}

impl sealed::ScalarOps for f32 {
    type RowCast = Vec<f32>;
    type ColCast = Mat<f32>;

    const ROUNDS_FROM_F64: bool = true;
    const DTYPE: safetensors::Dtype = safetensors::Dtype::F32;

    fn empty_rows() -> Self::RowCast {
        Vec::new()
    }

    fn empty_cols() -> Self::ColCast {
        Mat::zeros(0, 0)
    }

    fn storage_rows<'a>(y: &'a [f64], cast: &'a mut Self::RowCast) -> &'a [Self] {
        if cast.len() != y.len() {
            cast.resize(y.len(), 0.0);
        }
        for (slot, &value) in cast.iter_mut().zip(y.iter()) {
            *slot = value as f32;
        }
        cast.as_slice()
    }

    fn storage_cols<'a>(x: MatRef<'a, f64>, cast: &'a mut Self::ColCast) -> MatRef<'a, Self> {
        if cast.nrows() != x.nrows() || cast.ncols() != x.ncols() {
            *cast = Mat::zeros(x.nrows(), x.ncols());
        }
        for col in 0..x.ncols() {
            for row in 0..x.nrows() {
                cast[(row, col)] = x[(row, col)] as f32;
            }
        }
        cast.as_ref()
    }

    #[inline(always)]
    fn fast_exp(self) -> Self {
        math::fast_exp_f32(self)
    }

    #[inline(always)]
    fn fast_jet(self) -> ExpJet<Self> {
        math::fast_jet_f32(self)
    }

    fn cholesky_lower(
        a: &mut Mat<Self>,
        _scratch: &mut dyn_stack::MemBuffer,
        jitter: f64,
        stage: crate::error::CholeskyStage,
    ) -> Result<(), crate::error::GprError> {
        crate::linalg::cholesky_lower_f64_accum(a, jitter, stage)
    }

    fn solve_llt_in_place(
        l: MatRef<'_, Self>,
        rhs: faer::MatMut<'_, Self>,
        _scratch: &mut dyn_stack::MemBuffer,
    ) {
        crate::linalg::solve_llt_f64_accum(l, rhs);
    }

    fn solve_llt_owned_scratch(l: MatRef<'_, Self>, rhs: faer::MatMut<'_, Self>) {
        crate::linalg::solve_llt_f64_accum(l, rhs);
    }

    fn inv_diag_from_chol_l(l: MatRef<'_, Self>, q_diag: &mut [Self]) {
        crate::linalg::inv_diag_from_chol_l_f64_accum(l, q_diag);
    }

    fn matvec_columns(a: MatRef<'_, Self>, y: &[Self], ay: faer::MatMut<'_, Self>) {
        crate::linalg::matvec_columns_f64_accum(a, y, ay);
    }

    fn gram_aat(a: MatRef<'_, Self>, b: faer::MatMut<'_, Self>) {
        crate::linalg::gram_aat_f64_accum(a, b);
    }

    fn solve_unit_lower_in_place(ld: MatRef<'_, Self>, v: faer::MatMut<'_, Self>) {
        crate::linalg::solve_unit_lower_f64_accum(ld, v);
    }

    fn ldlt_delete_row_col(
        ld: faer::MatMut<'_, Self>,
        index: usize,
        n: usize,
        _scratch: &mut dyn_stack::MemBuffer,
    ) {
        crate::linalg::ldlt_delete_via_f64(ld, index, n);
    }
}
