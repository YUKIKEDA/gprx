//! Compute scalar for [`super::CompiledKernel`] and the stored Gram factors.

use std::fmt;

use faer::{Mat, MatRef};

use crate::math::{self, ExpJet};

pub(crate) mod sealed {
    use dyn_stack::MemBuffer;
    use faer::{Mat, MatMut, MatRef};

    use crate::error::{CholeskyStage, GprError};
    use crate::kernel::KernelTerm;
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

        /// Whether fit fills and reads the ARD `(Δx_d)²` cache at this scalar.
        const READS_ARD_CACHE: bool;

        /// Writes pairwise squared Euclidean distances of the rows of `x`.
        fn write_squared(x: MatRef<'_, Self>, dist: MatMut<'_, Self>, scratch: &mut [Mat<Self>]);

        /// Writes train × test squared Euclidean distances.
        fn write_cross(
            x_train: MatRef<'_, Self>,
            x_test: MatRef<'_, Self>,
            dist: MatMut<'_, Self>,
            scratch: &mut [Mat<Self>],
        );

        /// Writes the packed raw `(Δx_d)²` cache of
        /// [`crate::kernel::dist::ArdSqDiffBuf`].
        fn write_ard(x: MatRef<'_, Self>, cache: &mut [Self]);

        /// This slice as `f64` when the scalar is `f64`, for the SIMD paths.
        fn as_f64_slice(values: &[Self]) -> Option<&[f64]>;

        /// [`Self::as_f64_slice`] for a mutable slice.
        fn as_f64_slice_mut(values: &mut [Self]) -> Option<&mut [f64]>;

        fn empty_rows() -> Self::RowCast;

        fn empty_cols() -> Self::ColCast;

        /// Views `y` as this scalar. `f64` returns `y`. `f32` fills `cast`.
        fn storage_rows<'a>(y: &'a [f64], cast: &'a mut Self::RowCast) -> &'a [Self];

        /// Views `x` as this scalar. `f64` returns `x`. `f32` fills `cast`.
        fn storage_cols<'a>(x: MatRef<'a, f64>, cast: &'a mut Self::ColCast) -> MatRef<'a, Self>;

        /// This view as `f64` when the scalar is `f64`, for the SIMD paths.
        fn as_f64_ref(m: MatRef<'_, Self>) -> Option<MatRef<'_, f64>>;

        /// This view as `f64` when the scalar is `f64`, for the SIMD paths.
        fn as_f64_mut(m: MatMut<'_, Self>) -> Option<MatMut<'_, f64>>;

        /// Picks a [`crate::kernel::CustomKernel`] leaf's implementation at this scalar.
        fn pick_term<'a>(
            f64_term: &'a dyn KernelTerm<f64>,
            f32_term: &'a dyn KernelTerm<f32>,
        ) -> &'a dyn KernelTerm<Self>
        where
            Self: crate::kernel::KernelScalar;

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

        /// Solves `L w = v` in place for unit-lower `L` (`n = w.len()`), where
        /// row `i` of `L` is the head `lt[0..i, i]` of column `i` of `Lᵀ`.
        fn solve_unit_lower_rows(lt: &faer::Mat<Self>, w: &mut [Self]);

        /// Solves `L D Lᵀ x = b` for the leading `n` of the unit-lower `ld`
        /// (`D` on its diagonal), in place on every column of `rhs`.
        fn solve_ldlt_in_place(ld: MatRef<'_, Self>, rhs: MatMut<'_, Self>, n: usize);

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

    /// `selfᵉ` for a real exponent.
    fn powf(self, e: Self) -> Self;

    /// Sine (radians).
    fn sin(self) -> Self;

    /// Cosine (radians).
    fn cos(self) -> Self;

    /// The larger of two values (`NaN` loses).
    fn max(self, other: Self) -> Self;

    /// The smaller of two values (`NaN` loses).
    fn min(self, other: Self) -> Self;
}

/// `(src, dest)` as `f64` views when `T` is `f64`, for the SIMD paths.
#[inline(always)]
pub(crate) fn f64_pair<'a, T: KernelScalar>(
    src: MatRef<'a, T>,
    dest: faer::MatMut<'a, T>,
) -> Option<(MatRef<'a, f64>, faer::MatMut<'a, f64>)> {
    Some((T::as_f64_ref(src)?, T::as_f64_mut(dest)?))
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

            #[inline(always)]
            fn powf(self, e: Self) -> Self {
                <$t>::powf(self, e)
            }

            #[inline(always)]
            fn sin(self) -> Self {
                <$t>::sin(self)
            }

            #[inline(always)]
            fn cos(self) -> Self {
                <$t>::cos(self)
            }

            #[inline(always)]
            fn max(self, other: Self) -> Self {
                <$t>::max(self, other)
            }

            #[inline(always)]
            fn min(self, other: Self) -> Self {
                <$t>::min(self, other)
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
    const READS_ARD_CACHE: bool = true;

    fn write_squared(x: MatRef<'_, Self>, dist: faer::MatMut<'_, Self>, scratch: &mut [Mat<Self>]) {
        super::dist::fill_squared_euclidean(x, dist, scratch);
    }

    fn write_cross(
        x_train: MatRef<'_, Self>,
        x_test: MatRef<'_, Self>,
        dist: faer::MatMut<'_, Self>,
        scratch: &mut [Mat<Self>],
    ) {
        super::dist::fill_squared_euclidean_cross(x_train, x_test, dist, scratch);
    }

    fn write_ard(x: MatRef<'_, Self>, cache: &mut [Self]) {
        super::dist::fill_ard_squared_diff(x, cache);
    }

    fn as_f64_slice(values: &[Self]) -> Option<&[f64]> {
        Some(values)
    }

    fn as_f64_slice_mut(values: &mut [Self]) -> Option<&mut [f64]> {
        Some(values)
    }

    fn empty_rows() -> Self::RowCast {}

    fn empty_cols() -> Self::ColCast {}

    fn storage_rows<'a>(y: &'a [f64], _cast: &'a mut Self::RowCast) -> &'a [Self] {
        y
    }

    fn storage_cols<'a>(x: MatRef<'a, f64>, _cast: &'a mut Self::ColCast) -> MatRef<'a, Self> {
        x
    }

    fn pick_term<'a>(
        f64_term: &'a dyn crate::kernel::KernelTerm<f64>,
        _f32_term: &'a dyn crate::kernel::KernelTerm<f32>,
    ) -> &'a dyn crate::kernel::KernelTerm<Self> {
        f64_term
    }

    #[inline(always)]
    fn as_f64_ref(m: MatRef<'_, Self>) -> Option<MatRef<'_, f64>> {
        Some(m)
    }

    #[inline(always)]
    fn as_f64_mut(m: faer::MatMut<'_, Self>) -> Option<faer::MatMut<'_, f64>> {
        Some(m)
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

    fn solve_unit_lower_rows(lt: &faer::Mat<Self>, w: &mut [Self]) {
        crate::linalg::solve_unit_lower_rows_f64(lt, w);
    }

    fn solve_ldlt_in_place(ld: MatRef<'_, Self>, rhs: faer::MatMut<'_, Self>, n: usize) {
        crate::linalg::solve_ldlt_faer(ld, rhs, n);
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
    const READS_ARD_CACHE: bool = false;

    fn write_squared(
        x: MatRef<'_, Self>,
        dist: faer::MatMut<'_, Self>,
        _scratch: &mut [Mat<Self>],
    ) {
        super::dist::fill_squared_scalar(x, dist);
    }

    fn write_cross(
        x_train: MatRef<'_, Self>,
        x_test: MatRef<'_, Self>,
        dist: faer::MatMut<'_, Self>,
        _scratch: &mut [Mat<Self>],
    ) {
        super::dist::fill_cross_scalar(x_train, x_test, dist);
    }

    fn write_ard(x: MatRef<'_, Self>, cache: &mut [Self]) {
        super::dist::fill_ard_scalar(x, cache);
    }

    fn as_f64_slice(_values: &[Self]) -> Option<&[f64]> {
        None
    }

    fn as_f64_slice_mut(_values: &mut [Self]) -> Option<&mut [f64]> {
        None
    }

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

    fn pick_term<'a>(
        _f64_term: &'a dyn crate::kernel::KernelTerm<f64>,
        f32_term: &'a dyn crate::kernel::KernelTerm<f32>,
    ) -> &'a dyn crate::kernel::KernelTerm<Self> {
        f32_term
    }

    #[inline(always)]
    fn as_f64_ref(_m: MatRef<'_, Self>) -> Option<MatRef<'_, f64>> {
        None
    }

    #[inline(always)]
    fn as_f64_mut(_m: faer::MatMut<'_, Self>) -> Option<faer::MatMut<'_, f64>> {
        None
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

    fn solve_unit_lower_rows(lt: &faer::Mat<Self>, w: &mut [Self]) {
        crate::linalg::solve_unit_lower_rows_f32(lt, w);
    }

    fn solve_ldlt_in_place(ld: MatRef<'_, Self>, rhs: faer::MatMut<'_, Self>, n: usize) {
        crate::linalg::solve_ldlt_f64_accum(ld, rhs, n);
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
