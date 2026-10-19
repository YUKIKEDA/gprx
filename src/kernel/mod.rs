//! Describes kernel leaves and composition ([`KernelSpec`] / [`CompiledKernel`]).
//!
//! Built-in leaves are enum arms. User distance leaves implement [`KernelTerm`]
//! and enter the tree as [`KernelSpec::Custom`].
//!
//! Each built-in leaf stores its positive parameters as `log`. `new` takes
//! user units, and the `from_log_*` constructors take the optimizer
//! coordinate. `bounds` / `with_bounds` use an open [`Interval`](crate::Interval)
//! (default [`Interval::DEFAULT_POSITIVE`](crate::Interval::DEFAULT_POSITIVE),
//! `(1e-5, 1e5)`).
//!
//! | Leaf | Construct | Parameters |
//! | --- | --- | --- |
//! | [`RbfKernel`] | `new(ℓ)` | isotropic lengthscale |
//! | [`RbfArdKernel`] | `new(&[ℓ_d])` | one lengthscale per feature ([`ArdLengthscales`]) |
//! | [`MaternKernel`] | `new(ℓ, nu)` | lengthscale; [`MaternNu`] `Half`, `ThreeHalves`, or `FiveHalves` is fixed, not optimized |
//! | [`MaternArdKernel`] | `new(&[ℓ_d], nu)` | ARD lengthscales, fixed `ν` |
//! | [`PeriodicKernel`] | `new(ℓ, period)` | lengthscale and period |
//! | [`RationalQuadraticKernel`] | `new(ℓ, alpha)` | lengthscale and `α` |
//! | [`RationalQuadraticArdKernel`] | `new(&[ℓ_d], alpha)` | ARD lengthscales, then `α` |
//! | [`ConstantKernel`] | `new(c)` | signal variance; multiply by another leaf for `c · k` |
//! | [`LinearKernel`] | `new(variance)` | `σ² xᵀ x'` |
//! | [`WhiteKernel`] | `new(variance)` | diagonal nugget |
//!
//! Observation noise belongs in [`GaussianLikelihood`](crate::GaussianLikelihood);
//! a large likelihood noise and a large [`WhiteKernel`] count the noise twice.
//!
//! The evaluators of a leaf and of a [`CompiledKernel`] (`apply`,
//! `apply_cross`, `grad`, `hess`, …) take `faer` matrix views (`MatRef` /
//! `MatMut`): add `faer = "0.24"` to `Cargo.toml` to call them. A leaf
//! evaluates `exp` with [`Accurate`](crate::Accurate); a [`CompiledKernel`]
//! takes the [`KernelMath`](crate::KernelMath) type as its `M` parameter.
//!
//! A model can also read squared distances the caller supplies instead of
//! computing them from coordinates. A slot ([`ScalarDistance`], or
//! [`ArdDistance`] with one table per dimension) names such a supply, its
//! leaves form a [`DistanceKernel`], and a [`DistanceSource`] binds the
//! tables, or a [`DistanceFill`] that writes them, to the slot for one call.
//! [`DistanceKernel<WithPoints>`] also holds coordinate leaves of a
//! [`KernelSpec`]. [`ModelKernel`] is what the models accept: a coordinate
//! [`KernelSpec`] or a [`DistanceKernel`].
//!
//! A model of a [`DistanceKernel`] takes one [`DistanceSource`] per slot
//! instead of coordinates (and as well as them for [`WithPoints`]). `n` is
//! the training count, `q` the query count, and `inducing` the training
//! indices of the inducing points of [`Sgpr`](crate::Sgpr) /
//! [`Svgp`](crate::Svgp):
//!
//! | Model | `fit` / `factor` | `predict` family | Covariance and `sample` |
//! | --- | --- | --- | --- |
//! | `Gpr`, [`DistanceOnly`] | `(sources, n, y)` | `(sources, q)` | `(cross, square, q)` |
//! | `Gpr`, [`WithPoints`] | `(sources, n, x, n_cols, y)` | `(sources, xs, q, n_cols)` | `(cross, square, xs, q, n_cols)` |
//! | `Sgpr` / `Svgp`, [`DistanceOnly`] | `(sources, n, y, inducing)` | `(sources, q)` | `(cross, square, q)` |
//! | `Sgpr` / `Svgp`, [`WithPoints`] | `(sources, n, x, n_cols, y, inducing)` | `(sources, xs, q, n_cols)` | `(cross, square, xs, q, n_cols)` |
//!
//! A training source of `Gpr` is the `n × n` square among the training
//! points; one of `Sgpr` / `Svgp` is the `n × m` block from the training
//! points to the inducing points. A prediction source (`cross`) is the
//! `n × q` block from the training points to the queries for `Gpr`, and the
//! `m × q` block from the inducing points for `Sgpr` / `Svgp`; `square` is
//! the `q × q` block among the queries.
//!
//! Isotropic RBF, Matérn, Periodic, and rational quadratic evaluate from a
//! squared-Euclidean distance matrix (Periodic then takes the square root). ARD RBF, ARD
//! Matérn, and ARD rational quadratic evaluate from coordinates via [`ArdLengthscales`]
//! (`θ_d = log(ℓ_d)`). Callers pass column-major matrix views; this module does not
//! re-export matrix-library types.
//!
//! Squared-Euclidean fills and [`Triangle::Lower`] writes run on the process-wide thread
//! pool. Isotropic RBF, ARD RBF, and the distance / `(Δx_d)²` row loops use four-wide
//! `f64` SIMD when the matrix view is column-major with unit row stride. There is no
//! parallel on/off flag; `RAYON_NUM_THREADS=1` is sequential. Limit threads with
//! `RAYON_NUM_THREADS` or the global thread-pool builder before the first fill. Linear and
//! other points-mode leaves stay sequential. See the [crate-level parallelism
//! notes](crate).

mod ard;
mod compiled;
mod constant;
mod dist;
mod leaf_params;
mod lengthscale;
mod linear;
mod matern;
mod matern_ard;
mod periodic;
mod radial;
mod rbf;
mod rbf_ard;
mod rq;
mod rq_ard;
mod scalar;
mod simd;
mod sources;
mod spec;
mod supply;
mod term;
mod tree;
mod white;

pub use compiled::CompiledKernel;
pub(crate) use compiled::CrossViews;
pub(crate) use compiled::ensure_nested_levels;
pub(crate) use compiled::gram::GramInputs;
pub(crate) use compiled::supplied::unbound;
pub(crate) use compiled::weighted::{DiagAccum, WeightedWalk};
pub use constant::ConstantKernel;
pub(crate) use dist::ArdSqDiffBuf;
#[cfg(any(test, feature = "bench-internals"))]
pub(crate) use dist::fill_squared_euclidean;
pub(crate) use dist::packed_len;
pub use lengthscale::ArdLengthscales;
pub use linear::LinearKernel;
pub use matern::{MaternKernel, MaternNu};
pub use matern_ard::MaternArdKernel;
pub use periodic::PeriodicKernel;
pub use rbf::RbfKernel;
pub use rbf_ard::RbfArdKernel;
pub use rq::RationalQuadraticKernel;
pub use rq_ard::RationalQuadraticArdKernel;
pub use scalar::KernelScalar;
pub(crate) use scalar::sealed::ScalarOps;
pub(crate) use sources::BlockAt;
pub(crate) use sources::{
    BlockStore, PackedRuns, QueryScratch, QuerySources, RefinedSources, SourceStore, TrainSources,
    bind_inducing, column_into, lower_column, new_inducing_column, place_in_shape, widened,
};
pub use spec::{KernelSpec, ParameterBinding};
pub(crate) use supply::{
    ArdData, CompiledOf, ModelKernelParts, ScalarData, SlotId, SlotShape, SourceData, SpecOf,
    SuppliedLeafSpec, SuppliedSpec, Tidy, spec_slots,
};
pub use supply::{
    ArdDistance, ArdDistanceLeaf, DistanceFill, DistanceKernel, DistanceOnly, DistanceSlot,
    DistanceSource, JoinPoints, ModelKernel, PointKernel, PointUse, ScalarDistance,
    ScalarDistanceLeaf, WithPoints,
};
pub use term::{CustomKernel, KernelTerm};
pub(crate) use tree::sealed::Supply as SupplyViews;

/// What a model of kernel `K` holds only when `K` reads supplied distances
/// ([`SupplyViews::Held`]): `X`, or nothing for a coordinate kernel.
pub(crate) type Held<K, X> = <<K as ModelKernelParts>::Supply as SupplyViews>::Held<X>;
pub use tree::{NoSupply, Supply};
pub use white::WhiteKernel;

use crate::error::GprError;
use dist::par_lower_cols;
pub(crate) use dist::worker_count;
use faer::{MatMut, MatRef};

/// Records which triangle of a symmetric kernel matrix to write.
///
/// # Examples
///
/// ```rust
/// use faer::Mat;
/// use gprx::kernel::{RbfKernel, Triangle};
///
/// # fn main() -> Result<(), gprx::GprError> {
/// assert_ne!(Triangle::Lower, Triangle::Upper);
/// let rbf = RbfKernel::new(1.0)?;
/// let dist = Mat::from_fn(2, 2, |i, j| if i == j { 0.0 } else { 1.0 });
/// // `Lower` leaves the strict upper triangle as it was.
/// let mut lower = Mat::<f64>::zeros(2, 2);
/// rbf.apply(dist.as_ref(), lower.as_mut(), Triangle::Lower)?;
/// assert!(lower[(0, 1)].abs() < 1e-12 && lower[(1, 0)] > 0.0);
/// // `Upper` writes the other side, `Full` both.
/// let mut upper = Mat::<f64>::zeros(2, 2);
/// rbf.apply(dist.as_ref(), upper.as_mut(), Triangle::Upper)?;
/// assert!(upper[(1, 0)].abs() < 1e-12 && upper[(0, 1)] > 0.0);
/// let mut full = Mat::<f64>::zeros(2, 2);
/// rbf.apply(dist.as_ref(), full.as_mut(), Triangle::Full)?;
/// assert!((full[(0, 1)] - full[(1, 0)]).abs() < 1e-12);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Triangle {
    /// Marks entries with `row >= col`.
    ///
    /// This is the default for Cholesky.
    Lower,
    /// Marks entries with `row <= col`.
    Upper,
    /// Marks every entry.
    ///
    /// Upper and lower must match when `dist` is symmetric.
    Full,
}

pub(crate) fn visit_triangle(n: usize, uplo: Triangle, mut visit: impl FnMut(usize, usize)) {
    match uplo {
        Triangle::Lower => {
            for col in 0..n {
                for row in col..n {
                    visit(row, col);
                }
            }
        }
        Triangle::Upper => {
            for col in 0..n {
                for row in 0..=col {
                    visit(row, col);
                }
            }
        }
        Triangle::Full => {
            for col in 0..n {
                for row in 0..n {
                    visit(row, col);
                }
            }
        }
    }
}

fn require_square_pair<T>(dist: MatRef<'_, T>, out: MatRef<'_, T>) -> Result<usize, GprError> {
    if dist.nrows() != dist.ncols() {
        return Err(GprError::ShapeMismatch {
            reason: format!(
                "distance matrix must be square, got {}x{}",
                dist.nrows(),
                dist.ncols()
            ),
        });
    }
    if out.nrows() != dist.nrows() || out.ncols() != dist.ncols() {
        return Err(GprError::ShapeMismatch {
            reason: format!(
                "output is {}x{}, expected {}x{}",
                out.nrows(),
                out.ncols(),
                dist.nrows(),
                dist.ncols()
            ),
        });
    }
    if dist.nrows() == 0 {
        return Err(GprError::EmptyInput);
    }
    Ok(dist.nrows())
}

pub(crate) fn require_coord_grad<T>(
    x1: MatRef<'_, T>,
    x2: MatRef<'_, T>,
    d_k: MatRef<'_, T>,
    dim: usize,
) -> Result<(), GprError> {
    if x1.nrows() == 0 || x2.nrows() == 0 || x1.ncols() == 0 {
        return Err(GprError::EmptyInput);
    }
    if x1.ncols() != x2.ncols() {
        return Err(GprError::DimensionMismatch {
            x_dim: x2.ncols(),
            expected_dim: x1.ncols(),
        });
    }
    if dim >= x1.ncols() {
        return Err(GprError::IndexOutOfRange {
            reason: format!(
                "coordinate dimension {dim} is out of range for d={}",
                x1.ncols()
            ),
        });
    }
    if d_k.nrows() != x1.nrows() || d_k.ncols() != x2.nrows() {
        return Err(GprError::ShapeMismatch {
            reason: format!(
                "output is {}x{}, expected {}x{}",
                d_k.nrows(),
                d_k.ncols(),
                x1.nrows(),
                x2.nrows()
            ),
        });
    }
    Ok(())
}

fn require_same_shape<T>(dist: MatRef<'_, T>, out: MatRef<'_, T>) -> Result<(), GprError> {
    if out.nrows() != dist.nrows() || out.ncols() != dist.ncols() {
        return Err(GprError::ShapeMismatch {
            reason: format!(
                "output is {}x{}, expected {}x{}",
                out.nrows(),
                out.ncols(),
                dist.nrows(),
                dist.ncols()
            ),
        });
    }
    if dist.nrows() == 0 || dist.ncols() == 0 {
        return Err(GprError::EmptyInput);
    }
    Ok(())
}

fn write_dense<T: KernelScalar>(
    dist: MatRef<'_, T>,
    mut out: MatMut<'_, T>,
    mut kernel: impl FnMut(T) -> Result<T, GprError>,
) -> Result<(), GprError> {
    require_same_shape(dist, out.as_ref())?;
    let mut err = None;
    for col in 0..dist.ncols() {
        for row in 0..dist.nrows() {
            if err.is_some() {
                continue;
            }
            match kernel(dist[(row, col)]) {
                Ok(value) => out[(row, col)] = value,
                Err(e) => err = Some(e),
            }
        }
    }
    match err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

fn write_triangle<T: KernelScalar>(
    dist: MatRef<'_, T>,
    mut out: MatMut<'_, T>,
    uplo: Triangle,
    kernel: impl Fn(T) -> Result<T, GprError> + Sync,
) -> Result<(), GprError> {
    let n = require_square_pair(dist, out.as_ref())?;
    if n > 0 && matches!(uplo, Triangle::Lower) {
        return write_lower_parallel(dist, out, kernel);
    }
    let mut err = None;
    visit_triangle(n, uplo, |row, col| {
        if err.is_some() {
            return;
        }
        match kernel(dist[(row, col)]) {
            Ok(value) => out[(row, col)] = value,
            Err(e) => err = Some(e),
        }
    });
    match err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

fn write_lower_parallel<T: KernelScalar>(
    dist: MatRef<'_, T>,
    out: MatMut<'_, T>,
    kernel: impl Fn(T) -> Result<T, GprError> + Sync,
) -> Result<(), GprError> {
    par_lower_cols(out, &|col, mut rows| {
        for i in 0..rows.nrows() {
            rows[i] = kernel(dist[(col + i, col)])?;
        }
        Ok(())
    })
}

/// Returns `value` if it is finite, else [`GprError::NonFiniteKernelValue`].
#[inline(always)]
pub(crate) fn finite_kernel<T: KernelScalar>(value: T) -> Result<T, GprError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(GprError::NonFiniteKernelValue)
    }
}

fn finite_dist<T: KernelScalar>(d: T) -> Result<T, GprError> {
    if d.is_finite() {
        Ok(d)
    } else {
        Err(GprError::NonFiniteInput)
    }
}

pub(crate) fn pair_squared_euclidean<T: KernelScalar>(x: MatRef<'_, T>, i: usize, j: usize) -> T {
    let d = x.ncols();
    let mut sum = T::from_f64(0.0);
    for dim in 0..d {
        let diff = x[(i, dim)] - x[(j, dim)];
        sum += diff * diff;
    }
    sum
}

/// `‖x1_row − x2_col‖²` of two point sets.
pub(crate) fn cross_squared_euclidean<T: KernelScalar>(
    x1: MatRef<'_, T>,
    row: usize,
    x2: MatRef<'_, T>,
    col: usize,
) -> T {
    let mut sum = T::from_f64(0.0);
    for dim in 0..x1.ncols() {
        let diff = x1[(row, dim)] - x2[(col, dim)];
        sum += diff * diff;
    }
    sum
}

/// Writes a rectangular kernel block (`x1.nrows() × x2.nrows()`) from
/// coordinates: `kernel(‖x1_i − x2_j‖²)` per entry, without a distance matrix.
pub(crate) fn write_rect_from_coords<T: KernelScalar>(
    x1: MatRef<'_, T>,
    x2: MatRef<'_, T>,
    out: MatMut<'_, T>,
    kernel: impl Fn(T) -> Result<T, GprError>,
) -> Result<(), GprError> {
    require_coord_grad(x1, x2, out.as_ref(), 0)?;
    write_rect(out, |row, col| {
        kernel(finite_dist(cross_squared_euclidean(x1, row, x2, col))?)
    })
}

/// Writes a kernel triangle from coordinates. Each pair computes `‖x_i-x_j‖²`
/// without a distance matrix.
pub(crate) fn write_square_from_coords<T: KernelScalar>(
    x: MatRef<'_, T>,
    out: MatMut<'_, T>,
    uplo: Triangle,
    kernel: impl Fn(T) -> Result<T, GprError> + Sync,
) -> Result<(), GprError> {
    let n = out.nrows();
    if out.ncols() != n {
        return Err(GprError::ShapeMismatch {
            reason: format!("output must be square, got {}x{}", out.nrows(), out.ncols()),
        });
    }
    if n == 0 || x.nrows() != n || x.ncols() == 0 {
        return Err(GprError::EmptyInput);
    }
    if matches!(uplo, Triangle::Lower) {
        return write_lower_from_coords_parallel(x, out, kernel);
    }
    write_square(out, uplo, |row, col| {
        kernel(finite_dist(pair_squared_euclidean(x, row, col))?)
    })
}

fn write_lower_from_coords_parallel<T: KernelScalar>(
    x: MatRef<'_, T>,
    out: MatMut<'_, T>,
    kernel: impl Fn(T) -> Result<T, GprError> + Sync,
) -> Result<(), GprError> {
    par_lower_cols(out, &|col, mut rows| {
        for i in 0..rows.nrows() {
            rows[i] = kernel(finite_dist(pair_squared_euclidean(x, col + i, col))?)?;
        }
        Ok(())
    })
}

/// Writes every entry of a rectangular `out`. `pair(row, col)` is the value.
pub(crate) fn write_rect<T: KernelScalar>(
    mut out: MatMut<'_, T>,
    mut pair: impl FnMut(usize, usize) -> Result<T, GprError>,
) -> Result<(), GprError> {
    for col in 0..out.ncols() {
        for row in 0..out.nrows() {
            out[(row, col)] = pair(row, col)?;
        }
    }
    Ok(())
}

fn write_square<T: KernelScalar>(
    mut out: MatMut<'_, T>,
    uplo: Triangle,
    mut kernel: impl FnMut(usize, usize) -> Result<T, GprError>,
) -> Result<(), GprError> {
    if out.nrows() != out.ncols() {
        return Err(GprError::ShapeMismatch {
            reason: format!("output must be square, got {}x{}", out.nrows(), out.ncols()),
        });
    }
    if out.nrows() == 0 {
        return Err(GprError::EmptyInput);
    }
    let n = out.nrows();
    let mut err = None;
    visit_triangle(n, uplo, |row, col| {
        if err.is_some() {
            return;
        }
        match kernel(row, col) {
            Ok(value) => out[(row, col)] = value,
            Err(e) => err = Some(e),
        }
    });
    match err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

fn validate_positive_finite(value: f64, what: &str) -> Result<(), GprError> {
    if !value.is_finite() {
        return Err(GprError::InvalidHyperparameter {
            reason: format!("{what} must be finite"),
        });
    }
    if value <= 0.0 {
        return Err(GprError::InvalidHyperparameter {
            reason: format!("{what} must be positive"),
        });
    }
    Ok(())
}

fn validate_log_positive(theta: f64, what: &str) -> Result<f64, GprError> {
    if !theta.is_finite() {
        return Err(GprError::InvalidHyperparameter {
            reason: format!("log {what} must be finite"),
        });
    }
    let value = theta.exp();
    if !value.is_finite() {
        return Err(GprError::InvalidHyperparameter {
            reason: format!("{what} overflowed to a non-finite value"),
        });
    }
    if value <= 0.0 {
        return Err(GprError::InvalidHyperparameter {
            reason: format!("{what} underflowed to zero"),
        });
    }
    Ok(theta)
}
