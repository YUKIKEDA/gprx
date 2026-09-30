//! Coordinate derivatives (`Sgpr<FreeInducing>`) of the leaves that are not
//! radial, of `Custom`, and of Product trees.
//!
//! Radial leaves are in [`crate::kernel::radial`]. A Product differentiates by
//! the product rule over each term's value and first and second derivative.

use super::CompiledKernel;
use crate::error::GprError;
use crate::kernel::{
    ConstantKernel, CustomKernel, KernelScalar, LinearKernel, cross_squared_euclidean,
    require_coord_grad, write_rect,
};
use faer::{Mat, MatMut, MatRef};

/// Writes zeros (`Constant` has no coordinate dependence).
pub(super) fn zero_block<T: KernelScalar>(
    x1: MatRef<'_, T>,
    x2: MatRef<'_, T>,
    mut out: MatMut<'_, T>,
    dims: &[usize],
) -> Result<(), GprError> {
    for &dim in dims {
        require_coord_grad(x1, x2, out.as_ref(), dim)?;
    }
    fill(out.as_mut(), T::from_f64(0.0));
    Ok(())
}

pub(super) fn constant_param(leaf: &ConstantKernel, param: usize) -> Result<(), GprError> {
    if param >= leaf.num_params() {
        return Err(GprError::IndexOutOfRange {
            reason: format!("constant kernel has a single parameter; got index {param}"),
        });
    }
    Ok(())
}

fn fill<T: KernelScalar>(mut out: MatMut<'_, T>, value: T) {
    for col in 0..out.ncols() {
        for row in 0..out.nrows() {
            out[(row, col)] = value;
        }
    }
}

/// `∂(σ² x1·x2)/∂x2[*, dim] = σ² x1[*, dim]`, and the same for `∂/∂θ` of it
/// (`θ = log σ²`).
pub(super) fn linear_grad_dim<T: KernelScalar>(
    leaf: &LinearKernel,
    x1: MatRef<'_, T>,
    x2: MatRef<'_, T>,
    d_k: MatMut<'_, T>,
    dim: usize,
) -> Result<(), GprError> {
    require_coord_grad(x1, x2, d_k.as_ref(), dim)?;
    let var = T::from_f64(leaf.variance());
    write_rect(d_k, |row, _| Ok(var * x1[(row, dim)]))
}

/// `∂²(σ² x1·x2)/∂x1[*, a] ∂x2[*, b] = σ² [a = b]`.
pub(super) fn linear_mixed<T: KernelScalar>(
    leaf: &LinearKernel,
    x1: MatRef<'_, T>,
    x2: MatRef<'_, T>,
    d2_k: MatMut<'_, T>,
    (dim_x1, dim_x2): (usize, usize),
) -> Result<(), GprError> {
    require_coord_grad(x1, x2, d2_k.as_ref(), dim_x1)?;
    require_coord_grad(x1, x2, d2_k.as_ref(), dim_x2)?;
    let value = if dim_x1 == dim_x2 {
        leaf.variance()
    } else {
        0.0
    };
    fill(d2_k, T::from_f64(value));
    Ok(())
}

pub(super) fn linear_param(param: usize) -> Result<(), GprError> {
    if param != 0 {
        return Err(GprError::IndexOutOfRange {
            reason: "linear kernel has a single parameter at index 0".to_owned(),
        });
    }
    Ok(())
}

/// The squared distances of a rectangular block.
fn sq_dist<T: KernelScalar>(x1: MatRef<'_, T>, x2: MatRef<'_, T>) -> Mat<T> {
    Mat::from_fn(x1.nrows(), x2.nrows(), |row, col| {
        cross_squared_euclidean(x1, row, x2, col)
    })
}

/// `∂K/∂x2[*, dim] = −2 k' Δ_dim` of a `Custom` leaf, `k' = ∂k/∂(d²)`.
pub(super) fn custom_grad_dim<T: KernelScalar>(
    leaf: &CustomKernel<T>,
    x1: MatRef<'_, T>,
    x2: MatRef<'_, T>,
    d_k: MatMut<'_, T>,
    dim: usize,
) -> Result<(), GprError> {
    require_coord_grad(x1, x2, d_k.as_ref(), dim)?;
    let dist = sq_dist(x1, x2);
    let mut g1 = Mat::zeros(dist.nrows(), dist.ncols());
    leaf.grad_wrt_sq_dist(dist.as_ref(), g1.as_mut())?;
    write_rect(d_k, |row, col| {
        Ok(T::from_f64(-2.0) * g1[(row, col)] * (x1[(row, dim)] - x2[(col, dim)]))
    })
}

/// `4 k'' Δ_a Δ_b + 2 k' [a = b]` (negated when `mixed`).
pub(super) fn custom_hess_dims<T: KernelScalar>(
    leaf: &CustomKernel<T>,
    x1: MatRef<'_, T>,
    x2: MatRef<'_, T>,
    d2_k: MatMut<'_, T>,
    (dim_a, dim_b): (usize, usize),
    mixed: bool,
) -> Result<(), GprError> {
    require_coord_grad(x1, x2, d2_k.as_ref(), dim_a)?;
    require_coord_grad(x1, x2, d2_k.as_ref(), dim_b)?;
    let dist = sq_dist(x1, x2);
    let mut g1 = Mat::zeros(dist.nrows(), dist.ncols());
    let mut g2 = Mat::zeros(dist.nrows(), dist.ncols());
    leaf.grad_wrt_sq_dist(dist.as_ref(), g1.as_mut())?;
    leaf.hess_wrt_sq_dist(dist.as_ref(), g2.as_mut())?;
    let sign = T::from_f64(if mixed { -1.0 } else { 1.0 });
    write_rect(d2_k, |row, col| {
        let da = x1[(row, dim_a)] - x2[(col, dim_a)];
        let db = x1[(row, dim_b)] - x2[(col, dim_b)];
        let mut value = T::from_f64(4.0) * g2[(row, col)] * da * db;
        if dim_a == dim_b {
            value += T::from_f64(2.0) * g1[(row, col)];
        }
        Ok(sign * value)
    })
}

/// `−2 Δ_dim ∂²k/∂θ ∂(d²)` of a `Custom` leaf.
pub(super) fn custom_theta_dim<T: KernelScalar>(
    leaf: &CustomKernel<T>,
    x1: MatRef<'_, T>,
    x2: MatRef<'_, T>,
    d2_k: MatMut<'_, T>,
    param: usize,
    dim: usize,
) -> Result<(), GprError> {
    require_coord_grad(x1, x2, d2_k.as_ref(), dim)?;
    let dist = sq_dist(x1, x2);
    let mut dg1 = Mat::zeros(dist.nrows(), dist.ncols());
    leaf.grad_wrt_sq_dist_theta(dist.as_ref(), dg1.as_mut(), param)?;
    write_rect(d2_k, |row, col| {
        Ok(T::from_f64(-2.0) * dg1[(row, col)] * (x1[(row, dim)] - x2[(col, dim)]))
    })
}

/// A `Custom` leaf's `∂K(x1, x2)/∂θ` and `∂²K(x1, x2)/∂θ_i ∂θ_j`.
pub(super) fn custom_cross_grad<T: KernelScalar>(
    leaf: &CustomKernel<T>,
    x1: MatRef<'_, T>,
    x2: MatRef<'_, T>,
    d_k: MatMut<'_, T>,
    param: usize,
) -> Result<(), GprError> {
    require_coord_grad(x1, x2, d_k.as_ref(), 0)?;
    leaf.grad_cross(sq_dist(x1, x2).as_ref(), d_k, param)
}

pub(super) fn custom_cross_hess<T: KernelScalar>(
    leaf: &CustomKernel<T>,
    x1: MatRef<'_, T>,
    x2: MatRef<'_, T>,
    d2_k: MatMut<'_, T>,
    (i, j): (usize, usize),
) -> Result<(), GprError> {
    require_coord_grad(x1, x2, d2_k.as_ref(), 0)?;
    leaf.hess_cross(sq_dist(x1, x2).as_ref(), d2_k, i, j)
}

/// A derivative direction of a Product's coordinate derivative.
#[derive(Clone, Copy)]
pub(super) enum Dir {
    /// `∂/∂x1[*, dim]`.
    X1(usize),
    /// `∂/∂x2[*, dim]`.
    X2(usize),
    /// `∂/∂θ_p`, `p` counted over the whole product.
    Theta(usize),
}

/// One term with its value and the derivatives a Product's rule needs
/// (`None` is identically zero).
struct TermJet<T> {
    v: Mat<T>,
    a: Option<Mat<T>>,
    b: Option<Mat<T>>,
    ab: Option<Mat<T>>,
}

fn hadamard<T: KernelScalar>(x: &Mat<T>, y: &Mat<T>) -> Mat<T> {
    Mat::from_fn(x.nrows(), x.ncols(), |row, col| {
        x[(row, col)] * y[(row, col)]
    })
}

/// `sum += x ∘ y` for the parts that are not zero.
fn accumulate<T: KernelScalar>(sum: &mut Option<Mat<T>>, x: Option<&Mat<T>>, y: Option<&Mat<T>>) {
    let (Some(x), Some(y)) = (x, y) else { return };
    let prod = hadamard(x, y);
    match sum {
        Some(total) => {
            for col in 0..total.ncols() {
                for row in 0..total.nrows() {
                    total[(row, col)] += prod[(row, col)];
                }
            }
        }
        None => *sum = Some(prod),
    }
}

fn combine<T: KernelScalar>(acc: &TermJet<T>, t: &TermJet<T>) -> TermJet<T> {
    let mut a = None;
    accumulate(&mut a, acc.a.as_ref(), Some(&t.v));
    accumulate(&mut a, Some(&acc.v), t.a.as_ref());
    let mut b = None;
    accumulate(&mut b, acc.b.as_ref(), Some(&t.v));
    accumulate(&mut b, Some(&acc.v), t.b.as_ref());
    let mut ab = None;
    accumulate(&mut ab, acc.ab.as_ref(), Some(&t.v));
    accumulate(&mut ab, acc.a.as_ref(), t.b.as_ref());
    accumulate(&mut ab, acc.b.as_ref(), t.a.as_ref());
    accumulate(&mut ab, Some(&acc.v), t.ab.as_ref());
    TermJet {
        v: hadamard(&acc.v, &t.v),
        a,
        b,
        ab,
    }
}

impl<T: KernelScalar> CompiledKernel<T> {
    /// `∂K/∂dir` of one term, or `None` when `dir` is a parameter of another term.
    fn first_dir<M: crate::math::KernelMath>(
        &self,
        x1: MatRef<'_, T>,
        x2: MatRef<'_, T>,
        dir: Dir,
        local_theta: Option<usize>,
    ) -> Result<Option<Mat<T>>, GprError> {
        let (rows, cols) = (x1.nrows(), x2.nrows());
        Ok(match dir {
            Dir::X2(dim) => {
                let mut out = Mat::zeros(rows, cols);
                self.grad_wrt_coord_dim::<M>(x1, x2, out.as_mut(), dim)?;
                Some(out)
            }
            // k is symmetric: ∂k(x1, x2)/∂x1 is the transpose of ∂k(x2, x1)/∂x2.
            Dir::X1(dim) => {
                let mut swapped = Mat::zeros(cols, rows);
                self.grad_wrt_coord_dim::<M>(x2, x1, swapped.as_mut(), dim)?;
                Some(Mat::from_fn(rows, cols, |row, col| swapped[(col, row)]))
            }
            Dir::Theta(_) => match local_theta {
                Some(local) => {
                    let mut out = Mat::zeros(rows, cols);
                    let scratch = Mat::zeros(rows, cols);
                    let mut scratch = scratch;
                    self.grad_cross_points::<M>(x1, x2, out.as_mut(), local, scratch.as_mut())?;
                    Some(out)
                }
                None => None,
            },
        })
    }

    /// `∂²K/∂a ∂b` of one term for the directions the coordinate Hessians use:
    /// `(X2, X2)`, `(X1, X2)`, `(Theta, X2)`.
    fn second_dirs<M: crate::math::KernelMath>(
        &self,
        x1: MatRef<'_, T>,
        x2: MatRef<'_, T>,
        (a, b): (Dir, Dir),
        local_theta: Option<usize>,
    ) -> Result<Option<Mat<T>>, GprError> {
        let (rows, cols) = (x1.nrows(), x2.nrows());
        let mut out = Mat::zeros(rows, cols);
        match (a, b) {
            (Dir::X2(da), Dir::X2(db)) => {
                let mut scratch = Mat::zeros(rows, cols);
                self.hess_wrt_coord_dims::<M>(x1, x2, out.as_mut(), da, db, scratch.as_mut())?;
            }
            (Dir::X1(da), Dir::X2(db)) => {
                let mut scratch = Mat::zeros(rows, cols);
                self.hess_wrt_coord_mixed::<M>(x1, x2, out.as_mut(), da, db, scratch.as_mut())?;
            }
            (Dir::Theta(_), Dir::X2(db)) => match local_theta {
                Some(local) => self.hess_theta_coord_dim::<M>(x1, x2, out.as_mut(), local, db)?,
                None => return Ok(None),
            },
            _ => return Err(GprError::CoordGradientUnsupported),
        }
        Ok(Some(out))
    }

    /// The product rule over `terms` for `∂K/∂a` (`b = None`) or `∂²K/∂a ∂b`
    /// of a Product tree, written into `out`.
    pub(super) fn product_coord<M: crate::math::KernelMath>(
        terms: &[CompiledKernel<T>],
        x1: MatRef<'_, T>,
        x2: MatRef<'_, T>,
        mut out: MatMut<'_, T>,
        (a, b): (Dir, Option<Dir>),
    ) -> Result<(), GprError> {
        let (rows, cols) = (x1.nrows(), x2.nrows());
        let owner_of = |dir: Dir| match dir {
            Dir::Theta(p) => super::grad::term_index_for_param(terms, p).map(Some),
            _ => Ok(None),
        };
        let theta_owner = match (
            owner_of(a)?,
            match b {
                Some(dir) => owner_of(dir)?,
                None => None,
            },
        ) {
            (Some(owner), _) | (None, Some(owner)) => Some(owner),
            (None, None) => None,
        };
        let mut folded: Option<TermJet<T>> = None;
        for (index, term) in terms.iter().enumerate() {
            let local = theta_owner.and_then(|(owner, local)| (owner == index).then_some(local));
            let mut v = Mat::zeros(rows, cols);
            let scratch = Mat::zeros(rows, cols);
            let mut scratch = scratch;
            term.apply_cross_points::<M>(x1, x2, v.as_mut(), scratch.as_mut())?;
            let first = |dir: Dir| term.first_dir::<M>(x1, x2, dir, local);
            let jet = TermJet {
                v,
                a: first(a)?,
                b: match b {
                    Some(dir) => first(dir)?,
                    None => None,
                },
                ab: match b {
                    Some(dir) => term.second_dirs::<M>(x1, x2, (a, dir), local)?,
                    None => None,
                },
            };
            folded = Some(match folded {
                Some(acc) => combine(&acc, &jet),
                None => jet,
            });
        }
        let folded = folded.ok_or(GprError::CoordGradientUnsupported)?;
        let result = if b.is_some() { folded.ab } else { folded.a };
        match result {
            Some(mat) => write_rect(out, |row, col| Ok(mat[(row, col)])),
            None => {
                fill(out.as_mut(), T::from_f64(0.0));
                Ok(())
            }
        }
    }
}
