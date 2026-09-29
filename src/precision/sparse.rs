//! Mixed-precision refinement for the sparse models: Sgpr's predict
//! weights and Svgp's predictive mean.
//!
//! The model passes its `f64` reference (re-assembled VFE factors, or `K_mm`
//! and `k_*` evaluated in `f64`) as a closure, so this module does not
//! depend on `sgpr` or `svgp`. The closure runs only when the
//! [`super::ResidualFormula`] or the fallback needs it.

use faer::{Mat, MatRef};

use crate::error::GprError;
use crate::kernel::KernelScalar;
use crate::linalg::{
    forward_substitute, gram_aat_plus_noise_in_scalar, matvec_promoted, solve_llt,
};

use super::{RefineSystem, ResidualFormula, refine};

/// The collapsed VFE system assembled in `f64`: `A = L_mm⁻¹ K(Z, X)` and
/// the weights `w` of `B w = A y`.
pub struct F64Vfe {
    pub(crate) a: Mat<f64>,
    pub(crate) w: Vec<f64>,
}

/// `B w = A y` with `B = A Aᵀ + σn² I` through the stored `f32` factor of `B`.
///
/// [`super::PromoteStorage`] forms `B` from the stored `f32` `A`;
/// [`super::ReevaluateKernel`] from the `f64` `A` of `reference`. The
/// fallback is the `f64` `w` of `reference`.
pub(crate) fn refine_vfe_weights<R: ResidualFormula>(
    a: MatRef<'_, f32>,
    b_l: MatRef<'_, f32>,
    w_storage: &[f32],
    y: &[f64],
    noise: f64,
    reference: &dyn Fn() -> Result<F64Vfe, GprError>,
) -> Result<Vec<f64>, GprError> {
    if w_storage.is_empty() {
        return Ok(Vec::new());
    }
    let (b32, b64, rhs) = if R::READS_STORAGE {
        (
            Some(gram_aat_plus_noise_in_scalar(a, noise)),
            None,
            matvec_promoted(a, y),
        )
    } else {
        let f64_vfe = reference()?;
        (
            None,
            Some(gram_aat_plus_noise_in_scalar(f64_vfe.a.as_ref(), noise)),
            matvec_promoted(f64_vfe.a.as_ref(), y),
        )
    };
    let system = WeightSystem {
        b32,
        b64,
        rhs,
        b_l,
        reference,
    };
    let start = w_storage.iter().map(|value| f64::from(*value)).collect();
    refine(&system, start)
}

struct WeightSystem<'a> {
    b32: Option<Mat<f32>>,
    b64: Option<Mat<f64>>,
    rhs: Vec<f64>,
    b_l: MatRef<'a, f32>,
    reference: &'a dyn Fn() -> Result<F64Vfe, GprError>,
}

impl RefineSystem for WeightSystem<'_> {
    fn rhs(&self) -> &[f64] {
        &self.rhs
    }

    fn residual(&self, w: &[f64], r: &mut [f64]) -> Result<f64, GprError> {
        Ok(residual_inf(
            self.b32.as_ref().map(Mat::as_ref),
            self.b64.as_ref().map(Mat::as_ref),
            w,
            &self.rhs,
            r,
        ))
    }

    fn correct(&self, r: &[f64], w: &mut [f64]) {
        let mut delta = Mat::<f32>::from_fn(r.len(), 1, |i, _| r[i] as f32);
        solve_llt(self.b_l, delta.as_mut());
        for (i, slot) in w.iter_mut().enumerate() {
            *slot += f64::from(delta[(i, 0)]);
        }
    }

    fn fallback(&self) -> Result<Vec<f64>, GprError> {
        Ok((self.reference)()?.w)
    }
}

fn residual_inf(
    b32: Option<MatRef<'_, f32>>,
    b64: Option<MatRef<'_, f64>>,
    w: &[f64],
    rhs: &[f64],
    r: &mut [f64],
) -> f64 {
    let m = rhs.len();
    let mut b_inf = 0.0f64;
    for i in 0..m {
        let mut row = 0.0;
        let mut sum = 0.0;
        for j in 0..m {
            let bij = if let Some(b) = b32 {
                f64::from(b[(i, j)])
            } else if let Some(b) = b64 {
                b[(i, j)]
            } else {
                0.0
            };
            row += bij.abs();
            sum += bij * w[j];
        }
        b_inf = b_inf.max(row);
        r[i] = rhs[i] - sum;
    }
    b_inf
}

/// Svgp mean: `v = L⁻¹ k_*` refined in `f64`, dotted with `q_mean`.
///
/// [`super::PromoteStorage`] refines against the stored `f32` `L` and `k_*`
/// promoted to `f64`; [`super::ReevaluateKernel`] against the `f64` `L` and
/// `k_*` of `reference`.
pub(crate) fn refine_svgp_mean<R: ResidualFormula>(
    k_mm_l: MatRef<'_, f32>,
    solved: &[f32],
    rhs: &[f32],
    q_mean: &[f64],
    reference: &dyn Fn() -> Result<(Mat<f64>, Vec<f64>), GprError>,
) -> Result<f64, GprError> {
    let m = solved.len();
    let (l64, rhs64) = if R::READS_STORAGE {
        let mut l64 = Mat::<f64>::zeros(m, m);
        let mut rhs64 = vec![0.0; m];
        for i in 0..m {
            rhs64[i] = rhs[i].to_f64();
            for j in 0..=i {
                l64[(i, j)] = k_mm_l[(i, j)].to_f64();
            }
        }
        (l64, rhs64)
    } else {
        reference()?
    };
    let system = TriangularSystem {
        l: k_mm_l,
        l64: l64.as_ref(),
        rhs64: &rhs64,
    };
    let start = solved.iter().map(|value| f64::from(*value)).collect();
    let v = refine(&system, start)?;
    let mut sum = 0.0;
    for (weight, mean) in v.iter().zip(q_mean.iter()) {
        sum += *weight * *mean;
    }
    Ok(sum)
}

/// `L v = b` with `L` lower-triangular: refined through the stored `f32` `L`
/// against an `f64` reference `(L₆₄, b₆₄)`, which is also the fallback.
struct TriangularSystem<'a> {
    l: MatRef<'a, f32>,
    l64: MatRef<'a, f64>,
    rhs64: &'a [f64],
}

impl RefineSystem for TriangularSystem<'_> {
    fn rhs(&self) -> &[f64] {
        self.rhs64
    }

    fn residual(&self, v: &[f64], r: &mut [f64]) -> Result<f64, GprError> {
        let mut l_inf = 0.0f64;
        for (i, (ri, &bi)) in r.iter_mut().zip(self.rhs64).enumerate() {
            let mut row = 0.0;
            let mut sum = 0.0;
            for (j, &vj) in v.iter().enumerate().take(i + 1) {
                let lij = f64::from(self.l[(i, j)]);
                row += lij.abs();
                sum += lij * vj;
            }
            l_inf = l_inf.max(row);
            *ri = bi - sum;
        }
        Ok(l_inf)
    }

    fn correct(&self, r: &[f64], v: &mut [f64]) {
        let r32: Vec<f32> = r.iter().map(|value| *value as f32).collect();
        let delta = forward_substitute(self.l, &r32);
        for (slot, step) in v.iter_mut().zip(delta) {
            *slot += f64::from(step);
        }
    }

    fn fallback(&self) -> Result<Vec<f64>, GprError> {
        Ok(forward_substitute(self.l64, self.rhs64))
    }
}

/// Sgpr mean: storage `k_*` column dotted with storage weights.
pub(super) fn storage_dot<T: KernelScalar>(column: &[T], weights: &[T]) -> T {
    let mut sum = T::from_f64(0.0);
    for (kernel, weight) in column.iter().zip(weights.iter()) {
        sum += *kernel * *weight;
    }
    sum
}

/// Sgpr mean: `f32` `k_*` column promoted and dotted with `f64` weights.
pub(super) fn promoted_dot(column: &[f32], weights: &[f64]) -> f64 {
    let mut sum = 0.0;
    for (kernel, weight) in column.iter().zip(weights.iter()) {
        sum += kernel.to_f64() * *weight;
    }
    sum
}

/// Svgp mean: storage `L⁻¹ k_*` dotted with the variational mean.
pub(super) fn storage_q_dot<T: KernelScalar>(solved: &[T], q_mean: &[f64]) -> T {
    let mut sum = T::from_f64(0.0);
    for (weight, mean) in solved.iter().zip(q_mean.iter()) {
        sum += *weight * T::from_f64(*mean);
    }
    sum
}
