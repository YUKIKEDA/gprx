//! Leave-one-out prediction of the collapsed VFE posterior.
//!
//! With `A = L_mm⁻¹ K_mn`, `B = σn² I + A Aᵀ`, and `w = B⁻¹ A y`, leaving
//! out point `i` at fixed `θ` and `Z` subtracts `a_i a_iᵀ` from `B` and
//! `a_i y_i` from `A y`. Sherman–Morrison with `h = a_iᵀ B⁻¹ a_i` and
//! `g = a_iᵀ w` gives the prediction at `x_i` without the point:
//! mean `(g − h y_i) / (1 − h)` and latent variance
//! `k(x_i, x_i) − ‖a_i‖² + σn² h / (1 − h)`. One triangular solve
//! `L_B⁻¹ A` makes the whole pass `O(n m²)`.

use faer::{Mat, MatRef};

use crate::data::pack_points;
use crate::error::GprError;
use crate::kernel::{KernelScalar, ScalarOps};
use crate::linalg::solve_lower;
use crate::policy::with_kernel_exp;
use crate::precision::{InverseBuffers, ModelPrecision};
use crate::sparse::{KernelScratch, SparseCore};
use crate::{PredictOptions, Prediction, VarianceKind};

use super::assemble_vfe;

/// Leave-one-out mean and variance at every training point, mapped back
/// through the target transform.
///
/// A rounding storage (`f32`) assembles the VFE system again in `f64`, as
/// its prediction factors `K_mm` in `f64`; `f64` storage reads the stored
/// `A`, `L_B`, and `w`.
///
/// # Errors
///
/// Returns [`GprError::NonPositiveDefiniteMatrix`] when `1 − h` is not
/// positive and finite for some point, or the error of the `f64` assembly.
pub(crate) fn vfe_loo<P: ModelPrecision>(
    core: &SparseCore,
    a: MatRef<'_, P::Storage>,
    b_l: MatRef<'_, P::Storage>,
    w: &[P::Storage],
    options: PredictOptions,
) -> Result<Prediction<P::Refine>, GprError> {
    let (n, m, d) = (core.n, core.m, core.d);
    let (a64, b_l64, w64) = if <P::Storage as ScalarOps>::ROUNDS_FROM_F64 {
        let state = with_kernel_exp!(core.math, M => assemble_vfe::<M, f64>(
            &core.kernel,
            core.jitter,
            core.likelihood,
            &core.x_train,
            n,
            d,
            &core.y_train,
            &core.z_train,
            m,
            &mut KernelScratch::new(),
            &mut KernelScratch::new(),
        ))?;
        (state.a, state.b_l, state.w)
    } else {
        (
            promote(a),
            promote(b_l),
            w.iter().map(|value| value.to_f64()).collect(),
        )
    };
    let mut s = a64.clone();
    solve_lower(b_l64.as_ref(), s.as_mut());
    let x = pack_points(&core.x_train, n, d);
    let mut k_diag = vec![0.0; n];
    core.kernel
        .compile()
        .fill_diag_points(x.as_ref(), &mut k_diag)?;
    let noise = core.likelihood.noise_variance();
    let zero = P::Refine::from_f64(0.0);
    let mut out = Prediction {
        mean: vec![zero; n],
        variance: vec![zero; n],
        variance_kind: options.variance_kind,
    };
    for i in 0..n {
        let (mut h, mut g, mut a_norm) = (0.0, 0.0, 0.0);
        for k in 0..m {
            let a_ki = a64[(k, i)];
            h += s[(k, i)] * s[(k, i)];
            g += a_ki * w64[k];
            a_norm += a_ki * a_ki;
        }
        let keep = 1.0 - h;
        if !keep.is_finite() || keep <= 0.0 {
            return Err(GprError::NonPositiveDefiniteMatrix);
        }
        let mean = (g - h * core.y_train[i]) / keep;
        let latent = (k_diag[i] - a_norm + noise * h / keep).max(0.0);
        out.mean[i] = P::Refine::from_f64(mean);
        out.variance[i] = P::Refine::from_f64(match options.variance_kind {
            VarianceKind::Latent => latent,
            VarianceKind::Observation => latent + noise,
        });
    }
    core.inverse_prediction_in_place::<P>(&mut out, &mut InverseBuffers::default())?;
    Ok(out)
}

fn promote<T: KernelScalar>(src: MatRef<'_, T>) -> Mat<f64> {
    Mat::from_fn(src.nrows(), src.ncols(), |row, col| {
        src[(row, col)].to_f64()
    })
}
