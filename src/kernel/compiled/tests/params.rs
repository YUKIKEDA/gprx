//! Parameter vector round trips and argument checks.

use super::*;

#[test]
fn get_set_params_roundtrip_and_atomic() {
    let mut compiled = (rbf(1.0) + rbf(2.0)).compile();
    let mut params = [0.0; 2];
    compiled.get_params(&mut params).expect("len 2");
    assert_close(params[0], 1.0_f64.ln(), TOL);
    assert_close(params[1], 2.0_f64.ln(), TOL);
    params[0] = 0.5_f64.ln();
    compiled.set_params(&params).expect("valid");
    compiled.get_params(&mut params).expect("len 2");
    assert_close(params[0], 0.5_f64.ln(), TOL);
    let before = compiled.clone();
    assert!(compiled.set_params(&[0.0, f64::INFINITY]).is_err());
    assert_eq!(compiled, before);
}

#[test]
fn rejects_bad_index_and_scratch_shape() {
    let compiled = (rbf(1.0) + rbf(2.0)).compile();
    let dist = sq_dist_1d(&[0.0, 1.0]);
    let mut dk = fill(2, 0.0);
    let mut scratch = fill(2, 0.0);
    assert!(matches!(
        compiled.grad::<crate::math::Accurate>(
            dist.as_ref(),
            dk.as_mut(),
            2,
            Triangle::Lower,
            scratch.as_mut()
        ),
        Err(crate::error::GprError::IndexOutOfRange { .. })
    ));
    let mut small = fill(1, 0.0);
    assert!(matches!(
        compiled.apply::<crate::math::Accurate>(
            dist.as_ref(),
            dk.as_mut(),
            Triangle::Full,
            small.as_mut()
        ),
        Err(crate::error::GprError::WorkspaceTooSmall)
    ));
}

#[test]
fn set_params_in_place_restores_written_leaves_on_error() {
    let spec = rbf(1.0) + rbf(2.0);
    let mut compiled = spec.clone().compile();
    let mut spec_in_place = spec.clone();
    let prev = [1.0_f64.ln(), 2.0_f64.ln()];
    let bad = [0.5_f64.ln(), f64::INFINITY];
    let before = compiled.clone();
    assert!(compiled.set_params_in_place(&bad, &prev).is_err());
    assert_eq!(compiled, before);
    assert!(spec_in_place.set_params_in_place(&bad, &prev).is_err());
    assert_eq!(spec_in_place, spec);
    let good = [0.5_f64.ln(), 3.0_f64.ln()];
    compiled.set_params_in_place(&good, &prev).expect("valid");
    let mut got = [0.0; 2];
    compiled.get_params(&mut got).expect("len 2");
    assert_close(got[0], good[0], TOL);
    assert_close(got[1], good[1], TOL);
}

#[test]
fn ard_set_params_is_all_or_nothing() {
    let mut scales = crate::kernel::ArdLengthscales::new(&[1.0, 2.0, 3.0]).expect("valid");
    let before = scales.clone();
    assert!(scales.set_params(&[0.0, f64::NAN, 0.0]).is_err());
    assert_eq!(scales, before);
    scales
        .set_params(&[0.5_f64.ln(), 0.0, 1.5_f64.ln()])
        .expect("valid");
    assert_close(scales.lengthscale(0).expect("dim 0"), 0.5, TOL);
    assert_close(scales.lengthscale(2).expect("dim 2"), 1.5, TOL);
    assert_close(scales.inv_ell_sq()[1], 1.0, TOL);
}
