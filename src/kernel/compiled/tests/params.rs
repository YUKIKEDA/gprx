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
