//! A coordinate tree of Constant and White leaves, which read only the rows
//! of `x`, still refuses coordinates without a column at every entry point
//! ([`CompiledKernel::require_tree_columns`]).

use crate::error::GprError;
use crate::kernel::compiled::CrossViews;
use crate::kernel::compiled::gram::GramInputs;
use crate::kernel::compiled::weighted::{DiagAccum, WeightedWalk};
use crate::kernel::{CompiledKernel, ConstantKernel, KernelSpec, Triangle, WhiteKernel};
use crate::math::Accurate;
use faer::Mat;

fn rows_only() -> CompiledKernel<f64> {
    (KernelSpec::from(ConstantKernel::new(1.5).expect("c"))
        + KernelSpec::from(WhiteKernel::new(0.1).expect("w")))
    .compile()
}

fn empty(result: Result<(), GprError>) {
    assert!(matches!(result, Err(GprError::EmptyInput)), "{result:?}");
}

#[test]
fn every_entry_point_refuses_coordinates_without_a_column() {
    let k = rows_only();
    let n = 3;
    let x = Mat::<f64>::zeros(n, 0);
    let inputs = GramInputs::points(x.as_ref());
    let (mut out, mut scratch) = (Mat::<f64>::zeros(n, n), Mat::<f64>::zeros(n, n));
    let mut nested = Vec::new();
    empty(k.eval_gram::<Accurate>(
        inputs,
        out.as_mut(),
        Triangle::Lower,
        scratch.as_mut(),
        &mut nested,
    ));
    empty(k.grad_gram::<Accurate>(
        inputs,
        out.as_mut(),
        0,
        Triangle::Lower,
        scratch.as_mut(),
        &mut nested,
    ));
    empty(k.hess_gram::<Accurate>(
        inputs,
        out.as_mut(),
        (0, 0),
        Triangle::Lower,
        scratch.as_mut(),
        &mut nested,
    ));
    empty(k.eval_cross_slots::<Accurate>(
        x.as_ref(),
        x.as_ref(),
        None,
        None,
        out.as_mut(),
        scratch.as_mut(),
        &mut nested,
        &mut [],
    ));
    let views = CrossViews::points(x.as_ref(), x.as_ref());
    empty(k.grad_cross_views::<Accurate>(views, out.as_mut(), 0, scratch.as_mut(), &mut nested));
    empty(k.hess_cross_views::<Accurate>(
        views,
        out.as_mut(),
        (0, 0),
        scratch.as_mut(),
        &mut nested,
    ));
    let mut term = Mat::<f64>::zeros(n, n);
    empty(k.eval_gram_keeping::<Accurate>(
        inputs,
        out.as_mut(),
        scratch.as_mut(),
        term.as_mut(),
        &mut nested,
        &mut [],
        0,
    ));
    let weight = Mat::<f64>::identity(n, n);
    let mut grads = vec![0.0; k.num_params()];
    let mut bufs: Vec<Mat<f64>> = (0..k.contraction_buffers())
        .map(|_| Mat::zeros(n, n))
        .collect();
    let mut fold = Vec::new();
    let mut walk = WeightedWalk {
        inputs,
        scratch: scratch.as_mut(),
        nested: &mut nested,
        kept: &[],
        kept_products: 0,
        fold: &mut fold,
    };
    empty(k.weighted_grads::<Accurate>(&mut walk, weight.as_ref(), &mut grads, &mut bufs));
    let mut jobs = Vec::new();
    empty(k.weighted_cross_grads_views::<Accurate>(
        views,
        weight.as_ref(),
        &mut grads,
        &mut bufs,
        scratch.as_mut(),
        &mut nested,
        &mut jobs,
    ));
    empty(k.weighted_diag_sums::<Accurate>(x.as_ref(), &mut grads, &mut DiagAccum::new()));
    empty(k.eval_diag(x.as_ref(), &mut [0.0; 3]));
}
