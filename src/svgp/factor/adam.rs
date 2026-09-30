//! Mini-batch Adam over kernel `θ`, likelihood `θ`, and `q`.

use super::gradient::svgp_value_and_gradient;
use crate::error::GprError;
use crate::optimizer::{Adam, chain_logit_grad, log_theta_to_z, z_to_log_theta};
use crate::param::Interval;
use crate::rng::small_rng;
use crate::svgp::FittedSvgp;
use rand::RngExt;
use rand::rngs::SmallRng;

pub(super) fn user_to_unconstrained(
    user: &[f64],
    n_theta: usize,
    m: usize,
    intervals: &[Interval],
) -> Result<Vec<f64>, GprError> {
    let mut z = vec![0.0; user.len()];
    let mapped = log_theta_to_z(&user[..n_theta], intervals)?;
    z[..n_theta].copy_from_slice(&mapped);
    z[n_theta..n_theta + m].copy_from_slice(&user[n_theta..n_theta + m]);
    let mut packed = 0;
    for j in 0..m {
        for i in j..m {
            let idx = n_theta + m + packed;
            z[idx] = if i == j { user[idx].ln() } else { user[idx] };
            packed += 1;
        }
    }
    Ok(z)
}

pub(super) fn unconstrained_to_user(
    z: &[f64],
    n_theta: usize,
    m: usize,
    intervals: &[Interval],
) -> Result<Vec<f64>, GprError> {
    let mut user = vec![0.0; z.len()];
    let mapped = z_to_log_theta(&z[..n_theta], intervals)?;
    user[..n_theta].copy_from_slice(&mapped);
    user[n_theta..n_theta + m].copy_from_slice(&z[n_theta..n_theta + m]);
    let mut packed = 0;
    for j in 0..m {
        for i in j..m {
            let idx = n_theta + m + packed;
            user[idx] = if i == j { z[idx].exp() } else { z[idx] };
            packed += 1;
        }
    }
    Ok(user)
}

pub(super) fn user_grad_to_unconstrained(
    user: &[f64],
    z: &[f64],
    intervals: &[Interval],
    g_user: &[f64],
    g_z: &mut [f64],
    n_theta: usize,
    m: usize,
) {
    g_z.copy_from_slice(g_user);
    chain_logit_grad(
        &z[..n_theta],
        intervals,
        &user[..n_theta],
        &mut g_z[..n_theta],
    );
    let mut packed = 0;
    for j in 0..m {
        for i in j..m {
            let idx = n_theta + m + packed;
            if i == j {
                g_z[idx] = g_user[idx] * user[idx];
            }
            packed += 1;
        }
    }
}

pub(super) fn shuffle_indices(idx: &mut [usize], rng: &mut SmallRng) {
    for i in (1..idx.len()).rev() {
        let j = rng.random_range(0..=i);
        idx.swap(i, j);
    }
}

pub(crate) fn run_adam_fit<M: crate::math::KernelMath, P>(
    model: &mut FittedSvgp<P>,
    adam: &Adam,
) -> Result<(), GprError>
where
    P: crate::precision::GpScalar,
{
    let n = model.core.n;
    let m = model.core.m;
    let n_theta = model.core.kernel.num_params() + model.core.likelihood.num_params();
    let p = model.num_params();
    let mut intervals = vec![Interval::DEFAULT_POSITIVE; model.core.theta_len()];
    model.core.theta_intervals(&mut intervals)?;
    let mut user = vec![0.0; p];
    model.get_params(&mut user)?;
    let mut z = user_to_unconstrained(&user, n_theta, m, &intervals)?;
    let mut moment1 = vec![0.0; p];
    let mut moment2 = vec![0.0; p];
    let mut g_user = vec![0.0; p];
    let mut g_z = vec![0.0; p];
    let mut order: Vec<usize> = (0..n).collect();
    let mut rng = small_rng(adam.seed());
    let mut timestep = 0_u64;
    let batch_size = adam.batch_size();
    for _ in 0..adam.epochs() {
        shuffle_indices(&mut order, &mut rng);
        let mut start = 0;
        while start < n {
            let end = start.saturating_add(batch_size).min(n);
            let batch = &order[start..end];
            user = unconstrained_to_user(&z, n_theta, m, &intervals)?;
            model.set_params_light(&user)?;
            let mut scratch = std::mem::take(&mut model.scratch);
            let result = svgp_value_and_gradient::<M, _>(model, &mut g_user, batch, &mut scratch);
            model.scratch = scratch;
            result?;
            user_grad_to_unconstrained(&user, &z, &intervals, &g_user, &mut g_z, n_theta, m);
            adam.step(&mut z, &g_z, &mut moment1, &mut moment2, &mut timestep);
            start = end;
        }
    }
    user = unconstrained_to_user(&z, n_theta, m, &intervals)?;
    model.set_params_light(&user)?;
    // The steps left `A` and `k_diag` stale: one pass over all n rebuilds them.
    model.rebuild_data_terms()
}
