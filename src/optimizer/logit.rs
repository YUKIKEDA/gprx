//! Logit map from user-unit [`Interval`] to unconstrained argmin coordinates.

use std::cell::RefCell;

use argmin::core::{CostFunction, Error as ArgminError, Gradient};

use rand::rngs::SmallRng;

use crate::error::GprError;
use crate::objective::{Differentiable, Objective, TwiceDifferentiable};
use crate::param::Interval;
use crate::rng::open_unit;

use super::OptResult;

pub(super) struct LogitMapped<'a, P> {
    pub(super) inner: &'a mut P,
    pub(super) intervals: &'a [Interval],
    pub(super) log_scratch: Vec<f64>,
}

impl<P: Objective> Objective for LogitMapped<'_, P> {
    fn num_params(&self) -> usize {
        self.inner.num_params()
    }

    fn value(&mut self, params: &[f64]) -> Result<f64, GprError> {
        z_to_log_theta_into(params, self.intervals, &mut self.log_scratch)?;
        self.inner.value(&self.log_scratch)
    }
}

impl<P: Differentiable> Differentiable for LogitMapped<'_, P> {
    fn gradient_into(&mut self, params: &[f64], out: &mut [f64]) -> Result<(), GprError> {
        self.value_and_gradient_into(params, out).map(|_| ())
    }

    fn value_and_gradient_into(
        &mut self,
        params: &[f64],
        out: &mut [f64],
    ) -> Result<f64, GprError> {
        z_to_log_theta_into(params, self.intervals, &mut self.log_scratch)?;
        let value = self.inner.value_and_gradient_into(&self.log_scratch, out)?;
        chain_logit_grad(params, self.intervals, &self.log_scratch, out);
        Ok(value)
    }
}

impl<P: TwiceDifferentiable> TwiceDifferentiable for LogitMapped<'_, P> {
    fn hessian_into(&mut self, params: &[f64], out: &mut [f64]) -> Result<(), GprError> {
        z_to_log_theta_into(params, self.intervals, &mut self.log_scratch)?;
        let n = params.len();
        if out.len() != n * n {
            return Err(GprError::InvalidHyperparameter {
                reason: format!("expected {} Hessian entries, got {}", n * n, out.len()),
            });
        }
        let mut grad = vec![0.0; n];
        self.inner
            .value_and_gradient_into(&self.log_scratch, &mut grad)?;
        self.inner.hessian_into(&self.log_scratch, out)?;
        chain_logit_hess(params, self.intervals, &self.log_scratch, &grad, out);
        Ok(())
    }
}

pub(super) fn consider_value_run<P, F>(
    objective: &mut P,
    intervals: &[Interval],
    init_z: &[f64],
    best: &mut Option<OptResult>,
    mut run: F,
) -> Result<(), GprError>
where
    P: Objective,
    F: FnMut(&mut LogitMapped<'_, P>, &[f64]) -> Result<OptResult, GprError>,
{
    let mut mapped = LogitMapped {
        inner: objective,
        intervals,
        log_scratch: vec![0.0; init_z.len()],
    };
    let run = run(&mut mapped, init_z)?;
    let log_theta = z_to_log_theta(&run.params, intervals)?;
    let value = objective.value(&log_theta)?;
    keep_better(
        best,
        OptResult {
            params: log_theta,
            value,
            iterations: run.iterations,
        },
    );
    Ok(())
}

pub(super) fn keep_better(best: &mut Option<OptResult>, candidate: OptResult) {
    match best {
        None => *best = Some(candidate),
        Some(current) if candidate.value < current.value => *best = Some(candidate),
        Some(_) => {}
    }
}

pub(super) fn log_theta_to_z(
    log_theta: &[f64],
    intervals: &[Interval],
) -> Result<Vec<f64>, GprError> {
    let mut z = vec![0.0; log_theta.len()];
    for (i, &theta) in log_theta.iter().enumerate() {
        let x = theta.exp();
        z[i] = user_to_z(x, intervals[i])?;
    }
    Ok(z)
}

pub(super) fn z_to_log_theta(z: &[f64], intervals: &[Interval]) -> Result<Vec<f64>, GprError> {
    let mut log_theta = vec![0.0; z.len()];
    z_to_log_theta_into(z, intervals, &mut log_theta)?;
    Ok(log_theta)
}

fn z_to_log_theta_into(z: &[f64], intervals: &[Interval], out: &mut [f64]) -> Result<(), GprError> {
    if z.len() != intervals.len() || out.len() != z.len() {
        return Err(GprError::InvalidHyperparameter {
            reason: "logit map length mismatch".to_owned(),
        });
    }
    for i in 0..z.len() {
        let x = z_to_user(z[i], intervals[i]);
        out[i] = x.ln();
    }
    Ok(())
}

fn chain_logit_grad(z: &[f64], intervals: &[Interval], log_theta: &[f64], grad: &mut [f64]) {
    for i in 0..z.len() {
        grad[i] *= dlog_dz(z[i], intervals[i], log_theta[i]);
    }
}

fn chain_logit_hess(
    z: &[f64],
    intervals: &[Interval],
    log_theta: &[f64],
    grad_theta: &[f64],
    hess: &mut [f64],
) {
    let n = z.len();
    let mut jac = vec![0.0; n];
    for i in 0..n {
        jac[i] = dlog_dz(z[i], intervals[i], log_theta[i]);
    }
    for i in 0..n {
        for j in 0..n {
            hess[i * n + j] *= jac[i] * jac[j];
        }
        hess[i * n + i] += grad_theta[i] * d2log_dz2(z[i], intervals[i], log_theta[i]);
    }
}

fn dlog_dz(z: f64, interval: Interval, log_theta: f64) -> f64 {
    let scale = logit_scale(interval);
    let t = z / scale;
    let s = sigmoid(t);
    let ds = s * (1.0 - s);
    let dlog_dt = if interval.lo() > 0.0 {
        (interval.hi().ln() - interval.lo().ln()) * ds
    } else {
        let x = log_theta.exp();
        interval.width() * ds / x
    };
    dlog_dt / scale
}

fn d2log_dz2(z: f64, interval: Interval, log_theta: f64) -> f64 {
    let scale = logit_scale(interval);
    let t = z / scale;
    let s = sigmoid(t);
    let ds = s * (1.0 - s);
    if interval.lo() > 0.0 {
        4.0 * ds * (1.0 - 2.0 * s) / scale
    } else {
        let x = log_theta.exp();
        let dnum_dz = 4.0 * (1.0 - 2.0 * s) * ds / scale;
        let dx_dz = 4.0 * ds;
        let num = 4.0 * ds;
        (dnum_dz * x - num * dx_dz) / (x * x)
    }
}

fn user_to_z(x: f64, interval: Interval) -> Result<f64, GprError> {
    if !interval.contains(x) {
        return Err(GprError::from(crate::param::IntervalError::OutOfRange {
            value: x,
            lo: interval.lo(),
            hi: interval.hi(),
        }));
    }
    let u = unit_from_user(x, interval);
    Ok(logit(u.clamp(f64::EPSILON, 1.0 - f64::EPSILON)) * logit_scale(interval))
}

fn z_to_user(z: f64, interval: Interval) -> f64 {
    let t = z / logit_scale(interval);
    let u = sigmoid(t);
    if interval.lo() > 0.0 {
        let ln_lo = interval.lo().ln();
        let ln_hi = interval.hi().ln();
        (ln_lo + u * (ln_hi - ln_lo)).exp()
    } else {
        interval.lo() + u * interval.width()
    }
}

fn logit_scale(interval: Interval) -> f64 {
    let span = if interval.lo() > 0.0 {
        interval.hi().ln() - interval.lo().ln()
    } else {
        interval.width()
    };
    span / 4.0
}

fn unit_from_user(x: f64, interval: Interval) -> f64 {
    if interval.lo() > 0.0 {
        let ln_lo = interval.lo().ln();
        let ln_hi = interval.hi().ln();
        (x.ln() - ln_lo) / (ln_hi - ln_lo)
    } else {
        (x - interval.lo()) / interval.width()
    }
}

fn logit(u: f64) -> f64 {
    (u / (1.0 - u)).ln()
}

fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        let e = (-z).exp();
        1.0 / (1.0 + e)
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

pub(super) fn sample_log_uniform_z(
    intervals: &[Interval],
    rng: &mut SmallRng,
) -> Result<Vec<f64>, GprError> {
    let mut z = vec![0.0; intervals.len()];
    for (slot, interval) in z.iter_mut().zip(intervals.iter().copied()) {
        let x = log_uniform_open(rng, interval);
        *slot = user_to_z(x, interval)?;
    }
    Ok(z)
}

fn log_uniform_open(rng: &mut SmallRng, interval: Interval) -> f64 {
    let u = open_unit(rng);
    if interval.lo() > 0.0 {
        let ln_lo = interval.lo().ln();
        let ln_hi = interval.hi().ln();
        (ln_lo + u * (ln_hi - ln_lo)).exp()
    } else {
        interval.lo() + u * interval.width()
    }
}

/// Cost returned to argmin when a trial point is non-finite or rejected (for
/// example outside an open [`Interval`]). More–Thuente can then backtrack
/// instead of aborting the whole solve.
const BARRIER_COST: f64 = 1.0e300;

pub(super) struct EvalCache<'a, P: ?Sized> {
    objective: &'a mut P,
    params: Vec<f64>,
    value: Option<f64>,
    grad: Vec<f64>,
}

impl<'a, P: Differentiable + ?Sized> EvalCache<'a, P> {
    pub(super) fn new(objective: &'a mut P, n: usize) -> Self {
        Self {
            objective,
            params: Vec::new(),
            value: None,
            grad: vec![0.0; n],
        }
    }

    fn eval(&mut self, param: &[f64]) -> Result<f64, GprError> {
        if let Some(value) = self.value {
            if same_params(&self.params, param) {
                return Ok(value);
            }
        }
        if self.grad.len() != param.len() {
            self.grad.resize(param.len(), 0.0);
        }
        match self
            .objective
            .value_and_gradient_into(param, &mut self.grad)
        {
            Ok(value) if value.is_finite() && self.grad.iter().all(|g| g.is_finite()) => {
                self.params.clear();
                self.params.extend_from_slice(param);
                self.value = Some(value);
                Ok(value)
            }
            Ok(_) | Err(_) => {
                self.params.clear();
                self.params.extend_from_slice(param);
                self.grad.fill(0.0);
                self.value = Some(BARRIER_COST);
                Ok(BARRIER_COST)
            }
        }
    }
}

pub(super) struct CachedProblem<'a, P: ?Sized> {
    pub(super) inner: RefCell<EvalCache<'a, P>>,
}

impl<P: Differentiable + ?Sized> CostFunction for CachedProblem<'_, P> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, ArgminError> {
        self.inner
            .borrow_mut()
            .eval(param)
            .map_err(ArgminError::from)
    }
}

impl<P: Differentiable + ?Sized> Gradient for CachedProblem<'_, P> {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, ArgminError> {
        let mut inner = self.inner.borrow_mut();
        inner.eval(param).map_err(ArgminError::from)?;
        Ok(inner.grad.clone())
    }
}

pub(super) struct ValueCache<'a, P: ?Sized> {
    objective: &'a mut P,
    params: Vec<f64>,
    value: Option<f64>,
}

impl<'a, P: Objective + ?Sized> ValueCache<'a, P> {
    pub(super) fn new(objective: &'a mut P) -> Self {
        Self {
            objective,
            params: Vec::new(),
            value: None,
        }
    }

    fn eval(&mut self, param: &[f64]) -> Result<f64, GprError> {
        if let Some(value) = self.value {
            if same_params(&self.params, param) {
                return Ok(value);
            }
        }
        match self.objective.value(param) {
            Ok(value) if value.is_finite() => {
                self.params.clear();
                self.params.extend_from_slice(param);
                self.value = Some(value);
                Ok(value)
            }
            Ok(_) | Err(_) => {
                self.params.clear();
                self.params.extend_from_slice(param);
                self.value = Some(BARRIER_COST);
                Ok(BARRIER_COST)
            }
        }
    }
}

pub(super) struct ValueProblem<'a, P: ?Sized> {
    pub(super) inner: RefCell<ValueCache<'a, P>>,
}

impl<P: Objective + ?Sized> CostFunction for ValueProblem<'_, P> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, ArgminError> {
        self.inner
            .borrow_mut()
            .eval(param)
            .map_err(ArgminError::from)
    }
}

fn same_params(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

pub(super) fn map_argmin_error(err: ArgminError) -> GprError {
    if let Some(gpr) = err.downcast_ref::<GprError>() {
        return gpr.clone();
    }
    GprError::InvalidHyperparameter {
        reason: err.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CachedProblem, CostFunction, EvalCache, Gradient, logit, sigmoid, user_to_z, z_to_user,
    };
    use crate::error::GprError;
    use crate::gpr::Gpr;
    use crate::kernel::{KernelSpec, RbfKernel};
    use crate::likelihood::GaussianLikelihood;
    use crate::objective::{Differentiable, Objective, TwiceDifferentiable};
    use crate::optimizer::Fixed;
    use crate::param::Interval;
    use std::cell::RefCell;

    const TOL: f64 = 1e-6;

    fn assert_close(actual: f64, expected: f64) {
        let scale = expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= TOL * scale,
            "actual={actual}, expected={expected}"
        );
    }

    struct Quadratic {
        joint_evals: usize,
    }

    impl Objective for Quadratic {
        fn num_params(&self) -> usize {
            2
        }

        fn value(&mut self, params: &[f64]) -> Result<f64, GprError> {
            let mut dummy = [0.0; 2];
            self.value_and_gradient_into(params, &mut dummy)
        }
    }

    impl Differentiable for Quadratic {
        fn gradient_into(&mut self, params: &[f64], out: &mut [f64]) -> Result<(), GprError> {
            self.value_and_gradient_into(params, out).map(|_| ())
        }

        fn value_and_gradient_into(
            &mut self,
            params: &[f64],
            out: &mut [f64],
        ) -> Result<f64, GprError> {
            if params.len() != 2 || out.len() != 2 {
                return Err(GprError::InvalidHyperparameter {
                    reason: "quadratic is 2-D".to_owned(),
                });
            }
            self.joint_evals += 1;
            out[0] = params[0];
            out[1] = params[1];
            Ok(0.5 * (params[0] * params[0] + params[1] * params[1]))
        }
    }

    #[test]
    fn shared_point_uses_one_joint_eval() {
        let mut obj = Quadratic { joint_evals: 0 };
        let param = vec![0.3, -0.2];
        {
            let problem = CachedProblem {
                inner: RefCell::new(EvalCache::new(&mut obj, 2)),
            };
            let cost = problem.cost(&param).expect("cost");
            let grad = problem.gradient(&param).expect("grad");
            assert_close(cost, 0.5 * (0.3 * 0.3 + 0.2 * 0.2));
            assert_close(grad[0], 0.3);
            assert_close(grad[1], -0.2);
        }
        assert_eq!(obj.joint_evals, 1);
        {
            let problem = CachedProblem {
                inner: RefCell::new(EvalCache::new(&mut obj, 2)),
            };
            let other = vec![-1.0, 0.5];
            let _ = problem.cost(&param).expect("cost");
            let _ = problem.gradient(&param).expect("grad");
            let _ = problem.cost(&other).expect("other");
            let _ = problem.gradient(&other).expect("other grad");
        }
        assert_eq!(obj.joint_evals, 3);
    }

    #[test]
    fn logit_mapped_hess_matches_finite_difference() {
        let kernel = KernelSpec::from(RbfKernel::new(1.25).expect("valid"));
        let likelihood = GaussianLikelihood::new(0.16).expect("valid");
        let mut gpr = Gpr::new(kernel, likelihood)
            .with_optimizer(Fixed)
            .factor(&[0.0, 0.8, 1.7], 3, 1, &[0.4, -0.2, 0.9])
            .unwrap_or_else(|(_, e)| panic!("{e}"));
        let mut log_theta = [0.0; 2];
        gpr.get_params(&mut log_theta).expect("len 2");
        let intervals = [Interval::DEFAULT_POSITIVE, Interval::DEFAULT_POSITIVE];
        let z = super::log_theta_to_z(&log_theta, &intervals).expect("z");
        let mut obj = gpr.objective();
        let mut mapped = super::LogitMapped {
            inner: &mut obj,
            intervals: &intervals,
            log_scratch: vec![0.0; 2],
        };
        let mut hess = [0.0; 4];
        mapped.hessian_into(&z, &mut hess).expect("hess");
        let h = 1e-6;
        let mut g_plus = [0.0; 2];
        let mut g_minus = [0.0; 2];
        for j in 0..2 {
            let mut plus = z.clone();
            let mut minus = z.clone();
            plus[j] += h;
            minus[j] -= h;
            mapped
                .value_and_gradient_into(&plus, &mut g_plus)
                .expect("plus");
            mapped
                .value_and_gradient_into(&minus, &mut g_minus)
                .expect("minus");
            for i in 0..2 {
                let fd = (g_plus[i] - g_minus[i]) / (2.0 * h);
                assert!(
                    (hess[i * 2 + j] - fd).abs() <= 2e-4 * fd.abs().max(1.0),
                    "H[{i},{j}] analytic={} fd={}",
                    hess[i * 2 + j],
                    fd
                );
            }
        }
    }

    #[test]
    fn logit_mapped_grad_matches_finite_difference() {
        let kernel = KernelSpec::from(RbfKernel::new(1.0).expect("valid"));
        let likelihood = GaussianLikelihood::new(0.1).expect("valid");
        let mut gpr = Gpr::new(kernel, likelihood)
            .with_optimizer(Fixed)
            .factor(&[0.0, 1.0], 2, 1, &[0.5, -0.25])
            .unwrap_or_else(|(_, e)| panic!("{e}"));
        let mut log_theta = [0.0; 2];
        gpr.get_params(&mut log_theta).expect("len 2");
        let intervals = [Interval::DEFAULT_POSITIVE, Interval::DEFAULT_POSITIVE];
        let z = super::log_theta_to_z(&log_theta, &intervals).expect("z");
        let mut obj = gpr.objective();
        let mut mapped = super::LogitMapped {
            inner: &mut obj,
            intervals: &intervals,
            log_scratch: vec![0.0; 2],
        };
        let mut analytic = [0.0; 2];
        let value = mapped
            .value_and_gradient_into(&z, &mut analytic)
            .expect("grad");
        let h = 1e-6;
        for i in 0..2 {
            let mut plus = z.clone();
            let mut minus = z.clone();
            plus[i] += h;
            minus[i] -= h;
            let vp = mapped.value(&plus).expect("plus");
            let vm = mapped.value(&minus).expect("minus");
            let fd = (vp - vm) / (2.0 * h);
            assert!(
                (analytic[i] - fd).abs() <= 1e-4 * analytic[i].abs().max(1.0),
                "i={i} analytic={} fd={} value={value}",
                analytic[i],
                fd
            );
        }
    }

    #[test]
    fn logit_roundtrip_stays_inside_interval() {
        let interval = Interval::DEFAULT_POSITIVE;
        let x = 2.5;
        let z = user_to_z(x, interval).expect("inside");
        let back = z_to_user(z, interval);
        assert_close(back, x);
        assert!(interval.contains(back));
        assert_close(sigmoid(logit(0.25)), 0.25);
    }
}
