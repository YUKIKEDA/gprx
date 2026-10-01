//! Fast Simulated Annealing (Cauchy / Metropolis) as a homemade [`Optimizer`].

use std::num::NonZeroU32;

use crate::rng::SeededRng;

use crate::error::GprError;
use crate::objective::{HasBounds, Objective};
use crate::param::Interval;
use crate::rng::{open_unit, seeded_rng};

use super::logit::keep_better;
use super::{OptResult, Optimizer, Restarts};

/// How a proposed coordinate is folded back into an open parameter interval.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum BoundaryPolicy {
    /// Projects onto the open interval by clamping just inside the endpoints.
    #[default]
    Clamp,
    /// Wraps through the opposite side of the open interval.
    Periodic,
}

/// Fast Simulated Annealing with a Cauchy generating distribution.
///
/// Walks the same vector [`Optimizer::minimize`] receives (log-`θ` for
/// [`crate::Gpr`]). Positive user-unit [`Interval`]s map to
/// `(ln lo, ln hi)`. Component-wise Cauchy proposals use the Szu–Hartley
/// inverse CDF; worse points follow the Metropolis rule. Temperature follows
/// Ingber’s dimension-normalized exponential schedule. The first
/// evaluation and each restart use [`Objective::value`]. Each coordinate
/// step uses [`Objective::value_at_changes`]. A proposal the model cannot
/// evaluate (not positive definite, not finite) is rejected like an infinite
/// energy.
///
/// References: Szu & Hartley (1987), “Fast simulated annealing”; Ingber
/// (1989), “Very fast simulated re-annealing”.
///
/// # Examples
///
/// ```rust
/// use gprx::kernel::{KernelSpec, RbfKernel};
/// use gprx::{FastSimulatedAnnealing, GaussianLikelihood, Gpr};
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let kernel = KernelSpec::from(RbfKernel::new(1.0)?);
/// let likelihood = GaussianLikelihood::new(0.1)?;
/// let fsa = FastSimulatedAnnealing::new().with_seed(7);
/// let gpr = Gpr::new(kernel, likelihood).with_optimizer(fsa);
/// let fitted = gpr.fit(&[0.0, 1.0], 2, 1, &[0.5, -0.25]).map_err(|(_, e)| e)?;
/// let _nlml = fitted.neg_log_marginal_likelihood()?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct FastSimulatedAnnealing {
    max_iterations: u64,
    restarts: Option<Restarts>,
    initial_temperature: f64,
    cooling_rate: f64,
    seed: u64,
    boundary: BoundaryPolicy,
}

impl Default for FastSimulatedAnnealing {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            restarts: None,
            initial_temperature: 1.0,
            cooling_rate: 3.0,
            seed: 0,
            boundary: BoundaryPolicy::Clamp,
        }
    }
}

impl FastSimulatedAnnealing {
    /// Builds FSA with 100 iterations, temperature 1, cooling rate 3, and seed 0.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::num::NonZeroU32;
    /// use gprx::{BoundaryPolicy, FastSimulatedAnnealing};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let _fsa = FastSimulatedAnnealing::new()
    ///     .with_max_iterations(50)
    ///     .with_initial_temperature(2.0)?
    ///     .with_cooling_rate(3.0)?
    ///     .with_seed(1)
    ///     .with_boundary(BoundaryPolicy::Periodic)
    ///     .with_restarts(NonZeroU32::MIN, 0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the iteration cap (default 100). One iteration updates every coordinate.
    pub fn with_max_iterations(mut self, max_iterations: u64) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Adds `n` extra starts (`n ≥ 1`) and keeps the lowest objective value.
    ///
    /// Extra starts are drawn log-uniform in each positive interval (uniform
    /// in the parameter space of `init` otherwise). The first start is `init`.
    /// Failed extra starts are discarded.
    pub fn with_restarts(mut self, n: NonZeroU32, seed: u64) -> Self {
        self.restarts = Some(Restarts { n, seed });
        self
    }

    /// Sets the initial temperature `T0` used by the Cauchy step and Metropolis rule.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidConfig`] if `temperature` is not
    /// finite or is not strictly positive.
    pub fn with_initial_temperature(mut self, temperature: f64) -> Result<Self, GprError> {
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(GprError::InvalidConfig {
                reason: "FSA initial temperature must be finite and > 0".to_owned(),
            });
        }
        self.initial_temperature = temperature;
        Ok(self)
    }

    /// Sets the Ingber cooling coefficient `c` in `T(k) = T0 exp(-c k^{1/D})`.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidConfig`] if `cooling_rate` is not
    /// finite or is not strictly positive.
    pub fn with_cooling_rate(mut self, cooling_rate: f64) -> Result<Self, GprError> {
        if !cooling_rate.is_finite() || cooling_rate <= 0.0 {
            return Err(GprError::InvalidConfig {
                reason: "FSA cooling rate must be finite and > 0".to_owned(),
            });
        }
        self.cooling_rate = cooling_rate;
        Ok(self)
    }

    /// Sets the seed passed to gprx's seeded generator (Xoshiro256++, the same on every platform).
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Sets how proposals that leave the open interval are folded back.
    pub fn with_boundary(mut self, boundary: BoundaryPolicy) -> Self {
        self.boundary = boundary;
        self
    }
}

impl<P: Objective + HasBounds> Optimizer<P> for FastSimulatedAnnealing {
    const USES_CHANGE_INDICES: bool = true;

    fn minimize(&self, objective: &mut P, init: &[f64]) -> Result<OptResult, GprError> {
        let n = objective.num_params();
        if init.len() != n {
            return Err(GprError::LengthMismatch {
                reason: format!("expected {n} parameters, got {}", init.len()),
            });
        }
        if n == 0 {
            return Err(GprError::LengthMismatch {
                reason: "FSA requires at least one parameter".to_owned(),
            });
        }
        let mut intervals = vec![Interval::DEFAULT_POSITIVE; n];
        objective.fill_intervals(&mut intervals)?;
        let mut best: Option<OptResult> = None;
        keep_better(&mut best, anneal(self, objective, init, &intervals)?);
        if let Some(restarts) = self.restarts {
            let mut rng = seeded_rng(restarts.seed);
            for _ in 0..restarts.n.get() {
                let start = sample_in_param_space(&intervals, &mut rng, self.boundary);
                let _ = anneal(self, objective, &start, &intervals)
                    .map(|candidate| keep_better(&mut best, candidate));
            }
        }
        best.ok_or(GprError::OptimizationNotConverged { iterations: 0 })
    }
}

fn anneal<P: Objective>(
    fsa: &FastSimulatedAnnealing,
    objective: &mut P,
    init: &[f64],
    intervals: &[Interval],
) -> Result<OptResult, GprError> {
    let n = init.len();
    let bounds: Vec<(f64, f64)> = intervals.iter().copied().map(param_bounds).collect();
    let mut current = init.to_vec();
    for (slot, &(lo, hi)) in current.iter_mut().zip(bounds.iter()) {
        *slot = apply_boundary(*slot, lo, hi, fsa.boundary);
    }
    let mut current_energy = objective.value(&current)?;
    if !current_energy.is_finite() {
        return Err(GprError::InvalidConfig {
            reason: "FSA initial objective value must be finite".to_owned(),
        });
    }
    let mut best = current.clone();
    let mut best_energy = current_energy;
    let mut proposed = current.clone();
    let mut rng = seeded_rng(fsa.seed);
    let dim = n as f64;
    // Coordinate of the last rejected proposal. The objective last evaluated
    // that proposal, so the next step also lists it as changed.
    let mut reverted: Option<usize> = None;
    for iteration in 0..fsa.max_iterations {
        let temperature = fsa.temperature(iteration, dim);
        for i in 0..n {
            proposed.copy_from_slice(&current);
            let (lo, hi) = bounds[i];
            let step = cauchy_step(&mut rng, temperature) * (hi - lo);
            proposed[i] = apply_boundary(current[i] + step, lo, hi, fsa.boundary);
            let changes = match reverted {
                Some(r) if r != i => [r, i],
                _ => [i, i],
            };
            let changed = if changes[0] == changes[1] {
                &changes[..1]
            } else {
                &changes[..]
            };
            // A proposal the model cannot evaluate (not positive definite, not
            // finite) is rejected like an infinite energy; any other error is
            // the caller's.
            let proposed_energy = match objective.value_at_changes(&proposed, changed) {
                Ok(value) => value,
                Err(
                    GprError::CholeskyFailed { .. }
                    | GprError::NonFiniteKernelValue
                    | GprError::NonPositiveDefiniteMatrix,
                ) => f64::INFINITY,
                Err(err) => return Err(err),
            };
            reverted = Some(i);
            if !proposed_energy.is_finite() {
                continue;
            }
            if metropolis_accept(&mut rng, proposed_energy - current_energy, temperature) {
                reverted = None;
                current.copy_from_slice(&proposed);
                current_energy = proposed_energy;
                if current_energy < best_energy {
                    best.copy_from_slice(&current);
                    best_energy = current_energy;
                }
            }
        }
    }
    Ok(OptResult {
        params: best,
        value: best_energy,
        iterations: fsa.max_iterations,
    })
}

impl FastSimulatedAnnealing {
    fn temperature(&self, iteration: u64, dim: f64) -> f64 {
        let k = iteration as f64;
        (self.initial_temperature * (-self.cooling_rate * k.powf(1.0 / dim)).exp()).max(1e-12)
    }
}

fn param_bounds(interval: Interval) -> (f64, f64) {
    if interval.lo() > 0.0 {
        (interval.lo().ln(), interval.hi().ln())
    } else {
        (interval.lo(), interval.hi())
    }
}

fn apply_boundary(x: f64, lo: f64, hi: f64, policy: BoundaryPolicy) -> f64 {
    let width = hi - lo;
    debug_assert!(width > 0.0);
    let eps = (width * f64::EPSILON).max(f64::MIN_POSITIVE);
    let lo_in = lo + eps;
    let hi_in = hi - eps;
    match policy {
        BoundaryPolicy::Clamp => x.clamp(lo_in, hi_in),
        BoundaryPolicy::Periodic => {
            let mut t = (x - lo) % width;
            if t < 0.0 {
                t += width;
            }
            (lo + t).clamp(lo_in, hi_in)
        }
    }
}

fn cauchy_step(rng: &mut SeededRng, temperature: f64) -> f64 {
    let u = open_unit(rng);
    let sign = if u >= 0.5 { 1.0 } else { -1.0 };
    let t = temperature.max(1e-12);
    sign * t * ((1.0 + 1.0 / t).powf((2.0 * u - 1.0).abs()) - 1.0)
}

fn metropolis_accept(rng: &mut SeededRng, delta: f64, temperature: f64) -> bool {
    if delta <= 0.0 {
        true
    } else {
        let t = temperature.max(1e-12);
        open_unit(rng) < (-delta / t).exp()
    }
}

fn sample_in_param_space(
    intervals: &[Interval],
    rng: &mut SeededRng,
    boundary: BoundaryPolicy,
) -> Vec<f64> {
    intervals
        .iter()
        .copied()
        .map(|interval| {
            let (lo, hi) = param_bounds(interval);
            apply_boundary(lo + open_unit(rng) * (hi - lo), lo, hi, boundary)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{BoundaryPolicy, FastSimulatedAnnealing, apply_boundary, metropolis_accept};
    use crate::error::GprError;
    use crate::gpr::Gpr;
    use crate::kernel::{KernelSpec, RbfKernel};
    use crate::likelihood::GaussianLikelihood;
    use crate::objective::{HasBounds, Objective};
    use crate::optimizer::{Fixed, Optimizer};
    use crate::param::Interval;
    use crate::rng::seeded_rng;

    struct Rosenbrock;

    impl Objective for Rosenbrock {
        fn num_params(&self) -> usize {
            2
        }

        fn value(&mut self, params: &[f64]) -> Result<f64, GprError> {
            if params.len() != 2 {
                return Err(GprError::ShapeMismatch {
                    reason: "Rosenbrock is 2-D".to_owned(),
                });
            }
            let a = 1.0 - params[0];
            let b = params[1] - params[0] * params[0];
            Ok(a * a + 100.0 * b * b)
        }
    }

    impl HasBounds for Rosenbrock {
        fn fill_intervals(&self, out: &mut [Interval]) -> Result<(), GprError> {
            if out.len() != 2 {
                return Err(GprError::ShapeMismatch {
                    reason: "Rosenbrock is 2-D".to_owned(),
                });
            }
            let interval = Interval::new(-5.0, 5.0).expect("finite");
            out[0] = interval;
            out[1] = interval;
            Ok(())
        }
    }

    struct Bowl1d {
        target: f64,
    }

    impl Objective for Bowl1d {
        fn num_params(&self) -> usize {
            1
        }

        fn value(&mut self, params: &[f64]) -> Result<f64, GprError> {
            if params.len() != 1 {
                return Err(GprError::ShapeMismatch {
                    reason: "bowl is 1-D".to_owned(),
                });
            }
            let d = params[0] - self.target;
            Ok(d * d)
        }
    }

    impl HasBounds for Bowl1d {
        fn fill_intervals(&self, out: &mut [Interval]) -> Result<(), GprError> {
            if out.len() != 1 {
                return Err(GprError::ShapeMismatch {
                    reason: "bowl is 1-D".to_owned(),
                });
            }
            out[0] = Interval::new(-2.0, 2.0).expect("finite");
            Ok(())
        }
    }

    #[test]
    fn metropolis_accepts_improvement_and_rejects_huge_increase() {
        let mut rng = seeded_rng(1);
        assert!(metropolis_accept(&mut rng, -0.25, 1.0));
        assert!(!metropolis_accept(&mut rng, 1.0e9, 1.0e-12));
    }

    #[test]
    fn clamp_stays_strictly_inside() {
        let y = apply_boundary(5.0, 0.0, 1.0, BoundaryPolicy::Clamp);
        assert!(y > 0.0 && y < 1.0);
        assert!(y > 0.9);
        let z = apply_boundary(-3.0, 0.0, 1.0, BoundaryPolicy::Clamp);
        assert!(z > 0.0 && z < 1.0);
        assert!(z < 0.1);
    }

    #[test]
    fn periodic_wraps_into_open_interval() {
        let y = apply_boundary(1.25, 0.0, 1.0, BoundaryPolicy::Periodic);
        assert!(y > 0.0 && y < 1.0);
        assert!((y - 0.25).abs() <= 1e-12);
        let z = apply_boundary(-0.25, 0.0, 1.0, BoundaryPolicy::Periodic);
        assert!(z > 0.0 && z < 1.0);
        assert!((z - 0.75).abs() <= 1e-12);
    }

    #[test]
    fn rosenbrock_lands_in_loose_box() {
        let mut obj = Rosenbrock;
        let result = FastSimulatedAnnealing::new()
            .with_max_iterations(400)
            .with_initial_temperature(2.0)
            .expect("T")
            .with_seed(1234)
            .minimize(&mut obj, &[0.0, 0.0])
            .expect("fsa");
        assert!(
            result.params[0] > -0.5
                && result.params[0] < 2.5
                && result.params[1] > -0.5
                && result.params[1] < 2.5,
            "params={:?}",
            result.params
        );
    }

    #[test]
    fn gpr_fit_returns_fitted_and_lowers_nlml() {
        let kernel = KernelSpec::from(RbfKernel::new(4.0).expect("valid"));
        let likelihood = GaussianLikelihood::new(1.0).expect("valid");
        let fsa = FastSimulatedAnnealing::new().with_seed(7);
        let gpr = Gpr::new(kernel, likelihood).with_optimizer(fsa);
        let x = [0.0, 1.0];
        let y = [0.5, -0.25];
        let start = gpr
            .clone()
            .with_optimizer(Fixed)
            .factor(&x, 2, 1, &y)
            .unwrap_or_else(|(_, e)| panic!("{e}"))
            .neg_log_marginal_likelihood()
            .expect("start");
        let fitted = gpr.fit(&x, 2, 1, &y).unwrap_or_else(|(_, e)| panic!("{e}"));
        let best = fitted.neg_log_marginal_likelihood().expect("fitted nlml");
        assert!(best < start, "start={start}, best={best}");
    }

    #[test]
    fn one_d_bowl_moves_toward_target() {
        let mut obj = Bowl1d { target: 0.5 };
        let start = obj.value(&[-1.0]).expect("start");
        let result = FastSimulatedAnnealing::new()
            .with_seed(3)
            .minimize(&mut obj, &[-1.0])
            .expect("fsa");
        assert!(result.value < start, "start={start}, best={}", result.value);
    }
    /// `x²` that cannot be evaluated (not positive definite) for `x > 0.5`.
    struct Unevaluable;

    impl Objective for Unevaluable {
        fn num_params(&self) -> usize {
            1
        }

        fn value(&mut self, params: &[f64]) -> Result<f64, GprError> {
            if params[0] > 0.5 {
                return Err(GprError::NonPositiveDefiniteMatrix);
            }
            Ok(params[0] * params[0])
        }
    }

    impl HasBounds for Unevaluable {
        fn fill_intervals(&self, out: &mut [Interval]) -> Result<(), GprError> {
            out[0] = Interval::new(-2.0, 2.0).expect("finite");
            Ok(())
        }
    }

    #[test]
    fn a_proposal_that_cannot_be_evaluated_is_rejected_not_fatal() {
        let result = FastSimulatedAnnealing::new()
            .with_seed(3)
            .minimize(&mut Unevaluable, &[0.4])
            .expect("annealing survives unevaluable proposals");
        assert!(result.params[0] <= 0.5);
        assert!(result.value <= 0.16 + 1e-12);
    }
}
