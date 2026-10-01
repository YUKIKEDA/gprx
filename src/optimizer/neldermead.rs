//! Nelder–Mead via argmin. Uses [`Objective::value`] only.

use std::cell::RefCell;
use std::num::NonZeroU32;

use argmin::core::{Executor, State};
use argmin::solver::neldermead::NelderMead as ArgminNelderMead;

use crate::error::GprError;
use crate::objective::Objective;

use super::logit::{
    ValueCache, ValueProblem, best_value, consider_value_run, log_theta_to_z, map_argmin_error,
    sample_log_uniform_z,
};
use super::{OptResult, Optimizer, Restarts};

/// Nelder–Mead via argmin. Uses [`Objective::value`] only.
#[derive(Clone, Debug, PartialEq)]
pub struct NelderMead {
    max_iterations: u64,
    tolerance: f64,
    restarts: Option<Restarts>,
}

impl Default for NelderMead {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: f64::EPSILON.sqrt(),
            restarts: None,
        }
    }
}

impl NelderMead {
    /// Builds Nelder–Mead with 100 iterations and simplex SD tolerance `sqrt(ε)`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::num::NonZeroU32;
    /// use gprx::NelderMead;
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let _nm = NelderMead::new()
    ///     .with_max_iterations(50)
    ///     .with_tolerance(1e-8)?
    ///     .with_restarts(NonZeroU32::MIN, 0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the iteration cap passed to argmin (default 100).
    pub fn with_max_iterations(mut self, max_iterations: u64) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Sets the simplex standard-deviation tolerance (default `sqrt(ε)`).
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidConfig`] if `tolerance` is not finite
    /// or is negative.
    pub fn with_tolerance(mut self, tolerance: f64) -> Result<Self, GprError> {
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(GprError::InvalidConfig {
                reason: "Nelder-Mead simplex tolerance must be finite and >= 0".to_owned(),
            });
        }
        self.tolerance = tolerance;
        Ok(self)
    }

    /// Adds `n` extra log-uniform starts (`n ≥ 1`) and keeps the lowest NLML.
    ///
    /// The first start is the model `θ`. Failed extra starts are discarded.
    /// The default trainer has no restarts and no seed.
    pub fn with_restarts(mut self, n: NonZeroU32, seed: u64) -> Self {
        self.restarts = Some(Restarts { n, seed });
        self
    }

    #[cfg(test)]
    pub(crate) fn minimize_unconstrained<P: Objective>(
        &self,
        objective: &mut P,
        init: &[f64],
    ) -> Result<OptResult, GprError> {
        run_neldermead(self, objective, init)
    }
}

impl<P: Objective> Optimizer<P> for NelderMead {
    fn minimize(&self, objective: &mut P, init: &[f64]) -> Result<OptResult, GprError> {
        super::minimize_with_restarts(
            objective,
            init,
            self.restarts,
            log_theta_to_z,
            sample_log_uniform_z,
            |objective, intervals, z, _restart, best| {
                consider_value_run(objective, intervals, z, best, |mapped, z| {
                    run_neldermead(self, mapped, z)
                })
            },
        )
    }
}

fn run_neldermead<P: Objective>(
    nm: &NelderMead,
    objective: &mut P,
    init: &[f64],
) -> Result<OptResult, GprError> {
    let n = objective.num_params();
    if init.len() != n {
        return Err(GprError::LengthMismatch {
            reason: format!("expected {n} parameters, got {}", init.len()),
        });
    }
    let problem = ValueProblem {
        inner: RefCell::new(ValueCache::new(objective)),
    };
    let simplex = initial_simplex(init);
    let solver = ArgminNelderMead::new(simplex)
        .with_sd_tolerance(nm.tolerance)
        .map_err(map_argmin_error)?;
    let (params, value, iterations) =
        {
            let result = Executor::new(problem, solver)
                .configure(|state| state.param(init.to_vec()).max_iters(nm.max_iterations))
                .ctrlc(false)
                .run()
                .map_err(map_argmin_error)?;
            let state = result.state();
            let params = state.get_best_param().cloned().ok_or_else(|| {
                GprError::OptimizationNotConverged {
                    iterations: state.get_iter() as usize,
                }
            })?;
            (params, state.get_best_cost(), state.get_iter())
        };
    let value = best_value(value, iterations, || objective.value(&params))?;
    Ok(OptResult {
        params,
        value,
        iterations,
    })
}

fn initial_simplex(z: &[f64]) -> Vec<Vec<f64>> {
    let n = z.len();
    let mut verts = Vec::with_capacity(n + 1);
    verts.push(z.to_vec());
    for i in 0..n {
        let mut v = z.to_vec();
        if v[i].abs() > 0.0 {
            v[i] *= 1.05;
        } else {
            v[i] = 2.5e-4;
        }
        verts.push(v);
    }
    verts
}

#[cfg(test)]
mod tests {
    use super::NelderMead;
    use crate::error::GprError;
    use crate::gpr::Gpr;
    use crate::kernel::{KernelSpec, RbfKernel};
    use crate::likelihood::GaussianLikelihood;
    use crate::objective::Objective;
    use crate::optimizer::{Fixed, Optimizer};

    const TOL: f64 = 1e-6;

    use crate::test_check::assert_close;

    struct ValueBowl {
        evals: usize,
    }

    impl Objective for ValueBowl {
        fn num_params(&self) -> usize {
            2
        }

        fn value(&mut self, params: &[f64]) -> Result<f64, GprError> {
            if params.len() != 2 {
                return Err(GprError::ShapeMismatch {
                    reason: "value bowl is 2-D".to_owned(),
                });
            }
            self.evals += 1;
            Ok(0.5 * (params[0] * params[0] + params[1] * params[1]))
        }
    }

    #[test]
    fn neldermead_minimizes_value_only_bowl() {
        let mut obj = ValueBowl { evals: 0 };
        let result = NelderMead::new()
            .with_max_iterations(200)
            .minimize_unconstrained(&mut obj, &[1.0, -0.5])
            .expect("bowl");
        assert!(
            result.params[0].abs() <= 1e-3 && result.params[1].abs() <= 1e-3,
            "params={:?}",
            result.params
        );
        assert!(result.value < 1e-6, "value={}", result.value);
        assert!(obj.evals > 0);
    }

    #[test]
    fn neldermead_lowers_gpr_nlml() {
        let kernel = KernelSpec::from(RbfKernel::new(1.0).expect("valid"));
        let likelihood = GaussianLikelihood::new(0.1).expect("valid");
        let mut gpr = Gpr::new(kernel, likelihood)
            .with_optimizer(Fixed)
            .factor(&[0.0, 1.0], 2, 1, &[0.5, -0.25])
            .unwrap_or_else(|(_, e)| panic!("{e}"));
        let mut init = [0.0; 2];
        gpr.get_params(&mut init).expect("len 2");
        init[0] = 2.0_f64.ln();
        let start;
        let result;
        {
            let mut obj = gpr.objective();
            start = obj.value(&init).expect("start");
            result = NelderMead::new()
                .with_max_iterations(80)
                .minimize(&mut obj, &init)
                .expect("nm");
        }
        assert!(
            result.value <= start + 1e-9,
            "start={start}, best={}",
            result.value
        );
        // The trainer, not the solver, puts the model at the result (#313):
        // the value it reports is the objective's at those parameters.
        let at_result = gpr.objective().value(&result.params).expect("result");
        assert_close(at_result, result.value, TOL);
    }
}
