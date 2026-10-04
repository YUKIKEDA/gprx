"""GPyTorch exact cell: ARD ``Scale(RBF)`` + Gaussian likelihood, float64, CPU.

GPyTorch has no default optimizer, so ``native`` is the one its exact-GP
tutorial uses (Adam, lr 0.1, 50 steps). ``matched`` hands the same objective
(the marginal log likelihood times n, autograd gradient) to scipy
``L-BFGS-B`` with the case's iteration cap and gradient tolerance.
"""

from __future__ import annotations

import contextlib
import sys
import time
from pathlib import Path

import gpytorch
import linear_operator
import numpy as np
import torch
from common.records import load_case, write_result
from common.rss import peak_rss_bytes
from common.timeline import phase
from gpytorch.constraints import GreaterThan
from scipy.optimize import minimize

from . import maunaloa_kernels
from .metrics import case_metrics
from .timing import warmup_fits

# Parameters are set from Python floats before `.double()`: keep them in float64.
torch.set_default_dtype(torch.float64)

ADAM_LR = 0.1
ADAM_STEPS = 50


@contextlib.contextmanager
def exact_cholesky(n: int):
    """Factor every covariance of order ``n`` with Cholesky.

    The default ``max_cholesky_size`` is 800. Above that, the marginal
    likelihood, the solves, and the predictive cache switch to conjugate
    gradients and Lanczos. sklearn, GPy, and gprx factor ``K`` with Cholesky
    at every ``n`` this harness fits, so GPyTorch stays on that factorization.
    """
    cap = max(int(n), 1)
    with (
        gpytorch.settings.max_cholesky_size(cap),
        gpytorch.settings.fast_pred_var(False),
        gpytorch.settings.fast_computations(
            covar_root_decomposition=False,
            log_prob=False,
            solves=False,
        ),
    ):
        yield


class ExactModel(gpytorch.models.ExactGP):
    def __init__(self, x, y, likelihood, d, lengthscale, outputscale):
        super().__init__(x, y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=d)
        )
        self.covar_module.base_kernel.lengthscale = lengthscale
        self.covar_module.outputscale = outputscale

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def build(case: dict, x: torch.Tensor, y: torch.Tensor):
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=GreaterThan(1e-5))
    if case.get("kernel") == "mauna_loa":
        likelihood.noise = case["theta_init"][-1]
        model = ExactModel(x, y, likelihood, x.shape[1], 1.0, 1.0)
        model.covar_module = maunaloa_kernels.gpytorch_kernel(case["theta_init"])
        return model.double(), likelihood.double()
    likelihood.noise = case["noise_variance_init"]
    model = ExactModel(
        x, y, likelihood, x.shape[1], case["lengthscale_init"], case["signal_variance_init"]
    )
    return model.double(), likelihood.double()


def optimize_lbfgsb_or_adam(case: dict, model, likelihood, x, y) -> dict:
    """Runs the case's optimizer; returns the call count and stop reason."""
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    n = x.shape[0]
    model.train()
    likelihood.train()
    params = [p for p in model.parameters() if p.requires_grad]
    calls = 0
    protocol = case["protocol"]
    if protocol == "fixed":
        return {"calls": 0, "iterations": 0, "message": "fixed"}
    if protocol == "native":
        optimizer = torch.optim.Adam(params, lr=ADAM_LR)
        for _ in range(ADAM_STEPS):
            optimizer.zero_grad()
            loss = -mll(model(x), y)
            loss.backward()
            optimizer.step()
            calls += 1
        return {"calls": calls, "iterations": ADAM_STEPS, "message": f"Adam lr={ADAM_LR} steps={ADAM_STEPS}"}

    shapes = [p.shape for p in params]
    sizes = [p.numel() for p in params]

    def assign(flat: np.ndarray) -> None:
        offset = 0
        for p, size, shape in zip(params, sizes, shapes):
            p.data = torch.as_tensor(flat[offset : offset + size].reshape(shape), dtype=torch.float64)
            offset += size

    def objective(flat: np.ndarray):
        nonlocal calls
        calls += 1
        assign(flat)
        for p in params:
            p.grad = None
        try:
            loss = -mll(model(x), y) * n
            loss.backward()
        except (
            gpytorch.utils.errors.NotPSDError,
            linear_operator.utils.errors.NanError,
            torch.linalg.LinAlgError,
        ):
            # a trial θ whose K is not positive definite: a huge value and no
            # slope, as gprx does (SgprObjective returns 1e300)
            return 1.0e300, np.zeros_like(flat)
        grad = np.concatenate([p.grad.reshape(-1).numpy() for p in params])
        return float(loss.item()), grad

    x0 = np.concatenate([p.detach().reshape(-1).numpy() for p in params])
    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": int(case["max_iterations"]),
            "gtol": float(case["gtol"]),
            "ftol": 0.0,
            "maxcor": 10,
        },
    )
    assign(result.x)
    return {"calls": calls, "iterations": int(result.nit), "message": str(result.message)}


def run(case: dict) -> dict:
    phase("load")
    n, d = int(case["n_rows"]), int(case["n_cols"])
    m = int(case["xs_n_rows"])
    x = torch.as_tensor(np.asarray(case["x"]).reshape(d, n).T.copy(), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(case["y"]), dtype=torch.float64)
    xs = torch.as_tensor(np.asarray(case["xs"]).reshape(d, m).T.copy(), dtype=torch.float64)

    # Train and test can each be the matrix that is factored.
    with exact_cholesky(max(n, m)):
        for _ in range(warmup_fits(n)):
            phase("warmup")
            model, likelihood = build(case, x, y)
            optimize_lbfgsb_or_adam(case, model, likelihood, x, y)
        phase("fit")
        model, likelihood = build(case, x, y)
        t0 = time.perf_counter()
        info = optimize_lbfgsb_or_adam(case, model, likelihood, x, y)
        fit_s = time.perf_counter() - t0

        model.train()
        likelihood.train()
        with torch.no_grad():
            nlml = float(-gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)(model(x), y).item() * n)
        model.eval()
        likelihood.eval()
        phase("predict")
        t0 = time.perf_counter()
        with torch.no_grad():
            pred = likelihood(model(xs))
            mean, var = pred.mean.numpy(), pred.variance.numpy()
        predict_s = time.perf_counter() - t0

    row = {
        "lib": "gpytorch",
        "name": case["name"],
        "status": "ok",
        "protocol": case["protocol"],
        "fit_s": fit_s,
        "predict_s": predict_s,
        "joint_evals": info["calls"],
        "value_evals": 0,
        "iterations": info["iterations"],
        "nlml": nlml,
        "peak_rss_bytes": peak_rss_bytes(),
        "note": info["message"],
    }
    row.update(case_metrics(case, mean, var))
    return row


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m perf.real.gpytorch_fit CASE.json", file=sys.stderr)
        return 2
    write_result(run(load_case(Path(sys.argv[1]))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
