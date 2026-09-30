"""GPyTorch sparse cells on a real dataset, float64, CPU.

``sgpr``: ``ExactGP`` with ``InducingPointKernel`` (fixed ``Z``) and the exact
marginal log likelihood, fitted with L-BFGS-B (``native`` runs the tutorial
Adam, lr 0.1, 50 steps; ``matched`` scipy L-BFGS-B). ``svgp``: variational
strategy on fixed ``Z``, ELBO, minibatch Adam with the case's settings.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import gpytorch
import numpy as np
import torch
from gpytorch.constraints import GreaterThan

from common.records import load_case, write_result
from common.rss import peak_rss_bytes
from common.timeline import phase

from .gpytorch_fit import ADAM_LR, ADAM_STEPS, optimize_lbfgsb_or_adam
from .metrics import score
from .timing import warmup_fits

torch.set_default_dtype(torch.float64)


def _base_kernel(case: dict, d: int):
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=d))
    kernel.base_kernel.lengthscale = case["lengthscale_init"]
    kernel.outputscale = 1.0
    # gprx's sparse models cannot fit a signal variance (see gprx fit.rs): fixed at 1 here too.
    kernel.raw_outputscale.requires_grad_(False)
    return kernel


def _likelihood(case: dict):
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=GreaterThan(1e-5))
    likelihood.noise = case["noise_variance_init"]
    return likelihood


class Sgpr(gpytorch.models.ExactGP):
    def __init__(self, x, y, likelihood, z, case):
        super().__init__(x, y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.InducingPointKernel(
            _base_kernel(case, x.shape[1]), inducing_points=z.clone(), likelihood=likelihood
        )
        self.covar_module.inducing_points.requires_grad_(False)

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


class Svgp(gpytorch.models.ApproximateGP):
    def __init__(self, z, case, d):
        dist = gpytorch.variational.CholeskyVariationalDistribution(z.shape[0])
        strategy = gpytorch.variational.VariationalStrategy(
            self, z.clone(), dist, learn_inducing_locations=False
        )
        super().__init__(strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = _base_kernel(case, d)

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def fit_sgpr(case, x, y, z):
    likelihood = _likelihood(case)
    model = Sgpr(x, y, likelihood, z, case)
    info = optimize_lbfgsb_or_adam(case, model, likelihood, x, y)
    return model, likelihood, info


def fit_svgp(case, x, y, z):
    likelihood = _likelihood(case)
    model = Svgp(z, case, x.shape[1])
    model.train()
    likelihood.train()
    params = [p for p in list(model.parameters()) + list(likelihood.parameters()) if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=float(case["adam_lr"]))
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=x.shape[0])
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y),
        batch_size=int(case["adam_batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(0),
    )
    steps = 0
    for _ in range(int(case["adam_epochs"])):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = -mll(model(xb), yb)
            loss.backward()
            optimizer.step()
            steps += 1
    return model, likelihood, {
        "calls": steps,
        "iterations": steps,
        "message": f"Adam lr={case['adam_lr']} batch={case['adam_batch_size']} epochs={case['adam_epochs']}",
    }


def run(case: dict) -> dict:
    phase("load")
    n, d = int(case["n_rows"]), int(case["n_cols"])
    m = int(case["xs_n_rows"])
    x = torch.as_tensor(np.asarray(case["x"]).reshape(d, n).T.copy())
    y = torch.as_tensor(np.asarray(case["y"]))
    xs = torch.as_tensor(np.asarray(case["xs"]).reshape(d, m).T.copy())
    z = torch.as_tensor(np.asarray(case["z"]).reshape(d, int(case["n_inducing"])).T.copy())
    fit = fit_sgpr if case["model"] == "sgpr" else fit_svgp

    for _ in range(warmup_fits(n)):
        phase("warmup")
        fit(case, x, y, z)
    phase("fit")
    t0 = time.perf_counter()
    model, likelihood, info = fit(case, x, y, z)
    fit_s = time.perf_counter() - t0

    nlml = None
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
    row.update(score(mean, var, np.asarray(case["ys"]), case["y_mean"], case["y_std"]))
    return row


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m perf.real.gpytorch_sparse_fit CASE.json", file=sys.stderr)
        return 2
    write_result(run(load_case(Path(sys.argv[1]))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
