"""GPyTorch SGPR / SVGP cell. CPU, float64. cargo test must not run this."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(max(1, os.cpu_count() or 1))

from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import InducingPointKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ZeroMean
from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO
from gpytorch.models import ApproximateGP, ExactGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

from common.problems import unpack_column_major as unpack_rows
from common.records import load_case, write_result
from common.rss import peak_rss_bytes
from common.timing import median, min_max, timed_reps, warmup_count

DEVICE = torch.device("cpu")


class SgprModel(ExactGP):
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: GaussianLikelihood,
        z: torch.Tensor,
        ard: bool,
        lengthscales: list[float],
    ) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ZeroMean()
        base = RBFKernel(ard_num_dims=train_x.size(-1) if ard else None)
        scales = torch.tensor(lengthscales, dtype=torch.float64).view(1, -1)
        base.initialize(lengthscale=scales)
        self.covar_module = InducingPointKernel(base, inducing_points=z, likelihood=likelihood)

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))


class SvgpModel(ApproximateGP):
    def __init__(self, z: torch.Tensor, ard: bool, lengthscales: list[float]) -> None:
        m = z.size(0)
        variational_distribution = CholeskyVariationalDistribution(m, mean_init_std=0.0)
        variational_strategy = VariationalStrategy(
            self,
            z.clone(),
            variational_distribution,
            learn_inducing_locations=False,
        )
        super().__init__(variational_strategy)
        self.mean_module = ZeroMean()
        kernel = RBFKernel(ard_num_dims=z.size(-1) if ard else None)
        scales = torch.tensor(lengthscales, dtype=torch.float64).view(1, -1)
        kernel.initialize(lengthscale=scales)
        self.covar_module = kernel

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))


def tensors(case: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.tensor(
        unpack_rows(case["x"], case["n_rows"], case["n_cols"]),
        dtype=torch.float64,
        device=DEVICE,
    )
    y = torch.tensor(case["y"], dtype=torch.float64, device=DEVICE)
    z = torch.tensor(
        unpack_rows(case["z"], case["n_inducing"], case["n_cols"]),
        dtype=torch.float64,
        device=DEVICE,
    )
    xs = torch.tensor(
        unpack_rows(case["xs"], case["xs_n_rows"], case["xs_n_cols"]),
        dtype=torch.float64,
        device=DEVICE,
    )
    return x, y, z, xs


def make_sgpr(case: dict, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> tuple[SgprModel, GaussianLikelihood]:
    likelihood = GaussianLikelihood()
    likelihood.initialize(noise=float(case["noise_variance_init"]))
    model = SgprModel(x, y, likelihood, z, bool(case["ard"]), case["lengthscales_init"])
    model.double()
    likelihood.double()
    model.to(DEVICE)
    likelihood.to(DEVICE)
    return model, likelihood


def make_svgp(case: dict, z: torch.Tensor) -> tuple[SvgpModel, GaussianLikelihood]:
    likelihood = GaussianLikelihood()
    likelihood.initialize(noise=float(case["noise_variance_init"]))
    model = SvgpModel(z, bool(case["ard"]), case["lengthscales_init"])
    model.double()
    likelihood.double()
    model.to(DEVICE)
    likelihood.to(DEVICE)
    return model, likelihood


def sgpr_factor(case: dict, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> tuple[SgprModel, GaussianLikelihood]:
    model, likelihood = make_sgpr(case, x, y, z)
    model.train()
    likelihood.train()
    mll = ExactMarginalLogLikelihood(likelihood, model)
    output = model(*model.train_inputs)
    _ = -mll(output, model.train_targets)
    return model, likelihood


def svgp_factor(
    case: dict, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> tuple[SvgpModel, GaussianLikelihood]:
    model, likelihood = make_svgp(case, z)
    model.train()
    likelihood.train()
    mll = VariationalELBO(likelihood, model, num_data=int(case["n_rows"]), beta=1.0)
    output = model(x)
    _ = -mll(output, y)
    return model, likelihood


def sgpr_joint(model: SgprModel, likelihood: GaussianLikelihood) -> None:
    model.train()
    likelihood.train()
    model.zero_grad()
    likelihood.zero_grad()
    mll = ExactMarginalLogLikelihood(likelihood, model)
    output = model(*model.train_inputs)
    loss = -mll(output, model.train_targets)
    loss.backward()


def svgp_joint(
    model: SvgpModel,
    likelihood: GaussianLikelihood,
    x: torch.Tensor,
    y: torch.Tensor,
    n: int,
) -> None:
    model.train()
    likelihood.train()
    model.zero_grad()
    likelihood.zero_grad()
    mll = VariationalELBO(likelihood, model, num_data=n, beta=1.0)
    output = model(x)
    loss = -mll(output, y)
    loss.backward()


def predict_obs(model: ExactGP | ApproximateGP, likelihood: GaussianLikelihood, xs: torch.Tensor) -> None:
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        dist = likelihood(model(xs))
        _ = dist.mean, dist.variance


def run(case: dict) -> dict:
    x, y, z, xs = tensors(case)
    n_evals = int(case["joint_evals"])
    warmup = warmup_count()
    reps = timed_reps(int(case["n_rows"]))
    model_name = case["model"]
    factor_samples: list[float] = []
    model = None
    likelihood = None
    for i in range(warmup + reps):
        model = None
        likelihood = None
        t0 = time.perf_counter()
        if model_name == "sgpr":
            model, likelihood = sgpr_factor(case, x, y, z)
        elif model_name == "svgp":
            model, likelihood = svgp_factor(case, x, y, z)
        else:
            raise ValueError(f"unknown sparse model {model_name}")
        dt = time.perf_counter() - t0
        if i >= warmup:
            factor_samples.append(dt)
    assert model is not None
    assert likelihood is not None

    eval_samples: list[float] = []
    for i in range(warmup + reps):
        t0 = time.perf_counter()
        if model_name == "sgpr":
            sgpr_joint(model, likelihood)
        else:
            svgp_joint(model, likelihood, x, y, int(case["n_rows"]))
        dt = time.perf_counter() - t0
        if i >= warmup:
            eval_samples.append(dt)
    eval_scale = [s * n_evals for s in eval_samples]

    predict_samples: list[float] = []
    for i in range(warmup + reps):
        t0 = time.perf_counter()
        predict_obs(model, likelihood, xs)
        dt = time.perf_counter() - t0
        if i >= warmup:
            predict_samples.append(dt)

    factor_lo, factor_hi = min_max(factor_samples)
    eval_lo, eval_hi = min_max(eval_scale)
    predict_lo, predict_hi = min_max(predict_samples)
    return {
        "lib": "gpytorch",
        "name": case["name"],
        "status": "ok",
        "factor_s": median(factor_samples),
        "factor_min_s": factor_lo,
        "factor_max_s": factor_hi,
        "eval_s": median(eval_scale),
        "eval_min_s": eval_lo,
        "eval_max_s": eval_hi,
        "predict_s": median(predict_samples),
        "predict_min_s": predict_lo,
        "predict_max_s": predict_hi,
        "joint_evals": n_evals,
        "peak_rss_bytes": peak_rss_bytes(),
        "warmup": warmup,
        "reps": reps,
        "note": (
            f"GPyTorch {model_name} CPU f64 + {n_evals}× median of one joint; "
            f"discard {warmup} then {reps} timed"
        ),
    }


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: gpytorch_sparse.py CASE.json", file=sys.stderr)
        return 2
    write_result(run(load_case(Path(sys.argv[1]))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
