"""Write GPyTorch SGPR / SVGP goldens (P4-11). cargo test must not run this."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import gpytorch
import numpy as np
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel, MaternKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ZeroMean
from gpytorch.mlls import VariationalELBO
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parent
PERF = ROOT / "perf"
GOLDENS = ROOT / "goldens"

sys.path.insert(0, str(PERF))
from problems import (  # noqa: E402
    ELL_ARD,
    ELL_ISO,
    NOISE_VARIANCE_INIT,
    make_forrester,
    make_sphere,
    pack_column_major,
)

M = 16
KMEANS_SEED = 0
WHITE_VARIANCE = 0.05
FORRESTER_XS = np.array([[0.25], [0.5], [0.75]], dtype=np.float64)
SPHERE_XS = np.array([[0.25, 0.75], [0.25, 0.25], [0.5, 0.5]], dtype=np.float64)


class DiagWhiteKernel(Kernel):
    """Index-diagonal white kernel matching gprx `WhiteKernel`."""

    is_stationary = True
    has_lengthscale = False

    def __init__(self, variance: float) -> None:
        super().__init__()
        self.white_variance = float(variance)

    def forward(self, x1, x2, diag=False, **kwargs):  # noqa: ANN001, ANN003
        if diag:
            return x1.new_full(x1.shape[:-1], self.white_variance)
        rows = x1.size(-2)
        cols = x2.size(-2)
        out = x1.new_zeros(*x1.shape[:-2], rows, cols)
        if x1.data_ptr() == x2.data_ptr() and rows == cols:
            eye = torch.eye(rows, dtype=x1.dtype, device=x1.device)
            out = out + self.white_variance * eye
        return out


class SvgpModel(ApproximateGP):
    def __init__(self, z: torch.Tensor, kernel: Kernel) -> None:
        m = z.size(0)
        variational_distribution = CholeskyVariationalDistribution(m, mean_init_std=0.0)
        variational_strategy = VariationalStrategy(
            self,
            z.clone(),
            variational_distribution,
            learn_inducing_locations=False,
            jitter_val=0.0,
        )
        super().__init__(variational_strategy)
        self.mean_module = ZeroMean()
        self.covar_module = kernel

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))


def unpack_column_major(values: list[float], n: int, d: int) -> np.ndarray:
    packed = np.asarray(values, dtype=np.float64)
    coords = np.empty((n, d), dtype=np.float64)
    for dim in range(d):
        coords[:, dim] = packed[dim * n : (dim + 1) * n]
    return coords


def kmeans_z(coords: np.ndarray, m: int) -> np.ndarray:
    model = KMeans(n_clusters=m, random_state=KMEANS_SEED, n_init=10)
    model.fit(coords)
    return np.asarray(model.cluster_centers_, dtype=np.float64)


def set_lengthscale(kernel: RBFKernel | MaternKernel, lengthscales: list[float]) -> None:
    tensor = torch.tensor(lengthscales, dtype=torch.float64).view(1, -1)
    kernel.initialize(lengthscale=tensor)


def make_kernel(name: str, n_cols: int) -> tuple[Kernel, list[float], float | None]:
    if name == "rbf":
        kernel = RBFKernel()
        set_lengthscale(kernel, [ELL_ISO])
        return kernel, [ELL_ISO], None
    if name == "matern32":
        kernel = MaternKernel(nu=1.5)
        set_lengthscale(kernel, [ELL_ISO])
        return kernel, [ELL_ISO], None
    if name == "rbf_ard":
        scales = [ELL_ARD] * n_cols if n_cols > 1 else [ELL_ISO]
        kernel = RBFKernel(ard_num_dims=n_cols)
        set_lengthscale(kernel, scales)
        return kernel, scales, None
    if name == "rbf_white":
        rbf = RBFKernel()
        set_lengthscale(rbf, [ELL_ISO])
        return rbf + DiagWhiteKernel(WHITE_VARIANCE), [ELL_ISO], WHITE_VARIANCE
    raise ValueError(f"unknown kernel {name}")


def predict_latent_obs(
    model: ApproximateGP,
    likelihood: GaussianLikelihood,
    xs: torch.Tensor,
) -> tuple[list[float], list[float], list[float]]:
    model.eval()
    likelihood.eval()
    with (
        torch.no_grad(),
        gpytorch.settings.sgpr_diagonal_correction(True),
        gpytorch.settings.cholesky_jitter(0.0),
        gpytorch.settings.cholesky_max_tries(1),
        gpytorch.settings.fast_pred_var(False),
        gpytorch.settings.fast_computations(
            covar_root_decomposition=False,
            log_prob=False,
            solves=False,
        ),
    ):
        latent = model(xs)
        observed = likelihood(latent)
        mean = latent.mean.detach().cpu().reshape(-1).tolist()
        latent_var = latent.variance.detach().cpu().reshape(-1).tolist()
        obs_var = observed.variance.detach().cpu().reshape(-1).tolist()
    return mean, obs_var, latent_var


def _chol(mat: torch.Tensor) -> torch.Tensor:
    try:
        return torch.linalg.cholesky(mat)
    except RuntimeError:
        return torch.linalg.cholesky(mat + 1e-8 * torch.eye(mat.size(0), dtype=mat.dtype))


def titsias_sgpr(
    kernel: Kernel,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    z: torch.Tensor,
    xs: torch.Tensor,
    noise: float,
) -> tuple[float, list[float], list[float], list[float]]:
    """Collapsed VFE (Titsias) using GPyTorch kernel values.

    GPyTorch ``InducingPointKernel`` + ExactGP is a few 1e-6 off Titsias, so
    the oracle is GPyTorch ``k`` plus the same assemble as gprx.
    """
    n = train_x.size(0)
    m = z.size(0)
    kernel.eval()
    with torch.no_grad():
        k_mm = kernel(z, z, diag=False)
        if hasattr(k_mm, "to_dense"):
            k_mm = k_mm.to_dense()
        k_mn = kernel(z, train_x, diag=False)
        if hasattr(k_mn, "to_dense"):
            k_mn = k_mn.to_dense()
        k_diag = kernel(train_x, train_x, diag=True)
        if hasattr(k_diag, "to_dense"):
            k_diag = k_diag.to_dense()
        k_sz = kernel(z, xs, diag=False)
        if hasattr(k_sz, "to_dense"):
            k_sz = k_sz.to_dense()
        kss = kernel(xs, xs, diag=True)
        if hasattr(kss, "to_dense"):
            kss = kss.to_dense()
        l_mm = _chol(k_mm)
        a = torch.linalg.solve_triangular(l_mm, k_mn, upper=False)
        a_star = torch.linalg.solve_triangular(l_mm, k_sz, upper=False)
        b = a @ a.T + noise * torch.eye(m, dtype=a.dtype)
        l_b = _chol(b)
        ay = a @ train_y.reshape(-1, 1)
        w = torch.linalg.solve_triangular(
            l_b.T,
            torch.linalg.solve_triangular(l_b, ay, upper=False),
            upper=True,
        ).reshape(-1)
        a_fro = float((a * a).sum().item())
        k_diag_sum = float(k_diag.sum().item())
        y_norm2 = float((train_y * train_y).sum().item())
        ay_dot_w = float((ay.reshape(-1) * w).sum().item())
        quad = (y_norm2 - ay_dot_w) / noise
        logdet_b = float(2.0 * torch.log(torch.diag(l_b)).sum().item())
        log_det = (n - m) * float(np.log(noise)) + logdet_b
        trace = (k_diag_sum - a_fro) / (2.0 * noise)
        nlml = 0.5 * (n * float(np.log(2.0 * np.pi)) + log_det + quad) + trace
        mean = (a_star.T @ w).reshape(-1)
        a_norm = (a_star * a_star).sum(dim=0)
        binv = torch.linalg.solve_triangular(
            l_b.T,
            torch.linalg.solve_triangular(l_b, a_star, upper=False),
            upper=True,
        )
        binv_norm = (a_star * binv).sum(dim=0)
        latent = kss - a_norm + noise * binv_norm
        latent = torch.clamp(latent, min=0.0)
        obs = latent + noise
    return (
        float(nlml),
        mean.cpu().tolist(),
        obs.cpu().tolist(),
        latent.cpu().tolist(),
    )


def sgpr_case(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    z: torch.Tensor,
    xs: torch.Tensor,
    name: str,
    n_cols: int,
) -> dict:
    kernel, lengthscales, white = make_kernel(name, n_cols)
    kernel.double()
    nlml, mean, obs_var, latent_var = titsias_sgpr(
        kernel, train_x, train_y, z, xs, NOISE_VARIANCE_INIT
    )
    return {
        "name": name,
        "lengthscales": lengthscales,
        "white_variance": white,
        "neg_log_marginal_likelihood": nlml,
        "mean": mean,
        "observation_variance": obs_var,
        "latent_variance": latent_var,
    }


def svgp_case(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    z: torch.Tensor,
    xs: torch.Tensor,
    name: str,
    n_cols: int,
) -> dict:
    kernel, lengthscales, white = make_kernel(name, n_cols)
    likelihood = GaussianLikelihood()
    likelihood.initialize(noise=NOISE_VARIANCE_INIT)
    model = SvgpModel(z, kernel)
    model.double()
    likelihood.double()
    model.train()
    likelihood.train()
    n = train_x.size(0)
    mll = VariationalELBO(likelihood, model, num_data=n, beta=1.0)
    with (
        torch.no_grad(),
        gpytorch.settings.cholesky_jitter(0.0),
        gpytorch.settings.cholesky_max_tries(1),
        gpytorch.settings.fast_computations(
            covar_root_decomposition=False,
            log_prob=False,
            solves=False,
        ),
    ):
        output = model(train_x)
        elbo = mll(output, train_y)
        # VariationalELBO is a mean over n; gprx `neg_elbo` is the sum.
        neg_elbo = float((-elbo * n).item())
    mean, obs_var, latent_var = predict_latent_obs(model, likelihood, xs)
    return {
        "name": name,
        "lengthscales": lengthscales,
        "white_variance": white,
        "neg_elbo": neg_elbo,
        "mean": mean,
        "observation_variance": obs_var,
        "latent_variance": latent_var,
    }


def problem_payload(case: dict, xs: np.ndarray) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = int(case["n_rows"])
    d = int(case["n_cols"])
    coords = unpack_column_major(case["x"], n, d)
    z_coords = kmeans_z(coords, M)
    train_x = torch.as_tensor(coords, dtype=torch.float64)
    train_y = torch.as_tensor(case["y"], dtype=torch.float64).reshape(-1)
    z = torch.as_tensor(z_coords, dtype=torch.float64)
    xs_t = torch.as_tensor(xs, dtype=torch.float64)
    payload = {
        "problem": case["problem"],
        "n_rows": n,
        "n_cols": d,
        "x": case["x"],
        "y": case["y"],
        "z": pack_column_major(z_coords),
        "m": M,
        "kmeans_seed": KMEANS_SEED,
        "xs_n_rows": int(xs.shape[0]),
        "xs_n_cols": int(xs.shape[1]),
        "xs": pack_column_major(xs),
        "noise_variance": NOISE_VARIANCE_INIT,
    }
    return payload, train_x, train_y, z, xs_t


def write_json(path: Path, payload: dict) -> None:
    GOLDENS.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    torch.set_default_dtype(torch.float64)
    names = ("rbf", "matern32", "rbf_ard", "rbf_white")
    jobs = [
        ("sgpr_forrester", make_forrester(256), FORRESTER_XS, sgpr_case),
        ("sgpr_sphere", make_sphere(16), SPHERE_XS, sgpr_case),
        ("svgp_forrester", make_forrester(256), FORRESTER_XS, svgp_case),
        ("svgp_sphere", make_sphere(16), SPHERE_XS, svgp_case),
    ]
    for stem, raw, xs, runner in jobs:
        payload, train_x, train_y, z, xs_t = problem_payload(raw, xs)
        payload["kernels"] = [
            runner(train_x, train_y, z, xs_t, name, payload["n_cols"]) for name in names
        ]
        write_json(GOLDENS / f"{stem}.json", payload)
        print(f"wrote {stem}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
