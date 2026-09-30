"""The comparison regression, GPyTorch: ARD RBF, learned from a fixed start."""
import gpytorch
import numpy as np
import torch

rng = np.random.default_rng(0)
x = torch.tensor(rng.normal(size=(200, 3)))
xs = torch.tensor(rng.normal(size=(50, 3)))
y = torch.tensor(np.sin(x.numpy().sum(1)) + 0.1 * rng.normal(size=200))
ys = torch.tensor(np.sin(xs.numpy().sum(1)))
d = x.shape[1]

# snippet:begin
class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, x, y, likelihood):
        super().__init__(x, y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=d))

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
model = ExactGP(x, y, likelihood).double()
model.train(); likelihood.train()
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # there is no default optimizer
for _ in range(50):
    optimizer.zero_grad()
    loss = -mll(model(x), y)
    loss.backward()
    optimizer.step()
model.eval(); likelihood.eval()
with torch.no_grad():
    pred = likelihood(model(xs))  # the observation distribution
    mean, var = pred.mean, pred.variance
nlpd = torch.mean(0.5 * torch.log(2 * torch.pi * var) + 0.5 * (ys - mean) ** 2 / var)
# snippet:end
print(f"gpytorch mean[0]={mean[0]:.4f} std[0]={var[0].sqrt():.4f} nlpd={nlpd:.4f}")
