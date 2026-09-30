"""The comparison regression, scikit-learn: ARD RBF, learned from a fixed start."""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

rng = np.random.default_rng(0)
x, xs = rng.normal(size=(200, 3)), rng.normal(size=(50, 3))
y, ys = np.sin(x.sum(1)) + 0.1 * rng.normal(size=200), np.sin(xs.sum(1))
d = x.shape[1]

# snippet:begin
kernel = ConstantKernel(1.0) * RBF([1.0] * d) + WhiteKernel(0.1)
gp = GaussianProcessRegressor(kernel).fit(x, y)
mean, std = gp.predict(xs, return_std=True)  # std includes the noise
nlpd = np.mean(0.5 * np.log(2 * np.pi * std**2) + 0.5 * (ys - mean) ** 2 / std**2)
# snippet:end
print(f"sklearn  mean[0]={mean[0]:.4f} std[0]={std[0]:.4f} nlpd={nlpd:.4f}")
