"""The comparison regression, GPy: ARD RBF, learned from a fixed start."""
import GPy
import numpy as np

rng = np.random.default_rng(0)
x, xs = rng.normal(size=(200, 3)), rng.normal(size=(50, 3))
y, ys = np.sin(x.sum(1)) + 0.1 * rng.normal(size=200), np.sin(xs.sum(1))
d = x.shape[1]

# snippet:begin
kernel = GPy.kern.RBF(d, variance=1.0, lengthscale=[1.0] * d, ARD=True)
model = GPy.models.GPRegression(x, y[:, None], kernel)
model.Gaussian_noise.variance = 0.1
model.optimize()
mean, var = model.predict(xs)  # includes the noise
nlpd = np.mean(0.5 * np.log(2 * np.pi * var) + 0.5 * (ys[:, None] - mean) ** 2 / var)
# snippet:end
print(f"gpy      mean[0]={mean[0, 0]:.4f} std[0]={var[0, 0] ** 0.5:.4f} nlpd={nlpd:.4f}")
