"""The Mauna Loa kernel of Rasmussen & Williams §5.4.3 in each library.

``θ = [s1², ℓ1, s2², ℓ2, ℓ3, p, s3², ℓ4, α, s4², ℓ5, σn²]`` (amplitudes as
variances, in the standardized ``y`` units of the case):

    k = s1² RBF(ℓ1) + s2² RBF(ℓ2) Periodic(ℓ3, p) + s3² RQ(ℓ4, α) + s4² RBF(ℓ5)  (+ noise σn²)

    RBF      exp(-d² / 2ℓ²)
    Periodic exp(-2 sin²(π d / p) / ℓ²)
    RQ       (1 + d² / (2αℓ²))^-α

Each builder maps this to the library's own parametrization; ``check`` proves
the four agree at a fixed θ.
"""

from __future__ import annotations

import math


def sklearn_kernel(theta):
    from sklearn.gaussian_process.kernels import (
        RBF,
        ConstantKernel as C,
        ExpSineSquared,
        RationalQuadratic,
        WhiteKernel,
    )

    s1, l1, s2, l2, l3, p, s3, l4, a, s4, l5, noise = theta
    return (
        C(s1) * RBF(l1)
        + C(s2) * RBF(l2) * ExpSineSquared(l3, p)
        + C(s3) * RationalQuadratic(l4, a)
        + C(s4) * RBF(l5)
        + WhiteKernel(noise)
    )


def gpytorch_kernel(theta):
    """The kernel; the likelihood noise is set by the caller."""
    import gpytorch

    s1, l1, s2, l2, l3, p, s3, l4, a, s4, l5, _ = theta
    k = gpytorch.kernels

    def scaled(base, variance):
        kernel = k.ScaleKernel(base)
        kernel.outputscale = variance
        return kernel

    rbf1 = k.RBFKernel()
    rbf1.lengthscale = l1
    rbf2 = k.RBFKernel()
    rbf2.lengthscale = l2
    periodic = k.PeriodicKernel()
    periodic.lengthscale = l3**2  # GPyTorch divides sin² by λ, not λ²
    periodic.period_length = p
    rq = k.RQKernel()
    rq.lengthscale = l4
    rq.alpha = a
    rbf5 = k.RBFKernel()
    rbf5.lengthscale = l5
    return scaled(rbf1, s1) + scaled(rbf2 * periodic, s2) + scaled(rq, s3) + scaled(rbf5, s4)


def gpy_kernel(theta):
    import GPy

    s1, l1, s2, l2, l3, p, s3, l4, a, s4, l5, _ = theta
    periodic = GPy.kern.StdPeriodic(1, variance=1.0, period=p, lengthscale=l3 / 2.0)
    periodic.variance.fix(warning=False)  # the amplitude is s2² on the RBF factor
    return (
        GPy.kern.RBF(1, variance=s1, lengthscale=l1)
        + GPy.kern.RBF(1, variance=s2, lengthscale=l2) * periodic
        + GPy.kern.RatQuad(1, variance=s3, lengthscale=l4 * math.sqrt(a), power=a)
        + GPy.kern.RBF(1, variance=s4, lengthscale=l5)
    )
