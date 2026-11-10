"""Pathwise function samples from random-feature GPs.

A fitted ``RWFGP``/``RFFGP`` has the exact weight posterior ``w ~ N(m, A^-1)``, so each draw
``f_k(x) = z(x)^T w_k + b`` is a whole function: it is evaluated anywhere at ``O(D)`` cost per
point, is consistent across calls, and is differentiable in ``x``.

    >>> f = PosteriorFunction(model, n_samples=16, seed=0)
    >>> values = f(x)              # (16, N), original output scale
    >>> grads = f.gradient(x)      # (16, N, d)
"""

from __future__ import annotations

import numpy as np
import torch

from .model import _FeatureGP


class PosteriorFunction:
    """``K`` frozen function draws from the posterior (or prior) of a fitted feature GP.

    Args:
        model: a fitted ``RWFGP`` or ``RFFGP``.
        n_samples: number of draws ``K``.
        prior: draw from the prior ``w ~ N(0, tau^2 I)`` instead of the posterior.
        seed: seed of the weight draws.
    """

    def __init__(self, model: _FeatureGP, n_samples: int = 1, prior: bool = False, seed: int = 0) -> None:
        blr = model.blr
        g = torch.Generator().manual_seed(int(seed))
        if prior:
            eps = torch.randn(blr.features.num_features, int(n_samples), dtype=torch.float64, generator=g)
            w, bias = eps.to(model.device) * blr.prior_var.sqrt(), 0.0
        else:
            w = blr.posterior_weights(int(n_samples), generator=g)
            bias = float(blr.bias)
        self.features = blr.features
        self.weights = (w * model.y_sd).detach()
        self.offset = bias * model.y_sd + model.y_mu
        self.x_mu, self.x_sd = model.x_mu, model.x_sd
        self.device = model.device
        self.n_samples = int(n_samples)

    def _design(self, x: torch.Tensor) -> torch.Tensor:
        return self.features((x.to(self.device, torch.float32) - self.x_mu) / self.x_sd).double()

    def __call__(self, x) -> torch.Tensor:
        """Evaluate every draw at ``x``.

        ``x`` of shape ``(N, d)`` returns ``(K, N)``; ``x`` of shape ``(K, M, d)`` evaluates draw
        ``k`` on its own points ``x[k]`` and returns ``(K, M)``.
        """
        x = torch.as_tensor(x, device=self.device) if not isinstance(x, torch.Tensor) else x
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        if x.ndim == 2:
            return (self._design(x) @ self.weights).t() + self.offset
        if x.shape[0] != self.n_samples:
            raise ValueError(f"expected {self.n_samples} point sets, got {x.shape[0]}")
        k, m, d = x.shape
        z = self._design(x.reshape(k * m, d)).reshape(k, m, -1)
        return torch.einsum("kmd,dk->km", z, self.weights) + self.offset

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        """``d f_k / d x`` at paired points ``x`` of shape ``(K, M, d)``."""
        x = x.detach().to(self.device, torch.float32).requires_grad_(True)
        with torch.enable_grad():
            (g,) = torch.autograd.grad(self(x).sum(), x)
        return g

    def numpy(self, x) -> np.ndarray:
        with torch.no_grad():
            return self(torch.as_tensor(np.asarray(x, dtype=np.float32).reshape(len(x), -1))).cpu().numpy()
