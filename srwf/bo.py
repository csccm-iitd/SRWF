"""Bayesian optimisation with feature GPs: pathwise Thompson sampling and analytic acquisitions.

Thompson sampling fits a surrogate, draws ``q`` posterior functions and evaluates the objective
at the maximiser of each draw. A draw is maximised by scoring a Sobol set plus perturbations of
the best observed inputs, then refining the top candidates with projected Adam on ``grad_x f_k``.

    >>> sampler = FeatureSampler(lambda x: RWFGP(num_features=512, levels=6, steps=200))
    >>> trace = thompson_sampling(objective, bounds, sampler, budget=100)

``PosteriorMoments`` exposes the predictive mean and latent standard deviation of a fitted model
as differentiable functions of the inputs, for gradient-based maximisation of expected improvement
(``log_expected_improvement``) or an upper confidence bound (``upper_confidence_bound``).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import torch
from torch.quasirandom import SobolEngine

from .model import _FeatureGP
from .sampling import PosteriorFunction


class SampleFunction(Protocol):
    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...


class Sampler(Protocol):
    def __call__(self, x: np.ndarray, y: np.ndarray, n_samples: int, seed: int) -> SampleFunction: ...


class FeatureSampler:
    """Fit a feature GP built by ``make(x)`` and return its pathwise posterior draws."""

    def __init__(self, make: Callable[[np.ndarray], _FeatureGP]) -> None:
        self.make = make
        self.model: _FeatureGP | None = None

    def __call__(self, x: np.ndarray, y: np.ndarray, n_samples: int, seed: int) -> PosteriorFunction:
        self.model = self.make(x).fit(x, y)
        return PosteriorFunction(self.model, n_samples, seed=seed)


class PosteriorMoments:
    """Predictive mean and latent standard deviation of a fitted feature GP, in the output units.

    Unlike ``predict`` the moments are torch tensors with gradients with respect to the inputs,
    so analytic acquisition functions can be maximised with gradient steps.
    """

    def __init__(self, model: _FeatureGP) -> None:
        self.blr = model.blr
        self.x_mu, self.x_sd = model.x_mu, model.x_sd
        self.y_mu, self.y_sd = float(model.y_mu), float(model.y_sd)
        self.device = model.device

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean and latent standard deviation at the rows of ``x``, each of shape ``(N,)``."""
        xs = (x.to(self.device, torch.float32) - self.x_mu) / self.x_sd
        z = self.blr.design(xs)
        mean = z @ self.blr.mean_w + self.blr.bias
        std = self.blr._latent_var(z).clamp_min(1e-18).sqrt()
        return mean * self.y_sd + self.y_mu, std * self.y_sd


def log_expected_improvement(mean: torch.Tensor, std: torch.Tensor, best: float) -> torch.Tensor:
    """``log E[max(f - best, 0)]`` for ``f ~ N(mean, std^2)``, stable for very negative ``(mean - best) / std``."""
    u = (mean - best) / std
    normal = torch.distributions.Normal(0.0, 1.0)
    upper = torch.log(torch.exp(normal.log_prob(u)) + u * normal.cdf(u).clamp_min(0.0) + 1e-300)
    v = u.clamp(max=-1.0)
    w = torch.log(torch.special.erfcx(-v / math.sqrt(2.0)) * v.abs()) + 0.5 * math.log(math.pi / 2.0)
    lower = normal.log_prob(v) + torch.log(-torch.expm1(w).clamp(max=-1e-300))
    return torch.where(u > -1.0, upper, lower) + torch.log(std)


def upper_confidence_bound(mean: torch.Tensor, std: torch.Tensor, beta: float = 2.0) -> torch.Tensor:
    return mean + beta * std


@dataclass
class Trace:
    x: np.ndarray
    y: np.ndarray
    n_init: int
    fit_seconds: list[float] = field(default_factory=list)
    acq_seconds: list[float] = field(default_factory=list)

    @property
    def best(self) -> np.ndarray:
        """Best observed value after each evaluation."""
        return np.maximum.accumulate(self.y)


def sobol(n: int, bounds: np.ndarray, seed: int) -> np.ndarray:
    lo, hi = bounds[:, 0], bounds[:, 1]
    u = SobolEngine(bounds.shape[0], scramble=True, seed=seed).draw(n).double().numpy()
    return lo + u * (hi - lo)


def maximise(
    f: SampleFunction,
    bounds: np.ndarray,
    x_obs: np.ndarray | None = None,
    y_obs: np.ndarray | None = None,
    n_candidates: int = 2048,
    n_local: int = 512,
    n_starts: int = 4,
    steps: int = 50,
    lr: float = 0.01,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate argmax and max over the box of each draw of ``f`` (rows of ``f(x)``)."""
    lo = torch.as_tensor(bounds[:, 0], dtype=torch.float32)
    span = torch.as_tensor(bounds[:, 1] - bounds[:, 0], dtype=torch.float32)
    d = len(lo)
    g = torch.Generator().manual_seed(seed)
    u = SobolEngine(d, scramble=True, seed=seed).draw(n_candidates).float()
    if x_obs is not None and y_obs is not None and n_local > 0:
        top = np.argsort(-y_obs)[: min(5, len(y_obs))]
        centre = (torch.as_tensor(x_obs[top], dtype=torch.float32) - lo) / span
        sd = torch.tensor([0.1, 0.01]).repeat_interleave(n_local // 2)
        idx = torch.randint(0, len(top), (len(sd),), generator=g)
        local = centre[idx] + sd[:, None] * torch.randn(len(sd), d, generator=g)
        u = torch.cat([u, local.clamp(0.0, 1.0)])
    with torch.no_grad():
        vals = torch.cat([f(lo + c * span).float().cpu() for c in u.split(4096)], dim=1)
    k = min(n_starts, u.shape[0])
    start = u[vals.topk(k, dim=1).indices]
    best_u = start[:, 0].clone()
    best_v = vals.max(1).values.clone()
    if steps > 0:
        z = start.clone().requires_grad_(True)
        opt = torch.optim.Adam([z], lr=lr)
        for _ in range(steps):
            (z.grad,) = torch.autograd.grad(-f(lo + z * span).sum(), z)
            opt.step()
            with torch.no_grad():
                z.clamp_(0.0, 1.0)
        with torch.no_grad():
            v = f(lo + z * span).float().cpu()
            vk, ik = v.max(1)
            better = vk > best_v
            best_u[better] = z.detach()[better, ik[better]]
            best_v = torch.where(better, vk, best_v)
    x = (lo + best_u * span).double().numpy()
    return np.clip(x, bounds[:, 0], bounds[:, 1]), best_v.double().numpy()


def thompson_sampling(
    objective: Callable[[np.ndarray], np.ndarray],
    bounds: np.ndarray,
    sampler: Sampler,
    budget: int,
    n_init: int | None = None,
    batch_size: int = 1,
    seed: int = 0,
    callback: Callable[[int, SampleFunction, np.ndarray, Trace], None] | None = None,
    **maximise_kwargs,
) -> Trace:
    """Maximise ``objective`` over ``bounds`` (shape ``(d, 2)``) with batch Thompson sampling."""
    bounds = np.asarray(bounds, dtype=np.float64)
    d = bounds.shape[0]
    n_init = n_init or 2 * d + 1
    x = sobol(n_init, bounds, seed)
    trace = Trace(x, np.asarray(objective(x), dtype=np.float64), n_init)
    it = 0
    while len(trace.y) < budget:
        q = min(batch_size, budget - len(trace.y))
        t0 = time.perf_counter()
        f = sampler(trace.x, trace.y, q, seed * 100003 + it)
        t1 = time.perf_counter()
        x_next, _ = maximise(f, bounds, trace.x, trace.y, seed=seed * 100003 + it, **maximise_kwargs)
        t2 = time.perf_counter()
        if callback is not None:
            callback(it, f, x_next, trace)
        trace.x = np.concatenate([trace.x, x_next])
        trace.y = np.concatenate([trace.y, np.asarray(objective(x_next), dtype=np.float64)])
        trace.fit_seconds.append(t1 - t0)
        trace.acq_seconds.append(t2 - t1)
        it += 1
    return trace
