"""Exact Bayesian linear regression on an explicit feature map.

Model: ``y = z(x)^T w + b + eps``, ``w ~ N(0, tau^2 I_D)``, ``eps ~ N(0, sigma^2)``. With
``r = y - b`` and ``A = tau^-2 I + sigma^-2 Z^T Z``:

    m              = sigma^-2 A^-1 Z^T r
    log p(y | th)  = -1/2 [sigma^-2 ||r||^2 - m^T A m] - 1/2 [N log sigma^2 + D log tau^2 + log|A|] - N/2 log 2 pi
    mean(x*)       = z*^T m + b
    var(y* | x*)   = z*^T A^-1 z* + sigma^2

Data enter only through ``(Z^T Z, Z^T r, ||r||^2, N)``, accumulated in row chunks, so the
cost is ``O(N D^2)`` time and ``O(D^2)`` memory for any ``N``. When ``N < D`` the same posterior
is available in the dual form, with ``K = tau^2 Z Z^T + sigma^2 I`` and ``alpha = K^-1 r``:

    m              = tau^2 Z^T alpha
    var(y* | x*)   = tau^2 ||z*||^2 - tau^4 ||L^-1 Z z*||^2 + sigma^2,   K = L L^T

at ``O(N^2 D)`` time and ``O(N D)`` memory. Linear algebra is in float64.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class Stats:
    ztz: torch.Tensor
    ztr: torch.Tensor
    rss: torch.Tensor
    n: int

    def __add__(self, other: Stats) -> Stats:
        return Stats(self.ztz + other.ztz, self.ztr + other.ztr, self.rss + other.rss, self.n + other.n)

    def add_(self, z: torch.Tensor, r: torch.Tensor) -> Stats:
        """Accumulate a chunk of design rows ``z`` and residuals ``r`` in place."""
        self.ztz.addmm_(z.t(), z)
        self.ztr.addmv_(z.t(), r)
        self.rss = self.rss + r.dot(r)
        self.n += int(r.numel())
        return self


def _chunks(n: int, chunk: int) -> Iterator[slice]:
    for start in range(0, n, chunk):
        yield slice(start, min(start + chunk, n))


def cholesky(a: torch.Tensor, retries: int = 6) -> torch.Tensor:
    """Lower Cholesky factor with an escalating diagonal jitter; at most one extra ``D x D`` buffer."""
    chol, info = torch.linalg.cholesky_ex(a)
    if int(info) == 0:
        return chol
    jitter = 1e-8 * torch.diagonal(a).mean().detach().abs().clamp_min(1e-12)
    if a.requires_grad:
        for _ in range(retries):
            chol, info = torch.linalg.cholesky_ex(a + torch.diag_embed(jitter.expand(a.shape[0])))
            if int(info) == 0:
                return chol
            jitter = jitter * 10.0
        raise torch.linalg.LinAlgError("Cholesky failed after jitter retries")
    diag = torch.diagonal(a)
    added = torch.zeros_like(jitter)
    try:
        for _ in range(retries):
            del chol
            diag.add_(jitter - added)
            added = jitter
            chol, info = torch.linalg.cholesky_ex(a)
            if int(info) == 0:
                return chol
            jitter = jitter * 10.0
    finally:
        diag.sub_(added)
    raise torch.linalg.LinAlgError("Cholesky failed after jitter retries")


class BayesianLinearRegression(nn.Module):
    """Closed-form Gaussian posterior over the weights of ``features`` with learnable ``sigma^2, tau^2``."""

    def __init__(self, features: nn.Module, noise: float = 0.1, prior_var: float = 1.0) -> None:
        super().__init__()
        self.features = features
        self.log_noise = nn.Parameter(torch.tensor(math.log(noise)))
        self.log_prior_var = nn.Parameter(torch.tensor(math.log(prior_var)))
        self.register_buffer("bias", torch.zeros((), dtype=torch.float64))
        self.register_buffer("chol", torch.zeros(0, 0, dtype=torch.float64))
        self.register_buffer("mean_w", torch.zeros(0, dtype=torch.float64))
        self.register_buffer("dual_z", torch.zeros(0, 0, dtype=torch.float64))

    @property
    def noise(self) -> torch.Tensor:
        return torch.exp(self.log_noise.clamp(-25.0, 15.0)).double()

    @property
    def prior_var(self) -> torch.Tensor:
        return torch.exp(self.log_prior_var.clamp(-25.0, 15.0)).double()

    @property
    def fitted(self) -> bool:
        return self.mean_w.numel() > 0

    @property
    def dual(self) -> bool:
        """Whether the cached posterior is in the dual form (``chol`` is then the factor of ``K``)."""
        return self.dual_z.numel() > 0

    def design(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).double()

    def stats(self, x: torch.Tensor, y: torch.Tensor, chunk: int = 16384) -> Stats:
        if x.shape[0] == 0:
            raise ValueError("empty training set")
        out: Stats | None = None
        for sl in _chunks(x.shape[0], chunk):
            z = self.design(x[sl])
            r = y[sl].double() - self.bias
            if out is None:
                out = Stats(z.t() @ z, z.t() @ r, r.dot(r), int(r.numel()))
            else:
                out.add_(z, r)
            del z
        return out

    def _posterior(self, s: Stats, overwrite: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Cholesky factor of ``A`` and posterior mean; ``overwrite`` builds ``A`` in the storage of ``s.ztz``."""
        a = s.ztz.div_(self.noise) if overwrite else s.ztz / self.noise
        a.diagonal().add_(1.0 / self.prior_var)
        chol = cholesky(a)
        half = torch.linalg.solve_triangular(chol, (s.ztr / self.noise).unsqueeze(-1), upper=False)
        mean = torch.linalg.solve_triangular(chol.t(), half, upper=True).squeeze(-1)
        return chol, mean

    def log_marginal_likelihood_from_stats(self, s: Stats) -> torch.Tensor:
        d = s.ztz.shape[0]
        chol, mean = self._posterior(s)
        quad = s.rss / self.noise - mean.dot(s.ztr) / self.noise
        logdet = 2.0 * torch.log(torch.diagonal(chol)).sum()
        return (
            -0.5 * quad
            - 0.5 * (s.n * torch.log(self.noise) + d * torch.log(self.prior_var) + logdet)
            - 0.5 * s.n * math.log(2.0 * math.pi)
        )

    def log_marginal_likelihood(self, x: torch.Tensor, y: torch.Tensor, chunk: int = 16384) -> torch.Tensor:
        """Exact ``log p(y | X, theta)``; uses the ``N x N`` dual form when ``N < D``."""
        n = x.shape[0]
        d = self.features.num_features
        if n < d and n <= 8192:
            z = self.design(x)
            r = (y.double() - self.bias).unsqueeze(-1)
            k = self.prior_var * (z @ z.t()) + self.noise * torch.eye(n, dtype=z.dtype, device=z.device)
            chol = cholesky(k)
            alpha = torch.cholesky_solve(r, chol)
            return -0.5 * (r * alpha).sum() - torch.log(torch.diagonal(chol)).sum() - 0.5 * n * math.log(2.0 * math.pi)
        return self.log_marginal_likelihood_from_stats(self.stats(x, y, chunk))

    def loo(self, x: torch.Tensor, y: torch.Tensor, chunk: int = 16384) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact leave-one-out predictive means and variances (no refitting)."""
        s = self.stats(x, y, chunk)
        chol, mean = self._posterior(s)
        mus, vs = [], []
        for sl in _chunks(x.shape[0], chunk):
            z = self.design(x[sl])
            yy = y[sl].double()
            e = yy - self.bias - z @ mean
            h = torch.linalg.solve_triangular(chol, z.t(), upper=False).pow(2).sum(0) / self.noise
            one_minus_h = (1.0 - h).clamp_min(1e-9)
            mus.append(yy - e / one_minus_h)
            vs.append(self.noise / one_minus_h)
        return torch.cat(mus), torch.cat(vs)

    @torch.no_grad()
    def condition(
        self, x: torch.Tensor, y: torch.Tensor, chunk: int = 16384, solver: str = "primal"
    ) -> BayesianLinearRegression:
        """Compute and cache the exact weight posterior on all rows.

        ``solver`` is ``"primal"`` (factor the ``D x D`` precision), ``"dual"`` (factor the
        ``N x N`` kernel matrix) or ``"auto"`` (dual when ``N < D`` and ``N <= 8192``).
        """
        n, d = x.shape[0], self.features.num_features
        if solver not in ("primal", "dual", "auto"):
            raise ValueError(f"unknown solver {solver!r}")
        self.chol, self.mean_w = self.chol.new_zeros(0, 0), self.mean_w.new_zeros(0)
        self.dual_z = self.dual_z.new_zeros(0, 0)
        if solver == "dual" or (solver == "auto" and n < d and n <= 8192):
            z = self.design(x)
            k = z @ z.t()
            k.mul_(self.prior_var).diagonal().add_(self.noise)
            self.chol = cholesky(k)
            alpha = self._dual_solve((y.double() - self.bias).unsqueeze(-1))
            self.mean_w = (z.t() @ alpha).squeeze(-1).mul_(self.prior_var)
            self.dual_z = z
            return self
        self.chol, self.mean_w = self._posterior(self.stats(x, y, chunk), overwrite=True)
        return self

    def _dual_solve(self, b: torch.Tensor) -> torch.Tensor:
        half = torch.linalg.solve_triangular(self.chol, b, upper=False)
        return torch.linalg.solve_triangular(self.chol.t(), half, upper=True)

    def _latent_var(self, z: torch.Tensor) -> torch.Tensor:
        if self.dual:
            proj = torch.linalg.solve_triangular(self.chol, self.dual_z @ z.t(), upper=False)
            return (self.prior_var * z.pow(2).sum(1) - self.prior_var**2 * proj.pow(2).sum(0)).clamp_min(0.0)
        return torch.linalg.solve_triangular(self.chol, z.t(), upper=False).pow(2).sum(0)

    @torch.no_grad()
    def posterior_weights(self, n_samples: int = 1, generator: torch.Generator | None = None) -> torch.Tensor:
        """Draws ``w ~ N(m, A^-1)`` of the weight posterior, shape ``(D, n_samples)``.

        In the dual form the draws use Matheron's rule, ``w = w0 + tau^2 Z^T K^-1 (r - Z w0 - eps)``
        with ``w0 ~ N(0, tau^2 I)`` and ``eps ~ N(0, sigma^2 I)``.
        """
        eps = torch.randn(self.mean_w.numel(), n_samples, dtype=torch.float64, generator=generator)
        eps = eps.to(self.chol.device)
        if not self.dual:
            return self.mean_w.unsqueeze(-1) + torch.linalg.solve_triangular(self.chol.t(), eps, upper=True)
        noise = torch.randn(self.dual_z.shape[0], n_samples, dtype=torch.float64, generator=generator)
        w0 = eps * self.prior_var.sqrt()
        resid = self.dual_z @ w0 + noise.to(self.chol.device) * self.noise.sqrt()
        return self.mean_w.unsqueeze(-1) + w0 - self.prior_var * (self.dual_z.t() @ self._dual_solve(resid))

    @torch.no_grad()
    def predict(
        self, x: torch.Tensor, chunk: int = 16384, include_noise: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predictive mean and variance; ``include_noise=False`` returns the latent ``f`` variance."""
        if not self.fitted:
            raise RuntimeError("call condition() before predict()")
        means, variances = [], []
        for sl in _chunks(x.shape[0], chunk):
            z = self.design(x[sl])
            means.append(z @ self.mean_w + self.bias)
            v = self._latent_var(z)
            variances.append(v + self.noise if include_noise else v)
        return torch.cat(means), torch.cat(variances)

    @torch.no_grad()
    def sample_posterior(
        self, x: torch.Tensor, n_samples: int = 1, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Joint posterior samples of the latent function, shape ``(n_samples, len(x))``."""
        return (self.design(x) @ self.posterior_weights(n_samples, generator) + self.bias).t()

    @torch.no_grad()
    def sample_prior(
        self, x: torch.Tensor, n_samples: int = 1, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        d = self.features.num_features
        w = torch.randn(d, n_samples, dtype=torch.float64, generator=generator).to(x.device) * self.prior_var.sqrt()
        return (self.design(x) @ w).t()
