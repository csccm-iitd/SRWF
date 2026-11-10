"""Random feature maps: Random Wavelet Features (RWF) and Random Fourier Features (RFF).

RWF approximates the non-stationary kernel

    k(x, x') = sum_j c_j  E_{(s, t) ~ p_j} [ psi_{s,t}(x) psi_{s,t}(x') ],

with ``psi_{s,t}(x) = psi((x - t) / (s * ell))`` and one sampling density ``p_j`` per dyadic
octave ``s in s_0 * 2^{-(j + [0, 1))}`` (pooled over octaves this is the log-uniform scale law
of the paper). The Monte Carlo estimate ``z(x)^T z(x')`` uses ``n_j`` atoms per octave. All
random draws are frozen at construction; only the hyperparameters of ``p`` (``s_0``, per-octave
width offsets, per-octave weights ``c_j``, ARD lengthscales ``ell``) are learned, so the feature
map is a reparameterised Monte Carlo estimator of a valid kernel for every hyperparameter value.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn.functional import softplus

from .wavelets import get_wavelet


def _inv_softplus(x: float) -> float:
    return x + math.log(-math.expm1(-x))


def split_features(total: int, levels: int, growth: float) -> list[int]:
    """Split ``total`` atoms over ``levels`` octaves proportionally to ``growth ** j``."""
    w = torch.tensor([growth**j for j in range(levels)], dtype=torch.float64)
    raw = w / w.sum() * total
    counts = torch.floor(raw).long()
    rest = int(total - counts.sum())
    order = torch.argsort(raw - counts, descending=True)
    counts[order[:rest]] += 1
    return [max(int(c), 1) for c in counts]


class RandomWaveletFeatures(nn.Module):
    """Multi-resolution random wavelet feature map ``z: R^d -> R^D``.

    Args:
        input_dim: input dimension ``d``.
        num_features: total number of wavelet atoms ``D`` (before the optional low-pass block).
        levels: number of dyadic octaves ``J``.
        wavelet: ``"mexican_hat"``, ``"morlet"`` or ``"gaussian"``.
        growth: atoms per octave grow as ``growth ** j`` (finer octaves need more atoms).
        lowpass: number of Gaussian scaling functions at the coarsest scale (0 disables).
        anchors: if positive, octave weights vary with location, ``log c_j(t) = a_j + sum_k beta_jk phi_k(t)``
            with Gaussian bumps ``phi_k`` at ``anchors`` points spread over the inputs; the
            sampling measure stays non-negative, so the kernel is positive definite for any ``beta``.
        anchor_prior: variance of the Gaussian hyperprior on ``beta``.
        ard: learn one lengthscale per input dimension (otherwise a single shared one).
        omega: Morlet carrier frequency.
        seed: seed of the frozen random draws.
    """

    def __init__(
        self,
        input_dim: int,
        num_features: int = 1024,
        levels: int = 5,
        wavelet: str = "mexican_hat",
        growth: float = 2.0,
        lowpass: int = 0,
        ard: bool = True,
        omega: float = 5.0,
        anchors: int = 0,
        anchor_prior: float = 1.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.n_anchors = int(anchors)
        self.anchor_prior = float(anchor_prior)
        self.levels = int(levels)
        self.wavelet = get_wavelet(wavelet)
        self.omega = float(omega)
        self.counts = split_features(int(num_features), self.levels, float(growth))
        if lowpass > 0:
            self.counts = self.counts + [int(lowpass)]
        self.has_lowpass = lowpass > 0
        self.num_features = int(sum(self.counts))
        n_blocks = len(self.counts)

        g = torch.Generator().manual_seed(int(seed))
        level_index = torch.cat([torch.full((c,), j, dtype=torch.long) for j, c in enumerate(self.counts)])
        octave = level_index.double().clone()
        if self.has_lowpass:
            octave[level_index == n_blocks - 1] = -1.0
        u = torch.rand(self.num_features, generator=g, dtype=torch.float64)
        self.register_buffer("level_index", level_index)
        self.register_buffer("log2_rel_scale", -(octave + u).float())
        directions = torch.randn(self.num_features, self.input_dim, generator=g)
        self.register_buffer("directions", directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-12))
        self.register_buffer("translations", torch.zeros(self.num_features, self.input_dim))
        self.register_buffer("block_rms", torch.ones(n_blocks))
        self.seed = int(seed)

        n_ls = self.input_dim if ard else 1
        self.raw_lengthscale = nn.Parameter(torch.full((n_ls,), _inv_softplus(1.0)))
        self.log_base_scale = nn.Parameter(torch.tensor(0.0))
        self.level_log_scale = nn.Parameter(torch.zeros(n_blocks))
        self.level_log_weight = nn.Parameter(torch.zeros(n_blocks))
        if self.n_anchors > 0:
            self.register_buffer("anchor_basis", torch.zeros(self.num_features, self.n_anchors))
            self.anchor_log_weight = nn.Parameter(torch.zeros(n_blocks, self.n_anchors))

    @property
    def lengthscale(self) -> torch.Tensor:
        return softplus(self.raw_lengthscale) + 1e-6

    def scales(self) -> torch.Tensor:
        log_s = self.log_base_scale + self.log2_rel_scale * math.log(2.0) + self.level_log_scale[self.level_index]
        return torch.exp(log_s.clamp(-12.0, 12.0))

    def atom_log_weights(self) -> torch.Tensor:
        """``log c_j(t_i)`` for every atom: per-octave weight, optionally modulated by location."""
        logw = self.level_log_weight[self.level_index]
        if self.n_anchors > 0:
            logw = logw + (self.anchor_basis * self.anchor_log_weight[self.level_index]).sum(1)
        return logw.clamp(-15.0, 15.0)

    def penalty(self) -> torch.Tensor:
        """Negative log hyperprior ``sum beta^2 / (2 rho)`` of the location-dependent weights."""
        if self.n_anchors == 0:
            return self.level_log_weight.new_zeros(())
        return 0.5 * self.anchor_log_weight.pow(2).sum() / self.anchor_prior

    @torch.no_grad()
    def initialize(
        self,
        x: torch.Tensor,
        centres: str | torch.Tensor = "data",
        base_scale: float | None = None,
        max_rows: int = 20000,
    ) -> RandomWaveletFeatures:
        """Draw the translations and set the coarsest scale from the training inputs.

        ``centres`` is ``"data"`` (training inputs, the empirical ``p_t``), ``"box"`` (uniform on
        the bounding box) or a tensor of candidate locations to draw from.
        """
        x = x.to(self.translations)
        g = torch.Generator().manual_seed(self.seed + 1)
        n = self.num_features
        if isinstance(centres, torch.Tensor):
            pool = centres.to(self.translations)
            idx = torch.randint(0, pool.shape[0], (n,), generator=g)
            t = pool[idx.to(pool.device)]
        elif centres == "data":
            idx = torch.randint(0, x.shape[0], (n,), generator=g)
            t = x[idx.to(x.device)]
        elif centres == "box":
            lo, hi = x.min(0).values, x.max(0).values
            t = lo + torch.rand(n, self.input_dim, generator=g).to(x) * (hi - lo)
        else:
            raise ValueError(f"unknown centres {centres!r}")
        self.translations.copy_(t)
        if self.n_anchors > 0:
            self._set_anchors(x, g)
        if base_scale is None:
            span = (x.max(0).values - x.min(0).values).clamp_min(1e-6)
            base_scale = float(span.mean()) / 2.0
        self.log_base_scale.fill_(math.log(base_scale))
        self.block_rms.fill_(1.0)
        sub = x[torch.randperm(x.shape[0], generator=g)[:max_rows].to(x.device)]
        raw = self._raw(sub)
        sq = torch.zeros(len(self.counts), device=raw.device, dtype=raw.dtype).index_add_(
            0, self.level_index, raw.pow(2).mean(0)
        )
        cnt = torch.tensor(self.counts, device=raw.device, dtype=sq.dtype)
        self.block_rms.copy_((sq / cnt).sqrt().clamp_min(1e-8))
        return self

    def _set_anchors(self, x: torch.Tensor, g: torch.Generator) -> None:
        k = min(self.n_anchors, x.shape[0])
        pool = x[torch.randperm(x.shape[0], generator=g)[: min(x.shape[0], 20000)].to(x.device)]
        idx = [int(torch.randint(0, pool.shape[0], (1,), generator=g))]
        dist = (pool - pool[idx[0]]).pow(2).sum(1)
        for _ in range(1, k):
            idx.append(int(torch.argmax(dist)))
            dist = torch.minimum(dist, (pool - pool[idx[-1]]).pow(2).sum(1))
        a = pool[idx]
        nn_d2 = torch.cdist(a, a).fill_diagonal_(float("inf")).min(1).values.pow(2)
        bw2 = nn_d2.median().clamp_min(1e-8)
        basis = torch.exp(-0.5 * torch.cdist(self.translations, a).pow(2) / bw2)
        self.anchor_basis.zero_()
        self.anchor_basis[:, :k] = basis

    def _raw(self, x: torch.Tensor) -> torch.Tensor:
        ell = self.lengthscale
        s = self.scales()
        xs = x / ell
        ts = self.translations / ell
        r2 = (xs.pow(2).sum(1, keepdim=True) + ts.pow(2).sum(1) - 2.0 * xs @ ts.t()).clamp_min(0.0) / s.pow(2)
        p = None
        if self.wavelet.directional:
            p = (xs @ self.directions.t() - (ts * self.directions).sum(1)) / s
        d = self.input_dim
        if self.has_lowpass:
            lp = self.level_index == len(self.counts) - 1
            out = self._mother(r2, p, d)
            return torch.where(lp, torch.exp(-0.5 * r2), out)
        return self._mother(r2, p, d)

    def _mother(self, r2: torch.Tensor, p: torch.Tensor | None, d: int) -> torch.Tensor:
        if self.wavelet.name == "morlet":
            return self.wavelet.fn(r2, p, d, self.omega)
        return self.wavelet.fn(r2, p, d)

    def column_scale(self) -> torch.Tensor:
        """Per-atom amplitude ``sqrt(c_j(t_i) / n_j) / rms_j``."""
        cnt = torch.tensor(self.counts, device=self.block_rms.device, dtype=self.block_rms.dtype)
        per_block = (cnt.rsqrt() / self.block_rms)[self.level_index]
        return per_block * torch.exp(0.5 * self.atom_log_weights())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._raw(x) * self.column_scale()

    def kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return self(x1) @ self(x2).t()


class RandomFourierFeatures(nn.Module):
    """Random Fourier features for the squared-exponential kernel, ``sqrt(2c/D) cos(w^T x / ell + b)``."""

    def __init__(self, input_dim: int, num_features: int = 1024, ard: bool = True, seed: int = 0) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_features = int(num_features)
        g = torch.Generator().manual_seed(int(seed))
        self.register_buffer("omega", torch.randn(self.num_features, self.input_dim, generator=g))
        self.register_buffer("phase", torch.rand(self.num_features, generator=g) * 2.0 * math.pi)
        n_ls = self.input_dim if ard else 1
        self.raw_lengthscale = nn.Parameter(torch.full((n_ls,), _inv_softplus(1.0)))
        self.log_outputscale = nn.Parameter(torch.tensor(0.0))

    @property
    def lengthscale(self) -> torch.Tensor:
        return softplus(self.raw_lengthscale) + 1e-6

    @torch.no_grad()
    def initialize(self, x: torch.Tensor, **_: object) -> RandomFourierFeatures:
        span = (x.max(0).values - x.min(0).values).clamp_min(1e-6)
        ls = (span / 4.0) if self.raw_lengthscale.numel() > 1 else span.mean().view(1) / 4.0
        self.raw_lengthscale.copy_(torch.log(torch.expm1(ls.to(self.raw_lengthscale))))
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = (x / self.lengthscale) @ self.omega.t() + self.phase
        return torch.sqrt(2.0 * torch.exp(self.log_outputscale) / self.num_features) * torch.cos(z)

    def kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return self(x1) @ self(x2).t()


def concat_features(blocks: Sequence[nn.Module]) -> nn.Module:
    return _Concat(blocks)


class _Concat(nn.Module):
    def __init__(self, blocks: Sequence[nn.Module]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.num_features = int(sum(b.num_features for b in blocks))

    @torch.no_grad()
    def initialize(self, x: torch.Tensor, **kw: object) -> _Concat:
        for b in self.blocks:
            b.initialize(x, **kw)
        return self

    def penalty(self) -> torch.Tensor:
        return sum(b.penalty() for b in self.blocks if hasattr(b, "penalty"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([b(x) for b in self.blocks], dim=1)
