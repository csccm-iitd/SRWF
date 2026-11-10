"""Mother wavelets on R^d.

Every wavelet is written as a function of two quantities computed from the scaled offset
``u = (x - t) / s``: the squared radius ``r2 = ||u||^2`` and, for directional wavelets, the
projection ``p = v^T u`` onto a unit direction ``v``. Both are obtained with matrix products,
so a feature block costs ``O(N D d)`` time and ``O(N D)`` memory.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

WaveletFn = Callable[[torch.Tensor, torch.Tensor | None, int], torch.Tensor]


def mexican_hat(r2: torch.Tensor, p: torch.Tensor | None, d: int) -> torch.Tensor:
    """Negative Laplacian of a Gaussian, ``(d - ||u||^2) exp(-||u||^2 / 2)``; zero mean on R^d."""
    return (d - r2) * torch.exp(-0.5 * r2)


def morlet(r2: torch.Tensor, p: torch.Tensor | None, d: int, omega: float = 5.0) -> torch.Tensor:
    """Real Morlet (Gabor) wavelet ``exp(-||u||^2/2) (cos(omega v^T u) - exp(-omega^2/2))``.

    The constant correction makes the integral exactly zero (admissibility) for any ``omega``.
    """
    return torch.exp(-0.5 * r2) * (torch.cos(omega * p) - math.exp(-0.5 * omega**2))


def gaussian(r2: torch.Tensor, p: torch.Tensor | None, d: int) -> torch.Tensor:
    """Gaussian scaling function (father wavelet), the low-pass channel of a multiresolution analysis."""
    return torch.exp(-0.5 * r2)


@dataclass(frozen=True)
class Wavelet:
    name: str
    fn: WaveletFn
    directional: bool
    sup_abs: Callable[[int], float]
    lipschitz: Callable[[int], float]


def _mh_sup(d: int) -> float:
    return float(d)


def _mh_lip(d: int) -> float:
    # sup_r |d/dr (d - r^2) e^{-r^2/2}| = sup_r |r (r^2 - d - 2)| e^{-r^2/2}
    r = torch.linspace(0.0, 10.0, 20001, dtype=torch.float64)
    return float((r * (r**2 - d - 2)).abs().mul(torch.exp(-0.5 * r**2)).max())


WAVELETS: dict[str, Wavelet] = {
    "mexican_hat": Wavelet("mexican_hat", mexican_hat, False, _mh_sup, _mh_lip),
    "morlet": Wavelet("morlet", morlet, True, lambda d: 1.0 + math.exp(-12.5), lambda d: 1.0 + 5.0),
    "gaussian": Wavelet("gaussian", gaussian, False, lambda d: 1.0, lambda d: math.exp(-0.5)),
}


def get_wavelet(name: str) -> Wavelet:
    key = name.lower().replace("-", "_")
    if key in ("mexhat", "ricker"):
        key = "mexican_hat"
    if key not in WAVELETS:
        raise ValueError(f"unknown wavelet {name!r}; available: {sorted(WAVELETS)}")
    return WAVELETS[key]
