"""Benchmark objectives, written for maximisation (``value = -f`` for minimisation problems)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class Problem:
    name: str
    fn: Callable[[np.ndarray], np.ndarray]
    bounds: np.ndarray
    optimum: float
    budget: int
    stationary: bool

    @property
    def dim(self) -> int:
        return self.bounds.shape[0]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.fn(np.atleast_2d(x))


def gramacy_lee(x: np.ndarray) -> np.ndarray:
    """Gramacy & Lee (2012): fast oscillation for small ``x``, a smooth quartic elsewhere."""
    x = x[:, 0]
    return -(np.sin(10 * np.pi * x) / (2 * x) + (x - 1) ** 4)


def gramacy_2d(x: np.ndarray) -> np.ndarray:
    """``x1 exp(-x1^2 - x2^2)`` on ``[-2, 6]^2``: all structure in one corner, flat elsewhere."""
    return -(x[:, 0] * np.exp(-(x[:, 0] ** 2) - x[:, 1] ** 2))


def branin(x: np.ndarray) -> np.ndarray:
    a, b, c, r, s, t = 1.0, 5.1 / (4 * np.pi**2), 5 / np.pi, 6.0, 10.0, 1 / (8 * np.pi)
    x1, x2 = x[:, 0], x[:, 1]
    return -(a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s)


BUMP_CENTRE = np.array([3.8, 3.0])
BUMP_WIDTH = 0.3
BUMP_DEPTH = 5.0


def bumped_branin(x: np.ndarray) -> np.ndarray:
    """Branin plus a narrow well (width 2% of the range) on the flank of the minimum at ``(pi, 2.275)``.

    The global optimum (about -1.68) sits 0.9 away from a smooth Branin basin (0.398): a surrogate
    must resolve a fine-scale feature inside an otherwise smooth landscape.
    """
    r2 = ((x - BUMP_CENTRE) ** 2).sum(1) / BUMP_WIDTH**2
    return branin(x) + BUMP_DEPTH * np.exp(-0.5 * r2)


_H_A = np.array(
    [[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]]
)
_H_P = 1e-4 * np.array(
    [
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381],
    ]
)
_H_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])


def hartmann6(x: np.ndarray) -> np.ndarray:
    inner = (_H_A[None] * (x[:, None, :] - _H_P[None]) ** 2).sum(-1)
    return (_H_ALPHA * np.exp(-inner)).sum(-1)


def levy(x: np.ndarray) -> np.ndarray:
    w = 1 + (x - 1) / 4
    head = np.sin(np.pi * w[:, 0]) ** 2
    mid = ((w[:, :-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:, :-1] + 1) ** 2)).sum(1)
    tail = (w[:, -1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[:, -1]) ** 2)
    return -(head + mid + tail)


def gramacy_embedded(x: np.ndarray) -> np.ndarray:
    """``gramacy_2d`` on the first two of ten coordinates; the rest are inactive."""
    return gramacy_2d(x[:, :2])


def _bumped_branin_optimum() -> float:
    best = -np.inf
    for x0 in BUMP_CENTRE + 0.2 * np.random.default_rng(0).standard_normal((20, 2)):
        res = minimize(lambda z: -bumped_branin(z[None])[0], x0, method="L-BFGS-B", bounds=[(-5, 10), (0, 15)])
        best = max(best, -res.fun)
    return float(best)


def _box(lo: float, hi: float, d: int) -> np.ndarray:
    return np.tile([lo, hi], (d, 1)).astype(np.float64)


PROBLEMS: dict[str, Problem] = {
    p.name: p
    for p in [
        Problem("gramacy_lee", gramacy_lee, np.array([[0.5, 2.5]]), 0.8690111349894997, 40, False),
        Problem("gramacy_2d", gramacy_2d, _box(-2, 6, 2), np.exp(-0.5) / np.sqrt(2), 80, False),
        Problem(
            "bumped_branin", bumped_branin, np.array([[-5.0, 10.0], [0.0, 15.0]]), _bumped_branin_optimum(), 100, False
        ),
        Problem("gramacy_10d", gramacy_embedded, _box(-2, 6, 10), np.exp(-0.5) / np.sqrt(2), 150, False),
        Problem("branin", branin, np.array([[-5.0, 10.0], [0.0, 15.0]]), -0.39788735772973816, 80, True),
        Problem("hartmann6", hartmann6, _box(0, 1, 6), 3.3223680114155147, 150, True),
        Problem("levy10", levy, _box(-10, 10, 10), 0.0, 200, True),
    ]
}
