"""Validation objectives on S^2 for designing the Bayesian-optimisation method.

Every design decision of ``bo.py`` (surrogate, acquisition, optimiser settings) is selected on
these functions; the benchmark target ``srwf.data.sphere.irregular_bo`` is not among them and is
evaluated once with the frozen configuration. The set follows the construction of the benchmark,
real spherical harmonics with colatitude and longitude swapped, but with other harmonics, other
singular locations (random rotations) and other kinds of irregularity:

    fan-l-m   ``Y_l^m`` (angles swapped) times ``(1 + z)(1 - colatitude)``: a point singularity at
              a random location where the function takes every value of its range, the infimum
              approached along one or more meridians
    kink-k    the regression target of the benchmark and its negative, randomly rotated: minima
              on lines where the function is continuous but not differentiable
    smooth-k  a random sum of real spherical harmonics of degree at most three
    well-k    a smooth background with a narrow Gaussian well (half-width 6 and 10 degrees)

Minima are located numerically: the best points of a 400k-point lattice are refined by a
ring search with a geometrically shrinking radius, which converges to a singular infimum as
well as to a smooth minimum.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import lpmv

from srwf.data.sphere import irregular
from srwf.geometry import fibonacci_sphere


@dataclass(frozen=True)
class Objective:
    name: str
    f: Callable[[np.ndarray], np.ndarray]
    f_min: float
    scale: float

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.f(np.asarray(x, dtype=np.float64))


def rotation(seed: int) -> np.ndarray:
    """A uniformly random rotation matrix."""
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q = q * np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q


def real_harmonic(l: int, m: int, polar: np.ndarray, azimuth: np.ndarray) -> np.ndarray:
    """Real spherical harmonic ``sqrt(2) N_lm P_l^m(cos polar) cos(m azimuth)`` for ``m >= 1``."""
    norm = math.sqrt((2 * l + 1) / (4 * math.pi) * math.factorial(l - m) / math.factorial(l + m))
    return math.sqrt(2.0) * norm * lpmv(m, l, np.cos(polar)) * np.cos(m * azimuth)


def _angles(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.arccos(np.clip(x[..., 2], -1.0, 1.0)), np.mod(np.arctan2(x[..., 1], x[..., 0]), 2.0 * np.pi)


def fan(l: int, m: int, rot: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    def f(x: np.ndarray) -> np.ndarray:
        x = x @ rot
        colat, lon = _angles(x)
        return real_harmonic(l, m, lon, colat) * (x[..., 2] + 1.0) * (1.0 - colat)

    return f


def kink(rot: np.ndarray, sign: float) -> Callable[[np.ndarray], np.ndarray]:
    def f(x: np.ndarray) -> np.ndarray:
        return sign * irregular(x @ rot)

    return f


def smooth(seed: int, max_degree: int = 3) -> Callable[[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    terms = [(l, m, rng.standard_normal()) for l in range(1, max_degree + 1) for m in range(1, l + 1)]
    rot = rotation(seed + 1)

    def f(x: np.ndarray) -> np.ndarray:
        colat, lon = _angles(x @ rot)
        return sum(c * real_harmonic(l, m, colat, lon) for l, m, c in terms) / math.sqrt(len(terms))

    return f


def well(seed: int, half_width_deg: float, depth: float = 0.6) -> Callable[[np.ndarray], np.ndarray]:
    background = smooth(seed, max_degree=2)
    centre = rotation(seed + 2)[:, 2]
    width = math.radians(half_width_deg)

    def f(x: np.ndarray) -> np.ndarray:
        angle = np.arccos(np.clip(x @ centre, -1.0, 1.0))
        return 0.3 * background(x) - depth * np.exp(-0.5 * (angle / width) ** 2)

    return f


def _refine(
    f: Callable, x0: np.ndarray, radius: float = 0.05, floor: float = 1e-10, n_ring: int = 128, moves: int = 1000
) -> tuple[np.ndarray, float]:
    """Descend from ``x0`` by moving to the best point of a ring of geodesic radius ``radius``
    around the current point, halving the radius whenever the ring brings no improvement."""
    x, fx = x0 / np.linalg.norm(x0), float(f(x0[None])[0])
    angle = np.linspace(0.0, 2.0 * np.pi, n_ring, endpoint=False)
    while radius > floor:
        for _ in range(moves):
            u = np.cross(x, [0.0, 0.0, 1.0] if abs(x[2]) < 0.9 else [1.0, 0.0, 0.0])
            u /= np.linalg.norm(u)
            v = np.cross(x, u)
            cand = x + radius * (np.cos(angle)[:, None] * u + np.sin(angle)[:, None] * v)
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            vals = f(cand)
            k = int(np.argmin(vals))
            if not vals[k] < fx:
                break
            x, fx = cand[k], float(vals[k])
        radius *= 0.5
    return x, fx


def locate_minimum(f: Callable, n_lattice: int = 400_000, n_refine: int = 20) -> tuple[float, float]:
    """Numerical infimum of ``f`` over the sphere and the standard deviation of ``f`` on a lattice."""
    x = fibonacci_sphere(n_lattice)
    vals = f(x)
    best = float("inf")
    for i in np.argsort(vals)[:n_refine]:
        _, fx = _refine(f, x[i])
        best = min(best, fx)
    return best, float(vals.std())


def validation_objectives(cache: Path | None = None) -> dict[str, Objective]:
    """The twelve validation objectives, keyed by name; ``cache`` stores the located minima."""
    specs: list[tuple[str, Callable]] = []
    for k, (l, m) in enumerate([(2, 1), (3, 1), (3, 3), (4, 2), (4, 3), (5, 3)]):
        specs.append((f"fan-{l}-{m}", fan(l, m, rotation(100 + k))))
    specs.append(("kink-0", kink(rotation(200), 1.0)))
    specs.append(("kink-1", kink(rotation(201), -1.0)))
    specs.append(("smooth-0", smooth(300)))
    specs.append(("smooth-1", smooth(301)))
    specs.append(("well-0", well(400, 6.0)))
    specs.append(("well-1", well(401, 10.0)))
    minima = json.loads(cache.read_text()) if cache is not None and cache.exists() else {}
    missing = [(name, f) for name, f in specs if name not in minima]
    for name, f in missing:
        minima[name] = dict(zip(("f_min", "scale"), locate_minimum(f), strict=True))
    if cache is not None and missing:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(minima, indent=1))
    return {name: Objective(name, f, minima[name]["f_min"], minima[name]["scale"]) for name, f in specs}
