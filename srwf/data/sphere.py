"""Irregular benchmark functions on the sphere S^2 of Wyrwal, Krause and Borovitskiy (2025).

Wyrwal, K., Krause, A. and Borovitskiy, V. "Residual Deep Gaussian Processes on Manifolds",
ICLR 2025 (arXiv:2411.00161), Section 4.1, 4.2 and Appendix A.3, A.4. The formulas follow the
released code (github.com/KacperWyrwal/residual-deep-gps), which uses real spherical harmonics
evaluated with colatitude and longitude swapped; that swap makes the functions singular at the
poles and, for the term composed with a quarter-turn roll, along the equator.

Inputs are unit vectors ``x`` of shape ``(n, 3)``.
"""

from __future__ import annotations

import numpy as np

NOISE_STD = 0.01


def fibonacci_lattice(n: int) -> np.ndarray:
    """The ``n``-point Fibonacci lattice of the paper (Appendix A.3, equation 28)."""
    i = np.arange(n)
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    colat = np.arccos(1.0 - 2.0 * (i + 0.5) / n)
    lon = 2.0 * np.pi * i / golden
    return np.stack([np.sin(colat) * np.cos(lon), np.sin(colat) * np.sin(lon), np.cos(colat)], axis=-1)


def _colat_lon(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.arccos(np.clip(x[..., 2], -1.0, 1.0)), np.mod(np.arctan2(x[..., 1], x[..., 0]), 2.0 * np.pi)


def irregular(x: np.ndarray) -> np.ndarray:
    """Regression target ``f*`` of Section 4.1: ``Re Y_2^1 + Re Y_1^1`` (rolled by pi / 2), angles swapped."""
    x = np.asarray(x, dtype=np.float64)
    colat, lon = _colat_lon(x)
    a = -3.0 * np.sqrt(5.0 / (24.0 * np.pi)) * np.cos(lon) * np.abs(np.sin(lon)) * np.cos(colat)
    roll_colat = np.arccos(np.clip(x[..., 1], -1.0, 1.0))
    roll_lon = np.arctan2(-x[..., 2], x[..., 0])
    b = -np.sqrt(3.0 / (8.0 * np.pi)) * np.abs(np.sin(roll_lon)) * np.cos(roll_colat)
    return a + b


def irregular_bo(x: np.ndarray) -> np.ndarray:
    """Bayesian-optimisation target of Section 4.2 (minimised): one global minimum near the north pole."""
    x = np.asarray(x, dtype=np.float64)
    colat, lon = _colat_lon(x)
    y23 = np.sqrt(35.0 / (96.0 * np.pi)) * 3.0 * np.sin(lon) ** 2 * np.cos(lon) * np.cos(2.0 * colat)
    return y23 * (x[..., 2] + 1.0) * (1.0 - colat)


def regression_dataset(n_train: int, seed: int, n_test: int = 5000, noise_std: float = NOISE_STD):
    """Lattice inputs, noisy targets for train and test, and the noise-free test targets."""
    rng = np.random.default_rng(seed)
    x_train, x_test = fibonacci_lattice(n_train), fibonacci_lattice(n_test)
    f_test = irregular(x_test)
    y_train = irregular(x_train) + noise_std * rng.standard_normal(n_train)
    y_test = f_test + noise_std * rng.standard_normal(n_test)
    return x_train, y_train, x_test, y_test, f_test
