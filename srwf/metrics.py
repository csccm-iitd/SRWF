from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm


def _arr(*xs):
    return [np.asarray(x, dtype=np.float64).ravel() for x in xs]


def rmse(y, mean) -> float:
    y, mean = _arr(y, mean)
    return float(np.sqrt(np.mean((y - mean) ** 2)))


def mae(y, mean) -> float:
    y, mean = _arr(y, mean)
    return float(np.mean(np.abs(y - mean)))


def gaussian_nll(y, mean, std) -> float:
    y, mean, std = _arr(y, mean, std)
    std = np.maximum(std, 1e-12)
    return float(np.mean(0.5 * math.log(2 * math.pi) + np.log(std) + 0.5 * ((y - mean) / std) ** 2))


def gaussian_crps(y, mean, std) -> float:
    y, mean, std = _arr(y, mean, std)
    std = np.maximum(std, 1e-12)
    z = (y - mean) / std
    return float(np.mean(std * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / math.sqrt(math.pi))))


def coverage(y, mean, std, level: float = 0.95) -> float:
    y, mean, std = _arr(y, mean, std)
    q = norm.ppf(0.5 + level / 2)
    return float(np.mean(np.abs(y - mean) <= q * std))


def interval_score(y, mean, std, level: float = 0.95) -> float:
    """Gneiting & Raftery (2007) interval score of the central ``level`` interval."""
    y, mean, std = _arr(y, mean, std)
    alpha = 1 - level
    q = norm.ppf(1 - alpha / 2)
    lo, hi = mean - q * std, mean + q * std
    return float(np.mean((hi - lo) + 2 / alpha * (lo - y) * (y < lo) + 2 / alpha * (y - hi) * (y > hi)))


def evaluate(y, mean, std, scale: float = 1.0) -> dict[str, float]:
    """Standard scores; ``scale`` maps standardised targets back to original units."""
    y, mean, std = _arr(y, mean, std)
    y, mean, std = y * scale, mean * scale, std * scale
    return {
        "rmse": rmse(y, mean),
        "mae": mae(y, mean),
        "nll": gaussian_nll(y, mean, std),
        "crps": gaussian_crps(y, mean, std),
        "cov95": coverage(y, mean, std),
        "int95": interval_score(y, mean, std),
    }
