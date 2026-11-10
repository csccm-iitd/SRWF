"""Shared pieces of the climate experiments: models, area-weighted scores, timing."""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import numpy as np
import torch
from scipy.stats import norm

from srwf import RFFGP, RWFGP, MixtureGP
from srwf.baselines import SVGP, ExactGP
from srwf.data import era5
from srwf.geometry import fibonacci_sphere
from srwf.model import Prediction

Z95 = float(norm.ppf(0.975))


def weighted_scores(y, mean, std, w=None) -> dict[str, float]:
    """RMSE, MAE, NLL, CRPS, 95% coverage and 95% interval score, weighted by ``w`` (area)."""
    y, mean, std = (np.asarray(a, dtype=np.float64).ravel() for a in (y, mean, std))
    w = np.full(y.shape, 1.0 / y.size) if w is None else np.asarray(w, dtype=np.float64) / np.sum(w)
    std = np.maximum(std, 1e-9)
    e = y - mean
    z = e / std
    nll = 0.5 * math.log(2 * math.pi) + np.log(std) + 0.5 * z**2
    crps = std * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / math.sqrt(math.pi))
    width = 2 * Z95 * std
    excess = np.maximum(np.abs(e) - Z95 * std, 0.0)
    return {
        "rmse": float(np.sqrt(np.sum(w * e**2))),
        "mae": float(np.sum(w * np.abs(e))),
        "nll": float(np.sum(w * nll)),
        "crps": float(np.sum(w * crps)),
        "cov95": float(np.sum(w * (np.abs(e) <= Z95 * std))),
        "int95": float(np.sum(w * (width + 2 / 0.05 * excess))),
    }


def sphere_centres(n: int, grid: era5.Grid | None = None, covariates: bool = False) -> np.ndarray:
    """Candidate wavelet centres: a Fibonacci lattice on S^2, with the covariates of the nearest
    grid cell appended when the inputs carry covariates."""
    c = fibonacci_sphere(n)
    if not covariates:
        return c
    if grid is None:
        raise ValueError("covariate centres need the grid")
    nearest = np.argmax(c @ grid.xyz.T, axis=1)
    return np.column_stack([c, grid.inputs(nearest, covariates=True)[:, 3:]])


def build_model(name: str, cfg: dict, seed: int, device: str | None = None):
    """``name`` in {rwf, rff, svgp, exact}; ``cfg`` holds the keyword arguments of the model.

    For the feature models, ``members: K`` builds the mixture of ``K`` independent feature draws.
    """
    cfg = dict(cfg)
    members = int(cfg.pop("members", 1))
    if name in ("rwf", "rff"):
        cls = RWFGP if name == "rwf" else RFFGP
        if members > 1:
            return MixtureGP.of(cls, members=members, seed=seed, device=device, **cfg)
        return cls(seed=seed, device=device, **cfg)
    if name == "svgp":
        return SVGP(seed=seed, device=device, **cfg)
    if name == "exact":
        torch.manual_seed(seed)
        return ExactGP(device=device, **cfg)
    raise ValueError(f"unknown model {name!r}")


def timed(fn: Callable[[], Prediction]) -> tuple[Prediction, float]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def fit_predict(model, x_train, y_train, x_test) -> tuple[Prediction, float]:
    return timed(lambda: model.fit(x_train, y_train).predict(x_test))
