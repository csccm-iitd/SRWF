from __future__ import annotations

import numpy as np

STEP_BREAKS = (0.2, 0.4, 0.6, 0.8)
STEP_LEVELS = (-1.0, 1.2, -0.25, 1.2, 0.25)


def multi_step(x: np.ndarray) -> np.ndarray:
    """Piecewise-constant multi-step function on [0, 1]."""
    return np.select([x < b for b in STEP_BREAKS], STEP_LEVELS[:-1], default=STEP_LEVELS[-1])


def multi_step_dataset(n_train: int = 4200, n_test: int = 1800, noise: float = 0.03, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n_train + n_test)
    y = multi_step(x) + noise * rng.standard_normal(x.shape)
    perm = rng.permutation(x.size)
    tr, te = perm[:n_train], perm[n_train:]
    return x[tr, None], y[tr], x[te, None], y[te]


def burst(x: np.ndarray) -> np.ndarray:
    """Smooth background with one localised high-frequency burst around ``x = 0.62``."""
    smooth = 0.55 * np.sin(2 * np.pi * 1.1 * x + 0.4) - 0.45 * np.exp(-(((x - 0.33) / 0.12) ** 2))
    envelope = np.exp(-(((x - 0.62) / 0.035) ** 2))
    return smooth + 1.3 * envelope * np.sin(2 * np.pi * 38.0 * (x - 0.62))


_BLOCK_T = np.array([0.1, 0.13, 0.15, 0.23, 0.25, 0.40, 0.44, 0.65, 0.76, 0.78, 0.81])
_BLOCK_H = np.array([4.0, -5.0, 3.0, -4.0, 5.0, -4.2, 2.1, 4.3, -3.1, 2.1, -4.2])
_BUMP_T = _BLOCK_T
_BUMP_H = np.array([4.0, 5.0, 3.0, 4.0, 5.0, 4.2, 2.1, 4.3, 3.1, 5.1, 4.2])
_BUMP_W = np.array([0.005, 0.005, 0.006, 0.01, 0.01, 0.03, 0.01, 0.01, 0.005, 0.008, 0.005])


def blocks(x: np.ndarray) -> np.ndarray:
    """Donoho & Johnstone (1994) 'Blocks', rescaled to unit standard deviation (signal-to-noise ratio 7 at noise 1/7)."""
    f = (_BLOCK_H * (1 + np.sign(x[:, None] - _BLOCK_T)) / 2).sum(1)
    return f / 1.914


def bumps(x: np.ndarray) -> np.ndarray:
    """Donoho & Johnstone (1994) 'Bumps', rescaled."""
    f = (_BUMP_H * (1 + np.abs((x[:, None] - _BUMP_T) / _BUMP_W)) ** -4).sum(1)
    return f / 0.665


def heavisine(x: np.ndarray) -> np.ndarray:
    """Donoho & Johnstone (1994) 'HeaviSine', rescaled."""
    return (4 * np.sin(4 * np.pi * x) - np.sign(x - 0.3) - np.sign(0.72 - x)) / 3.0


def doppler(x: np.ndarray) -> np.ndarray:
    """Donoho & Johnstone (1994) 'Doppler', rescaled."""
    return np.sqrt(x * (1 - x)) * np.sin(2.1 * np.pi / (x + 0.05)) / 0.29


FUNCTIONS = {
    "multi_step": (multi_step, 0.03),
    "burst": (burst, 0.04),
    "blocks": (blocks, 1 / 7),
    "bumps": (bumps, 1 / 7),
    "heavisine": (heavisine, 1 / 7),
    "doppler": (doppler, 1 / 7),
}


def sample(name: str, n_train: int, n_test: int = 2000, seed: int = 0):
    """Uniform design of ``n_train`` noisy observations and a dense noisy test grid."""
    f, noise = FUNCTIONS[name]
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.0, 1.0, n_train))
    x_test = np.linspace(0.0, 1.0, n_test)
    return (
        x[:, None],
        f(x) + noise * rng.standard_normal(n_train),
        x_test[:, None],
        f(x_test) + noise * rng.standard_normal(n_test),
    )
