"""Embeddings of the sphere S^2 in R^3.

A positive-definite kernel on R^3 restricted to S^2 is positive definite on S^2, so wavelet
features evaluated on the embedded coordinates give a valid non-stationary kernel on the
sphere; distances are chordal, ``||x - x'|| = 2 sin(theta / 2)`` for great-circle angle ``theta``.
"""

from __future__ import annotations

import numpy as np


def lonlat_to_xyz(lon_deg, lat_deg) -> np.ndarray:
    lon, lat = np.deg2rad(np.asarray(lon_deg, dtype=np.float64)), np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def fibonacci_sphere(n: int) -> np.ndarray:
    """``n`` near-uniform points on S^2 (golden-angle spiral)."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(1.0 - z**2)
    phi = np.pi * (1.0 + 5.0**0.5) * i
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=-1)
