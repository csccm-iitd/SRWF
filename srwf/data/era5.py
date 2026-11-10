"""ERA5 reanalysis regridded to the T30 Gaussian grid (the LUCIE training data).

Each variable is a float32 array ``(16538, 48, 96)`` stored as ``<root>/<name>.npy``: 6-hourly
fields from 2000-01-01 00 UTC on 48 Gauss-Legendre latitudes (row 0 southernmost) and 96
equiangular longitudes ``3.75 k`` degrees. ``orography.npy`` is static, ``(48, 96)`` in metres.
Cell areas are proportional to the Gauss quadrature weights of the latitude row.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from ..geometry import lonlat_to_xyz

N_LAT, N_LON = 48, 96
STEP_HOURS = 6
T0 = np.datetime64("2000-01-01T00", "h")
VARIABLES = ("temperature", "precipitation", "humidity", "surface_pressure", "u_wind", "v_wind", "tisr")
UNITS = {
    "temperature": "K",
    "precipitation": "log1p(mm)",
    "humidity": "g/kg",
    "surface_pressure": "hPa",
    "u_wind": "m/s",
    "v_wind": "m/s",
    "tisr": "MJ/m2",
}


def data_root() -> Path:
    default = Path(__file__).resolve().parents[2] / "data" / "era5_t30"
    return Path(os.environ.get("SRWF_ERA5", default))


@dataclass(frozen=True)
class Grid:
    """The T30 Gaussian grid; cell arrays are flattened row-major, ``cell = 96 * row + col``."""

    lat: np.ndarray
    lon: np.ndarray
    lat_weights: np.ndarray
    orography: np.ndarray

    @property
    def n_cells(self) -> int:
        return self.lat.size * self.lon.size

    @cached_property
    def cell_lonlat(self) -> np.ndarray:
        lon, lat = np.meshgrid(self.lon, self.lat)
        return np.stack([lon.ravel(), lat.ravel()], axis=1)

    @cached_property
    def xyz(self) -> np.ndarray:
        return lonlat_to_xyz(self.cell_lonlat[:, 0], self.cell_lonlat[:, 1])

    @cached_property
    def area(self) -> np.ndarray:
        """Cell area weights summing to one."""
        w = np.repeat(self.lat_weights, self.lon.size)
        return w / w.sum()

    @cached_property
    def land(self) -> np.ndarray:
        """Land indicator ``orography > 10 m`` (the files carry no land-sea mask)."""
        return (self.orography.ravel() > 10.0).astype(np.float64)

    def inputs(self, cells: np.ndarray | None = None, covariates: bool = False) -> np.ndarray:
        """Sphere embedding ``(x, y, z)``, optionally with orography (km) and the land flag."""
        cells = np.arange(self.n_cells) if cells is None else np.asarray(cells)
        x = self.xyz[cells]
        if covariates:
            x = np.column_stack([x, self.orography.ravel()[cells] / 1000.0, self.land[cells]])
        return x


def gaussian_latitudes(n: int = N_LAT) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre latitudes in degrees (ascending, south first) and quadrature weights."""
    nodes, weights = np.polynomial.legendre.leggauss(n)
    return np.rad2deg(np.arcsin(nodes)), weights


def load_grid(root: str | Path | None = None) -> Grid:
    root = Path(root) if root is not None else data_root()
    lat, w = gaussian_latitudes(N_LAT)
    lon = np.arange(N_LON) * (360.0 / N_LON)
    oro = np.load(root / "orography.npy").reshape(-1, N_LAT, N_LON)[0].astype(np.float64)
    return Grid(lat, lon, w, oro)


def load_variable(name: str, root: str | Path | None = None) -> np.ndarray:
    """Memory-mapped ``(T, 48, 96)`` array of the raw variable."""
    if name not in VARIABLES:
        raise ValueError(f"unknown variable {name!r}; available: {VARIABLES}")
    root = Path(root) if root is not None else data_root()
    return np.load(root / f"{name}.npy", mmap_mode="r")


def transform(name: str, values: np.ndarray) -> np.ndarray:
    """Raw values to modelling units: precipitation ``log1p(1000 P)`` (P in m), humidity in g/kg,
    pressure in hPa, tisr in MJ/m^2, the rest unchanged."""
    v = np.asarray(values, dtype=np.float64)
    if name == "precipitation":
        return np.log1p(1000.0 * np.maximum(v, 0.0))
    if name == "humidity":
        return 1000.0 * v
    if name == "surface_pressure":
        return v / 100.0
    if name == "tisr":
        return v / 1e6
    return v


def field(name: str, step: int, root: str | Path | None = None) -> np.ndarray:
    """One snapshot in modelling units, flattened to ``(4608,)``."""
    return transform(name, load_variable(name, root)[int(step)]).ravel()


def step_index(time: str | np.datetime64) -> int:
    return int((np.datetime64(time, "h") - T0) / np.timedelta64(STEP_HOURS, "h"))


def step_time(index: int | np.ndarray) -> np.ndarray:
    return T0 + np.asarray(index, dtype=np.int64) * np.timedelta64(STEP_HOURS, "h")


def snapshot_steps(start: str, count: int, spacing_days: float) -> np.ndarray:
    """``count`` step indices from ``start``, every ``spacing_days`` days."""
    stride = int(round(spacing_days * 24 / STEP_HOURS))
    return step_index(start) + stride * np.arange(int(count))


def sample_stations(grid: Grid, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Area-weighted sample of ``round(fraction * cells)`` station cells without replacement.

    Returns sorted ``(stations, held_out)`` cell indices.
    """
    n = int(round(fraction * grid.n_cells))
    rng = np.random.default_rng(seed)
    stations = np.sort(rng.choice(grid.n_cells, size=n, replace=False, p=grid.area))
    held_out = np.setdiff1d(np.arange(grid.n_cells), stations)
    return stations, held_out
