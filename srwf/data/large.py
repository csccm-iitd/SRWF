"""Large-scale UCI regression benchmarks of the scalable-GP literature.

Data and splits are those of the ``uci_datasets`` collection (Evans; used by Wang et al., 2019
and Lin et al., 2023, 2024): the target is the last column and ``test_mask.csv.gz`` defines ten
fixed 90/10 train/test splits. Files are downloaded once and cached as ``.npy`` arrays.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np

from .uci import Split

BASE_URL = "https://github.com/treforevans/uci_datasets/raw/master/uci_datasets"

DATASETS: dict[str, tuple[int, int]] = {
    "pol": (15000, 26),
    "elevators": (16599, 18),
    "bike": (17379, 17),
    "protein": (45730, 9),
    "keggdirected": (48827, 20),
    "3droad": (434874, 3),
    "song": (515345, 90),
    "buzz": (583250, 77),
    "houseelectric": (2049280, 11),
}
_PARTS = {"song": ("data.csv.gz", "data1.csv.gz")}


def data_root() -> Path:
    root = Path(os.environ.get("SRWF_DATA", Path(__file__).resolve().parents[2] / "data"))
    return root / "large"


def _fetch(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(url, timeout=600) as r:
            tmp.write_bytes(r.read())
        tmp.rename(path)
    return path


def _cached(path: Path, parse) -> np.ndarray:
    npy = path.with_suffix(".npy")
    if not npy.exists():
        np.save(npy, parse())
    return np.load(npy, mmap_mode="r")


def load(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``(N, d + 1)`` float64 data matrix (target last) and the ``(N, 10)`` test mask."""
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; available: {tuple(DATASETS)}")
    root = data_root() / name
    parts = [_fetch(f"{BASE_URL}/{name}/{f}", root / f) for f in _PARTS.get(name, ("data.csv.gz",))]
    mask_gz = _fetch(f"{BASE_URL}/{name}/test_mask.csv.gz", root / "test_mask.csv.gz")
    xy = _cached(root / "data", lambda: np.concatenate([np.loadtxt(p, delimiter=",", ndmin=2) for p in parts]))
    mask = _cached(root / "test_mask", lambda: np.loadtxt(mask_gz, delimiter=",", dtype=bool, ndmin=2))
    if xy.shape[0] != mask.shape[0]:
        raise ValueError(f"{name}: {xy.shape[0]} rows but {mask.shape[0]} mask rows")
    return xy, mask


def split(name: str, k: int) -> Split:
    """Split ``k`` (``0 <= k < 10``) of the fixed 90/10 partitions."""
    xy, mask = load(name)
    te = np.asarray(mask[:, k])
    xy = np.asarray(xy)
    return Split(xy[~te, :-1], xy[~te, -1], xy[te, :-1], xy[te, -1])


def n_splits(name: str) -> int:
    return int(load(name)[1].shape[1])
