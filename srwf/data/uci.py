"""UCI regression benchmarks with the standard 90/10 splits used in the GP literature.

Energy, Concrete, Kin8nm, Naval and Protein use the public splits of Hernandez-Lobato & Adams
(2015) / Gal & Ghahramani (2016). Airfoil is not part of that collection; it is read from the
UCI archive and split with seeded random permutations.
"""

from __future__ import annotations

import io
import os
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SPLITS_URL = "https://raw.githubusercontent.com/yaringal/DropoutUncertaintyExps/master/UCI_Datasets"
UCI_URL = "https://archive.ics.uci.edu/static/public"

STANDARD = {
    "energy": "energy",
    "concrete": "concrete",
    "kin8nm": "kin8nm",
    "naval": "naval-propulsion-plant",
    "protein": "protein-tertiary-structure",
}
DATASETS = (*STANDARD, "airfoil")


def data_root() -> Path:
    root = Path(os.environ.get("SRWF_DATA", Path(__file__).resolve().parents[2] / "data"))
    return root / "uci"


@dataclass
class Split:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def _fetch(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as r:
            path.write_bytes(r.read())
    return path


def _standard(name: str) -> tuple[np.ndarray, list[np.ndarray]]:
    key = STANDARD[name]
    root = data_root() / name
    base = f"{SPLITS_URL}/{key}/data"
    data = np.loadtxt(_fetch(f"{base}/data.txt", root / "data.txt"))
    feats = np.loadtxt(_fetch(f"{base}/index_features.txt", root / "index_features.txt")).astype(int)
    target = int(np.loadtxt(_fetch(f"{base}/index_target.txt", root / "index_target.txt")))
    n_splits = int(np.loadtxt(_fetch(f"{base}/n_splits.txt", root / "n_splits.txt")))
    train_idx = [
        np.loadtxt(_fetch(f"{base}/index_train_{k}.txt", root / f"index_train_{k}.txt")).astype(int)
        for k in range(n_splits)
    ]
    xy = np.column_stack([data[:, feats], data[:, target]])
    return xy, train_idx


def _airfoil() -> tuple[np.ndarray, list[np.ndarray]]:
    root = data_root() / "airfoil"
    raw = _fetch(f"{UCI_URL}/291/airfoil+self+noise.zip", root / "airfoil.zip")
    with zipfile.ZipFile(raw) as z:
        name = next(n for n in z.namelist() if n.endswith(".dat"))
        xy = np.loadtxt(io.BytesIO(z.read(name)))
    n = xy.shape[0]
    n_train = int(round(0.9 * n))
    train_idx = [np.sort(np.random.default_rng(k).permutation(n)[:n_train]) for k in range(20)]
    return xy, train_idx


def load(name: str) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return the ``(N, d + 1)`` data matrix (target last) and the list of training index sets."""
    if name == "airfoil":
        return _airfoil()
    if name in STANDARD:
        return _standard(name)
    raise ValueError(f"unknown dataset {name!r}; available: {DATASETS}")


def split(name: str, k: int) -> Split:
    xy, train_idx = load(name)
    tr = train_idx[k % len(train_idx)]
    te = np.setdiff1d(np.arange(xy.shape[0]), tr)
    return Split(xy[tr, :-1], xy[tr, -1], xy[te, :-1], xy[te, -1])


def n_splits(name: str) -> int:
    return len(load(name)[1])
