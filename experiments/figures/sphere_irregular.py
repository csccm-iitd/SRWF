"""The irregular function on S^2 of Wyrwal, Krause and Borovitskiy (ICLR 2025, arXiv:2411.00161)
on a globe, next to the RWF posterior mean and posterior standard deviation.

The model is the RWF GP chosen on validation data for ``--n-train`` lattice points (seed 0) by
``experiments/sphere/irregular.py``. Writes a static figure and a slowly rotating animation
with transparent backgrounds.

    python experiments/figures/sphere_irregular.py --n-train 400
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sphere"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from animations import rgba_frame, write  # noqa: E402
from irregular import build, candidates  # noqa: E402

from srwf import plotting  # noqa: E402
from srwf.data.sphere import irregular, regression_dataset  # noqa: E402
from srwf.geometry import lonlat_to_xyz  # noqa: E402

FIELD_CMAP, STD_CMAP = "magma", "viridis"
TITLES = ("$f^*$", "RWF posterior mean", "RWF posterior std")


def field(n_train: int, results: Path, cache: Path, device: str, res: float = 1.0) -> dict:
    if cache.exists():
        return dict(np.load(cache))
    runs = json.loads((results / "irregular_rwf.json").read_text())
    name = next(r["candidate"] for r in runs if r["n_train"] == n_train and r["seed"] == 0)
    x, y, *_ = regression_dataset(n_train, 0)
    model = build(candidates()[name], 0, device).fit(x, y)
    lon = np.arange(-180.0, 180.0 + res / 2, res)
    lat = np.arange(-90.0, 90.0 + res / 2, res)
    glon, glat = np.meshgrid(lon, lat)
    xyz = lonlat_to_xyz(glon.ravel(), glat.ravel())
    p = model.predict(xyz, return_f_std=True)
    out = {
        "lon": lon,
        "lat": lat,
        "truth": irregular(xyz).reshape(glon.shape),
        "mean": p.mean.reshape(glon.shape),
        "std": p.f_std.reshape(glon.shape),
        "x_train": x,
        "candidate": np.array(name),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


def draw(fig, d: dict, central_lon: float, colorbars: bool) -> None:
    import cartopy.crs as ccrs

    fig.clf()
    proj = ccrs.Orthographic(central_longitude=central_lon, central_latitude=25)
    flim = float(np.abs(d["truth"]).max())
    norms = (Normalize(-flim, flim), Normalize(-flim, flim), Normalize(0.0, float(np.percentile(d["std"], 99.5))))
    cmaps = (FIELD_CMAP, FIELD_CMAP, STD_CMAP)
    for i, key in enumerate(("truth", "mean", "std")):
        ax = fig.add_subplot(1, 3, i + 1, projection=proj)
        ax.set_global()
        ax.patch.set_alpha(0)
        ax.spines["geo"].set_edgecolor(plotting.INK_MUTED)
        ax.spines["geo"].set_linewidth(0.5)
        mesh = ax.pcolormesh(
            d["lon"],
            d["lat"],
            d[key],
            cmap=cmaps[i],
            norm=norms[i],
            transform=ccrs.PlateCarree(),
            shading="gouraud",
            rasterized=True,
        )
        if colorbars:
            ax.set_title(TITLES[i])
            cb = fig.colorbar(mesh, ax=ax, orientation="horizontal", fraction=0.05, pad=0.04, shrink=0.8)
            cb.outline.set_linewidth(0.4)
            cb.ax.tick_params(width=0.4, length=2)


def static(d: dict, path: Path) -> None:
    fig = plt.figure(figsize=(plotting.FULL_WIDTH, 2.6))
    fig.patch.set_alpha(0)
    draw(fig, d, central_lon=30.0, colorbars=True)
    fig.subplots_adjust(0.01, 0.02, 0.99, 0.92, wspace=0.06)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), transparent=True)
    fig.savefig(path.with_suffix(".png"), dpi=200, transparent=True)
    plt.close(fig)


def animate(d: dict, path: Path, seconds: float = 16.0, fps: int = 15, dpi: int = 60, quality: int = 50) -> None:
    fig = plt.figure(figsize=(9.6, 3.3), dpi=dpi)
    fig.patch.set_alpha(0)
    frames = []
    n = int(round(seconds * fps))
    for k in range(n):
        draw(fig, d, central_lon=-180.0 + 360.0 * k / n, colorbars=False)
        fig.subplots_adjust(0.005, 0.01, 0.995, 0.99, wspace=0.03)
        frames.append(rgba_frame(fig))
    plt.close(fig)
    write(frames, path, fps=fps, quality=quality)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--results", type=Path, default=Path("results/sphere"))
    ap.add_argument("--figures", type=Path, default=Path("assets/figures"))
    ap.add_argument("--docs", type=Path, default=Path("docs/static/img"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--quality", type=int, default=45)
    ap.add_argument("--dpi", type=int, default=85)
    args = ap.parse_args()
    plotting.apply()
    d = field(args.n_train, args.results, args.results / f"sphere_irregular_field_n{args.n_train}.npz", args.device)
    static(d, args.figures / "sphere_irregular")
    animate(d, args.figures / "anim" / "sphere_irregular", quality=args.quality, dpi=args.dpi)
    args.docs.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.figures / "sphere_irregular.png", args.docs / "sphere_irregular.png")
    shutil.copy(args.figures / "anim" / "sphere_irregular.webp", args.docs / "sphere_irregular.webp")


if __name__ == "__main__":
    main()
