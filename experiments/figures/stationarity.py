"""What non-stationarity means: covariance matrices, kernel slices and prior samples, plus real data.

Rows: a stationary kernel and a non-stationary random-wavelet kernel. Columns: the covariance
matrix K(x, x'), the slices k(x0, .) at three locations, and prior samples. A final panel shows the
day-to-day variability of ERA5 near-surface temperature, which changes about tenfold over the globe.

    python experiments/figures/stationarity.py --out assets/figures
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

from srwf import plotting
from srwf.features import RandomFourierFeatures, RandomWaveletFeatures

ANCHORS = (0.15, 0.5, 0.85)
LATITUDES = [r"$60^{\circ}\mathrm{S}$", r"$0^{\circ}$", r"$60^{\circ}\mathrm{N}$"]
SEQ = {
    0: LinearSegmentedColormap.from_list("seq_b", ["#ffffff00", "#bcd6f6", plotting.BLUE, "#12233d"]),
    1: LinearSegmentedColormap.from_list("seq_o", ["#ffffff00", "#f8c3a6", plotting.ORANGE, "#5a1e08"]),
}


def stationary(x: torch.Tensor) -> torch.Tensor:
    f = RandomFourierFeatures(1, 8000, ard=False, seed=0)
    with torch.no_grad():
        f.raw_lengthscale.fill_(math.log(math.expm1(0.06)))
        return f(x)


def non_stationary(x: torch.Tensor) -> torch.Tensor:
    f = RandomWaveletFeatures(1, 20000, levels=9, growth=1.0, anchors=1, seed=0)
    f.initialize(torch.linspace(-0.1, 1.1, 400).unsqueeze(1), centres="box", base_scale=0.4)
    with torch.no_grad():
        f.block_rms.fill_(1.0)
        f.level_log_weight.zero_()
        t = f.translations[:, 0].double()
        alpha = 0.25 + 1.25 * t.clamp(0, 1)
        f.anchor_basis[:, 0] = ((1 - 2 * alpha) * f.level_index.double() * math.log(2)).float()
        f.anchor_log_weight.fill_(1.0)
        return f(x)


def temperature_variability() -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    try:
        from srwf.data import era5
    except ImportError:
        return None
    try:
        grid = era5.load_grid()
        temp = era5.load_variable("temperature")
    except (FileNotFoundError, OSError):
        return None
    year = slice(era5.step_index("2004-01-01T00"), era5.step_index("2005-01-01T00"))
    daily = np.asarray(temp[year], dtype=np.float64)
    diffs = np.diff(daily[::4], axis=0)
    return grid.lon, grid.lat, diffs.std(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("assets/figures"))
    args = ap.parse_args()
    plotting.apply()
    x = torch.linspace(0, 1, 500).unsqueeze(1)
    xs = x[:, 0].numpy()
    gen = torch.Generator().manual_seed(3)
    rows = []
    for name, feats in (("stationary", stationary), ("non-stationary", non_stationary)):
        z = feats(x).numpy()
        k = z @ z.T
        k = k / np.sqrt(np.outer(np.diag(k), np.diag(k)))
        w = torch.randn(z.shape[1], 3, generator=gen).numpy()
        s = z @ w
        rows.append((name, k, s / s.std()))

    era = temperature_variability()
    ncols = 4 if era is not None else 3
    fig = plt.figure(figsize=(plotting.FULL_WIDTH, 2.5))
    gs = fig.add_gridspec(2, ncols, width_ratios=[1, 1.35, 1.6] + ([1.9] if era else []), wspace=0.36, hspace=0.25)
    for r, (name, k, s) in enumerate(rows):
        c = plotting.BLUE if r == 0 else plotting.ORANGE
        ax = fig.add_subplot(gs[r, 0])
        ax.imshow(k, cmap=SEQ[r], vmin=0, vmax=1, origin="lower", extent=(0, 1, 0, 1), aspect="auto")
        ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
        ax.set_ylabel(name)
        if r == 0:
            ax.set_title("covariance $k(x, x')$")
        ax = fig.add_subplot(gs[r, 1])
        for a in ANCHORS:
            i = int(a * (len(xs) - 1))
            ax.plot(xs, k[i], color=c)
            ax.plot([xs[i]], [1.0], "o", ms=2.5, color=c)
        ax.set_ylim(-0.4, 1.15), ax.set_yticks([0, 1]), ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"] if r else [])
        if r == 0:
            ax.set_title(r"slices $k(x_0, \cdot)$")
        ax = fig.add_subplot(gs[r, 2])
        for j, alpha in enumerate((1.0, 0.6, 0.35)):
            ax.plot(xs, s[:, j] + 3.2 * j, color=c, lw=0.6, alpha=alpha)
        ax.set_yticks([]), ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"] if r else [])
        if r == 0:
            ax.set_title("prior samples")

    if era is not None:
        lon, lat, sd = era
        ax = fig.add_subplot(gs[:, 3])
        im = ax.pcolormesh(lon, lat, sd, cmap="magma", shading="auto", rasterized=True, vmax=np.percentile(sd, 99.5))
        ax.set_xticks([]), ax.grid(False)
        ax.set_yticks([-60, 0, 60], LATITUDES)
        ax.set_title("ERA5 temperature, 2004")
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.06, pad=0.05)
        cb.outline.set_linewidth(0.5)
        cb.ax.tick_params(labelsize=6.5)
        cb.set_label("s.d. of daily change (K)", fontsize=7)
    plotting.save(fig, args.out / "stationarity")


if __name__ == "__main__":
    main()
