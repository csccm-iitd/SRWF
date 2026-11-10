"""Random Fourier vs random wavelet features in 2-D: individual features and induced kernels.

Top row: three random features of each family. Bottom row: the induced kernel ``k(x, .)`` at
three anchor points ``x``; RFF gives the same shape everywhere (stationary), RWF adapts it.

    python experiments/figures/features_2d.py --out assets/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from srwf import plotting
from srwf.features import RandomFourierFeatures, RandomWaveletFeatures

CMAP = LinearSegmentedColormap.from_list("div", [plotting.BLUE, "#bcd6f6", "#f7f6f2", "#f8c3a6", plotting.ORANGE])
ANCHORS = np.array([[-0.55, -0.5], [0.0, 0.05], [0.5, 0.5]])
ROUGH_CENTRE = torch.tensor([0.45, 0.45])


def grid(n: int = 220) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    g = np.linspace(-1, 1, n)
    xx, yy = np.meshgrid(g, g)
    return xx, yy, torch.tensor(np.stack([xx.ravel(), yy.ravel()], 1), dtype=torch.float32)


def wavelet_features(pts: torch.Tensor) -> RandomWaveletFeatures:
    f = RandomWaveletFeatures(2, 4000, levels=4, growth=2.0, anchors=1, ard=False, seed=1)
    f.initialize(pts, centres="box", base_scale=0.5)
    with torch.no_grad():
        dist = (f.translations - ROUGH_CENTRE).norm(dim=1)
        octave = f.level_index.float() / (f.levels - 1)
        local = torch.sigmoid((0.55 - dist) / 0.1)
        f.anchor_basis[:, 0] = torch.log(1e-4 + (1 - octave) ** 3 + 12.0 * octave**2 * local)
        f.anchor_log_weight.fill_(1.0)
        f.level_log_weight.zero_()
    return f


def fourier_features() -> RandomFourierFeatures:
    f = RandomFourierFeatures(2, 4000, ard=False, seed=1)
    with torch.no_grad():
        f.raw_lengthscale.fill_(float(np.log(np.expm1(0.25))))
    return f


def show(ax, xx, yy, v, marker=None) -> None:
    lim = np.abs(v).max() + 1e-12
    ax.pcolormesh(xx, yy, v, cmap=CMAP, norm=TwoSlopeNorm(0.0, -lim, lim), shading="gouraud", rasterized=True)
    if marker is not None:
        ax.plot(*marker, "o", ms=3, mfc=plotting.INK, mec="white", mew=0.7)
    ax.set_xticks([]), ax.set_yticks([]), ax.set_aspect("equal"), ax.grid(False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("assets/figures"))
    args = ap.parse_args()
    plotting.apply()
    xx, yy, pts = grid()
    rwf, rff = wavelet_features(pts), fourier_features()
    with torch.no_grad():
        zw, zf = rwf(pts).numpy(), rff(pts).numpy()
        aw = rwf(torch.tensor(ANCHORS, dtype=torch.float32)).numpy()
        af = rff(torch.tensor(ANCHORS, dtype=torch.float32)).numpy()
    rng = np.random.default_rng(4)
    pick_w = [int(rng.choice(np.where(rwf.level_index.numpy() == j)[0])) for j in (0, 1, 2)]
    pick_f = rng.choice(zf.shape[1], 3, replace=False)

    fig, axes = plt.subplots(
        2,
        7,
        figsize=(plotting.FULL_WIDTH, 2.2),
        gridspec_kw=dict(width_ratios=[1, 1, 1, 0.2, 1, 1, 1], wspace=0.06, hspace=0.08),
    )
    for ax in axes[:, 3]:
        ax.axis("off")
    for i, j in enumerate(pick_f):
        show(axes[0, i], xx, yy, zf[:, j].reshape(xx.shape))
    for i, j in enumerate(pick_w):
        show(axes[0, 4 + i], xx, yy, zw[:, j].reshape(xx.shape))
    for i, a in enumerate(ANCHORS):
        show(axes[1, i], xx, yy, (zf @ af[i]).reshape(xx.shape), marker=a)
        show(axes[1, 4 + i], xx, yy, (zw @ aw[i]).reshape(xx.shape), marker=a)
        axes[1, 4 + i].add_patch(
            plt.Circle(ROUGH_CENTRE.numpy(), 0.55, fill=False, ls=(0, (2, 2)), lw=0.6, color=plotting.INK)
        )
    axes[0, 1].set_title("Random Fourier features")
    axes[0, 5].set_title("Random wavelet features")
    axes[0, 0].set_ylabel("features")
    axes[1, 0].set_ylabel(r"kernel $k(x, \cdot)$")
    plotting.save(fig, args.out / "features_2d")


if __name__ == "__main__":
    main()
