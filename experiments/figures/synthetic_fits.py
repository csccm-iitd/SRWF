"""Predictive fits on the multi-step function: every model on the same data.

python experiments/figures/synthetic_fits.py --results results/synthetic --out assets/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from srwf import plotting
from srwf.data.synthetic import multi_step

ORDER = ["rwf", "exact", "svgp", "rff", "dgp"]


def fits(results: Path, models: list[str], width: float, height: float, rmse_label: bool = True):
    """One panel per model: training data, true function, mean and two-sigma band."""
    curves = np.load(results / "multi_step_curves.npz")
    rows = json.loads((results / "multi_step.json").read_text())
    score = {m: np.mean([r["rmse"] for r in rows if r["model"] == m]) for m in models}
    grid, xtr, ytr = curves["grid"], curves["x_train"], curves["y_train"]
    sub = np.random.default_rng(0).choice(len(xtr), 600, replace=False)

    fig, axes = plt.subplots(1, len(models), figsize=(width, height), sharey=True, gridspec_kw=dict(wspace=0.1))
    for ax, m in zip(axes, models, strict=True):
        mean, std = curves[m]
        ax.scatter(xtr[sub], ytr[sub], s=1, color=plotting.INK_MUTED, alpha=0.35, rasterized=True)
        plotting.truth(ax, grid, multi_step(grid))
        plotting.band(ax, grid, mean, std, plotting.color(m))
        ax.set_title(plotting.LABELS[m])
        if rmse_label:
            ax.text(0.97, 0.04, f"RMSE {score[m]:.3f}", transform=ax.transAxes, ha="right", fontsize=6.5)
        ax.set_xlim(0, 1), ax.set_ylim(-1.6, 1.8), ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
        ax.set_xlabel("$x$")
    axes[0].set_ylabel("$y$")
    return fig, axes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results/synthetic"))
    ap.add_argument("--out", type=Path, default=Path("assets/figures"))
    args = ap.parse_args()
    plotting.apply()
    fig, _ = fits(args.results, ORDER, plotting.FULL_WIDTH, 1.55)
    plotting.save(fig, args.out / "multi_step_fits")


if __name__ == "__main__":
    main()
