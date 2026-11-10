"""Simple regret of Bayesian optimisation on S^2 (Wyrwal, Krause and Borovitskiy, ICLR 2025,
Section 4.2): RWF with expected improvement against the exact Matern GP and the residual deep GP
baselines, median and interquartile range over 15 seeds.

    python experiments/figures/sphere_bo.py --results results/sphere --out assets/figures
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from srwf import plotting

CURVES = {
    "rwf_v2": ("bo_rwf_v2.json", "RWF + EI (ours)", plotting.ORANGE),
    "shallow": ("bo_baselines.json", "Exact Matern GP + EI", plotting.GREEN),
    "deep": ("bo_baselines.json", "Residual deep GP + EI (from step 180)", plotting.BROWN),
}
FLOOR = 1e-4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/sphere")
    ap.add_argument("--out", default="assets/figures")
    ap.add_argument("--docs", default="docs/static/img")
    args = ap.parse_args()
    results = Path(args.results)
    plotting.apply()
    fig, ax = plt.subplots(figsize=(plotting.TEXT_WIDTH * 0.6, 2.1))
    n_seeds = []
    for key, (file, label, c) in CURVES.items():
        d = json.loads((results / file).read_text())[key]
        r = np.maximum(10.0 ** np.array(d["log10_regret_runs"]), FLOOR)
        n_seeds.append(len(d["seeds"]))
        lo, med, hi = np.percentile(r, [25, 50, 75], axis=0)
        steps = np.arange(r.shape[1])
        ax.fill_between(steps, lo, hi, color=c, alpha=0.18, lw=0)
        ax.plot(steps, med, color=c, lw=1.0, label=label, zorder=3 if key == "rwf_v2" else 2)
    plotting.log_scale(ax, "y")
    ax.set_xlim(0, 200)
    ax.set_xlabel("evaluations after the initial design")
    ax.set_ylabel("simple regret")
    handles = [Line2D([], [], color=c, lw=1.0) for _, _, c in CURVES.values()]
    handles.append(Patch(color=plotting.INK_2, alpha=0.18, lw=0))
    labels = [label for _, label, _ in CURVES.values()] + ["interquartile range"]
    ax.legend(handles, labels, loc="lower left", title=f"median of {min(n_seeds)} seeds", title_fontsize=7)
    plotting.save(fig, Path(args.out) / "sphere_bo")
    docs = Path(args.docs)
    if docs.exists():
        shutil.copy(Path(args.out) / "sphere_bo.png", docs / "sphere_bo.png")


if __name__ == "__main__":
    main()
