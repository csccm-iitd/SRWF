"""Burst-signal fits and error-versus-sample-size curves on the Donoho-Johnstone suite.

python experiments/figures/sample_size.py --results results/synthetic --out assets/figures
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from srwf import plotting
from srwf.data.synthetic import FUNCTIONS

STYLE = {
    "rwf-adaptive": ("RWF, adaptive (ours)", plotting.ORANGE, "-"),
    "rwf": ("RWF (ours)", plotting.ORANGE, (0, (4, 2))),
    "exact-m12": (r"Exact GP, Mat$\acute{\mathrm{e}}$rn-1/2", plotting.GREEN, "-"),
    "exact": ("Exact GP, RBF", plotting.BLUE, "-"),
    "rff": ("RFF", plotting.PURPLE, "-"),
}
FIT_ORDER = ["rwf-adaptive", "exact-m12", "exact", "rff"]
CURVE_FUNCTIONS = ["burst", "doppler", "heavisine", "blocks", "bumps"]


def burst_fits(results: Path, out: Path, n: int) -> None:
    f, _ = FUNCTIONS["burst"]
    fig, axes = plt.subplots(
        1, len(FIT_ORDER), figsize=(plotting.FULL_WIDTH, 1.55), sharey=True, gridspec_kw=dict(wspace=0.1)
    )
    for ax, m in zip(axes, FIT_ORDER, strict=True):
        d = np.load(results / f"fit_burst_{n}_{m}.npz")
        label, c, _ = STYLE[m]
        ax.scatter(d["x_train"], d["y_train"], s=1.5, color=plotting.INK_MUTED, alpha=0.6)
        plotting.truth(ax, d["x"], f(d["x"]))
        plotting.band(ax, d["x"], d["mean"], d["f_std"], c)
        err = np.sqrt(np.mean((d["mean"] - f(d["x"])) ** 2))
        ax.set_title(label.removesuffix(" (ours)"))
        ax.text(0.03, 0.04, f"RMSE {err:.3f}", transform=ax.transAxes, fontsize=6.5)
        ax.set_xlim(0, 1), ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
        ax.set_xlabel("$x$")
    axes[0].set_ylabel("$y$")
    plotting.save(fig, out / "burst_fits")


def curves(results: Path, out: Path) -> None:
    rows = json.loads((results / "sample_size.json").read_text())
    agg: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        agg[(r["function"], r["model"], r["n"])].append(r["rmse_f"])
    fig, axes = plt.subplots(1, len(CURVE_FUNCTIONS), figsize=(plotting.FULL_WIDTH, 1.5), gridspec_kw=dict(wspace=0.38))
    for ax, fn in zip(axes, CURVE_FUNCTIONS, strict=True):
        span = [np.mean(v) for (g, _, _), v in agg.items() if g == fn]
        for m, (label, c, ls) in STYLE.items():
            ns = sorted({n for (g, mm, n) in agg if g == fn and mm == m})
            mu = np.array([np.mean(agg[(fn, m, n)]) for n in ns])
            se = np.array([np.std(agg[(fn, m, n)]) / np.sqrt(len(agg[(fn, m, n)])) for n in ns])
            ax.fill_between(ns, mu - se, mu + se, color=c, alpha=0.25, lw=0)
            ax.plot(ns, mu, color=c, ls=ls, marker="o", ms=2, label=label)
        plotting.log_scale(ax, "x", base=2), plotting.log_scale(ax, "y", plain=max(span) / min(span) < 10)
        ax.set_xticks([128, 512, 2048])
        ax.set_title("HeaviSine" if fn == "heavisine" else fn.capitalize())
        ax.set_xlabel("$N$")
    axes[0].set_ylabel("RMSE to $f$")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=len(STYLE))
    plotting.save(fig, out / "sample_size")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results/synthetic"))
    ap.add_argument("--out", type=Path, default=Path("assets/figures"))
    ap.add_argument("--n", type=int, default=256)
    args = ap.parse_args()
    plotting.apply()
    burst_fits(args.results, args.out, args.n)
    curves(args.results, args.out)


if __name__ == "__main__":
    main()
