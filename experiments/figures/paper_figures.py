"""The multi-step figures of the paper, re-created with the release code.

python experiments/figures/paper_figures.py --results results/synthetic --out assets/figures/paper
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory
from synthetic_fits import ORDER, fits

from srwf import plotting

SCALING_ORDER = ["rwf", "rff", "exact", "svgp", "dgp"]
MARKERS = {"rwf": "o", "rff": "^", "exact": "D", "svgp": "s", "dgp": "v"}


def comparison(results: Path, out: Path) -> None:
    fig, axes = fits(results, ORDER, plotting.FULL_WIDTH, 1.45, rmse_label=False)
    handles = [
        Line2D([], [], color=plotting.INK_2, lw=1.0, label="predictive mean"),
        Patch(color=plotting.INK_2, alpha=0.25, lw=0, label=r"$\pm 2\sigma$"),
        Line2D([], [], color=plotting.INK, lw=0.7, ls=(0, (3, 2)), label="true function"),
        Line2D([], [], color=plotting.INK_MUTED, marker="o", ms=2, lw=0, label="training data"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=4)
    plotting.save(fig, out / "extended_comparison_6models")


def scalability(results: Path, out: Path, budget: float = 600.0) -> None:
    rows = json.loads((results / "scalability.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(plotting.FULL_WIDTH, 1.9), gridspec_kw=dict(wspace=0.38))
    axes[0].axhline(budget, color=plotting.INK_MUTED, lw=0.6, ls=(0, (1, 1.5)))
    axes[0].text(1.1e3, budget * 0.62, "10 min", fontsize=6.5, color=plotting.INK_2)
    top = blended_transform_factory(axes[1].transData, axes[1].transAxes)
    for m in SCALING_ORDER:
        r = sorted((x for x in rows if x["model"] == m), key=lambda x: x["n"])
        ok = [x for x in r if x["status"] == "ok"]
        style = dict(color=plotting.color(m), marker=MARKERS[m], ms=3, lw=1.0)
        n = [x["n"] for x in ok]
        axes[0].plot(n, [x["seconds"] for x in ok], label=plotting.LABELS[m], **style)
        axes[1].plot(n, [x["memory_gb"] for x in ok], **style)
        axes[2].plot(n, [x["rmse"] for x in ok], **style)
        for x in r:
            if x["status"] == "ok":
                continue
            axes[0].plot([x["n"]], [budget], "x", color=plotting.color(m), ms=4.5, mew=1.0, clip_on=False)
            axes[1].plot([x["n"]], [0.97], "x", color=plotting.color(m), ms=4.5, mew=1.0, transform=top, clip_on=False)
    for ax, label, title in zip(
        axes,
        ("fit time (s)", "peak GPU memory (GB)", "test RMSE"),
        ("Training time", "Memory", "Accuracy"),
        strict=True,
    ):
        plotting.log_scale(ax)
        ax.set_xlabel("training points $N$")
        ax.set_ylabel(label)
        ax.set_title(title)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([], [], color=plotting.INK_2, marker="x", ms=4.5, mew=1.0, lw=0))
    labels.append("over 10 min or out of memory")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=len(labels))
    plotting.save(fig, out / "scalability_combined_linear")


def num_features(results: Path, out: Path) -> None:
    rows = json.loads((results / "num_features.json").read_text())
    agg: dict[tuple[str, int], list[float]] = defaultdict(list)
    for r in rows:
        agg[(r["model"], r["num_features"])].append(r["rmse"])
    fig, ax = plt.subplots(figsize=(2.75, 1.9))
    for m in ("rwf", "rff"):
        d = sorted(k for mm, k in agg if mm == m)
        mu = np.array([np.mean(agg[(m, k)]) for k in d])
        sd = np.array([np.std(agg[(m, k)]) for k in d])
        c = plotting.color(m)
        ax.fill_between(d, mu - sd, mu + sd, color=c, alpha=0.25, lw=0)
        ax.plot(d, mu, color=c, marker=MARKERS[m], ms=3, label=plotting.LABELS[m])
    plotting.log_scale(ax, "x", base=2)
    ax.set_xticks(d, [f"$2^{{{int(np.log2(k))}}}$" for k in d])
    plotting.log_scale(ax, "y", plain=True)
    ax.set_xlabel("number of features $D$")
    ax.set_ylabel("test RMSE")
    ax.legend(loc="upper right")
    plotting.save(fig, out / "feature_wrf_rff")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results/synthetic"))
    ap.add_argument("--out", type=Path, default=Path("assets/figures/paper"))
    ap.add_argument("--only", nargs="*", default=["comparison", "scalability", "num_features"])
    args = ap.parse_args()
    plotting.apply()
    for name in args.only:
        {"comparison": comparison, "scalability": scalability, "num_features": num_features}[name](
            args.results, args.out
        )


if __name__ == "__main__":
    main()
