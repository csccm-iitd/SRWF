"""Bayesian-optimisation figures: simple-regret curves, the cost of one iteration and of posterior
sample paths, and a Thompson-sampling animation of RWF against RFF on bumped Branin.

python experiments/figures/bo.py --results results/bo --out assets/figures
python experiments/figures/bo.py --only animation --seed 9
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from animations import rgba_frame, write
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory
from scipy.optimize import minimize

from srwf import plotting

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bo"))

from functions import BUMP_CENTRE, PROBLEMS  # noqa: E402
from methods import run_method  # noqa: E402

PROBLEM_ORDER = ["branin", "bumped_branin", "gramacy_2d", "gramacy_lee", "hartmann6", "gramacy_10d", "levy10"]
TITLES = {
    "branin": "Branin (2-D)",
    "bumped_branin": "Bumped Branin (2-D)",
    "gramacy_2d": "Gramacy (2-D)",
    "gramacy_lee": "Gramacy-Lee (1-D)",
    "hartmann6": "Hartmann (6-D)",
    "gramacy_10d": "Gramacy (10-D)",
    "levy10": "Levy (10-D)",
}
METHODS = {
    "rwf_ts": ("RWF-TS (ours)", plotting.ORANGE),
    "rff_ts": ("RFF-TS", plotting.BLUE),
    "gp_ts": ("GP-TS", plotting.GREEN),
    "gp_ei": ("GP-EI", plotting.PURPLE),
    "random": ("Random", plotting.INK_MUTED),
}
COST = {
    "rwf": ("RWF (ours)", plotting.ORANGE, "o", "rwf_ts", "rwf_pathwise"),
    "rff": ("RFF", plotting.BLUE, "^", "rff_ts", "rff_pathwise"),
    "exact": ("Exact GP", plotting.GREEN, "D", "gp_ts", "exact_cholesky"),
}
FLOOR = 1e-8
INK = "#7b879b"
LANDSCAPE = LinearSegmentedColormap.from_list("landscape", ["#141b2b", "#3a4a66", "#8c9bb3", "#e6e9ee"])


def regret(results: Path, out: Path) -> None:
    curves = np.load(results / "regret_curves.npz")
    fig, axes = plt.subplots(2, 4, figsize=(plotting.FULL_WIDTH, 3.15), gridspec_kw=dict(wspace=0.32, hspace=0.55))
    for ax, name in zip(axes.flat, PROBLEM_ORDER, strict=False):
        p = PROBLEMS[name]
        for m, (label, c) in METHODS.items():
            r = np.log10(np.maximum(curves[f"{name}__{m}"], FLOOR))
            n = np.arange(1, r.shape[1] + 1)
            lo, med, hi = np.percentile(r, [25, 50, 75], axis=0)
            ax.fill_between(n, 10**lo, 10**hi, color=c, alpha=0.18, lw=0)
            ax.plot(n, 10**med, color=c, lw=1.0, label=label, zorder=3 if m == "rwf_ts" else 2)
        ax.axvline(2 * p.dim + 1, color=plotting.INK_MUTED, lw=0.5, ls=(0, (1, 1.5)))
        plotting.log_scale(ax, "y")
        ax.set_xlim(1, p.budget)
        ax.set_ylim(top=ax.get_ylim()[1] * 3)
        ax.set_title(TITLES[name])
    for ax in axes[1]:
        ax.set_xlabel("evaluations")
    for ax in axes[:, 0]:
        ax.set_ylabel("simple regret")
    legend_ax = axes.flat[-1]
    legend_ax.set_axis_off()
    handles = [Line2D([], [], color=c, lw=1.0) for _, c in METHODS.values()]
    handles += [Patch(color=plotting.INK_2, alpha=0.18, lw=0)]
    handles += [Line2D([], [], color=plotting.INK_MUTED, lw=0.5, ls=(0, (1, 1.5)))]
    labels = [label for label, _ in METHODS.values()] + ["interquartile range", "initial design"]
    legend_ax.legend(handles, labels, loc="center", title="median of 20 seeds", title_fontsize=7)
    plotting.save(fig, out / "bo_regret")


def cost(results: Path, out: Path, budget: float = 600.0) -> None:
    it = pd.read_csv(results / "iteration_cost.csv")
    sc = pd.read_csv(results / "sampling_cost.csv")
    grid = sorted(sc["n"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(plotting.TEXT_WIDTH, 2.0), gridspec_kw=dict(wspace=0.3))
    axes[0].axhline(budget, color=plotting.INK_MUTED, lw=0.6, ls=(0, (1, 1.5)))
    axes[0].text(110, budget * 0.62, "10 min", fontsize=6.5, color=plotting.INK_2)
    top = blended_transform_factory(axes[1].transData, axes[1].transAxes)
    for label, c, marker, ts, sampler in COST.values():
        style = dict(color=c, marker=marker, ms=3, lw=1.0)
        r = it[it["method"] == ts].sort_values("n")
        ok = r[np.isfinite(r["total_seconds"])]
        axes[0].plot(ok["n"], ok["total_seconds"], label=label, **style)
        for n in r.loc[~np.isfinite(r["total_seconds"]), "n"]:
            axes[0].plot([n], [budget], "x", color=c, ms=4.5, mew=1.0, clip_on=False)
        s = sc[sc["method"] == sampler].sort_values("n")
        axes[1].plot(s["n"], s["seconds"], **style)
        for n in sorted(set(grid) - set(s["n"])):
            axes[1].plot([n], [0.95], "x", color=c, ms=4.5, mew=1.0, transform=top, clip_on=False)
    for ax, xlabel, ylabel, title in zip(
        axes,
        ("observations $N$", "evaluation points $M$"),
        ("seconds per iteration", "seconds for 16 paths"),
        ("One Thompson-sampling iteration", "Drawing posterior sample paths"),
        strict=True,
    ):
        plotting.log_scale(ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    axes[0].set_ylim(top=budget * 5)
    axes[1].set_ylim(top=axes[1].get_ylim()[1] * 4)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([], [], color=plotting.INK_2, marker="x", ms=4.5, mew=1.0, lw=0))
    labels.append("out of memory")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=4)
    plotting.save(fig, out / "bo_cost")


def minimiser(p) -> np.ndarray:
    res = minimize(lambda z: -p(z[None])[0], BUMP_CENTRE, method="L-BFGS-B", bounds=list(map(tuple, p.bounds)))
    return res.x


def record(results: Path, seed: int, side: int = 160) -> Path:
    p = PROBLEMS["bumped_branin"]
    g1, g2 = np.linspace(*p.bounds[0], side), np.linspace(*p.bounds[1], side)
    grid = np.stack(np.meshgrid(g1, g2), -1).reshape(-1, 2)
    gt = torch.as_tensor(grid, dtype=torch.float32)
    n_init = 2 * p.dim + 1
    rwf = {"levels": 4, "data_fraction": 0.5, "lowpass": 32, "fourier": 0, "anchors": 0, "weight_prior": 1.0}
    data: dict[str, np.ndarray] = {}
    torch.set_num_threads(1)
    for method in ("rwf_ts", "rff_ts"):
        paths = []

        def callback(it, f, x_next, trace, paths=paths):
            with torch.no_grad():
                paths.append(f(gt)[0].float().cpu().numpy().reshape(side, side))

        np.random.seed(seed)
        torch.manual_seed(seed)
        options = rwf if method == "rwf_ts" else {}
        trace = run_method(
            method, p, p.bounds, p.budget, n_init, seed, steps=100, warm_start=True, callback=callback, **options
        )
        data[f"{method}_path"] = np.array(paths)
        data[f"{method}_x"] = trace.x
        data[f"{method}_regret"] = np.maximum(p.optimum - trace.best, 0.0)
        stored = pd.read_json(results / "runs" / f"bumped_branin__{method}__{seed}.json", typ="series")
        print(f"{method}: max |x - stored x| = {np.abs(trace.x - np.array(stored['x'])).max():.2e}")
    path = results / f"animation_bumped_branin_seed{seed}.npz"
    np.savez_compressed(
        path, g1=g1, g2=g2, truth=-p(grid).reshape(side, side), minimiser=minimiser(p), n_init=n_init, **data
    )
    return path


def animation(results: Path, out: Path, docs: Path, seed: int, iterations: int, hold: int = 12) -> None:
    path = results / f"animation_bumped_branin_seed{seed}.npz"
    if not path.exists():
        record(results, seed)
    d = np.load(path)
    n_init, star = int(d["n_init"]), d["minimiser"]
    extent = (d["g1"][0], d["g1"][-1], d["g2"][0], d["g2"][-1])
    fig = plt.figure(figsize=(7.2, 4.6), dpi=100)
    fig.patch.set_alpha(0)
    grid = fig.add_gridspec(2, 2, height_ratios=(1, 0.2), left=0.07, right=0.98, top=0.87, bottom=0.04)
    grid.update(wspace=0.1, hspace=0.08)
    panels = []
    for j, m in enumerate(("rwf_ts", "rff_ts")):
        label, c = METHODS[m]
        ax, tr = fig.add_subplot(grid[0, j]), fig.add_subplot(grid[1, j])
        ax.set_xticks([]), ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(INK)
        image = ax.imshow(np.zeros((2, 2)), extent=extent, origin="lower", cmap=LANDSCAPE, aspect="auto")
        ax.plot(*star, "*", ms=13, mfc="#ffd23f", mec="#141b2b", mew=0.8, zorder=7)
        old = ax.plot([], [], "o", ms=3.2, mfc="white", mec="#141b2b", mew=0.5, zorder=4)[0]
        new = ax.plot([], [], "o", ms=7, mfc=c, mec="white", mew=1.2, zorder=6)[0]
        ax.set_xlim(extent[:2]), ax.set_ylim(extent[2:])
        ax.set_title(label.removesuffix(" (ours)"), color=c, fontsize=13, pad=5)
        r = np.log10(np.maximum(d[f"{m}_regret"][n_init - 1 : n_init + iterations], 1e-6))
        tr.set_facecolor("none")
        tr.set_xlim(0, iterations + 1), tr.set_ylim(-6.3, 1.2)
        tr.set_yticks([-6, -3, 0], ["$10^{-6}$", "$10^{-3}$", "$10^{0}$"], fontsize=8, color=INK)
        tr.set_xticks([])
        tr.grid(False)
        tr.tick_params(colors=INK, length=2)
        for spine in tr.spines.values():
            spine.set_color(INK)
        line = tr.plot([], [], color=c, lw=1.6)[0]
        dot = tr.plot([], [], "o", ms=4, color=c)[0]
        panels.append((m, image, old, new, line, dot, r))
    fig.axes[1].set_ylabel("regret", color=INK, fontsize=9)
    counter = fig.text(0.5, 0.97, "", ha="center", va="center", color=INK, fontsize=12)
    frames = []
    for k in range(1, iterations + 1):
        counter.set_text(f"iteration {k}")
        for m, image, old, new, line, dot, r in panels:
            v = -d[f"{m}_path"][k - 1]
            v = np.log1p(v - v.min())
            image.set_data(v)
            image.set_clim(0, np.percentile(v, 99))
            x = d[f"{m}_x"]
            old.set_data(x[: n_init + k - 1, 0], x[: n_init + k - 1, 1])
            new.set_data([x[n_init + k - 1, 0]], [x[n_init + k - 1, 1]])
            line.set_data(np.arange(k + 1), r[: k + 1])
            dot.set_data([k], [r[k]])
        frames += [rgba_frame(fig)] * 2
    frames += [frames[-1]] * hold
    plt.close(fig)
    write(frames, out / "bo_thompson", fps=5, quality=80)
    docs.mkdir(parents=True, exist_ok=True)
    shutil.copy(out / "bo_thompson.webp", docs / "bo_thompson.webp")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results/bo"))
    ap.add_argument("--out", type=Path, default=Path("assets/figures"))
    ap.add_argument("--docs", type=Path, default=Path("docs/static/img"))
    ap.add_argument("--only", nargs="*", default=["regret", "cost"])
    ap.add_argument("--seed", type=int, default=9, help="animation seed; its outcome is near the 20-seed median")
    ap.add_argument("--iterations", type=int, default=52)
    ap.add_argument("--record", action="store_true", help="re-run the animation's BO traces")
    args = ap.parse_args()
    plotting.apply()
    for name in args.only:
        if name == "animation":
            if args.record:
                record(args.results, args.seed)
            animation(args.results, args.out / "anim", args.docs, args.seed, args.iterations)
            continue
        {"regret": regret, "cost": cost}[name](args.results, args.out)
        args.docs.mkdir(parents=True, exist_ok=True)
        shutil.copy(args.out / f"bo_{name}.png", args.docs / f"bo_{name}.png")


if __name__ == "__main__":
    main()
