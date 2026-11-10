"""Animations of the three theorems: positive definiteness, unbiasedness with O(1/D) variance, uniform convergence.

python experiments/figures/theorems.py --out assets/figures/anim
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from animations import alpha_map, rgba_frame, write
from matplotlib import patheffects
from matplotlib.colors import LightSource, LinearSegmentedColormap, to_rgb
from matplotlib.ticker import MaxNLocator

from srwf import plotting
from srwf.features import RandomWaveletFeatures, split_features

INK = "#4a5466"
FAINT = "#9aa3b2"
HALO = [patheffects.withStroke(linewidth=2.2, foreground="white", alpha=0.55)]
DPI = 180
QUALITY = 60
HEAT = LinearSegmentedColormap.from_list("heat", ["#fde7dc", plotting.ORANGE, "#7a2408"])


def ease(u: np.ndarray) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(u, 0.0, 1.0))


def pingpong(fps: int, rest: float, rise: float, top: float, fall: float) -> np.ndarray:
    """Eased progress 0 -> 1 -> 0 with pauses at both ends; the last frame flows into the first."""
    n = [round(fps * t) for t in (rest, rise, top, fall)]
    return np.concatenate(
        [
            np.zeros(n[0]),
            ease(np.arange(1, n[1] + 1) / n[1]),
            np.ones(n[2]),
            ease(1.0 - np.arange(1, n[3] + 1) / n[3]),
        ]
    )


def style(ax) -> None:
    ax.set_facecolor("none")
    ax.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
    ax.tick_params(colors=INK, which="both", labelcolor=INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def opaque(rgba: np.ndarray) -> np.ndarray:
    """Composite onto white: smooth colour ramps compress far better than smooth alpha ramps."""
    a = rgba[..., 3:]
    return np.concatenate([rgba[..., :3] * a + (1.0 - a), np.ones_like(a)], -1)


def quantise(frames: list[np.ndarray], step: int) -> list[np.ndarray]:
    for f in frames:
        f[..., 3] = np.minimum(255, np.round(f[..., 3] / step) * step).astype(np.uint8)
    return frames


def sci(v: float) -> str:
    e = math.floor(math.log10(v))
    return rf"{v / 10**e:.1f}\times 10^{{{e}}}"


def features(
    dim: int, total: int, levels: int, seed: int, pool: torch.Tensor, rms=None, base_scale: float = 0.3, **kw
) -> RandomWaveletFeatures:
    f = RandomWaveletFeatures(dim, total, levels=levels, ard=False, seed=seed, **kw).double()
    f.initialize(pool, centres="box", base_scale=base_scale)
    if rms is not None:
        f.block_rms.copy_(rms)
    return f


def atom_terms(f: RandomWaveletFeatures, x: torch.Tensor) -> torch.Tensor:
    """Per-atom features rescaled so that the kernel of the first ``n_j`` atoms of every octave is a block mean."""
    counts = torch.tensor(f.counts, dtype=torch.float64)
    with torch.no_grad():
        return f(x) * counts[f.level_index].sqrt()


def nested_slice(terms_a: torch.Tensor, terms_b: torch.Tensor, f, sizes: list[int]) -> np.ndarray:
    """``k_D(a, x)`` for nested sub-samples with ``D`` atoms, using the first ``n_j(D)`` atoms of each octave."""
    prod = terms_a[:, None] * terms_b.t()
    out = np.zeros((len(sizes), terms_b.shape[0]))
    starts = np.cumsum([0] + f.counts[:-1])
    for j, (s, c) in enumerate(zip(starts, f.counts, strict=True)):
        csum = prod[s : s + c].cumsum(0).numpy()
        for i, d in enumerate(sizes):
            n = split_features(d, f.levels, 2.0)[j]
            out[i] += csum[n - 1] / n
    return out


def theorem_pd(out: Path, fps: int = 24, seconds: float = 8.0, n: int = 64) -> None:
    x = torch.linspace(0, 1, n, dtype=torch.float64)[:, None]
    grid = np.linspace(0, 1, 400)
    centres = np.linspace(0.0, 1.0, 6)
    bw = 0.13
    f = features(1, 3000, 4, 3, torch.linspace(-0.15, 1.15, 200, dtype=torch.float64)[:, None], anchors=len(centres))
    with torch.no_grad():
        f.anchor_basis.copy_(torch.exp(-0.5 * (f.translations - torch.tensor(centres)) ** 2 / bw**2))
    basis = np.exp(-0.5 * (grid[:, None] - centres) ** 2 / bw**2)
    rng = np.random.default_rng(4)
    a, b = rng.normal(0, 1.1, (2, f.levels, len(centres)))
    la, lb = rng.normal(0, 0.7, (2, f.levels))
    theta = np.linspace(0, 2 * np.pi, round(fps * seconds), endpoint=False)
    states = []
    for th in theta:
        beta = np.cos(th) * a + np.sin(th) * b
        lw = np.cos(th) * la + np.sin(th) * lb
        with torch.no_grad():
            f.anchor_log_weight.copy_(torch.tensor(beta))
            f.level_log_weight.copy_(torch.tensor(lw))
            z = f(x)
        weights = np.exp(lw[:, None] + beta @ basis.T)
        states.append((weights, (z @ z.t()).numpy(), (torch.linalg.svdvals(z) ** 2).numpy()))
    lam = np.concatenate([s[2] for s in states])
    lo, hi = 10 ** math.floor(math.log10(lam.min())), 10 ** math.ceil(math.log10(lam.max()))

    fig = plt.figure(figsize=(3.4, 2.75), dpi=DPI)
    fig.patch.set_alpha(0)
    gs = fig.add_gridspec(2, 2, height_ratios=(0.42, 1.0), width_ratios=(1.0, 1.15), hspace=0.45, wspace=0.42)
    gs.update(left=0.1, right=0.97, top=0.97, bottom=0.14)
    ax_w, ax_k, ax_l = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    for ax in (ax_w, ax_l):
        style(ax)
    strip = ax_w.imshow(
        np.zeros((f.levels, 400, 4)), extent=(0, 1, f.levels - 0.5, -0.5), aspect="auto", interpolation="bilinear"
    )
    ax_w.set_yticks(range(f.levels))
    ax_w.set_yticklabels([str(j) for j in range(f.levels)])
    ax_w.set_xticks([0, 1])
    ax_w.set_xticklabels(["0", "1"])
    ax_w.set_xlabel(r"location $t$", labelpad=-6)
    ax_w.set_ylabel(r"octave $j$")
    ax_w.tick_params(axis="y", length=0)
    ax_w.spines["left"].set_visible(False)
    ax_k.set_axis_off()
    image = ax_k.imshow(np.zeros((n, n, 4)), extent=(0, 1, 1, 0), interpolation="bilinear")
    ax_k.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, lw=0.6, color=INK, alpha=0.6))
    ax_k.text(0.5, -0.06, r"$K = Z Z^{\top}$", color=INK, ha="center", va="top", transform=ax_k.transAxes)
    plotting.log_scale(ax_l, "y")
    ax_l.set_ylim(lo, hi)
    ax_l.set_xlim(0, n + 1)
    ax_l.set_xticks([1, n // 2, n])
    ax_l.set_xlabel(r"index $i$")
    ax_l.set_ylabel(r"eigenvalue $\lambda_i$")
    dots = ax_l.plot([], [], "o", ms=1.9, color=plotting.ORANGE, mec="none")[0]
    tail = ax_l.plot([], [], "o", ms=3.2, mfc="none", mec=INK, mew=0.6)[0]
    note = ax_l.text(0.96, 0.95, "", color=INK, ha="right", va="top", transform=ax_l.transAxes, fontsize=7)
    ink = np.array(to_rgb(plotting.ORANGE))
    frames = []
    for weights, k, ev in states:
        w = (weights / weights.max()) ** 0.5
        strip.set_data(opaque(np.concatenate([np.broadcast_to(ink, w.shape + (3,)), w[..., None]], -1)))
        image.set_data(opaque(alpha_map(k, np.abs(k).max())))
        dots.set_data(np.arange(1, n + 1), ev)
        tail.set_data([n], [ev[-1]])
        note.set_text(rf"$\lambda_{{\min}} = {sci(ev[-1])}$")
        frames.append(rgba_frame(fig))
    plt.close(fig)
    write(frames, out / "theorem_pd", fps=fps, quality=QUALITY)


def theorem_unbiased(out: Path, fps: int = 24, reps: int = 200, shown: int = 48, d_max: int = 4096) -> None:
    pool = torch.linspace(-0.3, 1.3, 400, dtype=torch.float64)[:, None]
    grid = torch.linspace(0, 1, 300, dtype=torch.float64)[:, None]
    x0 = torch.tensor([[0.42]], dtype=torch.float64)
    rms = features(1, d_max, 4, 0, pool).block_rms.clone()
    sizes = sorted({int(round(v)) for v in np.geomspace(16, d_max, 160)})
    slices = []
    for r in range(reps + 64):
        f = features(1, d_max, 4, 1000 + r, pool, rms)
        slices.append(nested_slice(atom_terms(f, x0)[0], atom_terms(f, grid), f, sizes if r < reps else [d_max]))
    est = np.stack(slices[:reps])
    truth = np.mean([s[-1] for s in slices[reps:]], 0)
    var = est.var(0).mean(1)
    logd = np.log(sizes)
    u = pingpong(fps, 0.7, 4.2, 1.2, 2.4)
    xs = grid[:, 0].numpy()

    fig = plt.figure(figsize=(3.4, 2.2), dpi=DPI)
    fig.patch.set_alpha(0)
    gs = fig.add_gridspec(1, 2, width_ratios=(1.75, 1.0), wspace=0.55, left=0.13, right=0.97, top=0.95, bottom=0.2)
    ax, ax_v = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    for a in (ax, ax_v):
        style(a)
    spread = est[:shown, 0]
    ax.set_xlim(0, 1)
    ax.set_ylim(truth.min() - 0.18 * np.ptp(truth) - 0.6 * spread.std(0).max(), truth.max() + 0.6 * spread.std(0).max())
    ax.set_xticks([0, 0.5, 1])
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\hat{k}(x_0, x)$")
    curves = [ax.plot(xs, xs * 0, color=plotting.ORANGE, lw=0.45, alpha=0.32)[0] for _ in range(shown)]
    ax.plot(xs, truth, color="#141a24", lw=0.9, ls=(0, (3.5, 2)), path_effects=HALO)
    ax.axvline(float(x0), color=FAINT, lw=0.5, ls=":", zorder=0)
    label = ax.text(0.04, 0.95, "", color=INK, ha="left", va="top", transform=ax.transAxes)
    plotting.log_scale(ax_v)
    ax_v.plot(sizes, var, color=plotting.ORANGE, lw=1.0)
    ref = var[0] * (sizes[0] / np.array(sizes, dtype=float))
    ax_v.plot(sizes, ref * 1.9, color=INK, lw=0.6, ls=(0, (3, 2)))
    ax_v.text(sizes[len(sizes) // 2], ref[len(sizes) // 2] * 3.4, r"$D^{-1}$", color=INK, fontsize=7, ha="left")
    ax_v.set_xlabel(r"$D$")
    ax_v.set_ylabel("variance")
    ax_v.set_xticks([1e2, 1e3])
    dot = ax_v.plot([], [], "o", ms=4, mfc=plotting.ORANGE, mec="white", mew=0.7)[0]
    frames = []
    for p in u:
        ld = logd[0] + p * (logd[-1] - logd[0])
        i = min(int(np.searchsorted(logd, ld, side="right")) - 1, len(sizes) - 2)
        w = (ld - logd[i]) / (logd[i + 1] - logd[i])
        cur = (1 - w) * est[:shown, i] + w * est[:shown, i + 1]
        for line, c in zip(curves, cur, strict=True):
            line.set_ydata(c)
        d = math.exp(ld)
        label.set_text(rf"$D = {int(round(d))}$")
        dot.set_data([d], [math.exp((1 - w) * math.log(var[i]) + w * math.log(var[i + 1]))])
        frames.append(rgba_frame(fig))
    plt.close(fig)
    write(quantise(frames, 16), out / "theorem_unbiased", fps=fps, quality=QUALITY)


def theorem_uniform(out: Path, fps: int = 24, side: int = 40, d_max: int = 4096, truth_reps: int = 64) -> None:
    g = torch.linspace(0, 1, side, dtype=torch.float64)
    pts = torch.stack(torch.meshgrid(g, g, indexing="xy"), -1).reshape(-1, 2)
    make = dict(dim=2, total=d_max, levels=3, pool=pts, base_scale=0.5)
    rms = features(seed=0, **make).block_rms.clone()
    truth = torch.zeros(len(pts), len(pts), dtype=torch.float64)
    with torch.no_grad():
        for r in range(truth_reps):
            z = features(seed=5000 + r, rms=rms, **make)(pts)
            truth += z @ z.t()
    truth /= truth_reps
    sizes = sorted({int(round(v)) for v in np.geomspace(32, d_max, 120)})
    f = features(seed=11, rms=rms, **make)
    terms = atom_terms(f, pts)
    starts = np.cumsum([0] + f.counts[:-1])
    acc = [torch.zeros_like(truth) for _ in f.counts]
    used = [0] * f.levels
    fields = []
    for d in sizes:
        k = torch.zeros_like(truth)
        for j, n in enumerate(split_features(d, f.levels, 2.0)):
            block = terms[:, starts[j] + used[j] : starts[j] + n]
            acc[j] += block @ block.t()
            used[j] = n
            k += acc[j] / n
        fields.append((k - truth).abs().max(1).values.reshape(side, side).numpy())
    fields = np.stack(fields)
    sup = fields.max((1, 2))
    logd = np.log(sizes)
    u = pingpong(fps, 0.7, 4.6, 1.2, 2.6)
    top = 1.05 * sup[0]
    ticks = [v for v in MaxNLocator(3, integer=True).tick_values(0, top) if 0 < v <= top]
    xx, yy = np.meshgrid(g.numpy(), g.numpy())
    light = LightSource(azdeg=315, altdeg=35)
    cmap = LinearSegmentedColormap.from_list("surface", ["#f4a27f", plotting.ORANGE, "#8f2c0a"])

    fig = plt.figure(figsize=(3.6, 2.3), dpi=DPI)
    fig.patch.set_alpha(0)
    ax = fig.add_axes((-0.02, -0.06, 0.68, 1.1), projection="3d")
    ax_s = fig.add_axes((0.775, 0.2, 0.205, 0.72))
    style(ax_s)
    ax.set_axis_off()
    ax.set_facecolor("none")
    ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.set_zlim(0, top)
    ax.set_box_aspect((1, 1, 0.75))
    edge = dict(color=INK, lw=0.6, alpha=0.75, zorder=0)
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], [0] * 5, **edge)
    ax.plot([0, 0], [1, 1], [0, top], **edge)
    for v in ticks:
        ax.plot([0, -0.04], [1, 1], [v, v], **edge)
        ax.text(-0.07, 1, v, f"{v:g}", color=INK, fontsize=6.5, ha="right", va="center")
    ax.text(0.5, -0.14, 0, r"$x_1$", color=INK, ha="center", va="top")
    ax.text(1.13, 0.5, 0, r"$x_2$", color=INK, ha="left", va="center")
    label = fig.text(0.04, 0.9, "", color=INK)
    plotting.log_scale(ax_s)
    ax_s.plot(sizes, sup, color=plotting.ORANGE, lw=0.9)
    ref = sup[0] * np.sqrt(sizes[0] / np.array(sizes, dtype=float))
    ax_s.plot(sizes, ref * 1.7, color=INK, lw=0.6, ls=(0, (3, 2)))
    ax_s.text(sizes[len(sizes) // 2], ref[len(sizes) // 2] * 2.4, r"$D^{-1/2}$", color=INK, fontsize=7, ha="left")
    ax_s.set_xlabel(r"$D$")
    ax_s.set_ylabel(r"$\sup\,|\hat{k} - k|$")
    ax_s.set_xticks([1e2, 1e3])
    dot = ax_s.plot([], [], "o", ms=4, mfc=plotting.ORANGE, mec="white", mew=0.7)[0]
    surf = None
    frames = []
    for i_frame, p in enumerate(u):
        ld = logd[0] + p * (logd[-1] - logd[0])
        i = min(int(np.searchsorted(logd, ld, side="right")) - 1, len(sizes) - 2)
        w = (ld - logd[i]) / (logd[i + 1] - logd[i])
        field = (1 - w) * fields[i] + w * fields[i + 1]
        if surf is not None:
            surf.remove()
        colours = light.shade(field, cmap, vmin=0, vmax=top, blend_mode="soft", vert_exag=0.02)
        surf = ax.plot_surface(
            xx, yy, field, facecolors=colours, rstride=1, cstride=1, lw=0, antialiased=False, shade=False
        )
        ax.view_init(elev=26, azim=-58 + 6 * math.sin(2 * math.pi * i_frame / len(u)))
        d = math.exp(ld)
        label.set_text(rf"$D = {int(round(d))}$")
        dot.set_data([d], [sup[i] ** (1 - w) * sup[i + 1] ** w])
        frames.append(rgba_frame(fig))
    plt.close(fig)
    write(frames, out / "theorem_uniform", fps=fps, quality=QUALITY)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("assets/figures/anim"))
    ap.add_argument("--only", nargs="*", default=["pd", "unbiased", "uniform"])
    args = ap.parse_args()
    plotting.apply()
    torch.set_num_threads(4)
    if "pd" in args.only:
        theorem_pd(args.out)
    if "unbiased" in args.only:
        theorem_unbiased(args.out)
    if "uniform" in args.only:
        theorem_uniform(args.out)


if __name__ == "__main__":
    main()
