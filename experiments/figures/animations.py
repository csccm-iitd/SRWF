"""Explanatory animations with transparent backgrounds (animated WebP and VP9 WebM with alpha).

kernel_morph   an anchor glides through a rough region: the RWF kernel tightens, RFF never changes
multires       the burst-signal posterior as wavelet octaves are switched on, coarse to fine
globe          rotating globes: an ERA5 field with its stations and the reconstruction errors

python experiments/figures/animations.py --out assets/figures/anim
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import imageio_ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import to_rgb

from srwf import RFFGP, RWFGP, plotting
from srwf.data.synthetic import burst, sample
from srwf.features import RandomFourierFeatures, RandomWaveletFeatures

INK = "#7b879b"


def rgba_frame(fig) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()


def write(frames: list[np.ndarray], path: Path, fps: int = 30, quality: int = 70) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    raw = b"".join(f.tobytes() for f in frames)
    base = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba"]
    base += ["-s", f"{w}x{h}", "-r", str(fps), "-i", "-"]
    webp = [
        "-c:v",
        "libwebp_anim",
        "-lossless",
        "0",
        "-q:v",
        str(quality),
        "-loop",
        "0",
        str(path.with_suffix(".webp")),
    ]
    webm = ["-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-b:v", "0", "-crf", "30", str(path.with_suffix(".webm"))]
    for tail in (webp, webm):
        subprocess.run(base + tail, input=raw, check=True)


def alpha_map(values: np.ndarray, lim: float) -> np.ndarray:
    """Diverging colours whose opacity grows with |value|: zero is fully transparent."""
    v = np.clip(values / lim, -1, 1)
    pos, neg = np.array(to_rgb(plotting.ORANGE)), np.array(to_rgb(plotting.BLUE))
    rgb = np.where(v[..., None] >= 0, pos, neg)
    a = np.abs(v) ** 0.7
    return np.concatenate([rgb, a[..., None]], -1)


def kernel_morph(out: Path, frames: int = 150) -> None:
    n = 200
    g = np.linspace(-1, 1, n)
    xx, yy = np.meshgrid(g, g)
    pts = torch.tensor(np.stack([xx.ravel(), yy.ravel()], 1), dtype=torch.float32)
    centre = torch.tensor([0.4, 0.35])
    rwf = RandomWaveletFeatures(2, 6000, levels=4, growth=2.0, anchors=1, ard=False, seed=1)
    rwf.initialize(pts, centres="box", base_scale=0.5)
    with torch.no_grad():
        dist = (rwf.translations - centre).norm(dim=1)
        octave = rwf.level_index.float() / (rwf.levels - 1)
        local = torch.sigmoid((0.5 - dist) / 0.08)
        rwf.anchor_basis[:, 0] = torch.log(1e-4 + (1 - octave) ** 3 + 12.0 * octave**2 * local)
        rwf.anchor_log_weight.fill_(1.0)
        rff = RandomFourierFeatures(2, 6000, ard=False, seed=1)
        rff.raw_lengthscale.fill_(float(np.log(np.expm1(0.22))))
        zw, zf = rwf(pts).numpy(), rff(pts).numpy()
    t = np.linspace(0, 2 * np.pi, frames, endpoint=False)
    path = np.stack([0.05 + 0.62 * np.cos(t), 0.0 + 0.55 * np.sin(t)], 1)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=110)
    fig.patch.set_alpha(0)
    images = []
    for ax in axes:
        ax.set_axis_off()
        ax.set_xlim(-1, 1), ax.set_ylim(-1, 1), ax.set_aspect("equal")
        images.append(ax.imshow(np.zeros((n, n, 4)), extent=(-1, 1, -1, 1), origin="lower", interpolation="bilinear"))
    axes[1].add_patch(plt.Circle(centre.numpy(), 0.5, fill=False, ls=(0, (2, 3)), lw=1.0, color=INK))
    for ax in axes:
        ax.add_patch(plt.Rectangle((-1, -1), 2, 2, fill=False, lw=0.8, color=INK, alpha=0.5))
    dots = [ax.plot([], [], "o", ms=6, mfc="white", mec="#12233d", mew=1.5)[0] for ax in axes]
    fig.subplots_adjust(0.01, 0.01, 0.99, 0.99, wspace=0.06)
    out_frames = []
    with torch.no_grad():
        for p in path:
            a = torch.tensor(p, dtype=torch.float32).unsqueeze(0)
            kf, kw = zf @ rff(a).numpy()[0], zw @ rwf(a).numpy()[0]
            images[0].set_data(alpha_map(kf.reshape(n, n), kf.max()))
            images[1].set_data(alpha_map(kw.reshape(n, n), kw.max()))
            for d in dots:
                d.set_data([p[0]], [p[1]])
            out_frames.append(rgba_frame(fig))
    plt.close(fig)
    write(out_frames, out / "kernel_morph")


def multires(out: Path, n_train: int = 256, hold: int = 10, blend: int = 14) -> None:
    xtr, ytr, _, _ = sample("burst", n_train, seed=0)
    model = RWFGP(num_features=1024, levels=9, anchors=16, steps=500, seed=0, device="cpu").fit(xtr, ytr)
    rff = RFFGP(num_features=1024, steps=500, seed=0, device="cpu").fit(xtr, ytr)
    grid = np.linspace(0, 1, 1500)
    blr = model.blr
    feats = blr.features
    with torch.no_grad():
        xs, gs = model._x(xtr), model._x(grid[:, None])
        z, zg = blr.design(xs), blr.design(gs)
        y = torch.as_tensor((ytr - model.y_mu) / model.y_sd, dtype=torch.float64)
        levels = feats.level_index.cpu().numpy()
        curves = []
        for j in range(feats.levels):
            keep = torch.as_tensor(levels <= j)
            zk, zgk = z[:, keep], zg[:, keep]
            a = zk.t() @ zk / blr.noise + torch.eye(int(keep.sum()), dtype=torch.float64) / blr.prior_var
            chol = torch.linalg.cholesky(a)
            m = torch.cholesky_solve((zk.t() @ (y - blr.bias) / blr.noise).unsqueeze(-1), chol).squeeze(-1)
            mean = (zgk @ m + blr.bias).numpy() * model.y_sd + model.y_mu
            sd = torch.linalg.solve_triangular(chol, zgk.t(), upper=False).pow(2).sum(0).sqrt().numpy() * model.y_sd
            curves.append((mean, sd))
    pf = rff.predict(grid[:, None], return_f_std=True)

    fig, axes = plt.subplots(2, 1, figsize=(8, 4.2), dpi=110, sharex=True)
    fig.patch.set_alpha(0)
    for ax in axes:
        ax.set_facecolor("none")
        ax.set_axis_off()
        ax.set_xlim(0, 1), ax.set_ylim(-2.1, 1.0)
        ax.axvspan(0.54, 0.70, color=INK, alpha=0.08, lw=0)
        ax.scatter(xtr[:, 0], ytr, s=5, color=INK, alpha=0.55, lw=0)
        ax.plot(grid, burst(grid), color=INK, lw=0.8, ls=(0, (3, 2)))
    axes[1].fill_between(grid, pf.mean - 2 * pf.f_std, pf.mean + 2 * pf.f_std, color=plotting.BLUE, alpha=0.2, lw=0)
    axes[1].plot(grid, pf.mean, color=plotting.BLUE, lw=1.8)
    line = axes[0].plot([], [], color=plotting.ORANGE, lw=1.8)[0]
    fill = [None]
    ticks = [axes[0].plot([], [], "s", ms=5, color=plotting.ORANGE, alpha=0.9)[0] for _ in range(feats.levels)]
    fig.subplots_adjust(0.01, 0.01, 0.99, 0.99, hspace=0.04)
    frames = []

    def draw(mean, sd, active):
        if fill[0] is not None:
            fill[0].remove()
        fill[0] = axes[0].fill_between(grid, mean - 2 * sd, mean + 2 * sd, color=plotting.ORANGE, alpha=0.2, lw=0)
        line.set_data(grid, mean)
        for j, tk in enumerate(ticks):
            tk.set_data([0.02 + 0.025 * j], [0.85])
            tk.set_alpha(0.95 if j <= active else 0.15)
        frames.append(rgba_frame(fig))

    for j in range(len(curves)):
        for _ in range(hold):
            draw(*curves[j], j)
        if j + 1 < len(curves):
            for s in np.linspace(0, 1, blend)[1:-1]:
                w = 0.5 - 0.5 * np.cos(np.pi * s)
                draw((1 - w) * curves[j][0] + w * curves[j + 1][0], (1 - w) * curves[j][1] + w * curves[j + 1][1], j)
    for _ in range(3 * hold):
        draw(*curves[-1], len(curves) - 1)
    plt.close(fig)
    write(frames, out / "multires", fps=24)


def globe(
    out: Path,
    field: Path,
    frames: int = 240,
    fps: int = 15,
    dpi: int = 60,
    quality: int = 45,
    models: tuple[str, ...] = ("rwf", "exact", "rff"),
) -> None:
    import cartopy.crs as ccrs

    d = np.load(field)
    lon, lat, truth = d["lon"], d["lat"], d["truth"]
    lon_c = np.concatenate([lon, lon[:1] + 360.0])

    lat_c = np.concatenate([[-89.5], lat, [89.5]])

    def wrap(a: np.ndarray) -> np.ndarray:
        a = np.concatenate([a, a[:, :1]], axis=1)
        return np.concatenate([a[:1], a, a[-1:]], axis=0)

    errors = {m: np.abs(d[f"{m}_mean"] - truth) for m in models}
    emax = np.percentile(errors[models[-1]], 97)
    tlim = tuple(np.percentile(truth, [5, 99.5]))
    sy, sx = np.nonzero(d["station_mask"])
    fig = plt.figure(figsize=(3.2 * (1 + len(models)), 3.4), dpi=dpi)
    fig.patch.set_alpha(0)
    out_frames = []
    for k in range(frames):
        fig.clf()
        proj = ccrs.Orthographic(central_longitude=-180 + 360 * k / frames, central_latitude=20)
        panels = [("truth", wrap(truth), "viridis", tlim)] + [(m, wrap(errors[m]), "magma", (0, emax)) for m in models]
        for i, (_, values, cmap, (vmin, vmax)) in enumerate(panels):
            ax = fig.add_subplot(1, len(panels), i + 1, projection=proj)
            ax.set_global()
            ax.patch.set_alpha(0)
            ax.spines["geo"].set_edgecolor(INK)
            ax.spines["geo"].set_linewidth(0.6)
            ax.pcolormesh(
                lon_c, lat_c, values, cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(), shading="gouraud"
            )
            ax.coastlines("110m", color="white" if i else "#12233d", linewidth=0.5, alpha=0.8)
            if i == 0:
                ax.scatter(lon[sx], lat[sy], s=3.5, color="#12233d", alpha=0.75, transform=ccrs.PlateCarree(), lw=0)
        fig.subplots_adjust(0.005, 0.01, 0.995, 0.99, wspace=0.03)
        out_frames.append(rgba_frame(fig))
    plt.close(fig)
    write(out_frames, out / "globe", fps=fps, quality=quality)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("assets/figures/anim"))
    ap.add_argument("--only", nargs="*", default=["kernel_morph", "multires", "globe"])
    ap.add_argument("--field", type=Path, default=Path("results/climate/showcase/surface_pressure_2005-01-15T00.npz"))
    args = ap.parse_args()
    plotting.apply()
    if "kernel_morph" in args.only:
        kernel_morph(args.out)
    if "multires" in args.only:
        multires(args.out)
    if "globe" in args.only:
        globe(args.out, args.field)


if __name__ == "__main__":
    main()
