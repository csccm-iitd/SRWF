"""Figure style shared by all figures and animations.

Computer Modern text at small sizes, full box frames with short ticks, thin lines, translucent
posterior bands under an opaque mean, black dashed ground truth and posterior samples. Colour
encodes the method: RWF is orange, a stationary baseline is blue, and further baselines sharing a
panel take muted Tableau colours.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

SURFACE = "#ffffff"
INK = "#000000"
INK_2 = "#3a3a3a"
INK_MUTED = "#8c8c8c"
GRID = "#ebebeb"
AXIS = "#262626"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#4e9f8f"
GREEN = "#59a14f"
PURPLE = "#9467bd"
BROWN = "#8c6d5c"
GAP = "#f2f2f2"

METHOD_COLORS = {"rwf": ORANGE, "rff": BLUE, "exact": GREEN, "svgp": PURPLE, "dgp": BROWN}
LABELS = {"rwf": "RWF (ours)", "rff": "RFF", "exact": "Exact GP", "svgp": "SVGP", "dgp": "Deep GP"}
TEXT_WIDTH = 5.5
FULL_WIDTH = 6.75


def color(method: str) -> str:
    return METHOD_COLORS.get(method.lower(), BLUE)


def apply(scale: float = 1.0) -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["cmr10", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "axes.unicode_minus": False,
            "font.size": 8 * scale,
            "axes.titlesize": 9 * scale,
            "axes.titlelocation": "center",
            "axes.titlepad": 4,
            "axes.titlecolor": INK,
            "axes.labelsize": 8 * scale,
            "axes.labelcolor": INK,
            "axes.labelpad": 2.5,
            "xtick.labelsize": 7 * scale,
            "ytick.labelsize": 7 * scale,
            "xtick.color": AXIS,
            "ytick.color": AXIS,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.minor.size": 1.4,
            "ytick.minor.size": 1.4,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            "xtick.major.pad": 2,
            "ytick.major.pad": 2,
            "legend.fontsize": 7 * scale,
            "legend.frameon": False,
            "legend.handlelength": 1.6,
            "legend.handletextpad": 0.5,
            "legend.borderaxespad": 0.3,
            "legend.labelspacing": 0.3,
            "legend.columnspacing": 1.2,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.5,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.grid": True,
            "axes.grid.axis": "both",
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "axes.facecolor": SURFACE,
            "figure.facecolor": SURFACE,
            "lines.linewidth": 1.0,
            "lines.markersize": 3,
            "lines.solid_capstyle": "round",
            "scatter.edgecolors": "none",
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "figure.dpi": 110,
            "pdf.fonttype": 42,
        }
    )


def band(ax, x, mean, std, c: str, k: float = 2.0, alpha: float = 0.25, label: str | None = None) -> None:
    """Translucent ``mean +- k std`` band under an opaque mean."""
    ax.fill_between(x, mean - k * std, mean + k * std, color=c, alpha=alpha, lw=0)
    ax.plot(x, mean, color=c, lw=1.0, label=label)


def samples(ax, x, f: np.ndarray, label: str | None = None) -> None:
    """Posterior function draws (columns of ``f``) as thin black dashed lines."""
    ax.plot(x, f, color=INK, lw=0.5, ls=(0, (3, 2)), alpha=0.8)
    if label:
        ax.plot([], [], color=INK, lw=0.5, ls=(0, (3, 2)), label=label)


def truth(ax, x, f, label: str | None = None) -> None:
    ax.plot(x, f, color=INK, lw=0.7, ls=(0, (3, 2)), label=label)


def log_scale(ax, axis: str = "both", base: int = 10, plain: bool = False) -> None:
    """Logarithmic axes labelled ``base^k`` at the decades; ``plain`` labels 1-2-5 in decimals."""
    for name in ("x", "y") if axis == "both" else (axis,):
        getattr(ax, f"set_{name}scale")("log", base=base)
        a = getattr(ax, f"{name}axis")
        if plain:
            a.set_major_locator(LogLocator(base=base, subs=(1.0, 2.0, 5.0), numticks=12))
            a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        a.set_minor_locator(LogLocator(base=base, subs="auto", numticks=12))
        a.set_minor_formatter(NullFormatter())


def save(fig, path: str | Path, png: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))
    if png:
        fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)
