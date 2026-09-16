from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import set_publication_style


SCALABILITY_COLORS = {
    "RWF-GP": "#1f77b4",
    "RFF-GP": "#d62728",
    "SVGP": "#2ca02c",
    "Deep GP": "#9467bd",
}

SCALABILITY_MARKERS = {
    "RWF-GP": "o",
    "RFF-GP": "^",
    "SVGP": "s",
    "Deep GP": "D",
}

FEATURE_COLORS = {"RWF-GP": "#1f77b4", "RFF-GP": "#d62728"}
FEATURE_MARKERS = {"RWF-GP": "o", "RFF-GP": "^"}


def create_scalability_combined_linear_plot(
    csv_path: Path | str,
    output_stem: Path | str,
    png_dpi: int = 600,
) -> Tuple[Path, Path]:
    set_publication_style()
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    x = pd.to_numeric(df["Samples"], errors="coerce").to_numpy(dtype=float)
    labels = [_sample_label(v) for v in x]

    time_map = (
        ("RWF-GP", "RWF-GP_time_min", "RWF-GP_std_min"),
        ("RFF-GP", "RFF-GP_time_min", "RFF-GP_std_min"),
        ("SVGP", "SVGP_time_min", "SVGP_std_min"),
        ("Deep GP", "DeepGP_time_min", "DeepGP_std_min"),
    )
    mem_map = (
        ("RWF-GP", "RWF-GP_mem_gb", "RWF-GP_std_gb"),
        ("RFF-GP", "RFF-GP_mem_gb", "RFF-GP_std_gb"),
        ("SVGP", "SVGP_mem_gb", "SVGP_std_gb"),
        ("Deep GP", "DeepGP_mem_gb", "DeepGP_std_gb"),
    )

    fig, (ax_time, ax_mem) = plt.subplots(1, 2, figsize=(16.0, 6.0))
    _plot_scalability_panel(df, x, labels, time_map, ax_time, "Training time (min)", "(a) Training Time")
    _plot_scalability_panel(df, x, labels, mem_map, ax_mem, "Memory (GB)", "(b) Memory Usage")
    ax_mem.legend(loc="upper left", frameon=False)
    fig.tight_layout(pad=0.4, w_pad=1.0)
    return _save_fig(fig, output_stem, png_dpi)


def create_feature_wrf_rff_plot(
    csv_path: Path | str,
    output_stem: Path | str,
    png_dpi: int = 600,
) -> Tuple[Path, Path]:
    set_publication_style()
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    features = pd.to_numeric(df["Features"], errors="coerce").to_numpy(dtype=float)
    rwf_mean, rwf_std = _parse_pm(df["RWF_RMSE"])
    rff_mean, rff_std = _parse_pm(df["RFF_RMSE"])

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(
        features,
        rwf_mean,
        marker=FEATURE_MARKERS["RWF-GP"],
        linestyle=":",
        color=FEATURE_COLORS["RWF-GP"],
        label="RWF-GP",
    )
    if np.isfinite(rwf_std).any():
        ax.fill_between(
            features,
            rwf_mean - rwf_std,
            rwf_mean + rwf_std,
            color=FEATURE_COLORS["RWF-GP"],
            alpha=0.22,
            linewidth=0.0,
        )

    ax.plot(
        features,
        rff_mean,
        marker=FEATURE_MARKERS["RFF-GP"],
        linestyle=":",
        color=FEATURE_COLORS["RFF-GP"],
        label="RFF-GP",
    )
    if np.isfinite(rff_std).any():
        ax.fill_between(
            features,
            rff_mean - rff_std,
            rff_mean + rff_std,
            color=FEATURE_COLORS["RFF-GP"],
            alpha=0.22,
            linewidth=0.0,
        )

    ax.set_xticks(features)
    ax.set_xlabel(r"Number of features $D$")
    ax.set_ylabel("RMSE")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout(pad=0.35)
    return _save_fig(fig, output_stem, png_dpi)


def _plot_scalability_panel(
    df: pd.DataFrame,
    x: np.ndarray,
    x_labels: Iterable[str],
    metric_map: Iterable[Tuple[str, str, str]],
    ax: plt.Axes,
    y_label: str,
    title: str,
) -> None:
    for model_name, value_col, std_col in metric_map:
        if value_col not in df.columns:
            continue
        y = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        if not np.any(valid):
            continue
        x_valid, y_valid = x[valid], y[valid]
        ax.plot(
            x_valid,
            y_valid,
            marker=SCALABILITY_MARKERS[model_name],
            linestyle=":",
            color=SCALABILITY_COLORS[model_name],
            label=model_name,
        )
        if std_col in df.columns:
            y_std = pd.to_numeric(df[std_col], errors="coerce").to_numpy(dtype=float)
            valid_std = valid & np.isfinite(y_std)
            if np.any(valid_std):
                x_std = x[valid_std]
                y_std_mean = y[valid_std]
                std = y_std[valid_std]
                lower = np.clip(y_std_mean - std, a_min=0.0, a_max=None)
                upper = y_std_mean + std
                ax.fill_between(
                    x_std,
                    lower,
                    upper,
                    color=SCALABILITY_COLORS[model_name],
                    alpha=0.12,
                    linewidth=0.0,
                )

    ax.set_xscale("log", base=10)
    ax.set_xticks(x)
    ax.set_xticklabels(list(x_labels))
    ax.set_xlabel("Number of samples")
    ax.set_ylabel(y_label)
    ax.set_title(title)


def _parse_pm(series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    means, stds = [], []
    pattern = re.compile(
        r"\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(?:\u00B1|\+/-)\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*"
    )
    for raw in series.astype(str):
        token = raw.strip()
        match = pattern.fullmatch(token)
        if match:
            means.append(float(match.group(1)))
            stds.append(float(match.group(2)))
            continue
        means.append(float(token))
        stds.append(np.nan)
    return np.asarray(means, dtype=float), np.asarray(stds, dtype=float)


def _sample_label(value: float) -> str:
    if int(value) == 500:
        return "500"
    return f"{int(value / 1000)}K"


def _save_fig(fig: plt.Figure, output_stem: Path | str, png_dpi: int) -> Tuple[Path, Path]:
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=png_dpi, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)
    return png_path, pdf_path

